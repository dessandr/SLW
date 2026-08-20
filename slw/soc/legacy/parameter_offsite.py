
"""Fit site-pair SOC parameters from Wannier90 position matrices.

This is the archived offsite extension of ``slw.soc.legacy.parameter``. It uses
Wannier90 ``*_r.dat`` position matrix elements and a collinear charge-channel
Hamiltonian to build k-dependent templates

    L(k) ~= r(k) x dH0/dk
    H_soc(k) = sum_ab lambda_ab P_ab [L(k) . S]

where ``P_ab`` masks orbital rows on site type ``a`` and columns on site type
``b``.  The implementation follows the SLW spin-major convention
``[up orbitals | down orbitals]``.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from slw.core.wannier_io import write_wannier_hr
from slw.exchange.kernels.epr import _full_k_mesh
from slw.exchange.kernels.j_tensor_epr import (
    _win_projection_groups,
)
from slw.exchange.kernels.soc_downfold import (
    _normalize_projection_groups_for_hamiltonian,
)
from slw.exchange.kernels.spinor import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    d_orbital_l_matrices,
    p_orbital_l_matrices,
    spinor_from_collinear,
)
from slw.soc.manifold import _clean_species
from slw.soc.legacy.parameter import (
    _cfg_value,
    _cfg_vec,
    _hmap_to_hk,
    _read_wannier_hr_compat,
    band_weights,
    relative_energies,
)


@dataclass(frozen=True)
class PairSOCSpec:
    label: str
    row_element: str
    col_element: str
    lambda0: float
    lower: float
    upper: float


def _read_parameter_offsite_input(path: str) -> dict:
    cfg: dict[str, object] = {}
    pair_rows: list[str] = []
    orbital_group_rows: list[str] = []
    block = None
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.split("#", 1)[0].split("!", 1)[0].strip()
            if not line:
                continue
            words = line.split()
            head = words[0].lower()
            if head == "begin":
                if len(words) != 2:
                    raise ValueError(f"{path}:{lineno}: expected 'begin <block>'")
                block = words[1].lower()
                if block not in {"pair_soc_params", "soc_params", "orbital_groups"}:
                    raise ValueError(f"{path}:{lineno}: unsupported block {block!r}")
                continue
            if head == "end":
                if block is None:
                    raise ValueError(f"{path}:{lineno}: 'end' without active block")
                end_name = words[1].lower() if len(words) > 1 else block
                if end_name != block:
                    raise ValueError(f"{path}:{lineno}: end {end_name!r} does not match begin {block!r}")
                block = None
                continue
            if block in {"pair_soc_params", "soc_params"}:
                pair_rows.append(_parse_pair_soc_param_row(line, path=path, lineno=lineno))
                continue
            if block == "orbital_groups":
                orbital_group_rows.append(line)
                continue
            if block is not None:
                raise ValueError(f"{path}:{lineno}: unsupported content inside block {block!r}")
            if "=" in line:
                key, value = line.split("=", 1)
                cfg[key.strip().lower()] = value.strip()
            else:
                if len(words) < 2:
                    raise ValueError(f"{path}:{lineno}: expected 'key = value' or 'key value'")
                cfg[words[0].strip().lower()] = " ".join(words[1:]).strip()
    if block is not None:
        raise ValueError(f"{path}: missing 'end {block}'")
    if pair_rows:
        cfg["pair_soc_params"] = ";".join(pair_rows)
    if orbital_group_rows:
        cfg["orbital_groups"] = "\n".join(orbital_group_rows)
    return cfg


def _parse_pair_soc_param_row(line: str, *, path: str, lineno: int) -> str:
    parts = [x.strip() for x in line.replace(",", " ").split() if x.strip()]
    if len(parts) == 5:
        a, b, lam0, lo, hi = parts
        return f"{a}:{b}:{lam0}:{lo}:{hi}"
    if len(parts) == 3 and ":" in parts[0]:
        a, b = parts[0].split(":", 1)
        return f"{a}:{b}:{parts[1]}:{parts[2]}:{parts[2]}"
    if len(parts) == 4 and ":" in parts[0]:
        a, b = parts[0].split(":", 1)
        return f"{a}:{b}:{parts[1]}:{parts[2]}:{parts[3]}"
    raise ValueError(
        f"{path}:{lineno}: pair_soc_params row must be 'ElemRow ElemCol lambda0 lower upper', got {line!r}"
    )




def _normalize_pair_label(text: str) -> str:
    return str(text).strip().lower()


def _orbital_alias(name: str) -> str:
    aliases = {
        "dz^2": "dz2",
        "d_z2": "dz2",
        "d_z^2": "dz2",
        "dx2y2": "dx2-y2",
        "dx2-y2": "dx2-y2",
        "dx^2-y^2": "dx2-y2",
        "d_x2-y2": "dx2-y2",
        "d_x^2-y^2": "dx2-y2",
    }
    return aliases.get(str(name).strip().lower(), str(name).strip().lower())


def _group_orbital_names(orbital: str, *, p_order=WANNIER90_P_ORDER, d_order=WANNIER90_D_ORDER):
    orb = str(orbital).lower()
    if orb == "p":
        return [x.strip().lower() for x in str(p_order).replace(";", ",").split(",") if x.strip()]
    if orb == "d":
        return [_orbital_alias(x) for x in str(d_order).replace(";", ",").split(",") if x.strip()]
    if orb == "s":
        return ["s"]
    return []


def parse_orbital_group_specs(text: str) -> list[dict]:
    specs = []
    if text is None or str(text).strip() == "":
        return specs
    for lineno, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if not line:
            continue
        parts = [x.strip() for x in line.replace(",", " ").split() if x.strip()]
        if len(parts) < 3:
            raise ValueError(
                "orbital_groups row must be 'name element orbital_name ...', "
                f"got line {lineno}: {line!r}"
            )
        name = _normalize_pair_label(parts[0])
        elem = _normalize_pair_label(parts[1])
        names = [_orbital_alias(x) for x in parts[2:]]
        specs.append({"name": name, "element": elem, "orbital_names": names})
    names = [g["name"] for g in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate orbital group names: {names}")
    return specs


def _build_named_orbital_groups(win_path, nwan, orbital_group_specs, *, p_order=WANNIER90_P_ORDER, d_order=WANNIER90_D_ORDER):
    site_element, site_label, groups = _projection_site_metadata(win_path, nwan)
    named = {}
    for group in groups:
        elem = _normalize_pair_label(group["element"])
        default_name = f"{elem}_{str(group['orbital']).lower()}"
        idx = np.asarray(group["indices"], dtype=np.int64)
        named.setdefault(elem, np.zeros(int(nwan), dtype=bool))
        named[elem][idx] = True
        named.setdefault(default_name, np.zeros(int(nwan), dtype=bool))
        named[default_name][idx] = True
    summaries = []
    for spec in orbital_group_specs:
        mask = np.zeros(int(nwan), dtype=bool)
        for group in groups:
            if _normalize_pair_label(group["element"]) != spec["element"]:
                continue
            names = _group_orbital_names(group["orbital"], p_order=p_order, d_order=d_order)
            idx = np.asarray(group["indices"], dtype=np.int64)
            if len(names) != idx.size:
                raise ValueError(f"Orbital name count mismatch for group {group}")
            for name, oid in zip(names, idx):
                if _orbital_alias(name) in spec["orbital_names"]:
                    mask[int(oid)] = True
        if not np.any(mask):
            raise ValueError(f"Orbital group {spec['name']} matched no orbitals")
        named[spec["name"]] = mask
        summaries.append(f"{spec['name']}:{int(np.count_nonzero(mask))} orbitals")
    return named, summaries, groups

def parse_pair_soc_specs(text: str) -> list[PairSOCSpec]:
    specs: list[PairSOCSpec] = []
    for raw in str(text).split(";"):
        item = raw.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.replace(",", ":").split(":") if x.strip()]
        if len(parts) != 5:
            raise ValueError(
                "Each pair SOC spec must be row_element:col_element:lambda0:lower:upper, "
                f"got {item!r}"
            )
        a = _normalize_pair_label(parts[0])
        b = _normalize_pair_label(parts[1])
        lam0, lo, hi = map(float, parts[2:5])
        if hi < lo:
            raise ValueError(f"Invalid bounds for {item!r}: upper < lower")
        specs.append(PairSOCSpec(f"{a}:{b}", a, b, lam0, lo, hi))
    if not specs:
        raise ValueError("No pair SOC parameters specified")
    labels = [s.label for s in specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate pair SOC parameter labels: {labels}")
    return specs


def read_wannier_r_dat(path: str, nwan_expected: int | None = None) -> dict[tuple[int, int, int], np.ndarray]:
    """Read Wannier90 ``*_r.dat`` into R -> (3,nwan,nwan).

    Supports the standard Wannier90 layout with comment, ``num_wann``, ``nrpts``,
    degeneracy lines, then ``R i j Re(x) Im(x) Re(y) Im(y) Re(z) Im(z)`` rows.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"{path}: too few lines for Wannier90 r.dat")
    idx = 0
    try:
        nwan = int(lines[idx].split()[0])
        idx += 1
    except ValueError:
        idx += 1
        nwan = int(lines[idx].split()[0])
        idx += 1
    nrpts = int(lines[idx].split()[0])
    idx += 1
    degens: list[int] = []
    while len(degens) < nrpts and idx < len(lines):
        toks = lines[idx].split()
        if len(toks) >= 8:
            break
        degens.extend(int(x) for x in toks)
        idx += 1
    if nwan_expected is not None and int(nwan_expected) != int(nwan):
        raise ValueError(f"{path}: r.dat nwan={nwan}, expected {nwan_expected}")
    out: dict[tuple[int, int, int], np.ndarray] = {}
    for lineno, line in enumerate(lines[idx:], start=idx + 1):
        parts = line.split()
        if len(parts) < 11:
            raise ValueError(f"{path}:{lineno}: expected at least 11 columns, got {len(parts)}")
        r = (int(parts[0]), int(parts[1]), int(parts[2]))
        i = int(parts[3]) - 1
        j = int(parts[4]) - 1
        vals = np.array([
            float(parts[5]) + 1j * float(parts[6]),
            float(parts[7]) + 1j * float(parts[8]),
            float(parts[9]) + 1j * float(parts[10]),
        ], dtype=np.complex128)
        if r not in out:
            out[r] = np.zeros((3, nwan, nwan), dtype=np.complex128)
        out[r][:, i, j] = vals
    if not out:
        raise ValueError(f"{path}: no r.dat matrix rows parsed")
    return out


def _projection_site_metadata(win_path: str, nwan: int):
    _atoms, groups, nproj = _win_projection_groups(win_path)
    groups, _nproj_eff, _mode = _normalize_projection_groups_for_hamiltonian(win_path, groups, nproj, nwan)
    site_element = np.full(int(nwan), None, dtype=object)
    site_label = np.full(int(nwan), None, dtype=object)
    for group in groups:
        elem = _clean_species(group["element"])
        label = str(group["atom_label"])
        for idx in group["indices"]:
            site_element[int(idx)] = elem
            site_label[int(idx)] = label
    missing = [i for i, x in enumerate(site_element.tolist()) if x is None]
    if missing:
        raise ValueError(f"{win_path}: no projection site metadata for orbital indices {missing}")
    return site_element, site_label, groups


def _charge_hmap(h_up: dict, h_dn: dict, nwan: int) -> dict:
    keys = sorted(set(h_up) | set(h_dn))
    zero = np.zeros((int(nwan), int(nwan)), dtype=np.complex128)
    return {tuple(r): 0.5 * (np.asarray(h_up.get(r, zero)) + np.asarray(h_dn.get(r, zero))) for r in keys}


def _fourier_position_and_dhdk(rmap: dict, h0map: dict, kpts: np.ndarray, nwan: int, *, use_2pi=True):
    r_keys = list(rmap.keys())
    rvec_r = np.asarray(r_keys, dtype=np.float64)
    r_blocks = np.asarray([rmap[r] for r in r_keys], dtype=np.complex128)  # nR,3,n,n
    phase_r = np.exp(2.0j * np.pi * (np.asarray(kpts, dtype=np.float64) @ rvec_r.T))
    rk = np.einsum("kr,raij->kaij", phase_r, r_blocks, optimize=True)

    h_keys = list(h0map.keys())
    rvec_h = np.asarray(h_keys, dtype=np.float64)
    h_blocks = np.asarray([h0map[r] for r in h_keys], dtype=np.complex128)
    phase_h = np.exp(2.0j * np.pi * (np.asarray(kpts, dtype=np.float64) @ rvec_h.T))
    pref = (2.0j * np.pi if use_2pi else 1.0j) * rvec_h
    dhdk = np.einsum("kr,ra,rij->kaij", phase_h, pref, h_blocks, optimize=True)
    return rk, dhdk


def _orbital_l_from_r_cross_dhdk(rk: np.ndarray, dhdk: np.ndarray, *, hermitianize=True) -> np.ndarray:
    lx = rk[:, 1] * dhdk[:, 2] - rk[:, 2] * dhdk[:, 1]
    ly = rk[:, 2] * dhdk[:, 0] - rk[:, 0] * dhdk[:, 2]
    lz = rk[:, 0] * dhdk[:, 1] - rk[:, 1] * dhdk[:, 0]
    lk = np.stack([lx, ly, lz], axis=1)
    if hermitianize:
        lk = 0.5 * (lk + np.swapaxes(lk.conj(), 2, 3))
    return lk


def _spinor_template_from_lk(lk: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lk = np.asarray(lk, dtype=np.complex128) * np.asarray(mask, dtype=np.complex128)[None, None, :, :]
    nk, _three, nwan, _ = lk.shape
    out = np.zeros((nk, 2 * nwan, 2 * nwan), dtype=np.complex128)
    lx, ly, lz = lk[:, 0], lk[:, 1], lk[:, 2]
    out[:, :nwan, :nwan] = 0.5 * lz
    out[:, :nwan, nwan:] = 0.5 * (lx - 1j * ly)
    out[:, nwan:, :nwan] = 0.5 * (lx + 1j * ly)
    out[:, nwan:, nwan:] = -0.5 * lz
    return 0.5 * (out + np.swapaxes(out.conj(), 1, 2))




def _l_matrices_for_group(group, *, p_order=WANNIER90_P_ORDER, d_order=WANNIER90_D_ORDER):
    orb = str(group["orbital"]).lower()
    if orb == "p":
        return p_orbital_l_matrices(order=p_order)
    if orb == "d":
        return d_orbital_l_matrices(order=d_order)
    raise ValueError(f"Unsupported onsite-L orbital type {orb!r}; supported p,d")


def _bridge_matrix(nrow: int, ncol: int, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "ones":
        return np.ones((int(nrow), int(ncol)), dtype=np.complex128) / np.sqrt(float(int(nrow) * int(ncol)))
    if mode in {"identity", "rect_identity"}:
        out = np.zeros((int(nrow), int(ncol)), dtype=np.complex128)
        m = min(int(nrow), int(ncol))
        out[np.arange(m), np.arange(m)] = 1.0
        return out
    raise ValueError(f"Unsupported onsite_l_bridge={mode!r}; use ones or identity")


def build_pair_soc_templates_onsite_l(
    win_path,
    specs: Sequence[PairSOCSpec],
    kpts,
    nwan,
    *,
    bridge="ones",
    p_order=WANNIER90_P_ORDER,
    d_order=WANNIER90_D_ORDER,
    hermitianize=True,
    orbital_group_specs=None,
):
    """Build effective pair templates from row-site onsite atomic L only.

    For a pair parameter A:B, each row site of element A contributes its local
    atomic L matrix.  The local row-site L texture is extended to each column
    site of element B through a simple rectangular bridge.  This is an
    effective one-particle template for lambda_AB L_A . S_B; lambda_AB absorbs
    the intersite strength and bridge normalization.
    """
    _site_element, _site_label, groups = _projection_site_metadata(win_path, nwan)
    named_groups, group_summaries, groups = _build_named_orbital_groups(
        win_path,
        nwan,
        orbital_group_specs or [],
        p_order=p_order,
        d_order=d_order,
    )
    nk = len(kpts)
    templates = []
    summaries = []
    for spec in specs:
        l_pair = np.zeros((3, int(nwan), int(nwan)), dtype=np.complex128)
        nblock = 0
        row_mask = named_groups.get(spec.row_element)
        col_mask = named_groups.get(spec.col_element)
        if row_mask is None or col_mask is None:
            known = sorted(named_groups)
            raise ValueError(f"Unknown pair group in {spec.label}; known groups={known}")
        same_group = spec.row_element == spec.col_element
        for grow in groups:
            row_idx_all = np.asarray(grow["indices"], dtype=np.int64)
            row_idx = row_idx_all[row_mask[row_idx_all]]
            if row_idx.size == 0:
                continue
            l_full = _l_matrices_for_group(grow, p_order=p_order, d_order=d_order)
            row_pos = np.flatnonzero(row_mask[row_idx_all])
            l_mats = [np.asarray(lmat, dtype=np.complex128)[np.ix_(row_pos, row_pos)] for lmat in l_full]
            for gcol in groups:
                col_idx_all = np.asarray(gcol["indices"], dtype=np.int64)
                col_idx = col_idx_all[col_mask[col_idx_all]]
                if col_idx.size == 0:
                    continue
                same_site = int(grow.get("atom_index", -1)) == int(gcol.get("atom_index", -2))
                if same_group and not same_site:
                    continue
                if same_site and np.array_equal(row_idx, col_idx):
                    bridge_mat = np.eye(row_idx.size, dtype=np.complex128)
                else:
                    bridge_mat = _bridge_matrix(row_idx.size, col_idx.size, bridge)
                for ia, lmat in enumerate(l_mats):
                    l_pair[ia][np.ix_(row_idx, col_idx)] += lmat @ bridge_mat
                nblock += 1
        if nblock == 0:
            known = sorted({_clean_species(g["element"]) for g in groups})
            raise ValueError(f"No site-pair blocks matched {spec.label}; known elements={known}")
        lk = np.broadcast_to(l_pair[None, :, :, :], (nk, 3, int(nwan), int(nwan))).copy()
        templates.append(_spinor_template_from_lk(lk, np.ones((int(nwan), int(nwan)), dtype=bool)))
        mode_note = "onsite_orbit" if spec.row_element == spec.col_element else f"bridge={bridge}"
        summaries.append(f"{spec.label}: onsite_l blocks={nblock} {mode_note}")
    arr = np.asarray(templates, dtype=np.complex128)
    if not hermitianize:
        return arr, summaries, groups
    return arr, summaries, groups

def build_pair_soc_templates(rmap, h_up, h_dn, win_path, specs: Sequence[PairSOCSpec], kpts, nwan, *, use_2pi=True, hermitianize=True):
    site_element, _site_label, groups = _projection_site_metadata(win_path, nwan)
    h0map = _charge_hmap(h_up, h_dn, nwan)
    rk, dhdk = _fourier_position_and_dhdk(rmap, h0map, kpts, nwan, use_2pi=use_2pi)
    lk = _orbital_l_from_r_cross_dhdk(rk, dhdk, hermitianize=hermitianize)
    templates = []
    summaries = []
    row_elems = np.asarray(site_element, dtype=object)
    col_elems = np.asarray(site_element, dtype=object)
    for spec in specs:
        mask = (row_elems[:, None] == spec.row_element) & (col_elems[None, :] == spec.col_element)
        if not np.any(mask):
            known = sorted(set(str(x) for x in site_element.tolist()))
            raise ValueError(f"No orbital pairs matched {spec.label}; known elements={known}")
        templates.append(_spinor_template_from_lk(lk, mask))
        summaries.append(f"{spec.label}: npair={int(np.count_nonzero(mask))}")
    return np.asarray(templates, dtype=np.complex128), summaries, groups




def fit_soc_parameters_tracked(
    fitter,
    x0,
    bounds,
    *,
    maxiter=200,
    ftol=1e-12,
    gtol=1e-8,
    labels=None,
    history_path=None,
    verbose=True,
):
    from scipy.optimize import minimize

    labels = list(labels or [f"p{i}" for i in range(len(x0))])
    state = {"nfev": 0, "njev": 0, "nit": 0, "last_loss": None, "last_grad": None}
    t0 = time.time()
    hist_f = None
    if history_path:
        os.makedirs(os.path.dirname(os.path.abspath(history_path)) or ".", exist_ok=True)
        hist_f = open(history_path, "w", encoding="utf-8")
        hist_f.write("# iter nfev njev elapsed_s loss rms grad_norm " + " ".join(labels) + "\n")
        hist_f.flush()

    def log_line(iter_idx, x, loss, grad, *, tag="iter"):
        rms = float(np.sqrt(max(float(loss), 0.0)))
        gnorm = float(np.linalg.norm(grad)) if grad is not None else float("nan")
        elapsed = time.time() - t0
        vals = " ".join(f"{float(v):.10g}" for v in np.asarray(x, dtype=np.float64))
        line = (
            f"[soc-parameter-offsite][{tag}] iter={iter_idx} nfev={state['nfev']} njev={state['njev']} "
            f"elapsed={elapsed:.1f}s loss={float(loss):.12e} rms={rms:.6e} grad_norm={gnorm:.6e} params=[{vals}]"
        )
        if verbose:
            print(line, flush=True)
        if hist_f is not None:
            hist_f.write(
                f"{iter_idx} {state['nfev']} {state['njev']} {elapsed:.8e} {float(loss):.16e} "
                f"{rms:.16e} {gnorm:.16e} {vals}\n"
            )
            hist_f.flush()

    def fun(x):
        state["nfev"] += 1
        loss = fitter.loss(x)
        state["last_loss"] = float(loss)
        return loss

    def jac(x):
        state["njev"] += 1
        grad = fitter.gradient(x)
        state["last_grad"] = np.asarray(grad, dtype=np.float64)
        return grad

    def callback(xk):
        state["nit"] += 1
        loss = state["last_loss"] if state["last_loss"] is not None else fitter.loss(xk)
        grad = state["last_grad"] if state["last_grad"] is not None else fitter.gradient(xk)
        log_line(state["nit"], xk, loss, grad)

    x0 = np.asarray(x0, dtype=np.float64)
    log_line(0, x0, fitter.loss(x0), fitter.gradient(x0), tag="start")
    try:
        result = minimize(
            fun=fun,
            x0=x0,
            jac=jac,
            method="L-BFGS-B",
            bounds=list(bounds),
            callback=callback,
            options={"maxiter": int(maxiter), "ftol": float(ftol), "gtol": float(gtol)},
        )
        final_grad = fitter.gradient(result.x)
        log_line(int(result.nit), result.x, float(result.fun), final_grad, tag="final")
        return result
    finally:
        if hist_f is not None:
            hist_f.close()

class OffsiteSOCParameterFitter:
    def __init__(self, hk_up, hk_dn, hk_target, templates_k, *, model_efermi, target_efermi, spin_direction=(0, 0, 1), alpha=0.5, band_window=None):
        self.hk_up = np.asarray(hk_up, dtype=np.complex128)
        self.hk_dn = np.asarray(hk_dn, dtype=np.complex128)
        self.hk_target = np.asarray(hk_target, dtype=np.complex128)
        self.templates_k = np.asarray(templates_k, dtype=np.complex128)
        self.model_efermi = float(model_efermi)
        self.target_efermi = float(target_efermi)
        self.spin_direction = tuple(float(x) for x in spin_direction)
        self.alpha = float(alpha)
        self.band_window = None if band_window is None else int(band_window)
        if self.hk_up.shape != self.hk_dn.shape:
            raise ValueError(f"up/down H(k) shape mismatch: {self.hk_up.shape} vs {self.hk_dn.shape}")
        if self.hk_target.shape[-1] != 2 * self.hk_up.shape[-1]:
            raise ValueError(f"target dim={self.hk_target.shape[-1]}, expected {2*self.hk_up.shape[-1]}")
        if self.templates_k.shape[1:] != self.hk_target.shape:
            raise ValueError(f"template shape {self.templates_k.shape} incompatible with target {self.hk_target.shape}")
        self.h_base = spinor_from_collinear(self.hk_up, self.hk_dn, n=self.spin_direction)
        self.e_target = np.linalg.eigvalsh(self.hk_target)
        self.rel_target, self.ref_target = relative_energies(self.e_target, self.target_efermi)
        self.weights = band_weights(self.e_target.shape[1], self.ref_target, alpha=self.alpha, window=self.band_window)
        self.weight_norm = float(np.sum(self.weights))
        if self.weight_norm <= 0.0:
            raise ValueError("Band weights are all zero; increase band_window or adjust weight_alpha")

    def h_model(self, params: Sequence[float]) -> np.ndarray:
        p = np.asarray(params, dtype=np.float64).reshape(-1)
        if p.size != self.templates_k.shape[0]:
            raise ValueError(f"Expected {self.templates_k.shape[0]} parameters, got {p.size}")
        hsoc = np.einsum("a,akij->kij", p, self.templates_k, optimize=True)
        return self.h_base + hsoc

    def eigensystem(self, params: Sequence[float]):
        return np.linalg.eigh(self.h_model(params))

    def residual_data(self, params: Sequence[float]):
        evals, evecs = self.eigensystem(params)
        rel_model, ref_model = relative_energies(evals, self.model_efermi)
        return evals, evecs, rel_model, ref_model, rel_model - self.rel_target

    def loss(self, params: Sequence[float]) -> float:
        *_unused, residual = self.residual_data(params)
        return float(np.sum(self.weights * residual * residual) / self.weight_norm)

    def gradient(self, params: Sequence[float]) -> np.ndarray:
        _evals, evecs, _rel, ref_model, residual = self.residual_data(params)
        deps = np.einsum("kib,akij,kjb->akb", evecs.conj(), self.templates_k, evecs, optimize=True).real
        dref = deps[:, np.arange(deps.shape[1]), ref_model]
        drel = deps - dref[:, :, None]
        return 2.0 * np.einsum("kb,akb->a", self.weights * residual, drel, optimize=True) / self.weight_norm

    def rms(self, params: Sequence[float]) -> float:
        return float(np.sqrt(max(self.loss(params), 0.0)))


def _hk_to_hr_grid(hk: np.ndarray, kmesh: Sequence[int], drop_tol: float):
    kmesh = tuple(int(x) for x in kmesh)
    grid = np.asarray(hk, dtype=np.complex128).reshape(kmesh + hk.shape[-2:])
    hr_grid = np.fft.fftn(grid, axes=(0, 1, 2)) / float(np.prod(kmesh))
    out = {}
    for i in range(kmesh[0]):
        for j in range(kmesh[1]):
            for k in range(kmesh[2]):
                r = (
                    i if i <= kmesh[0] // 2 else i - kmesh[0],
                    j if j <= kmesh[1] // 2 else j - kmesh[1],
                    k if k <= kmesh[2] // 2 else k - kmesh[2],
                )
                block = np.asarray(hr_grid[i, j, k], dtype=np.complex128)
                if np.linalg.norm(block) > float(drop_tol):
                    out[r] = block
    return out


def _resolve_args(args):
    cfg = _read_parameter_offsite_input(args.input) if args.input else {}

    def pick(name, default=None):
        val = getattr(args, name)
        return val if val is not None else _cfg_value(cfg, name, default)

    args.up_hr = pick("up_hr")
    args.dn_hr = pick("dn_hr")
    args.target_soc_hr = pick("target_soc_hr")
    args.win = pick("win")
    args.r_dat = pick("r_dat")
    args.r_up_dat = pick("r_up_dat")
    args.r_dn_dat = pick("r_dn_dat")
    args.template_mode = str(pick("template_mode", "onsite_l")).strip().lower()
    args.orbital_groups = pick("orbital_groups")
    args.onsite_l_bridge = str(pick("onsite_l_bridge", "ones")).strip().lower()
    args.p_order = pick("p_order", WANNIER90_P_ORDER)
    args.d_order = pick("d_order", WANNIER90_D_ORDER)
    args.pair_soc_params = pick("pair_soc_params")
    args.efermi = float(pick("efermi")) if pick("efermi") is not None else None
    args.model_efermi = float(pick("model_efermi")) if pick("model_efermi") is not None else None
    args.target_efermi = float(pick("target_efermi")) if pick("target_efermi") is not None else None
    args.spin_direction = args.spin_direction if args.spin_direction is not None else _cfg_vec(cfg, "spin_direction", 3, [0.0, 0.0, 1.0], float)
    args.kmesh = args.kmesh if args.kmesh is not None else _cfg_vec(cfg, "kmesh", 3, [8, 8, 8], int)
    args.weight_alpha = float(pick("weight_alpha", 0.5))
    args.band_window = int(pick("band_window", -1))
    args.maxiter = int(pick("maxiter", 200))
    args.ftol = float(pick("ftol", 1e-12))
    args.gtol = float(pick("gtol", 1e-8))
    args.drop_tol = float(pick("drop_tol", 1e-10))
    args.out_hr = pick("out_hr")
    args.report = pick("report")
    args.save_npz = pick("save_npz")
    args.history = pick("history")
    args.verbose_iterations = int(pick("verbose_iterations", 1))
    required = ["up_hr", "dn_hr", "target_soc_hr", "win", "pair_soc_params"]
    missing = [name for name in required if getattr(args, name) in {None, ""}]
    has_single_r = getattr(args, "r_dat", None) not in {None, ""}
    has_spin_r = getattr(args, "r_up_dat", None) not in {None, ""} or getattr(args, "r_dn_dat", None) not in {None, ""}
    if has_spin_r and not (getattr(args, "r_up_dat", None) not in {None, ""} and getattr(args, "r_dn_dat", None) not in {None, ""}):
        missing.append("r_up_dat/r_dn_dat pair")
    if args.template_mode not in {"onsite_l", "r_cross"}:
        missing.append("template_mode must be onsite_l or r_cross")
    if args.template_mode == "r_cross" and not has_single_r and not has_spin_r:
        missing.append("r_dat or r_up_dat+r_dn_dat")
    if missing:
        src = f" in {args.input}" if args.input else ""
        raise ValueError(f"Missing required offsite SOC input(s){src}: {', '.join(missing)}")
    if args.model_efermi is None and args.efermi is None:
        raise ValueError("Provide --model_efermi or --efermi")
    if args.target_efermi is None and args.efermi is None:
        raise ValueError("Provide --target_efermi or --efermi")
    args.model_efermi = float(args.model_efermi if args.model_efermi is not None else args.efermi)
    args.target_efermi = float(args.target_efermi if args.target_efermi is not None else args.efermi)
    return args




def _bound_status(value: float, lower: float, upper: float, *, tol=1.0e-8) -> str:
    v = float(value)
    lo = float(lower)
    hi = float(upper)
    scale = max(1.0, abs(lo), abs(hi), abs(v))
    eps = float(tol) * scale
    if abs(v - lo) <= eps and abs(v - hi) <= eps:
        return "fixed"
    if abs(v - lo) <= eps:
        return "lower"
    if abs(v - hi) <= eps:
        return "upper"
    return "free"


def _format_param_summary(specs, x0, xfinal):
    rows = []
    for spec, initial, final in zip(specs, np.asarray(x0, dtype=float), np.asarray(xfinal, dtype=float)):
        rows.append(
            {
                "label": spec.label,
                "initial": float(initial),
                "final": float(final),
                "delta": float(final - initial),
                "lower": float(spec.lower),
                "upper": float(spec.upper),
                "bound": _bound_status(final, spec.lower, spec.upper),
            }
        )
    return rows

def run(args):
    args = _resolve_args(args)
    dim_up, _deg_up, h_up = _read_wannier_hr_compat(args.up_hr)
    dim_dn, _deg_dn, h_dn = _read_wannier_hr_compat(args.dn_hr)
    dim_t, _deg_t, h_t = _read_wannier_hr_compat(args.target_soc_hr)
    if dim_up != dim_dn:
        raise ValueError(f"up/dn dimension mismatch: {dim_up} vs {dim_dn}")
    if dim_t != 2 * dim_up:
        raise ValueError(f"target_soc_hr dim={dim_t}, expected {2*dim_up}")
    specs = parse_pair_soc_specs(args.pair_soc_params)
    orbital_group_specs = parse_orbital_group_specs(args.orbital_groups)
    kmesh = tuple(int(x) for x in args.kmesh)
    kpts = _full_k_mesh(kmesh)
    hk_up = _hmap_to_hk(h_up, kpts, dim_up)
    hk_dn = _hmap_to_hk(h_dn, kpts, dim_dn)
    hk_t = _hmap_to_hk(h_t, kpts, dim_t)
    if args.template_mode == "r_cross":
        if getattr(args, "r_up_dat", None) and getattr(args, "r_dn_dat", None):
            r_up = read_wannier_r_dat(args.r_up_dat, nwan_expected=dim_up)
            r_dn = read_wannier_r_dat(args.r_dn_dat, nwan_expected=dim_up)
            keys = sorted(set(r_up) | set(r_dn))
            zero_r = np.zeros((3, dim_up, dim_up), dtype=np.complex128)
            rmap = {tuple(r): 0.5 * (np.asarray(r_up.get(r, zero_r)) + np.asarray(r_dn.get(r, zero_r))) for r in keys}
            r_source = f"avg({args.r_up_dat},{args.r_dn_dat})"
        else:
            rmap = read_wannier_r_dat(args.r_dat, nwan_expected=dim_up)
            r_source = str(args.r_dat)
        templates_k, summaries, _groups = build_pair_soc_templates(
            rmap,
            h_up,
            h_dn,
            args.win,
            specs,
            kpts,
            dim_up,
            use_2pi=not bool(args.no_2pi_dhdk),
            hermitianize=not bool(args.no_hermitianize_templates),
        )
    else:
        r_source = "onsite_l"
        templates_k, summaries, _groups = build_pair_soc_templates_onsite_l(
            args.win,
            specs,
            kpts,
            dim_up,
            bridge=args.onsite_l_bridge,
            p_order=args.p_order,
            d_order=args.d_order,
            hermitianize=not bool(args.no_hermitianize_templates),
            orbital_group_specs=orbital_group_specs,
        )
    band_window = None if int(args.band_window) < 0 else int(args.band_window)
    fitter = OffsiteSOCParameterFitter(
        hk_up,
        hk_dn,
        hk_t,
        templates_k,
        model_efermi=args.model_efermi,
        target_efermi=args.target_efermi,
        spin_direction=args.spin_direction,
        alpha=args.weight_alpha,
        band_window=band_window,
    )
    x0 = np.asarray([s.lambda0 for s in specs], dtype=np.float64)
    bounds = [(s.lower, s.upper) for s in specs]
    print(f"[soc-parameter-offsite] nk={len(kpts)} nwan={dim_up} spinor_dim={dim_t} nparam={len(specs)}", flush=True)
    print(f"[soc-parameter-offsite] template_mode={args.template_mode} r_source={r_source}", flush=True)
    if orbital_group_specs:
        print("[soc-parameter-offsite] orbital_groups=" + ", ".join(g["name"] for g in orbital_group_specs), flush=True)
    for spec, summary in zip(specs, summaries):
        print(f"[soc-parameter-offsite] param {spec.label}: x0={spec.lambda0:g} bounds=({spec.lower:g},{spec.upper:g}) {summary}", flush=True)
    print(f"[soc-parameter-offsite] initial loss={fitter.loss(x0):.12e} rms={fitter.rms(x0):.12e}", flush=True)
    result = fit_soc_parameters_tracked(
        fitter,
        x0,
        bounds,
        maxiter=args.maxiter,
        ftol=args.ftol,
        gtol=args.gtol,
        labels=[s.label for s in specs],
        history_path=args.history,
        verbose=bool(args.verbose_iterations),
    )
    print(f"[soc-parameter-offsite] success={bool(result.success)} status={result.status} message={result.message}", flush=True)
    print(f"[soc-parameter-offsite] final loss={float(result.fun):.12e} rms={fitter.rms(result.x):.12e}", flush=True)
    param_rows = _format_param_summary(specs, x0, result.x)
    for row in param_rows:
        print(
            f"[soc-parameter-offsite] lambda {row['label']} = {row['final']:.12g} "
            f"(initial={row['initial']:.12g} delta={row['delta']:.3e} "
            f"bounds=[{row['lower']:.12g},{row['upper']:.12g}] bound={row['bound']})",
            flush=True,
        )
    h_final = fitter.h_model(result.x)
    if args.out_hr:
        out_hmap = _hk_to_hr_grid(h_final, kmesh, args.drop_tol)
        write_wannier_hr(
            args.out_hr,
            dim_t,
            [1] * len(out_hmap),
            out_hmap,
            header="SLW fitted offsite SOC model; " + "; ".join(f"{s.label}={v:.12g}" for s, v in zip(specs, result.x)),
        )
        print(f"[soc-parameter-offsite] wrote {args.out_hr}", flush=True)
    if args.save_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)) or ".", exist_ok=True)
        np.savez_compressed(
            args.save_npz,
            kpts_frac=kpts,
            params=result.x,
            labels=np.asarray([s.label for s in specs], dtype=object),
            templates_k=templates_k,
            h_final=h_final,
            loss=np.asarray(float(result.fun)),
        )
        print(f"[soc-parameter-offsite] wrote {args.save_npz}", flush=True)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("# SLW offsite SOC parameter fitting report\n")
            f.write(f"success = {bool(result.success)}\n")
            f.write(f"message = {result.message}\n")
            f.write(f"kmesh = {kmesh}\n")
            f.write(f"model_efermi = {args.model_efermi:.12g}\n")
            f.write(f"target_efermi = {args.target_efermi:.12g}\n")
            f.write(f"template_mode = {args.template_mode}\n")
            f.write(f"r_source = {r_source}\n")
            if orbital_group_specs:
                f.writelines(f"orbital_group_{group['name']} = {group['element']} {' '.join(group['orbital_names'])}\n" for group in orbital_group_specs)
            f.write(f"initial_loss = {fitter.loss(x0):.12e}\n")
            f.write(f"final_loss = {float(result.fun):.12e}\n")
            f.write(f"final_rms = {fitter.rms(result.x):.12e}\n")
            for row, summary in zip(param_rows, summaries):
                f.write(f"lambda_{row['label']} = {row['final']:.12g}\n")
                f.write(f"initial_{row['label']} = {row['initial']:.12g}\n")
                f.write(f"delta_{row['label']} = {row['delta']:.12e}\n")
                f.write(f"bounds_{row['label']} = {row['lower']:.12g} {row['upper']:.12g}\n")
                f.write(f"bound_status_{row['label']} = {row['bound']}\n")
                f.write(f"summary_{row['label']} = {summary}\n")
        print(f"[soc-parameter-offsite] wrote {args.report}", flush=True)
    return result


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Fit k-dependent site-pair SOC parameters from Wannier90 r.dat")
    ap.add_argument("--input", "--in_file", default=None, help="Offsite SOC fitting .in file")
    ap.add_argument("--up_hr", default=None)
    ap.add_argument("--dn_hr", default=None)
    ap.add_argument("--target_soc_hr", default=None)
    ap.add_argument("--win", default=None)
    ap.add_argument("--r_dat", default=None, help="Spin-independent/charge-channel Wannier90 r.dat")
    ap.add_argument("--r_up_dat", default=None, help="Spin-up Wannier90 r.dat; averaged with --r_dn_dat")
    ap.add_argument("--r_dn_dat", default=None, help="Spin-down Wannier90 r.dat; averaged with --r_up_dat")
    ap.add_argument("--template_mode", choices=["onsite_l", "r_cross"], default=None, help="SOC template definition. Default onsite_l uses row-site atomic L only; r_cross uses r x dH/dk.")
    ap.add_argument("--orbital_groups", default=None, help="Inline orbital group definitions; usually provided by begin orbital_groups block in --input")
    ap.add_argument("--onsite_l_bridge", choices=["ones", "identity"], default=None, help="Rectangular bridge for onsite_l offsite blocks")
    ap.add_argument("--p_order", default=None)
    ap.add_argument("--d_order", default=None)
    ap.add_argument("--pair_soc_params", default=None, help="Short form: 'Cr:I:0:0:1;I:Cr:0:0:1;I:I:0.5:0:2'")
    ap.add_argument("--efermi", type=float, default=None)
    ap.add_argument("--model_efermi", type=float, default=None)
    ap.add_argument("--target_efermi", type=float, default=None)
    ap.add_argument("--spin_direction", type=float, nargs=3, default=None)
    ap.add_argument("--kmesh", type=int, nargs=3, default=None)
    ap.add_argument("--weight_alpha", type=float, default=None)
    ap.add_argument("--band_window", type=int, default=None)
    ap.add_argument("--maxiter", type=int, default=None)
    ap.add_argument("--ftol", type=float, default=None)
    ap.add_argument("--gtol", type=float, default=None)
    ap.add_argument("--drop_tol", type=float, default=None)
    ap.add_argument("--no_2pi_dhdk", action="store_true", help="Use iR instead of 2*pi*iR in dH/dk")
    ap.add_argument("--no_hermitianize_templates", action="store_true")
    ap.add_argument("--out_hr", default=None)
    ap.add_argument("--save_npz", default=None)
    ap.add_argument("--history", default=None, help="Optional iteration history text file")
    ap.add_argument("--verbose_iterations", type=int, default=None, help="Print per-iteration optimizer progress; default 1")
    ap.add_argument("--report", default=None)
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
