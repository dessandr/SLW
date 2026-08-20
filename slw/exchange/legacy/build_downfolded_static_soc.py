# Legacy exchange implementation; use the native engine for new workflows.
"""Build a static downfolded ligand-SOC spin-flip Hamiltonian.

The calculator starts from collinear up/down Hamiltonians and constructs an
effective spin-flip term on a selected low-energy subspace.  Two downfolding
modes are available:

    perturbative:   dH_ud(k) = H_up[d,p] G_up[p] Lambda_ud G_down[p] H_down[p,d]
    soc_propagator: dH(k)    = H_dI(k) [E - H_II(k) - Lambda_SOC]^-1 H_Id(k)

For the default full-basis model, soc_propagator embeds only the SOC-induced
part, Sigma(lambda)-Sigma(0), so that lambda=0 leaves the original collinear
Hamiltonian unchanged.  Use --socprop_reference none to write the raw
Schur-complement contribution.

By default the output is the full spin-major Hamiltonian: the original
collinear base embedded as [up|down], plus the selected downfolded correction
embedded back into the full Wannier basis.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from slw.core.wannier_io import read_wannier_hr, write_wannier_hr
from slw.exchange.legacy.diagnose_J_epr_kspace import _build_hk_from_epr, _full_k_mesh
from slw.exchange.legacy.plot_epr_soc_bands import _clean_species, _win_projection_groups
from slw.exchange.legacy.spinor_model import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    atomic_d_soc_block,
    atomic_p_soc_block,
    spinor_from_collinear,
)


def _read_wannier_hr_compat(path):
    parsed = read_wannier_hr(path)
    if len(parsed) == 4:
        dim, _nrpts, degens, hmap = parsed
        return int(dim), list(degens), hmap
    if len(parsed) == 3:
        dim, degens, hmap = parsed
        return int(dim), list(degens), hmap
    raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")


def _parse_selector(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.replace(",", ":").split(":") if x.strip()]
        if len(parts) != 2:
            raise ValueError(f"Expected selector element:orbital, got {item!r}")
        elem = _clean_species(parts[0])
        orb = parts[1].lower()
        if orb not in {"s", "p", "d"}:
            raise ValueError(f"Unsupported orbital selector {orb!r}; use s, p, or d")
        out.append((elem, orb))
    if not out:
        raise ValueError("Empty selector")
    return out


def _read_win_scalar_int(win_path, key):
    key_l = str(key).strip().lower()
    with open(win_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("!", 1)[0].split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            if lhs.strip().lower() != key_l:
                continue
            return int(rhs.split()[0])
    return None


def _read_win_spinor_flag(win_path):
    with open(win_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("!", 1)[0].split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            if lhs.strip().lower() == "spin":
                return rhs.strip().lower().split()[0] == "spinor"
    return False


def _normalize_projection_groups_for_hamiltonian(win_path, groups, nproj, nwan):
    nproj = int(nproj)
    nwan = int(nwan)
    num_wann = _read_win_scalar_int(win_path, "num_wann")
    spinor_flag = _read_win_spinor_flag(win_path)

    if nproj == nwan:
        return groups, nproj, "projection_count_matches_hamiltonian"

    if nproj == 2 * nwan:
        normalized = []
        for group in groups:
            idx = [int(x) for x in group["indices"]]
            if all((x % 2) == 0 for x in idx):
                new_idx = [x // 2 for x in idx]
            elif max(idx) < nwan:
                new_idx = idx
            else:
                raise ValueError(
                    f"{win_path}: projection count is 2*nwan ({nproj} vs {nwan}), "
                    f"but group {group['atom_label']}:{group['orbital']} indices={idx} cannot be collapsed safely."
                )
            g2 = dict(group)
            g2["indices"] = new_idx
            normalized.append(g2)
        return normalized, nwan, "collapsed_explicit_spin_duplicated_projections"

    detail = (
        f"{win_path}: projection count from begin projections is {nproj}, "
        f"Hamiltonian nwan is {nwan}, num_wann in win is {num_wann}, spinor={spinor_flag}. "
        "This tool needs collinear orbital indices matching up_hr/dn_hr. "
        "Use the collinear .win for up/dn HR, or provide a projections block whose orbital count matches nwan."
    )
    raise ValueError(detail)


def _selected_groups(win_path, selector, nwan):
    _atoms, groups, nproj = _win_projection_groups(win_path)
    groups, nproj_eff, projection_mode = _normalize_projection_groups_for_hamiltonian(
        win_path, groups, nproj, nwan
    )
    specs = _parse_selector(selector)
    matched = []
    for elem, orb in specs:
        found = [g for g in groups if g["element"] == elem and g["orbital"] == orb]
        if not found:
            known = sorted({(g["element"], g["orbital"]) for g in groups})
            raise ValueError(
                f"No {elem}:{orb} groups found in {win_path}; known={known}; "
                f"projection_mode={projection_mode}, effective_nproj={nproj_eff}, Hamiltonian nwan={nwan}"
            )
        matched.extend(found)
    return matched


def _group_indices(groups, *, expected_orbital=None):
    idx = []
    summary = []
    for group in groups:
        if expected_orbital is not None and group["orbital"] != expected_orbital:
            raise ValueError(f"Expected {expected_orbital} group, got {group}")
        vals = [int(x) for x in group["indices"]]
        idx.extend(vals)
        summary.append(f"{group['atom_label']}:{group['orbital']}{vals}")
    return np.asarray(idx, dtype=np.int64), summary


def _spinor_slice_suggestion(groups):
    spans = []
    offset = 0
    for isite, group in enumerate(groups):
        n = len(group["indices"])
        spans.append((isite, offset, offset + n))
        offset += n
    out = []
    for isite, up0, up1 in spans:
        out.append(f"{isite}:{up0}:{up1}:{offset + up0}:{offset + up1}")
    return ",".join(out)


def _spinor_slice_suggestion_full(groups, nwan):
    out = []
    for isite, group in enumerate(groups):
        idx = np.asarray(group["indices"], dtype=np.int64).reshape(-1)
        up0 = int(idx.min())
        up1 = int(idx.max()) + 1
        if not np.array_equal(idx, np.arange(up0, up1, dtype=np.int64)):
            vals = idx.tolist()
            raise ValueError(
                "Full-space spinor_slices currently require each d-subspace group to be contiguous; "
                f"group {group.get('atom_label', isite)} has indices {vals}."
            )
        out.append(f"{isite}:{up0}:{up1}:{int(nwan) + up0}:{int(nwan) + up1}")
    return ",".join(out)


def _embed_spinor_block(full, block, d_idx, nwan):
    nd = int(d_idx.size)
    up = np.asarray(d_idx, dtype=np.int64)
    dn = up + int(nwan)
    full[np.ix_(up, up)] += block[:nd, :nd]
    full[np.ix_(up, dn)] += block[:nd, nd:]
    full[np.ix_(dn, up)] += block[nd:, :nd]
    full[np.ix_(dn, dn)] += block[nd:, nd:]


def _full_spinor_hr_from_collinear_maps(h_up, h_dn, nwan, *, add_base, hsoc=None):
    keys = sorted(set(h_up) | set(h_dn) | {(0, 0, 0)})
    out = {}
    zero = np.zeros((int(nwan), int(nwan)), dtype=np.complex128)
    for r in keys:
        up = np.asarray(h_up.get(r, zero), dtype=np.complex128)
        dn = np.asarray(h_dn.get(r, zero), dtype=np.complex128)
        block = np.zeros((2 * int(nwan), 2 * int(nwan)), dtype=np.complex128)
        if add_base:
            block[:nwan, :nwan] = up
            block[nwan:, nwan:] = dn
        if tuple(r) == (0, 0, 0) and hsoc is not None:
            block += np.asarray(hsoc, dtype=np.complex128)
        if np.linalg.norm(block) > 0.0:
            out[tuple(r)] = block
    return out


def _soc_matrix_for_groups(groups, nwan, lambda_ev, *, p_order, d_order):
    out = np.zeros((2 * int(nwan), 2 * int(nwan)), dtype=np.complex128)
    summary = []
    for group in groups:
        orb = group["orbital"]
        if orb == "p":
            block = atomic_p_soc_block(lambda_ev, order=p_order)
        elif orb == "d":
            block = atomic_d_soc_block(lambda_ev, order=d_order)
        else:
            raise ValueError(f"SOC-active orbital must be p or d, got {orb!r}")
        norb = block.shape[0] // 2
        idx = np.asarray(group["indices"], dtype=np.int64).reshape(norb)
        sidx = np.concatenate([idx, idx + int(nwan)])
        out[np.ix_(sidx, sidx)] += block
        summary.append(f"{group['atom_label']}:{orb}{idx.tolist()}")
    return out, summary


def _build_hk_from_hr(hmap, kpts):
    keys = list(hmap.keys())
    dim = next(iter(hmap.values())).shape[0]
    rvec = np.asarray(keys, dtype=np.float64)
    blocks = np.asarray([hmap[r] for r in keys], dtype=np.complex128)
    phase = np.exp(2.0j * np.pi * (np.asarray(kpts, dtype=np.float64) @ rvec.T))
    hk = np.einsum("kr,rij->kij", phase, blocks, optimize=True)
    return 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))


def _load_collinear_hk(args, kpts):
    if args.epr_up or args.epr_dn:
        if not (args.epr_up and args.epr_dn):
            raise ValueError("Provide both --epr_up and --epr_dn")
        hk_up = _build_hk_from_epr(args.epr_up, kpts, unit=args.hr_unit)
        hk_dn = _build_hk_from_epr(args.epr_dn, kpts, unit=args.hr_unit)
        return hk_up, hk_dn, None
    if not (args.up_hr and args.dn_hr):
        raise ValueError("Provide either --epr_up/--epr_dn or --up_hr/--dn_hr")
    dim_up, _deg_up, h_up = _read_wannier_hr_compat(args.up_hr)
    dim_dn, _deg_dn, h_dn = _read_wannier_hr_compat(args.dn_hr)
    if dim_up != dim_dn:
        raise ValueError(f"HR dimension mismatch: up={dim_up} dn={dim_dn}")
    return _build_hk_from_hr(h_up, kpts), _build_hk_from_hr(h_dn, kpts), (h_up, h_dn)


def _downfold_green(h, p_idx, e0, eta):
    eye_p = np.eye(int(p_idx.size), dtype=np.complex128)
    return np.linalg.inv((float(e0) + 1j * float(eta)) * eye_p - h[np.ix_(p_idx, p_idx)])


def _downfold_spin_conserving_base_one_k(h, d_idx, p_idx, e0, eta):
    g = _downfold_green(h, p_idx, e0, eta)
    h_dd = h[np.ix_(d_idx, d_idx)]
    h_dp = h[np.ix_(d_idx, p_idx)]
    h_pd = h[np.ix_(p_idx, d_idx)]
    return h_dd + h_dp @ g @ h_pd


def _downfold_one_k(hup, hdn, d_idx, p_idx, lambda_uu, lambda_ud, lambda_du, lambda_dd, e0, eta):
    nd = int(d_idx.size)
    gp_up = _downfold_green(hup, p_idx, e0, eta)
    gp_dn = _downfold_green(hdn, p_idx, e0, eta)
    h_up_dp = hup[np.ix_(d_idx, p_idx)]
    h_dn_dp = hdn[np.ix_(d_idx, p_idx)]
    h_up_pd = hup[np.ix_(p_idx, d_idx)]
    h_dn_pd = hdn[np.ix_(p_idx, d_idx)]

    # Downfold the full ligand SOC block, including spin-conserving Lz Sz terms.
    uu = h_up_dp @ gp_up @ lambda_uu @ gp_up @ h_up_pd
    ud = h_up_dp @ gp_up @ lambda_ud @ gp_dn @ h_dn_pd
    du = h_dn_dp @ gp_dn @ lambda_du @ gp_up @ h_up_pd
    dd = h_dn_dp @ gp_dn @ lambda_dd @ gp_dn @ h_dn_pd

    out = np.zeros((2 * nd, 2 * nd), dtype=np.complex128)
    out[:nd, :nd] = uu
    out[:nd, nd:] = ud
    out[nd:, :nd] = du
    out[nd:, nd:] = dd
    return out


def _downfold_soc_propagator_k(hk_up, hk_dn, d_idx, p_idx, hsoc_p, e0, eta, *, reference="lambda0"):
    """Batch Schur complement H_dI (z-H_II-SOC)^-1 H_Id.

    The returned object is not only the first-order SOC vertex.  It is the
    full ligand-mediated contribution in the SOC-split ligand subspace.
    """
    z = complex(float(e0), float(eta))
    nk = int(hk_up.shape[0])
    nd = int(d_idx.size)
    np_ = int(p_idx.size)

    hpp = spinor_from_collinear(
        hk_up[:, p_idx[:, None], p_idx],
        hk_dn[:, p_idx[:, None], p_idx],
    )
    hpp = hpp + np.asarray(hsoc_p, dtype=np.complex128)[None, :, :]

    hdp = np.zeros((nk, 2 * nd, 2 * np_), dtype=np.complex128)
    hpd = np.zeros((nk, 2 * np_, 2 * nd), dtype=np.complex128)
    hdp[:, :nd, :np_] = hk_up[:, d_idx[:, None], p_idx]
    hdp[:, nd:, np_:] = hk_dn[:, d_idx[:, None], p_idx]
    hpd[:, :np_, :nd] = hk_up[:, p_idx[:, None], d_idx]
    hpd[:, np_:, nd:] = hk_dn[:, p_idx[:, None], d_idx]

    eye = np.eye(2 * np_, dtype=np.complex128)
    g = np.linalg.inv(z * eye[None, :, :] - hpp)
    sigma = hdp @ g @ hpd

    ref = str(reference).strip().lower()
    if ref in {"none", "raw"}:
        return sigma
    if ref in {"lambda0", "zero", "zero_soc"}:
        hpp0 = hpp - np.asarray(hsoc_p, dtype=np.complex128)[None, :, :]
        g0 = np.linalg.inv(z * eye[None, :, :] - hpp0)
        return sigma - (hdp @ g0 @ hpd)
    raise ValueError(f"Unsupported soc_propagator reference={reference!r}; use lambda0 or none")



def _write_text_dump(path, args, *, d_summary, soc_summary, spinor_slices, norms, herm_rel, out_hmap):
    rows = []
    for r, block in out_hmap.items():
        nd = block.shape[0] // 2
        uu = block[:nd, :nd]
        ud = block[:nd, nd:]
        du = block[nd:, :nd]
        dd = block[nd:, nd:]
        rows.append(
            (
                float(np.linalg.norm(block)),
                tuple(int(x) for x in r),
                float(np.linalg.norm(uu)),
                float(np.linalg.norm(ud)),
                float(np.linalg.norm(du)),
                float(np.linalg.norm(dd)),
                float(np.max(np.abs(block))) if block.size else 0.0,
            )
        )
    rows.sort(reverse=True, key=lambda x: x[0])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SLW downfolded static SOC report\n")
        f.write(f"# output_hr = {os.path.abspath(args.output)}\n")
        f.write(f"# d_subspace = {args.d_subspace}\n")
        f.write(f"# soc_active = {args.soc_active}\n")
        f.write(f"# d_groups = {d_summary}\n")
        f.write(f"# soc_groups = {soc_summary}\n")
        f.write(f"# suggested_spinor_slices = {spinor_slices}\n")
        f.write(f"# output_space = {args.output_space}\n")
        if str(args.output_space).strip().lower() == "full":
            f.write(f"# full_mode = {args.full_mode}\n")
        f.write(f"# downfold_mode = {args.downfold_mode}\n")
        if str(args.downfold_mode).strip().lower() == "soc_propagator":
            f.write(f"# socprop_reference = {args.socprop_reference}\n")
        f.write(f"# lambda_soc_eV = {float(args.lambda_soc):.12g}\n")
        f.write(f"# e0_eV = {float(args.e0):.12g}\n")
        f.write(f"# eta_eV = {float(args.eta):.12g}\n")
        f.write(f"# kmesh = {tuple(int(x) for x in args.kmesh)}\n")
        f.write(f"# add_base = {bool(args.add_base)}\n")
        if bool(args.add_base):
            f.write(f"# base_mode = {args.base_mode}\n")
        f.write(f"# hermitianize = {bool(args.hermitianize)}\n")
        f.write(f"# nrpts_out = {len(out_hmap)}\n")
        f.write(f"# k_norm_min = {float(norms.min()):.12e}\n")
        f.write(f"# k_norm_mean = {float(norms.mean()):.12e}\n")
        f.write(f"# k_norm_max = {float(norms.max()):.12e}\n")
        f.write(f"# hermiticity_rel_max = {float(herm_rel):.12e}\n\n")
        f.write("# R_x R_y R_z  norm_full  norm_uu  norm_ud  norm_du  norm_dd  max_abs\n")
        for norm, r, nuu, nud, ndu, ndd, mx in rows:
            f.write(
                f"{r[0]:5d} {r[1]:5d} {r[2]:5d} "
                f"{norm: .12e} {nuu: .12e} {nud: .12e} {ndu: .12e} {ndd: .12e} {mx: .12e}\n"
            )

def build_downfolded_static_soc(args):
    kmesh = tuple(int(x) for x in args.kmesh)
    kpts = _full_k_mesh(kmesh)
    hk_up, hk_dn, hr_maps = _load_collinear_hk(args, kpts)
    if hk_up.shape != hk_dn.shape:
        raise ValueError(f"H(k) shape mismatch: {hk_up.shape} vs {hk_dn.shape}")
    nwan = int(hk_up.shape[-1])

    d_groups = _selected_groups(args.win, args.d_subspace, nwan)
    p_groups = _selected_groups(args.win, args.soc_active, nwan)
    d_idx, d_summary = _group_indices(d_groups)
    output_space = str(args.output_space).strip().lower()
    if output_space not in {"full", "d"}:
        raise ValueError(f"Unsupported --output_space={args.output_space!r}; use full or d")
    spinor_slices = _spinor_slice_suggestion_full(d_groups, nwan)
    p_idx, _p_summary = _group_indices(p_groups)
    if len(set(d_idx.tolist())) != int(d_idx.size):
        raise ValueError(f"Duplicate d-subspace orbital indices: {d_idx.tolist()}")
    if len(set(p_idx.tolist())) != int(p_idx.size):
        raise ValueError(f"Duplicate SOC-active orbital indices: {p_idx.tolist()}")

    hsoc, soc_summary = _soc_matrix_for_groups(
        p_groups,
        nwan,
        args.lambda_soc,
        p_order=args.p_order,
        d_order=args.d_order,
    )
    hsoc_p = hsoc[np.ix_(np.concatenate([p_idx, p_idx + nwan]), np.concatenate([p_idx, p_idx + nwan]))]
    np_ = int(p_idx.size)
    lambda_uu = hsoc_p[:np_, :np_]
    lambda_ud = hsoc_p[:np_, np_:]
    lambda_du = hsoc_p[np_:, :np_]
    lambda_dd = hsoc_p[np_:, np_:]

    print(
        f"[downfold-static-soc] kmesh={kmesh} nk={len(kpts)} nwan={nwan} nd={d_idx.size} np={p_idx.size}",
        flush=True,
    )
    print(f"[downfold-static-soc] d_subspace={d_summary}", flush=True)
    print(f"[downfold-static-soc] soc_active={soc_summary}", flush=True)
    downfold_mode = str(args.downfold_mode).strip().lower()
    print(
        f"[downfold-static-soc] E0={args.e0:g} eta={args.eta:g} "
        f"lambda={args.lambda_soc:g} eV mode={downfold_mode}",
        flush=True,
    )

    direct_out_hmap = None
    if downfold_mode == "perturbative":
        delta_k = np.empty((len(kpts), 2 * d_idx.size, 2 * d_idx.size), dtype=np.complex128)
        for ik in range(len(kpts)):
            delta_k[ik] = _downfold_one_k(
                hk_up[ik],
                hk_dn[ik],
                d_idx,
                p_idx,
                lambda_uu,
                lambda_ud,
                lambda_du,
                lambda_dd,
                args.e0,
                args.eta,
            )
    elif downfold_mode == "soc_propagator":
        delta_k = _downfold_soc_propagator_k(
            hk_up,
            hk_dn,
            d_idx,
            p_idx,
            hsoc_p,
            args.e0,
            args.eta,
            reference=args.socprop_reference,
        )
    else:
        raise ValueError(f"Unsupported --downfold_mode={args.downfold_mode!r}")

    if args.hermitianize:
        delta_k = 0.5 * (delta_k + np.swapaxes(delta_k.conj(), 1, 2))

    base_mode = str(args.base_mode).strip().lower()

    if output_space == "d":
        if downfold_mode == "soc_propagator":
            raise ValueError(
                "--downfold_mode soc_propagator now writes the full Hamiltonian by design; "
                "omit --output_space d."
            )
        if args.correction_only:
            pass
        elif base_mode == "bare":
            base = spinor_from_collinear(
                hk_up[:, d_idx[:, None], d_idx],
                hk_dn[:, d_idx[:, None], d_idx],
            )
            delta_k = base + delta_k
        elif base_mode == "downfolded":
            base_up = np.empty((len(kpts), d_idx.size, d_idx.size), dtype=np.complex128)
            base_dn = np.empty_like(base_up)
            for ik in range(len(kpts)):
                base_up[ik] = _downfold_spin_conserving_base_one_k(hk_up[ik], d_idx, p_idx, args.e0, args.eta)
                base_dn[ik] = _downfold_spin_conserving_base_one_k(hk_dn[ik], d_idx, p_idx, args.e0, args.eta)
            if args.hermitianize:
                base_up = 0.5 * (base_up + np.swapaxes(base_up.conj(), 1, 2))
                base_dn = 0.5 * (base_dn + np.swapaxes(base_dn.conj(), 1, 2))
            delta_k = spinor_from_collinear(base_up, base_dn) + delta_k
        else:
            raise ValueError(f"Unsupported --base_mode={args.base_mode!r}; use bare or downfolded")
    else:
        full_k = np.zeros((len(kpts), 2 * nwan, 2 * nwan), dtype=np.complex128)
        if not args.correction_only:
            full_k = spinor_from_collinear(hk_up, hk_dn)
        for ik in range(len(kpts)):
            _embed_spinor_block(full_k[ik], delta_k[ik], d_idx, nwan)
        delta_k = full_k

    norms = np.linalg.norm(delta_k.reshape(delta_k.shape[0], -1), axis=1)
    herm_rel = np.linalg.norm((delta_k - np.swapaxes(delta_k.conj(), 1, 2)).reshape(delta_k.shape[0], -1), axis=1)
    herm_rel = float(np.max(herm_rel / np.maximum(norms, 1.0e-30)))

    tol = float(args.drop_tol)
    if direct_out_hmap is not None:
        out_hmap = {r: block for r, block in direct_out_hmap.items() if np.linalg.norm(block) > tol}
    else:
        grid = delta_k.reshape(kmesh + (delta_k.shape[-2], delta_k.shape[-1]))
        # _build_hk_from_hr uses exp(+2*pi*i*k.R), so invert with the FFT sign.
        hr_grid = np.fft.fftn(grid, axes=(0, 1, 2)) / float(len(kpts))
        out_hmap = {}
        for i in range(kmesh[0]):
            for j in range(kmesh[1]):
                for k in range(kmesh[2]):
                    r = (
                        i if i <= kmesh[0] // 2 else i - kmesh[0],
                        j if j <= kmesh[1] // 2 else j - kmesh[1],
                        k if k <= kmesh[2] // 2 else k - kmesh[2],
                    )
                    block = np.asarray(hr_grid[i, j, k], dtype=np.complex128)
                    if np.linalg.norm(block) > tol:
                        out_hmap[r] = block
    if not out_hmap:
        raise ValueError("Downfolded SOC HR is empty; lower --drop_tol or check selections")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    write_wannier_hr(
        args.output,
        int(delta_k.shape[-1]),
        [1] * len(out_hmap),
        out_hmap,
        header=(
            "SLW downfolded static SOC"
            f"; mode={args.downfold_mode}; d={args.d_subspace}; p={args.soc_active}; "
            f"lambda={args.lambda_soc:g}; E0={args.e0:g}"
        ),
    )
    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            kpts_frac=kpts,
            delta_k=delta_k,
            d_indices=d_idx,
            p_indices=p_idx,
            d_summary=np.asarray(d_summary, dtype=object),
            soc_summary=np.asarray(soc_summary, dtype=object),
            spinor_slices=np.asarray(spinor_slices, dtype=object),
            e0=np.asarray(args.e0),
            eta=np.asarray(args.eta),
            lambda_soc=np.asarray(args.lambda_soc),
            downfold_mode=np.asarray(args.downfold_mode),
            socprop_reference=np.asarray(args.socprop_reference),
            hermitianized=np.asarray(bool(args.hermitianize)),
            correction_only=np.asarray(bool(args.correction_only)),
            output_space=np.asarray(output_space),
            base_mode=np.asarray(args.base_mode),
        )
    print("[downfold-static-soc] summary", flush=True)
    print(f"  output: {args.output}", flush=True)
    print(f"  spinor_dim: {delta_k.shape[-1]}", flush=True)
    print(f"  output_space: {output_space}", flush=True)
    print(f"  downfold_mode: {args.downfold_mode}", flush=True)
    if downfold_mode == "soc_propagator":
        print(f"  socprop_reference: {args.socprop_reference}", flush=True)
    print(f"  correction_only: {bool(args.correction_only)}", flush=True)
    print(f"  suggested_spinor_slices: {spinor_slices}", flush=True)
    print(f"  nrpts_out: {len(out_hmap)} drop_tol={tol:g}", flush=True)
    print(f"  k_norm_min_mean_max: {float(norms.min()):.6g} {float(norms.mean()):.6g} {float(norms.max()):.6g}", flush=True)
    print(f"  hermiticity_rel_max: {herm_rel:.3e}", flush=True)
    if args.dump_txt:
        _write_text_dump(
            args.dump_txt,
            args,
            d_summary=d_summary,
            soc_summary=soc_summary,
            spinor_slices=spinor_slices,
            norms=norms,
            herm_rel=herm_rel,
            out_hmap=out_hmap,
        )
        print(f"  dump_txt: {args.dump_txt}", flush=True)
    if args.save_npz:
        print(f"  save_npz: {args.save_npz}", flush=True)
    return args.output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_argument_group("collinear Hamiltonian input")
    src.add_argument("--epr_up", default=None)
    src.add_argument("--epr_dn", default=None)
    src.add_argument("--up_hr", default=None)
    src.add_argument("--dn_hr", default=None)
    src.add_argument("--hr_unit", default="ry", help="Unit for EPR electron_wannier input")
    ap.add_argument("--win", required=True, help="Wannier90 .win defining projection groups")
    ap.add_argument("--d_subspace", required=True, help="Low-energy target subspace, e.g. M:d")
    ap.add_argument("--soc_active", required=True, help="SOC-active ligand subspace, e.g. X:p")
    ap.add_argument("--lambda_soc", type=float, required=True, help="Atomic SOC lambda in eV")
    ap.add_argument(
        "--downfold_mode",
        choices=["perturbative", "soc_propagator"],
        default="perturbative",
        help=(
            "perturbative uses H_dI G_I Lambda_SOC G_I H_Id; "
            "soc_propagator uses H_dI (E-H_II-Lambda_SOC)^-1 H_Id directly."
        ),
    )
    ap.add_argument("--e0", type=float, default=0.0, help="Downfolding energy E0 in eV")
    ap.add_argument("--eta", type=float, default=0.0, help="Small broadening in eV")
    ap.add_argument(
        "--socprop_reference",
        choices=["lambda0", "none"],
        default="lambda0",
        help=(
            "Reference subtracted in soc_propagator mode. "
            "lambda0 writes Sigma(lambda)-Sigma(0); none writes raw Sigma(lambda)."
        ),
    )
    ap.add_argument("--kmesh", type=int, nargs=3, required=True)
    ap.add_argument("--p_order", default=WANNIER90_P_ORDER)
    ap.add_argument("--d_order", default=WANNIER90_D_ORDER)
    ap.add_argument("--hermitianize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--output_space",
        choices=["full", "d"],
        default="full",
        help="Output basis. Default full writes the original full spinor H plus embedded correction.",
    )
    ap.add_argument("--correction_only", action="store_true", help="Write only the downfolded correction instead of H_base + correction")
    ap.add_argument(
        "--base_mode",
        choices=["downfolded", "bare"],
        default="bare",
        help="Base used for d-subspace output. Full output always uses the original full H as base unless --correction_only is set.",
    )
    ap.add_argument("--drop_tol", type=float, default=1.0e-10)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--save_npz", default=None)
    ap.add_argument("--dump_txt", default=None, help="Human-readable text report path")
    # Deprecated compatibility: accepted but ignored; full output now always embeds the downfolded correction.
    ap.add_argument("--add_base", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--full_mode", choices=["atomic", "embed_downfolded"], default="embed_downfolded", help=argparse.SUPPRESS)
    args = ap.parse_args()
    build_downfolded_static_soc(args)


if __name__ == "__main__":
    main()
