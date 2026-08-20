"""Magnon band plot from compute_J_epr_tensor HDF5 output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import h5py
import numpy as np

from slw.magph.legacy.plotting.common import build_kpath, point_table, prepare_matplotlib
from slw.magph.legacy.lswt import build_spin_frame_info, bdg_matrix_from_tensor, solve_tensor_lswt
from slw.magph.legacy.utils import parsing_POSCAR


@dataclass(frozen=True)
class JPayload:
    path: Path
    lattice_ang: np.ndarray
    atom_pos_ang: np.ndarray
    atom_labels: list[str]
    ii: np.ndarray
    jj: np.ndarray
    rr: np.ndarray
    j_mev: np.ndarray
    shell: np.ndarray | None
    distance_ang: np.ndarray | None
    j_source: str
    mag_atom_ids: np.ndarray


def _decode_strings(arr) -> list[str]:
    out = []
    for value in np.asarray(arr).tolist():
        if isinstance(value, bytes):
            out.append(value.decode())
        else:
            out.append(str(value))
    return out


def _read_structure_from_h5(h5) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lattice = np.asarray(h5["basic_data/lattice_ang"], dtype=np.float64).reshape(3, 3)
    atom_pos = np.asarray(h5["basic_data/tau_cart_ang"], dtype=np.float64).reshape(-1, 3)
    if "basic_data/atom_labels" in h5:
        labels = _decode_strings(h5["basic_data/atom_labels"][:])
    else:
        labels = [f"Atom{i + 1}" for i in range(atom_pos.shape[0])]
    return lattice, atom_pos, labels


def _read_structure(config, h5) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    structure = config.get("structure")
    if structure:
        path = config.path_value("structure", must_exist=True)
        lattice, labels, atom_pos = parsing_POSCAR(path)
        return (
            np.asarray(lattice, dtype=np.float64).reshape(3, 3),
            np.asarray(atom_pos, dtype=np.float64).reshape(-1, 3),
            list(labels),
            str(path),
        )
    return (*_read_structure_from_h5(h5), "h5:basic_data")


def _extract_j_values(h5, source: str, component: str) -> tuple[np.ndarray, str]:
    source = source.strip().lower()
    component = component.strip().lower()
    if source == "auto":
        candidates = ["J_iso_r", "tb2j_extra/jiso_tb2j", "J_iso_tensor_r", "J_tensor_r"]
    else:
        candidates = [source]

    for key in candidates:
        if key not in h5:
            continue
        data = np.asarray(h5[key], dtype=np.float64)
        if data.ndim == 1:
            return data, key
        if data.ndim == 3 and data.shape[1:] == (3, 3):
            if component == "iso":
                return np.trace(data, axis1=1, axis2=2) / 3.0, key
            if component in {"xx", "yy", "zz"}:
                idx = {"xx": 0, "yy": 1, "zz": 2}[component]
                return data[:, idx, idx], key
            raise ValueError(f"Unsupported component '{component}' for tensor J data")
        raise ValueError(f"Unsupported J dataset shape for {key}: {data.shape}")

    raise KeyError(f"Could not find J data source '{source}' in HDF5")


def load_j_h5(config) -> tuple[JPayload, str]:
    path = config.path_value("input", must_exist=True)
    j_source = config.get_str("j_source", "auto")
    component = config.get_str("component", "iso")

    with h5py.File(path, "r") as h5:
        required = [
            "bonds/mag_i_atom",
            "bonds/mag_j_atom",
            "bonds/R",
            "basic_data/lattice_ang",
            "basic_data/tau_cart_ang",
        ]
        missing = [key for key in required if key not in h5]
        if missing:
            raise KeyError(f"Missing HDF5 datasets in {path}: {missing}")

        lattice, atom_pos, labels, structure_source = _read_structure(config, h5)
        mag_i = np.asarray(h5["bonds/mag_i_atom"], dtype=np.int64).reshape(-1)
        mag_j = np.asarray(h5["bonds/mag_j_atom"], dtype=np.int64).reshape(-1)
        rr = np.asarray(h5["bonds/R"], dtype=np.int64).reshape(-1, 3)
        shell = np.asarray(h5["bonds/shell"], dtype=np.int64).reshape(-1) if "bonds/shell" in h5 else None
        distance_ang = (
            np.asarray(h5["bonds/distance_ang"], dtype=np.float64).reshape(-1)
            if "bonds/distance_ang" in h5
            else None
        )
        j_mev, used_j_source = _extract_j_values(h5, j_source, component)

    if not (mag_i.size == mag_j.size == rr.shape[0] == j_mev.size):
        raise ValueError(
            "Bond array size mismatch: "
            f"i={mag_i.size}, j={mag_j.size}, R={rr.shape[0]}, J={j_mev.size}"
        )

    mag_atom_ids = np.unique(np.concatenate([mag_i, mag_j]))
    local = {int(atom_id): idx for idx, atom_id in enumerate(mag_atom_ids)}
    ii = np.asarray([local[int(x)] for x in mag_i], dtype=np.int64)
    jj = np.asarray([local[int(x)] for x in mag_j], dtype=np.int64)

    mask = _bond_filter_mask(config, j_mev, shell, distance_ang)
    n_before = int(j_mev.size)
    if not np.all(mask):
        ii = ii[mask]
        jj = jj[mask]
        rr = rr[mask]
        j_mev = j_mev[mask]
        if shell is not None:
            shell = shell[mask]
        if distance_ang is not None:
            distance_ang = distance_ang[mask]
    if j_mev.size == 0:
        raise ValueError("All bonds were removed by plot.in filters")
    if config.get_bool("debug_bonds", False) or config.get_bool("debug_bdg", False):
        _print_filter_summary(n_before, j_mev, shell, distance_ang)

    if np.max(mag_atom_ids) >= atom_pos.shape[0]:
        raise ValueError(
            f"Magnetic atom index exceeds structure atoms: max={np.max(mag_atom_ids)}, nat={atom_pos.shape[0]}"
        )

    return (
        JPayload(
            path=path,
            lattice_ang=lattice,
            atom_pos_ang=atom_pos[mag_atom_ids],
            atom_labels=[labels[int(i)] if int(i) < len(labels) else f"Atom{int(i) + 1}" for i in mag_atom_ids],
            ii=ii,
            jj=jj,
            rr=rr,
            j_mev=np.asarray(j_mev, dtype=np.float64),
            shell=None if shell is None else np.asarray(shell, dtype=np.int64),
            distance_ang=None if distance_ang is None else np.asarray(distance_ang, dtype=np.float64),
            j_source=used_j_source,
            mag_atom_ids=mag_atom_ids,
        ),
        structure_source,
    )


def _parse_int_set(tokens: list[str], key: str) -> set[int]:
    out: set[int] = set()
    for token in tokens:
        try:
            out.add(int(token))
        except ValueError as exc:
            raise ValueError(f"{key} must contain integer shell indices, got '{token}'") from exc
    return out


def _bond_filter_mask(config, j_mev: np.ndarray, shell, distance_ang) -> np.ndarray:
    j = np.asarray(j_mev, dtype=np.float64).reshape(-1)
    mask = np.ones(j.shape[0], dtype=bool)

    include_shells = _parse_int_set(config.get_list("include_shells"), "include_shells")
    exclude_shells = _parse_int_set(config.get_list("exclude_shells"), "exclude_shells")
    if include_shells or exclude_shells:
        if shell is None:
            raise KeyError("plot.in requested shell filtering, but HDF5 has no bonds/shell dataset")
        sh = np.asarray(shell, dtype=np.int64).reshape(-1)
        if include_shells:
            mask &= np.isin(sh, np.fromiter(include_shells, dtype=np.int64))
        if exclude_shells:
            mask &= ~np.isin(sh, np.fromiter(exclude_shells, dtype=np.int64))

    min_distance = config.get("min_distance")
    max_distance = config.get("max_distance")
    if min_distance is not None or max_distance is not None:
        if distance_ang is None:
            raise KeyError("plot.in requested distance filtering, but HDF5 has no bonds/distance_ang dataset")
        dist = np.asarray(distance_ang, dtype=np.float64).reshape(-1)
        if min_distance is not None:
            mask &= dist >= float(min_distance)
        if max_distance is not None:
            mask &= dist <= float(max_distance)

    min_abs_j = config.get("min_abs_j")
    max_abs_j = config.get("max_abs_j")
    if min_abs_j is not None:
        mask &= np.abs(j) >= float(min_abs_j)
    if max_abs_j is not None:
        mask &= np.abs(j) <= float(max_abs_j)

    return mask


def _print_filter_summary(n_before: int, j_mev: np.ndarray, shell, distance_ang) -> None:
    print("[magnon_h5:debug] bond filter")
    print(f"  bonds = {int(j_mev.size)} / {int(n_before)}")
    print(f"  J range = {float(np.min(j_mev)):.8g} .. {float(np.max(j_mev)):.8g} meV")
    if distance_ang is not None:
        print(f"  distance range = {float(np.min(distance_ang)):.8g} .. {float(np.max(distance_ang)):.8g} Ang")
    if shell is None:
        return
    for sh in np.unique(shell):
        m = shell == sh
        vals = j_mev[m]
        if distance_ang is not None:
            dvals = distance_ang[m]
            dtext = f", d={float(np.min(dvals)):.6g}..{float(np.max(dvals)):.6g} Ang"
        else:
            dtext = ""
        print(
            f"  shell {int(sh)}: count={int(np.count_nonzero(m))}, "
            f"sumJ={float(np.sum(vals)):.8g}, min={float(np.min(vals)):.8g}, max={float(np.max(vals)):.8g}{dtext}"
        )


def _spin_pattern(config, nmag: int) -> np.ndarray:
    tokens = config.get_list("spin_pattern")
    if not tokens:
        return np.where(np.arange(nmag) % 2 == 0, 1.0, -1.0)
    if len(tokens) == 1:
        mode = tokens[0].strip().lower()
        if mode in {"auto", "afm"}:
            return np.where(np.arange(nmag) % 2 == 0, 1.0, -1.0)
        if mode in {"fm", "ferro", "ferromagnetic"}:
            return np.ones(nmag, dtype=np.float64)
    vals = np.asarray([float(tok) for tok in tokens], dtype=np.float64)
    if vals.size != nmag:
        raise ValueError(f"spin_pattern length {vals.size} does not match magnetic atoms {nmag}")
    signs = np.sign(vals)
    if np.any(signs == 0.0):
        raise ValueError("spin_pattern entries must be nonzero")
    return signs


def _j_prefactor(config, spin: float) -> float:
    token = config.get_str("j_prefactor", "auto").strip().lower().replace(" ", "")
    if token in {"auto", "1/s", "unit-vector", "unit_vector"}:
        return 1.0 / spin
    if token in {"1", "none"}:
        return 1.0
    if token == "s":
        return spin
    return float(token)


def _bdg_energies_at_k(
    kfrac: np.ndarray,
    payload: JPayload,
    spin: float,
    spin_pattern: np.ndarray,
    prefactor: float,
    bond_factor: float,
    anisotropy_mev: float,
    hermitize: bool,
    zero_mode_tol: float,
    bond_class: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nmag = spin_pattern.size
    h = _bdg_matrix_at_k(
        kfrac,
        payload,
        spin_pattern,
        prefactor,
        bond_factor,
        anisotropy_mev,
        hermitize,
        bond_class,
    )
    eta = np.diag(np.r_[np.ones(nmag), -np.ones(nmag)])

    eigvals, eigvecs = np.linalg.eig(eta @ h)
    order = np.argsort(eigvals.real)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    norms = np.real(np.einsum("ij,ij->j", eigvecs.conj(), eta @ eigvecs))
    imag_tol = 1.0e-7
    mask = (eigvals.real >= -1.0e-9) & (np.abs(eigvals.imag) <= imag_tol) & (norms > 1.0e-8)
    positive = eigvals.real[mask]
    positive = positive[np.argsort(positive)]

    if positive.size < nmag:
        zero_count = int(np.count_nonzero(np.abs(eigvals) <= float(zero_mode_tol)) // 2)
        modes = np.full(nmag, np.nan, dtype=np.float64)
        fill = min(zero_count, nmag)
        if fill:
            modes[:fill] = 0.0
        tail = min(positive.size, nmag - fill)
        if tail:
            modes[fill:fill + tail] = positive[:tail]
        return modes, eigvals, norms
    return positive[:nmag], eigvals, norms


def _bdg_matrix_at_k(
    kfrac: np.ndarray,
    payload: JPayload,
    spin_pattern: np.ndarray,
    prefactor: float,
    bond_factor: float,
    anisotropy_mev: float,
    hermitize: bool,
    bond_class: str = "spin_pattern",
) -> np.ndarray:
    nmag = spin_pattern.size
    delta = payload.rr @ payload.lattice_ang + payload.atom_pos_ang[payload.jj] - payload.atom_pos_ang[payload.ii]
    reciprocal = 2.0 * np.pi * np.linalg.inv(payload.lattice_ang).T
    kcart = np.asarray(kfrac, dtype=np.float64) @ reciprocal
    j_eff = np.asarray(payload.j_mev, dtype=np.float64) * prefactor * bond_factor
    mode = str(bond_class).strip().lower().replace("-", "_")
    if mode in {"spin", "spin_pattern", "order"}:
        parallel = spin_pattern[payload.ii] == spin_pattern[payload.jj]
        anti = ~parallel
    elif mode in {"sign", "j_sign", "j"}:
        parallel = j_eff >= 0.0
        anti = j_eff < 0.0
    else:
        raise ValueError("bond_class must be one of: spin_pattern, sign")

    def blocks_at(sign: float) -> tuple[np.ndarray, np.ndarray]:
        phase = np.exp(1j * sign * (delta @ kcart))
        a = np.zeros((nmag, nmag), dtype=np.complex128)
        b = np.zeros((nmag, nmag), dtype=np.complex128)

        if np.any(parallel):
            idx = np.where(parallel)[0]
            val = j_eff[idx]
            np.add.at(a, (payload.ii[idx], payload.ii[idx]), val)
            np.add.at(a, (payload.ii[idx], payload.jj[idx]), -val * phase[idx])

        if np.any(anti):
            idx = np.where(anti)[0]
            val = -j_eff[idx]
            np.add.at(a, (payload.ii[idx], payload.ii[idx]), val)
            np.add.at(b, (payload.ii[idx], payload.jj[idx]), val * phase[idx])

        if anisotropy_mev:
            a += np.eye(nmag, dtype=np.complex128) * float(anisotropy_mev)
        return a, b

    a_k, b_k = blocks_at(1.0)
    a_mk, b_mk = blocks_at(-1.0)
    h = np.block([[a_k, b_k], [b_mk.conj(), a_mk.conj()]])
    if hermitize:
        h = 0.5 * (h + h.conj().T)
    return h


def solve_magnon_bands(config, payload: JPayload, kfrac: np.ndarray) -> np.ndarray:
    _LAST_FULL_BDG.clear()
    spin = config.get_float("S", 2.5)
    if spin <= 0.0:
        raise ValueError("S must be positive")

    spin_pattern = _spin_pattern(config, payload.atom_pos_ang.shape[0])
    solver = config.get_str("solver", "full_bdg").strip().lower()
    if solver in {"full", "full_bdg", "full-bdg", "all"}:
        if config.get_str("bond_class", "sign").strip().lower().replace("-", "_") in {"sign", "j_sign", "j"}:
            return _solve_magnon_bands_full_bdg_manual(config, payload, kfrac, spin, spin_pattern)
        return _solve_magnon_bands_full_bdg(config, payload, kfrac, spin, spin_pattern)
    if solver in {"local_frame", "local-frame", "lswt"}:
        return _solve_magnon_bands_local_frame(config, payload, kfrac, spin, spin_pattern)
    if solver not in {"strict", "debug", "manual"}:
        raise ValueError("solver must be one of: local_frame, full_bdg, strict")

    prefactor = _j_prefactor(config, spin)
    bond_factor = config.get_float("bond_factor", 1.0)
    anisotropy = config.get_float("anisotropy_mev", 0.0)
    hermitize = config.get_bool("hermitize", True)
    zero_mode_tol = config.get_float("zero_mode_tol", 1.0e-8)
    bond_class = config.get_str("bond_class", "spin_pattern")

    bands = np.zeros((kfrac.shape[0], payload.atom_pos_ang.shape[0]), dtype=np.float64)
    max_imag = np.zeros(kfrac.shape[0], dtype=np.float64)
    missing = np.zeros(kfrac.shape[0], dtype=np.int64)
    for ik, k in enumerate(kfrac):
        vals, eigvals, _ = _bdg_energies_at_k(
            k,
            payload,
            spin,
            spin_pattern,
            prefactor,
            bond_factor,
            anisotropy,
            hermitize,
            zero_mode_tol,
            bond_class,
        )
        bands[ik] = vals
        max_imag[ik] = float(np.max(np.abs(eigvals.imag)))
        missing[ik] = int(np.count_nonzero(~np.isfinite(vals)))
    if np.any(~np.isfinite(bands)):
        bad = np.where(np.any(~np.isfinite(bands), axis=1))[0]
        preview = ", ".join(str(int(x)) for x in bad[:8])
        warnings.warn(
            "BdG spectrum has complex/unstable or incomplete positive-norm modes; "
            f"marking them as NaN instead of clipping. bad k indices: {preview}",
            RuntimeWarning,
        )
    finite = bands[np.isfinite(bands)]
    if finite.size and np.min(finite) < -1.0e-7:
        warnings.warn(f"Negative magnon mode found: min={np.min(finite):.6g} meV", RuntimeWarning)
    if config.get_bool("debug_bdg", False):
        _print_bdg_debug(config, payload, kfrac, bands, max_imag, missing, spin_pattern, prefactor, bond_factor, anisotropy, hermitize, bond_class)
    return bands


def _solve_magnon_bands_full_bdg(
    config,
    payload: JPayload,
    kfrac: np.ndarray,
    spin: float,
    spin_pattern: np.ndarray,
) -> np.ndarray:
    prefactor = _j_prefactor(config, spin)
    bond_factor = config.get_float("bond_factor", 1.0)
    anisotropy = config.get_float("anisotropy_mev", 0.0)
    spin_direction = np.asarray(
        [float(x) for x in config.get_list("spin_direction", ["0", "1", "0"])],
        dtype=np.float64,
    )
    if spin_direction.size != 3:
        raise ValueError("spin_direction must contain three numbers")
    spin_directions = spin_pattern[:, None] * spin_direction[None, :]
    spin_info = build_spin_frame_info(
        spin_direction=spin_direction,
        spin_directions=spin_directions,
    )
    tensor = payload.j_mev[:, None, None] * prefactor * np.eye(3, dtype=np.float64)[None, :, :]
    reciprocal = 2.0 * np.pi * np.linalg.inv(payload.lattice_ang).T
    kcart = np.asarray(kfrac, dtype=np.float64) @ reciprocal
    nmag = spin_pattern.size
    eta = np.diag(np.r_[np.ones(nmag), -np.ones(nmag)]).astype(np.complex128)
    eigvals_all = np.zeros((kcart.shape[0], 2 * nmag), dtype=np.complex128)
    norms_all = np.zeros((kcart.shape[0], 2 * nmag), dtype=np.float64)
    for ik, k in enumerate(kcart):
        h = bdg_matrix_from_tensor(
            k,
            1.0,
            tensor,
            payload.ii,
            payload.jj,
            payload.rr,
            payload.lattice_ang,
            payload.atom_pos_ang,
            spin_info.frames,
            bond_factor=bond_factor,
            anisotropy_mev=anisotropy,
        )
        vals, vecs = np.linalg.eig(eta @ h)
        order = np.lexsort((vals.imag, vals.real))
        vals = vals[order]
        vecs = vecs[:, order]
        eigvals_all[ik] = vals
        norms_all[ik] = np.real(np.einsum("ij,ij->j", vecs.conj(), eta @ vecs))

    _LAST_FULL_BDG["eigvals"] = eigvals_all
    _LAST_FULL_BDG["norms"] = norms_all
    if config.get_bool("debug_bdg", False):
        print("[magnon_h5:debug] full BdG eigenvalues")
        print(f"  channels = {2 * nmag}")
        print(f"  spin_pattern = {spin_pattern.tolist()}")
        print(f"  spin_direction = {spin_direction.tolist()}")
        print(f"  prefactor = {prefactor}, bond_factor = {bond_factor}, anisotropy_mev = {anisotropy}")
        print(f"  real_range = {float(np.min(eigvals_all.real)):.8g} .. {float(np.max(eigvals_all.real)):.8g} meV")
        print(f"  max_abs_imag = {float(np.max(np.abs(eigvals_all.imag))):.8g} meV")
        _print_full_bdg_debug(config, kfrac, eigvals_all, norms_all)
    return eigvals_all.real


def _solve_magnon_bands_full_bdg_manual(
    config,
    payload: JPayload,
    kfrac: np.ndarray,
    spin: float,
    spin_pattern: np.ndarray,
) -> np.ndarray:
    prefactor = _j_prefactor(config, spin)
    bond_factor = config.get_float("bond_factor", 1.0)
    anisotropy = config.get_float("anisotropy_mev", 0.0)
    hermitize = config.get_bool("hermitize", True)
    bond_class = config.get_str("bond_class", "sign")
    nmag = spin_pattern.size
    eta = np.diag(np.r_[np.ones(nmag), -np.ones(nmag)]).astype(np.complex128)
    eigvals_all = np.zeros((kfrac.shape[0], 2 * nmag), dtype=np.complex128)
    norms_all = np.zeros((kfrac.shape[0], 2 * nmag), dtype=np.float64)
    for ik, k in enumerate(kfrac):
        h = _bdg_matrix_at_k(
            k,
            payload,
            spin_pattern,
            prefactor,
            bond_factor,
            anisotropy,
            hermitize,
            bond_class,
        )
        vals, vecs = np.linalg.eig(eta @ h)
        order = np.lexsort((vals.imag, vals.real))
        vals = vals[order]
        vecs = vecs[:, order]
        eigvals_all[ik] = vals
        norms_all[ik] = np.real(np.einsum("ij,ij->j", vecs.conj(), eta @ vecs))

    _LAST_FULL_BDG["eigvals"] = eigvals_all
    _LAST_FULL_BDG["norms"] = norms_all
    if config.get_bool("debug_bdg", False):
        _print_bond_class_summary(payload, spin_pattern, prefactor, bond_class)
        print("[magnon_h5:debug] full BdG eigenvalues")
        print(f"  builder = manual, bond_class = {bond_class}")
        print(f"  channels = {2 * nmag}")
        print(f"  spin_pattern = {spin_pattern.tolist()}")
        print(f"  prefactor = {prefactor}, bond_factor = {bond_factor}, anisotropy_mev = {anisotropy}, hermitize = {hermitize}")
        print(f"  real_range = {float(np.min(eigvals_all.real)):.8g} .. {float(np.max(eigvals_all.real)):.8g} meV")
        print(f"  max_abs_imag = {float(np.max(np.abs(eigvals_all.imag))):.8g} meV")
        _print_full_bdg_debug(config, kfrac, eigvals_all, norms_all)
    return eigvals_all.real


_LAST_FULL_BDG: dict[str, np.ndarray] = {}


def _energy_scale(config) -> tuple[str, float, str, str]:
    raw = config.get_str("energy_unit", "meV").strip().lower().replace(" ", "")
    aliases = {
        "mev": ("meV", 1.0, "Energy (meV)", "energy_mev"),
        "thz": ("THz", 1.0 / 4.135667696, "Frequency (THz)", "energy_thz"),
        "cm-1": ("cm^-1", 8.065543937, r"Energy (cm$^{-1}$)", "energy_cm-1"),
        "cm^-1": ("cm^-1", 8.065543937, r"Energy (cm$^{-1}$)", "energy_cm-1"),
        "cm1": ("cm^-1", 8.065543937, r"Energy (cm$^{-1}$)", "energy_cm-1"),
        "1/cm": ("cm^-1", 8.065543937, r"Energy (cm$^{-1}$)", "energy_cm-1"),
    }
    if raw not in aliases:
        raise ValueError("energy_unit must be one of: meV, THz, cm-1")
    return aliases[raw]


def _print_full_bdg_debug(config, kfrac: np.ndarray, eigvals: np.ndarray, norms: np.ndarray) -> None:
    labels = config.get_list("debug_points", ["G", "M", "K"])
    points = point_table(config)
    for item in labels:
        key = item.strip().lower()
        if key in points:
            target = points[key]
            ik = int(np.argmin(np.linalg.norm(kfrac - target[None, :], axis=1)))
        else:
            ik = int(item)
        print(f"  k[{ik}] frac={kfrac[ik].tolist()}")
        for ic, (val, norm) in enumerate(zip(eigvals[ik], norms[ik])):
            print(f"    ch{ic}: eig = {val.real:.10g} {val.imag:+.3g}j, norm = {norm:.6g}")


def _print_bond_class_summary(payload: JPayload, spin_pattern: np.ndarray, prefactor: float, bond_class: str) -> None:
    j_eff = np.asarray(payload.j_mev, dtype=np.float64) * float(prefactor)
    align = spin_pattern[payload.ii] * spin_pattern[payload.jj]
    cases = [
        ("same_spin,J<0", (align > 0.0) & (j_eff < 0.0)),
        ("same_spin,J>0", (align > 0.0) & (j_eff > 0.0)),
        ("opp_spin,J<0", (align < 0.0) & (j_eff < 0.0)),
        ("opp_spin,J>0", (align < 0.0) & (j_eff > 0.0)),
    ]
    print("[magnon_h5:debug] bond classification")
    print(f"  bond_class = {bond_class}")
    for label, mask in cases:
        count = int(np.count_nonzero(mask))
        if count:
            vals = j_eff[mask]
            print(
                f"  {label}: count={count}, sumJ={float(np.sum(vals)):.8g}, "
                f"min={float(np.min(vals)):.8g}, max={float(np.max(vals)):.8g}"
            )
        else:
            print(f"  {label}: count=0, sumJ=0")


def _solve_magnon_bands_local_frame(
    config,
    payload: JPayload,
    kfrac: np.ndarray,
    spin: float,
    spin_pattern: np.ndarray,
) -> np.ndarray:
    prefactor = _j_prefactor(config, spin)
    bond_factor = config.get_float("bond_factor", 1.0)
    anisotropy = config.get_float("anisotropy_mev", 0.0)
    spin_direction = np.asarray(
        [float(x) for x in config.get_list("spin_direction", ["0", "1", "0"])],
        dtype=np.float64,
    )
    if spin_direction.size != 3:
        raise ValueError("spin_direction must contain three numbers")
    spin_directions = spin_pattern[:, None] * spin_direction[None, :]
    spin_info = build_spin_frame_info(
        spin_direction=spin_direction,
        spin_directions=spin_directions,
    )

    tensor = payload.j_mev[:, None, None] * prefactor * np.eye(3, dtype=np.float64)[None, :, :]
    reciprocal = 2.0 * np.pi * np.linalg.inv(payload.lattice_ang).T
    kcart = np.asarray(kfrac, dtype=np.float64) @ reciprocal
    bands, _ = solve_tensor_lswt(
        kcart,
        1.0,
        tensor,
        payload.ii,
        payload.jj,
        payload.rr,
        payload.lattice_ang,
        payload.atom_pos_ang,
        spin_info,
        bond_factor=bond_factor,
        anisotropy_mev=anisotropy,
    )
    if config.get_bool("debug_bdg", False):
        finite = bands[np.isfinite(bands)]
        print("[magnon_h5:debug] local-frame LSWT")
        print(f"  spin_pattern = {spin_pattern.tolist()}")
        print(f"  spin_direction = {spin_direction.tolist()}")
        print(f"  prefactor = {prefactor}, bond_factor = {bond_factor}, anisotropy_mev = {anisotropy}")
        if finite.size:
            print(f"  energy_range = {float(np.min(finite)):.8g} .. {float(np.max(finite)):.8g} meV")
        _print_local_frame_debug(config, payload, kfrac, bands)
    return bands


def _print_local_frame_debug(config, payload: JPayload, kfrac: np.ndarray, bands: np.ndarray) -> None:
    labels = config.get_list("debug_points", ["G", "M", "K"])
    points = point_table(config)
    for item in labels:
        key = item.strip().lower()
        if key in points:
            target = points[key]
            ik = int(np.argmin(np.linalg.norm(kfrac - target[None, :], axis=1)))
        else:
            ik = int(item)
        print(f"  k[{ik}] frac={kfrac[ik].tolist()} band={bands[ik].tolist()}")


def _print_bdg_debug(
    config,
    payload: JPayload,
    kfrac: np.ndarray,
    bands: np.ndarray,
    max_imag: np.ndarray,
    missing: np.ndarray,
    spin_pattern: np.ndarray,
    prefactor: float,
    bond_factor: float,
    anisotropy: float,
    hermitize: bool,
    bond_class: str,
) -> None:
    nmag = spin_pattern.size
    eta = np.diag(np.r_[np.ones(nmag), -np.ones(nmag)])
    labels = config.get_list("debug_points", ["0", "G", "M", "K"])
    points = point_table(config)
    print("[magnon_h5:debug] BdG diagnostics")
    print(f"  spin_pattern = {spin_pattern.tolist()}")
    print(f"  prefactor = {prefactor}, bond_factor = {bond_factor}, anisotropy_mev = {anisotropy}, hermitize = {hermitize}")
    print(f"  bond_class = {bond_class}")
    print(f"  unstable_or_incomplete_k = {int(np.count_nonzero(missing))}/{kfrac.shape[0]}")
    print(f"  max_abs_imag_etaH = {float(np.max(max_imag)):.8g}")
    for item in labels:
        key = item.strip().lower()
        if key in points:
            target = points[key]
            ik = int(np.argmin(np.linalg.norm(kfrac - target[None, :], axis=1)))
        else:
            ik = int(item)
        h = _bdg_matrix_at_k(kfrac[ik], payload, spin_pattern, prefactor, bond_factor, anisotropy, hermitize, bond_class)
        vals, vecs = np.linalg.eig(eta @ h)
        order = np.argsort(vals.real)
        vals = vals[order]
        vecs = vecs[:, order]
        norms = np.real(np.einsum("ij,ij->j", vecs.conj(), eta @ vecs))
        print(f"  k[{ik}] frac={kfrac[ik].tolist()} band={bands[ik].tolist()} max_imag={float(np.max(np.abs(vals.imag))):.8g}")
        for val, norm in zip(vals, norms):
            print(f"    eig = {val.real:.10g} {val.imag:+.3g}j, norm = {norm:.6g}")


def plot_magnon_h5(config):
    payload, structure_source = load_j_h5(config)
    kfrac, kdist, ticks, tick_labels = build_kpath(config, payload.lattice_ang)
    bands_mev = solve_magnon_bands(config, payload, kfrac)
    unit_label, unit_factor, ylabel, energy_key = _energy_scale(config)
    unit_suffix = energy_key.split("energy_", 1)[1]
    bands = bands_mev * unit_factor

    output = config.path_value("output", "magnon.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    plt = prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for ib in range(bands.shape[1]):
        ax.plot(kdist, bands[:, ib], color="black", lw=1.15)
    if config.get_bool("plot_imag", False) and "eigvals" in _LAST_FULL_BDG:
        imag = _LAST_FULL_BDG["eigvals"].imag * unit_factor
        for ib in range(imag.shape[1]):
            ax.plot(kdist, imag[:, ib], color="tab:red", lw=0.8, ls="--", alpha=0.75)
    for tick in ticks:
        ax.axvline(kdist[tick], color="0.78", lw=0.8)
    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_xticks([kdist[t] for t in ticks], tick_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Wave vector")
    ymin = config.get("ymin")
    ymax = config.get("ymax")
    if ymin is not None or ymax is not None:
        ax.set_ylim(None if ymin is None else float(ymin), None if ymax is None else float(ymax))
    fig.savefig(output, dpi=config.get_int("dpi", 200))
    plt.close(fig)

    save_npz = config.get("save_npz")
    if save_npz:
        npz_path = config.path_value("save_npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        payload_npz = {
            "kfrac": kfrac,
            "kdist": kdist,
            "energy_mev": bands_mev,
            energy_key: bands,
            "energy_unit": np.asarray(unit_label),
            "ticks": np.asarray(ticks, dtype=np.int64),
            "tick_labels": np.asarray(tick_labels, dtype=object),
        }
        if "eigvals" in _LAST_FULL_BDG:
            payload_npz["bdg_eigvals_mev"] = _LAST_FULL_BDG["eigvals"]
            payload_npz["bdg_real_mev"] = _LAST_FULL_BDG["eigvals"].real
            payload_npz["bdg_imag_mev"] = _LAST_FULL_BDG["eigvals"].imag
            payload_npz[f"bdg_eigvals_{unit_suffix}"] = _LAST_FULL_BDG["eigvals"] * unit_factor
            payload_npz[f"bdg_real_{unit_suffix}"] = _LAST_FULL_BDG["eigvals"].real * unit_factor
            payload_npz[f"bdg_imag_{unit_suffix}"] = _LAST_FULL_BDG["eigvals"].imag * unit_factor
            payload_npz["bdg_norms"] = _LAST_FULL_BDG["norms"]
        np.savez(npz_path, **payload_npz)

    print(f"kind = magnon_h5")
    print(f"input = {payload.path}")
    print(f"J source = {payload.j_source}")
    print(f"structure = {structure_source}")
    print(f"magnetic atoms = {', '.join(payload.atom_labels)}")
    finite = bands[np.isfinite(bands)]
    if finite.size:
        print(f"energy range = {np.min(finite):.8g} .. {np.max(finite):.8g} {unit_label}")
    else:
        print("energy range = no finite stable BdG modes")
    print(f"output = {output}")
    return bands
