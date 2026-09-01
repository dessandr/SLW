"""Native tensor LKAG kernel for qe2pert EPR inputs."""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import h5py
import numpy as np

from slw.exchange.kernels.epr import _build_hk_from_epr, _full_k_mesh, _load_slices
from slw.exchange.kernels.j_epr import (
    _atom_labels,
    _copy_epr_structure_basic,
    _find_nearest_neighbours_from_epr,
    _group_orbits_epr,
    _mirror_indices,
    _normalize_mag_atoms,
    _orbit_label_map,
    _split_chunks,
)
from slw.exchange.kernels.lkag import (
    _accumulate_tb2j_A_numba,
    get_cfr_ozaki_mesh,
    get_cfr_pole_mesh,
    get_semicircle_contour,
)
from slw.exchange.kernels.parallel import collective_sum, partition_sequence, rank_size
from slw.exchange.kernels.soc_downfold import (
    _downfold_one_k,
    _group_indices,
    _selected_groups,
    _soc_matrix_for_groups,
)
from slw.exchange.kernels.spinor import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    add_atomic_d_soc,
    add_atomic_p_soc,
    spinor_from_collinear,
)
from slw.soc.manifold import _clean_species, _win_projection_groups

_AXES = ("x", "y", "z")


def _decode_str_array(arr):
    out = []
    for x in np.asarray(arr).tolist():
        if isinstance(x, bytes):
            out.append(x.decode())
        else:
            out.append(str(x))
    return out


def _parse_axes(text):
    raw = str(text).replace(",", "").lower()
    out = []
    for ch in raw:
        if ch not in _AXES:
            raise ValueError(f"Unsupported tensor axis '{ch}'. Use subset of xyz.")
        if ch not in out:
            out.append(ch)
    return tuple(out) if out else _AXES


def _fmt_r(r):
    return f"({int(r[0])},{int(r[1])},{int(r[2])})"


def _ensure_out(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    return path


def _load_scalar_jr_h5(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Jr HDF5 not found: {path}")
    with h5py.File(path, "r") as h5:
        required = ["bonds/mag_i_atom", "bonds/mag_j_atom", "bonds/R", "J_r/value"]
        missing = [k for k in required if k not in h5]
        if missing:
            raise KeyError(f"Missing scalar Jr datasets in {path}: {missing}")

        gi = np.asarray(h5["bonds/mag_i_atom"], dtype=np.int64).reshape(-1)
        gj = np.asarray(h5["bonds/mag_j_atom"], dtype=np.int64).reshape(-1)
        R = np.asarray(h5["bonds/R"], dtype=np.int64).reshape(-1, 3)
        j_iso = np.asarray(h5["J_r/value"], dtype=np.float64).reshape(-1)
        if len(j_iso) != len(gi):
            raise ValueError(f"J_r/value length mismatch: {len(j_iso)} != {len(gi)}")

        dist = (
            np.asarray(h5["bonds/distance_ang"], dtype=np.float64).reshape(-1)
            if "bonds/distance_ang" in h5
            else np.full(len(gi), np.nan, dtype=np.float64)
        )
        shell = (
            np.asarray(h5["bonds/shell"], dtype=np.int64).reshape(-1)
            if "bonds/shell" in h5
            else np.zeros(len(gi), dtype=np.int64)
        )
        orbit = (
            _decode_str_array(h5["bonds/orbit_label"][:])
            if "bonds/orbit_label" in h5
            else ["NA"] * len(gi)
        )
        mirror = (
            np.asarray(h5["bonds/mirror_index"], dtype=np.int64).reshape(-1)
            if "bonds/mirror_index" in h5
            else np.full(len(gi), -1, dtype=np.int64)
        )
        atom_labels = (
            _decode_str_array(h5["basic_data/atom_labels"][:])
            if "basic_data/atom_labels" in h5
            else []
        )
        basic = {}
        for key in ("lattice_ang", "tau_frac", "tau_cart_ang", "kmesh", "nspin"):
            full = f"basic_data/{key}"
            if full in h5:
                basic[key] = np.asarray(h5[full])

    return {
        "gi": gi,
        "gj": gj,
        "R": R,
        "j_iso": j_iso,
        "dist": dist,
        "shell": shell,
        "orbit": orbit,
        "mirror": mirror,
        "atom_labels": atom_labels,
        "basic": basic,
    }


def _apply_bond_filter(payload, drop_onsite_self=False):
    if not drop_onsite_self:
        return payload
    mask = ~(
        (payload["gi"] == payload["gj"])
        & np.all(payload["R"] == np.array([0, 0, 0], dtype=np.int64)[None, :], axis=1)
    )
    out = {}
    for key, val in payload.items():
        if key in {"gi", "gj", "R", "j_iso", "dist", "shell", "orbit", "mirror"}:
            arr = np.asarray(val)
            out[key] = arr[mask].tolist() if key == "orbit" else arr[mask]
        else:
            out[key] = val
    return out


def _scalar_to_tensor(j_iso, tensor_axes):
    n_axis = len(tensor_axes)
    out = np.zeros((len(j_iso), n_axis, n_axis), dtype=np.float64)
    idx = np.arange(n_axis)
    out[:, idx, idx] = np.asarray(j_iso, dtype=np.float64)[:, None]
    return out


def _tensor_decomp(mat):
    j_iso = float(np.trace(mat) / 3.0) if mat.shape == (3, 3) else float(np.trace(mat) / mat.shape[0])
    sym = 0.5 * (mat + mat.T)
    ani = sym - np.eye(mat.shape[0]) * (np.trace(sym) / mat.shape[0])
    dmi = np.zeros(3, dtype=np.float64)
    if mat.shape == (3, 3):
        dmi[:] = [0.5 * (mat[1, 2] - mat[2, 1]), 0.5 * (mat[2, 0] - mat[0, 2]), 0.5 * (mat[0, 1] - mat[1, 0])]
    return j_iso, ani, dmi


def _write_h5(path, payload, tensor, tensor_axes, source):
    _ensure_out(path)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        def put_str(group, name, value):
            group.create_dataset(name, data=np.array(str(value), dtype=object), dtype=str_dt)

        basic = h5.create_group("basic_data")
        put_str(basic, "unit", "meV")
        put_str(basic, "source", source)
        put_str(basic, "tensor_mode", "isotropic_from_scalar_collinear")
        put_str(
            basic,
            "hamiltonian_convention",
            "E=-1/2 sum_directed J_ij e_i.e_j; mate-complete pair list",
        )
        put_str(basic, "hamiltonian_sign", "minus")
        put_str(basic, "spin_normalization", "unit_vector")
        put_str(basic, "kernel_family", "scalar_lkag")
        put_str(basic, "bond_coverage", "directed_mate_complete")
        put_str(basic, "realspace_gauge", "i_at_0_j_at_R")
        basic.create_dataset("source_spin_magnitude", data=np.array(1.0, dtype=np.float64))
        basic.create_dataset("directed_bond_weight", data=np.array(0.5, dtype=np.float64))
        put_str(basic, "command", " ".join(sys.argv))
        basic.create_dataset("tensor_axes", data=np.asarray(tensor_axes, dtype=object), dtype=str_dt)
        if payload["atom_labels"]:
            basic.create_dataset("atom_labels", data=np.asarray(payload["atom_labels"], dtype=object), dtype=str_dt)
        for key, val in payload["basic"].items():
            basic.create_dataset(key, data=val)

        bonds = h5.create_group("bonds")
        bonds.create_dataset("mag_i_atom", data=payload["gi"])
        bonds.create_dataset("mag_j_atom", data=payload["gj"])
        bonds.create_dataset("R", data=payload["R"])
        bonds.create_dataset("distance_ang", data=payload["dist"])
        bonds.create_dataset("shell", data=payload["shell"])
        bonds.create_dataset("orbit_label", data=np.asarray(payload["orbit"], dtype=object), dtype=str_dt)
        bonds.create_dataset("mirror_index", data=payload["mirror"])

        h5.create_dataset("J_iso_r", data=payload["j_iso"])
        dset = h5.create_dataset("J_tensor_r", data=tensor)
        dset.attrs["shape"] = "(nBond,tensor_axis_a,tensor_axis_b)"
        dset.attrs["tensor_axis_order"] = ",".join(tensor_axes)
        dset.attrs["unit"] = "meV"


def _label_for(labels, idx):
    return str(labels[idx]) if 0 <= int(idx) < len(labels) else f"Atom{int(idx) + 1}"


def _atom_label(payload, idx):
    return _label_for(payload["atom_labels"], idx)


def _write_text(path, payload, tensor, tensor_axes):
    _ensure_out(path)
    with open(path, "w") as f:
        f.write("# J tensor Results - EPR scalar J lifted to isotropic tensor\n")
        f.write("# tensor_mode = isotropic_from_scalar_collinear\n")
        f.write("# convention = E=-1/2 sum_directed J_ij e_i.e_j; mate-complete pair list\n")
        f.write("# unit = meV\n\n")
        for ib in range(len(payload["gi"])):
            gi = int(payload["gi"][ib])
            gj = int(payload["gj"][ib])
            ilab = _atom_label(payload, gi)
            jlab = _atom_label(payload, gj)
            f.write(
                f"Orbit: {payload['orbit'][ib]}   pair i={gi}({ilab}, atom #{gi + 1}) "
                f"j={gj}({jlab}, atom #{gj + 1})   R = {_fmt_r(payload['R'][ib])} "
                f"dist = {float(payload['dist'][ib]):.6f} A\n"
            )
            f.write(f"J_iso = {float(payload['j_iso'][ib]):+.10e} meV\n")
            f.write("J_tensor:\n")
            for ia, _a in enumerate(tensor_axes):
                row = [f"{tensor[ib, ia, ibb]:+.10e}" for ibb in range(len(tensor_axes))]
                f.write("  " + " ".join(row) + "\n")
            j_iso, ani, dmi = _tensor_decomp(tensor[ib])
            f.write(f"decomp_iso = {j_iso:+.10e} meV\n")
            if ani.shape == (3, 3):
                f.write("anisotropy_symmetric_traceless:\n")
                for ia in range(3):
                    f.write("  " + " ".join(f"{ani[ia, ibb]:+.10e}" for ibb in range(3)) + "\n")
                f.write(f"DMI = ({dmi[0]:+.10e}, {dmi[1]:+.10e}, {dmi[2]:+.10e}) meV\n")
            f.write("\n")


def _write_tsv(path, payload, tensor, tensor_axes):
    tsv = os.path.splitext(path)[0] + ".all_bonds.tsv"
    _ensure_out(tsv)
    with open(tsv, "w") as f:
        f.write(
            "bond_index\torbit\tgi\tgj\ti_atom\tj_atom\tR1\tR2\tR3\tdist_A\tshell\t"
            "tensor_a\ttensor_b\tJ_tensor\tJ_iso\n"
        )
        for ib in range(len(payload["gi"])):
            for ia, a in enumerate(tensor_axes):
                f.writelines(f"{ib}\t{payload['orbit'][ib]}\t{int(payload['gi'][ib])}\t{int(payload['gj'][ib])}\t"
                        f"{int(payload['gi'][ib]) + 1}\t{int(payload['gj'][ib]) + 1}\t"
                        f"{int(payload['R'][ib,0])}\t{int(payload['R'][ib,1])}\t{int(payload['R'][ib,2])}\t"
                        f"{float(payload['dist'][ib]):.12e}\t{int(payload['shell'][ib])}\t"
                        f"{a}\t{b}\t{tensor[ib, ia, ibb]:.12e}\t{float(payload['j_iso'][ib]):.12e}\n" for ibb, b in enumerate(tensor_axes))
    return tsv


_WORKER_STATIC = None


def _infer_win_path(epr_path, explicit=None):
    if explicit:
        return explicit
    base = os.path.dirname(os.path.abspath(epr_path)) or "."
    stem = os.path.basename(epr_path)
    candidates = []
    for suffix in ("_epr.h5", ".h5"):
        if stem.endswith(suffix):
            candidates.append(os.path.join(base, stem[: -len(suffix)] + ".win"))
    candidates.extend(sorted(glob.glob(os.path.join(base, "*.win"))))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _resolve_soc_entries(args, nwan):
    entries = []
    card_specs = tuple(getattr(args, "soc_manifolds", ()) or ())
    if not card_specs:
        return entries, None
    win_path = getattr(args, "win", None)
    if not win_path:
        raise ValueError("SOC (atomic) requires an explicit Wannier90 .win file")
    _atoms, all_groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(
            f"{win_path} projection count={nproj} but Hamiltonian nwan={nwan}"
        )
    used: set[int] = set()
    for spec in card_specs:
        selector = str(spec["selector"])
        label, orb = selector.rsplit("-", 1)
        exact = [
            group
            for group in all_groups
            if str(group["atom_label"]).casefold() == label.casefold()
            and str(group["orbital"]).lower() == orb
        ]
        matching = exact or [
            group
            for group in all_groups
            if str(group["element"]).casefold()
            == _clean_species(label).casefold()
            and str(group["orbital"]).lower() == orb
        ]
        if not matching:
            known = sorted(
                {
                    f"{group['atom_label']}-{group['orbital']}"
                    for group in all_groups
                }
            )
            raise ValueError(
                f"SOC selector {selector!r} matched no Wannier manifold; "
                f"known site manifolds={known}"
            )
        indices = {
            int(index) for group in matching for index in group["indices"]
        }
        overlap = sorted(used & indices)
        if overlap:
            raise ValueError(
                f"SOC selector {selector!r} overlaps a previous selector at "
                f"Wannier indices {overlap}"
            )
        used.update(indices)
        entries.append(
            {
                "selector": selector,
                "element": _clean_species(label),
                "orbital": orb,
                "lambda_ev": float(spec["lambda_ev"]),
                "groups": [group["indices"] for group in matching],
                "matched_labels": [group["atom_label"] for group in matching],
                "source": "SOC (atomic)",
            }
        )
    return entries, win_path


def _apply_model_soc(h_spin, args, nwan):
    entries, win_path = _resolve_soc_entries(args, nwan)
    out = h_spin
    for entry in entries:
        orb = entry["orbital"]
        lam = float(entry["lambda_ev"])
        groups = entry["groups"]
        if orb == "p":
            out = add_atomic_p_soc(
                out,
                groups,
                lambda_ev=lam,
                order=WANNIER90_P_ORDER,
                inplace=False,
            )
        elif orb == "d":
            out = add_atomic_d_soc(
                out,
                groups,
                lambda_ev=lam,
                order=WANNIER90_D_ORDER,
                inplace=False,
            )
        else:
            raise ValueError(f"Unsupported SOC orbital: {orb}")
        print(
            f"[J-epr-tensor] added model {entry['element']}:{orb} SOC "
            f"lambda={lam:g} eV groups={groups} source={entry['source']}",
            flush=True,
        )
    return out, entries, win_path


def _selector_indices(selector, nwan, *, site=None):
    """Return one validated collinear-half orbital selector as an index array."""
    if int(nwan) <= 0:
        raise ValueError(f"Collinear-half Hamiltonian dimension must be positive, got {nwan}")
    context = "" if site is None else f" for magnetic site {site}"
    if isinstance(selector, slice):
        if selector.start is None or selector.stop is None:
            raise ValueError(f"Orbital slice{context} requires explicit start and stop")
        if selector.step not in (None, 1):
            raise ValueError(f"Orbital slice{context} must have unit stride")
        start = int(selector.start)
        stop = int(selector.stop)
        if start < 0 or stop <= start or stop > int(nwan):
            raise ValueError(
                f"Invalid orbital slice{context}: [{start}:{stop}] for nwan={nwan}"
            )
        return np.arange(start, stop, dtype=np.int64)

    raw = np.asarray(selector)
    if raw.ndim != 1:
        raise ValueError(
            f"Orbital index selector{context} must be one-dimensional, got shape={raw.shape}"
        )
    if raw.size == 0:
        raise ValueError(f"Orbital index selector{context} must not be empty")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.integer
    ):
        raise ValueError(f"Orbital index selector{context} must contain integers")
    indices = np.asarray(raw, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= int(nwan)):
        raise ValueError(
            f"Orbital index selector{context}={indices.tolist()} is outside [0,{nwan})"
        )
    if np.unique(indices).size != indices.size:
        raise ValueError(
            f"Orbital index selector{context} contains duplicate indices: {indices.tolist()}"
        )
    return indices


def _validated_spinor_hamiltonian(h_spin):
    array = np.asarray(h_spin, dtype=np.complex128)
    if (
        array.ndim != 3
        or array.shape[0] == 0
        or array.shape[1] == 0
        or array.shape[1] != array.shape[2]
        or array.shape[1] % 2
    ):
        raise ValueError(
            f"Spinor Hamiltonian must have shape (nk,2*nwan,2*nwan), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Spinor Hamiltonian contains non-finite matrix elements")
    return array, int(array.shape[1] // 2)


def _normalise_site_selectors(selectors, nwan):
    if not selectors:
        raise ValueError("Magnetic orbital selector mapping must not be empty")
    out = {}
    orbital_owner = np.full(int(nwan), -1, dtype=np.int64)
    for site, selector in selectors.items():
        if isinstance(site, (bool, np.bool_)) or not isinstance(site, (int, np.integer)):
            raise TypeError(f"Magnetic site selector key must be an integer, got {site!r}")
        site_index = int(site)
        if site_index < 0:
            raise ValueError(f"Magnetic site selector key must be non-negative, got {site_index}")
        if site_index in out:
            raise ValueError(f"Duplicate magnetic site selector after integer conversion: {site!r}")
        indices = _selector_indices(selector, nwan, site=site_index)
        overlap = indices[orbital_owner[indices] >= 0]
        if overlap.size:
            owners = orbital_owner[overlap].tolist()
            raise ValueError(
                f"Magnetic site {site_index} overlaps existing site selectors at "
                f"orbitals={overlap.tolist()} owned_by={owners}"
            )
        orbital_owner[indices] = site_index
        out[site_index] = indices
    return out


def _spinor_atom_indices(selector, nwan):
    up = _selector_indices(selector, nwan)
    return np.concatenate([up, up + int(nwan)])


def _selectors_in_dynamic_basis(selectors, nwan, d_idx):
    selectors = _normalise_site_selectors(selectors, nwan)
    d_indices = _selector_indices(d_idx, nwan, site="dynamic d subspace")
    global_to_local = np.full(int(nwan), -1, dtype=np.int64)
    global_to_local[d_indices] = np.arange(d_indices.size, dtype=np.int64)
    out = {}
    for site, indices in selectors.items():
        local = global_to_local[indices]
        if np.any(local < 0):
            missing = indices[local < 0].tolist()
            raise ValueError(
                f"Magnetic site {site} orbitals {missing} are absent from the dynamic d subspace"
            )
        out[site] = np.concatenate([local, local + d_indices.size])
    return out


def _validate_pair_selector_sites(pair_meta, selectors):
    used = {
        int(meta[key])
        for meta in pair_meta
        for key in ("li", "lj")
    }
    missing = sorted(used - set(selectors))
    if missing:
        raise ValueError(
            f"Exchange pairs reference magnetic sites without orbital selectors: {missing}"
        )


def _extract_exchange_fields_from_spinor_block(h_loc):
    n2 = int(h_loc.shape[0])
    if n2 % 2:
        raise ValueError(f"Local spinor block dimension must be even, got {h_loc.shape}")
    n = n2 // 2
    huu = h_loc[:n, :n]
    hud = h_loc[:n, n:]
    hdu = h_loc[n:, :n]
    hdd = h_loc[n:, n:]
    bx = 0.5 * (hud + hdu)
    by = (hdu - hud) / (2.0j)
    bz = 0.5 * (huu - hdd)
    return bx, by, bz


def _compose_spinor_operator(axis, bmat):
    n = int(bmat.shape[0])
    z = np.zeros((n, n), dtype=np.complex128)
    if axis == "x":
        return np.block([[z, bmat], [bmat, z]])
    if axis == "y":
        return np.block([[z, -1j * bmat], [1j * bmat, z]])
    if axis == "z":
        return np.block([[bmat, z], [z, -bmat]])
    raise ValueError(f"Unsupported tensor axis: {axis}")


def _pauli_block_all_contiguous(block):
    n2 = int(block.shape[0])
    if n2 % 2:
        raise ValueError(f"Spinor block must have even shape, got {block.shape}")
    n = n2 // 2
    guu = block[:n, :n]
    gud = block[:n, n:]
    gdu = block[n:, :n]
    gdd = block[n:, n:]
    g0 = 0.5 * (guu + gdd)
    gx = 0.5 * (gud + gdu)
    gy = 0.5j * (gud - gdu)
    gz = 0.5 * (guu - gdd)
    return g0, gx, gy, gz


def _build_tb2j_projectors(h_spin_mean, slices):
    nwan = int(h_spin_mean.shape[0] // 2)
    selectors = _normalise_site_selectors(slices, nwan)
    out = {}
    for site, selector in selectors.items():
        idx = _spinor_atom_indices(selector, nwan)
        hloc = h_spin_mean[np.ix_(idx, idx)]
        _m0, mx, my, mz = _pauli_block_all_contiguous(hloc)
        evec = np.array([np.trace(mx), np.trace(my), np.trace(mz)], dtype=np.complex128)
        norm = float(np.linalg.norm(evec))
        if norm < 1.0e-14:
            out[int(site)] = np.zeros_like(mx)
        else:
            evec = evec / norm
            out[int(site)] = mx * evec[0] + my * evec[1] + mz * evec[2]
    return out


def _precompute_spinor_kdata(h_spin, slices, efermi):
    h_spin, nwan = _validated_spinor_hamiltonian(h_spin)
    nk, dim, _ = h_spin.shape
    selectors = _normalise_site_selectors(slices, nwan)
    eye = np.eye(dim, dtype=np.complex128)
    evals = np.zeros((nk, dim), dtype=np.float64)
    evecs = np.zeros((nk, dim, dim), dtype=np.complex128)
    coeffs = {
        int(site): np.zeros((nk, 2 * selector.size, dim), dtype=np.complex128)
        for site, selector in selectors.items()
    }
    for ik in range(nk):
        hk = 0.5 * (h_spin[ik] + h_spin[ik].conj().T) - float(efermi) * eye
        ww, cc = np.linalg.eigh(hk)
        evals[ik] = np.real(ww)
        evecs[ik] = cc
        for site, selector in selectors.items():
            idx = _spinor_atom_indices(selector, nwan)
            coeffs[int(site)][ik] = cc[idx, :]

    h0 = np.mean(h_spin, axis=0)
    d_ops = {}
    for site, selector in selectors.items():
        idx = _spinor_atom_indices(selector, nwan)
        hloc = h0[np.ix_(idx, idx)]
        bx, by, bz = _extract_exchange_fields_from_spinor_block(hloc)
        d_ops[int(site)] = {
            "x": _compose_spinor_operator("x", bx),
            "y": _compose_spinor_operator("y", by),
            "z": _compose_spinor_operator("z", bz),
        }
    return {
        "evals": evals,
        "evecs": evecs,
        "coeffs": coeffs,
        "d_ops": d_ops,
        "efermi": efermi,
        "selectors": selectors,
    }


def _downfold_one_k_dynamic(hup, hdn, d_idx, p_idx, lambda_uu, lambda_ud, lambda_du, lambda_dd, z):
    nd = int(d_idx.size)
    eye_p = np.eye(int(p_idx.size), dtype=np.complex128)

    gp_up = np.linalg.inv(z * eye_p - hup[np.ix_(p_idx, p_idx)])
    gp_dn = np.linalg.inv(z * eye_p - hdn[np.ix_(p_idx, p_idx)])

    h_up_dp = hup[np.ix_(d_idx, p_idx)]
    h_dn_dp = hdn[np.ix_(d_idx, p_idx)]
    h_up_pd = hup[np.ix_(p_idx, d_idx)]
    h_dn_pd = hdn[np.ix_(p_idx, d_idx)]

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


def _build_dynamic_h_eff(hup, hdn, d_idx, p_idx, lambda_uu, lambda_ud, lambda_du, lambda_dd, z):
    nk = hup.shape[0]
    nd = d_idx.size
    h_eff = np.zeros((nk, 2 * nd, 2 * nd), dtype=np.complex128)
    for ik in range(nk):
        h_eff[ik, :nd, :nd] = hup[ik][np.ix_(d_idx, d_idx)]
        h_eff[ik, nd:, nd:] = hdn[ik][np.ix_(d_idx, d_idx)]
        corr = _downfold_one_k_dynamic(
            hup[ik], hdn[ik], d_idx, p_idx,
            lambda_uu, lambda_ud, lambda_du, lambda_dd, z
        )
        h_eff[ik] += corr
    return h_eff


def _compute_tensor_chunk(energy_mesh, kdata, pair_meta, phase, kweights, axes):
    n_pair = len(pair_meta)
    n_ax = len(axes)
    acc = np.zeros((n_pair, n_ax, n_ax), dtype=np.complex128)
    trace_acc = np.zeros_like(acc)
    pair_ctx = []
    for meta in pair_meta:
        li = int(meta["li"])
        lj = int(meta["lj"])
        pair_ctx.append(
            (
                kdata["coeffs"][li],
                kdata["coeffs"][lj],
                np.stack([kdata["d_ops"][li][ax] for ax in axes], axis=0),
                np.stack([kdata["d_ops"][lj][ax] for ax in axes], axis=0),
            )
        )

    for z, dz in energy_mesh:
        inv = 1.0 / (z - kdata["evals"])
        for ip, (ci, cj, di, dj) in enumerate(pair_ctx):
            gij_k = np.einsum("kia,ka,kja->kij", ci, inv, np.conjugate(cj), optimize=True)
            gji_k = np.einsum("kja,ka,kia->kji", cj, inv, np.conjugate(ci), optimize=True)
            wk_phase = kweights * phase[ip]
            gij = np.einsum("k,kij->ij", wk_phase, gij_k, optimize=True)
            gji = np.einsum("k,kji->ji", np.conjugate(wk_phase), gji_k, optimize=True)
            left = np.einsum("aij,jk->aik", di, gij, optimize=True)
            right = np.einsum("bij,jk->bik", dj, gji, optimize=True)
            tr = np.einsum("aij,bji->ab", left, right, optimize=True)
            trace_acc[ip] += tr
            acc[ip] += tr * dz
    return acc, trace_acc


def _worker_init(static_payload):
    global _WORKER_STATIC
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _WORKER_STATIC = static_payload


def _compute_tensor_chunk(
    energy_mesh, kdata, pair_meta, phase, kweights, axes,
    dynamic_soc=False,
    hk_up=None,
    hk_dn=None,
    d_idx=None,
    p_idx=None,
    lambda_uu=None,
    lambda_ud=None,
    lambda_du=None,
    lambda_dd=None,
    slices=None,
):
    n_pair = len(pair_meta)
    n_ax = len(axes)
    acc = np.zeros((n_pair, n_ax, n_ax), dtype=np.complex128)
    trace_acc = np.zeros_like(acc)
    pair_ctx = []

    if dynamic_soc:
        nd = int(d_idx.size)
        local_idxs = _selectors_in_dynamic_basis(
            slices, int(hk_up.shape[1]), d_idx
        )

        for meta in pair_meta:
            li = int(meta["li"])
            lj = int(meta["lj"])
            pair_ctx.append(
                (
                    li,
                    lj,
                    np.stack([kdata["d_ops"][li][ax] for ax in axes], axis=0),
                    np.stack([kdata["d_ops"][lj][ax] for ax in axes], axis=0),
                )
            )

        nk = hk_up.shape[0]
        for z, dz in energy_mesh:
            z_abs = z + float(kdata["efermi"])
            h_eff_z = _build_dynamic_h_eff(hk_up, hk_dn, d_idx, p_idx, lambda_uu, lambda_ud, lambda_du, lambda_dd, z_abs)
            evals_z = np.zeros((nk, 2 * nd), dtype=np.float64)
            evecs_z = np.zeros((nk, 2 * nd, 2 * nd), dtype=np.complex128)
            for ik in range(nk):
                ww, cc = np.linalg.eigh(h_eff_z[ik] - float(kdata["efermi"]) * np.eye(2 * nd))
                evals_z[ik] = ww
                evecs_z[ik] = cc

            inv = 1.0 / (z - evals_z)
            for ip, (li, lj, di, dj) in enumerate(pair_ctx):
                idx_i = local_idxs[li]
                idx_j = local_idxs[lj]
                ci = evecs_z[:, idx_i, :]
                cj = evecs_z[:, idx_j, :]

                gij_k = np.einsum("kia,ka,kja->kij", ci, inv, np.conjugate(cj), optimize=True)
                gji_k = np.einsum("kja,ka,kia->kji", cj, inv, np.conjugate(ci), optimize=True)
                wk_phase = kweights * phase[ip]
                gij = np.einsum("k,kij->ij", wk_phase, gij_k, optimize=True)
                gji = np.einsum("k,kji->ji", np.conjugate(wk_phase), gji_k, optimize=True)
                left = np.einsum("aij,jk->aik", di, gij, optimize=True)
                right = np.einsum("bij,jk->bik", dj, gji, optimize=True)
                tr = np.einsum("aij,bji->ab", left, right, optimize=True)
                trace_acc[ip] += tr
                acc[ip] += tr * dz
    else:
        for meta in pair_meta:
            li = int(meta["li"])
            lj = int(meta["lj"])
            pair_ctx.append(
                (
                    kdata["coeffs"][li],
                    kdata["coeffs"][lj],
                    np.stack([kdata["d_ops"][li][ax] for ax in axes], axis=0),
                    np.stack([kdata["d_ops"][lj][ax] for ax in axes], axis=0),
                )
            )
        for z, dz in energy_mesh:
            inv = 1.0 / (z - kdata["evals"])
            for ip, (ci, cj, di, dj) in enumerate(pair_ctx):
                gij_k = np.einsum("kia,ka,kja->kij", ci, inv, np.conjugate(cj), optimize=True)
                gji_k = np.einsum("kja,ka,kia->kji", cj, inv, np.conjugate(ci), optimize=True)
                wk_phase = kweights * phase[ip]
                gij = np.einsum("k,kij->ij", wk_phase, gij_k, optimize=True)
                gji = np.einsum("k,kji->ji", np.conjugate(wk_phase), gji_k, optimize=True)
                left = np.einsum("aij,jk->aik", di, gij, optimize=True)
                right = np.einsum("bij,jk->bik", dj, gji, optimize=True)
                tr = np.einsum("aij,bji->ab", left, right, optimize=True)
                trace_acc[ip] += tr
                acc[ip] += tr * dz
    return acc, trace_acc


def _worker_init(static_payload):
    global _WORKER_STATIC
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _WORKER_STATIC = static_payload


def _worker_compute_tensor_chunk(energy_chunk):
    if _WORKER_STATIC is None:
        raise RuntimeError("Worker static payload is not initialized.")
    return _compute_tensor_chunk(
        energy_chunk,
        _WORKER_STATIC["kdata"],
        _WORKER_STATIC["pair_meta"],
        _WORKER_STATIC["phase"],
        _WORKER_STATIC["kweights"],
        _WORKER_STATIC["axes"],
        dynamic_soc=_WORKER_STATIC.get("dynamic_soc", False),
        hk_up=_WORKER_STATIC.get("hk_up"),
        hk_dn=_WORKER_STATIC.get("hk_dn"),
        d_idx=_WORKER_STATIC.get("d_idx"),
        p_idx=_WORKER_STATIC.get("p_idx"),
        lambda_uu=_WORKER_STATIC.get("lambda_uu"),
        lambda_ud=_WORKER_STATIC.get("lambda_ud"),
        lambda_du=_WORKER_STATIC.get("lambda_du"),
        lambda_dd=_WORKER_STATIC.get("lambda_dd"),
        slices=_WORKER_STATIC.get("slices"),
    )


def _compute_tensor_direct(
    h_spin, slices, pair_meta, kpts, energy_mesh, efermi, axes, *, nproc=1, collinear_override=False,
    dynamic_soc=False,
    hk_up=None,
    hk_dn=None,
    d_idx=None,
    p_idx=None,
    lambda_uu=None,
    lambda_ud=None,
    lambda_du=None,
    lambda_dd=None,
):
    h_spin, static_nwan = _validated_spinor_hamiltonian(h_spin)
    nwan = int(hk_up.shape[1]) if dynamic_soc else static_nwan
    slices = _normalise_site_selectors(slices, nwan)
    _validate_pair_selector_sites(pair_meta, slices)
    if dynamic_soc:
        delta_k = np.empty((hk_up.shape[0], 2 * d_idx.size, 2 * d_idx.size), dtype=np.complex128)
        e0_ref = float(efermi)
        eta_ref = 0.01
        for ik in range(hk_up.shape[0]):
            delta_k[ik] = _downfold_one_k(
                hk_up[ik], hk_dn[ik], d_idx, p_idx,
                lambda_uu, lambda_ud, lambda_du, lambda_dd,
                e0_ref, eta_ref
            )
        nwan = hk_up.shape[1]
        h_spin_static = np.array(h_spin, dtype=np.complex128, copy=True)
        nd = d_idx.size
        up = np.asarray(d_idx, dtype=np.int64)
        dn = up + int(nwan)
        h_spin_static[:, up[:, None], up] += delta_k[:, :nd, :nd]
        h_spin_static[:, up[:, None], dn] += delta_k[:, :nd, nd:]
        h_spin_static[:, dn[:, None], up] += delta_k[:, nd:, :nd]
        h_spin_static[:, dn[:, None], dn] += delta_k[:, nd:, nd:]
        kdata = _precompute_spinor_kdata(h_spin_static, slices, efermi)
    else:
        kdata = _precompute_spinor_kdata(h_spin, slices, efermi)


    nk = len(kpts)
    kweights = np.full(nk, 1.0 / float(max(1, nk)), dtype=np.float64)
    r_arr = np.asarray([m["R"] for m in pair_meta], dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * (r_arr @ kpts.T))

    nproc = max(1, int(nproc))
    if nproc <= 1 or len(energy_mesh) < 2:
        acc, trace_acc = _compute_tensor_chunk(
            energy_mesh, kdata, pair_meta, phase, kweights, axes,
            dynamic_soc=dynamic_soc,
            hk_up=hk_up,
            hk_dn=hk_dn,
            d_idx=d_idx,
            p_idx=p_idx,
            lambda_uu=lambda_uu,
            lambda_ud=lambda_ud,
            lambda_du=lambda_du,
            lambda_dd=lambda_dd,
            slices=slices,
        )
        tensor_raw = 1000.0 * np.imag(acc) / (4.0 * np.pi)
    else:
        chunks = _split_chunks(energy_mesh, nproc)
        static_payload = {
            "kdata": kdata,
            "pair_meta": pair_meta,
            "phase": phase,
            "kweights": kweights,
            "axes": tuple(axes),
            "dynamic_soc": dynamic_soc,
            "hk_up": hk_up,
            "hk_dn": hk_dn,
            "d_idx": d_idx,
            "p_idx": p_idx,
            "lambda_uu": lambda_uu,
            "lambda_ud": lambda_ud,
            "lambda_du": lambda_du,
            "lambda_dd": lambda_dd,
            "slices": slices,
        }
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ctx.Pool(processes=len(chunks), initializer=_worker_init, initargs=(static_payload,)) as pool:
            partials = pool.map(_worker_compute_tensor_chunk, chunks)

        acc = np.zeros((len(pair_meta), len(axes), len(axes)), dtype=np.complex128)
        trace_acc = np.zeros_like(acc)
        for a_part, tr_part in partials:
            acc += a_part
            trace_acc += tr_part
        tensor_raw = 1000.0 * np.imag(acc) / (4.0 * np.pi)

    if collinear_override and "x" in axes and "y" in axes and "z" in axes:
        ix = axes.index("x")
        iy = axes.index("y")
        iz = axes.index("z")
        tensor_raw[:, iz, iz] = 0.5 * (tensor_raw[:, ix, ix] + tensor_raw[:, iy, iy])

    n_chunks_val = len(chunks) if (nproc > 1 and len(energy_mesh) >= 2) else 1
    return tensor_raw, trace_acc, {"n_chunks": n_chunks_val, "nproc": n_chunks_val}


def _ordered_unique(seq):
    out = []
    seen = set()
    for item in seq:
        key = tuple(item)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _decompose_tb2j_pair(
    val,
    val_m,
    *,
    collinear_override=False,
    collinear_direction=None,
):
    """Decompose TB2J A tensors into scalar J, Gamma, DMI, and debug Aab tensor."""
    val = np.asarray(val, dtype=np.complex128)
    val_m = np.asarray(val_m, dtype=np.complex128)

    m_raw = np.imag(val[1:4, 1:4] + val_m[1:4, 1:4]).astype(np.float64, copy=False)
    ms = 0.5 * (m_raw + m_raw.T)
    if collinear_override:
        ms = np.array(ms, dtype=np.float64, copy=True)
        if collinear_direction is None:
            direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            direction = np.asarray(collinear_direction, dtype=np.float64)
            if direction.shape != (3,) or not np.all(np.isfinite(direction)):
                raise ValueError("collinear_direction must be a finite 3-vector")
            norm = float(np.linalg.norm(direction))
            if norm <= 0.0:
                raise ValueError("collinear_direction must be nonzero")
            direction = direction / norm
        longitudinal = float(direction @ ms @ direction)
        transverse = 0.5 * (float(np.trace(ms)) - longitudinal)
        ms += (transverse - longitudinal) * np.outer(direction, direction)

    jiso_tb2j = float(np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3]))
    gamma = ms - np.eye(3, dtype=np.float64) * (float(np.trace(ms)) / 3.0)
    ma = 0.5 * (m_raw - m_raw.T)

    dmi_tb2j = np.array(
        [
            np.real(val[0, 1] - val[1, 0]),
            np.real(val[0, 2] - val[2, 0]),
            np.real(val[0, 3] - val[3, 0]),
        ],
        dtype=np.float64,
    )
    dmi_mat = np.array(
        [
            [0.0, dmi_tb2j[2], -dmi_tb2j[1]],
            [-dmi_tb2j[2], 0.0, dmi_tb2j[0]],
            [dmi_tb2j[1], -dmi_tb2j[0], 0.0],
        ],
        dtype=np.float64,
    )
    eye = np.eye(3, dtype=np.float64)
    return {
        "jiso": jiso_tb2j,
        "gamma": gamma,
        "m_raw": m_raw,
        "ma": ma,
        "dmi": dmi_tb2j,
        "dmi_mat": dmi_mat,
        "jfull_aab": gamma + eye * jiso_tb2j + ma,
        "jfull_dmi": gamma + eye * jiso_tb2j + dmi_mat,
    }


def _green_mesh_from_eigensystem(evals, evecs, z):
    inv = 1.0 / (z - evals)
    return np.einsum("kia,ka,kja->kij", evecs, inv, np.conjugate(evecs), optimize=True)


def _sum_gr_flat(gk, r_list, kpts):
    r_arr = np.asarray(r_list, dtype=np.float64).reshape(-1, 3)
    phase = np.exp(-1j * 2.0 * np.pi * (r_arr @ kpts.T))
    nk = float(max(1, len(kpts)))
    return np.einsum("rk,kij->rij", phase, gk, optimize=True) / nk


def _compute_tensor_tb2j(
    h_spin, slices, pair_meta, kpts, energy_mesh, efermi, axes,
    collinear_override=False,
    dynamic_soc=False,
    hk_up=None,
    hk_dn=None,
    d_idx=None,
    p_idx=None,
    lambda_uu=None,
    lambda_ud=None,
    lambda_du=None,
    lambda_dd=None,
):
    h_spin, static_nwan = _validated_spinor_hamiltonian(h_spin)
    if dynamic_soc:
        # collinear_override must be enabled for dynamic collinear downfolding base
        nwan = int(hk_up.shape[1])
    else:
        nwan = static_nwan
    slices = _normalise_site_selectors(slices, nwan)
    _validate_pair_selector_sites(pair_meta, slices)

    req_pairs = _ordered_unique((int(m["li"]), int(m["lj"])) for m in pair_meta)
    acc_pairs = list(req_pairs)
    pair_set = set(acc_pairs)
    for i, j in req_pairs:
        if (j, i) not in pair_set:
            acc_pairs.append((j, i))
            pair_set.add((j, i))

    req_r = _ordered_unique(tuple(int(x) for x in m["R"]) for m in pair_meta)
    acc_r = list(req_r)
    r_set = set(acc_r)
    for r in req_r:
        rm = (-r[0], -r[1], -r[2])
        if rm not in r_set:
            acc_r.append(rm)
            r_set.add(rm)

    n_pairs = len(acc_pairs)
    n_r = len(acc_r)
    ni_arr = np.zeros(n_pairs, dtype=np.int64)
    nj_arr = np.zeros(n_pairs, dtype=np.int64)
    max_ni = 0
    max_nj = 0
    for ip, (li, lj) in enumerate(acc_pairs):
        ni = slices[li].size
        nj = slices[lj].size
        ni_arr[ip] = ni
        nj_arr[ip] = nj
        max_ni = max(max_ni, ni)
        max_nj = max(max_nj, nj)

    h_spin_mean = np.mean(h_spin, axis=0)

    projectors = _build_tb2j_projectors(h_spin_mean, slices)
    p_i = np.zeros((n_pairs, max_ni, max_ni), dtype=np.complex128)
    p_j = np.zeros((n_pairs, max_nj, max_nj), dtype=np.complex128)

    # For standard calculation
    iu_up = np.zeros((n_pairs, max_ni), dtype=np.int64)
    iu_dn = np.zeros((n_pairs, max_ni), dtype=np.int64)
    jv_up = np.zeros((n_pairs, max_nj), dtype=np.int64)
    jv_dn = np.zeros((n_pairs, max_nj), dtype=np.int64)

    for ip, (li, lj) in enumerate(acc_pairs):
        si = slices[li]
        sj = slices[lj]
        ni = int(ni_arr[ip])
        nj = int(nj_arr[ip])
        p_i[ip, :ni, :ni] = projectors[li]
        p_j[ip, :nj, :nj] = projectors[lj]
        if not dynamic_soc:
            iu_up[ip, :ni] = si
            iu_dn[ip, :ni] = si + nwan
            jv_up[ip, :nj] = sj
            jv_dn[ip, :nj] = sj + nwan

    acc_a = np.zeros((n_pairs, n_r, 4, 4), dtype=np.complex128)
    r_all = list(acc_r) + [(-r[0], -r[1], -r[2]) for r in acc_r]

    if dynamic_soc:
        nd = int(d_idx.size)
        iu_up_local = np.zeros((n_pairs, max_ni), dtype=np.int64)
        iu_dn_local = np.zeros((n_pairs, max_ni), dtype=np.int64)
        jv_up_local = np.zeros((n_pairs, max_nj), dtype=np.int64)
        jv_dn_local = np.zeros((n_pairs, max_nj), dtype=np.int64)

        dynamic_indices = _selectors_in_dynamic_basis(slices, nwan, d_idx)
        for ip, (li, lj) in enumerate(acc_pairs):
            ni = int(ni_arr[ip])
            nj = int(nj_arr[ip])
            idx_i = dynamic_indices[li]
            idx_j = dynamic_indices[lj]
            iu_up_local[ip, :ni] = idx_i[:ni]
            iu_dn_local[ip, :ni] = idx_i[ni:]
            jv_up_local[ip, :nj] = idx_j[:nj]
            jv_dn_local[ip, :nj] = idx_j[nj:]

        for z, weight in energy_mesh:
            z_abs = z + float(efermi)
            h_eff_z = _build_dynamic_h_eff(hk_up, hk_dn, d_idx, p_idx, lambda_uu, lambda_ud, lambda_du, lambda_dd, z_abs)
            evals_z = np.zeros((hk_up.shape[0], 2 * nd), dtype=np.float64)
            evecs_z = np.zeros((hk_up.shape[0], 2 * nd, 2 * nd), dtype=np.complex128)
            for ik in range(hk_up.shape[0]):
                ww, cc = np.linalg.eigh(h_eff_z[ik] - float(efermi) * np.eye(2 * nd))
                evals_z[ik] = ww
                evecs_z[ik] = cc

            gk = _green_mesh_from_eigensystem(evals_z, evecs_z, z)
            gr_all = _sum_gr_flat(gk, r_all, kpts)
            _accumulate_tb2j_A_numba(
                GR=gr_all[:n_r],
                GRm=gr_all[n_r:],
                ni_arr=ni_arr,
                nj_arr=nj_arr,
                iu_up=iu_up_local,
                iu_dn=iu_dn_local,
                jv_up=jv_up_local,
                jv_dn=jv_dn_local,
                P_i=p_i,
                P_j=p_j,
                acc_A=acc_a,
                weight_over_pi=weight / np.pi,
            )
    else:
        kdata = _precompute_spinor_kdata(h_spin, slices, efermi)
        for z, weight in energy_mesh:
            gk = _green_mesh_from_eigensystem(kdata["evals"], kdata["evecs"], z)
            gr_all = _sum_gr_flat(gk, r_all, kpts)
            _accumulate_tb2j_A_numba(
                GR=gr_all[:n_r],
                GRm=gr_all[n_r:],
                ni_arr=ni_arr,
                nj_arr=nj_arr,
                iu_up=iu_up,
                iu_dn=iu_dn,
                jv_up=jv_up,
                jv_dn=jv_dn,
                P_i=p_i,
                P_j=p_j,
                acc_A=acc_a,
                weight_over_pi=weight / np.pi,
            )

    pair_to_ip = {pair: ip for ip, pair in enumerate(acc_pairs)}
    r_to_ir = {r: ir for ir, r in enumerate(acc_r)}
    tensor = np.zeros((len(pair_meta), len(axes), len(axes)), dtype=np.float64)
    extra = {
        "jiso_tb2j": np.zeros(len(pair_meta), dtype=np.float64),
        "dmi_tb2j": np.zeros((len(pair_meta), 3), dtype=np.float64),
        "jani_tb2j": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
        "J_iso_tensor_r": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
        "J_gamma_r": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
        "J_antisym_aab_r": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
        "J_dmi_tensor_r": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
        "J_aab_full_r": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
    }
    ax_to_full = {"x": 0, "y": 1, "z": 2}
    for ib, meta in enumerate(pair_meta):
        li = int(meta["li"])
        lj = int(meta["lj"])
        rt = tuple(int(x) for x in meta["R"])
        ip = pair_to_ip[(li, lj)]
        ir = r_to_ir[rt]
        ip_m = pair_to_ip.get((lj, li))
        ir_m = r_to_ir.get((-rt[0], -rt[1], -rt[2]))
        val = acc_a[ip, ir]
        val_m = acc_a[ip_m, ir_m] if ip_m is not None and ir_m is not None else np.zeros((4, 4), dtype=np.complex128)

        dec = _decompose_tb2j_pair(val, val_m, collinear_override=collinear_override)
        jiso_tb2j = dec["jiso"]
        gamma = dec["gamma"]
        m_raw = dec["m_raw"]
        ma = dec["ma"]
        dmi_tb2j = dec["dmi"]
        dmi_mat = dec["dmi_mat"]
        jphys = dec["jfull_dmi"]
        jfull = dec["jfull_aab"]

        for ia, a in enumerate(axes):
            for jb, b in enumerate(axes):
                tensor[ib, ia, jb] = 1000.0 * jphys[ax_to_full[a], ax_to_full[b]]
        extra["jiso_tb2j"][ib] = 1000.0 * jiso_tb2j
        extra["dmi_tb2j"][ib] = 1000.0 * dmi_tb2j
        extra["jani_tb2j"][ib] = 1000.0 * m_raw
        extra["J_iso_tensor_r"][ib] = 1000.0 * np.eye(3, dtype=np.float64) * jiso_tb2j
        extra["J_gamma_r"][ib] = 1000.0 * gamma
        extra["J_antisym_aab_r"][ib] = 1000.0 * ma
        extra["J_dmi_tensor_r"][ib] = 1000.0 * dmi_mat
        extra["J_aab_full_r"][ib] = 1000.0 * jfull
    return tensor, acc_a, extra, {"n_chunks": 1, "nproc": 1, "n_pairs_acc": n_pairs, "n_R_acc": n_r}


def _build_pair_meta(epr_path, mag_atoms, neighbours):
    global_to_local = {g: i for i, g in enumerate(mag_atoms)}
    out = []
    for n in neighbours:
        gi, gj = int(n["i"]), int(n["j"])
        if gi not in global_to_local or gj not in global_to_local:
            continue
        out.append(
            {
                "gi": gi,
                "gj": gj,
                "li": int(global_to_local[gi]),
                "lj": int(global_to_local[gj]),
                "R": tuple(int(x) for x in n["R"]),
                "dist": float(n["distance"]),
                "shell": int(n.get("shell_idx", 0)),
            }
        )
    if not out:
        raise RuntimeError(f"No directed bonds matched --mag_atoms for {epr_path}")
    return out


def _write_tensor_h5(path, args, labels, pair_meta, tensor, trace_acc, orbits, elapsed, nk, nE, exe_info, axes, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    orbit_label_by_key = _orbit_label_map(pair_meta, orbits)
    with h5py.File(path, "w") as h5:
        def put_string(group, name, value):
            group.create_dataset(name, data=np.array(str(value), dtype=object), dtype=str_dt)

        basic = h5.create_group("basic_data")
        basic.create_dataset("nat", data=np.array(len(labels), dtype=np.int64))
        basic.create_dataset("atom_labels", data=np.asarray(labels, dtype=object), dtype=str_dt)
        basic.create_dataset("kmesh", data=np.asarray(args.kmesh, dtype=np.int64))
        basic.create_dataset("efermi_ev", data=np.array(float(args.efermi), dtype=np.float64))
        basic.create_dataset("spin_direction", data=np.asarray(args.spin_direction, dtype=np.float64))
        basic.create_dataset("tensor_axes", data=np.asarray(axes, dtype=object), dtype=str_dt)
        put_string(basic, "unit", "meV")
        put_string(basic, "hr_unit", args.hr_unit)
        put_string(basic, "integrator", args.integrator)
        put_string(basic, "kernel", args.kernel)
        put_string(basic, "kernel_family", args.kernel)
        put_string(basic, "hamiltonian_sign", "minus")
        spin_magnitude = float(getattr(args, "spin_magnitude", 1.0))
        put_string(
            basic,
            "spin_normalization",
            "unit_vector" if spin_magnitude == 1.0 else "spin_operator",
        )
        put_string(
            basic,
            "bond_coverage",
            "directed_mate_complete" if bool(args.all_bonds) else "canonical_half",
        )
        put_string(basic, "realspace_gauge", "i_at_0_j_at_R")
        basic.create_dataset("source_spin_magnitude", data=np.array(spin_magnitude, dtype=np.float64))
        directed_weight = 1.0 if args.kernel == "tb2j" else 0.5
        stored_bond_weight = directed_weight if bool(args.all_bonds) else 2.0 * directed_weight
        basic.create_dataset(
            "directed_bond_weight",
            data=np.array(stored_bond_weight, dtype=np.float64),
        )
        put_string(basic, "base_hamiltonian", "epr_up_down")
        put_string(basic, "input_groupby", "")
        put_string(basic, "internal_groupby", "spin")
        put_string(basic, "win_path", getattr(args, "_soc_win_path", "") or "")
        soc_entries = getattr(args, "_soc_entries", []) or []
        basic.create_dataset(
            "additional_soc",
            data=np.array(bool(soc_entries), dtype=np.bool_),
        )
        put_string(basic, "soc_mode", "atomic" if soc_entries else "none")
        basic.create_dataset(
            "soc_entries",
            data=np.asarray(
                [
                    f"{e.get('selector', e['element'] + '-' + e['orbital'])}:"
                    f"{float(e['lambda_ev']):.16g}"
                    for e in soc_entries
                ],
                dtype=object,
            ),
            dtype=str_dt,
        )
        basic.create_dataset(
            "soc_resolved_groups",
            data=np.asarray(
                [
                    ";".join(
                        ",".join(str(int(index)) for index in group)
                        for group in entry["groups"]
                    )
                    for entry in soc_entries
                ],
                dtype=object,
            ),
            dtype=str_dt,
        )
        put_string(basic, "command", " ".join(sys.argv))
        basic.create_dataset("empoints", data=np.array(int(args.empoints), dtype=np.int64))
        basic.create_dataset("nproc", data=np.array(int(args.nproc), dtype=np.int64))
        basic.create_dataset("nk", data=np.array(int(nk), dtype=np.int64))
        basic.create_dataset("nE", data=np.array(int(nE), dtype=np.int64))
        basic.create_dataset("elapsed_s", data=np.array(float(elapsed), dtype=np.float64))
        basic.create_dataset("n_chunks", data=np.array(int(exe_info.get("n_chunks", 1)), dtype=np.int64))
        basic.create_dataset(
            "mpi_size", data=np.array(int(exe_info.get("mpi_size", 1)), dtype=np.int64)
        )
        _copy_epr_structure_basic(h5, args.epr_up)

        bonds = h5.create_group("bonds")
        bonds.create_dataset("mag_i_atom", data=np.asarray([m["gi"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_j_atom", data=np.asarray([m["gj"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_i_local", data=np.asarray([m["li"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_j_local", data=np.asarray([m["lj"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("R", data=np.asarray([m["R"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("distance_ang", data=np.asarray([m["dist"] for m in pair_meta], dtype=np.float64))
        bonds.create_dataset("shell", data=np.asarray([m["shell"] for m in pair_meta], dtype=np.int64))
        orbit_labels = [orbit_label_by_key.get((m["gi"], m["gj"], tuple(m["R"])), "NA") for m in pair_meta]
        bonds.create_dataset("orbit_label", data=np.asarray(orbit_labels, dtype=object), dtype=str_dt)
        bonds.create_dataset("mirror_index", data=_mirror_indices(pair_meta))

        dset = h5.create_dataset("J_tensor_r", data=np.asarray(tensor, dtype=np.float64))
        dset.attrs["shape"] = "(nBond,tensor_axis_a,tensor_axis_b)"
        dset.attrs["tensor_axis_order"] = ",".join(axes)
        dset.attrs["unit"] = "meV"
        h5.create_dataset("trace_acc", data=np.asarray(trace_acc, dtype=np.complex128))
        if extra:
            if "jiso_tb2j" in extra:
                iso_dataset = h5.create_dataset(
                    "J_iso_r",
                    data=np.asarray(extra["jiso_tb2j"], dtype=np.float64),
                )
                iso_dataset.attrs["unit"] = "meV"
            for key in (
                "J_iso_tensor_r",
                "J_gamma_r",
                "J_dmi_tensor_r",
                "J_antisym_aab_r",
                "J_aab_full_r",
            ):
                if key in extra:
                    ds = h5.create_dataset(key, data=np.asarray(extra[key], dtype=np.float64))
                    ds.attrs["shape"] = "(nBond,3,3)"
                    ds.attrs["tensor_axis_order"] = "x,y,z"
                    ds.attrs["unit"] = "meV"
            if "dmi_tb2j" in extra:
                ds = h5.create_dataset("DMI_r", data=np.asarray(extra["dmi_tb2j"], dtype=np.float64))
                ds.attrs["shape"] = "(nBond,3)"
                ds.attrs["axis_order"] = "x,y,z"
                ds.attrs["unit"] = "meV"
            grp = h5.create_group("tb2j_extra")
            for key, val in extra.items():
                grp.create_dataset(key, data=np.asarray(val))


def _write_tensor_text(
    path,
    labels,
    pair_meta,
    tensor,
    orbits,
    axes,
    kernel,
    extra=None,
    *,
    source_label="EPR spinor H(k)",
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    orbit_label_by_key = _orbit_label_map(pair_meta, orbits)

    def mat_for(key, ib, default=None):
        if extra and key in extra:
            return np.asarray(extra[key][ib], dtype=np.float64)
        if default is not None:
            return np.asarray(default, dtype=np.float64)
        return None

    def write_matrix(f, title, mat):
        f.write(f"{title}:\n")
        arr = np.asarray(mat, dtype=np.float64)
        for row in arr:
            f.write("  " + " ".join(f"{float(x):+.10e}" for x in row) + "\n")

    with open(path, "w") as f:
        f.write(f"# Exchange Tensor J^{{ab}}(R) Results - {source_label}\n")
        f.write(f"# kernel = {kernel}, unit = meV\n\n")
        f.write("# Tensor decomposition convention\n")
        f.write("#   J_full = J_iso*I + Gamma + J_DMI\n")
        f.write("#   Gamma is symmetric traceless; DMI vector is (Dx,Dy,Dz)\n")
        f.write("#   J_DMI matrix is [[0,Dz,-Dy],[-Dz,0,Dx],[Dy,-Dx,0]] for E=-S_i^a J_ab S_j^b\n")
        f.write("#   Raw Aab tensors are printed only for comparison/debug.\n\n")
        f.write("# Detailed Directed Bond Summary\n")
        f.write("Orbit\tBond\tDistance(A)\tR\tJiso\tDx\tDy\tDz\n")
        f.write("-" * 96 + "\n")
        for ib, m in enumerate(pair_meta):
            mat = np.asarray(tensor[ib], dtype=float)
            full = mat_for("J_tensor_r", ib, mat)
            jiso_tensor = mat_for("J_iso_tensor_r", ib)
            gamma = mat_for("J_gamma_r", ib)
            dmi_tensor = mat_for("J_dmi_tensor_r", ib)
            aab_full = mat_for("J_aab_full_r", ib)
            aab_anti = mat_for("J_antisym_aab_r", ib)
            dmi = np.zeros(3, dtype=np.float64)
            jiso = float(np.trace(full) / 3.0)
            if extra and "jiso_tb2j" in extra:
                jiso = float(extra["jiso_tb2j"][ib])
            if extra and "dmi_tb2j" in extra:
                dmi = np.asarray(extra["dmi_tb2j"][ib], dtype=float)
            orbit = orbit_label_by_key.get((m["gi"], m["gj"], tuple(m["R"])), "NA")
            bond = f"{_label_for(labels, m['gi'])}-{_label_for(labels, m['gj'])}"
            f.write(
                f"{orbit}\t{bond}\t{float(m['dist']):.8f}\t{_fmt_r(m['R'])}\t"
                f"{jiso:+.10e}\t{dmi[0]:+.10e}\t{dmi[1]:+.10e}\t{dmi[2]:+.10e}\n"
            )
            f.write(
                f"Bond detail: orbit={orbit} pair i={int(m['gi'])}({_label_for(labels, m['gi'])}) "
                f"j={int(m['gj'])}({_label_for(labels, m['gj'])}) R={_fmt_r(m['R'])} "
                f"dist={float(m['dist']):.8f} A\n"
            )
            f.write(f"J_iso = {jiso:+.10e} meV\n")
            f.write(f"DMI = ({dmi[0]:+.10e}, {dmi[1]:+.10e}, {dmi[2]:+.10e}) meV\n")
            if jiso_tensor is not None:
                write_matrix(f, "J_iso_tensor = J_iso*I", jiso_tensor)
            if gamma is not None:
                write_matrix(f, "Gamma_symmetric_traceless", gamma)
            if dmi_tensor is not None:
                write_matrix(f, "J_DMI_antisymmetric_from_DMI", dmi_tensor)
            write_matrix(f, "J_full = J_iso_tensor + Gamma + J_DMI", full)
            if aab_full is not None:
                write_matrix(f, "Raw_Aab_full_debug", aab_full)
            if aab_anti is not None:
                write_matrix(f, "Raw_Aab_antisymmetric_debug", aab_anti)
            f.write("\n")


def run(args, comm=None):
    axes = _parse_axes(args.axes)
    kpts = _full_k_mesh(args.kmesh)
    print(f"[J-epr-tensor] building H_up/H_down: nk={len(kpts)} kmesh={tuple(args.kmesh)}", flush=True)
    hk_up = _build_hk_from_epr(args.epr_up, kpts, unit=args.hr_unit)
    hk_dn = _build_hk_from_epr(args.epr_dn, kpts, unit=args.hr_unit)
    if hk_up.shape != hk_dn.shape:
        raise ValueError(f"up/down H(k) shape mismatch: {hk_up.shape} vs {hk_dn.shape}")
    dim = int(hk_up.shape[1])
    slices = _load_slices(args, dim)
    print(f"[J-epr-tensor] slices={ {int(k): (v.start, v.stop) for k, v in slices.items()} }", flush=True)

    h_spin = spinor_from_collinear(hk_up, hk_dn, n=args.spin_direction)
    h_spin, soc_entries, soc_win_path = _apply_model_soc(h_spin, args, dim)
    args._soc_entries = soc_entries
    args._soc_win_path = soc_win_path

    mag_atoms = _normalize_mag_atoms(args)
    neighbours = _find_nearest_neighbours_from_epr(
        args.epr_up,
        mag_atom_indices=mag_atoms,
        n_shells=int(args.n_shells),
        d_max=float(args.d_max),
        all_bonds=bool(args.all_bonds),
    )
    if args.nn_only:
        neighbours = [n for n in neighbours if int(n.get("shell_idx", 0)) == 1]
    pair_meta = _build_pair_meta(args.epr_up, mag_atoms, neighbours)
    labels = _atom_labels(args.epr_up, args.atom_labels, species_labels=args.species_labels)
    orbits = _group_orbits_epr(
        args.epr_up,
        neighbours,
        use_symmetry=not args.no_symmetry,
        symprec=float(args.symprec),
        labels=labels,
        species_labels=args.species_labels,
        orbit_grouping=args.orbit_grouping,
        angle_tolerance=float(args.angle_tolerance),
        debug_orbits=bool(args.debug_orbits),
        debug_orbit_shell=args.debug_orbit_shell,
        debug_epr_positions=bool(args.debug_epr_positions),
    )

    if args.integrator == "contour":
        energy_mesh = get_semicircle_contour(emin=args.emin, emax=0.0, npoints=args.empoints)
    elif args.integrator == "cfr_ozaki":
        energy_mesh = get_cfr_ozaki_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)
    else:
        energy_mesh = get_cfr_pole_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)

    dynamic_soc_val = getattr(args, "dynamic_soc", False)
    d_idx, p_idx = None, None
    lambda_uu, lambda_ud, lambda_du, lambda_dd = None, None, None, None
    if dynamic_soc_val:
        win_path = _infer_win_path(args.epr_up, explicit=args.win)
        if not win_path:
            raise ValueError("--dynamic_soc requires --win for subspace selection")
        if getattr(args, "lambda_soc", None) is None:
            raise ValueError("--dynamic_soc requires --lambda_soc")
        if not str(args.d_subspace).strip() or not str(args.soc_active).strip():
            raise ValueError("--dynamic_soc requires --d_subspace and --soc_active")
        d_groups = _selected_groups(win_path, args.d_subspace, dim)
        p_groups = _selected_groups(win_path, args.soc_active, dim)
        d_idx, d_summary = _group_indices(d_groups)
        p_idx, p_summary = _group_indices(p_groups)
        hsoc, _soc_summary = _soc_matrix_for_groups(
            p_groups,
            dim,
            float(args.lambda_soc),
            p_order=args.p_order,
            d_order=args.d_order,
        )
        hsoc_p = hsoc[np.ix_(np.concatenate([p_idx, p_idx + dim]), np.concatenate([p_idx, p_idx + dim]))]
        np_ = int(p_idx.size)
        lambda_uu = hsoc_p[:np_, :np_]
        lambda_ud = hsoc_p[:np_, np_:]
        lambda_du = hsoc_p[np_:, :np_]
        lambda_dd = hsoc_p[np_:, np_:]
        print(
            f"[J-epr-tensor] Setup dynamic SOC downfolding: d={d_summary} p={p_summary} lambda={args.lambda_soc:g} eV",
            flush=True,
        )

    rank, size = rank_size(comm)
    local_energy_mesh = partition_sequence(energy_mesh, comm)
    print(
        f"[J-epr-tensor] computing spinor tensor: kernel={args.kernel} nBond={len(pair_meta)} "
        f"nE={len(energy_mesh)} local_nE={len(local_energy_mesh)} "
        f"axes={axes} mpi={size}",
        flush=True,
    )
    t0 = time.time()
    collinear_override_val = getattr(args, "collinear_override", False)

    def integrate_local():
        if args.kernel == "direct":
            tensor_local, trace_local, _info = _compute_tensor_direct(
                h_spin,
                slices,
                pair_meta,
                kpts,
                local_energy_mesh,
                args.efermi,
                axes,
                nproc=int(args.nproc) if size == 1 else 1,
                collinear_override=collinear_override_val,
                dynamic_soc=dynamic_soc_val,
                hk_up=hk_up,
                hk_dn=hk_dn,
                d_idx=d_idx,
                p_idx=p_idx,
                lambda_uu=lambda_uu,
                lambda_ud=lambda_ud,
                lambda_du=lambda_du,
                lambda_dd=lambda_dd,
            )
            return tensor_local, trace_local, None
        tensor_local, trace_local, extra_local, _info = _compute_tensor_tb2j(
            h_spin,
            slices,
            pair_meta,
            kpts,
            local_energy_mesh,
            args.efermi,
            axes,
            collinear_override=collinear_override_val,
            dynamic_soc=dynamic_soc_val,
            hk_up=hk_up,
            hk_dn=hk_dn,
            d_idx=d_idx,
            p_idx=p_idx,
            lambda_uu=lambda_uu,
            lambda_ud=lambda_ud,
            lambda_du=lambda_du,
            lambda_dd=lambda_dd,
        )
        return tensor_local, trace_local, extra_local

    reduced = collective_sum(comm, integrate_local)
    if rank != 0:
        return
    if reduced is None:  # pragma: no cover - defensive communicator guard
        raise RuntimeError("MPI root did not receive tensor J reduction")
    tensor, trace_acc, extra = reduced
    exe_info = {
        "n_chunks": max(1, size),
        "nproc": size if size > 1 else int(args.nproc),
        "mpi_size": size,
    }
    elapsed = time.time() - t0

    # Scale J by S^2
    spin_magnitude_val = getattr(args, "spin_magnitude", 1.0)
    if spin_magnitude_val != 1.0:
        s2 = float(spin_magnitude_val) ** 2
        tensor = tensor / s2
        if extra:
            for key in (
                "jiso_tb2j",
                "dmi_tb2j",
                "jani_tb2j",
                "J_iso_tensor_r",
                "J_gamma_r",
                "J_antisym_aab_r",
                "J_dmi_tensor_r",
                "J_aab_full_r",
            ):
                if key in extra:
                    extra[key] = extra[key] / s2

    os.makedirs(args.out_dir, exist_ok=True)
    out_h5 = args.out_h5 if os.path.isabs(args.out_h5) else os.path.join(args.out_dir, args.out_h5)
    out_txt = args.out_name if os.path.isabs(args.out_name) else os.path.join(args.out_dir, args.out_name)
    _write_tensor_h5(
        out_h5,
        args,
        labels,
        pair_meta,
        tensor,
        trace_acc,
        orbits,
        elapsed,
        len(kpts),
        len(energy_mesh),
        exe_info,
        axes,
        extra=extra,
    )
    _write_tensor_text(out_txt, labels, pair_meta, tensor, orbits, axes, args.kernel, extra=extra)
    print(
        f"[J-epr-tensor] done elapsed_s={elapsed:.2f} "
        f"n_chunks={exe_info.get('n_chunks', 1)} nproc={exe_info.get('nproc', 1)}",
        flush=True,
    )
    print(f"[J-epr-tensor] wrote {out_txt}", flush=True)
    print(f"[J-epr-tensor] wrote {out_h5}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Spinor exchange tensor J^{ab}(R) directly from EPR H(k)")
    ap.add_argument("--epr_up", required=True, help="Spin-up EPR HDF5")
    ap.add_argument("--epr_dn", required=True, help="Spin-down EPR HDF5")
    ap.add_argument("--efermi", type=float, required=True, help="Fermi energy in eV")
    ap.add_argument("--kmesh", type=int, nargs=3, required=True)
    ap.add_argument("--mag_atoms", type=int, nargs="+", required=True, help="Magnetic atom indices")
    ap.add_argument("--mag_atoms_base", type=int, default=0, choices=[0, 1])
    ap.add_argument("--slices", required=True, help="Manual local orbital slices, e.g. '0:0:5,1:5:10'")
    ap.add_argument("--hr_unit", default="ry", choices=["ry", "ev", "ha"], help="EPR hopping unit")
    ap.add_argument(
        "--kernel",
        choices=["tb2j", "direct"],
        default="tb2j",
        help="tb2j: TB2J-like Pauli A-tensor mapping. direct: raw D_i^a G D_j^b G trace.",
    )
    ap.add_argument("--axes", default="xyz", help="Tensor axes to compute, subset of xyz")
    ap.add_argument("--spin_direction", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    ap.add_argument(
        "--soc",
        default="",
        help="Generic model SOC specs, e.g. 'Te:p:0.5;Cr:d:0.05'. Groups are inferred from --win.",
    )
    ap.add_argument("--win", default=None, help="wannier90 .win file used to infer p/d SOC orbital groups")
    ap.add_argument("--soc_element", default="", help="Element for compatibility --lambda_te p-SOC mode")
    ap.add_argument("--lambda_te", type=float, default=0.0, help="Compatibility model onsite p-SOC lambda in eV; prefer --soc")
    ap.add_argument("--soc_p_groups", default="", help="Semicolon-separated p orbital groups, e.g. '10,11,12;25,26,27'")
    ap.add_argument("--soc_groups_base", type=int, default=0, choices=[0, 1], help="Index base for --soc_p_groups")
    ap.add_argument("--soc_p_groups_base", dest="soc_groups_base", type=int, choices=[0, 1], help=argparse.SUPPRESS)
    ap.add_argument("--p_order", default=WANNIER90_P_ORDER, help="p orbital order inside each SOC group")
    ap.add_argument("--d_order", default=WANNIER90_D_ORDER, help="d orbital order inside each SOC group")
    ap.add_argument("--n_shells", type=int, default=10)
    ap.add_argument("--d_max", type=float, default=20.0)
    ap.add_argument("--all_bonds", action="store_true", default=True, help="Keep directed bonds")
    ap.add_argument("--canonical_bonds", dest="all_bonds", action="store_false", help="Fold equivalent directed bonds")
    ap.add_argument("--nn_only", action="store_true")
    ap.add_argument("--atom_labels", default="", help="Comma-separated atom labels")
    ap.add_argument("--species_labels", nargs="*", default=None)
    ap.add_argument("--no_symmetry", action="store_true")
    ap.add_argument("--orbit_grouping", choices=["spglib", "shell"], default="spglib")
    ap.add_argument("--symprec", type=float, default=1.0e-4)
    ap.add_argument("--angle_tolerance", type=float, default=-1.0)
    ap.add_argument("--debug_orbits", action="store_true")
    ap.add_argument("--debug_orbit_shell", type=int, default=None)
    ap.add_argument("--debug_epr_positions", action="store_true")
    ap.add_argument("--integrator", choices=["contour", "cfr_ozaki", "cfr_pole"], default="contour")
    ap.add_argument("--emin", type=float, default=-25.0)
    ap.add_argument("--empoints", type=int, default=500)
    ap.add_argument("--cfr_beta", type=float, default=1000.0)
    ap.add_argument("--nproc", type=int, default=1, help="Parallelize over energy chunks")
    ap.add_argument("--collinear_override", action="store_true", help="Override J_zz with (J_xx+J_yy)/2 in collinear calculations")
    ap.add_argument("--spin_magnitude", type=float, default=1.0, help="Spin magnitude S to scale J by 1/S^2")
    ap.add_argument("--d_subspace", default="", help="Target low-energy subspace for dynamic SOC downfolding")
    ap.add_argument("--soc_active", default="", help="SOC-active ligand subspace for dynamic SOC downfolding")
    ap.add_argument("--lambda_soc", type=float, default=None, help="Atomic SOC lambda in eV for dynamic SOC downfolding")
    ap.add_argument("--dynamic_soc", action="store_true", help="Perform dynamic ligand SOC downfolding inside the energy loop")
    ap.add_argument("--out_dir", default="J_epr_tensor")
    ap.add_argument("--out_name", default="J_epr_tensor.txt")
    ap.add_argument("--out_h5", default="J_epr_tensor.h5")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
