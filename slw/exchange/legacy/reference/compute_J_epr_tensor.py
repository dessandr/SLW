# Numerical parity implementation retained behind the native exchange engine.
"""Compute spinor exchange tensor J^{ab}(R) directly from EPR H(k)."""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import re
import sys
import time

import h5py
import numpy as np

from slw.exchange.legacy.reference.compute_J_epr_kspace import (
    _atom_labels,
    _copy_epr_structure_basic,
    _find_nearest_neighbours_from_epr,
    _group_orbits_epr,
    _mirror_indices,
    _normalize_mag_atoms,
    _orbit_label_map,
    _split_chunks,
)
from slw.exchange.legacy.diagnose_J_epr_kspace import _build_hk_from_epr, _full_k_mesh, _load_slices
from slw.exchange.legacy.lkag_solver import (
    _accumulate_tb2j_A_numba,
    get_cfr_ozaki_mesh,
    get_cfr_pole_mesh,
    get_semicircle_contour,
)
from slw.exchange.legacy.spinor_model import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    add_atomic_d_soc,
    add_atomic_p_soc,
    spinor_from_collinear,
)
from slw.exchange.legacy.build_downfolded_static_soc import (
    _group_indices,
    _selected_groups,
    _soc_matrix_for_groups,
    _downfold_one_k,
)


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
        put_str(basic, "hamiltonian_convention", "E=-sum_{i!=j} J_ij e_i.e_j; tensor follows same pair list")
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
        f.write("# convention = E=-sum_{i!=j} J_ij e_i.e_j; same directed pair list as Jr.h5\n")
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
                for ibb, b in enumerate(tensor_axes):
                    f.write(
                        f"{ib}\t{payload['orbit'][ib]}\t{int(payload['gi'][ib])}\t{int(payload['gj'][ib])}\t"
                        f"{int(payload['gi'][ib]) + 1}\t{int(payload['gj'][ib]) + 1}\t"
                        f"{int(payload['R'][ib,0])}\t{int(payload['R'][ib,1])}\t{int(payload['R'][ib,2])}\t"
                        f"{float(payload['dist'][ib]):.12e}\t{int(payload['shell'][ib])}\t"
                        f"{a}\t{b}\t{tensor[ib, ia, ibb]:.12e}\t{float(payload['j_iso'][ib]):.12e}\n"
                    )
    return tsv


_WORKER_STATIC = None


def _parse_soc_p_groups(text, *, base=0):
    if not text:
        return []
    groups = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        vals = [int(x.strip()) - int(base) for x in item.replace(",", " ").split()]
        if len(vals) != 3:
            raise ValueError(f"Each --soc_p_groups entry must contain 3 orbital indices, got {item!r}")
        groups.append(vals)
    return groups


def _clean_species(label):
    return re.sub(r"\d+$", "", str(label).strip()).lower()


def _projection_orbitals(text):
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*0\b", "s", str(text), flags=re.I)
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*1\b", "p", clean, flags=re.I)
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*2\b", "d", clean, flags=re.I)
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*3\b", "f", clean, flags=re.I)
    fields = [x.strip().lower() for x in re.split(r"[,; \t]+", clean) if x.strip()]
    out = []
    for item in fields:
        item = item.strip("{}()")
        if item in {"s", "p", "d", "f"}:
            out.append(item)
    return out


def _orbital_count(kind):
    counts = {"s": 1, "p": 3, "d": 5, "f": 7}
    if kind not in counts:
        raise ValueError(f"Unsupported projection orbital kind '{kind}'")
    return counts[kind]


def _parse_win_atoms_and_projections(win_path):
    atoms = []
    projections = []
    block = None
    with open(win_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("!", 1)[0].split("#", 1)[0].strip()
            low = line.lower()
            if not line:
                continue
            if low.startswith("begin atoms_frac") or low.startswith("begin atoms_cart"):
                block = "atoms"
                continue
            if low.startswith("begin projections"):
                block = "projections"
                continue
            if low.startswith("end atoms_frac") or low.startswith("end atoms_cart") or low.startswith("end projections"):
                block = None
                continue
            if block == "atoms":
                toks = line.split()
                if len(toks) >= 4:
                    atoms.append(toks[0])
            elif block == "projections":
                if ":" in line:
                    lhs, rhs = line.split(":", 1)
                else:
                    lhs, rhs = line, line
                orbitals = _projection_orbitals(rhs)
                if orbitals:
                    projections.append((lhs.strip(), orbitals))
    return atoms, projections


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


def _win_projection_groups(win_path):
    atoms, projections = _parse_win_atoms_and_projections(win_path)
    if not atoms:
        raise ValueError(f"No atoms_frac/atoms_cart block found in {win_path}")
    if not projections:
        raise ValueError(f"No projections block found in {win_path}")

    groups = []
    offset = 0
    used_atom_projection = [False] * len(atoms)
    for label, orbitals in projections:
        label_clean = _clean_species(label)
        exact_matches = [i for i, atom in enumerate(atoms) if atom.lower() == label.lower()]
        species_matches = [i for i, atom in enumerate(atoms) if _clean_species(atom) == label_clean]
        matches = exact_matches if exact_matches else species_matches
        if not matches:
            raise ValueError(f"Projection label '{label}' did not match any atom in {win_path}")
        for ia in matches:
            if used_atom_projection[ia]:
                raise ValueError(
                    f"Atom '{atoms[ia]}' is matched by multiple projection lines in {win_path}; "
                    "automatic SOC group inference would be ambiguous."
                )
            used_atom_projection[ia] = True
            for orb in orbitals:
                n = _orbital_count(orb)
                groups.append(
                    {
                        "atom_index": ia,
                        "atom_label": atoms[ia],
                        "element": _clean_species(atoms[ia]),
                        "orbital": orb,
                        "indices": list(range(offset, offset + n)),
                    }
                )
                offset += n
    return atoms, groups, offset


def _parse_soc_specs(text):
    specs = []
    if not text:
        return specs
    for item in re.split(r"[;]+", str(text)):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in re.split(r"[:,= \t]+", item) if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"Each --soc entry must be element:orbital:lambda_eV, got {item!r}")
        elem, orb, lam = parts
        orb = orb.lower()
        if orb not in {"p", "d"}:
            raise ValueError(f"Unsupported SOC orbital {orb!r}; supported: p, d")
        specs.append((_clean_species(elem), orb, float(lam)))
    return specs


def _resolve_soc_entries(args, nwan):
    entries = []
    specs = _parse_soc_specs(args.soc)
    legacy_groups = _parse_soc_p_groups(args.soc_p_groups, base=args.soc_groups_base)
    legacy_lambda = float(args.lambda_te)
    legacy_element = _clean_species(args.soc_element)

    if legacy_groups:
        if abs(legacy_lambda) <= 0.0:
            raise ValueError("--soc_p_groups was provided but --lambda_te is zero")
        entries.append(
            {
                "element": legacy_element,
                "orbital": "p",
                "lambda_ev": legacy_lambda,
                "groups": legacy_groups,
                "source": "manual --soc_p_groups",
            }
        )
    elif abs(legacy_lambda) > 0.0:
        matching = [lam for elem, orb, lam in specs if elem == legacy_element and orb == "p"]
        if matching:
            if any(abs(float(lam) - legacy_lambda) > 1e-14 for lam in matching):
                raise ValueError(
                    f"Conflicting SOC lambda for {legacy_element}:p between --soc and --lambda_te={legacy_lambda}"
                )
        else:
            specs.append((legacy_element, "p", legacy_lambda))

    if not specs:
        return entries, None

    win_path = _infer_win_path(args.epr_up, explicit=args.win)
    if not win_path:
        raise ValueError("--win is required for automatic SOC group inference")
    atoms, all_groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(f"{win_path} projection count={nproj} but EPR H(k) nwan={nwan}")

    for elem, orb, lam in specs:
        if legacy_groups and elem == legacy_element and orb == "p" and abs(lam - legacy_lambda) < 1e-14:
            continue
        groups = [g["indices"] for g in all_groups if g["element"] == elem and g["orbital"] == orb]
        if not groups:
            known = sorted({(g["element"], g["orbital"]) for g in all_groups})
            raise ValueError(f"No {elem}:{orb} projection groups found in {win_path}; known={known}")
        entries.append(
            {
                "element": elem,
                "orbital": orb,
                "lambda_ev": float(lam),
                "groups": groups,
                "source": win_path,
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
            out = add_atomic_p_soc(out, groups, lambda_ev=lam, order=args.p_order, inplace=False)
        elif orb == "d":
            out = add_atomic_d_soc(out, groups, lambda_ev=lam, order=args.d_order, inplace=False)
        else:
            raise ValueError(f"Unsupported SOC orbital: {orb}")
        print(
            f"[J-epr-tensor] added model {entry['element']}:{orb} SOC "
            f"lambda={lam:g} eV groups={groups} source={entry['source']}",
            flush=True,
        )
    return out, entries, win_path


def _spinor_atom_indices(slc, nwan):
    up = np.arange(slc.start, slc.stop, dtype=np.int64)
    return np.concatenate([up, up + int(nwan)])


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
    out = {}
    for site, slc in slices.items():
        idx = _spinor_atom_indices(slc, nwan)
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
    nk, dim, _ = h_spin.shape
    nwan = dim // 2
    eye = np.eye(dim, dtype=np.complex128)
    evals = np.zeros((nk, dim), dtype=np.float64)
    evecs = np.zeros((nk, dim, dim), dtype=np.complex128)
    coeffs = {
        int(site): np.zeros((nk, 2 * (slc.stop - slc.start), dim), dtype=np.complex128)
        for site, slc in slices.items()
    }
    for ik in range(nk):
        hk = 0.5 * (h_spin[ik] + h_spin[ik].conj().T) - float(efermi) * eye
        ww, cc = np.linalg.eigh(hk)
        evals[ik] = np.real(ww)
        evecs[ik] = cc
        for site, slc in slices.items():
            idx = _spinor_atom_indices(slc, nwan)
            coeffs[int(site)][ik] = cc[idx, :]

    h0 = np.mean(h_spin, axis=0)
    d_ops = {}
    for site, slc in slices.items():
        idx = _spinor_atom_indices(slc, nwan)
        hloc = h0[np.ix_(idx, idx)]
        bx, by, bz = _extract_exchange_fields_from_spinor_block(hloc)
        d_ops[int(site)] = {
            "x": _compose_spinor_operator("x", bx),
            "y": _compose_spinor_operator("y", by),
            "z": _compose_spinor_operator("z", bz),
        }
    return {"evals": evals, "evecs": evecs, "coeffs": coeffs, "d_ops": d_ops, "efermi": efermi}


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
        global_to_local_d = {g_idx: idx for idx, g_idx in enumerate(d_idx)}
        local_idxs = {}
        for site, slc in slices.items():
            idx_local = np.array([global_to_local_d[x] for x in range(slc.start, slc.stop)], dtype=np.int64)
            local_idxs[int(site)] = np.concatenate([idx_local, idx_local + nd])

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


    nk = int(len(kpts))
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


def _decompose_tb2j_pair(val, val_m, *, collinear_override=False):
    """Decompose TB2J A tensors into scalar J, Gamma, DMI, and debug Aab tensor."""
    val = np.asarray(val, dtype=np.complex128)
    val_m = np.asarray(val_m, dtype=np.complex128)

    m_raw = np.imag(val[1:4, 1:4] + val_m[1:4, 1:4]).astype(np.float64, copy=False)
    ms = 0.5 * (m_raw + m_raw.T)
    if collinear_override:
        ms = np.array(ms, dtype=np.float64, copy=True)
        ms[2, 2] = 0.5 * (ms[0, 0] + ms[1, 1])

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
    if dynamic_soc:
        # collinear_override must be enabled for dynamic collinear downfolding base
        nwan = int(hk_up.shape[1])
    else:
        nwan = int(h_spin.shape[1] // 2)

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
        ni = slices[li].stop - slices[li].start
        nj = slices[lj].stop - slices[lj].start
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
            iu_up[ip, :ni] = np.arange(si.start, si.stop)
            iu_dn[ip, :ni] = np.arange(si.start, si.stop) + nwan
            jv_up[ip, :nj] = np.arange(sj.start, sj.stop)
            jv_dn[ip, :nj] = np.arange(sj.start, sj.stop) + nwan

    acc_a = np.zeros((n_pairs, n_r, 4, 4), dtype=np.complex128)
    r_all = list(acc_r) + [(-r[0], -r[1], -r[2]) for r in acc_r]

    if dynamic_soc:
        nd = int(d_idx.size)
        iu_up_local = np.zeros((n_pairs, max_ni), dtype=np.int64)
        iu_dn_local = np.zeros((n_pairs, max_ni), dtype=np.int64)
        jv_up_local = np.zeros((n_pairs, max_nj), dtype=np.int64)
        jv_dn_local = np.zeros((n_pairs, max_nj), dtype=np.int64)

        global_to_local_d = {g_idx: idx for idx, g_idx in enumerate(d_idx)}
        for ip, (li, lj) in enumerate(acc_pairs):
            si = slices[li]
            sj = slices[lj]
            ni = si.stop - si.start
            nj = sj.stop - sj.start
            iu_up_local[ip, :ni] = np.array([global_to_local_d[x] for x in range(si.start, si.stop)])
            iu_dn_local[ip, :ni] = iu_up_local[ip, :ni] + nd
            jv_up_local[ip, :nj] = np.array([global_to_local_d[x] for x in range(sj.start, sj.stop)])
            jv_dn_local[ip, :nj] = jv_up_local[ip, :nj] + nd

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
        basic.create_dataset("lambda_te_ev", data=np.array(float(args.lambda_te), dtype=np.float64))
        basic.create_dataset("tensor_axes", data=np.asarray(axes, dtype=object), dtype=str_dt)
        put_string(basic, "unit", "meV")
        put_string(basic, "hr_unit", args.hr_unit)
        put_string(basic, "integrator", args.integrator)
        put_string(basic, "kernel", args.kernel)
        put_string(basic, "p_order", args.p_order)
        put_string(basic, "d_order", args.d_order)
        put_string(basic, "soc", args.soc)
        put_string(basic, "soc_element", args.soc_element)
        put_string(basic, "win_path", getattr(args, "_soc_win_path", "") or "")
        soc_entries = getattr(args, "_soc_entries", []) or []
        basic.create_dataset(
            "soc_entries",
            data=np.asarray(
                [f"{e['element']}:{e['orbital']}:{float(e['lambda_ev']):.16g}" for e in soc_entries],
                dtype=object,
            ),
            dtype=str_dt,
        )
        soc_p_groups = []
        soc_d_groups = []
        for entry in soc_entries:
            if entry["orbital"] == "p":
                soc_p_groups.extend(entry["groups"])
            elif entry["orbital"] == "d":
                soc_d_groups.extend(entry["groups"])
        basic.create_dataset("soc_p_groups", data=np.asarray(soc_p_groups, dtype=np.int64).reshape(-1, 3))
        basic.create_dataset("soc_d_groups", data=np.asarray(soc_d_groups, dtype=np.int64).reshape(-1, 5))
        put_string(basic, "command", " ".join(sys.argv))
        basic.create_dataset("empoints", data=np.array(int(args.empoints), dtype=np.int64))
        basic.create_dataset("nproc", data=np.array(int(args.nproc), dtype=np.int64))
        basic.create_dataset("nk", data=np.array(int(nk), dtype=np.int64))
        basic.create_dataset("nE", data=np.array(int(nE), dtype=np.int64))
        basic.create_dataset("elapsed_s", data=np.array(float(elapsed), dtype=np.float64))
        basic.create_dataset("n_chunks", data=np.array(int(exe_info.get("n_chunks", 1)), dtype=np.int64))
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
                h5.create_dataset("J_iso_r", data=np.asarray(extra["jiso_tb2j"], dtype=np.float64))
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


def _write_tensor_text(path, labels, pair_meta, tensor, orbits, axes, kernel, extra=None):
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
        f.write("# Exchange Tensor J^{ab}(R) Results - EPR spinor H(k)\n")
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


def run(args):
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
        hsoc, soc_summary = _soc_matrix_for_groups(
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

    print(
        f"[J-epr-tensor] computing spinor tensor: kernel={args.kernel} nBond={len(pair_meta)} "
        f"nE={len(energy_mesh)} axes={axes} nproc={args.nproc}",
        flush=True,
    )
    t0 = time.time()
    extra = None
    collinear_override_val = getattr(args, "collinear_override", False)
    if args.kernel == "direct":
        tensor, trace_acc, exe_info = _compute_tensor_direct(
            h_spin,
            slices,
            pair_meta,
            kpts,
            energy_mesh,
            args.efermi,
            axes,
            nproc=args.nproc,
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
    else:
        tensor, trace_acc, extra, exe_info = _compute_tensor_tb2j(
            h_spin,
            slices,
            pair_meta,
            kpts,
            energy_mesh,
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
