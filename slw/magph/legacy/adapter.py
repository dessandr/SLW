import os
import re
import json
import multiprocessing as mp
import time
import tempfile
from typing import Dict, Tuple, Any

import h5py
import numpy as np

from slw.core.constants import BOHR_TO_ANG
from slw.core.qe2pert_ws import init_rvec_images, set_wigner_seitz_cell, triangular_pair_index
from slw.epc.check_qe2pert_epr_dispersion import (
    RYD2MEV,
    _apply_loto_override,
    _dyn_mat_longrange,
    _phase,
    _polar_onsite_correction,
    _read_h5_meta,
)


def _parse_qe_fc_force_constants(fc_path: str, expected_nat: int):
    """Read the real-space IFC tensor from a QE q2r text file."""
    if not os.path.exists(fc_path):
        raise FileNotFoundError(f"QE force-constant file not found: {fc_path}")
    with open(fc_path, "r") as handle:
        lines = handle.readlines()

    candidates = []
    for iline, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            mesh = tuple(int(x) for x in parts)
        except ValueError:
            continue
        if any(x <= 0 for x in mesh):
            continue
        ngrid = int(np.prod(mesh))
        required = 9 * int(expected_nat) ** 2 * (ngrid + 1)
        if len(lines) - iline - 1 < required:
            continue
        candidates.append((iline, mesh))

    for mesh_line, mesh in candidates:
        nr1, nr2, nr3 = mesh
        frc = np.zeros((nr1, nr2, nr3, 3, 3, expected_nat, expected_nat), dtype=np.float64)
        cursor = mesh_line + 1
        valid = True
        try:
            for ipol in range(3):
                for jpol in range(3):
                    for ia in range(expected_nat):
                        for ja in range(expected_nat):
                            header = tuple(int(x) for x in lines[cursor].split())
                            cursor += 1
                            if header != (ipol + 1, jpol + 1, ia + 1, ja + 1):
                                valid = False
                                break
                            for _ in range(nr1 * nr2 * nr3):
                                row = lines[cursor].split()
                                cursor += 1
                                if len(row) < 4:
                                    valid = False
                                    break
                                r = tuple(int(x) - 1 for x in row[:3])
                                if not (0 <= r[0] < nr1 and 0 <= r[1] < nr2 and 0 <= r[2] < nr3):
                                    valid = False
                                    break
                                frc[r + (ipol, jpol, ia, ja)] = float(row[3].replace("D", "E"))
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
        except (IndexError, ValueError):
            valid = False
        if valid:
            return mesh, frc
    raise ValueError(f"Could not locate a valid QE q2r IFC section in {fc_path}")


def _qe_fc_mirror(frc):
    inv = [(-np.arange(n, dtype=np.int64)) % n for n in frc.shape[:3]]
    return frc[np.ix_(inv[0], inv[1], inv[2])].transpose(0, 1, 2, 4, 3, 6, 5)


def _apply_qe_fc_asr(frc, zstar, mode: str):
    """Apply QE matdyn's no/simple/crystal ASR to IFCs and Born charges."""
    mode = str(mode).strip().lower()
    if mode not in {"none", "simple", "crystal"}:
        raise ValueError("phonon_asr must be one of: none, simple, crystal")
    out = np.array(frc, dtype=np.float64, copy=True)
    zeu = np.array(zstar, dtype=np.float64, copy=True)
    if mode == "none":
        return out, zeu

    zeu -= np.mean(zeu, axis=0, keepdims=True)
    if mode == "simple":
        row_sum = np.sum(out, axis=(0, 1, 2, 6))
        for ia in range(out.shape[5]):
            out[0, 0, 0, :, :, ia, ia] -= row_sum[:, :, ia]
        return out, zeu

    # This is the orthogonal projection used by QE set_asr(asr='crystal'):
    # first enforce Phi(R,ij,ab)=Phi(-R,ji,ba), then remove the 9*nat
    # translational constraint vectors after projecting them into that
    # permutation-symmetric subspace.
    out = 0.5 * (out + _qe_fc_mirror(out))
    nat = int(out.shape[5])
    constraints = np.empty((9 * nat, out.size), dtype=np.float64)
    iconstraint = 0
    for ipol in range(3):
        for jpol in range(3):
            for ia in range(nat):
                vec = np.zeros_like(out)
                vec[:, :, :, ipol, jpol, ia, :] = 1.0
                vec = 0.5 * (vec + _qe_fc_mirror(vec))
                constraints[iconstraint] = vec.reshape(-1)
                iconstraint += 1

    # Build an orthonormal basis from the small (9*nat)^2 Gram matrix.  The
    # expensive operations over all IFC entries are BLAS matrix products,
    # rather than Python Gram-Schmidt loops.
    gram = constraints @ constraints.T
    evals, evecs = np.linalg.eigh(gram)
    keep = evals > 1.0e-8
    basis = (evecs[:, keep].T @ constraints) / np.sqrt(evals[keep])[:, None]
    flat = out.reshape(-1)
    flat -= basis.T @ (basis @ flat)
    return out, zeu


def _qe_fc_blocks(frc, meta):
    """Convert a QE supercell IFC tensor to qe2pert WS blocks."""
    mesh = tuple(int(x) for x in frc.shape[:3])
    images = init_rvec_images(mesh, meta["at"])
    tau = np.asarray(meta["tau"], dtype=np.float64)
    nat = int(meta["nat"])
    mesh_arr = np.asarray(mesh, dtype=np.int64)
    blocks = []
    for ja in range(nat):
        for ia in range(ja + 1):
            ws = set_wigner_seitz_cell(images, meta["at"], tau[ia], tau[ja])
            mats = np.empty((ws.nr, 3, 3), dtype=np.complex128)
            for ir, (rvec, ndeg) in enumerate(zip(ws.vectors, ws.ndeg)):
                # QE frc_blk uses exp(-iq.R); the magph WS convention uses
                # exp(+iq.R), hence the minus sign in the modulo lookup.
                r = np.mod(-np.asarray(rvec, dtype=np.int64), mesh_arr)
                mats[ir] = frc[r[0], r[1], r[2], :, :, ia, ja] / float(ndeg)
            blocks.append((3 * ia, 3 * (ia + 1), 3 * ja, 3 * (ja + 1), ws.vectors, mats))
    return blocks


def _qe_fc_from_epr_blocks(blocks, meta):
    """Fold qe2pert WS IFC blocks back onto their native q-grid supercell."""
    mesh = tuple(int(x) for x in meta["qc_dim"])
    nat = int(meta["nat"])
    shape = mesh + (3, 3, nat, nat)
    accum = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int32)
    mesh_arr = np.asarray(mesh, dtype=np.int64)
    images = init_rvec_images(mesh, meta["at"])

    for si0, _si1, sj0, _sj1, rvecs, mats in blocks:
        ia = int(si0) // 3
        ja = int(sj0) // 3
        ws = set_wigner_seitz_cell(
            images,
            meta["at"],
            meta["tau"][ia],
            meta["tau"][ja],
        )
        if len(rvecs) != ws.nr or not np.array_equal(np.asarray(rvecs), ws.vectors):
            raise ValueError(f"EPR WS-vector mismatch while folding IFC block ({ia}, {ja})")
        if np.max(np.abs(np.asarray(mats).imag), initial=0.0) > 1.0e-12:
            raise ValueError("EPR force constants must be real before crystal ASR")
        for rvec, ndeg, mat in zip(ws.vectors, ws.ndeg, mats):
            r = np.mod(-np.asarray(rvec, dtype=np.int64), mesh_arr)
            value = np.asarray(mat.real, dtype=np.float64) * float(ndeg)
            idx = (r[0], r[1], r[2], slice(None), slice(None), ia, ja)
            accum[idx] += value
            counts[idx] += 1

            rm = np.mod(-r, mesh_arr)
            mirror_idx = (rm[0], rm[1], rm[2], slice(None), slice(None), ja, ia)
            accum[mirror_idx] += value.T
            counts[mirror_idx] += 1

    if np.any(counts == 0):
        missing = int(np.count_nonzero(counts == 0))
        raise ValueError(f"Could not reconstruct dense EPR IFC tensor: {missing} entries are missing")
    return accum / counts


def _as_str_list(arr):
    return [str(x) for x in np.array(arr).tolist()]


def _decode_str_array(arr):
    out = []
    for x in np.array(arr).tolist():
        if isinstance(x, bytes):
            out.append(x.decode())
        else:
            out.append(str(x))
    return out


def _decode_scalar_text(value, *, name: str) -> str:
    """Decode one scalar string without permitting object-array deserialization."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar text, got shape {array.shape}")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="strict")
    return str(item)


def _load_npz_schema_array(payload, key: str, source: str) -> np.ndarray:
    """Read one documented NPZ field while clearly rejecting pickle payloads."""
    try:
        array = np.array(payload[key], copy=True)
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                f"Refusing unsafe legacy object array for NPZ key {key!r} in "
                f"{source}. Regenerate it with a numeric, fixed-width Unicode, "
                "or fixed-width bytes dtype."
            ) from exc
        raise
    if array.dtype.hasobject:
        # This is normally caught by NumPy with allow_pickle=False, but retain
        # an explicit invariant in case its archive-loading behavior changes.
        raise ValueError(
            f"Refusing unsafe legacy object array for NPZ key {key!r} in {source}"
        )
    return array


def _canonical_unit_token(value: str) -> str:
    """Normalize common spellings only for detecting conflicting metadata."""
    normalized = str(value).strip().lower().replace(" ", "")
    normalized = (
        normalized.replace("ångström", "angstrom")
        .replace("Å", "angstrom")
        .replace("å", "angstrom")
        .replace("angstroms", "angstrom")
    )
    if normalized in {
        "mev/a",
        "mev/ang",
        "mev/angstrom",
        "mev*a^-1",
        "mev*ang^-1",
        "mev*angstrom^-1",
        "mevang^-1",
        "mevangstrom^-1",
    }:
        return "mev/angstrom"
    return normalized


def _resolve_djr_h5_units(h5, djr_key: str) -> tuple[str, bool, str]:
    """Resolve explicit dJ unit metadata and report its exact HDF5 source."""
    candidates: list[tuple[str, str]] = []

    def append(source: str, raw) -> None:
        candidates.append(
            (source, _decode_scalar_text(raw, name=f"dJ unit metadata {source}"))
        )

    for key in ("basic_data/unit", "basic_data/units", "unit", "units"):
        if key in h5 and isinstance(h5[key], h5py.Dataset):
            append(key, h5[key][()])
    for attr in ("unit", "units"):
        if attr in h5.attrs:
            append(f"@{attr}", h5.attrs[attr])

    group = h5[djr_key]
    for attr in ("unit", "units"):
        if attr in group.attrs:
            append(f"{djr_key}@{attr}", group.attrs[attr])
    for name in sorted(group.keys()):
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset):
            continue
        for attr in ("unit", "units"):
            if attr in dataset.attrs:
                append(f"{djr_key}/{name}@{attr}", dataset.attrs[attr])

    if not candidates:
        return "meV/A", False, "missing"

    tokens = {_canonical_unit_token(value) for _, value in candidates}
    if len(tokens) != 1:
        details = ", ".join(
            f"{source}={value!r}" for source, value in candidates[:8]
        )
        if len(candidates) > 8:
            details += f", ... ({len(candidates)} sources total)"
        raise ValueError(f"Conflicting dJ unit metadata in HDF5: {details}")
    source, value = candidates[0]
    return value, True, source


def _require_matching_nonempty_lengths(source: str, **arrays) -> int:
    """Refuse partial exchange payloads instead of truncating them silently."""
    lengths = {name: len(array) for name, array in arrays.items()}
    if not lengths or next(iter(lengths.values())) < 1:
        raise ValueError(f"Empty J payload in {source}: lengths={lengths}")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"J payload length mismatch in {source}: {lengths}")
    return next(iter(lengths.values()))


def _label_to_atom_index(label: str) -> int:
    """
    Convert labels like 'Mn1', 'Te2' -> zero-based atom index (0, 1, ...).
    Falls back to -1 when no trailing integer exists.
    """
    m = re.search(r"(\d+)$", str(label))
    if not m:
        return -1
    return int(m.group(1)) - 1


def load_djdu_npz(npz_path: str) -> Dict[str, Any]:
    """
    Load SLW compute_dJ_bulk NPZ payload.
    Required keys:
      - values_rp : (n_rp, n_targets, n_axes, n_bonds)
      - rp_list   : (n_rp, 3)
      - targets   : (n_targets,)
      - axes      : (n_axes,)
      - pair_R    : (n_bonds, 5) as [i, j, R1, R2, R3]
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"dJ/du npz not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        keys = set(payload.files)
        has_multi = {"values_rp", "rp_list"} <= keys
        has_single = {"values", "rp_idx"} <= keys
        if not has_multi and not has_single:
            raise KeyError(
                f"Missing Rp payload in {npz_path}: need "
                "(values_rp,rp_list) or (values,rp_idx)."
            )
        for key in ("targets", "axes", "pair_R"):
            if key not in keys:
                raise KeyError(f"Missing key in {npz_path}: {key}")

        if has_multi:
            values_rp = np.asarray(
                _load_npz_schema_array(payload, "values_rp", npz_path),
                dtype=np.float64,
            )
            rp_list = np.asarray(
                _load_npz_schema_array(payload, "rp_list", npz_path),
                dtype=np.int32,
            )
        else:
            # single-rp compatibility: values -> values_rp[1,...],
            # rp_idx -> rp_list[1,3]
            values_rp = np.asarray(
                _load_npz_schema_array(payload, "values", npz_path),
                dtype=np.float64,
            )[np.newaxis, ...]
            rp_list = np.asarray(
                _load_npz_schema_array(payload, "rp_idx", npz_path),
                dtype=np.int32,
            )[np.newaxis, ...]

        targets = _decode_str_array(
            _load_npz_schema_array(payload, "targets", npz_path)
        )
        axes = _decode_str_array(_load_npz_schema_array(payload, "axes", npz_path))
        pair_R = np.asarray(
            _load_npz_schema_array(payload, "pair_R", npz_path), dtype=np.int32
        )
        if "units" in keys:
            units = _decode_scalar_text(
                _load_npz_schema_array(payload, "units", npz_path),
                name="NPZ dJ units",
            )
            units_explicit = True
            units_source = "units"
        else:
            units = "meV/A"
            units_explicit = False
            units_source = "missing"

    out = {
        "values_rp": values_rp,
        "rp_list": rp_list,
        "targets": targets,
        "axes": axes,
        "pair_R": pair_R,
        "units": units,
        "units_explicit": units_explicit,
        "units_source": units_source,
    }
    return out


def load_djr_h5(h5_path: str) -> Dict[str, Any]:
    """
    Load direct real-space dJ(R,Rp) HDF5 payload from compute_dJ_epr_kspace.

    Returned schema matches load_djdu_npz enough for build_legacy_djdu_tuple:
      - values_rp : (n_rp, n_targets, 3, n_bonds)
      - rp_list   : (n_rp, 3)
      - targets   : human labels
      - target_indices : 0-based global atom indices
      - axes      : ["x", "y", "z"]
      - pair_R    : (n_bonds, 5) [mag_i_atom, mag_j_atom, R1, R2, R3]
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"dJr HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        required = ["bonds/mag_i_atom", "bonds/mag_j_atom", "bonds/R", "displacements/target_atom", "displacements/Rp"]
        missing = [k for k in required if k not in h5]
        if missing:
            raise KeyError(f"Missing dJr HDF5 datasets/groups in {h5_path}: {missing}")

        # Look for possible dJ keys:
        # 1. dJ_iso_r (isotropic only, shape (n_rp, 3))
        # 2. dJ_r     (isotropic legacy, shape (n_rp, 3))
        # 3. dJ_tensor_r (full 3x3 tensor, shape (n_rp, 3, 3, 3))
        djr_key = None
        for k in ("dJ_iso_r", "dJ_r", "dJ_tensor_r"):
            if k in h5:
                djr_key = k
                break
        if djr_key is None:
            raise KeyError(
                f"Missing dJr HDF5 datasets/groups in {h5_path}. "
                "Looked for: 'dJ_iso_r', 'dJ_r', or 'dJ_tensor_r'"
            )

        target_indices = np.asarray(h5["displacements/target_atom"], dtype=np.int32).reshape(-1)
        target_labels = (
            _decode_str_array(h5["displacements/target_label"][:])
            if "displacements/target_label" in h5
            else [f"Atom{i + 1}" for i in target_indices]
        )
        targets = [f"{lab}{int(idx) + 1}" if not str(lab).endswith(str(int(idx) + 1)) else str(lab) for idx, lab in zip(target_indices, target_labels)]
        axes = (
            [a.lower() for a in _decode_str_array(h5["displacements/axes"][:])]
            if "displacements/axes" in h5
            else ["x", "y", "z"]
        )
        if axes != ["x", "y", "z"]:
            # dJ_r_m*_b* stores columns in x,y,z order by construction.
            axes = ["x", "y", "z"]

        rp_list = np.asarray(h5["displacements/Rp"], dtype=np.int32)
        gi = np.asarray(h5["bonds/mag_i_atom"], dtype=np.int32).reshape(-1)
        gj = np.asarray(h5["bonds/mag_j_atom"], dtype=np.int32).reshape(-1)
        R = np.asarray(h5["bonds/R"], dtype=np.int32).reshape(-1, 3)
        pair_R = np.column_stack([gi, gj, R]).astype(np.int32, copy=False)

        n_rp = int(rp_list.shape[0])
        n_targets = int(target_indices.shape[0])
        n_bonds = int(pair_R.shape[0])
        values_rp = np.zeros((n_rp, n_targets, 3, n_bonds), dtype=np.float64)
        grp = h5[djr_key]
        for it, atom_idx in enumerate(target_indices):
            for ib in range(n_bonds):
                name_legacy = f"dJ_r_m{int(atom_idx) + 1}_b{ib + 1}"
                name_tensor = f"m{int(atom_idx) + 1}_b{ib + 1}"

                if name_legacy in grp:
                    name = name_legacy
                elif name_tensor in grp:
                    name = name_tensor
                else:
                    raise KeyError(f"Missing /{djr_key}/[{name_legacy} or {name_tensor}] in {h5_path}")

                arr = np.asarray(grp[name], dtype=np.float64)

                # If we are loading the full tensor (n_rp, n_disp, 3, 3), extract the isotropic component:
                # Tr(J) / 3 for each displacement axis
                if djr_key == "dJ_tensor_r":
                    if arr.ndim != 4 or arr.shape[2] != 3 or arr.shape[3] != 3:
                        raise ValueError(
                            f"/{djr_key}/{name} shape={arr.shape}, expected full tensor shape (n_rp, n_disp, 3, 3)"
                        )
                    # Trace over the last two dimensions (spin space row/col), divided by 3:
                    # np.trace(arr, axis1=2, axis2=3) -> (n_rp, n_disp)
                    arr_iso = np.trace(arr, axis1=2, axis2=3) / 3.0
                else:
                    arr_iso = arr

                if arr_iso.shape != (n_rp, 3):
                    raise ValueError(f"/{djr_key}/{name} resolved isotropic shape={arr_iso.shape}, expected={(n_rp, 3)}")
                values_rp[:, it, :, ib] = arr_iso

        units, units_explicit, units_source = _resolve_djr_h5_units(h5, djr_key)

    return {
        "values_rp": values_rp,
        "rp_list": rp_list,
        "targets": targets,
        "target_indices": target_indices,
        "axes": axes,
        "pair_R": pair_R,
        "units": units,
        "units_explicit": units_explicit,
        "units_source": units_source,
        "source": h5_path,
    }


def select_rp_slice(dj_payload: Dict[str, Any], rp_idx: Tuple[int, int, int]) -> np.ndarray:
    """
    Select values[target, axis, bond] for a specific Rp index.
    """
    rp_arr = np.array(rp_idx, dtype=np.int32)
    rp_list = dj_payload["rp_list"]
    hits = np.where(np.all(rp_list == rp_arr[None, :], axis=1))[0]
    if hits.size == 0:
        raise ValueError(f"Rp index {tuple(rp_idx)} not found in rp_list. Available count={len(rp_list)}")
    return dj_payload["values_rp"][int(hits[0])]


def _axis_index_map(axes):
    amap = {}
    for i, ax in enumerate(axes):
        amap[str(ax).lower()] = i
    return amap


def build_legacy_djdu_tuple(dj_payload: Dict[str, Any], rp_idx: Tuple[int, int, int] = (0, 0, 0)):
    """
    Build legacy tuple:
      (moved_atom_idx, i_idx, j_idx, R, grad_J_vec)
    expected by historical magph kernels.

    grad_J_vec is constructed from x/y/z components in the selected Rp slice.
    """
    vals = select_rp_slice(dj_payload, rp_idx=rp_idx)  # (n_targets, n_axes, n_bonds)
    targets = dj_payload["targets"]
    axes = [a.lower() for a in dj_payload["axes"]]
    pair_R = dj_payload["pair_R"]  # [i,j,R1,R2,R3]

    amap = _axis_index_map(axes)
    ix = amap.get("x", None)
    iy = amap.get("y", None)
    iz = amap.get("z", None)

    moved_atom_idx = []
    i_idx = []
    j_idx = []
    R = []
    grad = []

    n_targets = len(targets)
    n_bonds = pair_R.shape[0]
    for it in range(n_targets):
        if "target_indices" in dj_payload:
            atom_idx = int(np.asarray(dj_payload["target_indices"], dtype=np.int32)[it])
        else:
            atom_label = targets[it]
            atom_idx = _label_to_atom_index(atom_label)
        for ib in range(n_bonds):
            moved_atom_idx.append(atom_idx)
            i_idx.append(int(pair_R[ib, 0]))
            j_idx.append(int(pair_R[ib, 1]))
            R.append([int(pair_R[ib, 2]), int(pair_R[ib, 3]), int(pair_R[ib, 4])])

            gx = float(vals[it, ix, ib]) if ix is not None else 0.0
            gy = float(vals[it, iy, ib]) if iy is not None else 0.0
            gz = float(vals[it, iz, ib]) if iz is not None else 0.0
            grad.append([gx, gy, gz])

    return (
        np.array(moved_atom_idx, dtype=np.int32),
        np.array(i_idx, dtype=np.int32),
        np.array(j_idx, dtype=np.int32),
        np.array(R, dtype=np.int32),
        np.array(grad, dtype=np.float64),
    )


_PHONON_CACHE_CORE_KEYS = (
    "q_mesh_flat_frac",
    "q_mesh_flat_cart",
    "ph_en_flat",
    "ph_vec_flat",
)
_PHONON_CACHE_METADATA_KEYS = (
    "phonon_cache_schema_version",
    "q_mesh_shape",
    "q_mesh_shift_grid",
    "q_mesh_origin_frac",
    "q_mesh_sampling_convention",
    "atom_mass_electron",
    "atom_frac",
    "atom_labels",
    "lattice_ang",
    "phonon_vector_convention",
    "fourier_phase_convention",
    "mass_unit",
    "energy_unit",
    "q_cart_unit",
    "epr_source",
    "phonon_source",
    "phonon_asr",
    "phonon_loto",
)


def load_phonon_cache(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load phonon cache generated by legacy magph solver or compatible preprocessing.
    Required keys:
      - q_mesh_flat_frac
      - q_mesh_flat_cart
      - ph_en_flat
      - ph_vec_flat
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"phonon cache not found: {npz_path}")
    # Never deserialize arbitrary object arrays from an input cache.  Read
    # only the documented numeric/string schema so unrelated large extras do
    # not get materialized either.
    with np.load(npz_path, allow_pickle=False) as data:
        keys = list(data.files)
        selected = set(_PHONON_CACHE_CORE_KEYS) | set(_PHONON_CACHE_METADATA_KEYS)
        selected.add("q_mesh_flat")
        loaded = {
            key: np.array(data[key], copy=True)
            for key in keys
            if key in selected
        }

    has_new = all(
        key in loaded
        for key in _PHONON_CACHE_CORE_KEYS
    )
    if has_new:
        # Preserve optional provenance/geometry fields.  Older callers use
        # only the four required arrays, while rotational observables need the
        # mass metric and must not silently guess it from chemical symbols.
        return loaded

    # Legacy compatibility:
    # - q_mesh_flat (historically cartesian in older scripts)
    # - ph_en_flat
    # - ph_vec_flat
    has_old = all(key in loaded for key in ["q_mesh_flat", "ph_en_flat", "ph_vec_flat"])
    if has_old:
        loaded["q_mesh_flat_frac"] = np.array(loaded["q_mesh_flat"], copy=True)
        loaded["q_mesh_flat_cart"] = np.array(loaded["q_mesh_flat"], copy=True)
        return loaded

    raise KeyError(
        f"Unsupported phonon cache schema in {npz_path}. "
        f"Found keys={keys}"
    )


def _validated_qmesh(qmesh) -> np.ndarray:
    """Return a three-component positive integer q mesh."""
    raw = np.asarray(qmesh)
    if raw.shape != (3,):
        raise ValueError(f"qmesh must contain exactly three values, got shape {raw.shape}")
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("qmesh must contain three positive integers") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("qmesh must contain only finite values")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"qmesh must contain integers, got {numeric.tolist()}")
    if np.any(rounded <= 0.0):
        raise ValueError(f"qmesh values must be positive, got {numeric.tolist()}")
    if np.any(rounded > np.iinfo(np.int32).max):
        raise ValueError("qmesh values exceed the cache metadata integer range")
    return rounded.astype(np.int64)


def _canonical_qmesh_shift_grid(qmesh_shift_grid=None) -> np.ndarray:
    """Canonicalize a q-grid shift in grid-index units modulo integer shifts."""
    if qmesh_shift_grid is None:
        return np.zeros(3, dtype=np.float64)
    raw = np.asarray(qmesh_shift_grid, dtype=np.float64)
    if raw.shape != (3,):
        raise ValueError(
            "qmesh_shift_grid must contain exactly three values, "
            f"got shape {raw.shape}"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("qmesh_shift_grid must contain only finite values")
    return np.mod(raw, 1.0)


def _full_q_mesh(qmesh, qmesh_shift_grid=None):
    """Build a C-ordered uniform fractional q grid, with the third axis fastest."""
    mesh = _validated_qmesh(qmesh)
    shift = _canonical_qmesh_shift_grid(qmesh_shift_grid)
    grid_indices = np.indices(tuple(mesh), dtype=np.float64).reshape(3, -1).T
    return np.mod((grid_indices + shift[None, :]) / mesh[None, :], 1.0)


def _diagonalize_phonon_dyn_chunk(task):
    start, dyn_chunk, sqrt_mass, ryd2mev = task
    evals, evecs = np.linalg.eigh(dyn_chunk)
    freqs = np.sqrt(np.abs(np.real(evals)))
    freqs[np.real(evals) <= 0.0] *= -1.0
    nq, nm = freqs.shape
    nat = int(sqrt_mass.shape[0])
    ph_vec = evecs.transpose(0, 2, 1).reshape(nq, nm, nat, 3) / sqrt_mass[None, None, :, :]
    return int(start), freqs * float(ryd2mev), ph_vec


def _diagonalize_phonon_dyn(dyn, sqrt_mass, *, nproc=1):
    nq = int(dyn.shape[0])
    nm = int(dyn.shape[1])
    nat = int(sqrt_mass.shape[0])
    nproc = max(1, min(int(nproc), nq))
    if nproc == 1:
        _start, ph_en, ph_vec = _diagonalize_phonon_dyn_chunk((0, dyn, sqrt_mass, RYD2MEV))
        return ph_en, ph_vec

    edges = np.linspace(0, nq, nproc + 1, dtype=np.int64)
    tasks = [
        (int(edges[i]), dyn[int(edges[i]): int(edges[i + 1])], sqrt_mass, RYD2MEV)
        for i in range(nproc)
        if int(edges[i + 1]) > int(edges[i])
    ]
    ph_en = np.zeros((nq, nm), dtype=np.float64)
    ph_vec = np.zeros((nq, nm, nat, 3), dtype=np.complex128)
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(processes=len(tasks)) as pool:
        for start, en_chunk, vec_chunk in pool.map(_diagonalize_phonon_dyn_chunk, tasks):
            stop = int(start) + int(en_chunk.shape[0])
            ph_en[int(start):stop] = en_chunk
            ph_vec[int(start):stop] = vec_chunk
    return ph_en, ph_vec


_PHONON_EPR_STATE = {}


def _preload_phonon_ifc_blocks(epr_path, meta):
    at = np.asarray(meta["at"], dtype=np.float64)
    tau = np.asarray(meta["tau"], dtype=np.float64)
    nat = int(meta["nat"])
    images = init_rvec_images(meta["qc_dim"], at)
    blocks = []

    with h5py.File(epr_path, "r") as h5:
        for ja in range(1, nat + 1):
            for ia in range(1, ja + 1):
                tri = triangular_pair_index(ia, ja)
                key = f"force_constant/ifc{tri}"
                if key not in h5:
                    raise KeyError(f"Missing EPR IFC dataset: {key}")
                ws = set_wigner_seitz_cell(images, at, tau[ia - 1], tau[ja - 1])
                ifc = np.asarray(h5[key]).astype(np.complex128, copy=False)
                if ifc.shape[0] != ws.nr:
                    raise ValueError(f"IFC length mismatch pair={ia},{ja}: h5={ifc.shape[0]} ws={ws.nr}")
                blocks.append(
                    (
                        3 * (ia - 1),
                        3 * ia,
                        3 * (ja - 1),
                        3 * ja,
                        np.asarray(ws.vectors, dtype=np.float64),
                        # As with zstar, h5py moves the Fortran real-space
                        # index to the front but preserves both Cartesian
                        # axes: (i, j, ir) -> (ir, i, j).
                        np.ascontiguousarray(ifc),
                    )
                )
    return blocks


def _set_phonon_epr_state(meta, blocks):
    at = np.asarray(meta["at"], dtype=np.float64)
    nat = int(meta["nat"])
    mass = np.asarray(meta["mass"], dtype=np.float64).reshape(-1)
    if mass.shape != (nat,):
        raise ValueError(f"mass length mismatch: mass={mass.shape} nat={nat}")
    if np.any(mass <= 0.0):
        raise ValueError("All atomic masses in EPR basic_data/mass must be positive.")

    alat_ang = float(meta["alat"]) * BOHR_TO_ANG
    lattice_ang = at.T * alat_ang
    recip_ang = 2.0 * np.pi * np.linalg.inv(lattice_ang).T
    mode_mass = np.repeat(mass, 3)
    mass_factor = 1.0 / np.sqrt(mode_mass[:, None] * mode_mass[None, :])

    _PHONON_EPR_STATE.clear()
    _PHONON_EPR_STATE.update(
        {
            "meta": meta,
            "blocks": blocks,
            "nat": nat,
            "nm": 3 * nat,
            "recip_ang": recip_ang,
            "mass_factor": mass_factor,
            "sqrt_mass": np.sqrt(mass)[:, None],
            "polar_onsite": _polar_onsite_correction(meta) if meta["lpolar"] else None,
        }
    )


def _build_phonon_cache_from_epr_qchunk(task):
    start, qpts = task
    qpts = np.asarray(qpts, dtype=np.float64).reshape(-1, 3)
    if not _PHONON_EPR_STATE:
        raise RuntimeError("phonon EPR worker state is not initialized")
    meta = _PHONON_EPR_STATE["meta"]
    nat = int(_PHONON_EPR_STATE["nat"])
    nm = int(_PHONON_EPR_STATE["nm"])
    q_cart = qpts @ _PHONON_EPR_STATE["recip_ang"]
    dyn = np.zeros((len(qpts), nm, nm), dtype=np.complex128)

    for si0, si1, sj0, sj1, rvecs, mats in _PHONON_EPR_STATE["blocks"]:
        vals = np.einsum("qr,rij->qij", _phase(qpts, rvecs), mats, optimize=True)
        dyn[:, si0:si1, sj0:sj1] = vals
        if si0 != sj0:
            dyn[:, sj0:sj1, si0:si1] = np.swapaxes(vals.conj(), 1, 2)

    if meta["lpolar"]:
        for iq, qpt in enumerate(qpts):
            dyn_lr = _dyn_mat_longrange(meta, qpt, _PHONON_EPR_STATE["polar_onsite"])
            n = 0
            for ja in range(1, nat + 1):
                for ia in range(1, ja + 1):
                    si = slice(3 * (ia - 1), 3 * ia)
                    sj = slice(3 * (ja - 1), 3 * ja)
                    dyn[iq, si, sj] += dyn_lr[n]
                    if ia != ja:
                        dyn[iq, sj, si] += dyn_lr[n].conj().T
                    n += 1

    dyn = 0.5 * (dyn + np.swapaxes(dyn.conj(), 1, 2))
    dyn *= _PHONON_EPR_STATE["mass_factor"][None, :, :]

    sqrt_mass = _PHONON_EPR_STATE["sqrt_mass"]
    _chunk_start, ph_en, ph_vec = _diagonalize_phonon_dyn_chunk((start, dyn, sqrt_mass, RYD2MEV))
    return int(start), q_cart, ph_en, ph_vec


def _phonon_cache_payload(
    qpts,
    q_cart,
    ph_en,
    ph_vec,
    *,
    meta,
    epr_path,
    qmesh_shape,
    qmesh_shift_grid,
    phonon_asr,
    phonon_loto,
    phonon_source,
):
    """Attach the geometry and convention needed for physical observables."""
    lattice_ang = (
        np.asarray(meta["at"], dtype=np.float64).T
        * float(meta["alat"])
        * BOHR_TO_ANG
    )
    mesh = np.asarray(qmesh_shape, dtype=np.int32).reshape(-1)
    shift = np.asarray(qmesh_shift_grid, dtype=np.float64).reshape(-1)
    if mesh.size == 3:
        if shift.size != 3:
            raise ValueError("Uniform q meshes require a three-component q-grid shift")
        origin_frac = shift / mesh.astype(np.float64)
        sampling_convention = (
            "uniform fractional grid q=(integer_index+q_mesh_shift_grid)/"
            "q_mesh_shape mod 1; C-order flattening with third axis fastest"
        )
    else:
        if mesh.size != 0 or shift.size != 0:
            raise ValueError(
                "Explicit q-point caches require empty mesh-shape and shift metadata"
            )
        origin_frac = np.asarray([], dtype=np.float64)
        sampling_convention = "explicit fractional q points in stored order"

    return {
        "q_mesh_flat_frac": np.asarray(qpts, dtype=np.float64),
        "q_mesh_flat_cart": np.asarray(q_cart, dtype=np.float64),
        "ph_en_flat": np.asarray(ph_en, dtype=np.float64),
        "ph_vec_flat": np.asarray(ph_vec, dtype=np.complex128),
        "phonon_cache_schema_version": np.asarray(3, dtype=np.int32),
        "q_mesh_shape": mesh,
        "q_mesh_shift_grid": shift,
        "q_mesh_origin_frac": origin_frac,
        "q_mesh_sampling_convention": np.asarray(sampling_convention),
        "atom_mass_electron": np.asarray(meta["mass"], dtype=np.float64),
        "atom_frac": np.mod(np.asarray(meta["tau"], dtype=np.float64), 1.0),
        "lattice_ang": lattice_ang,
        "phonon_vector_convention": np.asarray(
            "cell-gauge physical displacement u=e/sqrt(M); sum_kappa M_kappa|u|^2=1"
        ),
        "fourier_phase_convention": np.asarray("u_(l,kappa) proportional exp(+i2pi q dot R_l)"),
        "mass_unit": np.asarray("electron_mass"),
        "energy_unit": np.asarray("meV"),
        "q_cart_unit": np.asarray("1/angstrom"),
        "epr_source": np.asarray(os.path.realpath(epr_path)),
        "phonon_source": np.asarray(str(phonon_source)),
        "phonon_asr": np.asarray(str(phonon_asr)),
        "phonon_loto": np.asarray(str(phonon_loto)),
    }


def build_phonon_cache_from_epr(
    epr_path: str,
    qmesh=None,
    qpts_frac=None,
    nproc=1,
    verbose=False,
    phonon_fc=None,
    phonon_asr="none",
    phonon_loto="auto",
    qmesh_shift_grid=None,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct a magph phonon cache directly from qe2pert/Perturbo EPR IFC data.

    Returned arrays:
      - q_mesh_flat_frac : fractional q points on qmesh or explicit qpts_frac
      - q_mesh_flat_cart : cartesian q points in 1/Angstrom
      - ph_en_flat       : phonon frequencies in meV, shape (nq, 3*nat)
      - ph_vec_flat      : mass-normalized eigenvectors, shape (nq, 3*nat, nat, 3)
    """
    if qpts_frac is not None and qmesh_shift_grid is not None:
        raise ValueError(
            "qmesh_shift_grid cannot be combined with explicit qpts_frac; "
            "apply any desired shift directly to qpts_frac"
        )
    if qpts_frac is None:
        qpts_raw = None
        qmesh_shift = _canonical_qmesh_shift_grid(qmesh_shift_grid)
        requested_qmesh = None if qmesh is None else _validated_qmesh(qmesh)
    else:
        qpts_raw = np.asarray(qpts_frac, dtype=np.float64)
        if qpts_raw.ndim != 2 or qpts_raw.shape[1:] != (3,) or qpts_raw.shape[0] == 0:
            raise ValueError(
                "qpts_frac must have nonempty shape (nq, 3), "
                f"got {qpts_raw.shape}"
            )
        if not np.all(np.isfinite(qpts_raw)):
            raise ValueError("qpts_frac must contain only finite values")
        qmesh_shift = np.asarray([], dtype=np.float64)
        requested_qmesh = None

    if not os.path.exists(epr_path):
        raise FileNotFoundError(f"EPR HDF5 not found: {epr_path}")
    meta = _apply_loto_override(_read_h5_meta(epr_path), loto_dim=phonon_loto)
    if qpts_raw is None:
        qmesh_array = (
            _validated_qmesh(meta["qc_dim"])
            if requested_qmesh is None
            else requested_qmesh
        )
        qpts = _full_q_mesh(qmesh_array, qmesh_shift)
        qmesh_shape = qmesh_array.astype(np.int32)
    else:
        qpts = np.mod(qpts_raw, 1.0)
        qmesh_shape = np.asarray([], dtype=np.int32)

    nat = int(meta["nat"])
    mass = np.asarray(meta["mass"], dtype=np.float64).reshape(-1)
    if mass.shape != (nat,):
        raise ValueError(f"mass length mismatch: mass={mass.shape} nat={nat}")
    if np.any(mass <= 0.0):
        raise ValueError("All atomic masses in EPR basic_data/mass must be positive.")
    nm = 3 * nat
    nq = int(qpts.shape[0])
    nproc = max(1, min(int(nproc), nq))
    if verbose:
        print(f"[magph-phonon] building EPR phonon cache: nq={nq} nat={nat} nproc={nproc}", flush=True)
    q_cart = np.zeros((nq, 3), dtype=np.float64)
    ph_en = np.zeros((nq, nm), dtype=np.float64)
    ph_vec = np.zeros((nq, nm, nat, 3), dtype=np.complex128)

    t0 = time.perf_counter()
    if phonon_fc is None:
        blocks = _preload_phonon_ifc_blocks(epr_path, meta)
        if str(phonon_asr).strip().lower() == "none":
            phonon_source = "EPR IFC"
        else:
            frc = _qe_fc_from_epr_blocks(blocks, meta)
            frc, zstar = _apply_qe_fc_asr(frc, meta["zstar"], phonon_asr)
            meta = dict(meta)
            meta["zstar"] = zstar
            blocks = _qe_fc_blocks(frc, meta)
            phonon_source = f"EPR IFC ({phonon_asr} ASR)"
    else:
        fc_mesh, frc = _parse_qe_fc_force_constants(phonon_fc, nat)
        if tuple(fc_mesh) != tuple(int(x) for x in meta["qc_dim"]):
            raise ValueError(
                f"QE FC mesh {fc_mesh} does not match EPR qc_dim {tuple(meta['qc_dim'])}"
            )
        frc, zstar = _apply_qe_fc_asr(frc, meta["zstar"], phonon_asr)
        meta = dict(meta)
        meta["zstar"] = zstar
        blocks = _qe_fc_blocks(frc, meta)
        phonon_source = f"QE FC ({phonon_asr} ASR)"
    if verbose:
        n_ifc = sum(int(mats.shape[0]) for *_prefix, mats in blocks)
        print(
            f"[magph-phonon] preloaded {len(blocks)} {phonon_source} blocks ({n_ifc} R blocks) "
            f"in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )
    _set_phonon_epr_state(meta, blocks)

    if nproc == 1:
        t1 = time.perf_counter()
        if verbose:
            print("[magph-phonon] running phonon dynamical matrices serially", flush=True)
        start, q_cart_chunk, ph_en_chunk, ph_vec_chunk = _build_phonon_cache_from_epr_qchunk(
            (0, qpts)
        )
        q_cart[start: start + q_cart_chunk.shape[0]] = q_cart_chunk
        ph_en[start: start + ph_en_chunk.shape[0]] = ph_en_chunk
        ph_vec[start: start + ph_vec_chunk.shape[0]] = ph_vec_chunk
        if verbose:
            print(f"[magph-phonon] serial phonon build finished in {time.perf_counter() - t1:.2f}s", flush=True)
    else:
        if "fork" not in mp.get_all_start_methods():
            nproc = 1
            if verbose:
                print("[magph-phonon] fork is unavailable; falling back to serial phonon build", flush=True)
            start, q_cart_chunk, ph_en_chunk, ph_vec_chunk = _build_phonon_cache_from_epr_qchunk((0, qpts))
            q_cart[start: start + q_cart_chunk.shape[0]] = q_cart_chunk
            ph_en[start: start + ph_en_chunk.shape[0]] = ph_en_chunk
            ph_vec[start: start + ph_vec_chunk.shape[0]] = ph_vec_chunk
            return _phonon_cache_payload(
                qpts,
                q_cart,
                ph_en,
                ph_vec,
                meta=meta,
                epr_path=epr_path,
                qmesh_shape=qmesh_shape,
                qmesh_shift_grid=qmesh_shift,
                phonon_asr=phonon_asr,
                phonon_loto=phonon_loto,
                phonon_source=phonon_source,
            )
        edges = np.linspace(0, nq, nproc + 1, dtype=np.int64)
        tasks = [
            (int(edges[i]), qpts[int(edges[i]): int(edges[i + 1])])
            for i in range(nproc)
            if int(edges[i + 1]) > int(edges[i])
        ]
        ctx = mp.get_context("fork")
        t1 = time.perf_counter()
        if verbose:
            chunks = [int(task[1].shape[0]) for task in tasks]
            print(f"[magph-phonon] starting fork pool with {len(tasks)} chunks: {chunks}", flush=True)
        with ctx.Pool(processes=len(tasks)) as pool:
            for start, q_cart_chunk, ph_en_chunk, ph_vec_chunk in pool.map(_build_phonon_cache_from_epr_qchunk, tasks):
                stop = int(start) + int(ph_en_chunk.shape[0])
                q_cart[int(start):stop] = q_cart_chunk
                ph_en[int(start):stop] = ph_en_chunk
                ph_vec[int(start):stop] = ph_vec_chunk
        if verbose:
            print(f"[magph-phonon] parallel phonon build finished in {time.perf_counter() - t1:.2f}s", flush=True)

    return _phonon_cache_payload(
        qpts,
        q_cart,
        ph_en,
        ph_vec,
        meta=meta,
        epr_path=epr_path,
        qmesh_shape=qmesh_shape,
        qmesh_shift_grid=qmesh_shift,
        phonon_asr=phonon_asr,
        phonon_loto=phonon_loto,
        phonon_source=phonon_source,
    )


def write_phonon_cache_from_epr(
    epr_path: str,
    out_npz: str,
    qmesh=None,
    qpts_frac=None,
    nproc=1,
    compressed=True,
    phonon_fc=None,
    phonon_asr="none",
    phonon_loto="auto",
    overwrite=False,
    qmesh_shift_grid=None,
    verbose=False,
) -> str:
    output = os.path.abspath(os.path.expanduser(out_npz))
    directory = os.path.dirname(output) or "."
    os.makedirs(directory, exist_ok=True)
    if os.path.lexists(output) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite phonon cache: {output}")

    cache = build_phonon_cache_from_epr(
        epr_path,
        qmesh=qmesh,
        qpts_frac=qpts_frac,
        nproc=nproc,
        phonon_fc=phonon_fc,
        phonon_asr=phonon_asr,
        phonon_loto=phonon_loto,
        qmesh_shift_grid=qmesh_shift_grid,
        verbose=verbose,
    )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{os.path.basename(output)}.",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as handle:
            temp_path = handle.name
            writer = np.savez_compressed if compressed else np.savez
            writer(handle, **cache)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temp_path, output)
        else:
            # Same-filesystem hard linking is an atomic no-clobber publish.
            try:
                os.link(temp_path, output)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite phonon cache: {output}"
                ) from exc
            os.unlink(temp_path)
        temp_path = None
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
    return output


def write_manifest(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _parse_neighbor_token(token: str):
    """
    Parse neighbor token like 'Mn1-Mn2' -> (0, 1)
    """
    m = re.match(r"[A-Za-z]+(\d+)-[A-Za-z]+(\d+)", token.strip())
    if not m:
        return None
    return int(m.group(1)) - 1, int(m.group(2)) - 1


def _parse_r_tuple(token: str):
    """
    Parse '(h,k,l)' string.
    """
    m = re.match(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", token.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_j_cache_npz(npz_path: str):
    """
    Load legacy J cache npz.
    Expected keys:
      - J_iso
      - i_idx
      - j_idx
      - vector (R vectors)
      - R (distance)
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"J cache npz not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        required = ("J_iso", "i_idx", "j_idx", "vector", "R")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"Missing keys in J cache npz: {missing}")
        J_iso = np.asarray(
            _load_npz_schema_array(payload, "J_iso", npz_path), dtype=np.float64
        ).reshape(-1)
        i_idx = np.asarray(
            _load_npz_schema_array(payload, "i_idx", npz_path), dtype=np.int32
        ).reshape(-1)
        j_idx = np.asarray(
            _load_npz_schema_array(payload, "j_idx", npz_path), dtype=np.int32
        ).reshape(-1)
        vec = np.asarray(
            _load_npz_schema_array(payload, "vector", npz_path), dtype=np.int32
        ).reshape(-1, 3)
        dist = np.asarray(
            _load_npz_schema_array(payload, "R", npz_path), dtype=np.float64
        ).reshape(-1)

    _require_matching_nonempty_lengths(
        npz_path, J_iso=J_iso, i_idx=i_idx, j_idx=j_idx, vector=vec, R=dist
    )
    return (J_iso, i_idx, j_idx, [tuple(x) for x in vec], dist)


def load_jr_h5(h5_path: str):
    """
    Load direct real-space J(R) HDF5 payload from compute_J_epr_kspace.

    Returns the legacy magph J tuple:
      (J_iso, i_idx, j_idx, Vec, dist)
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Jr HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        if "J_r/value" in h5:
            j_key = "J_r/value"
            J_iso = np.asarray(h5[j_key], dtype=np.float64).reshape(-1)
        elif "tb2j_extra/jiso_tb2j" in h5:
            j_key = "tb2j_extra/jiso_tb2j"
            J_iso = np.asarray(h5[j_key], dtype=np.float64).reshape(-1)
        elif "J_tensor_r" in h5:
            j_key = "J_tensor_r"
            tensor = np.asarray(h5[j_key], dtype=np.float64)
            if tensor.ndim != 3 or tensor.shape[1] != tensor.shape[2]:
                raise ValueError(f"Unsupported J_tensor_r shape in {h5_path}: {tensor.shape}")
            J_iso = np.trace(tensor, axis1=1, axis2=2) / float(tensor.shape[1])
        else:
            j_key = None
            J_iso = None

        required = ["bonds/mag_i_atom", "bonds/mag_j_atom", "bonds/R"]
        missing = [k for k in required if k not in h5]
        if j_key is None:
            missing.append("J_r/value or tb2j_extra/jiso_tb2j or J_tensor_r")
        if missing:
            raise KeyError(f"Missing Jr HDF5 datasets/groups in {h5_path}: {missing}")

        i_idx = np.asarray(h5["bonds/mag_i_atom"], dtype=np.int32).reshape(-1)
        j_idx = np.asarray(h5["bonds/mag_j_atom"], dtype=np.int32).reshape(-1)
        vec = np.asarray(h5["bonds/R"], dtype=np.int32).reshape(-1, 3)
        if "bonds/distance_ang" in h5:
            dist = np.asarray(h5["bonds/distance_ang"], dtype=np.float64).reshape(-1)
        else:
            dist = np.full(J_iso.shape, np.nan, dtype=np.float64)

    _require_matching_nonempty_lengths(
        h5_path, J_iso=J_iso, i_idx=i_idx, j_idx=j_idx, vector=vec, R=dist
    )
    return (
        J_iso,
        i_idx,
        j_idx,
        [tuple(int(x) for x in r) for r in vec],
        dist,
    )


def load_j_from_compute_j_bulk_txt(path: str):
    """
    Parse slw.exchange.legacy.compute_J_bulk text output.
    Reads the '# Detailed Bond List (Reference)' section.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"compute_J_bulk output not found: {path}")

    JJ, ii, jj, vecs, dist = [], [], [], [], []
    in_detail = False
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# Detailed Bond List"):
                in_detail = True
                continue
            if not in_detail:
                continue
            if line.startswith("-") or line.startswith("Orbit"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            neigh = parts[1]
            pair = _parse_neighbor_token(neigh)
            if pair is None:
                continue
            try:
                d = float(parts[2])
            except Exception:
                continue
            r_tok = parts[3]
            r = _parse_r_tuple(r_tok)
            if r is None:
                continue
            try:
                jv = float(parts[4])
            except Exception:
                continue
            ii.append(pair[0])
            jj.append(pair[1])
            vecs.append(r)
            dist.append(d)
            JJ.append(jv)

    if len(JJ) == 0:
        raise ValueError(f"Could not parse detailed bond list from {path}")
    return (
        np.array(JJ, dtype=np.float64),
        np.array(ii, dtype=np.int32),
        np.array(jj, dtype=np.int32),
        vecs,
        np.array(dist, dtype=np.float64),
    )


def load_j_from_tensor_txt(path: str):
    """
    Parse slw.exchange.legacy.compute_J_tensor_bulk decomposition blocks:
      [4a] Mn1-Mn1  dist=6.7100 A  R=(0,0,1)
      J_iso: +0.0070
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"compute_J_tensor_bulk output not found: {path}")

    header_re = re.compile(
        r"^\[[^\]]+\]\s+([A-Za-z]+\d+)-([A-Za-z]+\d+)\s+dist=([0-9eE+\-.]+)\s+A\s+R=\(([-0-9,\s]+)\)"
    )
    jiso_re = re.compile(r"^J_iso:\s*([0-9eE+\-.]+)")

    JJ, ii, jj, vecs, dist = [], [], [], [], []
    pending = None
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = header_re.match(line)
            if m:
                a1, a2 = m.group(1), m.group(2)
                d = float(m.group(3))
                r = tuple(int(x.strip()) for x in m.group(4).split(","))
                i0 = _label_to_atom_index(a1)
                j0 = _label_to_atom_index(a2)
                pending = (i0, j0, r, d)
                continue
            jm = jiso_re.match(line)
            if jm and pending is not None:
                jv = float(jm.group(1))
                i0, j0, r, d = pending
                ii.append(i0)
                jj.append(j0)
                vecs.append(r)
                dist.append(d)
                JJ.append(jv)
                pending = None

    if len(JJ) == 0:
        raise ValueError(f"Could not parse J_iso decomposition blocks from {path}")
    return (
        np.array(JJ, dtype=np.float64),
        np.array(ii, dtype=np.int32),
        np.array(jj, dtype=np.int32),
        vecs,
        np.array(dist, dtype=np.float64),
    )


def load_j_payload(j_path: str):
    """
    Unified loader for J payloads used by magph kernels.
    Supported:
      - legacy npz: J_iso/i_idx/j_idx/vector/R
      - compute_J_epr_kspace HDF5: bonds/* + J_r/value
      - compute_J_bulk text output
      - compute_J_tensor_bulk text output (decomposition blocks)
    Returns:
      (JJ, ii, jj, Vec, dist)
    """
    lower = j_path.lower()
    if lower.endswith(".npz"):
        return load_j_cache_npz(j_path)
    if lower.endswith((".h5", ".hdf5")):
        return load_jr_h5(j_path)

    # text mode
    # Try compute_J_bulk detailed list first.
    try:
        return load_j_from_compute_j_bulk_txt(j_path)
    except Exception:
        pass
    # Try tensor decomposition block format.
    return load_j_from_tensor_txt(j_path)
