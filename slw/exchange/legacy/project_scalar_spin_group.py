# Legacy exchange implementation; use the native engine for new workflows.
"""Audit and project scalar exchange data onto an SOC-off spin group.

The real-space convention implemented here is the EPR atomic gauge

    J(i,j,R) -> J(p(i),p(j), W R + s_j - s_i)

and, for a displacement of atom ``kappa`` in cell ``Rp`` anchored at the
first bond endpoint,

    dJ(i,j,R;kappa,Rp) ->
      Q dJ(p(i),p(j),W R+s_j-s_i;
            p(kappa),W Rp+s_kappa-s_i),

where ``W tau_a + t = tau_p(a) + s_a`` and
``Q = L.T @ W @ inv(L.T)``.  Scalar exchange has no extra sign when a
collinear spin pattern is globally reversed by a retained operation.

Only scalar/isotropic pieces are projected.  If full exchange tensors are
present, their scalar trace is replaced while their old traceless part is
preserved.  Raw response data (for example ``dA_r`` and ``trace_acc``) are
never changed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import spglib

from slw.exchange.legacy.reference.compute_J_epr_kspace import (
    _read_epr_spglib_cell,
    _species_numbers_from_labels,
)


_SCHEMA_VERSION = 1
_STATIC_SCALAR_PATHS = (
    "J_r/value",
    "tb2j_extra/jiso_tb2j",
    "J_iso_r",
    "jiso_tb2j",
)
_STATIC_ISO_TENSOR_PATHS = (
    "J_iso_tensor_r",
    "tb2j_extra/J_iso_tensor_r",
)
_STATIC_FULL_TENSOR_PATHS = (
    "J_tensor_r",
    "J_aab_full_r",
    "tb2j_extra/J_aab_full_r",
)
_DJ_SCALAR_GROUPS = ("dJ_iso_r", "dJ_r")
_DJ_ISO_TENSOR_GROUPS = ("dJ_iso_tensor_r",)
_DJ_FULL_TENSOR_GROUPS = ("dJ_tensor_r", "dJ_aab_full_r")


@dataclass(frozen=True)
class SymmetryOperation:
    """One retained structural operation and its atom mapping."""

    source_index: int
    rotation: np.ndarray
    translation: np.ndarray
    atom_map: np.ndarray
    atom_shift: np.ndarray
    cart_rotation: np.ndarray
    spin_action: str


@dataclass
class ScalarPayload:
    """Canonical in-memory scalar payload."""

    gi: np.ndarray
    gj: np.ndarray
    bond_r: np.ndarray
    static_j: np.ndarray | None
    static_source: str | None
    dj: np.ndarray | None  # (target, Rp, bond, xyz)
    dj_source: str | None
    targets: np.ndarray | None
    rp: np.ndarray | None
    qmesh: np.ndarray | None
    axes: tuple[str, ...] | None
    alias_consistency: dict[str, Any] | None = None


@dataclass
class OperationMap:
    operation: SymmetryOperation
    bond_map: np.ndarray
    target_map: np.ndarray | None
    rp_map: np.ndarray | None  # (target, Rp, bond)
    diagnostics: dict[str, Any]


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_decode(x) for x in value.reshape(-1)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _read_text_dataset(h5: h5py.File, path: str) -> str | None:
    if path not in h5:
        return None
    raw = _decode(h5[path][()])
    if isinstance(raw, list):
        return ";".join(str(x) for x in raw)
    return str(raw)


def _soc_value_active(value: Any, *, atol: float = 1.0e-14) -> bool:
    value = _decode(value)
    if isinstance(value, list):
        return any(_soc_value_active(x, atol=atol) for x in value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, complex, np.number)):
        return bool(np.any(np.abs(np.asarray(value)) > atol))
    text = str(value).strip().lower()
    if text in {"", "0", "0.0", "none", "off", "false", "no", "disabled", "[]"}:
        return False
    # Unknown non-empty SOC specifications are treated conservatively as active.
    return True


def detect_soc(h5: h5py.File) -> dict[str, Any]:
    """Conservatively detect explicit SOC metadata in a result HDF5."""

    evidence: list[dict[str, Any]] = []
    direct_paths = (
        "basic_data/soc",
        "soc",
        "basic_data/soc_entries",
        "soc_entries",
        "basic_data/spinor_hr",
        "spinor_hr",
    )
    for path in direct_paths:
        if path in h5:
            val = _decode(h5[path][()])
            evidence.append({"path": path, "value": _jsonable(val), "active": _soc_value_active(val)})

    def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset) and obj.name.rsplit("/", 1)[-1].lower().startswith("lambda_"):
            val = _decode(obj[()])
            evidence.append({"path": name, "value": _jsonable(val), "active": _soc_value_active(val)})

    h5.visititems(visit)
    for key, val in h5.attrs.items():
        low = str(key).lower()
        if low in {"soc", "soc_entries", "spinor_hr"} or low.startswith("lambda_"):
            decoded = _decode(val)
            evidence.append(
                {"path": f"@{key}", "value": _jsonable(decoded), "active": _soc_value_active(decoded)}
            )
        elif low == "base_hamiltonian":
            decoded = _decode(val)
            active = str(decoded).strip().lower() == "spinor_hr"
            evidence.append({"path": f"@{key}", "value": _jsonable(decoded), "active": active})
    return {"active": any(item["active"] for item in evidence), "evidence": evidence}


def _dataset_paths(h5: h5py.File, candidates: Iterable[str]) -> list[str]:
    return [path for path in candidates if path in h5 and isinstance(h5[path], h5py.Dataset)]


def _validate_bonds(h5: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("bonds/mag_i_atom", "bonds/mag_j_atom", "bonds/R")
    missing = [path for path in required if path not in h5]
    if missing:
        raise KeyError(f"Missing bond metadata: {missing}")
    gi = np.asarray(h5[required[0]], dtype=np.int64).reshape(-1)
    gj = np.asarray(h5[required[1]], dtype=np.int64).reshape(-1)
    bond_r = np.asarray(h5[required[2]], dtype=np.int64).reshape(-1, 3)
    if not (len(gi) == len(gj) == len(bond_r)) or len(gi) == 0:
        raise ValueError("Bond metadata arrays must have the same nonzero length")
    keys = [(int(i), int(j), *(int(x) for x in r)) for i, j, r in zip(gi, gj, bond_r)]
    if len(set(keys)) != len(keys):
        raise ValueError("Bond list contains duplicate directed (i,j,R) keys")
    return gi, gj, bond_r


def _dataset_name(group: h5py.Group, atom: int, bond: int) -> str | None:
    names = (f"m{atom + 1}_b{bond + 1}", f"dJ_r_m{atom + 1}_b{bond + 1}")
    hits = [name for name in names if name in group and isinstance(group[name], h5py.Dataset)]
    if len(hits) > 1:
        raise ValueError(f"Ambiguous dJ datasets in /{group.name}: {hits}")
    return hits[0] if hits else None


def _read_dj_group(
    group: h5py.Group,
    targets: np.ndarray,
    n_rp: int,
    n_bond: int,
    axis_to_xyz: np.ndarray,
    *,
    tensor: bool,
) -> np.ndarray:
    out = np.empty((len(targets), n_rp, n_bond, 3), dtype=np.float64)
    for it, atom in enumerate(targets):
        for ib in range(n_bond):
            name = _dataset_name(group, int(atom), ib)
            if name is None:
                raise KeyError(f"Missing target={int(atom)} bond={ib} dataset in /{group.name}")
            arr = np.asarray(group[name], dtype=np.float64)
            if tensor:
                if arr.shape != (n_rp, 3, 3, 3):
                    raise ValueError(
                        f"/{group.name}/{name} has shape {arr.shape}; expected {(n_rp, 3, 3, 3)}"
                    )
                scalar_file_axes = np.trace(arr, axis1=2, axis2=3) / 3.0
            else:
                if arr.shape != (n_rp, 3):
                    raise ValueError(f"/{group.name}/{name} has shape {arr.shape}; expected {(n_rp, 3)}")
                scalar_file_axes = arr
            out[it, :, ib, :] = scalar_file_axes[:, axis_to_xyz]
    return out


def _alias_comparison(name: str, actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"Scalar alias {name} shape {actual.shape} != canonical {reference.shape}")
    diff = actual - reference
    scale = max(1.0, float(np.max(np.abs(reference), initial=0.0)))
    tolerance = 1.0e-12 * scale + 1.0e-10 * float(np.max(np.abs(reference), initial=0.0))
    max_abs = float(np.max(np.abs(diff), initial=0.0))
    result = {"path": name, **_relative_metrics(diff, reference), "tolerance": tolerance}
    result["consistent"] = max_abs <= tolerance
    if not result["consistent"]:
        raise ValueError(
            f"Scalar aliases are inconsistent: {name} differs from the canonical scalar "
            f"by max_abs={max_abs:.3e} (tolerance={tolerance:.3e})"
        )
    return result


def validate_alias_consistency(h5: h5py.File, payload: ScalarPayload) -> dict[str, Any]:
    """Require all consumer-visible scalar decompositions to agree pre-projection."""

    report: dict[str, Any] = {
        "policy": (
            "strict: scalar aliases, isotropic tensors, and trace/3 of explicitly named full exchange "
            "tensors must agree; raw dA_r and trace_acc are excluded"
        ),
        "static_J": [],
        "dJ": [],
    }
    if payload.static_j is not None:
        reference = payload.static_j
        for path in _dataset_paths(h5, _STATIC_SCALAR_PATHS):
            report["static_J"].append(
                _alias_comparison(path, np.asarray(h5[path], dtype=np.float64).reshape(-1), reference)
            )
        if "J_r" in h5 and isinstance(h5["J_r"], h5py.Group):
            individual_names = [f"J_r_b{ib + 1}" for ib in range(len(reference))]
            present = [name in h5["J_r"] for name in individual_names]
            if any(present) and not all(present):
                raise ValueError("J_r/J_r_b* scalar aliases are only partially present")
            if all(present):
                individual = np.asarray([h5["J_r"][name][()] for name in individual_names], dtype=np.float64)
                report["static_J"].append(_alias_comparison("J_r/J_r_b*", individual, reference))
        for path in _dataset_paths(h5, (*_STATIC_ISO_TENSOR_PATHS, *_STATIC_FULL_TENSOR_PATHS)):
            arr = np.asarray(h5[path], dtype=np.float64)
            if arr.shape != (len(reference), 3, 3):
                raise ValueError(f"/{path} has unsupported full-tensor shape {arr.shape}")
            trace = np.trace(arr, axis1=1, axis2=2) / 3.0
            report["static_J"].append(_alias_comparison(f"trace({path})/3", trace, reference))

    if payload.dj is not None:
        assert payload.targets is not None and payload.rp is not None and payload.axes is not None
        axis_to_xyz = np.asarray([payload.axes.index(axis) for axis in ("x", "y", "z")], dtype=np.int64)
        for group_name in _DJ_SCALAR_GROUPS:
            if group_name in h5:
                actual = _read_dj_group(
                    h5[group_name], payload.targets, len(payload.rp), len(payload.gi), axis_to_xyz, tensor=False
                )
                report["dJ"].append(_alias_comparison(group_name, actual, payload.dj))
        for group_name in (*_DJ_ISO_TENSOR_GROUPS, *_DJ_FULL_TENSOR_GROUPS):
            if group_name in h5:
                actual = _read_dj_group(
                    h5[group_name], payload.targets, len(payload.rp), len(payload.gi), axis_to_xyz, tensor=True
                )
                report["dJ"].append(_alias_comparison(f"trace({group_name})/3", actual, payload.dj))
    report["n_checked"] = len(report["static_J"]) + len(report["dJ"])
    return report


def load_scalar_payload(h5: h5py.File) -> ScalarPayload:
    """Load every required index and one canonical scalar source."""

    gi, gj, bond_r = _validate_bonds(h5)
    n_bond = len(gi)

    static_paths = _dataset_paths(h5, _STATIC_SCALAR_PATHS)
    static_j: np.ndarray | None = None
    static_source: str | None = None
    if static_paths:
        static_source = static_paths[0]
        static_j = np.asarray(h5[static_source], dtype=np.float64).reshape(-1)
        if static_j.shape != (n_bond,):
            raise ValueError(f"/{static_source} has shape {static_j.shape}; expected {(n_bond,)}")
    elif "J_iso_tensor_r" in h5:
        arr = np.asarray(h5["J_iso_tensor_r"], dtype=np.float64)
        if arr.shape != (n_bond, 3, 3):
            raise ValueError(f"/J_iso_tensor_r has unsupported shape {arr.shape}")
        static_source = "J_iso_tensor_r(trace/3)"
        static_j = np.trace(arr, axis1=1, axis2=2) / 3.0
    elif "J_tensor_r" in h5:
        arr = np.asarray(h5["J_tensor_r"], dtype=np.float64)
        if arr.shape != (n_bond, 3, 3):
            raise ValueError(f"/J_tensor_r has unsupported shape {arr.shape}")
        static_source = "J_tensor_r(trace/3)"
        static_j = np.trace(arr, axis1=1, axis2=2) / 3.0

    has_dj = any(name in h5 for name in (*_DJ_SCALAR_GROUPS, *_DJ_ISO_TENSOR_GROUPS, *_DJ_FULL_TENSOR_GROUPS))
    dj = None
    dj_source = None
    targets = rp = qmesh = None
    axes: tuple[str, ...] | None = None
    if has_dj:
        required = ("displacements/target_atom", "displacements/Rp", "displacements/axes", "basic_data/qmesh")
        missing = [path for path in required if path not in h5]
        if missing:
            raise KeyError(f"Missing dJ metadata: {missing}")
        targets = np.asarray(h5[required[0]], dtype=np.int64).reshape(-1)
        rp = np.asarray(h5[required[1]], dtype=np.int64).reshape(-1, 3)
        qmesh = np.asarray(h5[required[3]], dtype=np.int64).reshape(-1)
        if qmesh.shape != (3,) or np.any(qmesh <= 0):
            raise ValueError(f"Invalid qmesh: {qmesh.tolist()}")
        raw_axes = [_decode(x) for x in np.asarray(h5[required[2]]).reshape(-1)]
        axes = tuple(str(x).strip().lower() for x in raw_axes)
        if len(axes) != 3 or set(axes) != {"x", "y", "z"}:
            raise ValueError(f"Displacement axes must be a permutation of x,y,z, got {axes}")
        axis_to_xyz = np.asarray([axes.index(axis) for axis in ("x", "y", "z")], dtype=np.int64)
        if len(set(int(x) for x in targets)) != len(targets):
            raise ValueError("displacements/target_atom contains duplicates")
        if len(rp) != int(np.prod(qmesh)):
            raise ValueError(f"Rp count {len(rp)} does not equal prod(qmesh)={int(np.prod(qmesh))}")
        rp_mod = np.mod(rp, qmesh[None, :])
        if len({tuple(int(x) for x in row) for row in rp_mod}) != len(rp):
            raise ValueError("displacements/Rp is not unique modulo qmesh")

        scalar_groups = [name for name in _DJ_SCALAR_GROUPS if name in h5]
        iso_tensor_groups = [name for name in _DJ_ISO_TENSOR_GROUPS if name in h5]
        full_groups = [name for name in _DJ_FULL_TENSOR_GROUPS if name in h5]
        if scalar_groups:
            dj_source = scalar_groups[0]
            dj = _read_dj_group(
                h5[dj_source], targets, len(rp), n_bond, axis_to_xyz, tensor=False
            )
        elif iso_tensor_groups:
            dj_source = f"{iso_tensor_groups[0]}(trace/3)"
            dj = _read_dj_group(
                h5[iso_tensor_groups[0]], targets, len(rp), n_bond, axis_to_xyz, tensor=True
            )
        elif full_groups:
            dj_source = f"{full_groups[0]}(trace/3)"
            dj = _read_dj_group(
                h5[full_groups[0]], targets, len(rp), n_bond, axis_to_xyz, tensor=True
            )

    if static_j is None and dj is None:
        raise KeyError("No supported scalar J_iso or dJ_iso payload was found")
    payload = ScalarPayload(
        gi=gi,
        gj=gj,
        bond_r=bond_r,
        static_j=static_j,
        static_source=static_source,
        dj=dj,
        dj_source=dj_source,
        targets=targets,
        rp=rp,
        qmesh=qmesh,
        axes=axes,
    )
    payload.alias_consistency = validate_alias_consistency(h5, payload)
    return payload


def _atom_mapping(
    rotation: np.ndarray,
    translation: np.ndarray,
    tau: np.ndarray,
    *,
    tolerance: float,
    lattice: np.ndarray,
    numbers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.einsum("ab,nb->na", rotation, tau) + translation[None, :]
    # Pairwise minimum-image fractional mismatch.  spglib operations should
    # leave a unique match well within the requested symmetry tolerance.
    delta = raw[:, None, :] - tau[None, :, :]
    wrapped = delta - np.rint(delta)
    lattice = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    norm = np.linalg.norm(wrapped @ lattice, axis=2)
    if numbers is not None:
        species = np.asarray(numbers).reshape(-1)
        if species.shape != (len(tau),):
            raise ValueError("Species-number array does not match atomic positions")
        allowed = species[:, None] == species[None, :]
        norm = np.where(allowed, norm, np.inf)
    atom_map = np.argmin(norm, axis=1).astype(np.int64)
    minimum = norm[np.arange(len(tau)), atom_map]
    if np.any(minimum > tolerance):
        bad = np.flatnonzero(minimum > tolerance)
        raise ValueError(f"Symmetry operation failed to map atoms {bad.tolist()}; max mismatch={minimum.max():.3e}")
    if len(np.unique(atom_map)) != len(tau):
        raise ValueError("Symmetry operation atom mapping is not bijective")
    shift = np.rint(raw - tau[atom_map]).astype(np.int64)
    residual = raw - tau[atom_map] - shift
    residual_cart_norm = np.linalg.norm(residual @ lattice, axis=1)
    if np.max(residual_cart_norm, initial=0.0) > tolerance:
        raise ValueError("Nonintegral atom cell shift encountered")
    return atom_map, shift


def _classify_spin_action(atom_map: np.ndarray, pattern: np.ndarray, *, tolerance: float) -> str | None:
    mapped = np.empty_like(pattern, dtype=np.float64)
    mapped[atom_map] = pattern
    if np.allclose(mapped, pattern, atol=tolerance, rtol=0.0):
        return "preserve"
    if np.allclose(mapped, -pattern, atol=tolerance, rtol=0.0):
        return "flip"
    return None


def build_symmetry_operations(
    lattice: np.ndarray,
    tau: np.ndarray,
    numbers: np.ndarray,
    *,
    symprec: float,
    angle_tolerance: float,
    spin_pattern: Sequence[float] | None,
) -> tuple[list[SymmetryOperation], dict[str, Any]]:
    """Build and optionally spin-filter spglib structural operations."""

    symmetry = spglib.get_symmetry(
        (lattice, tau, numbers),
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    if symmetry is None or len(symmetry.get("rotations", [])) == 0:
        raise RuntimeError("spglib returned no structural symmetry operations")
    rotations = np.asarray(symmetry["rotations"], dtype=np.int64)
    translations = np.asarray(symmetry["translations"], dtype=np.float64)
    pattern = None if spin_pattern is None else np.asarray(spin_pattern, dtype=np.float64).reshape(-1)
    if pattern is not None and pattern.shape != (len(tau),):
        raise ValueError(f"spin_pattern length {len(pattern)} does not equal nat={len(tau)}")

    inv_lt = np.linalg.inv(lattice.T)
    atom_tol = max(float(symprec), 1.0e-8)
    spin_tol = max(float(symprec), 1.0e-10)
    orth_tol = max(10.0 * float(symprec), 1.0e-8)
    kept: list[SymmetryOperation] = []
    rejected: list[int] = []
    operation_summary: list[dict[str, Any]] = []
    for index, (rotation, translation) in enumerate(zip(rotations, translations)):
        atom_map, atom_shift = _atom_mapping(
            rotation, translation, tau, tolerance=atom_tol, lattice=lattice, numbers=numbers
        )
        if not np.array_equal(np.asarray(numbers)[atom_map], np.asarray(numbers)):
            raise ValueError(f"Symmetry operation {index} maps atoms across chemical species")
        q_cart = lattice.T @ rotation @ inv_lt
        orth_error = float(np.max(np.abs(q_cart.T @ q_cart - np.eye(3))))
        if orth_error > orth_tol:
            raise ValueError(
                f"Cartesian rotation for operation {index} is not orthogonal: max error={orth_error:.3e}"
            )
        action = "structural_parent" if pattern is None else _classify_spin_action(
            atom_map, pattern, tolerance=spin_tol
        )
        selected = action is not None
        if selected:
            kept.append(
                SymmetryOperation(
                    source_index=index,
                    rotation=rotation,
                    translation=translation,
                    atom_map=atom_map,
                    atom_shift=atom_shift,
                    cart_rotation=q_cart,
                    spin_action=str(action),
                )
            )
        else:
            rejected.append(index)
        operation_summary.append(
            {
                "index": index,
                "selected": selected,
                "spin_action": action if action is not None else "rejected_not_global_preserve_or_flip",
                "det_cart": float(np.linalg.det(q_cart)),
                "orthogonality_max_abs": orth_error,
                "translation": translation.tolist(),
                "rotation_fractional": rotation.tolist(),
                "atom_map": atom_map.tolist(),
                "atom_shift": atom_shift.tolist(),
                "cartesian_rotation": q_cart.tolist(),
            }
        )
    if not kept:
        raise ValueError("Spin-pattern filter rejected every structural symmetry operation")
    return kept, {
        "n_structural_operations": len(rotations),
        "n_selected_operations": len(kept),
        "rejected_operation_indices": rejected,
        "spin_pattern": None if pattern is None else pattern.tolist(),
        "group_assumption": (
            "full_SOC_off_parent_structural_group"
            if pattern is None
            else "collinear_pattern_preserve_or_global_flip"
        ),
        "operations": operation_summary,
    }


def _linear_rp_hash(rp_mod: np.ndarray, qmesh: np.ndarray) -> np.ndarray:
    return (rp_mod[..., 0] * qmesh[1] + rp_mod[..., 1]) * qmesh[2] + rp_mod[..., 2]


def build_operation_maps(payload: ScalarPayload, operations: Sequence[SymmetryOperation]) -> list[OperationMap]:
    """Build exact bond/target/Rp permutations; retain diagnostics if incomplete."""

    bond_lookup = {
        (int(i), int(j), *(int(x) for x in r)): ib
        for ib, (i, j, r) in enumerate(zip(payload.gi, payload.gj, payload.bond_r))
    }
    target_lookup = None
    rp_lut = None
    if payload.dj is not None:
        assert payload.targets is not None and payload.rp is not None and payload.qmesh is not None
        target_lookup = {int(atom): it for it, atom in enumerate(payload.targets)}
        ncell = int(np.prod(payload.qmesh))
        rp_lut = np.full(ncell, -1, dtype=np.int64)
        hashes = _linear_rp_hash(np.mod(payload.rp, payload.qmesh[None, :]), payload.qmesh)
        rp_lut[hashes] = np.arange(len(payload.rp), dtype=np.int64)

    maps: list[OperationMap] = []
    for operation in operations:
        p = operation.atom_map
        shifts = operation.atom_shift
        transformed_r = np.einsum("ab,nb->na", operation.rotation, payload.bond_r)
        transformed_r += shifts[payload.gj] - shifts[payload.gi]
        bond_keys = [
            (int(p[i]), int(p[j]), *(int(x) for x in r))
            for i, j, r in zip(payload.gi, payload.gj, transformed_r)
        ]
        bond_map = np.asarray([bond_lookup.get(key, -1) for key in bond_keys], dtype=np.int64)
        valid_bond = bond_map >= 0
        bond_bijective = bool(np.all(valid_bond) and len(np.unique(bond_map)) == len(bond_map))

        target_map = None
        rp_map = None
        target_bijective = True
        full_bijective = bond_bijective
        missing_targets = 0
        qmesh_compatible = True
        if payload.dj is not None:
            assert target_lookup is not None and rp_lut is not None
            assert payload.targets is not None and payload.rp is not None and payload.qmesh is not None
            target_map = np.asarray([target_lookup.get(int(p[a]), -1) for a in payload.targets], dtype=np.int64)
            target_bijective = bool(np.all(target_map >= 0) and len(np.unique(target_map)) == len(target_map))
            missing_targets = int(np.count_nonzero(target_map < 0))

            # W must induce an automorphism of Z^3 / diag(qmesh) Z^3.  Merely
            # obtaining a permutation for one representative list is not enough:
            # equivalent representatives Rp and Rp+q_a e_a must remain equivalent.
            wq = operation.rotation * payload.qmesh[None, :]
            qmesh_compatible = bool(
                np.all(np.mod(wq, payload.qmesh[:, None]) == 0)
            )

            # Rp' = W Rp + s_kappa - s_i, with i the source-bond origin.
            wrp = np.einsum("ab,rb->ra", operation.rotation, payload.rp)
            raw_rp = (
                wrp[None, :, None, :]
                + shifts[payload.targets][:, None, None, :]
                - shifts[payload.gi][None, None, :, :]
            )
            hashes = _linear_rp_hash(np.mod(raw_rp, payload.qmesh[None, None, None, :]), payload.qmesh)
            rp_map = rp_lut[hashes]
            valid_full = (
                (target_map[:, None, None] >= 0)
                & (rp_map >= 0)
                & (bond_map[None, None, :] >= 0)
            )
            if np.all(valid_full):
                n_rp, n_bond = len(payload.rp), len(payload.gi)
                destinations = (
                    (target_map[:, None, None] * n_rp + rp_map) * n_bond
                    + bond_map[None, None, :]
                )
                full_bijective = bool(qmesh_compatible and
                    len(np.unique(destinations)) == int(np.prod(destinations.shape))
                )
            else:
                full_bijective = False

        maps.append(
            OperationMap(
                operation=operation,
                bond_map=bond_map,
                target_map=target_map,
                rp_map=rp_map,
                diagnostics={
                    "operation_index": operation.source_index,
                    "spin_action": operation.spin_action,
                    "missing_bonds": int(np.count_nonzero(~valid_bond)),
                    "bond_bijective": bond_bijective,
                    "missing_targets": missing_targets,
                    "target_bijective": target_bijective,
                    "qmesh_compatible": qmesh_compatible,
                    "full_dj_bijective": full_bijective,
                },
            )
        )
    return maps


def build_mate_maps(payload: ScalarPayload) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Build directed-mate permutations, including Rp -> Rp-R for dJ."""

    lookup = {
        (int(i), int(j), *(int(x) for x in r)): ib
        for ib, (i, j, r) in enumerate(zip(payload.gi, payload.gj, payload.bond_r))
    }
    keys = [
        (int(j), int(i), *(int(-x) for x in r))
        for i, j, r in zip(payload.gi, payload.gj, payload.bond_r)
    ]
    bond_mate = np.asarray([lookup.get(key, -1) for key in keys], dtype=np.int64)
    valid = bond_mate >= 0
    bond_bijective = bool(np.all(valid) and len(np.unique(bond_mate)) == len(bond_mate))
    involutive = bool(bond_bijective and np.array_equal(bond_mate[bond_mate], np.arange(len(bond_mate))))

    rp_mate = None
    full_bijective = bond_bijective
    if payload.dj is not None:
        assert payload.rp is not None and payload.qmesh is not None
        rp_mod = np.mod(payload.rp, payload.qmesh[None, :])
        lut = np.full(int(np.prod(payload.qmesh)), -1, dtype=np.int64)
        lut[_linear_rp_hash(rp_mod, payload.qmesh)] = np.arange(len(payload.rp))
        shifted = payload.rp[:, None, :] - payload.bond_r[None, :, :]
        rp_mate = lut[_linear_rp_hash(np.mod(shifted, payload.qmesh[None, None, :]), payload.qmesh)]
        if np.all(rp_mate >= 0) and bond_bijective:
            dest = rp_mate * len(payload.gi) + bond_mate[None, :]
            full_bijective = bool(len(np.unique(dest)) == dest.size)
        else:
            full_bijective = False
    return bond_mate, rp_mate, {
        "missing_bond_mates": int(np.count_nonzero(~valid)),
        "bond_mate_bijective": bond_bijective,
        "bond_mate_involutive": involutive,
        "full_dj_mate_bijective": full_bijective,
    }


def _relative_metrics(diff: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    d = np.asarray(diff, dtype=np.float64).reshape(-1)
    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(r))
    numerator = float(np.linalg.norm(d))
    return {
        "n_components": int(d.size),
        "l2_abs": numerator,
        "l2_relative": numerator / max(denominator, np.finfo(np.float64).tiny),
        "rms_abs": float(np.sqrt(np.mean(d * d))) if d.size else 0.0,
        "max_abs": float(np.max(np.abs(d), initial=0.0)),
    }


def _audit_static_symmetry(values: np.ndarray, maps: Sequence[OperationMap]) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    all_diff: list[np.ndarray] = []
    all_ref: list[np.ndarray] = []
    for mapping in maps:
        valid = mapping.bond_map >= 0
        source = np.flatnonzero(valid)
        diff = values[mapping.bond_map[source]] - values[source]
        metrics = _relative_metrics(diff, values[source])
        metrics.update(mapping.diagnostics)
        operations.append(metrics)
        if source.size:
            all_diff.append(diff)
            all_ref.append(values[source])
    aggregate = _relative_metrics(
        np.concatenate(all_diff) if all_diff else np.empty(0),
        np.concatenate(all_ref) if all_ref else np.empty(0),
    )
    return {"aggregate": aggregate, "operations": operations}


def _valid_dj_sources(mapping: OperationMap, shape: tuple[int, int, int, int]) -> tuple[np.ndarray, ...]:
    nt, nrp, nb, _ = shape
    assert mapping.target_map is not None and mapping.rp_map is not None
    valid = (
        (mapping.target_map[:, None, None] >= 0)
        & (mapping.rp_map >= 0)
        & (mapping.bond_map[None, None, :] >= 0)
    )
    return np.nonzero(valid)


def _audit_dj_symmetry(values: np.ndarray, maps: Sequence[OperationMap]) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    all_diff: list[np.ndarray] = []
    all_ref: list[np.ndarray] = []
    for mapping in maps:
        st, sr, sb = _valid_dj_sources(mapping, values.shape)
        assert mapping.target_map is not None and mapping.rp_map is not None
        transformed = values[st, sr, sb] @ mapping.operation.cart_rotation.T
        destination = values[
            mapping.target_map[st],
            mapping.rp_map[st, sr, sb],
            mapping.bond_map[sb],
        ]
        diff = destination - transformed
        metrics = _relative_metrics(diff, transformed)
        metrics.update(mapping.diagnostics)
        operations.append(metrics)
        if len(st):
            all_diff.append(diff.reshape(-1))
            all_ref.append(transformed.reshape(-1))
    aggregate = _relative_metrics(
        np.concatenate(all_diff) if all_diff else np.empty(0),
        np.concatenate(all_ref) if all_ref else np.empty(0),
    )
    return {"aggregate": aggregate, "operations": operations}


def _audit_static_reciprocity(values: np.ndarray, bond_mate: np.ndarray) -> dict[str, Any]:
    valid = bond_mate >= 0
    src = np.flatnonzero(valid)
    return _relative_metrics(values[bond_mate[src]] - values[src], values[src])


def _audit_dj_reciprocity(
    values: np.ndarray,
    bond_mate: np.ndarray,
    rp_mate: np.ndarray | None,
) -> dict[str, Any]:
    assert rp_mate is not None
    valid_b = bond_mate >= 0
    sr, sb = np.nonzero((rp_mate >= 0) & valid_b[None, :])
    destination = values[:, rp_mate[sr, sb], bond_mate[sb], :]
    source = values[:, sr, sb, :]
    return _relative_metrics(destination - source, source)


def _audit_asr(values: np.ndarray) -> dict[str, Any]:
    residual = values.sum(axis=(0, 1))  # (bond, xyz)
    norms = np.linalg.norm(residual, axis=1)
    return {
        "n_targets_times_rp": int(values.shape[0] * values.shape[1]),
        "l2_residual": float(np.linalg.norm(residual)),
        "max_abs_component": float(np.max(np.abs(residual), initial=0.0)),
        "max_vector_norm": float(np.max(norms, initial=0.0)),
        "rms_vector_norm": float(np.sqrt(np.mean(norms * norms))) if norms.size else 0.0,
    }


def audit_payload(
    payload: ScalarPayload,
    maps: Sequence[OperationMap],
    bond_mate: np.ndarray,
    rp_mate: np.ndarray | None,
    mate_diagnostics: dict[str, Any],
    *,
    nat: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"mate_mapping": mate_diagnostics}
    if payload.static_j is not None:
        result["static_J"] = {
            "source": payload.static_source,
            "symmetry": _audit_static_symmetry(payload.static_j, maps),
            "reciprocity": _audit_static_reciprocity(payload.static_j, bond_mate),
        }
    if payload.dj is not None:
        assert payload.targets is not None
        result["dJ"] = {
            "source": payload.dj_source,
            "target_coverage": {
                "nat": int(nat),
                "n_targets": int(len(payload.targets)),
                "full": set(int(x) for x in payload.targets) == set(range(nat)),
                "missing_atoms": sorted(set(range(nat)) - set(int(x) for x in payload.targets)),
            },
            "symmetry": _audit_dj_symmetry(payload.dj, maps),
            "reciprocity": _audit_dj_reciprocity(payload.dj, bond_mate, rp_mate),
            "asr": _audit_asr(payload.dj),
        }
    return result


def _require_projectable(
    payload: ScalarPayload,
    maps: Sequence[OperationMap],
    mate_diagnostics: dict[str, Any],
    *,
    nat: int,
    enforce_asr: bool,
) -> None:
    failures: list[str] = []
    for mapping in maps:
        d = mapping.diagnostics
        if not d["bond_bijective"]:
            failures.append(f"op {d['operation_index']}: bond map not bijective")
        if payload.dj is not None and not d["full_dj_bijective"]:
            failures.append(f"op {d['operation_index']}: target/Rp/bond map not bijective")
    if not mate_diagnostics["bond_mate_bijective"] or not mate_diagnostics["bond_mate_involutive"]:
        failures.append("directed bond mate map is not a bijective involution")
    if payload.dj is not None and not mate_diagnostics["full_dj_mate_bijective"]:
        failures.append("dJ mate (bond,Rp) map is not bijective")
    if payload.dj is not None and enforce_asr:
        assert payload.targets is not None
        if set(int(x) for x in payload.targets) != set(range(nat)):
            failures.append("ASR projection requires all crystal atoms in displacements/target_atom")
    if failures:
        raise ValueError("Input is not closed under the requested projection:\n  - " + "\n  - ".join(failures))


def project_static(values: np.ndarray, maps: Sequence[OperationMap], bond_mate: np.ndarray) -> np.ndarray:
    """Orthogonally project scalar static J under symmetry and mate maps."""

    acc = np.zeros_like(values, dtype=np.float64)
    for mapping in maps:
        action = np.empty_like(values, dtype=np.float64)
        action[mapping.bond_map] = values
        acc += action
    projected = acc / float(len(maps))
    mate_action = np.empty_like(projected)
    mate_action[bond_mate] = projected
    return 0.5 * (projected + mate_action)


def project_dj(
    values: np.ndarray,
    maps: Sequence[OperationMap],
    bond_mate: np.ndarray,
    rp_mate: np.ndarray,
    *,
    enforce_asr: bool,
) -> np.ndarray:
    """Orthogonally project scalar dJ using vectorized permutation actions."""

    nt, nrp, nb, _ = values.shape
    acc = np.zeros_like(values, dtype=np.float64)
    flat_acc = acc.reshape(-1, 3)
    for mapping in maps:
        assert mapping.target_map is not None and mapping.rp_map is not None
        destination = (
            (mapping.target_map[:, None, None] * nrp + mapping.rp_map) * nb
            + mapping.bond_map[None, None, :]
        )
        transformed = values @ mapping.operation.cart_rotation.T
        # The map has already been validated as a permutation, so direct indexed
        # accumulation is safe and substantially faster than scalar loops.
        flat_acc[destination.reshape(-1)] += transformed.reshape(-1, 3)
    projected = acc / float(len(maps))

    mate_action = np.empty_like(projected)
    destination = rp_mate * nb + bond_mate[None, :]
    mate_action.reshape(nt, -1, 3)[:, destination.reshape(-1), :] = projected.reshape(nt, -1, 3)
    projected = 0.5 * (projected + mate_action)
    if enforce_asr:
        projected -= projected.sum(axis=(0, 1), keepdims=True) / float(nt * nrp)
    return projected


def _fail_closed_validation(
    after: dict[str, Any],
    payload: ScalarPayload,
    *,
    enforce_asr: bool,
    idempotency: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scale = 1.0
    if payload.static_j is not None:
        scale = max(scale, float(np.max(np.abs(payload.static_j), initial=0.0)))
    if payload.dj is not None:
        scale = max(scale, float(np.max(np.abs(payload.dj), initial=0.0)))
    abs_tol = 1.0e-10 * scale
    rel_tol = 1.0e-10
    failures: list[str] = []

    def require_metrics(label: str, metrics: dict[str, Any]) -> None:
        if float(metrics["max_abs"]) > abs_tol and float(metrics["l2_relative"]) > rel_tol:
            failures.append(
                f"{label}: max_abs={float(metrics['max_abs']):.3e}, "
                f"relative={float(metrics['l2_relative']):.3e}"
            )

    if "static_J" in after:
        require_metrics("static symmetry", after["static_J"]["symmetry"]["aggregate"])
        require_metrics("static reciprocity", after["static_J"]["reciprocity"])
    if "dJ" in after:
        require_metrics("dJ symmetry", after["dJ"]["symmetry"]["aggregate"])
        require_metrics("dJ reciprocity", after["dJ"]["reciprocity"])
        if enforce_asr and float(after["dJ"]["asr"]["max_abs_component"]) > abs_tol:
            failures.append(
                f"dJ ASR: max_abs_component={float(after['dJ']['asr']['max_abs_component']):.3e}"
            )
    for label, metrics in idempotency.items():
        require_metrics(f"{label} idempotency", metrics)
    result = {
        "passed": not failures,
        "absolute_tolerance": abs_tol,
        "relative_tolerance": rel_tol,
        "failures": failures,
        "idempotency": idempotency,
    }
    if failures:
        raise RuntimeError("Projected payload failed closed validation:\n  - " + "\n  - ".join(failures))
    return result


def _write_static_values(h5: h5py.File, values: np.ndarray) -> dict[str, list[str]]:
    updated: list[str] = []
    for path in _dataset_paths(h5, _STATIC_SCALAR_PATHS):
        if h5[path].shape != values.shape:
            raise ValueError(f"Cannot update /{path}: shape {h5[path].shape} != {values.shape}")
        h5[path][...] = values
        updated.append(path)
    if "J_r" in h5 and isinstance(h5["J_r"], h5py.Group):
        group = h5["J_r"]
        for ib, value in enumerate(values):
            name = f"J_r_b{ib + 1}"
            if name in group:
                if group[name].shape != ():
                    raise ValueError(f"Cannot update /J_r/{name}: expected a scalar dataset")
                group[name][...] = float(value)
                updated.append(f"J_r/{name}")
    eye = np.eye(3, dtype=np.float64)
    for path in _dataset_paths(h5, _STATIC_ISO_TENSOR_PATHS):
        expected = (len(values), 3, 3)
        if h5[path].shape != expected:
            raise ValueError(f"Cannot update /{path}: shape {h5[path].shape} != {expected}")
        h5[path][...] = values[:, None, None] * eye[None, :, :]
        updated.append(path)

    full_trace_updated: list[str] = []
    for path in _dataset_paths(h5, _STATIC_FULL_TENSOR_PATHS):
        arr = np.asarray(h5[path], dtype=np.float64)
        expected = (len(values), 3, 3)
        if arr.shape != expected:
            raise ValueError(f"Cannot update scalar trace of /{path}: shape {arr.shape} != {expected}")
        delta = values - np.trace(arr, axis1=1, axis2=2) / 3.0
        arr += delta[:, None, None] * eye[None, :, :]
        h5[path][...] = arr
        full_trace_updated.append(path)
    return {"scalar_and_isotropic": updated, "full_tensor_scalar_trace_only": full_trace_updated}


def _file_axis_values(values_xyz: np.ndarray, axes: tuple[str, ...]) -> np.ndarray:
    out = np.empty_like(values_xyz)
    for file_axis, label in enumerate(axes):
        out[..., file_axis] = values_xyz[..., ("x", "y", "z").index(label)]
    return out


def _write_dj_group_scalar(
    group: h5py.Group,
    values_file: np.ndarray,
    targets: np.ndarray,
) -> list[str]:
    updated: list[str] = []
    nt, nrp, nb, _ = values_file.shape
    for it, atom in enumerate(targets):
        for ib in range(nb):
            name = _dataset_name(group, int(atom), ib)
            if name is None:
                raise KeyError(f"Missing target={int(atom)} bond={ib} dataset in /{group.name}")
            if group[name].shape != (nrp, 3):
                raise ValueError(f"Unexpected shape in /{group.name}/{name}: {group[name].shape}")
            group[name][...] = values_file[it, :, ib, :]
            updated.append(f"{group.name.lstrip('/')}/{name}")
    return updated


def _write_dj_group_tensor(
    group: h5py.Group,
    values_file: np.ndarray,
    targets: np.ndarray,
    *,
    isotropic_only: bool,
) -> list[str]:
    updated: list[str] = []
    nt, nrp, nb, _ = values_file.shape
    eye = np.eye(3, dtype=np.float64)
    for it, atom in enumerate(targets):
        for ib in range(nb):
            name = _dataset_name(group, int(atom), ib)
            if name is None:
                raise KeyError(f"Missing target={int(atom)} bond={ib} dataset in /{group.name}")
            arr = np.asarray(group[name], dtype=np.float64)
            if arr.shape != (nrp, 3, 3, 3):
                raise ValueError(f"Unexpected shape in /{group.name}/{name}: {arr.shape}")
            scalar = values_file[it, :, ib, :]
            if isotropic_only:
                arr = scalar[:, :, None, None] * eye[None, None, :, :]
            else:
                old_scalar = np.trace(arr, axis1=2, axis2=3) / 3.0
                arr += (scalar - old_scalar)[:, :, None, None] * eye[None, None, :, :]
            group[name][...] = arr
            updated.append(f"{group.name.lstrip('/')}/{name}")
    return updated


def _write_dj_values(h5: h5py.File, payload: ScalarPayload, values: np.ndarray) -> dict[str, list[str]]:
    assert payload.targets is not None and payload.axes is not None
    values_file = _file_axis_values(values, payload.axes)
    scalar_updated: list[str] = []
    for name in _DJ_SCALAR_GROUPS:
        if name in h5:
            scalar_updated.extend(_write_dj_group_scalar(h5[name], values_file, payload.targets))
    for name in _DJ_ISO_TENSOR_GROUPS:
        if name in h5:
            scalar_updated.extend(
                _write_dj_group_tensor(h5[name], values_file, payload.targets, isotropic_only=True)
            )
    trace_updated: list[str] = []
    for name in _DJ_FULL_TENSOR_GROUPS:
        if name in h5:
            trace_updated.extend(
                _write_dj_group_tensor(h5[name], values_file, payload.targets, isotropic_only=False)
            )
    return {"scalar_and_isotropic": scalar_updated, "full_tensor_scalar_trace_only": trace_updated}


def _assert_array_close(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
    if not np.allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12 * scale):
        error = float(np.max(np.abs(actual - expected), initial=0.0))
        raise RuntimeError(f"Post-write verification failed for {name}: max_abs={error:.3e}")


def _verify_written_file(
    path: Path,
    original_payload: ScalarPayload,
    expected_static: np.ndarray | None,
    expected_dj: np.ndarray | None,
) -> None:
    """Reopen the temporary HDF5 and verify every consumer-visible alias."""

    with h5py.File(path, "r") as h5:
        reloaded = load_scalar_payload(h5)
        if expected_static is not None:
            assert reloaded.static_j is not None
            _assert_array_close("canonical static scalar", reloaded.static_j, expected_static)
            for scalar_path in _dataset_paths(h5, _STATIC_SCALAR_PATHS):
                _assert_array_close(scalar_path, np.asarray(h5[scalar_path]).reshape(-1), expected_static)
            if "J_r" in h5:
                for ib, value in enumerate(expected_static):
                    name = f"J_r/J_r_b{ib + 1}"
                    if name in h5:
                        _assert_array_close(name, np.asarray(h5[name]).reshape(()), np.asarray(value).reshape(()))
            for tensor_path in _dataset_paths(h5, (*_STATIC_ISO_TENSOR_PATHS, *_STATIC_FULL_TENSOR_PATHS)):
                arr = np.asarray(h5[tensor_path], dtype=np.float64)
                _assert_array_close(
                    f"trace({tensor_path})/3",
                    np.trace(arr, axis1=1, axis2=2) / 3.0,
                    expected_static,
                )

        if expected_dj is not None:
            assert original_payload.targets is not None and original_payload.rp is not None
            assert original_payload.axes is not None
            axis_to_xyz = np.asarray(
                [original_payload.axes.index(axis) for axis in ("x", "y", "z")], dtype=np.int64
            )
            for group_name in _DJ_SCALAR_GROUPS:
                if group_name in h5:
                    actual = _read_dj_group(
                        h5[group_name],
                        original_payload.targets,
                        len(original_payload.rp),
                        len(original_payload.gi),
                        axis_to_xyz,
                        tensor=False,
                    )
                    _assert_array_close(group_name, actual, expected_dj)
            for group_name in (*_DJ_ISO_TENSOR_GROUPS, *_DJ_FULL_TENSOR_GROUPS):
                if group_name in h5:
                    actual = _read_dj_group(
                        h5[group_name],
                        original_payload.targets,
                        len(original_payload.rp),
                        len(original_payload.gi),
                        axis_to_xyz,
                        tensor=True,
                    )
                    _assert_array_close(f"trace({group_name})/3", actual, expected_dj)
            assert reloaded.dj is not None
            _assert_array_close("canonical dJ scalar", reloaded.dj, expected_dj)


def _completion_status(h5: h5py.File, payload: ScalarPayload) -> dict[str, Any]:
    if "complete" in h5.attrs:
        complete_attr: bool | None = bool(int(h5.attrs["complete"]))
    else:
        complete_attr = None
    result: dict[str, Any] = {"complete_attribute": complete_attr}
    if payload.dj is not None:
        tensor_checkpoint_schema = bool(
            any(name in h5 for name in ("dA_r", "dJ_iso_r", "dJ_iso_tensor_r", "dJ_tensor_r"))
            or "n_completed_target_axis" in h5.attrs
        )
        legacy_scalar_schema = bool("dJ_r" in h5 and not tensor_checkpoint_schema)
        result["schema"] = "tensor_checkpoint" if tensor_checkpoint_schema else "legacy_scalar_dJ_r"
        result["required_for_dJ"] = tensor_checkpoint_schema
        # Explicit incomplete is always fatal.  Only the structurally complete
        # legacy /dJ_r schema may omit the historical checkpoint attribute.
        result["valid"] = complete_attr is not False and (
            complete_attr is True or legacy_scalar_schema
        )
        if payload.targets is not None and "n_completed_target_axis" in h5.attrs:
            got = int(h5.attrs["n_completed_target_axis"])
            expected = int(len(payload.targets) * 3)
            result.update({"n_completed_target_axis": got, "expected_target_axis": expected})
            result["valid"] = bool(result["valid"] and got == expected)
    else:
        result["required_for_dJ"] = False
        result["valid"] = complete_attr is not False
    return result


def _publish_noreplace(temp_path: Path, destination: Path) -> tuple[int, int]:
    """Atomically publish by hard link, which fails rather than clobbers."""

    try:
        os.link(temp_path, destination, follow_symlinks=False)
    except FileExistsError:
        raise FileExistsError(f"Refusing to overwrite concurrently created path: {destination}") from None
    except OSError as exc:
        raise RuntimeError(
            f"Atomic no-clobber publication is unavailable for {destination}: {exc}"
        ) from exc
    stat = destination.stat()
    temp_path.unlink()
    return int(stat.st_dev), int(stat.st_ino)


def _unlink_if_owned(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        stat = path.stat()
    except FileNotFoundError:
        return
    if (int(stat.st_dev), int(stat.st_ino)) == identity:
        path.unlink()


def _atomic_json(path: Path, report: dict[str, Any]) -> tuple[int, int]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite report JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(_jsonable(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return _publish_noreplace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _prepare(
    input_path: str | os.PathLike[str],
    epr_h5: str | os.PathLike[str],
    *,
    symprec: float,
    angle_tolerance: float,
    spin_pattern: Sequence[float] | None,
) -> tuple[ScalarPayload, list[OperationMap], np.ndarray, np.ndarray | None, dict[str, Any], dict[str, Any]]:
    embedded_nat = None
    embedded_geometry_kind = None
    embedded_labels = None
    with h5py.File(input_path, "r") as h5:
        payload = load_scalar_payload(h5)
        soc = detect_soc(h5)
        completion = _completion_status(h5, payload)
        if "basic_data/nat" in h5:
            embedded_nat = int(np.asarray(h5["basic_data/nat"]).reshape(()))
        if "basic_data/atom_labels" in h5:
            embedded_labels = [str(_decode(x)) for x in np.asarray(h5["basic_data/atom_labels"]).reshape(-1)]
        if all(path in h5 for path in ("basic_data/lattice_ang", "basic_data/tau_frac")):
            embedded_geometry_kind = "normalized_lattice_ang_tau_frac"
        elif all(
            path in h5 for path in ("basic_data/at", "basic_data/tau", "basic_data/alat", "basic_data/nat")
        ):
            embedded_geometry_kind = "raw_epr_at_tau"
    labels = None
    with h5py.File(epr_h5, "r") as epr:
        if "basic_data/atom_labels" in epr:
            labels = [str(_decode(x)) for x in np.asarray(epr["basic_data/atom_labels"]).reshape(-1)]
    lattice, tau, numbers, species_source = _read_epr_spglib_cell(str(epr_h5), labels=labels)
    if species_source == "all_same_fallback":
        raise ValueError(
            "Canonical EPR contains no reliable species metadata; refusing spglib projection with "
            "the all-same fallback"
        )
    if embedded_nat is not None and embedded_nat != len(tau):
        raise ValueError(f"Embedded result nat={embedded_nat} differs from canonical EPR nat={len(tau)}")
    species_order_check: dict[str, Any] = {
        "result_labels_available": embedded_labels is not None,
        "canonical_labels_available": labels is not None,
        "checked": False,
    }
    labels_are_generic = bool(
        embedded_labels is not None
        and all(re.fullmatch(r"(?:atom|site)\s*\d+", str(label).strip(), flags=re.IGNORECASE) for label in embedded_labels)
    )
    species_order_check["result_labels_generic"] = labels_are_generic
    if embedded_labels is not None and not labels_are_generic:
        result_numbers = _species_numbers_from_labels(embedded_labels, len(tau))
        if result_numbers is None:
            raise ValueError("Result atom_labels cannot be reconciled with canonical EPR nat")
        canonical_equivalence = np.asarray(numbers)[:, None] == np.asarray(numbers)[None, :]
        result_equivalence = result_numbers[:, None] == result_numbers[None, :]
        species_order_check["checked"] = True
        species_order_check["equivalence_partition_match"] = bool(
            np.array_equal(canonical_equivalence, result_equivalence)
        )
        if not species_order_check["equivalence_partition_match"]:
            raise ValueError(
                "Result atom-label species ordering disagrees with canonical EPR species ordering"
            )
    elif labels_are_generic:
        species_order_check["skipped_reason"] = "generic AtomN/SiteN labels carry no species information"
    embedded_geometry_check: dict[str, Any] = {
        "available": embedded_geometry_kind is not None,
        "kind": embedded_geometry_kind,
    }
    if embedded_geometry_kind is not None:
        if embedded_geometry_kind == "normalized_lattice_ang_tau_frac":
            with h5py.File(input_path, "r") as embedded_h5:
                lattice_raw = np.asarray(embedded_h5["basic_data/lattice_ang"], dtype=np.float64)
                embedded_tau = np.mod(
                    np.asarray(embedded_h5["basic_data/tau_frac"], dtype=np.float64), 1.0
                )
            if lattice_raw.shape != (3, 3) or embedded_tau.shape != tau.shape:
                raise ValueError("Embedded normalized lattice/tau arrays have incompatible shapes")
            direct_error = float(np.max(np.abs(lattice_raw - lattice)))
            transpose_error = float(np.max(np.abs(lattice_raw.T - lattice)))
            if transpose_error < direct_error:
                embedded_lattice = lattice_raw.T
                orientation = "transposed_to_canonical_rows"
            else:
                embedded_lattice = lattice_raw
                orientation = "canonical_rows"
            embedded_geometry_check["lattice_orientation"] = orientation
        else:
            embedded_lattice, embedded_tau, _embedded_numbers, _embedded_source = _read_epr_spglib_cell(
                str(input_path), labels=embedded_labels
            )
            orientation = "decoded_by_epr_convention"
            embedded_geometry_check["lattice_orientation"] = orientation
        lattice_error = float(np.max(np.abs(embedded_lattice - lattice)))
        tau_delta = embedded_tau - tau
        tau_delta -= np.rint(tau_delta)
        tau_error = float(np.max(np.abs(tau_delta)))
        embedded_geometry_check.update(
            {"lattice_max_abs_ang": lattice_error, "tau_fractional_max_abs": tau_error}
        )
        lattice_tol = max(float(symprec), 1.0e-8)
        if lattice_error > lattice_tol or tau_error > max(float(symprec), 1.0e-8):
            raise ValueError(
                "Embedded result geometry disagrees with canonical EPR: "
                f"lattice_max_abs={lattice_error:.3e} A, tau_max_abs={tau_error:.3e}"
            )
    if np.any(payload.gi < 0) or np.any(payload.gj < 0) or np.any(payload.gi >= len(tau)) or np.any(payload.gj >= len(tau)):
        raise ValueError("Bond atom indices lie outside the EPR structure")
    if payload.targets is not None and (
        np.any(payload.targets < 0) or np.any(payload.targets >= len(tau))
    ):
        raise ValueError("Displacement target indices lie outside the EPR structure")
    operations, symmetry_summary = build_symmetry_operations(
        lattice,
        tau,
        numbers,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        spin_pattern=spin_pattern,
    )
    maps = build_operation_maps(payload, operations)
    bond_mate, rp_mate, mate_diagnostics = build_mate_maps(payload)
    context = {
        "soc": soc,
        "completion": completion,
        "alias_consistency": payload.alias_consistency,
        "structure": {
            "nat": int(len(tau)),
            "species_source": species_source,
            "lattice_ang": lattice.tolist(),
            "embedded_geometry_check": embedded_geometry_check,
            "species_order_check": species_order_check,
        },
        "symmetry": symmetry_summary,
        "mapping_convention": {
            "atom": "W*tau_a+t=tau_p(a)+s_a",
            "bond": "(i,j,R)->(p_i,p_j,W*R+s_j-s_i)",
            "displacement": "(kappa,Rp)->(p_kappa,W*Rp+s_kappa-s_i)_mod_qmesh",
            "cartesian_gradient": "g' = Q*g; Q=L.T*W*inv(L.T)",
            "spin_flip_scalar_sign": "+1 (no minus)",
            "directed_mate": "(i,j,R;kappa,Rp)->(j,i,-R;kappa,Rp-R)",
        },
    }
    return payload, maps, bond_mate, rp_mate, mate_diagnostics, context


def audit_file(
    input_path: str | os.PathLike[str],
    epr_h5: str | os.PathLike[str],
    *,
    symprec: float = 1.0e-4,
    angle_tolerance: float = -1.0,
    spin_pattern: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Read-only covariance, reciprocity, and ASR audit."""

    payload, maps, bond_mate, rp_mate, mate_diag, context = _prepare(
        input_path,
        epr_h5,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        spin_pattern=spin_pattern,
    )
    report = {
        "schema_version": _SCHEMA_VERSION,
        "mode": "audit_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(input_path).resolve()),
        "epr_h5": str(Path(epr_h5).resolve()),
        **context,
        "before": audit_payload(
            payload,
            maps,
            bond_mate,
            rp_mate,
            mate_diag,
            nat=context["structure"]["nat"],
        ),
        "after": None,
    }
    return _jsonable(report)


def project_file(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    epr_h5: str | os.PathLike[str],
    *,
    symprec: float = 1.0e-4,
    angle_tolerance: float = -1.0,
    spin_pattern: Sequence[float] | None = None,
    enforce_asr: bool = True,
    allow_soc: bool = False,
    report_json: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Project into a new HDF5 file and return its before/after report."""

    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    if spin_pattern is None:
        raise ValueError("Projection requires an explicit collinear spin_pattern; audit-only may omit it")
    if allow_soc:
        raise ValueError("SOC override is disabled for writes; active SOC may only be inspected in audit-only mode")
    if source == output:
        raise ValueError("--output must be distinct from the input HDF5")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    report_path = Path(report_json).resolve() if report_json is not None else Path(str(output) + ".projection.json")
    if len({source, output, report_path}) != 3:
        raise ValueError("Input, output HDF5, and report JSON paths must be pairwise distinct")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report JSON: {report_path}")

    payload, maps, bond_mate, rp_mate, mate_diag, context = _prepare(
        source,
        epr_h5,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        spin_pattern=spin_pattern,
    )
    if not context["completion"]["valid"]:
        raise ValueError(f"Input is not marked complete: {context['completion']}")
    if context["soc"]["active"]:
        raise ValueError(
            "Active/nonempty SOC metadata detected. Scalar parent spin-group projection writes are "
            "forbidden; use audit-only for diagnostics."
        )
    nat = int(context["structure"]["nat"])
    _require_projectable(payload, maps, mate_diag, nat=nat, enforce_asr=enforce_asr)
    before = audit_payload(payload, maps, bond_mate, rp_mate, mate_diag, nat=nat)

    projected_static = None
    projected_dj = None
    correction_report: dict[str, Any] = {}
    if payload.static_j is not None:
        projected_static = project_static(payload.static_j, maps, bond_mate)
        correction_report["static_J_total"] = _relative_metrics(
            projected_static - payload.static_j, payload.static_j
        )
    if payload.dj is not None:
        assert rp_mate is not None
        projected_dj_before_asr = project_dj(
            payload.dj,
            maps,
            bond_mate,
            rp_mate,
            enforce_asr=False,
        )
        if enforce_asr:
            asr_uniform = -projected_dj_before_asr.sum(axis=(0, 1), keepdims=True) / float(
                projected_dj_before_asr.shape[0] * projected_dj_before_asr.shape[1]
            )
            projected_dj = projected_dj_before_asr + asr_uniform
        else:
            asr_uniform = np.zeros((1, 1, projected_dj_before_asr.shape[2], 3), dtype=np.float64)
            projected_dj = projected_dj_before_asr
        asr_total_correction = projected_dj - projected_dj_before_asr
        correction_report["dJ"] = {
            "symmetry_and_mate": _relative_metrics(
                projected_dj_before_asr - payload.dj, payload.dj
            ),
            "asr": {
                "applied": bool(enforce_asr),
                "method": "Euclidean minimum-norm uniform correction over complete (target,Rp) support",
                "uniform_per_target_rp_l2": float(np.linalg.norm(asr_uniform)),
                "uniform_per_target_rp_max_abs": float(np.max(np.abs(asr_uniform), initial=0.0)),
                "total_correction": _relative_metrics(
                    asr_total_correction, projected_dj_before_asr
                ),
                "residual_before": _audit_asr(projected_dj_before_asr),
                "residual_after": _audit_asr(projected_dj),
            },
            "total": _relative_metrics(projected_dj - payload.dj, payload.dj),
        }
    projected_payload = ScalarPayload(
        **{
            **payload.__dict__,
            "static_j": projected_static,
            "dj": projected_dj,
        }
    )
    after = audit_payload(projected_payload, maps, bond_mate, rp_mate, mate_diag, nat=nat)
    idempotency: dict[str, dict[str, Any]] = {}
    if projected_static is not None:
        projected_static_twice = project_static(projected_static, maps, bond_mate)
        idempotency["static_J"] = _relative_metrics(
            projected_static_twice - projected_static, projected_static
        )
    if projected_dj is not None:
        assert rp_mate is not None
        projected_dj_twice = project_dj(
            projected_dj,
            maps,
            bond_mate,
            rp_mate,
            enforce_asr=enforce_asr,
        )
        idempotency["dJ"] = _relative_metrics(projected_dj_twice - projected_dj, projected_dj)
    validation = _fail_closed_validation(
        after,
        projected_payload,
        enforce_asr=enforce_asr,
        idempotency=idempotency,
    )
    report: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "mode": "project",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source),
        "output": str(output),
        "epr_h5": str(Path(epr_h5).resolve()),
        **context,
        "options": {
            "symprec": float(symprec),
            "angle_tolerance": float(angle_tolerance),
            "enforce_asr": bool(enforce_asr),
            "allow_soc_override": False,
        },
        "before": before,
        "after": after,
        "projection_corrections": correction_report,
        "fail_closed_validation": validation,
        "updates": {},
        "full_tensor_policy": (
            "replace trace/3 by projected scalar using delta*I; preserve old traceless symmetric and "
            "antisymmetric components; raw dA_r and trace_acc untouched"
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    os.close(fd)
    temp_path = Path(temp_name)
    published_output_identity: tuple[int, int] | None = None
    try:
        shutil.copy2(source, temp_path)
        with h5py.File(temp_path, "r+") as h5:
            updates: dict[str, Any] = {}
            if projected_static is not None:
                updates["static_J"] = _write_static_values(h5, projected_static)
            if projected_dj is not None:
                updates["dJ"] = _write_dj_values(h5, payload, projected_dj)
            report["updates"] = updates
            encoded = json.dumps(_jsonable(report), sort_keys=True, separators=(",", ":"))
            h5.attrs["scalar_spin_group_projected"] = 1
            h5.attrs["scalar_spin_group_projection_schema"] = _SCHEMA_VERSION
            h5.attrs["scalar_spin_group_projection_source"] = str(source)
            h5.attrs["scalar_spin_group_projection_epr_h5"] = str(Path(epr_h5).resolve())
            h5.attrs["scalar_spin_group_projection_n_operations"] = len(maps)
            h5.attrs["scalar_spin_group_projection_enforce_asr"] = int(bool(enforce_asr))
            h5.attrs["scalar_spin_group_projection_allow_soc_override"] = 0
            h5.attrs["scalar_spin_group_projection_full_tensor"] = "scalar_trace_only;traceless_untouched"
            if "scalar_spin_group_projection_report_json" in h5.attrs:
                del h5.attrs["scalar_spin_group_projection_report_json"]
            provenance = h5.require_group("scalar_spin_group_projection")
            if "report_json" in provenance:
                del provenance["report_json"]
            provenance.create_dataset(
                "report_json",
                data=np.array(encoded, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            h5.attrs["scalar_spin_group_projection_report"] = "/scalar_spin_group_projection/report_json"
            h5.flush()
        _verify_written_file(temp_path, payload, projected_static, projected_dj)
        published_output_identity = _publish_noreplace(temp_path, output)
        _atomic_json(report_path, report)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        _unlink_if_owned(output, published_output_identity)
        raise
    return _jsonable(report)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit or non-destructively project scalar J_iso/dJ_iso under SOC-off spin-group symmetry."
    )
    parser.add_argument("input", help="Input static-J and/or dJ HDF5")
    parser.add_argument("--epr-h5", required=True, help="EPR HDF5 supplying lattice, positions, and species")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", help="New projected HDF5; existing paths are refused")
    mode.add_argument("--audit-only", action="store_true", help="Read-only audit; no HDF5 is written")
    parser.add_argument("--report-json", default=None, help="JSON report path (default: OUTPUT.projection.json)")
    parser.add_argument("--symprec", type=float, default=1.0e-4)
    parser.add_argument("--angle-tolerance", type=float, default=-1.0)
    parser.add_argument(
        "--spin-pattern",
        type=float,
        nargs="+",
        default=None,
        help="One collinear scalar per crystal atom; retain operations mapping it to itself or its global negative",
    )
    parser.add_argument(
        "--no-asr",
        dest="enforce_asr",
        action="store_false",
        help="Do not impose sum_(kappa,Rp) dJ=0 (ASR is enabled by default)",
    )
    parser.set_defaults(enforce_asr=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.audit_only:
        report = audit_file(
            args.input,
            args.epr_h5,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            spin_pattern=args.spin_pattern,
        )
        if args.report_json is not None:
            _atomic_json(Path(args.report_json).resolve(), report)
    else:
        report = project_file(
            args.input,
            args.output,
            args.epr_h5,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            spin_pattern=args.spin_pattern,
            enforce_asr=args.enforce_asr,
            report_json=args.report_json,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
