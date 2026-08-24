"""Space-group covariance screening for scalar exchange derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import spglib
from numpy.typing import NDArray

from .j_epr import _read_epr_spglib_cell


@dataclass(frozen=True)
class CovariantDerivativeSymmetryResult:
    """Canonical derivative and compact space-group projection provenance."""

    values_mev_per_ang: NDArray[np.float64]
    raw_values_mev_per_ang: NDArray[np.float64]
    policy: str
    tolerance_mev_per_ang: float
    applied: bool
    max_abs_residual_mev_per_ang: float
    rms_residual_mev_per_ang: float
    n_operations: int
    spacegroup: str
    species_source: str


def _spacegroup_label(dataset: Any) -> str:
    if dataset is None:
        return "unknown"
    try:
        return f"{dataset.international} #{int(dataset.number)}"
    except (AttributeError, TypeError, ValueError):
        try:
            return f"{dataset['international']} #{int(dataset['number'])}"
        except (KeyError, TypeError, ValueError):
            return "unknown"


def _atom_maps(
    positions_frac: NDArray[np.float64],
    lattice_ang: NDArray[np.float64],
    species_numbers: NDArray[np.int32],
    rotations: NDArray[np.int64],
    translations: NDArray[np.float64],
    *,
    symprec: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    n_operations = rotations.shape[0]
    nat = positions_frac.shape[0]
    mapped = np.empty((n_operations, nat), dtype=np.int64)
    shifts = np.empty((n_operations, nat, 3), dtype=np.int64)
    for operation, (rotation, translation) in enumerate(
        zip(rotations, translations, strict=True)
    ):
        raw = positions_frac @ rotation.T + translation[None, :]
        for atom in range(nat):
            candidates = np.flatnonzero(species_numbers == species_numbers[atom])
            delta = raw[atom][None, :] - positions_frac[candidates]
            delta -= np.rint(delta)
            distances = np.linalg.norm(delta @ lattice_ang, axis=1)
            nearest_slot = int(np.argmin(distances))
            if float(distances[nearest_slot]) > symprec:
                raise ValueError(
                    "space-group operation does not map an atom within symprec: "
                    f"operation={operation} atom={atom} "
                    f"distance={float(distances[nearest_slot]):.6g} angstrom"
                )
            destination = int(candidates[nearest_slot])
            mapped[operation, atom] = destination
            shifts[operation, atom] = np.rint(
                raw[atom] - positions_frac[destination]
            ).astype(np.int64)
        if np.unique(mapped[operation]).size != nat:
            raise ValueError(
                f"space-group atom map is not bijective for operation {operation}"
            )
    return mapped, shifts


def apply_covariant_derivative_symmetry(
    raw_values_mev_per_ang: object,
    *,
    bond_i_atom: object,
    bond_j_atom: object,
    bond_cell_shifts: object,
    target_atom_indices: object,
    rp_cell_shifts: object,
    q_mesh_shape: tuple[int, int, int],
    lattice_ang: object,
    positions_frac: object,
    species_numbers: object,
    rotations: object,
    translations: object,
    policy: str = "project",
    tolerance_mev_per_ang: float = 1.0e-8,
    symprec: float = 1.0e-4,
    spacegroup: str = "unknown",
    species_source: str = "",
) -> CovariantDerivativeSymmetryResult:
    """Apply the Reynolds projector for ``dJ(i,j,R)/du(kappa,Rp)``.

    The derivative is a Cartesian polar vector. Each operation therefore maps
    the target atom, directed bond, periodic ``Rp`` cell, and vector component
    together. The source arrays use shape ``(target,bond,Rp,3)``.
    """

    selected_policy = str(policy).strip().lower()
    if selected_policy not in {"none", "report", "project", "fail"}:
        raise ValueError(
            "covariant_symmetry must be one of none, report, project, or fail"
        )
    tolerance = float(tolerance_mev_per_ang)
    symmetry_tolerance = float(symprec)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "covariant_symmetry_tolerance_mev_per_ang must be finite and non-negative"
        )
    if not np.isfinite(symmetry_tolerance) or symmetry_tolerance <= 0.0:
        raise ValueError("symprec must be finite and positive")

    raw = np.asarray(raw_values_mev_per_ang, dtype=np.float64)
    bond_i = np.asarray(bond_i_atom, dtype=np.int64)
    bond_j = np.asarray(bond_j_atom, dtype=np.int64)
    bond_r = np.asarray(bond_cell_shifts, dtype=np.int64)
    targets = np.asarray(target_atom_indices, dtype=np.int64)
    rp = np.asarray(rp_cell_shifts, dtype=np.int64)
    lattice = np.asarray(lattice_ang, dtype=np.float64)
    positions = np.mod(np.asarray(positions_frac, dtype=np.float64), 1.0)
    numbers = np.asarray(species_numbers, dtype=np.int32)
    rotation_array = np.asarray(rotations, dtype=np.int64)
    translation_array = np.asarray(translations, dtype=np.float64)
    mesh = np.asarray(q_mesh_shape, dtype=np.int64)

    n_targets = targets.size
    n_bonds = bond_i.size
    n_rp = rp.shape[0]
    if raw.shape != (n_targets, n_bonds, n_rp, 3):
        raise ValueError(
            "raw derivative must have shape "
            f"{(n_targets, n_bonds, n_rp, 3)}, got {raw.shape}"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw derivative contains non-finite values")
    if bond_j.shape != (n_bonds,) or bond_r.shape != (n_bonds, 3):
        raise ValueError("directed bond arrays have inconsistent shapes")
    if targets.ndim != 1 or targets.size == 0 or np.unique(targets).size != targets.size:
        raise ValueError("target_atom_indices must be a nonempty unique vector")
    if rp.ndim != 2 or rp.shape[1:] != (3,):
        raise ValueError("rp_cell_shifts must have shape (nRp,3)")
    if mesh.shape != (3,) or np.any(mesh <= 0) or int(np.prod(mesh)) != n_rp:
        raise ValueError("q_mesh_shape must match the complete periodic Rp cell")
    if lattice.shape != (3, 3) or abs(float(np.linalg.det(lattice))) <= 1.0e-12:
        raise ValueError("lattice_ang must be a nonsingular 3x3 matrix")
    nat = positions.shape[0]
    if positions.shape != (nat, 3) or numbers.shape != (nat,):
        raise ValueError("positions/species arrays have inconsistent atom counts")
    if (
        np.any(bond_i < 0)
        or np.any(bond_i >= nat)
        or np.any(bond_j < 0)
        or np.any(bond_j >= nat)
        or np.any(targets < 0)
        or np.any(targets >= nat)
    ):
        raise ValueError("bond or target atom index exceeds the structure")
    if rotation_array.ndim != 3 or rotation_array.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (n_operation,3,3)")
    if translation_array.shape != (rotation_array.shape[0], 3):
        raise ValueError("translations must have shape (n_operation,3)")
    if rotation_array.shape[0] == 0:
        raise ValueError("at least one space-group operation is required")

    raw_copy = np.array(raw, copy=True)
    if selected_policy == "none":
        return CovariantDerivativeSymmetryResult(
            values_mev_per_ang=raw_copy,
            raw_values_mev_per_ang=raw_copy.copy(),
            policy=selected_policy,
            tolerance_mev_per_ang=tolerance,
            applied=False,
            max_abs_residual_mev_per_ang=0.0,
            rms_residual_mev_per_ang=0.0,
            n_operations=0,
            spacegroup="not_evaluated",
            species_source=str(species_source),
        )

    atom_map, atom_shift = _atom_maps(
        positions,
        lattice,
        numbers,
        rotation_array,
        translation_array,
        symprec=symmetry_tolerance,
    )
    target_lookup = {int(atom): slot for slot, atom in enumerate(targets)}
    bond_lookup = {
        (int(i), int(j), int(cell[0]), int(cell[1]), int(cell[2])): slot
        for slot, (i, j, cell) in enumerate(zip(bond_i, bond_j, bond_r, strict=True))
    }
    if len(bond_lookup) != n_bonds:
        raise ValueError("directed derivative bond keys must be unique")
    rp_mod = np.mod(rp, mesh[None, :])
    rp_codes = (rp_mod[:, 0] * mesh[1] + rp_mod[:, 1]) * mesh[2] + rp_mod[:, 2]
    if np.unique(rp_codes).size != n_rp:
        raise ValueError("Rp cells must form one complete periodic q-mesh cell")
    rp_lookup = np.empty(n_rp, dtype=np.int64)
    rp_lookup[rp_codes] = np.arange(n_rp, dtype=np.int64)

    projected_flat = np.zeros((raw.size // 3, 3), dtype=np.float64)
    raw_flat = raw.reshape(-1, 3)
    inverse_lattice_t = np.linalg.inv(lattice.T)
    for operation, rotation in enumerate(rotation_array):
        cartesian_rotation = lattice.T @ rotation @ inverse_lattice_t
        if not np.allclose(
            cartesian_rotation.T @ cartesian_rotation,
            np.eye(3),
            atol=1.0e-8,
            rtol=1.0e-8,
        ):
            raise ValueError(
                f"operation {operation} does not yield an orthogonal Cartesian rotation"
            )
        try:
            target_destination = np.asarray(
                [target_lookup[int(atom_map[operation, atom])] for atom in targets],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                "derivative target set is not closed under the detected space group"
            ) from exc
        if np.unique(target_destination).size != n_targets:
            raise ValueError(
                f"derivative target map is not bijective for operation {operation}"
            )

        bond_destination = np.empty(n_bonds, dtype=np.int64)
        origin_shifts = atom_shift[operation, bond_i]
        for bond in range(n_bonds):
            destination_i = int(atom_map[operation, bond_i[bond]])
            destination_j = int(atom_map[operation, bond_j[bond]])
            destination_r = (
                rotation @ bond_r[bond]
                + atom_shift[operation, bond_j[bond]]
                - origin_shifts[bond]
            )
            key = (
                destination_i,
                destination_j,
                int(destination_r[0]),
                int(destination_r[1]),
                int(destination_r[2]),
            )
            try:
                bond_destination[bond] = bond_lookup[key]
            except KeyError as exc:
                raise ValueError(
                    "derivative bond set is not closed under the detected space group: "
                    f"operation={operation} mapped_key={key}"
                ) from exc
        if np.unique(bond_destination).size != n_bonds:
            raise ValueError(
                f"derivative bond map is not bijective for operation {operation}"
            )

        rotated_rp = rp @ rotation.T
        target_shifts = atom_shift[operation, targets]
        destination_rp = (
            rotated_rp[None, None, :, :]
            + target_shifts[:, None, None, :]
            - origin_shifts[None, :, None, :]
        )
        destination_rp = np.mod(destination_rp, mesh[None, None, None, :])
        destination_codes = (
            (destination_rp[..., 0] * mesh[1] + destination_rp[..., 1]) * mesh[2]
            + destination_rp[..., 2]
        )
        rp_destination = rp_lookup[destination_codes]
        if np.unique(rp_destination[0, 0]).size != n_rp:
            raise ValueError(
                "q mesh is incompatible with a detected space-group operation: "
                f"operation={operation} qmesh={tuple(int(value) for value in mesh)}"
            )
        flat_destination = (
            (target_destination[:, None, None] * n_bonds)
            + bond_destination[None, :, None]
        ) * n_rp + rp_destination
        flat_destination = flat_destination.reshape(-1)
        projected_flat[flat_destination] += raw_flat @ cartesian_rotation.T

    projected = (
        projected_flat / float(rotation_array.shape[0])
    ).reshape(raw.shape)
    residual = projected - raw
    max_residual = float(np.max(np.abs(residual), initial=0.0))
    rms_residual = float(np.sqrt(np.mean(np.square(residual))))
    if selected_policy == "fail" and max_residual > tolerance:
        raise ValueError(
            "scalar dJ/du violates space-group covariance: "
            f"maximum residual={max_residual:.6g} meV/angstrom exceeds "
            f"{tolerance:.6g}"
        )
    values = projected if selected_policy == "project" else raw_copy
    values.setflags(write=False)
    raw_copy.setflags(write=False)
    return CovariantDerivativeSymmetryResult(
        values_mev_per_ang=values,
        raw_values_mev_per_ang=raw_copy,
        policy=selected_policy,
        tolerance_mev_per_ang=tolerance,
        applied=selected_policy == "project",
        max_abs_residual_mev_per_ang=max_residual,
        rms_residual_mev_per_ang=rms_residual,
        n_operations=int(rotation_array.shape[0]),
        spacegroup=str(spacegroup),
        species_source=str(species_source),
    )


def project_epr_scalar_derivative(
    epr_path: str | Path,
    raw_values_mev_per_ang: object,
    *,
    bond_i_atom: object,
    bond_j_atom: object,
    bond_cell_shifts: object,
    target_atom_indices: object,
    rp_cell_shifts: object,
    q_mesh_shape: tuple[int, int, int],
    labels: list[str] | tuple[str, ...] | None,
    species_labels: list[str] | tuple[str, ...] | None,
    policy: str,
    tolerance_mev_per_ang: float,
    symprec: float,
    angle_tolerance: float,
) -> CovariantDerivativeSymmetryResult:
    """Discover EPR structure symmetry and apply the native projector."""

    selected_policy = str(policy).strip().lower()
    if selected_policy == "none":
        tolerance = float(tolerance_mev_per_ang)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "covariant_symmetry_tolerance_mev_per_ang must be finite and "
                "non-negative"
            )
        raw = np.array(raw_values_mev_per_ang, dtype=np.float64, copy=True)
        if raw.ndim != 4 or raw.shape[-1] != 3 or not np.all(np.isfinite(raw)):
            raise ValueError("raw derivative must have finite shape (target,bond,Rp,3)")
        raw.setflags(write=False)
        raw_provenance = raw.copy()
        raw_provenance.setflags(write=False)
        return CovariantDerivativeSymmetryResult(
            values_mev_per_ang=raw,
            raw_values_mev_per_ang=raw_provenance,
            policy=selected_policy,
            tolerance_mev_per_ang=tolerance,
            applied=False,
            max_abs_residual_mev_per_ang=0.0,
            rms_residual_mev_per_ang=0.0,
            n_operations=0,
            spacegroup="not_evaluated",
            species_source="not_evaluated",
        )

    lattice, positions, numbers, species_source = _read_epr_spglib_cell(
        epr_path,
        labels=labels,
        species_labels=species_labels,
    )
    cell = (lattice, positions, numbers)
    operations = spglib.get_symmetry(
        cell,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    dataset = spglib.get_symmetry_dataset(
        cell,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    if operations is None or len(operations.get("rotations", ())) == 0:
        raise ValueError("spglib returned no operations for dJ/du covariance")
    return apply_covariant_derivative_symmetry(
        raw_values_mev_per_ang,
        bond_i_atom=bond_i_atom,
        bond_j_atom=bond_j_atom,
        bond_cell_shifts=bond_cell_shifts,
        target_atom_indices=target_atom_indices,
        rp_cell_shifts=rp_cell_shifts,
        q_mesh_shape=q_mesh_shape,
        lattice_ang=lattice,
        positions_frac=positions,
        species_numbers=numbers,
        rotations=np.asarray(operations["rotations"], dtype=np.int64),
        translations=np.asarray(operations["translations"], dtype=np.float64),
        policy=selected_policy,
        tolerance_mev_per_ang=tolerance_mev_per_ang,
        symprec=symprec,
        spacegroup=_spacegroup_label(dataset),
        species_source=str(species_source or ""),
    )


__all__ = [
    "CovariantDerivativeSymmetryResult",
    "apply_covariant_derivative_symmetry",
    "project_epr_scalar_derivative",
]
