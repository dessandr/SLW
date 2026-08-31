"""Magnetic-subspace masks, site positions, local frames, and spin lengths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.wtorque.config import MagneticSubspaceConfig
from slw.wtorque.errors import MagneticSubspaceError
from slw.wtorque.model.local_frames import validate_local_frames
from slw.wtorque.model.magnetic_subspace import MagneticSubspace


@dataclass(frozen=True)
class MagneticData:
    subspace: MagneticSubspace
    atom_indices: NDArray[np.int64]
    site_positions: NDArray[np.float64]
    local_frames: NDArray[np.float64]
    spin_lengths: NDArray[np.float64] | None


def load_magnetic_data(
    path: str | Path,
    config: MagneticSubspaceConfig,
    *,
    norb: int,
    orbital_labels: tuple[str, ...],
) -> MagneticData:
    masks = np.zeros((len(config.sites), norb), dtype=bool)
    atoms = np.empty(len(config.sites), dtype=np.int64)
    for index, site in enumerate(config.sites):
        if any(orbital < 0 or orbital >= norb for orbital in site.orbitals):
            raise MagneticSubspaceError(
                f"site {index} contains an orbital outside [0,{norb})"
            )
        masks[index, list(site.orbitals)] = True
        atoms[index] = site.atom
    subspace = MagneticSubspace.from_masks(masks, orbital_labels=orbital_labels)
    with h5py.File(path, "r") as handle:
        if "spin" not in handle:
            raise MagneticSubspaceError("electronic input must contain /spin metadata")
        group = handle["spin"]
        required = ("magnetic_atom_index", "magnetic_site_position", "local_frames")
        missing = [name for name in required if name not in group]
        if missing:
            raise MagneticSubspaceError(f"/spin is missing {', '.join(missing)}")
        stored_atoms = np.asarray(group["magnetic_atom_index"][...], dtype=np.int64)
        if stored_atoms.shape != atoms.shape or not np.array_equal(stored_atoms, atoms):
            raise MagneticSubspaceError(
                "configured magnetic site atoms do not match /spin/magnetic_atom_index"
            )
        if "magnetic_orbital_mask" in group:
            stored_masks = np.asarray(group["magnetic_orbital_mask"][...], dtype=bool)
            if stored_masks.shape != masks.shape or not np.array_equal(stored_masks, masks):
                raise MagneticSubspaceError(
                    "configured orbital masks do not match /spin/magnetic_orbital_mask"
                )
        positions = np.asarray(group["magnetic_site_position"][...])
        if positions.dtype != np.float64 or positions.shape != (len(config.sites), 3):
            raise MagneticSubspaceError(
                "/spin/magnetic_site_position must be float64 (nmag,3) reduced coordinates"
            )
        frames_raw = np.asarray(group["local_frames"][...])
        if frames_raw.dtype != np.float64:
            raise MagneticSubspaceError("/spin/local_frames must use float64")
        frames = validate_local_frames(frames_raw)
        spins = None
        if "spin_length" in group:
            spins = np.asarray(group["spin_length"][...])
            if spins.dtype != np.float64 or spins.shape != (len(config.sites),) or np.any(spins <= 0):
                raise MagneticSubspaceError("/spin/spin_length must be positive float64 [nmag]")
    return MagneticData(subspace, atoms, positions, frames, spins)


__all__ = ["MagneticData", "load_magnetic_data"]

