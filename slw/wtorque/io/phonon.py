"""Phonon q-group HDF5 data boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import require_complex128


@dataclass(frozen=True)
class PhononQData:
    frequencies_eV: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]


class HDF5PhononProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        if "phonon" not in self.handle:
            self.close()
            raise KeyError("phonon input must contain /phonon")
        group = self.handle["phonon"]
        mass_name = "mass" if "mass" in group else "masses" if "masses" in group else None
        if mass_name is None:
            self.close()
            raise KeyError("phonon input must contain /phonon/mass in amu")
        self.masses_amu = np.asarray(group[mass_name][...])
        if self.masses_amu.dtype != np.float64 or self.masses_amu.ndim != 1 or np.any(self.masses_amu <= 0):
            self.close()
            raise TypeError("/phonon/mass must be positive float64 [natom] in amu")
        self.qpoints = None
        self.mass_weighted_mode_scale = None
        if "mass_weighted_mode_scale" in group:
            self.mass_weighted_mode_scale = np.asarray(
                group["mass_weighted_mode_scale"][...], dtype=np.float64
            )
        if "qpoints" in group:
            self.qpoints = np.asarray(group["qpoints"][...])
            if self.qpoints.dtype != np.float64 or self.qpoints.ndim != 2 or self.qpoints.shape[1] != 3:
                self.close()
                raise TypeError("/phonon/qpoints must be float64 [nq,3]")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            self.handle.close()

    def q(self, iq: int) -> PhononQData:
        name = f"phonon/q_{int(iq):06d}"
        if name not in self.handle:
            raise KeyError(f"phonon input is missing /{name}")
        group = self.handle[name]
        if "frequency" not in group or "eigenvector" not in group:
            raise KeyError(f"/{name} requires frequency and eigenvector")
        frequency = np.asarray(group["frequency"][...])
        if frequency.dtype != np.float64 or frequency.ndim != 1:
            raise TypeError(f"/{name}/frequency must be float64 [nmode] in eV")
        vectors = require_complex128(f"/{name}/eigenvector", group["eigenvector"][...])
        if vectors.shape != (self.masses_amu.size, 3, frequency.size):
            raise ValueError(f"/{name}/eigenvector has incompatible dimensions")
        return PhononQData(frequency, vectors)


__all__ = ["HDF5PhononProvider", "PhononQData"]
