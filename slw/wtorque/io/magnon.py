"""Externally supplied magnon paraunitary HDF5 data boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import require_complex128
from slw.wtorque.errors import GaugeMismatchError, ParaunitarityError
from slw.wtorque.model.local_frames import validate_local_frames
from slw.wtorque.projection.magnon import paraunitarity_residual


@dataclass(frozen=True)
class MagnonQData:
    energies_eV: NDArray[np.float64]
    transform: NDArray[np.complex128]


class HDF5MagnonProvider:
    def __init__(
        self,
        path: str | Path,
        *,
        expected_frames: object,
        expected_spin_lengths: object,
        tolerance: float = 1.0e-9,
    ) -> None:
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        if "magnon" not in self.handle:
            self.close()
            raise KeyError("magnon input must contain /magnon")
        group = self.handle["magnon"]
        required = ("qpoints", "energy", "T_para", "metric", "local_frames", "spin_length")
        missing = [name for name in required if name not in group]
        if missing:
            self.close()
            raise KeyError(f"/magnon is missing {', '.join(missing)}")
        self.qpoints = np.asarray(group["qpoints"][...])
        self.energies = np.asarray(group["energy"][...])
        self.transforms = require_complex128("/magnon/T_para", group["T_para"][...])
        self.metric = np.asarray(group["metric"][...])
        self.frames = validate_local_frames(group["local_frames"][...])
        self.spin_lengths = np.asarray(group["spin_length"][...])
        nmag = self.frames.shape[0]
        if self.qpoints.dtype != np.float64 or self.qpoints.ndim != 2 or self.qpoints.shape[1] != 3:
            self.close()
            raise TypeError("/magnon/qpoints must be float64 [nq,3]")
        if self.energies.dtype != np.float64 or self.energies.shape != (self.qpoints.shape[0], nmag):
            self.close()
            raise TypeError("/magnon/energy must be float64 [nq,nbranch]")
        if self.transforms.shape != (self.qpoints.shape[0], 2 * nmag, 2 * nmag):
            self.close()
            raise ValueError("/magnon/T_para must have shape [nq,2*nmag,2*nmag]")
        expected_metric = np.r_[np.ones(nmag), -np.ones(nmag)]
        if self.metric.shape != expected_metric.shape or not np.array_equal(self.metric, expected_metric):
            self.close()
            raise ParaunitarityError("/magnon/metric or Nambu order is not [I,-I]")
        expected_frame_array = np.asarray(expected_frames, dtype=np.float64)
        expected_spins = np.asarray(expected_spin_lengths, dtype=np.float64)
        if self.frames.shape != expected_frame_array.shape or not np.allclose(
            self.frames, expected_frame_array, atol=tolerance
        ):
            self.close()
            raise GaugeMismatchError("magnon local frames differ from the electronic local frames")
        if self.spin_lengths.shape != expected_spins.shape or not np.allclose(
            self.spin_lengths, expected_spins, atol=tolerance
        ):
            self.close()
            raise GaugeMismatchError("magnon spin lengths differ from the electronic input")
        residuals = np.array([paraunitarity_residual(value) for value in self.transforms])
        if np.any(residuals > tolerance):
            self.close()
            raise ParaunitarityError(
                f"magnon input paraunitarity residual reaches {residuals.max():.3e}"
            )
        self.paraunitarity_residuals = residuals

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            self.handle.close()

    def q(self, iq: int) -> MagnonQData:
        return MagnonQData(
            np.asarray(self.energies[int(iq)], dtype=np.float64),
            np.asarray(self.transforms[int(iq)], dtype=np.complex128),
        )


__all__ = ["HDF5MagnonProvider", "MagnonQData"]
