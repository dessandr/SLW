"""Gauge-invariant spectra and subspace diagnostics under mode rephasing."""

from __future__ import annotations

import numpy as np

from slw.wtorque.basis import require_complex128


def rephase_modes(value: object, phases: object, *, axis: int = -1) -> np.ndarray:
    array = require_complex128("mode array", value)
    phase = np.asarray(phases, dtype=np.float64)
    if array.shape[axis] != phase.size:
        raise ValueError("phase count does not match selected mode axis")
    shape = [1] * array.ndim
    shape[axis] = phase.size
    return np.asarray(array * np.exp(1j * phase).reshape(shape), dtype=np.complex128)


def spectral_norm_invariant(value: object) -> float:
    array = require_complex128("coupling", value)
    return float(np.real(np.vdot(array, array)))


__all__ = ["rephase_modes", "spectral_norm_invariant"]

