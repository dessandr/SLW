"""Reusable Hermiticity, frame, and decomposition residuals."""

from __future__ import annotations

import numpy as np

from slw.wtorque.basis import require_complex128
from slw.wtorque.errors import HermiticityError


def hermiticity_residual(value: object) -> float:
    matrix = require_complex128("matrix", value)
    return float(np.max(np.abs(matrix - np.swapaxes(matrix.conj(), -1, -2)), initial=0.0))


def require_hermitian(value: object, *, tolerance: float, label: str) -> float:
    residual = hermiticity_residual(value)
    if residual > tolerance:
        raise HermiticityError(
            f"{label} Hermiticity residual {residual:.3e} exceeds {tolerance:.3e}"
        )
    return residual


__all__ = ["hermiticity_residual", "require_hermitian"]

