"""Reusable complex128 Green matrices (WT-E01)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import require_complex128


def green_matrix(hamiltonian: object, z_eV: complex) -> NDArray[np.complex128]:
    h = require_complex128("Hamiltonian", hamiltonian)
    if h.ndim < 2 or h.shape[-1] != h.shape[-2]:
        raise ValueError("Hamiltonian must contain square matrices")
    identity = np.eye(h.shape[-1], dtype=np.complex128)
    return np.asarray(np.linalg.inv(complex(z_eV) * identity - h), dtype=np.complex128)


def green_from_eigendecomposition(
    eigenvalues: object,
    eigenvectors: object,
    z_eV: complex,
) -> NDArray[np.complex128]:
    values = np.asarray(eigenvalues, dtype=np.float64)
    vectors = require_complex128("eigenvectors", eigenvectors)
    if vectors.shape[:-1] != values.shape or vectors.shape[-2] != values.shape[-1]:
        raise ValueError("eigenvalue/eigenvector shapes are incompatible")
    inverse = 1.0 / (complex(z_eV) - values)
    return np.asarray((vectors * inverse[..., None, :]) @ np.swapaxes(vectors.conj(), -1, -2), dtype=np.complex128)


__all__ = ["green_from_eigendecomposition", "green_matrix"]

