"""Small Green-function kernels shared by Wannier and EPR workflows."""

from __future__ import annotations

import numpy as np


def green_subblock_from_coefficients(
    coefficients_i: np.ndarray,
    coefficients_j: np.ndarray,
    inverse_denominator: np.ndarray,
) -> np.ndarray:
    """Construct ``G_ij = C_i diag(d) C_j^H`` without a dense diagonal matrix."""
    coefficients_i = np.asarray(coefficients_i, dtype=np.complex128)
    coefficients_j = np.asarray(coefficients_j, dtype=np.complex128)
    inverse_denominator = np.asarray(inverse_denominator, dtype=np.complex128)
    if coefficients_i.shape[-1] != inverse_denominator.shape[0]:
        raise ValueError(
            "Coefficient/eigenvalue mismatch: "
            f"{coefficients_i.shape[-1]} != {inverse_denominator.shape[0]}"
        )
    if coefficients_j.shape[-1] != inverse_denominator.shape[0]:
        raise ValueError(
            "Coefficient/eigenvalue mismatch: "
            f"{coefficients_j.shape[-1]} != {inverse_denominator.shape[0]}"
        )
    return (coefficients_i * inverse_denominator[None, :]) @ coefficients_j.conj().T
