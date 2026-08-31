"""Exchange-field extraction routes (WT-E02 through WT-E04)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import require_complex128
from slw.wtorque.errors import ExchangeDecompositionError

from .time_reversal import time_reversed


def _check_reconstruction(
    full: NDArray[np.complex128],
    trs: NDArray[np.complex128],
    xc: NDArray[np.complex128],
    tolerance: float,
) -> None:
    residual = float(np.max(np.abs(full - trs - xc)))
    if residual > tolerance:
        raise ExchangeDecompositionError(
            f"H_TRS + H_XC reconstruction residual {residual:.3e} exceeds {tolerance:.3e}"
        )


def split_explicit(
    full: object,
    h_trs: object,
    h_xc: object,
    *,
    tolerance: float = 1.0e-10,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    h = require_complex128("H", full)
    trs = require_complex128("H_TRS", h_trs)
    xc = require_complex128("H_XC", h_xc)
    if h.shape != trs.shape or h.shape != xc.shape:
        raise ExchangeDecompositionError("explicit H, H_TRS, and H_XC shapes must match")
    _check_reconstruction(h, trs, xc, tolerance)
    return trs, xc


def split_time_reversal(
    h_k: object,
    h_minus_k: object,
    sewing_k: object,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Separate TR-even and TR-odd fields in the same k-basis (WT-E04)."""

    h = require_complex128("H_k", h_k)
    transformed = time_reversed(h_minus_k, sewing_k)
    if transformed.shape != h.shape:
        raise ExchangeDecompositionError("time-reversed Hamiltonian shape mismatch")
    trs = np.asarray(0.5 * (h + transformed), dtype=np.complex128)
    xc = np.asarray(0.5 * (h - transformed), dtype=np.complex128)
    _check_reconstruction(h, trs, xc, 1.0e-12)
    return trs, xc


def split_collinear(
    h_up: object,
    h_down: object,
    reference_direction: object,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Lift common-gauge collinear blocks into interleaved spinor order."""

    up = require_complex128("H_up", h_up)
    down = require_complex128("H_down", h_down)
    if up.shape != down.shape or up.shape[-1] != up.shape[-2]:
        raise ExchangeDecompositionError("H_up and H_down must have matching square matrices")
    direction = np.asarray(reference_direction, dtype=np.float64)
    if direction.shape != (3,) or not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-10):
        raise ExchangeDecompositionError("reference_direction must be a unit 3-vector")
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sigma_n = np.einsum("i,iab->ab", direction, np.stack((sx, sy, sz)))
    average = 0.5 * (up + down)
    difference = 0.5 * (up - down)
    leading = up.shape[:-2]
    norb = up.shape[-1]
    trs = np.einsum("...ij,ab->...iajb", average, np.eye(2)).reshape(
        *leading, 2 * norb, 2 * norb
    )
    xc = np.einsum("...ij,ab->...iajb", difference, sigma_n).reshape(
        *leading, 2 * norb, 2 * norb
    )
    return np.asarray(trs, dtype=np.complex128), np.asarray(xc, dtype=np.complex128)


__all__ = ["split_collinear", "split_explicit", "split_time_reversal"]

