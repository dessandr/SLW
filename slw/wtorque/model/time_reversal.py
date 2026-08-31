"""Time-reversal sewing operations for arbitrary spinor Wannier gauges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import SpinOrder, require_complex128
from slw.wtorque.errors import TimeReversalMetadataError


@dataclass(frozen=True)
class ProjectionAnchoredSewing:
    """Polar atomic frame and time-reversal sewing in a Wannier gauge."""

    atomic_frame: NDArray[np.complex128]
    sewing: NDArray[np.complex128]
    singular_values: NDArray[np.float64]


def atomic_spin_time_reversal(
    dimension: int,
    spin_order: SpinOrder | str,
) -> NDArray[np.complex128]:
    """Return the real-orbital spin-1/2 unitary ``I_orb x i sigma_y``.

    This standard identity assumes real spatial trial orbitals and a declared
    up/down ordering.  Complex spherical-harmonic projections require an
    upstream orbital time-reversal matrix instead.
    """

    size = int(dimension)
    if size < 2 or size % 2:
        raise ValueError("atomic spinor dimension must be positive and even")
    orbital_count = size // 2
    spin_part = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
    if SpinOrder(spin_order) is SpinOrder.INTERLEAVED:
        return np.kron(np.eye(orbital_count, dtype=np.complex128), spin_part)
    return np.kron(spin_part, np.eye(orbital_count, dtype=np.complex128))


def projection_anchored_sewing(
    wannier_atomic_overlap: object,
    minus_k_index: object,
    atomic_sewing: object,
) -> ProjectionAnchoredSewing:
    """Construct a sewing matrix through a polar atomic projection frame.

    ``C[k,n,alpha]=<w_nk|g_alpha,k>`` is polar-orthonormalized as
    ``C=Q P`` and the result is ``B(k)=Q(k) J Q(-k)^T``.  This is a
    PROJECT-EXTENSION that is physically anchored only when the declared
    atomic trial functions span the selected Wannier subspace.  Callers must
    retain singular-value and projected-subspace time-reversal diagnostics.
    """

    overlap = require_complex128("Wannier/atomic overlap", wannier_atomic_overlap)
    if overlap.ndim != 3 or overlap.shape[-1] != overlap.shape[-2]:
        raise ValueError("Wannier/atomic overlap must have shape (nk,nw,nw)")
    minus = np.asarray(minus_k_index, dtype=np.int64)
    if minus.shape != (overlap.shape[0],):
        raise ValueError("minus_k_index must have shape (nk,)")
    if np.any(minus < 0) or np.any(minus >= minus.size):
        raise ValueError("minus_k_index contains out-of-range entries")
    if not np.array_equal(minus[minus], np.arange(minus.size)):
        raise ValueError("minus_k_index must be an involution")
    atomic = validate_sewing_matrix(atomic_sewing)
    if atomic.shape != overlap.shape[-2:]:
        raise ValueError("atomic sewing matrix and projection dimension differ")
    fermion_residual = float(
        np.max(np.abs(atomic @ atomic.conj() + np.eye(atomic.shape[0])))
    )
    if fermion_residual > 1.0e-10:
        raise TimeReversalMetadataError(
            "atomic sewing matrix does not satisfy spin-1/2 Theta^2=-1: "
            f"residual={fermion_residual:.3e}"
        )

    left, singular_values, right_h = np.linalg.svd(overlap, full_matrices=False)
    frame = np.asarray(left @ right_h, dtype=np.complex128)
    sewing = np.asarray(
        frame @ atomic @ np.swapaxes(frame[minus], -1, -2),
        dtype=np.complex128,
    )
    return ProjectionAnchoredSewing(
        atomic_frame=frame,
        sewing=sewing,
        singular_values=np.asarray(singular_values, dtype=np.float64),
    )


def validate_sewing_matrix(
    value: object, *, tolerance: float = 1.0e-10
) -> NDArray[np.complex128]:
    sewing = require_complex128("B_theta", value)
    if sewing.ndim < 2 or sewing.shape[-1] != sewing.shape[-2]:
        raise TimeReversalMetadataError("B_theta must contain square matrices")
    identity = np.eye(sewing.shape[-1], dtype=np.complex128)
    residual = float(
        np.max(np.abs(sewing @ np.swapaxes(sewing.conj(), -1, -2) - identity))
    )
    if residual > tolerance:
        raise TimeReversalMetadataError(
            f"B_theta is not unitary: residual {residual:.3e}"
        )
    return sewing


def time_reversed(
    h_minus_k: object,
    sewing_k: object,
) -> NDArray[np.complex128]:
    """Return ``B_theta(k) H(-k)^* B_theta(k)^dagger`` (WT-E03)."""

    h_minus = require_complex128("H_minus_k", h_minus_k)
    sewing = validate_sewing_matrix(sewing_k)
    if h_minus.shape[-2:] != sewing.shape[-2:]:
        raise TimeReversalMetadataError(
            "H(-k) and B_theta have incompatible dimensions"
        )
    return sewing @ h_minus.conj() @ np.swapaxes(sewing.conj(), -1, -2)


__all__ = [
    "ProjectionAnchoredSewing",
    "atomic_spin_time_reversal",
    "projection_anchored_sewing",
    "time_reversed",
    "validate_sewing_matrix",
]
