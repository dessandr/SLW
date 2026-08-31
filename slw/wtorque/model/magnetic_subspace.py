"""Explicit, auditable projectors onto localized magnetic orbitals.

The legacy :class:`MagneticSubspace` represents a fixed orbital mask in an
already atomic-like Wannier gauge.  Native spinor Wannier90 calculations need
the more general projection-anchored construction below: the electronic
Hamiltonian remains in the full Wannier space, while only a declared atomic
projection subspace is used to identify and rotate the collinear exchange
field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from slw.wtorque.basis import (
    SpinOrder,
    canonicalize_spin_order,
    require_complex128,
    restore_spin_order,
)
from slw.wtorque.errors import MagneticSubspaceError

from .time_reversal import atomic_spin_time_reversal

_PAULI = np.asarray(
    (
        ((0.0, 1.0), (1.0, 0.0)),
        ((0.0, -1.0j), (1.0j, 0.0)),
        ((1.0, 0.0), (0.0, -1.0)),
    ),
    dtype=np.complex128,
)


@dataclass(frozen=True)
class MagneticSubspace:
    orbital_masks: NDArray[np.bool_]
    projectors: NDArray[np.complex128]
    labels: tuple[tuple[str, ...], ...]

    @classmethod
    def from_masks(
        cls,
        masks: object,
        *,
        orbital_labels: tuple[str, ...] | None = None,
    ) -> MagneticSubspace:
        array = np.asarray(masks, dtype=bool)
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
            raise MagneticSubspaceError("magnetic masks must have shape (nmag,norb)")
        if np.any(array.sum(axis=1) == 0):
            raise MagneticSubspaceError(
                "each magnetic site must select at least one orbital"
            )
        if np.any(array.sum(axis=0) > 1):
            raise ValueError("magnetic orbital masks must be disjoint")
        norb = array.shape[1]
        if orbital_labels is None:
            orbital_labels = tuple(str(index) for index in range(norb))
        if len(orbital_labels) != norb:
            raise MagneticSubspaceError("orbital label count does not match mask width")
        spin_masks = np.repeat(array, 2, axis=1)
        projectors = np.zeros((array.shape[0], 2 * norb, 2 * norb), dtype=np.complex128)
        diagonal = np.arange(2 * norb)
        projectors[:, diagonal, diagonal] = spin_masks
        labels = tuple(
            tuple(orbital_labels[index] for index in np.flatnonzero(mask))
            for mask in array
        )
        return cls(array.copy(), projectors, labels)

    @property
    def magnetic_projector(self) -> NDArray[np.complex128]:
        return self.projectors.sum(axis=0)


@dataclass(frozen=True)
class ProjectedCollinearExchange:
    """A collinear magnetic exchange field embedded in the full W gauge.

    ``atomic_frame`` is the rectangular polar isometry ``Q(k)`` obtained from
    the explicitly selected AMN columns.  The full Hamiltonian is never
    truncated: only ``Q^dagger H Q`` is used to identify the time-reversal-odd
    magnetic field, and the resulting field/vertices are embedded back with
    ``Q (...) Q^dagger``.

    This is a labelled d-only (or otherwise input-selected) collinear exchange
    approximation.  ``exchange_subspace_raw`` retains the unprojected TR-odd
    field so the discarded non-collinear component remains auditable.
    """

    atomic_frame: NDArray[np.complex128]
    singular_values: NDArray[np.float64]
    exchange_subspace_raw: NDArray[np.complex128]
    exchange_subspace_collinear: NDArray[np.complex128]
    exchange_full: NDArray[np.complex128]
    transverse_vertices: NDArray[np.complex128]
    longitudinal_vertex: NDArray[np.complex128]
    local_frame: NDArray[np.float64]
    magnetization_direction: NDArray[np.float64]
    spin_order: SpinOrder

    @property
    def noncollinear_fraction_per_k(self) -> NDArray[np.float64]:
        difference = self.exchange_subspace_raw - self.exchange_subspace_collinear
        numerator = np.linalg.norm(difference.reshape(difference.shape[0], -1), axis=1)
        denominator = np.linalg.norm(
            self.exchange_subspace_raw.reshape(self.exchange_subspace_raw.shape[0], -1),
            axis=1,
        )
        return np.asarray(
            numerator / np.maximum(denominator, np.finfo(np.float64).tiny),
            dtype=np.float64,
        )

    @property
    def raw_rotation_vertices(self) -> NDArray[np.complex128]:
        """Return angle derivatives of the uncollinearized exchange field.

        The axes follow ``local_frame`` and the matrices are embedded back in
        the full Wannier space.  These vertices are diagnostic only: the
        production transverse vertices deliberately rotate the declared
        collinear exchange component.
        """

        dimension = self.exchange_subspace_raw.shape[-1]
        vertices = []
        for axis in self.local_frame:
            sigma = atomic_spin_matrix(axis, dimension, self.spin_order)
            gamma_subspace = 0.5j * (
                self.exchange_subspace_raw @ sigma
                - sigma @ self.exchange_subspace_raw
            )
            vertices.append(_embed(self.atomic_frame, gamma_subspace))
        return np.asarray(np.stack(vertices, axis=1), dtype=np.complex128)


def _unit_direction(value: object, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise MagneticSubspaceError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise MagneticSubspaceError(f"{name} must be nonzero")
    return np.asarray(vector / norm, dtype=np.float64)


def transverse_frame(direction: object) -> NDArray[np.float64]:
    """Return deterministic rows ``[t1,t2,n]`` for a magnetic direction."""

    normal = _unit_direction(direction, name="magnetization direction")
    seed = np.zeros(3, dtype=np.float64)
    seed[int(np.argmin(np.abs(normal)))] = 1.0
    first = np.cross(seed, normal)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return np.asarray(np.stack((first, second, normal)), dtype=np.float64)


def atomic_spin_matrix(
    axis: object,
    dimension: int,
    spin_order: SpinOrder | str,
) -> NDArray[np.complex128]:
    """Return ``I_orb tensor axis.sigma`` in the declared atomic ordering."""

    size = int(dimension)
    if size < 2 or size % 2:
        raise MagneticSubspaceError(
            "magnetic projection dimension must be positive and even"
        )
    direction = _unit_direction(axis, name="spin axis")
    sigma = np.einsum("a,aij->ij", direction, _PAULI, optimize=True)
    interleaved = np.kron(
        np.eye(size // 2, dtype=np.complex128),
        sigma,
    )
    return np.asarray(
        restore_spin_order(interleaved, size // 2, SpinOrder(spin_order)),
        dtype=np.complex128,
    )


def _collinear_component(
    exchange: NDArray[np.complex128],
    direction: NDArray[np.float64],
    spin_order: SpinOrder,
) -> NDArray[np.complex128]:
    dimension = exchange.shape[-1]
    norb = dimension // 2
    interleaved = np.asarray(
        canonicalize_spin_order(exchange, norb, spin_order),
        dtype=np.complex128,
    )
    blocks = interleaved.reshape(exchange.shape[0], norb, 2, norb, 2)
    sigma = np.einsum("a,aij->ij", direction, _PAULI, optimize=True)
    orbital_field = 0.5 * np.einsum(
        "ts,kisjt->kij",
        sigma,
        blocks,
        optimize=True,
    )
    orbital_field = 0.5 * (orbital_field + np.swapaxes(orbital_field.conj(), 1, 2))
    collinear = np.einsum(
        "kij,st->kisjt",
        orbital_field,
        sigma,
        optimize=True,
    ).reshape(exchange.shape)
    return np.asarray(
        restore_spin_order(collinear, norb, spin_order),
        dtype=np.complex128,
    )


def _embed(
    atomic_frame: NDArray[np.complex128],
    subspace_matrix: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    return np.asarray(
        atomic_frame @ subspace_matrix @ np.swapaxes(atomic_frame.conj(), 1, 2),
        dtype=np.complex128,
    )


def build_projected_collinear_exchange(
    hamiltonian_k: object,
    wannier_atomic_overlap: object,
    minus_k_index: object,
    *,
    magnetization_direction: object,
    atomic_spin_order: SpinOrder | str,
    rank_tolerance: float = 1.0e-8,
) -> ProjectedCollinearExchange:
    """Extract and embed a collinear exchange field from a magnetic subspace.

    Gauge contract: under an arbitrary Wannier rotation ``R(k)``, callers pass
    ``H' = R^dagger H R`` and ``C' = R^dagger C``.  The returned full-space
    field and vertices then transform covariantly with the same ``R(k)``.
    """

    hamiltonian = require_complex128("full Hamiltonian", hamiltonian_k)
    overlap = require_complex128(
        "selected Wannier/atomic overlap", wannier_atomic_overlap
    )
    if hamiltonian.ndim != 3 or hamiltonian.shape[-1] != hamiltonian.shape[-2]:
        raise MagneticSubspaceError("full Hamiltonian must have shape (nk,nw,nw)")
    if overlap.ndim != 3 or overlap.shape[:2] != hamiltonian.shape[:2]:
        raise MagneticSubspaceError(
            "selected overlap must have shape (nk,nw,nmagnetic_projection)"
        )
    nk, nw, magnetic_dimension = overlap.shape
    if magnetic_dimension < 2 or magnetic_dimension % 2 or magnetic_dimension > nw:
        raise MagneticSubspaceError(
            "selected magnetic projection dimension must be even and no larger than nw"
        )
    minus = np.asarray(minus_k_index, dtype=np.int64)
    if minus.shape != (nk,) or np.any(minus < 0) or np.any(minus >= nk):
        raise MagneticSubspaceError("minus_k_index must contain one valid index per k")
    if not np.array_equal(minus[minus], np.arange(nk, dtype=np.int64)):
        raise MagneticSubspaceError("minus_k_index must be an involution")
    threshold = float(rank_tolerance)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise MagneticSubspaceError("rank_tolerance must be finite and positive")
    order = SpinOrder(atomic_spin_order)
    direction = _unit_direction(
        magnetization_direction,
        name="magnetization direction",
    )

    left, singular_values, right_h = np.linalg.svd(overlap, full_matrices=False)
    minimum_singular = float(np.min(singular_values, initial=np.inf))
    if minimum_singular < threshold:
        raise MagneticSubspaceError(
            "selected magnetic projection is rank deficient: "
            f"minimum singular value {minimum_singular:.6e} < {threshold:.6e}"
        )
    atomic_frame = np.asarray(left @ right_h, dtype=np.complex128)
    frame_dagger = np.swapaxes(atomic_frame.conj(), 1, 2)
    h_subspace = np.asarray(
        frame_dagger @ hamiltonian @ atomic_frame,
        dtype=np.complex128,
    )
    atomic_sewing = atomic_spin_time_reversal(magnetic_dimension, order)
    theta_h = np.asarray(
        atomic_sewing @ h_subspace[minus].conj() @ atomic_sewing.conj().T,
        dtype=np.complex128,
    )
    exchange_raw = np.asarray(0.5 * (h_subspace - theta_h), dtype=np.complex128)
    exchange_raw = np.asarray(
        0.5 * (exchange_raw + np.swapaxes(exchange_raw.conj(), 1, 2)),
        dtype=np.complex128,
    )
    exchange_collinear = _collinear_component(exchange_raw, direction, order)
    exchange_full = _embed(atomic_frame, exchange_collinear)

    local_frame = transverse_frame(direction)
    rotation_vertices = []
    for axis in local_frame:
        sigma = atomic_spin_matrix(axis, magnetic_dimension, order)
        gamma_subspace = 0.5j * (
            exchange_collinear @ sigma - sigma @ exchange_collinear
        )
        rotation_vertices.append(_embed(atomic_frame, gamma_subspace))
    gamma_t1, gamma_t2, gamma_longitudinal = rotation_vertices
    transverse = np.asarray(
        np.stack((gamma_t2, -gamma_t1), axis=1),
        dtype=np.complex128,
    )

    return ProjectedCollinearExchange(
        atomic_frame=atomic_frame,
        singular_values=np.asarray(singular_values, dtype=np.float64),
        exchange_subspace_raw=exchange_raw,
        exchange_subspace_collinear=exchange_collinear,
        exchange_full=exchange_full,
        transverse_vertices=transverse,
        longitudinal_vertex=np.asarray(gamma_longitudinal, dtype=np.complex128),
        local_frame=local_frame,
        magnetization_direction=direction,
        spin_order=order,
    )


def rotate_projected_exchange(
    exchange: ProjectedCollinearExchange,
    axis: object,
    angle: float,
) -> NDArray[np.complex128]:
    """Rigidly rotate only the selected collinear exchange subspace."""

    theta = float(angle)
    if not np.isfinite(theta):
        raise MagneticSubspaceError("rotation angle must be finite")
    dimension = exchange.exchange_subspace_collinear.shape[-1]
    sigma = atomic_spin_matrix(axis, dimension, exchange.spin_order)
    unitary = expm(-0.5j * theta * sigma)
    rotated = np.asarray(
        unitary @ exchange.exchange_subspace_collinear @ unitary.conj().T,
        dtype=np.complex128,
    )
    return _embed(exchange.atomic_frame, rotated)


__all__ = [
    "MagneticSubspace",
    "ProjectedCollinearExchange",
    "atomic_spin_matrix",
    "build_projected_collinear_exchange",
    "rotate_projected_exchange",
    "transverse_frame",
]
