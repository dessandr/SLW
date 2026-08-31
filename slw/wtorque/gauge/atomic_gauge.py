"""Atomic-position gauge wrapping matrices from WT convention 01."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def atomic_gauge_matrix(
    reciprocal_vector: object,
    orbital_centers: object,
) -> NDArray[np.complex128]:
    """Return ``D_G=diag(exp(i 2pi G.tau_n)) tensor I_spin``."""

    g = np.asarray(reciprocal_vector, dtype=np.float64)
    centers = np.asarray(orbital_centers, dtype=np.float64)
    if g.shape[-1:] != (3,) or centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("G and orbital_centers must end in 3 Cartesian/reduced components")
    scalar = g.ndim == 1
    if scalar:
        g = g[None, :]
    if g.ndim != 2:
        raise ValueError("reciprocal_vector must have shape (3,) or (n,3)")
    orbital_phase = np.exp(2j * np.pi * (g @ centers.T))
    spinor_phase = np.repeat(orbital_phase, 2, axis=1)
    result = np.zeros((g.shape[0], 2 * centers.shape[0], 2 * centers.shape[0]), dtype=np.complex128)
    diagonal = np.arange(result.shape[-1])
    result[:, diagonal, diagonal] = spinor_phase
    return result[0] if scalar else result


def wrap_hamiltonian(
    wrapped_matrix: object,
    reciprocal_vector: object,
    orbital_centers: object,
) -> NDArray[np.complex128]:
    """Convert ``H(kbar)`` to ``H(kbar+G)=D_G^dagger H D_G``."""

    matrix = np.asarray(wrapped_matrix, dtype=np.complex128)
    d_g = atomic_gauge_matrix(reciprocal_vector, orbital_centers)
    return np.swapaxes(d_g.conj(), -1, -2) @ matrix @ d_g


def unwrap_final_state_vertex(
    wrapped_vertex: object,
    reciprocal_vector: object,
    orbital_centers: object,
) -> NDArray[np.complex128]:
    """Convert the final index of ``g`` from wrapped to unwrapped atomic gauge."""

    vertex = np.asarray(wrapped_vertex)
    if vertex.dtype != np.complex128:
        raise TypeError(f"wrapped_vertex must use complex128; got {vertex.dtype}")
    d_g = atomic_gauge_matrix(reciprocal_vector, orbital_centers)
    if d_g.ndim == 2:
        if vertex.shape[-2:] != d_g.shape:
            raise ValueError("wrapped vertex and gauge matrix dimensions do not match")
    else:
        if vertex.shape[0] != d_g.shape[0] or vertex.shape[-2:] != d_g.shape[-2:]:
            raise ValueError("batched wrapped vertex and gauge matrices do not match")
        while d_g.ndim < vertex.ndim:
            d_g = np.expand_dims(d_g, axis=1)
    return np.swapaxes(d_g.conj(), -1, -2) @ vertex


__all__ = ["atomic_gauge_matrix", "unwrap_final_state_vertex", "wrap_hamiltonian"]
