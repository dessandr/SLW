"""Exact commensurate k+q index and reciprocal-wrap construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.errors import MeshCommensurabilityError


@dataclass(frozen=True)
class KQMap:
    indices: NDArray[np.int64]
    G_wrap: NDArray[np.int64]
    residual: NDArray[np.float64]

    @property
    def max_residual(self) -> float:
        return float(np.max(np.abs(self.residual), initial=0.0))


def build_kq_map(
    kpoints: object,
    qpoint: object,
    *,
    tolerance: float = 1.0e-9,
) -> KQMap:
    """Vectorized exact mapping on a reduced-coordinate commensurate mesh."""

    mesh = np.asarray(kpoints, dtype=np.float64)
    q = np.asarray(qpoint, dtype=np.float64)
    if mesh.ndim != 2 or mesh.shape[1] != 3 or q.shape != (3,):
        raise ValueError("kpoints and qpoint must have shapes (nk,3) and (3,)")
    target = mesh + q
    delta = target[:, None, :] - mesh[None, :, :]
    periodic_delta = delta - np.rint(delta)
    distances = np.max(np.abs(periodic_delta), axis=-1)
    indices = np.argmin(distances, axis=1).astype(np.int64)
    best = distances[np.arange(mesh.shape[0]), indices]
    if np.any(best > tolerance):
        offender = int(np.argmax(best))
        raise MeshCommensurabilityError(
            f"q is not commensurate with the k mesh: ik={offender}, residual={best[offender]:.3e}"
        )
    g_wrap_float = target - mesh[indices]
    g_wrap = np.rint(g_wrap_float).astype(np.int64)
    residual = np.asarray(g_wrap_float - g_wrap, dtype=np.float64)
    if np.max(np.abs(residual), initial=0.0) > tolerance:
        raise MeshCommensurabilityError("k+q reciprocal wrapping has a noninteger residual")
    return KQMap(indices=indices, G_wrap=g_wrap, residual=residual)


__all__ = ["KQMap", "build_kq_map"]

