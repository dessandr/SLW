"""Runtime bridge from SLW atomic-SOC inputs to canonical wtorque matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slw.soc import ResolvedAtomicSOCManifold, SpinorGroupBy, build_atomic_soc_matrix

from ..basis import require_complex128
from ..config import OnsiteSOCConfig


def resolve_onsite_soc(
    config: OnsiteSOCConfig,
    *,
    norb: int,
) -> tuple[NDArray[np.complex128], tuple[ResolvedAtomicSOCManifold, ...]]:
    """Resolve an input SOC card into canonical orbital-major spin order."""

    matrix, resolved = build_atomic_soc_matrix(
        config.spec,
        config.win_file,
        nwan=norb,
        groupby=SpinorGroupBy.ORBITAL,
        scale=config.scale,
        p_order=config.p_order,
        d_order=config.d_order,
    )
    return require_complex128("onsite SOC matrix", matrix), resolved


def add_onsite_to_realspace(
    matrices_r: object,
    r_vectors: object,
    onsite_matrix: object | None,
) -> NDArray[np.complex128]:
    """Add one onsite matrix to the unique ``R=(0,0,0)`` block."""

    matrices = require_complex128("real-space matrices", matrices_r)
    if onsite_matrix is None:
        return np.array(matrices, copy=True)
    onsite = require_complex128("onsite SOC matrix", onsite_matrix)
    vectors = np.asarray(r_vectors)
    if vectors.shape != (matrices.shape[0], 3) or vectors.dtype.kind not in "iu":
        raise ValueError("R_vectors must be integer with shape (nR,3)")
    zero = np.flatnonzero(np.all(vectors == 0, axis=1))
    if zero.size != 1:
        raise ValueError(
            f"onsite SOC requires exactly one R=(0,0,0) block; found {zero.size}"
        )
    if onsite.shape != matrices.shape[-2:]:
        raise ValueError(
            f"onsite SOC shape {onsite.shape} does not match {matrices.shape[-2:]}"
        )
    output = np.array(matrices, copy=True)
    output[int(zero[0])] += onsite
    return np.asarray(output, dtype=np.complex128)


__all__ = ["add_onsite_to_realspace", "resolve_onsite_soc"]
