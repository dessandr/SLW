"""Hermitian local exchange-field partition (WT-L01)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import require_complex128
from slw.wtorque.config import SiteProjection
from slw.wtorque.errors import LocalProjectionError


def localize_exchange(
    exchange: object,
    projectors: object,
    *,
    strategy: SiteProjection | str = SiteProjection.LOCAL_PARTITION,
    user_supplied: object | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return local fields and the unassigned support residual.

    For ``exchange[...,nw,nw]`` the local result has shape
    ``[nmag,...,nw,nw]``. The production default splits cross-boundary blocks
    equally; onsite-only is retained as a diagnostic.
    """

    x = require_complex128("H_XC", exchange)
    p = require_complex128("magnetic projectors", projectors)
    if p.ndim != 3 or p.shape[1:] != x.shape[-2:]:
        raise LocalProjectionError("projectors must have shape (nmag,nw,nw)")
    mode = SiteProjection(strategy)
    local: NDArray[np.complex128]
    if mode is SiteProjection.USER_SUPPLIED:
        if user_supplied is None:
            raise LocalProjectionError("user_supplied projection requires local matrices")
        local = require_complex128("user supplied local fields", user_supplied)
        if local.shape != (p.shape[0], *x.shape):
            raise LocalProjectionError("user supplied local-field shape mismatch")
    else:
        expand = (p.shape[0],) + (1,) * (x.ndim - 2) + p.shape[1:]
        pp = p.reshape(expand)
        xx = x[None, ...]
        if mode in {SiteProjection.LOCAL_PARTITION, SiteProjection.FULL_ATOM}:
            local = np.asarray(0.5 * (pp @ xx + xx @ pp), dtype=np.complex128)
        elif mode is SiteProjection.ONSITE_ONLY:
            local = np.asarray(pp @ xx @ pp, dtype=np.complex128)
        else:  # pragma: no cover - exhaustive enum
            raise LocalProjectionError(f"unsupported local projection {mode.value}")
    residual = x - np.sum(local, axis=0)
    return np.asarray(local, dtype=np.complex128), np.asarray(residual, dtype=np.complex128)


def support_residual_norm(exchange: object, projectors: object) -> float:
    values = require_complex128("H_XC", exchange)
    _, residual = localize_exchange(values, projectors)
    denominator = max(float(np.linalg.norm(values)), float(np.finfo(float).tiny))
    return float(np.linalg.norm(residual) / denominator)


__all__ = ["localize_exchange", "support_residual_norm"]
