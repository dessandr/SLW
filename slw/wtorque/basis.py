"""Canonical interleaved spinor-basis conversions (WT data contract 09)."""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray


class SpinOrder(str, Enum):
    INTERLEAVED = "interleaved"
    BLOCKED = "blocked"


def require_complex128(name: str, value: object) -> NDArray[np.complex128]:
    """Return an array only when it is already complex128.

    Physics-core inputs are rejected rather than silently promoted so an
    upstream precision loss cannot be disguised by a later cast.
    """

    array = np.asarray(value)
    if array.dtype != np.dtype(np.complex128):
        raise TypeError(f"{name} must use complex128; got {array.dtype}")
    return array


def _blocked_to_interleaved(norb: int) -> NDArray[np.int64]:
    if norb < 1:
        raise ValueError("norb must be positive")
    return np.column_stack(
        (np.arange(norb, dtype=np.int64), np.arange(norb, 2 * norb, dtype=np.int64))
    ).reshape(-1)


def _permute_matrix(value: object, permutation: NDArray[np.int64]) -> np.ndarray:
    array = np.asarray(value)
    expected = permutation.size
    if array.ndim < 2 or array.shape[-2:] != (expected, expected):
        raise ValueError(
            f"spinor matrix must end in ({expected}, {expected}); got {array.shape}"
        )
    return np.take(np.take(array, permutation, axis=-2), permutation, axis=-1)


def canonicalize_spin_order(
    value: object,
    norb: int,
    source: SpinOrder | str,
) -> np.ndarray:
    """Convert matrix trailing axes to orbital-major interleaved spin order."""

    order = SpinOrder(source)
    if order is SpinOrder.INTERLEAVED:
        return np.array(value, copy=True)
    return _permute_matrix(value, _blocked_to_interleaved(norb))


def restore_spin_order(
    value: object,
    norb: int,
    target: SpinOrder | str,
) -> np.ndarray:
    """Convert an interleaved matrix to the requested output spin order."""

    order = SpinOrder(target)
    if order is SpinOrder.INTERLEAVED:
        return np.array(value, copy=True)
    permutation = _blocked_to_interleaved(norb)
    inverse = np.argsort(permutation)
    return _permute_matrix(value, inverse)


__all__ = [
    "SpinOrder",
    "canonicalize_spin_order",
    "require_complex128",
    "restore_spin_order",
]

