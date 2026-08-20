"""Vectorized conversions between the two TB2J spinor basis layouts."""

from __future__ import annotations

from typing import Any

import numpy as np

from .model import SpinorGroupBy


def normalize_groupby(value: str | SpinorGroupBy) -> SpinorGroupBy:
    if isinstance(value, SpinorGroupBy):
        return value
    try:
        return SpinorGroupBy(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(f"groupby must be 'spin' or 'orbital', got {value!r}") from exc


def spin_major_to_groupby_indices(
    nwan: int, groupby: str | SpinorGroupBy
) -> np.ndarray:
    """Return file-order indices into a spin-major vector.

    ``spin`` is ``[all up | all down]``. ``orbital`` is
    ``[orb1 up, orb1 down, orb2 up, orb2 down, ...]``.
    """

    count = int(nwan)
    if count <= 0:
        raise ValueError(f"nwan must be positive, got {nwan!r}")
    mode = normalize_groupby(groupby)
    if mode is SpinorGroupBy.SPIN:
        return np.arange(2 * count, dtype=np.int64)
    indices = np.empty(2 * count, dtype=np.int64)
    indices[0::2] = np.arange(count, dtype=np.int64)
    indices[1::2] = np.arange(count, 2 * count, dtype=np.int64)
    return indices


def groupby_to_spin_major_indices(
    nwan: int, groupby: str | SpinorGroupBy
) -> np.ndarray:
    """Return spin-major-order indices into a file-order vector."""

    forward = spin_major_to_groupby_indices(nwan, groupby)
    inverse = np.empty_like(forward)
    inverse[forward] = np.arange(forward.size, dtype=np.int64)
    return inverse


def reorder_spinor_matrix(
    matrix: Any,
    *,
    source: str | SpinorGroupBy,
    target: str | SpinorGroupBy,
) -> np.ndarray:
    """Reorder the final two matrix axes without Python element loops."""

    values = np.asarray(matrix)
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise ValueError(f"spinor matrix must be square, got shape={values.shape}")
    if values.shape[-1] % 2:
        raise ValueError(
            f"spinor matrix dimension must be even, got {values.shape[-1]}"
        )
    source_mode = normalize_groupby(source)
    target_mode = normalize_groupby(target)
    if source_mode is target_mode:
        return np.array(values, copy=True)
    nwan = values.shape[-1] // 2
    if source_mode is SpinorGroupBy.SPIN:
        indices = spin_major_to_groupby_indices(nwan, target_mode)
    else:
        indices = groupby_to_spin_major_indices(nwan, source_mode)
    return values[..., indices[:, None], indices]


def reorder_spinor_rows(
    rows: list[str] | tuple[str, ...],
    *,
    source: str | SpinorGroupBy,
    target: str | SpinorGroupBy,
) -> list[str]:
    """Apply the identical basis permutation to Wannier-centre rows."""

    if len(rows) % 2:
        raise ValueError(f"spinor row count must be even, got {len(rows)}")
    source_mode = normalize_groupby(source)
    target_mode = normalize_groupby(target)
    if source_mode is target_mode:
        return list(rows)
    indices = (
        spin_major_to_groupby_indices(len(rows) // 2, target_mode)
        if source_mode is SpinorGroupBy.SPIN
        else groupby_to_spin_major_indices(len(rows) // 2, source_mode)
    )
    return [rows[int(index)] for index in indices]


__all__ = [
    "groupby_to_spin_major_indices",
    "normalize_groupby",
    "reorder_spinor_matrix",
    "reorder_spinor_rows",
    "spin_major_to_groupby_indices",
]
