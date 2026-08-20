"""Collective, deterministic MPI primitives for native magph calculations.

External k points are the first distribution axis because they are independent
and require no q-point reduction.  A rank-local worker receives one contiguous
index block and is expected to vectorize internally over that block and over
its q/mode axes.  Only rank zero materializes the assembled array by default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from slw.cli.mpi import MPIContext


@dataclass(frozen=True)
class IndexPartition:
    """One balanced contiguous slice of a global task axis."""

    total_count: int
    rank: int
    size: int
    start: int
    stop: int

    @property
    def count(self) -> int:
        return self.stop - self.start

    @property
    def indices(self) -> NDArray[np.int64]:
        values = np.arange(self.start, self.stop, dtype=np.int64)
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class RankFailure:
    rank: int
    stage: str
    exception_type: str
    message: str


class CollectiveExecutionError(RuntimeError):
    """An identical, pickle-safe summary raised on every participating rank."""

    def __init__(self, failures: tuple[RankFailure, ...]):
        self.failures = failures
        details = "; ".join(
            f"rank {item.rank} {item.exception_type}: {item.message}"
            for item in failures
        )
        stage = failures[0].stage if failures else "unknown"
        super().__init__(f"collective stage {stage!r} failed: {details}")


@dataclass(frozen=True)
class DistributedArrayResult:
    """Local ownership plus a root-only or broadcast global result."""

    local_indices: NDArray[np.int64]
    global_values: NDArray[Any] | None
    total_count: int
    item_shape: tuple[int, ...]
    mpi_size: int
    root_rank: int


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative; got {value!r}")
    return result


def balanced_partition(total_count: int, rank: int, size: int) -> IndexPartition:
    """Return a deterministic contiguous partition with imbalance at most one."""

    total = _nonnegative_integer("total_count", total_count)
    rank_value = _nonnegative_integer("rank", rank)
    size_value = _nonnegative_integer("size", size)
    if size_value < 1:
        raise ValueError("size must be positive")
    if rank_value >= size_value:
        raise ValueError(f"rank {rank_value} is outside communicator size {size_value}")
    quotient, remainder = divmod(total, size_value)
    start = rank_value * quotient + min(rank_value, remainder)
    stop = start + quotient + (1 if rank_value < remainder else 0)
    return IndexPartition(total, rank_value, size_value, start, stop)


def _failure(rank: int, stage: str, exc: BaseException) -> RankFailure:
    message = str(exc).strip() or repr(exc)
    return RankFailure(
        rank=int(rank),
        stage=str(stage),
        exception_type=type(exc).__name__,
        message=message,
    )


def _collective_failures(
    context: MPIContext,
    failure: RankFailure | None,
) -> tuple[RankFailure, ...]:
    gathered = context.allgather(failure)
    return tuple(item for item in gathered if item is not None)


def _readonly_array(value: object, *, dtype: np.dtype[Any]) -> NDArray[Any]:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _assemble_indexed_arrays(
    gathered: list[tuple[NDArray[np.int64], NDArray[Any]]],
    *,
    total_count: int,
    item_shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    output = np.empty((total_count, *item_shape), dtype=dtype)
    seen = np.zeros(total_count, dtype=bool)
    for indices_raw, values_raw in gathered:
        indices = np.asarray(indices_raw, dtype=np.int64)
        values = np.asarray(values_raw, dtype=dtype)
        if indices.ndim != 1 or values.shape != (indices.size, *item_shape):
            raise ValueError(
                "gathered index/value shapes are inconsistent: "
                f"indices={indices.shape}, values={values.shape}, item={item_shape}"
            )
        if np.any(indices < 0) or np.any(indices >= total_count):
            raise ValueError("gathered result contains an out-of-range global index")
        if np.unique(indices).size != indices.size or np.any(seen[indices]):
            raise ValueError("gathered result contains a duplicate global index")
        output[indices] = values
        seen[indices] = True
    if not np.all(seen):
        missing = np.flatnonzero(~seen)
        preview = ", ".join(str(int(index)) for index in missing[:8])
        suffix = "" if missing.size <= 8 else ", ..."
        raise ValueError(f"gathered result is missing global indices {preview}{suffix}")
    output.setflags(write=False)
    return output


def distributed_array_map(
    total_count: int,
    worker: Callable[[NDArray[np.int64]], object],
    *,
    item_shape: tuple[int, ...] = (),
    dtype: DTypeLike = np.float64,
    context: MPIContext | None = None,
    root: int = 0,
    broadcast_result: bool = False,
    stage: str = "external_k",
) -> DistributedArrayResult:
    """Run one vectorized local worker and deterministically gather its output.

    This function is collective whenever the discovered communicator has more
    than one rank.  Every local exception is exchanged before the gather, and
    every root assembly exception is broadcast before returning.  Therefore a
    normal Python failure cannot leave peer ranks waiting in the next phase.
    Process termination inside MPI itself remains the launcher's responsibility.
    """

    total = _nonnegative_integer("total_count", total_count)
    shape = tuple(
        _nonnegative_integer("item_shape entry", value) for value in item_shape
    )
    if any(value == 0 for value in shape):
        raise ValueError(f"item_shape entries must be positive; got {shape}")
    mpi = MPIContext.discover() if context is None else context
    root_rank = _nonnegative_integer("root", root)
    if root_rank >= mpi.size:
        raise ValueError(f"root {root_rank} is outside communicator size {mpi.size}")
    partition = balanced_partition(total, mpi.rank, mpi.size)
    indices = partition.indices
    resolved_dtype = np.dtype(dtype)
    local_values: NDArray[Any] | None = None
    local_failure: RankFailure | None = None
    try:
        local_values = np.asarray(worker(indices), dtype=resolved_dtype)
        expected = (indices.size, *shape)
        if local_values.shape != expected:
            raise ValueError(
                f"rank-local worker returned shape {local_values.shape}; expected {expected}"
            )
        if local_values.dtype.kind in {"f", "c"} and not np.all(
            np.isfinite(local_values)
        ):
            raise ValueError("rank-local worker returned non-finite values")
        local_values = _readonly_array(local_values, dtype=resolved_dtype)
    except Exception as exc:  # noqa: BLE001 - serialize before the next collective
        local_failure = _failure(mpi.rank, stage, exc)

    failures = _collective_failures(mpi, local_failure)
    if failures:
        raise CollectiveExecutionError(failures)
    if local_values is None:  # pragma: no cover - guarded by collective failure
        raise RuntimeError("rank-local worker produced neither a result nor a failure")

    gathered = mpi.gather((indices, local_values), root=root_rank)
    global_values: NDArray[Any] | None = None
    assembly_failure: RankFailure | None = None
    if mpi.rank == root_rank:
        try:
            if gathered is None:  # pragma: no cover - communicator contract guard
                raise RuntimeError("root rank did not receive gathered values")
            global_values = _assemble_indexed_arrays(
                gathered,
                total_count=total,
                item_shape=shape,
                dtype=resolved_dtype,
            )
        except Exception as exc:  # noqa: BLE001 - release non-root ranks
            assembly_failure = _failure(mpi.rank, f"{stage}_assembly", exc)
    assembly_failure = mpi.bcast(assembly_failure, root=root_rank)
    if assembly_failure is not None:
        raise CollectiveExecutionError((assembly_failure,))
    if broadcast_result:
        global_values = mpi.bcast(global_values, root=root_rank)
        if global_values is None:  # pragma: no cover - communicator contract guard
            raise RuntimeError("root rank did not broadcast the assembled result")
        global_values = _readonly_array(global_values, dtype=resolved_dtype)

    return DistributedArrayResult(
        local_indices=indices,
        global_values=global_values,
        total_count=total,
        item_shape=shape,
        mpi_size=mpi.size,
        root_rank=root_rank,
    )


__all__ = [
    "CollectiveExecutionError",
    "DistributedArrayResult",
    "IndexPartition",
    "RankFailure",
    "balanced_partition",
    "distributed_array_map",
]
