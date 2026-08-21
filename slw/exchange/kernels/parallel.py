"""Small collective helpers for native exchange kernels.

Every numerical mode uses the same contract: expensive setup is rank-local,
the integration mesh is distributed deterministically, rank-local failures are
agreed before the gather, and only rank zero receives the reduced payload.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def rank_size(comm: Any | None) -> tuple[int, int]:
    """Return ``(rank, size)`` for an mpi4py-compatible communicator."""

    if comm is None:
        return 0, 1
    return int(comm.Get_rank()), int(comm.Get_size())


def partition_sequence(values: Sequence[T], comm: Any | None) -> list[T]:
    """Return a deterministic strided rank partition without material copies."""

    rank, size = rank_size(comm)
    return list(values[rank::size])


def _sum_payloads(values: Sequence[Any]) -> Any:
    first = values[0]
    if first is None:
        if any(value is not None for value in values):
            raise TypeError("inconsistent optional payload across MPI ranks")
        return None
    if isinstance(first, np.ndarray):
        result = np.zeros_like(first)
        for value in values:
            array = np.asarray(value)
            if array.shape != first.shape:
                raise ValueError(
                    f"MPI payload shape mismatch: {array.shape} != {first.shape}"
                )
            result += array
        return result
    if isinstance(first, Mapping):
        keys = tuple(first)
        if any(tuple(value) != keys for value in values):
            raise ValueError("MPI mapping payload keys differ across ranks")
        return {key: _sum_payloads([value[key] for value in values]) for key in keys}
    if isinstance(first, tuple):
        if any(len(value) != len(first) for value in values):
            raise ValueError("MPI tuple payload lengths differ across ranks")
        return tuple(
            _sum_payloads([value[index] for value in values])
            for index in range(len(first))
        )
    if isinstance(first, (float, complex, np.number)):
        return sum(values)
    raise TypeError(f"unsupported MPI reduction payload {type(first).__name__}")


def collective_sum(comm: Any | None, work: Callable[[], T]) -> T | None:
    """Run local work, agree failures, and sum its numerical payload on root."""

    if comm is None:
        return work()
    rank, _size = rank_size(comm)
    error: tuple[int, str, str] | None
    try:
        local = work()
    except Exception as exc:  # noqa: BLE001 - agree before another collective
        local = None
        error = (rank, type(exc).__name__, str(exc))
    else:
        error = None
    errors = list(comm.allgather(error))
    failures = [item for item in errors if item is not None]
    if failures:
        rendered = "; ".join(
            f"rank {item[0]} {item[1]}: {item[2]}" for item in failures
        )
        raise RuntimeError(f"native exchange MPI work failed: {rendered}")
    gathered = comm.gather(local, root=0)
    if rank != 0:
        return None
    if gathered is None or not gathered:
        raise RuntimeError("native exchange MPI gather returned no root payload")
    return _sum_payloads(gathered)


def collective_call(
    comm: Any | None,
    work: Callable[[], T],
    *,
    phase: str,
) -> T:
    """Run rank-local work and agree failures before later collectives."""

    if comm is None:
        return work()
    rank, _size = rank_size(comm)
    error: tuple[int, str, str] | None
    try:
        result = work()
    except Exception as exc:  # noqa: BLE001 - synchronize rank-local setup
        result = None
        error = (rank, type(exc).__name__, str(exc))
    else:
        error = None
    errors = list(comm.allgather(error))
    failures = [item for item in errors if item is not None]
    if failures:
        rendered = "; ".join(
            f"rank {item[0]} {item[1]}: {item[2]}" for item in failures
        )
        raise RuntimeError(f"native exchange MPI {phase} failed: {rendered}")
    return result  # type: ignore[return-value]


def collective_root_call(
    comm: Any | None,
    work: Callable[[], T],
    *,
    phase: str,
) -> T | None:
    """Run root-only finalization and broadcast its success or failure."""

    if comm is None:
        return work()
    rank, _size = rank_size(comm)
    result: T | None = None
    error: tuple[int, str, str] | None = None
    if rank == 0:
        try:
            result = work()
        except Exception as exc:  # noqa: BLE001 - peers await root status
            error = (rank, type(exc).__name__, str(exc))
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(
            f"native exchange MPI {phase} failed: "
            f"rank {error[0]} {error[1]}: {error[2]}"
        )
    return result


__all__ = [
    "collective_call",
    "collective_root_call",
    "collective_sum",
    "partition_sequence",
    "rank_size",
]
