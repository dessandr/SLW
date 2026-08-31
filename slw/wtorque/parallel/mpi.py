"""Optional MPI facade used to distribute independent q pairs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class MPIUnavailableError(RuntimeError):
    """Raised when a multi-rank launcher is detected without mpi4py."""


_SIZE_ENV = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
)


def _launched_size() -> int:
    sizes: list[int] = []
    for name in _SIZE_ENV:
        try:
            sizes.append(int(os.environ[name]))
        except (KeyError, ValueError):
            continue
    return max(sizes, default=1)


@dataclass(frozen=True)
class MPIContext:
    comm: Any = None
    rank: int = 0
    size: int = 1

    @classmethod
    def discover(cls) -> MPIContext:
        try:
            from mpi4py import MPI
        except ImportError as exc:
            launched = _launched_size()
            if launched > 1:
                raise MPIUnavailableError(
                    f"detected a {launched}-rank launch but mpi4py is unavailable; install SLW with .[mpi]"
                ) from exc
            return cls()
        comm = MPI.COMM_WORLD
        return cls(comm=comm, rank=int(comm.Get_rank()), size=int(comm.Get_size()))

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def bcast(self, value: Any, root: int = 0) -> Any:
        return value if self.comm is None else self.comm.bcast(value, root=root)

    def gather(self, value: Any, root: int = 0) -> list[Any] | None:
        if self.comm is None:
            return [value] if self.rank == root else None
        result = self.comm.gather(value, root=root)
        return None if result is None else list(result)

    def allgather(self, value: Any) -> list[Any]:
        return [value] if self.comm is None else list(self.comm.allgather(value))

    def scatter(self, values: Any, root: int = 0) -> Any:
        if self.comm is None:
            if not isinstance(values, (list, tuple)) or len(values) != 1:
                raise ValueError("serial MPI scatter requires one payload")
            return values[0]
        return self.comm.scatter(values, root=root)

    def barrier(self) -> None:
        if self.comm is not None:
            self.comm.Barrier()


__all__ = ["MPIContext", "MPIUnavailableError"]
