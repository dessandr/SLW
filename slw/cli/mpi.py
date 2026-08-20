"""Small optional-MPI facade for rank-safe input and dispatch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class MPIUnavailableError(RuntimeError):
    """Raised when a multi-rank launch is detected without mpi4py."""


_SIZE_ENV = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
)


def _launched_size() -> int:
    sizes = []
    for name in _SIZE_ENV:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            sizes.append(int(raw))
        except ValueError:
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
        except (ImportError, ModuleNotFoundError) as exc:
            size = _launched_size()
            if size > 1:
                raise MPIUnavailableError(
                    f"detected a {size}-rank launch but mpi4py is unavailable; "
                    "install SLW with `python -m pip install -e '.[mpi]'` "
                    "using the cluster MPI environment"
                ) from exc
            return cls()

        comm = MPI.COMM_WORLD
        return cls(comm=comm, rank=int(comm.Get_rank()), size=int(comm.Get_size()))

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def bcast(self, value: Any, root: int = 0) -> Any:
        if self.comm is None:
            return value
        return self.comm.bcast(value, root=root)

    def barrier(self) -> None:
        if self.comm is not None:
            self.comm.Barrier()

    def allreduce_max(self, value: int) -> int:
        if self.comm is None:
            return int(value)
        from mpi4py import MPI

        return int(self.comm.allreduce(int(value), op=MPI.MAX))

    def allreduce_max_float(self, value: float) -> float:
        """Return the largest floating-point value across ranks."""
        if self.comm is None:
            return float(value)
        from mpi4py import MPI

        return float(self.comm.allreduce(float(value), op=MPI.MAX))

    def allreduce_sum_float(self, value: float) -> float:
        """Return the floating-point sum across ranks."""
        if self.comm is None:
            return float(value)
        from mpi4py import MPI

        return float(self.comm.allreduce(float(value), op=MPI.SUM))

    def allgather(self, value: Any) -> list[Any]:
        if self.comm is None:
            return [value]
        return list(self.comm.allgather(value))
