"""Validation and normalization for stage namelists."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .namelist import NamelistError, expand_placeholders

STAGES = ("epr", "exchange", "magph", "post")
_CONTROL_KEYS = {
    "calculation",
    "prefix",
    "outdir",
    "verbosity",
    "dry_run",
    "create_save",
}
_PARALLEL_KEYS = {
    "execution",
    "workers_per_rank",
    "threads_per_worker",
    "precache_workers",
    "blas_threads",
    "numba_threads",
    "q_chunk_size",
    "bond_chunk_size",
    "vertex_q_chunk_size",
    "self_energy_q_chunk_size",
    "channel_chunk_size",
}
_DEPRECATED_PARALLEL_KEYS = {
    "nproc": "workers_per_rank",
    "omp_threads": "threads_per_worker",
    "phonon_nproc": "workers_per_rank",
    "hybrid_nproc": "workers_per_rank",
    "num_threads": "numba_threads",
    "q_chunk": "q_chunk_size",
    "bond_chunk": "bond_chunk_size",
    "vertex_q_chunk": "vertex_q_chunk_size",
    "vertex_bond_chunk": "bond_chunk_size",
}
_STAGE_PARALLEL_KEYS = {
    "exchange": {
        "nproc",
        "omp_threads",
        "precache_workers",
        "blas_threads",
        "numba_threads",
        "workers_per_rank",
        "threads_per_worker",
    },
    "magph": {
        "nproc",
        "omp_threads",
        "phonon_nproc",
        "hybrid_nproc",
        "num_threads",
        "precache_workers",
        "blas_threads",
        "numba_threads",
        "workers_per_rank",
        "threads_per_worker",
        "q_chunk_size",
        "bond_chunk_size",
        "vertex_q_chunk_size",
        "self_energy_q_chunk_size",
        "channel_chunk_size",
        "q_chunk",
        "bond_chunk",
        "vertex_q_chunk",
        "vertex_bond_chunk",
    },
}


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", ".true.", "yes", "on", "1"}:
        return True
    if text in {"false", ".false.", "no", "off", "0"}:
        return False
    raise NamelistError(f"{name} must be a boolean, got {value!r}")


def _unknown_keys(values: dict[str, Any], allowed: set[str], group: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise NamelistError(f"unknown key(s) in &{group}: {', '.join(unknown)}")


@dataclass(frozen=True)
class ControlConfig:
    calculation: str
    prefix: str
    outdir: str
    savedir: str
    verbosity: str
    dry_run: bool
    create_save: bool


@dataclass(frozen=True)
class ParallelConfig:
    execution: str = "auto"
    workers_per_rank: int = 1
    threads_per_worker: int = 1
    precache_workers: int | None = None
    blas_threads: int | None = None
    numba_threads: int | None = None
    q_chunk_size: int | None = None
    bond_chunk_size: int | None = None
    vertex_q_chunk_size: int | None = None
    self_energy_q_chunk_size: int | None = None
    channel_chunk_size: int | None = None

    @property
    def effective_precache_workers(self) -> int:
        return (
            self.workers_per_rank
            if self.precache_workers is None
            else self.precache_workers
        )

    @property
    def effective_blas_threads(self) -> int:
        return (
            self.threads_per_worker if self.blas_threads is None else self.blas_threads
        )

    @property
    def effective_numba_threads(self) -> int:
        return (
            self.threads_per_worker
            if self.numba_threads is None
            else self.numba_threads
        )


@dataclass(frozen=True)
class RunConfig:
    stage: str
    control: ControlConfig
    parallel: ParallelConfig
    parameters: dict[str, Any]


def parse_run_config(
    groups: dict[str, dict[str, Any]],
    *,
    stage: str,
    cwd: str | os.PathLike[str] | None = None,
) -> RunConfig:
    stage = stage.lower()
    if stage not in STAGES:
        raise NamelistError(f"unknown SLW stage: {stage}")

    allowed_groups = {"control", "parallel", "input", stage}
    if stage == "exchange":
        allowed_groups.add("soc_card")
    unknown_groups = sorted(set(groups) - allowed_groups)
    if unknown_groups:
        rendered = ", ".join(f"&{name}" for name in unknown_groups)
        raise NamelistError(
            f"unexpected namelist group(s) for slw_{stage}.x: {rendered}"
        )

    control_raw = dict(groups.get("control", {}))
    _unknown_keys(control_raw, _CONTROL_KEYS, "control")
    calculation = str(control_raw.get("calculation", "")).strip().lower()
    if not calculation:
        raise NamelistError("&control must define calculation")

    prefix = str(control_raw.get("prefix", "slw")).strip()
    if not prefix:
        raise NamelistError("prefix cannot be empty")
    if prefix in {".", ".."} or any(sep in prefix for sep in ("/", "\\")):
        raise NamelistError("prefix must be a file-name prefix, not a path")

    base = Path(cwd or os.getcwd()).expanduser().resolve()
    outdir_raw = str(control_raw.get("outdir", ".")).strip()
    outdir_path = Path(outdir_raw).expanduser()
    if not outdir_path.is_absolute():
        outdir_path = base / outdir_path
    outdir = str(outdir_path.resolve())
    savedir = str((outdir_path / f"{prefix}.save").resolve())

    verbosity = str(control_raw.get("verbosity", "normal")).strip().lower()
    if verbosity not in {"quiet", "normal", "high", "debug"}:
        raise NamelistError("verbosity must be one of quiet, normal, high, or debug")
    dry_run = _as_bool(control_raw.get("dry_run", False), name="dry_run")
    create_save = _as_bool(control_raw.get("create_save", True), name="create_save")

    parallel_raw = dict(groups.get("parallel", {}))
    deprecated_parallel = sorted(set(parallel_raw) & set(_DEPRECATED_PARALLEL_KEYS))
    if deprecated_parallel:
        rendered = ", ".join(
            f"{key}->{_DEPRECATED_PARALLEL_KEYS[key]}" for key in deprecated_parallel
        )
        raise NamelistError(f"deprecated &parallel key(s): {rendered}")
    _unknown_keys(parallel_raw, _PARALLEL_KEYS, "parallel")
    execution = str(parallel_raw.get("execution", "auto")).strip().lower()
    if execution not in {"auto", "serial", "mpi"}:
        raise NamelistError("execution must be auto, serial, or mpi")

    def positive_parallel_integer(name: str, *, optional: bool) -> int | None:
        raw = parallel_raw.get(name)
        if raw is None and optional:
            return None
        if raw is None:
            raw = 1
        if isinstance(raw, bool):
            raise NamelistError(f"&parallel {name} must be a positive integer")
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as exc:
            raise NamelistError(f"&parallel {name} must be a positive integer") from exc
        if not numeric.is_integer() or numeric < 1.0:
            raise NamelistError(f"&parallel {name} must be a positive integer")
        return int(numeric)

    workers_per_rank = positive_parallel_integer("workers_per_rank", optional=False)
    threads_per_worker = positive_parallel_integer("threads_per_worker", optional=False)
    assert workers_per_rank is not None
    assert threads_per_worker is not None
    optional_parallel = {
        name: positive_parallel_integer(name, optional=True)
        for name in _PARALLEL_KEYS
        if name
        not in {
            "execution",
            "workers_per_rank",
            "threads_per_worker",
        }
    }

    generic = dict(groups.get("input", {}))
    specific = dict(groups.get(stage, {}))
    misplaced = sorted(
        (set(generic) | set(specific)) & _STAGE_PARALLEL_KEYS.get(stage, set())
    )
    if misplaced:
        rendered_misplaced = ", ".join(
            (
                f"{name}->{_DEPRECATED_PARALLEL_KEYS[name]}"
                if name in _DEPRECATED_PARALLEL_KEYS
                else name
            )
            for name in misplaced
        )
        raise NamelistError(
            "parallel setting(s) belong in &parallel, not the stage/input "
            f"namelist: {rendered_misplaced}"
        )
    overlap = sorted(set(generic) & set(specific))
    if overlap:
        raise NamelistError(
            f"parameters are defined in both &input and &{stage}: {', '.join(overlap)}"
        )
    parameters = {**generic, **specific}
    if "soc_card" in groups:
        parameters["soc_card"] = groups["soc_card"]
    variables = {"prefix": prefix, "outdir": outdir, "savedir": savedir}
    parameters = expand_placeholders(parameters, variables)

    return RunConfig(
        stage=stage,
        control=ControlConfig(
            calculation=calculation,
            prefix=prefix,
            outdir=outdir,
            savedir=savedir,
            verbosity=verbosity,
            dry_run=dry_run,
            create_save=create_save,
        ),
        parallel=ParallelConfig(
            execution=execution,
            workers_per_rank=workers_per_rank,
            threads_per_worker=threads_per_worker,
            precache_workers=optional_parallel["precache_workers"],
            blas_threads=optional_parallel["blas_threads"],
            numba_threads=optional_parallel["numba_threads"],
            q_chunk_size=optional_parallel["q_chunk_size"],
            bond_chunk_size=optional_parallel["bond_chunk_size"],
            vertex_q_chunk_size=optional_parallel["vertex_q_chunk_size"],
            self_energy_q_chunk_size=optional_parallel["self_energy_q_chunk_size"],
            channel_chunk_size=optional_parallel["channel_chunk_size"],
        ),
        parameters=parameters,
    )
