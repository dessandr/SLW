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
_PARALLEL_KEYS = {"execution"}


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
        raise NamelistError(
            f"unknown key(s) in &{group}: {', '.join(unknown)}"
        )


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
    execution: str


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
        raise NamelistError(
            "verbosity must be one of quiet, normal, high, or debug"
        )
    dry_run = _as_bool(control_raw.get("dry_run", False), name="dry_run")
    create_save = _as_bool(
        control_raw.get("create_save", True), name="create_save"
    )

    parallel_raw = dict(groups.get("parallel", {}))
    _unknown_keys(parallel_raw, _PARALLEL_KEYS, "parallel")
    execution = str(parallel_raw.get("execution", "auto")).strip().lower()
    if execution not in {"auto", "serial", "mpi"}:
        raise NamelistError("execution must be auto, serial, or mpi")

    generic = dict(groups.get("input", {}))
    specific = dict(groups.get(stage, {}))
    overlap = sorted(set(generic) & set(specific))
    if overlap:
        raise NamelistError(
            "parameters are defined in both &input and "
            f"&{stage}: {', '.join(overlap)}"
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
        parallel=ParallelConfig(execution=execution),
        parameters=parameters,
    )
