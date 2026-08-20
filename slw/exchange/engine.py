"""Native orchestration for static and displacement-derivative exchange.

The public mode space is intentionally small: ``j`` or ``dj`` combined with
``ltensor``.  Scalar and tensor integrands remain separate numerical kernels;
the boolean is interpreted only at this dispatch boundary because it changes
the LKAG/TB2J convention, not merely the result shape.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

from slw.cli.logging import RunLogger, TimerBook
from slw.cli.mpi import MPIContext
from slw.cli.native import NativeExecutionPlan
from slw.cli.schema import RunConfig

from .config import ExchangeInputError, build_exchange_request
from .kernels.dispatch import KernelArtifacts
from .kernels.dispatch import execute as execute_kernel
from .model import ExchangeCalculation, ExchangeRequest

_IMPORTANT_BACKEND_MESSAGES = (
    "build",
    "load",
    "setup",
    "comput",
    "diagonal",
    "precache",
    "assemble",
    "completed",
    "checkpoint",
    "wrote",
    "done",
)
_BACKEND_PREFIX = re.compile(r"^(?:\[[^]]+\])+\s*")


@dataclass(frozen=True)
class ExchangeRunResult:
    """Completed native-engine run with explicit products and timings."""

    request: ExchangeRequest
    artifacts: tuple[str, ...]
    cpu_seconds: float
    wall_seconds: float
    mpi_size: int


class _BackendOutput(io.TextIOBase):
    """Turn native-kernel prints into stable rank-aware log lines."""

    def __init__(self, logger: RunLogger) -> None:
        super().__init__()
        self._logger = logger
        self._buffer = ""
        self._lock = threading.RLock()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        value = str(text)
        with self._lock:
            self._buffer += value
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit(line)
        return len(value)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                self._emit(self._buffer)
                self._buffer = ""

    def _emit(self, line: str) -> None:
        clean = _BACKEND_PREFIX.sub("", line.strip())
        if not clean:
            return
        lowered = clean.lower()
        if "warn" in lowered:
            self._logger.warning(clean)
        elif any(token in lowered for token in _IMPORTANT_BACKEND_MESSAGES):
            self._logger.info(clean)
        else:
            self._logger.high(clean)


def _mode_label(request: ExchangeRequest) -> str:
    quantity = "dJ/du" if request.calculation is ExchangeCalculation.DJ else "J"
    form = "tensor" if request.ltensor else "scalar"
    return f"{form} {quantity}"


def _mpi_capable(request: ExchangeRequest) -> bool:
    del request
    return True


def _parallel_policy(
    request: ExchangeRequest,
    *,
    requested: str,
    context: MPIContext,
) -> tuple[bool, str | None]:
    capable = _mpi_capable(request)
    if requested == "mpi":
        if context.comm is None:
            raise ExchangeInputError(
                "execution='mpi' requires mpi4py, even for a one-rank run"
            )
        return True, None

    if requested == "serial":
        warning = None
        if context.size > 1:
            warning = "serial execution requested under MPI; only rank 0 will calculate"
        return False, warning

    if context.size > 1 and capable:
        return True, None
    if context.size > 1:
        return False, "this exchange mode is serial; only rank 0 will calculate"
    return False, None


def _validate_runtime(request: ExchangeRequest, *, all_ranks: bool) -> None:
    if (
        request.calculation is ExchangeCalculation.DJ
        and request.ltensor
        and request.tensor_kernel.value != "tb2j"
    ):
        raise ExchangeInputError("tensor dJ currently supports tensor_kernel='tb2j' only")
    if all_ranks:
        nproc = int(request.options.as_dict().get("nproc", 1))
        if nproc != 1:
            raise ExchangeInputError(
                "MPI exchange requires nproc=1 per rank to avoid nested process "
                "oversubscription; use numba_threads/blas_threads for rank-local work"
            )


def _check_artifacts(artifacts: KernelArtifacts) -> None:
    missing = [path for path in artifacts.paths if not Path(path).is_file()]
    if missing:
        rendered = ", ".join(missing)
        raise RuntimeError(f"exchange kernel did not create expected output(s): {rendered}")


def _execute(
    request: ExchangeRequest,
    *,
    context: MPIContext,
    program: str,
    verbosity: str,
    all_ranks: bool,
) -> ExchangeRunResult:
    timers = TimerBook()
    logger = RunLogger(
        sys.stdout,
        rank=context.rank,
        verbosity=verbosity,
        timers=timers,
    )
    output = _BackendOutput(logger)
    artifacts: KernelArtifacts

    with timers.phase("total"):
        logger.info("Exchange calculation")
        logger.info(f"mode          = {_mode_label(request)}")
        logger.info(f"k-point mesh  = {' x '.join(map(str, request.kmesh))}")
        logger.info(
            "magnetic atoms = " + ", ".join(str(item) for item in request.mag_atoms)
        )
        if request.ltensor:
            logger.info(f"tensor kernel = {request.tensor_kernel.value}")
        logger.high("numerical backend = native exchange kernels")
        with logger.phase(
            "exchange_kernel",
            label=f"Running {_mode_label(request)} kernel",
        ):
            try:
                with contextlib.redirect_stdout(cast(TextIO, output)):
                    artifacts = execute_kernel(
                        request,
                        mpi=all_ranks,
                        comm=context.comm if all_ranks else None,
                    )
            finally:
                output.flush()
        validation_error: str | None = None
        with logger.phase(
            "output_validation",
            label="Validating exchange output",
            level="high",
        ):
            if context.is_root:
                try:
                    _check_artifacts(artifacts)
                except Exception as exc:  # noqa: BLE001 - broadcast before collective
                    validation_error = f"{type(exc).__name__}: {exc}"
        if all_ranks:
            validation_error = context.bcast(validation_error, root=0)
        if validation_error is not None:
            raise RuntimeError(validation_error)
        if context.is_root:
            logger.info("Exchange output")
            for path in artifacts.paths:
                logger.info(f"written       = {path}")

    total = timers.get("total")
    timing_phase = "total"
    summary_cpu = total.cpu_seconds if total is not None else 0.0
    summary_wall = total.wall_seconds if total is not None else 0.0
    if all_ranks and total is not None:
        wall = context.allreduce_max_float(total.wall_seconds)
        cpu = context.allreduce_sum_float(total.cpu_seconds)
        timers.record("mpi_total", cpu_seconds=cpu, wall_seconds=wall)
        timing_phase = "mpi_total"
        summary_cpu = cpu
        summary_wall = wall
    logger.timing_summary(program, phase=timing_phase)
    return ExchangeRunResult(
        request=request,
        artifacts=artifacts.paths,
        cpu_seconds=summary_cpu,
        wall_seconds=summary_wall,
        mpi_size=context.size if all_ranks else 1,
    )


def run_exchange(
    request: ExchangeRequest,
    *,
    context: MPIContext | None = None,
    execution: str = "auto",
    program: str = "slw_exchange.x",
    verbosity: str = "normal",
) -> ExchangeRunResult:
    """Run a validated request without entering any command-line parser.

    Under MPI this function is collective for every exchange mode. Static J
    and scalar dJ distribute energy points, while tensor dJ distributes
    target-axis tasks. Pass an explicit size-one ``MPIContext`` to opt out of
    launch discovery.
    """

    mpi = context if context is not None else MPIContext.discover()
    requested = str(execution).strip().lower()
    if mpi.size > 1:
        control_payload = (
            (request, requested, str(program), str(verbosity))
            if mpi.is_root
            else None
        )
        canonical = mpi.bcast(control_payload, root=0)
        if canonical is None:  # pragma: no cover - defensive communicator guard
            raise RuntimeError("rank 0 did not broadcast exchange control data")
        request, requested, program, verbosity = canonical
    if requested not in {"auto", "serial", "mpi"}:
        raise ExchangeInputError("execution must be auto, serial, or mpi")
    all_ranks, _warning = _parallel_policy(
        request,
        requested=requested,
        context=mpi,
    )
    _validate_runtime(request, all_ranks=all_ranks)
    if all_ranks or mpi.size == 1:
        return _execute(
            request,
            context=mpi,
            program=program,
            verbosity=verbosity,
            all_ranks=all_ranks,
        )

    payload: tuple[ExchangeRunResult | None, str | None]
    if mpi.is_root:
        try:
            result = _execute(
                request,
                context=mpi,
                program=program,
                verbosity=verbosity,
                all_ranks=False,
            )
        except Exception as exc:  # noqa: BLE001 - release all collective callers
            payload = (None, f"{type(exc).__name__}: {exc}")
        else:
            payload = (result, None)
    else:
        payload = (None, None)
    result, error = mpi.bcast(payload, root=0)
    if error is not None:
        raise RuntimeError(f"exchange calculation failed on rank 0: {error}")
    if result is None:  # pragma: no cover - defensive communicator guard
        raise RuntimeError("rank 0 did not broadcast an exchange result")
    return result


def prepare_run(
    *,
    calculation: str,
    requested_name: str,
    parameters: dict[str, Any],
    config: RunConfig,
    context: MPIContext,
    program: str,
) -> NativeExecutionPlan:
    """Validate one native exchange input and return an executable plan."""

    request = build_exchange_request(
        requested_name if requested_name in {"j_tensor", "dj_tensor"} else calculation,
        parameters,
        prefix=config.control.prefix,
        savedir=config.control.savedir,
    )
    all_ranks, warning = _parallel_policy(
        request,
        requested=config.parallel.execution,
        context=context,
    )
    _validate_runtime(request, all_ranks=all_ranks)
    mode = _mode_label(request)
    summary: tuple[tuple[str, Any], ...] = (
        ("form", "tensor" if request.ltensor else "scalar"),
        ("k-point mesh", " x ".join(map(str, request.kmesh))),
        ("magnetic sites", len(request.mag_atoms)),
        ("output", request.output.h5_path),
    )

    def run() -> int:
        _execute(
            request,
            context=context,
            program=program,
            verbosity=config.control.verbosity,
            all_ranks=all_ranks,
        )
        return 0

    return NativeExecutionPlan(
        backend_label=f"slw.exchange.engine ({mode}, {request.source.value})",
        run=run,
        all_ranks=all_ranks,
        warning=warning,
        summary=summary,
    )


def format_help(
    *,
    calculation: str,
    requested_name: str,
    source: str,
    parameters: dict[str, Any],
) -> str:
    """Return concise native exchange input help for the common frontend."""

    del requested_name, parameters
    source_label = " or ".join(f"'{item}'" for item in source.split("|"))
    lines = [
        f"slw_exchange.x calculation='{calculation}' input_format={source_label}",
        "",
        "Required &exchange keys:",
        "  input_format  = 'epr' or 'wannier' (dj accepts epr only)",
        "  efermi        = Fermi energy in eV",
        "  kmesh         = three positive integers",
        "  mag_atoms     = explicit magnetic atom indices",
        "  slices        = 'site:start:stop,...'",
        "  epr_up/epr_dn = required for EPR input",
        "  up_hr/dn_hr   = required for scalar Wannier input",
        "",
        "Mode selection:",
        "  ltensor       = .false. for scalar, .true. for tensor",
        "  tensor_kernel = 'tb2j' or 'direct' for tensor J",
        "",
        "Output defaults to ${savedir}/${prefix}.<mode>.{h5,txt}.",
        "See docs/INPUT_REFERENCE.md for advanced numerical options.",
        "",
    ]
    return "\n".join(lines)


__all__ = ["ExchangeRunResult", "prepare_run", "run_exchange"]
