"""Shared implementation behind ``slw_<stage>.x``."""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from threadpoolctl import threadpool_limits

from slw import __version__

from .adapter import (
    ArgumentAdapterError,
    arguments_from_parameters,
    invoke,
    parser_for,
)
from .logging import print_footer, print_header
from .mpi import MPIContext, MPIUnavailableError
from .namelist import NamelistError, loads
from .native import (
    NativeExecutionPlan,
    NativeHandlerError,
    native_help,
    prepare_native,
)
from .registry import (
    Backend,
    RegistryError,
    actions_for,
    find_action,
    resolve_action,
)
from .schema import STAGES, RunConfig, parse_run_config


@dataclass(frozen=True)
class ExecutionPlan:
    module: str
    all_ranks: bool
    warning: str | None = None


def _front_parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"slw_{stage}.x",
        description=(
            f"SLW {stage} stage. Read a QE-style namelist with -in FILE "
            "or from standard input."
        ),
    )
    parser.add_argument(
        "-in",
        "-inp",
        "-input",
        "--input",
        dest="input_file",
        metavar="FILE",
        help="Input namelist (default: standard input)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate the selected calculation without running it",
    )
    parser.add_argument(
        "--list-calculations",
        action="store_true",
        help="List calculations available in this stage",
    )
    parser.add_argument(
        "--legacy-help",
        "--help-calculation",
        metavar="CALCULATION",
        help="Show detailed input help for one calculation",
    )
    parser.add_argument(
        "--source",
        choices=("epr", "wannier"),
        help="Backend source for --legacy-help when a calculation has variants",
    )
    parser.add_argument("--version", action="version", version=f"SLW {__version__}")
    return parser


def _print_actions(stage: str) -> None:
    print(f"Calculations available in slw_{stage}.x:")
    width = max(len(action.name) for action in actions_for(stage))
    for action in actions_for(stage):
        suffix = ""
        if action.requires_source:
            suffix = f" [input_format: {'|'.join(action.backends)}]"
        print(f"  {action.name:<{width}}  {action.description}{suffix}")


def _print_setting(name: str, value: object) -> None:
    """Print one stable QE-style key/value setting line."""
    print(f"     {name:<15} = {value}")


def _read_input(path: str | None) -> str:
    if path is None or path == "-":
        if path is None and getattr(sys.stdin, "isatty", lambda: False)():
            raise NamelistError(
                "no input supplied; use -in FILE or pipe a namelist to standard input"
            )
        return sys.stdin.read()
    with Path(path).open("r", encoding="utf-8") as handle:
        return handle.read()


def _root_config(
    context: MPIContext,
    *,
    stage: str,
    input_file: str | None,
) -> RunConfig:
    payload: tuple[RunConfig | None, str | None]
    if context.is_root:
        try:
            config = parse_run_config(loads(_read_input(input_file)), stage=stage)
        except Exception as exc:  # noqa: BLE001 - broadcast parse failures to every rank
            payload = (None, f"{type(exc).__name__}: {exc}")
        else:
            payload = (config, None)
    else:
        payload = (None, None)
    config, error = context.bcast(payload, root=0)
    if error is not None:
        raise NamelistError(error)
    if config is None:  # pragma: no cover - defensive MPI guard
        raise RuntimeError("rank 0 did not broadcast an SLW configuration")
    return config


def _execution_plan(
    backend: Backend,
    *,
    requested: str,
    context: MPIContext,
) -> ExecutionPlan:
    if backend.module is None:
        raise RegistryError("native backends prepare their own execution plan")
    if requested == "mpi":
        if context.comm is None:
            raise MPIUnavailableError(
                "execution='mpi' requires mpi4py, even for a one-rank run"
            )
        if backend.mpi_module is None:
            raise RegistryError("the selected calculation has no MPI backend")
        return ExecutionPlan(module=backend.mpi_module, all_ranks=True)

    if requested == "serial":
        warning = None
        if context.size > 1:
            warning = "serial execution requested under MPI; only rank 0 will calculate"
        return ExecutionPlan(
            module=backend.module,
            all_ranks=False,
            warning=warning,
        )

    if context.size > 1 and backend.mpi_module is not None:
        return ExecutionPlan(module=backend.mpi_module, all_ranks=True)
    warning = None
    if context.size > 1:
        warning = "this calculation has no MPI backend; only rank 0 will calculate"
    return ExecutionPlan(module=backend.module, all_ranks=False, warning=warning)


@contextlib.contextmanager
def _workflow_environment(config: RunConfig, context: MPIContext):
    blas_threads = _runtime_blas_threads(config)
    numba_threads = config.parallel.effective_numba_threads
    updates = {
        "SLW_PREFIX": config.control.prefix,
        "SLW_OUTDIR": config.control.outdir,
        "SLW_SAVEDIR": config.control.savedir,
        "SLW_MPI_RANK": str(context.rank),
        "SLW_MPI_SIZE": str(context.size),
    }
    manage_compute_threads = config.stage in {"exchange", "magph"}
    if manage_compute_threads:
        updates.update(
            {
                "OMP_NUM_THREADS": str(config.parallel.threads_per_worker),
                "MKL_NUM_THREADS": str(blas_threads),
                "OPENBLAS_NUM_THREADS": str(blas_threads),
                "NUMEXPR_NUM_THREADS": str(blas_threads),
                "NUMBA_NUM_THREADS": str(numba_threads),
            }
        )
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        if manage_compute_threads:
            with threadpool_limits(limits=blas_threads):
                yield
        else:
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _runtime_blas_threads(config: RunConfig) -> int:
    """Resolve a safe rank-local BLAS limit for the selected algorithm."""
    explicit = config.parallel.blas_threads
    if explicit is not None:
        return explicit
    calculation = config.control.calculation
    if config.stage == "exchange":
        ltensor = config.parameters.get("ltensor", False)
        if calculation == "dj_tensor" or (calculation == "dj" and bool(ltensor)):
            return 1
    if config.stage == "magph" and calculation in {
        "scattering_kbz",
        "scattering_qbz",
        "rotational_coupling",
        "chirality_plane",
    }:
        return 1
    return config.parallel.threads_per_worker


def _uses_numba_threads(config: RunConfig) -> bool:
    calculation = config.control.calculation
    if config.stage == "exchange":
        return calculation in {"j_tensor", "dj_tensor"} or bool(
            config.parameters.get("ltensor", False)
        )
    return config.stage == "magph" and calculation not in {
        "lifetime",
        "prepare_lifetime",
    }


def _reject_unused_parallel_settings(
    calculation: str,
    config: RunConfig,
) -> None:
    parallel = config.parallel
    unsupported: list[str] = []
    if parallel.precache_workers is not None:
        unsupported.append("precache_workers")
    if parallel.workers_per_rank != 1 and calculation not in {
        "hybrid",
        "prepare_lifetime",
    }:
        unsupported.append("workers_per_rank")

    optional_chunks = {
        "q_chunk_size": parallel.q_chunk_size,
        "bond_chunk_size": parallel.bond_chunk_size,
        "vertex_q_chunk_size": parallel.vertex_q_chunk_size,
        "self_energy_q_chunk_size": parallel.self_energy_q_chunk_size,
        "channel_chunk_size": parallel.channel_chunk_size,
    }
    accepted_chunks = {
        "scattering_kbz": {"bond_chunk_size"},
        "scattering_qbz": {"bond_chunk_size", "vertex_q_chunk_size"},
        "rotational_coupling": {"q_chunk_size", "bond_chunk_size"},
        "chirality_plane": {"q_chunk_size", "bond_chunk_size"},
    }.get(calculation, set())
    unsupported.extend(
        name
        for name, value in optional_chunks.items()
        if value is not None and name not in accepted_chunks
    )
    if unsupported:
        rendered = ", ".join(sorted(set(unsupported)))
        raise ArgumentAdapterError(
            f"&parallel setting(s) are not used by magph calculation "
            f"{calculation!r}: {rendered}"
        )


def _legacy_parallel_parameters(
    stage: str,
    calculation: str,
    parameters: dict[str, object],
    config: RunConfig,
) -> dict[str, object]:
    """Translate common execution controls at the retained-driver boundary."""
    if stage != "magph":
        return parameters
    _reject_unused_parallel_settings(calculation, config)
    result = dict(parameters)
    parallel = config.parallel
    workers = parallel.workers_per_rank
    numba_threads = parallel.effective_numba_threads

    if calculation == "hybrid":
        result["phonon_nproc"] = workers
        result["hybrid_nproc"] = workers
    elif calculation in {"scattering_kbz", "scattering_qbz"}:
        result["nproc"] = numba_threads
    elif calculation in {"rotational_coupling", "chirality_plane"}:
        result["num_threads"] = numba_threads
        q_chunk = parallel.q_chunk_size
        bond_chunk = parallel.bond_chunk_size
        if q_chunk is not None:
            result["q_chunk"] = q_chunk
        if bond_chunk is not None:
            result["bond_chunk"] = bond_chunk
        if calculation == "chirality_plane":
            result["blas_threads"] = _runtime_blas_threads(config)
    elif calculation == "prepare_lifetime":
        result["phonon_nproc"] = workers

    if calculation in {"scattering_kbz", "scattering_qbz"}:
        q_chunk = (
            parallel.vertex_q_chunk_size
            if parallel.vertex_q_chunk_size is not None
            else parallel.q_chunk_size
        )
        bond_chunk = (
            parallel.bond_chunk_size if parallel.bond_chunk_size is not None else None
        )
        if q_chunk is not None and calculation == "scattering_qbz":
            result["vertex_q_chunk"] = q_chunk
        if bond_chunk is not None:
            result["vertex_bond_chunk"] = bond_chunk
    return result


def _invoke_root_only(
    context: MPIContext,
    *,
    module: str,
    argv: list[str],
    program: str,
    debug: bool,
) -> int:
    payload: tuple[int, str | None, str | None]
    if context.is_root:
        try:
            code = invoke(module, argv, program=program)
        except Exception as exc:  # noqa: BLE001 - keep non-root MPI ranks from hanging
            detail = traceback.format_exc() if debug else None
            payload = (1, f"{type(exc).__name__}: {exc}", detail)
        else:
            payload = (code, None, None)
    else:
        payload = (0, None, None)
    code, error, detail = context.bcast(payload, root=0)
    if context.is_root and error:
        print(f"Error in {program}: {error}", file=sys.stderr)
        if detail:
            print(detail.rstrip(), file=sys.stderr)
    return int(code)


def _invoke_all_ranks(
    context: MPIContext,
    *,
    module: str,
    argv: list[str],
    program: str,
    debug: bool,
) -> int:
    error: str | None = None
    detail: str | None = None
    try:
        local_code = invoke(module, argv, program=program)
    except Exception as exc:  # noqa: BLE001 - report the failing MPI rank collectively
        local_code = 1
        error = f"rank {context.rank}: {type(exc).__name__}: {exc}"
        if debug:
            detail = traceback.format_exc()

    code = context.allreduce_max(local_code)
    errors = context.allgather((error, detail))
    if context.is_root and code:
        for rank_error, rank_detail in errors:
            if rank_error:
                print(f"Error in {program}: {rank_error}", file=sys.stderr)
                if rank_detail:
                    print(rank_detail.rstrip(), file=sys.stderr)
    return code


def _invoke_native_root_only(
    context: MPIContext,
    *,
    plan: NativeExecutionPlan,
    program: str,
    debug: bool,
) -> int:
    payload: tuple[int, str | None, str | None]
    if context.is_root:
        try:
            result = plan.run()
            code = int(result) if isinstance(result, int) else 0
        except Exception as exc:  # noqa: BLE001 - broadcast failure to idle ranks
            detail = traceback.format_exc() if debug else None
            payload = (1, f"{type(exc).__name__}: {exc}", detail)
        else:
            payload = (code, None, None)
    else:
        payload = (0, None, None)
    code, error, detail = context.bcast(payload, root=0)
    if context.is_root and error:
        print(f"Error in {program}: {error}", file=sys.stderr)
        if detail:
            print(detail.rstrip(), file=sys.stderr)
    return int(code)


def _invoke_native_all_ranks(
    context: MPIContext,
    *,
    plan: NativeExecutionPlan,
    program: str,
    debug: bool,
) -> int:
    error: str | None = None
    detail: str | None = None
    try:
        result = plan.run()
        local_code = int(result) if isinstance(result, int) else 0
    except Exception as exc:  # noqa: BLE001 - report rank failures collectively
        local_code = 1
        error = f"rank {context.rank}: {type(exc).__name__}: {exc}"
        if debug:
            detail = traceback.format_exc()

    code = context.allreduce_max(local_code)
    errors = context.allgather((error, detail))
    if context.is_root and code:
        for rank_error, rank_detail in errors:
            if rank_error:
                print(f"Error in {program}: {rank_error}", file=sys.stderr)
                if rank_detail:
                    print(rank_detail.rstrip(), file=sys.stderr)
    return code


def _show_legacy_help(stage: str, calculation: str, source: str | None) -> int:
    if source is None:
        action = find_action(stage, calculation)
        native_handlers = {
            backend.handler for backend in action.backends.values() if backend.is_native
        }
        if len(native_handlers) == 1 and all(
            backend.is_native for backend in action.backends.values()
        ):
            requested_name = calculation.strip().lower().replace("-", "_")
            parameters = dict(action.alias_parameters.get(requested_name, {}))
            print(
                native_help(
                    next(iter(native_handlers)) or "",
                    calculation=action.name,
                    requested_name=requested_name,
                    source="|".join(action.backends),
                    parameters=parameters,
                ),
                end="",
            )
            return 0
    parameters = {"input_format": source} if source else {}
    resolved = resolve_action(
        stage,
        calculation,
        parameters,
        validate_requirements=False,
    )
    if resolved.backend.is_native:
        parameters = dict(resolved.parameters)
        parameters["input_format"] = resolved.source
        print(
            native_help(
                resolved.backend.handler or "",
                calculation=resolved.action.name,
                requested_name=resolved.requested_name,
                source=resolved.source,
                parameters=parameters,
            ),
            end="",
        )
        return 0
    if resolved.backend.module is None:  # pragma: no cover - dataclass invariant
        raise RegistryError("legacy backend has no module")
    parser = parser_for(resolved.backend.module)
    parser.prog = f"slw_{stage}.x ({calculation} backend)"
    print(parser.format_help(), end="")
    return 0


def _run(stage: str, argv: Sequence[str] | None) -> int:
    parser = _front_parser(stage)
    args = parser.parse_args(list(argv) if argv is not None else None)
    context = MPIContext.discover()

    if args.list_calculations:
        if context.is_root:
            _print_actions(stage)
        return 0
    if args.legacy_help:
        if context.is_root:
            return _show_legacy_help(stage, args.legacy_help, args.source)
        return 0

    config = _root_config(
        context,
        stage=stage,
        input_file=args.input_file,
    )
    resolved = resolve_action(
        stage,
        config.control.calculation,
        config.parameters,
    )
    dry_run = bool(args.dry_run or config.control.dry_run)
    program = f"slw_{stage}.x"

    native_plan: NativeExecutionPlan | None = None
    legacy_argv: list[str] = []
    if resolved.backend.is_native:
        parameters = dict(resolved.parameters)
        parameters["input_format"] = resolved.source
        native_plan = prepare_native(
            resolved.backend.handler or "",
            calculation=resolved.action.name,
            requested_name=resolved.requested_name,
            parameters=parameters,
            config=config,
            context=context,
            program=program,
        )
        backend_label = native_plan.backend_label
        all_ranks = native_plan.all_ranks
        warning = native_plan.warning
    else:
        plan = _execution_plan(
            resolved.backend,
            requested=config.parallel.execution,
            context=context,
        )
        legacy_parser = parser_for(plan.module)
        legacy_parser.prog = f"slw_{stage}.x ({resolved.action.name} backend)"
        legacy_parameters = _legacy_parallel_parameters(
            stage,
            resolved.action.name,
            dict(resolved.parameters),
            config,
        )
        legacy_argv = arguments_from_parameters(
            legacy_parser,
            legacy_parameters,
        )
        backend_label = plan.module
        all_ranks = plan.all_ranks
        warning = plan.warning

    if context.is_root:
        print_header(
            sys.stdout,
            program=program,
            version=__version__,
            calculation=resolved.action.name,
            prefix=config.control.prefix,
            outdir=config.control.outdir,
            savedir=config.control.savedir,
            mpi_size=context.size,
        )
        _print_setting("backend", backend_label)
        _print_setting("execution", "MPI" if all_ranks else "rank-0 serial")
        if stage in {"exchange", "magph"}:
            _print_setting("workers/rank", config.parallel.workers_per_rank)
            _print_setting("threads/worker", config.parallel.threads_per_worker)
            _print_setting("BLAS threads", _runtime_blas_threads(config))
            if _uses_numba_threads(config):
                _print_setting("Numba threads", config.parallel.effective_numba_threads)
        if resolved.action.requires_source:
            _print_setting("input format", resolved.source)
        if native_plan is not None:
            common_parallel_labels = {
                "workers/rank",
                "threads/worker",
                "BLAS threads",
                "Numba threads",
            }
            for key, value in native_plan.summary:
                if key in common_parallel_labels:
                    continue
                _print_setting(key, value)
        if warning:
            print(f"     WARNING: {warning}")
        if native_plan is None and (
            dry_run or config.control.verbosity in {"high", "debug"}
        ):
            _print_setting("command", shlex.join([program, *legacy_argv]))
        print(flush=True)

    if dry_run:
        if context.is_root:
            print(
                "Input and calculation parameters validated; calculation was not run."
            )
            print_footer(sys.stdout, program=program)
        return 0

    if config.control.create_save and context.is_root:
        Path(config.control.savedir).mkdir(parents=True, exist_ok=True)
    context.barrier()

    debug = config.control.verbosity == "debug"
    with _workflow_environment(config, context):
        if native_plan is not None and native_plan.all_ranks:
            code = _invoke_native_all_ranks(
                context,
                plan=native_plan,
                program=program,
                debug=debug,
            )
        elif native_plan is not None:
            code = _invoke_native_root_only(
                context,
                plan=native_plan,
                program=program,
                debug=debug,
            )
        elif all_ranks:
            code = _invoke_all_ranks(
                context,
                module=backend_label,
                argv=legacy_argv,
                program=program,
                debug=debug,
            )
        else:
            code = _invoke_root_only(
                context,
                module=backend_label,
                argv=legacy_argv,
                program=program,
                debug=debug,
            )

    if context.is_root and code == 0:
        print_footer(sys.stdout, program=program)
    return int(code)


def run_stage(stage: str, argv: Sequence[str] | None = None) -> int:
    """Run one of the four stage frontends and return a process exit code."""
    normalized = stage.strip().lower()
    if normalized not in STAGES:
        raise ValueError(f"unknown SLW stage: {stage}")
    try:
        return _run(normalized, argv)
    except (
        ArgumentAdapterError,
        MPIUnavailableError,
        NamelistError,
        NativeHandlerError,
        RegistryError,
    ) as exc:
        print(f"Error in slw_{normalized}.x: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Error in slw_{normalized}.x: {exc}", file=sys.stderr)
        return 2
