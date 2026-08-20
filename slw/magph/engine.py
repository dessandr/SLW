"""Native MPI-aware orchestration for magnon self-energy and lifetime."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from slw.cli.logging import RunLogger, TimerBook
from slw.cli.mpi import MPIContext
from slw.cli.native import NativeExecutionPlan
from slw.cli.schema import RunConfig

from .config import MagphInputError, MagphLifetimeRequest, build_lifetime_request
from .coupling import build_mode_resolved_isotropic_derivative
from .derivative import load_exchange_derivative_h5
from .lswt import uniform_fractional_mesh
from .output import write_lifetime_npz
from .parallel import CollectiveExecutionError, RankFailure
from .phonon import load_phonon_cache, zero_point_displacements
from .pipeline import compute_lifetime_grid
from .screening import load_exchange_h5, screen_magnetic_configuration


@dataclass(frozen=True)
class MagphRunResult:
    output: Path
    mpi_size: int
    k_point_count: int
    magnetic_site_count: int


def _rank_failure(rank: int, stage: str, exc: BaseException) -> RankFailure:
    return RankFailure(
        rank=int(rank),
        stage=stage,
        exception_type=type(exc).__name__,
        message=str(exc).strip() or repr(exc),
    )


def _load_native_problem(request: MagphLifetimeRequest) -> tuple[Any, ...]:
    exchange, exchange_report = load_exchange_h5(request.exchange_h5)
    derivative, derivative_report = load_exchange_derivative_h5(
        request.derivative_h5,
        exchange,
        asr_policy=request.asr_policy,
    )
    phonons = load_phonon_cache(request.phonon_cache)
    zero_point = zero_point_displacements(
        phonons,
        frequency_floor_mev=request.frequency_floor_mev,
    )
    spin_magnitudes: float | tuple[float, ...] = request.spin_magnitudes
    if len(request.spin_magnitudes) == 1:
        spin_magnitudes = request.spin_magnitudes[0]
    configuration = screen_magnetic_configuration(
        exchange,
        order=request.magnetic_order,
        spin_pattern=request.spin_pattern,
        spin_magnitudes=spin_magnitudes,
        quantization_axis=request.quantization_axis,
    )
    coupling = build_mode_resolved_isotropic_derivative(
        exchange,
        derivative,
        phonons,
        zero_point,
        require_complete_targets=request.require_complete_targets,
        q_chunk_size=request.q_chunk_size,
        bond_chunk_size=request.bond_chunk_size,
    )
    k_points = uniform_fractional_mesh(request.kmesh, shift=request.kshift)
    return (
        exchange,
        exchange_report,
        derivative,
        derivative_report,
        phonons,
        configuration,
        coupling,
        k_points,
    )


def _load_collectively(
    request: MagphLifetimeRequest,
    context: MPIContext,
) -> tuple[Any, ...]:
    loaded: tuple[Any, ...] | None = None
    failure: RankFailure | None = None
    try:
        loaded = _load_native_problem(request)
    except Exception as exc:  # noqa: BLE001 - exchange before next collective
        failure = _rank_failure(context.rank, "magph_input_screening", exc)
    failures = tuple(item for item in context.allgather(failure) if item is not None)
    if failures:
        raise CollectiveExecutionError(failures)
    if loaded is None:  # pragma: no cover - guarded by collective failure
        raise RuntimeError("native magph input produced no result or failure")
    return loaded


def _output_metadata(
    request: MagphLifetimeRequest,
    *,
    exchange: Any,
    exchange_report: Any,
    derivative: Any,
    derivative_report: Any,
    phonons: Any,
    configuration: Any,
    mpi_size: int,
) -> dict[str, Any]:
    return {
        "calculation": "lifetime",
        "exchange_h5": str(request.exchange_h5),
        "derivative_h5": str(request.derivative_h5),
        "phonon_cache": str(request.phonon_cache),
        "exchange_dataset": exchange_report.source_dataset,
        "exchange_representation": exchange_report.representation.value,
        "exchange_convention_origin": exchange_report.convention_origin,
        "exchange_kernel_family": exchange.convention.kernel_family,
        "exchange_realspace_gauge": exchange.convention.realspace_gauge,
        "exchange_capabilities": [
            capability.value for capability in exchange_report.capabilities
        ],
        "derivative_dataset": derivative_report.source_dataset,
        "derivative_asr_policy": derivative_report.asr_policy.value,
        "derivative_asr_projected": bool(derivative_report.projected_asr),
        "derivative_max_asr_residual_mev_per_ang": float(
            derivative_report.max_asr_residual_after_mev_per_ang
        ),
        "derivative_fourier_phase_convention": derivative.fourier_phase_convention,
        "derivative_directed_bond_mate": derivative.directed_bond_mate,
        "phonon_schema_version": int(phonons.schema_version),
        "phonon_mass_unit": phonons.mass_unit.value,
        "phonon_vector_convention": phonons.vector_convention,
        "phonon_fourier_phase_convention": phonons.fourier_phase_convention,
        "frequency_floor_mev": request.frequency_floor_mev,
        "magnetic_order": request.magnetic_order.value,
        "spin_magnitudes": configuration.spin_magnitudes.tolist(),
        "spin_pattern": configuration.spin_pattern.tolist(),
        "quantization_axis": configuration.quantization_axis.tolist(),
        "kmesh": list(request.kmesh),
        "kshift": list(request.kshift),
        "q_weight_policy": "uniform_normalized",
        "metric_energy_tolerance_mev": request.metric_energy_tolerance_mev,
        "require_complete_targets": request.require_complete_targets,
        "chunk_sizes": {
            "mode_q": request.q_chunk_size,
            "mode_bond": request.bond_chunk_size,
            "vertex_q": request.vertex_q_chunk_size,
            "self_energy_q": request.self_energy_q_chunk_size,
            "self_energy_channel": request.channel_chunk_size,
        },
        "mpi_size": int(mpi_size),
    }


def run_lifetime(
    request: MagphLifetimeRequest,
    *,
    context: MPIContext | None = None,
    program: str = "slw_magph.x",
    verbosity: str = "normal",
) -> MagphRunResult:
    """Run the complete native lifetime route, discovering MPI by default."""

    mpi = MPIContext.discover() if context is None else context
    if mpi.size > 1:
        request = mpi.bcast(request if mpi.is_root else None, root=0)
        if request is None:  # pragma: no cover - communicator contract guard
            raise RuntimeError("rank zero did not broadcast the lifetime request")
    timers = TimerBook()
    logger = RunLogger(
        sys.stdout,
        rank=mpi.rank,
        verbosity=verbosity,
        timers=timers,
    )
    with timers.phase("total"):
        with logger.phase("input_screening", label="Screening native inputs"):
            (
                exchange,
                exchange_report,
                derivative,
                derivative_report,
                phonons,
                configuration,
                coupling,
                k_points,
            ) = _load_collectively(request, mpi)
        logger.info(f"magnetic order = {configuration.order.value}")
        logger.info(f"magnetic sites = {configuration.n_magnetic_sites}")
        logger.info(f"k-point count  = {k_points.shape[0]}")
        logger.info(f"phonon q count = {coupling.nq}")
        logger.info(
            "k distribution = balanced contiguous rank blocks; "
            "q/mode contraction = rank-local vectorized"
        )

        with logger.phase(
            "self_energy",
            label="Computing on-shell magnon self-energy and lifetime",
        ):
            distributed = compute_lifetime_grid(
                exchange,
                configuration,
                coupling,
                k_points,
                temperature_k=request.temperature_k,
                broadening_mev=request.broadening_mev,
                metric_energy_tolerance_mev=request.metric_energy_tolerance_mev,
                negative_tolerance_mev=request.negative_tolerance_mev,
                vertex_q_chunk_size=request.vertex_q_chunk_size,
                self_energy_q_chunk_size=request.self_energy_q_chunk_size,
                channel_chunk_size=request.channel_chunk_size,
                context=mpi,
            )

        output_failure: RankFailure | None = None
        if mpi.is_root:
            try:
                if distributed.global_result is None:
                    raise RuntimeError("rank zero did not receive the lifetime grid")
                metadata = _output_metadata(
                    request,
                    exchange=exchange,
                    exchange_report=exchange_report,
                    derivative=derivative,
                    derivative_report=derivative_report,
                    phonons=phonons,
                    configuration=configuration,
                    mpi_size=mpi.size,
                )
                with logger.phase("output", label="Writing lifetime output"):
                    write_lifetime_npz(
                        request.output,
                        distributed.global_result,
                        metadata=metadata,
                        overwrite=request.overwrite,
                    )
            except Exception as exc:  # noqa: BLE001 - release non-root ranks
                output_failure = _rank_failure(mpi.rank, "magph_output", exc)
        output_failure = mpi.bcast(output_failure, root=0)
        if output_failure is not None:
            raise CollectiveExecutionError((output_failure,))
        logger.info(f"written       = {request.output}")

    total = timers.get("total")
    if total is not None and mpi.size > 1:
        timers.record(
            "mpi_total",
            cpu_seconds=mpi.allreduce_sum_float(total.cpu_seconds),
            wall_seconds=mpi.allreduce_max_float(total.wall_seconds),
        )
        logger.timing_summary(program, phase="mpi_total")
    else:
        logger.timing_summary(program, phase="total")
    return MagphRunResult(
        output=request.output,
        mpi_size=mpi.size,
        k_point_count=int(np.prod(np.asarray(request.kmesh, dtype=np.int64))),
        magnetic_site_count=configuration.n_magnetic_sites,
    )


def _parallel_policy(config: RunConfig, context: MPIContext) -> tuple[bool, str | None]:
    requested = config.parallel.execution
    if requested == "mpi":
        if context.comm is None:
            raise MagphInputError(
                "execution='mpi' requires mpi4py, even for a one-rank run"
            )
        return True, None
    if requested == "serial":
        warning = None
        if context.size > 1:
            warning = "serial execution requested under MPI; only rank 0 will calculate"
        return False, warning
    return (context.size > 1), None


def prepare_run(
    *,
    calculation: str,
    requested_name: str,
    parameters: dict[str, Any],
    config: RunConfig,
    context: MPIContext,
    program: str,
) -> NativeExecutionPlan:
    """Prepare the native lifetime calculation for the common CLI runner."""

    if calculation != "lifetime" or requested_name != "lifetime":
        raise MagphInputError(
            f"native magph engine does not implement {requested_name!r}"
        )
    request = build_lifetime_request(
        parameters,
        prefix=config.control.prefix,
        savedir=config.control.savedir,
    )
    all_ranks, warning = _parallel_policy(config, context)
    execution_context = context if all_ranks else MPIContext()

    def run() -> int:
        run_lifetime(
            request,
            context=execution_context,
            program=program,
            verbosity=config.control.verbosity,
        )
        return 0

    return NativeExecutionPlan(
        backend_label="slw.magph.engine (native lifetime)",
        run=run,
        all_ranks=all_ranks,
        warning=warning,
        summary=(
            ("magnetic order", request.magnetic_order.value),
            ("k-point mesh", " x ".join(map(str, request.kmesh))),
            ("k-grid shift", ", ".join(map(str, request.kshift))),
            ("output", request.output),
        ),
    )


def format_help(
    *,
    calculation: str,
    requested_name: str,
    source: str,
    parameters: dict[str, Any],
) -> str:
    del calculation, requested_name, source, parameters
    return (
        "slw_magph.x calculation='lifetime' (native)\n\n"
        "Required &magph keys:\n"
        "  exchange_h5       = canonical static J HDF5\n"
        "  derivative_h5     = scalar dJ/du HDF5\n"
        "  phonon_cache      = schema-v3 phonon NPZ\n"
        "  magnetic_order    = 'fm' or 'collinear_afm'\n"
        "  spin_magnitudes   = scalar or one value per magnetic site\n"
        "  quantization_axis = three Cartesian components\n"
        "  kmesh             = three positive integers\n"
        "  kshift            = explicit three-component grid-unit shift\n"
        "  temperature_k     = non-negative temperature\n"
        "  broadening_mev    = positive retarded broadening\n\n"
        "Optional: spin_pattern, frequency_floor_mev, asr_policy, output,\n"
        "overwrite, and q/bond/vertex/self-energy/channel chunk sizes.\n"
        "MPI is selected automatically under mpirun; external k points are\n"
        "distributed across ranks and q/mode contractions stay vectorized.\n"
    )


__all__ = ["MagphRunResult", "prepare_run", "run_lifetime"]
