"""Native MPI-aware orchestration for magnon self-energy and lifetime."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from socket import gethostname
from typing import Any

import numpy as np

from slw.cli.logging import RunLogger, TimerBook
from slw.cli.mpi import MPIContext
from slw.cli.native import NativeExecutionPlan
from slw.cli.schema import ParallelConfig, RunConfig

from .config import MagphInputError, MagphLifetimeRequest, build_lifetime_request
from .coupling import build_mode_resolved_isotropic_derivative_distributed
from .derivative import load_exchange_derivative_h5
from .lswt import uniform_fractional_mesh
from .mesh import build_magnon_mesh_cache
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


def _validate_parallel(parallel: ParallelConfig) -> None:
    if parallel.workers_per_rank != 1:
        raise MagphInputError(
            "native lifetime currently requires &parallel workers_per_rank=1; "
            "use MPI ranks over external k and threads_per_worker within each rank"
        )
    if parallel.precache_workers is not None:
        raise MagphInputError("native lifetime does not use &parallel precache_workers")
    if parallel.numba_threads is not None:
        raise MagphInputError(
            "native lifetime does not use &parallel numba_threads; "
            "use threads_per_worker or blas_threads for vectorized contractions"
        )


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
    k_points = uniform_fractional_mesh(request.kmesh, shift=request.kshift)
    return (
        exchange,
        exchange_report,
        derivative,
        derivative_report,
        phonons,
        zero_point,
        configuration,
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


def _array_storage_bytes(*values: Any) -> int:
    return sum(int(np.asarray(value).nbytes) for value in values)


def _format_storage(byte_count: int) -> str:
    value = float(byte_count)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for candidate in units:
        unit = candidate
        if value < 1024.0 or candidate == units[-1]:
            break
        value /= 1024.0
    return f"{value:.2f} {unit}"


def _cache_storage(coupling: Any, magnon_cache: Any) -> tuple[int, int]:
    coupling_bytes = _array_storage_bytes(
        coupling.q_points_frac,
        coupling.phonon_energy_mev,
        coupling.lambda_mev,
        coupling.target_atom_indices,
        coupling.frequency_regularized,
    )
    lswt_bytes = _array_storage_bytes(
        magnon_cache.union_points_frac,
        magnon_cache.signed_energies_mev,
        magnon_cache.transformation,
        magnon_cache.metric,
        magnon_cache.external_k_grid_indices,
        magnon_cache.q_grid_indices,
    )
    return coupling_bytes, lswt_bytes


def _output_metadata(
    request: MagphLifetimeRequest,
    *,
    parallel: ParallelConfig,
    exchange: Any,
    exchange_report: Any,
    derivative: Any,
    derivative_report: Any,
    phonons: Any,
    configuration: Any,
    coupling: Any,
    magnon_cache: Any,
    mpi_size: int,
) -> dict[str, Any]:
    coupling_bytes, lswt_bytes = _cache_storage(coupling, magnon_cache)
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
            "mode_q": parallel.q_chunk_size,
            "mode_bond": parallel.bond_chunk_size,
            "vertex_q": parallel.vertex_q_chunk_size,
            "self_energy_q": parallel.self_energy_q_chunk_size,
            "self_energy_channel": parallel.channel_chunk_size,
        },
        "parallel": {
            "workers_per_rank": parallel.workers_per_rank,
            "threads_per_worker": parallel.threads_per_worker,
            "blas_threads": parallel.effective_blas_threads,
        },
        "algorithm": {
            "coupling": "mpi_q_distributed_cache",
            "lswt": "uniform_k_plus_q_union_cache",
            "vertex_self_energy": "q_block_streaming",
            "full_vertex_materialized": False,
            "mpi_distribution": "external_k",
        },
        "union_kq_mesh": [
            int(np.lcm(k_value, q_value))
            for k_value, q_value in zip(
                request.kmesh,
                derivative.q_mesh_shape,
                strict=True,
            )
        ],
        "cache_bytes_per_rank": {
            "coupling": coupling_bytes,
            "lswt": lswt_bytes,
            "total": coupling_bytes + lswt_bytes,
        },
        "mpi_size": int(mpi_size),
    }


def run_lifetime(
    request: MagphLifetimeRequest,
    *,
    context: MPIContext | None = None,
    parallel: ParallelConfig | None = None,
    program: str = "slw_magph.x",
    verbosity: str = "normal",
) -> MagphRunResult:
    """Run the complete native lifetime route, discovering MPI by default."""

    mpi = MPIContext.discover() if context is None else context
    runtime = ParallelConfig() if parallel is None else parallel
    _validate_parallel(runtime)
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
                zero_point,
                configuration,
                k_points,
            ) = _load_collectively(request, mpi)
        logger.info(f"magnetic order = {configuration.order.value}")
        logger.info(f"magnetic sites = {configuration.n_magnetic_sites}")
        logger.info(f"k-point count  = {k_points.shape[0]}")
        logger.info(f"phonon q count = {phonons.nq}")

        with logger.phase(
            "coupling_cache",
            label="Building MPI-distributed mode coupling cache",
        ):
            coupling = build_mode_resolved_isotropic_derivative_distributed(
                exchange,
                derivative,
                phonons,
                zero_point,
                require_complete_targets=request.require_complete_targets,
                q_chunk_size=runtime.q_chunk_size,
                bond_chunk_size=runtime.bond_chunk_size,
                context=mpi,
            )
        union_mesh = tuple(
            int(np.lcm(k_value, q_value))
            for k_value, q_value in zip(
                request.kmesh,
                derivative.q_mesh_shape,
                strict=True,
            )
        )
        logger.info(
            "unique k+q mesh = "
            + " x ".join(map(str, union_mesh))
            + f" ({int(np.prod(union_mesh))} LSWT points)"
        )
        logger.info(
            "k distribution = balanced contiguous rank blocks; "
            "vertex/self-energy = rank-local q-block streaming"
        )

        with logger.phase(
            "lswt_cache",
            label="Precomputing unique k+q magnon mesh",
        ):
            magnon_cache = build_magnon_mesh_cache(
                exchange,
                configuration,
                coupling,
                k_points,
                k_mesh_shape=request.kmesh,
                kshift_grid=request.kshift,
                context=mpi,
            )

        coupling_bytes, lswt_bytes = _cache_storage(coupling, magnon_cache)
        cache_bytes_per_rank = coupling_bytes + lswt_bytes
        hostnames = mpi.allgather(gethostname())
        maximum_ranks_per_node = max(Counter(hostnames).values())
        logger.info(
            "persistent cache/rank = coupling "
            f"{_format_storage(coupling_bytes)} + LSWT "
            f"{_format_storage(lswt_bytes)} = "
            f"{_format_storage(cache_bytes_per_rank)}"
        )
        logger.info(
            "maximum replicated cache/node = "
            f"{_format_storage(cache_bytes_per_rank * maximum_ranks_per_node)} "
            f"({maximum_ranks_per_node} rank(s)/node)"
        )

        def report_progress(completed: int, total: int) -> None:
            if total < 1:
                return
            label = (
                "external k (balanced-rank estimate)" if mpi.size > 1 else "external k"
            )
            logger.progress(
                label,
                completed,
                total,
                every=max(1, total // 20),
                min_interval=1.0,
                force=completed == total,
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
                vertex_q_chunk_size=runtime.vertex_q_chunk_size,
                self_energy_q_chunk_size=runtime.self_energy_q_chunk_size,
                channel_chunk_size=runtime.channel_chunk_size,
                magnon_cache=magnon_cache,
                context=mpi,
                progress=report_progress,
            )

        output_failure: RankFailure | None = None
        if mpi.is_root:
            try:
                if distributed.global_result is None:
                    raise RuntimeError("rank zero did not receive the lifetime grid")
                metadata = _output_metadata(
                    request,
                    parallel=runtime,
                    exchange=exchange,
                    exchange_report=exchange_report,
                    derivative=derivative,
                    derivative_report=derivative_report,
                    phonons=phonons,
                    configuration=configuration,
                    coupling=coupling,
                    magnon_cache=magnon_cache,
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
    _validate_parallel(config.parallel)
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
            parallel=config.parallel,
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
            ("workers/rank", config.parallel.workers_per_rank),
            ("threads/worker", config.parallel.threads_per_worker),
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
        "Optional &magph: spin_pattern, frequency_floor_mev, asr_policy,\n"
        "output, and overwrite. Put worker/thread and q/bond/vertex/\n"
        "self-energy/channel chunk controls in &parallel.\n"
        "MPI is selected automatically under mpirun; external k points are\n"
        "distributed across ranks and q/mode contractions stay vectorized.\n"
    )


__all__ = ["MagphRunResult", "prepare_run", "run_lifetime"]
