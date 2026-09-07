"""Native MPI-aware orchestration for magnon self-energy and lifetime."""

from __future__ import annotations

import hashlib
import json
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

from .config import (
    MagphDispersionRequest,
    MagphInputError,
    MagphLifetimeRequest,
    MagphPhononRenormalizationRequest,
    RestartMode,
    build_dispersion_request,
    build_lifetime_request,
    build_phonon_renormalization_request,
)
from .coupling import build_mode_resolved_isotropic_derivative_distributed
from .derivative import (
    load_exchange_derivative_h5,
    read_exchange_derivative_q_mesh,
)
from .dispersion import (
    MagnonDispersionResult,
    MagnonKPath,
    build_wannier90_kpath,
    compute_magnon_dispersion,
)
from .epr_phonon import EPRPhononBuildReport, write_epr_phonon_cache
from .lswt import magnon_mode_chirality, uniform_fractional_mesh
from .mesh import build_magnon_mesh_cache
from .output import (
    DISPERSION_OUTPUT_SCHEMA_VERSION,
    LIFETIME_OUTPUT_SCHEMA_VERSION,
    PHONON_RENORMALIZATION_OUTPUT_SCHEMA_VERSION,
    write_dispersion_npz,
    write_dispersion_plot,
    write_lifetime_npz,
    write_phonon_renormalization_npz,
)
from .parallel import CollectiveExecutionError, RankFailure
from .phonon import load_phonon_cache, zero_point_displacements
from .pipeline import LifetimeGridResult, compute_lifetime_grid
from .phonon_renormalization import compute_phonon_renormalization_grid
from .screening import load_exchange_h5, screen_magnetic_configuration


@dataclass(frozen=True)
class MagphRunResult:
    output: Path
    mpi_size: int
    k_point_count: int
    magnetic_site_count: int
    plot_output: Path | None = None


def _source_stamp(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _anisotropy_signature(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "model": value.model,
        "energy_mev": list(value.energy_mev),
        "axis": list(value.axis),
        "spin_normalization": value.spin_normalization.value,
    }


def _request_signature(
    request: (
        MagphLifetimeRequest
        | MagphPhononRenormalizationRequest
        | MagphDispersionRequest
    ),
) -> str:
    """Hash scientific inputs and source-file identities for safe restart reuse."""

    common: dict[str, Any] = {
        "exchange_h5": _source_stamp(request.exchange_h5),
        "magnetic_order": request.magnetic_order.value,
        "spin_magnitudes": list(request.spin_magnitudes),
        "spin_pattern": None
        if request.spin_pattern is None
        else list(request.spin_pattern),
        "quantization_axis": list(request.quantization_axis),
        "single_ion_anisotropy": _anisotropy_signature(request.anisotropy),
    }
    if isinstance(request, MagphLifetimeRequest):
        calculation = (
            "phonon_renormalization"
            if isinstance(request, MagphPhononRenormalizationRequest)
            else "lifetime"
        )
        payload = {
            **common,
            "calculation": calculation,
            "derivative_h5": _source_stamp(request.derivative_h5),
            "phonon_cache": _source_stamp(request.phonon_cache),
            "phonon_epr": (
                None
                if request.phonon_epr is None
                else _source_stamp(request.phonon_epr)
            ),
            "phonon_loto": request.phonon_loto,
            "phonon_qmesh": (
                None
                if request.phonon_qmesh is None
                else list(request.phonon_qmesh)
            ),
            "phonon_imaginary_tolerance_mev": (
                request.phonon_imaginary_tolerance_mev
            ),
            "phonon_cache_compressed": request.phonon_cache_compressed,
            "kmesh": list(request.kmesh),
            "kshift": list(request.kshift),
            "temperature_k": request.temperature_k,
            "broadening_mev": request.broadening_mev,
            "frequency_floor_mev": request.frequency_floor_mev,
            "asr_policy": request.asr_policy.value,
            "exchange_reciprocity_atol_mev": (
                request.exchange_reciprocity_atol_mev
            ),
            "derivative_reciprocity_atol_mev_per_ang": (
                request.derivative_reciprocity_atol_mev_per_ang
            ),
            "metric_energy_tolerance_mev": request.metric_energy_tolerance_mev,
            "negative_tolerance_mev": request.negative_tolerance_mev,
            "require_complete_targets": request.require_complete_targets,
        }
    else:
        payload = {
            **common,
            "calculation": "dispersion",
            "kpath_file": _source_stamp(request.kpath_file),
            "points_per_segment": request.points_per_segment,
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _completed_payload(
    path: Path,
    *,
    calculation: str,
    schema_version: int,
    restart_signature: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as source:
            schema = np.asarray(source["schema_version"])
            if schema.size != 1 or int(schema.reshape(())) != schema_version:
                raise ValueError(
                    f"expected schema_version={schema_version}, got {schema.tolist()}"
                )
            metadata = json.loads(str(np.asarray(source["metadata_json"]).reshape(())))
            if not isinstance(metadata, dict):
                raise TypeError("metadata_json must decode to an object")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except Exception as exc:
        raise ValueError(
            f"invalid completed {calculation} output {path}: {exc}"
        ) from exc
    if metadata.get("calculation") != calculation:
        raise ValueError(
            f"restart output {path} has calculation={metadata.get('calculation')!r}, "
            f"expected {calculation!r}"
        )
    recorded = metadata.get("restart_signature")
    if recorded != restart_signature:
        reason = "missing" if recorded is None else "does not match the current input"
        raise ValueError(
            f"restart signature {reason} in {path}; use "
            "restart_mode='from_scratch' to replace it"
        )
    return metadata, arrays


def _restart_dispersion_result(
    request: MagphDispersionRequest,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> MagnonDispersionResult:
    required = {
        "energy_mev",
        "goldstone_mask",
        "k_points_frac",
        "segment_offsets",
        "tick_labels",
        "tick_positions_inv_ang",
        "x_coordinate_inv_ang",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError("completed dispersion is missing " + ", ".join(missing))
    path = MagnonKPath(
        source=request.kpath_file,
        k_points_frac=arrays["k_points_frac"],
        x_coordinate_inv_ang=arrays["x_coordinate_inv_ang"],
        segment_offsets=arrays["segment_offsets"],
        tick_positions_inv_ang=arrays["tick_positions_inv_ang"],
        tick_labels=tuple(str(value) for value in arrays["tick_labels"].tolist()),
        points_per_segment=request.points_per_segment,
    )
    return MagnonDispersionResult(
        path=path,
        energy_mev=arrays["energy_mev"],
        goldstone_mask=arrays["goldstone_mask"],
        mpi_size=int(metadata.get("mpi_size", 1)),
    )


def _existing_output_policy(
    request: (
        MagphLifetimeRequest
        | MagphPhononRenormalizationRequest
        | MagphDispersionRequest
    ),
    context: MPIContext,
) -> MagphRunResult | None:
    """Apply one root-decided output policy and broadcast the result or error."""

    reused: MagphRunResult | None = None
    failure: RankFailure | None = None
    if context.is_root:
        try:
            auxiliaries = (
                (request.plot_output,)
                if isinstance(request, MagphDispersionRequest)
                and request.plot_output is not None
                else ()
            )
            existing = tuple(
                path for path in (request.output, *auxiliaries) if path.exists()
            )
            if request.restart_mode is RestartMode.ERROR and existing:
                rendered = ", ".join(str(path) for path in existing)
                raise FileExistsError(f"native magph output already exists: {rendered}")
            if request.restart_mode is RestartMode.RESTART:
                if not request.output.exists():
                    if existing:
                        raise FileExistsError(
                            "restart found auxiliary output without the primary NPZ: "
                            + ", ".join(str(path) for path in existing)
                        )
                else:
                    signature = _request_signature(request)
                    if isinstance(request, MagphDispersionRequest):
                        calculation = "dispersion"
                        schema = DISPERSION_OUTPUT_SCHEMA_VERSION
                    elif isinstance(request, MagphPhononRenormalizationRequest):
                        calculation = "phonon_renormalization"
                        schema = PHONON_RENORMALIZATION_OUTPUT_SCHEMA_VERSION
                    else:
                        calculation = "lifetime"
                        schema = LIFETIME_OUTPUT_SCHEMA_VERSION
                    metadata, arrays = _completed_payload(
                        request.output,
                        calculation=calculation,
                        schema_version=schema,
                        restart_signature=signature,
                    )
                    if isinstance(request, MagphDispersionRequest):
                        result = _restart_dispersion_result(request, metadata, arrays)
                        if (
                            request.plot_output is not None
                            and not request.plot_output.exists()
                        ):
                            write_dispersion_plot(
                                request.plot_output,
                                result,
                                title=request.output.stem,
                                dpi=request.plot_dpi,
                                overwrite=False,
                            )
                        k_count = result.path.n_points
                    elif isinstance(request, MagphPhononRenormalizationRequest):
                        required = {
                            "q_points_frac",
                            "renormalized_frequency_mev",
                        }
                        missing = sorted(required.difference(arrays))
                        if missing:
                            raise ValueError(
                                "completed phonon-renormalization output is missing "
                                + ", ".join(missing)
                            )
                        q_points = np.asarray(arrays["q_points_frac"])
                        frequency = np.asarray(
                            arrays["renormalized_frequency_mev"]
                        )
                        if q_points.ndim != 2 or q_points.shape[1:] != (3,):
                            raise ValueError(
                                "completed phonon-renormalization q_points_frac "
                                "is invalid"
                            )
                        if (
                            frequency.ndim != 2
                            or frequency.shape[0] != q_points.shape[0]
                        ):
                            raise ValueError(
                                "completed phonon-renormalization frequency grid "
                                "is invalid"
                            )
                        kmesh = metadata.get("kmesh")
                        if not isinstance(kmesh, list) or len(kmesh) != 3:
                            raise ValueError(
                                "completed phonon-renormalization output lacks "
                                "kmesh provenance"
                            )
                        k_count = int(np.prod(np.asarray(kmesh, dtype=np.int64)))
                    else:
                        if "k_points_frac" not in arrays or "energy_mev" not in arrays:
                            raise ValueError(
                                "completed lifetime output is missing k_points_frac "
                                "or energy_mev"
                            )
                        k_points = np.asarray(arrays["k_points_frac"])
                        energy = np.asarray(arrays["energy_mev"])
                        if k_points.ndim != 2 or k_points.shape[1:] != (3,):
                            raise ValueError(
                                "completed lifetime k_points_frac is invalid"
                            )
                        if energy.ndim != 2 or energy.shape[0] != k_points.shape[0]:
                            raise ValueError("completed lifetime energy_mev is invalid")
                        k_count = int(k_points.shape[0])
                    magnetic_atoms = metadata.get("magnetic_atom_indices")
                    if not isinstance(magnetic_atoms, list) or not magnetic_atoms:
                        raise ValueError(
                            "completed output lacks magnetic_atom_indices provenance"
                        )
                    reused = MagphRunResult(
                        output=request.output,
                        plot_output=(
                            request.plot_output
                            if isinstance(request, MagphDispersionRequest)
                            else None
                        ),
                        mpi_size=context.size,
                        k_point_count=k_count,
                        magnetic_site_count=len(magnetic_atoms),
                    )
        except Exception as exc:  # noqa: BLE001 - release all ranks together
            failure = _rank_failure(context.rank, "magph_restart", exc)
    reused, failure = context.bcast((reused, failure), root=0)
    if failure is not None:
        raise CollectiveExecutionError((failure,))
    return reused


def _validate_parallel(parallel: ParallelConfig, *, calculation: str) -> None:
    if parallel.workers_per_rank != 1:
        work_axis = (
            "external phonon q"
            if calculation == "phonon_renormalization"
            else "external magnon k"
        )
        raise MagphInputError(
            f"native {calculation} currently requires &parallel workers_per_rank=1; "
            f"use MPI ranks over {work_axis} and threads_per_worker within each rank"
        )
    if parallel.precache_workers is not None:
        raise MagphInputError(
            f"native {calculation} does not use &parallel precache_workers"
        )
    if parallel.numba_threads is not None:
        raise MagphInputError(
            f"native {calculation} does not use &parallel numba_threads; "
            "use threads_per_worker or blas_threads for vectorized contractions"
        )
    if calculation == "dispersion":
        chunks = {
            "q_chunk_size": parallel.q_chunk_size,
            "bond_chunk_size": parallel.bond_chunk_size,
            "vertex_q_chunk_size": parallel.vertex_q_chunk_size,
            "self_energy_q_chunk_size": parallel.self_energy_q_chunk_size,
            "channel_chunk_size": parallel.channel_chunk_size,
        }
        unused = [name for name, value in chunks.items() if value is not None]
        if unused:
            raise MagphInputError(
                "native dispersion does not use &parallel " + ", ".join(unused)
            )


def _rank_failure(rank: int, stage: str, exc: BaseException) -> RankFailure:
    return RankFailure(
        rank=int(rank),
        stage=stage,
        exception_type=type(exc).__name__,
        message=str(exc).strip() or repr(exc),
    )


def _ensure_phonon_cache(
    request: MagphLifetimeRequest,
    context: MPIContext,
    parallel: ParallelConfig,
    logger: RunLogger,
) -> EPRPhononBuildReport | None:
    """Create a missing EPR-derived cache once on rank zero."""

    report: EPRPhononBuildReport | None = None
    failure: RankFailure | None = None
    if context.is_root:
        try:
            if request.phonon_cache.is_file():
                logger.info(f"phonon cache = reusing {request.phonon_cache}")
            else:
                if request.phonon_cache.exists():
                    raise ValueError(
                        "phonon_cache exists but is not a regular file: "
                        f"{request.phonon_cache}"
                    )
                if request.phonon_epr is None:
                    raise FileNotFoundError(
                        f"phonon cache not found: {request.phonon_cache}; "
                        "set phonon_epr to build it automatically"
                    )
                q_mesh = (
                    read_exchange_derivative_q_mesh(request.derivative_h5)
                    if request.phonon_qmesh is None
                    else request.phonon_qmesh
                )
                logger.info(
                    "phonon cache = absent; building from EPR on rank 0"
                )
                logger.info(
                    "phonon q mesh = " + " x ".join(map(str, q_mesh))
                )
                report = write_epr_phonon_cache(
                    request.phonon_epr,
                    request.phonon_cache,
                    q_mesh_shape=q_mesh,
                    q_chunk_size=parallel.q_chunk_size,
                    loto_mode=request.phonon_loto,
                    imaginary_tolerance_mev=(
                        request.phonon_imaginary_tolerance_mev
                    ),
                    compressed=request.phonon_cache_compressed,
                )
        except Exception as exc:  # noqa: BLE001 - release every rank together
            failure = _rank_failure(context.rank, "phonon_cache_build", exc)
    report, failure = context.bcast((report, failure), root=0)
    if failure is not None:
        raise CollectiveExecutionError((failure,))
    if report is not None:
        logger.info(
            "phonon cache = built "
            f"{report.q_point_count} q points, {report.mode_count} modes in "
            f"{report.elapsed_seconds:.2f} s"
        )
        logger.info(
            "phonon cleanup = rounded "
            f"{report.rounded_frequency_count} modes within "
            f"{report.imaginary_tolerance_mev:.3g} meV"
        )
    return report


def _load_native_problem(request: MagphLifetimeRequest) -> tuple[Any, ...]:
    exchange, exchange_report = load_exchange_h5(
        request.exchange_h5,
        reciprocity_atol_mev=request.exchange_reciprocity_atol_mev,
    )
    anisotropy = (
        None
        if request.anisotropy is None
        else request.anisotropy.build(exchange.n_magnetic_sites)
    )
    derivative, derivative_report = load_exchange_derivative_h5(
        request.derivative_h5,
        exchange,
        reciprocity_atol_mev_per_ang=(
            request.derivative_reciprocity_atol_mev_per_ang
        ),
        asr_policy=request.asr_policy,
    )
    phonons = load_phonon_cache(request.phonon_cache)
    if phonons.q_mesh_shape is None:
        raise ValueError(
            "native lifetime requires phonon q_mesh_shape for dense-q "
            "interpolation and exact k+q indexing"
        )
    if (
        request.phonon_qmesh is not None
        and phonons.q_mesh_shape != request.phonon_qmesh
    ):
        raise ValueError(
            f"phonon cache q mesh {phonons.q_mesh_shape} != requested "
            f"phonon_qmesh {request.phonon_qmesh}"
        )
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
        anisotropy=anisotropy,
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
        anisotropy,
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


def _external_union_indices(magnon_cache: Any) -> np.ndarray:
    """Map every external k point to the broadcast union cache vectorially."""

    union = np.asarray(magnon_cache.union_mesh_shape, dtype=np.int64)
    external_mesh = np.asarray(magnon_cache.k_mesh_shape, dtype=np.int64)
    coordinates = magnon_cache.external_k_grid_indices * (union // external_mesh)
    return np.asarray(
        np.ravel_multi_index(
            (coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]),
            magnon_cache.union_mesh_shape,
        ),
        dtype=np.int64,
    )


def _canonicalize_lifetime_mode_order(
    result: LifetimeGridResult,
    magnon_cache: Any,
    configuration: Any,
) -> tuple[LifetimeGridResult, np.ndarray | None]:
    """Order bipartite-AFM observables as chirality ``+1, -1``."""

    if configuration.order.value != "collinear_afm":
        return result, None
    physical_count = int(magnon_cache.physical_mode_count)
    external_union = _external_union_indices(magnon_cache)
    cache_energy = magnon_cache.signed_energies_mev[
        external_union, :physical_count
    ]
    energy_error = float(
        np.max(np.abs(cache_energy - result.energy_mev), initial=0.0)
    )
    energy_scale = max(float(np.max(np.abs(cache_energy), initial=0.0)), 1.0)
    if energy_error > 256.0 * np.finfo(np.float64).eps * energy_scale:
        raise ValueError(
            "lifetime energy grid is inconsistent with the external LSWT cache; "
            f"maximum error={energy_error:.6g} meV"
        )
    transformation = magnon_cache.transformation[
        external_union, :, :physical_count
    ]
    raw_chirality = magnon_mode_chirality(
        transformation,
        configuration.spin_pattern,
        physical_mode_count=physical_count,
    )
    permutation = np.argsort(-raw_chirality, axis=1, kind="stable")

    def reorder(value: Any) -> np.ndarray:
        return np.take_along_axis(np.asarray(value), permutation, axis=1)

    canonical = LifetimeGridResult(
        k_points_frac=result.k_points_frac,
        energy_mev=reorder(result.energy_mev),
        self_energy_onshell_mev=reorder(result.self_energy_onshell_mev),
        gamma_hwhm_mev=reorder(result.gamma_hwhm_mev),
        fwhm_mev=reorder(result.fwhm_mev),
        scattering_rate_ps_inv=reorder(result.scattering_rate_ps_inv),
        lifetime_ps=reorder(result.lifetime_ps),
        valid_damping=reorder(result.valid_damping),
        temperature_k=result.temperature_k,
        broadening_mev=result.broadening_mev,
        negative_tolerance_mev=result.negative_tolerance_mev,
    )
    chirality = reorder(raw_chirality)
    chirality.setflags(write=False)
    return canonical, chirality


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
    anisotropy: Any,
    coupling: Any,
    magnon_cache: Any,
    mpi_size: int,
) -> dict[str, Any]:
    coupling_bytes, lswt_bytes = _cache_storage(coupling, magnon_cache)
    return {
        "calculation": "lifetime",
        "exchange_h5": str(request.exchange_h5),
        "lattice_ang": (
            None
            if exchange.lattice_ang is None
            else exchange.lattice_ang.tolist()
        ),
        "tau_frac": (
            None if exchange.tau_frac is None else exchange.tau_frac.tolist()
        ),
        "atom_labels": list(exchange.atom_labels),
        "derivative_h5": str(request.derivative_h5),
        "phonon_cache": str(request.phonon_cache),
        "phonon_epr": (
            None if request.phonon_epr is None else str(request.phonon_epr)
        ),
        "phonon_loto": request.phonon_loto,
        "phonon_qmesh_requested": (
            None
            if request.phonon_qmesh is None
            else list(request.phonon_qmesh)
        ),
        "phonon_imaginary_tolerance_mev": (
            request.phonon_imaginary_tolerance_mev
        ),
        "phonon_cache_compressed": request.phonon_cache_compressed,
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
        "exchange_reciprocity_atol_mev": request.exchange_reciprocity_atol_mev,
        "derivative_reciprocity_atol_mev_per_ang": (
            request.derivative_reciprocity_atol_mev_per_ang
        ),
        "derivative_asr_projected": bool(derivative_report.projected_asr),
        "derivative_static_bond_count": int(derivative_report.static_bond_count),
        "derivative_source_bond_count": int(derivative_report.source_bond_count),
        "derivative_zero_filled_static_bond_count": int(
            derivative_report.zero_filled_static_bond_count
        ),
        "derivative_bond_coverage_complete": bool(
            derivative_report.bond_coverage_complete
        ),
        "derivative_covariant_symmetry_policy": (
            derivative_report.covariant_symmetry_policy
        ),
        "derivative_covariant_symmetry_applied": bool(
            derivative_report.covariant_symmetry_applied
        ),
        "derivative_max_covariance_residual_mev_per_ang": (
            None
            if derivative_report.max_covariance_residual_mev_per_ang is None
            else float(derivative_report.max_covariance_residual_mev_per_ang)
        ),
        "derivative_covariant_symmetry_operation_count": int(
            derivative_report.covariant_symmetry_operation_count
        ),
        "derivative_covariant_symmetry_spacegroup": (
            derivative_report.covariant_symmetry_spacegroup
        ),
        "derivative_max_asr_residual_mev_per_ang": float(
            derivative_report.max_asr_residual_after_mev_per_ang
        ),
        "derivative_fourier_phase_convention": derivative.fourier_phase_convention,
        "derivative_directed_bond_mate": derivative.directed_bond_mate,
        "derivative_source_qmesh": list(derivative.q_mesh_shape),
        "phonon_evaluation_qmesh": list(phonons.q_mesh_shape),
        "derivative_q_interpolation": bool(
            derivative.q_mesh_shape != phonons.q_mesh_shape
        ),
        "phonon_schema_version": int(phonons.schema_version),
        "phonon_mass_unit": phonons.mass_unit.value,
        "phonon_vector_convention": phonons.vector_convention,
        "phonon_fourier_phase_convention": phonons.fourier_phase_convention,
        "frequency_floor_mev": request.frequency_floor_mev,
        "magnetic_order": request.magnetic_order.value,
        "magnetic_atom_indices": exchange.magnetic_atom_indices.tolist(),
        "spin_magnitudes": configuration.spin_magnitudes.tolist(),
        "spin_pattern": configuration.spin_pattern.tolist(),
        "quantization_axis": configuration.quantization_axis.tolist(),
        "magnon_mode_order": (
            "chirality_descending"
            if configuration.order.value == "collinear_afm"
            else "energy_ascending"
        ),
        "magnon_chirality_definition": (
            "-sum_i eta_i (abs(u_i)^2-abs(v_i)^2) / "
            "sum_i abs(abs(u_i)^2-abs(v_i)^2)"
        ),
        "magnon_chirality_axis": "ordered_spin_axis",
        "single_ion_anisotropy": _anisotropy_metadata(anisotropy),
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
            "derivative_q_interpolation": "realspace_fourier_from_dJ_Rp",
            "lswt": "uniform_k_plus_q_union_cache",
            "vertex_self_energy": "q_block_streaming",
            "full_vertex_materialized": False,
            "mpi_distribution": "external_k",
        },
        "union_kq_mesh": [
            int(np.lcm(k_value, q_value))
            for k_value, q_value in zip(
                request.kmesh,
                phonons.q_mesh_shape,
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
    _validate_parallel(runtime, calculation="lifetime")
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
    with logger.phase("phonon_cache", label="Preparing phonon cache"):
        _ensure_phonon_cache(request, mpi, runtime, logger)
    reused = _existing_output_policy(request, mpi)
    if reused is not None:
        logger.info(f"restart      = reused completed {request.output}")
        return reused
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
                anisotropy,
                k_points,
            ) = _load_collectively(request, mpi)
        logger.info(f"magnetic order = {configuration.order.value}")
        logger.info(f"magnetic sites = {configuration.n_magnetic_sites}")
        logger.info(f"k-point count  = {k_points.shape[0]}")
        logger.info(f"phonon q count = {phonons.nq}")
        logger.info(
            "dJ source q mesh = "
            + " x ".join(map(str, derivative.q_mesh_shape))
        )
        logger.info(
            "phonon evaluation q mesh = "
            + " x ".join(map(str, phonons.q_mesh_shape))
        )
        logger.info(
            "dJ q interpolation = real-space Fourier "
            + (
                "(active)"
                if derivative.q_mesh_shape != phonons.q_mesh_shape
                else "(source and evaluation meshes coincide)"
            )
        )
        logger.info(
            "dJ bond coverage = "
            f"{derivative_report.source_bond_count}/"
            f"{derivative_report.static_bond_count} explicit; "
            f"{derivative_report.zero_filled_static_bond_count} zero-filled"
        )
        logger.info(
            "dJ covariance   = "
            f"{derivative_report.covariant_symmetry_policy}; "
            f"applied={derivative_report.covariant_symmetry_applied}; "
            f"spacegroup={derivative_report.covariant_symmetry_spacegroup}"
        )

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
                phonons.q_mesh_shape,
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
                anisotropy=anisotropy,
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
                anisotropy=anisotropy,
                context=mpi,
                progress=report_progress,
            )

        output_failure: RankFailure | None = None
        if mpi.is_root:
            try:
                if distributed.global_result is None:
                    raise RuntimeError("rank zero did not receive the lifetime grid")
                canonical_result, magnon_chirality = (
                    _canonicalize_lifetime_mode_order(
                        distributed.global_result,
                        magnon_cache,
                        configuration,
                    )
                )
                metadata = _output_metadata(
                    request,
                    parallel=runtime,
                    exchange=exchange,
                    exchange_report=exchange_report,
                    derivative=derivative,
                    derivative_report=derivative_report,
                    phonons=phonons,
                    configuration=configuration,
                    anisotropy=anisotropy,
                    coupling=coupling,
                    magnon_cache=magnon_cache,
                    mpi_size=mpi.size,
                )
                metadata["restart_signature"] = _request_signature(request)
                metadata["restart_mode"] = request.restart_mode.value
                with logger.phase("output", label="Writing lifetime output"):
                    write_lifetime_npz(
                        request.output,
                        canonical_result,
                        metadata=metadata,
                        magnon_chirality=magnon_chirality,
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


def run_phonon_renormalization(
    request: MagphPhononRenormalizationRequest,
    *,
    context: MPIContext | None = None,
    parallel: ParallelConfig | None = None,
    program: str = "slw_magph.x",
    verbosity: str = "normal",
) -> MagphRunResult:
    """Run the exchange-striction magnon bubble for every phonon q/mode."""

    mpi = MPIContext.discover() if context is None else context
    runtime = ParallelConfig() if parallel is None else parallel
    _validate_parallel(runtime, calculation="phonon_renormalization")
    if mpi.size > 1:
        request = mpi.bcast(request if mpi.is_root else None, root=0)
        if request is None:  # pragma: no cover - communicator contract guard
            raise RuntimeError(
                "rank zero did not broadcast the phonon-renormalization request"
            )
    timers = TimerBook()
    logger = RunLogger(
        sys.stdout,
        rank=mpi.rank,
        verbosity=verbosity,
        timers=timers,
    )
    with logger.phase("phonon_cache", label="Preparing phonon cache"):
        _ensure_phonon_cache(request, mpi, runtime, logger)
    reused = _existing_output_policy(request, mpi)
    if reused is not None:
        logger.info(f"restart      = reused completed {request.output}")
        return reused

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
                anisotropy,
                k_points,
            ) = _load_collectively(request, mpi)
        logger.info("external mode  = phonon")
        logger.info("self-energy    = one-loop exchange-striction magnon bubble")
        logger.info("static J''     = not included")
        logger.info(f"magnetic order = {configuration.order.value}")
        logger.info(f"magnon k count = {k_points.shape[0]}")
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
                phonons.q_mesh_shape,
                strict=True,
            )
        )
        logger.info(
            "unique k+q mesh = "
            + " x ".join(map(str, union_mesh))
            + f" ({int(np.prod(union_mesh))} LSWT points)"
        )
        with logger.phase("lswt_cache", label="Precomputing k+q magnons"):
            magnon_cache = build_magnon_mesh_cache(
                exchange,
                configuration,
                coupling,
                k_points,
                k_mesh_shape=request.kmesh,
                kshift_grid=request.kshift,
                anisotropy=anisotropy,
                context=mpi,
            )

        q_blocks = [
            value
            for value in (
                runtime.vertex_q_chunk_size,
                runtime.self_energy_q_chunk_size,
            )
            if value is not None
        ]
        renormalization_q_chunk = 1 if not q_blocks else min(q_blocks)
        logger.info(
            "q distribution = balanced contiguous MPI blocks; "
            f"rank-local materialization chunk={renormalization_q_chunk}"
        )

        def report_progress(completed: int, total: int) -> None:
            if total < 1:
                return
            label = (
                "phonon q (balanced-rank estimate)"
                if mpi.size > 1
                else "phonon q"
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
            "phonon_self_energy",
            label="Computing on-shell phonon self-energy",
        ):
            distributed = compute_phonon_renormalization_grid(
                exchange,
                configuration,
                coupling,
                magnon_cache,
                phonons.frequencies_mev,
                temperature_k=request.temperature_k,
                broadening_mev=request.broadening_mev,
                metric_energy_tolerance_mev=(
                    request.metric_energy_tolerance_mev
                ),
                negative_tolerance_mev=request.negative_tolerance_mev,
                q_chunk_size=renormalization_q_chunk,
                channel_chunk_size=runtime.channel_chunk_size,
                context=mpi,
                progress=report_progress,
            )

        output_failure: RankFailure | None = None
        if mpi.is_root:
            try:
                if distributed.global_result is None:
                    raise RuntimeError(
                        "rank zero did not receive the phonon-renormalization grid"
                    )
                metadata = _output_metadata(
                    request,
                    parallel=runtime,
                    exchange=exchange,
                    exchange_report=exchange_report,
                    derivative=derivative,
                    derivative_report=derivative_report,
                    phonons=phonons,
                    configuration=configuration,
                    anisotropy=anisotropy,
                    coupling=coupling,
                    magnon_cache=magnon_cache,
                    mpi_size=mpi.size,
                )
                metadata["calculation"] = "phonon_renormalization"
                metadata["external_quasiparticle"] = "phonon"
                metadata["static_exchange_second_derivative_included"] = False
                metadata.pop("q_weight_policy", None)
                metadata["k_weight_policy"] = "uniform_normalized"
                metadata["phonon_mode_self_energy"] = "diagonal"
                metadata["nambu_prefactor"] = (
                    0.5
                    if configuration.order.value == "collinear_afm"
                    else 1.0
                )
                metadata["algorithm"] = {
                    **metadata["algorithm"],
                    "self_energy": "one_loop_exchange_striction_magnon_bubble",
                    "mpi_distribution": "phonon_q",
                    "dyson": "onshell_frequency_squared",
                }
                metadata["restart_signature"] = _request_signature(request)
                metadata["restart_mode"] = request.restart_mode.value
                with logger.phase(
                    "output",
                    label="Writing phonon-renormalization output",
                ):
                    write_phonon_renormalization_npz(
                        request.output,
                        distributed.global_result,
                        metadata=metadata,
                        overwrite=request.overwrite,
                    )
            except Exception as exc:  # noqa: BLE001 - release non-root ranks
                output_failure = _rank_failure(
                    mpi.rank,
                    "phonon_renormalization_output",
                    exc,
                )
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


def _load_dispersion_problem(request: MagphDispersionRequest) -> tuple[Any, ...]:
    exchange, exchange_report = load_exchange_h5(request.exchange_h5)
    if exchange.lattice_ang is None:
        raise MagphInputError(
            "native dispersion requires lattice_ang in the exchange HDF5 to "
            "construct the reciprocal-path distance"
        )
    spin_magnitudes: float | tuple[float, ...] = request.spin_magnitudes
    if len(request.spin_magnitudes) == 1:
        spin_magnitudes = request.spin_magnitudes[0]
    anisotropy = (
        None
        if request.anisotropy is None
        else request.anisotropy.build(exchange.n_magnetic_sites)
    )
    configuration = screen_magnetic_configuration(
        exchange,
        order=request.magnetic_order,
        spin_pattern=request.spin_pattern,
        spin_magnitudes=spin_magnitudes,
        quantization_axis=request.quantization_axis,
        anisotropy=anisotropy,
    )
    path = build_wannier90_kpath(
        request.kpath_file,
        lattice_ang=exchange.lattice_ang,
        points_per_segment=request.points_per_segment,
    )
    return exchange, exchange_report, configuration, anisotropy, path


def _load_dispersion_collectively(
    request: MagphDispersionRequest,
    context: MPIContext,
) -> tuple[Any, ...]:
    loaded: tuple[Any, ...] | None = None
    failure: RankFailure | None = None
    try:
        loaded = _load_dispersion_problem(request)
    except Exception as exc:  # noqa: BLE001 - exchange before next collective
        failure = _rank_failure(context.rank, "magph_dispersion_input", exc)
    failures = tuple(item for item in context.allgather(failure) if item is not None)
    if failures:
        raise CollectiveExecutionError(failures)
    if loaded is None:  # pragma: no cover - guarded by collective failure
        raise RuntimeError("native magph dispersion produced no input or failure")
    return loaded


def _anisotropy_metadata(anisotropy: Any) -> dict[str, Any] | None:
    if anisotropy is None:
        return None
    return {
        "model": anisotropy.model,
        "energy_mev": anisotropy.energy_mev.tolist(),
        "axis": anisotropy.axis.tolist(),
        "spin_normalization": anisotropy.spin_normalization.value,
        "hamiltonian_sign": anisotropy.hamiltonian_sign,
    }


def run_dispersion(
    request: MagphDispersionRequest,
    *,
    context: MPIContext | None = None,
    parallel: ParallelConfig | None = None,
    program: str = "slw_magph.x",
    verbosity: str = "normal",
) -> MagphRunResult:
    """Run an MPI-distributed native magnon-band calculation and plot."""

    mpi = MPIContext.discover() if context is None else context
    runtime = ParallelConfig() if parallel is None else parallel
    _validate_parallel(runtime, calculation="dispersion")
    if mpi.size > 1:
        request = mpi.bcast(request if mpi.is_root else None, root=0)
        if request is None:  # pragma: no cover - communicator contract guard
            raise RuntimeError("rank zero did not broadcast the dispersion request")
    timers = TimerBook()
    logger = RunLogger(
        sys.stdout,
        rank=mpi.rank,
        verbosity=verbosity,
        timers=timers,
    )
    reused = _existing_output_policy(request, mpi)
    if reused is not None:
        logger.info(f"restart      = reused completed {request.output}")
        return reused
    with timers.phase("total"):
        with logger.phase("input_screening", label="Screening magnon-band inputs"):
            (
                exchange,
                exchange_report,
                configuration,
                anisotropy,
                path,
            ) = _load_dispersion_collectively(request, mpi)
        logger.info(f"magnetic order = {configuration.order.value}")
        logger.info(f"magnetic sites = {configuration.n_magnetic_sites}")
        logger.info(f"path segments  = {path.n_segments}")
        logger.info(f"path k points  = {path.n_points}")
        logger.info(
            "single-ion anisotropy = "
            + ("none" if anisotropy is None else anisotropy.spin_normalization.value)
        )
        with logger.phase(
            "dispersion",
            label="Computing MPI-distributed magnon dispersion",
        ):
            distributed = compute_magnon_dispersion(
                exchange,
                configuration,
                path,
                anisotropy=anisotropy,
                context=mpi,
            )

        output_failure: RankFailure | None = None
        if mpi.is_root:
            try:
                if distributed.global_result is None:
                    raise RuntimeError(
                        "rank zero did not receive the magnon dispersion"
                    )
                metadata = {
                    "calculation": "dispersion",
                    "exchange_h5": str(request.exchange_h5),
                    "exchange_dataset": exchange_report.source_dataset,
                    "exchange_representation": exchange_report.representation.value,
                    "exchange_convention_origin": exchange_report.convention_origin,
                    "exchange_kernel_family": exchange.convention.kernel_family,
                    "magnetic_order": configuration.order.value,
                    "magnetic_atom_indices": exchange.magnetic_atom_indices.tolist(),
                    "spin_magnitudes": configuration.spin_magnitudes.tolist(),
                    "spin_pattern": configuration.spin_pattern.tolist(),
                    "quantization_axis": configuration.quantization_axis.tolist(),
                    "single_ion_anisotropy": _anisotropy_metadata(anisotropy),
                    "kpath_file": str(request.kpath_file),
                    "points_per_segment": request.points_per_segment,
                    "restart_mode": request.restart_mode.value,
                    "restart_signature": _request_signature(request),
                    "fourier_phase_convention": "exp(+i2pi_k_dot_R)",
                    "exact_goldstone_energy_only": True,
                    "goldstone_point_count": int(
                        np.count_nonzero(distributed.global_result.goldstone_mask)
                    ),
                    "mpi_size": mpi.size,
                    "parallel": {
                        "workers_per_rank": runtime.workers_per_rank,
                        "threads_per_worker": runtime.threads_per_worker,
                        "blas_threads": runtime.effective_blas_threads,
                        "distribution": "kpath_points",
                    },
                }
                with logger.phase("output", label="Writing magnon dispersion"):
                    write_dispersion_npz(
                        request.output,
                        distributed.global_result,
                        metadata=metadata,
                        overwrite=request.overwrite,
                    )
                    if request.plot_output is not None:
                        write_dispersion_plot(
                            request.plot_output,
                            distributed.global_result,
                            title=request.output.stem,
                            dpi=request.plot_dpi,
                            overwrite=request.overwrite,
                        )
            except Exception as exc:  # noqa: BLE001 - release non-root ranks
                output_failure = _rank_failure(mpi.rank, "magph_dispersion_output", exc)
        output_failure = mpi.bcast(output_failure, root=0)
        if output_failure is not None:
            raise CollectiveExecutionError((output_failure,))
        logger.info(f"written       = {request.output}")
        if request.plot_output is not None:
            logger.info(f"plot          = {request.plot_output}")

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
        plot_output=request.plot_output,
        mpi_size=mpi.size,
        k_point_count=path.n_points,
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
    """Prepare a native magph calculation for the common CLI runner."""

    if calculation not in {
        "dispersion",
        "lifetime",
        "phonon_renormalization",
    } or requested_name != calculation:
        raise MagphInputError(
            f"native magph engine does not implement {requested_name!r}"
        )
    _validate_parallel(config.parallel, calculation=calculation)
    all_ranks, warning = _parallel_policy(config, context)
    execution_context = context if all_ranks else MPIContext()
    if calculation in {"lifetime", "phonon_renormalization"}:
        if calculation == "phonon_renormalization":
            dynamic_request = build_phonon_renormalization_request(
                parameters,
                prefix=config.control.prefix,
                savedir=config.control.savedir,
            )

            def run_dynamic_plan() -> int:
                run_phonon_renormalization(
                    dynamic_request,
                    context=execution_context,
                    parallel=config.parallel,
                    program=program,
                    verbosity=config.control.verbosity,
                )
                return 0

            return NativeExecutionPlan(
                backend_label=(
                    "slw.magph.engine (native phonon renormalization)"
                ),
                run=run_dynamic_plan,
                all_ranks=all_ranks,
                warning=warning,
                summary=(
                    ("magnetic order", dynamic_request.magnetic_order.value),
                    ("magnon k mesh", " x ".join(map(str, dynamic_request.kmesh))),
                    ("k-grid shift", ", ".join(map(str, dynamic_request.kshift))),
                    (
                        "phonon q mesh",
                        "cache / dJ mesh default"
                        if dynamic_request.phonon_qmesh is None
                        else " x ".join(map(str, dynamic_request.phonon_qmesh)),
                    ),
                    ("temperature", f"{dynamic_request.temperature_k} K"),
                    ("output", dynamic_request.output),
                ),
            )

        lifetime_request = build_lifetime_request(
            parameters,
            prefix=config.control.prefix,
            savedir=config.control.savedir,
        )

        def run_lifetime_plan() -> int:
            run_lifetime(
                lifetime_request,
                context=execution_context,
                parallel=config.parallel,
                program=program,
                verbosity=config.control.verbosity,
            )
            return 0

        return NativeExecutionPlan(
            backend_label="slw.magph.engine (native lifetime)",
            run=run_lifetime_plan,
            all_ranks=all_ranks,
            warning=warning,
            summary=(
                ("magnetic order", lifetime_request.magnetic_order.value),
                ("k-point mesh", " x ".join(map(str, lifetime_request.kmesh))),
                ("k-grid shift", ", ".join(map(str, lifetime_request.kshift))),
                (
                    "phonon q mesh",
                    "cache / dJ mesh default"
                    if lifetime_request.phonon_qmesh is None
                    else " x ".join(map(str, lifetime_request.phonon_qmesh)),
                ),
                ("workers/rank", config.parallel.workers_per_rank),
                ("threads/worker", config.parallel.threads_per_worker),
                ("output", lifetime_request.output),
            ),
        )

    dispersion_request = build_dispersion_request(
        parameters,
        prefix=config.control.prefix,
        savedir=config.control.savedir,
    )

    def run_dispersion_plan() -> int:
        run_dispersion(
            dispersion_request,
            context=execution_context,
            parallel=config.parallel,
            program=program,
            verbosity=config.control.verbosity,
        )
        return 0

    return NativeExecutionPlan(
        backend_label="slw.magph.engine (native dispersion)",
        run=run_dispersion_plan,
        all_ranks=all_ranks,
        warning=warning,
        summary=(
            ("magnetic order", dispersion_request.magnetic_order.value),
            ("k-path", dispersion_request.kpath_file),
            ("points/segment", dispersion_request.points_per_segment),
            ("workers/rank", config.parallel.workers_per_rank),
            ("threads/worker", config.parallel.threads_per_worker),
            ("output", dispersion_request.output),
        ),
    )


def format_help(
    *,
    calculation: str,
    requested_name: str,
    source: str,
    parameters: dict[str, Any],
) -> str:
    del requested_name, source, parameters
    if calculation == "dispersion":
        return (
            "slw_magph.x calculation='dispersion' (native)\n\n"
            "Required &magph keys:\n"
            "  exchange_h5       = canonical static J HDF5\n"
            "  magnetic_order    = 'fm' or 'collinear_afm'\n"
            "  spin_magnitudes   = scalar or one value per magnetic site\n"
            "  quantization_axis = three Cartesian components\n"
            "  kpath_file        = Wannier90 file/snippet with kpoint_path\n\n"
            "Optional &magph: spin_pattern, points_per_segment, output, plot,\n"
            "plot_output, plot_dpi, restart_mode, and the complete uniaxial SIA\n"
            "set anisotropy_model/mev/axis/normalization. MPI distributes path\n"
            "points.\n"
        )
    if calculation == "phonon_renormalization":
        return (
            "slw_magph.x calculation='phonon_renormalization' (native)\n\n"
            "Required &magph keys:\n"
            "  exchange_h5       = canonical static J HDF5\n"
            "  derivative_h5     = scalar dJ/du HDF5\n"
            "  phonon_cache/epr  = schema-v3 NPZ or EPR IFC source\n"
            "  magnetic_order    = 'fm' or 'collinear_afm'\n"
            "  spin_magnitudes   = scalar or one value per magnetic site\n"
            "  quantization_axis = three Cartesian components\n"
            "  kmesh, kshift     = magnon integration mesh and grid shift\n"
            "  temperature_k     = non-negative temperature\n"
            "  broadening_mev    = positive retarded broadening\n\n"
            "This computes the on-shell one-loop phonon self-energy from the\n"
            "linear exchange-striction vertex dJ/du. It does not include the\n"
            "static mean-field J'' correction used by Lee and Rabe. MPI\n"
            "distributes external phonon q points. Optional controls match the\n"
            "native lifetime input, including phonon_qmesh, frequency_floor_mev,\n"
            "ASR/tolerance settings, anisotropy, output, and restart_mode.\n"
        )
    return (
        "slw_magph.x calculation='lifetime' (native)\n\n"
        "Required &magph keys:\n"
        "  exchange_h5       = canonical static J HDF5\n"
        "  derivative_h5     = scalar dJ/du HDF5\n"
        "  phonon_cache/epr  = existing schema-v3 NPZ or EPR IFC source\n"
        "  magnetic_order    = 'fm' or 'collinear_afm'\n"
        "  spin_magnitudes   = scalar or one value per magnetic site\n"
        "  quantization_axis = three Cartesian components\n"
        "  kmesh             = three positive integers\n"
        "  kshift            = explicit three-component grid-unit shift\n"
        "  temperature_k     = non-negative temperature\n"
        "  broadening_mev    = positive retarded broadening\n\n"
        "Optional &magph: phonon_cache, phonon_epr, phonon_qmesh, phonon_loto,\n"
        "phonon_imaginary_tolerance_mev, phonon_cache_compressed,\n"
        "spin_pattern, frequency_floor_mev, asr_policy,\n"
        "the complete anisotropy_model/mev/axis/normalization set, output, and\n"
        "restart_mode. Put worker/thread and q/bond/vertex/\n"
        "self-energy/channel chunk controls in &parallel.\n"
        "MPI is selected automatically under mpirun; external k points are\n"
        "distributed across ranks and q/mode contractions stay vectorized.\n"
    )


__all__ = [
    "MagphRunResult",
    "prepare_run",
    "run_dispersion",
    "run_lifetime",
    "run_phonon_renormalization",
]
