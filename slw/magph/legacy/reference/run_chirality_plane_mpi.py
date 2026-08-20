"""MPI driver for a fixed-kz magnon chirality-selectivity plane.

The expensive q integration remains threaded inside
``analyze_rotational_coupling``.  MPI distributes restartable blocks of
independent external magnon k points across ranks.  Each block is written as
an atomic, no-clobber shard; rank zero validates and merges the complete plane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_rotational_coupling import (
    RotationalCouplingResult,
    analyze_rotational_coupling,
    coupling_kwargs_from_namespace,
)
from .analyze_rotational_coupling import (
    build_parser as build_coupling_parser,
)
from ..analyze_rotational_selectivity import (
    RotationalAnalysisResult,
    write_rotational_analysis,
)

PLANE_SCHEMA_VERSION = 2

# These arrays carry one leading entry per external magnon k point.  Keeping
# this contract explicit prevents a future schema field from being merged on
# the wrong axis merely because one of its dimensions happens to equal nk.
K_DEPENDENT_KEYS = (
    "k_points_frac",
    "k_points_cart_inv_ang",
    "magnon_energies_mev",
    "gamma_atom_component_matrix_mev",
    "gamma_helicity_grouped_matrix_mev",
    "vertex_atom_component_matrix_mev2",
    "vertex_helicity_grouped_matrix_mev2",
    "vertex_physical_chirality_transition_mev2",
    "vertex_forbidden_chirality_flip_mev2",
    "vertex_chirality_flip_fraction",
    "gamma_process_mev",
    "gamma_hwhm_mev",
    "fwhm_mev",
    "lifetime_ps",
    "scattering_rate_ps_inv",
    "valid_damping",
    "paraunitary_residual_max_abs",
    "external_paraunitary_residual_max_abs",
    "internal_paraunitary_residual_max_abs",
    "internal_paraunitary_residual_max_abs_before_goldstone_omission",
    "external_metric_energy_minimum_mev",
    "internal_metric_energy_minimum_mev",
    "internal_metric_energy_minimum_mev_before_goldstone_omission",
    "vertex_selectivity_bins_mev2",
    "vertex_transition_selectivity_bins_mev2",
    "phase_space_selectivity_gamma_mev",
    "physical_lte_gamma_histogram_mev",
    "selectivity_mode_counts",
    "vertex_selectivity_contrast",
    "phase_space_selectivity_signed_contrast",
)

_BLOCK_SUMMARY_KEYS = (
    "interpretation_status",
    "diagnostic_reasons",
    "reconstruction",
    "bdg_validation",
    "max_paraunitary_residual",
    "runtime_seconds",
    "n_omitted",
    "n_qpoints_used",
    "normalization_q_count",
)

_COMMON_SUMMARY_EXCLUDED = {
    "n_kpoints",
    "physical_results",
    "vertex_chirality",
    "component_accounting",
    "goldstone_node_handling",
    "reconstruction",
    "bdg_validation",
    "max_paraunitary_residual",
    "runtime_seconds",
    "saved_vertices_memory",
    "saved_vertices_estimated_payload_bytes",
    "saved_vertices_estimated_peak_gb",
    "output_npz",
    "output_json",
}


def _positive_integer(value: int, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def build_kz_plane_grid(
    mesh: Sequence[int],
    kz_frac: float,
    shift_grid: Sequence[float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return C-order ``(kx, ky, kz)`` points and their integer plane indices."""

    shape = np.asarray(mesh, dtype=np.int64).reshape(-1)
    if shape.size != 2 or np.any(shape < 1):
        raise ValueError(f"mesh must contain two positive integers, got {mesh!r}")
    shift = np.asarray(shift_grid, dtype=np.float64).reshape(-1)
    if shift.size != 2 or not np.all(np.isfinite(shift)):
        raise ValueError(
            f"shift_grid must contain two finite grid-unit shifts, got {shift_grid!r}"
        )
    # Integer indices are kept separately so the exact original grid ordering
    # survives periodic wrapping and output/plotting round trips.
    indices = np.indices(tuple(int(value) for value in shape), dtype=np.int64)
    indices = np.moveaxis(indices, 0, -1).reshape(-1, 2)
    in_plane = np.mod(
        (indices.astype(np.float64) + shift[None, :]) / shape[None, :], 1.0
    )
    points = np.empty((indices.shape[0], 3), dtype=np.float64)
    points[:, :2] = in_plane
    points[:, 2] = np.mod(_finite(kz_frac, "kz_frac"), 1.0)
    return points, indices


def plane_blocks(n_kpoints: int, block_size: int) -> list[tuple[int, int, int]]:
    """Build contiguous restart blocks as ``(block_id, start, stop)``."""

    total = _positive_integer(n_kpoints, "n_kpoints")
    width = _positive_integer(block_size, "block_size")
    return [
        (block_id, start, min(start + width, total))
        for block_id, start in enumerate(range(0, total, width))
    ]


def blocks_for_rank(
    blocks: Sequence[tuple[int, int, int]], rank: int, size: int
) -> list[tuple[int, int, int]]:
    """Static round-robin block ownership with at most one block per MPI round."""

    mpi_size = _positive_integer(size, "size")
    mpi_rank = int(rank)
    if mpi_rank < 0 or mpi_rank >= mpi_size:
        raise ValueError(f"rank={rank} is outside [0, {mpi_size})")
    return [block for block in blocks if block[0] % mpi_size == mpi_rank]


def _detected_mpi_size_from_environment() -> int:
    for name in (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "SLURM_NTASKS",
    ):
        text = os.environ.get(name)
        if text:
            try:
                return max(int(text), 1)
            except ValueError:
                continue
    return 1


def _mpi_context():
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm, int(comm.Get_rank()), int(comm.Get_size())
    except ImportError as exc:
        detected = _detected_mpi_size_from_environment()
        if detected > 1:
            raise RuntimeError(
                "multi-rank launch detected but mpi4py is unavailable; install "
                "mpi4py built against the cluster MPI module"
            ) from exc
        return None, 0, 1


def _allgather(comm, value):
    return [value] if comm is None else comm.allgather(value)


def _broadcast(comm, value, *, root: int = 0):
    return value if comm is None else comm.bcast(value, root=root)


def _barrier(comm) -> None:
    if comm is not None:
        comm.Barrier()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _file_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    absolute = Path(path).expanduser().resolve(strict=True)
    stat = absolute.stat()
    if not absolute.is_file():
        raise FileNotFoundError(f"input is not a regular file: {absolute}")
    return {
        "path": str(absolute),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_plane_configuration(
    args,
    *,
    k_points: np.ndarray,
    k_grid_indices: np.ndarray,
) -> dict[str, Any]:
    """Build the restart identity; MPI size is intentionally not part of it."""

    operational = {
        "output",
        "summary_json",
        "work_dir",
        "resume",
        "dry_run",
        "reference_seconds_per_k",
        "compressed",
        "blas_threads",
        "kmesh",
        "kz",
        "k_shift_grid",
        "block_size",
    }
    analysis = {
        key: _jsonable(value)
        for key, value in vars(args).items()
        if key not in operational
    }
    analysis["num_threads"] = args.num_threads
    inputs = {
        "jr": _file_identity(args.jr),
        "djr": _file_identity(args.djr),
        "phonon_cache": _file_identity(args.phonon_cache),
    }
    if args.geometry_epr is not None:
        inputs["geometry_epr"] = _file_identity(args.geometry_epr)
    config = {
        "plane_schema_version": PLANE_SCHEMA_VERSION,
        "plane": {
            "mesh": [int(value) for value in args.kmesh],
            "kz_frac": float(np.mod(args.kz, 1.0)),
            "shift_grid": [float(value) for value in args.k_shift_grid],
            "flattening": "C-order; ky changes fastest",
            "n_kpoints": int(k_points.shape[0]),
            "first_k_frac": k_points[0].tolist(),
            "last_k_frac": k_points[-1].tolist(),
            "first_grid_index": k_grid_indices[0].tolist(),
            "last_grid_index": k_grid_indices[-1].tolist(),
        },
        "block_size": int(args.block_size),
        "analysis": analysis,
        "execution": {
            "numba_threads_per_rank": args.num_threads,
            "blas_threads_per_rank": int(args.blas_threads),
        },
        "inputs": inputs,
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config["configuration_sha256"] = hashlib.sha256(encoded).hexdigest()
    return config


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _pair_state(npz_path: Path, json_path: Path) -> bool:
    exists = (npz_path.exists(), json_path.exists())
    if exists[0] != exists[1]:
        raise RuntimeError(
            "incomplete checkpoint pair; preserve it for inspection and choose a "
            f"new work directory: {npz_path}, {json_path}"
        )
    return bool(exists[0])


def _shard_paths(work_dir: Path, block_id: int) -> tuple[Path, Path]:
    stem = f"block_{int(block_id):06d}"
    return work_dir / "shards" / f"{stem}.npz", work_dir / "shards" / f"{stem}.json"


def _common_paths(work_dir: Path) -> tuple[Path, Path]:
    return work_dir / "common_payload.npz", work_dir / "common_summary.json"


def _validate_shard_metadata(
    npz_path: Path,
    json_path: Path,
    *,
    config_hash: str,
    block: tuple[int, int, int],
) -> None:
    block_id, start, stop = block
    summary = _load_json(json_path)
    if summary.get("configuration_sha256") != config_hash:
        raise ValueError(f"checkpoint configuration mismatch: {json_path}")
    expected = {
        "block_id": int(block_id),
        "global_start": int(start),
        "global_stop": int(stop),
    }
    for key, value in expected.items():
        if int(summary.get(key, -1)) != value:
            raise ValueError(
                f"checkpoint {key} mismatch in {json_path}: "
                f"{summary.get(key)!r} != {value}"
            )
    with np.load(npz_path, allow_pickle=False) as loaded:
        required = set(K_DEPENDENT_KEYS) | {
            "plane_global_k_indices",
            "plane_k_grid_indices",
        }
        missing = required - set(loaded.files)
        if missing:
            raise KeyError(f"checkpoint {npz_path} lacks arrays: {sorted(missing)}")
        expected_indices = np.arange(start, stop, dtype=np.int64)
        np.testing.assert_array_equal(
            np.asarray(loaded["plane_global_k_indices"], dtype=np.int64),
            expected_indices,
        )
        for key in K_DEPENDENT_KEYS:
            if np.asarray(loaded[key]).shape[0] != stop - start:
                raise ValueError(
                    f"checkpoint {key} leading size is inconsistent in {npz_path}"
                )


def _split_result_payload(
    result: RotationalCouplingResult,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    payload = result.payload
    missing = set(K_DEPENDENT_KEYS) - set(payload)
    if missing:
        raise KeyError(f"coupling result lacks k-dependent arrays: {sorted(missing)}")
    k_payload = {key: np.asarray(payload[key]) for key in K_DEPENDENT_KEYS}
    common_payload = {
        key: np.asarray(value)
        for key, value in payload.items()
        if key not in K_DEPENDENT_KEYS
    }
    nk = k_payload["k_points_frac"].shape[0]
    for key, value in k_payload.items():
        if value.shape[0] != nk:
            raise ValueError(
                f"k-dependent payload {key} has leading size {value.shape[0]} != {nk}"
            )
    return k_payload, common_payload


def _common_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in _COMMON_SUMMARY_EXCLUDED
    }


def _block_summary(
    summary: Mapping[str, Any],
    *,
    config_hash: str,
    block: tuple[int, int, int],
    rank: int,
    grid_indices: np.ndarray,
) -> dict[str, Any]:
    block_id, start, stop = block
    return {
        "plane_schema_version": PLANE_SCHEMA_VERSION,
        "configuration_sha256": config_hash,
        "block_id": int(block_id),
        "global_start": int(start),
        "global_stop": int(stop),
        "n_kpoints": int(stop - start),
        "producer_rank": int(rank),
        "grid_indices_first": grid_indices[0].tolist(),
        "grid_indices_last": grid_indices[-1].tolist(),
        "analysis": {
            key: summary[key] for key in _BLOCK_SUMMARY_KEYS if key in summary
        },
    }


def _common_compatibility_error(
    reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]
) -> str | None:
    if set(reference) != set(candidate):
        return (
            "common payload keys differ: "
            f"missing={sorted(set(reference) - set(candidate))}, "
            f"extra={sorted(set(candidate) - set(reference))}"
        )
    for key in sorted(reference):
        left = np.asarray(reference[key])
        right = np.asarray(candidate[key])
        if left.shape != right.shape or left.dtype.kind != right.dtype.kind:
            return (
                f"common payload {key} shape/type differs: "
                f"{left.shape}/{left.dtype} != {right.shape}/{right.dtype}"
            )
        if left.dtype.kind == "b":
            if not np.array_equal(left, right):
                return f"common payload {key} differs"
        elif left.dtype.kind in "iufc":
            if not np.allclose(
                left,
                right,
                rtol=2.0e-12,
                atol=2.0e-13,
                equal_nan=True,
            ):
                residual = float(np.max(np.abs(left - right), initial=0.0))
                return f"common payload {key} differs; max_abs={residual:.6e}"
        elif not np.array_equal(left, right):
            return f"common payload {key} differs"
    return None


def _write_common(
    work_dir: Path,
    *,
    config_hash: str,
    common_payload: Mapping[str, np.ndarray],
    analysis_summary: Mapping[str, Any],
    compressed: bool,
) -> None:
    npz_path, json_path = _common_paths(work_dir)
    payload = dict(common_payload)
    payload.update(
        {
            "chirality_plane_schema_version": np.asarray(
                PLANE_SCHEMA_VERSION, dtype=np.int32
            ),
            "plane_configuration_sha256": np.asarray(config_hash),
        }
    )
    summary = {
        "plane_schema_version": PLANE_SCHEMA_VERSION,
        "configuration_sha256": config_hash,
        "analysis": _common_summary(analysis_summary),
    }
    write_rotational_analysis(
        npz_path,
        json_path,
        RotationalAnalysisResult(payload=payload, summary=summary),
        compressed=compressed,
    )


def _write_shard(
    work_dir: Path,
    *,
    config_hash: str,
    block: tuple[int, int, int],
    rank: int,
    grid_indices: np.ndarray,
    k_payload: Mapping[str, np.ndarray],
    analysis_summary: Mapping[str, Any],
    compressed: bool,
) -> None:
    npz_path, json_path = _shard_paths(work_dir, block[0])
    start, stop = block[1], block[2]
    payload = dict(k_payload)
    payload.update(
        {
            "plane_global_k_indices": np.arange(start, stop, dtype=np.int64),
            "plane_k_grid_indices": np.asarray(grid_indices, dtype=np.int64),
        }
    )
    summary = _block_summary(
        analysis_summary,
        config_hash=config_hash,
        block=block,
        rank=rank,
        grid_indices=grid_indices,
    )
    write_rotational_analysis(
        npz_path,
        json_path,
        RotationalAnalysisResult(payload=payload, summary=summary),
        compressed=compressed,
    )


def _aggregate_numeric_dicts(
    dictionaries: Sequence[Mapping[str, Any]],
    *,
    minimum_keys: Sequence[str] = (),
) -> dict[str, float]:
    minimum = set(minimum_keys)
    keys = (
        set.intersection(
            *[
                {key for key, value in item.items() if isinstance(value, (int, float))}
                for item in dictionaries
            ]
        )
        if dictionaries
        else set()
    )
    return {
        key: float(
            (min if key in minimum else max)(float(item[key]) for item in dictionaries)
        )
        for key in sorted(keys)
    }


def merge_plane_shards(
    *,
    work_dir: Path,
    blocks: Sequence[tuple[int, int, int]],
    config: Mapping[str, Any],
    k_points: np.ndarray,
    k_grid_indices: np.ndarray,
    output: Path,
    summary_json: Path,
    mpi_size: int,
    wall_seconds: float,
    compressed: bool,
) -> tuple[str, str]:
    """Validate all shards and atomically publish one plane NPZ/JSON pair."""

    config_hash = str(config["configuration_sha256"])
    common_npz, common_json = _common_paths(work_dir)
    if not _pair_state(common_npz, common_json):
        raise RuntimeError("common plane payload is missing")
    common_summary = _load_json(common_json)
    if common_summary.get("configuration_sha256") != config_hash:
        raise ValueError("common plane payload has a different configuration")
    payload = _load_npz(common_npz)
    payload.pop("plane_configuration_sha256", None)
    payload.pop("chirality_plane_schema_version", None)

    arrays: dict[str, list[np.ndarray]] = {key: [] for key in K_DEPENDENT_KEYS}
    shard_summaries = []
    merged_indices = []
    merged_grid_indices = []
    for block in blocks:
        npz_path, json_path = _shard_paths(work_dir, block[0])
        if not _pair_state(npz_path, json_path):
            raise RuntimeError(f"missing completed checkpoint for block {block[0]}")
        _validate_shard_metadata(
            npz_path,
            json_path,
            config_hash=config_hash,
            block=block,
        )
        shard_summaries.append(_load_json(json_path))
        shard = _load_npz(npz_path)
        for key in K_DEPENDENT_KEYS:
            arrays[key].append(np.asarray(shard[key]))
        merged_indices.append(np.asarray(shard["plane_global_k_indices"]))
        merged_grid_indices.append(np.asarray(shard["plane_k_grid_indices"]))

    global_indices = np.concatenate(merged_indices, axis=0)
    grid_indices = np.concatenate(merged_grid_indices, axis=0)
    np.testing.assert_array_equal(global_indices, np.arange(k_points.shape[0]))
    np.testing.assert_array_equal(grid_indices, k_grid_indices)
    for key, chunks in arrays.items():
        payload[key] = np.concatenate(chunks, axis=0)
    np.testing.assert_allclose(
        payload["k_points_frac"], k_points, rtol=0.0, atol=2.0e-15
    )

    chirality = np.asarray(payload["physical_channel_chirality"], dtype=np.int8)
    gamma = np.asarray(payload["gamma_hwhm_mev"], dtype=np.float64)
    nphysical = int(chirality.size)
    if gamma.ndim != 2 or gamma.shape[1] < nphysical:
        raise ValueError(
            "merged linewidth shape is inconsistent with chirality metadata"
        )
    minus = np.flatnonzero(chirality == -1)
    plus = np.flatnonzero(chirality == 1)
    if minus.size != 1 or plus.size != 1:
        raise ValueError(
            "plane chirality contrast requires one physical chi=-1 and one chi=+1 channel"
        )
    gamma_minus = gamma[:, int(minus[0])]
    gamma_plus = gamma[:, int(plus[0])]
    denominator = gamma_plus + gamma_minus
    contrast = np.divide(
        gamma_plus - gamma_minus,
        denominator,
        out=np.zeros_like(denominator),
        where=np.abs(denominator) > 0.0,
    )
    ratio = np.divide(
        gamma_plus,
        gamma_minus,
        out=np.full_like(gamma_plus, np.nan),
        where=gamma_minus != 0.0,
    )

    payload.update(
        {
            "chirality_plane_schema_version": np.asarray(
                PLANE_SCHEMA_VERSION, dtype=np.int32
            ),
            "plane_configuration_sha256": np.asarray(config_hash),
            "k_plane_mesh": np.asarray(config["plane"]["mesh"], dtype=np.int32),
            "k_plane_grid_indices": np.asarray(k_grid_indices, dtype=np.int32),
            "k_plane_shift_grid": np.asarray(
                config["plane"]["shift_grid"], dtype=np.float64
            ),
            "k_plane_kz_frac": np.asarray(config["plane"]["kz_frac"], dtype=np.float64),
            "k_plane_flattening": np.asarray(config["plane"]["flattening"]),
            "physical_channel_count": np.asarray(nphysical, dtype=np.int32),
            "chirality_linewidth_contrast_plus_minus": contrast,
            "chirality_linewidth_ratio_plus_over_minus": ratio,
            # Compatibility aliases for the existing lifetime/BZ plotting tools.
            "k_mesh_frac": np.asarray(payload["k_points_frac"]),
            "linewidth": np.asarray(payload["gamma_hwhm_mev"]),
            "energy": np.asarray(payload["magnon_energies_mev"]),
        }
    )

    analysis_blocks = [item["analysis"] for item in shard_summaries]
    diagnostic_reasons = sorted(
        {
            reason
            for item in analysis_blocks
            for reason in item.get("diagnostic_reasons", [])
        }
    )
    statuses = sorted(
        {str(item.get("interpretation_status")) for item in analysis_blocks}
    )
    reconstruction = _aggregate_numeric_dicts(
        [item.get("reconstruction", {}) for item in analysis_blocks]
    )
    bdg = _aggregate_numeric_dicts(
        [item.get("bdg_validation", {}) for item in analysis_blocks],
        minimum_keys=(
            "external_metric_energy_minimum_mev",
            "internal_metric_energy_minimum_mev",
            "internal_metric_energy_minimum_mev_before_goldstone_omission",
        ),
    )
    block_runtimes = [
        float(item.get("runtime_seconds", 0.0)) for item in analysis_blocks
    ]
    finite_lifetime = np.asarray(payload["lifetime_ps"][:, :nphysical], dtype=float)
    finite_lifetime = finite_lifetime[np.isfinite(finite_lifetime)]
    summary = {
        "plane_schema_version": PLANE_SCHEMA_VERSION,
        "configuration_sha256": config_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "analysis_provenance": common_summary["analysis"],
        "interpretation_statuses": statuses,
        "diagnostic_reasons": diagnostic_reasons,
        "grid": {
            **config["plane"],
            "completed_blocks": len(blocks),
            "block_size": int(config["block_size"]),
        },
        "parallel": {
            "completion_mpi_size": int(mpi_size),
            "numba_threads_per_rank": config["execution"]["numba_threads_per_rank"],
            "blas_threads_per_rank": config["execution"]["blas_threads_per_rank"],
            "rank_assignment": "block_id modulo MPI size",
        },
        "validation": {
            "all_blocks_present": True,
            "global_k_indices_contiguous": True,
            "grid_indices_exact": True,
            "k_points_exact": True,
            "reconstruction_maxima": reconstruction,
            "bdg_extrema": bdg,
        },
        "physical_ranges": {
            "gamma_hwhm_mev_min": float(np.min(gamma[:, :nphysical])),
            "gamma_hwhm_mev_max": float(np.max(gamma[:, :nphysical])),
            "lifetime_ps_min_finite": (
                float(np.min(finite_lifetime)) if finite_lifetime.size else None
            ),
            "lifetime_ps_max_finite": (
                float(np.max(finite_lifetime)) if finite_lifetime.size else None
            ),
            "chirality_contrast_min": float(np.min(contrast)),
            "chirality_contrast_max": float(np.max(contrast)),
        },
        "runtime": {
            "current_invocation_wall_seconds": float(wall_seconds),
            "sum_block_runtime_seconds": float(sum(block_runtimes)),
            "maximum_block_runtime_seconds": float(max(block_runtimes, default=0.0)),
        },
        "output_npz": str(output.absolute()),
        "output_json": str(summary_json.absolute()),
        "checkpoint_directory": str(work_dir.absolute()),
    }
    return write_rotational_analysis(
        output,
        summary_json,
        RotationalAnalysisResult(payload=payload, summary=summary),
        compressed=compressed,
    )


def build_parser():
    parser = build_coupling_parser(
        require_k_points=False,
        include_output_arguments=False,
        description=(
            "Compute a restartable fixed-kz chirality-selectivity plane with MPI "
            "over external magnon k blocks and Numba threads over phonon q."
        ),
    )
    parser.add_argument(
        "--kmesh",
        type=int,
        nargs=2,
        required=True,
        metavar=("NKX", "NKY"),
        help="Periodic in-plane k mesh; ky changes fastest",
    )
    parser.add_argument("--kz", type=float, required=True, help="Fixed fractional kz")
    parser.add_argument(
        "--k-shift-grid",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("SX", "SY"),
        help="Optional in-plane shift in units of one k-grid step",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="k points per atomic checkpoint block (default: 8)",
    )
    parser.add_argument("--output", required=True, help="Merged plane NPZ")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Merged JSON report (default: output suffix changed to .json)",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Checkpoint directory (default: OUTPUT stem plus .parts)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse matching complete shards (default: true)",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="BLAS threads per MPI rank; keep at one with Numba q threads",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs/configuration and print the MPI/block layout only",
    )
    parser.add_argument(
        "--reference-seconds-per-k",
        type=float,
        default=None,
        help="Optional measured seconds/k used only for a wall-time estimate",
    )
    return parser


def _preflight_root(
    *,
    args,
    config: Mapping[str, Any],
    work_dir: Path,
    output: Path,
    summary_json: Path,
    n_blocks: int,
) -> dict[str, Any]:
    paths = [output.absolute(), summary_json.absolute(), work_dir.absolute()]
    if len({str(path) for path in paths}) != len(paths):
        raise ValueError("output, summary JSON, and work directory must be distinct")
    final_complete = _pair_state(output, summary_json)
    if final_complete:
        previous = _load_json(summary_json)
        if previous.get("configuration_sha256") != config["configuration_sha256"]:
            raise FileExistsError(
                "final output exists with a different configuration: "
                f"{output}, {summary_json}"
            )
        if not args.resume:
            raise FileExistsError(f"final output already exists: {output}")
        return {"already_complete": True}

    manifest_path = work_dir / "manifest.json"
    if manifest_path.exists():
        previous = _load_json(manifest_path)
        if previous.get("configuration_sha256") != config["configuration_sha256"]:
            raise ValueError(
                "checkpoint work directory belongs to a different configuration: "
                f"{work_dir}"
            )
        if not args.resume:
            raise FileExistsError(
                f"checkpoint directory exists and --no-resume was requested: {work_dir}"
            )
    else:
        if work_dir.exists() and any(work_dir.iterdir()):
            raise FileExistsError(
                f"nonempty checkpoint directory has no manifest: {work_dir}"
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "shards").mkdir(parents=True, exist_ok=True)
        manifest = {
            **config,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "n_blocks": int(n_blocks),
        }
        _atomic_json(manifest_path, manifest)
    return {"already_complete": False}


def _print_layout(args, *, rank: int, size: int, n_kpoints: int, n_blocks: int) -> None:
    if rank != 0:
        return
    print(
        "[chirality-plane] "
        f"kmesh={tuple(args.kmesh)} kz={float(np.mod(args.kz, 1.0)):.12g} "
        f"nk={n_kpoints} blocks={n_blocks} block_size={args.block_size}",
        flush=True,
    )
    print(
        "[chirality-plane] "
        f"mpi_ranks={size} numba_threads_per_rank={args.num_threads} "
        f"blas_threads_per_rank={args.blas_threads} "
        f"active_cpu_budget={size * (args.num_threads or 1)}",
        flush=True,
    )
    if args.reference_seconds_per_k is not None:
        seconds = _finite(args.reference_seconds_per_k, "reference_seconds_per_k")
        if seconds <= 0.0:
            raise ValueError("reference_seconds_per_k must be positive")
        ideal = n_kpoints * seconds / max(size, 1)
        print(
            "[chirality-plane] ideal wall estimate from supplied reference: "
            f"{ideal / 3600.0:.3f} h (excludes I/O/load imbalance)",
            flush=True,
        )


def _run(args, comm, rank: int, size: int) -> int:
    started = time.perf_counter()
    if args.save_vertices:
        raise ValueError(
            "the plane driver refuses --save-vertices; store accumulated observables only"
        )
    if args.goldstone_node_policy != "fail":
        raise ValueError(
            "the plane driver currently requires --goldstone-node-policy fail; a "
            "block-local union omission would make results depend on block partitioning"
        )
    if size > 1 and args.num_threads is None:
        raise ValueError(
            "--num-threads is required for MPI runs to prevent node oversubscription"
        )
    args.block_size = _positive_integer(args.block_size, "block_size")
    args.blas_threads = _positive_integer(args.blas_threads, "blas_threads")
    if args.num_threads is not None:
        args.num_threads = _positive_integer(args.num_threads, "num_threads")
        from numba import set_num_threads

        set_num_threads(args.num_threads)

    k_points, k_grid_indices = build_kz_plane_grid(
        args.kmesh, args.kz, args.k_shift_grid
    )
    blocks = plane_blocks(k_points.shape[0], args.block_size)
    output = Path(args.output).expanduser().absolute()
    summary_json = (
        Path(args.summary_json).expanduser().absolute()
        if args.summary_json is not None
        else output.with_suffix(".json")
    )
    work_dir = (
        Path(args.work_dir).expanduser().absolute()
        if args.work_dir is not None
        else output.with_name(f"{output.stem}.parts")
    )
    config = build_plane_configuration(
        args, k_points=k_points, k_grid_indices=k_grid_indices
    )
    config_hash = str(config["configuration_sha256"])
    rank_hashes = _allgather(comm, config_hash)
    if any(value != rank_hashes[0] for value in rank_hashes):
        raise RuntimeError(
            "MPI ranks resolved different input/configuration identities: "
            + ", ".join(rank_hashes)
        )
    _print_layout(
        args,
        rank=rank,
        size=size,
        n_kpoints=k_points.shape[0],
        n_blocks=len(blocks),
    )

    root_state = None
    if rank == 0:
        try:
            root_state = {
                "ok": True,
                **_preflight_root(
                    args=args,
                    config=config,
                    work_dir=work_dir,
                    output=output,
                    summary_json=summary_json,
                    n_blocks=len(blocks),
                ),
            }
        except Exception as exc:  # noqa: BLE001 - broadcast root failure to all ranks
            root_state = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    root_state = _broadcast(comm, root_state, root=0)
    if not root_state["ok"]:
        raise RuntimeError(root_state["error"])
    if root_state["already_complete"]:
        if rank == 0:
            print(f"[chirality-plane] already complete: {output}", flush=True)
        return 0
    if args.dry_run:
        if rank == 0:
            print(f"[chirality-plane] dry run manifest: {work_dir / 'manifest.json'}")
        return 0

    common_npz, common_json = _common_paths(work_dir)
    common_exists = _pair_state(common_npz, common_json)
    common_payload = _load_npz(common_npz) if common_exists else None
    if common_payload is not None:
        common_payload.pop("chirality_plane_schema_version", None)
        common_payload.pop("plane_configuration_sha256", None)

    nrounds = (len(blocks) + size - 1) // size
    for round_index in range(nrounds):
        block_id = round_index * size + rank
        block = blocks[block_id] if block_id < len(blocks) else None
        candidate = None
        local_error = None
        if block is not None:
            shard_npz, shard_json = _shard_paths(work_dir, block[0])
            try:
                complete = _pair_state(shard_npz, shard_json)
                if complete:
                    _validate_shard_metadata(
                        shard_npz,
                        shard_json,
                        config_hash=config_hash,
                        block=block,
                    )
                    print(
                        f"[chirality-plane][rank {rank}] resume block "
                        f"{block[0]} ({block[1]}:{block[2]})",
                        flush=True,
                    )
                else:
                    print(
                        f"[chirality-plane][rank {rank}] compute block "
                        f"{block[0]} ({block[1]}:{block[2]})",
                        flush=True,
                    )
                    result = analyze_rotational_coupling(
                        **coupling_kwargs_from_namespace(
                            args, k_points_frac=k_points[block[1] : block[2]]
                        )
                    )
                    k_payload, block_common = _split_result_payload(result)
                    candidate = (k_payload, block_common, result.summary)
            except Exception as exc:  # noqa: BLE001 - synchronize rank failure
                local_error = (
                    f"rank {rank}, block {block[0]}: {type(exc).__name__}: {exc}"
                )
        errors = [error for error in _allgather(comm, local_error) if error]
        if errors:
            raise RuntimeError("MPI block calculation failed:\n" + "\n".join(errors))

        common_write_error = None
        if round_index == 0 and rank == 0 and common_payload is None:
            try:
                if candidate is None:
                    raise RuntimeError(
                        "common payload is missing although block zero is already complete"
                    )
                _write_common(
                    work_dir,
                    config_hash=config_hash,
                    common_payload=candidate[1],
                    analysis_summary=candidate[2],
                    compressed=args.compressed,
                )
            except Exception as exc:  # noqa: BLE001 - broadcast root failure
                common_write_error = f"{type(exc).__name__}: {exc}"
        common_write_error = _broadcast(comm, common_write_error, root=0)
        if common_write_error:
            raise RuntimeError(
                f"failed to publish common payload: {common_write_error}"
            )
        _barrier(comm)
        if common_payload is None:
            common_payload = _load_npz(common_npz)
            common_payload.pop("chirality_plane_schema_version", None)
            common_payload.pop("plane_configuration_sha256", None)

        write_error = None
        if candidate is not None and block is not None:
            try:
                mismatch = _common_compatibility_error(common_payload, candidate[1])
                if mismatch is not None:
                    raise ValueError(mismatch)
                _write_shard(
                    work_dir,
                    config_hash=config_hash,
                    block=block,
                    rank=rank,
                    grid_indices=k_grid_indices[block[1] : block[2]],
                    k_payload=candidate[0],
                    analysis_summary=candidate[2],
                    compressed=args.compressed,
                )
            except Exception as exc:  # noqa: BLE001 - synchronize rank failure
                write_error = (
                    f"rank {rank}, block {block[0]}: {type(exc).__name__}: {exc}"
                )
        errors = [error for error in _allgather(comm, write_error) if error]
        if errors:
            raise RuntimeError(
                "MPI checkpoint publication failed:\n" + "\n".join(errors)
            )

    _barrier(comm)
    if rank == 0:
        written = merge_plane_shards(
            work_dir=work_dir,
            blocks=blocks,
            config=config,
            k_points=k_points,
            k_grid_indices=k_grid_indices,
            output=output,
            summary_json=summary_json,
            mpi_size=size,
            wall_seconds=time.perf_counter() - started,
            compressed=args.compressed,
        )
        print(f"[chirality-plane] wrote {written[0]}", flush=True)
        print(f"[chirality-plane] wrote {written[1]}", flush=True)
    _barrier(comm)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comm, rank, size = _mpi_context()
    # Limit BLAS only; Numba's OpenMP/TBB q parallelism keeps --num-threads.
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=int(args.blas_threads), user_api="blas"):
            return _run(args, comm, rank, size)
    except ImportError:
        if rank == 0:
            print(
                "[chirality-plane][WARN] threadpoolctl is unavailable; set "
                "OPENBLAS_NUM_THREADS=1 and MKL_NUM_THREADS=1 in the job script",
                flush=True,
            )
        return _run(args, comm, rank, size)


if __name__ == "__main__":
    raise SystemExit(main())
