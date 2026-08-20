"""Analyze atom-resolved phonon rotation in the physical-displacement metric.

This command consumes a phonon cache whose eigenvectors obey

``u[kappa] = epsilon[kappa] / sqrt(M[kappa])``

and therefore ``sum_kappa M[kappa] |u[kappa]|**2 = 1``.  The mass-metric
normalization is checked rather than inferred from chemical symbols.  Circular
weights and angular momentum can optionally be evaluated after diagonalizing
helicity within energy-degenerate mode subspaces.  The returned mode rotations
are the same linear transformations that must later be applied to a
magnon--phonon vertex.

The analysis intentionally stops before a coupling/self-energy calculation.
Its payload and mode rotations are reusable by that stage without tying this
tool to a particular material, lifetime manifest, or magnetic model.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import load_phonon_cache
from .rotational import (
    COMPONENT_LABELS,
    chiralize_degenerate_subspaces,
    decompose_atomistic_rotation,
)

ANALYSIS_SCHEMA_VERSION = 1


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    return result


def _positive_integer(value: int, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def _scalar_string(value: Any, default: str) -> str:
    if value is None:
        return default
    array = np.asarray(value)
    if array.size != 1:
        return default
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values)), initial=0.0))


@dataclass(frozen=True)
class ValidatedPhononCache:
    """The cache arrays needed for rotational observables."""

    q_frac: np.ndarray
    q_cart: np.ndarray
    energies_mev: np.ndarray
    displacements: np.ndarray
    masses: np.ndarray
    mass_norm: np.ndarray
    mass_unit: str
    vector_convention: str

    @property
    def n_qpoints(self) -> int:
        return int(self.energies_mev.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.energies_mev.shape[1])

    @property
    def n_atoms(self) -> int:
        return int(self.displacements.shape[2])


@dataclass(frozen=True)
class RotationalAnalysisResult:
    """Serializable arrays plus a compact JSON-compatible report."""

    payload: dict[str, np.ndarray]
    summary: dict[str, Any]


def validate_physical_phonon_cache(
    cache: Mapping[str, Any],
    *,
    normalization_tolerance: float = 1.0e-7,
) -> ValidatedPhononCache:
    """Validate shapes, masses, and physical-displacement normalization.

    A cache without ``atom_mass_electron`` is refused.  Guessing masses from
    atom labels would make the circular weights look plausible while using the
    wrong metric, so this strict behavior is deliberate.
    """

    tolerance = _finite_nonnegative(
        normalization_tolerance, "normalization_tolerance"
    )
    required = (
        "q_mesh_flat_frac",
        "q_mesh_flat_cart",
        "ph_en_flat",
        "ph_vec_flat",
        "atom_mass_electron",
    )
    missing = [key for key in required if key not in cache]
    if missing:
        raise KeyError(
            "phonon cache lacks fields required for a physical mass-metric "
            f"analysis: {missing}"
        )

    q_frac = np.asarray(cache["q_mesh_flat_frac"], dtype=np.float64)
    q_cart = np.asarray(cache["q_mesh_flat_cart"], dtype=np.float64)
    energies = np.asarray(cache["ph_en_flat"], dtype=np.float64)
    displacements = np.asarray(cache["ph_vec_flat"], dtype=np.complex128)
    masses = np.asarray(cache["atom_mass_electron"], dtype=np.float64)

    if q_frac.ndim != 2 or q_frac.shape[1] != 3 or q_frac.shape[0] < 1:
        raise ValueError(
            f"q_mesh_flat_frac must have nonempty shape (nq, 3), got {q_frac.shape}"
        )
    if q_cart.shape != q_frac.shape:
        raise ValueError(
            "q_mesh_flat_cart shape must match q_mesh_flat_frac: "
            f"{q_cart.shape} != {q_frac.shape}"
        )
    if (
        energies.ndim != 2
        or energies.shape[0] != q_frac.shape[0]
        or energies.shape[1] < 1
    ):
        raise ValueError(
            "ph_en_flat must have nonempty shape (nq, nmode) consistent with q points, "
            f"got {energies.shape} and nq={q_frac.shape[0]}"
        )
    if masses.ndim != 1 or masses.size < 1:
        raise ValueError(
            f"atom_mass_electron must be a nonempty one-dimensional array, got {masses.shape}"
        )
    expected_vectors = energies.shape + (masses.size, 3)
    if displacements.shape != expected_vectors:
        raise ValueError(
            "ph_vec_flat must have shape (nq, nmode, nat, 3), with nat from "
            f"atom_mass_electron; got {displacements.shape}, expected {expected_vectors}"
        )
    arrays = (q_frac, q_cart, energies, masses, displacements.real, displacements.imag)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("phonon cache arrays must contain only finite values")
    if np.any(masses <= 0.0):
        raise ValueError("all atom_mass_electron values must be strictly positive")

    mass_norm = np.einsum(
        "a,qmai,qmai->qm",
        masses,
        displacements.conj(),
        displacements,
        optimize=True,
    ).real
    normalization_error = _max_abs(mass_norm - 1.0)
    if normalization_error > tolerance:
        flat = int(np.argmax(np.abs(mass_norm - 1.0)))
        q_index, mode_index = np.unravel_index(flat, mass_norm.shape)
        raise ValueError(
            "phonon cache is not normalized as physical displacements "
            "sum_kappa M_kappa|u_kappa|^2=1: "
            f"max error={normalization_error:.6e} at q={q_index}, mode={mode_index} "
            f"(norm={mass_norm[q_index, mode_index]:.16g}), tolerance={tolerance:.6e}"
        )

    return ValidatedPhononCache(
        q_frac=q_frac,
        q_cart=q_cart,
        energies_mev=energies,
        displacements=displacements,
        masses=masses,
        mass_norm=mass_norm,
        mass_unit=_scalar_string(cache.get("mass_unit"), "electron_mass"),
        vector_convention=_scalar_string(
            cache.get("phonon_vector_convention"),
            "physical displacement u=e/sqrt(M)",
        ),
    )


def _empty_block_payload() -> dict[str, np.ndarray]:
    return {
        "degenerate_block_q_index": np.empty(0, dtype=np.int64),
        "degenerate_block_offsets": np.zeros(1, dtype=np.int64),
        "degenerate_block_mode_indices": np.empty(0, dtype=np.int32),
        "degenerate_block_helicity_eigenvalues": np.empty(0, dtype=np.float64),
        "degenerate_block_energy_min_mev": np.empty(0, dtype=np.float64),
        "degenerate_block_energy_max_mev": np.empty(0, dtype=np.float64),
        "degenerate_block_energy_spread_mev": np.empty(0, dtype=np.float64),
        "degenerate_block_helicity_offdiagonal_max": np.empty(0, dtype=np.float64),
        "degenerate_block_energy_offdiagonal_max_mev": np.empty(0, dtype=np.float64),
    }


def _pack_block_diagnostics(blocks: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not blocks:
        return _empty_block_payload()
    mode_lists = [np.asarray(item["mode_indices"], dtype=np.int32) for item in blocks]
    helicity_lists = [
        np.asarray(item["helicity_eigenvalues"], dtype=np.float64) for item in blocks
    ]
    counts = np.asarray([values.size for values in mode_lists], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    return {
        "degenerate_block_q_index": np.asarray(
            [item["q_index"] for item in blocks], dtype=np.int64
        ),
        "degenerate_block_offsets": offsets,
        "degenerate_block_mode_indices": np.concatenate(mode_lists),
        "degenerate_block_helicity_eigenvalues": np.concatenate(helicity_lists),
        "degenerate_block_energy_min_mev": np.asarray(
            [item["energy_min"] for item in blocks], dtype=np.float64
        ),
        "degenerate_block_energy_max_mev": np.asarray(
            [item["energy_max"] for item in blocks], dtype=np.float64
        ),
        "degenerate_block_energy_spread_mev": np.asarray(
            [item["energy_spread"] for item in blocks], dtype=np.float64
        ),
        "degenerate_block_helicity_offdiagonal_max": np.asarray(
            [item["helicity_offdiagonal_max"] for item in blocks], dtype=np.float64
        ),
        "degenerate_block_energy_offdiagonal_max_mev": np.asarray(
            [item["energy_offdiagonal_max"] for item in blocks], dtype=np.float64
        ),
    }


def analyze_phonon_rotations(
    cache: Mapping[str, Any],
    axis: np.ndarray,
    *,
    degeneracy_tolerance_mev: float | None = None,
    chiralization_atom_weights: np.ndarray | None = None,
    normalization_tolerance: float = 1.0e-7,
    orthonormal_tolerance: float = 1.0e-7,
    chunk_size: int = 256,
    save_chiralized_vectors: bool = True,
) -> RotationalAnalysisResult:
    """Build a material-independent atomistic rotational-analysis payload.

    If ``degeneracy_tolerance_mev`` is supplied, helicity is diagonalized only
    within mode blocks whose full energy span obeys that absolute tolerance.
    Relative energy tolerance is deliberately zero so the CLI option retains
    an unambiguous meV meaning.  Processing is vectorized within configurable
    q-point chunks.
    """

    validated = validate_physical_phonon_cache(
        cache, normalization_tolerance=normalization_tolerance
    )
    chunk = _positive_integer(chunk_size, "chunk_size")
    overlap_tolerance = _finite_nonnegative(
        orthonormal_tolerance, "orthonormal_tolerance"
    )
    if degeneracy_tolerance_mev is None:
        degeneracy_tolerance = None
    else:
        degeneracy_tolerance = _finite_nonnegative(
            degeneracy_tolerance_mev, "degeneracy_tolerance_mev"
        )
    if chiralization_atom_weights is None:
        operator_atom_weights = np.ones(validated.n_atoms, dtype=np.float64)
    else:
        operator_atom_weights = np.asarray(
            chiralization_atom_weights, dtype=np.float64
        )
        if degeneracy_tolerance is None:
            raise ValueError(
                "chiralization_atom_weights requires degeneracy_tolerance_mev"
            )
        if operator_atom_weights.shape != (validated.n_atoms,):
            raise ValueError(
                "chiralization_atom_weights must contain exactly one value per "
                f"atom: got {operator_atom_weights.shape}, expected "
                f"({validated.n_atoms},)"
            )
        if not np.all(np.isfinite(operator_atom_weights)):
            raise ValueError("chiralization_atom_weights must be finite")
        if not np.any(operator_atom_weights != 0.0):
            raise ValueError(
                "chiralization_atom_weights must contain at least one nonzero value"
            )

    nq = validated.n_qpoints
    nmode = validated.n_modes
    nat = validated.n_atoms
    atom_angular = np.empty((nq, nmode, nat, 3), dtype=np.float64)
    total_angular = np.empty((nq, nmode, 3), dtype=np.float64)
    atom_weights = np.empty((nq, nmode, nat, 3), dtype=np.float64)
    total_weights = np.empty((nq, nmode, 3), dtype=np.float64)
    analyzed_energies = np.array(validated.energies_mev, copy=True)
    analyzed_mass_norm = np.empty((nq, nmode), dtype=np.float64)

    mode_rotations = None
    helicity_eigenvalues = None
    chiralized_vectors = None
    block_records: list[dict[str, Any]] = []
    chiral_summary: dict[str, Any] = {
        "enabled": degeneracy_tolerance is not None,
        "abs_tolerance_mev": degeneracy_tolerance,
        "relative_tolerance": 0.0,
        "orthonormal_tolerance": overlap_tolerance,
        "operator_atom_weights": [float(x) for x in operator_atom_weights],
        "vectors_saved": bool(
            degeneracy_tolerance is not None and save_chiralized_vectors
        ),
        "n_degenerate_blocks": 0,
        "n_rotated_modes": 0,
        "max_mass_overlap_error_before": 0.0,
        "max_mass_overlap_error_after": 0.0,
        "max_rotation_unitarity_error": 0.0,
        "max_helicity_offdiagonal_after": 0.0,
        "max_energy_offdiagonal_after_mev": 0.0,
        "max_energy_spread_mev": 0.0,
    }
    if degeneracy_tolerance is not None:
        mode_rotations = np.empty((nq, nmode, nmode), dtype=np.complex128)
        helicity_eigenvalues = np.empty((nq, nmode), dtype=np.float64)
        if save_chiralized_vectors:
            chiralized_vectors = np.empty_like(validated.displacements)

    decomposition_maxima = {
        "max_normalization_error": 0.0,
        "max_relative_reconstruction_error": 0.0,
        "max_relative_weight_closure_error": 0.0,
        "max_angular_momentum_identity_error": 0.0,
        "max_angular_momentum_imaginary_residual": 0.0,
    }

    for start in range(0, nq, chunk):
        stop = min(start + chunk, nq)
        chunk_energies = validated.energies_mev[start:stop]
        chunk_vectors = validated.displacements[start:stop]

        if degeneracy_tolerance is not None:
            chiralized = chiralize_degenerate_subspaces(
                chunk_energies,
                chunk_vectors,
                validated.masses,
                axis,
                atom_weights=operator_atom_weights,
                abs_tolerance=degeneracy_tolerance,
                rel_tolerance=0.0,
                orthonormal_tolerance=overlap_tolerance,
            )
            chunk_energies = chiralized.energies
            chunk_vectors = chiralized.displacements
            analyzed_energies[start:stop] = chunk_energies
            mode_rotations[start:stop] = chiralized.rotations
            helicity_eigenvalues[start:stop] = chiralized.helicity_eigenvalues
            if chiralized_vectors is not None:
                chiralized_vectors[start:stop] = chunk_vectors

            diagnostics = chiralized.diagnostics
            chiral_summary["n_degenerate_blocks"] += diagnostics.n_degenerate_blocks
            chiral_summary["n_rotated_modes"] += diagnostics.n_rotated_modes
            for key, value in (
                ("max_mass_overlap_error_before", diagnostics.max_mass_overlap_error_before),
                ("max_mass_overlap_error_after", diagnostics.max_mass_overlap_error_after),
                ("max_rotation_unitarity_error", diagnostics.max_rotation_unitarity_error),
                ("max_helicity_offdiagonal_after", diagnostics.max_helicity_offdiagonal_after),
                ("max_energy_offdiagonal_after_mev", diagnostics.max_energy_offdiagonal_after),
                ("max_energy_spread_mev", diagnostics.max_energy_spread),
            ):
                chiral_summary[key] = max(float(chiral_summary[key]), float(value))
            for block in diagnostics.blocks:
                local_q = int(block.q_index[0])
                record = block.as_dict()
                record["q_index"] = start + local_q
                block_records.append(record)

        decomposition = decompose_atomistic_rotation(
            chunk_vectors, validated.masses, axis
        )
        atom_angular[start:stop] = decomposition.atom_angular_momentum_over_hbar
        total_angular[start:stop] = decomposition.angular_momentum_over_hbar
        atom_weights[start:stop] = decomposition.atom_weights
        total_weights[start:stop] = decomposition.weights
        analyzed_mass_norm[start:stop] = decomposition.diagnostics.mass_norm
        for key, value in decomposition.diagnostics.summary().items():
            decomposition_maxima[key] = max(decomposition_maxima[key], float(value))

    axis_vector = decomposition.axis_vector
    weight_closure = np.sum(total_weights, axis=-1) - analyzed_mass_norm
    angular_axis = np.einsum(
        "qmi,i->qm", total_angular, axis_vector, optimize=True
    )
    atom_angular_axis = np.einsum(
        "qmai,i->qma", atom_angular, axis_vector, optimize=True
    )
    operator_angular = np.einsum(
        "a,qmai->qmi", operator_atom_weights, atom_angular, optimize=True
    )
    operator_angular_axis = np.einsum(
        "qmi,i->qm", operator_angular, axis_vector, optimize=True
    )
    helicity_identity_error = angular_axis - (
        total_weights[..., 0] - total_weights[..., 1]
    )

    payload: dict[str, np.ndarray] = {
        "rotational_analysis_schema_version": np.asarray(
            ANALYSIS_SCHEMA_VERSION, dtype=np.int32
        ),
        "q_mesh_flat_frac": np.array(validated.q_frac, copy=True),
        "q_mesh_flat_cart": np.array(validated.q_cart, copy=True),
        "ph_en_original_mev": np.array(validated.energies_mev, copy=True),
        "ph_en_analyzed_mev": analyzed_energies,
        "atom_mass_electron": np.array(validated.masses, copy=True),
        "analysis_axis_cart": np.asarray(axis_vector, dtype=np.float64),
        "helicity_component_labels": np.asarray(COMPONENT_LABELS),
        "mode_mass_norm": analyzed_mass_norm,
        "atom_angular_momentum_over_hbar": atom_angular,
        "angular_momentum_over_hbar": total_angular,
        "angular_momentum_axis_over_hbar": angular_axis,
        "atom_angular_momentum_axis_over_hbar": atom_angular_axis,
        "chiralization_operator_angular_momentum_over_hbar": operator_angular,
        "chiralization_operator_angular_momentum_axis_over_hbar": operator_angular_axis,
        "atom_helicity_weights": atom_weights,
        "helicity_weights": total_weights,
        "helicity_weight_closure_error": weight_closure,
        "angular_momentum_helicity_identity_error": helicity_identity_error,
        "mass_unit": np.asarray(validated.mass_unit),
        "phonon_vector_convention": np.asarray(validated.vector_convention),
        "chiralization_enabled": np.asarray(
            degeneracy_tolerance is not None, dtype=np.bool_
        ),
        "chiralization_operator_atom_weights": operator_atom_weights,
    }
    payload.update(_pack_block_diagnostics(block_records))
    if mode_rotations is not None:
        payload["mode_rotations_new_from_old"] = mode_rotations
        payload["chiralization_helicity_eigenvalues"] = helicity_eigenvalues
    if chiralized_vectors is not None:
        payload["ph_vec_chiralized"] = chiralized_vectors

    # Preserve geometry/provenance that downstream atom-resolved plotting may
    # need, without copying arbitrary object arrays from an untrusted NPZ.
    for key in (
        "atom_frac",
        "lattice_ang",
        "q_mesh_shape",
        "fourier_phase_convention",
        "energy_unit",
        "q_cart_unit",
        "epr_source",
        "phonon_source",
        "phonon_asr",
        "phonon_loto",
    ):
        if key in cache:
            value = np.asarray(cache[key])
            if value.dtype.kind != "O":
                payload[key] = np.array(value, copy=True)

    normalization_error = validated.mass_norm - 1.0
    summary: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "n_qpoints": nq,
        "n_modes": nmode,
        "n_atoms": nat,
        "axis_cart": [float(x) for x in axis_vector],
        "mass_unit": validated.mass_unit,
        "energy_unit": "meV",
        "phonon_vector_convention": validated.vector_convention,
        "normalization": {
            "tolerance": float(normalization_tolerance),
            "minimum_mass_norm": float(np.min(validated.mass_norm)),
            "maximum_mass_norm": float(np.max(validated.mass_norm)),
            "max_abs_error": _max_abs(normalization_error),
        },
        "decomposition_diagnostics": decomposition_maxima,
        "max_abs_helicity_weight_closure_error": _max_abs(weight_closure),
        "max_abs_angular_momentum_helicity_identity_error": _max_abs(
            helicity_identity_error
        ),
        "max_angular_momentum_norm_over_hbar": float(
            np.max(np.linalg.norm(total_angular, axis=-1), initial=0.0)
        ),
        "max_abs_angular_momentum_axis_over_hbar": _max_abs(angular_axis),
        "max_atom_angular_momentum_norm_over_hbar": float(
            np.max(np.linalg.norm(atom_angular, axis=-1), initial=0.0)
        ),
        "max_abs_atom_angular_momentum_axis_over_hbar": _max_abs(
            atom_angular_axis
        ),
        "max_chiralization_operator_angular_momentum_norm_over_hbar": float(
            np.max(np.linalg.norm(operator_angular, axis=-1), initial=0.0)
        ),
        "max_abs_chiralization_operator_angular_momentum_axis_over_hbar": _max_abs(
            operator_angular_axis
        ),
        "chiralization": chiral_summary,
        "interpretation": {
            "component_order": list(COMPONENT_LABELS),
            "angular_momentum_formula": "-i sum_kappa M_kappa u_kappa^* cross u_kappa",
            "caution": (
                "nonzero circular-component weight is rotational participation; "
                "net phonon angular momentum is the signed plus-minus difference"
            ),
        },
    }
    return RotationalAnalysisResult(payload=payload, summary=summary)


def _write_npz_temp(
    directory: Path,
    name: str,
    payload: Mapping[str, np.ndarray],
    *,
    compressed: bool,
) -> Path:
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{name}.",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as handle:
            path = Path(handle.name)
            writer = np.savez_compressed if compressed else np.savez
            writer(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        return path
    except BaseException:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


def _write_json_temp(directory: Path, name: str, summary: Mapping[str, Any]) -> Path:
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{name}.",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as handle:
            path = Path(handle.name)
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return path
    except BaseException:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


def _commit_temp(temp_path: Path, final_path: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temp_path, final_path)
        return
    # A hard link is an atomic no-clobber commit on the same filesystem.
    # Unlike a check followed by os.replace, this also protects against races.
    os.link(temp_path, final_path)
    temp_path.unlink()


def write_rotational_analysis(
    output_npz: str | os.PathLike[str],
    summary_json: str | os.PathLike[str],
    result: RotationalAnalysisResult,
    *,
    overwrite: bool = False,
    compressed: bool = True,
) -> tuple[str, str]:
    """Atomically write NPZ and JSON outputs, refusing clobber by default."""

    npz_path = Path(output_npz).expanduser().absolute()
    json_path = Path(summary_json).expanduser().absolute()
    if npz_path == json_path:
        raise ValueError("NPZ and JSON output paths must be different")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = [str(path) for path in (npz_path, json_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing rotational-analysis output(s): "
                + ", ".join(existing)
            )

    npz_temp: Path | None = None
    json_temp: Path | None = None
    committed: list[Path] = []
    try:
        npz_temp = _write_npz_temp(
            npz_path.parent, npz_path.name, result.payload, compressed=compressed
        )
        json_temp = _write_json_temp(json_path.parent, json_path.name, result.summary)
        _commit_temp(npz_temp, npz_path, overwrite=overwrite)
        npz_temp = None
        committed.append(npz_path)
        _commit_temp(json_temp, json_path, overwrite=overwrite)
        json_temp = None
        committed.append(json_path)
    except BaseException:
        if not overwrite:
            # Roll back only files created by this attempted no-clobber commit.
            for path in committed:
                path.unlink(missing_ok=True)
        raise
    finally:
        if npz_temp is not None:
            npz_temp.unlink(missing_ok=True)
        if json_temp is not None:
            json_temp.unlink(missing_ok=True)
    return str(npz_path), str(json_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze atom-resolved circular phonon motion using the physical "
            "mass metric stored in a magph phonon cache."
        )
    )
    parser.add_argument("--phonon-cache", required=True, help="Input phonon cache NPZ")
    parser.add_argument(
        "--axis",
        required=True,
        type=float,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        help="Cartesian analysis axis (normalized internally)",
    )
    parser.add_argument("--output", required=True, help="Output analysis NPZ path")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="JSON summary path (default: --output with suffix .json)",
    )
    parser.add_argument(
        "--degeneracy-tol-mev",
        type=float,
        default=None,
        help=(
            "Absolute energy-span tolerance for helicity diagonalization in "
            "degenerate mode blocks; omit to preserve the input mode gauge"
        ),
    )
    parser.add_argument(
        "--normalization-tol",
        type=float,
        default=1.0e-7,
        help="Maximum allowed |sum M|u|^2 - 1| (default: 1e-7)",
    )
    parser.add_argument(
        "--chiralization-atom-weights",
        type=float,
        nargs="+",
        default=None,
        metavar="W",
        help=(
            "Optional one-weight-per-atom local helicity operator used during "
            "degenerate chiralization (e.g. signed weights for a staggered mode)"
        ),
    )
    parser.add_argument(
        "--orthonormal-tol",
        type=float,
        default=1.0e-7,
        help="Maximum mass-overlap error allowed before chiralization (default: 1e-7)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of q points processed per vectorized chunk (default: 256)",
    )
    parser.add_argument(
        "--save-chiralized-vectors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store rotated eigenvectors when chiralization is enabled (default: true)",
    )
    parser.add_argument(
        "--compressed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress the output NPZ (default: true)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs; default behavior is no-clobber",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_path = Path(args.phonon_cache).expanduser().absolute()
    cache = load_phonon_cache(str(cache_path))
    result = analyze_phonon_rotations(
        cache,
        np.asarray(args.axis, dtype=np.float64),
        degeneracy_tolerance_mev=args.degeneracy_tol_mev,
        chiralization_atom_weights=args.chiralization_atom_weights,
        normalization_tolerance=args.normalization_tol,
        orthonormal_tolerance=args.orthonormal_tol,
        chunk_size=args.chunk_size,
        save_chiralized_vectors=args.save_chiralized_vectors,
    )

    output_path = Path(args.output).expanduser()
    summary_path = (
        Path(args.summary_json).expanduser()
        if args.summary_json is not None
        else output_path.with_suffix(".json")
    )
    summary = dict(result.summary)
    summary["source_phonon_cache"] = str(cache_path)
    summary["output_npz"] = str(output_path.absolute())
    summary["output_json"] = str(summary_path.absolute())
    result = RotationalAnalysisResult(payload=result.payload, summary=summary)
    written_npz, written_json = write_rotational_analysis(
        output_path,
        summary_path,
        result,
        overwrite=args.overwrite,
        compressed=args.compressed,
    )
    print(f"[magph-rotation] wrote {written_npz}")
    print(f"[magph-rotation] wrote {written_json}")
    print(
        "[magph-rotation] "
        f"max|L_axis|/hbar={result.summary['max_abs_angular_momentum_axis_over_hbar']:.6g} "
        f"degenerate_blocks={result.summary['chiralization']['n_degenerate_blocks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
