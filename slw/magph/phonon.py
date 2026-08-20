"""Strict phonon-cache contracts and zero-point displacement conversion.

The retained cache builder stores displacement polarizations
``p = e / sqrt(M / m0)``.  A mass unit is therefore part of the numerical
contract: using an amu-based zero-point prefactor with masses stored in
electron-mass atomic units changes the vertex by roughly ``sqrt(1822.9)``.
This module keeps that choice explicit and never guesses it from the magnitude
of an atomic mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from slw.core.constants import (
    ZPF_ANG_SQRT_MEV_AMU,
    ZPF_ANG_SQRT_MEV_ELECTRON_MASS,
)


class PhononInputError(ValueError):
    """Raised when a phonon cache violates the native numerical contract."""


class PhononMassUnit(str, Enum):
    """Mass unit used to normalize stored phonon eigenvectors."""

    ELECTRON_MASS = "electron_mass"
    AMU = "amu"


class FrequencyFloorProvenance(str, Enum):
    """How a zero-point displacement frequency floor entered the result."""

    NONE = "none"
    EXPLICIT_ARGUMENT = "explicit_argument"


SUPPORTED_PHONON_CACHE_SCHEMA_VERSION = 3
CELL_GAUGE_VECTOR_CONVENTION = (
    "cell-gauge physical displacement u=e/sqrt(M); sum_kappa M_kappa|u|^2=1"
)
CELL_GAUGE_FOURIER_PHASE_CONVENTION = "u_(l,kappa) proportional exp(+i2pi q dot R_l)"


def _readonly_copy(value: Any, *, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _scalar_text(value: Any, *, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise PhononInputError(f"{name} must be scalar, got shape {array.shape}")
    item = array.item()
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="strict")
    if isinstance(item, str):
        return item
    raise PhononInputError(f"{name} must be text, got {type(item).__name__}")


def _schema_version(value: Any) -> int:
    array = np.asarray(value)
    if array.ndim != 0:
        raise PhononInputError("phonon_cache_schema_version must be scalar")
    item = array.item()
    if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
        raise PhononInputError("phonon_cache_schema_version must be an integer scalar")
    version = int(item)
    if version != SUPPORTED_PHONON_CACHE_SCHEMA_VERSION:
        raise PhononInputError(
            "unsupported phonon cache schema version "
            f"{version}; only {SUPPORTED_PHONON_CACHE_SCHEMA_VERSION} is supported"
        )
    return version


def _mass_unit(value: Any) -> PhononMassUnit:
    if isinstance(value, PhononMassUnit):
        return value
    text = _scalar_text(value, name="mass_unit")
    normalized = text.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"electron_mass", "electron_masses", "electronmass", "m_e"}:
        return PhononMassUnit.ELECTRON_MASS
    if normalized in {"amu", "u", "dalton", "atomic_mass_unit"}:
        return PhononMassUnit.AMU
    raise PhononInputError(
        f"mass_unit must explicitly be electron_mass or amu; got {text!r}"
    )


def _floor_provenance(value: Any) -> FrequencyFloorProvenance:
    if isinstance(value, FrequencyFloorProvenance):
        return value
    text = _scalar_text(value, name="frequency_floor_provenance")
    try:
        return FrequencyFloorProvenance(text)
    except ValueError as exc:
        choices = ", ".join(item.value for item in FrequencyFloorProvenance)
        raise PhononInputError(
            f"frequency_floor_provenance must be one of {choices}; got {text!r}"
        ) from exc


@dataclass(frozen=True)
class PhononCache:
    """Validated numerical view with ``sum (M/m0) |p|^2 = 1`` modes."""

    source: str
    schema_version: int
    q_points_frac: np.ndarray
    frequencies_mev: np.ndarray
    eigenvectors_mass_normalized: np.ndarray
    masses: np.ndarray
    mass_unit: PhononMassUnit
    q_mesh_shape: tuple[int, int, int] | None
    vector_convention: str
    fourier_phase_convention: str
    normalization_atol: float = 1.0e-7

    def __post_init__(self) -> None:
        source = str(self.source)
        if not source:
            raise PhononInputError("source must be a nonempty path or identifier")
        schema_version = _schema_version(self.schema_version)
        mass_unit = _mass_unit(self.mass_unit)
        tolerance = float(self.normalization_atol)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise PhononInputError("normalization_atol must be positive and finite")
        if self.vector_convention != CELL_GAUGE_VECTOR_CONVENTION:
            raise PhononInputError(
                "unsupported phonon_vector_convention; expected exactly "
                f"{CELL_GAUGE_VECTOR_CONVENTION!r}"
            )
        if self.fourier_phase_convention != CELL_GAUGE_FOURIER_PHASE_CONVENTION:
            raise PhononInputError(
                "unsupported fourier_phase_convention; expected exactly "
                f"{CELL_GAUGE_FOURIER_PHASE_CONVENTION!r}"
            )

        q_points = _readonly_copy(self.q_points_frac, dtype=np.float64)
        frequencies = _readonly_copy(self.frequencies_mev, dtype=np.float64)
        eigenvectors = _readonly_copy(
            self.eigenvectors_mass_normalized,
            dtype=np.complex128,
        )
        masses_raw = np.asarray(self.masses)
        if masses_raw.ndim != 1:
            raise PhononInputError(
                f"atomic masses must have shape (nat,), got {masses_raw.shape}"
            )
        masses = _readonly_copy(masses_raw, dtype=np.float64)

        if q_points.ndim != 2 or q_points.shape[1:] != (3,) or q_points.shape[0] == 0:
            raise PhononInputError(
                "q_mesh_flat_frac must have nonempty shape (nq,3), "
                f"got {q_points.shape}"
            )
        if frequencies.ndim != 2 or frequencies.shape[0] != q_points.shape[0]:
            raise PhononInputError(
                "ph_en_flat must have shape (nq,nmode) matching "
                f"q_mesh_flat_frac; got {frequencies.shape} and {q_points.shape}"
            )
        if masses.size == 0 or not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
            raise PhononInputError(
                "atomic masses must be nonempty, positive, and finite"
            )
        if frequencies.shape[1] != 3 * masses.size:
            raise PhononInputError(
                "a schema-v3 cache must contain exactly 3*nat phonon modes; "
                f"got nmode={frequencies.shape[1]} and nat={masses.size}"
            )
        expected_vectors = (
            frequencies.shape[0],
            frequencies.shape[1],
            masses.size,
            3,
        )
        if eigenvectors.shape != expected_vectors:
            raise PhononInputError(
                f"ph_vec_flat shape {eigenvectors.shape} != {expected_vectors}"
            )
        if not np.all(np.isfinite(q_points)):
            raise PhononInputError("q_mesh_flat_frac contains non-finite values")
        if not np.all(np.isfinite(frequencies)):
            raise PhononInputError("ph_en_flat contains non-finite values")
        if not np.all(np.isfinite(eigenvectors)):
            raise PhononInputError("ph_vec_flat contains non-finite values")

        mass_norm = np.einsum(
            "qvka,k,qvka->qv",
            eigenvectors.conj(),
            masses,
            eigenvectors,
            optimize=True,
        ).real
        if not np.allclose(mass_norm, 1.0, rtol=0.0, atol=tolerance):
            worst = float(np.max(np.abs(mass_norm - 1.0)))
            raise PhononInputError(
                "phonon eigenvectors violate sum_kappa M_kappa |u|^2 = 1; "
                f"maximum residual={worst:.3e}"
            )

        q_mesh_shape: tuple[int, int, int] | None
        if self.q_mesh_shape is None:
            q_mesh_shape = None
        else:
            mesh_float = np.asarray(self.q_mesh_shape, dtype=np.float64).reshape(-1)
            if (
                mesh_float.shape != (3,)
                or not np.all(np.isfinite(mesh_float))
                or not np.array_equal(mesh_float, np.rint(mesh_float))
                or np.any(mesh_float <= 0.0)
            ):
                raise PhononInputError(
                    "q_mesh_shape must contain three positive integers, "
                    f"got {self.q_mesh_shape!r}"
                )
            q_mesh_shape = (
                int(mesh_float[0]),
                int(mesh_float[1]),
                int(mesh_float[2]),
            )
            mesh_size = int(np.prod(np.asarray(q_mesh_shape, dtype=np.int64)))
            if mesh_size != frequencies.shape[0]:
                raise PhononInputError(
                    f"q_mesh_shape product {mesh_size} != nq {frequencies.shape[0]}"
                )

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "q_points_frac", q_points)
        object.__setattr__(self, "frequencies_mev", frequencies)
        object.__setattr__(self, "eigenvectors_mass_normalized", eigenvectors)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "mass_unit", mass_unit)
        object.__setattr__(self, "q_mesh_shape", q_mesh_shape)
        object.__setattr__(self, "normalization_atol", tolerance)

    @property
    def nq(self) -> int:
        return int(self.frequencies_mev.shape[0])

    @property
    def nmode(self) -> int:
        return int(self.frequencies_mev.shape[1])

    @property
    def nat(self) -> int:
        return int(self.masses.size)


@dataclass(frozen=True)
class ZeroPointDisplacement:
    """Physical mode displacements and explicit low-frequency regularization."""

    values_ang: np.ndarray
    input_frequencies_mev: np.ndarray
    effective_frequencies_mev: np.ndarray
    regularized: np.ndarray
    frequency_floor_mev: float | None
    frequency_floor_provenance: FrequencyFloorProvenance

    def __post_init__(self) -> None:
        values = _readonly_copy(self.values_ang, dtype=np.complex128)
        input_frequencies = _readonly_copy(
            self.input_frequencies_mev,
            dtype=np.float64,
        )
        effective_frequencies = _readonly_copy(
            self.effective_frequencies_mev,
            dtype=np.float64,
        )
        regularized_raw = np.asarray(self.regularized)
        if regularized_raw.dtype.kind != "b":
            raise PhononInputError("regularized must be a boolean array")
        regularized = _readonly_copy(regularized_raw, dtype=bool)
        provenance = _floor_provenance(self.frequency_floor_provenance)

        if values.ndim != 4 or values.shape[-1] != 3:
            raise PhononInputError(
                f"values_ang must have shape (nq,nmode,nat,3), got {values.shape}"
            )
        expected_frequency_shape = values.shape[:2]
        for name, array in (
            ("input_frequencies_mev", input_frequencies),
            ("effective_frequencies_mev", effective_frequencies),
            ("regularized", regularized),
        ):
            if array.shape != expected_frequency_shape:
                raise PhononInputError(
                    f"{name} shape {array.shape} != {expected_frequency_shape}"
                )
        if not np.all(np.isfinite(values)):
            raise PhononInputError("values_ang contains non-finite values")
        if not np.all(np.isfinite(input_frequencies)) or np.any(
            input_frequencies < 0.0
        ):
            raise PhononInputError(
                "input_frequencies_mev must be nonnegative and finite"
            )
        if not np.all(np.isfinite(effective_frequencies)) or np.any(
            effective_frequencies <= 0.0
        ):
            raise PhononInputError(
                "effective_frequencies_mev must be positive and finite"
            )

        if self.frequency_floor_mev is None:
            floor = None
            if provenance is not FrequencyFloorProvenance.NONE:
                raise PhononInputError(
                    "a missing frequency floor requires provenance='none'"
                )
            if np.any(regularized) or not np.array_equal(
                effective_frequencies,
                input_frequencies,
            ):
                raise PhononInputError(
                    "without a frequency floor, effective frequencies must equal "
                    "the input and no mode may be regularized"
                )
        else:
            floor = float(self.frequency_floor_mev)
            if not np.isfinite(floor) or floor <= 0.0:
                raise PhononInputError(
                    "frequency_floor_mev must be positive and finite"
                )
            if provenance is not FrequencyFloorProvenance.EXPLICIT_ARGUMENT:
                raise PhononInputError(
                    "frequency_floor_mev requires provenance='explicit_argument'"
                )
            expected_effective = np.maximum(input_frequencies, floor)
            expected_regularized = input_frequencies < floor
            if not np.array_equal(effective_frequencies, expected_effective):
                raise PhononInputError(
                    "effective_frequencies_mev is inconsistent with frequency_floor_mev"
                )
            if not np.array_equal(regularized, expected_regularized):
                raise PhononInputError(
                    "regularized is inconsistent with frequency_floor_mev"
                )

        object.__setattr__(self, "values_ang", values)
        object.__setattr__(self, "input_frequencies_mev", input_frequencies)
        object.__setattr__(self, "effective_frequencies_mev", effective_frequencies)
        object.__setattr__(self, "regularized", regularized)
        object.__setattr__(self, "frequency_floor_mev", floor)
        object.__setattr__(self, "frequency_floor_provenance", provenance)


def load_phonon_cache(
    path: str | Path,
    *,
    normalization_atol: float = 1.0e-7,
) -> PhononCache:
    """Load and strictly validate a schema-v3 phonon cache.

    Legacy caches without explicit units are intentionally rejected.  The
    native linewidth pipeline needs a physical displacement convention, not
    merely arrays with compatible shapes.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"phonon cache not found: {source}")
    tolerance = float(normalization_atol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise PhononInputError("normalization_atol must be positive and finite")

    required = {
        "phonon_cache_schema_version",
        "q_mesh_flat_frac",
        "ph_en_flat",
        "ph_vec_flat",
        "mass_unit",
        "energy_unit",
        "phonon_vector_convention",
        "fourier_phase_convention",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise PhononInputError(
                f"phonon cache is missing required fields: {', '.join(missing)}"
            )
        schema_version = _schema_version(archive["phonon_cache_schema_version"])

        mass_unit = _mass_unit(archive["mass_unit"])
        mass_key = (
            "atom_mass_electron"
            if mass_unit is PhononMassUnit.ELECTRON_MASS
            else "atom_mass_amu"
        )
        if mass_key not in archive.files:
            raise PhononInputError(
                f"mass_unit={mass_unit.value!r} requires cache field {mass_key!r}"
            )
        energy_unit = _scalar_text(archive["energy_unit"], name="energy_unit")
        if energy_unit.strip().lower() != "mev":
            raise PhononInputError(
                f"phonon energy_unit must be meV, got {energy_unit!r}"
            )
        convention = _scalar_text(
            archive["phonon_vector_convention"],
            name="phonon_vector_convention",
        )
        if convention != CELL_GAUGE_VECTOR_CONVENTION:
            raise PhononInputError(
                "unsupported phonon_vector_convention; expected exactly "
                f"{CELL_GAUGE_VECTOR_CONVENTION!r}"
            )
        fourier_phase_convention = _scalar_text(
            archive["fourier_phase_convention"],
            name="fourier_phase_convention",
        )
        if fourier_phase_convention != CELL_GAUGE_FOURIER_PHASE_CONVENTION:
            raise PhononInputError(
                "unsupported fourier_phase_convention; expected exactly "
                f"{CELL_GAUGE_FOURIER_PHASE_CONVENTION!r}"
            )

        q_points = _readonly_copy(archive["q_mesh_flat_frac"], dtype=np.float64)
        frequencies = _readonly_copy(archive["ph_en_flat"], dtype=np.float64)
        eigenvectors = _readonly_copy(archive["ph_vec_flat"], dtype=np.complex128)
        masses = _readonly_copy(archive[mass_key], dtype=np.float64).reshape(-1)
        mesh_raw = (
            np.asarray(archive["q_mesh_shape"])
            if "q_mesh_shape" in archive.files
            else np.asarray([], dtype=np.int64)
        )

    if q_points.ndim != 2 or q_points.shape[1:] != (3,) or q_points.shape[0] == 0:
        raise PhononInputError(
            f"q_mesh_flat_frac must have nonempty shape (nq,3), got {q_points.shape}"
        )
    if frequencies.ndim != 2 or frequencies.shape[0] != q_points.shape[0]:
        raise PhononInputError(
            "ph_en_flat must have shape (nq,nmode) matching q_mesh_flat_frac; "
            f"got {frequencies.shape} and {q_points.shape}"
        )
    expected_vectors = (frequencies.shape[0], frequencies.shape[1], masses.size, 3)
    if frequencies.shape[1] != 3 * masses.size:
        raise PhononInputError(
            "a schema-v3 cache must contain exactly 3*nat phonon modes; "
            f"got nmode={frequencies.shape[1]} and nat={masses.size}"
        )
    if eigenvectors.shape != expected_vectors:
        raise PhononInputError(
            f"ph_vec_flat shape {eigenvectors.shape} != {expected_vectors}"
        )
    if masses.size == 0 or not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise PhononInputError("atomic masses must be nonempty, positive, and finite")
    if not np.all(np.isfinite(q_points)):
        raise PhononInputError("q_mesh_flat_frac contains non-finite values")
    if not np.all(np.isfinite(frequencies)):
        raise PhononInputError("ph_en_flat contains non-finite values")
    if not np.all(np.isfinite(eigenvectors)):
        raise PhononInputError("ph_vec_flat contains non-finite values")

    mass_norm = np.einsum(
        "qvka,k,qvka->qv",
        eigenvectors.conj(),
        masses,
        eigenvectors,
        optimize=True,
    ).real
    if not np.allclose(mass_norm, 1.0, rtol=0.0, atol=tolerance):
        worst = float(np.max(np.abs(mass_norm - 1.0)))
        raise PhononInputError(
            "phonon eigenvectors violate sum_kappa M_kappa |u|^2 = 1; "
            f"maximum residual={worst:.3e}"
        )

    q_mesh_shape: tuple[int, int, int] | None
    if mesh_raw.size == 0:
        q_mesh_shape = None
    else:
        mesh_float = np.asarray(mesh_raw, dtype=np.float64).reshape(-1)
        if (
            mesh_float.shape != (3,)
            or not np.all(np.isfinite(mesh_float))
            or not np.array_equal(mesh_float, np.rint(mesh_float))
            or np.any(mesh_float <= 0.0)
        ):
            raise PhononInputError(
                f"q_mesh_shape must contain three positive integers, got {mesh_raw!r}"
            )
        q_mesh_shape = (
            int(mesh_float[0]),
            int(mesh_float[1]),
            int(mesh_float[2]),
        )
        mesh_size = int(np.prod(np.asarray(q_mesh_shape, dtype=np.int64)))
        if mesh_size != frequencies.shape[0]:
            raise PhononInputError(
                f"q_mesh_shape product {mesh_size} != nq {frequencies.shape[0]}"
            )

    return PhononCache(
        source=str(source),
        schema_version=schema_version,
        q_points_frac=q_points,
        frequencies_mev=frequencies,
        eigenvectors_mass_normalized=eigenvectors,
        masses=masses,
        mass_unit=mass_unit,
        q_mesh_shape=q_mesh_shape,
        vector_convention=convention,
        fourier_phase_convention=fourier_phase_convention,
        normalization_atol=tolerance,
    )


def zero_point_displacements(
    cache: PhononCache,
    *,
    frequency_floor_mev: float | None = None,
) -> ZeroPointDisplacement:
    r"""Convert ``p=e/sqrt(M/m0)`` modes to displacements in Angstrom.

    The factor follows ``sqrt(hbar^2/(2 M E))``.  A zero-frequency acoustic
    mode is singular, so any regularization must be supplied explicitly.
    Negative frequencies indicate an unstable phonon and are never clipped.
    """

    frequencies = cache.frequencies_mev
    if np.any(frequencies < 0.0):
        minimum = float(np.min(frequencies))
        raise PhononInputError(
            f"negative phonon frequency {minimum:.6g} meV cannot enter a linewidth"
        )
    if frequency_floor_mev is None:
        if np.any(frequencies == 0.0):
            raise PhononInputError(
                "zero-frequency modes require an explicit positive frequency_floor_mev"
            )
        effective = frequencies
        regularized = np.zeros(frequencies.shape, dtype=bool)
        floor = None
    else:
        floor = float(frequency_floor_mev)
        if not np.isfinite(floor) or floor <= 0.0:
            raise PhononInputError("frequency_floor_mev must be positive and finite")
        regularized = frequencies < floor
        effective = np.maximum(frequencies, floor)

    prefactor = (
        ZPF_ANG_SQRT_MEV_ELECTRON_MASS
        if cache.mass_unit is PhononMassUnit.ELECTRON_MASS
        else ZPF_ANG_SQRT_MEV_AMU
    )
    values = (
        cache.eigenvectors_mass_normalized
        * (prefactor / np.sqrt(effective))[:, :, None, None]
    )
    values = _readonly_copy(values, dtype=np.complex128)
    regularized = _readonly_copy(regularized, dtype=bool)
    return ZeroPointDisplacement(
        values_ang=values,
        input_frequencies_mev=frequencies,
        effective_frequencies_mev=effective,
        regularized=regularized,
        frequency_floor_mev=floor,
        frequency_floor_provenance=(
            FrequencyFloorProvenance.NONE
            if floor is None
            else FrequencyFloorProvenance.EXPLICIT_ARGUMENT
        ),
    )


__all__ = [
    "CELL_GAUGE_FOURIER_PHASE_CONVENTION",
    "CELL_GAUGE_VECTOR_CONVENTION",
    "SUPPORTED_PHONON_CACHE_SCHEMA_VERSION",
    "FrequencyFloorProvenance",
    "PhononCache",
    "PhononInputError",
    "PhononMassUnit",
    "ZeroPointDisplacement",
    "load_phonon_cache",
    "zero_point_displacements",
]
