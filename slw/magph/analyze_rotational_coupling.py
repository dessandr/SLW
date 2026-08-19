r"""Atom- and helicity-resolved scalar magnon--phonon scattering analysis.

This command resolves the physical-displacement phonon eigenvector of every
atom into circular ``plus``, circular ``minus``, and axial components about a
user supplied Cartesian axis.  The corresponding amplitudes are propagated
through the same atomic-gauge scalar-exchange vertex and bosonic BdG metric as
the lifetime calculation.  Both self terms and interference terms are kept:

``sum_(c,d) Gamma[c,d] == Gamma[sum_c g_c]``.

Here a component is ``(atom, helicity)``.  Consequently local counter-rotating
motion is retained even when the unit-cell angular momentum cancels.  The
scalar Heisenberg vertex describes one-phonon/two-magnon scattering; a
helicity correlation found here is therefore a spatial form-factor
selectivity, not by itself evidence for direct spin--phonon angular-momentum
conversion.

The code momentum ``q`` labels the absorbed phonon.  At the emission pole the
physical emitted phonon carries ``-q``.  Sign-resolved selectivity bins thus
use ``chi_initial * L(q)`` for absorption and ``-chi_initial * L(q)`` for
emission.  Vertex-only intensities and phase-space weighted linewidths are
stored separately.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numba import njit, prange

from slw.core.constants import BOHR_TO_ANG

from .adapter import load_djdu_npz, load_djr_h5, load_j_payload, load_phonon_cache
from .analyze_rotational_selectivity import (
    RotationalAnalysisResult,
    validate_physical_phonon_cache,
    write_rotational_analysis,
)
from .kernels import Magnon_Hamiltonian_v1
from .lswt import parse_spin_pattern
from .numerics import (
    bosonic_metric_diag,
    compute_linewidth_components_vectorized,
    compute_linewidth_vectorized,
    linewidth_observables,
    paraunitary_residual,
)
from .rotational import (
    COMPONENT_LABELS,
    chiralize_degenerate_subspaces,
    decompose_atomistic_rotation,
    helicity_projectors,
)
from .runtime import filter_exchange_shells
from .scattering import (
    apply_exchange_derivative_asr,
    build_atomic_gauge_lambda_components,
    filter_dj_payload_bonds,
    g_kq_loop_atomic_gauge_components,
    target_atom_indices,
)

COUPLING_SCHEMA_VERSION = 4
PROCESS_LABELS = ("emission_minus_q", "absorption_plus_q")
SIGN_BIN_LABELS = ("negative", "zero", "positive")
PHYSICAL_CHANNEL_CHIRALITY = np.asarray([-1, 1], dtype=np.int8)
BDG_SECTOR_CHIRALITY = np.asarray([-1, 1, 1, -1], dtype=np.int8)
HOLE_CHANNEL_INDICES = np.asarray([2, 3], dtype=np.int8)
HOLE_PHYSICAL_PARTNERS = np.asarray([1, 0], dtype=np.int8)
STRICT_CHIRAL_ENERGY_OFFDIAG_MEV = 1.0e-10


@dataclass(frozen=True)
class CouplingCacheGeometry:
    """Validated cache geometry and a uniform q-grid subset."""

    q_indices: np.ndarray
    q_mesh_shape: tuple[int, int, int]
    q_stride: tuple[int, int, int]
    q_mesh_shift_grid: np.ndarray
    q_mesh_origin_frac: np.ndarray
    q_mesh_sampling_convention: str
    selected_q_grid_indices: np.ndarray
    lattice_ang: np.ndarray
    reciprocal_ang_inv: np.ndarray
    atom_frac: np.ndarray
    atom_cart_ang: np.ndarray


@dataclass(frozen=True)
class RotationalCouplingResult:
    """Serializable coupling arrays and their JSON-compatible summary."""

    payload: dict[str, np.ndarray]
    summary: dict[str, Any]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _positive_integer(value: int, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values)), initial=0.0))


def _json_number(value: float) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar text metadata, got shape {array.shape}")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def validate_coupling_units(
    *,
    mass_unit: Any,
    static_exchange_unit: Any,
    static_unit_explicit: bool,
    dynamic_derivative_unit: Any,
    dynamic_unit_explicit: bool,
    allow_missing_explicit_units_diagnostic: bool = False,
) -> dict[str, Any]:
    """Enforce the legacy zero-point conversion's explicit unit contract."""

    mass_text = _scalar_text(mass_unit).strip()
    mass_normalized = mass_text.lower().replace("-", "_").replace(" ", "_")
    if mass_normalized not in {
        "electron_mass",
        "electron_masses",
        "electronmass",
        "m_e",
        "me",
    }:
        raise ValueError(
            "phonon cache mass_unit must be electron_mass for the magph "
            f"zero-point conversion, got {mass_text!r}"
        )

    missing = []
    if not bool(static_unit_explicit):
        missing.append("static_J_unit")
    if not bool(dynamic_unit_explicit):
        missing.append("dynamic_dJ_unit")
    if missing and not allow_missing_explicit_units_diagnostic:
        raise ValueError(
            "explicit unit metadata is required for a production linewidth: "
            f"missing={missing}. Use --allow-missing-units-diagnostic only to "
            "run with the legacy meV and meV/angstrom assumptions."
        )

    static_text = (
        _scalar_text(static_exchange_unit).strip()
        if bool(static_unit_explicit)
        else "meV (diagnostic assumption)"
    )
    if bool(static_unit_explicit):
        static_normalized = static_text.lower().replace(" ", "")
        if static_normalized not in {"mev", "millielectronvolt", "millielectronvolts"}:
            raise ValueError(f"static J energy unit must be meV, got {static_text!r}")

    derivative_text = (
        _scalar_text(dynamic_derivative_unit).strip()
        if bool(dynamic_unit_explicit)
        else "meV/angstrom (diagnostic assumption)"
    )
    derivative_normalized = (
        derivative_text.lower()
        .replace("å", "angstrom")
        .replace("Å", "angstrom")
        .replace("ångström", "angstrom")
        .replace("ang.", "ang")
        .replace(" ", "")
    )
    if bool(dynamic_unit_explicit) and derivative_normalized not in {
        "mev/a",
        "mev/ang",
        "mev/angstrom",
        "mev*a^-1",
        "mev*ang^-1",
        "mev*angstrom^-1",
        "mevang^-1",
        "mevangstrom^-1",
    }:
        raise ValueError(
            "dynamic dJ units must be meV/angstrom for the lifetime vertex, "
            f"got {derivative_text!r}"
        )
    return {
        "phonon_mass_unit": mass_text,
        "static_exchange_unit": static_text,
        "static_unit_explicit": bool(static_unit_explicit),
        "dynamic_derivative_unit": derivative_text,
        "dynamic_unit_explicit": bool(dynamic_unit_explicit),
        "missing_explicit_units": missing,
        "diagnostic_assumption_used": bool(missing),
        "vertex_energy_unit": "meV",
        "linewidth_unit": "meV",
        "lifetime_unit": "ps",
    }


def _decode_h5_scalar(value: Any) -> Any:
    """Decode HDF5 scalar/array metadata into JSON-compatible Python values."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode_h5_scalar(value.item())
        return [_decode_h5_scalar(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _decode_h5_scalar(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _soc_metadata_value_active(value: Any, *, tolerance: float = 1.0e-14) -> bool:
    """Conservatively classify an explicit SOC metadata value."""

    if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
        return bool(np.any(np.abs(value) > tolerance))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, complex)):
        return abs(value) > tolerance
    decoded = _decode_h5_scalar(value)
    if isinstance(decoded, list):
        return any(
            _soc_metadata_value_active(item, tolerance=tolerance) for item in decoded
        )
    text = str(decoded).strip().lower()
    return text not in {
        "",
        "0",
        "0.0",
        "none",
        "off",
        "false",
        "no",
        "disabled",
        "[]",
    }


def read_static_j_soc_metadata(path: str) -> dict[str, Any]:
    """Read explicit SOC evidence from a static-J HDF5 conservatively."""

    source = Path(path).expanduser()
    if source.suffix.lower() not in {".h5", ".hdf5"}:
        return {
            "hdf5": False,
            "metadata_present": False,
            "active": None,
            "evidence": [],
        }
    evidence: list[dict[str, Any]] = []
    with h5py.File(source, "r") as handle:
        for key in (
            "basic_data/soc",
            "soc",
            "basic_data/soc_entries",
            "soc_entries",
            "basic_data/spinor_hr",
            "spinor_hr",
        ):
            if key in handle:
                dataset = handle[key]
                if key.endswith("spinor_hr"):
                    evidence.append(
                        {
                            "path": key,
                            "value": {
                                "dataset_present": True,
                                "shape": list(dataset.shape),
                                "dtype": str(dataset.dtype),
                            },
                            "active": True,
                        }
                    )
                    continue
                raw = dataset[()]
                value = _decode_h5_scalar(raw)
                evidence.append(
                    {
                        "path": key,
                        "value": value,
                        "active": _soc_metadata_value_active(raw),
                    }
                )
        for key, raw in handle.attrs.items():
            lowered = str(key).lower()
            if lowered in {"soc", "soc_entries", "spinor_hr"}:
                value = _decode_h5_scalar(raw)
                evidence.append(
                    {
                        "path": f"@{key}",
                        "value": value,
                        "active": _soc_metadata_value_active(value),
                    }
                )
            elif lowered == "base_hamiltonian":
                value = _decode_h5_scalar(raw)
                evidence.append(
                    {
                        "path": f"@{key}",
                        "value": value,
                        "active": str(value).strip().lower() == "spinor_hr",
                    }
                )
    return {
        "hdf5": True,
        "metadata_present": bool(evidence),
        "active": any(item["active"] for item in evidence) if evidence else None,
        "evidence": evidence,
    }


def _h5_explicit_unit(
    handle: h5py.File, candidates: Sequence[str]
) -> tuple[str | None, str | None]:
    for key in candidates:
        if key in handle and isinstance(handle[key], h5py.Dataset):
            return _scalar_text(handle[key][()]), key
    for key in ("unit", "units"):
        if key in handle.attrs:
            return _scalar_text(handle.attrs[key]), f"@{key}"
    return None, None


def inspect_static_j_provenance(path: str) -> dict[str, Any]:
    """Record the exact scalar source and explicit static-J energy unit."""

    source = Path(path).expanduser()
    if source.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(source, "r") as handle:
            if "J_r/value" in handle:
                selected = "J_r/value"
                reduction = "direct_scalar"
            elif "tb2j_extra/jiso_tb2j" in handle:
                selected = "tb2j_extra/jiso_tb2j"
                reduction = "direct_scalar"
            elif "J_tensor_r" in handle:
                selected = "J_tensor_r"
                reduction = "trace_over_spin_axes_divided_by_3"
            else:
                raise KeyError("static J HDF5 has no loader-supported exchange dataset")
            unit, unit_source = _h5_explicit_unit(
                handle, ("basic_data/unit", "basic_data/units")
            )
        return {
            "format": "hdf5",
            "selected_dataset": selected,
            "scalar_reduction": reduction,
            "unit": unit,
            "unit_explicit": unit_source is not None,
            "unit_source": unit_source,
        }
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as handle:
            unit = _scalar_text(handle["units"]) if "units" in handle else None
            explicit_flag = (
                bool(np.asarray(handle["units_explicit"]).reshape(()).item())
                if "units_explicit" in handle
                else unit is not None
            )
        return {
            "format": "npz",
            "selected_dataset": "J_iso",
            "scalar_reduction": "direct_scalar",
            "unit": unit,
            "unit_explicit": bool(explicit_flag),
            "unit_source": "units" if unit is not None else None,
        }
    return {
        "format": "text",
        "selected_dataset": "parsed_scalar_J_iso",
        "scalar_reduction": "direct_scalar",
        "unit": None,
        "unit_explicit": False,
        "unit_source": None,
    }


def _extract_h5_geometry(path: str) -> dict[str, Any]:
    """Read lattice/positions from supported exchange or EPR HDF5 schemas."""

    with h5py.File(path, "r") as handle:
        nat = (
            int(np.asarray(handle["basic_data/nat"]).reshape(()).item())
            if "basic_data/nat" in handle
            else None
        )
        lattice = None
        lattice_source = None
        if "basic_data/lattice_ang" in handle:
            lattice = np.asarray(handle["basic_data/lattice_ang"], dtype=np.float64)
            lattice_source = "basic_data/lattice_ang"
        elif "basic_data/at" in handle and "basic_data/alat" in handle:
            lattice = (
                np.asarray(handle["basic_data/at"], dtype=np.float64)
                * float(handle["basic_data/alat"][()])
                * BOHR_TO_ANG
            )
            lattice_source = "basic_data/at * basic_data/alat"

        atom_frac = None
        position_source = None
        if "basic_data/tau_frac" in handle:
            atom_frac = np.asarray(handle["basic_data/tau_frac"], dtype=np.float64)
            position_source = "basic_data/tau_frac"
        elif lattice is not None and "basic_data/tau_cart_ang" in handle:
            atom_cart = np.asarray(handle["basic_data/tau_cart_ang"], dtype=np.float64)
            atom_frac = atom_cart @ np.linalg.inv(lattice)
            position_source = "basic_data/tau_cart_ang"
        elif "basic_data/tau" in handle and "basic_data/at" in handle:
            tau_alat = np.asarray(handle["basic_data/tau"], dtype=np.float64)
            raw_at = np.asarray(handle["basic_data/at"], dtype=np.float64)
            atom_frac = tau_alat @ np.linalg.inv(raw_at)
            position_source = "basic_data/tau with basic_data/at"
        if nat is None and atom_frac is not None:
            nat = int(atom_frac.shape[0])
    available = lattice is not None and atom_frac is not None and nat is not None
    return {
        "available": bool(available),
        "source_path": str(Path(path).expanduser().absolute()),
        "lattice_source": lattice_source,
        "position_source": position_source,
        "nat": nat,
        "lattice_ang": lattice,
        "atom_frac": None if atom_frac is None else np.mod(atom_frac, 1.0),
    }


def inspect_dynamic_geometry(
    path: str,
    provenance: Mapping[str, Any],
    *,
    geometry_epr_path: str | None = None,
) -> dict[str, Any]:
    """Use direct dJ geometry, then its recorded projection EPR geometry."""

    source = Path(path).expanduser()
    if source.suffix.lower() in {".h5", ".hdf5"}:
        direct = _extract_h5_geometry(str(source))
        if direct["available"]:
            direct["geometry_route"] = "direct_dynamic_hdf5"
            return direct
        if geometry_epr_path is not None:
            explicit_epr = Path(geometry_epr_path).expanduser().resolve(strict=True)
            inherited = _extract_h5_geometry(str(explicit_epr))
            inherited["geometry_route"] = "explicit_geometry_epr_override"
            inherited["recorded_projection_epr_h5"] = provenance.get(
                "projection_epr_h5"
            )
            return inherited
        projection_epr = provenance.get("projection_epr_h5")
        if projection_epr and Path(str(projection_epr)).expanduser().exists():
            inherited = _extract_h5_geometry(str(projection_epr))
            inherited["geometry_route"] = "scalar_projection_epr_h5"
            return inherited
        direct["geometry_route"] = "unavailable"
        return direct
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as handle:
            lattice = (
                np.asarray(handle["lattice_ang"], dtype=np.float64)
                if "lattice_ang" in handle
                else None
            )
            atom_frac = (
                np.asarray(handle["atom_frac"], dtype=np.float64)
                if "atom_frac" in handle
                else None
            )
            nat = (
                int(np.asarray(handle["nat"]).reshape(()).item())
                if "nat" in handle
                else (None if atom_frac is None else int(atom_frac.shape[0]))
            )
        return {
            "available": lattice is not None
            and atom_frac is not None
            and nat is not None,
            "source_path": str(source.absolute()),
            "geometry_route": "direct_dynamic_npz",
            "lattice_source": "lattice_ang" if lattice is not None else None,
            "position_source": "atom_frac" if atom_frac is not None else None,
            "nat": nat,
            "lattice_ang": lattice,
            "atom_frac": None if atom_frac is None else np.mod(atom_frac, 1.0),
        }
    return {
        "available": False,
        "source_path": str(source.absolute()),
        "geometry_route": "unsupported_format",
        "lattice_source": None,
        "position_source": None,
        "nat": None,
        "lattice_ang": None,
        "atom_frac": None,
    }


def compare_geometry_to_cache(
    source_geometry: Mapping[str, Any],
    cache_geometry: CouplingCacheGeometry,
    *,
    n_cache_atoms: int,
    tolerance: float,
    label: str,
) -> dict[str, Any]:
    """Strictly compare atom order, fractional positions, and lattice."""

    tol = _finite_nonnegative(tolerance, "geometry_tolerance")
    if not source_geometry.get("available", False):
        return {
            "label": label,
            "available": False,
            "valid": False,
            "issues": ["geometry_unavailable"],
            "geometry_route": source_geometry.get("geometry_route"),
            "source_path": source_geometry.get("source_path"),
        }
    lattice = np.asarray(source_geometry["lattice_ang"], dtype=np.float64)
    atom_frac = np.asarray(source_geometry["atom_frac"], dtype=np.float64)
    issues = []
    if int(source_geometry["nat"]) != int(n_cache_atoms):
        issues.append("nat_mismatch")
    lattice_error = (
        _max_abs(lattice - cache_geometry.lattice_ang)
        if lattice.shape == (3, 3)
        else np.inf
    )
    lattice_scale = max(_max_abs(cache_geometry.lattice_ang), 1.0)
    if lattice_error > tol * lattice_scale:
        issues.append("lattice_mismatch")
    expected_shape = (int(n_cache_atoms), 3)
    if atom_frac.shape == expected_shape:
        periodic_delta = np.mod(atom_frac - cache_geometry.atom_frac + 0.5, 1.0) - 0.5
        position_error = _max_abs(periodic_delta)
    else:
        position_error = np.inf
    if position_error > tol:
        issues.append("atom_fractional_position_or_order_mismatch")
    return {
        "label": label,
        "available": True,
        "valid": not issues,
        "issues": issues,
        "geometry_route": source_geometry.get("geometry_route", "direct"),
        "source_path": source_geometry.get("source_path"),
        "lattice_source": source_geometry.get("lattice_source"),
        "position_source": source_geometry.get("position_source"),
        "nat": int(source_geometry["nat"]),
        "max_lattice_abs_error_ang": _json_number(lattice_error),
        "max_periodic_fractional_position_error": _json_number(position_error),
        "tolerance": tol,
    }


def _bond_keys(i_atom: np.ndarray, j_atom: np.ndarray, cells: np.ndarray) -> list:
    """Return hashable directed real-space bond keys, retaining duplicates."""

    return [
        (int(i), int(j), tuple(int(value) for value in cell))
        for i, j, cell in zip(i_atom, j_atom, np.asarray(cells).reshape(-1, 3))
    ]


def compare_static_dynamic_bond_keys(
    j_tuple: tuple[np.ndarray, ...], dynamic_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Report exact directed bond-key consistency after any shell filter."""

    pair = np.asarray(dynamic_payload["pair_R"], dtype=np.int32).reshape(-1, 5)
    static_rows = _bond_keys(j_tuple[1], j_tuple[2], j_tuple[3])
    dynamic_rows = _bond_keys(pair[:, 0], pair[:, 1], pair[:, 2:5])
    static_counter = Counter(static_rows)
    dynamic_counter = Counter(dynamic_rows)
    static = set(static_counter)
    dynamic = set(dynamic_counter)
    static_only = static - dynamic
    dynamic_only = dynamic - static

    def ordered(values: set) -> list[list[Any]]:
        return [[item[0], item[1], list(item[2])] for item in sorted(values, key=str)]

    def multiplicities(
        counter: Counter, *, duplicate_only: bool
    ) -> list[dict[str, Any]]:
        return [
            {
                "key": [item[0], item[1], list(item[2])],
                "multiplicity": int(count),
            }
            for item, count in sorted(counter.items(), key=lambda pair: str(pair[0]))
            if not duplicate_only or count > 1
        ]

    mismatched = {
        key for key in static & dynamic if static_counter[key] != dynamic_counter[key]
    }

    return {
        "static_row_count": len(static_rows),
        "dynamic_row_count": len(dynamic_rows),
        "static_count": len(static),
        "dynamic_count": len(dynamic),
        "intersection_count": len(static & dynamic),
        "unique_keys_equal": static == dynamic,
        "multiplicities_equal": static_counter == dynamic_counter,
        "exactly_equal": static_counter == dynamic_counter,
        "static_only_count": len(static_only),
        "dynamic_only_count": len(dynamic_only),
        "static_only_keys": ordered(static_only),
        "dynamic_only_keys": ordered(dynamic_only),
        "static_duplicate_extra_count": len(static_rows) - len(static),
        "dynamic_duplicate_extra_count": len(dynamic_rows) - len(dynamic),
        "static_duplicate_keys": multiplicities(static_counter, duplicate_only=True),
        "dynamic_duplicate_keys": multiplicities(dynamic_counter, duplicate_only=True),
        "multiplicity_mismatch_count": len(mismatched),
        "multiplicity_mismatches": [
            {
                "key": [key[0], key[1], list(key[2])],
                "static_multiplicity": int(static_counter[key]),
                "dynamic_multiplicity": int(dynamic_counter[key]),
            }
            for key in sorted(mismatched, key=str)
        ],
    }


def validate_asr_report(
    report: Mapping[str, Any], *, requested_mode: str
) -> dict[str, Any]:
    """Fail closed when a checked/projected dJ ASR residual exceeds tolerance."""

    normalized = {
        "off": "none",
        "false": "none",
        "on": "project",
        "enforce": "project",
    }.get(str(requested_mode).strip().lower(), str(requested_mode).strip().lower())
    if normalized not in {"none", "check", "project"}:
        raise ValueError("dJ_asr must be one of: none, check, project")
    reported_mode = str(report.get("mode", "")).strip().lower()
    if reported_mode != normalized:
        raise RuntimeError(
            "dJ ASR implementation returned a different mode than requested: "
            f"requested={normalized!r}, reported={reported_mode!r}"
        )
    maximum = _finite_nonnegative(report.get("max_abs_after"), "ASR max_abs_after")
    tolerance = _finite_nonnegative(report.get("tolerance"), "ASR tolerance")
    violated = bool(report.get("violated_after", maximum > tolerance))
    if normalized in {"check", "project"} and (maximum > tolerance or violated):
        raise ValueError(
            f"dJ ASR {normalized} failed: max residual after={maximum:.6e} "
            f"exceeds tolerance={tolerance:.6e}"
        )
    return dict(report)


def estimate_saved_vertex_memory_bytes(
    *,
    n_components: int,
    n_qpoints: int,
    n_modes: int,
    n_bonds: int,
    n_kpoints: int,
    n_channels: int,
    complex_itemsize: int = np.dtype(np.complex128).itemsize,
) -> dict[str, int]:
    """Conservative peak estimate for the explicit in-memory vertex save path.

    The chunk lists coexist with the concatenated Lambda arrays (two Lambda
    copies).  Per-k concatenated ``g`` arrays then coexist with the final
    stacked output and the original chunk lists (three ``g`` copies).
    """

    dimensions = {
        "n_components": n_components,
        "n_qpoints": n_qpoints,
        "n_modes": n_modes,
        "n_bonds": n_bonds,
        "n_kpoints": n_kpoints,
        "n_channels": n_channels,
        "complex_itemsize": complex_itemsize,
    }
    checked = {
        name: _positive_integer(value, name) for name, value in dimensions.items()
    }
    lambda_bytes = (
        checked["complex_itemsize"]
        * checked["n_components"]
        * checked["n_qpoints"]
        * checked["n_modes"]
        * checked["n_bonds"]
    )
    g_bytes = (
        checked["complex_itemsize"]
        * checked["n_kpoints"]
        * checked["n_components"]
        * checked["n_qpoints"]
        * checked["n_modes"]
        * checked["n_channels"]
        * checked["n_channels"]
    )
    return {
        "lambda_payload_bytes": int(lambda_bytes),
        "g_payload_bytes": int(g_bytes),
        "combined_payload_bytes": int(lambda_bytes + g_bytes),
        "estimated_peak_bytes": int(2 * lambda_bytes + 3 * g_bytes),
    }


def select_uniform_q_indices(
    q_mesh_shape: Sequence[int],
    q_stride: Sequence[int],
    *,
    n_qpoints: int | None = None,
) -> np.ndarray:
    """Return cache-flat indices of a periodic uniform 3-D q submesh.

    Every mesh dimension must be divisible by its stride.  This excludes a
    shortened final interval at the periodic boundary, which would otherwise
    make equal q weights physically inconsistent.
    """

    shape = tuple(int(value) for value in q_mesh_shape)
    stride = tuple(int(value) for value in q_stride)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(
            f"q_mesh_shape must contain three positive integers, got {shape}"
        )
    if len(stride) != 3 or any(value <= 0 for value in stride):
        raise ValueError(f"q_stride must contain three positive integers, got {stride}")
    expected = int(np.prod(shape, dtype=np.int64))
    if n_qpoints is not None and int(n_qpoints) != expected:
        raise ValueError(
            f"q_mesh_shape product {expected} does not match cache nq={int(n_qpoints)}"
        )
    nondivisible = [
        (dimension, step)
        for dimension, step in zip(shape, stride)
        if dimension % step != 0
    ]
    if nondivisible:
        raise ValueError(
            "q_stride must divide every q_mesh_shape dimension for a periodic "
            f"uniform submesh; incompatible pairs={nondivisible}"
        )
    grid = np.arange(expected, dtype=np.int64).reshape(shape)
    return np.asarray(
        grid[:: stride[0], :: stride[1], :: stride[2]], dtype=np.int64
    ).ravel()


def validate_coupling_cache_geometry(
    cache: Mapping[str, Any],
    q_stride: Sequence[int],
    *,
    normalization_tolerance: float = 1.0e-7,
    geometry_tolerance: float = 1.0e-8,
) -> tuple[Any, CouplingCacheGeometry]:
    """Validate mass normalization, geometry, q ordering, and q stride."""

    validated = validate_physical_phonon_cache(
        cache, normalization_tolerance=normalization_tolerance
    )
    required = ("atom_frac", "lattice_ang", "q_mesh_shape", "mass_unit")
    missing = [key for key in required if key not in cache]
    if missing:
        raise KeyError(
            "phonon cache lacks coupling geometry fields; regenerate it with "
            f"the current adapter. Missing: {missing}"
        )

    atom_frac = np.asarray(cache["atom_frac"], dtype=np.float64)
    lattice = np.asarray(cache["lattice_ang"], dtype=np.float64)
    mesh_raw = np.asarray(cache["q_mesh_shape"], dtype=np.int64).reshape(-1)
    if atom_frac.shape != (validated.n_atoms, 3):
        raise ValueError(
            f"atom_frac shape {atom_frac.shape} != {(validated.n_atoms, 3)}"
        )
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError(
            f"lattice_ang must be a finite 3x3 matrix, got {lattice.shape}"
        )
    if not np.all(np.isfinite(atom_frac)):
        raise ValueError("atom_frac contains non-finite values")
    determinant = float(np.linalg.det(lattice))
    if not np.isfinite(determinant) or abs(determinant) <= np.finfo(float).tiny:
        raise ValueError("lattice_ang is singular")
    if mesh_raw.size != 3:
        raise ValueError(
            "q_mesh_shape must contain three dimensions; explicit non-grid q caches "
            "cannot be uniformly strided"
        )
    mesh = tuple(int(value) for value in mesh_raw)
    stride = tuple(int(value) for value in q_stride)
    indices = select_uniform_q_indices(mesh, stride, n_qpoints=validated.n_qpoints)

    tolerance = _finite_nonnegative(geometry_tolerance, "geometry_tolerance")
    cache_schema = (
        int(np.asarray(cache["phonon_cache_schema_version"]).reshape(()).item())
        if "phonon_cache_schema_version" in cache
        else None
    )
    shift_metadata = (
        "q_mesh_shift_grid",
        "q_mesh_origin_frac",
        "q_mesh_sampling_convention",
    )
    has_shift_metadata = [key in cache for key in shift_metadata]
    if (cache_schema is not None and cache_schema >= 3) or any(has_shift_metadata):
        missing_shift = [
            key
            for key, present in zip(shift_metadata, has_shift_metadata)
            if not present
        ]
        if missing_shift:
            raise KeyError(
                "shift-aware phonon cache metadata is incomplete; missing "
                f"{missing_shift}"
            )
        shift_grid = np.asarray(cache["q_mesh_shift_grid"], dtype=np.float64)
        origin_frac = np.asarray(cache["q_mesh_origin_frac"], dtype=np.float64)
        sampling_convention = _scalar_text(cache["q_mesh_sampling_convention"]).strip()
        if shift_grid.shape != (3,) or not np.all(np.isfinite(shift_grid)):
            raise ValueError("q_mesh_shift_grid must be a finite length-3 vector")
        if np.any(shift_grid < 0.0) or np.any(shift_grid >= 1.0):
            raise ValueError(
                "q_mesh_shift_grid metadata must already be canonical modulo 1 "
                "in grid-index units"
            )
        if origin_frac.shape != (3,) or not np.all(np.isfinite(origin_frac)):
            raise ValueError("q_mesh_origin_frac must be a finite length-3 vector")
        expected_origin = shift_grid / np.asarray(mesh, dtype=np.float64)
        if _max_abs(origin_frac - expected_origin) > tolerance:
            raise ValueError(
                "q_mesh_origin_frac is inconsistent with q_mesh_shift_grid/q_mesh_shape"
            )
        if not sampling_convention:
            raise ValueError("q_mesh_sampling_convention must be a nonempty string")
    else:
        shift_grid = np.zeros(3, dtype=np.float64)
        origin_frac = np.zeros(3, dtype=np.float64)
        sampling_convention = (
            "legacy unshifted uniform fractional grid; inferred q_mesh_shift_grid=0"
        )
    grid_indices = np.indices(mesh, dtype=np.int64).reshape(3, -1).T
    expected_frac = np.mod(
        (grid_indices + shift_grid[None, :])
        / np.asarray(mesh, dtype=np.float64)[None, :],
        1.0,
    )
    periodic_delta = np.mod(validated.q_frac - expected_frac + 0.5, 1.0) - 0.5
    frac_error = _max_abs(periodic_delta)
    if frac_error > tolerance:
        raise ValueError(
            "q_mesh_flat_frac does not follow the q_mesh_shape flattening order "
            f"required by --q-stride: max periodic error={frac_error:.6e}"
        )
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    expected_cart = validated.q_frac @ reciprocal
    cart_error = _max_abs(expected_cart - validated.q_cart)
    cart_scale = max(_max_abs(expected_cart), 1.0)
    if cart_error > tolerance * cart_scale:
        raise ValueError(
            "q_mesh_flat_cart is inconsistent with q_mesh_flat_frac and lattice_ang: "
            f"max error={cart_error:.6e}, scaled tolerance={tolerance * cart_scale:.6e}"
        )
    atom_cart = np.mod(atom_frac, 1.0) @ lattice
    return validated, CouplingCacheGeometry(
        q_indices=indices,
        q_mesh_shape=mesh,
        q_stride=stride,
        q_mesh_shift_grid=shift_grid,
        q_mesh_origin_frac=origin_frac,
        q_mesh_sampling_convention=sampling_convention,
        selected_q_grid_indices=grid_indices[indices],
        lattice_ang=lattice,
        reciprocal_ang_inv=reciprocal,
        atom_frac=np.mod(atom_frac, 1.0),
        atom_cart_ang=atom_cart,
    )


def build_atom_helicity_projectors(n_atoms: int, axis: Sequence[float]) -> np.ndarray:
    """Return a complete atom x (plus, minus, axis) resolution of identity."""

    nat = _positive_integer(n_atoms, "n_atoms")
    cartesian = helicity_projectors(np.asarray(axis, dtype=np.float64)).stacked
    atom_identity = np.eye(nat, dtype=np.complex128)
    resolved = atom_identity[:, None, :, None, None] * cartesian[None, :, None, :, :]
    return resolved.reshape(nat * len(COMPONENT_LABELS), nat, 3, 3)


def group_atom_component_matrix_by_helicity(
    matrix: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Sum both atom indices of a component interference matrix.

    The input ends in ``(ncomp, ncomp, nchannel)`` and may have arbitrary
    leading dimensions.  Components must be atom-major and helicity-minor.
    """

    values = np.asarray(matrix)
    nat = _positive_integer(n_atoms, "n_atoms")
    nhelicity = len(COMPONENT_LABELS)
    ncomp = nat * nhelicity
    if values.ndim < 3 or values.shape[-3] != ncomp or values.shape[-2] != ncomp:
        raise ValueError(
            "matrix must end in (nat*3,nat*3,nchannel), "
            f"got {values.shape} for nat={nat}"
        )
    leading = values.shape[:-3]
    nchannel = values.shape[-1]
    reshaped = values.reshape(leading + (nat, nhelicity, nat, nhelicity, nchannel))
    first_atom_axis = len(leading)
    second_atom_axis = len(leading) + 2
    return np.sum(reshaped, axis=(first_atom_axis, second_atom_axis))


def signed_selectivity_bin_indices(
    angular_momentum_axis_over_hbar: np.ndarray,
    initial_chirality: float,
    process: str,
    *,
    zero_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Classify ``chi*L`` into negative/zero/positive process-aware bins.

    ``process='emission'`` reverses the sign because the physical emitted
    phonon has momentum ``-q`` while the code vertex is indexed by ``q``.
    """

    values = np.asarray(angular_momentum_axis_over_hbar, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("angular momentum contains non-finite values")
    chirality = _finite(initial_chirality, "initial_chirality")
    selected = str(process).strip().lower().replace("-", "_")
    if selected in {"emission", "emit", "emission_minus_q"}:
        process_sign = -1.0
    elif selected in {"absorption", "absorb", "absorption_plus_q"}:
        process_sign = 1.0
    else:
        raise ValueError("process must be emission or absorption")
    tolerance = _finite_nonnegative(zero_tolerance, "zero_tolerance")
    signed = process_sign * chirality * values
    output = np.full(values.shape, 1, dtype=np.int8)
    output[signed < -tolerance] = 0
    output[signed > tolerance] = 2
    return output


def physical_lte_histogram_edges(
    angular_momentum_axis_over_hbar: np.ndarray,
    n_bins: int,
    *,
    maximum_abs_over_hbar: float | None = None,
    zero_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Build symmetric bin edges for physical Te-staggered angular momentum.

    An odd number of bins places zero at the center of one bin instead of on
    an edge.  The observed range is used by default; an explicit maximum is a
    strict coverage contract, not a clipping request.
    """

    angular = np.asarray(angular_momentum_axis_over_hbar, dtype=np.float64)
    if not np.all(np.isfinite(angular)):
        raise ValueError("angular momentum contains non-finite values")
    bins = int(n_bins)
    if bins == 0:
        if maximum_abs_over_hbar is not None:
            raise ValueError(
                "maximum_abs_over_hbar requires a positive histogram bin count"
            )
        return np.empty(0, dtype=np.float64)
    if bins < 3 or bins % 2 == 0:
        raise ValueError("LTe Gamma histogram bins must be odd and at least 3")
    tolerance = _finite_nonnegative(zero_tolerance, "zero_tolerance")
    observed = _max_abs(angular)
    if maximum_abs_over_hbar is None:
        limit = max(observed, tolerance, np.finfo(np.float64).eps)
    else:
        limit = _finite_positive(
            maximum_abs_over_hbar, "maximum_abs_over_hbar"
        )
        coverage_tolerance = 32.0 * np.finfo(np.float64).eps * max(limit, 1.0)
        if observed > limit + coverage_tolerance:
            raise ValueError(
                "LTe histogram range does not cover the analyzed modes: "
                f"observed max={observed:.12g}, requested max={limit:.12g}"
            )
    # Expand by one representable number so an extremal value is never lost
    # through a roundoff-level endpoint comparison in chunked histograms.
    limit = float(np.nextafter(limit, np.inf))
    return np.linspace(-limit, limit, bins + 1, dtype=np.float64)


def _bose_array(energy_mev: np.ndarray, temperature_k: float) -> np.ndarray:
    """Vectorized form of the lifetime kernel's signed-energy Bose function."""

    energy = np.asarray(energy_mev, dtype=np.float64)
    temperature = _finite_nonnegative(temperature_k, "temperature_k")
    output = np.zeros(energy.shape, dtype=np.float64)
    negative = energy < -1.0e-9
    positive = energy > 1.0e-9
    if temperature < 1.0e-9:
        output[negative] = -1.0
        return output

    # Keep this constant identical to slw.magph.kernels.KB_MEV without
    # embedding an independent thermodynamic convention.
    from .kernels import KB_MEV

    absolute = np.abs(energy)
    active = negative | positive
    x = np.zeros_like(absolute)
    x[active] = absolute[active] / (float(KB_MEV) * temperature)
    occupation = np.zeros_like(absolute)
    small = active & (x < 1.0e-10)
    regular = active & ~small & (x <= 700.0)
    occupation[small] = 1.0 / x[small]
    occupation[regular] = 1.0 / np.expm1(x[regular])
    output[positive] = occupation[positive]
    output[negative] = -(1.0 + occupation[negative])
    return output


def compute_vertex_phase_space_selectivity(
    full_vertex: np.ndarray,
    internal_energies_mev: np.ndarray,
    phonon_energies_mev: np.ndarray,
    external_energies_mev: np.ndarray,
    angular_momentum_axis_over_hbar: np.ndarray,
    *,
    temperature_k: float,
    eta_mev: float,
    metric_diag: np.ndarray,
    physical_channel_chirality: np.ndarray = PHYSICAL_CHANNEL_CHIRALITY,
    zero_tolerance: float = 1.0e-10,
    normalization_q_count: int | None = None,
    physical_lte_histogram_edges_over_hbar: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Bin vertex intensity and linewidth phase space by ``sign(chi*L)``.

    Vertex bins use only physical final magnon channels and contain no Bose,
    metric, or energy-conservation factors.  Phase-space bins use every BdG
    intermediate channel and exactly the retarded linewidth kernel.  Their
    separation prevents a density-of-states resonance from being mistaken
    for a polarization-selective matrix element.
    """

    vertex = np.asarray(full_vertex, dtype=np.complex128)
    internal = np.asarray(internal_energies_mev, dtype=np.float64)
    phonon = np.asarray(phonon_energies_mev, dtype=np.float64)
    external = np.asarray(external_energies_mev, dtype=np.float64).reshape(-1)
    angular = np.asarray(angular_momentum_axis_over_hbar, dtype=np.float64)
    metric = np.asarray(metric_diag, dtype=np.float64).reshape(-1)
    chirality = np.asarray(physical_channel_chirality, dtype=np.int8).reshape(-1)
    if vertex.ndim != 4:
        raise ValueError(
            "full_vertex must have shape (nq,nmode,ninternal,nexternal), "
            f"got {vertex.shape}"
        )
    nq, nmode, ninternal, nexternal = vertex.shape
    if internal.shape != (nq, ninternal):
        raise ValueError(f"internal energy shape {internal.shape} != {(nq, ninternal)}")
    if phonon.shape != (nq, nmode):
        raise ValueError(f"phonon energy shape {phonon.shape} != {(nq, nmode)}")
    if angular.shape != (nq, nmode):
        raise ValueError(f"angular momentum shape {angular.shape} != {(nq, nmode)}")
    if external.shape != (nexternal,) or metric.shape != (ninternal,):
        raise ValueError("external energy or BdG metric channel shape mismatch")
    nphysical = chirality.size
    if nphysical < 1 or nphysical > min(ninternal, nexternal):
        raise ValueError("physical_channel_chirality has an invalid length")
    if not np.all(np.isin(chirality, (-1, 1))):
        raise ValueError("physical channel chiralities must be +/-1")
    normalization = nq if normalization_q_count is None else int(normalization_q_count)
    if normalization < nq or normalization < 1:
        raise ValueError("normalization_q_count must be at least the local q count")
    broadening = _finite_positive(eta_mev, "eta_mev")
    temperature = _finite_nonnegative(temperature_k, "temperature_k")

    histogram_edges = None
    if physical_lte_histogram_edges_over_hbar is not None:
        histogram_edges = np.asarray(
            physical_lte_histogram_edges_over_hbar, dtype=np.float64
        ).reshape(-1)
        if (
            histogram_edges.size < 2
            or not np.all(np.isfinite(histogram_edges))
            or not np.all(np.diff(histogram_edges) > 0.0)
        ):
            raise ValueError(
                "physical LTe histogram edges must be finite and strictly increasing"
            )
        physical_extrema = np.concatenate((angular.ravel(), -angular.ravel()))
        endpoint_tolerance = 32.0 * np.finfo(np.float64).eps * max(
            _max_abs(histogram_edges), 1.0
        )
        if (
            np.min(physical_extrema, initial=0.0)
            < histogram_edges[0] - endpoint_tolerance
            or np.max(physical_extrema, initial=0.0)
            > histogram_edges[-1] + endpoint_tolerance
        ):
            raise ValueError("physical LTe histogram edges do not cover all modes")

    vertex_bins = np.zeros((2, nphysical, 3), dtype=np.float64)
    transition_bins = np.zeros((2, nphysical, nphysical, 3), dtype=np.float64)
    phase_bins = np.zeros((2, nphysical, 3), dtype=np.float64)
    counts = np.zeros((2, nphysical, 3), dtype=np.int64)
    lte_gamma_histogram = (
        np.zeros((2, nphysical, histogram_edges.size - 1), dtype=np.float64)
        if histogram_edges is not None
        else None
    )
    amplitude_physical = np.abs(vertex[:, :, :nphysical, :nphysical]) ** 2
    nb = _bose_array(phonon, temperature)
    nm = _bose_array(internal, temperature)
    metric_view = metric[None, None, :]

    for process_index, process in enumerate(("emission", "absorption")):
        physical_angular = -angular if process_index == 0 else angular
        for initial in range(nphysical):
            bins = signed_selectivity_bin_indices(
                angular,
                int(chirality[initial]),
                process,
                zero_tolerance=zero_tolerance,
            )
            flat_bins = bins.ravel()
            counts[process_index, initial] = np.bincount(flat_bins, minlength=3)
            transition = amplitude_physical[:, :, :, initial]
            for final in range(nphysical):
                transition_bins[process_index, final, initial] = np.bincount(
                    flat_bins,
                    weights=transition[:, :, final].ravel(),
                    minlength=3,
                ) / float(normalization)
            vertex_bins[process_index, initial] = np.sum(
                transition_bins[process_index, :, initial], axis=0
            )

            amplitude_bdg = np.abs(vertex[:, :, :, initial]) ** 2
            omega = external[initial]
            if process_index == 0:
                mismatch = omega - internal[:, None, :] - phonon[:, :, None]
                thermal = nb[:, :, None] + 1.0 + nm[:, None, :]
            else:
                mismatch = omega - internal[:, None, :] + phonon[:, :, None]
                thermal = nb[:, :, None] - nm[:, None, :]
            im_pole = -broadening / (mismatch * mismatch + broadening * broadening)
            # Gamma=-Im Sigma.  Sum the BdG intermediate channel only; q/mode
            # remains available for the helicity sign binning.
            gamma_qmode = -np.sum(
                amplitude_bdg * metric_view * thermal * im_pole,
                axis=-1,
            )
            phase_bins[process_index, initial] = np.bincount(
                flat_bins,
                weights=gamma_qmode.ravel(),
                minlength=3,
            ) / float(normalization)

            if lte_gamma_histogram is not None:
                lte_gamma_histogram[process_index, initial] = np.histogram(
                    physical_angular.ravel(),
                    bins=histogram_edges,
                    weights=gamma_qmode.ravel(),
                )[0] / float(normalization)

    result = {
        "vertex_intensity_mev2": vertex_bins,
        "vertex_transition_intensity_mev2": transition_bins,
        "phase_space_gamma_mev": phase_bins,
        "mode_counts": counts,
    }
    if lte_gamma_histogram is not None:
        result["physical_lte_gamma_histogram_mev"] = lte_gamma_histogram
    return result


def component_vertex_intensity_matrix(
    component_vertex: np.ndarray,
    *,
    physical_count: int,
    normalization_q_count: int | None = None,
) -> np.ndarray:
    """Return the real component Gram matrix for physical magnon transitions."""

    values = np.asarray(component_vertex, dtype=np.complex128)
    if values.ndim != 5:
        raise ValueError(
            "component_vertex must have shape (ncomp,nq,nmode,nfinal,ninitial)"
        )
    count = _positive_integer(physical_count, "physical_count")
    if count > min(values.shape[-2:]):
        raise ValueError("physical_count exceeds a vertex channel dimension")
    normalization = (
        values.shape[1] if normalization_q_count is None else int(normalization_q_count)
    )
    if normalization < values.shape[1] or normalization < 1:
        raise ValueError("normalization_q_count must be at least the local q count")
    physical = values[..., :count, :count]
    return np.einsum(
        "cqnlm,dqnlm->cdm",
        physical,
        physical.conj(),
        optimize=True,
    ).real / float(normalization)


@njit(parallel=True, fastmath=True)
def _magnon_q_metric_diagnostics(
    q_cart,
    k_cart,
    spin_S,
    j_tuple,
    lattice,
    atom_cart,
    spin_pattern,
    bond_factor,
    anisotropy_mev,
    metric_diag,
):
    """Check every internal k+q eigensystem, not only the external k."""

    nq = q_cart.shape[0]
    nchannel = metric_diag.shape[0]
    residual = np.zeros(nq, dtype=np.float64)
    metric_energy_minimum = np.zeros(nq, dtype=np.float64)
    for iq in prange(nq):
        energy, transform = Magnon_Hamiltonian_v1(
            k_cart + q_cart[iq],
            spin_S,
            j_tuple,
            lattice,
            atom_cart,
            spin_pattern,
            bond_factor,
            anisotropy_mev,
        )
        local_max = 0.0
        for first in range(nchannel):
            for second in range(nchannel):
                overlap = 0.0j
                for row in range(nchannel):
                    overlap += (
                        np.conjugate(transform[row, first])
                        * metric_diag[row]
                        * transform[row, second]
                    )
                if first == second:
                    overlap -= metric_diag[first]
                magnitude = np.abs(overlap)
                local_max = max(local_max, magnitude)
        residual[iq] = local_max
        local_energy_min = metric_diag[0] * energy[0]
        for channel in range(1, nchannel):
            value = metric_diag[channel] * energy[channel]
            local_energy_min = min(local_energy_min, value)
        metric_energy_minimum[iq] = local_energy_min
    return residual, metric_energy_minimum


def resolve_goldstone_quadrature_nodes(
    paraunitary_residual_by_kq: np.ndarray,
    metric_energy_minimum_by_kq_mev: np.ndarray,
    q_flat_indices: np.ndarray,
    q_frac: np.ndarray,
    k_frac: np.ndarray,
    *,
    anisotropy_mev: float,
    paraunitary_tolerance: float,
    metric_energy_tolerance_mev: float,
    goldstone_energy_tolerance_mev: float,
    policy: str = "fail",
    omission_prerequisites: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Identify defective/near-exact Goldstone quadrature nodes fail-closed.

    A candidate must have a non-negative (up to the independent metric-energy
    tolerance) metric-positive magnon energy no larger than the explicit
    Goldstone energy tolerance.  It must additionally either violate the
    paraunitary tolerance or be an exact-zero-anisotropy gapless node.  The
    latter branch is deliberately disabled for every nonzero anisotropy.

    With multiple source k points, the returned keep mask omits the union of
    candidates.  The report retains the diagnostics for every k at every
    omitted q so that a node triggered by only one source k is not hidden.
    """

    selected_policy = str(policy).strip().lower()
    if selected_policy not in {"fail", "omit"}:
        raise ValueError("goldstone_node_policy must be one of: fail, omit")
    para_tol = _finite_nonnegative(paraunitary_tolerance, "paraunitary_tolerance")
    metric_tol = _finite_nonnegative(
        metric_energy_tolerance_mev, "metric_energy_tolerance_mev"
    )
    energy_tol = _finite_nonnegative(
        goldstone_energy_tolerance_mev, "goldstone_energy_tolerance_mev"
    )
    anisotropy = _finite(anisotropy_mev, "anisotropy_mev")
    residual = np.asarray(paraunitary_residual_by_kq, dtype=np.float64)
    energy = np.asarray(metric_energy_minimum_by_kq_mev, dtype=np.float64)
    flat = np.asarray(q_flat_indices, dtype=np.int64).reshape(-1)
    q_values = np.asarray(q_frac, dtype=np.float64)
    k_values = np.asarray(k_frac, dtype=np.float64)
    if residual.ndim != 2 or energy.shape != residual.shape:
        raise ValueError(
            "Goldstone diagnostics must have matching (nk,nq) shapes, got "
            f"residual={residual.shape}, energy={energy.shape}"
        )
    nk, nq = residual.shape
    if flat.shape != (nq,) or q_values.shape != (nq, 3):
        raise ValueError("Goldstone q metadata is inconsistent with diagnostic nq")
    if k_values.shape != (nk, 3):
        raise ValueError("Goldstone k metadata is inconsistent with diagnostic nk")
    if not (
        np.all(np.isfinite(residual))
        and np.all(np.isfinite(energy))
        and np.all(np.isfinite(q_values))
        and np.all(np.isfinite(k_values))
    ):
        raise ValueError("Goldstone diagnostics or momentum metadata are non-finite")

    metric_nonnegative = energy >= -metric_tol
    near_gapless = metric_nonnegative & (energy <= energy_tol)
    para_defective = residual > para_tol
    zero_anisotropy_gapless = bool(anisotropy == 0.0)
    candidate = near_gapless & (
        para_defective | np.full(residual.shape, zero_anisotropy_gapless)
    )
    unstable = energy < -metric_tol
    if np.any(unstable):
        first_k, first_q = np.argwhere(unstable)[0]
        raise ValueError(
            "internal k+q magnon spectrum violates positive metric energy outside "
            "the Goldstone policy: "
            f"k_index={int(first_k)}, q_flat_index={int(flat[first_q])}, "
            f"energy={energy[first_k, first_q]:.6e} meV"
        )
    non_goldstone_defect = para_defective & ~candidate
    if np.any(non_goldstone_defect):
        first_k, first_q = np.argwhere(non_goldstone_defect)[0]
        raise ValueError(
            "internal k+q magnon BdG transform violates paraunitarity at a "
            "non-Goldstone node, which cannot be omitted by a multi-k union mask: "
            f"k_index={int(first_k)}, q_flat_index={int(flat[first_q])}, "
            f"energy={energy[first_k, first_q]:.6e} meV, "
            f"residual={residual[first_k, first_q]:.6e}"
        )
    union = np.any(candidate, axis=0)
    omitted_positions = np.flatnonzero(union)

    prerequisite_report = dict(omission_prerequisites or {})
    prerequisites_passed = bool(prerequisite_report.get("passed", False))
    prerequisite_report.setdefault("passed", prerequisites_passed)
    prerequisite_report.setdefault(
        "physical_basis",
        "dJ translational ASR/Ward identity makes the exact Goldstone-node vertex vanish",
    )

    per_k = []
    for ik in range(nk):
        nodes = []
        for iq in np.flatnonzero(candidate[ik]):
            reasons = []
            if para_defective[ik, iq]:
                reasons.append("paraunitary_residual_above_tolerance")
            if zero_anisotropy_gapless:
                reasons.append("zero_anisotropy_near_gapless_metric_energy")
            k_plus_q_unwrapped = k_values[ik] + q_values[iq]
            nodes.append(
                {
                    "selected_q_position": int(iq),
                    "q_flat_index": int(flat[iq]),
                    "q_frac": [float(value) for value in q_values[iq]],
                    "k_plus_q_frac": [
                        float(value) for value in np.mod(k_plus_q_unwrapped, 1.0)
                    ],
                    "k_plus_q_frac_unwrapped": [
                        float(value) for value in k_plus_q_unwrapped
                    ],
                    "paraunitary_residual": float(residual[ik, iq]),
                    "metric_positive_energy_minimum_mev": float(energy[ik, iq]),
                    "reasons": reasons,
                }
            )
        per_k.append(
            {
                "k_index": ik,
                "k_frac": [float(value) for value in k_values[ik]],
                "candidate_count": len(nodes),
                "candidates": nodes,
            }
        )

    omitted_nodes = []
    for iq in omitted_positions:
        diagnostics = []
        for ik in range(nk):
            reasons = []
            if candidate[ik, iq]:
                if para_defective[ik, iq]:
                    reasons.append("paraunitary_residual_above_tolerance")
                if zero_anisotropy_gapless:
                    reasons.append("zero_anisotropy_near_gapless_metric_energy")
            k_plus_q_unwrapped = k_values[ik] + q_values[iq]
            diagnostics.append(
                {
                    "k_index": ik,
                    "triggered_union_omission": bool(candidate[ik, iq]),
                    "k_plus_q_frac": [
                        float(value) for value in np.mod(k_plus_q_unwrapped, 1.0)
                    ],
                    "k_plus_q_frac_unwrapped": [
                        float(value) for value in k_plus_q_unwrapped
                    ],
                    "paraunitary_residual": float(residual[ik, iq]),
                    "metric_positive_energy_minimum_mev": float(energy[ik, iq]),
                    "reasons": reasons,
                }
            )
        omitted_nodes.append(
            {
                "selected_q_position": int(iq),
                "q_flat_index": int(flat[iq]),
                "q_frac": [float(value) for value in q_values[iq]],
                "per_k_diagnostics": diagnostics,
            }
        )

    report = {
        "policy": selected_policy,
        "goldstone_energy_tolerance_mev": energy_tol,
        "paraunitary_tolerance": para_tol,
        "metric_energy_tolerance_mev": metric_tol,
        "anisotropy_mev": anisotropy,
        "automatic_zero_anisotropy_identification_enabled": (zero_anisotropy_gapless),
        "n_qpoints_selected_before_omission": nq,
        "n_candidate_kq_pairs": int(np.count_nonzero(candidate)),
        "n_omitted": int(omitted_positions.size if selected_policy == "omit" else 0),
        "normalization_q_count": nq,
        "candidate_count_by_k": [int(np.count_nonzero(row)) for row in candidate],
        "per_k": per_k,
        "omitted_q_flat_indices": (
            [int(flat[iq]) for iq in omitted_positions]
            if selected_policy == "omit"
            else []
        ),
        "omitted_nodes": omitted_nodes if selected_policy == "omit" else [],
        "ward_asr_omission_prerequisites": prerequisite_report,
    }
    if omitted_positions.size and selected_policy == "fail":
        first_k, first_q = np.argwhere(candidate)[0]
        raise ValueError(
            "defective or near-exact Goldstone quadrature node detected under "
            "goldstone_node_policy=fail: "
            f"k_index={int(first_k)}, q_flat_index={int(flat[first_q])}, "
            f"q_frac={q_values[first_q].tolist()}, "
            f"metric-positive energy={energy[first_k, first_q]:.6e} meV, "
            f"paraunitary residual={residual[first_k, first_q]:.6e}. "
            "Use --goldstone-node-policy omit only when the recorded scalar-projection "
            "and dJ ASR/Ward prerequisites are satisfied."
        )
    if (
        omitted_positions.size
        and selected_policy == "omit"
        and not prerequisites_passed
    ):
        raise ValueError(
            "refusing Goldstone-node omission because its dJ Ward prerequisites "
            "failed: scalar_spin_group_projected, complete target-atom coverage, "
            "and a passing checked/projected dJ ASR are required; "
            f"report={prerequisite_report}"
        )
    keep = np.ones(nq, dtype=bool)
    if selected_policy == "omit":
        keep[omitted_positions] = False
        if not np.any(keep):
            raise ValueError("Goldstone-node omission removed every selected q point")
    return keep, report


def quadrature_chunk_weight(local_q_count: int, normalization_q_count: int) -> float:
    """Weight a chunk average by the original pre-omission quadrature count."""

    local = _positive_integer(local_q_count, "local_q_count")
    normalization = _positive_integer(normalization_q_count, "normalization_q_count")
    if local > normalization:
        raise ValueError("local_q_count cannot exceed normalization_q_count")
    return float(local) / float(normalization)


def _canonical_j_tuple(path: str) -> tuple[np.ndarray, ...]:
    raw = load_j_payload(path)
    result = (
        np.asarray(raw[0], dtype=np.float64).reshape(-1),
        np.asarray(raw[1], dtype=np.int32).reshape(-1),
        np.asarray(raw[2], dtype=np.int32).reshape(-1),
        np.asarray(raw[3], dtype=np.int32).reshape(-1, 3),
        np.asarray(raw[4], dtype=np.float64).reshape(-1),
    )
    lengths = [array.shape[0] for array in result]
    if not lengths or lengths[0] < 1 or len(set(lengths)) != 1:
        raise ValueError(f"invalid or empty static J payload lengths: {lengths}")
    return result


def _load_dynamic_payload(
    path: str,
    *,
    allow_unsafe_tensor_trace: bool,
    allow_unprojected_scalar_diagnostic: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load scalar dJ while refusing silent full-tensor trace fallback."""

    lower = str(path).lower()
    if lower.endswith((".h5", ".hdf5")):
        with h5py.File(path, "r") as handle:
            selected = next(
                (key for key in ("dJ_iso_r", "dJ_r", "dJ_tensor_r") if key in handle),
                None,
            )
            if selected is None:
                raise KeyError("dynamic HDF5 has none of dJ_iso_r, dJ_r, dJ_tensor_r")
            projected = bool(int(handle.attrs.get("scalar_spin_group_projected", 0)))
            explicit_unit, unit_source = _h5_explicit_unit(
                handle, ("basic_data/unit", "basic_data/units")
            )
            provenance = {
                "format": "hdf5",
                "selected_dataset": selected,
                "tensor_trace_fallback": selected == "dJ_tensor_r",
                "scalar_spin_group_projected": projected,
                "projection_schema": _decode_h5_scalar(
                    handle.attrs.get("scalar_spin_group_projection_schema", "")
                ),
                "projection_source": _decode_h5_scalar(
                    handle.attrs.get("scalar_spin_group_projection_source", "")
                ),
                "projection_epr_h5": _decode_h5_scalar(
                    handle.attrs.get("scalar_spin_group_projection_epr_h5", "")
                ),
                "unit": explicit_unit,
                "unit_explicit": unit_source is not None,
                "unit_source": unit_source,
            }
        if selected == "dJ_tensor_r" and not allow_unsafe_tensor_trace:
            raise ValueError(
                "--djr has no scalar dJ_iso_r/dJ_r dataset; load_djr_h5 would "
                "silently trace the full dJ_tensor_r, which is forbidden in this "
                "workflow. Use --allow-unsafe-tensor-trace only for a labelled diagnostic."
            )
        if not projected and not allow_unprojected_scalar_diagnostic:
            raise ValueError(
                "--djr is not marked scalar_spin_group_projected. Use the "
                "non-destructive scalar spin-group projected file, or pass "
                "--allow-unprojected-scalar-diagnostic for a labelled diagnostic."
            )
        payload = load_djr_h5(path)
        if "units_explicit" in payload:
            provenance["unit_explicit"] = bool(payload["units_explicit"])
        provenance["unit_source"] = str(
            payload.get("units_source", provenance.get("unit_source") or "missing")
        )
        if provenance["unit_explicit"]:
            provenance["unit"] = _scalar_text(payload.get("units", provenance["unit"]))
        return payload, provenance
    if lower.endswith(".npz"):
        with np.load(path, allow_pickle=False) as handle:
            projected = (
                bool(
                    np.asarray(handle["scalar_spin_group_projected"]).reshape(()).item()
                )
                if "scalar_spin_group_projected" in handle
                else False
            )
            explicit_unit = _scalar_text(handle["units"]) if "units" in handle else None
            unit_explicit = (
                bool(np.asarray(handle["units_explicit"]).reshape(()).item())
                if "units_explicit" in handle
                else explicit_unit is not None
            )
        if not projected and not allow_unprojected_scalar_diagnostic:
            raise ValueError(
                "NPZ --djr lacks scalar_spin_group_projected provenance; pass "
                "--allow-unprojected-scalar-diagnostic only for a labelled diagnostic"
            )
        payload = load_djdu_npz(path)
        return payload, {
            "format": "npz",
            "selected_dataset": "values_rp",
            "tensor_trace_fallback": False,
            "scalar_spin_group_projected": projected,
            "unit": explicit_unit,
            "unit_explicit": unit_explicit,
            "unit_source": "units" if explicit_unit is not None else None,
            "projection_epr_h5": "",
        }
    raise ValueError("--djr must be a real-space dJ HDF5 or NPZ payload")


def _apply_shell_filter(
    j_tuple: tuple[np.ndarray, ...],
    dj_payload: dict[str, Any],
    *,
    exclude_shells: Sequence[int],
    exclude_shell_apply: str,
    shell_tolerance: float,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any], dict[str, Any]]:
    pair = np.asarray(dj_payload["pair_R"], dtype=np.int32).reshape(-1, 5)
    # filter_exchange_shells needs only dynamic bond keys.  One synthetic tuple
    # entry per pair avoids choosing or hardcoding a particular Rp cell.
    dynamic_keys = (
        np.zeros(pair.shape[0], dtype=np.int32),
        pair[:, 0],
        pair[:, 1],
        pair[:, 2:5],
        np.zeros((pair.shape[0], 3), dtype=np.float64),
    )
    filtered_j, filtered_dynamic, report = filter_exchange_shells(
        j_tuple,
        dynamic_keys,
        {
            "exclude_shells": [int(value) for value in exclude_shells],
            "exclude_shell_apply": str(exclude_shell_apply),
            "shell_tol": float(shell_tolerance),
        },
    )
    if report.get("enabled", False) and report.get("apply") in {"both", "dJ"}:
        allowed = {
            (int(i), int(j), tuple(int(x) for x in r))
            for i, j, r in zip(
                filtered_dynamic[1], filtered_dynamic[2], filtered_dynamic[3]
            )
        }
        dj_payload = filter_dj_payload_bonds(dj_payload, allowed)
    return filtered_j, dj_payload, report


def _relative_reconstruction_error(actual: np.ndarray, reference: np.ndarray) -> float:
    difference = np.asarray(actual) - np.asarray(reference)
    return float(
        np.linalg.norm(difference.ravel())
        / max(np.linalg.norm(np.asarray(reference).ravel()), np.finfo(float).tiny)
    )


def _selectivity_contrast(bins: np.ndarray) -> np.ndarray:
    """Return (positive-negative)/(positive+negative) for nonnegative bins."""

    values = np.asarray(bins, dtype=np.float64)
    denominator = values[..., 2] + values[..., 0]
    return np.divide(
        values[..., 2] - values[..., 0],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )


def _signed_selectivity_contrast(bins: np.ndarray) -> np.ndarray:
    """Bounded contrast for possibly signed phase-space contributions."""

    values = np.asarray(bins, dtype=np.float64)
    denominator = np.abs(values[..., 2]) + np.abs(values[..., 0])
    return np.divide(
        values[..., 2] - values[..., 0],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )


def analyze_rotational_coupling(
    *,
    jr_path: str,
    djr_path: str,
    phonon_cache_path: str,
    axis: Sequence[float],
    spin_pattern: str,
    k_points_frac: Sequence[Sequence[float]],
    temperature_k: float,
    eta_mev: float,
    spin_S: float,
    bond_factor: float,
    anisotropy_mev: float,
    q_stride: Sequence[int] = (1, 1, 1),
    exclude_shells: Sequence[int] = (),
    exclude_shell_apply: str = "both",
    shell_tolerance: float = 1.0e-4,
    degeneracy_tolerance_mev: float | None = 1.0e-8,
    chiralization_atom_weights: Sequence[float] | None = None,
    chiralization_energy_offdiag_tolerance_mev: float = STRICT_CHIRAL_ENERGY_OFFDIAG_MEV,
    normalization_tolerance: float = 1.0e-7,
    geometry_tolerance: float = 1.0e-8,
    negative_phonon_tolerance_mev: float = 1.0e-6,
    orthonormal_tolerance: float = 1.0e-7,
    angular_zero_tolerance: float = 1.0e-10,
    lte_gamma_histogram_bins: int = 0,
    lte_gamma_histogram_max_abs_over_hbar: float | None = None,
    phonon_floor_mev: float = 1.0e-3,
    dj_asr_mode: str = "check",
    dj_asr_tolerance: float = 1.0e-8,
    reconstruction_tolerance: float = 1.0e-9,
    q_chunk: int = 256,
    bond_chunk: int = 64,
    save_vertices: bool = False,
    max_saved_vertices_gb: float = 2.0,
    paraunitary_tolerance: float = 1.0e-7,
    metric_energy_tolerance_mev: float = 1.0e-8,
    goldstone_node_policy: str = "fail",
    goldstone_energy_tolerance_mev: float = 1.0e-8,
    allow_active_static_soc_diagnostic: bool = False,
    allow_unsafe_tensor_trace: bool = False,
    allow_unprojected_scalar_diagnostic: bool = False,
    allow_missing_units_diagnostic: bool = False,
    allow_geometry_mismatch_diagnostic: bool = False,
    geometry_epr_path: str | None = None,
    allow_unstable_phonon_diagnostic: bool = False,
    allow_asr_none_diagnostic: bool = False,
    allow_duplicate_bonds_diagnostic: bool = False,
) -> RotationalCouplingResult:
    """Run a chunked rotational coupling and linewidth pilot calculation."""

    started = time.perf_counter()
    temperature = _finite_nonnegative(temperature_k, "temperature_k")
    broadening = _finite_positive(eta_mev, "eta_mev")
    spin_length = _finite_positive(spin_S, "spin_S")
    bond_scale = _finite(bond_factor, "bond_factor")
    anisotropy = _finite(anisotropy_mev, "anisotropy_mev")
    shell_tol = _finite_nonnegative(shell_tolerance, "shell_tolerance")
    negative_phonon_tol = _finite_nonnegative(
        negative_phonon_tolerance_mev, "negative_phonon_tolerance_mev"
    )
    overlap_tol = _finite_nonnegative(orthonormal_tolerance, "orthonormal_tolerance")
    chiral_energy_tol = _finite_nonnegative(
        chiralization_energy_offdiag_tolerance_mev,
        "chiralization_energy_offdiag_tolerance_mev",
    )
    recon_tol = _finite_nonnegative(
        reconstruction_tolerance, "reconstruction_tolerance"
    )
    zero_tol = _finite_nonnegative(angular_zero_tolerance, "angular_zero_tolerance")
    lte_histogram_bin_count = int(lte_gamma_histogram_bins)
    if lte_histogram_bin_count < 0:
        raise ValueError("lte_gamma_histogram_bins must be non-negative")
    floor = _finite_positive(phonon_floor_mev, "phonon_floor_mev")
    asr_tol = _finite_nonnegative(dj_asr_tolerance, "dj_asr_tolerance")
    chunk_size = _positive_integer(q_chunk, "q_chunk")
    bond_chunk_size = _positive_integer(bond_chunk, "bond_chunk")
    saved_vertex_limit_gb = _finite_positive(
        max_saved_vertices_gb, "max_saved_vertices_gb"
    )
    para_tol = _finite_nonnegative(paraunitary_tolerance, "paraunitary_tolerance")
    metric_energy_tol = _finite_nonnegative(
        metric_energy_tolerance_mev, "metric_energy_tolerance_mev"
    )
    goldstone_energy_tol = _finite_nonnegative(
        goldstone_energy_tolerance_mev, "goldstone_energy_tolerance_mev"
    )
    normalized_goldstone_policy = str(goldstone_node_policy).strip().lower()
    if normalized_goldstone_policy not in {"fail", "omit"}:
        raise ValueError("goldstone_node_policy must be one of: fail, omit")
    normalized_asr_mode = {
        "off": "none",
        "false": "none",
        "on": "project",
        "enforce": "project",
    }.get(str(dj_asr_mode).strip().lower(), str(dj_asr_mode).strip().lower())
    if normalized_asr_mode == "none" and not allow_asr_none_diagnostic:
        raise ValueError(
            "dJ ASR must be checked or projected for a production linewidth; "
            "use --allow-asr-none-diagnostic only for a labelled diagnostic"
        )

    cache = load_phonon_cache(phonon_cache_path)
    validated, geometry = validate_coupling_cache_geometry(
        cache,
        q_stride,
        normalization_tolerance=normalization_tolerance,
        geometry_tolerance=geometry_tolerance,
    )
    axis_vector = helicity_projectors(np.asarray(axis, dtype=np.float64)).axis_vector
    q_indices = geometry.q_indices
    q_grid_indices = np.array(geometry.selected_q_grid_indices, copy=True)
    q_frac = validated.q_frac[q_indices]
    q_cart = validated.q_cart[q_indices]
    phonon_energies = np.array(validated.energies_mev[q_indices], copy=True)
    phonon_vectors = np.array(validated.displacements[q_indices], copy=True)
    nat = validated.n_atoms
    nmode = validated.n_modes

    if chiralization_atom_weights is None:
        operator_weights = None
    else:
        operator_weights = np.asarray(chiralization_atom_weights, dtype=np.float64)
        if operator_weights.shape != (nat,):
            raise ValueError(
                f"chiralization_atom_weights must have one value per atom: "
                f"expected {(nat,)}, got {operator_weights.shape}"
            )
        if degeneracy_tolerance_mev is None:
            raise ValueError(
                "chiralization_atom_weights requires degeneracy_tolerance_mev"
            )
    if degeneracy_tolerance_mev is None:
        raise ValueError(
            "sign-resolved phonon-helicity selectivity requires degenerate-subspace "
            "chiralization; provide --degeneracy-tol-mev"
        )
    chiral_summary: dict[str, Any] = {
        "enabled": degeneracy_tolerance_mev is not None,
        "operator_atom_weights": None
        if operator_weights is None
        else [float(value) for value in operator_weights],
    }
    chiral_energy_offdiagonal = 0.0
    if degeneracy_tolerance_mev is not None:
        degeneracy_tol = _finite_nonnegative(
            degeneracy_tolerance_mev, "degeneracy_tolerance_mev"
        )
        chiralized = chiralize_degenerate_subspaces(
            phonon_energies,
            phonon_vectors,
            validated.masses,
            axis_vector,
            atom_weights=operator_weights,
            abs_tolerance=degeneracy_tol,
            rel_tolerance=0.0,
            orthonormal_tolerance=overlap_tol,
        )
        phonon_energies = chiralized.energies
        phonon_vectors = chiralized.displacements
        chiral_summary.update(chiralized.diagnostics.summary())
        energy_offdiagonal = float(chiralized.diagnostics.max_energy_offdiagonal_after)
        chiral_energy_offdiagonal = energy_offdiagonal
        chiral_summary["coupling_energy_offdiagonal_tolerance_mev"] = chiral_energy_tol
        if energy_offdiagonal > chiral_energy_tol:
            raise ValueError(
                "degenerate-subspace chiralization mixed nondegenerate phonon "
                "energies but the coupling kernel keeps only diagonal phonon "
                f"energies: max offdiagonal={energy_offdiagonal:.6e} meV exceeds "
                f"tolerance={chiral_energy_tol:.6e} meV. Tighten "
                "--degeneracy-tol-mev or explicitly relax "
                "--chiralization-energy-offdiag-tol-mev."
            )

    minimum_phonon_energy = float(np.min(phonon_energies))
    negative_mask = phonon_energies < 0.0
    unstable_mask = phonon_energies < -negative_phonon_tol
    negative_phonon_count = int(np.count_nonzero(negative_mask))
    unstable_phonon_count = int(np.count_nonzero(unstable_mask))
    if unstable_phonon_count and not allow_unstable_phonon_diagnostic:
        raise ValueError(
            "phonon cache contains frequencies below the allowed acoustic-noise "
            f"floor: count={unstable_phonon_count}, min={minimum_phonon_energy:.6e} "
            f"meV, tolerance={negative_phonon_tol:.6e} meV. Use "
            "--allow-unstable-phonon-diagnostic only for a labelled diagnostic."
        )
    coupling_phonon_energies = np.maximum(phonon_energies, 0.0)
    decomposition = decompose_atomistic_rotation(
        phonon_vectors, validated.masses, axis_vector
    )
    atom_angular_axis = np.einsum(
        "qmai,i->qma",
        decomposition.atom_angular_momentum_over_hbar,
        axis_vector,
        optimize=True,
    )
    selection_weights = (
        np.ones(nat, dtype=np.float64) if operator_weights is None else operator_weights
    )
    selection_angular = np.einsum(
        "a,qma->qm", selection_weights, atom_angular_axis, optimize=True
    )
    selection_operator_kind = (
        "total_angular_momentum"
        if operator_weights is None
        else "user_weighted_local_angular_momentum"
    )

    static_soc = read_static_j_soc_metadata(jr_path)
    static_provenance = inspect_static_j_provenance(jr_path)
    if static_soc["active"] is True and not allow_active_static_soc_diagnostic:
        raise ValueError(
            "active SOC metadata was detected in static J. An exact SOC-off "
            "spin-group interpretation is forbidden; pass "
            "--allow-active-static-soc-diagnostic only for an explicitly "
            "labelled scalar-trace diagnostic"
        )
    j_tuple = _canonical_j_tuple(jr_path)
    dynamic_payload, dynamic_provenance = _load_dynamic_payload(
        djr_path,
        allow_unsafe_tensor_trace=allow_unsafe_tensor_trace,
        allow_unprojected_scalar_diagnostic=allow_unprojected_scalar_diagnostic,
    )
    unit_report = validate_coupling_units(
        mass_unit=cache["mass_unit"],
        static_exchange_unit=static_provenance.get("unit"),
        static_unit_explicit=bool(static_provenance.get("unit_explicit", False)),
        dynamic_derivative_unit=dynamic_provenance.get("unit"),
        dynamic_unit_explicit=bool(dynamic_provenance.get("unit_explicit", False)),
        allow_missing_explicit_units_diagnostic=allow_missing_units_diagnostic,
    )
    static_geometry_report = compare_geometry_to_cache(
        _extract_h5_geometry(jr_path)
        if str(jr_path).lower().endswith((".h5", ".hdf5"))
        else {"available": False, "source_path": str(jr_path)},
        geometry,
        n_cache_atoms=nat,
        tolerance=geometry_tolerance,
        label="static_J",
    )
    dynamic_geometry_source = inspect_dynamic_geometry(
        djr_path,
        dynamic_provenance,
        geometry_epr_path=geometry_epr_path,
    )
    dynamic_geometry_report = compare_geometry_to_cache(
        dynamic_geometry_source,
        geometry,
        n_cache_atoms=nat,
        tolerance=geometry_tolerance,
        label="dynamic_dJ",
    )
    cache_epr_source = (
        _scalar_text(cache["epr_source"]) if "epr_source" in cache else None
    )
    recorded_projected_epr_source = dynamic_provenance.get("projection_epr_h5") or None
    explicit_geometry_epr = (
        str(Path(geometry_epr_path).expanduser().resolve(strict=True))
        if geometry_epr_path is not None
        else None
    )
    projected_epr_source = explicit_geometry_epr or recorded_projected_epr_source
    if cache_epr_source and projected_epr_source:
        epr_source_matches = os.path.realpath(cache_epr_source) == os.path.realpath(
            str(projected_epr_source)
        )
    else:
        epr_source_matches = None
    dynamic_geometry_report["cache_epr_source"] = cache_epr_source
    dynamic_geometry_report["projection_epr_source"] = projected_epr_source
    dynamic_geometry_report["recorded_projection_epr_source"] = (
        recorded_projected_epr_source
    )
    dynamic_geometry_report["explicit_geometry_epr_override"] = explicit_geometry_epr
    dynamic_geometry_report["epr_source_paths_match"] = epr_source_matches
    dynamic_geometry_report["path_mismatch_accepted_by_explicit_geometry_override"] = (
        bool(epr_source_matches is False and explicit_geometry_epr is not None)
    )
    if epr_source_matches is False and explicit_geometry_epr is None:
        dynamic_geometry_report["valid"] = False
        dynamic_geometry_report["issues"].append("epr_source_path_mismatch")
    geometry_reports = {
        "static_J_vs_cache": static_geometry_report,
        "dynamic_dJ_vs_cache": dynamic_geometry_report,
    }
    invalid_geometry = [
        key for key, report in geometry_reports.items() if not report["valid"]
    ]
    if invalid_geometry and not allow_geometry_mismatch_diagnostic:
        raise ValueError(
            "exchange/phonon geometry provenance is unavailable or mismatched: "
            f"{invalid_geometry}. Use --allow-geometry-mismatch-diagnostic only "
            "for a labelled diagnostic."
        )
    j_tuple, dynamic_payload, shell_report = _apply_shell_filter(
        j_tuple,
        dynamic_payload,
        exclude_shells=exclude_shells,
        exclude_shell_apply=exclude_shell_apply,
        shell_tolerance=shell_tol,
    )
    bond_key_report = compare_static_dynamic_bond_keys(j_tuple, dynamic_payload)
    if bond_key_report["dynamic_only_count"]:
        raise ValueError(
            "dynamic dJ contains directed bond keys absent from the post-filter "
            "static J Hamiltonian; the vertex is not its derivative. "
            f"dynamic-only count={bond_key_report['dynamic_only_count']}"
        )
    duplicate_or_mismatch = bool(
        bond_key_report["static_duplicate_extra_count"]
        or bond_key_report["dynamic_duplicate_extra_count"]
        or bond_key_report["multiplicity_mismatch_count"]
    )
    if duplicate_or_mismatch and not allow_duplicate_bonds_diagnostic:
        raise ValueError(
            "duplicate directed bond keys or J/dJ multiplicity mismatch detected; "
            "the static kernel and its derivative are ambiguous. Use "
            "--allow-duplicate-bonds-diagnostic only for a labelled diagnostic."
        )
    _, goldstone_asr_object = apply_exchange_derivative_asr(
        dynamic_payload["values_rp"],
        mode=normalized_asr_mode,
        tolerance=asr_tol,
    )
    goldstone_asr_preflight = validate_asr_report(
        goldstone_asr_object.as_dict(), requested_mode=normalized_asr_mode
    )
    dynamic_targets = target_atom_indices(dynamic_payload)
    target_coverage_complete = bool(
        dynamic_targets.size == nat
        and np.array_equal(np.sort(np.unique(dynamic_targets)), np.arange(nat))
    )
    goldstone_ward_prerequisites = {
        "scalar_spin_group_projected": bool(
            dynamic_provenance["scalar_spin_group_projected"]
        ),
        "dj_asr_mode": normalized_asr_mode,
        "dj_asr_checked_or_projected": normalized_asr_mode in {"check", "project"},
        "dj_asr_after_within_tolerance": bool(
            goldstone_asr_preflight["max_abs_after"]
            <= goldstone_asr_preflight["tolerance"]
            and not goldstone_asr_preflight["violated_after"]
        ),
        "dj_asr_report": goldstone_asr_preflight,
        "target_atom_coverage_complete": target_coverage_complete,
        "target_atom_indices": [int(value) for value in dynamic_targets],
    }
    goldstone_ward_prerequisites["passed"] = bool(
        goldstone_ward_prerequisites["scalar_spin_group_projected"]
        and goldstone_ward_prerequisites["dj_asr_checked_or_projected"]
        and goldstone_ward_prerequisites["dj_asr_after_within_tolerance"]
        and goldstone_ward_prerequisites["target_atom_coverage_complete"]
    )
    nmag = int(max(np.max(j_tuple[1]), np.max(j_tuple[2])) + 1)
    if nmag != 2:
        raise ValueError(
            "the chirality-resolved Magnon_Hamiltonian_v1 analysis requires "
            f"exactly two magnetic sublattices, found {nmag}"
        )
    if int(max(np.max(j_tuple[1]), np.max(j_tuple[2]))) >= nat:
        raise ValueError("static J atom index exceeds phonon-cache atom count")
    spin = parse_spin_pattern(spin_pattern, nmag)
    if spin[0] * spin[1] >= 0.0:
        raise ValueError(
            "chirality-resolved Magnon_Hamiltonian_v1 requires an antiparallel "
            "two-sublattice spin pattern, e.g. 1,-1 or -1,1"
        )
    physical_channel_chirality = np.asarray(-spin, dtype=np.int8)
    bdg_sector_chirality = physical_channel_chirality[
        np.asarray([0, 1, 1, 0], dtype=np.int8)
    ]

    k_frac = np.asarray(k_points_frac, dtype=np.float64)
    if k_frac.ndim != 2 or k_frac.shape[0] < 1 or k_frac.shape[1] != 3:
        raise ValueError(
            f"k_points_frac must have nonempty shape (nk,3), got {k_frac.shape}"
        )
    if not np.all(np.isfinite(k_frac)):
        raise ValueError("k_points_frac contains non-finite values")
    k_cart = k_frac @ geometry.reciprocal_ang_inv
    nk = k_frac.shape[0]
    nchannel = 2 * nmag
    metric = bosonic_metric_diag(nchannel, physical_count=nmag)
    external_energies = np.empty((nk, nchannel), dtype=np.float64)
    initial_modes = np.empty((nk, nchannel, nchannel), dtype=np.complex128)
    paraunitary_error = np.empty(nk, dtype=np.float64)
    for ik in range(nk):
        energies_k, transform_k = Magnon_Hamiltonian_v1(
            k_cart[ik],
            spin_length,
            j_tuple,
            geometry.lattice_ang,
            geometry.atom_cart_ang,
            spin,
            bond_scale,
            anisotropy,
        )
        external_energies[ik] = energies_k
        initial_modes[ik] = transform_k
        paraunitary_error[ik] = _max_abs(paraunitary_residual(transform_k, metric))
    if not (
        np.all(np.isfinite(external_energies))
        and np.all(np.isfinite(initial_modes))
        and np.all(np.isfinite(paraunitary_error))
    ):
        raise ValueError("external magnon BdG eigensystem contains non-finite values")
    external_metric_energy_minimum = np.min(external_energies * metric[None, :], axis=1)
    if _max_abs(paraunitary_error) > para_tol:
        raise ValueError(
            "external magnon BdG transform violates paraunitarity: "
            f"max residual={_max_abs(paraunitary_error):.6e}, "
            f"tolerance={para_tol:.6e}"
        )
    if float(np.min(external_metric_energy_minimum)) < -metric_energy_tol:
        raise ValueError(
            "external magnon spectrum violates positive metric energy: "
            f"minimum={float(np.min(external_metric_energy_minimum)):.6e} meV"
        )

    normalization_q_count = int(q_indices.size)
    internal_paraunitary_residual_by_kq = np.empty(
        (nk, normalization_q_count), dtype=np.float64
    )
    internal_metric_energy_by_kq = np.empty(
        (nk, normalization_q_count), dtype=np.float64
    )
    for ik in range(nk):
        q_residual, q_metric_energy = _magnon_q_metric_diagnostics(
            q_cart,
            k_cart[ik],
            spin_length,
            j_tuple,
            geometry.lattice_ang,
            geometry.atom_cart_ang,
            spin,
            bond_scale,
            anisotropy,
            metric,
        )
        internal_paraunitary_residual_by_kq[ik] = q_residual
        internal_metric_energy_by_kq[ik] = q_metric_energy
    if not (
        np.all(np.isfinite(internal_paraunitary_residual_by_kq))
        and np.all(np.isfinite(internal_metric_energy_by_kq))
    ):
        raise ValueError(
            "internal k+q magnon BdG diagnostics contain non-finite values"
        )
    internal_paraunitary_error_before_omission = np.max(
        internal_paraunitary_residual_by_kq, axis=1
    )
    internal_metric_energy_minimum_before_omission = np.min(
        internal_metric_energy_by_kq, axis=1
    )
    if (
        float(np.min(internal_metric_energy_minimum_before_omission))
        < -metric_energy_tol
    ):
        raise ValueError(
            "internal k+q magnon spectrum violates positive metric energy: "
            f"minimum={float(np.min(internal_metric_energy_minimum_before_omission)):.6e} "
            f"meV, tolerance={metric_energy_tol:.6e} meV"
        )

    goldstone_keep, goldstone_report = resolve_goldstone_quadrature_nodes(
        internal_paraunitary_residual_by_kq,
        internal_metric_energy_by_kq,
        q_indices,
        q_frac,
        k_frac,
        anisotropy_mev=anisotropy,
        paraunitary_tolerance=para_tol,
        metric_energy_tolerance_mev=metric_energy_tol,
        goldstone_energy_tolerance_mev=goldstone_energy_tol,
        policy=normalized_goldstone_policy,
        omission_prerequisites=goldstone_ward_prerequisites,
    )
    for per_k_report in goldstone_report["per_k"]:
        for node in per_k_report["candidates"]:
            node["q_grid_index"] = [
                int(value) for value in q_grid_indices[node["selected_q_position"]]
            ]
    for node in goldstone_report["omitted_nodes"]:
        node["q_grid_index"] = [
            int(value) for value in q_grid_indices[node["selected_q_position"]]
        ]
    omitted_goldstone_q_flat_indices = np.asarray(
        goldstone_report["omitted_q_flat_indices"], dtype=np.int64
    )
    omitted_goldstone_q_frac = np.asarray(
        [node["q_frac"] for node in goldstone_report["omitted_nodes"]],
        dtype=np.float64,
    ).reshape(-1, 3)
    omitted_goldstone_q_grid_indices = q_grid_indices[~goldstone_keep]
    q_indices = q_indices[goldstone_keep]
    q_grid_indices = q_grid_indices[goldstone_keep]
    q_frac = q_frac[goldstone_keep]
    q_cart = q_cart[goldstone_keep]
    phonon_energies = phonon_energies[goldstone_keep]
    coupling_phonon_energies = coupling_phonon_energies[goldstone_keep]
    phonon_vectors = phonon_vectors[goldstone_keep]
    atom_angular_axis = atom_angular_axis[goldstone_keep]
    selection_angular = selection_angular[goldstone_keep]
    lte_histogram_edges = physical_lte_histogram_edges(
        selection_angular,
        lte_histogram_bin_count,
        maximum_abs_over_hbar=lte_gamma_histogram_max_abs_over_hbar,
        zero_tolerance=zero_tol,
    )
    lte_histogram_centers = (
        0.5 * (lte_histogram_edges[:-1] + lte_histogram_edges[1:])
        if lte_histogram_edges.size
        else np.empty(0, dtype=np.float64)
    )
    lte_histogram_mode_counts = (
        np.stack(
            (
                np.histogram(-selection_angular.ravel(), bins=lte_histogram_edges)[0],
                np.histogram(selection_angular.ravel(), bins=lte_histogram_edges)[0],
            ),
            axis=0,
        ).astype(np.int64, copy=False)
        if lte_histogram_edges.size
        else np.empty((2, 0), dtype=np.int64)
    )
    kept_paraunitary_residual = internal_paraunitary_residual_by_kq[:, goldstone_keep]
    kept_metric_energy = internal_metric_energy_by_kq[:, goldstone_keep]
    internal_paraunitary_error = np.max(kept_paraunitary_residual, axis=1)
    internal_metric_energy_minimum = np.min(kept_metric_energy, axis=1)
    total_q = int(q_indices.size)
    goldstone_report["n_qpoints_used"] = total_q
    if _max_abs(internal_paraunitary_error) > para_tol:
        raise ValueError(
            "internal k+q magnon BdG transform violates paraunitarity: "
            f"max residual={_max_abs(internal_paraunitary_error):.6e}, "
            f"tolerance={para_tol:.6e}"
        )

    projectors = build_atom_helicity_projectors(nat, axis_vector)
    nhelicity = len(COMPONENT_LABELS)
    ncomp = nat * nhelicity
    pair = np.asarray(dynamic_payload["pair_R"], dtype=np.int32).reshape(-1, 5)
    saved_vertex_memory = estimate_saved_vertex_memory_bytes(
        n_components=ncomp,
        n_qpoints=total_q,
        n_modes=nmode,
        n_bonds=pair.shape[0],
        n_kpoints=nk,
        n_channels=nchannel,
    )
    saved_vertex_payload_bytes = saved_vertex_memory["combined_payload_bytes"]
    saved_vertex_peak_gb = saved_vertex_memory["estimated_peak_bytes"] / 1.0e9
    if save_vertices and saved_vertex_peak_gb > saved_vertex_limit_gb:
        raise MemoryError(
            "refusing --save-vertices because estimated peak memory "
            f"{saved_vertex_peak_gb:.3f} GB exceeds --max-saved-vertices-gb "
            f"{saved_vertex_limit_gb:.3f} GB; increase q stride or the explicit limit"
        )
    component_gamma = np.zeros((nk, 2, ncomp, ncomp, nchannel), dtype=np.float64)
    component_vertex_gram = np.zeros((nk, ncomp, ncomp, nmag), dtype=np.float64)
    physical_transition_intensity = np.zeros((nk, nmag, nmag), dtype=np.float64)
    full_gamma = np.zeros((nk, nchannel), dtype=np.float64)
    process_gamma = np.zeros((nk, 2, nchannel), dtype=np.float64)
    vertex_bins = np.zeros((nk, 2, nmag, 3), dtype=np.float64)
    transition_bins = np.zeros((nk, 2, nmag, nmag, 3), dtype=np.float64)
    phase_bins = np.zeros((nk, 2, nmag, 3), dtype=np.float64)
    lte_gamma_histogram = np.zeros(
        (nk, 2, nmag, lte_histogram_centers.size), dtype=np.float64
    )
    selectivity_counts = np.zeros((nk, 2, nmag, 3), dtype=np.int64)

    saved_lambda_chunks: list[np.ndarray] = []
    saved_g_chunks: list[list[np.ndarray]] = [[] for _ in range(nk)]
    lambda_reconstruction_max_abs = 0.0
    lambda_reconstruction_relative_l2 = 0.0
    linewidth_reconstruction_max_abs = 0.0
    linewidth_reconstruction_relative_l2 = 0.0
    asr_report: dict[str, Any] | None = None

    for start in range(0, total_q, chunk_size):
        stop = min(start + chunk_size, total_q)
        local_count = stop - start
        lambda_components, lambda_report = build_atomic_gauge_lambda_components(
            dynamic_payload,
            q_frac[start:stop],
            coupling_phonon_energies[start:stop],
            phonon_vectors[start:stop],
            geometry.atom_frac,
            projectors,
            phonon_floor_mev=floor,
            asr_mode=normalized_asr_mode,
            asr_tolerance=asr_tol,
            reconstruction_tolerance=recon_tol,
            q_chunk=local_count,
            bond_chunk=bond_chunk_size,
        )
        local_component_report = lambda_report["component_decomposition"]
        if not local_component_report["lambda_reconstructs_full"]:
            raise RuntimeError(
                "atom-helicity Lambda components do not reconstruct the identity vertex"
            )
        lambda_reconstruction_max_abs = max(
            lambda_reconstruction_max_abs,
            float(local_component_report["lambda_sum_minus_full_max_abs"]),
        )
        lambda_reconstruction_relative_l2 = max(
            lambda_reconstruction_relative_l2,
            float(local_component_report["lambda_sum_minus_full_relative_l2"]),
        )
        asr_report = validate_asr_report(
            lambda_report["asr"], requested_mode=normalized_asr_mode
        )
        if save_vertices:
            saved_lambda_chunks.append(np.array(lambda_components, copy=True))

        for ik in range(nk):
            g_components, internal_energies = g_kq_loop_atomic_gauge_components(
                q_cart[start:stop],
                k_cart[ik],
                lambda_components,
                pair,
                initial_modes[ik],
                spin_length,
                j_tuple,
                geometry.lattice_ang,
                geometry.atom_cart_ang,
                spin,
                anisotropy,
                bond_scale,
            )
            full_vertex = np.sum(g_components, axis=0)
            resolved_raw = compute_linewidth_components_vectorized(
                g_components,
                internal_energies,
                coupling_phonon_energies[start:stop],
                external_energies[ik],
                temperature,
                broadening,
                local_count,
                metric,
            )
            full_raw = compute_linewidth_vectorized(
                full_vertex,
                internal_energies,
                coupling_phonon_energies[start:stop],
                external_energies[ik],
                temperature,
                broadening,
                local_count,
                metric,
            )
            reconstructed_raw = np.sum(resolved_raw, axis=(0, 1, 2))
            chunk_max = _max_abs(reconstructed_raw - full_raw)
            chunk_relative = _relative_reconstruction_error(reconstructed_raw, full_raw)
            linewidth_reconstruction_max_abs = max(
                linewidth_reconstruction_max_abs, chunk_max
            )
            linewidth_reconstruction_relative_l2 = max(
                linewidth_reconstruction_relative_l2, chunk_relative
            )
            scale = max(_max_abs(full_raw), 1.0)
            if chunk_max > recon_tol * scale and chunk_relative > recon_tol:
                raise RuntimeError(
                    "component linewidth matrix failed to reconstruct the full vertex "
                    f"linewidth at k index {ik}, q chunk {start}:{stop}"
                )

            q_weight = quadrature_chunk_weight(local_count, normalization_q_count)
            component_gamma[ik] += -resolved_raw * q_weight
            process_gamma[ik] += -np.sum(resolved_raw, axis=(1, 2)) * q_weight
            full_gamma[ik] += -full_raw * q_weight
            component_vertex_gram[ik] += component_vertex_intensity_matrix(
                g_components,
                physical_count=nmag,
                normalization_q_count=normalization_q_count,
            )
            physical_transition_intensity[ik] += np.sum(
                np.abs(full_vertex[:, :, :nmag, :nmag]) ** 2,
                axis=(0, 1),
            ) / float(normalization_q_count)
            selection = compute_vertex_phase_space_selectivity(
                full_vertex,
                internal_energies,
                coupling_phonon_energies[start:stop],
                external_energies[ik],
                selection_angular[start:stop],
                temperature_k=temperature,
                eta_mev=broadening,
                metric_diag=metric,
                physical_channel_chirality=physical_channel_chirality,
                zero_tolerance=zero_tol,
                normalization_q_count=normalization_q_count,
                physical_lte_histogram_edges_over_hbar=(
                    lte_histogram_edges if lte_histogram_edges.size else None
                ),
            )
            vertex_bins[ik] += selection["vertex_intensity_mev2"]
            transition_bins[ik] += selection["vertex_transition_intensity_mev2"]
            phase_bins[ik] += selection["phase_space_gamma_mev"]
            if lte_histogram_edges.size:
                lte_gamma_histogram[ik] += selection[
                    "physical_lte_gamma_histogram_mev"
                ]
            selectivity_counts[ik] += selection["mode_counts"]
            if save_vertices:
                saved_g_chunks[ik].append(np.array(g_components, copy=True))

    matrix_total = np.sum(component_gamma, axis=(1, 2, 3))
    gamma_error = _max_abs(matrix_total - full_gamma)
    gamma_relative = _relative_reconstruction_error(matrix_total, full_gamma)
    gamma_scale = max(_max_abs(full_gamma), 1.0)
    if gamma_error > recon_tol * gamma_scale and gamma_relative > recon_tol:
        raise RuntimeError(
            "final atom-component linewidth matrix reconstruction failed"
        )
    process_error = _max_abs(np.sum(process_gamma, axis=1) - full_gamma)
    if process_error > recon_tol * gamma_scale:
        raise RuntimeError(
            "emission plus absorption does not reconstruct total linewidth"
        )
    phase_total = np.sum(phase_bins, axis=(1, 3))
    phase_reference = full_gamma[:, :nmag]
    phase_error = _max_abs(phase_total - phase_reference)
    phase_relative = _relative_reconstruction_error(phase_total, phase_reference)
    if phase_error > recon_tol * gamma_scale and phase_relative > recon_tol:
        raise RuntimeError("sign-binned phase space does not reconstruct linewidth")
    lte_histogram_error = 0.0
    lte_histogram_relative = 0.0
    if lte_histogram_edges.size:
        lte_process_reference = process_gamma[:, :, :nmag]
        lte_process_total = np.sum(lte_gamma_histogram, axis=-1)
        lte_histogram_error = _max_abs(lte_process_total - lte_process_reference)
        lte_histogram_relative = _relative_reconstruction_error(
            lte_process_total, lte_process_reference
        )
        if (
            lte_histogram_error > recon_tol * gamma_scale
            and lte_histogram_relative > recon_tol
        ):
            raise RuntimeError(
                "physical-LTe histogram does not reconstruct process linewidth"
            )

    grouped_gamma = group_atom_component_matrix_by_helicity(component_gamma, nat)
    grouped_vertex_gram = group_atom_component_matrix_by_helicity(
        component_vertex_gram, nat
    )
    observables = linewidth_observables(full_gamma)
    chirality_mismatch = (
        physical_channel_chirality[:, None] != physical_channel_chirality[None, :]
    )
    forbidden_transition_intensity = np.sum(
        physical_transition_intensity[:, chirality_mismatch], axis=1
    )
    total_transition_intensity = np.sum(physical_transition_intensity, axis=(1, 2))
    chirality_flip_fraction = np.divide(
        forbidden_transition_intensity,
        total_transition_intensity,
        out=np.zeros_like(forbidden_transition_intensity),
        where=total_transition_intensity > 0.0,
    )
    atom_matrix = component_gamma.reshape(
        nk, 2, nat, nhelicity, nat, nhelicity, nchannel
    )
    atom_vertex_matrix = component_vertex_gram.reshape(
        nk, nat, nhelicity, nat, nhelicity, nmag
    )

    payload: dict[str, np.ndarray] = {
        "rotational_coupling_schema_version": np.asarray(
            COUPLING_SCHEMA_VERSION, dtype=np.int32
        ),
        "analysis_axis_cart": np.asarray(axis_vector, dtype=np.float64),
        "atom_mass_electron": np.asarray(validated.masses, dtype=np.float64),
        "atom_frac": np.asarray(geometry.atom_frac, dtype=np.float64),
        "lattice_ang": np.asarray(geometry.lattice_ang, dtype=np.float64),
        "q_mesh_shape_full": np.asarray(geometry.q_mesh_shape, dtype=np.int32),
        "q_stride": np.asarray(geometry.q_stride, dtype=np.int32),
        "q_mesh_shift_grid": np.asarray(geometry.q_mesh_shift_grid, dtype=np.float64),
        "q_mesh_origin_frac": np.asarray(geometry.q_mesh_origin_frac, dtype=np.float64),
        "q_mesh_sampling_convention": np.asarray(geometry.q_mesh_sampling_convention),
        "n_qpoints_selected_before_omission": np.asarray(
            normalization_q_count, dtype=np.int64
        ),
        "n_omitted": np.asarray(omitted_goldstone_q_flat_indices.size, dtype=np.int64),
        "n_qpoints_used": np.asarray(total_q, dtype=np.int64),
        "normalization_q_count": np.asarray(normalization_q_count, dtype=np.int64),
        "goldstone_omitted_q_flat_indices": omitted_goldstone_q_flat_indices,
        "goldstone_omitted_q_frac": omitted_goldstone_q_frac,
        "goldstone_omitted_q_grid_indices": omitted_goldstone_q_grid_indices,
        "q_flat_indices": np.asarray(q_indices, dtype=np.int64),
        "q_grid_indices": np.asarray(q_grid_indices, dtype=np.int64),
        "q_mesh_flat_frac": np.asarray(q_frac, dtype=np.float64),
        "q_mesh_flat_cart": np.asarray(q_cart, dtype=np.float64),
        "phonon_energies_analyzed_mev": np.asarray(phonon_energies, dtype=np.float64),
        "phonon_energies_coupling_mev": np.asarray(
            coupling_phonon_energies, dtype=np.float64
        ),
        "atom_angular_momentum_axis_over_hbar": atom_angular_axis,
        "k_points_frac": np.asarray(k_frac, dtype=np.float64),
        "k_points_cart_inv_ang": np.asarray(k_cart, dtype=np.float64),
        "magnon_energies_mev": external_energies,
        "bdg_metric_diag": metric,
        "physical_channel_chirality": physical_channel_chirality,
        "bdg_sector_chirality": bdg_sector_chirality,
        "hole_channel_indices": HOLE_CHANNEL_INDICES,
        "hole_physical_partner": HOLE_PHYSICAL_PARTNERS,
        "process_labels": np.asarray(PROCESS_LABELS),
        "helicity_component_labels": np.asarray(COMPONENT_LABELS),
        "gamma_helicity_labels_by_process": np.asarray(
            [
                (
                    "minus_helicity_of_emitted_minus_q",
                    "plus_helicity_of_emitted_minus_q",
                    "axis_of_emitted_minus_q",
                ),
                (
                    "plus_helicity_of_absorbed_plus_q",
                    "minus_helicity_of_absorbed_plus_q",
                    "axis_of_absorbed_plus_q",
                ),
            ]
        ),
        "sign_bin_labels": np.asarray(SIGN_BIN_LABELS),
        "physical_lte_histogram_edges_over_hbar": lte_histogram_edges,
        "physical_lte_histogram_centers_over_hbar": lte_histogram_centers,
        "physical_lte_histogram_mode_counts": lte_histogram_mode_counts,
        "component_labels": np.asarray(
            [
                f"atom_{atom}:{helicity}"
                for atom in range(nat)
                for helicity in COMPONENT_LABELS
            ]
        ),
        "gamma_atom_component_matrix_mev": atom_matrix,
        "gamma_helicity_grouped_matrix_mev": grouped_gamma,
        "vertex_atom_component_matrix_mev2": atom_vertex_matrix,
        "vertex_helicity_grouped_matrix_mev2": grouped_vertex_gram,
        "vertex_physical_chirality_transition_mev2": physical_transition_intensity,
        "vertex_forbidden_chirality_flip_mev2": forbidden_transition_intensity,
        "vertex_chirality_flip_fraction": chirality_flip_fraction,
        "gamma_process_mev": process_gamma,
        "gamma_hwhm_mev": observables["gamma_hwhm_mev"],
        "fwhm_mev": observables["fwhm_mev"],
        "lifetime_ps": observables["lifetime_ps"],
        "scattering_rate_ps_inv": observables["scattering_rate_ps_inv"],
        "valid_damping": observables["valid_damping"],
        # Keep the schema-v1 key as an external-k alias, while schema v2
        # publishes the external/internal distinction explicitly.
        "paraunitary_residual_max_abs": paraunitary_error,
        "external_paraunitary_residual_max_abs": paraunitary_error,
        "internal_paraunitary_residual_max_abs": internal_paraunitary_error,
        "internal_paraunitary_residual_max_abs_before_goldstone_omission": (
            internal_paraunitary_error_before_omission
        ),
        "external_metric_energy_minimum_mev": external_metric_energy_minimum,
        "internal_metric_energy_minimum_mev": internal_metric_energy_minimum,
        "internal_metric_energy_minimum_mev_before_goldstone_omission": (
            internal_metric_energy_minimum_before_omission
        ),
    }
    payload["selectivity_operator_atom_weights"] = selection_weights
    payload["selectivity_operator_angular_momentum_axis_over_hbar"] = selection_angular
    payload["vertex_selectivity_bins_mev2"] = vertex_bins
    payload["vertex_transition_selectivity_bins_mev2"] = transition_bins
    payload["phase_space_selectivity_gamma_mev"] = phase_bins
    payload["physical_lte_gamma_histogram_mev"] = lte_gamma_histogram
    payload["selectivity_mode_counts"] = selectivity_counts
    payload["vertex_selectivity_contrast"] = _selectivity_contrast(vertex_bins)
    payload["phase_space_selectivity_signed_contrast"] = _signed_selectivity_contrast(
        phase_bins
    )
    if operator_weights is not None:
        payload["chiralization_operator_atom_weights"] = operator_weights
        payload["chiralization_operator_angular_momentum_axis_over_hbar"] = (
            selection_angular
        )
    if save_vertices:
        payload["lambda_atom_helicity_qnu_b"] = np.concatenate(
            saved_lambda_chunks, axis=1
        )
        payload["g_atom_helicity_kqnu_channel_channel"] = np.stack(
            [np.concatenate(chunks, axis=1) for chunks in saved_g_chunks],
            axis=0,
        )

    physical_summary = []
    for ik in range(nk):
        channels = []
        for channel in range(nmag):
            channels.append(
                {
                    "channel": channel,
                    "chirality": int(physical_channel_chirality[channel]),
                    "energy_mev": _json_number(external_energies[ik, channel]),
                    "gamma_hwhm_mev": _json_number(full_gamma[ik, channel]),
                    "fwhm_mev": _json_number(observables["fwhm_mev"][ik, channel]),
                    "lifetime_ps": _json_number(
                        observables["lifetime_ps"][ik, channel]
                    ),
                    "scattering_rate_ps_inv": _json_number(
                        observables["scattering_rate_ps_inv"][ik, channel]
                    ),
                }
            )
        physical_summary.append(
            {
                "k_index": ik,
                "k_frac": [float(value) for value in k_frac[ik]],
                "channels": channels,
            }
        )

    component_diagonal = np.einsum("kpccm->kpm", component_gamma, optimize=True)
    diagnostic_reasons = []
    if static_soc["active"] is True:
        diagnostic_reasons.append("active_static_j_soc")
    if dynamic_provenance["tensor_trace_fallback"]:
        diagnostic_reasons.append("unsafe_dynamic_tensor_trace")
    if not dynamic_provenance["scalar_spin_group_projected"]:
        diagnostic_reasons.append("unprojected_dynamic_scalar")
    if chiral_energy_offdiagonal > STRICT_CHIRAL_ENERGY_OFFDIAG_MEV:
        diagnostic_reasons.append("discarded_phonon_energy_offdiagonal")
    if unit_report["diagnostic_assumption_used"]:
        diagnostic_reasons.append("missing_explicit_units")
    if invalid_geometry:
        diagnostic_reasons.append("geometry_unavailable_or_mismatch")
    if unstable_phonon_count:
        diagnostic_reasons.append("unstable_phonon_modes_clipped")
    if normalized_asr_mode == "none":
        diagnostic_reasons.append("asr_unchecked")
    if duplicate_or_mismatch:
        diagnostic_reasons.append("duplicate_or_mismatched_bond_multiplicity")
    if omitted_goldstone_q_flat_indices.size:
        diagnostic_reasons.append("omitted_goldstone_quadrature_node")
    if diagnostic_reasons:
        interpretation_status = "diagnostic_only"
    elif static_soc["active"] is None:
        interpretation_status = "static_soc_metadata_unknown"
    else:
        interpretation_status = "soc_off_projected_scalar_interpretation"
    summary: dict[str, Any] = {
        "schema_version": COUPLING_SCHEMA_VERSION,
        "formalism": "atomic_gauge_scalar_exchange_bosonic_bdg",
        "interpretation_status": interpretation_status,
        "diagnostic_reasons": diagnostic_reasons,
        "source_jr": str(Path(jr_path).expanduser().absolute()),
        "source_djr": str(Path(djr_path).expanduser().absolute()),
        "source_phonon_cache": str(Path(phonon_cache_path).expanduser().absolute()),
        "axis_cart": [float(value) for value in axis_vector],
        "n_atoms": nat,
        "n_modes": nmode,
        "n_qpoints_full": validated.n_qpoints,
        "n_qpoints_selected_before_omission": normalization_q_count,
        "n_omitted": int(omitted_goldstone_q_flat_indices.size),
        "n_qpoints_used": int(total_q),
        "normalization_q_count": normalization_q_count,
        "q_mesh_shape_full": list(geometry.q_mesh_shape),
        "q_stride": list(geometry.q_stride),
        "q_mesh_shift_grid": [float(value) for value in geometry.q_mesh_shift_grid],
        "q_mesh_origin_frac": [float(value) for value in geometry.q_mesh_origin_frac],
        "q_mesh_sampling_convention": geometry.q_mesh_sampling_convention,
        "n_kpoints": nk,
        "temperature_k": temperature,
        "eta_mev": broadening,
        "spin_S": spin_length,
        "bond_factor": bond_scale,
        "anisotropy_mev": anisotropy,
        "spin_pattern": [float(value) for value in spin],
        "physical_channel_chirality": [
            int(value) for value in physical_channel_chirality
        ],
        "hole_mapping": {
            "hole_channels": [2, 3],
            "physical_partners": [1, 0],
            "sector_chirality": [
                int(bdg_sector_chirality[2]),
                int(bdg_sector_chirality[3]),
            ],
        },
        "shell_filter": shell_report,
        "post_filter_bond_keys": bond_key_report,
        "static_j_soc": static_soc,
        "static_j_provenance": static_provenance,
        "dynamic_dj_provenance": dynamic_provenance,
        "unit_contract": unit_report,
        "geometry_contract": {
            "tolerance": float(geometry_tolerance),
            "invalid_sources": invalid_geometry,
            "diagnostic_override_used": bool(invalid_geometry),
            **geometry_reports,
        },
        "dj_asr": asr_report,
        "goldstone_node_handling": goldstone_report,
        "chiralization": chiral_summary,
        "phonon_regularization": {
            "negative_count_clipped_to_zero": negative_phonon_count,
            "acoustic_noise_count_clipped_to_zero": (
                negative_phonon_count - unstable_phonon_count
            ),
            "unstable_below_negative_tolerance_count": unstable_phonon_count,
            "negative_phonon_tolerance_mev": negative_phonon_tol,
            "unstable_diagnostic_override_used": bool(unstable_phonon_count),
            "minimum_original_mev": minimum_phonon_energy,
            "zero_point_denominator_floor_mev": floor,
        },
        "reconstruction": {
            "tolerance": recon_tol,
            "lambda_max_abs": lambda_reconstruction_max_abs,
            "lambda_max_relative_l2": lambda_reconstruction_relative_l2,
            "linewidth_chunk_max_abs_mev": linewidth_reconstruction_max_abs,
            "linewidth_chunk_max_relative_l2": linewidth_reconstruction_relative_l2,
            "linewidth_final_max_abs_mev": gamma_error,
            "linewidth_final_relative_l2": gamma_relative,
            "process_sum_max_abs_mev": process_error,
            "phase_space_bin_sum_max_abs_mev": phase_error,
            "phase_space_bin_sum_relative_l2": phase_relative,
            "physical_lte_histogram_sum_max_abs_mev": lte_histogram_error,
            "physical_lte_histogram_sum_relative_l2": lte_histogram_relative,
        },
        "max_paraunitary_residual": _max_abs(paraunitary_error),
        "bdg_validation": {
            "paraunitary_tolerance": para_tol,
            "external_paraunitary_residual_max_abs": _max_abs(paraunitary_error),
            "internal_paraunitary_residual_max_abs": _max_abs(
                internal_paraunitary_error
            ),
            "internal_paraunitary_residual_max_abs_before_goldstone_omission": (
                _max_abs(internal_paraunitary_error_before_omission)
            ),
            "metric_energy_tolerance_mev": metric_energy_tol,
            "external_metric_energy_minimum_mev": float(
                np.min(external_metric_energy_minimum)
            ),
            "internal_metric_energy_minimum_mev": float(
                np.min(internal_metric_energy_minimum)
            ),
            "internal_metric_energy_minimum_mev_before_goldstone_omission": (
                float(np.min(internal_metric_energy_minimum_before_omission))
            ),
            "fail_closed": True,
        },
        "vertex_chirality": {
            "transition_matrix_final_by_initial_mev2": physical_transition_intensity.tolist(),
            "forbidden_flip_intensity_mev2": forbidden_transition_intensity.tolist(),
            "forbidden_flip_fraction": chirality_flip_fraction.tolist(),
            "maximum_forbidden_flip_fraction": float(
                np.max(chirality_flip_fraction, initial=0.0)
            ),
        },
        "physical_results": physical_summary,
        "component_accounting": {
            "component_order": "atom-major, then plus/minus/axis",
            "matrix_diagonal_is_signed_bdg_process_contribution": True,
            "matrix_offdiagonal_is_ordered_signed_bdg_interference": True,
            "signed_bdg_process_diagonal_component_gamma_mev": (
                component_diagonal.tolist()
            ),
            "signed_bdg_total_offdiagonal_interference_gamma_mev": (
                full_gamma - np.sum(component_diagonal, axis=1)
            ).tolist(),
            "sign_semantics": (
                "These are signed contributions after the bosonic metric, Bose "
                "factors, and retarded emission/absorption poles. A diagonal "
                "entry is not a positive-definite partial scattering rate."
            ),
        },
        "selectivity": {
            "enabled": True,
            "operator_kind": selection_operator_kind,
            "operator_atom_weights": [float(value) for value in selection_weights],
            "sign_bins": list(SIGN_BIN_LABELS),
            "absorption_sign": "sign(chi_initial * L_operator(q))",
            "emission_sign": "sign(-chi_initial * L_operator(q)); emitted phonon is -q",
            "vertex_bins": "matrix element only; physical final magnons",
            "phase_space_bins": "full BdG metric, Bose factors, and broadened poles",
            "physical_lte_histogram": {
                "enabled": bool(lte_histogram_edges.size),
                "n_bins": int(lte_histogram_centers.size),
                "edges_over_hbar": lte_histogram_edges.tolist(),
                "physical_emission_coordinate": "L_physical(-q) = -L_code(q)",
                "physical_absorption_coordinate": "L_physical(+q) = +L_code(q)",
                "values": (
                    "signed BdG/process-resolved contributions; summing bins "
                    "reconstructs gamma_process for physical magnon channels"
                ),
            },
            "component_label_note": (
                "The emission process stores code-q Lambda components. Therefore "
                "plus_q is minus helicity for the physical emitted -q phonon; use "
                "gamma_helicity_labels_by_process."
            ),
        },
        "interpretation_caution": (
            ("DIAGNOSTIC-ONLY PROVENANCE: " + ", ".join(diagnostic_reasons) + ". ")
            if diagnostic_reasons
            else ""
        )
        + (
            (
                "ACTIVE SOC METADATA IS PRESENT IN STATIC J: this scalar-trace "
                "result is diagnostic only and is not an exact SOC-off spin-group "
                "or altermagnetic-symmetry test. "
            )
            if static_soc["active"] is True
            else ""
        )
        + (
            "This scalar dJ model preserves the collinear chirality sectors. "
            "Any helicity correlation is a scattering form-factor selectivity; "
            "it is not a direct one-magnon/one-phonon angular-momentum-transfer vertex."
        ),
        "vertices_saved": bool(save_vertices),
        "saved_vertices_memory": {
            **saved_vertex_memory,
            "peak_formula": "2 * lambda_payload_bytes + 3 * g_payload_bytes",
            "decimal_gb": saved_vertex_peak_gb,
        },
        "saved_vertices_estimated_payload_bytes": saved_vertex_payload_bytes,
        "saved_vertices_estimated_peak_gb": saved_vertex_peak_gb,
        "saved_vertices_limit_gb": saved_vertex_limit_gb,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return RotationalCouplingResult(payload=payload, summary=summary)


def write_rotational_coupling(
    output_npz: str | os.PathLike[str],
    summary_json: str | os.PathLike[str],
    result: RotationalCouplingResult,
    *,
    overwrite: bool = False,
    compressed: bool = True,
) -> tuple[str, str]:
    """Atomically write the small analysis products, with no-clobber default."""

    compatible = RotationalAnalysisResult(
        payload=result.payload, summary=result.summary
    )
    return write_rotational_analysis(
        output_npz,
        summary_json,
        compatible,
        overwrite=overwrite,
        compressed=compressed,
    )


def build_parser(
    *,
    require_k_points: bool = True,
    include_output_arguments: bool = True,
    description: str | None = None,
) -> argparse.ArgumentParser:
    """Build the coupling CLI parser.

    The optional switches let MPI/front-end drivers reuse exactly the same
    physical and numerical arguments without copying defaults.  The public
    no-argument behavior remains the standalone single-process CLI.
    """

    parser = argparse.ArgumentParser(
        description=description
        or (
            "Resolve scalar magnon-phonon scattering into atomistic circular "
            "phonon components, interference, and chirality-selective linewidths."
        ),
    )
    parser.add_argument("--jr", required=True, help="Static scalar J real-space file")
    parser.add_argument("--djr", required=True, help="Scalar dJ(R,Rp) HDF5/NPZ file")
    parser.add_argument(
        "--phonon-cache", required=True, help="Current-schema phonon cache NPZ"
    )
    parser.add_argument(
        "--axis", required=True, type=float, nargs=3, metavar=("NX", "NY", "NZ")
    )
    parser.add_argument(
        "--spin-pattern",
        required=True,
        help="Two-sublattice collinear signs, e.g. 1,-1",
    )
    if require_k_points:
        parser.add_argument(
            "--k-point",
            required=True,
            action="append",
            type=float,
            nargs=3,
            metavar=("K1", "K2", "K3"),
            help="Fractional initial magnon k; repeat for multiple pilot points",
        )
    parser.add_argument(
        "--temperature-k", "--T", dest="temperature_k", required=True, type=float
    )
    parser.add_argument("--eta-mev", "--eta", dest="eta_mev", required=True, type=float)
    parser.add_argument("--spin-S", "--S", dest="spin_S", required=True, type=float)
    parser.add_argument("--bond-factor", required=True, type=float)
    parser.add_argument("--anisotropy-mev", required=True, type=float)
    parser.add_argument(
        "--q-stride",
        type=int,
        nargs=3,
        default=(1, 1, 1),
        metavar=("S1", "S2", "S3"),
        help="Uniform periodic q-grid stride; each stride must divide q_mesh_shape",
    )
    parser.add_argument(
        "--exclude-shell", action="append", type=int, default=[], metavar="N"
    )
    parser.add_argument(
        "--exclude-shell-apply", choices=("both", "J", "dJ"), default="both"
    )
    parser.add_argument("--shell-tol", type=float, default=1.0e-4)
    parser.add_argument(
        "--degeneracy-tol-mev",
        type=float,
        default=1.0e-8,
        help="Absolute phonon-degeneracy tolerance for helicity gauge fixing",
    )
    parser.add_argument(
        "--chiralization-energy-offdiag-tol-mev",
        type=float,
        default=1.0e-10,
        help=(
            "Maximum discarded phonon-Hamiltonian offdiagonal after finite-tolerance "
            "chiralization (default: 1e-10 meV)"
        ),
    )
    parser.add_argument(
        "--chiralize-atom-weights",
        type=float,
        nargs="+",
        default=None,
        metavar="W",
        help="One signed helicity-operator weight per phonon atom",
    )
    parser.add_argument("--normalization-tol", type=float, default=1.0e-7)
    parser.add_argument("--geometry-tol", type=float, default=1.0e-8)
    parser.add_argument(
        "--negative-phonon-tol-mev",
        type=float,
        default=1.0e-6,
        help=(
            "Maximum negative phonon energy treated as acoustic numerical noise; "
            "more-negative modes fail closed"
        ),
    )
    parser.add_argument("--orthonormal-tol", type=float, default=1.0e-7)
    parser.add_argument("--angular-zero-tol", type=float, default=1.0e-10)
    parser.add_argument(
        "--lte-gamma-bins",
        type=int,
        default=0,
        help=(
            "Odd number >=3 of physical LTe/hbar bins for Gamma histograms; "
            "zero disables the histogram (default: 0)"
        ),
    )
    parser.add_argument(
        "--lte-gamma-max-abs-over-hbar",
        type=float,
        default=None,
        help=(
            "Optional strict symmetric physical-LTe histogram bound; the "
            "observed mode range is used when omitted"
        ),
    )
    parser.add_argument("--phonon-floor-mev", type=float, default=1.0e-3)
    parser.add_argument(
        "--dJ-asr",
        "--dj-asr",
        dest="dj_asr",
        choices=("none", "check", "project"),
        default="check",
    )
    parser.add_argument(
        "--dJ-asr-tol", "--dj-asr-tol", dest="dj_asr_tol", type=float, default=1.0e-8
    )
    parser.add_argument("--reconstruction-tol", type=float, default=1.0e-9)
    parser.add_argument("--paraunitary-tol", type=float, default=1.0e-7)
    parser.add_argument("--metric-energy-tol-mev", type=float, default=1.0e-8)
    parser.add_argument(
        "--goldstone-node-policy",
        choices=("fail", "omit"),
        default="fail",
        help=(
            "Fail on a defective/near-exact internal Goldstone quadrature node, "
            "or explicitly omit it while preserving the original q normalization"
        ),
    )
    parser.add_argument(
        "--goldstone-energy-tol-mev",
        type=float,
        default=1.0e-8,
        help="Metric-positive magnon-energy threshold for Goldstone-node detection",
    )
    parser.add_argument("--q-chunk", type=int, default=256)
    parser.add_argument("--bond-chunk", type=int, default=64)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Numba worker threads; defaults to the environment setting",
    )
    if include_output_arguments:
        parser.add_argument("--output", required=True, help="Output NPZ path")
        parser.add_argument("--summary-json", default=None)
    parser.add_argument(
        "--save-vertices",
        action="store_true",
        help="Explicitly store potentially huge component Lambda and g arrays",
    )
    parser.add_argument(
        "--allow-active-static-soc-diagnostic",
        action="store_true",
        help=(
            "Allow a scalar-trace diagnostic with SOC metadata in static J; "
            "the output is marked invalid for exact SOC-off symmetry claims"
        ),
    )
    parser.add_argument(
        "--allow-unsafe-tensor-trace",
        action="store_true",
        help="Allow diagnostic trace(dJ_tensor_r)/3 when no scalar dJ dataset exists",
    )
    parser.add_argument(
        "--allow-unprojected-scalar-diagnostic",
        action="store_true",
        help="Allow dJ without scalar_spin_group_projected provenance as diagnostic",
    )
    parser.add_argument(
        "--allow-missing-units-diagnostic",
        action="store_true",
        help=(
            "Assume legacy static-J meV and/or dJ meV/angstrom only when explicit "
            "unit metadata is absent; marks output diagnostic-only"
        ),
    )
    parser.add_argument(
        "--allow-geometry-mismatch-diagnostic",
        action="store_true",
        help="Allow unavailable/mismatched J/dJ/cache geometry as diagnostic-only",
    )
    parser.add_argument(
        "--geometry-epr",
        default=None,
        help=(
            "Explicit EPR HDF5 used only to validate dynamic-dJ geometry after "
            "moving a projected file whose recorded absolute EPR path is stale"
        ),
    )
    parser.add_argument(
        "--allow-unstable-phonon-diagnostic",
        action="store_true",
        help="Clip phonons below the negative tolerance as diagnostic-only",
    )
    parser.add_argument(
        "--allow-asr-none-diagnostic",
        action="store_true",
        help="Allow --dJ-asr none as diagnostic-only",
    )
    parser.add_argument(
        "--allow-duplicate-bonds-diagnostic",
        action="store_true",
        help="Allow duplicate J/dJ bond keys or multiplicity mismatch as diagnostic-only",
    )
    parser.add_argument(
        "--max-saved-vertices-gb",
        type=float,
        default=2.0,
        help="Peak-memory safety limit used only with --save-vertices",
    )
    parser.add_argument(
        "--compressed", action=argparse.BooleanOptionalAction, default=True
    )
    if include_output_arguments:
        parser.add_argument("--overwrite", action="store_true")
    return parser


def coupling_kwargs_from_namespace(
    args: argparse.Namespace,
    *,
    k_points_frac: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Translate shared CLI options into ``analyze_rotational_coupling`` kwargs."""

    points = getattr(args, "k_point", None) if k_points_frac is None else k_points_frac
    if points is None:
        raise ValueError("k_points_frac must be supplied by the caller")
    return {
        "jr_path": str(Path(args.jr).expanduser().absolute()),
        "djr_path": str(Path(args.djr).expanduser().absolute()),
        "phonon_cache_path": str(Path(args.phonon_cache).expanduser().absolute()),
        "axis": args.axis,
        "spin_pattern": args.spin_pattern,
        "k_points_frac": points,
        "temperature_k": args.temperature_k,
        "eta_mev": args.eta_mev,
        "spin_S": args.spin_S,
        "bond_factor": args.bond_factor,
        "anisotropy_mev": args.anisotropy_mev,
        "q_stride": args.q_stride,
        "exclude_shells": args.exclude_shell,
        "exclude_shell_apply": args.exclude_shell_apply,
        "shell_tolerance": args.shell_tol,
        "degeneracy_tolerance_mev": args.degeneracy_tol_mev,
        "chiralization_atom_weights": args.chiralize_atom_weights,
        "chiralization_energy_offdiag_tolerance_mev": (
            args.chiralization_energy_offdiag_tol_mev
        ),
        "normalization_tolerance": args.normalization_tol,
        "geometry_tolerance": args.geometry_tol,
        "negative_phonon_tolerance_mev": args.negative_phonon_tol_mev,
        "orthonormal_tolerance": args.orthonormal_tol,
        "angular_zero_tolerance": args.angular_zero_tol,
        "lte_gamma_histogram_bins": args.lte_gamma_bins,
        "lte_gamma_histogram_max_abs_over_hbar": (
            args.lte_gamma_max_abs_over_hbar
        ),
        "phonon_floor_mev": args.phonon_floor_mev,
        "dj_asr_mode": args.dj_asr,
        "dj_asr_tolerance": args.dj_asr_tol,
        "reconstruction_tolerance": args.reconstruction_tol,
        "q_chunk": args.q_chunk,
        "bond_chunk": args.bond_chunk,
        "save_vertices": args.save_vertices,
        "max_saved_vertices_gb": args.max_saved_vertices_gb,
        "paraunitary_tolerance": args.paraunitary_tol,
        "metric_energy_tolerance_mev": args.metric_energy_tol_mev,
        "goldstone_node_policy": args.goldstone_node_policy,
        "goldstone_energy_tolerance_mev": args.goldstone_energy_tol_mev,
        "allow_active_static_soc_diagnostic": (args.allow_active_static_soc_diagnostic),
        "allow_unsafe_tensor_trace": args.allow_unsafe_tensor_trace,
        "allow_unprojected_scalar_diagnostic": (
            args.allow_unprojected_scalar_diagnostic
        ),
        "allow_missing_units_diagnostic": args.allow_missing_units_diagnostic,
        "allow_geometry_mismatch_diagnostic": (args.allow_geometry_mismatch_diagnostic),
        "geometry_epr_path": args.geometry_epr,
        "allow_unstable_phonon_diagnostic": (args.allow_unstable_phonon_diagnostic),
        "allow_asr_none_diagnostic": args.allow_asr_none_diagnostic,
        "allow_duplicate_bonds_diagnostic": (args.allow_duplicate_bonds_diagnostic),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_threads is not None:
        from numba import set_num_threads

        set_num_threads(_positive_integer(args.num_threads, "num_threads"))
    result = analyze_rotational_coupling(**coupling_kwargs_from_namespace(args))
    output = Path(args.output).expanduser()
    summary_path = (
        Path(args.summary_json).expanduser()
        if args.summary_json is not None
        else output.with_suffix(".json")
    )
    summary = dict(result.summary)
    summary["output_npz"] = str(output.absolute())
    summary["output_json"] = str(summary_path.absolute())
    result = RotationalCouplingResult(payload=result.payload, summary=summary)
    written_npz, written_json = write_rotational_coupling(
        output,
        summary_path,
        result,
        overwrite=args.overwrite,
        compressed=args.compressed,
    )
    print(f"[magph-rotational-coupling] wrote {written_npz}")
    print(f"[magph-rotational-coupling] wrote {written_json}")
    if result.summary["static_j_soc"]["active"] is True:
        print(
            "[magph-rotational-coupling][WARN] active static-J SOC metadata: "
            "diagnostic scalar trace only; no exact SOC-off symmetry claim",
            flush=True,
        )
    elif result.summary["static_j_soc"]["active"] is None:
        print(
            "[magph-rotational-coupling][WARN] static-J SOC metadata is unknown; "
            "verify provenance before a symmetry claim",
            flush=True,
        )
    if result.summary["diagnostic_reasons"]:
        print(
            "[magph-rotational-coupling][WARN] diagnostic-only provenance: "
            + ", ".join(result.summary["diagnostic_reasons"]),
            flush=True,
        )
    for item in result.summary["physical_results"]:
        for channel in item["channels"]:
            print(
                "[magph-rotational-coupling] "
                f"k={item['k_index']} chi={channel['chirality']:+d} "
                f"Gamma={channel['gamma_hwhm_mev']} meV "
                f"tau={channel['lifetime_ps']} ps"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
