"""Validation of QE-style namelist values for the native exchange engine."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from pathlib import Path
from typing import Any

from slw.soc.model import AtomicSOCManifold, AtomicSOCSpec, SpinorGroupBy

from .model import (
    ExchangeCalculation,
    ExchangeFiles,
    ExchangeOptions,
    ExchangeOutput,
    ExchangeRequest,
    ExchangeSource,
    OrbitalSlice,
    TensorKernel,
)


class ExchangeInputError(ValueError):
    """Raised when a native exchange input is incomplete or inconsistent."""


_FILE_KEYS = {
    "epr_up",
    "epr_dn",
    "up_hr",
    "dn_hr",
    "spinor_hr",
    "win",
    "centres",
}

_J_SCALAR_EPR_OPTIONS = {
    "atom_labels",
    "species_labels",
    "hr_unit",
    "n_shells",
    "d_max",
    "emin",
    "empoints",
    "integrator",
    "cfr_beta",
    "nproc",
    "symprec",
    "angle_tolerance",
    "orbit_grouping",
    "orbit_symmetry",
    "orbit_symmetry_tolerance_mev",
    "debug_orbits",
    "debug_orbit_shell",
    "debug_epr_positions",
    "debug_shell",
    "debug_bond",
    "debug_out",
    "no_symmetry_orbits",
    "no_h5",
}
_J_TENSOR_EPR_OPTIONS = {
    "hr_unit",
    "axes",
    "spin_direction",
    "n_shells",
    "d_max",
    "all_bonds",
    "canonical_bonds",
    "nn_only",
    "atom_labels",
    "species_labels",
    "no_symmetry",
    "orbit_grouping",
    "symprec",
    "angle_tolerance",
    "debug_orbits",
    "debug_orbit_shell",
    "debug_epr_positions",
    "integrator",
    "emin",
    "empoints",
    "cfr_beta",
    "nproc",
    "collinear_override",
    "spin_magnitude",
}
_J_TENSOR_WANNIER_OPTIONS = {
    "hr_unit",
    "ref_epr_up",
    "ref_epr_dn",
    "ref_hr_unit",
    "apply_degeneracy",
    "axes",
    "spin_direction",
    "n_shells",
    "d_max",
    "all_bonds",
    "canonical_bonds",
    "nn_only",
    "orbit_grouping",
    "integrator",
    "emin",
    "empoints",
    "cfr_beta",
    "nproc",
    "collinear_override",
    "spin_magnitude",
}
_J_SCALAR_WANNIER_OPTIONS = {
    "hr_unit",
    "apply_degeneracy",
    "axes",
    "spin_direction",
    "n_shells",
    "d_max",
    "all_bonds",
    "canonical_bonds",
    "nn_only",
    "orbit_grouping",
    "integrator",
    "emin",
    "empoints",
    "cfr_beta",
    "nproc",
    "collinear_override",
    "spin_magnitude",
    "ref_epr_up",
    "ref_epr_dn",
    "ref_hr_unit",
}
_DJ_SCALAR_OPTIONS = {
    "hr_unit",
    "eph_unit",
    "atom_labels",
    "species_labels",
    "targets",
    "axes",
    "qmesh",
    "rp_idx",
    "g_transform",
    "g_kernel",
    "n_shells",
    "d_max",
    "emin",
    "empoints",
    "integrator",
    "cfr_beta",
    "nproc",
    "omp_threads",
    "precache_workers",
    "rotation_mode",
    "ddelta_mode",
    "symprec",
    "angle_tolerance",
    "orbit_grouping",
    "debug_orbits",
    "debug_orbit_shell",
    "debug_epr_positions",
    "no_symmetry_orbits",
}
_DJ_TENSOR_OPTIONS = {
    "spinor_hr_unit",
    "apply_degeneracy",
    "hr_unit",
    "eph_unit",
    "qmesh",
    "n_shells",
    "d_max",
    "targets",
    "disp_axes",
    "tensor_axes",
    "spin_direction",
    "emin",
    "empoints",
    "integrator",
    "nproc",
    "numba_threads",
    "blas_threads",
    "progress_every",
    "verbose_worker_init",
    "checkpoint",
    "onsite_deriv_projector",
}


@dataclass(frozen=True)
class _OptionSpec:
    """Declarative validation rule for one advanced numerical option."""

    kind: str
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    strict_minimum: bool = False
    length: int | None = None
    allow_none: bool = False
    allow_empty: bool = True
    nonzero: bool = False


_ADVANCED_OPTION_SPECS: dict[str, _OptionSpec] = {
    **{
        name: _OptionSpec("bool")
        for name in (
            "all_bonds",
            "apply_degeneracy",
            "canonical_bonds",
            "collinear_override",
            "debug_epr_positions",
            "debug_orbits",
            "nn_only",
            "no_h5",
            "no_symmetry",
            "no_symmetry_orbits",
            "onsite_deriv_projector",
        )
    },
    **{name: _OptionSpec("str") for name in ("atom_labels",)},
    "debug_out": _OptionSpec("str", allow_empty=False),
    "debug_bond": _OptionSpec("int_csv_vector", length=5, allow_empty=True),
    **{
        name: _OptionSpec("int", minimum=1.0)
        for name in (
            "blas_threads",
            "empoints",
            "n_shells",
            "nproc",
            "numba_threads",
            "omp_threads",
            "precache_workers",
        )
    },
    **{name: _OptionSpec("int", minimum=0.0) for name in ("checkpoint", "progress_every", "verbose_worker_init")},
    "debug_orbit_shell": _OptionSpec("int", minimum=0.0, allow_none=True),
    "debug_shell": _OptionSpec("int", minimum=0.0, allow_none=True),
    **{name: _OptionSpec("float") for name in ("angle_tolerance", "emin")},
    "cfr_beta": _OptionSpec("float", minimum=0.0, strict_minimum=True),
    "d_max": _OptionSpec("float", minimum=0.0, strict_minimum=True),
    "spin_magnitude": _OptionSpec("float", minimum=0.0, strict_minimum=True),
    "symprec": _OptionSpec("float", minimum=0.0, strict_minimum=True),
    "orbit_symmetry_tolerance_mev": _OptionSpec("float", minimum=0.0),
    "qmesh": _OptionSpec("int_vector", length=3, minimum=0.0, strict_minimum=True, allow_none=True),
    "rp_idx": _OptionSpec("int_vector", length=3),
    "spin_direction": _OptionSpec("float_vector", length=3, nonzero=True),
    "axes": _OptionSpec("axes"),
    "disp_axes": _OptionSpec("axes"),
    "tensor_axes": _OptionSpec("axes"),
    "eph_unit": _OptionSpec("choice", choices=("ev", "ry", "ha")),
    "hr_unit": _OptionSpec("choice", choices=("ev", "ry", "ha")),
    "ref_hr_unit": _OptionSpec("choice", choices=("ev", "ry", "ha")),
    "spinor_hr_unit": _OptionSpec("choice", choices=("ev", "ry", "ha")),
    "ddelta_mode": _OptionSpec("choice", choices=("off", "local", "onsite")),
    "g_transform": _OptionSpec("choice", choices=("kq", "k_only_rp")),
    "g_kernel": _OptionSpec("choice", choices=("direct", "spectral")),
    "rotation_mode": _OptionSpec("choice", choices=("none",)),
    "orbit_symmetry": _OptionSpec("choice", choices=("report", "project", "fail")),
    "ref_epr_up": _OptionSpec("path", allow_none=True),
    "ref_epr_dn": _OptionSpec("path", allow_none=True),
}


def _normalized_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(parameters, Mapping):
        raise ExchangeInputError("exchange parameters must be a mapping")
    result: dict[str, Any] = {}
    for raw_key, value in parameters.items():
        key = str(raw_key).strip().lower().replace("-", "_")
        if not key:
            raise ExchangeInputError("exchange parameter names cannot be empty")
        if key in result:
            raise ExchangeInputError(f"exchange parameter {key!r} is defined more than once")
        result[key] = value
    return result


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
    raise ExchangeInputError(f"{name} must be a boolean, got {value!r}")


def _mode_option_specs(
    calculation: ExchangeCalculation,
    ltensor: bool,
    source: ExchangeSource,
) -> dict[str, _OptionSpec]:
    specs = dict(_ADVANCED_OPTION_SPECS)
    integrators: tuple[str, ...]
    if calculation is ExchangeCalculation.DJ and ltensor:
        integrators = ("contour", "cfr")
    elif ltensor or source is ExchangeSource.WANNIER:
        integrators = ("contour", "cfr_ozaki", "cfr_pole")
    else:
        integrators = ("contour", "cfr", "cfr_ozaki")
    specs["integrator"] = _OptionSpec("choice", choices=integrators)

    orbit_groupings: tuple[str, ...]
    if source is ExchangeSource.WANNIER:
        orbit_groupings = ("none", "distance", "shell")
    else:
        orbit_groupings = ("spglib", "shell")
    specs["orbit_grouping"] = _OptionSpec("choice", choices=orbit_groupings)

    if calculation is ExchangeCalculation.DJ and ltensor:
        specs["targets"] = _OptionSpec("int_vector", minimum=0.0, allow_none=True)
    else:
        specs["targets"] = _OptionSpec("str", allow_empty=False)

    if calculation is ExchangeCalculation.J and ltensor:
        specs["species_labels"] = _OptionSpec("string_list", allow_none=True)
    else:
        specs["species_labels"] = _OptionSpec("str", allow_none=True)
    return specs


def _checked_number(value: Any, *, name: str, integral: bool) -> int | float:
    if isinstance(value, bool):
        expected = "an integer" if integral else "a finite real number"
        raise ExchangeInputError(f"{name} must be {expected}, got {value!r}")
    if integral:
        if not isinstance(value, Integral):
            raise ExchangeInputError(f"{name} must be an integer, got {value!r}")
        return int(value)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExchangeInputError(f"{name} must be a finite real number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ExchangeInputError(f"{name} must be finite, got {value!r}")
    return result


def _check_bound(value: float, spec: _OptionSpec, *, name: str) -> None:
    if spec.minimum is None:
        return
    invalid = value <= spec.minimum if spec.strict_minimum else value < spec.minimum
    if invalid:
        relation = "greater than" if spec.strict_minimum else "at least"
        raise ExchangeInputError(f"{name} must be {relation} {spec.minimum:g}")


def _validated_vector(value: Any, spec: _OptionSpec, *, name: str) -> tuple[Any, ...]:
    items = _items(value)
    if spec.length is not None and len(items) != spec.length:
        raise ExchangeInputError(f"{name} must contain exactly {spec.length} values, got {items!r}")
    if spec.length is None and not items:
        raise ExchangeInputError(f"{name} must contain at least one value")
    integral = spec.kind == "int_vector"
    result = []
    for item in items:
        number = _checked_number(item, name=name, integral=integral)
        _check_bound(number, spec, name=name)
        result.append(number)
    if spec.nonzero and not any(float(item) != 0.0 for item in result):
        raise ExchangeInputError(f"{name} must not be the zero vector")
    return tuple(result)


def _validated_option(value: Any, spec: _OptionSpec, *, name: str) -> Any:
    if value is None:
        if spec.allow_none:
            return None
        raise ExchangeInputError(f"{name} cannot be null")
    if spec.kind == "bool":
        return _as_bool(value, name=name)
    if spec.kind == "int":
        int_value = int(_checked_number(value, name=name, integral=True))
        _check_bound(int_value, spec, name=name)
        if spec.choices and int_value not in spec.choices:
            choices = ", ".join(map(str, spec.choices))
            raise ExchangeInputError(f"{name} must be one of {choices}, got {value!r}")
        return int_value
    if spec.kind == "float":
        float_value = float(_checked_number(value, name=name, integral=False))
        _check_bound(float_value, spec, name=name)
        return float_value
    if spec.kind in {"int_vector", "float_vector"}:
        return _validated_vector(value, spec, name=name)
    if spec.kind == "choice":
        choice_value = str(value).strip().lower().replace("-", "_")
        if choice_value not in spec.choices:
            choices = ", ".join(map(str, spec.choices))
            raise ExchangeInputError(f"{name} must be one of {choices}, got {value!r}")
        return choice_value
    if spec.kind == "axes":
        if not isinstance(value, str):
            raise ExchangeInputError(f"{name} must be a string containing x, y, and/or z")
        axes_value = "".join(char for char in value.strip().lower() if char not in {",", " ", "\t"})
        if not axes_value or any(char not in "xyz" for char in axes_value):
            raise ExchangeInputError(f"{name} must contain only x, y, and/or z")
        if len(set(axes_value)) != len(axes_value):
            raise ExchangeInputError(f"{name} contains duplicate axes: {value!r}")
        return axes_value
    if spec.kind == "string_list":
        if isinstance(value, str):
            strings = tuple(item.strip() for item in value.split(",") if item.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            if any(not isinstance(item, str) for item in value):
                raise ExchangeInputError(f"{name} must contain only strings")
            strings = tuple(item.strip() for item in value if item.strip())
        else:
            raise ExchangeInputError(f"{name} must be a string or sequence of strings")
        return strings
    if spec.kind == "path":
        try:
            path_value = os.fspath(value)
        except TypeError as exc:
            raise ExchangeInputError(f"{name} must be a filesystem path") from exc
        if not isinstance(path_value, str):
            raise ExchangeInputError(f"{name} must be a text filesystem path")
        path_value = path_value.strip()
        if not path_value:
            if spec.allow_none:
                return None
            raise ExchangeInputError(f"{name} cannot be empty")
        return path_value
    if spec.kind == "int_csv_vector":
        if not isinstance(value, str):
            raise ExchangeInputError(f"{name} must be a comma-separated integer string")
        text = value.strip()
        if not text and spec.allow_empty:
            return ""
        fields = [item.strip() for item in text.replace(":", ",").split(",")]
        if spec.length is not None and len(fields) != spec.length:
            raise ExchangeInputError(f"{name} must contain exactly {spec.length} integers")
        try:
            integers = tuple(int(item) for item in fields)
        except ValueError as exc:
            raise ExchangeInputError(f"{name} must contain only integers") from exc
        return ",".join(map(str, integers))
    if spec.kind == "str":
        if not isinstance(value, str):
            raise ExchangeInputError(f"{name} must be a string, got {value!r}")
        if not spec.allow_empty and not value.strip():
            raise ExchangeInputError(f"{name} cannot be empty")
        return value
    raise ExchangeInputError(f"{name} is unsupported by the native option schema")


def _validate_advanced_options(
    values: dict[str, Any],
    allowed: set[str],
    *,
    calculation: ExchangeCalculation,
    ltensor: bool,
    source: ExchangeSource,
) -> None:
    specs = _mode_option_specs(calculation, ltensor, source)
    unsupported = sorted(allowed - set(specs))
    if unsupported:
        raise ExchangeInputError("advanced option(s) lack a native validation schema and are unsupported: " + ", ".join(unsupported))
    for name in sorted(values):
        values[name] = _validated_option(values[name], specs[name], name=name)


def _enum_value(enum_type, value: Any, *, name: str):
    if isinstance(value, enum_type):
        return value
    text = str(value).strip().lower().replace("-", "_")
    try:
        return enum_type(text)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ExchangeInputError(f"{name} must be one of {choices}, got {value!r}") from exc


def _normalize_mode(calculation: Any, values: dict[str, Any]) -> tuple[ExchangeCalculation, bool]:
    raw = calculation.value if isinstance(calculation, ExchangeCalculation) else str(calculation).strip().lower().replace("-", "_")
    aliases = {
        "j": (ExchangeCalculation.J, False),
        "j_tensor": (ExchangeCalculation.J, True),
        "dj": (ExchangeCalculation.DJ, False),
        "dj_tensor": (ExchangeCalculation.DJ, True),
    }
    try:
        mode, alias_tensor = aliases[raw]
    except KeyError as exc:
        raise ExchangeInputError("calculation must be j or dj (j_tensor/dj_tensor are compatibility aliases)") from exc

    duplicate = values.pop("calculation", None)
    if duplicate is not None:
        duplicate_values: dict[str, Any] = {}
        duplicate_mode, duplicate_tensor = _normalize_mode(duplicate, duplicate_values)
        if duplicate_mode != mode or duplicate_tensor != alias_tensor:
            raise ExchangeInputError(f"calculation argument {calculation!r} conflicts with parameter {duplicate!r}")

    explicit = values.pop("ltensor", None)
    if explicit is None:
        return mode, alias_tensor
    requested = _as_bool(explicit, name="ltensor")
    if alias_tensor and not requested:
        raise ExchangeInputError(f"calculation={raw!r} requires ltensor=true")
    return mode, requested or alias_tensor


def _source(values: dict[str, Any], calculation: ExchangeCalculation) -> ExchangeSource:
    input_format = values.pop("input_format", None)
    source = values.pop("source", None)
    if input_format is not None and source is not None:
        left = _enum_value(ExchangeSource, input_format, name="input_format")
        right = _enum_value(ExchangeSource, source, name="source")
        if left != right:
            raise ExchangeInputError("source and input_format select different backends")
        selected = left
    elif input_format is not None:
        selected = _enum_value(ExchangeSource, input_format, name="input_format")
    elif source is not None:
        selected = _enum_value(ExchangeSource, source, name="source")
    else:
        choices = "'epr' or 'wannier'" if calculation is ExchangeCalculation.J else "'epr'"
        raise ExchangeInputError(f"calculation={calculation.value!r} requires explicit input_format={choices}")
    if calculation is ExchangeCalculation.DJ and selected is not ExchangeSource.EPR:
        raise ExchangeInputError("calculation='dj' supports only input_format='epr'")
    return selected


def _tensor_kernel(values: dict[str, Any], *, ltensor: bool) -> TensorKernel:
    has_native = "tensor_kernel" in values
    has_legacy = "kernel" in values
    native = values.pop("tensor_kernel", None)
    legacy = values.pop("kernel", None)
    if not ltensor and (has_native or has_legacy):
        names = " and ".join(name for name, present in (("tensor_kernel", has_native), ("kernel", has_legacy)) if present)
        raise ExchangeInputError(f"{names} is only valid when ltensor=true")
    if has_native and has_legacy:
        left = _enum_value(TensorKernel, native, name="tensor_kernel")
        right = _enum_value(TensorKernel, legacy, name="kernel")
        if left != right:
            raise ExchangeInputError("kernel and tensor_kernel specify different tensor kernels")
        return left
    if has_native:
        return _enum_value(TensorKernel, native, name="tensor_kernel")
    if has_legacy:
        return _enum_value(TensorKernel, legacy, name="kernel")
    return TensorKernel.TB2J


def _normalize_dj_tensor_aliases(values: dict[str, Any]) -> None:
    """Map historical argparse spellings to canonical backend field names."""

    alias = "onsite_deriv_exchange_field"
    canonical = "onsite_deriv_projector"
    if alias not in values:
        return
    alias_value = _as_bool(values.pop(alias), name=alias)
    if canonical in values:
        canonical_value = _as_bool(values[canonical], name=canonical)
        if alias_value != canonical_value:
            raise ExchangeInputError(f"{alias} and {canonical} specify different boolean values")
    values[canonical] = alias_value


def _required_float(values: dict[str, Any], name: str) -> float:
    if name not in values:
        raise ExchangeInputError(f"missing required exchange parameter: {name}")
    raw = values.pop(name)
    if isinstance(raw, bool):
        raise ExchangeInputError(f"{name} must be a finite real number")
    try:
        result = float(raw)
    except (TypeError, ValueError) as exc:
        raise ExchangeInputError(f"{name} must be a finite real number, got {raw!r}") from exc
    if not math.isfinite(result):
        raise ExchangeInputError(f"{name} must be finite, got {raw!r}")
    return result


def _items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _positive_mesh(values: dict[str, Any]) -> tuple[int, int, int]:
    if "kmesh" not in values:
        raise ExchangeInputError("missing required exchange parameter: kmesh")
    raw = _items(values.pop("kmesh"))
    if len(raw) != 3:
        raise ExchangeInputError(f"kmesh must contain exactly three integers, got {raw!r}")
    mesh = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ExchangeInputError(f"kmesh must contain integers, got {item!r}")
        if item <= 0:
            raise ExchangeInputError(f"kmesh entries must be positive, got {raw!r}")
        mesh.append(int(item))
    return tuple(mesh)  # type: ignore[return-value]


def _magnetic_atoms(values: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    if "mag_atoms" not in values:
        raise ExchangeInputError("missing required exchange parameter: mag_atoms")
    raw = _items(values.pop("mag_atoms"))
    if not raw:
        raise ExchangeInputError("mag_atoms must contain at least one atom index")
    base_raw = values.pop("mag_atoms_base", 0)
    if isinstance(base_raw, bool) or not isinstance(base_raw, Integral) or int(base_raw) not in (0, 1):
        raise ExchangeInputError("mag_atoms_base must be 0 or 1")
    atoms = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ExchangeInputError(f"mag_atoms must contain integers, got {item!r}")
        atom = int(item) - int(base_raw)
        if atom < 0:
            raise ExchangeInputError(f"mag_atoms contains an invalid index after base conversion: {item!r}")
        atoms.append(atom)
    if len(set(atoms)) != len(atoms):
        raise ExchangeInputError("mag_atoms contains duplicate atom indices")
    return tuple(atoms), int(base_raw)


def _one_slice(item: Any) -> OrbitalSlice:
    if isinstance(item, OrbitalSlice):
        result = item
    elif isinstance(item, str):
        fields = item.strip().split(":")
        if len(fields) != 3:
            raise ExchangeInputError(f"slice {item!r} must have site:start:stop syntax")
        try:
            result = OrbitalSlice(*(int(field.strip()) for field in fields))
        except ValueError as exc:
            raise ExchangeInputError(f"slice {item!r} must contain integers") from exc
    elif isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
        fields = list(item)
        if len(fields) != 3:
            raise ExchangeInputError(f"slice entry must contain site, start, stop: {item!r}")
        if any(isinstance(field, bool) or not isinstance(field, Integral) for field in fields):
            raise ExchangeInputError(f"slice entry must contain integers: {item!r}")
        result = OrbitalSlice(int(fields[0]), int(fields[1]), int(fields[2]))
    else:
        raise ExchangeInputError(f"unsupported slices entry: {item!r}")
    if result.site < 0 or result.start < 0 or result.stop <= result.start:
        raise ExchangeInputError(f"invalid slice {result.site}:{result.start}:{result.stop}; require site>=0 and 0<=start<stop")
    return result


def _orbital_slices(values: dict[str, Any]) -> tuple[OrbitalSlice, ...]:
    if "slices" not in values:
        raise ExchangeInputError("missing required exchange parameter: slices")
    raw = values.pop("slices")
    entries: list[Any]
    if isinstance(raw, str):
        entries = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, Mapping):
        entries = []
        for site, bounds in raw.items():
            parts = _items(bounds)
            if len(parts) != 2:
                raise ExchangeInputError(f"slice bounds for site {site!r} must have start and stop")
            entries.append((site, parts[0], parts[1]))
    else:
        entries = _items(raw)
    if not entries:
        raise ExchangeInputError("slices must contain at least one orbital interval")
    parsed = tuple(sorted((_one_slice(item) for item in entries), key=lambda item: item.site))
    sites = [item.site for item in parsed]
    if len(set(sites)) != len(sites):
        raise ExchangeInputError("slices contains duplicate site keys")
    by_start = sorted(parsed, key=lambda item: (item.start, item.stop))
    for left, right in pairwise(by_start):
        if right.start < left.stop:
            raise ExchangeInputError(f"orbital slices overlap: {left.site}:{left.start}:{left.stop} and {right.site}:{right.start}:{right.stop}")
    return parsed


def _validate_site_contract(
    mag_atoms: tuple[int, ...],
    slices: tuple[OrbitalSlice, ...],
    *,
    calculation: ExchangeCalculation,
    ltensor: bool,
    source: ExchangeSource,
) -> None:
    needs_two_sites = source is ExchangeSource.EPR and not (calculation is ExchangeCalculation.DJ and ltensor)
    if needs_two_sites and len(mag_atoms) < 2:
        raise ExchangeInputError(f"{calculation.value} EPR {'tensor' if ltensor else 'scalar'} requires mag_atoms to contain at least two atoms")

    keys = {item.site for item in slices}
    local_keys = set(range(len(mag_atoms)))
    if calculation is ExchangeCalculation.DJ and ltensor:
        global_keys = set(mag_atoms)
        if not (local_keys <= keys or global_keys <= keys):
            raise ExchangeInputError(
                "tensor dJ slices must cover either local magnetic-site keys "
                f"{sorted(local_keys)} or normalized global atom keys "
                f"{sorted(global_keys)}; got {sorted(keys)}"
            )
        return
    if not local_keys <= keys:
        raise ExchangeInputError(f"slices must cover zero-based local magnetic-site keys {sorted(local_keys)}; got {sorted(keys)}")


def _optional_path(values: dict[str, Any], name: str) -> str | None:
    raw = values.pop(name, None)
    if raw is None:
        return None
    try:
        path = os.fspath(raw)
    except TypeError as exc:
        raise ExchangeInputError(f"{name} must be a filesystem path, got {raw!r}") from exc
    if not isinstance(path, str):
        raise ExchangeInputError(f"{name} must be a text filesystem path")
    path = path.strip()
    if not path:
        return None
    return path


def _input_files(
    values: dict[str, Any],
    *,
    calculation: ExchangeCalculation,
    ltensor: bool,
    source: ExchangeSource,
) -> ExchangeFiles:
    files = ExchangeFiles(**{name: _optional_path(values, name) for name in _FILE_KEYS})
    epr_any = files.epr_up is not None or files.epr_dn is not None
    collinear_any = files.up_hr is not None or files.dn_hr is not None
    collinear_complete = files.up_hr is not None and files.dn_hr is not None

    if source is ExchangeSource.EPR:
        if files.epr_up is None or files.epr_dn is None:
            raise ExchangeInputError("input_format='epr' requires both epr_up and epr_dn")
        if collinear_any:
            raise ExchangeInputError("EPR input cannot also define up_hr or dn_hr")
        if calculation is ExchangeCalculation.J and files.spinor_hr is not None:
            raise ExchangeInputError("EPR J calculations do not accept spinor_hr")
        if calculation is ExchangeCalculation.J:
            if files.centres is not None:
                raise ExchangeInputError("EPR J calculations do not accept centres")
            if not ltensor and files.win is not None:
                raise ExchangeInputError("scalar EPR J does not accept win")
        elif not ltensor:
            irrelevant = [name for name in ("spinor_hr", "win", "centres") if getattr(files, name) is not None]
            if irrelevant:
                raise ExchangeInputError("scalar dJ does not accept " + ", ".join(irrelevant) + "; spinor inputs require ltensor=true")
        elif files.centres is not None and files.spinor_hr is None:
            raise ExchangeInputError("tensor dJ centres requires spinor_hr")
    else:
        if epr_any:
            raise ExchangeInputError("Wannier input cannot also define epr_up or epr_dn")
        if calculation is ExchangeCalculation.DJ:
            raise ExchangeInputError("dJ requires EPR electron-phonon input")
        if ltensor:
            if files.spinor_hr is not None and collinear_any:
                raise ExchangeInputError("Wannier tensor J requires either spinor_hr or the up_hr/dn_hr pair, not both")
            if files.spinor_hr is None and not collinear_complete:
                raise ExchangeInputError("Wannier tensor J requires spinor_hr or both up_hr and dn_hr")
        else:
            if files.spinor_hr is not None:
                raise ExchangeInputError("scalar Wannier J requires collinear up_hr and dn_hr")
            if not collinear_complete:
                raise ExchangeInputError("scalar Wannier J requires both up_hr and dn_hr")
        if files.centres is not None and files.spinor_hr is None:
            raise ExchangeInputError("Wannier centres requires spinor_hr")
    return files


def _has_value(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _spinor_groupby(values: dict[str, Any], files: ExchangeFiles) -> SpinorGroupBy | None:
    raw = values.pop("groupby", None)
    if files.spinor_hr is None:
        if raw is not None:
            raise ExchangeInputError("groupby is valid only with spinor_hr")
        return None
    if raw is None:
        raise ExchangeInputError("spinor_hr requires explicit groupby='spin' or groupby='orbital'")
    try:
        return SpinorGroupBy(str(raw).strip().lower())
    except ValueError as exc:
        raise ExchangeInputError(f"groupby must be spin or orbital, got {raw!r}") from exc


def _atomic_soc(
    values: dict[str, Any],
    files: ExchangeFiles,
    *,
    ltensor: bool,
) -> AtomicSOCSpec | None:
    raw = values.pop("soc_card", None)
    if raw is None:
        return None
    if not ltensor:
        raise ExchangeInputError("SOC (atomic) requires ltensor=true")
    if files.win is None:
        raise ExchangeInputError("SOC (atomic) requires an explicit win file")
    if not isinstance(raw, Mapping):
        raise ExchangeInputError("SOC card payload must be a mapping")
    mode = str(raw.get("mode", "")).strip().lower()
    if mode != "atomic":
        raise ExchangeInputError(f"unsupported SOC card mode {mode!r}; only atomic is supported")
    entries = raw.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        raise ExchangeInputError("SOC (atomic) entries must be a sequence")
    manifolds: list[AtomicSOCManifold] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            raise ExchangeInputError(f"SOC (atomic) entry {index} must contain selector and lambda_ev")
        unknown = sorted(set(entry) - {"selector", "lambda_ev"})
        if unknown or "selector" not in entry or "lambda_ev" not in entry:
            raise ExchangeInputError(f"SOC (atomic) entry {index} must contain only selector and lambda_ev")
        try:
            manifolds.append(
                AtomicSOCManifold(
                    selector=str(entry["selector"]),
                    lambda_ev=entry["lambda_ev"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ExchangeInputError(str(exc)) from exc
    try:
        return AtomicSOCSpec(tuple(manifolds))
    except ValueError as exc:
        raise ExchangeInputError(str(exc)) from exc


def _validate_advanced_combinations(
    values: dict[str, Any],
    files: ExchangeFiles,
    *,
    calculation: ExchangeCalculation,
    ltensor: bool,
    source: ExchangeSource,
) -> None:
    if calculation is ExchangeCalculation.J and not ltensor and source is ExchangeSource.EPR:
        orbit_policy = values.get("orbit_symmetry")
        if orbit_policy in {"project", "fail"} and (bool(values.get("no_symmetry_orbits", False)) or values.get("orbit_grouping", "spglib") != "spglib"):
            raise ExchangeInputError(f"orbit_symmetry={orbit_policy!r} requires orbit_grouping='spglib' with no_symmetry_orbits=false")

    if calculation is ExchangeCalculation.J and source is ExchangeSource.WANNIER:
        ref_up = _has_value(values.get("ref_epr_up"))
        ref_dn = _has_value(values.get("ref_epr_dn"))
        if ref_up != ref_dn:
            raise ExchangeInputError("ref_epr_up and ref_epr_dn must be provided together")

    # Spinor layout and SOC-card dependencies are normalized before the
    # advanced numerical options reach this compatibility bridge.


def _normalize_tensor_dj_targets(values: dict[str, Any], *, atom_index_base: int) -> None:
    if "targets" not in values or values["targets"] is None:
        return
    raw = _items(values["targets"])
    normalized: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ExchangeInputError("tensor dJ targets must contain integers")
        target = int(item) - atom_index_base
        if target < 0:
            raise ExchangeInputError(f"tensor dJ targets contains an invalid index: {item!r}")
        normalized.append(target)
    if len(set(normalized)) != len(normalized):
        raise ExchangeInputError("tensor dJ targets contains duplicate atom indices")
    values["targets"] = tuple(normalized)


def _output(
    prefix: str,
    savedir: str | os.PathLike[str],
    values: dict[str, Any],
    mode_name: str,
) -> ExchangeOutput:
    if not isinstance(prefix, str):
        raise ExchangeInputError("prefix must be a string")
    prefix = prefix.strip()
    if not prefix or prefix in {".", ".."} or any(sep in prefix for sep in ("/", "\\")):
        raise ExchangeInputError("prefix must be a non-empty file-name prefix")
    try:
        savedir_value = os.fspath(savedir)
    except TypeError as exc:
        raise ExchangeInputError("savedir must be a filesystem path") from exc
    if not isinstance(savedir_value, str):
        raise ExchangeInputError("savedir must be a text filesystem path")
    savedir = savedir_value.strip()
    if not savedir:
        raise ExchangeInputError("savedir cannot be empty")
    raw_directory = values.pop("out_dir", None)
    try:
        directory_value = savedir if raw_directory in (None, "") else os.fspath(raw_directory)
    except TypeError as exc:
        raise ExchangeInputError(f"out_dir must be a filesystem path, got {raw_directory!r}") from exc
    directory = Path(directory_value).expanduser()
    stem = f"{prefix}.{mode_name}"

    def resolved(name: str, default: str) -> str:
        raw = values.pop(name, None)
        try:
            path_value = default if raw in (None, "") else os.fspath(raw)
        except TypeError as exc:
            raise ExchangeInputError(f"{name} must be a filesystem path, got {raw!r}") from exc
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = directory / path
        return str(path)

    h5_path = resolved("out_h5", f"{stem}.h5")
    text_path = resolved("out_name", f"{stem}.txt")
    text = Path(text_path)
    table_path = str(text.with_name(f"{text.stem}.all_bonds.tsv"))
    return ExchangeOutput(
        directory=str(directory),
        h5_path=h5_path,
        text_path=text_path,
        table_path=table_path,
    )


def _allowed_options(
    calculation: ExchangeCalculation,
    ltensor: bool,
    source: ExchangeSource,
) -> set[str]:
    if calculation is ExchangeCalculation.DJ:
        return set(_DJ_TENSOR_OPTIONS if ltensor else _DJ_SCALAR_OPTIONS)
    if ltensor:
        return set(_J_TENSOR_EPR_OPTIONS if source is ExchangeSource.EPR else _J_TENSOR_WANNIER_OPTIONS)
    if source is ExchangeSource.EPR:
        return set(_J_SCALAR_EPR_OPTIONS)
    return set(_J_SCALAR_WANNIER_OPTIONS)


def build_exchange_request(
    calculation: str | ExchangeCalculation,
    parameters: Mapping[str, Any],
    *,
    prefix: str,
    savedir: str | os.PathLike[str],
) -> ExchangeRequest:
    """Validate namelist parameters and construct an immutable native request.

    Validation is deliberately structural and does not touch the filesystem,
    so this function is also suitable for ``--dry-run`` and MPI rank-0 input
    handling. Kernel-specific tuning values are accepted only from an
    explicit per-mode allowlist and are kept in ``options``.
    """

    values = _normalized_parameters(parameters)
    mode, ltensor = _normalize_mode(calculation, values)
    source = _source(values, mode)
    tensor_kernel = _tensor_kernel(values, ltensor=ltensor)
    if mode is ExchangeCalculation.DJ and ltensor:
        _normalize_dj_tensor_aliases(values)
        if tensor_kernel is not TensorKernel.TB2J:
            raise ExchangeInputError("tensor dJ supports tensor_kernel='tb2j' only")
    efermi = _required_float(values, "efermi")
    kmesh = _positive_mesh(values)
    mag_atoms, atom_base = _magnetic_atoms(values)
    slices = _orbital_slices(values)
    _validate_site_contract(
        mag_atoms,
        slices,
        calculation=mode,
        ltensor=ltensor,
        source=source,
    )
    files = _input_files(
        values,
        calculation=mode,
        ltensor=ltensor,
        source=source,
    )
    groupby = _spinor_groupby(values, files)
    soc = _atomic_soc(values, files, ltensor=ltensor)
    mode_name = f"{mode.value}{'_tensor' if ltensor else ''}"
    output = _output(prefix, savedir, values, mode_name)

    allowed = _allowed_options(mode, ltensor, source)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ExchangeInputError(
            f"unknown or unsupported option(s) for calculation={mode.value!r}, ltensor={ltensor}, input_format={source.value!r}: {', '.join(unknown)}"
        )
    _validate_advanced_options(
        values,
        allowed,
        calculation=mode,
        ltensor=ltensor,
        source=source,
    )
    if mode is ExchangeCalculation.DJ and ltensor:
        _normalize_tensor_dj_targets(values, atom_index_base=atom_base)
    _validate_advanced_combinations(
        values,
        files,
        calculation=mode,
        ltensor=ltensor,
        source=source,
    )
    options = dict(values)

    return ExchangeRequest(
        calculation=mode,
        ltensor=ltensor,
        tensor_kernel=tensor_kernel,
        source=source,
        efermi=efermi,
        kmesh=kmesh,
        mag_atoms=mag_atoms,
        atom_index_base=atom_base,
        slices=slices,
        files=files,
        output=output,
        groupby=groupby,
        soc=soc,
        options=ExchangeOptions.from_mapping(options),
    )


__all__ = ["ExchangeInputError", "ExchangeRequest", "build_exchange_request"]
