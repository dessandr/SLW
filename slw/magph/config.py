"""Typed QE-namelist configuration for native magnon lifetimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .derivative import DerivativeASRPolicy
from .model import MagneticOrder


class MagphInputError(ValueError):
    """Raised when native magnon--phonon input is incomplete or ambiguous."""


_LIFETIME_KEYS = {
    "asr_policy",
    "bond_chunk_size",
    "broadening_mev",
    "channel_chunk_size",
    "derivative_h5",
    "exchange_h5",
    "frequency_floor_mev",
    "input_format",
    "kmesh",
    "kshift",
    "magnetic_order",
    "metric_energy_tolerance_mev",
    "negative_tolerance_mev",
    "output",
    "overwrite",
    "phonon_cache",
    "q_chunk_size",
    "quantization_axis",
    "require_complete_targets",
    "self_energy_q_chunk_size",
    "spin_magnitudes",
    "spin_pattern",
    "temperature_k",
    "vertex_q_chunk_size",
}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, np.ndarray):
        return tuple(value.reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _float_tuple(
    value: Any, *, name: str, length: int | None = None
) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in _sequence(value))
    except (TypeError, ValueError) as exc:
        raise MagphInputError(f"{name} must contain real numbers") from exc
    if length is not None and len(result) != length:
        raise MagphInputError(f"{name} must contain exactly {length} values")
    if not result or not np.all(np.isfinite(result)):
        raise MagphInputError(f"{name} must contain finite real numbers")
    return result


def _positive_mesh(value: Any, *, name: str) -> tuple[int, int, int]:
    raw = np.asarray(_sequence(value))
    if raw.shape != (3,) or np.iscomplexobj(raw):
        raise MagphInputError(f"{name} must contain three positive integers")
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MagphInputError(f"{name} must contain three positive integers") from exc
    if (
        not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, np.rint(numeric))
        or np.any(numeric <= 0.0)
    ):
        raise MagphInputError(f"{name} must contain three positive integers")
    return int(numeric[0]), int(numeric[1]), int(numeric[2])


def _finite_scalar(
    value: Any,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MagphInputError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result):
        raise MagphInputError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise MagphInputError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise MagphInputError(f"{name} must be non-negative")
    return result


def _optional_positive_integer(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    raw = _finite_scalar(value, name=name, positive=True)
    if not raw.is_integer():
        raise MagphInputError(f"{name} must be a positive integer")
    return int(raw)


def _boolean(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", ".true.", "yes", "on", "1"}:
        return True
    if text in {"false", ".false.", "no", "off", "0"}:
        return False
    raise MagphInputError(f"{name} must be a boolean")


def _required(parameters: dict[str, Any], name: str) -> Any:
    if name not in parameters:
        raise MagphInputError(f"native lifetime input requires {name}")
    return parameters[name]


@dataclass(frozen=True)
class MagphLifetimeRequest:
    exchange_h5: Path
    derivative_h5: Path
    phonon_cache: Path
    output: Path
    magnetic_order: MagneticOrder
    spin_magnitudes: tuple[float, ...]
    spin_pattern: tuple[float, ...] | None
    quantization_axis: tuple[float, float, float]
    kmesh: tuple[int, int, int]
    kshift: tuple[float, float, float]
    temperature_k: float
    broadening_mev: float
    frequency_floor_mev: float | None
    asr_policy: DerivativeASRPolicy
    metric_energy_tolerance_mev: float
    negative_tolerance_mev: float
    q_chunk_size: int | None
    bond_chunk_size: int | None
    vertex_q_chunk_size: int | None
    self_energy_q_chunk_size: int | None
    channel_chunk_size: int | None
    require_complete_targets: bool
    overwrite: bool


def build_lifetime_request(
    parameters: dict[str, Any],
    *,
    prefix: str,
    savedir: str | Path,
) -> MagphLifetimeRequest:
    """Validate one native lifetime namelist without opening scientific files."""

    values = {str(key).strip().lower(): value for key, value in parameters.items()}
    unknown = sorted(set(values) - _LIFETIME_KEYS)
    if unknown:
        raise MagphInputError(
            "unknown native lifetime parameter(s): " + ", ".join(unknown)
        )
    input_format = str(values.pop("input_format", "default")).strip().lower()
    if input_format not in {"default", "native"}:
        raise MagphInputError("native lifetime accepts input_format='native' only")

    exchange_h5 = Path(_required(values, "exchange_h5")).expanduser().resolve()
    derivative_h5 = Path(_required(values, "derivative_h5")).expanduser().resolve()
    phonon_cache = Path(_required(values, "phonon_cache")).expanduser().resolve()
    output = (
        Path(values.get("output", Path(savedir) / f"{prefix}.lifetime.npz"))
        .expanduser()
        .resolve()
    )
    if output.suffix.lower() != ".npz":
        raise MagphInputError("output must use the .npz suffix")

    try:
        magnetic_order = MagneticOrder(
            str(_required(values, "magnetic_order")).strip().lower()
        )
    except ValueError as exc:
        choices = ", ".join(item.value for item in MagneticOrder)
        raise MagphInputError(f"magnetic_order must be one of {choices}") from exc
    spin_magnitudes = _float_tuple(
        _required(values, "spin_magnitudes"), name="spin_magnitudes"
    )
    if any(value <= 0.0 for value in spin_magnitudes):
        raise MagphInputError("spin_magnitudes must be strictly positive")
    spin_pattern = (
        None
        if "spin_pattern" not in values
        else _float_tuple(values["spin_pattern"], name="spin_pattern")
    )
    quantization_axis = _float_tuple(
        _required(values, "quantization_axis"),
        name="quantization_axis",
        length=3,
    )
    if np.linalg.norm(quantization_axis) <= np.finfo(np.float64).eps:
        raise MagphInputError("quantization_axis must be nonzero")

    kmesh = _positive_mesh(_required(values, "kmesh"), name="kmesh")
    kshift_raw = _float_tuple(_required(values, "kshift"), name="kshift", length=3)
    kshift = (kshift_raw[0], kshift_raw[1], kshift_raw[2])
    temperature_k = _finite_scalar(
        _required(values, "temperature_k"),
        name="temperature_k",
        nonnegative=True,
    )
    broadening_mev = _finite_scalar(
        _required(values, "broadening_mev"),
        name="broadening_mev",
        positive=True,
    )
    frequency_floor_mev = (
        None
        if "frequency_floor_mev" not in values
        else _finite_scalar(
            values["frequency_floor_mev"],
            name="frequency_floor_mev",
            positive=True,
        )
    )
    try:
        asr_policy = DerivativeASRPolicy(
            str(values.get("asr_policy", "fail")).strip().lower()
        )
    except ValueError as exc:
        choices = ", ".join(item.value for item in DerivativeASRPolicy)
        raise MagphInputError(f"asr_policy must be one of {choices}") from exc

    return MagphLifetimeRequest(
        exchange_h5=exchange_h5,
        derivative_h5=derivative_h5,
        phonon_cache=phonon_cache,
        output=output,
        magnetic_order=magnetic_order,
        spin_magnitudes=spin_magnitudes,
        spin_pattern=spin_pattern,
        quantization_axis=(
            quantization_axis[0],
            quantization_axis[1],
            quantization_axis[2],
        ),
        kmesh=kmesh,
        kshift=kshift,
        temperature_k=temperature_k,
        broadening_mev=broadening_mev,
        frequency_floor_mev=frequency_floor_mev,
        asr_policy=asr_policy,
        metric_energy_tolerance_mev=_finite_scalar(
            values.get("metric_energy_tolerance_mev", 0.0),
            name="metric_energy_tolerance_mev",
            nonnegative=True,
        ),
        negative_tolerance_mev=_finite_scalar(
            values.get("negative_tolerance_mev", 0.0),
            name="negative_tolerance_mev",
            nonnegative=True,
        ),
        q_chunk_size=_optional_positive_integer(
            values.get("q_chunk_size"), name="q_chunk_size"
        ),
        bond_chunk_size=_optional_positive_integer(
            values.get("bond_chunk_size"), name="bond_chunk_size"
        ),
        vertex_q_chunk_size=_optional_positive_integer(
            values.get("vertex_q_chunk_size"), name="vertex_q_chunk_size"
        ),
        self_energy_q_chunk_size=_optional_positive_integer(
            values.get("self_energy_q_chunk_size"),
            name="self_energy_q_chunk_size",
        ),
        channel_chunk_size=_optional_positive_integer(
            values.get("channel_chunk_size"), name="channel_chunk_size"
        ),
        require_complete_targets=_boolean(
            values.get("require_complete_targets", True),
            name="require_complete_targets",
        ),
        overwrite=_boolean(values.get("overwrite", False), name="overwrite"),
    )


__all__ = ["MagphInputError", "MagphLifetimeRequest", "build_lifetime_request"]
