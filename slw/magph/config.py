"""Typed QE-namelist configuration for native magnon calculations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .derivative import DerivativeASRPolicy
from .model import (
    ExchangeSpinNormalization,
    MagneticOrder,
    SingleIonAnisotropy,
)

_ANISOTROPY_KEYS = {
    "anisotropy_axis",
    "anisotropy_mev",
    "anisotropy_model",
    "anisotropy_normalization",
}


class MagphInputError(ValueError):
    """Raised when native magnon--phonon input is incomplete or ambiguous."""


class RestartMode(str, Enum):
    """Output policy for one native magph calculation."""

    ERROR = "error"
    RESTART = "restart"
    FROM_SCRATCH = "from_scratch"


_LIFETIME_KEYS = {
    "asr_policy",
    "broadening_mev",
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
    "quantization_axis",
    "require_complete_targets",
    "restart_mode",
    "spin_magnitudes",
    "spin_pattern",
    "temperature_k",
} | _ANISOTROPY_KEYS

_DISPERSION_KEYS = {
    "exchange_h5",
    "input_format",
    "kpath_file",
    "magnetic_order",
    "output",
    "overwrite",
    "plot",
    "plot_dpi",
    "plot_output",
    "points_per_segment",
    "quantization_axis",
    "restart_mode",
    "spin_magnitudes",
    "spin_pattern",
} | _ANISOTROPY_KEYS


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


def _restart_mode(values: dict[str, Any]) -> RestartMode:
    """Parse the native output policy, retaining ``overwrite`` as an alias."""

    if "restart_mode" in values and "overwrite" in values:
        raise MagphInputError(
            "restart_mode and the deprecated overwrite key are mutually exclusive"
        )
    if "overwrite" in values:
        return (
            RestartMode.FROM_SCRATCH
            if _boolean(values["overwrite"], name="overwrite")
            else RestartMode.ERROR
        )
    raw = str(values.get("restart_mode", RestartMode.ERROR.value)).strip().lower()
    try:
        return RestartMode(raw)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in RestartMode)
        raise MagphInputError(f"restart_mode must be one of {choices}") from exc


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


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise MagphInputError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise MagphInputError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or numeric != round(numeric):
        raise MagphInputError(f"{name} must be a positive integer")
    return int(numeric)


@dataclass(frozen=True)
class SingleIonAnisotropyInput:
    """Namelist form of uniaxial SIA before the magnetic site count is known."""

    energy_mev: tuple[float, ...]
    axis: tuple[float, ...]
    spin_normalization: ExchangeSpinNormalization
    model: str = "uniaxial"

    def build(self, n_magnetic_sites: int) -> SingleIonAnisotropy:
        count = int(n_magnetic_sites)
        if count < 1:
            raise MagphInputError("n_magnetic_sites must be positive")
        if len(self.energy_mev) == 1:
            energy = np.full(count, self.energy_mev[0], dtype=np.float64)
        elif len(self.energy_mev) == count:
            energy = np.asarray(self.energy_mev, dtype=np.float64)
        else:
            raise MagphInputError(
                "anisotropy_mev must contain one value or one value per magnetic site"
            )
        raw_axis = np.asarray(self.axis, dtype=np.float64)
        if raw_axis.size == 3:
            axes = np.tile(raw_axis.reshape(1, 3), (count, 1))
        elif raw_axis.size == 3 * count:
            axes = raw_axis.reshape(count, 3)
        else:
            raise MagphInputError(
                "anisotropy_axis must contain one Cartesian axis or one axis "
                "per magnetic site"
            )
        return SingleIonAnisotropy(
            energy_mev=energy,
            axis=axes,
            spin_normalization=self.spin_normalization,
            model=self.model,
        )


def _anisotropy_input(
    values: dict[str, Any],
) -> SingleIonAnisotropyInput | None:
    present = sorted(set(values) & _ANISOTROPY_KEYS)
    if not present:
        return None
    missing = [
        name
        for name in (
            "anisotropy_model",
            "anisotropy_mev",
            "anisotropy_axis",
            "anisotropy_normalization",
        )
        if name not in values
    ]
    if missing:
        raise MagphInputError("single-ion anisotropy requires " + ", ".join(missing))
    model = str(values["anisotropy_model"]).strip().lower()
    if model != "uniaxial":
        raise MagphInputError("anisotropy_model must be 'uniaxial'")
    try:
        normalization = ExchangeSpinNormalization(
            str(values["anisotropy_normalization"]).strip().lower()
        )
    except ValueError as exc:
        choices = ", ".join(item.value for item in ExchangeSpinNormalization)
        raise MagphInputError(
            f"anisotropy_normalization must be one of {choices}"
        ) from exc
    return SingleIonAnisotropyInput(
        energy_mev=_float_tuple(values["anisotropy_mev"], name="anisotropy_mev"),
        axis=_float_tuple(values["anisotropy_axis"], name="anisotropy_axis"),
        spin_normalization=normalization,
        model=model,
    )


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
    anisotropy: SingleIonAnisotropyInput | None
    kmesh: tuple[int, int, int]
    kshift: tuple[float, float, float]
    temperature_k: float
    broadening_mev: float
    frequency_floor_mev: float | None
    asr_policy: DerivativeASRPolicy
    metric_energy_tolerance_mev: float
    negative_tolerance_mev: float
    require_complete_targets: bool
    restart_mode: RestartMode

    @property
    def overwrite(self) -> bool:
        """Whether atomic writers may replace completed products."""

        return self.restart_mode is RestartMode.FROM_SCRATCH


@dataclass(frozen=True)
class MagphDispersionRequest:
    exchange_h5: Path
    kpath_file: Path
    output: Path
    plot_output: Path | None
    magnetic_order: MagneticOrder
    spin_magnitudes: tuple[float, ...]
    spin_pattern: tuple[float, ...] | None
    quantization_axis: tuple[float, float, float]
    anisotropy: SingleIonAnisotropyInput | None
    points_per_segment: int
    plot_dpi: int
    restart_mode: RestartMode

    @property
    def overwrite(self) -> bool:
        """Whether atomic writers may replace completed products."""

        return self.restart_mode is RestartMode.FROM_SCRATCH


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
        anisotropy=_anisotropy_input(values),
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
        require_complete_targets=_boolean(
            values.get("require_complete_targets", True),
            name="require_complete_targets",
        ),
        restart_mode=_restart_mode(values),
    )


def build_dispersion_request(
    parameters: dict[str, Any],
    *,
    prefix: str,
    savedir: str | Path,
) -> MagphDispersionRequest:
    """Validate native magnon-dispersion input without opening scientific files."""

    values = {str(key).strip().lower(): value for key, value in parameters.items()}
    unknown = sorted(set(values) - _DISPERSION_KEYS)
    if unknown:
        raise MagphInputError(
            "unknown native dispersion parameter(s): " + ", ".join(unknown)
        )
    input_format = str(values.pop("input_format", "default")).strip().lower()
    if input_format not in {"default", "native"}:
        raise MagphInputError("native dispersion accepts input_format='native' only")
    exchange_h5 = Path(_required(values, "exchange_h5")).expanduser().resolve()
    kpath_file = Path(_required(values, "kpath_file")).expanduser().resolve()
    output = (
        Path(values.get("output", Path(savedir) / f"{prefix}.dispersion.npz"))
        .expanduser()
        .resolve()
    )
    if output.suffix.lower() != ".npz":
        raise MagphInputError("output must use the .npz suffix")
    plot_enabled = _boolean(values.get("plot", True), name="plot")
    plot_output = (
        Path(values.get("plot_output", Path(savedir) / f"{prefix}.dispersion.png"))
        .expanduser()
        .resolve()
        if plot_enabled
        else None
    )
    if plot_output is not None and plot_output.suffix.lower() not in {
        ".png",
        ".pdf",
        ".svg",
    }:
        raise MagphInputError("plot_output must use .png, .pdf, or .svg")
    if not plot_enabled and "plot_output" in values:
        raise MagphInputError("plot_output cannot be set when plot=.false.")
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
    return MagphDispersionRequest(
        exchange_h5=exchange_h5,
        kpath_file=kpath_file,
        output=output,
        plot_output=plot_output,
        magnetic_order=magnetic_order,
        spin_magnitudes=spin_magnitudes,
        spin_pattern=spin_pattern,
        quantization_axis=(
            quantization_axis[0],
            quantization_axis[1],
            quantization_axis[2],
        ),
        anisotropy=_anisotropy_input(values),
        points_per_segment=_positive_integer(
            values.get("points_per_segment", 50), name="points_per_segment"
        ),
        plot_dpi=_positive_integer(values.get("plot_dpi", 180), name="plot_dpi"),
        restart_mode=_restart_mode(values),
    )


__all__ = [
    "MagphDispersionRequest",
    "MagphInputError",
    "MagphLifetimeRequest",
    "RestartMode",
    "SingleIonAnisotropyInput",
    "build_dispersion_request",
    "build_lifetime_request",
]
