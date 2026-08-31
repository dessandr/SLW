"""Gauge metadata records; conversions are never inferred from array shape."""

from __future__ import annotations

from dataclasses import dataclass

from slw.wtorque.config import BlochGauge


@dataclass(frozen=True)
class GaugeConvention:
    bloch_gauge: BlochGauge
    bloch_exponent_sign: int
    field_fourier_exponent_sign: int
    centers_unit: str = "reduced"
    final_state_representation: str = "unwrapped"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bloch_gauge", BlochGauge(self.bloch_gauge))
        if self.bloch_exponent_sign not in {-1, 1}:
            raise ValueError("bloch_exponent_sign must be +1 or -1")
        if self.field_fourier_exponent_sign not in {-1, 1}:
            raise ValueError("field_fourier_exponent_sign must be +1 or -1")
        if self.centers_unit not in {"reduced", "angstrom"}:
            raise ValueError("centers_unit must be reduced or angstrom")
        if self.final_state_representation not in {"wrapped", "unwrapped"}:
            raise ValueError("final_state_representation must be wrapped or unwrapped")


__all__ = ["GaugeConvention"]

