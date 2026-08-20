"""Typed public models for spinor-basis and atomic-SOC inputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum


class SpinorGroupBy(str, Enum):
    """The two spinor basis layouts supported by TB2J."""

    SPIN = "spin"
    ORBITAL = "orbital"


_MANIFOLD = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9_]*)-(?P<orbital>[pdPD])$")


@dataclass(frozen=True, order=True)
class AtomicSOCManifold:
    """One atomic ``lambda L.S`` term selected from Wannier90 projections."""

    selector: str
    lambda_ev: float

    def __post_init__(self) -> None:
        selector = str(self.selector).strip()
        match = _MANIFOLD.fullmatch(selector)
        if match is None:
            raise ValueError(
                "SOC selector must have LABEL-p or LABEL-d syntax, "
                f"got {self.selector!r}"
            )
        value = float(self.lambda_ev)
        if not math.isfinite(value):
            raise ValueError(f"SOC lambda must be finite, got {self.lambda_ev!r}")
        canonical = f"{match.group('label')}-{match.group('orbital').lower()}"
        object.__setattr__(self, "selector", canonical)
        object.__setattr__(self, "lambda_ev", value)

    @property
    def label(self) -> str:
        return self.selector.rsplit("-", 1)[0]

    @property
    def orbital(self) -> str:
        return self.selector.rsplit("-", 1)[1]


@dataclass(frozen=True)
class AtomicSOCSpec:
    """Validated atomic onsite SOC card."""

    manifolds: tuple[AtomicSOCManifold, ...]

    def __post_init__(self) -> None:
        manifolds = tuple(self.manifolds)
        if not manifolds:
            raise ValueError("SOC (atomic) must contain at least one LAMBDA entry")
        if any(not isinstance(item, AtomicSOCManifold) for item in manifolds):
            raise TypeError(
                "AtomicSOCSpec.manifolds must contain AtomicSOCManifold values"
            )
        keys = [item.selector.casefold() for item in manifolds]
        if len(keys) != len(set(keys)):
            raise ValueError("SOC (atomic) contains duplicate manifold selectors")
        object.__setattr__(self, "manifolds", manifolds)


__all__ = ["AtomicSOCManifold", "AtomicSOCSpec", "SpinorGroupBy"]
