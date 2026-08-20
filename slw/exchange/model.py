"""Typed, file-I/O-free models for the native exchange engine."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from slw.soc.model import AtomicSOCSpec, SpinorGroupBy


class ExchangeCalculation(str, Enum):
    """Order of the requested exchange calculation."""

    J = "j"
    DJ = "dj"


class ExchangeSource(str, Enum):
    """Electronic Hamiltonian representation used by the calculation."""

    EPR = "epr"
    WANNIER = "wannier"


class TensorKernel(str, Enum):
    """Numerical convention used for a full exchange tensor."""

    TB2J = "tb2j"
    DIRECT = "direct"


@dataclass(frozen=True, order=True)
class OrbitalSlice:
    """Contiguous half-open Wannier-orbital interval for one site key."""

    site: int
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class ExchangeFiles:
    """Explicit scientific input files, separated from numerical options."""

    epr_up: str | None = None
    epr_dn: str | None = None
    up_hr: str | None = None
    dn_hr: str | None = None
    spinor_hr: str | None = None
    win: str | None = None
    centres: str | None = None


@dataclass(frozen=True)
class ExchangeOutput:
    """Resolved deterministic output locations for a native calculation."""

    directory: str
    h5_path: str
    text_path: str
    table_path: str


def _freeze_option_value(value: Any) -> Any:
    """Recursively remove mutable containers from numerical option values."""

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_option_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_option_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_option_value(item) for item in value), key=repr))
    return value


@dataclass(frozen=True)
class ExchangeOptions(Mapping[str, Any]):
    """Immutable, pickle-friendly mapping of validated numerical options."""

    entries: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ExchangeOptions:
        entries = tuple(
            (str(key), _freeze_option_value(value))
            for key, value in sorted(values.items())
        )
        return cls(entries=entries)

    def __getitem__(self, key: str) -> Any:
        for name, value in self.entries:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _value in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow mutable copy for private kernel dispatch."""

        return dict(self.entries)


@dataclass(frozen=True)
class ExchangeRequest:
    """Validated native exchange request.

    ``mag_atoms`` is always normalized to zero-based global atom indices.
    Orbital-slice ``site`` keys retain the explicit convention supplied by the
    caller because existing data may key them by either magnetic-site order or
    global atom index.
    """

    calculation: ExchangeCalculation
    ltensor: bool
    tensor_kernel: TensorKernel
    source: ExchangeSource
    efermi: float
    kmesh: tuple[int, int, int]
    mag_atoms: tuple[int, ...]
    atom_index_base: int
    slices: tuple[OrbitalSlice, ...]
    files: ExchangeFiles
    output: ExchangeOutput
    groupby: SpinorGroupBy | None = None
    soc: AtomicSOCSpec | None = None
    options: ExchangeOptions = ExchangeOptions()

    @property
    def mode_name(self) -> str:
        suffix = "_tensor" if self.ltensor else ""
        return f"{self.calculation.value}{suffix}"

    @property
    def slice_map(self) -> dict[int, slice]:
        return {item.site: item.as_slice() for item in self.slices}
