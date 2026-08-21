"""MPI-distributed native magnon dispersion on explicit Wannier90 paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from slw.cli.mpi import MPIContext
from slw.core.structure import read_wannier90_kpoint_path

from .lswt import solve_isotropic_lswt_energies
from .model import ExchangeModel, MagneticConfiguration, SingleIonAnisotropy
from .parallel import DistributedArrayResult, distributed_array_map


def _readonly(value: object, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _display_label(value: object) -> str:
    label = str(value).strip()
    if label.lower() in {"g", "gam", "gamma", "gm", "Γ"}:
        return "Γ"
    return label


@dataclass(frozen=True)
class MagnonKPath:
    """Interpolated reciprocal path with explicit plot topology."""

    source: Path
    k_points_frac: NDArray[np.float64]
    x_coordinate_inv_ang: NDArray[np.float64]
    segment_offsets: NDArray[np.int64]
    tick_positions_inv_ang: NDArray[np.float64]
    tick_labels: tuple[str, ...]
    points_per_segment: int

    def __post_init__(self) -> None:
        source = Path(self.source)
        points = _readonly(self.k_points_frac, dtype=np.dtype(np.float64))
        x_value = _readonly(self.x_coordinate_inv_ang, dtype=np.dtype(np.float64))
        offsets = _readonly(self.segment_offsets, dtype=np.dtype(np.int64))
        ticks = _readonly(self.tick_positions_inv_ang, dtype=np.dtype(np.float64))
        labels = tuple(str(label) for label in self.tick_labels)
        per_segment = int(self.points_per_segment)
        if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] == 0:
            raise ValueError("k_points_frac must have nonempty shape (nk,3)")
        if x_value.shape != (points.shape[0],):
            raise ValueError("x_coordinate_inv_ang must contain one value per k point")
        if offsets.ndim != 1 or offsets.size < 2:
            raise ValueError("segment_offsets must contain at least one segment")
        if offsets[0] != 0 or offsets[-1] != points.shape[0]:
            raise ValueError("segment_offsets must span the complete k path")
        if np.any(np.diff(offsets) < 2):
            raise ValueError("each k-path segment must contain at least two points")
        if ticks.ndim != 1 or ticks.size != len(labels) or ticks.size < 2:
            raise ValueError("tick positions and labels are inconsistent")
        if per_segment < 1:
            raise ValueError("points_per_segment must be positive")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(x_value)):
            raise ValueError("k path contains non-finite values")
        if not np.all(np.isfinite(ticks)) or np.any(np.diff(ticks) < 0.0):
            raise ValueError("k-path ticks must be finite and nondecreasing")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "k_points_frac", points)
        object.__setattr__(self, "x_coordinate_inv_ang", x_value)
        object.__setattr__(self, "segment_offsets", offsets)
        object.__setattr__(self, "tick_positions_inv_ang", ticks)
        object.__setattr__(self, "tick_labels", labels)
        object.__setattr__(self, "points_per_segment", per_segment)

    @property
    def n_points(self) -> int:
        return int(self.k_points_frac.shape[0])

    @property
    def n_segments(self) -> int:
        return int(self.segment_offsets.size - 1)


@dataclass(frozen=True)
class MagnonDispersionResult:
    """Root-assembled physical energy bands on one explicit path."""

    path: MagnonKPath
    energy_mev: NDArray[np.float64]
    goldstone_mask: NDArray[np.bool_]
    mpi_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, MagnonKPath):
            raise TypeError("path must be a MagnonKPath")
        energy = _readonly(self.energy_mev, dtype=np.dtype(np.float64))
        goldstone = _readonly(self.goldstone_mask, dtype=np.dtype(bool))
        if energy.ndim != 2 or energy.shape[0] != self.path.n_points:
            raise ValueError("energy_mev must have shape (nk,nmode)")
        if goldstone.shape != energy.shape:
            raise ValueError("goldstone_mask must match energy_mev")
        if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
            raise ValueError("magnon energies must be finite and non-negative")
        size = int(self.mpi_size)
        if size < 1:
            raise ValueError("mpi_size must be positive")
        object.__setattr__(self, "energy_mev", energy)
        object.__setattr__(self, "goldstone_mask", goldstone)
        object.__setattr__(self, "mpi_size", size)


@dataclass(frozen=True)
class DistributedMagnonDispersion:
    local_indices: NDArray[np.int64]
    global_result: MagnonDispersionResult | None
    mpi_size: int
    root_rank: int


def build_wannier90_kpath(
    path: str | Path,
    *,
    lattice_ang: ArrayLike,
    points_per_segment: int,
) -> MagnonKPath:
    """Read and interpolate an explicit Wannier90 ``kpoint_path`` block."""

    source = Path(path).expanduser().resolve()
    lattice = np.asarray(lattice_ang, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("lattice_ang must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(lattice))) <= np.finfo(np.float64).eps:
        raise ValueError("lattice_ang must be nonsingular")
    count = int(points_per_segment)
    if count < 1:
        raise ValueError("points_per_segment must be positive")
    segments = read_wannier90_kpoint_path(source)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    all_points: list[NDArray[np.float64]] = []
    all_x: list[NDArray[np.float64]] = []
    offsets = [0]
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    current_x = 0.0

    def add_tick(position: float, label: str) -> None:
        displayed = _display_label(label)
        if tick_positions and np.isclose(
            position, tick_positions[-1], rtol=0.0, atol=1.0e-12
        ):
            existing = tick_labels[-1].split("|")
            if displayed not in existing:
                tick_labels[-1] = tick_labels[-1] + "|" + displayed
            return
        tick_positions.append(float(position))
        tick_labels.append(displayed)

    for start_label, start, stop_label, stop in segments:
        start_frac = np.asarray(start, dtype=np.float64)
        stop_frac = np.asarray(stop, dtype=np.float64)
        if not np.all(np.isfinite(start_frac)) or not np.all(np.isfinite(stop_frac)):
            raise ValueError(f"{source}: kpoint_path contains non-finite coordinates")
        delta_cart = (stop_frac - start_frac) @ reciprocal
        length = float(np.linalg.norm(delta_cart))
        if length <= np.finfo(np.float64).eps:
            raise ValueError(f"{source}: kpoint_path contains a zero-length segment")
        fraction = np.linspace(0.0, 1.0, count + 1)
        points = np.asarray(
            (1.0 - fraction[:, None]) * start_frac[None, :]
            + fraction[:, None] * stop_frac[None, :],
            dtype=np.float64,
        )
        x_value = current_x + fraction * length
        add_tick(current_x, str(start_label))
        current_x += length
        add_tick(current_x, str(stop_label))
        all_points.append(points)
        all_x.append(x_value)
        offsets.append(offsets[-1] + points.shape[0])
    return MagnonKPath(
        source=source,
        k_points_frac=np.concatenate(all_points, axis=0),
        x_coordinate_inv_ang=np.concatenate(all_x),
        segment_offsets=np.asarray(offsets, dtype=np.int64),
        tick_positions_inv_ang=np.asarray(tick_positions, dtype=np.float64),
        tick_labels=tuple(tick_labels),
        points_per_segment=count,
    )


def compute_magnon_dispersion(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    path: MagnonKPath,
    *,
    anisotropy: SingleIonAnisotropy | None = None,
    context: MPIContext | None = None,
    root: int = 0,
    goldstone_tolerance_mev: float = 1.0e-10,
) -> DistributedMagnonDispersion:
    """Distribute path points over MPI ranks and assemble energies on root."""

    physical_modes = configuration.n_magnetic_sites

    def solve(indices: NDArray[np.int64]) -> NDArray[np.float64]:
        if indices.size == 0:
            return np.empty((0, 2 * physical_modes), dtype=np.float64)
        dispersion = solve_isotropic_lswt_energies(
            exchange,
            configuration,
            path.k_points_frac[indices],
            anisotropy=anisotropy,
            goldstone_tolerance_mev=goldstone_tolerance_mev,
        )
        return np.concatenate(
            (
                dispersion.energy_mev,
                dispersion.goldstone_mask.astype(np.float64),
            ),
            axis=1,
        )

    distributed: DistributedArrayResult = distributed_array_map(
        path.n_points,
        solve,
        item_shape=(2 * physical_modes,),
        dtype=np.float64,
        context=context,
        root=root,
        stage="magph_dispersion_kpath",
    )
    result = None
    if distributed.global_values is not None:
        result = MagnonDispersionResult(
            path=path,
            energy_mev=distributed.global_values[:, :physical_modes],
            goldstone_mask=distributed.global_values[:, physical_modes:].astype(bool),
            mpi_size=distributed.mpi_size,
        )
    return DistributedMagnonDispersion(
        local_indices=distributed.local_indices,
        global_result=result,
        mpi_size=distributed.mpi_size,
        root_rank=distributed.root_rank,
    )


__all__ = [
    "DistributedMagnonDispersion",
    "MagnonDispersionResult",
    "MagnonKPath",
    "build_wannier90_kpath",
    "compute_magnon_dispersion",
]
