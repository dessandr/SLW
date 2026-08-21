"""Reusable magnon eigensystems on the uniform ``k+q`` union mesh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from slw.cli.mpi import MPIContext

from .coupling import ModeResolvedExchangeDerivative
from .lswt import solve_isotropic_lswt, uniform_fractional_mesh
from .model import (
    ExchangeModel,
    MagneticConfiguration,
    MagneticOrder,
    SingleIonAnisotropy,
)
from .parallel import distributed_array_map


def _readonly(value: object, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _mesh_tuple(value: ArrayLike, *, name: str) -> tuple[int, int, int]:
    raw = np.asarray(value)
    numeric = np.asarray(raw, dtype=np.float64)
    if (
        numeric.shape != (3,)
        or not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, np.rint(numeric))
        or np.any(numeric <= 0.0)
    ):
        raise ValueError(f"{name} must contain three positive integers")
    return (int(numeric[0]), int(numeric[1]), int(numeric[2]))


def _grid_indices(
    points_frac: NDArray[np.float64],
    mesh: tuple[int, int, int],
    *,
    shift_grid: NDArray[np.float64],
    tolerance: float,
    name: str,
) -> NDArray[np.int64]:
    dimensions = np.asarray(mesh, dtype=np.int64)
    scaled = points_frac * dimensions[None, :] - shift_grid[None, :]
    nearest = np.rint(scaled)
    periodic_residual = scaled - nearest
    if float(np.max(np.abs(periodic_residual), initial=0.0)) > tolerance:
        raise ValueError(f"{name} points are not commensurate with mesh {mesh}")
    result = np.mod(nearest.astype(np.int64), dimensions[None, :])
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MagnonMeshCache:
    """Magnon energies/transforms indexed on the exact uniform ``k+q`` mesh."""

    union_points_frac: NDArray[np.float64]
    signed_energies_mev: NDArray[np.float64]
    transformation: NDArray[np.complex128]
    metric: NDArray[np.float64]
    physical_mode_count: int
    k_mesh_shape: tuple[int, int, int]
    q_mesh_shape: tuple[int, int, int]
    union_mesh_shape: tuple[int, int, int]
    kshift_grid: tuple[float, float, float]
    external_k_grid_indices: NDArray[np.int64]
    q_grid_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        points = _readonly(self.union_points_frac, dtype=np.dtype(np.float64))
        energies = _readonly(self.signed_energies_mev, dtype=np.dtype(np.float64))
        transformation = _readonly(self.transformation, dtype=np.dtype(np.complex128))
        metric = _readonly(self.metric, dtype=np.dtype(np.float64))
        external_indices = _readonly(
            self.external_k_grid_indices, dtype=np.dtype(np.int64)
        )
        q_indices = _readonly(self.q_grid_indices, dtype=np.dtype(np.int64))
        k_mesh = _mesh_tuple(self.k_mesh_shape, name="k_mesh_shape")
        q_mesh = _mesh_tuple(self.q_mesh_shape, name="q_mesh_shape")
        union_mesh = _mesh_tuple(self.union_mesh_shape, name="union_mesh_shape")
        expected_union = tuple(
            int(np.lcm(k_value, q_value))
            for k_value, q_value in zip(k_mesh, q_mesh, strict=True)
        )
        if union_mesh != expected_union:
            raise ValueError(
                f"union_mesh_shape {union_mesh} != lcm(k,q) {expected_union}"
            )
        if points.shape != (int(np.prod(union_mesh)), 3):
            raise ValueError("union_points_frac has the wrong union-mesh shape")
        if energies.ndim != 2 or energies.shape[0] != points.shape[0]:
            raise ValueError("signed_energies_mev must have shape (nunion,nchannel)")
        channels = energies.shape[1]
        if transformation.shape != (points.shape[0], channels, channels):
            raise ValueError(
                "transformation must have shape (nunion,nchannel,nchannel)"
            )
        if metric.shape != (channels,) or not np.all(np.isin(metric, (-1.0, 1.0))):
            raise ValueError("metric must contain one +/-1 value per channel")
        physical = int(self.physical_mode_count)
        if physical < 1 or physical > channels:
            raise ValueError("physical_mode_count is outside the channel range")
        if external_indices.ndim != 2 or external_indices.shape[1:] != (3,):
            raise ValueError("external_k_grid_indices must have shape (nk,3)")
        if q_indices.shape != (int(np.prod(q_mesh)), 3):
            raise ValueError("q_grid_indices must contain one complete q mesh")
        if np.unique(external_indices, axis=0).shape[0] != external_indices.shape[0]:
            raise ValueError("external k mesh contains duplicate grid indices")
        if np.unique(q_indices, axis=0).shape[0] != q_indices.shape[0]:
            raise ValueError("q mesh contains duplicate grid indices")
        for indices, mesh, name in (
            (external_indices, k_mesh, "external k"),
            (q_indices, q_mesh, "q"),
        ):
            dimensions = np.asarray(mesh, dtype=np.int64)
            if np.any(indices < 0) or np.any(indices >= dimensions[None, :]):
                raise ValueError(f"{name} grid indices are outside their mesh")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(energies)):
            raise ValueError("magnon mesh cache contains non-finite real data")
        if not np.all(np.isfinite(transformation)):
            raise ValueError("magnon mesh cache contains non-finite transformations")
        kshift = tuple(float(item) for item in self.kshift_grid)
        if len(kshift) != 3 or not np.all(np.isfinite(kshift)):
            raise ValueError("kshift_grid must contain three finite values")
        object.__setattr__(self, "union_points_frac", points)
        object.__setattr__(self, "signed_energies_mev", energies)
        object.__setattr__(self, "transformation", transformation)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "physical_mode_count", physical)
        object.__setattr__(self, "k_mesh_shape", k_mesh)
        object.__setattr__(self, "q_mesh_shape", q_mesh)
        object.__setattr__(self, "union_mesh_shape", union_mesh)
        object.__setattr__(self, "kshift_grid", kshift)
        object.__setattr__(self, "external_k_grid_indices", external_indices)
        object.__setattr__(self, "q_grid_indices", q_indices)

    @property
    def nchannel(self) -> int:
        return int(self.metric.size)

    @property
    def nk(self) -> int:
        return int(self.external_k_grid_indices.shape[0])

    @property
    def nq(self) -> int:
        return int(self.q_grid_indices.shape[0])

    def external_union_index(self, external_k_index: int) -> int:
        index = int(external_k_index)
        if index < 0 or index >= self.nk:
            raise IndexError("external k index is outside the cached mesh")
        union = np.asarray(self.union_mesh_shape, dtype=np.int64)
        k_mesh = np.asarray(self.k_mesh_shape, dtype=np.int64)
        coordinates = self.external_k_grid_indices[index] * (union // k_mesh)
        return int(np.ravel_multi_index(tuple(coordinates), self.union_mesh_shape))

    def internal_union_indices(
        self,
        external_k_index: int,
        q_start: int = 0,
        q_stop: int | None = None,
    ) -> NDArray[np.int64]:
        index = int(external_k_index)
        if index < 0 or index >= self.nk:
            raise IndexError("external k index is outside the cached mesh")
        start = int(q_start)
        stop = self.nq if q_stop is None else int(q_stop)
        if start < 0 or stop < start or stop > self.nq:
            raise IndexError("q slice is outside the cached mesh")
        union = np.asarray(self.union_mesh_shape, dtype=np.int64)
        k_mesh = np.asarray(self.k_mesh_shape, dtype=np.int64)
        q_mesh = np.asarray(self.q_mesh_shape, dtype=np.int64)
        coordinates = np.mod(
            self.external_k_grid_indices[index][None, :] * (union // k_mesh)[None, :]
            + self.q_grid_indices[start:stop] * (union // q_mesh)[None, :],
            union[None, :],
        )
        multi_index = (
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
        )
        flattened = np.ravel_multi_index(multi_index, self.union_mesh_shape)
        flattened = np.asarray(flattened, dtype=np.int64)
        flattened.setflags(write=False)
        return flattened


def build_magnon_mesh_cache(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    coupling: ModeResolvedExchangeDerivative,
    k_points_frac: ArrayLike,
    *,
    k_mesh_shape: tuple[int, int, int],
    kshift_grid: ArrayLike,
    context: MPIContext | None = None,
    root: int = 0,
    mesh_tolerance: float = 1.0e-8,
    lswt_options: dict[str, float] | None = None,
    anisotropy: SingleIonAnisotropy | None = None,
) -> MagnonMeshCache:
    """Solve each unique uniform ``k+q`` point once and broadcast the cache."""

    if coupling.q_mesh_shape is None:
        raise ValueError("optimized magnon caching requires coupling.q_mesh_shape")
    tolerance = float(mesh_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("mesh_tolerance must be finite and non-negative")
    k_mesh = _mesh_tuple(k_mesh_shape, name="k_mesh_shape")
    q_mesh = _mesh_tuple(coupling.q_mesh_shape, name="q_mesh_shape")
    kshift_raw = np.asarray(kshift_grid, dtype=np.float64)
    if kshift_raw.shape != (3,) or not np.all(np.isfinite(kshift_raw)):
        raise ValueError("kshift_grid must contain three finite values")
    k_points = np.asarray(k_points_frac, dtype=np.float64)
    if k_points.shape != (int(np.prod(k_mesh)), 3):
        raise ValueError("k_points_frac must contain one complete external k mesh")
    if not np.all(np.isfinite(k_points)):
        raise ValueError("k_points_frac contains non-finite values")
    external_indices = _grid_indices(
        k_points,
        k_mesh,
        shift_grid=kshift_raw,
        tolerance=tolerance,
        name="external k",
    )
    q_indices = _grid_indices(
        coupling.q_points_frac,
        q_mesh,
        shift_grid=np.zeros(3, dtype=np.float64),
        tolerance=tolerance,
        name="phonon q",
    )
    union_mesh = (
        int(np.lcm(k_mesh[0], q_mesh[0])),
        int(np.lcm(k_mesh[1], q_mesh[1])),
        int(np.lcm(k_mesh[2], q_mesh[2])),
    )
    union_shift = kshift_raw * (
        np.asarray(union_mesh, dtype=np.float64) / np.asarray(k_mesh, dtype=np.float64)
    )
    union_points = uniform_fractional_mesh(union_mesh, shift=union_shift)
    channels = configuration.n_channels
    options = {} if lswt_options is None else dict(lswt_options)

    def solve(indices: NDArray[np.int64]) -> NDArray[np.complex128]:
        if indices.size == 0:
            return np.empty((0, channels + channels * channels), dtype=np.complex128)
        spectrum = solve_isotropic_lswt(
            exchange,
            configuration,
            union_points[indices],
            anisotropy=anisotropy,
            **options,
        )
        packed = np.empty(
            (indices.size, channels + channels * channels), dtype=np.complex128
        )
        packed[:, :channels] = spectrum.signed_energies_mev
        packed[:, channels:] = spectrum.transformation.reshape(indices.size, -1)
        return packed

    distributed = distributed_array_map(
        union_points.shape[0],
        solve,
        item_shape=(channels + channels * channels,),
        dtype=np.complex128,
        context=context,
        root=root,
        broadcast_result=True,
        stage="magph_union_lswt",
    )
    if distributed.global_values is None:  # pragma: no cover - broadcast contract
        raise RuntimeError("union-mesh LSWT cache was not broadcast")
    packed = distributed.global_values
    energies = np.asarray(packed[:, :channels].real, dtype=np.float64)
    transformation = np.asarray(packed[:, channels:], dtype=np.complex128).reshape(
        union_points.shape[0], channels, channels
    )
    metric = (
        np.ones(channels, dtype=np.float64)
        if configuration.order is MagneticOrder.FM
        else np.concatenate(
            (
                np.ones(configuration.n_magnetic_sites, dtype=np.float64),
                -np.ones(configuration.n_magnetic_sites, dtype=np.float64),
            )
        )
    )
    return MagnonMeshCache(
        union_points_frac=union_points,
        signed_energies_mev=energies,
        transformation=transformation,
        metric=metric,
        physical_mode_count=configuration.n_magnetic_sites,
        k_mesh_shape=k_mesh,
        q_mesh_shape=q_mesh,
        union_mesh_shape=union_mesh,
        kshift_grid=(
            float(kshift_raw[0]),
            float(kshift_raw[1]),
            float(kshift_raw[2]),
        ),
        external_k_grid_indices=external_indices,
        q_grid_indices=q_indices,
    )


__all__ = ["MagnonMeshCache", "build_magnon_mesh_cache"]
