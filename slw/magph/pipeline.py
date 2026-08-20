"""End-to-end native self-energy and lifetime orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from slw.cli.mpi import MPIContext

from .coupling import ModeResolvedExchangeDerivative
from .lifetime import (
    LifetimeResult,
    lifetime_from_onshell_self_energy,
    linewidth_observables,
)
from .model import ExchangeModel, MagneticConfiguration
from .parallel import DistributedArrayResult, distributed_array_map
from .self_energy import OnShellSelfEnergyResult, compute_onshell_self_energy_diagonal
from .vertex import MagnonPhononScatteringProblem, build_scattering_problem


def _readonly(value: object, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LifetimeAtK:
    problem: MagnonPhononScatteringProblem
    self_energy: OnShellSelfEnergyResult
    lifetime: LifetimeResult

    @property
    def physical_sigma_mev(self) -> NDArray[np.complex128]:
        values = self.self_energy.sigma_diagonal_mev[: self.problem.physical_mode_count]
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class LifetimeGridResult:
    k_points_frac: NDArray[np.float64]
    energy_mev: NDArray[np.float64]
    self_energy_onshell_mev: NDArray[np.complex128]
    gamma_hwhm_mev: NDArray[np.float64]
    fwhm_mev: NDArray[np.float64]
    scattering_rate_ps_inv: NDArray[np.float64]
    lifetime_ps: NDArray[np.float64]
    valid_damping: NDArray[np.bool_]
    temperature_k: float
    broadening_mev: float
    negative_tolerance_mev: float

    def __post_init__(self) -> None:
        k_points = _readonly(self.k_points_frac, dtype=np.dtype(np.float64))
        energy = _readonly(self.energy_mev, dtype=np.dtype(np.float64))
        sigma = _readonly(self.self_energy_onshell_mev, dtype=np.dtype(np.complex128))
        gamma = _readonly(self.gamma_hwhm_mev, dtype=np.dtype(np.float64))
        fwhm = _readonly(self.fwhm_mev, dtype=np.dtype(np.float64))
        rate = _readonly(self.scattering_rate_ps_inv, dtype=np.dtype(np.float64))
        lifetime = _readonly(self.lifetime_ps, dtype=np.dtype(np.float64))
        valid = _readonly(self.valid_damping, dtype=np.dtype(bool))
        if k_points.ndim != 2 or k_points.shape[1:] != (3,) or k_points.shape[0] == 0:
            raise ValueError("k_points_frac must have nonempty shape (nk,3)")
        if energy.ndim != 2 or energy.shape[0] != k_points.shape[0]:
            raise ValueError("energy_mev must have shape (nk,nphysical)")
        expected = energy.shape
        for name, array in (
            ("self_energy_onshell_mev", sigma),
            ("gamma_hwhm_mev", gamma),
            ("fwhm_mev", fwhm),
            ("scattering_rate_ps_inv", rate),
            ("lifetime_ps", lifetime),
            ("valid_damping", valid),
        ):
            if array.shape != expected:
                raise ValueError(f"{name} shape {array.shape} != {expected}")
        if not np.all(np.isfinite(k_points)) or not np.all(np.isfinite(energy)):
            raise ValueError("lifetime grid contains non-finite k points or energies")
        if not np.all(np.isfinite(sigma)):
            raise ValueError("lifetime grid contains non-finite self-energy")
        for name, value, positive in (
            ("temperature_k", self.temperature_k, False),
            ("broadening_mev", self.broadening_mev, True),
            ("negative_tolerance_mev", self.negative_tolerance_mev, False),
        ):
            numeric = float(value)
            if not np.isfinite(numeric) or (
                numeric <= 0.0 if positive else numeric < 0.0
            ):
                comparator = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {comparator}")
            object.__setattr__(self, name, numeric)
        object.__setattr__(self, "k_points_frac", k_points)
        object.__setattr__(self, "energy_mev", energy)
        object.__setattr__(self, "self_energy_onshell_mev", sigma)
        object.__setattr__(self, "gamma_hwhm_mev", gamma)
        object.__setattr__(self, "fwhm_mev", fwhm)
        object.__setattr__(self, "scattering_rate_ps_inv", rate)
        object.__setattr__(self, "lifetime_ps", lifetime)
        object.__setattr__(self, "valid_damping", valid)


@dataclass(frozen=True)
class DistributedLifetimeGridResult:
    local_indices: NDArray[np.int64]
    global_result: LifetimeGridResult | None
    mpi_size: int
    root_rank: int


def evaluate_scattering_problem(
    problem: MagnonPhononScatteringProblem,
    *,
    temperature_k: float,
    broadening_mev: float,
    q_weights: ArrayLike | None = None,
    metric_energy_tolerance_mev: float = 0.0,
    negative_tolerance_mev: float = 0.0,
    q_chunk_size: int | None = None,
    channel_chunk_size: int | None = None,
) -> LifetimeAtK:
    """Evaluate one on-shell retarded self-energy and linewidth convention."""

    self_energy = compute_onshell_self_energy_diagonal(
        problem.external_energy_mev,
        problem.vertex_mev,
        problem.internal_energy_mev,
        problem.phonon_energy_mev,
        temperature_k=temperature_k,
        broadening_mev=broadening_mev,
        metric=problem.metric,
        metric_energy_tolerance_mev=metric_energy_tolerance_mev,
        q_weights=q_weights,
        q_chunk_size=q_chunk_size,
        channel_chunk_size=channel_chunk_size,
    )
    lifetime = lifetime_from_onshell_self_energy(
        self_energy.sigma_diagonal_mev,
        negative_tolerance_mev=negative_tolerance_mev,
    )
    return LifetimeAtK(problem=problem, self_energy=self_energy, lifetime=lifetime)


def _grid_from_packed(
    k_points: NDArray[np.float64],
    packed: NDArray[np.float64],
    *,
    temperature_k: float,
    broadening_mev: float,
    negative_tolerance_mev: float,
) -> LifetimeGridResult:
    energy = packed[..., 0]
    sigma = packed[..., 1] + 1.0j * packed[..., 2]
    observables = linewidth_observables(
        -np.imag(sigma),
        negative_tolerance_mev=negative_tolerance_mev,
    )
    return LifetimeGridResult(
        k_points_frac=k_points,
        energy_mev=energy,
        self_energy_onshell_mev=sigma,
        gamma_hwhm_mev=observables.gamma_hwhm_mev,
        fwhm_mev=observables.fwhm_mev,
        scattering_rate_ps_inv=observables.scattering_rate_ps_inv,
        lifetime_ps=observables.lifetime_ps,
        valid_damping=observables.valid_damping,
        temperature_k=temperature_k,
        broadening_mev=broadening_mev,
        negative_tolerance_mev=negative_tolerance_mev,
    )


def compute_lifetime_grid(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    coupling: ModeResolvedExchangeDerivative,
    k_points_frac: ArrayLike,
    *,
    temperature_k: float,
    broadening_mev: float,
    q_weights: ArrayLike | None = None,
    metric_energy_tolerance_mev: float = 0.0,
    negative_tolerance_mev: float = 0.0,
    vertex_q_chunk_size: int | None = None,
    self_energy_q_chunk_size: int | None = None,
    channel_chunk_size: int | None = None,
    lswt_options: dict[str, float] | None = None,
    context: MPIContext | None = None,
    root: int = 0,
) -> DistributedLifetimeGridResult:
    """Compute physical magnon lifetimes with external-k MPI distribution.

    MPI discovery is the default.  Each rank owns a contiguous k block and
    keeps q/mode contractions vectorized locally.  Only root receives the
    assembled scientific grid.
    """

    k_points = np.asarray(k_points_frac, dtype=np.float64)
    if k_points.ndim != 2 or k_points.shape[1:] != (3,) or k_points.shape[0] == 0:
        raise ValueError("k_points_frac must have nonempty shape (nk,3)")
    if not np.all(np.isfinite(k_points)):
        raise ValueError("k_points_frac contains non-finite values")
    physical_count = configuration.n_magnetic_sites

    def worker(indices: NDArray[np.int64]) -> NDArray[np.float64]:
        packed = np.empty((indices.size, physical_count, 3), dtype=np.float64)
        for local_index, global_index in enumerate(indices):
            problem = build_scattering_problem(
                exchange,
                configuration,
                coupling,
                k_points[int(global_index)],
                q_chunk_size=vertex_q_chunk_size,
                lswt_options=lswt_options,
            )
            evaluated = evaluate_scattering_problem(
                problem,
                temperature_k=temperature_k,
                broadening_mev=broadening_mev,
                q_weights=q_weights,
                metric_energy_tolerance_mev=metric_energy_tolerance_mev,
                negative_tolerance_mev=negative_tolerance_mev,
                q_chunk_size=self_energy_q_chunk_size,
                channel_chunk_size=channel_chunk_size,
            )
            sigma = evaluated.physical_sigma_mev
            packed[local_index, :, 0] = problem.external_energy_mev[:physical_count]
            packed[local_index, :, 1] = sigma.real
            packed[local_index, :, 2] = sigma.imag
        return packed

    distributed: DistributedArrayResult = distributed_array_map(
        k_points.shape[0],
        worker,
        item_shape=(physical_count, 3),
        dtype=np.float64,
        context=context,
        root=root,
        stage="magph_lifetime_external_k",
    )
    global_result = (
        None
        if distributed.global_values is None
        else _grid_from_packed(
            k_points,
            distributed.global_values,
            temperature_k=temperature_k,
            broadening_mev=broadening_mev,
            negative_tolerance_mev=negative_tolerance_mev,
        )
    )
    return DistributedLifetimeGridResult(
        local_indices=distributed.local_indices,
        global_result=global_result,
        mpi_size=distributed.mpi_size,
        root_rank=distributed.root_rank,
    )


__all__ = [
    "DistributedLifetimeGridResult",
    "LifetimeAtK",
    "LifetimeGridResult",
    "compute_lifetime_grid",
    "evaluate_scattering_problem",
]
