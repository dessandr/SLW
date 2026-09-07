"""MPI phonon renormalization from the exchange-striction magnon bubble."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from slw.cli.mpi import MPIContext

from .coupling import ModeResolvedExchangeDerivative
from .lifetime import linewidth_observables
from .mesh import MagnonMeshCache
from .model import ExchangeModel, MagneticConfiguration, MagneticOrder
from .parallel import DistributedArrayResult, distributed_array_map
from .phonon_self_energy import compute_phonon_self_energy_onshell
from .vertex import build_bare_isotropic_vertex_block


def _readonly(value: object, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PhononRenormalizationResult:
    """On-shell phonon poles and damping on an explicit q mesh."""

    q_points_frac: NDArray[np.float64]
    bare_frequency_mev: NDArray[np.float64]
    effective_bare_frequency_mev: NDArray[np.float64]
    self_energy_onshell_mev: NDArray[np.complex128]
    frequency_squared_mev2: NDArray[np.float64]
    renormalized_frequency_mev: NDArray[np.float64]
    frequency_shift_mev: NDArray[np.float64]
    raw_gamma_hwhm_mev: NDArray[np.float64]
    gamma_hwhm_mev: NDArray[np.float64]
    fwhm_mev: NDArray[np.float64]
    scattering_rate_ps_inv: NDArray[np.float64]
    lifetime_ps: NDArray[np.float64]
    dynamically_stable: NDArray[np.bool_]
    valid_damping: NDArray[np.bool_]
    frequency_regularized: NDArray[np.bool_]
    valid_renormalization: NDArray[np.bool_]
    temperature_k: float
    broadening_mev: float
    negative_tolerance_mev: float
    dyson_convention: str = "Omega^2=omega0^2+2*omega0*RePi(omega0)"
    approximation: str = "one_loop_magnon_bubble_onshell"

    def __post_init__(self) -> None:
        q_points = _readonly(self.q_points_frac, dtype=np.dtype(np.float64))
        bare = _readonly(self.bare_frequency_mev, dtype=np.dtype(np.float64))
        effective = _readonly(
            self.effective_bare_frequency_mev, dtype=np.dtype(np.float64)
        )
        if q_points.ndim != 2 or q_points.shape[1:] != (3,) or q_points.shape[0] < 1:
            raise ValueError("q_points_frac must have nonempty shape (nq,3)")
        if bare.ndim != 2 or bare.shape[0] != q_points.shape[0]:
            raise ValueError("bare_frequency_mev must have shape (nq,nmode)")
        if effective.shape != bare.shape or np.any(effective <= 0.0):
            raise ValueError(
                "effective_bare_frequency_mev must be positive and match "
                "bare frequencies"
            )
        arrays = {
            "self_energy_onshell_mev": (
                self.self_energy_onshell_mev,
                np.dtype(np.complex128),
            ),
            "frequency_squared_mev2": (
                self.frequency_squared_mev2,
                np.dtype(np.float64),
            ),
            "renormalized_frequency_mev": (
                self.renormalized_frequency_mev,
                np.dtype(np.float64),
            ),
            "frequency_shift_mev": (self.frequency_shift_mev, np.dtype(np.float64)),
            "raw_gamma_hwhm_mev": (self.raw_gamma_hwhm_mev, np.dtype(np.float64)),
            "gamma_hwhm_mev": (self.gamma_hwhm_mev, np.dtype(np.float64)),
            "fwhm_mev": (self.fwhm_mev, np.dtype(np.float64)),
            "scattering_rate_ps_inv": (
                self.scattering_rate_ps_inv,
                np.dtype(np.float64),
            ),
            "lifetime_ps": (self.lifetime_ps, np.dtype(np.float64)),
            "dynamically_stable": (self.dynamically_stable, np.dtype(bool)),
            "valid_damping": (self.valid_damping, np.dtype(bool)),
            "frequency_regularized": (self.frequency_regularized, np.dtype(bool)),
            "valid_renormalization": (self.valid_renormalization, np.dtype(bool)),
        }
        converted: dict[str, np.ndarray] = {}
        for name, (value, dtype) in arrays.items():
            array = _readonly(value, dtype=dtype)
            if array.shape != bare.shape:
                raise ValueError(f"{name} shape {array.shape} != {bare.shape}")
            converted[name] = array
        finite_required = (
            q_points,
            bare,
            effective,
            converted["self_energy_onshell_mev"],
            converted["frequency_squared_mev2"],
            converted["renormalized_frequency_mev"],
            converted["frequency_shift_mev"],
            converted["raw_gamma_hwhm_mev"],
        )
        if any(not np.all(np.isfinite(value)) for value in finite_required):
            raise ValueError("phonon renormalization contains non-finite core data")
        for name in ("temperature_k", "broadening_mev", "negative_tolerance_mev"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or (
                name == "broadening_mev" and value == 0.0
            ):
                qualifier = "positive" if name == "broadening_mev" else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "q_points_frac", q_points)
        object.__setattr__(self, "bare_frequency_mev", bare)
        object.__setattr__(self, "effective_bare_frequency_mev", effective)
        for name, value in converted.items():
            object.__setattr__(self, name, value)

    @property
    def has_dynamic_instability(self) -> bool:
        return bool(np.any(~self.dynamically_stable & ~self.frequency_regularized))


@dataclass(frozen=True)
class DistributedPhononRenormalizationResult:
    local_q_indices: NDArray[np.int64]
    global_result: PhononRenormalizationResult | None
    mpi_size: int
    root_rank: int


def _external_union_indices(cache: MagnonMeshCache) -> NDArray[np.int64]:
    return np.asarray(
        [cache.external_union_index(index) for index in range(cache.nk)],
        dtype=np.int64,
    )


def _band_vertices_for_q_block(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    coupling: ModeResolvedExchangeDerivative,
    cache: MagnonMeshCache,
    q_start: int,
    q_stop: int,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]:
    """Materialize one bounded q block over the complete magnon k mesh."""

    q_count = int(q_stop - q_start)
    vertices = np.empty(
        (cache.nk, q_count, coupling.nmode, cache.nchannel, cache.nchannel),
        dtype=np.complex128,
    )
    initial_energy = np.empty((cache.nk, cache.nchannel), dtype=np.float64)
    final_energy = np.empty(
        (cache.nk, q_count, cache.nchannel), dtype=np.float64
    )
    external_indices = _external_union_indices(cache)
    for k_index, external_index in enumerate(external_indices):
        k_point = cache.union_points_frac[external_index]
        internal_indices = cache.internal_union_indices(k_index, q_start, q_stop)
        bare = build_bare_isotropic_vertex_block(
            exchange,
            configuration,
            coupling,
            k_point,
            q_start,
            q_stop,
        )
        initial_transform = cache.transformation[external_index]
        final_transform = cache.transformation[internal_indices]
        vertices[k_index] = np.einsum(
            "qai,qmij,jb->qmab",
            np.swapaxes(final_transform.conj(), 1, 2),
            bare,
            initial_transform,
            optimize=True,
        )
        initial_energy[k_index] = cache.signed_energies_mev[external_index]
        final_energy[k_index] = cache.signed_energies_mev[internal_indices]
    return vertices, initial_energy, final_energy


def compute_phonon_renormalization_grid(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    coupling: ModeResolvedExchangeDerivative,
    magnon_cache: MagnonMeshCache,
    bare_frequency_mev: ArrayLike,
    *,
    temperature_k: float,
    broadening_mev: float,
    metric_energy_tolerance_mev: float = 0.0,
    negative_tolerance_mev: float = 0.0,
    q_chunk_size: int | None = None,
    mode_chunk_size: int | None = None,
    channel_chunk_size: int | None = None,
    context: MPIContext | None = None,
    root: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> DistributedPhononRenormalizationResult:
    """Compute one-loop phonon poles with MPI distribution over phonon q."""

    if (
        magnon_cache.nq != coupling.nq
        or magnon_cache.nchannel != configuration.n_channels
    ):
        raise ValueError("magnon cache does not match the coupling/configuration")
    expected_metric = (
        np.ones(configuration.n_channels, dtype=np.float64)
        if configuration.order is MagneticOrder.FM
        else np.concatenate(
            (
                np.ones(configuration.n_magnetic_sites),
                -np.ones(configuration.n_magnetic_sites),
            )
        )
    )
    if not np.array_equal(magnon_cache.metric, expected_metric):
        raise ValueError("magnon cache metric does not match the magnetic order")
    bare = np.asarray(bare_frequency_mev, dtype=np.float64)
    if bare.shape != coupling.phonon_energy_mev.shape:
        raise ValueError("bare_frequency_mev must match the coupling phonon grid")
    if not np.all(np.isfinite(bare)):
        raise ValueError("bare_frequency_mev contains non-finite values")
    if q_chunk_size is None:
        q_chunk = 1
    elif isinstance(q_chunk_size, bool) or not isinstance(
        q_chunk_size, Integral
    ):
        raise TypeError("q_chunk_size must be a positive integer or None")
    else:
        q_chunk = int(q_chunk_size)
        if q_chunk < 1:
            raise ValueError("q_chunk_size must be positive")

    def worker(indices: NDArray[np.int64]) -> NDArray[np.complex128]:
        local = np.empty((indices.size, coupling.nmode), dtype=np.complex128)
        if indices.size == 0:
            return local
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("phonon-q MPI ownership must be contiguous")
        for local_start in range(0, indices.size, q_chunk):
            local_stop = min(local_start + q_chunk, indices.size)
            q_start = int(indices[local_start])
            q_stop = int(indices[local_stop - 1]) + 1
            vertices, initial_energy, final_energy = _band_vertices_for_q_block(
                exchange,
                configuration,
                coupling,
                magnon_cache,
                q_start,
                q_stop,
            )
            for offset, q_index in enumerate(range(q_start, q_stop)):
                self_energy = compute_phonon_self_energy_onshell(
                    coupling.phonon_energy_mev[q_index],
                    vertices[:, offset],
                    initial_energy,
                    final_energy[:, offset],
                    temperature_k=temperature_k,
                    broadening_mev=broadening_mev,
                    metric=magnon_cache.metric,
                    metric_energy_tolerance_mev=metric_energy_tolerance_mev,
                    mode_chunk_size=mode_chunk_size,
                    channel_chunk_size=channel_chunk_size,
                )
                local[local_start + offset] = self_energy.sigma_onshell_mev
                if progress is not None:
                    progress(local_start + offset + 1, indices.size)
        return local

    mpi = MPIContext.discover() if context is None else context
    distributed: DistributedArrayResult = distributed_array_map(
        coupling.nq,
        worker,
        item_shape=(coupling.nmode,),
        dtype=np.complex128,
        context=mpi,
        root=root,
        stage="phonon_renormalization_q",
    )
    result: PhononRenormalizationResult | None = None
    if distributed.global_values is not None:
        sigma = np.asarray(distributed.global_values, dtype=np.complex128)
        effective = np.asarray(coupling.phonon_energy_mev, dtype=np.float64)
        squared = effective**2 + 2.0 * effective * np.real(sigma)
        stable = squared >= 0.0
        renormalized = np.sqrt(np.abs(squared))
        renormalized[~stable] *= -1.0
        shift = renormalized - effective
        damping = linewidth_observables(
            -np.imag(sigma),
            negative_tolerance_mev=negative_tolerance_mev,
        )
        regularized = np.asarray(coupling.frequency_regularized, dtype=bool)
        valid = stable & damping.valid_damping & ~regularized
        result = PhononRenormalizationResult(
            q_points_frac=coupling.q_points_frac,
            bare_frequency_mev=bare,
            effective_bare_frequency_mev=effective,
            self_energy_onshell_mev=sigma,
            frequency_squared_mev2=squared,
            renormalized_frequency_mev=renormalized,
            frequency_shift_mev=shift,
            raw_gamma_hwhm_mev=damping.raw_gamma_hwhm_mev,
            gamma_hwhm_mev=damping.gamma_hwhm_mev,
            fwhm_mev=damping.fwhm_mev,
            scattering_rate_ps_inv=damping.scattering_rate_ps_inv,
            lifetime_ps=damping.lifetime_ps,
            dynamically_stable=stable,
            valid_damping=damping.valid_damping,
            frequency_regularized=regularized,
            valid_renormalization=valid,
            temperature_k=temperature_k,
            broadening_mev=broadening_mev,
            negative_tolerance_mev=negative_tolerance_mev,
        )
    return DistributedPhononRenormalizationResult(
        local_q_indices=distributed.local_indices,
        global_result=result,
        mpi_size=distributed.mpi_size,
        root_rank=distributed.root_rank,
    )


__all__ = [
    "DistributedPhononRenormalizationResult",
    "PhononRenormalizationResult",
    "compute_phonon_renormalization_grid",
]
