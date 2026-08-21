"""Native isotropic magnon--phonon scattering vertices."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .coupling import ModeResolvedExchangeDerivative
from .lswt import _local_frames, solve_isotropic_lswt
from .model import (
    ExchangeModel,
    ExchangeRepresentation,
    ExchangeSpinNormalization,
    MagneticConfiguration,
    MagneticOrder,
    SingleIonAnisotropy,
)


def _readonly(value: object, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MagnonPhononScatteringProblem:
    """One external-k self-energy problem in the magnon band basis."""

    external_k_frac: NDArray[np.float64]
    internal_kq_frac: NDArray[np.float64]
    external_energy_mev: NDArray[np.float64]
    internal_energy_mev: NDArray[np.float64]
    phonon_energy_mev: NDArray[np.float64]
    vertex_mev: NDArray[np.complex128]
    metric: NDArray[np.float64]
    external_transformation: NDArray[np.complex128]
    internal_transformation: NDArray[np.complex128]
    physical_mode_count: int

    def __post_init__(self) -> None:
        external_k = _readonly(self.external_k_frac, dtype=np.dtype(np.float64))
        internal_k = _readonly(self.internal_kq_frac, dtype=np.dtype(np.float64))
        external_energy = _readonly(
            self.external_energy_mev, dtype=np.dtype(np.float64)
        )
        internal_energy = _readonly(
            self.internal_energy_mev, dtype=np.dtype(np.float64)
        )
        phonon_energy = _readonly(self.phonon_energy_mev, dtype=np.dtype(np.float64))
        vertex = _readonly(self.vertex_mev, dtype=np.dtype(np.complex128))
        metric = _readonly(self.metric, dtype=np.dtype(np.float64))
        external_transform = _readonly(
            self.external_transformation, dtype=np.dtype(np.complex128)
        )
        internal_transform = _readonly(
            self.internal_transformation, dtype=np.dtype(np.complex128)
        )
        if external_k.shape != (3,):
            raise ValueError("external_k_frac must have shape (3,)")
        if internal_k.ndim != 2 or internal_k.shape[1:] != (3,):
            raise ValueError("internal_kq_frac must have shape (nq,3)")
        if external_energy.ndim != 1:
            raise ValueError("external_energy_mev must have shape (nchannel,)")
        channels = external_energy.size
        if internal_energy.shape != (internal_k.shape[0], channels):
            raise ValueError("internal_energy_mev must have shape (nq,nchannel)")
        if phonon_energy.ndim != 2 or phonon_energy.shape[0] != internal_k.shape[0]:
            raise ValueError("phonon_energy_mev must have shape (nq,nmode)")
        if vertex.shape != (
            internal_k.shape[0],
            phonon_energy.shape[1],
            channels,
            channels,
        ):
            raise ValueError(
                "vertex_mev must have shape (nq,nmode,ninternal,nexternal)"
            )
        if metric.shape != (channels,) or not np.all(np.isin(metric, (-1.0, 1.0))):
            raise ValueError("metric must contain one +/-1 entry per channel")
        if external_transform.shape != (channels, channels):
            raise ValueError("external_transformation has the wrong shape")
        if internal_transform.shape != (internal_k.shape[0], channels, channels):
            raise ValueError("internal_transformation has the wrong shape")
        physical = int(self.physical_mode_count)
        if physical < 1 or physical > channels:
            raise ValueError("physical_mode_count is outside the channel range")
        for name, array in (
            ("external_k_frac", external_k),
            ("internal_kq_frac", internal_k),
            ("external_energy_mev", external_energy),
            ("internal_energy_mev", internal_energy),
            ("phonon_energy_mev", phonon_energy),
            ("vertex_mev", vertex),
            ("external_transformation", external_transform),
            ("internal_transformation", internal_transform),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")
        object.__setattr__(self, "external_k_frac", external_k)
        object.__setattr__(self, "internal_kq_frac", internal_k)
        object.__setattr__(self, "external_energy_mev", external_energy)
        object.__setattr__(self, "internal_energy_mev", internal_energy)
        object.__setattr__(self, "phonon_energy_mev", phonon_energy)
        object.__setattr__(self, "vertex_mev", vertex)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "external_transformation", external_transform)
        object.__setattr__(self, "internal_transformation", internal_transform)
        object.__setattr__(self, "physical_mode_count", physical)


def _unit_exchange_coefficients(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    frames = _local_frames(configuration)
    local_unit = np.einsum(
        "bai,baj->bij",
        frames[exchange.bond_i],
        frames[exchange.bond_j],
        optimize=True,
    )
    spin_i = configuration.spin_magnitudes[exchange.bond_i]
    spin_j = configuration.spin_magnitudes[exchange.bond_j]
    hp_i = np.sqrt(spin_i / 2.0)[:, None] * np.asarray(
        (1.0, -1.0j, 0.0), dtype=np.complex128
    )
    hp_j = np.sqrt(spin_j / 2.0)[:, None] * np.asarray(
        (1.0, -1.0j, 0.0), dtype=np.complex128
    )
    weight = exchange.convention.directed_bond_weight
    normal_ij = -weight * np.einsum(
        "bi,bij,bj->b", hp_i.conj(), local_unit, hp_j, optimize=True
    )
    normal_ji = -weight * np.einsum(
        "bi,bij,bj->b", hp_i, local_unit, hp_j.conj(), optimize=True
    )
    pair_create = -weight * np.einsum(
        "bi,bij,bj->b", hp_i.conj(), local_unit, hp_j.conj(), optimize=True
    )
    pair_annihilate = -weight * np.einsum(
        "bi,bij,bj->b", hp_i, local_unit, hp_j, optimize=True
    )
    longitudinal_i = weight * local_unit[:, 2, 2] * spin_j
    longitudinal_j = weight * local_unit[:, 2, 2] * spin_i
    if exchange.convention.spin_normalization is ExchangeSpinNormalization.UNIT_VECTOR:
        scale = 1.0 / (spin_i * spin_j)
        normal_ij *= scale
        normal_ji *= scale
        pair_create *= scale
        pair_annihilate *= scale
        longitudinal_i *= scale
        longitudinal_j *= scale
    return (
        normal_ij,
        normal_ji,
        pair_create,
        pair_annihilate,
        longitudinal_i,
        longitudinal_j,
    )


def _validate_vertex_inputs(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    coupling: ModeResolvedExchangeDerivative,
    external_k_frac: ArrayLike,
) -> tuple[NDArray[np.float64], int]:
    if exchange.representation is not ExchangeRepresentation.ISOTROPIC:
        raise NotImplementedError(
            "native bare vertex currently supports isotropic exchange"
        )
    if coupling.n_bonds != exchange.n_bonds:
        raise ValueError("mode-resolved derivative and exchange bond counts differ")
    k = np.asarray(external_k_frac, dtype=np.float64)
    if k.shape != (3,) or not np.all(np.isfinite(k)):
        raise ValueError("external_k_frac must be a finite length-3 vector")
    n_site = exchange.n_magnetic_sites
    n_channel = n_site if configuration.order is MagneticOrder.FM else 2 * n_site
    if configuration.n_channels != n_channel:
        raise ValueError("magnetic configuration has an inconsistent channel count")
    return k, n_channel


def _build_bare_isotropic_vertex_block(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    coupling: ModeResolvedExchangeDerivative,
    k: NDArray[np.float64],
    q_start: int,
    q_stop: int,
    coefficients: tuple[
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> NDArray[np.complex128]:
    n_site = exchange.n_magnetic_sites
    n_channel = configuration.n_channels
    (
        normal_ij,
        normal_ji,
        pair_create,
        pair_annihilate,
        longitudinal_i,
        longitudinal_j,
    ) = coefficients
    shifts = exchange.cell_shift.astype(np.float64, copy=False)
    phase_k = np.exp(2.0j * np.pi * (shifts @ k))
    q_points = coupling.q_points_frac[q_start:q_stop]
    phase_minus_q = np.exp(-2.0j * np.pi * (q_points @ shifts.T))
    phase_minus_kq = phase_minus_q * phase_k.conj()[None, :]
    lambda_chunk = coupling.lambda_mev[q_start:q_stop]
    block = np.zeros(
        (q_stop - q_start, coupling.nmode, n_channel, n_channel),
        dtype=np.complex128,
    )
    for bond in range(exchange.n_bonds):
        i = int(exchange.bond_i[bond])
        j = int(exchange.bond_j[bond])
        amplitude = lambda_chunk[:, :, bond]
        block[:, :, i, i] += amplitude * longitudinal_i[bond]
        block[:, :, j, j] += (
            amplitude * longitudinal_j[bond] * phase_minus_q[:, bond, None]
        )
        block[:, :, i, j] += amplitude * normal_ij[bond] * phase_k[bond]
        block[:, :, j, i] += amplitude * normal_ji[bond] * phase_minus_kq[:, bond, None]
        if configuration.order is MagneticOrder.COLLINEAR_AFM:
            block[:, :, n_site + i, n_site + i] += amplitude * longitudinal_i[bond]
            block[:, :, n_site + j, n_site + j] += (
                amplitude * longitudinal_j[bond] * phase_minus_q[:, bond, None]
            )
            block[:, :, n_site + i, n_site + j] += (
                amplitude * normal_ji[bond] * phase_k[bond]
            )
            block[:, :, n_site + j, n_site + i] += (
                amplitude * normal_ij[bond] * phase_minus_kq[:, bond, None]
            )
            block[:, :, i, n_site + j] += amplitude * pair_create[bond] * phase_k[bond]
            block[:, :, j, n_site + i] += (
                amplitude * pair_create[bond] * phase_minus_kq[:, bond, None]
            )
            block[:, :, n_site + i, j] += (
                amplitude * pair_annihilate[bond] * phase_k[bond]
            )
            block[:, :, n_site + j, i] += (
                amplitude * pair_annihilate[bond] * phase_minus_kq[:, bond, None]
            )
    return block


def build_bare_isotropic_vertex_block(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    coupling: ModeResolvedExchangeDerivative,
    external_k_frac: ArrayLike,
    q_start: int,
    q_stop: int,
) -> NDArray[np.complex128]:
    """Build one contiguous q block of the cell-gauge site/Nambu vertex."""

    k, _n_channel = _validate_vertex_inputs(
        exchange, configuration, coupling, external_k_frac
    )
    start = int(q_start)
    stop = int(q_stop)
    if start < 0 or stop <= start or stop > coupling.nq:
        raise IndexError("q block is outside the mode-resolved coupling")
    block = _build_bare_isotropic_vertex_block(
        exchange,
        configuration,
        coupling,
        k,
        start,
        stop,
        _unit_exchange_coefficients(exchange, configuration),
    )
    block.setflags(write=False)
    return block


def build_bare_isotropic_vertex(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    coupling: ModeResolvedExchangeDerivative,
    external_k_frac: ArrayLike,
    *,
    q_chunk_size: int | None = None,
) -> NDArray[np.complex128]:
    """Build the cell-gauge site/Nambu vertex for all phonon q and modes."""

    _k, n_channel = _validate_vertex_inputs(
        exchange, configuration, coupling, external_k_frac
    )
    q_chunk = coupling.nq if q_chunk_size is None else int(q_chunk_size)
    if q_chunk < 1:
        raise ValueError("q_chunk_size must be positive")
    output = np.zeros(
        (coupling.nq, coupling.nmode, n_channel, n_channel), dtype=np.complex128
    )
    for q_start, q_stop, block in iter_bare_isotropic_vertex_blocks(
        exchange,
        configuration,
        coupling,
        external_k_frac,
        q_chunk_size=q_chunk,
    ):
        output[q_start:q_stop] = block
    output.setflags(write=False)
    return output


def iter_bare_isotropic_vertex_blocks(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    coupling: ModeResolvedExchangeDerivative,
    external_k_frac: ArrayLike,
    *,
    q_chunk_size: int,
) -> Iterator[tuple[int, int, NDArray[np.complex128]]]:
    """Yield bare q blocks while reusing static bond coefficients."""

    k, _n_channel = _validate_vertex_inputs(
        exchange, configuration, coupling, external_k_frac
    )
    q_chunk = int(q_chunk_size)
    if q_chunk < 1:
        raise ValueError("q_chunk_size must be positive")
    coefficients = _unit_exchange_coefficients(exchange, configuration)
    for q_start in range(0, coupling.nq, q_chunk):
        q_stop = min(q_start + q_chunk, coupling.nq)
        block = _build_bare_isotropic_vertex_block(
            exchange,
            configuration,
            coupling,
            k,
            q_start,
            q_stop,
            coefficients,
        )
        block.setflags(write=False)
        yield q_start, q_stop, block


def build_scattering_problem(
    exchange: ExchangeModel,
    configuration: MagneticConfiguration,
    coupling: ModeResolvedExchangeDerivative,
    external_k_frac: ArrayLike,
    *,
    q_chunk_size: int | None = None,
    lswt_options: dict[str, float] | None = None,
    anisotropy: SingleIonAnisotropy | None = None,
) -> MagnonPhononScatteringProblem:
    """Solve the external/internal magnons and rotate one vertex to band space."""

    k = np.asarray(external_k_frac, dtype=np.float64)
    if k.shape != (3,) or not np.all(np.isfinite(k)):
        raise ValueError("external_k_frac must be a finite length-3 vector")
    options = {} if lswt_options is None else dict(lswt_options)
    external = solve_isotropic_lswt(
        exchange,
        configuration,
        k[None, :],
        anisotropy=anisotropy,
        **options,
    )
    internal_points = np.mod(k[None, :] + coupling.q_points_frac, 1.0)
    internal = solve_isotropic_lswt(
        exchange,
        configuration,
        internal_points,
        anisotropy=anisotropy,
        **options,
    )
    if not np.array_equal(external.metric, internal.metric):
        raise ValueError("external and internal LSWT metrics differ")
    bare = build_bare_isotropic_vertex(
        exchange,
        configuration,
        coupling,
        k,
        q_chunk_size=q_chunk_size,
    )
    external_transform = external.transformation[0]
    band_vertex = np.einsum(
        "qai,qmij,jb->qmab",
        np.swapaxes(internal.transformation.conj(), 1, 2),
        bare,
        external_transform,
        optimize=True,
    )
    return MagnonPhononScatteringProblem(
        external_k_frac=k,
        internal_kq_frac=internal_points,
        external_energy_mev=external.signed_energies_mev[0],
        internal_energy_mev=internal.signed_energies_mev,
        phonon_energy_mev=coupling.phonon_energy_mev,
        vertex_mev=band_vertex,
        metric=external.metric,
        external_transformation=external_transform,
        internal_transformation=internal.transformation,
        physical_mode_count=external.physical_mode_count,
    )


__all__ = [
    "MagnonPhononScatteringProblem",
    "build_bare_isotropic_vertex",
    "build_bare_isotropic_vertex_block",
    "build_scattering_problem",
    "iter_bare_isotropic_vertex_blocks",
]
