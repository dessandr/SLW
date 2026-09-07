"""Phonon self-energy from the native exchange-striction vertex.

The native magnon self-energy treats a magnon as the external quasiparticle.
This module evaluates the reciprocal one-loop process: a phonon couples to a
magnon particle-hole pair, or to a two-magnon pair in a bosonic BdG basis.

For one phonon momentum ``q`` the band vertex is ordered as
``(mode, final_channel_at_k_plus_q, initial_channel_at_k)``.  Signed BdG
energies and their explicit metric make the particle-hole and pair channels
share one expression.  A factor one half removes Nambu double counting; it is
not applied to a normal (non-Nambu) FM basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .self_energy import bose_occupation


def _readonly(value: object, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _chunk_size(name: str, value: int | None, dimension: int) -> int:
    if value is None:
        return dimension
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer or None")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return min(result, dimension)


def _validate_metric_energy(
    name: str,
    energy: NDArray[np.float64],
    metric: NDArray[np.float64],
    tolerance_mev: float,
) -> float:
    metric_energy = energy * metric[None, :]
    minimum = float(np.min(metric_energy))
    if minimum < -tolerance_mev:
        count = int(np.count_nonzero(metric_energy < -tolerance_mev))
        raise ValueError(
            f"{name} is inconsistent with the signed BdG metric: {count} "
            f"eta*E value(s) are below -{tolerance_mev:.17g} meV "
            f"(minimum {minimum:.17g} meV)"
        )
    return minimum


@dataclass(frozen=True)
class PhononSelfEnergyResult:
    """Diagonal retarded phonon self-energy on the bare phonon shell."""

    phonon_energy_mev: NDArray[np.float64]
    sigma_onshell_mev: NDArray[np.complex128]
    temperature_k: float
    broadening_mev: float
    nambu_prefactor: float
    metric: NDArray[np.float64]
    k_weights: NDArray[np.float64]
    metric_energy_tolerance_mev: float
    minimum_metric_energy_mev: float

    def __post_init__(self) -> None:
        energy = _readonly(self.phonon_energy_mev, dtype=np.dtype(np.float64))
        sigma = _readonly(self.sigma_onshell_mev, dtype=np.dtype(np.complex128))
        metric = _readonly(self.metric, dtype=np.dtype(np.float64))
        weights = _readonly(self.k_weights, dtype=np.dtype(np.float64))
        if energy.ndim != 1 or energy.size == 0 or np.any(energy <= 0.0):
            raise ValueError("phonon_energy_mev must be a nonempty positive vector")
        if sigma.shape != energy.shape:
            raise ValueError("sigma_onshell_mev must match phonon_energy_mev")
        if metric.ndim != 1 or metric.size == 0 or not np.all(
            np.isin(metric, (-1.0, 1.0))
        ):
            raise ValueError("metric must be a nonempty +/-1 vector")
        if weights.ndim != 1 or weights.size == 0 or np.any(weights < 0.0):
            raise ValueError("k_weights must be a nonempty non-negative vector")
        if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1.0e-14):
            raise ValueError("k_weights must sum to one")
        if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(sigma)):
            raise ValueError("phonon self-energy result contains non-finite values")
        prefactor = _finite_positive("nambu_prefactor", self.nambu_prefactor)
        tolerance = _finite_nonnegative(
            "metric_energy_tolerance_mev", self.metric_energy_tolerance_mev
        )
        minimum = float(self.minimum_metric_energy_mev)
        if not np.isfinite(minimum):
            raise ValueError("minimum_metric_energy_mev must be finite")
        object.__setattr__(self, "phonon_energy_mev", energy)
        object.__setattr__(self, "sigma_onshell_mev", sigma)
        object.__setattr__(self, "temperature_k", _finite_nonnegative(
            "temperature_k", self.temperature_k
        ))
        object.__setattr__(self, "broadening_mev", _finite_positive(
            "broadening_mev", self.broadening_mev
        ))
        object.__setattr__(self, "nambu_prefactor", prefactor)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "k_weights", weights)
        object.__setattr__(self, "metric_energy_tolerance_mev", tolerance)
        object.__setattr__(self, "minimum_metric_energy_mev", minimum)

    @property
    def gamma_hwhm_mev(self) -> NDArray[np.float64]:
        result = -np.imag(self.sigma_onshell_mev)
        result.setflags(write=False)
        return result


def compute_phonon_self_energy_onshell(
    phonon_energy_mev: ArrayLike,
    vertex_mev: object,
    initial_magnon_energy_mev: object,
    final_magnon_energy_mev: object,
    *,
    temperature_k: float,
    broadening_mev: float,
    metric: object,
    k_weights: object | None = None,
    metric_energy_tolerance_mev: float = 0.0,
    mode_chunk_size: int | None = None,
    channel_chunk_size: int | None = None,
) -> PhononSelfEnergyResult:
    r"""Evaluate the diagonal phonon self-energy at ``omega=omega_qnu``.

    The implemented signed-BdG bubble is

    .. math::

       \Pi^R_\nu(q,\omega)=\frac{c_N}{N_k}\sum_{kab}
       \eta_a\eta_b |g^\nu_{ab}(k,q)|^2
       \frac{n_B(E_{bk})-n_B(E_{a,k+q})}
       {\omega+i\delta+E_{bk}-E_{a,k+q}}.

    ``c_N`` defaults to one for an all-positive normal basis and one half when
    negative-metric Nambu partners are present.  The returned self-energy has
    energy units because the supplied zero-point vertex is in meV.
    """

    phonon = np.asarray(phonon_energy_mev, dtype=np.float64)
    vertex = np.asarray(vertex_mev, dtype=np.complex128)
    initial = np.asarray(initial_magnon_energy_mev, dtype=np.float64)
    final = np.asarray(final_magnon_energy_mev, dtype=np.float64)
    metric_diag = np.asarray(metric, dtype=np.float64)
    if phonon.ndim != 1 or phonon.size == 0 or not np.all(np.isfinite(phonon)):
        raise ValueError("phonon_energy_mev must be a nonempty finite vector")
    if np.any(phonon <= 0.0):
        raise ValueError("phonon_energy_mev must be strictly positive")
    if vertex.ndim != 4 or any(size < 1 for size in vertex.shape):
        raise ValueError(
            "vertex_mev must have shape (nk,nmode,nfinal,ninitial)"
        )
    nk, nmode, nfinal, ninitial = vertex.shape
    if nmode != phonon.size:
        raise ValueError("vertex mode count does not match phonon_energy_mev")
    if initial.shape != (nk, ninitial):
        raise ValueError("initial_magnon_energy_mev has the wrong shape")
    if final.shape != (nk, nfinal):
        raise ValueError("final_magnon_energy_mev has the wrong shape")
    if ninitial != nfinal:
        raise ValueError("initial and final magnon channel counts must match")
    if metric_diag.shape != (ninitial,) or not np.all(
        np.isin(metric_diag, (-1.0, 1.0))
    ):
        raise ValueError("metric must contain one +/-1 entry per magnon channel")
    for name, value in (
        ("vertex_mev", vertex),
        ("initial_magnon_energy_mev", initial),
        ("final_magnon_energy_mev", final),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")

    temperature = _finite_nonnegative("temperature_k", temperature_k)
    broadening = _finite_positive("broadening_mev", broadening_mev)
    tolerance = _finite_nonnegative(
        "metric_energy_tolerance_mev", metric_energy_tolerance_mev
    )
    minimum = min(
        _validate_metric_energy(
            "initial_magnon_energy_mev", initial, metric_diag, tolerance
        ),
        _validate_metric_energy(
            "final_magnon_energy_mev", final, metric_diag, tolerance
        ),
    )
    if temperature > 0.0 and (np.any(initial == 0.0) or np.any(final == 0.0)):
        raise ValueError(
            "magnon energies contain an exact zero at finite temperature; "
            "use a shifted k mesh or a physical anisotropy gap"
        )

    if k_weights is None:
        weights = np.full(nk, 1.0 / float(nk), dtype=np.float64)
    else:
        weights = np.asarray(k_weights, dtype=np.float64)
        if weights.shape != (nk,) or not np.all(np.isfinite(weights)):
            raise ValueError("k_weights must be a finite vector with length nk")
        if np.any(weights < 0.0):
            raise ValueError("k_weights must be non-negative")
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("k_weights must have a finite positive sum")
        weights = weights / total

    prefactor = 0.5 if np.any(metric_diag < 0.0) else 1.0
    mode_block = _chunk_size("mode_chunk_size", mode_chunk_size, nmode)
    channel_block = _chunk_size(
        "channel_chunk_size", channel_chunk_size, nfinal
    )

    occupation_initial = bose_occupation(initial, temperature)
    occupation_final = bose_occupation(final, temperature)
    sigma = np.zeros(nmode, dtype=np.complex128)
    for channel_start in range(0, nfinal, channel_block):
        channel_stop = min(channel_start + channel_block, nfinal)
        channel_slice = slice(channel_start, channel_stop)
        metric_pair = (
            metric_diag[channel_slice, None] * metric_diag[None, :]
        )
        thermal = (
            prefactor
            * weights[:, None, None]
            * metric_pair[None, :, :]
            * (
                occupation_initial[:, None, :]
                - occupation_final[:, channel_slice, None]
            )
        )
        energy_difference = (
            initial[:, None, :] - final[:, channel_slice, None]
        )
        for mode_start in range(0, nmode, mode_block):
            mode_stop = min(mode_start + mode_block, nmode)
            mode_slice = slice(mode_start, mode_stop)
            denominator = (
                phonon[None, mode_slice, None, None]
                + 1.0j * broadening
                + energy_difference[:, None, :, :]
            )
            response = thermal[:, None, :, :] / denominator
            sigma[mode_slice] += np.einsum(
                "kmab,kmab->m",
                np.abs(vertex[:, mode_slice, channel_slice, :]) ** 2,
                response,
                optimize=True,
            )

    return PhononSelfEnergyResult(
        phonon_energy_mev=phonon,
        sigma_onshell_mev=sigma,
        temperature_k=temperature,
        broadening_mev=broadening,
        nambu_prefactor=prefactor,
        metric=metric_diag,
        k_weights=weights,
        metric_energy_tolerance_mev=tolerance,
        minimum_metric_energy_mev=minimum,
    )


__all__ = [
    "PhononSelfEnergyResult",
    "compute_phonon_self_energy_onshell",
]
