"""Retarded magnon--phonon self-energy kernels.

The numerical core in this module is deliberately independent of file formats,
magnetic-order labels, and MPI.  FM and bosonic-BdG (for example collinear AFM)
problems use the same expression; the latter is selected by supplying its signed
metric explicitly.

Array convention
----------------
``vertex_mev`` is ordered as
``(q, phonon_mode, internal_channel, external_channel)``.  The returned
self-energy is ordered as ``(frequency, external_channel, external_channel)``.
All q-point weights are normalized to sum to one before contraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from slw.core.constants import KB_MEV_PER_K


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validate_real_array(
    name: str,
    value: object,
    *,
    ndim: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real, got complex data")
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {result.shape}")
    if any(size < 1 for size in result.shape):
        raise ValueError(f"{name} must not contain an empty axis, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_nonnegative_scalar(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return result


def _validate_positive_scalar(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


def _validate_chunk_size(name: str, value: int | None, dimension: int) -> int:
    if value is None:
        return dimension
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer or None, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return min(result, dimension)


def bose_occupation(energy_mev: object, temperature_k: float) -> np.ndarray:
    r"""Return the Bose occupation, including signed BdG hole channels.

    Negative BdG energies obey
    :math:`n_B(-E)=-(1+n_B(E))`.  Consequently their zero-temperature value is
    ``-1``, not ``0``.  Exact zero energy is well-defined at zero temperature
    under this convention, but its finite-temperature Bose occupation diverges
    and is therefore rejected.  No implicit energy or temperature cutoff is
    applied.
    """

    temperature = _validate_nonnegative_scalar("temperature_k", temperature_k)
    raw = np.asarray(energy_mev)
    if np.iscomplexobj(raw):
        raise ValueError("energy_mev must be real, got complex data")
    energy = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(energy)):
        raise ValueError("energy_mev contains non-finite values")

    occupation = np.zeros(energy.shape, dtype=np.float64)
    negative = energy < 0.0

    if temperature == 0.0:
        occupation[negative] = -1.0
        return occupation

    if np.any(energy == 0.0):
        raise ValueError(
            "energy_mev contains an exact zero with temperature_k > 0; "
            "the Bose occupation diverges"
        )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scaled = np.abs(energy) / (KB_MEV_PER_K * temperature)
        absolute_occupation = 1.0 / np.expm1(scaled)
    if not np.all(np.isfinite(absolute_occupation)):
        raise ValueError(
            "Bose occupation is not representable for the supplied nonzero "
            "energy_mev and temperature_k"
        )

    occupation[...] = absolute_occupation
    occupation[negative] = -(1.0 + absolute_occupation[negative])
    return occupation


def normalize_q_weights(q_weights: object | None, nq: int) -> np.ndarray:
    """Validate q-point weights and return an immutable unit-sum copy."""

    if nq < 1:
        raise ValueError(f"nq must be positive, got {nq}")
    if q_weights is None:
        weights = np.ones(nq, dtype=np.float64)
    else:
        weights = _validate_real_array("q_weights", q_weights, ndim=1)
        if weights.shape != (nq,):
            raise ValueError(f"q_weights shape {weights.shape} != {(nq,)}")
        if np.any(weights < 0.0):
            raise ValueError("q_weights must be non-negative")
    total = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("q_weights must have a finite positive sum")
    return _readonly_copy(weights / total, dtype=np.dtype(np.float64))


def _validate_problem_arrays(
    vertex_mev: object,
    internal_energy_mev: object,
    phonon_energy_mev: object,
    metric: object,
    q_weights: object | None,
    *,
    temperature_k: float,
    metric_energy_tolerance_mev: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    vertex = np.asarray(vertex_mev, dtype=np.complex128)
    if vertex.ndim != 4:
        raise ValueError(
            "vertex_mev must have shape "
            "(nq,nmode,ninternal,nexternal), "
            f"got {vertex.shape}"
        )
    if any(size < 1 for size in vertex.shape):
        raise ValueError(
            f"vertex_mev must not contain an empty axis, got {vertex.shape}"
        )
    if not np.all(np.isfinite(vertex.real)) or not np.all(np.isfinite(vertex.imag)):
        raise ValueError("vertex_mev contains non-finite values")

    nq, nmode, ninternal, _ = vertex.shape
    internal_energy = _validate_real_array(
        "internal_energy_mev", internal_energy_mev, ndim=2
    )
    if internal_energy.shape != (nq, ninternal):
        raise ValueError(
            f"internal_energy_mev shape {internal_energy.shape} != {(nq, ninternal)}"
        )

    phonon_energy = _validate_real_array("phonon_energy_mev", phonon_energy_mev, ndim=2)
    if phonon_energy.shape != (nq, nmode):
        raise ValueError(
            f"phonon_energy_mev shape {phonon_energy.shape} != {(nq, nmode)}"
        )
    if np.any(phonon_energy < 0.0):
        raise ValueError("phonon_energy_mev contains a negative (unstable) mode")
    if temperature_k > 0.0 and np.any(phonon_energy <= 0.0):
        raise ValueError(
            "phonon_energy_mev must be strictly positive at finite temperature"
        )

    metric_diag = _validate_real_array("metric", metric, ndim=1)
    if metric_diag.shape != (ninternal,):
        raise ValueError(f"metric shape {metric_diag.shape} != {(ninternal,)}")
    if not np.all(np.isin(metric_diag, (-1.0, 1.0))):
        raise ValueError("metric entries must be exactly +1 or -1")

    metric_energy = internal_energy * metric_diag[None, :]
    minimum_metric_energy = float(np.min(metric_energy))
    if minimum_metric_energy < -metric_energy_tolerance_mev:
        mismatch_count = int(
            np.count_nonzero(metric_energy < -metric_energy_tolerance_mev)
        )
        raise ValueError(
            "internal_energy_mev is inconsistent with the signed BdG metric: "
            f"{mismatch_count} eta*E value(s) are below "
            f"-{metric_energy_tolerance_mev:.17g} meV "
            f"(minimum {minimum_metric_energy:.17g} meV)"
        )
    if temperature_k > 0.0 and np.any(internal_energy == 0.0):
        raise ValueError(
            "internal_energy_mev contains an exact zero at finite temperature; "
            "the Bose occupation diverges"
        )

    weights = normalize_q_weights(q_weights, nq)
    return (
        vertex,
        internal_energy,
        phonon_energy,
        metric_diag,
        weights,
        minimum_metric_energy,
    )


@dataclass(frozen=True)
class SelfEnergyResult:
    """Complete retarded self-energy evaluated at common frequencies.

    ``metric_energy_tolerance_mev`` records the explicit tolerance used to
    validate ``eta * E``.  ``minimum_metric_energy_mev`` makes any accepted
    roundoff-level sign mismatch visible to downstream audits.
    """

    omega_mev: np.ndarray
    sigma_retarded_mev: np.ndarray
    temperature_k: float
    broadening_mev: float
    metric: np.ndarray
    q_weights: np.ndarray
    metric_energy_tolerance_mev: float
    minimum_metric_energy_mev: float

    @property
    def diagonal_mev(self) -> np.ndarray:
        """Diagonal ``Sigma_mm(omega)`` with shape ``(nomega,nexternal)``."""

        diagonal = np.diagonal(self.sigma_retarded_mev, axis1=-2, axis2=-1)
        diagonal.setflags(write=False)
        return diagonal


@dataclass(frozen=True)
class OnShellSelfEnergyResult:
    """Directly contracted diagonal ``Sigma_mm(E_m)`` with sign provenance."""

    external_energy_mev: np.ndarray
    sigma_diagonal_mev: np.ndarray
    temperature_k: float
    broadening_mev: float
    metric: np.ndarray
    q_weights: np.ndarray
    metric_energy_tolerance_mev: float
    minimum_metric_energy_mev: float

    @property
    def gamma_hwhm_mev(self) -> np.ndarray:
        """Raw HWHM damping ``-Im Sigma_mm(E_m)`` (not clipped)."""

        gamma = -np.imag(self.sigma_diagonal_mev)
        gamma.setflags(write=False)
        return gamma


def compute_retarded_self_energy(
    omega_mev: object,
    vertex_mev: object,
    internal_energy_mev: object,
    phonon_energy_mev: object,
    *,
    temperature_k: float,
    broadening_mev: float,
    metric: object,
    metric_energy_tolerance_mev: float = 0.0,
    q_weights: object | None = None,
    q_chunk_size: int | None = None,
    omega_chunk_size: int | None = None,
) -> SelfEnergyResult:
    r"""Evaluate the retarded self-energy using chunked NumPy contractions.

    The implemented expression is

    .. math::

       \Sigma^R_{mn}(\omega) = \sum_{q\nu l} w_q\eta_l
       g^*_{q\nu lm}g_{q\nu ln}\left[
       \frac{n_\nu+1+n_l}{\omega+i\delta-E_l-\omega_\nu}+
       \frac{n_\nu-n_l}{\omega+i\delta-E_l+\omega_\nu}\right].

    ``q_chunk_size`` and ``omega_chunk_size`` bound temporary arrays while the
    actual channel contraction remains vectorized through ``numpy.einsum``.
    Signed BdG inputs must satisfy ``metric[l] * internal_energy[..., l] >= 0``.
    A caller may explicitly relax that check with
    ``metric_energy_tolerance_mev``; both the tolerance and observed minimum are
    retained in the result.  Its default is exactly zero, with no hidden
    numerical cutoff.
    """

    omega = _validate_real_array("omega_mev", omega_mev, ndim=1)
    broadening = _validate_positive_scalar("broadening_mev", broadening_mev)
    temperature = _validate_nonnegative_scalar("temperature_k", temperature_k)
    metric_energy_tolerance = _validate_nonnegative_scalar(
        "metric_energy_tolerance_mev", metric_energy_tolerance_mev
    )
    (
        vertex,
        internal_energy,
        phonon_energy,
        metric_diag,
        weights,
        minimum_metric_energy,
    ) = _validate_problem_arrays(
        vertex_mev,
        internal_energy_mev,
        phonon_energy_mev,
        metric,
        q_weights,
        temperature_k=temperature,
        metric_energy_tolerance_mev=metric_energy_tolerance,
    )

    nq, _, _, nexternal = vertex.shape
    q_block = _validate_chunk_size("q_chunk_size", q_chunk_size, nq)
    omega_block = _validate_chunk_size("omega_chunk_size", omega_chunk_size, omega.size)
    sigma = np.zeros((omega.size, nexternal, nexternal), dtype=np.complex128)

    for q_start in range(0, nq, q_block):
        q_stop = min(q_start + q_block, nq)
        vertex_q = vertex[q_start:q_stop]
        energy_q = internal_energy[q_start:q_stop, None, :]
        phonon_q = phonon_energy[q_start:q_stop, :, None]
        occupation_internal = bose_occupation(energy_q, temperature)
        occupation_phonon = bose_occupation(phonon_q, temperature)
        weighted_metric = (
            weights[q_start:q_stop, None, None] * metric_diag[None, None, :]
        )

        for omega_start in range(0, omega.size, omega_block):
            omega_stop = min(omega_start + omega_block, omega.size)
            frequency = omega[omega_start:omega_stop, None, None, None]
            denominator_emission = (
                frequency + 1j * broadening - energy_q[None] - phonon_q[None]
            )
            denominator_absorption = (
                frequency + 1j * broadening - energy_q[None] + phonon_q[None]
            )
            thermal = weighted_metric[None] * (
                (occupation_phonon[None] + 1.0 + occupation_internal[None])
                / denominator_emission
                + (occupation_phonon[None] - occupation_internal[None])
                / denominator_absorption
            )
            sigma[omega_start:omega_stop] += np.einsum(
                "wqvl,qvlm,qvln->wmn",
                thermal,
                np.conjugate(vertex_q),
                vertex_q,
                optimize=True,
            )

    return SelfEnergyResult(
        omega_mev=_readonly_copy(omega, dtype=np.dtype(np.float64)),
        sigma_retarded_mev=_readonly_copy(sigma, dtype=np.dtype(np.complex128)),
        temperature_k=temperature,
        broadening_mev=broadening,
        metric=_readonly_copy(metric_diag, dtype=np.dtype(np.float64)),
        q_weights=weights,
        metric_energy_tolerance_mev=metric_energy_tolerance,
        minimum_metric_energy_mev=minimum_metric_energy,
    )


def compute_onshell_self_energy_diagonal(
    external_energy_mev: object,
    vertex_mev: object,
    internal_energy_mev: object,
    phonon_energy_mev: object,
    *,
    temperature_k: float,
    broadening_mev: float,
    metric: object,
    metric_energy_tolerance_mev: float = 0.0,
    q_weights: object | None = None,
    q_chunk_size: int | None = None,
    channel_chunk_size: int | None = None,
) -> OnShellSelfEnergyResult:
    """Directly evaluate ``Sigma_mm(E_m)`` without allocating full matrices.

    The signed-energy validation and explicit tolerance provenance are the same
    as for :func:`compute_retarded_self_energy`.
    """

    external_energy = _validate_real_array(
        "external_energy_mev", external_energy_mev, ndim=1
    )
    broadening = _validate_positive_scalar("broadening_mev", broadening_mev)
    temperature = _validate_nonnegative_scalar("temperature_k", temperature_k)
    metric_energy_tolerance = _validate_nonnegative_scalar(
        "metric_energy_tolerance_mev", metric_energy_tolerance_mev
    )
    (
        vertex,
        internal_energy,
        phonon_energy,
        metric_diag,
        weights,
        minimum_metric_energy,
    ) = _validate_problem_arrays(
        vertex_mev,
        internal_energy_mev,
        phonon_energy_mev,
        metric,
        q_weights,
        temperature_k=temperature,
        metric_energy_tolerance_mev=metric_energy_tolerance,
    )
    nq, _, _, nexternal = vertex.shape
    if external_energy.shape != (nexternal,):
        raise ValueError(
            f"external_energy_mev shape {external_energy.shape} != {(nexternal,)}"
        )

    q_block = _validate_chunk_size("q_chunk_size", q_chunk_size, nq)
    channel_block = _validate_chunk_size(
        "channel_chunk_size", channel_chunk_size, nexternal
    )
    sigma_diagonal = np.zeros(nexternal, dtype=np.complex128)

    for q_start in range(0, nq, q_block):
        q_stop = min(q_start + q_block, nq)
        vertex_q = vertex[q_start:q_stop]
        energy_q = internal_energy[q_start:q_stop, None, :]
        phonon_q = phonon_energy[q_start:q_stop, :, None]
        occupation_internal = bose_occupation(energy_q, temperature)
        occupation_phonon = bose_occupation(phonon_q, temperature)
        weighted_metric = (
            weights[q_start:q_stop, None, None] * metric_diag[None, None, :]
        )

        for channel_start in range(0, nexternal, channel_block):
            channel_stop = min(channel_start + channel_block, nexternal)
            frequency = external_energy[channel_start:channel_stop, None, None, None]
            denominator_emission = (
                frequency + 1j * broadening - energy_q[None] - phonon_q[None]
            )
            denominator_absorption = (
                frequency + 1j * broadening - energy_q[None] + phonon_q[None]
            )
            thermal = weighted_metric[None] * (
                (occupation_phonon[None] + 1.0 + occupation_internal[None])
                / denominator_emission
                + (occupation_phonon[None] - occupation_internal[None])
                / denominator_absorption
            )
            vertex_squared = np.abs(vertex_q[..., channel_start:channel_stop]) ** 2
            sigma_diagonal[channel_start:channel_stop] += np.einsum(
                "cqvl,qvlc->c", thermal, vertex_squared, optimize=True
            )

    return OnShellSelfEnergyResult(
        external_energy_mev=_readonly_copy(external_energy, dtype=np.dtype(np.float64)),
        sigma_diagonal_mev=_readonly_copy(
            sigma_diagonal, dtype=np.dtype(np.complex128)
        ),
        temperature_k=temperature,
        broadening_mev=broadening,
        metric=_readonly_copy(metric_diag, dtype=np.dtype(np.float64)),
        q_weights=weights,
        metric_energy_tolerance_mev=metric_energy_tolerance,
        minimum_metric_energy_mev=minimum_metric_energy,
    )


__all__ = [
    "OnShellSelfEnergyResult",
    "SelfEnergyResult",
    "bose_occupation",
    "compute_onshell_self_energy_diagonal",
    "compute_retarded_self_energy",
    "normalize_q_weights",
]
