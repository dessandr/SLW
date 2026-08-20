"""Linewidth, scattering-rate, and lifetime observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slw.core.constants import HBAR_MEV_PS


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LifetimeResult:
    """Physical observables derived from retarded on-shell damping.

    ``raw_gamma_hwhm_mev`` always preserves the input.  Tiny negative values
    are clipped only when the caller supplies a non-zero
    ``negative_tolerance_mev``; their positions are recorded in
    ``roundoff_clipped``.  More negative values are recorded in
    ``invalid_negative_damping`` and all derived observables at those positions
    are ``NaN``.
    """

    raw_gamma_hwhm_mev: np.ndarray
    gamma_hwhm_mev: np.ndarray
    fwhm_mev: np.ndarray
    scattering_rate_ps_inv: np.ndarray
    lifetime_ps: np.ndarray
    valid_damping: np.ndarray
    roundoff_clipped: np.ndarray
    invalid_negative_damping: np.ndarray
    negative_tolerance_mev: float

    @property
    def has_invalid_damping(self) -> bool:
        return bool(np.any(~self.valid_damping))


def linewidth_observables(
    gamma_hwhm_mev: object,
    *,
    negative_tolerance_mev: float = 0.0,
) -> LifetimeResult:
    """Convert HWHM damping to FWHM, rate, and lifetime.

    The convention is

    ``FWHM = 2 gamma``, ``rate = FWHM / hbar``, and
    ``lifetime = hbar / FWHM``.

    A zero linewidth therefore has zero scattering rate and infinite lifetime.
    No absolute clipping scale is guessed: callers that want roundoff clipping
    must pass an explicit, physically justified tolerance in meV.
    """

    raw_input = np.asarray(gamma_hwhm_mev)
    if np.iscomplexobj(raw_input):
        raise ValueError("gamma_hwhm_mev must be real, got complex data")
    raw_gamma = np.asarray(raw_input, dtype=np.float64)

    tolerance = float(negative_tolerance_mev)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "negative_tolerance_mev must be finite and non-negative, "
            f"got {negative_tolerance_mev!r}"
        )

    finite = np.isfinite(raw_gamma)
    roundoff_clipped = finite & (raw_gamma < 0.0) & (raw_gamma >= -tolerance)
    invalid_negative = finite & (raw_gamma < -tolerance)
    valid = finite & ~invalid_negative

    gamma = np.full(raw_gamma.shape, np.nan, dtype=np.float64)
    gamma[valid] = raw_gamma[valid]
    gamma[roundoff_clipped] = 0.0

    fwhm = np.full(raw_gamma.shape, np.nan, dtype=np.float64)
    rate = np.full(raw_gamma.shape, np.nan, dtype=np.float64)
    lifetime = np.full(raw_gamma.shape, np.nan, dtype=np.float64)
    fwhm[valid] = 2.0 * gamma[valid]
    rate[valid] = fwhm[valid] / HBAR_MEV_PS
    positive = valid & (gamma > 0.0)
    zero = valid & (gamma == 0.0)
    lifetime[positive] = HBAR_MEV_PS / fwhm[positive]
    lifetime[zero] = np.inf

    return LifetimeResult(
        raw_gamma_hwhm_mev=_readonly_copy(raw_gamma, dtype=np.dtype(np.float64)),
        gamma_hwhm_mev=_readonly_copy(gamma, dtype=np.dtype(np.float64)),
        fwhm_mev=_readonly_copy(fwhm, dtype=np.dtype(np.float64)),
        scattering_rate_ps_inv=_readonly_copy(rate, dtype=np.dtype(np.float64)),
        lifetime_ps=_readonly_copy(lifetime, dtype=np.dtype(np.float64)),
        valid_damping=_readonly_copy(valid, dtype=np.dtype(np.bool_)),
        roundoff_clipped=_readonly_copy(roundoff_clipped, dtype=np.dtype(np.bool_)),
        invalid_negative_damping=_readonly_copy(
            invalid_negative, dtype=np.dtype(np.bool_)
        ),
        negative_tolerance_mev=tolerance,
    )


def lifetime_from_onshell_self_energy(
    sigma_onshell_mev: object,
    *,
    negative_tolerance_mev: float = 0.0,
) -> LifetimeResult:
    """Derive lifetime observables from retarded on-shell self-energy values.

    A one-dimensional input is interpreted directly as ``Sigma_mm(E_m)``.
    For an array ending in two equal channel axes, its diagonal is used.  The
    latter supports batches but assumes that the caller has already selected
    the appropriate on-shell frequency for each stored matrix.
    """

    sigma = np.asarray(sigma_onshell_mev, dtype=np.complex128)
    if sigma.ndim == 0:
        raise ValueError("sigma_onshell_mev must be at least one-dimensional")
    if sigma.ndim == 1:
        diagonal = sigma
    elif sigma.shape[-2] == sigma.shape[-1]:
        diagonal = np.diagonal(sigma, axis1=-2, axis2=-1)
    else:
        raise ValueError(
            "sigma_onshell_mev must be a diagonal vector or end in square axes, "
            f"got {sigma.shape}"
        )
    gamma = -np.imag(diagonal)
    return linewidth_observables(gamma, negative_tolerance_mev=negative_tolerance_mev)


def compute_lifetime(
    gamma_hwhm_mev: object,
    *,
    negative_tolerance_mev: float = 0.0,
) -> LifetimeResult:
    """Public shorthand for :func:`linewidth_observables`."""

    return linewidth_observables(
        gamma_hwhm_mev, negative_tolerance_mev=negative_tolerance_mev
    )


__all__ = [
    "LifetimeResult",
    "compute_lifetime",
    "lifetime_from_onshell_self_energy",
    "linewidth_observables",
]
