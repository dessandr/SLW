"""Cartesian and mode-normalized phonon projection (WT-P01/WT-P02)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import require_complex128
from slw.wtorque.config import Normalization
from slw.wtorque.errors import NormalizationError
from slw.wtorque.units import AMU_KG, ANGSTROM_M, EV_J, HBAR_J_S


def phonon_zero_point_displacement(
    frequencies_eV: object,
    masses_amu: object,
    eigenvectors: object,
) -> NDArray[np.complex128]:
    """Return ``xi[kappa,mu,nu]`` in Angstrom from WT-P01.

    Frequencies are phonon energies ``hbar*omega`` in eV, masses are amu, and
    eigenvectors are dimensionless Cartesian polarization vectors.
    """

    frequencies = np.asarray(frequencies_eV, dtype=np.float64)
    masses = np.asarray(masses_amu, dtype=np.float64)
    vectors = require_complex128("phonon eigenvectors", eigenvectors)
    if frequencies.ndim != 1 or masses.ndim != 1:
        raise NormalizationError("frequencies and masses must be one-dimensional")
    if vectors.shape != (masses.size, 3, frequencies.size):
        raise NormalizationError(
            "phonon eigenvectors must have shape (natom,3,nmode)"
        )
    if np.any(masses <= 0):
        raise NormalizationError("phonon masses must be positive")
    if np.any(frequencies <= 0):
        raise NormalizationError(
            "zero or imaginary phonon modes must be diagnosed before zero-point normalization"
        )
    length_m = np.sqrt(
        HBAR_J_S**2
        / (
            2.0
            * masses[:, None]
            * AMU_KG
            * frequencies[None, :]
            * EV_J
        )
    )
    return np.asarray(
        vectors * (length_m / ANGSTROM_M)[:, None, :],
        dtype=np.complex128,
    )


def project_phonons(
    kernel: object,
    *,
    normalization: Normalization | str,
    frequencies: object | None = None,
    masses: object | None = None,
    eigenvectors: object | None = None,
    mass_weighted_mode_scale: object | None = None,
) -> NDArray[np.complex128]:
    """Project the last ``(atom,cart)`` axes or pass mode-normalized data once.

    A mass-weighted input requires an explicit mode scale because upstream
    mass-coordinate units vary; no implicit conversion is guessed.
    """

    values = require_complex128("phonon input kernel", kernel)
    mode = Normalization(normalization)
    if mode is Normalization.PHONON_ZERO_POINT_MODE:
        if any(item is not None for item in (frequencies, masses, eigenvectors, mass_weighted_mode_scale)):
            raise NormalizationError(
                "phonon_zero_point_mode data must not receive another normalization factor"
            )
        return np.array(values, copy=True)
    if values.ndim < 2:
        raise NormalizationError("Cartesian kernel must end in (natom,3)")
    if frequencies is None or eigenvectors is None:
        raise NormalizationError("Cartesian phonon projection requires frequencies and eigenvectors")
    if mode is Normalization.CARTESIAN_DERIVATIVE:
        if masses is None:
            raise NormalizationError("Cartesian phonon projection requires atomic masses")
        xi = phonon_zero_point_displacement(frequencies, masses, eigenvectors)
    elif mode is Normalization.MASS_WEIGHTED_CARTESIAN:
        vectors = require_complex128("phonon eigenvectors", eigenvectors)
        scale = np.asarray(mass_weighted_mode_scale, dtype=np.float64)
        if mass_weighted_mode_scale is None or scale.shape != (vectors.shape[-1],):
            raise NormalizationError(
                "mass_weighted_cartesian requires explicit mass_weighted_mode_scale[nmode]"
            )
        xi = vectors * scale[None, None, :]
    else:  # pragma: no cover - exhaustive enum
        raise NormalizationError(f"unsupported normalization {mode.value}")
    if values.shape[-2:] != xi.shape[:2]:
        raise NormalizationError(
            f"kernel Cartesian axes {values.shape[-2:]} do not match modes {xi.shape[:2]}"
        )
    return np.asarray(np.einsum("...ac,acv->...v", values, xi, optimize=True), dtype=np.complex128)


def acoustic_translation_residual(kernel: object) -> tuple[float, float]:
    """Return absolute/relative q=0 rigid-translation residual (WT-S03)."""

    values = require_complex128("K_pi_u", kernel)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError("K_pi_u must end in (natom,3)")
    translated = values.sum(axis=-2)
    absolute = float(np.max(np.abs(translated), initial=0.0))
    scale = float(np.max(np.abs(values), initial=0.0))
    scale = max(scale, float(np.finfo(float).tiny))
    return absolute, absolute / scale


__all__ = [
    "acoustic_translation_residual",
    "phonon_zero_point_displacement",
    "project_phonons",
]
