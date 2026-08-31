"""Magnon-phonon rotating-wave block assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import require_complex128


def assemble_rwa(
    magnon_energies: object,
    phonon_energies: object,
    normal_coupling: object,
) -> NDArray[np.complex128]:
    magnons = np.asarray(magnon_energies, dtype=np.float64)
    phonons = np.asarray(phonon_energies, dtype=np.float64)
    coupling = require_complex128("g_mp_normal", normal_coupling)
    if magnons.ndim != 1 or phonons.ndim != 1 or coupling.shape != (magnons.size, phonons.size):
        raise ValueError("RWA energies/coupling dimensions are incompatible")
    top = np.concatenate((np.diag(magnons), coupling), axis=1)
    bottom = np.concatenate((coupling.conj().T, np.diag(phonons)), axis=1)
    return np.asarray(np.concatenate((top, bottom), axis=0), dtype=np.complex128)


@dataclass(frozen=True)
class BosonicBdGDiagnostics:
    """Full normal/anomalous quadratic-boson stability diagnostics."""

    hessian_mev: NDArray[np.complex128]
    dynamic_eigenvalues_mev: NDArray[np.complex128]
    minimum_hessian_eigenvalue_mev: float
    maximum_imaginary_eigenvalue_mev: float
    particle_hole_residual_mev: float
    hessian_hermiticity_residual_mev: float
    stable: bool


def assemble_bosonic_bdg(
    magnon_energies: object,
    phonon_energies: object,
    normal_coupling: object,
    anomalous_coupling: object,
) -> NDArray[np.complex128]:
    """Assemble the Nambu Hessian in ``(a,b,a†,b†)`` order."""

    magnons = np.asarray(magnon_energies, dtype=np.float64)
    phonons = np.asarray(phonon_energies, dtype=np.float64)
    normal = require_complex128("g_mp_normal", normal_coupling)
    anomalous = require_complex128("g_mp_anomalous", anomalous_coupling)
    expected = (magnons.size, phonons.size)
    if magnons.ndim != 1 or phonons.ndim != 1:
        raise ValueError("bosonic BdG energies must be one-dimensional")
    if normal.shape != expected or anomalous.shape != expected:
        raise ValueError("bosonic BdG coupling dimensions are incompatible")
    count = magnons.size + phonons.size
    normal_block = np.zeros((count, count), dtype=np.complex128)
    normal_block[: magnons.size, : magnons.size] = np.diag(magnons)
    normal_block[magnons.size :, magnons.size :] = np.diag(phonons)
    normal_block[: magnons.size, magnons.size :] = normal
    normal_block[magnons.size :, : magnons.size] = normal.conj().T
    pairing = np.zeros_like(normal_block)
    pairing[: magnons.size, magnons.size :] = anomalous
    pairing[magnons.size :, : magnons.size] = anomalous.T
    top = np.concatenate((normal_block, pairing), axis=1)
    bottom = np.concatenate((pairing.conj(), normal_block.conj()), axis=1)
    return np.asarray(np.concatenate((top, bottom), axis=0), dtype=np.complex128)


def diagnose_bosonic_bdg(
    magnon_energies: object,
    phonon_energies: object,
    normal_coupling: object,
    anomalous_coupling: object,
    *,
    stability_tolerance_mev: float = 1.0e-9,
    imaginary_tolerance_mev: float = 1.0e-9,
) -> BosonicBdGDiagnostics:
    """Diagonalize the full bosonic dynamic matrix without hiding instability."""

    stability_tolerance = float(stability_tolerance_mev)
    imaginary_tolerance = float(imaginary_tolerance_mev)
    if (
        not np.isfinite(stability_tolerance)
        or stability_tolerance < 0.0
        or not np.isfinite(imaginary_tolerance)
        or imaginary_tolerance < 0.0
    ):
        raise ValueError("bosonic BdG tolerances must be finite and non-negative")
    hessian = assemble_bosonic_bdg(
        magnon_energies,
        phonon_energies,
        normal_coupling,
        anomalous_coupling,
    )
    hermiticity = float(np.max(np.abs(hessian - hessian.conj().T), initial=0.0))
    hessian_eigenvalues = np.linalg.eigvalsh(
        0.5 * (hessian + hessian.conj().T)
    )
    minimum_hessian = float(np.min(hessian_eigenvalues))
    half = hessian.shape[0] // 2
    metric = np.r_[np.ones(half), -np.ones(half)]
    dynamic_eigenvalues = np.linalg.eigvals(metric[:, None] * hessian)
    order = np.lexsort((dynamic_eigenvalues.imag, dynamic_eigenvalues.real))
    dynamic_eigenvalues = np.asarray(
        dynamic_eigenvalues[order], dtype=np.complex128
    )
    maximum_imaginary = float(
        np.max(np.abs(dynamic_eigenvalues.imag), initial=0.0)
    )
    pair_distance = np.abs(
        dynamic_eigenvalues[:, None]
        + dynamic_eigenvalues.conj()[None, :]
    )
    particle_hole = float(
        np.max(np.min(pair_distance, axis=1), initial=0.0)
    )
    return BosonicBdGDiagnostics(
        hessian_mev=hessian,
        dynamic_eigenvalues_mev=dynamic_eigenvalues,
        minimum_hessian_eigenvalue_mev=minimum_hessian,
        maximum_imaginary_eigenvalue_mev=maximum_imaginary,
        particle_hole_residual_mev=particle_hole,
        hessian_hermiticity_residual_mev=hermiticity,
        stable=(
            minimum_hessian >= -stability_tolerance
            and maximum_imaginary <= imaginary_tolerance
        ),
    )


__all__ = [
    "BosonicBdGDiagnostics",
    "assemble_bosonic_bdg",
    "assemble_rwa",
    "diagnose_bosonic_bdg",
]
