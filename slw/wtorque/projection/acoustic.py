"""Rigid translations at Gamma, before zero-point mode normalization."""
from __future__ import annotations

import numpy as np


def translation_basis(masses_amu, *, phase=None):
    """Orthonormal mass-weighted translations (3*nat,3), in the caller gauge."""
    mass = np.asarray(masses_amu, float)
    if mass.ndim != 1 or not mass.size or np.any(mass <= 0) or not np.all(np.isfinite(mass)):
        raise ValueError("translation masses must be finite and positive")
    result = np.kron(np.sqrt(mass / mass.sum())[:, None], np.eye(3)).astype(complex)
    if phase is not None:
        p = np.asarray(phase, complex)
        if p.shape != mass.shape or not np.allclose(abs(p), 1., atol=1e-12, rtol=0):
            raise ValueError("translation phase needs one unit-modulus value per atom")
        result *= np.repeat(p, 3)[:, None]
    return result


def gamma_optical_modes(dynamical, masses_amu, *, phase=None, tolerance=1e-6):
    """Project translations, retaining raw ASR errors; never gap an acoustic mode.

    The caller's dynamical-matrix units are retained. Return exact zero
    eigenvalues and rigid-translation vectors before the optical columns.
    A large ASR violation fails instead of hiding a physically different model.
    """
    d = np.asarray(dynamical, complex)
    t = translation_basis(masses_amu, phase=phase)
    if d.shape != (len(t), len(t)) or not np.all(np.isfinite(d)):
        raise ValueError("Gamma dynamical matrix dimensions/values differ from masses")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Gamma ASR tolerance must be finite and nonnegative")
    absolute = float(np.linalg.norm(d @ t))
    relative = absolute / max(float(np.linalg.norm(d)), np.finfo(float).tiny)
    if relative > tolerance:
        raise ValueError(f"Gamma phonon translation residual {relative:.6g} exceeds {tolerance:.6g}")
    optical = np.linalg.qr(t, mode='complete')[0][:, 3:]
    values, rotations = np.linalg.eigh(optical.conj().T @ d @ optical)
    if np.any(values <= 0):
        raise ValueError("nonpositive Gamma optical mode; only rigid translations may be removed")
    vectors = np.concatenate((t, optical @ rotations), axis=1)
    values = np.r_[np.zeros(3), values]
    projected = (vectors * values[None, :]) @ vectors.conj().T
    return values, vectors, projected, {
        "raw_translation_absolute": absolute, "raw_translation_relative": relative,
        "matrix_relative_correction": float(np.linalg.norm(projected-d)/max(np.linalg.norm(d), np.finfo(float).tiny)),
        "translation_count": 3,
    }


def project_translation_kernel(kernel, masses_amu, *, phase=None):
    """Remove only rigid translations from a Cartesian mixed derivative.

    Orthogonality is imposed in mass-weighted coordinates. Optical mode
    contractions are unchanged. Apply to bubble/direct separately and test
    the total Ward identity; individual terms need not obey that identity.
    """
    k = np.asarray(kernel, complex)
    mass = np.asarray(masses_amu, float)
    if k.shape[-2:] != (len(mass), 3) or not np.all(np.isfinite(k)):
        raise ValueError("translation kernel must end in finite (atom,3) axes")
    t = translation_basis(mass, phase=phase)
    repeated = np.repeat(np.sqrt(mass), 3)
    weighted = k.reshape(*k.shape[:-2], -1) / repeated
    translation = weighted @ t
    projected = weighted - translation @ t.conj().T
    absolute = float(np.linalg.norm(translation))
    norm = max(float(np.linalg.norm(weighted)), np.finfo(float).tiny)
    return (projected * repeated).reshape(k.shape), {
        "raw_mass_weighted_translation_norm": absolute,
        "raw_translation_relative": absolute / norm,
        "projected_translation_relative": float(np.linalg.norm(projected @ t)) / norm,
        "cartesian_correction_norm_eV_per_angstrom": float(np.linalg.norm((projected-weighted)*repeated)),
    }
