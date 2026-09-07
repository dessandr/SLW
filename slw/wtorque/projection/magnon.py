"""External magnon BdG projection without an extra bosonic metric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import require_complex128
from slw.wtorque.errors import ParaunitarityError


@dataclass(frozen=True)
class MagnonProjection:
    normal: NDArray[np.complex128]
    anomalous: NDArray[np.complex128]
    full_nambu: NDArray[np.complex128]
    paraunitarity_residual: float


def magnon_coordinate_map(spin_lengths: object) -> NDArray[np.complex128]:
    """Build ``C_m`` for ``pi^T(-q)=Psi_m^dagger C_m`` (WT-M02)."""

    spins = np.asarray(spin_lengths, dtype=np.float64)
    if (
        spins.ndim != 1 or spins.size == 0
        or not np.all(np.isfinite(spins)) or np.any(spins <= 0)
    ):
        raise ValueError("spin_lengths must be a positive one-dimensional array")
    nmag = spins.size
    result = np.zeros((2 * nmag, 2 * nmag), dtype=np.complex128)
    factors = 1.0 / np.sqrt(2.0 * spins)
    sites = np.arange(nmag)
    result[sites, 2 * sites] = factors
    result[nmag + sites, 2 * sites] = factors
    result[sites, 2 * sites + 1] = 1j * factors
    result[nmag + sites, 2 * sites + 1] = -1j * factors
    return result


def paraunitarity_residual(value: object) -> float:
    transform = require_complex128("magnon T_para", value)
    if (
        transform.ndim != 2 or transform.shape[0] == 0
        or transform.shape[0] != transform.shape[1] or transform.shape[0] % 2
    ):
        raise ParaunitarityError("magnon T_para must be square with even dimension")
    if not np.all(np.isfinite(transform)):
        raise ParaunitarityError("T_para must contain only finite values")
    half = transform.shape[0] // 2
    metric = np.diag(np.r_[np.ones(half), -np.ones(half)]).astype(np.complex128)
    residual = float(np.max(np.abs(transform.conj().T @ metric @ transform - metric)))
    if not np.isfinite(residual):
        raise ParaunitarityError("T_para has a non-finite paraunitarity residual")
    return residual


def project_external_magnons(
    v_pi_ph: object,
    t_magnon: object,
    spin_lengths: object,
    *,
    t_phonon: object | None = None,
    tolerance: float = 1.0e-9,
    spin_coordinate: str = "transverse_direction",
) -> MagnonProjection:
    """Apply ordinary ``T_m^dagger V T_p`` congruence (WT-M03/WT-M04).

    No metric parameter exists here by design. ``Sigma`` is used only for the
    paraunitarity check above, never inserted into the coefficient transform.
    ``spin_coordinate`` describes the incoming coefficients; rotation-angle
    coefficients are converted to transverse directions before projection.
    """

    values = require_complex128("V_pi_ph", v_pi_ph)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not np.all(np.isfinite(values)):
        raise ValueError("V_pi_ph must contain only finite values")
    spins = np.asarray(spin_lengths, dtype=np.float64)
    coordinate_m = magnon_coordinate_map(spins)
    nmag = spins.size
    if values.ndim == 3:
        if values.shape[:2] != (nmag, 2):
            raise ValueError("V_pi_ph must have shape (nmag,2,nphonon)")
        flattened = values.reshape(2 * nmag, values.shape[-1])
    elif values.ndim == 2 and values.shape[0] == 2 * nmag:
        flattened = values
    else:
        raise ValueError("V_pi_ph must have shape (nmag,2,nphonon) or (2*nmag,nphonon)")
    if spin_coordinate == "rotation_angle":
        # pi_1=theta_2, pi_2=-theta_1: transform the coefficient covector.
        paired = flattened.reshape(nmag, 2, -1)
        flattened = np.stack((paired[:, 1], -paired[:, 0]), axis=1).reshape(flattened.shape)
    elif spin_coordinate != "transverse_direction":
        raise ValueError("spin_coordinate must be transverse_direction or rotation_angle")
    transform_m = require_complex128("magnon T_para", t_magnon)
    if transform_m.shape != (2 * nmag, 2 * nmag):
        raise ParaunitarityError("magnon T_para dimension does not match spin sites")
    residual = paraunitarity_residual(transform_m)
    if residual > tolerance:
        raise ParaunitarityError(
            f"magnon paraunitarity residual {residual:.3e} exceeds {tolerance:.3e}"
        )
    nphonon = flattened.shape[1]
    coordinate_p = np.concatenate(
        (
            np.eye(nphonon, dtype=np.complex128),
            np.eye(nphonon, dtype=np.complex128),
        ),
        axis=1,
    )
    bare_nambu = coordinate_m @ flattened @ coordinate_p
    if t_phonon is None:
        transform_p = np.eye(2 * nphonon, dtype=np.complex128)
    else:
        transform_p = require_complex128("phonon T_para", t_phonon)
        if transform_p.shape != (2 * nphonon, 2 * nphonon):
            raise ParaunitarityError("phonon T_para dimension mismatch")
        if paraunitarity_residual(transform_p) > tolerance:
            raise ParaunitarityError("phonon T_para is not paraunitary")
    transformed = transform_m.conj().T @ bare_nambu @ transform_p
    return MagnonProjection(
        normal=np.asarray(transformed[:nmag, :nphonon], dtype=np.complex128),
        anomalous=np.asarray(transformed[:nmag, nphonon:], dtype=np.complex128),
        full_nambu=np.asarray(transformed, dtype=np.complex128),
        paraunitarity_residual=residual,
    )


def degenerate_subspace_invariants(coupling: object) -> tuple[NDArray[np.float64], float]:
    values = require_complex128("coupling", coupling)
    singular = np.linalg.svd(values, compute_uv=False)
    norm = float(np.real(np.trace(values @ values.conj().T)))
    return np.asarray(singular, dtype=np.float64), norm


__all__ = [
    "MagnonProjection",
    "degenerate_subspace_invariants",
    "magnon_coordinate_map",
    "paraunitarity_residual",
    "project_external_magnons",
]
