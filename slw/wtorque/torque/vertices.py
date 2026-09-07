"""Exchange-field-only spin rotation vertices (WT-T02 through WT-T05)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from slw.wtorque.basis import require_complex128
from slw.wtorque.config import SiteProjection

PAULI = np.array(
    [
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=np.complex128,
)


def spin_matrix(axis: object, nw: int) -> NDArray[np.complex128]:
    """Return ``I_orb tensor (axis dot sigma)`` in interleaved order."""

    value = np.asarray(axis)
    if value.shape == (3,):
        direction = np.asarray(value, dtype=np.float64)
        if not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-10):
            raise ValueError("spin axis must be a unit vector")
        sigma = np.einsum("i,iab->ab", direction, PAULI, optimize=True)
    elif value.shape == (2, 2):
        sigma = require_complex128("spin axis matrix", value)
    else:
        raise ValueError("spin axis must be a unit 3-vector or complex128 (2,2) Pauli matrix")
    if nw % 2:
        raise ValueError("spinor dimension must be even")
    return np.kron(np.eye(nw // 2, dtype=np.complex128), sigma)


def rotation_vertex(exchange_local: object, axis: object) -> NDArray[np.complex128]:
    """Compute ``Gamma=i[H_XC,axis.sigma]/2`` (MC2023 Eq. 100, WT-T03)."""

    hxc = require_complex128("H_XC_local", exchange_local)
    if hxc.ndim < 2 or hxc.shape[-1] != hxc.shape[-2]:
        raise ValueError("H_XC_local must contain square matrices")
    sigma = spin_matrix(axis, hxc.shape[-1])
    return np.asarray(0.5j * (hxc @ sigma - sigma @ hxc), dtype=np.complex128)


def transverse_vertices(exchange_local: object, local_frame: object) -> NDArray[np.complex128]:
    """Return ``dH/dpi_1,dH/dpi_2`` using WT-T04's tangent map."""

    frame = np.asarray(local_frame, dtype=np.float64)
    if frame.shape != (3, 3):
        raise ValueError("local_frame must have rows [t1,t2,n]")
    gamma_t1 = rotation_vertex(exchange_local, frame[0])
    gamma_t2 = rotation_vertex(exchange_local, frame[1])
    return np.stack((gamma_t2, -gamma_t1), axis=0)


def finite_rotated_exchange(
    exchange_local: object,
    axis: object,
    angle: float,
) -> NDArray[np.complex128]:
    """Rigidly rotate only ``H_XC`` for central finite-difference checks."""

    hxc = require_complex128("H_XC_local", exchange_local)
    sigma = spin_matrix(axis, hxc.shape[-1])
    unitary = expm(-0.5j * float(angle) * sigma)
    return np.asarray(unitary @ hxc @ unitary.conj().T, dtype=np.complex128)


def site_selector_q(
    orbital_masks: object,
    q_red: object,
    orbital_centers: object,
    magnetic_site_positions: object,
) -> NDArray[np.complex128]:
    """Return diagonal ``P_ell(q)`` in the atomic-position gauge (WT-T05)."""

    masks = np.asarray(orbital_masks, dtype=bool)
    q = np.asarray(q_red, dtype=np.float64)
    centers = np.asarray(orbital_centers, dtype=np.float64)
    sites = np.asarray(magnetic_site_positions, dtype=np.float64)
    if masks.ndim != 2 or centers.shape != (masks.shape[1], 3):
        raise ValueError("orbital masks and centers are incompatible")
    if sites.shape != (masks.shape[0], 3) or q.shape != (3,):
        raise ValueError("magnetic site positions and q must use reduced coordinates")
    phase = np.exp(
        2j * np.pi * np.einsum("i,lni->ln", q, sites[:, None, :] - centers[None, :, :])
    )
    diagonal_values = np.repeat(masks * phase, 2, axis=1)
    nw = 2 * masks.shape[1]
    selectors = np.zeros((masks.shape[0], nw, nw), dtype=np.complex128)
    diagonal = np.arange(nw)
    selectors[:, diagonal, diagonal] = diagonal_values
    return selectors


def finite_q_vertices(
    hxc_source: object,
    hxc_final: object,
    *,
    orbital_masks: object,
    local_frames: object,
    q_red: object,
    orbital_centers: object,
    magnetic_site_positions: object,
    coordinate_type: str = "transverse_direction",
    site_projection: SiteProjection | str = SiteProjection.LOCAL_PARTITION,
) -> NDArray[np.complex128]:
    """Build batched finite-q site vertices mapping source k to final k+q.

    ``hxc_source`` and ``hxc_final`` have shape ``[nk,nw,nw]``. The return
    shape is ``[nk,nmag,2,nw,nw]``. Only the exchange field enters the
    commutators; SOC remains solely in the full Hamiltonian Green functions.
    """

    policy = SiteProjection(site_projection)
    if policy not in {SiteProjection.LOCAL_PARTITION, SiteProjection.FULL_ATOM}:
        raise NotImplementedError(
            f"production finite-q vertices do not support site_projection={policy.value}; "
            "onsite_only is a local diagnostic, and user_supplied needs explicit local fields"
        )
    source = require_complex128("H_XC(source)", hxc_source)
    final = require_complex128("H_XC(final)", hxc_final)
    masks = np.asarray(orbital_masks, dtype=bool)
    frames = np.asarray(local_frames, dtype=np.float64)
    if source.ndim == 2:
        source = source[None, ...]
        final = final[None, ...]
    if source.shape != final.shape or source.ndim != 3:
        raise ValueError("source/final exchange fields must have matching [nk,nw,nw] shapes")
    if masks.shape[0] != frames.shape[0] or frames.shape[1:] != (3, 3):
        raise ValueError("magnetic masks and local frames are incompatible")
    selectors = site_selector_q(
        masks,
        q_red,
        orbital_centers,
        magnetic_site_positions,
    )
    nk, nw, _ = source.shape
    result = np.empty((nk, masks.shape[0], 2, nw, nw), dtype=np.complex128)
    for ell, frame in enumerate(frames):
        gamma_source = [rotation_vertex(source, frame[index]) for index in (0, 1)]
        gamma_final = [rotation_vertex(final, frame[index]) for index in (0, 1)]
        localized = [
            0.5
            * (
                selectors[ell] @ gamma_source[index]
                + gamma_final[index] @ selectors[ell]
            )
            for index in (0, 1)
        ]
        if coordinate_type == "rotation_angle":
            result[:, ell, 0] = localized[0]
            result[:, ell, 1] = localized[1]
        elif coordinate_type == "transverse_direction":
            result[:, ell, 0] = localized[1]
            result[:, ell, 1] = -localized[0]
        else:
            raise ValueError("coordinate_type must be transverse_direction or rotation_angle")
    return result


__all__ = [
    "PAULI",
    "finite_q_vertices",
    "finite_rotated_exchange",
    "rotation_vertex",
    "site_selector_q",
    "spin_matrix",
    "transverse_vertices",
]
