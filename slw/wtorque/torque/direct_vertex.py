"""Fixed-projector exchange derivative and momentum-closed one-G loop.

The supplied derivative must describe the same exchange operator that is
rotated in the torque model. The full displacement derivative is not a
substitute: that would also rotate displacement-induced spin-orbit terms.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import require_complex128
from slw.wtorque.config import SiteProjection
from slw.wtorque.gauge.atomic_gauge import atomic_gauge_matrix
from slw.wtorque.gauge.kq_map import build_kq_map
from slw.wtorque.green.real_axis import RealAxisIntegrator
from slw.wtorque.model.local_frames import validate_local_frames
from slw.wtorque.model.local_projection import localize_exchange

from .vertices import rotation_vertex, site_selector_q


def direct_rotation_vertices(
    g_xc: object | None,
    projectors: object,
    axes: object,
    *,
    include: bool = False,
    projector_policy: str = "fixed",
) -> NDArray[np.complex128] | None:
    """Build ``i[L_ell(g_XC),sigma_c]/2`` only from declared exchange DFPT."""

    if not include:
        return None
    if g_xc is None:
        raise ValueError("include_direct_vertex requires exchange-resolved g_XC")
    if projector_policy != "fixed":
        raise ValueError("moving projectors require an explicit projector derivative dataset")
    local, _ = localize_exchange(g_xc, projectors)
    axis_array = np.asarray(axes, dtype=np.float64)
    if axis_array.shape != (local.shape[0], 3):
        raise ValueError("axes must have shape (nmag,3)")
    return np.stack(
        [rotation_vertex(local[index], axis_array[index]) for index in range(local.shape[0])],
        axis=0,
    )


def finite_q_direct_vertices(
    g_xc: object,
    *,
    kpoints: object,
    q_red: object,
    g_xc_at_k_minus_q: object | None = None,
    orbital_masks: object,
    local_frames: object,
    orbital_centers: object,
    magnetic_site_positions: object,
    coordinate_type: str = "transverse_direction",
    projector_policy: str = "fixed",
    site_projection: SiteProjection | str = SiteProjection.LOCAL_PARTITION,
) -> NDArray[np.complex128]:
    r"""Return the diagonal-k mixed insertion for spin(-q), displacement(q).

    ``g_xc[nk,npert,nw,nw]`` maps k to *unwrapped* k+q in the canonical
    interleaved atomic-position gauge. The result has shape
    ``[nk,nmag,2,npert,nw,nw]``. In the local-partition model its rotation
    components are

    ``D(k;q) = [P(-q) Gamma(g_XC(k,q))
                 + Gamma(g_XC(k-q,q)) P(-q)] / 2``.

    This follows by differentiating ``{P(-q),Gamma(H_XC)}/2`` in the full
    momentum-space operator. The two summands close the displacement momentum
    on opposite electronic endpoints. ``g_XC(k-q,q)`` is shifted at *both*
    endpoints by the reciprocal wrap, not merely looked up by its mesh index.
    Alternatively ``g_xc_at_k_minus_q`` supplies that second term explicitly,
    already in the atomic gauge with unwrapped endpoints k-q and k. This
    supports arbitrary q that does not close on the integration k mesh.
    Contracting the forward g_XC commutator directly with G(k) is incorrect at
    general q. Neither D nor g_XC is made Hermitian at an individual q.

    Atomic projectors, orbital centers, and the spin frame are held fixed;
    projector or basis-motion derivatives are outside this construction.
    """

    if projector_policy != "fixed":
        raise ValueError("moving projectors require explicit projector-response vertices")
    policy = SiteProjection(site_projection)
    if policy not in {SiteProjection.LOCAL_PARTITION, SiteProjection.FULL_ATOM}:
        raise NotImplementedError("finite-q direct vertices require local_partition or full_atom")
    g = require_complex128("exchange-resolved g_XC", g_xc)
    mesh = np.asarray(kpoints, dtype=np.float64)
    q = np.asarray(q_red, dtype=np.float64)
    masks = np.asarray(orbital_masks, dtype=bool)
    centers = np.asarray(orbital_centers, dtype=np.float64)
    sites = np.asarray(magnetic_site_positions, dtype=np.float64)
    frames = validate_local_frames(local_frames)
    if g.ndim != 4 or g.shape[-1] != g.shape[-2] or g.shape[-1] % 2:
        raise ValueError("g_XC must have shape (nk,npert,nw,nw) with even nw")
    nk, npert, nw, _ = g.shape
    if min(nk, npert, nw) < 1 or mesh.shape != (nk, 3) or q.shape != (3,):
        raise ValueError("g_XC, kpoints, and q dimensions are incompatible")
    if masks.ndim != 2 or masks.shape != (len(frames), nw // 2):
        raise ValueError("magnetic masks must have shape (nmag,nw/2)")
    if centers.shape != (nw // 2, 3) or sites.shape != (len(frames), 3):
        raise ValueError("orbital centers and magnetic site positions have incompatible shapes")
    if any(not np.all(np.isfinite(value)) for value in (g, mesh, q, centers, sites, frames)):
        raise ValueError("finite-q direct vertex inputs must be finite")
    if coordinate_type not in {"transverse_direction", "rotation_angle"}:
        raise ValueError("coordinate_type must be transverse_direction or rotation_angle")
    if g_xc_at_k_minus_q is None:
        mapping = build_kq_map(mesh, -q)
        wrap = atomic_gauge_matrix(mapping.G_wrap, centers)[:, None]
        shifted = np.swapaxes(wrap.conj(), -1, -2) @ g[mapping.indices] @ wrap
    else:
        shifted = require_complex128("g_XC(k-q,q)", g_xc_at_k_minus_q)
        if shifted.shape != g.shape or not np.all(np.isfinite(shifted)):
            raise ValueError("g_XC(k-q,q) must have the same finite shape as g_XC(k,q)")
    selectors = site_selector_q(masks, -q, centers, sites)
    result = np.empty((nk, len(frames), 2, npert, nw, nw), dtype=np.complex128)
    for ell, frame in enumerate(frames):
        axes = (frame[1], -frame[0]) if coordinate_type == "transverse_direction" else frame[:2]
        for component, axis in enumerate(axes):
            result[:, ell, component] = 0.5 * (
                selectors[ell] @ rotation_vertex(g, axis)
                + rotation_vertex(shifted, axis) @ selectors[ell]
            )
    return result


def _direct_loop_inputs(
    hamiltonian_k: object,
    direct_vertices: object,
    k_weights: object,
    eta_eV: float,
    perturbation_chunk: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    h = require_complex128("H_k", hamiltonian_k)
    direct = require_complex128("direct mixed vertices", direct_vertices)
    weights = np.asarray(k_weights, dtype=np.float64)
    if h.ndim != 3 or h.shape[-1] != h.shape[-2] or min(h.shape) < 1:
        raise ValueError("H_k must have shape (nk,nw,nw)")
    if not np.all(np.isfinite(h)) or not np.allclose(
        h, np.swapaxes(h.conj(), -1, -2), rtol=1.0e-10, atol=1.0e-10
    ):
        raise ValueError("H_k must be finite and Hermitian")
    if (direct.ndim != 5 or direct.shape[0] != h.shape[0]
            or direct.shape[-2:] != h.shape[-2:] or min(direct.shape) < 1):
        raise ValueError("direct vertices must have shape (nk,nt,npert,nw,nw) matching H_k")
    if not np.all(np.isfinite(direct)):
        raise ValueError("direct mixed vertices must be finite")
    if weights.shape != (h.shape[0],) or not np.all(np.isfinite(weights)):
        raise ValueError("k_weights must have finite shape (nk,)")
    if not np.isfinite(eta_eV) or eta_eV <= 0:
        raise ValueError("eta_eV must be finite and positive")
    chunk = direct.shape[2] if perturbation_chunk is None else int(perturbation_chunk)
    if chunk < 1:
        raise ValueError("perturbation_chunk must be positive")
    return h, direct, weights, chunk


def _contract_integrated_direct(
    eigenvectors: np.ndarray,
    integrated_poles: np.ndarray,
    direct: np.ndarray,
    weights: np.ndarray,
    chunk: int,
) -> NDArray[np.complex128]:
    integrated_green = ((eigenvectors * integrated_poles[:, None, :])
                        @ np.swapaxes(eigenvectors.conj(), -1, -2))
    result = np.empty(direct.shape[1:3], dtype=np.complex128)
    for start in range(0, direct.shape[2], chunk):
        stop = min(start + chunk, direct.shape[2])
        result[:, start:stop] = np.einsum(
            "k,ktpab,kba->tp", weights, direct[:, :, start:stop], integrated_green,
            optimize=True,
        )
    return result


def retarded_direct_loop_eigh_zero_temperature(
    hamiltonian_k: object,
    direct_vertices: object,
    k_weights: object,
    *,
    energy_min_eV: float,
    occupied_energy_max_eV: float,
    eta_eV: float,
    temperature_K: float = 0.0,
    perturbation_chunk: int | None = None,
) -> NDArray[np.complex128]:
    r"""Integrate ``A_direct(q)=sum_k w_k int Tr[D(k;q)G_k] dE`` exactly.

    D has shape ``[nk,nt,npert,nw,nw]`` and is already momentum closed by
    :func:`finite_q_direct_vertices`. The log primitive uses continuous upper
    half-plane branches over the finite occupied interval. The complex return
    ``[nt,npert]`` has neither ``-1/pi`` nor q/-q completion applied. K weights
    are used once without renormalization, supporting disjoint MPI k subsets.
    Energies, broadenings, and H use eV; a D in eV/Angstrom gives the completed
    physical direct kernel in eV/Angstrom.
    """

    h, direct, weights, chunk = _direct_loop_inputs(
        hamiltonian_k, direct_vertices, k_weights, eta_eV, perturbation_chunk
    )
    if not np.isfinite(temperature_K) or temperature_K != 0.0:
        raise ValueError("analytic integration requires temperature_K=0")
    lower, upper = float(energy_min_eV), float(occupied_energy_max_eV)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("occupied analytic energy interval must increase")
    energies, vectors = np.linalg.eigh(h)
    poles = np.log(upper - energies + 1j * eta_eV) - np.log(lower - energies + 1j * eta_eV)
    return _contract_integrated_direct(vectors, poles, direct, weights, chunk)


def retarded_direct_loop(
    hamiltonian_k: object,
    direct_vertices: object,
    k_weights: object,
    integrator: RealAxisIntegrator,
    *,
    eta_eV: float,
    perturbation_chunk: int | None = None,
) -> NDArray[np.complex128]:
    """Compute the same complex one-G loop with declared real-axis weights.

    The integrator owns the energy quadrature and occupations. Each H(k) is
    diagonalized once; spectral one-pole sums are accumulated before the
    direct-vertex contraction, keeping the matrix work independent of the
    number of energy nodes.
    """

    h, direct, weights, chunk = _direct_loop_inputs(
        hamiltonian_k, direct_vertices, k_weights, eta_eV, perturbation_chunk
    )
    energies, vectors = np.linalg.eigh(h)
    poles = np.zeros(energies.shape, dtype=np.complex128)
    for energy, weight in zip(integrator.nodes_eV, integrator.weighted_occupations(), strict=True):
        poles += weight / (energy + 1j * eta_eV - energies)
    return _contract_integrated_direct(vectors, poles, direct, weights, chunk)


__all__ = [
    "direct_rotation_vertices",
    "finite_q_direct_vertices",
    "retarded_direct_loop",
    "retarded_direct_loop_eigh_zero_temperature",
]
