"""Vectorized bubble contraction and retarded energy loop (WT-K01/WT-K02)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import require_complex128
from slw.wtorque.green.provider import green_from_eigendecomposition, green_matrix
from slw.wtorque.green.real_axis import RealAxisIntegrator


def contract_bubble_batch(
    torque_vertices: object,
    perturbations: object,
    green_kq: object,
    green_k: object,
    k_weights: object,
) -> NDArray[np.complex128]:
    """Contract all torque/perturbation pairs without a ``t,p,nw,nw`` tensor.

    Inputs have shapes ``vertices[nk,nt,nw,nw]``,
    ``g[nk,np,nw,nw]``, and ``G[nk,nw,nw]``. The returned trace is
    ``[nt,np]`` with k weights applied exactly once.
    """

    vertices = require_complex128("torque_vertices", torque_vertices)
    g = require_complex128("DFPT perturbations", perturbations)
    gkq = require_complex128("G_kq", green_kq)
    gk = require_complex128("G_k", green_k)
    weights = np.asarray(k_weights, dtype=np.float64)
    if vertices.ndim != 4 or g.ndim != 4 or gk.ndim != 3 or gkq.ndim != 3:
        raise ValueError("invalid bubble input ranks")
    nk, _, nw, nw2 = vertices.shape
    if nw != nw2 or g.shape[0] != nk or g.shape[-2:] != (nw, nw):
        raise ValueError("vertex and perturbation shapes are incompatible")
    if gk.shape != (nk, nw, nw) or gkq.shape != gk.shape or weights.shape != (nk,):
        raise ValueError("Green-function or k-weight shape mismatch")
    inserted = (gkq[:, None, :, :] @ g) @ gk[:, None, :, :]
    return np.asarray(
        np.einsum("k,ktab,kpba->tp", weights, vertices, inserted, optimize=True),
        dtype=np.complex128,
    )


def retarded_bubble_loop(
    hamiltonian_k: object,
    hamiltonian_kq: object,
    torque_vertices: object,
    perturbations: object,
    k_weights: object,
    integrator: RealAxisIntegrator,
    *,
    eta_eV: float,
    perturbation_chunk: int | None = None,
) -> NDArray[np.complex128]:
    """Integrate the complex retarded loop while streaming perturbation chunks."""

    hk = require_complex128("H_k", hamiltonian_k)
    hkq = require_complex128("H_kq", hamiltonian_kq)
    vertices = require_complex128("torque_vertices", torque_vertices)
    perturb = require_complex128("DFPT perturbations", perturbations)
    if eta_eV <= 0:
        raise ValueError("eta_eV must be positive")
    npert = perturb.shape[1]
    chunk = npert if perturbation_chunk is None else int(perturbation_chunk)
    if chunk < 1:
        raise ValueError("perturbation_chunk must be positive")
    result = np.zeros((vertices.shape[1], npert), dtype=np.complex128)
    for energy, energy_weight in zip(
        integrator.nodes_eV,
        integrator.weighted_occupations(),
        strict=True,
    ):
        green_k = green_matrix(hk, complex(energy, eta_eV))
        green_kq = green_matrix(hkq, complex(energy, eta_eV))
        for start in range(0, npert, chunk):
            stop = min(start + chunk, npert)
            result[:, start:stop] += energy_weight * contract_bubble_batch(
                vertices,
                perturb[:, start:stop],
                green_kq,
                green_k,
                k_weights,
            )
    return result


def retarded_bubble_loop_eigh(
    hamiltonian_k: object,
    hamiltonian_kq: object,
    torque_vertices: object,
    perturbations: object,
    k_weights: object,
    integrator: RealAxisIntegrator,
    *,
    eta_eV: float,
    perturbation_chunk: int | None = None,
) -> NDArray[np.complex128]:
    """Energy loop with each Hermitian Hamiltonian diagonalized only once.

    This is preferable for dense real-axis meshes.  At q=0, passing the same
    Hamiltonian object for both arguments also reuses the same eigensystem and
    Green matrix.
    """

    hk = require_complex128("H_k", hamiltonian_k)
    hkq = require_complex128("H_kq", hamiltonian_kq)
    vertices = require_complex128("torque_vertices", torque_vertices)
    perturb = require_complex128("DFPT perturbations", perturbations)
    weights = np.asarray(k_weights, dtype=np.float64)
    if hk.ndim != 3 or hk.shape[-1] != hk.shape[-2]:
        raise ValueError("H_k must have shape (nk,nw,nw)")
    if hkq.shape != hk.shape:
        raise ValueError("H_k and H_kq must have matching shapes")
    if vertices.ndim != 4 or vertices.shape[0] != hk.shape[0]:
        raise ValueError("torque vertices must have shape (nk,nt,nw,nw)")
    if perturb.ndim != 4 or perturb.shape[0] != hk.shape[0]:
        raise ValueError("perturbations must have shape (nk,np,nw,nw)")
    if weights.shape != (hk.shape[0],):
        raise ValueError("k_weights must have shape (nk,)")
    if eta_eV <= 0:
        raise ValueError("eta_eV must be positive")
    npert = perturb.shape[1]
    chunk = npert if perturbation_chunk is None else int(perturbation_chunk)
    if chunk < 1:
        raise ValueError("perturbation_chunk must be positive")

    eigenvalues_k, eigenvectors_k = np.linalg.eigh(hk)
    same_hamiltonian = hamiltonian_kq is hamiltonian_k
    if same_hamiltonian:
        eigenvalues_kq = eigenvalues_k
        eigenvectors_kq = eigenvectors_k
    else:
        eigenvalues_kq, eigenvectors_kq = np.linalg.eigh(hkq)
    result = np.zeros((vertices.shape[1], npert), dtype=np.complex128)
    for energy, energy_weight in zip(
        integrator.nodes_eV,
        integrator.weighted_occupations(),
        strict=True,
    ):
        z = complex(energy, eta_eV)
        green_k = green_from_eigendecomposition(
            eigenvalues_k,
            np.asarray(eigenvectors_k, dtype=np.complex128),
            z,
        )
        if same_hamiltonian:
            green_kq = green_k
        else:
            green_kq = green_from_eigendecomposition(
                eigenvalues_kq,
                np.asarray(eigenvectors_kq, dtype=np.complex128),
                z,
            )
        for start in range(0, npert, chunk):
            stop = min(start + chunk, npert)
            result[:, start:stop] += energy_weight * contract_bubble_batch(
                vertices,
                perturb[:, start:stop],
                green_kq,
                green_k,
                weights,
            )
    return result


def _zero_temperature_pair_integral(
    eigenvalues: NDArray[np.float64],
    energy_min_eV: float,
    occupied_energy_max_eV: float,
    eta_eV: float,
    *,
    eigenvalues_final: NDArray[np.float64] | None = None,
) -> NDArray[np.complex128]:
    """Integrate retarded pole pairs, with source/final band indices last.

    Individual logarithms use their principal branch: every argument lies in
    the upper half plane for positive ``eta``, so this is a continuous
    primitive along the real integration interval.  A midpoint series avoids
    cancellation of the divided logarithms for (nearly) degenerate poles.
    """

    lower = float(energy_min_eV)
    upper = float(occupied_energy_max_eV)
    eta = float(eta_eV)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("occupied analytic energy interval must increase")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta_eV must be finite and positive")
    energies = np.asarray(eigenvalues, dtype=np.float64)
    if energies.ndim != 2 or not np.all(np.isfinite(energies)):
        raise ValueError("eigenvalues must have finite shape (nk,nw)")
    final = (
        energies
        if eigenvalues_final is None
        else np.asarray(eigenvalues_final, dtype=np.float64)
    )
    if final.shape != energies.shape or not np.all(np.isfinite(final)):
        raise ValueError("final eigenvalues must match finite source eigenvalues")
    first = energies[:, :, None]
    second = final[:, None, :]
    difference = first - second
    numerator = (
        np.log(upper - first + 1j * eta)
        - np.log(upper - second + 1j * eta)
        - np.log(lower - first + 1j * eta)
        + np.log(lower - second + 1j * eta)
    )
    midpoint = 0.5 * (first + second)
    z_upper = upper - midpoint + 1j * eta
    z_lower = lower - midpoint + 1j * eta
    # The endpoint expansion has ratio |(first-second)/(2*z)| <= 0.01;
    # retaining powers through ratio**6 gives an O(1e-16) remainder.
    near_degenerate = np.abs(difference) <= 0.02 * np.minimum(
        np.abs(z_upper), np.abs(z_lower)
    )
    result = np.empty_like(numerator, dtype=np.complex128)
    np.divide(
        numerator,
        difference,
        out=result,
        where=~near_degenerate,
    )
    delta = difference[near_degenerate]

    def primitive(z: NDArray[np.complex128]) -> NDArray[np.complex128]:
        ratio_squared = (delta / (2.0 * z)) ** 2
        return -(1.0 + ratio_squared * (
            1.0 / 3.0 + ratio_squared * (1.0 / 5.0 + ratio_squared / 7.0)
        )) / z

    result[near_degenerate] = (
        primitive(z_upper[near_degenerate])
        - primitive(z_lower[near_degenerate])
    )
    return np.asarray(result, dtype=np.complex128)


def retarded_bubble_loop_eigh_zero_temperature(
    hamiltonian_k: object,
    torque_vertices: object,
    perturbations: object,
    k_weights: object,
    *,
    energy_min_eV: float,
    occupied_energy_max_eV: float,
    eta_eV: float,
    perturbation_chunk: int | None = None,
) -> NDArray[np.complex128]:
    """Backward-compatible q=0 wrapper for the analytic finite-q loop."""

    return retarded_bubble_loop_eigh_zero_temperature_finite_q(
        hamiltonian_k,
        hamiltonian_k,
        torque_vertices,
        perturbations,
        k_weights,
        energy_min_eV=energy_min_eV,
        occupied_energy_max_eV=occupied_energy_max_eV,
        eta_eV=eta_eV,
        perturbation_chunk=perturbation_chunk,
    )


def retarded_bubble_loop_eigh_zero_temperature_finite_q(
    hamiltonian_k: object,
    hamiltonian_kq: object,
    torque_vertices: object,
    perturbations: object,
    k_weights: object,
    *,
    energy_min_eV: float,
    occupied_energy_max_eV: float,
    eta_eV: float,
    perturbation_chunk: int | None = None,
    temperature_K: float = 0.0,
) -> NDArray[np.complex128]:
    """Analytically integrate the complex retarded bubble at general q.

    Return ``A(q) = sum_k w_k integral Tr[Gamma G_kq g G_k] dE`` over
    ``[energy_min_eV, occupied_energy_max_eV]``.  The occupied upper limit
    must be the smaller of the chemical potential and the requested energy
    cutoff.  This is a zero-temperature integral; finite temperature is
    rejected.  Neither a ``-1/pi`` factor nor q/-q completion is applied here.

    ``Gamma[nk,nt,nw,nw]`` is the reverse vertex (k <- k+q), whereas
    ``g[nk,np,nw,nw]`` is forward (k+q <- k).  Source and final Hamiltonians
    may have independent complex eigenvectors.  Both vertices are transformed
    with those same two eigengauges, retaining covariance without assuming
    that either vertex is Hermitian at finite q.  K weights are used once and
    are not renormalized, so MPI workers can integrate disjoint k subsets.
    """

    hamiltonian = require_complex128("H_k", hamiltonian_k)
    hamiltonian_final = require_complex128("H_kq", hamiltonian_kq)
    vertices = require_complex128("torque_vertices", torque_vertices)
    perturb = require_complex128("DFPT perturbations", perturbations)
    weights = np.asarray(k_weights, dtype=np.float64)
    if hamiltonian.ndim != 3 or hamiltonian.shape[-1] != hamiltonian.shape[-2]:
        raise ValueError("H_k must have shape (nk,nw,nw)")
    if hamiltonian_final.shape != hamiltonian.shape:
        raise ValueError("H_k and H_kq must have matching shapes")
    if not np.isfinite(temperature_K) or temperature_K != 0.0:
        raise ValueError("analytic integration requires temperature_K=0")
    for name, matrix in (("H_k", hamiltonian), ("H_kq", hamiltonian_final)):
        if not np.all(np.isfinite(matrix)) or not np.allclose(
            matrix, np.swapaxes(matrix.conj(), -1, -2), rtol=1.0e-10, atol=1.0e-10
        ):
            raise ValueError(f"{name} must be finite and Hermitian")
    nk, nw, _ = hamiltonian.shape
    if vertices.ndim != 4 or vertices.shape[0] != nk:
        raise ValueError("torque vertices must have shape (nk,nt,nw,nw)")
    if perturb.ndim != 4 or perturb.shape[0] != nk:
        raise ValueError("perturbations must have shape (nk,np,nw,nw)")
    if vertices.shape[-2:] != (nw, nw) or perturb.shape[-2:] != (nw, nw):
        raise ValueError("vertex and perturbation dimensions must match H_k")
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(perturb)):
        raise ValueError("vertices and perturbations must be finite")
    if weights.shape != (nk,) or not np.all(np.isfinite(weights)):
        raise ValueError("k_weights must have finite shape (nk,)")
    npert = perturb.shape[1]
    chunk = npert if perturbation_chunk is None else int(perturbation_chunk)
    if chunk < 1:
        raise ValueError("perturbation_chunk must be positive")

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    if hamiltonian_kq is hamiltonian_k:
        eigenvalues_final, eigenvectors_final = eigenvalues, eigenvectors
    else:
        eigenvalues_final, eigenvectors_final = np.linalg.eigh(hamiltonian_final)
    eigenvectors_dagger = np.swapaxes(eigenvectors.conj(), -1, -2)
    eigenvectors_final_dagger = np.swapaxes(eigenvectors_final.conj(), -1, -2)
    vertices_eigen = (
        eigenvectors_dagger[:, None]
        @ vertices
        @ eigenvectors_final[:, None]
    )
    pole_integral = _zero_temperature_pair_integral(
        np.asarray(eigenvalues, dtype=np.float64),
        energy_min_eV,
        occupied_energy_max_eV,
        eta_eV,
        eigenvalues_final=np.asarray(eigenvalues_final, dtype=np.float64),
    )
    result = np.zeros((vertices.shape[1], npert), dtype=np.complex128)
    for start in range(0, npert, chunk):
        stop = min(start + chunk, npert)
        perturb_eigen = (
            eigenvectors_final_dagger[:, None]
            @ perturb[:, start:stop]
            @ eigenvectors[:, None]
        )
        result[:, start:stop] = np.einsum(
            "k,ktab,kpba,kab->tp",
            weights,
            vertices_eigen,
            perturb_eigen,
            pole_integral,
            optimize=True,
        )
    return result


def retarded_bubble_loop_reference(
    hamiltonian_k: object,
    hamiltonian_kq: object,
    torque_vertices: object,
    perturbations: object,
    k_weights: object,
    integrator: RealAxisIntegrator,
    *,
    eta_eV: float,
) -> NDArray[np.complex128]:
    """Straightforward deterministic WT-K02 reference used by validation."""

    hk = require_complex128("H_k", hamiltonian_k)
    hkq = require_complex128("H_kq", hamiltonian_kq)
    vertices = require_complex128("torque_vertices", torque_vertices)
    perturb = require_complex128("DFPT perturbations", perturbations)
    weights = np.asarray(k_weights, dtype=np.float64)
    if eta_eV <= 0:
        raise ValueError("eta_eV must be positive")
    output = np.zeros((vertices.shape[1], perturb.shape[1]), dtype=np.complex128)
    for energy, energy_weight in zip(
        integrator.nodes_eV,
        integrator.weighted_occupations(),
        strict=True,
    ):
        green_k = green_matrix(hk, complex(energy, eta_eV))
        green_kq = green_matrix(hkq, complex(energy, eta_eV))
        for ik, k_weight in enumerate(weights):
            for itorque, torque in enumerate(vertices[ik]):
                for iperturbation, lattice_vertex in enumerate(perturb[ik]):
                    output[itorque, iperturbation] += (
                        energy_weight
                        * k_weight
                        * np.trace(torque @ green_kq[ik] @ lattice_vertex @ green_k[ik])
                    )
    return output


def tune_perturbation_chunk(
    nw: int,
    nk: int,
    npert: int,
    memory_limit_bytes: int,
) -> int:
    """Choose a non-hardcoded chunk from the configured memory budget."""

    if min(nw, nk, npert, memory_limit_bytes) < 1:
        raise ValueError("dimensions and memory_limit_bytes must be positive")
    fixed = 4 * nk * nw * nw * np.dtype(np.complex128).itemsize
    per_perturbation = 2 * nk * nw * nw * np.dtype(np.complex128).itemsize
    available = int(memory_limit_bytes) - fixed
    if available < per_perturbation:
        raise MemoryError(
            f"memory limit {memory_limit_bytes} bytes cannot hold one perturbation chunk"
        )
    return min(npert, max(1, available // per_perturbation))


__all__ = [
    "contract_bubble_batch",
    "retarded_bubble_loop",
    "retarded_bubble_loop_eigh",
    "retarded_bubble_loop_eigh_zero_temperature",
    "retarded_bubble_loop_eigh_zero_temperature_finite_q",
    "retarded_bubble_loop_reference",
    "tune_perturbation_chunk",
]
