"""Independent supercell derivatives and one-G integrals for the direct term."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

from slw.wtorque.green.real_axis import RealAxisIntegrator
from slw.wtorque.torque.direct_vertex import (
    finite_q_direct_vertices,
    retarded_direct_loop,
    retarded_direct_loop_eigh_zero_temperature,
)
from slw.wtorque.torque.kernel import retarded_bubble_loop_eigh_zero_temperature
from slw.wtorque.torque.qpair import finalize_q_pair


def _dagger(array):
    return np.swapaxes(array.conj(), -1, -2)


_SIGMA = np.array([
    [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]],
], dtype=np.complex128)


def _supercell_example(q):
    """Construct a displacement derivative in real space, before any k map."""
    nc, nw = 3, 4
    rng = np.random.default_rng(7171)
    centers = np.array([[.12, .1, 0], [.37, 0, 0]])
    sites = centers.copy()
    sites[1, 0] += 1  # Site image phases are not reduced away.
    kpoints = np.column_stack((np.arange(nc) / nc, np.zeros((nc, 2))))
    frames = np.array([np.eye(3), np.diag([1., -1., -1.])])
    translation = np.zeros((nc * nw, nc * nw), dtype=np.complex128)
    for cell in range(nc):
        translation[((cell + 1) % nc) * nw:((cell + 1) % nc + 1) * nw,
                    cell * nw:(cell + 1) * nw] = np.eye(nw)
    seed = rng.normal(size=(nc * nw, nc * nw)) + 1j * rng.normal(size=(nc * nw, nc * nw))
    seed = .5 * (seed + seed.conj().T)
    derivative = np.zeros_like(seed)
    for cell in range(nc):
        shift = np.linalg.matrix_power(translation, cell)
        derivative += np.exp(2j * np.pi * q[0] * (cell + .21)) * shift @ seed @ shift.conj().T

    def bloch(k):
        basis = np.zeros((nc * nw, nw), dtype=np.complex128)
        for cell in range(nc):
            phase = np.exp(2j * np.pi * (centers @ k + cell * k[0]))
            basis[cell * nw:(cell + 1) * nw] = np.diag(np.repeat(phase, 2)) / np.sqrt(nc)
        return basis

    g = np.array([bloch(k + q).conj().T @ derivative @ bloch(k) for k in kpoints])[:, None]
    options = dict(kpoints=kpoints, q_red=q, orbital_masks=np.eye(2, dtype=bool),
                   local_frames=frames, orbital_centers=centers, magnetic_site_positions=sites)
    return derivative, bloch, options, g


@pytest.mark.parametrize("coordinate_type", ["rotation_angle", "transverse_direction"])
def test_general_q_matches_real_space_mixed_finite_difference(coordinate_type):
    """A nonlocal supercell catches the missing k-q term and reciprocal wraps."""
    q = np.array([1 / 3, 0., 0.])
    derivative, bloch, options, g = _supercell_example(q)
    actual = finite_q_direct_vertices(g, coordinate_type=coordinate_type, **options)
    theta, displacement = 1.e-4, .13
    nc, nw = 3, 4
    for ell, frame in enumerate(options["local_frames"]):
        axes = (frame[1], -frame[0]) if coordinate_type == "transverse_direction" else frame[:2]
        for component, axis in enumerate(axes):
            sigma = np.kron(np.eye(nc * nw // 2), np.einsum("a,aij->ij", axis, _SIGMA))
            up, um = expm(-.5j * theta * sigma), expm(.5j * theta * sigma)
            mixed_fd = np.zeros_like(derivative)
            for cell in range(nc):
                projector = np.zeros_like(derivative)
                for spin in (0, 1):
                    index = cell * nw + 2 * ell + spin
                    projector[index, index] = 1

                def h(rotation, amplitude):
                    shifted_exchange = amplitude * derivative
                    local_exchange = .5 * (projector @ shifted_exchange + shifted_exchange @ projector)
                    return rotation @ local_exchange @ rotation.conj().T

                # Independent real-space spin rotations, followed by the
                # spin Fourier coefficient at -q. No k-q indexing is used.
                local_fd = (h(up, displacement) - h(up, -displacement)
                            - h(um, displacement) + h(um, -displacement)) / (4 * theta * displacement)
                position = np.array([cell, 0., 0.]) + options["magnetic_site_positions"][ell]
                mixed_fd += np.exp(-2j * np.pi * (q @ position)) * local_fd
            expected = np.array([bloch(k).conj().T @ mixed_fd @ bloch(k)
                                 for k in options["kpoints"]])
            np.testing.assert_allclose(actual[:, ell, component, 0], expected, rtol=3.e-8, atol=3.e-8)
            for ik, k in enumerate(options["kpoints"]):
                for jk, kp in enumerate(options["kpoints"]):
                    if ik != jk:
                        np.testing.assert_allclose(bloch(kp).conj().T @ mixed_fd @ bloch(k), 0, atol=1.e-10)
    assert np.max(abs(actual - _dagger(actual))) > .1


def test_direct_insertion_qpair_conjugation_without_forced_hermiticity():
    q = np.array([1 / 3, 0., 0.])
    _, _, options, g = _supercell_example(q)
    _, _, minus_options, gm = _supercell_example(-q)
    plus = finite_q_direct_vertices(g, **options)
    minus = finite_q_direct_vertices(gm, **minus_options)
    np.testing.assert_allclose(minus, _dagger(plus), rtol=1.e-12, atol=1.e-12)


def _loop_example():
    rng = np.random.default_rng(98721)
    raw = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
    h = .4 * (raw + _dagger(raw))
    direct = (rng.normal(size=(2, 2, 2, 3, 3))
              + 1j * rng.normal(size=(2, 2, 2, 3, 3)))
    return h, direct, np.array([.13, .48])


_INTERVAL = dict(energy_min_eV=-3.5, occupied_energy_max_eV=.15, eta_eV=.09)


def test_analytic_one_green_matches_direct_inverse_adaptive_quadrature():
    h, direct, weights = _loop_example()

    def integrand(energy):
        green = np.linalg.inv((energy + 1j * _INTERVAL["eta_eV"]) * np.eye(3) - h)
        result = np.zeros((2, 2), dtype=np.complex128)
        for ik in range(2):
            for it in range(2):
                for ip in range(2):
                    result[it, ip] += weights[ik] * np.trace(direct[ik, it, ip] @ green[ik])
        return result

    expected = quad_vec(integrand, _INTERVAL["energy_min_eV"], _INTERVAL["occupied_energy_max_eV"],
                        epsabs=1.e-11, epsrel=1.e-11)[0]
    actual = retarded_direct_loop_eigh_zero_temperature(h, direct, weights, perturbation_chunk=1, **_INTERVAL)
    np.testing.assert_allclose(actual, expected, rtol=1.e-11, atol=1.e-11)
    reverse = retarded_direct_loop_eigh_zero_temperature(h, _dagger(direct), weights, **_INTERVAL)
    physical = finalize_q_pair(actual, reverse)
    np.testing.assert_allclose(finalize_q_pair(reverse, actual), physical.conj(), rtol=1.e-13)
    assert np.max(abs(physical.imag)) > .1


def test_one_green_is_gauge_covariant_and_supports_distributed_k_weights():
    h, direct, weights = _loop_example()
    rng = np.random.default_rng(1192)
    unitary = np.linalg.qr(rng.normal(size=h.shape) + 1j * rng.normal(size=h.shape))[0]
    changed_h = _dagger(unitary) @ h @ unitary
    changed_direct = _dagger(unitary)[:, None, None] @ direct @ unitary[:, None, None]
    expected = retarded_direct_loop_eigh_zero_temperature(h, direct, weights, **_INTERVAL)
    actual = retarded_direct_loop_eigh_zero_temperature(changed_h, changed_direct, weights, **_INTERVAL)
    np.testing.assert_allclose(actual, expected, rtol=2.e-13, atol=2.e-13)
    partitioned = sum(retarded_direct_loop_eigh_zero_temperature(
        h[ik:ik+1], direct[ik:ik+1], weights[ik:ik+1], **_INTERVAL) for ik in range(2))
    np.testing.assert_allclose(partitioned, expected, rtol=1.e-13, atol=1.e-13)


def test_finite_temperature_quadrature_keeps_integrator_weights_once():
    h, direct, weights = _loop_example()
    integrator = RealAxisIntegrator.gauss_legendre(-3.5, .8, 23, chemical_potential_eV=.15, temperature_K=320)
    actual = retarded_direct_loop(h, direct, weights, integrator, eta_eV=.09, perturbation_chunk=1)
    expected = np.zeros((2, 2), dtype=np.complex128)
    for energy, energy_weight in zip(integrator.nodes_eV, integrator.weighted_occupations(), strict=True):
        green = np.linalg.inv((energy + .09j) * np.eye(3) - h)
        expected += energy_weight * np.einsum("k,ktpab,kba->tp", weights, direct, green)
    np.testing.assert_allclose(actual, expected, rtol=1.e-13, atol=1.e-13)


def test_bubble_plus_direct_matches_frozen_projector_energy_hessian():
    """Check the sign and scale against an occupied band-energy derivative."""
    rng = np.random.default_rng(10398)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h0 = np.diag([-2.2, -1.3, .7, 1.6]).astype(complex) + .09 * (raw + raw.conj().T)
    x = np.kron(np.diag([.7, -.5]), _SIGMA[2])
    seed = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    gxc = .1 * (seed + seed.conj().T)
    gtotal = gxc + np.kron(np.array([[.1, .2j], [-.2j, -.2]]), np.eye(2))
    projector = np.diag([1., 1., 0., 0.])
    sigma = np.kron(np.eye(2), _SIGMA[0])

    def energy(theta, displacement):
        xu = x + displacement * gxc
        local = .5 * (projector @ xu + xu @ projector)
        rotation = expm(-.5j * theta * sigma)
        h = h0 + displacement * gtotal + rotation @ local @ rotation.conj().T - local
        eigenvalues = np.linalg.eigvalsh(h)
        assert np.count_nonzero(eigenvalues < 0) == 2
        return eigenvalues[:2].sum()

    step = 3.e-4
    hessian = (energy(step, step) - energy(step, -step)
               - energy(-step, step) + energy(-step, -step)) / (4 * step**2)
    local = .5 * (projector @ x + x @ projector)
    torque = .5j * (local @ sigma - sigma @ local)
    mixed = finite_q_direct_vertices(
        gxc[None, None], kpoints=np.zeros((1, 3)), q_red=np.zeros(3),
        orbital_masks=[[True, False]], local_frames=np.eye(3)[None],
        orbital_centers=np.zeros((2, 3)), magnetic_site_positions=np.zeros((1, 3)),
        coordinate_type="rotation_angle",
    )[:, :, 0]
    interval = dict(energy_min_eV=-1.e5, occupied_energy_max_eV=0., eta_eV=1.e-9)
    bubble = retarded_bubble_loop_eigh_zero_temperature(
        h0[None], torque[None, None], gtotal[None, None], [1.], **interval)
    direct = retarded_direct_loop_eigh_zero_temperature(h0[None], mixed, [1.], **interval)
    full = -(bubble + direct).imag[0, 0] / np.pi
    np.testing.assert_allclose(full, hessian, atol=2.e-8, rtol=2.e-6)
    assert abs(direct.imag[0, 0] / np.pi) > 1.e-4
    assert abs(-bubble.imag[0, 0] / np.pi - hessian) > 1.e-4


@pytest.mark.parametrize("temperature", [1., -1., np.nan])
def test_analytic_direct_rejects_nonzero_or_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature_K=0"):
        retarded_direct_loop_eigh_zero_temperature(*_loop_example(), temperature_K=temperature, **_INTERVAL)


def test_direct_rejects_moving_projectors_and_nonhermitian_hamiltonian():
    _, _, options, g = _supercell_example(np.array([1 / 3, 0., 0.]))
    with pytest.raises(ValueError, match="moving projectors"):
        finite_q_direct_vertices(g, projector_policy="moving", **options)
    h, direct, weights = _loop_example()
    h[0, 0, 1] += .1j
    with pytest.raises(ValueError, match="H_k must be finite and Hermitian"):
        retarded_direct_loop_eigh_zero_temperature(h, direct, weights, **_INTERVAL)
