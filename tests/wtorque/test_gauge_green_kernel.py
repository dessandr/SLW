from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from slw.wtorque.gauge.atomic_gauge import atomic_gauge_matrix
from slw.wtorque.gauge.kq_map import build_kq_map
from slw.wtorque.green.provider import green_matrix
from slw.wtorque.green.real_axis import RealAxisIntegrator
from slw.wtorque.torque.kernel import (
    contract_bubble_batch,
    retarded_bubble_loop,
    retarded_bubble_loop_eigh,
    retarded_bubble_loop_eigh_zero_temperature,
    retarded_bubble_loop_reference,
)
from slw.wtorque.torque.qpair import finalize_q_pair


def test_exact_kq_map_and_atomic_gauge_group_composition():
    mesh = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [0.75, 0, 0]]
    )
    mapping = build_kq_map(mesh, np.array([0.25, 0.0, 0.0]))
    np.testing.assert_array_equal(mapping.indices, [1, 2, 3, 0])
    np.testing.assert_array_equal(mapping.G_wrap[-1], [1, 0, 0])
    centers = np.array([[0.1, 0.0, 0.0], [0.35, 0.0, 0.0]])
    d1 = atomic_gauge_matrix(np.array([1, 0, 0]), centers)
    d2 = atomic_gauge_matrix(np.array([-2, 0, 0]), centers)
    d12 = atomic_gauge_matrix(np.array([-1, 0, 0]), centers)
    np.testing.assert_allclose(d1 @ d2, d12)


def test_incommensurate_q_is_a_hard_failure():
    mesh = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    with pytest.raises(ValueError, match="commensurate"):
        build_kq_map(mesh, np.array([0.2, 0.0, 0.0]))


def test_green_derivative_and_batched_bubble_contraction():
    h = np.array([[0.2, 0.1j], [-0.1j, -0.4]], dtype=np.complex128)
    z = 0.7 + 0.03j
    g = green_matrix(h, z)
    delta = 1.0e-7
    numeric = (green_matrix(h, z + delta) - green_matrix(h, z - delta)) / (
        2 * delta
    )
    np.testing.assert_allclose(numeric, -(g @ g), rtol=2e-8, atol=2e-9)

    vertices = np.stack((np.eye(2), np.array([[0, 1], [1, 0]]))).astype(
        np.complex128
    )[None, ...]
    perturbations = np.stack((np.eye(2), h))[None, ...]
    got = contract_bubble_batch(
        vertices,
        perturbations,
        g[None, ...],
        g[None, ...],
        np.array([1.0]),
    )
    reference = np.array(
        [
            [np.trace(t @ g @ p @ g) for p in perturbations[0]]
            for t in vertices[0]
        ]
    )
    np.testing.assert_allclose(got, reference)


def test_general_q_finalizer_uses_both_retarded_loops():
    aq = np.array([1.2 + 0.7j, -0.2 + 0.4j])
    amq = np.array([0.9 - 0.1j, 0.5 + 0.8j])
    kq = finalize_q_pair(aq, amq)
    kmq = finalize_q_pair(amq, aq)
    np.testing.assert_allclose(kmq, kq.conj())
    assert not np.allclose(kq, -np.imag(aq) / np.pi)


def test_vectorized_retarded_loop_matches_deterministic_reference():
    h = np.array([[0.2, 0.1j], [-0.1j, -0.4]], dtype=np.complex128)[None]
    vertices = np.stack((np.eye(2), np.array([[0, 1], [1, 0]]))).astype(
        np.complex128
    )[None]
    perturbations = np.stack((np.eye(2), h[0]))[None].astype(np.complex128)
    integrator = RealAxisIntegrator.gauss_legendre(
        -1.0, 0.1, 12, chemical_potential_eV=0.0
    )
    expected = retarded_bubble_loop_reference(
        h, h, vertices, perturbations, [1.0], integrator, eta_eV=0.03
    )
    actual = retarded_bubble_loop(
        h,
        h,
        vertices,
        perturbations,
        [1.0],
        integrator,
        eta_eV=0.03,
        perturbation_chunk=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    accelerated = retarded_bubble_loop_eigh(
        h,
        h,
        vertices,
        perturbations,
        [1.0],
        integrator,
        eta_eV=0.03,
        perturbation_chunk=1,
    )
    np.testing.assert_allclose(accelerated, expected, rtol=1e-12, atol=1e-12)


def test_zero_temperature_q0_analytic_loop_matches_adaptive_quadrature():
    h = np.array([[0.2, 0.1j], [-0.1j, -0.4]], dtype=np.complex128)[None]
    vertices = np.stack((np.eye(2), np.array([[0, 1], [1, 0]]))).astype(
        np.complex128
    )[None]
    perturbations = np.stack((np.eye(2), h[0]))[None].astype(np.complex128)
    lower = -1.3
    upper = 0.15
    eta = 0.08
    expected = np.empty((2, 2), dtype=np.complex128)
    for itorque, vertex in enumerate(vertices[0]):
        for iperturbation, perturbation in enumerate(perturbations[0]):

            def integrand(
                energy: float,
                vertex_matrix: np.ndarray = vertex,
                perturbation_matrix: np.ndarray = perturbation,
            ) -> complex:
                green = green_matrix(h[0], complex(energy, eta))
                return complex(
                    np.trace(
                        vertex_matrix @ green @ perturbation_matrix @ green
                    )
                )

            real = quad(
                lambda energy: integrand(energy).real,
                lower,
                upper,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
            )[0]
            imag = quad(
                lambda energy: integrand(energy).imag,
                lower,
                upper,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
            )[0]
            expected[itorque, iperturbation] = real + 1j * imag

    actual = retarded_bubble_loop_eigh_zero_temperature(
        h,
        vertices,
        perturbations,
        [1.0],
        energy_min_eV=lower,
        occupied_energy_max_eV=upper,
        eta_eV=eta,
        perturbation_chunk=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=2.0e-11)
