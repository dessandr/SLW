"""Independent finite-q checks for the analytic retarded energy integral."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad_vec

from slw.wtorque.torque.kernel import (
    retarded_bubble_loop_eigh_zero_temperature,
    retarded_bubble_loop_eigh_zero_temperature_finite_q,
)
from slw.wtorque.torque.qpair import finalize_q_pair


def _dagger(array):
    return np.swapaxes(array.conj(), -1, -2)


def _unitary(rng, nk=2, nw=3):
    return np.linalg.qr(
        rng.normal(size=(nk, nw, nw)) + 1j * rng.normal(size=(nk, nw, nw))
    )[0]


def _example():
    rng = np.random.default_rng(5948)
    uk, uq = _unitary(rng), _unitary(rng)
    ek = np.array([[-0.8, 0.1, 1.2], [-0.5, -0.5, 1.1]])
    eq = np.array([[-0.6, 0.3, 1.6], [-0.7, 0.0, 0.9]])
    hk = (uk * ek[:, None]) @ _dagger(uk)
    hq = (uq * eq[:, None]) @ _dagger(uq)
    shape = (2, 2, 3, 3)
    torque = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    g = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    # A distributed worker need not own a normalized set of k weights.
    return hk, hq, torque, g, np.array([0.13, 0.48])


_INTERVAL = dict(energy_min_eV=-1.3, occupied_energy_max_eV=0.15, eta_eV=0.045)


def _quadrature(hk, hq, torque, g, weights, **interval):
    """Direct inverses and traces, independent of spectral pole formulas."""
    eta = interval["eta_eV"]
    identity = np.eye(hk.shape[-1])

    def integrand(energy):
        gk = np.linalg.inv((energy + 1j * eta) * identity - hk)
        gq = np.linalg.inv((energy + 1j * eta) * identity - hq)
        output = np.zeros((torque.shape[1], g.shape[1]), dtype=np.complex128)
        for ik, weight in enumerate(weights):
            for it, vertex in enumerate(torque[ik]):
                for ip, perturbation in enumerate(g[ik]):
                    output[it, ip] += weight * np.trace(
                        vertex @ gq[ik] @ perturbation @ gk[ik]
                    )
        return output

    return quad_vec(
        integrand,
        interval["energy_min_eV"],
        interval["occupied_energy_max_eV"],
        epsabs=1.0e-11,
        epsrel=1.0e-11,
    )[0]


def test_finite_q_and_reverse_loop_match_adaptive_quadrature():
    data = _example()
    hk, hq, torque, g, weights = data
    reverse = hq, hk, _dagger(torque), _dagger(g), weights
    actual = retarded_bubble_loop_eigh_zero_temperature_finite_q(
        *data, perturbation_chunk=1, **_INTERVAL
    )
    actual_reverse = retarded_bubble_loop_eigh_zero_temperature_finite_q(
        *reverse, **_INTERVAL
    )
    expected = _quadrature(*data, **_INTERVAL)
    expected_reverse = _quadrature(*reverse, **_INTERVAL)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(
        actual_reverse, expected_reverse, rtol=2.0e-11, atol=2.0e-11
    )
    kq = finalize_q_pair(actual, actual_reverse)
    np.testing.assert_allclose(
        kq, finalize_q_pair(expected, expected_reverse), rtol=2.0e-11, atol=2.0e-11
    )
    np.testing.assert_allclose(finalize_q_pair(actual_reverse, actual), kq.conj())
    assert np.max(np.abs(kq.imag)) > 0.1
    assert not np.allclose(kq, -actual.imag / np.pi)


def test_independent_source_final_gauges_and_k_partition_preserve_integral():
    data = _example()
    hk, hq, torque, g, weights = data
    rng = np.random.default_rng(2027)
    vk, vq = _unitary(rng), _unitary(rng)
    changed_gauge = (
        _dagger(vk) @ hk @ vk,
        _dagger(vq) @ hq @ vq,
        _dagger(vk)[:, None] @ torque @ vq[:, None],
        _dagger(vq)[:, None] @ g @ vk[:, None],
        weights,
    )
    expected = retarded_bubble_loop_eigh_zero_temperature_finite_q(*data, **_INTERVAL)
    actual = retarded_bubble_loop_eigh_zero_temperature_finite_q(
        *changed_gauge, **_INTERVAL
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    distributed = sum(
        retarded_bubble_loop_eigh_zero_temperature_finite_q(
            *(array[ik:ik + 1] for array in data), **_INTERVAL
        )
        for ik in range(len(weights))
    )
    np.testing.assert_allclose(distributed, expected, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize(
    "energy_source,energy_final",
    [(-0.4, -0.4), (-0.4, -0.4 + 1.0e-13), (-0.4, -0.399),
     (-1.5, 0.3), (-1.5, -0.5), (-0.3, 0.5), (0.3, 0.5),
     (-1.3, 0.15), (0.15 - 1.0e-10, 0.15 + 1.0e-10)],
)
def test_pole_branch_and_nearly_degenerate_limits(energy_source, energy_final):
    hk = np.array([[[energy_source]]], dtype=np.complex128)
    hq = np.array([[[energy_final]]], dtype=np.complex128)
    torque = np.array([[[[0.2 + 0.6j]]]], dtype=np.complex128)
    g = np.array([[[[1.1 - 0.7j]]]], dtype=np.complex128)
    data = hk, hq, torque, g, np.array([1.0])
    actual = retarded_bubble_loop_eigh_zero_temperature_finite_q(*data, **_INTERVAL)
    np.testing.assert_allclose(
        actual, _quadrature(*data, **_INTERVAL), rtol=2.0e-11, atol=2.0e-11
    )


def test_q0_api_keeps_same_result():
    hk, _, torque, g, weights = _example()
    expected = retarded_bubble_loop_eigh_zero_temperature(
        hk, torque, g, weights, **_INTERVAL
    )
    actual = retarded_bubble_loop_eigh_zero_temperature_finite_q(
        hk, hk.copy(), torque, g, weights, **_INTERVAL
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize("temperature", [1.0, -1.0, np.nan])
def test_analytic_route_rejects_nonzero_or_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature_K=0"):
        retarded_bubble_loop_eigh_zero_temperature_finite_q(
            *_example(), temperature_K=temperature, **_INTERVAL
        )


def test_nonhermitian_hamiltonian_is_not_silently_diagonalized():
    hk, hq, torque, g, weights = _example()
    hq[0, 0, 1] += 0.1j
    with pytest.raises(ValueError, match="H_kq must be finite and Hermitian"):
        retarded_bubble_loop_eigh_zero_temperature_finite_q(
            hk, hq, torque, g, weights, **_INTERVAL
        )
