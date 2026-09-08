from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from slw.wtorque.gauge.atomic_gauge import atomic_gauge_matrix
from slw.wtorque.gauge.kq_map import build_kq_map
from slw.wtorque.interpolated_response import ArbitraryQResponse
from slw.wtorque.model.native_spinor_frame import build_native_spinor_frame
from slw.wtorque.torque.direct_vertex import finite_q_direct_vertices, retarded_direct_loop_eigh_zero_temperature
from slw.wtorque.torque.kernel import retarded_bubble_loop_eigh_zero_temperature_finite_q
from slw.wtorque.torque.qpair import finalize_q_pair


def _dagger(value):
    return value.conj().swapaxes(-1, -2)


class _AnalyticBackend:
    def __init__(self):
        centers = np.array([[0., 0., 0.], [.37, .1, -.2]])
        self.metadata = SimpleNamespace(
            nk_grid=(3, 1, 1), nq_grid=(3, 1, 1), at=np.eye(3), wc=np.repeat(centers, 2, axis=0),
            tau=centers, nat=2, nwan=4,
        )
        rng = np.random.default_rng(252)
        self.hop = .07 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        self.glocal = .03 * (rng.normal(size=(6, 4, 4)) + 1j * rng.normal(size=(6, 4, 4)))
        self.glocal += _dagger(self.glocal)
        self.ghop = .02 * (rng.normal(size=(6, 4, 4)) + 1j * rng.normal(size=(6, 4, 4)))

    def evaluate_h(self, k):
        phase = np.exp(2j * np.pi * np.asarray(k)[:, 0])[:, None, None]
        return np.diag([-.8, .6, -.7, .9]) + phase * self.hop + phase.conj() * _dagger(self.hop)

    def evaluate_g(self, k, q):
        k = np.asarray(k)
        first = np.exp(2j * np.pi * k[:, 0])[:, None, None, None]
        second = np.exp(-2j * np.pi * (k[:, 0] + q[0]))[:, None, None, None]
        return self.glocal + first * self.ghop + second * _dagger(self.ghop)


def _fixture():
    backend = _AnalyticBackend()
    k = np.zeros((3, 3)); k[:, 0] = np.arange(3) / 3
    frame = build_native_spinor_frame(
        backend.evaluate_h(k), np.broadcast_to(np.eye(4, dtype=complex), (3, 4, 4)).copy(), [0, 2, 1],
        kpoints=k, orbital_centers=backend.metadata.tau, orbital_masks=np.eye(2, dtype=bool),
        magnetic_site_positions=backend.metadata.tau,
        local_frames=np.array([np.eye(3), np.diag([1., -1., -1.])]), atomic_spin_order="interleaved",
    )
    config = dict(energy_min_eV=-10., fermi_energy_eV=0., eta_eV=.02, perturbation_chunk=2,
                  include_direct_vertex=True, g_xc_source="fixed_frame_tr_odd", g_pair_policy="raw")
    return backend, frame, config


def _legacy_reference(backend, frame, config, q):
    result_b, result_d = [], []
    positions = np.repeat(backend.metadata.tau, 3, axis=0)
    integral = dict(energy_min_eV=config["energy_min_eV"], occupied_energy_max_eV=config["fermi_energy_eV"],
                    eta_eV=config["eta_eV"], perturbation_chunk=config["perturbation_chunk"])
    for point in (q, -q):
        mapping = build_kq_map(frame.kpoints, point)
        g = backend.evaluate_g(frame.kpoints, point)
        gm = backend.evaluate_g(frame.kpoints, -point)
        transformed = frame.transform_vertex(g, q_red=point, perturbation_positions=positions)
        torque = _dagger(frame.finite_q_vertices(point)).reshape(3, 4, 4, 4)
        result_b.append(retarded_bubble_loop_eigh_zero_temperature_finite_q(
            frame.hamiltonian_eV, frame.hamiltonian_at_indices(mapping.indices, mapping.G_wrap),
            torque, transformed, np.full(3, 1/3), **integral,
        ))
        gxc = frame.model_exchange_derivative_wannier(g, gm, q_red=point)
        gxc = frame.transform_vertex(gxc, q_red=point, perturbation_positions=positions)
        vertex = finite_q_direct_vertices(
            gxc, kpoints=frame.kpoints, q_red=point, orbital_masks=frame.orbital_masks,
            local_frames=frame.local_frames, orbital_centers=frame.orbital_centers,
            magnetic_site_positions=frame.magnetic_site_positions,
        )
        result_d.append(retarded_direct_loop_eigh_zero_temperature(
            frame.hamiltonian_eV, vertex.reshape(3, 4, 6, 4, 4), np.full(3, 1/3), **integral,
        ))
    return np.array(result_b), np.array(result_d)


@pytest.mark.parametrize("policy", ["raw", "hermitian_pair_average"])
def test_arbitrary_response_reproduces_commensurate_native_bubble_and_direct(policy):
    backend, frame, config = _fixture()
    config["g_pair_policy"] = policy
    q = np.array([1/3, 0., 0.])
    response = ArbitraryQResponse(frame, backend, [3, 1, 1], config)
    actual = response.evaluate_pair(q)
    bubble, direct = _legacy_reference(backend, frame, config, q)
    np.testing.assert_allclose(actual["retarded_bubble"], bubble, atol=3.e-15)
    np.testing.assert_allclose(actual["retarded_direct"], direct, atol=3.e-15)
    expected = finalize_q_pair(bubble[0] + direct[0], bubble[1] + direct[1]).reshape(2, 2, 2, 3)
    np.testing.assert_allclose(actual["total"][0], expected, atol=3.e-15)
    np.testing.assert_allclose(actual["total"][1], actual["total"][0].conj(), atol=1.e-16)


def test_arbitrary_response_works_without_commensurate_q_and_reports_raw_reciprocity():
    backend, frame, config = _fixture()
    response = ArbitraryQResponse(frame, backend, [5, 2, 1], config)
    q = np.array([.137, .223, -.071])
    result = response.evaluate_pair(q)
    assert result["total"].shape == (2, 2, 2, 2, 3)
    np.testing.assert_allclose(result["total"], result["bubble"] + result["direct"], atol=0.)
    assert max(result["diagnostics"]["g_reciprocity_relative"]) < 1.e-14
    assert max(result["diagnostics"]["direct"]["g_xc_hermitian_pair_relative"]) < 1.e-14
    assert np.max(abs(result["total"])) > 1.e-5
    np.testing.assert_allclose(response.evaluate_pair(-q)["total"], result["total"][::-1], atol=3.e-15)
    config["include_direct_vertex"] = False
    bubble = ArbitraryQResponse(frame, backend, [5, 2, 1], config).evaluate_pair(q)
    np.testing.assert_allclose(bubble["direct"], 0.)
    np.testing.assert_allclose(bubble["total"], result["bubble"], atol=1.e-15)


def test_explicit_shifted_direct_vertex_matches_wrapped_lookup_and_allows_off_grid():
    _, frame, _ = _fixture()
    rng = np.random.default_rng(171)
    g = rng.normal(size=(3, 6, 4, 4)) + 1j * rng.normal(size=(3, 6, 4, 4))
    q = np.array([1/3, 0., 0.])
    mapping = build_kq_map(frame.kpoints, -q)
    wrap = atomic_gauge_matrix(mapping.G_wrap, frame.orbital_centers)[:, None]
    shifted = _dagger(wrap) @ g[mapping.indices] @ wrap
    kwargs = dict(kpoints=frame.kpoints, q_red=q, orbital_masks=frame.orbital_masks,
                  local_frames=frame.local_frames, orbital_centers=frame.orbital_centers,
                  magnetic_site_positions=frame.magnetic_site_positions)
    np.testing.assert_allclose(finite_q_direct_vertices(g, g_xc_at_k_minus_q=shifted, **kwargs),
                               finite_q_direct_vertices(g, **kwargs), atol=0.)
    kwargs["q_red"] = np.array([.137, .223, -.071])
    result = finite_q_direct_vertices(g, g_xc_at_k_minus_q=shifted, **kwargs)
    assert result.shape == (3, 2, 2, 6, 4, 4)
    with pytest.raises(ValueError, match="same finite shape"):
        finite_q_direct_vertices(g, g_xc_at_k_minus_q=shifted[:, :1], **kwargs)


def test_bad_interpolated_response_configuration_rejected():
    backend, frame, config = _fixture()
    with pytest.raises(ValueError, match="three positive integers"):
        ArbitraryQResponse(frame, backend, [3., 1., 1.], config)
    config["g_xc_source"] = "total_g"
    with pytest.raises(ValueError, match="fixed_frame_tr_odd"):
        ArbitraryQResponse(frame, backend, [3, 1, 1], config)
