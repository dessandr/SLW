from __future__ import annotations

import numpy as np
import pytest

from slw.wtorque.gauge.kq_map import build_kq_map
from slw.wtorque.model.native_spinor_frame import (
    build_native_spinor_frame,
    read_spinor_projection_metadata,
)
from slw.wtorque.torque.vertices import PAULI


def _dagger(value):
    return value.conj().swapaxes(-1, -2)


def _unitaries(rng, nk, nw):
    return np.asarray(
        [np.linalg.qr(rng.normal(size=(nw, nw)) + 1j * rng.normal(size=(nw, nw)))[0] for _ in range(nk)],
        dtype=np.complex128,
    )


def _fixture():
    k = np.asarray([[0, 0, 0], [1 / 3, 0, 0], [2 / 3, 0, 0]], dtype=float)
    centers = np.asarray([[0, 0, 0], [0.37, 0.21, -0.1]], dtype=float)
    h = np.zeros((3, 4, 4), dtype=np.complex128)
    h[:, :2, :2] = 1.3 * PAULI[2]
    h[:, 2:, 2:] = -0.9 * PAULI[2]
    # Real hopping and TR-even imaginary-orbital SOC remain in the full H.
    h[:, :2, 2:] = 0.4 * np.eye(2) + 0.07j * PAULI[0]
    h[:, 2:, :2] = _dagger(h[:, :2, 2:])
    return h, np.broadcast_to(np.eye(4, dtype=np.complex128), h.shape).copy(), np.array([0, 2, 1]), {
        "kpoints": k,
        "orbital_centers": centers,
        "orbital_masks": np.eye(2, dtype=bool),
        "magnetic_site_positions": centers.copy(),
        "local_frames": np.asarray([np.eye(3), np.diag([1, -1, -1])]),
        "atomic_spin_order": "interleaved",
    }


def test_full_frame_removes_arbitrary_k_dependent_wannier_spin_gauge():
    h, overlap, minus, kwargs = _fixture()
    rng = np.random.default_rng(913)
    gauge = _unitaries(rng, 3, 4)
    spin = np.asarray([np.kron(np.eye(2), axis) for axis in PAULI])
    spin = np.broadcast_to(spin, (3, 3, 4, 4)).copy()
    reference = build_native_spinor_frame(h, overlap, minus, projected_spin_wannier=spin, **kwargs)
    transformed = build_native_spinor_frame(
        _dagger(gauge) @ h @ gauge,
        _dagger(gauge) @ overlap,
        minus,
        projected_spin_wannier=_dagger(gauge)[:, None] @ spin @ gauge[:, None],
        **kwargs,
    )
    np.testing.assert_allclose(transformed.hamiltonian_eV, reference.hamiltonian_eV, atol=3e-14)
    np.testing.assert_allclose(transformed.exchange_eV, reference.exchange_eV, atol=3e-14)
    # The scalar hoppings and time-reversal-even SOC are excluded from exchange.
    expected_xc = np.zeros_like(h)
    expected_xc[:, :2, :2] = 1.3 * PAULI[2]
    expected_xc[:, 2:, 2:] = -0.9 * PAULI[2]
    np.testing.assert_allclose(reference.exchange_eV, expected_xc, atol=3e-14)
    assert transformed.diagnostics["projected_spin_closure_residual"] < 3e-14
    assert transformed.diagnostics["projected_spin_atomic_frame_relative_residual"] < 3e-14

    q = np.array([1 / 3, 0, 0])
    mapping = build_kq_map(kwargs["kpoints"], q)
    g = np.asarray(rng.normal(size=(3, 2, 4, 4)) + 1j * rng.normal(size=(3, 2, 4, 4)), dtype=np.complex128)
    g_rotated = _dagger(gauge[mapping.indices])[:, None] @ g @ gauge[:, None]
    g_ref = reference.transform_vertex(g, q_red=q, perturbation_positions=kwargs["orbital_centers"])
    g_result = transformed.transform_vertex(g_rotated, q_red=q, perturbation_positions=kwargs["orbital_centers"])
    np.testing.assert_allclose(g_result, g_ref, atol=3e-14)
    np.testing.assert_allclose(transformed.finite_q_vertices(q), reference.finite_q_vertices(q), atol=3e-14)


def test_atomic_phase_uses_unwrapped_final_momentum_and_displacement_site():
    h, overlap, minus, kwargs = _fixture()
    frame = build_native_spinor_frame(h, overlap, minus, **kwargs)
    q = np.array([1 / 3, 0, 0])
    g = np.zeros((3, 2, 4, 4), dtype=np.complex128)
    g[:, 0, :2, :2] = np.eye(2)
    g[:, 1, 2:, 2:] = 2 * np.eye(2)
    # An on-site derivative co-located with its orbital has no remaining phase,
    # including the k=2/3 → k+q=1 boundary, independently of library gauge code.
    result = frame.transform_vertex(g, q_red=q, perturbation_positions=kwargs["orbital_centers"])
    np.testing.assert_allclose(result, g, atol=1e-14)
    # A nonlocal matrix element gives the directly evaluated physical phase.
    g[:, 0, 0, 2] = 1.0 + 2.0j
    result = frame.transform_vertex(g, q_red=q, perturbation_positions=kwargs["orbital_centers"])
    k = kwargs["kpoints"]
    centers = kwargs["orbital_centers"]
    phase = np.exp(2j * np.pi * (k @ centers[1] - (k + q) @ centers[0] + q @ centers[0]))
    np.testing.assert_allclose(result[:, 0, 0, 2], (1 + 2j) * phase, atol=1e-14)


def test_finite_q_afm_tangents_and_reciprocal_partner():
    h, overlap, minus, kwargs = _fixture()
    # Add real exchange hopping so the test covers different endpoint gauges.
    h[:, :2, 2:] += 0.23 * PAULI[2]
    h[:, 2:, :2] = _dagger(h[:, :2, 2:])
    frame = build_native_spinor_frame(h, overlap, minus, **kwargs)
    q = np.array([1 / 3, 0, 0])
    vertices = frame.finite_q_vertices(q)
    # Independent derivative of Δ n.sigma with respect to tangent components.
    np.testing.assert_allclose(vertices[:, 0, 0, :2, :2], np.broadcast_to(1.3 * PAULI[0], (3, 2, 2)), atol=1e-14)
    np.testing.assert_allclose(vertices[:, 0, 1, :2, :2], np.broadcast_to(1.3 * PAULI[1], (3, 2, 2)), atol=1e-14)
    np.testing.assert_allclose(vertices[:, 1, 0, 2:, 2:], np.broadcast_to(0.9 * PAULI[0], (3, 2, 2)), atol=1e-14)
    np.testing.assert_allclose(vertices[:, 1, 1, 2:, 2:], np.broadcast_to(-0.9 * PAULI[1], (3, 2, 2)), atol=1e-14)
    mapping = build_kq_map(kwargs["kpoints"], q)
    reverse = frame.finite_q_vertices(-q)[mapping.indices]
    centers_spinor = np.repeat(kwargs["orbital_centers"], 2, axis=0)
    phases = np.exp(2j * np.pi * (mapping.G_wrap @ centers_spinor.T))
    reverse_unwrapped = phases.conj()[:, None, None, :, None] * reverse * phases[:, None, None, None, :]
    np.testing.assert_allclose(_dagger(vertices), reverse_unwrapped, atol=2e-14)
    assert np.max(np.abs(vertices - _dagger(vertices))) > 0.01


def test_blocked_input_columns_become_common_interleaved_frame():
    h, overlap, minus, kwargs = _fixture()
    reference = build_native_spinor_frame(h, overlap, minus, **kwargs)
    kwargs["atomic_spin_order"] = "blocked"
    blocked = build_native_spinor_frame(h, overlap[:, :, [0, 2, 1, 3]], minus, **kwargs)
    np.testing.assert_allclose(blocked.hamiltonian_eV, reference.hamiltonian_eV, atol=1e-14)
    np.testing.assert_allclose(blocked.exchange_eV, reference.exchange_eV, atol=1e-14)


def test_rank_covariance_and_minus_mapping_fail_closed():
    h, overlap, minus, kwargs = _fixture()
    deficient = overlap.copy()
    deficient[:, :, -1] = 0
    with pytest.raises(ValueError, match="rank deficient"):
        build_native_spinor_frame(h, deficient, minus, **kwargs)
    asymmetric = overlap.copy()
    asymmetric[:, 0, 0] = 0.5
    with pytest.raises(ValueError, match="TR covariance"):
        build_native_spinor_frame(h, asymmetric, minus, **kwargs)
    with pytest.raises(ValueError, match="does not map"):
        build_native_spinor_frame(h, overlap, [0, 1, 2], **kwargs)


def test_projected_spin_nonclosure_is_reported_without_pauli_assumption():
    h, overlap, minus, kwargs = _fixture()
    spin = 0.9 * np.asarray([np.kron(np.eye(2), axis) for axis in PAULI])
    frame = build_native_spinor_frame(h, overlap, minus, projected_spin_wannier=np.broadcast_to(spin, (3, 3, 4, 4)).copy(), **kwargs)
    assert frame.diagnostics["projected_spin_closure_residual"] == pytest.approx(0.57)
    assert frame.diagnostics["projected_spin_atomic_frame_relative_residual"] == pytest.approx(0.1)


def _nnkp(path, order="interleaved", axis="0 0 1"):
    entries = []
    for position, angular in (("0 0 0", 2), ("0.5 0.5 -0.5", 1)):
        for sign in (1, -1):
            entries.append(f"{position} {angular} 1 1\n0 0 1 1 0 0 1\n{sign} {axis}")
    if order == "blocked":
        entries = [entries[index] for index in (0, 2, 1, 3)]
    path.write_text("begin spinor_projections\n4\n" + "\n".join(entries) + "\nend spinor_projections\n")


@pytest.mark.parametrize("order", ["interleaved", "blocked"])
def test_nnkp_pairs_masks_and_cell_images(tmp_path, order):
    path = tmp_path / "model.nnkp"
    _nnkp(path, order)
    metadata = read_spinor_projection_metadata(path, atomic_spin_order=order)
    np.testing.assert_allclose(metadata.orbital_centers, [[0, 0, 0], [0.5, 0.5, -0.5]])
    masks = metadata.masks_from_projection_groups(metadata.projection_pairs)
    np.testing.assert_array_equal(masks, np.eye(2, dtype=bool))
    with pytest.raises(ValueError, match="both spin"):
        metadata.masks_from_projection_groups([[0]])
    with pytest.raises(ValueError, match="disjoint"):
        metadata.masks_from_projection_groups([metadata.projection_pairs[0]] * 2)


def test_nnkp_rejects_unsupported_spin_axes_and_order(tmp_path):
    path = tmp_path / "model.nnkp"
    _nnkp(path, "interleaved", "1 0 0")
    with pytest.raises(ValueError, match="global z"):
        read_spinor_projection_metadata(path, atomic_spin_order="interleaved")
    _nnkp(path)
    with pytest.raises(ValueError, match="identical spatial"):
        read_spinor_projection_metadata(path, atomic_spin_order="blocked")
