from __future__ import annotations

import numpy as np
import pytest

from slw.wtorque.basis import SpinOrder, canonicalize_spin_order, restore_spin_order
from slw.wtorque.model.exchange_field import split_explicit, split_time_reversal
from slw.wtorque.model.local_frames import validate_local_frames
from slw.wtorque.model.local_projection import localize_exchange
from slw.wtorque.model.magnetic_subspace import MagneticSubspace
from slw.wtorque.model.spinor_wannier import SpinorWannierModel
from slw.wtorque.torque.direct_vertex import direct_rotation_vertices
from slw.wtorque.torque.vertices import (
    finite_rotated_exchange,
    rotation_vertex,
    transverse_vertices,
)


def _toy_model() -> SpinorWannierModel:
    hopping = np.diag([-0.4, -0.4]).astype(np.complex128)
    onsite = np.array([[0.2, 0.05j], [-0.05j, -0.2]], dtype=np.complex128)
    return SpinorWannierModel(
        lattice=np.eye(3),
        R_vectors=np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
        H_R=np.stack((hopping, onsite, hopping)),
        orbital_centers=np.zeros((1, 3)),
        orbital_site=np.array([0]),
        orbital_labels=("d",),
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        weights=np.array([0.5, 0.5]),
        fermi_energy=0.0,
    )


def test_spin_order_round_trip_and_vectorized_fourier_reconstruction():
    model = _toy_model()
    blocked = np.arange(16).reshape(4, 4).astype(np.complex128)
    canonical = canonicalize_spin_order(blocked, 2, SpinOrder.BLOCKED)
    np.testing.assert_array_equal(
        restore_spin_order(canonical, 2, SpinOrder.BLOCKED), blocked
    )

    kpoints = np.array([[0.13, 0.0, 0.0], [0.37, 0.0, 0.0]])
    expected = np.stack(
        [
            model.H_R[1]
            + np.exp(-2j * np.pi * k[0]) * model.H_R[0]
            + np.exp(2j * np.pi * k[0]) * model.H_R[2]
            for k in kpoints
        ]
    )
    np.testing.assert_allclose(model.hamiltonian_batch(kpoints), expected)


def test_exchange_decomposition_and_time_reversal_parity():
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    scalar = 0.3 * np.eye(2, dtype=np.complex128)
    exchange = 0.2 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    full = scalar + exchange
    trs, xc = split_explicit(full, scalar, exchange)
    np.testing.assert_allclose(trs + xc, full)

    sewing = 1j * sy
    trs, xc = split_time_reversal(full, full, sewing)
    np.testing.assert_allclose(trs, scalar)
    np.testing.assert_allclose(xc, exchange)


def test_local_partition_and_exchange_only_rotation_finite_difference():
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    hxc = np.kron(np.array([[1.0, 0.2], [0.2, 0.7]]), sz).astype(
        np.complex128
    )
    subspace = MagneticSubspace.from_masks(
        np.array([[True, False], [False, True]]),
        orbital_labels=("d0", "d1"),
    )
    local, residual = localize_exchange(hxc, subspace.projectors)
    np.testing.assert_allclose(local.sum(axis=0) + residual, hxc)
    assert all(np.allclose(x, x.conj().T) for x in local)

    frame = np.eye(3)[[0, 1, 2]][None, ...]
    validate_local_frames(frame)
    transverse = transverse_vertices(local[0], frame[0])
    np.testing.assert_allclose(transverse[0], rotation_vertex(local[0], sy))
    np.testing.assert_allclose(transverse[1], -rotation_vertex(local[0], sx))
    np.testing.assert_allclose(rotation_vertex(local[0], sz), 0.0, atol=1e-14)

    delta = 1.0e-6
    numeric = (
        finite_rotated_exchange(local[0], sy, delta)
        - finite_rotated_exchange(local[0], sy, -delta)
    ) / (2.0 * delta)
    np.testing.assert_allclose(numeric, transverse[0], rtol=2e-9, atol=2e-10)


def test_overlapping_magnetic_masks_are_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        MagneticSubspace.from_masks(np.array([[True, False], [True, False]]))


def test_direct_vertex_is_gated_and_uses_exchange_resolved_derivative_only():
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    g_xc = np.array([[0.2, 0], [0, -0.2]], dtype=np.complex128)
    projector = np.eye(2, dtype=np.complex128)[None]
    with pytest.raises(ValueError, match="g_XC"):
        direct_rotation_vertices(None, projector, [[0, 1, 0]], include=True)
    direct = direct_rotation_vertices(
        g_xc,
        projector,
        [[0, 1, 0]],
        include=True,
    )
    expected = 0.5j * (g_xc @ sy - sy @ g_xc)
    np.testing.assert_allclose(direct[0], expected)
