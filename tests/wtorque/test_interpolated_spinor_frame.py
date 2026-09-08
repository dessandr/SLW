from __future__ import annotations

import numpy as np
import pytest

from slw.wtorque.model.interpolated_spinor_frame import InterpolatedSpinorFrame
from slw.wtorque.model.native_spinor_frame import build_native_spinor_frame


def _dagger(value):
    return value.conj().swapaxes(-1, -2)


def _unitary(rng, dimension):
    return np.linalg.qr(rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension)))[0]


def _native(q, centers, k=None):
    nk, nw, _ = q.shape
    if k is None:
        k = np.zeros((nk, 3))
        k[:, 0] = np.arange(nk) / nk
    minus = np.array([np.argmin(np.max(abs(k + point - np.rint(k + point)), axis=1)) for point in k])
    return build_native_spinor_frame(
        np.broadcast_to(np.diag(np.arange(nw, dtype=np.complex128)), q.shape).copy(),
        np.asarray(q, dtype=np.complex128), minus, kpoints=k,
        orbital_centers=centers, orbital_masks=np.eye(nw // 2, dtype=bool),
        magnetic_site_positions=centers, local_frames=np.repeat(np.eye(3)[None], nw // 2, axis=0),
        atomic_spin_order="interleaved",
    )


def _interpolator(native, wannier_centers=None):
    if wannier_centers is None:
        wannier_centers = np.repeat(native.orbital_centers, 2, axis=0)
    return InterpolatedSpinorFrame(
        native, kmesh=[len(native.kpoints), 1, 1],
        lattice_column_vectors=np.eye(3), wannier_centers=wannier_centers,
    )


def test_ws_frame_reproduces_permuted_coarse_grid_and_is_unitary_off_grid():
    rng = np.random.default_rng(9701)
    centers = np.array([[0., 0., 0.], [.34, .1, -.2]])
    q = np.stack([_unitary(rng, 4) for _ in range(3)])
    order = np.array([2, 0, 1])
    k = np.zeros((3, 3))
    k[:, 0] = order / 3
    native = _native(q[order], centers, k)
    interpolation = _interpolator(native)
    np.testing.assert_allclose(interpolation.periodic_frame(k), native.atomic_frame, atol=2.e-14)
    points = np.array([[.11, .2, -.1], [.41, -.1, .2], [1.21, .7, -.8]])
    dense = interpolation.evaluate(points)
    np.testing.assert_allclose(_dagger(dense.periodic_frame) @ dense.periodic_frame,
                               np.broadcast_to(np.eye(4), (3, 4, 4)), atol=3.e-14)
    np.testing.assert_allclose(interpolation.periodic_frame(points + np.array([1, -2, 3])),
                               dense.periodic_frame, atol=3.e-14)
    assert dense.diagnostics["interpolated_min_singular_value"] > 0
    assert not interpolation.diagnostics["new_first_principles_projection_data"]


def test_ws_overlap_row_column_orientation_matches_single_real_space_hop():
    k = np.zeros((3, 3))
    k[:, 0] = np.arange(3) / 3
    q = np.exp(2j * np.pi * k[:, 0, None, None]) * np.eye(2)
    native = _native(q, np.zeros((1, 3)))
    # <w_0|trial_R> is local at R=+1 when row center is +0.8 and trial center 0.
    interpolation = _interpolator(native, np.array([[.8, 0., 0.], [.8, 0., 0.]]))
    off = np.array([[.17, .0, .0], [.57, .0, .0], [1.19, .0, .0]])
    expected = np.exp(2j * np.pi * off[:, 0, None, None]) * np.eye(2)
    np.testing.assert_allclose(interpolation.periodic_frame(off), expected, atol=3.e-14)


def test_even_mesh_boundary_images_are_shared_and_rank_loss_is_rejected():
    native = _native(np.array([np.eye(2), -np.eye(2)], complex), np.zeros((1, 3)))
    interpolation = _interpolator(native)
    # The ±1 Nyquist images contribute equally: Q_raw(k)=cos(2πk) I.
    shifts = interpolation.cell_shifts[:, 0]
    for sign in (-1, 1):
        np.testing.assert_allclose(interpolation.coefficients[shifts == sign].sum(axis=0), .5 * np.eye(2), atol=1.e-14)
    with pytest.raises(ValueError, match="rank deficient"):
        interpolation.evaluate([[.25, 0., 0.]])


def test_constant_equal_center_block_rotations_commute_with_interpolation():
    rng = np.random.default_rng(307)
    centers = np.array([[0., 0., 0.], [.37, .1, -.2]])
    q = np.stack([_unitary(rng, 4) for _ in range(3)])
    left, right = np.zeros((4, 4), complex), np.zeros((4, 4), complex)
    for start in (0, 2):
        left[start:start+2, start:start+2] = _unitary(rng, 2)
        right[start:start+2, start:start+2] = _unitary(rng, 2)
    reference = _interpolator(_native(q, centers))
    rotated = _interpolator(_native(left @ q @ right, centers))
    points = np.array([[.07, .2, .4], [.47, -.3, .4], [.77, .9, -.2]])
    np.testing.assert_allclose(rotated.periodic_frame(points),
                               left @ reference.periodic_frame(points) @ right, atol=5.e-14)


def test_continuous_vertex_matches_coarse_transform_and_sewing_squares_to_minus_one():
    rng = np.random.default_rng(616)
    centers = np.array([[0., 0., 0.], [.37, .1, -.2]])
    native = _native(np.stack([_unitary(rng, 4) for _ in range(3)]), centers)
    interpolation = _interpolator(native)
    q = np.array([1/3, 0., 0.])
    source = interpolation.evaluate(native.kpoints)
    final = interpolation.evaluate(native.kpoints + q)
    g = rng.normal(size=(3, 2, 4, 4)) + 1j * rng.normal(size=(3, 2, 4, 4))
    np.testing.assert_allclose(
        interpolation.transform_vertex(g, source, final, perturbation_positions=centers),
        native.transform_vertex(g, q_red=q, perturbation_positions=centers), atol=6.e-14,
    )
    points = np.array([[.17, .2, .4], [1.41, -.2, .8]])
    positive, negative = interpolation.evaluate(points), interpolation.evaluate(-points)
    b = interpolation.sewing(positive, negative)
    bm = interpolation.sewing(negative, positive)
    np.testing.assert_allclose(b @ bm.conj(), -np.broadcast_to(np.eye(4), b.shape), atol=3.e-14)
    with pytest.raises(ValueError, match="represent -k"):
        interpolation.sewing(positive, positive)


def test_continuous_onsite_vertex_phases_cancel_for_arbitrary_unwrapped_q():
    centers = np.array([[.3, .1, -.2]])
    interpolation = _interpolator(_native(np.repeat(np.eye(2, dtype=complex)[None], 3, axis=0), centers))
    k = np.array([[.1, .3, .2], [.81, -.7, .8]])
    source, final = interpolation.evaluate(k), interpolation.evaluate(k + [.57, -.21, .11])
    g = np.broadcast_to(np.eye(2, dtype=complex), (2, 1, 2, 2)).copy()
    result = interpolation.transform_vertex(g, source, final, perturbation_positions=centers)
    np.testing.assert_allclose(result, g, atol=3.e-14)


def test_invalid_mesh_and_grid_rejected():
    native = _native(np.repeat(np.eye(2, dtype=complex)[None], 3, axis=0), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="three positive integers"):
        InterpolatedSpinorFrame(native, kmesh=[3., 1., 1.], lattice_column_vectors=np.eye(3), wannier_centers=np.zeros((2, 3)))
    native.kpoints[1] += [.1, 0., 0.]
    with pytest.raises(ValueError, match="unshifted kmesh"):
        _interpolator(native)
