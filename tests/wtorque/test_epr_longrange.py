"""Independent controls for the reusable electronic polar LR models."""

import numpy as np
import pytest

from slw.wtorque.io.epr_longrange import (
    PolarLongRange3D,
    electronic_longrange_3d,
    electronic_longrange_3d_center,
)


def _metadata(*, alpha: float = 1.0e-3, tau_cart=None, zstar=None):
    if tau_cart is None:
        tau_cart = np.array([[0.13, -0.21, 0.17]])
    if zstar is None:
        zstar = np.array(
            [[[2.0, 0.3, -0.1], [0.7, 1.8, 0.4], [0.1, 0.5, 2.2]]]
        )
    return {
        "nat": 1,
        "qc_dim": (2, 2, 2),
        "alat": 1.0,
        "volume": 1.0,
        "bg": np.eye(3),
        "epsil": np.diag([2.0, 3.0, 4.0]),
        "zstar": np.asarray(zstar, dtype=float),
        "tau_cart": np.asarray(tau_cart, dtype=float),
        "polar_alpha": alpha,
    }


def _manual_g0(meta, q, center=None):
    bg = np.asarray(meta["bg"])
    epsil = np.asarray(meta["epsil"])
    zstar = np.asarray(meta["zstar"])
    tau_cart = np.asarray(meta["tau_cart"])
    cart = np.asarray(q) @ bg.T
    norm = cart @ epsil @ cart
    weight = np.exp(-norm / (4.0 * meta["polar_alpha"])) / norm
    phase = np.exp(-2j * np.pi * (cart @ tau_cart.T))
    charge = np.einsum("i,aij->aj", cart, zstar)
    value = weight * phase[:, None] * charge
    if center is not None:
        value *= np.exp(2j * np.pi * (np.asarray(q) @ np.asarray(center).T))[None, :]
    return (value * (8j * np.pi / (meta["volume"] * (2.0 * np.pi / meta["alat"]))))


def test_localized_point_charge_matches_physical_single_g_limit():
    meta = _metadata(alpha=1.0e-3)
    q = np.array([0.01, 0.005, -0.007])
    centers = np.array([[0.37, -0.11, 0.23]])
    # alpha is small enough that the source's finite box retains only G=0 for
    # this q; this is an independent analytic point-charge oracle.
    expected_source = _manual_g0(meta, q).ravel()
    expected_center = _manual_g0(meta, q, centers).reshape(3, 1)
    np.testing.assert_allclose(electronic_longrange_3d(meta, q), expected_source, atol=2.0e-15)
    np.testing.assert_allclose(
        electronic_longrange_3d_center(meta, q, centers), expected_center, atol=2.0e-15
    )


def test_center_model_is_invariant_to_common_origin_translation():
    meta = _metadata(alpha=0.47)
    centers = np.array([[0.37, -0.11, 0.23]])
    q = np.array([0.137, -0.241, 0.193])
    shift = np.array([0.23, -0.17, 0.09])
    translated = dict(meta, tau_cart=meta["tau_cart"] + shift)
    center_translated = electronic_longrange_3d_center(
        translated, q, centers + shift
    )
    center = electronic_longrange_3d_center(meta, q, centers)
    np.testing.assert_allclose(center_translated, center, atol=4.0e-14)
    # The source scalar has an origin phase and therefore is not the
    # translation-invariant localized matrix element by itself.
    assert (
        np.linalg.norm(
            electronic_longrange_3d(translated, q)
            - electronic_longrange_3d(meta, q)
        )
        > 1.0e-7
    )


def test_center_and_source_have_q_minus_q_conjugacy():
    meta = _metadata(alpha=0.47)
    centers = np.array([[0.37, -0.11, 0.23]])
    q = np.array([0.127, -0.318, 0.216])
    np.testing.assert_allclose(
        electronic_longrange_3d(meta, -q), electronic_longrange_3d(meta, q).conj(), atol=5.0e-15
    )
    np.testing.assert_allclose(
        electronic_longrange_3d_center(meta, -q, centers),
        electronic_longrange_3d_center(meta, q, centers).conj(),
        atol=5.0e-15,
    )
    np.testing.assert_array_equal(
        electronic_longrange_3d_center(meta, [0.0, 0.0, 0.0], centers), 0.0
    )


def test_resplitting_correction_preserves_every_coarse_point():
    meta = _metadata(alpha=0.47)
    centers = np.array([[0.37, -0.11, 0.23]])
    qpoints = np.asarray(
        [(i / 2.0, j / 2.0, k / 2.0) for k in range(2) for j in range(2) for i in range(2)]
    )
    model = PolarLongRange3D(
        meta,
        centers,
        coarse_qpoints=qpoints,
        qmesh=(2, 2, 2),
        at=np.eye(3),
        tau=meta["tau_cart"],
    )
    coarse_difference = model._coarse_difference
    assert coarse_difference is not None
    assert coarse_difference.shape == (len(qpoints), 3, 1)
    for q in qpoints:
        # Adding the source-SR correction and selected center LR gives the
        # original source LR at every stored coarse representative.
        np.testing.assert_allclose(
            model.sr_correction(q) + model.center(q), model.source(q)[:, None], atol=3.0e-13
        )

    # An arbitrary fine q receives the exact negative interpolated difference;
    # explicitly repeating the WS interpolation checks the sign and axes.
    fine_q = np.array([0.173, -0.269, 0.311])
    expected = np.empty((3, 1), dtype=complex)
    for wannier in range(1):
        vectors, ndeg = model.ws_cells[0, wannier]
        weights = (
            np.exp(2j * np.pi * ((fine_q - qpoints) @ vectors.T)) / ndeg
        ).sum(axis=1) / len(qpoints)
        expected[:, wannier] = -(weights @ coarse_difference[:, :, wannier])
    np.testing.assert_allclose(model.sr_correction(fine_q), expected, atol=2.0e-14)


def test_centered_coarse_grid_tolerates_roundoff_and_rejects_missing_point():
    meta = _metadata(alpha=0.47)
    meta["qc_dim"] = (3, 3, 3)
    qpoints = np.asarray(
        [(i / 3.0, j / 3.0, k / 3.0) for k in range(3) for j in range(3) for i in range(3)]
    )
    qpoints = (qpoints + 0.5) % 1.0 - 0.5
    qpoints += np.resize(np.array([1.0, -1.0, 0.5]) * 2.0e-16, qpoints.shape)
    model = PolarLongRange3D(
        meta, [[0.37, -0.11, 0.23]], coarse_qpoints=qpoints,
        qmesh=(3, 3, 3), at=np.eye(3), tau=meta["tau_cart"],
    )
    assert model.coarse_qpoints is not None
    missing = qpoints.copy()
    missing[-1] = missing[0]
    with pytest.raises(ValueError, match="complete qmesh"):
        PolarLongRange3D(
            meta, [[0.37, -0.11, 0.23]], coarse_qpoints=missing,
            qmesh=(3, 3, 3), at=np.eye(3), tau=meta["tau_cart"],
        )


def test_gamma_has_the_expected_one_over_q_leading_limit():
    meta = _metadata(alpha=0.47)
    model = PolarLongRange3D(meta, [[0.37, -0.11, 0.23]])
    direction = np.array([1.0, 2.0, -1.0])
    direction /= np.linalg.norm(direction)
    scaled = [
        scale * np.linalg.norm(model.source(scale * direction))
        for scale in (1.0e-3, 1.0e-4, 1.0e-5)
    ]
    np.testing.assert_allclose(scaled[1], scaled[0], rtol=2.0e-3, atol=0.0)
    np.testing.assert_allclose(scaled[2], scaled[1], rtol=2.0e-4, atol=0.0)
    np.testing.assert_array_equal(model.source([0.0, 0.0, 0.0]), 0.0)


def test_nonsymmetric_born_tensor_is_retained_without_symmetrization():
    meta = _metadata(alpha=0.47)
    q = np.array([0.13, -0.19, 0.17])
    full = electronic_longrange_3d(meta, q)
    symmetric = dict(meta, zstar=0.5 * (meta["zstar"] + meta["zstar"].swapaxes(-1, -2)))
    sym = electronic_longrange_3d(symmetric, q)
    assert np.linalg.norm(full - sym) > 1.0e-7


@pytest.mark.parametrize(
    "override, exception, message",
    [
        ({"system_2d": True}, NotImplementedError, "3D"),
        ({"lquad": True}, NotImplementedError, "quadrupole"),
        ({"polar_alpha": 0.0}, ValueError, "polar_alpha"),
    ],
)
def test_polar_capability_and_alpha_guards(override, exception, message):
    with pytest.raises(exception, match=message):
        electronic_longrange_3d(dict(_metadata(), **override), [0.1, 0.2, 0.3])


def test_shape_and_interpolation_configuration_guards():
    meta = _metadata()
    with pytest.raises(ValueError, match="qpoint"):
        electronic_longrange_3d(meta, [0.1, 0.2])
    with pytest.raises(ValueError, match="wannier_centers"):
        electronic_longrange_3d_center(meta, [0.1, 0.2, 0.3], [[0.1, 0.2]])
    model = PolarLongRange3D(meta, [[0.37, -0.11, 0.23]])
    with pytest.raises(ValueError, match="coarse_qpoints"):
        model.sr_correction([0.1, 0.2, 0.3])
