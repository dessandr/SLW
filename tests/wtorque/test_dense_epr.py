"""Arbitrary k,q interpolation, stored WS weights, and electronic polar port."""

from itertools import product

import h5py
import numpy as np
import pytest
from scipy.constants import physical_constants

from slw.core.constants import RY_TO_EV
from slw.core.qe2pert_ws import init_rvec_images, set_wigner_seitz_cell, triangular_pair_index
from slw.wtorque.io.dense_epr import DenseEPREvaluator, electronic_longrange_3d
from slw.wtorque.io.native_epr_h import reconstruct_epr_hamiltonian
from slw.wtorque.io.native_epr_g import reconstruct_epr_g_at_q


def _fixture(path):
    rng = np.random.default_rng(210)
    lattice = np.array([[1., .2, -.1], [0., 1.2, .1], [.1, -.2, .9]])
    centres = np.array([[.07, .2, .3], [.48, .6, -.1]])
    tau = np.array([[.17, -.23, .31]])
    kmesh, qmesh = (2, 2, 2), (2, 2, 2)
    eimages, pimages = init_rvec_images(kmesh, lattice), init_rvec_images(qmesh, lattice)
    expected_g, expected_h = {}, {}
    with h5py.File(path, "w") as handle:
        basic = handle.create_group("basic_data")
        for name, value in {
            "nat": 1, "num_wann": 2, "kc_dim": kmesh, "qc_dim": qmesh,
            "at": lattice.T, "tau": tau @ lattice.T,
            "wannier_center_cryst": centres, "spinor": 1, "lpolar": 0, "lquad": 0,
        }.items():
            basic[name] = value
        hgroup, ggroup = handle.create_group("electron_wannier"), handle.create_group("eph_matrix_wannier")
        for i, j in product(range(2), repeat=2):
            re = set_wigner_seitz_cell(eimages, lattice, centres[i], centres[j])
            rp = set_wigner_seitz_cell(pimages, lattice, centres[i], tau[0])
            # These coefficients already include weights, deliberately not all
            # weights are one.  The direct sum below must not divide again.
            coeff = rng.normal(size=(rp.nr, re.nr, 3)) + 1j*rng.normal(size=(rp.nr, re.nr, 3))
            coeff /= rp.ndeg[:, None, None] * re.ndeg[None, :, None]
            ggroup[f"ep_hop_r_1_{j+1}_{i+1}"] = coeff.real
            ggroup[f"ep_hop_i_1_{j+1}_{i+1}"] = coeff.imag
            expected_g[i, j] = re.vectors, rp.vectors, coeff
            if i <= j:
                h = rng.normal(size=re.nr) + 1j*rng.normal(size=re.nr)
                pair = triangular_pair_index(i+1, j+1)
                hgroup[f"hopping_r{pair}"], hgroup[f"hopping_i{pair}"] = h.real, h.imag
                expected_h[i, j] = re.vectors, h
    return expected_h, expected_g


def test_arbitrary_fourier_against_direct_sum_and_periodic_k(tmp_path):
    path = tmp_path / "input.h5"
    hdata, gdata = _fixture(path)
    backend = DenseEPREvaluator(path, energy_unit="ev", displacement_unit="angstrom")
    k = np.array([[.137, -.243, 1.219], [.21, .47, -.16]])
    q = np.array([.127, -.374, .298])
    h, g = backend.evaluate_h(k), backend.evaluate_g(k, q)
    for (i, j), (re, coeff) in hdata.items():
        expected = np.array([sum(np.exp(2j*np.pi*np.dot(point, r))*value for r, value in zip(re, coeff)) for point in k])
        np.testing.assert_allclose(h[:, i, j], expected, atol=2.e-14)
        if i != j:
            np.testing.assert_allclose(h[:, j, i], expected.conj(), atol=2.e-14)
    for (i, j), (re, rp, coeff) in gdata.items():
        expected = np.zeros((len(k), 3), complex)
        for ik, point in enumerate(k):
            for ip, p in enumerate(rp):
                for ie, e in enumerate(re):
                    expected[ik] += np.exp(2j*np.pi*(point@e + q@p))*coeff[ip, ie]
        np.testing.assert_allclose(g[:, :, i, j], expected, atol=6.e-14)
    np.testing.assert_allclose(backend.evaluate_g(k+[2,-3,1], q), g, atol=3.e-13)
    np.testing.assert_allclose(backend.evaluate_h(k+[2,-3,1]), h, atol=2.e-13)
    assert np.linalg.norm(g - g.swapaxes(-1,-2).conj()) > 1.


def test_coarse_fft_agreement_and_cache_reuse(tmp_path):
    path = tmp_path / "input.h5"
    _fixture(path)
    backend = DenseEPREvaluator(path, energy_unit="ev", displacement_unit="angstrom", cache_dir=tmp_path/"cache")
    coarse = reconstruct_epr_g_at_q(path, 5, energy_unit="ev", displacement_unit="angstrom")
    np.testing.assert_allclose(backend.evaluate_g(coarse.kpoints, coarse.qpoint), coarse.values, atol=3.e-14)
    reconstructed_h = reconstruct_epr_hamiltonian(path, energy_unit="ev", expected_spinor=True)
    direct_h = backend.evaluate_h(reconstructed_h.kpoints)
    np.testing.assert_allclose((direct_h+direct_h.swapaxes(-1,-2).conj())/2, reconstructed_h.values, atol=2.e-14)
    cached = DenseEPREvaluator(path, energy_unit="ev", displacement_unit="angstrom", cache_dir=tmp_path/"cache")
    assert all(isinstance(group.values, np.memmap) for group in cached._ggroups)
    assert cached.cache_path == backend.cache_path
    np.testing.assert_array_equal(cached.evaluate_g([[.13,.47,.24]], [.18,-.22,.15]), backend.evaluate_g([[.13,.47,.24]], [.18,-.22,.15]))
    assert cached.diagnostics["eph_pairs"] == 4


def test_point_center_rejects_nonpolar_input(tmp_path):
    path = tmp_path / "nonpolar.h5"
    _fixture(path)
    with pytest.raises(ValueError, match="requires a polar EPR"):
        DenseEPREvaluator(
            path, energy_unit="ev", displacement_unit="angstrom",
            longrange_model="point_center",
            longrange_coarse_qpoints=list(product([0., .5], repeat=3)),
        )


def test_combined_resplitting_uses_actual_sr_images_near_ws_tie(tmp_path):
    # A source single-distance tie is split by the two-distance criterion.
    # Construct g_coarse=0; the selected full result must be
    # L_new(q) - I_selected[L_new(qc)] regardless of the old LR split.
    from slw.wtorque.io.epr_longrange import PolarLongRange3D

    path = tmp_path / "near_tie.h5"
    at = np.eye(3)
    wc = np.array([[.2, 0., 0.], [.2, 0., 0.]])
    tau = np.array([[1.2 + .375e-6, 0., 0.]])
    qc = np.array([[0., 0., 0.], [.5, 0., 0.]])
    meta = dict(nat=1, qc_dim=(2, 1, 1), alat=6., volume=100., bg=at,
                epsil=4*at, zstar=np.array([2*at]), tau_cart=tau,
                polar_alpha=1., system_2d=False)
    source_lr = np.array([electronic_longrange_3d(meta, q) for q in qc])
    pws = set_wigner_seitz_cell(init_rvec_images((2, 1, 1), at), at, wc[0], tau[0])
    assert pws.nr == 3  # two tied images of residue 0, one of residue 1
    old_sr = (np.exp(-2j*np.pi*(pws.vectors @ qc.T)) @ (-source_lr)
              / len(qc) / pws.ndeg[:, None])
    with h5py.File(path, "w") as h:
        basic = h.create_group("basic_data")
        for name, value in dict(nat=1, num_wann=2, kc_dim=(1, 1, 1), qc_dim=(2, 1, 1),
                                at=at.T, tau=tau, wannier_center_cryst=wc, spinor=1,
                                lpolar=1, lquad=0, mass=[1800.], alat=6., volume=100.,
                                bg=at, epsil=4*at, zstar=np.array([2*at]),
                                polar_alpha=1., system_2d=0).items():
            basic[name] = value
        hg, gg = h.create_group("electron_wannier"), h.create_group("eph_matrix_wannier")
        for i, j in product(range(2), repeat=2):
            value = old_sr[:, None, :] if i == j else np.zeros((pws.nr, 1, 3), complex)
            gg[f"ep_hop_r_1_{j+1}_{i+1}"], gg[f"ep_hop_i_1_{j+1}_{i+1}"] = value.real, value.imag
            if i <= j:
                index = triangular_pair_index(i+1, j+1)
                hg[f"hopping_r{index}"], hg[f"hopping_i{index}"] = [.1], [0.]
    backend = DenseEPREvaluator(path, energy_unit="ry", displacement_unit="bohr",
                                short_range_model="two_center", longrange_model="point_center",
                                longrange_coarse_qpoints=qc)
    q = np.array([.217, 0., 0.])
    # With the stated geometry, the nearest two-center images are -2 and -1.
    selected_rp = np.array([[-2, 0, 0], [-1, 0, 0]])
    polar = PolarLongRange3D(meta, wc)
    new_lr_coarse = np.array([polar.center(point) for point in qc])
    weights = np.exp(2j*np.pi*((q[None, :] - qc) @ selected_rp.T)).sum(axis=1) / len(qc)
    expected_diagonal = polar.center(q) - np.einsum("q,qpi->pi", weights, new_lr_coarse)
    expected_diagonal *= RY_TO_EV / physical_constants["Bohr radius"][0] / 1e10
    actual = backend.evaluate_g([[.131, 0., 0.]], q)
    np.testing.assert_allclose(actual[0, :, np.arange(2), np.arange(2)].T,
                               expected_diagonal, atol=2e-14)
    np.testing.assert_allclose(actual[0, :, 0, 1], 0., atol=2e-14)


def _polar_meta():
    return {
        "nat": 2, "qc_dim": (3,3,3), "alat": 6.3, "volume": 171.2,
        "bg": np.array([[1.1,.2,-.1], [.0,1.4,.2], [.1,-.1,.9]]),
        "epsil": np.array([[3.,.2,.1],[.2,5.,-.1],[.1,-.1,4.]]),
        "zstar": np.array([[[2.,.3,-.1],[.7,1.8,.4],[.1,.5,2.2]], [[-2.,-.4,.1],[-.6,-1.9,-.3],[-.1,-.5,-2.1]]]),
        "tau_cart": np.array([[.1,.2,.3], [.3,-.2,.4]]),
        "polar_alpha": .47, "loto_alpha": 7., "system_2d": False,
    }


def test_electronic_longrange_independent_scalar_sum_and_gamma():
    meta = _polar_meta()
    q = np.array([.127, -.318, .216])
    alpha, bg, eps = meta["polar_alpha"], meta["bg"], meta["epsil"]
    bounds = [int(np.ceil(np.sqrt(56*alpha/(bg[:,a]@eps@bg[:,a])))) for a in range(3)]
    expected = np.zeros((2,3), complex)
    for shift in product(*(range(-n,n+1) for n in bounds)):
        cart = bg @ (q+shift)
        norm = cart @ eps @ cart
        if norm < 1.e-14 or norm > 56*alpha:
            continue
        for atom, axis in product(range(2), range(3)):
            charge = sum(cart[field]*meta["zstar"][atom,field,axis] for field in range(3))
            expected[atom,axis] += charge*np.exp(-norm/(4*alpha))/norm*np.exp(-2j*np.pi*(cart@meta["tau_cart"][atom]))
    expected *= 8j*np.pi/(meta["volume"]*(2*np.pi/meta["alat"]))
    np.testing.assert_allclose(electronic_longrange_3d(meta, q), expected.ravel(), atol=1.e-16)
    np.testing.assert_allclose(electronic_longrange_3d(meta,-q), expected.ravel().conj(), atol=1.e-16)
    np.testing.assert_array_equal(electronic_longrange_3d(meta, [0,0,0]), np.zeros(6))
    with pytest.raises(NotImplementedError, match="3D"):
        electronic_longrange_3d(dict(meta, system_2d=True), q)
    with pytest.raises(NotImplementedError, match="quadrupole"):
        electronic_longrange_3d(dict(meta, lquad=True), q)


@pytest.mark.parametrize("q", [[.5, 0, 0], [.5, .25, -.5], [1.31, -.72, .49]])
def test_source_polar_sum_keeps_exact_q_minus_q_partners(q):
    meta = _polar_meta()
    plus = electronic_longrange_3d(meta, q)
    minus = electronic_longrange_3d(meta, -np.asarray(q))
    np.testing.assert_allclose(plus, minus.conj(), atol=3.e-16)


def test_source_finite_g_sum_is_explicitly_not_reciprocal_periodic():
    # The source has a Gamma-only reduced-norm cutoff and a fixed G box.
    # Folding here would silently change that source prescription; at half
    # boundaries, such folding also introduces a finite discontinuity.
    meta = _polar_meta()
    gamma = electronic_longrange_3d(meta, [0, 0, 0])
    shifted = electronic_longrange_3d(meta, [1, 0, 0])
    assert np.linalg.norm(shifted - gamma) > 1.e-8


def test_polar_hdf_tensor_orientation_addback_and_units(tmp_path):
    path = tmp_path/"polar.h5"
    _fixture(path)
    meta = _polar_meta()
    # One atom fixture with nonsymmetric Z: transposition errors must show up.
    with h5py.File(path,"a") as f:
        basic=f["basic_data"]
        basic["lpolar"][...] = 1
        for name, value in {
            "mass": [1800.], "alat": meta["alat"], "volume": meta["volume"],
            "bg": meta["bg"].T, "epsil": meta["epsil"].T,
            "zstar": meta["zstar"][:1].swapaxes(-1,-2), "polar_alpha": meta["polar_alpha"],
            "loto_alpha": meta["loto_alpha"], "system_2d": 0,
        }.items():
            basic[name] = value
    backend = DenseEPREvaluator(path, energy_unit="ev", displacement_unit="angstrom")
    k, q = [[.2,.3,.4]], [.14,-.25,.17]
    short, full = backend.evaluate_g(k,q,include_longrange=False), backend.evaluate_g(k,q)
    expected = electronic_longrange_3d(backend.polar_metadata,q) * RY_TO_EV/(physical_constants["Bohr radius"][0]/1.e-10)
    np.testing.assert_allclose(full-short, expected[None,:,None,None]*np.eye(2), atol=2.e-15)
    np.testing.assert_array_equal(backend.polar_metadata["zstar"], meta["zstar"][:1])


def test_incomplete_payload_and_spinor_boundary_fail(tmp_path):
    path = tmp_path/"incomplete.h5"
    _fixture(path)
    with pytest.raises(ValueError, match="spinor"):
        DenseEPREvaluator(path,energy_unit="ev",displacement_unit="angstrom",expected_spinor=False)
    with h5py.File(path,"a") as f:
        del f["eph_matrix_wannier/ep_hop_i_1_2_1"]
    with pytest.raises(KeyError,match="incomplete"):
        DenseEPREvaluator(path,energy_unit="ev",displacement_unit="angstrom")


@pytest.mark.parametrize("short_range_model", ["source", "two_center"])
def test_point_center_resplitting_preserves_coarse_data_and_separates_sr(tmp_path, short_range_model):
    path = tmp_path / "resplit.h5"
    _fixture(path)
    polar = _polar_meta()
    with h5py.File(path, "a") as f:
        basic = f["basic_data"]
        basic["lpolar"][...] = 1
        for key, value in {
            "mass": [1800.], "alat": polar["alat"], "volume": polar["volume"],
            "bg": polar["bg"].T, "epsil": polar["epsil"].T,
            "zstar": polar["zstar"][:1].swapaxes(-1, -2),
            "polar_alpha": polar["polar_alpha"], "system_2d": 0,
        }.items():
            basic[key] = value
    original_bytes = path.read_bytes()
    q_coarse = np.indices((2, 2, 2)).reshape(3, -1).T / 2
    source = DenseEPREvaluator(path, energy_unit="ev", displacement_unit="angstrom")
    selected = DenseEPREvaluator(
        path, energy_unit="ev", displacement_unit="angstrom",
        longrange_model="point_center", longrange_coarse_qpoints=q_coarse,
        short_range_model=short_range_model,
    )
    k = np.array([[.137, -.243, .219], [.21, .47, -.16]])
    for q in q_coarse:
        np.testing.assert_allclose(selected.evaluate_g(k, q), source.evaluate_g(k, q),
                                   atol=2.e-12, rtol=2.e-13)
    q = np.array([.127, -.374, .298])
    full = selected.evaluate_g(k, q)
    sr = selected.evaluate_g(k, q, include_longrange=False)
    lr = selected.longrange_diagonal(q)[None, :, :, None] * np.eye(2)
    np.testing.assert_allclose(full, sr + lr, atol=1.e-13)
    assert np.linalg.norm(sr-source.evaluate_g(k, q, include_longrange=False)) > 1.e-4
    # The legacy scalar remains explicitly the source model.
    np.testing.assert_array_equal(selected.longrange(q), source.longrange(q))
    assert selected.diagnostics["longrange_resplitting"] is True
    assert path.read_bytes() == original_bytes


def test_point_center_requires_actual_coarse_q_representatives(tmp_path):
    with pytest.raises(ValueError, match="actual source q"):
        DenseEPREvaluator(tmp_path / "not_read.h5", energy_unit="ev", displacement_unit="angstrom",
                          longrange_model="point_center")
    with pytest.raises(ValueError, match="requires longrange_model"):
        DenseEPREvaluator(tmp_path / "not_read.h5", energy_unit="ev", displacement_unit="angstrom",
                          longrange_coarse_qpoints=[[0., 0., 0.]])
