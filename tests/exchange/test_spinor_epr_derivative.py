import numpy as np
import pytest

from slw.exchange.kernels.spinor_epr_derivative import (
    uniform_points, commensurate_map, periodic_spinor_transform, prepare_kdata,
    derivative_A, semicircle_gauss, wannier_pair_masks,
)
from slw.exchange.kernels.dj_tensor_epr import (
    _pauli_block_all, _build_local_tb2j_projector_deriv_from_block,
)


def unitary(rng, shape):
    x = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    return np.linalg.qr(x)[0]


def test_endpoint_gauge_and_spin_flip():
    rng = np.random.default_rng(318)
    source, final = unitary(rng, (3, 4, 4)), unitary(rng, (3, 4, 4))
    trial = rng.normal(size=(3, 2, 4, 4)) + 1j*rng.normal(size=(3, 2, 4, 4))
    wannier = final[:, None] @ trial @ source[:, None].conj().swapaxes(-1, -2)
    order = np.array([0, 2, 1, 3])
    actual = periodic_spinor_transform(wannier, source, final)
    np.testing.assert_allclose(actual, trial[..., order[:, None], order], atol=3.e-14)
    assert np.linalg.norm(actual[..., :2, 2:]) > 1


def test_commensurate_actual_q_representative():
    k = uniform_points((6, 3, 3))
    a = commensurate_map(k, [-1/3, 0, 0], (6, 3, 3))
    b = commensurate_map(k, [2/3, 0, 0], (6, 3, 3))
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="commensurate"):
        commensurate_map(k, [.1, 0, 0], (6, 3, 3))


def test_declared_wannier_pair_geometry():
    centers=np.repeat([[0.,0,0],[.5,.5,0],[.25,0,0]],2,axis=0)
    masks=wannier_pair_masks(centers,[[0,0,0],[.5,.5,0]],np.eye(3),
        maximum_pair_distance_ang=.01,site_radius_ang=.1)
    np.testing.assert_array_equal(masks,[[True,False,False],[False,True,False]])
    centers[1,0]=.2
    with pytest.raises(ValueError,match="colocated"):
        wannier_pair_masks(centers,[[0,0,0]],np.eye(3),maximum_pair_distance_ang=.01,site_radius_ang=.1)


@pytest.mark.parametrize("q_index", [0, 1, -1])
def test_four_terms_against_complex_supercell_finite_difference(q_index):
    """Independent real-space resolvent derivative, including both dP sites.

    A complex displacement Fourier wave is allowed analytically; q and -q
    together form a real displacement. This tests the nonzero-q endpoint
    phase as well as the previously missing remote-site projector response.
    """
    rng = np.random.default_rng(712)
    mesh = (3, 1, 1)
    k = uniform_points(mesh)
    q = np.array([q_index/3, 0., 0.])
    h0 = np.diag([-2., 1.7, 2.1, -1.8]).astype(complex)
    t = (rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))) * .08
    h = np.array([h0 + t*np.exp(2j*np.pi*ki[0]) + t.conj().T*np.exp(-2j*np.pi*ki[0]) for ki in k])
    g = (rng.normal(size=(3, 4, 4)) + 1j*rng.normal(size=(3, 4, 4))) * .13
    data = prepare_kdata(h, [[1, 0], [0, 1]], 0.)
    pair = [dict(li=0, lj=1, R=(1, 0, 0))]
    contour = semicircle_gauss(-5., 0., 16)
    analytic = derivative_A(data, g, k, q, mesh, pair, contour)[0]
    cells = np.arange(3)
    h_real = np.empty((3, 4, 3, 4), complex)
    dh_real = np.empty_like(h_real)
    for x in cells:
        for y in cells:
            phase = np.exp(2j*np.pi*k[:, 0]*(x-y))/3
            h_real[x, :, y, :] = np.einsum("k,kij->ij", phase, h)
            dh_real[x, :, y, :] = np.exp(2j*np.pi*q[0]*x)*np.einsum("k,kij->ij", phase, g)
    hr, dhr = h_real.reshape(12, 12), dh_real.reshape(12, 12)
    ii, jj = np.array([0, 2]), 4+np.array([1, 3])
    dp = {}
    for site, indices in data["site_indices"].items():
        dp[site] = _build_local_tb2j_projector_deriv_from_block(
            g.mean(axis=0)[np.ix_(indices, indices)], data["p_dirs"][site])

    def value(eps):
        out = np.zeros((4, 4), complex)
        pi = data["p_ops"][0] + eps*dp[0]
        pj = data["p_ops"][1] + eps*np.exp(2j*np.pi*q[0])*dp[1]
        for z, dz in contour:
            gr = np.linalg.inv(z*np.eye(12) - hr - eps*dhr)
            x = pi @ np.asarray(_pauli_block_all(gr[np.ix_(ii, jj)]))
            y = pj @ np.asarray(_pauli_block_all(gr[np.ix_(jj, ii)]))
            out += np.einsum("uij,vji->uv", x, y)*dz/np.pi
        return out

    eps = 1.e-5
    numerical = (value(eps)-value(-eps))/(2*eps)
    np.testing.assert_allclose(analytic, numerical, atol=2.e-11, rtol=2.e-7)
