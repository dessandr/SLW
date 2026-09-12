"""Physical q-pair, oscillator-spectrum, and instability checks."""

import numpy as np
import pytest

from slw.wtorque.polaron_bands import (
    assemble_qpair_bdg, plot_polaron_bands, sample_wannier_path, solve_polaron_bands,
)


def _minus(value):
    nm, np_ = value.shape[0] // 2, value.shape[1] // 2
    return value[np.r_[np.arange(nm, 2 * nm), np.arange(nm)]][
        :, np.r_[np.arange(np_, 2 * np_), np.arange(np_)]
    ].conj()


def _solve(em, ep, value, emm=None, epm=None):
    return solve_polaron_bands(
        [em], [ep], [value], [em if emm is None else emm],
        [ep if epm is None else epm], [_minus(value)],
    )


def test_finite_q_particle_hole_identity_uses_actual_minus_q():
    value = np.array([[.002 + .001j, .004 - .0002j], [.0001 + .001j, .002j]])
    hq = assemble_qpair_bdg([.03], [.05], value, [.04], [.06], _minus(value))
    hm = assemble_qpair_bdg([.04], [.06], _minus(value), [.03], [.05], value)
    swap = [2, 3, 0, 1]
    np.testing.assert_allclose(hq, hm[swap][:, swap].conj())
    np.testing.assert_allclose(hq, hq.conj().T)
    # Finite-q pairing does not generally equal its own transpose.
    assert not np.allclose(hq[:2, 2:], hq[:2, 2:].T)
    metric = np.array([1, 1, -1, -1])
    wq = np.sort(np.linalg.eigvals(metric[:, None] * hq).real)
    wm = np.sort(np.linalg.eigvals(metric[:, None] * hm).real)
    np.testing.assert_allclose(wq, -wm[::-1], atol=1.e-14)


def test_real_coordinate_coupling_matches_two_oscillators():
    em, ep, g = .04, .06, .005
    result = _solve([em], [ep], np.full((2, 2), g))
    radical = np.sqrt((em**2 - ep**2)**2 + 16 * g**2 * em * ep)
    expected = np.sqrt(np.array([em**2 + ep**2 - radical, em**2 + ep**2 + radical]) / 2)
    np.testing.assert_allclose(result.energies_eV[0], expected, atol=1.e-14)
    assert result.valid.tolist() == [True]
    assert result.paraunitarity_residual[0] < 1.e-13
    assert result.eigen_residual_eV[0] < 1.e-13
    assert np.all((result.magnon_weights >= 0) & (result.magnon_weights <= 1))


def test_number_conserving_coupling_reduces_to_q_rwa():
    value = np.diag([.003 + .002j, .006 - .001j])
    result = _solve([.03], [.05], value, [.025], [.04])
    expected = np.linalg.eigvalsh([[.03, value[0, 0]], [value[0, 0].conj(), .05]])
    np.testing.assert_allclose(result.energies_eV[0], expected, atol=1.e-14)
    np.testing.assert_allclose(result.rwa_energies_eV, result.energies_eV, atol=1.e-14)


def test_bare_degeneracy_is_metric_orthonormal():
    result = _solve([.04, .04], [.04], np.zeros((4, 2)))
    np.testing.assert_allclose(result.energies_eV, .04)
    assert result.paraunitarity_residual[0] < 1.e-14
    np.testing.assert_allclose(np.sort(result.magnon_weights[0]), [0, 1, 1])


def test_mode_gauge_covariance():
    rng = np.random.default_rng(421)
    value = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) * .001
    left = np.exp(1j * rng.normal(size=4))
    right = np.exp(1j * rng.normal(size=4))
    transformed = left[:, None].conj() * value * right[None, :]
    original = _solve([.025, .035], [.04, .07], value, [.03, .037], [.045, .06])
    gauged = _solve([.025, .035], [.04, .07], transformed, [.03, .037], [.045, .06])
    np.testing.assert_allclose(gauged.energies_eV, original.energies_eV, atol=1.e-14)
    np.testing.assert_allclose(gauged.magnon_weights, original.magnon_weights, atol=1.e-13)


def test_qpair_mismatch_is_rejected_not_averaged():
    with pytest.raises(ValueError, match="q-pair residual"):
        assemble_qpair_bdg([.03], [.05], np.ones((2, 2)) * .001,
                           [.03], [.05], np.ones((2, 2)) * .002)


def test_instability_and_zero_modes_are_retained_without_shifts():
    unstable = _solve([.04], [.06], np.full((2, 2), .03))
    assert unstable.status == ("unstable",)
    assert np.all(np.isnan(unstable.energies_eV))
    assert unstable.minimum_hessian_eigenvalue_eV[0] < 0
    assert unstable.maximum_imaginary_eigenvalue_eV[0] > .001
    zero = _solve([0.], [.06], np.zeros((2, 2)))
    assert zero.status == ("zero_or_unresolved_mode",)
    assert zero.minimum_hessian_eigenvalue_eV[0] == 0
    assert np.any(zero.dynamic_eigenvalues_eV[0] == 0)


def test_masked_endpoints_can_contain_nan():
    result = solve_polaron_bands(
        [[np.nan], [.04]], [[np.nan], [.06]], np.zeros((2, 2, 2)),
        [[np.nan], [.04]], [[np.nan], [.06]], np.zeros((2, 2, 2)),
        valid_q_mask=[False, True],
    )
    assert result.status == ("masked_by_caller", "stable")
    assert np.all(np.isnan(result.energies_eV[0]))
    np.testing.assert_allclose(result.energies_eV[1], [.04, .06])


def test_cell_path_keeps_breaks_and_plot(tmp_path):
    source = tmp_path / "bands.win"
    source.write_text("begin kpoint_path\nG 0 0 0 X .5 0 0\nL .5 .5 .5 G 0 0 0\nend kpoint_path\n")
    path = sample_wannier_path(source, np.diag([2., 3., 4.]), points_per_segment=3)
    assert path.qpoints.shape == (8, 3)
    assert path.tick_labels == ("G", "X | L", "G")
    assert path.gamma_mask.tolist() == [True, False, False, False, False, False, False, True]
    assert path.distance_inv_ang[3] == path.distance_inv_ang[4]
    np.testing.assert_allclose(path.distance_inv_ang[3], np.pi / 2)
    result = solve_polaron_bands(np.full((8, 1), .04), np.full((8, 1), .06),
                                 np.zeros((8, 2, 2)), np.full((8, 1), .04),
                                 np.full((8, 1), .06), np.zeros((8, 2, 2)),
                                 valid_q_mask=~path.gamma_mask)
    plot_polaron_bands(path, result, tmp_path / "bands.png")
    assert (tmp_path / "bands.png").stat().st_size > 1000


def test_gamma_reduced_bdg_retains_zero_energies_but_no_acoustic_bosons():
    em=[.01];ep=[0,0,0,.06];g=.002
    coupling=np.zeros((2,8),complex);coupling[:,[3,7]]=g
    r=solve_polaron_bands([em],[ep],[coupling],[em],[ep],[_minus(coupling)],
        phonon_translation_mask=[[1,1,1,0]])
    ref=_solve(em,[.06],np.full((2,2),g))
    np.testing.assert_allclose(r.energies_eV[0,:3],0)
    np.testing.assert_allclose(r.energies_eV[0,3:],ref.energies_eV[0],atol=1e-14)
    assert r.valid[0] and r.mode_valid[0].tolist()==[False,False,False,True,True]
    assert np.isnan(r.eigenvectors[0,:,:3]).all()
    assert np.isnan(r.magnon_weights[0,:3]).all()
    t=r.eigenvectors[0,:,3:];metric=np.r_[np.ones(5),-np.ones(5)]
    np.testing.assert_allclose(t.conj().T@(metric[:,None]*t),np.eye(2),atol=1e-14)
    coupling[:,[0,4]]=1e-4
    with pytest.raises(ValueError,match='decouple'):
        solve_polaron_bands([em],[ep],[coupling],[em],[ep],[_minus(coupling)],
            phonon_translation_mask=[[1,1,1,0]])
