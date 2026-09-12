"""Physical sewing, endpoint gauge, joint projection, and SOC-retention oracles."""

from types import SimpleNamespace

import numpy as np
import pytest

from slw.wtorque.model.afm_symmetry import AFMInversionSymmetry, native_afm_inversion_symmetry


def dagger(x):
    return x.conj().swapaxes(-1, -2)


def symmetry(norb_per_site=1):
    orbital_swap = np.kron([[0, 1], [1, 0]], np.eye(norb_per_site))
    return AFMInversionSymmetry(np.kron(orbital_swap, [[0, 1], [-1, 0]]), np.array([1, 0]))


def test_joint_projection_is_orthogonal_idempotent_and_preserves_both_constraints():
    sym = symmetry(); rng = np.random.default_rng(58)
    a, b = [rng.normal(size=(3, 6, 4, 4))+1j*rng.normal(size=(3, 6, 4, 4)) for _ in range(2)]
    p, pr, report = sym.project_pair(a, b)
    np.testing.assert_allclose(sym.transform_vertex(p), p, atol=1.e-15)
    np.testing.assert_allclose(pr, dagger(p), atol=0.)
    np.testing.assert_allclose(sym.project_pair(p, pr)[0], p, atol=1.e-15)
    np.testing.assert_allclose(sym.transform_vertex(sym.transform_vertex(a)), a, atol=1.e-15)
    np.testing.assert_allclose(sym.project_pair(b, a)[0], pr, atol=1.e-15)
    # Orthogonal residual against every allowed direction, not just its own image.
    trial = sym.project_pair(1j*b, a-b)[0]
    inner = np.vdot(a-p, trial) + np.vdot(b-pr, dagger(trial))
    assert abs(inner.real) < 1.e-12
    assert report["afm_pair_commutator"]["relative"] < 1.e-15
    assert np.linalg.norm(p-dagger(p)) > 1.  # finite-q forward g is not Hermitian


def test_endpoint_frames_and_polar_covector_phases_match_full_sewing_oracle():
    sym = symmetry(); rng = np.random.default_rng(19)
    fs, ff = [np.linalg.qr(rng.normal(size=(2, 4, 4))+1j*rng.normal(size=(2, 4, 4)))[0] for _ in range(2)]
    k = np.array([[.83, -.2, 1.7], [1.9, .4, -.2]]); q = np.array([.41, .19, -.73])
    tau = np.array([.3, -.2, .8]); pos = np.repeat([[0, 0, 0], tau], 3, axis=0)
    a, b = [rng.normal(size=(2, 6, 4, 4))+1j*rng.normal(size=(2, 6, 4, 4)) for _ in range(2)]
    s = sym.electronic_sewing
    ua0 = np.exp(-2j*np.pi*(k@tau))[:, None, None]*s
    ua1 = np.exp(-2j*np.pi*((k+q)@tau))[:, None, None]*s
    u0, u1 = fs @ ua0 @ fs.swapaxes(-1, -2), ff @ ua1 @ ff.swapaxes(-1, -2)
    # Full atomic displacement representation, conjugated because g is a covector.
    uu = -np.exp(-2j*np.pi*(q@tau))*np.kron([[0, 1], [1, 0]], np.eye(3))
    phase = np.exp(2j*np.pi*(pos@q))[None, :, None, None]
    ga = dagger(ff)[:, None] @ a @ fs[:, None]*phase
    expected_a = np.einsum('cd,kdij->kcij', uu.conj(), ua1[:, None] @ ga.conj() @ dagger(ua0)[:, None])
    np.testing.assert_allclose(expected_a, sym.transform_vertex(ga), atol=2.e-14)
    # Cell-coordinate displacement sewing and Wannier endpoints: independent full formula.
    uu_cell = phase[0, :, 0, 0, None]*uu*phase[0, None, :, 0, 0]
    aw = np.einsum('cd,kdij->kcij', uu_cell.conj(), u1[:, None] @ a.conj() @ dagger(u0)[:, None])
    np.testing.assert_allclose(dagger(ff)[:, None] @ aw @ fs[:, None]*phase, expected_a, atol=3.e-14)
    pa, pb, report = sym.project_wannier_pair(a, b, source_frame=fs, final_frame=ff, q_red=q, perturbation_positions=pos)
    direct = sym.project_pair(ga, dagger(fs)[:, None] @ b @ ff[:, None]*phase.conj())[0]
    np.testing.assert_allclose(dagger(ff)[:, None] @ pa @ fs[:, None]*phase, direct, atol=3.e-14)
    np.testing.assert_allclose(pa, dagger(pb), atol=3.e-14)
    np.testing.assert_allclose(sym.project_wannier_pair(pa, pb, source_frame=fs, final_frame=ff, q_red=q, perturbation_positions=pos)[0], pa, atol=3.e-14)
    assert report['projected_Wannier_reciprocity']['relative'] < 1.e-14


def test_allowed_orbital_spin_soc_vertex_survives_exactly():
    sym = symmetry(3)
    # Real p orbitals: L matrices are imaginary Hermitian. L.S is TR even,
    # inversion even and has spin-flip matrix elements. Its displacement
    # derivative changes sign on the inversion-partner perturbation.
    lx = np.array([[0,0,0],[0,0,-1j],[0,1j,0]])
    ly = np.array([[0,0,1j],[0,0,0],[-1j,0,0]])
    lz = np.array([[0,-1j,0],[1j,0,0],[0,0,0]])
    sigma = [np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.diag([1.,-1.])]
    ls = sum(np.kron(l, s) for l, s in zip((lx,ly,lz), sigma))
    g = np.zeros((1,6,12,12), complex); g[0,0,:6,:6] = ls; g[0,3,6:,6:] = -ls
    p, pr, _ = sym.project_pair(g, dagger(g))
    np.testing.assert_array_equal(p, g)
    assert np.linalg.norm(p.imag) > 0
    assert np.linalg.norm(p[..., ::2, 1::2]) > 0


def inputs(tmp_path):
    xml = tmp_path/'qe.xml'; nnkp = tmp_path/'trial.nnkp'
    xml.write_text('''<root><input><atomic_species>
    <species name="M1"><pseudo_file>M.upf</pseudo_file><mass>10</mass><starting_magnetization>0.5</starting_magnetization></species>
    <species name="M2"><pseudo_file>M.upf</pseudo_file><mass>10</mass><starting_magnetization>-0.5</starting_magnetization></species>
    </atomic_species></input><output><atomic_structure><cell><a1>1 0 0</a1><a2>0 1 0</a2><a3>0 0 1</a3></cell>
    <atomic_positions><atom name="M1">0 0 0</atom><atom name="M2">0.3 0.2 0.1</atom></atomic_positions>
    </atomic_structure></output></root>''')
    records = []
    for pos in ('0 0 0', '0.3 0.2 0.1'):
        for spin in (1,-1):
            records.append(f'{pos} 1 1 1\n0 0 1 1 0 0 1\n{spin} 0 0 1')
    nnkp.write_text('begin spinor_projections\n4\n'+'\n'.join(records)+'\nend spinor_projections\n')
    centers = np.array([[0,0,0],[.3,.2,.1]])
    frame = SimpleNamespace(orbital_centers=centers, orbital_masks=np.eye(2,dtype=bool),
                            magnetic_site_positions=centers,local_frames=np.array([np.eye(3),np.diag([1.,-1.,-1.])]))
    meta = SimpleNamespace(tau=centers,nwan=4)
    config = dict(qe_xml=str(xml),nnkp=str(nnkp),atomic_spin_order='interleaved',
                  afm_inversion_translation=[.3,.2,.1],spin_lengths=[1,1])
    return frame,meta,config


def test_factory_parses_species_real_trial_parity_and_local_moments(tmp_path):
    frame, meta, cfg = inputs(tmp_path)
    actual = native_afm_inversion_symmetry(frame, meta, cfg)
    np.testing.assert_allclose(actual.electronic_sewing, -symmetry().electronic_sewing)
    assert actual.diagnostics['orbital_parities'] == [-1.,-1.]


@pytest.mark.parametrize('failure', ['geometry','species','orbital','moment','local_axis','spin_length','hybrid'])
def test_factory_rejects_incompatible_physical_inputs(tmp_path, failure):
    frame, meta, cfg = inputs(tmp_path)
    xml, nnkp = tmp_path/'qe.xml', tmp_path/'trial.nnkp'
    if failure == 'geometry': cfg['afm_inversion_translation'] = [.1,0,0]
    if failure == 'species': xml.write_text(xml.read_text().replace('M.upf','X.upf',1))
    if failure == 'orbital': nnkp.write_text(nnkp.read_text().replace('0.3 0.2 0.1 1 1 1','0.3 0.2 0.1 2 1 1'))
    if failure == 'moment': xml.write_text(xml.read_text().replace('-0.5','0.5'))
    if failure == 'local_axis': frame.local_frames[1] = np.eye(3)
    if failure == 'spin_length': cfg['spin_lengths'] = [1,2]
    if failure == 'hybrid': nnkp.write_text(nnkp.read_text().replace(' 1 1 1\n',' -1 1 1\n'))
    with pytest.raises(ValueError): native_afm_inversion_symmetry(frame, meta, cfg)


@pytest.mark.parametrize('sewing,permutation', [(np.eye(4),[1,0]), (np.eye(4),[0,0]), (np.zeros((4,3)),[1,0])])
def test_invalid_representation_rejected(sewing,permutation):
    with pytest.raises(ValueError): AFMInversionSymmetry(sewing,np.array(permutation))
