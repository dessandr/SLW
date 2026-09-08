"""Native file conventions and an analytic complete mode-projection artifact."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.constants import hbar, atomic_mass, electron_volt, physical_constants

from slw.wtorque.io.native_coarse import periodic_grid_indices, read_native_coarse_vertex
from slw.wtorque.native_workflow import _write_result


def test_native_fortran_axes_mesh_order_and_units(tmp_path):
    path = tmp_path / "elph.h5"
    reference = np.array([[[1+2j, 3+4j], [5+6j, 7+8j]], [[9+10j, 11+12j], [13+14j, 15+16j]]])
    with h5py.File(path, "w") as out:
        out["wannier_gauge"] = 1
        for cart in range(3):
            data = (cart+1)*reference.swapaxes(-1, -2)[None]
            out[f"elph_1_{cart+1}_r"] = data.real
            out[f"elph_1_{cart+1}_i"] = data.imag
    indices = periodic_grid_indices([[.5,0,0], [0,0,0]], [[0,0,0], [-.5,0,0]])
    actual = read_native_coarse_vertex(path, q_index=0, k_indices=indices,
        nat=1,num_wann=2,nq=1,nk=2,energy_unit="ev",displacement_unit="bohr")
    bohr_ang = physical_constants["Bohr radius"][0]/1.e-10
    np.testing.assert_allclose(actual[:, 1], reference[::-1]*2/bohr_ang)
    with pytest.raises(ValueError, match="dimensions"):
        read_native_coarse_vertex(path,q_index=0,k_indices=indices,nat=1,num_wann=4,nq=1,nk=2,energy_unit="ev",displacement_unit="bohr")
    with pytest.raises(ValueError, match="ambiguous"):
        periodic_grid_indices([[0,0,0], [1,0,0]], [[0,0,0]])


@pytest.mark.parametrize("direct_fraction", [0., -.9])
def test_complex_bubble_to_mode_artifact_and_no_overwrite(tmp_path, direct_fraction):
    energies = np.array([[.1,.2,.3], [.1,.2,.3]])
    phonons = SimpleNamespace(energies_eV=energies, masses_amu=np.array([10.]),
        eigenvectors=np.tile(np.eye(3,dtype=complex).reshape(1,1,3,3),(2,1,1,1)))
    magnons = SimpleNamespace(energies_eV=np.full((2,1),.4),
        spin_lengths=np.array([2.]), transform=np.tile(np.eye(2,dtype=complex),(2,1,1)),
        local_frames=np.eye(3)[None],magnetic_site_positions=np.zeros((1,3)))
    k = np.zeros((2,3),complex)
    k[0,0] = 2+3j  # Only the first spin tangent and Cartesian x couple.
    loops = {0:{"loop":-1j*np.pi*k,"g_reciprocity_relative":0.},
             1:{"loop":-1j*np.pi*k.conj(),"g_reciprocity_relative":0.}}
    for item in loops.values():
        item['direct_loop'] = direct_fraction * item['loop']
    config = {"qpoints":[[1/3,0,0],[-1/3,0,0]],"output":str(tmp_path/'result.h5'),"include_direct_vertex":direct_fraction != 0}
    inputs = dict(phonons=phonons,magnons=magnons,meta=SimpleNamespace(nat=1),summary={})
    report = _write_result(config,inputs,loops,1)
    # Independent dimensional expression: K * sqrt(hbar/(2M omega)) / sqrt(2S).
    bubble_expected = (2+3j)*hbar / np.sqrt(2*10*atomic_mass*.1*electron_volt) / 1.e-10 / 2
    expected = (1+direct_fraction)*bubble_expected
    with h5py.File(config['output']) as result:
        np.testing.assert_allclose(result['coupling/g_mp_normal'][0,0], [expected,0,0])
        np.testing.assert_allclose(result['coupling/g_mp_normal'][1,0], [expected.conjugate(),0,0])
        assert np.all(np.isfinite(result['bands/rwa_energies_eV']))
        np.testing.assert_allclose(result['coupling/g_mp_normal_bubble'][0,0], [bubble_expected,0,0])
        np.testing.assert_allclose(result['coupling/g_mp_normal_direct'][0,0], [direct_fraction*bubble_expected,0,0])
        assert result.attrs['direct_term_enabled'] == (direct_fraction != 0)
    assert report['q_pair_kernel_residual_eV_per_angstrom'] == 0
    assert json.loads((tmp_path/'result.json').read_text())['status'] == 'complete'
    with pytest.raises(FileExistsError):
        _write_result(config,inputs,loops,1)
