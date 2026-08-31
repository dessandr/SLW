from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest

from slw.core.wannier_io import write_wannier_hr
from slw.exchange.kernels.j_tensor_epr import (
    _compute_tensor_direct,
    _compute_tensor_tb2j,
    _normalise_site_selectors,
    _precompute_spinor_kdata,
    _selectors_in_dynamic_basis,
)
from slw.exchange.kernels.j_wannier import (
    _infer_spinor_magnetic_selectors,
    _pbc_nearest_atom_assignments,
    _spinor_canonical_index_map,
    _write_tensor_h5_simple,
)
from slw.exchange.kernels.j_wannier import run as run_wannier_tensor


def _write_win(path: Path) -> None:
    path.write_text(
        """begin unit_cell_cart
ang
10 0 0
0 10 0
0 0 10
end unit_cell_cart
begin atoms_frac
Mn1 0.05 0 0
Mn2 0.55 0 0
Te1 0.30 0 0
end atoms_frac
""",
        encoding="utf-8",
    )


def _write_spinor_centres(path: Path, groupby: str) -> None:
    up = np.asarray(
        [[0.50, 0, 0], [3.00, 0, 0], [5.50, 0, 0], [3.10, 0, 0], [0.60, 0, 0]]
    )
    down = up + np.asarray([0.01, 0.0, 0.0])
    if groupby == "spin":
        centres = np.concatenate([up, down], axis=0)
    else:
        centres = np.stack([up, down], axis=1).reshape(-1, 3)
    rows = [f"X {x:.8f} {y:.8f} {z:.8f}" for x, y, z in centres]
    rows.extend(["Mn1 0.5 0 0", "Mn2 5.5 0 0", "Te1 3.0 0 0"])
    path.write_text(
        f"{len(rows)}\nWannier centres and atoms\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("groupby", ["spin", "orbital"])
def test_auto_subspace_uses_x_rows_groupby_and_mag_atom_order(
    tmp_path: Path, groupby: str
) -> None:
    win = tmp_path / "model.win"
    centres = tmp_path / "model_centres.xyz"
    _write_win(win)
    _write_spinor_centres(centres, groupby)

    selectors, metadata = _infer_spinor_magnetic_selectors(
        win_path=win,
        centres_path=centres,
        spinor_dim=10,
        groupby=groupby,
        mag_atoms=[1, 0],
        centre_tolerance_ang=0.2,
    )

    assert list(selectors) == [0, 1]
    np.testing.assert_array_equal(selectors[0], [2])
    np.testing.assert_array_equal(selectors[1], [0, 4])
    np.testing.assert_array_equal(metadata["selected_mag_atoms"], [1, 0])
    np.testing.assert_array_equal(metadata["orbital_atom_index"], [0, 2, 1, 2, 0])
    assert metadata["partner_file_indices"].shape == (5, 2)
    assert np.max(metadata["partner_distance_ang"]) == pytest.approx(0.11)

    # A structure-only .win must not require a projections block merely for
    # spinor basis canonicalisation or centre matching.
    canonical, mode, labels, file_labels = _spinor_canonical_index_map(
        10, groupby=groupby, win=win
    )
    assert canonical.shape == (10,)
    assert mode == groupby
    assert labels == []
    assert file_labels == []


def test_pbc_nearest_atom_uses_cartesian_metric_for_skew_cell() -> None:
    lattice = np.asarray([[1.0, 0.0, 0.0], [0.9, 0.2, 0.0], [0.0, 0.0, 1.0]])
    centre = np.asarray([[0.49, 0.49, 0.0]]) @ lattice
    atom, distance = _pbc_nearest_atom_assignments(
        centre, lattice, np.asarray([[0.0, 0.0, 0.0]])
    )
    np.testing.assert_array_equal(atom, [0])
    assert distance[0] == pytest.approx(0.10660675, rel=1.0e-7)


def test_auto_subspace_fails_when_spin_partners_map_to_different_atoms(
    tmp_path: Path,
) -> None:
    win = tmp_path / "model.win"
    _write_win(win)
    centres = tmp_path / "bad.xyz"
    centres.write_text(
        "5\ncentres\nX 0.5 0 0\nX 5.5 0 0\nMn1 0.5 0 0\nMn2 5.5 0 0\nTe1 3 0 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Spin partners were assigned to different atoms"):
        _infer_spinor_magnetic_selectors(
            win_path=win,
            centres_path=centres,
            spinor_dim=2,
            groupby="spin",
            mag_atoms=[0],
        )


def test_auto_subspace_tolerance_and_missing_orbitals_fail_closed(
    tmp_path: Path,
) -> None:
    win = tmp_path / "model.win"
    _write_win(win)
    centres = tmp_path / "one.xyz"
    centres.write_text(
        "5\ncentres\nX 0.7 0 0\nX 0.7 0 0\nMn1 0.5 0 0\nMn2 5.5 0 0\nTe1 3 0 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exceed centre_tolerance_ang"):
        _infer_spinor_magnetic_selectors(
            win_path=win,
            centres_path=centres,
            spinor_dim=2,
            groupby="spin",
            mag_atoms=[0],
            centre_tolerance_ang=0.1,
        )
    with pytest.raises(ValueError, match="has no Wannier orbitals"):
        _infer_spinor_magnetic_selectors(
            win_path=win,
            centres_path=centres,
            spinor_dim=2,
            groupby="spin",
            mag_atoms=[1],
        )


def _small_spinor_hamiltonian() -> np.ndarray:
    up = np.diag([-1.2, -0.9, -0.6, -0.3]).astype(np.complex128)
    down = np.diag([0.8, 1.1, 1.4, 1.7]).astype(np.complex128)
    up[0, 1] = up[1, 0] = 0.08
    up[2, 3] = up[3, 2] = -0.04
    down[0, 1] = down[1, 0] = 0.05
    down[2, 3] = down[3, 2] = -0.03
    out = np.zeros((1, 8, 8), dtype=np.complex128)
    out[0, :4, :4] = up
    out[0, 4:, 4:] = down
    return out


def _kernel_inputs():
    pair_meta = [
        {"li": 0, "lj": 1, "gi": 0, "gj": 1, "R": (0, 0, 0), "dist": 1.0, "shell": 1}
    ]
    kpts = np.zeros((1, 3), dtype=np.float64)
    energy_mesh = [(-2.0 + 0.15j, 0.2 + 0.0j)]
    return pair_meta, kpts, energy_mesh


def test_tensor_kernels_accept_index_arrays_and_keep_full_green_space() -> None:
    h_spin = _small_spinor_hamiltonian()
    selectors = {0: np.asarray([0, 2]), 1: np.asarray([1, 3])}
    pair_meta, kpts, energy_mesh = _kernel_inputs()

    kdata = _precompute_spinor_kdata(h_spin, selectors, efermi=0.0)
    assert kdata["evals"].shape == (1, 8)
    assert kdata["evecs"].shape == (1, 8, 8)
    assert kdata["coeffs"][0].shape == (1, 4, 8)

    direct, _trace, _info = _compute_tensor_direct(
        h_spin,
        selectors,
        pair_meta,
        kpts,
        energy_mesh,
        0.0,
        ("x", "y", "z"),
    )
    tb2j, _acc, _extra, _info = _compute_tensor_tb2j(
        h_spin,
        selectors,
        pair_meta,
        kpts,
        energy_mesh,
        0.0,
        ("x", "y", "z"),
    )
    assert direct.shape == (1, 3, 3)
    assert tb2j.shape == (1, 3, 3)
    assert np.all(np.isfinite(direct))
    assert np.all(np.isfinite(tb2j))


@pytest.mark.parametrize("kernel", [_compute_tensor_direct, _compute_tensor_tb2j])
def test_contiguous_index_arrays_match_legacy_slices(kernel) -> None:
    h_spin = _small_spinor_hamiltonian()
    pair_meta, kpts, energy_mesh = _kernel_inputs()
    legacy = {0: slice(0, 2), 1: slice(2, 4)}
    arrays = {0: np.asarray([0, 1]), 1: np.asarray([2, 3])}

    old = kernel(h_spin, legacy, pair_meta, kpts, energy_mesh, 0.0, ("x", "y", "z"))[0]
    new = kernel(h_spin, arrays, pair_meta, kpts, energy_mesh, 0.0, ("x", "y", "z"))[0]
    np.testing.assert_allclose(new, old, atol=1.0e-12, rtol=1.0e-12)


def test_selector_validation_rejects_overlap_and_dynamic_missing_orbitals() -> None:
    with pytest.raises(ValueError, match="overlaps existing site selectors"):
        _normalise_site_selectors({0: [0, 2], 1: [1, 2]}, 4)
    with pytest.raises(ValueError, match="absent from the dynamic d subspace"):
        _selectors_in_dynamic_basis({0: [0, 2]}, 4, np.asarray([0, 1]))


def test_auto_mapping_metadata_is_written_to_h5(tmp_path: Path) -> None:
    args = Namespace(
        kmesh=[1, 1, 1],
        efermi=0.0,
        apply_degeneracy=True,
        spin_direction=[0.0, 0.0, 1.0],
        _mpi_size=1,
        kernel="tb2j",
        integrator="contour",
        hr_unit="ev",
        up_hr=None,
        dn_hr=None,
        spinor_hr="model_hr.dat",
        groupby="spin",
        centres="model_centres.xyz",
        mag_subspace="",
        win="model.win",
        ref_epr_up=None,
        ref_epr_dn=None,
        _soc_entries=[],
        spin_magnitude=1.0,
        all_bonds=True,
        _magnetic_subspace_meta={
            "source": "centres_xyz_pbc_nearest_atom",
            "structure_source": "explicit",
            "win_path": "model.win",
            "centres_path": "model_centres.xyz",
            "groupby": "spin",
            "centre_tolerance_ang": 0.2,
            "selected_mag_atoms": np.asarray([1, 0]),
            "selector_offsets": np.asarray([0, 1, 3]),
            "selector_indices": np.asarray([2, 0, 4]),
            "site_max_distance_ang": np.asarray([0.01, 0.1]),
            "orbital_atom_index": np.asarray([0, 2, 1, 2, 0]),
            "partner_file_indices": np.asarray([[0, 5], [1, 6]]),
            "partner_distance_ang": np.asarray([[0.0, 0.01], [0.0, 0.01]]),
        },
    )
    output = tmp_path / "tensor.h5"
    pair_meta = [
        {"gi": 1, "gj": 0, "li": 0, "lj": 1, "R": (0, 0, 0), "dist": 1.0, "shell": 1}
    ]
    _write_tensor_h5_simple(
        output,
        args,
        ["Mn1", "Mn2", "Te1"],
        pair_meta,
        np.zeros((1, 3, 3)),
        np.zeros((1, 3, 3), dtype=np.complex128),
        0.0,
        1,
        1,
        ("x", "y", "z"),
    )
    with h5py.File(output, "r") as h5:
        assert h5["basic_data/magnetic_subspace_source"].asstr()[()] == "centres_xyz_pbc_nearest_atom"
        np.testing.assert_array_equal(h5["magnetic_subspace/selected_mag_atoms"][:], [1, 0])
        np.testing.assert_array_equal(h5["magnetic_subspace/selector_indices"][:], [2, 0, 4])
        assert h5["basic_data/centre_tolerance_ang"][()] == pytest.approx(0.2)


def test_spinor_auto_subspace_run_writes_finite_tensor(tmp_path: Path) -> None:
    win = tmp_path / "two_site.win"
    win.write_text(
        """begin unit_cell_cart
ang
3 0 0
0 3 0
0 0 3
end unit_cell_cart
begin atoms_frac
Mn1 0 0 0
Mn2 0.5 0 0
end atoms_frac
""",
        encoding="utf-8",
    )
    centres = tmp_path / "two_site_centres.xyz"
    centres.write_text(
        "6\ncentres\n"
        "X 0 0 0\nX 1.5 0 0\nX 0 0 0\nX 1.5 0 0\n"
        "Mn1 0 0 0\nMn2 1.5 0 0\n",
        encoding="utf-8",
    )
    block = np.zeros((4, 4), dtype=np.complex128)
    block[:2, :2] = [[-1.0, 0.12], [0.12, -0.8]]
    block[2:, 2:] = [[0.9, 0.08], [0.08, 1.1]]
    hr_path = tmp_path / "two_site_hr.dat"
    write_wannier_hr(hr_path, 4, [1], {(0, 0, 0): block})

    output = tmp_path / "result.h5"
    args = Namespace(
        up_hr=None,
        dn_hr=None,
        spinor_hr=str(hr_path),
        groupby="spin",
        centres=str(centres),
        centre_tolerance_ang=0.1,
        efermi=0.0,
        hr_unit="ev",
        ref_epr_up=None,
        ref_epr_dn=None,
        ref_hr_unit="ry",
        kmesh=[1, 1, 1],
        mag_atoms=[1, 0],
        mag_atoms_base=0,
        slices="",
        apply_degeneracy=True,
        kernel="tb2j",
        axes="xyz",
        spin_direction=[0.0, 0.0, 1.0],
        soc="",
        soc_manifolds=(),
        win=str(win),
        mag_subspace="",
        soc_element="",
        lambda_te=0.0,
        soc_p_groups="",
        soc_groups_base=0,
        p_order="pz,px,py",
        d_order="dz2,dxz,dyz,dx2-y2,dxy",
        intersite_soc=False,
        d_subspace="",
        soc_active="",
        lambda_soc=None,
        e0=0.0,
        eta=0.0,
        hermitianize_soc=True,
        n_shells=1,
        d_max=2.0,
        all_bonds=True,
        nn_only=False,
        orbit_grouping="distance",
        integrator="contour",
        emin=-3.0,
        empoints=8,
        cfr_beta=1000.0,
        nproc=1,
        collinear_override=False,
        spin_magnitude=1.0,
        dynamic_soc=False,
        out_dir=str(tmp_path),
        out_name="result.txt",
        out_h5=str(output),
    )
    run_wannier_tensor(args)

    with h5py.File(output, "r") as h5:
        tensor = h5["J_tensor_r"][:]
        assert tensor.size > 0
        assert np.all(np.isfinite(tensor))
        assert h5["basic_data/magnetic_subspace_source"].asstr()[()] == "centres_xyz_pbc_nearest_atom"
        np.testing.assert_array_equal(
            h5["magnetic_subspace/selected_mag_atoms"][:], [1, 0]
        )
        np.testing.assert_array_equal(
            h5["magnetic_subspace/selector_indices"][:], [1, 0]
        )
