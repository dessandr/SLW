from __future__ import annotations

from pathlib import Path

import numpy as np

from slw.magph.phonon import (
    CELL_GAUGE_FOURIER_PHASE_CONVENTION,
    CELL_GAUGE_VECTOR_CONVENTION,
    SUPPORTED_PHONON_CACHE_SCHEMA_VERSION,
    PhononCache,
)
from slw.magph.tb2j import load_projected_scalar_tb2j_h5
from slw.wtorque.cli import _parser
from slw.wtorque.projection.polaron import diagnose_bosonic_bdg
from slw.wtorque.q0_polaron import project_q0_magnon_polaron
from tests.magph.test_tb2j_scalar import write_projected_tb2j_afm


def _phonons() -> PhononCache:
    masses = np.asarray((1000.0, 2000.0))
    vectors = np.zeros((1, 6, 2, 3), dtype=np.complex128)
    for mode in range(6):
        atom, cart = divmod(mode, 3)
        vectors[0, mode, atom, cart] = 1.0 / np.sqrt(masses[atom])
    return PhononCache(
        source="synthetic",
        schema_version=SUPPORTED_PHONON_CACHE_SCHEMA_VERSION,
        q_points_frac=np.zeros((1, 3)),
        frequencies_mev=np.asarray(((0.0, 0.0, 0.0, 10.0, 12.0, 15.0),)),
        eigenvectors_mass_normalized=vectors,
        masses=masses,
        mass_unit="electron_mass",
        q_mesh_shape=(1, 1, 1),
        vector_convention=CELL_GAUGE_VECTOR_CONVENTION,
        fourier_phase_convention=CELL_GAUGE_FOURIER_PHASE_CONVENTION,
    )


def test_q0_polaron_cli_keeps_material_and_conventions_explicit() -> None:
    args = _parser().parse_args(
        [
            "test-collinear-epr-q0-polaron",
            "--up-epr",
            "up.h5",
            "--down-epr",
            "down.h5",
            "--magnetic-orbitals",
            "1-5;6-10",
            "--magnetic-atoms",
            "1,2",
            "--local-direction",
            "0",
            "1",
            "0",
            "--local-direction",
            "0",
            "-1",
            "0",
            "--spin-quantization-direction",
            "0",
            "1",
            "0",
            "--spin-coordinate",
            "rotation_angle",
            "--fermi-energy-ev",
            "11.4",
            "--energy-min-ev",
            "3.5",
            "--energy-max-ev",
            "11.4",
            "--energy-points",
            "64",
            "--eta-ev",
            "0.05",
            "--temperature-k",
            "0",
            "--epr-energy-unit",
            "ry",
            "--epr-displacement-unit",
            "bohr",
            "--exchange-h5",
            "J.h5",
            "--exchange-source-directed-bond-weight",
            "1",
            "--exchange-spin-normalization",
            "unit_vector",
            "--spin-length",
            "2.5",
            "--anisotropy-mev",
            "0.0005",
            "--anisotropy-spin-normalization",
            "unit_vector",
        ]
    )
    assert args.command == "test-collinear-epr-q0-polaron"
    assert args.exchange_source_directed_bond_weight == 1.0
    assert args.spin_length == [2.5]
    assert args.anisotropy_mev == [0.0005]


def test_rotation_angle_and_transverse_direction_give_same_mode_vertex(
    tmp_path: Path,
) -> None:
    path = tmp_path / "projected_j.h5"
    write_projected_tb2j_afm(path)
    exchange, _report = load_projected_scalar_tb2j_h5(
        path,
        source_directed_bond_weight=1.0,
        spin_normalization="unit_vector",
        source_spin_magnitude=1.0,
    )
    kernel_pi = np.arange(24, dtype=np.float64).reshape(2, 2, 2, 3) / 1000.0
    kernel_theta = np.stack((-kernel_pi[:, 1], kernel_pi[:, 0]), axis=1)
    common = {
        "local_magnetization_directions": ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        "spin_quantization_direction": (0.0, 1.0, 0.0),
        "magnetic_atom_indices": (0, 1),
        "exchange": exchange,
        "spin_lengths": (1.0, 1.0),
        "anisotropy_mev": (0.1, 0.1),
        "anisotropy_spin_normalization": "unit_vector",
        "phonons": _phonons(),
        "phonon_min_energy_mev": 1.0e-6,
    }
    transverse = project_q0_magnon_polaron(
        kernel_pi,
        spin_coordinate="transverse_direction",
        **common,
    )
    rotation = project_q0_magnon_polaron(
        kernel_theta,
        spin_coordinate="rotation_angle",
        **common,
    )
    np.testing.assert_allclose(
        transverse.normal_coupling_mev,
        rotation.normal_coupling_mev,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        transverse.anomalous_coupling_mev,
        rotation.anomalous_coupling_mev,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        transverse.phonon_mode_indices_one_based,
        (4, 5, 6),
    )
    assert transverse.normal_coupling_mev.shape == (2, 3)
    assert transverse.rwa_hamiltonian_mev.shape == (5, 5)
    assert transverse.rwa_hermiticity_residual_mev == 0.0
    assert transverse.magnon_paraunitarity_residual < 1.0e-10
    assert transverse.full_bdg_dynamic_eigenvalues_mev.shape == (10,)


def test_full_bosonic_diagnostic_retains_anomalous_instability() -> None:
    uncoupled = diagnose_bosonic_bdg(
        (1.0,),
        (2.0,),
        np.zeros((1, 1), dtype=np.complex128),
        np.zeros((1, 1), dtype=np.complex128),
    )
    assert uncoupled.stable
    np.testing.assert_allclose(
        uncoupled.dynamic_eigenvalues_mev.real,
        (-2.0, -1.0, 1.0, 2.0),
    )
    unstable = diagnose_bosonic_bdg(
        (0.2,),
        (1.0,),
        np.zeros((1, 1), dtype=np.complex128),
        np.asarray(((1.0 + 0.0j,),)),
    )
    assert not unstable.stable
    assert unstable.minimum_hessian_eigenvalue_mev < 0.0
