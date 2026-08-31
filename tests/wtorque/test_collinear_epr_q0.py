from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from slw.core.qe2pert_ws import (
    init_rvec_images,
    set_wigner_seitz_cell,
    triangular_pair_index,
)
from slw.wtorque.cli import _parser
from slw.wtorque.collinear_q0 import (
    parse_magnetic_orbital_groups,
    parse_onsite_soc_manifolds,
    run_collinear_epr_q0_test,
)
from slw.wtorque.io.native_epr_h import reconstruct_epr_hamiltonian
from slw.wtorque.parallel.mpi import MPIContext


def _write_scalar_epr(
    path: Path,
    onsite: tuple[float, ...],
    *,
    g_scale: float,
    g_offdiagonal: float = 0.0,
    centres: np.ndarray | None = None,
    atom_positions: np.ndarray | None = None,
) -> None:
    kmesh = (1, 1, 1)
    qmesh = (1, 1, 1)
    lattice = np.eye(3, dtype=np.float64)
    nwan = len(onsite)
    if centres is None:
        centres = np.zeros((nwan, 3), dtype=np.float64)
        centres[:, 2] = np.arange(nwan, dtype=np.float64) / nwan
    else:
        centres = np.asarray(centres, dtype=np.float64)
    if atom_positions is None:
        atom_positions = centres.copy()
    else:
        atom_positions = np.asarray(atom_positions, dtype=np.float64)
    nat = atom_positions.shape[0]
    electron_images = init_rvec_images(kmesh, lattice)
    phonon_images = init_rvec_images(qmesh, lattice)
    with h5py.File(path, "w") as handle:
        basic = handle.create_group("basic_data")
        basic.create_dataset("nat", data=np.int32(nat))
        basic.create_dataset("num_wann", data=np.int32(nwan))
        basic.create_dataset("kc_dim", data=np.asarray(kmesh, dtype=np.int32))
        basic.create_dataset("qc_dim", data=np.asarray(qmesh, dtype=np.int32))
        basic.create_dataset("at", data=lattice.T)
        basic.create_dataset("tau", data=atom_positions)
        basic.create_dataset("wannier_center_cryst", data=centres)
        basic.create_dataset("spinor", data=np.int32(0))

        electron = handle.create_group("electron_wannier")
        for jw in range(1, nwan + 1):
            for iw in range(1, jw + 1):
                pair = triangular_pair_index(iw, jw)
                ws = set_wigner_seitz_cell(
                    electron_images,
                    lattice,
                    centres[iw - 1],
                    centres[jw - 1],
                )
                target = onsite[iw - 1] if iw == jw else 0.0
                hopping = np.full(ws.nr, target / ws.nr, dtype=np.float64)
                electron.create_dataset(f"hopping_r{pair}", data=hopping)
                electron.create_dataset(
                    f"hopping_i{pair}", data=np.zeros(ws.nr, dtype=np.float64)
                )

        eph = handle.create_group("eph_matrix_wannier")
        for ia in range(1, nat + 1):
            for orbital in range(1, nwan + 1):
                ws_electron = set_wigner_seitz_cell(
                    electron_images,
                    lattice,
                    centres[orbital - 1],
                    centres[orbital - 1],
                )
                ws_phonon = set_wigner_seitz_cell(
                    phonon_images,
                    lattice,
                    centres[orbital - 1],
                    atom_positions[ia - 1],
                )
                value = g_scale * (ia + 0.2 * orbital)
                block = np.empty(
                    (3, ws_electron.nr, ws_phonon.nr),
                    dtype=np.float64,
                )
                for axis in range(3):
                    block[axis] = value * (axis + 1) / (
                        ws_electron.nr * ws_phonon.nr
                    )
                stored = block.transpose(2, 1, 0)
                eph.create_dataset(
                    f"ep_hop_r_{ia}_{orbital}_{orbital}", data=stored
                )
                eph.create_dataset(
                    f"ep_hop_i_{ia}_{orbital}_{orbital}",
                    data=np.zeros_like(stored),
                )
            if g_offdiagonal != 0.0:
                for jw in range(1, nwan + 1):
                    for iw in range(1, nwan + 1):
                        if iw == jw:
                            continue
                        ws_electron = set_wigner_seitz_cell(
                            electron_images,
                            lattice,
                            centres[iw - 1],
                            centres[jw - 1],
                        )
                        ws_phonon = set_wigner_seitz_cell(
                            phonon_images,
                            lattice,
                            centres[iw - 1],
                            atom_positions[ia - 1],
                        )
                        value = g_scale * g_offdiagonal * (iw + jw + ia)
                        block = np.empty(
                            (3, ws_electron.nr, ws_phonon.nr),
                            dtype=np.float64,
                        )
                        for axis in range(3):
                            block[axis] = value * (axis + 1) / (
                                ws_electron.nr * ws_phonon.nr
                            )
                        stored = block.transpose(2, 1, 0)
                        eph.create_dataset(
                            f"ep_hop_r_{ia}_{jw}_{iw}", data=stored
                        )
                        eph.create_dataset(
                            f"ep_hop_i_{ia}_{jw}_{iw}",
                            data=np.zeros_like(stored),
                        )


def test_magnetic_orbital_groups_are_one_based_and_disjoint() -> None:
    assert parse_magnetic_orbital_groups("1-3,5;6-8", 8) == (
        (0, 1, 2, 4),
        (5, 6, 7),
    )
    with pytest.raises(ValueError, match="disjoint"):
        parse_magnetic_orbital_groups("1-3;3-5", 8)
    with pytest.raises(ValueError, match="1..8"):
        parse_magnetic_orbital_groups("1-3;9", 8)

    soc = parse_onsite_soc_manifolds(("p:11-13:0.5", "d:1-5:0.04"), 16)
    assert soc[0].orbital_indices == (10, 11, 12)
    assert soc[0].lambda_eV == 0.5
    with pytest.raises(ValueError, match="requires 3 orbitals"):
        parse_onsite_soc_manifolds(("p:11-12:0.5",), 16)


def test_collinear_epr_q0_cli_requires_explicit_physical_inputs() -> None:
    args = _parser().parse_args(
        [
            "test-collinear-epr-q0",
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
            "0",
            "1",
            "--local-direction",
            "0",
            "0",
            "-1",
            "--spin-quantization-direction",
            "0",
            "0",
            "1",
            "--spin-coordinate",
            "rotation_angle",
            "--onsite-soc",
            "p:11-13:0.5",
            "--onsite-soc",
            "p:14-16:0.5",
            "--soc-p-order",
            "pz,px,py",
            "--fermi-energy-ev",
            "11.4",
            "--energy-min-ev",
            "3.5",
            "--energy-max-ev",
            "11.4",
            "--energy-points",
            "16",
            "--integration-backend",
            "analytic_zero_temperature",
            "--eta-ev",
            "0.05",
            "--temperature-k",
            "0",
            "--epr-energy-unit",
            "ry",
            "--epr-displacement-unit",
            "bohr",
        ]
    )
    assert args.command == "test-collinear-epr-q0"
    assert args.local_direction == [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    assert args.onsite_soc == ["p:11-13:0.5", "p:14-16:0.5"]
    assert args.integration_backend == "analytic_zero_temperature"


def test_scalar_epr_hamiltonian_and_full_collinear_null(tmp_path: Path) -> None:
    up_path = tmp_path / "up_epr.h5"
    down_path = tmp_path / "down_epr.h5"
    _write_scalar_epr(up_path, (-1.3, 0.8), g_scale=0.07)
    _write_scalar_epr(down_path, (1.2, -0.7), g_scale=0.11)

    up_h = reconstruct_epr_hamiltonian(
        up_path,
        energy_unit="ev",
        expected_spinor=False,
    )
    np.testing.assert_allclose(up_h.values[0], np.diag((-1.3, 0.8)))
    assert up_h.hermiticity_residual_eV == 0.0
    with pytest.raises(ValueError, match="expected 1"):
        reconstruct_epr_hamiltonian(
            up_path,
            energy_unit="ev",
            expected_spinor=True,
        )

    result = run_collinear_epr_q0_test(
        up_epr_path=up_path,
        down_epr_path=down_path,
        magnetic_orbital_specification="1;2",
        magnetic_atom_indices=(0, 1),
        local_magnetization_directions=((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        spin_quantization_direction=(0.0, 0.0, 1.0),
        spin_coordinate="rotation_angle",
        fermi_energy_eV=0.0,
        energy_min_eV=-3.0,
        energy_max_eV=0.0,
        energy_points=8,
        eta_eV=0.1,
        temperature_K=0.0,
        epr_energy_unit="ev",
        epr_displacement_unit="angstrom",
        perturbation_chunk=2,
        mpi=MPIContext(),
    )
    assert result is not None
    assert result.green_function_dimension == 4
    assert result.magnetic_orbital_indices_one_based == ((1,), (2,))
    np.testing.assert_allclose(result.onsite_exchange_mean_eV, (-2.5, 1.5))
    assert result.hamiltonian_spin_flip_max_eV == 0.0
    assert result.g_spin_flip_max_eV_per_angstrom == 0.0
    assert result.torque_spin_conserving_max_eV == 0.0
    assert result.torque_vertex_rms_eV > 0.0
    assert result.kernel_null_max_abs_eV_per_angstrom < 1.0e-12
    assert result.selection_rule_expected_zero
    assert result.onsite_soc_matrix_norm_eV == 0.0

    result_y = run_collinear_epr_q0_test(
        up_epr_path=up_path,
        down_epr_path=down_path,
        magnetic_orbital_specification="1;2",
        magnetic_atom_indices=(0, 1),
        local_magnetization_directions=((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        spin_quantization_direction=(0.0, 1.0, 0.0),
        spin_coordinate="rotation_angle",
        fermi_energy_eV=0.0,
        energy_min_eV=-3.0,
        energy_max_eV=0.0,
        energy_points=8,
        integration_backend="analytic_zero_temperature",
        eta_eV=0.1,
        temperature_K=0.0,
        epr_energy_unit="ev",
        epr_displacement_unit="angstrom",
        perturbation_chunk=2,
        mpi=MPIContext(),
    )
    assert result_y is not None
    assert result_y.kernel_null_max_abs_eV_per_angstrom < 1.0e-12
    assert result_y.g_spin_flip_max_eV_per_angstrom > 0.0
    assert (
        result_y.g_spin_quantization_commutator_max_eV_per_angstrom
        < 1.0e-13
    )


def test_p_onsite_soc_breaks_collinear_selection_rule(tmp_path: Path) -> None:
    up_path = tmp_path / "p_up_epr.h5"
    down_path = tmp_path / "p_down_epr.h5"
    centres = np.zeros((3, 3), dtype=np.float64)
    atoms = np.zeros((1, 3), dtype=np.float64)
    _write_scalar_epr(
        up_path,
        (-1.1, -0.2, 0.4),
        g_scale=0.07,
        g_offdiagonal=0.4,
        centres=centres,
        atom_positions=atoms,
    )
    _write_scalar_epr(
        down_path,
        (1.0, 0.3, 0.8),
        g_scale=0.12,
        g_offdiagonal=0.6,
        centres=centres,
        atom_positions=atoms,
    )

    result = run_collinear_epr_q0_test(
        up_epr_path=up_path,
        down_epr_path=down_path,
        magnetic_orbital_specification="1",
        magnetic_atom_indices=(0,),
        local_magnetization_directions=((0.0, 0.0, 1.0),),
        spin_quantization_direction=(0.0, 0.0, 1.0),
        spin_coordinate="rotation_angle",
        onsite_soc_specifications=("p:1-3:0.5",),
        p_orbital_order="pz,px,py",
        fermi_energy_eV=0.0,
        energy_min_eV=-3.0,
        energy_max_eV=0.0,
        energy_points=16,
        integration_backend="analytic_zero_temperature",
        eta_eV=0.1,
        temperature_K=0.0,
        epr_energy_unit="ev",
        epr_displacement_unit="angstrom",
        mpi=MPIContext(),
    )
    assert result is not None
    assert not result.selection_rule_expected_zero
    assert result.onsite_soc_manifolds == ("p:1,2,3:0.5",)
    assert result.onsite_soc_matrix_norm_eV > 0.0
    assert result.onsite_soc_spin_flip_max_eV > 0.0
    assert result.onsite_soc_hermiticity_residual_eV < 1.0e-14
    assert result.onsite_soc_time_reversal_residual_eV < 1.0e-14
    assert result.kernel_max_abs_eV_per_angstrom > 1.0e-8
