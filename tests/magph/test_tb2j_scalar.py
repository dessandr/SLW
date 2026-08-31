from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from slw.magph.lswt import solve_isotropic_lswt_energies
from slw.magph.screening import screen_magnetic_configuration
from slw.magph.tb2j import load_projected_scalar_tb2j_h5


def write_projected_tb2j_afm(path: Path, *, source_weight: float | None = None) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["scalar_spin_group_projected"] = 1
        basic = handle.create_group("basic_data")
        basic.create_dataset("kernel", data=np.bytes_("tb2j"))
        basic.create_dataset("unit", data=np.bytes_("meV"))
        basic.create_dataset("atom_labels", data=np.asarray((b"A", b"B")))
        basic.create_dataset("lattice_ang", data=np.eye(3, dtype=np.float64))
        basic.create_dataset(
            "tau_frac", data=np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)))
        )
        basic.create_dataset(
            "tau_cart_ang",
            data=np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))),
        )
        if source_weight is not None:
            basic.create_dataset("directed_bond_weight", data=source_weight)
        bonds = handle.create_group("bonds")
        bonds.create_dataset("mag_i_local", data=np.asarray((0, 1, 0, 1)))
        bonds.create_dataset("mag_j_local", data=np.asarray((1, 0, 1, 0)))
        bonds.create_dataset("mag_i_atom", data=np.asarray((0, 1, 0, 1)))
        bonds.create_dataset("mag_j_atom", data=np.asarray((1, 0, 1, 0)))
        bonds.create_dataset(
            "R",
            data=np.asarray(
                ((0, 0, 0), (0, 0, 0), (-1, 0, 0), (1, 0, 0))
            ),
        )
        bonds.create_dataset("mirror_index", data=np.asarray((1, 0, 3, 2)))
        bonds.create_dataset("distance_ang", data=np.full(4, 0.5))
        handle.create_dataset("J_iso_r", data=np.full(4, -2.0))


def test_explicit_tb2j_weight_conversion_matches_double_directed_lswt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "projected_j.h5"
    write_projected_tb2j_afm(path)
    exchange, report = load_projected_scalar_tb2j_h5(
        path,
        source_directed_bond_weight=1.0,
        spin_normalization="unit_vector",
        source_spin_magnitude=1.0,
    )
    np.testing.assert_array_equal(exchange.isotropic_mev, np.full(4, -4.0))
    assert report.exchange_scale == 2.0
    assert report.source_weight_origin == "explicit_input"
    configuration = screen_magnetic_configuration(
        exchange,
        order="collinear_afm",
        spin_magnitudes=1.0,
    )
    point = np.asarray(((0.25, 0.0, 0.0),))
    result = solve_isotropic_lswt_energies(exchange, configuration, point)
    expected = 2.0 * 4.0 * np.sin(np.pi * 0.25)
    np.testing.assert_allclose(result.energy_mev[0], (expected, expected))


def test_tb2j_adapter_cross_checks_declared_source_weight(tmp_path: Path) -> None:
    path = tmp_path / "projected_j.h5"
    write_projected_tb2j_afm(path, source_weight=1.0)
    exchange, report = load_projected_scalar_tb2j_h5(
        path,
        source_directed_bond_weight=1.0,
        spin_normalization="unit_vector",
        source_spin_magnitude=2.0,
    )
    assert exchange.n_bonds == 4
    assert report.source_weight_origin == "file_and_explicit_input"

