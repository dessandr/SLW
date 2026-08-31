from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from slw.exchange.kernels.j_epr import _write_h5
from slw.magph.screening import load_exchange_h5


def _write_scalar_lkag_fixture(path, epr_path) -> tuple[np.ndarray, np.ndarray]:
    args = SimpleNamespace(
        kmesh=(2, 3, 4),
        efermi=1.25,
        hr_unit="eV",
        integrator="contour",
        empoints=32,
        nproc=1,
        epr_up=str(epr_path),
    )
    pair_meta = [
        {
            "gi": 0,
            "gj": 1,
            "li": 0,
            "lj": 1,
            "R": (0, 0, 0),
            "dist": 2.0,
            "shell": 1,
        },
        {
            "gi": 1,
            "gj": 0,
            "li": 1,
            "lj": 0,
            "R": (0, 0, 0),
            "dist": 2.0,
            "shell": 1,
        },
    ]
    orbits = [
        [
            {
                "i": 0,
                "j": 1,
                "R": (0, 0, 0),
                "distance": 2.0,
                "shell_idx": 1,
            },
            {
                "i": 1,
                "j": 0,
                "R": (0, 0, 0),
                "distance": 2.0,
                "shell_idx": 1,
            },
        ]
    ]
    projected_source = np.array([-3.25, -3.25], dtype=np.float64)
    raw_source = np.array([-3.0, -3.5], dtype=np.float64)
    _write_h5(
        str(path),
        args,
        ["A", "B"],
        pair_meta,
        projected_source,
        orbits,
        0.25,
        24,
        32,
        {"n_chunks": 1, "mpi_size": 1},
        raw_j_mev=raw_source,
    )
    return projected_source, raw_source


def test_scalar_lkag_writer_declares_source_directed_weight_without_rescaling(
    tmp_path,
) -> None:
    epr_path = tmp_path / "source_epr.h5"
    with h5py.File(epr_path, "w"):
        pass
    output = tmp_path / "scalar_j.h5"
    projected_source, raw_source = _write_scalar_lkag_fixture(output, epr_path)

    with h5py.File(output, "r") as handle:
        basic = handle["basic_data"]
        assert basic["kernel_family"].asstr()[()] == "scalar_lkag"
        assert basic["bond_coverage"].asstr()[()] == "directed_mate_complete"
        assert basic["spin_normalization"].asstr()[()] == "unit_vector"
        assert basic["hamiltonian_sign"].asstr()[()] == "minus"
        assert float(basic["directed_bond_weight"][()]) == 1.0

        np.testing.assert_array_equal(handle["J_r/value"], projected_source)
        np.testing.assert_array_equal(handle["J_r/value_raw"], raw_source)
        np.testing.assert_array_equal(
            handle["J_r/projection_delta"], projected_source - raw_source
        )
        assert "source coefficient" in handle["J_r/value"].attrs["meaning"]

    # The numerical payload remains in the source pair-twice convention.  A
    # canonical mate-complete half-weight representation would require 2*J.
    np.testing.assert_array_equal(
        projected_source * 1.0,
        (2.0 * projected_source) * 0.5,
    )


def test_scalar_lkag_source_weight_fails_closed_in_native_loader(tmp_path) -> None:
    epr_path = tmp_path / "source_epr.h5"
    with h5py.File(epr_path, "w"):
        pass
    output = tmp_path / "scalar_j.h5"
    _write_scalar_lkag_fixture(output, epr_path)

    with pytest.raises(ValueError, match="directed_bond_weight=0.5"):
        load_exchange_h5(output)
