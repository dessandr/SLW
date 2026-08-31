from __future__ import annotations

import h5py
import numpy as np

from slw.core.constants import BOHR_TO_ANG, RY_TO_EV
from slw.core.wannier_io import write_wannier_hr
from slw.wtorque.io.slw_adapters import (
    load_collinear_slw_gkq,
    load_collinear_wannier_hr,
)


def test_slw_wannier_adapter_normalizes_degeneracy_and_lifts_spin(tmp_path):
    up_path = tmp_path / "up_hr.dat"
    down_path = tmp_path / "down_hr.dat"
    r_vectors = [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]
    degeneracies = [2, 1, 2]
    up = {
        (-1, 0, 0): np.array([[0.2]], dtype=np.complex128),
        (0, 0, 0): np.array([[-0.3]], dtype=np.complex128),
        (1, 0, 0): np.array([[0.2]], dtype=np.complex128),
    }
    down = {
        (-1, 0, 0): np.array([[0.4]], dtype=np.complex128),
        (0, 0, 0): np.array([[0.1]], dtype=np.complex128),
        (1, 0, 0): np.array([[0.4]], dtype=np.complex128),
    }
    write_wannier_hr(up_path, 1, degeneracies, up)
    write_wannier_hr(down_path, 1, degeneracies, down)

    data = load_collinear_wannier_hr(
        up_path,
        down_path,
        reference_direction=[0.0, 0.0, 2.0],
    )
    np.testing.assert_array_equal(data.r_vectors, np.asarray(r_vectors))
    np.testing.assert_allclose(data.h_up_r[:, 0, 0], [0.1, -0.3, 0.1])
    np.testing.assert_allclose(data.h_down_r[:, 0, 0], [0.2, 0.1, 0.2])
    np.testing.assert_allclose(data.h_r, data.h_trs_r + data.h_xc_r)
    np.testing.assert_allclose(data.reference_direction, [0.0, 0.0, 1.0])


def _write_slw_gkq(path, value: float) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "g_wannier",
            data=np.full((1, 2, 1, 1, 3), value, dtype=np.complex128),
        )
        handle.create_dataset(
            "k_fracs",
            data=np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64),
        )
        handle.create_dataset(
            "q_fracs", data=np.zeros((1, 3), dtype=np.float64)
        )
        handle.attrs["nat"] = 1


def test_slw_epc_adapter_requires_units_and_lifts_collinear_vertices(tmp_path):
    up_path = tmp_path / "g_up.h5"
    down_path = tmp_path / "g_down.h5"
    _write_slw_gkq(up_path, 1.0)
    _write_slw_gkq(down_path, 2.0)

    data = load_collinear_slw_gkq(
        up_path,
        down_path,
        energy_unit="ry",
        displacement_unit="bohr",
    )
    assert data.values.shape == (1, 2, 3, 2, 2)
    scale = RY_TO_EV / BOHR_TO_ANG
    np.testing.assert_allclose(data.values[..., 0, 0], scale)
    np.testing.assert_allclose(data.values[..., 1, 1], 2.0 * scale)
    np.testing.assert_allclose(data.values[..., 0, 1], 0.0)
    np.testing.assert_array_equal(data.pert_atom, [0, 0, 0])
    np.testing.assert_array_equal(data.pert_cart, [0, 1, 2])
    assert data.units == "eV/angstrom"
