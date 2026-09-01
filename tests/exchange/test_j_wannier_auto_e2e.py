from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from slw.cli.mpi import MPIContext
from slw.core.wannier_io import write_wannier_hr
from slw.exchange.config import build_exchange_request
from slw.exchange.engine import run_exchange
from slw.soc.spinor import reorder_spinor_matrix, reorder_spinor_rows


def _write_structure(path: Path) -> None:
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


def _spin_major_hamiltonian() -> np.ndarray:
    half_dim = 5
    matrix = np.zeros((2 * half_dim, 2 * half_dim), dtype=np.complex128)
    matrix[:half_dim, :half_dim] = np.diag([-1.20, -0.55, -0.95, -0.35, -0.75])
    matrix[half_dim:, half_dim:] = np.diag([0.85, -0.15, 1.05, -0.05, 1.20])
    for left, right, up, down in (
        (0, 1, 0.12, 0.09),
        (1, 2, 0.10, 0.07),
        (2, 3, 0.08, 0.06),
        (3, 4, 0.07, 0.05),
        (0, 2, 0.04, 0.03),
    ):
        matrix[left, right] = matrix[right, left] = up
        matrix[left + half_dim, right + half_dim] = down
        matrix[right + half_dim, left + half_dim] = down

    # Keep a small spin-mixing term so the parity check exercises a genuinely
    # spinorial matrix permutation rather than two independent scalar blocks.
    matrix[1, half_dim + 1] = 0.025j
    matrix[half_dim + 1, 1] = -0.025j
    matrix[3, half_dim + 3] = -0.015j
    matrix[half_dim + 3, 3] = 0.015j
    return matrix


def _write_spinor_inputs(root: Path, groupby: str) -> tuple[Path, Path]:
    spin_major = _spin_major_hamiltonian()
    file_order = reorder_spinor_matrix(
        spin_major,
        source="spin",
        target=groupby,
    )
    hr_path = root / f"model_{groupby}_hr.dat"
    write_wannier_hr(hr_path, file_order.shape[0], [1], {(0, 0, 0): file_order})

    up_centres = (
        "X 0.50 0 0",
        "X 3.00 0 0",
        "X 5.50 0 0",
        "X 3.10 0 0",
        "X 0.60 0 0",
    )
    down_centres = tuple(
        f"X {float(row.split()[1]) + 0.01:.2f} 0 0" for row in up_centres
    )
    centre_rows = reorder_spinor_rows(
        up_centres + down_centres,
        source="spin",
        target=groupby,
    )
    atom_rows = ("Mn1 0.5 0 0", "Mn2 5.5 0 0", "Te1 3.0 0 0")
    centres_path = root / f"model_{groupby}_centres.xyz"
    rows = centre_rows + list(atom_rows)
    centres_path.write_text(
        f"{len(rows)}\nWannier centres and atoms\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return hr_path, centres_path


def _run_groupby(root: Path, win: Path, groupby: str) -> dict[str, np.ndarray]:
    hr_path, centres_path = _write_spinor_inputs(root, groupby)
    request = build_exchange_request(
        "j_tensor",
        {
            "input_format": "wannier",
            "spinor_hr": str(hr_path),
            "win": str(win),
            "centres": str(centres_path),
            "groupby": groupby,
            "spin_operator": "pauli",
            "centre_tolerance_ang": 0.2,
            "efermi": 0.0,
            "kmesh": (1, 1, 1),
            "mag_atoms": (1, 0),
            "tensor_kernel": "tb2j",
            "n_shells": 1,
            "d_max": 5.1,
            "emin": -2.0,
            "empoints": 6,
            "nproc": 1,
        },
        prefix=f"auto_{groupby}",
        savedir=root / "save",
    )
    assert request.slices == ()
    run_exchange(
        request,
        context=MPIContext(),
        execution="serial",
        verbosity="quiet",
    )

    with h5py.File(request.output.h5_path, "r") as handle:
        assert handle["basic_data/input_groupby"].asstr()[()] == groupby
        assert (
            handle["basic_data/magnetic_subspace_source"].asstr()[()]
            == "centres_xyz_pbc_nearest_atom"
        )
        np.testing.assert_array_equal(
            handle["basic_data/atom_labels"].asstr()[:],
            ["Mn1", "Mn2", "Te1"],
        )
        np.testing.assert_array_equal(
            handle["magnetic_subspace/selected_mag_atoms"][:], [1, 0]
        )
        selected = handle["magnetic_subspace/selector_indices"][:]
        full_half_basis = handle["magnetic_subspace/orbital_atom_index"][:]
        assert full_half_basis.size == 5
        assert selected.size == 3
        assert full_half_basis.size > selected.size

        tensor = handle["J_tensor_r"][:]
        assert tensor.size > 0
        assert np.all(np.isfinite(tensor))
        return {
            "tensor": tensor,
            "mag_i": handle["bonds/mag_i_atom"][:],
            "mag_j": handle["bonds/mag_j_atom"][:],
            "r": handle["bonds/R"][:],
        }


def test_automatic_subspace_is_groupby_invariant_end_to_end(tmp_path: Path) -> None:
    win = tmp_path / "model.win"
    _write_structure(win)

    spin = _run_groupby(tmp_path, win, "spin")
    orbital = _run_groupby(tmp_path, win, "orbital")

    for key in ("mag_i", "mag_j", "r"):
        np.testing.assert_array_equal(orbital[key], spin[key])
    np.testing.assert_allclose(
        orbital["tensor"],
        spin["tensor"],
        rtol=1.0e-11,
        atol=1.0e-11,
    )
