from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from slw.cli.mpi import MPIContext
from slw.core.wannier_io import write_wannier_hr
from slw.exchange.config import build_exchange_request
from slw.exchange.engine import run_exchange
from slw.exchange.kernels.spinor_hr import build_spinor_hr


def _literal_spin_major_to_file_indices(nwan: int, groupby: str) -> np.ndarray:
    """Independent TB2J-order oracle; do not reuse production permutations."""

    if groupby == "spin":
        return np.arange(2 * nwan, dtype=np.int64)
    if groupby == "orbital":
        return np.column_stack(
            [np.arange(nwan), np.arange(nwan, 2 * nwan)]
        ).reshape(-1)
    raise ValueError(groupby)


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
    permutation = _literal_spin_major_to_file_indices(5, groupby)
    file_order = spin_major[np.ix_(permutation, permutation)]
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
    spin_major_rows = up_centres + down_centres
    centre_rows = [spin_major_rows[int(index)] for index in permutation]
    atom_rows = ("Mn1 0.5 0 0", "Mn2 5.5 0 0", "Te1 3.0 0 0")
    centres_path = root / f"model_{groupby}_centres.xyz"
    rows = centre_rows + list(atom_rows)
    centres_path.write_text(
        f"{len(rows)}\nWannier centres and atoms\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return hr_path, centres_path


def _run_groupby(
    root: Path,
    win: Path,
    groupby: str,
    *,
    verbosity: str = "quiet",
) -> dict[str, np.ndarray]:
    hr_path, centres_path = _write_spinor_inputs(root, groupby)
    request = build_exchange_request(
        "j_tensor",
        {
            "input_format": "wannier",
            "spinor_hr": str(hr_path),
            "win": str(win),
            "centres": str(centres_path),
            "groupby": groupby,
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
        verbosity=verbosity,
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
        np.testing.assert_array_equal(
            handle["magnetic_subspace/selector_offsets"][:], [0, 1, 3]
        )
        np.testing.assert_array_equal(selected, [2, 0, 4])
        np.testing.assert_array_equal(full_half_basis, [0, 2, 1, 2, 0])
        expected_partners = (
            np.column_stack([np.arange(5), np.arange(5, 10)])
            if groupby == "spin"
            else np.arange(10).reshape(5, 2)
        )
        np.testing.assert_array_equal(
            handle["magnetic_subspace/partner_file_indices"][:],
            expected_partners,
        )
        assert (
            handle["basic_data/spin_operator_resolved"].asstr()[()]
            == "pauli_product_basis"
        )
        assert "spin_operator_validation" not in handle

        tensor = handle["J_tensor_r"][:]
        assert tensor.size > 0
        assert np.all(np.isfinite(tensor))
        assert np.linalg.norm(tensor) > 0.0
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


def test_standard_spinor_high_verbosity_omits_internal_basis_dumps(
    tmp_path: Path,
    capsys,
) -> None:
    win = tmp_path / "model.win"
    _write_structure(win)

    _run_groupby(tmp_path, win, "orbital", verbosity="high")
    output = capsys.readouterr().out

    assert "separability" not in output
    assert "basis_groups" not in output
    assert "orbital_to_atom" not in output
    assert "partner_distances" not in output
    assert "mag_subspace_file_order" not in output
    assert output.count("magnetic subspace") == 1
    assert "orbitals_per_site={0: 1, 1: 2}" in output


def _write_two_site_collinear_inputs(
    root: Path,
) -> tuple[Path, Path, Path, Path]:
    win = root / "collinear.win"
    win.write_text(
        """begin unit_cell_cart
ang
4 0 0
0 4 0
0 0 4
end unit_cell_cart
begin atoms_frac
Mn1 0 0 0
Mn2 0.5 0 0
end atoms_frac
""",
        encoding="utf-8",
    )
    up_hr = root / "collinear_up_hr.dat"
    dn_hr = root / "collinear_dn_hr.dat"
    up = np.asarray([[-1.10, 0.18], [0.18, -0.72]], dtype=np.complex128)
    down = np.asarray([[0.92, 0.11], [0.11, 1.16]], dtype=np.complex128)
    write_wannier_hr(up_hr, 2, [1], {(0, 0, 0): up})
    write_wannier_hr(dn_hr, 2, [1], {(0, 0, 0): down})

    centres_up = root / "collinear_up_centres.xyz"
    centres_dn = root / "collinear_dn_centres.xyz"
    rows = (
        "4\nWannier centres and atoms\n"
        "X 0 0 0\nX 2 0 0\n"
        "Mn1 0 0 0\nMn2 2 0 0\n"
    )
    centres_up.write_text(rows, encoding="utf-8")
    centres_dn.write_text(rows, encoding="utf-8")
    return win, up_hr, dn_hr, centres_up


def _run_two_site_request(
    root: Path,
    *,
    prefix: str,
    files: dict[str, object],
) -> dict[str, np.ndarray]:
    request = build_exchange_request(
        "j_tensor",
        {
            "input_format": "wannier",
            "win": str(root / "collinear.win"),
            "efermi": 0.0,
            "kmesh": (1, 1, 1),
            "mag_atoms": (0, 1),
            "tensor_kernel": "tb2j",
            "n_shells": 1,
            "d_max": 2.1,
            "emin": -2.0,
            "empoints": 8,
            "nproc": 1,
            **files,
        },
        prefix=prefix,
        savedir=root / "save",
    )
    run_exchange(
        request,
        context=MPIContext(),
        execution="serial",
        verbosity="quiet",
    )
    with h5py.File(request.output.h5_path, "r") as handle:
        return {
            "tensor": handle["J_tensor_r"][:],
            "mag_i": handle["bonds/mag_i_atom"][:],
            "mag_j": handle["bonds/mag_j_atom"][:],
            "r": handle["bonds/R"][:],
        }


def test_collinear_and_native_spinor_paths_have_identical_tb2j_tensor(
    tmp_path: Path,
) -> None:
    _win, up_hr, dn_hr, centres_up = _write_two_site_collinear_inputs(tmp_path)
    collinear = _run_two_site_request(
        tmp_path,
        prefix="collinear_reference",
        files={
            "up_hr": str(up_hr),
            "dn_hr": str(dn_hr),
            "slices": "0:0:1,1:1:2",
        },
    )

    for groupby in ("spin", "orbital"):
        output_hr = tmp_path / f"built_{groupby}_hr.dat"
        output_centres = tmp_path / f"built_{groupby}_centres.xyz"
        build_spinor_hr(
            argparse.Namespace(
                up_hr=str(up_hr),
                dn_hr=str(dn_hr),
                prefix_up=None,
                prefix_dn=None,
                output=str(output_hr),
                out_prefix=None,
                prefix=None,
                groupby=groupby,
                centres_up=str(centres_up),
                centres_dn=str(tmp_path / "collinear_dn_centres.xyz"),
                centres_output=str(output_centres),
                spin_direction=(0.0, 0.0, 1.0),
            )
        )
        native = _run_two_site_request(
            tmp_path,
            prefix=f"native_{groupby}",
            files={
                "spinor_hr": str(output_hr),
                "centres": str(output_centres),
                "groupby": groupby,
                "centre_tolerance_ang": 0.1,
            },
        )
        for key in ("mag_i", "mag_j", "r"):
            np.testing.assert_array_equal(native[key], collinear[key])
        np.testing.assert_allclose(
            native["tensor"], collinear["tensor"], rtol=1.0e-11, atol=1.0e-11
        )
        assert np.linalg.norm(native["tensor"]) > 0.0
