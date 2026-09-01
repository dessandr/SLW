from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from slw.core.wannier_io import read_wannier_hr, write_wannier_hr
from slw.exchange.kernels.j_wannier import _load_spinor_hr_hk
from slw.exchange.kernels.spinor_hr import build_spinor_hr
from slw.soc.spinor import (
    groupby_to_spin_major_indices,
    reorder_spinor_matrix,
    spin_major_to_groupby_indices,
)


def test_tb2j_groupby_permutations_are_exact() -> None:
    np.testing.assert_array_equal(
        spin_major_to_groupby_indices(3, "spin"), np.arange(6)
    )
    np.testing.assert_array_equal(
        spin_major_to_groupby_indices(3, "orbital"), [0, 3, 1, 4, 2, 5]
    )
    np.testing.assert_array_equal(
        groupby_to_spin_major_indices(3, "orbital"), [0, 2, 4, 1, 3, 5]
    )

    matrix = np.arange(36).reshape(6, 6)
    orbital = reorder_spinor_matrix(matrix, source="spin", target="orbital")
    restored = reorder_spinor_matrix(orbital, source="orbital", target="spin")
    np.testing.assert_array_equal(restored, matrix)


def _write_centres(path: Path, wannier_x: tuple[float, float], atom_x: float) -> None:
    path.write_text(
        "4\ncentres\n"
        f"X {wannier_x[0]} 0 0\n"
        f"X {wannier_x[1]} 0 0\n"
        f"Mn1 {atom_x} 0 0\n"
        "Te1 0.5 0.5 0.5\n",
        encoding="utf-8",
    )


def _builder_args(tmp_path: Path, *, groupby: str) -> argparse.Namespace:
    return argparse.Namespace(
        up_hr=str(tmp_path / "up_hr.dat"),
        dn_hr=str(tmp_path / "dn_hr.dat"),
        prefix_up=None,
        prefix_dn=None,
        output=str(tmp_path / f"spinor_{groupby}_hr.dat"),
        out_prefix=None,
        prefix=None,
        groupby=groupby,
        centres_up=str(tmp_path / "up_centres.xyz"),
        centres_dn=str(tmp_path / "dn_centres.xyz"),
        centres_output=str(tmp_path / f"spinor_{groupby}_centres.xyz"),
        spin_direction=(0.0, 0.0, 1.0),
    )


@pytest.mark.parametrize("groupby", ["spin", "orbital"])
def test_collinear_builder_permutes_hr_and_centres_together(
    tmp_path: Path, groupby: str
) -> None:
    r0 = (0, 0, 0)
    r1 = (1, 0, 0)
    up = {
        r0: np.diag([1.0, 2.0]).astype(np.complex128),
        r1: np.diag([0.1, 0.2]).astype(np.complex128),
    }
    down = {
        r0: np.diag([3.0, 4.0]).astype(np.complex128),
        r1: np.diag([0.3, 0.4]).astype(np.complex128),
    }
    write_wannier_hr(tmp_path / "up_hr.dat", 2, [1, 2], up)
    write_wannier_hr(tmp_path / "dn_hr.dat", 2, [1, 2], down)
    _write_centres(tmp_path / "up_centres.xyz", (0.1, 0.2), 0.0)
    _write_centres(tmp_path / "dn_centres.xyz", (0.3, 0.4), 0.0)

    args = _builder_args(tmp_path, groupby=groupby)
    build_spinor_hr(args)
    dimension, degeneracies, blocks = read_wannier_hr(args.output)
    assert dimension == 4
    assert degeneracies == [1, 2]
    expected_spin = np.diag([1.0, 2.0, 3.0, 4.0])
    # Keep this oracle independent from the production reorder helper used by
    # the loader: C=[up...|down...], F=[w0 up,w0 down,...].
    permutation = (
        np.arange(4, dtype=np.int64)
        if groupby == "spin"
        else np.asarray([0, 2, 1, 3], dtype=np.int64)
    )
    expected = expected_spin[np.ix_(permutation, permutation)]
    np.testing.assert_allclose(blocks[r0], expected)

    canonical_hk, metadata = _load_spinor_hr_hk(
        args.output,
        np.zeros((1, 3), dtype=np.float64),
        apply_degeneracy=False,
        groupby=groupby,
    )
    np.testing.assert_allclose(
        canonical_hk[0],
        np.diag([1.1, 2.2, 3.3, 4.4]),
    )
    assert metadata["input_groupby"] == groupby
    assert metadata["internal_groupby"] == "spin"

    lines = Path(args.centres_output).read_text(encoding="utf-8").splitlines()
    assert int(lines[0]) == 6
    wannier_rows = lines[2:6]
    spin_rows = ["X 0.1 0 0", "X 0.2 0 0", "X 0.3 0 0", "X 0.4 0 0"]
    assert wannier_rows == [spin_rows[int(index)] for index in permutation]
    assert lines[6:] == ["Mn1 0.0 0 0", "Te1 0.5 0.5 0.5"]


def test_centres_require_two_consistent_spin_channels(tmp_path: Path) -> None:
    block = {(0, 0, 0): np.eye(2, dtype=np.complex128)}
    write_wannier_hr(tmp_path / "up_hr.dat", 2, [1], block)
    write_wannier_hr(tmp_path / "dn_hr.dat", 2, [1], block)
    _write_centres(tmp_path / "up_centres.xyz", (0.1, 0.2), 0.0)
    _write_centres(tmp_path / "dn_centres.xyz", (0.3, 0.4), 0.1)
    args = _builder_args(tmp_path, groupby="spin")

    args.centres_dn = None
    with pytest.raises(ValueError, match="both centres_up and centres_dn"):
        build_spinor_hr(args)
    args.centres_dn = str(tmp_path / "dn_centres.xyz")
    with pytest.raises(ValueError, match="differs between spin channels"):
        build_spinor_hr(args)
