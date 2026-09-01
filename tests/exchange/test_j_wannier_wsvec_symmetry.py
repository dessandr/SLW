from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slw.core.structure import find_nearest_neighbours
from slw.core.wannier_io import write_wannier_hr
from slw.exchange.kernels.j_epr import _orbit_label_map
from slw.exchange.kernels.j_wannier import (
    _group_orbits,
    _load_spinor_hr_hk,
    _resolve_wsvec_path,
)


def _write_two_orbital_wsvec(path: Path) -> None:
    records = {
        (1, 1): ((0, 0, 0),),
        (1, 2): ((1, 0, 0),),
        (2, 1): ((-1, 0, 0),),
        (2, 2): ((0, 0, 0),),
    }
    rows = ["# use_ws_distance=.true."]
    for (row, column), shifts in records.items():
        rows.append(f"0 0 0 {row} {column}")
        rows.append(str(len(shifts)))
        rows.extend(" ".join(str(value) for value in shift) for shift in shifts)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_wsvec_applies_matrix_element_specific_fourier_phases(
    tmp_path: Path,
) -> None:
    hr = tmp_path / "model_hr.dat"
    wsvec = tmp_path / "model_wsvec.dat"
    block = np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.complex128)
    write_wannier_hr(hr, 2, (1,), {(0, 0, 0): block})
    _write_two_orbital_wsvec(wsvec)
    kpoints = np.asarray(((0.25, 0.0, 0.0),), dtype=np.float64)

    plain, _plain_meta = _load_spinor_hr_hk(
        hr, kpoints, groupby="spin"
    )
    mdrs, metadata = _load_spinor_hr_hk(
        hr, kpoints, groupby="spin", wsvec=wsvec
    )

    np.testing.assert_allclose(plain[0], block, atol=1.0e-14)
    np.testing.assert_allclose(
        mdrs[0],
        np.asarray(((0.0, 1.0j), (-1.0j, 0.0)), dtype=np.complex128),
        atol=1.0e-14,
    )
    assert metadata["wsvec_applied"] is True
    assert metadata["wsvec_matrix_elements"] == 4
    assert metadata["wsvec_shifted_r_points"] == 3


def test_wsvec_standard_sibling_is_auto_detected_and_can_be_disabled(
    tmp_path: Path,
) -> None:
    hr = tmp_path / "seed_hr.dat"
    wsvec = tmp_path / "seed_wsvec.dat"
    hr.touch()
    wsvec.touch()

    resolved, source = _resolve_wsvec_path(hr)
    assert resolved == str(wsvec)
    assert source == "auto_sibling"
    assert _resolve_wsvec_path(hr, enabled=False) == (None, "disabled")
    with pytest.raises(ValueError, match="conflicts with use_wsvec=false"):
        _resolve_wsvec_path(hr, wsvec, enabled=False)


def _write_mnte_structure(path: Path) -> None:
    path.write_text(
        """begin unit_cell_cart
ang
4.15 0 0
-2.075 3.594005426 0
0 0 6.71
end unit_cell_cart
begin atoms_frac
Mn1 0 0 0
Mn2 0 0 0.5
Te1 0.3333333333 0.6666666666 0.25
Te2 0.6666666666 0.3333333333 0.75
end atoms_frac
""",
        encoding="utf-8",
    )


def test_spglib_orbits_mix_equivalent_site_pairs_in_same_crystal_orbit(
    tmp_path: Path,
) -> None:
    win = tmp_path / "MnTe.win"
    _write_mnte_structure(win)
    neighbours = find_nearest_neighbours(
        win,
        (0, 1),
        n_shells=10,
        d_max=20.0,
        all_bonds=True,
    )
    orbits = _group_orbits(
        neighbours,
        "spglib",
        structure_path=win,
        symprec=1.0e-5,
    )
    shell_ten = [
        orbit for orbit in orbits if int(orbit[0]["shell_idx"]) == 10
    ]

    assert len(shell_ten) == 2
    assert sorted(len(orbit) for orbit in shell_ten) == [12, 12]
    for orbit in shell_ten:
        assert {(int(bond["i"]), int(bond["j"])) for bond in orbit} == {
            (0, 0),
            (1, 1),
        }

    pair_meta = [
        {
            "gi": int(bond["i"]),
            "gj": int(bond["j"]),
            "R": tuple(int(value) for value in bond["R"]),
            "shell": int(bond["shell_idx"]),
        }
        for bond in neighbours
    ]
    label_map = _orbit_label_map(pair_meta, orbits)
    shell_ten_labels = {
        label_map[(item["gi"], item["gj"], item["R"])]
        for item in pair_meta
        if item["shell"] == 10
    }
    assert shell_ten_labels == {"10a", "10b"}
    for label in shell_ten_labels:
        pairs = {
            (item["gi"], item["gj"])
            for item in pair_meta
            if item["shell"] == 10
            and label_map[(item["gi"], item["gj"], item["R"])] == label
        }
        assert pairs == {(0, 0), (1, 1)}
