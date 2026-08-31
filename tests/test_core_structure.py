from __future__ import annotations

import numpy as np
import pytest

from slw.core.structure import atom_names_from_structure, read_wannier90_structure


def _write_win(tmp_path, atoms_block: str):
    path = tmp_path / "material.win"
    path.write_text(
        """
begin unit_cell_cart
ang
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
end unit_cell_cart

"""
        + atoms_block.strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_numbered_site_labels_atoms_frac_preserve_labels_and_elements(tmp_path):
    path = _write_win(
        tmp_path,
        """
begin atoms_frac
Mn1 0.0 0.0 0.0
Mn2 0.5 0.5 0.5
Te1 0.25 0.25 0.25
end atoms_frac
""",
    )

    lattice, species, positions, numbers = read_wannier90_structure(path)

    np.testing.assert_allclose(lattice, np.eye(3) * 2.0)
    assert species == ["Mn1", "Mn2", "Te1"]
    np.testing.assert_allclose(
        positions,
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )
    np.testing.assert_array_equal(numbers, [25, 25, 52])
    assert atom_names_from_structure(path) == {0: "Mn1", 1: "Mn2", 2: "Te1"}


def test_numbered_site_labels_atoms_cart_map_to_fractional_coordinates(tmp_path):
    path = _write_win(
        tmp_path,
        """
begin atoms_cart
ang
Mn1 0.0 0.0 0.0
Mn2 1.0 1.0 1.0
Te1 0.5 1.5 0.0
end atoms_cart
""",
    )

    _, species, positions, numbers = read_wannier90_structure(path)

    assert species == ["Mn1", "Mn2", "Te1"]
    np.testing.assert_allclose(
        positions,
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.75, 0.0]],
    )
    np.testing.assert_array_equal(numbers, [25, 25, 52])
    assert atom_names_from_structure(path) == {0: "Mn1", 1: "Mn2", 2: "Te1"}


def test_plain_repeated_species_receive_stable_one_based_site_labels(tmp_path):
    path = _write_win(
        tmp_path,
        """
begin atoms_frac
Mn 0.0 0.0 0.0
Mn 0.5 0.5 0.5
Te 0.25 0.25 0.25
end atoms_frac
""",
    )

    _, species, _, numbers = read_wannier90_structure(path)

    assert species == ["Mn", "Mn", "Te"]
    np.testing.assert_array_equal(numbers, [25, 25, 52])
    assert atom_names_from_structure(path) == {0: "Mn1", 1: "Mn2", 2: "Te1"}


@pytest.mark.parametrize("invalid_label", ["Xx1", "Mn_site1", "1Mn"])
def test_invalid_numbered_chemical_symbol_fails_closed(tmp_path, invalid_label):
    path = _write_win(
        tmp_path,
        f"""
begin atoms_frac
{invalid_label} 0.0 0.0 0.0
end atoms_frac
""",
    )

    with pytest.raises(ValueError, match="Unknown chemical symbols"):
        read_wannier90_structure(path)
