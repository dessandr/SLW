from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from slw.exchange.kernels.j_tensor_epr import _apply_model_soc, _resolve_soc_entries


def _write_win(path: Path) -> None:
    path.write_text(
        """
begin atoms_frac
Mn1 0.0 0.0 0.0
Mn2 0.5 0.5 0.0
Te1 0.333 0.667 0.25
Te2 0.667 0.333 0.75
end atoms_frac
begin projections
Mn1:s
Mn1:d
Mn2:d
Te:p
end projections
""",
        encoding="utf-8",
    )


def _args(win: Path, manifolds) -> argparse.Namespace:
    return argparse.Namespace(
        win=str(win),
        soc_manifolds=tuple(manifolds),
        epr_up="",
    )


def test_species_and_site_manifolds_resolve_without_manual_indices(
    tmp_path: Path,
) -> None:
    win = tmp_path / "model.win"
    _write_win(win)
    args = _args(
        win,
        (
            {"selector": "Te-p", "lambda_ev": 0.5},
            {"selector": "Mn1-d", "lambda_ev": 0.05},
        ),
    )
    entries, resolved = _resolve_soc_entries(args, nwan=17)
    assert resolved == str(win)
    assert entries[0]["matched_labels"] == ["Te1", "Te2"]
    assert entries[0]["groups"] == [[11, 12, 13], [14, 15, 16]]
    assert entries[1]["matched_labels"] == ["Mn1"]
    assert entries[1]["groups"] == [[1, 2, 3, 4, 5]]

    hamiltonian = np.zeros((2, 34, 34), dtype=np.complex128)
    with_soc, applied, _ = _apply_model_soc(hamiltonian, args, 17)
    assert len(applied) == 2
    assert np.linalg.norm(with_soc) > 0.0


def test_overlapping_species_and_site_selectors_fail(tmp_path: Path) -> None:
    win = tmp_path / "model.win"
    _write_win(win)
    args = _args(
        win,
        (
            {"selector": "Te-p", "lambda_ev": 0.5},
            {"selector": "Te1-p", "lambda_ev": 0.4},
        ),
    )
    with pytest.raises(ValueError, match="overlaps a previous selector"):
        _resolve_soc_entries(args, nwan=17)


def test_unknown_manifold_and_projection_count_fail(tmp_path: Path) -> None:
    win = tmp_path / "model.win"
    _write_win(win)
    with pytest.raises(ValueError, match="matched no Wannier manifold"):
        _resolve_soc_entries(
            _args(win, ({"selector": "O-p", "lambda_ev": 0.2},)),
            nwan=17,
        )
    with pytest.raises(ValueError, match="projection count"):
        _resolve_soc_entries(
            _args(win, ({"selector": "Te-p", "lambda_ev": 0.2},)),
            nwan=16,
        )
