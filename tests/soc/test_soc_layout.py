from __future__ import annotations

import importlib.util
from pathlib import Path

import slw.soc


def test_soc_root_contains_only_active_typed_modules() -> None:
    root = Path(slw.soc.__file__).resolve().parent
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py",
        "atomic.py",
        "manifold.py",
        "model.py",
        "spinor.py",
        "wannier_soc.py",
    }


def test_abandoned_fitting_modules_are_quarantined() -> None:
    for name in (
        "band_splitting",
        "build_collinear_spinor",
        "nonlocal_fit",
        "parameter",
        "parameter_offsite",
        "plot_band",
        "plot_dmi_bonds",
        "plot_dos",
        "prep_tb2j",
        "soc_SK",
        "view_hr_realspace",
    ):
        assert importlib.util.find_spec(f"slw.soc.{name}") is None
        assert importlib.util.find_spec(f"slw.soc.legacy.{name}") is not None
