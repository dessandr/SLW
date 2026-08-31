from __future__ import annotations

from pathlib import Path

import slw.wtorque


def test_wtorque_is_an_slw_subpackage_without_top_level_duplicate() -> None:
    package = Path(slw.wtorque.__file__).resolve().parent
    repository = Path(__file__).resolve().parents[2]
    assert package.parent.name == "slw"
    assert package == repository / "slw" / "wtorque"
    assert not (repository / "wtorque").exists()
