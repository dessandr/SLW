from __future__ import annotations

from pathlib import Path

import pytest

from slw.cli.namelist import load
from slw.cli.schema import parse_run_config
from slw.exchange.config import build_exchange_request

EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize(
    ("filename", "calculation", "ltensor", "source"),
    (
        ("exchange_j_epr.in", "j", False, "epr"),
        ("exchange_j_tensor_epr.in", "j", True, "epr"),
        ("exchange_dj_epr.in", "dj", False, "epr"),
        ("exchange_dj_tensor_epr.in", "dj", True, "epr"),
        ("exchange_j_tensor_wannier_soc.in", "j", True, "wannier"),
    ),
)
def test_exchange_examples_validate_without_scientific_io(
    tmp_path: Path,
    filename: str,
    calculation: str,
    ltensor: bool,
    source: str,
) -> None:
    config = parse_run_config(
        load(EXAMPLE_ROOT / filename),
        stage="exchange",
        cwd=tmp_path,
    )
    request = build_exchange_request(
        config.control.calculation,
        config.parameters,
        prefix=config.control.prefix,
        savedir=config.control.savedir,
    )

    assert request.calculation.value == calculation
    assert request.ltensor is ltensor
    assert request.source.value == source
    assert config.parallel.workers_per_rank == 1
    assert config.parallel.threads_per_worker == 1
    assert not Path(config.control.savedir).exists()

    if filename.endswith("wannier_soc.in"):
        assert request.groupby is not None
        assert request.groupby.value == "orbital"
        assert request.slices == ()
        assert request.files.win is not None
        assert request.files.centres is not None
        assert request.soc is not None
        assert tuple(item.selector for item in request.soc.manifolds) == (
            "Ligand-p",
            "Mag1-d",
        )
    else:
        assert request.soc is None


def test_exchange_example_inventory_is_explicit() -> None:
    assert {path.name for path in EXAMPLE_ROOT.glob("exchange_*.in")} == {
        "exchange_j_epr.in",
        "exchange_j_tensor_epr.in",
        "exchange_dj_epr.in",
        "exchange_dj_tensor_epr.in",
        "exchange_j_tensor_wannier_soc.in",
    }
