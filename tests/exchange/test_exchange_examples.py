from __future__ import annotations

from pathlib import Path

import pytest

from slw.cli.namelist import load
from slw.cli.schema import parse_run_config
from slw.exchange.config import build_exchange_request
from slw.exchange.kernels.dispatch import build_namespace

EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize(
    ("filename", "calculation", "ltensor", "source"),
    (
        ("exchange_j_epr.in", "j", False, "epr"),
        ("exchange_j_tensor_epr.in", "j", True, "epr"),
        ("exchange_dj_epr.in", "dj", False, "epr"),
        ("exchange_dj_tensor_epr.in", "dj", True, "epr"),
        ("exchange_j_tensor_wannier_collinear.in", "j", True, "wannier"),
        ("exchange_j_tensor_wannier_spinor.in", "j", True, "wannier"),
        ("exchange_j_tensor_wannier_spn.in", "j", True, "wannier"),
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

    if filename.endswith("wannier_spinor.in"):
        assert request.groupby is not None
        assert request.groupby.value == "orbital"
        assert request.slices == ()
        assert request.files.win is not None
        assert request.files.centres is not None
        assert all(
            getattr(request.files, name) is None
            for name in ("amn", "eig", "spn", "u_mat", "u_dis_mat")
        )
        assert "spin_operator" not in config.parameters
        _module, _function, namespace = build_namespace(request)
        assert namespace.spin_operator == "pauli"
        assert request.soc is None
    elif filename.endswith("wannier_collinear.in"):
        assert request.groupby is None
        assert request.files.up_hr is not None
        assert request.files.dn_hr is not None
        assert request.files.spinor_hr is None
        assert request.files.win is not None
        assert request.files.centres is None
        assert request.slices
        assert request.soc is None
    elif filename.endswith("wannier_spn.in"):
        assert request.groupby is not None
        assert request.groupby.value == "orbital"
        assert request.slices == ()
        assert request.files.win is not None
        assert request.files.centres is None
        assert all(
            getattr(request.files, name) is not None
            for name in ("amn", "eig", "spn", "u_mat", "u_dis_mat")
        )
        assert request.options["spin_operator"] == "spn"
        assert request.options["u_dis_layout"] == "global_bands"
        assert request.soc is None
    else:
        assert request.soc is None


def test_exchange_example_inventory_is_explicit() -> None:
    assert {path.name for path in EXAMPLE_ROOT.glob("exchange_*.in")} == {
        "exchange_j_epr.in",
        "exchange_j_tensor_epr.in",
        "exchange_dj_epr.in",
        "exchange_dj_tensor_epr.in",
        "exchange_j_tensor_wannier_collinear.in",
        "exchange_j_tensor_wannier_spinor.in",
        "exchange_j_tensor_wannier_spn.in",
    }
