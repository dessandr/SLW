from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from slw.magph.legacy.plotting.config import PlotConfig
from slw.magph.legacy.plotting.magnon_h5 import (
    _resolve_bond_factor,
    load_j_h5,
)


def _write_exchange(
    path: Path,
    *,
    mate_complete: bool = True,
    kernel: str | None = None,
    kernel_family: str | None = None,
    hamiltonian_sign: str | None = None,
    bond_coverage: str | None = None,
    directed_bond_weight: float | None = None,
    command: str | None = None,
    j_dataset: str = "J_iso_r",
) -> None:
    atom_i = np.asarray((0, 1) if mate_complete else (0,), dtype=np.int64)
    atom_j = np.asarray((1, 0) if mate_complete else (1,), dtype=np.int64)
    shifts = np.zeros((atom_i.size, 3), dtype=np.int64)
    with h5py.File(path, "w") as handle:
        basic = handle.create_group("basic_data")
        basic.create_dataset("lattice_ang", data=np.eye(3, dtype=np.float64))
        basic.create_dataset(
            "tau_cart_ang",
            data=np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))),
        )
        basic.create_dataset("atom_labels", data=np.asarray((b"A", b"B")))
        for name, value in (
            ("kernel", kernel),
            ("kernel_family", kernel_family),
            ("hamiltonian_sign", hamiltonian_sign),
            ("bond_coverage", bond_coverage),
            ("command", command),
        ):
            if value is not None:
                basic.create_dataset(name, data=np.bytes_(value))
        if directed_bond_weight is not None:
            basic.create_dataset("directed_bond_weight", data=directed_bond_weight)

        bonds = handle.create_group("bonds")
        bonds.create_dataset("mag_i_atom", data=atom_i)
        bonds.create_dataset("mag_j_atom", data=atom_j)
        bonds.create_dataset("R", data=shifts)
        parent, _, name = j_dataset.rpartition("/")
        group = handle.require_group(parent) if parent else handle
        group.create_dataset(name, data=np.full(atom_i.size, -2.0))


def _config(path: Path, **values: str) -> PlotConfig:
    return PlotConfig(
        path=path.parent / "plot.in",
        values={"input": str(path), **values},
    )


@pytest.mark.parametrize(
    ("source_weight", "expected_factor"),
    ((0.5, 1.0), (1.0, 2.0)),
)
def test_metadata_weight_is_mapped_to_legacy_half_directed_sum(
    tmp_path: Path,
    source_weight: float,
    expected_factor: float,
) -> None:
    path = tmp_path / "exchange.h5"
    _write_exchange(
        path,
        kernel_family="scalar_lkag",
        hamiltonian_sign="minus",
        bond_coverage="directed_mate_complete",
        directed_bond_weight=source_weight,
    )
    config = _config(path)
    payload, _ = load_j_h5(config)

    assert _resolve_bond_factor(config, payload) == expected_factor


def test_old_tb2j_all_bonds_payload_infers_unit_source_weight(tmp_path: Path) -> None:
    path = tmp_path / "old_tb2j.h5"
    _write_exchange(
        path,
        kernel="tb2j",
        command="compute_J_epr_tensor.py --kernel tb2j --all_bonds --nproc 4",
    )
    config = _config(path)
    payload, _ = load_j_h5(config)

    assert payload.convention.command_all_bonds
    assert payload.convention.mate_complete
    assert _resolve_bond_factor(config, payload) == 2.0


def test_scalar_writer_dataset_uses_declared_source_weight(tmp_path: Path) -> None:
    path = tmp_path / "scalar_j.h5"
    _write_exchange(
        path,
        kernel_family="scalar_lkag",
        hamiltonian_sign="minus",
        bond_coverage="directed_mate_complete",
        directed_bond_weight=1.0,
        j_dataset="J_r/value",
    )
    config = _config(path)
    payload, _ = load_j_h5(config)

    assert payload.j_source == "J_r/value"
    assert _resolve_bond_factor(config, payload) == 2.0


def test_missing_weight_without_auditable_tb2j_provenance_fails(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.h5"
    _write_exchange(path, hamiltonian_sign="minus")
    config = _config(path)
    payload, _ = load_j_h5(config)

    with pytest.raises(ValueError, match="no basic_data/directed_bond_weight"):
        _resolve_bond_factor(config, payload)


def test_tb2j_without_all_bonds_or_coverage_does_not_guess(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous_tb2j.h5"
    _write_exchange(path, kernel="tb2j")
    config = _config(path)
    payload, _ = load_j_h5(config)

    with pytest.raises(ValueError, match="not an auditable raw TB2J"):
        _resolve_bond_factor(config, payload)


def test_declared_mate_complete_bonds_are_checked_against_topology(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.h5"
    _write_exchange(
        path,
        mate_complete=False,
        kernel="tb2j",
        hamiltonian_sign="minus",
        bond_coverage="directed_mate_complete",
        directed_bond_weight=1.0,
    )
    config = _config(path)
    payload, _ = load_j_h5(config)

    with pytest.raises(ValueError, match="not mate-complete"):
        _resolve_bond_factor(config, payload)


def test_explicit_bond_factor_remains_an_override_for_legacy_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "external_legacy.h5"
    _write_exchange(path, mate_complete=False)
    config = _config(path, bond_factor="0.75")
    payload, _ = load_j_h5(config)

    assert _resolve_bond_factor(config, payload) == 0.75


@pytest.mark.parametrize("value", ("nan", "0", "-1"))
def test_nonpositive_or_nonfinite_explicit_bond_factor_is_rejected(
    tmp_path: Path,
    value: str,
) -> None:
    path = tmp_path / "exchange.h5"
    _write_exchange(path)
    config = _config(path, bond_factor=value)
    payload, _ = load_j_h5(config)

    with pytest.raises(ValueError, match="positive finite number"):
        _resolve_bond_factor(config, payload)
