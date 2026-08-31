from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from slw.wtorque.basis import require_complex128
from slw.wtorque.config import ConfigurationError, RunConfig
from slw.wtorque.equations import lint_equation_registry
from slw.wtorque.provenance import build_run_manifest, canonical_manifest_json


def _config(electrons: str, dfpt: str, output: str) -> dict[str, object]:
    return {
        "electrons": {
            "file": electrons,
            "bloch_gauge": "atomic_position",
            "exchange_extraction": "explicit",
            "spin_order": "interleaved",
        },
        "magnetic_subspace": {
            "policy": "explicit_indices",
            "sites": [{"atom": 0, "orbitals": [0]}],
            "site_projection": "local_partition",
            "spin_coordinate": "transverse_direction",
        },
        "dfpt": {
            "file": dfpt,
            "normalization": "cartesian_derivative",
            "spinor_lift": "native_spinor",
        },
        "kernel": {
            "include_direct_vertex": False,
            "fixed_chemical_potential": True,
            "q_pair_completion": True,
        },
        "integration": {"backend": "real_axis", "eta_eV": 0.01},
        "output": {"file": output, "resume": True},
    }


def test_config_rejects_unknown_convention_and_missing_normalization(tmp_path):
    mapping = _config("e.h5", "g.h5", "out.h5")
    mapping["electrons"]["bloch_gauge"] = "guess_for_me"
    with pytest.raises(ConfigurationError, match="bloch_gauge"):
        RunConfig.from_mapping(mapping, base_dir=tmp_path)

    mapping = _config("e.h5", "g.h5", "out.h5")
    del mapping["dfpt"]["normalization"]
    with pytest.raises(ConfigurationError, match="normalization"):
        RunConfig.from_mapping(mapping, base_dir=tmp_path)


def test_complex64_is_never_silently_promoted():
    with pytest.raises(TypeError, match="complex128"):
        require_complex128("H_R", np.eye(2, dtype=np.complex64))


def test_run_manifest_is_deterministic_and_hashes_inputs(tmp_path):
    electrons = tmp_path / "electrons.h5"
    dfpt = tmp_path / "dfpt.h5"
    electrons.write_bytes(b"electrons")
    dfpt.write_bytes(b"dfpt")
    config = RunConfig.from_mapping(
        _config(str(electrons), str(dfpt), str(tmp_path / "out.h5")),
        base_dir=tmp_path,
    )

    first = build_run_manifest(config)
    second = build_run_manifest(config)
    assert canonical_manifest_json(first) == canonical_manifest_json(second)
    decoded = json.loads(canonical_manifest_json(first))
    assert decoded["source_hashes"]["electrons"] == decoded["source_hashes"][
        "electrons"
    ]
    assert "created_at" not in decoded
    assert "hostname" not in decoded


def test_all_implemented_equations_are_registered_in_the_plan():
    index = (
        Path(__file__).parents[2]
        / "new_feature/wannier_torque_magnon_polaron_impl_plan_v2/docs/17_equation_index.md"
    )
    assert lint_equation_registry(index) == ()
