from __future__ import annotations

import contextlib
import io

import h5py
import numpy as np
import yaml

from slw.wtorque.cli import main
from slw.wtorque.config import RunConfig
from slw.wtorque.pipeline import run_compute_kernel
from slw.wtorque.stages import (
    run_assemble_polaron,
    run_project_magnons,
    run_project_phonons,
)
from slw.wtorque.validation.reports import validate_run

from .test_io_pipeline import _config as _electronic_config
from .test_io_pipeline import _write_inputs


def _write_bosons(tmp_path):
    phonon_path = tmp_path / "phonons.h5"
    with h5py.File(phonon_path, "w") as handle:
        group = handle.create_group("phonon")
        group.create_dataset("qpoints", data=np.zeros((1, 3), dtype=np.float64))
        group.create_dataset("mass", data=np.array([12.0], dtype=np.float64))
        qgroup = group.create_group("q_000000")
        qgroup.create_dataset(
            "frequency", data=np.array([0.05, 0.06, 0.07], dtype=np.float64)
        )
        qgroup.create_dataset(
            "eigenvector", data=np.eye(3, dtype=np.complex128)[None]
        )
    magnon_path = tmp_path / "magnons.h5"
    with h5py.File(magnon_path, "w") as handle:
        group = handle.create_group("magnon")
        group.attrs["nambu_order"] = "a_q,a_minus_q_dagger"
        group.create_dataset("qpoints", data=np.zeros((1, 3), dtype=np.float64))
        group.create_dataset("energy", data=np.array([[0.08]], dtype=np.float64))
        group.create_dataset("T_para", data=np.eye(2, dtype=np.complex128)[None])
        group.create_dataset("metric", data=np.array([1.0, -1.0], dtype=np.float64))
        group.create_dataset("local_frames", data=np.eye(3, dtype=np.float64)[None])
        group.create_dataset("spin_length", data=np.array([1.5], dtype=np.float64))
    return phonon_path, magnon_path


def _full_config(tmp_path):
    electrons, dfpt = _write_inputs(tmp_path)
    phonons, magnons = _write_bosons(tmp_path)
    mapping = _electronic_config(tmp_path, electrons, dfpt).raw
    mapping["phonons"] = {"file": str(phonons)}
    mapping["magnons"] = {
        "file": str(magnons),
        "require_external_paraunitary": True,
    }
    return RunConfig.from_mapping(mapping, base_dir=tmp_path)


def test_projection_stages_write_existing_polaron_contract(tmp_path):
    config = _full_config(tmp_path)
    run_compute_kernel(config)
    run_project_phonons(config)
    run_project_magnons(config)
    run_assemble_polaron(config)
    with h5py.File(config.output.file, "r") as handle:
        assert handle["kernel/V_pi_ph"].shape == (1, 1, 2, 3)
        assert handle["coupling/g_mp_normal"].shape == (1, 1, 3)
        assert handle["coupling/g_mp_anomalous"].shape == (1, 1, 3)
        assert handle["polaron/H_rwa"].shape == (1, 4, 4)
        assert handle["polaron/energy"].shape == (1, 4)


def test_inspect_cli_and_deterministic_validation_report(tmp_path):
    config = _full_config(tmp_path)
    config_path = tmp_path / "run.yml"
    config_path.write_text(yaml.safe_dump(config.raw), encoding="utf-8")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert main(["inspect", str(config_path)]) == 0
    assert '"resolved_magnetic_orbitals"' in stdout.getvalue()
    run_compute_kernel(config)
    first = validate_run(config)
    second = validate_run(config)
    assert first.to_json() == second.to_json()
    with h5py.File(config.output.file, "r") as handle:
        assert "validation/report_json" in handle

