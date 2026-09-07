from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest

from slw.wtorque.config import RunConfig
from slw.wtorque.errors import ParaunitarityError
from slw.wtorque.io.magnon import HDF5MagnonProvider
from slw.wtorque.parallel.benchmark import benchmark_q
from slw.wtorque.pipeline import run_compute_kernel
from slw.wtorque.projection.magnon import project_external_magnons
from slw.wtorque.stages import (
    run_assemble_polaron,
    run_project_magnons,
    run_project_phonons,
)

from .test_stages_cli_validation import _full_config, _write_bosons


def test_production_coordinate_choice_preserves_coupling_and_hybrid_energies(tmp_path):
    outputs = []
    for coordinate in ("transverse_direction", "rotation_angle"):
        directory = tmp_path / coordinate
        directory.mkdir()
        raw = copy.deepcopy(_full_config(directory).raw)
        raw["magnetic_subspace"]["spin_coordinate"] = coordinate
        config = RunConfig.from_mapping(raw, base_dir=directory)
        with h5py.File(config.magnons_file, "r+") as handle:
            c, s = np.cosh(0.5), np.sinh(0.5)
            handle["magnon/T_para"][0] = np.array([[c, s], [s, c]], dtype=np.complex128)
        run_compute_kernel(config)
        run_project_phonons(config)
        run_project_magnons(config)
        run_assemble_polaron(config)
        with h5py.File(config.output.file) as handle:
            outputs.append([handle[name][...] for name in (
                "coupling/g_mp_normal", "coupling/g_mp_anomalous", "polaron/energy"
            )])
    assert np.linalg.norm(outputs[0][0]) > 0.01
    for first, second in zip(*outputs, strict=True):
        np.testing.assert_allclose(first, second, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("policy", ["onsite_only", "user_supplied"])
def test_unsupported_projection_cannot_silently_compute_local_partition(tmp_path, policy):
    raw = copy.deepcopy(_full_config(tmp_path).raw)
    raw["magnetic_subspace"]["site_projection"] = policy
    config = RunConfig.from_mapping(raw, base_dir=tmp_path)
    with pytest.raises((RuntimeError, NotImplementedError), match=f"site_projection={policy}"):
        run_compute_kernel(config)
    with pytest.raises(NotImplementedError, match=f"site_projection={policy}"):
        benchmark_q(config, q_index=0)


@pytest.mark.parametrize("dataset,value", [
    ("energy", -0.08), ("energy", 0.0), ("energy", np.nan), ("energy", np.inf),
    ("T_para", np.nan), ("T_para", np.inf),
    ("qpoints", np.nan), ("spin_length", np.nan), ("spin_length", -1.0),
])
def test_magnon_provider_rejects_invalid_numerical_data(tmp_path, dataset, value):
    _, path = _write_bosons(tmp_path)
    with h5py.File(path, "r+") as handle:
        data = handle[f"magnon/{dataset}"][...]
        data.flat[0] = value
        handle[f"magnon/{dataset}"][...] = data
    with pytest.raises((ValueError, ParaunitarityError)):
        HDF5MagnonProvider(path, expected_frames=np.eye(3)[None], expected_spin_lengths=[1.5])


@pytest.mark.parametrize("argument", ["v_pi_ph", "t_magnon", "spin_lengths", "t_phonon", "tolerance"])
def test_public_projection_rejects_nan(argument):
    arguments = dict(
        v_pi_ph=np.ones((1, 2, 1), dtype=np.complex128),
        t_magnon=np.eye(2, dtype=np.complex128),
        spin_lengths=np.array([1.5]),
        t_phonon=np.eye(2, dtype=np.complex128),
        tolerance=1e-9,
    )
    if argument == "tolerance":
        arguments[argument] = np.nan
    else:
        arguments[argument].flat[0] = np.nan
    with pytest.raises((ValueError, ParaunitarityError)):
        project_external_magnons(**arguments)


def test_flat_rotation_coefficients_match_transverse_coefficients():
    rotation = np.array([[2+1j], [3-2j]], dtype=np.complex128)
    transverse = np.array([[3-2j], [-2-1j]], dtype=np.complex128)
    transform = np.array([[np.cosh(.5), np.sinh(.5)], [np.sinh(.5), np.cosh(.5)]], dtype=np.complex128)
    first = project_external_magnons(rotation, transform, [1.5], spin_coordinate="rotation_angle")
    second = project_external_magnons(transverse, transform, [1.5])
    np.testing.assert_allclose(first.full_nambu, second.full_nambu)
