from __future__ import annotations

import h5py
import numpy as np
import pytest
from scipy import constants

from slw.wtorque.errors import IncompleteRunError, ParaunitarityError
from slw.wtorque.io.hdf5 import RestartableHDF5
from slw.wtorque.parallel.mpi import MPIContext, MPIUnavailableError
from slw.wtorque.parallel.scheduler import build_q_pair_schedule, partition_pairs
from slw.wtorque.projection.magnon import project_external_magnons
from slw.wtorque.projection.phonon import (
    phonon_zero_point_displacement,
    project_phonons,
)


def test_phonon_cartesian_projection_and_mode_path_agree():
    frequencies = np.array([0.025])
    masses = np.array([12.0])
    eigenvectors = np.zeros((1, 3, 1), dtype=np.complex128)
    eigenvectors[0, 0, 0] = 1.0
    kernel = np.zeros((1, 2, 1, 3), dtype=np.complex128)
    kernel[0, 0, 0, 0] = 2.0
    xi = phonon_zero_point_displacement(frequencies, masses, eigenvectors)
    expected_xi = np.sqrt(
        constants.hbar**2
        / (2 * 12.0 * constants.atomic_mass * 0.025 * constants.electron_volt)
    ) / constants.angstrom
    np.testing.assert_allclose(xi[0, 0, 0], expected_xi)
    cartesian = project_phonons(
        kernel,
        normalization="cartesian_derivative",
        frequencies=frequencies,
        masses=masses,
        eigenvectors=eigenvectors,
    )
    mode = project_phonons(
        cartesian,
        normalization="phonon_zero_point_mode",
    )
    np.testing.assert_allclose(mode, cartesian)


def test_magnon_projection_matches_circular_combination_and_checks_metric():
    v = np.array([[[2.0 + 0.5j], [-0.3 + 0.4j]]], dtype=np.complex128)
    result = project_external_magnons(v, np.eye(2, dtype=np.complex128), [2.0])
    expected = (v[0, 0] + 1j * v[0, 1]) / np.sqrt(4.0)
    np.testing.assert_allclose(result.normal[0], expected)
    assert result.anomalous.shape == (1, 1)

    invalid = np.array([[1.0, 0.2], [0.0, 1.0]], dtype=np.complex128)
    with pytest.raises(ParaunitarityError):
        project_external_magnons(v, invalid, [2.0])


def test_restart_checksum_and_manifest_mismatch_are_rejected(tmp_path):
    output = tmp_path / "restart.h5"
    qpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    manifest = {"config": {"tag": "one"}, "source_hashes": {}}
    with RestartableHDF5(output, qpoints, manifest, resume=False) as writer:
        writer.write_q(0, {"kernel/K_pi_u": np.ones((1, 2, 1, 3))})
        writer.write_q(1, {"kernel/K_pi_u": np.zeros((1, 2, 1, 3))})
        writer.materialize()

    with h5py.File(output, "r+") as handle:
        handle["q_data/q_000000/kernel/K_pi_u"][0, 0, 0, 0] = 9.0
    with (
        pytest.raises(IncompleteRunError, match="checksum"),
        RestartableHDF5(output, qpoints, manifest, resume=True) as writer,
    ):
        writer.verify_complete_q(0)

    changed = {"config": {"tag": "two"}, "source_hashes": {}}
    with pytest.raises(IncompleteRunError, match="manifest"):
        RestartableHDF5(output, qpoints, changed, resume=True)


def test_q_pairs_are_unique_and_balanced_for_mpi_ownership():
    qpoints = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 0.0, 0.0], [0.5, 0, 0]]
    )
    pairs = build_q_pair_schedule(qpoints)
    assert {(pair.positive, pair.negative) for pair in pairs} == {
        (0, 0),
        (1, 2),
        (3, 3),
    }
    owned = [partition_pairs(pairs, rank, 2) for rank in range(2)]
    assert sorted(pair.positive for chunk in owned for pair in chunk) == [0, 1, 3]
    assert abs(len(owned[0]) - len(owned[1])) <= 1


def test_multi_rank_launch_without_mpi4py_fails_instead_of_writing_serially(
    monkeypatch,
):
    monkeypatch.setenv("PMI_SIZE", "2")
    monkeypatch.setitem(__import__("sys").modules, "mpi4py", None)
    with pytest.raises(MPIUnavailableError, match="2-rank"):
        MPIContext.discover()
