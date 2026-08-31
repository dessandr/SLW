from __future__ import annotations

import json

import h5py
import numpy as np

from slw.wtorque.config import RunConfig
from slw.wtorque.io.dfpt import lift_collinear, lift_spin_scalar
from slw.wtorque.model.local_projection import localize_exchange
from slw.wtorque.model.magnetic_subspace import MagneticSubspace
from slw.wtorque.parallel.benchmark import benchmark_q
from slw.wtorque.pipeline import run_compute_kernel
from slw.wtorque.torque.vertices import finite_q_vertices, transverse_vertices


def _write_inputs(tmp_path):
    electron_path = tmp_path / "electrons.h5"
    dfpt_path = tmp_path / "dfpt.h5"
    identity = np.eye(2, dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    h_trs = -0.1 * identity
    h_xc = 0.35 * sz
    with h5py.File(electron_path, "w") as handle:
        electrons = handle.create_group("electrons")
        electrons.attrs["bloch_gauge"] = "atomic_position"
        electrons.create_dataset("lattice", data=np.eye(3, dtype=np.float64))
        electrons.create_dataset("R_vectors", data=np.zeros((1, 3), dtype=np.int32))
        electrons.create_dataset("H_R", data=(h_trs + h_xc)[None])
        electrons.create_dataset("H_TRS_R", data=h_trs[None])
        electrons.create_dataset("H_XC_R", data=h_xc[None])
        electrons.create_dataset("orbital_centers", data=np.zeros((1, 3), dtype=np.float64))
        electrons.create_dataset("orbital_site", data=np.array([0], dtype=np.int32))
        electrons.create_dataset(
            "orbital_labels",
            data=np.array(["d"], dtype=h5py.string_dtype("utf-8")),
        )
        electrons.create_dataset("kpoints", data=np.zeros((1, 3), dtype=np.float64))
        electrons.create_dataset("weights", data=np.ones(1, dtype=np.float64))
        electrons.create_dataset("fermi_energy", data=np.float64(0.0))
        spin = handle.create_group("spin")
        spin.create_dataset("magnetic_atom_index", data=np.array([0], dtype=np.int32))
        spin.create_dataset("magnetic_site_position", data=np.zeros((1, 3), dtype=np.float64))
        spin.create_dataset("magnetic_orbital_mask", data=np.ones((1, 1), dtype=bool))
        spin.create_dataset("local_frames", data=np.eye(3, dtype=np.float64)[None])
        spin.create_dataset("spin_length", data=np.array([1.5], dtype=np.float64))
    with h5py.File(dfpt_path, "w") as handle:
        dfpt = handle.create_group("dfpt")
        dfpt.attrs["normalization"] = "cartesian_derivative"
        dfpt.create_dataset("qpoints", data=np.zeros((1, 3), dtype=np.float64))
        dfpt.create_dataset("pert_atom", data=np.zeros(3, dtype=np.int32))
        dfpt.create_dataset("pert_cart", data=np.arange(3, dtype=np.int32))
        qgroup = dfpt.create_group("q_000000")
        qgroup.create_dataset("g_cart", data=np.stack((sx, sy, identity))[None])
    return electron_path, dfpt_path


def _config(tmp_path, electron_path, dfpt_path):
    return RunConfig.from_mapping(
        {
            "electrons": {
                "file": str(electron_path),
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
                "file": str(dfpt_path),
                "normalization": "cartesian_derivative",
                "spinor_lift": "native_spinor",
                "final_state_representation": "unwrapped",
            },
            "kernel": {
                "include_direct_vertex": False,
                "fixed_chemical_potential": True,
                "q_pair_completion": True,
            },
            "integration": {
                "backend": "real_axis",
                "eta_eV": 0.04,
                "energy_min_eV": -1.0,
                "energy_max_eV": 0.0,
                "energy_points": 24,
            },
            "output": {"file": str(tmp_path / "wtorque.h5"), "resume": True},
        },
        base_dir=tmp_path,
    )


def test_spin_lifting_is_orbital_major_and_has_no_spin_flip_guess():
    scalar = np.array([[[1.0 + 0j]]], dtype=np.complex128)
    lifted = lift_spin_scalar(scalar)
    np.testing.assert_allclose(lifted[0], np.eye(2))
    up = np.array([[[2.0 + 0j]]], dtype=np.complex128)
    down = np.array([[[3.0 + 0j]]], dtype=np.complex128)
    np.testing.assert_allclose(lift_collinear(up, down)[0], np.diag([2.0, 3.0]))


def test_finite_q_q0_provider_reduces_to_local_partition():
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    hxc = np.kron(np.array([[1.0, 0.3], [0.3, 0.8]]), sz).astype(np.complex128)
    subspace = MagneticSubspace.from_masks(np.array([[True, False], [False, True]]))
    local, _ = localize_exchange(hxc, subspace.projectors)
    frames = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
    got = finite_q_vertices(
        hxc[None],
        hxc[None],
        orbital_masks=subspace.orbital_masks,
        local_frames=frames,
        q_red=np.zeros(3),
        orbital_centers=np.zeros((2, 3)),
        magnetic_site_positions=np.zeros((2, 3)),
    )
    expected = np.stack([transverse_vertices(item, frame) for item, frame in zip(local, frames)])
    np.testing.assert_allclose(got[0], expected)


def test_file_pipeline_computes_and_resumes_q_group(tmp_path):
    electron_path, dfpt_path = _write_inputs(tmp_path)
    config = _config(tmp_path, electron_path, dfpt_path)
    result = run_compute_kernel(config)
    assert result.computed_q_indices == (0,)
    with h5py.File(config.output.file, "r") as handle:
        assert handle["kernel/K_pi_u"].shape == (1, 1, 2, 1, 3)
        assert handle["kernel/K_pi_u"].dtype == np.complex128
        checksum = str(handle["q_data/q_000000"].attrs["payload_sha256"])
        assert json.loads(handle["meta/run_manifest_json"][()].decode())["schema_version"] == 1
    resumed = run_compute_kernel(config)
    assert resumed.computed_q_indices == ()
    with h5py.File(config.output.file, "r") as handle:
        assert str(handle["q_data/q_000000"].attrs["payload_sha256"]) == checksum


def test_reference_benchmark_agrees_with_vectorized_file_kernel(tmp_path):
    electron_path, dfpt_path = _write_inputs(tmp_path)
    config = _config(tmp_path, electron_path, dfpt_path)
    result = benchmark_q(config, 0, repeats=1)
    assert result.agrees
    assert result.max_absolute_difference < 1e-12
