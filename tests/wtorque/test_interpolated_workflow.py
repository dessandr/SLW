"""Restart/provenance and complete Cartesian-to-boson artifact checks."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from slw.wtorque import interpolated_workflow as workflow
from slw.wtorque.polaron_bands import NativePathModes


def _modes(qpoints):
    nq = len(qpoints)
    valid = np.max(np.abs(qpoints - np.rint(qpoints)), axis=1) > 1.e-12
    em = np.full((nq, 1), .04)
    ep = np.tile([.02, .06, .08], (nq, 1))
    tm = np.tile(np.eye(2, dtype=complex), (nq, 1, 1))
    tp = np.tile(np.eye(3, dtype=complex).reshape(1, 1, 3, 3), (nq, 1, 1, 1))
    for value in (em, ep, tm, tp):
        value[~valid] = np.nan
    magnons = SimpleNamespace(spin_lengths=np.array([1.]), local_frames=np.eye(3)[None])
    phonons = SimpleNamespace(masses_amu=np.array([10.]))
    return NativePathModes(np.flatnonzero(valid), valid, tuple("stable" if ok else "gamma_zero_mode_omitted" for ok in valid),
                           magnons, phonons, em, ep, em.copy(), ep.copy(), tm, tp, tm.copy(), tp.copy())


def _pair(q):
    kernel = np.zeros((2, 1, 2, 1, 3), complex)
    kernel[0, 0, 0, 0, 0] = .01 + .02j
    kernel[1] = kernel[0].conj()
    return dict(qpoints=np.asarray([q, -q]), bubble=kernel, direct=-.9 * kernel,
                total=.1 * kernel, diagnostics={"synthetic": True})


def _fake_workflow(tmp_path, monkeypatch, qpoints):
    import slw.wtorque.io.dense_epr as backend_module
    import slw.wtorque.interpolated_response as response_module

    native = dict(epr="fixture.epr", exchange_out="fixture.exchange", spin_lengths=[1.],
                  magnetic_atom_labels=["M1"], source_directed_bond_weight=1.,
                  ep_energy_unit="ev", ep_displacement_unit="angstrom", include_direct_vertex=True)
    monkeypatch.setattr(workflow, "load_native_workflow_config", lambda path: native.copy())
    monkeypatch.setattr(workflow, "_root_inputs", lambda cfg: {
        "frame": None, "phonons": SimpleNamespace(lattice_ang=np.eye(3)),
        "summary": {"source_sha256": {"fixture": "same"}},
    })
    monkeypatch.setattr(workflow, "native_path_modes", lambda epr, ex, q, **kw: _modes(q))
    monkeypatch.setattr(backend_module, "DenseEPREvaluator", lambda *a, **k: SimpleNamespace(diagnostics={"polar_policy": "test"}))

    class Response:
        calls = []
        fail_at = None

        def __init__(self, *args):
            self.diagnostics = {"fixture_response": True}

        def evaluate_pair(self, q):
            type(self).calls.append(q.copy())
            if type(self).fail_at is not None and np.allclose(q, type(self).fail_at):
                raise ValueError("simulated interruption")
            return _pair(q)

    monkeypatch.setattr(response_module, "ArbitraryQResponse", Response)
    config = dict(native_config="native.json", kmesh=[2, 2, 2], output="result.h5", cache_dir="cache", qpoints=qpoints)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config))
    return path, Response


def test_signed_pair_deduplication_without_reciprocal_folding():
    points = np.array([[.2, 0, 0], [-.2, 0, 0], [1.2, 0, 0], [.2, 0, 0], [0, 0, 0]])
    tasks, owners, signs = workflow._tasks(points, np.array([1, 1, 1, 1, 0], bool))
    np.testing.assert_allclose(tasks, [[.2, 0, 0], [1.2, 0, 0]])
    np.testing.assert_array_equal(owners, [0, 0, 1, 0, -1])
    np.testing.assert_array_equal(signs, [0, 1, 0, 0, 0])


def test_independent_lr_sr_options_reach_backend_and_change_restart_identity(tmp_path, monkeypatch):
    import slw.wtorque.io.dense_epr as backend_module

    configpath, _ = _fake_workflow(tmp_path, monkeypatch, [[.2, 0, 0]])
    calls = []

    def backend(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(diagnostics={"longrange_model": kwargs["longrange_model"],
                                            "short_range_model": kwargs["short_range_model"]})

    monkeypatch.setattr(backend_module, "DenseEPREvaluator", backend)
    original = workflow.run_interpolated_workflow(configpath)
    assert calls[-1]["longrange_model"] == calls[-1]["short_range_model"] == "source"
    config = json.loads(configpath.read_text())
    config.update(longrange_model="point_center", short_range_model="two_center",
                  longrange_coarse_qpoints=[[0., 0., 0.]])
    configpath.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="provenance"):
        workflow.run_interpolated_workflow(configpath)
    config["output"] = "corrected.h5"
    configpath.write_text(json.dumps(config))
    corrected = workflow.run_interpolated_workflow(configpath)
    assert calls[-1]["longrange_model"] == "point_center"
    assert calls[-1]["short_range_model"] == "two_center"
    assert corrected["fingerprint"] != original["fingerprint"]


@pytest.mark.parametrize("key,value", [("longrange_model", "full_overlap"),
                                       ("short_range_model", "average"),
                                       ("longrange_model", []),
                                       ("short_range_model", None)])
def test_invalid_lr_sr_option_rejected_before_inputs(tmp_path, key, value):
    config = dict(native_config="missing.json", kmesh=[2, 2, 2], output="out.h5",
                  cache_dir="cache", **{key: value})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        workflow.load_interpolated_workflow_config(path)


def test_shard_round_trip_checks_provenance_pair_and_total(tmp_path):
    target = tmp_path / "q.npz"
    q = np.array([.2, 0, 0])
    workflow._save_shard(target, _pair(q), "hash")
    actual = workflow._load_shard(target, "hash", q)
    np.testing.assert_allclose(actual["total"], _pair(q)["total"])
    with pytest.raises(ValueError, match="provenance"):
        workflow._load_shard(target, "changed", q)
    with pytest.raises(ValueError, match="q pair"):
        workflow._load_shard(target, "hash", -q)
    corrupt = _pair(q)
    corrupt["total"] *= 2
    workflow._save_shard(target, corrupt, "hash")
    with pytest.raises(ValueError, match="total differs"):
        workflow._load_shard(target, "hash", q)


def test_complete_signed_modes_and_restart_does_not_recompute(tmp_path, monkeypatch):
    config, response = _fake_workflow(tmp_path, monkeypatch, [[.2, 0, 0], [-.2, 0, 0], [0, 0, 0]])
    report = workflow.run_interpolated_workflow(config)
    assert report["response_q_pair_count"] == 1
    assert report["stable_polaron_q_count"] == 2
    assert len(response.calls) == 1
    with h5py.File(tmp_path / "result.h5") as handle:
        assert handle["coupling/full_nambu_total"].shape == (3, 2, 6)
        assert np.all(np.isnan(handle["bands/energies_eV"][-1]))
        np.testing.assert_allclose(handle["kernel/K_pi_u_total"][0].conj(), handle["kernel/K_pi_u_total"][1])
        assert handle["bands/paraunitarity_residual"][:2].max() < 1.e-13
    rerun = workflow.run_interpolated_workflow(config)
    assert rerun["fingerprint"] == report["fingerprint"]
    assert len(response.calls) == 1
    assert (tmp_path / "result_low_energy.pdf").exists()
    changed = json.loads(config.read_text())
    changed["kmesh"] = [3, 3, 3]
    config.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="different/incomplete provenance"):
        workflow.run_interpolated_workflow(config)


def test_interrupted_q_task_resumes_from_completed_shard(tmp_path, monkeypatch):
    config, response = _fake_workflow(tmp_path, monkeypatch, [[.2, 0, 0], [.3, 0, 0]])
    response.fail_at = [.3, 0, 0]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        workflow.run_interpolated_workflow(config)
    assert len(list((tmp_path / "cache").glob("responses-*/q_*.npz"))) == 1
    response.fail_at = None
    workflow.run_interpolated_workflow(config)
    np.testing.assert_allclose(response.calls, [[.2, 0, 0], [.3, 0, 0], [.3, 0, 0]])


def test_post_hdf5_plot_interruption_recovers_missing_artifacts(tmp_path, monkeypatch):
    config, response = _fake_workflow(tmp_path, monkeypatch, [[.2, 0, 0]])
    original = workflow.plot_polaron_bands
    monkeypatch.setattr(workflow, "plot_polaron_bands", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("plot interrupted")))
    with pytest.raises(RuntimeError, match="plot interrupted"):
        workflow.run_interpolated_workflow(config)
    assert (tmp_path / "result.h5").exists()
    assert not (tmp_path / "result.json").exists()
    monkeypatch.setattr(workflow, "plot_polaron_bands", original)
    workflow.run_interpolated_workflow(config)
    assert len(response.calls) == 1
    for suffix in (".json", ".csv", "_bands.png", "_bands.pdf", "_low_energy.png", "_low_energy.pdf"):
        assert (tmp_path / ("result" + suffix)).exists()


def test_uniform_qmesh_is_centered_and_sampling_choices_exclusive(tmp_path):
    path = tmp_path / "config.json"
    config = dict(native_config="native.json", kmesh=[2, 3, 4], output="out.h5", cache_dir="cache", qmesh=[3, 1, 1])
    path.write_text(json.dumps(config))
    loaded = workflow.load_interpolated_workflow_config(path)
    sampled, kind = workflow._path(loaded, {}, np.eye(3))
    assert kind == "uniform_qmesh"
    np.testing.assert_allclose(sampled.qpoints[:, 0], [0, 1/3, -1/3])
    config["qpoints"] = [[.2, 0, 0]]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mutually exclusive"):
        workflow.load_interpolated_workflow_config(path)


def test_partial_figure_or_table_is_not_installed(tmp_path):
    target = tmp_path / "artifact.png"
    def interrupted(temporary):
        temporary.write_bytes(b"partial")
        raise RuntimeError("writer interrupted")
    with pytest.raises(RuntimeError, match="writer interrupted"):
        workflow._atomic_file(target, interrupted)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
    workflow._atomic_file(target, lambda temporary: temporary.write_bytes(b"complete"))
    assert target.read_bytes() == b"complete"
