"""Restartable dense electronic-mesh mixed response and magnon-polaron bands."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import h5py
import numpy as np

from slw.wtorque.native_workflow import load_native_workflow_config, _root_inputs, _hash
from slw.wtorque.parallel.mpi import MPIContext
from slw.wtorque.polaron_bands import (
    NativePathModes, PolaronBands, SampledBandPath, native_path_modes, plot_polaron_bands,
    sample_wannier_path, solve_polaron_bands,
)
from slw.wtorque.projection.magnon import project_external_magnons
from slw.wtorque.projection.phonon import project_phonons


def _json(value: object) -> str:
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"not JSON serializable: {type(item)}")
    return json.dumps(value, sort_keys=True, default=encode, allow_nan=False)


def _mesh(value: object, name: str) -> tuple[int, int, int]:
    array = np.asarray(value)
    if (array.shape != (3,) or not np.issubdtype(array.dtype, np.number)
            or np.any(~np.isfinite(array)) or np.any(array < 1)
            or np.any(array != np.rint(array))):
        raise ValueError(f"{name} must contain three positive integers")
    return tuple(int(item) for item in array)


def load_interpolated_workflow_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    config = json.loads(source.read_text())
    required = {"native_config", "kmesh", "output", "cache_dir"}
    optional = {"qpoints", "qmesh", "points_per_segment",
                "longrange_model", "short_range_model", "longrange_coarse_qpoints"}
    if not isinstance(config, dict) or required - config.keys() or config.keys() - required - optional:
        raise ValueError(f"interpolated config requires {sorted(required)}; optional {sorted(optional)}")
    for key in ("native_config", "output", "cache_dir"):
        target = Path(config[key]).expanduser()
        config[key] = str((source.parent / target).resolve() if not target.is_absolute() else target.resolve())
    config["kmesh"] = list(_mesh(config["kmesh"], "kmesh"))
    for key, allowed in (("longrange_model", {"source", "point_center"}),
                         ("short_range_model", {"source", "two_center"})):
        config.setdefault(key, "source")
        if not isinstance(config[key], str) or config[key] not in allowed:
            raise ValueError(f"{key} must be one of {sorted(allowed)}")
    if config["longrange_model"] == "point_center":
        coarse_q = np.asarray(config.get("longrange_coarse_qpoints"), dtype=float)
        if (coarse_q.ndim != 2 or coarse_q.shape[1:] != (3,) or not len(coarse_q)
                or not np.all(np.isfinite(coarse_q))):
            raise ValueError("point_center requires longrange_coarse_qpoints containing the actual source q representatives")
        config["longrange_coarse_qpoints"] = coarse_q.tolist()
    elif "longrange_coarse_qpoints" in config:
        raise ValueError("longrange_coarse_qpoints requires longrange_model=point_center")
    if sum(key in config for key in ("qpoints", "qmesh", "points_per_segment")) > 1:
        raise ValueError("qpoints, qmesh, and points_per_segment are mutually exclusive")
    if "qpoints" in config:
        q = np.asarray(config["qpoints"], float)
        if q.ndim != 2 or q.shape[1] != 3 or not q.shape[0] or not np.all(np.isfinite(q)):
            raise ValueError("qpoints must have finite nonempty shape (nq,3)")
    elif "qmesh" in config:
        config["qmesh"] = list(_mesh(config["qmesh"], "qmesh"))
    else:
        value = config.setdefault("points_per_segment", 40)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("points_per_segment must be a positive integer")
    return config


def _path(config: dict[str, Any], native: dict[str, Any], lattice: np.ndarray) -> tuple[SampledBandPath, str]:
    if "qpoints" not in config and "qmesh" not in config:
        return sample_wannier_path(native["win"], lattice, points_per_segment=config["points_per_segment"]), "wannier90_path"
    if "qmesh" in config:
        mesh = np.asarray(config["qmesh"], int)
        q = np.indices(mesh).reshape(3, -1).T / mesh
        q -= np.floor(q + .5)
        kind = "uniform_qmesh"
    else:
        q = np.asarray(config["qpoints"], float)
        kind = "explicit_qpoints"
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0) @ reciprocal, axis=1))]
    ticks = distance[[0, -1]] if len(q) > 1 else distance.copy()
    labels = ("q start", "q end") if len(q) > 1 else ("q",)
    gamma = np.max(np.abs(q - np.rint(q)), axis=1) < 1.e-12
    return SampledBandPath(q, distance, np.zeros(len(q), int), ticks, labels, gamma), kind


def _tasks(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate identical signed representatives, without q+G folding."""
    lookup, q_tasks = {}, []
    owners, signs = np.full(len(points), -1, int), np.zeros(len(points), int)
    for iq in np.flatnonzero(valid):
        q = points[iq]
        nonzero = np.flatnonzero(np.abs(q) > 1.e-13)
        reverse = bool(nonzero.size and q[nonzero[0]] < 0)
        representative = -q if reverse else q
        key = tuple(np.round(representative, 14))
        if key not in lookup:
            lookup[key] = len(q_tasks)
            q_tasks.append(representative)
        owners[iq], signs[iq] = lookup[key], int(reverse)
    return np.asarray(q_tasks, float).reshape(-1, 3), owners, signs


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(_json(value) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_file(path: Path, writer) -> None:
    """Install a figure/table only after its complete write succeeds."""
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + ".tmp-", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_shard(path: Path, value: dict[str, Any], fingerprint: str) -> None:
    arrays = {key: np.asarray(value[key]) for key in ("qpoints", "bubble", "direct", "total")}
    for key in ("retarded_bubble", "retarded_direct"):
        if key in value:
            arrays[key] = np.asarray(value[key])
    arrays["diagnostics_json"] = np.asarray(_json(value.get("diagnostics", {})))
    arrays["fingerprint"] = np.asarray(fingerprint)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_shard(path: Path, fingerprint: str, qpoint: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        if str(source["fingerprint"]) != fingerprint:
            raise ValueError(f"response shard provenance differs: {path}")
        result = {key: source[key] for key in source.files if key not in {"fingerprint", "diagnostics_json"}}
        result["diagnostics"] = json.loads(str(source["diagnostics_json"]))
    if result["qpoints"].shape != (2, 3) or not np.allclose(result["qpoints"], [qpoint, -qpoint], atol=1.e-13, rtol=0):
        raise ValueError(f"response shard q pair differs: {path}")
    for name in ("bubble", "direct", "total"):
        array = result[name]
        if array.ndim != 5 or array.shape[0] != 2 or array.shape[2] != 2 or array.shape[-1] != 3 or not np.all(np.isfinite(array)):
            raise ValueError(f"response shard {name} has invalid values or dimensions: {path}")
    if result["bubble"].shape != result["direct"].shape or result["total"].shape != result["bubble"].shape:
        raise ValueError(f"response shard component dimensions differ: {path}")
    if not np.allclose(result["total"], result["bubble"] + result["direct"], atol=1.e-13, rtol=1.e-11):
        raise ValueError(f"response shard total differs from bubble+direct: {path}")
    return result


def _project_results(path: SampledBandPath, modes: NativePathModes, results: dict[int, dict[str, Any]],
                     owners: np.ndarray, signs: np.ndarray) -> dict[str, Any]:
    nq = len(path.qpoints)
    nm = modes.magnon_energies_eV.shape[1]
    np_ = modes.phonon_energies_eV.shape[1]
    nat = len(modes.phonons.masses_amu)
    kernels = {name: np.full((nq, 2, nm, 2, nat, 3), np.nan + 0j) for name in ("bubble", "direct", "total")}
    full_nambu = {name: np.full((nq, 2, 2 * nm, 2 * np_), np.nan + 0j) for name in kernels}
    for iq in np.flatnonzero(modes.valid):
        result = results[int(owners[iq])]
        for sign in range(2):
            source_sign = (int(signs[iq]) + sign) % 2
            energies = modes.phonon_energies_eV[iq] if sign == 0 else modes.phonon_energies_minus_eV[iq]
            phonon_vectors = modes.phonon_eigenvectors[iq] if sign == 0 else modes.phonon_eigenvectors_minus[iq]
            magnon_vectors = modes.magnon_transform[iq] if sign == 0 else modes.magnon_transform_minus[iq]
            for component in kernels:
                kernels[component][iq, sign] = result[component][source_sign]
                coefficient = project_phonons(
                    kernels[component][iq, sign], normalization="cartesian_derivative",
                    frequencies=energies, masses=modes.phonons.masses_amu, eigenvectors=phonon_vectors,
                )
                full_nambu[component][iq, sign] = project_external_magnons(
                    coefficient, magnon_vectors, modes.magnons.spin_lengths,
                    spin_coordinate="transverse_direction",
                ).full_nambu
    bands = solve_polaron_bands(
        modes.magnon_energies_eV, modes.phonon_energies_eV, full_nambu["total"][:, 0],
        modes.magnon_energies_minus_eV, modes.phonon_energies_minus_eV, full_nambu["total"][:, 1],
        valid_q_mask=modes.valid,
    )
    return {"kernels": kernels, "full_nambu": full_nambu, "bands": bands}


def _write_outputs(config, native, inputs, path, kind, modes, results, owners, signs,
                   fingerprint, response_diagnostics, mpi_size, elapsed):
    projected = _project_results(path, modes, results, owners, signs)
    bands = projected["bands"]
    target = Path(config["output"])
    summary = dict(inputs["summary"])
    summary.update({
        "status": "complete", "workflow": "interpolated_native_magnon_polaron",
        "interpolated_config": config, "fingerprint": fingerprint,
        "kmesh": config["kmesh"], "q_sampling": kind, "q_count": len(path.qpoints),
        "response_q_pair_count": len(results), "mpi_size": mpi_size, "elapsed_seconds": elapsed,
        "valid_mode_q_count": int(modes.valid.sum()), "stable_polaron_q_count": int(bands.valid.sum()),
        "mode_status": modes.status, "polaron_status": bands.status,
        "ep_input": "EPR pair-specific WS interpolation with analytic 3D dipole add-back",
        "electronic_interpolation": response_diagnostics,
        "dense_k_and_path_are_interpolated": True, "coarse_q_convergence_established": False,
        "g_xc_exported": False,
        "gamma_policy": "omit zero-point/Goldstone singular endpoints; no gap or stabilizing shift",
        "magnon_weight_definition": "Euclidean particle+hole magnon fraction of full Nambu eigenvector",
        "q_diagnostics": {str(key): value["diagnostics"] for key, value in results.items()},
    })
    if bands.valid.any():
        summary["max_paraunitarity_residual"] = float(np.max(bands.paraunitarity_residual[bands.valid]))
        summary["max_eigen_residual_eV"] = float(np.max(bands.eigen_residual_eV[bands.valid]))
        summary["max_mode_qpair_residual"] = float(np.max(bands.qpair_residual[bands.valid]))
        summary["maximum_normal_vertex_meV"] = float(np.nanmax(np.abs(projected["full_nambu"]["total"][:, 0, :modes.magnon_energies_eV.shape[1], :modes.phonon_energies_eV.shape[1]])) * 1.e3)
    if target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite interpolated results: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    os.close(descriptor)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(status="complete", fingerprint=fingerprint, energy_unit="eV", kernel_unit="eV/angstrom",
                                direct_term_enabled=native["include_direct_vertex"], projector_motion_enabled=False)
            handle["qpoints"] = path.qpoints
            handle["qpoints_minus"] = -path.qpoints
            for field in dataclasses.fields(path):
                value = getattr(path, field.name)
                handle[f"path/{field.name}"] = np.asarray(value, dtype=h5py.string_dtype()) if field.name == "tick_labels" else value
            handle["path/response_task_index"] = owners
            handle["path/response_pair_sign"] = signs
            handle["modes/valid"] = modes.valid
            handle["modes/status"] = np.asarray(modes.status, dtype=h5py.string_dtype())
            for name, value in (("magnons/energies_eV", modes.magnon_energies_eV),
                                ("magnons/energies_minus_eV", modes.magnon_energies_minus_eV),
                                ("phonons/energies_eV", modes.phonon_energies_eV),
                                ("phonons/energies_minus_eV", modes.phonon_energies_minus_eV),
                                ("magnons/T_para", modes.magnon_transform),
                                ("magnons/T_para_minus", modes.magnon_transform_minus),
                                ("phonons/eigenvectors", modes.phonon_eigenvectors),
                                ("phonons/eigenvectors_minus", modes.phonon_eigenvectors_minus),
                                ("magnons/spin_lengths", modes.magnons.spin_lengths),
                                ("magnons/local_frames", modes.magnons.local_frames),
                                ("phonons/masses_amu", modes.phonons.masses_amu)):
                handle[name] = value
            for name in ("bubble", "direct", "total"):
                handle[f"kernel/K_pi_u_{name}"] = projected["kernels"][name][:, 0]
                handle[f"kernel/K_pi_u_{name}_minus"] = projected["kernels"][name][:, 1]
                handle[f"coupling/full_nambu_{name}"] = projected["full_nambu"][name][:, 0]
                handle[f"coupling/full_nambu_{name}_minus"] = projected["full_nambu"][name][:, 1]
            for field in dataclasses.fields(bands):
                value = getattr(bands, field.name)
                handle[f"bands/{field.name}"] = np.asarray(value, dtype=h5py.string_dtype()) if field.name == "status" else value
            handle["metadata/summary_json"] = _json(summary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    _write_auxiliary(config, path, kind, modes, bands, summary)
    return summary


def _write_auxiliary(config, path, kind, modes, bands, summary):
    """Complete missing plot/table/JSON artifacts, including after a restart."""
    target = Path(config["output"])
    # A uniform 3-D q mesh is a sampling volume, not a labelled band path.
    if kind != "uniform_qmesh":
        for suffix, limits in (("bands", None), ("low_energy", (0., 80.))):
            for extension in ("png", "pdf"):
                figure = target.with_name(target.stem + f"_{suffix}.{extension}")
                if figure.exists():
                    continue
                _atomic_file(figure, lambda temporary: plot_polaron_bands(path, bands, temporary,
                                   magnon_energies=modes.magnon_energies_eV,
                                   phonon_energies=modes.phonon_energies_eV,
                                   energy_limits_meV=limits,
                                   title="Magnon–polaron: interpolated fixed-frame response"))
        table = np.column_stack((np.arange(len(path.qpoints)), path.qpoints, path.distance_inv_ang,
                                 bands.valid.astype(int), bands.energies_eV * 1.e3, bands.magnon_weights))
        count = bands.energies_eV.shape[1]
        columns = ["q_index", "qx", "qy", "qz", "distance_inv_ang", "valid"]
        columns += [f"energy_{i}_meV" for i in range(count)] + [f"magnon_weight_{i}" for i in range(count)]
        if not target.with_suffix(".csv").exists():
            _atomic_file(target.with_suffix(".csv"), lambda temporary: np.savetxt(
                temporary, table, delimiter=",", header=",".join(columns), comments=""))
    if not target.with_suffix(".json").exists():
        _atomic_json(target.with_suffix(".json"), summary)


def run_interpolated_workflow(configpath: str | Path, mpi_enabled: bool = False, *, mpi: MPIContext | None = None) -> dict[str, Any] | None:
    """Run q-pair tasks with root-only artifacts, atomic shards and restart."""
    from slw.wtorque.io.dense_epr import DenseEPREvaluator
    from slw.wtorque.interpolated_response import ArbitraryQResponse

    context = mpi if mpi is not None else (MPIContext.discover() if mpi_enabled else MPIContext())
    started = time.monotonic()

    def root_call(function):
        payload = None
        if context.is_root:
            try:
                payload = (True, function())
            except Exception as exc:
                payload = (False, f"{type(exc).__name__}: {exc}")
        ok, value = context.bcast(payload)
        if not ok:
            raise RuntimeError(value)
        return value

    config = root_call(lambda: load_interpolated_workflow_config(configpath))
    native = root_call(lambda: load_native_workflow_config(config["native_config"]))
    if context.is_root:
        print("[interpolated] checking native source, frame, geometry and hashes", flush=True)

    def prepare():
        scratch = dict(native)
        scratch["output"] = str(Path(config["cache_dir"]) / f"unused-preflight-{os.getpid()}.h5")
        inputs = _root_inputs(scratch)
        inputs["summary"]["resolved_config"] = native
        # Include shared LSWT, EPR and WS implementations in restart identity.
        source_root = Path(__file__).parents[1]
        inputs["summary"]["implementation_sha256"] = {
            str(source.relative_to(source_root)): _hash(str(source))
            for source in sorted(source_root.rglob("*.py"))
        }
        path, kind = _path(config, native, inputs["phonons"].lattice_ang)
        modes = native_path_modes(native["epr"], native["exchange_out"], path.qpoints,
                                  spin_lengths=native["spin_lengths"], magnetic_atom_labels=native["magnetic_atom_labels"],
                                  source_directed_bond_weight=native["source_directed_bond_weight"])
        qtasks, owners, signs = _tasks(path.qpoints, modes.valid)
        identity = {"config": config, "native_config": native,
                    "source_sha256": inputs["summary"]["source_sha256"],
                    "implementation_sha256": inputs["summary"]["implementation_sha256"],
                    "q_tasks": qtasks}
        fingerprint = hashlib.sha256(_json(identity).encode()).hexdigest()
        directory = Path(config["cache_dir"]) / ("responses-" + fingerprint)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "manifest.json"
        if manifest.exists() and manifest.read_text().strip() != _json(identity):
            raise ValueError("response restart manifest mismatch")
        if not manifest.exists():
            _atomic_json(manifest, identity)
        existing = None
        target = Path(config["output"])
        if target.exists():
            with h5py.File(target, "r") as handle:
                if handle.attrs.get("fingerprint") != fingerprint or handle.attrs.get("status") != "complete":
                    raise FileExistsError(f"existing output has different/incomplete provenance: {target}")
                raw = handle["metadata/summary_json"][()]
                existing = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        elif target.with_suffix(".json").exists():
            raise FileExistsError(f"existing summary has no matching HDF5: {target.with_suffix('.json')}")
        results = {}
        for index, q in enumerate(qtasks):
            shard = directory / f"q_{index:06d}.npz"
            if shard.exists():
                results[index] = _load_shard(shard, fingerprint, q)
        return inputs, path, kind, modes, qtasks, owners, signs, fingerprint, str(directory), results, existing

    prepared = root_call(prepare)
    inputs, path, kind, modes, qtasks, owners, signs, fingerprint, directory, results, existing = prepared
    if existing is not None:
        def repair_artifacts():
            with h5py.File(config["output"], "r") as handle:
                values = {field.name: handle[f"bands/{field.name}"][()] for field in dataclasses.fields(PolaronBands)}
            values["status"] = tuple(item.decode() if isinstance(item, bytes) else str(item) for item in values["status"])
            _write_auxiliary(config, path, kind, modes, PolaronBands(**values), existing)
        root_call(repair_artifacts)
        if context.is_root:
            print(f"[interpolated] 100.0%: verified completed output {config['output']}", flush=True)
        return existing if context.is_root else None
    if context.is_root:
        print(f"[interpolated] {len(qtasks)} q-pair tasks; {len(results)} restored; preparing WS cache", flush=True)
    # Every rank owns a light evaluator over shared read-only coefficient maps.
    local_error, response = None, None
    try:
        backend = DenseEPREvaluator(native["epr"], energy_unit=native["ep_energy_unit"],
                                    displacement_unit=native["ep_displacement_unit"], cache_dir=config["cache_dir"],
                                    longrange_model=config.get("longrange_model", "source"),
                                    short_range_model=config.get("short_range_model", "source"),
                                    longrange_coarse_qpoints=config.get("longrange_coarse_qpoints"))
        response = ArbitraryQResponse(inputs["frame"], backend, config["kmesh"], native)
        response.diagnostics["epr_backend"] = backend.diagnostics
    except Exception as exc:
        local_error = f"rank {context.rank}: {type(exc).__name__}: {exc}"
    errors = context.allgather(local_error)
    if any(errors):
        raise RuntimeError("; ".join(error for error in errors if error))
    pending = [index for index in range(len(qtasks)) if index not in results]
    for offset in range(0, len(pending), context.size):
        slot = offset + context.rank
        item = None
        if slot < len(pending):
            index = pending[slot]
            try:
                item = (index, response.evaluate_pair(qtasks[index]), None)
            except Exception as exc:
                item = (index, None, f"rank {context.rank}, q task {index}: {type(exc).__name__}: {exc}")
        gathered = context.allgather(item)
        step_errors = [item[2] for item in gathered if item is not None and item[2] is not None]
        for completed in gathered:
            if completed is None or completed[2] is not None:
                continue
            index, value, _ = completed
            results[index] = value
        def save_completed():
            for completed in gathered:
                if completed is not None and completed[2] is None:
                    index, value, _ = completed
                    _save_shard(Path(directory) / f"q_{index:06d}.npz", value, fingerprint)
            print(f"[interpolated] {100. * len(results) / len(qtasks):6.2f}% ({len(results)}/{len(qtasks)} q pairs), elapsed {time.monotonic()-started:.1f}s", flush=True)
        root_call(save_completed)
        if step_errors:
            raise RuntimeError("; ".join(step_errors))
    summary = root_call(lambda: _write_outputs(
        config, native, inputs, path, kind, modes, results, owners, signs, fingerprint,
        response.diagnostics, context.size, time.monotonic() - started,
    ))
    if context.is_root:
        print(f"[interpolated] complete: {config['output']}", flush=True)
    return summary if context.is_root else None


__all__ = ["load_interpolated_workflow_config", "run_interpolated_workflow"]
