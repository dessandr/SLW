"""Native spinor coarse-grid mixed response → phonon → magnon workflow.

This is a fixed-projector, projection-anchored rigid-spin approximation.  It
retains SOC in H and rotates only its model TR-odd field. The optional direct
term differentiates this same model at fixed AMN frame; it is not a separately
isolated QE XC potential. All momenta, sites and choices come from inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import h5py
import numpy as np

from slw.epc.epr_io import read_epr_metadata
from slw.wtorque.gauge.kq_map import build_kq_map
from slw.wtorque.io.native_coarse import (
    periodic_grid_indices, read_native_coarse_grids, read_native_coarse_vertex,
    validate_native_xml_geometry,
)
from slw.wtorque.io.native_epr import load_native_spinor_basis
from slw.wtorque.io.native_epr_h import reconstruct_epr_hamiltonian
from slw.wtorque.model.native_spinor_frame import (
    build_native_spinor_frame, read_spinor_projection_metadata,
)
from slw.wtorque.native_modes import native_magnon_modes, native_phonon_modes, validate_phonons_against_qe_dyn_xml
from slw.wtorque.parallel.mpi import MPIContext
from slw.wtorque.projection.magnon import project_external_magnons
from slw.wtorque.projection.phonon import project_phonons
from slw.wtorque.projection.polaron import assemble_rwa
from slw.wtorque.torque.kernel import retarded_bubble_loop_eigh_zero_temperature_finite_q
from slw.wtorque.torque.qpair import finalize_q_pair
from slw.wtorque.torque.direct_vertex import (
    finite_q_direct_vertices, retarded_direct_loop_eigh_zero_temperature,
)


_FILES = ("epr", "elph", "qe_xml", "win", "amn", "eig", "u_mat", "u_dis_mat", "nnkp", "exchange_out")
_REQUIRED = set(_FILES) | {
    "dynamical_xml", "qpoints", "output", "u_dis_layout", "atomic_spin_order",
    "spin_lengths", "magnetic_atom_labels", "source_directed_bond_weight",
    "fermi_energy_eV", "energy_min_eV", "eta_eV", "ep_energy_unit",
    "ep_displacement_unit", "approximation",
}
_OPTIONAL = {"perturbation_chunk", "projector_tr_tolerance", "g_reciprocity_tolerance", "hamiltonian_tolerance_eV", "g_pair_policy", "include_direct_vertex", "g_xc_source", "export_g_xc"}


def load_native_workflow_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    config = json.loads(source.read_text())
    if not isinstance(config, dict):
        raise ValueError("native workflow config must be a JSON object")
    if _REQUIRED - config.keys() or config.keys() - _REQUIRED - _OPTIONAL:
        raise ValueError(f"native config missing={sorted(_REQUIRED - config.keys())}, unknown={sorted(config.keys() - _REQUIRED - _OPTIONAL)}")
    if config["approximation"] not in {"projection_anchored_bubble", "projection_anchored_frozen_frame_total"}:
        raise ValueError("native approximation must be projection_anchored_bubble or projection_anchored_frozen_frame_total")
    config.setdefault("include_direct_vertex", False)
    if not isinstance(config["include_direct_vertex"], bool):
        raise ValueError("include_direct_vertex must be boolean")
    if config["include_direct_vertex"] != (config["approximation"] == "projection_anchored_frozen_frame_total"):
        raise ValueError("direct flag and declared bubble/frozen-frame-total approximation disagree")
    if config["include_direct_vertex"] and config.get("g_xc_source") != "fixed_frame_tr_odd":
        raise ValueError("native direct term requires explicit g_xc_source=fixed_frame_tr_odd; separately supplied g_XC uses the strict HDF5 workflow")
    if not config["include_direct_vertex"] and config.get("g_xc_source") is not None:
        raise ValueError("g_xc_source requires include_direct_vertex=true")
    config.setdefault("export_g_xc", config["include_direct_vertex"])
    if not isinstance(config["export_g_xc"], bool) or (config["export_g_xc"] and not config["include_direct_vertex"]):
        raise ValueError("export_g_xc is boolean and requires an enabled direct term")
    def resolve(value: str) -> str:
        target = Path(value).expanduser()
        return str((source.parent / target).resolve() if not target.is_absolute() else target.resolve())
    for key in (*_FILES, "output"):
        config[key] = resolve(config[key])
    config["dynamical_xml"] = [resolve(item) for item in config["dynamical_xml"]]
    if not config["dynamical_xml"]:
        raise ValueError("dynamical_xml must list the PH files in qe2pert reading order")
    for key in ("fermi_energy_eV", "energy_min_eV", "eta_eV"):
        if not np.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if config["eta_eV"] <= 0 or config["energy_min_eV"] >= config["fermi_energy_eV"]:
        raise ValueError("require eta>0 and energy_min<fermi_energy")
    for key, default in (("projector_tr_tolerance", .05), ("g_reciprocity_tolerance", 1.e-5), ("hamiltonian_tolerance_eV", 1.e-5)):
        config.setdefault(key, default)
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    config.setdefault("perturbation_chunk", 3)
    config.setdefault("g_pair_policy", "raw")
    if config["g_pair_policy"] not in {"raw", "hermitian_pair_average"}:
        raise ValueError("g_pair_policy must be raw or hermitian_pair_average")
    if not isinstance(config["perturbation_chunk"], int) or config["perturbation_chunk"] < 1:
        raise ValueError("perturbation_chunk must be a positive integer")
    qpoints = np.asarray(config["qpoints"], float)
    if qpoints.ndim != 2 or qpoints.shape[1] != 3 or not qpoints.size or not np.all(np.isfinite(qpoints)):
        raise ValueError("qpoints must have finite shape (nq,3)")
    # Use explicit signed representatives. In atomic gauge q+G carries site
    # phases, so modulo matching alone is insufficient for coefficient pairs.
    for q in qpoints:
        if np.sum(np.max(np.abs(qpoints + q), axis=1) < 1.e-9) != 1:
            raise ValueError("qpoints must contain each exact -q representative once")
    return config


def _hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_inputs(config: dict[str, Any]) -> dict[str, Any]:
    target = Path(config["output"])
    if target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite native workflow results: {target}")
    q = np.asarray(config["qpoints"], float)
    with h5py.File(config["epr"], "r") as handle:
        meta = read_epr_metadata(handle)
    basis = load_native_spinor_basis(
        config["epr"], win_path=config["win"], amn_path=config["amn"], eig_path=config["eig"],
        u_matrix_path=config["u_mat"], u_dis_matrix_path=config["u_dis_mat"],
        u_dis_layout=config["u_dis_layout"], projection_indices=np.arange(meta.nwan),
    )
    native_h = reconstruct_epr_hamiltonian(config["epr"], energy_unit=config["ep_energy_unit"], expected_spinor=True)
    h_error = float(np.max(np.abs(native_h.values - basis.hamiltonian_eV)))
    if h_error > config["hamiltonian_tolerance_eV"]:
        raise ValueError(f"EPR and Wannier side files have different H(k): {h_error:.6g} eV")
    grids = read_native_coarse_grids(config["qe_xml"], config["dynamical_xml"])
    if grids.qpoints.shape[0] != np.prod(meta.nq_grid) or grids.kpoints.shape[0] != np.prod(meta.nk_grid):
        raise ValueError("native XML meshes disagree with EPR metadata")
    k_order = periodic_grid_indices(grids.kpoints, basis.kpoints)
    q_order = periodic_grid_indices(grids.qpoints, q)
    for point in q:
        build_kq_map(basis.kpoints, point)
    magnons = native_magnon_modes(
        config["exchange_out"], q, spin_lengths=config["spin_lengths"],
        source_directed_bond_weight=config["source_directed_bond_weight"],
        magnetic_atom_labels=config["magnetic_atom_labels"], gauge="atomic",
    )
    phonons = native_phonon_modes(config["epr"], q, gauge="atomic")
    validate_native_xml_geometry(config["qe_xml"], config["dynamical_xml"],
                                lattice_ang=phonons.lattice_ang, atom_positions_frac=phonons.atom_positions_frac)
    phonon_reference = validate_phonons_against_qe_dyn_xml(phonons, config["dynamical_xml"])
    if not np.allclose(magnons.lattice_ang, phonons.lattice_ang, atol=1.e-6, rtol=0):
        raise ValueError("TB2J and EPR lattices differ")
    site_atoms = periodic_grid_indices(meta.tau, magnons.magnetic_site_positions)
    # Use EPR representatives as the common field origins. A different TB2J
    # representative would require rephasing its magnon coordinate rows.
    if not np.allclose(meta.tau[site_atoms], magnons.magnetic_site_positions, atol=1.e-7, rtol=0):
        raise ValueError("TB2J/EPR magnetic site cell representatives differ; align input cells first")
    projection = read_spinor_projection_metadata(config["nnkp"], atomic_spin_order=config["atomic_spin_order"])
    groups = []
    for position in magnons.magnetic_site_positions:
        delta = projection.orbital_centers - position
        selected = np.max(np.abs(delta - np.rint(delta)), axis=1) < 1.e-7
        groups.append(projection.projection_pairs[selected].ravel())
    frame = build_native_spinor_frame(
        basis.hamiltonian_eV, basis.selected_overlap, basis.minus_k_index,
        kpoints=basis.kpoints, orbital_centers=projection.orbital_centers,
        orbital_masks=projection.masks_from_projection_groups(groups),
        magnetic_site_positions=meta.tau[site_atoms], local_frames=magnons.local_frames,
        atomic_spin_order=config["atomic_spin_order"],
        projector_tr_tolerance=config["projector_tr_tolerance"],
    )
    paths = {key: config[key] for key in _FILES}
    paths.update({f"dynamical_xml_{i}": value for i, value in enumerate(config["dynamical_xml"])})
    eigenvalues = np.linalg.eigvalsh(basis.hamiltonian_eV)
    mu = config["fermi_energy_eV"]
    summary = {
        "approximation": "projection-anchored rigid atomic spin; TR-odd model exchange; local_partition; " + ("bubble + fixed-frame direct; " if config["include_direct_vertex"] else "bubble only; ") + "isotropic TB2J magnons",
        "direct_term_enabled": config["include_direct_vertex"], "projector_motion_enabled": False,
        "g_xc_definition": ("d[H-Theta H Theta^-1]/(2 du), fixed AMN Q/B; model_exchange_derivative, not isolated QE XC potential" if config["include_direct_vertex"] else None),
        "temperature_K": 0., "fixed_chemical_potential": True,
        "spin_coordinate": "transverse_direction", "bloch_gauge": "atomic",
        "ep_input": "full coarse Cartesian derivative (before polar subtraction)",
        "kernel_units": "eV/angstrom", "mode_vertex_units": "eV",
        "kmesh": list(meta.nk_grid), "qmesh": list(meta.nq_grid), "num_wann": meta.nwan,
        "nmag": len(site_atoms), "natom": meta.nat,
        "hamiltonian_side_file_max_residual_eV": h_error,
        "wannier_u_isometry_residual": basis.u_isometry_residual,
        "input_k_rounding_residual": basis.kpoint_residual,
        "occupied_band_counts": sorted(set(np.sum(eigenvalues < mu, axis=1).tolist())),
        "valence_max_eV": float(eigenvalues[eigenvalues < mu].max()),
        "conduction_min_eV": float(eigenvalues[eigenvalues > mu].min()),
        "frame": frame.diagnostics, "phonon": phonons.diagnostics, "magnon": magnons.diagnostics,
        "phonon_qe_reference": phonon_reference,
        "nominal_projector_tr_tolerance": .05,
        "nominal_projector_tr_gate_passed": frame.diagnostics["projector_tr_covariance_max_relative"] <= .05,
        "source_files": paths, "source_sha256": {key: _hash(value) for key, value in paths.items()},
        "implementation_sha256": {
            str(source.relative_to(Path(__file__).parents[1])): _hash(str(source))
            for source in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "resolved_config": config,
    }
    return dict(meta=meta, frame=frame, grids=grids, k_order=k_order, q_order=q_order,
                magnons=magnons, phonons=phonons, summary=summary)


def _one_loop(index: int, config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    meta, frame = inputs["meta"], inputs["frame"]
    q = np.asarray(config["qpoints"][index], float)
    mapping = build_kq_map(frame.kpoints, q)
    def read(qindex: int) -> np.ndarray:
        return read_native_coarse_vertex(
            config["elph"], q_index=qindex, k_indices=inputs["k_order"],
            nat=meta.nat, num_wann=meta.nwan, nq=len(inputs["grids"].qpoints), nk=len(frame.kpoints),
            energy_unit=config["ep_energy_unit"], displacement_unit=config["ep_displacement_unit"],
        )
    g = read(int(inputs["q_order"][index]))
    minus_index = int(periodic_grid_indices(inputs["grids"].qpoints, [-q])[0])
    gm = read(minus_index)
    residual = float(np.linalg.norm(g - gm[mapping.indices].conj().swapaxes(-1, -2)) / max(np.linalg.norm(g), np.finfo(float).tiny))
    if residual > config["g_reciprocity_tolerance"]:
        raise ValueError(f"native g Hermitian-pair residual at q={q.tolist()} is {residual:.6g}")
    if config["g_pair_policy"] == "hermitian_pair_average":
        averaged = .5 * (g + gm[mapping.indices].conj().swapaxes(-1, -2))
        gm = .5 * (gm + g[build_kq_map(frame.kpoints, -q).indices].conj().swapaxes(-1, -2))
        g = averaged
    forward_g = frame.transform_vertex(g, q_red=q, perturbation_positions=np.repeat(meta.tau, 3, axis=0))
    forward_torque = frame.finite_q_vertices(q)
    reverse_torque = forward_torque.conj().swapaxes(-1, -2).reshape(len(frame.kpoints), -1, meta.nwan, meta.nwan)
    loop = retarded_bubble_loop_eigh_zero_temperature_finite_q(
        frame.hamiltonian_eV, frame.hamiltonian_at_indices(mapping.indices, mapping.G_wrap),
        reverse_torque, forward_g, np.full(len(frame.kpoints), 1. / len(frame.kpoints)),
        energy_min_eV=config["energy_min_eV"], occupied_energy_max_eV=config["fermi_energy_eV"],
        eta_eV=config["eta_eV"], perturbation_chunk=config["perturbation_chunk"],
    )
    result = {"loop": loop, "g_reciprocity_relative": residual, "g_rms_eV_per_angstrom": float(np.sqrt(np.mean(abs(g)**2)))}
    if config["include_direct_vertex"]:
        gxc_w = frame.model_exchange_derivative_wannier(g, gm, q_red=q)
        gmxc_w = frame.model_exchange_derivative_wannier(gm, g, q_red=-q)
        gxc = frame.transform_vertex(gxc_w, q_red=q, perturbation_positions=np.repeat(meta.tau, 3, axis=0))
        direct = np.empty_like(loop)
        chunk = config["perturbation_chunk"]
        for start in range(0, gxc.shape[1], chunk):
            stop = min(start + chunk, gxc.shape[1])
            vertex = finite_q_direct_vertices(
                gxc[:, start:stop], kpoints=frame.kpoints, q_red=q,
                orbital_masks=frame.orbital_masks, local_frames=frame.local_frames,
                orbital_centers=frame.orbital_centers,
                magnetic_site_positions=frame.magnetic_site_positions,
            )
            direct[:, start:stop] = retarded_direct_loop_eigh_zero_temperature(
                frame.hamiltonian_eV, vertex.reshape(len(frame.kpoints), -1, stop-start, meta.nwan, meta.nwan),
                np.full(len(frame.kpoints), 1. / len(frame.kpoints)),
                energy_min_eV=config["energy_min_eV"], occupied_energy_max_eV=config["fermi_energy_eV"],
                eta_eV=config["eta_eV"], perturbation_chunk=chunk,
            )
        result["direct_loop"] = direct
        result["g_xc_diagnostics"] = {
            "relative_norm_to_total_g": float(np.linalg.norm(gxc_w)/max(np.linalg.norm(g), np.finfo(float).tiny)),
            "hermitian_partner_relative_residual": float(np.linalg.norm(gxc_w-gmxc_w[mapping.indices].conj().swapaxes(-1,-2))/max(np.linalg.norm(gxc_w), np.finfo(float).tiny)),
            "tr_odd_projection_idempotency_max": float(abs(frame.model_exchange_derivative_wannier(gxc_w,gmxc_w,q_red=q)-gxc_w).max()),
        }
        if config["export_g_xc"]:
            result["g_xc_cart"] = gxc
    return result


def _write_result(config: dict[str, Any], inputs: dict[str, Any], loops: dict[int, dict[str, Any]], mpi_size: int) -> dict[str, Any]:
    points = np.asarray(config["qpoints"], float)
    phonons, magnons, meta = inputs["phonons"], inputs["magnons"], inputs["meta"]
    nmag = len(magnons.spin_lengths)
    raw = np.array([loops[i]["loop"] for i in range(len(points))])
    partners = np.argmin(np.max(abs(points[:, None] + points[None]), axis=2), axis=1)
    bubble = np.array([finalize_q_pair(raw[i], raw[j]) for i, j in enumerate(partners)]).reshape(len(points), nmag, 2, meta.nat, 3)
    raw_direct = np.array([loops[i].get("direct_loop", np.zeros_like(raw[i])) for i in range(len(points))])
    direct = np.array([finalize_q_pair(raw_direct[i], raw_direct[j]) for i, j in enumerate(partners)]).reshape(bubble.shape)
    kernel = bubble + direct
    phonon_vertex, normal, anomalous, nambu, rwa = [], [], [], [], []
    for i in range(len(points)):
        value = project_phonons(kernel[i], normalization="cartesian_derivative", frequencies=phonons.energies_eV[i], masses=phonons.masses_amu, eigenvectors=phonons.eigenvectors[i])
        projected = project_external_magnons(value, magnons.transform[i], magnons.spin_lengths)
        phonon_vertex.append(value)
        normal.append(projected.normal)
        anomalous.append(projected.anomalous)
        nambu.append(projected.full_nambu)
        rwa.append(np.linalg.eigvalsh(assemble_rwa(magnons.energies_eV[i], phonons.energies_eV[i], projected.normal)))
    data = {
        "qpoints": points, "kernel/A_retarded": raw + raw_direct, "kernel/K_pi_u": kernel,
        "kernel/A_retarded_bubble": raw, "kernel/A_retarded_direct": raw_direct,
        "kernel/A_retarded_total": raw + raw_direct,
        "kernel/K_pi_u_bubble": bubble, "kernel/K_pi_u_direct": direct, "kernel/K_pi_u_total": kernel,
        "phonons/energies_eV": phonons.energies_eV, "phonons/eigenvectors": phonons.eigenvectors,
        "phonons/masses_amu": phonons.masses_amu, "magnons/energies_eV": magnons.energies_eV,
        "magnons/T_para": magnons.transform, "magnons/local_frames": magnons.local_frames,
        "magnons/spin_lengths": magnons.spin_lengths, "magnons/site_positions": magnons.magnetic_site_positions,
        "coupling/V_pi_ph": np.array(phonon_vertex), "coupling/g_mp_normal": np.array(normal),
        "coupling/g_mp_anomalous": np.array(anomalous), "coupling/full_nambu": np.array(nambu),
        "bands/rwa_energies_eV": np.array(rwa),
    }
    # Keep component mode coefficients in the same magnon/phonon gauges, so
    # cancellation can be inspected without comparing independently phased modes.
    for name, component in (("bubble", bubble), ("direct", direct)):
        values = [project_phonons(component[i], normalization="cartesian_derivative", frequencies=phonons.energies_eV[i], masses=phonons.masses_amu, eigenvectors=phonons.eigenvectors[i]) for i in range(len(points))]
        projected = [project_external_magnons(value, magnons.transform[i], magnons.spin_lengths) for i,value in enumerate(values)]
        data[f"coupling/V_pi_ph_{name}"] = np.array(values)
        data[f"coupling/g_mp_normal_{name}"] = np.array([item.normal for item in projected])
        data[f"coupling/g_mp_anomalous_{name}"] = np.array([item.anomalous for item in projected])
    if config.get("export_g_xc", False):
        data["dfpt/qpoints"] = points
        data["dfpt/kpoints"] = inputs["frame"].kpoints
        data["dfpt/pert_atom"] = np.repeat(np.arange(meta.nat), 3)
        data["dfpt/pert_cart"] = np.tile(np.arange(3), meta.nat)
        for i in range(len(points)):
            data[f"dfpt/q_{i:06d}/g_xc_cart"] = loops[i]["g_xc_cart"]
    if any(not np.all(np.isfinite(value)) for value in data.values()):
        raise ValueError("non-finite native workflow result")
    summary = inputs["summary"] | {
        "status": "complete", "mpi_size": mpi_size, "qpoints": points.tolist(),
        "g_reciprocity_max_relative": max(item["g_reciprocity_relative"] for item in loops.values()),
        "nominal_g_reciprocity_tolerance": 1.e-5,
        "nominal_g_reciprocity_gate_passed": max(item["g_reciprocity_relative"] for item in loops.values()) <= 1.e-5,
        "q_pair_kernel_residual_eV_per_angstrom": float(np.max(abs(kernel[partners] - kernel.conj()))),
        "kernel_max_eV_per_angstrom": float(np.max(abs(kernel))),
        "normal_vertex_max_meV": float(np.max(np.abs(normal)) * 1.e3),
        "normal_vertex_frobenius_meV_per_q": (np.linalg.norm(np.array(normal), axis=(1,2)) * 1.e3).tolist(),
        "bubble_normal_vertex_max_meV": float(abs(data["coupling/g_mp_normal_bubble"]).max()*1.e3),
        "direct_normal_vertex_max_meV": float(abs(data["coupling/g_mp_normal_direct"]).max()*1.e3),
        "total_to_bubble_normal_frobenius_ratio": float(np.linalg.norm(normal)/max(np.linalg.norm(data["coupling/g_mp_normal_bubble"]),np.finfo(float).tiny)),
        "g_xc_diagnostics_per_q": [loops[i].get("g_xc_diagnostics") for i in range(len(points))],
        "phonon_energy_range_meV": [float(phonons.energies_eV.min() * 1.e3), float(phonons.energies_eV.max() * 1.e3)],
        "magnon_energy_range_meV": [float(magnons.energies_eV.min() * 1.e3), float(magnons.energies_eV.max() * 1.e3)],
        "validation_scope": "Numerical algebra and input consistency; mesh/eta and physical-spin projection convergence require separate comparisons. q-pair kernel residual is enforced, not an independent DFPT check; g reciprocity is independent.",
    }
    target = Path(config["output"])
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = "slw.wtorque.native_mixed.v2"
            handle.attrs["direct_term_enabled"] = config.get("include_direct_vertex", False)
            handle.attrs["bloch_gauge"] = "atomic"
            handle.create_dataset("metadata/summary_json", data=json.dumps(summary, sort_keys=True))
            for key, value in data.items():
                compression = dict(compression="gzip", compression_opts=1) if key.endswith("/g_xc_cart") else {}
                handle.create_dataset(key, data=value, **compression)
            if "dfpt" in handle:
                handle["dfpt"].attrs.update(normalization="cartesian_derivative", bloch_gauge="atomic", spin_order="interleaved",
                                          derivative_kind="fixed_frame_TR_odd_model_exchange", units="eV/angstrom",
                                          basis="AMN_polar_atomic; use exported atomic_frame and orbital_centers, not original MLWF basis")
                handle["dfpt"].create_dataset("atomic_frame", data=inputs["frame"].atomic_frame)
                handle["dfpt"].create_dataset("orbital_centers", data=inputs["frame"].orbital_centers)
        # Atomic no-clobber publication on the same filesystem.
        os.link(temporary, target)
        with target.with_suffix(".json").open("x") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
    finally:
        os.unlink(temporary)
    return summary


def run_native_workflow(path: str | Path, *, mpi: MPIContext | None = None) -> dict[str, Any] | None:
    """Run independent q tasks with MPI and collective error propagation."""
    context = MPIContext.discover() if mpi is None else mpi
    config = inputs = None
    error = None
    if context.is_root:
        try:
            config = load_native_workflow_config(path)
            print("native wtorque: 0% — checking spinor inputs and modes", flush=True)
            inputs = _root_inputs(config)
            print("native wtorque: 25% — inputs checked; computing q/-q bubbles", flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = context.bcast(error)
    if error:
        raise RuntimeError(f"native workflow initialization failed: {error}")
    config, inputs = context.bcast((config, inputs))
    local = {}
    error = None
    try:
        for index in range(context.rank, len(config["qpoints"]), context.size):
            local[index] = _one_loop(index, config, inputs)
            print(f"native wtorque rank {context.rank}: q {index+1}/{len(config['qpoints'])} complete", flush=True)
    except Exception as exc:
        error = f"rank {context.rank}: {type(exc).__name__}: {exc}"
    errors = [value for value in context.allgather(error) if value]
    if errors:
        raise RuntimeError("native q computation failed: " + "; ".join(errors))
    gathered = context.gather(local)
    result = None
    error = None
    if context.is_root:
        try:
            combined = {index: item for rank_data in gathered for index, item in rank_data.items()}
            print("native wtorque: 75% — projecting phonon and magnon modes", flush=True)
            result = _write_result(config, inputs, combined, context.size)
            print(f"native wtorque: 100% — {config['output']}", flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = context.bcast(error)
    if error:
        raise RuntimeError(f"native result assembly failed: {error}")
    return result
