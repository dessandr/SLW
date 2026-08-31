"""Stable command-line interface for the restartable wtorque workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from slw.wtorque.collinear_q0 import run_collinear_epr_q0_test
from slw.wtorque.config import RunConfig
from slw.wtorque.gauge.kq_map import build_kq_map
from slw.wtorque.io.dfpt import HDF5DFPTProvider
from slw.wtorque.io.exchange import exchange_at_k_and_kq, load_exchange_field
from slw.wtorque.io.hdf5 import RestartableHDF5
from slw.wtorque.io.native_epr import audit_native_spinor_epr
from slw.wtorque.io.spin import load_magnetic_data
from slw.wtorque.io.wannier import load_spinor_wannier
from slw.wtorque.native_q0 import run_native_q0_test
from slw.wtorque.parallel.benchmark import benchmark_q
from slw.wtorque.parallel.mpi import MPIContext
from slw.wtorque.parallel.scheduler import build_q_pair_schedule
from slw.wtorque.pipeline import run_compute_kernel
from slw.wtorque.provenance import build_run_manifest
from slw.wtorque.q0_polaron import run_collinear_epr_q0_polaron_test
from slw.wtorque.stages import (
    run_assemble_polaron,
    run_project_magnons,
    run_project_phonons,
)
from slw.wtorque.torque.vertices import finite_q_vertices
from slw.wtorque.validation.reports import validate_run


def _config(path: str) -> RunConfig:
    return RunConfig.load(path)


def inspect_config(config: RunConfig) -> dict[str, Any]:
    model = load_spinor_wannier(
        config.electrons.file,
        bloch_gauge=config.electrons.bloch_gauge,
        spin_order=config.electrons.spin_order,
        onsite_soc=config.electrons.onsite_soc,
    )
    exchange = load_exchange_field(
        config.electrons.file,
        model,
        route=config.electrons.exchange_extraction,
        spin_order=config.electrons.spin_order,
    )
    magnetic = load_magnetic_data(
        config.electrons.file,
        config.magnetic_subspace,
        norb=model.norb,
        orbital_labels=model.orbital_labels,
    )
    with HDF5DFPTProvider(
        config.dfpt.file,
        normalization=config.dfpt.normalization,
        spinor_lift=config.dfpt.spinor_lift,
        spin_order=config.electrons.spin_order,
        norb=model.norb,
    ) as dfpt:
        q_pairs = len(build_q_pair_schedule(dfpt.qpoints))
        dimensions = {
            "norb": model.norb,
            "nw": model.nw,
            "nk": int(model.kpoints.shape[0]),
            "nq": int(dfpt.qpoints.shape[0]),
            "nq_pairs": q_pairs,
            "nmag": int(magnetic.subspace.projectors.shape[0]),
            "npert_q0": int(dfpt.g(0).shape[1]),
        }
    manifest = build_run_manifest(config)
    return {
        "valid": True,
        "dimensions": dimensions,
        "conventions": {
            "bloch_gauge": config.electrons.bloch_gauge.value,
            "spin_order": config.electrons.spin_order.value,
            "exchange_extraction": config.electrons.exchange_extraction.value,
            "site_projection": config.magnetic_subspace.site_projection.value,
            "spin_coordinate": config.magnetic_subspace.spin_coordinate.value,
            "dfpt_normalization": config.dfpt.normalization.value,
            "dfpt_final_state": config.dfpt.final_state_representation,
            "fixed_chemical_potential": config.kernel.fixed_chemical_potential,
            "q_pair_completion": config.kernel.q_pair_completion,
        },
        "resolved_magnetic_orbitals": [
            list(labels) for labels in magnetic.subspace.labels
        ],
        "exchange_reconstruction_residual": exchange.reconstruction_residual,
        "onsite_soc": {
            "enabled": config.electrons.onsite_soc is not None,
            "matrix_norm_eV": (
                0.0
                if model.onsite_soc_matrix is None
                else float(np.linalg.norm(model.onsite_soc_matrix))
            ),
            "selectors": (
                []
                if config.electrons.onsite_soc is None
                else [
                    item.selector for item in config.electrons.onsite_soc.spec.manifolds
                ]
            ),
            "scale": (
                0.0
                if config.electrons.onsite_soc is None
                else config.electrons.onsite_soc.scale
            ),
        },
        "source_hashes": manifest["source_hashes"],
        "source_provenance_ids": manifest["source_provenance_ids"],
    }


def _artifact_path(config: RunConfig, suffix: str, explicit: str | None) -> Path:
    return (
        Path(explicit).resolve() if explicit else config.output.file.with_suffix(suffix)
    )


def _extract_exchange(config: RunConfig, output: str | None) -> Path:
    model = load_spinor_wannier(
        config.electrons.file,
        bloch_gauge=config.electrons.bloch_gauge,
        spin_order=config.electrons.spin_order,
        onsite_soc=config.electrons.onsite_soc,
    )
    exchange = load_exchange_field(
        config.electrons.file,
        model,
        route=config.electrons.exchange_extraction,
        spin_order=config.electrons.spin_order,
    )
    target = _artifact_path(config, ".exchange.h5", output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    with h5py.File(target, "w") as handle:
        group = handle.create_group("exchange")
        group.attrs["route"] = exchange.route.value
        group.attrs["reconstruction_residual"] = exchange.reconstruction_residual
        for name in ("h_trs_r", "h_xc_r", "h_trs_k", "h_xc_k"):
            value = getattr(exchange, name)
            if value is not None:
                group.create_dataset(name.upper(), data=value)
    return target


def _build_vertices(config: RunConfig, output: str | None) -> Path:
    model = load_spinor_wannier(
        config.electrons.file,
        bloch_gauge=config.electrons.bloch_gauge,
        spin_order=config.electrons.spin_order,
        onsite_soc=config.electrons.onsite_soc,
    )
    exchange = load_exchange_field(
        config.electrons.file,
        model,
        route=config.electrons.exchange_extraction,
        spin_order=config.electrons.spin_order,
    )
    magnetic = load_magnetic_data(
        config.electrons.file,
        config.magnetic_subspace,
        norb=model.norb,
        orbital_labels=model.orbital_labels,
    )
    target = _artifact_path(config, ".vertices.h5", output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    with (
        HDF5DFPTProvider(
            config.dfpt.file,
            normalization=config.dfpt.normalization,
            spinor_lift=config.dfpt.spinor_lift,
            spin_order=config.electrons.spin_order,
            norb=model.norb,
        ) as dfpt,
        h5py.File(target, "w") as handle,
    ):
        handle.create_dataset("qpoints", data=dfpt.qpoints)
        handle.attrs["coordinate_type"] = config.magnetic_subspace.spin_coordinate.value
        handle.attrs["resolved_orbitals_json"] = json.dumps(magnetic.subspace.labels)
        for index, q in enumerate(dfpt.qpoints):
            mapping = build_kq_map(model.kpoints, q)
            hxc_k, hxc_kq = exchange_at_k_and_kq(exchange, model, q, mapping)
            vertices = finite_q_vertices(
                hxc_k,
                hxc_kq,
                orbital_masks=magnetic.subspace.orbital_masks,
                local_frames=magnetic.local_frames,
                q_red=q,
                orbital_centers=model.orbital_centers,
                magnetic_site_positions=magnetic.site_positions,
                coordinate_type=config.magnetic_subspace.spin_coordinate.value,
            )
            handle.create_dataset(f"q_{index:06d}/vertex", data=vertices)
    return target


def merge_outputs(directory: str | Path, output: str | Path | None = None) -> Path:
    source_dir = Path(directory).resolve()
    sources = sorted(path for path in source_dir.glob("*.h5") if path.is_file())
    if not sources:
        raise FileNotFoundError(f"no HDF5 shards found in {source_dir}")
    target = Path(output).resolve() if output else source_dir / "wtorque-merged.h5"
    sources = [path for path in sources if path != target]
    reference_manifest: dict[str, Any] | None = None
    qpoints: np.ndarray | None = None
    payloads: dict[int, dict[str, object]] = {}
    for source in sources:
        with h5py.File(source, "r") as handle:
            if "meta/run_manifest_json" not in handle or "q_data" not in handle:
                continue
            raw = handle["meta/run_manifest_json"][()]
            manifest = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            mesh = np.asarray(handle["qpoints"][...], dtype=np.float64)
        if reference_manifest is None:
            reference_manifest = manifest
            qpoints = mesh
        elif (
            manifest != reference_manifest
            or qpoints is None
            or mesh.shape != qpoints.shape
            or not np.allclose(mesh, qpoints)
        ):
            raise ValueError(f"shard manifest or q mesh mismatch: {source}")
        with RestartableHDF5(source, mesh, manifest, resume=True) as reader:
            for index in range(mesh.shape[0]):
                if reader.status(index) != "complete":
                    continue
                reader.verify_complete_q(index)
                group = reader.handle[f"q_data/q_{index:06d}"]
                current: dict[str, object] = {}

                def collect(
                    name: str,
                    value: h5py.Dataset | h5py.Group,
                    target: dict[str, object] = current,
                ) -> None:
                    if isinstance(value, h5py.Dataset):
                        target[name] = value[...]

                group.visititems(collect)
                if index in payloads:
                    if current.keys() != payloads[index].keys() or any(
                        not np.array_equal(
                            np.asarray(current[key]), np.asarray(payloads[index][key])
                        )
                        for key in current
                    ):
                        raise ValueError(
                            f"conflicting complete q index {index} in {source}"
                        )
                else:
                    payloads[index] = current
    if reference_manifest is None or qpoints is None:
        raise ValueError("no restart-compatible wtorque shards were found")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite merged output: {target}")
    with RestartableHDF5(target, qpoints, reference_manifest, resume=False) as writer:
        for index in sorted(payloads):
            writer.write_q(index, payloads[index])
        if len(payloads) != qpoints.shape[0]:
            missing = sorted(set(range(qpoints.shape[0])) - payloads.keys())
            raise ValueError(
                f"merged shards remain incomplete; missing q indices {missing}"
            )
        writer.materialize()
    return target


def _add_collinear_q0_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--up-epr", required=True)
    parser.add_argument("--down-epr", required=True)
    parser.add_argument(
        "--magnetic-orbitals",
        required=True,
        help=(
            "one-based orbital groups separated by semicolons, for example "
            "'1-5;6-10'"
        ),
    )
    parser.add_argument(
        "--magnetic-atoms",
        required=True,
        help="one-based atom indices paired with the orbital groups, e.g. 1,2",
    )
    parser.add_argument(
        "--local-direction",
        required=True,
        action="append",
        nargs=3,
        type=float,
        metavar=("MX", "MY", "MZ"),
        help="repeat once for each magnetic site",
    )
    parser.add_argument(
        "--spin-quantization-direction",
        required=True,
        nargs=3,
        type=float,
        metavar=("SX", "SY", "SZ"),
    )
    parser.add_argument(
        "--spin-coordinate",
        required=True,
        choices=("rotation_angle", "transverse_direction"),
    )
    parser.add_argument(
        "--onsite-soc",
        action="append",
        default=[],
        metavar="ORBITAL:INDICES:LAMBDA_EV",
        help=(
            "repeat for each explicit p/d manifold, for example "
            "p:11-13:0.5"
        ),
    )
    parser.add_argument(
        "--soc-p-order",
        help="explicit p order, for example pz,px,py",
    )
    parser.add_argument(
        "--soc-d-order",
        help="explicit d order, for example dz2,dxz,dyz,dx2-y2,dxy",
    )
    parser.add_argument("--fermi-energy-ev", required=True, type=float)
    parser.add_argument("--energy-min-ev", required=True, type=float)
    parser.add_argument("--energy-max-ev", required=True, type=float)
    parser.add_argument("--energy-points", required=True, type=int)
    parser.add_argument(
        "--integration-backend",
        choices=("gauss_legendre", "analytic_zero_temperature"),
        default="gauss_legendre",
    )
    parser.add_argument("--eta-ev", required=True, type=float)
    parser.add_argument("--temperature-k", required=True, type=float)
    parser.add_argument(
        "--epr-energy-unit",
        required=True,
        choices=("ev", "ry", "ha"),
    )
    parser.add_argument(
        "--epr-displacement-unit",
        required=True,
        choices=("angstrom", "bohr"),
    )
    parser.add_argument("--divide-by-ws-degeneracy", action="store_true")
    parser.add_argument("--perturbation-chunk", type=int)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wtorque")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "inspect",
        "extract-exchange",
        "build-vertices",
        "compute-kernel",
        "project-phonons",
        "project-magnons",
        "assemble-polaron",
        "validate",
    ):
        item = sub.add_parser(command)
        item.add_argument("config")
        if command in {"extract-exchange", "build-vertices"}:
            item.add_argument("--output")
    validate = sub.choices["validate"]
    validate.add_argument("--report")
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("config")
    benchmark.add_argument("--q-index", required=True, type=int)
    benchmark.add_argument("--repeats", type=int, default=3)
    merge = sub.add_parser("merge")
    merge.add_argument("output_dir")
    merge.add_argument("--output")
    native = sub.add_parser(
        "inspect-native-epr",
        help="audit a qe2pert native-spinor EPR/Wannier basis before conversion",
    )
    native.add_argument("--epr", required=True)
    native.add_argument("--win", required=True)
    native.add_argument("--amn", required=True)
    native.add_argument("--eig", required=True)
    native.add_argument("--spn", required=True)
    native.add_argument("--hr")
    native.add_argument("--u-mat", required=True)
    native.add_argument("--u-dis-mat", required=True)
    native.add_argument(
        "--u-dis-layout",
        required=True,
        choices=("global_bands", "compact_outer_window"),
    )
    native.add_argument(
        "--atomic-spin-order",
        required=True,
        choices=("interleaved", "blocked"),
    )
    native.add_argument(
        "--magnetization-direction",
        required=True,
        nargs=3,
        type=float,
        metavar=("MX", "MY", "MZ"),
    )
    native.add_argument("--algebra-tolerance", type=float, default=1.0e-6)
    native.add_argument("--hamiltonian-tolerance-ev", type=float, default=1.0e-4)
    native.add_argument("--projection-rank-tolerance", type=float, default=1.0e-4)
    native.add_argument("--projector-tr-tolerance", type=float, default=5.0e-2)
    native.add_argument("--longitudinal-torque-tolerance", type=float, default=5.0e-2)
    native.add_argument(
        "--strict",
        action="store_true",
        help="return exit status 2 when production validation gates fail",
    )
    native_q0 = sub.add_parser(
        "test-native-epr-q0",
        help=(
            "test full-spinor Green functions with an input-selected magnetic "
            "projection subspace at q=0"
        ),
    )
    native_q0.add_argument("--epr", required=True)
    native_q0.add_argument("--win", required=True)
    native_q0.add_argument("--amn", required=True)
    native_q0.add_argument("--eig", required=True)
    native_q0.add_argument("--u-mat", required=True)
    native_q0.add_argument("--u-dis-mat", required=True)
    native_q0.add_argument(
        "--u-dis-layout",
        required=True,
        choices=("global_bands", "compact_outer_window"),
    )
    native_q0.add_argument(
        "--magnetic-projections",
        required=True,
        help="one-based AMN columns, for example 9-18 or 9,10,13-18",
    )
    native_q0.add_argument(
        "--atomic-spin-order",
        required=True,
        choices=("interleaved", "blocked"),
    )
    native_q0.add_argument(
        "--magnetization-direction",
        required=True,
        nargs=3,
        type=float,
        metavar=("MX", "MY", "MZ"),
    )
    native_q0.add_argument("--fermi-energy-ev", required=True, type=float)
    native_q0.add_argument("--energy-min-ev", required=True, type=float)
    native_q0.add_argument("--energy-max-ev", required=True, type=float)
    native_q0.add_argument("--energy-points", required=True, type=int)
    native_q0.add_argument("--eta-ev", required=True, type=float)
    native_q0.add_argument("--temperature-k", required=True, type=float)
    native_q0.add_argument(
        "--ep-energy-unit",
        required=True,
        choices=("ev", "ry", "ha"),
    )
    native_q0.add_argument(
        "--ep-displacement-unit",
        required=True,
        choices=("angstrom", "bohr"),
    )
    native_q0.add_argument("--divide-by-ws-degeneracy", action="store_true")
    native_q0.add_argument("--finite-difference-step", type=float, default=1.0e-6)
    native_q0.add_argument("--perturbation-chunk", type=int)
    collinear_q0 = sub.add_parser(
        "test-collinear-epr-q0",
        help=(
            "test the no-SOC torque null condition from paired scalar "
            "spin-up/down qe2pert EPR files"
        ),
    )
    _add_collinear_q0_arguments(collinear_q0)
    polaron_q0 = sub.add_parser(
        "test-collinear-epr-q0-polaron",
        help=(
            "project the paired-scalar EPR Gamma torque into input J, magnon, "
            "and phonon modes"
        ),
    )
    _add_collinear_q0_arguments(polaron_q0)
    polaron_q0.add_argument("--exchange-h5", required=True)
    polaron_q0.add_argument(
        "--exchange-source-directed-bond-weight",
        required=True,
        type=float,
    )
    polaron_q0.add_argument(
        "--exchange-spin-normalization",
        required=True,
        choices=("unit_vector", "spin_operator"),
    )
    polaron_q0.add_argument(
        "--spin-length",
        required=True,
        nargs="+",
        type=float,
        help="one value to broadcast or one value per magnetic site",
    )
    polaron_q0.add_argument(
        "--anisotropy-mev",
        required=True,
        nargs="+",
        type=float,
        help="easy-axis SIA: one value to broadcast or one per magnetic site",
    )
    polaron_q0.add_argument(
        "--anisotropy-spin-normalization",
        required=True,
        choices=("unit_vector", "spin_operator"),
    )
    polaron_q0.add_argument(
        "--phonon-min-energy-mev", type=float, default=1.0e-6
    )
    polaron_q0.add_argument(
        "--phonon-imaginary-tolerance-mev", type=float, default=1.0e-6
    )
    polaron_q0.add_argument(
        "--phonon-loto-mode",
        choices=("auto", "none", "2d", "3d"),
        default="auto",
    )
    polaron_q0.add_argument("--geometry-tolerance", type=float, default=1.0e-8)
    polaron_q0.add_argument(
        "--torque-imaginary-tolerance-ev-per-angstrom",
        type=float,
        default=1.0e-8,
    )
    return parser


def _parse_magnetic_atoms(specification: object) -> tuple[int, ...]:
    atom_tokens = tuple(
        token.strip() for token in str(specification).split(",")
    )
    if not atom_tokens or any(not token.isdigit() for token in atom_tokens):
        raise ValueError("--magnetic-atoms must be a comma-separated integer list")
    atoms = tuple(int(token) - 1 for token in atom_tokens)
    if any(atom < 0 for atom in atoms):
        raise ValueError("--magnetic-atoms uses positive one-based indices")
    return atoms


def _collinear_q0_kwargs(
    args: argparse.Namespace,
    mpi: MPIContext,
) -> dict[str, Any]:
    return {
        "up_epr_path": args.up_epr,
        "down_epr_path": args.down_epr,
        "magnetic_orbital_specification": args.magnetic_orbitals,
        "magnetic_atom_indices": _parse_magnetic_atoms(args.magnetic_atoms),
        "local_magnetization_directions": args.local_direction,
        "spin_quantization_direction": args.spin_quantization_direction,
        "spin_coordinate": args.spin_coordinate,
        "onsite_soc_specifications": tuple(args.onsite_soc),
        "p_orbital_order": args.soc_p_order,
        "d_orbital_order": args.soc_d_order,
        "fermi_energy_eV": args.fermi_energy_ev,
        "energy_min_eV": args.energy_min_ev,
        "energy_max_eV": args.energy_max_ev,
        "energy_points": args.energy_points,
        "integration_backend": args.integration_backend,
        "eta_eV": args.eta_ev,
        "temperature_K": args.temperature_k,
        "epr_energy_unit": args.epr_energy_unit,
        "epr_displacement_unit": args.epr_displacement_unit,
        "divide_by_ws_degeneracy": args.divide_by_ws_degeneracy,
        "perturbation_chunk": args.perturbation_chunk,
        "mpi": mpi,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mpi = MPIContext.discover()
    if args.command == "merge":
        if mpi.is_root:
            print(merge_outputs(args.output_dir, args.output))
        mpi.barrier()
        return 0
    if args.command == "inspect-native-epr":
        if mpi.is_root:
            audit = audit_native_spinor_epr(
                args.epr,
                win_path=args.win,
                amn_path=args.amn,
                eig_path=args.eig,
                spn_path=args.spn,
                hr_path=args.hr,
                u_matrix_path=args.u_mat,
                u_dis_matrix_path=args.u_dis_mat,
                u_dis_layout=args.u_dis_layout,
                atomic_spin_order=args.atomic_spin_order,
                magnetization_direction=args.magnetization_direction,
                algebra_tolerance=args.algebra_tolerance,
                hamiltonian_tolerance_eV=args.hamiltonian_tolerance_ev,
                projection_rank_tolerance=args.projection_rank_tolerance,
                projector_tr_tolerance=args.projector_tr_tolerance,
                longitudinal_torque_tolerance=(args.longitudinal_torque_tolerance),
            )
            print(json.dumps(audit.as_dict(), indent=2, sort_keys=True))
            exit_code = 0 if audit.production_ready or not args.strict else 2
        else:
            exit_code = 0
        exit_code = int(mpi.bcast(exit_code, root=0))
        mpi.barrier()
        return exit_code
    if args.command == "test-native-epr-q0":
        result = run_native_q0_test(
            epr_path=args.epr,
            win_path=args.win,
            amn_path=args.amn,
            eig_path=args.eig,
            u_matrix_path=args.u_mat,
            u_dis_matrix_path=args.u_dis_mat,
            u_dis_layout=args.u_dis_layout,
            magnetic_projection_specification=args.magnetic_projections,
            atomic_spin_order=args.atomic_spin_order,
            magnetization_direction=args.magnetization_direction,
            fermi_energy_eV=args.fermi_energy_ev,
            energy_min_eV=args.energy_min_ev,
            energy_max_eV=args.energy_max_ev,
            energy_points=args.energy_points,
            eta_eV=args.eta_ev,
            temperature_K=args.temperature_k,
            ep_energy_unit=args.ep_energy_unit,
            ep_displacement_unit=args.ep_displacement_unit,
            divide_by_ws_degeneracy=args.divide_by_ws_degeneracy,
            finite_difference_step=args.finite_difference_step,
            perturbation_chunk=args.perturbation_chunk,
            mpi=mpi,
        )
        if mpi.is_root:
            if result is None:
                raise RuntimeError("native q=0 root did not receive a result")
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "test-collinear-epr-q0":
        collinear_result = run_collinear_epr_q0_test(
            **_collinear_q0_kwargs(args, mpi)
        )
        if mpi.is_root:
            if collinear_result is None:
                raise RuntimeError("collinear EPR q=0 root did not receive a result")
            print(json.dumps(collinear_result.as_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "test-collinear-epr-q0-polaron":
        polaron_result = run_collinear_epr_q0_polaron_test(
            exchange_h5_path=args.exchange_h5,
            exchange_source_directed_bond_weight=(
                args.exchange_source_directed_bond_weight
            ),
            exchange_spin_normalization=args.exchange_spin_normalization,
            spin_lengths=args.spin_length,
            anisotropy_mev=args.anisotropy_mev,
            anisotropy_spin_normalization=(
                args.anisotropy_spin_normalization
            ),
            phonon_min_energy_mev=args.phonon_min_energy_mev,
            phonon_imaginary_tolerance_mev=(
                args.phonon_imaginary_tolerance_mev
            ),
            phonon_loto_mode=args.phonon_loto_mode,
            geometry_tolerance=args.geometry_tolerance,
            torque_imaginary_tolerance_eV_per_angstrom=(
                args.torque_imaginary_tolerance_ev_per_angstrom
            ),
            **_collinear_q0_kwargs(args, mpi),
        )
        if mpi.is_root:
            if polaron_result is None:
                raise RuntimeError(
                    "Gamma magnon-polaron root did not receive a result"
                )
            print(json.dumps(polaron_result.as_dict(), indent=2, sort_keys=True))
        return 0
    config = _config(args.config)
    exit_code = 0
    if args.command == "inspect":
        if mpi.is_root:
            print(json.dumps(inspect_config(config), indent=2, sort_keys=True))
    elif args.command == "extract-exchange":
        if mpi.is_root:
            print(_extract_exchange(config, args.output))
    elif args.command == "build-vertices":
        if mpi.is_root:
            print(_build_vertices(config, args.output))
    elif args.command == "compute-kernel":
        kernel_result = run_compute_kernel(config, mpi=mpi)
        if mpi.is_root:
            print(kernel_result.output_file)
    elif args.command == "project-phonons":
        stage_result = run_project_phonons(config, mpi=mpi)
        if mpi.is_root:
            print(stage_result.output_file)
    elif args.command == "project-magnons":
        stage_result = run_project_magnons(config, mpi=mpi)
        if mpi.is_root:
            print(stage_result.output_file)
    elif args.command == "assemble-polaron":
        stage_result = run_assemble_polaron(config, mpi=mpi)
        if mpi.is_root:
            print(stage_result.output_file)
    elif args.command == "validate" and mpi.is_root:
        report = validate_run(config, markdown_path=args.report)
        print(report.to_json())
        exit_code = 0 if report.valid else 2
    elif args.command == "benchmark" and mpi.is_root:
        benchmark_result = benchmark_q(config, args.q_index, repeats=args.repeats)
        print(json.dumps(benchmark_result.as_dict(), indent=2, sort_keys=True))
        exit_code = 0 if benchmark_result.agrees else 2
    exit_code = int(mpi.bcast(exit_code, root=0))
    mpi.barrier()
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["inspect_config", "main", "merge_outputs"]
