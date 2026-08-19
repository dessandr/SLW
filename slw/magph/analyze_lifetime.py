"""Analyze magnon lifetime observables and altermagnetic channel symmetry."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .lswt import parse_spin_pattern
from .numerics import linewidth_observables
from .utils import parsing_POSCAR


@dataclass(frozen=True)
class MagneticSymmetryOperation:
    index: int
    rotation: np.ndarray
    translation: np.ndarray
    atom_map: np.ndarray
    channel_map: np.ndarray
    rotation_order: int
    determinant: int


def rotate_fractional_momenta(kpoints_frac, direct_rotation):
    """Apply a direct-space fractional rotation to reciprocal row vectors."""
    kpoints = np.asarray(kpoints_frac, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(direct_rotation, dtype=np.float64).reshape(3, 3)
    return np.mod(kpoints @ np.linalg.inv(rotation), 1.0)


def periodic_mesh_map(kpoints_frac, transformed_frac, *, tolerance, workers=-1):
    """Map transformed fractional points back to a periodic mesh bijectively."""
    points = np.mod(np.asarray(kpoints_frac, dtype=np.float64).reshape(-1, 3), 1.0)
    transformed = np.mod(
        np.asarray(transformed_frac, dtype=np.float64).reshape(-1, 3), 1.0
    )
    if points.shape != transformed.shape:
        raise ValueError(
            f"point shape {points.shape} != transformed shape {transformed.shape}"
        )
    tree = cKDTree(points, boxsize=1.0)
    distance, index = tree.query(transformed, k=1, workers=int(workers))
    if np.any(distance > float(tolerance)):
        worst = int(np.argmax(distance))
        raise ValueError(
            "Rotated momentum is not present on the stored mesh: "
            f"source={worst}, distance={distance[worst]:.6g}, "
            f"tolerance={float(tolerance):.6g}"
        )
    if np.unique(index).size != points.shape[0]:
        raise ValueError("Rotated momentum map is not bijective on the stored mesh")
    return np.asarray(index, dtype=np.int64), np.asarray(distance, dtype=np.float64)


def compare_rotated_channels(values, momentum_map, channel_map, *, relative_floor=None):
    """Compare ``value_n(k)`` with ``value_P(n)(Rk)`` vectorially."""
    source = np.asarray(values)
    momentum_map = np.asarray(momentum_map, dtype=np.int64).reshape(-1)
    channel_map = np.asarray(channel_map, dtype=np.int64).reshape(-1)
    if source.ndim != 2:
        raise ValueError(f"values must have shape (nk,nchannel), got {source.shape}")
    if momentum_map.shape != (source.shape[0],):
        raise ValueError(
            f"momentum_map shape {momentum_map.shape} != {(source.shape[0],)}"
        )
    if channel_map.shape != (source.shape[1],):
        raise ValueError(
            f"channel_map shape {channel_map.shape} != {(source.shape[1],)}"
        )
    if np.unique(channel_map).size != source.shape[1] or np.any(channel_map < 0) or np.any(
        channel_map >= source.shape[1]
    ):
        raise ValueError(f"channel_map must be a channel permutation, got {channel_map}")

    transformed = source[momentum_map][:, channel_map]
    difference = source - transformed
    absolute = np.abs(difference)
    if relative_floor is None:
        finite_scale = np.max(np.abs(source[np.isfinite(source)]), initial=1.0)
        relative_floor = np.finfo(np.float64).eps * max(float(finite_scale), 1.0)
    denominator = np.abs(source) + np.abs(transformed) + float(relative_floor)
    symmetric_relative = 2.0 * absolute / denominator
    return {
        "source": source,
        "transformed": transformed,
        "difference": difference,
        "absolute_difference": absolute,
        "symmetric_relative_error": symmetric_relative,
        "relative_floor": float(relative_floor),
    }


def summarize_comparison(comparison, *, absolute_tolerance, relative_tolerance):
    absolute = np.asarray(comparison["absolute_difference"], dtype=np.float64)
    relative = np.asarray(comparison["symmetric_relative_error"], dtype=np.float64)
    summary = []
    for channel in range(absolute.shape[1]):
        valid = np.isfinite(absolute[:, channel]) & np.isfinite(relative[:, channel])
        abs_values = absolute[valid, channel]
        rel_values = relative[valid, channel]
        violation = (abs_values > float(absolute_tolerance)) & (
            rel_values > float(relative_tolerance)
        )
        summary.append(
            {
                "channel": int(channel),
                "count": int(abs_values.size),
                "rms_absolute": (
                    float(np.sqrt(np.mean(abs_values * abs_values)))
                    if abs_values.size
                    else None
                ),
                "max_absolute": float(np.max(abs_values)) if abs_values.size else None,
                "p95_absolute": (
                    float(np.percentile(abs_values, 95.0)) if abs_values.size else None
                ),
                "mean_symmetric_relative": (
                    float(np.mean(rel_values)) if rel_values.size else None
                ),
                "max_symmetric_relative": (
                    float(np.max(rel_values)) if rel_values.size else None
                ),
                "violation_count": int(np.count_nonzero(violation)),
                "violation_fraction": (
                    float(np.mean(violation)) if violation.size else None
                ),
            }
        )
    return summary


def _rotation_order(rotation, *, maximum_order=24):
    rotation = np.asarray(rotation, dtype=np.int64).reshape(3, 3)
    product = np.eye(3, dtype=np.int64)
    for order in range(1, int(maximum_order) + 1):
        product = product @ rotation
        if np.array_equal(product, np.eye(3, dtype=np.int64)):
            return order
    return 0


def _species_from_labels(labels):
    species = [re.sub(r"\d+$", "", str(label)) for label in labels]
    unique = {name: index + 1 for index, name in enumerate(dict.fromkeys(species))}
    return species, np.asarray([unique[name] for name in species], dtype=np.int32)


def _map_atoms_under_operation(
    positions_frac,
    species,
    lattice,
    rotation,
    translation,
    *,
    tolerance_ang,
):
    positions = np.mod(np.asarray(positions_frac, dtype=np.float64), 1.0)
    lattice = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    rotation = np.asarray(rotation, dtype=np.int64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    atom_map = np.full(positions.shape[0], -1, dtype=np.int64)
    distances = np.full(positions.shape[0], np.inf, dtype=np.float64)
    transformed = np.mod(positions @ rotation.T + translation, 1.0)
    for source, target_position in enumerate(transformed):
        candidates = np.asarray(
            [index for index, name in enumerate(species) if name == species[source]],
            dtype=np.int64,
        )
        delta = target_position[None, :] - positions[candidates]
        delta -= np.rint(delta)
        cartesian_distance = np.linalg.norm(delta @ lattice, axis=1)
        best = int(np.argmin(cartesian_distance))
        atom_map[source] = int(candidates[best])
        distances[source] = float(cartesian_distance[best])
    if np.any(distances > float(tolerance_ang)):
        worst = int(np.argmax(distances))
        raise ValueError(
            f"Atom {worst} maps with error {distances[worst]:.6g} Ang, "
            f"above tolerance {float(tolerance_ang):.6g} Ang"
        )
    if np.unique(atom_map).size != positions.shape[0]:
        raise ValueError("Symmetry operation did not produce a bijective atom map")
    return atom_map


def find_magnetic_transposing_operations(
    lattice,
    positions_frac,
    species,
    magnetic_atom_indices,
    spin_pattern,
    *,
    symprec,
    atom_tolerance_ang,
):
    """Find crystal operations that exchange every opposite-spin sublattice."""
    try:
        import spglib
    except ImportError as exc:
        raise RuntimeError("spglib is required for automatic symmetry discovery") from exc

    lattice = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    positions = np.mod(np.asarray(positions_frac, dtype=np.float64).reshape(-1, 3), 1.0)
    magnetic = np.asarray(magnetic_atom_indices, dtype=np.int64).reshape(-1)
    spins = np.asarray(spin_pattern, dtype=np.float64).reshape(-1)
    if magnetic.shape != spins.shape:
        raise ValueError(
            f"magnetic atom shape {magnetic.shape} != spin pattern shape {spins.shape}"
        )
    species_names, numbers = _species_from_labels(species)
    symmetry = spglib.get_symmetry(
        (lattice, positions, numbers), symprec=float(symprec)
    )
    if symmetry is None:
        raise RuntimeError("spglib did not find crystal symmetry operations")

    local_from_global = {int(atom): index for index, atom in enumerate(magnetic)}
    operations = []
    for index, (rotation, translation) in enumerate(
        zip(symmetry["rotations"], symmetry["translations"])
    ):
        try:
            atom_map = _map_atoms_under_operation(
                positions,
                species_names,
                lattice,
                rotation,
                translation,
                tolerance_ang=atom_tolerance_ang,
            )
        except ValueError:
            continue
        mapped_global = atom_map[magnetic]
        if any(int(atom) not in local_from_global for atom in mapped_global):
            continue
        channel_map = np.asarray(
            [local_from_global[int(atom)] for atom in mapped_global], dtype=np.int64
        )
        if not np.all(spins[channel_map] == -spins):
            continue
        operations.append(
            MagneticSymmetryOperation(
                index=int(index),
                rotation=np.asarray(rotation, dtype=np.int64),
                translation=np.asarray(translation, dtype=np.float64),
                atom_map=atom_map,
                channel_map=channel_map,
                rotation_order=_rotation_order(rotation),
                determinant=round(np.linalg.det(rotation)),
            )
        )
    return operations


def select_transposing_operation(operations, *, operation_index=None):
    if not operations:
        raise ValueError("No crystal symmetry operation exchanges all magnetic sublattices")
    if operation_index is not None:
        matches = [operation for operation in operations if operation.index == operation_index]
        if not matches:
            available = [operation.index for operation in operations]
            raise ValueError(
                f"symmetry operation {operation_index} is not transposing; available={available}"
            )
        return matches[0]
    # Prefer a proper operation of the highest crystallographic order.  For
    # This can select a sublattice-transposing screw operation without naming it.
    return max(
        operations,
        key=lambda operation: (
            operation.determinant == 1,
            operation.rotation_order,
            -operation.index,
        ),
    )


def _scalar_from_npz(payload, key, default=None):
    if key not in payload:
        return default
    array = np.asarray(payload[key])
    if array.size != 1:
        raise ValueError(f"{key} must be scalar, got {array.shape}")
    return array.reshape(-1)[0].item()


def _comparison_payload(prefix, comparison):
    return {
        f"{prefix}_source": comparison["source"],
        f"{prefix}_transformed": comparison["transformed"],
        f"{prefix}_difference": comparison["difference"],
        f"{prefix}_absolute_difference": comparison["absolute_difference"],
        f"{prefix}_symmetric_relative_error": comparison["symmetric_relative_error"],
    }


def _summarize_observables(observables, physical_count):
    keys = (
        "gamma_hwhm_mev",
        "fwhm_mev",
        "lifetime_ps",
        "scattering_rate_ps_inv",
    )
    summary = []
    for channel in range(int(physical_count)):
        item = {"channel": int(channel)}
        for key in keys:
            values = np.asarray(observables[key], dtype=np.float64)[:, channel]
            finite = values[np.isfinite(values)]
            item[key] = {
                "min": float(np.min(finite)) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
                "infinite_count": int(np.count_nonzero(np.isinf(values))),
                "invalid_count": int(np.count_nonzero(np.isnan(values))),
            }
        summary.append(item)
    return summary


def _summarize_sign_audit(payload, physical_count):
    stored_tolerance = _scalar_from_npz(payload, "negative_damping_tol_mev")
    tolerance = float(stored_tolerance) if stored_tolerance is not None else 0.0
    audit = {
        "negative_damping_tolerance_mev": tolerance,
        "raw_physical_damping_min_mev": None,
        "raw_physical_damping_negative_count": None,
        "paraunitary_residual_max": None,
        "paraunitary_violation_count": None,
        "metric_energy_min_mev": None,
        "metric_energy_sign_mismatch_count": None,
        "common_damping_min_eigenvalue_mev": None,
        "common_damping_non_psd_count": None,
    }
    if "linewidth_raw" in payload:
        raw_gamma = -np.asarray(payload["linewidth_raw"], dtype=np.float64)[
            :, :physical_count
        ]
        audit["raw_physical_damping_min_mev"] = float(np.min(raw_gamma))
        audit["raw_physical_damping_negative_count"] = int(
            np.count_nonzero(raw_gamma < -tolerance)
        )
    if "paraunitary_residual_max" in payload:
        residual = np.asarray(payload["paraunitary_residual_max"], dtype=np.float64)
        para_tolerance = float(
            _scalar_from_npz(payload, "paraunitary_tolerance", default=0.0)
        )
        audit["paraunitary_tolerance"] = para_tolerance
        audit["paraunitary_residual_max"] = float(np.max(residual, initial=0.0))
        audit["paraunitary_violation_count"] = int(
            np.count_nonzero(residual > para_tolerance)
        )
    if "metric_energy_min_mev" in payload:
        values = np.asarray(payload["metric_energy_min_mev"], dtype=np.float64)
        metric_tolerance = float(
            _scalar_from_npz(
                payload, "metric_energy_tolerance_mev", default=tolerance
            )
        )
        audit["metric_energy_tolerance_mev"] = metric_tolerance
        audit["metric_energy_min_mev"] = float(np.min(values))
        audit["metric_energy_sign_mismatch_count"] = int(
            np.count_nonzero(values < -metric_tolerance)
        )
    if "gamma_phys_common_eigvals" in payload:
        values = np.asarray(payload["gamma_phys_common_eigvals"], dtype=np.float64)
        audit["common_damping_min_eigenvalue_mev"] = float(np.min(values))
        audit["common_damping_non_psd_count"] = int(
            np.count_nonzero(np.min(values, axis=1) < -tolerance)
        )
    return audit


def _plot_difference(path, kpoints, difference, *, plane_axes, slice_value, slice_tolerance, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = tuple(int(axis) for axis in plane_axes)
    if len(set(axes)) != 2 or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError(f"plane_axes must be two different axes, got {axes}")
    normal = next(axis for axis in (0, 1, 2) if axis not in axes)
    wrapped_normal = ((kpoints[:, normal] - float(slice_value) + 0.5) % 1.0) - 0.5
    mask = np.abs(wrapped_normal) <= float(slice_tolerance)
    if not np.any(mask):
        raise ValueError("No momentum points lie on the requested plot slice")
    points = ((kpoints[mask][:, axes] + 0.5) % 1.0) - 0.5
    values = np.asarray(difference)[mask]
    channel_count = values.shape[1]
    figure, panels = plt.subplots(
        1,
        channel_count,
        figsize=(4.5 * channel_count, 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    limit = np.percentile(np.abs(values[np.isfinite(values)]), 99.0)
    limit = max(float(limit), np.finfo(np.float64).eps)
    artist = None
    for channel, axis in enumerate(panels.flat):
        artist = axis.scatter(
            points[:, 0],
            points[:, 1],
            c=values[:, channel],
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            s=18.0,
            linewidths=0.0,
        )
        axis.set_title(f"channel {channel}")
        axis.set_xlabel(f"fractional k[{axes[0]}]")
        axis.set_ylabel(f"fractional k[{axes[1]}]")
        axis.set_aspect("equal", adjustable="box")
    figure.colorbar(artist, ax=panels.ravel().tolist(), label="signed linewidth difference (meV)")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi))
    plt.close(figure)
    return output


def analyze(args):
    input_path = Path(args.input).expanduser().resolve()
    with np.load(input_path, allow_pickle=True) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    if args.linewidth_key not in payload:
        raise KeyError(
            f"Missing {args.linewidth_key!r} in {input_path}; keys={sorted(payload)}"
        )
    linewidth = np.asarray(payload[args.linewidth_key], dtype=np.float64)
    kpoints = np.asarray(payload[args.kpoints_key], dtype=np.float64)
    if linewidth.ndim != 2 or kpoints.shape != (linewidth.shape[0], 3):
        raise ValueError(
            f"Incompatible linewidth/k-point shapes: {linewidth.shape}, {kpoints.shape}"
        )

    stored_physical = _scalar_from_npz(payload, "physical_channel_count")
    if args.physical_channel_count is not None:
        physical_count = int(args.physical_channel_count)
    elif stored_physical is not None:
        physical_count = int(stored_physical)
    elif linewidth.shape[1] % 2 == 0:
        physical_count = linewidth.shape[1] // 2
    else:
        physical_count = linewidth.shape[1]
    if physical_count < 1 or physical_count > linewidth.shape[1]:
        raise ValueError(
            f"physical_channel_count={physical_count} is invalid for {linewidth.shape[1]} channels"
        )

    operation = None
    if args.rotation is not None:
        rotation = np.asarray(args.rotation, dtype=np.int64).reshape(3, 3)
        translation = np.zeros(3, dtype=np.float64)
    else:
        if args.structure is None:
            raise ValueError("Automatic rotation discovery requires --structure")
        lattice, labels, atom_cart = parsing_POSCAR(args.structure)
        lattice = np.asarray(lattice, dtype=np.float64)
        positions_frac = np.mod(np.asarray(atom_cart) @ np.linalg.inv(lattice), 1.0)
        if args.magnetic_atom_indices is not None:
            magnetic_atoms = np.asarray(args.magnetic_atom_indices, dtype=np.int64)
        elif "magnetic_atom_indices" in payload:
            magnetic_atoms = np.asarray(payload["magnetic_atom_indices"], dtype=np.int64)
        else:
            magnetic_atoms = np.arange(physical_count, dtype=np.int64)
        if args.spin_pattern is not None:
            spins = parse_spin_pattern(args.spin_pattern, physical_count)
        elif "spin_pattern" in payload:
            spins = np.asarray(payload["spin_pattern"], dtype=np.float64)
        else:
            spins = parse_spin_pattern("auto", physical_count)
        operations = find_magnetic_transposing_operations(
            lattice,
            positions_frac,
            labels,
            magnetic_atoms,
            spins,
            symprec=args.symprec,
            atom_tolerance_ang=args.atom_tolerance_ang,
        )
        operation = select_transposing_operation(
            operations, operation_index=args.symmetry_operation_index
        )
        rotation = operation.rotation
        translation = operation.translation

    if args.channel_map is not None:
        channel_map = np.asarray(args.channel_map, dtype=np.int64)
    elif operation is not None:
        channel_map = operation.channel_map
    else:
        raise ValueError("Explicit --rotation also requires --channel-map")
    if channel_map.shape != (physical_count,):
        raise ValueError(
            f"channel map shape {channel_map.shape} != physical channel count {(physical_count,)}"
        )

    rotated_kpoints = rotate_fractional_momenta(kpoints, rotation)
    momentum_map, momentum_distance = periodic_mesh_map(
        kpoints,
        rotated_kpoints,
        tolerance=args.mesh_tolerance,
        workers=args.workers,
    )
    qpoints = None
    rotated_qpoints = None
    q_momentum_map = None
    q_momentum_distance = None
    if args.qpoints_key in payload:
        qpoints = np.asarray(payload[args.qpoints_key], dtype=np.float64).reshape(-1, 3)
        rotated_qpoints = rotate_fractional_momenta(qpoints, rotation)
        q_momentum_map, q_momentum_distance = periodic_mesh_map(
            qpoints,
            rotated_qpoints,
            tolerance=args.mesh_tolerance,
            workers=args.workers,
        )
    gamma_physical = linewidth[:, :physical_count]
    gamma_comparison = compare_rotated_channels(
        gamma_physical,
        momentum_map,
        channel_map,
        relative_floor=args.relative_floor,
    )
    gamma_summary = summarize_comparison(
        gamma_comparison,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )

    energy_comparison = None
    energy_summary = None
    if args.energy_key in payload:
        energy = np.asarray(payload[args.energy_key], dtype=np.float64)
        if energy.shape[0] != linewidth.shape[0] or energy.shape[1] < physical_count:
            raise ValueError(f"Invalid energy shape {energy.shape}")
        energy_comparison = compare_rotated_channels(
            energy[:, :physical_count],
            momentum_map,
            channel_map,
            relative_floor=args.relative_floor,
        )
        energy_summary = summarize_comparison(
            energy_comparison,
            absolute_tolerance=args.energy_absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )

    observables = linewidth_observables(linewidth)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_suffix(".lifetime_analysis.npz")
    )
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary
        else output_path.with_suffix(".json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    output_payload = {
        **observables,
        **_comparison_payload("linewidth", gamma_comparison),
        "kpoints_frac": kpoints,
        "rotated_kpoints_frac": rotated_kpoints,
        "momentum_map": momentum_map,
        "momentum_map_distance": momentum_distance,
        "channel_map": channel_map,
        "rotation_direct_frac": rotation,
        "translation_frac": translation,
        "physical_channel_count": np.asarray(physical_count, dtype=np.int32),
    }
    if "linewidth_raw" in payload:
        output_payload["raw_gamma_hwhm_mev"] = -np.asarray(
            payload["linewidth_raw"], dtype=np.float64
        )
    if energy_comparison is not None:
        output_payload.update(_comparison_payload("energy", energy_comparison))
    if qpoints is not None:
        output_payload.update(
            {
                "qpoints_frac": qpoints,
                "rotated_qpoints_frac": rotated_qpoints,
                "q_momentum_map": q_momentum_map,
                "q_momentum_map_distance": q_momentum_distance,
            }
        )
    np.savez_compressed(output_path, **output_payload)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "linewidth_key": args.linewidth_key,
        "linewidth_definition": "gamma_hwhm_mev = -Im Sigma^R_mm(E_m)",
        "physical_channel_count": int(physical_count),
        "channel_map": channel_map.tolist(),
        "rotation_direct_frac": np.asarray(rotation).tolist(),
        "translation_frac": np.asarray(translation).tolist(),
        "symmetry_operation_index": operation.index if operation is not None else None,
        "rotation_order": (
            operation.rotation_order if operation is not None else _rotation_order(rotation)
        ),
        "rotation_determinant": round(np.linalg.det(rotation)),
        "mesh_max_mapping_distance": float(np.max(momentum_distance, initial=0.0)),
        "q_mesh_present": qpoints is not None,
        "q_mesh_max_mapping_distance": (
            float(np.max(q_momentum_distance, initial=0.0))
            if q_momentum_distance is not None
            else None
        ),
        "absolute_tolerance_mev": float(args.absolute_tolerance),
        "relative_tolerance": float(args.relative_tolerance),
        "observables": _summarize_observables(observables, physical_count),
        "sign_audit": _summarize_sign_audit(payload, physical_count),
        "linewidth_symmetry": gamma_summary,
        "energy_symmetry": energy_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = None
    if args.plot:
        plot_path = _plot_difference(
            args.plot,
            kpoints,
            gamma_comparison["difference"],
            plane_axes=args.plane_axes,
            slice_value=args.slice_value,
            slice_tolerance=args.slice_tolerance,
            dpi=args.dpi,
        )
    return output_path, summary_path, plot_path, summary


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert magph linewidths to lifetime/scattering rate and audit "
            "altermagnetic rotation-channel symmetry"
        )
    )
    parser.add_argument("input", help="NPZ from slw.magph.lifetime_mpi_dynamic")
    parser.add_argument("--structure", help="POSCAR/card structure for automatic symmetry discovery")
    parser.add_argument("--linewidth-key", default="linewidth")
    parser.add_argument("--energy-key", default="energy")
    parser.add_argument("--kpoints-key", default="k_mesh_frac")
    parser.add_argument("--qpoints-key", default="q_mesh_frac")
    parser.add_argument("--physical-channel-count", type=int)
    parser.add_argument("--magnetic-atom-indices", nargs="+", type=int)
    parser.add_argument("--spin-pattern")
    parser.add_argument(
        "--rotation",
        nargs=9,
        type=int,
        metavar=("R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"),
        help="Explicit direct-space fractional rotation matrix",
    )
    parser.add_argument("--channel-map", nargs="+", type=int)
    parser.add_argument("--symmetry-operation-index", type=int)
    parser.add_argument("--symprec", type=float, default=1.0e-5)
    parser.add_argument("--atom-tolerance-ang", type=float, default=1.0e-4)
    parser.add_argument("--mesh-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--relative-floor", type=float)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--energy-absolute-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-6)
    parser.add_argument("-o", "--output")
    parser.add_argument("--summary")
    parser.add_argument("--plot")
    parser.add_argument("--plane-axes", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--slice-value", type=float, default=0.0)
    parser.add_argument("--slice-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main():
    args = build_parser().parse_args()
    output, summary_path, plot, summary = analyze(args)
    print(
        "[magph:lifetime-analysis] "
        f"rotation_order={summary['rotation_order']} "
        f"operation={summary['symmetry_operation_index']} "
        f"channel_map={summary['channel_map']}"
    )
    for item in summary["linewidth_symmetry"]:
        print(
            "[magph:lifetime-analysis] "
            f"channel={item['channel']} rms_abs={item['rms_absolute']:.6g} meV "
            f"max_abs={item['max_absolute']:.6g} meV "
            f"max_rel={item['max_symmetric_relative']:.6g} "
            f"violations={item['violation_count']}/{item['count']}"
        )
    for item in summary["observables"]:
        lifetime = item["lifetime_ps"]
        rate = item["scattering_rate_ps_inv"]
        print(
            "[magph:lifetime-analysis] "
            f"channel={item['channel']} "
            f"lifetime_ps[min,median,max]=({lifetime['min']},{lifetime['median']},{lifetime['max']}) "
            f"rate_ps^-1[min,median,max]=({rate['min']},{rate['median']},{rate['max']})"
        )
    sign_audit = summary["sign_audit"]
    print(
        "[magph:lifetime-analysis] sign-audit "
        f"negative_raw={sign_audit['raw_physical_damping_negative_count']} "
        f"non_paraunitary={sign_audit['paraunitary_violation_count']} "
        f"metric_energy_mismatch={sign_audit['metric_energy_sign_mismatch_count']} "
        f"non_psd_common={sign_audit['common_damping_non_psd_count']}"
    )
    print(f"[magph:lifetime-analysis] wrote {output}")
    print(f"[magph:lifetime-analysis] wrote {summary_path}")
    if plot is not None:
        print(f"[magph:lifetime-analysis] wrote {plot}")


if __name__ == "__main__":
    main()
