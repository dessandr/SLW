"""Publication-style BZ maps for magnon lifetime and channel symmetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from .analyze_lifetime import (
    compare_rotated_channels,
    find_magnetic_transposing_operations,
    periodic_mesh_map,
    rotate_fractional_momenta,
    select_transposing_operation,
)
from .reference.hybrid_berry import (
    _first_bz_polygon_2d,
    _periodic_voronoi_polygons,
    reciprocal_plane_basis_2d,
)
from .numerics import linewidth_observables
from .plot_coupling_kbz import (
    _plane_points,
    _select_plane_slice,
    _slice_axis,
    _validate_plane_axes,
)
from .utils import parsing_POSCAR


def _scalar(payload, key, default=None):
    if key not in payload:
        return default
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got {value.shape}")
    return value.reshape(-1)[0].item()


def _finite_percentile(values, percentile, *, default):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float(default)
    return float(np.percentile(finite, float(percentile)))


def _safe_limits(vmin, vmax):
    low = float(vmin)
    high = float(vmax)
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if high < low:
        low, high = high, low
    if abs(high - low) <= np.finfo(np.float64).eps * max(abs(low), abs(high), 1.0):
        pad = max(abs(low), abs(high), 1.0) * 1.0e-6
        low -= pad
        high += pad
    return low, high


def _format_slice(value):
    return f"{float(value):.4f}".replace("-", "m").replace(".", "p")


def _load_payload(path):
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _physical_channel_count(payload, requested=None):
    if requested is not None:
        count = int(requested)
    else:
        count = int(_scalar(payload, "physical_channel_count", 0))
    linewidth = np.asarray(payload["linewidth"])
    if count < 1 or count > linewidth.shape[1]:
        raise ValueError(
            f"physical_channel_count={count} is invalid for linewidth {linewidth.shape}"
        )
    return count


def _automatic_operation(payload, structure, physical_count, args):
    lattice, labels, atom_cart = parsing_POSCAR(str(structure))
    lattice = np.asarray(lattice, dtype=np.float64)
    positions = np.mod(
        np.asarray(atom_cart, dtype=np.float64) @ np.linalg.inv(lattice), 1.0
    )
    magnetic_atoms = (
        np.asarray(args.magnetic_atom_indices, dtype=np.int64)
        if args.magnetic_atom_indices is not None
        else np.asarray(payload["magnetic_atom_indices"], dtype=np.int64)
    )
    spins = (
        np.asarray(args.spin_pattern, dtype=np.float64)
        if args.spin_pattern is not None
        else np.asarray(payload["spin_pattern"], dtype=np.float64)
    )
    if magnetic_atoms.shape != (physical_count,) or spins.shape != (physical_count,):
        raise ValueError(
            "magnetic atom/spin metadata must contain one entry per physical channel: "
            f"atoms={magnetic_atoms.shape}, spins={spins.shape}, channels={physical_count}"
        )
    operations = find_magnetic_transposing_operations(
        lattice,
        positions,
        labels,
        magnetic_atoms,
        spins,
        symprec=float(args.symprec),
        atom_tolerance_ang=float(args.atom_tolerance_ang),
    )
    operation = select_transposing_operation(
        operations, operation_index=args.symmetry_operation_index
    )
    return (
        lattice,
        operation.rotation,
        operation.translation,
        operation.channel_map,
        operation,
    )


def _resolve_operation(payload, physical_count, args):
    if args.rotation is None:
        if args.structure is None:
            raise ValueError("--structure is required for automatic symmetry discovery")
        return _automatic_operation(payload, args.structure, physical_count, args)
    if args.channel_map is None:
        raise ValueError("Explicit --rotation requires --channel-map")
    lattice = np.asarray(payload["lattice_ang"], dtype=np.float64)
    rotation = np.asarray(args.rotation, dtype=np.int64).reshape(3, 3)
    channel_map = np.asarray(args.channel_map, dtype=np.int64)
    if channel_map.shape != (physical_count,):
        raise ValueError(
            f"channel map shape {channel_map.shape} != {(physical_count,)}"
        )
    return lattice, rotation, np.zeros(3), channel_map, None


def build_plot_data(payload, *, physical_count, rotation, channel_map, workers=-1, mesh_tolerance=1.0e-8):
    """Build physical observables and the rotation/channel comparison arrays."""
    kpoints = np.asarray(payload["k_mesh_frac"], dtype=np.float64)
    linewidth = np.asarray(payload["linewidth"], dtype=np.float64)
    energy = np.asarray(payload["energy"], dtype=np.float64)
    if kpoints.shape != (linewidth.shape[0], 3) or energy.shape[0] != kpoints.shape[0]:
        raise ValueError(
            f"incompatible k/linewidth/energy shapes: {kpoints.shape}, "
            f"{linewidth.shape}, {energy.shape}"
        )
    gamma = linewidth[:, :physical_count]
    physical_energy = energy[:, :physical_count]
    rotated = rotate_fractional_momenta(kpoints, rotation)
    momentum_map, distance = periodic_mesh_map(
        kpoints,
        rotated,
        tolerance=float(mesh_tolerance),
        workers=int(workers),
    )
    comparison = compare_rotated_channels(gamma, momentum_map, channel_map)
    energy_comparison = compare_rotated_channels(
        physical_energy, momentum_map, channel_map
    )
    observables = linewidth_observables(gamma)
    lifetime = np.asarray(observables["lifetime_ps"], dtype=np.float64)
    log_lifetime = np.full_like(lifetime, np.nan)
    positive_finite = np.isfinite(lifetime) & (lifetime > 0.0)
    log_lifetime[positive_finite] = np.log10(lifetime[positive_finite])
    return {
        "kpoints_frac": kpoints,
        "gamma_hwhm_mev": gamma,
        "scattering_rate_ps_inv": np.asarray(
            observables["scattering_rate_ps_inv"], dtype=np.float64
        ),
        "log10_lifetime_ps": log_lifetime,
        "energy_mev": physical_energy,
        "linewidth_absolute_error_mev": comparison["absolute_difference"],
        "linewidth_relative_error_percent": 100.0
        * comparison["symmetric_relative_error"],
        "linewidth_signed_difference_mev": comparison["difference"],
        "energy_absolute_error_mev": energy_comparison["absolute_difference"],
        "momentum_map": momentum_map,
        "momentum_map_distance": distance,
    }


def _select_slices(kpoints, requested, plane_axes, decimals):
    normal_axis = _slice_axis(plane_axes)
    selected = []
    for value in requested:
        mask, chosen = _select_plane_slice(
            kpoints, axis=normal_axis, value=float(value), decimals=int(decimals)
        )
        if not any(np.isclose(chosen, item[0], atol=10.0 ** (-int(decimals))) for item in selected):
            selected.append((chosen, mask))
    if not selected:
        raise ValueError("At least one slice is required")
    return selected


def _polygon_intersection_mask(polygons, minimum, maximum):
    return np.asarray(
        [
            np.all(np.max(polygon, axis=0) >= minimum)
            and np.all(np.min(polygon, axis=0) <= maximum)
            for polygon in polygons
        ],
        dtype=bool,
    )


def _display_geometry(
    kpoints_slice,
    values_slice,
    lattice,
    plane_axes,
    *,
    tile,
    bz_mode,
    periodic_repeats,
    periodic_view="central",
    periodic_padding=0.06,
):
    b1, b2 = reciprocal_plane_basis_2d(lattice, plane_axes)
    points = _plane_points(kpoints_slice, plane_axes, b1, b2)
    boundary = _first_bz_polygon_2d(b1, b2)
    mode = str(bz_mode).strip().lower()
    if mode not in {"clip", "periodic"}:
        raise ValueError(f"bz_mode must be clip or periodic, got {bz_mode!r}")
    repeats = int(periodic_repeats)
    if repeats < 0:
        raise ValueError(f"periodic_repeats must be non-negative, got {repeats}")
    view = str(periodic_view).strip().lower()
    if view not in {"central", "neighbors"}:
        raise ValueError(
            f"periodic_view must be central or neighbors, got {periodic_view!r}"
        )
    padding = float(periodic_padding)
    if padding < 0.0:
        raise ValueError(f"periodic_padding must be non-negative, got {padding}")
    if mode == "periodic" and view == "neighbors":
        construction_tile = max(int(tile), repeats + 1)
    elif mode == "periodic":
        # One extra image shell guarantees finite Voronoi cells throughout the
        # rectangular view surrounding the central first BZ.
        construction_tile = max(int(tile), 2)
    else:
        construction_tile = max(int(tile), 1)
    polygons, polygon_values = _periodic_voronoi_polygons(
        points, values_slice, b1, b2, construction_tile
    )
    if not polygons:
        raise RuntimeError("Voronoi construction produced no finite cells")
    outlines = []
    clip_boundary = None
    add_axis_margin = True
    if boundary is not None and mode == "periodic" and view == "neighbors":
        outlines = [
            (boundary + ia * b1 + ib * b2, ia == 0 and ib == 0)
            for ia in range(-repeats, repeats + 1)
            for ib in range(-repeats, repeats + 1)
        ]
        all_vertices = np.vstack([polygon for polygon, _ in outlines])
        minimum = np.min(all_vertices, axis=0)
        maximum = np.max(all_vertices, axis=0)
        visible = _polygon_intersection_mask(polygons, minimum, maximum)
        polygons = [polygon for polygon, keep in zip(polygons, visible) if keep]
        polygon_values = polygon_values[visible]
    elif boundary is not None and mode == "periodic":
        # Keep the first BZ large and centered, but do not clip the collection.
        # The rectangular strip immediately outside the boundary is populated
        # by periodic images, avoiding the white background of clipped maps.
        outlines = [(boundary, True)]
        minimum = np.min(boundary, axis=0)
        maximum = np.max(boundary, axis=0)
        span = np.maximum(maximum - minimum, np.finfo(np.float64).eps)
        minimum = minimum - padding * span
        maximum = maximum + padding * span
        visible = _polygon_intersection_mask(polygons, minimum, maximum)
        polygons = [polygon for polygon, keep in zip(polygons, visible) if keep]
        polygon_values = polygon_values[visible]
        add_axis_margin = False
    elif boundary is not None:
        outlines = [(boundary, True)]
        clip_boundary = boundary
        minimum = np.min(boundary, axis=0)
        maximum = np.max(boundary, axis=0)
        visible = _polygon_intersection_mask(polygons, minimum, maximum)
        polygons = [polygon for polygon, keep in zip(polygons, visible) if keep]
        polygon_values = polygon_values[visible]
    else:
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
    return {
        "points": points,
        "polygons": polygons,
        "polygon_values": polygon_values,
        "outlines": outlines,
        "clip_boundary": clip_boundary,
        "minimum": minimum,
        "maximum": maximum,
        "add_axis_margin": add_axis_margin,
    }


def _decorate_axis(axis, collection, geometry, plane_axes, args):
    clip_boundary = geometry["clip_boundary"]
    if clip_boundary is not None:
        clip = plt.Polygon(clip_boundary, closed=True, facecolor="none", edgecolor="none")
        axis.add_patch(clip)
        collection.set_clip_path(clip)
    for outline, central in geometry["outlines"]:
        closed = np.vstack((outline, outline[0]))
        axis.plot(
            closed[:, 0],
            closed[:, 1],
            color=args.bz_color,
            lw=float(args.bz_lw if central else args.neighbor_bz_lw),
            alpha=1.0 if central else float(args.neighbor_bz_alpha),
            zorder=4,
        )
    minimum = geometry["minimum"]
    maximum = geometry["maximum"]
    span = np.maximum(maximum - minimum, np.finfo(np.float64).eps)
    margin = float(args.margin) if geometry["add_axis_margin"] else 0.0
    axis.set_xlim(minimum[0] - margin * span[0], maximum[0] + margin * span[0])
    axis.set_ylim(minimum[1] - margin * span[1], maximum[1] + margin * span[1])
    axis.set_aspect("equal")
    axis.set_xlabel(rf"$k_{{{plane_axes[0] + 1}}}$ ($\AA^{{-1}}$)")
    axis.set_ylabel(rf"$k_{{{plane_axes[1] + 1}}}$ ($\AA^{{-1}}$)")


def _add_collection(axis, polygons, values, *, cmap, limits, edgecolor, linewidth):
    collection = PolyCollection(
        polygons,
        array=np.asarray(values, dtype=np.float64),
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=float(linewidth),
    )
    collection.set_clim(*limits)
    axis.add_collection(collection)
    return collection


def _save_figure(figure, output_dir, stem, formats, dpi):
    outputs = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def _plot_channels(
    *,
    values,
    kpoints,
    mask,
    lattice,
    plane_axes,
    title,
    channel_labels,
    colorbar_label,
    cmap,
    limits,
    args,
    output_dir,
    stem,
):
    sliced = np.asarray(values)[mask]
    geometry = _display_geometry(
        kpoints[mask],
        sliced,
        lattice,
        plane_axes,
        tile=args.tile,
        bz_mode=args.bz_mode,
        periodic_repeats=args.periodic_repeats,
        periodic_view=args.periodic_view,
        periodic_padding=args.periodic_padding,
    )
    count = sliced.shape[1]
    figure, axes = plt.subplots(
        1,
        count,
        figsize=(float(args.panel_width) * count, float(args.panel_height)),
        squeeze=False,
        constrained_layout=True,
    )
    collection = None
    for channel in range(count):
        axis = axes.flat[channel]
        collection = _add_collection(
            axis,
            geometry["polygons"],
            geometry["polygon_values"][:, channel],
            cmap=cmap,
            limits=limits,
            edgecolor=args.edgecolor,
            linewidth=args.linewidth,
        )
        _decorate_axis(axis, collection, geometry, plane_axes, args)
        axis.set_title(str(channel_labels[channel]))
    figure.suptitle(title)
    figure.colorbar(collection, ax=list(axes.flat), label=colorbar_label, shrink=0.86)
    return _save_figure(figure, output_dir, stem, args.formats, args.dpi)


def _plot_rotation_error(
    *, data, kpoints, mask, lattice, plane_axes, title, channel_labels, limits, args, output_dir, stem
):
    absolute = np.asarray(data["linewidth_absolute_error_mev"])[mask] * float(
        args.linewidth_scale
    )
    relative = np.asarray(data["linewidth_relative_error_percent"])[mask]
    combined = np.concatenate((absolute, relative), axis=1)
    geometry = _display_geometry(
        kpoints[mask],
        combined,
        lattice,
        plane_axes,
        tile=args.tile,
        bz_mode=args.bz_mode,
        periodic_repeats=args.periodic_repeats,
        periodic_view=args.periodic_view,
        periodic_padding=args.periodic_padding,
    )
    channel_count = absolute.shape[1]
    figure, axes = plt.subplots(
        2,
        channel_count,
        figsize=(float(args.panel_width) * channel_count, float(args.panel_height) * 1.85),
        squeeze=False,
        constrained_layout=True,
    )
    row_collections = []
    for row in range(2):
        row_collection = None
        for channel in range(channel_count):
            axis = axes[row, channel]
            index = channel if row == 0 else channel_count + channel
            row_collection = _add_collection(
                axis,
                geometry["polygons"],
                geometry["polygon_values"][:, index],
                cmap=args.error_cmap,
                limits=limits[row],
                edgecolor=args.edgecolor,
                linewidth=args.linewidth,
            )
            _decorate_axis(axis, row_collection, geometry, plane_axes, args)
            axis.set_title(f"{channel_labels[channel]}: " + (r"$|\Delta\Gamma|$" if row == 0 else r"relative error"))
        row_collections.append(row_collection)
    figure.suptitle(title)
    figure.colorbar(
        row_collections[0],
        ax=list(axes[0]),
        label=rf"$|\Gamma_n(\mathbf{{k}})-\Gamma_{{P(n)}}(R\mathbf{{k}})|$ ({args.linewidth_unit})",
        shrink=0.84,
    )
    figure.colorbar(
        row_collections[1],
        ax=list(axes[1]),
        label="symmetric relative error (%)",
        shrink=0.84,
    )
    return _save_figure(figure, output_dir, stem, args.formats, args.dpi)


def _plot_channel_splitting(
    *, data, channels, kpoints, mask, lattice, plane_axes, title, limits, args, output_dir, stem
):
    first, second = (int(channels[0]), int(channels[1]))
    energy = data["energy_mev"][:, first] - data["energy_mev"][:, second]
    gamma = (
        data["gamma_hwhm_mev"][:, first] - data["gamma_hwhm_mev"][:, second]
    ) * float(args.linewidth_scale)
    combined = np.column_stack((energy[mask], gamma[mask]))
    geometry = _display_geometry(
        kpoints[mask],
        combined,
        lattice,
        plane_axes,
        tile=args.tile,
        bz_mode=args.bz_mode,
        periodic_repeats=args.periodic_repeats,
        periodic_view=args.periodic_view,
        periodic_padding=args.periodic_padding,
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(float(args.panel_width) * 2, float(args.panel_height)),
        squeeze=False,
        constrained_layout=True,
    )
    labels = (
        rf"$E_{{{first}}}-E_{{{second}}}$ (meV)",
        rf"$\Gamma_{{{first}}}-\Gamma_{{{second}}}$ ({args.linewidth_unit})",
    )
    titles = ("magnon energy splitting", "linewidth splitting")
    for panel in range(2):
        collection = _add_collection(
            axes[0, panel],
            geometry["polygons"],
            geometry["polygon_values"][:, panel],
            cmap=args.diverging_cmap,
            limits=limits[panel],
            edgecolor=args.edgecolor,
            linewidth=args.linewidth,
        )
        _decorate_axis(axes[0, panel], collection, geometry, plane_axes, args)
        axes[0, panel].set_title(titles[panel])
        figure.colorbar(collection, ax=axes[0, panel], label=labels[panel], shrink=0.86)
    figure.suptitle(title)
    return _save_figure(figure, output_dir, stem, args.formats, args.dpi)


def _style():
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.0,
            "axes.titlesize": 11.0,
            "figure.titlesize": 12.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_lifetime_symmetry(args):
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_payload(input_path)
    physical_count = _physical_channel_count(payload, args.physical_channel_count)
    lattice, rotation, translation, channel_map, operation = _resolve_operation(
        payload, physical_count, args
    )
    data = build_plot_data(
        payload,
        physical_count=physical_count,
        rotation=rotation,
        channel_map=channel_map,
        workers=args.workers,
        mesh_tolerance=args.mesh_tolerance,
    )
    plane_axes = _validate_plane_axes(args.plane_axes)
    normal_axis = _slice_axis(plane_axes)
    slice_requests = (
        np.unique(
            np.round(
                np.mod(data["kpoints_frac"][:, normal_axis], 1.0),
                int(args.k_round),
            )
        )
        if args.all_slices
        else args.slices
    )
    slices = _select_slices(
        data["kpoints_frac"], slice_requests, plane_axes, args.k_round
    )
    plot_types = set(args.plots)
    if "all" in plot_types:
        plot_types = {
            "linewidth",
            "scattering-rate",
            "lifetime",
            "rotation-error",
            "splitting",
        }
    union = np.logical_or.reduce([mask for _, mask in slices])
    gamma_display = data["gamma_hwhm_mev"] * float(args.linewidth_scale)
    rate_display = data["scattering_rate_ps_inv"] * float(args.scattering_rate_scale)
    log_lifetime = data["log10_lifetime_ps"]
    absolute_error = (
        data["linewidth_absolute_error_mev"] * float(args.linewidth_scale)
    )
    relative_error = data["linewidth_relative_error_percent"]
    split_first, split_second = (
        int(args.split_channels[0]),
        int(args.split_channels[1]),
    )
    if "splitting" in plot_types and any(
        index < 0 or index >= physical_count for index in (split_first, split_second)
    ):
        raise ValueError(
            f"split channels {args.split_channels} are invalid for {physical_count} channels"
        )
    if "splitting" in plot_types:
        energy_split = (
            data["energy_mev"][:, split_first]
            - data["energy_mev"][:, split_second]
        )
        gamma_split = (
            gamma_display[:, split_first] - gamma_display[:, split_second]
        )
    else:
        energy_split = np.zeros(data["kpoints_frac"].shape[0], dtype=np.float64)
        gamma_split = np.zeros_like(energy_split)

    gamma_limits = _safe_limits(
        _finite_percentile(gamma_display[union], args.vmin_percentile, default=0.0),
        _finite_percentile(gamma_display[union], args.vmax_percentile, default=1.0),
    )
    rate_limits = _safe_limits(
        _finite_percentile(rate_display[union], args.vmin_percentile, default=0.0),
        _finite_percentile(rate_display[union], args.vmax_percentile, default=1.0),
    )
    lifetime_limits = _safe_limits(
        _finite_percentile(log_lifetime[union], args.vmin_percentile, default=0.0),
        _finite_percentile(log_lifetime[union], args.vmax_percentile, default=1.0),
    )
    absolute_error_limits = _safe_limits(
        0.0,
        _finite_percentile(absolute_error[union], args.error_percentile, default=1.0),
    )
    relative_error_limits = _safe_limits(
        0.0,
        _finite_percentile(relative_error[union], args.error_percentile, default=100.0),
    )
    energy_split_limit = _finite_percentile(
        np.abs(energy_split[union]), args.error_percentile, default=1.0
    )
    gamma_split_limit = _finite_percentile(
        np.abs(gamma_split[union]), args.error_percentile, default=1.0
    )
    energy_split_limits = _safe_limits(-energy_split_limit, energy_split_limit)
    gamma_split_limits = _safe_limits(-gamma_split_limit, gamma_split_limit)
    channel_labels = [f"channel {channel}" for channel in range(physical_count)]

    _style()
    outputs = []
    for selected, mask in slices:
        slice_tag = _format_slice(selected)
        slice_title = rf"$k_{{{normal_axis + 1}}}={selected:.4g}$"
        if "linewidth" in plot_types:
            outputs.extend(
                _plot_channels(
                    values=gamma_display,
                    kpoints=data["kpoints_frac"],
                    mask=mask,
                    lattice=lattice,
                    plane_axes=plane_axes,
                    title=rf"Magnon linewidth, {slice_title}",
                    channel_labels=channel_labels,
                    colorbar_label=rf"$\Gamma=-\mathrm{{Im}}\,\Sigma^R$ ({args.linewidth_unit})",
                    cmap=args.linewidth_cmap,
                    limits=gamma_limits,
                    args=args,
                    output_dir=output_dir,
                    stem=f"linewidth.kz{slice_tag}",
                )
            )
        if "scattering-rate" in plot_types:
            outputs.extend(
                _plot_channels(
                    values=rate_display,
                    kpoints=data["kpoints_frac"],
                    mask=mask,
                    lattice=lattice,
                    plane_axes=plane_axes,
                    title=rf"Magnon scattering rate, {slice_title}",
                    channel_labels=channel_labels,
                    colorbar_label=rf"$\tau^{{-1}}=2\Gamma/\hbar$ ({args.scattering_rate_unit})",
                    cmap=args.scattering_rate_cmap,
                    limits=rate_limits,
                    args=args,
                    output_dir=output_dir,
                    stem=f"scattering_rate.kz{slice_tag}",
                )
            )
        if "lifetime" in plot_types:
            outputs.extend(
                _plot_channels(
                    values=log_lifetime,
                    kpoints=data["kpoints_frac"],
                    mask=mask,
                    lattice=lattice,
                    plane_axes=plane_axes,
                    title=rf"Magnon lifetime, {slice_title}",
                    channel_labels=channel_labels,
                    colorbar_label=r"$\log_{10}[\tau\,/\,\mathrm{ps}]$",
                    cmap=args.lifetime_cmap,
                    limits=lifetime_limits,
                    args=args,
                    output_dir=output_dir,
                    stem=f"lifetime_log10ps.kz{slice_tag}",
                )
            )
        if "rotation-error" in plot_types:
            outputs.extend(
                _plot_rotation_error(
                    data=data,
                    kpoints=data["kpoints_frac"],
                    mask=mask,
                    lattice=lattice,
                    plane_axes=plane_axes,
                    title=rf"Rotation + channel-swap residual, {slice_title}",
                    channel_labels=channel_labels,
                    limits=(absolute_error_limits, relative_error_limits),
                    args=args,
                    output_dir=output_dir,
                    stem=f"rotation_error.kz{slice_tag}",
                )
            )
        if "splitting" in plot_types:
            outputs.extend(
                _plot_channel_splitting(
                    data=data,
                    channels=args.split_channels,
                    kpoints=data["kpoints_frac"],
                    mask=mask,
                    lattice=lattice,
                    plane_axes=plane_axes,
                    title=rf"Channel splitting, {slice_title}",
                    limits=(energy_split_limits, gamma_split_limits),
                    args=args,
                    output_dir=output_dir,
                    stem=f"channel_splitting.kz{slice_tag}",
                )
            )

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "physical_channel_count": physical_count,
        "rotation_direct_frac": np.asarray(rotation, dtype=int).tolist(),
        "translation_frac": np.asarray(translation, dtype=float).tolist(),
        "channel_map": np.asarray(channel_map, dtype=int).tolist(),
        "symmetry_operation_index": operation.index if operation is not None else None,
        "rotation_order": operation.rotation_order if operation is not None else None,
        "plane_axes": list(plane_axes),
        "requested_slices": [float(value) for value in slice_requests],
        "selected_slices": [float(value) for value, _ in slices],
        "plots": sorted(plot_types),
        "bz_mode": str(args.bz_mode),
        "periodic_view": str(args.periodic_view),
        "periodic_repeats": int(args.periodic_repeats),
        "periodic_padding": float(args.periodic_padding),
        "mesh_max_mapping_distance": float(
            np.max(data["momentum_map_distance"], initial=0.0)
        ),
        "linewidth_scale": float(args.linewidth_scale),
        "linewidth_unit": str(args.linewidth_unit),
        "scattering_rate_scale": float(args.scattering_rate_scale),
        "scattering_rate_unit": str(args.scattering_rate_unit),
        "color_limits": {
            "linewidth": list(gamma_limits),
            "scattering_rate": list(rate_limits),
            "log10_lifetime_ps": list(lifetime_limits),
            "rotation_absolute_error": list(absolute_error_limits),
            "rotation_relative_error_percent": list(relative_error_limits),
            "energy_splitting_mev": list(energy_split_limits),
            "linewidth_splitting": list(gamma_split_limits),
        },
        "files": [str(path) for path in outputs],
    }
    summary_path = output_dir / "plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return outputs, summary_path, summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot magnon linewidth, lifetime, rotation residual, and channel splitting"
    )
    parser.add_argument("input", help="NPZ from slw.magph.legacy.reference.lifetime_mpi_dynamic")
    parser.add_argument("--structure", help="Structure/input file for symmetry discovery")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slices", nargs="+", type=float, default=(0.0, 0.5))
    parser.add_argument(
        "--all-slices",
        action="store_true",
        help="Plot every stored plane along the axis normal to --plane-axes",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=(
            "all",
            "linewidth",
            "scattering-rate",
            "lifetime",
            "rotation-error",
            "splitting",
        ),
        default=("all",),
        help="Quantities to generate; use scattering-rate for tau^-1 BZ maps only",
    )
    parser.add_argument("--plane-axes", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--physical-channel-count", type=int)
    parser.add_argument("--magnetic-atom-indices", nargs="+", type=int)
    parser.add_argument("--spin-pattern", nargs="+", type=float)
    parser.add_argument("--rotation", nargs=9, type=int)
    parser.add_argument("--channel-map", nargs="+", type=int)
    parser.add_argument("--symmetry-operation-index", type=int)
    parser.add_argument("--split-channels", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--symprec", type=float, default=1.0e-5)
    parser.add_argument("--atom-tolerance-ang", type=float, default=1.0e-4)
    parser.add_argument("--mesh-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--k-round", type=int, default=8)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--panel-width", type=float, default=4.4)
    parser.add_argument("--panel-height", type=float, default=4.0)
    parser.add_argument("--tile", type=int, default=1)
    parser.add_argument(
        "--bz-mode",
        choices=("clip", "periodic"),
        default="clip",
        help="Clip to the first BZ or display periodic copies with BZ outlines",
    )
    parser.add_argument(
        "--periodic-repeats",
        type=int,
        default=1,
        help="Neighboring BZ repeats for --periodic-view neighbors",
    )
    parser.add_argument(
        "--periodic-view",
        choices=("central", "neighbors"),
        default="central",
        help=(
            "Keep the central first BZ large with periodic data outside, or "
            "show multiple neighboring BZs"
        ),
    )
    parser.add_argument(
        "--periodic-padding",
        type=float,
        default=0.06,
        help="Fractional padding outside the central first-BZ bounding box",
    )
    parser.add_argument("--margin", type=float, default=0.035)
    parser.add_argument("--linewidth-scale", type=float, default=1.0e6)
    parser.add_argument("--linewidth-unit", default="neV")
    parser.add_argument("--scattering-rate-scale", type=float, default=1.0)
    parser.add_argument("--scattering-rate-unit", default=r"ps$^{-1}$")
    parser.add_argument("--vmin-percentile", type=float, default=1.0)
    parser.add_argument("--vmax-percentile", type=float, default=99.0)
    parser.add_argument("--error-percentile", type=float, default=99.0)
    parser.add_argument("--linewidth-cmap", default="magma")
    parser.add_argument("--lifetime-cmap", default="viridis")
    parser.add_argument("--scattering-rate-cmap", default="magma")
    parser.add_argument("--error-cmap", default="cividis")
    parser.add_argument("--diverging-cmap", default="RdBu_r")
    parser.add_argument("--edgecolor", default="none")
    parser.add_argument("--linewidth", type=float, default=0.0)
    parser.add_argument("--bz-color", default="0.25")
    parser.add_argument("--bz-lw", type=float, default=1.1)
    parser.add_argument("--neighbor-bz-lw", type=float, default=0.65)
    parser.add_argument("--neighbor-bz-alpha", type=float, default=0.55)
    return parser


def main():
    outputs, summary_path, summary = plot_lifetime_symmetry(
        build_parser().parse_args()
    )
    print(
        "[magph:lifetime-plot] "
        f"operation={summary['symmetry_operation_index']} "
        f"channel_map={summary['channel_map']} "
        f"slices={summary['selected_slices']}"
    )
    for output in outputs:
        print(f"[magph:lifetime-plot] wrote {output}")
    print(f"[magph:lifetime-plot] wrote {summary_path}")


if __name__ == "__main__":
    main()
