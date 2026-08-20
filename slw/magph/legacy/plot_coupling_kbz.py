"""Voronoi map of magnon-phonon coupling over a magnon BZ at fixed phonon q."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from .reference.hybrid import _load_structure
from .reference.hybrid_berry import (
    _first_bz_polygon_2d,
    _periodic_voronoi_polygons,
    reciprocal_plane_basis_2d,
)
from .plot_coupling_bz import _finite_percentile, _mode_index


def _validate_plane_axes(axes) -> tuple[int, int]:
    plane_axes = tuple(int(axis) for axis in axes)
    if len(plane_axes) != 2 or len(set(plane_axes)) != 2:
        raise ValueError(f"plane_axes must contain two different axes, got {plane_axes}")
    if any(axis < 0 or axis > 2 for axis in plane_axes):
        raise ValueError(f"plane_axes entries must be in [0, 2], got {plane_axes}")
    return plane_axes


def _slice_axis(plane_axes: tuple[int, int]) -> int:
    return next(iter({0, 1, 2} - set(plane_axes)))


def _select_plane_slice(kpts_frac, *, axis: int, value: float, decimals: int):
    """Select the periodic mesh plane closest to ``value`` along ``axis``."""
    kpts = np.asarray(kpts_frac, dtype=np.float64) % 1.0
    coord = np.round(kpts[:, int(axis)], int(decimals))
    available = np.unique(coord)
    if available.size == 0:
        raise ValueError("kpts_frac is empty")
    target = float(value) % 1.0
    periodic_distance = np.abs(((available - target + 0.5) % 1.0) - 0.5)
    chosen = float(available[int(np.argmin(periodic_distance))])
    mask = np.isclose(coord, chosen, atol=10.0 ** (-int(decimals)))
    return mask, chosen


def _plane_points(frac_points, axes, b1_2d, b2_2d):
    """Project fractional reciprocal coordinates into a centered 2D plane."""
    frac = np.asarray(frac_points, dtype=np.float64)
    axis1, axis2 = _validate_plane_axes(axes)
    f1 = ((frac[:, axis1] + 0.5) % 1.0) - 0.5
    f2 = ((frac[:, axis2] + 0.5) % 1.0) - 0.5
    return f1[:, None] * b1_2d[None, :] + f2[:, None] * b2_2d[None, :]


def _select_qpoint(qpts_frac, *, iq=None, q=None, reciprocal=None):
    """Resolve a fixed phonon momentum by index or nearest periodic coordinate."""
    qpts = np.asarray(qpts_frac, dtype=np.float64)
    if qpts.ndim != 2 or qpts.shape[1] != 3 or qpts.shape[0] == 0:
        raise ValueError(f"Expected qpts_frac shape (nq,3), got {qpts.shape}")
    if iq is not None and q is not None:
        raise ValueError("Specify only one of iq and q")
    if q is None:
        index = 0 if iq is None else int(iq)
        if index < 0 or index >= qpts.shape[0]:
            raise ValueError(f"iq out of range: {index}, nq={qpts.shape[0]}")
        return index, qpts[index] % 1.0, 0.0

    target = np.asarray(q, dtype=np.float64).reshape(-1)
    if target.size != 3:
        raise ValueError(f"q must contain three fractional coordinates, got {target}")
    delta_frac = ((qpts - target[None, :] + 0.5) % 1.0) - 0.5
    if reciprocal is None:
        delta = delta_frac
    else:
        recip = np.asarray(reciprocal, dtype=np.float64)
        if recip.shape != (3, 3):
            raise ValueError(f"Expected reciprocal lattice shape (3,3), got {recip.shape}")
        delta = delta_frac @ recip
    distances = np.linalg.norm(delta, axis=1)
    index = int(np.argmin(distances))
    return index, qpts[index] % 1.0, float(distances[index])


def _coupling_values_kbz(g, *, iq, phonon_mode, magnon_mode, quantity):
    """Vectorized reduction of g(k,q,phonon,magnon) over the selected modes."""
    arr = np.asarray(g, dtype=np.complex128)
    if arr.ndim != 4:
        raise ValueError(f"Expected magnon_vertex shape (nk,nq,nph,nmag), got {arr.shape}")
    _, nq, nph, nmag = arr.shape
    if iq < 0 or iq >= nq:
        raise ValueError(f"iq out of range: {iq}, nq={nq}")
    if phonon_mode < 0 or phonon_mode >= nph:
        raise ValueError(f"phonon_mode out of range: {phonon_mode}, nph={nph}")

    selected = arr[:, int(iq), int(phonon_mode), :]
    if magnon_mode is None:
        amplitude_squared = np.sum(np.abs(selected) ** 2, axis=1)
        label = rf"$\sqrt{{\sum_m |g_{{\lambda m}}|^2}}$"
    else:
        if magnon_mode < 0 or magnon_mode >= nmag:
            raise ValueError(f"magnon_mode out of range: {magnon_mode}, nmag={nmag}")
        amplitude_squared = np.abs(selected[:, int(magnon_mode)]) ** 2
        label = rf"$|g_{{\lambda,{magnon_mode + 1}}}|$"

    normalized_quantity = str(quantity).strip().lower()
    if normalized_quantity == "norm":
        return np.sqrt(amplitude_squared), f"{label} (meV)"
    if normalized_quantity == "norm2":
        return amplitude_squared, rf"{label}$^2$ (meV$^2$)"
    if normalized_quantity == "log10_abs":
        return np.log10(np.sqrt(amplitude_squared) + np.finfo(np.float64).tiny), rf"$\log_{{10}}$ {label}"
    raise ValueError(f"Unknown quantity={quantity!r}; use norm, norm2, or log10_abs")


def _load_hybrid_arrays(path):
    with np.load(path, allow_pickle=True) as data:
        missing = {"magnon_vertex", "kpts_frac", "qpts_frac"} - set(data.files)
        if missing:
            raise KeyError(f"{path} is missing required arrays: {', '.join(sorted(missing))}")
        vertex = np.asarray(data["magnon_vertex"], dtype=np.complex128)
        kpts = np.asarray(data["kpts_frac"], dtype=np.float64)
        qpts = np.asarray(data["qpts_frac"], dtype=np.float64)
    if vertex.ndim != 4:
        raise ValueError(f"Expected magnon_vertex shape (nk,nq,nph,nmag), got {vertex.shape}")
    if kpts.shape != (vertex.shape[0], 3):
        raise ValueError(f"kpts/magnon_vertex mismatch: kpts={kpts.shape}, vertex={vertex.shape}")
    if qpts.shape != (vertex.shape[1], 3):
        raise ValueError(f"qpts/magnon_vertex mismatch: qpts={qpts.shape}, vertex={vertex.shape}")
    return vertex, kpts % 1.0, qpts % 1.0


def plot_coupling_kbz(args):
    vertex, kpts, qpts = _load_hybrid_arrays(args.input)
    _, _, nph, nmag = vertex.shape
    plane_axes = _validate_plane_axes(args.plane_axes)
    normal_axis = _slice_axis(plane_axes)

    lattice = None
    reciprocal = None
    if args.structure:
        lattice, _, _ = _load_structure(args.structure, fallback_h5=None)
        reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=np.float64)).T

    iq, selected_q, q_distance = _select_qpoint(
        qpts,
        iq=args.iq,
        q=args.q,
        reciprocal=reciprocal,
    )
    phonon_mode = _mode_index(args.phonon_mode, args.mode_base, nph, "phonon_mode")
    magnon_mode = (
        None
        if args.magnon_mode is None
        else _mode_index(args.magnon_mode, args.mode_base, nmag, "magnon_mode")
    )
    values_all, colorbar_label = _coupling_values_kbz(
        vertex,
        iq=iq,
        phonon_mode=phonon_mode,
        magnon_mode=magnon_mode,
        quantity=args.quantity,
    )

    mask, selected_slice = _select_plane_slice(
        kpts,
        axis=normal_axis,
        value=args.slice_value,
        decimals=args.k_round,
    )
    k_slice = kpts[mask]
    values = values_all[mask]
    if k_slice.shape[0] < 4:
        raise RuntimeError(
            "Selected k-plane has too few points for a Voronoi map: "
            f"n={k_slice.shape[0]}. Re-run the hybrid calculation with a 2D/3D kmesh; "
            f"the input contains nk={kpts.shape[0]}."
        )

    unique_plane_points = np.unique(np.round(k_slice[:, plane_axes], args.k_round), axis=0)
    if unique_plane_points.shape[0] != k_slice.shape[0]:
        raise ValueError(
            "Selected k-plane contains duplicate in-plane points; check kpts_frac or increase --k_round. "
            f"points={k_slice.shape[0]}, unique={unique_plane_points.shape[0]}"
        )

    if lattice is None:
        b1 = np.array([1.0, 0.0], dtype=np.float64)
        b2 = np.array([0.0, 1.0], dtype=np.float64)
        xlabel = rf"$k_{plane_axes[0] + 1}$"
        ylabel = rf"$k_{plane_axes[1] + 1}$"
    else:
        b1, b2 = reciprocal_plane_basis_2d(lattice, plane_axes)
        xlabel = rf"$k_{plane_axes[0] + 1}$ ($\AA^{{-1}}$)"
        ylabel = rf"$k_{plane_axes[1] + 1}$ ($\AA^{{-1}}$)"

    points = _plane_points(k_slice, plane_axes, b1, b2)
    polygons, colors = _periodic_voronoi_polygons(points, values, b1, b2, args.tile)
    if len(polygons) == 0:
        raise RuntimeError("Voronoi construction produced no finite cells")

    vmin = args.vmin if args.vmin is not None else _finite_percentile(colors, args.vmin_percentile)
    vmax = args.vmax if args.vmax is not None else _finite_percentile(colors, args.vmax_percentile)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if abs(float(vmax) - float(vmin)) < np.finfo(np.float64).eps:
        padding = max(abs(float(vmax)), 1.0) * 1.0e-6
        vmin, vmax = float(vmin) - padding, float(vmax) + padding

    figure, axis = plt.subplots(figsize=(args.fig_width, args.fig_height))
    collection = PolyCollection(
        polygons,
        array=colors,
        cmap=args.cmap,
        edgecolor=args.edgecolor,
        linewidth=args.linewidth,
    )
    collection.set_clim(float(vmin), float(vmax))
    axis.add_collection(collection)

    bz_polygon = _first_bz_polygon_2d(b1, b2)
    if bz_polygon is not None:
        clip_patch = plt.Polygon(bz_polygon, closed=True, facecolor="none", edgecolor="none")
        axis.add_patch(clip_patch)
        collection.set_clip_path(clip_patch)
        closed_bz = np.vstack((bz_polygon, bz_polygon[0]))
        axis.plot(closed_bz[:, 0], closed_bz[:, 1], color=args.bz_color, lw=args.bz_lw, alpha=args.bz_alpha)
        minimum = np.min(bz_polygon, axis=0)
        maximum = np.max(bz_polygon, axis=0)
    else:
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)

    span = np.maximum(maximum - minimum, np.finfo(np.float64).eps)
    axis.set_xlim(minimum[0] - args.margin * span[0], maximum[0] + args.margin * span[0])
    axis.set_ylim(minimum[1] - args.margin * span[1], maximum[1] + args.margin * span[1])
    axis.set_aspect("equal")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if args.title:
        title = args.title
    else:
        mode_text = "all magnons" if magnon_mode is None else f"magnon {args.magnon_mode}"
        q_text = ", ".join(f"{coordinate:.5g}" for coordinate in selected_q)
        title = f"q=({q_text}), phonon {args.phonon_mode}, {mode_text}"
    axis.set_title(title)
    figure.colorbar(collection, ax=axis, label=colorbar_label)
    figure.tight_layout()

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)
    print(
        "[magph-coupling-kbz] "
        f"input={args.input}, output={output}, iq={iq}, q={selected_q.tolist()}, "
        f"q_distance={q_distance:.6g}, plane_axes={plane_axes}, "
        f"slice_axis={normal_axis}, slice={selected_slice:.8g}, n_slice={int(mask.sum())}, "
        f"value[min,max]=({float(np.nanmin(values)):.6g},{float(np.nanmax(values)):.6g})"
    )
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot g(k,q) over a magnon BZ plane for one fixed phonon momentum and mode"
    )
    parser.add_argument("input", help="hybrid.npz containing magnon_vertex, kpts_frac, and qpts_frac")
    parser.add_argument("-o", "--output", required=True, help="Output image path")
    parser.add_argument("--structure", default=None, help="Structure file used to construct the Cartesian first BZ")
    q_group = parser.add_mutually_exclusive_group()
    q_group.add_argument("--iq", type=int, default=None, help="Zero-based q index; defaults to 0")
    q_group.add_argument("--q", type=float, nargs=3, metavar=("Q1", "Q2", "Q3"), help="Fractional q; nearest periodic mesh point is used")
    parser.add_argument("--phonon_mode", "--phonon-mode", dest="phonon_mode", type=int, required=True)
    parser.add_argument("--magnon_mode", "--magnon-mode", dest="magnon_mode", type=int, default=None, help="Omit to take the norm over all magnon modes")
    parser.add_argument("--mode_base", "--mode-base", dest="mode_base", type=int, choices=(0, 1), default=1)
    parser.add_argument("--plane_axes", "--plane-axes", dest="plane_axes", type=int, nargs=2, default=(0, 1), metavar=("A1", "A2"), help="Fractional reciprocal axes spanning the plotted plane")
    parser.add_argument("--slice", "--kz", dest="slice_value", type=float, default=0.0, help="Fractional coordinate along the axis normal to --plane-axes")
    parser.add_argument("--k_round", "--k-round", dest="k_round", type=int, default=8)
    parser.add_argument("--quantity", choices=("norm", "norm2", "log10_abs"), default="norm")
    parser.add_argument("--tile", type=int, default=2, help="Periodic image range used to construct finite Voronoi cells")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--vmin_percentile", "--vmin-percentile", dest="vmin_percentile", type=float, default=1.0)
    parser.add_argument("--vmax_percentile", "--vmax-percentile", dest="vmax_percentile", type=float, default=99.0)
    parser.add_argument("--edgecolor", default="none")
    parser.add_argument("--linewidth", type=float, default=0.0)
    parser.add_argument("--bz_color", "--bz-color", dest="bz_color", default="0.35")
    parser.add_argument("--bz_lw", "--bz-lw", dest="bz_lw", type=float, default=1.2)
    parser.add_argument("--bz_alpha", "--bz-alpha", dest="bz_alpha", type=float, default=0.95)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--fig_width", "--fig-width", dest="fig_width", type=float, default=5.4)
    parser.add_argument("--fig_height", "--fig-height", dest="fig_height", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--title", default=None)
    return parser


def main(argv=None):
    plot_coupling_kbz(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
