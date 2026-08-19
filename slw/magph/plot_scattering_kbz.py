"""Compute and plot branch-resolved fixed-Q magnon-phonon scattering norms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from . import kernels
from .adapter import build_legacy_djdu_tuple, load_djr_h5, load_j_payload, load_phonon_cache
from .hybrid import _load_structure
from .hybrid_berry import (
    _first_bz_polygon_2d,
    _periodic_voronoi_polygons,
    reciprocal_plane_basis_2d,
)
from .lswt import parse_spin_pattern
from .plot_coupling_bz import _finite_percentile, _mode_index
from .plot_coupling_kbz import (
    _plane_points,
    _select_plane_slice,
    _select_qpoint,
    _slice_axis,
    _validate_plane_axes,
)
from .utils import generate_k_mesh_flat
from .scattering import build_atomic_gauge_lambda, fixed_q_gbar_atomic_gauge


def reduce_scattering_vertex(vertex, *, nfinal: int | None = None):
    """Return sqrt(sum_(m,nu)|G_nu,m,n|^2/Nm), retaining initial mode n."""
    values = np.asarray(vertex, dtype=np.complex128)
    if values.ndim != 3:
        raise ValueError(f"Expected vertex shape (nphonon,nfinal,ninitial), got {values.shape}")
    final_count = values.shape[1] if nfinal is None else int(nfinal)
    if final_count < 1 or final_count > values.shape[1] or final_count > values.shape[2]:
        raise ValueError(f"Invalid nfinal={final_count} for vertex shape {values.shape}")
    norm_squared = np.sum(np.abs(values[:, :final_count, :final_count]) ** 2, axis=(0, 1))
    return np.sqrt(norm_squared / float(final_count))


def _positive_mesh(values):
    mesh = tuple(int(value) for value in values)
    if len(mesh) != 3 or any(value <= 0 for value in mesh):
        raise ValueError(f"kmesh must contain three positive integers, got {mesh}")
    return mesh


def _magnetic_count(j_tuple) -> int:
    if len(j_tuple[0]) == 0:
        raise ValueError("Static J payload contains no bonds")
    return int(max(np.max(j_tuple[1]), np.max(j_tuple[2])) + 1)


def compute_fixed_q_gbar(args):
    """Compute the requested fixed-Q quantity using the lifetime scattering vertex."""
    lattice, _, atom_pos = _load_structure(args.structure, fallback_h5=args.jr)
    lattice = np.asarray(lattice, dtype=np.float64)
    atom_pos = np.asarray(atom_pos, dtype=np.float64)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T

    j_tuple = load_j_payload(args.jr)
    j_tuple = (
        np.asarray(j_tuple[0], dtype=np.float64),
        np.asarray(j_tuple[1], dtype=np.int32),
        np.asarray(j_tuple[2], dtype=np.int32),
        np.asarray(j_tuple[3], dtype=np.int32).reshape(-1, 3),
        np.asarray(j_tuple[4], dtype=np.float64),
    )
    nmag = _magnetic_count(j_tuple)
    if atom_pos.shape[0] <= max(np.max(j_tuple[1]), np.max(j_tuple[2])):
        raise ValueError(f"Structure has too few atoms for J indices: atoms={atom_pos.shape[0]}, nmag={nmag}")

    dj_payload = load_djr_h5(args.djr)
    djdu = build_legacy_djdu_tuple(dj_payload, rp_idx=tuple(args.rp_idx))
    phonon = load_phonon_cache(args.phonon_cache)
    qpts_frac = np.asarray(phonon["q_mesh_flat_frac"], dtype=np.float64)
    iq, selected_q, q_distance = _select_qpoint(qpts_frac, q=args.q, reciprocal=reciprocal)
    q_cart = np.asarray(phonon["q_mesh_flat_cart"][iq], dtype=np.float64)
    ph_freq = np.asarray(phonon["ph_en_flat"][iq], dtype=np.float64)
    ph_pol = np.asarray(phonon["ph_vec_flat"][iq], dtype=np.complex128)

    kmesh = _positive_mesh(args.kmesh)
    kpts_frac = generate_k_mesh_flat(*kmesh, shift=args.shift_kmesh)
    kpts_cart = kpts_frac @ reciprocal
    if args.nproc is not None:
        from numba import set_num_threads

        set_num_threads(int(args.nproc))

    started = time.perf_counter()
    vertex_report = {"formalism": "legacy_fm"}
    if args.mag_order == "fm":
        bond_vectors = np.asarray(djdu[3], dtype=np.float64) @ lattice
        bond_vectors += atom_pos[np.asarray(djdu[2], dtype=np.int32)]
        bond_vectors -= atom_pos[np.asarray(djdu[1], dtype=np.int32)]
        exp_iqd = kernels.precompute_phase_factors(q_cart.reshape(1, 3), bond_vectors)
        exp_ikd = kernels.precompute_phase_factors(kpts_cart, bond_vectors)
        gbar = kernels.fixed_q_gbar_fm_normal(
            kpts_cart,
            q_cart,
            exp_iqd,
            exp_ikd,
            ph_freq,
            ph_pol,
            djdu,
            float(args.S),
            j_tuple,
            lattice,
            atom_pos,
            nmag,
            float(args.anisotropy_mev),
            float(args.bond_factor),
        )
        physical_only = True
    else:
        if nmag != 2:
            raise ValueError(f"The AFM lifetime kernel requires two magnetic sublattices, found nmag={nmag}")
        spin_pattern = parse_spin_pattern(args.spin_pattern, nmag)
        atom_frac = atom_pos @ np.linalg.inv(lattice)
        lambda_one_q, vertex_report = build_atomic_gauge_lambda(
            dj_payload,
            np.asarray(selected_q, dtype=np.float64).reshape(1, 3),
            ph_freq.reshape(1, -1),
            ph_pol.reshape(1, *ph_pol.shape),
            atom_frac,
            phonon_floor_mev=float(args.phonon_floor_mev),
            asr_mode=str(args.dJ_asr),
            asr_tolerance=float(args.dJ_asr_tolerance),
            q_chunk=1,
            bond_chunk=int(args.vertex_bond_chunk),
        )
        gbar = fixed_q_gbar_atomic_gauge(
            kpts_cart,
            q_cart,
            lambda_one_q[0],
            np.asarray(dj_payload["pair_R"], dtype=np.int32),
            float(args.S),
            j_tuple,
            lattice,
            atom_pos,
            spin_pattern,
            float(args.anisotropy_mev),
            float(args.bond_factor),
            bool(args.physical_only),
        )
        physical_only = bool(args.physical_only)

    return {
        "gbar_n": np.asarray(gbar, dtype=np.float64),
        "kpts_frac": kpts_frac,
        "kpts_cart": kpts_cart,
        "q_requested_frac": np.asarray(args.q, dtype=np.float64),
        "q_selected_frac": selected_q,
        "q_selected_cart": q_cart,
        "q_index": np.asarray(iq, dtype=np.int64),
        "q_distance_inv_ang": np.asarray(q_distance, dtype=np.float64),
        "kmesh": np.asarray(kmesh, dtype=np.int32),
        "n_final_modes": np.asarray(gbar.shape[1], dtype=np.int32),
        "n_phonon_modes": np.asarray(ph_freq.shape[0], dtype=np.int32),
        "physical_only": np.asarray(physical_only),
        "mag_order": np.asarray(args.mag_order),
        "lattice_ang": lattice,
        "runtime_s": np.asarray(time.perf_counter() - started, dtype=np.float64),
        "vertex_report_json": np.asarray(json.dumps(vertex_report, sort_keys=True)),
    }


def save_fixed_q_gbar(path, result):
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return output


def _selected_initial_modes(args, nmode):
    if args.initial_mode is None:
        return list(range(int(nmode)))
    return [_mode_index(value, args.mode_base, nmode, "initial_mode") for value in args.initial_mode]


def plot_fixed_q_gbar(args, result):
    plane_axes = _validate_plane_axes(args.plane_axes)
    normal_axis = _slice_axis(plane_axes)
    mask, selected_slice = _select_plane_slice(
        result["kpts_frac"], axis=normal_axis, value=args.slice_value, decimals=args.k_round
    )
    k_slice = np.asarray(result["kpts_frac"])[mask]
    values = np.asarray(result["gbar_n"])[mask]
    if k_slice.shape[0] < 4:
        raise RuntimeError(f"Selected k-plane has too few points for Voronoi plotting: {k_slice.shape[0]}")

    lattice = np.asarray(result["lattice_ang"], dtype=np.float64)
    b1, b2 = reciprocal_plane_basis_2d(lattice, plane_axes)
    points = _plane_points(k_slice, plane_axes, b1, b2)
    polygons, polygon_values = _periodic_voronoi_polygons(points, values, b1, b2, args.tile)
    if not polygons:
        raise RuntimeError("Voronoi construction produced no finite cells")

    modes = _selected_initial_modes(args, values.shape[1])
    displayed = polygon_values[:, modes]
    vmin = args.vmin if args.vmin is not None else _finite_percentile(displayed, args.vmin_percentile)
    vmax = args.vmax if args.vmax is not None else _finite_percentile(displayed, args.vmax_percentile)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if abs(float(vmax) - float(vmin)) < np.finfo(np.float64).eps:
        padding = max(abs(float(vmax)), 1.0) * 1.0e-6
        vmin, vmax = float(vmin) - padding, float(vmax) + padding

    ncols = min(len(modes), max(1, int(math.ceil(math.sqrt(len(modes))))))
    nrows = int(math.ceil(len(modes) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(args.panel_width * ncols, args.panel_height * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    bz_polygon = _first_bz_polygon_2d(b1, b2)
    minimum = np.min(bz_polygon if bz_polygon is not None else points, axis=0)
    maximum = np.max(bz_polygon if bz_polygon is not None else points, axis=0)
    span = np.maximum(maximum - minimum, np.finfo(np.float64).eps)
    collection = None
    for panel, mode in enumerate(modes):
        axis = axes.flat[panel]
        collection = PolyCollection(
            polygons,
            array=polygon_values[:, mode],
            cmap=args.cmap,
            edgecolor=args.edgecolor,
            linewidth=args.linewidth,
        )
        collection.set_clim(float(vmin), float(vmax))
        axis.add_collection(collection)
        if bz_polygon is not None:
            clip = plt.Polygon(bz_polygon, closed=True, facecolor="none", edgecolor="none")
            axis.add_patch(clip)
            collection.set_clip_path(clip)
            closed = np.vstack((bz_polygon, bz_polygon[0]))
            axis.plot(closed[:, 0], closed[:, 1], color=args.bz_color, lw=args.bz_lw)
        axis.set_xlim(minimum[0] - args.margin * span[0], maximum[0] + args.margin * span[0])
        axis.set_ylim(minimum[1] - args.margin * span[1], maximum[1] + args.margin * span[1])
        axis.set_aspect("equal")
        axis.set_xlabel(rf"$k_{plane_axes[0] + 1}$ ($\AA^{{-1}}$)")
        axis.set_ylabel(rf"$k_{plane_axes[1] + 1}$ ($\AA^{{-1}}$)")
        axis.set_title(f"initial magnon n={mode + args.mode_base}")
    for panel in range(len(modes), axes.size):
        axes.flat[panel].set_visible(False)

    q_text = ", ".join(f"{value:.5g}" for value in np.asarray(result["q_selected_frac"]))
    figure.suptitle(args.title or rf"$\bar{{G}}_{{n,\mathbf{{Q}}}}(\mathbf{{k}})$, Q=({q_text}), slice={selected_slice:.5g}")
    figure.colorbar(collection, ax=list(axes.flat[: len(modes)]), label=r"$\sqrt{\sum_{m\nu}|G_{nm\nu}|^2/N_m}$ (meV)")
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description="Compute and plot fixed-Q, n-resolved magnon-phonon scattering strength")
    jr_group = parser.add_mutually_exclusive_group(required=True)
    jr_group.add_argument("--jr", help="Static Jr HDF5 (preferred) or prepared legacy J NPZ")
    jr_group.add_argument("--j-cache", dest="jr", help=argparse.SUPPRESS)
    djr_group = parser.add_mutually_exclusive_group(required=True)
    djr_group.add_argument("--djr", help="Real-space dJr HDF5")
    djr_group.add_argument("--dj-tensor-h5", dest="djr", help=argparse.SUPPRESS)
    parser.add_argument("--phonon-cache", required=True, help="Prepared phonon mesh NPZ")
    parser.add_argument("--structure", default=None, help="Structure file; optional when --jr HDF5 embeds lattice and positions")
    parser.add_argument("--q", type=float, nargs=3, required=True, metavar=("Q1", "Q2", "Q3"), help="Requested fractional phonon momentum")
    parser.add_argument("--kmesh", type=int, nargs=3, required=True, metavar=("NK1", "NK2", "NK3"))
    parser.add_argument("--shift-kmesh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mag-order", choices=("afm", "fm"), default="afm")
    parser.add_argument("--S", type=float, default=2.5)
    parser.add_argument("--spin-pattern", default="auto")
    parser.add_argument("--physical-only", action=argparse.BooleanOptionalAction, default=True, help="For AFM, use only the positive-energy half of the Nambu channels")
    parser.add_argument("--anisotropy-mev", type=float, default=0.0)
    parser.add_argument("--bond-factor", type=float, default=1.0)
    parser.add_argument("--rp-idx", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--phonon-floor-mev", type=float, default=1.0e-3)
    parser.add_argument("--dJ-asr", choices=("none", "check", "project"), default="check")
    parser.add_argument("--dJ-asr-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--vertex-bond-chunk", type=int, default=64)
    parser.add_argument("--nproc", type=int, default=None, help="Number of Numba threads; default uses Numba's configured maximum")
    parser.add_argument("--initial-mode", type=int, nargs="+", default=None, help="Initial magnon modes to plot; default plots all")
    parser.add_argument("--mode-base", type=int, choices=(0, 1), default=1)
    parser.add_argument("--data-output", default=None, help="Output NPZ; default is OUTPUT with suffix .npz")
    parser.add_argument("-o", "--output", required=True, help="Output image")
    parser.add_argument("--plane-axes", type=int, nargs=2, default=(0, 1), metavar=("A1", "A2"))
    parser.add_argument("--slice", dest="slice_value", type=float, default=0.0)
    parser.add_argument("--k-round", type=int, default=8)
    parser.add_argument("--tile", type=int, default=2)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--vmin-percentile", type=float, default=1.0)
    parser.add_argument("--vmax-percentile", type=float, default=99.0)
    parser.add_argument("--edgecolor", default="none")
    parser.add_argument("--linewidth", type=float, default=0.0)
    parser.add_argument("--bz-color", default="0.35")
    parser.add_argument("--bz-lw", type=float, default=1.2)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--panel-width", type=float, default=5.0)
    parser.add_argument("--panel-height", type=float, default=4.6)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--title", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = compute_fixed_q_gbar(args)
    data_output = Path(args.data_output) if args.data_output else Path(args.output).with_suffix(".npz")
    data_path = save_fixed_q_gbar(data_output, result)
    plot_path = plot_fixed_q_gbar(args, result)
    print(
        "[magph-scattering-kbz] "
        f"plot={plot_path}, data={data_path}, q_index={int(result['q_index'])}, "
        f"q={result['q_selected_frac'].tolist()}, kmesh={result['kmesh'].tolist()}, "
        f"shape={result['gbar_n'].shape}, runtime_s={float(result['runtime_s']):.3f}"
    )


if __name__ == "__main__":
    main()
