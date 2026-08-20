"""Map phonon q for a fixed initial magnon momentum and branch."""

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

from .. import kernels
from ..adapter import build_legacy_djdu_tuple, load_djr_h5, load_j_payload, load_phonon_cache
from .hybrid import _load_structure
from .hybrid_berry import (
    _first_bz_polygon_2d,
    _periodic_voronoi_polygons,
    reciprocal_plane_basis_2d,
)
from ..lswt import parse_spin_pattern
from ..plot_coupling_bz import _finite_percentile
from ..plot_coupling_kbz import (
    _plane_points,
    _select_plane_slice,
    _slice_axis,
    _validate_plane_axes,
)
from .plot_scattering_kbz import _magnetic_count, _selected_initial_modes, save_fixed_q_gbar
from .solver_mpi import _canonical_bonds_for_fm
from ..scattering import build_atomic_gauge_lambda, g_kq_loop_atomic_gauge


def reduce_scattering_qgrid(vertex, *, nphysical: int | None = None):
    """Reduce G[q,nu,m,n] to sqrt(sum_(m,nu)|G|^2/Nm) at every q and n."""
    values = np.asarray(vertex, dtype=np.complex128)
    if values.ndim != 4:
        raise ValueError(f"Expected vertex shape (nq,nphonon,nfinal,ninitial), got {values.shape}")
    mode_count = values.shape[2] if nphysical is None else int(nphysical)
    if mode_count < 1 or mode_count > values.shape[2] or mode_count > values.shape[3]:
        raise ValueError(f"Invalid nphysical={mode_count} for vertex shape {values.shape}")
    physical = values[:, :, :mode_count, :mode_count]
    norm_squared = np.sum(np.abs(physical) ** 2, axis=(1, 2))
    return np.sqrt(norm_squared / float(mode_count))


def _load_inputs(args):
    lattice, _, atom_pos = _load_structure(args.structure, fallback_h5=args.jr)
    lattice = np.asarray(lattice, dtype=np.float64)
    atom_pos = np.asarray(atom_pos, dtype=np.float64)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T

    raw_j = load_j_payload(args.jr)
    j_tuple = (
        np.asarray(raw_j[0], dtype=np.float64),
        np.asarray(raw_j[1], dtype=np.int32),
        np.asarray(raw_j[2], dtype=np.int32),
        np.asarray(raw_j[3], dtype=np.int32).reshape(-1, 3),
        np.asarray(raw_j[4], dtype=np.float64),
    )
    nmag = _magnetic_count(j_tuple)
    largest_atom_index = int(max(np.max(j_tuple[1]), np.max(j_tuple[2])))
    if atom_pos.shape[0] <= largest_atom_index:
        raise ValueError(f"Structure has too few atoms for J indices: atoms={atom_pos.shape[0]}, nmag={nmag}")

    dj_payload = load_djr_h5(args.djr)
    djdu = build_legacy_djdu_tuple(dj_payload, rp_idx=tuple(args.rp_idx))
    phonon = load_phonon_cache(args.phonon_cache)
    qpts_frac = np.asarray(phonon["q_mesh_flat_frac"], dtype=np.float64) % 1.0
    qpts_cart = np.asarray(phonon["q_mesh_flat_cart"], dtype=np.float64)
    ph_energy = np.asarray(phonon["ph_en_flat"], dtype=np.float64)
    ph_vector = np.asarray(phonon["ph_vec_flat"], dtype=np.complex128)
    if qpts_frac.shape != qpts_cart.shape or qpts_frac.ndim != 2 or qpts_frac.shape[1] != 3:
        raise ValueError(f"Invalid phonon q-point shapes: frac={qpts_frac.shape}, cart={qpts_cart.shape}")
    if ph_energy.shape[:1] != qpts_frac.shape[:1] or ph_vector.shape[:1] != qpts_frac.shape[:1]:
        raise ValueError(
            f"Phonon cache q mismatch: q={qpts_frac.shape}, energy={ph_energy.shape}, vector={ph_vector.shape}"
        )
    return lattice, atom_pos, reciprocal, j_tuple, nmag, dj_payload, djdu, qpts_frac, qpts_cart, ph_energy, ph_vector


def compute_fixed_magnon_q_gbar(args):
    """Fix initial magnon Q and calculate the coupling norm on the phonon q mesh."""
    if args.nproc is not None:
        from numba import set_num_threads

        set_num_threads(int(args.nproc))

    (
        lattice,
        atom_pos,
        reciprocal,
        j_tuple,
        nmag,
        dj_payload,
        djdu,
        qpts_frac,
        qpts_cart,
        ph_energy,
        ph_vector,
    ) = _load_inputs(args)
    magnon_q_frac = np.asarray(args.magnon_q, dtype=np.float64) % 1.0
    magnon_q_cart = magnon_q_frac @ reciprocal

    started = time.perf_counter()
    vertex_report = {"formalism": "legacy_fm"}
    if args.mag_order == "fm":
        bond_vectors = np.asarray(djdu[3], dtype=np.float64) @ lattice
        bond_vectors += atom_pos[np.asarray(djdu[2], dtype=np.int32)]
        bond_vectors -= atom_pos[np.asarray(djdu[1], dtype=np.int32)]
        exp_iqd = kernels.precompute_phase_factors(qpts_cart, bond_vectors)
        exp_ikd = kernels.precompute_phase_factors(magnon_q_cart.reshape(1, 3), bond_vectors)
        j_kernel = _canonical_bonds_for_fm(j_tuple)
        _, initial_modes = kernels.Magnon_Hamiltonian_fm_normal(
            magnon_q_cart,
            float(args.S),
            j_kernel,
            lattice,
            atom_pos,
            nmag,
            float(args.anisotropy_mev),
            float(args.bond_factor),
        )
        vertex, _ = kernels.g_kq_loop_fm_normal(
            ik=0,
            exp_iqd=exp_iqd,
            exp_ikd=exp_ikd,
            total_q=qpts_cart.shape[0],
            q_mesh_flat_cart=qpts_cart,
            k_cart=magnon_q_cart,
            dJdu=djdu,
            Uk=initial_modes,
            ph_en_flat=ph_energy,
            ph_vec_flat=ph_vector,
            S=float(args.S),
            J0=j_kernel,
            R_vec=lattice,
            G_vec=reciprocal,
            atom_pos=atom_pos,
            nmag=nmag,
            anisotropy=float(args.anisotropy_mev),
            bond_factor=float(args.bond_factor),
        )
        physical_count = nmag
        physical_only = True
    else:
        if nmag != 2:
            raise ValueError(f"The AFM lifetime kernel requires two magnetic sublattices, found nmag={nmag}")
        spin_pattern = parse_spin_pattern(args.spin_pattern, nmag)
        atom_frac = atom_pos @ np.linalg.inv(lattice)
        lambda_qnu_b, vertex_report = build_atomic_gauge_lambda(
            dj_payload,
            qpts_frac,
            ph_energy,
            ph_vector,
            atom_frac,
            phonon_floor_mev=float(args.phonon_floor_mev),
            asr_mode=str(args.dJ_asr),
            asr_tolerance=float(args.dJ_asr_tolerance),
            q_chunk=int(args.vertex_q_chunk),
            bond_chunk=int(args.vertex_bond_chunk),
        )
        _, initial_modes = kernels.Magnon_Hamiltonian_v1(
            magnon_q_cart,
            float(args.S),
            j_tuple,
            lattice,
            atom_pos,
            spin_pattern,
            float(args.bond_factor),
            float(args.anisotropy_mev),
        )
        vertex, _ = g_kq_loop_atomic_gauge(
            q_mesh_flat_cart=qpts_cart,
            k_cart=magnon_q_cart,
            lambda_qnu_b=lambda_qnu_b,
            pair_R=np.asarray(dj_payload["pair_R"], dtype=np.int32),
            Uk=initial_modes,
            S=float(args.S),
            J0=j_tuple,
            lattice=lattice,
            atom_pos=atom_pos,
            spin_pattern=spin_pattern,
            anisotropy=float(args.anisotropy_mev),
            bond_factor=float(args.bond_factor),
        )
        physical_count = nmag if args.physical_only else 2 * nmag
        physical_only = bool(args.physical_only)

    gbar = reduce_scattering_qgrid(vertex, nphysical=physical_count)
    return {
        "gbar_n": gbar,
        "qpts_frac": qpts_frac,
        "qpts_cart": qpts_cart,
        "magnon_q_frac": magnon_q_frac,
        "magnon_q_cart": magnon_q_cart,
        "n_final_modes": np.asarray(physical_count, dtype=np.int32),
        "n_phonon_modes": np.asarray(ph_energy.shape[1], dtype=np.int32),
        "physical_only": np.asarray(physical_only),
        "mag_order": np.asarray(args.mag_order),
        "lattice_ang": lattice,
        "runtime_s": np.asarray(time.perf_counter() - started, dtype=np.float64),
        "vertex_report_json": np.asarray(json.dumps(vertex_report, sort_keys=True)),
    }


def plot_fixed_magnon_q_gbar(args, result):
    plane_axes = _validate_plane_axes(args.plane_axes)
    normal_axis = _slice_axis(plane_axes)
    mask, selected_slice = _select_plane_slice(
        result["qpts_frac"], axis=normal_axis, value=args.slice_value, decimals=args.q_round
    )
    q_slice = np.asarray(result["qpts_frac"])[mask]
    values = np.asarray(result["gbar_n"])[mask]
    if q_slice.shape[0] < 4:
        raise RuntimeError(f"Selected phonon q-plane has too few points for Voronoi plotting: {q_slice.shape[0]}")

    lattice = np.asarray(result["lattice_ang"], dtype=np.float64)
    b1, b2 = reciprocal_plane_basis_2d(lattice, plane_axes)
    points = _plane_points(q_slice, plane_axes, b1, b2)
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
    boundary_points = bz_polygon if bz_polygon is not None else points
    minimum = np.min(boundary_points, axis=0)
    maximum = np.max(boundary_points, axis=0)
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
        axis.set_xlabel(rf"$q_{plane_axes[0] + 1}$ ($\AA^{{-1}}$)")
        axis.set_ylabel(rf"$q_{plane_axes[1] + 1}$ ($\AA^{{-1}}$)")
        axis.set_title(f"initial magnon n={mode + args.mode_base}")
    for panel in range(len(modes), axes.size):
        axes.flat[panel].set_visible(False)

    magnon_q_text = ", ".join(f"{value:.5g}" for value in np.asarray(result["magnon_q_frac"]))
    figure.suptitle(
        args.title
        or rf"$\bar{{G}}_{{n,\mathbf{{Q}}}}(\mathbf{{q}})$, magnon Q=({magnon_q_text}), q-slice={selected_slice:.5g}"
    )
    figure.colorbar(
        collection,
        ax=list(axes.flat[: len(modes)]),
        label=r"$\sqrt{\sum_{m\nu}|G_{nm\nu}|^2/N_m}$ (meV)",
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description="Map phonon q coupling for a fixed initial magnon momentum Q and branch n"
    )
    jr_group = parser.add_mutually_exclusive_group(required=True)
    jr_group.add_argument("--jr", help="Static Jr HDF5 (preferred) or prepared legacy J NPZ")
    jr_group.add_argument("--j-cache", dest="jr", help=argparse.SUPPRESS)
    djr_group = parser.add_mutually_exclusive_group(required=True)
    djr_group.add_argument("--djr", help="Real-space dJr HDF5")
    djr_group.add_argument("--dj-tensor-h5", dest="djr", help=argparse.SUPPRESS)
    parser.add_argument("--phonon-cache", required=True, help="Prepared phonon q-mesh NPZ")
    parser.add_argument("--structure", default=None, help="Structure file; optional when --jr embeds lattice and positions")
    parser.add_argument(
        "--magnon-q",
        type=float,
        nargs=3,
        required=True,
        metavar=("Q1", "Q2", "Q3"),
        help="Fixed fractional momentum of the initial magnon",
    )
    parser.add_argument("--mag-order", choices=("afm", "fm"), required=True)
    parser.add_argument("--S", type=float, required=True, help="Local spin magnitude")
    parser.add_argument("--spin-pattern", default="auto")
    parser.add_argument(
        "--physical-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For AFM, retain only positive-energy Nambu channels",
    )
    parser.add_argument("--anisotropy-mev", type=float, default=0.0)
    parser.add_argument("--bond-factor", type=float, default=1.0)
    parser.add_argument("--rp-idx", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--phonon-floor-mev", type=float, default=1.0e-3)
    parser.add_argument("--dJ-asr", choices=("none", "check", "project"), default="check")
    parser.add_argument("--dJ-asr-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--vertex-q-chunk", type=int, default=32)
    parser.add_argument("--vertex-bond-chunk", type=int, default=64)
    parser.add_argument("--nproc", type=int, default=None, help="Number of Numba threads")
    parser.add_argument("--initial-mode", type=int, nargs="+", default=None, help="Initial magnon modes to plot; default plots all")
    parser.add_argument("--mode-base", type=int, choices=(0, 1), default=1)
    parser.add_argument("--data-output", default=None, help="Output NPZ; default is OUTPUT with suffix .npz")
    parser.add_argument("-o", "--output", required=True, help="Output image")
    parser.add_argument("--plane-axes", type=int, nargs=2, default=(0, 1), metavar=("A1", "A2"))
    parser.add_argument("--slice", dest="slice_value", type=float, default=0.0)
    parser.add_argument("--q-round", type=int, default=8)
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
    result = compute_fixed_magnon_q_gbar(args)
    data_output = Path(args.data_output) if args.data_output else Path(args.output).with_suffix(".npz")
    data_path = save_fixed_q_gbar(data_output, result)
    plot_path = plot_fixed_magnon_q_gbar(args, result)
    print(
        "[magph-scattering-qbz] "
        f"plot={plot_path}, data={data_path}, magnon_q={result['magnon_q_frac'].tolist()}, "
        f"nq={result['qpts_frac'].shape[0]}, shape={result['gbar_n'].shape}, "
        f"runtime_s={float(result['runtime_s']):.3f}"
    )


if __name__ == "__main__":
    main()
