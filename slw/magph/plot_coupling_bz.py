"""Voronoi BZ map of magnon-phonon coupling constants from hybrid.npz."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from .hybrid import _load_structure
from .hybrid_berry import _first_bz_polygon_2d, _periodic_voronoi_polygons, reciprocal_plane_basis_2d


def _finite_percentile(values, pct):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, float(pct)))


def _mode_index(value: int, base: int, size: int, name: str) -> int:
    idx = int(value) - int(base)
    if idx < 0 or idx >= int(size):
        raise ValueError(f"{name} index out of range: value={value}, base={base}, size={size}")
    return idx


def _select_kz_slice(qpts_frac, kz, k_round):
    q = np.asarray(qpts_frac, dtype=np.float64) % 1.0
    kz_round = np.round(q[:, 2], int(k_round))
    kz_vals = np.unique(kz_round)
    target = float(kz) % 1.0
    dist = np.abs(((kz_vals - target + 0.5) % 1.0) - 0.5)
    chosen = kz_vals[int(np.argmin(dist))]
    mask = np.isclose(kz_round, chosen, atol=10.0 ** (-int(k_round)))
    if not np.any(mask):
        raise RuntimeError(f"No q points found for kz={kz}")
    return mask, float(chosen)


def _plane_points_from_q(qpts_frac, b1_2d, b2_2d):
    q = np.asarray(qpts_frac, dtype=np.float64)
    f1 = ((q[:, 0] + 0.5) % 1.0) - 0.5
    f2 = ((q[:, 1] + 0.5) % 1.0) - 0.5
    return f1[:, None] * b1_2d[None, :] + f2[:, None] * b2_2d[None, :]


def _coupling_values(g, *, ik, phonon_mode, magnon_mode, quantity):
    arr = np.asarray(g, dtype=np.complex128)
    if arr.ndim != 4:
        raise ValueError(f"Expected magnon_vertex shape (nk,nq,nph,nmag), got {arr.shape}")
    nk, nq, nph, nmag = arr.shape
    if ik < 0 or ik >= nk:
        raise ValueError(f"ik out of range: {ik}, nk={nk}")
    if phonon_mode < 0 or phonon_mode >= nph:
        raise ValueError(f"phonon_mode out of range: {phonon_mode}, nph={nph}")

    gp = arr[int(ik), :, int(phonon_mode), :]
    if magnon_mode is None:
        amp2 = np.sum(np.abs(gp) ** 2, axis=1)
        label_base = rf"$\sqrt{{\sum_m |g_{{\lambda m}}|^2}}$"
    else:
        if magnon_mode < 0 or magnon_mode >= nmag:
            raise ValueError(f"magnon_mode out of range: {magnon_mode}, nmag={nmag}")
        amp2 = np.abs(gp[:, int(magnon_mode)]) ** 2
        label_base = rf"$|g_{{\lambda,{magnon_mode}}}|$"

    q = str(quantity).strip().lower()
    if q in {"norm", "abs", "amplitude"}:
        return np.sqrt(amp2), f"{label_base} (meV)"
    if q in {"norm2", "abs2", "power"}:
        return amp2, rf"{label_base}$^2$ (meV$^2$)"
    if q in {"log", "log10", "log10_abs"}:
        return np.log10(np.sqrt(amp2) + 1.0e-30), rf"$\log_{{10}}$ {label_base}"
    raise ValueError(f"Unknown quantity={quantity!r}; use norm, norm2, or log10_abs")


def plot_coupling_bz(args):
    data = np.load(args.input, allow_pickle=True)
    if "magnon_vertex" not in data:
        raise KeyError(f"{args.input} does not contain 'magnon_vertex'")
    if "qpts_frac" not in data:
        raise KeyError(f"{args.input} does not contain 'qpts_frac'")

    g = np.asarray(data["magnon_vertex"], dtype=np.complex128)
    qpts = np.asarray(data["qpts_frac"], dtype=np.float64) % 1.0
    _, nq, nph, nmag = g.shape
    if qpts.shape[0] != nq:
        raise ValueError(f"qpts/magnon_vertex mismatch: qpts={qpts.shape}, magnon_vertex={g.shape}")

    ph = _mode_index(args.phonon_mode, args.mode_base, nph, "phonon_mode")
    mag = None if args.magnon_mode is None else _mode_index(args.magnon_mode, args.mode_base, nmag, "magnon_mode")
    vals_all, cbar_label = _coupling_values(
        g,
        ik=int(args.ik),
        phonon_mode=ph,
        magnon_mode=mag,
        quantity=args.quantity,
    )

    mask, kz = _select_kz_slice(qpts, args.kz, args.k_round)
    q_slice = qpts[mask]
    vals = vals_all[mask]
    if q_slice.shape[0] < 4:
        raise RuntimeError(f"kz slice has too few q points for Voronoi plot: n={q_slice.shape[0]}")

    if args.structure:
        lattice, _, _ = _load_structure(args.structure, fallback_h5=None)
        b1, b2 = reciprocal_plane_basis_2d(lattice, (0, 1))
        xlabel = r"$q_x$ ($\AA^{-1}$)"
        ylabel = r"$q_y$ ($\AA^{-1}$)"
    else:
        b1 = np.array([1.0, 0.0], dtype=np.float64)
        b2 = np.array([0.0, 1.0], dtype=np.float64)
        xlabel = r"$q_1$"
        ylabel = r"$q_2$"

    pts = _plane_points_from_q(q_slice, b1, b2)
    polys, colors = _periodic_voronoi_polygons(pts, vals, b1, b2, args.tile)
    if len(polys) == 0:
        raise RuntimeError("Voronoi construction produced no finite cells")

    vmin = args.vmin
    vmax = args.vmax
    if vmin is None:
        vmin = _finite_percentile(colors, args.vmin_percentile)
    if vmax is None:
        vmax = _finite_percentile(colors, args.vmax_percentile)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if abs(float(vmax) - float(vmin)) < 1.0e-14:
        pad = max(abs(float(vmax)), 1.0) * 1.0e-6
        vmin = float(vmin) - pad
        vmax = float(vmax) + pad

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    pc = PolyCollection(polys, array=colors, cmap=args.cmap, edgecolor=args.edgecolor, linewidth=args.linewidth)
    pc.set_clim(float(vmin), float(vmax))
    ax.add_collection(pc)

    bz = _first_bz_polygon_2d(b1, b2)
    if bz is not None:
        clip_patch = plt.Polygon(bz, closed=True, facecolor="none", edgecolor="none")
        ax.add_patch(clip_patch)
        pc.set_clip_path(clip_patch)
        closed = np.vstack([bz, bz[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=args.bz_color, lw=args.bz_lw, alpha=args.bz_alpha)
        xmin, ymin = np.min(bz, axis=0)
        xmax, ymax = np.max(bz, axis=0)
    else:
        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)

    dx = max(float(xmax - xmin), 1.0e-12)
    dy = max(float(ymax - ymin), 1.0e-12)
    ax.set_xlim(float(xmin) - args.margin * dx, float(xmax) + args.margin * dx)
    ax.set_ylim(float(ymin) - args.margin * dy, float(ymax) + args.margin * dy)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = args.title
    if not title:
        mag_part = "all magnons" if mag is None else f"magnon {args.magnon_mode}"
        title = f"phonon {args.phonon_mode}, {mag_part}, kz={kz:.6g}"
    ax.set_title(title)
    fig.colorbar(pc, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)

    print(
        "[magph-coupling-bz] "
        f"input={args.input}, output={args.output}, n_slice={int(mask.sum())}, "
        f"kz={kz:.8g}, phonon_mode={args.phonon_mode} (idx={ph}), "
        f"magnon_mode={'all' if mag is None else args.magnon_mode}, "
        f"value[min,max]=({float(np.nanmin(vals)):.6g},{float(np.nanmax(vals)):.6g})"
    )


def main():
    ap = argparse.ArgumentParser(description="Plot magnon-phonon coupling constant on a fixed-kz BZ slice")
    ap.add_argument("input", help="hybrid.npz containing magnon_vertex and qpts_frac")
    ap.add_argument("-o", "--output", required=True, help="Output image path")
    ap.add_argument("--structure", default=None, help="Structure file for reciprocal BZ shape; omit for fractional square coordinates")
    ap.add_argument("--phonon_mode", type=int, required=True, help="Phonon mode number")
    ap.add_argument("--magnon_mode", type=int, default=None, help="Magnon mode number; omit to use norm over all magnon modes")
    ap.add_argument("--mode_base", type=int, choices=[0, 1], default=1, help="Whether mode numbers are 0-based or 1-based")
    ap.add_argument("--ik", type=int, default=0, help="k index in magnon_vertex")
    ap.add_argument("--kz", type=float, default=0.0, help="Target fractional kz slice")
    ap.add_argument("--k_round", type=int, default=8, help="Rounding decimals used to identify kz slices")
    ap.add_argument("--quantity", choices=["norm", "norm2", "log10_abs"], default="norm")
    ap.add_argument("--tile", type=int, default=2, help="Periodic image range for Voronoi fill")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--vmin_percentile", type=float, default=1.0)
    ap.add_argument("--vmax_percentile", type=float, default=99.0)
    ap.add_argument("--edgecolor", default="none")
    ap.add_argument("--linewidth", type=float, default=0.0)
    ap.add_argument("--bz_color", default="0.35")
    ap.add_argument("--bz_lw", type=float, default=1.2)
    ap.add_argument("--bz_alpha", type=float, default=0.95)
    ap.add_argument("--margin", type=float, default=0.04)
    ap.add_argument("--fig_width", type=float, default=5.4)
    ap.add_argument("--fig_height", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    plot_coupling_bz(args)


if __name__ == "__main__":
    main()
