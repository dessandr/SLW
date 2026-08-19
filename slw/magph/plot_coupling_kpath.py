"""Plot magnon-phonon coupling constant along a k-path for a fixed q from hybrid.npz."""

from __future__ import annotations

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slw.magph.plot_coupling_bz import _mode_index

def _coupling_values_kpath(g, *, iq, phonon_mode, magnon_mode, quantity):
    arr = np.asarray(g, dtype=np.complex128)
    nk, nq, nph, nmag = arr.shape
    if iq < 0 or iq >= nq:
        raise ValueError(f"iq out of range: {iq}, nq={nq}")
    if phonon_mode < 0 or phonon_mode >= nph:
        raise ValueError(f"phonon_mode out of range: {phonon_mode}, nph={nph}")

    gp = arr[:, int(iq), int(phonon_mode), :]
    if magnon_mode is None:
        amp2 = np.sum(np.abs(gp) ** 2, axis=1)
        label_base = rf"$\sqrt{{\sum_m |g_{{\lambda m}}|^2}}$"
    else:
        amp2 = np.abs(gp[:, int(magnon_mode)]) ** 2
        label_base = rf"$|g_{{\lambda,{magnon_mode}}}|$"

    q = str(quantity).strip().lower()
    if q in {"norm", "abs", "amplitude"}:
        return np.sqrt(amp2), f"{label_base} (meV)"
    if q in {"norm2", "abs2", "power"}:
        return amp2, rf"{label_base}$^2$ (meV$^2$)"
    if q in {"log", "log10", "log10_abs"}:
        return np.log10(np.sqrt(amp2) + 1.0e-30), rf"$\log_{{10}}$ {label_base}"
    raise ValueError(f"Unknown quantity={quantity!r}")

def plot_coupling_kpath(args):
    data = np.load(args.input, allow_pickle=True)
    g = np.asarray(data["magnon_vertex"], dtype=np.complex128)
    nk, nq, nph, nmag = g.shape

    if "kdist" not in data:
        raise KeyError(f"{args.input} does not contain 'kdist'. The calculation might not have used a k-path.")
    kdist = np.asarray(data["kdist"], dtype=np.float64)

    ph = _mode_index(args.phonon_mode, args.mode_base, nph, "phonon_mode")
    mag = None if args.magnon_mode is None else _mode_index(args.magnon_mode, args.mode_base, nmag, "magnon_mode")
    vals, ylabel = _coupling_values_kpath(g, iq=int(args.iq), phonon_mode=ph, magnon_mode=mag, quantity=args.quantity)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    # Plot the coupling strength
    ax.plot(kdist, vals, color=args.color, lw=args.linewidth)

    # Plot k-path ticks if available
    if "tick_indices" in data and "kpath_symbols" in data:
        ticks = np.asarray(data["tick_indices"], dtype=np.int64)
        tick_labels = np.asarray(data["kpath_symbols"], dtype=object)
        for tick in ticks:
            ax.axvline(kdist[tick], color="0.78", lw=0.8)
        ax.set_xticks([kdist[t] for t in ticks], tick_labels)
    else:
        ax.set_xlabel("k path distance")

    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_ylabel(ylabel)

    title = args.title or f"phonon {args.phonon_mode}, {'all magnons' if mag is None else f'magnon {args.magnon_mode}'} (fixed Q)"
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved 1D k-path plot to {args.output}")

def main():
    ap = argparse.ArgumentParser(description="Plot magnon-phonon coupling constant along k-path for a fixed Q")
    ap.add_argument("input", help="hybrid.npz")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--phonon_mode", type=int, required=True)
    ap.add_argument("--magnon_mode", type=int, default=None)
    ap.add_argument("--mode_base", type=int, choices=[0, 1], default=1)
    ap.add_argument("--iq", type=int, default=0, help="q index in magnon_vertex")
    ap.add_argument("--quantity", choices=["norm", "norm2", "log10_abs"], default="norm")
    ap.add_argument("--color", default="black")
    ap.add_argument("--linewidth", type=float, default=1.5)
    ap.add_argument("--fig_width", type=float, default=6.4)
    ap.add_argument("--fig_height", type=float, default=4.2)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--title", default=None)
    plot_coupling_kpath(ap.parse_args())

if __name__ == "__main__":
    main()
