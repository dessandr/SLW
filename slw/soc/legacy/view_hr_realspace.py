"""Archived interactive hr.dat viewer."""

import argparse
import os

import numpy as np

from slw.core.cli_paths import resolve_path, resolve_workdir
from slw.core.wannier_io import read_wannier_hr


def load_hr_stack(path, divide_degen=False, sort_mode="norm"):
    dim, degens, hmap = read_wannier_hr(os.fspath(path))
    if dim <= 0 or not hmap:
        raise ValueError(f"Failed to read non-empty hr.dat: {path}")
    if sort_mode == "lex":
        r_keys = sorted(hmap.keys(), key=lambda r: (int(r[0]), int(r[1]), int(r[2])))
    elif sort_mode == "norm":
        r_keys = sorted(
            hmap.keys(),
            key=lambda r: (int(r[0]) ** 2 + int(r[1]) ** 2 + int(r[2]) ** 2, int(r[0]), int(r[1]), int(r[2])),
        )
    else:
        r_keys = list(hmap.keys())

    degen_by_r = {r: int(degens[i]) for i, r in enumerate(sorted(hmap.keys(), key=lambda x: (int(x[0]), int(x[1]), int(x[2]))))}
    mats = []
    for r in r_keys:
        mat = np.asarray(hmap[r], dtype=np.complex128)
        if divide_degen:
            mat = mat / float(degen_by_r[r])
        mats.append(mat)
    return dim, r_keys, np.stack(mats, axis=0)


def matrix_view(matrix, mode):
    if mode == "abs":
        return np.abs(matrix)
    if mode == "real":
        return np.real(matrix)
    if mode == "imag":
        return np.imag(matrix)
    if mode == "phase":
        return np.angle(matrix)
    if mode == "signed_abs":
        return np.sign(np.real(matrix)) * np.abs(matrix)
    raise ValueError(f"Unknown mode: {mode}")


def default_cmap(mode):
    if mode in {"real", "imag", "signed_abs", "phase"}:
        return "RdBu_r"
    return "magma"


def robust_clim(view, mode, percentile):
    finite = np.isfinite(view)
    if not np.any(finite):
        return 0.0, 1.0
    values = view[finite]
    if mode in {"real", "imag", "signed_abs", "phase"}:
        vmax = float(np.percentile(np.abs(values), percentile))
        if vmax <= 0.0:
            vmax = 1e-12
        return -vmax, vmax
    vmin = float(np.percentile(values, 100.0 - percentile))
    vmax = float(np.percentile(values, percentile))
    if vmin == vmax:
        vmax = vmin + 1e-12
    return vmin, vmax


def parse_r(value):
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError("--r requires three integers")
    return tuple(int(x) for x in value)


def nearest_r_index(r_keys, target):
    arr = np.asarray(r_keys, dtype=int)
    target = np.asarray(target, dtype=int)
    dist2 = np.sum((arr - target[None, :]) ** 2, axis=1)
    return int(np.argmin(dist2))


def top_elements(r_keys, mats, n_top):
    if n_top <= 0:
        return []
    vals = np.abs(mats)
    flat = vals.ravel()
    n_take = min(int(n_top), flat.size)
    if n_take == flat.size:
        idx = np.argsort(flat)[::-1]
    else:
        part = np.argpartition(flat, -n_take)[-n_take:]
        idx = part[np.argsort(flat[part])[::-1]]
    dim = mats.shape[1]
    rows = []
    for flat_idx in idx:
        ir, rem = divmod(int(flat_idx), dim * dim)
        i, j = divmod(rem, dim)
        rows.append({"R": r_keys[ir], "i": i + 1, "j": j + 1, "abs": float(vals[ir, i, j])})
    return rows


def show_viewer(hr_path, dim, r_keys, mats, args):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, CheckButtons, Slider

    r_to_idx = {r: i for i, r in enumerate(r_keys)}
    start_r = parse_r(args.r)
    start_idx = r_to_idx[start_r] if start_r in r_to_idx else nearest_r_index(r_keys, start_r or (0, 0, 0))
    top = top_elements(r_keys, mats, args.top)
    state = {
        "idx": start_idx,
        "mode": args.mode,
        "log": bool(args.log),
        "highlight": None,
        "message": "",
        "top_idx": 0,
    }

    fig = plt.figure(figsize=(11.5, 7.5))
    fig.suptitle(f"Wannier real-space HR viewer: {os.path.basename(hr_path)}", fontsize=12)
    ax_img = fig.add_axes([0.07, 0.17, 0.60, 0.73])
    ax_info = fig.add_axes([0.73, 0.60, 0.25, 0.30])
    ax_mode = fig.add_axes([0.73, 0.37, 0.12, 0.18])
    ax_log = fig.add_axes([0.88, 0.47, 0.10, 0.08])
    ax_top = fig.add_axes([0.73, 0.12, 0.25, 0.20])
    ax_slider = fig.add_axes([0.07, 0.07, 0.60, 0.05])
    ax_info.set_axis_off()
    ax_top.set_axis_off()

    view = matrix_view(mats[state["idx"]], state["mode"])
    if state["log"]:
        view = np.log10(np.abs(view) + float(args.eps))
    im = ax_img.imshow(view, origin="lower", interpolation="nearest", aspect="equal", cmap=default_cmap(state["mode"]))
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.035, pad=0.02)
    marker, = ax_img.plot([], [], "wo", markerfacecolor="none", markersize=10, markeredgewidth=1.5)
    ax_img.set_xlabel("orbital j")
    ax_img.set_ylabel("orbital i")
    ax_img.set_xticks(np.arange(0, dim, max(1, dim // 8)))
    ax_img.set_yticks(np.arange(0, dim, max(1, dim // 8)))

    info_text = ax_info.text(0.0, 1.0, "", va="top", family="monospace", fontsize=9)
    top_text = ax_top.text(0.0, 1.0, "", va="top", family="monospace", fontsize=8)
    modes = ["abs", "real", "imag", "phase", "signed_abs"]
    mode_radio = RadioButtons(ax_mode, labels=modes, active=modes.index(state["mode"]))
    log_check = CheckButtons(ax_log, labels=["log"], actives=[state["log"]])
    slider = Slider(ax_slider, "R index", 0, len(r_keys) - 1, valinit=start_idx, valstep=1, valfmt="%d")

    def update():
        idx = int(state["idx"])
        r = r_keys[idx]
        mat = mats[idx]
        view_now = matrix_view(mat, state["mode"])
        if state["log"]:
            view_now = np.log10(np.abs(view_now) + float(args.eps))
        im.set_data(view_now)
        im.set_cmap(default_cmap(state["mode"]))
        im.set_clim(*robust_clim(view_now, state["mode"], args.percentile))
        cbar.update_normal(im)

        abs_mat = np.abs(mat)
        i0, j0 = np.unravel_index(np.argmax(abs_mat), abs_mat.shape)
        mate = r_to_idx.get((-r[0], -r[1], -r[2]))
        herm = np.nan if mate is None else float(np.max(np.abs(mat - mats[mate].conj().T)))
        info_text.set_text(
            f"R = {r}\n"
            f"index = {idx + 1}/{len(r_keys)}\n"
            f"dim = {dim}\n"
            f"mode = {state['mode']}\n"
            f"log = {state['log']}\n\n"
            f"max|H| = {float(abs_mat[i0, j0]):.6e}\n"
            f"argmax = ({i0 + 1},{j0 + 1})\n"
            f"mean|H| = {float(np.mean(abs_mat)):.6e}\n"
            f"max|H(R)-H(-R)^†| = {herm:.6e}\n\n"
            f"{state['message']}"
        )

        lines = ["keys: arrows R1/R2, z/x R3, n/p next"]
        for k, row in enumerate(top[: min(len(top), 8)]):
            mark = ">>" if k == state["top_idx"] else "  "
            lines.append(f"{mark}{k + 1:2d} {row['abs']:.2e} R={row['R']} ({row['i']},{row['j']})")
        top_text.set_text("\n".join(lines))

        if state["highlight"] is None:
            marker.set_data([], [])
        else:
            marker.set_data([state["highlight"][1] - 1], [state["highlight"][0] - 1])
        ax_img.set_title(f"R={r}, {state['mode']}{', log10' if state['log'] else ''}")
        fig.canvas.draw_idle()

    def goto_idx(idx):
        state["idx"] = int(np.clip(idx, 0, len(r_keys) - 1))
        state["message"] = ""
        slider.set_val(state["idx"])

    def goto_r(target):
        target = tuple(int(x) for x in target)
        if target in r_to_idx:
            state["highlight"] = None
            goto_idx(r_to_idx[target])
        else:
            near = r_keys[nearest_r_index(r_keys, target)]
            state["message"] = f"R {target} not found; nearest existing is {near}"
            update()

    def on_slider(val):
        state["idx"] = int(val)
        state["highlight"] = None
        state["message"] = ""
        update()

    def on_mode(label):
        state["mode"] = str(label)
        update()

    def on_log(_label):
        state["log"] = not state["log"]
        update()

    def on_key(event):
        r = r_keys[state["idx"]]
        if event.key == "right":
            goto_r((r[0] + 1, r[1], r[2]))
        elif event.key == "left":
            goto_r((r[0] - 1, r[1], r[2]))
        elif event.key == "up":
            goto_r((r[0], r[1] + 1, r[2]))
        elif event.key == "down":
            goto_r((r[0], r[1] - 1, r[2]))
        elif event.key in ("x", "pageup"):
            goto_r((r[0], r[1], r[2] + 1))
        elif event.key in ("z", "pagedown"):
            goto_r((r[0], r[1], r[2] - 1))
        elif event.key in ("n", "d", "]"):
            state["highlight"] = None
            goto_idx(state["idx"] + 1)
        elif event.key in ("p", "a", "["):
            state["highlight"] = None
            goto_idx(state["idx"] - 1)
        elif event.key in ("t",):
            if top:
                state["top_idx"] = (state["top_idx"] + 1) % len(top)
                row = top[state["top_idx"]]
                state["highlight"] = (row["i"], row["j"])
                goto_idx(r_to_idx[row["R"]])

    slider.on_changed(on_slider)
    mode_radio.on_clicked(on_mode)
    log_check.on_clicked(on_log)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update()

    if args.save:
        fig.savefig(args.save, dpi=args.dpi)
    if not args.no_show:
        plt.show()


def build_argparser():
    parser = argparse.ArgumentParser(description="Interactive heatmap viewer for Wannier90 real-space hr.dat blocks.")
    parser.add_argument("hr", help="Wannier90 *_hr.dat file")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--r", nargs=3, type=int, default=None, metavar=("R1", "R2", "R3"), help="Initial R vector")
    parser.add_argument("--mode", choices=["abs", "real", "imag", "phase", "signed_abs"], default="abs")
    parser.add_argument("--sort", choices=["norm", "lex", "file"], default="norm", help="R block ordering for n/p navigation")
    parser.add_argument("--divide-degen", action="store_true", help="Show H(R)/ndegen(R), matching the Fourier-sum convention")
    parser.add_argument("--log", action="store_true", help="Start with log10(abs(value)+eps)")
    parser.add_argument("--eps", type=float, default=1e-16)
    parser.add_argument("--percentile", type=float, default=99.0, help="Robust color scale percentile")
    parser.add_argument("--top", type=int, default=50, help="Number of largest elements tracked by t key")
    parser.add_argument("--save", default=None, help="Optional image output path")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-show", action="store_true", help="Save or initialize without opening GUI")
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    workdir = resolve_workdir(args.workdir)
    hr_path = resolve_path(workdir, args.hr)
    dim, r_keys, mats = load_hr_stack(hr_path, divide_degen=args.divide_degen, sort_mode=args.sort)
    show_viewer(hr_path, dim, r_keys, mats, args)


if __name__ == "__main__":
    main()
