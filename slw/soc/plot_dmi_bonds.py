"""Plot TB2J exchange bonds and DMI vectors in real-space 3D coordinates."""

import argparse
import os
import re

import numpy as np

from slw.core.cli_paths import resolve_path, resolve_workdir


def parse_exchange_out(path):
    cell = []
    atoms = {}
    rows = []
    current = None
    section = None
    header_re = re.compile(
        r"^\s*(\w+)\s+(\w+)\s+\(\s*([+-]?\d+),\s*([+-]?\d+),\s*([+-]?\d+)\)\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+\(\s*([+-]?\d+(?:\.\d+)?),\s*"
        r"([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\)\s+([+-]?\d+(?:\.\d+)?)"
    )
    dmi_re = re.compile(
        r"DMI:\s*\(\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\)"
    )
    atom_re = re.compile(
        r"^\s*(\w+)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)"
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Cell (Angstrom)"):
                section = "cell"
                continue
            if stripped.startswith("Atoms:"):
                section = "atoms"
                continue
            if stripped.startswith("Exchange:"):
                section = "exchange"
                continue

            if section == "cell" and len(cell) < 3:
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        cell.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        pass
                continue

            if section == "atoms":
                match = atom_re.match(line)
                if match and match.group(1).lower() not in {"atom_number", "total"}:
                    atoms[match.group(1)] = np.asarray(
                        [float(match.group(2)), float(match.group(3)), float(match.group(4))],
                        dtype=float,
                    )
                continue

            if section == "exchange":
                match = header_re.match(line)
                if match:
                    current = {
                        "i": match.group(1),
                        "j": match.group(2),
                        "R": tuple(int(match.group(i)) for i in (3, 4, 5)),
                        "J_iso": float(match.group(6)),
                        "vector": np.asarray(
                            [float(match.group(7)), float(match.group(8)), float(match.group(9))],
                            dtype=float,
                        ),
                        "distance": float(match.group(10)),
                    }
                    continue
                match = dmi_re.search(line)
                if match and current is not None:
                    current["DMI"] = np.asarray(
                        [float(match.group(1)), float(match.group(2)), float(match.group(3))],
                        dtype=float,
                    )
                    rows.append(current)
                    current = None

    if len(cell) != 3:
        raise ValueError(f"{path}: failed to parse 3 cell vectors")
    if not atoms:
        raise ValueError(f"{path}: failed to parse atom coordinates")
    if not rows:
        raise ValueError(f"{path}: failed to parse exchange rows")
    return np.asarray(cell, dtype=float), atoms, rows


def filter_rows(rows, args):
    selected = []
    center = tuple(args.center_R)
    for row in rows:
        if args.elements and (row["i"] not in args.elements or row["j"] not in args.elements):
            continue
        if args.pair is not None:
            a, b = args.pair
            if row["i"] != a or row["j"] != b:
                continue
        if args.only_center_R and tuple(row["R"]) != center:
            continue
        if args.max_abs_R is not None and max(abs(x) for x in row["R"]) > int(args.max_abs_R):
            continue
        if args.max_distance is not None and row["distance"] > float(args.max_distance):
            continue
        if args.min_distance is not None and row["distance"] < float(args.min_distance):
            continue
        selected.append(row)
    return selected


def set_axes_equal(ax):
    limits = np.asarray([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_bonds(exchange_path, cell, atoms, rows, args):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons

    fig = plt.figure(figsize=(args.fig_width, args.fig_height))
    if args.controls:
        ax = fig.add_axes([0.06, 0.08, 0.68, 0.84], projection="3d")
        ax_pair = fig.add_axes([0.78, 0.70, 0.18, 0.14])
        ax_shell = fig.add_axes([0.78, 0.32, 0.18, 0.32])
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax_pair = None
        ax_shell = None

    colors = {
        "Mn1-Mn2": "tab:blue",
        "Mn2-Mn1": "tab:orange",
    }
    other_color = "0.55"

    # Draw atom positions in the reference cell.
    for label, pos in atoms.items():
        if not label.startswith("Mn") and not args.show_all_atoms:
            continue
        color = "crimson" if label.startswith("Mn") else "0.4"
        ax.scatter([pos[0]], [pos[1]], [pos[2]], s=70, color=color, depthshade=True)
        ax.text(pos[0], pos[1], pos[2], f" {label}", fontsize=9)

    d_norms = [float(np.linalg.norm(row["DMI"])) for row in rows]
    max_d = max(d_norms) if d_norms else 1.0
    if max_d <= 0.0:
        max_d = 1.0

    pair_groups = {}
    shell_groups = {}
    row_artists = []

    shell_values = sorted({round(float(row["distance"]), 3) for row in rows})
    shell_label_by_value = {value: f"{value:.3f} A" for value in shell_values}

    for row in rows:
        if row["i"] not in atoms:
            continue
        start = atoms[row["i"]]
        end = start + row["vector"]
        mid = 0.5 * (start + end)
        pair = f"{row['i']}-{row['j']}"
        color = colors.get(pair, other_color)
        linestyle = "-" if pair == "Mn1-Mn2" else "--" if pair == "Mn2-Mn1" else ":"
        line, = ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linestyle=linestyle,
            linewidth=args.bond_width,
            alpha=args.alpha,
        )
        endpoint = ax.scatter([end[0]], [end[1]], [end[2]], s=18, color=color, alpha=args.alpha)

        dvec = row["DMI"] * float(args.dmi_scale)
        arrow = ax.quiver(
            mid[0],
            mid[1],
            mid[2],
            dvec[0],
            dvec[1],
            dvec[2],
            color=color,
            linewidth=args.arrow_width,
            arrow_length_ratio=0.25,
            normalize=False,
        )
        artists = [line, endpoint, arrow]
        if args.labels:
            label = ax.text(
                mid[0],
                mid[1],
                mid[2],
                f" {row['i']}->{row['j']} R={row['R']}\n |D|={np.linalg.norm(row['DMI']):.3f}",
                fontsize=7,
                color=color,
            )
            artists.append(label)
        shell_key = round(float(row["distance"]), 3)
        row_artists.append({"pair": pair, "shell": shell_key, "artists": artists})
        pair_groups.setdefault(pair, []).extend(artists)
        shell_groups.setdefault(shell_key, []).extend(artists)

    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_zlabel("z (A)")
    ax.set_title(
        f"DMI bonds from {os.path.basename(exchange_path)}\n"
        f"N={len(rows)}, DMI scale={args.dmi_scale:g} A/meV"
    )
    set_axes_equal(ax)
    ax.view_init(elev=args.elev, azim=args.azim)

    pair_state = {pair: True for pair in sorted(pair_groups)}
    shell_state = {shell: True for shell in shell_values}

    def apply_visibility():
        for item in row_artists:
            visible = pair_state.get(item["pair"], True) and shell_state.get(item["shell"], True)
            for artist in item["artists"]:
                artist.set_visible(visible)
        fig.canvas.draw_idle()

    if args.controls:
        pair_labels = sorted(pair_groups)
        pair_checks = CheckButtons(ax_pair, pair_labels, [pair_state[label] for label in pair_labels])
        ax_pair.set_title("pair", fontsize=9)

        shell_labels = [shell_label_by_value[value] for value in shell_values]
        shell_checks = CheckButtons(ax_shell, shell_labels, [shell_state[value] for value in shell_values])
        ax_shell.set_title("distance", fontsize=9)

        def on_pair(label):
            pair_state[label] = not pair_state[label]
            apply_visibility()

        def on_shell(label):
            reverse = {text: value for value, text in shell_label_by_value.items()}
            shell = reverse[label]
            shell_state[shell] = not shell_state[shell]
            apply_visibility()

        pair_checks.on_clicked(on_pair)
        shell_checks.on_clicked(on_shell)
        fig._lamp_dmi_widgets = (pair_checks, shell_checks)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
    if not args.no_show:
        plt.show()


def build_argparser():
    parser = argparse.ArgumentParser(description="Plot TB2J exchange bonds with DMI arrows.")
    parser.add_argument("exchange_out", help="TB2J exchange.out")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--pair", nargs=2, default=None, metavar=("I", "J"), help="Directed atom-label pair")
    parser.add_argument("--elements", nargs="*", default=None, help="Allowed atom labels")
    parser.add_argument("--center-R", nargs=3, type=int, default=(0, 0, 0), metavar=("R1", "R2", "R3"))
    parser.add_argument("--only-center-R", action="store_true", help="Plot only bonds with R equal to --center-R")
    parser.add_argument("--max-abs-R", type=int, default=1, help="Keep bonds with max(abs(R)) <= this value")
    parser.add_argument("--min-distance", type=float, default=None)
    parser.add_argument("--max-distance", type=float, default=None)
    parser.add_argument("--dmi-scale", type=float, default=6.0, help="Arrow scale in Angstrom per meV")
    parser.add_argument("--bond-width", type=float, default=1.5)
    parser.add_argument("--arrow-width", type=float, default=1.6)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--show-all-atoms", action="store_true")
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--fig-width", type=float, default=9.5)
    parser.add_argument("--fig-height", type=float, default=8.0)
    parser.add_argument("--controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", default=None)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    workdir = resolve_workdir(args.workdir)
    exchange_path = resolve_path(workdir, args.exchange_out)
    cell, atoms, rows = parse_exchange_out(exchange_path)
    del cell  # Bond vectors in exchange.out are already Cartesian.
    rows = filter_rows(rows, args)
    if not rows:
        raise ValueError("No bonds selected; relax --pair/--max-distance/--max-abs-R filters")
    plot_bonds(exchange_path, None, atoms, rows, args)


if __name__ == "__main__":
    main()
