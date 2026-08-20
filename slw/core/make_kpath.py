"""
Generate high-symmetry k paths for QE and Wannier90.

The tool reads POSCAR-like files and QE scf inputs through the existing SLW
structure parser. If seekpath is installed it can be used as the backend;
otherwise a lightweight input-cell fallback covers common Bravais shapes.
"""

import argparse
import math
import os
import re

import numpy as np

from slw.core.cli_paths import resolve_path, resolve_workdir
from slw.core.structure_io import read_structure


GAMMA_LABELS = {"G", "Gamma", "GAMMA", "\\Gamma", "Γ"}


def strip_species(label):
    return re.sub(r"\d+$", "", str(label))


def structure_to_spglib_cell(lattice, labels, cart_pos):
    species = []
    mapping = {}
    for label in labels:
        name = strip_species(label)
        if name not in mapping:
            mapping[name] = len(mapping) + 1
        species.append(mapping[name])
    frac = (np.asarray(cart_pos, dtype=float) @ np.linalg.inv(np.asarray(lattice, dtype=float))) % 1.0
    return np.asarray(lattice, dtype=float), frac, np.asarray(species, dtype=int)


def lattice_lengths_angles(lattice):
    a_vec, b_vec, c_vec = np.asarray(lattice, dtype=float)
    lengths = np.asarray([np.linalg.norm(a_vec), np.linalg.norm(b_vec), np.linalg.norm(c_vec)])

    def angle(u, v):
        cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return math.degrees(math.acos(float(np.clip(cosang, -1.0, 1.0))))

    angles = np.asarray([angle(b_vec, c_vec), angle(a_vec, c_vec), angle(a_vec, b_vec)])
    return lengths, angles


def close(x, y, tol):
    return abs(float(x) - float(y)) <= tol


def classify_lattice(lattice, tol_len=1e-3, tol_ang=1e-2):
    lengths, angles = lattice_lengths_angles(lattice)
    a, b, c = lengths
    alpha, beta, gamma = angles
    right = close(alpha, 90.0, tol_ang) and close(beta, 90.0, tol_ang) and close(gamma, 90.0, tol_ang)
    same_ab = abs(a - b) <= tol_len * max(a, b)
    same_bc = abs(b - c) <= tol_len * max(b, c)
    same_ac = abs(a - c) <= tol_len * max(a, c)

    if right and same_ab and same_bc:
        return "cubic"
    if right and same_ab:
        return "tetragonal"
    if right:
        return "orthorhombic"
    if same_ab and close(alpha, 90.0, tol_ang) and close(beta, 90.0, tol_ang) and (
        close(gamma, 120.0, tol_ang) or close(gamma, 60.0, tol_ang)
    ):
        return "hexagonal"
    if same_ab and same_bc and close(alpha, beta, tol_ang) and close(beta, gamma, tol_ang):
        return "rhombohedral"
    return "generic"


def fallback_points_path(lattice):
    kind = classify_lattice(lattice)
    if kind == "hexagonal":
        points = {
            "G": (0.0, 0.0, 0.0),
            "M": (0.5, 0.0, 0.0),
            "K": (1.0 / 3.0, 1.0 / 3.0, 0.0),
            "A": (0.0, 0.0, 0.5),
            "L": (0.5, 0.0, 0.5),
            "H": (1.0 / 3.0, 1.0 / 3.0, 0.5),
        }
        path = ["G", "M", "K", "G", "A", "L", "H", "A"]
    elif kind in {"cubic", "tetragonal"}:
        points = {
            "G": (0.0, 0.0, 0.0),
            "X": (0.5, 0.0, 0.0),
            "M": (0.5, 0.5, 0.0),
            "Z": (0.0, 0.0, 0.5),
            "R": (0.5, 0.5, 0.5),
            "A": (0.5, 0.5, 0.5),
        }
        path = ["G", "X", "M", "G", "Z", "R", "X"]
    elif kind == "orthorhombic":
        points = {
            "G": (0.0, 0.0, 0.0),
            "X": (0.5, 0.0, 0.0),
            "Y": (0.0, 0.5, 0.0),
            "Z": (0.0, 0.0, 0.5),
            "S": (0.5, 0.5, 0.0),
            "T": (0.0, 0.5, 0.5),
            "U": (0.5, 0.0, 0.5),
            "R": (0.5, 0.5, 0.5),
        }
        path = ["G", "X", "S", "Y", "G", "Z", "U", "R", "T", "Z"]
    else:
        points = {
            "G": (0.0, 0.0, 0.0),
            "X": (0.5, 0.0, 0.0),
            "Y": (0.0, 0.5, 0.0),
            "Z": (0.0, 0.0, 0.5),
        }
        path = ["G", "X", "G", "Y", "G", "Z"]
    return kind, points, path


def seekpath_points_path(lattice, labels, cart_pos, symprec):
    try:
        import seekpath
    except ImportError as exc:
        raise RuntimeError("seekpath backend requested but seekpath is not installed") from exc
    cell = structure_to_spglib_cell(lattice, labels, cart_pos)
    data = seekpath.get_path(cell, symprec=float(symprec))
    raw_points = data["point_coords"]
    points = {("G" if label in GAMMA_LABELS else label): tuple(coords) for label, coords in raw_points.items()}
    path = []
    for start, stop in data["path"]:
        start = "G" if start in GAMMA_LABELS else start
        stop = "G" if stop in GAMMA_LABELS else stop
        if not path:
            path.append(start)
        elif path[-1] != start:
            path.append(start)
        path.append(stop)
    return "seekpath", points, path


def get_points_path(lattice, labels, cart_pos, backend="auto", symprec=1e-5):
    if backend in {"auto", "seekpath"}:
        try:
            return seekpath_points_path(lattice, labels, cart_pos, symprec)
        except Exception:
            if backend == "seekpath":
                raise
    return fallback_points_path(lattice)


def format_label(label):
    return "G" if label in GAMMA_LABELS else str(label)


def qe_kpoints_text(points, path, points_per_segment):
    lines = ["K_POINTS crystal_b", f"{len(path)}"]
    for idx, label in enumerate(path):
        k = points[label]
        npts = int(points_per_segment) if idx < len(path) - 1 else 1
        lines.append(
            f"  {k[0]: .10f} {k[1]: .10f} {k[2]: .10f}  {npts:d}  ! {format_label(label)}"
        )
    return "\n".join(lines) + "\n"


def wannier_kpoint_path_text(points, path):
    lines = ["begin kpoint_path"]
    for start, stop in zip(path[:-1], path[1:]):
        ks = points[start]
        ke = points[stop]
        lines.append(
            f" {format_label(start):<4s} {ks[0]: .10f} {ks[1]: .10f} {ks[2]: .10f} "
            f" {format_label(stop):<4s} {ke[0]: .10f} {ke[1]: .10f} {ke[2]: .10f}"
        )
    lines.append("end kpoint_path")
    return "\n".join(lines) + "\n"


def write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def build_argparser():
    parser = argparse.ArgumentParser(description="Generate QE and Wannier90 high-symmetry k paths")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--input", required=True, help="POSCAR-like file or QE scf input")
    parser.add_argument("--backend", choices=["auto", "seekpath", "fallback"], default="auto")
    parser.add_argument("--symprec", type=float, default=1e-5)
    parser.add_argument("--points-per-segment", type=int, default=50)
    parser.add_argument("--qe-out", default=None, help="Output QE K_POINTS crystal_b file")
    parser.add_argument("--win-out", default=None, help="Output Wannier90 kpoint_path snippet")
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    workdir = resolve_workdir(args.workdir)
    input_path = resolve_path(workdir, args.input)
    lattice, labels, cart_pos = read_structure(input_path)
    kind, points, path = get_points_path(
        lattice,
        labels,
        cart_pos,
        backend=args.backend,
        symprec=args.symprec,
    )
    qe_text = qe_kpoints_text(points, path, args.points_per_segment)
    win_text = wannier_kpoint_path_text(points, path)

    if args.qe_out:
        qe_out = resolve_path(workdir, args.qe_out)
        write_text(qe_out, qe_text)
        print(f"Saved QE K_POINTS crystal_b: {qe_out}")
    if args.win_out:
        win_out = resolve_path(workdir, args.win_out)
        write_text(win_out, win_text)
        print(f"Saved Wannier90 kpoint_path: {win_out}")
    if not args.qe_out and not args.win_out:
        print("# QE")
        print(qe_text, end="")
        print("# Wannier90")
        print(win_text, end="")

    print(f"Backend/path type: {kind}")
    print(f"Path labels: {' '.join(path)}")


if __name__ == "__main__":
    main()
