"""
Archived collinear-to-spinor builder with superseded basis layouts.

The collinear spin channels are combined as

    H = I * (H_up + H_down) / 2 + (n.sigma) * (H_up - H_down) / 2

and can be written either in separated-spin order

    [orb1 up, orb2 up, ..., orbN up, orb1 down, ...]

or spinor-block order

    [orb1 up, orb1 down, orb2 up, orb2 down, ...].
"""

import argparse
import glob
import math
import os
import re

import numpy as np

from slw.core.cli_paths import resolve_path, resolve_workdir
from slw.core.wannier_io import read_wannier_hr, write_wannier_hr


def read_hr_or_raise(path):
    dim, degens, dict_h = read_wannier_hr(os.fspath(path))
    if dim <= 0 or not dict_h:
        raise ValueError(f"Failed to read Wannier Hamiltonian: {path}")
    return dim, list(degens), dict_h


def spin_direction_from_angles(theta_deg, phi_deg):
    theta = math.radians(float(theta_deg))
    phi = math.radians(float(phi_deg))
    return np.asarray(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=float,
    )


def normalize_vector(vec):
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("Spin direction vector must be non-zero")
    return vec / norm


def zero_or_normalize_vector(vec, tol=1e-14):
    if vec is None:
        return None
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= tol or not np.isfinite(norm):
        return None
    return vec / norm


def spin_rotation_from_z(direction):
    """Return U with U sigma_z U^dagger = direction.sigma."""
    n = normalize_vector(direction)
    nx, ny, nz = n
    theta = math.acos(float(np.clip(nz, -1.0, 1.0)))
    phi = math.atan2(float(ny), float(nx))
    c = math.cos(0.5 * theta)
    s = math.sin(0.5 * theta)
    eip = complex(math.cos(phi), math.sin(phi))
    eim = complex(math.cos(phi), -math.sin(phi))
    return np.asarray([[c, -s * eim], [s * eip, c]], dtype=np.complex128)


def separated_to_spinor_block_permutation(n_orb):
    perm = []
    for iorb in range(int(n_orb)):
        perm.append(iorb)
        perm.append(iorb + n_orb)
    return np.asarray(perm, dtype=int)


def _strip_inline_comment(line):
    return line.split("!", 1)[0].split("#", 1)[0].strip()


def read_win_block(path, block_name):
    path = os.fspath(path)
    block_name = block_name.lower()
    lines_out = []
    in_block = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = _strip_inline_comment(line)
            low = stripped.lower()
            if not in_block:
                if low.startswith("begin") and block_name in low:
                    in_block = True
                continue
            if low.startswith("end") and block_name in low:
                break
            if stripped:
                lines_out.append(stripped)
    return lines_out


def species_name(label):
    return re.sub(r"\d+$", "", str(label))


def orbital_l_from_token(token):
    token = str(token).strip().lower()
    mapping = {"s": 0, "p": 1, "d": 2, "f": 3}
    if token in mapping:
        return mapping[token]
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(f"Unsupported projection orbital token: {token}") from exc


def read_atom_labels_from_win(path):
    for block_name in ("atoms_frac", "atoms_cart"):
        lines = read_win_block(path, block_name)
        if lines:
            return [line.split()[0] for line in lines if line.split()]
    raise ValueError(f"{path}: missing atoms_frac/atoms_cart block")


def read_projection_specs_from_win(path):
    specs = []
    for line in read_win_block(path, "projections"):
        if ":" not in line:
            continue
        lhs, rhs = line.split(":", 1)
        atom_sel = lhs.strip()
        tokens = [tok.strip().strip(",") for tok in re.split(r"[\s,;]+", rhs.strip()) if tok.strip()]
        for tok in tokens:
            if tok.lower() in ("random", "sp", "sp2", "sp3"):
                raise ValueError(f"Projection token '{tok}' is not supported for block inference")
            specs.append((atom_sel, orbital_l_from_token(tok)))
    if not specs:
        raise ValueError(f"{path}: no parseable projection specs found")
    return specs


def infer_spinless_blocks_from_win(win_path, dim):
    atom_labels = read_atom_labels_from_win(win_path)
    specs = read_projection_specs_from_win(win_path)
    used = np.zeros(len(atom_labels), dtype=bool)
    blocks = []
    start = 0
    for selector, l_val in specs:
        exact = [i for i, label in enumerate(atom_labels) if label == selector and not used[i]]
        if exact:
            indices = exact
        else:
            selector_species = species_name(selector)
            indices = [
                i
                for i, label in enumerate(atom_labels)
                if species_name(label) == selector_species and not used[i]
            ]
        if not indices:
            raise ValueError(f"{win_path}: projection selector '{selector}' did not match unused atoms")
        for idx in indices:
            used[idx] = True
            size = 2 * int(l_val) + 1
            blocks.append(
                {
                    "start": start,
                    "stop": start + size,
                    "l": int(l_val),
                    "atom_label": atom_labels[idx],
                }
            )
            start += size
    if start != int(dim):
        raise ValueError(f"{win_path}: inferred {start} orbitals from projections, expected dim={dim}")
    return blocks


def read_centres_xyz(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid centres file: {path}")
    total = int(lines[0].strip())
    body = lines[2:]
    if len(body) < total:
        raise ValueError(f"centres count mismatch in {path}: header={total}, rows={len(body)}")
    return body[:total]


def split_centres_body(body, n_wann):
    if len(body) < int(n_wann):
        raise ValueError(f"centres body has {len(body)} rows, expected at least {n_wann}")
    return body[: int(n_wann)], body[int(n_wann) :]


def default_centres_output(out_hr):
    base = os.path.basename(os.fspath(out_hr))
    if base.endswith("_hr.dat"):
        return os.path.join(os.path.dirname(os.path.abspath(out_hr)), base[:-7] + "_centres.xyz")
    stem, _ext = os.path.splitext(os.path.abspath(out_hr))
    return stem + "_centres.xyz"


def write_spinor_centres(
    out_path,
    centres_up,
    centres_down,
    n_orb,
    output_layout="spinor-block",
):
    up_wann, up_atoms = split_centres_body(read_centres_xyz(centres_up), n_orb)
    if centres_down:
        down_wann, down_atoms = split_centres_body(read_centres_xyz(centres_down), n_orb)
        if len(up_atoms) != len(down_atoms):
            raise ValueError(
                f"Atom centre row count mismatch: up={len(up_atoms)}, down={len(down_atoms)}"
            )
    else:
        down_wann = up_wann
        down_atoms = up_atoms

    spin_major = up_wann + down_wann
    if output_layout == "separated-spin":
        ordered_wann = spin_major
    elif output_layout == "spinor-block":
        perm = separated_to_spinor_block_permutation(n_orb)
        ordered_wann = [spin_major[int(i)] for i in perm]
    else:
        raise ValueError(f"Unknown output layout: {output_layout}")

    atom_lines = up_atoms
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"    {len(ordered_wann) + len(atom_lines)}\n")
        f.write(f" SLW spinor Wannier centres layout={output_layout}\n")
        for line in ordered_wann:
            f.write(line if line.endswith("\n") else line + "\n")
        for line in atom_lines:
            f.write(line if line.endswith("\n") else line + "\n")


def build_spinor_block(h_up, h_down, spin_direction, output_layout="spinor-block"):
    if h_up.shape != h_down.shape:
        raise ValueError(f"H_up/H_down shape mismatch: {h_up.shape} vs {h_down.shape}")
    n_orb = h_up.shape[0]
    h_avg = 0.5 * (h_up + h_down)
    h_diff = 0.5 * (h_up - h_down)
    nx, ny, nz = normalize_vector(spin_direction)

    h_sep = np.zeros((2 * n_orb, 2 * n_orb), dtype=np.complex128)
    h_sep[:n_orb, :n_orb] = h_avg + nz * h_diff
    h_sep[n_orb:, n_orb:] = h_avg - nz * h_diff
    h_sep[:n_orb, n_orb:] = (nx - 1j * ny) * h_diff
    h_sep[n_orb:, :n_orb] = (nx + 1j * ny) * h_diff

    if output_layout == "separated-spin":
        return h_sep
    if output_layout == "spinor-block":
        perm = separated_to_spinor_block_permutation(n_orb)
        return h_sep[np.ix_(perm, perm)]
    raise ValueError(f"Unknown output layout: {output_layout}")


def build_site_rotated_spinor_block(h_up, h_down, orbital_directions, output_layout="spinor-block"):
    if h_up.shape != h_down.shape:
        raise ValueError(f"H_up/H_down shape mismatch: {h_up.shape} vs {h_down.shape}")
    n_orb = h_up.shape[0]
    if len(orbital_directions) != n_orb:
        raise ValueError(f"Need {n_orb} orbital spin directions, got {len(orbital_directions)}")

    h_avg = 0.5 * (h_up + h_down)
    rotations = []
    active = np.zeros(n_orb, dtype=bool)
    for iorb, direction in enumerate(orbital_directions):
        direction = zero_or_normalize_vector(direction)
        if direction is None:
            rotations.append(None)
        else:
            rotations.append(spin_rotation_from_z(direction))
            active[iorb] = True

    h_sep = np.zeros((2 * n_orb, 2 * n_orb), dtype=np.complex128)
    eye2 = np.eye(2, dtype=np.complex128)
    for i in range(n_orb):
        for j in range(n_orb):
            if active[i] and active[j]:
                block = rotations[i] @ np.diag([h_up[i, j], h_down[i, j]]) @ rotations[j].conj().T
            else:
                block = h_avg[i, j] * eye2
            h_sep[i, j] = block[0, 0]
            h_sep[i, j + n_orb] = block[0, 1]
            h_sep[i + n_orb, j] = block[1, 0]
            h_sep[i + n_orb, j + n_orb] = block[1, 1]

    if output_layout == "separated-spin":
        return h_sep
    if output_layout == "spinor-block":
        perm = separated_to_spinor_block_permutation(n_orb)
        return h_sep[np.ix_(perm, perm)]
    raise ValueError(f"Unknown output layout: {output_layout}")


def pauli_from_direction(direction):
    nx, ny, nz = normalize_vector(direction)
    return np.asarray([[nz, nx - 1j * ny], [nx + 1j * ny, -nz]], dtype=np.complex128)


def build_weak_ferro_spinor_block(
    h_up,
    h_down,
    sublattice_signs,
    afm_axis,
    canting_axis,
    canting_angle_deg,
    output_layout="spinor-block",
):
    if h_up.shape != h_down.shape:
        raise ValueError(f"H_up/H_down shape mismatch: {h_up.shape} vs {h_down.shape}")
    n_orb = h_up.shape[0]
    signs = np.asarray(sublattice_signs, dtype=float)
    if signs.shape != (n_orb,):
        raise ValueError(f"Need {n_orb} sublattice signs, got {signs.shape}")

    h_avg = 0.5 * (h_up + h_down)
    h_diff = 0.5 * (h_up - h_down)
    theta = math.radians(float(canting_angle_deg))

    sigma_afm = pauli_from_direction(afm_axis)
    sigma_fm = pauli_from_direction(canting_axis)

    # H_diff already carries the AFM sublattice sign. Multiplying by the
    # endpoint sign turns the magnetic-block part into a common weak-FM field.
    sign_factor = 0.5 * (signs[:, None] + signs[None, :])
    h_fm = sign_factor * h_diff

    h_sep = np.kron(np.eye(2, dtype=np.complex128), h_avg)
    h_sep += math.cos(theta) * np.kron(sigma_afm, h_diff)
    h_sep += math.sin(theta) * np.kron(sigma_fm, h_fm)

    if output_layout == "separated-spin":
        return h_sep
    if output_layout == "spinor-block":
        perm = separated_to_spinor_block_permutation(n_orb)
        return h_sep[np.ix_(perm, perm)]
    raise ValueError(f"Unknown output layout: {output_layout}")


def build_spinor_hr(dict_up, dict_down, spin_direction, output_layout="spinor-block"):
    keys = sorted(
        set(dict_up.keys()) | set(dict_down.keys()),
        key=lambda r: (int(r[0]), int(r[1]), int(r[2])),
    )
    n_orb = next(iter(dict_up.values())).shape[0]
    out = {}
    zero = np.zeros((n_orb, n_orb), dtype=np.complex128)
    for r in keys:
        h_up = dict_up.get(r, zero)
        h_down = dict_down.get(r, zero)
        out[r] = build_spinor_block(
            h_up,
            h_down,
            spin_direction,
            output_layout=output_layout,
        )
    return out


def build_site_rotated_spinor_hr(dict_up, dict_down, orbital_directions, output_layout="spinor-block"):
    keys = sorted(
        set(dict_up.keys()) | set(dict_down.keys()),
        key=lambda r: (int(r[0]), int(r[1]), int(r[2])),
    )
    n_orb = next(iter(dict_up.values())).shape[0]
    out = {}
    zero = np.zeros((n_orb, n_orb), dtype=np.complex128)
    for r in keys:
        h_up = dict_up.get(r, zero)
        h_down = dict_down.get(r, zero)
        out[r] = build_site_rotated_spinor_block(
            h_up,
            h_down,
            orbital_directions,
            output_layout=output_layout,
        )
    return out


def build_weak_ferro_spinor_hr(
    dict_up,
    dict_down,
    sublattice_signs,
    afm_axis,
    canting_axis,
    canting_angle_deg,
    output_layout="spinor-block",
):
    keys = sorted(
        set(dict_up.keys()) | set(dict_down.keys()),
        key=lambda r: (int(r[0]), int(r[1]), int(r[2])),
    )
    n_orb = next(iter(dict_up.values())).shape[0]
    out = {}
    zero = np.zeros((n_orb, n_orb), dtype=np.complex128)
    for r in keys:
        h_up = dict_up.get(r, zero)
        h_down = dict_down.get(r, zero)
        out[r] = build_weak_ferro_spinor_block(
            h_up,
            h_down,
            sublattice_signs,
            afm_axis,
            canting_axis,
            canting_angle_deg,
            output_layout=output_layout,
        )
    return out


def weak_ferro_sublattice_signs(blocks, dim, magnetic_l=2):
    signs = np.zeros(int(dim), dtype=float)
    magnetic_blocks = [block for block in blocks if int(block["l"]) == int(magnetic_l)]
    if not magnetic_blocks:
        raise ValueError(f"No l={magnetic_l} magnetic blocks inferred from .win")
    summaries = []
    for imag, block in enumerate(magnetic_blocks):
        sign = 1.0 if imag % 2 == 0 else -1.0
        for iorb in range(block["start"], block["stop"]):
            signs[iorb] = sign
        summaries.append(
            {
                "atom_label": block.get("atom_label", f"block{imag + 1}"),
                "start": block["start"] + 1,
                "stop": block["stop"],
                "sign": int(sign),
            }
        )
    return signs, summaries


def strip_spin_suffix(seed, spin):
    suffixes = {
        "up": ("_up", "_u"),
        "down": ("_down", "_dn", "_dw", "_d"),
    }[spin]
    for suffix in suffixes:
        if seed.endswith(suffix):
            return seed[: -len(suffix)]
    return seed


def hr_seed_from_path(path):
    base = os.path.basename(path)
    if not base.endswith("_hr.dat"):
        return None
    return base[:-7]


def centres_seed_from_path(path):
    base = os.path.basename(path)
    if not base.endswith("_centres.xyz"):
        return None
    return base[:-12]


def find_seeded_file(directory, seed, kind, spin):
    if kind == "hr":
        suffix = "_hr.dat"
        spin_suffixes = {"up": ("_up", "_u"), "down": ("_down", "_dn", "_dw", "_d")}[spin]
    elif kind == "centres":
        suffix = "_centres.xyz"
        spin_suffixes = {"up": ("_up", "_u"), "down": ("_down", "_dn", "_dw", "_d")}[spin]
    elif kind == "win":
        suffix = ".win"
        spin_suffixes = {"up": ("_up", "_u"), "down": ("_down", "_dn", "_dw", "_d")}[spin]
    else:
        raise ValueError(f"Unknown file kind: {kind}")

    candidates = [os.path.join(directory, f"{seed}{suffix}")]
    candidates.extend(os.path.join(directory, f"{seed}{spin_suffix}{suffix}") for spin_suffix in spin_suffixes)
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def infer_seed(up_dir, down_dir):
    up_files = glob.glob(os.path.join(up_dir, "*_hr.dat"))
    down_files = glob.glob(os.path.join(down_dir, "*_hr.dat"))
    up_base = {strip_spin_suffix(hr_seed_from_path(path), "up") for path in up_files}
    down_base = {strip_spin_suffix(hr_seed_from_path(path), "down") for path in down_files}
    common = sorted(seed for seed in (up_base & down_base) if seed)
    if len(common) == 1:
        return common[0]
    if not common:
        raise ValueError(f"No compatible *_hr.dat seed in {up_dir} and {down_dir}")
    raise ValueError(f"Multiple compatible seeds {common}; pass --seed explicitly")


def build_argparser():
    parser = argparse.ArgumentParser(description="Build spinor noSOC hr.dat from collinear up/down hr.dat")
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--up-dir", type=str, default="up")
    parser.add_argument("--down-dir", type=str, default="down")
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--up-hr", type=str, default=None)
    parser.add_argument("--down-hr", type=str, default=None)
    parser.add_argument("--centres", type=str, default=None, help="Single centres.xyz duplicated for both spins")
    parser.add_argument("--centres-up", type=str, default=None, help="Spin-up centres.xyz")
    parser.add_argument("--centres-down", type=str, default=None, help="Spin-down centres.xyz")
    parser.add_argument("--centres-out", type=str, default=None, help="Output spinor centres.xyz")
    parser.add_argument("--win", type=str, default=None, help="Wannier .win for automatic orbital block inference")
    parser.add_argument(
        "--no-centres",
        action="store_true",
        help="Do not write centres.xyz even if input centres files are found",
    )
    parser.add_argument(
        "--output-layout",
        choices=["spinor-block", "separated-spin"],
        default="spinor-block",
    )
    parser.add_argument("--theta", type=float, default=0.0, help="Spin direction polar angle in degrees")
    parser.add_argument("--phi", type=float, default=0.0, help="Spin direction azimuthal angle in degrees")
    parser.add_argument(
        "--spin-direction",
        nargs=3,
        type=float,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help="Override theta/phi with explicit spin direction vector",
    )
    parser.add_argument(
        "--weak-ferro-canting",
        action="store_true",
        help="Test mode: infer magnetic blocks from .win and add a common weak-FM canting component",
    )
    parser.add_argument(
        "--afm-axis",
        nargs=3,
        type=float,
        default=(0.0, 1.0, 0.0),
        metavar=("NX", "NY", "NZ"),
        help="AFM axis for --weak-ferro-canting",
    )
    parser.add_argument(
        "--canting-axis",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("NX", "NY", "NZ"),
        help="Common weak-ferro canting axis for --weak-ferro-canting",
    )
    parser.add_argument(
        "--canting-angle-deg",
        type=float,
        default=0.0,
        help="Canting angle away from +/-AFM axis toward --canting-axis",
    )
    parser.add_argument(
        "--magnetic-l",
        type=int,
        default=2,
        help="Angular momentum channel treated as magnetic in automatic block inference",
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--out-prefix", type=str, default=None)
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    workdir = resolve_workdir(args.workdir)
    root = resolve_path(workdir, args.root)
    up_dir = resolve_path(root, args.up_dir)
    down_dir = resolve_path(root, args.down_dir)
    seed = args.seed or infer_seed(up_dir, down_dir)

    up_path = resolve_path(workdir, args.up_hr) if args.up_hr else find_seeded_file(up_dir, seed, "hr", "up")
    down_path = resolve_path(workdir, args.down_hr) if args.down_hr else find_seeded_file(down_dir, seed, "hr", "down")
    if up_path is None:
        raise ValueError(f"Could not find up hr.dat for seed={seed} in {up_dir}")
    if down_path is None:
        raise ValueError(f"Could not find down hr.dat for seed={seed} in {down_dir}")
    if args.out:
        out_path = resolve_path(workdir, args.out)
    elif args.out_prefix:
        out_path = resolve_path(workdir, f"{args.out_prefix}_hr.dat")
    else:
        out_path = os.path.join(root, f"{seed}_spinor_{args.output_layout}_hr.dat")
    centres_up = None
    centres_down = None
    centres_out = None
    if not args.no_centres:
        if args.centres:
            centres_up = resolve_path(workdir, args.centres)
            centres_down = None
        else:
            centres_up = (
                resolve_path(workdir, args.centres_up)
                if args.centres_up
                else find_seeded_file(up_dir, seed, "centres", "up")
            )
            centres_down = (
                resolve_path(workdir, args.centres_down)
                if args.centres_down
                else find_seeded_file(down_dir, seed, "centres", "down")
            )
            if centres_down is not None and not os.path.exists(centres_down):
                centres_down = None
        if centres_up is None or not os.path.exists(centres_up):
            centres_up = None
            centres_down = None
        centres_out = resolve_path(workdir, args.centres_out) if args.centres_out else default_centres_output(out_path)

    dim_up, degens_up, dict_up = read_hr_or_raise(up_path)
    dim_down, degens_down, dict_down = read_hr_or_raise(down_path)
    if dim_up != dim_down:
        raise ValueError(f"Up/down dimensions differ: {dim_up} vs {dim_down}")
    if len(degens_up) != len(degens_down):
        raise ValueError("Up/down degeneracy lists have different lengths")

    win_path = None
    block_summaries = []
    if args.weak_ferro_canting:
        win_path = resolve_path(workdir, args.win) if args.win else (
            find_seeded_file(up_dir, seed, "win", "up") or find_seeded_file(down_dir, seed, "win", "down")
        )
        if win_path is None or not os.path.exists(win_path):
            raise ValueError("--weak-ferro-canting requires .win; pass --win or keep it near up/down hr.dat")
        blocks = infer_spinless_blocks_from_win(win_path, dim_up)
        sublattice_signs, block_summaries = weak_ferro_sublattice_signs(
            blocks,
            dim_up,
            magnetic_l=args.magnetic_l,
        )
        dict_spinor = build_weak_ferro_spinor_hr(
            dict_up,
            dict_down,
            sublattice_signs,
            args.afm_axis,
            args.canting_axis,
            args.canting_angle_deg,
            output_layout=args.output_layout,
        )
        spin_direction = None
    else:
        spin_direction = (
            normalize_vector(args.spin_direction)
            if args.spin_direction is not None
            else spin_direction_from_angles(args.theta, args.phi)
        )
        dict_spinor = build_spinor_hr(
            dict_up,
            dict_down,
            spin_direction,
            output_layout=args.output_layout,
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_wannier_hr(
        out_path,
        2 * dim_up,
        degens_up,
        dict_spinor,
        header=(
            "SLW collinear up/down spinor Hamiltonian "
            f"layout={args.output_layout}"
        ),
    )
    if centres_up:
        write_spinor_centres(
            centres_out,
            centres_up,
            centres_down,
            dim_up,
            output_layout=args.output_layout,
        )

    print(f"Using seed: {seed}")
    print(f"Loaded up: dim={dim_up}, R={len(dict_up)} from {up_path}")
    print(f"Loaded down: dim={dim_down}, R={len(dict_down)} from {down_path}")
    if args.weak_ferro_canting:
        print(f"Weak-ferro canting mode: win={win_path}")
        print(
            "AFM axis: "
            f"[{float(args.afm_axis[0]):.8g}, {float(args.afm_axis[1]):.8g}, {float(args.afm_axis[2]):.8g}]"
        )
        print(
            "Canting axis/angle: "
            f"[{float(args.canting_axis[0]):.8g}, {float(args.canting_axis[1]):.8g}, "
            f"{float(args.canting_axis[2]):.8g}], {float(args.canting_angle_deg):.8g} deg"
        )
        print("Magnetic block AFM signs:")
        afm_axis = normalize_vector(args.afm_axis)
        canting_axis = normalize_vector(args.canting_axis)
        theta = math.radians(float(args.canting_angle_deg))
        for item in block_summaries:
            direction = normalize_vector(
                item["sign"] * math.cos(theta) * afm_axis + math.sin(theta) * canting_axis
            )
            print(
                f"  {item['atom_label']} [{item['start']}:{item['stop']}] "
                f"sign={item['sign']:+d} "
                f"n=[{direction[0]:.8g}, {direction[1]:.8g}, {direction[2]:.8g}]"
            )
    else:
        print(f"Spin direction: [{spin_direction[0]:.8g}, {spin_direction[1]:.8g}, {spin_direction[2]:.8g}]")
    print(f"Output layout: {args.output_layout}")
    print(f"Saved spinor hr: {out_path}")
    if centres_up:
        print(f"Saved spinor centres: {centres_out}")
    elif not args.no_centres:
        print("No centres.xyz written: input centres file was not found")


if __name__ == "__main__":
    main()
