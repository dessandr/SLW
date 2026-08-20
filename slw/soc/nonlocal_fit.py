"""
Fit effective SOC corrections to a spinful collinear Hamiltonian.

The default model keeps atomic L.S as onsite terms and represents nonlocal
spin-flip processes with bond-resolved Slater-Koster d-p operators. The output
hr.dat contains the fitted correction added to the input collinear hr.dat.
"""

import argparse
import concurrent.futures
import csv
import glob
import os
import re
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from slw.core.constants import BOHR_TO_ANG
from slw.core.cli_paths import resolve_path, resolve_workdir
from slw.core.wannier_io import read_wannier_hr, write_wannier_hr
from slw.soc.band_splitting import (
    compute_alignment_shift,
    infer_seed,
    parse_band_selection,
    read_kpoints,
)
from slw.exchange.legacy.spinor_model import atomic_soc_block_diagonal


@dataclass(frozen=True)
class HRData:
    dim: int
    degens: list
    r_keys: list
    r_vecs: np.ndarray
    h_raw: np.ndarray
    h_norm: np.ndarray


def load_hr(path):
    dim, degens, dict_h = read_wannier_hr(os.fspath(path))
    if dim <= 0 or not dict_h:
        raise ValueError(f"Failed to read Wannier Hamiltonian: {path}")
    r_keys = sorted(dict_h.keys(), key=lambda r: (int(r[0]), int(r[1]), int(r[2])))
    if len(degens) != len(r_keys):
        raise ValueError(
            f"Degeneracy count mismatch in {path}: len(degens)={len(degens)}, "
            f"len(R)={len(r_keys)}"
        )
    h_raw = np.stack([dict_h[r] for r in r_keys], axis=0)
    h_norm = np.stack([dict_h[r] / float(degens[i]) for i, r in enumerate(r_keys)], axis=0)
    return HRData(
        dim=dim,
        degens=list(degens),
        r_keys=r_keys,
        r_vecs=np.asarray(r_keys, dtype=float),
        h_raw=h_raw,
        h_norm=h_norm,
    )


def hamiltonian_k(hr, kpts):
    phase = np.exp(2j * np.pi * (np.asarray(kpts, dtype=float) @ hr.r_vecs.T))
    hk = np.einsum("kr,rij->kij", phase, hr.h_norm, optimize=True)
    return 0.5 * (hk + np.swapaxes(hk.conj(), -1, -2))


def make_shifted_kmesh(mesh, shift):
    """Build a shifted uniform fractional k-mesh."""
    mesh = np.asarray(mesh, dtype=int)
    shift = np.asarray(shift, dtype=float)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError(f"--kmesh must contain three positive integers, got {mesh}")
    if shift.shape != (3,):
        raise ValueError(f"--mesh-shift must contain three values, got {shift}")

    grid = np.indices(tuple(mesh), dtype=float)
    kx = (grid[0].ravel() + shift[0]) / float(mesh[0])
    ky = (grid[1].ravel() + shift[1]) / float(mesh[1])
    kz = (grid[2].ravel() + shift[2]) / float(mesh[2])
    return np.column_stack([kx, ky, kz]) % 1.0


def parse_blocks(block_args):
    """Parse orbital blocks of the form start:stop:l, all 1-based inclusive."""
    blocks = []
    tokens = []
    for item in block_args or []:
        parts = item.split(":")
        if len(parts) == 3:
            tokens.append(parts)
        elif len(parts) > 3 and len(parts) % 3 == 0:
            tokens.extend(parts[i:i + 3] for i in range(0, len(parts), 3))
        else:
            raise ValueError(
                "Blocks must look like 'start:stop:l'. "
                f"Got {item}. Use e.g. --blocks 1:10:2 11:20:2 "
                "or --blocks 1:10:2:11:20:2."
            )
    for parts in tokens:
        start, stop, l_val = (int(x) for x in parts)
        if start <= 0 or stop < start or l_val < 0:
            raise ValueError(f"Invalid orbital block: {':'.join(parts)}")
        blocks.append({"start": start - 1, "stop": stop, "l": l_val})
    return blocks


def parse_moment_components(text):
    raw = str(text or "z").replace(",", "").lower()
    out = []
    for char in raw:
        if char not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported moment component {char!r}; use any subset of xyz")
        if char not in out:
            out.append(char)
    return out or ["z"]


def moment_blocks_from_selector(blocks, selector):
    if not selector:
        selected = [block for block in blocks if int(block["l"]) == 2]
        if not selected:
            raise ValueError("--moment-constraint needs --moment-subspace because no d blocks were inferred")
        return selected

    selected = []
    seen = set()
    for item in re.split(r"[,;]+", str(selector)):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.replace("=", ":").split(":") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Moment subspace selector must look like element:orbital, got {item!r}")
        atom_sel, orb_sel = parts
        l_val = orbital_l_from_token(orb_sel)
        atom_species = species_name(atom_sel)
        for iblock, block in enumerate(blocks):
            label = str(block.get("atom_label", ""))
            if int(block["l"]) != int(l_val):
                continue
            if label != atom_sel and species_name(label) != atom_species:
                continue
            if iblock not in seen:
                selected.append(block)
                seen.add(iblock)
    if not selected:
        known = sorted({f"{block.get('atom_label', '')}:l{block['l']}" for block in blocks})
        raise ValueError(f"--moment-subspace={selector!r} matched no blocks; known={known}")
    return selected


def _strip_inline_comment(line):
    return line.split("!", 1)[0].split("#", 1)[0].strip()


def read_win_block(path, block_name):
    path = os.fspath(path)
    block_name = block_name.lower()
    lines_out = []
    in_block = False
    with open(path, "r") as f:
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


def make_unique_atom_labels(labels):
    labels = [str(label) for label in labels]
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    running = {}
    unique = []
    for label in labels:
        if counts[label] == 1:
            unique.append(label)
            continue
        running[label] = running.get(label, 0) + 1
        unique.append(f"{species_name(label)}{running[label]}")
    return unique


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
            return make_unique_atom_labels([line.split()[0] for line in lines if line.split()])
    raise ValueError(f"{path}: missing atoms_frac/atoms_cart block")


def read_atoms_from_win(path):
    cell = read_unit_cell_cart_from_win(path)
    for block_name in ("atoms_frac", "atoms_cart"):
        lines = read_win_block(path, block_name)
        if not lines:
            continue
        labels = []
        coords = []
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            labels.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        coords = np.asarray(coords, dtype=float)
        if block_name == "atoms_frac":
            frac = coords
        else:
            frac = coords @ np.linalg.inv(cell)
        unique_labels = make_unique_atom_labels(labels)
        species = {name: idx + 1 for idx, name in enumerate(sorted({species_name(x) for x in unique_labels}))}
        numbers = np.asarray([species[species_name(label)] for label in unique_labels], dtype=int)
        return unique_labels, np.mod(frac, 1.0), numbers
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
                raise ValueError(f"Projection token '{tok}' is not supported for automatic L.S blocks")
            specs.append((atom_sel, orbital_l_from_token(tok)))
    if not specs:
        raise ValueError(f"{path}: no parseable projection specs found")
    return specs


def expand_projection_l_sequence(win_path):
    atom_labels = read_atom_labels_from_win(win_path)
    specs = read_projection_specs_from_win(win_path)
    used = np.zeros(len(atom_labels), dtype=bool)
    sequence = []

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
            raise ValueError(
                f"{win_path}: projection selector '{selector}' did not match unused atoms {atom_labels}"
            )
        for idx in indices:
            used[idx] = True
            sequence.append({"atom_label": atom_labels[idx], "l": l_val})

    return sequence


def read_centres_count(path):
    with open(path, "r") as f:
        return int(f.readline().strip())


def read_centres_coords(path, dim, natoms, block_layout):
    path = os.fspath(path)
    with open(path, "r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid centres file: {path}")
    total = int(lines[0].strip())
    body = lines[2:]
    if len(body) < total:
        raise ValueError(f"{path}: centres count mismatch header={total}, rows={len(body)}")
    n_wann = total - int(natoms)
    wann_lines = body[:n_wann]
    coords = np.asarray([[float(x) for x in line.split()[1:4]] for line in wann_lines], dtype=float)
    if n_wann == dim:
        return coords
    if 2 * n_wann != dim:
        raise ValueError(f"{path}: n_wann={n_wann} is incompatible with dim={dim}")
    if block_layout == "spinor-block":
        return np.repeat(coords, 2, axis=0)
    if block_layout == "separated-spin":
        return np.vstack([coords, coords])
    raise ValueError(f"Unknown block layout: {block_layout}")


def find_default_centres_path(root, soc_dir, seed):
    candidates = [
        os.path.join(soc_dir, f"{seed}_centres.xyz"),
        os.path.join(root, f"{seed}_centres.xyz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def hr_seed_from_path(path):
    base = os.path.basename(os.fspath(path))
    if not base.endswith("_hr.dat"):
        return None
    return base[:-7]


def infer_seed_from_paths(paths):
    seeds = sorted({hr_seed_from_path(path) for path in paths if hr_seed_from_path(path)})
    if len(seeds) == 1:
        return seeds[0]
    if not seeds:
        raise ValueError("Could not infer seed: no *_hr.dat candidates found")
    raise ValueError(f"Could not infer unique seed from {seeds}; pass --seed explicitly")


def list_seeded_hr_candidates(root, seed):
    pattern = os.path.join(os.fspath(root), "**", f"{seed}_hr.dat")
    return sorted(glob.glob(pattern, recursive=True))


def choose_unique_candidate(candidates, role, exclude=None):
    exclude = {os.path.abspath(path) for path in (exclude or []) if path}
    filtered = [path for path in candidates if os.path.abspath(path) not in exclude]
    if len(filtered) == 1:
        return filtered[0]
    if not filtered:
        raise ValueError(f"Could not find {role} hr.dat candidate")
    joined = "\n  ".join(filtered)
    raise ValueError(
        f"Ambiguous {role} hr.dat candidates. Pass --{role}-hr explicitly:\n  {joined}"
    )


def find_seeded_sidecar(anchor_hr, seed, suffix):
    anchor_dir = os.path.dirname(os.path.abspath(anchor_hr))
    candidates = [
        os.path.join(anchor_dir, f"{seed}{suffix}"),
        os.path.join(os.path.dirname(anchor_dir), f"{seed}{suffix}"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def find_hr_seed_sidecar(anchor_hr, suffix):
    seed = hr_seed_from_path(anchor_hr)
    if seed is None:
        return None
    return find_seeded_sidecar(anchor_hr, seed, suffix)


def resolve_role_dir(root, subdir):
    if not subdir:
        return None
    path = resolve_path(root, subdir)
    if not os.path.isdir(path):
        raise ValueError(f"Role directory does not exist: {path}")
    return path


def infer_blocks_from_win(win_path, dim, block_layout, centres_path=None):
    sequence = expand_projection_l_sequence(win_path)
    blocks = []
    start = 1
    for item in sequence:
        l_val = int(item["l"])
        spinless_size = 2 * l_val + 1
        if block_layout == "spinor-block":
            size = 2 * spinless_size
        elif block_layout == "separated-spin":
            size = spinless_size
        else:
            raise ValueError(f"Unknown block layout: {block_layout}")
        stop = start + size - 1
        blocks.append(
            {
                "start": start - 1,
                "stop": stop,
                "l": l_val,
                "atom_label": item["atom_label"],
            }
        )
        start = stop + 1

    expected_dim = start - 1
    compare_dim = dim if block_layout == "spinor-block" else dim // 2
    if expected_dim != compare_dim:
        raise ValueError(
            f"{win_path}: inferred block dimension {expected_dim} does not match "
            f"{block_layout} dimension {compare_dim}"
        )

    if centres_path and os.path.exists(centres_path):
        total = read_centres_count(centres_path)
        natoms = len(read_atom_labels_from_win(win_path))
        nwann = total - natoms
        if nwann != dim:
            raise ValueError(
                f"{centres_path}: inferred nwann={nwann} from centres count, expected dim={dim}"
            )

    return blocks


def build_ls_operator_separated_spin(dim, blocks, target_l=None):
    """Build L.S for [all up orbitals, all down orbitals] ordering."""
    if dim % 2 != 0:
        raise ValueError(f"Spinful Hamiltonian dimension must be even, got {dim}")
    n_orb = dim // 2
    op = np.zeros((dim, dim), dtype=np.complex128)
    used = []

    for block in blocks:
        l_val = block["l"]
        if target_l is not None and l_val != target_l:
            continue
        start, stop = block["start"], block["stop"]
        if stop > n_orb:
            raise ValueError(f"Orbital block {block} exceeds spinless dimension {n_orb}")
        size = stop - start
        expected = 2 * l_val + 1
        if size != expected:
            raise ValueError(
                f"Separated-spin block {block} has size {size}, but l={l_val} requires {expected}"
            )
        local = atomic_soc_block_diagonal([l_val], 1.0)
        op[start:stop, start:stop] += local[:size, :size]
        op[start + n_orb:stop + n_orb, start + n_orb:stop + n_orb] += local[size:, size:]
        op[start:stop, start + n_orb:stop + n_orb] += local[:size, size:]
        op[start + n_orb:stop + n_orb, start:stop] += local[size:, :size]
        used.append(block)

    if not used:
        raise ValueError(f"No orbital blocks selected for target_l={target_l}")
    return 0.5 * (op + op.conj().T), used


def build_ls_operator_spinor_blocks(dim, blocks, target_l=None):
    """Build L.S for atomic spinor blocks [orb1 up/down, orb2 up/down, ...]."""
    op = np.zeros((dim, dim), dtype=np.complex128)
    used = []

    for block in blocks:
        l_val = block["l"]
        if target_l is not None and l_val != target_l:
            continue
        start, stop = block["start"], block["stop"]
        if stop > dim:
            raise ValueError(f"Orbital block {block} exceeds dimension {dim}")
        size = stop - start
        expected = 2 * (2 * l_val + 1)
        if size != expected:
            raise ValueError(
                f"Spinor block {block} has size {size}, but l={l_val} requires {expected}"
            )

        local_separated = atomic_soc_block_diagonal([l_val], 1.0)
        n_local_orb = 2 * l_val + 1
        perm = []
        for iorb in range(n_local_orb):
            perm.append(iorb)
            perm.append(iorb + n_local_orb)
        local_interleaved = local_separated[np.ix_(perm, perm)]
        op[start:stop, start:stop] += local_interleaved
        used.append(block)

    if not used:
        raise ValueError(f"No orbital blocks selected for target_l={target_l}")
    return 0.5 * (op + op.conj().T), used


def build_ls_operator(dim, blocks, target_l=None, block_layout="spinor-block"):
    if block_layout == "spinor-block":
        return build_ls_operator_spinor_blocks(dim, blocks, target_l=target_l)
    if block_layout == "separated-spin":
        return build_ls_operator_separated_spin(dim, blocks, target_l=target_l)
    raise ValueError(f"Unknown block layout: {block_layout}")


def read_unit_cell_cart_from_win(path):
    """Read Wannier90 unit_cell_cart from a .win file in Angstrom."""
    path = os.fspath(path)
    with open(path, "r") as f:
        lines = f.readlines()

    in_block = False
    unit = "ang"
    vecs = []
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if not in_block:
            if low.startswith("begin") and "unit_cell_cart" in low:
                in_block = True
            continue
        if low.startswith("end") and "unit_cell_cart" in low:
            break
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        parts = stripped.split()
        if len(parts) == 1 and not vecs:
            unit = parts[0].lower()
            continue
        if len(parts) < 3:
            continue
        vecs.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if len(vecs) < 3:
        raise ValueError(f"{path}: unit_cell_cart has fewer than 3 vectors")
    cell = np.asarray(vecs[:3], dtype=float)
    if unit in ("ang", "angstrom", "angstroms"):
        return cell
    if unit in ("bohr", "a.u.", "au"):
        return cell * BOHR_TO_ANG
    raise ValueError(f"{path}: unsupported unit_cell_cart unit '{unit}'")


def group_shells_by_index(r_keys, num_shells, include_onsite=True):
    """Group R-vectors by integer index-space |R|^2."""
    grouped = {}
    for idx, r in enumerate(r_keys):
        shell_key = int(r[0]) ** 2 + int(r[1]) ** 2 + int(r[2]) ** 2
        grouped.setdefault(shell_key, []).append(idx)

    selected = []
    if include_onsite and 0 in grouped:
        selected.append({"name": "onsite", "key": 0, "indices": grouped[0]})

    nonzero_keys = sorted(k for k in grouped if k != 0)
    for key in nonzero_keys[: int(num_shells)]:
        selected.append({"name": f"shell_{key}", "key": key, "indices": grouped[key]})
    if not selected:
        raise ValueError("No R-vector shells selected")
    return selected


def group_shells_by_metric(r_keys, cell, num_shells, include_onsite=True, tol=1e-6):
    """Group R-vectors by Cartesian |R_cart| from unit_cell_cart."""
    r_arr = np.asarray(r_keys, dtype=float)
    r_cart = r_arr @ np.asarray(cell, dtype=float)
    distances = np.linalg.norm(r_cart, axis=1)

    onsite = []
    shell_groups = []
    for idx in np.argsort(distances):
        dist = float(distances[idx])
        if dist <= tol:
            onsite.append(int(idx))
            continue
        for group in shell_groups:
            if abs(group["distance_ang"] - dist) <= tol:
                group["indices"].append(int(idx))
                break
        else:
            shell_groups.append({"distance_ang": dist, "indices": [int(idx)]})

    selected = []
    if include_onsite and onsite:
        selected.append({"name": "onsite", "key": 0.0, "distance_ang": 0.0, "indices": onsite})

    for group in shell_groups[: int(num_shells)]:
        selected.append(
            {
                "name": f"shell_{group['distance_ang']:.6g}A",
                "key": group["distance_ang"],
                "distance_ang": group["distance_ang"],
                "indices": group["indices"],
            }
        )
    if not selected:
        raise ValueError("No R-vector shells selected")
    return selected


def k_phases(kpts, r_vecs):
    return np.exp(2j * np.pi * (np.asarray(kpts, dtype=float) @ np.asarray(r_vecs, dtype=float).T))


def shell_form_factors_from_phases(phase, shells):
    factors = np.empty((phase.shape[0], len(shells)), dtype=np.complex128)
    for ishell, shell in enumerate(shells):
        factors[:, ishell] = np.sum(phase[:, shell["indices"]], axis=1)
    return factors


def shell_form_factors(kpts, r_vecs, shells):
    phase = k_phases(kpts, r_vecs)
    return shell_form_factors_from_phases(phase, shells)


def operator_name(l_val):
    labels = {0: "s_ls", 1: "p_ls", 2: "d_ls", 3: "f_ls"}
    return labels.get(int(l_val), f"l{int(l_val)}_ls")


def block_mask(dim, blocks, target_l, block_layout):
    mask = np.zeros(dim, dtype=bool)
    if block_layout == "spinor-block":
        for block in blocks:
            if int(block["l"]) == int(target_l):
                mask[block["start"]:block["stop"]] = True
    elif block_layout == "separated-spin":
        if dim % 2 != 0:
            raise ValueError(f"Separated-spin dimension must be even, got {dim}")
        n_orb = dim // 2
        for block in blocks:
            if int(block["l"]) == int(target_l):
                mask[block["start"]:block["stop"]] = True
                mask[block["start"] + n_orb:block["stop"] + n_orb] = True
    else:
        raise ValueError(f"Unknown block layout: {block_layout}")
    if not np.any(mask):
        raise ValueError(f"No basis functions found for l={target_l}")
    return mask


def build_dp_hybrid_operator(coll_hr, blocks, block_layout, include_onsite=False):
    """Build R-dependent d-p hybrid operator dressed by p L.S."""
    p_ls, p_blocks = build_ls_operator(
        coll_hr.dim,
        blocks,
        target_l=1,
        block_layout=block_layout,
    )
    p_mask = block_mask(coll_hr.dim, blocks, 1, block_layout)
    d_mask = block_mask(coll_hr.dim, blocks, 2, block_layout)
    p_proj = np.diag(p_mask.astype(float))
    d_proj = np.diag(d_mask.astype(float))

    r_ops = np.empty_like(coll_hr.h_norm)
    for ir, h_r in enumerate(coll_hr.h_norm):
        if not include_onsite and tuple(coll_hr.r_keys[ir]) == (0, 0, 0):
            op_r = np.zeros_like(h_r)
        else:
            op_r = d_proj @ h_r @ p_proj @ p_ls + p_ls @ p_proj @ h_r @ d_proj
        r_ops[ir] = op_r
    return {
        "name": "dp_p_lsoc",
        "l": "dp",
        "kind": "rdep",
        "r_matrices": r_ops,
        "blocks": p_blocks,
    }


def real_harmonic_values(l_val, xyz):
    xyz = np.asarray(xyz, dtype=float)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    if int(l_val) == 1:
        return np.column_stack([z, x, y])
    if int(l_val) == 2:
        return np.column_stack([
            0.5 * (3.0 * z * z - 1.0),
            x * z,
            y * z,
            x * x - y * y,
            x * y,
        ])
    raise ValueError(f"SK real harmonics implemented only for p/d, got l={l_val}")


def real_harmonic_rotation(l_val, frame):
    samples = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [2.0, 1.0, 0.5],
            [0.5, 2.0, 1.0],
        ],
        dtype=float,
    )
    samples /= np.linalg.norm(samples, axis=1)[:, None]
    x_global = real_harmonic_values(l_val, samples)
    local_coords = samples @ np.asarray(frame, dtype=float)
    y_local_rotated = real_harmonic_values(l_val, local_coords)
    coeff, *_ = np.linalg.lstsq(x_global, y_local_rotated, rcond=None)
    return coeff


def bond_frame(unit):
    zhat = np.asarray(unit, dtype=float)
    zhat /= np.linalg.norm(zhat)
    ref = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, zhat))) > 0.9:
        ref = np.asarray([1.0, 0.0, 0.0])
    xhat = np.cross(ref, zhat)
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    return np.column_stack([xhat, yhat, zhat])


def sk_pd_sigma_pi(unit):
    frame = bond_frame(unit)
    d_p = real_harmonic_rotation(1, frame)
    d_d = real_harmonic_rotation(2, frame)
    local_sigma = np.zeros((3, 5), dtype=float)
    local_pi = np.zeros((3, 5), dtype=float)
    local_sigma[0, 0] = 1.0
    local_pi[1, 1] = 1.0
    local_pi[2, 2] = 1.0
    return d_p @ local_sigma @ d_d.T, d_p @ local_pi @ d_d.T


def block_orbital_index(block, iorb, spin, dim, block_layout):
    if block_layout == "spinor-block":
        return block["start"] + 2 * iorb + spin
    n_orb = dim // 2
    return block["start"] + iorb + spin * n_orb


def pauli_matrices():
    return {
        "s0": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
        "sx": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "sy": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        "sz": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    }


def default_spin_channels(spin_frame):
    frame = str(spin_frame).strip().lower()
    if frame in {"c3", "bond"}:
        return ("transverse", "normal")
    return ("sx", "sy")


def normalize_spin_channels(channels, spin_frame="global"):
    if channels is None:
        channels = default_spin_channels(spin_frame)
    allowed = {"s0", "sx", "sy", "sz", "radial", "transverse", "normal"}
    out = []
    for channel in channels:
        key = str(channel).strip().lower()
        if key not in out:
            if key not in allowed:
                raise ValueError(
                    f"Unknown spin channel {channel!r}; use any of "
                    "s0,sx,sy,sz,radial,transverse,normal"
                )
            out.append(key)
    return out


def spin_matrix_for_channel(channel, bond_unit, layer_normal):
    pauli = pauli_matrices()
    channel = str(channel).strip().lower()
    if channel in pauli:
        return pauli[channel]

    normal = np.asarray(layer_normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    radial = np.asarray(bond_unit, dtype=float)
    radial = radial - normal * float(np.dot(radial, normal))
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1e-12:
        radial = np.asarray(bond_unit, dtype=float)
        radial_norm = float(np.linalg.norm(radial))
    radial = radial / radial_norm
    transverse = np.cross(normal, radial)
    transverse = transverse / np.linalg.norm(transverse)

    if channel == "radial":
        vec = radial
    elif channel == "transverse":
        vec = transverse
    elif channel == "normal":
        vec = normal
    else:
        raise ValueError(f"Unknown spin channel {channel!r}")
    return vec[0] * pauli["sx"] + vec[1] * pauli["sy"] + vec[2] * pauli["sz"]


def local_spin_moments_from_evecs(evecs, blocks, components, nocc, dim, block_layout):
    nocc = int(nocc)
    if nocc <= 0 or nocc > int(dim):
        raise ValueError(f"moment occupation must be in [1, {dim}], got {nocc}")
    moments = np.zeros((len(blocks), len(components)), dtype=float)
    nk = int(evecs.shape[0])
    for iblock, block in enumerate(blocks):
        l_val = int(block["l"])
        norb = 2 * l_val + 1
        for iorb in range(norb):
            up = block_orbital_index(block, iorb, 0, dim, block_layout)
            dn = block_orbital_index(block, iorb, 1, dim, block_layout)
            cu = evecs[:, up, :nocc]
            cd = evecs[:, dn, :nocc]
            coh = np.conj(cu) * cd
            for icomp, comp in enumerate(components):
                if comp == "x":
                    moments[iblock, icomp] += float(np.sum(2.0 * np.real(coh)))
                elif comp == "y":
                    moments[iblock, icomp] += float(np.sum(2.0 * np.imag(coh)))
                elif comp == "z":
                    moments[iblock, icomp] += float(np.sum(np.abs(cu) ** 2 - np.abs(cd) ** 2))
    return moments / float(nk)


def local_spin_moments(hk, blocks, components, nocc, dim, block_layout):
    _evals, evecs = np.linalg.eigh(hk)
    return local_spin_moments_from_evecs(evecs, blocks, components, nocc, dim, block_layout)


def block_center_fractional(block, centres_frac, dim, block_layout):
    l_val = int(block["l"])
    norb = 2 * l_val + 1
    indices = [block_orbital_index(block, iorb, 0, dim, block_layout) for iorb in range(norb)]
    return np.mean(centres_frac[indices], axis=0)


def atom_fractional_positions_from_win(win_path):
    labels, frac, _numbers = read_atoms_from_win(win_path)
    pos = {}
    for label, coord in zip(labels, frac):
        if label in pos:
            raise ValueError(f"{win_path}: duplicate atom label '{label}' cannot be used for SK geometry")
        pos[label] = np.asarray(coord, dtype=float)
    return pos


def block_atom_fractional(block, atom_frac_by_label, win_path):
    label = block.get("atom_label")
    if label not in atom_frac_by_label:
        raise ValueError(
            f"{win_path}: block atom label '{label}' was not found in atoms_frac/atoms_cart"
        )
    return atom_frac_by_label[label]


def require_spglib_symmetry(win_path, symprec):
    try:
        import spglib
    except ImportError as exc:
        raise RuntimeError("--include-dp-sk requires spglib for symmetry-orbit grouping") from exc

    labels, frac, numbers = read_atoms_from_win(win_path)
    cell = read_unit_cell_cart_from_win(win_path)
    sym = spglib.get_symmetry((cell, frac, numbers), symprec=float(symprec))
    if sym is None or len(sym.get("rotations", [])) == 0:
        raise RuntimeError(f"spglib failed to find symmetry operations for {win_path}")
    return {
        "labels": labels,
        "frac": np.asarray(frac, dtype=float),
        "numbers": np.asarray(numbers, dtype=int),
        "cell": np.asarray(cell, dtype=float),
        "rotations": np.asarray(sym["rotations"], dtype=int),
        "translations": np.asarray(sym["translations"], dtype=float),
    }


def wrap_fractional_delta(delta):
    return np.asarray(delta, dtype=float) - np.rint(delta)


def cart_rotation_from_fractional(rotation, cell):
    """Cartesian column-vector rotation for a spglib fractional rotation."""
    cell = np.asarray(cell, dtype=float)
    return cell.T @ np.asarray(rotation, dtype=float) @ np.linalg.inv(cell.T)


def symmetry_site_mapping(rotation, translation, frac, numbers, cell, tol):
    rotation = np.asarray(rotation, dtype=int)
    translation = np.asarray(translation, dtype=float)
    frac = np.asarray(frac, dtype=float)
    numbers = np.asarray(numbers, dtype=int)
    cell = np.asarray(cell, dtype=float)
    site_map = np.full(len(frac), -1, dtype=int)
    shifts = np.zeros((len(frac), 3), dtype=int)

    transformed = frac @ rotation.T + translation
    for iatom, pos in enumerate(transformed):
        same_species = np.flatnonzero(numbers == numbers[iatom])
        deltas = pos[None, :] - frac[same_species]
        wrapped = deltas - np.rint(deltas)
        distances = np.linalg.norm(wrapped @ cell, axis=1)
        ibest = int(np.argmin(distances))
        if float(distances[ibest]) > float(tol):
            raise ValueError(
                "Failed to map atom under symmetry operation: "
                f"atom={iatom + 1}, min_distance={float(distances[ibest]):.6g} A"
            )
        target = int(same_species[ibest])
        site_map[iatom] = target
        shifts[iatom] = np.rint(pos - frac[target]).astype(int)
    return site_map, shifts


def read_spin_texture(path):
    """Read spin texture rows: atom_label mx my mz. Zero vectors mark nonmagnetic sites."""
    spins = {}
    with open(path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = _strip_inline_comment(line)
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 4:
                raise ValueError(f"{path}:{lineno}: spin texture row must be 'label mx my mz'")
            label = parts[0]
            vec = np.asarray([float(x) for x in parts[1:4]], dtype=float)
            if label in spins:
                raise ValueError(f"{path}:{lineno}: duplicate spin texture label {label!r}")
            spins[label] = vec
    if not spins:
        raise ValueError(f"{path}: no spin texture rows were read")
    return spins


def spin_vectors_for_atoms(labels, spin_texture, path):
    vectors = []
    for label in labels:
        if label in spin_texture:
            vectors.append(spin_texture[label])
            continue
        species = species_name(label)
        if species in spin_texture:
            vectors.append(spin_texture[species])
            continue
        vectors.append(np.zeros(3, dtype=float))
    return np.asarray(vectors, dtype=float)


def spin_rotation_preserves_texture(site_map, spin_vectors, axial_rotation, time_reversal, tol):
    sign = -1.0 if time_reversal else 1.0
    transformed = sign * (np.asarray(spin_vectors, dtype=float) @ np.asarray(axial_rotation, dtype=float).T)
    target = spin_vectors[np.asarray(site_map, dtype=int)]
    scale = np.maximum(np.linalg.norm(target, axis=1), 1.0)
    err = np.linalg.norm(transformed - target, axis=1) / scale
    return bool(np.max(err) <= float(tol))


def magnetic_symmetry_operations(
    sym_data,
    *,
    mode,
    spin_texture=None,
    include_time_reversal=False,
    spin_tol=1e-5,
    map_tol=1e-5,
):
    mode = str(mode).strip().lower()
    if mode not in {"none", "crystal", "magnetic"}:
        raise ValueError(f"Unknown d-p SK symmetry mode {mode!r}; use none, crystal, or magnetic")
    if mode == "none":
        return []

    labels = list(sym_data["labels"])
    frac = np.asarray(sym_data["frac"], dtype=float)
    numbers = np.asarray(sym_data["numbers"], dtype=int)
    cell = np.asarray(sym_data["cell"], dtype=float)
    spin_vectors = None
    if mode == "magnetic":
        if spin_texture is None:
            raise ValueError("--dp-sk-symmetry magnetic requires --spin-texture")
        spin_vectors = spin_vectors_for_atoms(labels, spin_texture, "--spin-texture")

    operations = []
    for rotation, translation in zip(sym_data["rotations"], sym_data["translations"]):
        site_map, shifts = symmetry_site_mapping(rotation, translation, frac, numbers, cell, map_tol)
        polar_cart = cart_rotation_from_fractional(rotation, cell)
        axial_cart = float(np.linalg.det(polar_cart)) * polar_cart
        for time_reversal in ((False, True) if include_time_reversal else (False,)):
            if mode == "magnetic" and not spin_rotation_preserves_texture(
                site_map,
                spin_vectors,
                axial_cart,
                time_reversal,
                spin_tol,
            ):
                continue
            operations.append(
                {
                    "rotation": np.asarray(rotation, dtype=int),
                    "translation": np.asarray(translation, dtype=float),
                    "site_map": site_map,
                    "shifts": shifts,
                    "polar_cart": polar_cart,
                    "axial_cart": axial_cart,
                    "time_reversal": bool(time_reversal),
                }
            )
    if not operations:
        raise ValueError(f"No {mode} symmetry operations survived the d-p SK symmetry filter")
    return operations


def group_bonds_by_symmetry_orbit(bonds, rotations, cell, tol):
    """Group d-p bond records by species pair and spglib rotation orbit."""
    cell = np.asarray(cell, dtype=float)
    unused = set(range(len(bonds)))
    groups = []
    while unused:
        seed_idx = min(unused)
        seed = bonds[seed_idx]
        orbit = []
        targets = [np.asarray(seed["bond_frac"] @ rot.T, dtype=float) for rot in rotations]
        for idx in list(unused):
            cand = bonds[idx]
            if cand["pair_key"] != seed["pair_key"]:
                continue
            diffs = np.asarray([wrap_fractional_delta(cand["bond_frac"] - target) @ cell for target in targets])
            if np.min(np.linalg.norm(diffs, axis=1)) <= tol:
                orbit.append(idx)
        for idx in orbit:
            unused.remove(idx)
        groups.append(orbit)
    groups.sort(
        key=lambda group: (
            float(np.mean([bonds[idx]["distance_ang"] for idx in group])),
            min(group),
        )
    )
    return groups


def group_bonds_by_distance_shell(bonds, tol):
    groups = []
    for idx, bond in enumerate(sorted(range(len(bonds)), key=lambda i: (bonds[i]["distance_ang"], i))):
        dist = float(bonds[bond]["distance_ang"])
        pair_key = bonds[bond]["pair_key"]
        placed = False
        for group in groups:
            ref = bonds[group[0]]
            if ref["pair_key"] == pair_key and abs(float(ref["distance_ang"]) - dist) <= tol:
                group.append(bond)
                placed = True
                break
        if not placed:
            groups.append([bond])
    groups.sort(
        key=lambda group: (
            float(np.mean([bonds[idx]["distance_ang"] for idx in group])),
            min(group),
        )
    )
    return groups


def filter_bonds_by_distance_shells(bonds, num_shells, tol):
    if num_shells is None:
        return bonds
    num_shells = int(num_shells)
    if num_shells <= 0:
        return bonds
    ordered = sorted(bonds, key=lambda item: item["distance_ang"])
    selected = []
    shell_distances = []
    for bond in ordered:
        dist = float(bond["distance_ang"])
        if not shell_distances or abs(shell_distances[-1] - dist) > tol:
            if len(shell_distances) >= num_shells:
                break
            shell_distances.append(dist)
        selected.append(bond)
    return selected


def proper_rotation_to_su2(rotation):
    rotation = np.asarray(rotation, dtype=float)
    det = float(np.linalg.det(rotation))
    if abs(det - 1.0) > 1e-6:
        raise ValueError(f"Spin SU(2) representation needs a proper rotation, det={det:.6g}")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diag = np.diag(rotation)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qw, qx, qy, qz], dtype=float)
    quat /= np.linalg.norm(quat)
    qw, qx, qy, qz = quat
    return np.asarray(
        [
            [qw - 1.0j * qz, -qy - 1.0j * qx],
            [qy - 1.0j * qx, qw + 1.0j * qz],
        ],
        dtype=np.complex128,
    )


def block_occurrence_keys(blocks):
    counts = {}
    keys = {}
    for iblock, block in enumerate(blocks):
        key = (block.get("atom_label"), int(block["l"]))
        counts[key] = counts.get(key, 0) + 1
        keys[iblock] = (key[0], key[1], counts[key])
    return keys


def build_symmetry_basis_transform(operation, blocks, atom_labels, dim, block_layout):
    label_to_atom = {label: iatom for iatom, label in enumerate(atom_labels)}
    block_keys = block_occurrence_keys(blocks)
    target_by_key = {value: blocks[iblock] for iblock, value in block_keys.items()}

    spin_rotation = proper_rotation_to_su2(operation["axial_cart"])
    if operation.get("time_reversal", False):
        spin_rotation = spin_rotation @ np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)

    transform = np.zeros((dim, dim), dtype=np.complex128)
    basis_shifts = np.zeros((dim, 3), dtype=int)

    for iblock, block in enumerate(blocks):
        atom_label = block.get("atom_label")
        if atom_label not in label_to_atom:
            raise ValueError(f"Cannot symmetrize block without a valid atom label: {block}")
        source_atom = label_to_atom[atom_label]
        target_atom = int(operation["site_map"][source_atom])
        target_label = atom_labels[target_atom]
        key = block_keys[iblock]
        target_key = (target_label, key[1], key[2])
        if target_key not in target_by_key:
            raise ValueError(f"Symmetry maps block {key} to missing block {target_key}")
        target_block = target_by_key[target_key]
        l_val = int(block["l"])
        norb = 2 * l_val + 1
        orbital_rotation = real_harmonic_rotation(l_val, operation["polar_cart"].T)
        shift = np.asarray(operation["shifts"][source_atom], dtype=int)
        for source_orb in range(norb):
            for source_spin in range(2):
                source_idx = block_orbital_index(block, source_orb, source_spin, dim, block_layout)
                basis_shifts[source_idx] = shift
                for target_orb in range(norb):
                    orb_value = orbital_rotation[target_orb, source_orb]
                    if abs(orb_value) <= 1e-14:
                        continue
                    for target_spin in range(2):
                        spin_value = spin_rotation[target_spin, source_spin]
                        if abs(spin_value) <= 1e-14:
                            continue
                        target_idx = block_orbital_index(
                            target_block,
                            target_orb,
                            target_spin,
                            dim,
                            block_layout,
                        )
                        transform[target_idx, source_idx] += orb_value * spin_value
    return transform, basis_shifts


def transform_r_matrices_by_symmetry(
    r_matrices,
    operation,
    transform,
    basis_shifts,
    r_keys,
    r_to_ir,
    *,
    missing_r_policy="skip",
    zero_tol=1e-14,
    stats=None,
):
    source = np.asarray(r_matrices, dtype=np.complex128)
    if operation.get("time_reversal", False):
        source = np.conj(source)
    out = np.zeros_like(source)
    rotation = np.asarray(operation["rotation"], dtype=int)
    r_arr = np.asarray(r_keys, dtype=int)
    stats = stats if stats is not None else {}
    unique_shifts = np.unique(basis_shifts, axis=0)
    for row_shift in unique_shifts:
        row_idx = np.flatnonzero(np.all(basis_shifts == row_shift, axis=1))
        left = transform[:, row_idx]
        if not np.any(np.abs(left) > zero_tol):
            continue
        for col_shift in unique_shifts:
            col_idx = np.flatnonzero(np.all(basis_shifts == col_shift, axis=1))
            right = transform[:, col_idx]
            if not np.any(np.abs(right) > zero_tol):
                continue
            block = source[:, row_idx[:, None], col_idx]
            active = np.linalg.norm(block.reshape(block.shape[0], -1), axis=1) > zero_tol
            if not np.any(active):
                continue
            r_new = r_arr @ rotation.T + np.asarray(col_shift - row_shift, dtype=int)
            ir_new = np.full(len(r_new), -1, dtype=int)
            missing = []
            for ir, rvec in enumerate(r_new):
                if not active[ir]:
                    continue
                key = tuple(int(x) for x in rvec)
                if key in r_to_ir:
                    ir_new[ir] = r_to_ir[key]
                else:
                    missing.append((ir, key))
            if missing:
                if missing_r_policy == "error":
                    raise ValueError(
                        "Symmetry-projected d-p SK term needs an R-vector absent from hr.dat: "
                        f"{missing[0][1]}"
                    )
                if missing_r_policy != "skip":
                    raise ValueError(
                        f"Unknown missing R-vector policy {missing_r_policy!r}; use error or skip"
                    )
                stats["missing_r_skipped"] = int(stats.get("missing_r_skipped", 0)) + len(missing)
            keep = active & (ir_new >= 0)
            if not np.any(keep):
                continue
            transformed = np.einsum("as,rst,bt->rab", left, block[keep], right.conj(), optimize=True)
            ir_new = ir_new[keep]
            np.add.at(out, ir_new, transformed)
    return out


def project_r_matrices_to_symmetry(
    r_matrices,
    operations,
    blocks,
    atom_labels,
    dim,
    block_layout,
    r_keys,
    *,
    missing_r_policy="skip",
    zero_tol=1e-14,
    stats=None,
):
    if not operations:
        return np.asarray(r_matrices, dtype=np.complex128)
    r_to_ir = {tuple(int(x) for x in r): ir for ir, r in enumerate(r_keys)}
    projected = np.zeros_like(r_matrices, dtype=np.complex128)
    stats = stats if stats is not None else {}
    transforms = [
        (
            operation,
            *build_symmetry_basis_transform(operation, blocks, atom_labels, dim, block_layout),
        )
        for operation in operations
    ]
    for operation, transform, basis_shifts in transforms:
        projected += transform_r_matrices_by_symmetry(
            r_matrices,
            operation,
            transform,
            basis_shifts,
            r_keys,
            r_to_ir,
            missing_r_policy=missing_r_policy,
            zero_tol=zero_tol,
            stats=stats,
        )
    projected /= float(len(operations))
    return projected


def term_vector_real(r_matrices):
    flat = np.ravel(np.asarray(r_matrices, dtype=np.complex128))
    return np.concatenate([flat.real, flat.imag])


def prune_dependent_terms(terms, tol):
    kept = []
    basis = []
    tol = float(tol)
    for term in terms:
        vec = term_vector_real(term["r_matrices"])
        norm = float(np.linalg.norm(vec))
        term.setdefault("metadata", {})["symmetry_norm"] = norm
        if norm <= tol:
            continue
        residual = np.array(vec, copy=True)
        for basis_vec in basis:
            residual -= float(np.dot(basis_vec, residual)) * basis_vec
        residual_norm = float(np.linalg.norm(residual))
        rel = residual_norm / max(norm, 1.0)
        term["metadata"]["symmetry_independent_residual"] = residual_norm
        if rel <= tol:
            continue
        basis.append(residual / residual_norm)
        kept.append(term)
    if not kept:
        raise ValueError("All d-p SK terms vanished or became linearly dependent after symmetry projection")
    return kept


def project_terms_to_symmetry(
    terms,
    operations,
    blocks,
    atom_labels,
    dim,
    block_layout,
    r_keys,
    tol,
    *,
    missing_r_policy="skip",
):
    if not operations:
        return terms
    projected = []
    for term in terms:
        new_term = dict(term)
        metadata = dict(term.get("metadata", {}))
        pre_norm = float(np.linalg.norm(term_vector_real(term["r_matrices"])))
        stats = {}
        new_term["r_matrices"] = project_r_matrices_to_symmetry(
            term["r_matrices"],
            operations,
            blocks,
            atom_labels,
            dim,
            block_layout,
            r_keys,
            missing_r_policy=missing_r_policy,
            zero_tol=max(float(tol) * 1.0e-6, 1.0e-14),
            stats=stats,
        )
        metadata["symmetry_projected"] = True
        metadata["symmetry_operations"] = len(operations)
        metadata["pre_projection_norm"] = pre_norm
        metadata["post_projection_norm"] = float(np.linalg.norm(term_vector_real(new_term["r_matrices"])))
        metadata["antiunitary_operations"] = int(sum(1 for op in operations if op.get("time_reversal", False)))
        metadata["missing_r_policy"] = missing_r_policy
        metadata["missing_r_skipped"] = int(stats.get("missing_r_skipped", 0))
        new_term["metadata"] = metadata
        projected.append(new_term)
    return prune_dependent_terms(projected, tol)


def build_dp_sk_operators(coll_hr, blocks, block_layout, centres_path, win_path):
    if centres_path is None or not os.path.exists(centres_path):
        raise ValueError("--include-dp-sk requires centres.xyz; pass --centres explicitly")
    atom_labels = read_atom_labels_from_win(win_path)
    centres = read_centres_coords(centres_path, coll_hr.dim, len(atom_labels), block_layout)
    cell = read_unit_cell_cart_from_win(win_path)
    p_ls, p_blocks = build_ls_operator(
        coll_hr.dim,
        blocks,
        target_l=1,
        block_layout=block_layout,
    )
    p_mask = block_mask(coll_hr.dim, blocks, 1, block_layout)
    d_mask = block_mask(coll_hr.dim, blocks, 2, block_layout)
    p_proj = np.diag(p_mask.astype(float))
    d_proj = np.diag(d_mask.astype(float))

    sigma_h = np.zeros_like(coll_hr.h_norm)
    pi_h = np.zeros_like(coll_hr.h_norm)
    d_blocks = [b for b in blocks if int(b["l"]) == 2]
    p_blocks_only = [b for b in blocks if int(b["l"]) == 1]
    for ir, r in enumerate(coll_hr.r_keys):
        r_cart = np.asarray(r, dtype=float) @ cell
        h_sigma = np.zeros((coll_hr.dim, coll_hr.dim), dtype=np.complex128)
        h_pi = np.zeros_like(h_sigma)
        for d_block in d_blocks:
            for p_block in p_blocks_only:
                for id_orb in range(5):
                    for ip_orb in range(3):
                        for spin in range(2):
                            if block_layout == "spinor-block":
                                d_idx = d_block["start"] + 2 * id_orb + spin
                                p_idx = p_block["start"] + 2 * ip_orb + spin
                            else:
                                n_orb = coll_hr.dim // 2
                                d_idx = d_block["start"] + id_orb + spin * n_orb
                                p_idx = p_block["start"] + ip_orb + spin * n_orb
                            bond = r_cart + centres[p_idx] - centres[d_idx]
                            norm = float(np.linalg.norm(bond))
                            if norm < 1e-10:
                                continue
                            sk_sigma, sk_pi = sk_pd_sigma_pi(bond / norm)
                            val_sigma = sk_sigma[ip_orb, id_orb]
                            val_pi = sk_pi[ip_orb, id_orb]
                            h_sigma[p_idx, d_idx] += val_sigma
                            h_sigma[d_idx, p_idx] += val_sigma
                            h_pi[p_idx, d_idx] += val_pi
                            h_pi[d_idx, p_idx] += val_pi
        sigma_h[ir] = d_proj @ h_sigma @ p_proj @ p_ls + p_ls @ p_proj @ h_sigma @ d_proj
        pi_h[ir] = d_proj @ h_pi @ p_proj @ p_ls + p_ls @ p_proj @ h_pi @ d_proj
    return [
        {"name": "dp_sk_sigma", "l": "dp", "kind": "rdep", "r_matrices": sigma_h, "blocks": p_blocks},
        {"name": "dp_sk_pi", "l": "dp", "kind": "rdep", "r_matrices": pi_h, "blocks": p_blocks},
    ]


def build_dp_sk_orbit_terms(
    coll_hr,
    blocks,
    block_layout,
    centres_path,
    win_path,
    symprec,
    tol,
    num_shells,
    spin_dependent=False,
    spin_channels=None,
    grouping="orbit",
    spin_frame="global",
    symmetry_mode="none",
    spin_texture=None,
    include_time_reversal=False,
    spin_symprec=1e-5,
    projector_tol=1e-8,
    missing_r_policy="skip",
):
    if centres_path is None or not os.path.exists(centres_path):
        raise ValueError("--include-dp-sk requires centres.xyz; pass --centres explicitly")
    atom_labels = read_atom_labels_from_win(win_path)
    cell = read_unit_cell_cart_from_win(win_path)
    atom_frac_by_label = atom_fractional_positions_from_win(win_path)
    sym_data = require_spglib_symmetry(win_path, symprec)
    rotations = sym_data["rotations"]
    symmetry_operations = magnetic_symmetry_operations(
        sym_data,
        mode=symmetry_mode,
        spin_texture=spin_texture,
        include_time_reversal=include_time_reversal,
        spin_tol=spin_symprec,
        map_tol=max(float(symprec), float(tol)),
    )

    p_ls, p_blocks = build_ls_operator(
        coll_hr.dim,
        blocks,
        target_l=1,
        block_layout=block_layout,
    )
    d_blocks = [b for b in blocks if int(b["l"]) == 2]
    p_blocks_only = [b for b in blocks if int(b["l"]) == 1]
    if not d_blocks or not p_blocks_only:
        raise ValueError("--include-dp-sk requires both d and p blocks")

    bonds = []
    for ir, r in enumerate(coll_hr.r_keys):
        r_frac = np.asarray(r, dtype=float)
        for id_block, d_block in enumerate(d_blocks):
            d_frac = block_atom_fractional(d_block, atom_frac_by_label, win_path)
            d_label = d_block.get("atom_label", f"d{id_block + 1}")
            for ip_block, p_block in enumerate(p_blocks_only):
                p_frac = block_atom_fractional(p_block, atom_frac_by_label, win_path)
                p_label = p_block.get("atom_label", f"p{ip_block + 1}")
                bond_frac = r_frac + p_frac - d_frac
                bond_cart = bond_frac @ cell
                distance = float(np.linalg.norm(bond_cart))
                if distance <= tol:
                    continue
                bonds.append(
                    {
                        "ir": ir,
                        "r": tuple(r),
                        "d_block": d_block,
                        "p_block": p_block,
                        "bond_frac": bond_frac,
                        "distance_ang": distance,
                        "pair_key": (species_name(d_label), species_name(p_label)),
                    }
                )
    if not bonds:
        raise ValueError("No d-p bonds were generated for SK orbit terms")
    bonds = filter_bonds_by_distance_shells(bonds, num_shells, tol)
    if not bonds:
        raise ValueError("No d-p bonds remain after distance-shell selection")

    grouping = str(grouping).strip().lower()
    if grouping == "orbit":
        groups = group_bonds_by_symmetry_orbit(bonds, rotations, cell, tol)
    elif grouping == "shell":
        groups = group_bonds_by_distance_shell(bonds, tol)
    else:
        raise ValueError(f"Unknown d-p SK grouping {grouping!r}; use orbit or shell")
    r_to_ir = {tuple(r): ir for ir, r in enumerate(coll_hr.r_keys)}
    terms = []
    spin_frame = str(spin_frame).strip().lower()
    if spin_frame not in {"global", "c3", "bond"}:
        raise ValueError(f"Unknown --dp-sk-spin-frame {spin_frame!r}; use global, c3, or bond")
    spin_channels = normalize_spin_channels(spin_channels, spin_frame) if spin_dependent else []
    layer_normal = np.asarray(cell[2], dtype=float)
    layer_normal = layer_normal / np.linalg.norm(layer_normal)
    group_label = "orbit" if grouping == "orbit" else "shell"
    for iorbit, group in enumerate(groups):
        sigma_base = np.zeros_like(coll_hr.h_norm)
        pi_base = np.zeros_like(coll_hr.h_norm)
        sigma_spin = {
            name: np.zeros_like(coll_hr.h_norm) for name in spin_channels
        }
        pi_spin = {
            name: np.zeros_like(coll_hr.h_norm) for name in spin_channels
        }
        for ibond in group:
            bond = bonds[ibond]
            ir = bond["ir"]
            r_neg = tuple(-int(x) for x in bond["r"])
            if r_neg not in r_to_ir:
                raise ValueError(f"Missing Hermitian partner R={r_neg} for SK bond R={bond['r']}")
            ir_neg = r_to_ir[r_neg]
            unit = (bond["bond_frac"] @ cell) / bond["distance_ang"]
            sk_sigma, sk_pi = sk_pd_sigma_pi(unit)
            d_block = bond["d_block"]
            p_block = bond["p_block"]
            for id_orb in range(5):
                for ip_orb in range(3):
                    val_sigma = sk_sigma[ip_orb, id_orb]
                    val_pi = sk_pi[ip_orb, id_orb]
                    if abs(val_sigma) <= 1e-14 and abs(val_pi) <= 1e-14:
                        continue
                    for spin in range(2):
                        d_idx = block_orbital_index(d_block, id_orb, spin, coll_hr.dim, block_layout)
                        p_idx = block_orbital_index(p_block, ip_orb, spin, coll_hr.dim, block_layout)
                        sigma_base[ir, d_idx, p_idx] += val_sigma
                        sigma_base[ir_neg, p_idx, d_idx] += np.conj(val_sigma)
                        pi_base[ir, d_idx, p_idx] += val_pi
                        pi_base[ir_neg, p_idx, d_idx] += np.conj(val_pi)
                    for spin_name in spin_channels:
                        spin_matrix = spin_matrix_for_channel(spin_name, unit, layer_normal)
                        for d_spin in range(2):
                            d_idx = block_orbital_index(d_block, id_orb, d_spin, coll_hr.dim, block_layout)
                            for p_spin in range(2):
                                spin_value = spin_matrix[d_spin, p_spin]
                                if abs(spin_value) <= 1e-14:
                                    continue
                                p_idx = block_orbital_index(p_block, ip_orb, p_spin, coll_hr.dim, block_layout)
                                sigma_val = val_sigma * spin_value
                                pi_val = val_pi * spin_value
                                sigma_spin[spin_name][ir, d_idx, p_idx] += sigma_val
                                sigma_spin[spin_name][ir_neg, p_idx, d_idx] += np.conj(sigma_val)
                                pi_spin[spin_name][ir, d_idx, p_idx] += pi_val
                                pi_spin[spin_name][ir_neg, p_idx, d_idx] += np.conj(pi_val)
        sigma_h = np.einsum("rij,jk->rik", sigma_base, p_ls, optimize=True)
        sigma_h += np.einsum("ij,rjk->rik", p_ls, sigma_base, optimize=True)
        pi_h = np.einsum("rij,jk->rik", pi_base, p_ls, optimize=True)
        pi_h += np.einsum("ij,rjk->rik", p_ls, pi_base, optimize=True)
        distances = np.asarray([bonds[idx]["distance_ang"] for idx in group], dtype=float)
        pair_keys = sorted({bonds[idx]["pair_key"] for idx in group})
        meta = {
            "orbit": iorbit,
            "grouping": grouping,
            "num_bonds": len(group),
            "distance_mean_ang": float(np.mean(distances)),
            "distance_std_ang": float(np.std(distances)),
            "distance_min_ang": float(np.min(distances)),
            "distance_max_ang": float(np.max(distances)),
            "pair_keys": ";".join(f"{a}-{b}" for a, b in pair_keys),
            "dp_sk_symmetry": symmetry_mode,
        }
        terms.append(
            {
                "name": f"dp_sk_sigma_{group_label}_{iorbit}",
                "operator": "dp_sk_sigma",
                "l": "dp",
                "kind": "rdep",
                "r_matrices": sigma_h,
                "metadata": meta,
                "blocks": p_blocks,
            }
        )
        terms.append(
            {
                "name": f"dp_sk_pi_{group_label}_{iorbit}",
                "operator": "dp_sk_pi",
                "l": "dp",
                "kind": "rdep",
                "r_matrices": pi_h,
                "metadata": meta,
                "blocks": p_blocks,
            }
        )
        for spin_name in spin_channels:
            terms.append(
                {
                    "name": f"dp_sk_sigma_{spin_name}_{group_label}_{iorbit}",
                    "operator": f"dp_sk_sigma_{spin_name}",
                    "l": "dp",
                    "kind": "rdep",
                    "r_matrices": sigma_spin[spin_name],
                    "metadata": dict(meta, spin_channel=spin_name, spin_frame=spin_frame),
                    "blocks": p_blocks,
                }
            )
            terms.append(
                {
                    "name": f"dp_sk_pi_{spin_name}_{group_label}_{iorbit}",
                    "operator": f"dp_sk_pi_{spin_name}",
                    "l": "dp",
                    "kind": "rdep",
                    "r_matrices": pi_spin[spin_name],
                    "metadata": dict(meta, spin_channel=spin_name, spin_frame=spin_frame),
                    "blocks": p_blocks,
                }
            )
    return project_terms_to_symmetry(
        terms,
        symmetry_operations,
        blocks,
        atom_labels,
        coll_hr.dim,
        block_layout,
        coll_hr.r_keys,
        projector_tol,
        missing_r_policy=missing_r_policy,
    )


def prepare_operator_shell_hk(operators, form_factors, phases, shells):
    prepared = []
    for op in operators:
        op_new = dict(op)
        shell_hk = []
        if op.get("kind", "static") == "static":
            for ishell in range(len(shells)):
                shell_hk.append(form_factors[:, ishell, None, None] * op["matrix"][None, :, :])
        elif op.get("kind") == "rdep":
            r_mats = op["r_matrices"]
            for shell in shells:
                idx = shell["indices"]
                shell_hk.append(np.einsum("kr,rij->kij", phases[:, idx], r_mats[idx], optimize=True))
        else:
            raise ValueError(f"Unknown operator kind: {op.get('kind')}")
        op_new["shell_hk"] = shell_hk
        prepared.append(op_new)
    return prepared


def build_static_onsite_term(coll_hr, name, l_val, matrix, used_blocks):
    r_mats = np.zeros_like(coll_hr.h_norm)
    for ir, r in enumerate(coll_hr.r_keys):
        if tuple(r) == (0, 0, 0):
            r_mats[ir] = matrix
            break
    else:
        raise ValueError("Input hr.dat has no R=(0,0,0) block for onsite L.S")
    return {
        "name": name,
        "operator": name,
        "l": int(l_val),
        "kind": "rdep",
        "r_matrices": r_mats,
        "metadata": {
            "orbit": "onsite",
            "num_bonds": 1,
            "distance_mean_ang": 0.0,
            "distance_std_ang": 0.0,
            "pair_keys": "",
        },
        "blocks": used_blocks,
    }


def prepare_terms_hk(terms, phases):
    prepared = []
    for term in terms:
        term_new = dict(term)
        if term.get("kind") != "rdep":
            raise ValueError(f"Term-based fitting expects rdep terms, got {term.get('kind')}")
        term_new["hk"] = np.einsum("kr,rij->kij", phases, term["r_matrices"], optimize=True)
        prepared.append(term_new)
    return prepared


def model_hk(hk_coll, terms, lambdas):
    lambdas = np.asarray(lambdas, dtype=float)
    if len(lambdas) != len(terms):
        raise ValueError(f"lambda count {len(lambdas)} does not match term count {len(terms)}")
    hk = np.array(hk_coll, copy=True)
    for value, term in zip(lambdas, terms):
        hk = hk + float(value) * term["hk"]
    return 0.5 * (hk + np.swapaxes(hk.conj(), -1, -2))


def k_chunk_slices(nk, batch_size):
    nk = int(nk)
    batch_size = nk if batch_size is None or int(batch_size) <= 0 else int(batch_size)
    return [slice(start, min(start + batch_size, nk)) for start in range(0, nk, batch_size)]


def model_hk_slice(hk_coll, terms, lambdas, slc):
    lambdas = np.asarray(lambdas, dtype=float)
    hk = np.array(hk_coll[slc], copy=True)
    for value, term in zip(lambdas, terms):
        hk += float(value) * term["hk"][slc]
    return 0.5 * (hk + np.swapaxes(hk.conj(), -1, -2))


def residual_chunk(
    slc,
    lambdas,
    hk_coll,
    terms,
    target_evals,
    bands,
    remove_center,
    moment_blocks,
    moment_components,
    moment_nocc,
    dim,
    block_layout,
):
    hk = model_hk_slice(hk_coll, terms, lambdas, slc)
    if moment_blocks:
        evals, evecs = np.linalg.eigh(hk)
    else:
        evals = np.linalg.eigvalsh(hk)
        evecs = None
    residual = evals[:, bands] - target_evals[slc][:, bands]
    if remove_center:
        residual = residual - np.mean(residual, axis=1, keepdims=True)
    moment_sum = None
    nk_chunk = int(hk.shape[0])
    if moment_blocks:
        moment_avg = local_spin_moments_from_evecs(
            evecs,
            moment_blocks,
            moment_components,
            moment_nocc,
            dim,
            block_layout,
        )
        moment_sum = moment_avg * float(nk_chunk)
    return np.ravel(residual), moment_sum, nk_chunk


def selected_residual_kparallel(
    lambdas,
    hk_coll,
    terms,
    target_evals,
    bands,
    remove_center=False,
    *,
    workers=1,
    batch_size=None,
    moment_blocks=None,
    moment_components=None,
    moment_nocc=None,
    dim=None,
    block_layout="spinor-block",
):
    nk = int(hk_coll.shape[0])
    chunks = k_chunk_slices(nk, batch_size)
    moment_blocks = list(moment_blocks or [])
    moment_components = list(moment_components or [])

    def run_one(slc):
        return residual_chunk(
            slc,
            lambdas,
            hk_coll,
            terms,
            target_evals,
            bands,
            remove_center,
            moment_blocks,
            moment_components,
            moment_nocc,
            dim if dim is not None else hk_coll.shape[1],
            block_layout,
        )

    if int(workers) <= 1 or len(chunks) <= 1:
        results = [run_one(slc) for slc in chunks]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as executor:
            results = list(executor.map(run_one, chunks))

    data_resid = np.concatenate([item[0] for item in results])
    moment_avg = None
    if moment_blocks:
        moment_sum = None
        total_nk = 0
        for _resid, chunk_moment_sum, nk_chunk in results:
            if moment_sum is None:
                moment_sum = np.zeros_like(chunk_moment_sum)
            moment_sum += chunk_moment_sum
            total_nk += int(nk_chunk)
        moment_avg = moment_sum / float(total_nk)
    return data_resid, moment_avg


def selected_residual(
    lambdas,
    hk_coll,
    terms,
    target_evals,
    bands,
    remove_center=False,
    weights=None,
):
    evals = np.linalg.eigvalsh(model_hk(hk_coll, terms, lambdas))
    residual = evals[:, bands] - target_evals[:, bands]
    if remove_center:
        residual = residual - np.mean(residual, axis=1, keepdims=True)
    if weights is not None:
        residual = residual * weights[:, None]
    return np.ravel(residual)


def ridge_initial_guess(hk_coll, terms, target_evals, bands, ridge):
    """Perturbative linear initial guess using collinear eigenvectors."""
    evals, evecs = np.linalg.eigh(hk_coll)
    y = target_evals[:, bands] - evals[:, bands]
    x_cols = []
    for term in terms:
        expvals = np.einsum(
            "kib,kij,kjb->kb",
            evecs[:, :, bands].conj(),
            term["hk"],
            evecs[:, :, bands],
            optimize=True,
        )
        x_cols.append(np.real(expvals))
    x = np.stack(x_cols, axis=-1).reshape(-1, len(x_cols))
    y_flat = np.ravel(y)
    a = x.T @ x + float(ridge) * np.eye(x.shape[1])
    b = x.T @ y_flat
    return np.linalg.solve(a, b)


def parse_term_bound_specs(items):
    specs = []
    for item in items or []:
        parts = str(item).replace(",", ":").split(":")
        if len(parts) != 3:
            raise ValueError(f"Term bound must look like name:lower:upper, got {item!r}")
        name, lower, upper = parts
        lo = -np.inf if lower.lower() in {"-inf", "none"} else float(lower)
        hi = np.inf if upper.lower() in {"inf", "+inf", "none"} else float(upper)
        if lo > hi:
            raise ValueError(f"Invalid bound for {name}: lower {lo} > upper {hi}")
        specs.append((name, lo, hi))
    return specs


def build_lambda_bounds(terms, *, onsite_positive=False, term_bounds=None):
    lower = np.full(len(terms), -np.inf, dtype=float)
    upper = np.full(len(terms), np.inf, dtype=float)
    if onsite_positive:
        for idx, term in enumerate(terms):
            if term.get("metadata", {}).get("orbit") == "onsite":
                lower[idx] = max(lower[idx], 0.0)
    name_to_indices = {}
    for idx, term in enumerate(terms):
        keys = {term["name"], term.get("operator", term["name"])}
        for key in keys:
            name_to_indices.setdefault(str(key), []).append(idx)
    for name, lo, hi in parse_term_bound_specs(term_bounds):
        if name not in name_to_indices:
            known = ", ".join(term["name"] for term in terms)
            raise ValueError(f"--term-bound name {name!r} matched no terms. Known terms: {known}")
        for idx in name_to_indices[name]:
            lower[idx] = max(lower[idx], lo)
            upper[idx] = min(upper[idx], hi)
            if lower[idx] > upper[idx]:
                raise ValueError(
                    f"Inconsistent bounds for {terms[idx]['name']}: "
                    f"lower {lower[idx]} > upper {upper[idx]}"
                )
    return lower, upper


def clip_initial_to_bounds(x0, lower, upper):
    x0 = np.asarray(x0, dtype=float)
    clipped = np.minimum(np.maximum(x0, lower), upper)
    finite_lo = np.isfinite(lower)
    clipped[finite_lo & (clipped == lower)] = lower[finite_lo & (clipped == lower)] + 1.0e-12
    finite_hi = np.isfinite(upper)
    clipped[finite_hi & (clipped == upper)] = upper[finite_hi & (clipped == upper)] - 1.0e-12
    return np.minimum(np.maximum(clipped, lower), upper)


def write_lambda_csv(path, terms, lambdas, lower_bounds=None, upper_bounds=None):
    lambdas = np.asarray(lambdas, dtype=float)
    lower_bounds = np.full_like(lambdas, np.nan) if lower_bounds is None else np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.full_like(lambdas, np.nan) if upper_bounds is None else np.asarray(upper_bounds, dtype=float)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "term",
                "operator",
                "l",
                "orbit",
                "grouping",
                "num_bonds",
                "distance_mean_ang",
                "distance_std_ang",
                "distance_min_ang",
                "distance_max_ang",
                "pair_keys",
                "spin_channel",
                "spin_frame",
                "dp_sk_symmetry",
                "symmetry_projected",
                "symmetry_operations",
                "antiunitary_operations",
                "missing_r_policy",
                "missing_r_skipped",
                "pre_projection_norm",
                "post_projection_norm",
                "lower_bound_eV",
                "upper_bound_eV",
                "lambda_eV",
            ]
        )
        for term, value, lo, hi in zip(terms, lambdas, lower_bounds, upper_bounds):
            meta = term.get("metadata", {})
            writer.writerow(
                [
                    term["name"],
                    term.get("operator", term["name"]),
                    term["l"],
                    meta.get("orbit", ""),
                    meta.get("grouping", ""),
                    meta.get("num_bonds", ""),
                    f"{float(meta.get('distance_mean_ang', np.nan)):.12g}",
                    f"{float(meta.get('distance_std_ang', np.nan)):.12g}",
                    f"{float(meta.get('distance_min_ang', np.nan)):.12g}",
                    f"{float(meta.get('distance_max_ang', np.nan)):.12g}",
                    meta.get("pair_keys", ""),
                    meta.get("spin_channel", ""),
                    meta.get("spin_frame", ""),
                    meta.get("dp_sk_symmetry", ""),
                    meta.get("symmetry_projected", ""),
                    meta.get("symmetry_operations", ""),
                    meta.get("antiunitary_operations", ""),
                    meta.get("missing_r_policy", ""),
                    meta.get("missing_r_skipped", ""),
                    meta.get("pre_projection_norm", ""),
                    meta.get("post_projection_norm", ""),
                    f"{float(lo):.12g}",
                    f"{float(hi):.12g}",
                    f"{float(value):.12g}",
                ]
            )


def write_fit_summary_csv(path, kpts, residual, selected_evals, target_evals, bands):
    residual_by_k = residual.reshape(len(kpts), len(bands))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ik", "kx", "ky", "kz", "rms_residual_eV", "max_abs_residual_eV"])
        for ik, k in enumerate(kpts):
            row = residual_by_k[ik]
            writer.writerow(
                [
                    ik + 1,
                    f"{k[0]:.12g}",
                    f"{k[1]:.12g}",
                    f"{k[2]:.12g}",
                    f"{np.sqrt(np.mean(row * row)):.12g}",
                    f"{np.max(np.abs(row)):.12g}",
                ]
            )


def write_fitted_hr(path, coll_hr, terms, lambdas):
    """Write raw hr.dat with fitted correction added."""
    lambdas = np.asarray(lambdas, dtype=float)
    dict_out = {}
    for ir, r in enumerate(coll_hr.r_keys):
        dict_out[r] = np.array(coll_hr.h_raw[ir], copy=True)

    for value, term in zip(lambdas, terms):
        if term.get("kind") != "rdep":
            raise ValueError(f"Term-based output expects rdep terms, got {term.get('kind')}")
        for ir, r in enumerate(coll_hr.r_keys):
            matrix = term["r_matrices"][ir]
            if np.any(np.abs(matrix) > 0.0):
                dict_out[r] = dict_out[r] + coll_hr.degens[ir] * float(value) * matrix

    write_wannier_hr(
        path,
        coll_hr.dim,
        coll_hr.degens,
        dict_out,
        header="SLW fitted nonlocal SOC Hamiltonian",
    )


def build_argparser():
    parser = argparse.ArgumentParser(description="Fit nonlocal shell SOC correction.")
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--root", type=str, default="src/soc")
    parser.add_argument("--wsoc-dir", type=str, default=None, help="Legacy alias for --target-dir")
    parser.add_argument("--wosoc-dir", type=str, default=None, help="Legacy alias for --coll-dir")
    parser.add_argument("--target-dir", type=str, default=None, help="Directory containing target SOC files")
    parser.add_argument("--coll-dir", type=str, default=None, help="Directory containing collinear/noSOC files")
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--coll-hr", type=str, default=None, help="Spinful collinear/noSOC hr.dat")
    parser.add_argument("--target-hr", type=str, default=None, help="Full SOC target hr.dat")
    parser.add_argument("--kpt", type=str, default=None)
    parser.add_argument(
        "--kmesh",
        nargs=3,
        type=int,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help="Use shifted uniform k-mesh for fitting instead of --kpt",
    )
    parser.add_argument(
        "--mesh-shift",
        nargs=3,
        type=float,
        default=(0.173205, 0.318310, 0.414214),
        metavar=("SX", "SY", "SZ"),
        help="Grid-cell shift for --kmesh; k=(i+s)/N",
    )
    parser.add_argument("--win", type=str, default=None, help="Wannier .win used for metric shell grouping")
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=None,
        help="Orbital blocks start:stop:l, interpreted by --block-layout",
    )
    parser.add_argument(
        "--centres",
        type=str,
        default=None,
        help="Optional centres.xyz for inferred block sanity check",
    )
    parser.add_argument(
        "--block-layout",
        choices=["spinor-block", "separated-spin"],
        default="spinor-block",
        help=(
            "spinor-block: each atomic block already contains spinor pairs; "
            "separated-spin: blocks refer to spinless orbitals in [all up, all down] order"
        ),
    )
    parser.add_argument(
        "--target-l",
        nargs="+",
        type=int,
        default=None,
        help="Angular momentum channels to fit, e.g. --target-l 1 or --target-l 1 2",
    )
    parser.add_argument(
        "--include-dp-hybrid",
        action="store_true",
        help="Add R-dependent d-p hybrid operator dressed by p L.S",
    )
    parser.add_argument(
        "--dp-hybrid-include-onsite",
        action="store_true",
        help="Also include the R=(0,0,0) block in the d-p hybrid operator",
    )
    parser.add_argument(
        "--include-dp-sk",
        action="store_true",
        help="Add spglib-orbit Slater-Koster d-p sigma/pi operators dressed by p L.S",
    )
    parser.add_argument(
        "--dp-sk-grouping",
        choices=["orbit", "shell"],
        default="orbit",
        help="Parameter grouping for d-p SK terms: spglib orbit or distance shell",
    )
    parser.add_argument(
        "--dp-sk-spin-dependent",
        action="store_true",
        help="Also add direct spin-dependent SK channels for each d-p orbit",
    )
    parser.add_argument(
        "--dp-sk-spin-frame",
        choices=["global", "c3", "bond"],
        default="global",
        help=(
            "Spin operator frame for direct SK channels. global uses fixed s0/sx/sy/sz; "
            "c3/bond also allow radial/transverse/normal directions from bond geometry."
        ),
    )
    parser.add_argument(
        "--dp-sk-spin-channels",
        nargs="+",
        default=None,
        metavar="CHANNEL",
        help=(
            "Spin channels for --dp-sk-spin-dependent. Defaults: sx sy for global; "
            "transverse normal for c3/bond. Allowed: s0 sx sy sz radial transverse normal"
        ),
    )
    parser.add_argument(
        "--dp-sk-symmetry",
        choices=["none", "crystal", "magnetic"],
        default="none",
        help=(
            "Project d-p SK templates onto no additional symmetry, the crystal space group, "
            "or the magnetic subgroup selected by --spin-texture"
        ),
    )
    parser.add_argument(
        "--spin-texture",
        type=str,
        default=None,
        help=(
            "Spin texture file for --dp-sk-symmetry magnetic; rows are 'atom_label mx my mz'. "
            "Omitted atoms are treated as nonmagnetic zero-spin sites."
        ),
    )
    parser.add_argument(
        "--dp-sk-include-time-reversal",
        action="store_true",
        help="Allow antiunitary space-group times time-reversal operations in magnetic symmetry filtering",
    )
    parser.add_argument(
        "--spin-symprec",
        type=float,
        default=1e-5,
        help="Tolerance for matching rotated spin vectors in --dp-sk-symmetry magnetic",
    )
    parser.add_argument(
        "--symmetry-projector-tol",
        type=float,
        default=1e-8,
        help="Tolerance for dropping zero or linearly dependent symmetry-projected d-p SK templates",
    )
    parser.add_argument(
        "--symmetry-missing-r",
        choices=["skip", "error"],
        default="skip",
        help=(
            "How to handle symmetry-projected d-p SK contributions whose transformed R-vector "
            "is absent from the finite hr.dat support"
        ),
    )
    parser.add_argument("--symprec", type=float, default=1e-5, help="spglib symmetry tolerance")
    parser.add_argument("--bands", type=str, required=True, help="1-based band selection, e.g. 7:12")
    parser.add_argument("--num-shells", type=int, default=1, help="Number of nonzero |R|^2 shells")
    parser.add_argument(
        "--shell-mode",
        choices=["metric", "index"],
        default="metric",
        help="Group shells using .win lattice metric or integer R-index norm",
    )
    parser.add_argument("--shell-tol", type=float, default=1e-6, help="Distance tolerance in Angstrom")
    parser.add_argument("--no-onsite", action="store_true", help="Exclude R=(0,0,0) shell")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument(
        "--onsite-positive",
        action="store_true",
        help="Constrain onsite L.S lambda terms to be nonnegative",
    )
    parser.add_argument(
        "--term-bound",
        action="append",
        default=None,
        metavar="NAME:LOWER:UPPER",
        help=(
            "Constrain matching term/operator lambda values, e.g. "
            "--term-bound p_ls:0:1 --term-bound d_ls:0:0.2"
        ),
    )
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument(
        "--k-workers",
        type=int,
        default=1,
        help="Number of thread workers for k-chunk residual diagonalization",
    )
    parser.add_argument(
        "--k-batch-size",
        type=int,
        default=0,
        help="K-points per residual chunk for --k-workers; 0 uses an automatic chunk size",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print compact optimizer progress every N residual evaluations; 0 disables",
    )
    parser.add_argument("--remove-center", action="store_true", help="Fit splitting only after removing per-k mean residual")
    parser.add_argument("--align", choices=["none", "bottom", "vbm"], default="none")
    parser.add_argument("--align-bands", type=str, default="1")
    parser.add_argument("--align-reducer", choices=["mean", "min", "max"], default="mean")
    parser.add_argument("--num-occupied", type=int, default=None)
    parser.add_argument(
        "--moment-constraint",
        choices=["none", "collinear", "target"],
        default="none",
        help="Add local spin-moment residuals referenced to collinear or target HR",
    )
    parser.add_argument(
        "--moment-subspace",
        default="",
        help="Local moment selector such as 'Cr:d,I:p'. Empty selects all d blocks.",
    )
    parser.add_argument(
        "--moment-components",
        default="z",
        help="Moment components to constrain, e.g. z or xyz",
    )
    parser.add_argument(
        "--moment-weight",
        type=float,
        default=1.0,
        help="Weight multiplying local moment residuals in the least-squares vector",
    )
    parser.add_argument(
        "--moment-num-occupied",
        type=int,
        default=None,
        help="Occupied band count for local moment constraint; defaults to --num-occupied",
    )
    parser.add_argument("--out-prefix", type=str, default="nonlocal_soc_fit")
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    workdir = resolve_workdir(args.workdir)
    root = resolve_path(workdir, args.root)
    target_dir_arg = args.target_dir if args.target_dir else args.wsoc_dir
    coll_dir_arg = args.coll_dir if args.coll_dir else args.wosoc_dir
    target_dir = resolve_role_dir(root, target_dir_arg)
    coll_dir = resolve_role_dir(root, coll_dir_arg)

    if args.seed:
        seed = args.seed
    elif args.target_hr:
        seed = infer_seed_from_paths([resolve_path(workdir, args.target_hr)])
    elif target_dir:
        seed = infer_seed_from_paths(glob.glob(os.path.join(target_dir, "*_hr.dat")))
    elif args.coll_hr:
        seed = infer_seed_from_paths([resolve_path(workdir, args.coll_hr)])
    elif coll_dir:
        seed = infer_seed_from_paths(glob.glob(os.path.join(coll_dir, "*_hr.dat")))
    else:
        seed = infer_seed_from_paths(glob.glob(os.path.join(root, "**", "*_hr.dat"), recursive=True))

    if args.coll_hr:
        coll_path = resolve_path(workdir, args.coll_hr)
    elif coll_dir:
        coll_path = os.path.join(coll_dir, f"{seed}_hr.dat")
    else:
        coll_path = choose_unique_candidate(
            list_seeded_hr_candidates(root, seed),
            "coll",
            exclude=[resolve_path(workdir, args.target_hr)] if args.target_hr else [],
        )

    if args.target_hr:
        target_path = resolve_path(workdir, args.target_hr)
    elif target_dir:
        target_path = os.path.join(target_dir, f"{seed}_hr.dat")
    else:
        target_path = choose_unique_candidate(
            list_seeded_hr_candidates(root, seed),
            "target",
            exclude=[coll_path],
        )

    if not os.path.exists(coll_path):
        raise ValueError(f"Collinear/noSOC hr.dat not found: {coll_path}")
    if not os.path.exists(target_path):
        raise ValueError(f"Target SOC hr.dat not found: {target_path}")

    if args.kpt:
        kpt_path = resolve_path(workdir, args.kpt)
    else:
        kpt_path = find_seeded_sidecar(target_path, seed, "_band.kpt")
        if kpt_path is None and args.kmesh is None:
            raise ValueError("Could not find kpt file near target hr.dat; pass --kpt or --kmesh")

    if args.win:
        win_path = resolve_path(workdir, args.win)
    else:
        win_path = find_seeded_sidecar(target_path, seed, ".win")
        if win_path is None:
            win_path = find_seeded_sidecar(coll_path, seed, ".win")
        if win_path is None and args.shell_mode == "metric":
            raise ValueError("Could not find .win near input hr.dat files; pass --win or use --shell-mode index")

    centres_path = (
        resolve_path(workdir, args.centres)
        if args.centres
        else (
            find_hr_seed_sidecar(coll_path, "_centres.xyz")
            or find_seeded_sidecar(coll_path, seed, "_centres.xyz")
            or find_hr_seed_sidecar(target_path, "_centres.xyz")
            or find_seeded_sidecar(target_path, seed, "_centres.xyz")
        )
    )

    if args.kmesh is not None:
        kpts = make_shifted_kmesh(args.kmesh, args.mesh_shift)
        k_source = (
            f"shifted mesh {tuple(args.kmesh)} "
            f"with shift {tuple(float(x) for x in args.mesh_shift)}"
        )
    else:
        kpts, _weights = read_kpoints(kpt_path)
        k_source = kpt_path
    coll_hr = load_hr(coll_path)
    target_hr = load_hr(target_path)
    if coll_hr.dim != target_hr.dim:
        raise ValueError(f"Dimension mismatch: coll={coll_hr.dim}, target={target_hr.dim}")

    blocks = parse_blocks(args.blocks)
    if not blocks:
        blocks = infer_blocks_from_win(
            win_path,
            coll_hr.dim,
            args.block_layout,
            centres_path=centres_path,
        )
    target_ls = args.target_l
    if target_ls is None:
        target_ls = sorted({int(block["l"]) for block in blocks})
    terms = []
    used_blocks_by_operator = []
    for l_val in target_ls:
        op_matrix, used_blocks = build_ls_operator(
            coll_hr.dim,
            blocks,
            target_l=int(l_val),
            block_layout=args.block_layout,
        )
        ls_term = build_static_onsite_term(
            coll_hr,
            operator_name(l_val),
            int(l_val),
            op_matrix,
            used_blocks,
        )
        terms.append(ls_term)
        used_blocks_by_operator.append({"operator": operator_name(l_val), "l": int(l_val), "blocks": used_blocks})
    if args.include_dp_hybrid:
        hybrid_op = build_dp_hybrid_operator(
            coll_hr,
            blocks,
            args.block_layout,
            include_onsite=args.dp_hybrid_include_onsite,
        )
        hybrid_op["operator"] = hybrid_op["name"]
        hybrid_op["metadata"] = {
            "orbit": "all_r",
            "num_bonds": len(coll_hr.r_keys),
            "distance_mean_ang": np.nan,
            "distance_std_ang": np.nan,
            "pair_keys": "d-p",
        }
        terms.append(hybrid_op)
        used_blocks_by_operator.append(
            {
                "operator": hybrid_op["name"],
                "l": hybrid_op["l"],
                "blocks": hybrid_op["blocks"],
            }
        )
    sk_orbit_tol = None
    if args.include_dp_sk:
        sk_orbit_tol = max(float(args.shell_tol), 1.0e-2)
        spin_texture = None
        if args.dp_sk_symmetry == "magnetic":
            spin_texture_path = resolve_path(workdir, args.spin_texture) if args.spin_texture else None
            if spin_texture_path is None:
                raise ValueError("--dp-sk-symmetry magnetic requires --spin-texture")
            spin_texture = read_spin_texture(spin_texture_path)
        sk_terms = build_dp_sk_orbit_terms(
            coll_hr,
            blocks,
            args.block_layout,
            centres_path,
            win_path,
            symprec=args.symprec,
            tol=sk_orbit_tol,
            num_shells=args.num_shells,
            spin_dependent=args.dp_sk_spin_dependent,
            spin_channels=args.dp_sk_spin_channels,
            grouping=args.dp_sk_grouping,
            spin_frame=args.dp_sk_spin_frame,
            symmetry_mode=args.dp_sk_symmetry,
            spin_texture=spin_texture,
            include_time_reversal=args.dp_sk_include_time_reversal,
            spin_symprec=args.spin_symprec,
            projector_tol=args.symmetry_projector_tol,
            missing_r_policy=args.symmetry_missing_r,
        )
        for sk_op in sk_terms:
            terms.append(sk_op)
            used_blocks_by_operator.append(
                {
                    "operator": sk_op["name"],
                    "l": sk_op["l"],
                    "blocks": sk_op["blocks"],
                }
            )
    bands = parse_band_selection(args.bands, coll_hr.dim, name="fit bands")
    align_bands = parse_band_selection(args.align_bands, coll_hr.dim, name="align bands")

    hk_coll = hamiltonian_k(coll_hr, kpts)
    hk_target = hamiltonian_k(target_hr, kpts)
    evals_coll = np.linalg.eigvalsh(hk_coll)
    evals_target = np.linalg.eigvalsh(hk_target)

    shift, alignment = compute_alignment_shift(
        evals_target,
        evals_coll,
        args.align,
        bottom_bands=align_bands,
        bottom_reducer=args.align_reducer,
        num_occupied=args.num_occupied,
    )
    evals_target_aligned = evals_target - shift

    moment_active = args.moment_constraint != "none"
    moment_blocks = []
    moment_components = []
    moment_reference = None
    moment_nocc = None
    if moment_active:
        moment_nocc = args.moment_num_occupied if args.moment_num_occupied is not None else args.num_occupied
        if moment_nocc is None:
            raise ValueError("--moment-constraint requires --moment-num-occupied or --num-occupied")
        moment_nocc = int(moment_nocc)
        if moment_nocc <= 0 or moment_nocc > coll_hr.dim:
            raise ValueError(f"moment occupied count must be in [1, {coll_hr.dim}], got {moment_nocc}")
        moment_blocks = moment_blocks_from_selector(blocks, args.moment_subspace)
        moment_components = parse_moment_components(args.moment_components)
        ref_hk = hk_coll if args.moment_constraint == "collinear" else hk_target
        moment_reference = local_spin_moments(
            ref_hk,
            moment_blocks,
            moment_components,
            moment_nocc,
            coll_hr.dim,
            args.block_layout,
        )

    phases = k_phases(kpts, coll_hr.r_vecs)
    terms = prepare_terms_hk(terms, phases)
    x0 = ridge_initial_guess(
        hk_coll,
        terms,
        evals_target_aligned,
        bands,
        ridge=args.ridge,
    )
    lower_bounds, upper_bounds = build_lambda_bounds(
        terms,
        onsite_positive=args.onsite_positive,
        term_bounds=args.term_bound,
    )
    x0 = clip_initial_to_bounds(x0, lower_bounds, upper_bounds)

    progress_every = max(0, int(args.progress_every))
    progress_state = {"calls": 0, "t0": time.time()}
    progress_block = len(x0) + 1
    k_workers = max(1, int(args.k_workers))
    if int(args.k_batch_size) > 0:
        k_batch_size = int(args.k_batch_size)
    elif k_workers > 1:
        k_batch_size = max(1, int(np.ceil(len(kpts) / float(4 * k_workers))))
    else:
        k_batch_size = len(kpts)

    def residual_fn(values):
        moment_resid = np.empty(0, dtype=float)
        if k_workers > 1 or moment_active:
            data_resid, moment_current = selected_residual_kparallel(
                values,
                hk_coll,
                terms,
                evals_target_aligned,
                bands,
                remove_center=args.remove_center,
                workers=k_workers,
                batch_size=k_batch_size,
                moment_blocks=moment_blocks if moment_active else None,
                moment_components=moment_components if moment_active else None,
                moment_nocc=moment_nocc,
                dim=coll_hr.dim,
                block_layout=args.block_layout,
            )
            if moment_active:
                moment_resid = float(args.moment_weight) * np.ravel(moment_current - moment_reference)
        else:
            data_resid = selected_residual(
                values,
                hk_coll,
                terms,
                evals_target_aligned,
                bands,
                remove_center=args.remove_center,
            )
        reg = np.sqrt(float(args.ridge)) * np.asarray(values, dtype=float)
        progress_state["calls"] += 1
        ncall = int(progress_state["calls"])
        nfev = 1 + (ncall - 1) // progress_block
        is_base_eval = (ncall - 1) % progress_block == 0
        max_nfev = int(args.max_nfev) if args.max_nfev is not None else 0
        should_print = (
            progress_every > 0
            and is_base_eval
            and (nfev == 1 or nfev % progress_every == 0 or (max_nfev > 0 and nfev == max_nfev))
        )
        if should_print:
            total_resid = np.concatenate([data_resid, moment_resid, reg])
            cost = 0.5 * float(np.dot(total_resid, total_resid))
            rms = float(np.sqrt(np.mean(data_resid * data_resid)))
            elapsed = time.time() - progress_state["t0"]
            total = f"/{max_nfev}" if max_nfev > 0 else ""
            print(
                f"     iter {nfev:5d}{total:<6s} cost={cost:.6e} "
                f"rms={rms:.6e} elapsed={elapsed:.1f}s",
                flush=True,
            )
        return np.concatenate([data_resid, moment_resid, reg])

    result = least_squares(
        residual_fn,
        x0,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=args.max_nfev,
    )
    lambdas = result.x
    final_residual = selected_residual(
        lambdas,
        hk_coll,
        terms,
        evals_target_aligned,
        bands,
        remove_center=args.remove_center,
    )
    hk_fit = model_hk(hk_coll, terms, lambdas)
    evals_fit = np.linalg.eigvalsh(hk_fit)
    moment_fit = None
    if moment_active:
        moment_fit = local_spin_moments(
            hk_fit,
            moment_blocks,
            moment_components,
            moment_nocc,
            coll_hr.dim,
            args.block_layout,
        )

    out_prefix = resolve_path(workdir, args.out_prefix)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    lambda_csv = f"{out_prefix}_lambdas.csv"
    summary_csv = f"{out_prefix}_fit_summary.csv"
    npz_path = f"{out_prefix}.npz"
    hr_path = f"{out_prefix}_hr.dat"
    for input_path, role in ((coll_path, "collinear"), (target_path, "target")):
        if os.path.abspath(hr_path) == os.path.abspath(input_path):
            raise ValueError(
                f"Refusing to overwrite {role} input hr.dat: {hr_path}. "
                "Use a different --out-prefix."
            )

    write_lambda_csv(lambda_csv, terms, lambdas, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    write_fit_summary_csv(summary_csv, kpts, final_residual, evals_fit[:, bands], evals_target_aligned[:, bands], bands)
    write_fitted_hr(hr_path, coll_hr, terms, lambdas)
    np.savez(
        npz_path,
        kpts=kpts,
        bands_1based=bands + 1,
        lambdas=lambdas,
        lambda_lower_bounds=lower_bounds,
        lambda_upper_bounds=upper_bounds,
        term_names=np.asarray([term["name"] for term in terms], dtype=object),
        operator_names=np.asarray([term.get("operator", term["name"]) for term in terms], dtype=object),
        operator_l=np.asarray([term["l"] for term in terms], dtype=object),
        term_metadata=np.asarray([term.get("metadata", {}) for term in terms], dtype=object),
        kmesh=np.asarray(args.kmesh if args.kmesh is not None else [], dtype=int),
        mesh_shift=np.asarray(args.mesh_shift, dtype=float),
        evals_coll=evals_coll,
        evals_target=evals_target,
        evals_target_aligned=evals_target_aligned,
        evals_fit=evals_fit,
        residual=final_residual.reshape(len(kpts), len(bands)),
        energy_shift_target_minus_coll_eV=float(shift),
        alignment=np.asarray(alignment, dtype=object),
        used_blocks=np.asarray(used_blocks_by_operator, dtype=object),
        block_layout=args.block_layout,
        success=bool(result.success),
        cost=float(result.cost),
        message=str(result.message),
        moment_constraint=args.moment_constraint,
        moment_subspace=args.moment_subspace,
        moment_components=np.asarray(moment_components, dtype=object),
        moment_blocks=np.asarray(moment_blocks, dtype=object),
        moment_num_occupied=-1 if moment_nocc is None else int(moment_nocc),
        moment_reference=np.asarray([] if moment_reference is None else moment_reference),
        moment_fit=np.asarray([] if moment_fit is None else moment_fit),
        moment_weight=float(args.moment_weight),
    )

    rms = float(np.sqrt(np.mean(final_residual * final_residual)))
    max_abs = float(np.max(np.abs(final_residual)))
    print(f"Using seed: {seed}")
    print(f"Loaded k-points: {len(kpts)} from {k_source}")
    print(f"Loaded collinear: dim={coll_hr.dim}, R={len(coll_hr.r_keys)} from {coll_path}")
    print(f"Loaded target SOC: dim={target_hr.dim}, R={len(target_hr.r_keys)} from {target_path}")
    if args.shell_mode == "metric":
        print(f"Loaded lattice: {win_path}")
    if args.blocks:
        print("Orbital blocks: explicit CLI")
    else:
        print(f"Orbital blocks: inferred from {win_path}")
        if centres_path:
            print(f"Centres sanity check: {centres_path}")
    if args.include_dp_sk:
        sk_spin_channels_log = (
            normalize_spin_channels(args.dp_sk_spin_channels, args.dp_sk_spin_frame)
            if args.dp_sk_spin_dependent
            else []
        )
        print(
            f"SK grouping: spglib symprec={args.symprec:g}, "
            f"bond_tol={sk_orbit_tol:g} A, spin_dependent={bool(args.dp_sk_spin_dependent)}, "
            f"spin_frame={args.dp_sk_spin_frame}, spin_channels={sk_spin_channels_log}, "
            f"grouping={args.dp_sk_grouping}, symmetry={args.dp_sk_symmetry}, "
            f"include_time_reversal={bool(args.dp_sk_include_time_reversal)}"
        )
    print(f"Block layout: {args.block_layout}")
    preview_terms = [term["name"] for term in terms[:12]]
    suffix = " ..." if len(terms) > len(preview_terms) else ""
    print(f"Selected terms ({len(terms)}): {preview_terms}{suffix}")
    bounded = np.flatnonzero(np.isfinite(lower_bounds) | np.isfinite(upper_bounds))
    if len(bounded):
        bound_preview = [
            f"{terms[idx]['name']}=[{lower_bounds[idx]:.6g},{upper_bounds[idx]:.6g}]"
            for idx in bounded[:8]
        ]
        bound_suffix = " ..." if len(bounded) > len(bound_preview) else ""
        print(f"Lambda bounds ({len(bounded)}): {bound_preview}{bound_suffix}")
    print(f"Selected blocks/operators: {len(used_blocks_by_operator)} entries")
    print(f"Selected bands: {bands + 1}")
    print(f"K residual workers: {k_workers}, batch_size={k_batch_size}")
    print(f"Alignment mode: {args.align}, shift={shift:.12g} eV")
    if moment_active:
        labels = [f"{block.get('atom_label', '')}:l{block['l']}" for block in moment_blocks]
        delta = moment_fit - moment_reference
        print(
            f"Moment constraint: ref={args.moment_constraint}, blocks={labels}, "
            f"components={moment_components}, nocc={moment_nocc}, weight={float(args.moment_weight):.6g}"
        )
        print(
            f"Moment residual max_abs={float(np.max(np.abs(delta))):.6g}, "
            f"rms={float(np.sqrt(np.mean(delta * delta))):.6g}"
        )
    print(f"Fit success: {result.success}, cost={result.cost:.12g}")
    print(f"Fit residual RMS={rms:.12g} eV, max_abs={max_abs:.12g} eV")
    print(f"Saved lambdas: {lambda_csv}")
    print(f"Saved fit summary: {summary_csv}")
    print(f"Saved fitted hr: {hr_path}")
    print(f"Saved npz: {npz_path}")


if __name__ == "__main__":
    main()
