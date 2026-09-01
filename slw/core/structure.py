"""Material-independent structure helpers for Wannier90 workflows."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import spglib
from scipy.spatial import cKDTree

from .constants import BOHR_TO_ANG

_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ELEMENT_TO_Z = {symbol.upper(): index for index, symbol in enumerate(_ELEMENTS, start=1)}
_ELEMENT_CANONICAL = {symbol.casefold(): symbol for symbol in _ELEMENTS}
_SITE_LABEL_RE = re.compile(r"^(?P<symbol>[A-Za-z]+)(?P<site_index>\d*)$")


def chemical_symbol_from_site_label(label: str) -> str:
    """Return the canonical element symbol encoded by a Wannier site label.

    Wannier90 commonly distinguishes equivalent sites with a trailing integer,
    for example ``Mn1`` and ``Mn2``. The suffix identifies the site and is not
    part of the chemical symbol used to construct an spglib cell.
    """

    text = str(label).strip()
    match = _SITE_LABEL_RE.fullmatch(text)
    symbol = None if match is None else _ELEMENT_CANONICAL.get(match["symbol"].casefold())
    if symbol is None:
        raise ValueError(f"Unknown chemical site label: {label!r}")
    return symbol


def _has_explicit_site_index(label: str) -> bool:
    match = _SITE_LABEL_RE.fullmatch(str(label).strip())
    return bool(match is not None and match["site_index"])


def _read_win_blocks(path: Path) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    active: str | None = None
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.split("!", 1)[0].split("#", 1)[0].strip()
            lowered = line.lower()
            if lowered.startswith("begin "):
                active = lowered.split(None, 1)[1]
                blocks[active] = []
                continue
            if lowered.startswith("end "):
                active = None
                continue
            if active is not None and line:
                blocks[active].append(line)
    return blocks


def read_wannier90_structure(
    path: str | Path,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Return lattice vectors in Angstrom and fractional atomic positions from ``.win``."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Wannier90 input not found: {input_path}")
    blocks = _read_win_blocks(input_path)

    lattice_rows = list(blocks.get("unit_cell_cart", []))
    if not lattice_rows:
        raise ValueError(f"{input_path}: missing 'begin unit_cell_cart' block")
    lattice_unit = "ang"
    if lattice_rows[0].lower() in {"ang", "angstrom", "bohr"}:
        lattice_unit = lattice_rows.pop(0).lower()
    if len(lattice_rows) != 3:
        raise ValueError(f"{input_path}: unit_cell_cart must contain exactly three vectors")
    lattice_ang = np.asarray(
        [[float(value) for value in row.split()[:3]] for row in lattice_rows],
        dtype=np.float64,
    )
    if lattice_unit == "bohr":
        lattice_ang *= BOHR_TO_ANG

    species: list[str] = []
    positions_fractional: list[list[float]] = []
    if "atoms_frac" in blocks:
        for row in blocks["atoms_frac"]:
            fields = row.split()
            if len(fields) < 4:
                raise ValueError(f"{input_path}: malformed atoms_frac row: {row}")
            species.append(fields[0])
            positions_fractional.append([float(value) for value in fields[1:4]])
    elif "atoms_cart" in blocks:
        cart_rows = list(blocks["atoms_cart"])
        cart_unit = "ang"
        if cart_rows and cart_rows[0].lower() in {"ang", "angstrom", "bohr"}:
            cart_unit = cart_rows.pop(0).lower()
        positions_cartesian = []
        for row in cart_rows:
            fields = row.split()
            if len(fields) < 4:
                raise ValueError(f"{input_path}: malformed atoms_cart row: {row}")
            species.append(fields[0])
            positions_cartesian.append([float(value) for value in fields[1:4]])
        positions_cartesian_array = np.asarray(positions_cartesian, dtype=np.float64)
        if cart_unit == "bohr":
            positions_cartesian_array *= BOHR_TO_ANG
        positions_fractional = (
            positions_cartesian_array @ np.linalg.inv(lattice_ang)
        ).tolist()
    else:
        raise ValueError(f"{input_path}: missing atoms_frac or atoms_cart block")

    positions = np.mod(np.asarray(positions_fractional, dtype=np.float64), 1.0)
    canonical_species: list[str] = []
    unknown: set[str] = set()
    for label in species:
        try:
            canonical_species.append(chemical_symbol_from_site_label(label))
        except ValueError:
            canonical_species.append("")
            unknown.add(label)
    if unknown:
        raise ValueError(f"Unknown chemical symbols in {input_path}: {sorted(unknown)}")
    numbers = np.asarray(
        [ELEMENT_TO_Z[symbol.upper()] for symbol in canonical_species], dtype=np.int32
    )
    return lattice_ang, species, positions, numbers


def read_wannier90_kpoint_path(path: str | Path):
    """Read labelled line segments from a Wannier90 ``kpoint_path`` block."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Wannier90 input not found: {input_path}")
    rows = _read_win_blocks(input_path).get("kpoint_path", [])
    segments = []
    for row in rows:
        fields = row.split()
        if len(fields) < 8:
            raise ValueError(f"{input_path}: malformed kpoint_path row: {row}")
        segments.append(
            (
                fields[0],
                np.asarray([float(value) for value in fields[1:4]], dtype=np.float64),
                fields[4],
                np.asarray([float(value) for value in fields[5:8]], dtype=np.float64),
            )
        )
    if not segments:
        raise ValueError(f"{input_path}: missing or empty kpoint_path block")
    return segments


def resolve_wannier_structure(
    root: str | Path,
    win_path: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve an explicit ``.win`` file or the sole candidate below ``root``."""
    root_path = Path(root).resolve()
    if win_path is not None:
        candidate = Path(win_path)
        if not candidate.is_absolute():
            candidate = root_path / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Wannier90 input not found: {candidate}")
        return candidate, "explicit"

    candidates = sorted(path.resolve() for path in root_path.glob("*.win") if path.is_file())
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .win file in {root_path}; found {len(candidates)}. "
            "Pass --win explicitly."
        )
    return candidates[0], "inferred"


def atom_names_from_structure(path: str | Path) -> dict[int, str]:
    """Build unique site labels while preserving explicit Wannier labels."""
    _, species, _, _ = read_wannier90_structure(path)
    canonical = [chemical_symbol_from_site_label(label) for label in species]
    label_counts: dict[str, int] = {}
    for label in species:
        key = label.strip().casefold()
        label_counts[key] = label_counts.get(key, 0) + 1

    reserved = {
        label.strip().casefold()
        for label in species
        if _has_explicit_site_index(label)
        and label_counts[label.strip().casefold()] == 1
    }
    used: set[str] = set()
    next_index: dict[str, int] = {}
    labels: dict[int, str] = {}
    for index, (raw_label, symbol) in enumerate(zip(species, canonical)):
        raw_key = raw_label.strip().casefold()
        if _has_explicit_site_index(raw_label) and label_counts[raw_key] == 1:
            site_label = raw_label.strip()
        else:
            candidate_index = next_index.get(symbol, 1)
            while True:
                site_label = f"{symbol}{candidate_index}"
                candidate_index += 1
                key = site_label.casefold()
                if key not in reserved and key not in used:
                    break
            next_index[symbol] = candidate_index
        used.add(site_label.casefold())
        labels[index] = site_label
    return labels


def compute_lattice(
    structure_path: str | Path | None = None,
    *,
    cell_ang: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    atomic_numbers: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return real and reciprocal lattice data in Angstrom-based units."""
    if structure_path is not None:
        cell_ang, _, positions, atomic_numbers = read_wannier90_structure(structure_path)
    if cell_ang is None:
        raise ValueError("Provide structure_path or cell_ang")
    cell_ang = np.asarray(cell_ang, dtype=np.float64)
    if cell_ang.shape != (3, 3):
        raise ValueError(f"cell_ang must have shape (3, 3), got {cell_ang.shape}")
    if positions is None:
        positions = np.empty((0, 3), dtype=np.float64)
    if atomic_numbers is None:
        atomic_numbers = np.ones(len(positions), dtype=np.int32)
    reciprocal_ang = np.linalg.inv(cell_ang).T * (2.0 * np.pi)
    return {
        "R_vec_ang": cell_ang,
        "G_vec_ang": reciprocal_ang,
        "positions": np.asarray(positions, dtype=np.float64),
        "atomic_numbers": np.asarray(atomic_numbers, dtype=np.int32),
    }


def find_nearest_neighbours(
    structure_path: str | Path,
    mag_atom_indices: Iterable[int],
    n_shells: int = 10,
    d_max: float = 20.0,
    all_bonds: bool = False,
):
    """Find magnetic bonds using vectorized supercell construction and a KD tree."""
    lattice = compute_lattice(structure_path)
    cell = lattice["R_vec_ang"]
    positions = lattice["positions"]
    magnetic = np.asarray(sorted({int(index) for index in mag_atom_indices}), dtype=np.int32)
    if magnetic.size == 0 or np.any(magnetic < 0) or np.any(magnetic >= len(positions)):
        raise ValueError(f"Invalid magnetic atom indices: {magnetic.tolist()}")

    extent = int(np.ceil(float(d_max) / np.min(np.linalg.norm(cell, axis=1)))) + 1
    axis = np.arange(-extent, extent + 1, dtype=np.int32)
    shifts = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    translated = positions[None, :, :] + shifts[:, None, :]
    cartesian = translated.reshape(-1, 3) @ cell
    atom_ids = np.tile(np.arange(len(positions), dtype=np.int32), len(shifts))
    shift_ids = np.repeat(shifts, len(positions), axis=0)
    tree = cKDTree(cartesian)

    candidates: list[dict[str, object]] = []
    seen: set[tuple[int, int, tuple[int, int, int]]] = set()
    magnetic_set = set(magnetic.tolist())
    for atom_i in magnetic:
        origin = positions[atom_i] @ cell
        indices = np.asarray(tree.query_ball_point(origin, r=float(d_max)), dtype=np.int64)
        distances = np.linalg.norm(cartesian[indices] - origin[None, :], axis=1)
        order = np.argsort(distances)
        for flat_index, distance in zip(indices[order], distances[order]):
            distance = float(distance)
            if distance < 1.0e-4 or distance > float(d_max):
                continue
            atom_j = int(atom_ids[int(flat_index)])
            if atom_j not in magnetic_set:
                continue
            translation = tuple(int(value) for value in shift_ids[int(flat_index)])
            if all_bonds:
                key = (int(atom_i), atom_j, translation)
            elif atom_i < atom_j:
                key = (int(atom_i), atom_j, translation)
            elif atom_i > atom_j:
                key = (atom_j, int(atom_i), tuple(-value for value in translation))
            elif any(translation):
                first_nonzero = next(value for value in translation if value)
                key = (
                    int(atom_i),
                    int(atom_i),
                    translation if first_nonzero > 0 else tuple(-value for value in translation),
                )
            else:
                continue
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"i": key[0], "j": key[1], "R": key[2], "distance": distance})

    candidates.sort(key=lambda item: float(item["distance"]))
    if not candidates:
        return []
    shells: list[list[dict[str, object]]] = []
    for candidate in candidates:
        if not shells or abs(float(candidate["distance"]) - float(shells[-1][0]["distance"])) >= 1.0e-4:
            if len(shells) >= int(n_shells):
                break
            shells.append([])
        shells[-1].append(candidate)
    return [
        {**bond, "shell_idx": shell_index}
        for shell_index, shell in enumerate(shells, start=1)
        for bond in shell
    ]


def get_symmetry_orbits(
    structure_path: str | Path,
    neighbours,
    *,
    symprec: float = 1.0e-5,
    angle_tolerance: float = -1.0,
):
    """Group bonds into crystallographic symmetry orbits."""
    lattice = compute_lattice(structure_path)
    positions = lattice["positions"]
    symmetry = spglib.get_symmetry(
        (lattice["R_vec_ang"], positions, lattice["atomic_numbers"]),
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    if symmetry is None:
        return [[neighbour] for neighbour in neighbours]

    def mapped_atom(atom_index, rotation, translation):
        transformed = rotation @ positions[atom_index] + translation
        differences = transformed[None, :] - positions
        differences -= np.rint(differences)
        norms = np.linalg.norm(differences @ lattice["R_vec_ang"], axis=1)
        target = int(np.argmin(norms))
        return target if norms[target] < float(symprec) else None

    unused = set(range(len(neighbours)))
    orbits = []
    while unused:
        seed_index = min(unused)
        unused.remove(seed_index)
        seed = neighbours[seed_index]
        orbit = [seed]
        vector = positions[seed["j"]] + np.asarray(seed["R"]) - positions[seed["i"]]
        for rotation, translation in zip(symmetry["rotations"], symmetry["translations"]):
            atom_i = mapped_atom(seed["i"], rotation, translation)
            atom_j = mapped_atom(seed["j"], rotation, translation)
            if atom_i is None or atom_j is None:
                continue
            translated_vector = rotation @ vector
            r_vector = np.rint(
                translated_vector - (positions[atom_j] - positions[atom_i])
            ).astype(np.int32)
            for candidate_index in tuple(unused):
                candidate = neighbours[candidate_index]
                direct = (
                    candidate["i"] == atom_i
                    and candidate["j"] == atom_j
                    and np.array_equal(candidate["R"], r_vector)
                )
                reverse = (
                    candidate["i"] == atom_j
                    and candidate["j"] == atom_i
                    and np.array_equal(candidate["R"], -r_vector)
                )
                if direct or reverse:
                    unused.remove(candidate_index)
                    orbit.append(candidate)
                    break
        orbits.append(orbit)
    return orbits
