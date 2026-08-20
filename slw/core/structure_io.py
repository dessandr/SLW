"""Material-independent structure readers used by native SLW stages.

The reader accepts a VASP POSCAR or the explicit ``CELL_PARAMETERS`` and
``ATOMIC_POSITIONS`` cards used by Quantum ESPRESSO.  It intentionally does
not infer ``alat`` because doing so without the complete QE system namelist is
ambiguous.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .constants import BOHR_TO_ANG


def _strip_inline_comment(line: str) -> str:
    return line.split("#", 1)[0].split("!", 1)[0].strip()


def _card_name(line: str) -> str:
    stripped = _strip_inline_comment(line)
    return stripped.split()[0].strip().upper() if stripped else ""


def _card_unit(line: str, default: str) -> str:
    tokens = _strip_inline_comment(line).replace("(", " ").replace(")", " ").split()
    return tokens[1].lower() if len(tokens) >= 2 else default


def _find_card(lines: list[str], names: set[str]) -> int | None:
    normalized = {name.upper() for name in names}
    return next(
        (index for index, line in enumerate(lines) if _card_name(line) in normalized),
        None,
    )


def _read_three_vectors(lines: list[str], start: int, unit: str) -> NDArray[np.float64]:
    if start + 3 > len(lines):
        raise ValueError("CELL_PARAMETERS card does not contain three vectors")
    rows: list[list[float]] = []
    for line in lines[start : start + 3]:
        clean = _strip_inline_comment(line)
        fields = clean.split()
        if len(fields) < 3:
            raise ValueError("CELL_PARAMETERS card has a malformed vector line")
        rows.append([float(value) for value in fields[:3]])
    lattice = np.asarray(rows, dtype=np.float64)
    if unit in {"bohr", "a.u.", "au"}:
        lattice *= BOHR_TO_ANG
    elif unit not in {"angstrom", "ang", "angs", "a"}:
        raise ValueError(
            f"Unsupported CELL_PARAMETERS unit {unit!r}; use angstrom or bohr"
        )
    return lattice


def _numbered_atom_labels(species: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    labels: list[str] = []
    for symbol in species:
        counts[symbol] = counts.get(symbol, 0) + 1
        labels.append(f"{symbol}{counts[symbol]}")
    return labels


def _read_qe_cards(
    source: Path, lines: list[str]
) -> tuple[NDArray[np.float64], list[str], NDArray[np.float64]] | None:
    cell_index = _find_card(lines, {"CELL_PARAMETERS"})
    position_index = _find_card(lines, {"ATOMIC_POSITIONS", "ATOMIC_POSITION"})
    if cell_index is None or position_index is None:
        return None

    lattice = _read_three_vectors(
        lines,
        cell_index + 1,
        _card_unit(lines[cell_index], "angstrom"),
    )
    position_unit = _card_unit(lines[position_index], "crystal")
    species: list[str] = []
    coordinates: list[NDArray[np.float64]] = []
    stop_cards = {
        "CELL_PARAMETERS",
        "ATOMIC_POSITIONS",
        "ATOMIC_POSITION",
        "K_POINTS",
        "OCCUPATIONS",
        "CONSTRAINTS",
        "ATOMIC_SPECIES",
        "ATOMIC_FORCES",
    }
    for raw_line in lines[position_index + 1 :]:
        clean = _strip_inline_comment(raw_line)
        if not clean:
            continue
        if _card_name(clean) in stop_cards:
            break
        fields = clean.split()
        if len(fields) < 4:
            break
        try:
            xyz = np.asarray([float(value) for value in fields[1:4]], dtype=np.float64)
        except ValueError:
            break
        species.append(fields[0])
        coordinates.append(xyz)
    if not species:
        raise ValueError(f"ATOMIC_POSITIONS card in {source} has no atoms")

    positions = np.asarray(coordinates, dtype=np.float64)
    if position_unit in {"crystal", "cryst", "fractional", "frac", "direct"}:
        fractional = positions % 1.0
        cartesian = fractional @ lattice
    elif position_unit in {"angstrom", "ang", "angs", "cartesian", "cart", "a"}:
        cartesian = positions
    elif position_unit in {"bohr", "a.u.", "au"}:
        cartesian = positions * BOHR_TO_ANG
    elif position_unit == "alat":
        raise ValueError(
            "ATOMIC_POSITIONS alat is ambiguous without an explicit lattice "
            "parameter; use crystal, angstrom, or bohr"
        )
    else:
        raise ValueError(
            f"Unsupported ATOMIC_POSITIONS unit {position_unit!r}; "
            "use crystal, angstrom/cartesian, or bohr"
        )
    return lattice, _numbered_atom_labels(species), cartesian


def _read_poscar(
    source: Path, lines: list[str]
) -> tuple[NDArray[np.float64], list[str], NDArray[np.float64]]:
    if len(lines) < 8:
        raise ValueError(f"POSCAR {source} is too short")
    scale_fields = lines[1].split()
    if not scale_fields:
        raise ValueError(f"POSCAR {source} has no scale factor")
    scale = float(scale_fields[0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"POSCAR {source} requires one positive scalar scale factor; got {scale!r}"
        )
    lattice = scale * np.asarray(
        [[float(value) for value in line.split()[:3]] for line in lines[2:5]],
        dtype=np.float64,
    )
    if lattice.shape != (3, 3) or abs(float(np.linalg.det(lattice))) <= 0.0:
        raise ValueError(f"POSCAR {source} has a singular or malformed lattice")

    species = lines[5].split()
    try:
        counts = [int(value) for value in lines[6].split()]
    except ValueError as exc:
        raise ValueError(
            f"POSCAR {source} must include an explicit species-name line"
        ) from exc
    if not species or len(species) != len(counts) or any(count < 0 for count in counts):
        raise ValueError(f"POSCAR {source} has inconsistent species/count rows")
    labels = [
        f"{symbol}{local_index + 1}"
        for symbol, count in zip(species, counts)
        for local_index in range(count)
    ]

    coordinate_line = 7
    if lines[coordinate_line].strip().lower().startswith("s"):
        coordinate_line += 1
    if coordinate_line >= len(lines):
        raise ValueError(f"POSCAR {source} is missing its coordinate mode")
    mode = lines[coordinate_line].strip().lower()
    start = coordinate_line + 1
    stop = start + len(labels)
    if stop > len(lines):
        raise ValueError(
            f"POSCAR {source} declares {len(labels)} atoms but has too few positions"
        )
    coordinates = np.asarray(
        [[float(value) for value in line.split()[:3]] for line in lines[start:stop]],
        dtype=np.float64,
    )
    if coordinates.shape != (len(labels), 3):
        raise ValueError(f"POSCAR {source} has a malformed atomic-position row")
    if mode.startswith("d"):
        cartesian = (coordinates % 1.0) @ lattice
    elif mode.startswith(("c", "k")):
        cartesian = coordinates * scale
    else:
        raise ValueError(f"POSCAR {source} has unsupported coordinate mode {mode!r}")
    return lattice, labels, cartesian


def read_structure(
    path: str | Path,
) -> tuple[NDArray[np.float64], list[str], NDArray[np.float64]]:
    """Read a POSCAR or explicit QE structure in Angstrom Cartesian units."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"structure file not found: {source}")
    source = source.resolve()
    lines = source.read_text(encoding="utf-8").splitlines()
    parsed = _read_qe_cards(source, lines)
    return _read_poscar(source, lines) if parsed is None else parsed


__all__ = ["read_structure"]
