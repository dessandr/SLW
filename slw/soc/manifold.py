"""Resolve Wannier90 projection manifolds without material-specific indices."""

from __future__ import annotations

import re
from pathlib import Path


def clean_species(label: str) -> str:
    """Return a case-normalized species key from a site label."""

    return re.sub(r"\d+$", "", str(label).strip()).casefold()


def _projection_orbitals(text: str) -> list[str]:
    clean = re.sub(
        r"\b(l|ang|angular)_?mom(entum)?\s*=\s*0\b",
        "s",
        str(text),
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(l|ang|angular)_?mom(entum)?\s*=\s*1\b",
        "p",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(l|ang|angular)_?mom(entum)?\s*=\s*2\b",
        "d",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(l|ang|angular)_?mom(entum)?\s*=\s*3\b",
        "f",
        clean,
        flags=re.IGNORECASE,
    )
    return [
        field.strip("{}()")
        for field in re.split(r"[,; \t]+", clean.casefold())
        if field.strip("{}()") in {"s", "p", "d", "f"}
    ]


def _orbital_count(kind: str) -> int:
    try:
        return {"s": 1, "p": 3, "d": 5, "f": 7}[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported projection orbital kind {kind!r}") from exc


def read_projection_groups(
    win_path: str | Path,
) -> tuple[list[str], list[dict[str, object]], int]:
    """Read ordered atom/orbital groups from a Wannier90 projections block."""

    path = Path(win_path)
    atoms: list[str] = []
    projections: list[tuple[str, list[str]]] = []
    block: str | None = None
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.split("!", 1)[0].split("#", 1)[0].strip()
            lower = line.casefold()
            if not line:
                continue
            if lower.startswith(("begin atoms_frac", "begin atoms_cart")):
                block = "atoms"
                continue
            if lower.startswith("begin projections"):
                block = "projections"
                continue
            if lower.startswith(
                ("end atoms_frac", "end atoms_cart", "end projections")
            ):
                block = None
                continue
            if block == "atoms":
                fields = line.split()
                if len(fields) >= 4:
                    atoms.append(fields[0])
            elif block == "projections":
                left, right = line.split(":", 1) if ":" in line else (line, line)
                orbitals = _projection_orbitals(right)
                if orbitals:
                    projections.append((left.strip(), orbitals))
    if not atoms:
        raise ValueError(f"no atoms_frac/atoms_cart block found in {path}")
    if not projections:
        raise ValueError(f"no projections block found in {path}")

    groups: list[dict[str, object]] = []
    offset = 0
    claimed: set[tuple[int, str]] = set()
    for label, orbitals in projections:
        exact = [
            index
            for index, atom in enumerate(atoms)
            if atom.casefold() == label.casefold()
        ]
        species = [
            index
            for index, atom in enumerate(atoms)
            if clean_species(atom) == clean_species(label)
        ]
        matches = exact or species
        if not matches:
            raise ValueError(f"projection label {label!r} matched no atom in {path}")
        for atom_index in matches:
            for orbital in orbitals:
                key = (atom_index, orbital)
                if key in claimed:
                    raise ValueError(
                        f"atom manifold {atoms[atom_index]}-{orbital} is matched "
                        f"by multiple projection entries in {path}"
                    )
                claimed.add(key)
                size = _orbital_count(orbital)
                groups.append(
                    {
                        "atom_index": atom_index,
                        "atom_label": atoms[atom_index],
                        "element": clean_species(atoms[atom_index]),
                        "orbital": orbital,
                        "indices": list(range(offset, offset + size)),
                    }
                )
                offset += size
    return atoms, groups, offset


# Private aliases retained only while numerical kernels finish their typed API
# conversion. They resolve to the same strict implementation.
_clean_species = clean_species
_win_projection_groups = read_projection_groups


__all__ = ["clean_species", "read_projection_groups"]
