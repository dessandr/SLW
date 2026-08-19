"""I/O helpers for Wannier90 Hamiltonian and rotation files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


RVector = tuple[int, int, int]


def read_wannier_hr(path: str | Path) -> tuple[int, list[int], dict[RVector, np.ndarray]]:
    """Read a Wannier90 ``*_hr.dat`` Hamiltonian."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Wannier90 Hamiltonian not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as stream:
        stream.readline()  # comment/header
        dim_line = stream.readline()
        nrpts_line = stream.readline()
        if not dim_line or not nrpts_line:
            raise ValueError(f"Malformed Wannier90 Hamiltonian header: {input_path}")
        dim = int(dim_line.strip())
        nrpts = int(nrpts_line.strip())

        degeneracies: list[int] = []
        while len(degeneracies) < nrpts:
            line = stream.readline()
            if not line:
                raise ValueError(
                    f"Unexpected EOF while reading {nrpts} degeneracies from {input_path}"
                )
            degeneracies.extend(int(value) for value in line.split())
        degeneracies = degeneracies[:nrpts]

        hamiltonian: dict[RVector, np.ndarray] = {}
        for line_number, line in enumerate(stream, start=4):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 7:
                raise ValueError(
                    f"Malformed Hamiltonian row at {input_path}:{line_number}: {line.rstrip()}"
                )
            r_vector = tuple(int(value) for value in fields[:3])
            row = int(fields[3]) - 1
            column = int(fields[4]) - 1
            if not (0 <= row < dim and 0 <= column < dim):
                raise ValueError(
                    f"Orbital index out of range at {input_path}:{line_number}: "
                    f"({row + 1}, {column + 1}) for dimension {dim}"
                )
            block = hamiltonian.setdefault(
                r_vector, np.zeros((dim, dim), dtype=np.complex128)
            )
            block[row, column] = float(fields[5]) + 1j * float(fields[6])

    if len(hamiltonian) != nrpts:
        raise ValueError(
            f"R-point count mismatch in {input_path}: header={nrpts}, parsed={len(hamiltonian)}"
        )
    return dim, degeneracies, hamiltonian


def write_wannier_hr(
    path: str | Path,
    dim: int,
    degeneracies: Sequence[int],
    hamiltonian: Mapping[RVector, np.ndarray],
    header: str = "Generated Wannier90 Hamiltonian",
) -> None:
    """Write a Wannier90 ``*_hr.dat`` Hamiltonian."""
    output_path = Path(path)
    r_vectors = sorted(hamiltonian)
    if len(degeneracies) != len(r_vectors):
        raise ValueError(
            "A degeneracy is required for every R point: "
            f"degeneracies={len(degeneracies)}, R points={len(r_vectors)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(f" {header}\n")
        stream.write(f"{int(dim):12d}\n")
        stream.write(f"{len(r_vectors):12d}\n")
        for offset in range(0, len(degeneracies), 15):
            stream.write(
                " ".join(f"{int(value):5d}" for value in degeneracies[offset : offset + 15])
                + "\n"
            )

        for r_vector in r_vectors:
            block = np.asarray(hamiltonian[r_vector], dtype=np.complex128)
            if block.shape != (dim, dim):
                raise ValueError(
                    f"Block {r_vector} has shape {block.shape}; expected {(dim, dim)}"
                )
            for row in range(dim):
                for column in range(dim):
                    value = block[row, column]
                    stream.write(
                        f"{r_vector[0]:5d}{r_vector[1]:5d}{r_vector[2]:5d}"
                        f"{row + 1:5d}{column + 1:5d}"
                        f"{value.real:22.14e}{value.imag:22.14e}\n"
                    )


def read_wannier_u_matrix(path: str | Path, expected_kpoints: int | None = None) -> np.ndarray:
    """Read a Wannier90 ``*_u.mat`` rotation matrix."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Wannier90 rotation matrix not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as stream:
        stream.readline()
        dimensions = stream.readline().split()
        if len(dimensions) < 3:
            raise ValueError(f"Malformed Wannier90 rotation header: {input_path}")
        nk_file, n_wannier, n_bands = (int(value) for value in dimensions[:3])
        if expected_kpoints is not None and nk_file != int(expected_kpoints):
            raise ValueError(
                f"k-point mismatch for {input_path}: file={nk_file}, expected={expected_kpoints}"
            )

        rotations = np.empty((nk_file, n_bands, n_wannier), dtype=np.complex128)
        for ik in range(nk_file):
            if not stream.readline():
                raise ValueError(f"Unexpected EOF before k point {ik} in {input_path}")
            for iw in range(n_wannier):
                for ib in range(n_bands):
                    fields = stream.readline().split()
                    if len(fields) < 2:
                        raise ValueError(
                            f"Malformed rotation value at k={ik}, band={ib}, wannier={iw}"
                        )
                    rotations[ik, ib, iw] = float(fields[0]) + 1j * float(fields[1])
    return rotations


def parse_wannier_win_parameters(path: str | Path) -> dict[str, object]:
    """Read basic dimensions and projection entries from a Wannier90 ``.win`` file."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Wannier90 input not found: {input_path}")

    parameters: dict[str, object] = {
        "num_bands": None,
        "num_wann": None,
        "projections": [],
    }
    in_projections = False
    with input_path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.split("!", 1)[0].split("#", 1)[0].strip()
            lowered = line.lower()
            if lowered == "begin projections":
                in_projections = True
                continue
            if lowered == "end projections":
                in_projections = False
                continue
            if in_projections and line:
                parameters["projections"].append(line)
                continue
            if "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if key.lower() in {"num_bands", "num_wann"}:
                parameters[key.lower()] = int(value.split()[0])
    return parameters
