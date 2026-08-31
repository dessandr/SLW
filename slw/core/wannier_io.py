"""I/O helpers for Wannier90 Hamiltonian and rotation files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

RVector = tuple[int, int, int]


@dataclass(frozen=True)
class WannierUMatrixData:
    """Wannier90 rotation matrices and their reduced k points.

    ``matrices`` uses the physical ``(nk, nband, nwann)`` layout.  This is
    also the layout of ``U_dis``; a square ``U`` file simply has
    ``nband == nwann``.
    """

    kpoints: np.ndarray
    matrices: np.ndarray


@dataclass(frozen=True)
class WannierEigenvalueData:
    """Wannier90 eigenvalues with layout ``(nk, nband)`` in electron-volts."""

    eigenvalues: np.ndarray


def read_wannier_hr(
    path: str | Path,
) -> tuple[int, list[int], dict[RVector, np.ndarray]]:
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
            r_vector: RVector = (
                int(fields[0]),
                int(fields[1]),
                int(fields[2]),
            )
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
                " ".join(
                    f"{int(value):5d}" for value in degeneracies[offset : offset + 15]
                )
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


def _next_nonempty_line(stream: TextIO, *, context: str) -> str:
    line = stream.readline()
    while line and not line.strip():
        line = stream.readline()
    if not line:
        raise ValueError(f"Unexpected EOF while reading {context}")
    return str(line)


def read_wannier_u_matrix_data(
    path: str | Path,
    expected_kpoints: int | None = None,
) -> WannierUMatrixData:
    """Read a Wannier90 ``*_u.mat`` or ``*_u_dis.mat`` file.

    Wannier90 writes a blank separator, a three-component k-point header,
    and then matrices with the band index varying fastest.  Keeping the
    k-point header is essential when the matrix is combined with EPR data.
    """

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

        kpoints = np.empty((nk_file, 3), dtype=np.float64)
        rotations = np.empty((nk_file, n_bands, n_wannier), dtype=np.complex128)
        for ik in range(nk_file):
            kpoint_line = _next_nonempty_line(
                stream,
                context=f"k point {ik} in {input_path}",
            )
            kpoint_fields = kpoint_line.split()
            if len(kpoint_fields) != 3:
                raise ValueError(
                    f"Malformed k-point header at k={ik} in {input_path}: "
                    f"{kpoint_line.rstrip()}"
                )
            kpoints[ik] = [float(value) for value in kpoint_fields]
            for iw in range(n_wannier):
                for ib in range(n_bands):
                    fields = _next_nonempty_line(
                        stream,
                        context=(
                            f"rotation value at k={ik}, band={ib}, "
                            f"wannier={iw} in {input_path}"
                        ),
                    ).split()
                    if len(fields) < 2:
                        raise ValueError(
                            f"Malformed rotation value at k={ik}, band={ib}, wannier={iw}"
                        )
                    rotations[ik, ib, iw] = float(fields[0]) + 1j * float(fields[1])
    return WannierUMatrixData(kpoints=kpoints, matrices=rotations)


def read_wannier_u_matrix(
    path: str | Path,
    expected_kpoints: int | None = None,
) -> np.ndarray:
    """Read only the matrices from a Wannier90 rotation file."""

    return read_wannier_u_matrix_data(path, expected_kpoints).matrices


def read_wannier_eig(
    path: str | Path,
    *,
    expected_bands: int | None = None,
    expected_kpoints: int | None = None,
) -> WannierEigenvalueData:
    """Read and validate a Wannier90 ``*.eig`` file.

    Rows must use Wannier90's band-fast ordering ``band, kpoint, energy``.
    The file format stores energies in electron-volts.
    """

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Wannier90 eigenvalue file not found: {input_path}")
    rows = np.loadtxt(input_path, dtype=np.float64, ndmin=2)
    if rows.ndim != 2 or rows.shape[1] < 3 or rows.shape[0] == 0:
        raise ValueError(f"Malformed Wannier90 eigenvalue file: {input_path}")
    integer_indices = rows[:, :2]
    rounded_indices = np.rint(integer_indices).astype(np.int64)
    if not np.array_equal(integer_indices, rounded_indices):
        raise ValueError(f"Non-integer band/k-point index in {input_path}")
    n_bands = int(rounded_indices[:, 0].max(initial=0))
    n_kpoints = int(rounded_indices[:, 1].max(initial=0))
    if expected_bands is not None and n_bands != int(expected_bands):
        raise ValueError(
            f"band mismatch for {input_path}: file={n_bands}, expected={expected_bands}"
        )
    if expected_kpoints is not None and n_kpoints != int(expected_kpoints):
        raise ValueError(
            f"k-point mismatch for {input_path}: file={n_kpoints}, "
            f"expected={expected_kpoints}"
        )
    expected_rows = n_bands * n_kpoints
    if rows.shape[0] != expected_rows:
        raise ValueError(
            f"Wannier90 eigenvalue row count {rows.shape[0]} does not match "
            f"{n_bands} bands * {n_kpoints} k points"
        )
    expected_band = np.tile(np.arange(1, n_bands + 1), n_kpoints)
    expected_kpoint = np.repeat(np.arange(1, n_kpoints + 1), n_bands)
    if not np.array_equal(rounded_indices[:, 0], expected_band) or not np.array_equal(
        rounded_indices[:, 1], expected_kpoint
    ):
        raise ValueError(
            f"Wannier90 eigenvalues in {input_path} are not in band-fast order"
        )
    eigenvalues = np.asarray(rows[:, 2].reshape(n_kpoints, n_bands), dtype=np.float64)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(
            f"Wannier90 eigenvalues contain non-finite values: {input_path}"
        )
    return WannierEigenvalueData(eigenvalues=eigenvalues)


def expand_compact_wannier_u_dis(
    matrices: object,
    eigenvalues: object,
    *,
    outer_window_min: float,
    outer_window_max: float,
    zero_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Expand an old compact ``U_dis`` layout to global band indices.

    Older Wannier90 versions wrote the rows inside the disentanglement outer
    window first and padded the remaining rows with zero.  The returned array
    places those compact rows back at their original band indices.  This is an
    explicit format conversion: callers must select it from file provenance,
    never from an array-shape guess.
    """

    raw = np.asarray(matrices)
    energies = np.asarray(eigenvalues)
    if raw.dtype != np.dtype(np.complex128) or raw.ndim != 3:
        raise TypeError("compact U_dis must be complex128 with shape (nk,nband,nwann)")
    if energies.dtype != np.dtype(np.float64) or energies.shape != raw.shape[:2]:
        raise TypeError(
            "eigenvalues must be float64 with the compact U_dis (nk,nband) shape"
        )
    lower = float(outer_window_min)
    upper = float(outer_window_max)
    threshold = float(zero_tolerance)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError(
            "outer disentanglement window bounds must be finite and ordered"
        )
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("U_dis zero tolerance must be finite and non-negative")

    expanded = np.zeros_like(raw)
    for ik in range(raw.shape[0]):
        window_indices = np.flatnonzero(
            (energies[ik] >= lower) & (energies[ik] <= upper)
        )
        count = int(window_indices.size)
        if count < raw.shape[2]:
            raise ValueError(
                f"outer window at k={ik} contains {count} bands for "
                f"{raw.shape[2]} Wannier functions"
            )
        tail_residual = float(np.max(np.abs(raw[ik, count:]), initial=0.0))
        if tail_residual > threshold:
            raise ValueError(
                f"compact U_dis has nonzero padded rows at k={ik}: "
                f"residual={tail_residual:.3e}, tolerance={threshold:.3e}"
            )
        expanded[ik, window_indices] = raw[ik, :count]
    return np.asarray(expanded, dtype=np.complex128)


def parse_wannier_win_parameters(path: str | Path) -> dict[str, object]:
    """Read basic dimensions and projection entries from a Wannier90 ``.win`` file."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Wannier90 input not found: {input_path}")

    projections: list[str] = []
    parameters: dict[str, object] = {
        "num_bands": None,
        "num_wann": None,
        "projections": projections,
        "dis_win_min": None,
        "dis_win_max": None,
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
                projections.append(line)
                continue
            if "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            normalized_key = key.lower()
            if normalized_key in {"num_bands", "num_wann"}:
                parameters[normalized_key] = int(value.split()[0])
            elif normalized_key in {"dis_win_min", "dis_win_max"}:
                parameters[normalized_key] = float(value.split()[0])
    return parameters
