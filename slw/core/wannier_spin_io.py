"""Wannier90 spin-matrix and projection-file readers.

The binary ``*.spn`` layout is a sequence of Fortran unformatted records.
Matrices are expanded from Wannier90's packed upper triangle into Hermitian
``complex128`` arrays as they are iterated, so callers need not retain the
whole file in memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import FortranFile


@dataclass(frozen=True)
class WannierSPNHeader:
    comment: str
    num_bands: int
    num_kpoints: int
    formatted: bool


@dataclass(frozen=True)
class WannierAMNHeader:
    comment: str
    num_bands: int
    num_kpoints: int
    num_projections: int


def _is_binary_fortran(path: Path) -> bool:
    with path.open("rb") as stream:
        prefix = stream.read(16)
    return b"\x00" in prefix


def _decode_fortran_text(record: NDArray[np.bytes_]) -> str:
    return b"".join(record.tolist()).decode("utf-8", "replace").strip()


def read_wannier_spn_header(path: str | Path) -> WannierSPNHeader:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Wannier90 spin matrix not found: {input_path}")
    binary = _is_binary_fortran(input_path)
    if binary:
        with FortranFile(input_path, "r") as stream:
            comment = _decode_fortran_text(stream.read_record(np.dtype("S1")))
            dimensions = stream.read_record(np.int32)
    else:
        with input_path.open("r", encoding="utf-8") as stream:
            comment = stream.readline().strip()
            dimensions = np.asarray(
                [int(value) for value in stream.readline().split()[:2]],
                dtype=np.int32,
            )
    if dimensions.shape != (2,):
        raise ValueError(f"Malformed Wannier90 SPN dimensions in {input_path}")
    num_bands, num_kpoints = (int(value) for value in dimensions)
    if num_bands < 1 or num_kpoints < 1:
        raise ValueError(
            f"Wannier90 SPN dimensions must be positive; got {num_bands}, {num_kpoints}"
        )
    return WannierSPNHeader(comment, num_bands, num_kpoints, not binary)


def _expand_spn_triangle(
    packed: NDArray[np.complex128],
    num_bands: int,
) -> NDArray[np.complex128]:
    expected = 3 * num_bands * (num_bands + 1) // 2
    if packed.size != expected:
        raise ValueError(
            f"Wannier90 SPN record has {packed.size} complex values; expected {expected}"
        )
    triangle = np.asarray(packed, dtype=np.complex128).reshape(
        (3, expected // 3), order="F"
    )
    matrices = np.zeros((3, num_bands, num_bands), dtype=np.complex128)
    packed_index = 0
    for column in range(num_bands):
        for row in range(column + 1):
            value = triangle[:, packed_index]
            matrices[:, row, column] = value
            matrices[:, column, row] = value.conj()
            packed_index += 1
    diagonal = np.arange(num_bands)
    matrices[:, diagonal, diagonal] = matrices[:, diagonal, diagonal].real
    return matrices


def iter_wannier_spn(
    path: str | Path,
) -> Iterator[NDArray[np.complex128]]:
    """Yield ``(3, nband, nband)`` Pauli matrices for each k point."""

    input_path = Path(path)
    header = read_wannier_spn_header(input_path)
    ntriangle = header.num_bands * (header.num_bands + 1) // 2
    if not header.formatted:
        with FortranFile(input_path, "r") as stream:
            stream.read_record(np.dtype("S1"))
            stream.read_record(np.int32)
            for _ in range(header.num_kpoints):
                yield _expand_spn_triangle(
                    stream.read_record(np.complex128), header.num_bands
                )
        return

    with input_path.open("r", encoding="utf-8") as stream:
        stream.readline()
        stream.readline()
        for ik in range(header.num_kpoints):
            packed = np.empty(3 * ntriangle, dtype=np.complex128)
            for pair in range(ntriangle):
                line = stream.readline()
                while line and not line.strip():
                    line = stream.readline()
                if not line:
                    raise ValueError(
                        f"Unexpected EOF in formatted SPN at k={ik}, pair={pair}"
                    )
                fields = line.split()
                if len(fields) < 6:
                    raise ValueError(
                        f"Formatted SPN row requires six real values at k={ik}, "
                        f"pair={pair}"
                    )
                for axis in range(3):
                    packed[3 * pair + axis] = float(fields[2 * axis]) + 1j * float(
                        fields[2 * axis + 1]
                    )
            yield _expand_spn_triangle(packed, header.num_bands)


def read_wannier_amn_header(path: str | Path) -> WannierAMNHeader:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Wannier90 projection matrix not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as stream:
        comment = stream.readline().strip()
        fields = stream.readline().split()
    if len(fields) < 3:
        raise ValueError(f"Malformed Wannier90 AMN dimensions in {input_path}")
    num_bands, num_kpoints, num_projections = (int(value) for value in fields[:3])
    if min(num_bands, num_kpoints, num_projections) < 1:
        raise ValueError(
            "Wannier90 AMN dimensions must all be positive; got "
            f"{num_bands}, {num_kpoints}, {num_projections}"
        )
    return WannierAMNHeader(comment, num_bands, num_kpoints, num_projections)


def iter_wannier_amn_chunks(
    path: str | Path,
    *,
    kpoint_chunk: int = 32,
) -> Iterator[tuple[slice, NDArray[np.complex128]]]:
    """Yield vectorized ``AMN[k,band,projection]`` chunks.

    The text parser uses ``numpy.loadtxt`` over several k points at once,
    avoiding both a Python loop over every matrix element and retention of the
    full projection file.  Integer indices and Wannier90's band-fast ordering
    are validated at the I/O boundary.
    """

    input_path = Path(path)
    header = read_wannier_amn_header(input_path)
    chunk_size = int(kpoint_chunk)
    if chunk_size < 1:
        raise ValueError("AMN k-point chunk must be positive")
    rows_per_kpoint = header.num_bands * header.num_projections
    with input_path.open("r", encoding="utf-8") as stream:
        stream.readline()
        stream.readline()
        for start in range(0, header.num_kpoints, chunk_size):
            stop = min(start + chunk_size, header.num_kpoints)
            count = stop - start
            rows = np.loadtxt(
                stream,
                dtype=np.float64,
                max_rows=count * rows_per_kpoint,
                ndmin=2,
            )
            expected_shape = (count * rows_per_kpoint, 5)
            if rows.shape != expected_shape:
                raise ValueError(
                    f"AMN chunk k={start}:{stop} in {input_path} has shape "
                    f"{rows.shape}; expected {expected_shape}"
                )
            indices = rows[:, :3]
            integer_indices = np.rint(indices).astype(np.int64)
            if not np.array_equal(indices, integer_indices):
                raise ValueError(
                    f"AMN chunk k={start}:{stop} contains non-integer indices"
                )
            expected_band = np.tile(
                np.arange(1, header.num_bands + 1),
                count * header.num_projections,
            )
            expected_projection = np.tile(
                np.repeat(
                    np.arange(1, header.num_projections + 1),
                    header.num_bands,
                ),
                count,
            )
            expected_kpoint = np.repeat(np.arange(start + 1, stop + 1), rows_per_kpoint)
            if (
                not np.array_equal(integer_indices[:, 0], expected_band)
                or not np.array_equal(integer_indices[:, 1], expected_projection)
                or not np.array_equal(integer_indices[:, 2], expected_kpoint)
            ):
                raise ValueError(
                    f"AMN chunk k={start}:{stop} is not in band-fast ordering"
                )
            values = rows[:, 3] + 1j * rows[:, 4]
            matrices = values.reshape(
                count,
                header.num_projections,
                header.num_bands,
            ).transpose(0, 2, 1)
            if not np.all(np.isfinite(matrices)):
                raise ValueError(
                    f"AMN chunk k={start}:{stop} contains non-finite values"
                )
            yield slice(start, stop), np.asarray(matrices, dtype=np.complex128)


__all__ = [
    "WannierAMNHeader",
    "WannierSPNHeader",
    "iter_wannier_amn_chunks",
    "iter_wannier_spn",
    "read_wannier_amn_header",
    "read_wannier_spn_header",
]
