"""Native, vectorized phonon-cache construction from qe2pert EPR IFCs."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.core.constants import BOHR_TO_ANG, RY_TO_EV
from slw.core.qe2pert_ws import (
    init_rvec_images,
    set_wigner_seitz_cell,
    triangular_pair_index,
)
from slw.epc.epr_phonon import (
    apply_loto_override,
    dynamical_longrange,
    phase_table,
    polar_onsite_correction,
    read_epr_phonon_metadata,
)

from .phonon import (
    CELL_GAUGE_FOURIER_PHASE_CONVENTION,
    CELL_GAUGE_VECTOR_CONVENTION,
    SUPPORTED_PHONON_CACHE_SCHEMA_VERSION,
)

_RY_TO_MEV = 1000.0 * RY_TO_EV


@dataclass(frozen=True)
class EPRPhononBuildReport:
    source: Path
    output: Path
    q_mesh_shape: tuple[int, int, int]
    q_point_count: int
    mode_count: int
    q_chunk_size: int
    loto_mode: str
    imaginary_tolerance_mev: float
    rounded_frequency_count: int
    minimum_raw_frequency_mev: float
    elapsed_seconds: float


def _positive_mesh(value: object) -> tuple[int, int, int]:
    raw = np.asarray(value)
    if raw.shape != (3,) or np.iscomplexobj(raw):
        raise ValueError("phonon q mesh must contain three positive integers")
    numeric = np.asarray(raw, dtype=np.float64)
    if (
        not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, np.rint(numeric))
        or np.any(numeric <= 0.0)
    ):
        raise ValueError("phonon q mesh must contain three positive integers")
    return int(numeric[0]), int(numeric[1]), int(numeric[2])


def _uniform_q_mesh(mesh: tuple[int, int, int]) -> NDArray[np.float64]:
    grid = np.indices(mesh, dtype=np.float64).reshape(3, -1).T
    return grid / np.asarray(mesh, dtype=np.float64)[None, :]


def _preload_ifc_blocks(
    source: Path, meta: dict[str, object]
) -> list[tuple[int, int, int, int, NDArray[np.float64], NDArray[np.complex128]]]:
    lattice = np.asarray(meta["at"], dtype=np.float64)
    positions = np.asarray(meta["tau"], dtype=np.float64)
    nat = int(cast(int, meta["nat"]))
    images = init_rvec_images(meta["qc_dim"], lattice)
    blocks = []
    with h5py.File(source, "r") as handle:
        for atom_j in range(1, nat + 1):
            for atom_i in range(1, atom_j + 1):
                pair = triangular_pair_index(atom_i, atom_j)
                path = f"force_constant/ifc{pair}"
                if path not in handle or not isinstance(handle[path], h5py.Dataset):
                    raise KeyError(f"missing required EPR IFC dataset {path}")
                ws = set_wigner_seitz_cell(
                    images,
                    lattice,
                    positions[atom_i - 1],
                    positions[atom_j - 1],
                )
                values = np.asarray(handle[path], dtype=np.float64).astype(
                    np.complex128, copy=False
                )
                if values.shape != (ws.nr, 3, 3):
                    raise ValueError(
                        f"EPR IFC {path} shape {values.shape} != {(ws.nr, 3, 3)}"
                    )
                blocks.append(
                    (
                        3 * (atom_i - 1),
                        3 * atom_i,
                        3 * (atom_j - 1),
                        3 * atom_j,
                        np.asarray(ws.vectors, dtype=np.float64),
                        np.ascontiguousarray(values),
                    )
                )
    return blocks


def _assemble_chunk(
    q_points: NDArray[np.float64],
    *,
    meta: dict[str, object],
    blocks: list[
        tuple[int, int, int, int, NDArray[np.float64], NDArray[np.complex128]]
    ],
    mass_factor: NDArray[np.float64],
    polar_onsite: NDArray[np.float64] | None,
) -> NDArray[np.complex128]:
    nat = int(cast(int, meta["nat"]))
    dynamical = np.zeros((q_points.shape[0], 3 * nat, 3 * nat), dtype=np.complex128)
    for row0, row1, col0, col1, shifts, values in blocks:
        transformed = np.einsum(
            "qr,rij->qij",
            phase_table(q_points, shifts),
            values,
            optimize=True,
        )
        dynamical[:, row0:row1, col0:col1] = transformed
        if row0 != col0:
            dynamical[:, col0:col1, row0:row1] = np.swapaxes(
                transformed.conj(), 1, 2
            )
    if polar_onsite is not None:
        for q_index, q_point in enumerate(q_points):
            longrange = dynamical_longrange(meta, q_point, polar_onsite)
            pair = 0
            for atom_j in range(nat):
                for atom_i in range(atom_j + 1):
                    row = slice(3 * atom_i, 3 * (atom_i + 1))
                    col = slice(3 * atom_j, 3 * (atom_j + 1))
                    dynamical[q_index, row, col] += longrange[pair]
                    if atom_i != atom_j:
                        dynamical[q_index, col, row] += longrange[pair].conj().T
                    pair += 1
    dynamical = 0.5 * (dynamical + np.swapaxes(dynamical.conj(), 1, 2))
    dynamical *= mass_factor[None, :, :]
    return dynamical


def build_epr_phonon_cache_payload(
    epr_path: str | Path,
    *,
    q_mesh_shape: tuple[int, int, int] | None = None,
    q_chunk_size: int | None = None,
    loto_mode: str = "auto",
    imaginary_tolerance_mev: float = 1.0e-6,
) -> tuple[dict[str, np.ndarray], EPRPhononBuildReport]:
    """Build one schema-v3 cache payload with chunked batched diagonalization."""

    started = time.perf_counter()
    source = Path(epr_path).expanduser().resolve()
    meta = apply_loto_override(read_epr_phonon_metadata(source), loto_mode)
    mesh = _positive_mesh(meta["qc_dim"] if q_mesh_shape is None else q_mesh_shape)
    q_points = _uniform_q_mesh(mesh)
    nq = q_points.shape[0]
    chunk = nq if q_chunk_size is None else int(q_chunk_size)
    if chunk <= 0:
        raise ValueError("phonon q_chunk_size must be positive")
    chunk = min(chunk, nq)
    tolerance = float(imaginary_tolerance_mev)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("phonon imaginary tolerance must be finite and non-negative")

    nat = int(meta["nat"])
    masses = np.asarray(meta["mass"], dtype=np.float64)
    blocks = _preload_ifc_blocks(source, meta)
    repeated_mass = np.repeat(masses, 3)
    mass_factor = 1.0 / np.sqrt(
        repeated_mass[:, None] * repeated_mass[None, :]
    )
    sqrt_mass = np.sqrt(masses)[:, None]
    polar_onsite = polar_onsite_correction(meta) if bool(meta["lpolar"]) else None

    frequencies = np.empty((nq, 3 * nat), dtype=np.float64)
    vectors = np.empty((nq, 3 * nat, nat, 3), dtype=np.complex128)
    minimum_raw = np.inf
    rounded_count = 0
    for start in range(0, nq, chunk):
        stop = min(start + chunk, nq)
        dynamical = _assemble_chunk(
            q_points[start:stop],
            meta=meta,
            blocks=blocks,
            mass_factor=mass_factor,
            polar_onsite=polar_onsite,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(dynamical)
        signed = np.sqrt(np.abs(eigenvalues.real)) * _RY_TO_MEV
        signed[eigenvalues.real < 0.0] *= -1.0
        minimum_raw = min(minimum_raw, float(np.min(signed)))
        unstable = signed < -tolerance
        if np.any(unstable):
            worst = float(np.min(signed))
            raise ValueError(
                "EPR phonons contain an imaginary mode outside the configured "
                f"roundoff tolerance: minimum={worst:.6g} meV, "
                f"tolerance={tolerance:.6g} meV"
            )
        roundoff = np.abs(signed) <= tolerance
        rounded_count += int(np.count_nonzero(roundoff))
        signed[roundoff] = 0.0
        frequencies[start:stop] = signed
        vectors[start:stop] = (
            eigenvectors.transpose(0, 2, 1).reshape(stop - start, 3 * nat, nat, 3)
            / sqrt_mass[None, None, :, :]
        )

    alat_ang = float(meta["alat"]) * BOHR_TO_ANG
    lattice_ang = np.asarray(meta["at"], dtype=np.float64).T * alat_ang
    reciprocal_ang = 2.0 * np.pi * np.linalg.inv(lattice_ang).T
    source_stat = source.stat()
    payload: dict[str, np.ndarray] = {
        "phonon_cache_schema_version": np.asarray(
            SUPPORTED_PHONON_CACHE_SCHEMA_VERSION, dtype=np.int32
        ),
        "q_mesh_flat_frac": q_points,
        "q_mesh_flat_cart": q_points @ reciprocal_ang,
        "q_mesh_shape": np.asarray(mesh, dtype=np.int32),
        "q_mesh_shift_grid": np.zeros(3, dtype=np.float64),
        "ph_en_flat": frequencies,
        "ph_vec_flat": vectors,
        "atom_mass_electron": masses,
        "atom_frac": np.mod(np.asarray(meta["tau"], dtype=np.float64), 1.0),
        "lattice_ang": lattice_ang,
        "phonon_vector_convention": np.asarray(CELL_GAUGE_VECTOR_CONVENTION),
        "fourier_phase_convention": np.asarray(
            CELL_GAUGE_FOURIER_PHASE_CONVENTION
        ),
        "mass_unit": np.asarray("electron_mass"),
        "energy_unit": np.asarray("meV"),
        "q_cart_unit": np.asarray("1/angstrom"),
        "epr_source": np.asarray(str(source)),
        "epr_source_size": np.asarray(source_stat.st_size, dtype=np.int64),
        "epr_source_mtime_ns": np.asarray(source_stat.st_mtime_ns, dtype=np.int64),
        "phonon_source": np.asarray("native EPR IFC"),
        "phonon_loto": np.asarray(str(loto_mode).strip().lower()),
        "imaginary_tolerance_mev": np.asarray(tolerance, dtype=np.float64),
        "rounded_frequency_count": np.asarray(rounded_count, dtype=np.int64),
        "minimum_raw_frequency_mev": np.asarray(minimum_raw, dtype=np.float64),
    }
    report = EPRPhononBuildReport(
        source=source,
        output=Path("<memory>"),
        q_mesh_shape=mesh,
        q_point_count=nq,
        mode_count=3 * nat,
        q_chunk_size=chunk,
        loto_mode=str(loto_mode).strip().lower(),
        imaginary_tolerance_mev=tolerance,
        rounded_frequency_count=rounded_count,
        minimum_raw_frequency_mev=minimum_raw,
        elapsed_seconds=time.perf_counter() - started,
    )
    return payload, report


def write_epr_phonon_cache(
    epr_path: str | Path,
    output_path: str | Path,
    *,
    q_mesh_shape: tuple[int, int, int] | None = None,
    q_chunk_size: int | None = None,
    loto_mode: str = "auto",
    imaginary_tolerance_mev: float = 1.0e-6,
    compressed: bool = False,
) -> EPRPhononBuildReport:
    """Build and atomically write one native cache without clobbering."""

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite phonon cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload, report = build_epr_phonon_cache_payload(
        epr_path,
        q_mesh_shape=q_mesh_shape,
        q_chunk_size=q_chunk_size,
        loto_mode=loto_mode,
        imaginary_tolerance_mev=imaginary_tolerance_mev,
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = np.savez_compressed if compressed else np.savez
            writer(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return EPRPhononBuildReport(
        source=report.source,
        output=output,
        q_mesh_shape=report.q_mesh_shape,
        q_point_count=report.q_point_count,
        mode_count=report.mode_count,
        q_chunk_size=report.q_chunk_size,
        loto_mode=report.loto_mode,
        imaginary_tolerance_mev=report.imaginary_tolerance_mev,
        rounded_frequency_count=report.rounded_frequency_count,
        minimum_raw_frequency_mev=report.minimum_raw_frequency_mev,
        elapsed_seconds=report.elapsed_seconds,
    )


__all__ = [
    "EPRPhononBuildReport",
    "build_epr_phonon_cache_payload",
    "write_epr_phonon_cache",
]
