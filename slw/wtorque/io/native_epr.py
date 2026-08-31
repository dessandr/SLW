"""Diagnostics for native spinor qe2pert EPR/Wannier inputs.

The production observable may use an arbitrary maximally-localized Wannier
gauge.  This module therefore never requires projected spin matrices to equal
``I_orb tensor sigma``.  Instead it anchors the antiunitary time-reversal
operation to declared real atomic projections and reports the conditioning and
closure of that construction before extracting the time-reversal-odd field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.core.wannier_io import (
    expand_compact_wannier_u_dis,
    parse_wannier_win_parameters,
    read_wannier_eig,
    read_wannier_hr,
    read_wannier_u_matrix_data,
)
from slw.core.wannier_spin_io import (
    iter_wannier_amn_chunks,
    iter_wannier_spn,
    read_wannier_amn_header,
    read_wannier_spn_header,
)

from ..basis import SpinOrder
from ..model.time_reversal import (
    atomic_spin_time_reversal,
    projection_anchored_sewing,
    time_reversed,
)

UDisLayout = Literal["global_bands", "compact_outer_window"]


def _uniform_reduced_grid(mesh: tuple[int, int, int]) -> NDArray[np.float64]:
    n1, n2, n3 = mesh
    return np.asarray(
        [
            (i / n1, j / n2, k / n3)
            for i in range(n1)
            for j in range(n2)
            for k in range(n3)
        ],
        dtype=np.float64,
    )


def _periodic_residual(first: object, second: object) -> float:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    delta -= np.rint(delta)
    return float(np.max(np.linalg.norm(delta, axis=-1), initial=0.0))


def _minus_k_indices(mesh: tuple[int, int, int]) -> NDArray[np.int64]:
    n1, n2, n3 = mesh
    grid = np.arange(n1 * n2 * n3, dtype=np.int64).reshape(mesh)
    result = np.empty(grid.size, dtype=np.int64)
    offset = 0
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                result[offset] = grid[(-i) % n1, (-j) % n2, (-k) % n3]
                offset += 1
    return result


def _relative_matrix_norms(
    difference: NDArray[np.complex128],
    reference: NDArray[np.complex128],
) -> NDArray[np.float64]:
    numerator = np.linalg.norm(difference.reshape(difference.shape[0], -1), axis=1)
    denominator = np.linalg.norm(reference.reshape(reference.shape[0], -1), axis=1)
    return np.asarray(
        numerator / np.maximum(denominator, np.finfo(np.float64).tiny),
        dtype=np.float64,
    )


def _hamiltonian_from_eigenvalues(
    transform: NDArray[np.complex128],
    eigenvalues: NDArray[np.float64],
) -> NDArray[np.complex128]:
    hamiltonian = np.einsum(
        "kmi,km,kmj->kij",
        transform.conj(),
        eigenvalues,
        transform,
        optimize=True,
    )
    return np.asarray(
        0.5 * (hamiltonian + np.swapaxes(hamiltonian.conj(), 1, 2)),
        dtype=np.complex128,
    )


@dataclass(frozen=True)
class NativeSpinorBasis:
    """Full Wannier Hamiltonian plus an explicitly selected AMN subspace."""

    num_bands: int
    num_wann: int
    kmesh: tuple[int, int, int]
    qmesh: tuple[int, int, int]
    kpoints: NDArray[np.float64]
    minus_k_index: NDArray[np.int64]
    transform: NDArray[np.complex128]
    hamiltonian_eV: NDArray[np.complex128]
    selected_overlap: NDArray[np.complex128]
    projection_indices: tuple[int, ...]
    u_isometry_residual: float
    kpoint_residual: float


def parse_projection_indices(
    specification: str,
    projection_count: int,
) -> tuple[int, ...]:
    """Parse a one-based list/range such as ``9-18`` into zero-based indices."""

    count = int(projection_count)
    if count < 1:
        raise ValueError("projection_count must be positive")
    text = str(specification).strip()
    if not text:
        raise ValueError("magnetic projection selection must not be empty")
    selected: list[int] = []
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"invalid empty projection token in {specification!r}")
        if "-" in token:
            fields = token.split("-")
            if len(fields) != 2 or not all(field.strip().isdigit() for field in fields):
                raise ValueError(f"invalid projection range {token!r}")
            first, last = (int(field) for field in fields)
            if last < first:
                raise ValueError(f"projection range must increase: {token!r}")
            selected.extend(range(first, last + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"invalid projection index {token!r}")
            selected.append(int(token))
    if any(index < 1 or index > count for index in selected):
        raise ValueError(f"projection indices must lie in 1..{count}; got {selected}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"projection selection contains duplicates: {selected}")
    return tuple(index - 1 for index in selected)


def load_native_spinor_basis(
    epr_path: str | Path,
    *,
    win_path: str | Path,
    amn_path: str | Path,
    eig_path: str | Path,
    u_matrix_path: str | Path,
    u_dis_matrix_path: str | Path,
    u_dis_layout: UDisLayout,
    projection_indices: object,
    isometry_tolerance: float = 1.0e-7,
    kpoint_tolerance: float = 1.0e-7,
) -> NativeSpinorBasis:
    """Load the full native spinor Hamiltonian and selected atomic overlap.

    Projection indices are zero based at this API boundary.  The CLI uses
    :func:`parse_projection_indices` to make its one-based user convention
    explicit.
    """

    layout = str(u_dis_layout)
    if layout not in {"global_bands", "compact_outer_window"}:
        raise ValueError("u_dis_layout must be global_bands or compact_outer_window")
    isometry_threshold = float(isometry_tolerance)
    kpoint_threshold = float(kpoint_tolerance)
    if (
        not np.isfinite(isometry_threshold)
        or isometry_threshold <= 0.0
        or not np.isfinite(kpoint_threshold)
        or kpoint_threshold <= 0.0
    ):
        raise ValueError("native basis tolerances must be finite and positive")

    epr = Path(epr_path)
    with h5py.File(epr, "r") as handle:
        required = (
            "basic_data/spinor",
            "basic_data/num_wann",
            "basic_data/kc_dim",
            "basic_data/qc_dim",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"{epr} is missing EPR metadata: {missing}")
        if not bool(handle["basic_data/spinor"][()]):
            raise ValueError(f"{epr} does not declare basic_data/spinor=1")
        num_wann = int(handle["basic_data/num_wann"][()])
        kmesh = tuple(int(value) for value in handle["basic_data/kc_dim"][()])
        qmesh = tuple(int(value) for value in handle["basic_data/qc_dim"][()])
    if num_wann < 2 or num_wann % 2:
        raise ValueError(
            f"native spinor num_wann must be positive and even; got {num_wann}"
        )
    if len(kmesh) != 3 or len(qmesh) != 3 or min(*kmesh, *qmesh) <= 0:
        raise ValueError(f"invalid EPR meshes kc_dim={kmesh}, qc_dim={qmesh}")
    num_kpoints = int(np.prod(kmesh))

    win = parse_wannier_win_parameters(win_path)
    amn = read_wannier_amn_header(amn_path)
    if win["num_wann"] != num_wann or amn.num_projections != num_wann:
        raise ValueError("WIN/AMN projection counts do not match EPR num_wann")
    if amn.num_kpoints != num_kpoints:
        raise ValueError("AMN k-point count does not match the EPR k mesh")
    num_bands = amn.num_bands
    indices_array = np.asarray(projection_indices, dtype=np.int64)
    if indices_array.ndim != 1 or indices_array.size < 2 or indices_array.size % 2:
        raise ValueError(
            "magnetic projection selection must contain a positive even count"
        )
    if (
        np.any(indices_array < 0)
        or np.any(indices_array >= amn.num_projections)
        or np.unique(indices_array).size != indices_array.size
    ):
        raise ValueError("magnetic projection indices are duplicate or out of range")
    indices = tuple(int(value) for value in indices_array)

    u = read_wannier_u_matrix_data(u_matrix_path, num_kpoints)
    u_dis_raw = read_wannier_u_matrix_data(u_dis_matrix_path, num_kpoints)
    if u.matrices.shape != (num_kpoints, num_wann, num_wann):
        raise ValueError(f"unexpected U matrix shape {u.matrices.shape}")
    if u_dis_raw.matrices.shape != (num_kpoints, num_bands, num_wann):
        raise ValueError(f"unexpected U_dis matrix shape {u_dis_raw.matrices.shape}")
    eigenvalues = read_wannier_eig(
        eig_path,
        expected_bands=num_bands,
        expected_kpoints=num_kpoints,
    ).eigenvalues
    if layout == "compact_outer_window":
        lower = win.get("dis_win_min")
        upper = win.get("dis_win_max")
        if not isinstance(lower, float) or not isinstance(upper, float):
            raise ValueError(
                "compact U_dis expansion requires dis_win_min and dis_win_max"
            )
        u_dis = expand_compact_wannier_u_dis(
            u_dis_raw.matrices,
            eigenvalues,
            outer_window_min=lower,
            outer_window_max=upper,
        )
    else:
        u_dis = np.asarray(u_dis_raw.matrices, dtype=np.complex128)

    kpoints = _uniform_reduced_grid(kmesh)
    residual = max(
        _periodic_residual(u.kpoints, kpoints),
        _periodic_residual(u_dis_raw.kpoints, kpoints),
        _periodic_residual(u.kpoints, u_dis_raw.kpoints),
    )
    if residual > kpoint_threshold:
        raise ValueError(
            f"native Wannier k-point alignment residual {residual:.6e} exceeds "
            f"{kpoint_threshold:.6e}"
        )
    transform = np.asarray(u_dis @ u.matrices, dtype=np.complex128)
    identity = np.eye(num_wann, dtype=np.complex128)
    isometry = float(
        np.max(
            np.abs(np.swapaxes(transform.conj(), 1, 2) @ transform - identity),
            initial=0.0,
        )
    )
    if isometry > isometry_threshold:
        raise ValueError(
            f"native U_dis U isometry residual {isometry:.6e} exceeds "
            f"{isometry_threshold:.6e}"
        )

    selected_overlap = np.empty(
        (num_kpoints, num_wann, len(indices)),
        dtype=np.complex128,
    )
    records = 0
    for selection, projection in iter_wannier_amn_chunks(amn_path):
        selected = projection[..., indices_array]
        selected_overlap[selection] = (
            np.swapaxes(transform[selection].conj(), 1, 2) @ selected
        )
        records += selection.stop - selection.start
    if records != num_kpoints:
        raise ValueError(f"AMN yielded {records} k points; expected {num_kpoints}")

    return NativeSpinorBasis(
        num_bands=num_bands,
        num_wann=num_wann,
        kmesh=kmesh,
        qmesh=qmesh,
        kpoints=kpoints,
        minus_k_index=_minus_k_indices(kmesh),
        transform=transform,
        hamiltonian_eV=_hamiltonian_from_eigenvalues(transform, eigenvalues),
        selected_overlap=selected_overlap,
        projection_indices=indices,
        u_isometry_residual=isometry,
        kpoint_residual=residual,
    )


def _hr_hamiltonian(
    path: str | Path,
    kpoints: NDArray[np.float64],
    expected_dimension: int,
) -> NDArray[np.complex128]:
    dimension, degeneracies, mapping = read_wannier_hr(path)
    if dimension != expected_dimension:
        raise ValueError(
            f"Wannier HR dimension {dimension} does not match {expected_dimension}"
        )
    r_vectors = np.asarray(list(mapping), dtype=np.float64)
    if len(degeneracies) != r_vectors.shape[0]:
        raise ValueError("Wannier HR degeneracy and R-vector counts differ")
    matrices = np.stack(
        [
            np.asarray(
                mapping[(int(vector[0]), int(vector[1]), int(vector[2]))],
                dtype=np.complex128,
            )
            / int(degeneracies[index])
            for index, vector in enumerate(r_vectors)
        ]
    )
    phases = np.exp(2j * np.pi * (kpoints @ r_vectors.T))
    hamiltonian = np.einsum("kr,rij->kij", phases, matrices, optimize=True)
    return np.asarray(
        0.5 * (hamiltonian + np.swapaxes(hamiltonian.conj(), 1, 2)),
        dtype=np.complex128,
    )


def _transverse_frame(direction: object) -> NDArray[np.float64]:
    normal = np.asarray(direction, dtype=np.float64)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("magnetization direction must be a finite 3-vector")
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("magnetization direction must be nonzero")
    normal /= norm
    seed = np.zeros(3, dtype=np.float64)
    seed[int(np.argmin(np.abs(normal)))] = 1.0
    first = np.cross(seed, normal)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return np.stack((first, second, normal))


@dataclass(frozen=True)
class NativeSpinorEPRAudit:
    production_ready: bool
    u_dis_layout: str
    atomic_spin_order: str
    num_bands: int
    num_wann: int
    num_kpoints: int
    kmesh: tuple[int, int, int]
    qmesh: tuple[int, int, int]
    spinor_flag: bool
    u_isometry_residual: float
    kpoint_residual: float
    hr_hamiltonian_rms_residual_eV: float | None
    hr_hamiltonian_max_residual_eV: float | None
    projection_min_singular_value: float
    projection_p01_singular_value: float
    projection_median_min_singular_value: float
    projection_max_condition_number: float
    projector_tr_covariance_median_relative: float
    projector_tr_covariance_max_relative: float
    sewing_unitarity_residual: float
    sewing_theta_square_residual: float
    sewing_neighbor_jump_median_relative: float
    sewing_neighbor_jump_p95_relative: float
    sewing_neighbor_jump_max_relative: float
    sewing_realspace_r0_weight_fraction: float
    sewing_realspace_top8_weight_fraction: float
    h_hermiticity_residual_eV: float
    h_reconstruction_residual_eV: float
    h_trs_even_residual_eV: float
    h_xc_odd_residual_eV: float
    h_xc_median_frobenius_eV: float
    h_xc_max_frobenius_eV: float
    spn_hermiticity_residual: float
    projected_spin_closure_residual: float
    torque_hermiticity_residual_eV: float
    transverse_torque_median_frobenius_eV: tuple[float, float]
    longitudinal_torque_median_frobenius_eV: float
    longitudinal_to_transverse_torque_ratio: float
    projection_rank_tolerance: float
    projector_tr_tolerance: float
    longitudinal_torque_tolerance: float
    failure_reasons: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        """Compatibility alias retained for the diagnostic CLI."""

        return self.production_ready

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_native_spinor_epr(
    epr_path: str | Path,
    *,
    win_path: str | Path,
    amn_path: str | Path,
    eig_path: str | Path,
    spn_path: str | Path,
    u_matrix_path: str | Path,
    u_dis_matrix_path: str | Path,
    u_dis_layout: UDisLayout,
    atomic_spin_order: SpinOrder | str,
    magnetization_direction: object,
    hr_path: str | Path | None = None,
    algebra_tolerance: float = 1.0e-6,
    hamiltonian_tolerance_eV: float = 1.0e-4,
    projection_rank_tolerance: float = 1.0e-4,
    projector_tr_tolerance: float = 5.0e-2,
    longitudinal_torque_tolerance: float = 5.0e-2,
) -> NativeSpinorEPRAudit:
    """Audit projection-anchored time reversal and exchange extraction.

    The atomic polar-frame construction is a PROJECT-EXTENSION.  It assumes
    that the real, spin-paired AMN trial functions span a time-reversal-closed
    Wannier subspace.  The returned rank, covariance, locality, and
    longitudinal-torque diagnostics test that assumption; canonical Pauli
    form in the final MLWF gauge is neither required nor expected.
    """

    algebra_threshold = float(algebra_tolerance)
    hamiltonian_threshold = float(hamiltonian_tolerance_eV)
    rank_threshold = float(projection_rank_tolerance)
    covariance_threshold = float(projector_tr_tolerance)
    longitudinal_threshold = float(longitudinal_torque_tolerance)
    tolerances = (
        algebra_threshold,
        hamiltonian_threshold,
        rank_threshold,
        covariance_threshold,
        longitudinal_threshold,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError("native EPR audit tolerances must be finite and positive")
    layout = str(u_dis_layout)
    if layout not in {"global_bands", "compact_outer_window"}:
        raise ValueError("u_dis_layout must be global_bands or compact_outer_window")
    order = SpinOrder(atomic_spin_order)

    epr = Path(epr_path)
    with h5py.File(epr, "r") as handle:
        required = (
            "basic_data/spinor",
            "basic_data/num_wann",
            "basic_data/kc_dim",
            "basic_data/qc_dim",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"{epr} is missing EPR metadata: {missing}")
        spinor_flag = bool(handle["basic_data/spinor"][()])
        num_wann = int(handle["basic_data/num_wann"][()])
        kmesh = tuple(int(value) for value in handle["basic_data/kc_dim"][()])
        qmesh = tuple(int(value) for value in handle["basic_data/qc_dim"][()])
    if not spinor_flag:
        raise ValueError(f"{epr} does not declare basic_data/spinor=1")
    if num_wann % 2:
        raise ValueError(f"native spinor num_wann must be even; got {num_wann}")
    if len(kmesh) != 3 or len(qmesh) != 3 or min(*kmesh, *qmesh) <= 0:
        raise ValueError(f"invalid EPR meshes kc_dim={kmesh}, qc_dim={qmesh}")
    num_kpoints = int(np.prod(kmesh))

    win = parse_wannier_win_parameters(win_path)
    amn = read_wannier_amn_header(amn_path)
    spn = read_wannier_spn_header(spn_path)
    if win["num_wann"] != num_wann or amn.num_projections != num_wann:
        raise ValueError("WIN/AMN projection counts do not match EPR num_wann")
    if amn.num_bands != spn.num_bands or amn.num_kpoints != spn.num_kpoints:
        raise ValueError("AMN and SPN dimensions differ")
    if amn.num_kpoints != num_kpoints:
        raise ValueError("AMN/SPN k-point count does not match the EPR k mesh")
    num_bands = amn.num_bands

    u = read_wannier_u_matrix_data(u_matrix_path, num_kpoints)
    u_dis_raw = read_wannier_u_matrix_data(u_dis_matrix_path, num_kpoints)
    if u.matrices.shape != (num_kpoints, num_wann, num_wann):
        raise ValueError(f"unexpected U matrix shape {u.matrices.shape}")
    if u_dis_raw.matrices.shape != (num_kpoints, num_bands, num_wann):
        raise ValueError(f"unexpected U_dis matrix shape {u_dis_raw.matrices.shape}")
    eigenvalues = read_wannier_eig(
        eig_path,
        expected_bands=num_bands,
        expected_kpoints=num_kpoints,
    ).eigenvalues
    if layout == "compact_outer_window":
        lower = win.get("dis_win_min")
        upper = win.get("dis_win_max")
        if not isinstance(lower, float) or not isinstance(upper, float):
            raise ValueError(
                "compact U_dis expansion requires dis_win_min and dis_win_max"
            )
        u_dis = expand_compact_wannier_u_dis(
            u_dis_raw.matrices,
            eigenvalues,
            outer_window_min=lower,
            outer_window_max=upper,
        )
    else:
        u_dis = np.asarray(u_dis_raw.matrices, dtype=np.complex128)

    expected_kpoints = _uniform_reduced_grid(kmesh)
    kpoint_residual = max(
        _periodic_residual(u.kpoints, expected_kpoints),
        _periodic_residual(u_dis_raw.kpoints, expected_kpoints),
        _periodic_residual(u.kpoints, u_dis_raw.kpoints),
    )
    transform = np.asarray(u_dis @ u.matrices, dtype=np.complex128)
    identity = np.eye(num_wann, dtype=np.complex128)
    u_isometry_residual = float(
        np.max(
            np.abs(np.swapaxes(transform.conj(), 1, 2) @ transform - identity),
            initial=0.0,
        )
    )

    overlap = np.empty((num_kpoints, num_wann, num_wann), dtype=np.complex128)
    chunk_records = 0
    for selection, projection in iter_wannier_amn_chunks(amn_path):
        overlap[selection] = np.swapaxes(transform[selection].conj(), 1, 2) @ projection
        chunk_records += selection.stop - selection.start
    if chunk_records != num_kpoints:
        raise ValueError(
            f"AMN yielded {chunk_records} k points; expected {num_kpoints}"
        )

    minus = _minus_k_indices(kmesh)
    atomic_sewing = atomic_spin_time_reversal(num_wann, order)
    projected = projection_anchored_sewing(overlap, minus, atomic_sewing)
    sewing = projected.sewing
    singular_values = projected.singular_values
    minimum_singular = singular_values[:, -1]
    condition = singular_values[:, 0] / np.maximum(
        minimum_singular, np.finfo(np.float64).tiny
    )

    gram = np.swapaxes(overlap.conj(), 1, 2) @ overlap
    gram_theta = atomic_sewing @ gram[minus].conj() @ atomic_sewing.conj().T
    gram_relative = _relative_matrix_norms(gram - gram_theta, gram)
    sewing_unitarity = float(
        np.max(
            np.abs(sewing @ np.swapaxes(sewing.conj(), 1, 2) - identity),
            initial=0.0,
        )
    )
    sewing_theta_square = float(
        np.max(np.abs(sewing @ sewing[minus].conj() + identity), initial=0.0)
    )
    sewing_grid = sewing.reshape(*kmesh, num_wann, num_wann)
    neighbor_jumps = np.concatenate(
        [
            (
                np.linalg.norm(
                    (np.roll(sewing_grid, -1, axis=axis) - sewing_grid).reshape(
                        *kmesh, -1
                    ),
                    axis=-1,
                )
                / np.sqrt(num_wann)
            ).ravel()
            for axis in range(3)
        ]
    )
    sewing_realspace = np.fft.ifftn(sewing_grid, axes=(0, 1, 2))
    sewing_weights = np.sum(np.abs(sewing_realspace) ** 2, axis=(-2, -1))
    sorted_weights = np.sort(sewing_weights.ravel())[::-1]
    weight_total = float(np.sum(sewing_weights))

    hamiltonian = _hamiltonian_from_eigenvalues(transform, eigenvalues)
    h_hermiticity = float(
        np.max(
            np.abs(hamiltonian - np.swapaxes(hamiltonian.conj(), 1, 2)),
            initial=0.0,
        )
    )
    h_theta = time_reversed(hamiltonian[minus], sewing)
    h_trs = np.asarray(0.5 * (hamiltonian + h_theta), dtype=np.complex128)
    h_xc = np.asarray(0.5 * (hamiltonian - h_theta), dtype=np.complex128)
    h_reconstruction = float(np.max(np.abs(hamiltonian - h_trs - h_xc), initial=0.0))
    h_trs_even = float(
        np.max(np.abs(time_reversed(h_trs[minus], sewing) - h_trs), initial=0.0)
    )
    h_xc_odd = float(
        np.max(np.abs(time_reversed(h_xc[minus], sewing) + h_xc), initial=0.0)
    )
    h_xc_norm = np.linalg.norm(h_xc.reshape(num_kpoints, -1), axis=1)

    hr_rms: float | None = None
    hr_max: float | None = None
    if hr_path is not None:
        h_hr = _hr_hamiltonian(hr_path, u.kpoints, num_wann)
        hr_difference = hamiltonian - h_hr
        hr_rms = float(np.sqrt(np.mean(np.abs(hr_difference) ** 2)))
        hr_max = float(np.max(np.abs(hr_difference), initial=0.0))

    frame = _transverse_frame(magnetization_direction)
    spn_hermiticity = 0.0
    spin_closure = 0.0
    torque_hermiticity = 0.0
    torque_norm = np.empty((num_kpoints, 3), dtype=np.float64)
    spn_records = 0
    for ik, spin_bloch in enumerate(iter_wannier_spn(spn_path)):
        spn_hermiticity = max(
            spn_hermiticity,
            float(
                np.max(
                    np.abs(spin_bloch - np.swapaxes(spin_bloch.conj(), 1, 2)),
                    initial=0.0,
                )
            ),
        )
        spin_wannier = transform[ik].conj().T @ spin_bloch @ transform[ik]
        spin_squared = sum(axis @ axis for axis in spin_wannier)
        spin_closure = max(
            spin_closure,
            float(np.max(np.abs(spin_squared - 3.0 * identity), initial=0.0)),
        )
        spin_local = np.einsum("ca,aij->cij", frame, spin_wannier, optimize=True)
        torque = 0.5j * (
            h_xc[ik][None, ...] @ spin_local - spin_local @ h_xc[ik][None, ...]
        )
        torque_hermiticity = max(
            torque_hermiticity,
            float(
                np.max(
                    np.abs(torque - np.swapaxes(torque.conj(), 1, 2)),
                    initial=0.0,
                )
            ),
        )
        torque_norm[ik] = np.linalg.norm(torque.reshape(3, -1), axis=1)
        spn_records += 1
    if spn_records != num_kpoints:
        raise ValueError(f"SPN yielded {spn_records} k points; expected {num_kpoints}")
    torque_median = np.median(torque_norm, axis=0)
    transverse_scale = float(np.sqrt(np.mean(torque_median[:2] ** 2)))
    longitudinal_ratio = float(
        torque_median[2] / max(transverse_scale, np.finfo(np.float64).tiny)
    )

    reasons: list[str] = []
    algebra_checks = {
        "U_dis U isometry": u_isometry_residual,
        "k-point alignment": kpoint_residual,
        "sewing unitarity": sewing_unitarity,
        "sewing Theta^2": sewing_theta_square,
        "Hamiltonian Hermiticity": h_hermiticity,
        "H reconstruction": h_reconstruction,
        "H_TRS time-reversal evenness": h_trs_even,
        "H_XC time-reversal oddness": h_xc_odd,
        "SPN Hermiticity": spn_hermiticity,
        "torque Hermiticity": torque_hermiticity,
    }
    for label, residual in algebra_checks.items():
        if residual > algebra_threshold:
            reasons.append(
                f"{label} residual {residual:.6e} exceeds {algebra_threshold:.6e}"
            )
    if hr_max is not None and hr_max > hamiltonian_threshold:
        reasons.append(
            f"U/eig versus HR Hamiltonian residual {hr_max:.6e} eV exceeds "
            f"{hamiltonian_threshold:.6e} eV"
        )
    if float(np.min(minimum_singular)) < rank_threshold:
        reasons.append(
            f"atomic projection minimum singular value "
            f"{float(np.min(minimum_singular)):.6e} is below "
            f"{rank_threshold:.6e}"
        )
    if float(np.max(gram_relative)) > covariance_threshold:
        reasons.append(
            f"atomic projector time-reversal covariance "
            f"{float(np.max(gram_relative)):.6e} exceeds "
            f"{covariance_threshold:.6e}"
        )
    if longitudinal_ratio > longitudinal_threshold:
        reasons.append(
            f"longitudinal/transverse torque ratio {longitudinal_ratio:.6e} "
            f"exceeds {longitudinal_threshold:.6e}"
        )

    return NativeSpinorEPRAudit(
        production_ready=not reasons,
        u_dis_layout=layout,
        atomic_spin_order=order.value,
        num_bands=num_bands,
        num_wann=num_wann,
        num_kpoints=num_kpoints,
        kmesh=kmesh,
        qmesh=qmesh,
        spinor_flag=spinor_flag,
        u_isometry_residual=u_isometry_residual,
        kpoint_residual=kpoint_residual,
        hr_hamiltonian_rms_residual_eV=hr_rms,
        hr_hamiltonian_max_residual_eV=hr_max,
        projection_min_singular_value=float(np.min(minimum_singular)),
        projection_p01_singular_value=float(np.quantile(minimum_singular, 0.01)),
        projection_median_min_singular_value=float(np.median(minimum_singular)),
        projection_max_condition_number=float(np.max(condition)),
        projector_tr_covariance_median_relative=float(np.median(gram_relative)),
        projector_tr_covariance_max_relative=float(np.max(gram_relative)),
        sewing_unitarity_residual=sewing_unitarity,
        sewing_theta_square_residual=sewing_theta_square,
        sewing_neighbor_jump_median_relative=float(np.median(neighbor_jumps)),
        sewing_neighbor_jump_p95_relative=float(np.quantile(neighbor_jumps, 0.95)),
        sewing_neighbor_jump_max_relative=float(np.max(neighbor_jumps)),
        sewing_realspace_r0_weight_fraction=float(
            sewing_weights[(0, 0, 0)] / weight_total
        ),
        sewing_realspace_top8_weight_fraction=float(
            np.sum(sorted_weights[: min(8, sorted_weights.size)]) / weight_total
        ),
        h_hermiticity_residual_eV=h_hermiticity,
        h_reconstruction_residual_eV=h_reconstruction,
        h_trs_even_residual_eV=h_trs_even,
        h_xc_odd_residual_eV=h_xc_odd,
        h_xc_median_frobenius_eV=float(np.median(h_xc_norm)),
        h_xc_max_frobenius_eV=float(np.max(h_xc_norm)),
        spn_hermiticity_residual=spn_hermiticity,
        projected_spin_closure_residual=spin_closure,
        torque_hermiticity_residual_eV=torque_hermiticity,
        transverse_torque_median_frobenius_eV=(
            float(torque_median[0]),
            float(torque_median[1]),
        ),
        longitudinal_torque_median_frobenius_eV=float(torque_median[2]),
        longitudinal_to_transverse_torque_ratio=longitudinal_ratio,
        projection_rank_tolerance=rank_threshold,
        projector_tr_tolerance=covariance_threshold,
        longitudinal_torque_tolerance=longitudinal_threshold,
        failure_reasons=tuple(reasons),
    )


__all__ = [
    "NativeSpinorBasis",
    "NativeSpinorEPRAudit",
    "UDisLayout",
    "audit_native_spinor_epr",
    "load_native_spinor_basis",
    "parse_projection_indices",
]
