"""Projection-anchored spinor-Wannier exchange support.

The legacy Wannier tensor-J kernel assumes that rows of ``*_hr.dat`` form a
global orbital x spin product basis.  Maximally-localized spinor Wannier
functions do not generally satisfy that assumption.  This module instead
uses the declared atomic trial projections to construct a gauge-covariant
magnetic endpoint frame while retaining the full Wannier Hamiltonian in every
Green function.

The SPN matrices are physical Pauli matrices (not spin operators divided by
two).  They are transformed to the Wannier gauge and used as a fail-closed
validation of the projection frame.  They are not inserted as a k-dependent
exchange vertex.  The TB2J algebra itself uses atomic-trial Pauli matrices in
the polar frame, whose columns have an explicit interleaved orbital-spin
ordering inherited from the input projections.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from slw.core.wannier_io import (
    expand_compact_wannier_u_dis,
    parse_wannier_win_parameters,
    read_wannier_eig,
    read_wannier_hr,
    read_wannier_u_matrix_data,
    remap_wannier_hr_with_wsvec,
)
from slw.core.wannier_spin_io import (
    iter_wannier_amn_chunks,
    iter_wannier_spn,
    read_wannier_amn_header,
    read_wannier_spn_header,
)
from slw.exchange.kernels.eph_provider import _unit_scale_to_ev
from slw.exchange.kernels.lkag import _accumulate_tb2j_A_numba
from slw.soc.manifold import read_projection_groups
from slw.soc.spinor import normalize_groupby

_PAULI = np.asarray(
    (
        ((0.0, 1.0), (1.0, 0.0)),
        ((0.0, -1.0j), (1.0j, 0.0)),
        ((1.0, 0.0), (0.0, -1.0)),
    ),
    dtype=np.complex128,
)
_I_SIGMA_Y = np.asarray(((0.0, 1.0), (-1.0, 0.0)), dtype=np.complex128)


@dataclass(frozen=True)
class ProjectedWannierContext:
    """Numerical data for a projection-anchored TB2J integration.

    ``coefficients[k,p,n]`` is ``Q(k)^dagger V_H(k)``.  Consequently Green
    functions built from it include propagation through all ``nw`` Wannier
    states even though their endpoints have only ``nmag_projection`` rows.
    Site offsets refer to the interleaved atomic projection columns in this
    endpoint space.
    """

    eigenvalues: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    kpoints: NDArray[np.float64]
    site_offsets: NDArray[np.int64]
    site_orbital_counts: NDArray[np.int64]
    exchange_projectors: tuple[NDArray[np.complex128], ...]
    site_spin_directions: NDArray[np.float64]
    selected_amn_columns: NDArray[np.int64]
    selected_spatial_indices: NDArray[np.int64]
    diagnostics: dict[str, Any]


def _periodic_kpoint_residual(first: object, second: object) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    difference = left - right
    difference -= np.rint(difference)
    return float(np.max(np.linalg.norm(difference, axis=-1), initial=0.0))


def _canonical_kpoints(kpoints: object, tolerance: float) -> NDArray[np.float64]:
    points = np.mod(np.asarray(kpoints, dtype=np.float64), 1.0)
    points[np.isclose(points, 1.0, atol=tolerance, rtol=0.0)] = 0.0
    points[np.isclose(points, 0.0, atol=tolerance, rtol=0.0)] = 0.0
    return points


def _validate_native_uniform_mesh(
    kpoints: NDArray[np.float64],
    mesh: tuple[int, int, int],
    *,
    tolerance: float = 1.0e-7,
) -> None:
    points = _canonical_kpoints(kpoints, tolerance)
    quantized = np.rint(points / tolerance).astype(np.int64)
    if np.unique(quantized, axis=0).shape[0] != points.shape[0]:
        raise ValueError("native U k-point mesh contains duplicate periodic points")
    counts = tuple(
        int(np.unique(quantized[:, axis]).size) for axis in range(3)
    )
    if counts != mesh:
        raise ValueError(
            f"native U k-point grid dimensions={counts} differ from kmesh={mesh}"
        )
    for axis, count in enumerate(mesh):
        coordinates = np.unique(points[:, axis])
        coordinates.sort()
        spacings = np.diff(np.concatenate((coordinates, coordinates[:1] + 1.0)))
        if not np.allclose(
            spacings,
            1.0 / float(count),
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                f"native U k points are not a uniform mesh along axis {axis}"
            )


def _minus_k_indices_from_points(
    kpoints: NDArray[np.float64],
    *,
    tolerance: float = 1.0e-7,
) -> NDArray[np.int64]:
    """Resolve ``-k`` partners for native U ordering and optional MP shifts."""

    points = _canonical_kpoints(kpoints, tolerance)
    quantized = np.rint(points / tolerance).astype(np.int64)
    lookup: dict[tuple[int, int, int], int] = {}
    for index, row in enumerate(quantized):
        key = tuple(int(value) for value in row)
        if key in lookup:
            raise ValueError("native U k-point mesh contains duplicate periodic points")
        lookup[key] = index
    result = np.empty(points.shape[0], dtype=np.int64)
    for index, point in enumerate(points):
        target = _canonical_kpoints(-point[None, :], tolerance)[0]
        key = tuple(int(value) for value in np.rint(target / tolerance))
        partner = lookup.get(key)
        if partner is None:
            raise ValueError(
                "native U mesh is not closed under k -> -k; missing partner for "
                f"k-point {index} {point.tolist()}"
            )
        residual = _periodic_kpoint_residual(
            points[partner][None, :], target[None, :]
        )
        if residual > tolerance:
            raise ValueError(
                f"native U -k lookup residual={residual:.6e} exceeds {tolerance:.1e}"
            )
        result[index] = partner
    if np.unique(result).size != len(result) or not np.array_equal(
        result[result], np.arange(len(result), dtype=np.int64)
    ):
        raise ValueError("native U -k partner mapping is not a bijective involution")
    return result


def _hamiltonian_from_hr(
    path: str | Path,
    kpoints: NDArray[np.float64],
    *,
    apply_degeneracy: bool,
    unit: str,
    wsvec: str | Path | None = None,
) -> NDArray[np.complex128]:
    dimension, degeneracies, mapping = read_wannier_hr(path)
    if wsvec is not None:
        mapping, _wsvec_info = remap_wannier_hr_with_wsvec(
            wsvec,
            mapping,
            degeneracies,
            apply_degeneracy=apply_degeneracy,
            unit_scale=float(_unit_scale_to_ev(unit)),
        )
        apply_degeneracy = False
        unit = "ev"
    r_vectors = np.asarray(list(mapping), dtype=np.float64)
    blocks = np.stack(
        [np.asarray(value, dtype=np.complex128) for value in mapping.values()],
        axis=0,
    )
    if len(degeneracies) != blocks.shape[0]:
        raise ValueError(
            f"Wannier HR degeneracy count={len(degeneracies)} differs from "
            f"R-point count={blocks.shape[0]}"
        )
    if apply_degeneracy:
        divisor = np.asarray(degeneracies, dtype=np.float64)
        if np.any(divisor <= 0.0):
            raise ValueError("Wannier HR degeneracies must be positive")
        blocks = blocks / divisor[:, None, None]
    blocks *= float(_unit_scale_to_ev(unit))
    phases = np.exp(2.0j * np.pi * (np.asarray(kpoints) @ r_vectors.T))
    hamiltonian = np.einsum("kr,rij->kij", phases, blocks, optimize=True)
    hamiltonian = 0.5 * (
        hamiltonian + np.swapaxes(hamiltonian.conj(), 1, 2)
    )
    if hamiltonian.shape[1:] != (dimension, dimension):
        raise ValueError("internal Wannier HR interpolation dimension mismatch")
    return np.asarray(hamiltonian, dtype=np.complex128)


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


def _expanded_projection_columns(
    spatial_indices: list[int],
    *,
    spatial_count: int,
    groupby: str,
) -> list[int]:
    mode = normalize_groupby(groupby).value
    columns: list[int] = []
    for spatial in spatial_indices:
        if mode == "orbital":
            columns.extend((2 * int(spatial), 2 * int(spatial) + 1))
        elif mode == "spin":
            columns.extend((int(spatial), int(spatial_count) + int(spatial)))
        else:  # pragma: no cover - guarded by normalize_groupby
            raise ValueError(f"unsupported spinor projection order {mode!r}")
    return columns


def _resolve_magnetic_projection_columns(
    win_path: str | Path,
    mag_atoms: tuple[int, ...] | list[int],
    *,
    groupby: str,
    amn_projection_count: int,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    list[str],
]:
    atoms, groups, spatial_count = read_projection_groups(win_path)
    if int(amn_projection_count) != 2 * int(spatial_count):
        raise ValueError(
            f"{win_path} defines {spatial_count} spatial trial functions but AMN "
            f"contains {amn_projection_count} projections; the physical spinor "
            "path requires exactly two declared spin partners per spatial trial"
        )
    selected_columns: list[int] = []
    selected_spatial: list[int] = []
    offsets = [0]
    orbital_counts: list[int] = []
    labels: list[str] = []
    for atom_index in mag_atoms:
        site_groups = [
            group for group in groups if int(group["atom_index"]) == int(atom_index)
        ]
        if not site_groups:
            label = atoms[int(atom_index)] if 0 <= int(atom_index) < len(atoms) else "?"
            raise ValueError(
                f"magnetic atom {int(atom_index)} ({label}) has no declared "
                f"Wannier projection in {win_path}"
            )
        spatial = [
            int(index)
            for group in site_groups
            for index in list(group["indices"])
        ]
        if len(set(spatial)) != len(spatial):
            raise ValueError(
                f"magnetic atom {int(atom_index)} has overlapping projection groups"
            )
        columns = _expanded_projection_columns(
            spatial,
            spatial_count=spatial_count,
            groupby=groupby,
        )
        selected_spatial.extend(spatial)
        selected_columns.extend(columns)
        offsets.append(len(selected_columns))
        orbital_counts.append(len(spatial))
        labels.extend(
            f"{group['atom_label']}:{group['orbital']}" for group in site_groups
        )
    if len(set(selected_columns)) != len(selected_columns):
        raise ValueError("magnetic AMN projection groups overlap between sites")
    return (
        np.asarray(selected_columns, dtype=np.int64),
        np.asarray(selected_spatial, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(orbital_counts, dtype=np.int64),
        labels,
    )


def _pauli_components_interleaved(
    matrices: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2] or matrices.shape[1] % 2:
        raise ValueError(
            "interleaved spinor matrices must have shape (nk,2*norb,2*norb)"
        )
    norb = matrices.shape[1] // 2
    blocks = matrices.reshape(matrices.shape[0], norb, 2, norb, 2)
    return np.asarray(
        0.5 * np.einsum("ats,kisjt->kaij", _PAULI, blocks, optimize=True),
        dtype=np.complex128,
    )


def _site_exchange_projector(
    site_exchange: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.float64], float, float]:
    components = _pauli_components_interleaved(site_exchange)
    mean_components = np.mean(components, axis=0)
    trace_vector = np.real(
        np.trace(mean_components, axis1=-2, axis2=-1)
    )
    norm = float(np.linalg.norm(trace_vector))
    if norm <= 64.0 * np.finfo(np.float64).eps:
        raise ValueError(
            "magnetic projection has no resolvable time-reversal-odd onsite field"
        )
    direction = np.asarray(trace_vector / norm, dtype=np.float64)
    orbital_k = np.einsum("a,kaij->kij", direction, components, optimize=True)
    exchange_projector = np.asarray(np.mean(orbital_k, axis=0), dtype=np.complex128)
    exchange_projector = 0.5 * (
        exchange_projector + exchange_projector.conj().T
    )
    sigma = np.einsum("a,ast->st", direction, _PAULI, optimize=True)
    collinear = np.einsum(
        "kij,st->kisjt", orbital_k, sigma, optimize=True
    ).reshape(site_exchange.shape)
    denominator = np.linalg.norm(site_exchange.reshape(site_exchange.shape[0], -1), axis=1)
    residual = np.linalg.norm(
        (site_exchange - collinear).reshape(site_exchange.shape[0], -1), axis=1
    ) / np.maximum(denominator, np.finfo(np.float64).tiny)
    k_dependence = float(
        np.linalg.norm(orbital_k - exchange_projector[None, ...])
        / max(float(np.linalg.norm(orbital_k)), np.finfo(np.float64).tiny)
    )
    return exchange_projector, direction, float(np.max(residual)), k_dependence


def spin_product_separability(
    hamiltonian_spin_major: object,
) -> dict[str, Any]:
    """Measure whether spin dependence has one global Pauli direction.

    This diagnostic is meaningful for a declared no/additional-SOC-free bare
    Pauli fast path.  It is not a test for SOC in a known product basis; such
    inputs must opt into ``spin_operator='pauli'`` explicitly.
    """

    hamiltonian = np.asarray(hamiltonian_spin_major, dtype=np.complex128)
    if (
        hamiltonian.ndim != 3
        or hamiltonian.shape[1] != hamiltonian.shape[2]
        or hamiltonian.shape[1] % 2
    ):
        raise ValueError(
            f"spin-major Hamiltonian must have shape (nk,2*n,2*n), got {hamiltonian.shape}"
        )
    norb = hamiltonian.shape[1] // 2
    up_up = hamiltonian[:, :norb, :norb]
    up_down = hamiltonian[:, :norb, norb:]
    down_up = hamiltonian[:, norb:, :norb]
    down_down = hamiltonian[:, norb:, norb:]
    components = np.stack(
        (
            0.5 * (up_down + down_up),
            (down_up - up_down) / (2.0j),
            0.5 * (up_up - down_down),
        ),
        axis=1,
    )
    gram = np.real(
        np.einsum("kaij,kbij->ab", components.conj(), components, optimize=True)
    ) / float(max(1, hamiltonian.shape[0]))
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    total = float(np.sum(eigenvalues))
    if total <= 256.0 * np.finfo(np.float64).eps:
        return {
            "residual": 1.0,
            "axis": np.zeros(3, dtype=np.float64),
            "eigenvalues": np.asarray(eigenvalues, dtype=np.float64),
            "spin_dependent_norm": 0.0,
        }
    residual = float((eigenvalues[0] + eigenvalues[1]) / total)
    axis = np.asarray(eigenvectors[:, -1], dtype=np.float64)
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis *= -1.0
    return {
        "residual": residual,
        "axis": axis,
        "eigenvalues": np.asarray(eigenvalues[::-1], dtype=np.float64),
        "spin_dependent_norm": float(np.sqrt(total)),
    }


def load_projected_wannier_context(
    *,
    spinor_hr: str | Path,
    win: str | Path,
    amn: str | Path,
    eig: str | Path,
    spn: str | Path,
    u_mat: str | Path,
    u_dis_mat: str | Path,
    u_dis_layout: str,
    groupby: str,
    mag_atoms: tuple[int, ...] | list[int],
    kmesh: tuple[int, int, int] | list[int],
    kpoints: object,
    efermi: float,
    apply_degeneracy: bool = True,
    hr_unit: str = "ev",
    wsvec: str | Path | None = None,
    projection_rank_tolerance: float = 1.0e-4,
    spin_projection_tolerance: float = 0.4,
    hamiltonian_tolerance_ev: float = 1.0e-4,
    noncollinear_tolerance: float = 0.25,
    intersite_xc_tolerance: float = 0.1,
) -> ProjectedWannierContext:
    """Load and validate one physical spinor Wannier exchange context."""

    layout = str(u_dis_layout).strip().lower()
    if layout not in {"global_bands", "compact_outer_window"}:
        raise ValueError(
            "u_dis_layout must be global_bands or compact_outer_window"
        )
    tolerances = {
        "projection_rank_tolerance": float(projection_rank_tolerance),
        "spin_projection_tolerance": float(spin_projection_tolerance),
        "hamiltonian_tolerance_ev": float(hamiltonian_tolerance_ev),
        "noncollinear_tolerance": float(noncollinear_tolerance),
        "intersite_xc_tolerance": float(intersite_xc_tolerance),
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in tolerances.values()):
        raise ValueError("physical spinor tolerances must be finite and positive")

    mesh = tuple(int(value) for value in kmesh)
    if len(mesh) != 3 or min(mesh, default=0) < 1:
        raise ValueError(
            f"k mesh must contain three positive integers, got {mesh}"
        )
    fermi_energy = float(efermi)
    if not np.isfinite(fermi_energy):
        raise ValueError(f"efermi must be finite, got {efermi!r}")
    input_mesh_kpoints = np.asarray(kpoints, dtype=np.float64)
    if input_mesh_kpoints.shape != (int(np.prod(mesh)), 3):
        raise ValueError(
            f"requested k points have shape {input_mesh_kpoints.shape}; expected "
            f"({int(np.prod(mesh))},3) from kmesh={mesh}"
        )

    win_parameters = parse_wannier_win_parameters(win)
    amn_header = read_wannier_amn_header(amn)
    spn_header = read_wannier_spn_header(spn)
    if (
        amn_header.num_bands != spn_header.num_bands
        or amn_header.num_kpoints != spn_header.num_kpoints
    ):
        raise ValueError("AMN and SPN band/k-point dimensions differ")
    if amn_header.num_kpoints != input_mesh_kpoints.shape[0]:
        raise ValueError(
            "physical spinor exchange requires kmesh dimensions to equal the "
            f"Wannier90 U/AMN/SPN mesh: requested nk={input_mesh_kpoints.shape[0]}, "
            f"file nk={amn_header.num_kpoints}"
        )

    u_data = read_wannier_u_matrix_data(u_mat, amn_header.num_kpoints)
    u_dis_data = read_wannier_u_matrix_data(u_dis_mat, amn_header.num_kpoints)
    nwann = int(u_data.matrices.shape[1])
    if u_data.matrices.shape != (amn_header.num_kpoints, nwann, nwann):
        raise ValueError(f"unexpected U matrix shape {u_data.matrices.shape}")
    if u_dis_data.matrices.shape != (
        amn_header.num_kpoints,
        amn_header.num_bands,
        nwann,
    ):
        raise ValueError(f"unexpected U_dis matrix shape {u_dis_data.matrices.shape}")
    if int(win_parameters.get("num_wann") or -1) != nwann:
        raise ValueError("WIN num_wann does not match U matrix dimension")
    for name, array in (
        ("U k points", u_data.kpoints),
        ("U_dis k points", u_dis_data.kpoints),
        ("U matrices", u_data.matrices),
        ("U_dis matrices", u_dis_data.matrices),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contain non-finite values")

    native_kpoints = np.asarray(u_data.kpoints, dtype=np.float64)
    _validate_native_uniform_mesh(native_kpoints, mesh)
    kpoint_residual = _periodic_kpoint_residual(
        native_kpoints, u_dis_data.kpoints
    )
    if kpoint_residual > 1.0e-7:
        raise ValueError(
            "Wannier U and U_dis k-point order differs; "
            f"periodic residual={kpoint_residual:.6e}"
        )
    input_gamma_mesh_residual = _periodic_kpoint_residual(
        native_kpoints, input_mesh_kpoints
    )

    eigenvalues = read_wannier_eig(
        eig,
        expected_bands=amn_header.num_bands,
        expected_kpoints=amn_header.num_kpoints,
    ).eigenvalues
    if layout == "compact_outer_window":
        lower = win_parameters.get("dis_win_min")
        upper = win_parameters.get("dis_win_max")
        if not isinstance(lower, float) or not isinstance(upper, float):
            raise ValueError(
                "compact U_dis expansion requires dis_win_min/dis_win_max in WIN"
            )
        u_dis = expand_compact_wannier_u_dis(
            u_dis_data.matrices,
            eigenvalues,
            outer_window_min=lower,
            outer_window_max=upper,
        )
    else:
        u_dis = np.asarray(u_dis_data.matrices, dtype=np.complex128)
    transform = np.asarray(u_dis @ u_data.matrices, dtype=np.complex128)
    identity = np.eye(nwann, dtype=np.complex128)
    isometry_residual = float(
        np.max(
            np.abs(np.swapaxes(transform.conj(), 1, 2) @ transform - identity),
            initial=0.0,
        )
    )
    if isometry_residual > 1.0e-7:
        raise ValueError(
            f"U_dis U isometry residual {isometry_residual:.6e} exceeds 1e-7"
        )

    hamiltonian_hr = _hamiltonian_from_hr(
        spinor_hr,
        native_kpoints,
        apply_degeneracy=bool(apply_degeneracy),
        unit=hr_unit,
        wsvec=wsvec,
    )
    if hamiltonian_hr.shape[1:] != (nwann, nwann):
        raise ValueError(
            f"HR dimension={hamiltonian_hr.shape[1]} does not match U num_wann={nwann}"
        )
    hamiltonian_rotations = _hamiltonian_from_eigenvalues(transform, eigenvalues)
    hamiltonian_difference = hamiltonian_hr - hamiltonian_rotations
    hamiltonian_rms = float(np.sqrt(np.mean(np.abs(hamiltonian_difference) ** 2)))
    hamiltonian_max = float(np.max(np.abs(hamiltonian_difference), initial=0.0))
    if hamiltonian_max > tolerances["hamiltonian_tolerance_ev"]:
        raise ValueError(
            "U/U_dis/eig Hamiltonian does not reconstruct spinor_hr in the same "
            f"gauge: max residual={hamiltonian_max:.6e} eV exceeds "
            f"{tolerances['hamiltonian_tolerance_ev']:.6e} eV"
        )

    (
        selected_columns,
        selected_spatial,
        site_offsets,
        site_orbital_counts,
        projection_labels,
    ) = _resolve_magnetic_projection_columns(
        win,
        mag_atoms,
        groupby=groupby,
        amn_projection_count=amn_header.num_projections,
    )
    if selected_columns.size > nwann:
        raise ValueError(
            "selected magnetic AMN projection count exceeds the Wannier "
            f"dimension: selected={selected_columns.size}, num_wann={nwann}"
        )
    overlap = np.empty(
        (amn_header.num_kpoints, nwann, selected_columns.size),
        dtype=np.complex128,
    )
    records = 0
    for selection, projection in iter_wannier_amn_chunks(amn):
        selected = projection[..., selected_columns]
        overlap[selection] = (
            np.swapaxes(transform[selection].conj(), 1, 2) @ selected
        )
        records += int(selection.stop - selection.start)
    if records != amn_header.num_kpoints:
        raise ValueError(f"AMN yielded {records} k points; expected {amn_header.num_kpoints}")
    left, singular_values, right_h = np.linalg.svd(overlap, full_matrices=False)
    minimum_singular = float(np.min(singular_values, initial=np.inf))
    if minimum_singular < tolerances["projection_rank_tolerance"]:
        raise ValueError(
            "magnetic AMN projection is rank deficient: minimum singular value "
            f"{minimum_singular:.6e} is below "
            f"{tolerances['projection_rank_tolerance']:.6e}"
        )
    atomic_frame = np.asarray(left @ right_h, dtype=np.complex128)
    frame_identity = np.eye(selected_columns.size, dtype=np.complex128)
    frame_isometry = float(
        np.max(
            np.abs(
                np.swapaxes(atomic_frame.conj(), 1, 2) @ atomic_frame
                - frame_identity
            ),
            initial=0.0,
        )
    )
    if frame_isometry > 1.0e-10:
        raise ValueError(
            "polar magnetic projection frame is not isometric: residual="
            f"{frame_isometry:.6e} exceeds 1e-10"
        )

    bare_spin = np.asarray(
        [
            np.kron(
                np.eye(selected_columns.size // 2, dtype=np.complex128), axis
            )
            for axis in _PAULI
        ],
        dtype=np.complex128,
    )
    bare_norm = float(np.linalg.norm(bare_spin))
    spin_relative = np.empty(amn_header.num_kpoints, dtype=np.float64)
    spin_axis_relative = np.empty((amn_header.num_kpoints, 3), dtype=np.float64)
    spn_hermiticity = 0.0
    spn_records = 0
    for ik, spin_bloch in enumerate(iter_wannier_spn(spn)):
        spin_bloch = np.asarray(spin_bloch, dtype=np.complex128)
        if not np.all(np.isfinite(spin_bloch)):
            raise ValueError(f"SPN contains non-finite values at k-point {ik}")
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
        spin_atomic = atomic_frame[ik].conj().T @ spin_wannier @ atomic_frame[ik]
        spin_relative[ik] = np.linalg.norm(spin_atomic - bare_spin) / bare_norm
        for axis in range(3):
            spin_axis_relative[ik, axis] = (
                np.linalg.norm(spin_atomic[axis] - bare_spin[axis])
                / np.linalg.norm(bare_spin[axis])
            )
        spn_records += 1
    if spn_records != amn_header.num_kpoints:
        raise ValueError(
            f"SPN yielded {spn_records} k points; expected {amn_header.num_kpoints}"
        )
    spin_projection_max = float(np.max(spin_relative))
    spin_axis_projection_max = float(np.max(spin_axis_relative))
    if not np.all(np.isfinite(spin_relative)) or not np.all(
        np.isfinite(spin_axis_relative)
    ):
        raise ValueError("physical SPN projection residuals are non-finite")
    spin_projection_worst = max(
        spin_projection_max, spin_axis_projection_max
    )
    if spin_projection_worst > tolerances["spin_projection_tolerance"]:
        raise ValueError(
            "physical SPN matrices are incompatible with the declared magnetic "
            "atomic projection frame: worst combined/per-axis relative "
            f"residual={spin_projection_worst:.6e} "
            f"exceeds {tolerances['spin_projection_tolerance']:.6e}"
        )

    frame_dagger = np.swapaxes(atomic_frame.conj(), 1, 2)
    hamiltonian_atomic = np.asarray(
        frame_dagger @ hamiltonian_hr @ atomic_frame,
        dtype=np.complex128,
    )
    minus = _minus_k_indices_from_points(native_kpoints)
    atomic_sewing = np.kron(
        np.eye(selected_columns.size // 2, dtype=np.complex128),
        _I_SIGMA_Y,
    )
    theta_hamiltonian = np.asarray(
        atomic_sewing
        @ hamiltonian_atomic[minus].conj()
        @ atomic_sewing.conj().T,
        dtype=np.complex128,
    )
    exchange = np.asarray(
        0.5 * (hamiltonian_atomic - theta_hamiltonian),
        dtype=np.complex128,
    )
    exchange = np.asarray(
        0.5 * (exchange + np.swapaxes(exchange.conj(), 1, 2)),
        dtype=np.complex128,
    )

    exchange_projectors: list[NDArray[np.complex128]] = []
    directions = np.empty((len(site_orbital_counts), 3), dtype=np.float64)
    noncollinear_residuals = np.empty(len(site_orbital_counts), dtype=np.float64)
    exchange_k_dependence = np.empty(len(site_orbital_counts), dtype=np.float64)
    block_mask = np.zeros((selected_columns.size, selected_columns.size), dtype=bool)
    for site in range(len(site_orbital_counts)):
        start = int(site_offsets[site])
        stop = int(site_offsets[site + 1])
        block_mask[start:stop, start:stop] = True
        projector, direction, residual, k_dependence = _site_exchange_projector(
            exchange[:, start:stop, start:stop]
        )
        exchange_projectors.append(projector)
        directions[site] = direction
        noncollinear_residuals[site] = residual
        exchange_k_dependence[site] = k_dependence
    noncollinear_max = float(np.max(noncollinear_residuals))
    if noncollinear_max > tolerances["noncollinear_tolerance"]:
        raise ValueError(
            "time-reversal-odd field is not sufficiently collinear in the "
            f"magnetic projection: residual={noncollinear_max:.6e} exceeds "
            f"{tolerances['noncollinear_tolerance']:.6e}"
        )
    exchange_k_dependence_max = float(np.max(exchange_k_dependence))
    if exchange_k_dependence_max > tolerances["intersite_xc_tolerance"]:
        raise ValueError(
            "time-reversal-odd site field is excessively nonlocal in lattice "
            f"translations: k-dependence={exchange_k_dependence_max:.6e} "
            f"exceeds {tolerances['intersite_xc_tolerance']:.6e}"
        )
    exchange_norm = max(float(np.linalg.norm(exchange)), np.finfo(np.float64).tiny)
    intersite_fraction = float(np.linalg.norm(exchange[:, ~block_mask]) / exchange_norm)
    if intersite_fraction > tolerances["intersite_xc_tolerance"]:
        raise ValueError(
            "time-reversal-odd field has excessive intersite magnetic support: "
            f"fraction={intersite_fraction:.6e} exceeds "
            f"{tolerances['intersite_xc_tolerance']:.6e}"
        )

    shifted = hamiltonian_hr - fermi_energy * np.eye(
        nwann, dtype=np.complex128
    )
    energy_eigenvalues, energy_eigenvectors = np.linalg.eigh(shifted)
    coefficients = np.asarray(
        frame_dagger @ energy_eigenvectors,
        dtype=np.complex128,
    )
    diagnostics: dict[str, Any] = {
        "operator_mode": "amn_anchored_atomic_pauli_spn_validated",
        "exchange_vertex_operator": "atomic_trial_pauli",
        "spn_usage": "projection_frame_validation_only",
        "u_dis_layout": layout,
        "num_bands": amn_header.num_bands,
        "num_wann": nwann,
        "num_amn_projections": amn_header.num_projections,
        "u_isometry_residual": isometry_residual,
        "atomic_frame_isometry_residual": frame_isometry,
        "kpoint_residual": kpoint_residual,
        "input_gamma_mesh_residual": input_gamma_mesh_residual,
        "hamiltonian_rms_residual_ev": hamiltonian_rms,
        "hamiltonian_max_residual_ev": hamiltonian_max,
        "projection_min_singular_value": minimum_singular,
        "projection_median_min_singular_value": float(
            np.median(singular_values[:, -1])
        ),
        "projection_max_condition_number": float(
            np.max(
                singular_values[:, 0]
                / np.maximum(singular_values[:, -1], np.finfo(np.float64).tiny)
            )
        ),
        "spn_hermiticity_residual": spn_hermiticity,
        "spin_projection_relative_median": float(np.median(spin_relative)),
        "spin_projection_relative_max": spin_projection_max,
        "spin_axis_projection_relative_max": spin_axis_projection_max,
        "spin_projection_worst_relative_max": spin_projection_worst,
        "site_noncollinear_fraction_max": noncollinear_max,
        "site_noncollinear_fraction": noncollinear_residuals,
        "site_exchange_k_dependence": exchange_k_dependence,
        "site_exchange_k_dependence_max": exchange_k_dependence_max,
        "intersite_xc_fraction": intersite_fraction,
        "projection_labels": tuple(projection_labels),
    }
    diagnostics.update(
        {
            f"threshold_{name}": value
            for name, value in tolerances.items()
        }
    )
    return ProjectedWannierContext(
        eigenvalues=np.asarray(energy_eigenvalues, dtype=np.float64),
        coefficients=coefficients,
        kpoints=native_kpoints,
        site_offsets=site_offsets,
        site_orbital_counts=site_orbital_counts,
        exchange_projectors=tuple(exchange_projectors),
        site_spin_directions=directions,
        selected_amn_columns=selected_columns,
        selected_spatial_indices=selected_spatial,
        diagnostics=diagnostics,
    )


def _ordered_unique(values: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for value in values:
        item = tuple(int(component) for component in value)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _sum_green_realspace(
    green_k: NDArray[np.complex128],
    r_vectors: list[tuple[int, ...]],
    kpoints: NDArray[np.float64],
) -> NDArray[np.complex128]:
    r_array = np.asarray(r_vectors, dtype=np.float64).reshape(-1, 3)
    phase = np.exp(-2.0j * np.pi * (r_array @ kpoints.T))
    return np.asarray(
        np.einsum("rk,kij->rij", phase, green_k, optimize=True)
        / float(max(1, len(kpoints))),
        dtype=np.complex128,
    )


def compute_projected_tb2j(
    context: ProjectedWannierContext,
    pair_meta: list[dict[str, Any]],
    energy_mesh: list[tuple[np.complex128, np.complex128]],
    axes: tuple[str, ...],
    *,
    collinear_override: bool = False,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    dict[str, NDArray[np.float64]],
    dict[str, int],
]:
    """Evaluate TB2J using full-G propagation and atomic-frame endpoints."""

    # Imported lazily to keep the low-level loader independent of the legacy
    # tensor driver while reusing its tested decomposition convention.
    from slw.exchange.kernels.j_tensor_epr import _decompose_tb2j_pair

    site_count = int(context.site_orbital_counts.size)
    used_sites = {
        int(meta[key]) for meta in pair_meta for key in ("li", "lj")
    }
    missing = sorted(site for site in used_sites if site < 0 or site >= site_count)
    if missing:
        raise ValueError(
            f"exchange pairs reference missing projected magnetic sites: {missing}"
        )
    requested_pairs = _ordered_unique(
        [(int(meta["li"]), int(meta["lj"])) for meta in pair_meta]
    )
    accumulated_pairs = list(requested_pairs)
    pair_set = set(accumulated_pairs)
    for first, second in requested_pairs:
        if (second, first) not in pair_set:
            accumulated_pairs.append((second, first))
            pair_set.add((second, first))

    requested_r = _ordered_unique(
        [tuple(int(value) for value in meta["R"]) for meta in pair_meta]
    )
    accumulated_r = list(requested_r)
    r_set = set(accumulated_r)
    for r_vector in requested_r:
        reverse = tuple(-value for value in r_vector)
        if reverse not in r_set:
            accumulated_r.append(reverse)
            r_set.add(reverse)

    n_pairs = len(accumulated_pairs)
    n_r = len(accumulated_r)
    ni = np.asarray(
        [context.site_orbital_counts[first] for first, _ in accumulated_pairs],
        dtype=np.int64,
    )
    nj = np.asarray(
        [context.site_orbital_counts[second] for _, second in accumulated_pairs],
        dtype=np.int64,
    )
    max_ni = int(np.max(ni, initial=0))
    max_nj = int(np.max(nj, initial=0))
    p_i = np.zeros((n_pairs, max_ni, max_ni), dtype=np.complex128)
    p_j = np.zeros((n_pairs, max_nj, max_nj), dtype=np.complex128)
    iu_up = np.zeros((n_pairs, max_ni), dtype=np.int64)
    iu_down = np.zeros((n_pairs, max_ni), dtype=np.int64)
    jv_up = np.zeros((n_pairs, max_nj), dtype=np.int64)
    jv_down = np.zeros((n_pairs, max_nj), dtype=np.int64)
    for pair_index, (first, second) in enumerate(accumulated_pairs):
        first_count = int(ni[pair_index])
        second_count = int(nj[pair_index])
        first_offset = int(context.site_offsets[first])
        second_offset = int(context.site_offsets[second])
        p_i[pair_index, :first_count, :first_count] = context.exchange_projectors[first]
        p_j[pair_index, :second_count, :second_count] = context.exchange_projectors[second]
        iu_up[pair_index, :first_count] = first_offset + 2 * np.arange(first_count)
        iu_down[pair_index, :first_count] = iu_up[pair_index, :first_count] + 1
        jv_up[pair_index, :second_count] = second_offset + 2 * np.arange(second_count)
        jv_down[pair_index, :second_count] = jv_up[pair_index, :second_count] + 1

    accumulator = np.zeros((n_pairs, n_r, 4, 4), dtype=np.complex128)
    all_r = accumulated_r + [tuple(-value for value in r) for r in accumulated_r]
    coefficients = context.coefficients
    for energy, weight in energy_mesh:
        inverse = 1.0 / (energy - context.eigenvalues)
        green_magnetic_k = np.einsum(
            "kia,ka,kja->kij",
            coefficients,
            inverse,
            coefficients.conj(),
            optimize=True,
        )
        green_all_r = _sum_green_realspace(
            green_magnetic_k,
            all_r,
            context.kpoints,
        )
        _accumulate_tb2j_A_numba(
            GR=green_all_r[:n_r],
            GRm=green_all_r[n_r:],
            ni_arr=ni,
            nj_arr=nj,
            iu_up=iu_up,
            iu_dn=iu_down,
            jv_up=jv_up,
            jv_dn=jv_down,
            P_i=p_i,
            P_j=p_j,
            acc_A=accumulator,
            weight_over_pi=weight / np.pi,
        )

    pair_to_index = {pair: index for index, pair in enumerate(accumulated_pairs)}
    r_to_index = {r: index for index, r in enumerate(accumulated_r)}
    tensor = np.zeros((len(pair_meta), len(axes), len(axes)), dtype=np.float64)
    extra = {
        "jiso_tb2j": np.zeros(len(pair_meta), dtype=np.float64),
        "dmi_tb2j": np.zeros((len(pair_meta), 3), dtype=np.float64),
        "jani_tb2j": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
        "J_iso_tensor_r": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
        "J_gamma_r": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
        "J_antisym_aab_r": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
        "J_dmi_tensor_r": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
        "J_aab_full_r": np.zeros((len(pair_meta), 3, 3), dtype=np.float64),
    }
    axis_index = {"x": 0, "y": 1, "z": 2}
    for bond, metadata in enumerate(pair_meta):
        first = int(metadata["li"])
        second = int(metadata["lj"])
        r_vector = tuple(int(value) for value in metadata["R"])
        pair_index = pair_to_index[(first, second)]
        r_index = r_to_index[r_vector]
        reverse_pair = pair_to_index.get((second, first))
        reverse_r = r_to_index.get(tuple(-value for value in r_vector))
        value = accumulator[pair_index, r_index]
        reverse_value = (
            accumulator[reverse_pair, reverse_r]
            if reverse_pair is not None and reverse_r is not None
            else np.zeros((4, 4), dtype=np.complex128)
        )
        decomposition = _decompose_tb2j_pair(
            value,
            reverse_value,
            collinear_override=collinear_override,
            collinear_direction=context.site_spin_directions[first],
        )
        physical = decomposition["jfull_dmi"]
        for first_axis, axis_a in enumerate(axes):
            for second_axis, axis_b in enumerate(axes):
                tensor[bond, first_axis, second_axis] = (
                    1000.0 * physical[axis_index[axis_a], axis_index[axis_b]]
                )
        extra["jiso_tb2j"][bond] = 1000.0 * decomposition["jiso"]
        extra["dmi_tb2j"][bond] = 1000.0 * decomposition["dmi"]
        extra["jani_tb2j"][bond] = 1000.0 * decomposition["m_raw"]
        extra["J_iso_tensor_r"][bond] = (
            1000.0 * np.eye(3) * decomposition["jiso"]
        )
        extra["J_gamma_r"][bond] = 1000.0 * decomposition["gamma"]
        extra["J_antisym_aab_r"][bond] = 1000.0 * decomposition["ma"]
        extra["J_dmi_tensor_r"][bond] = 1000.0 * decomposition["dmi_mat"]
        extra["J_aab_full_r"][bond] = 1000.0 * decomposition["jfull_aab"]
    return tensor, accumulator, extra, {
        "n_chunks": 1,
        "nproc": 1,
        "n_pairs_acc": n_pairs,
        "n_R_acc": n_r,
    }


__all__ = [
    "ProjectedWannierContext",
    "compute_projected_tb2j",
    "load_projected_wannier_context",
    "spin_product_separability",
]
