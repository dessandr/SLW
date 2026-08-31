"""MPI q=0 null test from a pair of scalar collinear EPR files."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.epc.epr_io import EPRMetadata, read_epr_metadata
from slw.soc import (
    add_atomic_d_soc,
    add_atomic_p_soc,
    reorder_spinor_matrix,
)
from slw.wtorque.basis import SpinOrder
from slw.wtorque.green.real_axis import RealAxisIntegrator
from slw.wtorque.io.native_epr_g import reconstruct_epr_g_at_q
from slw.wtorque.io.native_epr_h import reconstruct_epr_hamiltonian
from slw.wtorque.model.exchange_field import split_collinear
from slw.wtorque.model.magnetic_subspace import (
    MagneticSubspace,
    transverse_frame,
)
from slw.wtorque.model.time_reversal import atomic_spin_time_reversal
from slw.wtorque.parallel.mpi import MPIContext
from slw.wtorque.torque.kernel import (
    retarded_bubble_loop_eigh,
    retarded_bubble_loop_eigh_zero_temperature,
)
from slw.wtorque.torque.qpair import finalize_q_pair
from slw.wtorque.torque.vertices import finite_q_vertices, spin_matrix


@dataclass(frozen=True)
class CollinearEPRQ0Result:
    approximation: str
    orbital_dimension_per_spin: int
    green_function_dimension: int
    magnetic_site_count: int
    magnetic_atom_indices_one_based: tuple[int, ...]
    magnetic_orbital_indices_one_based: tuple[tuple[int, ...], ...]
    local_magnetization_directions: tuple[tuple[float, float, float], ...]
    spin_quantization_direction: tuple[float, float, float]
    spin_coordinate: str
    kmesh: tuple[int, int, int]
    qmesh: tuple[int, int, int]
    qpoint_reduced: tuple[float, float, float]
    mpi_size: int
    fermi_energy_eV: float
    integration_energy_range_eV: tuple[float, float]
    integration_energy_points: int
    integration_backend: str
    eta_eV: float
    temperature_K: float
    epr_input_energy_unit: str
    epr_input_displacement_unit: str
    output_units: str
    direct_term_enabled: bool
    selection_rule_expected_zero: bool
    onsite_soc_manifolds: tuple[str, ...]
    onsite_soc_matrix_norm_eV: float
    onsite_soc_spin_flip_max_eV: float
    onsite_soc_hermiticity_residual_eV: float
    onsite_soc_time_reversal_residual_eV: float
    wannier_center_pair_max_periodic_delta: float
    hamiltonian_hermiticity_residual_eV: float
    g_q0_hermiticity_residual_eV_per_angstrom: float
    hamiltonian_spin_flip_max_eV: float
    g_spin_flip_max_eV_per_angstrom: float
    g_spin_quantization_commutator_max_eV_per_angstrom: float
    torque_spin_conserving_max_eV: float
    torque_vertex_rms_eV: float
    onsite_exchange_mean_eV: tuple[float, ...]
    hamiltonian_band_range_eV: tuple[float, float]
    g_q0_rms_eV_per_angstrom: float
    retarded_loop_real: list[list[list[list[float]]]]
    retarded_loop_imag: list[list[list[list[float]]]]
    kernel_eV_per_angstrom: list[list[list[list[float]]]]
    kernel_imaginary_residual_eV_per_angstrom: float
    kernel_max_abs_eV_per_angstrom: float
    kernel_null_max_abs_eV_per_angstrom: float
    acoustic_sum_kernel_eV_per_angstrom: list[list[list[float]]]
    acoustic_sum_max_eV_per_angstrom: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnsiteSOCManifoldInput:
    orbital: str
    orbital_indices: tuple[int, ...]
    lambda_eV: float

    @property
    def label_one_based(self) -> str:
        indices = ",".join(str(index + 1) for index in self.orbital_indices)
        return f"{self.orbital}:{indices}:{self.lambda_eV:.16g}"


def parse_magnetic_orbital_groups(
    specification: str,
    orbital_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Parse one-based groups such as ``1-5;6-10`` into zero-based indices."""

    count = int(orbital_count)
    if count < 1:
        raise ValueError("orbital_count must be positive")
    raw_groups = str(specification).split(";")
    if not raw_groups or any(not group.strip() for group in raw_groups):
        raise ValueError(
            "magnetic orbital groups must be semicolon-separated nonempty selections"
        )
    groups: list[tuple[int, ...]] = []
    used: set[int] = set()
    for raw_group in raw_groups:
        selected: list[int] = []
        for raw_token in raw_group.split(","):
            token = raw_token.strip()
            if not token:
                raise ValueError(f"empty orbital token in {specification!r}")
            if "-" in token:
                fields = token.split("-")
                if len(fields) != 2 or not all(
                    field.strip().isdigit() for field in fields
                ):
                    raise ValueError(f"invalid orbital range {token!r}")
                first, last = (int(field) for field in fields)
                if last < first:
                    raise ValueError(f"orbital range must increase: {token!r}")
                selected.extend(range(first - 1, last))
            else:
                if not token.isdigit():
                    raise ValueError(f"invalid orbital index {token!r}")
                selected.append(int(token) - 1)
        if any(index < 0 or index >= count for index in selected):
            raise ValueError(
                f"orbital indices must lie in 1..{count}; got "
                f"{[index + 1 for index in selected]}"
            )
        if len(set(selected)) != len(selected):
            raise ValueError(f"duplicate orbital within group {raw_group!r}")
        overlap = used.intersection(selected)
        if overlap:
            raise ValueError(
                "magnetic orbital groups must be disjoint; overlap="
                f"{[index + 1 for index in sorted(overlap)]}"
            )
        used.update(selected)
        groups.append(tuple(selected))
    return tuple(groups)


def parse_onsite_soc_manifolds(
    specifications: Iterable[object],
    orbital_count: int,
) -> tuple[OnsiteSOCManifoldInput, ...]:
    """Parse repeated ``p:11-13:0.5`` or ``d:1-5:0.05`` specifications."""

    raw_values = tuple(str(value).strip() for value in specifications)
    parsed: list[OnsiteSOCManifoldInput] = []
    used: set[int] = set()
    for raw in raw_values:
        fields = raw.split(":")
        if len(fields) != 3:
            raise ValueError(
                "onsite SOC must use ORBITAL:ONE_BASED_INDICES:LAMBDA_EV; "
                f"got {raw!r}"
            )
        orbital = fields[0].strip().lower()
        if orbital not in {"p", "d"}:
            raise ValueError(f"onsite SOC orbital must be p or d; got {orbital!r}")
        groups = parse_magnetic_orbital_groups(fields[1], orbital_count)
        if len(groups) != 1:
            raise ValueError("each onsite SOC entry must select exactly one manifold")
        indices = groups[0]
        required = 3 if orbital == "p" else 5
        if len(indices) != required:
            raise ValueError(
                f"onsite {orbital} SOC requires {required} orbitals; "
                f"got {[index + 1 for index in indices]}"
            )
        try:
            lambda_eV = float(fields[2])
        except ValueError as exc:
            raise ValueError(f"invalid onsite SOC lambda in {raw!r}") from exc
        if not np.isfinite(lambda_eV):
            raise ValueError(f"onsite SOC lambda must be finite; got {lambda_eV}")
        overlap = used.intersection(indices)
        if overlap:
            raise ValueError(
                "onsite SOC manifolds must be disjoint; overlap="
                f"{[index + 1 for index in sorted(overlap)]}"
            )
        used.update(indices)
        parsed.append(
            OnsiteSOCManifoldInput(
                orbital=orbital,
                orbital_indices=indices,
                lambda_eV=lambda_eV,
            )
        )
    return tuple(parsed)


def _build_onsite_soc(
    orbital_count: int,
    manifolds: tuple[OnsiteSOCManifoldInput, ...],
    *,
    p_orbital_order: str | None,
    d_orbital_order: str | None,
) -> NDArray[np.complex128]:
    blocked = np.zeros(
        (2 * orbital_count, 2 * orbital_count),
        dtype=np.complex128,
    )
    for manifold in manifolds:
        if manifold.orbital == "p":
            if p_orbital_order is None:
                raise ValueError("p onsite SOC requires an explicit p orbital order")
            blocked = add_atomic_p_soc(
                blocked,
                (manifold.orbital_indices,),
                manifold.lambda_eV,
                order=p_orbital_order,
                inplace=True,
            )
        else:
            if d_orbital_order is None:
                raise ValueError("d onsite SOC requires an explicit d orbital order")
            blocked = add_atomic_d_soc(
                blocked,
                (manifold.orbital_indices,),
                manifold.lambda_eV,
                order=d_orbital_order,
                inplace=True,
            )
    return np.asarray(
        reorder_spinor_matrix(blocked, source="spin", target="orbital"),
        dtype=np.complex128,
    )


def _metadata(path: str | Path) -> EPRMetadata:
    with h5py.File(path, "r") as handle:
        return read_epr_metadata(handle)


def _periodic_center_residual(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> float:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    delta -= np.rint(delta)
    return float(np.max(np.linalg.norm(delta, axis=1), initial=0.0))


def _validate_metadata_pair(
    up: EPRMetadata,
    down: EPRMetadata,
) -> float:
    scalar_fields = (
        ("nat", up.nat, down.nat),
        ("nwan", up.nwan, down.nwan),
        ("nk_grid", up.nk_grid, down.nk_grid),
        ("nq_grid", up.nq_grid, down.nq_grid),
    )
    mismatched = [
        name for name, first, second in scalar_fields if first != second
    ]
    if mismatched:
        raise ValueError(
            "spin-resolved EPR metadata differ: " + ", ".join(mismatched)
        )
    if not np.allclose(
        up.lattice_dimensionless,
        down.lattice_dimensionless,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("spin-resolved EPR lattice matrices differ")
    if not np.allclose(
        up.atom_positions_fractional,
        down.atom_positions_fractional,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError("spin-resolved EPR atom positions differ")
    return _periodic_center_residual(
        up.wannier_centres_fractional,
        down.wannier_centres_fractional,
    )


def _normalized_directions(
    value: object,
    count: int,
    *,
    name: str,
) -> NDArray[np.float64]:
    directions = np.asarray(value, dtype=np.float64)
    if directions.shape != (count, 3) or not np.all(np.isfinite(directions)):
        raise ValueError(f"{name} must have finite shape ({count},3)")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} entries must be nonzero")
    return np.asarray(directions / norms[:, None], dtype=np.float64)


def _float3(value: object) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"expected a three-vector, got shape {array.shape}")
    return (float(array[0]), float(array[1]), float(array[2]))


def _int3(value: object) -> tuple[int, int, int]:
    array = np.asarray(value, dtype=np.int64)
    if array.shape != (3,):
        raise ValueError(f"expected three mesh dimensions, got shape {array.shape}")
    return (int(array[0]), int(array[1]), int(array[2]))


def _float2(value: object) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"expected two bounds, got shape {array.shape}")
    return (float(array[0]), float(array[1]))


def _spin_off_diagonal_max(value: NDArray[np.complex128], norb: int) -> float:
    blocks = value.reshape(*value.shape[:-2], norb, 2, norb, 2)
    return float(
        max(
            np.max(np.abs(blocks[..., :, 0, :, 1]), initial=0.0),
            np.max(np.abs(blocks[..., :, 1, :, 0]), initial=0.0),
        )
    )


def _spin_diagonal_max(value: NDArray[np.complex128], norb: int) -> float:
    blocks = value.reshape(*value.shape[:-2], norb, 2, norb, 2)
    return float(
        max(
            np.max(np.abs(blocks[..., :, 0, :, 0]), initial=0.0),
            np.max(np.abs(blocks[..., :, 1, :, 1]), initial=0.0),
        )
    )


def _root_setup(
    *,
    up_epr_path: str | Path,
    down_epr_path: str | Path,
    magnetic_orbital_specification: str,
    magnetic_atom_indices: tuple[int, ...],
    local_magnetization_directions: object,
    spin_quantization_direction: object,
    spin_coordinate: str,
    onsite_soc_specifications: tuple[str, ...],
    p_orbital_order: str | None,
    d_orbital_order: str | None,
    epr_energy_unit: str,
    epr_displacement_unit: str,
    divide_by_ws_degeneracy: bool,
    mpi_size: int,
) -> tuple[list[tuple[np.ndarray, ...]], dict[str, Any]]:
    up_metadata = _metadata(up_epr_path)
    down_metadata = _metadata(down_epr_path)
    center_residual = _validate_metadata_pair(up_metadata, down_metadata)
    norb = up_metadata.nwan
    magnetic_orbital_groups = parse_magnetic_orbital_groups(
        magnetic_orbital_specification,
        norb,
    )
    onsite_soc_manifolds = parse_onsite_soc_manifolds(
        onsite_soc_specifications,
        norb,
    )
    nmag = len(magnetic_orbital_groups)
    if nmag < 1:
        raise ValueError("at least one magnetic orbital group is required")
    if len(magnetic_atom_indices) != nmag:
        raise ValueError(
            "magnetic atom count must match the magnetic orbital group count"
        )
    atoms = np.asarray(magnetic_atom_indices, dtype=np.int64)
    if np.any(atoms < 0) or np.any(atoms >= up_metadata.nat):
        raise ValueError(
            f"magnetic atom indices must lie in 0..{up_metadata.nat - 1}"
        )
    if np.unique(atoms).size != atoms.size:
        raise ValueError("magnetic atom indices must be unique")
    masks = np.zeros((nmag, norb), dtype=bool)
    for site, group in enumerate(magnetic_orbital_groups):
        indices = np.asarray(group, dtype=np.int64)
        if indices.ndim != 1 or indices.size < 1:
            raise ValueError("each magnetic orbital group must be nonempty")
        if np.any(indices < 0) or np.any(indices >= norb):
            raise ValueError(f"magnetic orbital index outside 0..{norb - 1}")
        masks[site, indices] = True
    magnetic = MagneticSubspace.from_masks(masks)
    directions = _normalized_directions(
        local_magnetization_directions,
        nmag,
        name="local_magnetization_directions",
    )
    quantization = _normalized_directions(
        np.asarray(spin_quantization_direction, dtype=np.float64)[None, :],
        1,
        name="spin_quantization_direction",
    )[0]
    coordinate = str(spin_coordinate)
    if coordinate not in {"rotation_angle", "transverse_direction"}:
        raise ValueError(
            "spin_coordinate must be rotation_angle or transverse_direction"
        )

    up_h = reconstruct_epr_hamiltonian(
        up_epr_path,
        energy_unit=epr_energy_unit,
        expected_spinor=False,
        divide_by_ws_degeneracy=divide_by_ws_degeneracy,
    )
    down_h = reconstruct_epr_hamiltonian(
        down_epr_path,
        energy_unit=epr_energy_unit,
        expected_spinor=False,
        divide_by_ws_degeneracy=divide_by_ws_degeneracy,
    )
    if not np.array_equal(up_h.kpoints, down_h.kpoints):
        raise ValueError("spin-resolved EPR Hamiltonian k-point orders differ")
    h_trs, h_xc = split_collinear(
        up_h.values,
        down_h.values,
        quantization,
    )
    hamiltonian = np.asarray(h_trs + h_xc, dtype=np.complex128)
    magnetic_projector = magnetic.magnetic_projector
    h_xc_magnetic = np.asarray(
        magnetic_projector @ h_xc @ magnetic_projector,
        dtype=np.complex128,
    )
    frames = np.stack(
        [transverse_frame(direction) for direction in directions]
    )
    vertices_site = finite_q_vertices(
        h_xc_magnetic,
        h_xc_magnetic,
        orbital_masks=masks,
        local_frames=frames,
        q_red=np.zeros(3, dtype=np.float64),
        orbital_centers=up_metadata.wannier_centres_fractional,
        magnetic_site_positions=up_metadata.atom_positions_fractional[atoms],
        coordinate_type=coordinate,
    )
    vertices = np.asarray(
        vertices_site.reshape(
            vertices_site.shape[0],
            nmag * 2,
            2 * norb,
            2 * norb,
        ),
        dtype=np.complex128,
    )
    onsite_soc = _build_onsite_soc(
        norb,
        onsite_soc_manifolds,
        p_orbital_order=p_orbital_order,
        d_orbital_order=d_orbital_order,
    )
    hamiltonian = np.asarray(
        hamiltonian + onsite_soc[None, :, :],
        dtype=np.complex128,
    )
    onsite_soc_hermiticity = float(
        np.max(np.abs(onsite_soc - onsite_soc.conj().T), initial=0.0)
    )
    time_reversal = atomic_spin_time_reversal(
        2 * norb,
        SpinOrder.INTERLEAVED,
    )
    time_reversed_soc = (
        time_reversal @ onsite_soc.conj() @ time_reversal.conj().T
    )
    onsite_soc_time_reversal = float(
        np.max(np.abs(time_reversed_soc - onsite_soc), initial=0.0)
    )

    up_g = reconstruct_epr_g_at_q(
        up_epr_path,
        0,
        energy_unit=epr_energy_unit,
        displacement_unit=epr_displacement_unit,
        expected_spinor=False,
        divide_by_ws_degeneracy=divide_by_ws_degeneracy,
    )
    down_g = reconstruct_epr_g_at_q(
        down_epr_path,
        0,
        energy_unit=epr_energy_unit,
        displacement_unit=epr_displacement_unit,
        expected_spinor=False,
        divide_by_ws_degeneracy=divide_by_ws_degeneracy,
    )
    if not np.array_equal(up_g.kpoints, up_h.kpoints):
        raise ValueError("spin-up EPR H and g k-point orders differ")
    if not np.array_equal(down_g.kpoints, down_h.kpoints):
        raise ValueError("spin-down EPR H and g k-point orders differ")
    if up_g.values.shape != down_g.values.shape:
        raise ValueError(
            "spin-resolved EPR g shapes differ: "
            f"{up_g.values.shape} vs {down_g.values.shape}"
        )
    if not np.array_equal(up_g.pert_atom, down_g.pert_atom) or not np.array_equal(
        up_g.pert_cart, down_g.pert_cart
    ):
        raise ValueError("spin-resolved EPR perturbation metadata differ")
    if not np.array_equal(up_g.qpoint, down_g.qpoint):
        raise ValueError("spin-resolved EPR q points differ")
    g_average, g_spin_dependent = split_collinear(
        up_g.values,
        down_g.values,
        quantization,
    )
    perturbations = np.asarray(
        g_average + g_spin_dependent,
        dtype=np.complex128,
    )
    sigma_quantization = spin_matrix(quantization, 2 * norb)
    g_spin_commutator = perturbations @ sigma_quantization - (
        sigma_quantization @ perturbations
    )

    onsite_means: list[float] = []
    onsite_difference = np.mean(up_h.values - down_h.values, axis=0)
    for group in magnetic_orbital_groups:
        indices = np.asarray(group, dtype=np.int64)
        block = onsite_difference[np.ix_(indices, indices)]
        block = 0.5 * (block + block.conj().T)
        onsite_means.append(float(np.mean(np.linalg.eigvalsh(block))))

    h_eigenvalues = np.linalg.eigvalsh(hamiltonian)
    g_hermiticity = max(
        float(up_g.q0_hermiticity_residual or 0.0),
        float(down_g.q0_hermiticity_residual or 0.0),
    )
    metadata: dict[str, Any] = {
        "norb": norb,
        "nmag": nmag,
        "nat": up_metadata.nat,
        "atoms": tuple(int(atom) for atom in atoms),
        "groups": magnetic_orbital_groups,
        "directions": directions,
        "quantization": quantization,
        "coordinate": coordinate,
        "onsite_soc_labels": tuple(
            manifold.label_one_based for manifold in onsite_soc_manifolds
        ),
        "onsite_soc_norm": float(np.linalg.norm(onsite_soc)),
        "onsite_soc_spin_flip": _spin_off_diagonal_max(onsite_soc, norb),
        "onsite_soc_hermiticity": onsite_soc_hermiticity,
        "onsite_soc_time_reversal": onsite_soc_time_reversal,
        "kmesh": up_metadata.nk_grid,
        "qmesh": up_metadata.nq_grid,
        "qpoint": tuple(float(value) for value in up_g.qpoint),
        "center_residual": center_residual,
        "h_hermiticity": max(
            up_h.hermiticity_residual_eV,
            down_h.hermiticity_residual_eV,
        ),
        "g_hermiticity": g_hermiticity,
        "h_spin_flip": _spin_off_diagonal_max(hamiltonian, norb),
        "g_spin_flip": _spin_off_diagonal_max(perturbations, norb),
        "g_spin_commutator": float(
            np.max(np.abs(g_spin_commutator), initial=0.0)
        ),
        "torque_spin_conserving": _spin_diagonal_max(vertices, norb),
        "torque_rms": float(np.sqrt(np.mean(np.abs(vertices) ** 2))),
        "onsite_means": tuple(onsite_means),
        "band_range": (
            float(np.min(h_eigenvalues)),
            float(np.max(h_eigenvalues)),
        ),
        "g_rms": float(np.sqrt(np.mean(np.abs(perturbations) ** 2))),
    }

    nk = hamiltonian.shape[0]
    rank_indices = [
        np.arange(rank, nk, mpi_size, dtype=np.int64)
        for rank in range(mpi_size)
    ]
    payloads: list[tuple[np.ndarray, ...]] = []
    for selected in rank_indices:
        payloads.append(
            (
                np.asarray(hamiltonian[selected], dtype=np.complex128),
                np.asarray(vertices[selected], dtype=np.complex128),
                np.asarray(perturbations[selected], dtype=np.complex128),
                np.full(selected.size, 1.0 / nk, dtype=np.float64),
            )
        )
    return payloads, metadata


def run_collinear_epr_q0_test(
    *,
    up_epr_path: str | Path,
    down_epr_path: str | Path,
    magnetic_orbital_specification: str,
    magnetic_atom_indices: tuple[int, ...],
    local_magnetization_directions: object,
    spin_quantization_direction: object,
    spin_coordinate: str,
    fermi_energy_eV: float,
    energy_min_eV: float,
    energy_max_eV: float,
    energy_points: int,
    eta_eV: float,
    temperature_K: float,
    epr_energy_unit: str,
    epr_displacement_unit: str,
    integration_backend: str = "gauss_legendre",
    onsite_soc_specifications: tuple[str, ...] = (),
    p_orbital_order: str | None = None,
    d_orbital_order: str | None = None,
    divide_by_ws_degeneracy: bool = False,
    perturbation_chunk: int | None = None,
    mpi: MPIContext | None = None,
) -> CollinearEPRQ0Result | None:
    """Evaluate the full-G/full-g collinear no-SOC q=0 null test."""

    context = MPIContext.discover() if mpi is None else mpi
    scalars = {
        "fermi_energy_eV": float(fermi_energy_eV),
        "energy_min_eV": float(energy_min_eV),
        "energy_max_eV": float(energy_max_eV),
        "eta_eV": float(eta_eV),
        "temperature_K": float(temperature_K),
    }
    if any(not np.isfinite(value) for value in scalars.values()):
        raise ValueError("collinear q=0 integration inputs must be finite")
    if scalars["eta_eV"] <= 0.0:
        raise ValueError("eta_eV must be positive")
    if scalars["temperature_K"] < 0.0:
        raise ValueError("temperature_K must be non-negative")
    backend = str(integration_backend)
    if backend not in {"gauss_legendre", "analytic_zero_temperature"}:
        raise ValueError(
            "integration_backend must be gauss_legendre or "
            "analytic_zero_temperature"
        )
    if backend == "analytic_zero_temperature" and scalars["temperature_K"] != 0.0:
        raise ValueError("analytic_zero_temperature requires temperature_K=0")
    if scalars["energy_max_eV"] <= scalars["energy_min_eV"]:
        raise ValueError("energy_max_eV must exceed energy_min_eV")

    setup_error: tuple[str, str] | None = None
    payloads: list[tuple[np.ndarray, ...]] | None = None
    metadata: dict[str, Any] | None = None
    if context.is_root:
        try:
            payloads, metadata = _root_setup(
                up_epr_path=up_epr_path,
                down_epr_path=down_epr_path,
                magnetic_orbital_specification=(
                    magnetic_orbital_specification
                ),
                magnetic_atom_indices=magnetic_atom_indices,
                local_magnetization_directions=(local_magnetization_directions),
                spin_quantization_direction=spin_quantization_direction,
                spin_coordinate=spin_coordinate,
                onsite_soc_specifications=onsite_soc_specifications,
                p_orbital_order=p_orbital_order,
                d_orbital_order=d_orbital_order,
                epr_energy_unit=epr_energy_unit,
                epr_displacement_unit=epr_displacement_unit,
                divide_by_ws_degeneracy=divide_by_ws_degeneracy,
                mpi_size=context.size,
            )
        except Exception as exc:  # noqa: BLE001 - release MPI peers
            setup_error = (type(exc).__name__, str(exc))
    setup_error = context.bcast(setup_error, root=0)
    if setup_error is not None:
        error_type, message = setup_error
        raise RuntimeError(
            f"collinear EPR q=0 setup failed ({error_type}): {message}"
        )
    metadata = context.bcast(metadata, root=0)
    local_h, local_vertices, local_g, local_weights = context.scatter(
        payloads,
        root=0,
    )

    integrator = (
        RealAxisIntegrator.gauss_legendre(
            float(energy_min_eV),
            float(energy_max_eV),
            int(energy_points),
            chemical_potential_eV=float(fermi_energy_eV),
            temperature_K=float(temperature_K),
        )
        if backend == "gauss_legendre"
        else None
    )
    local_error: tuple[int, str, str] | None = None
    if local_h.shape[0] == 0:
        if metadata is None:
            raise RuntimeError("collinear q=0 metadata was not broadcast")
        local_loop = np.zeros(
            (2 * int(metadata["nmag"]), 3 * int(metadata["nat"])),
            dtype=np.complex128,
        )
    else:
        try:
            if backend == "analytic_zero_temperature":
                occupied_max = min(
                    float(energy_max_eV),
                    float(fermi_energy_eV),
                )
                local_loop = retarded_bubble_loop_eigh_zero_temperature(
                    local_h,
                    local_vertices,
                    local_g,
                    local_weights,
                    energy_min_eV=float(energy_min_eV),
                    occupied_energy_max_eV=occupied_max,
                    eta_eV=float(eta_eV),
                    perturbation_chunk=perturbation_chunk,
                )
            else:
                if integrator is None:
                    raise RuntimeError("Gauss-Legendre integrator was not built")
                local_loop = retarded_bubble_loop_eigh(
                    local_h,
                    local_h,
                    local_vertices,
                    local_g,
                    local_weights,
                    integrator,
                    eta_eV=float(eta_eV),
                    perturbation_chunk=perturbation_chunk,
                )
        except Exception as exc:  # noqa: BLE001 - agree before gather
            local_loop = np.empty((0, 0), dtype=np.complex128)
            local_error = (context.rank, type(exc).__name__, str(exc))
    failures = tuple(
        item for item in context.allgather(local_error) if item is not None
    )
    if failures:
        rendered = "; ".join(
            f"rank {rank} {error_type}: {message}"
            for rank, error_type, message in failures
        )
        raise RuntimeError(f"collinear EPR q=0 MPI integration failed: {rendered}")

    gathered = context.gather(local_loop, root=0)
    finalization_error: tuple[str, str] | None = None
    root_result: CollinearEPRQ0Result | None = None
    if context.is_root:
        try:
            if gathered is None or metadata is None:
                raise RuntimeError(
                    "collinear q=0 root received no integration payload"
                )
            retarded = np.sum(np.stack(gathered), axis=0)
            kernel = finalize_q_pair(retarded, retarded)
            nmag = int(metadata["nmag"])
            nat = int(metadata["nat"])
            retarded_shaped = retarded.reshape(nmag, 2, nat, 3)
            kernel_shaped = kernel.reshape(nmag, 2, nat, 3)
            acoustic_sum = np.sum(kernel_shaped, axis=2)
            onsite_soc_enabled = bool(metadata["onsite_soc_labels"])
            kernel_max = float(
                np.max(np.abs(kernel_shaped), initial=0.0)
            )
            root_result = CollinearEPRQ0Result(
                approximation=(
                    "bubble-only collinear EPR with optional TR-even atomic "
                    "onsite SOC in full H/G propagation; spin-diagonal DFPT g; "
                    "input-selected magnetic-subspace exchange"
                ),
                orbital_dimension_per_spin=int(metadata["norb"]),
                green_function_dimension=2 * int(metadata["norb"]),
                magnetic_site_count=nmag,
                magnetic_atom_indices_one_based=tuple(
                    int(atom) + 1 for atom in metadata["atoms"]
                ),
                magnetic_orbital_indices_one_based=tuple(
                    tuple(int(index) + 1 for index in group)
                    for group in metadata["groups"]
                ),
                local_magnetization_directions=tuple(
                    _float3(direction)
                    for direction in metadata["directions"]
                ),
                spin_quantization_direction=_float3(metadata["quantization"]),
                spin_coordinate=str(metadata["coordinate"]),
                kmesh=_int3(metadata["kmesh"]),
                qmesh=_int3(metadata["qmesh"]),
                qpoint_reduced=_float3(metadata["qpoint"]),
                mpi_size=context.size,
                fermi_energy_eV=float(fermi_energy_eV),
                integration_energy_range_eV=(
                    float(energy_min_eV),
                    float(energy_max_eV),
                ),
                integration_energy_points=int(energy_points),
                integration_backend=backend,
                eta_eV=float(eta_eV),
                temperature_K=float(temperature_K),
                epr_input_energy_unit=str(epr_energy_unit),
                epr_input_displacement_unit=str(epr_displacement_unit),
                output_units="eV/angstrom",
                direct_term_enabled=False,
                selection_rule_expected_zero=not onsite_soc_enabled,
                onsite_soc_manifolds=tuple(metadata["onsite_soc_labels"]),
                onsite_soc_matrix_norm_eV=float(metadata["onsite_soc_norm"]),
                onsite_soc_spin_flip_max_eV=float(
                    metadata["onsite_soc_spin_flip"]
                ),
                onsite_soc_hermiticity_residual_eV=float(
                    metadata["onsite_soc_hermiticity"]
                ),
                onsite_soc_time_reversal_residual_eV=float(
                    metadata["onsite_soc_time_reversal"]
                ),
                wannier_center_pair_max_periodic_delta=float(
                    metadata["center_residual"]
                ),
                hamiltonian_hermiticity_residual_eV=float(
                    metadata["h_hermiticity"]
                ),
                g_q0_hermiticity_residual_eV_per_angstrom=float(
                    metadata["g_hermiticity"]
                ),
                hamiltonian_spin_flip_max_eV=float(metadata["h_spin_flip"]),
                g_spin_flip_max_eV_per_angstrom=float(metadata["g_spin_flip"]),
                g_spin_quantization_commutator_max_eV_per_angstrom=float(
                    metadata["g_spin_commutator"]
                ),
                torque_spin_conserving_max_eV=float(
                    metadata["torque_spin_conserving"]
                ),
                torque_vertex_rms_eV=float(metadata["torque_rms"]),
                onsite_exchange_mean_eV=tuple(
                    float(value) for value in metadata["onsite_means"]
                ),
                hamiltonian_band_range_eV=_float2(metadata["band_range"]),
                g_q0_rms_eV_per_angstrom=float(metadata["g_rms"]),
                retarded_loop_real=retarded_shaped.real.tolist(),
                retarded_loop_imag=retarded_shaped.imag.tolist(),
                kernel_eV_per_angstrom=kernel_shaped.real.tolist(),
                kernel_imaginary_residual_eV_per_angstrom=float(
                    np.max(np.abs(kernel_shaped.imag), initial=0.0)
                ),
                kernel_max_abs_eV_per_angstrom=kernel_max,
                kernel_null_max_abs_eV_per_angstrom=kernel_max,
                acoustic_sum_kernel_eV_per_angstrom=(
                    acoustic_sum.real.tolist()
                ),
                acoustic_sum_max_eV_per_angstrom=float(
                    np.max(np.abs(acoustic_sum), initial=0.0)
                ),
            )
        except Exception as exc:  # noqa: BLE001 - release MPI peers
            finalization_error = (type(exc).__name__, str(exc))
    finalization_error = context.bcast(finalization_error, root=0)
    if finalization_error is not None:
        error_type, message = finalization_error
        raise RuntimeError(
            f"collinear EPR q=0 finalization failed ({error_type}): {message}"
        )
    context.barrier()
    return root_result


__all__ = [
    "CollinearEPRQ0Result",
    "parse_magnetic_orbital_groups",
    "run_collinear_epr_q0_test",
]
