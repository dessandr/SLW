"""MPI smoke test for native spinor EPR at the Gamma phonon point."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from slw.core.wannier_spin_io import read_wannier_amn_header
from slw.wtorque.green.real_axis import RealAxisIntegrator
from slw.wtorque.io.native_epr import (
    UDisLayout,
    load_native_spinor_basis,
    parse_projection_indices,
)
from slw.wtorque.io.native_epr_g import reconstruct_epr_g_at_q
from slw.wtorque.model.magnetic_subspace import (
    build_projected_collinear_exchange,
    rotate_projected_exchange,
)
from slw.wtorque.parallel.mpi import MPIContext
from slw.wtorque.torque.kernel import retarded_bubble_loop_eigh
from slw.wtorque.torque.qpair import finalize_q_pair


@dataclass(frozen=True)
class NativeQ0Result:
    approximation: str
    green_function_dimension: int
    magnetic_subspace_dimension: int
    magnetic_projection_indices_one_based: tuple[int, ...]
    kmesh: tuple[int, int, int]
    qmesh: tuple[int, int, int]
    qpoint_reduced: tuple[float, float, float]
    mpi_size: int
    fermi_energy_eV: float
    integration_energy_range_eV: tuple[float, float]
    integration_energy_points: int
    eta_eV: float
    temperature_K: float
    ep_input_energy_unit: str
    ep_input_displacement_unit: str
    output_units: str
    direct_term_enabled: bool
    projection_min_singular_value: float
    projection_median_min_singular_value: float
    projection_max_condition_number: float
    raw_exchange_noncollinear_fraction_median: float
    raw_exchange_noncollinear_fraction_max: float
    raw_longitudinal_vertex_max_frobenius_eV: float
    raw_longitudinal_to_transverse_vertex_ratio: float
    longitudinal_vertex_max_frobenius_eV: float
    longitudinal_to_transverse_vertex_ratio: float
    central_difference_relative_residual: float
    g_q0_hermiticity_residual_eV_per_angstrom: float
    g_q0_rms_eV_per_angstrom: float
    retarded_loop_real: list[list[float]]
    retarded_loop_imag: list[list[float]]
    kernel_eV_per_angstrom: list[list[list[float]]]
    kernel_imaginary_residual_eV_per_angstrom: float
    acoustic_sum_kernel_eV_per_angstrom: list[list[float]]
    acoustic_sum_max_eV_per_angstrom: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _relative_norm(
    first: NDArray[np.complex128], second: NDArray[np.complex128]
) -> float:
    denominator = max(
        float(np.linalg.norm(second)),
        float(np.finfo(np.float64).tiny),
    )
    return float(np.linalg.norm(first - second) / denominator)


def _root_setup(
    *,
    epr_path: str | Path,
    win_path: str | Path,
    amn_path: str | Path,
    eig_path: str | Path,
    u_matrix_path: str | Path,
    u_dis_matrix_path: str | Path,
    u_dis_layout: UDisLayout,
    magnetic_projection_specification: str,
    atomic_spin_order: str,
    magnetization_direction: object,
    ep_energy_unit: str,
    ep_displacement_unit: str,
    divide_by_ws_degeneracy: bool,
    finite_difference_step: float,
    mpi_size: int,
) -> tuple[list[tuple[np.ndarray, ...]], dict[str, Any]]:
    header = read_wannier_amn_header(amn_path)
    indices = parse_projection_indices(
        magnetic_projection_specification,
        header.num_projections,
    )
    basis = load_native_spinor_basis(
        epr_path,
        win_path=win_path,
        amn_path=amn_path,
        eig_path=eig_path,
        u_matrix_path=u_matrix_path,
        u_dis_matrix_path=u_dis_matrix_path,
        u_dis_layout=u_dis_layout,
        projection_indices=indices,
    )
    projected = build_projected_collinear_exchange(
        basis.hamiltonian_eV,
        basis.selected_overlap,
        basis.minus_k_index,
        magnetization_direction=magnetization_direction,
        atomic_spin_order=atomic_spin_order,
    )
    g_q0 = reconstruct_epr_g_at_q(
        epr_path,
        0,
        energy_unit=ep_energy_unit,
        displacement_unit=ep_displacement_unit,
        divide_by_ws_degeneracy=divide_by_ws_degeneracy,
    )
    if not np.array_equal(g_q0.kpoints, basis.kpoints):
        raise ValueError("native Hamiltonian and EPR g(k,q=0) k-point orders differ")
    if not np.allclose(g_q0.qpoint, 0.0, atol=0.0, rtol=0.0):
        raise ValueError("native q index zero is not Gamma")

    delta = float(finite_difference_step)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("finite_difference_step must be finite and positive")
    rotated_plus = rotate_projected_exchange(
        projected,
        projected.local_frame[0],
        delta,
    )
    rotated_minus = rotate_projected_exchange(
        projected,
        projected.local_frame[0],
        -delta,
    )
    numerical_vertex = (rotated_plus - rotated_minus) / (2.0 * delta)
    analytic_vertex = -projected.transverse_vertices[:, 1]
    finite_difference_residual = _relative_norm(numerical_vertex, analytic_vertex)

    singular = projected.singular_values
    minimum_singular = singular[:, -1]
    condition = singular[:, 0] / np.maximum(
        minimum_singular,
        np.finfo(np.float64).tiny,
    )
    transverse_norms = np.linalg.norm(
        projected.transverse_vertices.reshape(
            projected.transverse_vertices.shape[0],
            2,
            -1,
        ),
        axis=2,
    )
    longitudinal_norms = np.linalg.norm(
        projected.longitudinal_vertex.reshape(
            projected.longitudinal_vertex.shape[0],
            -1,
        ),
        axis=1,
    )
    raw_rotation_vertices = projected.raw_rotation_vertices
    raw_rotation_norms = np.linalg.norm(
        raw_rotation_vertices.reshape(
            raw_rotation_vertices.shape[0],
            3,
            -1,
        ),
        axis=2,
    )
    transverse_scale = max(
        float(np.sqrt(np.mean(transverse_norms**2))),
        float(np.finfo(np.float64).tiny),
    )
    raw_transverse_scale = max(
        float(np.sqrt(np.mean(raw_rotation_norms[:, :2] ** 2))),
        float(np.finfo(np.float64).tiny),
    )
    noncollinear_fraction = projected.noncollinear_fraction_per_k
    g_rms = float(np.sqrt(np.mean(np.abs(g_q0.values) ** 2)))
    metadata: dict[str, Any] = {
        "num_wann": basis.num_wann,
        "magnetic_dimension": len(indices),
        "indices": indices,
        "kmesh": basis.kmesh,
        "qmesh": basis.qmesh,
        "qpoint": tuple(float(value) for value in g_q0.qpoint),
        "projection_min": float(np.min(minimum_singular)),
        "projection_median": float(np.median(minimum_singular)),
        "projection_condition": float(np.max(condition)),
        "noncollinear_median": float(np.median(noncollinear_fraction)),
        "noncollinear_max": float(np.max(noncollinear_fraction)),
        "raw_longitudinal_max": float(
            np.max(raw_rotation_norms[:, 2], initial=0.0)
        ),
        "raw_longitudinal_ratio": float(
            float(np.max(raw_rotation_norms[:, 2], initial=0.0))
            / raw_transverse_scale
        ),
        "longitudinal_max": float(np.max(longitudinal_norms, initial=0.0)),
        "longitudinal_ratio": float(
            float(np.max(longitudinal_norms, initial=0.0)) / transverse_scale
        ),
        "finite_difference_residual": finite_difference_residual,
        "g_hermiticity": float(g_q0.q0_hermiticity_residual or 0.0),
        "g_rms": g_rms,
        "nat": int(g_q0.pert_atom[-1]) + 1,
    }

    nk = basis.kpoints.shape[0]
    rank_indices = [
        np.arange(rank, nk, mpi_size, dtype=np.int64) for rank in range(mpi_size)
    ]
    payloads: list[tuple[np.ndarray, ...]] = []
    for selected in rank_indices:
        payloads.append(
            (
                np.asarray(basis.hamiltonian_eV[selected], dtype=np.complex128),
                np.asarray(
                    projected.transverse_vertices[selected],
                    dtype=np.complex128,
                ),
                np.asarray(g_q0.values[selected], dtype=np.complex128),
                np.full(selected.size, 1.0 / nk, dtype=np.float64),
            )
        )
    return payloads, metadata


def run_native_q0_test(
    *,
    epr_path: str | Path,
    win_path: str | Path,
    amn_path: str | Path,
    eig_path: str | Path,
    u_matrix_path: str | Path,
    u_dis_matrix_path: str | Path,
    u_dis_layout: UDisLayout,
    magnetic_projection_specification: str,
    atomic_spin_order: str,
    magnetization_direction: object,
    fermi_energy_eV: float,
    energy_min_eV: float,
    energy_max_eV: float,
    energy_points: int,
    eta_eV: float,
    temperature_K: float,
    ep_energy_unit: str,
    ep_displacement_unit: str,
    divide_by_ws_degeneracy: bool = False,
    finite_difference_step: float = 1.0e-6,
    perturbation_chunk: int | None = None,
    mpi: MPIContext | None = None,
) -> NativeQ0Result | None:
    """Run the bubble-only Gamma test with full Green functions and d-like torque."""

    context = MPIContext.discover() if mpi is None else mpi
    scalar_inputs = {
        "fermi_energy_eV": float(fermi_energy_eV),
        "energy_min_eV": float(energy_min_eV),
        "energy_max_eV": float(energy_max_eV),
        "eta_eV": float(eta_eV),
        "temperature_K": float(temperature_K),
    }
    if any(not np.isfinite(value) for value in scalar_inputs.values()):
        raise ValueError("native q=0 integration inputs must be finite")
    if scalar_inputs["eta_eV"] <= 0.0:
        raise ValueError("eta_eV must be positive")
    if scalar_inputs["temperature_K"] < 0.0:
        raise ValueError("temperature_K must be non-negative")
    setup_error: tuple[str, str] | None = None
    payloads: list[tuple[np.ndarray, ...]] | None = None
    metadata: dict[str, Any] | None = None
    if context.is_root:
        try:
            payloads, metadata = _root_setup(
                epr_path=epr_path,
                win_path=win_path,
                amn_path=amn_path,
                eig_path=eig_path,
                u_matrix_path=u_matrix_path,
                u_dis_matrix_path=u_dis_matrix_path,
                u_dis_layout=u_dis_layout,
                magnetic_projection_specification=(magnetic_projection_specification),
                atomic_spin_order=atomic_spin_order,
                magnetization_direction=magnetization_direction,
                ep_energy_unit=ep_energy_unit,
                ep_displacement_unit=ep_displacement_unit,
                divide_by_ws_degeneracy=divide_by_ws_degeneracy,
                finite_difference_step=finite_difference_step,
                mpi_size=context.size,
            )
        except Exception as exc:  # noqa: BLE001 - release MPI peers
            setup_error = (type(exc).__name__, str(exc))
    setup_error = context.bcast(setup_error, root=0)
    if setup_error is not None:
        error_type, message = setup_error
        raise RuntimeError(f"native q=0 setup failed ({error_type}): {message}")
    metadata = context.bcast(metadata, root=0)
    local_h, local_vertices, local_g, local_weights = context.scatter(
        payloads,
        root=0,
    )

    integrator = RealAxisIntegrator.gauss_legendre(
        float(energy_min_eV),
        float(energy_max_eV),
        int(energy_points),
        chemical_potential_eV=float(fermi_energy_eV),
        temperature_K=float(temperature_K),
    )
    local_error: tuple[int, str, str] | None = None
    if local_h.shape[0] == 0:
        if metadata is None:
            raise RuntimeError("native q=0 metadata was not broadcast")
        local_loop = np.zeros((2, 3 * int(metadata["nat"])), dtype=np.complex128)
    else:
        try:
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
        raise RuntimeError(f"native q=0 MPI integration failed: {rendered}")
    gathered = context.gather(local_loop, root=0)
    finalization_error: tuple[str, str] | None = None
    root_result: NativeQ0Result | None = None
    if context.is_root:
        try:
            if gathered is None or metadata is None:
                raise RuntimeError("native q=0 root received no integration payload")
            retarded_loop = np.sum(np.stack(gathered), axis=0)
            kernel = finalize_q_pair(retarded_loop, retarded_loop)
            nat = int(metadata["nat"])
            kernel_shaped = kernel.reshape(2, nat, 3)
            acoustic_sum = np.sum(kernel_shaped, axis=1)
            imaginary_residual = float(np.max(np.abs(kernel_shaped.imag), initial=0.0))
            root_result = NativeQ0Result(
                approximation=(
                    "bubble-only; full spinor H/g Green functions; "
                    "input-selected collinear magnetic-subspace exchange"
                ),
                green_function_dimension=int(metadata["num_wann"]),
                magnetic_subspace_dimension=int(metadata["magnetic_dimension"]),
                magnetic_projection_indices_one_based=tuple(
                    int(index) + 1 for index in metadata["indices"]
                ),
                kmesh=(
                    int(metadata["kmesh"][0]),
                    int(metadata["kmesh"][1]),
                    int(metadata["kmesh"][2]),
                ),
                qmesh=(
                    int(metadata["qmesh"][0]),
                    int(metadata["qmesh"][1]),
                    int(metadata["qmesh"][2]),
                ),
                qpoint_reduced=(
                    float(metadata["qpoint"][0]),
                    float(metadata["qpoint"][1]),
                    float(metadata["qpoint"][2]),
                ),
                mpi_size=context.size,
                fermi_energy_eV=float(fermi_energy_eV),
                integration_energy_range_eV=(
                    float(energy_min_eV),
                    float(energy_max_eV),
                ),
                integration_energy_points=int(energy_points),
                eta_eV=float(eta_eV),
                temperature_K=float(temperature_K),
                ep_input_energy_unit=str(ep_energy_unit),
                ep_input_displacement_unit=str(ep_displacement_unit),
                output_units="eV/angstrom",
                direct_term_enabled=False,
                projection_min_singular_value=float(metadata["projection_min"]),
                projection_median_min_singular_value=float(
                    metadata["projection_median"]
                ),
                projection_max_condition_number=float(metadata["projection_condition"]),
                raw_exchange_noncollinear_fraction_median=float(
                    metadata["noncollinear_median"]
                ),
                raw_exchange_noncollinear_fraction_max=float(
                    metadata["noncollinear_max"]
                ),
                raw_longitudinal_vertex_max_frobenius_eV=float(
                    metadata["raw_longitudinal_max"]
                ),
                raw_longitudinal_to_transverse_vertex_ratio=float(
                    metadata["raw_longitudinal_ratio"]
                ),
                longitudinal_vertex_max_frobenius_eV=float(
                    metadata["longitudinal_max"]
                ),
                longitudinal_to_transverse_vertex_ratio=float(
                    metadata["longitudinal_ratio"]
                ),
                central_difference_relative_residual=float(
                    metadata["finite_difference_residual"]
                ),
                g_q0_hermiticity_residual_eV_per_angstrom=float(
                    metadata["g_hermiticity"]
                ),
                g_q0_rms_eV_per_angstrom=float(metadata["g_rms"]),
                retarded_loop_real=retarded_loop.real.tolist(),
                retarded_loop_imag=retarded_loop.imag.tolist(),
                kernel_eV_per_angstrom=kernel_shaped.real.tolist(),
                kernel_imaginary_residual_eV_per_angstrom=imaginary_residual,
                acoustic_sum_kernel_eV_per_angstrom=acoustic_sum.real.tolist(),
                acoustic_sum_max_eV_per_angstrom=float(
                    np.max(np.abs(acoustic_sum), initial=0.0)
                ),
            )
        except Exception as exc:  # noqa: BLE001 - release MPI peers
            finalization_error = (type(exc).__name__, str(exc))
    finalization_error = context.bcast(finalization_error, root=0)
    if finalization_error is not None:
        error_type, message = finalization_error
        raise RuntimeError(f"native q=0 finalization failed ({error_type}): {message}")
    context.barrier()
    return root_result


__all__ = ["NativeQ0Result", "run_native_q0_test"]
