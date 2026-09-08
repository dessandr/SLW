"""MPI q-pair pipeline for bubble and fixed-projector direct kernels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.config import BlochGauge, Normalization, ProjectorPolicy, RunConfig
from slw.wtorque.gauge.atomic_gauge import unwrap_final_state_vertex
from slw.wtorque.gauge.kq_map import build_kq_map
from slw.wtorque.green.real_axis import RealAxisIntegrator
from slw.wtorque.io.dfpt import HDF5DFPTProvider, expand_cartesian_perturbations
from slw.wtorque.io.exchange import (
    ExchangeFieldData,
    exchange_at_k_and_kq,
    load_exchange_field,
)
from slw.wtorque.io.hdf5 import RestartableHDF5
from slw.wtorque.io.spin import MagneticData, load_magnetic_data
from slw.wtorque.io.wannier import load_spinor_wannier
from slw.wtorque.model.spinor_wannier import SpinorWannierModel
from slw.wtorque.parallel.mpi import MPIContext
from slw.wtorque.parallel.scheduler import (
    QPointPair,
    build_q_pair_schedule,
    partition_pairs,
)
from slw.wtorque.provenance import build_run_manifest
from slw.wtorque.torque.direct_vertex import (
    finite_q_direct_vertices,
    retarded_direct_loop,
)
from slw.wtorque.torque.kernel import retarded_bubble_loop, tune_perturbation_chunk
from slw.wtorque.torque.qpair import finalize_q_pair, q_conjugation_residual
from slw.wtorque.torque.vertices import finite_q_vertices


@dataclass(frozen=True)
class KernelRunResult:
    output_file: Path
    computed_q_indices: tuple[int, ...]
    mpi_size: int


@dataclass(frozen=True)
class _Inputs:
    model: SpinorWannierModel
    exchange: ExchangeFieldData
    magnetic: MagneticData
    integrator: RealAxisIntegrator


def _load_inputs(config: RunConfig) -> _Inputs:
    model = load_spinor_wannier(
        config.electrons.file,
        bloch_gauge=config.electrons.bloch_gauge,
        spin_order=config.electrons.spin_order,
        onsite_soc=config.electrons.onsite_soc,
    )
    exchange = load_exchange_field(
        config.electrons.file,
        model,
        route=config.electrons.exchange_extraction,
        spin_order=config.electrons.spin_order,
    )
    magnetic = load_magnetic_data(
        config.electrons.file,
        config.magnetic_subspace,
        norb=model.norb,
        orbital_labels=model.orbital_labels,
    )
    integration = config.integration
    if integration.backend != "real_axis":
        raise NotImplementedError(
            "the contour interface is reserved for an upstream integrator; use real_axis for this build"
        )
    energy_min = integration.energy_min_eV
    energy_max = integration.energy_max_eV
    energy_points = integration.energy_points
    if energy_min is None or energy_max is None or energy_points is None:
        raise ValueError(
            "compute-kernel requires integration.energy_min_eV, energy_max_eV, and energy_points"
        )
    integrator = RealAxisIntegrator.gauss_legendre(
        energy_min,
        energy_max,
        energy_points,
        chemical_potential_eV=model.fermi_energy,
        temperature_K=integration.temperature_K,
    )
    return _Inputs(model, exchange, magnetic, integrator)


def _canonical_perturbations(
    values: NDArray[np.complex128],
    *,
    config: RunConfig,
    model: SpinorWannierModel,
    g_wrap: NDArray[np.int64],
) -> NDArray[np.complex128]:
    if config.dfpt.final_state_representation == "unwrapped":
        return values
    if config.electrons.bloch_gauge is BlochGauge.CELL_PERIODIC:
        return values
    return np.asarray(
        unwrap_final_state_vertex(values, g_wrap, model.orbital_centers),
        dtype=np.complex128,
    )


def _vertex_centers(model: SpinorWannierModel) -> NDArray[np.float64]:
    # Cell-periodic Bloch vectors do not carry the atomic-position phases.
    if model.bloch_gauge is BlochGauge.CELL_PERIODIC:
        return np.zeros_like(model.orbital_centers)
    return model.orbital_centers


def _retarded_loop_for_q(
    iq: int,
    *,
    config: RunConfig,
    inputs: _Inputs,
    dfpt: HDF5DFPTProvider,
    perturbation_chunk: int,
) -> NDArray[np.complex128]:
    model = inputs.model
    q = dfpt.qpoints[int(iq)]
    mapping = build_kq_map(model.kpoints, q)
    h_k = model.hamiltonian_batch(model.kpoints)
    h_kq = model.hamiltonian_batch(model.kpoints + q)
    hxc_k, hxc_kq = exchange_at_k_and_kq(inputs.exchange, model, q, mapping)
    # T(-q) maps the k+q source basis back to the k final basis.
    vertices = finite_q_vertices(
        hxc_kq,
        hxc_k,
        orbital_masks=inputs.magnetic.subspace.orbital_masks,
        local_frames=inputs.magnetic.local_frames,
        q_red=-q,
        orbital_centers=_vertex_centers(model),
        magnetic_site_positions=inputs.magnetic.site_positions,
        coordinate_type=config.magnetic_subspace.spin_coordinate.value,
        site_projection=config.magnetic_subspace.site_projection,
    )
    vertices = vertices.reshape(
        model.kpoints.shape[0],
        2 * inputs.magnetic.subspace.projectors.shape[0],
        model.nw,
        model.nw,
    )
    perturbations = _canonical_perturbations(
        dfpt.g(iq),
        config=config,
        model=model,
        g_wrap=mapping.G_wrap,
    )
    if perturbations.shape[0] != model.kpoints.shape[0]:
        raise ValueError(
            f"DFPT q index {iq} has nk={perturbations.shape[0]}, expected {model.kpoints.shape[0]}"
        )
    return retarded_bubble_loop(
        h_k,
        h_kq,
        vertices,
        perturbations,
        model.weights,
        inputs.integrator,
        eta_eV=config.integration.eta_eV,
        perturbation_chunk=perturbation_chunk,
    )


def _direct_retarded_loop_for_q(
    iq: int,
    *,
    config: RunConfig,
    inputs: _Inputs,
    dfpt: HDF5DFPTProvider,
    perturbation_chunk: int,
) -> NDArray[np.complex128]:
    model = inputs.model
    q = dfpt.qpoints[int(iq)]
    mapping = build_kq_map(model.kpoints, q)
    g_xc = dfpt.g_xc(iq)
    if g_xc is None:
        raise ValueError(
            f"direct term requires the declared exchange-resolved g_XC dataset "
            f"{config.dfpt.g_xc_dataset!r} at q index {iq}"
        )
    g_xc = _canonical_perturbations(
        g_xc, config=config, model=model, g_wrap=mapping.G_wrap
    )
    if g_xc.shape[0] != model.kpoints.shape[0]:
        raise ValueError("g_XC k-point count does not match the electronic model")
    nmag = inputs.magnetic.subspace.projectors.shape[0]
    result = np.zeros((2 * nmag, g_xc.shape[1]), dtype=np.complex128)
    h_k = model.hamiltonian_batch(model.kpoints)
    # The mixed insertion carries both torque and displacement labels. Build
    # only a perturbation chunk so its large nk*nt*np*nw*nw tensor is bounded.
    for start in range(0, g_xc.shape[1], perturbation_chunk):
        stop = min(start + perturbation_chunk, g_xc.shape[1])
        direct = finite_q_direct_vertices(
            g_xc[:, start:stop],
            kpoints=model.kpoints,
            q_red=q,
            orbital_masks=inputs.magnetic.subspace.orbital_masks,
            local_frames=inputs.magnetic.local_frames,
            orbital_centers=_vertex_centers(model),
            magnetic_site_positions=inputs.magnetic.site_positions,
            coordinate_type=config.magnetic_subspace.spin_coordinate.value,
            projector_policy=config.kernel.projector_policy.value,
            site_projection=config.magnetic_subspace.site_projection,
        )
        result[:, start:stop] = retarded_direct_loop(
            h_k,
            direct.reshape(model.kpoints.shape[0], 2 * nmag, stop - start, model.nw, model.nw),
            model.weights,
            inputs.integrator,
            eta_eV=config.integration.eta_eV,
            perturbation_chunk=perturbation_chunk,
        )
    return result


def _payload(
    iq: int,
    loop: NDArray[np.complex128],
    kernel: NDArray[np.complex128],
    *,
    config: RunConfig,
    inputs: _Inputs,
    dfpt: HDF5DFPTProvider,
    conjugation_residual: tuple[float, float],
    direct_loop: NDArray[np.complex128] | None = None,
    direct_kernel: NDArray[np.complex128] | None = None,
) -> dict[str, object]:
    nmag = inputs.magnetic.subspace.projectors.shape[0]
    if direct_loop is None:
        direct_loop = np.zeros_like(loop)
    if direct_kernel is None:
        direct_kernel = np.zeros_like(kernel)
    total_loop = loop + direct_loop
    total_kernel = kernel + direct_kernel
    result: dict[str, object] = {
        "kernel/A_retarded": total_loop.reshape(nmag, 2, loop.shape[-1]),
        "validation/q_conjugation_absolute": np.float64(conjugation_residual[0]),
        "validation/q_conjugation_relative": np.float64(conjugation_residual[1]),
        "validation/direct_term_enabled": np.bool_(config.kernel.include_direct_vertex),
    }
    for name, loop_component, kernel_component in (
        ("bubble", loop, kernel),
        ("direct", direct_loop, direct_kernel),
        ("total", total_loop, total_kernel),
    ):
        result[f"kernel/A_retarded_{name}"] = loop_component.reshape(nmag, 2, loop.shape[-1])
        kernel_shaped = kernel_component.reshape(nmag, 2, kernel.shape[-1])
        if config.dfpt.normalization is Normalization.PHONON_ZERO_POINT_MODE:
            key = "kernel/V_pi_ph"
            projected = kernel_shaped
        else:
            if dfpt.pert_atom is None or dfpt.pert_cart is None:
                raise ValueError("Cartesian kernel lacks perturbation metadata")
            key = "kernel/K_pi_u"
            projected = expand_cartesian_perturbations(
                kernel_shaped, dfpt.pert_atom, dfpt.pert_cart
            )
        result[f"{key}_{name}"] = projected
        if name == "total":
            result[key] = projected
    return result


def _compute_pair(
    pair: QPointPair,
    *,
    config: RunConfig,
    inputs: _Inputs,
    dfpt: HDF5DFPTProvider,
    perturbation_chunk: int,
) -> dict[int, dict[str, object]]:
    a_positive = _retarded_loop_for_q(
        pair.positive,
        config=config,
        inputs=inputs,
        dfpt=dfpt,
        perturbation_chunk=perturbation_chunk,
    )
    if pair.self_inverse:
        a_negative = a_positive
    else:
        a_negative = _retarded_loop_for_q(
            pair.negative,
            config=config,
            inputs=inputs,
            dfpt=dfpt,
            perturbation_chunk=perturbation_chunk,
        )
    k_positive = finalize_q_pair(a_positive, a_negative)
    k_negative = k_positive.conj()
    direct_positive = np.zeros_like(a_positive)
    direct_negative = np.zeros_like(a_negative)
    if config.kernel.include_direct_vertex:
        direct_positive = _direct_retarded_loop_for_q(
            pair.positive,
            config=config,
            inputs=inputs,
            dfpt=dfpt,
            perturbation_chunk=perturbation_chunk,
        )
        direct_negative = direct_positive if pair.self_inverse else _direct_retarded_loop_for_q(
            pair.negative,
            config=config,
            inputs=inputs,
            dfpt=dfpt,
            perturbation_chunk=perturbation_chunk,
        )
    direct_k_positive = finalize_q_pair(direct_positive, direct_negative)
    direct_k_negative = direct_k_positive.conj()
    residual = q_conjugation_residual(
        k_positive + direct_k_positive, k_negative + direct_k_negative
    )
    result = {
        pair.positive: _payload(
            pair.positive,
            a_positive,
            k_positive,
            config=config,
            inputs=inputs,
            dfpt=dfpt,
            conjugation_residual=residual,
            direct_loop=direct_positive,
            direct_kernel=direct_k_positive,
        )
    }
    if not pair.self_inverse:
        result[pair.negative] = _payload(
            pair.negative,
            a_negative,
            k_negative,
            config=config,
            inputs=inputs,
            dfpt=dfpt,
            conjugation_residual=residual,
            direct_loop=direct_negative,
            direct_kernel=direct_k_negative,
        )
    return result


def run_compute_kernel(
    config: RunConfig,
    *,
    mpi: MPIContext | None = None,
    memory_limit_bytes: int | None = None,
) -> KernelRunResult:
    """Compute restartable q-paired kernels with deterministic MPI ownership."""

    context = MPIContext.discover() if mpi is None else mpi
    if config.performance.backend == "cupy":
        raise NotImplementedError(
            "CuPy remains an optional feature gate; this validated build uses the NumPy backend"
        )
    if config.kernel.include_direct_vertex and config.kernel.projector_policy is ProjectorPolicy.MOVING:
        raise NotImplementedError(
            "moving-projector direct terms require projector derivatives and are not implemented"
        )
    inputs = _load_inputs(config)
    manifest = build_run_manifest(config)
    writer: RestartableHDF5 | None = None
    with HDF5DFPTProvider(
        config.dfpt.file,
        normalization=config.dfpt.normalization,
        spinor_lift=config.dfpt.spinor_lift,
        spin_order=config.electrons.spin_order,
        norb=inputs.model.norb,
        g_xc_dataset=config.dfpt.g_xc_dataset,
    ) as dfpt:
        sample = dfpt.g(0)
        if context.is_root:
            try:
                writer = RestartableHDF5(
                    config.output.file,
                    dfpt.qpoints,
                    manifest,
                    resume=config.output.resume,
                )
                pending = set(writer.pending_indices())
                initialization_error = None
            except Exception as exc:  # noqa: BLE001 - release peer ranks
                initialization_error = (type(exc).__name__, str(exc))
                pending = set()
        else:
            pending = set()
            initialization_error = None
        initialization_error = context.bcast(initialization_error, root=0)
        if initialization_error is not None:
            error_type, message = initialization_error
            raise RuntimeError(f"root output initialization failed ({error_type}): {message}")
        pending = set(context.bcast(tuple(sorted(pending)), root=0))
        schedule = build_q_pair_schedule(dfpt.qpoints)
        needed = tuple(
            pair
            for pair in schedule
            if pair.positive in pending or pair.negative in pending
        )
        owned = partition_pairs(needed, context.rank, context.size)
        resolved_memory = (
            config.performance.memory_limit_mb * 1024**2
            if memory_limit_bytes is None
            else int(memory_limit_bytes)
        )
        tuned_chunk = tune_perturbation_chunk(
            inputs.model.nw,
            inputs.model.kpoints.shape[0],
            sample.shape[1],
            resolved_memory,
        )
        chunk = (
            tuned_chunk
            if config.performance.perturbation_chunk is None
            else min(config.performance.perturbation_chunk, tuned_chunk)
        )
        if config.kernel.include_direct_vertex:
            matrix_bytes = inputs.model.kpoints.shape[0] * inputs.model.nw**2 * 16
            nmag = inputs.magnetic.subspace.projectors.shape[0]
            fixed_bytes = (4 * sample.shape[1] + 8) * matrix_bytes
            bytes_per_perturbation = (2 * nmag + 6) * matrix_bytes
            direct_chunk = (resolved_memory - fixed_bytes) // bytes_per_perturbation
            if direct_chunk < 1:
                if writer is not None:
                    writer.close()
                raise MemoryError("memory budget cannot hold one direct-vertex perturbation chunk")
            chunk = min(chunk, direct_chunk)
        local: dict[int, dict[str, object]] = {}
        local_error: tuple[int, str, str] | None = None
        try:
            for pair in owned:
                local.update(
                    _compute_pair(
                        pair,
                        config=config,
                        inputs=inputs,
                        dfpt=dfpt,
                        perturbation_chunk=chunk,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - exchange before the next collective
            local_error = (context.rank, type(exc).__name__, str(exc))
        failures = tuple(error for error in context.allgather(local_error) if error is not None)
        if failures:
            if writer is not None:
                writer.close()
            details = "; ".join(
                f"rank {rank} {error_type}: {message}"
                for rank, error_type, message in failures
            )
            raise RuntimeError(f"collective q-pair calculation failed: {details}")
        gathered = context.gather(local, root=0)
        computed: tuple[int, ...] = ()
        assembly_error: tuple[str, str] | None = None
        if context.is_root:
            try:
                if writer is None or gathered is None:
                    raise RuntimeError("MPI root did not initialize output collection")
                combined: dict[int, dict[str, object]] = {}
                for item in gathered:
                    overlap = combined.keys() & item.keys()
                    if overlap:
                        raise RuntimeError(f"duplicate MPI q ownership: {sorted(overlap)}")
                    combined.update(item)
                computed = tuple(sorted(index for index in combined if index in pending))
                for index in computed:
                    writer.write_q(index, combined[index])
                writer.materialize()
            except Exception as exc:  # noqa: BLE001 - release non-root ranks
                assembly_error = (type(exc).__name__, str(exc))
            finally:
                if writer is not None:
                    writer.close()
        assembly_error = context.bcast(assembly_error, root=0)
        if assembly_error is not None:
            error_type, message = assembly_error
            raise RuntimeError(f"root q-output assembly failed ({error_type}): {message}")
        computed = tuple(context.bcast(computed, root=0))
        context.barrier()
    return KernelRunResult(config.output.file, computed, context.size)


__all__ = ["KernelRunResult", "run_compute_kernel"]
