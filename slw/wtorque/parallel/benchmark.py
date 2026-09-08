"""Reference-versus-vectorized q-point benchmark without output mutation."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from slw.wtorque.config import RunConfig
from slw.wtorque.gauge.kq_map import build_kq_map
from slw.wtorque.io.dfpt import HDF5DFPTProvider
from slw.wtorque.io.exchange import exchange_at_k_and_kq
from slw.wtorque.pipeline import _canonical_perturbations, _load_inputs, _vertex_centers
from slw.wtorque.torque.kernel import (
    retarded_bubble_loop,
    retarded_bubble_loop_reference,
    tune_perturbation_chunk,
)
from slw.wtorque.torque.vertices import finite_q_vertices


@dataclass(frozen=True)
class BenchmarkResult:
    q_index: int
    reference_seconds: float
    vectorized_seconds: float
    speedup: float
    max_absolute_difference: float
    agrees: bool
    kernel_component: str = "bubble"

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def benchmark_q(
    config: RunConfig,
    q_index: int,
    *,
    repeats: int = 3,
    tolerance: float = 1.0e-10,
) -> BenchmarkResult:
    """Compare bubble contractions; direct-term construction is not timed."""

    if repeats < 1:
        raise ValueError("benchmark repeats must be positive")
    inputs = _load_inputs(config)
    model = inputs.model
    with HDF5DFPTProvider(
        config.dfpt.file,
        normalization=config.dfpt.normalization,
        spinor_lift=config.dfpt.spinor_lift,
        spin_order=config.electrons.spin_order,
        norb=model.norb,
        g_xc_dataset=config.dfpt.g_xc_dataset,
    ) as dfpt:
        if q_index < 0 or q_index >= dfpt.qpoints.shape[0]:
            raise ValueError(f"q_index {q_index} is outside [0,{dfpt.qpoints.shape[0]})")
        q = dfpt.qpoints[q_index]
        mapping = build_kq_map(model.kpoints, q)
        h_k = model.hamiltonian_batch(model.kpoints)
        h_kq = model.hamiltonian_batch(model.kpoints + q)
        hxc_k, hxc_kq = exchange_at_k_and_kq(inputs.exchange, model, q, mapping)
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
        ).reshape(model.kpoints.shape[0], -1, model.nw, model.nw)
        perturbations = _canonical_perturbations(
            dfpt.g(q_index),
            config=config,
            model=model,
            g_wrap=mapping.G_wrap,
        )
    start = time.perf_counter()
    reference = retarded_bubble_loop_reference(
        h_k,
        h_kq,
        vertices,
        perturbations,
        model.weights,
        inputs.integrator,
        eta_eV=config.integration.eta_eV,
    )
    reference_seconds = time.perf_counter() - start
    chunk = tune_perturbation_chunk(
        model.nw,
        model.kpoints.shape[0],
        perturbations.shape[1],
        config.performance.memory_limit_mb * 1024**2,
    )
    if config.performance.perturbation_chunk is not None:
        chunk = min(chunk, config.performance.perturbation_chunk)
    vectorized_seconds = float("inf")
    vectorized = reference
    for _ in range(repeats):
        start = time.perf_counter()
        vectorized = retarded_bubble_loop(
            h_k,
            h_kq,
            vertices,
            perturbations,
            model.weights,
            inputs.integrator,
            eta_eV=config.integration.eta_eV,
            perturbation_chunk=chunk,
        )
        vectorized_seconds = min(vectorized_seconds, time.perf_counter() - start)
    difference = float(abs(vectorized - reference).max(initial=0.0))
    return BenchmarkResult(
        q_index=int(q_index),
        reference_seconds=reference_seconds,
        vectorized_seconds=vectorized_seconds,
        speedup=reference_seconds / max(vectorized_seconds, float.fromhex("0x1p-1022")),
        max_absolute_difference=difference,
        agrees=difference <= tolerance,
    )


__all__ = ["BenchmarkResult", "benchmark_q"]
