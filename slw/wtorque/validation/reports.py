"""Deterministic physics validation report generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from slw.wtorque.config import RunConfig, SiteProjection
from slw.wtorque.gauge.kq_map import build_kq_map
from slw.wtorque.io.exchange import exchange_at_k_and_kq, load_exchange_field
from slw.wtorque.io.hdf5 import RestartableHDF5
from slw.wtorque.io.spin import load_magnetic_data
from slw.wtorque.io.wannier import load_spinor_wannier
from slw.wtorque.model.local_projection import localize_exchange
from slw.wtorque.projection.phonon import acoustic_translation_residual
from slw.wtorque.provenance import build_run_manifest
from slw.wtorque.torque.vertices import (
    finite_rotated_exchange,
    rotation_vertex,
    transverse_vertices,
)
from slw.wtorque.validation.algebra import hermiticity_residual


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    residual: float | None
    tolerance: float | None
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    gates: tuple[GateResult, ...]
    warnings: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def to_markdown(self) -> str:
        lines = ["# wtorque physics validation", "", f"Overall: {'PASS' if self.valid else 'FAIL'}", ""]
        lines.append("| Gate | Status | Residual | Tolerance | Detail |")
        lines.append("|---|---:|---:|---:|---|")
        for gate in self.gates:
            residual = "" if gate.residual is None else f"{gate.residual:.6e}"
            tolerance = "" if gate.tolerance is None else f"{gate.tolerance:.6e}"
            lines.append(
                f"| {gate.name} | {gate.status} | {residual} | {tolerance} | {gate.detail} |"
            )
        if self.warnings:
            lines.extend(("", "## Warnings", ""))
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines) + "\n"


def _gate(name: str, residual: float, tolerance: float, detail: str) -> GateResult:
    return GateResult(name, "pass" if residual <= tolerance else "fail", residual, tolerance, detail)


def validate_run(
    config: RunConfig,
    *,
    tolerance: float = 1.0e-8,
    finite_difference_step: float = 1.0e-6,
    markdown_path: str | Path | None = None,
) -> ValidationReport:
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
    gates: list[GateResult] = []
    warnings: list[str] = []
    hk = model.hamiltonian_batch(model.kpoints)
    gates.append(_gate("H Hermiticity", hermiticity_residual(hk), tolerance, "full spinor propagation"))
    if model.onsite_soc_matrix is not None:
        soc = model.onsite_soc_matrix
        gates.append(
            _gate(
                "onsite SOC Hermiticity",
                hermiticity_residual(soc),
                tolerance,
                "input-resolved lambda L.S",
            )
        )
        i_sigma_y = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
        sewing = np.kron(np.eye(model.norb, dtype=np.complex128), i_sigma_y)
        time_reversed_soc = sewing @ soc.conj() @ sewing.conj().T
        gates.append(
            _gate(
                "onsite SOC time reversal",
                float(np.max(np.abs(time_reversed_soc - soc), initial=0.0)),
                tolerance,
                "SOC must remain in H_TRS",
            )
        )
        base_model = load_spinor_wannier(
            config.electrons.file,
            bloch_gauge=config.electrons.bloch_gauge,
            spin_order=config.electrons.spin_order,
            onsite_soc=None,
        )
        base_exchange = load_exchange_field(
            config.electrons.file,
            base_model,
            route=config.electrons.exchange_extraction,
            spin_order=config.electrons.spin_order,
        )
        if exchange.h_xc_r is not None and base_exchange.h_xc_r is not None:
            isolation = float(
                np.max(np.abs(exchange.h_xc_r - base_exchange.h_xc_r), initial=0.0)
            )
        elif exchange.h_xc_k is not None and base_exchange.h_xc_k is not None:
            isolation = float(
                np.max(np.abs(exchange.h_xc_k - base_exchange.h_xc_k), initial=0.0)
            )
        else:
            isolation = float("inf")
        gates.append(
            _gate(
                "SOC isolation from exchange",
                isolation,
                tolerance,
                "scaling SOC must not change H_XC or torque vertices",
            )
        )
    gates.append(
        _gate(
            "exchange reconstruction",
            exchange.reconstruction_residual,
            tolerance,
            config.electrons.exchange_extraction.value,
        )
    )
    zero_mapping = build_kq_map(model.kpoints, np.zeros(3))
    hxc, _ = exchange_at_k_and_kq(exchange, model, np.zeros(3), zero_mapping)
    local, residual = localize_exchange(
        hxc,
        magnetic.subspace.projectors,
        strategy=config.magnetic_subspace.site_projection,
    )
    closure = float(np.max(np.abs(hxc - local.sum(axis=0) - residual), initial=0.0))
    gates.append(_gate("local partition closure", closure, tolerance, "WT-L01"))
    local_hermiticity = hermiticity_residual(local)
    gates.append(_gate("local exchange Hermiticity", local_hermiticity, tolerance, "exchange only"))
    vertex_residual = 0.0
    longitudinal = 0.0
    for ell, frame in enumerate(magnetic.local_frames):
        analytic = transverse_vertices(local[ell], frame)
        for axis_index, rotation_axis in enumerate((frame[1], frame[0])):
            sign = 1.0 if axis_index == 0 else -1.0
            numeric = sign * (
                finite_rotated_exchange(local[ell], rotation_axis, finite_difference_step)
                - finite_rotated_exchange(local[ell], rotation_axis, -finite_difference_step)
            ) / (2.0 * finite_difference_step)
            vertex_residual = max(
                vertex_residual,
                float(np.max(np.abs(numeric - analytic[axis_index]), initial=0.0)),
            )
        longitudinal = max(
            longitudinal,
            float(np.max(np.abs(rotation_vertex(local[ell], frame[2])), initial=0.0)),
        )
    gates.append(_gate("torque central difference", vertex_residual, 10 * tolerance, "WT-T02/WT-T03"))
    gates.append(_gate("longitudinal rotation", longitudinal, tolerance, "must vanish"))
    onsite, _ = localize_exchange(
        hxc,
        magnetic.subspace.projectors,
        strategy=SiteProjection.ONSITE_ONLY,
    )
    sensitivity = float(np.linalg.norm(local - onsite) / max(np.linalg.norm(local), np.finfo(float).tiny))
    if sensitivity > 0.1:
        warnings.append(
            f"local_partition versus onsite_only relative spread is {sensitivity:.3e}; audit the magnetic subspace"
        )

    qpoints: np.ndarray
    with h5py.File(config.output.file, "r") as output:
        qpoints = np.asarray(output["qpoints"][...], dtype=np.float64)
        if "kernel/K_pi_u" in output:
            kernels = np.asarray(output["kernel/K_pi_u"][...], dtype=np.complex128)
        elif "kernel/V_pi_ph" in output:
            kernels = np.asarray(output["kernel/V_pi_ph"][...], dtype=np.complex128)
        else:
            raise KeyError("output has neither K_pi_u nor V_pi_ph")
        qpair_residual = 0.0
        for index, q in enumerate(qpoints):
            delta = qpoints + q
            delta -= np.rint(delta)
            partner = int(np.argmin(np.max(np.abs(delta), axis=1)))
            qpair_residual = max(
                qpair_residual,
                float(np.max(np.abs(kernels[partner] - kernels[index].conj()), initial=0.0)),
            )
        gates.append(_gate("q conjugation", qpair_residual, tolerance, "WT-S01"))
        gamma_indices = np.flatnonzero(np.max(np.abs(qpoints - np.rint(qpoints)), axis=1) < tolerance)
        if gamma_indices.size and "kernel/K_pi_u" in output:
            acoustic = max(
                acoustic_translation_residual(kernels[index])[0]
                for index in gamma_indices
            )
            gates.append(_gate("rigid translation", acoustic, tolerance, "WT-S03 before mode normalization"))

    with RestartableHDF5(
        config.output.file,
        qpoints,
        build_run_manifest(config),
        resume=True,
    ) as writer:
        for index in range(qpoints.shape[0]):
            writer.verify_complete_q(index)
    report = ValidationReport(
        valid=all(gate.status == "pass" for gate in gates),
        gates=tuple(gates),
        warnings=tuple(warnings),
    )
    with h5py.File(config.output.file, "r+") as output:
        group = output.require_group("validation")
        if "report_json" in group:
            del group["report_json"]
        group.create_dataset("report_json", data=report.to_json(), dtype=h5py.string_dtype("utf-8"))
    if markdown_path is not None:
        Path(markdown_path).write_text(report.to_markdown(), encoding="utf-8")
    return report


__all__ = ["GateResult", "ValidationReport", "validate_run"]
