"""Restart-safe phonon, magnon, and polaron projection stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from slw.wtorque.config import Normalization, RunConfig
from slw.wtorque.io.hdf5 import RestartableHDF5
from slw.wtorque.io.magnon import HDF5MagnonProvider
from slw.wtorque.io.phonon import HDF5PhononProvider
from slw.wtorque.io.spin import load_magnetic_data
from slw.wtorque.io.wannier import load_spinor_wannier
from slw.wtorque.parallel.mpi import MPIContext
from slw.wtorque.projection.magnon import project_external_magnons
from slw.wtorque.projection.phonon import acoustic_translation_residual, project_phonons
from slw.wtorque.projection.polaron import assemble_rwa
from slw.wtorque.provenance import build_run_manifest


@dataclass(frozen=True)
class StageResult:
    output_file: Path
    updated_q_indices: tuple[int, ...]
    mpi_size: int


def _output_qpoints(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["qpoints"][...], dtype=np.float64)


def _distributed_stage(
    config: RunConfig,
    worker: Callable[[int], dict[str, object]],
    *,
    mpi: MPIContext,
) -> StageResult:
    qpoints = _output_qpoints(config.output.file)
    local_indices = np.arange(mpi.rank, qpoints.shape[0], mpi.size, dtype=np.int64)
    local: dict[int, dict[str, object]] = {}
    local_error: tuple[int, str, str] | None = None
    try:
        local = {int(index): worker(int(index)) for index in local_indices}
    except Exception as exc:  # noqa: BLE001 - exchange before the next collective
        local_error = (mpi.rank, type(exc).__name__, str(exc))
    failures = tuple(error for error in mpi.allgather(local_error) if error is not None)
    if failures:
        details = "; ".join(
            f"rank {rank} {error_type}: {message}"
            for rank, error_type, message in failures
        )
        raise RuntimeError(f"collective projection stage failed: {details}")
    gathered = mpi.gather(local, root=0)
    updated: tuple[int, ...] = ()
    assembly_error: tuple[str, str] | None = None
    if mpi.is_root:
        try:
            if gathered is None:
                raise RuntimeError("MPI root did not receive stage payloads")
            combined: dict[int, dict[str, object]] = {}
            for item in gathered:
                overlap = combined.keys() & item.keys()
                if overlap:
                    raise RuntimeError(f"duplicate q-stage ownership: {sorted(overlap)}")
                combined.update(item)
            with RestartableHDF5(
                config.output.file,
                qpoints,
                build_run_manifest(config),
                resume=True,
            ) as writer:
                for index in sorted(combined):
                    writer.write_q(index, combined[index])
                writer.materialize()
            updated = tuple(sorted(combined))
        except Exception as exc:  # noqa: BLE001 - release non-root ranks
            assembly_error = (type(exc).__name__, str(exc))
    assembly_error = mpi.bcast(assembly_error, root=0)
    if assembly_error is not None:
        error_type, message = assembly_error
        raise RuntimeError(f"root projection output failed ({error_type}): {message}")
    updated = tuple(mpi.bcast(updated, root=0))
    mpi.barrier()
    return StageResult(config.output.file, updated, mpi.size)


def run_project_phonons(
    config: RunConfig,
    *,
    mpi: MPIContext | None = None,
) -> StageResult:
    context = MPIContext.discover() if mpi is None else mpi
    if config.dfpt.normalization is Normalization.PHONON_ZERO_POINT_MODE:
        with h5py.File(config.output.file, "r") as handle:
            if "kernel/V_pi_ph" not in handle:
                raise KeyError("mode-normalized kernel output is missing /kernel/V_pi_ph")
        return StageResult(config.output.file, (), context.size)
    if config.phonons_file is None:
        raise ValueError("project-phonons requires phonons.file")
    qpoints = _output_qpoints(config.output.file)
    with h5py.File(config.output.file, "r") as output:
        if "kernel/K_pi_u" not in output:
            raise KeyError("electronic output is missing /kernel/K_pi_u")
        kernels = np.asarray(output["kernel/K_pi_u"][...], dtype=np.complex128)
    with HDF5PhononProvider(config.phonons_file) as phonons:
        if phonons.qpoints is not None and (
            phonons.qpoints.shape != qpoints.shape
            or not np.allclose(phonons.qpoints, qpoints, atol=1e-9)
        ):
            raise ValueError("phonon and electronic q-point meshes do not match")

        def worker(index: int) -> dict[str, object]:
            qdata = phonons.q(index)
            v = project_phonons(
                kernels[index],
                normalization=config.dfpt.normalization,
                frequencies=qdata.frequencies_eV,
                masses=phonons.masses_amu,
                eigenvectors=qdata.eigenvectors,
                mass_weighted_mode_scale=phonons.mass_weighted_mode_scale,
            )
            payload: dict[str, object] = {"kernel/V_pi_ph": v}
            if np.max(np.abs(qpoints[index] - np.rint(qpoints[index]))) < 1e-9:
                absolute, relative = acoustic_translation_residual(kernels[index])
                payload["validation/acoustic_translation_absolute"] = np.float64(absolute)
                payload["validation/acoustic_translation_relative"] = np.float64(relative)
            return payload

        return _distributed_stage(config, worker, mpi=context)


def run_project_magnons(
    config: RunConfig,
    *,
    mpi: MPIContext | None = None,
) -> StageResult:
    context = MPIContext.discover() if mpi is None else mpi
    if config.magnons_file is None:
        raise ValueError("project-magnons requires magnons.file")
    model = load_spinor_wannier(
        config.electrons.file,
        bloch_gauge=config.electrons.bloch_gauge,
        spin_order=config.electrons.spin_order,
        onsite_soc=config.electrons.onsite_soc,
    )
    magnetic = load_magnetic_data(
        config.electrons.file,
        config.magnetic_subspace,
        norb=model.norb,
        orbital_labels=model.orbital_labels,
    )
    if magnetic.spin_lengths is None:
        raise ValueError("external magnon projection requires /spin/spin_length")
    qpoints = _output_qpoints(config.output.file)
    with h5py.File(config.output.file, "r") as output:
        if "kernel/V_pi_ph" not in output:
            raise KeyError("phonon-projected output is missing /kernel/V_pi_ph")
        v_pi_ph = np.asarray(output["kernel/V_pi_ph"][...], dtype=np.complex128)
    with HDF5MagnonProvider(
        config.magnons_file,
        expected_frames=magnetic.local_frames,
        expected_spin_lengths=magnetic.spin_lengths,
    ) as magnons:
        if magnons.qpoints.shape != qpoints.shape or not np.allclose(
            magnons.qpoints, qpoints, atol=1e-9
        ):
            raise ValueError("magnon and electronic q-point meshes do not match")

        def worker(index: int) -> dict[str, object]:
            qdata = magnons.q(index)
            projected = project_external_magnons(
                v_pi_ph[index],
                qdata.transform,
                magnetic.spin_lengths,
                spin_coordinate=config.magnetic_subspace.spin_coordinate.value,
            )
            return {
                "coupling/g_mp_normal": projected.normal,
                "coupling/g_mp_anomalous": projected.anomalous,
                "validation/magnon_paraunitarity": np.float64(
                    projected.paraunitarity_residual
                ),
            }

        return _distributed_stage(config, worker, mpi=context)


def run_assemble_polaron(
    config: RunConfig,
    *,
    mpi: MPIContext | None = None,
) -> StageResult:
    context = MPIContext.discover() if mpi is None else mpi
    if config.phonons_file is None or config.magnons_file is None:
        raise ValueError("assemble-polaron requires phonons.file and magnons.file")
    model = load_spinor_wannier(
        config.electrons.file,
        bloch_gauge=config.electrons.bloch_gauge,
        spin_order=config.electrons.spin_order,
        onsite_soc=config.electrons.onsite_soc,
    )
    magnetic = load_magnetic_data(
        config.electrons.file,
        config.magnetic_subspace,
        norb=model.norb,
        orbital_labels=model.orbital_labels,
    )
    if magnetic.spin_lengths is None:
        raise ValueError("polaron assembly requires /spin/spin_length")
    with h5py.File(config.output.file, "r") as output:
        if "coupling/g_mp_normal" not in output:
            raise KeyError("magnon-projected output is missing /coupling/g_mp_normal")
        coupling = np.asarray(output["coupling/g_mp_normal"][...], dtype=np.complex128)
    with HDF5PhononProvider(config.phonons_file) as phonons, HDF5MagnonProvider(
        config.magnons_file,
        expected_frames=magnetic.local_frames,
        expected_spin_lengths=magnetic.spin_lengths,
    ) as magnons:
        def worker(index: int) -> dict[str, object]:
            phonon_q = phonons.q(index)
            magnon_q = magnons.q(index)
            h_rwa = assemble_rwa(
                magnon_q.energies_eV,
                phonon_q.frequencies_eV,
                coupling[index],
            )
            energies = np.linalg.eigvalsh(h_rwa)
            return {
                "polaron/H_rwa": h_rwa,
                "polaron/energy": np.asarray(energies, dtype=np.float64),
            }

        return _distributed_stage(config, worker, mpi=context)


__all__ = [
    "StageResult",
    "run_assemble_polaron",
    "run_project_magnons",
    "run_project_phonons",
]
