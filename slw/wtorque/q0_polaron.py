"""End-to-end Gamma-point magnon-polaron projection for scalar EPR torque."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from slw.magph.epr_phonon import build_epr_phonon_cache_payload
from slw.magph.lswt import (
    local_spin_frames,
    magnon_mode_chirality,
    solve_isotropic_lswt,
)
from slw.magph.model import (
    ExchangeModel,
    ExchangeSpinNormalization,
    SingleIonAnisotropy,
)
from slw.magph.phonon import PhononCache, PhononMassUnit, zero_point_displacements
from slw.magph.screening import screen_magnetic_configuration
from slw.magph.tb2j import (
    ProjectedScalarTB2JReport,
    load_projected_scalar_tb2j_h5,
)

from .collinear_q0 import CollinearEPRQ0Result, run_collinear_epr_q0_test
from .model.magnetic_subspace import transverse_frame
from .parallel.mpi import MPIContext
from .projection.magnon import (
    degenerate_subspace_invariants,
    project_external_magnons,
)
from .projection.polaron import assemble_rwa, diagnose_bosonic_bdg


def _complex_payload(value: NDArray[np.complex128]) -> dict[str, object]:
    return {"real": value.real.tolist(), "imag": value.imag.tolist()}


@dataclass(frozen=True)
class Q0PolaronProjection:
    """Mode-resolved Gamma-point coupling in a common LSWT/phonon gauge."""

    spin_lengths: NDArray[np.float64]
    spin_pattern: NDArray[np.float64]
    single_ion_anisotropy_mev: NDArray[np.float64]
    torque_to_lswt_coordinate: NDArray[np.float64]
    magnon_energies_mev: NDArray[np.float64]
    magnon_chirality: NDArray[np.float64]
    phonon_mode_indices_one_based: NDArray[np.int64]
    phonon_energies_mev: NDArray[np.float64]
    v_pi_ph_mev: NDArray[np.complex128]
    normal_coupling_mev: NDArray[np.complex128]
    anomalous_coupling_mev: NDArray[np.complex128]
    phonon_coupling_norm_mev: NDArray[np.float64]
    coupling_singular_values_mev: NDArray[np.float64]
    detuning_mev: NDArray[np.float64]
    pair_resonant_gap_mev: NDArray[np.float64]
    rwa_hamiltonian_mev: NDArray[np.complex128]
    rwa_energies_mev: NDArray[np.float64]
    full_bdg_dynamic_eigenvalues_mev: NDArray[np.complex128]
    full_bdg_minimum_hessian_mev: float
    full_bdg_maximum_imaginary_mev: float
    full_bdg_particle_hole_residual_mev: float
    full_bdg_stable: bool
    magnon_paraunitarity_residual: float
    magnon_eigen_residual_mev: float
    rwa_hermiticity_residual_mev: float
    closest_pair_indices_one_based: tuple[int, int]
    closest_pair_detuning_mev: float
    closest_pair_coupling_mev: float
    closest_pair_resonant_gap_mev: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "spin_lengths": self.spin_lengths.tolist(),
            "spin_pattern": self.spin_pattern.tolist(),
            "single_ion_anisotropy_mev": (
                self.single_ion_anisotropy_mev.tolist()
            ),
            "torque_to_lswt_coordinate": self.torque_to_lswt_coordinate.tolist(),
            "magnon_energies_mev": self.magnon_energies_mev.tolist(),
            "magnon_chirality": self.magnon_chirality.tolist(),
            "phonon_mode_indices_one_based": (
                self.phonon_mode_indices_one_based.tolist()
            ),
            "phonon_energies_mev": self.phonon_energies_mev.tolist(),
            "v_pi_ph_mev": _complex_payload(self.v_pi_ph_mev),
            "normal_coupling_mev": _complex_payload(self.normal_coupling_mev),
            "anomalous_coupling_mev": _complex_payload(
                self.anomalous_coupling_mev
            ),
            "phonon_coupling_norm_mev": self.phonon_coupling_norm_mev.tolist(),
            "coupling_singular_values_mev": (
                self.coupling_singular_values_mev.tolist()
            ),
            "detuning_mev": self.detuning_mev.tolist(),
            "pair_resonant_gap_mev": self.pair_resonant_gap_mev.tolist(),
            "rwa_hamiltonian_mev": _complex_payload(self.rwa_hamiltonian_mev),
            "rwa_energies_mev": self.rwa_energies_mev.tolist(),
            "full_bdg_dynamic_eigenvalues_mev": _complex_payload(
                self.full_bdg_dynamic_eigenvalues_mev
            ),
            "full_bdg_minimum_hessian_mev": (
                self.full_bdg_minimum_hessian_mev
            ),
            "full_bdg_maximum_imaginary_mev": (
                self.full_bdg_maximum_imaginary_mev
            ),
            "full_bdg_particle_hole_residual_mev": (
                self.full_bdg_particle_hole_residual_mev
            ),
            "full_bdg_stable": self.full_bdg_stable,
            "magnon_paraunitarity_residual": (
                self.magnon_paraunitarity_residual
            ),
            "magnon_eigen_residual_mev": self.magnon_eigen_residual_mev,
            "rwa_hermiticity_residual_mev": (
                self.rwa_hermiticity_residual_mev
            ),
            "closest_pair_indices_one_based": list(
                self.closest_pair_indices_one_based
            ),
            "closest_pair_detuning_mev": self.closest_pair_detuning_mev,
            "closest_pair_coupling_mev": self.closest_pair_coupling_mev,
            "closest_pair_resonant_gap_mev": (
                self.closest_pair_resonant_gap_mev
            ),
        }


@dataclass(frozen=True)
class CollinearEPRQ0PolaronResult:
    """Torque, exchange-conversion, phonon, and hybridization diagnostics."""

    torque: CollinearEPRQ0Result
    exchange: ProjectedScalarTB2JReport
    projection: Q0PolaronProjection
    phonon_source: Path
    phonon_loto_mode: str
    phonon_minimum_raw_frequency_mev: float
    phonon_rounded_frequency_count: int
    exchange_lattice_max_residual_ang: float
    exchange_position_max_periodic_residual: float

    def as_dict(self) -> dict[str, Any]:
        exchange = {
            "source": str(self.exchange.source),
            "source_dataset": self.exchange.source_dataset,
            "source_directed_bond_weight": (
                self.exchange.source_directed_bond_weight
            ),
            "canonical_directed_bond_weight": (
                self.exchange.canonical_directed_bond_weight
            ),
            "exchange_scale": self.exchange.exchange_scale,
            "spin_normalization": self.exchange.spin_normalization.value,
            "source_spin_magnitude": self.exchange.source_spin_magnitude,
            "bond_count": self.exchange.bond_count,
            "magnetic_atom_indices_one_based": [
                value + 1 for value in self.exchange.magnetic_atom_indices
            ],
            "maximum_source_reciprocity_residual_mev": (
                self.exchange.maximum_source_reciprocity_residual_mev
            ),
            "scalar_spin_group_projected": (
                self.exchange.scalar_spin_group_projected
            ),
            "source_weight_origin": self.exchange.source_weight_origin,
        }
        return {
            "approximation": (
                "Gamma-only RWA magnon-polaron projection from bubble torque; "
                "scalar isotropic LSWT; optical phonons only"
            ),
            "torque": self.torque.as_dict(),
            "exchange": exchange,
            "projection": self.projection.as_dict(),
            "phonon_source": str(self.phonon_source),
            "phonon_loto_mode": self.phonon_loto_mode,
            "phonon_minimum_raw_frequency_mev": (
                self.phonon_minimum_raw_frequency_mev
            ),
            "phonon_rounded_frequency_count": (
                self.phonon_rounded_frequency_count
            ),
            "exchange_lattice_max_residual_ang": (
                self.exchange_lattice_max_residual_ang
            ),
            "exchange_position_max_periodic_residual": (
                self.exchange_position_max_periodic_residual
            ),
        }


def _site_values(value: object, count: int, *, name: str) -> NDArray[np.float64]:
    raw = np.asarray(value, dtype=np.float64).reshape(-1)
    if raw.size == 1:
        result = np.full(count, float(raw[0]), dtype=np.float64)
    elif raw.size == count:
        result = np.array(raw, dtype=np.float64, copy=True)
    else:
        raise ValueError(f"{name} must contain one value or {count} site values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _unit_directions(value: object, count: int) -> NDArray[np.float64]:
    directions = np.asarray(value, dtype=np.float64)
    if directions.shape != (count, 3) or not np.all(np.isfinite(directions)):
        raise ValueError(f"local directions must have finite shape ({count},3)")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("local directions must be nonzero")
    return np.asarray(directions / norms[:, None], dtype=np.float64)


def _unit_axis(value: object) -> NDArray[np.float64]:
    axis = np.asarray(value, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("spin quantization direction must be a finite length-3 vector")
    norm = float(np.linalg.norm(axis))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("spin quantization direction must be nonzero")
    return np.asarray(axis / norm, dtype=np.float64)


def _torque_to_transverse_direction(
    kernel: NDArray[np.complex128],
    coordinate: str,
) -> NDArray[np.complex128]:
    if coordinate == "transverse_direction":
        return np.array(kernel, dtype=np.complex128, copy=True)
    if coordinate != "rotation_angle":
        raise ValueError(
            "spin coordinate must be rotation_angle or transverse_direction"
        )
    # pi_1=theta_2 and pi_2=-theta_1 in a right-handed [t1,t2,n] frame.
    return np.asarray(
        np.stack((kernel[:, 1], -kernel[:, 0]), axis=1),
        dtype=np.complex128,
    )


def _phonon_cache_from_payload(
    source: str | Path,
    payload: dict[str, np.ndarray],
) -> PhononCache:
    mesh = np.asarray(payload["q_mesh_shape"], dtype=np.int64)
    if mesh.shape != (3,):
        raise ValueError("EPR phonon payload q_mesh_shape must contain three values")
    return PhononCache(
        source=str(Path(source).expanduser().resolve()),
        schema_version=int(np.asarray(payload["phonon_cache_schema_version"])),
        q_points_frac=payload["q_mesh_flat_frac"],
        frequencies_mev=payload["ph_en_flat"],
        eigenvectors_mass_normalized=payload["ph_vec_flat"],
        masses=payload["atom_mass_electron"],
        mass_unit=PhononMassUnit.ELECTRON_MASS,
        q_mesh_shape=(int(mesh[0]), int(mesh[1]), int(mesh[2])),
        vector_convention=str(np.asarray(payload["phonon_vector_convention"])),
        fourier_phase_convention=str(
            np.asarray(payload["fourier_phase_convention"])
        ),
    )


def project_q0_magnon_polaron(
    kernel_eV_per_angstrom: object,
    *,
    spin_coordinate: str,
    local_magnetization_directions: object,
    spin_quantization_direction: object,
    magnetic_atom_indices: object,
    exchange: ExchangeModel,
    spin_lengths: object,
    anisotropy_mev: object,
    anisotropy_spin_normalization: ExchangeSpinNormalization | str,
    phonons: PhononCache,
    phonon_min_energy_mev: float,
) -> Q0PolaronProjection:
    """Project one Cartesian Gamma torque kernel into magnon/phonon modes."""

    kernel = np.asarray(kernel_eV_per_angstrom, dtype=np.complex128)
    if kernel.ndim != 4 or kernel.shape[1] != 2 or kernel.shape[3] != 3:
        raise ValueError("q=0 torque kernel must have shape (nmag,2,natom,3)")
    n_torque_site = kernel.shape[0]
    torque_atoms = np.asarray(magnetic_atom_indices, dtype=np.int64)
    if torque_atoms.shape != (n_torque_site,) or np.unique(torque_atoms).size != n_torque_site:
        raise ValueError("magnetic_atom_indices must identify each torque site once")
    exchange_atoms = np.asarray(exchange.magnetic_atom_indices, dtype=np.int64)
    if exchange_atoms.size != n_torque_site or set(exchange_atoms) != set(torque_atoms):
        raise ValueError(
            "torque and exchange inputs contain different magnetic atoms"
        )
    lookup = {int(atom): site for site, atom in enumerate(torque_atoms)}
    torque_order = np.asarray(
        [lookup[int(atom)] for atom in exchange_atoms], dtype=np.int64
    )
    kernel = kernel[torque_order]
    directions = _unit_directions(
        local_magnetization_directions, n_torque_site
    )[torque_order]
    axis = _unit_axis(spin_quantization_direction)
    projections = directions @ axis
    if not np.allclose(np.abs(projections), 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            "local magnetization directions must be collinear with the LSWT axis"
        )
    pattern = np.where(projections >= 0.0, 1.0, -1.0)
    if n_torque_site != 2 or pattern[0] != -pattern[1]:
        raise ValueError("Gamma collinear-AFM projection requires two opposite sites")
    spins_torque = _site_values(spin_lengths, n_torque_site, name="spin_lengths")
    anisotropy_torque = _site_values(
        anisotropy_mev,
        n_torque_site,
        name="anisotropy_mev",
    )
    spins = spins_torque[torque_order]
    anisotropy_values = anisotropy_torque[torque_order]
    if np.any(spins <= 0.0):
        raise ValueError("spin_lengths must be positive")
    anisotropy = SingleIonAnisotropy(
        energy_mev=anisotropy_values,
        axis=np.repeat(axis[None, :], n_torque_site, axis=0),
        spin_normalization=ExchangeSpinNormalization(
            anisotropy_spin_normalization
        ),
    )
    configuration = screen_magnetic_configuration(
        exchange,
        order="collinear_afm",
        spin_pattern=pattern,
        spin_magnitudes=spins,
        quantization_axis=axis,
        anisotropy=anisotropy,
    )
    spectrum = solve_isotropic_lswt(
        exchange,
        configuration,
        np.zeros((1, 3), dtype=np.float64),
        anisotropy=anisotropy,
    )

    torque_frames = np.stack([transverse_frame(direction) for direction in directions])
    lswt_frames = local_spin_frames(configuration)
    coordinate_map = np.einsum(
        "sac,scb->sab",
        torque_frames[:, :2, :],
        lswt_frames[:, :, :2],
        optimize=True,
    )
    kernel_pi_torque = _torque_to_transverse_direction(kernel, spin_coordinate)
    flattened = kernel_pi_torque.reshape(n_torque_site, 2, -1)
    kernel_pi_lswt = np.einsum(
        "sai,sab->sbi", flattened, coordinate_map, optimize=True
    ).reshape(kernel.shape)

    floor = float(phonon_min_energy_mev)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("phonon_min_energy_mev must be positive and finite")
    gamma_indices = np.flatnonzero(
        np.max(np.abs(phonons.q_points_frac - np.rint(phonons.q_points_frac)), axis=1)
        <= 1.0e-10
    )
    if gamma_indices.size != 1:
        raise ValueError("phonon input must contain exactly one Gamma point")
    gamma = int(gamma_indices[0])
    if phonons.nat != kernel.shape[2]:
        raise ValueError(
            "torque atom count and phonon atom count differ: "
            f"{kernel.shape[2]} != {phonons.nat}"
        )
    zero_point = zero_point_displacements(
        phonons,
        frequency_floor_mev=floor,
    )
    selected = np.flatnonzero(~zero_point.regularized[gamma])
    if selected.size == 0:
        raise ValueError("no phonon mode lies above phonon_min_energy_mev")
    displacements = zero_point.values_ang[gamma, selected]
    v_pi_ph = np.einsum(
        "siac,vac->siv",
        1000.0 * kernel_pi_lswt,
        displacements,
        optimize=True,
    )
    magnon_projection = project_external_magnons(
        v_pi_ph,
        spectrum.transformation[0],
        spins,
        tolerance=max(1.0e-9, 10.0 * spectrum.max_paraunitary_residual),
    )
    magnon_energies = np.asarray(
        spectrum.physical_energies_mev[0], dtype=np.float64
    )
    phonon_energies = np.asarray(
        phonons.frequencies_mev[gamma, selected], dtype=np.float64
    )
    normal = magnon_projection.normal
    anomalous = magnon_projection.anomalous
    rwa = assemble_rwa(magnon_energies, phonon_energies, normal)
    rwa_energies = np.linalg.eigvalsh(rwa)
    full_bdg = diagnose_bosonic_bdg(
        magnon_energies,
        phonon_energies,
        normal,
        anomalous,
    )
    detuning = magnon_energies[:, None] - phonon_energies[None, :]
    gaps = 2.0 * np.abs(normal)
    closest_flat = int(np.argmin(np.abs(detuning)))
    closest_magnon, closest_phonon = np.unravel_index(
        closest_flat, detuning.shape
    )
    singular, _norm = degenerate_subspace_invariants(normal)
    chirality = magnon_mode_chirality(
        spectrum.transformation,
        configuration.spin_pattern,
        physical_mode_count=spectrum.physical_mode_count,
    )[0]
    hermiticity = float(np.max(np.abs(rwa - rwa.conj().T), initial=0.0))
    return Q0PolaronProjection(
        spin_lengths=np.asarray(spins, dtype=np.float64),
        spin_pattern=np.asarray(pattern, dtype=np.float64),
        single_ion_anisotropy_mev=np.asarray(
            anisotropy_values, dtype=np.float64
        ),
        torque_to_lswt_coordinate=np.asarray(coordinate_map, dtype=np.float64),
        magnon_energies_mev=magnon_energies,
        magnon_chirality=np.asarray(chirality, dtype=np.float64),
        phonon_mode_indices_one_based=np.asarray(selected + 1, dtype=np.int64),
        phonon_energies_mev=phonon_energies,
        v_pi_ph_mev=np.asarray(v_pi_ph, dtype=np.complex128),
        normal_coupling_mev=np.asarray(normal, dtype=np.complex128),
        anomalous_coupling_mev=np.asarray(anomalous, dtype=np.complex128),
        phonon_coupling_norm_mev=np.asarray(
            np.linalg.norm(normal, axis=0), dtype=np.float64
        ),
        coupling_singular_values_mev=np.asarray(singular, dtype=np.float64),
        detuning_mev=np.asarray(detuning, dtype=np.float64),
        pair_resonant_gap_mev=np.asarray(gaps, dtype=np.float64),
        rwa_hamiltonian_mev=np.asarray(rwa, dtype=np.complex128),
        rwa_energies_mev=np.asarray(rwa_energies, dtype=np.float64),
        full_bdg_dynamic_eigenvalues_mev=np.asarray(
            full_bdg.dynamic_eigenvalues_mev, dtype=np.complex128
        ),
        full_bdg_minimum_hessian_mev=float(
            full_bdg.minimum_hessian_eigenvalue_mev
        ),
        full_bdg_maximum_imaginary_mev=float(
            full_bdg.maximum_imaginary_eigenvalue_mev
        ),
        full_bdg_particle_hole_residual_mev=float(
            full_bdg.particle_hole_residual_mev
        ),
        full_bdg_stable=bool(full_bdg.stable),
        magnon_paraunitarity_residual=float(
            magnon_projection.paraunitarity_residual
        ),
        magnon_eigen_residual_mev=float(spectrum.max_eigen_residual_mev),
        rwa_hermiticity_residual_mev=hermiticity,
        closest_pair_indices_one_based=(
            int(closest_magnon) + 1,
            int(selected[closest_phonon]) + 1,
        ),
        closest_pair_detuning_mev=float(detuning[closest_magnon, closest_phonon]),
        closest_pair_coupling_mev=float(abs(normal[closest_magnon, closest_phonon])),
        closest_pair_resonant_gap_mev=float(gaps[closest_magnon, closest_phonon]),
    )


def _geometry_residuals(
    exchange: ExchangeModel,
    payload: dict[str, np.ndarray],
) -> tuple[float, float]:
    if exchange.lattice_ang is None or exchange.tau_frac is None:
        raise ValueError(
            "projected exchange requires embedded lattice_ang and tau_frac"
        )
    lattice = np.asarray(payload["lattice_ang"], dtype=np.float64)
    positions = np.asarray(payload["atom_frac"], dtype=np.float64)
    if exchange.lattice_ang.shape != lattice.shape:
        raise ValueError("exchange and EPR lattice shapes differ")
    if exchange.tau_frac.shape != positions.shape:
        raise ValueError("exchange and EPR position shapes differ")
    lattice_residual = float(np.max(np.abs(exchange.lattice_ang - lattice)))
    delta = np.asarray(exchange.tau_frac - positions, dtype=np.float64)
    delta -= np.rint(delta)
    position_residual = float(np.max(np.abs(delta), initial=0.0))
    return lattice_residual, position_residual


def run_collinear_epr_q0_polaron_test(
    *,
    exchange_h5_path: str | Path,
    exchange_source_directed_bond_weight: float,
    exchange_spin_normalization: ExchangeSpinNormalization | str,
    spin_lengths: object,
    anisotropy_mev: object,
    anisotropy_spin_normalization: ExchangeSpinNormalization | str,
    phonon_min_energy_mev: float,
    phonon_imaginary_tolerance_mev: float,
    phonon_loto_mode: str,
    geometry_tolerance: float,
    torque_imaginary_tolerance_eV_per_angstrom: float,
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
) -> CollinearEPRQ0PolaronResult | None:
    """Run the MPI torque bubble and root-only Gamma mode projection."""

    context = MPIContext.discover() if mpi is None else mpi
    torque = run_collinear_epr_q0_test(
        up_epr_path=up_epr_path,
        down_epr_path=down_epr_path,
        magnetic_orbital_specification=magnetic_orbital_specification,
        magnetic_atom_indices=magnetic_atom_indices,
        local_magnetization_directions=local_magnetization_directions,
        spin_quantization_direction=spin_quantization_direction,
        spin_coordinate=spin_coordinate,
        fermi_energy_eV=fermi_energy_eV,
        energy_min_eV=energy_min_eV,
        energy_max_eV=energy_max_eV,
        energy_points=energy_points,
        eta_eV=eta_eV,
        temperature_K=temperature_K,
        epr_energy_unit=epr_energy_unit,
        epr_displacement_unit=epr_displacement_unit,
        integration_backend=integration_backend,
        onsite_soc_specifications=onsite_soc_specifications,
        p_orbital_order=p_orbital_order,
        d_orbital_order=d_orbital_order,
        divide_by_ws_degeneracy=divide_by_ws_degeneracy,
        perturbation_chunk=perturbation_chunk,
        mpi=context,
    )
    final_error: tuple[str, str] | None = None
    result: CollinearEPRQ0PolaronResult | None = None
    if context.is_root:
        try:
            if torque is None:
                raise RuntimeError("MPI root did not receive the q=0 torque result")
            imaginary_tolerance = float(
                torque_imaginary_tolerance_eV_per_angstrom
            )
            if not np.isfinite(imaginary_tolerance) or imaginary_tolerance < 0.0:
                raise ValueError(
                    "torque imaginary tolerance must be finite and non-negative"
                )
            if (
                torque.kernel_imaginary_residual_eV_per_angstrom
                > imaginary_tolerance
            ):
                raise ValueError(
                    "q=0 torque has a non-negligible imaginary residual: "
                    f"{torque.kernel_imaginary_residual_eV_per_angstrom:.6g} eV/A"
                )
            spins = _site_values(
                spin_lengths,
                torque.magnetic_site_count,
                name="spin_lengths",
            )
            if not np.allclose(spins, spins[0], rtol=0.0, atol=1.0e-12):
                raise ValueError(
                    "the current exchange convention stores one source spin "
                    "magnitude; all site spin lengths must match"
                )
            exchange, exchange_report = load_projected_scalar_tb2j_h5(
                exchange_h5_path,
                source_directed_bond_weight=(
                    exchange_source_directed_bond_weight
                ),
                spin_normalization=exchange_spin_normalization,
                source_spin_magnitude=float(spins[0]),
            )
            payload, phonon_report = build_epr_phonon_cache_payload(
                up_epr_path,
                q_mesh_shape=(1, 1, 1),
                q_chunk_size=1,
                loto_mode=phonon_loto_mode,
                imaginary_tolerance_mev=phonon_imaginary_tolerance_mev,
            )
            lattice_residual, position_residual = _geometry_residuals(
                exchange, payload
            )
            tolerance = float(geometry_tolerance)
            if not np.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("geometry_tolerance must be finite and non-negative")
            if max(lattice_residual, position_residual) > tolerance:
                raise ValueError(
                    "exchange and EPR structures differ: "
                    f"lattice={lattice_residual:.3e} A, "
                    f"fractional_positions={position_residual:.3e}"
                )
            phonons = _phonon_cache_from_payload(up_epr_path, payload)
            projection = project_q0_magnon_polaron(
                np.asarray(torque.kernel_eV_per_angstrom, dtype=np.complex128),
                spin_coordinate=torque.spin_coordinate,
                local_magnetization_directions=(
                    torque.local_magnetization_directions
                ),
                spin_quantization_direction=torque.spin_quantization_direction,
                magnetic_atom_indices=np.asarray(
                    torque.magnetic_atom_indices_one_based, dtype=np.int64
                )
                - 1,
                exchange=exchange,
                spin_lengths=spins,
                anisotropy_mev=anisotropy_mev,
                anisotropy_spin_normalization=(
                    anisotropy_spin_normalization
                ),
                phonons=phonons,
                phonon_min_energy_mev=phonon_min_energy_mev,
            )
            result = CollinearEPRQ0PolaronResult(
                torque=torque,
                exchange=exchange_report,
                projection=projection,
                phonon_source=Path(up_epr_path).expanduser().resolve(),
                phonon_loto_mode=str(phonon_loto_mode).strip().lower(),
                phonon_minimum_raw_frequency_mev=float(
                    phonon_report.minimum_raw_frequency_mev
                ),
                phonon_rounded_frequency_count=int(
                    phonon_report.rounded_frequency_count
                ),
                exchange_lattice_max_residual_ang=lattice_residual,
                exchange_position_max_periodic_residual=position_residual,
            )
        except Exception as exc:  # noqa: BLE001 - release MPI peers
            final_error = (type(exc).__name__, str(exc))
    final_error = context.bcast(final_error, root=0)
    if final_error is not None:
        error_type, message = final_error
        raise RuntimeError(
            f"Gamma magnon-polaron projection failed ({error_type}): {message}"
        )
    context.barrier()
    return result


__all__ = [
    "CollinearEPRQ0PolaronResult",
    "Q0PolaronProjection",
    "project_q0_magnon_polaron",
    "run_collinear_epr_q0_polaron_test",
]
