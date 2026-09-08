"""Native EPR phonons and isotropic TB2J/SLW magnons for torque projection.

All returned energies are eV.  Polarization vectors are dimensionless,
Euclidean-normalized Cartesian eigenvectors; masses are amu.  In particular,
qe2pert ``basic_data/mass`` is in *Rydberg* atomic units (electron mass 1/2),
and must not be interpreted as a mass in electron-mass units.

The atomic Fourier convention uses exp(+i 2pi q.(R+tau)); consequently
atomic-gauge coordinate vectors are exp(-i 2pi q.tau) times cell-gauge
vectors.  The same row phase applies to both sectors of (a(q), a†(-q)).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from slw.core.constants import RY_TO_EV
from slw.epc.epr_phonon import (
    apply_loto_override, polar_onsite_correction, read_epr_phonon_metadata,
)
from slw.magph.epr_phonon import _assemble_chunk, _preload_ifc_blocks
from slw.magph.lswt import local_spin_frames, solve_isotropic_lswt
from slw.magph.model import (
    ExchangeConvention, ExchangeModel, ExchangeRepresentation,
    ExchangeSpinNormalization,
)
from slw.magph.screening import screen_magnetic_configuration
from slw.wtorque.projection.magnon import paraunitarity_residual

# QE Modules/constants.f90, NIST 2018 values used by the native EPR writer.
QE_BOHR_TO_ANG = 0.529177210903
QE_AMU_RY = 1.66053906660e-27 / (2.0 * 9.1093837015e-31)


def _qpoints(value: object) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] == 0:
        raise ValueError("qpoints must have nonempty shape (nq,3)")
    if not np.all(np.isfinite(result)):
        raise ValueError("qpoints must be finite")
    return result


def _gauge(value: str) -> str:
    if value not in {"cell", "atomic"}:
        raise ValueError("gauge must be 'cell' or 'atomic'")
    return value


@dataclass(frozen=True)
class NativePhononModes:
    qpoints: NDArray[np.float64]
    energies_eV: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]  # [nq,natom,3,nmode]
    masses_amu: NDArray[np.float64]
    lattice_ang: NDArray[np.float64]  # row lattice vectors
    atom_positions_frac: NDArray[np.float64]
    dynamical_matrix_ry2: NDArray[np.complex128]
    gauge: str
    diagnostics: dict[str, object]


def native_phonon_modes(
    epr_path: str | Path,
    qpoints: object,
    *,
    gauge: str = "atomic",
    loto_mode: str = "auto",
) -> NativePhononModes:
    """Reconstruct native IFCs, add polar IFCs, and diagonalize explicit q.

    No acoustic rounding, energy shift, or mass normalization is hidden here:
    nonpositive modes raise before zero-point projection.  The q list can be
    partitioned across MPI ranks by the workflow owner.
    """
    points = _qpoints(qpoints)
    selected_gauge = _gauge(gauge)
    source = Path(epr_path).expanduser().resolve()
    meta = apply_loto_override(read_epr_phonon_metadata(source), loto_mode)
    # The Fortran writer stores ifc(cart_i,cart_j,R) and zstar(field,disp,atom).
    # h5py reverses those axes.  The legacy cache helper leaves cart_i/cart_j
    # reversed; restore the physical matrices before reusing its assembly.
    blocks = [(*block[:5], np.ascontiguousarray(block[5].swapaxes(-1, -2)))
              for block in _preload_ifc_blocks(source, meta)]
    if bool(meta["lpolar"]):
        meta["zstar"] = np.asarray(meta["zstar"]).swapaxes(-1, -2)
    mass_ry = np.asarray(meta["mass"], dtype=np.float64)
    repeated_mass = np.repeat(mass_ry, 3)
    mass_factor = 1.0 / np.sqrt(repeated_mass[:, None] * repeated_mass[None, :])
    polar_onsite = polar_onsite_correction(meta) if bool(meta["lpolar"]) else None
    # Evaluate both signs even for a caller supplying one q.  A deterministic
    # representative (first nonzero component positive) anchors the entire
    # degenerate polarization subspace, so e(-q)=e(q)* also holds across calls.
    paired_points = np.concatenate((points, -points), axis=0)
    paired_dynamical = _assemble_chunk(
        paired_points, meta=meta, blocks=blocks, mass_factor=mass_factor,
        polar_onsite=polar_onsite,
    )
    positions = np.asarray(meta["tau"], dtype=np.float64)
    if selected_gauge == "atomic":
        phase = np.repeat(np.exp(-2j * np.pi * (paired_points @ positions.T)), 3, axis=1)
        paired_dynamical = phase[:, :, None] * paired_dynamical * phase[:, None, :].conj()
    dynamical, minus_dynamical = np.split(paired_dynamical, 2)
    qpair_residual = float(np.max(
        np.linalg.norm(dynamical - minus_dynamical.conj(), axis=(1, 2)) /
        np.maximum(np.linalg.norm(dynamical, axis=(1, 2)), np.finfo(float).tiny)
    ))
    if not np.isfinite(qpair_residual) or qpair_residual > 1.0e-10:
        raise ValueError(f"native phonon D(-q) differs from D(q)*: relative residual={qpair_residual:.6g}")
    first_nonzero = np.argmax(np.abs(points) > 1.0e-12, axis=1)
    negative_representative = points[np.arange(points.shape[0]), first_nonzero] < 0
    canonical_dynamical = np.where(negative_representative[:, None, None], minus_dynamical, dynamical)
    eigenvalues, vectors = np.linalg.eigh(canonical_dynamical)
    vectors = np.where(negative_representative[:, None, None], vectors.conj(), vectors)
    signed_energy = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * RY_TO_EV
    if np.any(signed_energy <= 0):
        iq, mode = np.unravel_index(np.argmin(signed_energy), signed_energy.shape)
        raise ValueError(
            "nonpositive native phonon mode cannot receive zero-point normalization: "
            f"q={points[iq].tolist()}, mode={mode}, energy={signed_energy[iq, mode]:.9g} eV"
        )
    eigen_residual = np.max(np.abs(
        dynamical @ vectors - vectors * eigenvalues[:, None, :]
    ))
    ortho_residual = np.max(np.abs(vectors.swapaxes(1, 2).conj() @ vectors - np.eye(vectors.shape[1])))
    return NativePhononModes(
        qpoints=points,
        energies_eV=np.asarray(signed_energy),
        eigenvectors=np.asarray(vectors.reshape(points.shape[0], mass_ry.size, 3, -1)),
        masses_amu=mass_ry / QE_AMU_RY,
        lattice_ang=np.asarray(meta["at"]).T * float(meta["alat"]) * QE_BOHR_TO_ANG,
        atom_positions_frac=positions,
        dynamical_matrix_ry2=np.asarray(dynamical),
        gauge=selected_gauge,
        diagnostics={
            "source": str(source), "source_mass_unit": "rydberg_atomic_mass",
            "mass_conversion_amu_ry": QE_AMU_RY,
            "ifc_cartesian_axis_transpose": True,
            "loto_mode": loto_mode, "polar_ifc_restored": bool(meta["lpolar"]),
            "eigen_residual_ry2": float(eigen_residual),
            "orthonormality_residual": float(ortho_residual),
            "qpair_matrix_relative_residual": qpair_residual,
            "qpair_mode_gauge": "e(-q)=conj(e(q)); first_nonzero_q_component_positive_anchor",
            "minimum_energy_eV": float(signed_energy.min()),
            "fourier_phase": "exp(+i2pi_q_dot_R_plus_tau)" if gauge == "atomic" else "exp(+i2pi_q_dot_R)",
        },
    )


def validate_phonons_against_qe_dyn_xml(
    modes: NativePhononModes,
    dyn_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    q_tolerance: float = 1.0e-7,
    matrix_relative_tolerance: float = 1.0e-9,
) -> dict[str, object]:
    """Independently compare every requested q with raw QE PH XML matrices.

    ``PHI.i.j`` is read as Fortran column-major complex Cartesian data.  This
    check bypasses IFC interpolation and polar splitting; it catches axis,
    mass, and Fourier errors that frequency-only checks can miss.
    """
    mass = np.repeat(modes.masses_amu * QE_AMU_RY, 3)
    mass_factor = 1.0 / np.sqrt(mass[:, None] * mass[None, :])
    nat = modes.masses_amu.size
    matches: dict[int, dict[str, object]] = {}
    for path in dyn_paths:
        source = Path(path).expanduser().resolve()
        root = ET.parse(source).getroot()
        at_xml = root.find("GEOMETRY_INFO/AT")
        if at_xml is None or at_xml.text is None:
            raise ValueError(f"QE dynamical XML has no GEOMETRY_INFO/AT: {source}")
        at_rows = np.fromstring(at_xml.text, sep=" ").reshape(3, 3)
        for dyn in root:
            if not dyn.tag.startswith("DYNAMICAL_MAT_"):
                continue
            q_xml = dyn.find("Q_POINT")
            if q_xml is None or q_xml.text is None:
                raise ValueError(f"QE dynamical XML has no Q_POINT: {source}:{dyn.tag}")
            q_reduced = np.fromstring(q_xml.text, sep=" ") @ at_rows.T
            delta = modes.qpoints - q_reduced
            indices = np.flatnonzero(np.max(np.abs(delta - np.rint(delta)), axis=1) <= q_tolerance)
            if not indices.size:
                continue
            matrix = np.empty((3*nat, 3*nat), dtype=np.complex128)
            for atom_i in range(nat):
                for atom_j in range(nat):
                    phi = dyn.find(f"PHI.{atom_i+1}.{atom_j+1}")
                    if phi is None or phi.text is None:
                        raise ValueError(f"QE PHI atom-pair data missing: {source}:{dyn.tag}")
                    pairs = np.fromstring(phi.text, sep=" ").reshape(9, 2)
                    values = pairs[:, 0] + 1j*pairs[:, 1]
                    matrix[3*atom_i:3*atom_i+3, 3*atom_j:3*atom_j+3] = values.reshape(3, 3, order="F")
            matrix *= mass_factor
            hermiticity = float(np.max(np.abs(matrix - matrix.conj().T)))
            eigenvalues = np.linalg.eigvalsh(matrix)
            energies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * RY_TO_EV
            for index in indices:
                expected = matrix
                if modes.gauge == "atomic":
                    phase = np.repeat(np.exp(-2j*np.pi*(modes.atom_positions_frac @ modes.qpoints[index])), 3)
                    expected = phase[:, None] * matrix * phase[None, :].conj()
                residual = float(np.linalg.norm(expected - modes.dynamical_matrix_ry2[index]) / max(np.linalg.norm(expected), np.finfo(float).tiny))
                if not np.isfinite(residual) or residual > matrix_relative_tolerance:
                    raise ValueError(
                        f"native EPR phonons disagree with QE PH XML at q index {index}: "
                        f"matrix relative residual={residual:.6g}"
                    )
                matches[int(index)] = {
                    "q_index": int(index), "source": str(source), "xml_tag": dyn.tag,
                    "matrix_relative_residual": residual,
                    "frequency_max_abs_residual_eV": float(np.max(np.abs(energies - modes.energies_eV[index]))),
                    "qe_matrix_hermiticity_residual_ry2": hermiticity,
                }
    missing = sorted(set(range(modes.qpoints.shape[0])) - set(matches))
    if missing:
        raise ValueError(f"QE dynamical XML lacks requested q indices: {missing}")
    return {
        "status": "pass", "q_count": len(matches),
        "max_matrix_relative_residual": max(float(v["matrix_relative_residual"]) for v in matches.values()),
        "max_frequency_abs_residual_eV": max(float(v["frequency_max_abs_residual_eV"]) for v in matches.values()),
        "matches": [matches[index] for index in sorted(matches)],
    }


_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
_BOND = re.compile(
    rf"^\s*(\S+)\s+(\S+)\s+\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    rf"\s+({_FLOAT})\s+\([^)]*\)\s+({_FLOAT})\s*$", re.MULTILINE,
)


@dataclass(frozen=True)
class TB2JIsotropicModel:
    exchange: ExchangeModel
    magnetic_atom_labels: tuple[str, ...]
    moments_muB: NDArray[np.float64]
    diagnostics: dict[str, object]


def load_tb2j_isotropic_model(
    exchange_path: str | Path,
    *,
    source_directed_bond_weight: float,
    magnetic_atom_labels: tuple[str, ...] | list[str] | None = None,
    reciprocity_tolerance_meV: float = 1.0e-8,
) -> TB2JIsotropicModel:
    """Parse TB2J's text J_iso only and explicitly convert directed weights.

    This intentionally selects the isotropic exchange approximation, leaving
    DMI/Jani unused.  The source Hamiltonian weight must be supplied; for the
    TB2J convention H=-sum_ij J_ij e_i.e_j it is 1 and J_native=2*J_source.
    Every mate must exist; differing mate values are rejected (no averaging).
    """
    source = Path(exchange_path).expanduser().resolve()
    content = source.read_text()
    if "generated by TB2J" not in content or "J_iso(meV)" not in content:
        raise ValueError("expected a TB2J exchange.out with J_iso(meV)")
    try:
        lattice = np.asarray([
            [float(item) for item in line.split()]
            for line in content.split("Cell (Angstrom):", 1)[1].strip().splitlines()[:3]
        ])
        atom_text = content.split("Atoms:", 1)[1].split("Exchange:", 1)[0]
    except (IndexError, ValueError) as exc:
        raise ValueError("TB2J exchange.out lacks valid cell/atom sections") from exc
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("TB2J lattice must be finite with shape (3,3)")
    labels, positions, moments = [], [], []
    for line in atom_text.splitlines():
        fields = line.split()
        if len(fields) != 8 or fields[0] in {"Total", "Atom_number"}:
            continue
        try:
            values = [float(item) for item in fields[1:]]
        except ValueError:
            continue
        labels.append(fields[0])
        positions.append(values[:3])
        moments.append(values[4:7])
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("TB2J atoms must have unique labels")
    bonds = list(_BOND.finditer(content))
    if not bonds:
        raise ValueError("TB2J exchange.out contains no J_iso bonds")
    bonded_labels = {match[i] for match in bonds for i in (1, 2)}
    selected = tuple(label for label in labels if label in bonded_labels) if magnetic_atom_labels is None else tuple(magnetic_atom_labels)
    if not selected or len(set(selected)) != len(selected) or any(label not in labels for label in selected):
        raise ValueError("magnetic atom labels must be a unique nonempty subset of source atoms")
    if set(selected) != bonded_labels:
        raise ValueError("magnetic atom labels must cover exactly all atoms in the source bond list")
    local_index = {label: i for i, label in enumerate(selected)}
    global_index = {label: i for i, label in enumerate(labels)}
    atom_indices = np.asarray([global_index[label] for label in selected], dtype=np.int64)
    bond_i = np.asarray([local_index[m[1]] for m in bonds], dtype=np.int64)
    bond_j = np.asarray([local_index[m[2]] for m in bonds], dtype=np.int64)
    shifts = np.asarray([[int(m[i]) for i in (3, 4, 5)] for m in bonds], dtype=np.int64)
    source_values = np.asarray([float(m[6].replace("D", "E").replace("d", "e")) for m in bonds])
    weight = float(source_directed_bond_weight)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("source_directed_bond_weight must be finite and positive")
    lookup = {(int(i), int(j), *r): index for index, (i, j, r) in enumerate(zip(bond_i, bond_j, shifts))}
    if len(lookup) != len(bonds):
        raise ValueError("duplicate TB2J directed bond")
    try:
        mirror = np.asarray([lookup[(int(j), int(i), *(-r))] for i, j, r in zip(bond_i, bond_j, shifts)], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"missing reciprocal TB2J mate: {exc.args[0]}") from exc
    residual = float(np.max(np.abs(source_values - source_values[mirror])))
    if residual > reciprocity_tolerance_meV or not np.array_equal(source_values, source_values[mirror]):
        raise ValueError(f"TB2J J_iso mates differ by {residual:.6g} meV; no implicit averaging is applied")
    values = source_values * (weight / 0.5)
    positions_array = np.asarray(positions, dtype=np.float64)
    model = ExchangeModel(
        source=source, representation=ExchangeRepresentation.ISOTROPIC,
        source_dataset="exchange.out:J_iso[explicit_isotropic_projection_and_bond_weight_conversion]",
        convention=ExchangeConvention(
            spin_normalization=ExchangeSpinNormalization.UNIT_VECTOR,
            source_spin_magnitude=1.0, kernel_family="tb2j_text_converted_isotropic",
        ),
        isotropic_mev=values,
        tensor_mev=values[:, None, None] * np.eye(3),
        bond_i=bond_i, bond_j=bond_j,
        bond_i_atom=atom_indices[bond_i], bond_j_atom=atom_indices[bond_j],
        cell_shift=shifts, magnetic_atom_indices=atom_indices, mirror_index=mirror,
        distance_ang=np.asarray([float(m[7]) for m in bonds]),
        lattice_ang=lattice, tau_cart_ang=positions_array,
        tau_frac=positions_array @ np.linalg.inv(lattice), atom_labels=tuple(labels),
    )
    return TB2JIsotropicModel(
        exchange=model, magnetic_atom_labels=selected,
        moments_muB=np.asarray(moments, dtype=np.float64)[atom_indices],
        diagnostics={
            "source": str(source), "exchange_projection": "isotropic_J_iso_only",
            "source_directed_bond_weight": weight, "native_directed_bond_weight": 0.5,
            "J_source_to_native_factor": weight / 0.5, "mate_residual_meV": residual,
            "bond_count": len(bonds), "spin_normalization": "unit_vector",
        },
    )


@dataclass(frozen=True)
class NativeMagnonModes:
    qpoints: NDArray[np.float64]
    energies_eV: NDArray[np.float64]
    transform: NDArray[np.complex128]  # [nq,2*nmag,2*nmag], [a(q),a†(-q)]
    local_frames: NDArray[np.float64]  # ROWS [t1,t2,n]
    spin_lengths: NDArray[np.float64]
    magnetic_atom_labels: tuple[str, ...]
    magnetic_atom_indices: NDArray[np.int64]
    magnetic_site_positions: NDArray[np.float64]
    lattice_ang: NDArray[np.float64]
    gauge: str
    diagnostics: dict[str, object]


def native_magnon_modes(
    exchange_path: str | Path,
    qpoints: object,
    *,
    spin_lengths: object,
    source_directed_bond_weight: float,
    magnetic_atom_labels: tuple[str, ...] | list[str] | None = None,
    gauge: str = "atomic",
    collinearity_tolerance: float = 1.0e-3,
) -> NativeMagnonModes:
    """Solve SLW isotropic LSWT using TB2J's actual collinear spin direction.

    S is an explicit spin-Hamiltonian input, never inferred by equating a
    Wannier partial moment to 2S.  The normalized first source moment defines
    the axis; all other normalized moments must be parallel/antiparallel.
    Exact AFM Goldstone and unstable modes are rejected by native SLW.
    """
    points = _qpoints(qpoints)
    selected_gauge = _gauge(gauge)
    loaded = load_tb2j_isotropic_model(
        exchange_path, source_directed_bond_weight=source_directed_bond_weight,
        magnetic_atom_labels=magnetic_atom_labels,
    )
    model = loaded.exchange
    moments = loaded.moments_muB
    norms = np.linalg.norm(moments, axis=1)
    if np.any(norms <= np.finfo(float).eps) or not np.all(np.isfinite(norms)):
        raise ValueError("every magnetic atom needs a finite nonzero source moment")
    directions = moments / norms[:, None]
    axis = directions[0]
    pattern = np.sign(directions @ axis)
    collinearity = float(np.max(np.linalg.norm(directions - pattern[:, None] * axis, axis=1)))
    if not np.isfinite(collinearity_tolerance) or collinearity_tolerance < 0:
        raise ValueError("collinearity_tolerance must be finite and nonnegative")
    if collinearity > collinearity_tolerance:
        raise ValueError(f"source moments are not collinear: residual={collinearity:.6g}")
    order = "fm" if np.all(pattern == pattern[0]) else "collinear_afm"
    configuration = screen_magnetic_configuration(
        model, order=order, spin_pattern=pattern,
        spin_magnitudes=spin_lengths, quantization_axis=axis,
    )
    spectrum = solve_isotropic_lswt(model, configuration, points)
    minus_spectrum = solve_isotropic_lswt(model, configuration, -points)
    nmag = model.n_magnetic_sites
    transform = np.zeros((points.shape[0], 2*nmag, 2*nmag), dtype=np.complex128)
    if spectrum.transformation.shape[1] == nmag:
        # The FM native solver returns only particles.
        transform[:, :nmag, :nmag] = spectrum.transformation
        transform[:, nmag:, nmag:] = minus_spectrum.transformation.conj()
        original_dynamic = None
    else:
        # Independent negative-eigenvalue columns carry unrelated mode
        # phases.  The physical hole modes are the conjugate particle modes
        # at -q with their two Nambu sectors exchanged.
        transform[:, :, :nmag] = spectrum.transformation[:, :, :nmag]
        swapped_rows = np.r_[np.arange(nmag, 2*nmag), np.arange(nmag)]
        transform[:, :, nmag:] = minus_spectrum.transformation[:, swapped_rows, :nmag].conj()
        original_dynamic = (
            spectrum.transformation * spectrum.signed_energies_mev[:, None, :]
        ) @ np.linalg.inv(spectrum.transformation)
    positions = np.asarray(model.tau_frac)[model.magnetic_atom_indices]
    if selected_gauge == "atomic":
        site_phase = np.exp(-2j * np.pi * (points @ positions.T))
        nambu_phase = np.concatenate((site_phase, site_phase), axis=1)
        transform *= nambu_phase[:, :, None]
        if original_dynamic is not None:
            original_dynamic = nambu_phase[:, :, None] * original_dynamic * nambu_phase[:, None, :].conj()
    paired_eigen_residual = max(spectrum.max_eigen_residual_mev, minus_spectrum.max_eigen_residual_mev)
    if original_dynamic is not None:
        paired_signed_energies = np.concatenate(
            (spectrum.physical_energies_mev, -minus_spectrum.physical_energies_mev), axis=1,
        )
        paired_eigen_residual = float(np.max(np.abs(
            original_dynamic @ transform - transform * paired_signed_energies[:, None, :]
        )))
    paired_paraunitarity_residual = max(paraunitarity_residual(t) for t in transform)
    if paired_eigen_residual > 1.0e-8 or paired_paraunitarity_residual > 1.0e-9:
        raise ValueError(
            "particle-hole pairing invalidated magnon modes: "
            f"eigen residual={paired_eigen_residual:.6g} meV, "
            f"paraunitarity residual={paired_paraunitarity_residual:.6g}"
        )
    energies = spectrum.physical_energies_mev * 1.0e-3
    if np.any(energies <= 0):
        raise ValueError("zero or unstable magnon modes need separate treatment before vertex projection")
    diagnostics = dict(loaded.diagnostics)
    diagnostics.update({
        "source_moments_muB": moments.tolist(), "magnetic_order": order,
        "source_collinearity_residual": collinearity,
        "quantization_axis": axis.tolist(), "spin_pattern": pattern.tolist(),
        "lswt_engine": "slw.magph.lswt.solve_isotropic_lswt",
        "max_hermiticity_residual_meV": spectrum.max_hermiticity_residual_mev,
        "max_eigen_residual_meV": paired_eigen_residual,
        "max_paraunitarity_residual": paired_paraunitarity_residual,
        "qpair_mode_gauge": "T(q)_hole=swap_particle_hole(conj(T(-q)_particle))",
        "minimum_energy_eV": float(energies.min()),
        "fourier_phase": "exp(+i2pi_q_dot_R_plus_tau)" if gauge == "atomic" else "exp(+i2pi_q_dot_R)",
        "frame_convention": "rows_t1_t2_n",
        "nambu_order": "a(q),a_dagger(-q)",
    })
    return NativeMagnonModes(
        qpoints=points, energies_eV=energies, transform=transform,
        local_frames=np.swapaxes(local_spin_frames(configuration), 1, 2),
        spin_lengths=configuration.spin_magnitudes,
        magnetic_atom_labels=loaded.magnetic_atom_labels,
        magnetic_atom_indices=model.magnetic_atom_indices,
        magnetic_site_positions=positions, lattice_ang=np.asarray(model.lattice_ang),
        gauge=selected_gauge, diagnostics=diagnostics,
    )


__all__ = [
    "NativeMagnonModes", "NativePhononModes", "TB2JIsotropicModel",
    "load_tb2j_isotropic_model", "native_magnon_modes", "native_phonon_modes",
    "validate_phonons_against_qe_dyn_xml",
]
