r"""Atom-resolved circular polarization and phonon chiralization tools.

The phonon cache used by :mod:`slw.magph` stores *physical displacement*
eigenvectors ``u = epsilon / sqrt(M)``.  Consequently all norms and angular
momenta in this module use the mass metric,

.. math::

   \langle u|v\rangle_M = \sum_a M_a u_a^*\cdot v_a,
   \qquad
   \frac{\boldsymbol L}{\hbar}
       = -i\sum_a M_a u_a^*\times u_a.

For a unit analysis axis ``n``, ``i (n x)`` is a Hermitian helicity operator.
Its ``+1``, ``-1``, and zero eigenspaces give coordinate-independent circular
``+``, circular ``-``, and axial projectors.  A real transverse displacement
has equal ``+`` and ``-`` amplitudes and therefore zero angular momentum; the
individual projector weights must not by themselves be interpreted as a net
rotation.

Near a numerical degeneracy, eigenvectors returned by a dynamical-matrix
solver have an arbitrary unitary gauge.  :func:`chiralize_degenerate_subspaces`
diagonalizes the same helicity operator inside each energy-degenerate block.
It returns the full row rotation so that any mode-linear quantity (including a
magnon--phonon vertex) can be transformed with exactly the same convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

COMPONENT_LABELS = ("plus", "minus", "axis")


def _validate_axis(axis: np.ndarray) -> np.ndarray:
    """Return a finite unit vector without modifying the caller's array."""
    if np.iscomplexobj(axis):
        raise ValueError("rotation axis must be real")
    vector = np.asarray(axis, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"rotation axis must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("rotation axis must contain only finite values")
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise ValueError("rotation axis must have nonzero length")
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def _validate_masses(masses: np.ndarray, nat: int) -> np.ndarray:
    """Validate the atom masses used by the physical-displacement convention."""
    if np.iscomplexobj(masses):
        raise ValueError("atomic masses must be real")
    values = np.asarray(masses, dtype=np.float64)
    if values.shape != (nat,):
        raise ValueError(f"masses must have shape ({nat},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("atomic masses must contain only finite values")
    if np.any(values <= 0.0):
        raise ValueError("all atomic masses must be strictly positive")
    return values


def _validate_operator_atom_weights(
    atom_weights: np.ndarray | None,
    nat: int,
) -> np.ndarray:
    """Validate optional weights in a local/staggered helicity operator."""
    if atom_weights is None:
        return np.ones(nat, dtype=np.float64)
    if np.iscomplexobj(atom_weights):
        raise ValueError("helicity-operator atom weights must be real")
    values = np.asarray(atom_weights, dtype=np.float64)
    if values.shape != (nat,):
        raise ValueError(
            f"helicity-operator atom weights must have shape ({nat},), "
            f"got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "helicity-operator atom weights must contain only finite values"
        )
    if not np.any(values != 0.0):
        raise ValueError("at least one helicity-operator atom weight must be nonzero")
    return values


def _validate_displacements(displacements: np.ndarray) -> np.ndarray:
    """Validate an array with arbitrary leading dimensions and trailing (atom,3)."""
    values = np.asarray(displacements, dtype=np.complex128)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError(
            "physical displacement eigenvectors must have shape (..., nat, 3), "
            f"got {values.shape}"
        )
    if values.shape[-2] < 1:
        raise ValueError(
            "physical displacement eigenvectors must contain at least one atom"
        )
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError(
            "physical displacement eigenvectors must contain only finite values"
        )
    return values


def _max_abs(values: np.ndarray) -> float:
    array = np.asarray(values)
    return float(np.max(np.abs(array), initial=0.0))


@dataclass(frozen=True)
class HelicityProjectors:
    """Hermitian Cartesian projectors for rotations about ``axis``.

    ``helicity`` is the Hermitian generator ``i (axis x)``.  The sign is fixed
    so that ``(x + i y)/sqrt(2)`` has helicity ``+1`` for ``axis = z``.
    """

    axis_vector: np.ndarray
    plus: np.ndarray
    minus: np.ndarray
    axial: np.ndarray
    transverse: np.ndarray
    helicity: np.ndarray

    @property
    def stacked(self) -> np.ndarray:
        """Return ``(P_plus, P_minus, P_axis)`` with shape ``(3,3,3)``."""
        return np.stack((self.plus, self.minus, self.axial), axis=0)


def helicity_projectors(axis: np.ndarray) -> HelicityProjectors:
    """Construct circular/axial projectors about an arbitrary Cartesian axis.

    Reversing the supplied axis exchanges ``plus`` and ``minus`` but leaves the
    axial projector unchanged.  A non-unit input axis is normalized.
    """
    unit = _validate_axis(axis)
    nx, ny, nz = unit
    # C @ v = n x v.  i*C is Hermitian because C is real antisymmetric.
    cross_generator = np.asarray(
        [[0.0, -nz, ny], [nz, 0.0, -nx], [-ny, nx, 0.0]],
        dtype=np.float64,
    )
    helicity = 1.0j * cross_generator
    axial = np.outer(unit, unit).astype(np.complex128)
    transverse = np.eye(3, dtype=np.complex128) - axial
    plus = 0.5 * (transverse + helicity)
    minus = 0.5 * (transverse - helicity)
    return HelicityProjectors(
        axis_vector=unit,
        plus=plus,
        minus=minus,
        axial=axial,
        transverse=transverse,
        helicity=helicity,
    )


def project_helicity_components(
    displacements: np.ndarray,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ``(..., nat, 3)`` displacements onto ``+``, ``-``, and axial parts."""
    values = _validate_displacements(displacements)
    projectors = helicity_projectors(axis)
    projected = np.einsum(
        "cij,...aj->c...ai", projectors.stacked, values, optimize=True
    )
    return projected[0], projected[1], projected[2]


@dataclass(frozen=True)
class RotationNormalizationDiagnostics:
    """Per-mode consistency checks for an atomistic helicity decomposition."""

    mass_norm: np.ndarray
    normalization_error: np.ndarray
    relative_reconstruction_error: np.ndarray
    relative_weight_closure_error: np.ndarray
    angular_momentum_identity_error: np.ndarray
    angular_momentum_imaginary_residual: np.ndarray

    def summary(self) -> dict[str, float]:
        """Return maximum errors suitable for logs or serialized reports."""
        return {
            "max_normalization_error": _max_abs(self.normalization_error),
            "max_relative_reconstruction_error": _max_abs(
                self.relative_reconstruction_error
            ),
            "max_relative_weight_closure_error": _max_abs(
                self.relative_weight_closure_error
            ),
            "max_angular_momentum_identity_error": _max_abs(
                self.angular_momentum_identity_error
            ),
            "max_angular_momentum_imaginary_residual": _max_abs(
                self.angular_momentum_imaginary_residual
            ),
        }


@dataclass(frozen=True)
class AtomicRotationDecomposition:
    """Mass-metric atomistic circular decomposition of physical displacements."""

    axis_vector: np.ndarray
    plus: np.ndarray
    minus: np.ndarray
    axial: np.ndarray
    # Last index follows COMPONENT_LABELS = (plus, minus, axis).
    atom_weights: np.ndarray
    atom_weight_fractions: np.ndarray
    weights: np.ndarray
    weight_fractions: np.ndarray
    atom_angular_momentum_over_hbar: np.ndarray
    angular_momentum_over_hbar: np.ndarray
    normalized_angular_momentum_over_hbar: np.ndarray
    angular_momentum_axis_over_hbar: np.ndarray
    normalized_angular_momentum_axis_over_hbar: np.ndarray
    diagnostics: RotationNormalizationDiagnostics

    @property
    def reconstructed(self) -> np.ndarray:
        """Reconstruct the input displacement from the three orthogonal pieces."""
        return self.plus + self.minus + self.axial


def decompose_atomistic_rotation(
    displacements: np.ndarray,
    masses: np.ndarray,
    axis: np.ndarray,
) -> AtomicRotationDecomposition:
    """Resolve physical phonon displacements into atomistic helicity components.

    Parameters
    ----------
    displacements
        Physical displacement eigenvectors ``u = epsilon/sqrt(M)`` with shape
        ``(..., nat, 3)``.  Any number of q-point/mode leading dimensions is
        accepted.
    masses
        Positive masses with shape ``(nat,)`` in the same units used to form
        ``u``.  Only the internally consistent mass metric matters here.
    axis
        Cartesian axis about which helicity is measured; it is normalized.

    Notes
    -----
    For a mass-normalized eigenvector ``sum_a M_a |u_a|^2 = 1``.  Raw weights
    and ``L/hbar`` remain meaningful for non-normalized input, while the
    corresponding normalized fields divide by the measured mass norm.
    """
    values = _validate_displacements(displacements)
    mass = _validate_masses(masses, values.shape[-2])
    projectors = helicity_projectors(axis)
    projected = np.einsum(
        "cij,...aj->c...ai", projectors.stacked, values, optimize=True
    )
    plus, minus, axial = projected[0], projected[1], projected[2]

    mass_broadcast = mass.reshape((1,) * (values.ndim - 2) + (mass.size, 1))
    atom_norm = mass_broadcast[..., 0] * np.sum(np.abs(values) ** 2, axis=-1)
    atom_weights = mass_broadcast[..., 0, None] * np.sum(
        np.abs(np.moveaxis(projected, 0, -2)) ** 2,
        axis=-1,
    )
    weights = np.sum(atom_weights, axis=-2)
    mass_norm = np.sum(atom_norm, axis=-1)

    denominator = mass_norm[..., None]
    weight_fractions = np.divide(
        weights,
        denominator,
        out=np.zeros_like(weights, dtype=np.float64),
        where=denominator > 0.0,
    )
    atom_weight_fractions = np.divide(
        atom_weights,
        mass_norm[..., None, None],
        out=np.zeros_like(atom_weights, dtype=np.float64),
        where=mass_norm[..., None, None] > 0.0,
    )

    angular_complex = -1.0j * mass_broadcast * np.cross(values.conj(), values, axis=-1)
    atom_angular = angular_complex.real
    angular = np.sum(atom_angular, axis=-2)
    normalized_angular = np.divide(
        angular,
        denominator,
        out=np.zeros_like(angular, dtype=np.float64),
        where=denominator > 0.0,
    )
    angular_axis = np.einsum(
        "...i,i->...", angular, projectors.axis_vector, optimize=True
    )
    normalized_angular_axis = np.divide(
        angular_axis,
        mass_norm,
        out=np.zeros_like(angular_axis, dtype=np.float64),
        where=mass_norm > 0.0,
    )

    reconstructed = np.sum(projected, axis=0)
    reconstruction_norm = np.sqrt(
        np.sum(
            mass_broadcast * np.abs(reconstructed - values) ** 2,
            axis=(-2, -1),
        )
    )
    input_norm = np.sqrt(np.maximum(mass_norm, 0.0))
    reconstruction_relative = np.divide(
        reconstruction_norm,
        input_norm,
        out=np.zeros_like(reconstruction_norm, dtype=np.float64),
        where=input_norm > 0.0,
    )
    weight_closure = np.sum(weights, axis=-1) - mass_norm
    relative_weight_closure = np.divide(
        weight_closure,
        mass_norm,
        out=np.zeros_like(weight_closure, dtype=np.float64),
        where=mass_norm > 0.0,
    )
    # With H = i(n x), <u|H|u>_M = -i n.(u* x u).
    helicity_from_weights = weights[..., 0] - weights[..., 1]
    angular_identity = angular_axis - helicity_from_weights
    angular_imaginary = np.max(np.abs(angular_complex.imag), axis=(-2, -1))
    diagnostics = RotationNormalizationDiagnostics(
        mass_norm=mass_norm,
        normalization_error=mass_norm - 1.0,
        relative_reconstruction_error=reconstruction_relative,
        relative_weight_closure_error=relative_weight_closure,
        angular_momentum_identity_error=angular_identity,
        angular_momentum_imaginary_residual=angular_imaginary,
    )
    return AtomicRotationDecomposition(
        axis_vector=projectors.axis_vector,
        plus=plus,
        minus=minus,
        axial=axial,
        atom_weights=atom_weights,
        atom_weight_fractions=atom_weight_fractions,
        weights=weights,
        weight_fractions=weight_fractions,
        atom_angular_momentum_over_hbar=atom_angular,
        angular_momentum_over_hbar=angular,
        normalized_angular_momentum_over_hbar=normalized_angular,
        angular_momentum_axis_over_hbar=angular_axis,
        normalized_angular_momentum_axis_over_hbar=normalized_angular_axis,
        diagnostics=diagnostics,
    )


def phonon_angular_momentum_over_hbar(
    displacements: np.ndarray,
    masses: np.ndarray,
    *,
    normalize: bool = False,
    atom_resolved: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute ``-i sum_a M_a u_a^* x u_a`` for arbitrary leading shapes.

    If ``atom_resolved`` is true, return ``(total, per_atom)``.  With
    ``normalize=True`` both arrays are divided by the total mass norm of each
    mode.  Zero-norm inputs produce zeros rather than NaNs.
    """
    values = _validate_displacements(displacements)
    mass = _validate_masses(masses, values.shape[-2])
    mass_broadcast = mass.reshape((1,) * (values.ndim - 2) + (mass.size, 1))
    per_atom = (-1.0j * mass_broadcast * np.cross(values.conj(), values, axis=-1)).real
    total = np.sum(per_atom, axis=-2)
    if normalize:
        norm = np.sum(mass_broadcast * np.abs(values) ** 2, axis=(-2, -1))
        total = np.divide(
            total,
            norm[..., None],
            out=np.zeros_like(total),
            where=norm[..., None] > 0.0,
        )
        per_atom = np.divide(
            per_atom,
            norm[..., None, None],
            out=np.zeros_like(per_atom),
            where=norm[..., None, None] > 0.0,
        )
    if atom_resolved:
        return total, per_atom
    return total


def group_degenerate_energies(
    energies: np.ndarray,
    *,
    abs_tolerance: float = 1.0e-8,
    rel_tolerance: float = 1.0e-8,
) -> tuple[tuple[int, ...], ...]:
    """Group one mode-energy vector without transitive tolerance chaining.

    A candidate joins a sorted block only when it is close to the lowest-energy
    member of that block.  Thus every pair in a returned block obeys the stated
    absolute/relative span tolerance, unlike adjacent-linkage grouping.
    """
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"energies must be one-dimensional, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("energies must contain only finite values")
    atol = float(abs_tolerance)
    rtol = float(rel_tolerance)
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0.0 or rtol < 0.0:
        raise ValueError("energy tolerances must be finite and nonnegative")
    if values.size == 0:
        return ()

    order = np.argsort(values, kind="stable")
    groups: list[list[int]] = [[int(order[0])]]
    anchor = float(values[order[0]])
    for raw_index in order[1:]:
        index = int(raw_index)
        candidate = float(values[index])
        threshold = atol + rtol * max(abs(anchor), abs(candidate))
        if abs(candidate - anchor) <= threshold:
            groups[-1].append(index)
        else:
            groups.append([index])
            anchor = candidate
    # Output slots are always ascending original mode indices.  This makes the
    # placement of helicity-ordered states deterministic for unsorted input.
    return tuple(tuple(sorted(group)) for group in groups)


def _phase_fix_rows(rows: np.ndarray) -> np.ndarray:
    """Choose a deterministic U(1) phase from the first largest component."""
    output = np.array(rows, dtype=np.complex128, copy=True)
    if output.shape[0] == 0:
        return output
    magnitudes = np.abs(output)
    row_maximum = np.max(magnitudes, axis=1)
    # Circular vectors have exactly tied Cartesian magnitudes.  Tiny solver
    # noise must not switch the phase pivot from x to y and break gauge
    # determinism, so choose the first component within a scale-aware tie band.
    tie_band = 128.0 * np.finfo(np.float64).eps
    eligible = magnitudes >= row_maximum[:, None] * (1.0 - tie_band)
    pivots = np.argmax(eligible, axis=1)
    values = output[np.arange(output.shape[0]), pivots]
    phases = np.ones(values.shape, dtype=np.complex128)
    nonzero = np.abs(values) > np.finfo(np.float64).eps
    phases[nonzero] = np.exp(-1.0j * np.angle(values[nonzero]))
    output *= phases[:, None]
    return output


def _deterministic_row_basis(rows: np.ndarray, tolerance: float) -> np.ndarray:
    """Return a basis-gauge-independent orthonormal basis for a row space."""
    source = np.asarray(rows, dtype=np.complex128)
    count, dimension = source.shape
    if count == 1:
        return _phase_fix_rows(source)

    selected: list[np.ndarray] = []
    # Project canonical Cartesian basis vectors into the physical eigenspace.
    # This depends only on the eigenspace projector, not on the input mode gauge.
    for coordinate in range(dimension):
        candidate = source[:, coordinate].conj() @ source
        for previous in selected:
            candidate -= np.vdot(previous, candidate) * previous
        length = float(np.linalg.norm(candidate))
        if length > tolerance:
            candidate /= length
            candidate = _phase_fix_rows(candidate[None, :])[0]
            selected.append(candidate)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise np.linalg.LinAlgError(
            "failed to construct a deterministic basis for a helicity eigenspace"
        )
    return np.stack(selected, axis=0)


def _canonicalize_helicity_eigenvectors(
    eigenvalues: np.ndarray,
    physical_rows: np.ndarray,
    *,
    eigenvalue_tolerance: float,
) -> np.ndarray:
    """Fix phases and repeated-helicity gauges in physical coordinate space."""
    groups = group_degenerate_energies(
        eigenvalues,
        abs_tolerance=eigenvalue_tolerance,
        rel_tolerance=eigenvalue_tolerance,
    )
    result = np.empty_like(physical_rows)
    basis_tolerance = 64.0 * np.finfo(np.float64).eps * max(1, physical_rows.shape[1])
    for group in groups:
        indices = np.asarray(group, dtype=np.int64)
        result[indices] = _deterministic_row_basis(
            physical_rows[indices], basis_tolerance
        )
    return result


@dataclass(frozen=True)
class DegenerateBlockDiagnostics:
    """Diagnostics for one q-point/energy block."""

    q_index: tuple[int, ...]
    mode_indices: tuple[int, ...]
    energy_min: float
    energy_max: float
    energy_spread: float
    helicity_eigenvalues: tuple[float, ...]
    helicity_offdiagonal_max: float
    energy_offdiagonal_max: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_index": self.q_index,
            "mode_indices": self.mode_indices,
            "energy_min": self.energy_min,
            "energy_max": self.energy_max,
            "energy_spread": self.energy_spread,
            "helicity_eigenvalues": self.helicity_eigenvalues,
            "helicity_offdiagonal_max": self.helicity_offdiagonal_max,
            "energy_offdiagonal_max": self.energy_offdiagonal_max,
        }


@dataclass(frozen=True)
class ChiralizationDiagnostics:
    """Global and blockwise consistency checks for mode chiralization."""

    abs_tolerance: float
    rel_tolerance: float
    orthonormal_tolerance: float
    operator_atom_weights: tuple[float, ...]
    n_qpoints: int
    n_modes: int
    n_degenerate_blocks: int
    n_rotated_modes: int
    max_mass_overlap_error_before: float
    max_mass_overlap_error_after: float
    max_rotation_unitarity_error: float
    max_helicity_offdiagonal_after: float
    max_energy_offdiagonal_after: float
    max_energy_spread: float
    blocks: tuple[DegenerateBlockDiagnostics, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "abs_tolerance": self.abs_tolerance,
            "rel_tolerance": self.rel_tolerance,
            "orthonormal_tolerance": self.orthonormal_tolerance,
            "operator_atom_weights": self.operator_atom_weights,
            "n_qpoints": self.n_qpoints,
            "n_modes": self.n_modes,
            "n_degenerate_blocks": self.n_degenerate_blocks,
            "n_rotated_modes": self.n_rotated_modes,
            "max_mass_overlap_error_before": self.max_mass_overlap_error_before,
            "max_mass_overlap_error_after": self.max_mass_overlap_error_after,
            "max_rotation_unitarity_error": self.max_rotation_unitarity_error,
            "max_helicity_offdiagonal_after": self.max_helicity_offdiagonal_after,
            "max_energy_offdiagonal_after": self.max_energy_offdiagonal_after,
            "max_energy_spread": self.max_energy_spread,
        }


@dataclass(frozen=True)
class ChiralizedPhononModes:
    """Result of helicity diagonalization in energy-degenerate subspaces.

    ``rotations[..., new_mode, old_mode]`` follows the row convention

    ``new_u = einsum('...nm,...mai->...nai', rotations, old_u)``.

    ``energies`` are diagonal expectation values of the original diagonal
    energy matrix.  For an exact degeneracy they equal the original energy;
    the reported off-diagonal energy residual quantifies any approximation
    introduced when a finite grouping tolerance is used.
    """

    original_energies: np.ndarray
    energies: np.ndarray
    displacements: np.ndarray
    rotations: np.ndarray
    helicity_eigenvalues: np.ndarray
    axis_vector: np.ndarray
    operator_atom_weights: np.ndarray
    diagnostics: ChiralizationDiagnostics


def chiralize_degenerate_subspaces(
    energies: np.ndarray,
    displacements: np.ndarray,
    masses: np.ndarray,
    axis: np.ndarray,
    *,
    atom_weights: np.ndarray | None = None,
    abs_tolerance: float = 1.0e-8,
    rel_tolerance: float = 1.0e-8,
    orthonormal_tolerance: float = 1.0e-7,
    helicity_eigenvalue_tolerance: float = 1.0e-10,
) -> ChiralizedPhononModes:
    """Diagonalize atomistic helicity inside each energy-degenerate mode block.

    ``energies`` has shape ``(..., nmode)`` and ``displacements`` must have
    shape ``(..., nmode, nat, 3)`` with identical leading q-point dimensions.
    Singleton blocks are left *exactly unchanged*, including their phase.

    ``atom_weights`` defines the analyzed local operator
    ``sum_a atom_weights[a] * i(axis x)``.  Its default is all ones (total
    phonon angular momentum); zero/one masks select atoms, while signed weights
    resolve staggered local angular momentum that cancels in the cell total.

    The input modes must be orthonormal in the mass metric.  Refusing a badly
    normalized basis is intentional: a non-unitary repair would not be a pure
    rotation of a degenerate eigenspace and could silently change a coupling.
    """
    energy = np.asarray(energies, dtype=np.float64)
    values = _validate_displacements(displacements)
    if energy.ndim < 1:
        raise ValueError("energies must have at least one mode dimension")
    if not np.all(np.isfinite(energy)):
        raise ValueError("energies must contain only finite values")
    if values.ndim < 3 or values.shape[:-2] != energy.shape:
        raise ValueError(
            "displacements must have shape energies.shape + (nat, 3), "
            f"got energies={energy.shape}, displacements={values.shape}"
        )
    mass = _validate_masses(masses, values.shape[-2])
    operator_weights = _validate_operator_atom_weights(atom_weights, values.shape[-2])
    projectors = helicity_projectors(axis)

    atol = float(abs_tolerance)
    rtol = float(rel_tolerance)
    overlap_tol = float(orthonormal_tolerance)
    eigen_tol = float(helicity_eigenvalue_tolerance)
    tolerances = np.asarray([atol, rtol, overlap_tol, eigen_tol])
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0.0):
        raise ValueError("all chiralization tolerances must be finite and nonnegative")

    q_shape = energy.shape[:-1]
    nmode = int(energy.shape[-1])
    nq = int(np.prod(q_shape, dtype=np.int64)) if q_shape else 1
    nat = int(values.shape[-2])
    energy_flat = energy.reshape(nq, nmode)
    value_flat = values.reshape(nq, nmode, nat, 3)
    canonical = value_flat * np.sqrt(mass)[None, None, :, None]
    overlap_before = np.einsum(
        "qmai,qnai->qmn", canonical.conj(), canonical, optimize=True
    )
    identity = np.eye(nmode, dtype=np.complex128)
    overlap_error_before = _max_abs(overlap_before - identity[None, :, :])
    if overlap_error_before > overlap_tol:
        raise ValueError(
            "phonon modes are not orthonormal in the mass metric: "
            f"max|S-I|={overlap_error_before:.6e} exceeds "
            f"orthonormal_tolerance={overlap_tol:.6e}"
        )

    rotations = np.broadcast_to(identity, (nq, nmode, nmode)).copy()
    helicity_values = np.zeros((nq, nmode), dtype=np.float64)
    block_reports: list[DegenerateBlockDiagnostics] = []
    n_rotated_modes = 0

    for iq in range(nq):
        groups = group_degenerate_energies(
            energy_flat[iq], abs_tolerance=atol, rel_tolerance=rtol
        )
        q_index = (
            tuple(int(x) for x in np.unravel_index(iq, q_shape)) if q_shape else ()
        )
        for group in groups:
            indices = np.asarray(group, dtype=np.int64)
            block = canonical[iq, indices]
            operated = np.einsum(
                "ij,maj->mai", projectors.helicity, block, optimize=True
            )
            operated *= operator_weights[None, :, None]
            helicity_matrix = np.einsum(
                "mai,nai->mn", block.conj(), operated, optimize=True
            )
            helicity_matrix = 0.5 * (helicity_matrix + helicity_matrix.conj().T)
            if indices.size == 1:
                helicity_values[iq, indices[0]] = float(helicity_matrix[0, 0].real)
                continue

            eigenvalues, eigenvectors = np.linalg.eigh(helicity_matrix)
            order = np.argsort(-eigenvalues, kind="stable")
            eigenvalues = np.asarray(eigenvalues[order].real, dtype=np.float64)
            # Columns of eigenvectors contain old-basis coefficients.  Rows of
            # `mixed_physical` are the corresponding canonical displacement
            # eigenvectors in Cartesian coordinates.
            initial_mix = eigenvectors[:, order].T
            mixed_physical = initial_mix @ block.reshape(indices.size, -1)
            mixed_physical = _canonicalize_helicity_eigenvectors(
                eigenvalues,
                mixed_physical,
                eigenvalue_tolerance=eigen_tol,
            )
            # Recover the row rotation from the physical states.  Because block
            # is orthonormal, these are exactly their old-basis coefficients.
            mix = mixed_physical @ block.reshape(indices.size, -1).conj().T
            rotations[iq][np.ix_(indices, indices)] = mix
            helicity_values[iq, indices] = eigenvalues
            n_rotated_modes += int(indices.size)

            helicity_after = mix.conj() @ helicity_matrix @ mix.T
            block_energies = energy_flat[iq, indices]
            energy_after = (
                mix.conj() @ np.diag(block_energies.astype(np.complex128)) @ mix.T
            )
            helicity_offdiag = helicity_after - np.diag(np.diag(helicity_after))
            energy_offdiag = energy_after - np.diag(np.diag(energy_after))
            block_reports.append(
                DegenerateBlockDiagnostics(
                    q_index=q_index,
                    mode_indices=tuple(int(x) for x in indices),
                    energy_min=float(np.min(block_energies)),
                    energy_max=float(np.max(block_energies)),
                    energy_spread=float(np.ptp(block_energies)),
                    helicity_eigenvalues=tuple(float(x) for x in eigenvalues),
                    helicity_offdiagonal_max=_max_abs(helicity_offdiag),
                    energy_offdiagonal_max=_max_abs(energy_offdiag),
                )
            )

    rotated_values = np.einsum("qnm,qmai->qnai", rotations, value_flat, optimize=True)
    rotated_canonical = rotated_values * np.sqrt(mass)[None, None, :, None]
    overlap_after = np.einsum(
        "qmai,qnai->qmn",
        rotated_canonical.conj(),
        rotated_canonical,
        optimize=True,
    )
    unitarity = np.einsum("qnm,qpm->qnp", rotations, rotations.conj(), optimize=True)
    rotated_energies = np.einsum(
        "qnm,qm->qn", np.abs(rotations) ** 2, energy_flat, optimize=True
    )

    diagnostics = ChiralizationDiagnostics(
        abs_tolerance=atol,
        rel_tolerance=rtol,
        orthonormal_tolerance=overlap_tol,
        operator_atom_weights=tuple(float(x) for x in operator_weights),
        n_qpoints=nq,
        n_modes=nmode,
        n_degenerate_blocks=len(block_reports),
        n_rotated_modes=n_rotated_modes,
        max_mass_overlap_error_before=overlap_error_before,
        max_mass_overlap_error_after=_max_abs(overlap_after - identity[None, :, :]),
        max_rotation_unitarity_error=_max_abs(unitarity - identity[None, :, :]),
        max_helicity_offdiagonal_after=max(
            (report.helicity_offdiagonal_max for report in block_reports),
            default=0.0,
        ),
        max_energy_offdiagonal_after=max(
            (report.energy_offdiagonal_max for report in block_reports),
            default=0.0,
        ),
        max_energy_spread=max(
            (report.energy_spread for report in block_reports), default=0.0
        ),
        blocks=tuple(block_reports),
    )
    return ChiralizedPhononModes(
        original_energies=np.array(energy, copy=True),
        energies=rotated_energies.reshape(energy.shape),
        displacements=rotated_values.reshape(values.shape),
        rotations=rotations.reshape(q_shape + (nmode, nmode)),
        helicity_eigenvalues=helicity_values.reshape(energy.shape),
        axis_vector=projectors.axis_vector,
        operator_atom_weights=operator_weights,
        diagnostics=diagnostics,
    )


def apply_mode_rotations(values: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Apply returned chiralization rotations to a mode-leading array.

    ``rotations`` has shape ``(..., nmode, nmode)`` and ``values`` must have
    shape ``(..., nmode, *payload_shape)``.  This is the correct linear
    transformation for displacement-linear magnon--phonon amplitudes.
    """
    transform = np.asarray(rotations, dtype=np.complex128)
    array = np.asarray(values)
    if transform.ndim < 2 or transform.shape[-1] != transform.shape[-2]:
        raise ValueError(
            f"rotations must have shape (..., nmode, nmode), got {transform.shape}"
        )
    leading = transform.shape[:-2]
    nmode = transform.shape[-1]
    if array.shape[: len(leading)] != leading:
        raise ValueError(
            f"value leading shape {array.shape} is incompatible with rotations {transform.shape}"
        )
    if array.ndim <= len(leading) or array.shape[len(leading)] != nmode:
        raise ValueError(
            f"values must have mode dimension {nmode} after leading shape {leading}, "
            f"got {array.shape}"
        )
    payload_shape = array.shape[len(leading) + 1 :]
    nq = int(np.prod(leading, dtype=np.int64)) if leading else 1
    flat_transform = transform.reshape(nq, nmode, nmode)
    flat_values = array.reshape(nq, nmode, -1)
    rotated = np.einsum("qnm,qmp->qnp", flat_transform, flat_values, optimize=True)
    return rotated.reshape(leading + (nmode,) + payload_shape)


# Compact aliases for callers that use "circular" rather than "helicity".
circular_projectors = helicity_projectors
decompose_atomic_rotations = decompose_atomistic_rotation


__all__ = [
    "COMPONENT_LABELS",
    "AtomicRotationDecomposition",
    "ChiralizationDiagnostics",
    "ChiralizedPhononModes",
    "DegenerateBlockDiagnostics",
    "HelicityProjectors",
    "RotationNormalizationDiagnostics",
    "apply_mode_rotations",
    "chiralize_degenerate_subspaces",
    "circular_projectors",
    "decompose_atomic_rotations",
    "decompose_atomistic_rotation",
    "group_degenerate_energies",
    "helicity_projectors",
    "phonon_angular_momentum_over_hbar",
    "project_helicity_components",
]
