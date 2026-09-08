"""AMN-anchored common atomic frame for native spinor finite-q response.

The polar frame is an explicit rigid atomic-spin approximation, not a claim
that arbitrary MLWFs carry canonical Pauli matrices.  H, exchange, and g are
all transformed with the same full-rank frame. Energies remain eV and Cartesian
derivatives remain eV/Angstrom; all positions and momenta here are reduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import SpinOrder, require_complex128
from slw.wtorque.gauge.atomic_gauge import atomic_gauge_matrix, wrap_hamiltonian
from slw.wtorque.gauge.kq_map import build_kq_map
from slw.wtorque.model.exchange_field import split_time_reversal
from slw.wtorque.model.local_frames import validate_local_frames
from slw.wtorque.model.local_projection import support_residual_norm
from slw.wtorque.model.magnetic_subspace import MagneticSubspace
from slw.wtorque.model.time_reversal import (
    atomic_spin_time_reversal,
    projection_anchored_sewing,
)
from slw.wtorque.torque.vertices import PAULI, finite_q_vertices


def _dagger(value: np.ndarray) -> np.ndarray:
    return np.swapaxes(value.conj(), -1, -2)


def _canonical_columns(dimension: int, order: SpinOrder) -> np.ndarray:
    if order is SpinOrder.INTERLEAVED:
        return np.arange(dimension, dtype=np.int64)
    count = dimension // 2
    return np.column_stack((np.arange(count), np.arange(count, dimension))).ravel()


@dataclass(frozen=True)
class SpinorProjectionMetadata:
    """Paired real Wannier90 trial orbitals, in canonical orbital order."""

    orbital_centers: NDArray[np.float64]
    projection_pairs: NDArray[np.int64]
    orbital_labels: tuple[str, ...]
    spin_order: SpinOrder

    def masks_from_projection_groups(self, groups: object) -> NDArray[np.bool_]:
        """Make site masks from disjoint, zero-based, complete spin pairs."""

        masks = []
        for group in groups:
            selected = np.asarray(group, dtype=np.int64)
            if selected.ndim != 1 or selected.size == 0:
                raise ValueError("each projection group must be a nonempty index list")
            if np.unique(selected).size != selected.size:
                raise ValueError("projection groups must not contain duplicate indices")
            if np.any(selected < 0) or np.any(selected >= self.projection_pairs.size):
                raise ValueError("projection group contains out-of-range indices")
            contained = np.isin(self.projection_pairs, selected)
            if np.any(contained[:, 0] != contained[:, 1]):
                raise ValueError("each magnetic orbital must select both spin projections")
            masks.append(np.all(contained, axis=1))
        return MagneticSubspace.from_masks(np.asarray(masks, dtype=bool)).orbital_masks


def read_spinor_projection_metadata(
    nnkp_path: str | Path,
    *,
    atomic_spin_order: SpinOrder | str,
    tolerance: float = 1.0e-7,
) -> SpinorProjectionMetadata:
    """Read paired real ``spinor_projections`` from a Wannier90 NNKP file.

    The currently supported spin basis is global z, with + then - members in
    the declared interleaved/blocked ordering. Other spin axes require explicit
    orbital-dependent spin rotations and are rejected rather than guessed.
    NNKP positions are reduced, including their supplied cell-image offsets.
    """

    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("projection tolerance must be finite and positive")
    lines = [line.split("!")[0].strip() for line in Path(nnkp_path).read_text().splitlines()]
    lowered = [line.lower() for line in lines]
    try:
        begin = lowered.index("begin spinor_projections")
        end = lowered.index("end spinor_projections", begin)
    except ValueError as exc:
        raise ValueError("NNKP requires a spinor_projections block") from exc
    block = [line for line in lines[begin + 1 : end] if line]
    count = int(block[0])
    if count < 2 or count % 2 or len(block) != 1 + 3 * count:
        raise ValueError("invalid NNKP spinor projection count or record length")
    spatial = []
    spins = []
    for index in range(count):
        first = np.asarray(block[1 + 3 * index].split(), dtype=float)
        second = np.asarray(block[2 + 3 * index].split(), dtype=float)
        spin = np.asarray(block[3 + 3 * index].split(), dtype=float)
        if first.shape != (6,) or second.shape != (7,) or spin.shape != (4,):
            raise ValueError("invalid NNKP spinor projection record")
        if not np.all(np.isfinite(np.concatenate((first, second, spin)))):
            raise ValueError("NNKP projection metadata must be finite")
        if not np.allclose(spin[1:], [0, 0, 1], atol=threshold, rtol=0):
            raise ValueError("NNKP projections require the supported global z spin axis")
        spatial.append(np.concatenate((first, second)))
        spins.append(spin[0])
    spatial = np.asarray(spatial)
    spins = np.asarray(spins)
    order = SpinOrder(atomic_spin_order)
    pairs = _canonical_columns(count, order).reshape(-1, 2)
    if not np.allclose(spatial[pairs[:, 0]], spatial[pairs[:, 1]], atol=threshold, rtol=0):
        raise ValueError("declared spin ordering does not pair identical spatial projections")
    if not np.all(spins[pairs] == np.asarray([1, -1])):
        raise ValueError("declared spin ordering must pair +1 and -1 projections")
    return SpinorProjectionMetadata(
        orbital_centers=np.asarray(spatial[pairs[:, 0], :3], dtype=np.float64),
        projection_pairs=pairs,
        orbital_labels=tuple(
            f"l={int(record[3])},mr={int(record[4])},r={int(record[5])}"
            for record in spatial[pairs[:, 0]]
        ),
        spin_order=order,
    )


@dataclass(frozen=True)
class NativeSpinorFrame:
    """Full electronic Hilbert space in a canonical atomic-position gauge.

    ``atomic_frame[k]`` maps canonical atomic *cell-gauge* coordinates to the
    original Wannier coordinates. ``hamiltonian_eV`` and ``exchange_eV``
    additionally include atomic-position Bloch phases. No band truncation,
    collinearization, or on-site projection is applied to the Green functions.
    """

    kpoints: NDArray[np.float64]
    minus_k_index: NDArray[np.int64]
    atomic_frame: NDArray[np.complex128]
    hamiltonian_eV: NDArray[np.complex128]
    exchange_eV: NDArray[np.complex128]
    orbital_centers: NDArray[np.float64]
    orbital_masks: NDArray[np.bool_]
    magnetic_site_positions: NDArray[np.float64]
    local_frames: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    diagnostics: dict[str, Any]

    def model_exchange_derivative_wannier(
        self, g_q: object, g_minus_q: object, *, q_red: object,
    ) -> NDArray[np.complex128]:
        """Differentiate the frozen-frame TR-odd *model* exchange at finite q.

        In the original periodic Wannier gauge this is
        1/2 [g(k,q) - B(k+q) g(-k,-q)* B(k)†].  Both endpoints matter:
        using the Hamiltonian's same-k sewing on a finite-q vertex is wrong.
        This excludes TR-even scalar/SOC derivatives within the declared
        atomic projection model. It is not a separately resolved QE XC
        potential and does not include derivatives of the AMN frame/sewing.
        Use transform_vertex on the result to obtain the atomic gauge.
        """
        forward = require_complex128("g(k,q)", g_q)
        partner = require_complex128("g(k,-q)", g_minus_q)
        nk, nw, _ = self.atomic_frame.shape
        if (forward.ndim != 4 or forward.shape != partner.shape
                or forward.shape[0] != nk or forward.shape[-2:] != (nw, nw)):
            raise ValueError("g and its -q partner must have matching (nk,npert,nw,nw) shapes")
        if not np.all(np.isfinite(forward)) or not np.all(np.isfinite(partner)):
            raise ValueError("exchange derivative input must be finite")
        mapping = build_kq_map(self.kpoints, q_red)
        j = atomic_spin_time_reversal(nw, SpinOrder.INTERLEAVED)
        sewing = self.atomic_frame @ j @ self.atomic_frame[self.minus_k_index].swapaxes(-1, -2)
        reversed_vertex = (sewing[mapping.indices, None]
                           @ partner[self.minus_k_index].conj()
                           @ sewing.conj().swapaxes(-1, -2)[:, None])
        return np.asarray(.5 * (forward - reversed_vertex), dtype=np.complex128)

    def hamiltonian_at_indices(
        self, indices: object, reciprocal_shift: object
    ) -> NDArray[np.complex128]:
        """Return H(kbar+G), including atomic-gauge reciprocal wrapping."""

        return wrap_hamiltonian(
            self.hamiltonian_eV[np.asarray(indices, dtype=np.int64)],
            reciprocal_shift,
            self.orbital_centers,
        )

    def exchange_at_indices(
        self, indices: object, reciprocal_shift: object
    ) -> NDArray[np.complex128]:
        return wrap_hamiltonian(
            self.exchange_eV[np.asarray(indices, dtype=np.int64)],
            reciprocal_shift,
            self.orbital_centers,
        )

    def transform_vertex(
        self,
        vertex_wannier: object,
        *,
        q_red: object,
        perturbation_positions: object,
    ) -> NDArray[np.complex128]:
        """Map cell-periodic g[k,pert,final,source] to atomic-position gauge.

        The output uses the unwrapped final momentum k+q and atomic-position
        displacement Fourier amplitudes. Thus cell-gauge phonon eigenvectors
        must be multiplied by exp(-2pi i q.tau_atom) before mode projection.
        Both g and a separately supplied g_XC use this identical transform.
        """

        vertex = require_complex128("native Cartesian vertex", vertex_wannier)
        q = np.asarray(q_red, dtype=np.float64)
        positions = np.asarray(perturbation_positions, dtype=np.float64)
        nk, nw, _ = self.atomic_frame.shape
        if vertex.ndim != 4 or vertex.shape[0] != nk or vertex.shape[-2:] != (nw, nw):
            raise ValueError("native vertex must have shape (nk,npert,nw,nw)")
        if positions.shape != (vertex.shape[1], 3) or not np.all(np.isfinite(positions)):
            raise ValueError("perturbation_positions must have finite shape (npert,3)")
        mapping = build_kq_map(self.kpoints, q)
        source = self.atomic_frame @ atomic_gauge_matrix(self.kpoints, self.orbital_centers)
        final = self.atomic_frame[mapping.indices] @ atomic_gauge_matrix(
            self.kpoints + q, self.orbital_centers
        )
        field_phase = np.exp(2j * np.pi * (positions @ q))
        return np.asarray(
            (_dagger(final)[:, None] @ vertex @ source[:, None])
            * field_phase[None, :, None, None],
            dtype=np.complex128,
        )

    def finite_q_vertices(self, q_red: object) -> NDArray[np.complex128]:
        """Return forward torque [nk,nmag,2,final,source] in eV/radian.

        The bubble trace needs the reverse insertion, obtained by Hermitian
        conjugating the last two axes of this result. No individual finite-q
        matrix is forcibly Hermitian.
        """

        mapping = build_kq_map(self.kpoints, q_red)
        return finite_q_vertices(
            self.exchange_eV,
            self.exchange_at_indices(mapping.indices, mapping.G_wrap),
            orbital_masks=self.orbital_masks,
            local_frames=self.local_frames,
            q_red=q_red,
            orbital_centers=self.orbital_centers,
            magnetic_site_positions=self.magnetic_site_positions,
        )


def build_native_spinor_frame(
    hamiltonian_k: object,
    wannier_atomic_overlap: object,
    minus_k_index: object,
    *,
    kpoints: object,
    orbital_centers: object,
    orbital_masks: object,
    magnetic_site_positions: object,
    local_frames: object,
    atomic_spin_order: SpinOrder | str,
    projected_spin_wannier: object | None = None,
    rank_tolerance: float = 1.0e-4,
    projector_tr_tolerance: float = 5.0e-2,
    hermiticity_tolerance_eV: float = 1.0e-7,
) -> NativeSpinorFrame:
    """Construct an auditable full AMN frame for a finite-q magnetic response.

    H and overlap must refer to the same ordered periodic Wannier gauge. All
    trial columns must be supplied, in declared real-orbital spin-pair order.
    The TR-odd Hamiltonian is used as a *model exchange field*. Its extraction
    is not a replacement for a separately calculated XC-only operator.

    Projected SPN Pauli matrices can be supplied for independent physical-spin
    closure and frame diagnostics. They are reported without imposing exact
    Pauli closure on a truncated first-principles Wannier manifold.
    """

    tolerances = (rank_tolerance, projector_tr_tolerance, hermiticity_tolerance_eV)
    if any(not np.isfinite(value) or value <= 0 for value in tolerances):
        raise ValueError("native frame tolerances must be finite and positive")
    h = require_complex128("full Hamiltonian", hamiltonian_k)
    overlap = require_complex128("full Wannier/atomic overlap", wannier_atomic_overlap)
    k = np.asarray(kpoints, dtype=np.float64)
    centers = np.asarray(orbital_centers, dtype=np.float64)
    positions = np.asarray(magnetic_site_positions, dtype=np.float64)
    if h.ndim != 3 or h.shape[-1] != h.shape[-2] or h.shape != overlap.shape:
        raise ValueError("H and full AMN overlap must have matching shape (nk,nw,nw)")
    nk, nw, _ = h.shape
    if nw < 2 or nw % 2 or k.shape != (nk, 3) or centers.shape != (nw // 2, 3):
        raise ValueError("native spinor frame dimensions/positions are incompatible")
    if any(not np.all(np.isfinite(value)) for value in (h, overlap, k, centers, positions)):
        raise ValueError("native spinor frame inputs must be finite")
    minus = np.asarray(minus_k_index, dtype=np.int64)
    if minus.shape != (nk,) or np.any(minus < 0) or np.any(minus >= nk):
        raise ValueError("minus_k_index must contain one valid index per k point")
    residual_k = k + k[minus]
    if np.max(np.abs(residual_k - np.rint(residual_k)), initial=0.0) > 1.0e-7:
        raise ValueError("minus_k_index does not map the supplied k points to -k")
    magnetic = MagneticSubspace.from_masks(orbital_masks)
    frames = validate_local_frames(local_frames)
    if magnetic.orbital_masks.shape[1] != nw // 2:
        raise ValueError("magnetic masks do not span the full atomic orbital dimension")
    if positions.shape != (magnetic.orbital_masks.shape[0], 3) or len(frames) != len(positions):
        raise ValueError("magnetic site positions, masks, and local frames differ in count")
    h_error = float(np.max(np.abs(h - _dagger(h)), initial=0.0))
    if h_error > hermiticity_tolerance_eV:
        raise ValueError(f"native Hamiltonian is not Hermitian: {h_error:.6e} eV")
    order = SpinOrder(atomic_spin_order)
    sewing_atomic = atomic_spin_time_reversal(nw, order)
    projected = projection_anchored_sewing(overlap, minus, sewing_atomic)
    singular = projected.singular_values
    minimum = float(np.min(singular))
    if minimum < rank_tolerance:
        raise ValueError(f"full atomic projection is rank deficient: {minimum:.6e}")
    gram = _dagger(overlap) @ overlap
    gram_theta = sewing_atomic @ gram[minus].conj() @ sewing_atomic.conj().T
    gram_relative = np.linalg.norm((gram - gram_theta).reshape(nk, -1), axis=1) / np.maximum(
        np.linalg.norm(gram.reshape(nk, -1), axis=1), np.finfo(float).tiny
    )
    covariance = float(np.max(gram_relative))
    if covariance > projector_tr_tolerance:
        raise ValueError(
            f"atomic projector TR covariance {covariance:.6e} exceeds {projector_tr_tolerance:.6e}"
        )
    _, exchange_wannier = split_time_reversal(h, h[minus], projected.sewing)
    columns = _canonical_columns(nw, order)
    frame = projected.atomic_frame[:, :, columns]
    gauge_frame = frame @ atomic_gauge_matrix(k, centers)
    h_atomic = np.asarray(_dagger(gauge_frame) @ h @ gauge_frame, dtype=np.complex128)
    exchange = np.asarray(_dagger(gauge_frame) @ exchange_wannier @ gauge_frame, dtype=np.complex128)
    identity = np.eye(nw, dtype=np.complex128)
    diagnostics: dict[str, Any] = {
        "approximation": "full AMN polar frame; rigid atomic spin; TR-odd model exchange; local_partition",
        "projection_min_singular_value": minimum,
        "projection_max_condition_number": float(np.max(singular[:, 0] / singular[:, -1])),
        "projector_tr_covariance_max_relative": covariance,
        "projector_tr_covariance_median_relative": float(np.median(gram_relative)),
        "frame_unitarity_residual": float(np.max(np.abs(_dagger(frame) @ frame - identity))),
        "sewing_theta_square_residual": float(np.max(np.abs(projected.sewing @ projected.sewing[minus].conj() + identity))),
        "hamiltonian_hermiticity_residual_eV": h_error,
        "exchange_unassigned_relative_norm": support_residual_norm(exchange, magnetic.projectors),
        "projected_spin_closure_residual": None,
        "projected_spin_atomic_frame_relative_residual": None,
    }
    if projected_spin_wannier is not None:
        spin = require_complex128("projected Wannier SPN matrices", projected_spin_wannier)
        if spin.shape != (nk, 3, nw, nw) or not np.all(np.isfinite(spin)):
            raise ValueError("projected spin matrices must have finite shape (nk,3,nw,nw)")
        diagnostics["projected_spin_closure_residual"] = float(
            np.max(np.abs(np.sum(spin @ spin, axis=1) - 3 * identity))
        )
        atomic_spin = _dagger(gauge_frame)[:, None] @ spin @ gauge_frame[:, None]
        pauli = np.asarray([np.kron(np.eye(nw // 2), matrix) for matrix in PAULI])
        diagnostics["projected_spin_atomic_frame_relative_residual"] = float(
            np.linalg.norm(atomic_spin - pauli[None]) / np.sqrt(nk * 3 * nw)
        )
    return NativeSpinorFrame(
        kpoints=k.copy(),
        minus_k_index=minus.copy(),
        atomic_frame=frame,
        hamiltonian_eV=h_atomic,
        exchange_eV=exchange,
        orbital_centers=centers.copy(),
        orbital_masks=magnetic.orbital_masks,
        magnetic_site_positions=positions.copy(),
        local_frames=frames,
        singular_values=singular,
        diagnostics=diagnostics,
    )


__all__ = [
    "NativeSpinorFrame",
    "SpinorProjectionMetadata",
    "build_native_spinor_frame",
    "read_spinor_projection_metadata",
]
