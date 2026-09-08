"""Continuous rigid atomic-spin frames from a sampled native AMN polar frame.

This is a specified interpolation model, not new first-principles projection
data. The periodic matrix Q_iα(k) is interpolated using overlaps between the
Wannier center i in cell 0 and trial center α in cell R, then polar-unitarized.
The atomic-position Bloch phases are applied only after this interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from slw.core.qe2pert_ws import init_rvec_images, set_wigner_seitz_cell
from slw.wtorque.gauge.atomic_gauge import atomic_gauge_matrix
from slw.wtorque.model.native_spinor_frame import NativeSpinorFrame
from slw.wtorque.model.time_reversal import atomic_spin_time_reversal


def _dagger(value: np.ndarray) -> np.ndarray:
    return value.conj().swapaxes(-1, -2)


def _points(value: object) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (3,) or not len(result) or not np.all(np.isfinite(result)):
        raise ValueError("kpoints must have nonempty finite shape (nk,3)")
    return result


@dataclass(frozen=True)
class InterpolatedFramePoints:
    """Frame values at unwrapped momenta, with interpolation diagnostics.

    periodic_frame is Q(k). atomic_frame is Q(k) D(k), with the same
    canonical interleaved atomic spin convention as NativeSpinorFrame.
    """

    kpoints: NDArray[np.float64]
    periodic_frame: NDArray[np.complex128]
    atomic_frame: NDArray[np.complex128]
    singular_values: NDArray[np.float64]
    diagnostics: dict[str, Any]


class InterpolatedSpinorFrame:
    """Pair-specific WS interpolation of a full native periodic polar Q(k).

    Matrix elements use ``Q_iα(k)=sum_R exp(+i2pi k.R) Q_iα(R)``.
    Each coarse Fourier coefficient is shared equally among its shortest WS
    images, with row center ``wannier_centers[i]`` and column trial center
    ``coarse_native_frame.orbital_centers[α//2]``. This preserves every coarse
    matrix without choosing one of several equally short boundary images.

    The model is covariant under constant basis rotations within equal-center
    blocks. A new k-dependent Wannier gauge defines a different truncated
    interpolation model; no arbitrary off-grid gauge covariance is asserted.
    Reunitarization preserves the rigid-spin assumption, not missing physical
    spin/projector closure. Diagnostics of the original AMN frame remain valid
    limitations and are retained in ``diagnostics['coarse_frame']``.
    """

    def __init__(
        self,
        coarse_native_frame: NativeSpinorFrame,
        *,
        kmesh: object,
        lattice_column_vectors: object,
        wannier_centers: object,
        rank_tolerance: float = 1.e-8,
    ) -> None:
        raw_mesh = np.asarray(kmesh)
        if (raw_mesh.shape != (3,) or not np.issubdtype(raw_mesh.dtype, np.integer)
                or np.any(raw_mesh < 1)):
            raise ValueError("kmesh must contain three positive integers")
        mesh = raw_mesh.astype(np.int64)
        if not np.isfinite(rank_tolerance) or rank_tolerance <= 0:
            raise ValueError("rank_tolerance must be finite and positive")
        lattice = np.asarray(lattice_column_vectors, dtype=np.float64)
        if (lattice.shape != (3, 3) or not np.all(np.isfinite(lattice))
                or abs(np.linalg.det(lattice)) <= np.finfo(float).eps):
            raise ValueError("lattice_column_vectors must be a finite nonsingular 3x3 matrix")
        k = _points(coarse_native_frame.kpoints)
        frame = np.asarray(coarse_native_frame.atomic_frame, dtype=np.complex128)
        nw = frame.shape[-1] if frame.ndim == 3 else 0
        if (frame.shape != (int(np.prod(mesh)), nw, nw) or nw < 2 or nw % 2
                or len(k) != len(frame) or not np.all(np.isfinite(frame))):
            raise ValueError("coarse atomic_frame must have shape (prod(kmesh),nw,nw), even nw")
        centers = np.asarray(wannier_centers, dtype=np.float64)
        atomic_centers = np.asarray(coarse_native_frame.orbital_centers, dtype=np.float64)
        if (centers.shape != (nw, 3) or atomic_centers.shape != (nw // 2, 3)
                or not np.all(np.isfinite(centers)) or not np.all(np.isfinite(atomic_centers))):
            raise ValueError("Wannier and atomic centers must match the frame dimensions")
        unitary_error = float(np.max(abs(_dagger(frame) @ frame - np.eye(nw))))
        if unitary_error > 1.e-8:
            raise ValueError(f"coarse frame is not unitary: {unitary_error:.6g}")

        integer_k = np.rint(k * mesh).astype(np.int64)
        residual = k - integer_k / mesh
        if np.max(abs(residual - np.rint(residual))) > 1.e-7:
            raise ValueError("coarse kpoints do not lie on the unshifted kmesh")
        integer_k %= mesh
        linear_k = (integer_k[:, 0] * mesh[1] + integer_k[:, 1]) * mesh[2] + integer_k[:, 2]
        if len(np.unique(linear_k)) != len(k):
            raise ValueError("coarse kpoints do not cover the full kmesh exactly once")
        ordered = np.empty_like(frame)
        ordered[linear_k] = frame
        coefficients = (np.fft.fftn(ordered.reshape(*mesh, nw, nw), axes=(0, 1, 2))
                        / len(k)).reshape(len(k), nw, nw)

        images = init_rvec_images(tuple(mesh), lattice)
        spinor_centers = np.repeat(atomic_centers, 2, axis=0)
        cells = []
        cache = {}
        for row in range(nw):
            for column in range(nw):
                key = (tuple(centers[row]), tuple(spinor_centers[column]))
                if key not in cache:
                    cache[key] = set_wigner_seitz_cell(images, lattice, centers[row], spinor_centers[column])
                cells.append(cache[key])
        selected = np.unique(np.concatenate([cell.raw_indices for cell in cells]))
        hopping = np.zeros((len(selected), nw, nw), dtype=np.complex128)
        for pair, cell in enumerate(cells):
            row, column = divmod(pair, nw)
            aliases = cell.vectors % mesh
            indices = (aliases[:, 0] * mesh[1] + aliases[:, 1]) * mesh[2] + aliases[:, 2]
            hopping[np.searchsorted(selected, cell.raw_indices), row, column] = (
                coefficients[indices, row, column] / cell.ndeg
            )
        self.cell_shifts = images.vec[selected].copy()
        self.coefficients = hopping
        self.orbital_centers = atomic_centers.copy()
        self.rank_tolerance = float(rank_tolerance)
        self.dimension = nw
        self.diagnostics = {
            "model": "pair-specific WS Fourier interpolation of periodic AMN polar Q, followed by polar unitarization",
            "kmesh": mesh.tolist(),
            "real_space_vector_count": len(selected),
            "coarse_frame_unitarity_residual": unitary_error,
            "rank_tolerance": self.rank_tolerance,
            "coarse_frame": dict(coarse_native_frame.diagnostics),
            "new_first_principles_projection_data": False,
        }

    def evaluate(self, kpoints: object) -> InterpolatedFramePoints:
        """Evaluate periodic and atomic-gauge frames without a k+q grid map."""

        k = _points(kpoints)
        phase = np.exp(2j * np.pi * (np.remainder(k, 1.) @ self.cell_shifts.T))
        interpolated = (phase @ self.coefficients.reshape(len(self.cell_shifts), -1)).reshape(
            len(k), self.dimension, self.dimension
        )
        left, singular, right = np.linalg.svd(interpolated)
        minimum = float(np.min(singular))
        if minimum < self.rank_tolerance:
            raise ValueError(f"interpolated atomic frame is rank deficient: min singular value {minimum:.6g}")
        periodic = left @ right
        diagnostics = {
            "interpolated_min_singular_value": minimum,
            "interpolated_max_condition_number": float(np.max(singular[:, 0] / singular[:, -1])),
            "polar_correction_max_frobenius": float(np.max(np.linalg.norm(periodic - interpolated, axis=(1, 2)))),
            "frame_unitarity_residual": float(np.max(abs(_dagger(periodic) @ periodic - np.eye(self.dimension)))),
        }
        return InterpolatedFramePoints(
            k.copy(), periodic, periodic @ atomic_gauge_matrix(k, self.orbital_centers),
            singular, diagnostics,
        )

    def periodic_frame(self, kpoints: object) -> NDArray[np.complex128]:
        return self.evaluate(kpoints).periodic_frame

    def sewing(
        self, source: InterpolatedFramePoints, minus: InterpolatedFramePoints,
    ) -> NDArray[np.complex128]:
        """Return B(k)=Q(k) J Q(-k)^T in the original periodic Wannier gauge."""

        if source.kpoints.shape != minus.kpoints.shape:
            raise ValueError("source and minus momenta must have matching shapes")
        mismatch = source.kpoints + minus.kpoints
        if np.max(abs(mismatch - np.rint(mismatch))) > 1.e-7:
            raise ValueError("minus points must represent -k")
        j = atomic_spin_time_reversal(self.dimension, "interleaved")
        return source.periodic_frame @ j @ minus.periodic_frame.swapaxes(-1, -2)

    @staticmethod
    def transform_vertex(
        vertex_wannier: object,
        source: InterpolatedFramePoints,
        final: InterpolatedFramePoints,
        *,
        perturbation_positions: object,
    ) -> NDArray[np.complex128]:
        """Transform g(k,q) with both endpoints and the displacement phase."""

        g = np.asarray(vertex_wannier, dtype=np.complex128)
        positions = np.asarray(perturbation_positions, dtype=np.float64)
        nk, nw, _ = source.atomic_frame.shape
        if (g.ndim != 4 or g.shape[0] != nk or g.shape[-2:] != (nw, nw)
                or final.atomic_frame.shape != source.atomic_frame.shape
                or positions.shape != (g.shape[1], 3)
                or not np.all(np.isfinite(g)) or not np.all(np.isfinite(positions))):
            raise ValueError("vertex/endpoints/perturbation positions have incompatible shapes or values")
        q = final.kpoints - source.kpoints
        if not np.allclose(q, q[0], atol=1.e-10, rtol=0):
            raise ValueError("all vertex endpoints must have one common unwrapped q")
        field_phase = np.exp(2j * np.pi * (positions @ q[0]))
        return (_dagger(final.atomic_frame)[:, None] @ g @ source.atomic_frame[:, None]
                * field_phase[None, :, None, None])


__all__ = ["InterpolatedFramePoints", "InterpolatedSpinorFrame"]
