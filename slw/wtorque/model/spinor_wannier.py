"""Orthonormal spinor-Wannier Fourier model (WT-E01)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import require_complex128
from slw.wtorque.config import BlochGauge
from slw.wtorque.errors import HermiticityError


def fourier_matrix_batch(
    r_vectors: object,
    matrices_r: object,
    k_red: object,
    *,
    orbital_centers: object,
    bloch_gauge: BlochGauge | str,
) -> NDArray[np.complex128]:
    """Vectorized Fourier transform shared by H, H_TRS, and H_XC."""

    vectors = np.asarray(r_vectors, dtype=np.float64)
    matrices = require_complex128("real-space matrices", matrices_r)
    k = np.asarray(k_red, dtype=np.float64)
    centers = np.asarray(orbital_centers, dtype=np.float64)
    scalar = k.ndim == 1
    if scalar:
        k = k[None, :]
    if vectors.ndim != 2 or vectors.shape[1] != 3 or matrices.shape[0] != vectors.shape[0]:
        raise ValueError("R vectors and real-space matrices are incompatible")
    if k.ndim != 2 or k.shape[1] != 3:
        raise ValueError("k_red must have shape (3,) or (nk,3)")
    if matrices.shape[-1] != matrices.shape[-2] or matrices.shape[-1] != 2 * centers.shape[0]:
        raise ValueError("matrix dimension must equal twice the orbital-center count")
    phase_r = np.exp(2j * np.pi * (k @ vectors.T))
    result = np.einsum("kr,rab->kab", phase_r, matrices, optimize=True)
    if BlochGauge(bloch_gauge) is BlochGauge.ATOMIC_POSITION:
        spinor_centers = np.repeat(centers, 2, axis=0)
        orbital_phase = np.exp(2j * np.pi * (k @ spinor_centers.T))
        result = orbital_phase.conj()[:, :, None] * result * orbital_phase[:, None, :]
    output = np.asarray(result, dtype=np.complex128)
    return output[0] if scalar else output


def _real64(name: str, value: object, shape_tail: tuple[int, ...]) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must use float64; got {array.dtype}")
    if array.shape[-len(shape_tail) :] != shape_tail:
        raise ValueError(f"{name} must end in {shape_tail}; got {array.shape}")
    return array


@dataclass(frozen=True)
class SpinorWannierModel:
    """Real-space spinor model in canonical orbital-major spin order.

    Shapes are ``H_R[nR,nw,nw]``, ``R_vectors[nR,3]``, and
    ``orbital_centers[norb,3]``. Energies are eV, lattice vectors Angstrom,
    and centers/recriprocal points are reduced coordinates.
    """

    lattice: NDArray[np.float64]
    R_vectors: NDArray[np.integer]
    H_R: NDArray[np.complex128]
    orbital_centers: NDArray[np.float64]
    orbital_site: NDArray[np.integer]
    orbital_labels: tuple[str, ...]
    kpoints: NDArray[np.float64]
    weights: NDArray[np.float64]
    fermi_energy: float
    bloch_gauge: BlochGauge = BlochGauge.ATOMIC_POSITION
    hermiticity_tolerance: float = 1.0e-10
    onsite_soc_matrix: NDArray[np.complex128] | None = None

    def __post_init__(self) -> None:
        lattice = np.asarray(self.lattice)
        centers = np.asarray(self.orbital_centers)
        kpoints = np.asarray(self.kpoints)
        weights = np.asarray(self.weights)
        h_r = require_complex128("H_R", self.H_R)
        r_vectors = np.asarray(self.R_vectors)
        sites = np.asarray(self.orbital_site)
        if lattice.dtype != np.float64 or lattice.shape != (3, 3):
            raise TypeError("lattice must be float64 with shape (3, 3)")
        if centers.dtype != np.float64 or centers.ndim != 2 or centers.shape[1] != 3:
            raise TypeError("orbital_centers must be float64 with shape (norb, 3)")
        norb = centers.shape[0]
        nw = 2 * norb
        if h_r.ndim != 3 or h_r.shape[1:] != (nw, nw):
            raise ValueError(f"H_R must have shape (nR, {nw}, {nw}); got {h_r.shape}")
        if r_vectors.shape != (h_r.shape[0], 3) or r_vectors.dtype.kind not in "iu":
            raise ValueError("R_vectors must be an integer array with shape (nR, 3)")
        if sites.shape != (norb,) or sites.dtype.kind not in "iu":
            raise ValueError("orbital_site must be an integer array with shape (norb,)")
        if len(self.orbital_labels) != norb:
            raise ValueError("orbital_labels length must equal norb")
        if self.onsite_soc_matrix is not None:
            onsite_soc = require_complex128("onsite_soc_matrix", self.onsite_soc_matrix)
            if onsite_soc.shape != (nw, nw):
                raise ValueError(
                    f"onsite_soc_matrix must have shape {(nw, nw)}; got {onsite_soc.shape}"
                )
            residual = float(
                np.max(np.abs(onsite_soc - onsite_soc.conj().T), initial=0.0)
            )
            if residual > self.hermiticity_tolerance:
                raise HermiticityError(
                    f"onsite SOC Hermiticity residual {residual:.3e} exceeds "
                    f"{self.hermiticity_tolerance:.3e}"
                )
        if kpoints.dtype != np.float64 or kpoints.ndim != 2 or kpoints.shape[1] != 3:
            raise TypeError("kpoints must be float64 with shape (nk, 3)")
        if weights.dtype != np.float64 or weights.shape != (kpoints.shape[0],):
            raise TypeError("weights must be float64 with shape (nk,)")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
            raise ValueError("k-point weights must be non-negative and sum to exactly one")
        object.__setattr__(self, "bloch_gauge", BlochGauge(self.bloch_gauge))
        self._validate_real_space_hermiticity()

    @property
    def norb(self) -> int:
        return int(self.orbital_centers.shape[0])

    @property
    def nw(self) -> int:
        return 2 * self.norb

    @property
    def spinor_centers(self) -> NDArray[np.float64]:
        return np.repeat(self.orbital_centers, 2, axis=0)

    def _validate_real_space_hermiticity(self) -> None:
        lookup = {tuple(int(x) for x in row): index for index, row in enumerate(self.R_vectors)}
        if len(lookup) != len(self.R_vectors):
            raise ValueError("R_vectors contains duplicate lattice vectors")
        worst = 0.0
        missing: list[tuple[int, int, int]] = []
        for index, row in enumerate(self.R_vectors):
            opposite = (-int(row[0]), -int(row[1]), -int(row[2]))
            partner = lookup.get(opposite)
            if partner is None:
                missing.append(opposite)
                continue
            worst = max(worst, float(np.max(np.abs(self.H_R[index] - self.H_R[partner].conj().T))))
        if missing:
            raise HermiticityError(f"H_R is missing {-np.asarray(missing[0])} Hermitian partner")
        if worst > self.hermiticity_tolerance:
            raise HermiticityError(
                f"H_R(R)=H_R(-R)^dagger residual {worst:.3e} exceeds {self.hermiticity_tolerance:.3e}"
            )

    def hamiltonian_batch(self, k_red: object) -> NDArray[np.complex128]:
        """Vectorized Fourier transform using the declared Bloch gauge."""

        return fourier_matrix_batch(
            self.R_vectors,
            self.H_R,
            k_red,
            orbital_centers=self.orbital_centers,
            bloch_gauge=self.bloch_gauge,
        )

    def hamiltonian(self, k_red: object) -> NDArray[np.complex128]:
        return self.hamiltonian_batch(k_red)

    def green(self, k_red: object, z_eV: complex) -> NDArray[np.complex128]:
        hamiltonian = self.hamiltonian_batch(k_red)
        identity = np.eye(self.nw, dtype=np.complex128)
        return np.linalg.inv(complex(z_eV) * identity - hamiltonian)


__all__ = ["SpinorWannierModel", "fourier_matrix_batch"]
