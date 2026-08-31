"""Adapters from native SLW Wannier/EPC outputs to wtorque arrays.

These functions do not guess units, meshes, spin layouts, or material-specific
orbital assignments.  They provide the explicit boundary used by later data
preparation once calculation files are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.core.constants import BOHR_TO_ANG
from slw.core.wannier_io import read_wannier_hr
from slw.epc.epr_io import energy_scale_to_ev

from ..basis import require_complex128
from ..model.exchange_field import split_collinear
from .dfpt import lift_collinear


@dataclass(frozen=True)
class CollinearWannierArrays:
    """Degeneracy-normalized, canonical spinor arrays from two ``hr.dat`` files."""

    r_vectors: NDArray[np.int32]
    h_r: NDArray[np.complex128]
    h_trs_r: NDArray[np.complex128]
    h_xc_r: NDArray[np.complex128]
    h_up_r: NDArray[np.complex128]
    h_down_r: NDArray[np.complex128]
    reference_direction: NDArray[np.float64]


@dataclass(frozen=True)
class CollinearGKQArrays:
    """Canonical interleaved ``g(q,k,pert)`` reconstructed by native SLW EPC."""

    qpoints: NDArray[np.float64]
    kpoints: NDArray[np.float64]
    values: NDArray[np.complex128]
    pert_atom: NDArray[np.int32]
    pert_cart: NDArray[np.int32]
    normalization: str
    units: str


def _normalized_hr(
    path: str | Path,
) -> tuple[list[tuple[int, int, int]], NDArray[np.complex128]]:
    dimension, degeneracies, mapping = read_wannier_hr(path)
    r_vectors = list(mapping)
    if len(degeneracies) != len(r_vectors):
        raise ValueError(
            f"Wannier degeneracy count {len(degeneracies)} does not match "
            f"R count {len(r_vectors)} in {path}"
        )
    blocks = np.empty((len(r_vectors), dimension, dimension), dtype=np.complex128)
    for index, (r_vector, degeneracy) in enumerate(
        zip(r_vectors, degeneracies, strict=True)
    ):
        if int(degeneracy) <= 0:
            raise ValueError(f"Wannier degeneracy must be positive at R={r_vector}")
        blocks[index] = np.asarray(mapping[r_vector], dtype=np.complex128) / int(
            degeneracy
        )
    return r_vectors, blocks


def load_collinear_wannier_hr(
    up_path: str | Path,
    down_path: str | Path,
    *,
    reference_direction: object,
) -> CollinearWannierArrays:
    """Load two common-gauge Wannier90 Hamiltonians through ``slw.core``."""

    up_vectors, up = _normalized_hr(up_path)
    down_vectors, down_raw = _normalized_hr(down_path)
    if set(up_vectors) != set(down_vectors):
        raise ValueError("spin-up and spin-down hr.dat files have different R vectors")
    down_lookup = {
        r_vector: down_raw[index] for index, r_vector in enumerate(down_vectors)
    }
    down = np.stack([down_lookup[r_vector] for r_vector in up_vectors])
    direction = np.asarray(reference_direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("reference_direction must be a finite nonzero 3-vector")
    direction = direction / norm
    h_trs, h_xc = split_collinear(up, down, direction)
    return CollinearWannierArrays(
        r_vectors=np.asarray(up_vectors, dtype=np.int32),
        h_r=np.asarray(h_trs + h_xc, dtype=np.complex128),
        h_trs_r=np.asarray(h_trs, dtype=np.complex128),
        h_xc_r=np.asarray(h_xc, dtype=np.complex128),
        h_up_r=require_complex128("spin-up H_R", up),
        h_down_r=require_complex128("spin-down H_R", down),
        reference_direction=direction,
    )


def _read_slw_gkq(path: str | Path) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64], int]:
    with h5py.File(path, "r") as handle:
        required = ("g_wannier", "k_fracs", "q_fracs")
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"{path} is missing SLW g(k,q) datasets: {missing}")
        values = require_complex128("g_wannier", handle["g_wannier"][...])
        kpoints = np.asarray(handle["k_fracs"][...])
        qpoints = np.asarray(handle["q_fracs"][...])
        nat = int(handle.attrs.get("nat", -1))
    if values.ndim != 5:
        raise ValueError(
            "g_wannier must have shape (nq,nk,norb,norb,3*nat); "
            f"got {values.shape}"
        )
    if values.shape[2] != values.shape[3]:
        raise ValueError("g_wannier orbital axes must be square")
    if kpoints.dtype != np.dtype(np.float64) or kpoints.shape != (values.shape[1], 3):
        raise TypeError("k_fracs must use float64 with shape (nk,3)")
    if qpoints.dtype != np.dtype(np.float64) or qpoints.shape != (values.shape[0], 3):
        raise TypeError("q_fracs must use float64 with shape (nq,3)")
    if nat <= 0 or values.shape[-1] != 3 * nat:
        raise ValueError(
            f"SLW g(k,q) nat={nat} is incompatible with npert={values.shape[-1]}"
        )
    return values, kpoints, qpoints, nat


def load_collinear_slw_gkq(
    up_path: str | Path,
    down_path: str | Path,
    *,
    energy_unit: str,
    displacement_unit: str,
) -> CollinearGKQArrays:
    """Lift two native SLW ``g_wannier`` files into canonical spinor order.

    Values are returned as eV/Angstrom Cartesian derivatives with shape
    ``(nq,nk,3*nat,2*norb,2*norb)``.
    """

    up, kpoints, qpoints, nat = _read_slw_gkq(up_path)
    down, down_kpoints, down_qpoints, down_nat = _read_slw_gkq(down_path)
    if up.shape != down.shape:
        raise ValueError(f"spin-resolved g(k,q) shapes differ: {up.shape} vs {down.shape}")
    if down_nat != nat or not np.allclose(kpoints, down_kpoints, atol=0.0, rtol=0.0):
        raise ValueError("spin-resolved g(k,q) k meshes or atom counts differ")
    if not np.allclose(qpoints, down_qpoints, atol=0.0, rtol=0.0):
        raise ValueError("spin-resolved g(k,q) q meshes differ")
    displacement = str(displacement_unit).strip().lower()
    if displacement in {"angstrom", "ang", "a"}:
        length_to_angstrom = 1.0
    elif displacement in {"bohr", "au", "a.u."}:
        length_to_angstrom = BOHR_TO_ANG
    else:
        raise ValueError(
            "displacement_unit must be angstrom or bohr; "
            f"got {displacement_unit!r}"
        )
    scale = energy_scale_to_ev(energy_unit) / length_to_angstrom
    up_perturbation_first = np.transpose(up, (0, 1, 4, 2, 3)) * scale
    down_perturbation_first = np.transpose(down, (0, 1, 4, 2, 3)) * scale
    lifted = lift_collinear(
        np.asarray(up_perturbation_first, dtype=np.complex128),
        np.asarray(down_perturbation_first, dtype=np.complex128),
    )
    return CollinearGKQArrays(
        qpoints=qpoints,
        kpoints=kpoints,
        values=np.asarray(lifted, dtype=np.complex128),
        pert_atom=np.repeat(np.arange(nat, dtype=np.int32), 3),
        pert_cart=np.tile(np.arange(3, dtype=np.int32), nat),
        normalization="cartesian_derivative",
        units="eV/angstrom",
    )


__all__ = [
    "CollinearGKQArrays",
    "CollinearWannierArrays",
    "load_collinear_slw_gkq",
    "load_collinear_wannier_hr",
]
