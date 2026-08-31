"""Streaming single-q electron-phonon reconstruction from qe2pert EPR.

Only one q point is materialized.  The electron real-space axis is accumulated
on the uniform coarse mesh and transformed with a vectorized inverse FFT,
avoiding the multi-gigabyte ``(nq,nk,nw,nw,npert)`` allocation used by the
legacy all-q converter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.core.constants import BOHR_TO_ANG
from slw.core.qe2pert_ws import init_rvec_images, set_wigner_seitz_cell
from slw.epc.epr_io import energy_scale_to_ev, read_epr_metadata


@dataclass(frozen=True)
class NativeEPRSingleQ:
    q_index: int
    qpoint: NDArray[np.float64]
    kpoints: NDArray[np.float64]
    values: NDArray[np.complex128]
    pert_atom: NDArray[np.int32]
    pert_cart: NDArray[np.int32]
    units: str
    q0_hermiticity_residual: float | None


def _uniform_reduced_grid(mesh: tuple[int, int, int]) -> NDArray[np.float64]:
    n1, n2, n3 = mesh
    return np.asarray(
        [
            (i / n1, j / n2, k / n3)
            for i in range(n1)
            for j in range(n2)
            for k in range(n3)
        ],
        dtype=np.float64,
    )


def _mesh_linear_indices(
    vectors: NDArray[np.integer],
    mesh: tuple[int, int, int],
) -> NDArray[np.int64]:
    dims = np.asarray(mesh, dtype=np.int64)
    reduced = np.mod(np.asarray(vectors, dtype=np.int64), dims[None, :])
    return np.asarray(
        (reduced[:, 0] * dims[1] + reduced[:, 1]) * dims[2] + reduced[:, 2],
        dtype=np.int64,
    )


def _derivative_scale_to_ev_per_angstrom(
    energy_unit: str,
    displacement_unit: str,
) -> float:
    displacement = str(displacement_unit).strip().lower()
    if displacement in {"angstrom", "ang", "a"}:
        length_scale = 1.0
    elif displacement in {"bohr", "au", "a.u."}:
        length_scale = BOHR_TO_ANG
    else:
        raise ValueError(
            f"displacement_unit must be angstrom or bohr; got {displacement_unit!r}"
        )
    return float(energy_scale_to_ev(energy_unit) / length_scale)


def reconstruct_epr_g_at_q(
    epr_path: str | Path,
    q_index: int,
    *,
    energy_unit: str,
    displacement_unit: str,
    expected_spinor: bool = True,
    divide_by_ws_degeneracy: bool = False,
) -> NativeEPRSingleQ:
    """Return Cartesian ``dH/du`` as ``[nk,3*nat,nw,nw]`` in eV/Angstrom.

    ``energy_unit`` and ``displacement_unit`` are mandatory because the legacy
    EPR datasets carry no unit attributes.  No conversion is inferred from a
    filename or material.  ``expected_spinor`` guards the scalar/native-spinor
    boundary explicitly.
    """

    path = Path(epr_path)
    scale = _derivative_scale_to_ev_per_angstrom(
        energy_unit,
        displacement_unit,
    )
    with h5py.File(path, "r") as handle:
        metadata = read_epr_metadata(handle)
        if "basic_data/spinor" not in handle:
            raise KeyError(f"{path} is missing basic_data/spinor")
        declared_spinor = bool(handle["basic_data/spinor"][()])
        if declared_spinor is not bool(expected_spinor):
            expected = int(bool(expected_spinor))
            actual = int(declared_spinor)
            raise ValueError(
                f"{path} declares basic_data/spinor={actual}, expected {expected}"
            )
        if "eph_matrix_wannier" not in handle:
            raise KeyError(f"{path} is missing eph_matrix_wannier")
        group = handle["eph_matrix_wannier"]
        qpoints = _uniform_reduced_grid(metadata.nq_grid)
        iq = int(q_index)
        if iq < 0 or iq >= qpoints.shape[0]:
            raise IndexError(f"q_index {iq} is outside 0..{qpoints.shape[0] - 1}")
        qpoint = qpoints[iq]
        kpoints = _uniform_reduced_grid(metadata.nk_grid)
        nk = kpoints.shape[0]
        npert = 3 * metadata.nat
        values = np.zeros(
            (nk, npert, metadata.nwan, metadata.nwan),
            dtype=np.complex128,
        )
        electron_images = init_rvec_images(metadata.nk_grid, metadata.at)
        phonon_images = init_rvec_images(metadata.nq_grid, metadata.at)
        found = 0

        for ia in range(1, metadata.nat + 1):
            for h5_jw in range(1, metadata.nwan + 1):
                for h5_iw in range(1, metadata.nwan + 1):
                    real_name = f"ep_hop_r_{ia}_{h5_jw}_{h5_iw}"
                    imag_name = f"ep_hop_i_{ia}_{h5_jw}_{h5_iw}"
                    has_real = real_name in group
                    has_imag = imag_name in group
                    if has_real != has_imag:
                        raise KeyError(
                            f"EPR pair has only one complex component: "
                            f"{real_name}, {imag_name}"
                        )
                    if not has_real:
                        continue
                    found += 1
                    stored = np.asarray(
                        group[real_name], dtype=np.float64
                    ) + 1j * np.asarray(group[imag_name], dtype=np.float64)
                    # h5py sees Fortran (3,nre,nrp) as (nrp,nre,3).
                    block = np.asarray(stored.transpose(2, 1, 0), dtype=np.complex128)
                    iw0 = h5_iw - 1
                    jw0 = h5_jw - 1
                    ws_electron = set_wigner_seitz_cell(
                        electron_images,
                        metadata.at,
                        metadata.wc[iw0],
                        metadata.wc[jw0],
                    )
                    ws_phonon = set_wigner_seitz_cell(
                        phonon_images,
                        metadata.at,
                        metadata.wc[iw0],
                        metadata.tau[ia - 1],
                    )
                    if block.shape[1:] != (ws_electron.nr, ws_phonon.nr):
                        raise ValueError(
                            f"WS mismatch ia={ia}, h5_jw={h5_jw}, "
                            f"h5_iw={h5_iw}: EPR={block.shape[1:]}, "
                            f"WS={(ws_electron.nr, ws_phonon.nr)}"
                        )
                    if divide_by_ws_degeneracy:
                        denominator = (
                            ws_electron.ndeg[:, None] * ws_phonon.ndeg[None, :]
                        )
                        block = block / denominator[None, ...]

                    phase_q = np.exp(
                        2j
                        * np.pi
                        * (np.asarray(ws_phonon.vectors, dtype=np.float64) @ qpoint)
                    )
                    # Sum the phonon real-space axis for this q, then collect
                    # periodically equivalent electron vectors on one dense
                    # mesh before the +phase inverse FFT.
                    electron_axis = np.einsum(
                        "arp,p->ra",
                        block,
                        phase_q,
                        optimize=True,
                    )
                    dense_r = np.zeros((nk, 3), dtype=np.complex128)
                    re_index = _mesh_linear_indices(
                        ws_electron.vectors,
                        metadata.nk_grid,
                    )
                    np.add.at(dense_r, re_index, electron_axis)
                    dense_grid = dense_r.reshape(*metadata.nk_grid, 3)
                    values_pair = (
                        np.fft.ifftn(dense_grid, axes=(0, 1, 2)) * nk
                    ).reshape(nk, 3)
                    mode = slice(3 * (ia - 1), 3 * ia)
                    values[:, mode, iw0, jw0] = values_pair * scale

        if found == 0:
            raise KeyError(f"{path} contains no ep_hop complex dataset pairs")

    hermiticity: float | None
    if np.allclose(qpoint, 0.0, atol=0.0, rtol=0.0):
        hermiticity = float(
            np.max(
                np.abs(values - np.swapaxes(values.conj(), -1, -2)),
                initial=0.0,
            )
        )
    else:
        hermiticity = None
    return NativeEPRSingleQ(
        q_index=iq,
        qpoint=np.asarray(qpoint, dtype=np.float64),
        kpoints=kpoints,
        values=values,
        pert_atom=np.repeat(np.arange(metadata.nat, dtype=np.int32), 3),
        pert_cart=np.tile(np.arange(3, dtype=np.int32), metadata.nat),
        units="eV/angstrom",
        q0_hermiticity_residual=hermiticity,
    )


__all__ = ["NativeEPRSingleQ", "reconstruct_epr_g_at_q"]
