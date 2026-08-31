"""Direct Hamiltonian reconstruction from legacy qe2pert EPR files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.core.qe2pert_ws import (
    init_rvec_images,
    set_wigner_seitz_cell,
    triangular_pair_index,
)
from slw.epc.epr_io import energy_scale_to_ev, read_epr_metadata


@dataclass(frozen=True)
class NativeEPRHamiltonian:
    """Hamiltonian on the EPR coarse k mesh."""

    kpoints: NDArray[np.float64]
    values: NDArray[np.complex128]
    units: str
    spinor: bool
    hermiticity_residual_eV: float


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


def reconstruct_epr_hamiltonian(
    epr_path: str | Path,
    *,
    energy_unit: str,
    expected_spinor: bool,
    divide_by_ws_degeneracy: bool = False,
) -> NativeEPRHamiltonian:
    """Return ``H(k)`` as ``[nk,nw,nw]`` in eV.

    The legacy EPR datasets do not declare their energy unit.  Callers must
    therefore provide it explicitly.  ``expected_spinor`` prevents silently
    mixing scalar and native-spinor files.
    """

    path = Path(epr_path)
    scale = float(energy_scale_to_ev(energy_unit))
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
        if "electron_wannier" not in handle:
            raise KeyError(f"{path} is missing electron_wannier")

        kpoints = _uniform_reduced_grid(metadata.nk_grid)
        images = init_rvec_images(metadata.nk_grid, metadata.at)
        hamiltonian = np.zeros(
            (kpoints.shape[0], metadata.nwan, metadata.nwan),
            dtype=np.complex128,
        )
        found = 0
        group = handle["electron_wannier"]
        for jw in range(1, metadata.nwan + 1):
            for iw in range(1, jw + 1):
                pair = triangular_pair_index(iw, jw)
                real_name = f"hopping_r{pair}"
                imag_name = f"hopping_i{pair}"
                has_real = real_name in group
                has_imag = imag_name in group
                if has_real != has_imag:
                    raise KeyError(
                        "EPR hopping has only one complex component: "
                        f"{real_name}, {imag_name}"
                    )
                if not has_real:
                    continue
                found += 1
                hopping = np.asarray(
                    group[real_name], dtype=np.float64
                ) + 1j * np.asarray(group[imag_name], dtype=np.float64)
                ws = set_wigner_seitz_cell(
                    images,
                    metadata.at,
                    metadata.wc[iw - 1],
                    metadata.wc[jw - 1],
                )
                if hopping.shape != (ws.nr,):
                    raise ValueError(
                        f"hopping length mismatch pair=({iw},{jw}): "
                        f"EPR={hopping.shape}, WS={(ws.nr,)}"
                    )
                if divide_by_ws_degeneracy:
                    hopping = hopping / ws.ndeg
                phase = np.exp(
                    2j
                    * np.pi
                    * (
                        kpoints
                        @ np.asarray(ws.vectors, dtype=np.float64).T
                    )
                )
                values = (phase @ hopping) * scale
                hamiltonian[:, iw - 1, jw - 1] = values
                if iw != jw:
                    hamiltonian[:, jw - 1, iw - 1] = values.conj()
        if found == 0:
            raise KeyError(f"{path} contains no electron hopping pairs")

    residual = float(
        np.max(
            np.abs(
                hamiltonian
                - np.swapaxes(hamiltonian.conj(), -1, -2)
            ),
            initial=0.0,
        )
    )
    hamiltonian = np.asarray(
        0.5 * (hamiltonian + np.swapaxes(hamiltonian.conj(), -1, -2)),
        dtype=np.complex128,
    )
    return NativeEPRHamiltonian(
        kpoints=kpoints,
        values=hamiltonian,
        units="eV",
        spinor=declared_spinor,
        hermiticity_residual_eV=residual,
    )


__all__ = ["NativeEPRHamiltonian", "reconstruct_epr_hamiltonian"]
