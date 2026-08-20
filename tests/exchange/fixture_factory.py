"""Generated two-site fixtures for exchange integration tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from slw.core.qe2pert_ws import (
    init_rvec_images,
    set_wigner_seitz_cell,
    triangular_pair_index,
)
from slw.core.wannier_io import write_wannier_hr


def _onsite(spin: str) -> tuple[float, float]:
    return (-0.12, -0.10) if spin == "up" else (0.08, 0.10)


def write_toy_epr(path: Path, *, spin: str) -> None:
    """Write the smallest qe2pert-shaped H/g fixture used by all four modes."""

    lattice = np.eye(3, dtype=np.float64)
    positions = np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)))
    mesh = (1, 1, 1)
    electron_images = init_rvec_images(mesh, lattice)
    phonon_images = init_rvec_images(mesh, lattice)
    with h5py.File(path, "w") as handle:
        basic = handle.create_group("basic_data")
        basic.create_dataset("nat", data=2)
        basic.create_dataset("num_wann", data=2)
        basic.create_dataset("kc_dim", data=mesh)
        basic.create_dataset("qc_dim", data=mesh)
        basic.create_dataset("at", data=lattice.T)
        basic.create_dataset("tau", data=positions)
        basic.create_dataset("wannier_center_cryst", data=positions)
        basic.create_dataset("alat", data=10.0)

        electron = handle.create_group("electron_wannier")
        onsite = _onsite(spin)
        for jw in range(1, 3):
            for iw in range(1, jw + 1):
                pair = triangular_pair_index(iw, jw)
                ws = set_wigner_seitz_cell(
                    electron_images,
                    lattice,
                    positions[iw - 1],
                    positions[jw - 1],
                )
                values = np.zeros(ws.nr, dtype=np.float64)
                total = onsite[iw - 1] if iw == jw else 0.025
                values[:] = total / ws.nr
                electron.create_dataset(f"hopping_r{pair}", data=values)
                electron.create_dataset(f"hopping_i{pair}", data=np.zeros_like(values))

        eph = handle.create_group("eph_matrix_wannier")
        for atom in range(1, 3):
            for jw in range(1, 3):
                for iw in range(1, 3):
                    electron_ws = set_wigner_seitz_cell(
                        electron_images,
                        lattice,
                        positions[iw - 1],
                        positions[jw - 1],
                    )
                    phonon_ws = set_wigner_seitz_cell(
                        phonon_images,
                        lattice,
                        positions[iw - 1],
                        positions[atom - 1],
                    )
                    values = np.zeros(
                        (phonon_ws.nr, electron_ws.nr, 3), dtype=np.float64
                    )
                    if iw == jw == atom:
                        derivative = 0.002 if spin == "up" else -0.001
                        values[:, :, 0] = derivative / (
                            phonon_ws.nr * electron_ws.nr
                        )
                    name = f"{atom}_{jw}_{iw}"
                    eph.create_dataset(f"ep_hop_r_{name}", data=values)
                    eph.create_dataset(f"ep_hop_i_{name}", data=np.zeros_like(values))


def write_toy_wannier(root: Path) -> tuple[Path, Path, Path]:
    """Write collinear HR files and an explicit structure-bearing win file."""

    up = root / "toy_up_hr.dat"
    down = root / "toy_dn_hr.dat"
    for path, spin in ((up, "up"), (down, "down")):
        onsite = _onsite(spin)
        matrix = np.asarray(
            ((onsite[0], 0.025), (0.025, onsite[1])), dtype=np.complex128
        )
        write_wannier_hr(path, 2, (1,), {(0, 0, 0): matrix})
    win = root / "toy.win"
    win.write_text(
        """num_wann = 2
begin unit_cell_cart
bohr
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
end unit_cell_cart
begin atoms_frac
Mn 0.0 0.0 0.0
Mn 0.5 0.0 0.0
end atoms_frac
""",
        encoding="utf-8",
    )
    return up, down, win


def common_parameters(up: Path, down: Path) -> dict[str, object]:
    return {
        "input_format": "epr",
        "epr_up": str(up),
        "epr_dn": str(down),
        "efermi": 0.0,
        "kmesh": (1, 1, 1),
        "mag_atoms": (0, 1),
        "slices": "0:0:1,1:1:2",
        "hr_unit": "ev",
        "n_shells": 1,
        "d_max": 3.0,
        "emin": -0.5,
        "empoints": 6,
        "nproc": 1,
    }


__all__ = ["common_parameters", "write_toy_epr", "write_toy_wannier"]
