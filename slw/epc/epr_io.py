"""Small, dependency-neutral readers for qe2pert EPR metadata."""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.core.constants import RY_TO_EV


@dataclass(frozen=True)
class EPRMetadata:
    nat: int
    nwan: int
    nk_grid: tuple[int, int, int]
    nq_grid: tuple[int, int, int]
    lattice_dimensionless: NDArray[np.float64]
    atom_positions_fractional: NDArray[np.float64]
    wannier_centres_fractional: NDArray[np.float64]

    @property
    def at(self) -> NDArray[np.float64]:
        """Compatibility alias for the qe2pert lattice matrix."""

        return self.lattice_dimensionless

    @property
    def tau(self) -> NDArray[np.float64]:
        """Compatibility alias for fractional atomic positions."""

        return self.atom_positions_fractional

    @property
    def wc(self) -> NDArray[np.float64]:
        """Compatibility alias for fractional Wannier centres."""

        return self.wannier_centres_fractional


def energy_scale_to_ev(unit: str) -> float:
    """Return a finite energy conversion factor to electron-volts."""

    normalized = str(unit).strip().lower()
    if normalized == "ev":
        return 1.0
    if normalized in {"ry", "ryd", "rydberg"}:
        return RY_TO_EV
    if normalized in {"ha", "hartree"}:
        return 2.0 * RY_TO_EV
    raise ValueError(f"Unsupported energy unit {unit!r}; use ev, ry, or ha")


def _positive_mesh(dataset: h5py.Dataset, *, name: str) -> tuple[int, int, int]:
    raw = np.asarray(dataset)
    if raw.shape != (3,) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain exactly three integers; got {raw!r}")
    mesh = (int(raw[0]), int(raw[1]), int(raw[2]))
    if any(value <= 0 for value in mesh):
        raise ValueError(f"{name} must contain positive integers; got {mesh}")
    return mesh


def read_epr_metadata(handle: h5py.File) -> EPRMetadata:
    """Read and validate the shared structural subset of a qe2pert EPR file."""

    required = (
        "basic_data/nat",
        "basic_data/num_wann",
        "basic_data/kc_dim",
        "basic_data/qc_dim",
        "basic_data/at",
        "basic_data/tau",
        "basic_data/wannier_center_cryst",
    )
    missing = [name for name in required if name not in handle]
    if missing:
        raise KeyError("EPR file is missing required datasets: " + ", ".join(missing))

    nat = int(np.asarray(handle["basic_data/nat"]))
    nwan = int(np.asarray(handle["basic_data/num_wann"]))
    if nat <= 0 or nwan <= 0:
        raise ValueError(f"EPR nat and num_wann must be positive; got {nat}, {nwan}")
    nk_grid = _positive_mesh(handle["basic_data/kc_dim"], name="basic_data/kc_dim")
    nq_grid = _positive_mesh(handle["basic_data/qc_dim"], name="basic_data/qc_dim")

    lattice = np.asarray(handle["basic_data/at"], dtype=np.float64).T
    tau_cartesian_dimensionless = np.asarray(handle["basic_data/tau"], dtype=np.float64)
    centres = np.asarray(handle["basic_data/wannier_center_cryst"], dtype=np.float64)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("basic_data/at must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(lattice))) <= np.finfo(np.float64).eps:
        raise ValueError("basic_data/at must be nonsingular")
    if tau_cartesian_dimensionless.shape != (nat, 3) or not np.all(
        np.isfinite(tau_cartesian_dimensionless)
    ):
        raise ValueError(f"basic_data/tau must have finite shape ({nat}, 3)")
    if centres.shape != (nwan, 3) or not np.all(np.isfinite(centres)):
        raise ValueError(
            f"basic_data/wannier_center_cryst must have finite shape ({nwan}, 3)"
        )
    atom_positions_fractional = np.linalg.solve(
        lattice, tau_cartesian_dimensionless.T
    ).T
    for array in (lattice, atom_positions_fractional, centres):
        array.setflags(write=False)
    return EPRMetadata(
        nat=nat,
        nwan=nwan,
        nk_grid=nk_grid,
        nq_grid=nq_grid,
        lattice_dimensionless=lattice,
        atom_positions_fractional=atom_positions_fractional,
        wannier_centres_fractional=centres,
    )


__all__ = ["EPRMetadata", "energy_scale_to_ev", "read_epr_metadata"]
