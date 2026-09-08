"""Full, Cartesian qe2pert coarse vertices with explicit k/q ordering.

The intermediate ``*_elph.h5`` precedes the polar subtraction in qe2pert.
Unlike an EPR short-range Fourier reconstruction it therefore retains the
long-range contribution on the calculated mesh.  HDF5 reverses the Fortran
``(bra,ket,k,q)`` dimensions; the transpose below is part of the format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from slw.epc.epr_io import energy_scale_to_ev
from scipy.constants import physical_constants


def _numbers(text: str | None) -> np.ndarray:
    if text is None:
        raise ValueError("missing numeric XML field")
    return np.fromstring(text.replace("D", "E"), sep=" ")


@dataclass(frozen=True)
class NativeCoarseGrids:
    kpoints: np.ndarray
    qpoints: np.ndarray
    reciprocal_rows: np.ndarray


def validate_native_xml_geometry(
    qe_xml: str | Path, dynamical_xml: list[str | Path], *,
    lattice_ang: object, atom_positions_frac: object, tolerance_ang: float = 1.e-6,
) -> None:
    """Check the ordered physical cell/positions of all side files against EPR."""
    lattice = np.asarray(lattice_ang, float)
    cartesian = np.asarray(atom_positions_frac, float) @ lattice
    bohr_ang = physical_constants["Bohr radius"][0] / 1.e-10
    qe = ET.parse(qe_xml).getroot().find("output/atomic_structure")
    if qe is None:
        raise ValueError("QE XML has no output atomic structure")
    qe_lattice = np.array([_numbers(qe.findtext(f"cell/a{i}")) for i in (1,2,3)]) * bohr_ang
    qe_atoms = np.array([_numbers(atom.text) for atom in qe.findall("atomic_positions/atom")]) * bohr_ang
    geometries = [(str(qe_xml), qe_lattice, qe_atoms)]
    for path in dynamical_xml:
        geometry = ET.parse(path).getroot().find("GEOMETRY_INFO")
        if geometry is None:
            raise ValueError(f"PH XML has no geometry: {path}")
        alat = _numbers(geometry.findtext("CELL_DIMENSIONS"))[0] * bohr_ang
        ph_lattice = _numbers(geometry.findtext("AT")).reshape(3,3) * alat
        nat = int(geometry.findtext("NUMBER_OF_ATOMS"))
        ph_atoms = np.array([_numbers(geometry.find(f"ATOM.{i}").get("TAU")) for i in range(1,nat+1)]) * alat
        geometries.append((str(path), ph_lattice, ph_atoms))
    for name, actual_lattice, actual_cartesian in geometries:
        if (actual_lattice.shape != lattice.shape or actual_cartesian.shape != cartesian.shape
            or not np.allclose(actual_lattice, lattice, atol=tolerance_ang, rtol=0)
            or not np.allclose(actual_cartesian, cartesian, atol=tolerance_ang, rtol=0)):
            raise ValueError(f"native XML/EPR ordered geometry mismatch: {name}")


def read_native_coarse_grids(
    qe_xml: str | Path, dynamical_xml: list[str | Path] | tuple[str | Path, ...],
) -> NativeCoarseGrids:
    """Read NSCF k order and PH star-major q order (file order is explicit)."""
    root = ET.parse(qe_xml).getroot()
    reciprocal = np.array([
        _numbers(root.findtext(f"output/basis_set/reciprocal_lattice/b{i}"))
        for i in (1, 2, 3)
    ])
    if reciprocal.shape != (3, 3) or abs(np.linalg.det(reciprocal)) < 1.e-12:
        raise ValueError("invalid QE reciprocal lattice")
    inverse = np.linalg.inv(reciprocal)
    kcart = np.array([
        _numbers(item.findtext("k_point"))
        for item in root.findall("output/band_structure/ks_energies")
    ])
    qcart = np.array([
        _numbers(item.text)
        for path in dynamical_xml
        for item in ET.parse(path).getroot().iter("Q_POINT")
    ])
    if kcart.ndim != 2 or kcart.shape[1] != 3 or qcart.ndim != 2 or qcart.shape[1] != 3:
        raise ValueError("missing k/q records in QE/PH XML")
    for name, grid in (("k", kcart @ inverse), ("q", qcart @ inverse)):
        delta = grid[:, None] - grid[None]
        distance = np.max(np.abs(delta - np.rint(delta)), axis=-1)
        np.fill_diagonal(distance, np.inf)
        if np.any(distance < 1.e-7):
            raise ValueError(f"duplicate periodic {name} points; check XML file order")
    return NativeCoarseGrids(kcart @ inverse, qcart @ inverse, reciprocal)


def periodic_grid_indices(source: object, target: object, *, tolerance: float = 1.e-7) -> np.ndarray:
    """Return unique source indices for target points modulo reciprocal vectors."""
    original, wanted = np.asarray(source, float), np.asarray(target, float)
    if original.ndim != 2 or original.shape[1] != 3 or wanted.ndim != 2 or wanted.shape[1] != 3:
        raise ValueError("grids must have shape (n,3)")
    delta = wanted[:, None] - original[None]
    residual = np.max(np.abs(delta - np.rint(delta)), axis=-1)
    matches = residual < tolerance
    if np.any(matches.sum(axis=1) != 1):
        raise ValueError("requested points have missing or ambiguous native mesh matches")
    return np.argmax(matches, axis=1).astype(np.int64)


def read_native_coarse_vertex(
    path: str | Path, *, q_index: int, k_indices: object,
    nat: int, num_wann: int, nq: int, nk: int,
    energy_unit: str, displacement_unit: str,
) -> np.ndarray:
    """Return full g in eV/Angstrom, shaped ``(k,atom*cart,bra,ket)``."""
    scale = float(energy_scale_to_ev(energy_unit))
    if displacement_unit == "bohr":
        scale /= physical_constants["Bohr radius"][0] / 1.e-10
    elif displacement_unit != "angstrom":
        raise ValueError("displacement_unit must be bohr or angstrom")
    order = np.asarray(k_indices, dtype=np.int64)
    if order.shape != (nk,) or not np.array_equal(np.sort(order), np.arange(nk)):
        raise ValueError("k_indices must be a permutation of the complete native k mesh")
    if not 0 <= q_index < nq:
        raise ValueError("q_index is outside native q mesh")
    values = np.empty((nk, nat * 3, num_wann, num_wann), complex)
    with h5py.File(path, "r") as handle:
        if "wannier_gauge" not in handle or int(handle["wannier_gauge"][()]) != 1:
            raise ValueError("coarse vertex requires wannier_gauge=1")
        for atom in range(nat):
            for cart in range(3):
                names = [f"elph_{atom + 1}_{cart + 1}_{part}" for part in ("r", "i")]
                if any(name not in handle for name in names):
                    raise ValueError(f"missing complex coarse vertex components: {names}")
                if any(handle[name].shape != (nq, nk, num_wann, num_wann) for name in names):
                    raise ValueError("coarse vertex dimensions disagree with XML/EPR inputs")
                block = handle[names[0]][q_index] + 1j * handle[names[1]][q_index]
                values[:, 3 * atom + cart] = block[order].swapaxes(-1, -2) * scale
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite native coarse vertex")
    return values
