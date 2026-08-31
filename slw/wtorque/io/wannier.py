"""Canonical spinor-Wannier HDF5 loader."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from slw.wtorque.basis import SpinOrder, canonicalize_spin_order, require_complex128
from slw.wtorque.config import BlochGauge, OnsiteSOCConfig
from slw.wtorque.errors import GaugeMismatchError
from slw.wtorque.model.onsite_soc import add_onsite_to_realspace, resolve_onsite_soc
from slw.wtorque.model.spinor_wannier import SpinorWannierModel

_ELECTRON_DATASETS = (
    "lattice",
    "R_vectors",
    "H_R",
    "orbital_centers",
    "orbital_site",
    "orbital_labels",
    "kpoints",
    "weights",
    "fermi_energy",
)


def _require_group_datasets(group: h5py.Group, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in group]
    if missing:
        raise KeyError(f"missing HDF5 datasets in {group.name}: {', '.join(missing)}")


def load_spinor_wannier(
    path: str | Path,
    *,
    bloch_gauge: BlochGauge | str,
    spin_order: SpinOrder | str,
    onsite_soc: OnsiteSOCConfig | None = None,
) -> SpinorWannierModel:
    with h5py.File(path, "r") as handle:
        if "electrons" not in handle:
            raise KeyError("electronic input must contain /electrons")
        group = handle["electrons"]
        _require_group_datasets(group, _ELECTRON_DATASETS)
        centers = np.asarray(group["orbital_centers"][...])
        if centers.dtype != np.float64:
            raise TypeError("/electrons/orbital_centers must use float64")
        norb = centers.shape[0]
        h_r = require_complex128("/electrons/H_R", group["H_R"][...])
        canonical = canonicalize_spin_order(h_r, norb, spin_order)
        onsite_matrix = None
        if onsite_soc is not None:
            onsite_matrix, _ = resolve_onsite_soc(onsite_soc, norb=norb)
            canonical = add_onsite_to_realspace(
                np.asarray(canonical, dtype=np.complex128),
                group["R_vectors"][...],
                onsite_matrix,
            )
        labels = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in group["orbital_labels"][...]
        )
        declared = group.attrs.get("bloch_gauge")
        if declared is None:
            raise GaugeMismatchError("/electrons must declare the bloch_gauge attribute")
        declared_text = declared.decode() if isinstance(declared, bytes) else str(declared)
        if BlochGauge(declared_text) is not BlochGauge(bloch_gauge):
            raise GaugeMismatchError(
                f"file Bloch gauge {declared_text!r} does not match config {BlochGauge(bloch_gauge).value!r}"
            )
        return SpinorWannierModel(
            lattice=np.asarray(group["lattice"][...]),
            R_vectors=np.asarray(group["R_vectors"][...]),
            H_R=np.asarray(canonical, dtype=np.complex128),
            orbital_centers=centers,
            orbital_site=np.asarray(group["orbital_site"][...]),
            orbital_labels=labels,
            kpoints=np.asarray(group["kpoints"][...]),
            weights=np.asarray(group["weights"][...]),
            fermi_energy=float(group["fermi_energy"][()]),
            bloch_gauge=BlochGauge(bloch_gauge),
            onsite_soc_matrix=onsite_matrix,
        )


__all__ = ["load_spinor_wannier"]
