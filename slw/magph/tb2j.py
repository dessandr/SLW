"""Explicit conversion of projected scalar TB2J exchange HDF5 payloads.

The retained TB2J kernel writes a mate-complete directed bond list with a
source-side bond weight that is not the native magph weight.  This adapter is
deliberately separate from :func:`slw.magph.screening.load_exchange_h5`: the
caller must provide the source weight and spin normalization, and the report
records the resulting numerical conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from .model import (
    ExchangeConvention,
    ExchangeModel,
    ExchangeRepresentation,
    ExchangeSpinNormalization,
)

_CANONICAL_DIRECTED_BOND_WEIGHT = 0.5


@dataclass(frozen=True)
class ProjectedScalarTB2JReport:
    """Auditable provenance for one TB2J-to-native scalar conversion."""

    source: Path
    source_dataset: str
    source_directed_bond_weight: float
    canonical_directed_bond_weight: float
    exchange_scale: float
    spin_normalization: ExchangeSpinNormalization
    source_spin_magnitude: float
    bond_count: int
    magnetic_atom_indices: tuple[int, ...]
    maximum_source_reciprocity_residual_mev: float
    scalar_spin_group_projected: bool
    source_weight_origin: str


def _decode_scalar(value: Any, *, label: str) -> str:
    raw = np.asarray(value)
    if raw.size != 1:
        raise ValueError(f"{label} must contain one scalar value")
    item = raw.reshape(()).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _integer_dataset(
    handle: h5py.File,
    path: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.int64]:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        raise KeyError(f"projected TB2J input is missing {path}")
    raw = np.asarray(handle[path])
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{path} must contain real integers")
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.array_equal(
        numeric, np.rint(numeric)
    ):
        raise ValueError(f"{path} must contain finite integers")
    result = np.asarray(numeric, dtype=np.int64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{path} shape {result.shape} != {shape}")
    return result


def _optional_real_dataset(
    handle: h5py.File,
    path: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.float64] | None:
    if path not in handle:
        return None
    raw = np.asarray(handle[path])
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{path} must contain real numbers")
    result = np.asarray(raw, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{path} shape {result.shape} != {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{path} contains non-finite values")
    return result


def _magnetic_atom_map(
    bond_i: NDArray[np.int64],
    bond_j: NDArray[np.int64],
    atom_i: NDArray[np.int64],
    atom_j: NDArray[np.int64],
) -> NDArray[np.int64]:
    if np.any(bond_i < 0) or np.any(bond_j < 0):
        raise ValueError("local magnetic bond indices must be non-negative")
    site_count = int(max(np.max(bond_i), np.max(bond_j))) + 1
    magnetic_atoms = np.full(site_count, -1, dtype=np.int64)
    for local_values, atom_values in ((bond_i, atom_i), (bond_j, atom_j)):
        for local in range(site_count):
            selected = np.unique(atom_values[local_values == local])
            if selected.size == 0:
                continue
            if selected.size != 1:
                raise ValueError(
                    f"local magnetic site {local} maps to multiple global atoms"
                )
            atom = int(selected[0])
            if magnetic_atoms[local] not in {-1, atom}:
                raise ValueError(
                    f"inconsistent global atom mapping for local site {local}"
                )
            magnetic_atoms[local] = atom
    if np.any(magnetic_atoms < 0) or np.unique(magnetic_atoms).size != site_count:
        raise ValueError("local magnetic sites do not map bijectively to global atoms")
    return magnetic_atoms


def _read_atom_labels(handle: h5py.File) -> tuple[str, ...]:
    if "basic_data/atom_labels" not in handle:
        return ()
    raw = np.asarray(handle["basic_data/atom_labels"]).reshape(-1)
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in raw
    )


def load_projected_scalar_tb2j_h5(
    path: str | Path,
    *,
    source_directed_bond_weight: float,
    spin_normalization: ExchangeSpinNormalization | str,
    source_spin_magnitude: float,
    reciprocity_tolerance_mev: float = 1.0e-8,
) -> tuple[ExchangeModel, ProjectedScalarTB2JReport]:
    """Load ``J_iso_r`` and convert an explicit TB2J bond-count convention.

    The converted values satisfy

    ``J_native * 0.5 = J_source * source_directed_bond_weight``.

    No tensor dataset is consumed by this scalar-only adapter.  The source
    file must identify the TB2J kernel and a completed scalar spin-group
    projection; these conditions prevent an anisotropic tensor from being
    truncated accidentally.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"projected TB2J exchange file not found: {source}")
    weight = float(source_directed_bond_weight)
    magnitude = float(source_spin_magnitude)
    tolerance = float(reciprocity_tolerance_mev)
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("source_directed_bond_weight must be positive and finite")
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("source_spin_magnitude must be positive and finite")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("reciprocity_tolerance_mev must be finite and non-negative")
    normalization = ExchangeSpinNormalization(spin_normalization)
    scale = weight / _CANONICAL_DIRECTED_BOND_WEIGHT

    with h5py.File(source, "r") as handle:
        projected = bool(int(handle.attrs.get("scalar_spin_group_projected", 0)))
        if not projected:
            raise ValueError(
                "scalar TB2J adapter requires scalar_spin_group_projected=1"
            )
        if "basic_data/kernel" not in handle:
            raise KeyError("projected TB2J input is missing basic_data/kernel")
        kernel = _decode_scalar(
            handle["basic_data/kernel"][()], label="basic_data/kernel"
        ).strip().lower()
        if kernel != "tb2j":
            raise ValueError(
                f"scalar TB2J adapter requires basic_data/kernel='tb2j'; got {kernel!r}"
            )
        if "basic_data/unit" not in handle:
            raise KeyError("projected TB2J input is missing basic_data/unit")
        unit = _decode_scalar(
            handle["basic_data/unit"][()], label="basic_data/unit"
        ).strip().lower()
        if unit != "mev":
            raise ValueError(f"projected scalar exchange must use meV; got {unit!r}")
        if "J_iso_r" not in handle or not isinstance(handle["J_iso_r"], h5py.Dataset):
            raise KeyError("projected TB2J input is missing J_iso_r")
        source_values = np.asarray(handle["J_iso_r"], dtype=np.float64)
        if source_values.ndim != 1 or source_values.size == 0:
            raise ValueError("J_iso_r must have nonempty shape (n_bond,)")
        if not np.all(np.isfinite(source_values)):
            raise ValueError("J_iso_r contains non-finite values")
        bond_count = int(source_values.size)
        atom_i = _integer_dataset(
            handle, "bonds/mag_i_atom", shape=(bond_count,)
        )
        atom_j = _integer_dataset(
            handle, "bonds/mag_j_atom", shape=(bond_count,)
        )
        cell_shift = _integer_dataset(handle, "bonds/R", shape=(bond_count, 3))
        bond_i = _integer_dataset(
            handle, "bonds/mag_i_local", shape=(bond_count,)
        )
        bond_j = _integer_dataset(
            handle, "bonds/mag_j_local", shape=(bond_count,)
        )
        mirror = _integer_dataset(
            handle, "bonds/mirror_index", shape=(bond_count,)
        )
        magnetic_atoms = _magnetic_atom_map(bond_i, bond_j, atom_i, atom_j)
        if np.any(mirror < 0) or np.any(mirror >= bond_count):
            raise ValueError("bonds/mirror_index contains an out-of-range index")
        source_residual = float(
            np.max(np.abs(source_values - source_values[mirror]), initial=0.0)
        )
        if source_residual > tolerance:
            raise ValueError(
                "projected scalar TB2J exchange violates mate reciprocity: "
                f"maximum residual={source_residual:.6g} meV"
            )
        if "basic_data/directed_bond_weight" in handle:
            declared_weight = float(
                np.asarray(handle["basic_data/directed_bond_weight"]).reshape(())
            )
            if not np.isclose(declared_weight, weight, rtol=0.0, atol=0.0):
                raise ValueError(
                    "source directed-bond weight conflicts with the file: "
                    f"input={weight:g}, file={declared_weight:g}"
                )
            weight_origin = "file_and_explicit_input"
        else:
            weight_origin = "explicit_input"
        distance = _optional_real_dataset(
            handle, "bonds/distance_ang", shape=(bond_count,)
        )
        lattice = _optional_real_dataset(
            handle, "basic_data/lattice_ang", shape=(3, 3)
        )
        tau_frac = _optional_real_dataset(handle, "basic_data/tau_frac")
        tau_cart = _optional_real_dataset(handle, "basic_data/tau_cart_ang")
        labels = _read_atom_labels(handle)

    converted = np.asarray(source_values * scale, dtype=np.float64)
    convention = ExchangeConvention(
        spin_normalization=normalization,
        source_spin_magnitude=magnitude,
        kernel_family="scalar_lkag",
    )
    model = ExchangeModel(
        source=source,
        representation=ExchangeRepresentation.ISOTROPIC,
        source_dataset="J_iso_r[explicit_tb2j_weight_conversion]",
        convention=convention,
        isotropic_mev=converted,
        tensor_mev=converted[:, None, None] * np.eye(3, dtype=np.float64),
        bond_i=bond_i,
        bond_j=bond_j,
        bond_i_atom=atom_i,
        bond_j_atom=atom_j,
        cell_shift=cell_shift,
        magnetic_atom_indices=magnetic_atoms,
        mirror_index=mirror,
        distance_ang=distance,
        lattice_ang=lattice,
        tau_frac=tau_frac,
        tau_cart_ang=tau_cart,
        atom_labels=labels,
    )
    report = ProjectedScalarTB2JReport(
        source=source,
        source_dataset="J_iso_r",
        source_directed_bond_weight=weight,
        canonical_directed_bond_weight=_CANONICAL_DIRECTED_BOND_WEIGHT,
        exchange_scale=scale,
        spin_normalization=normalization,
        source_spin_magnitude=magnitude,
        bond_count=bond_count,
        magnetic_atom_indices=tuple(int(value) for value in magnetic_atoms),
        maximum_source_reciprocity_residual_mev=source_residual,
        scalar_spin_group_projected=projected,
        source_weight_origin=weight_origin,
    )
    return model, report


__all__ = [
    "ProjectedScalarTB2JReport",
    "load_projected_scalar_tb2j_h5",
]
