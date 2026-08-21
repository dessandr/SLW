"""Strict exchange-input screening for the native magnon--phonon workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from slw.core.constants import HA_TO_EV, RY_TO_EV

from .model import (
    ExchangeCapability,
    ExchangeConvention,
    ExchangeModel,
    ExchangeRepresentation,
    ExchangeScreeningReport,
    ExchangeSpinNormalization,
    MagneticConfiguration,
    MagneticOrder,
    SingleIonAnisotropy,
)

_SCALAR_DATASETS = (
    "J_r/value",
    "J_iso_r",
    "tb2j_extra/jiso_tb2j",
    "J_iso_tensor_r",
)
_TENSOR_DATASET = "J_tensor_r"
_CONVENTION_STRING_KEYS = (
    "hamiltonian_sign",
    "spin_normalization",
    "bond_coverage",
    "realspace_gauge",
)


def _nonnegative_finite_tolerance(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return result


def _decode_text(value: Any, *, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must contain one string; got shape {array.shape}")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _normalise_energy_unit(raw: Any, *, label: str) -> tuple[str, float]:
    text = _decode_text(raw, label=label).strip().lower().replace(" ", "")
    aliases = {
        "mev": ("meV", 1.0),
        "millielectronvolt": ("meV", 1.0),
        "millielectronvolts": ("meV", 1.0),
        "ev": ("eV", 1.0e3),
        "electronvolt": ("eV", 1.0e3),
        "electronvolts": ("eV", 1.0e3),
        "ry": ("Ry", RY_TO_EV * 1.0e3),
        "rydberg": ("Ry", RY_TO_EV * 1.0e3),
        "rydbergs": ("Ry", RY_TO_EV * 1.0e3),
        "ha": ("Ha", HA_TO_EV * 1.0e3),
        "hartree": ("Ha", HA_TO_EV * 1.0e3),
        "hartrees": ("Ha", HA_TO_EV * 1.0e3),
    }
    if text not in aliases:
        supported = "meV, eV, Ry, or Ha"
        raise ValueError(
            f"Unsupported exchange energy unit {text!r} in {label}; use {supported}"
        )
    return aliases[text]


def _dataset_energy_units(
    handle: h5py.File, energy_paths: tuple[str, ...]
) -> tuple[dict[str, tuple[str, float]], tuple[str, ...]]:
    """Resolve every energy dataset independently without sibling inference."""

    file_declaration: tuple[str, float] | None = None
    if "basic_data/unit" in handle:
        file_declaration = _normalise_energy_unit(
            _require_dataset(handle, "basic_data/unit")[()],
            label="basic_data/unit",
        )

    resolved: dict[str, tuple[str, float]] = {}
    for path in energy_paths:
        dataset = _require_dataset(handle, path)
        dataset_declaration: tuple[str, float] | None = None
        if "unit" in dataset.attrs:
            dataset_declaration = _normalise_energy_unit(
                dataset.attrs["unit"], label=f"{path}.attrs['unit']"
            )
        if dataset_declaration is None and file_declaration is None:
            raise ValueError(
                f"Exchange energy unit is missing for {path}; give that dataset a "
                "unit attribute or provide basic_data/unit"
            )
        if (
            dataset_declaration is not None
            and file_declaration is not None
            and dataset_declaration[1] != file_declaration[1]
        ):
            raise ValueError(
                "Conflicting exchange energy units: "
                f"basic_data/unit={file_declaration[0]}, "
                f"{path}.attrs['unit']={dataset_declaration[0]}"
            )
        effective_declaration = (
            dataset_declaration if dataset_declaration is not None else file_declaration
        )
        if effective_declaration is None:  # Guarded above; narrows the static type.
            raise AssertionError("unreachable missing energy-unit declaration")
        resolved[path] = effective_declaration

    units = tuple(dict.fromkeys(unit for unit, _factor in resolved.values()))
    return resolved, units


def _real_array(dataset: h5py.Dataset, *, label: str) -> NDArray[np.float64]:
    raw = np.asarray(dataset)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{label} must be a real numeric array; got dtype {raw.dtype}")
    if np.iscomplexobj(raw):
        raise ValueError(
            f"{label} must be real-valued; complex exchange is unsupported"
        )
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def _integer_array(
    dataset: h5py.Dataset,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.int64]:
    raw = np.asarray(dataset)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{label} must be an integer-valued numeric array")
    if np.iscomplexobj(raw):
        raise ValueError(f"{label} must be real and integer-valued")
    if np.issubdtype(raw.dtype, np.integer):
        if np.issubdtype(raw.dtype, np.unsignedinteger) and np.any(
            raw > np.iinfo(np.int64).max
        ):
            raise ValueError(f"{label} contains values outside the int64 range")
        array = np.asarray(raw, dtype=np.int64)
    else:
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.rint(numeric)):
            raise ValueError(f"{label} must contain finite integers")
        int64_upper_exclusive = float(2**63)
        if np.any(numeric < -int64_upper_exclusive) or np.any(
            numeric >= int64_upper_exclusive
        ):
            raise ValueError(f"{label} contains values outside the int64 range")
        array = np.asarray(numeric, dtype=np.int64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}; got {array.shape}")
    return array


def _require_dataset(handle: h5py.File, path: str) -> h5py.Dataset:
    if path not in handle:
        raise KeyError(f"Missing required exchange dataset: {path}")
    value = handle[path]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"{path} must be an HDF5 dataset")
    return value


def _read_safe_isotropic_dataset(
    dataset: h5py.Dataset,
    *,
    path: str,
    n_bond: int,
    unit_to_mev: float,
    isotropy_atol_mev: float,
    rtol: float,
) -> NDArray[np.float64]:
    array = _real_array(dataset, label=path) * unit_to_mev
    if array.shape == (n_bond,):
        return array
    if path == "J_iso_tensor_r" and array.shape == (n_bond, 3, 3):
        scalar = np.trace(array, axis1=1, axis2=2) / 3.0
        isotropic_tensor = scalar[:, None, None] * np.eye(3, dtype=np.float64)
        if not np.allclose(array, isotropic_tensor, atol=isotropy_atol_mev, rtol=rtol):
            error = float(np.max(np.abs(array - isotropic_tensor)))
            raise ValueError(
                "J_iso_tensor_r is not an isotropic tensor and cannot be used as "
                f"a scalar exchange alias (max deviation {error:.6g} meV)"
            )
        return scalar
    raise ValueError(f"{path} must have shape ({n_bond},); got {array.shape}")


def _canonical_local_indices(
    handle: h5py.File,
    bond_i_atom: NDArray[np.int64],
    bond_j_atom: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    has_i = "bonds/mag_i_local" in handle
    has_j = "bonds/mag_j_local" in handle
    if has_i != has_j:
        raise KeyError(
            "bonds/mag_i_local and bonds/mag_j_local must either both be present or both be absent"
        )

    if not has_i:
        magnetic_atoms = np.unique(np.concatenate((bond_i_atom, bond_j_atom)))
        derived_global_to_local = {
            int(atom): idx for idx, atom in enumerate(magnetic_atoms)
        }
        bond_i = np.fromiter(
            (derived_global_to_local[int(atom)] for atom in bond_i_atom),
            dtype=np.int64,
            count=bond_i_atom.size,
        )
        bond_j = np.fromiter(
            (derived_global_to_local[int(atom)] for atom in bond_j_atom),
            dtype=np.int64,
            count=bond_j_atom.size,
        )
        return bond_i, bond_j, magnetic_atoms

    n_bond = bond_i_atom.size
    source_i = _integer_array(
        _require_dataset(handle, "bonds/mag_i_local"),
        label="bonds/mag_i_local",
        shape=(n_bond,),
    )
    source_j = _integer_array(
        _require_dataset(handle, "bonds/mag_j_local"),
        label="bonds/mag_j_local",
        shape=(n_bond,),
    )
    if np.any(source_i < 0) or np.any(source_j < 0):
        raise ValueError("source local magnetic-site indices must be non-negative")

    local_to_global: dict[int, int] = {}
    global_to_local: dict[int, int] = {}
    for local, global_atom in zip(
        np.concatenate((source_i, source_j)),
        np.concatenate((bond_i_atom, bond_j_atom)),
    ):
        local_int = int(local)
        global_int = int(global_atom)
        previous_global = local_to_global.setdefault(local_int, global_int)
        if previous_global != global_int:
            raise ValueError(
                f"source local site {local_int} maps to multiple global atoms: "
                f"{previous_global} and {global_int}"
            )
        previous_local = global_to_local.setdefault(global_int, local_int)
        if previous_local != local_int:
            raise ValueError(
                f"global atom {global_int} maps to multiple source local sites: "
                f"{previous_local} and {local_int}"
            )

    source_sites = sorted(local_to_global)
    source_to_canonical = {source: idx for idx, source in enumerate(source_sites)}
    bond_i = np.fromiter(
        (source_to_canonical[int(site)] for site in source_i),
        dtype=np.int64,
        count=n_bond,
    )
    bond_j = np.fromiter(
        (source_to_canonical[int(site)] for site in source_j),
        dtype=np.int64,
        count=n_bond,
    )
    magnetic_atoms = np.asarray(
        [local_to_global[source] for source in source_sites], dtype=np.int64
    )
    return bond_i, bond_j, magnetic_atoms


def _screen_bonds(
    bond_i: NDArray[np.int64],
    bond_j: NDArray[np.int64],
    cell_shift: NDArray[np.int64],
    tensor_mev: NDArray[np.float64],
    representation: ExchangeRepresentation,
    *,
    source_mirror: NDArray[np.int64] | None,
    reciprocity_atol_mev: float,
    rtol: float,
) -> tuple[NDArray[np.int64], float]:
    keys = [
        (int(i), int(j), int(shift[0]), int(shift[1]), int(shift[2]))
        for i, j, shift in zip(bond_i, bond_j, cell_shift)
    ]
    by_key: dict[tuple[int, int, int, int, int], int] = {}
    for index, key in enumerate(keys):
        if key in by_key:
            raise ValueError(
                f"Duplicate directed bond key {key} at indices {by_key[key]} and {index}"
            )
        by_key[key] = index

    mirror = np.empty(len(keys), dtype=np.int64)
    missing: list[tuple[int, int, int, int, int]] = []
    for index, (i, j, rx, ry, rz) in enumerate(keys):
        mate_key = (j, i, -rx, -ry, -rz)
        mate = by_key.get(mate_key)
        if mate is None:
            missing.append(mate_key)
        else:
            mirror[index] = mate
    if missing:
        sample = ", ".join(str(item) for item in missing[:3])
        suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
        raise ValueError(
            f"Exchange bond list is not mate-complete; missing {sample}{suffix}"
        )

    if source_mirror is not None and not np.array_equal(source_mirror, mirror):
        mismatches = np.flatnonzero(source_mirror != mirror)
        first = int(mismatches[0])
        raise ValueError(
            "bonds/mirror_index is inconsistent with (j, i, -R): "
            f"bond {first} declares {int(source_mirror[first])}, expected {int(mirror[first])}"
        )

    max_error = 0.0
    for index, mate_value in enumerate(mirror):
        mate_index = int(mate_value)
        expected = (
            tensor_mev[index].T
            if representation is ExchangeRepresentation.TENSOR
            else tensor_mev[index]
        )
        error = float(np.max(np.abs(tensor_mev[mate_index] - expected)))
        max_error = max(max_error, error)
        if not np.allclose(
            tensor_mev[mate_index], expected, atol=reciprocity_atol_mev, rtol=rtol
        ):
            relation = (
                "J_mate = J.T"
                if representation is ExchangeRepresentation.TENSOR
                else "J_mate = J"
            )
            raise ValueError(
                f"Exchange reciprocity failed for bonds {index} and {mate_index} "
                f"({relation}; max error {error:.6g} meV)"
            )
    return mirror, max_error


def _canonicalize_reciprocal_tensor(
    tensor_mev: NDArray[np.float64],
    mirror: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Average screened mates so the canonical model has exact reciprocity."""

    reciprocal = np.transpose(tensor_mev[mirror], (0, 2, 1))
    return np.asarray(0.5 * (tensor_mev + reciprocal), dtype=np.float64)


def _optional_real_array(
    handle: h5py.File, path: str, *, shape: tuple[int, ...] | None = None
) -> NDArray[np.float64] | None:
    if path not in handle:
        return None
    array = _real_array(_require_dataset(handle, path), label=path)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{path} must have shape {shape}; got {array.shape}")
    return array


def _read_atom_labels(handle: h5py.File) -> tuple[str, ...]:
    if "basic_data/atom_labels" not in handle:
        return ()
    raw = np.asarray(_require_dataset(handle, "basic_data/atom_labels")).reshape(-1)
    labels: list[str] = []
    for value in raw:
        if isinstance(value, bytes):
            labels.append(value.decode("utf-8"))
        else:
            labels.append(str(value))
    return tuple(labels)


def _axis_order(value: Any, *, label: str) -> tuple[str, str, str]:
    array = np.asarray(value)
    if array.size == 1:
        text = _decode_text(value, label=label)
        tokens = tuple(
            item.strip().lower()
            for item in text.replace(";", ",").replace(" ", ",").split(",")
            if item.strip()
        )
    else:
        tokens = tuple(
            _decode_text(item, label=label).strip().lower()
            for item in array.reshape(-1)
        )
    if len(tokens) != 3 or set(tokens) != {"x", "y", "z"}:
        raise ValueError(f"{label} must be a permutation of x,y,z; got {tokens!r}")
    return tokens  # type: ignore[return-value]


def _tensor_axis_order(handle: h5py.File) -> tuple[str, str, str]:
    declarations: list[tuple[str, tuple[str, str, str]]] = []
    if "basic_data/tensor_axes" in handle:
        declarations.append(
            (
                "basic_data/tensor_axes",
                _axis_order(
                    _require_dataset(handle, "basic_data/tensor_axes")[()],
                    label="basic_data/tensor_axes",
                ),
            )
        )
    tensor = _require_dataset(handle, _TENSOR_DATASET)
    if "tensor_axis_order" in tensor.attrs:
        declarations.append(
            (
                "J_tensor_r.attrs['tensor_axis_order']",
                _axis_order(
                    tensor.attrs["tensor_axis_order"],
                    label="J_tensor_r.attrs['tensor_axis_order']",
                ),
            )
        )
    if not declarations:
        raise ValueError(
            "J_tensor_r requires explicit tensor axes in "
            "basic_data/tensor_axes or its tensor_axis_order attribute"
        )
    reference = declarations[0][1]
    if any(axes != reference for _label, axes in declarations[1:]):
        details = ", ".join(f"{label}={axes}" for label, axes in declarations)
        raise ValueError(f"Conflicting tensor axis declarations: {details}")
    return reference


def _optional_scalar_text(handle: h5py.File, path: str) -> str | None:
    if path not in handle:
        return None
    return _decode_text(_require_dataset(handle, path)[()], label=path).strip()


def _read_exchange_convention(
    handle: h5py.File,
    override: ExchangeConvention | None,
) -> tuple[ExchangeConvention, str, tuple[str, ...]]:
    if override is not None and not isinstance(override, ExchangeConvention):
        raise TypeError("convention override must be an ExchangeConvention")

    declared: dict[str, Any] = {}
    for key in _CONVENTION_STRING_KEYS:
        value = _optional_scalar_text(handle, f"basic_data/{key}")
        if value is not None:
            declared[key] = value
    for key in ("directed_bond_weight", "source_spin_magnitude"):
        path = f"basic_data/{key}"
        if path in handle:
            raw = np.asarray(_require_dataset(handle, path))
            if raw.size != 1 or np.iscomplexobj(raw):
                raise ValueError(f"{path} must contain one real number")
            numeric_value = float(raw.reshape(()).item())
            if not np.isfinite(numeric_value):
                raise ValueError(f"{path} must be finite")
            declared[key] = numeric_value

    kernel_declarations: list[tuple[str, str]] = []
    for path in ("basic_data/kernel_family", "basic_data/kernel"):
        value = _optional_scalar_text(handle, path)
        if value is not None:
            kernel = value.strip().lower()
            if kernel == "scalar":
                kernel = "scalar_lkag"
            kernel_declarations.append((path, kernel))
    if kernel_declarations:
        kernels = {value for _path, value in kernel_declarations}
        if len(kernels) != 1:
            details = ", ".join(
                f"{path}={value!r}" for path, value in kernel_declarations
            )
            raise ValueError(f"Conflicting exchange kernel declarations: {details}")
        declared["kernel_family"] = kernel_declarations[0][1]

    required = {
        *_CONVENTION_STRING_KEYS,
        "directed_bond_weight",
        "source_spin_magnitude",
        "kernel_family",
    }
    missing = sorted(required.difference(declared))
    if override is None:
        if missing:
            raise ValueError(
                "Exchange convention metadata is incomplete; missing "
                f"{', '.join('basic_data/' + name for name in missing)}. "
                "Regenerate the exchange file or pass an explicit ExchangeConvention."
            )
        return ExchangeConvention(**declared), "file", ()

    expected = {
        "hamiltonian_sign": override.hamiltonian_sign,
        "spin_normalization": override.spin_normalization.value,
        "bond_coverage": override.bond_coverage,
        "directed_bond_weight": override.directed_bond_weight,
        "realspace_gauge": override.realspace_gauge,
        "source_spin_magnitude": override.source_spin_magnitude,
        "kernel_family": override.kernel_family,
    }
    conflicts: list[str] = []
    for key, declared_value in declared.items():
        expected_value = expected[key]
        matches: bool
        if isinstance(expected_value, float):
            matches = bool(
                np.isclose(float(declared_value), expected_value, rtol=0.0, atol=0.0)
            )
        else:
            matches = (
                str(declared_value).strip().lower()
                == str(expected_value).strip().lower()
            )
        if not matches:
            conflicts.append(
                f"{key}: file={declared_value!r}, override={expected_value!r}"
            )
    if conflicts:
        raise ValueError(
            "Exchange convention override conflicts with file: " + "; ".join(conflicts)
        )
    return (
        override,
        "explicit_override",
        tuple(f"basic_data/{name}" for name in missing),
    )


def load_exchange_h5(
    path: str | Path,
    *,
    reciprocity_atol_mev: float = 1.0e-8,
    consistency_atol_mev: float = 1.0e-8,
    rtol: float = 1.0e-7,
    convention: ExchangeConvention | None = None,
) -> tuple[ExchangeModel, ExchangeScreeningReport]:
    """Load and strictly screen a native/legacy exchange HDF5 payload.

    A scalar source is promoted to ``J_iso * I`` for a uniform internal shape.
    This does not unlock tensor-only capabilities: provenance and capabilities
    are returned in ``ExchangeScreeningReport``.  A full tensor is always kept
    intact; ``isotropic_mev`` is only its trace/3 decomposition, never a
    replacement for ``tensor_mev``.
    """

    reciprocity_atol_mev = _nonnegative_finite_tolerance(
        "reciprocity_atol_mev", reciprocity_atol_mev
    )
    consistency_atol_mev = _nonnegative_finite_tolerance(
        "consistency_atol_mev", consistency_atol_mev
    )
    rtol = _nonnegative_finite_tolerance("rtol", rtol)
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Exchange HDF5 file not found: {source}")
    source = source.resolve()

    with h5py.File(source, "r") as handle:
        (
            exchange_convention,
            convention_origin,
            missing_convention_fields,
        ) = _read_exchange_convention(handle, convention)
        tensor_present = _TENSOR_DATASET in handle
        scalar_paths = tuple(name for name in _SCALAR_DATASETS if name in handle)
        if not tensor_present and not scalar_paths:
            choices = ", ".join((_TENSOR_DATASET, *_SCALAR_DATASETS))
            raise KeyError(
                f"No supported exchange dataset found; expected one of: {choices}"
            )

        bond_i_atom = _integer_array(
            _require_dataset(handle, "bonds/mag_i_atom"), label="bonds/mag_i_atom"
        )
        if bond_i_atom.ndim != 1:
            raise ValueError(
                f"bonds/mag_i_atom must have shape (n_bond,); got {bond_i_atom.shape}"
            )
        n_bond = bond_i_atom.size
        if n_bond == 0:
            raise ValueError("Exchange bond list must not be empty")
        bond_j_atom = _integer_array(
            _require_dataset(handle, "bonds/mag_j_atom"),
            label="bonds/mag_j_atom",
            shape=(n_bond,),
        )
        cell_shift = _integer_array(
            _require_dataset(handle, "bonds/R"),
            label="bonds/R",
            shape=(n_bond, 3),
        )
        if np.any(bond_i_atom < 0) or np.any(bond_j_atom < 0):
            raise ValueError("global atom indices must be non-negative")
        bond_i, bond_j, magnetic_atoms = _canonical_local_indices(
            handle, bond_i_atom, bond_j_atom
        )

        energy_paths = tuple(
            name for name in (_TENSOR_DATASET, *scalar_paths) if name in handle
        )
        energy_units, input_units = _dataset_energy_units(handle, energy_paths)

        scalar_values: list[tuple[str, NDArray[np.float64]]] = []
        for scalar_path in scalar_paths:
            dataset = _require_dataset(handle, scalar_path)
            scalar_values.append(
                (
                    scalar_path,
                    _read_safe_isotropic_dataset(
                        dataset,
                        path=scalar_path,
                        n_bond=n_bond,
                        unit_to_mev=energy_units[scalar_path][1],
                        isotropy_atol_mev=consistency_atol_mev,
                        rtol=rtol,
                    ),
                )
            )
        if scalar_values:
            reference_path, reference_scalar = scalar_values[0]
            for candidate_path, candidate_scalar in scalar_values[1:]:
                if not np.allclose(
                    reference_scalar,
                    candidate_scalar,
                    atol=consistency_atol_mev,
                    rtol=rtol,
                ):
                    error = float(np.max(np.abs(reference_scalar - candidate_scalar)))
                    raise ValueError(
                        f"Scalar exchange datasets {reference_path} and {candidate_path} "
                        f"disagree (max error {error:.6g} meV)"
                    )

        if tensor_present:
            tensor_mev = _real_array(
                _require_dataset(handle, _TENSOR_DATASET), label=_TENSOR_DATASET
            )
            if tensor_mev.shape != (n_bond, 3, 3):
                raise ValueError(
                    f"J_tensor_r must have shape ({n_bond}, 3, 3); got {tensor_mev.shape}"
                )
            input_tensor_axes = _tensor_axis_order(handle)
            tensor_mev = tensor_mev * energy_units[_TENSOR_DATASET][1]
            canonical_permutation = tuple(
                input_tensor_axes.index(axis) for axis in ("x", "y", "z")
            )
            tensor_mev = tensor_mev[:, canonical_permutation, :][
                :, :, canonical_permutation
            ]
            isotropic_mev = np.trace(tensor_mev, axis1=1, axis2=2) / 3.0
            if scalar_values and not np.allclose(
                scalar_values[0][1],
                isotropic_mev,
                atol=consistency_atol_mev,
                rtol=rtol,
            ):
                error = float(np.max(np.abs(scalar_values[0][1] - isotropic_mev)))
                raise ValueError(
                    f"{scalar_values[0][0]} disagrees with trace(J_tensor_r)/3 "
                    f"(max error {error:.6g} meV)"
                )
            tensor_mode = _optional_scalar_text(handle, "basic_data/tensor_mode")
            isotropic_tensor_source = exchange_convention.kernel_family in {
                "scalar",
                "scalar_lkag",
            } or (
                tensor_mode is not None
                and tensor_mode.strip().lower() == "isotropic_from_scalar_collinear"
            )
            if isotropic_tensor_source:
                isotropic_tensor = isotropic_mev[:, None, None] * np.eye(
                    3, dtype=np.float64
                )
                if not np.allclose(
                    tensor_mev,
                    isotropic_tensor,
                    atol=consistency_atol_mev,
                    rtol=rtol,
                ):
                    error = float(np.max(np.abs(tensor_mev - isotropic_tensor)))
                    raise ValueError(
                        "Scalar-lifted J_tensor_r contains anisotropic components "
                        f"(max deviation {error:.6g} meV)"
                    )
                tensor_mev = isotropic_tensor
                representation = ExchangeRepresentation.ISOTROPIC
                promoted = True
            else:
                representation = ExchangeRepresentation.TENSOR
                promoted = False
            source_dataset = _TENSOR_DATASET
        else:
            source_dataset, isotropic_mev = scalar_values[0]
            tensor_mev = isotropic_mev[:, None, None] * np.eye(3, dtype=np.float64)
            representation = ExchangeRepresentation.ISOTROPIC
            promoted = True
            input_tensor_axes = ("x", "y", "z")

        source_mirror: NDArray[np.int64] | None = None
        if "bonds/mirror_index" in handle:
            source_mirror = _integer_array(
                _require_dataset(handle, "bonds/mirror_index"),
                label="bonds/mirror_index",
                shape=(n_bond,),
            )
            if np.any(source_mirror < 0) or np.any(source_mirror >= n_bond):
                raise ValueError("bonds/mirror_index contains an out-of-range index")
        mirror, max_reciprocity_error = _screen_bonds(
            bond_i,
            bond_j,
            cell_shift,
            tensor_mev,
            representation,
            source_mirror=source_mirror,
            reciprocity_atol_mev=reciprocity_atol_mev,
            rtol=rtol,
        )
        tensor_mev = _canonicalize_reciprocal_tensor(tensor_mev, mirror)
        isotropic_mev = np.trace(tensor_mev, axis1=1, axis2=2) / 3.0
        if representation is ExchangeRepresentation.ISOTROPIC:
            tensor_mev = isotropic_mev[:, None, None] * np.eye(3, dtype=np.float64)

        distance = _optional_real_array(handle, "bonds/distance_ang", shape=(n_bond,))
        if distance is not None and np.any(distance < 0.0):
            raise ValueError("bonds/distance_ang must be non-negative")
        lattice = _optional_real_array(handle, "basic_data/lattice_ang", shape=(3, 3))
        tau_frac = _optional_real_array(handle, "basic_data/tau_frac")
        tau_cart = _optional_real_array(handle, "basic_data/tau_cart_ang")
        for name, positions in (
            ("basic_data/tau_frac", tau_frac),
            ("basic_data/tau_cart_ang", tau_cart),
        ):
            if positions is not None and (
                positions.ndim != 2 or positions.shape[1] != 3
            ):
                raise ValueError(
                    f"{name} must have shape (n_atom, 3); got {positions.shape}"
                )
        labels = _read_atom_labels(handle)

    capabilities = [ExchangeCapability.FM_ARBITRARY_N]
    if representation is ExchangeRepresentation.ISOTROPIC:
        capabilities.insert(0, ExchangeCapability.ISOTROPIC_STATIC_EXCHANGE)
    else:
        capabilities.insert(0, ExchangeCapability.FULL_TENSOR_STATIC_EXCHANGE)
    if magnetic_atoms.size == 2:
        capabilities.append(ExchangeCapability.COLLINEAR_AFM_BIPARTITE)

    warnings: list[str] = []
    if promoted:
        warnings.append(
            "Scalar exchange was promoted to J_iso*I(3); anisotropic exchange and DMI "
            "capabilities remain disabled."
        )
    converted_units = tuple(unit for unit in input_units if unit != "meV")
    if converted_units:
        warnings.append(
            "Exchange energies were converted from "
            f"{', '.join(converted_units)} to meV."
        )
    if convention_origin == "explicit_override":
        if missing_convention_fields:
            warnings.append(
                "Explicit ExchangeConvention supplied missing metadata: "
                f"{', '.join(missing_convention_fields)}."
            )
        else:
            warnings.append(
                "Explicit ExchangeConvention was checked against complete file metadata."
            )
    if input_tensor_axes != ("x", "y", "z"):
        warnings.append(
            f"Exchange tensor axes were reordered from {input_tensor_axes} to ('x', 'y', 'z')."
        )
    model = ExchangeModel(
        source=source,
        representation=representation,
        source_dataset=source_dataset,
        convention=exchange_convention,
        isotropic_mev=isotropic_mev,
        tensor_mev=tensor_mev,
        bond_i=bond_i,
        bond_j=bond_j,
        bond_i_atom=bond_i_atom,
        bond_j_atom=bond_j_atom,
        cell_shift=cell_shift,
        magnetic_atom_indices=magnetic_atoms,
        mirror_index=mirror,
        distance_ang=distance,
        lattice_ang=lattice,
        tau_frac=tau_frac,
        tau_cart_ang=tau_cart,
        atom_labels=labels,
    )
    report = ExchangeScreeningReport(
        source_dataset=source_dataset,
        representation=representation,
        promoted_isotropic=promoted,
        mate_complete=True,
        max_reciprocity_error_mev=max_reciprocity_error,
        capabilities=tuple(capabilities),
        input_units=input_units,
        reciprocity_atol_mev=reciprocity_atol_mev,
        consistency_atol_mev=consistency_atol_mev,
        relative_tolerance=rtol,
        convention_origin=convention_origin,
        missing_convention_fields=missing_convention_fields,
        warnings=tuple(warnings),
    )
    return model, report


def screen_magnetic_configuration(
    exchange: ExchangeModel,
    *,
    order: MagneticOrder | str,
    spin_pattern: ArrayLike | None = None,
    spin_magnitudes: ArrayLike | float | None = None,
    quantization_axis: ArrayLike | None = None,
    anisotropy: SingleIonAnisotropy | None = None,
    require_local_stability: bool = True,
    stability_atol_mev: float = 1.0e-10,
    torque_atol_mev: float = 1.0e-8,
) -> MagneticConfiguration:
    """Validate a declared FM or bipartite collinear-AFM reference state.

    The order is mandatory and is never inferred from the sign of ``J``.  This
    inexpensive real-space check verifies local alignment and torque for the
    exchange plus optional single-ion anisotropy; a later LSWT/BdG stage must
    still screen the full q-space spectrum.
    """

    stability_atol_mev = _nonnegative_finite_tolerance(
        "stability_atol_mev", stability_atol_mev
    )
    torque_atol_mev = _nonnegative_finite_tolerance("torque_atol_mev", torque_atol_mev)
    try:
        magnetic_order = MagneticOrder(order)
    except ValueError as exc:
        choices = ", ".join(item.value for item in MagneticOrder)
        raise ValueError(
            f"Unsupported magnetic order {order!r}; choose {choices}"
        ) from exc

    n_site = exchange.n_magnetic_sites
    missing_state = []
    if spin_magnitudes is None:
        missing_state.append("spin_magnitudes")
    if (
        exchange.representation is ExchangeRepresentation.TENSOR
        and quantization_axis is None
    ):
        missing_state.append("quantization_axis")
    if missing_state:
        qualifier = (
            "full tensor exchange"
            if exchange.representation is ExchangeRepresentation.TENSOR
            else "native magnetic screening"
        )
        raise ValueError(
            f"{qualifier} requires an explicit magnetic reference; "
            f"provide {', '.join(missing_state)}"
        )
    if magnetic_order is MagneticOrder.COLLINEAR_AFM and n_site != 2:
        raise NotImplementedError(
            "The first native AFM path supports exactly two magnetic sublattices; "
            f"the screened exchange model has {n_site}."
        )

    if spin_pattern is None:
        pattern = (
            np.ones(n_site, dtype=np.float64)
            if magnetic_order is MagneticOrder.FM
            else np.asarray((1.0, -1.0), dtype=np.float64)
        )
    else:
        pattern = np.asarray(spin_pattern, dtype=np.float64)
        if pattern.shape != (n_site,):
            raise ValueError(
                f"spin_pattern must have shape ({n_site},); got {pattern.shape}"
            )
        if not np.all(np.isfinite(pattern)) or not np.allclose(
            np.abs(pattern), 1.0, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("spin_pattern entries must be finite and exactly +/-1")
        pattern = np.sign(pattern)
    if magnetic_order is MagneticOrder.FM and not np.all(pattern == pattern[0]):
        raise ValueError("FM spin_pattern entries must all have the same sign")
    if magnetic_order is MagneticOrder.COLLINEAR_AFM and pattern[0] != -pattern[1]:
        raise ValueError("bipartite AFM spin_pattern must contain opposite signs")

    if spin_magnitudes is None:  # Guarded above; narrows the static type.
        raise AssertionError("unreachable missing spin_magnitudes")
    raw_magnitudes = np.asarray(spin_magnitudes, dtype=np.float64)
    if raw_magnitudes.ndim == 0:
        magnitudes = np.full(n_site, float(raw_magnitudes), dtype=np.float64)
    elif raw_magnitudes.shape == (n_site,):
        magnitudes = np.array(raw_magnitudes, copy=True)
    else:
        raise ValueError(
            f"spin_magnitudes must be a scalar or have shape ({n_site},); "
            f"got {raw_magnitudes.shape}"
        )
    if not np.all(np.isfinite(magnitudes)) or np.any(magnitudes <= 0.0):
        raise ValueError("spin_magnitudes must be finite and strictly positive")

    axis_input: ArrayLike = (
        (0.0, 0.0, 1.0) if quantization_axis is None else quantization_axis
    )
    axis = np.asarray(axis_input, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("quantization_axis must be a finite length-3 vector")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= np.finfo(np.float64).eps:
        raise ValueError("quantization_axis must be nonzero")
    axis = axis / axis_norm
    directions = pattern[:, None] * axis[None, :]
    spin_vectors = magnitudes[:, None] * directions

    if exchange.convention.spin_normalization is ExchangeSpinNormalization.UNIT_VECTOR:
        exchange_states = directions
    else:
        expected_magnitude = exchange.convention.source_spin_magnitude
        if not np.allclose(
            magnitudes,
            expected_magnitude,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "spin_operator exchange was generated for "
                f"S={expected_magnitude:g}, but spin_magnitudes={magnitudes.tolist()}"
            )
        exchange_states = spin_vectors

    neighbour_spins = exchange_states[exchange.bond_j]
    bond_fields = np.einsum(
        "bij,bj->bi", exchange.tensor_mev, neighbour_spins, optimize=True
    )
    exchange_fields = np.zeros((n_site, 3), dtype=np.float64)
    np.add.at(exchange_fields, exchange.bond_i, bond_fields)
    fields = (
        exchange_fields
        if exchange.convention.spin_normalization
        is ExchangeSpinNormalization.UNIT_VECTOR
        else magnitudes[:, None] * exchange_fields
    )
    if anisotropy is not None:
        if anisotropy.n_magnetic_sites != n_site:
            raise ValueError(
                "single-ion anisotropy and exchange magnetic site counts differ"
            )
        anisotropy_states = (
            directions
            if anisotropy.spin_normalization is ExchangeSpinNormalization.UNIT_VECTOR
            else spin_vectors
        )
        projections = np.einsum(
            "ij,ij->i", anisotropy_states, anisotropy.axis, optimize=True
        )
        anisotropy_fields = (
            2.0
            * anisotropy.energy_mev[:, None]
            * projections[:, None]
            * anisotropy.axis
        )
        if anisotropy.spin_normalization is ExchangeSpinNormalization.SPIN_OPERATOR:
            anisotropy_fields = magnitudes[:, None] * anisotropy_fields
        fields = fields + anisotropy_fields
    stiffness = np.einsum("ij,ij->i", directions, fields, optimize=True)
    torque = np.cross(directions, fields)
    max_torque = float(np.max(np.linalg.norm(torque, axis=1)))
    minimum_stiffness = float(np.min(stiffness))
    locally_stable = (
        minimum_stiffness >= -stability_atol_mev and max_torque <= torque_atol_mev
    )
    if require_local_stability and not locally_stable:
        raise ValueError(
            "Declared magnetic reference is not locally stationary/stable under the "
            "screened exchange/SIA model: "
            f"min longitudinal stiffness={minimum_stiffness:.6g} meV, "
            f"max torque={max_torque:.6g} meV. The order is not inferred from J; "
            "check the declared order/spin_pattern or explicitly disable this local check."
        )

    metric = (
        np.ones(n_site, dtype=np.float64)
        if magnetic_order is MagneticOrder.FM
        else np.concatenate(
            (np.ones(n_site, dtype=np.float64), -np.ones(n_site, dtype=np.float64))
        )
    )
    return MagneticConfiguration(
        order=magnetic_order,
        spin_pattern=pattern,
        spin_magnitudes=magnitudes,
        quantization_axis=axis,
        spin_directions=directions,
        local_exchange_field_mev=fields,
        longitudinal_stiffness_mev=stiffness,
        torque_mev=torque,
        bosonic_metric=metric,
        locally_stable=locally_stable,
    )


__all__ = (
    "load_exchange_h5",
    "screen_magnetic_configuration",
)
