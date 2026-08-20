"""Validated data models for the native magnon--phonon pipeline.

The native layer deliberately keeps the representation found on disk separate
from its canonical in-memory representation.  In particular, scalar exchange
is represented internally by an isotropic 3x3 tensor without claiming that the
input contained anisotropic exchange information.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ExchangeRepresentation(str, Enum):
    """Exchange representation provided by the source file."""

    ISOTROPIC = "isotropic"
    TENSOR = "tensor"


class ExchangeSpinNormalization(str, Enum):
    """Spin variables used by the exchange Hamiltonian on disk."""

    UNIT_VECTOR = "unit_vector"
    SPIN_OPERATOR = "spin_operator"


class ExchangeCapability(str, Enum):
    """Capabilities that can be enabled without inventing input data."""

    ISOTROPIC_STATIC_EXCHANGE = "isotropic_static_exchange"
    FULL_TENSOR_STATIC_EXCHANGE = "full_tensor_static_exchange"
    FM_ARBITRARY_N = "fm_arbitrary_n"
    COLLINEAR_AFM_BIPARTITE = "collinear_afm_bipartite"


class MagneticOrder(str, Enum):
    """Magnetic orders supported by the first native workflow."""

    FM = "fm"
    COLLINEAR_AFM = "collinear_afm"


@dataclass(frozen=True)
class ExchangeConvention:
    """Explicit Hamiltonian and real-space convention for one J payload.

    The initial native pipeline canonicalizes only the minus-sign,
    mate-complete directed representation.  Other conventions must be
    converted at ingestion rather than being guessed downstream.
    """

    spin_normalization: ExchangeSpinNormalization
    source_spin_magnitude: float
    kernel_family: str
    hamiltonian_sign: str = "minus"
    bond_coverage: str = "directed_mate_complete"
    directed_bond_weight: float = 0.5
    realspace_gauge: str = "i_at_0_j_at_R"

    def __post_init__(self) -> None:
        normalization = ExchangeSpinNormalization(self.spin_normalization)
        magnitude = float(self.source_spin_magnitude)
        if not np.isfinite(magnitude) or magnitude <= 0.0:
            raise ValueError("source_spin_magnitude must be positive and finite")
        kernel = str(self.kernel_family).strip().lower()
        if kernel == "scalar":
            kernel = "scalar_lkag"
        if not kernel:
            raise ValueError("kernel_family must not be empty")
        sign = str(self.hamiltonian_sign).strip().lower()
        coverage = str(self.bond_coverage).strip().lower()
        gauge = str(self.realspace_gauge).strip()
        weight = float(self.directed_bond_weight)
        if sign != "minus":
            raise ValueError(
                "native exchange requires hamiltonian_sign='minus'; "
                f"got {self.hamiltonian_sign!r}"
            )
        if coverage != "directed_mate_complete":
            raise ValueError(
                "native exchange requires bond_coverage='directed_mate_complete'; "
                f"got {self.bond_coverage!r}"
            )
        if not np.isfinite(weight) or not np.isclose(weight, 0.5, rtol=0.0, atol=0.0):
            raise ValueError(
                "native directed mate-complete exchange requires "
                f"directed_bond_weight=0.5; got {self.directed_bond_weight!r}"
            )
        if gauge != "i_at_0_j_at_R":
            raise ValueError(
                "native exchange requires realspace_gauge='i_at_0_j_at_R'; "
                f"got {self.realspace_gauge!r}"
            )
        if kernel == "tb2j":
            raise ValueError(
                "kernel_family='tb2j' is not in the native directed-weight=0.5 "
                "convention: audited TB2J payloads carry weight 1.0 for a "
                "mate-complete directed list. Convert the payload explicitly "
                "before native ingestion."
            )
        object.__setattr__(self, "spin_normalization", normalization)
        object.__setattr__(self, "source_spin_magnitude", magnitude)
        object.__setattr__(self, "kernel_family", kernel)
        object.__setattr__(self, "hamiltonian_sign", sign)
        object.__setattr__(self, "bond_coverage", coverage)
        object.__setattr__(self, "directed_bond_weight", weight)
        object.__setattr__(self, "realspace_gauge", gauge)


def _readonly_array(
    name: str,
    value: ArrayLike,
    *,
    dtype: np.dtype | type,
    ndim: int | None = None,
    shape: tuple[int | None, ...] | None = None,
) -> NDArray:
    """Return an owned, immutable NumPy array after structural validation."""

    array = np.array(value, dtype=dtype, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {array.shape}")
    if shape is not None and (
        array.ndim != len(shape)
        or any(
            expected is not None and actual != expected
            for actual, expected in zip(array.shape, shape)
        )
    ):
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    array.setflags(write=False)
    return array


def _readonly_integer_array(
    name: str,
    value: ArrayLike,
    *,
    ndim: int | None = None,
    shape: tuple[int | None, ...] | None = None,
) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if (
        np.iscomplexobj(raw)
        or not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must contain real integers")
    if np.issubdtype(raw.dtype, np.integer):
        if np.issubdtype(raw.dtype, np.unsignedinteger) and np.any(
            raw > np.iinfo(np.int64).max
        ):
            raise ValueError(f"{name} contains values outside the int64 range")
        integer = np.asarray(raw, dtype=np.int64)
    else:
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.array_equal(
            numeric, np.rint(numeric)
        ):
            raise ValueError(f"{name} must contain finite integers")
        int64_upper_exclusive = float(2**63)
        if np.any(numeric < -int64_upper_exclusive) or np.any(
            numeric >= int64_upper_exclusive
        ):
            raise ValueError(f"{name} contains values outside the int64 range")
        integer = np.asarray(numeric, dtype=np.int64)
    return _readonly_array(
        name,
        integer,
        dtype=np.int64,
        ndim=ndim,
        shape=shape,
    )


def _require_finite(name: str, array: NDArray) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class ExchangeModel:
    """Canonical real-space exchange data, always expressed in meV.

    ``tensor_mev`` always has shape ``(n_bond, 3, 3)``.  For a scalar source it
    is the exact promotion ``J_iso * identity(3)``; ``representation`` and the
    accompanying :class:`ExchangeScreeningReport` retain that provenance.
    Bond indices in ``bond_i`` and ``bond_j`` are contiguous local indices,
    whereas the ``*_atom`` arrays preserve the source's global atom indices.
    """

    source: Path
    representation: ExchangeRepresentation
    source_dataset: str
    convention: ExchangeConvention
    isotropic_mev: NDArray[np.float64]
    tensor_mev: NDArray[np.float64]
    bond_i: NDArray[np.int64]
    bond_j: NDArray[np.int64]
    bond_i_atom: NDArray[np.int64]
    bond_j_atom: NDArray[np.int64]
    cell_shift: NDArray[np.int64]
    magnetic_atom_indices: NDArray[np.int64]
    mirror_index: NDArray[np.int64]
    distance_ang: NDArray[np.float64] | None = None
    lattice_ang: NDArray[np.float64] | None = None
    tau_frac: NDArray[np.float64] | None = None
    tau_cart_ang: NDArray[np.float64] | None = None
    atom_labels: tuple[str, ...] = ()
    unit: str = "meV"

    def __post_init__(self) -> None:
        source = Path(self.source)
        representation = ExchangeRepresentation(self.representation)
        source_dataset = str(self.source_dataset).strip()
        if not source_dataset:
            raise ValueError("source_dataset must not be empty")
        if not isinstance(self.convention, ExchangeConvention):
            raise TypeError("convention must be an ExchangeConvention")
        if self.unit != "meV":
            raise ValueError(
                f"ExchangeModel unit must be canonical 'meV'; got {self.unit!r}"
            )

        isotropic = _readonly_array(
            "isotropic_mev", self.isotropic_mev, dtype=np.float64, ndim=1
        )
        n_bond = isotropic.size
        if n_bond == 0:
            raise ValueError("exchange model must contain at least one bond")
        tensor = _readonly_array(
            "tensor_mev", self.tensor_mev, dtype=np.float64, shape=(n_bond, 3, 3)
        )
        _require_finite("isotropic_mev", isotropic)
        _require_finite("tensor_mev", tensor)
        tensor_isotropic = np.trace(tensor, axis1=1, axis2=2) / 3.0
        if representation is ExchangeRepresentation.ISOTROPIC:
            expected_tensor = isotropic[:, None, None] * np.eye(3, dtype=np.float64)
            if not np.array_equal(tensor, expected_tensor):
                raise ValueError(
                    "isotropic representation requires tensor_mev to equal "
                    "isotropic_mev*I(3) exactly"
                )
        elif not np.array_equal(isotropic, tensor_isotropic):
            raise ValueError(
                "tensor representation requires isotropic_mev to equal "
                "trace(tensor_mev)/3 exactly"
            )

        vector_fields = {
            "bond_i": self.bond_i,
            "bond_j": self.bond_j,
            "bond_i_atom": self.bond_i_atom,
            "bond_j_atom": self.bond_j_atom,
            "mirror_index": self.mirror_index,
        }
        frozen_vectors: dict[str, NDArray[np.int64]] = {}
        for name, value in vector_fields.items():
            frozen_vectors[name] = _readonly_integer_array(name, value, shape=(n_bond,))
        cell_shift = _readonly_integer_array(
            "cell_shift", self.cell_shift, shape=(n_bond, 3)
        )
        magnetic_atoms = _readonly_integer_array(
            "magnetic_atom_indices",
            self.magnetic_atom_indices,
            ndim=1,
        )
        if magnetic_atoms.size == 0:
            raise ValueError("magnetic_atom_indices must not be empty")
        if np.any(magnetic_atoms < 0):
            raise ValueError("magnetic_atom_indices must be non-negative")
        if np.unique(magnetic_atoms).size != magnetic_atoms.size:
            raise ValueError("magnetic_atom_indices must be unique")
        if np.any(frozen_vectors["bond_i"] < 0) or np.any(
            frozen_vectors["bond_i"] >= magnetic_atoms.size
        ):
            raise ValueError("bond_i contains an out-of-range local site index")
        if np.any(frozen_vectors["bond_j"] < 0) or np.any(
            frozen_vectors["bond_j"] >= magnetic_atoms.size
        ):
            raise ValueError("bond_j contains an out-of-range local site index")
        if np.any(frozen_vectors["bond_i_atom"] < 0) or np.any(
            frozen_vectors["bond_j_atom"] < 0
        ):
            raise ValueError("global atom indices must be non-negative")
        if np.any(frozen_vectors["mirror_index"] < 0) or np.any(
            frozen_vectors["mirror_index"] >= n_bond
        ):
            raise ValueError("mirror_index contains an out-of-range bond index")
        mirror = frozen_vectors["mirror_index"]
        if not np.array_equal(mirror[mirror], np.arange(n_bond, dtype=np.int64)):
            raise ValueError("mirror_index must be an involution")
        if not np.array_equal(
            magnetic_atoms[frozen_vectors["bond_i"]],
            frozen_vectors["bond_i_atom"],
        ) or not np.array_equal(
            magnetic_atoms[frozen_vectors["bond_j"]],
            frozen_vectors["bond_j_atom"],
        ):
            raise ValueError(
                "local bond indices are inconsistent with global atom indices"
            )
        mate_keys_ok = (
            np.array_equal(frozen_vectors["bond_i"][mirror], frozen_vectors["bond_j"])
            and np.array_equal(
                frozen_vectors["bond_j"][mirror], frozen_vectors["bond_i"]
            )
            and np.array_equal(cell_shift[mirror], -cell_shift)
        )
        if not mate_keys_ok:
            raise ValueError("mirror_index does not map each bond to (j,i,-R)")
        reciprocal_tensor = np.transpose(tensor[mirror], (0, 2, 1))
        if not np.array_equal(tensor, reciprocal_tensor):
            raise ValueError(
                "tensor_mev is not exactly mate-reciprocal: canonical data require "
                "J(i,j,R) = J(j,i,-R).T"
            )

        optional_arrays: dict[str, NDArray | None] = {
            "distance_ang": None,
            "lattice_ang": None,
            "tau_frac": None,
            "tau_cart_ang": None,
        }
        if self.distance_ang is not None:
            distance = _readonly_array(
                "distance_ang", self.distance_ang, dtype=np.float64, shape=(n_bond,)
            )
            _require_finite("distance_ang", distance)
            if np.any(distance < 0.0):
                raise ValueError("distance_ang must be non-negative")
            optional_arrays["distance_ang"] = distance
        if self.lattice_ang is not None:
            lattice = _readonly_array(
                "lattice_ang", self.lattice_ang, dtype=np.float64, shape=(3, 3)
            )
            _require_finite("lattice_ang", lattice)
            if abs(float(np.linalg.det(lattice))) <= np.finfo(np.float64).eps:
                raise ValueError("lattice_ang must be nonsingular")
            optional_arrays["lattice_ang"] = lattice
        for position_name, position_value in (
            ("tau_frac", self.tau_frac),
            ("tau_cart_ang", self.tau_cart_ang),
        ):
            if position_value is None:
                continue
            positions = _readonly_array(
                position_name, position_value, dtype=np.float64, ndim=2
            )
            if positions.shape[1:] != (3,):
                raise ValueError(
                    f"{position_name} must have shape (n_atom, 3); got {positions.shape}"
                )
            _require_finite(position_name, positions)
            optional_arrays[position_name] = positions

        labels = tuple(str(label) for label in self.atom_labels)
        per_atom_counts = {
            name: int(array.shape[0])
            for name, array in (
                ("tau_frac", optional_arrays["tau_frac"]),
                ("tau_cart_ang", optional_arrays["tau_cart_ang"]),
            )
            if array is not None
        }
        if per_atom_counts:
            count_values = set(per_atom_counts.values())
            if len(count_values) != 1:
                details = ", ".join(
                    f"{name}={count}" for name, count in per_atom_counts.items()
                )
                raise ValueError(
                    f"per-atom geometry arrays disagree in size: {details}"
                )
            n_geometry_atom = next(iter(count_values))
            referenced_atoms = np.concatenate(
                (
                    frozen_vectors["bond_i_atom"],
                    frozen_vectors["bond_j_atom"],
                    magnetic_atoms,
                )
            )
            required_atoms = int(np.max(referenced_atoms)) + 1
            if n_geometry_atom < required_atoms:
                raise ValueError(
                    "per-atom geometry does not cover all referenced global atoms: "
                    f"need at least {required_atoms} rows, got {n_geometry_atom}"
                )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "source_dataset", source_dataset)
        object.__setattr__(self, "convention", self.convention)
        object.__setattr__(self, "isotropic_mev", isotropic)
        object.__setattr__(self, "tensor_mev", tensor)
        object.__setattr__(self, "cell_shift", cell_shift)
        object.__setattr__(self, "magnetic_atom_indices", magnetic_atoms)
        object.__setattr__(self, "atom_labels", labels)
        for vector_name, vector_value in frozen_vectors.items():
            object.__setattr__(self, vector_name, vector_value)
        for optional_name, optional_value in optional_arrays.items():
            object.__setattr__(self, optional_name, optional_value)

    @property
    def n_bonds(self) -> int:
        return int(self.isotropic_mev.size)

    @property
    def n_magnetic_sites(self) -> int:
        return int(self.magnetic_atom_indices.size)


@dataclass(frozen=True)
class ExchangeScreeningReport:
    """Auditable outcome of loading and screening an exchange file."""

    source_dataset: str
    representation: ExchangeRepresentation
    promoted_isotropic: bool
    mate_complete: bool
    max_reciprocity_error_mev: float
    capabilities: tuple[ExchangeCapability, ...]
    input_units: tuple[str, ...]
    reciprocity_atol_mev: float
    consistency_atol_mev: float
    relative_tolerance: float
    convention_origin: str = "file"
    missing_convention_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        representation = ExchangeRepresentation(self.representation)
        capabilities = tuple(ExchangeCapability(item) for item in self.capabilities)
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("capabilities must not contain duplicates")
        error = float(self.max_reciprocity_error_mev)
        if not np.isfinite(error) or error < 0.0:
            raise ValueError(
                "max_reciprocity_error_mev must be finite and non-negative"
            )
        tolerances = (
            float(self.reciprocity_atol_mev),
            float(self.consistency_atol_mev),
            float(self.relative_tolerance),
        )
        if any(not np.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError(
                "screening report tolerances must be finite and non-negative"
            )
        convention_origin = str(self.convention_origin).strip().lower()
        if convention_origin not in {"file", "explicit_override"}:
            raise ValueError(
                "convention_origin must be 'file' or 'explicit_override'; "
                f"got {self.convention_origin!r}"
            )
        missing_fields = tuple(str(item) for item in self.missing_convention_fields)
        if len(set(missing_fields)) != len(missing_fields):
            raise ValueError("missing_convention_fields must not contain duplicates")
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "max_reciprocity_error_mev", error)
        object.__setattr__(self, "reciprocity_atol_mev", tolerances[0])
        object.__setattr__(self, "consistency_atol_mev", tolerances[1])
        object.__setattr__(self, "relative_tolerance", tolerances[2])
        object.__setattr__(self, "convention_origin", convention_origin)
        object.__setattr__(self, "missing_convention_fields", missing_fields)
        object.__setattr__(self, "input_units", tuple(str(x) for x in self.input_units))
        object.__setattr__(self, "warnings", tuple(str(x) for x in self.warnings))

    def supports(self, capability: ExchangeCapability | str) -> bool:
        """Return whether a capability is supported by the screened input."""

        return ExchangeCapability(capability) in self.capabilities


@dataclass(frozen=True)
class MagneticConfiguration:
    """A validated collinear reference state for the native workflow."""

    order: MagneticOrder
    spin_pattern: NDArray[np.float64]
    spin_magnitudes: NDArray[np.float64]
    quantization_axis: NDArray[np.float64]
    spin_directions: NDArray[np.float64]
    local_exchange_field_mev: NDArray[np.float64]
    longitudinal_stiffness_mev: NDArray[np.float64]
    torque_mev: NDArray[np.float64]
    bosonic_metric: NDArray[np.float64]
    locally_stable: bool

    def __post_init__(self) -> None:
        order = MagneticOrder(self.order)
        pattern = _readonly_array(
            "spin_pattern", self.spin_pattern, dtype=np.float64, ndim=1
        )
        n_site = pattern.size
        if n_site == 0:
            raise ValueError("spin_pattern must not be empty")
        magnitudes = _readonly_array(
            "spin_magnitudes", self.spin_magnitudes, dtype=np.float64, shape=(n_site,)
        )
        axis = _readonly_array(
            "quantization_axis", self.quantization_axis, dtype=np.float64, shape=(3,)
        )
        directions = _readonly_array(
            "spin_directions", self.spin_directions, dtype=np.float64, shape=(n_site, 3)
        )
        fields = _readonly_array(
            "local_exchange_field_mev",
            self.local_exchange_field_mev,
            dtype=np.float64,
            shape=(n_site, 3),
        )
        stiffness = _readonly_array(
            "longitudinal_stiffness_mev",
            self.longitudinal_stiffness_mev,
            dtype=np.float64,
            shape=(n_site,),
        )
        torque = _readonly_array(
            "torque_mev", self.torque_mev, dtype=np.float64, shape=(n_site, 3)
        )
        expected_channels = n_site if order is MagneticOrder.FM else 2 * n_site
        metric = _readonly_array(
            "bosonic_metric",
            self.bosonic_metric,
            dtype=np.float64,
            shape=(expected_channels,),
        )
        for name, array in (
            ("spin_pattern", pattern),
            ("spin_magnitudes", magnitudes),
            ("quantization_axis", axis),
            ("spin_directions", directions),
            ("local_exchange_field_mev", fields),
            ("longitudinal_stiffness_mev", stiffness),
            ("torque_mev", torque),
            ("bosonic_metric", metric),
        ):
            _require_finite(name, array)
        if not np.all(np.isin(pattern, (-1.0, 1.0))):
            raise ValueError("spin_pattern entries must be exactly +/-1")
        if np.any(magnitudes <= 0.0):
            raise ValueError("spin_magnitudes must be strictly positive")
        if not np.isclose(np.linalg.norm(axis), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("quantization_axis must be normalized")
        if not np.allclose(
            directions,
            pattern[:, None] * axis[None, :],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "spin_directions must equal spin_pattern*quantization_axis"
            )
        expected_metric = (
            np.ones(n_site, dtype=np.float64)
            if order is MagneticOrder.FM
            else np.concatenate(
                (np.ones(n_site, dtype=np.float64), -np.ones(n_site, dtype=np.float64))
            )
        )
        if not np.array_equal(metric, expected_metric):
            raise ValueError("bosonic_metric is inconsistent with magnetic order")
        if not isinstance(self.locally_stable, (bool, np.bool_)):
            raise TypeError("locally_stable must be boolean")
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "spin_pattern", pattern)
        object.__setattr__(self, "spin_magnitudes", magnitudes)
        object.__setattr__(self, "quantization_axis", axis)
        object.__setattr__(self, "spin_directions", directions)
        object.__setattr__(self, "local_exchange_field_mev", fields)
        object.__setattr__(self, "longitudinal_stiffness_mev", stiffness)
        object.__setattr__(self, "torque_mev", torque)
        object.__setattr__(self, "bosonic_metric", metric)
        object.__setattr__(self, "locally_stable", bool(self.locally_stable))

    @property
    def n_magnetic_sites(self) -> int:
        return int(self.spin_pattern.size)

    @property
    def n_channels(self) -> int:
        return int(self.bosonic_metric.size)
