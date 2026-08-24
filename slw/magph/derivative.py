"""Strict native ingestion of real-space exchange derivatives.

The first production route accepts the scalar LKAG derivative written as
``dJ_r(R,Rp)/du`` and promotes it to an isotropic tensor only to provide one
internal shape.  The promotion never grants tensor-coupling capabilities.
Static and dynamic bonds are matched by their global ``(i,j,R)`` keys, not by
file order.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.core.constants import BOHR_TO_ANG, HA_TO_EV, RY_TO_EV

from .model import ExchangeModel, ExchangeRepresentation


class DerivativeASRPolicy(str, Enum):
    FAIL = "fail"
    PROJECT = "project"
    REPORT = "report"


@dataclass(frozen=True)
class ExchangeDerivativeReport:
    representation: ExchangeRepresentation
    source_dataset: str
    input_unit: str
    max_reciprocity_error_mev_per_ang: float
    asr_policy: DerivativeASRPolicy
    asr_tolerance_mev_per_ang: float
    max_asr_residual_before_mev_per_ang: float
    max_asr_residual_after_mev_per_ang: float
    projected_asr: bool
    static_bond_count: int
    source_bond_count: int
    zero_filled_static_bond_count: int
    bond_coverage_complete: bool
    covariant_symmetry_policy: str
    covariant_symmetry_applied: bool
    max_covariance_residual_mev_per_ang: float | None
    covariant_symmetry_operation_count: int
    covariant_symmetry_spacegroup: str


def _readonly(value: object, *, dtype: np.dtype[Any]) -> NDArray[Any]:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ExchangeDerivativeModel:
    """Canonical ``dJ(R,Rp)/du`` in meV/Angstrom.

    ``tensor_mev_per_ang`` is ordered as
    ``(target,bond,Rp,displacement_axis,spin_axis,spin_axis)``.
    """

    source: Path
    static_exchange_source: Path
    representation: ExchangeRepresentation
    source_dataset: str
    target_atom_indices: NDArray[np.int64]
    rp_cell_shifts: NDArray[np.int64]
    q_mesh_shape: tuple[int, int, int]
    isotropic_mev_per_ang: NDArray[np.float64]
    tensor_mev_per_ang: NDArray[np.float64]
    source_static_bond_indices: NDArray[np.int64] | None = None
    displacement_axes: tuple[str, str, str] = ("x", "y", "z")
    fourier_phase_convention: str = "exp(+i2pi_q_dot_Rp)"
    directed_bond_mate: str = "(j,i,-R;Rp-R)_periodic"
    unit: str = "meV/angstrom"

    def __post_init__(self) -> None:
        source = Path(self.source)
        static_source = Path(self.static_exchange_source)
        representation = ExchangeRepresentation(self.representation)
        target = _readonly(self.target_atom_indices, dtype=np.dtype(np.int64))
        rp = _readonly(self.rp_cell_shifts, dtype=np.dtype(np.int64))
        isotropic = _readonly(self.isotropic_mev_per_ang, dtype=np.dtype(np.float64))
        tensor = _readonly(self.tensor_mev_per_ang, dtype=np.dtype(np.float64))
        if target.ndim != 1 or target.size == 0 or np.any(target < 0):
            raise ValueError("target_atom_indices must be nonempty and non-negative")
        if np.unique(target).size != target.size:
            raise ValueError("target_atom_indices must be unique")
        if rp.ndim != 2 or rp.shape[1:] != (3,) or rp.shape[0] == 0:
            raise ValueError("rp_cell_shifts must have nonempty shape (nRp,3)")
        if isotropic.ndim != 4:
            raise ValueError("isotropic_mev_per_ang must have shape (target,bond,Rp,3)")
        expected_prefix = (target.size, isotropic.shape[1], rp.shape[0], 3)
        if isotropic.shape != expected_prefix:
            raise ValueError(
                f"isotropic_mev_per_ang shape {isotropic.shape} != {expected_prefix}"
            )
        if tensor.shape != (*expected_prefix, 3, 3):
            raise ValueError(
                f"tensor_mev_per_ang shape {tensor.shape} != {(*expected_prefix, 3, 3)}"
            )
        source_bonds = (
            np.arange(isotropic.shape[1], dtype=np.int64)
            if self.source_static_bond_indices is None
            else np.asarray(self.source_static_bond_indices)
        )
        if (
            source_bonds.ndim != 1
            or source_bonds.size == 0
            or not np.issubdtype(source_bonds.dtype, np.integer)
        ):
            raise ValueError(
                "source_static_bond_indices must be a nonempty integer vector"
            )
        source_bonds = _readonly(source_bonds, dtype=np.dtype(np.int64))
        if (
            np.any(source_bonds < 0)
            or np.any(source_bonds >= isotropic.shape[1])
            or np.unique(source_bonds).size != source_bonds.size
        ):
            raise ValueError(
                "source_static_bond_indices must contain unique in-range bond indices"
            )
        source_mask = np.zeros(isotropic.shape[1], dtype=bool)
        source_mask[source_bonds] = True
        if not np.array_equal(
            isotropic[:, ~source_mask], np.zeros_like(isotropic[:, ~source_mask])
        ) or not np.array_equal(
            tensor[:, ~source_mask], np.zeros_like(tensor[:, ~source_mask])
        ):
            raise ValueError(
                "static bonds absent from source_static_bond_indices must be exactly zero"
            )
        if not np.all(np.isfinite(isotropic)) or not np.all(np.isfinite(tensor)):
            raise ValueError("exchange derivative contains non-finite values")
        tensor_trace = np.trace(tensor, axis1=-2, axis2=-1) / 3.0
        if representation is ExchangeRepresentation.ISOTROPIC:
            expected_tensor = isotropic[..., None, None] * np.eye(3)
            if not np.array_equal(tensor, expected_tensor):
                raise ValueError(
                    "isotropic derivative requires tensor=dJ_iso*I exactly"
                )
        elif not np.array_equal(tensor_trace, isotropic):
            raise ValueError(
                "tensor derivative requires isotropic=trace(tensor)/3 exactly"
            )
        mesh = (
            int(self.q_mesh_shape[0]),
            int(self.q_mesh_shape[1]),
            int(self.q_mesh_shape[2]),
        )
        if len(mesh) != 3 or any(value <= 0 for value in mesh):
            raise ValueError("q_mesh_shape must contain three positive integers")
        if int(np.prod(np.asarray(mesh, dtype=np.int64))) != rp.shape[0]:
            raise ValueError("q_mesh_shape product must equal the number of Rp cells")
        axes = tuple(str(value).strip().lower() for value in self.displacement_axes)
        if axes != ("x", "y", "z"):
            raise ValueError("canonical displacement_axes must be ('x','y','z')")
        if self.fourier_phase_convention != "exp(+i2pi_q_dot_Rp)":
            raise ValueError("unsupported derivative Fourier phase convention")
        if self.directed_bond_mate != "(j,i,-R;Rp-R)_periodic":
            raise ValueError("unsupported derivative directed-bond mate convention")
        if self.unit != "meV/angstrom":
            raise ValueError("canonical derivative unit must be 'meV/angstrom'")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "static_exchange_source", static_source)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "target_atom_indices", target)
        object.__setattr__(self, "rp_cell_shifts", rp)
        object.__setattr__(self, "q_mesh_shape", mesh)
        object.__setattr__(self, "isotropic_mev_per_ang", isotropic)
        object.__setattr__(self, "tensor_mev_per_ang", tensor)
        object.__setattr__(self, "source_static_bond_indices", source_bonds)
        object.__setattr__(self, "displacement_axes", axes)

    @property
    def n_targets(self) -> int:
        return int(self.target_atom_indices.size)

    @property
    def n_bonds(self) -> int:
        return int(self.isotropic_mev_per_ang.shape[1])

    @property
    def n_rp(self) -> int:
        return int(self.rp_cell_shifts.shape[0])

    @property
    def n_source_bonds(self) -> int:
        indices = self.source_static_bond_indices
        if indices is None:  # pragma: no cover - canonicalized by __post_init__
            raise AssertionError("source bond indices were not canonicalized")
        return int(indices.size)

    @property
    def bond_coverage_complete(self) -> bool:
        return self.n_source_bonds == self.n_bonds


def _decode_scalar(value: Any, *, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must contain one scalar value")
    item = array.reshape(()).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _integer_dataset(
    handle: h5py.File,
    path: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.int64]:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        raise KeyError(f"missing required derivative dataset {path}")
    raw = np.asarray(handle[path])
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
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


def read_exchange_derivative_q_mesh(
    path: str | Path,
) -> tuple[int, int, int]:
    """Read and validate the source q mesh represented by derivative ``Rp``."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"exchange derivative HDF5 not found: {source}")
    with h5py.File(source, "r") as handle:
        raw = _integer_dataset(handle, "basic_data/qmesh", shape=(3,))
    mesh = (int(raw[0]), int(raw[1]), int(raw[2]))
    if any(value <= 0 for value in mesh):
        raise ValueError("basic_data/qmesh must contain positive integers")
    return mesh


def _axis_order(value: Any, *, label: str) -> tuple[str, str, str]:
    raw = np.asarray(value).reshape(-1)
    axes = tuple(_decode_scalar(item, label=label).strip().lower() for item in raw)
    if len(axes) != 3 or set(axes) != {"x", "y", "z"}:
        raise ValueError(f"{label} must be a permutation of x,y,z; got {axes}")
    return axes  # type: ignore[return-value]


def _derivative_unit(raw: Any, *, label: str) -> tuple[str, float]:
    text = (
        _decode_scalar(raw, label=label)
        .strip()
        .lower()
        .replace("å", "angstrom")
        .replace(" ", "")
    )
    aliases = {
        "mev/a": ("meV/angstrom", 1.0),
        "mev/ang": ("meV/angstrom", 1.0),
        "mev/angstrom": ("meV/angstrom", 1.0),
        "ev/a": ("eV/angstrom", 1.0e3),
        "ev/ang": ("eV/angstrom", 1.0e3),
        "ev/angstrom": ("eV/angstrom", 1.0e3),
        "ry/bohr": ("Ry/bohr", RY_TO_EV * 1.0e3 / BOHR_TO_ANG),
        "rydberg/bohr": ("Ry/bohr", RY_TO_EV * 1.0e3 / BOHR_TO_ANG),
        "ha/bohr": ("Ha/bohr", HA_TO_EV * 1.0e3 / BOHR_TO_ANG),
        "hartree/bohr": ("Ha/bohr", HA_TO_EV * 1.0e3 / BOHR_TO_ANG),
    }
    if text not in aliases:
        raise ValueError(
            f"unsupported derivative unit {text!r} in {label}; use meV/A, eV/A, Ry/bohr, or Ha/bohr"
        )
    return aliases[text]


def _dataset_unit(
    handle: h5py.File,
    group: h5py.Group,
    dataset: h5py.Dataset,
    *,
    label: str,
) -> tuple[str, float]:
    declarations: list[tuple[str, tuple[str, float]]] = []
    if "unit" in dataset.attrs:
        declarations.append(
            (
                f"{label}.attrs['unit']",
                _derivative_unit(dataset.attrs["unit"], label=label),
            )
        )
    if "unit" in group.attrs:
        declarations.append(
            (
                f"{group.name}.attrs['unit']",
                _derivative_unit(group.attrs["unit"], label=group.name),
            )
        )
    if "basic_data/unit" in handle:
        declarations.append(
            (
                "basic_data/unit",
                _derivative_unit(
                    handle["basic_data/unit"][()], label="basic_data/unit"
                ),
            )
        )
    if not declarations:
        raise ValueError(f"derivative unit is missing for {label}")
    reference = declarations[0][1]
    if any(item[1][1] != reference[1] for item in declarations[1:]):
        details = ", ".join(f"{name}={item[0]}" for name, item in declarations)
        raise ValueError(f"conflicting derivative units for {label}: {details}")
    return reference


def _periodic_rp_index(
    rp: NDArray[np.int64], mesh: tuple[int, int, int]
) -> dict[tuple[int, int, int], int]:
    modulus = np.asarray(mesh, dtype=np.int64)
    keys = [
        (int(key[0]), int(key[1]), int(key[2]))
        for key in np.mod(rp, modulus)
    ]
    if len(set(keys)) != len(keys) or len(keys) != int(np.prod(modulus)):
        raise ValueError("displacements/Rp must be one complete periodic q-mesh cell")
    return {key: index for index, key in enumerate(keys)}


def _dynamic_bond_permutation(
    handle: h5py.File, exchange: ExchangeModel
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    source_i = _integer_dataset(handle, "bonds/mag_i_atom")
    n_bond = source_i.size
    source_j = _integer_dataset(handle, "bonds/mag_j_atom", shape=(n_bond,))
    source_r = _integer_dataset(handle, "bonds/R", shape=(n_bond, 3))
    source_keys = [
        (int(i), int(j), int(r[0]), int(r[1]), int(r[2]))
        for i, j, r in zip(source_i, source_j, source_r)
    ]
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("dynamic derivative contains duplicate directed bond keys")
    lookup = {key: index for index, key in enumerate(source_keys)}
    static_keys = [
        (int(i), int(j), int(r[0]), int(r[1]), int(r[2]))
        for i, j, r in zip(
            exchange.bond_i_atom, exchange.bond_j_atom, exchange.cell_shift
        )
    ]
    extra = sorted(set(source_keys).difference(static_keys))
    if extra:
        raise ValueError(
            "derivative bond keys must be a subset of the static exchange: "
            f"extra={extra[:3]}"
        )
    permutation = np.asarray(
        [lookup.get(key, -1) for key in static_keys], dtype=np.int64
    )
    source_static_bonds = np.flatnonzero(permutation >= 0).astype(
        np.int64, copy=False
    )
    if source_static_bonds.size == 0:
        raise ValueError("derivative bond subset must not be empty")
    if source_static_bonds.size != len(source_keys):
        raise AssertionError("unreachable derivative/static bond mapping mismatch")
    return permutation, source_static_bonds


def _reciprocal_derivative(
    tensor: NDArray[np.float64],
    *,
    exchange: ExchangeModel,
    rp: NDArray[np.int64],
    mesh: tuple[int, int, int],
) -> NDArray[np.float64]:
    lookup = _periodic_rp_index(rp, mesh)
    modulus = np.asarray(mesh, dtype=np.int64)
    reciprocal = np.empty_like(tensor)
    for bond in range(exchange.n_bonds):
        mate = int(exchange.mirror_index[bond])
        shift = exchange.cell_shift[bond]
        shifted_rp = np.mod(rp - shift[None, :], modulus[None, :])
        mate_rp = np.asarray(
            [lookup[(int(key[0]), int(key[1]), int(key[2]))] for key in shifted_rp],
            dtype=np.int64,
        )
        reciprocal[:, bond] = np.transpose(
            tensor[:, mate, mate_rp],
            (0, 1, 2, 4, 3),
        )
    return reciprocal


def load_exchange_derivative_h5(
    path: str | Path,
    exchange: ExchangeModel,
    *,
    reciprocity_atol_mev_per_ang: float = 1.0e-8,
    rtol: float = 1.0e-7,
    asr_policy: DerivativeASRPolicy | str = DerivativeASRPolicy.FAIL,
    asr_tolerance_mev_per_ang: float = 1.0e-8,
) -> tuple[ExchangeDerivativeModel, ExchangeDerivativeReport]:
    """Load, match, and canonicalize a scalar real-space exchange derivative."""

    reciprocity_atol = float(reciprocity_atol_mev_per_ang)
    relative_tolerance = float(rtol)
    asr_tolerance = float(asr_tolerance_mev_per_ang)
    for name, value in (
        ("reciprocity_atol_mev_per_ang", reciprocity_atol),
        ("rtol", relative_tolerance),
        ("asr_tolerance_mev_per_ang", asr_tolerance),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    policy = DerivativeASRPolicy(asr_policy)
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"exchange derivative HDF5 not found: {source}")
    source = source.resolve()

    with h5py.File(source, "r") as handle:
        if "complete" in handle.attrs and int(handle.attrs["complete"]) != 1:
            raise ValueError("exchange derivative file is an incomplete checkpoint")
        phase = str(handle.attrs.get("fourier_phase_convention", ""))
        mate_convention = str(handle.attrs.get("directed_bond_mate", ""))
        if phase != "exp(+i2pi_q_dot_Rp)":
            raise ValueError(
                "exchange derivative requires root attribute "
                "fourier_phase_convention='exp(+i2pi_q_dot_Rp)'"
            )
        if mate_convention != "(j,i,-R;Rp-R)_periodic":
            raise ValueError(
                "exchange derivative requires root attribute "
                "directed_bond_mate='(j,i,-R;Rp-R)_periodic'"
            )
        covariance_policy = "unreported"
        covariance_applied = False
        max_covariance_residual: float | None = None
        covariance_operation_count = 0
        covariance_spacegroup = "unreported"
        if "symmetry" in handle:
            symmetry = handle["symmetry"]
            if not isinstance(symmetry, h5py.Group):
                raise TypeError("symmetry must be an HDF5 group")
            if "policy" in symmetry:
                covariance_policy = _decode_scalar(
                    symmetry["policy"][()], label="symmetry/policy"
                ).strip().lower()
            if covariance_policy not in {
                "none",
                "report",
                "project",
                "fail",
                "unreported",
            }:
                raise ValueError(
                    f"unsupported derivative covariance policy {covariance_policy!r}"
                )
            if "applied" in symmetry:
                covariance_applied = bool(
                    np.asarray(symmetry["applied"][()]).reshape(()).item()
                )
            if covariance_applied and covariance_policy != "project":
                raise ValueError(
                    "derivative covariance provenance says applied=true without "
                    "policy='project'"
                )
            if "max_abs_residual_raw_mev_per_ang" in symmetry:
                max_covariance_residual = float(
                    np.asarray(
                        symmetry["max_abs_residual_raw_mev_per_ang"][()],
                        dtype=np.float64,
                    ).reshape(())
                )
                if not np.isfinite(max_covariance_residual) or max_covariance_residual < 0.0:
                    raise ValueError(
                        "symmetry/max_abs_residual_raw_mev_per_ang must be "
                        "finite and non-negative"
                    )
            if "n_operations" in symmetry:
                covariance_operation_count = int(
                    np.asarray(symmetry["n_operations"][()]).reshape(()).item()
                )
                if covariance_operation_count < 0:
                    raise ValueError("symmetry/n_operations must be non-negative")
            if "spacegroup" in symmetry:
                covariance_spacegroup = _decode_scalar(
                    symmetry["spacegroup"][()], label="symmetry/spacegroup"
                )
        if "dJ_r" in handle:
            source_dataset = "dJ_r"
        elif "dJ_iso_r" in handle:
            source_dataset = "dJ_iso_r"
        else:
            raise KeyError("native scalar derivative requires dJ_r or dJ_iso_r")
        group = handle[source_dataset]
        if not isinstance(group, h5py.Group):
            raise TypeError(f"{source_dataset} must be an HDF5 group")

        targets = _integer_dataset(handle, "displacements/target_atom")
        if targets.ndim != 1 or targets.size == 0 or np.any(targets < 0):
            raise ValueError(
                "displacements/target_atom must be nonempty and non-negative"
            )
        if np.unique(targets).size != targets.size:
            raise ValueError("displacements/target_atom must not contain duplicates")
        axes = _axis_order(handle["displacements/axes"][()], label="displacements/axes")
        axis_permutation = tuple(axes.index(axis) for axis in ("x", "y", "z"))
        rp = _integer_dataset(handle, "displacements/Rp")
        if rp.ndim != 2 or rp.shape[1:] != (3,) or rp.shape[0] == 0:
            raise ValueError("displacements/Rp must have nonempty shape (nRp,3)")
        mesh_raw = _integer_dataset(handle, "basic_data/qmesh", shape=(3,))
        mesh = (int(mesh_raw[0]), int(mesh_raw[1]), int(mesh_raw[2]))
        if any(value <= 0 for value in mesh):
            raise ValueError("basic_data/qmesh must contain positive integers")
        _periodic_rp_index(rp, mesh)
        permutation, source_static_bonds = _dynamic_bond_permutation(handle, exchange)

        values_source = np.zeros(
            (targets.size, permutation.size, rp.shape[0], 3), dtype=np.float64
        )
        unit_label: str | None = None
        source_bond_count = int(source_static_bonds.size)
        for target_slot, target in enumerate(targets):
            for static_bond, source_bond in enumerate(permutation):
                if source_bond < 0:
                    continue
                candidates = (
                    f"dJ_r_m{int(target) + 1}_b{int(source_bond) + 1}",
                    f"m{int(target) + 1}_b{int(source_bond) + 1}",
                )
                key = next((name for name in candidates if name in group), None)
                if key is None:
                    raise KeyError(
                        f"missing {source_dataset} dataset for target {int(target)} "
                        f"and source bond {int(source_bond)}"
                    )
                dataset = group[key]
                if not isinstance(dataset, h5py.Dataset):
                    raise TypeError(f"{source_dataset}/{key} must be a dataset")
                raw = np.asarray(dataset)
                if raw.shape != (rp.shape[0], 3) or np.iscomplexobj(raw):
                    raise ValueError(
                        f"{source_dataset}/{key} must have real shape {(rp.shape[0], 3)}"
                    )
                numeric = np.asarray(raw, dtype=np.float64)
                if not np.all(np.isfinite(numeric)):
                    raise ValueError(
                        f"{source_dataset}/{key} contains non-finite values"
                    )
                this_unit, factor = _dataset_unit(
                    handle, group, dataset, label=f"{source_dataset}/{key}"
                )
                if unit_label is None:
                    unit_label = this_unit
                elif unit_label != this_unit:
                    raise ValueError("derivative datasets use inconsistent input units")
                values_source[target_slot, static_bond] = (
                    numeric[:, axis_permutation] * factor
                )

    tensor = values_source[..., None, None] * np.eye(3, dtype=np.float64)
    reciprocal = _reciprocal_derivative(
        tensor,
        exchange=exchange,
        rp=rp,
        mesh=mesh,
    )
    max_reciprocity_error = float(np.max(np.abs(tensor - reciprocal), initial=0.0))
    if not np.allclose(
        tensor,
        reciprocal,
        atol=reciprocity_atol,
        rtol=relative_tolerance,
    ):
        raise ValueError(
            "exchange derivative violates the Rp-aware directed-bond mate relation; "
            f"maximum error={max_reciprocity_error:.6g} meV/angstrom"
        )
    tensor = 0.5 * (tensor + reciprocal)
    # Both operands are exact scalar multiples of I.  Select one diagonal
    # component and rebuild the tensor so the canonical model retains the
    # exact tensor=dJ_iso*I invariant even for non-binary floating values.
    isotropic = np.array(tensor[..., 0, 0], dtype=np.float64, copy=True)
    tensor = isotropic[..., None, None] * np.eye(3, dtype=np.float64)

    residual_before = np.sum(isotropic, axis=(0, 2))
    max_before = float(np.max(np.abs(residual_before), initial=0.0))
    projected = policy is DerivativeASRPolicy.PROJECT and max_before > asr_tolerance
    if policy is DerivativeASRPolicy.FAIL and max_before > asr_tolerance:
        raise ValueError(
            "exchange derivative acoustic sum rule failed: "
            f"maximum residual={max_before:.6g} meV/angstrom exceeds "
            f"{asr_tolerance:.6g}"
        )
    if projected:
        correction = residual_before[source_static_bonds][None, :, None, :] / float(
            isotropic.shape[0] * isotropic.shape[2]
        )
        isotropic[:, source_static_bonds] -= correction
        tensor = isotropic[..., None, None] * np.eye(3, dtype=np.float64)
    residual_after = np.sum(isotropic, axis=(0, 2))
    max_after = float(np.max(np.abs(residual_after), initial=0.0))

    model = ExchangeDerivativeModel(
        source=source,
        static_exchange_source=exchange.source,
        representation=ExchangeRepresentation.ISOTROPIC,
        source_dataset=source_dataset,
        target_atom_indices=targets,
        rp_cell_shifts=rp,
        q_mesh_shape=mesh,
        isotropic_mev_per_ang=isotropic,
        tensor_mev_per_ang=tensor,
        source_static_bond_indices=source_static_bonds,
    )
    report = ExchangeDerivativeReport(
        representation=ExchangeRepresentation.ISOTROPIC,
        source_dataset=source_dataset,
        input_unit=unit_label or "unknown",
        max_reciprocity_error_mev_per_ang=max_reciprocity_error,
        asr_policy=policy,
        asr_tolerance_mev_per_ang=asr_tolerance,
        max_asr_residual_before_mev_per_ang=max_before,
        max_asr_residual_after_mev_per_ang=max_after,
        projected_asr=projected,
        static_bond_count=exchange.n_bonds,
        source_bond_count=source_bond_count,
        zero_filled_static_bond_count=exchange.n_bonds - source_bond_count,
        bond_coverage_complete=source_bond_count == exchange.n_bonds,
        covariant_symmetry_policy=covariance_policy,
        covariant_symmetry_applied=covariance_applied,
        max_covariance_residual_mev_per_ang=max_covariance_residual,
        covariant_symmetry_operation_count=covariance_operation_count,
        covariant_symmetry_spacegroup=covariance_spacegroup,
    )
    return model, report


__all__ = [
    "DerivativeASRPolicy",
    "ExchangeDerivativeModel",
    "ExchangeDerivativeReport",
    "load_exchange_derivative_h5",
    "read_exchange_derivative_q_mesh",
]
