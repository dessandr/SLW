"""DFPT HDF5 provider, canonical spinor lifting, and perturbation streaming."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import SpinOrder, canonicalize_spin_order, require_complex128
from slw.wtorque.config import Normalization, SpinorLift
from slw.wtorque.errors import NormalizationError


def lift_spin_scalar(value: object) -> NDArray[np.complex128]:
    scalar = require_complex128("spin-scalar DFPT vertex", value)
    if scalar.shape[-1] != scalar.shape[-2]:
        raise ValueError("spin-scalar DFPT matrices must be square")
    norb = scalar.shape[-1]
    leading = scalar.shape[:-2]
    lifted = np.einsum("...ij,ab->...iajb", scalar, np.eye(2, dtype=np.complex128))
    return np.asarray(lifted.reshape(*leading, 2 * norb, 2 * norb), dtype=np.complex128)


def lift_collinear(value_up: object, value_down: object) -> NDArray[np.complex128]:
    up = require_complex128("spin-up DFPT vertex", value_up)
    down = require_complex128("spin-down DFPT vertex", value_down)
    if up.shape != down.shape or up.shape[-1] != up.shape[-2]:
        raise ValueError("spin-up/down DFPT matrices must have matching square shapes")
    leading = up.shape[:-2]
    norb = up.shape[-1]
    result = np.zeros((*leading, norb, 2, norb, 2), dtype=np.complex128)
    result[..., :, 0, :, 0] = up
    result[..., :, 1, :, 1] = down
    return result.reshape(*leading, 2 * norb, 2 * norb)


class HDF5DFPTProvider:
    """Chunked access to canonical ``g[k,pert,nw,nw]`` matrices."""

    def __init__(
        self,
        path: str | Path,
        *,
        normalization: Normalization | str,
        spinor_lift: SpinorLift | str,
        spin_order: SpinOrder | str,
        norb: int,
        g_xc_dataset: str | None = "g_xc_cart",
    ) -> None:
        self.path = Path(path)
        self.normalization = Normalization(normalization)
        self.spinor_lift = SpinorLift(spinor_lift)
        self.spin_order = SpinOrder(spin_order)
        self.norb = int(norb)
        self.g_xc_dataset = g_xc_dataset
        self.handle = h5py.File(self.path, "r")
        if "dfpt/qpoints" not in self.handle:
            self.close()
            raise KeyError("DFPT input must contain /dfpt/qpoints")
        self.qpoints = np.asarray(self.handle["dfpt/qpoints"][...])
        if self.qpoints.dtype != np.float64 or self.qpoints.ndim != 2 or self.qpoints.shape[1] != 3:
            self.close()
            raise TypeError("/dfpt/qpoints must be float64 with shape (nq,3)")
        declared = self.handle["dfpt"].attrs.get("normalization")
        if declared is None:
            self.close()
            raise NormalizationError("/dfpt must declare the normalization attribute")
        text = declared.decode() if isinstance(declared, bytes) else str(declared)
        if Normalization(text) is not self.normalization:
            self.close()
            raise NormalizationError(
                f"DFPT file normalization {text!r} does not match config {self.normalization.value!r}"
            )
        self.pert_atom = None
        self.pert_cart = None
        if self.normalization is not Normalization.PHONON_ZERO_POINT_MODE:
            if "dfpt/pert_atom" not in self.handle or "dfpt/pert_cart" not in self.handle:
                self.close()
                raise KeyError("Cartesian DFPT input requires /dfpt/pert_atom and /dfpt/pert_cart")
            self.pert_atom = np.asarray(self.handle["dfpt/pert_atom"][...], dtype=np.int64)
            self.pert_cart = np.asarray(self.handle["dfpt/pert_cart"][...], dtype=np.int64)
            if self.pert_atom.shape != self.pert_cart.shape or np.any((self.pert_cart < 0) | (self.pert_cart > 2)):
                self.close()
                raise ValueError("DFPT perturbation atom/Cartesian metadata is invalid")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            self.handle.close()

    def _group(self, iq: int) -> h5py.Group:
        name = f"dfpt/q_{int(iq):06d}"
        if name not in self.handle:
            raise KeyError(f"DFPT input is missing /{name}")
        return self.handle[name]

    def _read_native(self, dataset: h5py.Dataset, selection: object = np.s_[:]) -> NDArray[np.complex128]:
        raw = require_complex128(dataset.name, dataset[selection])
        return np.asarray(canonicalize_spin_order(raw, self.norb, self.spin_order), dtype=np.complex128)

    def g(self, iq: int) -> NDArray[np.complex128]:
        group = self._group(iq)
        dataset_name = "g_mode" if self.normalization is Normalization.PHONON_ZERO_POINT_MODE else "g_cart"
        if self.spinor_lift is SpinorLift.NATIVE_SPINOR:
            if dataset_name not in group:
                raise KeyError(f"{group.name} is missing {dataset_name}")
            result = self._read_native(group[dataset_name])
        elif self.spinor_lift is SpinorLift.SPIN_SCALAR:
            result = lift_spin_scalar(group[dataset_name][...])
        else:
            base = "g_mode" if dataset_name == "g_mode" else "g_cart"
            result = lift_collinear(group[f"{base}_up"][...], group[f"{base}_down"][...])
        if result.shape[-2:] != (2 * self.norb, 2 * self.norb) or result.ndim != 4:
            raise ValueError(
                f"canonical DFPT data must have shape (nk,npert,{2*self.norb},{2*self.norb}); got {result.shape}"
            )
        return result

    def g_cart_chunk(self, iq: int, ik: int, perturbation_slice: slice) -> NDArray[np.complex128]:
        values = self.g(iq)
        return np.asarray(values[int(ik), perturbation_slice], dtype=np.complex128)

    def g_xc(self, iq: int) -> NDArray[np.complex128] | None:
        """Read an explicitly selected exchange response, never infer it from g.

        The dataset path is relative to each ``/dfpt/q_NNNNN`` group. An
        absolute path may use ``{iq:06d}`` to select the corresponding q group.
        Its units, perturbation order and final-state gauge are those of g.
        """

        group = self._group(iq)
        if self.g_xc_dataset is None:
            return None
        name = str(self.g_xc_dataset).format(iq=int(iq))
        if name not in group:
            return None
        if self.spinor_lift is not SpinorLift.NATIVE_SPINOR:
            raise ValueError("g_XC currently requires native spinor input")
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"g_XC path {dataset.name} must select a dataset")
        total_name = "g_mode" if self.normalization is Normalization.PHONON_ZERO_POINT_MODE else "g_cart"
        if total_name in group and dataset.id == group[total_name].id:
            raise ValueError("g_XC must be a separate exchange-resolved response, not the total DFPT g")
        result = self._read_native(dataset)
        if (
            result.ndim != 4
            or min(result.shape) < 1
            or result.shape[-2:] != (2 * self.norb, 2 * self.norb)
            or (total_name in group and result.shape != group[total_name].shape)
            or (self.pert_atom is not None and result.shape[1] != self.pert_atom.size)
        ):
            raise ValueError("g_XC must have (nk,npert,nw,nw) shape matching perturbation metadata and any native DFPT g")
        if "dfpt/kpoints" in self.handle and result.shape[0] != self.handle["dfpt/kpoints"].shape[0]:
            raise ValueError("g_XC k-point count does not match /dfpt/kpoints")
        if not np.isfinite(result).all():
            raise ValueError("g_XC contains non-finite values")
        return result


def expand_cartesian_perturbations(
    values: object,
    pert_atom: object,
    pert_cart: object,
    *,
    natom: int | None = None,
) -> NDArray[np.complex128]:
    array = require_complex128("Cartesian perturbation result", values)
    atoms = np.asarray(pert_atom, dtype=np.int64)
    carts = np.asarray(pert_cart, dtype=np.int64)
    if atoms.shape != carts.shape or array.shape[-1] != atoms.size:
        raise ValueError("perturbation result and metadata dimensions do not match")
    count = int(atoms.max(initial=-1) + 1) if natom is None else int(natom)
    if count < 1 or np.any(atoms < 0) or np.any(atoms >= count) or np.any((carts < 0) | (carts > 2)):
        raise ValueError("perturbation atom/Cartesian indices are out of range")
    pairs = np.column_stack((atoms, carts))
    if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
        raise ValueError("duplicate atom/Cartesian perturbations are ambiguous")
    output = np.zeros((*array.shape[:-1], count, 3), dtype=np.complex128)
    output[..., atoms, carts] = array
    return output


__all__ = [
    "HDF5DFPTProvider",
    "expand_cartesian_perturbations",
    "lift_collinear",
    "lift_spin_scalar",
]
