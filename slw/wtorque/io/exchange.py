"""Validated exchange-field decomposition input routes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from slw.wtorque.basis import SpinOrder, canonicalize_spin_order, require_complex128
from slw.wtorque.config import BlochGauge, ExchangeExtraction
from slw.wtorque.errors import ExchangeDecompositionError, TimeReversalMetadataError
from slw.wtorque.gauge.atomic_gauge import wrap_hamiltonian
from slw.wtorque.gauge.kq_map import KQMap
from slw.wtorque.model.exchange_field import (
    split_collinear,
    split_explicit,
    split_time_reversal,
)
from slw.wtorque.model.onsite_soc import add_onsite_to_realspace
from slw.wtorque.model.spinor_wannier import SpinorWannierModel, fourier_matrix_batch


@dataclass(frozen=True)
class ExchangeFieldData:
    route: ExchangeExtraction
    h_trs_r: NDArray[np.complex128] | None = None
    h_xc_r: NDArray[np.complex128] | None = None
    h_trs_k: NDArray[np.complex128] | None = None
    h_xc_k: NDArray[np.complex128] | None = None
    reconstruction_residual: float = 0.0


def _canonical_dataset(
    group: h5py.Group,
    name: str,
    *,
    norb: int,
    spin_order: SpinOrder,
) -> NDArray[np.complex128]:
    if name not in group:
        raise KeyError(f"{group.name} is missing {name}")
    raw = require_complex128(f"{group.name}/{name}", group[name][...])
    return np.asarray(canonicalize_spin_order(raw, norb, spin_order), dtype=np.complex128)


def load_exchange_field(
    path: str | Path,
    model: SpinorWannierModel,
    *,
    route: ExchangeExtraction | str,
    spin_order: SpinOrder | str,
    tolerance: float = 1.0e-9,
) -> ExchangeFieldData:
    mode = ExchangeExtraction(route)
    order = SpinOrder(spin_order)
    with h5py.File(path, "r") as handle:
        electrons = handle["electrons"]
        if mode is ExchangeExtraction.EXPLICIT:
            h_trs = _canonical_dataset(electrons, "H_TRS_R", norb=model.norb, spin_order=order)
            h_trs = add_onsite_to_realspace(
                h_trs,
                model.R_vectors,
                model.onsite_soc_matrix,
            )
            h_xc = _canonical_dataset(electrons, "H_XC_R", norb=model.norb, spin_order=order)
            split_explicit(model.H_R, h_trs, h_xc, tolerance=tolerance)
            residual = float(np.max(np.abs(model.H_R - h_trs - h_xc)))
            return ExchangeFieldData(mode, h_trs_r=h_trs, h_xc_r=h_xc, reconstruction_residual=residual)

        if mode is ExchangeExtraction.COLLINEAR_SPLIT:
            if "H_up_R" not in electrons or "H_dn_R" not in electrons:
                raise KeyError("collinear split requires /electrons/H_up_R and H_dn_R")
            up = require_complex128("H_up_R", electrons["H_up_R"][...])
            down = require_complex128("H_dn_R", electrons["H_dn_R"][...])
            direction = np.asarray(electrons.attrs.get("collinear_reference_direction", [0.0, 0.0, 1.0]), dtype=np.float64)
            h_trs, h_xc = split_collinear(up, down, direction)
            h_trs = add_onsite_to_realspace(
                h_trs,
                model.R_vectors,
                model.onsite_soc_matrix,
            )
            residual = float(np.max(np.abs(model.H_R - h_trs - h_xc)))
            if residual > tolerance:
                raise ExchangeDecompositionError(
                    f"collinear split does not reconstruct H_R: residual={residual:.3e}"
                )
            return ExchangeFieldData(mode, h_trs_r=h_trs, h_xc_r=h_xc, reconstruction_residual=residual)

        if "time_reversal" not in handle:
            raise TimeReversalMetadataError("time-reversal extraction requires /time_reversal")
        tr = handle["time_reversal"]
        if "B_theta_k" not in tr or "minus_k_index" not in tr:
            raise TimeReversalMetadataError(
                "time-reversal extraction requires B_theta_k and minus_k_index"
            )
        sewing = _canonical_dataset(tr, "B_theta_k", norb=model.norb, spin_order=order)
        minus = np.asarray(tr["minus_k_index"][...], dtype=np.int64)
        if minus.shape != (model.kpoints.shape[0],) or np.any(minus < 0) or np.any(minus >= minus.size):
            raise TimeReversalMetadataError("minus_k_index is invalid")
        periodic = model.kpoints + model.kpoints[minus]
        if np.max(np.abs(periodic - np.rint(periodic))) > tolerance:
            raise TimeReversalMetadataError("minus_k_index does not map k to -k modulo G")
        full_k = model.hamiltonian_batch(model.kpoints)
        h_trs, h_xc = split_time_reversal(full_k, full_k[minus], sewing)
        residual = float(np.max(np.abs(full_k - h_trs - h_xc)))
        return ExchangeFieldData(mode, h_trs_k=h_trs, h_xc_k=h_xc, reconstruction_residual=residual)


def exchange_at_k_and_kq(
    data: ExchangeFieldData,
    model: SpinorWannierModel,
    q_red: object,
    mapping: KQMap,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return exchange fields in the actual atomic bases at k and k+q."""

    q = np.asarray(q_red, dtype=np.float64)
    if data.h_xc_r is not None:
        h_k = fourier_matrix_batch(
            model.R_vectors,
            data.h_xc_r,
            model.kpoints,
            orbital_centers=model.orbital_centers,
            bloch_gauge=model.bloch_gauge,
        )
        h_kq = fourier_matrix_batch(
            model.R_vectors,
            data.h_xc_r,
            model.kpoints + q,
            orbital_centers=model.orbital_centers,
            bloch_gauge=model.bloch_gauge,
        )
        return h_k, h_kq
    if data.h_xc_k is None:
        raise ExchangeDecompositionError("exchange field has neither real-space nor k-space data")
    h_k = data.h_xc_k
    if model.bloch_gauge is BlochGauge.ATOMIC_POSITION:
        h_kq = wrap_hamiltonian(
            h_k[mapping.indices],
            mapping.G_wrap,
            model.orbital_centers,
        )
    else:
        h_kq = h_k[mapping.indices]
    return np.asarray(h_k, dtype=np.complex128), np.asarray(h_kq, dtype=np.complex128)


__all__ = ["ExchangeFieldData", "exchange_at_k_and_kq", "load_exchange_field"]
