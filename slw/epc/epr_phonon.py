"""Production qe2pert EPR phonon metadata and polar corrections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

_TWO_PI = 2.0 * np.pi
_FOUR_PI = 4.0 * np.pi
_E2_RY = 2.0
_POLAR_GMAX = 14.0


def _required_dataset(handle: h5py.File, path: str) -> h5py.Dataset:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        raise KeyError(f"missing required EPR phonon dataset {path}")
    return handle[path]


def read_epr_phonon_metadata(path: str | Path) -> dict[str, Any]:
    """Read only structure, mass, mesh, and optional polar EPR metadata."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"EPR HDF5 not found: {source}")
    with h5py.File(source, "r") as handle:
        at = np.asarray(_required_dataset(handle, "basic_data/at"), dtype=np.float64).T
        tau_cart = np.asarray(
            _required_dataset(handle, "basic_data/tau"), dtype=np.float64
        )
        if at.shape != (3, 3) or tau_cart.ndim != 2 or tau_cart.shape[1:] != (3,):
            raise ValueError("EPR lattice/atomic positions have invalid shapes")
        tau = np.linalg.solve(at, tau_cart.T).T
        mass = np.asarray(
            _required_dataset(handle, "basic_data/mass"), dtype=np.float64
        ).reshape(-1)
        qmesh = tuple(
            int(value)
            for value in np.asarray(
                _required_dataset(handle, "basic_data/qc_dim"), dtype=np.int64
            )
        )
        nat = int(_required_dataset(handle, "basic_data/nat")[()])
        alat = float(_required_dataset(handle, "basic_data/alat")[()])
        lpolar = bool(handle["basic_data/lpolar"][()]) if "basic_data/lpolar" in handle else False
        system_2d = (
            bool(handle["basic_data/system_2d"][()])
            if "basic_data/system_2d" in handle
            else False
        )
        if system_2d and "basic_data/thickness_2d" in handle:
            thickness_2d = float(handle["basic_data/thickness_2d"][()])
        elif system_2d:
            thickness_2d = 6.0 / 0.52917721092
        else:
            thickness_2d = -1.0
        result: dict[str, Any] = {
            "source": source,
            "at": at,
            "tau": tau,
            "tau_cart": tau_cart,
            "mass": mass,
            "qc_dim": qmesh,
            "nat": nat,
            "alat": alat,
            "lpolar": lpolar,
            "system_2d": system_2d,
            "thickness_2d": thickness_2d,
            "loto_alpha": (
                float(handle["basic_data/loto_alpha"][()])
                if "basic_data/loto_alpha" in handle
                else 1.0
            ),
        }
        if lpolar:
            result.update(
                {
                    "bg": np.asarray(
                        _required_dataset(handle, "basic_data/bg"),
                        dtype=np.float64,
                    ).T,
                    "epsil": np.asarray(
                        _required_dataset(handle, "basic_data/epsil"),
                        dtype=np.float64,
                    ).T,
                    "zstar": np.asarray(
                        _required_dataset(handle, "basic_data/zstar"),
                        dtype=np.float64,
                    ),
                    "volume": float(
                        _required_dataset(handle, "basic_data/volume")[()]
                    ),
                }
            )

    if len(qmesh) != 3 or any(value <= 0 for value in qmesh):
        raise ValueError("EPR basic_data/qc_dim must contain three positive integers")
    if mass.shape != (nat,) or np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("EPR masses must contain nat positive finite values")
    if not np.isfinite(alat) or alat <= 0.0:
        raise ValueError("EPR basic_data/alat must be positive and finite")
    return result


def apply_loto_override(meta: dict[str, Any], mode: str = "auto") -> dict[str, Any]:
    """Return metadata with an explicit auto/none/2d/3d LO-TO policy."""

    result = dict(meta)
    selected = str(mode).strip().lower()
    if selected in {"auto", ""}:
        return result
    if selected in {"none", "off"}:
        result["lpolar"] = False
        return result
    if selected == "2d":
        result["lpolar"] = True
        result["system_2d"] = True
        if result.get("thickness_2d", -1.0) <= 0.0:
            result["thickness_2d"] = 6.0 / 0.52917721092
        return result
    if selected == "3d":
        result["lpolar"] = True
        result["system_2d"] = False
        result["thickness_2d"] = -1.0
        return result
    raise ValueError("phonon_loto must be one of auto, none, 2d, or 3d")


def phase_table(points_frac: object, cell_shifts: object) -> NDArray[np.complex128]:
    """Return ``exp(+i 2pi q.R)`` for row-vector q and R arrays."""

    points = np.asarray(points_frac, dtype=np.float64)
    shifts = np.asarray(cell_shifts, dtype=np.float64)
    return np.exp(2j * np.pi * (points @ shifts.T))


def _cart_from_cryst(vectors: object, reciprocal: object) -> NDArray[np.float64]:
    return np.asarray(vectors, dtype=np.float64) @ np.asarray(
        reciprocal, dtype=np.float64
    ).T


def _polar_g_vectors(extents: object) -> NDArray[np.float64]:
    n1, n2, n3 = (int(value) for value in np.asarray(extents).reshape(3))
    return np.asarray(
        [
            (i, j, k)
            for i in range(-n1, n1 + 1)
            for j in range(-n2, n2 + 1)
            for k in range(-n3, n3 + 1)
        ],
        dtype=np.float64,
    )


def _polar_extents(meta: dict[str, Any]) -> NDArray[np.int64]:
    alpha4 = 4.0 * float(meta["loto_alpha"])
    maximum = _POLAR_GMAX * alpha4
    reciprocal = np.asarray(meta["bg"], dtype=np.float64)
    extents = np.asarray(
        [
            int(np.ceil(np.sqrt(maximum / float(reciprocal[:, axis] @ reciprocal[:, axis]))))
            for axis in range(3)
        ],
        dtype=np.int64,
    )
    extents[np.asarray(meta["qc_dim"], dtype=np.int64) < 2] = 0
    return extents


def _dynamical_longrange_raw_3d(
    meta: dict[str, Any], q_point: object
) -> NDArray[np.complex128]:
    nat = int(meta["nat"])
    epsilon = np.asarray(meta["epsil"], dtype=np.float64)
    zstar = np.asarray(meta["zstar"], dtype=np.float64)
    tau = np.asarray(meta["tau_cart"], dtype=np.float64)
    alpha4 = 4.0 * float(meta["loto_alpha"])
    maximum = _POLAR_GMAX * alpha4
    qg_cryst = np.asarray(q_point, dtype=np.float64)[None, :] + _polar_g_vectors(
        _polar_extents(meta)
    )
    qg = _cart_from_cryst(qg_cryst, meta["bg"])
    dielectric_norm = np.einsum("gi,ij,gj->g", qg, epsilon, qg, optimize=True)
    keep = (dielectric_norm >= 1.0e-14) & (dielectric_norm <= maximum)
    qg = qg[keep]
    dielectric_norm = dielectric_norm[keep]
    result = np.zeros((nat * (nat + 1) // 2, 3, 3), dtype=np.complex128)
    if qg.shape[0] == 0:
        return result
    weights = np.exp(-dielectric_norm / alpha4) / dielectric_norm
    outer = qg[:, :, None] * qg[:, None, :] * weights[:, None, None]
    pair = 0
    for atom_j in range(nat):
        for atom_i in range(atom_j + 1):
            phase = np.exp(1j * _TWO_PI * (qg @ (tau[atom_i] - tau[atom_j])))
            contracted = np.einsum("g,gij->ij", phase, outer, optimize=True)
            result[pair] = zstar[atom_i].T @ contracted @ zstar[atom_j]
            pair += 1
    return result * (_FOUR_PI * _E2_RY / float(meta["volume"]))


def _dynamical_longrange_raw_2d(
    meta: dict[str, Any], q_point: object
) -> NDArray[np.complex128]:
    nat = int(meta["nat"])
    epsilon = np.asarray(meta["epsil"], dtype=np.float64)
    zstar = np.asarray(meta["zstar"], dtype=np.float64)
    tau = np.asarray(meta["tau_cart"], dtype=np.float64)
    reciprocal = np.asarray(meta["bg"], dtype=np.float64)
    inverse_alat_factor = _TWO_PI / float(meta["alat"])
    alpha4 = 4.0 * float(meta["loto_alpha"])
    maximum = _POLAR_GMAX * alpha4
    factor = (
        _FOUR_PI
        * _E2_RY
        / float(meta["volume"])
        * 0.5
        * float(meta["alat"])
        / reciprocal[2, 2]
    )
    effective = epsilon[:2, :2] * (0.5 * _TWO_PI / reciprocal[2, 2])
    effective[0, 0] -= 0.5 * _TWO_PI / reciprocal[2, 2]
    effective[1, 1] -= 0.5 * _TWO_PI / reciprocal[2, 2]
    qg_cryst = np.asarray(q_point, dtype=np.float64)[None, :] + _polar_g_vectors(
        _polar_extents(meta)
    )
    qg = _cart_from_cryst(qg_cryst, reciprocal)
    norm_sq = np.einsum("gi,gi->g", qg, qg, optimize=True)
    inplane_sq = np.einsum("gi,gi->g", qg[:, :2], qg[:, :2], optimize=True)
    screening = np.zeros_like(norm_sq)
    nonzero = inplane_sq > 1.0e-8
    screening[nonzero] = np.einsum(
        "gi,ij,gj->g", qg[nonzero, :2], effective, qg[nonzero, :2], optimize=True
    ) / inplane_sq[nonzero]
    keep = (norm_sq >= 1.0e-14) & (norm_sq <= maximum)
    qg = qg[keep]
    norm_sq = norm_sq[keep]
    screening = screening[keep]
    result = np.zeros((nat * (nat + 1) // 2, 3, 3), dtype=np.complex128)
    if qg.shape[0] == 0:
        return result
    norm = np.sqrt(norm_sq)
    weights = (
        inverse_alat_factor
        * np.exp(-norm_sq / alpha4)
        / norm
        / (1.0 + screening * norm)
    )
    outer = qg[:, :, None] * qg[:, None, :] * weights[:, None, None]
    pair = 0
    for atom_j in range(nat):
        for atom_i in range(atom_j + 1):
            phase = np.exp(1j * _TWO_PI * (qg @ (tau[atom_i] - tau[atom_j])))
            contracted = np.einsum("g,gij->ij", phase, outer, optimize=True)
            result[pair] = zstar[atom_i].T @ contracted @ zstar[atom_j]
            pair += 1
    return result * factor


def dynamical_longrange_raw(
    meta: dict[str, Any], q_point: object
) -> NDArray[np.complex128]:
    if float(meta["thickness_2d"]) > 0.0:
        return _dynamical_longrange_raw_2d(meta, q_point)
    return _dynamical_longrange_raw_3d(meta, q_point)


def polar_onsite_correction(meta: dict[str, Any]) -> NDArray[np.float64]:
    """Return the real onsite term that restores polar translational balance."""

    nat = int(meta["nat"])
    gamma = dynamical_longrange_raw(meta, np.zeros(3, dtype=np.float64))
    if np.max(np.abs(gamma.imag), initial=0.0) > 1.0e-16:
        raise ValueError("onsite polar correction is not real")
    dense = np.zeros((nat, nat, 3, 3), dtype=np.float64)
    pair = 0
    for atom_j in range(nat):
        for atom_i in range(atom_j + 1):
            block = gamma[pair].real
            if atom_i == atom_j:
                dense[atom_i, atom_i] = 0.5 * (block + block.T)
            else:
                dense[atom_i, atom_j] = block
                dense[atom_j, atom_i] = block.T
            pair += 1
    return -np.sum(dense, axis=1)


def dynamical_longrange(
    meta: dict[str, Any],
    q_point: object,
    onsite: NDArray[np.float64],
) -> NDArray[np.complex128]:
    result = dynamical_longrange_raw(meta, q_point)
    for atom in range(int(meta["nat"])):
        result[atom * (atom + 3) // 2] += onsite[atom]
    return result


__all__ = [
    "apply_loto_override",
    "dynamical_longrange",
    "phase_table",
    "polar_onsite_correction",
    "read_epr_phonon_metadata",
]
