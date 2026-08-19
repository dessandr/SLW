"""
Build frozen reference density matrix P_ref in Wannier gauge from hr.dat.

Usage:
    python -m slw.interactions.build_pref_wannier \
        --in_dir /path/to/workdir \
        --up_hr spin_up_hr.dat \
        --dn_hr spin_dn_hr.dat \
        --kmesh 3 3 2 \
        --efermi 11.9
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np

from slw.core.wannier_io import read_wannier_hr

KB_EV_PER_K = 8.617333262145e-5


def _resolve_path(base_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(base_dir, path_value))


def _load_hr(path: str):
    parsed = read_wannier_hr(path)
    if len(parsed) == 3:
        dim, degens, h_dict = parsed
    elif len(parsed) == 4:
        dim, _nrpts, degens, h_dict = parsed
    else:
        raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")
    return int(dim), list(degens), h_dict


def _fermi_occupations(evals: np.ndarray, efermi: float, temperature_k: float) -> np.ndarray:
    if temperature_k <= 0.0:
        return (evals <= efermi).astype(np.float64)
    beta = 1.0 / (KB_EV_PER_K * float(temperature_k))
    x = np.clip((evals - efermi) * beta, -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)


def _build_kmesh(nkmesh):
    nk1, nk2, nk3 = nkmesh
    return np.array(
        [(i / nk1, j / nk2, k / nk3) for i in range(nk1) for j in range(nk2) for k in range(nk3)],
        dtype=np.float64,
    )


def _compute_hk(r_dict, kpts):
    r_list = np.array(list(r_dict.keys()), dtype=np.int32)
    h_r = np.array([r_dict[tuple(r)] for r in r_list], dtype=np.complex128)
    phase = np.exp(1j * 2.0 * np.pi * (kpts @ r_list.T))
    hk = np.einsum("kr,rij->kij", phase, h_r, optimize=True)
    return hk


def _build_pref_for_spin(r_dict, kpts, efermi, temperature_k):
    hk = _compute_hk(r_dict, kpts)
    nk, dim, _ = hk.shape
    p_ref = np.zeros((nk, dim, dim), dtype=np.complex128)

    trace_vals = np.zeros(nk, dtype=np.float64)
    herm_err = 0.0

    for ik in range(nk):
        evals, evecs = np.linalg.eigh(hk[ik])
        occ = _fermi_occupations(evals, efermi, temperature_k)
        p_ref[ik] = (evecs * occ[np.newaxis, :]) @ evecs.conj().T
        trace_vals[ik] = float(np.real(np.trace(p_ref[ik])))
        herm_err = max(herm_err, float(np.max(np.abs(p_ref[ik] - p_ref[ik].conj().T))))

    diag = {
        "trace_mean": float(np.mean(trace_vals)),
        "trace_min": float(np.min(trace_vals)),
        "trace_max": float(np.max(trace_vals)),
        "hermiticity_max_error": float(herm_err),
    }
    return p_ref, diag


def main():
    parser = argparse.ArgumentParser(description="Build frozen reference density matrix in Wannier gauge.")
    parser.add_argument("--in_dir", type=str, required=True, help="Input directory containing hr.dat files")
    parser.add_argument("--up_hr", type=str, required=True, help="Spin-up Wannier hr.dat filename or path")
    parser.add_argument("--dn_hr", type=str, required=True, help="Spin-down Wannier hr.dat filename or path")
    parser.add_argument("--kmesh", type=int, nargs=3, required=True, help="Uniform Monkhorst-like grid, e.g. 3 3 2")
    parser.add_argument("--temperature_K", type=float, default=0.0, help="Fermi temperature in K (default: 0)")
    parser.add_argument("--efermi", type=float, required=True, help="Fermi energy in eV")
    parser.add_argument("--out_prefix", type=str, default="pref_wannier", help="Output basename prefix")
    args = parser.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    up_hr = _resolve_path(in_dir, args.up_hr)
    dn_hr = _resolve_path(in_dir, args.dn_hr)

    if not os.path.exists(up_hr):
        raise FileNotFoundError(f"Spin-up hr.dat not found: {up_hr}")
    if not os.path.exists(dn_hr):
        raise FileNotFoundError(f"Spin-down hr.dat not found: {dn_hr}")

    dim_up, deg_up, h_up = _load_hr(up_hr)
    dim_dn, deg_dn, h_dn = _load_hr(dn_hr)
    if dim_up != dim_dn:
        raise ValueError(f"hr.dat dimensions differ: up={dim_up}, dn={dim_dn}")

    kpts = _build_kmesh(tuple(args.kmesh))
    p_up, diag_up = _build_pref_for_spin(h_up, kpts, args.efermi, args.temperature_K)
    p_dn, diag_dn = _build_pref_for_spin(h_dn, kpts, args.efermi, args.temperature_K)

    out_prefix = os.path.abspath(args.out_prefix)
    out_npz = out_prefix + ".npz"
    out_meta = out_prefix + ".meta.json"

    np.savez_compressed(
        out_npz,
        p_ref_wannier_up=p_up,
        p_ref_wannier_dn=p_dn,
        kmesh=np.array(args.kmesh, dtype=np.int32),
        efermi_eV=np.array(float(args.efermi), dtype=np.float64),
        temperature_K=np.array(float(args.temperature_K), dtype=np.float64),
    )

    meta = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "in_dir": in_dir,
        "up_hr": up_hr,
        "dn_hr": dn_hr,
        "kmesh": [int(x) for x in args.kmesh],
        "nk_points": int(len(kpts)),
        "dim": int(dim_up),
        "efermi_eV": float(args.efermi),
        "temperature_K": float(args.temperature_K),
        "diagnostics": {
            "up": diag_up,
            "dn": diag_dn,
            "trace_mean_difference": float(abs(diag_up["trace_mean"] - diag_dn["trace_mean"])),
        },
        "output": {
            "npz": out_npz,
            "meta_json": out_meta,
        },
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved NPZ:  {out_npz}")
    print(f"Saved META: {out_meta}")


if __name__ == "__main__":
    main()
