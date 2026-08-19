import os
import json
import argparse
from datetime import datetime

import numpy as np

from slw.core.wannier_io import read_wannier_hr, write_wannier_hr


def _resolve_path(base_dir, path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(base_dir, path_value))


def _read_hr_compat(path):
    parsed = read_wannier_hr(path)
    if len(parsed) == 3:
        dim, degens, h_dict = parsed
        return dim, degens, h_dict
    if len(parsed) == 4:
        dim, _nrpts, degens, h_dict = parsed
        return dim, degens, h_dict
    raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")


def _parse_index_spec(text, nmax):
    if text is None:
        return []
    idx = set()
    for tok in [x.strip() for x in text.split(",") if x.strip()]:
        if "-" in tok:
            a, b = tok.split("-", 1)
            ia, ib = int(a), int(b)
            if ia > ib:
                ia, ib = ib, ia
            for v in range(ia, ib + 1):
                idx.add(v)
        else:
            idx.add(int(tok))
    out = sorted(idx)
    for v in out:
        if v < 0 or v >= nmax:
            raise ValueError(f"Index {v} out of range [0, {nmax - 1}]")
    return out


def _build_pd_mask(dim, p_idx, d_idx):
    mask = np.zeros((dim, dim), dtype=np.float64)
    for i in p_idx:
        for j in d_idx:
            mask[i, j] = 1.0
            mask[j, i] = 1.0
    return mask


def _enforce_offdiag(mat):
    out = mat.copy()
    np.fill_diagonal(out, 0.0)
    return out


def _load_p_ref_npz(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    keys = set(data.files)

    def pick_for_spin(spin):
        candidates = [
            f"p_ref_wannier_{spin}",
            f"p_ref_{spin}",
            "p_ref_wannier",
            "p_ref",
        ]
        for key in candidates:
            if key in keys:
                return data[key]
        raise KeyError(
            f"Could not find p_ref array for spin '{spin}' in {npz_path}. "
            f"Available keys: {sorted(keys)}"
        )

    return {"up": pick_for_spin("up"), "down": pick_for_spin("down")}


def _normalize_p_ref_array(arr, spin, dim):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        if arr.shape != (dim, dim):
            raise ValueError(f"p_ref_{spin} must have shape ({dim},{dim}) or (Nk,{dim},{dim}); got {arr.shape}")
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        if arr.shape[1:] != (dim, dim):
            raise ValueError(f"p_ref_{spin} must have shape (Nk,{dim},{dim}); got {arr.shape}")
        return arr
    raise ValueError(f"p_ref_{spin} must be 2D or 3D array, got shape {arr.shape}")


def _build_delta_from_p_ref_wannier(p_ref_dict, v_value, lambda_v=1.0, pd_mask=None, offdiag_only=False, enforce_hermitian=False):
    result = {}
    for spin in ("up", "down"):
        p_ref = np.asarray(p_ref_dict[spin], dtype=np.complex128, copy=False)
        if p_ref.ndim == 2:
            p_ref = p_ref[np.newaxis, ...]
        if p_ref.ndim != 3 or p_ref.shape[1] != p_ref.shape[2]:
            raise ValueError(f"p_ref_{spin} must be a 2D square matrix or a 3D (Nk,dim,dim) tensor, got shape={p_ref.shape}")

        delta_k = -float(v_value) * float(lambda_v) * p_ref
        if pd_mask is not None:
            delta_k = delta_k * pd_mask[np.newaxis, :, :]
        if offdiag_only:
            delta_k = delta_k.copy()
            for ik in range(delta_k.shape[0]):
                delta_k[ik] = _enforce_offdiag(delta_k[ik])
        delta_r0 = np.mean(delta_k, axis=0)
        if enforce_hermitian:
            delta_r0 = 0.5 * (delta_r0 + delta_r0.conj().T)

        result[spin] = {
            "delta_k": delta_k,
            "delta_r0": delta_r0,
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build a scalar-V Hamiltonian correction from P_ref and write corrected Wannier hr.dat files."
    )
    parser.add_argument("--workdir", type=str, default=None, help="Workflow root directory (default: current directory)")
    parser.add_argument("--in_dir", type=str, required=True, help="Input directory containing hr.dat files")
    parser.add_argument("--up_hr", type=str, required=True, help="Spin-up hr.dat filename")
    parser.add_argument("--dn_hr", type=str, required=True, help="Spin-down hr.dat filename")
    parser.add_argument("--p_ref_wannier", type=str, required=True, help="NPZ from script#1 containing p_ref_wannier arrays")
    parser.add_argument("--v_value", type=float, required=True, help="Scalar V (eV)")
    parser.add_argument("--p_idx", type=str, default=None, help="Wannier p-orbital indices, e.g. '10-15'")
    parser.add_argument("--d_idx", type=str, default=None, help="Wannier d-orbital indices, e.g. '0-9'")
    parser.add_argument("--offdiag_only", action="store_true", help="Apply only off-diagonal correction")
    parser.add_argument("--lambda_v", type=float, default=1.0, help="Global scale factor for correction")
    parser.add_argument("--enforce_hermitian", action="store_true", help="Project delta_R0 to Hermitian before applying")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: in_dir)")
    parser.add_argument("--out_prefix", type=str, required=True, help="Output hr prefix")
    args = parser.parse_args()

    cwd = os.getcwd()
    workdir = _resolve_path(cwd, args.workdir) if args.workdir else cwd
    in_dir = _resolve_path(workdir, args.in_dir)
    out_dir = _resolve_path(workdir, args.out_dir) if args.out_dir else in_dir
    os.makedirs(out_dir, exist_ok=True)

    up_path = _resolve_path(in_dir, args.up_hr)
    dn_path = _resolve_path(in_dir, args.dn_hr)
    pref_path = _resolve_path(workdir, args.p_ref_wannier)

    dim_up, degens_up, h_up = _read_hr_compat(up_path)
    dim_dn, degens_dn, h_dn = _read_hr_compat(dn_path)
    if dim_up != dim_dn:
        raise ValueError(f"Up/Dn hr dimensions differ: {dim_up} vs {dim_dn}")
    if set(h_up.keys()) != set(h_dn.keys()):
        raise ValueError("Up/Dn hr R-lists differ. Align hr files before correction.")
    dim = dim_up
    r_list = sorted(h_up.keys())

    p_ref_loaded = _load_p_ref_npz(pref_path)
    p_idx = _parse_index_spec(args.p_idx, dim) if args.p_idx else []
    d_idx = _parse_index_spec(args.d_idx, dim) if args.d_idx else []
    if (p_idx and not d_idx) or (d_idx and not p_idx):
        raise ValueError("Provide both --p_idx and --d_idx, or neither.")
    pd_mask = _build_pd_mask(dim, p_idx, d_idx) if (p_idx and d_idx) else None

    built = _build_delta_from_p_ref_wannier(
        p_ref_loaded,
        args.v_value,
        lambda_v=args.lambda_v,
        pd_mask=pd_mask,
        offdiag_only=args.offdiag_only,
        enforce_hermitian=args.enforce_hermitian,
    )

    if (0, 0, 0) not in h_up:
        raise KeyError("R=(0,0,0) block not found in hr.dat. Cannot apply onsite correction.")

    h_up_corr = {r: h.copy() for r, h in h_up.items()}
    h_dn_corr = {r: h.copy() for r, h in h_dn.items()}
    h_up_corr[(0, 0, 0)] = h_up_corr[(0, 0, 0)] + built["up"]["delta_r0"]
    h_dn_corr[(0, 0, 0)] = h_dn_corr[(0, 0, 0)] + built["down"]["delta_r0"]

    out_up = os.path.join(out_dir, f"{args.out_prefix}_up_hr.dat")
    out_dn = os.path.join(out_dir, f"{args.out_prefix}_dn_hr.dat")
    out_meta = os.path.join(out_dir, f"{args.out_prefix}_v.meta.json")

    write_wannier_hr(out_up, dim, degens_up, h_up_corr, header="SLW scalar-V corrected (up)")
    write_wannier_hr(out_dn, dim, degens_dn, h_dn_corr, header="SLW scalar-V corrected (down)")

    base_up_norm = float(np.linalg.norm(h_up[(0, 0, 0)]))
    base_dn_norm = float(np.linalg.norm(h_dn[(0, 0, 0)]))
    delta_up_k = built["up"]["delta_k"]
    delta_dn_k = built["down"]["delta_k"]
    delta_up_r0 = built["up"]["delta_r0"]
    delta_dn_r0 = built["down"]["delta_r0"]

    meta = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "mode": "build_v_hamiltonian",
        "input": {
            "up_hr": up_path,
            "dn_hr": dn_path,
            "p_ref_wannier": pref_path,
        },
        "settings": {
            "v_value_eV": float(args.v_value),
            "lambda_v": float(args.lambda_v),
            "p_idx": args.p_idx,
            "d_idx": args.d_idx,
            "offdiag_only": bool(args.offdiag_only),
            "enforce_hermitian": bool(args.enforce_hermitian),
            "apply_R": [0, 0, 0],
        },
        "diagnostics": {
            "dim": int(dim),
            "nR": int(len(r_list)),
            "p_ref_shape_up": list(np.asarray(p_ref_loaded["up"]).shape),
            "p_ref_shape_dn": list(np.asarray(p_ref_loaded["down"]).shape),
            "delta_k_norm_up": float(np.linalg.norm(delta_up_k)),
            "delta_k_norm_dn": float(np.linalg.norm(delta_dn_k)),
            "delta_r0_norm_up": float(np.linalg.norm(delta_up_r0)),
            "delta_r0_norm_dn": float(np.linalg.norm(delta_dn_r0)),
            "base_r0_norm_up": base_up_norm,
            "base_r0_norm_dn": base_dn_norm,
            "delta_over_base_r0_up": float(np.linalg.norm(delta_up_r0) / (base_up_norm + 1e-16)),
            "delta_over_base_r0_dn": float(np.linalg.norm(delta_dn_r0) / (base_dn_norm + 1e-16)),
        },
        "output": {
            "up_hr": out_up,
            "dn_hr": out_dn,
        },
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved corrected up hr: {out_up}")
    print(f"Saved corrected dn hr: {out_dn}")
    print(f"Saved metadata:        {out_meta}")


if __name__ == "__main__":
    main()
