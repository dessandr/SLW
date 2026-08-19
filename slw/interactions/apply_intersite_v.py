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


def _load_wannier_matrix_npz(npz_path, base_key):
    data = np.load(npz_path, allow_pickle=False)
    keys = set(data.files)
    result = {}
    key_up = f"{base_key}_up"
    key_dn = f"{base_key}_dn"
    key_common = base_key

    if key_up in keys:
        result["up"] = data[key_up]
    if key_dn in keys:
        result["down"] = data[key_dn]
    if key_common in keys:
        common = data[key_common]
        result.setdefault("up", common)
        result.setdefault("down", common)
    if not result:
        raise KeyError(
            f"No supported key in npz for '{base_key}'. "
            f"Use one of: {base_key}, {base_key}_up, {base_key}_dn"
        )
    return result


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
            raise ValueError(f"Index {v} out of range [0, {nmax-1}]")
    return out


def _build_pd_mask(dim, p_idx, d_idx):
    mask = np.zeros((dim, dim), dtype=np.float64)
    if not p_idx or not d_idx:
        return mask
    for i in p_idx:
        for j in d_idx:
            mask[i, j] = 1.0
            mask[j, i] = 1.0
    return mask


def _enforce_offdiag(mat):
    out = mat.copy()
    np.fill_diagonal(out, 0.0)
    return out


def _build_delta_from_p_ref_wannier(p_ref_dict, v_value, pd_mask=None, offdiag_only=False):
    result = {}
    for spin in ("up", "down"):
        p_ref = p_ref_dict[spin].astype(np.complex128, copy=False)
        if p_ref.ndim != 2 or p_ref.shape[0] != p_ref.shape[1]:
            raise ValueError(f"p_ref_{spin} must be 2D square matrix, got shape={p_ref.shape}")
        delta = -float(v_value) * p_ref
        if pd_mask is not None:
            delta = delta * pd_mask
        if offdiag_only:
            delta = _enforce_offdiag(delta)
        result[spin] = 0.5 * (delta + delta.conj().T)
    return result


def _enforce_hermitian_dict(h_dict):
    out = {}
    for r, h in h_dict.items():
        out[r] = 0.5 * (h + h.conj().T)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Apply perturbative intersite-V correction directly in Wannier gauge."
    )
    parser.add_argument("--workdir", type=str, default=None, help="Workflow root directory (default: current directory)")
    parser.add_argument("--in_dir", type=str, default=".", help="Input directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: in_dir)")
    parser.add_argument("--up_hr", type=str, required=True, help="Spin-up hr.dat filename")
    parser.add_argument("--dn_hr", type=str, required=True, help="Spin-down hr.dat filename")
    parser.add_argument("--delta_wannier", type=str, default=None, help="NPZ with delta_h_wannier(_up/_dn) in Wannier gauge")
    parser.add_argument("--p_ref_wannier", type=str, default=None, help="NPZ with p_ref_wannier(_up/_dn) in Wannier gauge")
    parser.add_argument("--v_value", type=float, default=None, help="Scalar V (eV) for building delta = -V * P_ref_wannier")
    parser.add_argument("--p_idx", type=str, default=None, help="Wannier p-orbital indices, e.g. '10-15'")
    parser.add_argument("--d_idx", type=str, default=None, help="Wannier d-orbital indices, e.g. '0-9'")
    parser.add_argument("--lambda_v", type=float, default=1.0, help="Global scale factor for correction")
    parser.add_argument("--offdiag_only", action="store_true", help="Apply only off-diagonal correction")
    parser.add_argument("--enforce_hermitian", action="store_true", help="Project corrected H(R) to Hermitian blocks")
    parser.add_argument("--out_prefix", type=str, required=True, help="Output hr prefix")
    args = parser.parse_args()

    cwd = os.getcwd()
    workdir = _resolve_path(cwd, args.workdir) if args.workdir else cwd
    in_dir = _resolve_path(workdir, args.in_dir)
    out_dir = _resolve_path(workdir, args.out_dir) if args.out_dir else in_dir
    os.makedirs(out_dir, exist_ok=True)

    up_path = _resolve_path(in_dir, args.up_hr)
    dn_path = _resolve_path(in_dir, args.dn_hr)
    delta_path = _resolve_path(workdir, args.delta_wannier) if args.delta_wannier else None
    pref_path = _resolve_path(workdir, args.p_ref_wannier) if args.p_ref_wannier else None

    dim_up, degens_up, h_up = _read_hr_compat(up_path)
    dim_dn, degens_dn, h_dn = _read_hr_compat(dn_path)
    if dim_up != dim_dn:
        raise ValueError(f"Up/Dn hr dimensions differ: {dim_up} vs {dim_dn}")
    if set(h_up.keys()) != set(h_dn.keys()):
        raise ValueError("Up/Dn hr R-lists differ. Align hr files before correction.")
    dim = dim_up
    r_list = sorted(h_up.keys())

    p_idx = _parse_index_spec(args.p_idx, dim) if args.p_idx else []
    d_idx = _parse_index_spec(args.d_idx, dim) if args.d_idx else []
    pd_mask = _build_pd_mask(dim, p_idx, d_idx) if (p_idx and d_idx) else None

    if delta_path is not None:
        delta_loaded = _load_wannier_matrix_npz(delta_path, "delta_h_wannier")
        delta_source = "delta_wannier_file"
        delta_up_0 = delta_loaded["up"]
        delta_dn_0 = delta_loaded["down"]
    else:
        if pref_path is None or args.v_value is None:
            raise ValueError(
                "Provide --delta_wannier, or provide both --p_ref_wannier and --v_value."
            )
        p_ref_loaded = _load_wannier_matrix_npz(pref_path, "p_ref_wannier")
        built = _build_delta_from_p_ref_wannier(
            p_ref_loaded, args.v_value, pd_mask=pd_mask, offdiag_only=args.offdiag_only
        )
        delta_source = "built_from_p_ref_wannier"
        delta_up_0 = built["up"]
        delta_dn_0 = built["down"]

    def _shape_check(mat, name):
        if mat.ndim != 2 or mat.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim},{dim}), got {mat.shape}")

    _shape_check(delta_up_0, "delta_up")
    _shape_check(delta_dn_0, "delta_dn")

    if pd_mask is not None and delta_source == "delta_wannier_file":
        delta_up_0 = delta_up_0 * pd_mask
        delta_dn_0 = delta_dn_0 * pd_mask

    if args.offdiag_only and delta_source == "delta_wannier_file":
        delta_up_0 = _enforce_offdiag(delta_up_0)
        delta_dn_0 = _enforce_offdiag(delta_dn_0)

    delta_up_0 = args.lambda_v * delta_up_0
    delta_dn_0 = args.lambda_v * delta_dn_0

    # Minimal one-shot model: add correction on R=(0,0,0) block.
    if (0, 0, 0) not in h_up:
        raise KeyError("R=(0,0,0) block not found in hr.dat. Cannot apply onsite/bond correction.")

    h_up_corr = {r: h.copy() for r, h in h_up.items()}
    h_dn_corr = {r: h.copy() for r, h in h_dn.items()}
    h_up_corr[(0, 0, 0)] = h_up_corr[(0, 0, 0)] + delta_up_0
    h_dn_corr[(0, 0, 0)] = h_dn_corr[(0, 0, 0)] + delta_dn_0

    if args.enforce_hermitian:
        h_up_corr = _enforce_hermitian_dict(h_up_corr)
        h_dn_corr = _enforce_hermitian_dict(h_dn_corr)

    out_up = os.path.join(out_dir, f"{args.out_prefix}_up_hr.dat")
    out_dn = os.path.join(out_dir, f"{args.out_prefix}_dn_hr.dat")
    out_meta = os.path.join(out_dir, f"{args.out_prefix}_v.meta.json")

    write_wannier_hr(out_up, dim, degens_up, h_up_corr, header="SLW intersite-V corrected (up)")
    write_wannier_hr(out_dn, dim, degens_dn, h_dn_corr, header="SLW intersite-V corrected (down)")

    up_ratio = float(np.linalg.norm(delta_up_0) / (np.linalg.norm(h_up[(0, 0, 0)]) + 1e-16))
    dn_ratio = float(np.linalg.norm(delta_dn_0) / (np.linalg.norm(h_dn[(0, 0, 0)]) + 1e-16))

    meta = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "mode": "post-wannier perturbative intersite V (direct Wannier gauge)",
        "frozen_density": True,
        "input": {
            "up_hr": up_path,
            "dn_hr": dn_path,
            "delta_wannier": delta_path,
            "p_ref_wannier": pref_path,
        },
        "apply": {
            "delta_source": delta_source,
            "v_value_eV": None if args.v_value is None else float(args.v_value),
            "p_idx": args.p_idx,
            "d_idx": args.d_idx,
            "lambda_v": float(args.lambda_v),
            "offdiag_only": bool(args.offdiag_only),
            "enforce_hermitian": bool(args.enforce_hermitian),
            "apply_R": [0, 0, 0],
        },
        "output": {
            "up_hr": out_up,
            "dn_hr": out_dn,
        },
        "diagnostics": {
            "nR": int(len(r_list)),
            "dim": int(dim),
            "up_delta_over_base_norm_R0": up_ratio,
            "dn_delta_over_base_norm_R0": dn_ratio,
        },
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved corrected up hr: {out_up}")
    print(f"Saved corrected dn hr: {out_dn}")
    print(f"Saved metadata:        {out_meta}")


if __name__ == "__main__":
    main()
