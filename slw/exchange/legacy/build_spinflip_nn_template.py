# Legacy exchange implementation; use the native engine for new workflows.
"""Build an analytic nearest-neighbor spin-flip hopping template HR."""

from __future__ import annotations

import argparse
import os

import numpy as np

from slw.core.wannier_io import read_wannier_hr, write_wannier_hr


def _read_wannier_hr_compat(path):
    parsed = read_wannier_hr(path)
    if len(parsed) == 4:
        dim, _nrpts, degens, hmap = parsed
        return int(dim), list(degens), hmap
    if len(parsed) == 3:
        dim, degens, hmap = parsed
        return int(dim), list(degens), hmap
    raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")


def _parse_r_list(text):
    if not text:
        return None
    out = []
    for item in str(text).replace(";", "\n").splitlines():
        vals = [int(x) for x in item.replace(",", " ").split()]
        if not vals:
            continue
        if len(vals) != 3:
            raise ValueError(f"Each R entry must have 3 integers, got {item!r}")
        out.append(tuple(vals))
    return set(out)


def _r_norm_xy(r):
    rx, ry, _rz = [float(x) for x in r]
    return float(np.hypot(rx, ry))


def _auto_nn_set(hmap):
    candidates = []
    for r in hmap:
        rt = tuple(int(x) for x in r)
        if rt == (0, 0, 0):
            continue
        rn = _r_norm_xy(rt)
        if rn > 1.0e-12:
            candidates.append((rn, rt))
    if not candidates:
        return set()
    rmin = min(x[0] for x in candidates)
    return {r for rn, r in candidates if abs(rn - rmin) < 1.0e-8}


def _all_nonzero_r_set(hmap):
    return {
        tuple(int(x) for x in r)
        for r in hmap
        if tuple(int(x) for x in r) != (0, 0, 0)
    }


def _phase_factor(r, mode):
    rx, ry, rz = [float(x) for x in r]
    mode = str(mode).strip().lower()
    if mode == "px_minus_ipy":
        norm = np.hypot(rx, ry)
        if norm <= 1.0e-14:
            return 0.0j
        return (rx - 1.0j * ry) / norm
    if mode == "px_plus_ipy":
        norm = np.hypot(rx, ry)
        if norm <= 1.0e-14:
            return 0.0j
        return (rx + 1.0j * ry) / norm
    if mode == "i":
        return 1.0j
    if mode == "one":
        return 1.0 + 0.0j
    if mode == "rz_i":
        return 1.0j * np.sign(rz)
    raise ValueError(f"Unknown --phase_model {mode!r}")


def build_template(args):
    dim, degens, hbase = _read_wannier_hr_compat(args.base_hr)
    if dim % 2 != 0:
        raise ValueError(f"spin-major spinor dimension must be even, got {dim}")
    n = dim // 2
    keep_r = _parse_r_list(args.keep_R)
    if keep_r is None:
        if args.all_R:
            keep_r = _all_nonzero_r_set(hbase)
        else:
            keep_r = _auto_nn_set(hbase)
    if not keep_r:
        raise ValueError("No R vectors selected; use --keep_R or --all_R")

    out = {}
    for r, h in hbase.items():
        r = tuple(int(x) for x in r)
        tmpl = np.zeros((dim, dim), dtype=np.complex128)
        if r in keep_r:
            arr = np.asarray(h, dtype=np.complex128)
            hup = arr[:n, :n]
            hdn = arr[n:, n:]
            horb = 0.5 * (hup + hdn)
            if args.remove_diagonal:
                horb = np.array(horb, copy=True)
                np.fill_diagonal(horb, 0.0)
            fac = complex(_phase_factor(r, args.phase_model))
            tud = fac * horb
            tmpl[:n, n:] = tud
            # Enforce H(R)=H^dag(-R) if the reverse block is not separately selected
            rneg = tuple(-x for x in r)
            if rneg not in keep_r:
                tmpl[n:, :n] = tud.conj().T
        out[r] = tmpl

    # Fill missing Hermitian partners from selected R blocks where possible.
    for r in list(keep_r):
        rneg = tuple(-x for x in r)
        if rneg in out and r in out:
            out[rneg][n:, :n] = out[r][:n, n:].conj().T

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    write_wannier_hr(
        args.output,
        dim,
        degens,
        out,
        header=f"SLW analytic NN spin-flip template phase={args.phase_model}",
    )
    print("[spinflip-nn-template] summary")
    print(f"  base_hr: {args.base_hr}")
    print(f"  output: {args.output}")
    print(f"  dim: {dim} nwan: {n}")
    print(f"  selection: {'all nonzero R' if args.all_R else 'explicit/auto shell'}")
    print(f"  selected_R: {sorted(keep_r)}")
    print(f"  phase_model: {args.phase_model}")
    return args.output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_hr", required=True, help="No-SOC spin-major spinor hr.dat")
    ap.add_argument("-o", "--output", required=True, help="Output template hr.dat")
    ap.add_argument(
        "--keep_R",
        default=None,
        help="Explicit R list, e.g. '1 0 0; -1 0 0; 0 1 0; 0 -1 0; 1 -1 0; -1 1 0'",
    )
    ap.add_argument(
        "--all_R",
        action="store_true",
        help="Apply the nonlocal spin-flip template to every nonzero R block in base_hr.",
    )
    ap.add_argument(
        "--phase_model",
        choices=["px_minus_ipy", "px_plus_ipy", "i", "one", "rz_i"],
        default="px_minus_ipy",
        help="Complex direction factor multiplying the spin-conserving hopping matrix.",
    )
    ap.add_argument("--remove_diagonal", action="store_true", help="Zero orbital-diagonal entries in the NN template")
    args = ap.parse_args()
    build_template(args)


if __name__ == "__main__":
    main()
