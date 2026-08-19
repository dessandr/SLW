"""Diagnose spin-flip blocks in a spin-major spinor Wannier90 hr.dat."""

from __future__ import annotations

import argparse
import os

import numpy as np

from slw.core.wannier_io import read_wannier_hr


def _read_wannier_hr_compat(path):
    parsed = read_wannier_hr(path)
    if len(parsed) == 4:
        dim, _nrpts, _degens, hmap = parsed
        return int(dim), hmap
    if len(parsed) == 3:
        dim, _degens, hmap = parsed
        return int(dim), hmap
    raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")


def _fro_norm(x):
    arr = np.asarray(x, dtype=np.complex128)
    return float(np.sqrt(np.vdot(arr, arr).real))


def spinflip_stats(path, *, top=20, tol=0.0):
    dim, hmap = _read_wannier_hr_compat(path)
    if dim % 2 != 0:
        raise ValueError(f"spin-major spinor hr.dat dimension must be even, got {dim}")
    n = dim // 2
    rows = []
    total_ud2 = 0.0
    total_du2 = 0.0
    onsite_ud = 0.0
    onsite_du = 0.0
    off_ud2 = 0.0
    off_du2 = 0.0
    for r in sorted(hmap):
        h = np.asarray(hmap[r], dtype=np.complex128)
        if h.shape != (dim, dim):
            raise ValueError(f"{path}: R={r} block shape={h.shape}, expected={(dim, dim)}")
        ud = _fro_norm(h[:n, n:])
        du = _fro_norm(h[n:, :n])
        rows.append((tuple(int(x) for x in r), ud, du, max(ud, du)))
        total_ud2 += ud * ud
        total_du2 += du * du
        if tuple(r) == (0, 0, 0):
            onsite_ud = ud
            onsite_du = du
        else:
            off_ud2 += ud * ud
            off_du2 += du * du
    rows = [x for x in rows if x[3] >= float(tol)]
    rows.sort(key=lambda x: x[3], reverse=True)
    return {
        "path": path,
        "dim": dim,
        "nwan": n,
        "nrpts": len(hmap),
        "onsite_ud": onsite_ud,
        "onsite_du": onsite_du,
        "offsite_ud": float(np.sqrt(off_ud2)),
        "offsite_du": float(np.sqrt(off_du2)),
        "total_ud": float(np.sqrt(total_ud2)),
        "total_du": float(np.sqrt(total_du2)),
        "rows": rows[: int(top)],
    }


def _kmesh_points(mesh):
    n1, n2, n3 = [int(x) for x in mesh]
    return np.asarray(
        [(i / n1, j / n2, k / n3) for i in range(n1) for j in range(n2) for k in range(n3)],
        dtype=np.float64,
    )


def _parse_kpoints(text):
    if not text:
        return None
    rows = []
    for item in str(text).replace(";", "\n").splitlines():
        vals = [float(x) for x in item.replace(",", " ").split()]
        if not vals:
            continue
        if len(vals) != 3:
            raise ValueError(f"Each --kpoints entry must have 3 numbers, got {item!r}")
        rows.append(vals)
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def _hk_from_hmap(hmap, kpts):
    rvec = np.asarray(list(hmap.keys()), dtype=np.float64)
    h_r = np.asarray([hmap[tuple(int(x) for x in r)] for r in rvec], dtype=np.complex128)
    phase = np.exp(2j * np.pi * (np.asarray(kpts, dtype=np.float64) @ rvec.T))
    hk = np.einsum("kr,rij->kij", phase, h_r, optimize=True)
    return 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))


def kspace_stats(path, kpts):
    dim, hmap = _read_wannier_hr_compat(path)
    if dim % 2 != 0:
        raise ValueError(f"spin-major spinor hr.dat dimension must be even, got {dim}")
    n = dim // 2
    hk = _hk_from_hmap(hmap, kpts)
    ud = np.linalg.norm(hk[:, :n, n:].reshape(len(kpts), -1), axis=1)
    du = np.linalg.norm(hk[:, n:, :n].reshape(len(kpts), -1), axis=1)
    evals = np.linalg.eigvalsh(hk)
    sz_diag = np.r_[np.ones(n), -np.ones(n)]
    spin_split = np.einsum("i,kii->k", sz_diag, hk.real, optimize=True) / float(n)
    return {
        "path": path,
        "dim": dim,
        "nwan": n,
        "nk": len(kpts),
        "kpts": np.asarray(kpts, dtype=np.float64),
        "ud": ud,
        "du": du,
        "evals": evals,
        "spin_split_trace": spin_split,
    }


def _summary_arr(x):
    arr = np.asarray(x, dtype=np.float64)
    return {
        "min": float(np.nanmin(arr)),
        "mean": float(np.nanmean(arr)),
        "rms": float(np.sqrt(np.nanmean(arr * arr))),
        "max": float(np.nanmax(arr)),
    }


def print_kspace_stats(stats, *, label=None, top=10):
    name = label or os.path.basename(stats["path"])
    print(f"[spinflip-k] {name}")
    print(f"  path: {stats['path']}")
    print(f"  dim: {stats['dim']}  nwan: {stats['nwan']}  nk: {stats['nk']}")
    for key in ("ud", "du"):
        s = _summary_arr(stats[key])
        print(
            f"  ||H_{key}(k)||_F min/mean/rms/max: "
            f"{s['min']:.10g} {s['mean']:.10g} {s['rms']:.10g} {s['max']:.10g}"
        )
    ss = _summary_arr(stats["spin_split_trace"])
    print(
        "  trace spin splitting proxy min/mean/rms/max: "
        f"{ss['min']:.10g} {ss['mean']:.10g} {ss['rms']:.10g} {ss['max']:.10g}"
    )
    order = np.argsort(np.maximum(stats["ud"], stats["du"]))[::-1][: int(top)]
    print("  top k by spin-flip norm:")
    for ik in order:
        k = stats["kpts"][int(ik)]
        print(
            f"    k=({k[0]:.6g},{k[1]:.6g},{k[2]:.6g}) "
            f"ud={stats['ud'][ik]:.10g} du={stats['du'][ik]:.10g}"
        )


def print_kspace_compare(a, b, *, label_a="A", label_b="B"):
    if a["evals"].shape != b["evals"].shape:
        raise ValueError(f"eigenvalue shape mismatch: {a['evals'].shape} vs {b['evals'].shape}")
    de = b["evals"] - a["evals"]
    print("[spinflip-k] compare summary")
    for key in ("ud", "du"):
        ratio = np.divide(b[key], a[key], out=np.full_like(b[key], np.nan), where=np.abs(a[key]) > 1.0e-30)
        s = _summary_arr(ratio[np.isfinite(ratio)])
        print(
            f"  {key} ratio {label_b}/{label_a} min/mean/rms/max: "
            f"{s['min']:.10g} {s['mean']:.10g} {s['rms']:.10g} {s['max']:.10g}"
        )
    print(f"  eigenvalue diff rms: {float(np.sqrt(np.mean(de * de))):.10g}")
    print(f"  eigenvalue diff max_abs: {float(np.max(np.abs(de))):.10g}")
    split_diff = b["spin_split_trace"] - a["spin_split_trace"]
    print(f"  spin-split trace proxy diff rms: {float(np.sqrt(np.mean(split_diff * split_diff))):.10g}")


def _ratio(a, b):
    return np.nan if abs(float(b)) < 1.0e-30 else float(a) / float(b)


def print_stats(stats, *, label=None):
    name = label or os.path.basename(stats["path"])
    print(f"[spinflip-hr] {name}")
    print(f"  path: {stats['path']}")
    print(f"  dim: {stats['dim']}  nwan: {stats['nwan']}  nrpts: {stats['nrpts']}")
    print(f"  onsite ||H_ud(R=0)||_F: {stats['onsite_ud']:.10g}")
    print(f"  onsite ||H_du(R=0)||_F: {stats['onsite_du']:.10g}")
    print(f"  offsite sqrt(sum_R!=0 ||H_ud(R)||_F^2): {stats['offsite_ud']:.10g}")
    print(f"  offsite sqrt(sum_R!=0 ||H_du(R)||_F^2): {stats['offsite_du']:.10g}")
    print(f"  total   sqrt(sum_R    ||H_ud(R)||_F^2): {stats['total_ud']:.10g}")
    print(f"  total   sqrt(sum_R    ||H_du(R)||_F^2): {stats['total_du']:.10g}")
    print("  top R by spin-flip norm:")
    for r, ud, du, mx in stats["rows"]:
        print(f"    R={r!s:>14s}  ud={ud:.10g}  du={du:.10g}  max={mx:.10g}")


def print_compare(a, b, *, label_a="A", label_b="B"):
    print("[spinflip-hr] ratio summary")
    for key in ("onsite_ud", "offsite_ud", "total_ud", "onsite_du", "offsite_du", "total_du"):
        print(f"  {key}: {label_b}/{label_a} = {_ratio(b[key], a[key]):.10g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hr", nargs="+", help="Spin-major spinor Wannier90 hr.dat file(s)")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels for input files")
    ap.add_argument("--top", type=int, default=20, help="Number of largest R blocks to print")
    ap.add_argument("--tol", type=float, default=0.0, help="Omit R blocks with max spin-flip norm below this value")
    ap.add_argument("--kmesh", type=int, nargs=3, default=None, help="Also diagnose H_ud(k) on a regular fractional k mesh")
    ap.add_argument("--kpoints", default=None, help="Also diagnose explicit fractional k points, e.g. '0 0 0; 0.5 0 0'")
    ap.add_argument("--no_rspace", action="store_true", help="Skip R-space diagnostics")
    args = ap.parse_args()

    labels = args.labels or []
    stats = []
    if not args.no_rspace:
        for i, path in enumerate(args.hr):
            label = labels[i] if i < len(labels) else None
            st = spinflip_stats(path, top=args.top, tol=args.tol)
            stats.append(st)
            print_stats(st, label=label)
        if len(stats) == 2:
            la = labels[0] if len(labels) > 0 else os.path.basename(args.hr[0])
            lb = labels[1] if len(labels) > 1 else os.path.basename(args.hr[1])
            print_compare(stats[0], stats[1], label_a=la, label_b=lb)

    kpts = _parse_kpoints(args.kpoints)
    if args.kmesh is not None:
        mesh_kpts = _kmesh_points(args.kmesh)
        kpts = mesh_kpts if kpts is None else np.vstack([kpts, mesh_kpts])
    if kpts is not None:
        kstats = []
        for i, path in enumerate(args.hr):
            label = labels[i] if i < len(labels) else None
            st = kspace_stats(path, kpts)
            kstats.append(st)
            print_kspace_stats(st, label=label, top=args.top)
        if len(kstats) == 2:
            la = labels[0] if len(labels) > 0 else os.path.basename(args.hr[0])
            lb = labels[1] if len(labels) > 1 else os.path.basename(args.hr[1])
            print_kspace_compare(kstats[0], kstats[1], label_a=la, label_b=lb)


if __name__ == "__main__":
    main()
