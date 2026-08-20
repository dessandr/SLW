# Legacy exchange implementation; use the native engine for new workflows.
"""Diagnose orbital-block ratios of spin-flip Hamiltonian weight."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from slw.exchange.legacy.fit_spinflip_kspace import (
    _hk_from_hmap,
    _kmesh_points,
    _parse_kpoints,
    _read_wannier_hr_compat,
    _reconstruct_hk_from_w90,
)
from slw.exchange.legacy.plot_epr_soc_bands import _win_projection_groups


def _spin_index_maps(dim, order, win_path):
    if dim % 2 != 0:
        raise ValueError(f"spinor dimension must be even, got {dim}")
    nwan = dim // 2
    mode = str(order).strip().lower().replace("-", "_")
    up = np.empty(nwan, dtype=np.int64)
    dn = np.empty(nwan, dtype=np.int64)
    if mode in {"spin_major", "spin"}:
        up[:] = np.arange(nwan, dtype=np.int64)
        dn[:] = np.arange(nwan, 2 * nwan, dtype=np.int64)
        return up, dn
    if mode in {"orbital_interleaved", "interleaved", "pair_interleaved"}:
        up[:] = np.arange(0, 2 * nwan, 2, dtype=np.int64)
        dn[:] = np.arange(1, 2 * nwan, 2, dtype=np.int64)
        return up, dn
    if mode in {"win_interleaved", "site_interleaved", "atom_interleaved", "group_interleaved"}:
        if not win_path:
            raise ValueError("--spin_order win_interleaved requires --win")
        _atoms, groups, nproj = _win_projection_groups(win_path)
        if int(nproj) != int(nwan):
            raise ValueError(f"{win_path} projection count={nproj}, but spinor nwan={nwan}")
        offset = 0
        for group in groups:
            idx = [int(x) for x in group["indices"]]
            norb = len(idx)
            up[idx] = np.arange(offset, offset + norb, dtype=np.int64)
            dn[idx] = np.arange(offset + norb, offset + 2 * norb, dtype=np.int64)
            offset += 2 * norb
        if offset != dim:
            raise ValueError(f"win_interleaved index count={offset}, expected dim={dim}")
        return up, dn
    raise ValueError(f"Unknown --spin_order {order!r}")


def _group_key(group, group_by):
    mode = str(group_by).strip().lower()
    if mode == "element_orbital":
        return f"{group['element']}:{group['orbital']}"
    if mode == "atom_orbital":
        return f"{group['atom_label']}:{group['orbital']}"
    if mode == "group":
        return f"{group['atom_label']}:{group['orbital']}[{group['indices'][0]}:{group['indices'][-1] + 1}]"
    raise ValueError(f"Unknown --group_by {group_by!r}")


def _orbital_groups(win_path, nwan, group_by):
    _atoms, groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(f"{win_path} projection count={nproj}, but spinor nwan={nwan}")
    merged = {}
    for group in groups:
        key = _group_key(group, group_by)
        merged.setdefault(key, []).extend(int(x) for x in group["indices"])
    return [(key, np.asarray(sorted(set(idx)), dtype=np.int64)) for key, idx in merged.items()]


def _load_hk(args, kpts):
    if args.hr:
        dim, _degens, hmap = _read_wannier_hr_compat(args.hr)
        hk = _hk_from_hmap(hmap, kpts, dim)
        source = args.hr
    else:
        hk = _reconstruct_hk_from_w90(
            args.eig,
            args.u_mat,
            args.u_dis_mat,
            kpts,
            rotation=args.w90_rotation,
            tol=args.u_match_tol,
            nearest=bool(args.u_nearest),
            label="diag",
        )
        dim = int(hk.shape[1])
        source = args.eig
    if dim % 2 != 0:
        raise ValueError(f"spinor dimension must be even, got {dim}")
    return hk, dim, source


def _block_weights(hk, groups, up_map, dn_map, blocks):
    mode = str(blocks).strip().lower()
    rows = []
    total = 0.0
    for src_label, src_orb in groups:
        src_up = up_map[src_orb]
        src_dn = dn_map[src_orb]
        for dst_label, dst_orb in groups:
            dst_up = up_map[dst_orb]
            dst_dn = dn_map[dst_orb]
            w_ud = 0.0
            w_du = 0.0
            if mode in {"ud", "both"}:
                block = hk[:, src_up[:, None], dst_dn[None, :]]
                w_ud = float(np.vdot(block, block).real)
            if mode in {"du", "both"}:
                block = hk[:, src_dn[:, None], dst_up[None, :]]
                w_du = float(np.vdot(block, block).real)
            weight = w_ud + w_du
            total += weight
            rows.append(
                {
                    "src": src_label,
                    "dst": dst_label,
                    "weight": weight,
                    "norm": float(np.sqrt(max(weight, 0.0))),
                    "weight_ud": w_ud,
                    "weight_du": w_du,
                }
            )
    rows.sort(key=lambda x: x["weight"], reverse=True)
    for row in rows:
        row["ratio"] = row["weight"] / total if total > 0.0 else np.nan
    return rows, total


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "dst", "ratio", "norm", "weight", "weight_ud", "weight_du"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hr", default=None, help="Spinor hr.dat source")
    src.add_argument("--eig", default=None, help="Wannier90 .eig source")
    ap.add_argument("--u_dis_mat", default=None, help="Wannier90 *_u_dis.mat for --eig reconstruction")
    ap.add_argument("--u_mat", default=None, help="Optional Wannier90 *_u.mat; omit to stay before final U rotation")
    ap.add_argument("--w90_rotation", choices=["udag_h_u", "u_h_udag"], default="udag_h_u")
    ap.add_argument("--u_match_tol", type=float, default=1.0e-7)
    ap.add_argument("--u_nearest", action="store_true")
    ap.add_argument("--win", required=True, help="Wannier90 .win defining projection groups")
    ap.add_argument("--kmesh", type=int, nargs=3, required=True)
    ap.add_argument("--kpoints", default=None)
    ap.add_argument(
        "--spin_order",
        choices=["spin_major", "orbital_interleaved", "win_interleaved"],
        default="spin_major",
    )
    ap.add_argument("--group_by", choices=["element_orbital", "atom_orbital", "group"], default="element_orbital")
    ap.add_argument("--blocks", choices=["ud", "du", "both"], default="both")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--csv", default=None, help="Optional CSV output for all block ratios")
    args = ap.parse_args()

    if args.eig and not (args.u_dis_mat or args.u_mat):
        raise ValueError("--eig requires --u_dis_mat and/or --u_mat")

    kpts = _kmesh_points(args.kmesh)
    extra = _parse_kpoints(args.kpoints)
    if extra is not None:
        kpts = np.vstack([kpts, extra])

    hk, dim, source = _load_hk(args, kpts)
    nwan = dim // 2
    groups = _orbital_groups(args.win, nwan, args.group_by)
    up_map, dn_map = _spin_index_maps(dim, args.spin_order, args.win)
    rows, total = _block_weights(hk, groups, up_map, dn_map, args.blocks)

    print("[orbital-spinflip] summary")
    print(f"  source: {source}")
    print(f"  dim: {dim} nwan: {nwan} nk: {len(kpts)}")
    print(f"  spin_order: {args.spin_order}")
    print(f"  group_by: {args.group_by}")
    print(f"  blocks: {args.blocks}")
    print(f"  groups: {[label for label, _idx in groups]}")
    print(f"  total_weight: {total:.12g}")
    print(f"  total_norm: {np.sqrt(max(total, 0.0)):.12g}")
    print("  top orbital spin-flip channels:")
    for row in rows[: max(0, int(args.top))]:
        print(
            f"    {row['src']:>18s} -> {row['dst']:<18s} "
            f"ratio={row['ratio']:.8f} norm={row['norm']:.8g} "
            f"ud={np.sqrt(max(row['weight_ud'], 0.0)):.8g} du={np.sqrt(max(row['weight_du'], 0.0)):.8g}"
        )
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"[orbital-spinflip] wrote {args.csv}")


if __name__ == "__main__":
    main()
