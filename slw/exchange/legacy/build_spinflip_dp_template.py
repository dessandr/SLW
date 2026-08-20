# Legacy exchange implementation; use the native engine for new workflows.
"""Build an effective Cr-d/I-p spin-flip template with p-orbital selection rules."""

from __future__ import annotations

import argparse
import os

import numpy as np

from slw.core.wannier_io import read_wannier_hr, write_wannier_hr
from slw.exchange.legacy.plot_epr_soc_bands import _clean_species, _win_projection_groups
from slw.exchange.legacy.spinor_model import WANNIER90_D_ORDER, WANNIER90_P_ORDER, atomic_p_soc_block


def _read_wannier_hr_compat(path):
    parsed = read_wannier_hr(path)
    if len(parsed) == 4:
        dim, _nrpts, degens, hmap = parsed
        return int(dim), list(degens), hmap
    if len(parsed) == 3:
        dim, degens, hmap = parsed
        return int(dim), list(degens), hmap
    raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")


def _groups_by_element_orbital(win_path, nwan, element, orbital):
    _atoms, groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(f"{win_path} projection count={nproj}, but spinor nwan={nwan}")
    elem = _clean_species(element)
    orb = str(orbital).strip().lower()
    matched = [g for g in groups if g["element"] == elem and g["orbital"] == orb]
    if not matched:
        known = sorted({(g["element"], g["orbital"]) for g in groups})
        raise ValueError(f"No {elem}:{orb} groups found in {win_path}; known={known}")
    return [(g["atom_label"], np.asarray(g["indices"], dtype=np.int64)) for g in matched]


def _p_spinflip_blocks(order):
    block = atomic_p_soc_block(1.0, order=order)
    norb = block.shape[0] // 2
    return block[:norb, norb:], block[norb:, :norb]


def _constant_like(mat, tol):
    arr = np.asarray(mat, dtype=np.complex128)
    mask = np.abs(arr) > float(tol)
    if not np.any(mask):
        return np.zeros_like(arr)
    phase = np.ones_like(arr, dtype=np.complex128)
    nonzero = mask & (np.abs(arr) > 0.0)
    phase[nonzero] = arr[nonzero] / np.abs(arr[nonzero])
    return np.where(mask, phase, 0.0)


def _p_complex_transform(order):
    order_list = [x.strip().lower() for x in str(order).replace(";", ",").split(",") if x.strip()]
    if sorted(order_list) != ["px", "py", "pz"]:
        raise ValueError(f"Unsupported p order for lm selection: {order!r}")
    real = ["px", "py", "pz"]
    cols = []
    mvals = []
    vec = {"px": np.array([1.0, 0.0, 0.0]), "py": np.array([0.0, 1.0, 0.0]), "pz": np.array([0.0, 0.0, 1.0])}
    complex_states = [
        (0, vec["pz"]),
        (1, -(vec["px"] + 1.0j * vec["py"]) / np.sqrt(2.0)),
        (-1, (vec["px"] - 1.0j * vec["py"]) / np.sqrt(2.0)),
    ]
    perm = [real.index(x) for x in order_list]
    for m, v in complex_states:
        cols.append(v[perm])
        mvals.append(m)
    return np.column_stack(cols).astype(np.complex128), np.asarray(mvals, dtype=np.int64)


def _d_complex_transform(order):
    aliases = {
        "dz^2": "dz2",
        "d_z2": "dz2",
        "d_z^2": "dz2",
        "dx2y2": "dx2-y2",
        "dx^2-y^2": "dx2-y2",
        "d_x2-y2": "dx2-y2",
        "d_x^2-y^2": "dx2-y2",
    }
    order_list = [x.strip().lower() for x in str(order).replace(";", ",").split(",") if x.strip()]
    order_list = [aliases.get(x, x) for x in order_list]
    base = ["dz2", "dxz", "dyz", "dx2-y2", "dxy"]
    if sorted(order_list) != sorted(base):
        raise ValueError(f"Unsupported d order for lm selection: {order!r}")
    vec = {name: np.eye(5, dtype=np.complex128)[:, i] for i, name in enumerate(base)}
    complex_states = [
        (0, vec["dz2"]),
        (1, -(vec["dxz"] + 1.0j * vec["dyz"]) / np.sqrt(2.0)),
        (-1, (vec["dxz"] - 1.0j * vec["dyz"]) / np.sqrt(2.0)),
        (2, (vec["dx2-y2"] + 1.0j * vec["dxy"]) / np.sqrt(2.0)),
        (-2, (vec["dx2-y2"] - 1.0j * vec["dxy"]) / np.sqrt(2.0)),
    ]
    perm = [base.index(x) for x in order_list]
    cols = []
    mvals = []
    for m, v in complex_states:
        cols.append(v[perm])
        mvals.append(m)
    return np.column_stack(cols).astype(np.complex128), np.asarray(mvals, dtype=np.int64)


def _lm_filter_dp(hdp, hpd, args):
    mode = str(args.selection_model).strip().lower()
    if mode == "all":
        return hdp, hpd
    if mode != "lm_global":
        raise ValueError(f"Unknown --selection_model {args.selection_model!r}")
    cd, md = _d_complex_transform(args.d_order)
    cp, mp = _p_complex_transform(args.p_order)
    mask_dp = (md[:, None] == mp[None, :]).astype(np.complex128)
    hdp_c = cd.conj().T @ hdp @ cp
    hpd_c = cp.conj().T @ hpd @ cd
    hdp_f = cd @ (hdp_c * mask_dp) @ cp.conj().T
    hpd_f = cp @ (hpd_c * mask_dp.T) @ cd.conj().T
    return hdp_f, hpd_f


def _template_block(hup, hdn, d_idx, p_idx, lud_p, ldu_p, args):
    hdp = 0.5 * (hup[np.ix_(d_idx, p_idx)] + hdn[np.ix_(d_idx, p_idx)])
    hpd = 0.5 * (hup[np.ix_(p_idx, d_idx)] + hdn[np.ix_(p_idx, d_idx)])
    if args.amplitude_model == "constant":
        hdp = _constant_like(hdp, args.hopping_tol)
        hpd = _constant_like(hpd, args.hopping_tol)
    elif args.amplitude_model != "hopping":
        raise ValueError(f"Unknown --amplitude_model {args.amplitude_model!r}")
    hdp, hpd = _lm_filter_dp(hdp, hpd, args)
    if np.linalg.norm(hdp) <= float(args.hopping_tol) and np.linalg.norm(hpd) <= float(args.hopping_tol):
        return None
    ud_dp = hdp @ lud_p
    ud_pd = lud_p @ hpd
    du_dp = hdp @ ldu_p
    du_pd = ldu_p @ hpd
    bond_norm = float(np.sqrt(np.linalg.norm(hdp) ** 2 + np.linalg.norm(hpd) ** 2))
    return ud_dp, ud_pd, du_dp, du_pd, bond_norm


def _selected_candidates(candidates, args):
    if not candidates:
        return []
    norms = np.asarray([c["bond_norm"] for c in candidates], dtype=np.float64)
    max_norm = float(np.max(norms)) if norms.size else 0.0
    keep = norms >= float(args.block_norm_cutoff)
    if float(args.relative_cutoff) > 0.0:
        keep &= norms >= float(args.relative_cutoff) * max_norm
    selected = [c for c, ok in zip(candidates, keep) if bool(ok)]
    if int(args.top_blocks) > 0 and len(selected) > int(args.top_blocks):
        selected = sorted(selected, key=lambda x: x["bond_norm"], reverse=True)[: int(args.top_blocks)]
    return selected


def build_template(args):
    dim, degens, hbase = _read_wannier_hr_compat(args.base_hr)
    if dim % 2 != 0:
        raise ValueError(f"spin-major spinor dimension must be even, got {dim}")
    nwan = dim // 2
    d_groups = _groups_by_element_orbital(args.win, nwan, args.d_element, "d")
    p_groups = _groups_by_element_orbital(args.win, nwan, args.p_element, "p")
    lud_p, ldu_p = _p_spinflip_blocks(args.p_order)
    if lud_p.shape != (3, 3):
        raise ValueError(f"p spin-flip block must be 3x3, got {lud_p.shape}")

    candidates = []
    for r, h in hbase.items():
        r = tuple(int(x) for x in r)
        arr = np.asarray(h, dtype=np.complex128)
        hup = arr[:nwan, :nwan]
        hdn = arr[nwan:, nwan:]
        for d_label, d_idx in d_groups:
            for p_label, p_idx in p_groups:
                blocks = _template_block(hup, hdn, d_idx, p_idx, lud_p, ldu_p, args)
                if blocks is None:
                    continue
                ud_dp, ud_pd, du_dp, du_pd, bond_norm = blocks
                if args.zero_onsite and r == (0, 0, 0):
                    continue
                candidates.append(
                    {
                        "r": r,
                        "d_label": d_label,
                        "p_label": p_label,
                        "d_idx": d_idx,
                        "p_idx": p_idx,
                        "ud_dp": ud_dp,
                        "ud_pd": ud_pd,
                        "du_dp": du_dp,
                        "du_pd": du_pd,
                        "bond_norm": bond_norm,
                    }
                )

    selected_candidates = _selected_candidates(candidates, args)
    out = {tuple(int(x) for x in r): np.zeros((dim, dim), dtype=np.complex128) for r in hbase}
    for item in selected_candidates:
        r = item["r"]
        d_idx = item["d_idx"]
        p_idx = item["p_idx"]
        tmpl = out[r]
        tmpl[np.ix_(d_idx, p_idx + nwan)] += item["ud_dp"]
        tmpl[np.ix_(p_idx, d_idx + nwan)] += item["ud_pd"]
        tmpl[np.ix_(d_idx + nwan, p_idx)] += item["du_dp"]
        tmpl[np.ix_(p_idx + nwan, d_idx)] += item["du_pd"]

    selected = len(selected_candidates)
    if selected == 0:
        raise ValueError("No d-p hopping blocks selected; check --win, --hopping_tol, and base_hr")

    raw = {r: np.array(h, dtype=np.complex128, copy=True) for r, h in out.items()}
    for r in list(raw):
        rneg = tuple(-x for x in r)
        if rneg in raw:
            out[r] = 0.5 * (raw[r] + raw[rneg].conj().T)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    write_wannier_hr(
        args.output,
        dim,
        degens,
        out,
        header=(
            "SLW alpha_dp spin-flip template "
            f"d={args.d_element}:d p={args.p_element}:p p_order={args.p_order} "
            f"mode={args.amplitude_model} selection={args.selection_model}"
        ),
    )
    print("[spinflip-dp-template] summary")
    print(f"  base_hr: {args.base_hr}")
    print(f"  win: {args.win}")
    print(f"  output: {args.output}")
    print(f"  dim: {dim} nwan: {nwan}")
    print(f"  d_groups: {[f'{label}:{idx.tolist()}' for label, idx in d_groups]}")
    print(f"  p_groups: {[f'{label}:{idx.tolist()}' for label, idx in p_groups]}")
    print(f"  p_order: {args.p_order}")
    print(f"  d_order: {args.d_order}")
    print(f"  amplitude_model: {args.amplitude_model}")
    print(f"  selection_model: {args.selection_model}")
    if candidates:
        norms = np.asarray([c["bond_norm"] for c in candidates], dtype=np.float64)
        print(f"  candidate_dp_blocks: {len(candidates)}")
        print(f"  bond_norm min/mean/max: {float(np.min(norms)):.8g} {float(np.mean(norms)):.8g} {float(np.max(norms)):.8g}")
    print(f"  block_norm_cutoff: {args.block_norm_cutoff:g}")
    print(f"  relative_cutoff: {args.relative_cutoff:g}")
    print(f"  top_blocks: {int(args.top_blocks)}")
    print(f"  selected_dp_blocks: {selected}")
    print("  top_selected_blocks:")
    for item in sorted(selected_candidates, key=lambda x: x["bond_norm"], reverse=True)[: min(12, selected)]:
        print(f"    R={item['r']} {item['d_label']}->{item['p_label']} norm={item['bond_norm']:.8g}")
    print(f"  zero_onsite: {args.zero_onsite}")
    return args.output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_hr", required=True, help="No-SOC spin-major spinor hr.dat")
    ap.add_argument("--win", required=True, help="Wannier90 .win file defining projection groups")
    ap.add_argument("-o", "--output", required=True, help="Output template hr.dat")
    ap.add_argument("--d_element", required=True, help="Magnetic d element")
    ap.add_argument("--p_element", required=True, help="Ligand p element")
    ap.add_argument("--p_order", default=WANNIER90_P_ORDER)
    ap.add_argument("--d_order", default=WANNIER90_D_ORDER)
    ap.add_argument(
        "--selection_model",
        choices=["all", "lm_global"],
        default="all",
        help="all keeps every d-p orbital channel; lm_global keeps only same-m channels in global spherical basis before applying p L±.",
    )
    ap.add_argument(
        "--amplitude_model",
        choices=["hopping", "constant"],
        default="hopping",
        help="Use existing d-p hopping amplitudes, or only their nonzero topology/phases.",
    )
    ap.add_argument("--hopping_tol", type=float, default=1.0e-12, help="Tolerance for selecting nonzero d-p blocks")
    ap.add_argument("--block_norm_cutoff", type=float, default=0.0, help="Keep d-p group blocks with combined hopping norm above this value")
    ap.add_argument("--relative_cutoff", type=float, default=0.0, help="Keep d-p group blocks with norm >= this fraction of the maximum")
    ap.add_argument("--top_blocks", type=int, default=0, help="Keep only this many strongest d-p group/R blocks after cutoffs")
    ap.add_argument("--zero_onsite", action="store_true", help="Drop R=(0,0,0) d-p spin-flip block")
    args = ap.parse_args()
    build_template(args)


if __name__ == "__main__":
    main()
