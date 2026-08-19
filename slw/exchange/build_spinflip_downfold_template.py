"""Build a SOC-downfolding spin-flip template HR.

The template is based on the second-order structure

    M_ud(k) = H_up(k) P L_ud P H_down(k) / denom^2
    M_du(k) = H_down(k) P L_du P H_up(k) / denom^2

where P selects SOC-active orbitals from the .win projections.  This is an
effective model for ligand-SOC-induced spin-flip hopping, not a replacement
for the direct onsite atomic SOC term.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from slw.core.wannier_io import read_wannier_hr, write_wannier_hr
from slw.exchange.plot_epr_soc_bands import _clean_species, _win_projection_groups
from slw.exchange.spinor_model import WANNIER90_D_ORDER, WANNIER90_P_ORDER, atomic_d_soc_block, atomic_p_soc_block


def _read_wannier_hr_compat(path):
    parsed = read_wannier_hr(path)
    if len(parsed) == 4:
        dim, _nrpts, degens, hmap = parsed
        return int(dim), list(degens), hmap
    if len(parsed) == 3:
        dim, degens, hmap = parsed
        return int(dim), list(degens), hmap
    raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")


def _parse_specs(text):
    out = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.replace(",", ":").split(":") if x.strip()]
        if len(parts) != 2:
            raise ValueError(f"Each SOC spec must be element:orbital, got {item!r}")
        elem, orb = _clean_species(parts[0]), parts[1].lower()
        if orb not in {"p", "d"}:
            raise ValueError(f"Unsupported orbital {orb!r}; use p or d")
        out.append((elem, orb))
    if not out:
        raise ValueError("At least one --soc_active element:orbital entry is required")
    return out


def _soc_spinflip_mats(win_path, specs, nwan, *, p_order=WANNIER90_P_ORDER, d_order=WANNIER90_D_ORDER):
    _atoms, groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(f"{win_path} projection count={nproj}, but spinor nwan={nwan}")
    lud = np.zeros((nwan, nwan), dtype=np.complex128)
    ldu = np.zeros((nwan, nwan), dtype=np.complex128)
    matched_summary = []
    for elem, orb in specs:
        matched = [g for g in groups if g["element"] == elem and g["orbital"] == orb]
        if not matched:
            known = sorted({(g["element"], g["orbital"]) for g in groups})
            raise ValueError(f"No {elem}:{orb} projection groups found in {win_path}; known={known}")
        block = atomic_p_soc_block(1.0, order=p_order) if orb == "p" else atomic_d_soc_block(1.0, order=d_order)
        norb = block.shape[0] // 2
        for group in matched:
            idx = np.asarray(group["indices"], dtype=np.int64).reshape(norb)
            lud[idx[:, None], idx[None, :]] += block[:norb, norb:]
            ldu[idx[:, None], idx[None, :]] += block[norb:, :norb]
            matched_summary.append(f"{group['atom_label']}:{orb}{idx.tolist()}")
    return lud, ldu, matched_summary


def _empty_spinor(dim):
    return np.zeros((dim, dim), dtype=np.complex128)


def _build_downfold_hmap(hbase, dim, lud, ldu, denom, *, truncate_to_base_r=False, zero_onsite=False):
    nwan = dim // 2
    keys = [tuple(int(x) for x in r) for r in sorted(hbase)]
    hup = {r: np.asarray(hbase[r], dtype=np.complex128)[:nwan, :nwan] for r in keys}
    hdn = {r: np.asarray(hbase[r], dtype=np.complex128)[nwan:, nwan:] for r in keys}
    scale = 1.0 / (float(denom) ** 2)
    allowed = set(keys) if truncate_to_base_r else None
    out = {}

    # Convolution in real space corresponding to H_up(k) L_ud H_down(k).
    # Matrix products are BLAS-backed; the outer R-pair loop is the practical
    # sparse convolution over available Wannier90 R blocks.
    for r1 in keys:
        h1u_lud = hup[r1] @ lud
        h1d_ldu = hdn[r1] @ ldu
        if not np.any(h1u_lud) and not np.any(h1d_ldu):
            continue
        for r2 in keys:
            rout = (r1[0] + r2[0], r1[1] + r2[1], r1[2] + r2[2])
            if allowed is not None and rout not in allowed:
                continue
            if zero_onsite and rout == (0, 0, 0):
                continue
            block = out.get(rout)
            if block is None:
                block = _empty_spinor(dim)
                out[rout] = block
            block[:nwan, nwan:] += scale * (h1u_lud @ hdn[r2])
            block[nwan:, :nwan] += scale * (h1d_ldu @ hup[r2])

    raw = {r: np.array(h, dtype=np.complex128, copy=True) for r, h in out.items()}
    all_keys = set(raw) | {tuple(-x for x in r) for r in raw}
    herm = {}
    for r in all_keys:
        rneg = tuple(-x for x in r)
        if zero_onsite and r == (0, 0, 0):
            continue
        if allowed is not None and r not in allowed:
            continue
        herm[r] = 0.5 * (raw.get(r, _empty_spinor(dim)) + raw.get(rneg, _empty_spinor(dim)).conj().T)
    return herm


def build_template(args):
    dim, _degens, hbase = _read_wannier_hr_compat(args.base_hr)
    if dim % 2 != 0:
        raise ValueError(f"spin-major spinor dimension must be even, got {dim}")
    nwan = dim // 2
    specs = _parse_specs(args.soc_active)
    lud, ldu, matched = _soc_spinflip_mats(
        args.win,
        specs,
        nwan,
        p_order=args.p_order,
        d_order=args.d_order,
    )
    out = _build_downfold_hmap(
        hbase,
        dim,
        lud,
        ldu,
        args.denom,
        truncate_to_base_r=args.truncate_to_base_R,
        zero_onsite=args.zero_onsite,
    )
    if not out:
        raise ValueError("Downfold template is empty; check --soc_active and base_hr")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    degens = [1] * len(out)
    write_wannier_hr(
        args.output,
        dim,
        degens,
        out,
        header=f"SLW SOC-downfold spin-flip template active={args.soc_active} denom={args.denom:g}",
    )
    print("[spinflip-downfold-template] summary")
    print(f"  base_hr: {args.base_hr}")
    print(f"  win: {args.win}")
    print(f"  output: {args.output}")
    print(f"  dim: {dim} nwan: {nwan}")
    print(f"  soc_active: {args.soc_active}")
    print(f"  matched_groups: {matched}")
    print(f"  denom: {args.denom:g} eV")
    print(f"  nrpts_out: {len(out)}")
    print(f"  truncate_to_base_R: {args.truncate_to_base_R}")
    print(f"  zero_onsite: {args.zero_onsite}")
    return args.output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_hr", required=True, help="No-SOC spin-major spinor hr.dat")
    ap.add_argument("--win", required=True, help="Wannier90 .win file defining projection groups")
    ap.add_argument("--soc_active", required=True, help="SOC-active subspace, e.g. 'X:p' or 'X:p;M:d'")
    ap.add_argument("-o", "--output", required=True, help="Output template hr.dat")
    ap.add_argument("--denom", type=float, default=1.0, help="Energy denominator in eV; coefficient absorbs this if left at 1")
    ap.add_argument("--p_order", default=WANNIER90_P_ORDER)
    ap.add_argument("--d_order", default=WANNIER90_D_ORDER)
    ap.add_argument("--truncate_to_base_R", action="store_true", help="Keep only R vectors already present in base_hr")
    ap.add_argument("--zero_onsite", action="store_true", help="Drop R=(0,0,0) from the downfolded template")
    args = ap.parse_args()
    build_template(args)


if __name__ == "__main__":
    main()
