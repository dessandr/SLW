"""Build a spinor Wannier90 hr.dat from collinear up/down hr.dat plus model SOC."""

from __future__ import annotations

import argparse
import os

import numpy as np

from slw.core.wannier_io import read_wannier_hr, write_wannier_hr
from slw.exchange.plot_epr_soc_bands import _resolve_soc_groups_from_win, _win_projection_groups
from slw.exchange.spinor_model import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    add_atomic_d_soc,
    add_atomic_p_soc,
    normalize_spin_direction,
    spinor_from_collinear,
)


def _read_wannier_hr_compat(path):
    parsed = read_wannier_hr(path)
    if len(parsed) == 4:
        dim, _nrpts, degens, hmap = parsed
        return int(dim), list(degens), hmap
    if len(parsed) == 3:
        dim, degens, hmap = parsed
        return int(dim), list(degens), hmap
    raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")


def _resolve_hr_path(path=None, prefix=None):
    if path:
        return path
    if not prefix:
        raise ValueError("Provide either explicit hr path or prefix")
    return f"{prefix}_hr.dat"


def _apply_soc_to_onsite(h_spin_onsite, args, nwan):
    entries, win_path = _resolve_soc_groups_from_win(args, nwan)
    out = np.asarray(h_spin_onsite, dtype=np.complex128)
    for entry in entries:
        orb = entry["orbital"]
        lam = float(entry["lambda_ev"])
        groups = entry["groups"]
        if orb == "p":
            out = add_atomic_p_soc(out, groups, lambda_ev=lam, order=args.p_order, inplace=False)
        elif orb == "d":
            out = add_atomic_d_soc(out, groups, lambda_ev=lam, order=args.d_order, inplace=False)
        else:
            raise ValueError(f"Unsupported SOC orbital: {orb}")
        print(
            f"[spinor-soc-hr] added {entry['element']}:{orb} SOC "
            f"lambda={lam:g} eV groups={groups} source={entry['source']}",
            flush=True,
        )
    return out, entries, win_path


def _clean_selector_element(text):
    return str(text).strip().split()[0].strip().capitalize()


def _parse_subspace_selector(text):
    specs = []
    for item in str(text or "").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.replace(",", ":").split(":") if x.strip()]
        if len(parts) != 2:
            raise ValueError(f"Expected magnetic subspace selector element:orbital, got {item!r}")
        specs.append((_clean_selector_element(parts[0]), parts[1].lower()))
    return specs


def _suggest_spinor_slices_from_win(win_path, nwan, selector):
    specs = _parse_subspace_selector(selector)
    if not specs:
        return "", []
    if not win_path:
        raise ValueError("--mag_subspace requires --win")
    _atoms, groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(f"{win_path} projection count={nproj} but hr.dat nwan={nwan}")
    matched = []
    for elem, orb in specs:
        found = [g for g in groups if str(g["element"]).capitalize() == elem and str(g["orbital"]).lower() == orb]
        if not found:
            known = sorted({(g["element"], g["orbital"]) for g in groups})
            raise ValueError(f"No {elem}:{orb} groups found in {win_path}; known={known}")
        matched.extend(found)
    out = []
    labels = []
    for isite, group in enumerate(matched):
        idx = np.asarray(group["indices"], dtype=np.int64).reshape(-1)
        up0 = int(idx.min())
        up1 = int(idx.max()) + 1
        if not np.array_equal(idx, np.arange(up0, up1, dtype=np.int64)):
            raise ValueError(
                "--mag_subspace spinor_slices currently require contiguous group indices; "
                f"{group['atom_label']}:{group['orbital']} has {idx.tolist()}"
            )
        out.append(f"{isite}:{up0}:{up1}:{int(nwan) + up0}:{int(nwan) + up1}")
        labels.append(f"{group['atom_label']}:{group['orbital']}[{up0}:{up1}]")
    return ",".join(out), labels


def _basis_permutation_from_win(win_path, nwan, basis_order):
    order = str(basis_order).strip().lower().replace("-", "_")
    if order in {"spin_major", "spin"}:
        return np.arange(2 * int(nwan), dtype=np.int64), "spin_major", []
    if order not in {"win_interleaved", "site_interleaved", "atom_interleaved", "group_interleaved"}:
        raise ValueError(f"Unsupported --basis_order {basis_order!r}; use spin_major or win_interleaved")
    if not win_path:
        raise ValueError("--basis_order win_interleaved requires --win")
    _atoms, groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(f"{win_path} projection count={nproj} but hr.dat nwan={nwan}")
    seen = []
    for group in groups:
        seen.extend(int(x) for x in group["indices"])
    if sorted(seen) != list(range(int(nwan))):
        raise ValueError(f".win projection groups do not cover 0..{int(nwan)-1}: got {sorted(seen)[:12]}...")
    perm = []
    for group in groups:
        idx = [int(x) for x in group["indices"]]
        perm.extend(idx)
        perm.extend([int(nwan) + i for i in idx])
    labels = [
        f"{g['atom_label']}:{g['orbital']}[{g['indices'][0]}:{g['indices'][-1] + 1}]"
        for g in groups
    ]
    return np.asarray(perm, dtype=np.int64), "win_interleaved", labels


def _read_centres_xyz(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid centres file: {path}")
    total = int(lines[0].strip())
    body = lines[2:]
    if len(body) < total:
        raise ValueError(f"centres count mismatch in {path}: header={total}, body={len(body)}")
    return body[:total]


def _split_centres_body(body, nwan):
    if len(body) < int(nwan):
        raise ValueError(f"centres body has {len(body)} rows, expected at least nwan={nwan}")
    return body[: int(nwan)], body[int(nwan) :]


def _centres_default_output(out_hr):
    base = os.path.basename(str(out_hr))
    if base.endswith("_hr.dat"):
        return os.path.join(os.path.dirname(os.path.abspath(out_hr)), base[:-7] + "_centres.xyz")
    stem, _ext = os.path.splitext(os.path.abspath(out_hr))
    return stem + "_centres.xyz"


def _write_spinor_centres(args, nwan, perm, out_hr):
    up_path = args.centres_up or args.centres
    dn_path = args.centres_dn
    if not up_path and not dn_path:
        return None
    if not up_path:
        raise ValueError("Provide --centres or --centres_up for spinor centres output")
    up_wann, up_atoms = _split_centres_body(_read_centres_xyz(up_path), nwan)
    if dn_path:
        dn_wann, dn_atoms = _split_centres_body(_read_centres_xyz(dn_path), nwan)
        if len(up_atoms) != len(dn_atoms):
            raise ValueError(f"atom centre row count mismatch: up={len(up_atoms)}, dn={len(dn_atoms)}")
        atom_lines = up_atoms
    else:
        dn_wann = up_wann
        atom_lines = up_atoms
    spin_major = up_wann + dn_wann
    ordered_wann = [spin_major[int(i)] for i in np.asarray(perm, dtype=np.int64).tolist()]
    out_path = args.centres_output or _centres_default_output(out_hr)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"    {len(ordered_wann) + len(atom_lines)}\n")
        f.write(f" spinor Wannier centres generated by SLW basis_order={args.basis_order}\n")
        for line in ordered_wann:
            f.write(line if line.endswith("\n") else line + "\n")
        for line in atom_lines:
            f.write(line if line.endswith("\n") else line + "\n")
    return out_path


def build_spinor_soc_hr(args):
    up_path = _resolve_hr_path(args.up_hr, args.prefix_up)
    dn_path = _resolve_hr_path(args.dn_hr, args.prefix_dn)
    out_path = args.output
    if out_path is None:
        out_prefix = args.out_prefix or args.prefix or "spinor_soc"
        out_path = f"{out_prefix}_hr.dat"

    dim_up, degens_up, h_up = _read_wannier_hr_compat(up_path)
    dim_dn, degens_dn, h_dn = _read_wannier_hr_compat(dn_path)
    if dim_up != dim_dn:
        raise ValueError(f"hr.dat dimension mismatch: up={dim_up}, dn={dim_dn}")
    if set(h_up) != set(h_dn):
        missing_dn = sorted(set(h_up) - set(h_dn))
        missing_up = sorted(set(h_dn) - set(h_up))
        raise ValueError(f"R-vector mismatch: missing_dn={missing_dn[:5]} missing_up={missing_up[:5]}")
    if len(degens_up) != len(h_up):
        raise ValueError(f"up degeneracy count={len(degens_up)} does not match nrpts={len(h_up)}")
    if len(degens_dn) != len(h_dn):
        raise ValueError(f"dn degeneracy count={len(degens_dn)} does not match nrpts={len(h_dn)}")

    nwan = int(dim_up)
    nvec = normalize_spin_direction(args.spin_direction)
    spinor = {}
    for r in h_up:
        spinor[r] = spinor_from_collinear(h_up[r], h_dn[r], n=nvec)

    entries = []
    win_path = ""
    if args.soc or args.lambda_te != 0.0 or args.soc_p_groups:
        onsite = (0, 0, 0)
        if onsite not in spinor:
            raise KeyError("R=(0,0,0) onsite block not found; cannot add onsite model SOC")
        spinor[onsite], entries, win_path = _apply_soc_to_onsite(spinor[onsite], args, nwan)
    if not win_path and args.win:
        win_path = args.win

    perm, basis_order, basis_groups = _basis_permutation_from_win(win_path, nwan, args.basis_order)
    if basis_order != "spin_major":
        spinor = {r: h[np.ix_(perm, perm)] for r, h in spinor.items()}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    header = (
        "SLW spinor hr from collinear up/down"
        f"; spin_direction={','.join(f'{x:.10g}' for x in nvec)}"
        f"; soc={args.soc or 'legacy'}"
        f"; basis_order={basis_order}"
    )
    write_wannier_hr(out_path, 2 * nwan, degens_up, spinor, header=header)
    centres_path = _write_spinor_centres(args, nwan, perm, out_path)
    print("[spinor-soc-hr] summary", flush=True)
    print(f"  up_hr: {up_path}", flush=True)
    print(f"  dn_hr: {dn_path}", flush=True)
    print(f"  output: {out_path}", flush=True)
    print(f"  nwan: {nwan} -> spinor_dim: {2 * nwan}", flush=True)
    print(f"  nrpts: {len(spinor)}", flush=True)
    print(f"  spin_direction: {nvec.tolist()}", flush=True)
    print(f"  basis_order: {basis_order}", flush=True)
    if basis_groups:
        print(f"  basis_groups: {basis_groups}", flush=True)
    print(f"  win_path: {win_path or ''}", flush=True)
    if centres_path:
        print(f"  centres_output: {centres_path}", flush=True)
    soc_summary = [
        f"{e['element']}:{e['orbital']}:{float(e['lambda_ev']):.8g}"
        for e in entries
    ]
    print(f"  soc_entries: {soc_summary}", flush=True)
    if args.mag_subspace:
        slices, labels = _suggest_spinor_slices_from_win(win_path or args.win, nwan, args.mag_subspace)
        print(f"  mag_subspace: {labels}", flush=True)
        print(f"  suggested_spinor_slices: {slices}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default=None, help="Base prefix; defaults up/dn/output to <prefix>_up/_dn/_hr.dat patterns")
    ap.add_argument("--prefix_up", default=None, help="Spin-up prefix; reads <prefix_up>_hr.dat")
    ap.add_argument("--prefix_dn", default=None, help="Spin-down prefix; reads <prefix_dn>_hr.dat")
    ap.add_argument("--up_hr", default=None, help="Explicit spin-up hr.dat path")
    ap.add_argument("--dn_hr", default=None, help="Explicit spin-down hr.dat path")
    ap.add_argument("-o", "--output", default=None, help="Output spinor hr.dat path")
    ap.add_argument("--out_prefix", default=None, help="Output prefix; writes <out_prefix>_hr.dat if --output is omitted")
    ap.add_argument(
        "--basis_order",
        choices=["spin_major", "win_interleaved", "site_interleaved", "atom_interleaved", "group_interleaved"],
        default="spin_major",
        help="Output spinor basis. Default spin_major writes all up orbitals then all down orbitals.",
    )
    ap.add_argument("--centres", default=None, help="Single collinear centres.xyz file; duplicated if spin-down centres is omitted")
    ap.add_argument("--centres_up", default=None, help="Spin-up centres.xyz file")
    ap.add_argument("--centres_dn", default=None, help="Spin-down centres.xyz file")
    ap.add_argument("--centres_output", default=None, help="Output spinor centres.xyz path")
    ap.add_argument("--spin_direction", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    ap.add_argument("--soc", default="", help="Model SOC specs inferred from .win projections, e.g. 'I:p:0.6;Cr:d:0.05'")
    ap.add_argument("--win", default=None, help="Wannier90 .win file used to infer p/d SOC orbital groups")
    ap.add_argument("--mag_subspace", default="", help="Magnetic subspace selector for suggested spinor_slices, e.g. 'Cr:d'")
    ap.add_argument("--soc_element", default="", help="Element for compatibility --lambda_te p-SOC mode")
    ap.add_argument("--lambda_te", type=float, default=0.0, help="Compatibility onsite p SOC lambda in eV; prefer --soc")
    ap.add_argument("--soc_p_groups", default="", help="Manual p groups, e.g. '10,11,12;25,26,27'")
    ap.add_argument("--soc_p_groups_base", type=int, choices=[0, 1], default=0)
    ap.add_argument("--p_order", default=WANNIER90_P_ORDER)
    ap.add_argument("--d_order", default=WANNIER90_D_ORDER)
    args = ap.parse_args()

    if args.prefix:
        args.prefix_up = args.prefix_up or f"{args.prefix}_up"
        args.prefix_dn = args.prefix_dn or f"{args.prefix}_dn"
        args.out_prefix = args.out_prefix or args.prefix
    args.epr_up = ""
    build_spinor_soc_hr(args)


if __name__ == "__main__":
    main()
