"""Build a spinor Wannier90 hr.dat from collinear up/down hr.dat.

This module is kept as the legacy ``slw.soc.wannier_soc`` entry point, but
the SOC layout is now inferred from a Wannier90 ``.win`` projections block
instead of assuming a material-specific orbital order.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from slw.core.cli_paths import resolve_out_path, resolve_path, resolve_workdir
from slw.core.wannier_io import read_wannier_hr, write_wannier_hr
from slw.exchange.build_spinor_soc_hr import _apply_soc_to_onsite, _basis_permutation_from_win
from slw.exchange.spinor_model import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
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


def _read_centres_xyz(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid centres file: {path}")
    total = int(lines[0].strip())
    body = lines[2:]
    if len(body) < total:
        raise ValueError(f"centres count mismatch in {path}: header={total}, rows={len(body)}")
    return body[:total]


def _split_centres_body(body, nwan):
    if len(body) < int(nwan):
        raise ValueError(f"centres body has {len(body)} rows, expected at least {nwan}")
    return body[: int(nwan)], body[int(nwan) :]


def _default_centres_out(output_path):
    if str(output_path).endswith("_hr.dat"):
        return str(output_path)[:-7] + "_centres.xyz"
    stem, _ext = os.path.splitext(str(output_path))
    return stem + "_centres.xyz"


def _write_spinor_centres(out_path, centres_up, centres_down, nwan, perm, basis_order):
    up_wann, up_atoms = _split_centres_body(_read_centres_xyz(centres_up), nwan)
    if centres_down:
        down_wann, down_atoms = _split_centres_body(_read_centres_xyz(centres_down), nwan)
        if len(up_atoms) != len(down_atoms):
            raise ValueError(
                f"Atom centre row count mismatch: up={len(up_atoms)}, down={len(down_atoms)}"
            )
    else:
        down_wann = up_wann
        down_atoms = up_atoms

    spin_major = up_wann + down_wann
    ordered_wann = [spin_major[int(i)] for i in np.asarray(perm, dtype=np.int64)]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"    {len(ordered_wann) + len(up_atoms)}\n")
        f.write(f" SLW spinor Wannier centres basis_order={basis_order}\n")
        for line in ordered_wann:
            f.write(line if line.endswith("\n") else line + "\n")
        for line in up_atoms:
            f.write(line if line.endswith("\n") else line + "\n")


def _separated_to_wannier_orbital_permutation(nwan):
    perm = np.empty(2 * int(nwan), dtype=np.int64)
    perm[0::2] = np.arange(int(nwan), dtype=np.int64)
    perm[1::2] = np.arange(int(nwan), dtype=np.int64) + int(nwan)
    return perm


def _basis_permutation(win_path, nwan, basis_order):
    order = str(basis_order).strip().lower().replace("-", "_")
    if order in {"wannier_orbital", "orbital_interleaved", "spinor_block"}:
        return _separated_to_wannier_orbital_permutation(nwan), "wannier_orbital", []
    if order in {"spin_major", "spin"}:
        return np.arange(2 * int(nwan), dtype=np.int64), "spin_major", []
    return _basis_permutation_from_win(win_path, nwan, order)


def _theta_phi_direction(theta_deg, phi_deg):
    theta = np.deg2rad(float(theta_deg))
    phi = np.deg2rad(float(phi_deg))
    return normalize_spin_direction(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def _normalize_soc_args(args):
    raw = str(args.soc or "").strip()
    low = raw.lower()
    if low in {"off", "false", "none", "0", "no"}:
        args.soc = ""
        args.lambda_te = 0.0
        args.soc_p_groups = ""
        return False
    if low in {"on", "true", "1", "yes"}:
        args.soc = ""
        return True
    return bool(raw or abs(float(args.lambda_te)) > 0.0 or args.soc_p_groups)


def _infer_centres(path):
    candidate = str(path).replace("_hr.dat", "_centres.xyz")
    return candidate if os.path.exists(candidate) else None


def build_wannier_soc(args):
    workdir = resolve_workdir(args.workdir)
    in_dir = resolve_path(workdir, args.in_dir) if args.in_dir else workdir
    up_path = resolve_path(in_dir, args.up)
    dn_path = resolve_path(in_dir, args.dn)
    output_path = resolve_out_path(
        workdir=workdir,
        out_dir=args.out_dir,
        out_name=args.out_name,
        default_dir=".",
        default_name="spinor_soc_wan_hr.dat",
    )
    if args.win:
        args.win = resolve_path(workdir, args.win)

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
    nvec = normalize_spin_direction(args.spin_direction) if args.spin_direction else _theta_phi_direction(args.theta, args.phi)
    soc_requested = _normalize_soc_args(args)
    args.epr_up = ""

    spinor = {r: spinor_from_collinear(h_up[r], h_dn[r], n=nvec) for r in h_up}
    entries = []
    win_path = args.win or ""
    if soc_requested:
        onsite = (0, 0, 0)
        if onsite not in spinor:
            raise KeyError("R=(0,0,0) onsite block not found; cannot add onsite model SOC")
        spinor[onsite], entries, resolved_win = _apply_soc_to_onsite(spinor[onsite], args, nwan)
        win_path = resolved_win or win_path

    perm, basis_order, basis_groups = _basis_permutation(win_path, nwan, args.basis_order)
    if basis_order != "spin_major":
        spinor = {r: h[np.ix_(perm, perm)] for r, h in spinor.items()}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    soc_label = ";".join(f"{e['element']}:{e['orbital']}:{float(e['lambda_ev']):.10g}" for e in entries)
    header = (
        "SLW spinor Wannier Hamiltonian"
        f"; spin_direction={','.join(f'{x:.10g}' for x in nvec)}"
        f"; soc={soc_label or 'none'}"
        f"; basis_order={basis_order}"
    )
    write_wannier_hr(output_path, 2 * nwan, degens_up, spinor, header=header)

    print("[wannier-soc] summary", flush=True)
    print(f"  up_hr: {up_path}", flush=True)
    print(f"  dn_hr: {dn_path}", flush=True)
    print(f"  output: {output_path}", flush=True)
    print(f"  nwan: {nwan} -> spinor_dim: {2 * nwan}", flush=True)
    print(f"  nrpts: {len(spinor)}", flush=True)
    print(f"  spin_direction: {nvec.tolist()}", flush=True)
    print(f"  basis_order: {basis_order}", flush=True)
    if basis_groups:
        print(f"  basis_groups: {basis_groups}", flush=True)
    print(f"  win_path: {win_path or ''}", flush=True)
    print(f"  soc_entries: {soc_label or 'none'}", flush=True)

    centres_up = args.centres_up or _infer_centres(up_path)
    centres_dn = args.centres_dn or _infer_centres(dn_path)
    if centres_up:
        centres_out = args.centres_out or _default_centres_out(output_path)
        _write_spinor_centres(centres_out, centres_up, centres_dn, nwan, perm, basis_order)
        print(f"  centres_output: {centres_out}", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generic Wannier90 spinor/SOC downfolding")
    parser.add_argument("--workdir", type=str, default=None, help="Workflow root directory (default: current directory)")
    parser.add_argument("--in_dir", type=str, default=None, help="Input directory containing up/dn hr.dat")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--out_name", type=str, default=None, help="Output filename")
    parser.add_argument("--up", type=str, default="up_hr.dat", help="Spin-up hr.dat path, relative to --in_dir/workdir unless absolute")
    parser.add_argument("--dn", type=str, default="dn_hr.dat", help="Spin-down hr.dat path, relative to --in_dir/workdir unless absolute")
    parser.add_argument("--centres_up", type=str, default=None, help="Spin-up centres.xyz file")
    parser.add_argument("--centres_dn", type=str, default=None, help="Spin-down centres.xyz file")
    parser.add_argument("--centres_out", type=str, default=None, help="Output spinor centres.xyz path")
    parser.add_argument("--theta", type=float, default=0.0, help="Spin polar angle in degrees, used if --spin_direction is omitted")
    parser.add_argument("--phi", type=float, default=0.0, help="Spin azimuth angle in degrees, used if --spin_direction is omitted")
    parser.add_argument("--spin_direction", type=float, nargs=3, default=None, help="Spin quantization direction vector")
    parser.add_argument(
        "--basis_order",
        default="wannier_orbital",
        choices=[
            "wannier_orbital",
            "orbital_interleaved",
            "spinor_block",
            "spin_major",
            "win_interleaved",
            "site_interleaved",
            "atom_interleaved",
            "group_interleaved",
        ],
        help="Output spinor basis. Default keeps the old orbital-interleaved layout.",
    )
    parser.add_argument("--win", default=None, help="Wannier90 .win file used to infer p/d SOC orbital groups")
    parser.add_argument(
        "--soc",
        default="",
        help="Model SOC specs from .win projections, e.g. 'Te:p:0.5;Mn:d:0.05'. Use 'off' to disable.",
    )
    parser.add_argument("--soc_element", default="", help="Element for compatibility --lambda_te p-SOC mode")
    parser.add_argument("--lambda_te", type=float, default=0.0, help="Compatibility onsite p SOC lambda in eV; prefer --soc")
    parser.add_argument("--soc_p_groups", default="", help="Manual p groups, e.g. '10,11,12;25,26,27'")
    parser.add_argument("--soc_p_groups_base", type=int, choices=[0, 1], default=0)
    parser.add_argument("--p_order", default=WANNIER90_P_ORDER)
    parser.add_argument("--d_order", default=WANNIER90_D_ORDER)
    args = parser.parse_args()

    build_wannier_soc(args)


if __name__ == "__main__":
    main()
