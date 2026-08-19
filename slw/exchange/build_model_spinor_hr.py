"""Build a model spinor hr.dat by adding onsite SOC and template HR terms."""

from __future__ import annotations

import argparse
import os

import numpy as np

from slw.core.wannier_io import read_wannier_hr, write_wannier_hr
from slw.exchange.fit_spinflip_kspace import _onsite_soc_template
from slw.exchange.plot_epr_soc_bands import _clean_species


def _read_wannier_hr_compat(path):
    parsed = read_wannier_hr(path)
    if len(parsed) == 4:
        dim, _nrpts, degens, hmap = parsed
        if int(dim) <= 0 or not hmap:
            raise ValueError(f"Failed to read non-empty Wannier90 hr.dat: {path}")
        return int(dim), list(degens), hmap
    if len(parsed) == 3:
        dim, degens, hmap = parsed
        if int(dim) <= 0 or not hmap:
            raise ValueError(f"Failed to read non-empty Wannier90 hr.dat: {path}")
        return int(dim), list(degens), hmap
    raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")


def _parse_terms(items):
    terms = []
    for item in items or []:
        parts = [x.strip() for x in str(item).split(":") if x.strip()]
        if len(parts) < 2:
            raise ValueError(f"Template term must be coeff:path, got {item!r}")
        coeff = float(parts[0])
        path = ":".join(parts[1:])
        terms.append((coeff, path))
    return terms


def _parse_onsite(items):
    specs = []
    for item in items or []:
        parts = [x.strip() for x in str(item).replace(",", ":").split(":") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"Onsite SOC term must be element:orbital:lambda_eV, got {item!r}")
        elem, orb, lam = parts
        specs.append((_clean_species(elem), orb.lower(), float(lam)))
    return specs


def _add_hmap(out, term, coeff):
    keys = set(out) | set(term)
    dim = next(iter(out.values())).shape[0]
    zero = np.zeros((dim, dim), dtype=np.complex128)
    for r in keys:
        if r not in out:
            out[r] = zero.copy()
        if r in term:
            out[r] += complex(coeff) * np.asarray(term[r], dtype=np.complex128)


def build_model(args):
    dim, degens, hbase = _read_wannier_hr_compat(args.base_hr)
    if dim % 2 != 0:
        raise ValueError(f"spinor dimension must be even, got {dim}")
    out = {tuple(r): np.array(h, dtype=np.complex128, copy=True) for r, h in hbase.items()}

    onsite_terms = _parse_onsite(args.onsite)
    for elem, orb, lam in onsite_terms:
        tmpl = _onsite_soc_template(
            args.win,
            [(elem, orb)],
            dim,
            p_order=args.p_order,
            d_order=args.d_order,
        )
        _add_hmap(out, tmpl, lam)

    template_terms = _parse_terms(args.template)
    for coeff, path in template_terms:
        tdim, _tdeg, thmap = _read_wannier_hr_compat(path)
        if tdim != dim:
            raise ValueError(f"template {path} dim={tdim} differs from base dim={dim}")
        _add_hmap(out, thmap, coeff)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    write_wannier_hr(
        args.output,
        dim,
        [1] * len(out) if len(out) != len(degens) else degens,
        out,
        header="SLW model spinor HR",
    )
    print("[model-spinor-hr] summary")
    print(f"  base_hr: {args.base_hr}")
    print(f"  output: {args.output}")
    print(f"  dim: {dim} nrpts: {len(out)}")
    print(f"  onsite: {[f'{e}:{o}:{lam:g}' for e, o, lam in onsite_terms]}")
    print(f"  templates: {[f'{c:g}:{p}' for c, p in template_terms]}")
    return args.output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_hr", required=True, help="Base spin-major spinor hr.dat")
    ap.add_argument("--win", required=True, help="Wannier90 .win for onsite SOC groups")
    ap.add_argument("--onsite", action="append", default=[], help="Onsite SOC term element:orbital:lambda_eV, e.g. I:p:0.5")
    ap.add_argument("--template", action="append", default=[], help="Template term coeff:path, e.g. 0.02:M_alpha_dp_hr.dat")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--p_order", default="pz,px,py")
    ap.add_argument("--d_order", default="dz2,dxz,dyz,dx2-y2,dxy")
    args = ap.parse_args()
    build_model(args)


if __name__ == "__main__":
    main()
