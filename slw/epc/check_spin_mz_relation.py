import argparse
import json
import os

import h5py
import numpy as np


def _frac_key(frac, mesh):
    out = []
    for x, n in zip(frac, mesh):
        xi = int(np.rint((float(x) % 1.0) * int(n))) % int(n)
        out.append(xi)
    return tuple(out)


def _infer_mesh_from_kpts(k_fracs, tol=1e-8):
    nk = []
    for iax in range(3):
        vals = np.sort(np.mod(k_fracs[:, iax], 1.0))
        uniq = []
        for v in vals:
            if not uniq or abs(v - uniq[-1]) > tol:
                uniq.append(float(v))
        nk.append(len(uniq))
    return tuple(int(x) for x in nk)


def _load_h5(path):
    ap = os.path.abspath(path)
    with h5py.File(ap, "r") as h5:
        g = np.asarray(h5["g_band"], dtype=np.complex128)
        k = np.asarray(h5["k_fracs"], dtype=np.float64)
        q = np.asarray(h5["q_fracs"], dtype=np.float64)
        attrs = {k_: v for k_, v in h5.attrs.items()}
    nk = _infer_mesh_from_kpts(k)
    nq = tuple(int(x) for x in attrs.get("supercell_qmesh", (len(q), 1, 1)))
    if len(nq) != 3:
        nq = (len(q), 1, 1)
    return {"path": ap, "g": g, "k": k, "q": q, "nk": nk, "nq": nq, "attrs": attrs}


def _build_lookup(fracs, mesh):
    return {_frac_key(fr, mesh): i for i, fr in enumerate(fracs)}


def _mz_map(frac):
    return np.mod(np.array([frac[0], frac[1], -frac[2]], dtype=np.float64), 1.0)


def _norm_rel(a, b):
    d = a - b
    ea = float(np.linalg.norm(d))
    den = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-15)
    return ea, ea / den


def main():
    p = argparse.ArgumentParser(
        description="Check spin-flip relation under Mz in k/q space: g_dn(Mz k, Mz q) ~ g_up(k,q)."
    )
    p.add_argument("--g_up", required=True, help="g(k,q) h5 for spin-up")
    p.add_argument("--g_dn", required=True, help="g(k,q) h5 for spin-down")
    p.add_argument("--conjugate", action="store_true", help="Apply complex conjugation to RHS matrix before compare")
    p.add_argument("--dagger", action="store_true", help="Apply Hermitian conjugation to RHS matrix before compare")
    p.add_argument("--out_json", default=None, help="Optional report json")
    args = p.parse_args()

    up = _load_h5(args.g_up)
    dn = _load_h5(args.g_dn)

    if up["g"].shape != dn["g"].shape:
        raise ValueError(f"shape mismatch: up={up['g'].shape}, dn={dn['g'].shape}")
    if up["k"].shape != dn["k"].shape or up["q"].shape != dn["q"].shape:
        raise ValueError("k/q grid shape mismatch between up/down files")

    k_lookup_dn = _build_lookup(dn["k"], up["nk"])
    q_lookup_dn = _build_lookup(dn["q"], up["nq"])

    nq, nk, _, _ = up["g"].shape
    abs_terms = []
    rel_terms = []
    miss_k = 0
    miss_q = 0

    for iq in range(nq):
        q_t = _mz_map(up["q"][iq])
        iq_t = q_lookup_dn.get(_frac_key(q_t, up["nq"]))
        if iq_t is None:
            miss_q += 1
            continue

        for ik in range(nk):
            k_t = _mz_map(up["k"][ik])
            ik_t = k_lookup_dn.get(_frac_key(k_t, up["nk"]))
            if ik_t is None:
                miss_k += 1
                continue

            lhs = dn["g"][iq_t, ik_t]
            rhs = up["g"][iq, ik]
            if args.conjugate:
                rhs = rhs.conj()
            if args.dagger:
                rhs = rhs.conj().T
            ea, er = _norm_rel(lhs, rhs)
            abs_terms.append(ea)
            rel_terms.append(er)

    report = {
        "relation": "g_dn(Mz k, Mz q) ~= g_up(k,q)",
        "g_up": up["path"],
        "g_dn": dn["path"],
        "conjugate": bool(args.conjugate),
        "dagger": bool(args.dagger),
        "n_compared": int(len(abs_terms)),
        "nq": int(nq),
        "nk": int(nk),
        "missing_q_map": int(miss_q),
        "missing_k_map": int(miss_k),
        "mean_abs": float(np.mean(abs_terms)) if abs_terms else None,
        "max_abs": float(np.max(abs_terms)) if abs_terms else None,
        "mean_rel": float(np.mean(rel_terms)) if rel_terms else None,
        "max_rel": float(np.max(rel_terms)) if rel_terms else None,
    }

    print(json.dumps(report, indent=2))
    if args.out_json:
        with open(os.path.abspath(args.out_json), "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
