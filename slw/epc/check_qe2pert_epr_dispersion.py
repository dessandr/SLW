"""Check qe2pert epr.h5 hopping/IFC by reconstructing dispersions."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import spglib

from slw.core.qe2pert_ws import (
    init_rvec_images,
    set_wigner_seitz_cell,
    triangular_pair_index,
)


TPI = 2.0 * np.pi
FPI = 4.0 * np.pi
E2_RY = 2.0
POLAR_GMAX = 14.0
RYD2MEV = 13.605698066e3
Ry_to_eV = 13.605698066


DEFAULT_HSP = {
    "G": np.array([0.0, 0.0, 0.0]),
    "Gamma": np.array([0.0, 0.0, 0.0]),
    "K": np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
    "M": np.array([0.5, 0.0, 0.0]),
    "A": np.array([0.0, 0.0, 0.5]),
    "H": np.array([1.0 / 3.0, 1.0 / 3.0, 0.5]),
    "L": np.array([0.5, 0.0, 0.5]),
    "Lprime": np.array([0.0, 0.5, 0.5]),
}


def _read_h5_meta(epr_path):
    with h5py.File(epr_path, "r") as h5:
        at = np.asarray(h5["basic_data/at"], dtype=np.float64).T
        bg = np.asarray(h5["basic_data/bg"], dtype=np.float64).T
        wc = np.asarray(h5["basic_data/wannier_center_cryst"], dtype=np.float64)
        tau_cart = np.asarray(h5["basic_data/tau"], dtype=np.float64)
        tau = np.linalg.solve(at, tau_cart.T).T
        mass = np.asarray(h5["basic_data/mass"], dtype=np.float64)
        system_2d = bool(h5["basic_data/system_2d"][()]) if "basic_data/system_2d" in h5 else False
        if system_2d and "basic_data/thickness_2d" in h5:
            thickness_2d = float(h5["basic_data/thickness_2d"][()])
        elif system_2d:
            thickness_2d = 6.0 / 0.52917721092
        else:
            thickness_2d = -1.0
        return {
            "at": at,
            "bg": bg,
            "wc": wc,
            "tau": tau,
            "tau_cart": tau_cart,
            "mass": mass,
            "epsil": np.asarray(h5["basic_data/epsil"], dtype=np.float64).T,
            # h5py exposes Perturbo's zeu(3, 3, nat) as (nat, 3, 3),
            # while preserving the two Cartesian axes.  Keep those axes in
            # QE's (electric-field, displacement) order: rgd_blk contracts
            # g_a * zeu(a, i, atom).  Transposing here swaps Z_xz and Z_zx
            # and corrupts the polar correction whenever Z* is nonsymmetric.
            "zstar": np.asarray(h5["basic_data/zstar"], dtype=np.float64),
            "volume": float(h5["basic_data/volume"][()]),
            "alat": float(h5["basic_data/alat"][()]),
            "loto_alpha": float(h5["basic_data/loto_alpha"][()])
            if "basic_data/loto_alpha" in h5
            else 1.0,
            "system_2d": system_2d,
            "thickness_2d": thickness_2d,
            "kc_dim": tuple(int(x) for x in h5["basic_data/kc_dim"][()]),
            "qc_dim": tuple(int(x) for x in h5["basic_data/qc_dim"][()]),
            "nat": int(h5["basic_data/nat"][()]),
            "num_wann": int(h5["basic_data/num_wann"][()]),
            "lpolar": bool(h5["basic_data/lpolar"][()]) if "basic_data/lpolar" in h5 else False,
        }


def _parse_win_path(win_path):
    if not win_path:
        return [
            ("A", np.array([0.0, 0.0, 0.5])),
            ("G", np.array([0.0, 0.0, 0.0])),
            ("M", np.array([0.5, 0.0, 0.0])),
            ("K", np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])),
            ("G", np.array([0.0, 0.0, 0.0])),
        ]
    points = []
    in_path = False
    with open(win_path, "r") as f:
        for line in f:
            s = line.strip()
            low = s.lower()
            if low == "begin kpoint_path":
                in_path = True
                continue
            if low == "end kpoint_path":
                break
            if not in_path or not s:
                continue
            parts = s.split()
            if len(parts) >= 8:
                a = (parts[0], np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
                b = (parts[4], np.array([float(parts[5]), float(parts[6]), float(parts[7])]))
                if not points:
                    points.append(a)
                points.append(b)
    if not points:
        raise ValueError(f"No begin/end kpoint_path block found in {win_path}")
    return points


def _norm_hsp_label(label):
    s = str(label).strip().strip("\"'")
    low = s.lower()
    if low in {"g", "gamma", "gam", "gm", "Γ"}:
        return "G"
    if low in {"l'", "lp", "lprime", "l_prime"}:
        return "Lprime"
    return s


def _display_hsp_label(label):
    s = _norm_hsp_label(label)
    if s == "G":
        return "G"
    if s == "Lprime":
        return "L'"
    return s


def _parse_kpath_arg(text):
    if not text or str(text).strip().lower() in {"", "default", "win"}:
        return None
    tokens = [x for x in str(text).replace(",", " ").split() if x]
    if not tokens:
        return None

    def _is_float(tok):
        try:
            float(tok)
            return True
        except ValueError:
            return False

    if len(tokens) >= 4 and _is_float(tokens[1]):
        if len(tokens) % 4 != 0:
            raise ValueError("--kpath coordinate form must be repeated as: LABEL qx qy qz")
        out = []
        for i in range(0, len(tokens), 4):
            out.append((_display_hsp_label(tokens[i]), np.array([float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])])))
        return out

    out = []
    missing = []
    for tok in tokens:
        lab = _norm_hsp_label(tok)
        if lab not in DEFAULT_HSP:
            missing.append(tok)
            continue
        out.append((_display_hsp_label(lab), DEFAULT_HSP[lab].copy()))
    if missing:
        raise ValueError(f"Unknown --kpath labels {missing}; known={sorted(DEFAULT_HSP.keys())}")
    if len(out) < 2:
        raise ValueError("--kpath must contain at least two points")
    return out


def _resolve_path_points(kpath, win_path):
    custom = _parse_kpath_arg(kpath)
    if custom is not None:
        return custom
    return _parse_win_path(win_path)


def _interpolate_path(points, nseg, bg=None):
    bg_arr = None if bg is None else np.asarray(bg, dtype=np.float64)

    def dist(a, b):
        d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        if bg_arr is not None:
            d = bg_arr @ d
        return float(np.linalg.norm(d))

    kpts = []
    labels = []
    label_pos = []
    xvals = []
    x = 0.0
    for iseg, ((la, ka), (lb, kb)) in enumerate(zip(points[:-1], points[1:])):
        if iseg == 0:
            label_pos.append(x)
            labels.append(la)
        for i in range(nseg):
            t = i / float(nseg)
            k = (1.0 - t) * ka + t * kb
            if kpts:
                x += dist(k, kpts[-1])
            kpts.append(k)
            xvals.append(x)
        x += dist(kb, kpts[-1])
        kpts.append(kb)
        xvals.append(x)
        label_pos.append(x)
        labels.append(lb)
    return np.asarray(kpts), np.asarray(xvals), labels, np.asarray(label_pos)


def _phase(fracs, rvecs):
    return np.exp(2j * np.pi * (np.asarray(fracs, dtype=np.float64) @ np.asarray(rvecs, dtype=np.float64).T))


def _cart_from_cryst(vecs, bg):
    arr = np.asarray(vecs, dtype=np.float64)
    return arr @ np.asarray(bg, dtype=np.float64).T


def _polar_g_vectors(nrx):
    n1, n2, n3 = (int(x) for x in nrx)
    return np.asarray(
        [(m1, m2, m3) for m1 in range(-n1, n1 + 1) for m2 in range(-n2, n2 + 1) for m3 in range(-n3, n3 + 1)],
        dtype=np.float64,
    )


def _polar_nrx_ph(qc_dim, bg, epsil, alpha):
    falph = 4.0 * float(alpha)
    ggmax = POLAR_GMAX * falph
    bg = np.asarray(bg, dtype=np.float64)
    epsil = np.asarray(epsil, dtype=np.float64)
    nrx = []
    for i in range(3):
        denom = float(bg[:, i] @ bg[:, i])
        nrx.append(int(np.ceil(np.sqrt(ggmax / denom))))
    nrx = np.asarray(nrx, dtype=np.int64)
    qdim = np.asarray(qc_dim, dtype=np.int64)
    nrx[qdim < 2] = 0
    return nrx


def _dyn_mat_longrange_raw_3d(meta, qpt):
    nat = meta["nat"]
    nelem = nat * (nat + 1) // 2
    bg = meta["bg"]
    epsil = meta["epsil"]
    zstar = meta["zstar"]
    tau = meta["tau_cart"]
    omega = meta["volume"]
    alpha = meta["loto_alpha"]
    falph = 4.0 * alpha
    ggmax = POLAR_GMAX * falph
    nrx = _polar_nrx_ph(meta["qc_dim"], bg, epsil, alpha)

    xqr_cryst = np.asarray(qpt, dtype=np.float64)[None, :] + _polar_g_vectors(nrx)
    xqr = _cart_from_cryst(xqr_cryst, bg)
    qeq = np.einsum("gi,ij,gj->g", xqr, epsil, xqr, optimize=True)
    keep = (qeq >= 1.0e-14) & (qeq <= ggmax)
    xqr_cryst = xqr_cryst[keep]
    xqr = xqr[keep]
    qeq = qeq[keep]

    dynq = np.zeros((nelem, 3, 3), dtype=np.complex128)
    if xqr.shape[0] == 0:
        return dynq
    weights = np.exp(-qeq / falph) / qeq
    xouter = xqr[:, :, None] * xqr[:, None, :] * weights[:, None, None]

    n = 0
    for ja in range(nat):
        for ia in range(ja + 1):
            phase = np.exp(1j * TPI * (xqr @ (tau[ia] - tau[ja])))
            dd0 = np.einsum("g,gij->ij", phase, xouter, optimize=True)
            dynq[n] = zstar[ia].T @ dd0 @ zstar[ja]
            n += 1
    return dynq * (FPI * E2_RY / omega)


def _dyn_mat_longrange_raw_2d(meta, qpt):
    nat = meta["nat"]
    nelem = nat * (nat + 1) // 2
    bg = meta["bg"]
    epsil = meta["epsil"]
    zstar = meta["zstar"]
    tau = meta["tau_cart"]
    omega = meta["volume"]
    tpiba = TPI / meta["alat"]
    alpha = meta["loto_alpha"]
    falph = 4.0 * alpha
    ggmax = POLAR_GMAX * falph
    nrx = _polar_nrx_ph(meta["qc_dim"], bg, epsil, alpha)

    alat = TPI / tpiba
    fac = FPI * E2_RY / omega * 0.5 * alat / bg[2, 2]
    reff = epsil[:2, :2] * (0.5 * TPI / bg[2, 2])
    reff[0, 0] -= 0.5 * TPI / bg[2, 2]
    reff[1, 1] -= 0.5 * TPI / bg[2, 2]

    xqr_cryst = np.asarray(qpt, dtype=np.float64)[None, :] + _polar_g_vectors(nrx)
    xqr = _cart_from_cryst(xqr_cryst, bg)
    qeq = np.einsum("gi,gi->g", xqr, xqr, optimize=True)
    gp2 = np.einsum("gi,gi->g", xqr[:, :2], xqr[:, :2], optimize=True)
    r = np.zeros_like(qeq)
    gp_keep = gp2 > 1.0e-8
    r[gp_keep] = np.einsum("gi,ij,gj->g", xqr[gp_keep, :2], reff, xqr[gp_keep, :2], optimize=True) / gp2[
        gp_keep
    ]
    keep = (qeq >= 1.0e-14) & (qeq <= ggmax)
    xqr_cryst = xqr_cryst[keep]
    xqr = xqr[keep]
    qeq = qeq[keep]
    r = r[keep]

    dynq = np.zeros((nelem, 3, 3), dtype=np.complex128)
    if xqr.shape[0] == 0:
        return dynq
    sqrt_qeq = np.sqrt(qeq)
    weights = tpiba * np.exp(-qeq / falph) / sqrt_qeq / (1.0 + r * sqrt_qeq)
    xouter = xqr[:, :, None] * xqr[:, None, :] * weights[:, None, None]

    n = 0
    for ja in range(nat):
        for ia in range(ja + 1):
            phase = np.exp(1j * TPI * (xqr @ (tau[ia] - tau[ja])))
            dd0 = np.einsum("g,gij->ij", phase, xouter, optimize=True)
            dynq[n] = zstar[ia].T @ dd0 @ zstar[ja]
            n += 1
    return dynq * fac


def _dyn_mat_longrange_raw(meta, qpt):
    if meta["thickness_2d"] > 0.0:
        return _dyn_mat_longrange_raw_2d(meta, qpt)
    return _dyn_mat_longrange_raw_3d(meta, qpt)


def _polar_onsite_correction(meta):
    nat = meta["nat"]
    dyn_gamma = _dyn_mat_longrange_raw(meta, np.zeros(3, dtype=np.float64))
    if np.max(np.abs(dyn_gamma.imag), initial=0.0) > 1.0e-16:
        raise ValueError("On-site polar correction is not real")

    dd = np.zeros((nat, nat, 3, 3), dtype=np.float64)
    n = 0
    for ja in range(nat):
        for ia in range(ja + 1):
            dd0 = dyn_gamma[n].real
            if ia == ja:
                dd[ia, ia] = 0.5 * (dd0 + dd0.T)
            else:
                dd[ia, ja] = dd0
                dd[ja, ia] = dd0.T
            n += 1
    return -np.sum(dd, axis=1)


def _dyn_mat_longrange(meta, qpt, onsite):
    dynq = _dyn_mat_longrange_raw(meta, qpt)
    for ia in range(meta["nat"]):
        idx = ia * (ia + 3) // 2
        dynq[idx] += onsite[ia]
    return dynq


def compute_bands(epr_path, kpts):
    meta = _read_h5_meta(epr_path)
    at = meta["at"]
    wc = meta["wc"]
    nwan = meta["num_wann"]
    images = init_rvec_images(meta["kc_dim"], at)
    evals = np.zeros((len(kpts), nwan), dtype=np.float64)
    with h5py.File(epr_path, "r") as h5:
        hk = np.zeros((len(kpts), nwan, nwan), dtype=np.complex128)
        for jw in range(1, nwan + 1):
            for iw in range(1, jw + 1):
                x = triangular_pair_index(iw, jw)
                ws = set_wigner_seitz_cell(images, at, wc[iw - 1], wc[jw - 1])
                hop = np.asarray(h5[f"electron_wannier/hopping_r{x}"]) + 1j * np.asarray(
                    h5[f"electron_wannier/hopping_i{x}"]
                )
                if hop.shape[0] != ws.nr:
                    raise ValueError(f"hopping length mismatch pair={iw},{jw}: h5={hop.shape[0]} ws={ws.nr}")
                vals = _phase(kpts, ws.vectors) @ hop
                hk[:, iw - 1, jw - 1] = vals
                if iw != jw:
                    hk[:, jw - 1, iw - 1] = np.conjugate(vals)
        hk = 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))
        for ik in range(len(kpts)):
            evals[ik] = np.linalg.eigvalsh(hk[ik])
    return Ry_to_eV * evals


def _read_hr_dat(hr_path):
    with open(hr_path, "r") as f:
        f.readline()
        nwan = int(f.readline().strip())
        nrpts = int(f.readline().strip())
        ndegen = []
        while len(ndegen) < nrpts:
            ndegen.extend(int(x) for x in f.readline().split())
        rows = []
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append(
                (
                    int(parts[0]),
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]) - 1,
                    int(parts[4]) - 1,
                    float(parts[5]) + 1j * float(parts[6]),
                )
            )
    return nwan, nrpts, np.asarray(ndegen, dtype=np.float64), rows


def _read_wsvec_dat(wsvec_path):
    """Read Wannier90 *_wsvec.dat MDRS shifts keyed by (R1,R2,R3,i,j)."""

    if not wsvec_path:
        return None
    out = {}
    with open(wsvec_path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0].startswith("#"):
                continue
            if len(parts) != 5:
                raise ValueError(f"Invalid wsvec header line in {wsvec_path}: {line.rstrip()}")
            r1, r2, r3, i, j = (int(x) for x in parts)
            n = int(f.readline().strip())
            shifts = []
            for _ in range(n):
                sp = f.readline().split()
                if len(sp) != 3:
                    raise ValueError(f"Invalid wsvec shift line in {wsvec_path} for {(r1, r2, r3, i, j)}")
                shifts.append((int(sp[0]), int(sp[1]), int(sp[2])))
            out[(r1, r2, r3, i - 1, j - 1)] = np.asarray(shifts, dtype=np.float64)
    return out


def compute_bands_hr(hr_path, kpts, wsvec_path=None):
    nwan, nrpts, ndegen, rows = _read_hr_dat(hr_path)
    wsvec = _read_wsvec_dat(wsvec_path)
    hk = np.zeros((len(kpts), nwan, nwan), dtype=np.complex128)
    for idx, (r1, r2, r3, i, j, val) in enumerate(rows):
        ir = idx // (nwan * nwan)
        rvec = np.asarray([r1, r2, r3], dtype=np.float64)
        if wsvec is None:
            phase = np.exp(2j * np.pi * (kpts @ rvec))
        else:
            shifts = wsvec.get((r1, r2, r3, i, j))
            if shifts is None:
                phase = np.exp(2j * np.pi * (kpts @ rvec))
            else:
                phase = np.mean(np.exp(2j * np.pi * (kpts @ (rvec[None, :] + shifts).T)), axis=1)
        hk[:, i, j] += val * phase / ndegen[ir]
    hk = 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))
    evals = np.zeros((len(kpts), nwan), dtype=np.float64)
    for ik in range(len(kpts)):
        evals[ik] = np.linalg.eigvalsh(hk[ik])
    return evals


def _gamma_equiv_mask(qpts, tol=1.0e-10):
    q = np.asarray(qpts, dtype=np.float64)
    centered = q - np.rint(q)
    return np.linalg.norm(centered, axis=1) <= float(tol)


def _acoustic_projector_mass_weighted(mass):
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    nat = mass.shape[0]
    t = np.zeros((3 * nat, 3), dtype=np.complex128)
    sqrt_m = np.sqrt(mass)
    for ia in range(nat):
        for ax in range(3):
            t[3 * ia + ax, ax] = sqrt_m[ia]
    q, _ = np.linalg.qr(t)
    return np.eye(3 * nat, dtype=np.complex128) - q @ q.conj().T


def _accum_ifc_entry(store, counts, key, mat):
    if key in store:
        store[key] += mat
        counts[key] += 1
    else:
        store[key] = np.array(mat, dtype=np.complex128, copy=True)
        counts[key] = 1


def _load_ifc_realspace_dict(h5, meta, images):
    nat = meta["nat"]
    at = meta["at"]
    tau = meta["tau"]
    store = {}
    counts = {}
    for ja in range(1, nat + 1):
        for ia in range(1, ja + 1):
            x = triangular_pair_index(ia, ja)
            ws = set_wigner_seitz_cell(images, at, tau[ia - 1], tau[ja - 1])
            ifc = np.asarray(h5[f"force_constant/ifc{x}"]).astype(np.complex128, copy=False)
            if ifc.shape[0] != ws.nr:
                raise ValueError(f"IFC length mismatch pair={ia},{ja}: h5={ifc.shape[0]} ws={ws.nr}")
            mats = ifc  # h5py exposes Fortran ifc(i,j,ir) as (ir,i,j)
            i0 = ia - 1
            j0 = ja - 1
            for ir, rv in enumerate(ws.vectors):
                r = tuple(int(v) for v in rv)
                mat = mats[ir]
                _accum_ifc_entry(store, counts, (i0, j0, r), mat)
                if i0 != j0 or any(v != 0 for v in r):
                    _accum_ifc_entry(store, counts, (j0, i0, tuple(-v for v in r)), mat.conj().T)
    for key, n in list(counts.items()):
        store[key] /= float(n)
    return store


def _symmetrize_ifc_permutation_asr(phi, nat, *, iterations=5, apply_asr=True):
    out = {k: np.array(v, dtype=np.complex128, copy=True) for k, v in phi.items()}
    niter = max(1, int(iterations))
    for _ in range(niter):
        seen = set()
        for key in list(out.keys()):
            if key in seen:
                continue
            i, j, r = key
            mirror = (j, i, tuple(-x for x in r))
            a = out.get(key, np.zeros((3, 3), dtype=np.complex128))
            b = out.get(mirror, np.zeros((3, 3), dtype=np.complex128)).conj().T
            avg = 0.5 * (a + b)
            out[key] = avg
            out[mirror] = avg.conj().T
            seen.add(key)
            seen.add(mirror)

        if apply_asr:
            for ia in range(nat):
                row_sum = np.zeros((3, 3), dtype=np.complex128)
                for (i, _j, _r), mat in out.items():
                    if i == ia:
                        row_sum += mat
                key0 = (ia, ia, (0, 0, 0))
                out[key0] = out.get(key0, np.zeros((3, 3), dtype=np.complex128)) - row_sum
    return out


def _numbers_from_masses(mass, tol=1.0e-5):
    nums = []
    reps = []
    for m in np.asarray(mass, dtype=np.float64):
        found = None
        for i, r in enumerate(reps):
            if abs(float(m) - float(r)) <= float(tol):
                found = i + 1
                break
        if found is None:
            reps.append(float(m))
            found = len(reps)
        nums.append(found)
    return np.asarray(nums, dtype=np.int32)


def _build_sym_atom_maps(tau, rotations, translations, tol=1.0e-5):
    tau = np.asarray(tau, dtype=np.float64)
    maps = []
    shifts = []
    for rot, trans in zip(rotations, translations):
        amap = []
        ashift = []
        for pos in tau:
            p = pos @ np.asarray(rot, dtype=np.float64).T + np.asarray(trans, dtype=np.float64)
            best = -1
            best_shift = None
            best_err = 1.0e30
            for ib, tb in enumerate(tau):
                d = p - tb
                n = np.rint(d)
                err = float(np.linalg.norm(d - n))
                if err < best_err:
                    best = ib
                    best_shift = n.astype(np.int64)
                    best_err = err
            if best < 0 or best_err > tol:
                raise ValueError(f"Failed to map atom under symmetry op; best_err={best_err:.3e} tol={tol:.3e}")
            amap.append(best)
            ashift.append(best_shift)
        maps.append(np.asarray(amap, dtype=np.int64))
        shifts.append(np.asarray(ashift, dtype=np.int64))
    return maps, shifts


def _cart_rotation_from_frac(at, rot):
    a = np.asarray(at, dtype=np.float64)
    r = np.asarray(rot, dtype=np.float64)
    return a @ r @ np.linalg.inv(a)


def _map_pair_under_op(i, j, rvec, rot, trans, tau, amap, ashift):
    ip = int(amap[i])
    li = ashift[i]
    pos_jr = np.asarray(tau[j], dtype=np.float64) + np.asarray(rvec, dtype=np.float64)
    mapped_jr = pos_jr @ np.asarray(rot, dtype=np.float64).T + np.asarray(trans, dtype=np.float64)
    jp = -1
    lj = None
    best_err = 1.0e30
    for ib, tb in enumerate(tau):
        d = mapped_jr - tb
        n = np.rint(d)
        err = float(np.linalg.norm(d - n))
        if err < best_err:
            jp = ib
            lj = n.astype(np.int64)
            best_err = err
    if jp < 0 or best_err > 1.0e-5:
        raise ValueError(f"Failed to map IFC pair under symmetry op; best_err={best_err:.3e}")
    return ip, jp, tuple(int(x) for x in (lj - li))


def _symmetrize_ifc_crystal(phi, meta, *, symprec=1.0e-5, iterations=3, apply_asr=True):
    lattice = np.asarray(meta["at"], dtype=np.float64).T
    positions = np.asarray(meta["tau"], dtype=np.float64) % 1.0
    numbers = _numbers_from_masses(meta["mass"])
    dataset = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=float(symprec))
    if dataset is None:
        raise RuntimeError(f"spglib failed to find symmetry with symprec={symprec}")
    rotations = np.asarray(dataset["rotations"], dtype=np.int64)
    translations = np.asarray(dataset["translations"], dtype=np.float64)
    atom_maps, atom_shifts = _build_sym_atom_maps(positions, rotations, translations, tol=max(1.0e-5, 10.0 * float(symprec)))

    out = {k: np.array(v, dtype=np.complex128, copy=True) for k, v in phi.items()}
    for _ in range(max(1, int(iterations))):
        accum = {}
        counts = {}
        for key, mat in out.items():
            i, j, rvec = key
            for rot, trans, amap, ashift in zip(rotations, translations, atom_maps, atom_shifts):
                mapped = _map_pair_under_op(i, j, rvec, rot, trans, positions, amap, ashift)
                qrot = _cart_rotation_from_frac(meta["at"], rot)
                rmat = qrot @ mat @ qrot.T
                _accum_ifc_entry(accum, counts, mapped, rmat)
        out = {k: accum[k] / float(counts[k]) for k in accum}
        out = _symmetrize_ifc_permutation_asr(out, meta["nat"], iterations=1, apply_asr=bool(apply_asr))
    try:
        sg_name = dataset["international"]
    except Exception:
        sg_name = "unknown"
    print(
        f"[qe2pert-epr] crystal IFC symmetry: spacegroup={sg_name} "
        f"ops={len(rotations)} symprec={float(symprec):.3e} keys={len(out)}",
        flush=True,
    )
    return out


def _fourier_ifc_dict(phi, qpts, nat):
    q = np.asarray(qpts, dtype=np.float64)
    nm = 3 * nat
    dq = np.zeros((len(q), nm, nm), dtype=np.complex128)
    for (i, j, r), mat in phi.items():
        phase = np.exp(2j * np.pi * (q @ np.asarray(r, dtype=np.float64)))
        si = slice(3 * i, 3 * (i + 1))
        sj = slice(3 * j, 3 * (j + 1))
        dq[:, si, sj] += phase[:, None, None] * mat[None, :, :]
    return dq


def _apply_loto_override(meta, loto_dim=None, no_loto=False):
    out = dict(meta)
    if no_loto:
        out["lpolar"] = False
        return out
    mode = str(loto_dim or "auto").strip().lower()
    if mode in {"auto", ""}:
        return out
    if mode in {"none", "off", "false", "0"}:
        out["lpolar"] = False
        return out
    out["lpolar"] = True
    if mode in {"2d", "2"}:
        out["system_2d"] = True
        if out.get("thickness_2d", -1.0) <= 0.0:
            out["thickness_2d"] = 6.0 / 0.52917721092
        return out
    if mode in {"3d", "3"}:
        out["system_2d"] = False
        out["thickness_2d"] = -1.0
        return out
    raise ValueError(f"Unsupported loto_dim={loto_dim!r}; use auto|2d|3d|none")


def compute_phonons(
    epr_path,
    qpts,
    *,
    asr=False,
    asr_gamma_project=True,
    gamma_tol=1.0e-10,
    debug_gamma=False,
    fc_symmetry=False,
    fc_symmetry_iter=5,
    crystal_asr=False,
    symprec=1.0e-5,
    no_loto=False,
    loto_dim="auto",
):
    meta = _apply_loto_override(_read_h5_meta(epr_path), loto_dim=loto_dim, no_loto=no_loto)
    at = meta["at"]
    tau = meta["tau"]
    nat = meta["nat"]
    mass = np.asarray(meta["mass"], dtype=np.float64)
    if mass.shape != (nat,):
        raise ValueError(f"mass length mismatch: mass={mass.shape} nat={nat}")
    if np.any(mass <= 0.0):
        raise ValueError("All atomic masses in basic_data/mass must be positive")
    polar_onsite = _polar_onsite_correction(meta) if meta["lpolar"] else None
    images = init_rvec_images(meta["qc_dim"], at)
    nm = 3 * nat
    evals = np.zeros((len(qpts), nm), dtype=np.float64)
    mode_mass = np.repeat(mass, 3)
    mass_factor = 1.0 / np.sqrt(mode_mass[:, None] * mode_mass[None, :])
    with h5py.File(epr_path, "r") as h5:
        if crystal_asr:
            phi = _load_ifc_realspace_dict(h5, meta, images)
            phi = _symmetrize_ifc_crystal(
                phi,
                meta,
                symprec=float(symprec),
                iterations=int(fc_symmetry_iter),
                apply_asr=True,
            )
            dq = _fourier_ifc_dict(phi, qpts, nat)
        elif fc_symmetry:
            phi = _load_ifc_realspace_dict(h5, meta, images)
            phi = _symmetrize_ifc_permutation_asr(
                phi,
                nat,
                iterations=int(fc_symmetry_iter),
                apply_asr=True,
            )
            dq = _fourier_ifc_dict(phi, qpts, nat)
        else:
            dq = np.zeros((len(qpts), nm, nm), dtype=np.complex128)
            gamma_sr = np.zeros((nm, nm), dtype=np.complex128) if asr else None
            for ja in range(1, nat + 1):
                for ia in range(1, ja + 1):
                    x = triangular_pair_index(ia, ja)
                    ws = set_wigner_seitz_cell(images, at, tau[ia - 1], tau[ja - 1])
                    ifc = np.asarray(h5[f"force_constant/ifc{x}"]).astype(np.complex128, copy=False)
                    if ifc.shape[0] != ws.nr:
                        raise ValueError(f"IFC length mismatch pair={ia},{ja}: h5={ifc.shape[0]} ws={ws.nr}")
                    mats = ifc  # h5py exposes Fortran ifc(i,j,ir) as (ir,i,j)
                    vals = np.einsum("qr,rij->qij", _phase(qpts, ws.vectors), mats, optimize=True)
                    si = slice(3 * (ia - 1), 3 * ia)
                    sj = slice(3 * (ja - 1), 3 * ja)
                    dq[:, si, sj] = vals
                    if asr:
                        gamma_pair = np.sum(mats, axis=0)
                        gamma_sr[si, sj] = gamma_pair
                    if ia != ja:
                        dq[:, sj, si] = np.swapaxes(vals.conj(), 1, 2)
                        if asr:
                            gamma_sr[sj, si] = gamma_pair.conj().T
            if asr:
                corr = np.zeros((nm, nm), dtype=np.complex128)
                for ia in range(nat):
                    si = slice(3 * ia, 3 * (ia + 1))
                    row_sum = np.zeros((3, 3), dtype=np.complex128)
                    for ja in range(nat):
                        sj = slice(3 * ja, 3 * (ja + 1))
                        row_sum += gamma_sr[si, sj]
                    corr[si, si] -= row_sum
                dq += corr[None, :, :]
        if meta["lpolar"]:
            for iq, qpt in enumerate(qpts):
                dyn_lr = _dyn_mat_longrange(meta, qpt, polar_onsite)
                n = 0
                for ja in range(1, nat + 1):
                    for ia in range(1, ja + 1):
                        si = slice(3 * (ia - 1), 3 * ia)
                        sj = slice(3 * (ja - 1), 3 * ja)
                        dq[iq, si, sj] += dyn_lr[n]
                        if ia != ja:
                            dq[iq, sj, si] += dyn_lr[n].conj().T
                        n += 1
        dq = 0.5 * (dq + np.swapaxes(dq.conj(), 1, 2))
        dq *= mass_factor[None, :, :]
        gamma_mask = _gamma_equiv_mask(qpts, tol=gamma_tol)
        if asr and asr_gamma_project:
            proj = _acoustic_projector_mass_weighted(mass)
            for iq in np.where(gamma_mask)[0]:
                if debug_gamma:
                    before = np.linalg.eigvalsh(dq[iq])
                    before_freq = np.sqrt(np.abs(before))
                    before_freq[before <= 0.0] *= -1.0
                    print(
                        "[qe2pert-epr][asr-debug] "
                        f"iq={int(iq)} q={np.asarray(qpts[iq]).tolist()} "
                        f"before_acoustic_meV={(before_freq[:3] * RYD2MEV).tolist()}",
                        flush=True,
                    )
                dq[iq] = proj @ dq[iq] @ proj
                dq[iq] = 0.5 * (dq[iq] + dq[iq].conj().T)
        for iq in range(len(qpts)):
            evals[iq] = np.linalg.eigvalsh(dq[iq])
        if debug_gamma:
            gamma_ids = np.where(gamma_mask)[0]
            print(
                "[qe2pert-epr][asr-debug] "
                f"gamma_like_indices={gamma_ids.tolist()} gamma_tol={float(gamma_tol):.3e} "
                f"asr={bool(asr)} gamma_project={bool(asr_gamma_project)}",
                flush=True,
            )
    # Match Perturbo phdisp output: solve_phonon_modes returns Ry, calc_bands writes meV.
    freqs = np.sqrt(np.abs(evals))
    freqs[evals <= 0.0] *= -1.0
    return freqs * RYD2MEV


def _write_gnu(path, xvals, vals):
    with open(path, "w") as f:
        for i, x in enumerate(xvals):
            row = " ".join(f"{float(v): .12e}" for v in vals[i])
            f.write(f"{float(x): .12e} {row}\n")


def _read_perturbo_phdisp(path):
    blocks = []
    cur = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                if cur:
                    blocks.append(np.asarray(cur, dtype=np.float64))
                    cur = []
                continue
            if len(parts) < 5:
                continue
            cur.append([float(x) for x in parts[:5]])
    if cur:
        blocks.append(np.asarray(cur, dtype=np.float64))
    if not blocks:
        raise ValueError(f"No phonon blocks found in {path}")
    npts = blocks[0].shape[0]
    for ib, block in enumerate(blocks):
        if block.shape != (npts, 5):
            raise ValueError(f"Inconsistent phdisp block {ib}: got {block.shape}, expected {(npts, 5)}")
    xvals = blocks[0][:, 0]
    qpts = blocks[0][:, 1:4]
    vals = np.column_stack([block[:, 4] for block in blocks])
    return qpts, xvals, vals


def _compare_phonons(label, calc_vals, ref_vals):
    if calc_vals.shape != ref_vals.shape:
        raise ValueError(f"{label} shape mismatch: calc={calc_vals.shape} ref={ref_vals.shape}")
    diff = calc_vals - ref_vals
    flat = int(np.argmax(np.abs(diff)))
    iq, im = np.unravel_index(flat, diff.shape)
    print(
        f"{label} phdisp diff (meV): "
        f"rms={float(np.sqrt(np.mean(diff * diff))):.8e} "
        f"max={float(np.max(np.abs(diff))):.8e} at iq={iq} mode={im + 1}"
    )
    print(
        f"{label} worst values (meV): "
        f"calc={float(calc_vals[iq, im]):.10f} ref={float(ref_vals[iq, im]):.10f} "
        f"diff={float(diff[iq, im]):.10f}"
    )


def _decorate_path_axis(ax, labels, label_pos):
    for xpos in label_pos:
        ax.axvline(float(xpos), color="0.75", lw=0.6, zorder=0)
    ax.set_xlim(float(label_pos[0]), float(label_pos[-1]))
    ax.set_xticks(label_pos)
    ax.set_xticklabels([str(x).replace("G", r"$\Gamma$") for x in labels])
    ax.grid(axis="y", color="0.9", lw=0.6)


def _plot_single(path, xvals, vals, labels, label_pos, ylabel, title=None, *, zero_line=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(xvals, vals, color="black", lw=0.8)
    _decorate_path_axis(ax, labels, label_pos)
    if zero_line:
        ax.axhline(0.0, color="0.25", lw=0.7)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_updn(path, xvals, up_vals, dn_vals, labels, label_pos, ylabel, title=None, *, panels=True):
    if not panels:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        ax.plot(xvals, up_vals, color="#1f77b4", lw=0.8, alpha=0.95)
        ax.plot(xvals, dn_vals, color="#d62728", lw=0.8, alpha=0.8, ls="--")
        _decorate_path_axis(ax, labels, label_pos)
        ax.axhline(0.0, color="0.25", lw=0.7)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.plot([], [], color="#1f77b4", lw=1.2, label="up")
        ax.plot([], [], color="#d62728", lw=1.2, ls="--", label="down")
        ax.legend(frameon=False, loc="best")
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True, constrained_layout=True)
    for ax, vals, color, lab in (
        (axes[0], up_vals, "#1f77b4", "up"),
        (axes[1], dn_vals, "#d62728", "down"),
    ):
        ax.plot(xvals, vals, color=color, lw=0.8)
        _decorate_path_axis(ax, labels, label_pos)
        ax.axhline(0.0, color="0.25", lw=0.7)
        ax.text(0.01, 0.92, lab, transform=ax.transAxes, ha="left", va="top", fontsize=11)
        ax.set_ylabel(ylabel)
    if title:
        axes[0].set_title(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _band_plot_reference(bands, ref, efermi, nvalence):
    vals = [np.asarray(x, dtype=np.float64) for x in bands]
    mode = str(ref).strip().lower()
    if mode == "zero":
        return 0.0, "zero"
    if mode == "fermi":
        return float(efermi), "fermi"
    if mode == "max":
        return float(max(np.max(x) for x in vals)), "max"
    if mode != "vbm":
        raise ValueError(f"Unknown --band_ref {ref!r}")
    if nvalence is None:
        raise ValueError("--band_ref vbm requires --nvalence")
    nval = int(nvalence)
    if any(nval < 1 or nval > x.shape[1] for x in vals):
        shapes = [x.shape for x in vals]
        raise ValueError(f"Invalid --nvalence={nval} for band shapes={shapes}")
    return float(max(np.max(x[:, nval - 1]) for x in vals)), f"vbm:nvalence={nval}"


def main():
    p = argparse.ArgumentParser(description="Reconstruct qe2pert epr.h5 electronic and phonon dispersions.")
    p.add_argument("--epr", default=None, help="Input prefix_epr.h5 for single-spin mode")
    p.add_argument("--epr_up", default=None, help="Input up-spin prefix_epr.h5")
    p.add_argument("--epr_dn", default=None, help="Input down-spin prefix_epr.h5")
    p.add_argument("--hr_up", default=None, help="Optional up-spin Wannier90 *_hr.dat for band plotting")
    p.add_argument("--hr_dn", default=None, help="Optional down-spin Wannier90 *_hr.dat for band plotting")
    p.add_argument("--wsvec_up", default=None, help="Optional up-spin Wannier90 *_wsvec.dat for MDRS band phases")
    p.add_argument("--wsvec_dn", default=None, help="Optional down-spin Wannier90 *_wsvec.dat for MDRS band phases")
    p.add_argument("--win", default=None, help="Wannier .win file with kpoint_path")
    p.add_argument(
        "--kpath",
        default=None,
        help=(
            "Custom path. Label form: 'G M K G' using built-in hexagonal labels. "
            "Coordinate form: 'G 0 0 0 X 0.5 0 0'. Use 'win' or omit to read --win/default."
        ),
    )
    p.add_argument("--nseg", type=int, default=40, help="Interpolated points per path segment")
    p.add_argument(
        "--asr",
        "--sumrule",
        nargs="?",
        const="simple",
        default="none",
        choices=["none", "simple", "crystal"],
        help="ASR mode: --asr is simple; --asr crystal applies spglib crystal IFC symmetrization plus ASR.",
    )
    p.add_argument("--fc_symmetry", action="store_true",
                   help="Apply SLW fallback force-constant symmetrization: permutation symmetry plus iterative ASR in real space.")
    p.add_argument("--fc_symmetry_iter", type=int, default=5,
                   help="Iterations for --fc_symmetry permutation+ASR projection.")
    p.add_argument("--no_loto", action="store_true",
                   help="Disable polar/nonanalytic LO-TO correction even when epr.h5 has lpolar=true.")
    p.add_argument("--loto_dim", choices=["auto", "2d", "3d", "none"], default="auto",
                   help="Override LO-TO correction dimensionality. auto follows epr.h5 system_2d; none disables LO-TO.")
    p.add_argument("--no_asr_gamma_project", action="store_true",
                   help="Disable the extra Gamma acoustic-subspace projection used with --asr.")
    p.add_argument("--gamma_tol", type=float, default=1.0e-10,
                   help="Fractional-coordinate tolerance for treating q as Gamma in the ASR projector.")
    p.add_argument("--symprec", type=float, default=1.0e-5,
                   help="spglib symmetry tolerance for --asr crystal.")
    p.add_argument("--debug_gamma", action="store_true",
                   help="Print Gamma-like q indices and acoustic frequencies before ASR projection.")
    p.add_argument("--out_prefix", default=None)
    p.add_argument("--plot", action="store_true", help="Write PNG plots")
    p.add_argument("--band_ref", choices=["zero", "fermi", "vbm", "max"], default="zero",
                   help="Plot-only electronic energy reference. max reproduces the legacy top-of-Wannier-window shift.")
    p.add_argument("--efermi", type=float, default=0.0,
                   help="Fermi energy in eV used by --band_ref fermi.")
    p.add_argument("--nvalence", type=int, default=None,
                   help="Occupied band count per spin used by --band_ref vbm.")
    p.add_argument("--compare_phdisp", default=None, help="Optional Perturbo *.phdisp reference for phonon comparison")
    args = p.parse_args()
    asr_mode = str(args.asr).strip().lower()
    asr_enabled = asr_mode in {"simple", "crystal"}
    crystal_asr = asr_mode == "crystal"

    ref_qpts = ref_xvals = ref_phonons = None
    if args.compare_phdisp:
        ref_qpts, ref_xvals, ref_phonons = _read_perturbo_phdisp(args.compare_phdisp)

    if args.epr_up or args.epr_dn:
        if not args.epr_up or not args.epr_dn:
            raise ValueError("--epr_up and --epr_dn must be provided together")
        if ref_qpts is None:
            points = _resolve_path_points(args.kpath, args.win)
            kpts, xvals, labels, label_pos = _interpolate_path(points, args.nseg, _read_h5_meta(args.epr_up)["bg"])
        else:
            kpts = ref_qpts
            xvals = ref_xvals
            labels = [""] * 2
            label_pos = np.asarray([float(xvals[0]), float(xvals[-1])], dtype=np.float64)
        out_prefix = args.out_prefix or "qe2pert_epr_check"
        if args.hr_up or args.hr_dn:
            if not args.hr_up or not args.hr_dn:
                raise ValueError("--hr_up and --hr_dn must be provided together")
            if (args.wsvec_up is None) != (args.wsvec_dn is None):
                raise ValueError("--wsvec_up and --wsvec_dn must be provided together")
            bands_up = compute_bands_hr(args.hr_up, kpts, args.wsvec_up)
            bands_dn = compute_bands_hr(args.hr_dn, kpts, args.wsvec_dn)
            band_source = "hr.dat"
            if args.wsvec_up or args.wsvec_dn:
                band_source = "hr.dat+wsvec.dat"
        else:
            bands_up = compute_bands(args.epr_up, kpts)
            bands_dn = compute_bands(args.epr_dn, kpts)
            band_source = "epr.h5/electron_wannier"
        phonons = compute_phonons(
            args.epr_up,
            kpts,
            asr=asr_enabled,
            asr_gamma_project=not bool(args.no_asr_gamma_project),
            gamma_tol=float(args.gamma_tol),
            debug_gamma=bool(args.debug_gamma),
            fc_symmetry=bool(args.fc_symmetry),
            fc_symmetry_iter=int(args.fc_symmetry_iter),
            crystal_asr=crystal_asr,
            symprec=float(args.symprec),
            no_loto=bool(args.no_loto),
            loto_dim=str(args.loto_dim),
        )
        if ref_phonons is not None:
            _compare_phonons("up", phonons, ref_phonons)
        _write_gnu(out_prefix + "_bands_up.dat", xvals, bands_up)
        _write_gnu(out_prefix + "_bands_dn.dat", xvals, bands_dn)
        _write_gnu(out_prefix + "_phonons.dat", xvals, phonons)
        np.savez(
            out_prefix + "_dispersion.npz",
            kpts=kpts,
            xvals=xvals,
            labels=np.asarray(labels, dtype=object),
            label_pos=label_pos,
            bands_up=bands_up,
            bands_dn=bands_dn,
            phonons=phonons,
            asr=np.array(asr_mode, dtype=object),
            fc_symmetry=np.array(bool(args.fc_symmetry)),
            crystal_asr=np.array(crystal_asr),
            no_loto=np.array(bool(args.no_loto)),
            loto_dim=np.array(str(args.loto_dim), dtype=object),
            asr_gamma_project=np.array(not bool(args.no_asr_gamma_project)),
            symprec=np.array(float(args.symprec)),
        )
        print(f"wrote {out_prefix}_bands_up.dat")
        print(f"wrote {out_prefix}_bands_dn.dat")
        print(f"wrote {out_prefix}_phonons.dat")
        print(f"wrote {out_prefix}_dispersion.npz")
        if args.plot:
            eref, eref_label = _band_plot_reference(
                [bands_up, bands_dn], args.band_ref, args.efermi, args.nvalence
            )
            bands_up_plot = bands_up - eref
            bands_dn_plot = bands_dn - eref
            _plot_updn(
                out_prefix + "_bands_updn.png",
                xvals,
                bands_up_plot,
                bands_dn_plot,
                labels,
                label_pos,
                "Energy - Eref",
                f"qe2pert bands from {band_source} ({eref_label}, Eref={eref:.6f})",
            )
            _plot_single(
                out_prefix + "_phonons.png",
                xvals,
                phonons,
                labels,
                label_pos,
                "Phonon frequency (meV)",
                f"qe2pert epr.h5 phonons (ASR={asr_mode})",
                zero_line=True,
            )
            print(f"wrote {out_prefix}_bands_updn.png")
            print(f"wrote {out_prefix}_phonons.png")
            print(f"plot band reference: {eref_label} Eref={eref:.12f}")
        print(f"up band range: {float(np.min(bands_up)):.8f} .. {float(np.max(bands_up)):.8f}")
        print(f"dn band range: {float(np.min(bands_dn)):.8f} .. {float(np.max(bands_dn)):.8f}")
        print(f"band source: {band_source}")
        print(f"phonon LO-TO: disabled={bool(args.no_loto)} dim={args.loto_dim}")
        print(f"phonon ASR/sumrule: {asr_mode}")
        print(f"phonon FC symmetry: {bool(args.fc_symmetry)} iter={int(args.fc_symmetry_iter)}")
        if crystal_asr:
            print(f"phonon crystal ASR symprec: {float(args.symprec):.3e}")
        if asr_enabled:
            print(f"phonon ASR Gamma projector: {not bool(args.no_asr_gamma_project)} tol={float(args.gamma_tol):.3e}")
        print(f"phonon frequency range (meV): {float(np.min(phonons)):.8e} .. {float(np.max(phonons)):.8e}")
        return

    if not args.epr:
        raise ValueError("Provide --epr for single-spin mode or --epr_up/--epr_dn for up/down mode")
    if ref_qpts is None:
        points = _resolve_path_points(args.kpath, args.win)
        kpts, xvals, labels, label_pos = _interpolate_path(points, args.nseg, _read_h5_meta(args.epr)["bg"])
    else:
        kpts = ref_qpts
        xvals = ref_xvals
        labels = [""] * 2
        label_pos = np.asarray([float(xvals[0]), float(xvals[-1])], dtype=np.float64)
    out_prefix = args.out_prefix or os.path.splitext(os.path.basename(args.epr))[0]
    bands = compute_bands(args.epr, kpts)
    phonons = compute_phonons(
        args.epr,
        kpts,
        asr=asr_enabled,
        asr_gamma_project=not bool(args.no_asr_gamma_project),
        gamma_tol=float(args.gamma_tol),
        debug_gamma=bool(args.debug_gamma),
        fc_symmetry=bool(args.fc_symmetry),
        fc_symmetry_iter=int(args.fc_symmetry_iter),
        crystal_asr=crystal_asr,
        symprec=float(args.symprec),
        no_loto=bool(args.no_loto),
        loto_dim=str(args.loto_dim),
    )
    if ref_phonons is not None:
        _compare_phonons("single", phonons, ref_phonons)

    _write_gnu(out_prefix + "_bands.dat", xvals, bands)
    _write_gnu(out_prefix + "_phonons.dat", xvals, phonons)
    np.savez(
        out_prefix + "_dispersion.npz",
        kpts=kpts,
        xvals=xvals,
        labels=np.asarray(labels, dtype=object),
        label_pos=label_pos,
        bands=bands,
        phonons=phonons,
        asr=np.array(asr_mode, dtype=object),
        fc_symmetry=np.array(bool(args.fc_symmetry)),
        crystal_asr=np.array(crystal_asr),
        no_loto=np.array(bool(args.no_loto)),
        loto_dim=np.array(str(args.loto_dim), dtype=object),
        asr_gamma_project=np.array(not bool(args.no_asr_gamma_project)),
        symprec=np.array(float(args.symprec)),
    )
    print(f"wrote {out_prefix}_bands.dat")
    print(f"wrote {out_prefix}_phonons.dat")
    print(f"wrote {out_prefix}_dispersion.npz")
    if args.plot:
        eref, eref_label = _band_plot_reference([bands], args.band_ref, args.efermi, args.nvalence)
        _plot_single(
            out_prefix + "_bands.png",
            xvals,
            bands - eref,
            labels,
            label_pos,
            "Energy - Eref",
            f"qe2pert epr.h5 bands ({eref_label}, Eref={eref:.6f})",
            zero_line=True,
        )
        _plot_single(
            out_prefix + "_phonons.png",
            xvals,
            phonons,
            labels,
            label_pos,
            "Phonon frequency (meV)",
            f"qe2pert epr.h5 phonons (ASR={asr_mode})",
            zero_line=True,
        )
        print(f"wrote {out_prefix}_bands.png")
        print(f"wrote {out_prefix}_phonons.png")
        print(f"plot band reference: {eref_label} Eref={eref:.12f}")
    print(f"band range: {float(np.min(bands)):.8f} .. {float(np.max(bands)):.8f}")
    print(f"phonon LO-TO: disabled={bool(args.no_loto)} dim={args.loto_dim}")
    print(f"phonon ASR/sumrule: {asr_mode}")
    print(f"phonon FC symmetry: {bool(args.fc_symmetry)} iter={int(args.fc_symmetry_iter)}")
    if crystal_asr:
        print(f"phonon crystal ASR symprec: {float(args.symprec):.3e}")
    if asr_enabled:
        print(f"phonon ASR Gamma projector: {not bool(args.no_asr_gamma_project)} tol={float(args.gamma_tol):.3e}")
    print(f"phonon frequency range (meV): {float(np.min(phonons)):.8e} .. {float(np.max(phonons)):.8e}")


if __name__ == "__main__":
    main()
