"""Compare qe2pert epr.h5 electronic H(k) against hr.dat and u.mat rotations."""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from slw.core.qe2pert_ws import init_rvec_images, set_wigner_seitz_cell, triangular_pair_index


def read_win_kpoints(path):
    kpts = []
    in_block = False
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            low = s.lower()
            if low == "begin kpoints":
                in_block = True
                continue
            if low == "end kpoints":
                break
            if not in_block or not s:
                continue
            parts = s.split()
            if len(parts) >= 3:
                kpts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not kpts:
        raise ValueError(f"No begin/end kpoints block found in {path}")
    return np.asarray(kpts, dtype=np.float64)


def read_u_mat(path):
    with open(path, "r") as f:
        f.readline()
        nk, n1, n2 = (int(x) for x in f.readline().split()[:3])
        mats = np.empty((nk, n1, n2), dtype=np.complex128)
        kpts = np.empty((nk, 3), dtype=np.float64)
        for ik in range(nk):
            line = f.readline()
            while line and not line.strip():
                line = f.readline()
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Missing k-point line before matrix {ik + 1} in {path}")
            kpts[ik] = [float(parts[0]), float(parts[1]), float(parts[2])]
            vals = np.empty(n1 * n2, dtype=np.complex128)
            for i in range(n1 * n2):
                re_im = f.readline().split()
                vals[i] = float(re_im[0]) + 1j * float(re_im[1])
            mats[ik] = vals.reshape((n1, n2))
    return kpts, mats


def read_hr_matrices(path):
    with open(path, "r") as f:
        f.readline()
        nwan = int(f.readline().strip())
        nrpts = int(f.readline().strip())
        ndegen = []
        while len(ndegen) < nrpts:
            ndegen.extend(int(x) for x in f.readline().split())
        r_keys = []
        mats = []
        current_r = None
        current = None
        ir = -1
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            r = (int(parts[0]), int(parts[1]), int(parts[2]))
            if r != current_r:
                if current is not None:
                    mats.append(current / float(ndegen[ir]))
                current_r = r
                r_keys.append(r)
                ir += 1
                current = np.zeros((nwan, nwan), dtype=np.complex128)
            i = int(parts[3]) - 1
            j = int(parts[4]) - 1
            current[i, j] = float(parts[5]) + 1j * float(parts[6])
        if current is not None:
            mats.append(current / float(ndegen[ir]))
    if len(r_keys) != nrpts:
        raise ValueError(f"Expected {nrpts} R points in {path}, got {len(r_keys)}")
    return np.asarray(r_keys, dtype=np.float64), np.asarray(mats, dtype=np.complex128)


def hk_from_hr(hr_path, kpts):
    rvecs, h_r = read_hr_matrices(hr_path)
    phase = np.exp(2j * np.pi * (np.asarray(kpts, dtype=np.float64) @ rvecs.T))
    hk = np.einsum("kr,rij->kij", phase, h_r, optimize=True)
    return 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))


def _read_epr_meta(epr_path):
    with h5py.File(epr_path, "r") as h5:
        at = np.asarray(h5["basic_data/at"], dtype=np.float64).T
        wc = np.asarray(h5["basic_data/wannier_center_cryst"], dtype=np.float64)
        return at, wc, tuple(int(x) for x in h5["basic_data/kc_dim"][()]), int(h5["basic_data/num_wann"][()])


def hk_from_epr(epr_path, kpts):
    at, wc, kc_dim, nwan = _read_epr_meta(epr_path)
    images = init_rvec_images(kc_dim, at)
    hk = np.zeros((len(kpts), nwan, nwan), dtype=np.complex128)
    with h5py.File(epr_path, "r") as h5:
        for jw in range(1, nwan + 1):
            for iw in range(1, jw + 1):
                x = triangular_pair_index(iw, jw)
                ws = set_wigner_seitz_cell(images, at, wc[iw - 1], wc[jw - 1])
                hop = np.asarray(h5[f"electron_wannier/hopping_r{x}"]) + 1j * np.asarray(
                    h5[f"electron_wannier/hopping_i{x}"]
                )
                phase = np.exp(2j * np.pi * (np.asarray(kpts, dtype=np.float64) @ ws.vectors.T))
                vals = phase @ hop
                hk[:, iw - 1, jw - 1] = vals
                if iw != jw:
                    hk[:, jw - 1, iw - 1] = np.conjugate(vals)
    return 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))


def _stats(name, cand, ref):
    diff = cand - ref
    mat = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    refn = np.linalg.norm(ref.reshape(ref.shape[0], -1), axis=1)
    ev_c = np.linalg.eigvalsh(0.5 * (cand + np.swapaxes(cand.conj(), 1, 2)))
    ev_r = np.linalg.eigvalsh(0.5 * (ref + np.swapaxes(ref.conj(), 1, 2)))
    ed = ev_c - ev_r
    print(
        f"{name:18s} mat_rms={float(np.sqrt(np.mean(mat * mat))):.8e} "
        f"rel_rms={float(np.sqrt(np.mean((mat / np.maximum(refn, 1e-30)) ** 2))):.8e} "
        f"mat_max={float(np.max(mat)):.8e} "
        f"eig_rms={float(np.sqrt(np.mean(ed * ed))):.8e} "
        f"eig_max={float(np.max(np.abs(ed))):.8e}"
    )


def compare_one(label, epr, hr, umat, win):
    kpts = read_win_kpoints(win)
    u_kpts, u = read_u_mat(umat)
    if len(kpts) != len(u_kpts) or np.max(np.abs(kpts - u_kpts)) > 1e-7:
        raise ValueError(f"{label}: kpoints in {win} and {umat} do not match")
    print(f"[{label}] loading H_hr(k)")
    h_hr = hk_from_hr(hr, kpts)
    print(f"[{label}] loading H_epr(k)")
    h_epr = hk_from_epr(epr, kpts)
    print(f"[{label}] nk={len(kpts)} nwan={h_hr.shape[1]}")
    _stats("hr", h_hr, h_epr)
    _stats("U H Udag", u @ h_hr @ np.swapaxes(u.conj(), 1, 2), h_epr)
    _stats("Udag H U", np.swapaxes(u.conj(), 1, 2) @ h_hr @ u, h_epr)
    ut = np.swapaxes(u, 1, 2)
    _stats("Ut H Ut_dag", ut @ h_hr @ np.swapaxes(ut.conj(), 1, 2), h_epr)
    _stats("Ut_dag H Ut", np.swapaxes(ut.conj(), 1, 2) @ h_hr @ ut, h_epr)
    print(f"[{label}] ranges hr={np.linalg.eigvalsh(h_hr).min():.8f}..{np.linalg.eigvalsh(h_hr).max():.8f} "
          f"epr={np.linalg.eigvalsh(h_epr).min():.8f}..{np.linalg.eigvalsh(h_epr).max():.8f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--win", required=True)
    p.add_argument("--epr_up")
    p.add_argument("--hr_up")
    p.add_argument("--u_up")
    p.add_argument("--epr_dn")
    p.add_argument("--hr_dn")
    p.add_argument("--u_dn")
    args = p.parse_args()
    if args.epr_up:
        compare_one("up", args.epr_up, args.hr_up, args.u_up, args.win)
    if args.epr_dn:
        compare_one("dn", args.epr_dn, args.hr_dn, args.u_dn, args.win)


if __name__ == "__main__":
    main()
