"""Fit spin-flip hopping templates to a target spinor hr.dat in k space."""

from __future__ import annotations

import argparse
import os

import numpy as np

from slw.core.wannier_io import read_wannier_hr, write_wannier_hr
from slw.exchange.plot_epr_soc_bands import _clean_species, _match_u_to_kpath, _read_u_matrix, _spinor_u, _win_projection_groups
from slw.exchange.spinor_model import WANNIER90_D_ORDER, WANNIER90_P_ORDER, atomic_d_soc_block, atomic_p_soc_block


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
    return None if not rows else np.asarray(rows, dtype=np.float64)


def _hk_from_hmap(hmap, kpts, dim):
    if not hmap:
        raise ValueError("Cannot build H(k): empty hr.dat R-map")
    rvec = np.asarray(sorted(hmap), dtype=np.float64)
    h_r = np.asarray([hmap[tuple(int(x) for x in r)] for r in rvec], dtype=np.complex128)
    for r, h in zip(rvec, h_r):
        if h.shape != (dim, dim):
            raise ValueError(f"R={tuple(r.astype(int))} block shape={h.shape}, expected={(dim, dim)}")
    phase = np.exp(2j * np.pi * (np.asarray(kpts, dtype=np.float64) @ rvec.T))
    hk = np.einsum("kr,rij->kij", phase, h_r, optimize=True)
    return 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))


def _match_matrix_to_kpath_with_indices(u_kpts, mats, kpts_frac, *, tol=1.0e-7, nearest=False, label="matrix"):
    arr = np.asarray(mats, dtype=np.complex128)
    if arr.ndim == 2:
        out = np.broadcast_to(arr, (len(kpts_frac),) + arr.shape).copy()
        return out, np.zeros(len(kpts_frac), dtype=np.int64)
    if arr.ndim != 3:
        raise ValueError(f"{label} must have shape (nk,n,m) or (n,m), got {arr.shape}")
    if arr.shape[0] == len(kpts_frac) and u_kpts is None:
        return arr, np.arange(len(kpts_frac), dtype=np.int64)
    if u_kpts is None:
        raise ValueError(f"{label} nk={arr.shape[0]} does not match requested nk={len(kpts_frac)}, and no kpts were provided")
    src = np.asarray(u_kpts, dtype=np.float64).reshape(-1, 3) % 1.0
    dst = np.asarray(kpts_frac, dtype=np.float64).reshape(-1, 3) % 1.0
    if src.shape[0] != arr.shape[0]:
        raise ValueError(f"{label} kpts length={src.shape[0]} does not match matrices nk={arr.shape[0]}")
    out = np.empty((dst.shape[0],) + arr.shape[1:], dtype=np.complex128)
    indices = np.empty(dst.shape[0], dtype=np.int64)
    max_dist = 0.0
    for ik, kp in enumerate(dst):
        diff = src - kp[None, :]
        diff -= np.rint(diff)
        dist = np.linalg.norm(diff, axis=1)
        idx = int(np.argmin(dist))
        dmin = float(dist[idx])
        max_dist = max(max_dist, dmin)
        if dmin > float(tol) and not nearest:
            raise ValueError(
                f"No matching {label}(k) for k[{ik}]={kp.tolist()} within tol={tol:g}; "
                f"nearest distance={dmin:g}. Use --target_u_nearest for diagnostic-only nearest matching."
            )
        out[ik] = arr[idx]
        indices[ik] = idx
    if max_dist > float(tol):
        print(f"[fit-spinflip-k][WARN] nearest {label}(k) matching max fractional distance={max_dist:g}", flush=True)
    return out, indices


def _read_wannier_eig(path):
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"{path} must contain columns: band_index k_index eig")
    bands = data[:, 0].astype(np.int64)
    kidx = data[:, 1].astype(np.int64)
    eig = data[:, 2].astype(np.float64)
    nb = int(bands.max())
    nk = int(kidx.max())
    out = np.empty((nk, nb), dtype=np.float64)
    seen = np.zeros((nk, nb), dtype=bool)
    out[kidx - 1, bands - 1] = eig
    seen[kidx - 1, bands - 1] = True
    if not bool(np.all(seen)):
        missing = np.argwhere(~seen)
        raise ValueError(f"{path} missing eig entries, first missing k/b={missing[0].tolist()}")
    return out


def _reconstruct_hk_from_w90(eig_path, u_mat_path, u_dis_mat_path, kpts, *, rotation, tol, nearest, label):
    if not eig_path:
        return None
    if not (u_mat_path or u_dis_mat_path):
        raise ValueError(f"{label} reconstruction requires --*_u_mat and/or --*_u_dis_mat")
    eig_all = _read_wannier_eig(eig_path)
    u = None
    u_indices = None
    if u_mat_path:
        u_kpts, u_raw = _read_u_matrix(u_mat_path)
        u, u_indices = _match_matrix_to_kpath_with_indices(
            u_kpts,
            u_raw,
            kpts,
            tol=tol,
            nearest=nearest,
            label=f"{label}_u",
        )
    if u_dis_mat_path:
        udis_kpts, udis_raw = _read_u_matrix(u_dis_mat_path)
        udis, udis_indices = _match_matrix_to_kpath_with_indices(
            udis_kpts,
            udis_raw,
            kpts,
            tol=tol,
            nearest=nearest,
            label=f"{label}_u_dis",
        )
        if u_indices is not None and not np.array_equal(u_indices, udis_indices):
            print(f"[fit-spinflip-k][WARN] {label}_u and {label}_u_dis matched different source k indices", flush=True)
        eig = eig_all[udis_indices]
        if u is not None:
            if u.shape[1] != u.shape[2]:
                raise ValueError(f"{label}_u must be square, got {u.shape}")
            nwan_u = int(u.shape[1])
        else:
            nwan_u = None
        if (nwan_u is None and udis.shape[1] <= udis.shape[2]) or (nwan_u is not None and udis.shape[1] == nwan_u):
            # Wannier90 commonly writes u_dis as (nk, num_wann, num_bands).
            nwan = int(udis.shape[1])
            nb_outer = int(udis.shape[2])
            if eig.shape[1] < nb_outer:
                raise ValueError(f"{eig_path} has nbands={eig.shape[1]}, but u_dis needs {nb_outer}")
            h_opt = (udis * eig[:, None, :nb_outer]) @ np.swapaxes(udis.conj(), 1, 2)
            udis_layout = "wann_by_band"
        elif nwan_u is not None and udis.shape[2] == nwan_u:
            # Also support transposed arrays from npz/preprocessed inputs:
            # (nk, num_bands, num_wann).
            nwan = int(udis.shape[2])
            nb_outer = int(udis.shape[1])
            if eig.shape[1] < nb_outer:
                raise ValueError(f"{eig_path} has nbands={eig.shape[1]}, but u_dis needs {nb_outer}")
            h_opt = np.swapaxes(udis.conj(), 1, 2) @ (eig[:, :nb_outer, None] * udis)
            udis_layout = "band_by_wann"
        else:
            raise ValueError(
                f"{label}_u_dis shape={udis.shape} is incompatible with {label}_u shape={None if u is None else u.shape}; "
                "expected u_dis layout (nk,num_wann,num_bands) or (nk,num_bands,num_wann)"
            )
    else:
        if u is None:
            raise ValueError(f"{label} reconstruction without u_dis requires a U matrix")
        eig = eig_all[u_indices]
        if u.shape[1] != u.shape[2]:
            raise ValueError(f"{label}_u without u_dis must be square, got {u.shape}")
        nwan = u.shape[1]
        if eig.shape[1] < nwan:
            raise ValueError(f"{eig_path} has nbands={eig.shape[1]}, but u needs {nwan}")
        h_opt = np.zeros((len(kpts), nwan, nwan), dtype=np.complex128)
        diag = np.arange(nwan)
        h_opt[:, diag, diag] = eig[:, :nwan]
    if u is None:
        hk = h_opt
    else:
        mode = str(rotation).strip().lower()
        if mode == "udag_h_u":
            hk = np.swapaxes(u.conj(), 1, 2) @ h_opt @ u
        elif mode == "u_h_udag":
            hk = u @ h_opt @ np.swapaxes(u.conj(), 1, 2)
        else:
            raise ValueError(f"Unknown W90 rotation {mode!r}")
    hk = 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))
    print(
        f"[fit-spinflip-k] reconstructed {label} H(k) from eig/U: "
        f"eig={eig_path} u={u_mat_path} u_dis={u_dis_mat_path or ''} "
        f"shape={hk.shape}",
        flush=True,
    )
    return hk


def _reconstruct_target_hk_from_w90(args, kpts):
    return _reconstruct_hk_from_w90(
        args.target_eig,
        args.target_u_mat,
        args.target_u_dis_mat,
        kpts,
        rotation=args.target_w90_rotation,
        tol=args.target_u_match_tol,
        nearest=bool(args.target_u_nearest),
        label="target",
    )


def _load_target_u(args, kpts, dim):
    if not (args.target_u_mat or args.target_u_up_mat or args.target_u_dn_mat):
        return None
    if args.target_u_mat and (args.target_u_up_mat or args.target_u_dn_mat):
        raise ValueError("Use either --target_u_mat or --target_u_up_mat/--target_u_dn_mat, not both")
    if args.target_u_up_mat or args.target_u_dn_mat:
        if not (args.target_u_up_mat and args.target_u_dn_mat):
            raise ValueError("Provide both --target_u_up_mat and --target_u_dn_mat")
        k_up, u_up_raw = _read_u_matrix(args.target_u_up_mat)
        k_dn, u_dn_raw = _read_u_matrix(args.target_u_dn_mat)
        u_up = _match_u_to_kpath(
            k_up,
            u_up_raw,
            kpts,
            tol=args.target_u_match_tol,
            nearest=bool(args.target_u_nearest),
        )
        u_dn = _match_u_to_kpath(
            k_dn,
            u_dn_raw,
            kpts,
            tol=args.target_u_match_tol,
            nearest=bool(args.target_u_nearest),
        )
        u = _spinor_u(u_up, u_dn)
    else:
        k_u, u_raw = _read_u_matrix(args.target_u_mat)
        u = _match_u_to_kpath(
            k_u,
            u_raw,
            kpts,
            tol=args.target_u_match_tol,
            nearest=bool(args.target_u_nearest),
        )
        if u.shape[1:] == (dim // 2, dim // 2):
            u = _spinor_u(u, u)
    if u.shape != (len(kpts), dim, dim):
        raise ValueError(f"target U(k) shape={u.shape}; expected {(len(kpts), dim, dim)}")
    print(f"[fit-spinflip-k] loaded target U(k) shape={u.shape}", flush=True)
    return u


def _rotate_hk(hk, u, mode):
    if u is None:
        return hk
    mode = str(mode).strip().lower()
    if mode == "u_h_udag":
        out = u @ hk @ np.swapaxes(u.conj(), 1, 2)
    elif mode == "udag_h_u":
        out = np.swapaxes(u.conj(), 1, 2) @ hk @ u
    else:
        raise ValueError(f"Unknown --target_u_rotation {mode!r}")
    return 0.5 * (out + np.swapaxes(out.conj(), 1, 2))


def _spin_indices(dim, order="spin_major", win_path=None):
    if dim % 2 != 0:
        raise ValueError(f"spinor dimension must be even, got {dim}")
    n = dim // 2
    mode = str(order).strip().lower().replace("-", "_")
    if mode in {"spin_major", "spin"}:
        return np.arange(n, dtype=np.int64), np.arange(n, 2 * n, dtype=np.int64)
    if mode in {"orbital_interleaved", "interleaved", "pair_interleaved"}:
        return np.arange(0, 2 * n, 2, dtype=np.int64), np.arange(1, 2 * n, 2, dtype=np.int64)
    if mode in {"win_interleaved", "site_interleaved", "atom_interleaved", "group_interleaved"}:
        if not win_path:
            raise ValueError(f"--{order} spin order requires --win")
        _atoms, groups, nproj = _win_projection_groups(win_path)
        if int(nproj) != int(n):
            raise ValueError(f"{win_path} projection count={nproj}, but spinor nwan={n}")
        up_idx = []
        dn_idx = []
        offset = 0
        for group in groups:
            norb = len(group["indices"])
            up_idx.extend(range(offset, offset + norb))
            dn_idx.extend(range(offset + norb, offset + 2 * norb))
            offset += 2 * norb
        if offset != dim:
            raise ValueError(f"win_interleaved index count={offset}, expected dim={dim}")
        return np.asarray(up_idx, dtype=np.int64), np.asarray(dn_idx, dtype=np.int64)
    raise ValueError(f"Unknown spin order {order!r}; use spin_major, orbital_interleaved, or win_interleaved")


def _spinflip_vector(hk, *, blocks="both", spin_order="spin_major", win_path=None):
    nk, dim, _ = hk.shape
    up_idx, dn_idx = _spin_indices(dim, spin_order, win_path=win_path)
    parts = []
    mode = str(blocks).strip().lower()
    if mode in {"ud", "both"}:
        parts.append(hk[:, up_idx[:, None], dn_idx[None, :]].reshape(nk, -1))
    if mode in {"du", "both"}:
        parts.append(hk[:, dn_idx[:, None], up_idx[None, :]].reshape(nk, -1))
    if not parts:
        raise ValueError("--blocks must be ud, du, or both")
    return np.concatenate(parts, axis=1).reshape(-1)


def _spinblock_norms(hk, *, spin_order="spin_major", win_path=None):
    nk, dim, _ = hk.shape
    up_idx, dn_idx = _spin_indices(dim, spin_order, win_path=win_path)
    uu = np.linalg.norm(hk[:, up_idx[:, None], up_idx[None, :]].reshape(nk, -1), axis=1)
    dd = np.linalg.norm(hk[:, dn_idx[:, None], dn_idx[None, :]].reshape(nk, -1), axis=1)
    ud = np.linalg.norm(hk[:, up_idx[:, None], dn_idx[None, :]].reshape(nk, -1), axis=1)
    du = np.linalg.norm(hk[:, dn_idx[:, None], up_idx[None, :]].reshape(nk, -1), axis=1)
    return {
        "uu": float(np.sqrt(np.mean(uu * uu))),
        "dd": float(np.sqrt(np.mean(dd * dd))),
        "ud": float(np.sqrt(np.mean(ud * ud))),
        "du": float(np.sqrt(np.mean(du * du))),
    }


def _print_target_convention_scan(args, kpts):
    if not args.target_eig:
        return
    print("[fit-spinflip-k] target convention scan")
    for rot in ("udag_h_u", "u_h_udag"):
        hk = _reconstruct_hk_from_w90(
            args.target_eig,
            args.target_u_mat,
            args.target_u_dis_mat,
            kpts,
            rotation=rot,
            tol=args.target_u_match_tol,
            nearest=bool(args.target_u_nearest),
            label=f"target_scan_{rot}",
        )
        for order in ("spin_major", "orbital_interleaved", "win_interleaved"):
            try:
                s = _spinblock_norms(hk, spin_order=order, win_path=args.win)
            except Exception as exc:
                print(f"  rot={rot:10s} order={order:20s} failed: {exc}")
                continue
            sf = np.sqrt(s["ud"] ** 2 + s["du"] ** 2)
            sc = np.sqrt(s["uu"] ** 2 + s["dd"] ** 2)
            ratio = sf / max(sc, 1.0e-30)
            print(
                f"  rot={rot:10s} order={order:20s} "
                f"uu={s['uu']:.6g} dd={s['dd']:.6g} ud={s['ud']:.6g} du={s['du']:.6g} "
                f"sf/sc={ratio:.6g}"
            )


def _zero_nonspinflip(hmap, dim, *, onsite_only=False, offsite_only=False):
    n = dim // 2
    out = {}
    for r, h in hmap.items():
        r = tuple(int(x) for x in r)
        keep = np.zeros((dim, dim), dtype=np.complex128)
        if onsite_only and r != (0, 0, 0):
            out[r] = keep
            continue
        if offsite_only and r == (0, 0, 0):
            out[r] = keep
            continue
        arr = np.asarray(h, dtype=np.complex128)
        keep[:n, n:] = arr[:n, n:]
        keep[n:, :n] = arr[n:, :n]
        out[r] = keep
    return out


def _combine_hmaps(base, templates, coeffs, dim):
    out = {tuple(r): np.array(h, dtype=np.complex128, copy=True) for r, h in base.items()}
    zeros = np.zeros((dim, dim), dtype=np.complex128)
    keys = set(out)
    for tmpl in templates:
        keys.update(tmpl)
    for r in keys:
        if r not in out:
            out[r] = zeros.copy()
    for coeff, tmpl in zip(coeffs, templates):
        c = complex(coeff)
        for r, h in tmpl.items():
            out[tuple(r)] += c * np.asarray(h, dtype=np.complex128)
    return out


def _fit_real(design, target):
    a = np.asarray(design, dtype=np.complex128)
    y = np.asarray(target, dtype=np.complex128)
    ar = np.vstack([a.real, a.imag])
    yr = np.concatenate([y.real, y.imag])
    coeff, *_ = np.linalg.lstsq(ar, yr, rcond=None)
    return coeff.astype(np.float64)


def _fit_complex(design, target):
    coeff, *_ = np.linalg.lstsq(np.asarray(design, dtype=np.complex128), np.asarray(target, dtype=np.complex128), rcond=None)
    return coeff


def _norm(x):
    arr = np.asarray(x, dtype=np.complex128).reshape(-1)
    return float(np.sqrt(np.vdot(arr, arr).real))


def _real_overlap(a, b):
    avec = np.asarray(a, dtype=np.complex128).reshape(-1)
    bvec = np.asarray(b, dtype=np.complex128).reshape(-1)
    return float(np.vdot(avec, bvec).real)


def _parse_onsite_specs(text):
    specs = []
    if not text:
        return specs
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.replace(",", ":").split(":") if x.strip()]
        if len(parts) != 2:
            raise ValueError(f"Each --onsite_soc entry must be element:orbital, got {item!r}")
        elem, orb = parts
        orb = orb.lower()
        if orb not in {"p", "d"}:
            raise ValueError(f"Unsupported onsite SOC orbital {orb!r}; use p or d")
        specs.append((_clean_species(elem), orb))
    return specs


def _onsite_soc_template(win_path, specs, dim, *, p_order=WANNIER90_P_ORDER, d_order=WANNIER90_D_ORDER):
    if not win_path:
        raise ValueError("--onsite_soc requires --win")
    if dim % 2 != 0:
        raise ValueError(f"spin-major spinor dimension must be even, got {dim}")
    nwan = dim // 2
    _atoms, groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(f"{win_path} projection count={nproj}, but spinor nwan={nwan}")
    mat = np.zeros((dim, dim), dtype=np.complex128)
    for elem, orb in specs:
        matched = [g for g in groups if g["element"] == elem and g["orbital"] == orb]
        if not matched:
            known = sorted({(g["element"], g["orbital"]) for g in groups})
            raise ValueError(f"No {elem}:{orb} projection groups found in {win_path}; known={known}")
        block = atomic_p_soc_block(1.0, order=p_order) if orb == "p" else atomic_d_soc_block(1.0, order=d_order)
        norb = block.shape[0] // 2
        for group in matched:
            idx0 = np.asarray(group["indices"], dtype=np.int64).reshape(norb)
            idx = np.concatenate([idx0, idx0 + nwan])
            mat[idx[:, None], idx[None, :]] += block
    tmpl = np.zeros((dim, dim), dtype=np.complex128)
    tmpl[:nwan, nwan:] = mat[:nwan, nwan:]
    tmpl[nwan:, :nwan] = mat[nwan:, :nwan]
    return {(0, 0, 0): tmpl}


def _soc_spinflip_mats(win_path, specs, nwan, *, p_order=WANNIER90_P_ORDER, d_order=WANNIER90_D_ORDER):
    if not win_path:
        raise ValueError("--downfold_soc requires --win")
    _atoms, groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(f"{win_path} projection count={nproj}, but collinear nwan={nwan}")
    lud = np.zeros((nwan, nwan), dtype=np.complex128)
    ldu = np.zeros((nwan, nwan), dtype=np.complex128)
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
    return lud, ldu


def _collinear_downfold_template_vec(args, kpts, dim_t):
    specs = _parse_onsite_specs(args.downfold_soc)
    if not specs:
        return None
    if not (args.col_up_eig and args.col_dn_eig):
        raise ValueError("--downfold_soc requires --col_up_eig and --col_dn_eig")
    if not (args.col_up_u_mat or args.col_up_u_dis_mat):
        raise ValueError("--downfold_soc requires --col_up_u_mat and/or --col_up_u_dis_mat")
    if not (args.col_dn_u_mat or args.col_dn_u_dis_mat):
        raise ValueError("--downfold_soc requires --col_dn_u_mat and/or --col_dn_u_dis_mat")
    hup = _reconstruct_hk_from_w90(
        args.col_up_eig,
        args.col_up_u_mat,
        args.col_up_u_dis_mat,
        kpts,
        rotation=args.col_w90_rotation,
        tol=args.target_u_match_tol,
        nearest=bool(args.target_u_nearest),
        label="col_up",
    )
    hdn = _reconstruct_hk_from_w90(
        args.col_dn_eig,
        args.col_dn_u_mat,
        args.col_dn_u_dis_mat,
        kpts,
        rotation=args.col_w90_rotation,
        tol=args.target_u_match_tol,
        nearest=bool(args.target_u_nearest),
        label="col_dn",
    )
    if hup.shape != hdn.shape:
        raise ValueError(f"collinear up/down H(k) shape mismatch: {hup.shape} vs {hdn.shape}")
    nwan = int(hup.shape[1])
    if dim_t != 2 * nwan:
        raise ValueError(f"target dim={dim_t}, but collinear nwan={nwan} implies spinor dim={2 * nwan}")
    lud, ldu = _soc_spinflip_mats(
        args.win,
        specs,
        nwan,
        p_order=args.p_order,
        d_order=args.d_order,
    )
    scale = 1.0 / (float(args.downfold_denom) ** 2)
    hk = np.zeros((len(kpts), dim_t, dim_t), dtype=np.complex128)
    hk[:, :nwan, nwan:] = scale * (hup @ lud @ hdn)
    hk[:, nwan:, :nwan] = scale * (hdn @ ldu @ hup)
    hk = 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))
    return _spinflip_vector(
        hk,
        blocks=args.blocks,
        spin_order=args.template_spin_order,
        win_path=args.win,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target_hr", default=None, help="SCF/reference spinor hr.dat")
    ap.add_argument("--target_eig", default=None, help="Wannier90 .eig file for direct target H(k) reconstruction")
    ap.add_argument("--target_u_dis_mat", default=None, help="Wannier90 *_u_dis.mat for --target_eig disentanglement reconstruction")
    ap.add_argument(
        "--target_w90_rotation",
        choices=["udag_h_u", "u_h_udag"],
        default="udag_h_u",
        help="Final Wannier rotation for --target_eig reconstruction.",
    )
    ap.add_argument("--template_hr", nargs="*", default=[], help="Spin-flip template spinor hr.dat file(s)")
    ap.add_argument("--template_labels", nargs="*", default=None)
    ap.add_argument("--downfold_soc", default="", help="Build a k-space collinear downfold template from --col_up/--col_dn eig/U, e.g. 'I:p'")
    ap.add_argument("--downfold_label", default="col_downfold", help="Label for --downfold_soc template")
    ap.add_argument("--downfold_denom", type=float, default=1.0, help="Energy denominator in eV for --downfold_soc")
    ap.add_argument("--col_up_eig", default=None, help="Collinear spin-up Wannier90 .eig")
    ap.add_argument("--col_up_u_mat", default=None, help="Collinear spin-up Wannier90 *_u.mat")
    ap.add_argument("--col_up_u_dis_mat", default=None, help="Collinear spin-up Wannier90 *_u_dis.mat")
    ap.add_argument("--col_dn_eig", default=None, help="Collinear spin-down Wannier90 .eig")
    ap.add_argument("--col_dn_u_mat", default=None, help="Collinear spin-down Wannier90 *_u.mat")
    ap.add_argument("--col_dn_u_dis_mat", default=None, help="Collinear spin-down Wannier90 *_u_dis.mat")
    ap.add_argument(
        "--col_w90_rotation",
        choices=["udag_h_u", "u_h_udag"],
        default="udag_h_u",
        help="Final Wannier rotation for collinear eig/U reconstruction.",
    )
    ap.add_argument("--onsite_soc", default="", help="Analytic onsite SOC template(s), e.g. 'I:p;Cr:d'. Coefficient is lambda_eff in eV.")
    ap.add_argument("--win", default=None, help="Wannier90 .win file for --onsite_soc")
    ap.add_argument("--p_order", default=WANNIER90_P_ORDER)
    ap.add_argument("--d_order", default=WANNIER90_D_ORDER)
    ap.add_argument("--base_hr", default=None, help="Optional base hr.dat for --out_hr; templates are added to it")
    ap.add_argument("--out_hr", default=None, help="Optional fitted output hr.dat")
    ap.add_argument("--kmesh", type=int, nargs=3, required=True, help="Regular fractional k mesh for fitting")
    ap.add_argument("--kpoints", default=None, help="Additional explicit fractional k points")
    ap.add_argument(
        "--target_u_mat",
        default=None,
        help="U(k) matrix for target_hr gauge rotation or --target_eig reconstruction; accepts *_u.mat, .npy, or .npz",
    )
    ap.add_argument("--target_u_up_mat", default=None, help="Spin-up U(k) for block-diagonal target gauge rotation")
    ap.add_argument("--target_u_dn_mat", default=None, help="Spin-down U(k) for block-diagonal target gauge rotation")
    ap.add_argument(
        "--target_u_rotation",
        choices=["u_h_udag", "udag_h_u"],
        default="udag_h_u",
        help="Apply target gauge rotation as U H U^dag or U^dag H U.",
    )
    ap.add_argument("--target_u_match_tol", type=float, default=1.0e-7, help="Fractional k tolerance for matching U(k)")
    ap.add_argument("--target_u_nearest", action="store_true", help="Use nearest U(k) if exact k matching fails; diagnostic only")
    ap.add_argument("--scan_conventions", action="store_true", help="Print target spin-order/rotation block norms and exit")
    ap.add_argument("--blocks", choices=["ud", "du", "both"], default="both")
    ap.add_argument(
        "--target_spin_order",
        choices=["spin_major", "orbital_interleaved", "win_interleaved"],
        default="spin_major",
        help=(
            "Spin ordering of target_hr. spin_major=[all up, all down]; "
            "orbital_interleaved=[orb1 up, orb1 down, ...]; "
            "win_interleaved=[group1 up block, group1 down block, group2 up block, ...]."
        ),
    )
    ap.add_argument(
        "--template_spin_order",
        choices=["spin_major", "orbital_interleaved", "win_interleaved"],
        default="spin_major",
        help="Spin ordering of template/base HR files.",
    )
    ap.add_argument("--complex_coeff", action="store_true", help="Allow complex fit coefficients")
    ap.add_argument("--split_onsite_offsite", action="store_true", help="Split each template into R=0 and R!=0 templates")
    args = ap.parse_args()

    kpts = _kmesh_points(args.kmesh)
    extra = _parse_kpoints(args.kpoints)
    if extra is not None:
        kpts = np.vstack([kpts, extra])
    if args.scan_conventions:
        _print_target_convention_scan(args, kpts)
        return

    hk_target = _reconstruct_target_hk_from_w90(args, kpts)
    if hk_target is None:
        if not args.target_hr:
            raise ValueError("Provide --target_hr or --target_eig with --target_u_mat")
        dim_t, _deg_t, h_target = _read_wannier_hr_compat(args.target_hr)
        if dim_t % 2 != 0:
            raise ValueError(f"target spinor dimension must be even, got {dim_t}")
        hk_target = _hk_from_hmap(h_target, kpts, dim_t)
        target_u = _load_target_u(args, kpts, dim_t)
        hk_target = _rotate_hk(hk_target, target_u, args.target_u_rotation)
    else:
        dim_t = int(hk_target.shape[1])
        if dim_t % 2 != 0:
            raise ValueError(f"target spinor dimension must be even, got {dim_t}")
        target_u = None
    target_vec = _spinflip_vector(
        hk_target,
        blocks=args.blocks,
        spin_order=args.target_spin_order,
        win_path=args.win,
    )

    templates = []
    labels = []
    raw_labels = args.template_labels or []
    for elem, orb in _parse_onsite_specs(args.onsite_soc):
        templates.append(
            _onsite_soc_template(
                args.win,
                [(elem, orb)],
                dim_t,
                p_order=args.p_order,
                d_order=args.d_order,
            )
        )
        labels.append(f"onsite:{elem}:{orb}:lambda_eV")
    for i, path in enumerate(args.template_hr):
        dim, _deg, hmap = _read_wannier_hr_compat(path)
        if dim != dim_t:
            raise ValueError(f"template {path} dim={dim} differs from target dim={dim_t}")
        label = raw_labels[i] if i < len(raw_labels) else os.path.basename(path)
        if args.split_onsite_offsite:
            templates.append(_zero_nonspinflip(hmap, dim_t, onsite_only=True))
            labels.append(f"{label}:onsite")
            templates.append(_zero_nonspinflip(hmap, dim_t, offsite_only=True))
            labels.append(f"{label}:offsite")
        else:
            templates.append(_zero_nonspinflip(hmap, dim_t))
            labels.append(label)
    if not templates and not args.downfold_soc:
        raise ValueError("No templates provided. Use --onsite_soc, --template_hr, and/or --downfold_soc.")

    design_cols = []
    template_vecs = []
    has_kspace_template = False
    for hmap in templates:
        vec = _spinflip_vector(
            _hk_from_hmap(hmap, kpts, dim_t),
            blocks=args.blocks,
            spin_order=args.template_spin_order,
            win_path=args.win,
        )
        design_cols.append(vec)
        template_vecs.append(vec)
    downfold_vec = _collinear_downfold_template_vec(args, kpts, dim_t)
    if downfold_vec is not None:
        labels.append(str(args.downfold_label))
        design_cols.append(downfold_vec)
        template_vecs.append(downfold_vec)
        has_kspace_template = True
    design = np.stack(design_cols, axis=1)
    coeff = _fit_complex(design, target_vec) if args.complex_coeff else _fit_real(design, target_vec)
    fitted = design @ coeff
    residual = target_vec - fitted
    target_norm = _norm(target_vec)
    fitted_norm = _norm(fitted)
    residual_norm = _norm(residual)

    print("[fit-spinflip-k] summary")
    print(f"  target_hr: {args.target_hr or ''}")
    if args.target_eig:
        print(f"  target_eig: {args.target_eig}")
        print(f"  target_w90_rotation: {args.target_w90_rotation}")
    print(f"  nk: {len(kpts)}  dim: {dim_t}  blocks: {args.blocks}")
    if target_u is not None:
        print(f"  target_u_rotation: {args.target_u_rotation}")
    print(f"  target_spin_order: {args.target_spin_order}")
    print(f"  template_spin_order: {args.template_spin_order}")
    print(f"  target_norm: {target_norm:.10g}")
    print(f"  fitted_norm: {fitted_norm:.10g}")
    print(f"  residual_norm: {residual_norm:.10g}")
    print(f"  relative_residual: {residual_norm / max(target_norm, 1.0e-30):.10g}")
    print("  template diagnostics:")
    for label, vec in zip(labels, template_vecs):
        vec_norm = _norm(vec)
        ov = _real_overlap(vec, target_vec)
        cosine = ov / max(vec_norm * target_norm, 1.0e-30)
        print(f"    {label}: norm={vec_norm:.10g}  real_overlap={ov:.10g}  cosine={cosine:.10g}")
    print("  coefficients:")
    for label, c in zip(labels, coeff):
        if np.iscomplexobj(coeff):
            print(f"    {label}: {c.real:.12g} {c.imag:+.12g}j")
        else:
            print(f"    {label}: {float(c):.12g}")

    if args.out_hr:
        if has_kspace_template:
            raise ValueError("--out_hr is not supported with k-space --downfold_soc templates; omit --out_hr or use an HR template")
        if not args.base_hr:
            raise ValueError("--out_hr requires --base_hr")
        dim_b, deg_b, h_base = _read_wannier_hr_compat(args.base_hr)
        if dim_b != dim_t:
            raise ValueError(f"base dim={dim_b} differs from target dim={dim_t}")
        out = _combine_hmaps(h_base, templates, coeff, dim_t)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_hr)) or ".", exist_ok=True)
        write_wannier_hr(args.out_hr, dim_t, deg_b, out, header="SLW fitted spin-flip k-space model")
        print(f"[fit-spinflip-k] wrote {args.out_hr}")


if __name__ == "__main__":
    main()
