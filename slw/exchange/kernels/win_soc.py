"""Wannier projection and SOC helpers shared by native exchange kernels."""

from __future__ import annotations

import argparse
import glob
import os
import re

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slw.core.constants import BOHR_TO_ANG
from slw.core.structure import read_wannier90_kpoint_path
from slw.exchange.kernels.epr import _build_hk_from_epr
from slw.exchange.kernels.spinor import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    add_atomic_d_soc,
    add_atomic_p_soc,
    atomic_d_soc_block,
    atomic_p_soc_block,
    normalize_spin_direction,
    spinor_from_collinear,
)


def _read_epr_lattice_ang(epr_path):
    with h5py.File(epr_path, "r") as h5:
        at = np.asarray(h5["basic_data/at"], dtype=np.float64).T
        alat_bohr = float(h5["basic_data/alat"][()])
    return at.T * (alat_bohr * BOHR_TO_ANG)


def _display_label(tok):
    text = str(tok).strip()
    return r"$\Gamma$" if text.upper() in {"G", "GAMMA", "Γ"} else text


def _build_kpath(win_path, gvec_ang, npoints):
    segments = read_wannier90_kpoint_path(win_path)
    kpts = []
    kdist = []
    current = 0.0
    ticks = []
    tick_labels = []
    for index, (start_label, start, end_label, end) in enumerate(segments):
        fractions = np.linspace(
            0.0,
            1.0,
            int(npoints),
            endpoint=index == len(segments) - 1,
        )
        points = start[None, :] + fractions[:, None] * (end - start)[None, :]
        length = float(np.linalg.norm((end - start) @ gvec_ang))
        kpts.append(points)
        kdist.append(current + fractions * length)
        if not ticks or abs(ticks[-1] - current) > 1.0e-12:
            ticks.append(current)
            tick_labels.append(_display_label(start_label))
        current += length
        ticks.append(current)
        tick_labels.append(_display_label(end_label))
    return (
        np.concatenate(kpts, axis=0),
        np.concatenate(kdist),
        np.asarray(ticks, dtype=np.float64),
        tick_labels,
    )


def _parse_soc_p_groups(text, base=0):
    if not text:
        return []
    out = []
    offset = 1 if int(base) == 1 else 0
    for group in str(text).split(";"):
        group = group.strip()
        if not group:
            continue
        vals = [int(x.strip()) - offset for x in group.replace(":", ",").split(",") if x.strip()]
        if len(vals) != 3:
            raise ValueError(f"Each --soc_p_groups entry must contain 3 indices, got {vals}")
        out.append(vals)
    return out


def _clean_species(label):
    return re.sub(r"\d+$", "", str(label).strip()).lower()


def _projection_orbitals(text):
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*0\b", "s", str(text), flags=re.IGNORECASE)
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*1\b", "p", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*2\b", "d", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(l|ang|angular)_?mom(entum)?\s*=\s*3\b", "f", clean, flags=re.IGNORECASE)
    fields = [x.strip().lower() for x in re.split(r"[,; \t]+", clean) if x.strip()]
    out = []
    for item in fields:
        item = item.strip("{}()")
        if item in {"s", "p", "d", "f"}:
            out.append(item)
    return out


def _orbital_count(kind):
    counts = {"s": 1, "p": 3, "d": 5, "f": 7}
    if kind not in counts:
        raise ValueError(f"Unsupported projection orbital kind '{kind}'")
    return counts[kind]


def _parse_win_atoms_and_projections(win_path):
    atoms = []
    projections = []
    block = None
    with open(win_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("!", 1)[0].split("#", 1)[0].strip()
            low = line.lower()
            if not line:
                continue
            if low.startswith(("begin atoms_frac", "begin atoms_cart")):
                block = "atoms"
                continue
            if low.startswith("begin projections"):
                block = "projections"
                continue
            if low.startswith(("end atoms_frac", "end atoms_cart", "end projections")):
                block = None
                continue
            if block == "atoms":
                toks = line.split()
                if len(toks) >= 4:
                    atoms.append(toks[0])
            elif block == "projections":
                if ":" in line:
                    lhs, rhs = line.split(":", 1)
                else:
                    lhs, rhs = line, line
                orbitals = _projection_orbitals(rhs)
                if orbitals:
                    projections.append((lhs.strip(), orbitals))
    return atoms, projections


def _infer_win_path(epr_path, explicit=None):
    if explicit:
        return explicit
    base = os.path.dirname(os.path.abspath(epr_path)) or "."
    stem = os.path.basename(epr_path)
    candidates = []
    for suffix in ("_epr.h5", ".h5"):
        if stem.endswith(suffix):
            candidates.append(os.path.join(base, stem[: -len(suffix)] + ".win"))
    candidates.extend(sorted(glob.glob(os.path.join(base, "*.win"))))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _win_projection_groups(win_path):
    atoms, projections = _parse_win_atoms_and_projections(win_path)
    if not atoms:
        raise ValueError(f"No atoms_frac/atoms_cart block found in {win_path}")
    if not projections:
        raise ValueError(f"No projections block found in {win_path}")
    groups = []
    offset = 0
    used_atom_projection = [False] * len(atoms)
    for label, orbitals in projections:
        label_clean = _clean_species(label)
        exact_matches = [i for i, atom in enumerate(atoms) if atom.lower() == label.lower()]
        species_matches = [i for i, atom in enumerate(atoms) if _clean_species(atom) == label_clean]
        matches = exact_matches if exact_matches else species_matches
        if not matches:
            raise ValueError(f"Projection label '{label}' did not match any atom in {win_path}")
        for ia in matches:
            if used_atom_projection[ia]:
                raise ValueError(
                    f"Atom '{atoms[ia]}' is matched by multiple projection lines in {win_path}; "
                    "automatic SOC group inference would be ambiguous."
                )
            used_atom_projection[ia] = True
            for orb in orbitals:
                n = _orbital_count(orb)
                groups.append({
                    "atom_index": ia,
                    "atom_label": atoms[ia],
                    "element": _clean_species(atoms[ia]),
                    "orbital": orb,
                    "indices": list(range(offset, offset + n)),
                })
                offset += n
    return atoms, groups, offset


def _infer_soc_p_groups_from_win(win_path, element="Te"):
    _, all_groups, nproj = _win_projection_groups(win_path)
    target = _clean_species(element)
    groups = [g["indices"] for g in all_groups if g["element"] == target and g["orbital"] == "p"]
    return groups, nproj


def _parse_soc_specs(text):
    specs = []
    if not text:
        return specs
    for item in re.split(r"[;]+", str(text)):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in re.split(r"[:,= \t]+", item) if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"Each --soc entry must be element:orbital:lambda_eV, got {item!r}")
        elem, orb, lam = parts
        orb = orb.lower()
        if orb not in {"p", "d"}:
            raise ValueError(f"Unsupported SOC orbital {orb!r}; supported: p, d")
        specs.append((_clean_species(elem), orb, float(lam)))
    return specs


def _read_wannier_u_mat(path):
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        nk, n1, n2 = (int(x) for x in f.readline().split()[:3])
        kpts = np.empty((nk, 3), dtype=np.float64)
        mats = np.empty((nk, n1, n2), dtype=np.complex128)
        for ik in range(nk):
            line = f.readline()
            while line and not line.strip():
                line = f.readline()
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Missing k-point line before U matrix {ik + 1} in {path}")
            kpts[ik] = [float(parts[0]), float(parts[1]), float(parts[2])]
            vals = np.empty(n1 * n2, dtype=np.complex128)
            for i in range(n1 * n2):
                re_im = f.readline().split()
                if len(re_im) < 2:
                    raise ValueError(f"Missing complex entry {i + 1} for U matrix {ik + 1} in {path}")
                vals[i] = float(re_im[0]) + 1j * float(re_im[1])
            mats[ik] = vals.reshape((n1, n2))
    return kpts, mats


def _read_u_matrix(path):
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return None, np.asarray(arr, dtype=np.complex128)
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        key = "u" if "u" in data else "U" if "U" in data else "u_matrix" if "u_matrix" in data else None
        if key is None:
            raise ValueError(f"{path} must contain one of arrays: u, U, u_matrix")
        kpts = None
        for kkey in ("kpts", "kpoints", "kpts_frac"):
            if kkey in data:
                kpts = np.asarray(data[kkey], dtype=np.float64)
                break
        return kpts, np.asarray(data[key], dtype=np.complex128)
    return _read_wannier_u_mat(path)


def _match_u_to_kpath(u_kpts, u_mats, kpts_frac, *, tol=1.0e-7, nearest=False):
    u = np.asarray(u_mats, dtype=np.complex128)
    if u.ndim == 2:
        return np.broadcast_to(u, (len(kpts_frac),) + u.shape).copy()
    if u.ndim != 3:
        raise ValueError(f"U matrix must have shape (nk,n,n) or (n,n), got {u.shape}")
    if u.shape[0] == len(kpts_frac) and u_kpts is None:
        return u
    if u_kpts is None:
        raise ValueError(f"U matrix nk={u.shape[0]} does not match kpath nk={len(kpts_frac)}, and no kpts were provided")
    src = np.asarray(u_kpts, dtype=np.float64).reshape(-1, 3) % 1.0
    dst = np.asarray(kpts_frac, dtype=np.float64).reshape(-1, 3) % 1.0
    if src.shape[0] != u.shape[0]:
        raise ValueError(f"U kpts length={src.shape[0]} does not match U matrices nk={u.shape[0]}")
    out = np.empty((dst.shape[0],) + u.shape[1:], dtype=np.complex128)
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
                f"No matching U(k) for path k[{ik}]={kp.tolist()} within tol={tol:g}; "
                f"nearest distance={dmin:g}. Use --u_nearest for diagnostic-only nearest matching."
            )
        out[ik] = u[idx]
    if max_dist > float(tol):
        print(f"[epr-soc-band][WARN] nearest U(k) matching max fractional distance={max_dist:g}", flush=True)
    return out


def _atomic_soc_projector_matrix(entries, nproj, args):
    mat = np.zeros((2 * int(nproj), 2 * int(nproj)), dtype=np.complex128)
    for entry in entries:
        orb = entry["orbital"]
        lam = float(entry["lambda_ev"])
        groups = entry["groups"]
        if orb == "p":
            block = atomic_p_soc_block(lam, order=args.p_order)
        elif orb == "d":
            block = atomic_d_soc_block(lam, order=args.d_order)
        else:
            raise ValueError(f"Unsupported SOC orbital: {orb}")
        norb = block.shape[0] // 2
        for group in groups:
            oidx = np.asarray(group, dtype=np.int64).reshape(norb)
            if np.any(oidx < 0) or np.any(oidx >= int(nproj)):
                raise ValueError(f"SOC orbital index out of projector range nproj={nproj}: {oidx.tolist()}")
            idx = np.concatenate([oidx, oidx + int(nproj)])
            mat[idx[:, None], idx[None, :]] += block
    return mat


def _spinor_u(u_up, u_dn=None):
    up = np.asarray(u_up, dtype=np.complex128)
    dn = up if u_dn is None else np.asarray(u_dn, dtype=np.complex128)
    if up.ndim != 3 or up.shape[1] != up.shape[2]:
        raise ValueError(f"Rotated SOC currently requires square U_up(k) with shape (nk,n,n), got {up.shape}")
    if dn.ndim != 3 or dn.shape[1] != dn.shape[2]:
        raise ValueError(f"Rotated SOC currently requires square U_dn(k) with shape (nk,n,n), got {dn.shape}")
    if up.shape != dn.shape:
        raise ValueError(f"U_up/U_dn shape mismatch: {up.shape} vs {dn.shape}")
    nk, n, _ = up.shape
    out = np.zeros((nk, 2 * n, 2 * n), dtype=np.complex128)
    out[:, :n, :n] = up
    out[:, n:, n:] = dn
    return out


def _load_path_u(path, kpts_frac, args, *, label):
    u_kpts, u_raw = _read_u_matrix(path)
    u_path = _match_u_to_kpath(
        u_kpts,
        u_raw,
        kpts_frac,
        tol=args.u_match_tol,
        nearest=bool(args.u_nearest),
    )
    print(f"[epr-soc-band] loaded {label} U(k) {path} shape={u_path.shape}", flush=True)
    return u_path


def _apply_rotated_soc(hspin, args, nwan, kpts_frac):
    entries, win_path = _resolve_soc_groups_from_win(args, nwan)
    if not entries:
        return hspin, entries, win_path, {}
    if args.u_up_mat or args.u_dn_mat:
        if not (args.u_up_mat and args.u_dn_mat):
            raise ValueError("--soc_gauge rotated needs both --u_up_mat and --u_dn_mat, or a common --u_mat")
        u_up = _load_path_u(args.u_up_mat, kpts_frac, args, label="spin-up")
        u_dn = _load_path_u(args.u_dn_mat, kpts_frac, args, label="spin-down")
        u_paths = {"u_up_mat": str(args.u_up_mat), "u_dn_mat": str(args.u_dn_mat), "u_mat": ""}
    else:
        if not args.u_mat:
            raise ValueError("--soc_gauge rotated requires --u_mat or both --u_up_mat/--u_dn_mat")
        u_up = _load_path_u(args.u_mat, kpts_frac, args, label="common")
        u_dn = u_up
        u_paths = {"u_up_mat": "", "u_dn_mat": "", "u_mat": str(args.u_mat)}
    expected = (int(nwan), int(nwan))
    if u_up.shape[1:] != expected or u_dn.shape[1:] != expected:
        raise ValueError(f"U matrix shapes up={u_up.shape}, dn={u_dn.shape}; expected (nk,{nwan},{nwan})")
    hsoc_proj = _atomic_soc_projector_matrix(entries, int(nwan), args)
    uspin = _spinor_u(u_up, u_dn)
    if args.soc_rotation == "u_soc_udag":
        hsoc_w = uspin @ hsoc_proj @ np.swapaxes(uspin.conj(), 1, 2)
    elif args.soc_rotation == "udag_soc_u":
        hsoc_w = np.swapaxes(uspin.conj(), 1, 2) @ hsoc_proj @ uspin
    else:
        raise ValueError(f"Unknown --soc_rotation {args.soc_rotation!r}")
    out = np.asarray(hspin, dtype=np.complex128) + hsoc_w
    for entry in entries:
        print(
            f"[epr-soc-band] added rotated {entry['element']}:{entry['orbital']} SOC "
            f"lambda={float(entry['lambda_ev']):g} eV groups={entry['groups']} source={entry['source']}",
            flush=True,
        )
    print(
        f"[epr-soc-band] rotated SOC gauge={args.soc_rotation} "
        f"u_mat={u_paths['u_mat'] or 'spin-resolved'} "
        f"hsoc_norm={float(np.linalg.norm(hsoc_w.reshape(hsoc_w.shape[0], -1), axis=1).mean()):.6g}",
        flush=True,
    )
    return out, entries, win_path, u_paths


def _resolve_soc_groups_from_win(args, nwan):
    specs = _parse_soc_specs(args.soc)
    legacy_groups = _parse_soc_p_groups(args.soc_p_groups, base=args.soc_p_groups_base)
    if legacy_groups and abs(float(args.lambda_te)) > 0.0 or abs(float(args.lambda_te)) > 0.0:
        specs.append((_clean_species(args.soc_element), "p", float(args.lambda_te)))
    if not specs:
        return [], None

    entries = []
    if legacy_groups:
        entries.append({
            "element": _clean_species(args.soc_element),
            "orbital": "p",
            "lambda_ev": float(args.lambda_te),
            "groups": legacy_groups,
            "source": "manual --soc_p_groups",
        })

    win_path = _infer_win_path(args.epr_up, explicit=args.win)
    if not win_path and any(not legacy_groups or spec[1] != "p" for spec in specs):
        raise ValueError("--win is required for automatic SOC group inference")
    if win_path:
        _atoms, all_groups, nproj = _win_projection_groups(win_path)
        if nproj != int(nwan):
            raise ValueError(f"{win_path} projection count={nproj} but EPR H(k) nwan={nwan}")
        for elem, orb, lam in specs:
            if legacy_groups and elem == _clean_species(args.soc_element) and orb == "p" and abs(lam - float(args.lambda_te)) < 1e-14:
                continue
            groups = [g["indices"] for g in all_groups if g["element"] == elem and g["orbital"] == orb]
            if not groups:
                known = sorted({(g["element"], g["orbital"]) for g in all_groups})
                raise ValueError(f"No {elem}:{orb} projection groups found in {win_path}; known={known}")
            entries.append({
                "element": elem,
                "orbital": orb,
                "lambda_ev": lam,
                "groups": groups,
                "source": win_path,
            })
    return entries, win_path


def _resolve_soc_p_groups(args, nwan):
    groups = _parse_soc_p_groups(args.soc_p_groups, base=args.soc_p_groups_base)
    if groups:
        return groups
    win_path = _infer_win_path(args.epr_up, explicit=args.win)
    if not win_path:
        return []
    groups, nproj = _infer_soc_p_groups_from_win(win_path, element=args.soc_element)
    if nproj != int(nwan):
        raise ValueError(f"{win_path} projection count={nproj} but EPR H(k) nwan={nwan}")
    print(f"[epr-soc-band] auto SOC p groups from {win_path}: {groups}", flush=True)
    return groups


def _apply_model_soc(hspin, args, nwan):
    entries, win_path = _resolve_soc_groups_from_win(args, nwan)
    out = hspin
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
            f"[epr-soc-band] added {entry['element']}:{orb} SOC "
            f"lambda={lam:g} eV groups={groups} source={entry['source']}",
            flush=True,
        )
    return out, entries, win_path


def _band_reference(evals, args):
    ref = str(args.ref).strip().lower()
    vals = np.asarray(evals, dtype=np.float64)
    if ref in {"fermi", "ef", "efermi"}:
        return float(args.efermi), "efermi"
    if ref in {"zero", "none", "absolute"}:
        return 0.0, "zero"
    if ref in {"vbm", "valence"}:
        if args.nvalence is not None:
            nval = int(args.nvalence)
        else:
            raise ValueError("--ref vbm requires --nvalence. Half-filling is often wrong for magnetic Wannier subspaces.")
        if nval <= 0 or nval > vals.shape[1]:
            raise ValueError(f"Invalid nvalence={nval}; band count={vals.shape[1]}")
        vband = vals[:, nval - 1]
        cband = vals[:, nval] if nval < vals.shape[1] else np.full(vals.shape[0], np.nan)
        ik_vbm = int(np.nanargmax(vband))
        vbm = float(vband[ik_vbm])
        cbm = float(np.nanmin(cband)) if nval < vals.shape[1] else np.nan
        gap = cbm - vbm if np.isfinite(cbm) else np.nan
        return vbm, f"vbm:nvalence={nval}:ik={ik_vbm}:gap={gap:.10g}"
    raise ValueError(f"Unknown --ref {args.ref!r}; use fermi, vbm, or zero")


def build_soc_bands(args):
    rvec = _read_epr_lattice_ang(args.epr_up)
    gvec = 2.0 * np.pi * np.linalg.inv(rvec).T
    kpts_frac, kdist, ticks, tick_labels = _build_kpath(
        args.win, gvec, args.band_points
    )

    print(f"[epr-soc-band] building H_up/H_dn on kpath nk={len(kpts_frac)}", flush=True)
    hk_up = _build_hk_from_epr(args.epr_up, kpts_frac, unit=args.hr_unit)
    hk_dn = _build_hk_from_epr(args.epr_dn, kpts_frac, unit=args.hr_unit)
    if hk_up.shape != hk_dn.shape:
        raise ValueError(f"H_up/H_dn shape mismatch: {hk_up.shape} vs {hk_dn.shape}")

    nvec = normalize_spin_direction(args.spin_direction)
    hspin = spinor_from_collinear(hk_up, hk_dn, n=nvec)
    if args.soc_gauge == "wannier":
        hspin, soc_entries, win_path = _apply_model_soc(hspin, args, hk_up.shape[-1])
        u_paths = {}
    elif args.soc_gauge == "rotated":
        hspin, soc_entries, win_path, u_paths = _apply_rotated_soc(hspin, args, hk_up.shape[-1], kpts_frac)
    else:
        raise ValueError(f"Unknown --soc_gauge {args.soc_gauge!r}")
    hspin = 0.5 * (hspin + np.swapaxes(hspin.conj(), -1, -2))
    evals = np.linalg.eigvalsh(hspin)
    eref, ref_label = _band_reference(evals, args)
    bands = evals - eref
    print(f"[epr-soc-band] reference: {ref_label} E_ref={eref:.10g} eV", flush=True)
    if str(args.ref).strip().lower() == "vbm":
        nval = int(args.nvalence)
        if nval < evals.shape[1]:
            vbm = float(np.nanmax(evals[:, nval - 1]))
            cbm = float(np.nanmin(evals[:, nval]))
            print(
                f"[epr-soc-band] vbm diagnostics: nvalence={nval} "
                f"VBM={vbm:.10g} CBM={cbm:.10g} gap={cbm-vbm:.10g} eV",
                flush=True,
            )
    soc_p_groups = []
    soc_d_groups = []
    for entry in soc_entries:
        if entry["orbital"] == "p":
            soc_p_groups.extend(entry["groups"])
        elif entry["orbital"] == "d":
            soc_d_groups.extend(entry["groups"])
    return {
        "bands": np.asarray(bands, dtype=np.float64),
        "raw_bands": np.asarray(evals, dtype=np.float64),
        "eref": np.asarray(eref, dtype=np.float64),
        "ref": np.asarray(ref_label, dtype=object),
        "kdist": kdist,
        "kpts_frac": kpts_frac,
        "ticks": ticks,
        "tick_labels": tick_labels,
        "rvec_ang": rvec,
        "gvec_ang": gvec,
        "spin_direction": nvec,
        "soc_p_groups": np.asarray(soc_p_groups, dtype=np.int64) if soc_p_groups else np.zeros((0, 3), dtype=np.int64),
        "soc_d_groups": np.asarray(soc_d_groups, dtype=np.int64) if soc_d_groups else np.zeros((0, 5), dtype=np.int64),
        "soc_entries": np.asarray(
            [f"{e['element']}:{e['orbital']}:{float(e['lambda_ev']):.16g}" for e in soc_entries],
            dtype=object,
        ),
        "win_path": np.asarray(win_path or "", dtype=object),
        "soc_gauge": np.asarray(args.soc_gauge, dtype=object),
        "soc_rotation": np.asarray(args.soc_rotation, dtype=object),
        "u_mat_path": np.asarray(u_paths.get("u_mat", ""), dtype=object),
        "u_up_mat_path": np.asarray(u_paths.get("u_up_mat", ""), dtype=object),
        "u_dn_mat_path": np.asarray(u_paths.get("u_dn_mat", ""), dtype=object),
    }


def plot_bands(payload, args):
    bands = payload["bands"]
    kdist = payload["kdist"]
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    for ib in range(bands.shape[1]):
        ax.plot(kdist, bands[:, ib], color=args.color, lw=args.linewidth, alpha=args.alpha)
    ax.set_xlim(float(kdist[0]), float(kdist[-1]))
    ax.set_xticks(payload["ticks"])
    ax.set_xticklabels(payload["tick_labels"])
    for t in payload["ticks"]:
        ax.axvline(float(t), color="0.75", lw=0.5, alpha=0.7)
    ax.axhline(0.0, color="0.35", lw=0.6, alpha=0.7)
    if args.ymin is not None or args.ymax is not None:
        ymin = float(np.nanmin(bands)) if args.ymin is None else float(args.ymin)
        ymax = float(np.nanmax(bands)) if args.ymax is None else float(args.ymax)
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel("k path")
    ylabel = r"$E - E_F$ (eV)" if str(payload.get("ref", "")).startswith("efermi") else r"$E - E_{\rm ref}$ (eV)"
    ax.set_ylabel(ylabel)
    if args.title:
        ax.set_title(args.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(
        f"[epr-soc-band] wrote {args.output} "
        f"bands_shape={bands.shape} emin={np.nanmin(bands):.6g} emax={np.nanmax(bands):.6g}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Plot model-SOC spinor bands from EPR up/down H(k)")
    ap.add_argument("--epr_up", required=True)
    ap.add_argument("--epr_dn", required=True)
    ap.add_argument("--hr_unit", choices=["ev", "ry", "ha"], default="ry")
    ap.add_argument("--efermi", type=float, default=0.0, help="Fermi energy in eV subtracted from bands")
    ap.add_argument("--ref", choices=["fermi", "vbm", "zero"], default="fermi",
                    help="Energy reference: fermi subtracts --efermi; vbm subtracts valence-band maximum; zero subtracts nothing")
    ap.add_argument("--nvalence", type=int, default=None,
                    help="Number of occupied spinor bands for --ref vbm")
    ap.add_argument("--soc", default="",
                    help="Model SOC specs inferred from .win projections, e.g. 'I:p:0.6;Cr:d:0.05'")
    ap.add_argument("--lambda_te", type=float, default=0.0, help="Compatibility onsite p SOC lambda in eV; prefer --soc")
    ap.add_argument("--soc_p_groups", default="", help="Semicolon-separated p orbital groups, e.g. '10,11,12;25,26,27'")
    ap.add_argument("--soc_p_groups_base", type=int, choices=[0, 1], default=0)
    ap.add_argument("--win", required=True, help="Wannier90 .win with projections and kpoint_path")
    ap.add_argument("--soc_gauge", choices=["wannier", "rotated"], default="wannier",
                    help="wannier adds L.S directly in Wannier index space; rotated adds U(k) L.S U(k)^dagger")
    ap.add_argument("--u_mat", default=None, help="Common Wannier90 *_u.mat, .npy, or .npz U(k) file for --soc_gauge rotated")
    ap.add_argument("--u_up_mat", default=None, help="Spin-up U(k) file for --soc_gauge rotated")
    ap.add_argument("--u_dn_mat", default=None, help="Spin-down U(k) file for --soc_gauge rotated")
    ap.add_argument("--soc_rotation", choices=["u_soc_udag", "udag_soc_u"], default="u_soc_udag",
                    help="Rotation convention for --soc_gauge rotated")
    ap.add_argument("--u_match_tol", type=float, default=1.0e-7,
                    help="Fractional k tolerance for matching U(k) to the plot k-path")
    ap.add_argument("--u_nearest", action="store_true",
                    help="Diagnostic only: use nearest U(k) if exact k-path matching fails")
    ap.add_argument("--soc_element", default="", help="Element for compatibility --lambda_te p-SOC mode")
    ap.add_argument("--p_order", default=WANNIER90_P_ORDER, help="p orbital order inside each group")
    ap.add_argument("--d_order", default=WANNIER90_D_ORDER, help="d orbital order inside each SOC group")
    ap.add_argument("--spin_direction", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    ap.add_argument("--band_points", type=int, default=80, help="Points per k-path segment")
    ap.add_argument("-o", "--output", default="epr_soc_bands.png")
    ap.add_argument("--save_npz", default=None)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--fig_width", type=float, default=7.0)
    ap.add_argument("--fig_height", type=float, default=4.5)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--color", default="black")
    ap.add_argument("--linewidth", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    payload = build_soc_bands(args)
    if args.save_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)) or ".", exist_ok=True)
        np.savez_compressed(
            args.save_npz,
            bands=payload["bands"],
            raw_bands=payload["raw_bands"],
            kdist=payload["kdist"],
            kpts_frac=payload["kpts_frac"],
            ticks=payload["ticks"],
            tick_labels=np.asarray(payload["tick_labels"], dtype=object),
            rvec_ang=payload["rvec_ang"],
            gvec_ang=payload["gvec_ang"],
            spin_direction=payload["spin_direction"],
            soc_p_groups=payload["soc_p_groups"],
            soc_d_groups=payload["soc_d_groups"],
            soc_entries=payload["soc_entries"],
            win_path=payload["win_path"],
            soc_gauge=payload["soc_gauge"],
            soc_rotation=payload["soc_rotation"],
            u_mat_path=payload["u_mat_path"],
            u_up_mat_path=payload["u_up_mat_path"],
            u_dn_mat_path=payload["u_dn_mat_path"],
            eref=payload["eref"],
            ref=payload["ref"],
            lambda_te=np.array(float(args.lambda_te)),
            soc=np.array(args.soc, dtype=object),
            efermi=np.array(float(args.efermi)),
            nvalence=np.array(-1 if args.nvalence is None else int(args.nvalence), dtype=np.int64),
            p_order=np.array(args.p_order, dtype=object),
            d_order=np.array(args.d_order, dtype=object),
        )
        print(f"[epr-soc-band] wrote {args.save_npz}", flush=True)
    plot_bands(payload, args)


if __name__ == "__main__":
    main()
