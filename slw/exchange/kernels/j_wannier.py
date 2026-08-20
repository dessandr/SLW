"""Native scalar and tensor exchange kernel for Wannier90 Hamiltonians.

This is a thin input wrapper around the validated EPR tensor kernels in
``compute_J_epr_tensor``.  It avoids ``LKAGSolver.compute_J_tensor_bulk`` and
uses the same spinor-H(k), Green-function, and tensor accumulation code path as
the EPR implementation.
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np

from slw.core.structure import (
    atom_names_from_structure,
    find_nearest_neighbours,
    resolve_wannier_structure,
)
from slw.core.wannier_io import read_wannier_hr
from slw.exchange.kernels.eph_provider import _unit_scale_to_ev
from slw.exchange.kernels.epr import _build_hk_from_epr, _full_k_mesh, _load_slices
from slw.exchange.kernels.j_epr import (
    _compute_j_direct,
    _find_nearest_neighbours_from_epr,
)
from slw.exchange.kernels.j_tensor_epr import (
    _apply_model_soc,
    _compute_tensor_direct,
    _compute_tensor_tb2j,
    _parse_axes,
    _write_tensor_text,
)
from slw.exchange.kernels.lkag import (
    get_cfr_ozaki_mesh,
    get_cfr_pole_mesh,
    get_semicircle_contour,
)
from slw.exchange.kernels.parallel import collective_sum, partition_sequence, rank_size
from slw.exchange.kernels.soc_downfold import (
    _downfold_one_k,
    _group_indices,
    _selected_groups,
    _soc_matrix_for_groups,
)
from slw.exchange.kernels.spinor import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    spinor_from_collinear,
)
from slw.exchange.kernels.spinor_hr import _basis_permutation_from_win
from slw.exchange.kernels.win_soc import _win_projection_groups


def _read_wannier_hr_compat(path):
    parsed = read_wannier_hr(path)
    if len(parsed) == 4:
        dim, _nrpts, degens, hmap = parsed
        return int(dim), list(degens), hmap
    if len(parsed) == 3:
        dim, degens, hmap = parsed
        return int(dim), list(degens), hmap
    raise ValueError(f"Unexpected read_wannier_hr return length={len(parsed)} for {path}")


def _build_hk_from_hr_map(hmap, kpts, *, apply_degeneracy=True, degens=None, unit_scale=1.0):
    keys = list(hmap.keys())
    next(iter(hmap.values())).shape[0]
    rvec = np.asarray(keys, dtype=np.float64)
    blocks = np.asarray([hmap[r] for r in keys], dtype=np.complex128)
    if apply_degeneracy:
        if degens is None or len(degens) != len(keys):
            raise ValueError("--apply_degeneracy requires a degeneracy for every R point")
        blocks = blocks / np.asarray(degens, dtype=np.float64)[:, None, None]
    blocks = blocks * float(unit_scale)
    phase = np.exp(2.0j * np.pi * (np.asarray(kpts, dtype=np.float64) @ rvec.T))
    hk = np.einsum("kr,rij->kij", phase, blocks, optimize=True)
    return 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))


def _load_hr_hk(up_hr, dn_hr, kpts, *, apply_degeneracy=True, hr_unit="ev"):
    dim_up, deg_up, h_up = _read_wannier_hr_compat(up_hr)
    dim_dn, deg_dn, h_dn = _read_wannier_hr_compat(dn_hr)
    if dim_up != dim_dn:
        raise ValueError(f"hr.dat dimension mismatch: up={dim_up}, dn={dim_dn}")
    if set(h_up) != set(h_dn):
        missing_dn = sorted(set(h_up) - set(h_dn))
        missing_up = sorted(set(h_dn) - set(h_up))
        raise ValueError(f"R-vector mismatch: missing_dn={missing_dn[:5]} missing_up={missing_up[:5]}")
    scale = _unit_scale_to_ev(hr_unit)
    return (
        _build_hk_from_hr_map(h_up, kpts, apply_degeneracy=apply_degeneracy, degens=deg_up, unit_scale=scale),
        _build_hk_from_hr_map(h_dn, kpts, apply_degeneracy=apply_degeneracy, degens=deg_dn, unit_scale=scale),
    )



def _read_centres_xyz_count(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    try:
        return int(first.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid Wannier centres xyz header in {path!r}: {first!r}") from exc


def _read_centres_xyz_coords(path, n):
    coords = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[2:]
    for raw in lines:
        toks = raw.split()
        if len(toks) < 4:
            continue
        try:
            xyz = [float(toks[1]), float(toks[2]), float(toks[3])]
        except ValueError:
            continue
        coords.append(xyz)
        if len(coords) >= int(n):
            break
    if len(coords) < int(n):
        raise ValueError(f"{path}: found {len(coords)} coordinate rows, expected at least {n}")
    return np.asarray(coords, dtype=np.float64)


def _parse_collinear_label_bounds(labels):
    out = []
    for label in labels:
        text = str(label)
        if "[" not in text or ":" not in text or not text.endswith("]"):
            continue
        head, rest = text.rsplit("[", 1)
        try:
            lo_s, hi_s = rest[:-1].split(":", 1)
            lo = int(lo_s)
            hi = int(hi_s)
        except ValueError:
            continue
        out.append((head, lo, hi))
    return out


def _centres_pair_rms(coords_a, coords_b):
    diff = np.asarray(coords_a, dtype=np.float64) - np.asarray(coords_b, dtype=np.float64)
    if diff.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def _infer_spinor_order_from_centres(centres_path, spinor_dim, labels):
    dim = int(spinor_dim)
    nwan = dim // 2
    coords = _read_centres_xyz_coords(centres_path, dim)
    scores = {}
    scores["legacy_sector_spin"] = _centres_pair_rms(coords[:nwan], coords[nwan:])
    scores["wannier_orbital"] = _centres_pair_rms(coords[0::2], coords[1::2])
    parsed = _parse_collinear_label_bounds(labels)
    group_diffs = []
    offset = 0
    for _head, lo, hi in parsed:
        n = hi - lo
        if offset + 2 * n > dim:
            group_diffs = []
            break
        group_diffs.append(coords[offset:offset + n] - coords[offset + n:offset + 2 * n])
        offset += 2 * n
    if group_diffs and offset == dim:
        diff = np.concatenate(group_diffs, axis=0)
        scores["wannier_spin"] = float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))
    else:
        scores["wannier_spin"] = float("inf")
    best = min(scores, key=scores.get)
    return best, scores


def _win_collinear_group_labels(win_path, nwan):
    if not win_path:
        return []
    _atoms, groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(f"{win_path} projection count={nproj} but spinor hr.dat half-dim={nwan}")
    return [
        f"{g['atom_label']}:{g['orbital']}[{g['indices'][0]}:{g['indices'][-1] + 1}]"
        for g in groups
    ]


def _file_order_group_labels(labels, nwan, order):
    mode = str(order).strip().lower()
    parsed = []
    for label in labels:
        text = str(label)
        if "[" not in text or ":" not in text or not text.endswith("]"):
            continue
        head, rest = text.rsplit("[", 1)
        try:
            lo_s, hi_s = rest[:-1].split(":", 1)
            lo = int(lo_s)
            hi = int(hi_s)
        except ValueError:
            continue
        parsed.append((head, lo, hi))
    if not parsed:
        return []
    out = []
    if mode in {"spin_major", "sector_spin"}:
        for head, lo, hi in parsed:
            out.append(f"{head}:up file[{lo}:{hi}]")
        for head, lo, hi in parsed:
            out.append(f"{head}:down file[{int(nwan) + lo}:{int(nwan) + hi}]")
    elif mode in {"wannier_spin", "projection_spinor"}:
        offset = 0
        for head, lo, hi in parsed:
            n = hi - lo
            out.append(f"{head} file[up={offset}:{offset + n},dn={offset + n}:{offset + 2 * n}]")
            offset += 2 * n
    elif mode in {"wannier_orbital", "orbital_spinor"}:
        for head, lo, hi in parsed:
            out.append(f"{head} file[orbital_up_down_pairs={2 * lo}:{2 * hi}]")
    return out


def _spinor_canonical_index_map(spinor_dim, *, basis_order="spin_major", win=None):
    dim = int(spinor_dim)
    if dim % 2:
        raise ValueError(f"Spinor hr.dat dimension must be even, got {dim}")
    nwan = dim // 2
    order = str(basis_order).strip().lower().replace("-", "_")
    if order in {"spin_major", "sector_spin", "spin", "up_down", "wannier90_spin_major"}:
        labels = _win_collinear_group_labels(win, nwan)
        return np.arange(dim, dtype=np.int64), "spin_major", labels, _file_order_group_labels(labels, nwan, "spin_major")
    if order in {"wannier_orbital", "orbital_spinor", "orbital_projection", "projection_orbital", "orbital_interleaved", "up_down_interleaved", "wannier90_orbital_spinor"}:
        labels = _win_collinear_group_labels(win, nwan)
        perm_file_from_canonical = np.empty(dim, dtype=np.int64)
        perm_file_from_canonical[0::2] = np.arange(nwan, dtype=np.int64)
        perm_file_from_canonical[1::2] = np.arange(nwan, dtype=np.int64) + nwan
        inv = np.empty_like(perm_file_from_canonical)
        inv[perm_file_from_canonical] = np.arange(dim, dtype=np.int64)
        return inv, "wannier_orbital", labels, _file_order_group_labels(labels, nwan, "wannier_orbital")
    if order in {"wannier_spin", "projection_spinor", "group_spinor", "wannier90_spinor", "w90_spinor", "win_interleaved"}:
        perm_file_from_canonical, _resolved, labels = _basis_permutation_from_win(win, nwan, "win_interleaved")
        inv = np.empty_like(perm_file_from_canonical)
        inv[perm_file_from_canonical] = np.arange(dim, dtype=np.int64)
        return inv, "wannier_spin", labels, _file_order_group_labels(labels, nwan, "wannier_spin")
    perm_file_from_canonical, resolved, labels = _basis_permutation_from_win(win, nwan, order)
    inv = np.empty_like(perm_file_from_canonical)
    inv[perm_file_from_canonical] = np.arange(dim, dtype=np.int64)
    return inv, resolved, labels, _file_order_group_labels(labels, nwan, resolved)




def _spinor_label_from_collinear_label(label, nwan):
    text = str(label)
    if "[" not in text or ":" not in text or not text.endswith("]"):
        return text
    head, rest = text.rsplit("[", 1)
    bounds = rest[:-1].split(":", 1)
    if len(bounds) != 2:
        return text
    try:
        lo = int(bounds[0])
        hi = int(bounds[1])
    except ValueError:
        return text
    return f"{head}[up={lo}:{hi},dn={int(nwan) + lo}:{int(nwan) + hi}]"


def _spinor_labels_from_collinear(labels, nwan):
    return [_spinor_label_from_collinear_label(label, nwan) for label in labels]


def _spinor_slice_summary(slices, nwan):
    out = {}
    for site, slc in slices.items():
        up = (int(slc.start), int(slc.stop))
        dn = (int(nwan) + int(slc.start), int(nwan) + int(slc.stop))
        out[int(site)] = {"up": up, "dn": dn}
    return out


def _slice_file_order_summary(slices, nwan, basis_order):
    mode = str(basis_order).strip().lower().replace("-", "_")
    if mode in {"wannier_spin", "projection_spinor", "group_spinor", "wannier90_spinor", "w90_spinor", "win_interleaved", "site_interleaved", "atom_interleaved", "group_interleaved"}:
        out = {}
        for site, slc in slices.items():
            n = int(slc.stop) - int(slc.start)
            start = 2 * int(slc.start)
            out[int(site)] = {"up": (start, start + n), "dn": (start + n, start + 2 * n)}
        return out
    if mode in {"wannier_orbital", "orbital_spinor", "orbital_projection", "projection_orbital", "orbital_interleaved", "up_down_interleaved", "wannier90_orbital_spinor"}:
        return {
            int(site): {"orbital_up_down_pairs_file": (2 * int(slc.start), 2 * int(slc.stop))}
            for site, slc in slices.items()
        }
    if mode in {"spin_major", "sector_spin", "spin", "up_down", "wannier90_spin_major"}:
        out = {}
        for site, slc in slices.items():
            out[int(site)] = {
                "up_file": (int(slc.start), int(slc.stop)),
                "down_file": (int(nwan) + int(slc.start), int(nwan) + int(slc.stop)),
            }
        return out
    return _spinor_slice_summary(slices, nwan)

def _load_spinor_hr_hk(
    spinor_hr,
    kpts,
    *,
    apply_degeneracy=True,
    hr_unit="ev",
    basis_order="spin_major",
    win=None,
    centres=None,
):
    dim, degens, hmap = _read_wannier_hr_compat(spinor_hr)
    if int(dim) % 2:
        raise ValueError(f"Spinor hr.dat dimension must be even, got {dim}")
    if centres:
        ncentres = _read_centres_xyz_count(centres)
        if ncentres < int(dim):
            raise ValueError(f"{centres}: centres count={ncentres} is smaller than spinor_dim={dim}")
        if ncentres != int(dim):
            print(
                f"[J-wannier-tensor] centres count={ncentres}; using first spinor_dim={dim} Wannier centre rows",
                flush=True,
            )
    hk = _build_hk_from_hr_map(
        hmap,
        kpts,
        apply_degeneracy=apply_degeneracy,
        degens=degens,
        unit_scale=_unit_scale_to_ev(hr_unit),
    )
    canonical_idx, resolved_order, labels, file_labels = _spinor_canonical_index_map(dim, basis_order=basis_order, win=win)
    centres_order_guess = ""
    centres_order_scores = {}
    if centres and labels:
        centres_order_guess, centres_order_scores = _infer_spinor_order_from_centres(centres, dim, labels)
    if not np.array_equal(canonical_idx, np.arange(int(dim), dtype=np.int64)):
        hk = hk[:, canonical_idx[:, None], canonical_idx]
    meta = {
        "spinor_dim": int(dim),
        "nwan": int(dim) // 2,
        "basis_order": resolved_order,
        "basis_groups_internal": _spinor_labels_from_collinear(labels, int(dim) // 2),
        "basis_groups_file": file_labels,
        "basis_groups_collinear_half": labels,
        "centres": centres or "",
        "centres_order_guess": centres_order_guess,
        "centres_order_scores": centres_order_scores,
    }
    return hk, meta


def _parse_subspace_selector(text):
    specs = []
    for item in str(text or "").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.replace(",", ":").split(":") if x.strip()]
        if len(parts) != 2:
            raise ValueError(f"Expected subspace selector element:orbital, got {item!r}")
        specs.append((parts[0].capitalize(), parts[1].lower()))
    return specs


def _infer_collinear_slices_from_win(win_path, nwan, selector):
    specs = _parse_subspace_selector(selector)
    if not specs:
        return None, []
    if not win_path:
        raise ValueError("--mag_subspace requires --win")
    _atoms, groups, nproj = _win_projection_groups(win_path)
    if int(nproj) != int(nwan):
        raise ValueError(f"{win_path} projection count={nproj} but spinor hr.dat half-dim={nwan}")
    matched = []
    for elem, orb in specs:
        found = [g for g in groups if str(g["element"]).capitalize() == elem and str(g["orbital"]).lower() == orb]
        if not found:
            known = sorted({(g["element"], g["orbital"]) for g in groups})
            raise ValueError(f"No {elem}:{orb} groups found in {win_path}; known={known}")
        matched.extend(found)
    slices = {}
    labels = []
    for isite, group in enumerate(matched):
        idx = np.asarray(group["indices"], dtype=np.int64).reshape(-1)
        lo = int(idx.min())
        hi = int(idx.max()) + 1
        if not np.array_equal(idx, np.arange(lo, hi, dtype=np.int64)):
            raise ValueError(f"{group['atom_label']}:{group['orbital']} indices are not contiguous: {idx.tolist()}")
        slices[int(isite)] = slice(lo, hi)
        labels.append(f"{group['atom_label']}:{group['orbital']}[up={lo}:{hi},dn={int(nwan) + lo}:{int(nwan) + hi}]")
    return slices, labels

def _print_hk_diagnostics(tag, hk_up, hk_dn):
    eval_up = np.linalg.eigvalsh(hk_up)
    eval_dn = np.linalg.eigvalsh(hk_dn)
    split = np.linalg.norm((hk_up - hk_dn).reshape(hk_up.shape[0], -1), axis=1)
    print(
        f"[J-wannier-tensor] {tag} "
        f"up_band=({float(eval_up.min()):.6g},{float(eval_up.max()):.6g}) eV "
        f"dn_band=({float(eval_dn.min()):.6g},{float(eval_dn.max()):.6g}) eV "
        f"|Hup-Hdn|_F_mean={float(split.mean()):.6g} eV",
        flush=True,
    )


def _compare_reference_hk(args, kpts, hk_up, hk_dn):
    if not args.ref_epr_up and not args.ref_epr_dn:
        return
    if not args.ref_epr_up or not args.ref_epr_dn:
        raise ValueError("--ref_epr_up and --ref_epr_dn must be provided together")
    ref_up = _build_hk_from_epr(args.ref_epr_up, kpts, unit=args.ref_hr_unit)
    ref_dn = _build_hk_from_epr(args.ref_epr_dn, kpts, unit=args.ref_hr_unit)
    if ref_up.shape != hk_up.shape or ref_dn.shape != hk_dn.shape:
        raise ValueError(
            f"reference H(k) shape mismatch: wannier={hk_up.shape}/{hk_dn.shape} "
            f"ref={ref_up.shape}/{ref_dn.shape}"
        )
    _print_hk_diagnostics("ref EPR H(k)", ref_up, ref_dn)
    for name, got, ref in (("up", hk_up, ref_up), ("dn", hk_dn, ref_dn)):
        diff = got - ref
        denom = max(float(np.linalg.norm(ref.reshape(ref.shape[0], -1), axis=1).mean()), 1.0e-30)
        rel = float(np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1).mean()) / denom
        print(
            f"[J-wannier-tensor] compare_ref {name}: "
            f"max_abs_diff={float(np.max(np.abs(diff))):.6g} eV "
            f"mean_fro_rel={rel:.6g}",
            flush=True,
        )
    got_split = np.linalg.norm((hk_up - hk_dn).reshape(hk_up.shape[0], -1), axis=1).mean()
    ref_split = np.linalg.norm((ref_up - ref_dn).reshape(ref_up.shape[0], -1), axis=1).mean()
    ratio = float(got_split / ref_split) if ref_split != 0 else float("nan")
    print(f"[J-wannier-tensor] compare_ref split_ratio_wannier_over_ref={ratio:.6g}", flush=True)


def _embed_spinor_correction(h_spin, delta_k, d_idx, nwan):
    nd = len(d_idx)
    up = np.asarray(d_idx, dtype=np.int64)
    dn = up + int(nwan)
    out = np.array(h_spin, dtype=np.complex128, copy=True)
    out[:, up[:, None], up] += delta_k[:, :nd, :nd]
    out[:, up[:, None], dn] += delta_k[:, :nd, nd:]
    out[:, dn[:, None], up] += delta_k[:, nd:, :nd]
    out[:, dn[:, None], dn] += delta_k[:, nd:, nd:]
    return out


def _apply_intersite_soc(h_spin, hk_up, hk_dn, args):
    if not args.intersite_soc:
        return h_spin, {}
    if not args.win:
        raise ValueError("--intersite_soc requires --win for subspace selection")
    if args.lambda_soc is None:
        raise ValueError("--intersite_soc requires --lambda_soc")
    if not str(args.d_subspace).strip() or not str(args.soc_active).strip():
        raise ValueError("--intersite_soc requires --d_subspace and --soc_active")
    nwan = int(hk_up.shape[-1])
    d_groups = _selected_groups(args.win, args.d_subspace, nwan)
    p_groups = _selected_groups(args.win, args.soc_active, nwan)
    d_idx, d_summary = _group_indices(d_groups)
    p_idx, p_summary = _group_indices(p_groups)
    if len(set(d_idx.tolist())) != int(d_idx.size):
        raise ValueError(f"Duplicate d-subspace orbital indices: {d_idx.tolist()}")
    if len(set(p_idx.tolist())) != int(p_idx.size):
        raise ValueError(f"Duplicate SOC-active orbital indices: {p_idx.tolist()}")

    hsoc, soc_summary = _soc_matrix_for_groups(
        p_groups,
        nwan,
        float(args.lambda_soc),
        p_order=args.p_order,
        d_order=args.d_order,
    )
    hsoc_p = hsoc[np.ix_(np.concatenate([p_idx, p_idx + nwan]), np.concatenate([p_idx, p_idx + nwan]))]
    np_ = int(p_idx.size)
    lambda_uu = hsoc_p[:np_, :np_]
    lambda_ud = hsoc_p[:np_, np_:]
    lambda_du = hsoc_p[np_:, :np_]
    lambda_dd = hsoc_p[np_:, np_:]

    p_eval_up = np.linalg.eigvalsh(hk_up[:, p_idx[:, None], p_idx])
    p_eval_dn = np.linalg.eigvalsh(hk_dn[:, p_idx[:, None], p_idx])
    min_den_up = float(np.min(np.abs(float(args.e0) - p_eval_up)))
    min_den_dn = float(np.min(np.abs(float(args.e0) - p_eval_dn)))
    print(
        "[J-wannier-tensor] intersite SOC denominator "
        f"p_up_range=({float(p_eval_up.min()):.6g},{float(p_eval_up.max()):.6g}) eV "
        f"p_dn_range=({float(p_eval_dn.min()):.6g},{float(p_eval_dn.max()):.6g}) eV "
        f"min|E0-Hp|=({min_den_up:.6g},{min_den_dn:.6g}) eV",
        flush=True,
    )

    delta_k = np.empty((hk_up.shape[0], 2 * d_idx.size, 2 * d_idx.size), dtype=np.complex128)
    for ik in range(hk_up.shape[0]):
        delta_k[ik] = _downfold_one_k(
            hk_up[ik],
            hk_dn[ik],
            d_idx,
            p_idx,
            lambda_uu,
            lambda_ud,
            lambda_du,
            lambda_dd,
            args.e0,
            args.eta,
        )
    if args.hermitianize_soc:
        delta_k = 0.5 * (delta_k + np.swapaxes(delta_k.conj(), 1, 2))
    norms = np.linalg.norm(delta_k.reshape(delta_k.shape[0], -1), axis=1)
    print(
        "[J-wannier-tensor] added intersite/downfolded SOC "
        f"d={d_summary} p={p_summary} lambda={float(args.lambda_soc):g} "
        f"E0={float(args.e0):g} eta={float(args.eta):g} "
        f"norm_min_mean_max=({float(norms.min()):.6g},{float(norms.mean()):.6g},{float(norms.max()):.6g})",
        flush=True,
    )
    meta = {
        "d_subspace": str(args.d_subspace),
        "soc_active": str(args.soc_active),
        "lambda_soc": float(args.lambda_soc),
        "e0": float(args.e0),
        "eta": float(args.eta),
        "d_summary": d_summary,
        "soc_summary": soc_summary,
    }
    return _embed_spinor_correction(h_spin, delta_k, d_idx, nwan), meta


def _normalize_mag_atoms(args):
    return [int(x) - 1 if int(args.mag_atoms_base) == 1 else int(x) for x in args.mag_atoms]


def _build_pair_meta(mag_atoms, neighbours):
    global_to_local = {int(g): i for i, g in enumerate(mag_atoms)}
    out = []
    for n in neighbours:
        gi, gj = int(n["i"]), int(n["j"])
        if gi not in global_to_local or gj not in global_to_local:
            continue
        out.append(
            {
                "gi": gi,
                "gj": gj,
                "li": int(global_to_local[gi]),
                "lj": int(global_to_local[gj]),
                "R": tuple(int(x) for x in n["R"]),
                "dist": float(n["distance"]),
                "shell": int(n.get("shell_idx", 0)),
            }
        )
    if not out:
        raise RuntimeError("No directed bonds matched --mag_atoms")
    return out


def _compare_reference_bonds(args, mag_atoms, pair_meta):
    if not args.ref_epr_up:
        return
    ref_neigh = _find_nearest_neighbours_from_epr(
        args.ref_epr_up,
        mag_atom_indices=mag_atoms,
        n_shells=int(args.n_shells),
        d_max=float(args.d_max),
        all_bonds=bool(args.all_bonds),
    )
    ref_pair_meta = _build_pair_meta(mag_atoms, ref_neigh)
    got = {(int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"])) for m in pair_meta}
    ref = {(int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"])) for m in ref_pair_meta}
    missing = sorted(ref - got)
    extra = sorted(got - ref)
    dist_got = {k: float(m["dist"]) for m in pair_meta for k in [(int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"]))]}
    dist_ref = {k: float(m["dist"]) for m in ref_pair_meta for k in [(int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"]))]}
    common = sorted(got & ref)
    max_dd = max((abs(dist_got[k] - dist_ref[k]) for k in common), default=0.0)
    print(
        "[J-wannier-tensor] compare_ref bonds "
        f"n_wannier={len(pair_meta)} n_ref={len(ref_pair_meta)} n_common={len(common)} "
        f"missing_ref={len(missing)} extra_wannier={len(extra)} max_dist_diff_A={max_dd:.6g}",
        flush=True,
    )
    if missing[:5]:
        print(f"[J-wannier-tensor] compare_ref bonds missing_ref_sample={missing[:5]}", flush=True)
    if extra[:5]:
        print(f"[J-wannier-tensor] compare_ref bonds extra_wannier_sample={extra[:5]}", flush=True)


def _group_orbits(neighbours, mode):
    mode = str(mode).strip().lower()
    if mode == "none":
        return [[n] for n in neighbours]
    groups = {}
    for n in neighbours:
        if mode == "shell":
            key = (int(n.get("shell_idx", 0)), round(float(n["distance"]), 4))
        elif mode == "distance":
            pair = tuple(sorted((int(n["i"]), int(n["j"]))))
            key = (int(n.get("shell_idx", 0)), round(float(n["distance"]), 4), pair)
        else:
            raise ValueError(f"Unsupported --orbit_grouping={mode}; use none|distance|shell")
        groups.setdefault(key, []).append(n)
    return list(groups.values())


def _write_tensor_h5_simple(path, args, labels, pair_meta, tensor, trace_acc, elapsed, nk, nE, axes, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        basic = h5.create_group("basic_data")
        basic.create_dataset("atom_labels", data=np.asarray(labels, dtype=object), dtype=str_dt)
        basic.create_dataset("kmesh", data=np.asarray(args.kmesh, dtype=np.int64))
        basic.create_dataset("efermi_ev", data=np.asarray(float(args.efermi)))
        basic.create_dataset("apply_degeneracy", data=np.asarray(bool(args.apply_degeneracy)))
        basic.create_dataset("spin_direction", data=np.asarray(args.spin_direction, dtype=np.float64))
        basic.create_dataset("tensor_axes", data=np.asarray(axes, dtype=object), dtype=str_dt)
        basic.create_dataset("nk", data=np.asarray(int(nk), dtype=np.int64))
        basic.create_dataset("nE", data=np.asarray(int(nE), dtype=np.int64))
        basic.create_dataset("elapsed_s", data=np.asarray(float(elapsed)))
        basic.create_dataset(
            "mpi_size",
            data=np.asarray(int(getattr(args, "_mpi_size", 1)), dtype=np.int64),
        )
        for key, val in {
            "kernel": args.kernel,
            "integrator": args.integrator,
            "hr_unit": args.hr_unit,
            "up_hr": args.up_hr or "",
            "dn_hr": args.dn_hr or "",
            "spinor_hr": args.spinor_hr or "",
            "spinor_basis_order": getattr(args, "spinor_basis_order", ""),
            "centres": getattr(args, "centres", None) or "",
            "mag_subspace": getattr(args, "mag_subspace", ""),
            "win": args.win or "",
            "soc": args.soc,
            "ref_epr_up": args.ref_epr_up or "",
            "ref_epr_dn": args.ref_epr_dn or "",
        }.items():
            basic.create_dataset(key, data=np.array(str(val), dtype=object), dtype=str_dt)
        spin_magnitude = float(getattr(args, "spin_magnitude", 1.0))
        kernel_family = "scalar_lkag" if args.kernel == "scalar" else args.kernel
        for key, val in {
            "kernel_family": kernel_family,
            "hamiltonian_sign": "minus",
            "spin_normalization": (
                "unit_vector" if spin_magnitude == 1.0 else "spin_operator"
            ),
            "bond_coverage": (
                "directed_mate_complete" if bool(args.all_bonds) else "canonical_half"
            ),
            "realspace_gauge": "i_at_0_j_at_R",
        }.items():
            basic.create_dataset(key, data=np.array(str(val), dtype=object), dtype=str_dt)
        basic.create_dataset("source_spin_magnitude", data=np.asarray(spin_magnitude))
        directed_weight = 1.0 if args.kernel == "tb2j" else 0.5
        stored_bond_weight = directed_weight if bool(args.all_bonds) else 2.0 * directed_weight
        basic.create_dataset("directed_bond_weight", data=np.asarray(stored_bond_weight))
        bonds = h5.create_group("bonds")
        bonds.create_dataset("mag_i_atom", data=np.asarray([m["gi"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_j_atom", data=np.asarray([m["gj"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_i_local", data=np.asarray([m["li"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_j_local", data=np.asarray([m["lj"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("R", data=np.asarray([m["R"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("distance_ang", data=np.asarray([m["dist"] for m in pair_meta], dtype=np.float64))
        bonds.create_dataset("shell", data=np.asarray([m["shell"] for m in pair_meta], dtype=np.int64))
        ds = h5.create_dataset("J_tensor_r", data=np.asarray(tensor, dtype=np.float64))
        ds.attrs["shape"] = "(nBond,tensor_axis_a,tensor_axis_b)"
        ds.attrs["tensor_axis_order"] = ",".join(axes)
        ds.attrs["unit"] = "meV"
        h5.create_dataset("trace_acc", data=np.asarray(trace_acc, dtype=np.complex128))
        if extra:
            grp = h5.create_group("extra")
            for key, val in extra.items():
                grp.create_dataset(key, data=np.asarray(val))


def _scalar_j_to_tensor(j_mev, axes):
    """Embed scalar LKAG J into an isotropic tensor on the requested axes."""
    j_mev = np.asarray(j_mev, dtype=np.float64)
    tensor = np.zeros((j_mev.size, len(axes), len(axes)), dtype=np.float64)
    for ia, a in enumerate(axes):
        for ib, b in enumerate(axes):
            if a == b:
                tensor[:, ia, ib] = j_mev
    return tensor


def _scalar_trace_to_tensor(trace_acc, axes):
    trace_acc = np.asarray(trace_acc, dtype=np.complex128)
    tensor = np.zeros((trace_acc.size, len(axes), len(axes)), dtype=np.complex128)
    for ia, a in enumerate(axes):
        for ib, b in enumerate(axes):
            if a == b:
                tensor[:, ia, ib] = trace_acc
    return tensor


def _ensure_scalar_kernel_is_soc_free(args):
    active = []
    if str(args.soc).strip():
        active.append("--soc")
    if float(args.lambda_te) != 0.0:
        active.append("--lambda_te")
    if bool(args.intersite_soc):
        active.append("--intersite_soc")
    if active:
        raise ValueError(
            "--kernel scalar is the SOC-free collinear LKAG reference path; "
            f"remove {', '.join(active)} or use --kernel tb2j/direct for spinor-SOC tensor calculations"
        )


def run(args, comm=None):
    axes = _parse_axes(args.axes)
    kpts = _full_k_mesh(args.kmesh)
    spinor_input = bool(args.spinor_hr)
    if spinor_input and (args.up_hr or args.dn_hr):
        print("[J-wannier-tensor] --spinor_hr is set; ignoring --up_hr/--dn_hr", flush=True)
    if not spinor_input and (not args.up_hr or not args.dn_hr):
        raise ValueError("Provide either --spinor_hr or both --up_hr and --dn_hr")

    print(
        f"[J-wannier-tensor] building H(k): mode={'spinor_hr' if spinor_input else 'collinear_up_down'} "
        f"nk={len(kpts)} kmesh={tuple(args.kmesh)} hr_unit={args.hr_unit} "
        f"apply_degeneracy={bool(args.apply_degeneracy)}",
        flush=True,
    )

    hk_up, hk_dn = None, None
    soc_entries, soc_win_path = [], ""
    intersite_soc_meta = {}
    if spinor_input:
        active = []
        if str(args.soc).strip():
            active.append("--soc")
        if float(args.lambda_te) != 0.0:
            active.append("--lambda_te")
        if str(args.soc_p_groups).strip():
            active.append("--soc_p_groups")
        if bool(args.intersite_soc):
            active.append("--intersite_soc")
        if bool(getattr(args, "dynamic_soc", False)):
            active.append("--dynamic_soc")
        if active:
            raise ValueError(
                "--spinor_hr is already a full spinor Hamiltonian; remove "
                + ", ".join(active)
                + " to avoid adding model/downfolded SOC again"
            )
        if args.kernel == "scalar":
            raise ValueError("--kernel scalar requires collinear --up_hr/--dn_hr; use direct/tb2j for --spinor_hr")
        h_spin, spinor_meta = _load_spinor_hr_hk(
            args.spinor_hr,
            kpts,
            apply_degeneracy=bool(args.apply_degeneracy),
            hr_unit=args.hr_unit,
            basis_order=args.spinor_basis_order,
            win=args.win,
            centres=args.centres,
        )
        dim = int(spinor_meta["nwan"])
        intersite_soc_meta = dict(spinor_meta)
        spin_evals0 = np.linalg.eigvalsh(h_spin)
        print(
            f"[J-wannier-tensor] loaded spinor H(k) dim={h_spin.shape[1]} "
            f"canonical_half_dim={dim} basis_order={spinor_meta['basis_order']} "
            f"band=({float(spin_evals0.min()):.6g},{float(spin_evals0.max()):.6g}) eV",
            flush=True,
        )
        if spinor_meta.get("basis_groups_file"):
            print(f"[J-wannier-tensor] basis_groups_file_order={spinor_meta['basis_groups_file']}", flush=True)
        if spinor_meta.get("basis_groups_internal"):
            print(f"[J-wannier-tensor] basis_groups_internal_spin_major={spinor_meta['basis_groups_internal']}", flush=True)
        if spinor_meta.get("centres_order_guess"):
            scores = spinor_meta.get("centres_order_scores", {})
            score_txt = ", ".join(f"{k}={float(v):.3e}A" for k, v in sorted(scores.items()))
            print(
                f"[J-wannier-tensor] centres_order_guess={spinor_meta['centres_order_guess']} "
                f"scores({score_txt})",
                flush=True,
            )
            if str(spinor_meta.get("centres_order_guess")) != str(spinor_meta.get("basis_order")):
                print(
                    f"[J-wannier-tensor][WARN] --spinor_basis_order={spinor_meta.get('basis_order')} "
                    f"but centres suggest {spinor_meta.get('centres_order_guess')}; J can be badly wrong if this mapping is wrong.",
                    flush=True,
                )
    else:
        hk_up, hk_dn = _load_hr_hk(
            args.up_hr,
            args.dn_hr,
            kpts,
            apply_degeneracy=bool(args.apply_degeneracy),
            hr_unit=args.hr_unit,
        )
        _print_hk_diagnostics("H(k)", hk_up, hk_dn)
        _compare_reference_hk(args, kpts, hk_up, hk_dn)
        if hk_up.shape != hk_dn.shape:
            raise ValueError(f"up/down H(k) shape mismatch: {hk_up.shape} vs {hk_dn.shape}")
        dim = int(hk_up.shape[1])
        if args.kernel == "scalar":
            _ensure_scalar_kernel_is_soc_free(args)
        h_spin = spinor_from_collinear(hk_up, hk_dn, n=args.spin_direction)
        # Reuse the EPR SOC helper, but disable EPR-neighbor .win inference.
        # Wannier input must pass --win explicitly when --soc/--lambda_te needs groups.
        args.epr_up = ""
        h_spin, soc_entries, soc_win_path = _apply_model_soc(h_spin, args, dim)

    if spinor_input and not args.slices and args.mag_subspace:
        slices, slice_labels = _infer_collinear_slices_from_win(args.win, dim, args.mag_subspace)
        print(f"[J-wannier-tensor] mag_subspace={slice_labels}", flush=True)
    else:
        slices = _load_slices(args, dim)
    if spinor_input:
        print(
            f"[J-wannier-tensor] mag_subspace_file_order={_slice_file_order_summary(slices, dim, args.spinor_basis_order)}",
            flush=True,
        )
        print(
            f"[J-wannier-tensor] mag_subspace_internal_spin_major={_spinor_slice_summary(slices, dim)}",
            flush=True,
        )
    else:
        print(f"[J-wannier-tensor] slices={ {int(k): (v.start, v.stop) for k, v in slices.items()} }", flush=True)

    dynamic_soc_val = False if spinor_input else getattr(args, "dynamic_soc", False)
    d_idx, p_idx = None, None
    lambda_uu, lambda_ud, lambda_du, lambda_dd = None, None, None, None

    if dynamic_soc_val:
        if not args.win:
            raise ValueError("--dynamic_soc requires --win for subspace selection")
        if args.lambda_soc is None:
            raise ValueError("--dynamic_soc requires --lambda_soc")
        if not str(args.d_subspace).strip() or not str(args.soc_active).strip():
            raise ValueError("--dynamic_soc requires --d_subspace and --soc_active")
        d_groups = _selected_groups(args.win, args.d_subspace, dim)
        p_groups = _selected_groups(args.win, args.soc_active, dim)
        d_idx, d_summary = _group_indices(d_groups)
        p_idx, p_summary = _group_indices(p_groups)
        hsoc, soc_summary = _soc_matrix_for_groups(
            p_groups,
            dim,
            float(args.lambda_soc),
            p_order=args.p_order,
            d_order=args.d_order,
        )
        hsoc_p = hsoc[np.ix_(np.concatenate([p_idx, p_idx + dim]), np.concatenate([p_idx, p_idx + dim]))]
        np_ = int(p_idx.size)
        lambda_uu = hsoc_p[:np_, :np_]
        lambda_ud = hsoc_p[:np_, np_:]
        lambda_du = hsoc_p[np_:, :np_]
        lambda_dd = hsoc_p[np_:, np_:]
        intersite_soc_meta = {
            "d_subspace": str(args.d_subspace),
            "soc_active": str(args.soc_active),
            "lambda_soc": float(args.lambda_soc),
            "d_summary": d_summary,
            "soc_summary": soc_summary,
            "dynamic": True,
        }
        print(
            f"[J-wannier-tensor] Setup dynamic SOC downfolding: d={d_summary} p={p_summary} lambda={args.lambda_soc:g} eV",
            flush=True,
        )
    elif not spinor_input:
        h_spin, intersite_soc_meta = _apply_intersite_soc(h_spin, hk_up, hk_dn, args)

    spin_evals = np.linalg.eigvalsh(h_spin)
    print(
        f"[J-wannier-tensor] final spinor H(k) band=({float(spin_evals.min()):.6g},{float(spin_evals.max()):.6g}) eV",
        flush=True,
    )
    args._soc_entries = soc_entries
    args._soc_win_path = soc_win_path
    args._intersite_soc_meta = intersite_soc_meta

    structure_path, structure_source = resolve_wannier_structure(os.getcwd(), args.win)
    labels_map = atom_names_from_structure(structure_path)
    labels = [labels_map.get(i, f"Atom{i + 1}") for i in range(max(labels_map.keys(), default=-1) + 1)]
    mag_atoms = _normalize_mag_atoms(args)
    neighbours = find_nearest_neighbours(
        structure_path,
        mag_atom_indices=mag_atoms,
        n_shells=int(args.n_shells),
        d_max=float(args.d_max),
        all_bonds=bool(args.all_bonds),
    )
    if args.nn_only:
        neighbours = [n for n in neighbours if int(n.get("shell_idx", 0)) == 1]
    pair_meta = _build_pair_meta(mag_atoms, neighbours)
    _compare_reference_bonds(args, mag_atoms, pair_meta)
    orbits = _group_orbits(neighbours, args.orbit_grouping)

    if args.integrator == "contour":
        energy_mesh = get_semicircle_contour(emin=args.emin, emax=0.0, npoints=args.empoints)
    elif args.integrator == "cfr_ozaki":
        energy_mesh = get_cfr_ozaki_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)
    else:
        energy_mesh = get_cfr_pole_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)

    rank, size = rank_size(comm)
    local_energy_mesh = partition_sequence(energy_mesh, comm)
    print(
        f"[J-wannier-tensor] computing tensor: kernel={args.kernel} nBond={len(pair_meta)} "
        f"nE={len(energy_mesh)} local_nE={len(local_energy_mesh)} "
        f"axes={axes} mpi={size} structure={structure_source}",
        flush=True,
    )
    t0 = time.time()

    def integrate_local():
        if args.kernel == "scalar":
            _ensure_scalar_kernel_is_soc_free(args)
            j_local, trace_local, _kdata, _info = _compute_j_direct(
                hk_up,
                hk_dn,
                slices,
                pair_meta,
                kpts,
                local_energy_mesh,
                args.efermi,
                nproc=int(args.nproc) if size == 1 else 1,
            )
            return (
                _scalar_j_to_tensor(j_local, axes),
                _scalar_trace_to_tensor(trace_local, axes),
                None,
            )
        if args.kernel == "direct":
            tensor_local, trace_local, _info = _compute_tensor_direct(
                h_spin, slices, pair_meta, kpts, local_energy_mesh,
                args.efermi, axes,
                nproc=int(args.nproc) if size == 1 else 1,
                collinear_override=args.collinear_override,
                dynamic_soc=dynamic_soc_val,
                hk_up=hk_up,
                hk_dn=hk_dn,
                d_idx=d_idx,
                p_idx=p_idx,
                lambda_uu=lambda_uu,
                lambda_ud=lambda_ud,
                lambda_du=lambda_du,
                lambda_dd=lambda_dd,
            )
            return tensor_local, trace_local, None
        tensor_local, trace_local, extra_local, _info = _compute_tensor_tb2j(
            h_spin, slices, pair_meta, kpts, local_energy_mesh,
            args.efermi, axes,
            collinear_override=args.collinear_override,
            dynamic_soc=dynamic_soc_val,
            hk_up=hk_up,
            hk_dn=hk_dn,
            d_idx=d_idx,
            p_idx=p_idx,
            lambda_uu=lambda_uu,
            lambda_ud=lambda_ud,
            lambda_du=lambda_du,
            lambda_dd=lambda_dd,
        )
        return tensor_local, trace_local, extra_local

    reduced = collective_sum(comm, integrate_local)
    if rank != 0:
        return
    if reduced is None:  # pragma: no cover - defensive communicator guard
        raise RuntimeError("MPI root did not receive Wannier J reduction")
    tensor, trace_acc, extra = reduced
    exe_info = {
        "n_chunks": max(1, size),
        "nproc": size if size > 1 else int(args.nproc),
        "mpi_size": size,
    }
    args._mpi_size = size
    elapsed = time.time() - t0

    # Scale J by S^2
    spin_magnitude_val = getattr(args, "spin_magnitude", 1.0)
    if spin_magnitude_val != 1.0:
        s2 = float(spin_magnitude_val) ** 2
        tensor = tensor / s2
        if extra:
            for key in (
                "jiso_tb2j",
                "dmi_tb2j",
                "jani_tb2j",
                "J_iso_tensor_r",
                "J_gamma_r",
                "J_antisym_aab_r",
                "J_dmi_tensor_r",
                "J_aab_full_r",
            ):
                if key in extra:
                    extra[key] = extra[key] / s2

    os.makedirs(args.out_dir, exist_ok=True)
    out_txt = args.out_name if os.path.isabs(args.out_name) else os.path.join(args.out_dir, args.out_name)
    out_h5 = args.out_h5 if os.path.isabs(args.out_h5) else os.path.join(args.out_dir, args.out_h5)
    _write_tensor_text(out_txt, labels, pair_meta, tensor, orbits, axes, args.kernel, extra=extra)
    _write_tensor_h5_simple(out_h5, args, labels, pair_meta, tensor, trace_acc, elapsed, len(kpts), len(energy_mesh), axes, extra=extra)
    print(
        f"[J-wannier-tensor] done elapsed_s={elapsed:.2f} "
        f"n_chunks={exe_info.get('n_chunks', 1)} nproc={exe_info.get('nproc', 1)}",
        flush=True,
    )
    print(f"[J-wannier-tensor] wrote {out_txt}", flush=True)
    print(f"[J-wannier-tensor] wrote {out_h5}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Wannier90 hr.dat wrapper for the EPR spinor J tensor kernel")
    ap.add_argument("--up_hr", default=None, help="Spin-up Wannier90 hr.dat; required unless --spinor_hr is used")
    ap.add_argument("--dn_hr", default=None, help="Spin-down Wannier90 hr.dat; required unless --spinor_hr is used")
    ap.add_argument("--spinor_hr", default=None, help="Full spinor Wannier90 hr.dat from SOC/noncollinear Wannier90")
    ap.add_argument(
        "--spinor_basis_order",
        default="wannier_spin",
        metavar="{wannier_spin,wannier_orbital}",
        help=(
            "Spinor ordering in --spinor_hr. wannier_spin = each .win projection group as "
            "[group up block, group down block], e.g. Mn1 d up, Mn1 d down, Mn2 d up, ...; "
            "wannier_orbital = [orb1 up, orb1 down, orb2 up, orb2 down, ...]. "
            "Orbital/projection order always follows the .win projection order."
        ),
    )
    ap.add_argument("--centres", default=None, help="Optional Wannier90 centres.xyz for spinor_dim sanity check")
    ap.add_argument("--efermi", type=float, required=True, help="Fermi energy in eV")
    ap.add_argument("--hr_unit", choices=["ev", "ry", "ha"], default="ev", help="Unit of input hr.dat matrix elements; Wannier90 default is eV")
    ap.add_argument("--ref_epr_up", default=None, help="Optional reference EPR up HDF5 for H(k) scale/gauge diagnostics")
    ap.add_argument("--ref_epr_dn", default=None, help="Optional reference EPR down HDF5 for H(k) scale/gauge diagnostics")
    ap.add_argument("--ref_hr_unit", choices=["ev", "ry", "ha"], default="ry", help="Reference EPR hopping unit")
    ap.add_argument("--kmesh", type=int, nargs=3, required=True)
    ap.add_argument("--mag_atoms", type=int, nargs="+", required=True, help="Magnetic atom indices")
    ap.add_argument("--mag_atoms_base", type=int, default=0, choices=[0, 1])
    ap.add_argument("--slices", default="", help="Manual local orbital slices, e.g. '0:0:5,1:5:10'")
    ap.add_argument("--apply_degeneracy", action=argparse.BooleanOptionalAction, default=True, help="Divide HR blocks by Wannier90 degeneracy before H(k) construction; standard Wannier90 needs this")
    ap.add_argument(
        "--kernel",
        choices=["scalar", "direct", "tb2j"],
        default="tb2j",
        help="scalar is the SOC-free collinear LKAG reference; tb2j/direct use spinor tensor kernels",
    )
    ap.add_argument("--axes", default="xyz", help="Tensor axes to compute, subset of xyz")
    ap.add_argument("--spin_direction", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    ap.add_argument("--soc", default="", help="Model SOC specs inferred from --win, e.g. 'Te:p:0.5;Cr:d:0.05'")
    ap.add_argument("--win", default=None, help="Wannier90 .win file; inferred only when the working directory contains exactly one candidate")
    ap.add_argument("--mag_subspace", default="", help="Infer magnetic local slices from .win projections, e.g. 'Cr:d;Cr:d' or 'Cr:d'")
    ap.add_argument("--soc_element", default="", help="Element for compatibility --lambda_te p-SOC mode")
    ap.add_argument("--lambda_te", type=float, default=0.0)
    ap.add_argument("--soc_p_groups", default="")
    ap.add_argument("--soc_groups_base", type=int, default=0, choices=[0, 1])
    ap.add_argument("--soc_p_groups_base", dest="soc_groups_base", type=int, choices=[0, 1], help=argparse.SUPPRESS)
    ap.add_argument("--p_order", default=WANNIER90_P_ORDER)
    ap.add_argument("--d_order", default=WANNIER90_D_ORDER)
    ap.add_argument("--intersite_soc", action="store_true", help="Add downfolded intersite SOC correction to the selected d subspace")
    ap.add_argument("--d_subspace", default="", help="Target low-energy subspace for --intersite_soc")
    ap.add_argument("--soc_active", default="", help="SOC-active ligand subspace for --intersite_soc")
    ap.add_argument("--lambda_soc", type=float, default=None, help="Atomic SOC lambda in eV for --intersite_soc")
    ap.add_argument("--e0", type=float, default=0.0, help="Downfolding energy E0 in eV for --intersite_soc")
    ap.add_argument("--eta", type=float, default=0.0, help="Downfolding broadening in eV for --intersite_soc")
    ap.add_argument("--hermitianize_soc", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--n_shells", type=int, default=10)
    ap.add_argument("--d_max", type=float, default=20.0)
    ap.add_argument("--all_bonds", action="store_true", default=True, help="Keep directed bonds; default matches compute_J_epr_tensor")
    ap.add_argument("--canonical_bonds", dest="all_bonds", action="store_false", help="Fold equivalent directed bonds")
    ap.add_argument("--nn_only", action="store_true")
    ap.add_argument("--orbit_grouping", choices=["none", "distance", "shell"], default="distance")
    ap.add_argument("--integrator", choices=["contour", "cfr_ozaki", "cfr_pole"], default="contour")
    ap.add_argument("--emin", type=float, default=-25.0)
    ap.add_argument("--empoints", type=int, default=500)
    ap.add_argument("--cfr_beta", type=float, default=1000.0)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--collinear_override", action="store_true", help="Override J_zz with (J_xx+J_yy)/2 in collinear calculations")
    ap.add_argument("--spin_magnitude", type=float, default=1.0, help="Spin magnitude S to scale J by 1/S^2")
    ap.add_argument("--dynamic_soc", action="store_true", help="Perform dynamic ligand SOC downfolding inside the energy loop")
    ap.add_argument("--out_dir", default="J_wannier_tensor")
    ap.add_argument("--out_name", default="J_wannier_tensor.txt")
    ap.add_argument("--out_h5", default="J_wannier_tensor.h5")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
