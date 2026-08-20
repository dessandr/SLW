"""Native scalar exchange-derivative kernel in qe2pert EPR k,q space.

This path intentionally avoids materializing legacy g_real payloads.  It
pre-caches H(k), eigensystems, bond phases, k+q maps, and EPR g(k,q), then
assembles the dG terms in momentum space.  dDelta terms are scaffolded but not
included yet.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional runtime accelerator control
    threadpool_limits = None

from slw.core.constants import BOHR_TO_ANG
from slw.core.qe2pert_ws import init_rvec_images, set_wigner_seitz_cell
from slw.exchange.kernels.eph_provider import _read_meta, _unit_scale_to_ev
from slw.exchange.kernels.epr import (
    _find_nearest_neighbours_from_epr,
    _full_k_mesh,
    _load_slices,
)
from slw.exchange.kernels.j_epr import (
    _format_r,
    _group_orbits_epr,
    _normalize_mag_atoms,
    _split_chunks,
)
from slw.exchange.kernels.lkag import (
    get_cfr_ozaki_mesh,
    get_cfr_pole_mesh,
    get_semicircle_contour,
)
from slw.exchange.kernels.parallel import collective_sum, partition_sequence, rank_size

_WORKER_STATIC = None
_THREADPOOL_LIMITER = None


def _configure_threads(nthreads):
    """Limit BLAS/OpenMP libraries for this process when threadpoolctl exists."""
    global _THREADPOOL_LIMITER
    n = max(1, int(nthreads))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(n)
    if threadpool_limits is not None:
        if _THREADPOOL_LIMITER is not None:
            _THREADPOOL_LIMITER.__exit__(None, None, None)
        _THREADPOOL_LIMITER = threadpool_limits(limits=n)
        _THREADPOOL_LIMITER.__enter__()


def _axis_list(text):
    """Normalize compact or comma-separated Cartesian-axis input."""
    normalized = "".join(
        char
        for char in str(text).strip().lower()
        if char not in {",", " ", "\t"}
    )
    if not normalized:
        return ["x", "y", "z"]
    if any(char not in {"x", "y", "z"} for char in normalized):
        raise ValueError(f"Unsupported axes '{text}'")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate axes in '{text}'")
    return list(normalized)


def _labels_from_species_labels(species_labels, nat):
    if not species_labels:
        return None
    if len(species_labels) != int(nat):
        raise ValueError(f"--species_labels length={len(species_labels)} but nat={nat}")
    counts = {}
    labels = []
    for sp in species_labels:
        key = str(sp).strip()
        counts[key] = counts.get(key, 0) + 1
        labels.append(f"{key}{counts[key]}")
    return labels


def _atom_labels(epr_path, user_labels, species_labels=None):
    with h5py.File(epr_path, "r") as h5:
        nat = int(h5["basic_data/nat"][()])
    if user_labels:
        labels = [x.strip() for x in str(user_labels).split(",") if x.strip()]
        if len(labels) != nat:
            raise ValueError(f"--atom_labels length={len(labels)} but nat={nat}")
        return labels
    labels = _labels_from_species_labels(species_labels, nat)
    if labels is not None:
        return labels
    return [f"Atom{i + 1}" for i in range(nat)]


def _target_indices(text, labels):
    if not text or str(text).strip().lower() == "all":
        return list(range(len(labels)))
    out = []
    by_label = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        if t in by_label:
            out.extend(by_label[t])
        else:
            idx = int(t)
            if idx < 0 or idx >= len(labels):
                raise ValueError(f"Target index {idx} outside [0,{len(labels)-1}]")
            out.append(idx)
    return list(dict.fromkeys(out))


def _full_q_mesh(qmesh):
    return _full_k_mesh(qmesh)


def _rp_grid(qmesh):
    axes = [np.fft.fftfreq(int(n)) * int(n) for n in qmesh]
    return np.asarray([(int(i), int(j), int(k)) for i in axes[0] for j in axes[1] for k in axes[2]], dtype=np.int64)


def _kq_map(kmesh, qmesh):
    kmesh = np.asarray(kmesh, dtype=np.int64)
    qmesh = np.asarray(qmesh, dtype=np.int64)
    if np.any(kmesh % qmesh != 0):
        raise ValueError(f"kmesh={tuple(kmesh)} must be commensurate with qmesh={tuple(qmesh)}")
    kpts_idx = np.asarray(
        [(i, j, k) for i in range(kmesh[0]) for j in range(kmesh[1]) for k in range(kmesh[2])],
        dtype=np.int64,
    )
    qpts_idx = np.asarray(
        [(i, j, k) for i in range(qmesh[0]) for j in range(qmesh[1]) for k in range(qmesh[2])],
        dtype=np.int64,
    )
    scale = kmesh // qmesh
    shifts = qpts_idx * scale[None, :]
    kq = (kpts_idx[None, :, :] + shifts[:, None, :]) % kmesh[None, None, :]
    return (kq[:, :, 0] * (kmesh[1] * kmesh[2]) + kq[:, :, 1] * kmesh[2] + kq[:, :, 2]).astype(np.int64)


def _bond_target_q_phase(qpts, pair_meta):
    """Phase of the bond endpoint at cell R for EPR's atomic gauge.

    EPR/Wannier matrices use ``H_ij(R)=<i,0|H|j,R>`` and
    ``g(k,q)=<k+q|dH(q)|k>``.  Consequently, a derivative acting on the
    endpoint ``j@R`` carries ``exp(+i 2*pi*q.R)``.  The returned array is
    vectorized over the complete q mesh and directed-bond list.
    """
    q = np.asarray(qpts, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"qpts must have shape (nq,3), got {q.shape}")
    r = np.asarray([m["R"] for m in pair_meta], dtype=np.float64)
    if r.size == 0:
        return np.empty((len(q), 0), dtype=np.complex128)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"bond translations must have shape (nPair,3), got {r.shape}")
    return np.exp(2j * np.pi * (q @ r.T))


def _precompute_eigensystem(hk_up, hk_dn, slices, efermi):
    nk, dim, _ = hk_up.shape
    eye = np.eye(dim, dtype=np.complex128)
    evals = {"up": np.zeros((nk, dim)), "down": np.zeros((nk, dim))}
    coeffs = {"up": np.zeros((nk, dim, dim), dtype=np.complex128), "down": np.zeros((nk, dim, dim), dtype=np.complex128)}
    delta = {}
    signs = {}
    for spin, hk in (("up", hk_up), ("down", hk_dn)):
        for ik in range(nk):
            h = 0.5 * (hk[ik] + hk[ik].conj().T) - float(efermi) * eye
            w, c = np.linalg.eigh(h)
            evals[spin][ik] = np.real(w)
            coeffs[spin][ik] = c
    d0 = np.mean(hk_up, axis=0) - np.mean(hk_dn, axis=0)
    for site, slc in slices.items():
        sid = int(site)
        dloc = np.array(d0[slc, slc], dtype=np.complex128)
        delta[sid] = 0.5 * (dloc + dloc.conj().T)
        tr = float(np.real(np.trace(delta[sid])))
        signs[sid] = -1.0 if tr < 0.0 else 1.0
    return {"evals": evals, "coeffs": coeffs, "delta": delta, "signs": signs}


def _phase_cache_key(kind, vectors):
    vec = np.ascontiguousarray(np.asarray(vectors, dtype=np.int64))
    return (kind, vec.shape, vec.tobytes())


def _phase_table(points, vectors, cache, kind):
    """Return exp(+2*pi*i*k.R) with shape (n_points, n_vectors)."""
    vec = np.asarray(vectors, dtype=np.int64)
    key = _phase_cache_key(kind, vec)
    got = cache.get(key)
    if got is None:
        got = np.exp(2j * np.pi * (np.asarray(points, dtype=np.float64) @ vec.T))
        cache[key] = np.ascontiguousarray(got)
    return got


def _build_hk_epr_phase(epr_path, kpts, *, unit="ry", phase_cache=None):
    scale = _unit_scale_to_ev(unit)
    phase_cache = {} if phase_cache is None else phase_cache
    with h5py.File(epr_path, "r") as h5:
        meta = _read_meta(h5)
        images = init_rvec_images(meta.nk_grid, meta.at)
        nk = len(kpts)
        hk = np.zeros((nk, meta.nwan, meta.nwan), dtype=np.complex128)
        ws_cache = {}
        for jw in range(1, meta.nwan + 1):
            for iw in range(1, jw + 1):
                tri = jw * (jw - 1) // 2 + iw
                dr = f"electron_wannier/hopping_r{tri}"
                di = f"electron_wannier/hopping_i{tri}"
                if dr not in h5 or di not in h5:
                    continue
                hop = (np.asarray(h5[dr], dtype=np.float64) + 1j * np.asarray(h5[di], dtype=np.float64)) * scale
                key = (iw, jw)
                ws = ws_cache.get(key)
                if ws is None:
                    ws = set_wigner_seitz_cell(images, meta.at, meta.wc[iw - 1], meta.wc[jw - 1])
                    ws_cache[key] = ws
                if hop.shape[0] != ws.nr:
                    raise ValueError(f"H WS mismatch iw={iw} jw={jw}: h5={hop.shape[0]} ws={ws.nr}")
                phase_k = _phase_table(kpts, ws.vectors, phase_cache, "hk")
                vals = phase_k @ hop
                hk[:, iw - 1, jw - 1] += vals
                if iw != jw:
                    hk[:, jw - 1, iw - 1] += np.conjugate(vals)
        return 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2)), meta.nk_grid


def _build_gkq_one(epr_path, target_ia0, axis, kpts, qpts, *, unit="ry", rotation_mode="none", phase_cache=None):
    if rotation_mode != "none":
        raise NotImplementedError("rotation_mode scaffold exists, but U-rotation is not implemented yet.")
    ax_id = {"x": 0, "y": 1, "z": 2}[axis]
    scale = _unit_scale_to_ev(unit)
    phase_cache = {} if phase_cache is None else phase_cache
    with h5py.File(epr_path, "r") as h5:
        meta = _read_meta(h5)
        grp = h5["eph_matrix_wannier"]
        el_images = init_rvec_images(meta.nk_grid, meta.at)
        ph_images = init_rvec_images(meta.nq_grid, meta.at)
        nk = len(kpts)
        nq = len(qpts)
        gkq = np.zeros((nq, nk, meta.nwan, meta.nwan), dtype=np.complex128)
        ws_cache = {}
        ia = int(target_ia0) + 1
        for jw in range(1, meta.nwan + 1):
            for iw in range(1, meta.nwan + 1):
                dr = f"ep_hop_r_{ia}_{jw}_{iw}"
                di = f"ep_hop_i_{ia}_{jw}_{iw}"
                if dr not in grp or di not in grp:
                    continue
                arr = (np.asarray(grp[dr], dtype=np.float64) + 1j * np.asarray(grp[di], dtype=np.float64)).transpose(2, 1, 0)
                key = (jw, iw)
                got = ws_cache.get(key)
                if got is None:
                    iw0 = iw - 1
                    jw0 = jw - 1
                    ws_el = set_wigner_seitz_cell(el_images, meta.at, meta.wc[iw0], meta.wc[jw0])
                    ws_ph = set_wigner_seitz_cell(ph_images, meta.at, meta.wc[iw0], meta.tau[target_ia0])
                    got = (ws_el, ws_ph)
                    ws_cache[key] = got
                ws_el, ws_ph = got
                if arr.shape[1] != ws_el.nr or arr.shape[2] != ws_ph.nr:
                    raise ValueError(
                        f"WS mismatch ia={ia} jw={jw} iw={iw}: arr={arr.shape[1:]} ws={(ws_el.nr, ws_ph.nr)}"
                    )
                phase_k = _phase_table(kpts, ws_el.vectors, phase_cache, f"gk:{iw}:{jw}")
                phase_q = _phase_table(qpts, ws_ph.vectors, phase_cache, f"gq:{target_ia0}:{iw}")
                block = arr[ax_id] * scale
                # Pipeline equivalent:
                #   g(k,q) = sum_Re,Rp eph_hop(Re,Rp) exp(+ik.Re) exp(+iq.Rp)
                # Stored as (Nq,Nk,Nw,Nw) for q-loop locality in the assembler.
                gkq[:, :, iw - 1, jw - 1] += np.einsum(
                    "kr,rp,qp->qk", phase_k, block, phase_q, optimize=True
                )
        return gkq


def _build_gkq_onsite_one(epr_path, target_ia0, axis, kpts, qpts, *, unit="ry", rotation_mode="none", phase_cache=None):
    if rotation_mode != "none":
        raise NotImplementedError("rotation_mode scaffold exists, but U-rotation is not implemented yet.")
    ax_id = {"x": 0, "y": 1, "z": 2}[axis]
    scale = _unit_scale_to_ev(unit)
    phase_cache = {} if phase_cache is None else phase_cache
    with h5py.File(epr_path, "r") as h5:
        meta = _read_meta(h5)
        grp = h5["eph_matrix_wannier"]
        el_images = init_rvec_images(meta.nk_grid, meta.at)
        ph_images = init_rvec_images(meta.nq_grid, meta.at)
        nk = len(kpts)
        nq = len(qpts)
        gkq = np.zeros((nq, nk, meta.nwan, meta.nwan), dtype=np.complex128)
        ws_cache = {}
        ia = int(target_ia0) + 1
        for jw in range(1, meta.nwan + 1):
            for iw in range(1, meta.nwan + 1):
                dr = f"ep_hop_r_{ia}_{jw}_{iw}"
                di = f"ep_hop_i_{ia}_{jw}_{iw}"
                if dr not in grp or di not in grp:
                    continue
                arr = (np.asarray(grp[dr], dtype=np.float64) + 1j * np.asarray(grp[di], dtype=np.float64)).transpose(2, 1, 0)
                key = (jw, iw)
                got = ws_cache.get(key)
                if got is None:
                    iw0 = iw - 1
                    jw0 = jw - 1
                    ws_el = set_wigner_seitz_cell(el_images, meta.at, meta.wc[iw0], meta.wc[jw0])
                    ws_ph = set_wigner_seitz_cell(ph_images, meta.at, meta.wc[iw0], meta.tau[target_ia0])
                    got = (ws_el, ws_ph)
                    ws_cache[key] = got
                ws_el, ws_ph = got
                if arr.shape[1] != ws_el.nr or arr.shape[2] != ws_ph.nr:
                    raise ValueError(
                        f"WS mismatch ia={ia} jw={jw} iw={iw}: arr={arr.shape[1:]} ws={(ws_el.nr, ws_ph.nr)}"
                    )
                re_mask = np.all(np.asarray(ws_el.vectors, dtype=np.int64) == 0, axis=1)
                if not np.any(re_mask):
                    continue
                phase_q = _phase_table(qpts, ws_ph.vectors, phase_cache, f"gq:on:{target_ia0}:{iw}")
                block = np.sum(arr[ax_id][re_mask, :], axis=0) * scale
                vals_q = phase_q @ block
                gkq[:, :, iw - 1, jw - 1] += vals_q[:, None]
        return gkq


def _build_gk_rp_one(epr_path, target_ia0, axis, kpts, rp_idx, *, unit="ry", rotation_mode="none", phase_cache=None):
    if rotation_mode != "none":
        raise NotImplementedError("rotation_mode scaffold exists, but U-rotation is not implemented yet.")
    ax_id = {"x": 0, "y": 1, "z": 2}[axis]
    scale = _unit_scale_to_ev(unit)
    phase_cache = {} if phase_cache is None else phase_cache
    rp_idx = np.asarray(rp_idx, dtype=np.int64)
    with h5py.File(epr_path, "r") as h5:
        meta = _read_meta(h5)
        grp = h5["eph_matrix_wannier"]
        el_images = init_rvec_images(meta.nk_grid, meta.at)
        ph_images = init_rvec_images(meta.nq_grid, meta.at)
        nk = len(kpts)
        gk = np.zeros((1, nk, meta.nwan, meta.nwan), dtype=np.complex128)
        ws_cache = {}
        ia = int(target_ia0) + 1
        for jw in range(1, meta.nwan + 1):
            for iw in range(1, meta.nwan + 1):
                dr = f"ep_hop_r_{ia}_{jw}_{iw}"
                di = f"ep_hop_i_{ia}_{jw}_{iw}"
                if dr not in grp or di not in grp:
                    continue
                arr = (np.asarray(grp[dr], dtype=np.float64) + 1j * np.asarray(grp[di], dtype=np.float64)).transpose(2, 1, 0)
                key = (jw, iw)
                got = ws_cache.get(key)
                if got is None:
                    iw0 = iw - 1
                    jw0 = jw - 1
                    ws_el = set_wigner_seitz_cell(el_images, meta.at, meta.wc[iw0], meta.wc[jw0])
                    ws_ph = set_wigner_seitz_cell(ph_images, meta.at, meta.wc[iw0], meta.tau[target_ia0])
                    got = (ws_el, ws_ph)
                    ws_cache[key] = got
                ws_el, ws_ph = got
                if arr.shape[1] != ws_el.nr or arr.shape[2] != ws_ph.nr:
                    raise ValueError(
                        f"WS mismatch ia={ia} jw={jw} iw={iw}: arr={arr.shape[1:]} ws={(ws_el.nr, ws_ph.nr)}"
                    )
                rp_mask = np.all(np.asarray(ws_ph.vectors, dtype=np.int64) == rp_idx[None, :], axis=1)
                if not np.any(rp_mask):
                    continue
                phase_k = _phase_table(kpts, ws_el.vectors, phase_cache, f"gk:{iw}:{jw}")
                block_rp = np.sum(arr[ax_id][:, rp_mask], axis=1) * scale
                # Comparison path: keep Rp in real space and Fourier-transform only Re -> k.
                gk[0, :, iw - 1, jw - 1] += phase_k @ block_rp
        return gk


def _build_gk_rp_onsite_one(epr_path, target_ia0, axis, kpts, rp_idx, *, unit="ry", rotation_mode="none", phase_cache=None):
    if rotation_mode != "none":
        raise NotImplementedError("rotation_mode scaffold exists, but U-rotation is not implemented yet.")
    ax_id = {"x": 0, "y": 1, "z": 2}[axis]
    scale = _unit_scale_to_ev(unit)
    phase_cache = {} if phase_cache is None else phase_cache
    rp_idx = np.asarray(rp_idx, dtype=np.int64)
    with h5py.File(epr_path, "r") as h5:
        meta = _read_meta(h5)
        grp = h5["eph_matrix_wannier"]
        el_images = init_rvec_images(meta.nk_grid, meta.at)
        ph_images = init_rvec_images(meta.nq_grid, meta.at)
        nk = len(kpts)
        gk = np.zeros((1, nk, meta.nwan, meta.nwan), dtype=np.complex128)
        ws_cache = {}
        ia = int(target_ia0) + 1
        for jw in range(1, meta.nwan + 1):
            for iw in range(1, meta.nwan + 1):
                dr = f"ep_hop_r_{ia}_{jw}_{iw}"
                di = f"ep_hop_i_{ia}_{jw}_{iw}"
                if dr not in grp or di not in grp:
                    continue
                arr = (np.asarray(grp[dr], dtype=np.float64) + 1j * np.asarray(grp[di], dtype=np.float64)).transpose(2, 1, 0)
                key = (jw, iw)
                got = ws_cache.get(key)
                if got is None:
                    iw0 = iw - 1
                    jw0 = jw - 1
                    ws_el = set_wigner_seitz_cell(el_images, meta.at, meta.wc[iw0], meta.wc[jw0])
                    ws_ph = set_wigner_seitz_cell(ph_images, meta.at, meta.wc[iw0], meta.tau[target_ia0])
                    got = (ws_el, ws_ph)
                    ws_cache[key] = got
                ws_el, ws_ph = got
                if arr.shape[1] != ws_el.nr or arr.shape[2] != ws_ph.nr:
                    raise ValueError(
                        f"WS mismatch ia={ia} jw={jw} iw={iw}: arr={arr.shape[1:]} ws={(ws_el.nr, ws_ph.nr)}"
                    )
                re_mask = np.all(np.asarray(ws_el.vectors, dtype=np.int64) == 0, axis=1)
                rp_mask = np.all(np.asarray(ws_ph.vectors, dtype=np.int64) == rp_idx[None, :], axis=1)
                if not np.any(re_mask) or not np.any(rp_mask):
                    continue
                block = np.sum(arr[ax_id][re_mask][:, rp_mask]) * scale
                gk[0, :, iw - 1, jw - 1] += block
        return gk


def _build_g_cache_entry(task):
    spin, epr_path, ia0, ax, labels, g_transform, kpts, qpts, rp_idx, eph_unit, rotation_mode, onsite_only = task
    local_phase_cache = {}
    target_name = f"atom #{ia0 + 1} {labels[ia0]}"
    if g_transform == "kq":
        suffix = " onsite" if onsite_only else ""
        print(f"[dJ-epr-kspace] precache g(k,q){suffix}: {target_name}/{ax} {spin}", flush=True)
        builder = _build_gkq_onsite_one if onsite_only else _build_gkq_one
        arr = builder(epr_path, ia0, ax, kpts, qpts, unit=eph_unit, rotation_mode=rotation_mode, phase_cache=local_phase_cache)
    else:
        suffix = " onsite" if onsite_only else ""
        print(f"[dJ-epr-kspace] precache g(k,Rp={tuple(rp_idx)}){suffix}: {target_name}/{ax} {spin}", flush=True)
        builder = _build_gk_rp_onsite_one if onsite_only else _build_gk_rp_one
        arr = builder(
            epr_path, ia0, ax, kpts, rp_idx, unit=eph_unit, rotation_mode=rotation_mode, phase_cache=local_phase_cache
        )
    return (spin, ia0, ax, bool(onsite_only)), arr


def _precache(args, labels, target_ids, axes, slices, pair_meta):
    kpts = _full_k_mesh(args.kmesh)
    with h5py.File(args.epr_up, "r") as h5:
        epr_qmesh = tuple(int(x) for x in h5["basic_data/qc_dim"][()])
    if args.g_transform == "kq":
        qmesh = tuple(int(x) for x in (args.qmesh if args.qmesh is not None else epr_qmesh))
        qpts = _full_q_mesh(qmesh)
    else:
        qmesh = (1, 1, 1)
        qpts = np.zeros((1, 3), dtype=np.float64)
    phase_cache = {}
    print(f"[dJ-epr-kspace] precache phase tables + H(k): nk={len(kpts)} qmesh={qmesh} g_transform={args.g_transform}")
    hk_up, kc_up = _build_hk_epr_phase(args.epr_up, kpts, unit=args.hr_unit, phase_cache=phase_cache)
    hk_dn, kc_dn = _build_hk_epr_phase(args.epr_dn, kpts, unit=args.hr_unit, phase_cache=phase_cache)
    if tuple(kc_up) != tuple(kc_dn):
        raise ValueError(f"EPR up/dn kc_dim mismatch: up={kc_up} dn={kc_dn}")
    eig = _precompute_eigensystem(hk_up, hk_dn, slices, args.efermi)
    phase_R = np.exp(-1j * 2.0 * np.pi * (np.asarray([m["R"] for m in pair_meta], dtype=np.float64) @ kpts.T))
    bond_target_q_phase = _bond_target_q_phase(qpts, pair_meta)
    if args.g_transform == "kq":
        kq = _kq_map(tuple(args.kmesh), qmesh)
    else:
        kq = np.arange(len(kpts), dtype=np.int64)[None, :]
    g_cache = {}
    tasks = []
    onsite_mode = str(getattr(args, "ddelta_mode", "off")).lower()
    need_ddelta_onsite = onsite_mode == "onsite"
    for ia0 in target_ids:
        for ax in axes:
            tasks.append(("up", args.epr_up, ia0, ax, labels, args.g_transform, kpts, qpts, args.rp_idx, args.eph_unit, args.rotation_mode, False))
            tasks.append(("down", args.epr_dn, ia0, ax, labels, args.g_transform, kpts, qpts, args.rp_idx, args.eph_unit, args.rotation_mode, False))
            if need_ddelta_onsite:
                tasks.append(("up", args.epr_up, ia0, ax, labels, args.g_transform, kpts, qpts, args.rp_idx, args.eph_unit, args.rotation_mode, True))
                tasks.append(("down", args.epr_dn, ia0, ax, labels, args.g_transform, kpts, qpts, args.rp_idx, args.eph_unit, args.rotation_mode, True))
    npre = min(max(1, int(args.precache_workers)), len(tasks))
    if npre <= 1:
        for task in tasks:
            key, arr = _build_g_cache_entry(task)
            g_cache[key] = arr
    else:
        print(f"[dJ-epr-kspace] precache g entries in parallel: workers={npre} tasks={len(tasks)}", flush=True)
        with ThreadPoolExecutor(max_workers=npre) as pool:
            futures = [pool.submit(_build_g_cache_entry, task) for task in tasks]
            for fut in as_completed(futures):
                key, arr = fut.result()
                g_cache[key] = arr
    ddelta_cache = {}
    if onsite_mode in {"local", "onsite"}:
        for ia0 in target_ids:
            for ax in axes:
                if onsite_mode == "onsite":
                    gu = g_cache[("up", ia0, ax, True)]
                    gd = g_cache[("down", ia0, ax, True)]
                else:
                    gu = g_cache[("up", ia0, ax, False)]
                    gd = g_cache[("down", ia0, ax, False)]
                ddelta_cache[(ia0, ax)] = 0.5 * (gu - gd)
    return {
        "kpts": kpts,
        "qpts": qpts,
        "qmesh": qmesh,
        "rp_grid": _rp_grid(qmesh) if args.g_transform == "kq" else np.asarray([args.rp_idx], dtype=np.int64),
        "eig": eig,
        "phase_R": phase_R,
        "bond_target_q_phase": bond_target_q_phase,
        "kq_map": kq,
        "g": g_cache,
        "ddelta": ddelta_cache,
        "ddelta_mode": onsite_mode,
    }

def _compute_chunk(energy_chunk):
    st = _WORKER_STATIC
    eig = st["eig"]
    pair_meta = st["pair_meta"]
    slices = st["slices"]
    phase_R = st["phase_R"]
    bond_target_q_phase = st["bond_target_q_phase"]
    kq_map = st["kq_map"]
    g_cache = st["g"]
    ddelta_cache = st.get("ddelta", {})
    ddelta_mode = str(st.get("ddelta_mode", "off")).lower()
    target_ids = st["target_ids"]
    axes = st["axes"]
    nk = phase_R.shape[1]
    nq = kq_map.shape[0]
    wk = 1.0 / float(nk)
    out = {(ia, ax): np.zeros((nq, len(pair_meta)), dtype=np.complex128) for ia in target_ids for ax in axes}

    for z, dz in energy_chunk:
        inv_u = 1.0 / (z - eig["evals"]["up"])
        inv_d = 1.0 / (z - eig["evals"]["down"])
        Gu = np.einsum(
            "kni,ki,kmi->knm",
            eig["coeffs"]["up"],
            inv_u,
            np.conjugate(eig["coeffs"]["up"]),
            optimize=True,
        )
        Gd = np.einsum(
            "kni,ki,kmi->knm",
            eig["coeffs"]["down"],
            inv_d,
            np.conjugate(eig["coeffs"]["down"]),
            optimize=True,
        )

        GRu = []
        GRd = []
        for ip, meta in enumerate(pair_meta):
            li = int(meta["li"])
            lj = int(meta["lj"])
            sl_i = slices[li]
            sl_j = slices[lj]
            ph = phase_R[ip] * wk
            GRu.append(np.einsum("k,kij->ij", ph, Gu[:, sl_i, sl_j], optimize=True))
            GRd.append(np.einsum("k,kji->ji", np.conjugate(ph), Gd[:, sl_j, sl_i], optimize=True))

        for ia in target_ids:
            for ax in axes:
                gu = g_cache[("up", ia, ax, False)]
                gd = g_cache[("down", ia, ax, False)]
                ddelta = ddelta_cache.get((ia, ax))
                for iq in range(nq):
                    Gu_kq = Gu[kq_map[iq]]
                    Gd_kq = Gd[kq_map[iq]]
                    dGu = Gu_kq @ gu[iq] @ Gu
                    dGd = Gd_kq @ gd[iq] @ Gd
                    for ip, meta in enumerate(pair_meta):
                        li = int(meta["li"])
                        lj = int(meta["lj"])
                        sl_i = slices[li]
                        sl_j = slices[lj]
                        ph = phase_R[ip] * wk
                        endpoint_phase = bond_target_q_phase[iq, ip]
                        dGRu = np.einsum("k,kij->ij", ph, dGu[:, sl_i, sl_j], optimize=True)
                        dGRd = endpoint_phase * np.einsum(
                            "k,kji->ji", np.conjugate(ph), dGd[:, sl_j, sl_i], optimize=True
                        )
                        Di = eig["delta"][li]
                        Dj = eig["delta"][lj]
                        rel_sign = eig["signs"].get(li, 1.0) * eig["signs"].get(lj, 1.0)
                        tr = (
                            np.trace(Di @ dGRu @ Dj @ GRd[ip])
                            + np.trace(Di @ GRu[ip] @ Dj @ dGRd)
                        ) / rel_sign
                        if ddelta_mode != "off" and ddelta is not None:
                            ddel_li = np.mean(ddelta[iq, :, sl_i, sl_i], axis=0)
                            ddel_lj = endpoint_phase * np.mean(ddelta[iq, :, sl_j, sl_j], axis=0)
                            tr += (
                                np.trace(ddel_li @ GRu[ip] @ Dj @ GRd[ip])
                                + np.trace(Di @ GRu[ip] @ ddel_lj @ GRd[ip])
                            ) / rel_sign
                        out[(ia, ax)][iq, ip] += tr * dz
    return out


def _worker_init(static_payload):
    global _WORKER_STATIC
    _configure_threads(static_payload.get("omp_threads", 1))
    _WORKER_STATIC = static_payload


def _compute_dj(pre, slices, pair_meta, target_ids, axes, energy_mesh, nproc, omp_threads=1):
    static = {
        "eig": pre["eig"],
        "phase_R": pre["phase_R"],
        "bond_target_q_phase": pre["bond_target_q_phase"],
        "kq_map": pre["kq_map"],
        "qmesh": pre["qmesh"],
        "g": pre["g"],
        "ddelta": pre.get("ddelta", {}),
        "ddelta_mode": pre.get("ddelta_mode", "off"),
        "slices": slices,
        "pair_meta": pair_meta,
        "target_ids": target_ids,
        "axes": axes,
        "omp_threads": max(1, int(omp_threads)),
    }
    chunks = _split_chunks(energy_mesh, max(1, int(nproc)))
    if int(nproc) <= 1 or len(chunks) <= 1:
        global _WORKER_STATIC
        _configure_threads(omp_threads)
        _WORKER_STATIC = static
        parts = [_compute_chunk(chunks[0])]
    else:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ctx.Pool(processes=len(chunks), initializer=_worker_init, initargs=(static,)) as pool:
            parts = pool.map(_compute_chunk, chunks)
    nq = int(np.prod(pre["qmesh"]))
    out_q = {(ia, ax): np.zeros((nq, len(pair_meta)), dtype=np.complex128) for ia in target_ids for ax in axes}
    for part in parts:
        for key, val in part.items():
            out_q[key] += val
    qmesh = tuple(int(x) for x in pre["qmesh"])
    djr = {}
    for key, val in out_q.items():
        grid = val.reshape(qmesh[0], qmesh[1], qmesh[2], len(pair_meta))
        # The EPR phase path uses exp(+i q.Rp), so the inverse q->Rp transform
        # is the negative-sign DFT, i.e. numpy fft normalized by Nq.
        real_grid = (np.fft.fftn(grid, axes=(0, 1, 2)) / float(nq)).reshape(nq, len(pair_meta))
        djr[key] = 1000.0 * np.imag(real_grid) / (4.0 * np.pi)
    return djr, {"n_chunks": len(chunks), "qmesh": qmesh, "rp_grid": np.asarray(pre["rp_grid"], dtype=np.int64)}


def _rp_slice_for_text(djr, rp_grid, rp_idx):
    rp_idx = tuple(int(x) for x in rp_idx)
    matches = np.where(np.all(np.asarray(rp_grid, dtype=np.int64) == np.asarray(rp_idx, dtype=np.int64)[None, :], axis=1))[0]
    irp = int(matches[0]) if len(matches) else 0
    return {key: val[irp] for key, val in djr.items()}, tuple(int(x) for x in rp_grid[irp])


def _orbit_label_map(pair_meta, orbits):
    meta_by_key = {(m["gi"], m["gj"], tuple(m["R"])): m for m in pair_meta}
    orbit_counter = {}
    orbits = sorted(orbits, key=lambda o: (int(o[0].get("shell_idx", 0)), float(o[0]["distance"]), int(o[0]["i"]), int(o[0]["j"]), tuple(o[0]["R"])))
    orbit_label_by_key = {}
    for orbit in orbits:
        keys = [(int(n["i"]), int(n["j"]), tuple(int(x) for x in n["R"])) for n in orbit]
        keys = [k for k in keys if k in meta_by_key]
        if not keys:
            continue
        sh = int(meta_by_key[keys[0]]["shell"])
        orbit_counter.setdefault(sh, 0)
        label = f"{sh}{chr(97 + orbit_counter[sh])}"
        orbit_counter[sh] += 1
        for k in keys:
            orbit_label_by_key[k] = label
    return orbit_label_by_key


def _write_outputs(path, args, labels, axes, target_ids, pair_meta, dj, orbits, elapsed, exe_info, rp_text=(0, 0, 0)):
    orbit_label_by_key = _orbit_label_map(pair_meta, orbits)
    pair_index = {id(m): ip for ip, m in enumerate(pair_meta)}
    with open(path, "w") as f:
        f.write(f"# dJ/du Results - EPR direct k,q space, text Rp={tuple(rp_text)}\n")
        f.write(f"# EPR up: {os.path.abspath(args.epr_up)}\n")
        f.write(f"# EPR dn: {os.path.abspath(args.epr_dn)}\n")
        f.write(f"# H unit={args.hr_unit} eph unit={args.eph_unit}\n")
        f.write(f"# g_transform={args.g_transform} rp_idx={tuple(args.rp_idx)}\n")
        f.write(f"# terms: dG + dDelta_mode={args.ddelta_mode}\n")
        f.write(f"# kmesh={tuple(args.kmesh)} nE={args.empoints} nproc={args.nproc} n_chunks={exe_info['n_chunks']} elapsed_s={elapsed:.2f}\n\n")
        for m in sorted(pair_meta, key=lambda x: (x["shell"], x["dist"], x["gi"], x["gj"], tuple(x["R"]))):
            key = (m["gi"], m["gj"], tuple(m["R"]))
            label = orbit_label_by_key.get(key, "NA")
            gi = int(m["gi"])
            gj = int(m["gj"])
            li = int(m["li"])
            lj = int(m["lj"])
            ilab = labels[gi] if gi < len(labels) else f"Atom{gi + 1}"
            jlab = labels[gj] if gj < len(labels) else f"Atom{gj + 1}"
            f.write(
                f"Orbit: {label}   pair i={gi}({ilab}, atom #{gi + 1}, local {li}) "
                f"j={gj}({jlab}, atom #{gj + 1}, local {lj})   "
                f"R = {_format_r(m['R'])}, Rp = {_format_r(rp_text)} dist = {m['dist']:.6f} A\n"
            )
            ip = pair_index[id(m)]
            for ia in target_ids:
                vals = [float(dj[(ia, ax)][ip]) for ax in axes]
                comp = {ax: vals[i] for i, ax in enumerate(axes)}
                dx = comp.get("x", 0.0)
                dy = comp.get("y", 0.0)
                dz = comp.get("z", 0.0)
                f.write(
                    f"moved atom idx={ia} atom #{ia + 1} {labels[ia]} "
                    f"dJ : ({dx:.10e}, {dy:.10e}, {dz:.10e}) meV/A\n"
                )
            f.write("\n")

    tsv = os.path.splitext(path)[0] + ".all_bonds.tsv"
    with open(tsv, "w") as f:
        f.write(
            "orbit\tgi\tgj\ti_atom\tj_atom\ti_label\tj_label\tli\tlj\tR1\tR2\tR3\tdist_A\t"
            "target_idx\ttarget_atom\ttarget_label\tdJx\tdJy\tdJz\n"
        )
        for m in sorted(pair_meta, key=lambda x: (x["shell"], x["dist"], x["gi"], x["gj"], tuple(x["R"]))):
            key = (m["gi"], m["gj"], tuple(m["R"]))
            label = orbit_label_by_key.get(key, "NA")
            ip = pair_index[id(m)]
            gi = int(m["gi"])
            gj = int(m["gj"])
            li = int(m["li"])
            lj = int(m["lj"])
            ilab = labels[gi] if gi < len(labels) else f"Atom{gi + 1}"
            jlab = labels[gj] if gj < len(labels) else f"Atom{gj + 1}"
            for ia in target_ids:
                vals = {ax: float(dj[(ia, ax)][ip]) for ax in axes}
                f.write(
                    f"{label}\t{gi}\t{gj}\t{gi + 1}\t{gj + 1}\t{ilab}\t{jlab}\t{li}\t{lj}\t"
                    f"{m['R'][0]}\t{m['R'][1]}\t{m['R'][2]}\t{m['dist']:.12e}\t"
                    f"{ia}\t{ia + 1}\t{labels[ia]}\t"
                    f"{vals.get('x', 0.0):.12e}\t{vals.get('y', 0.0):.12e}\t{vals.get('z', 0.0):.12e}\n"
                )
    return tsv


def _mirror_indices(pair_meta):
    by_key = {(int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"])): i for i, m in enumerate(pair_meta)}
    out = np.full(len(pair_meta), -1, dtype=np.int64)
    for i, m in enumerate(pair_meta):
        key = (int(m["gj"]), int(m["gi"]), tuple(-int(x) for x in m["R"]))
        out[i] = by_key.get(key, -1)
    return out


def _write_h5(path, args, labels, axes, target_ids, pair_meta, djr, rp_grid, orbits, elapsed, exe_info):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    axis_to_col = {"x": 0, "y": 1, "z": 2}
    str_dt = h5py.string_dtype(encoding="utf-8")
    orbit_label_by_key = _orbit_label_map(pair_meta, orbits)
    with h5py.File(path, "w") as h5:
        h5.attrs["complete"] = 1
        h5.attrs["derivative_representation"] = "isotropic"
        h5.attrs["fourier_phase_convention"] = "exp(+i2pi_q_dot_Rp)"
        h5.attrs["directed_bond_mate"] = "(j,i,-R;Rp-R)_periodic"
        h5.attrs["atomic_gauge_bond_phase"] = (
            "endpoint_j_at_R_times_exp(+i2pi_q_dot_R)"
        )
        def put_string(group, name, value):
            group.create_dataset(name, data=np.array(str(value), dtype=object), dtype=str_dt)

        basic = h5.create_group("basic_data")
        basic.create_dataset("nat", data=np.array(len(labels), dtype=np.int64))
        basic.create_dataset("atom_labels", data=np.asarray(labels, dtype=object), dtype=str_dt)
        basic.create_dataset("kmesh", data=np.asarray(args.kmesh, dtype=np.int64))
        basic.create_dataset("qmesh", data=np.asarray(exe_info["qmesh"], dtype=np.int64))
        basic.create_dataset("efermi_ev", data=np.array(float(args.efermi), dtype=np.float64))
        put_string(basic, "unit", "meV/A")
        put_string(basic, "hamiltonian_sign", "minus")
        put_string(basic, "spin_normalization", "unit_vector")
        put_string(basic, "bond_coverage", "directed_mate_complete")
        basic.create_dataset(
            "directed_bond_weight", data=np.asarray(0.5, dtype=np.float64)
        )
        put_string(basic, "realspace_gauge", "i_at_0_j_at_R")
        basic.create_dataset(
            "source_spin_magnitude", data=np.asarray(1.0, dtype=np.float64)
        )
        put_string(basic, "kernel_family", "scalar_lkag")
        put_string(basic, "hr_unit", args.hr_unit)
        put_string(basic, "eph_unit", args.eph_unit)
        put_string(basic, "integrator", args.integrator)
        basic.create_dataset("empoints", data=np.array(int(args.empoints), dtype=np.int64))
        basic.create_dataset("nproc", data=np.array(int(args.nproc), dtype=np.int64))
        basic.create_dataset(
            "mpi_size", data=np.array(int(exe_info.get("mpi_size", 1)), dtype=np.int64)
        )
        basic.create_dataset("omp_threads", data=np.array(int(args.omp_threads), dtype=np.int64))
        basic.create_dataset("precache_workers", data=np.array(int(args.precache_workers), dtype=np.int64))
        put_string(basic, "g_transform", args.g_transform)
        put_string(basic, "ddelta_mode", getattr(args, "ddelta_mode", "off"))
        basic.create_dataset("elapsed_s", data=np.array(float(elapsed), dtype=np.float64))
        put_string(basic, "command", " ".join(sys.argv))
        with h5py.File(args.epr_up, "r") as src:
            meta = _read_meta(src)
            if "basic_data/alat" in src:
                alat_ang = float(src["basic_data/alat"][()]) * BOHR_TO_ANG
                lattice_ang = np.asarray(meta.at, dtype=np.float64) * alat_ang
                basic.create_dataset("lattice_ang", data=lattice_ang)
                basic.create_dataset("tau_frac", data=np.asarray(meta.tau, dtype=np.float64))
                basic.create_dataset("tau_cart_ang", data=np.asarray(meta.tau, dtype=np.float64) @ lattice_ang)

        bonds = h5.create_group("bonds")
        bonds.create_dataset("mag_i_atom", data=np.asarray([m["gi"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_j_atom", data=np.asarray([m["gj"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_i_local", data=np.asarray([m["li"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_j_local", data=np.asarray([m["lj"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("R", data=np.asarray([m["R"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("distance_ang", data=np.asarray([m["dist"] for m in pair_meta], dtype=np.float64))
        bonds.create_dataset("shell", data=np.asarray([m["shell"] for m in pair_meta], dtype=np.int64))
        orbit_labels = [orbit_label_by_key.get((m["gi"], m["gj"], tuple(m["R"])), "NA") for m in pair_meta]
        bonds.create_dataset("orbit_label", data=np.asarray(orbit_labels, dtype=object), dtype=str_dt)
        bonds.create_dataset("mirror_index", data=_mirror_indices(pair_meta))

        disp = h5.create_group("displacements")
        disp.create_dataset("target_atom", data=np.asarray(target_ids, dtype=np.int64))
        disp.create_dataset("target_label", data=np.asarray([labels[i] for i in target_ids], dtype=object), dtype=str_dt)
        disp.create_dataset("axes", data=np.asarray(["x", "y", "z"], dtype=object), dtype=str_dt)
        disp.create_dataset("Rp", data=np.asarray(rp_grid, dtype=np.int64))

        grp = h5.create_group("dJ_r")
        grp.attrs["dataset_shape"] = "(nRp, 3)"
        grp.attrs["axis_order"] = "x,y,z"
        grp.attrs["meaning"] = "dJ_r_m{moved_atom_1based}_b{bond_1based}[irp,axis] = dJ(R_bond,Rp_irp)/du_target_axis"
        n_rp = len(rp_grid)
        n_bond = len(pair_meta)
        for ia in target_ids:
            for ib in range(n_bond):
                arr = np.zeros((n_rp, 3), dtype=np.float64)
                for ax in axes:
                    arr[:, axis_to_col[ax]] = djr[(ia, ax)][:, ib]
                dset = grp.create_dataset(f"dJ_r_m{ia + 1}_b{ib + 1}", data=arr)
                dset.attrs["target_atom"] = int(ia)
                dset.attrs["bond_index"] = int(ib)
                dset.attrs["unit"] = "meV/A"


def run(args, comm=None):
    t0 = time.time()
    _configure_threads(args.omp_threads)
    ncpu = os.cpu_count() or 1
    if int(args.nproc) * int(args.omp_threads) > ncpu:
        print(
            f"[dJ-epr-kspace] warning: nproc*omp_threads={int(args.nproc) * int(args.omp_threads)} > cpu_count={ncpu}",
            flush=True,
        )
    species_labels = [x.strip() for x in str(args.species_labels).split(",") if x.strip()] if args.species_labels else None
    labels = _atom_labels(args.epr_up, args.atom_labels, species_labels=species_labels)
    target_ids = _target_indices(args.targets, labels)
    axes = _axis_list(args.axes)
    mag_atoms = _normalize_mag_atoms(args)
    with h5py.File(args.epr_up, "r") as h5:
        dim = int(h5["basic_data/num_wann"][()])
    slices = _load_slices(args, dim)
    neighbours = _find_nearest_neighbours_from_epr(args.epr_up, mag_atoms, n_shells=args.n_shells, d_max=args.d_max, all_bonds=True)
    global_to_local = {g: i for i, g in enumerate(mag_atoms)}
    pair_meta = []
    for n in neighbours:
        gi = int(n["i"])
        gj = int(n["j"])
        if gi not in global_to_local or gj not in global_to_local:
            continue
        pair_meta.append(
            {
                "gi": gi,
                "gj": gj,
                "li": global_to_local[gi],
                "lj": global_to_local[gj],
                "R": tuple(int(x) for x in n["R"]),
                "dist": float(n["distance"]),
                "shell": int(n.get("shell_idx", 0)),
            }
        )
    if not pair_meta:
        raise RuntimeError("No selected magnetic bonds found.")

    if args.integrator == "contour":
        energy_mesh = get_semicircle_contour(emin=args.emin, emax=0.0, npoints=args.empoints)
    elif args.integrator == "cfr_ozaki":
        energy_mesh = get_cfr_ozaki_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)
    else:
        energy_mesh = get_cfr_pole_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)

    pre = _precache(args, labels, target_ids, axes, slices, pair_meta)
    rank, size = rank_size(comm)
    local_energy_mesh = partition_sequence(energy_mesh, comm)
    print(
        f"[dJ-epr-kspace] assemble dJ: bonds={len(pair_meta)} "
        f"targets={len(target_ids)} axes={axes} nE={len(energy_mesh)} "
        f"local_nE={len(local_energy_mesh)} mpi={size}",
        flush=True,
    )
    local_info = {}

    def integrate_local():
        dj_local, info = _compute_dj(
            pre,
            slices,
            pair_meta,
            target_ids,
            axes,
            local_energy_mesh,
            int(args.nproc) if size == 1 else 1,
            args.omp_threads,
        )
        local_info.update(info)
        return dj_local

    reduced = collective_sum(comm, integrate_local)
    if rank != 0:
        return
    if reduced is None:  # pragma: no cover - defensive communicator guard
        raise RuntimeError("MPI root did not receive scalar dJ reduction")
    djr = reduced
    exe_info = {
        **local_info,
        "n_chunks": size if size > 1 else local_info.get("n_chunks", 1),
        "mpi_size": size,
    }
    orbits = _group_orbits_epr(
        args.epr_up,
        neighbours,
        use_symmetry=not bool(args.no_symmetry_orbits),
        symprec=float(args.symprec),
        labels=labels,
        species_labels=species_labels,
        orbit_grouping=args.orbit_grouping,
        angle_tolerance=float(args.angle_tolerance),
        debug_orbits=bool(args.debug_orbits),
        debug_orbit_shell=args.debug_orbit_shell,
        debug_epr_positions=bool(args.debug_epr_positions),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, args.out_name)
    rp_grid = np.asarray(exe_info["rp_grid"], dtype=np.int64)
    dj_text, rp_text = _rp_slice_for_text(djr, rp_grid, args.rp_idx)
    elapsed = time.time() - t0
    tsv = _write_outputs(out, args, labels, axes, target_ids, pair_meta, dj_text, orbits, elapsed, exe_info, rp_text=rp_text)
    out_h5 = args.out_h5 if os.path.isabs(args.out_h5) else os.path.join(args.out_dir, args.out_h5)
    _write_h5(out_h5, args, labels, axes, target_ids, pair_meta, djr, rp_grid, orbits, elapsed, exe_info)
    print(f"[dJ-epr-kspace] wrote {out}")
    print(f"[dJ-epr-kspace] wrote {tsv}")
    print(f"[dJ-epr-kspace] wrote {out_h5}")


def main():
    ap = argparse.ArgumentParser(description="Direct EPR k,q-space dJ/du calculator for Rp=(0,0,0)")
    ap.add_argument("--epr_up", required=True)
    ap.add_argument("--epr_dn", required=True)
    ap.add_argument("--hr_unit", choices=["ev", "ry", "ha"], default="ry")
    ap.add_argument("--eph_unit", choices=["ev", "ry", "ha"], default="ry")
    ap.add_argument("--atom_labels", default="")
    ap.add_argument("--species_labels", default="", help="Comma-separated species for spglib orbit grouping, e.g. Mn,Mn,Te,Te.")
    ap.add_argument("--targets", default="all", help="Comma labels or 0-based indices; default all")
    ap.add_argument(
        "--axes",
        default="xyz",
        help="Displacement axes in compact or comma-separated form, e.g. xyz or x,y,z",
    )
    ap.add_argument("--mag_atoms", type=int, nargs="+", required=True)
    ap.add_argument("--mag_atoms_base", type=int, choices=[0, 1], default=0)
    ap.add_argument("--slices", required=True, help="Local orbital slices, e.g. '0:0:5,1:5:10'")
    ap.add_argument("--efermi", type=float, required=True)
    ap.add_argument("--kmesh", type=int, nargs=3, required=True)
    ap.add_argument("--qmesh", type=int, nargs=3, default=None, help="Output q mesh; default EPR basic_data/qc_dim")
    ap.add_argument("--rp_idx", type=int, nargs=3, default=[0, 0, 0])
    ap.add_argument(
        "--g_transform",
        choices=["kq", "k_only_rp"],
        default="kq",
        help="kq: FT ep_hop over Re and Rp; k_only_rp: keep selected Rp real-space and FT only Re",
    )
    ap.add_argument("--n_shells", type=int, default=1)
    ap.add_argument("--d_max", type=float, default=20.0)
    ap.add_argument("--emin", type=float, default=-25.0)
    ap.add_argument("--empoints", type=int, default=100)
    ap.add_argument("--integrator", choices=["contour", "cfr", "cfr_ozaki"], default="contour")
    ap.add_argument("--cfr_beta", type=float, default=400.0)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--omp_threads", type=int, default=1, help="BLAS/OpenMP threads per energy worker process")
    ap.add_argument("--precache_workers", type=int, default=1, help="Thread workers for independent g(k,q) pre-cache entries")
    ap.add_argument("--rotation_mode", choices=["none"], default="none")
    ap.add_argument(
        "--ddelta_mode",
        choices=["off", "local", "onsite"],
        default="off",
        help=(
            "Include derivative of local exchange splitting Delta. "
            "off: legacy dG-only; local: use full local block of g_up-g_dn; "
            "onsite: use only electron Re=(0,0,0) onsite derivative."
        ),
    )
    ap.add_argument("--symprec", type=float, default=1.0e-4, help="spglib symmetry tolerance for orbit grouping.")
    ap.add_argument("--angle_tolerance", type=float, default=-1.0, help="spglib angle tolerance in degrees; -1 uses spglib default.")
    ap.add_argument(
        "--orbit_grouping",
        choices=["spglib", "shell"],
        default="spglib",
        help="Orbit grouping mode. shell groups all bonds with the same shell index and distance.",
    )
    ap.add_argument("--debug_orbits", action="store_true", help="Print spglib operation and bond-mapping diagnostics.")
    ap.add_argument("--debug_orbit_shell", type=int, default=None, help="Restrict --debug_orbits bond diagnostics to one shell.")
    ap.add_argument("--debug_epr_positions", action="store_true", help="Print EPR tau and Wannier-center position diagnostics.")
    ap.add_argument("--no_symmetry_orbits", action="store_true")
    ap.add_argument("--out_dir", default="dJ_epr_kspace")
    ap.add_argument("--out_name", default="dJ_epr_kspace.txt")
    ap.add_argument("--out_h5", default="dJr.h5")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
