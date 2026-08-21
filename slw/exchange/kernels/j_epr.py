"""Native scalar LKAG kernel for qe2pert EPR inputs."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import sys
import time
from dataclasses import dataclass

import h5py
import numpy as np
import spglib

from slw.core.constants import BOHR_TO_ANG
from slw.exchange.kernels.epr import (
    _build_hk_from_epr,
    _find_nearest_neighbours_from_epr,
    _full_k_mesh,
    _load_slices,
    _precompute_direct_kdata,
)
from slw.exchange.kernels.lkag import (
    get_cfr_ozaki_mesh,
    get_cfr_pole_mesh,
    get_semicircle_contour,
)
from slw.exchange.kernels.parallel import collective_sum, partition_sequence, rank_size

_WORKER_STATIC = None


@dataclass(frozen=True)
class _OrbitGrouping:
    orbits: list[list[dict]]
    source: str
    spacegroup: str = ""
    n_operations: int = 0
    species_source: str = ""


@dataclass(frozen=True)
class _OrbitSymmetryResult:
    values_mev: np.ndarray
    raw_values_mev: np.ndarray
    labels: tuple[str, ...]
    sizes: np.ndarray
    means_raw_mev: np.ndarray
    std_raw_mev: np.ndarray
    max_abs_deviation_raw_mev: np.ndarray
    policy: str
    tolerance_mev: float
    applied: bool


def _read_epr_positions_and_symops(epr_path):
    with h5py.File(epr_path, "r") as h5:
        at = np.asarray(h5["basic_data/at"], dtype=np.float64).T
        tau_cart = np.asarray(h5["basic_data/tau"], dtype=np.float64)
        tau = np.linalg.solve(at, tau_cart.T).T
        symop = np.asarray(h5["basic_data/symop"], dtype=np.int64) if "basic_data/symop" in h5 else None
    return tau, symop


def _species_numbers_from_labels(labels, nat):
    if not labels:
        return None
    cleaned = []
    for label in labels:
        tok = str(label).strip()
        m = re.match(r"([A-Za-z]+)", tok)
        cleaned.append(m.group(1).capitalize() if m else tok)
    if len(cleaned) != int(nat):
        return None
    by_species = {}
    out = []
    for sp in cleaned:
        if sp not in by_species:
            by_species[sp] = len(by_species) + 1
        out.append(by_species[sp])
    return np.asarray(out, dtype=np.int32)


def _read_epr_spglib_cell(epr_path, labels=None, species_labels=None):
    with h5py.File(epr_path, "r") as h5:
        at = np.asarray(h5["basic_data/at"], dtype=np.float64).T
        tau_cart = np.asarray(h5["basic_data/tau"], dtype=np.float64)
        tau = np.linalg.solve(at, tau_cart.T).T
        nat = int(h5["basic_data/nat"][()])
        alat_ang = float(h5["basic_data/alat"][()]) * BOHR_TO_ANG
        # EPR/QE stores lattice vectors as columns after the transpose above:
        # tau_frac @ at.T == tau_cart_alat, so spglib needs row vectors at.T.
        lattice_ang = at.T * alat_ang

        numbers = _species_numbers_from_labels(species_labels, nat)
        source = "species_labels" if numbers is not None else None
        for key in (
            "basic_data/atomic_numbers",
            "basic_data/atomic_number",
            "basic_data/zatom",
            "basic_data/ityp",
        ):
            if numbers is None and key in h5:
                arr = np.asarray(h5[key]).reshape(-1)
                if arr.size == nat:
                    numbers = arr.astype(np.int32)
                    source = key
                    break
        if numbers is None and "basic_data/mass" in h5:
            mass = np.asarray(h5["basic_data/mass"], dtype=np.float64).reshape(-1)
            if mass.size == nat:
                by_mass = {}
                out = []
                for val in mass:
                    # EPR masses are reliable for species equivalence; rounding avoids tiny text/binary noise.
                    key = round(float(val), 6)
                    if key not in by_mass:
                        by_mass[key] = len(by_mass) + 1
                    out.append(by_mass[key])
                numbers = np.asarray(out, dtype=np.int32)
                source = "basic_data/mass"

    if numbers is None:
        numbers = _species_numbers_from_labels(labels, nat)
        source = "atom_labels" if numbers is not None else None
    if numbers is None:
        numbers = np.ones(nat, dtype=np.int32)
        source = "all_same_fallback"
    return lattice_ang, np.mod(tau, 1.0), numbers, source


def _normalize_mag_atoms(args):
    vals = [int(x) - 1 if int(args.mag_atoms_base) == 1 else int(x) for x in args.mag_atoms]
    if len(vals) < 2:
        raise ValueError("--mag_atoms must contain at least two atoms")
    if any(x < 0 for x in vals):
        raise ValueError(f"Invalid --mag_atoms after base conversion: {vals}")
    return vals


def _mirror_key(key):
    i, j, r = key
    return (j, i, tuple(-int(x) for x in r))


def _group_orbits_epr_with_provenance(
    epr_path,
    neighbours,
    *,
    use_symmetry=True,
    symprec=1.0e-4,
    labels=None,
    species_labels=None,
    orbit_grouping="spglib",
    angle_tolerance=-1.0,
    debug_orbits=False,
    debug_orbit_shell=None,
    debug_epr_positions=False,
):
    """Group directed bonds and retain the exact grouping provenance.

    If EPR does not provide usable rotations, fall back to distance/pair grouping.
    The fallback still preserves every bond in the detailed list.
    """

    if not use_symmetry:
        return _OrbitGrouping(_group_orbits_fallback(neighbours), source="symmetry_disabled")
    if orbit_grouping == "shell":
        return _OrbitGrouping(_group_orbits_shell(neighbours), source="shell_distance")

    spglib_orbits = _group_orbits_epr_spglib(
        epr_path,
        neighbours,
        symprec=symprec,
        labels=labels,
        species_labels=species_labels,
        angle_tolerance=angle_tolerance,
        debug_orbits=debug_orbits,
        debug_orbit_shell=debug_orbit_shell,
        debug_epr_positions=debug_epr_positions,
    )
    if spglib_orbits is not None:
        return spglib_orbits

    pos, rotations = _read_epr_positions_and_symops(epr_path)
    if rotations is None or len(rotations) == 0:
        return _OrbitGrouping(_group_orbits_fallback(neighbours), source="distance_pair_fallback")

    key_to_idx = {(int(n["i"]), int(n["j"]), tuple(int(x) for x in n["R"])): idx for idx, n in enumerate(neighbours)}

    def apply_atom(atom_idx, rot):
        raw = rot @ pos[int(atom_idx)]
        mod = np.mod(raw, 1.0)
        for ia, p in enumerate(pos):
            diff = mod - p
            diff = diff - np.round(diff)
            if np.linalg.norm(diff) < float(symprec):
                shift = np.round(raw - p).astype(np.int64)
                return int(ia), shift
        return None, None

    used = set()
    orbits = []
    for idx, n in enumerate(neighbours):
        if idx in used:
            continue
        current = []
        stack = [idx]
        used.add(idx)
        while stack:
            cur_idx = stack.pop()
            cur = neighbours[cur_idx]
            current.append(cur)
            i = int(cur["i"])
            j = int(cur["j"])
            r = np.asarray(cur["R"], dtype=np.int64)
            for rot in rotations:
                ip, _ = apply_atom(i, rot)
                jp, _ = apply_atom(j, rot)
                if ip is None or jp is None:
                    continue
                v = pos[j] + r - pos[i]
                vp = rot @ v
                rp = np.round(vp - (pos[jp] - pos[ip])).astype(np.int64)
                for cand in (
                    (ip, jp, tuple(int(x) for x in rp)),
                    _mirror_key((ip, jp, tuple(int(x) for x in rp))),
                ):
                    target = key_to_idx.get(cand)
                    if target is not None and target not in used:
                        used.add(target)
                        stack.append(target)
        orbits.append(current)
    return _OrbitGrouping(
        orbits,
        source="epr_symop",
        n_operations=len(rotations),
    )


def _group_orbits_epr(*args, **kwargs):
    """Compatibility helper returning only the grouped bonds."""

    return _group_orbits_epr_with_provenance(*args, **kwargs).orbits


def _group_orbits_epr_spglib(
    epr_path,
    neighbours,
    *,
    symprec=1.0e-4,
    labels=None,
    species_labels=None,
    angle_tolerance=-1.0,
    debug_orbits=False,
    debug_orbit_shell=None,
    debug_epr_positions=False,
):
    try:
        lattice_ang, pos, numbers, species_source = _read_epr_spglib_cell(
            epr_path,
            labels=labels,
            species_labels=species_labels,
        )
        cell = (lattice_ang, pos, numbers)
        ops = spglib.get_symmetry(cell, symprec=float(symprec), angle_tolerance=float(angle_tolerance))
        dataset = spglib.get_symmetry_dataset(cell, symprec=float(symprec), angle_tolerance=float(angle_tolerance))
    except Exception as exc:  # noqa: BLE001 - spglib failures use the EPR fallback
        print(f"[J-epr-kspace][warn] spglib orbit grouping failed; falling back to EPR symop: {exc}")
        return None

    if ops is None or len(ops.get("rotations", [])) == 0:
        return None

    rotations = np.asarray(ops["rotations"], dtype=np.int64)
    translations = np.asarray(ops["translations"], dtype=np.float64)
    uniq_species = ",".join(str(int(x)) for x in np.unique(numbers))
    lengths = np.linalg.norm(lattice_ang, axis=1)
    angles = []
    for a, b in ((1, 2), (0, 2), (0, 1)):
        denom = max(float(lengths[a] * lengths[b]), 1.0e-30)
        cosang = float(np.dot(lattice_ang[a], lattice_ang[b]) / denom)
        angles.append(float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))))
    spg = "unknown"
    if dataset is not None:
        try:
            spg = f"{dataset.international} #{int(dataset.number)}"
        except (AttributeError, TypeError, ValueError):
            try:
                spg = f"{dataset['international']} #{int(dataset['number'])}"
            except (KeyError, TypeError, ValueError):
                spg = "unknown"
    print(
        f"[J-epr-kspace] orbit grouping: source=spglib n_ops={len(rotations)} "
        f"symprec={float(symprec):.3e} angle_tol={float(angle_tolerance):.3g} "
        f"spacegroup={spg} species_source={species_source} species_ids={uniq_species}"
    )
    print(
        "[J-epr-kspace] spglib cell metric: "
        f"lengths_A={np.asarray(lengths).round(8).tolist()} "
        f"angles_deg(alpha,beta,gamma)={np.asarray(angles).round(8).tolist()}"
    )
    if debug_epr_positions:
        _debug_epr_position_payload(epr_path, lattice_ang, pos)
    key_to_idx = {(int(n["i"]), int(n["j"]), tuple(int(x) for x in n["R"])): idx for idx, n in enumerate(neighbours)}

    def apply_op(atom_idx, rot, trans):
        raw = rot @ pos[int(atom_idx)] + trans
        mod = np.mod(raw, 1.0)
        for ia, p in enumerate(pos):
            diff = mod - p
            diff = diff - np.round(diff)
            if np.linalg.norm(diff) < float(symprec):
                shift = np.round(raw - p).astype(np.int64)
                return int(ia), shift
        return None, None

    if debug_orbits:
        _debug_spglib_orbit_maps(
            neighbours,
            pos,
            rotations,
            translations,
            key_to_idx,
            apply_op,
            labels=labels,
            shell=debug_orbit_shell,
        )

    used = set()
    orbits = []
    for idx, n in enumerate(neighbours):
        if idx in used:
            continue
        current = []
        stack = [idx]
        used.add(idx)
        while stack:
            cur_idx = stack.pop()
            cur = neighbours[cur_idx]
            current.append(cur)
            i = int(cur["i"])
            j = int(cur["j"])
            r = np.asarray(cur["R"], dtype=np.int64)
            for rot, trans in zip(rotations, translations):
                ip, _ = apply_op(i, rot, trans)
                jp, _ = apply_op(j, rot, trans)
                if ip is None or jp is None:
                    continue
                v = pos[j] + r - pos[i]
                vp = rot @ v
                rp = np.round(vp - (pos[jp] - pos[ip])).astype(np.int64)
                key = (ip, jp, tuple(int(x) for x in rp))
                for cand in (key, _mirror_key(key)):
                    target = key_to_idx.get(cand)
                    if target is not None and target not in used:
                        used.add(target)
                        stack.append(target)
        orbits.append(current)
    return _OrbitGrouping(
        orbits,
        source="spglib",
        spacegroup=spg,
        n_operations=len(rotations),
        species_source=str(species_source or ""),
    )


def _debug_spglib_orbit_maps(
    neighbours,
    pos,
    rotations,
    translations,
    key_to_idx,
    apply_op,
    *,
    labels=None,
    shell=None,
):
    if shell is None:
        selected = list(neighbours)
        shell_txt = "all"
    else:
        selected = [n for n in neighbours if int(n.get("shell_idx", 0)) == int(shell)]
        shell_txt = str(int(shell))
    print(f"[J-epr-kspace][orbit-debug] shell={shell_txt} selected_bonds={len(selected)}")
    print("[J-epr-kspace][orbit-debug] operations:")
    for iop, (rot, trans) in enumerate(zip(rotations, translations)):
        print(
            f"  op={iop:03d} det={round(np.linalg.det(rot)):+d} "
            f"rot={np.asarray(rot, dtype=int).tolist()} trans={np.asarray(trans, dtype=float).round(8).tolist()}"
        )

    for n in selected:
        i = int(n["i"])
        j = int(n["j"])
        r = np.asarray(n["R"], dtype=np.int64)
        src = (i, j, tuple(int(x) for x in r))
        src_label = _format_bond(i, j, r, labels=labels)
        hits = []
        misses = []
        for iop, (rot, trans) in enumerate(zip(rotations, translations)):
            ip, _ = apply_op(i, rot, trans)
            jp, _ = apply_op(j, rot, trans)
            if ip is None or jp is None:
                misses.append((iop, "atom-map-fail"))
                continue
            v = pos[j] + r - pos[i]
            vp = rot @ v
            rp = np.round(vp - (pos[jp] - pos[ip])).astype(np.int64)
            key = (ip, jp, tuple(int(x) for x in rp))
            mir = _mirror_key(key)
            if key in key_to_idx:
                hits.append((iop, key, "direct"))
            elif mir in key_to_idx:
                hits.append((iop, mir, "mirror"))
            else:
                misses.append((iop, key))
        unique_hits = []
        seen = set()
        for iop, key, mode in hits:
            if key in seen:
                continue
            seen.add(key)
            unique_hits.append((iop, key, mode))
        hit_txt = ", ".join(f"op{iop}->{_format_bond(key[0], key[1], key[2], labels=labels)}:{mode}" for iop, key, mode in unique_hits)
        if not hit_txt:
            hit_txt = "none"
        print(f"[J-epr-kspace][orbit-debug] {src_label} key={src} hits={hit_txt}")


def _debug_epr_position_payload(epr_path, lattice_ang, tau_frac):
    with h5py.File(epr_path, "r") as h5:
        keys = sorted(str(k) for k in h5["basic_data"]) if "basic_data" in h5 else []
        wc = np.asarray(h5["basic_data/wannier_center_cryst"], dtype=np.float64) if "basic_data/wannier_center_cryst" in h5 else None
        tau_raw = np.asarray(h5["basic_data/tau"], dtype=np.float64) if "basic_data/tau" in h5 else None
        at_raw = np.asarray(h5["basic_data/at"], dtype=np.float64) if "basic_data/at" in h5 else None
    print(f"[J-epr-kspace][epr-pos-debug] basic_data keys={keys}")
    if at_raw is not None:
        print(f"[J-epr-kspace][epr-pos-debug] raw at shape={at_raw.shape} values={np.asarray(at_raw).round(8).tolist()}")
    if tau_raw is not None:
        print(f"[J-epr-kspace][epr-pos-debug] raw tau shape={tau_raw.shape} values={np.asarray(tau_raw).round(8).tolist()}")
    print(f"[J-epr-kspace][epr-pos-debug] tau_frac_from_at shape={tau_frac.shape} values={np.asarray(tau_frac).round(8).tolist()}")
    if wc is None:
        print("[J-epr-kspace][epr-pos-debug] no basic_data/wannier_center_cryst dataset")
        return
    wc = np.mod(np.asarray(wc, dtype=np.float64), 1.0)
    print(f"[J-epr-kspace][epr-pos-debug] wannier_center_cryst shape={wc.shape}")
    tau_mod = np.mod(np.asarray(tau_frac, dtype=np.float64), 1.0)
    for ia, tau in enumerate(tau_mod):
        diff = wc - tau[None, :]
        diff = diff - np.round(diff)
        dist = np.linalg.norm(diff @ lattice_ang, axis=1)
        order = np.argsort(dist)
        nearest = [(int(i), float(dist[i])) for i in order[: min(8, len(order))]]
        n_025 = int(np.count_nonzero(dist < 0.25))
        n_050 = int(np.count_nonzero(dist < 0.50))
        n_100 = int(np.count_nonzero(dist < 1.00))
        print(
            f"[J-epr-kspace][epr-pos-debug] atom={ia} tau={tau.round(8).tolist()} "
            f"wc_within_A(0.25,0.50,1.00)=({n_025},{n_050},{n_100}) nearest_wc(index,dist_A)={nearest}"
        )


def _group_orbits_fallback(neighbours):
    groups = {}
    for n in neighbours:
        pair = tuple(sorted((int(n["i"]), int(n["j"]))))
        key = (int(n.get("shell_idx", 0)), round(float(n["distance"]), 4), pair)
        groups.setdefault(key, []).append(n)
    return list(groups.values())


def _group_orbits_shell(neighbours):
    groups = {}
    for n in neighbours:
        key = (int(n.get("shell_idx", 0)), round(float(n["distance"]), 4))
        groups.setdefault(key, []).append(n)
    print(f"[J-epr-kspace] orbit grouping: source=shell_distance n_orbits={len(groups)}")
    return list(groups.values())


def _split_chunks(seq, n_chunks):
    n = len(seq)
    n_chunks = max(1, min(int(n_chunks), n))
    base = n // n_chunks
    rem = n % n_chunks
    chunks = []
    i0 = 0
    for ic in range(n_chunks):
        sz = base + (1 if ic < rem else 0)
        chunks.append(seq[i0 : i0 + sz])
        i0 += sz
    return chunks


def _compute_j_chunk(energy_mesh, kdata, slices, pair_meta, phase, kweights):
    j_acc = np.zeros(len(pair_meta), dtype=np.complex128)
    trace_acc = np.zeros(len(pair_meta), dtype=np.complex128)

    for z, dz in energy_mesh:
        inv_up = 1.0 / (z - kdata["evals_up"])
        inv_dn = 1.0 / (z - kdata["evals_dn"])
        for ip, meta in enumerate(pair_meta):
            li = int(meta["li"])
            lj = int(meta["lj"])
            ciu = kdata["coeffs_up"][li]
            cju = kdata["coeffs_up"][lj]
            cjd = kdata["coeffs_dn"][lj]
            cid = kdata["coeffs_dn"][li]
            gu_k = np.einsum("kia,ka,kja->kij", ciu, inv_up, np.conjugate(cju), optimize=True)
            gd_k = np.einsum("kja,ka,kia->kji", cjd, inv_dn, np.conjugate(cid), optimize=True)
            wk_phase = kweights * phase[ip]
            gu = np.einsum("k,kij->ij", wk_phase, gu_k, optimize=True)
            gd = np.einsum("k,kji->ji", np.conjugate(wk_phase), gd_k, optimize=True)
            rel_sign = kdata["signs"].get(li, 1.0) * kdata["signs"].get(lj, 1.0)
            tr = np.trace(kdata["delta"][li] @ gu @ kdata["delta"][lj] @ gd) / rel_sign
            trace_acc[ip] += tr
            j_acc[ip] += tr * dz
    return j_acc, trace_acc


def _worker_init(static_payload):
    global _WORKER_STATIC
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _WORKER_STATIC = static_payload


def _worker_compute_chunk(energy_chunk):
    if _WORKER_STATIC is None:
        raise RuntimeError("Worker static payload is not initialized.")
    return _compute_j_chunk(
        energy_chunk,
        _WORKER_STATIC["kdata"],
        _WORKER_STATIC["slices"],
        _WORKER_STATIC["pair_meta"],
        _WORKER_STATIC["phase"],
        _WORKER_STATIC["kweights"],
    )


def _compute_j_direct(hk_up, hk_dn, slices, pair_meta, kpts, energy_mesh, efermi, *, nproc=1):
    kdata = _precompute_direct_kdata(hk_up, hk_dn, slices, efermi)
    nk = len(kpts)
    kweights = np.full(nk, 1.0 / float(max(1, nk)), dtype=np.float64)
    r_arr = np.asarray([m["R"] for m in pair_meta], dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * (r_arr @ kpts.T))

    nproc = max(1, int(nproc))
    if nproc <= 1 or len(energy_mesh) < 2:
        j_acc, trace_acc = _compute_j_chunk(energy_mesh, kdata, slices, pair_meta, phase, kweights)
        return (
            1000.0 * np.imag(j_acc) / (4.0 * np.pi),
            trace_acc,
            kdata,
            {"n_chunks": 1, "nproc": 1},
        )

    chunks = _split_chunks(energy_mesh, nproc)
    static_payload = {
        "kdata": kdata,
        "slices": slices,
        "pair_meta": pair_meta,
        "phase": phase,
        "kweights": kweights,
    }
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(processes=len(chunks), initializer=_worker_init, initargs=(static_payload,)) as pool:
        partials = pool.map(_worker_compute_chunk, chunks)

    j_acc = np.zeros(len(pair_meta), dtype=np.complex128)
    trace_acc = np.zeros(len(pair_meta), dtype=np.complex128)
    for j_part, tr_part in partials:
        j_acc += j_part
        trace_acc += tr_part
    return (
        1000.0 * np.imag(j_acc) / (4.0 * np.pi),
        trace_acc,
        kdata,
        {"n_chunks": len(chunks), "nproc": len(chunks)},
    )


def _atom_name(local_idx, global_idx, labels=None):
    gi = int(global_idx)
    if labels is not None and 0 <= gi < len(labels):
        return str(labels[gi])
    return f"Atom{gi + 1}"


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


def _atom_labels(epr_path, user_labels="", species_labels=None):
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


def _format_r(r):
    return f"({int(r[0])},{int(r[1])},{int(r[2])})"


def _format_bond(i, j, r, labels=None):
    return f"{_atom_name(0, i, labels)}-{_atom_name(0, j, labels)}@{_format_r(r)}"


def _mirror_indices(pair_meta):
    by_key = {(int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"])): i for i, m in enumerate(pair_meta)}
    out = np.full(len(pair_meta), -1, dtype=np.int64)
    for i, m in enumerate(pair_meta):
        key = (int(m["gj"]), int(m["gi"]), tuple(-int(x) for x in m["R"]))
        out[i] = by_key.get(key, -1)
    return out


def _orbit_label_map(pair_meta, orbits):
    meta_by_key = {(m["gi"], m["gj"], tuple(m["R"])): m for m in pair_meta}
    orbit_counter = {}
    orbits = sorted(
        orbits,
        key=lambda o: (
            int(o[0].get("shell_idx", 0)),
            float(o[0]["distance"]),
            int(o[0]["i"]),
            int(o[0]["j"]),
            tuple(o[0]["R"]),
        ),
    )
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


def _apply_orbit_symmetry(
    pair_meta,
    raw_values_mev,
    grouping,
    *,
    policy="project",
    tolerance_mev=1.0e-8,
):
    """Apply one scalar value per validated bond orbit.

    Projection is intentionally restricted to a successful spglib grouping.
    Shell/distance and EPR-symop fallbacks remain useful diagnostics, but they
    are not authoritative enough to alter the numerical result.
    """

    policy = str(policy).strip().lower()
    if policy not in {"report", "project", "fail"}:
        raise ValueError(f"orbit_symmetry must be one of report, project, or fail; got {policy!r}")
    tolerance_mev = float(tolerance_mev)
    if not np.isfinite(tolerance_mev) or tolerance_mev < 0.0:
        raise ValueError("orbit_symmetry_tolerance_mev must be finite and non-negative")
    raw = np.asarray(raw_values_mev, dtype=np.float64)
    if raw.shape != (len(pair_meta),) or not np.all(np.isfinite(raw)):
        raise ValueError("raw scalar J values must be a finite vector matching the bond list")
    if policy in {"project", "fail"} and grouping.source != "spglib":
        raise ValueError(f"orbit_symmetry={policy!r} requires a successful spglib grouping; the active grouping source is {grouping.source!r}")

    label_by_key = _orbit_label_map(pair_meta, grouping.orbits)
    bond_labels: list[str] = []
    for item in pair_meta:
        key = (int(item["gi"]), int(item["gj"]), tuple(int(x) for x in item["R"]))
        label = label_by_key.get(key)
        if label is None:
            raise RuntimeError(f"bond {key!r} was not assigned to a symmetry orbit")
        bond_labels.append(label)
    labels = tuple(dict.fromkeys(bond_labels))
    bond_label_array = np.asarray(bond_labels, dtype=object)

    values = raw.copy()
    sizes = np.empty(len(labels), dtype=np.int64)
    means = np.empty(len(labels), dtype=np.float64)
    stds = np.empty(len(labels), dtype=np.float64)
    max_deviations = np.empty(len(labels), dtype=np.float64)
    worst_label = ""
    worst_deviation = -1.0
    for index, label in enumerate(labels):
        members = np.flatnonzero(bond_label_array == label)
        orbit_values = raw[members]
        mean = float(np.mean(orbit_values))
        deviation = float(np.max(np.abs(orbit_values - mean)))
        sizes[index] = int(members.size)
        means[index] = mean
        stds[index] = float(np.std(orbit_values))
        max_deviations[index] = deviation
        if policy == "project":
            values[members] = mean
        if deviation > worst_deviation:
            worst_label = label
            worst_deviation = deviation

    if policy == "fail" and worst_deviation > tolerance_mev:
        raise ValueError(
            "scalar J violates the requested orbit-symmetry tolerance: "
            f"orbit={worst_label} max_abs_deviation={worst_deviation:.6g} meV "
            f"> tolerance={tolerance_mev:.6g} meV"
        )
    return _OrbitSymmetryResult(
        values_mev=values,
        raw_values_mev=raw.copy(),
        labels=labels,
        sizes=sizes,
        means_raw_mev=means,
        std_raw_mev=stds,
        max_abs_deviation_raw_mev=max_deviations,
        policy=policy,
        tolerance_mev=tolerance_mev,
        applied=policy == "project",
    )


def _copy_epr_structure_basic(h5_out, epr_path):
    """Copy canonical structure data from EPR basic_data when available."""
    basic = h5_out["basic_data"]
    with h5py.File(epr_path, "r") as src:
        if all(k in src for k in ["basic_data/alat", "basic_data/at", "basic_data/tau"]):
            alat_ang = float(src["basic_data/alat"][()]) * BOHR_TO_ANG
            at = np.asarray(src["basic_data/at"], dtype=np.float64).T
            tau_cart_alat = np.asarray(src["basic_data/tau"], dtype=np.float64)
            lattice_ang = at.T * alat_ang
            tau_frac = np.linalg.solve(at, tau_cart_alat.T).T
            tau_cart_ang = tau_frac @ lattice_ang
            basic.create_dataset("lattice_ang", data=np.asarray(lattice_ang, dtype=np.float64))
            basic.create_dataset("tau_frac", data=np.asarray(tau_frac, dtype=np.float64))
            basic.create_dataset("tau_cart_ang", data=np.asarray(tau_cart_ang, dtype=np.float64))


def _write_h5(
    path,
    args,
    labels,
    pair_meta,
    j_mev,
    orbits,
    elapsed,
    nk,
    nE,
    exe_info,
    *,
    raw_j_mev=None,
    orbit_symmetry=None,
    orbit_grouping=None,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    orbit_label_by_key = _orbit_label_map(pair_meta, orbits)
    canonical = np.asarray(j_mev, dtype=np.float64)
    raw = canonical if raw_j_mev is None else np.asarray(raw_j_mev, dtype=np.float64)
    if canonical.shape != raw.shape:
        raise ValueError("canonical and raw scalar J vectors must have the same shape")

    with h5py.File(path, "w") as h5:

        def put_string(group, name, value):
            group.create_dataset(name, data=np.array(str(value), dtype=object), dtype=str_dt)

        basic = h5.create_group("basic_data")
        basic.create_dataset("nat", data=np.array(len(labels), dtype=np.int64))
        basic.create_dataset("atom_labels", data=np.asarray(labels, dtype=object), dtype=str_dt)
        basic.create_dataset("kmesh", data=np.asarray(args.kmesh, dtype=np.int64))
        basic.create_dataset("efermi_ev", data=np.array(float(args.efermi), dtype=np.float64))
        put_string(basic, "unit", "meV")
        put_string(basic, "hamiltonian_sign", "minus")
        put_string(basic, "spin_normalization", "unit_vector")
        put_string(basic, "kernel_family", "scalar_lkag")
        put_string(basic, "bond_coverage", "directed_mate_complete")
        put_string(basic, "realspace_gauge", "i_at_0_j_at_R")
        basic.create_dataset("source_spin_magnitude", data=np.array(1.0, dtype=np.float64))
        basic.create_dataset("directed_bond_weight", data=np.array(0.5, dtype=np.float64))
        put_string(basic, "hr_unit", args.hr_unit)
        put_string(basic, "integrator", args.integrator)
        put_string(
            basic,
            "orbit_grouping_source",
            orbit_grouping.source if orbit_grouping is not None else "unspecified",
        )
        put_string(
            basic,
            "orbit_symmetry_policy",
            orbit_symmetry.policy if orbit_symmetry is not None else "report",
        )
        basic.create_dataset(
            "orbit_symmetry_applied",
            data=np.array(bool(orbit_symmetry.applied) if orbit_symmetry is not None else False),
        )
        basic.create_dataset(
            "orbit_symmetry_tolerance_mev",
            data=np.array(
                float(orbit_symmetry.tolerance_mev) if orbit_symmetry is not None else 0.0,
                dtype=np.float64,
            ),
        )
        basic.create_dataset("empoints", data=np.array(int(args.empoints), dtype=np.int64))
        basic.create_dataset("nproc", data=np.array(int(args.nproc), dtype=np.int64))
        basic.create_dataset("nk", data=np.array(int(nk), dtype=np.int64))
        basic.create_dataset("nE", data=np.array(int(nE), dtype=np.int64))
        basic.create_dataset("elapsed_s", data=np.array(float(elapsed), dtype=np.float64))
        basic.create_dataset("n_chunks", data=np.array(int(exe_info.get("n_chunks", 1)), dtype=np.int64))
        basic.create_dataset("mpi_size", data=np.array(int(exe_info.get("mpi_size", 1)), dtype=np.int64))
        put_string(basic, "command", " ".join(sys.argv))
        _copy_epr_structure_basic(h5, args.epr_up)

        bonds = h5.create_group("bonds")
        bonds.create_dataset("mag_i_atom", data=np.asarray([m["gi"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_j_atom", data=np.asarray([m["gj"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_i_local", data=np.asarray([m["li"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_j_local", data=np.asarray([m["lj"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("R", data=np.asarray([m["R"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset(
            "distance_ang",
            data=np.asarray([m["dist"] for m in pair_meta], dtype=np.float64),
        )
        bonds.create_dataset("shell", data=np.asarray([m["shell"] for m in pair_meta], dtype=np.int64))
        orbit_labels = [orbit_label_by_key.get((m["gi"], m["gj"], tuple(m["R"])), "NA") for m in pair_meta]
        bonds.create_dataset("orbit_label", data=np.asarray(orbit_labels, dtype=object), dtype=str_dt)
        bonds.create_dataset("mirror_index", data=_mirror_indices(pair_meta))

        grp = h5.create_group("J_r")
        grp.attrs["dataset_shape"] = "()"
        grp.attrs["meaning"] = "J_r_b{bond_1based} = J(R_bond)"
        grp.create_dataset("value", data=canonical)
        grp["value"].attrs["unit"] = "meV"
        grp["value"].attrs["meaning"] = "Canonical scalar J in bond order; spglib-orbit projected when enabled."
        grp.create_dataset("value_raw", data=raw)
        grp["value_raw"].attrs["unit"] = "meV"
        grp["value_raw"].attrs["meaning"] = "Unprojected numerical LKAG integration result in bond order."
        grp.create_dataset("projection_delta", data=canonical - raw)
        grp["projection_delta"].attrs["unit"] = "meV"
        for ib, val in enumerate(canonical):
            dset = grp.create_dataset(f"J_r_b{ib + 1}", data=np.array(float(val), dtype=np.float64))
            dset.attrs["bond_index"] = int(ib)
            dset.attrs["unit"] = "meV"

        if orbit_symmetry is not None:
            symmetry = h5.create_group("symmetry")
            put_string(symmetry, "grouping_source", orbit_grouping.source)
            put_string(symmetry, "policy", orbit_symmetry.policy)
            put_string(symmetry, "spacegroup", orbit_grouping.spacegroup)
            put_string(symmetry, "species_source", orbit_grouping.species_source)
            symmetry.create_dataset(
                "n_operations",
                data=np.array(int(orbit_grouping.n_operations), dtype=np.int64),
            )
            symmetry.create_dataset(
                "orbit_label",
                data=np.asarray(orbit_symmetry.labels, dtype=object),
                dtype=str_dt,
            )
            symmetry.create_dataset("orbit_size", data=orbit_symmetry.sizes)
            for name, values in (
                ("mean_raw_mev", orbit_symmetry.means_raw_mev),
                ("std_raw_mev", orbit_symmetry.std_raw_mev),
                (
                    "max_abs_deviation_raw_mev",
                    orbit_symmetry.max_abs_deviation_raw_mev,
                ),
            ):
                dataset = symmetry.create_dataset(name, data=values)
                dataset.attrs["unit"] = "meV"


def _write_outputs(
    args,
    output_path,
    neighbours,
    pair_meta,
    j_mev,
    orbits,
    mag_atoms,
    slices,
    elapsed,
    nk,
    nE,
    exe_info,
    labels=None,
    *,
    raw_j_mev=None,
    orbit_symmetry=None,
    orbit_grouping=None,
):
    raw_values = np.asarray(j_mev, dtype=np.float64) if raw_j_mev is None else np.asarray(raw_j_mev, dtype=np.float64)
    result_by_key = {(int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"])): float(j_mev[ip]) for ip, m in enumerate(pair_meta)}
    raw_by_key = {(int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"])): float(raw_values[ip]) for ip, m in enumerate(pair_meta)}
    meta_by_key = {(int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"])): m for m in pair_meta}

    orbit_rows = []
    detailed_rows = []
    orbit_counter = {}
    orbits = sorted(
        orbits,
        key=lambda o: (
            int(o[0].get("shell_idx", 0)),
            float(o[0]["distance"]),
            int(o[0]["i"]),
            int(o[0]["j"]),
            tuple(o[0]["R"]),
        ),
    )

    for orbit in orbits:
        keys = [(int(n["i"]), int(n["j"]), tuple(int(x) for x in n["R"])) for n in orbit]
        keys = [k for k in keys if k in result_by_key]
        if not keys:
            continue
        n0 = meta_by_key[keys[0]]
        shell = int(n0["shell"])
        orbit_counter.setdefault(shell, 0)
        orbit_label = f"{shell}{chr(97 + orbit_counter[shell])}"
        orbit_counter[shell] += 1
        vals = np.asarray([result_by_key[k] for k in keys], dtype=np.float64)
        raw_vals = np.asarray([raw_by_key[k] for k in keys], dtype=np.float64)
        name_i = _atom_name(n0["li"], n0["gi"], labels)
        name_j = _atom_name(n0["lj"], n0["gj"], labels)
        raw_mean = float(np.mean(raw_vals))
        orbit_rows.append(
            (
                shell,
                orbit_label,
                len(keys),
                f"{name_i}-{name_j}",
                float(n0["dist"]),
                float(np.mean(vals)),
                float(np.std(raw_vals)),
                float(np.max(np.abs(raw_vals - raw_mean))),
            )
        )
        for k in keys:
            m = meta_by_key[k]
            mir = _mirror_key(k)
            jm = result_by_key.get(mir, np.nan)
            detailed_rows.append((orbit_label, m, result_by_key[k], raw_by_key[k], jm))

    header = f"{'Shell':<6} {'Orbit':<8} {'Deg.':<6} {'Neighbor':<13} {'Dist (A)':<10} {'J (meV)':<14} {'Raw std':<10} {'Raw max dev':<12}"
    sep = "-" * 94
    with open(output_path, "w") as f:
        f.write("# Exchange Coupling (J) Results - EPR direct H(k)\n")
        f.write(f"# EPR up: {os.path.abspath(args.epr_up)}\n")
        f.write(f"# EPR dn: {os.path.abspath(args.epr_dn)}\n")
        f.write(f"# H unit interpreted as: {args.hr_unit}\n")
        f.write("# Neighbour source: EPR basic_data/at,tau with alat conversion\n")
        f.write(f"# K-mesh: {tuple(args.kmesh)}, Energy pts: {int(args.empoints)}, integrator={args.integrator}\n")
        f.write(f"# nk={int(nk)} nE={int(nE)} nproc={int(exe_info.get('nproc', 1))} n_chunks={int(exe_info.get('n_chunks', 1))} elapsed_s={elapsed:.2f}\n")
        f.write(f"# mag_atoms_global_0based={mag_atoms}\n")
        f.write(f"# slices={ {int(k): (v.start, v.stop) for k, v in slices.items()} }\n")
        if orbit_symmetry is not None:
            f.write(
                f"# orbit_symmetry={orbit_symmetry.policy} "
                f"applied={str(bool(orbit_symmetry.applied)).lower()} "
                f"tolerance_meV={orbit_symmetry.tolerance_mev:.12e} "
                f"grouping_source={orbit_grouping.source}\n"
            )
        f.write(f"# n_bonds={len(pair_meta)} n_orbits={len(orbit_rows)}\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        print("\n" + header)
        print(sep)
        for shell, label, deg, neigh, dist, avg, std, max_dev in orbit_rows:
            line = f"{shell:<6} {label:<8} {deg:<6} {neigh:<13} {dist:<10.4f} {avg:<14.6f} {std:<10.3e} {max_dev:<12.3e}"
            f.write(line + "\n")
            print(line)

        f.write("\n\n# Detailed Directed Bond List (all selected bonds, no omission)\n")
        f.write(
            f"{'Orbit':<8} {'Bond':<15} {'Mirror bond':<18} "
            f"{'Distance (A)':<15} {'R vector':<15} {'J (meV)':<14} "
            f"{'Raw J':<14} {'Delta':<14} {'Mirror J':<14} {'Diff':<14}\n"
        )
        f.write("-" * 154 + "\n")
        detailed_rows.sort(
            key=lambda x: (
                int(x[1]["shell"]),
                float(x[1]["dist"]),
                int(x[1]["gi"]),
                int(x[1]["gj"]),
                tuple(x[1]["R"]),
            )
        )
        for orbit_label, m, jv, raw_jv, jm in detailed_rows:
            bond = f"{_atom_name(m['li'], m['gi'], labels)}-{_atom_name(m['lj'], m['gj'], labels)}"
            key = (int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"]))
            mir = _mirror_key(key)
            diff = jv - jm if np.isfinite(jm) else np.nan
            f.write(
                f"{orbit_label:<8} {bond:<15} {_format_bond(*mir, labels=labels):<18} {float(m['dist']):<15.6f} "
                f"{_format_r(m['R']):<15} {jv:<14.8f} {raw_jv:<14.8f} "
                f"{jv - raw_jv:<14.8e} {jm:<14.8f} {diff:<14.8e}\n"
            )

    tsv_path = os.path.splitext(output_path)[0] + ".all_bonds.tsv"
    with open(tsv_path, "w") as f:
        f.write(
            "shell\torbit\tgi\tgj\tR1\tR2\tR3\tdist_A\tJ_meV\tJ_raw_meV\t"
            "projection_delta_meV\tmirror_gi\tmirror_gj\tmirror_R1\tmirror_R2\t"
            "mirror_R3\tmirror_J_meV\tdiff_meV\n"
        )
        for orbit_label, m, jv, raw_jv, jm in detailed_rows:
            key = (int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"]))
            mir = _mirror_key(key)
            diff = jv - jm if np.isfinite(jm) else np.nan
            f.write(
                f"{int(m['shell'])}\t{orbit_label}\t{key[0]}\t{key[1]}\t{key[2][0]}\t{key[2][1]}\t{key[2][2]}\t"
                f"{float(m['dist']):.12e}\t{jv:.12e}\t{raw_jv:.12e}\t{jv - raw_jv:.12e}\t"
                f"{mir[0]}\t{mir[1]}\t{mir[2][0]}\t{mir[2][1]}\t{mir[2][2]}\t"
                f"{jm:.12e}\t{diff:.12e}\n"
            )
    return tsv_path


def _parse_debug_bond(text):
    if not text:
        return None
    vals = [int(x.strip()) for x in str(text).replace(":", ",").split(",") if x.strip()]
    if len(vals) != 5:
        raise ValueError("--debug_bond must be gi,gj,R1,R2,R3")
    return vals[0], vals[1], (vals[2], vals[3], vals[4])


def _write_debug_pairs(args, out_dir, pair_meta, j_mev, trace_acc, kdata, labels=None):
    shell = args.debug_shell
    bond = _parse_debug_bond(args.debug_bond)
    if shell is None and bond is None:
        return None
    selected = []
    for ip, m in enumerate(pair_meta):
        ok = True
        if shell is not None:
            ok = ok and int(m["shell"]) == int(shell)
        if bond is not None:
            ok = ok and (int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"])) == bond
        if ok:
            selected.append((ip, m))
    if not selected:
        print(f"[J-epr-kspace][debug] no pairs matched debug_shell={shell} debug_bond={bond}")
        return None

    path = os.path.join(out_dir, args.debug_out or "J_debug_pairs.tsv")
    with open(path, "w") as f:
        f.write(
            "ip\tshell\tgi\tgj\tli\tlj\tlabel_i\tlabel_j\tR1\tR2\tR3\tdist_A\tJ_meV\t"
            "trace_re\ttrace_im\tdelta_i_trace_re\tdelta_j_trace_re\tdelta_i_frob\tdelta_j_frob\t"
            "delta_i_eigs_re\tdelta_j_eigs_re\n"
        )
        for ip, m in selected:
            li = int(m["li"])
            lj = int(m["lj"])
            Di = np.asarray(kdata["delta"][li], dtype=np.complex128)
            Dj = np.asarray(kdata["delta"][lj], dtype=np.complex128)
            eig_i = np.linalg.eigvalsh(0.5 * (Di + Di.conj().T)).real
            eig_j = np.linalg.eigvalsh(0.5 * (Dj + Dj.conj().T)).real
            gi = int(m["gi"])
            gj = int(m["gj"])
            ilab = labels[gi] if labels is not None and gi < len(labels) else f"Atom{gi + 1}"
            jlab = labels[gj] if labels is not None and gj < len(labels) else f"Atom{gj + 1}"
            tr = trace_acc[ip]
            f.write(
                f"{ip}\t{int(m['shell'])}\t{gi}\t{gj}\t{li}\t{lj}\t{ilab}\t{jlab}\t"
                f"{int(m['R'][0])}\t{int(m['R'][1])}\t{int(m['R'][2])}\t{float(m['dist']):.12e}\t"
                f"{float(j_mev[ip]):.12e}\t{float(np.real(tr)):.12e}\t{float(np.imag(tr)):.12e}\t"
                f"{float(np.real(np.trace(Di))):.12e}\t{float(np.real(np.trace(Dj))):.12e}\t"
                f"{float(np.linalg.norm(Di)):.12e}\t{float(np.linalg.norm(Dj)):.12e}\t"
                f"{','.join(f'{x:.8e}' for x in eig_i)}\t{','.join(f'{x:.8e}' for x in eig_j)}\n"
            )
    print(f"[J-epr-kspace][debug] wrote {path} rows={len(selected)}")
    pretty = os.path.splitext(path)[0] + ".pretty.txt"
    with open(pretty, "w") as f:
        f.write("# LKAG selected-pair debug\n")
        f.write(f"# debug_shell={shell} debug_bond={bond}\n\n")
        for ip, m in selected:
            li = int(m["li"])
            lj = int(m["lj"])
            Di = np.asarray(kdata["delta"][li], dtype=np.complex128)
            Dj = np.asarray(kdata["delta"][lj], dtype=np.complex128)
            eig_i = np.linalg.eigvalsh(0.5 * (Di + Di.conj().T)).real
            eig_j = np.linalg.eigvalsh(0.5 * (Dj + Dj.conj().T)).real
            gi = int(m["gi"])
            gj = int(m["gj"])
            ilab = labels[gi] if labels is not None and gi < len(labels) else f"Atom{gi + 1}"
            jlab = labels[gj] if labels is not None and gj < len(labels) else f"Atom{gj + 1}"
            tr = trace_acc[ip]
            f.write(f"[pair {ip}] shell={int(m['shell'])} {ilab}({gi},local {li}) -> {jlab}({gj},local {lj})\n")
            f.write(f"  R={_format_r(m['R'])}  dist_A={float(m['dist']):.8f}\n")
            f.write(f"  J_meV={float(j_mev[ip]): .10e}\n")
            f.write(f"  raw_trace={float(np.real(tr)): .10e} + {float(np.imag(tr)): .10e}i\n")
            f.write("  Delta_i:\n")
            f.write(f"    trace_re={float(np.real(np.trace(Di))): .10e}  frob={float(np.linalg.norm(Di)): .10e}\n")
            f.write("    eig_re=[" + ", ".join(f"{x: .6e}" for x in eig_i) + "]\n")
            f.write("    diag_re=[" + ", ".join(f"{x: .6e}" for x in np.real(np.diag(Di))) + "]\n")
            f.write("  Delta_j:\n")
            f.write(f"    trace_re={float(np.real(np.trace(Dj))): .10e}  frob={float(np.linalg.norm(Dj)): .10e}\n")
            f.write("    eig_re=[" + ", ".join(f"{x: .6e}" for x in eig_j) + "]\n")
            f.write("    diag_re=[" + ", ".join(f"{x: .6e}" for x in np.real(np.diag(Dj))) + "]\n")
            f.write("\n")
    print(f"[J-epr-kspace][debug] wrote {pretty}")
    return path


def run(args, comm=None):
    start = time.time()
    species_labels = [x.strip() for x in str(args.species_labels).split(",") if x.strip()] if args.species_labels else None
    labels = _atom_labels(args.epr_up, args.atom_labels, species_labels=species_labels)
    kpts = _full_k_mesh(args.kmesh)
    print(f"[J-epr-kspace] building direct H(k): nk={len(kpts)} kmesh={tuple(args.kmesh)}")
    hk_up = _build_hk_from_epr(args.epr_up, kpts, unit=args.hr_unit)
    hk_dn = _build_hk_from_epr(args.epr_dn, kpts, unit=args.hr_unit)
    if hk_up.shape != hk_dn.shape:
        raise ValueError(f"up/down H(k) shape mismatch: {hk_up.shape} vs {hk_dn.shape}")

    slices = _load_slices(args, int(hk_up.shape[1]))
    mag_atoms = _normalize_mag_atoms(args)
    neighbours = _find_nearest_neighbours_from_epr(
        args.epr_up,
        mag_atom_indices=mag_atoms,
        n_shells=int(args.n_shells),
        d_max=float(args.d_max),
        all_bonds=True,
    )
    if not neighbours:
        raise RuntimeError("No EPR neighbour bonds found.")

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
                "li": int(global_to_local[gi]),
                "lj": int(global_to_local[gj]),
                "R": tuple(int(x) for x in n["R"]),
                "dist": float(n["distance"]),
                "shell": int(n.get("shell_idx", 0)),
            }
        )
    if len(pair_meta) != len(neighbours):
        raise RuntimeError(f"Internal bond filtering mismatch: neighbours={len(neighbours)} pair_meta={len(pair_meta)}")

    if args.integrator == "contour":
        energy_mesh = get_semicircle_contour(emin=args.emin, emax=0.0, npoints=args.empoints)
    elif args.integrator == "cfr_ozaki":
        energy_mesh = get_cfr_ozaki_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)
    else:
        energy_mesh = get_cfr_pole_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)

    rank, size = rank_size(comm)
    local_energy_mesh = partition_sequence(energy_mesh, comm)
    print(
        f"[J-epr-kspace] computing J for {len(pair_meta)} directed bonds, nE={len(energy_mesh)} local_nE={len(local_energy_mesh)} mpi={size}",
        flush=True,
    )
    local_state = {}

    def integrate_local():
        j_local, trace_local, kdata_local, _info = _compute_j_direct(
            hk_up,
            hk_dn,
            slices,
            pair_meta,
            kpts,
            local_energy_mesh,
            args.efermi,
            nproc=int(args.nproc) if size == 1 else 1,
        )
        local_state["kdata"] = kdata_local
        return j_local, trace_local

    reduced = collective_sum(comm, integrate_local)
    if rank != 0:
        return
    if reduced is None:  # pragma: no cover - defensive communicator guard
        raise RuntimeError("MPI root did not receive scalar J reduction")
    j_raw_mev, trace_acc = reduced
    kdata = local_state["kdata"]
    exe_info = {
        "n_chunks": max(1, size),
        "nproc": size if size > 1 else int(args.nproc),
        "mpi_size": size,
    }
    grouping = _group_orbits_epr_with_provenance(
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
    orbit_symmetry = _apply_orbit_symmetry(
        pair_meta,
        j_raw_mev,
        grouping,
        policy=args.orbit_symmetry,
        tolerance_mev=args.orbit_symmetry_tolerance_mev,
    )
    j_mev = orbit_symmetry.values_mev
    max_residual = float(np.max(orbit_symmetry.max_abs_deviation_raw_mev, initial=0.0))
    print(
        "[J-epr-kspace] orbit symmetry: "
        f"policy={orbit_symmetry.policy} source={grouping.source} "
        f"n_orbits={len(orbit_symmetry.labels)} "
        f"max_raw_deviation={max_residual:.6e} meV",
        flush=True,
    )
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, args.out_name)
    elapsed = time.time() - start
    tsv_path = _write_outputs(
        args,
        output_path,
        neighbours,
        pair_meta,
        j_mev,
        grouping.orbits,
        mag_atoms,
        slices,
        elapsed,
        len(kpts),
        len(energy_mesh),
        exe_info,
        labels=labels,
        raw_j_mev=orbit_symmetry.raw_values_mev,
        orbit_symmetry=orbit_symmetry,
        orbit_grouping=grouping,
    )
    _write_debug_pairs(
        args,
        out_dir,
        pair_meta,
        orbit_symmetry.raw_values_mev,
        trace_acc,
        kdata,
        labels=labels,
    )
    if not args.no_h5:
        if args.out_h5:
            h5_path = args.out_h5
            if not os.path.isabs(h5_path):
                h5_path = os.path.join(out_dir, h5_path)
        else:
            h5_path = os.path.splitext(output_path)[0] + ".Jr.h5"
        _write_h5(
            h5_path,
            args,
            labels,
            pair_meta,
            j_mev,
            grouping.orbits,
            elapsed,
            len(kpts),
            len(energy_mesh),
            exe_info,
            raw_j_mev=orbit_symmetry.raw_values_mev,
            orbit_symmetry=orbit_symmetry,
            orbit_grouping=grouping,
        )
        print(f"[J-epr-kspace] wrote {h5_path}")
    print(f"[J-epr-kspace] wrote {output_path}")
    print(f"[J-epr-kspace] wrote {tsv_path}")


def main():
    ap = argparse.ArgumentParser(description="Direct EPR H(k) bond-resolved LKAG J calculator")
    ap.add_argument("--epr_up", required=True)
    ap.add_argument("--epr_dn", required=True)
    ap.add_argument(
        "--atom_labels",
        default="",
        help="Comma-separated labels for EPR atoms; default Atom1,Atom2,...",
    )
    ap.add_argument(
        "--species_labels",
        default="",
        help="Comma-separated species for spglib orbit grouping, e.g. Mn,Mn,Te,Te.",
    )
    ap.add_argument("--hr_unit", choices=["ev", "ry", "ha"], default="ry")
    ap.add_argument("--mag_atoms", type=int, nargs="+", required=True)
    ap.add_argument("--mag_atoms_base", type=int, choices=[0, 1], default=0)
    ap.add_argument("--slices", required=True, help="Local slices as '0:0:5,1:5:10'")
    ap.add_argument("--efermi", type=float, required=True)
    ap.add_argument("--kmesh", type=int, nargs=3, required=True)
    ap.add_argument("--n_shells", type=int, default=10)
    ap.add_argument("--d_max", type=float, default=20.0)
    ap.add_argument("--emin", type=float, default=-25.0)
    ap.add_argument("--empoints", type=int, default=500)
    ap.add_argument("--integrator", choices=["contour", "cfr", "cfr_ozaki"], default="contour")
    ap.add_argument("--cfr_beta", type=float, default=400.0)
    ap.add_argument("--nproc", type=int, default=1, help="Parallel contour chunks/processes")
    ap.add_argument(
        "--symprec",
        type=float,
        default=1.0e-4,
        help="spglib symmetry tolerance for orbit grouping.",
    )
    ap.add_argument(
        "--angle_tolerance",
        type=float,
        default=-1.0,
        help="spglib angle tolerance in degrees; -1 uses spglib default.",
    )
    ap.add_argument(
        "--orbit_grouping",
        choices=["spglib", "shell"],
        default="spglib",
        help="Orbit grouping mode. shell groups all bonds with the same shell index and distance.",
    )
    ap.add_argument(
        "--orbit_symmetry",
        choices=["report", "project", "fail"],
        default="project",
        help=("Treatment of scalar J within each spglib orbit: report raw values, project to the orbit mean, or fail above the requested tolerance."),
    )
    ap.add_argument(
        "--orbit_symmetry_tolerance_mev",
        type=float,
        default=1.0e-8,
        help="Maximum raw within-orbit deviation accepted by orbit_symmetry=fail.",
    )
    ap.add_argument(
        "--debug_orbits",
        action="store_true",
        help="Print spglib operation and bond-mapping diagnostics.",
    )
    ap.add_argument(
        "--debug_orbit_shell",
        type=int,
        default=None,
        help="Restrict --debug_orbits bond diagnostics to one shell.",
    )
    ap.add_argument(
        "--debug_epr_positions",
        action="store_true",
        help="Print EPR tau and Wannier-center position diagnostics.",
    )
    ap.add_argument(
        "--debug_shell",
        type=int,
        default=None,
        help="Write per-bond LKAG debug TSV for one shell, e.g. 2 for J2.",
    )
    ap.add_argument(
        "--debug_bond",
        default="",
        help="Write debug for one directed bond: gi,gj,R1,R2,R3.",
    )
    ap.add_argument(
        "--debug_out",
        default="J_debug_pairs.tsv",
        help="Debug TSV filename inside --out_dir.",
    )
    ap.add_argument(
        "--no_symmetry_orbits",
        action="store_true",
        help="Fallback to distance/pair orbit grouping.",
    )
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--out_name", default="J_epr_kspace.txt")
    ap.add_argument(
        "--out_h5",
        default=None,
        help="Optional J_r HDF5 filename/path. Default: <out_name basename>.Jr.h5",
    )
    ap.add_argument("--no_h5", action="store_true", help="Disable J_r HDF5 output.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
