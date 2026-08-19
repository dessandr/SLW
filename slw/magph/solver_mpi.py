import argparse
import json
import os
import time

import numpy as np

from slw.core.cli_paths import resolve_out_path, resolve_path, resolve_workdir
from slw.magph import kernels as internal_kernels
from slw.magph.backend import load_legacy_backend
from slw.magph.input_parser import parse_input_file
from slw.magph.lswt import parse_spin_pattern
from slw.magph.numerics import bosonic_metric_diag, compute_self_energy_at_frequencies
from slw.magph.runtime import (
    build_runtime_payload,
    build_shapes,
    estimate_solver_memory_gb,
    filter_exchange_shells,
    load_djdu_payload,
    load_djdu_tuple,
    load_legacy_j_tuple,
    load_manifest,
    recommend_solver_partition,
)
from slw.magph.scattering import (
    build_atomic_gauge_lambda,
    filter_dj_payload_bonds,
    g_kq_loop_atomic_gauge,
)
from slw.magph.utils import parsing_POSCAR


def _mpi_context():
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm, int(comm.Get_rank()), int(comm.Get_size())
    except ImportError as exc:
        env_size = int(
            os.environ.get("OMPI_COMM_WORLD_SIZE")
            or os.environ.get("PMI_SIZE")
            or os.environ.get("PMIX_SIZE")
            or "1"
        )
        if env_size > 1:
            raise RuntimeError("mpirun execution requires mpi4py for rank splitting") from exc
        return None, 0, 1



def _strip_inline_comment(line):
    return line.split("#", 1)[0].split("!", 1)[0].strip()


def _is_float_token(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def _default_kpath():
    sym_points = {
        r"$\Gamma$": [0.0, 0.0, 0.0],
        "K": [1.0 / 3.0, 1.0 / 3.0, 0.0],
        "M": [0.5, 0.0, 0.0],
        "A": [0.0, 0.0, 0.5],
        "H": [1.0 / 3.0, 1.0 / 3.0, 0.5],
        "L": [0.5, 0.0, 0.5],
        "L'": [0.0, 0.5, 0.5],
    }
    labels = ["L'", r"$\Gamma$", "L", "A", r"$\Gamma$", "M", "K", r"$\Gamma$"]
    return [{"label": label, "frac": sym_points[label], "npoints": None} for label in labels]


def _parse_kpoints_crystal_b(path, default_npoints):
    if not path:
        return None
    with open(path, "r") as f:
        lines = f.readlines()

    idx = None
    unit = ""
    for iline, raw in enumerate(lines):
        clean = _strip_inline_comment(raw)
        if not clean:
            continue
        parts = clean.replace("{", " ").replace("}", " ").replace("(", " ").replace(")", " ").split()
        if parts and parts[0].upper() == "K_POINTS":
            idx = iline
            unit = parts[1].lower() if len(parts) > 1 else ""
            break
    if idx is None or unit != "crystal_b":
        return None

    cursor = idx + 1
    while cursor < len(lines) and not _strip_inline_comment(lines[cursor]):
        cursor += 1
    if cursor >= len(lines):
        raise ValueError(f"K_POINTS crystal_b in {path} is missing the point count line")
    npoints_line = _strip_inline_comment(lines[cursor]).split()
    nlabels = int(npoints_line[0])
    cursor += 1

    records = []
    while cursor < len(lines) and len(records) < nlabels:
        clean = _strip_inline_comment(lines[cursor])
        cursor += 1
        if not clean:
            continue
        parts = clean.split()
        if _is_float_token(parts[0]):
            if len(parts) < 4:
                raise ValueError(f"Invalid K_POINTS crystal_b line in {path}: {clean}")
            label = parts[4] if len(parts) >= 5 else f"K{len(records)}"
            frac = [float(parts[0]), float(parts[1]), float(parts[2])]
            seg_points = int(parts[3])
        else:
            if len(parts) < 5:
                raise ValueError(f"Invalid K_POINTS crystal_b line in {path}: {clean}")
            label = parts[0]
            frac = [float(parts[1]), float(parts[2]), float(parts[3])]
            seg_points = int(parts[4])
        records.append({"label": label, "frac": frac, "npoints": seg_points})
    if len(records) != nlabels:
        raise ValueError(f"K_POINTS crystal_b in {path} declares {nlabels} points but {len(records)} were read")
    if len(records) < 2:
        raise ValueError("K_POINTS crystal_b needs at least two path points")
    for rec in records:
        if rec["npoints"] is None:
            rec["npoints"] = int(default_npoints)
    return records


def _build_kpath_from_records(records, g_vec, default_npoints):
    k_list = []
    distances = []
    tick_indices = []
    current_dist = 0.0
    for iseg in range(len(records) - 1):
        start = np.asarray(records[iseg]["frac"], dtype=np.float64)
        end = np.asarray(records[iseg + 1]["frac"], dtype=np.float64)
        nseg = max(1, int(records[iseg].get("npoints") or default_npoints))
        tick_indices.append(len(k_list))
        segment = np.linspace(start, end, nseg, endpoint=False)
        for kpt in segment:
            k_list.append(kpt)
            if len(k_list) > 1:
                current_dist += float(np.linalg.norm((k_list[-1] - k_list[-2]) @ g_vec))
            distances.append(current_dist)
    k_list.append(np.asarray(records[-1]["frac"], dtype=np.float64))
    if len(k_list) > 1:
        current_dist += float(np.linalg.norm((k_list[-1] - k_list[-2]) @ g_vec))
    distances.append(current_dist)
    tick_indices.append(len(k_list) - 1)
    return (
        np.asarray(k_list, dtype=np.float64),
        np.asarray(distances, dtype=np.float64),
        np.asarray(tick_indices, dtype=np.int64),
        [str(rec["label"]) for rec in records],
    )


def _kpath_records_from_cfg(cfg):
    default_npoints = int(cfg.get("band_points", 101))
    records = _parse_kpoints_crystal_b(cfg.get("_input_file_path"), default_npoints)
    if records is not None:
        return records
    records = _default_kpath()
    for rec in records[:-1]:
        rec["npoints"] = default_npoints
    records[-1]["npoints"] = 1
    return records


def _estimate_k_tasks_from_cfg(cfg):
    records = _kpath_records_from_cfg(cfg)
    return int(sum(max(1, int(rec.get("npoints") or cfg.get("band_points", 101))) for rec in records[:-1]) + 1)


def _resolve_structure_path(cfg):
    path = cfg.get("POSCAR") or cfg.get("structure_file") or cfg.get("_input_file_path")
    if not path:
        raise ValueError(
            "Structure is required for execute mode. Set POSCAR=... or structure_file=..., "
            "or include CELL_PARAMETERS and ATOMIC_POSITIONS cards in --input_file."
        )
    if os.path.isabs(path):
        return path
    return resolve_path(cfg.get("workdir", "."), path)


def _normalize_mag_order(cfg):
    raw = cfg.get("mag_order", cfg.get("magnetic_order", None))
    if raw is None:
        spin_raw = str(cfg.get("spin_pattern", "auto")).strip().lower()
        raw = "fm" if spin_raw in {"fm", "ferro", "ferromagnetic"} else "afm_bipartite"
    text = str(raw).strip().lower()
    aliases = {
        "afm": "afm_bipartite",
        "afm_bipartite": "afm_bipartite",
        "bipartite": "afm_bipartite",
        "fm": "fm",
        "ferro": "fm",
        "ferromagnetic": "fm",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported mag_order={raw!r}. Supported: afm_bipartite, fm")
    return aliases[text]


def _regularize_phonon_energies(ph_en_flat, rank=0):
    neg_mask = ph_en_flat < 0.0
    if not np.any(neg_mask):
        return ph_en_flat

    min_negative = float(np.min(ph_en_flat[neg_mask]))
    count_negative = int(np.count_nonzero(neg_mask))
    ph_en_regularized = ph_en_flat.copy()
    ph_en_regularized[neg_mask] = 0.0
    if rank == 0:
        print(
            "[magph:solver][WARN] clipped "
            f"{count_negative} negative phonon energies to 0.0 meV "
            f"(min={min_negative:.6g} meV)",
            flush=True,
        )
    return ph_en_regularized


def _canonical_bonds_for_fm(j0):
    jj, ii, kk, rr, dd = j0
    accum = {}
    for ib in range(len(jj)):
        i = int(ii[ib])
        j = int(kk[ib])
        r = tuple(int(x) for x in rr[ib])
        mirror = (j, i, tuple(-x for x in r))
        direct = (i, j, r)
        key = direct if str(direct) <= str(mirror) else mirror
        val = accum.setdefault(key, [0.0, 0.0, 0])
        val[0] += float(jj[ib])
        val[1] += float(dd[ib])
        val[2] += 1

    out_j, out_i, out_k, out_r, out_d = [], [], [], [], []
    for (i, j, r), (jsum, dsum, count) in accum.items():
        out_j.append(jsum / count)
        out_i.append(i)
        out_k.append(j)
        out_r.append(r)
        out_d.append(dsum / count)
    return (
        np.asarray(out_j, dtype=np.float64),
        np.asarray(out_i, dtype=np.int32),
        np.asarray(out_k, dtype=np.int32),
        np.asarray(out_r, dtype=np.int32),
        np.asarray(out_d, dtype=np.float64),
    )


def _prepare_context(manifest, cfg, kernel_source="kernels", rank=0):
    legacy_dir = cfg.get("legacy_magph_dir", None)
    if legacy_dir is None:
        legacy_dir = os.path.join(cfg.get("workdir", "."), "magph")
    backend = load_legacy_backend(legacy_dir, kernel_source=kernel_source)
    utils = backend["utils"]
    kernels = backend["kernels"]

    structure_path = _resolve_structure_path(cfg)
    R_vec, atom_list, atom_pos = parsing_POSCAR(structure_path)
    atom_pos = np.array(atom_pos, dtype=np.float64)
    R_vec = np.array(R_vec, dtype=np.float64)
    G_vec = 2 * np.pi * np.linalg.inv(R_vec).T

    j0 = load_legacy_j_tuple(manifest.legacy_j_npz)
    dJdu = load_djdu_tuple(manifest)
    dj_payload = load_djdu_payload(manifest)
    j0, dJdu, shell_filter = filter_exchange_shells(j0, dJdu, cfg)
    if rank == 0 and shell_filter.get("enabled", False):
        print(
            "[magph:solver] shell filter: "
            f"exclude={shell_filter['exclude_shells']} apply={shell_filter['apply']} "
            f"J {shell_filter['j_bonds_before']}->{shell_filter['j_bonds_after']} "
            f"dJ {shell_filter['dj_entries_before']}->{shell_filter['dj_entries_after']}",
            flush=True,
        )
    nmag = int(max(np.max(j0[1]), np.max(j0[2])) + 1) if len(j0[0]) else 2
    mag_order = _normalize_mag_order(cfg)
    if mag_order == "fm":
        j0 = _canonical_bonds_for_fm(j0)
    else:
        # Atomic-gauge V_H is the derivative of the in-tree BdG kernel; an
        # external legacy eigensolver would break that sign/counting identity.
        kernels = internal_kernels
    spin_pattern_default = "fm" if mag_order == "fm" else "auto"
    spin_pattern = parse_spin_pattern(cfg.get("spin_pattern", spin_pattern_default), nmag)

    ph_cache = np.load(manifest.phonon_cache, allow_pickle=True)
    if "q_mesh_flat_cart" in ph_cache:
        q_mesh_flat_cart = np.array(ph_cache["q_mesh_flat_cart"], dtype=np.float64)
        if "q_mesh_flat_frac" in ph_cache:
            q_mesh_flat_frac = np.array(ph_cache["q_mesh_flat_frac"], dtype=np.float64) % 1.0
        else:
            q_mesh_flat_frac = np.mod(q_mesh_flat_cart @ np.linalg.inv(G_vec), 1.0)
        ph_en_flat = np.array(ph_cache["ph_en_flat"], dtype=np.float64)
        ph_vec_flat = np.array(ph_cache["ph_vec_flat"], dtype=np.complex128)
    else:
        q_mesh_flat_cart = np.array(ph_cache["q_mesh_flat"], dtype=np.float64)
        q_mesh_flat_frac = np.mod(q_mesh_flat_cart @ np.linalg.inv(G_vec), 1.0)
        ph_en_flat = np.array(ph_cache["ph_en_flat"], dtype=np.float64)
        ph_vec_flat = np.array(ph_cache["ph_vec_flat"], dtype=np.complex128)
    ph_en_flat = _regularize_phonon_energies(ph_en_flat, rank=rank)
    total_q = int(q_mesh_flat_cart.shape[0])

    exp_iqd = None
    lambda_qnu_b = None
    dj_pair_R = None
    vertex_report = {"formalism": "legacy_fm"}
    if mag_order == "fm":
        n_bonds = len(dJdu[3])
        delta_bonds = np.zeros((n_bonds, 3), dtype=np.float64)
        for idx in range(n_bonds):
            i_idx = int(dJdu[1][idx])
            j_idx = int(dJdu[2][idx])
            delta_bonds[idx] = atom_pos[j_idx] - atom_pos[i_idx] + np.array(dJdu[3][idx], dtype=np.float64) @ R_vec
        exp_iqd = kernels.precompute_phase_factors(q_mesh_flat_cart, delta_bonds)
    else:
        allowed_dj_keys = {
            (int(i), int(j), tuple(int(x) for x in r))
            for i, j, r in zip(dJdu[1], dJdu[2], dJdu[3])
        }
        dj_payload = filter_dj_payload_bonds(dj_payload, allowed_dj_keys)
        atom_frac = atom_pos @ np.linalg.inv(R_vec)
        lambda_qnu_b, vertex_report = build_atomic_gauge_lambda(
            dj_payload,
            q_mesh_flat_frac,
            ph_en_flat,
            ph_vec_flat,
            atom_frac,
            phonon_floor_mev=float(cfg.get("phonon_floor_mev", 1.0e-3)),
            asr_mode=str(cfg.get("dJ_asr", "check")),
            asr_tolerance=float(cfg.get("dJ_asr_tolerance", 1.0e-8)),
            q_chunk=int(cfg.get("vertex_q_chunk", 32)),
            bond_chunk=int(cfg.get("vertex_bond_chunk", 64)),
        )
        dj_pair_R = np.asarray(dj_payload["pair_R"], dtype=np.int32)
        if rank == 0:
            asr = vertex_report["asr"]
            print(
                "[magph:solver] atomic-gauge vertex: "
                f"Rp={vertex_report['n_rp']} targets={vertex_report['n_targets']} "
                f"bonds={vertex_report['n_bonds']} dJ-ASR({asr['mode']}) "
                f"max={asr['max_abs_before']:.6g}->{asr['max_abs_after']:.6g} meV/A",
                flush=True,
            )
            if vertex_report["single_rp_legacy_input"]:
                print(
                    "[magph:solver][WARN] only a legacy single-Rp derivative payload is available; "
                    "use the original dJr HDF5/multi-Rp NPZ for the complete atomic-gauge sum",
                    flush=True,
                )
            if not vertex_report["target_coverage_complete"]:
                print(
                    "[magph:solver][WARN] dJ derivatives do not cover every phonon atom; "
                    "the exchange-derivative ASR and acoustic-q limit are incomplete",
                    flush=True,
                )

    band_points = int(cfg.get("band_points", 101))
    kpath_records = _kpath_records_from_cfg(cfg)
    kpts_frac, kdist, tick_indices, kpath_symbols = _build_kpath_from_records(kpath_records, G_vec, band_points)
    kpts_frac = np.array(kpts_frac, dtype=np.float64)
    kpts_cart = kpts_frac @ G_vec
    exp_ikd = None
    if mag_order == "fm":
        exp_ikd = kernels.precompute_phase_factors(kpts_cart, delta_bonds)

    ctx = {
        "utils": utils,
        "kernels": kernels,
        "R_vec": R_vec,
        "G_vec": G_vec,
        "atom_list": atom_list,
        "atom_pos": atom_pos,
        "j0": j0,
        "dJdu": dJdu,
        "mag_order": mag_order,
        "spin_pattern": spin_pattern,
        "q_mesh_flat_cart": q_mesh_flat_cart,
        "q_mesh_flat_frac": q_mesh_flat_frac,
        "ph_en_flat": ph_en_flat,
        "ph_vec_flat": ph_vec_flat,
        "total_q": total_q,
        "exp_iqd": exp_iqd,
        "lambda_qnu_b": lambda_qnu_b,
        "dj_pair_R": dj_pair_R,
        "vertex_report": vertex_report,
        "kpts_frac": kpts_frac,
        "kpts_cart": kpts_cart,
        "kdist": np.array(kdist, dtype=np.float64),
        "kpath_tick_indices": tick_indices,
        "exp_ikd": exp_ikd,
        "kpath_symbols": kpath_symbols,
        "shell_filter": shell_filter,
    }
    return ctx


def _execute_solver(manifest, cfg, out_json):
    kernel_source = cfg.get("kernel_source", "kernels")
    comm, rank, size = _mpi_context()
    ctx = _prepare_context(manifest, cfg, kernel_source=kernel_source, rank=rank)
    kernels = ctx["kernels"]
    utils = ctx["utils"]
    total_k = int(ctx["kpts_cart"].shape[0])
    total_q = int(ctx["total_q"])
    mag_order = ctx["mag_order"]
    nmag = int(ctx["spin_pattern"].shape[0])

    S = float(cfg.get("S", 2.5))
    T = float(cfg.get("T", 300.0))
    eta = float(cfg.get("eta", 1.0))
    omega_points = int(cfg.get("omega_points", 5000))
    omega_max = float(cfg.get("omega_max", 50.0))
    eta_A = float(cfg.get("eta_A", 0.01))
    anisotropy_mev = float(cfg.get("anisotropy_mev", cfg.get("anisotropy", 0.0)))
    bond_factor = float(cfg.get("bond_factor", 1.0))
    omega_axis = np.linspace(0.0, omega_max, omega_points)

    local_indices = np.arange(rank, total_k, size, dtype=np.int64)
    local_k = int(local_indices.shape[0])
    matrix_dim = nmag if mag_order == "fm" else 2 * nmag
    spectral_metric_diag = (
        np.ones(matrix_dim, dtype=np.float64)
        if mag_order == "fm"
        else bosonic_metric_diag(matrix_dim, physical_count=nmag)
    )
    local_sigma = np.zeros((local_k, omega_points, matrix_dim, matrix_dim), dtype=np.complex128)
    local_magnon_energies = np.zeros((local_k, matrix_dim), dtype=np.float64)

    t0 = time.time()
    progress_interval = max(1, int(cfg.get("progress_interval", 1)))
    if rank == 0:
        print(
            f"[magph:solver] execute start: k={total_k}, q={total_q}, "
            f"omega={omega_points}, mag_order={mag_order}, matrix_dim={matrix_dim}, mpi_size={size}",
            flush=True,
        )
    print(f"[magph:solver][rank {rank}] local_k={local_k}", flush=True)
    for iloc, ik in enumerate(local_indices):
        k_vec = ctx["kpts_cart"][ik]
        if mag_order == "fm":
            e_mag_k, Uk = kernels.Magnon_Hamiltonian_fm_normal(
                k_vec,
                S,
                ctx["j0"],
                ctx["R_vec"],
                ctx["atom_pos"],
                nmag,
                anisotropy_mev,
                bond_factor,
            )
            g_pq_cache, e_kq_cache = kernels.g_kq_loop_fm_normal(
                ik=ik,
                exp_iqd=ctx["exp_iqd"],
                exp_ikd=ctx["exp_ikd"],
                total_q=total_q,
                q_mesh_flat_cart=ctx["q_mesh_flat_cart"],
                k_cart=k_vec,
                dJdu=ctx["dJdu"],
                Uk=Uk,
                ph_en_flat=ctx["ph_en_flat"],
                ph_vec_flat=ctx["ph_vec_flat"],
                S=S,
                J0=ctx["j0"],
                R_vec=ctx["R_vec"],
                G_vec=ctx["G_vec"],
                atom_pos=ctx["atom_pos"],
                nmag=nmag,
                anisotropy=anisotropy_mev,
                bond_factor=bond_factor,
            )
            metric_diag = np.ones(matrix_dim, dtype=np.float64)
        else:
            e_mag_k, Uk = kernels.Magnon_Hamiltonian_v1(
                k_vec,
                S,
                ctx["j0"],
                ctx["R_vec"],
                ctx["atom_pos"],
                ctx["spin_pattern"],
                bond_factor,
                anisotropy_mev,
            )
            g_pq_cache, e_kq_cache = g_kq_loop_atomic_gauge(
                q_mesh_flat_cart=ctx["q_mesh_flat_cart"],
                k_cart=k_vec,
                lambda_qnu_b=ctx["lambda_qnu_b"],
                pair_R=ctx["dj_pair_R"],
                Uk=Uk,
                S=S,
                J0=ctx["j0"],
                lattice=ctx["R_vec"],
                atom_pos=ctx["atom_pos"],
                spin_pattern=ctx["spin_pattern"],
                anisotropy=anisotropy_mev,
                bond_factor=bond_factor,
            )
            metric_diag = bosonic_metric_diag(matrix_dim, physical_count=nmag)
        sigma_k = compute_self_energy_at_frequencies(
            omega_eval=omega_axis,
            g_pq_cache=g_pq_cache,
            e_kq_cache=e_kq_cache,
            ph_en_flat=ctx["ph_en_flat"],
            T=T,
            eta=eta,
            total_q=total_q,
            metric_diag=metric_diag,
        )
        local_sigma[iloc] = sigma_k
        local_magnon_energies[iloc] = e_mag_k
        done = iloc + 1
        if done % progress_interval == 0 or done == local_k:
            dt = time.time() - t0
            rate = done / max(dt, 1.0e-12)
            eta_s = (local_k - done) / max(rate, 1.0e-12)
            print(
                f"[magph:solver][rank {rank}] progress {done}/{local_k} "
                f"global_k={int(ik)+1}/{total_k} elapsed={dt:.1f}s eta={eta_s:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    A_kw = utils.calculate_spectral_function(
        local_sigma,
        local_magnon_energies,
        omega_axis,
        eta_A=eta_A,
        metric_diag=spectral_metric_diag,
    )

    base = os.path.splitext(out_json)[0]
    if size == 1:
        sigma_npz = base + ".sigma_kw.npz"
        spectral_npz = base + ".spectral.npz"
    else:
        sigma_npz = f"{base}.rank{rank:04d}.sigma_kw.npz"
        spectral_npz = f"{base}.rank{rank:04d}.spectral.npz"
    tick_indices_out = ctx["kpath_tick_indices"] if size == 1 else np.array([], dtype=np.int64)
    np.savez_compressed(
        sigma_npz,
        self_energy=local_sigma,
        omega_axis=omega_axis,
        e_mag=local_magnon_energies,
        k_indices=local_indices,
        kdist=ctx["kdist"][local_indices],
        kpts_frac=ctx["kpts_frac"][local_indices],
        kpts_cart=ctx["kpts_cart"][local_indices],
        mag_order=np.array(mag_order, dtype=object),
        matrix_dim=np.array([matrix_dim], dtype=np.int32),
        vertex_report_json=np.asarray(json.dumps(ctx["vertex_report"], sort_keys=True)),
        kpath_symbols=np.array(ctx["kpath_symbols"], dtype=object),
        kpath_tick_indices=tick_indices_out,
        kpath_tick_indices_global=ctx["kpath_tick_indices"],
    )
    np.savez_compressed(
        spectral_npz,
        A_kw=A_kw,
        omega_axis=omega_axis,
        e_mag=local_magnon_energies,
        k_indices=local_indices,
        kdist=ctx["kdist"][local_indices],
        kpts_frac=ctx["kpts_frac"][local_indices],
        kpts_cart=ctx["kpts_cart"][local_indices],
        mag_order=np.array(mag_order, dtype=object),
        matrix_dim=np.array([matrix_dim], dtype=np.int32),
        kpath_symbols=np.array(ctx["kpath_symbols"], dtype=object),
        kpath_tick_indices=tick_indices_out,
        kpath_tick_indices_global=ctx["kpath_tick_indices"],
        eta_A=np.array([eta_A], dtype=np.float64),
    )
    chunk_info = {
        "rank": rank,
        "local_k": local_k,
        "k_start": int(local_indices[0]) if local_k else None,
        "k_stop": int(local_indices[-1]) if local_k else None,
        "sigma_npz": sigma_npz,
        "spectral_npz": spectral_npz,
    }
    if comm is not None:
        from mpi4py import MPI

        chunks = comm.gather(chunk_info, root=0)
        elapsed = comm.reduce(elapsed, op=MPI.MAX, root=0)
    else:
        chunks = [chunk_info]
    if rank != 0:
        return None
    return {
        "mode": "execute",
        "kernel_source": (
            kernel_source if mag_order == "fm" else "internal_atomic_gauge"
        ),
        "total_k": total_k,
        "total_q": total_q,
        "omega_points": omega_points,
        "omega_max_meV": omega_max,
        "mag_order": mag_order,
        "matrix_dim": matrix_dim,
        "mpi_size": size,
        "runtime_s": elapsed,
        "sigma_npz": sigma_npz if size == 1 else None,
        "spectral_npz": spectral_npz if size == 1 else None,
        "chunks": chunks,
        "vertex": ctx["vertex_report"],
        "shell_filter": ctx.get("shell_filter", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="SLW magph spectral solver runner (reconstructed runtime)")
    parser.add_argument("--workdir", type=str, default=None, help="Workflow root directory (default: current directory)")
    parser.add_argument("--input_file", type=str, default=None, help="Legacy-style input file (e.g. input.in)")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest from slw.magph.prepare_lifetime")
    parser.add_argument("--mode", type=str, choices=["validate", "plan", "execute"], default="plan",
                        help="validate: contract check, plan: runtime estimates, execute: run sigma + spectral")
    parser.add_argument("--kernel_source", type=str, default=None, choices=["kernels", "kernels_lifetime"],
                        help="Kernel backend selection (default: kernels)")
    parser.add_argument("--mag_order", type=str, default=None, choices=["afm_bipartite", "afm", "fm", "ferro", "ferromagnetic"],
                        help="Magnon order/model for solver validation (FM execute is not implemented yet)")
    parser.add_argument("--omega_points", type=int, default=5000, help="Omega axis size for solver planning")
    parser.add_argument("--omega_max", type=float, default=None, help="Omega axis maximum in meV (default: input file or 50)")
    parser.add_argument("--progress_interval", type=int, default=None,
                        help="Print execute progress every N k-points (default: input file or 1)")
    parser.add_argument("--target_rank_gb", type=float, default=4.0,
                        help="Target per-rank working memory for omega chunk recommendation")
    parser.add_argument("--k_tasks", type=int, default=None, help="Number of external k tasks for runtime planning")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for runtime metadata")
    parser.add_argument("--out_name", type=str, default="magph_solver_runtime.json", help="Output metadata filename")
    args = parser.parse_args()

    workdir = resolve_workdir(args.workdir)
    cfg = {"workdir": workdir}
    if args.input_file:
        input_path = resolve_path(workdir, args.input_file)
        cfg.update(parse_input_file(input_path))
        cfg["_input_file_path"] = input_path

    manifest_arg = args.manifest if args.manifest else cfg.get("manifest")
    if not manifest_arg:
        raise ValueError("Manifest is required. Set --manifest or provide 'manifest=' in --input_file.")
    kernel_source = args.kernel_source if args.kernel_source else cfg.get("kernel_source", "kernels")
    cfg["kernel_source"] = kernel_source
    if args.mag_order is not None:
        cfg["mag_order"] = args.mag_order
    if args.progress_interval is not None:
        cfg["progress_interval"] = int(args.progress_interval)
    if kernel_source == "kernels_lifetime":
        print("[WARN] kernels_lifetime is legacy. kernels is recommended as latest backend.", flush=True)

    manifest = load_manifest(manifest_arg, workdir=workdir)
    shapes = build_shapes(manifest)
    mag_order = _normalize_mag_order(cfg)
    if mag_order == "fm":
        j0_for_dim = load_legacy_j_tuple(manifest.legacy_j_npz)
        matrix_dim = int(max(np.max(j0_for_dim[1]), np.max(j0_for_dim[2])) + 1) if len(j0_for_dim[0]) else 2
    else:
        matrix_dim = 4
    omega_points = int(args.omega_points if args.omega_points is not None else cfg.get("omega_points", 5000))
    cfg["omega_points"] = omega_points
    if args.omega_max is not None:
        cfg["omega_max"] = float(args.omega_max)
    target_rank_gb = float(args.target_rank_gb if args.target_rank_gb is not None else cfg.get("target_rank_gb", 4.0))
    mem = estimate_solver_memory_gb(shapes, omega_points=omega_points, matrix_dim=matrix_dim)
    partition = recommend_solver_partition(
        shapes,
        omega_points=omega_points,
        target_rank_gb=target_rank_gb,
        matrix_dim=matrix_dim,
    )
    if args.k_tasks is not None:
        k_tasks = int(args.k_tasks)
    else:
        k_tasks = _estimate_k_tasks_from_cfg(cfg)

    out_json = resolve_out_path(
        workdir=workdir,
        out_dir=args.out_dir,
        out_name=args.out_name,
        default_dir=".",
        default_name="magph_solver_runtime.json",
    )
    stage = "io_validated" if args.mode == "validate" else ("io_planned" if args.mode == "plan" else "executed")
    extra = {
        "kernel_source": kernel_source,
        "mag_order": mag_order,
        "matrix_dim": matrix_dim,
        "k_tasks": k_tasks,
        "omega_points": omega_points,
        "omega_max_meV": float(cfg.get("omega_max", 50.0)),
        "memory_gb": mem,
        "recommended_partition": partition,
    }
    _comm, rank, _ = _mpi_context()
    if args.mode == "execute":
        exec_info = _execute_solver(manifest, cfg, out_json=out_json)
        if rank != 0:
            return
        extra["execute"] = exec_info

    payload = build_runtime_payload(
        module_name="slw.magph.solver_mpi",
        manifest=manifest,
        shapes=shapes,
        stage=stage,
        extra=extra,
    )
    if rank == 0:
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        print("[magph:solver] Runtime contract ready.", flush=True)
        print(
            f"[magph:solver] J bonds={shapes.n_j_bonds}, "
            f"dJ entries={shapes.n_dj_entries}, qpoints={shapes.n_qpoints}",
            flush=True,
        )
        print(
            f"[magph:solver] est. working_set_per_rank={mem['working_set_per_rank_gb']:.3f} GB "
            f"(omega_points={omega_points})",
            flush=True,
        )
        print(
            f"[magph:solver] recommend omega_chunks={partition['omega_chunks']}, "
            f"omega_chunk_size={partition['omega_chunk_size']}",
            flush=True,
        )
        if args.mode == "execute":
            exec_payload = payload["analysis"]["execute"]
            if exec_payload["mpi_size"] == 1:
                print(f"[magph:solver] sigma npz: {exec_payload['sigma_npz']}", flush=True)
                print(f"[magph:solver] spectral npz: {exec_payload['spectral_npz']}", flush=True)
            else:
                print(f"[magph:solver] chunk files: {len(exec_payload['chunks'])}", flush=True)
        print(f"[magph:solver] runtime metadata: {out_json}", flush=True)


if __name__ == "__main__":
    main()
