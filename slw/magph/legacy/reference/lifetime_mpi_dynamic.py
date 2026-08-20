import argparse
import json
import os
import time

import numpy as np

from slw.core.cli_paths import resolve_out_path, resolve_path, resolve_workdir
from slw.magph.legacy import kernels as internal_kernels
from slw.magph.legacy.backend import load_legacy_backend
from slw.magph.legacy.input_parser import parse_input_file
from slw.magph.legacy.lswt import parse_spin_pattern
from slw.magph.legacy.numerics import (
    bosonic_metric_diag,
    compute_self_energy_at_frequencies,
    linewidth_observables,
    paraunitary_residual,
    retarded_damping_matrix,
)
from slw.magph.legacy.runtime import (
    build_runtime_payload,
    build_shapes,
    estimate_lifetime_memory_gb,
    filter_exchange_shells,
    load_djdu_payload,
    load_djdu_tuple,
    load_legacy_j_tuple,
    load_manifest,
)
from slw.magph.legacy.scattering import (
    build_atomic_gauge_lambda,
    filter_dj_payload_bonds,
    g_kq_loop_atomic_gauge,
)
from slw.magph.legacy.utils import parsing_POSCAR


def _mpi_context():
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm, int(comm.Get_rank()), int(comm.Get_size())
    except Exception:
        return None, 0, 1


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


def _prepare_context(manifest, cfg, kernel_source="kernels", rank=0):
    legacy_dir = cfg.get("legacy_magph_dir", None)
    if legacy_dir is None:
        legacy_dir = os.path.join(cfg.get("workdir", "."), "magph")
    backend = load_legacy_backend(legacy_dir, kernel_source=kernel_source)
    utils = backend["utils"]
    # The atomic-gauge vertex and its static BdG derivative must use the same
    # in-tree kernel implementation.  External legacy kernels do not expose
    # this contract.
    kernels = internal_kernels

    structure_path = _resolve_structure_path(cfg)
    R_vec, _atom_list, atom_pos = parsing_POSCAR(structure_path)
    atom_pos = np.array(atom_pos, dtype=np.float64)
    G_vec = 2 * np.pi * np.linalg.inv(np.array(R_vec, dtype=np.float64)).T

    j0 = load_legacy_j_tuple(manifest.legacy_j_npz)
    dJdu = load_djdu_tuple(manifest)
    dj_payload = load_djdu_payload(manifest)
    j0, dJdu, shell_filter = filter_exchange_shells(j0, dJdu, cfg)
    if rank == 0 and shell_filter.get("enabled", False):
        print(
            "[magph:lifetime] shell filter: "
            f"exclude={shell_filter['exclude_shells']} apply={shell_filter['apply']} "
            f"J {shell_filter['j_bonds_before']}->{shell_filter['j_bonds_after']} "
            f"dJ {shell_filter['dj_entries_before']}->{shell_filter['dj_entries_after']}",
            flush=True,
        )
    nmag = int(max(np.max(j0[1]), np.max(j0[2])) + 1) if len(j0[0]) else 2
    spin_pattern = parse_spin_pattern(cfg.get("spin_pattern", "auto"), nmag)

    ph_cache = np.load(manifest.phonon_cache, allow_pickle=True)
    # Through adapter schema in runtime already validated; keep compatibility:
    if "q_mesh_flat_cart" in ph_cache:
        q_mesh_flat_cart = ph_cache["q_mesh_flat_cart"]
        ph_en_flat = ph_cache["ph_en_flat"]
        ph_vec_flat = ph_cache["ph_vec_flat"]
    else:
        q_mesh_flat_cart = ph_cache["q_mesh_flat"]
        ph_en_flat = ph_cache["ph_en_flat"]
        ph_vec_flat = ph_cache["ph_vec_flat"]
    if "q_mesh_flat_frac" in ph_cache:
        q_mesh_flat_frac = np.mod(
            np.asarray(ph_cache["q_mesh_flat_frac"], dtype=np.float64), 1.0
        )
    else:
        q_mesh_flat_frac = np.mod(
            np.asarray(q_mesh_flat_cart, dtype=np.float64) @ np.linalg.inv(G_vec),
            1.0,
        )

    allowed_dj_keys = {
        (int(i), int(j), tuple(int(x) for x in r))
        for i, j, r in zip(dJdu[1], dJdu[2], dJdu[3])
    }
    dj_payload = filter_dj_payload_bonds(dj_payload, allowed_dj_keys)
    atom_frac = np.asarray(atom_pos, dtype=np.float64) @ np.linalg.inv(
        np.asarray(R_vec, dtype=np.float64)
    )
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
    if rank == 0:
        asr = vertex_report["asr"]
        print(
            "[magph:lifetime] atomic-gauge vertex: "
            f"Rp={vertex_report['n_rp']} targets={vertex_report['n_targets']} "
            f"bonds={vertex_report['n_bonds']} dJ-ASR({asr['mode']}) "
            f"max={asr['max_abs_before']:.6g}->{asr['max_abs_after']:.6g} meV/A",
            flush=True,
        )
        if asr["violated_after"]:
            print(
                "[magph:lifetime][WARN] exchange-derivative ASR is violated; "
                "set dJ_asr=project to apply the minimum-norm correction",
                flush=True,
            )
        if vertex_report["single_rp_legacy_input"]:
            print(
                "[magph:lifetime][WARN] only a legacy single-Rp derivative payload is available; "
                "use the original dJr HDF5/multi-Rp NPZ for the complete atomic-gauge sum",
                flush=True,
            )
        if not vertex_report["target_coverage_complete"]:
            print(
                "[magph:lifetime][WARN] dJ derivatives do not cover every phonon atom; "
                "the exchange-derivative ASR and acoustic-q limit are incomplete",
                flush=True,
            )

    tkm = cfg.get("target_k_mesh", [10, 10, 10])
    k_mesh_frac = utils.generate_k_mesh_flat(int(tkm[0]), int(tkm[1]), int(tkm[2]), shift=False)
    k_mesh_cart = k_mesh_frac @ G_vec

    ctx = {
        "utils": utils,
        "kernels": kernels,
        "R_vec": np.array(R_vec, dtype=np.float64),
        "G_vec": G_vec,
        "atom_pos": atom_pos,
        "j0": j0,
        "dJdu": dJdu,
        "dj_pair_R": np.asarray(dj_payload["pair_R"], dtype=np.int32),
        "lambda_qnu_b": lambda_qnu_b,
        "vertex_report": vertex_report,
        "spin_pattern": spin_pattern,
        "q_mesh_flat_cart": q_mesh_flat_cart,
        "q_mesh_flat_frac": q_mesh_flat_frac,
        "ph_en_flat": ph_en_flat,
        "ph_vec_flat": ph_vec_flat,
        "k_mesh_frac": k_mesh_frac,
        "k_mesh_cart": k_mesh_cart,
        "shell_filter": shell_filter,
    }
    return ctx


def _execute_lifetime(manifest, cfg, out_json):
    kernel_source = cfg.get("kernel_source", "kernels")
    comm, rank, size = _mpi_context()
    ctx = _prepare_context(manifest, cfg, kernel_source=kernel_source, rank=rank)

    kernels = ctx["kernels"]
    k_mesh_cart = ctx["k_mesh_cart"]
    total_k = int(k_mesh_cart.shape[0])
    total_q = int(ctx["q_mesh_flat_cart"].shape[0])
    S = float(cfg.get("S", 2.5))
    T = float(cfg.get("T", 300.0))
    eta = float(cfg.get("eta", 1.0))
    anisotropy_mev = float(cfg.get("anisotropy_mev", 0.0))
    bond_factor = float(cfg.get("bond_factor", 1.0))
    physical_count = int(ctx["spin_pattern"].shape[0])
    nchannel = 2 * physical_count
    metric_diag = bosonic_metric_diag(nchannel, physical_count=physical_count)

    linewidth_local = np.zeros((total_k, nchannel), dtype=np.float64)
    linewidth_raw_local = np.zeros((total_k, nchannel), dtype=np.float64)
    energies_local = np.zeros((total_k, nchannel), dtype=np.float64)
    sigma_onshell_local = np.zeros((total_k, nchannel, nchannel), dtype=np.complex128)
    sigma_at_channel_energies_local = np.zeros(
        (total_k, nchannel, nchannel, nchannel), dtype=np.complex128
    )
    sigma_phys_common_local = np.zeros(
        (total_k, physical_count, physical_count), dtype=np.complex128
    )
    common_energy_phys_local = np.zeros(total_k, dtype=np.float64)
    paraunitary_residual_local = np.zeros(total_k, dtype=np.float64)
    metric_energy_min_local = np.zeros(total_k, dtype=np.float64)

    t0 = time.time()
    progress_interval = max(1, int(cfg.get("progress_interval", 1)))
    local_indices = np.arange(rank, total_k, size, dtype=np.int64)
    if rank == 0:
        print(
            f"[magph:lifetime] execute start: k={total_k}, q={total_q}, "
            f"mpi_size={size}",
            flush=True,
        )
    print(f"[magph:lifetime][rank {rank}] local_k={len(local_indices)}", flush=True)
    for iloc, ik in enumerate(local_indices):
        k_vec = k_mesh_cart[int(ik)]
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
        if e_mag_k.shape != (nchannel,):
            raise ValueError(
                f"AFM kernel returned {e_mag_k.shape[0]} channels, expected {nchannel} "
                f"from spin_pattern length={physical_count}"
            )
        paraunitary_residual_local[int(ik)] = float(
            np.max(np.abs(paraunitary_residual(Uk, metric_diag)), initial=0.0)
        )
        metric_energy_min_local[int(ik)] = float(np.min(metric_diag * e_mag_k))
        common_energy_phys = float(np.mean(e_mag_k[:physical_count]))
        omega_eval = np.concatenate(
            [np.asarray(e_mag_k, dtype=np.float64), np.asarray([common_energy_phys])]
        )
        sigma_by_energy = compute_self_energy_at_frequencies(
            omega_eval=omega_eval,
            g_pq_cache=g_pq_cache,
            e_kq_cache=e_kq_cache,
            ph_en_flat=ctx["ph_en_flat"],
            T=T,
            eta=eta,
            total_q=total_q,
            metric_diag=metric_diag,
        )
        channel_index = np.arange(nchannel)
        sigma_at_channel_energies = sigma_by_energy[:nchannel]
        # Backward-compatible row-on-shell representation.  Row m is sampled
        # at E_m and the object is therefore not one common-frequency matrix.
        sigma_onshell = sigma_at_channel_energies[channel_index, channel_index, :]
        lw_raw = np.imag(
            sigma_at_channel_energies[channel_index, channel_index, channel_index]
        )
        sigma_phys_common = sigma_by_energy[-1, :physical_count, :physical_count]
        # The damping HWHM is Gamma = -Im Sigma^R on a diagonal channel.
        lw = -lw_raw
        linewidth_raw_local[int(ik)] = lw_raw
        linewidth_local[int(ik)] = np.maximum(lw, 0.0)
        energies_local[int(ik)] = e_mag_k
        sigma_onshell_local[int(ik)] = sigma_onshell
        sigma_at_channel_energies_local[int(ik)] = sigma_at_channel_energies
        sigma_phys_common_local[int(ik)] = sigma_phys_common
        common_energy_phys_local[int(ik)] = common_energy_phys
        done = iloc + 1
        if done % progress_interval == 0 or done == len(local_indices):
            dt = time.time() - t0
            rate = done / max(dt, 1.0e-12)
            eta_s = (len(local_indices) - done) / max(rate, 1.0e-12)
            print(
                f"[magph:lifetime][rank {rank}] progress {done}/{len(local_indices)} "
                f"global_k={int(ik)+1}/{total_k} elapsed={dt:.1f}s eta={eta_s:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    if comm is not None:
        linewidth = np.empty_like(linewidth_local) if rank == 0 else None
        linewidth_raw = np.empty_like(linewidth_raw_local) if rank == 0 else None
        energies = np.empty_like(energies_local) if rank == 0 else None
        sigma_onshell = np.empty_like(sigma_onshell_local) if rank == 0 else None
        sigma_at_channel_energies = (
            np.empty_like(sigma_at_channel_energies_local) if rank == 0 else None
        )
        sigma_phys_common = np.empty_like(sigma_phys_common_local) if rank == 0 else None
        common_energy_phys = np.empty_like(common_energy_phys_local) if rank == 0 else None
        paraunitary_residual_max = (
            np.empty_like(paraunitary_residual_local) if rank == 0 else None
        )
        metric_energy_min = np.empty_like(metric_energy_min_local) if rank == 0 else None
        from mpi4py import MPI
        comm.Reduce(linewidth_local, linewidth, op=MPI.SUM, root=0)
        comm.Reduce(linewidth_raw_local, linewidth_raw, op=MPI.SUM, root=0)
        comm.Reduce(energies_local, energies, op=MPI.SUM, root=0)
        comm.Reduce(sigma_onshell_local, sigma_onshell, op=MPI.SUM, root=0)
        comm.Reduce(
            sigma_at_channel_energies_local,
            sigma_at_channel_energies,
            op=MPI.SUM,
            root=0,
        )
        comm.Reduce(sigma_phys_common_local, sigma_phys_common, op=MPI.SUM, root=0)
        comm.Reduce(common_energy_phys_local, common_energy_phys, op=MPI.SUM, root=0)
        comm.Reduce(
            paraunitary_residual_local,
            paraunitary_residual_max,
            op=MPI.SUM,
            root=0,
        )
        comm.Reduce(metric_energy_min_local, metric_energy_min, op=MPI.SUM, root=0)
        elapsed = comm.reduce(elapsed, op=MPI.MAX, root=0)
    else:
        linewidth = linewidth_local
        linewidth_raw = linewidth_raw_local
        energies = energies_local
        sigma_onshell = sigma_onshell_local
        sigma_at_channel_energies = sigma_at_channel_energies_local
        sigma_phys_common = sigma_phys_common_local
        common_energy_phys = common_energy_phys_local
        paraunitary_residual_max = paraunitary_residual_local
        metric_energy_min = metric_energy_min_local

    if rank != 0:
        return None

    # Kept for compatibility only: rows of sigma_onshell use different
    # frequencies, and elementwise -imag is not an off-diagonal damping matrix.
    gamma_matrix_onshell = -np.imag(sigma_onshell)
    sigma_phys_row_onshell = sigma_onshell[:, :physical_count, :physical_count]
    gamma_phys_matrix_legacy = gamma_matrix_onshell[:, :physical_count, :physical_count]

    gamma_phys_common = retarded_damping_matrix(sigma_phys_common)
    gamma_phys_common_eigvals = np.linalg.eigvalsh(gamma_phys_common)
    energy_mixed_phys = np.zeros((total_k, physical_count), dtype=np.float64)
    linewidth_mixed_phys = np.zeros((total_k, physical_count), dtype=np.float64)
    for ik in range(total_k):
        h_eff = (
            np.diag(energies[ik, :physical_count].astype(np.complex128))
            + sigma_phys_common[ik]
        )
        vals = np.linalg.eigvals(h_eff)
        order = np.argsort(vals.real)
        vals = vals[order]
        energy_mixed_phys[ik] = vals.real
        linewidth_mixed_phys[ik] = np.maximum(-vals.imag, 0.0)

    scale = max(float(np.max(np.abs(linewidth_raw), initial=0.0)), 1.0)
    numerical_tol = np.sqrt(np.finfo(np.float64).eps) * scale
    negative_damping_tol = float(cfg.get("negative_damping_tol_mev", numerical_tol))
    n_clipped = int(np.count_nonzero((-linewidth_raw) < -negative_damping_tol))
    if n_clipped:
        print(f"[magph:lifetime][WARN] clipped {n_clipped} negative linewidth entries after sign conversion", flush=True)
    n_non_psd = int(
        np.count_nonzero(np.min(gamma_phys_common_eigvals, axis=1) < -negative_damping_tol)
    )
    if n_non_psd:
        print(
            f"[magph:lifetime][WARN] common-frequency physical damping matrix "
            f"is non-PSD at {n_non_psd}/{total_k} k-points "
            f"(tol={negative_damping_tol:.6g} meV)",
            flush=True,
        )
    paraunitary_tol = float(
        cfg.get(
            "paraunitary_tolerance",
            np.sqrt(np.finfo(np.float64).eps) * max(nchannel, 1),
        )
    )
    n_bad_paraunitary = int(
        np.count_nonzero(paraunitary_residual_max > paraunitary_tol)
    )
    if n_bad_paraunitary:
        print(
            f"[magph:lifetime][WARN] U^dagger eta U != eta at "
            f"{n_bad_paraunitary}/{total_k} k-points "
            f"(max={np.max(paraunitary_residual_max):.6g}, "
            f"tol={paraunitary_tol:.6g})",
            flush=True,
        )
    metric_energy_tol = float(cfg.get("metric_energy_tolerance_mev", negative_damping_tol))
    n_bad_metric_energy = int(np.count_nonzero(metric_energy_min < -metric_energy_tol))
    if n_bad_metric_energy:
        print(
            f"[magph:lifetime][WARN] sign(energy) is inconsistent with the BdG metric at "
            f"{n_bad_metric_energy}/{total_k} k-points "
            f"(min eta*E={np.min(metric_energy_min):.6g} meV)",
            flush=True,
        )

    observables = linewidth_observables(linewidth)
    magnetic_atom_indices = np.unique(
        np.concatenate(
            [
                np.asarray(ctx["j0"][1], dtype=np.int64).reshape(-1),
                np.asarray(ctx["j0"][2], dtype=np.int64).reshape(-1),
            ]
        )
    )
    if magnetic_atom_indices.shape != (physical_count,):
        raise ValueError(
            "Could not derive one magnetic atom index per physical channel: "
            f"indices={magnetic_atom_indices.tolist()}, physical_count={physical_count}"
        )
    base = os.path.splitext(out_json)[0]
    npz_path = base + ".npz"
    np.savez_compressed(
        npz_path,
        linewidth=linewidth,
        linewidth_raw=linewidth_raw,
        gamma_hwhm_mev=observables["gamma_hwhm_mev"],
        fwhm_mev=observables["fwhm_mev"],
        lifetime_ps=observables["lifetime_ps"],
        scattering_rate_ps_inv=observables["scattering_rate_ps_inv"],
        valid_damping=observables["valid_damping"],
        energy=energies,
        self_energy_onshell=sigma_onshell,
        self_energy_at_channel_energies=sigma_at_channel_energies,
        gamma_matrix_onshell=gamma_matrix_onshell,
        self_energy_phys_onshell=sigma_phys_row_onshell,
        gamma_phys_matrix_onshell=gamma_phys_matrix_legacy,
        self_energy_phys_common=sigma_phys_common,
        gamma_phys_common=gamma_phys_common,
        gamma_phys_common_eigvals=gamma_phys_common_eigvals,
        common_energy_phys_mev=common_energy_phys,
        energy_mixed_phys=energy_mixed_phys,
        linewidth_mixed_phys=linewidth_mixed_phys,
        k_mesh_cart=ctx["k_mesh_cart"],
        k_mesh_frac=ctx["k_mesh_frac"],
        q_mesh_frac=ctx["q_mesh_flat_frac"],
        bosonic_metric_diag=metric_diag,
        physical_channel_count=np.asarray(physical_count, dtype=np.int32),
        magnetic_atom_indices=magnetic_atom_indices,
        spin_pattern=np.asarray(ctx["spin_pattern"], dtype=np.float64),
        lattice_ang=np.asarray(ctx["R_vec"], dtype=np.float64),
        channel_labels=np.asarray(
            [f"particle_{index}" for index in range(physical_count)]
            + [f"hole_{index}" for index in range(physical_count)]
        ),
        linewidth_definition=np.asarray("gamma_hwhm_mev = -Im Sigma^R_mm(E_m)"),
        self_energy_onshell_convention=np.asarray(
            "row m is evaluated at omega=E_m; not a common-frequency matrix"
        ),
        negative_damping_tol_mev=np.asarray(negative_damping_tol, dtype=np.float64),
        paraunitary_residual_max=paraunitary_residual_max,
        paraunitary_tolerance=np.asarray(paraunitary_tol, dtype=np.float64),
        metric_energy_min_mev=metric_energy_min,
        metric_energy_tolerance_mev=np.asarray(metric_energy_tol, dtype=np.float64),
        vertex_formalism=np.asarray("atomic_gauge_full_Rp"),
        vertex_report_json=np.asarray(json.dumps(ctx["vertex_report"], sort_keys=True)),
        bond_factor=np.asarray(bond_factor, dtype=np.float64),
        anisotropy_mev=np.asarray(anisotropy_mev, dtype=np.float64),
    )
    return {
        "mode": "execute",
        "kernel_source": "internal_atomic_gauge",
        "total_k": total_k,
        "total_q": total_q,
        "mpi_size": size,
        "runtime_s": elapsed,
        "result_npz": npz_path,
        "physical_channel_count": physical_count,
        "negative_damping_entries": n_clipped,
        "non_psd_common_damping_kpoints": n_non_psd,
        "non_paraunitary_kpoints": n_bad_paraunitary,
        "metric_energy_sign_mismatch_kpoints": n_bad_metric_energy,
        "vertex": ctx["vertex_report"],
        "shell_filter": ctx.get("shell_filter", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="SLW magph lifetime runner (reconstructed runtime)")
    parser.add_argument("--workdir", type=str, default=None, help="Workflow root directory (default: current directory)")
    parser.add_argument("--input_file", type=str, default=None, help="Legacy-style input file (e.g. input.in)")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest from slw.magph.legacy.reference.prepare_lifetime")
    parser.add_argument("--mode", type=str, choices=["validate", "plan", "execute"], default="plan",
                        help="validate: contract check, plan: add memory estimate, execute: run lifetime kernel")
    parser.add_argument("--kernel_source", type=str, default=None, choices=["kernels", "kernels_lifetime"],
                        help="Kernel backend selection (default: kernels)")
    parser.add_argument("--k_tasks", type=int, default=None, help="Number of external k tasks for runtime planning")
    parser.add_argument("--progress_interval", type=int, default=None,
                        help="Print execute progress every N k-points (default: input file or 1)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for runtime metadata")
    parser.add_argument("--out_name", type=str, default="magph_lifetime_runtime.json", help="Output metadata filename")
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
    if args.progress_interval is not None:
        cfg["progress_interval"] = int(args.progress_interval)
    if kernel_source == "kernels_lifetime":
        print("[WARN] kernels_lifetime is legacy. kernels is recommended as latest backend.", flush=True)

    manifest = load_manifest(manifest_arg, workdir=workdir)
    shapes = build_shapes(manifest)
    analysis = estimate_lifetime_memory_gb(shapes)
    if args.k_tasks is not None:
        k_tasks = int(args.k_tasks)
    else:
        tkm = cfg.get("target_k_mesh", [10, 10, 10])
        k_tasks = int(tkm[0] * tkm[1] * tkm[2])

    out_json = resolve_out_path(
        workdir=workdir,
        out_dir=args.out_dir,
        out_name=args.out_name,
        default_dir=".",
        default_name="magph_lifetime_runtime.json",
    )
    stage = "io_validated" if args.mode == "validate" else ("io_planned" if args.mode == "plan" else "executed")
    extra = {
        "kernel_source": kernel_source,
        "k_tasks": k_tasks,
        "memory_gb": analysis,
    }
    _comm, rank, _ = _mpi_context()
    if args.mode == "execute":
        exec_info = _execute_lifetime(manifest, cfg, out_json=out_json)
        if rank != 0:
            return
        extra["execute"] = exec_info
    payload = build_runtime_payload(
        module_name="slw.magph.legacy.reference.lifetime_mpi_dynamic",
        manifest=manifest,
        shapes=shapes,
        stage=stage,
        extra=extra,
    )
    if rank == 0:
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

    print("[magph:lifetime] Runtime contract ready.", flush=True)
    print(
        f"[magph:lifetime] J bonds={shapes.n_j_bonds}, "
        f"dJ entries={shapes.n_dj_entries}, qpoints={shapes.n_qpoints}",
        flush=True,
    )
    print(f"[magph:lifetime] est. working_set_per_rank={analysis['working_set_per_rank_gb']:.3f} GB", flush=True)
    if args.mode == "execute":
        print(f"[magph:lifetime] result npz: {payload['analysis']['execute']['result_npz']}", flush=True)
    print(f"[magph:lifetime] runtime metadata: {out_json}", flush=True)


if __name__ == "__main__":
    main()
