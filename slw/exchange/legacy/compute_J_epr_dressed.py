# Legacy exchange implementation; use the native engine for new workflows.
"""EPR LKAG exchange with a diagonal Fan--Migdal dressed Green function.

This is deliberately a separate driver from :mod:`compute_J_epr_kspace` so
the validated bare-EPR path remains unchanged.  The implemented approximation
is a dressed LKAG bubble: the electronic propagators contain the diagonal,
frequency-dependent Fan self-energy, while vertex and Debye--Waller
corrections are not included.
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np

from slw.exchange.legacy.reference.compute_J_epr_kspace import (
    _find_nearest_neighbours_from_epr,
    _full_k_mesh,
    _normalize_mag_atoms,
)
from slw.exchange.legacy.reference.compute_dJ_epr_kspace import _build_gkq_one
from slw.exchange.legacy.diagnose_J_epr_kspace import _build_hk_from_epr, _load_slices
from slw.exchange.legacy.eph_epr_wrapper import _read_meta
from slw.exchange.legacy.lkag_solver import get_cfr_ozaki_mesh, get_cfr_pole_mesh, get_semicircle_contour
from slw.magph.legacy.adapter import RYD2MEV, build_phonon_cache_from_epr

KB_EV_K = 8.617333262145e-5


def _fermi(x_ev, temperature):
    if temperature <= 0.0:
        return (np.asarray(x_ev) < 0.0).astype(np.float64)
    x = np.clip(np.asarray(x_ev) / (KB_EV_K * float(temperature)), -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)


def _bose(omega_ev, temperature):
    if temperature <= 0.0:
        return np.zeros_like(omega_ev, dtype=np.float64)
    x = np.clip(np.asarray(omega_ev) / (KB_EV_K * float(temperature)), 1.0e-12, 700.0)
    return 1.0 / np.expm1(x)


def _eigensystem(hk, efermi):
    dim = int(hk.shape[1])
    shifted = 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))
    shifted = shifted - float(efermi) * np.eye(dim, dtype=np.complex128)[None, :, :]
    return np.linalg.eigh(shifted)


def _infer_magnetic_subspaces(epr_path, hk_up, hk_dn, relative_threshold):
    """Assign Wannier functions to atoms and select exchange-split sites."""
    with h5py.File(epr_path, "r") as h5:
        meta = _read_meta(h5)
    wc = np.mod(np.asarray(meta.wc, dtype=np.float64), 1.0)
    tau = np.mod(np.asarray(meta.tau, dtype=np.float64), 1.0)
    diff = wc[:, None, :] - tau[None, :, :]
    diff -= np.round(diff)
    cart = np.einsum("wac,cd->wad", diff, np.asarray(meta.at, dtype=np.float64).T, optimize=True)
    owner = np.argmin(np.linalg.norm(cart, axis=2), axis=1)
    split = np.mean(hk_up - hk_dn, axis=0)
    atom_orbitals = [np.flatnonzero(owner == ia) for ia in range(meta.nat)]
    scores = np.asarray(
        [np.linalg.norm(split[np.ix_(idx, idx)]) if idx.size else 0.0 for idx in atom_orbitals],
        dtype=np.float64,
    )
    if not np.any(scores > 0.0):
        raise ValueError("Could not infer magnetic atoms: every atom-local exchange-splitting score is zero.")
    selected = np.flatnonzero(scores >= float(relative_threshold) * float(np.max(scores)))
    if selected.size < 2:
        selected = np.argsort(scores)[-min(2, len(scores)):]
    selected = np.sort(selected)
    slices = {ilocal: atom_orbitals[int(ia)] for ilocal, ia in enumerate(selected)}
    print(
        f"[J-epr-dressed] inferred magnetic atoms={selected.tolist()} "
        f"exchange scores={scores.tolist()}",
        flush=True,
    )
    return selected.tolist(), slices


def _mesh_kplusq_map(kmesh, qmesh):
    km = np.asarray(kmesh, dtype=np.int64)
    qm = np.asarray(qmesh, dtype=np.int64)
    if np.any(km % qm != 0):
        raise ValueError(f"EPR qmesh must divide kmesh for exact k+q mapping: kmesh={tuple(km)}, qmesh={tuple(qm)}")
    kidx = np.indices(tuple(km)).reshape(3, -1).T
    qidx = np.indices(tuple(qm)).reshape(3, -1).T
    q_on_k = qidx * (km // qm)[None, :]
    kpq = (kidx[None, :, :] + q_on_k[:, None, :]) % km[None, None, :]
    return ((kpq[..., 0] * km[1] + kpq[..., 1]) * km[2] + kpq[..., 2]).astype(np.int64)


def _fan_sigma_diagonal(
    epr_path,
    kpts,
    qpts,
    evals,
    evecs,
    energy_mesh,
    phonons,
    kq_map,
    *,
    temperature,
    eph_unit,
    acoustic_cutoff_mev,
    q_block_size,
    energy_block_size,
    verbose=True,
):
    """Return diagonal Fan Sigma(z,k,n) in eV using q-block vectorization."""
    zvals = np.asarray([z for z, _dz in energy_mesh], dtype=np.complex128)
    nq, nk = len(qpts), len(kpts)
    nb = int(evals.shape[1])
    ph_en_mev = np.asarray(phonons["ph_en_flat"], dtype=np.float64)
    ph_vec = np.asarray(phonons["ph_vec_flat"], dtype=np.complex128)
    nat = int(ph_vec.shape[2])
    nm = int(ph_vec.shape[1])
    sigma = np.zeros((len(zvals), nk, nb), dtype=np.complex128)
    phase_cache = {}
    q_block_size = max(1, int(q_block_size))

    for q0 in range(0, nq, q_block_size):
        q1 = min(nq, q0 + q_block_size)
        qb = q1 - q0
        omega_mev = ph_en_mev[q0:q1]
        valid = omega_mev > float(acoustic_cutoff_mev)
        omega_ry = omega_mev / float(RYD2MEV)
        mode_factor = np.zeros_like(omega_ry)
        mode_factor[valid] = 1.0 / np.sqrt(2.0 * omega_ry[valid])

        # g_mode[q,k,i,j,nu].  Building Cartesian components serially keeps
        # peak memory bounded; the expensive matrix contractions are batched.
        g_mode = np.zeros((qb, nk, nb, nb, nm), dtype=np.complex128)
        for ia in range(nat):
            for ax, axis in enumerate(("x", "y", "z")):
                g_cart = _build_gkq_one(
                    epr_path,
                    ia,
                    axis,
                    kpts,
                    qpts[q0:q1],
                    unit=eph_unit,
                    phase_cache=phase_cache,
                )
                coeff = ph_vec[q0:q1, :, ia, ax] * mode_factor
                g_mode += g_cart[..., None] * coeff[:, None, None, None, :]

        maps = kq_map[q0:q1]
        u_k = evecs[None, :, :, :]
        u_kq = evecs[maps]
        # <m,k+q|g_nu|n,k>, axes=(q,k,m,n,nu)
        tmp = np.einsum("qkim,qkijv->qkmjv", np.conjugate(u_kq), g_mode, optimize=True)
        g_band = np.einsum("qkmjv,qkjn->qkmnv", tmp, u_k, optimize=True)
        abs2 = np.abs(g_band) ** 2
        del tmp, g_band, g_mode

        eps_final = evals[maps]  # (q,k,m), relative to EF
        f_final = _fermi(eps_final, temperature)
        omega_ev = omega_mev * 1.0e-3
        n_bose = _bose(np.where(valid, omega_ev, 1.0), temperature) * valid
        emit_num = n_bose[:, None, None, :] + 1.0 - f_final[:, :, :, None]
        absorb_num = n_bose[:, None, None, :] + f_final[:, :, :, None]
        emit_num *= valid[:, None, None, :]
        absorb_num *= valid[:, None, None, :]

        # Energy blocks prevent a (nE,q,k,m,n,nu) temporary from dominating memory.
        energy_block_size = max(1, int(energy_block_size))
        for iz0 in range(0, len(zvals), energy_block_size):
            iz1 = min(len(zvals), iz0 + energy_block_size)
            zblk = zvals[iz0:iz1, None, None, None, None]
            den_emit = zblk - eps_final[None, :, :, :, None] - omega_ev[None, :, None, None, :]
            den_abs = zblk - eps_final[None, :, :, :, None] + omega_ev[None, :, None, None, :]
            kernel = emit_num[None, ...] / den_emit + absorb_num[None, ...] / den_abs
            sigma[iz0:iz1] += np.einsum("qkmnv,eqkmv->ekn", abs2, kernel, optimize=True) / float(nq)
        if verbose:
            print(f"[J-epr-dressed] Fan Sigma q block {q0}:{q1}/{nq}", flush=True)
    return sigma


def _site_data(hk_up, hk_dn, slices, efermi):
    eu, uu = _eigensystem(hk_up, efermi)
    ed, ud = _eigensystem(hk_dn, efermi)
    h0u = np.mean(hk_up, axis=0)
    h0d = np.mean(hk_dn, axis=0)
    delta, signs = {}, {}
    for sid, slc in slices.items():
        idx = np.arange(hk_up.shape[1])[slc] if isinstance(slc, slice) else np.asarray(slc, dtype=np.int64)
        raw = (h0u - h0d)[np.ix_(idx, idx)]
        d = 0.5 * (raw + raw.conj().T)
        delta[int(sid)] = d
        signs[int(sid)] = -1.0 if np.real(np.trace(d)) < 0.0 else 1.0
    coeff_u = {int(s): uu[:, sl, :] for s, sl in slices.items()}
    coeff_d = {int(s): ud[:, sl, :] for s, sl in slices.items()}
    return eu, uu, ed, ud, coeff_u, coeff_d, delta, signs


def _integrate_j(energy_mesh, evals_up, evals_dn, sigma_up, sigma_dn, coeff_up, coeff_dn, delta, signs, pair_meta, phase):
    bare = np.zeros(len(pair_meta), dtype=np.complex128)
    dressed = np.zeros(len(pair_meta), dtype=np.complex128)
    nk = int(evals_up.shape[0])
    weight = 1.0 / float(nk)
    for iz, (z, dz) in enumerate(energy_mesh):
        inv0u = 1.0 / (z - evals_up)
        inv0d = 1.0 / (z - evals_dn)
        invu = 1.0 / (z - evals_up - sigma_up[iz])
        invd = 1.0 / (z - evals_dn - sigma_dn[iz])
        for ip, meta in enumerate(pair_meta):
            li, lj = int(meta["li"]), int(meta["lj"])
            wk = weight * phase[ip]
            rel = signs.get(li, 1.0) * signs.get(lj, 1.0)
            vals = []
            for iu, idn in ((inv0u, inv0d), (invu, invd)):
                gu_k = np.einsum("kia,ka,kja->kij", coeff_up[li], iu, np.conjugate(coeff_up[lj]), optimize=True)
                gd_k = np.einsum("kja,ka,kia->kji", coeff_dn[lj], idn, np.conjugate(coeff_dn[li]), optimize=True)
                gu = np.einsum("k,kij->ij", wk, gu_k, optimize=True)
                gd = np.einsum("k,kji->ji", np.conjugate(wk), gd_k, optimize=True)
                vals.append(np.trace(delta[li] @ gu @ delta[lj] @ gd) / rel)
            bare[ip] += vals[0] * dz
            dressed[ip] += vals[1] * dz
    scale = 1000.0 / (4.0 * np.pi)
    return scale * np.imag(bare), scale * np.imag(dressed)


def _energy_mesh(args):
    if args.integrator == "contour":
        return get_semicircle_contour(emin=args.emin, emax=0.0, npoints=args.empoints)
    if args.integrator == "cfr_ozaki":
        return get_cfr_ozaki_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)
    return get_cfr_pole_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)


def _write_results(path, h5_path, args, pair_meta, bare, dressed, elapsed, kmesh, qmesh):
    delta = dressed - bare
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as out:
        out.write("# EPR diagonal Fan-Migdal dressed LKAG exchange\n")
        out.write("# Approximation: dressed bubble; no vertex or Debye-Waller correction\n")
        out.write(f"# temperature_K={args.temperature} kmesh={tuple(kmesh)} qmesh={tuple(qmesh)} elapsed_s={elapsed:.3f}\n")
        out.write("bond\tgi\tgj\tR1\tR2\tR3\tdistance_A\tJ_bare_meV\tJ_dressed_meV\tdelta_J_meV\n")
        for ib, meta in enumerate(pair_meta):
            r = meta["R"]
            out.write(
                f"{ib}\t{meta['gi']}\t{meta['gj']}\t{r[0]}\t{r[1]}\t{r[2]}\t{meta['dist']:.12e}\t"
                f"{bare[ib]:.12e}\t{dressed[ib]:.12e}\t{delta[ib]:.12e}\n"
            )
    with h5py.File(h5_path, "w") as h5:
        basic = h5.create_group("basic_data")
        basic.create_dataset("temperature_K", data=float(args.temperature))
        basic.create_dataset("kmesh", data=np.asarray(kmesh, dtype=np.int64))
        basic.create_dataset("qmesh", data=np.asarray(qmesh, dtype=np.int64))
        basic.attrs["approximation"] = "diagonal_dynamic_Fan_dressed_bubble"
        basic.attrs["excluded"] = "vertex_corrections,Debye-Waller"
        bonds = h5.create_group("bonds")
        bonds.create_dataset("mag_i_atom", data=[m["gi"] for m in pair_meta])
        bonds.create_dataset("mag_j_atom", data=[m["gj"] for m in pair_meta])
        bonds.create_dataset("R", data=np.asarray([m["R"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("distance_ang", data=[m["dist"] for m in pair_meta])
        jgrp = h5.create_group("J_r")
        jgrp.create_dataset("bare_meV", data=bare)
        jgrp.create_dataset("dressed_meV", data=dressed)
        jgrp.create_dataset("delta_meV", data=delta)


def run(args):
    started = time.time()
    with h5py.File(args.epr_up, "r") as h5:
        meta = _read_meta(h5)
    kmesh, qmesh = tuple(meta.nk_grid), tuple(meta.nq_grid)
    kpts, qpts = _full_k_mesh(kmesh), _full_k_mesh(qmesh)
    kq_map = _mesh_kplusq_map(kmesh, qmesh)
    print(f"[J-epr-dressed] EPR meshes k={kmesh} q={qmesh}", flush=True)
    hk_up = _build_hk_from_epr(args.epr_up, kpts, unit=args.hr_unit)
    hk_dn = _build_hk_from_epr(args.epr_dn, kpts, unit=args.hr_unit)
    if args.mag_atoms is None and args.slices is None:
        mag_atoms, slices = _infer_magnetic_subspaces(
            args.epr_up, hk_up, hk_dn, args.magnetic_threshold
        )
    elif args.mag_atoms is not None and args.slices is not None:
        mag_atoms = _normalize_mag_atoms(args)
        slices = _load_slices(args, int(hk_up.shape[1]))
    else:
        raise ValueError("Provide both --mag_atoms and --slices, or omit both for automatic inference.")
    eu, uu, ed, ud, cu, cd, delta, signs = _site_data(hk_up, hk_dn, slices, args.efermi)
    energies = _energy_mesh(args)
    phonons = build_phonon_cache_from_epr(args.epr_up, qpts_frac=qpts, nproc=args.nproc, verbose=True)
    su = _fan_sigma_diagonal(
        args.epr_up, kpts, qpts, eu, uu, energies, phonons, kq_map,
        temperature=args.temperature, eph_unit=args.eph_unit,
        acoustic_cutoff_mev=args.acoustic_cutoff_mev, q_block_size=args.q_block, verbose=True,
        energy_block_size=args.energy_block,
    )
    sd = _fan_sigma_diagonal(
        args.epr_dn, kpts, qpts, ed, ud, energies, phonons, kq_map,
        temperature=args.temperature, eph_unit=args.eph_unit,
        acoustic_cutoff_mev=args.acoustic_cutoff_mev, q_block_size=args.q_block, verbose=True,
        energy_block_size=args.energy_block,
    )
    neighbours = _find_nearest_neighbours_from_epr(
        args.epr_up, mag_atom_indices=mag_atoms, n_shells=args.n_shells, d_max=args.d_max, all_bonds=True
    )
    glocal = {g: i for i, g in enumerate(mag_atoms)}
    pair_meta = [
        {"gi": int(n["i"]), "gj": int(n["j"]), "li": glocal[int(n["i"])], "lj": glocal[int(n["j"])],
         "R": tuple(int(x) for x in n["R"]), "dist": float(n["distance"]), "shell": int(n.get("shell_idx", 0))}
        for n in neighbours if int(n["i"]) in glocal and int(n["j"]) in glocal
    ]
    rvec = np.asarray([m["R"] for m in pair_meta], dtype=np.float64)
    phase = np.exp(-2j * np.pi * (rvec @ kpts.T))
    bare, dressed = _integrate_j(energies, eu, ed, su, sd, cu, cd, delta, signs, pair_meta, phase)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, args.out_name)
    h5_out = os.path.join(out_dir, os.path.splitext(args.out_name)[0] + ".h5")
    _write_results(out, h5_out, args, pair_meta, bare, dressed, time.time() - started, kmesh, qmesh)
    print(f"[J-epr-dressed] wrote {out}\n[J-epr-dressed] wrote {h5_out}")


def main():
    ap = argparse.ArgumentParser(description="EPR LKAG J with diagonal Fan-Migdal dressed Green functions")
    ap.add_argument("--epr_up", required=True)
    ap.add_argument("--epr_dn", required=True)
    ap.add_argument("--efermi", required=True, type=float, help="Fermi energy in eV")
    ap.add_argument("--temperature", type=float, default=300.0, help="Temperature in K")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--out_name", default="J_epr_dressed.tsv")
    advanced = ap.add_argument_group("advanced numerical/model options")
    advanced.add_argument("--hr_unit", choices=["ev", "ry", "ha"], default="ry")
    advanced.add_argument("--eph_unit", choices=["ev", "ry", "ha"], default="ry")
    advanced.add_argument("--empoints", type=int, default=128)
    advanced.add_argument("--emin", type=float, default=-25.0)
    advanced.add_argument("--integrator", choices=["contour", "cfr", "cfr_ozaki"], default="contour")
    advanced.add_argument("--cfr_beta", type=float, default=400.0)
    advanced.add_argument("--acoustic_cutoff_mev", type=float, default=0.1)
    advanced.add_argument("--q_block", type=int, default=4)
    advanced.add_argument("--energy_block", type=int, default=8)
    advanced.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    advanced.add_argument("--mag_atoms", type=int, nargs="+", default=None)
    advanced.add_argument("--mag_atoms_base", type=int, choices=[0, 1], default=0)
    advanced.add_argument("--slices", default=None)
    advanced.add_argument("--magnetic_threshold", type=float, default=0.25)
    advanced.add_argument("--n_shells", type=int, default=10)
    advanced.add_argument("--d_max", type=float, default=20.0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
