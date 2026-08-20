# Legacy exchange implementation; use the native engine for new workflows.
import argparse
import json
import os

import h5py
import numpy as np

from slw.core.structure import compute_lattice, resolve_wannier_structure
from slw.exchange.legacy.diagnose_J_epr_kspace import _parse_slices
from slw.exchange.legacy.lkag_solver import (
    LKAGSolver,
    _compute_Gk_mesh_njit,
    get_cfr_ozaki_mesh,
    get_cfr_pole_mesh,
    get_semicircle_contour,
)


def _frac_key(frac, mesh):
    out = []
    for x, n in zip(frac, mesh):
        xi = int(np.rint((x % 1.0) * n)) % n
        out.append(xi)
    return tuple(out)


def _build_q_minus_map(q_fracs, nq_mesh):
    lookup = {_frac_key(qf, nq_mesh): iq for iq, qf in enumerate(q_fracs)}
    out = np.zeros(len(q_fracs), dtype=np.int32)
    for iq, qf in enumerate(q_fracs):
        key = _frac_key(np.mod(-qf, 1.0), nq_mesh)
        if key not in lookup:
            raise KeyError(f"-q not found for iq={iq}, key={key}")
        out[iq] = lookup[key]
    return out


def _build_mesh_frac_order(nk_grid):
    nk1, nk2, nk3 = nk_grid
    fracs = np.zeros((nk1 * nk2 * nk3, 3), dtype=np.float64)
    it = 0
    for k1 in range(nk1):
        for k2 in range(nk2):
            for k3 in range(nk3):
                fracs[it] = np.array([k1 / nk1, k2 / nk2, k3 / nk3], dtype=np.float64)
                it += 1
    return fracs


def _build_file_to_mesh_map(k_file, nk_grid):
    mesh_fracs = _build_mesh_frac_order(nk_grid)
    lookup = {_frac_key(mesh_fracs[i], nk_grid): i for i in range(len(mesh_fracs))}
    out = np.zeros(len(k_file), dtype=np.int32)
    for ik, kf in enumerate(k_file):
        key = _frac_key(kf, nk_grid)
        if key not in lookup:
            raise KeyError(f"k-point {kf.tolist()} not found on k-grid {nk_grid}")
        out[ik] = lookup[key]
    return out


def _load_gkq(path):
    ap = os.path.abspath(path)
    with h5py.File(ap, "r") as h5:
        if "g_band" not in h5:
            raise KeyError(f"{ap} has no dataset 'g_band'")
        return {
            "path": ap,
            "g_kq": np.asarray(h5["g_band"], dtype=np.complex128),  # (Nq, Nk, dim, dim)
            "k_fracs": np.asarray(h5["k_fracs"], dtype=np.float64),
            "q_fracs": np.asarray(h5["q_fracs"], dtype=np.float64),
            "kq_map": np.asarray(h5["kq_map"], dtype=np.int32),
            "channel": str(h5.attrs.get("channel", "")),
            "label": str(h5.attrs.get("label", "")),
            "axis": str(h5.attrs.get("axis", "")),
            "nk_dense": tuple(int(x) for x in np.asarray(h5.attrs.get("nk_dense"))),
            "nq_mesh": tuple(int(x) for x in np.asarray(h5.attrs.get("supercell_qmesh"))),
        }


def _make_energy_mesh(args):
    if args.integrator == "contour":
        return get_semicircle_contour(emin=args.emin, emax=0.0, npoints=args.empoints)
    if args.integrator == "cfr":
        return get_cfr_pole_mesh(beta=args.cfr_beta, n_poles=args.empoints)
    if args.integrator == "cfr_ozaki":
        return get_cfr_ozaki_mesh(n_poles=args.empoints)
    raise ValueError(f"Unsupported integrator: {args.integrator}")


def _safe_sign(mat):
    return -1.0 if np.real(np.trace(mat)) < 0 else 1.0


def _norm_rel(a, b):
    abs_err = float(abs(a - b))
    denom = max(float(abs(a)), float(abs(b)), 1e-15)
    return abs_err, abs_err / denom


def _collect_stats(lhs, rhs, q_map, sign=1.0):
    abs_list = []
    rel_list = []
    for iq in range(len(lhs)):
        v1 = float(lhs[iq])
        v2 = float(sign) * float(rhs[int(q_map[iq])])
        abs_err, rel_err = _norm_rel(v1, v2)
        abs_list.append(abs_err)
        rel_list.append(rel_err)
    return {
        "mean_abs": float(np.mean(abs_list)) if abs_list else 0.0,
        "max_abs": float(np.max(abs_list)) if abs_list else 0.0,
        "mean_rel": float(np.mean(rel_list)) if rel_list else 0.0,
        "max_rel": float(np.max(rel_list)) if rel_list else 0.0,
    }


def _collect_mag_stats(lhs, rhs, q_map):
    abs_list = []
    rel_list = []
    for iq in range(len(lhs)):
        v1 = float(abs(lhs[iq]))
        v2 = float(abs(rhs[int(q_map[iq])]))
        abs_err, rel_err = _norm_rel(v1, v2)
        abs_list.append(abs_err)
        rel_list.append(rel_err)
    return {
        "mean_abs": float(np.mean(abs_list)) if abs_list else 0.0,
        "max_abs": float(np.max(abs_list)) if abs_list else 0.0,
        "mean_rel": float(np.mean(rel_list)) if rel_list else 0.0,
        "max_rel": float(np.max(rel_list)) if rel_list else 0.0,
    }


def _collect_phase_fit_stats(lhs, rhs_conj_mapped):
    """
    Find a single global phase phi minimizing || lhs - exp(i phi) rhs ||.
    """
    if len(lhs) == 0:
        return {"phi_rad": 0.0, "mean_abs": 0.0, "max_abs": 0.0, "mean_rel": 0.0, "max_rel": 0.0}
    z = np.vdot(rhs_conj_mapped, lhs)
    phi = float(np.angle(z)) if abs(z) > 0 else 0.0
    rhs_fit = np.exp(1j * phi) * rhs_conj_mapped
    abs_list = []
    rel_list = []
    for a, b in zip(lhs, rhs_fit):
        abs_err, rel_err = _norm_rel(complex(a), complex(b))
        abs_list.append(abs_err)
        rel_list.append(rel_err)
    return {
        "phi_rad": phi,
        "mean_abs": float(np.mean(abs_list)),
        "max_abs": float(np.max(abs_list)),
        "mean_rel": float(np.mean(rel_list)),
        "max_rel": float(np.max(rel_list)),
    }


def _phase_ratio_analysis(A_ij, A_ji_m_conj, q_fracs, nq_mesh, extra_phase=None):
    """
    Analyze ratio r(q) = A_ij(q) / (extra_phase(q) * A_ji(-q)^*).
    Returns:
      - amplitude mismatch stats
      - circular concentration of phase
      - best linear phase fit exp(i(2pi q.dr + phi0)) via coarse grid search
    """
    if extra_phase is None:
        extra_phase = np.ones(len(A_ij), dtype=np.complex128)
    denom = extra_phase * A_ji_m_conj

    eps = 1e-14
    mask = np.abs(denom) > eps
    if not np.any(mask):
        return {
            "n_valid": 0,
            "amp_mean_abs": 0.0,
            "amp_max_abs": 0.0,
            "phase_circular_R": 0.0,
            "best_linear_fit": None,
            "samples": [],
        }

    q_use = q_fracs[mask]
    r = A_ij[mask] / denom[mask]
    amp = np.abs(r)
    unit = r / np.maximum(amp, eps)

    amp_abs = np.abs(amp - 1.0)
    circ = np.mean(unit)
    circ_R = float(np.abs(circ))

    # Coarse grid search for linear phase model unit(q) ~ exp(i(2pi q.dr + phi0))
    gx = np.arange(-nq_mesh[0], nq_mesh[0] + 1, dtype=np.float64) / float(max(1, nq_mesh[0]))
    gy = np.arange(-nq_mesh[1], nq_mesh[1] + 1, dtype=np.float64) / float(max(1, nq_mesh[1]))
    gz = np.arange(-nq_mesh[2], nq_mesh[2] + 1, dtype=np.float64) / float(max(1, nq_mesh[2]))
    best = None
    for dx in gx:
        for dy in gy:
            for dz in gz:
                phase = np.exp(-1j * 2.0 * np.pi * (q_use @ np.array([dx, dy, dz], dtype=np.float64)))
                z = np.mean(unit * phase)
                R = float(np.abs(z))
                if (best is None) or (R > best["R"]):
                    best = {
                        "dr_frac": [float(dx), float(dy), float(dz)],
                        "phi0_rad": float(np.angle(z)),
                        "R": R,
                    }

    dr = np.array(best["dr_frac"], dtype=np.float64)
    pred = np.exp(1j * (2.0 * np.pi * (q_use @ dr) + best["phi0_rad"]))
    phase_err = np.angle(unit * np.conjugate(pred))
    best["phase_err_mean_abs_rad"] = float(np.mean(np.abs(phase_err)))
    best["phase_err_max_abs_rad"] = float(np.max(np.abs(phase_err)))

    samples = []
    q_idx = np.where(mask)[0]
    for i in range(min(8, len(q_idx))):
        iq = int(q_idx[i])
        samples.append(
            {
                "iq": iq,
                "q_frac": [float(x) for x in q_fracs[iq]],
                "ratio_abs": float(np.abs(r[i])),
                "ratio_arg_rad": float(np.angle(r[i])),
            }
        )

    return {
        "n_valid": int(np.sum(mask)),
        "amp_mean_abs": float(np.mean(amp_abs)),
        "amp_max_abs": float(np.max(amp_abs)),
        "phase_circular_R": circ_R,
        "best_linear_fit": best,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose dJ(k,q) pair symmetry using direct momentum-space derivative kernel."
    )
    parser.add_argument("--res_dir", required=True, help="Directory containing Wannier90 hr.dat files")
    parser.add_argument("--gkq_up", required=True, help="g(k,q) HDF5 for spin-up")
    parser.add_argument("--gkq_dn", required=True, help="g(k,q) HDF5 for spin-down")
    parser.add_argument("--efermi", type=float, required=True, help="Fermi energy in eV")
    parser.add_argument("--prefix_up", type=str, required=True)
    parser.add_argument("--prefix_dn", type=str, required=True)
    parser.add_argument("--kmesh", type=int, nargs=3, required=True, help="Green-function k-grid")
    parser.add_argument("--pair", type=int, nargs=2, required=True, metavar=("I", "J"), help="0-based magnetic atom pair (i j)")
    parser.add_argument("--with_tau_phase", action="store_true", help="Apply exp(i 2pi q.(tau_j-tau_i)) phase in complex symmetry diagnostics")
    parser.add_argument("--win", type=str, default=None, help="Wannier90 .win path for tau-phase correction")
    parser.add_argument("--slices", required=True, help="Local slices as '0:0:5,1:5:10'")
    parser.add_argument("--mag_atoms", type=int, nargs="+", required=True, help="0-based atomic indices corresponding to magnetic-site indices")
    parser.add_argument("--ddelta_mode", choices=["off", "qavg"], default="off", help="Onsite dDelta in momentum space")
    parser.add_argument("--integrator", choices=["contour", "cfr", "cfr_ozaki"], default="contour")
    parser.add_argument("--emin", type=float, default=-25.0)
    parser.add_argument("--empoints", type=int, default=120)
    parser.add_argument("--cfr_beta", type=float, default=400.0)
    parser.add_argument("--out_json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    gup = _load_gkq(args.gkq_up)
    gdn = _load_gkq(args.gkq_dn)
    if gup["g_kq"].shape != gdn["g_kq"].shape:
        raise ValueError(f"g_kq shape mismatch: up={gup['g_kq'].shape}, dn={gdn['g_kq'].shape}")
    if tuple(gup["nk_dense"]) != tuple(args.kmesh):
        raise ValueError(f"kmesh mismatch: gkq nk_dense={gup['nk_dense']} vs --kmesh={tuple(args.kmesh)}")
    if gup["label"] != gdn["label"] or gup["axis"] != gdn["axis"]:
        raise ValueError(
            f"up/down target mismatch: up=({gup['label']},{gup['axis']}), "
            f"down=({gdn['label']},{gdn['axis']})"
        )

    Nk1, Nk2, Nk3 = tuple(int(x) for x in args.kmesh)
    Nk = Nk1 * Nk2 * Nk3
    Nq = int(np.prod(gup["nq_mesh"]))
    q_minus = _build_q_minus_map(gup["q_fracs"], gup["nq_mesh"])
    file_to_mesh = _build_file_to_mesh_map(gup["k_fracs"], (Nk1, Nk2, Nk3))

    solver = LKAGSolver(os.path.abspath(args.res_dir), efermi=args.efermi, formalism="collinear")
    solver.load_hr(args.prefix_up, args.prefix_dn)
    solver.configure_magnetic_subspace(_parse_slices(args.slices, solver.dim))
    if solver.k_cache["up"] is None or solver.k_cache["grid"] != (Nk1, Nk2, Nk3):
        solver.precompute_k_mesh((Nk1, Nk2, Nk3))
    up_evals, up_evecs = solver.k_cache["up"]
    dn_evals, dn_evecs = solver.k_cache["down"]

    i_idx, j_idx = int(args.pair[0]), int(args.pair[1])
    slc_i = solver.orbital_slices[i_idx]
    slc_j = solver.orbital_slices[j_idx]
    H0_up = solver.H_R_tensor[0, solver.idx_000]
    H0_dn = solver.H_R_tensor[1, solver.idx_000]
    Di = H0_up[slc_i, slc_i] - H0_dn[slc_i, slc_i]
    Dj = H0_up[slc_j, slc_j] - H0_dn[slc_j, slc_j]
    rel_sign_ij = _safe_sign(Di) * _safe_sign(Dj)
    rel_sign_ji = _safe_sign(Dj) * _safe_sign(Di)

    tau_phase = np.ones(Nq, dtype=np.complex128)
    tau_info = None
    if args.with_tau_phase:
        structure_path, _ = resolve_wannier_structure(args.res_dir, args.win)
        lat = compute_lattice(structure_path=structure_path)
        pos_frac = np.asarray(lat["positions"], dtype=np.float64)
        mag_atoms = [int(x) for x in args.mag_atoms]
        if i_idx >= len(mag_atoms) or j_idx >= len(mag_atoms):
            raise IndexError(
                f"pair indices {(i_idx, j_idx)} exceed --mag_atoms length {len(mag_atoms)}. "
                f"Provide matching --mag_atoms mapping."
            )
        ai = mag_atoms[i_idx]
        aj = mag_atoms[j_idx]
        if ai < 0 or aj < 0 or ai >= len(pos_frac) or aj >= len(pos_frac):
            raise IndexError(f"Mapped atomic indices out of structure bounds: ai={ai}, aj={aj}, n_atoms={len(pos_frac)}")
        dtau = pos_frac[aj] - pos_frac[ai]
        tau_phase = np.exp(1j * 2.0 * np.pi * (gup["q_fracs"] @ dtau))
        tau_info = {
            "win": str(structure_path),
            "mag_atoms": mag_atoms,
            "site_i_atom": int(ai),
            "site_j_atom": int(aj),
            "delta_tau_frac": [float(x) for x in dtau],
        }

    energy_mesh = _make_energy_mesh(args)
    A_ij = np.zeros(Nq, dtype=np.complex128)
    A_ji = np.zeros(Nq, dtype=np.complex128)

    if args.ddelta_mode == "qavg":
        dDi_q = np.mean(gup["g_kq"][:, :, slc_i, slc_i] - gdn["g_kq"][:, :, slc_i, slc_i], axis=1)
        dDj_q = np.mean(gup["g_kq"][:, :, slc_j, slc_j] - gdn["g_kq"][:, :, slc_j, slc_j], axis=1)
    else:
        dDi_q = np.zeros((Nq, slc_i.stop - slc_i.start, slc_i.stop - slc_i.start), dtype=np.complex128)
        dDj_q = np.zeros((Nq, slc_j.stop - slc_j.start, slc_j.stop - slc_j.start), dtype=np.complex128)

    print(
        f"[dJkq] start pair=({i_idx},{j_idx}) target={gup['label']}/{gup['axis']} "
        f"Nk={Nk} Nq={Nq} nE={len(energy_mesh)} ddelta_mode={args.ddelta_mode}"
    )

    for iz, (z, weight) in enumerate(energy_mesh):
        if iz % 10 == 0 or iz == len(energy_mesh) - 1:
            print(f"[dJkq] energy {iz+1}/{len(energy_mesh)}")
        G_u_mesh = _compute_Gk_mesh_njit(up_evals, up_evecs, z + solver.efermi, Nk1, Nk2, Nk3)
        G_d_mesh = _compute_Gk_mesh_njit(dn_evals, dn_evecs, z + solver.efermi, Nk1, Nk2, Nk3)
        G_u = G_u_mesh.reshape((Nk, solver.dim, solver.dim))[file_to_mesh]
        G_d = G_d_mesh.reshape((Nk, solver.dim, solver.dim))[file_to_mesh]

        for iq in range(Nq):
            acc_ij = 0.0 + 0.0j
            acc_ji = 0.0 + 0.0j
            dDi = dDi_q[iq]
            dDj = dDj_q[iq]
            for ik in range(Nk):
                ikq = int(gup["kq_map"][ik, iq])
                Gu_k = G_u[ik]
                Gu_kq = G_u[ikq]
                Gd_k = G_d[ik]
                Gd_kq = G_d[ikq]
                gu = gup["g_kq"][iq, ik]
                gd = gdn["g_kq"][iq, ik]

                dGu = Gu_kq @ gu @ Gu_k
                dGd = Gd_kq @ gd @ Gd_k

                # ij channel
                Gij_u = Gu_kq[slc_i, slc_j]
                Gji_d = Gd_k[slc_j, slc_i]
                dGij_u = dGu[slc_i, slc_j]
                dGji_d = dGd[slc_j, slc_i]
                t_ij = (
                    np.trace(dDi @ Gij_u @ Dj @ Gji_d)
                    + np.trace(Di @ dGij_u @ Dj @ Gji_d)
                    + np.trace(Di @ Gij_u @ dDj @ Gji_d)
                    + np.trace(Di @ Gij_u @ Dj @ dGji_d)
                )
                acc_ij += t_ij

                # ji channel
                Gji_u = Gu_kq[slc_j, slc_i]
                Gij_d = Gd_k[slc_i, slc_j]
                dGji_u = dGu[slc_j, slc_i]
                dGij_d = dGd[slc_i, slc_j]
                t_ji = (
                    np.trace(dDj @ Gji_u @ Di @ Gij_d)
                    + np.trace(Dj @ dGji_u @ Di @ Gij_d)
                    + np.trace(Dj @ Gji_u @ dDi @ Gij_d)
                    + np.trace(Dj @ Gji_u @ Di @ dGij_d)
                )
                acc_ji += t_ji

            A_ij[iq] += (acc_ij / float(Nk)) * weight
            A_ji[iq] += (acc_ji / float(Nk)) * weight

    dJ_q_ij = 1000.0 * np.imag(A_ij) / (4.0 * np.pi * rel_sign_ij)
    dJ_q_ji = 1000.0 * np.imag(A_ji) / (4.0 * np.pi * rel_sign_ji)

    # Diagnose multiple plausible pair relations.
    id_map = np.arange(Nq, dtype=np.int32)
    cand_stats = {
        "ij(q)=ji(-q)": _collect_stats(dJ_q_ij, dJ_q_ji, q_minus, sign=+1.0),
        "ij(q)=-ji(-q)": _collect_stats(dJ_q_ij, dJ_q_ji, q_minus, sign=-1.0),
        "ij(q)=ji(q)": _collect_stats(dJ_q_ij, dJ_q_ji, id_map, sign=+1.0),
        "ij(q)=-ji(q)": _collect_stats(dJ_q_ij, dJ_q_ji, id_map, sign=-1.0),
    }
    Aji_m_conj = np.conjugate(A_ji[q_minus])
    complex_diag = {
        "|Aij(q)|=|Aji(-q)|": _collect_mag_stats(A_ij, A_ji, q_minus),
        "Aij(q)=phase*Aji(-q)^*": _collect_phase_fit_stats(A_ij, Aji_m_conj),
    }
    if args.with_tau_phase:
        complex_diag["Aij(q)=exp(i2pi q.dtau)*Aji(-q)^*"] = _collect_phase_fit_stats(A_ij, tau_phase * Aji_m_conj)

    phase_ratio = {
        "raw": _phase_ratio_analysis(A_ij, Aji_m_conj, gup["q_fracs"], gup["nq_mesh"], extra_phase=None)
    }
    if args.with_tau_phase:
        phase_ratio["tau_corrected"] = _phase_ratio_analysis(
            A_ij, Aji_m_conj, gup["q_fracs"], gup["nq_mesh"], extra_phase=tau_phase
        )

    # Keep per-q detail for the canonical check first.
    per_q = []
    for iq in range(Nq):
        iqm = int(q_minus[iq])
        lhs = float(dJ_q_ij[iq])
        rhs = float(dJ_q_ji[iqm])
        abs_err, rel_err = _norm_rel(lhs, rhs)
        per_q.append(
            {
                "iq": int(iq),
                "iq_minus": iqm,
                "dJ_ij_q_meV": lhs,
                "dJ_ji_minusq_meV": rhs,
                "abs_err": abs_err,
                "rel_err": rel_err,
            }
        )

    summary = {
        "method": "dJkq_symmetry_diagnostic",
        "pair": [i_idx, j_idx],
        "target_label": gup["label"],
        "target_axis": gup["axis"],
        "nk_grid": [Nk1, Nk2, Nk3],
        "nq_mesh": list(gup["nq_mesh"]),
        "n_energy": int(len(energy_mesh)),
        "integrator": args.integrator,
        "ddelta_mode": args.ddelta_mode,
        "with_tau_phase": bool(args.with_tau_phase),
        "tau_phase_info": tau_info,
        "check_relation": "dJ_ij(q) ?= dJ_ji(-q)",
        "stats": cand_stats["ij(q)=ji(-q)"],
        "candidate_relations": cand_stats,
        "complex_kernel_checks": complex_diag,
        "phase_ratio_analysis": phase_ratio,
        "per_q": per_q,
    }

    print(
        json.dumps(
            {
                "candidate_relations": summary["candidate_relations"],
                "complex_kernel_checks": summary["complex_kernel_checks"],
                "phase_ratio_analysis": summary["phase_ratio_analysis"],
            },
            indent=2,
        )
    )
    if args.out_json:
        outp = os.path.abspath(args.out_json)
        with open(outp, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[dJkq] wrote {outp}")


if __name__ == "__main__":
    main()
