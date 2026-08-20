# Legacy exchange implementation; use the native engine for new workflows.
"""Diagnose bond-resolved LKAG asymmetry directly from EPR electron_wannier.

This avoids the EPR -> hr.dat -> H(k) round trip used by compute_J_epr.py.
It builds H_up(k), H_dn(k) directly from the EPR HDF5 electron_wannier blocks
and reports forward J_ij(R) vs mirror J_ji(-R) LKAG components.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree

from slw.core.constants import BOHR_TO_ANG
from slw.core.qe2pert_ws import init_rvec_images, set_wigner_seitz_cell, triangular_pair_index
from slw.exchange.legacy.green import green_subblock_from_coefficients
from slw.exchange.legacy.eph_epr_wrapper import _unit_scale_to_ev
from slw.exchange.legacy.lkag_solver import get_cfr_ozaki_mesh, get_cfr_pole_mesh, get_semicircle_contour


def _read_epr_meta(path):
    with h5py.File(path, "r") as h5:
        at = np.asarray(h5["basic_data/at"], dtype=np.float64).T
        wc = np.asarray(h5["basic_data/wannier_center_cryst"], dtype=np.float64)
        tau_cart = np.asarray(h5["basic_data/tau"], dtype=np.float64)
        tau = np.linalg.solve(at, tau_cart.T).T
        return {
            "at": at,
            "wc": wc,
            "tau": tau,
            "kc_dim": tuple(int(x) for x in h5["basic_data/kc_dim"][()]),
            "num_wann": int(h5["basic_data/num_wann"][()]),
        }


def _full_k_mesh(kmesh):
    n1, n2, n3 = (int(x) for x in kmesh)
    return np.asarray(
        [(i / n1, j / n2, k / n3) for i in range(n1) for j in range(n2) for k in range(n3)],
        dtype=np.float64,
    )


def _read_epr_lattice_positions(epr_path):
    with h5py.File(epr_path, "r") as h5:
        at = np.asarray(h5["basic_data/at"], dtype=np.float64).T
        tau_cart_alat = np.asarray(h5["basic_data/tau"], dtype=np.float64)
        alat_bohr = float(h5["basic_data/alat"][()])
    tau_frac = np.linalg.solve(at, tau_cart_alat.T).T
    rvec_ang = at.T * (alat_bohr * BOHR_TO_ANG)
    return rvec_ang, tau_frac


def _find_nearest_neighbours_from_epr(epr_path, mag_atom_indices, n_shells=1, d_max=20.0, all_bonds=True):
    rvec_ang, pos_frac = _read_epr_lattice_positions(epr_path)
    rnorm = np.linalg.norm(rvec_ang, axis=1)
    rmax = int(np.ceil(float(d_max) / (float(np.min(rnorm)) + 1.0e-12))) + 1
    grid = np.arange(-rmax, rmax + 1, dtype=np.int64)
    shifts = np.asarray(np.meshgrid(grid, grid, grid, indexing="ij"), dtype=np.int64).reshape(3, -1).T

    nat = int(pos_frac.shape[0])
    all_frac = shifts[:, None, :] + pos_frac[None, :, :]
    all_pos = all_frac.reshape(-1, 3) @ rvec_ang
    atom_ids = np.tile(np.arange(nat, dtype=np.int64), len(shifts))
    shift_ids = np.repeat(shifts, nat, axis=0)
    tree = cKDTree(all_pos)

    neighbours = []
    seen = set()
    mag = [int(x) for x in mag_atom_indices]
    mag_set = set(mag)
    for ia in mag:
        i_pos = pos_frac[ia] @ rvec_ang
        idx_found = np.asarray(tree.query_ball_point(i_pos, r=float(d_max)), dtype=np.int64)
        d_found = np.linalg.norm(all_pos[idx_found] - i_pos[None, :], axis=1)
        order = np.argsort(d_found)
        for idx, d in zip(idx_found[order], d_found[order]):
            d = float(d)
            if d < 1.0e-4 or d > float(d_max):
                continue
            idx = int(idx)
            ja = int(atom_ids[idx])
            if ja not in mag_set:
                continue
            r = tuple(int(x) for x in shift_ids[idx])
            if all_bonds:
                if ia == ja and not any(x != 0 for x in r):
                    continue
                key = (ia, ja, r)
            else:
                if ia < ja:
                    key = (ia, ja, r)
                elif ia > ja:
                    key = (ja, ia, tuple(-x for x in r))
                else:
                    if not any(x != 0 for x in r):
                        continue
                    first_nz = next(x for x in r if x != 0)
                    key = (ia, ia, r) if first_nz > 0 else (ia, ia, tuple(-x for x in r))
            if key in seen:
                continue
            seen.add(key)
            neighbours.append({"i": key[0], "j": key[1], "R": key[2], "distance": d})

    neighbours.sort(key=lambda x: x["distance"])
    if not neighbours:
        return []
    shells = []
    curr = [neighbours[0]]
    curr_d = float(neighbours[0]["distance"])
    for n in neighbours[1:]:
        if abs(float(n["distance"]) - curr_d) < 1.0e-4:
            curr.append(n)
        else:
            shells.append(curr)
            if len(shells) >= int(n_shells):
                break
            curr = [n]
            curr_d = float(n["distance"])
    else:
        if len(shells) < int(n_shells):
            shells.append(curr)

    out = []
    for ishell, shell in enumerate(shells, start=1):
        for b in shell:
            row = dict(b)
            row["shell_idx"] = int(ishell)
            out.append(row)
    return out


def _build_hk_from_epr(path, kpts, *, unit="ry"):
    meta = _read_epr_meta(path)
    nwan = int(meta["num_wann"])
    images = init_rvec_images(meta["kc_dim"], meta["at"])
    hk = np.zeros((len(kpts), nwan, nwan), dtype=np.complex128)
    scale = _unit_scale_to_ev(unit)
    with h5py.File(path, "r") as h5:
        ws_cache = {}
        for jw in range(1, nwan + 1):
            for iw in range(1, jw + 1):
                tri = triangular_pair_index(iw, jw)
                dr = f"electron_wannier/hopping_r{tri}"
                di = f"electron_wannier/hopping_i{tri}"
                if dr not in h5 or di not in h5:
                    continue
                hop = np.asarray(h5[dr], dtype=np.float64) + 1j * np.asarray(h5[di], dtype=np.float64)
                key = (iw, jw)
                ws = ws_cache.get(key)
                if ws is None:
                    ws = set_wigner_seitz_cell(images, meta["at"], meta["wc"][iw - 1], meta["wc"][jw - 1])
                    ws_cache[key] = ws
                if hop.shape[0] != ws.nr:
                    raise ValueError(
                        f"hopping length mismatch pair=({iw},{jw}): h5={hop.shape[0]} ws={ws.nr}"
                    )
                phase = np.exp(2j * np.pi * (kpts @ np.asarray(ws.vectors, dtype=np.float64).T))
                vals = (phase @ hop) * scale
                hk[:, iw - 1, jw - 1] = vals
                if iw != jw:
                    hk[:, jw - 1, iw - 1] = np.conjugate(vals)
    return 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))


def _parse_slices(text, dim):
    if not text:
        raise ValueError(
            "--slices is required for EPR input, for example "
            "'0:0:5,1:5:10'. The package does not assume a material-specific orbital layout."
        )
    out = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        sid_s, rng = item.split(":", 1)
        a_s, b_s = rng.split(":", 1)
        sid = int(sid_s)
        a = int(a_s)
        b = int(b_s)
        if a < 0 or b <= a or b > dim:
            raise ValueError(f"Invalid slice spec '{item}' for dim={dim}")
        out[sid] = slice(a, b)
    if not out:
        raise ValueError("--slices parsed to an empty mapping")
    return out


def _load_slices(args, dim):
    return _parse_slices(args.slices, dim)


def _precompute_direct_kdata(
    hk_up, hk_dn, slices, efermi, *, keep_eigenvectors=False
):
    nk, dim, _ = hk_up.shape
    eye = np.eye(dim, dtype=np.complex128)
    evals_up = np.zeros((nk, dim), dtype=np.float64)
    evals_dn = np.zeros((nk, dim), dtype=np.float64)
    evecs_up = (
        np.zeros((nk, dim, dim), dtype=np.complex128)
        if keep_eigenvectors
        else None
    )
    evecs_dn = (
        np.zeros((nk, dim, dim), dtype=np.complex128)
        if keep_eigenvectors
        else None
    )
    coeffs_up = {int(site): np.zeros((nk, slc.stop - slc.start, dim), dtype=np.complex128) for site, slc in slices.items()}
    coeffs_dn = {int(site): np.zeros((nk, slc.stop - slc.start, dim), dtype=np.complex128) for site, slc in slices.items()}

    for ik in range(nk):
        hu = 0.5 * (hk_up[ik] + hk_up[ik].conj().T) - float(efermi) * eye
        hd = 0.5 * (hk_dn[ik] + hk_dn[ik].conj().T) - float(efermi) * eye
        wu, cu = np.linalg.eigh(hu)
        wd, cd = np.linalg.eigh(hd)
        evals_up[ik] = np.real(wu)
        evals_dn[ik] = np.real(wd)
        if keep_eigenvectors:
            evecs_up[ik] = cu
            evecs_dn[ik] = cd
        for site, slc in slices.items():
            coeffs_up[int(site)][ik] = cu[slc, :]
            coeffs_dn[int(site)][ik] = cd[slc, :]

    h0_up = np.mean(hk_up, axis=0)
    h0_dn = np.mean(hk_dn, axis=0)
    d0 = h0_up - h0_dn
    delta = {}
    signs = {}
    for site, slc in slices.items():
        sid = int(site)
        dloc = np.array(d0[slc, slc], dtype=np.complex128)
        delta[sid] = 0.5 * (dloc + dloc.conj().T)
        tr = float(np.real(np.trace(delta[sid])))
        signs[sid] = -1.0 if tr < 0.0 else 1.0
    result = {
        "evals_up": evals_up,
        "evals_dn": evals_dn,
        "coeffs_up": coeffs_up,
        "coeffs_dn": coeffs_dn,
        "delta": delta,
        "signs": signs,
    }
    if keep_eigenvectors:
        result["evecs_up"] = evecs_up
        result["evecs_dn"] = evecs_dn
    return result


def _bond_key(i, j, r):
    return f"{int(i)}-{int(j)}@({int(r[0])},{int(r[1])},{int(r[2])})"


def _write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")


def run(args):
    kpts = _full_k_mesh(args.kmesh)
    kweights = np.full(len(kpts), 1.0 / float(max(1, len(kpts))), dtype=np.float64)
    print(f"[J-epr-kdiag] building direct H(k): nk={len(kpts)} kmesh={tuple(args.kmesh)}")
    hk_up = _build_hk_from_epr(args.epr_up, kpts, unit=args.hr_unit)
    hk_dn = _build_hk_from_epr(args.epr_dn, kpts, unit=args.hr_unit)
    dim = int(hk_up.shape[1])
    if hk_dn.shape[1] != dim:
        raise ValueError(f"up/down dimension mismatch: up={dim}, down={hk_dn.shape[1]}")

    slices = _load_slices(args, dim)
    print(f"[J-epr-kdiag] slices={ {int(k): (v.start, v.stop) for k, v in slices.items()} }")
    kdata = _precompute_direct_kdata(hk_up, hk_dn, slices, args.efermi)

    mag_atoms = [int(x) - 1 if args.mag_atoms_base == 1 else int(x) for x in args.mag_atoms]
    neighbours = _find_nearest_neighbours_from_epr(
        args.epr_up,
        mag_atom_indices=mag_atoms[:2],
        n_shells=int(args.n_shells),
        d_max=float(args.d_max),
        all_bonds=True,
    )
    if args.nn_only:
        neighbours = [n for n in neighbours if int(n.get("shell_idx", 0)) == 1]
    global_to_local = {g: i for i, g in enumerate(mag_atoms)}
    pair_meta = []
    for n in neighbours:
        gi, gj = int(n["i"]), int(n["j"])
        if gi not in global_to_local or gj not in global_to_local:
            continue
        r = tuple(int(x) for x in n["R"])
        pair_meta.append(
            {
                "gi": gi,
                "gj": gj,
                "li": int(global_to_local[gi]),
                "lj": int(global_to_local[gj]),
                "R": r,
                "dist": float(n["distance"]),
                "shell": int(n.get("shell_idx", 0)),
            }
        )
    if not pair_meta:
        raise RuntimeError("No directed bonds matched --mag_atoms.")

    if args.integrator == "contour":
        energy_mesh = get_semicircle_contour(emin=args.emin, emax=0.0, npoints=args.empoints)
    elif args.integrator == "cfr_ozaki":
        energy_mesh = get_cfr_ozaki_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)
    else:
        energy_mesh = get_cfr_pole_mesh(npoles=args.empoints, beta_eV_inv=args.cfr_beta)

    os.makedirs(args.out_dir, exist_ok=True)
    phase = np.exp(-1j * 2.0 * np.pi * (np.asarray([m["R"] for m in pair_meta], dtype=np.float64) @ kpts.T))
    j_acc = np.zeros(len(pair_meta), dtype=np.complex128)
    energy_rows = []

    for iz, (z, dz) in enumerate(energy_mesh):
        inv_up = 1.0 / (z - kdata["evals_up"])
        inv_dn = 1.0 / (z - kdata["evals_dn"])
        tr_vals = np.zeros(len(pair_meta), dtype=np.complex128)
        gu_norm = np.zeros(len(pair_meta), dtype=np.float64)
        gd_norm = np.zeros(len(pair_meta), dtype=np.float64)

        for ip, meta in enumerate(pair_meta):
            li = meta["li"]
            lj = meta["lj"]
            gu = np.zeros((slices[li].stop - slices[li].start, slices[lj].stop - slices[lj].start), dtype=np.complex128)
            gd = np.zeros((slices[lj].stop - slices[lj].start, slices[li].stop - slices[li].start), dtype=np.complex128)
            for ik in range(len(kpts)):
                gij = green_subblock_from_coefficients(kdata["coeffs_up"][li][ik], kdata["coeffs_up"][lj][ik], inv_up[ik])
                gji = green_subblock_from_coefficients(kdata["coeffs_dn"][lj][ik], kdata["coeffs_dn"][li][ik], inv_dn[ik])
                gu += gij * phase[ip, ik] * kweights[ik]
                gd += gji * np.conjugate(phase[ip, ik]) * kweights[ik]
            rel_sign = kdata["signs"].get(li, 1.0) * kdata["signs"].get(lj, 1.0)
            tr = np.trace(kdata["delta"][li] @ gu @ kdata["delta"][lj] @ gd) / rel_sign
            tr_vals[ip] = tr
            gu_norm[ip] = float(np.linalg.norm(gu))
            gd_norm[ip] = float(np.linalg.norm(gd))
            j_acc[ip] += tr * dz

        tr_by_key = {(m["gi"], m["gj"], m["R"]): tr_vals[ip] for ip, m in enumerate(pair_meta)}
        for ip, meta in enumerate(pair_meta):
            key = (meta["gi"], meta["gj"], meta["R"])
            mir = (meta["gj"], meta["gi"], tuple(-x for x in meta["R"]))
            tm = tr_by_key.get(mir, np.nan + 1j * np.nan)
            diff = tr_vals[ip] - tm
            energy_rows.append(
                [
                    iz,
                    _bond_key(*key),
                    _bond_key(*mir),
                    f"{np.real(z):.16e}",
                    f"{np.imag(z):.16e}",
                    f"{np.real(dz):.16e}",
                    f"{np.imag(dz):.16e}",
                    f"{np.real(tr_vals[ip]):.16e}",
                    f"{np.imag(tr_vals[ip]):.16e}",
                    f"{np.real(tm):.16e}",
                    f"{np.imag(tm):.16e}",
                    f"{abs(diff):.16e}",
                    f"{gu_norm[ip]:.16e}",
                    f"{gd_norm[ip]:.16e}",
                ]
            )

    j_mev = 1000.0 * np.imag(j_acc) / (4.0 * np.pi)
    j_by_key = {(m["gi"], m["gj"], m["R"]): float(j_mev[ip]) for ip, m in enumerate(pair_meta)}
    summary_rows = []
    for ip, meta in enumerate(pair_meta):
        key = (meta["gi"], meta["gj"], meta["R"])
        mir = (meta["gj"], meta["gi"], tuple(-x for x in meta["R"]))
        jm = j_by_key.get(mir, np.nan)
        summary_rows.append(
            [
                int(meta["shell"]),
                f"{meta['dist']:.10f}",
                _bond_key(*key),
                _bond_key(*mir),
                f"{float(j_mev[ip]):.16e}",
                f"{float(jm):.16e}",
                f"{float(j_mev[ip] - jm):.16e}" if np.isfinite(jm) else "nan",
            ]
        )

    summary_path = os.path.join(args.out_dir, args.summary_name)
    energy_path = os.path.join(args.out_dir, args.energy_name)
    _write_tsv(summary_path, ["shell", "dist_A", "bond", "mirror", "J_meV", "J_mirror_meV", "diff_meV"], summary_rows)
    _write_tsv(
        energy_path,
        [
            "iz",
            "bond",
            "mirror",
            "z_real",
            "z_imag",
            "weight_real",
            "weight_imag",
            "trace_real",
            "trace_imag",
            "mirror_trace_real",
            "mirror_trace_imag",
            "abs_trace_diff",
            "Gup_R_fro",
            "Gdn_minusR_fro",
        ],
        energy_rows,
    )
    print(f"[J-epr-kdiag] neighbour source=EPR basic_data/tau from {os.path.abspath(args.epr_up)}")
    print(f"[J-epr-kdiag] wrote {summary_path}")
    print(f"[J-epr-kdiag] wrote {energy_path}")


def main():
    ap = argparse.ArgumentParser(description="Direct EPR H(k) LKAG forward/mirror asymmetry diagnostic")
    ap.add_argument("--epr_up", required=True)
    ap.add_argument("--epr_dn", required=True)
    ap.add_argument("--hr_unit", choices=["ev", "ry", "ha"], default="ry")
    ap.add_argument("--mag_atoms", type=int, nargs="+", required=True)
    ap.add_argument("--mag_atoms_base", type=int, choices=[0, 1], default=0)
    ap.add_argument("--slices", required=True, help="Local slices as '0:0:5,1:5:10'")
    ap.add_argument("--efermi", type=float, required=True)
    ap.add_argument("--kmesh", type=int, nargs=3, required=True)
    ap.add_argument("--n_shells", type=int, default=1)
    ap.add_argument("--nn_only", action="store_true", default=True)
    ap.add_argument("--d_max", type=float, default=20.0)
    ap.add_argument("--emin", type=float, default=-25.0)
    ap.add_argument("--empoints", type=int, default=500)
    ap.add_argument("--integrator", choices=["contour", "cfr", "cfr_ozaki"], default="contour")
    ap.add_argument("--cfr_beta", type=float, default=400.0)
    ap.add_argument("--out_dir", default="J_epr_kdiag")
    ap.add_argument("--summary_name", default="summary.tsv")
    ap.add_argument("--energy_name", default="energy_components.tsv")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
