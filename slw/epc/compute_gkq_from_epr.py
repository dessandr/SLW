import argparse
from concurrent.futures import ThreadPoolExecutor
import os

import h5py
import numpy as np

from slw.core.set_ws_cell import init_rvec_images, set_wigner_seitz_cell
from slw.core.wannier_io import read_wannier_u_matrix
from slw.epc.epr_io import read_epr_metadata


def _phase_plus(fracs, rvecs):
    # Fortran-compatible sign: exp(+i 2pi k·R)
    dots = 2.0 * np.pi * (np.asarray(fracs, dtype=np.float64) @ np.asarray(rvecs, dtype=np.float64).T)
    return np.exp(1j * dots)


def _uniform_frac_grid(mesh):
    n1, n2, n3 = (int(x) for x in mesh)
    return np.asarray(
        [(i / n1, j / n2, k / n3) for i in range(n1) for j in range(n2) for k in range(n3)],
        dtype=np.float64,
    )


def _frac_key(frac, mesh):
    out = []
    for x, n in zip(frac, mesh):
        xi = int(np.rint((x % 1.0) * n)) % n
        out.append(xi)
    return tuple(out)


def _build_kq_map(k_fracs, q_fracs, nk_dense):
    k_lookup = {_frac_key(kf, nk_dense): ik for ik, kf in enumerate(k_fracs)}
    kq_map = np.zeros((len(k_fracs), len(q_fracs)), dtype=np.int32)
    for ik, kf in enumerate(k_fracs):
        for iq, qf in enumerate(q_fracs):
            key = _frac_key(np.mod(kf + qf, 1.0), nk_dense)
            if key not in k_lookup:
                raise KeyError(f"k+q point not found for ik={ik}, iq={iq}, key={key}")
            kq_map[ik, iq] = k_lookup[key]
    return kq_map


def _load_u_pair(nk_tot, u_mat_path=None, u_dis_mat_path=None):
    u_rot = read_wannier_u_matrix(u_mat_path, nk_tot) if u_mat_path else None
    u_dis = read_wannier_u_matrix(u_dis_mat_path, nk_tot) if u_dis_mat_path else None
    if u_rot is None and u_dis is None:
        return None
    if u_rot is None:
        return u_dis
    if u_dis is None:
        return u_rot
    nk_u, nband, nwan = u_dis.shape
    out = np.zeros((nk_u, nband, nwan), dtype=np.complex128)
    for ik in range(nk_u):
        out[ik] = u_dis[ik] @ u_rot[ik]
    return out


def _rotate_iq_chunk(g_wannier_chunk, u_k, kq_map_chunk, mode_block):
    # g_wannier_chunk: (nq_chunk,nk,nwan,nwan,nmode)
    # u_k: (nk,nband,nwan)
    nq_chunk, nk, nwan, _, nmode = g_wannier_chunk.shape
    nband = u_k.shape[1]
    out = np.zeros((nq_chunk, nk, nband, nband, nmode), dtype=np.complex128)
    uk_right = np.transpose(u_k, (0, 2, 1))  # (nk,nwan,nband)

    for iq_loc in range(nq_chunk):
        ukq_conj = u_k[kq_map_chunk[iq_loc], :, :].conj()  # (nk,nband,nwan)
        for m0 in range(0, nmode, mode_block):
            m1 = min(m0 + mode_block, nmode)
            gw = g_wannier_chunk[iq_loc, :, :, :, m0:m1]  # (nk,nwan,nwan,mb)
            # tmp[k,a,j,m] = sum_i U*(k+q)[k,a,i] * g[k,i,j,m]
            tmp = np.einsum("kai,kijm->kajm", ukq_conj, gw, optimize=True)
            # out[k,a,b,m] = sum_j tmp[k,a,j,m] * U(k)[k,j,b]
            out[iq_loc, :, :, :, m0:m1] = np.einsum("kajm,kjb->kabm", tmp, uk_right, optimize=True)
    return out


def _rotate_gkq_to_band(g_wannier, u_k, kq_map, nproc=1, mode_block=6):
    nq, nk, nwan_i, nwan_j, nmode = g_wannier.shape
    if nwan_i != nwan_j:
        raise ValueError(f"g_wannier must be square in Wannier indices, got {g_wannier.shape}")
    if u_k.shape[0] != nk:
        raise ValueError(f"U(k) Nk mismatch: U has {u_k.shape[0]}, g has {nk}")
    nband, nwan_u = u_k.shape[1], u_k.shape[2]
    if nwan_u != nwan_i:
        raise ValueError(f"U(k) Wannier-dim mismatch: U has {nwan_u}, g has {nwan_i}")

    kq_map_qk = np.asarray(kq_map, dtype=np.int32).T  # (Nq,Nk)
    nproc = max(1, int(nproc))
    mode_block = max(1, int(mode_block))

    if nproc == 1 or nq == 1:
        return _rotate_iq_chunk(g_wannier, u_k, kq_map_qk, mode_block)

    # Split q axis across workers
    edges = np.linspace(0, nq, nproc + 1, dtype=np.int32)
    chunks = [(int(edges[i]), int(edges[i + 1])) for i in range(nproc) if int(edges[i + 1]) > int(edges[i])]
    out = np.zeros((nq, nk, nband, nband, nmode), dtype=np.complex128)
    with ThreadPoolExecutor(max_workers=nproc) as ex:
        futs = []
        for q0, q1 in chunks:
            futs.append(
                (
                    q0,
                    q1,
                    ex.submit(
                        _rotate_iq_chunk,
                        g_wannier[q0:q1],
                        u_k,
                        kq_map_qk[q0:q1],
                        mode_block,
                    ),
                )
            )
        for q0, q1, fut in futs:
            out[q0:q1] = fut.result()
    return out


def build_gkq_wannier_from_epr(epr_path):
    """
    Build g_wannier(k,q) from EPR ep_hop with the same phase convention as
    pert/elphon_coupling_matrix.f90:
      g_kerp(k, Rp) = sum_Re ep_hop(Re,Rp) exp(+i 2pi k·Re)
      gkq(k, q)     = sum_Rp g_kerp(k,Rp) exp(+i 2pi q·Rp)
    Returns:
      gkq: complex ndarray (Nq, Nk, nwan, nwan, 3*nat)
      k_fracs: (Nk,3), q_fracs: (Nq,3)
      meta: dict
    """
    with h5py.File(epr_path, "r") as h5:
        meta = read_epr_metadata(h5)
        grp = h5["eph_matrix_wannier"]
        nk = tuple(int(x) for x in meta.nk_grid)
        nq = tuple(int(x) for x in meta.nq_grid)
        nat = int(meta.nat)
        nwan = int(meta.nwan)
        nk_tot = int(np.prod(nk))
        nq_tot = int(np.prod(nq))

        k_fracs = _uniform_frac_grid(nk)
        q_fracs = _uniform_frac_grid(nq)
        gkq = np.zeros((nq_tot, nk_tot, nwan, nwan, 3 * nat), dtype=np.complex128)

        el_images = init_rvec_images(nk, meta.at)
        ph_images = init_rvec_images(nq, meta.at)
        ws_cache = {}

        for ia in range(1, nat + 1):
            for jw in range(1, nwan + 1):
                for iw in range(1, nwan + 1):
                    dr = f"ep_hop_r_{ia}_{jw}_{iw}"
                    di = f"ep_hop_i_{ia}_{jw}_{iw}"
                    if dr not in grp or di not in grp:
                        continue
                    arr = (np.asarray(grp[dr], dtype=np.float64) + 1j * np.asarray(grp[di], dtype=np.float64)).transpose(
                        2, 1, 0
                    )  # (3,nre,nrp)

                    key = (ia, jw, iw)
                    cached = ws_cache.get(key)
                    if cached is None:
                        ws_el = set_wigner_seitz_cell(el_images, meta.at, meta.wc[iw - 1], meta.wc[jw - 1])
                        ws_ph = set_wigner_seitz_cell(ph_images, meta.at, meta.wc[iw - 1], meta.tau[ia - 1])
                        ws_cache[key] = (ws_el, ws_ph)
                    else:
                        ws_el, ws_ph = cached

                    re_vecs = np.asarray(ws_el[0], dtype=np.float64)
                    rp_vecs = np.asarray(ws_ph[0], dtype=np.float64)
                    if arr.shape[1] != re_vecs.shape[0] or arr.shape[2] != rp_vecs.shape[0]:
                        raise ValueError(
                            f"WS mismatch ia={ia},jw={jw},iw={iw}: arr={arr.shape[1:]}, "
                            f"ws={(re_vecs.shape[0], rp_vecs.shape[0])}"
                        )

                    phase_k_re = _phase_plus(k_fracs, re_vecs)  # (Nk,nre)
                    phase_q_rp = _phase_plus(q_fracs, rp_vecs)  # (Nq,nrp)

                    # g_kerp(k, axis, rp): (Nk,3,nrp)
                    g_kerp = np.einsum("kr,arp->kap", phase_k_re, arr, optimize=True)
                    # gkq(q,k,axis) for this (iw,jw,ia)
                    g_qk_axis = np.einsum("qp,kap->qka", phase_q_rp, g_kerp, optimize=True)

                    for ax in range(3):
                        mode = (ia - 1) * 3 + ax
                        gkq[:, :, iw - 1, jw - 1, mode] = g_qk_axis[:, :, ax]

    info = {
        "nk_grid": list(nk),
        "nq_grid": list(nq),
        "nat": nat,
        "nwan": nwan,
        "source_epr": os.path.abspath(epr_path),
        "convention": "g_kerp=sum_Re exp(+i2pi k.Re), gkq=sum_Rp exp(+i2pi q.Rp)",
    }
    return gkq, k_fracs, q_fracs, info


def save_gkq_h5(path, gkq, k_fracs, q_fracs, meta, dataset_name="g_wannier", kq_map=None):
    out = os.path.abspath(path)
    with h5py.File(out, "w") as h5:
        h5.create_dataset(dataset_name, data=gkq, compression="lzf")
        h5.create_dataset("k_fracs", data=np.asarray(k_fracs, dtype=np.float64))
        h5.create_dataset("q_fracs", data=np.asarray(q_fracs, dtype=np.float64))
        if kq_map is not None:
            h5.create_dataset("kq_map", data=np.asarray(kq_map, dtype=np.int32))
        for k, v in meta.items():
            if isinstance(v, (list, tuple)):
                h5.attrs[k] = np.asarray(v)
            else:
                h5.attrs[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description="Build Wannier-gauge g(k,q) from EPR ep_hop (Fortran phase convention).")
    ap.add_argument("--epr", required=True, help="Input EPR HDF5 path")
    ap.add_argument("--out", default="gkq_wannier_from_epr.h5", help="Output HDF5 path")
    ap.add_argument("--u_mat", default=None, help="wannier90_u.mat path (optional)")
    ap.add_argument("--u_dis_mat", default=None, help="wannier90_u_dis.mat path (optional)")
    ap.add_argument(
        "--rotate",
        action="store_true",
        help="Apply U_{k+q}^dagger g U_k rotation and write dataset g_band",
    )
    ap.add_argument("--nproc", type=int, default=1, help="Thread parallelism for rotation over q points")
    ap.add_argument("--mode_block", type=int, default=6, help="Mode block size for batched einsum rotation")
    args = ap.parse_args()

    gkq, kf, qf, meta = build_gkq_wannier_from_epr(args.epr)
    if args.rotate:
        nk_mesh = tuple(int(x) for x in meta["nk_grid"])
        kq_map = _build_kq_map(kf, qf, nk_mesh)
        u_k = _load_u_pair(len(kf), u_mat_path=args.u_mat, u_dis_mat_path=args.u_dis_mat)
        if u_k is None:
            raise ValueError("--rotate requested but no U matrix provided. Use --u_mat (and optionally --u_dis_mat).")
        g_band = _rotate_gkq_to_band(gkq, u_k, kq_map, nproc=args.nproc, mode_block=args.mode_block)
        meta["rotation"] = "g_band = U(k+q)^dagger g_wannier U(k)"
        meta["rotation_nproc"] = int(args.nproc)
        meta["rotation_mode_block"] = int(args.mode_block)
        if args.u_mat:
            meta["u_mat"] = os.path.abspath(args.u_mat)
        if args.u_dis_mat:
            meta["u_dis_mat"] = os.path.abspath(args.u_dis_mat)
        out = save_gkq_h5(args.out, g_band, kf, qf, meta, dataset_name="g_band", kq_map=kq_map)
        print(f"wrote {out}")
        print(f"shape g_band = {g_band.shape} (Nq,Nk,nband,nband,nmode)")
    else:
        out = save_gkq_h5(args.out, gkq, kf, qf, meta, dataset_name="g_wannier")
        print(f"wrote {out}")
        print(f"shape g_wannier = {gkq.shape} (Nq,Nk,nwan,nwan,nmode)")


if __name__ == "__main__":
    main()
