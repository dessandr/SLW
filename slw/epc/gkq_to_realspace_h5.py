import argparse
import json
import os

import h5py
import numpy as np


def _idx_to_centered(idx, n):
    idx = int(idx) % int(n)
    return idx if idx < (n // 2) else idx - int(n)


def _ifft_to_real(g_kq_qfirst, nk_grid, nq_grid):
    """
    Convert g(k,q)[q,k] -> g(Re,Rp)
    output shape: (nk1,nk2,nk3,nq1,nq2,nq3,m,n)
    """
    nq, nk, m, n = g_kq_qfirst.shape
    g_mlwf = np.transpose(g_kq_qfirst, (1, 0, 2, 3))  # (Nk, Nq, m, n)
    if nk != int(np.prod(nk_grid)):
        raise ValueError(f"Nk mismatch: {nk} vs nk_grid={nk_grid}")
    if nq != int(np.prod(nq_grid)):
        raise ValueError(f"Nq mismatch: {nq} vs nq_grid={nq_grid}")

    temp = g_mlwf.reshape(tuple(nk_grid[::-1]) + (nq, m, n)).transpose(2, 1, 0, 3, 4, 5)
    g_re_q = np.fft.ifftn(temp, axes=(0, 1, 2))
    temp2 = g_re_q.reshape(tuple(nk_grid) + tuple(nq_grid[::-1]) + (m, n)).transpose(0, 1, 2, 5, 4, 3, 6, 7)
    g_re_rp = np.fft.ifftn(temp2, axes=(3, 4, 5))
    return g_re_rp


def _as_text(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    if isinstance(v, np.bytes_):
        return bytes(v).decode("utf-8", errors="ignore")
    return str(v)


def _load_gkq(path):
    ap = os.path.abspath(path)
    with h5py.File(ap, "r") as h5:
        if "g_band" not in h5:
            raise KeyError(f"{ap} has no dataset 'g_band'")
        g_kq = np.asarray(h5["g_band"], dtype=np.complex128)
        nk_grid = tuple(int(x) for x in np.asarray(h5.attrs.get("nk_dense")))
        nq_grid = tuple(int(x) for x in np.asarray(h5.attrs.get("supercell_qmesh")))
        channel = _as_text(h5.attrs.get("channel", "dH_up"))
        label = _as_text(h5.attrs.get("label", "X1"))
        axis = _as_text(h5.attrs.get("axis", "x")).lower()
        basis = _as_text(h5.attrs.get("basis", "unknown"))
    return {
        "path": ap,
        "g_kq": g_kq,
        "nk_grid": nk_grid,
        "nq_grid": nq_grid,
        "channel": channel,
        "label": label,
        "axis": axis,
        "basis": basis,
    }


def _flatten_real(g_real, nk_grid, nq_grid, drop_tol):
    nk1, nk2, nk3 = [int(x) for x in nk_grid]
    nq1, nq2, nq3 = [int(x) for x in nq_grid]
    mats = []
    rp_list = []
    re_list = []
    for ir1 in range(nk1):
        r1 = _idx_to_centered(ir1, nk1)
        for ir2 in range(nk2):
            r2 = _idx_to_centered(ir2, nk2)
            for ir3 in range(nk3):
                r3 = _idx_to_centered(ir3, nk3)
                for ip1 in range(nq1):
                    p1 = _idx_to_centered(ip1, nq1)
                    for ip2 in range(nq2):
                        p2 = _idx_to_centered(ip2, nq2)
                        for ip3 in range(nq3):
                            p3 = _idx_to_centered(ip3, nq3)
                            mat = np.asarray(g_real[ir1, ir2, ir3, ip1, ip2, ip3], dtype=np.complex128)
                            if drop_tol > 0.0 and float(np.linalg.norm(mat)) < float(drop_tol):
                                continue
                            rp_list.append((p1, p2, p3))
                            re_list.append((r1, r2, r3))
                            mats.append(mat)
    if mats:
        return (
            np.asarray(rp_list, dtype=np.int32),
            np.asarray(re_list, dtype=np.int32),
            np.stack(mats, axis=0),
        )
    dim = int(g_real.shape[-1])
    return (
        np.zeros((0, 3), dtype=np.int32),
        np.zeros((0, 3), dtype=np.int32),
        np.zeros((0, dim, dim), dtype=np.complex128),
    )


def main():
    p = argparse.ArgumentParser(
        description="Convert g(k,q) HDF5 to extract_v2-style real-space HDF5 for view_dhdu_gui."
    )
    p.add_argument("gkq_h5", help="Input g(k,q) h5 (contains g_band, attrs channel/label/axis)")
    p.add_argument("--out_h5", required=True, help="Output extract_v2-style h5")
    p.add_argument("--channel", default=None, help="Override output channel")
    p.add_argument("--label", default=None, help="Override output label")
    p.add_argument("--axis", default=None, choices=["x", "y", "z"], help="Override output axis")
    p.add_argument("--drop_tol", type=float, default=0.0, help="Drop blocks with ||mat||_F < drop_tol")
    p.add_argument("--compression", choices=["none", "lzf", "gzip"], default="lzf")
    p.add_argument("--gzip_level", type=int, default=4)
    args = p.parse_args()

    info = _load_gkq(args.gkq_h5)
    channel = args.channel if args.channel else info["channel"]
    label = args.label if args.label else info["label"]
    axis = args.axis if args.axis else info["axis"]

    g_real = _ifft_to_real(info["g_kq"], info["nk_grid"], info["nq_grid"])
    rp_arr, re_arr, mats = _flatten_real(g_real, info["nk_grid"], info["nq_grid"], drop_tol=float(args.drop_tol))

    out_path = os.path.abspath(args.out_h5)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    kwargs = {}
    if args.compression != "none":
        kwargs["compression"] = args.compression
        if args.compression == "gzip":
            kwargs["compression_opts"] = int(args.gzip_level)
        if mats.shape[0] > 0:
            kwargs["shuffle"] = True

    with h5py.File(out_path, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["format"] = "extract_v2_bundle"
        meta.attrs["source"] = "gkq_to_realspace_h5"
        meta.attrs["basis"] = info["basis"]
        meta.attrs["nk_dense"] = np.asarray(info["nk_grid"], dtype=np.int32)
        meta.attrs["supercell_qmesh"] = np.asarray(info["nq_grid"], dtype=np.int32)
        meta.attrs["origin_gkq_h5"] = info["path"]

        data = h5.create_group("data")
        ch = data.create_group(str(channel))
        lb = ch.create_group(str(label))
        ax = lb.create_group(str(axis))
        ax.attrs["channel"] = str(channel)
        ax.attrs["label"] = str(label)
        ax.attrs["axis"] = str(axis)
        ax.attrs["source"] = "ifft_gkq"
        ax.attrs["nk_dense"] = np.asarray(info["nk_grid"], dtype=np.int32)
        ax.attrs["supercell_qmesh"] = np.asarray(info["nq_grid"], dtype=np.int32)
        ax.attrs["drop_tol"] = float(args.drop_tol)
        ax.attrs["mapping_note"] = "matrices are g(Re,Rp) converted from g(k,q) by double IFFT"

        ax.create_dataset("Rp_list", data=rp_arr, **kwargs)
        ax.create_dataset("Rp_vec_list", data=rp_arr, **kwargs)
        ax.create_dataset("R_list", data=re_arr, **kwargs)
        ax.create_dataset("Re_list", data=re_arr, **kwargs)
        ax.create_dataset("matrices", data=mats, **kwargs)

    report = {
        "input": info["path"],
        "output": out_path,
        "channel": channel,
        "label": label,
        "axis": axis,
        "basis": info["basis"],
        "nk_grid": list(info["nk_grid"]),
        "nq_grid": list(info["nq_grid"]),
        "n_blocks": int(mats.shape[0]),
        "dim": int(mats.shape[-1]) if mats.ndim == 3 else 0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
