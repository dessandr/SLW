"""Compatibility g(k,q) reader retained by the native LKAG primitives."""
import argparse
import os
import pickle

import h5py
import numpy as np
from scipy.fft import ifftn


def _ifft_to_real(g_kq_qfirst, nk_grid, nq_grid):
    nq, nk, m, n = g_kq_qfirst.shape
    g_mlwf = np.transpose(g_kq_qfirst, (1, 0, 2, 3))

    if nk != int(np.prod(nk_grid)):
        raise ValueError(f"Nk={nk} incompatible with nk_grid={nk_grid}")
    if nq != int(np.prod(nq_grid)):
        raise ValueError(f"Nq={nq} incompatible with nq_grid={nq_grid}")

    temp = g_mlwf.reshape(tuple(nk_grid[::-1]) + (nq, m, n)).transpose(2, 1, 0, 3, 4, 5)
    g_re_q = ifftn(temp, axes=(0, 1, 2), workers=-1)

    temp2 = g_re_q.reshape(tuple(nk_grid) + tuple(nq_grid[::-1]) + (m, n)).transpose(0, 1, 2, 5, 4, 3, 6, 7)
    g_re_rp = ifftn(temp2, axes=(3, 4, 5), workers=-1)
    return g_re_rp

def _spin_from_channel(channel):
    if channel == "dH_up":
        return "up"
    if channel == "dH_dn":
        return "down"
    raise ValueError(f"Unsupported channel for exchange adapter: {channel}")

_G_REAL_CACHE = {}

def _load_single_gkq(path):
    ap = os.path.abspath(path)
    if ap in _G_REAL_CACHE:
        return _G_REAL_CACHE[ap]

    with h5py.File(path, "r") as h5:
        if "g_band" not in h5:
            raise KeyError(f"{path} has no 'g_band' dataset")
        g_kq = np.asarray(h5["g_band"], dtype=np.complex128)  # (Nq, Nk, m, n)

        channel = str(h5.attrs.get("channel", ""))
        label = str(h5.attrs.get("label", ""))
        axis = str(h5.attrs.get("axis", ""))
        basis = str(h5.attrs.get("basis", ""))
        nk_grid = tuple(int(x) for x in np.asarray(h5.attrs.get("nk_dense")))
        nq_grid = tuple(int(x) for x in np.asarray(h5.attrs.get("supercell_qmesh")))

    if channel == "" or label == "" or axis == "":
        raise ValueError(f"{path} missing required attrs (channel/label/axis)")

    spin = _spin_from_channel(channel)
    g_real = _ifft_to_real(g_kq, nk_grid=nk_grid, nq_grid=nq_grid)

    res = {
        "key": (spin, label, axis),
        "g_real": g_real,
        "nk_grid": nk_grid,
        "nq_grid": nq_grid,
        "basis": basis,
    }
    _G_REAL_CACHE[ap] = res
    return res


def main():
    parser = argparse.ArgumentParser(description="Adapt g_kq HDF5 files to legacy exchange g_real pickle format.")
    parser.add_argument("inputs", nargs="+", help="Input g_kq*.h5 files")
    parser.add_argument("--out", required=True, help="Output legacy *.real.pkl path")
    parser.add_argument(
        "--units",
        choices=["ha", "eV"],
        default="ha",
        help="Units encoded in output payload. 'ha' omits units tag so LKAG loader applies HA->eV conversion; 'eV' writes units='eV'.",
    )
    args = parser.parse_args()

    g_real_dict = {}
    nk_grid_ref = None
    nq_grid_ref = None
    bases = set()

    for p in args.inputs:
        ap = os.path.abspath(p)
        info = _load_single_gkq(ap)
        key = info["key"]
        if key in g_real_dict:
            raise ValueError(f"Duplicate key {key} from input {ap}")
        if nk_grid_ref is None:
            nk_grid_ref = info["nk_grid"]
            nq_grid_ref = info["nq_grid"]
        else:
            if tuple(info["nk_grid"]) != tuple(nk_grid_ref) or tuple(info["nq_grid"]) != tuple(nq_grid_ref):
                raise ValueError(
                    f"Grid mismatch for {ap}: nk/nq={info['nk_grid']}/{info['nq_grid']} vs ref {nk_grid_ref}/{nq_grid_ref}"
                )
        g_real_dict[key] = info["g_real"]
        if info["basis"]:
            bases.add(info["basis"])
        print(f"[adapter] loaded {ap} -> key={key}, shape={info['g_real'].shape}")

    payload = {
        "g_real_dict": g_real_dict,
        "nk_grid": list(nk_grid_ref),
        "nq_grid": list(nq_grid_ref),
        "basis": ",".join(sorted(bases)) if bases else "unknown",
    }
    if args.units == "eV":
        payload["units"] = "eV"

    out_path = os.path.abspath(args.out)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[adapter] wrote legacy g_real payload: {out_path}")
    print(f"[adapter] entries={len(g_real_dict)} nk_grid={nk_grid_ref} nq_grid={nq_grid_ref} units={args.units}")


if __name__ == "__main__":
    main()
