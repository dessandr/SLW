# Legacy exchange implementation; use the native engine for new workflows.
import os
import pickle
from dataclasses import dataclass

import h5py
import numpy as np

from slw.core.set_ws_cell import init_rvec_images, set_wigner_seitz_cell
from slw.core.qe2pert_ws import triangular_pair_index

RY_TO_EV = 13.605693122994
HA_TO_EV = 2.0 * RY_TO_EV


def _unit_scale_to_ev(unit: str) -> float:
    u = str(unit).strip().lower()
    if u == "ev":
        return 1.0
    if u in {"ry", "ryd", "rydberg"}:
        return RY_TO_EV
    if u in {"ha", "hartree"}:
        return HA_TO_EV
    raise ValueError(f"Unsupported unit='{unit}'. Use ev|ry|ha.")


@dataclass
class EPRMeta:
    nat: int
    nwan: int
    nk_grid: tuple[int, int, int]
    nq_grid: tuple[int, int, int]
    at: np.ndarray
    tau: np.ndarray
    wc: np.ndarray


def _read_meta(h5: h5py.File) -> EPRMeta:
    nat = int(np.asarray(h5["basic_data/nat"]))
    nwan = int(np.asarray(h5["basic_data/num_wann"]))
    nk = tuple(int(x) for x in np.asarray(h5["basic_data/kc_dim"]).reshape(3))
    nq = tuple(int(x) for x in np.asarray(h5["basic_data/qc_dim"]).reshape(3))
    # Keep the same convention already validated in check_qe2pert_epr_dispersion:
    # - 'at' in EPR HDF5 is transposed vs qe2pert_ws expectation.
    at = np.asarray(h5["basic_data/at"], dtype=np.float64).T
    tau_raw = np.asarray(h5["basic_data/tau"], dtype=np.float64)
    wc = np.asarray(h5["basic_data/wannier_center_cryst"], dtype=np.float64)
    tau = np.linalg.solve(at, tau_raw.T).T
    return EPRMeta(nat=nat, nwan=nwan, nk_grid=nk, nq_grid=nq, at=at, tau=tau, wc=wc)





def _normalize_atom_labels(atom_labels, nat: int) -> list[str]:
    if atom_labels is None:
        return [f"A{i + 1}" for i in range(nat)]
    labels = [str(x).strip() for x in atom_labels if str(x).strip()]
    if len(labels) != nat:
        raise ValueError(f"atom_labels length={len(labels)} but nat={nat}")
    return labels


def _idx_from_ws_vectors(ws_vec: np.ndarray, mesh: tuple[int, int, int]) -> np.ndarray:
    m = np.asarray(mesh, dtype=np.int64)
    v = np.asarray(ws_vec, dtype=np.int64)
    vv = np.mod(v, m[None, :])
    return (vv[:, 0] * (m[1] * m[2]) + vv[:, 1] * m[2] + vv[:, 2]).astype(np.int64, copy=False)


def _accumulate_pair_block(
    dense_pair: np.ndarray,
    re_idx: np.ndarray,
    rp_idx: np.ndarray,
    block_re_rp: np.ndarray,
):
    nqtot = dense_pair.shape[1]
    flat = dense_pair.reshape(-1)
    lin = (re_idx[:, None] * nqtot + rp_idx[None, :]).reshape(-1)
    np.add.at(flat, lin, block_re_rp.reshape(-1))


def _centered_r_vectors(mesh: tuple[int, int, int]) -> np.ndarray:
    n1, n2, n3 = (int(x) for x in mesh)
    r = np.asarray([(i, j, k) for i in range(n1) for j in range(n2) for k in range(n3)], dtype=np.int64)
    half = np.asarray([n1 // 2, n2 // 2, n3 // 2], dtype=np.int64)
    dims = np.asarray([n1, n2, n3], dtype=np.int64)
    gt = r > half[None, :]
    r = r - gt.astype(np.int64) * dims[None, :]
    return r


def build_hr_tensor_from_epr(epr_path: str, *, divide_by_ndeg: bool = False, unit: str = "ry"):
    with h5py.File(epr_path, "r") as h5:
        meta = _read_meta(h5)
        el_images = init_rvec_images(meta.nk_grid, meta.at)
        nk_tot = int(np.prod(meta.nk_grid))
        hr = np.zeros((nk_tot, meta.nwan, meta.nwan), dtype=np.complex128)
        ws_cache = {}
        for jw in range(1, meta.nwan + 1):
            for iw in range(1, jw + 1):
                tri = triangular_pair_index(iw, jw)
                d_r = f"electron_wannier/hopping_r{tri}"
                d_i = f"electron_wannier/hopping_i{tri}"
                if d_r not in h5 or d_i not in h5:
                    continue
                hop = np.asarray(h5[d_r], dtype=np.float64) + 1j * np.asarray(h5[d_i], dtype=np.float64)
                key = (iw, jw)
                ws = ws_cache.get(key)
                if ws is None:
                    ws = set_wigner_seitz_cell(el_images, meta.at, meta.wc[iw - 1], meta.wc[jw - 1])
                    ws_cache[key] = ws
                ws_vectors, ws_ndeg = ws
                if hop.shape[0] != len(ws_vectors):
                    raise ValueError(
                        f"hopping length mismatch for pair (iw={iw},jw={jw}): "
                        f"h5={hop.shape[0]} ws.nr={len(ws_vectors)}"
                    )
                idx = _idx_from_ws_vectors(ws_vectors, meta.nk_grid)
                if divide_by_ndeg:
                    hop_eff = hop / np.asarray(ws_ndeg, dtype=np.float64)
                else:
                    hop_eff = hop
                np.add.at(hr[:, iw - 1, jw - 1], idx, hop_eff)
                if iw != jw:
                    np.add.at(hr[:, jw - 1, iw - 1], idx, np.conjugate(hop_eff))
        # Hermitize for numerical stability.
        hr = 0.5 * (hr + np.swapaxes(hr.conj(), 1, 2))
        hr *= _unit_scale_to_ev(unit)
    return hr, _centered_r_vectors(meta.nk_grid)


def write_hr_dat_from_epr(epr_path: str, out_hr_path: str, *, divide_by_ndeg: bool = False, unit: str = "ry"):
    unit_scale = _unit_scale_to_ev(unit)
    with h5py.File(epr_path, "r") as h5:
        meta = _read_meta(h5)
        el_images = init_rvec_images(meta.nk_grid, meta.at)
        ws_cache = {}
        hdict = {}
        for jw in range(1, meta.nwan + 1):
            for iw in range(1, jw + 1):
                tri = triangular_pair_index(iw, jw)
                d_r = f"electron_wannier/hopping_r{tri}"
                d_i = f"electron_wannier/hopping_i{tri}"
                if d_r not in h5 or d_i not in h5:
                    continue
                hop = np.asarray(h5[d_r], dtype=np.float64) + 1j * np.asarray(h5[d_i], dtype=np.float64)
                key = (iw, jw)
                ws = ws_cache.get(key)
                if ws is None:
                    ws = set_wigner_seitz_cell(el_images, meta.at, meta.wc[iw - 1], meta.wc[jw - 1])
                    ws_cache[key] = ws
                ws_vectors, ws_ndeg = ws
                if hop.shape[0] != len(ws_vectors):
                    raise ValueError(
                        f"hopping length mismatch for pair (iw={iw},jw={jw}): "
                        f"h5={hop.shape[0]} ws.nr={len(ws_vectors)}"
                    )
                if divide_by_ndeg:
                    hop_eff = hop / np.asarray(ws_ndeg, dtype=np.float64)
                else:
                    hop_eff = hop
                hop_eff = hop_eff * unit_scale
                i0 = iw - 1
                j0 = jw - 1
                for t in range(len(ws_vectors)):
                    r = tuple(int(x) for x in ws_vectors[t])
                    mat = hdict.get(r)
                    if mat is None:
                        mat = np.zeros((meta.nwan, meta.nwan), dtype=np.complex128)
                        hdict[r] = mat
                    v = hop_eff[t]
                    mat[i0, j0] += v
                    rm = (-r[0], -r[1], -r[2])
                    matm = hdict.get(rm)
                    if matm is None:
                        matm = np.zeros((meta.nwan, meta.nwan), dtype=np.complex128)
                        hdict[rm] = matm
                    matm[j0, i0] += np.conjugate(v)

    # Enforce exact real-space Hermiticity: H(R) = H(-R)^\dagger.
    for r in list(hdict.keys()):
        rm = (-r[0], -r[1], -r[2])
        a = hdict[r]
        b = hdict.get(rm)
        if b is None:
            b = np.zeros_like(a)
            hdict[rm] = b
        avg = 0.5 * (a + b.conj().T)
        hdict[r] = avg
        hdict[rm] = avg.conj().T

    rvec = sorted(hdict.keys())
    nwan = int(meta.nwan)
    nr = int(len(rvec))
    degen = np.ones((nr,), dtype=np.int64)
    outp = os.path.abspath(out_hr_path)
    with open(outp, "w") as f:
        f.write("Generated from epr.h5 electron_wannier\n")
        f.write(f"{nwan}\n")
        f.write(f"{nr}\n")
        for i in range(0, nr, 15):
            row = " ".join(str(int(x)) for x in degen[i : i + 15])
            f.write(row + "\n")
        for ir in range(nr):
            r1, r2, r3 = rvec[ir]
            block = hdict[(r1, r2, r3)]
            for m in range(nwan):
                for n in range(nwan):
                    v = block[m, n]
                    f.write(f"{r1:5d}{r2:5d}{r3:5d}{m+1:5d}{n+1:5d}{v.real:22.14e}{v.imag:22.14e}\n")
    return outp


def write_hr_dat_from_epr_ifft(epr_path: str, out_hr_path: str, *, unit: str = "ry"):
    """
    Build H(k) on the full uniform kc grid directly from EPR electron_wannier,
    then inverse FFT to H(R) and write Wannier-style hr.dat.
    This path is gauge-consistent for band interpolation against EPR-direct H(k).
    """
    unit_scale = _unit_scale_to_ev(unit)
    with h5py.File(epr_path, "r") as h5:
        meta = _read_meta(h5)
        n1, n2, n3 = (int(x) for x in meta.nk_grid)
        nk_tot = n1 * n2 * n3
        kpts = np.asarray(
            [(i / n1, j / n2, k / n3) for i in range(n1) for j in range(n2) for k in range(n3)],
            dtype=np.float64,
        )
        images = init_rvec_images(meta.nk_grid, meta.at)
        hk = np.zeros((nk_tot, meta.nwan, meta.nwan), dtype=np.complex128)
        ws_cache = {}
        for jw in range(1, meta.nwan + 1):
            for iw in range(1, jw + 1):
                tri = triangular_pair_index(iw, jw)
                d_r = f"electron_wannier/hopping_r{tri}"
                d_i = f"electron_wannier/hopping_i{tri}"
                if d_r not in h5 or d_i not in h5:
                    continue
                hop = np.asarray(h5[d_r], dtype=np.float64) + 1j * np.asarray(h5[d_i], dtype=np.float64)
                key = (iw, jw)
                ws = ws_cache.get(key)
                if ws is None:
                    ws = set_wigner_seitz_cell(images, meta.at, meta.wc[iw - 1], meta.wc[jw - 1])
                    ws_cache[key] = ws
                ws_vectors = ws[0]
                if hop.shape[0] != len(ws_vectors):
                    raise ValueError(
                        f"hopping length mismatch for pair (iw={iw},jw={jw}): "
                        f"h5={hop.shape[0]} ws.nr={len(ws_vectors)}"
                    )
                phase = np.exp(2j * np.pi * (kpts @ np.asarray(ws_vectors, dtype=np.float64).T))
                vals = phase @ hop
                hk[:, iw - 1, jw - 1] = vals
                if iw != jw:
                    hk[:, jw - 1, iw - 1] = np.conjugate(vals)
        hk = 0.5 * (hk + np.swapaxes(hk.conj(), 1, 2))
        hk *= unit_scale

    hk_grid = hk.reshape(n1, n2, n3, meta.nwan, meta.nwan)
    hr_grid = np.fft.ifftn(hk_grid, axes=(0, 1, 2))
    hr_grid = np.fft.fftshift(hr_grid, axes=(0, 1, 2))
    r1 = np.arange(n1) - n1 // 2
    r2 = np.arange(n2) - n2 // 2
    r3 = np.arange(n3) - n3 // 2
    hdict = {}
    for i, a in enumerate(r1):
        for j, b in enumerate(r2):
            for k, c in enumerate(r3):
                hdict[(int(a), int(b), int(c))] = hr_grid[i, j, k]

    # enforce exact H(R)=H(-R)^dagger
    for r in list(hdict.keys()):
        rm = (-r[0], -r[1], -r[2])
        if rm in hdict:
            avg = 0.5 * (hdict[r] + hdict[rm].conj().T)
            hdict[r] = avg
            hdict[rm] = avg.conj().T

    rvec = sorted(hdict.keys())
    outp = os.path.abspath(out_hr_path)
    with open(outp, "w") as f:
        f.write("Generated from epr.h5 electron_wannier via H(k)->IFFT\n")
        f.write(f"{int(meta.nwan)}\n")
        f.write(f"{int(len(rvec))}\n")
        degen = np.ones((len(rvec),), dtype=np.int64)
        for i in range(0, len(rvec), 15):
            f.write(" ".join(str(int(x)) for x in degen[i : i + 15]) + "\n")
        for r in rvec:
            block = hdict[r]
            for m in range(meta.nwan):
                for n in range(meta.nwan):
                    v = block[m, n]
                    f.write(f"{r[0]:5d}{r[1]:5d}{r[2]:5d}{m+1:5d}{n+1:5d}{v.real:22.14e}{v.imag:22.14e}\n")
    return outp


def _build_spin_greal(
    epr_path: str,
    atom_labels: list[str],
    *,
    divide_by_ndeg: bool = False,
) -> dict[tuple[str, str, str], np.ndarray]:
    out = {}
    with h5py.File(epr_path, "r") as h5:
        meta = _read_meta(h5)
        grp = h5["eph_matrix_wannier"]
        el_images = init_rvec_images(meta.nk_grid, meta.at)
        ph_images = init_rvec_images(meta.nq_grid, meta.at)
        nk1, nk2, nk3 = meta.nk_grid
        nq1, nq2, nq3 = meta.nq_grid
        nk_tot = nk1 * nk2 * nk3
        nq_tot = nq1 * nq2 * nq3

        ws_cache = {}
        for ia in range(1, meta.nat + 1):
            label = atom_labels[ia - 1]
            dense_axis = np.zeros((3, nk_tot, nq_tot, meta.nwan, meta.nwan), dtype=np.complex128)
            for jw in range(1, meta.nwan + 1):
                for iw in range(1, meta.nwan + 1):
                    dr = f"ep_hop_r_{ia}_{jw}_{iw}"
                    di = f"ep_hop_i_{ia}_{jw}_{iw}"
                    if dr not in grp or di not in grp:
                        continue
                    arr_r = np.asarray(grp[dr], dtype=np.float64)
                    arr_i = np.asarray(grp[di], dtype=np.float64)
                    # HDF5 stores Fortran (3,nre,nrp) as (nrp,nre,3) in h5py.
                    arr = (arr_r + 1j * arr_i).transpose(2, 1, 0)  # (3,nre,nrp)

                    key = (ia, jw, iw)
                    cached = ws_cache.get(key)
                    if cached is None:
                        iw0 = iw - 1
                        jw0 = jw - 1
                        ia0 = ia - 1
                        ws_el = set_wigner_seitz_cell(el_images, meta.at, meta.wc[iw0], meta.wc[jw0])
                        ws_ph = set_wigner_seitz_cell(ph_images, meta.at, meta.wc[iw0], meta.tau[ia0])
                        ws_cache[key] = (ws_el, ws_ph)
                    else:
                        ws_el, ws_ph = cached
                    ws_el_vectors, ws_el_ndeg = ws_el
                    ws_ph_vectors, ws_ph_ndeg = ws_ph
                    nre = len(ws_el_vectors)
                    nrp = len(ws_ph_vectors)
                    if arr.shape[1] != nre or arr.shape[2] != nrp:
                        raise ValueError(
                            f"WS mismatch ia={ia},jw={jw},iw={iw}: dataset nre/nrp={arr.shape[1:]}, ws nre/nrp={(nre, nrp)}"
                        )

                    re_idx = _idx_from_ws_vectors(ws_el_vectors, meta.nk_grid)
                    rp_idx = _idx_from_ws_vectors(ws_ph_vectors, meta.nq_grid)

                    if divide_by_ndeg:
                        nde = np.asarray(ws_el_ndeg, dtype=np.float64)
                        ndp = np.asarray(ws_ph_ndeg, dtype=np.float64)
                        norm = (nde[:, None] * ndp[None, :])
                    else:
                        norm = 1.0

                    for ax in range(3):
                        block = arr[ax] / norm
                        _accumulate_pair_block(dense_axis[ax, :, :, iw - 1, jw - 1], re_idx, rp_idx, block)

            # Target shape used by LKAGSolver: (nk1,nk2,nk3,nq1,nq2,nq3,m,n)
            for ax_name, ax_id in (("x", 0), ("y", 1), ("z", 2)):
                g = dense_axis[ax_id].reshape(nk1, nk2, nk3, nq1, nq2, nq3, meta.nwan, meta.nwan)
                out[(label, ax_name)] = g
    return out


def build_greal_payload_from_epr(
    epr_up: str,
    epr_dn: str,
    *,
    atom_labels=None,
    units: str = "ha",
    divide_by_ndeg: bool = False,
):
    epr_up = os.path.abspath(epr_up)
    epr_dn = os.path.abspath(epr_dn)
    with h5py.File(epr_up, "r") as h5u, h5py.File(epr_dn, "r") as h5d:
        mu = _read_meta(h5u)
        md = _read_meta(h5d)
    if (mu.nat, mu.nwan, mu.nk_grid, mu.nq_grid) != (md.nat, md.nwan, md.nk_grid, md.nq_grid):
        raise ValueError("up/dn EPR metadata mismatch (nat/nwan/kmesh/qmesh)")

    labels = _normalize_atom_labels(atom_labels, mu.nat)
    up = _build_spin_greal(epr_up, labels, divide_by_ndeg=divide_by_ndeg)
    dn = _build_spin_greal(epr_dn, labels, divide_by_ndeg=divide_by_ndeg)

    g_real_dict = {}
    for (label, ax), g in up.items():
        g_real_dict[("up", label, ax)] = g
    for (label, ax), g in dn.items():
        g_real_dict[("down", label, ax)] = g

    payload = {
        "g_real_dict": g_real_dict,
        "nk_grid": list(mu.nk_grid),
        "nq_grid": list(mu.nq_grid),
        "basis": "wannier",
    }
    if units.lower() == "ev":
        payload["units"] = "eV"
    return payload


def write_greal_payload_from_epr(
    out_pkl: str,
    epr_up: str,
    epr_dn: str,
    *,
    atom_labels=None,
    units: str = "ha",
    divide_by_ndeg: bool = False,
):
    payload = build_greal_payload_from_epr(
        epr_up=epr_up,
        epr_dn=epr_dn,
        atom_labels=atom_labels,
        units=units,
        divide_by_ndeg=divide_by_ndeg,
    )
    outp = os.path.abspath(out_pkl)
    with open(outp, "wb") as f:
        pickle.dump(payload, f)
    return outp, payload


class EPRNativeProvider:
    """
    On-demand EPR-native g(Re, Rp) provider.
    Returns g_Rp with shape (nk1, nk2, nk3, m, n) for a requested (spin, atom, axis, Rp).
    """

    def __init__(
        self,
        epr_up: str,
        epr_dn: str,
        *,
        atom_labels=None,
        divide_by_ndeg: bool = False,
        value_unit: str = "ry",
    ):
        self.paths = {"up": os.path.abspath(epr_up), "down": os.path.abspath(epr_dn)}
        with h5py.File(self.paths["up"], "r") as h5u, h5py.File(self.paths["down"], "r") as h5d:
            mu = _read_meta(h5u)
            md = _read_meta(h5d)
        if (mu.nat, mu.nwan, mu.nk_grid, mu.nq_grid) != (md.nat, md.nwan, md.nk_grid, md.nq_grid):
            raise ValueError("up/dn EPR metadata mismatch (nat/nwan/kmesh/qmesh)")
        self.meta_up = mu
        self.meta_dn = md
        self.labels = _normalize_atom_labels(atom_labels, mu.nat)
        self.label_to_ia = {name: i + 1 for i, name in enumerate(self.labels)}
        self.divide_by_ndeg = bool(divide_by_ndeg)
        self.value_scale = _unit_scale_to_ev(value_unit)
        self.axis_to_idx = {"x": 0, "y": 1, "z": 2}
        self.el_images = init_rvec_images(mu.nk_grid, mu.at)
        self.ph_images = init_rvec_images(mu.nq_grid, mu.at)
        self._ws_cache = {}
        self._full_grid_cache = {}
        self._h5 = {}
        self._h5 = {}

    @property
    def nk_grid(self):
        return self.meta_up.nk_grid

    @property
    def nq_grid(self):
        return self.meta_up.nq_grid

    def close(self):
        for fh in self._h5.values():
            try:
                fh.close()
            except Exception:
                pass
        self._h5 = {}

    def _get_h5(self, spin: str):
        key = "up" if str(spin).strip().lower() == "up" else "down"
        if key not in self._h5:
            self._h5[key] = h5py.File(self.paths[key], "r")
        return self._h5[key]

    def _ws_pair(self, spin: str, ia: int, jw: int, iw: int):
        s = "up" if str(spin).strip().lower() == "up" else "down"
        key = (s, ia, jw, iw)
        got = self._ws_cache.get(key)
        if got is not None:
            return got
        iw0 = iw - 1
        jw0 = jw - 1
        ia0 = int(ia) - 1
        meta = self.meta_up if s == "up" else self.meta_dn
        ws_el_vectors, ws_el_ndeg = set_wigner_seitz_cell(self.el_images, meta.at, meta.wc[iw0], meta.wc[jw0])
        ws_ph_vectors, ws_ph_ndeg = set_wigner_seitz_cell(self.ph_images, meta.at, meta.wc[iw0], meta.tau[ia0])
        re_idx = _idx_from_ws_vectors(ws_el_vectors, meta.nk_grid)
        rp_lin = _idx_from_ws_vectors(ws_ph_vectors, meta.nq_grid)
        if self.divide_by_ndeg:
            norm = np.asarray(ws_el_ndeg, dtype=np.float64)[:, None] * np.asarray(ws_ph_ndeg, dtype=np.float64)[None, :]
        else:
            norm = None
        got = (re_idx, rp_lin, norm, len(ws_el_vectors), len(ws_ph_vectors))
        self._ws_cache[key] = got
        return got



    def get_gRp(self, spin: str, atom_label: str, axis: str, rp_idx):
        s = "up" if str(spin).strip().lower() == "up" else "down"
        ax = str(axis).strip().lower()
        if ax not in self.axis_to_idx:
            raise ValueError(f"Unsupported axis: {axis}")
        if atom_label not in self.label_to_ia:
            raise KeyError(f"Unknown atom label: {atom_label}. Known={sorted(self.label_to_ia)}")
        meta = self.meta_up if s == "up" else self.meta_dn
        nq = meta.nq_grid
        rp = tuple(int(rp_idx[i]) % int(nq[i]) for i in range(3))
        target_rp_lin = rp[0] * (nq[1] * nq[2]) + rp[1] * nq[2] + rp[2]

        cache_key = (s, atom_label, ax)
        cached_full = self._full_grid_cache.get(cache_key)

        if cached_full is None:
            ia = self.label_to_ia[atom_label]
            ax_id = self.axis_to_idx[ax]
            nk_tot = int(np.prod(meta.nk_grid))
            nq_tot = int(np.prod(meta.nq_grid))

            # Allocate (nq_tot, nk_tot, nwan, nwan) array for fast slicing over q
            dense_grid = np.zeros((nq_tot, nk_tot, meta.nwan, meta.nwan), dtype=np.complex128)
            h5 = self._get_h5(s)
            grp = h5["eph_matrix_wannier"]

            for jw in range(1, meta.nwan + 1):
                for iw in range(1, meta.nwan + 1):
                    dr = f"ep_hop_r_{ia}_{jw}_{iw}"
                    di = f"ep_hop_i_{ia}_{jw}_{iw}"
                    if dr not in grp or di not in grp:
                        continue

                    # Read from HDF5 and reshape
                    arr_r = np.asarray(grp[dr], dtype=np.float64)
                    arr_i = np.asarray(grp[di], dtype=np.float64)
                    arr = (arr_r + 1j * arr_i).transpose(2, 1, 0)  # (3, nre, nrp)

                    re_idx, rp_lin, norm, nre, nrp = self._ws_pair(s, ia, jw, iw)
                    if arr.shape[1] != nre or arr.shape[2] != nrp:
                        raise ValueError(f"WS mismatch ia={ia},jw={jw},iw={iw}: arr={arr.shape[1:]} ws={(nre, nrp)}")

                    blk = arr[ax_id]  # Shape (nre, nrp)
                    if norm is not None:
                        blk = blk / norm

                    # Vectorized mapping to dense (nq_tot, nk_tot) grid
                    flat_blk = blk.reshape(-1)
                    if self.value_scale != 1.0:
                        flat_blk = flat_blk * self.value_scale
                    # lin has shape (nre, nrp) flattened: re_idx + rp_lin * nk_tot
                    lin = (re_idx[:, None] + rp_lin[None, :] * nk_tot).reshape(-1)

                    target_view = dense_grid[:, :, iw - 1, jw - 1].reshape(-1)
                    np.add.at(target_view, lin, flat_blk)

            self._full_grid_cache[cache_key] = dense_grid
            cached_full = dense_grid

        nk1, nk2, nk3 = meta.nk_grid
        out_slice = cached_full[target_rp_lin].reshape(nk1, nk2, nk3, meta.nwan, meta.nwan)
        return out_slice
