"""Native analytic tensor exchange-derivative kernel for qe2pert EPR inputs.

This module implements the analytic spinor-based derivative method with
correct physics (full dg matrix, band-basis transformation) and multiprocessing.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

# Prevent OpenMP crash on fork
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["OMP_INIT_DEF_ALLOCATOR"] = "FALSE"

import h5py
import numba
import numpy as np
from numba import njit, prange

from slw.exchange.kernels.dj_epr import (
    _atom_labels,
    _bond_target_q_phase,
    _build_gkq_one,
)
from slw.exchange.kernels.dj_epr import (
    _kq_map as _commensurate_kq_map,
)

# Core utilities from SLW
from slw.exchange.kernels.epr import (
    _build_hk_from_epr,
    _find_nearest_neighbours_from_epr,
    _full_k_mesh,
    _load_slices,
)
from slw.exchange.kernels.j_epr import (
    _split_chunks,
)
from slw.exchange.kernels.j_tensor_epr import _apply_model_soc
from slw.exchange.kernels.j_wannier import (
    _infer_collinear_slices_from_win,
    _load_spinor_hr_hk,
    _slice_file_order_summary,
    _spinor_slice_summary,
)
from slw.exchange.kernels.lkag import (
    get_cfr_pole_mesh,
    get_semicircle_contour,
)
from slw.exchange.kernels.spinor import (
    WANNIER90_D_ORDER,
    WANNIER90_P_ORDER,
    spinor_from_collinear,
)

_AXES = ("x", "y", "z")

# Globals for worker inheritance
_G_KDATA = None
_G_KQMAP = None
_G_PAIR_META = None
_G_PHASE = None
_G_BOND_Q_PHASE = None
_G_WK = None
_G_AXES = None
_G_SLICES = None
_G_THREADPOOL_LIMITS = None
_G_DG_BAND = None
_G_G_TARGET_T = None
_G_TARGET_ID = None
_G_SPIN_DIRECTION = None
_G_PROGRESS_EVERY = 0
_G_ONSITE_DERIV_PROJECTOR = True
_G_VERBOSE_WORKER_INIT = 0


def _parse_axes(text):
    raw = str(text).replace(",", "").lower()
    out = []
    for ch in raw:
        if ch not in _AXES:
            raise ValueError(f"Unsupported tensor axis '{ch}'. Use subset of xyz.")
        if ch not in out:
            out.append(ch)
    return tuple(out) if out else _AXES


@njit
def _lkag_tensor_deriv_trace_select(Da_i, dDa_i, G_ij, dG_ij, Db_j, dDb_j, G_ji, dG_ji, use_dDa, use_dDb):
    """
    dJ tensor trace with optional dD terms.

    The two dG terms are always evaluated.  dD terms are evaluated only when
    the displaced atom is the corresponding magnetic site, avoiding zero-matrix
    multiplications for most bonds.
    """
    term2 = Da_i @ dG_ij @ Db_j @ G_ji
    term4 = Da_i @ G_ij @ Db_j @ dG_ji
    res = 0.0 + 0.0j
    for i in range(Da_i.shape[0]):
        res += term2[i, i] + term4[i, i]
    if use_dDa:
        term1 = dDa_i @ G_ij @ Db_j @ G_ji
        for i in range(Da_i.shape[0]):
            res += term1[i, i]
    if use_dDb:
        term3 = Da_i @ G_ij @ dDb_j @ G_ji
        for i in range(Da_i.shape[0]):
            res += term3[i, i]
    return res


def _normalize_mag_atoms(args):
    vals = [int(x) - 1 if int(args.mag_atoms_base) == 1 else int(x) for x in args.mag_atoms]
    if any(x < 0 for x in vals):
        raise ValueError(f"Invalid --mag_atoms after base conversion: {vals}")
    return vals


def _normalize_targets(args, default_targets):
    if args.targets is None:
        return [int(x) for x in default_targets]
    vals = [int(x) - 1 if int(args.mag_atoms_base) == 1 else int(x) for x in args.targets]
    if any(x < 0 for x in vals):
        raise ValueError(f"Invalid --targets after base conversion: {vals}")
    return vals


def _load_slices_for_global_atoms(args, dim, atom_ids):
    """Load orbital slices keyed by 0-based global atom id."""
    atom_ids = [int(x) for x in atom_ids]
    orig_mag_atoms = args.mag_atoms
    try:
        args.mag_atoms = [x + (1 if args.mag_atoms_base == 1 else 0) for x in atom_ids]
        slices = _load_slices(args, dim)
    finally:
        args.mag_atoms = orig_mag_atoms

    slice_keys = {int(k) for k in slices}
    has_global_keys = all(atom_id in slice_keys for atom_id in atom_ids)
    has_local_keys = all(iloc in slice_keys for iloc in range(len(atom_ids)))
    use_local_order = bool(args.slices) and not has_global_keys and has_local_keys

    if use_local_order:
        missing_local = [iloc for iloc in range(len(atom_ids)) if iloc not in slices]
        if missing_local:
            raise ValueError(
                "Slice loader did not return expected local keys "
                f"{missing_local}; available keys={sorted(int(k) for k in slices)}"
            )
        slices = {int(atom_id): slices[iloc] for iloc, atom_id in enumerate(atom_ids)}

    missing_atoms = [atom_id for atom_id in atom_ids if atom_id not in slices]
    if missing_atoms:
        raise ValueError(
            "Missing orbital slices for global atom ids "
            f"{missing_atoms}; available keys={sorted(int(k) for k in slices)}. "
            "Pass --slices keyed by the same "
            "0-based/1-based atom convention as --mag_atoms_base after conversion."
        )
    return slices


def _load_spinor_slices_for_global_atoms(args, dim, atom_ids):
    atom_ids = [int(x) for x in atom_ids]
    if args.slices or not args.mag_subspace:
        return _load_slices_for_global_atoms(args, dim, atom_ids), []
    if not args.win:
        raise ValueError("--spinor_hr with --mag_subspace requires --win")
    local_slices, slice_labels = _infer_collinear_slices_from_win(args.win, dim, args.mag_subspace)
    missing_local = [iloc for iloc in range(len(atom_ids)) if iloc not in local_slices]
    if missing_local:
        raise ValueError(
            f"--mag_subspace={args.mag_subspace!r} produced local slices {sorted(local_slices)}, "
            f"but mag_atoms needs local sites {list(range(len(atom_ids)))}"
        )
    slices = {int(atom_id): local_slices[iloc] for iloc, atom_id in enumerate(atom_ids)}
    return slices, slice_labels


def _parse_soc_p_groups(text, *, base=0):
    if not text:
        return []
    groups = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        vals = [int(x.strip()) - int(base) for x in item.replace(",", " ").split()]
        if len(vals) != 3:
            raise ValueError(f"Each --soc_p_groups entry must contain 3 orbital indices, got {item!r}")
        groups.append(vals)
    return groups


def _spinor_atom_indices(slc, nwan):
    up = np.arange(slc.start, slc.stop, dtype=np.int64)
    return np.concatenate([up, up + int(nwan)])


def _embed_collinear_g(g_up, g_dn, spin_direction):
    """Embed collinear EPR perturbations in the same spin frame as H_spin."""
    return spinor_from_collinear(g_up, g_dn, n=spin_direction)


def _compose_spinor_operator(axis, bmat):
    n = int(bmat.shape[0])
    z = np.zeros((n, n), dtype=np.complex128)
    if axis == "x":
        return np.block([[z, bmat], [bmat, z]])
    if axis == "y":
        return np.block([[z, -1j * bmat], [1j * bmat, z]])
    if axis == "z":
        return np.block([[bmat, z], [z, -bmat]])
    raise ValueError(f"Unsupported axis: {axis}")


def _extract_exchange_fields_from_spinor_block(h_loc):
    n2 = int(h_loc.shape[0])
    n = n2 // 2
    huu, hud, hdu, hdd = h_loc[:n, :n], h_loc[:n, n:], h_loc[n:, :n], h_loc[n:, n:]
    return 0.5*(hud+hdu), (hdu-hud)/(2.0j), 0.5*(huu-hdd)


def _pauli_block_all(block):
    nrow2, ncol2 = block.shape
    nrow = nrow2 // 2
    ncol = ncol2 // 2
    guu = block[:nrow, :ncol]
    gud = block[:nrow, ncol:]
    gdu = block[nrow:, :ncol]
    gdd = block[nrow:, ncol:]
    return (
        0.5 * (guu + gdd),
        0.5 * (gud + gdu),
        0.5j * (gud - gdu),
        0.5 * (guu - gdd),
    )


def _build_local_tb2j_projector_from_block(h_loc):
    """
    Spinor LKAG/TB2J local exchange projector.

    The spinor Hamiltonian is decomposed as H = H0 sigma_0 + M . sigma.
    This returns P_i = M_i . e_i, where e_i is the static local exchange-field
    direction. In the collinear z limit, P_i = (H_up - H_dn) / 2.
    """
    _, mx, my, mz = _pauli_block_all(h_loc)
    evec = np.array([np.trace(mx), np.trace(my), np.trace(mz)], dtype=np.complex128)
    norm = np.linalg.norm(evec)
    if norm < 1e-14:
        return np.zeros_like(mx)
    evec = evec / norm
    return mx * evec[0] + my * evec[1] + mz * evec[2]


def _local_tb2j_projector_direction(h_loc):
    _, mx, my, mz = _pauli_block_all(h_loc)
    evec = np.array([np.trace(mx), np.trace(my), np.trace(mz)], dtype=np.complex128)
    norm = np.linalg.norm(evec)
    if norm < 1e-14:
        return np.zeros(3, dtype=np.complex128)
    return evec / norm


def _build_local_tb2j_projector_deriv_from_block(dh_loc, evec):
    """
    Onsite derivative of the spinor exchange projector.

    For spinor matrices there is no generally valid H_up/H_dn block split.
    We therefore differentiate the exchange field, dP_i = dM_i . e_i.
    In the collinear z limit, dP_i = (g_up - g_dn) / 2, i.e. dDelta/2.
    """
    _, dmx, dmy, dmz = _pauli_block_all(dh_loc)
    return dmx * evec[0] + dmy * evec[1] + dmz * evec[2]


def _accumulate_dA_from_blocks(gij, gji, dgij, dgji, p_i, p_j, dp_i=None, dp_j=None):
    gij_u = _pauli_block_all(gij)
    gji_u = _pauli_block_all(gji)
    dgij_u = _pauli_block_all(dgij)
    dgji_u = _pauli_block_all(dgji)
    out = np.zeros((4, 4), dtype=np.complex128)
    for u in range(4):
        x = p_i @ gij_u[u]
        dx = p_i @ dgij_u[u]
        dpx = None if dp_i is None else dp_i @ gij_u[u]
        for v in range(4):
            y = p_j @ gji_u[v]
            dy = p_j @ dgji_u[v]
            val = np.einsum("ij,ji->", dx, y, optimize=True) + np.einsum("ij,ji->", x, dy, optimize=True)
            if dpx is not None:
                val += np.einsum("ij,ji->", dpx, y, optimize=True)
            if dp_j is not None:
                dpy = dp_j @ gji_u[v]
                val += np.einsum("ij,ji->", x, dpy, optimize=True)
            out[u, v] = val
    return out


def _periodic_rp_shift_indices(rp_grid, qmesh, shift):
    """Return indices for ``Rp -> Rp + shift`` on a full periodic q mesh."""
    rp = np.asarray(rp_grid, dtype=np.int64)
    qm = np.asarray(qmesh, dtype=np.int64)
    delta = np.asarray(shift, dtype=np.int64)
    if rp.ndim != 2 or rp.shape[1] != 3:
        raise ValueError(f"rp_grid must have shape (nRp,3), got {rp.shape}")
    if qm.shape != (3,) or np.any(qm <= 0):
        raise ValueError(f"qmesh must contain three positive integers, got {tuple(qm)}")
    if delta.shape != (3,):
        raise ValueError(f"shift must contain three integers, got shape={delta.shape}")
    n_expected = int(np.prod(qm))
    if len(rp) != n_expected:
        raise ValueError(f"rp_grid length={len(rp)} does not match prod(qmesh)={n_expected}")

    rp_mod = np.mod(rp, qm[None, :])
    flat = np.ravel_multi_index(tuple(rp_mod.T), tuple(int(x) for x in qm))
    if len(np.unique(flat)) != n_expected:
        raise ValueError("rp_grid is not a one-to-one representation of the periodic q mesh")
    index_from_flat = np.empty(n_expected, dtype=np.int64)
    index_from_flat[flat] = np.arange(n_expected, dtype=np.int64)

    shifted_mod = np.mod(rp + delta[None, :], qm[None, :])
    shifted_flat = np.ravel_multi_index(tuple(shifted_mod.T), tuple(int(x) for x in qm))
    return index_from_flat[shifted_flat]


def _decompose_tb2j_dA(dA_r, pair_meta, *, rp_grid=None, qmesh=None):
    """Decompose dA using the Rp-aware TB2J directed-bond mate.

    EPR uses the Wannier convention ``H_ij(R)=<i,0|H|j,R>`` and this solver
    follows it for ``G_ij(R)``.  Reversing the same physical bond and
    translating its first endpoint back to the origin therefore maps

        ``(i,j,R; Rp) -> (j,i,-R; Rp-R)``.

    The static tensor needs only the directed-bond mate.  Its displacement
    derivative additionally needs this periodic ``Rp-R`` reindexing.
    """
    pair_to_idx = {
        (int(m["gi"]), int(m["gj"]), tuple(int(x) for x in m["R"])): ip
        for ip, m in enumerate(pair_meta)
    }
    n_rp, n_pair, n_disp = dA_r.shape[:3]
    if rp_grid is None or qmesh is None:
        if n_rp != 1:
            raise ValueError("rp_grid and qmesh are required when decomposing more than one Rp")
        rp_grid = np.zeros((1, 3), dtype=np.int64)
        qmesh = (1, 1, 1)
    rp_shift_cache = {}
    d_jani = np.zeros((n_rp, n_pair, n_disp, 3, 3), dtype=np.float64)
    d_jiso = np.zeros((n_rp, n_pair, n_disp), dtype=np.float64)
    d_jiso_tensor = np.zeros((n_rp, n_pair, n_disp, 3, 3), dtype=np.float64)
    d_gamma = np.zeros((n_rp, n_pair, n_disp, 3, 3), dtype=np.float64)
    d_dmi_tensor = np.zeros((n_rp, n_pair, n_disp, 3, 3), dtype=np.float64)
    d_full = np.zeros((n_rp, n_pair, n_disp, 3, 3), dtype=np.float64)
    d_dmi = np.zeros((n_rp, n_pair, n_disp, 3), dtype=np.float64)
    for ib, m in enumerate(pair_meta):
        r_vec = tuple(int(x) for x in m["R"])
        mate = pair_to_idx.get((int(m["gj"]), int(m["gi"]), tuple(-x for x in r_vec)))
        val = dA_r[:, ib]
        if mate is None:
            val_m = np.zeros_like(val)
        else:
            mate_rp = rp_shift_cache.get(r_vec)
            if mate_rp is None:
                mate_rp = _periodic_rp_shift_indices(rp_grid, qmesh, tuple(-x for x in r_vec))
                rp_shift_cache[r_vec] = mate_rp
            val_m = dA_r[mate_rp, mate]
        for ia in range(3):
            for jb in range(3):
                d_jani[:, ib, :, ia, jb] = np.imag(val[:, :, ia + 1, jb + 1] + val_m[:, :, ia + 1, jb + 1])
        ms = 0.5 * (d_jani[:, ib] + np.swapaxes(d_jani[:, ib], -1, -2))
        tr = np.trace(ms, axis1=-2, axis2=-1) / 3.0
        d_jiso[:, ib] = np.imag(val[:, :, 0, 0] - val[:, :, 1, 1] - val[:, :, 2, 2] - val[:, :, 3, 3])
        for a in range(3):
            d_jiso_tensor[:, ib, :, a, a] = d_jiso[:, ib]
            d_gamma[:, ib, :, a, a] = ms[:, :, a, a] - tr
            for b in range(3):
                if a != b:
                    d_gamma[:, ib, :, a, b] = ms[:, :, a, b]
        d_dmi[:, ib, :, 0] = np.real(val[:, :, 0, 1] - val[:, :, 1, 0])
        d_dmi[:, ib, :, 1] = np.real(val[:, :, 0, 2] - val[:, :, 2, 0])
        d_dmi[:, ib, :, 2] = np.real(val[:, :, 0, 3] - val[:, :, 3, 0])
        d_dmi_tensor[:, ib, :, 0, 1] = d_dmi[:, ib, :, 2]
        d_dmi_tensor[:, ib, :, 1, 0] = -d_dmi[:, ib, :, 2]
        d_dmi_tensor[:, ib, :, 0, 2] = -d_dmi[:, ib, :, 1]
        d_dmi_tensor[:, ib, :, 2, 0] = d_dmi[:, ib, :, 1]
        d_dmi_tensor[:, ib, :, 1, 2] = d_dmi[:, ib, :, 0]
        d_dmi_tensor[:, ib, :, 2, 1] = -d_dmi[:, ib, :, 0]
        d_full[:, ib] = d_jiso_tensor[:, ib] + d_gamma[:, ib] + d_dmi_tensor[:, ib]
    return {
        "dJ_tensor_r": 1000.0 * d_full,
        "dJ_iso_r": 1000.0 * d_jiso,
        "dJ_iso_tensor_r": 1000.0 * d_jiso_tensor,
        "dJ_gamma_r": 1000.0 * d_gamma,
        "dJ_dmi_tensor_r": 1000.0 * d_dmi_tensor,
        "dDMI_r": 1000.0 * d_dmi,
        "dA_jani_r": 1000.0 * d_jani,
    }


@njit(parallel=True)
def _batch_eigh_njit(h_spin, efermi):
    nk, dim, _ = h_spin.shape
    evals = np.zeros((nk, dim), dtype=np.float64)
    evecs = np.zeros((nk, dim, dim), dtype=np.complex128)
    eye = np.eye(dim, dtype=np.complex128)
    for ik in prange(nk):
        hk = 0.5 * (h_spin[ik] + h_spin[ik].conj().T) - efermi * eye
        ww, cc = np.linalg.eigh(hk)
        evals[ik] = ww
        evecs[ik] = cc
    return evals, evecs


def _compute_band_dg(g_spin_full, evecs, kq_map_idx):
    """Transform dg from orbital basis to band basis: C(k+q)^dagger * dg * C(k)"""
    nq, nk, dim, _ = g_spin_full.shape
    dg_band = np.zeros((nq, nk, dim, dim), dtype=np.complex128)
    for iq in range(nq):
        kq_idx = kq_map_idx[iq]
        C_kq_dag = evecs[kq_idx].conj().transpose(0, 2, 1)
        C_k = evecs
        # Efficient batched matrix multiplication
        dg_band[iq] = np.matmul(C_kq_dag, np.matmul(g_spin_full[iq], C_k))
    return dg_band


def _worker_init(static_payload=None):
    global _G_THREADPOOL_LIMITS
    worker_threads = 1
    blas_threads = 1
    if static_payload:
        global _G_VERBOSE_WORKER_INIT, _G_ONSITE_DERIV_PROJECTOR
        global _G_KDATA, _G_KQMAP, _G_PAIR_META, _G_PHASE, _G_BOND_Q_PHASE
        global _G_WK, _G_AXES, _G_SLICES
        global _G_DG_BAND, _G_G_TARGET_T, _G_TARGET_ID, _G_SPIN_DIRECTION, _G_PROGRESS_EVERY
        _G_KDATA = static_payload["kdata"]
        _G_KQMAP = static_payload["kq_map"]
        _G_PAIR_META = static_payload["pair_meta"]
        _G_PHASE = static_payload["phase"]
        _G_BOND_Q_PHASE = static_payload["bond_target_q_phase"]
        _G_WK = static_payload["wk"]
        _G_AXES = static_payload["axes"]
        _G_SLICES = static_payload["slices"]
        _G_DG_BAND = static_payload.get("dg_band")
        _G_G_TARGET_T = static_payload.get("g_target_t")
        _G_TARGET_ID = static_payload.get("target_id")
        _G_SPIN_DIRECTION = static_payload.get("spin_direction", (0.0, 0.0, 1.0))
        _G_PROGRESS_EVERY = int(static_payload.get("progress_every", 0))
        _G_ONSITE_DERIV_PROJECTOR = bool(static_payload.get("onsite_deriv_projector", True))
        _G_VERBOSE_WORKER_INIT = int(static_payload.get("verbose_worker_init", 0))
        worker_threads = int(static_payload.get("numba_threads", 1))
        blas_threads = int(static_payload.get("blas_threads", 1))
    os.environ["OMP_NUM_THREADS"] = str(max(1, blas_threads))
    os.environ["MKL_NUM_THREADS"] = str(max(1, blas_threads))
    os.environ["OPENBLAS_NUM_THREADS"] = str(max(1, blas_threads))
    try:
        from threadpoolctl import threadpool_limits
        _G_THREADPOOL_LIMITS = threadpool_limits(limits=max(1, blas_threads), user_api="blas")
    except Exception:  # noqa: BLE001 - third-party runtime thread control is optional
        _G_THREADPOOL_LIMITS = None
    try:
        numba.set_num_threads(max(1, worker_threads))
    except Exception as exc:  # noqa: BLE001 - keep worker usable without thread tuning
        if int(_G_VERBOSE_WORKER_INIT) != 0:
            print(
                f"[dJ-epr-tensor][worker-init][warn] "
                f"failed to set numba_threads={worker_threads}: {exc}",
                flush=True,
            )
    if int(_G_VERBOSE_WORKER_INIT) != 0:
        print(
            f"[dJ-epr-tensor][worker-init] pid={os.getpid()} "
            f"blas_threads={max(1, blas_threads)} numba_threads={max(1, worker_threads)}",
            flush=True,
        )


def _energy_worker(energy_chunk):
    if _G_DG_BAND is None or _G_TARGET_ID is None:
        raise RuntimeError("Energy worker missing target/axis static payload.")
    return _compute_tensor_chunk_analytic(
        energy_chunk,
        _G_KDATA,
        _G_DG_BAND,
        _G_G_TARGET_T,
        int(_G_TARGET_ID),
        _G_KQMAP,
        _G_PAIR_META,
        _G_PHASE,
        _G_BOND_Q_PHASE,
        _G_WK,
        _G_AXES,
        progress_every=_G_PROGRESS_EVERY,
        progress_label=f"pid={os.getpid()} target={_G_TARGET_ID}",
        onsite_deriv_projector=bool(_G_ONSITE_DERIV_PROJECTOR),
    )


def _task_worker(task):
    """
    Processes one (target_atom, displacement_axis) task.
    """
    ia, dax, energy_mesh, epr_up, epr_dn, eph_unit = task
    t_task = time.time()
    print(
        f"[dJ-epr-tensor][worker] start pid={os.getpid()} target={ia} axis={dax} "
        f"nE={len(energy_mesh)} nq={len(_G_KDATA['qpts'])} nk={len(_G_KDATA['kpts'])} "
        f"nPair={len(_G_PAIR_META)}",
        flush=True,
    )

    # 1. Build FULL g_spin
    t_g = time.time()
    g_up = _build_gkq_one(epr_up, ia, dax, _G_KDATA["kpts"], _G_KDATA["qpts"], unit=eph_unit)
    g_dn = _build_gkq_one(epr_dn, ia, dax, _G_KDATA["kpts"], _G_KDATA["qpts"], unit=eph_unit)
    print(f"[dJ-epr-tensor][worker] target={ia} axis={dax} build_g={time.time()-t_g:.2f}s", flush=True)

    t_spin = time.time()
    nw = g_up.shape[-1]
    g_spin_full = _embed_collinear_g(g_up, g_dn, _G_SPIN_DIRECTION)
    print(f"[dJ-epr-tensor][worker] target={ia} axis={dax} spin_embed={time.time()-t_spin:.2f}s", flush=True)

    # 2. Extract local target block only for magnetic targets.
    # Non-magnetic displaced atoms still contribute through dg_band; they just
    # do not have an onsite exchange-field derivative dP/du.
    if ia in _G_SLICES:
        slc_t = _G_SLICES[ia]
        idx_t = _spinor_atom_indices(slc_t, nw)
        g_target_t = g_spin_full[:, :, idx_t[:, None], idx_t]
    else:
        g_target_t = None

    # 3. Transform g_spin_full to band basis (nk, dim, dim)
    t_band = time.time()
    dg_band = _compute_band_dg(g_spin_full, _G_KDATA["evecs"], _G_KQMAP)
    print(f"[dJ-epr-tensor][worker] target={ia} axis={dax} band_transform={time.time()-t_band:.2f}s", flush=True)

    # 4. Energy integration
    t_int = time.time()
    acc = _compute_tensor_chunk_analytic(
        energy_mesh, _G_KDATA, dg_band, g_target_t, ia, _G_KQMAP,
        _G_PAIR_META, _G_PHASE, _G_BOND_Q_PHASE, _G_WK, _G_AXES
    )
    print(
        f"[dJ-epr-tensor][worker] target={ia} axis={dax} integrate={time.time()-t_int:.2f}s "
        f"total={time.time()-t_task:.2f}s",
        flush=True,
    )
    return (ia, dax, acc)


def _run_target_axis_energy_parallel(task, common_payload, nproc):
    ia, dax, energy_mesh, epr_up, epr_dn, eph_unit = task
    t_task = time.time()
    print(
        f"[dJ-epr-tensor][target-axis] start target={ia} axis={dax} "
        f"nE={len(energy_mesh)} nproc={int(nproc)} nq={len(common_payload['kdata']['qpts'])} "
        f"nk={len(common_payload['kdata']['kpts'])} nPair={len(common_payload['pair_meta'])}",
        flush=True,
    )

    t_g = time.time()
    g_up = _build_gkq_one(epr_up, ia, dax, common_payload["kdata"]["kpts"], common_payload["kdata"]["qpts"], unit=eph_unit)
    g_dn = _build_gkq_one(epr_dn, ia, dax, common_payload["kdata"]["kpts"], common_payload["kdata"]["qpts"], unit=eph_unit)
    print(f"[dJ-epr-tensor][target-axis] target={ia} axis={dax} build_g={time.time()-t_g:.2f}s", flush=True)

    t_spin = time.time()
    g_spin_full = _embed_collinear_g(g_up, g_dn, common_payload["spin_direction"])
    print(f"[dJ-epr-tensor][target-axis] target={ia} axis={dax} spin_embed={time.time()-t_spin:.2f}s", flush=True)

    nw = g_up.shape[-1]
    if ia in common_payload["slices"]:
        slc_t = common_payload["slices"][ia]
        idx_t = _spinor_atom_indices(slc_t, nw)
        g_target_t = g_spin_full[:, :, idx_t[:, None], idx_t]
    else:
        g_target_t = None

    t_band = time.time()
    dg_band = _compute_band_dg(g_spin_full, common_payload["kdata"]["evecs"], common_payload["kq_map"])
    print(f"[dJ-epr-tensor][target-axis] target={ia} axis={dax} band_transform={time.time()-t_band:.2f}s", flush=True)

    t_int = time.time()
    nproc_eff = max(1, min(int(nproc), len(energy_mesh)))
    if nproc_eff <= 1:
        acc = _compute_tensor_chunk_analytic(
            energy_mesh,
            common_payload["kdata"],
            dg_band,
            g_target_t,
            ia,
            common_payload["kq_map"],
            common_payload["pair_meta"],
            common_payload["phase"],
            common_payload["bond_target_q_phase"],
            common_payload["wk"],
            common_payload["axes"],
            progress_every=int(common_payload.get("progress_every", 0)),
            progress_label=f"pid={os.getpid()} target={ia} axis={dax}",
            onsite_deriv_projector=bool(common_payload.get("onsite_deriv_projector", True)),
        )
    else:
        chunks = _split_chunks(list(energy_mesh), nproc_eff)
        static_payload = dict(common_payload)
        static_payload.update({"dg_band": dg_band, "g_target_t": g_target_t, "target_id": ia})
        # Fork keeps dg_band/g_target_t copy-on-write instead of serializing them
        # into every worker.  The code sets conservative OpenMP fork guards above.
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        with ctx.Pool(processes=nproc_eff, initializer=_worker_init, initargs=(static_payload,)) as pool:
            partials = pool.map(_energy_worker, chunks)
        acc = np.zeros_like(partials[0])
        for part in partials:
            acc += part
    print(
        f"[dJ-epr-tensor][target-axis] target={ia} axis={dax} integrate={time.time()-t_int:.2f}s "
        f"total={time.time()-t_task:.2f}s",
        flush=True,
    )
    return ia, dax, acc


def _compute_tensor_chunk_analytic(
    energy_chunk,
    kdata,
    dg_band,
    g_target_t,
    target_id,
    kq_map,
    pair_meta,
    phase,
    bond_target_q_phase,
    wk,
    axes,
    *,
    progress_every=0,
    progress_label="",
    onsite_deriv_projector=True,
):
    n_pair = len(pair_meta)
    nq, _nk, _dim, _ = dg_band.shape
    bond_target_q_phase = np.asarray(bond_target_q_phase, dtype=np.complex128)
    if bond_target_q_phase.shape != (nq, n_pair):
        raise ValueError(
            f"bond_target_q_phase must have shape {(nq, n_pair)}, "
            f"got {bond_target_q_phase.shape}"
        )
    acc = np.zeros((nq, n_pair, 4, 4), dtype=np.complex128)

    evals = kdata["evals"]
    dp_target_q = None
    if bool(onsite_deriv_projector) and g_target_t is not None and target_id in kdata.get("p_dirs", {}):
        evec_t = kdata["p_dirs"][int(target_id)]
        # q-dependent onsite derivative of the local exchange projector.
        # EPR g(k+q,k) is local in the target block here; use the k average
        # as the onsite part entering dP for that phonon q.
        g_loc_q = np.mean(np.asarray(g_target_t, dtype=np.complex128), axis=1)
        dp_target_q = np.asarray([
            _build_local_tb2j_projector_deriv_from_block(g_loc_q[iq], evec_t)
            for iq in range(g_loc_q.shape[0])
        ], dtype=np.complex128)

    # Pre-transpose and cache static matrices
    cj_conj_T = {}
    for li in set([m["li"] for m in pair_meta] + [m["lj"] for m in pair_meta]):
        cj_conj_T[li] = kdata["site_coeffs"][li].conj().transpose(0, 2, 1)

    n_energy = len(energy_chunk)
    progress_every = int(progress_every)
    t_chunk0 = time.time()
    t_mark = t_chunk0
    for ie, (z, dz) in enumerate(energy_chunk, start=1):
        t_e0 = time.time()
        inv = 1.0 / (z - evals) # (nk, dim)

        # Precompute scaled matrices
        c_inv = {}
        c_inv_cT = {}
        for li, c_dag in cj_conj_T.items():
            ci = kdata["site_coeffs"][li]
            c_inv[li] = ci * inv[:, None, :] # (nk, 2M, dim)
            c_inv_cT[li] = c_dag * inv[:, :, None] # (nk, dim, 2M)

        gij_r_dict = {}
        gji_r_dict = {}
        for ip, meta in enumerate(pair_meta):
            li, lj = int(meta["li"]), int(meta["lj"])

            gij_k = c_inv[li] @ cj_conj_T[lj]
            gji_k = c_inv[lj] @ cj_conj_T[li]

            wk_ph = wk * phase[ip]
            gij_r_dict[ip] = np.tensordot(wk_ph, gij_k, axes=(0, 0))
            gji_r_dict[ip] = np.tensordot(wk_ph.conj(), gji_k, axes=(0, 0))

        t_green = time.time()
        for iq in range(nq):
            kq_idx = kq_map[iq]
            dg_b = dg_band[iq] # (nk, dim, dim)

            for ip, meta in enumerate(pair_meta):
                li, lj = int(meta["li"]), int(meta["lj"])

                ci_kq_inv = c_inv[li][kq_idx]
                dgij_k = (ci_kq_inv @ dg_b) @ c_inv_cT[lj]

                cj_kq_inv = c_inv[lj][kq_idx]
                dgji_k = (cj_kq_inv @ dg_b) @ c_inv_cT[li]

                wk_ph = wk * phase[ip]
                dgij_r = np.tensordot(wk_ph, dgij_k, axes=(0, 0))
                endpoint_phase = bond_target_q_phase[iq, ip]
                dgji_r = endpoint_phase * np.tensordot(wk_ph.conj(), dgji_k, axes=(0, 0))
                dp_i = dp_target_q[iq] if dp_target_q is not None and int(target_id) == li else None
                dp_j = (
                    endpoint_phase * dp_target_q[iq]
                    if dp_target_q is not None and int(target_id) == lj
                    else None
                )
                acc[iq, ip] += _accumulate_dA_from_blocks(
                    gij_r_dict[ip],
                    gji_r_dict[ip],
                    dgij_r,
                    dgji_r,
                    kdata["p_ops"][li],
                    kdata["p_ops"][lj],
                    dp_i=dp_i,
                    dp_j=dp_j,
                ) * (dz / np.pi)
        if progress_every > 0 and (ie == 1 or ie == n_energy or ie % progress_every == 0):
            now = time.time()
            print(
                f"[dJ-epr-tensor][energy-progress] {progress_label} "
                f"iE={ie}/{n_energy} "
                f"green={t_green-t_e0:.2f}s qpair={now-t_green:.2f}s "
                f"step={now-t_e0:.2f}s since_last={now-t_mark:.2f}s "
                f"elapsed={now-t_chunk0:.2f}s",
                flush=True,
            )
            t_mark = now
    return acc


def _precompute_spinor_kdata(h_spin, slices, efermi):
    t_diag = time.time()
    nk, dim, _ = h_spin.shape
    nwan = dim // 2
    print(f"[dJ-epr-tensor] Diagonalizing {nk} k-points in parallel...", flush=True)
    evals, evecs = _batch_eigh_njit(h_spin, float(efermi))
    print(f"[dJ-epr-tensor] Diagonalization complete in {time.time()-t_diag:.2f}s", flush=True)
    site_coeffs = {}
    for site, slc in slices.items():
        idx = _spinor_atom_indices(slc, nwan)
        site_coeffs[int(site)] = evecs[:, idx, :]
    h_mean = np.mean(h_spin, axis=0)
    d_ops = {}
    p_ops = {}
    p_dirs = {}
    for site, slc in slices.items():
        idx = _spinor_atom_indices(slc, nwan)
        hloc = h_mean[np.ix_(idx, idx)]
        bx, by, bz = _extract_exchange_fields_from_spinor_block(hloc)
        d_ops[int(site)] = {
            "x": _compose_spinor_operator("x", bx),
            "y": _compose_spinor_operator("y", by),
            "z": _compose_spinor_operator("z", bz),
        }
        p_ops[int(site)] = _build_local_tb2j_projector_from_block(hloc)
        p_dirs[int(site)] = _local_tb2j_projector_direction(hloc)
    return {"evals": evals, "evecs": evecs, "d_ops": d_ops, "p_ops": p_ops, "p_dirs": p_dirs, "site_coeffs": site_coeffs}


def _kq_map(kmesh, qmesh):
    """Map every ``(q,k)`` pair to the exact commensurate ``k+q`` index.

    A q-grid step is ``1 / qmesh`` in fractional reciprocal coordinates,
    whereas a k-grid step is ``1 / kmesh``.  The corresponding integer k-grid
    shift is therefore ``q_index * (kmesh // qmesh)``, not ``q_index``.
    Reuse the validated vectorized implementation shared with the scalar EPR
    derivative path so incompatible meshes fail before the expensive energy
    integration starts.
    """
    return _commensurate_kq_map(kmesh, qmesh)


def run_analytic(args):
    t0 = time.time()
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=max(1, int(args.blas_threads)), user_api="blas")
        print(f"[dJ-epr-tensor] blas_threads={int(args.blas_threads)}", flush=True)
    except Exception as exc:  # noqa: BLE001 - optional third-party thread control
        print(f"[dJ-epr-tensor][warn] failed to set blas_threads={args.blas_threads}: {exc}", flush=True)
    try:
        numba.set_num_threads(max(1, int(args.numba_threads)))
        print(f"[dJ-epr-tensor] numba_threads={numba.get_num_threads()}", flush=True)
    except Exception as exc:  # noqa: BLE001 - optional Numba runtime tuning
        print(f"[dJ-epr-tensor][warn] failed to set numba_threads={args.numba_threads}: {exc}", flush=True)
    axes = _parse_axes(args.tensor_axes)
    kpts = _full_k_mesh(args.kmesh)

    with h5py.File(args.epr_up, "r") as h5:
        epr_qmesh = tuple(int(x) for x in h5["basic_data/qc_dim"][()])
    qmesh = tuple(int(x) for x in (args.qmesh if args.qmesh is not None else epr_qmesh))
    qpts = _full_k_mesh(qmesh)
    print(
        f"[dJ-epr-tensor] meshes: kmesh={tuple(args.kmesh)} nk={len(kpts)} "
        f"qmesh={qmesh} nq={len(qpts)} tensor_axes={axes}",
        flush=True,
    )

    spinor_input = bool(args.spinor_hr)
    mag_atoms = _normalize_mag_atoms(args)
    target_ids = _normalize_targets(args, [mag_atoms[0]])
    all_involved = sorted(set(mag_atoms + target_ids))
    print(
        f"[dJ-epr-tensor] atoms: mag_atoms_0based={mag_atoms} "
        f"targets_0based={target_ids} all_involved={all_involved}",
        flush=True,
    )

    if spinor_input:
        print(
            f"[dJ-epr-tensor] Loading base H(k) from spinor_hr={args.spinor_hr} "
            f"groupby={args.groupby} unit={args.spinor_hr_unit}",
            flush=True,
        )
        h_spin, spinor_meta = _load_spinor_hr_hk(
            args.spinor_hr,
            kpts,
            apply_degeneracy=bool(args.apply_degeneracy),
            hr_unit=args.spinor_hr_unit,
            groupby=args.groupby,
            win=args.win,
            centres=args.centres,
        )
        dim_col = int(spinor_meta["nwan"])
        if spinor_meta.get("basis_groups_file"):
            print(f"[dJ-epr-tensor] basis_groups_file_order={spinor_meta['basis_groups_file']}", flush=True)
        if spinor_meta.get("basis_groups_internal"):
            print(f"[dJ-epr-tensor] basis_groups_internal_spin_major={spinor_meta['basis_groups_internal']}", flush=True)
        h_spin, soc_entries, soc_win = _apply_model_soc(h_spin, args, dim_col)
        if soc_entries:
            print(
                f"[dJ-epr-tensor] added atomic SOC entries={len(soc_entries)} "
                f"win={soc_win}",
                flush=True,
            )
        slices, slice_labels = _load_spinor_slices_for_global_atoms(args, dim_col, mag_atoms)
        if slice_labels:
            print(f"[dJ-epr-tensor] mag_subspace={slice_labels}", flush=True)
        print(
            f"[dJ-epr-tensor] mag_subspace_file_order="
            f"{_slice_file_order_summary(slices, dim_col, args.groupby)}",
            flush=True,
        )
        print(
            f"[dJ-epr-tensor] mag_subspace_internal_spin_major={_spinor_slice_summary(slices, dim_col)}",
            flush=True,
        )
    else:
        print("[dJ-epr-tensor] Loading base H(k) up/dn from EPR", flush=True)
        hk_up = _build_hk_from_epr(args.epr_up, kpts, unit=args.hr_unit)
        hk_dn = _build_hk_from_epr(args.epr_dn, kpts, unit=args.hr_unit)
        dim_col = int(hk_up.shape[1])
        slices = _load_slices_for_global_atoms(args, dim_col, mag_atoms)
        print("[dJ-epr-tensor] Building spinor H(k) from collinear up/dn...", flush=True)
        h_spin = spinor_from_collinear(hk_up, hk_dn, n=args.spin_direction)
        h_spin, soc_entries, soc_win = _apply_model_soc(h_spin, args, dim_col)
        if soc_entries:
            print(
                f"[dJ-epr-tensor] applied model SOC entries={len(soc_entries)} "
                f"win={soc_win or '<manual>'}",
                flush=True,
            )

    spin_evals = np.linalg.eigvalsh(h_spin)
    args._soc_entries = soc_entries
    args._soc_win_path = soc_win
    print(
        f"[dJ-epr-tensor] base spinor H(k) dim={h_spin.shape[1]} "
        f"band=({float(spin_evals.min()):.6g},{float(spin_evals.max()):.6g}) eV",
        flush=True,
    )
    kdata = _precompute_spinor_kdata(h_spin, slices, args.efermi)
    kdata["kpts"] = kpts
    kdata["qpts"] = qpts

    kq_map_idx = _kq_map(args.kmesh, qmesh)
    labels = _atom_labels(args.epr_up, None)
    neighbours = _find_nearest_neighbours_from_epr(args.epr_up, mag_atoms, n_shells=args.n_shells, d_max=args.d_max)

    pair_meta = []
    for n in neighbours:
        gi, gj = int(n["i"]), int(n["j"])
        if gi in mag_atoms and gj in mag_atoms:
            pair_meta.append({
                "gi": gi, "gj": gj, "li": gi, "lj": gj,
                "R": n["R"], "dist": n["distance"], "shell": n.get("shell_idx", 0),
            })
    print(
        f"[dJ-epr-tensor] bonds: n_neighbours={len(neighbours)} n_pair_meta={len(pair_meta)} "
        f"n_shells={args.n_shells} d_max={args.d_max}",
        flush=True,
    )

    r_arr = np.asarray([m["R"] for m in pair_meta], dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * (r_arr @ kpts.T))
    bond_target_q_phase = _bond_target_q_phase(qpts, pair_meta)
    wk = 1.0 / float(len(kpts))

    energy_mesh = (get_semicircle_contour(emin=args.emin, emax=0.0, npoints=args.empoints)
                   if args.integrator == "contour" else get_cfr_pole_mesh(beta_eV_inv=400.0, npoles=args.empoints))

    disp_axes = list(args.disp_axes)
    tasks = []
    for ia in target_ids:
        for dax in disp_axes:
            tasks.append((ia, dax, energy_mesh, args.epr_up, args.epr_dn, args.eph_unit))
    print(
        f"[dJ-epr-tensor] tasks: n_targets={len(target_ids)} disp_axes={disp_axes} "
        f"n_tasks={len(tasks)} requested_nproc={int(args.nproc)}",
        flush=True,
    )
    print(
        f"[dJ-epr-tensor] parallel mode: outer loop = target x axis, "
        f"inner energy integration uses up to nproc={int(args.nproc)} processes per target/axis",
        flush=True,
    )
    print(
        f"[dJ-epr-tensor] thread budget per energy worker: "
        f"blas_threads={int(args.blas_threads)} numba_threads={int(args.numba_threads)}",
        flush=True,
    )
    print(
        "[dJ-epr-tensor] task list: "
        + ", ".join(f"(target={ia},axis={dax})" for ia, dax, *_ in tasks),
        flush=True,
    )
    common_payload = {
        "kdata": kdata, "kq_map": kq_map_idx, "pair_meta": pair_meta,
        "phase": phase, "bond_target_q_phase": bond_target_q_phase,
        "wk": wk, "axes": axes, "slices": slices,
        "spin_direction": tuple(float(x) for x in args.spin_direction),
        "numba_threads": int(args.numba_threads),
        "blas_threads": int(args.blas_threads),
        "progress_every": int(args.progress_every),
        "verbose_worker_init": int(args.verbose_worker_init),
        "onsite_deriv_projector": bool(args.onsite_deriv_projector),
    }

    task_results = []
    results = {}
    for task in tasks:
        ia, dax, acc = _run_target_axis_energy_parallel(task, common_payload, int(args.nproc))
        task_results.append((ia, dax, acc))
        nq = len(qpts)
        grid = acc.reshape(qmesh[0], qmesh[1], qmesh[2], len(pair_meta), 4, 4)
        real_grid = (np.fft.fftn(grid, axes=(0, 1, 2)) / float(nq))
        results[(ia, dax)] = real_grid.reshape(-1, len(pair_meta), 4, 4)
        if int(args.checkpoint) != 0:
            _save_results(args, results, pair_meta, labels, axes, disp_axes, qmesh, complete=False)
            print(
                f"[dJ-epr-tensor] checkpoint saved after target={ia} axis={dax}: "
                f"{os.path.join(args.out_dir, args.out_h5)}",
                flush=True,
            )

    _save_results(args, results, pair_meta, labels, axes, disp_axes, qmesh, complete=True)
    print(f"[dJ-epr-tensor] Done. Total time: {time.time()-t0:.2f}s")


def _save_results(args, results, pair_meta, labels, tensor_axes, disp_axes, qmesh, *, complete=True):
    os.makedirs(args.out_dir, exist_ok=True)
    out_h5 = os.path.join(args.out_dir, args.out_h5)
    from slw.exchange.kernels.dj_epr import _rp_grid
    rp_grid = _rp_grid(qmesh)
    n_rp, n_pair, n_disp = len(rp_grid), len(pair_meta), len(disp_axes)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_h5, "w") as h5:
        h5.attrs["complete"] = int(bool(complete))
        h5.attrs["n_completed_target_axis"] = len(results)
        h5.attrs["kernel"] = "tb2j_dA"
        h5.attrs["onsite_deriv_exchange_field"] = int(bool(args.onsite_deriv_projector))
        h5.attrs["onsite_deriv_convention"] = "spinor_exchange_field_P_equals_Delta_over_2"
        h5.attrs["atomic_gauge_bond_phase"] = "endpoint_j_at_R_times_exp(+i2pi_q_dot_R)"
        h5.attrs["directed_bond_mate"] = "(j,i,-R;Rp-R)_periodic"
        h5.attrs["base_hamiltonian"] = "spinor_hr" if getattr(args, "spinor_hr", None) else "epr_up_down"
        h5.attrs["input_groupby"] = str(getattr(args, "groupby", "") or "")
        h5.attrs["internal_groupby"] = "spin"
        basic = h5.create_group("basic_data")
        basic.create_dataset("tensor_axes", data=np.asarray(("x", "y", "z"), dtype=object), dtype=str_dt)
        basic.create_dataset("pauli_axes", data=np.asarray(("0", "x", "y", "z"), dtype=object), dtype=str_dt)
        basic.create_dataset("atom_labels", data=np.asarray(labels, dtype=object), dtype=str_dt)
        basic.create_dataset("epr_up", data=np.array(str(args.epr_up), dtype=object), dtype=str_dt)
        basic.create_dataset("epr_dn", data=np.array(str(args.epr_dn), dtype=object), dtype=str_dt)
        basic.create_dataset("spinor_hr", data=np.array(str(getattr(args, "spinor_hr", "") or ""), dtype=object), dtype=str_dt)
        basic.create_dataset("input_groupby", data=np.array(str(getattr(args, "groupby", "") or ""), dtype=object), dtype=str_dt)
        basic.create_dataset("internal_groupby", data=np.array("spin", dtype=object), dtype=str_dt)
        soc_entries = getattr(args, "_soc_entries", []) or []
        basic.create_dataset("additional_soc", data=np.asarray(bool(soc_entries)))
        basic.create_dataset("soc_mode", data=np.array("atomic" if soc_entries else "none", dtype=object), dtype=str_dt)
        basic.create_dataset(
            "soc_entries",
            data=np.asarray(
                [
                    f"{entry.get('selector', entry['element'] + '-' + entry['orbital'])}:"
                    f"{float(entry['lambda_ev']):.16g}"
                    for entry in soc_entries
                ],
                dtype=object,
            ),
            dtype=str_dt,
        )
        basic.create_dataset("win", data=np.array(str(getattr(args, "win", "") or ""), dtype=object), dtype=str_dt)
        basic.create_dataset("centres", data=np.array(str(getattr(args, "centres", "") or ""), dtype=object), dtype=str_dt)
        basic.create_dataset("command", data=np.array(" ".join(sys.argv), dtype=object), dtype=str_dt)
        basic.create_dataset("kmesh", data=np.asarray(args.kmesh, dtype=np.int64))
        basic.create_dataset("qmesh", data=np.asarray(qmesh, dtype=np.int64))
        basic.create_dataset("efermi_ev", data=np.array(float(args.efermi), dtype=np.float64))
        basic.create_dataset("spin_direction", data=np.asarray(args.spin_direction, dtype=np.float64))
        basic.create_dataset("empoints", data=np.array(int(args.empoints), dtype=np.int64))
        basic.create_dataset("nproc", data=np.array(int(args.nproc), dtype=np.int64))
        basic.create_dataset(
            "mpi_size",
            data=np.array(int(getattr(args, "_mpi_size", 1)), dtype=np.int64),
        )
        basic.create_dataset("numba_threads", data=np.array(int(args.numba_threads), dtype=np.int64))
        basic.create_dataset("blas_threads", data=np.array(int(args.blas_threads), dtype=np.int64))
        basic.create_dataset("integrator", data=np.array(str(args.integrator), dtype=object), dtype=str_dt)
        basic.create_dataset("unit", data=np.array("meV/A", dtype=object), dtype=str_dt)
        bonds = h5.create_group("bonds")
        bonds.create_dataset("mag_i_atom", data=np.asarray([m["gi"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("mag_j_atom", data=np.asarray([m["gj"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("R", data=np.asarray([m["R"] for m in pair_meta], dtype=np.int64))
        bonds.create_dataset("distance_ang", data=np.asarray([m["dist"] for m in pair_meta], dtype=np.float64))
        disp = h5.create_group("displacements")
        target_ids = sorted({k[0] for k in results})
        disp.create_dataset("target_atom", data=np.asarray(target_ids, dtype=np.int64))
        disp.create_dataset("axes", data=np.asarray(disp_axes, dtype=object), dtype=str_dt)
        disp.create_dataset("Rp", data=rp_grid)
        grp_a = h5.create_group("dA_r")
        grp_full = h5.create_group("dJ_tensor_r")
        grp_iso = h5.create_group("dJ_iso_r")
        grp_iso_tensor = h5.create_group("dJ_iso_tensor_r")
        grp_gamma = h5.create_group("dJ_gamma_r")
        grp_dmi_tensor = h5.create_group("dJ_dmi_tensor_r")
        grp_dmi = h5.create_group("dDMI_r")
        grp_jani = h5.create_group("dJani_from_A_r")
        derivative_groups = (
            grp_full,
            grp_iso,
            grp_iso_tensor,
            grp_gamma,
            grp_dmi_tensor,
            grp_dmi,
            grp_jani,
        )
        for group in derivative_groups:
            group.attrs["unit"] = "meV/A"
        for ia in target_ids:
            dA_all = np.zeros((n_rp, n_pair, n_disp, 4, 4), dtype=np.complex128)
            for idax, dax in enumerate(disp_axes):
                if (ia, dax) in results:
                    dA_all[:, :, idax] = np.asarray(results[(ia, dax)], dtype=np.complex128)
            dec = _decompose_tb2j_dA(dA_all, pair_meta, rp_grid=rp_grid, qmesh=qmesh)
            for ib in range(n_pair):
                key = f"m{ia+1}_b{ib+1}"
                grp_a.create_dataset(key, data=dA_all[:, ib])
                derivative_datasets = (
                    grp_full.create_dataset(key, data=dec["dJ_tensor_r"][:, ib]),
                    grp_iso.create_dataset(key, data=dec["dJ_iso_r"][:, ib]),
                    grp_iso_tensor.create_dataset(
                        key, data=dec["dJ_iso_tensor_r"][:, ib]
                    ),
                    grp_gamma.create_dataset(key, data=dec["dJ_gamma_r"][:, ib]),
                    grp_dmi_tensor.create_dataset(
                        key, data=dec["dJ_dmi_tensor_r"][:, ib]
                    ),
                    grp_dmi.create_dataset(key, data=dec["dDMI_r"][:, ib]),
                    grp_jani.create_dataset(key, data=dec["dA_jani_r"][:, ib]),
                )
                for dataset in derivative_datasets:
                    dataset.attrs["unit"] = "meV/A"


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Analytic dJ/du Tensor Calculator (Correct Structure)")
    ap.add_argument("--epr_up", required=True, help="EPR up HDF5; still required for g(k+q,k) and structure metadata")
    ap.add_argument("--epr_dn", required=True, help="EPR down HDF5; still required for g(k+q,k)")
    ap.add_argument("--spinor_hr", default=None, help="Optional full SOC/noncollinear Wannier90 spinor hr.dat used as base Hamiltonian")
    ap.add_argument(
        "--groupby",
        choices=["spin", "orbital"],
        default=None,
        help="Required for --spinor_hr: TB2J spin-major or orbital-interleaved layout.",
    )
    ap.add_argument("--spinor_hr_unit", choices=["ev", "ry", "ha"], default="ev", help="Unit of --spinor_hr matrix elements")
    ap.add_argument("--win", default=None, help="Wannier90 .win used to infer spinor projection order and magnetic slices")
    ap.add_argument("--centres", default=None, help="Optional Wannier90 centres.xyz for spinor ordering diagnostics")
    ap.add_argument("--mag_subspace", default="", help="Infer magnetic local slices from .win projections, e.g. 'Mn:d'")
    ap.add_argument("--apply_degeneracy", action=argparse.BooleanOptionalAction, default=True, help="Divide spinor_hr blocks by Wannier90 degeneracy before H(k)")
    ap.add_argument("--hr_unit", default="ry")
    ap.add_argument("--eph_unit", default="ry")
    ap.add_argument("--efermi", type=float, required=True)
    ap.add_argument("--kmesh", type=int, nargs=3, required=True)
    ap.add_argument("--qmesh", type=int, nargs=3, default=None)
    ap.add_argument("--n_shells", type=int, default=10)
    ap.add_argument("--d_max", type=float, default=20.0)
    ap.add_argument("--mag_atoms", type=int, nargs="+", required=True)
    ap.add_argument("--mag_atoms_base", type=int, choices=[0, 1], default=0)
    ap.add_argument("--targets", type=int, nargs="+", default=None)
    ap.add_argument("--disp_axes", default="xyz")
    ap.add_argument("--tensor_axes", default="xyz")
    ap.add_argument("--spin_direction", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    ap.add_argument("--soc", default="", help="Generic model SOC specs inferred from --win, e.g. 'Te:p:0.5;Mn:d:0.05'")
    ap.add_argument("--soc_element", default="", help="Element for compatibility --lambda_te p-SOC mode")
    ap.add_argument("--lambda_te", type=float, default=0.0, help="Compatibility model onsite p-SOC lambda in eV; prefer --soc")
    ap.add_argument("--soc_p_groups", default="", help="Semicolon-separated p orbital groups, e.g. '10,11,12;25,26,27'")
    ap.add_argument("--soc_groups_base", type=int, default=0, choices=[0, 1], help="Index base for --soc_p_groups")
    ap.add_argument("--soc_p_groups_base", dest="soc_groups_base", type=int, choices=[0, 1], help=argparse.SUPPRESS)
    ap.add_argument("--p_order", default=WANNIER90_P_ORDER)
    ap.add_argument("--d_order", default=WANNIER90_D_ORDER)
    ap.add_argument("--slices", required=True, help="Local orbital slices, e.g. '0:0:5,1:5:10'")
    ap.add_argument("--emin", type=float, default=-25.0)
    ap.add_argument("--empoints", type=int, default=300)
    ap.add_argument("--integrator", choices=["contour", "cfr"], default="contour")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--numba_threads", type=int, default=1, help="Numba threads per process")
    ap.add_argument("--blas_threads", type=int, default=1, help="BLAS threads per process for matmul/einsum")
    ap.add_argument("--progress_every", type=int, default=0, help="Print per-worker progress every N energy points; 0 disables")
    ap.add_argument("--verbose_worker_init", type=int, default=0, help="Print one worker-init line per local energy worker")
    ap.add_argument("--checkpoint", type=int, default=1, help="Write partial HDF5 after each completed target/axis; 0 disables")
    ap.add_argument("--onsite_deriv_exchange_field", "--onsite_deriv_projector",
                    dest="onsite_deriv_projector",
                    action=argparse.BooleanOptionalAction,
                    default=True,
                    help=("Include onsite derivative of the local spinor exchange field dP_i/du. "
                          "Convention: P=M.e=Delta/2 in the collinear limit, so dP=dDelta/2."))
    ap.add_argument("--out_dir", default="dJ_epr_tensor_analytic")
    ap.add_argument("--out_h5", default="dJ_tensor_analytic.h5")
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    run_analytic(args)


if __name__ == "__main__":
    main()
