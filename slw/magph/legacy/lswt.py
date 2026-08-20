"""Local-frame and BdG helpers for tensor LSWT magnon calculations."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np


def normalize_vectors(vec: np.ndarray, *, name: str = "vector") -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-30):
        raise ValueError(f"{name} contains a zero-length vector")
    return arr / norm


def parse_spin_pattern(text, nmag: int, *, default: str = "auto") -> np.ndarray:
    if text is None:
        text = default
    raw = str(text).strip().lower()
    if raw in {"auto", "afm"}:
        vals = np.ones(int(nmag), dtype=np.float64)
        vals[1::2] = -1.0
        return vals
    if raw in {"fm", "ferro", "ferromagnetic"}:
        return np.ones(int(nmag), dtype=np.float64)
    vals = np.asarray([float(x) for x in raw.replace(";", ",").split(",") if x.strip()], dtype=np.float64)
    if vals.size != int(nmag):
        raise ValueError(f"spin_pattern length={vals.size}, expected nmag={nmag}")
    if np.any(np.abs(vals) <= 1.0e-30):
        raise ValueError("spin_pattern entries must be nonzero")
    return np.where(vals >= 0.0, 1.0, -1.0)


def collinear_spin_directions(spin_direction, spin_pattern) -> np.ndarray:
    n = normalize_vectors(np.asarray(spin_direction, dtype=np.float64).reshape(3), name="spin_direction")
    signs = np.asarray(spin_pattern, dtype=np.float64).reshape(-1)
    return signs[:, None] * n[None, :]


def local_frames_from_spin_directions(spin_directions, reference_axis=None) -> np.ndarray:
    """Build local frames with columns `(e1, e2, e3)` for each magnetic site."""
    e3 = normalize_vectors(np.asarray(spin_directions, dtype=np.float64), name="spin_directions")
    nmag = e3.shape[0]
    if reference_axis is None:
        ref_x = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float64), (nmag, 1))
        ref_y = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), (nmag, 1))
        use_y = np.abs(np.einsum("ij,ij->i", ref_x, e3, optimize=True)) > 0.95
        ref = ref_x
        ref[use_y] = ref_y[use_y]
    else:
        ref = np.tile(
            normalize_vectors(np.asarray(reference_axis, dtype=np.float64).reshape(3), name="reference_axis"),
            (nmag, 1),
        )
        bad = np.abs(np.einsum("ij,ij->i", ref, e3, optimize=True)) > 0.999
        if np.any(bad):
            raise ValueError("reference_axis is parallel to at least one spin direction")

    e1 = ref - np.einsum("ij,ij->i", ref, e3, optimize=True)[:, None] * e3
    e1 = normalize_vectors(e1, name="local e1")
    e2 = np.cross(e3, e1)
    e2 = normalize_vectors(e2, name="local e2")
    return np.stack([e1, e2, e3], axis=2)


@dataclass
class SpinFrameInfo:
    spin_directions: np.ndarray
    frames: np.ndarray
    spin_pattern: np.ndarray
    eta: np.ndarray
    hp_annihilation: np.ndarray
    hp_creation: np.ndarray

    @property
    def nmag(self) -> int:
        return int(self.spin_directions.shape[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "nmag": self.nmag,
            "spin_directions": self.spin_directions,
            "frames": self.frames,
            "spin_pattern": self.spin_pattern,
            "eta": self.eta,
            "hp_annihilation": self.hp_annihilation,
            "hp_creation": self.hp_creation,
        }


def build_spin_frame_info(
    *,
    nmag: int | None = None,
    spin_direction=(0.0, 1.0, 0.0),
    spin_pattern="auto",
    spin_directions=None,
    reference_axis=None,
) -> SpinFrameInfo:
    if spin_directions is None:
        if nmag is None:
            raise ValueError("nmag is required when explicit spin_directions are not provided")
        pattern = parse_spin_pattern(spin_pattern, int(nmag))
        directions = collinear_spin_directions(spin_direction, pattern)
    else:
        directions = normalize_vectors(np.asarray(spin_directions, dtype=np.float64).reshape(-1, 3), name="spin_directions")
        pattern = np.sign(directions @ normalize_vectors(np.asarray(spin_direction, dtype=np.float64).reshape(3)))
        pattern[pattern == 0.0] = 1.0

    frames = local_frames_from_spin_directions(directions, reference_axis=reference_axis)
    nsite = int(frames.shape[0])
    eta = np.diag(np.r_[np.ones(nsite), -np.ones(nsite)]).astype(np.complex128)
    hp_annihilation = (frames[:, :, 0] - 1.0j * frames[:, :, 1]) / np.sqrt(2.0)
    hp_creation = (frames[:, :, 0] + 1.0j * frames[:, :, 1]) / np.sqrt(2.0)
    return SpinFrameInfo(
        spin_directions=directions,
        frames=frames,
        spin_pattern=pattern.astype(np.float64, copy=False),
        eta=eta,
        hp_annihilation=hp_annihilation,
        hp_creation=hp_creation,
    )


def dmi_vector_to_matrix(dmi: np.ndarray) -> np.ndarray:
    d = np.asarray(dmi, dtype=np.float64)
    out = np.zeros(d.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 1] = d[..., 2]
    out[..., 0, 2] = -d[..., 1]
    out[..., 1, 0] = -d[..., 2]
    out[..., 1, 2] = d[..., 0]
    out[..., 2, 0] = d[..., 1]
    out[..., 2, 1] = -d[..., 0]
    return out


def compose_exchange_tensor(iso=None, aniso=None, dmi=None) -> np.ndarray:
    parts = []
    if iso is not None:
        parts.append(np.asarray(iso, dtype=np.float64)[..., None, None] * np.eye(3, dtype=np.float64))
    if aniso is not None:
        parts.append(np.asarray(aniso, dtype=np.float64))
    if dmi is not None:
        parts.append(dmi_vector_to_matrix(dmi))
    if not parts:
        raise ValueError("At least one of iso, aniso, dmi is required")
    out = np.zeros(np.broadcast_shapes(*[p.shape for p in parts]), dtype=np.float64)
    for part in parts:
        out = out + part
    return out


def project_tensor_to_local(tensor, frames_i, frames_j) -> np.ndarray:
    return np.einsum("...ai,...ab,...bj->...ij", frames_i, tensor, frames_j, optimize=True)


def linear_hp_coefficients_from_tensor(tensor, ii, jj, frames, S: float) -> dict[str, np.ndarray]:
    """Project exchange tensors to local frames and return linear HP coefficients.

    For `H = - S_i^a J_ij^{ab} S_j^b`, local `e3` is the ordered spin axis.
    Keeping one transverse spin operator gives coefficients for
    `(a_i, a_i^dagger, a_j, a_j^dagger)` per bond-like entry.
    """
    tensor = np.asarray(tensor, dtype=np.float64).reshape(-1, 3, 3)
    ii = np.asarray(ii, dtype=np.int32).reshape(-1)
    jj = np.asarray(jj, dtype=np.int32).reshape(-1)
    if tensor.shape[0] != ii.size or ii.size != jj.size:
        raise ValueError(f"tensor/ii/jj size mismatch: tensor={tensor.shape}, ii={ii.shape}, jj={jj.shape}")
    K = project_tensor_to_local(tensor, np.asarray(frames, dtype=np.float64)[ii], np.asarray(frames, dtype=np.float64)[jj])
    pref = -float(S) * np.sqrt(float(S) / 2.0)
    return {
        "i_atom": ii,
        "j_atom": jj,
        "K_local": K,
        "annihilate_i": pref * (K[:, 0, 2] - 1.0j * K[:, 1, 2]),
        "create_i": pref * (K[:, 0, 2] + 1.0j * K[:, 1, 2]),
        "annihilate_j": pref * (K[:, 2, 0] - 1.0j * K[:, 2, 1]),
        "create_j": pref * (K[:, 2, 0] + 1.0j * K[:, 2, 1]),
    }


def _positive_bdg_modes(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nmag = H.shape[0] // 2
    eta = np.diag(np.r_[np.ones(nmag), -np.ones(nmag)]).astype(np.complex128)
    vals, vecs = np.linalg.eig(eta @ H)
    order = np.argsort(vals.real)
    omega = []
    modes = []
    for idx in order:
        val = vals[idx].real
        vec = vecs[:, idx]
        norm = float(np.real(np.vdot(vec[:nmag], vec[:nmag]) - np.vdot(vec[nmag:], vec[nmag:])))
        if val > -1.0e-9 and norm > 1.0e-9:
            omega.append(max(0.0, val))
            modes.append(vec / np.sqrt(norm))
        if len(omega) == nmag:
            break
    if len(omega) < nmag:
        pos = np.argsort(vals.real)[-nmag:]
        pos = pos[np.argsort(vals.real[pos])]
        omega = [float(vals[i].real) for i in pos]
        modes = [vecs[:, i] for i in pos]
    return np.asarray(omega, dtype=np.float64), np.asarray(modes, dtype=np.complex128).T


def tensor_ab_blocks(
    k,
    S,
    tensor,
    ii,
    jj,
    rr,
    lattice,
    atom_pos,
    frames,
    *,
    bond_factor: float = 1.0,
    anisotropy_mev: float = 0.0,
    zeeman_field_mev=None,
) -> tuple[np.ndarray, np.ndarray]:
    nmag = int(frames.shape[0])
    tensor = np.asarray(tensor, dtype=np.float64)
    ii = np.asarray(ii, dtype=np.int32).reshape(-1)
    jj = np.asarray(jj, dtype=np.int32).reshape(-1)
    rr = np.asarray(rr, dtype=np.float64).reshape(-1, 3)
    delta = np.asarray(atom_pos, dtype=np.float64)[jj] - np.asarray(atom_pos, dtype=np.float64)[ii]
    delta = delta + rr @ np.asarray(lattice, dtype=np.float64)
    phase = np.exp(1.0j * (delta @ np.asarray(k, dtype=np.float64).reshape(3)))

    K = project_tensor_to_local(tensor, frames[ii], frames[jj])
    kt = K[:, :2, :2]
    k33 = K[:, 2, 2]
    p = np.array([1.0 + 0.0j, -1.0j], dtype=np.complex128)
    q = np.array([1.0 + 0.0j, 1.0j], dtype=np.complex128)
    scale = float(bond_factor) / float(S)

    A = np.eye(nmag, dtype=np.complex128) * (float(anisotropy_mev) / float(S))
    if zeeman_field_mev is not None:
        field = np.asarray(zeeman_field_mev, dtype=np.float64).reshape(3)
        local_h = np.einsum("ia,a->i", frames[:, :, 2], field, optimize=True)
        A += np.diag(local_h.astype(np.complex128, copy=False))
    B = np.zeros((nmag, nmag), dtype=np.complex128)
    np.add.at(A, (ii, ii), scale * k33)
    Bij = -0.5 * scale * np.einsum("a,nab,b->n", p, kt, p, optimize=True) * phase
    Aij = -0.5 * scale * np.einsum("a,nab,b->n", q, kt, p, optimize=True) * phase
    np.add.at(B, (ii, jj), Bij)
    np.add.at(A, (ii, jj), Aij)
    return A, B


def bdg_matrix_from_tensor(k, S, tensor, ii, jj, rr, lattice, atom_pos, frames, **kwargs) -> np.ndarray:
    nmag = int(frames.shape[0])
    A_k, B_k = tensor_ab_blocks(k, S, tensor, ii, jj, rr, lattice, atom_pos, frames, **kwargs)
    A_mk, B_mk = tensor_ab_blocks(-np.asarray(k, dtype=np.float64), S, tensor, ii, jj, rr, lattice, atom_pos, frames, **kwargs)
    H = np.zeros((2 * nmag, 2 * nmag), dtype=np.complex128)
    H[:nmag, :nmag] = A_k
    H[:nmag, nmag:] = B_k
    H[nmag:, :nmag] = B_mk.conj()
    H[nmag:, nmag:] = A_mk.conj()
    return 0.5 * (H + H.conj().T)


_TENSOR_LSWT_STATE = {}


def _solve_tensor_lswt_serial(kpts, S, tensor, ii, jj, rr, lattice, atom_pos, spin_info: SpinFrameInfo, kwargs):
    kpts = np.asarray(kpts, dtype=np.float64).reshape(-1, 3)
    nmag = spin_info.nmag
    omega = np.zeros((kpts.shape[0], nmag), dtype=np.float64)
    modes = np.zeros((kpts.shape[0], 2 * nmag, nmag), dtype=np.complex128)
    for ik, k in enumerate(kpts):
        H = bdg_matrix_from_tensor(k, S, tensor, ii, jj, rr, lattice, atom_pos, spin_info.frames, **kwargs)
        omega[ik], modes[ik] = _positive_bdg_modes(H)
    return omega, modes


def _set_tensor_lswt_state(S, tensor, ii, jj, rr, lattice, atom_pos, spin_info, kwargs):
    _TENSOR_LSWT_STATE.clear()
    _TENSOR_LSWT_STATE.update(
        {
            "S": S,
            "tensor": tensor,
            "ii": ii,
            "jj": jj,
            "rr": rr,
            "lattice": lattice,
            "atom_pos": atom_pos,
            "spin_info": spin_info,
            "kwargs": kwargs,
        }
    )


def _solve_tensor_lswt_chunk(task):
    start, kpts = task
    if not _TENSOR_LSWT_STATE:
        raise RuntimeError("tensor LSWT worker state is not initialized")
    omega, modes = _solve_tensor_lswt_serial(
        kpts,
        _TENSOR_LSWT_STATE["S"],
        _TENSOR_LSWT_STATE["tensor"],
        _TENSOR_LSWT_STATE["ii"],
        _TENSOR_LSWT_STATE["jj"],
        _TENSOR_LSWT_STATE["rr"],
        _TENSOR_LSWT_STATE["lattice"],
        _TENSOR_LSWT_STATE["atom_pos"],
        _TENSOR_LSWT_STATE["spin_info"],
        _TENSOR_LSWT_STATE["kwargs"],
    )
    return int(start), omega, modes


def solve_tensor_lswt(kpts, S, tensor, ii, jj, rr, lattice, atom_pos, spin_info: SpinFrameInfo, nproc=1, **kwargs):
    kpts = np.asarray(kpts, dtype=np.float64).reshape(-1, 3)
    nk = int(kpts.shape[0])
    nmag = spin_info.nmag
    nproc = max(1, min(int(nproc), nk))
    if nproc == 1 or nk <= 1 or "fork" not in mp.get_all_start_methods():
        return _solve_tensor_lswt_serial(kpts, S, tensor, ii, jj, rr, lattice, atom_pos, spin_info, kwargs)

    omega = np.zeros((nk, nmag), dtype=np.float64)
    modes = np.zeros((nk, 2 * nmag, nmag), dtype=np.complex128)
    _set_tensor_lswt_state(S, tensor, ii, jj, rr, lattice, atom_pos, spin_info, kwargs)
    edges = np.linspace(0, nk, nproc + 1, dtype=np.int64)
    tasks = [
        (int(edges[i]), kpts[int(edges[i]): int(edges[i + 1])])
        for i in range(nproc)
        if int(edges[i + 1]) > int(edges[i])
    ]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=len(tasks)) as pool:
        for start, omega_chunk, modes_chunk in pool.map(_solve_tensor_lswt_chunk, tasks):
            stop = int(start) + int(omega_chunk.shape[0])
            omega[int(start):stop] = omega_chunk
            modes[int(start):stop] = modes_chunk
    return omega, modes


def infer_nmag_from_tensor_h5(path: str) -> int:
    with h5py.File(path, "r") as h5:
        ii = np.asarray(h5["bonds/mag_i_atom"], dtype=np.int32).reshape(-1)
        jj = np.asarray(h5["bonds/mag_j_atom"], dtype=np.int32).reshape(-1)
    if ii.size == 0 and jj.size == 0:
        raise ValueError(f"No magnetic bond metadata found in {path}")
    return int(max(int(ii.max(initial=0)), int(jj.max(initial=0))) + 1)


def main():
    ap = argparse.ArgumentParser(description="Build local-frame LSWT spin information")
    ap.add_argument("--tensor_h5", default=None, help="Optional tensor HDF5 used only to infer nmag")
    ap.add_argument("--nmag", type=int, default=None, help="Number of magnetic sites")
    ap.add_argument("--spin_direction", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    ap.add_argument("--spin_pattern", default="auto", help="'auto', 'fm', or comma-separated signs")
    ap.add_argument("--spin_directions", type=float, nargs="*", default=None, help="Explicit flattened per-site directions")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    nmag = args.nmag
    if nmag is None and args.tensor_h5:
        nmag = infer_nmag_from_tensor_h5(args.tensor_h5)
    explicit = None
    if args.spin_directions:
        if len(args.spin_directions) % 3 != 0:
            raise ValueError("--spin_directions length must be a multiple of 3")
        explicit = np.asarray(args.spin_directions, dtype=np.float64).reshape(-1, 3)
        nmag = explicit.shape[0]
    info = build_spin_frame_info(
        nmag=nmag,
        spin_direction=args.spin_direction,
        spin_pattern=args.spin_pattern,
        spin_directions=explicit,
    )
    print("[lswt] spin-frame summary")
    print(f"  nmag: {info.nmag}")
    print(f"  spin_pattern: {info.spin_pattern.tolist()}")
    print(f"  eta_shape: {info.eta.shape}")
    print(f"  frames_shape: {info.frames.shape}")
    for i in range(min(int(args.show), info.nmag)):
        print(f"  site={i} spin={info.spin_directions[i].tolist()}")
        print(f"    e1={info.frames[i, :, 0].tolist()}")
        print(f"    e2={info.frames[i, :, 1].tolist()}")
        print(f"    e3={info.frames[i, :, 2].tolist()}")


if __name__ == "__main__":
    main()
