import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np

from slw.core.cli_paths import resolve_path
from slw.magph.adapter import (
    build_legacy_djdu_tuple,
    load_djdu_npz,
    load_djr_h5,
    load_phonon_cache,
)


@dataclass
class MagphManifest:
    path: str
    workdir: str
    in_dir: str
    out_dir: str
    out_name: str
    legacy_j_npz: str
    phonon_cache: str
    legacy_djdu_npz: str = ""
    djr_h5: str = ""
    djdu_npz: str = ""
    rp_idx: tuple = (0, 0, 0)


@dataclass
class MagphShapes:
    n_j_bonds: int
    n_dj_entries: int
    n_qpoints: int
    n_modes: int
    n_atoms: int


def load_legacy_j_tuple(path: str):
    z = np.load(path, allow_pickle=True)
    return (
        np.array(z["J_iso"], dtype=np.float64),
        np.array(z["i_idx"], dtype=np.int32),
        np.array(z["j_idx"], dtype=np.int32),
        np.array(z["vector"], dtype=np.int32).reshape(-1, 3),
        np.array(z["R"], dtype=np.float64),
    )


def load_legacy_djdu_tuple(path: str, rp_idx=(0, 0, 0)):
    if str(path).lower().endswith((".h5", ".hdf5")):
        return build_legacy_djdu_tuple(load_djr_h5(path), rp_idx=tuple(int(x) for x in rp_idx))
    z = np.load(path, allow_pickle=True)
    return (
        np.array(z["moved_atom_idx"], dtype=np.int32),
        np.array(z["i_idx"], dtype=np.int32),
        np.array(z["j_idx"], dtype=np.int32),
        np.array(z["R"], dtype=np.int32),
        np.array(z["grad_J_vec"], dtype=np.complex128 if np.iscomplexobj(z["grad_J_vec"]) else np.float64),
    )


def load_manifest(manifest_path: str, workdir: str) -> MagphManifest:
    mpath = resolve_path(workdir, manifest_path)
    with open(mpath, "r") as f:
        raw = json.load(f)
    base = str(raw.get("workdir", workdir))
    def maybe_path(value):
        s = str(value or "")
        return resolve_path(base, s) if s else ""

    legacy_djdu_npz = maybe_path(raw.get("legacy_djdu_npz", ""))
    djr_h5 = maybe_path(raw.get("djr_h5", raw.get("djdu_h5", "")))
    djdu_npz = maybe_path(raw.get("djdu_npz", ""))
    if not legacy_djdu_npz and not djr_h5 and not djdu_npz:
        raise KeyError(
            f"Manifest {mpath} must contain legacy_djdu_npz, djr_h5, or djdu_npz"
        )
    return MagphManifest(
        path=mpath,
        workdir=str(raw.get("workdir", workdir)),
        in_dir=str(raw.get("in_dir", workdir)),
        out_dir=str(raw.get("out_dir", workdir)),
        out_name=str(raw.get("out_name", "magph_lifetime_prep")),
        legacy_j_npz=maybe_path(raw["legacy_j_npz"]),
        phonon_cache=maybe_path(raw["phonon_cache"]),
        legacy_djdu_npz=legacy_djdu_npz,
        djr_h5=djr_h5,
        djdu_npz=djdu_npz,
        rp_idx=tuple(int(x) for x in raw.get("rp_idx", [0, 0, 0])),
    )


def validate_legacy_j(path: str) -> int:
    z = np.load(path, allow_pickle=True)
    for k in ["J_iso", "i_idx", "j_idx", "vector", "R"]:
        if k not in z:
            raise KeyError(f"Missing key '{k}' in {path}")
    return int(len(z["J_iso"]))


def validate_legacy_djdu(path: str, rp_idx=(0, 0, 0)) -> int:
    if str(path).lower().endswith((".h5", ".hdf5")):
        dj = load_djr_h5(path)
        rp = np.asarray(rp_idx, dtype=np.int32)
        if not np.any(np.all(np.asarray(dj["rp_list"], dtype=np.int32) == rp[None, :], axis=1)):
            raise ValueError(f"Rp index {tuple(int(x) for x in rp)} not found in {path}")
        return int(len(dj["target_indices"]) * dj["pair_R"].shape[0])
    z = np.load(path, allow_pickle=True)
    for k in ["moved_atom_idx", "i_idx", "j_idx", "R", "grad_J_vec"]:
        if k not in z:
            raise KeyError(f"Missing key '{k}' in {path}")
    return int(len(z["i_idx"]))


def load_djdu_tuple(manifest: MagphManifest):
    path = (
        manifest.djr_h5
        if manifest.djr_h5
        else (manifest.legacy_djdu_npz if manifest.legacy_djdu_npz else manifest.djdu_npz)
    )
    return load_legacy_djdu_tuple(path, rp_idx=manifest.rp_idx)


def _payload_from_flat_legacy(path: str, rp_idx=(0, 0, 0)) -> Dict[str, Any]:
    """Lift a single-Rp legacy tuple into the real-space payload schema."""
    moved, ii, jj, rr, grad = load_legacy_djdu_tuple(path, rp_idx=rp_idx)
    targets = np.unique(np.asarray(moved, dtype=np.int32))
    target_slot = {int(atom): slot for slot, atom in enumerate(targets)}
    bond_rows = []
    bond_slot = {}
    for i, j, r in zip(ii, jj, rr):
        key = (int(i), int(j), tuple(int(x) for x in r))
        if key not in bond_slot:
            bond_slot[key] = len(bond_rows)
            bond_rows.append((key[0], key[1], *key[2]))
    values = np.zeros((1, targets.size, 3, len(bond_rows)), dtype=np.asarray(grad).dtype)
    for atom, i, j, r, vector in zip(moved, ii, jj, rr, grad):
        key = (int(i), int(j), tuple(int(x) for x in r))
        values[0, target_slot[int(atom)], :, bond_slot[key]] += vector
    return {
        "values_rp": values,
        "rp_list": np.asarray(rp_idx, dtype=np.int32).reshape(1, 3),
        "targets": [f"Atom{int(atom) + 1}" for atom in targets],
        "target_indices": targets,
        "axes": ["x", "y", "z"],
        "pair_R": np.asarray(bond_rows, dtype=np.int32),
        "units": "meV/A",
        "source": path,
        "single_rp_legacy": True,
    }


def load_djdu_payload(manifest: MagphManifest) -> Dict[str, Any]:
    """Load the complete dJ(R,Rp) payload used by atomic-gauge vertices."""
    if manifest.djr_h5:
        return load_djr_h5(manifest.djr_h5)
    if manifest.djdu_npz:
        return load_djdu_npz(manifest.djdu_npz)
    return _payload_from_flat_legacy(manifest.legacy_djdu_npz, manifest.rp_idx)


def _parse_shell_list(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, np.ndarray)):
        items = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"none", "false", "off", "no"}:
            return ()
        items = [x for x in re.split(r"[\s,;]+", text) if x]
    shells = []
    seen = set()
    for item in items:
        shell = int(item)
        if shell <= 0:
            raise ValueError(f"Shell indices are 1-based positive integers, got {shell}")
        if shell not in seen:
            seen.add(shell)
            shells.append(shell)
    return tuple(shells)


def _shell_ids_from_distances(distances, tol: float) -> np.ndarray:
    dist = np.asarray(distances, dtype=np.float64).reshape(-1)
    shells = np.zeros(dist.shape, dtype=np.int32)
    finite = np.isfinite(dist)
    if not np.any(finite):
        raise ValueError("Cannot apply exclude_shells because J bond distances are unavailable")

    tol = max(float(tol), 0.0)
    unique = []
    for val in np.sort(dist[finite]):
        if not unique or abs(float(val) - unique[-1]) > tol:
            unique.append(float(val))
    for ishell, ref in enumerate(unique, start=1):
        shells[np.abs(dist - ref) <= tol] = ishell
    if np.any(finite & (shells == 0)):
        raise RuntimeError("Internal shell assignment failed for finite J bond distances")
    return shells


def _bond_keys(i_idx, j_idx, r_vec) -> list[tuple[int, int, tuple[int, int, int]]]:
    return [
        (int(i), int(j), (int(r[0]), int(r[1]), int(r[2])))
        for i, j, r in zip(np.asarray(i_idx).reshape(-1), np.asarray(j_idx).reshape(-1), np.asarray(r_vec).reshape(-1, 3))
    ]


def _filter_j_tuple(j0, keep_mask):
    keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
    return (
        np.asarray(j0[0])[keep],
        np.asarray(j0[1])[keep],
        np.asarray(j0[2])[keep],
        np.asarray(j0[3], dtype=np.int32).reshape(-1, 3)[keep],
        np.asarray(j0[4])[keep],
    )


def _filter_djdu_tuple(djdu, keep_mask):
    keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
    return (
        np.asarray(djdu[0])[keep],
        np.asarray(djdu[1])[keep],
        np.asarray(djdu[2])[keep],
        np.asarray(djdu[3], dtype=np.int32).reshape(-1, 3)[keep],
        np.asarray(djdu[4])[keep],
    )


def filter_exchange_shells(j0, djdu, cfg: Dict[str, Any]):
    """
    Drop selected distance shells from static J and/or dynamic dJ tuples.

    Shell indices are reconstructed from sorted unique J-bond distances because
    the legacy runtime tuples do not store the original HDF5 shell column.
    """
    shells = _parse_shell_list(cfg.get("exclude_shells", cfg.get("remove_shells", None)))
    if not shells:
        return j0, djdu, {
            "enabled": False,
            "exclude_shells": [],
            "apply": "none",
            "shell_tol": float(cfg.get("shell_tol", 1.0e-4)),
        }

    apply = str(cfg.get("exclude_shell_apply", cfg.get("shell_filter_apply", "both"))).strip().lower()
    aliases = {
        "both": "both",
        "all": "both",
        "j+dJ": "both",
        "j+dj": "both",
        "jdj": "both",
        "j": "J",
        "static": "J",
        "dJ": "dJ",
        "dj": "dJ",
        "dynamic": "dJ",
    }
    apply = aliases.get(apply, apply)
    if apply not in {"both", "J", "dJ"}:
        raise ValueError("exclude_shell_apply must be one of: both, J, dJ")

    shell_tol = float(cfg.get("shell_tol", 1.0e-4))
    shell_ids = _shell_ids_from_distances(j0[4], shell_tol)
    excluded_shells = set(int(x) for x in shells)
    j_exclude_mask = np.isin(shell_ids, list(excluded_shells))
    excluded_keys = set(
        key for key, drop in zip(_bond_keys(j0[1], j0[2], j0[3]), j_exclude_mask) if bool(drop)
    )

    report = {
        "enabled": True,
        "exclude_shells": [int(x) for x in shells],
        "apply": apply,
        "shell_tol": shell_tol,
        "n_shells_detected": int(np.max(shell_ids)) if shell_ids.size else 0,
        "j_bonds_before": int(len(j0[0])),
        "dj_entries_before": int(len(djdu[0])),
        "j_bonds_dropped": 0,
        "dj_entries_dropped": 0,
    }

    if apply in {"both", "J"}:
        j_keep = ~j_exclude_mask
        report["j_bonds_dropped"] = int(np.count_nonzero(~j_keep))
        j0 = _filter_j_tuple(j0, j_keep)
        if len(j0[0]) == 0:
            raise ValueError(f"exclude_shells={shells} removed all static J bonds")

    if apply in {"both", "dJ"}:
        dj_keys = _bond_keys(djdu[1], djdu[2], djdu[3])
        dj_keep = np.asarray([key not in excluded_keys for key in dj_keys], dtype=bool)
        report["dj_entries_dropped"] = int(np.count_nonzero(~dj_keep))
        djdu = _filter_djdu_tuple(djdu, dj_keep)
        if len(djdu[0]) == 0:
            raise ValueError(f"exclude_shells={shells} removed all dynamic dJ entries")

    report["j_bonds_after"] = int(len(j0[0]))
    report["dj_entries_after"] = int(len(djdu[0]))
    return j0, djdu, report


def validate_manifest_djdu(manifest: MagphManifest) -> int:
    path = (
        manifest.djr_h5
        if manifest.djr_h5
        else (manifest.legacy_djdu_npz if manifest.legacy_djdu_npz else manifest.djdu_npz)
    )
    return validate_legacy_djdu(path, rp_idx=manifest.rp_idx)


def inspect_phonon_cache(path: str):
    ph = load_phonon_cache(path)
    n_q = int(ph["q_mesh_flat_frac"].shape[0])
    n_modes = int(ph["ph_en_flat"].shape[1])
    n_atoms = int(ph["ph_vec_flat"].shape[2])
    return n_q, n_modes, n_atoms


def build_shapes(manifest: MagphManifest) -> MagphShapes:
    n_j = validate_legacy_j(manifest.legacy_j_npz)
    n_dj = validate_manifest_djdu(manifest)
    n_q, n_modes, n_atoms = inspect_phonon_cache(manifest.phonon_cache)
    return MagphShapes(
        n_j_bonds=n_j,
        n_dj_entries=n_dj,
        n_qpoints=n_q,
        n_modes=n_modes,
        n_atoms=n_atoms,
    )


def _bytes_to_gb(nbytes: float) -> float:
    return float(nbytes) / (1024.0 ** 3)


def estimate_solver_memory_gb(shapes: MagphShapes, omega_points: int, matrix_dim: int = 4) -> Dict[str, float]:
    """
    Rough memory estimate for spectral solver kernels.
    """
    c128 = 16.0
    f64 = 8.0
    # g_pq_cache: (Nq, Nm, M, M) complex128
    g_pq = shapes.n_qpoints * shapes.n_modes * matrix_dim * matrix_dim * c128
    # e_kq_cache: (Nq, M) float64
    e_kq = shapes.n_qpoints * matrix_dim * f64
    # Atomic-gauge Lambda(q,nu,b), shared read-only by all k tasks on a rank.
    lambda_qnu_b = shapes.n_qpoints * shapes.n_modes * shapes.n_j_bonds * c128
    # sigma per k: (Nw, M, M) complex128
    sigma_k = omega_points * matrix_dim * matrix_dim * c128
    # conservative working buffer factor
    working = 1.25 * (g_pq + e_kq + lambda_qnu_b + sigma_k)
    return {
        "g_pq_cache_gb": _bytes_to_gb(g_pq),
        "e_kq_cache_gb": _bytes_to_gb(e_kq),
        "lambda_qnu_b_gb": _bytes_to_gb(lambda_qnu_b),
        "sigma_per_k_gb": _bytes_to_gb(sigma_k),
        "working_set_per_rank_gb": _bytes_to_gb(working),
    }


def estimate_lifetime_memory_gb(shapes: MagphShapes, matrix_dim: int = 4) -> Dict[str, float]:
    """
    Rough memory estimate for linewidth (lifetime) kernels.
    """
    c128 = 16.0
    f64 = 8.0
    g_pq = shapes.n_qpoints * shapes.n_modes * matrix_dim * matrix_dim * c128
    e_kq = shapes.n_qpoints * matrix_dim * f64
    lambda_qnu_b = shapes.n_qpoints * shapes.n_modes * shapes.n_j_bonds * c128
    linewidth_vec = matrix_dim * f64
    working = 1.2 * (g_pq + e_kq + lambda_qnu_b + linewidth_vec)
    return {
        "g_pq_cache_gb": _bytes_to_gb(g_pq),
        "e_kq_cache_gb": _bytes_to_gb(e_kq),
        "lambda_qnu_b_gb": _bytes_to_gb(lambda_qnu_b),
        "linewidth_vec_gb": _bytes_to_gb(linewidth_vec),
        "working_set_per_rank_gb": _bytes_to_gb(working),
    }


def recommend_solver_partition(shapes: MagphShapes, omega_points: int, target_rank_gb: float = 4.0, matrix_dim: int = 4) -> Dict[str, int]:
    """
    Recommend omega chunk count to keep per-rank working set under target.
    """
    est = estimate_solver_memory_gb(shapes, omega_points=omega_points, matrix_dim=matrix_dim)
    w = max(est["working_set_per_rank_gb"], 1e-9)
    chunks = int(np.ceil(w / max(target_rank_gb, 1e-6)))
    chunks = max(1, chunks)
    omega_chunk = int(np.ceil(omega_points / chunks))
    return {"omega_chunks": chunks, "omega_chunk_size": omega_chunk}


def build_runtime_payload(
    module_name: str,
    manifest: MagphManifest,
    shapes: MagphShapes,
    stage: str,
    extra: Dict[str, Any] = None,
) -> Dict[str, Any]:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module": module_name,
        "stage": stage,
        "manifest": manifest.path,
        "paths": {
            "legacy_j_npz": manifest.legacy_j_npz,
            "legacy_djdu_npz": manifest.legacy_djdu_npz,
            "djr_h5": manifest.djr_h5,
            "djdu_npz": manifest.djdu_npz,
            "phonon_cache": manifest.phonon_cache,
        },
        "shapes": asdict(shapes),
    }
    if extra:
        payload["analysis"] = extra
    return payload
