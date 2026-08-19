import argparse
import json
import os
from itertools import combinations

import h5py
import numpy as np


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U"}:
            return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in value.tolist()]
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _frac_key(frac, mesh):
    out = []
    for x, n in zip(frac, mesh):
        xi = int(np.rint((float(x) % 1.0) * int(n))) % int(n)
        out.append(xi)
    return tuple(out)


def _infer_mesh_from_kpts(k_fracs, tol=1e-8):
    nk = []
    for iax in range(3):
        vals = np.sort(np.mod(k_fracs[:, iax], 1.0))
        uniq = []
        for v in vals:
            if not uniq or abs(v - uniq[-1]) > tol:
                uniq.append(float(v))
        nk.append(len(uniq))
    return tuple(int(x) for x in nk)


def _build_maps(k_fracs, q_fracs, nk_dense, nq_mesh):
    k_lookup = {_frac_key(kf, nk_dense): ik for ik, kf in enumerate(k_fracs)}
    q_lookup = {_frac_key(qf, nq_mesh): iq for iq, qf in enumerate(q_fracs)}

    kq_map = np.zeros((len(k_fracs), len(q_fracs)), dtype=np.int32)
    for ik, kf in enumerate(k_fracs):
        for iq, qf in enumerate(q_fracs):
            tgt = np.mod(kf + qf, 1.0)
            key = _frac_key(tgt, nk_dense)
            if key not in k_lookup:
                raise KeyError(f"k+q point not found for ik={ik}, iq={iq}, key={key}")
            kq_map[ik, iq] = k_lookup[key]

    q_minus_map = np.zeros(len(q_fracs), dtype=np.int32)
    for iq, qf in enumerate(q_fracs):
        key = _frac_key(np.mod(-qf, 1.0), nq_mesh)
        if key not in q_lookup:
            raise KeyError(f"-q point not found for iq={iq}, key={key}")
        q_minus_map[iq] = q_lookup[key]

    return kq_map, q_minus_map


def _ifft_to_real(g_kq_qfirst, nk_grid, nq_grid):
    nq, nk, m, n = g_kq_qfirst.shape
    g_mlwf = np.transpose(g_kq_qfirst, (1, 0, 2, 3))  # (Nk, Nq, m, n)

    if nk != int(np.prod(nk_grid)):
        raise ValueError(f"Nk={nk} incompatible with nk_grid={nk_grid}")
    if nq != int(np.prod(nq_grid)):
        raise ValueError(f"Nq={nq} incompatible with nq_grid={nq_grid}")

    temp = g_mlwf.reshape(tuple(nk_grid[::-1]) + (nq, m, n)).transpose(2, 1, 0, 3, 4, 5)
    g_re_q = np.fft.ifftn(temp, axes=(0, 1, 2))
    temp2 = g_re_q.reshape(tuple(nk_grid) + tuple(nq_grid[::-1]) + (m, n)).transpose(0, 1, 2, 5, 4, 3, 6, 7)
    g_re_rp = np.fft.ifftn(temp2, axes=(3, 4, 5))
    return g_re_rp


def _matrix_metrics(a, b):
    diff = a - b
    abs_err = float(np.linalg.norm(diff))
    denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-15)
    return abs_err, abs_err / denom


def _load_gkq(path):
    ap = os.path.abspath(path)
    with h5py.File(ap, "r") as h5:
        if "g_band" not in h5:
            raise KeyError(f"{ap} has no dataset 'g_band'")
        g_kq = np.asarray(h5["g_band"], dtype=np.complex128)
        if "k_fracs" not in h5 or "q_fracs" not in h5:
            raise KeyError(f"{ap} must contain 'k_fracs' and 'q_fracs'")

        k_fracs = np.asarray(h5["k_fracs"], dtype=np.float64)
        q_fracs = np.asarray(h5["q_fracs"], dtype=np.float64)
        attrs = {key: _decode_attr(val) for key, val in h5.attrs.items()}
        kq_map = np.asarray(h5["kq_map"], dtype=np.int32) if "kq_map" in h5 else None
        q_minus_map = np.asarray(h5["q_minus_map"], dtype=np.int32) if "q_minus_map" in h5 else None

    nk_dense = tuple(int(x) for x in attrs.get("nk_dense", _infer_mesh_from_kpts(k_fracs)))
    nq_mesh = tuple(int(x) for x in attrs.get("supercell_qmesh", (len(q_fracs), 1, 1)))
    if len(nk_dense) != 3 or len(nq_mesh) != 3:
        raise ValueError(f"Invalid nk_dense/nq_mesh for {ap}: nk={nk_dense}, nq={nq_mesh}")

    if kq_map is None or tuple(kq_map.shape) != (len(k_fracs), len(q_fracs)):
        kq_map, q_minus_map_built = _build_maps(k_fracs, q_fracs, nk_dense, nq_mesh)
        if q_minus_map is None:
            q_minus_map = q_minus_map_built
    if q_minus_map is None or len(q_minus_map) != len(q_fracs):
        _, q_minus_map = _build_maps(k_fracs, q_fracs, nk_dense, nq_mesh)

    q0_list = np.where(np.all(np.isclose(q_fracs, 0.0, atol=1e-12), axis=1))[0]
    iq0 = int(q0_list[0]) if len(q0_list) > 0 else -1

    return {
        "path": ap,
        "name": os.path.basename(ap),
        "g_kq": g_kq,
        "k_fracs": k_fracs,
        "q_fracs": q_fracs,
        "kq_map": kq_map,
        "q_minus_map": q_minus_map,
        "attrs": attrs,
        "channel": str(attrs.get("channel", "")),
        "label": str(attrs.get("label", "")),
        "axis": str(attrs.get("axis", "")),
        "basis": str(attrs.get("basis", "")),
        "nk_dense": nk_dense,
        "nq_mesh": nq_mesh,
        "iq0": iq0,
    }


def _finite_metrics(g_kq):
    finite_mask = np.isfinite(g_kq.real) & np.isfinite(g_kq.imag)
    n_total = int(g_kq.size)
    n_finite = int(np.count_nonzero(finite_mask))
    abs_g = np.abs(g_kq)
    return {
        "n_total": n_total,
        "n_nonfinite": int(n_total - n_finite),
        "finite_fraction": float(n_finite / max(n_total, 1)),
        "max_abs": float(np.max(abs_g)) if abs_g.size else 0.0,
        "mean_abs": float(np.mean(abs_g)) if abs_g.size else 0.0,
        "median_abs": float(np.median(abs_g)) if abs_g.size else 0.0,
        "p99_abs": float(np.percentile(abs_g, 99.0)) if abs_g.size else 0.0,
    }


def _element_metrics(g_kq, top_n=8):
    if g_kq.ndim != 4:
        return {}
    elem_mean = np.mean(np.abs(g_kq), axis=(0, 1))
    dim = elem_mean.shape[0]
    diag = np.diag(elem_mean)
    offdiag_mask = ~np.eye(dim, dtype=bool)
    offdiag_vals = elem_mean[offdiag_mask]
    flat = elem_mean.reshape(-1)
    top_n = min(int(top_n), flat.size)
    if top_n > 0:
        top_idx = np.argpartition(flat, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
    else:
        top_idx = np.array([], dtype=np.int64)
    top_entries = []
    for idx in top_idx:
        i = int(idx // dim)
        j = int(idx % dim)
        top_entries.append({"i": i, "j": j, "mean_abs": float(elem_mean[i, j])})
    return {
        "dim": int(dim),
        "diag_mean_abs": float(np.mean(diag)) if diag.size else 0.0,
        "diag_max_abs": float(np.max(diag)) if diag.size else 0.0,
        "offdiag_mean_abs": float(np.mean(offdiag_vals)) if offdiag_vals.size else 0.0,
        "offdiag_max_abs": float(np.max(offdiag_vals)) if offdiag_vals.size else 0.0,
        "top_entries": top_entries,
    }


def _momentum_pair_metrics(info):
    g_kq = info["g_kq"]
    q_minus_map = info["q_minus_map"]
    kq_map = info["kq_map"]
    q_fracs = info["q_fracs"]
    iq0 = info["iq0"]

    per_q = []
    mean_abs_all = []
    mean_rel_all = []
    max_abs = 0.0
    max_rel = 0.0
    q0_pair = None
    q0_hermiticity = None
    self_inverse = []

    for iq in range(g_kq.shape[0]):
        iqm = int(q_minus_map[iq])
        abs_terms = []
        rel_terms = []
        for ik in range(g_kq.shape[1]):
            mate = g_kq[iqm, int(kq_map[ik, iq])].conj().T
            abs_err, rel_err = _matrix_metrics(g_kq[iq, ik], mate)
            abs_terms.append(abs_err)
            rel_terms.append(rel_err)

        q_entry = {
            "iq": int(iq),
            "iq_mate": int(iqm),
            "q_frac": [float(x) for x in q_fracs[iq]],
            "mode": "self" if iqm == iq else "pair",
            "mean_abs": float(np.mean(abs_terms)) if abs_terms else 0.0,
            "max_abs": float(np.max(abs_terms)) if abs_terms else 0.0,
            "mean_rel": float(np.mean(rel_terms)) if rel_terms else 0.0,
            "max_rel": float(np.max(rel_terms)) if rel_terms else 0.0,
        }
        per_q.append(q_entry)
        mean_abs_all.extend(abs_terms)
        mean_rel_all.extend(rel_terms)
        max_abs = max(max_abs, q_entry["max_abs"])
        max_rel = max(max_rel, q_entry["max_rel"])

        if iq == iq0:
            q0_pair = q_entry
            herm_abs = []
            herm_rel = []
            for ik in range(g_kq.shape[1]):
                abs_err, rel_err = _matrix_metrics(g_kq[iq, ik], g_kq[iq, ik].conj().T)
                herm_abs.append(abs_err)
                herm_rel.append(rel_err)
            q0_hermiticity = {
                "iq": int(iq),
                "q_frac": [float(x) for x in q_fracs[iq]],
                "mean_abs": float(np.mean(herm_abs)) if herm_abs else 0.0,
                "max_abs": float(np.max(herm_abs)) if herm_abs else 0.0,
                "mean_rel": float(np.mean(herm_rel)) if herm_rel else 0.0,
                "max_rel": float(np.max(herm_rel)) if herm_rel else 0.0,
            }
        if iqm == iq:
            self_inverse.append(q_entry)

    offenders = sorted(per_q, key=lambda x: x["max_rel"], reverse=True)
    return {
        "relation": "g(k,q) = [g(k+q,-q)]^dagger",
        "n_q": int(len(per_q)),
        "mean_abs": float(np.mean(mean_abs_all)) if mean_abs_all else 0.0,
        "max_abs": float(max_abs),
        "mean_rel": float(np.mean(mean_rel_all)) if mean_rel_all else 0.0,
        "max_rel": float(max_rel),
        "q0_pair": q0_pair,
        "q0_hermiticity": q0_hermiticity,
        "self_inverse_q": self_inverse,
        "top_offenders": offenders[: min(8, len(offenders))],
        "per_q": per_q,
    }


def _realspace_pair_metrics(g_real, nk_dense, nq_mesh, rp_rule):
    abs_terms = []
    rel_terms = []
    max_abs = 0.0
    max_rel = 0.0
    top = []

    for re1 in range(nk_dense[0]):
        for re2 in range(nk_dense[1]):
            for re3 in range(nk_dense[2]):
                mate_re = ((-re1) % nk_dense[0], (-re2) % nk_dense[1], (-re3) % nk_dense[2])
                for rp1 in range(nq_mesh[0]):
                    for rp2 in range(nq_mesh[1]):
                        for rp3 in range(nq_mesh[2]):
                            if rp_rule == "minus":
                                mate_rp = (
                                    (rp1 - re1) % nq_mesh[0],
                                    (rp2 - re2) % nq_mesh[1],
                                    (rp3 - re3) % nq_mesh[2],
                                )
                                relation = "g(Re,Rp) = [g(-Re,Rp-Re)]^dagger"
                            elif rp_rule == "plus":
                                mate_rp = (
                                    (rp1 + re1) % nq_mesh[0],
                                    (rp2 + re2) % nq_mesh[1],
                                    (rp3 + re3) % nq_mesh[2],
                                )
                                relation = "g(Re,Rp) = [g(-Re,Rp+Re)]^dagger"
                            elif rp_rule == "same":
                                mate_rp = (rp1, rp2, rp3)
                                relation = "g(Re,Rp) = [g(-Re,Rp)]^dagger"
                            else:
                                raise ValueError(f"Unsupported rp_rule: {rp_rule}")

                            mat = g_real[re1, re2, re3, rp1, rp2, rp3]
                            mate = g_real[
                                mate_re[0], mate_re[1], mate_re[2], mate_rp[0], mate_rp[1], mate_rp[2]
                            ].conj().T
                            abs_err, rel_err = _matrix_metrics(mat, mate)
                            abs_terms.append(abs_err)
                            rel_terms.append(rel_err)
                            if abs_err > max_abs:
                                max_abs = abs_err
                            if rel_err > max_rel:
                                max_rel = rel_err
                            if len(top) < 8 or rel_err > top[-1]["rel_err"]:
                                top.append(
                                    {
                                        "Re_idx": [int(re1), int(re2), int(re3)],
                                        "Rp_idx": [int(rp1), int(rp2), int(rp3)],
                                        "mate_Re_idx": [int(x) for x in mate_re],
                                        "mate_Rp_idx": [int(x) for x in mate_rp],
                                        "abs_err": float(abs_err),
                                        "rel_err": float(rel_err),
                                    }
                                )
                                top = sorted(top, key=lambda x: x["rel_err"], reverse=True)[:8]

    return {
        "relation": relation,
        "mean_abs": float(np.mean(abs_terms)) if abs_terms else 0.0,
        "max_abs": float(max_abs),
        "mean_rel": float(np.mean(rel_terms)) if rel_terms else 0.0,
        "max_rel": float(max_rel),
        "top_offenders": top,
    }


def _realspace_metrics(info):
    g_real = _ifft_to_real(info["g_kq"], info["nk_dense"], info["nq_mesh"])
    minus_rule = _realspace_pair_metrics(g_real, info["nk_dense"], info["nq_mesh"], rp_rule="minus")
    plus_rule = _realspace_pair_metrics(g_real, info["nk_dense"], info["nq_mesh"], rp_rule="plus")
    same_rule = _realspace_pair_metrics(g_real, info["nk_dense"], info["nq_mesh"], rp_rule="same")
    best_name, best_data = min(
        [
            ("minus_rule", minus_rule),
            ("plus_rule", plus_rule),
            ("same_rp_rule", same_rule),
        ],
        key=lambda item: item[1]["max_rel"],
    )
    return {
        "shape": [int(x) for x in g_real.shape],
        "minus_rule": minus_rule,
        "plus_rule": plus_rule,
        "same_rp_rule": same_rule,
        "best_rule": {
            "name": best_name,
            "max_rel": float(best_data["max_rel"]),
            "mean_rel": float(best_data["mean_rel"]),
        },
    }


def _q0_sum_rule(groups):
    reports = []
    for group_key, entries in groups.items():
        if len(entries) < 2:
            continue
        iq0_list = [x["iq0"] for x in entries]
        if any(iq < 0 for iq in iq0_list):
            continue

        g0_sum = np.zeros_like(entries[0]["g_kq"][entries[0]["iq0"]], dtype=np.complex128)
        abs_ref = 0.0
        labels = []
        for x in entries:
            g0 = x["g_kq"][x["iq0"]]
            g0_sum += g0
            abs_ref += float(np.mean([np.linalg.norm(g0[ik]) for ik in range(g0.shape[0])]))
            labels.append(x["label"])

        abs_terms = []
        rel_terms = []
        for ik in range(g0_sum.shape[0]):
            abs_err = float(np.linalg.norm(g0_sum[ik]))
            denom = max(abs_ref, 1e-15)
            rel_err = abs_err / denom
            abs_terms.append(abs_err)
            rel_terms.append(rel_err)

        reports.append(
            {
                "group": {
                    "channel": group_key[0],
                    "axis": group_key[1],
                    "basis": group_key[2],
                    "nk_dense": list(group_key[3]),
                    "nq_mesh": list(group_key[4]),
                },
                "labels": sorted(labels),
                "n_labels": int(len(labels)),
                "relation": "sum_label g_label(k,q=0) ~= 0",
                "mean_abs": float(np.mean(abs_terms)) if abs_terms else 0.0,
                "max_abs": float(np.max(abs_terms)) if abs_terms else 0.0,
                "mean_rel": float(np.mean(rel_terms)) if rel_terms else 0.0,
                "max_rel": float(np.max(rel_terms)) if rel_terms else 0.0,
            }
        )
    return reports


def _pair_compare_metrics(info_a, info_b, sample_points=128):
    if info_a["g_kq"].shape != info_b["g_kq"].shape:
        raise ValueError(f"Shape mismatch: {info_a['g_kq'].shape} vs {info_b['g_kq'].shape}")
    if tuple(info_a["nk_dense"]) != tuple(info_b["nk_dense"]) or tuple(info_a["nq_mesh"]) != tuple(info_b["nq_mesh"]):
        raise ValueError(
            f"Grid mismatch: A nk/nq={info_a['nk_dense']}/{info_a['nq_mesh']} "
            f"vs B {info_b['nk_dense']}/{info_b['nq_mesh']}"
        )

    nq, nk = info_a["g_kq"].shape[:2]
    total = nq * nk
    rng = np.random.default_rng(42)
    n_pick = min(int(sample_points), total)
    chosen = sorted(set(rng.integers(0, total, size=n_pick).tolist()))
    if info_a["iq0"] >= 0:
        chosen.append(int(info_a["iq0"] * nk))
    chosen = sorted(set(chosen))

    fro_rel = []
    sv_rel = []
    for idx in chosen:
        iq = int(idx // nk)
        ik = int(idx % nk)
        a = info_a["g_kq"][iq, ik]
        b = info_b["g_kq"][iq, ik]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        fro_rel.append(abs(na - nb) / max(na, nb, 1e-15))

        sva = np.linalg.svd(a, compute_uv=False)
        svb = np.linalg.svd(b, compute_uv=False)
        sv_rel.append(float(np.linalg.norm(sva - svb) / max(np.linalg.norm(sva), np.linalg.norm(svb), 1e-15)))

    return {
        "file_a": info_a["path"],
        "file_b": info_b["path"],
        "label_a": info_a["label"],
        "label_b": info_b["label"],
        "channel": info_a["channel"],
        "axis": info_a["axis"],
        "basis": info_a["basis"],
        "n_samples": int(len(chosen)),
        "relation": "Gauge-invariant compare via ||g||_F and singular-value spectrum",
        "max_fro_rel_diff": float(np.max(fro_rel)) if fro_rel else 0.0,
        "mean_fro_rel_diff": float(np.mean(fro_rel)) if fro_rel else 0.0,
        "max_svd_rel_diff": float(np.max(sv_rel)) if sv_rel else 0.0,
        "mean_svd_rel_diff": float(np.mean(sv_rel)) if sv_rel else 0.0,
    }


def _periodic_invert(arr, axes):
    out = arr
    for ax in axes:
        idx = (-np.arange(out.shape[ax])) % out.shape[ax]
        out = np.take(out, idx, axis=ax)
    return out


def _signed_shift(shift, shape):
    out = []
    for s, n in zip(shift, shape):
        s = int(s)
        n = int(n)
        out.append(s if s <= n // 2 else s - n)
    return tuple(out)


def _best_cyclic_shift(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch in shift compare: {a.shape} vs {b.shape}")
    corr = np.fft.ifftn(np.fft.fftn(a) * np.conjugate(np.fft.fftn(b))).real
    idx = tuple(int(x) for x in np.unravel_index(np.argmax(corr), corr.shape))
    shape = a.shape
    axes = tuple(range(a.ndim))
    cand_raw = [
        idx,
        tuple((-x) % n for x, n in zip(idx, shape)),
    ]
    tried = set()
    best = None
    for shift in cand_raw:
        if shift in tried:
            continue
        tried.add(shift)
        rolled = np.roll(b, shift=shift, axis=axes)
        abs_err = float(np.linalg.norm(a - rolled))
        rel_err = abs_err / max(float(np.linalg.norm(a)), float(np.linalg.norm(rolled)), 1e-15)
        rep = {
            "shift_raw": [int(x) for x in shift],
            "shift_signed": [int(x) for x in _signed_shift(shift, shape)],
            "abs_err": abs_err,
            "rel_err": rel_err,
        }
        if best is None or rep["rel_err"] < best["rel_err"]:
            best = rep
    return best


def _support_shift_compare(info_a, info_b):
    g_real_a = _ifft_to_real(info_a["g_kq"], info_a["nk_dense"], info_a["nq_mesh"])
    g_real_b = _ifft_to_real(info_b["g_kq"], info_b["nk_dense"], info_b["nq_mesh"])
    map_a = np.linalg.norm(g_real_a, axis=(-2, -1))
    map_b = np.linalg.norm(g_real_b, axis=(-2, -1))

    candidates = {
        "direct": map_b,
        "Re_inverted": _periodic_invert(map_b, axes=(0, 1, 2)),
        "Rp_inverted": _periodic_invert(map_b, axes=(3, 4, 5)),
        "ReRp_inverted": _periodic_invert(map_b, axes=(0, 1, 2, 3, 4, 5)),
    }

    reports = {}
    for name, cand in candidates.items():
        reports[name] = _best_cyclic_shift(map_a, cand)

    best_name, best_data = min(reports.items(), key=lambda item: item[1]["rel_err"])
    return {
        "file_a": info_a["path"],
        "file_b": info_b["path"],
        "label_a": info_a["label"],
        "label_b": info_b["label"],
        "channel": info_a["channel"],
        "axis": info_a["axis"],
        "basis": info_a["basis"],
        "relation": "Best cyclic shift on real-space support map ||g(Re,Rp)||_F",
        "best_candidate": best_name,
        "best_rel_err": float(best_data["rel_err"]),
        "best_abs_err": float(best_data["abs_err"]),
        "best_shift_signed": best_data["shift_signed"],
        "candidates": reports,
    }


def _status_from_metrics(checks, warn_rel, primary_realspace_rule="minus"):
    rel_values = []
    finite_bad = False

    finite = checks.get("finite")
    if finite is not None and finite.get("n_nonfinite", 0) > 0:
        finite_bad = True

    momentum = checks.get("momentum_pair")
    if momentum:
        rel_values.append(float(momentum.get("max_rel", 0.0)))
        q0 = momentum.get("q0_hermiticity")
        if q0:
            rel_values.append(float(q0.get("max_rel", 0.0)))

    realspace = checks.get("realspace")
    if realspace:
        key = {
            "minus": "minus_rule",
            "plus": "plus_rule",
            "same": "same_rp_rule",
            "none": None,
        }.get(str(primary_realspace_rule))
        if key is None and str(primary_realspace_rule) != "none":
            raise ValueError(f"Unsupported primary_realspace_rule: {primary_realspace_rule}")
        if key is not None:
            rel_values.append(float(realspace[key].get("max_rel", 0.0)))

    worst = max(rel_values) if rel_values else 0.0
    if finite_bad:
        return "FAIL"
    if worst > float(warn_rel):
        return "WARN"
    return "PASS"


def _print_file_report(report, show_top=5):
    print(f"[gkq-check] {report['path']}")
    print(
        f"  meta: channel={report['channel']} label={report['label']} axis={report['axis']} "
        f"basis={report['basis']} shape={tuple(report['shape'])}"
    )
    finite = report["checks"]["finite"]
    print(
        f"  finite: nonfinite={finite['n_nonfinite']} max|g|={finite['max_abs']:.6e} "
        f"mean|g|={finite['mean_abs']:.6e} p99|g|={finite['p99_abs']:.6e}"
    )
    momentum = report["checks"]["momentum_pair"]
    print(
        f"  kq-pair: mean_rel={momentum['mean_rel']:.6e} max_rel={momentum['max_rel']:.6e} "
        f"mean_abs={momentum['mean_abs']:.6e} max_abs={momentum['max_abs']:.6e}"
    )
    if momentum["q0_hermiticity"] is not None:
        q0 = momentum["q0_hermiticity"]
        print(
            f"  q=0 hermiticity: mean_rel={q0['mean_rel']:.6e} max_rel={q0['max_rel']:.6e} "
            f"mean_abs={q0['mean_abs']:.6e} max_abs={q0['max_abs']:.6e}"
        )
    self_inv = momentum.get("self_inverse_q", [])
    if self_inv:
        worst_self = max(self_inv, key=lambda x: x["max_rel"])
        print(
            f"  self-inverse q: n={len(self_inv)} worst_q={worst_self['q_frac']} "
            f"max_rel={worst_self['max_rel']:.6e}"
        )
    realspace = report["checks"].get("realspace")
    if realspace is not None:
        minus_rule = realspace["minus_rule"]
        plus_rule = realspace["plus_rule"]
        same_rule = realspace["same_rp_rule"]
        print(
            f"  real-space: minus_rule max_rel={minus_rule['max_rel']:.6e} "
            f"plus_rule max_rel={plus_rule['max_rel']:.6e} same_rp max_rel={same_rule['max_rel']:.6e}"
        )
        print(
            f"    best candidate: {realspace['best_rule']['name']} "
            f"(max_rel={realspace['best_rule']['max_rel']:.6e})"
        )
    elem = report["checks"]["elements"]
    if elem.get("top_entries"):
        print(
            f"  element norms: diag_mean={elem['diag_mean_abs']:.6e} "
            f"offdiag_mean={elem['offdiag_mean_abs']:.6e}"
        )
        for item in elem["top_entries"][: min(show_top, len(elem["top_entries"]))]:
            print(f"    top ({item['i']:>2d},{item['j']:>2d}) mean|g|={item['mean_abs']:.6e}")
    offenders = momentum.get("top_offenders", [])
    if offenders:
        print("  worst q-points:")
        for item in offenders[: min(show_top, len(offenders))]:
            print(
                f"    iq={item['iq']:>3d} q={item['q_frac']} mode={item['mode']:<4s} "
                f"max_rel={item['max_rel']:.6e} mean_rel={item['mean_rel']:.6e}"
            )
    print(f"  status: {report['status']}")


def _print_group_sum_rule(reports):
    if not reports:
        return
    print("[gkq-check] q=0 translational sum-rule diagnostics")
    for rep in reports:
        grp = rep["group"]
        print(
            f"  group channel={grp['channel']} axis={grp['axis']} basis={grp['basis']} "
            f"labels={rep['labels']}"
        )
        print(
            f"    relation: {rep['relation']} "
            f"mean_rel={rep['mean_rel']:.6e} max_rel={rep['max_rel']:.6e}"
        )


def _print_pair_compares(compares):
    if not compares:
        return
    print("[gkq-check] cross-file gauge-invariant comparisons")
    for rep in compares:
        print(
            f"  {os.path.basename(rep['file_a'])} vs {os.path.basename(rep['file_b'])} "
            f"[{rep['channel']}/{rep['axis']}/{rep['basis']}]"
        )
        print(
            f"    fro: mean={rep['mean_fro_rel_diff']:.6e} max={rep['max_fro_rel_diff']:.6e}  "
            f"svd: mean={rep['mean_svd_rel_diff']:.6e} max={rep['max_svd_rel_diff']:.6e}"
        )


def _print_shift_compares(compares):
    if not compares:
        return
    print("[gkq-check] cross-file real-space support shift comparisons")
    for rep in compares:
        print(
            f"  {os.path.basename(rep['file_a'])} vs {os.path.basename(rep['file_b'])} "
            f"[{rep['channel']}/{rep['axis']}/{rep['basis']}]"
        )
        print(
            f"    best={rep['best_candidate']} shift={rep['best_shift_signed']} "
            f"rel_err={rep['best_rel_err']:.6e}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive physical-constraint checker for g(k,q) HDF5 files produced by compute_gkq_full."
    )
    parser.add_argument("inputs", nargs="+", help="Input g(k,q) HDF5 files")
    parser.add_argument("--out_json", type=str, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--skip_realspace",
        action="store_true",
        help="Skip IFFT-based real-space relation checks",
    )
    parser.add_argument(
        "--compare_invariants",
        action="store_true",
        help="For compatible multi-file inputs, compare gauge-invariant Frobenius/SVD fingerprints",
    )
    parser.add_argument(
        "--sum_rule",
        action="store_true",
        help="For compatible multi-file inputs, check q=0 translational sum rule by summing over labels",
    )
    parser.add_argument("--compare_samples", type=int, default=128, help="Sample count for invariant pairwise comparisons")
    parser.add_argument(
        "--compare_shifts",
        action="store_true",
        help="For compatible multi-file inputs, compare real-space support maps ||g(Re,Rp)||_F under cyclic shifts/inversions.",
    )
    parser.add_argument("--top_n", type=int, default=8, help="How many dominant matrix elements to retain in reports")
    parser.add_argument("--show_top", type=int, default=5, help="How many offending/top entries to print")
    parser.add_argument("--warn_rel", type=float, default=1e-6, help="Relative-error threshold for PASS/WARN status")
    parser.add_argument(
        "--primary_realspace_rule",
        choices=["minus", "plus", "same", "none"],
        default="minus",
        help="Which real-space pairing rule participates in PASS/WARN status. Others are still reported diagnostically.",
    )
    args = parser.parse_args()

    infos = [_load_gkq(p) for p in args.inputs]
    reports = []

    for info in infos:
        checks = {
            "finite": _finite_metrics(info["g_kq"]),
            "elements": _element_metrics(info["g_kq"], top_n=args.top_n),
            "momentum_pair": _momentum_pair_metrics(info),
        }
        if not args.skip_realspace:
            checks["realspace"] = _realspace_metrics(info)
        report = {
            "path": info["path"],
            "channel": info["channel"],
            "label": info["label"],
            "axis": info["axis"],
            "basis": info["basis"],
            "shape": [int(x) for x in info["g_kq"].shape],
            "nk_dense": [int(x) for x in info["nk_dense"]],
            "nq_mesh": [int(x) for x in info["nq_mesh"]],
            "checks": checks,
        }
        report["status"] = _status_from_metrics(
            report["checks"],
            warn_rel=args.warn_rel,
            primary_realspace_rule=args.primary_realspace_rule,
        )
        reports.append(report)

    sum_rule_reports = []
    if args.sum_rule and len(infos) > 1:
        groups = {}
        for info in infos:
            key = (
                info["channel"],
                info["axis"],
                info["basis"],
                tuple(info["nk_dense"]),
                tuple(info["nq_mesh"]),
            )
            groups.setdefault(key, []).append(info)
        sum_rule_reports = _q0_sum_rule(groups)

    compare_reports = []
    shift_compare_reports = []
    if args.compare_invariants and len(infos) > 1:
        by_group = {}
        for info in infos:
            key = (
                info["channel"],
                info["axis"],
                info["basis"],
                tuple(info["nk_dense"]),
                tuple(info["nq_mesh"]),
                tuple(info["g_kq"].shape),
            )
            by_group.setdefault(key, []).append(info)
        for entries in by_group.values():
            if len(entries) < 2:
                continue
            entries = sorted(entries, key=lambda x: x["label"])
            for info_a, info_b in combinations(entries, 2):
                compare_reports.append(
                    _pair_compare_metrics(info_a, info_b, sample_points=args.compare_samples)
                )
                if args.compare_shifts:
                    shift_compare_reports.append(_support_shift_compare(info_a, info_b))

    for rep in reports:
        _print_file_report(rep, show_top=args.show_top)
    _print_group_sum_rule(sum_rule_reports)
    _print_pair_compares(compare_reports)
    _print_shift_compares(shift_compare_reports)

    payload = {
        "method": "gkq_constraint_checker",
        "n_inputs": int(len(reports)),
        "reports": reports,
        "sum_rule_reports": sum_rule_reports,
        "compare_reports": compare_reports,
        "shift_compare_reports": shift_compare_reports,
    }
    if args.out_json:
        out_path = os.path.abspath(args.out_json)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[gkq-check] wrote JSON report: {out_path}")


if __name__ == "__main__":
    main()
