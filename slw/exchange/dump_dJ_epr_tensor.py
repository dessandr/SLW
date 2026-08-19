"""Dump analytic dJ tensor HDF5 output to readable text/TSV."""

from __future__ import annotations

import argparse
import os
import re

import h5py
import numpy as np


def _decode_array(arr):
    out = []
    for x in np.asarray(arr):
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _fmt_r(vec):
    vals = [int(x) for x in vec]
    return f"({vals[0]},{vals[1]},{vals[2]})"


def _atom_label(labels, idx):
    idx = int(idx)
    label = labels[idx] if idx < len(labels) else f"Atom{idx + 1}"
    return f"{label} (atom #{idx + 1})"


def _select_rp_indices(rp_grid, rp, all_rp):
    if all_rp:
        return list(range(len(rp_grid)))
    want = tuple(int(x) for x in rp)
    for i, row in enumerate(rp_grid):
        if tuple(int(x) for x in row) == want:
            return [i]
    raise ValueError(f"Rp={want} not found in HDF5 Rp grid")


def _read_payload(path):
    with h5py.File(path, "r") as h5:
        labels = _decode_array(h5["basic_data/atom_labels"][()])
        tensor_axes = _decode_array(h5["basic_data/tensor_axes"][()])
        qmesh = np.asarray(h5["basic_data/qmesh"][()], dtype=int)
        gi = np.asarray(h5["bonds/mag_i_atom"][()], dtype=int)
        gj = np.asarray(h5["bonds/mag_j_atom"][()], dtype=int)
        rvec = np.asarray(h5["bonds/R"][()], dtype=int)
        dist = np.asarray(h5["bonds/distance_ang"][()], dtype=float)
        targets_from_meta = np.asarray(h5["displacements/target_atom"][()], dtype=int)
        disp_axes = _decode_array(h5["displacements/axes"][()])
        rp_grid = np.asarray(h5["displacements/Rp"][()], dtype=int)
        complete = int(h5.attrs.get("complete", 1))
        n_completed = int(h5.attrs.get("n_completed_target_axis", len(targets_from_meta) * len(disp_axes)))

        def scan_dataset_group(name):
            out = {}
            sources = []
            if name in h5 and isinstance(h5[name], h5py.Group):
                grp = h5[name]
                for key in grp:
                    match = re.fullmatch(r"m(\d+)_b(\d+)", str(key))
                    if match is None:
                        continue
                    target = int(match.group(1)) - 1
                    ib = int(match.group(2)) - 1
                    out[(target, ib)] = np.asarray(grp[key][()])
                    sources.append(f"{name}/{key}")
            return out, sources

        tensors = {}
        tensor_sources = []
        tensors, tensor_sources = scan_dataset_group("dJ_tensor_r")
        if not tensors:
            for key in h5:
                match = re.fullmatch(r"m(\d+)_b(\d+)", str(key))
                if match is None:
                    continue
                target = int(match.group(1)) - 1
                ib = int(match.group(2)) - 1
                tensors[(target, ib)] = np.asarray(h5[key][()], dtype=float)
                tensor_sources.append(f"/{key}")
        dA, dA_sources = scan_dataset_group("dA_r")
        dJ_iso, _ = scan_dataset_group("dJ_iso_r")
        dDMI, _ = scan_dataset_group("dDMI_r")
        dGamma, _ = scan_dataset_group("dJ_gamma_r")
        dDMI_tensor, _ = scan_dataset_group("dJ_dmi_tensor_r")
        targets_from_data = sorted({int(k[0]) for k in tensors})
        targets = np.asarray(sorted(set(int(x) for x in targets_from_meta) | set(targets_from_data)), dtype=int)

    return {
        "labels": labels,
        "tensor_axes": tensor_axes,
        "qmesh": qmesh,
        "gi": gi,
        "gj": gj,
        "rvec": rvec,
        "dist": dist,
        "targets": targets,
        "targets_from_meta": targets_from_meta,
        "disp_axes": disp_axes,
        "rp_grid": rp_grid,
        "complete": complete,
        "n_completed": n_completed,
        "tensors": tensors,
        "tensor_sources": tensor_sources,
        "dA": dA,
        "dA_sources": dA_sources,
        "dJ_iso": dJ_iso,
        "dDMI": dDMI,
        "dGamma": dGamma,
        "dDMI_tensor": dDMI_tensor,
    }


def _tensor_block_to_lines(mat, tensor_axes, indent="      "):
    lines = []
    header = " ".join(f"{ax:>14s}" for ax in tensor_axes)
    lines.append(f"{indent}{'':>6s}{header}")
    for ia, ax in enumerate(tensor_axes):
        vals = " ".join(f"{mat[ia, ib]:14.6e}" for ib in range(len(tensor_axes)))
        lines.append(f"{indent}{ax:>6s}{vals}")
    return lines


def dump_text(path, out_txt, out_tsv, rp=(0, 0, 0), all_rp=False, threshold=None):
    payload = _read_payload(path)
    labels = payload["labels"]
    tensor_axes = payload["tensor_axes"]
    disp_axes = payload["disp_axes"]
    rp_indices = _select_rp_indices(payload["rp_grid"], rp, all_rp)
    threshold = None if threshold is None else float(threshold)

    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_tsv) or ".", exist_ok=True)

    with open(out_txt, "w") as f:
        f.write("# dJ/du tensor from compute_dJ_epr_tensor\n")
        f.write(f"# source_h5 = {os.path.abspath(path)}\n")
        f.write(f"# unit = meV/A\n")
        f.write(f"# qmesh = {tuple(int(x) for x in payload['qmesh'])}\n")
        f.write(f"# tensor_axes = {tuple(tensor_axes)} disp_axes = {tuple(disp_axes)}\n")
        f.write(f"# complete = {payload['complete']} n_completed_target_axis = {payload['n_completed']}\n\n")
        f.write(f"# targets_from_meta = {tuple(int(x) for x in payload['targets_from_meta'])}\n")
        f.write(f"# targets_from_data = {tuple(sorted({int(k[0]) for k in payload['tensors']}))}\n")
        f.write(f"# n_tensor_datasets = {len(payload['tensors'])}\n\n")
        f.write(f"# n_dA_datasets = {len(payload['dA'])}\n")
        if payload["tensor_sources"]:
            preview = ", ".join(payload["tensor_sources"][:8])
            suffix = "" if len(payload["tensor_sources"]) <= 8 else ", ..."
            f.write(f"# tensor_dataset_examples = {preview}{suffix}\n\n")

        for irp in rp_indices:
            f.write(f"## Rp = {_fmt_r(payload['rp_grid'][irp])}\n\n")
            order = sorted(range(len(payload["gi"])), key=lambda i: (payload["dist"][i], payload["gi"][i], payload["gj"][i], tuple(payload["rvec"][i])))
            for ib in order:
                gi = int(payload["gi"][ib])
                gj = int(payload["gj"][ib])
                f.write(
                    f"Bond b{ib + 1}: {_atom_label(labels, gi)} - {_atom_label(labels, gj)} "
                    f"R={_fmt_r(payload['rvec'][ib])} dist={payload['dist'][ib]:.8f} A\n"
                )
                wrote_any = False
                missing_targets = []
                for target in payload["targets"]:
                    target = int(target)
                    data = payload["tensors"].get((target, ib))
                    if data is None:
                        missing_targets.append(target)
                        continue
                    for idisp, disp_axis in enumerate(disp_axes):
                        mat = data[irp, idisp]
                        if threshold is not None and np.max(np.abs(mat)) < threshold:
                            continue
                        wrote_any = True
                        f.write(f"  moved {_atom_label(labels, target)} disp={disp_axis}\n")
                        iso_data = payload["dJ_iso"].get((target, ib))
                        dmi_data = payload["dDMI"].get((target, ib))
                        if iso_data is not None:
                            f.write(f"    dJ_iso = {float(iso_data[irp, idisp]):+.10e} meV/A\n")
                        if dmi_data is not None:
                            dv = np.asarray(dmi_data[irp, idisp], dtype=float)
                            f.write(
                                "    dDMI = "
                                f"({dv[0]:+.10e}, {dv[1]:+.10e}, {dv[2]:+.10e}) meV/A\n"
                            )
                        gamma_data = payload["dGamma"].get((target, ib))
                        dmi_tensor_data = payload["dDMI_tensor"].get((target, ib))
                        aniso = None
                        if gamma_data is not None and dmi_tensor_data is not None:
                            aniso = np.asarray(gamma_data[irp, idisp], dtype=float) + np.asarray(dmi_tensor_data[irp, idisp], dtype=float)
                        elif iso_data is not None:
                            aniso = np.array(mat, dtype=float, copy=True)
                            for a in range(min(3, aniso.shape[0], aniso.shape[1])):
                                aniso[a, a] -= float(iso_data[irp, idisp])
                        if aniso is not None:
                            f.write("    dJ_aniso = dGamma + dDMI_tensor:\n")
                            f.write("\n".join(_tensor_block_to_lines(aniso, tensor_axes, indent="      ")))
                            f.write("\n")
                        f.write("    dJ_full = dJ_iso*I + dJ_aniso:\n")
                        f.write("\n".join(_tensor_block_to_lines(mat, tensor_axes)))
                        f.write("\n")
                if not wrote_any:
                    if len(payload["targets"]) == 0:
                        f.write("  no target entries found in HDF5\n")
                    elif len(missing_targets) == len(payload["targets"]):
                        available = sorted(int(k[0]) for k in payload["tensors"] if int(k[1]) == ib)
                        f.write(
                            "  no tensor dataset matched this bond/target "
                            f"(targets={tuple(int(x) for x in payload['targets'])}, available_targets_for_bond={tuple(available)})\n"
                        )
                    else:
                        f.write("  no entries above threshold\n")
                f.write("\n")

    with open(out_tsv, "w") as f:
        f.write(
            "Rp1\tRp2\tRp3\tbond_index\tgi\tgj\ti_atom\tj_atom\ti_label\tj_label\t"
            "R1\tR2\tR3\tdist_A\ttarget_idx\ttarget_atom\ttarget_label\tdisp_axis\t"
            "component\ttensor_row\ttensor_col\tvalue_meV_per_A\n"
        )
        for irp in rp_indices:
            rp_vec = payload["rp_grid"][irp]
            for ib in range(len(payload["gi"])):
                gi = int(payload["gi"][ib])
                gj = int(payload["gj"][ib])
                ilab = labels[gi] if gi < len(labels) else f"Atom{gi + 1}"
                jlab = labels[gj] if gj < len(labels) else f"Atom{gj + 1}"
                for target in payload["targets"]:
                    target = int(target)
                    data = payload["tensors"].get((target, ib))
                    if data is None:
                        continue
                    tlab = labels[target] if target < len(labels) else f"Atom{target + 1}"
                    for idisp, disp_axis in enumerate(disp_axes):
                        mat = data[irp, idisp]
                        iso_data = payload["dJ_iso"].get((target, ib))
                        gamma_data = payload["dGamma"].get((target, ib))
                        dmi_tensor_data = payload["dDMI_tensor"].get((target, ib))
                        matrices = [("full", mat)]
                        if gamma_data is not None and dmi_tensor_data is not None:
                            matrices.append((
                                "aniso",
                                np.asarray(gamma_data[irp, idisp], dtype=float) + np.asarray(dmi_tensor_data[irp, idisp], dtype=float),
                            ))
                        elif iso_data is not None:
                            aniso = np.array(mat, dtype=float, copy=True)
                            for a in range(min(3, aniso.shape[0], aniso.shape[1])):
                                aniso[a, a] -= float(iso_data[irp, idisp])
                            matrices.append(("aniso", aniso))
                        for component, matrix in matrices:
                            for ia, row_ax in enumerate(tensor_axes):
                                for ja, col_ax in enumerate(tensor_axes):
                                    val = float(matrix[ia, ja])
                                    if threshold is not None and abs(val) < threshold:
                                        continue
                                    f.write(
                                        f"{int(rp_vec[0])}\t{int(rp_vec[1])}\t{int(rp_vec[2])}\t"
                                        f"{ib + 1}\t{gi}\t{gj}\t{gi + 1}\t{gj + 1}\t{ilab}\t{jlab}\t"
                                        f"{int(payload['rvec'][ib, 0])}\t{int(payload['rvec'][ib, 1])}\t{int(payload['rvec'][ib, 2])}\t"
                                        f"{payload['dist'][ib]:.12e}\t{target}\t{target + 1}\t{tlab}\t{disp_axis}\t"
                                        f"{component}\t{row_ax}\t{col_ax}\t{val:.12e}\n"
                                    )


def main():
    ap = argparse.ArgumentParser(description="Dump compute_dJ_epr_tensor HDF5 to readable text and TSV")
    ap.add_argument("h5", help="dJ tensor HDF5 from compute_dJ_epr_tensor")
    ap.add_argument("-o", "--out", default=None, help="Readable text output path")
    ap.add_argument("--tsv", default=None, help="TSV output path")
    ap.add_argument("--rp", type=int, nargs=3, default=[0, 0, 0], help="Rp cell to dump; default 0 0 0")
    ap.add_argument("--all_rp", action="store_true", help="Dump every Rp instead of a single Rp")
    ap.add_argument("--threshold", type=float, default=None, help="Omit entries with absolute value below threshold")
    args = ap.parse_args()

    base = os.path.splitext(args.h5)[0]
    out_txt = args.out or f"{base}.txt"
    out_tsv = args.tsv or f"{base}.all_components.tsv"
    dump_text(args.h5, out_txt, out_tsv, rp=args.rp, all_rp=args.all_rp, threshold=args.threshold)
    print(f"[dump-dJ-epr-tensor] wrote {out_txt}")
    print(f"[dump-dJ-epr-tensor] wrote {out_tsv}")


if __name__ == "__main__":
    main()
