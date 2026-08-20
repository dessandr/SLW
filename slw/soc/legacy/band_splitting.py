"""
Archived SOC/noSOC Wannier band splitting analysis.

This module compares gauge-invariant eigenenergies from separately
Wannierized SOC and noSOC Hamiltonians. It does not subtract Hamiltonian
matrices, so it is suitable when the final Wannier gauges are unrelated.

Example
-------
python -m slw.soc.band_splitting \
    --root /path/to/soc --seed material --pairs 15,16 17,18 --lambda-denom 2.5
"""

import argparse
import csv
import glob
import os
from dataclasses import dataclass

import numpy as np

from slw.core.cli_paths import resolve_path, resolve_workdir
from slw.core.wannier_io import read_wannier_hr


@dataclass(frozen=True)
class WannierHR:
    dim: int
    r_vecs: np.ndarray
    h_r: np.ndarray


def _as_path(path):
    return os.fspath(path)


def infer_seed(wsoc_dir, wosoc_dir):
    """Infer a common Wannier seed from matching *_hr.dat files."""
    soc_seeds = {
        os.path.basename(path)[:-7]
        for path in glob.glob(os.path.join(_as_path(wsoc_dir), "*_hr.dat"))
    }
    nosoc_seeds = {
        os.path.basename(path)[:-7]
        for path in glob.glob(os.path.join(_as_path(wosoc_dir), "*_hr.dat"))
    }
    common = sorted(soc_seeds & nosoc_seeds)
    if len(common) == 1:
        return common[0]
    if not common:
        raise ValueError(
            f"Could not infer seed: no common *_hr.dat files in {wsoc_dir} and {wosoc_dir}"
        )
    raise ValueError(f"Could not infer unique seed from {common}; pass --seed explicitly")


def read_kpoints(path):
    """Read a Wannier90 *_band.kpt style file in fractional coordinates."""
    path = _as_path(path)
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty k-point file: {path}")

    try:
        nk = int(lines[0].split()[0])
        body = lines[1:]
    except ValueError:
        nk = len(lines)
        body = lines

    if len(body) < nk:
        raise ValueError(f"{path} declares {nk} k-points but only has {len(body)} rows")

    kpts = np.empty((nk, 3), dtype=float)
    weights = np.empty(nk, dtype=float)
    for i, line in enumerate(body[:nk]):
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid k-point row {i + 1} in {path}: {line}")
        kpts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        weights[i] = float(parts[3]) if len(parts) > 3 else 1.0
    return kpts, weights


def read_labelinfo(path):
    """Read Wannier90 *_band.labelinfo.dat if present."""
    labels = []
    path = _as_path(path)
    if not os.path.exists(path):
        return labels

    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            labels.append(
                {
                    "label": parts[0],
                    "index": int(parts[1]),
                    "distance": float(parts[2]),
                    "kx": float(parts[3]),
                    "ky": float(parts[4]),
                    "kz": float(parts[5]),
                }
            )
    return labels


def load_wannier_hr(path):
    """Load hr.dat into dense R-stack form with degeneracy normalization."""
    dim, degens, dict_h = read_wannier_hr(_as_path(path))
    if dim <= 0 or not dict_h:
        raise ValueError(f"Failed to read Wannier Hamiltonian: {path}")
    if len(degens) != len(dict_h):
        raise ValueError(
            f"Degeneracy count mismatch in {path}: "
            f"len(degens)={len(degens)}, len(R)={len(dict_h)}"
        )

    r_keys = sorted(dict_h.keys(), key=lambda r: (int(r[0]), int(r[1]), int(r[2])))
    deg_by_r = {r: float(degens[i]) for i, r in enumerate(r_keys)}
    h_r = np.stack([dict_h[r] / deg_by_r[r] for r in r_keys], axis=0)
    r_vecs = np.asarray(r_keys, dtype=float)
    return WannierHR(dim=dim, r_vecs=r_vecs, h_r=h_r)


def hamiltonian_k(wannier, kpts, batch_size=None):
    """Vectorized Fourier transform from H(R) to H(k)."""
    kpts = np.asarray(kpts, dtype=float)
    batch_size = len(kpts) if batch_size is None else int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    hk = np.empty((len(kpts), wannier.dim, wannier.dim), dtype=np.complex128)
    for start in range(0, len(kpts), batch_size):
        stop = min(start + batch_size, len(kpts))
        phase = np.exp(2j * np.pi * (kpts[start:stop] @ wannier.r_vecs.T))
        hk[start:stop] = np.einsum("kr,rij->kij", phase, wannier.h_r, optimize=True)
    return 0.5 * (hk + np.swapaxes(hk.conj(), -1, -2))


def solve_eigenvalues(wannier, kpts, batch_size=None):
    """Return sorted eigenvalues E_n(k) for all k-points."""
    hk = hamiltonian_k(wannier, kpts, batch_size=batch_size)
    evals = np.linalg.eigvalsh(hk)
    return np.real(evals)


def parse_band_pairs(pair_args):
    """Parse 1-based band pairs from CLI strings like '15,16'."""
    pairs = []
    for item in pair_args or []:
        parts = item.replace(":", ",").split(",")
        if len(parts) != 2:
            raise ValueError(f"Band pair must look like 'i,j': {item}")
        i, j = int(parts[0]), int(parts[1])
        if i <= 0 or j <= 0:
            raise ValueError(f"Band indices are 1-based and must be positive: {item}")
        pairs.append((i - 1, j - 1))
    return pairs


def parse_band_selection(text, nbands, name="bands"):
    """Parse 1-based band selection strings like '1', '1:4', or '1,3,5'."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    selected = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid {name} range: {item}")
            start, stop = int(parts[0]), int(parts[1])
            if start <= 0 or stop <= 0 or stop < start:
                raise ValueError(f"{name} ranges are 1-based inclusive: {item}")
            selected.extend(range(start - 1, stop))
        else:
            selected.append(int(item) - 1)

    if not selected:
        raise ValueError(f"No {name} selected from: {text}")
    if min(selected) < 0 or max(selected) >= nbands:
        raise ValueError(f"{name} selection {text} exceeds nbands={nbands}")
    return np.asarray(selected, dtype=int)


def _reduce_reference(values, reducer):
    if reducer == "mean":
        return float(np.mean(values))
    if reducer == "min":
        return float(np.min(values))
    if reducer == "max":
        return float(np.max(values))
    raise ValueError(f"Unknown reference reducer: {reducer}")


def compute_alignment_shift(
    evals_soc,
    evals_nosoc,
    mode,
    bottom_bands=None,
    bottom_reducer="mean",
    num_occupied=None,
):
    """Return constant shift C = E_ref_SOC - E_ref_noSOC."""
    mode = mode.lower()
    if mode == "none":
        return 0.0, {"mode": mode}

    if evals_soc.shape != evals_nosoc.shape:
        raise ValueError(
            f"Eigenvalue shape mismatch before alignment: SOC={evals_soc.shape}, "
            f"noSOC={evals_nosoc.shape}"
        )

    if mode == "bottom":
        bands = np.asarray([0] if bottom_bands is None else bottom_bands, dtype=int)
        ref_soc = _reduce_reference(evals_soc[:, bands], bottom_reducer)
        ref_nosoc = _reduce_reference(evals_nosoc[:, bands], bottom_reducer)
        return ref_soc - ref_nosoc, {
            "mode": mode,
            "bands_1based": bands + 1,
            "reducer": bottom_reducer,
            "ref_soc_eV": ref_soc,
            "ref_nosoc_eV": ref_nosoc,
        }

    if mode == "vbm":
        if num_occupied is None:
            raise ValueError("--num-occupied is required for --align vbm")
        nocc = int(num_occupied)
        if nocc <= 0 or nocc > evals_soc.shape[1]:
            raise ValueError(f"--num-occupied must be in [1, {evals_soc.shape[1]}]")
        ref_soc = float(np.max(evals_soc[:, :nocc]))
        ref_nosoc = float(np.max(evals_nosoc[:, :nocc]))
        return ref_soc - ref_nosoc, {
            "mode": mode,
            "num_occupied": nocc,
            "ref_soc_eV": ref_soc,
            "ref_nosoc_eV": ref_nosoc,
        }

    raise ValueError(f"Unknown alignment mode: {mode}")


def compare_pair_splittings(evals_soc, evals_nosoc, pairs, lambda_denom=1.0):
    """Compute SOC-induced splitting and effective lambda for band pairs."""
    evals_soc = np.asarray(evals_soc, dtype=float)
    evals_nosoc = np.asarray(evals_nosoc, dtype=float)
    if evals_soc.shape != evals_nosoc.shape:
        raise ValueError(
            f"Eigenvalue shape mismatch: SOC={evals_soc.shape}, noSOC={evals_nosoc.shape}"
        )
    if lambda_denom == 0:
        raise ValueError("lambda_denom must be non-zero")

    nk, nb = evals_soc.shape
    out = {}
    for i, j in pairs:
        if i >= nb or j >= nb:
            raise ValueError(f"Band pair {(i + 1, j + 1)} exceeds nbands={nb}")
        split_soc = evals_soc[:, j] - evals_soc[:, i]
        split_nosoc = evals_nosoc[:, j] - evals_nosoc[:, i]
        split_induced = split_soc - split_nosoc
        out[(i + 1, j + 1)] = {
            "split_soc": split_soc,
            "split_nosoc": split_nosoc,
            "split_induced": split_induced,
            "lambda_eff": split_induced / float(lambda_denom),
        }
    return out


def duplicate_spinless_evals(evals, target_nbands):
    """Duplicate spinless noSOC eigenvalues when comparing to spinful SOC."""
    if evals.shape[1] * 2 != target_nbands:
        return evals
    return np.repeat(evals, 2, axis=1)


def write_pair_csv(path, kpts, pair_results):
    """Write pair-resolved splitting data."""
    path = _as_path(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ik",
                "kx",
                "ky",
                "kz",
                "band_i",
                "band_j",
                "split_soc_eV",
                "split_nosoc_eV",
                "split_induced_eV",
                "lambda_eff_eV",
            ]
        )
        for ik, k in enumerate(kpts):
            for (i, j), data in pair_results.items():
                writer.writerow(
                    [
                        ik + 1,
                        f"{k[0]:.12g}",
                        f"{k[1]:.12g}",
                        f"{k[2]:.12g}",
                        i,
                        j,
                        f"{data['split_soc'][ik]:.12g}",
                        f"{data['split_nosoc'][ik]:.12g}",
                        f"{data['split_induced'][ik]:.12g}",
                        f"{data['lambda_eff'][ik]:.12g}",
                    ]
                )


def write_eigenvalue_csv(path, kpts, evals_soc, evals_nosoc):
    """Write full SOC/noSOC eigenvalue table."""
    path = _as_path(path)
    nb = evals_soc.shape[1]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["ik", "kx", "ky", "kz"]
        header.extend([f"E_soc_b{b + 1}_eV" for b in range(nb)])
        header.extend([f"E_nosoc_b{b + 1}_eV" for b in range(nb)])
        writer.writerow(header)
        for ik, k in enumerate(kpts):
            row = [ik + 1, f"{k[0]:.12g}", f"{k[1]:.12g}", f"{k[2]:.12g}"]
            row.extend(f"{x:.12g}" for x in evals_soc[ik])
            row.extend(f"{x:.12g}" for x in evals_nosoc[ik])
            writer.writerow(row)


def write_band_difference_csv(path, kpts, diff):
    """Write band-index resolved SOC-noSOC energy differences."""
    path = _as_path(path)
    nb = diff.shape[1]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["ik", "kx", "ky", "kz"]
        header.extend([f"dE_b{b + 1}_eV" for b in range(nb)])
        writer.writerow(header)
        for ik, k in enumerate(kpts):
            row = [ik + 1, f"{k[0]:.12g}", f"{k[1]:.12g}", f"{k[2]:.12g}"]
            row.extend(f"{x:.12g}" for x in diff[ik])
            writer.writerow(row)


def _stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values * values))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "max_abs": float(np.max(np.abs(values))),
    }


def write_band_summary_csv(path, band_diff):
    """Write per-band SOC-noSOC energy-difference statistics."""
    path = _as_path(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "band",
                "mean_dE_eV",
                "mean_abs_dE_eV",
                "rms_dE_eV",
                "min_dE_eV",
                "max_dE_eV",
                "max_abs_dE_eV",
            ]
        )
        for band in range(band_diff.shape[1]):
            stat = _stats(band_diff[:, band])
            writer.writerow(
                [
                    band + 1,
                    f"{stat['mean']:.12g}",
                    f"{stat['mean_abs']:.12g}",
                    f"{stat['rms']:.12g}",
                    f"{stat['min']:.12g}",
                    f"{stat['max']:.12g}",
                    f"{stat['max_abs']:.12g}",
                ]
            )


def write_adjacent_splitting_summary_csv(path, evals_soc_aligned, evals_nosoc):
    """Write per-adjacent-band SOC-induced splitting statistics."""
    path = _as_path(path)
    split_soc = evals_soc_aligned[:, 1:] - evals_soc_aligned[:, :-1]
    split_nosoc = evals_nosoc[:, 1:] - evals_nosoc[:, :-1]
    split_induced = split_soc - split_nosoc

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "band_i",
                "band_j",
                "mean_split_soc_eV",
                "mean_split_nosoc_eV",
                "mean_induced_split_eV",
                "mean_abs_induced_split_eV",
                "rms_induced_split_eV",
                "max_abs_induced_split_eV",
                "min_induced_split_eV",
                "max_induced_split_eV",
            ]
        )
        for i in range(split_induced.shape[1]):
            stat = _stats(split_induced[:, i])
            writer.writerow(
                [
                    i + 1,
                    i + 2,
                    f"{np.mean(split_soc[:, i]):.12g}",
                    f"{np.mean(split_nosoc[:, i]):.12g}",
                    f"{stat['mean']:.12g}",
                    f"{stat['mean_abs']:.12g}",
                    f"{stat['rms']:.12g}",
                    f"{stat['max_abs']:.12g}",
                    f"{stat['min']:.12g}",
                    f"{stat['max']:.12g}",
                ]
            )


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Compare SOC/noSOC Wannier eigenenergy splittings."
    )
    parser.add_argument("--workdir", type=str, default=None, help="Workflow root directory")
    parser.add_argument("--root", type=str, default="src/soc", help="Directory containing wsoc/wosoc")
    parser.add_argument("--wsoc-dir", type=str, default="wsoc", help="SOC-on subdirectory")
    parser.add_argument("--wosoc-dir", type=str, default="wosoc", help="SOC-off subdirectory")
    parser.add_argument("--seed", type=str, default=None, help="Wannier seed name")
    parser.add_argument("--soc-hr", type=str, default=None, help="SOC hr.dat path override")
    parser.add_argument("--nosoc-hr", type=str, default=None, help="noSOC hr.dat path override")
    parser.add_argument("--kpt", type=str, default=None, help="Wannier band kpt path override")
    parser.add_argument("--labelinfo", type=str, default=None, help="Wannier labelinfo path override")
    parser.add_argument("--pairs", nargs="*", default=[], help="1-based band pairs, e.g. 15,16 17,18")
    parser.add_argument(
        "--lambda-denom",
        type=float,
        default=1.0,
        help="L.S eigenvalue splitting denominator for lambda_eff",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of k-points per vectorized Fourier batch",
    )
    parser.add_argument(
        "--duplicate-spinless-nosoc",
        action="store_true",
        help="Duplicate noSOC bands if noSOC has half as many bands as SOC",
    )
    parser.add_argument(
        "--align",
        choices=["none", "bottom", "vbm"],
        default="none",
        help="Constant energy-zero alignment before band-index differences",
    )
    parser.add_argument(
        "--align-bands",
        type=str,
        default="1",
        help="1-based bands averaged for --align bottom, e.g. 1 or 1:4",
    )
    parser.add_argument(
        "--align-reducer",
        choices=["mean", "min", "max"],
        default="mean",
        help="Reference reducer for --align bottom over selected bands and k-points",
    )
    parser.add_argument(
        "--num-occupied",
        type=int,
        default=None,
        help="Occupied band count used to define VBM for --align vbm",
    )
    parser.add_argument("--out-prefix", type=str, default="soc_band_splitting")
    return parser


def main(argv=None):
    parser = build_argparser()
    args = parser.parse_args(argv)

    workdir = resolve_workdir(args.workdir)
    root = resolve_path(workdir, args.root)
    soc_dir = resolve_path(root, args.wsoc_dir)
    nosoc_dir = resolve_path(root, args.wosoc_dir)
    seed = args.seed or infer_seed(soc_dir, nosoc_dir)

    soc_hr = resolve_path(workdir, args.soc_hr) if args.soc_hr else os.path.join(soc_dir, f"{seed}_hr.dat")
    nosoc_hr = resolve_path(workdir, args.nosoc_hr) if args.nosoc_hr else os.path.join(nosoc_dir, f"{seed}_hr.dat")
    kpt_path = resolve_path(workdir, args.kpt) if args.kpt else os.path.join(soc_dir, f"{seed}_band.kpt")
    label_path = (
        resolve_path(workdir, args.labelinfo)
        if args.labelinfo
        else os.path.join(soc_dir, f"{seed}_band.labelinfo.dat")
    )

    kpts, _weights = read_kpoints(kpt_path)
    labels = read_labelinfo(label_path)
    soc = load_wannier_hr(soc_hr)
    nosoc = load_wannier_hr(nosoc_hr)

    print(f"Using seed: {seed}")
    print(f"Loaded k-points: {len(kpts)} from {kpt_path}")
    print(f"Loaded SOC:   dim={soc.dim}, R={len(soc.r_vecs)} from {soc_hr}")
    print(f"Loaded noSOC: dim={nosoc.dim}, R={len(nosoc.r_vecs)} from {nosoc_hr}")

    evals_soc = solve_eigenvalues(soc, kpts, batch_size=args.batch_size)
    evals_nosoc = solve_eigenvalues(nosoc, kpts, batch_size=args.batch_size)
    if args.duplicate_spinless_nosoc:
        evals_nosoc = duplicate_spinless_evals(evals_nosoc, evals_soc.shape[1])

    if evals_soc.shape != evals_nosoc.shape:
        raise ValueError(
            "SOC/noSOC eigenvalue shapes differ. Use --duplicate-spinless-nosoc "
            f"for spinless noSOC if applicable. SOC={evals_soc.shape}, noSOC={evals_nosoc.shape}"
        )

    align_bands = parse_band_selection(args.align_bands, evals_soc.shape[1], name="align bands")
    energy_shift, alignment = compute_alignment_shift(
        evals_soc,
        evals_nosoc,
        args.align,
        bottom_bands=align_bands,
        bottom_reducer=args.align_reducer,
        num_occupied=args.num_occupied,
    )
    evals_soc_aligned = evals_soc - energy_shift
    band_diff = evals_soc_aligned - evals_nosoc

    out_prefix = resolve_path(workdir, args.out_prefix)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    npz_path = f"{out_prefix}.npz"
    eig_csv_path = f"{out_prefix}_eigenvalues.csv"
    diff_csv_path = f"{out_prefix}_band_differences.csv"
    band_summary_csv_path = f"{out_prefix}_band_summary.csv"
    adjacent_summary_csv_path = f"{out_prefix}_adjacent_splitting_summary.csv"
    pair_csv_path = f"{out_prefix}_pairs.csv"

    pairs = parse_band_pairs(args.pairs)
    pair_results = compare_pair_splittings(
        evals_soc,
        evals_nosoc,
        pairs,
        lambda_denom=args.lambda_denom,
    ) if pairs else {}

    np.savez(
        npz_path,
        kpts=kpts,
        evals_soc=evals_soc,
        evals_nosoc=evals_nosoc,
        evals_soc_aligned=evals_soc_aligned,
        band_diff=band_diff,
        pair_bands=np.asarray(list(pair_results.keys()), dtype=int),
        labels=np.asarray(labels, dtype=object),
        lambda_denom=float(args.lambda_denom),
        energy_shift_soc_minus_nosoc_eV=float(energy_shift),
        alignment=np.asarray(alignment, dtype=object),
        **{
            f"pair_{i}_{j}_{name}": values
            for (i, j), data in pair_results.items()
            for name, values in data.items()
        },
    )
    write_eigenvalue_csv(eig_csv_path, kpts, evals_soc, evals_nosoc)
    write_band_difference_csv(diff_csv_path, kpts, band_diff)
    write_band_summary_csv(band_summary_csv_path, band_diff)
    write_adjacent_splitting_summary_csv(
        adjacent_summary_csv_path,
        evals_soc_aligned,
        evals_nosoc,
    )
    if pair_results:
        write_pair_csv(pair_csv_path, kpts, pair_results)

    print(f"Alignment mode: {args.align}")
    print(f"Energy shift C=E_ref_SOC-E_ref_noSOC: {energy_shift:.12g} eV")
    print(f"Saved eigenvalues: {eig_csv_path}")
    print(f"Saved band differences: {diff_csv_path}")
    print(f"Saved band summary: {band_summary_csv_path}")
    print(f"Saved adjacent splitting summary: {adjacent_summary_csv_path}")
    if pair_results:
        print(f"Saved pair splittings: {pair_csv_path}")
    print(f"Saved npz: {npz_path}")


if __name__ == "__main__":
    main()
