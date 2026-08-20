import os
import json
import argparse
from datetime import datetime, timezone

import numpy as np

from slw.core.cli_paths import resolve_workdir, resolve_path, resolve_in_dir, resolve_out_dir
from slw.magph.legacy.input_parser import parse_input_file
from slw.magph.legacy.adapter import (
    load_djdu_npz,
    load_djr_h5,
    load_phonon_cache,
    build_legacy_djdu_tuple,
    load_j_payload,
    write_phonon_cache_from_epr,
    write_manifest,
)


def _normalize_calculation_mode(value):
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "cache": "cache",
        "reuse": "cache",
        "from_scratch": "from_scratch",
        "recompute": "from_scratch",
    }
    if mode not in aliases:
        raise ValueError("calculation_mode must be cache/reuse or from_scratch/recompute")
    return aliases[mode]


def _finite_float_argument(value):
    number = float(value)
    if not np.isfinite(number):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {value!r}")
    return number


def _select_phonon_source(calculation_mode, phonon_cache, epr_phonon):
    mode = _normalize_calculation_mode(calculation_mode)
    if mode == "from_scratch":
        if not epr_phonon:
            raise ValueError("calculation_mode=from_scratch requires epr_phonon")
        return mode, "epr", epr_phonon
    if not phonon_cache:
        raise ValueError("calculation_mode=cache requires phonon_cache")
    return mode, "cache", phonon_cache


def _bond_key_set_from_j(J0):
    return {
        (int(i), int(j), tuple(int(x) for x in r))
        for i, j, r in zip(np.asarray(J0[1]).reshape(-1), np.asarray(J0[2]).reshape(-1), J0[3])
    }


def _bond_key_set_from_dj(dj):
    pair_R = np.asarray(dj["pair_R"], dtype=np.int32).reshape(-1, 5)
    return {
        (int(row[0]), int(row[1]), (int(row[2]), int(row[3]), int(row[4])))
        for row in pair_R
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare validated magph lifetime inputs from SLW outputs (adapter stage)."
    )
    parser.add_argument("--workdir", type=str, default=None, help="Workflow root directory (default: current directory)")
    parser.add_argument("--input_file", type=str, default=None, help="Optional input.in/magph.in with preparation variables")
    parser.add_argument("--in_dir", type=str, default=None, help="Input directory containing dJ/du npz and phonon cache")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for prepared artifacts")
    parser.add_argument("--out_name", type=str, default="magph_lifetime_prep", help="Output basename prefix")
    parser.add_argument("--djdu_npz", type=str, default=None, help="Legacy real-space dJ NPZ")
    parser.add_argument("--djr", type=str, default=None, help="Real-space dJ input (dJr HDF5)")
    parser.add_argument("--djr_h5", dest="djr", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--jr", type=str, default=None, help="Real-space J input (Jr HDF5, legacy NPZ, or text)")
    parser.add_argument("--j_cache", dest="jr", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--phonon_cache", type=str, default=None, help="Phonon cache NPZ used in calculation_mode=cache")
    parser.add_argument("--epr_phonon", type=str, default=None, help="EPR HDF5 used in calculation_mode=from_scratch")
    parser.add_argument("--phonon-fc", type=str, default=None, help="Optional QE q2r force-constant file used instead of EPR IFCs")
    parser.add_argument(
        "--phonon-asr",
        choices=("none", "simple", "crystal"),
        default=None,
        help="Acoustic sum rule for --phonon-fc (default: none)",
    )
    parser.add_argument(
        "--phonon-loto",
        choices=("auto", "2d", "3d", "none"),
        default=None,
        help="LO-TO treatment for EPR Born charges (default: auto)",
    )
    parser.add_argument("--calculation_mode", choices=("cache", "reuse", "from_scratch", "recompute"), default=None)
    parser.add_argument("--phonon_qmesh", type=int, nargs=3, default=None, help="q mesh for --epr_phonon; default EPR basic_data/qc_dim")
    parser.add_argument(
        "--phonon_qshift",
        type=_finite_float_argument,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help=(
            "Uniform q-mesh shift in grid-index units for --epr_phonon; "
            "components are canonicalized modulo integers (default: 0 0 0)"
        ),
    )
    parser.add_argument("--phonon_nproc", type=int, default=None, help="Worker processes for phonon q-point diagonalization")
    parser.add_argument(
        "--phonon_cache_compressed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write phonon cache with np.savez_compressed; use --no-phonon_cache_compressed for faster writes",
    )
    parser.add_argument(
        "--rp_idx",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        help="Rp index for the FM/legacy compatibility tuple; AFM atomic-gauge vertices use every Rp",
    )
    return parser


def main():
    args = build_parser().parse_args()

    workdir = resolve_workdir(args.workdir)
    cfg = {"workdir": workdir}
    if args.input_file:
        input_path = resolve_path(workdir, args.input_file)
        cfg.update(parse_input_file(input_path))
        cfg["_input_file_path"] = input_path

    in_dir_arg = args.in_dir if args.in_dir is not None else cfg.get("in_dir")
    out_dir_arg = args.out_dir if args.out_dir is not None else cfg.get("out_dir")
    out_name = args.out_name if args.out_name != "magph_lifetime_prep" else str(cfg.get("out_name", args.out_name))
    in_dir = resolve_in_dir(workdir, in_dir_arg, default_subdir=".")
    out_dir = resolve_out_dir(workdir, out_dir_arg, default_subdir=".")
    out_prefix = os.path.join(out_dir, out_name)

    djr_h5_arg = args.djr if args.djr is not None else cfg.get("djr")
    djdu_npz_arg = args.djdu_npz if args.djdu_npz is not None else cfg.get("djdu_npz")
    j_cache_arg = args.jr if args.jr is not None else cfg.get("jr")
    phonon_cache_arg = args.phonon_cache if args.phonon_cache is not None else cfg.get("phonon_cache")
    epr_phonon_arg = args.epr_phonon if args.epr_phonon is not None else cfg.get("epr_phonon")
    phonon_fc_arg = args.phonon_fc if args.phonon_fc is not None else cfg.get("phonon_fc")
    phonon_asr = str(args.phonon_asr if args.phonon_asr is not None else cfg.get("phonon_asr", "none"))
    phonon_loto = str(args.phonon_loto if args.phonon_loto is not None else cfg.get("phonon_loto", "auto"))
    calculation_mode = _normalize_calculation_mode(
        args.calculation_mode if args.calculation_mode is not None else cfg.get("calculation_mode", "cache")
    )
    phonon_qmesh = args.phonon_qmesh if args.phonon_qmesh is not None else cfg.get("phonon_qmesh")
    phonon_qshift = (
        args.phonon_qshift
        if args.phonon_qshift is not None
        else cfg.get("phonon_qshift")
    )
    phonon_nproc = max(1, int(args.phonon_nproc if args.phonon_nproc is not None else cfg.get("phonon_nproc", 1)))
    phonon_cache_compressed = (
        bool(args.phonon_cache_compressed)
        if args.phonon_cache_compressed is not None
        else bool(cfg.get("phonon_cache_compressed", True))
    )

    djr_h5_path = resolve_path(in_dir, djr_h5_arg) if djr_h5_arg else None
    djdu_path = resolve_path(in_dir, djdu_npz_arg) if djdu_npz_arg else None
    if djdu_path is None and djr_h5_path is None:
        raise ValueError("A dJ input is required. Set djr/--djr or djdu_npz/--djdu_npz explicitly.")
    if djdu_path is not None and djr_h5_path is not None:
        raise ValueError("Specify only one dJ input: djr or djdu_npz")

    if not j_cache_arg:
        raise ValueError("A J input is required. Set jr/--jr explicitly.")
    j_path = resolve_path(in_dir, j_cache_arg)

    _, phonon_source, phonon_source_arg = _select_phonon_source(
        calculation_mode, phonon_cache_arg, epr_phonon_arg
    )
    if phonon_source == "epr":
        epr_phonon = resolve_path(in_dir, phonon_source_arg)
        phonon_fc = resolve_path(in_dir, phonon_fc_arg) if phonon_fc_arg else None
        if phonon_qmesh is None:
            raise ValueError("calculation_mode=from_scratch requires phonon_qmesh to be explicit")
        phonon_path = out_prefix + ".phonon_cache.npz"
        print(
            f"[magph] calculation_mode=from_scratch; building phonon cache from EPR: {epr_phonon} "
            f"qmesh={tuple(int(x) for x in phonon_qmesh)} "
            f"qshift={tuple(float(x) for x in (phonon_qshift or (0.0, 0.0, 0.0)))} "
            f"nproc={phonon_nproc} compressed={phonon_cache_compressed} "
            f"fc={phonon_fc or 'EPR'} asr={phonon_asr} loto={phonon_loto}",
            flush=True,
        )
        write_phonon_cache_from_epr(
            epr_phonon,
            phonon_path,
            qmesh=phonon_qmesh,
            nproc=phonon_nproc,
            compressed=phonon_cache_compressed,
            phonon_fc=phonon_fc,
            phonon_asr=phonon_asr,
            phonon_loto=phonon_loto,
            qmesh_shift_grid=phonon_qshift,
        )
    else:
        phonon_fc = None
        phonon_path = resolve_path(in_dir, phonon_source_arg)
        print(f"[magph] calculation_mode=cache; loading phonon cache: {phonon_path}", flush=True)

    use_h5 = djr_h5_path is not None
    dj = load_djr_h5(djr_h5_path) if use_h5 else load_djdu_npz(djdu_path)
    J0 = load_j_payload(j_path)
    j_keys = _bond_key_set_from_j(J0)
    dj_keys = _bond_key_set_from_dj(dj)
    if j_keys != dj_keys:
        print(
            "[magph][warn] J and dJ bond sets differ: "
            f"J_only={len(j_keys - dj_keys)} dJ_only={len(dj_keys - j_keys)}"
        )
    ph = load_phonon_cache(phonon_path)
    rp_idx_raw = args.rp_idx if args.rp_idx != [0, 0, 0] else cfg.get("rp_idx", args.rp_idx)
    rp_idx = tuple(int(x) for x in rp_idx_raw)
    dJdu_legacy = build_legacy_djdu_tuple(dj, rp_idx=rp_idx)

    j_legacy_npz = out_prefix + ".J_legacy.npz"
    legacy_npz = out_prefix + ".dJdu_legacy.npz"
    manifest_path = out_prefix + ".manifest.json"

    np.savez_compressed(
        j_legacy_npz,
        J_iso=np.array(J0[0], dtype=np.float64),
        i_idx=np.array(J0[1], dtype=np.int32),
        j_idx=np.array(J0[2], dtype=np.int32),
        vector=np.array(J0[3], dtype=np.int32),
        R=np.array(J0[4], dtype=np.float64),
    )

    if not use_h5:
        np.savez_compressed(
            legacy_npz,
            moved_atom_idx=dJdu_legacy[0],
            i_idx=dJdu_legacy[1],
            j_idx=dJdu_legacy[2],
            R=dJdu_legacy[3],
            grad_J_vec=dJdu_legacy[4],
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module": "slw.magph.legacy.reference.prepare_lifetime",
        "workdir": workdir,
        "in_dir": in_dir,
        "out_dir": out_dir,
        "out_name": out_name,
        "j_cache": j_path,
        "djdu_npz": djdu_path if djdu_path else "",
        "djr_h5": djr_h5_path if use_h5 else "",
        "phonon_cache": phonon_path,
        "phonon_fc": phonon_fc or "",
        "phonon_asr": phonon_asr,
        "phonon_loto": phonon_loto,
        "calculation_mode": calculation_mode,
        "phonon_qmesh": None if phonon_qmesh is None else [int(x) for x in phonon_qmesh],
        "phonon_qshift": (
            np.asarray(ph["q_mesh_shift_grid"], dtype=np.float64).tolist()
            if "q_mesh_shift_grid" in ph
            else None
        ),
        "phonon_nproc": int(phonon_nproc),
        "phonon_cache_compressed": bool(phonon_cache_compressed),
        "rp_idx": list(rp_idx),
        "legacy_j_npz": j_legacy_npz,
        "legacy_djdu_npz": "" if use_h5 else legacy_npz,
        "djdu_units": dj.get("units", "meV/A"),
        "n_j_bonds": int(len(J0[0])),
        "n_targets": len(dj["targets"]),
        "n_axes": len(dj["axes"]),
        "n_bonds": int(dj["pair_R"].shape[0]),
        "n_qpoints": int(ph["q_mesh_flat_frac"].shape[0]),
    }
    write_manifest(manifest_path, manifest)

    print("[magph] Preparation complete.")
    print(f"[magph] J legacy tuple: {j_legacy_npz}")
    if use_h5:
        print(f"[magph] dJr HDF5: {djr_h5_path}")
    else:
        print(f"[magph] dJdu legacy tuple: {legacy_npz}")
    print(f"[magph] manifest: {manifest_path}")
    print(
        f"[magph] J_bonds={len(J0[0])}, targets={len(dj['targets'])}, "
        f"dJ_bonds={dj['pair_R'].shape[0]}, qpoints={ph['q_mesh_flat_frac'].shape[0]}"
    )


if __name__ == "__main__":
    main()
