"""Standalone, no-clobber phonon-cache builder for qe2pert EPR data."""

from __future__ import annotations

import argparse
import math

from slw.magph.adapter import write_phonon_cache_from_epr


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _finite_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a finite number, got {text!r}") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slw.magph.build_phonon_cache",
        description=(
            "Build a shifted uniform-q phonon cache from a qe2pert/Perturbo EPR "
            "file. Publication is atomic and refuses to replace an existing output."
        ),
    )
    parser.add_argument(
        "--epr",
        "--epr_phonon",
        dest="epr",
        required=True,
        help="Input qe2pert/Perturbo EPR HDF5",
    )
    parser.add_argument(
        "--qmesh",
        "--phonon_qmesh",
        dest="qmesh",
        type=_positive_int,
        nargs=3,
        required=True,
        metavar=("N1", "N2", "N3"),
        help="Three positive dimensions of the uniform q mesh",
    )
    parser.add_argument(
        "--qshift",
        "--phonon_qshift",
        dest="qshift",
        type=_finite_float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("SX", "SY", "SZ"),
        help="q-mesh shift in grid-index units, canonicalized modulo integers",
    )
    parser.add_argument(
        "--nproc",
        "--phonon_nproc",
        dest="nproc",
        type=_positive_int,
        default=1,
        help="Worker processes for q-point diagonalization (default: 1)",
    )
    parser.add_argument(
        "--asr",
        "--phonon_asr",
        dest="asr",
        choices=("none", "simple", "crystal"),
        default="none",
        help="Acoustic sum rule (default: none)",
    )
    parser.add_argument(
        "--loto",
        "--phonon_loto",
        dest="loto",
        choices=("auto", "2d", "3d", "none"),
        default="auto",
        help="LO-TO treatment (default: auto)",
    )
    parser.add_argument(
        "--phonon-fc",
        default=None,
        help="Optional QE q2r force-constant file used instead of EPR IFCs",
    )
    parser.add_argument(
        "--compressed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use compressed NPZ output (default: compressed)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="New output NPZ; an existing path is never overwritten",
    )
    return parser


def main(argv=None) -> str:
    args = build_parser().parse_args(argv)
    print(
        "[magph-phonon] requested cache: "
        f"epr={args.epr} qmesh={tuple(args.qmesh)} qshift={tuple(args.qshift)} "
        f"nproc={args.nproc} asr={args.asr} loto={args.loto} output={args.output}",
        flush=True,
    )
    written = write_phonon_cache_from_epr(
        args.epr,
        args.output,
        qmesh=args.qmesh,
        qmesh_shift_grid=args.qshift,
        nproc=args.nproc,
        compressed=args.compressed,
        phonon_fc=args.phonon_fc,
        phonon_asr=args.asr,
        phonon_loto=args.loto,
        verbose=True,
    )
    print(f"[magph-phonon] wrote {written}", flush=True)
    return written


if __name__ == "__main__":
    main()
