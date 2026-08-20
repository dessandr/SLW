"""Build a TB2J-compatible spinor hr.dat from collinear up/down files.

Atomic SOC is deliberately not configured here. Additional ``lambda L.S``
terms belong to the exchange input's ``SOC (atomic)`` card, so a spinor
Hamiltonian remains independent of whether SOC is added.
"""

from __future__ import annotations

import argparse

from slw.exchange.kernels.spinor_hr import build_spinor_hr


def build_wannier_spinor(args):
    """Run the strict collinear-to-spinor builder."""

    return build_spinor_hr(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--prefix_up", default=None)
    parser.add_argument("--prefix_dn", default=None)
    parser.add_argument("--up_hr", default=None)
    parser.add_argument("--dn_hr", default=None)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--out_prefix", default=None)
    parser.add_argument(
        "--groupby",
        required=True,
        choices=["spin", "orbital"],
        help="TB2J layout: [all up|all down] or per-orbital up/down pairs.",
    )
    parser.add_argument("--centres_up", default=None)
    parser.add_argument("--centres_dn", default=None)
    parser.add_argument("--centres_output", default=None)
    parser.add_argument(
        "--spin_direction",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
    )
    args = parser.parse_args()
    if args.prefix:
        args.prefix_up = args.prefix_up or f"{args.prefix}_up"
        args.prefix_dn = args.prefix_dn or f"{args.prefix}_dn"
        args.out_prefix = args.out_prefix or args.prefix
    build_wannier_spinor(args)


if __name__ == "__main__":
    main()


__all__ = ["build_wannier_spinor", "main"]
