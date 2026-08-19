"""Small plot.in driven plotting entrypoint for slw.magph."""

from __future__ import annotations

import argparse
import sys

from slw.magph.plotting.config import PlotConfig, load_plot_config
from slw.magph.plotting.registry import dispatch_plot, available_kinds


def _print_template() -> None:
    print(
        "\n".join(
            [
                "# plot.in",
                "kind = magnon_h5",
                "input = J_epr_tensor.h5",
                "output = magnon.png",
                "kpath = G M K G L",
                "# kpath_file = FeF2.kpt",
                "S = 2.5",
                "spin_pattern = 1 -1",
                "band_points = 101",
                "j_source = auto",
                "component = iso",
                "j_prefactor = auto",
                "solver = full_bdg",
                "bond_class = sign",
                "energy_unit = meV",
                "# exclude_shells = 2",
            ]
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slw.magph.plot",
        description="Read plot.in and generate the requested magph plot.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="plot.in",
        help="plot input file. Default: plot.in",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="print a minimal plot.in template and exit",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list supported plot kinds and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.template:
        _print_template()
        return 0

    if args.list:
        for kind in available_kinds():
            print(kind)
        return 0

    config: PlotConfig = load_plot_config(args.config)
    dispatch_plot(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
