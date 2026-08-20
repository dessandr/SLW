"""Entry point for ``slw_magph.x``."""

from .runner import run_stage


def main(argv=None) -> int:
    return run_stage("magph", argv)


if __name__ == "__main__":
    raise SystemExit(main())
