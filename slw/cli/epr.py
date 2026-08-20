"""Entry point for ``slw_epr.x``."""

from .runner import run_stage


def main(argv=None) -> int:
    return run_stage("epr", argv)


if __name__ == "__main__":
    raise SystemExit(main())
