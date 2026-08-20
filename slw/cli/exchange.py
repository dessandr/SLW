"""Entry point for ``slw_exchange.x``."""

from .runner import run_stage


def main(argv=None) -> int:
    return run_stage("exchange", argv)


if __name__ == "__main__":
    raise SystemExit(main())
