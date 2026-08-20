"""Entry point for ``slw_post.x``."""

from .runner import run_stage


def main(argv=None) -> int:
    return run_stage("post", argv)


if __name__ == "__main__":
    raise SystemExit(main())
