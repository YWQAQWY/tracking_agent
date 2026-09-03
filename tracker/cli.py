"""Allow Tracker to run as ``python -m tracker.cli``."""

from main import cli


if __name__ == "__main__":
    raise SystemExit(cli())
