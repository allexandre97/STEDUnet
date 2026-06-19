#!/usr/bin/env python
"""Validate the provisional blank-background pool manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.sted_blank_pools import read_csv, validate_blank_pools


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", required=True, type=Path)
    args = parser.parse_args()
    errors = validate_blank_pools(read_csv(args.pools))
    if errors:
        print("Blank-pool validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Blank-pool validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
