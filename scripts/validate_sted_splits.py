#!/usr/bin/env python
"""Validate provisional STED split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.sted_splits import validate_splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    args = parser.parse_args()
    errors = validate_splits(args.inventory_dir, args.splits)
    if errors:
        print("STED split validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("STED split validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

