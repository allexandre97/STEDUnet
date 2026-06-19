#!/usr/bin/env python
"""Validate IDs and parent inheritance across generated datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.synthetic.collection import validate_dataset_collection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    errors = validate_dataset_collection(args.dataset_dirs)
    if errors:
        print("Synthetic dataset collection validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Synthetic dataset collection validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
