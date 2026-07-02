#!/usr/bin/env python
"""Validate generated synthetic STED MVP samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.synthetic.storage import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    errors = validate_dataset(args.dataset_dir, num_workers=args.num_workers)
    if errors:
        print("Synthetic sample validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Synthetic sample validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
