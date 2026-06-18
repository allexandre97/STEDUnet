#!/usr/bin/env python
"""Create reproducible STED source-image manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.sted_inventory import run_inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fiber-root-id", required=True)
    parser.add_argument("--fiber-dir", required=True, type=Path)
    parser.add_argument("--blank-root-id", required=True)
    parser.add_argument("--blank-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    run_inventory(args.fiber_root_id, args.fiber_dir, args.blank_root_id, args.blank_dir, args.out, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

