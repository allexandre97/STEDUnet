#!/usr/bin/env python
"""Create provisional STED split manifests from inventory manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.sted_splits import create_splits, write_csv, write_split_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--strategy", default="pn_holdout")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    write_csv(args.out, create_splits(args.inventory_dir, args.strategy))
    if args.report:
        write_split_report(args.out, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
