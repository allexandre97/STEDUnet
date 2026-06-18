#!/usr/bin/env python
"""Generate bounded exploratory real-blank composite examples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.calibration.compositing import generate_composites
from fibras.synthetic.schema import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    generate_composites(load_yaml(args.config), args.inventory_dir, args.splits, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

