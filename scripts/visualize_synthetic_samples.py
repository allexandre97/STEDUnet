#!/usr/bin/env python
"""Visualize generated synthetic STED MVP samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.synthetic.visualization import visualize_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    visualize_dataset(args.dataset_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

