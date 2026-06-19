#!/usr/bin/env python
"""Run the bounded exploratory foreground-intensity sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.calibration.intensity_sweep import run_foreground_intensity_sweep
from fibras.synthetic.schema import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--sweep-config", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--composite-dir", required=True, type=Path)
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_foreground_intensity_sweep(
        load_yaml(args.base_config),
        load_yaml(args.sweep_config),
        args.artifact_dir,
        args.composite_dir,
        args.inventory_dir,
        args.splits,
        args.out,
        args.report_dir,
    )
    print(f"Foreground-intensity sweep: {len(result['settings'])} settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
