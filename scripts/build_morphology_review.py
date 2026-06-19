#!/usr/bin/env python
"""Build morphology review panels and condition-blind spatial diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.calibration.morphology import build_morphology_review
from fibras.synthetic.schema import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--synthetic-dir", required=True, type=Path)
    parser.add_argument("--composite-dir", required=True, type=Path)
    parser.add_argument("--previous-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--diagnostics-out", required=True, type=Path)
    args = parser.parse_args()
    build_morphology_review(
        load_yaml(args.config),
        args.synthetic_dir,
        args.composite_dir,
        args.previous_dir,
        args.artifact_dir,
        args.out,
        args.diagnostics_out,
    )
    print(f"Morphology review written to {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
