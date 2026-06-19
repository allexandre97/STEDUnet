#!/usr/bin/env python
"""Build display-only real-blank composite review panels and diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.calibration.visibility import build_composite_visibility_review


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--real-fiber-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    limits, rows = build_composite_visibility_review(
        args.composite_dir,
        args.artifact_dir,
        args.real_fiber_root,
        args.out,
    )
    print(
        f"Composite visibility review: {len(rows)} samples; "
        f"shared range {limits[0]:.6g}–{limits[1]:.6g}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
