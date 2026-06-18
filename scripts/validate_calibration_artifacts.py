#!/usr/bin/env python
"""Validate exploratory calibration artifacts and real-blank composites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.calibration.compositing import validate_composites
from fibras.calibration.schema import CALIBRATION_DATA_STATUSES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--composite-dir", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    if args.artifact_dir:
        summary_path = args.artifact_dir / "appearance_summary.json"
        required = [
            "real_fiber_stats.csv",
            "blank_stats.csv",
            "proxy_stats.csv",
            "background_decomposition_stats.csv",
            "appearance_summary.json",
        ]
        for name in required:
            if not (args.artifact_dir / name).exists():
                errors.append(f"missing calibration artifact: {name}")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            status = summary["metadata"].get("calibration_status")
            if status not in CALIBRATION_DATA_STATUSES:
                errors.append(f"invalid calibration_status: {status}")
            if not status.startswith("exploratory"):
                errors.append("calibration artifacts must remain exploratory in this phase")
    if args.composite_dir:
        errors.extend(validate_composites(args.composite_dir))
    if errors:
        print("Calibration artifact validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Calibration artifact validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

