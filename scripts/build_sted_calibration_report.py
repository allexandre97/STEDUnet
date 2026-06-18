#!/usr/bin/env python
"""Build an exploratory STED appearance calibration report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.calibration.reporting import build_report
from fibras.calibration.schema import write_json
from fibras.calibration.statistics import aggregate_numeric, read_manifest, summarize_array
from fibras.synthetic.schema import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--artificial-dir", required=True, type=Path)
    parser.add_argument("--composite-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    summary_path = args.artifact_dir / "appearance_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats_config = config.get("statistics", {})
    artificial = generated_stats(args.artificial_dir, "artificial_synthetic", stats_config)
    composite = generated_stats(args.composite_dir, "real_blank_composite", stats_config)
    write_csv(args.artifact_dir / "artificial_synthetic_stats.csv", artificial)
    write_csv(args.artifact_dir / "real_blank_composite_stats.csv", composite)
    fields = [
        "p50",
        "p95",
        "p99",
        "std",
        "zero_fraction",
        "saturation_fraction",
        "local_variance_p50",
        "row_variation",
        "column_variation",
        "radial_power_tail_median",
        "autocorrelation_tail_median",
        "directional_power_ratio",
    ]
    summary["groups"]["artificial_synthetic"] = aggregate_numeric(artificial, fields)
    summary["groups"]["real_blank_composite"] = aggregate_numeric(composite, fields)
    write_json(summary_path, summary)
    build_report(args.artifact_dir, args.artificial_dir, args.composite_dir, args.out, config)
    return 0


def generated_stats(dataset_dir: Path, source_kind: str, stats_config: dict) -> list[dict[str, str]]:
    rows = []
    with (dataset_dir / "dataset_manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    for row in manifest:
        with np.load(dataset_dir / row["npz_path"], allow_pickle=False) as data:
            image = data["render_uint8"].copy()
        stat = summarize_array(image, {"sample_id": row["sample_id"], "source_kind": source_kind}, stats_config)
        stat["stable_image_id"] = row["sample_id"]
        stat["source_kind"] = source_kind
        rows.append(stat)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
