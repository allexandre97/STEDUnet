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
    parser.add_argument("--pure-blank-dir", type=Path)
    parser.add_argument("--allow-qa-in-calibration", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    summary_path = args.artifact_dir / "appearance_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats_config = config.get("statistics", {})
    artificial = generated_stats(args.artificial_dir, "artificial_synthetic", stats_config, args.allow_qa_in_calibration)
    composite = generated_stats(args.composite_dir, "real_blank_composite", stats_config, args.allow_qa_in_calibration)
    matched = matched_blank_relative_stats(args.composite_dir, stats_config)
    write_csv(args.artifact_dir / "artificial_synthetic_stats.csv", artificial)
    write_csv(args.artifact_dir / "real_blank_composite_stats.csv", composite)
    write_csv(args.artifact_dir / "real_blank_composite_matched_blank_delta_stats.csv", matched)
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
        "normalized_radial_power_tail_median",
        "normalized_radial_power_low_band_fraction",
        "normalized_radial_power_high_band_fraction",
        "autocorrelation_tail_median",
        "directional_power_ratio",
    ]
    summary["groups"]["artificial_synthetic"] = aggregate_numeric(artificial, fields)
    summary["groups"]["real_blank_composite"] = aggregate_numeric(composite, fields)
    summary["groups"]["real_blank_composite_delta_vs_source_blank"] = aggregate_numeric(matched, ["delta_p50", "delta_p95", "delta_p99", "delta_mean", "delta_variance", "delta_zero_fraction", "delta_local_variance_p50", "foreground_added_integrated_signal"])
    write_json(summary_path, summary)
    build_report(args.artifact_dir, args.artificial_dir, args.composite_dir, args.out, config, args.pure_blank_dir)
    return 0


def generated_stats(dataset_dir: Path, source_kind: str, stats_config: dict, allow_qa: bool = False) -> list[dict[str, str]]:
    rows = []
    with (dataset_dir / "dataset_manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    for row in manifest:
        meta = json.loads((dataset_dir / row["json_path"]).read_text(encoding="utf-8"))
        category = meta.get("scenario_category", meta.get("rendering_report", {}).get("scenario_category", "not_reported"))
        if category != "realism_calibration" and not allow_qa:
            raise ValueError(f"{row['sample_id']}: scenario_category {category} is excluded from realism calibration statistics")
        with np.load(dataset_dir / row["npz_path"], allow_pickle=False) as data:
            image = data["render_uint8"].copy()
        stat = summarize_array(image, {"sample_id": row["sample_id"], "source_kind": source_kind}, stats_config)
        stat["stable_image_id"] = row["sample_id"]
        stat["source_kind"] = source_kind
        stat["scenario_category"] = category
        stat["scenario"] = meta.get("scenario", meta.get("rendering_report", {}).get("scenario", "not_reported"))
        stat["schema_version"] = meta.get("dataset_schema_version", row.get("schema_version", "not_reported"))
        rows.append(stat)
    return rows


def matched_blank_relative_stats(dataset_dir: Path, stats_config: dict) -> list[dict[str, str]]:
    rows = []
    with (dataset_dir / "dataset_manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    for row in manifest:
        meta = json.loads((dataset_dir / row["json_path"]).read_text(encoding="utf-8"))
        with np.load(dataset_dir / row["npz_path"], allow_pickle=False) as data:
            composite = data["render_uint8"].astype(np.float32)
            blank = data["blank_float"].astype(np.float32)
            signal = data["synthetic_signal_float"].astype(np.float32)
        c = summarize_array(composite, {"sample_id": row["sample_id"], "source_kind": "real_blank_composite"}, stats_config)
        b = summarize_array(blank, {"sample_id": row["sample_id"], "source_kind": "source_blank"}, stats_config)
        record = {
            "sample_id": row["sample_id"],
            "parent_synthetic_sample_id": meta.get("parent_synthetic_sample_id", "not_reported"),
            "blank_stable_image_id": meta["source_blank_provenance"]["blank_stable_image_id"],
            "scenario_category": meta.get("scenario_category", meta.get("rendering_report", {}).get("scenario_category", "not_reported")),
            "delta_p50": f"{float(c['p50']) - float(b['p50']):.6g}",
            "delta_p95": f"{float(c['p95']) - float(b['p95']):.6g}",
            "delta_p99": f"{float(c['p99']) - float(b['p99']):.6g}",
            "delta_mean": f"{float(c['mean']) - float(b['mean']):.6g}",
            "delta_variance": f"{float(c['std']) ** 2 - float(b['std']) ** 2:.6g}",
            "delta_zero_fraction": f"{float(c['zero_fraction']) - float(b['zero_fraction']):.6g}",
            "delta_local_variance_p50": f"{float(c['local_variance_p50']) - float(b['local_variance_p50']):.6g}",
            "delta_normalized_radial_power_low_band_fraction": f"{float(c['normalized_radial_power_low_band_fraction']) - float(b['normalized_radial_power_low_band_fraction']):.6g}",
            "delta_normalized_radial_power_high_band_fraction": f"{float(c['normalized_radial_power_high_band_fraction']) - float(b['normalized_radial_power_high_band_fraction']):.6g}",
            "foreground_added_integrated_signal": f"{float(signal.sum()):.6g}",
            "clipping_fraction": f"{float(meta.get('clipping_fraction', 0.0)):.6g}",
            "saturation_fraction": f"{float(meta.get('saturation_fraction', 0.0)):.6g}",
        }
        rows.append(record)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
