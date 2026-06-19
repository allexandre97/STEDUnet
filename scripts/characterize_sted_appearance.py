#!/usr/bin/env python
"""Characterize real STED fiber images and expert-validated blanks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.calibration.backgrounds import decomposition_stats
from fibras.calibration.proxies import proxy_measurements
from fibras.calibration.schema import artifact_metadata, write_json
from fibras.calibration.statistics import aggregate_numeric, deterministic_subset, read_manifest, source_path, summarize_image
from fibras.sted_inventory import read_image
from fibras.synthetic.schema import load_yaml


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    status = config.get("calibration_data_status", "exploratory_unpartitioned")
    source_roots = config["source_roots"]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    images = read_manifest(args.inventory_dir / "sted_images.csv")
    blanks = read_manifest(args.inventory_dir / "sted_blanks.csv")
    heavy_count = int(config.get("heavy_analysis_max_images_per_kind", 48))
    fiber_subset = deterministic_subset(images, heavy_count)
    blank_subset = deterministic_subset(blanks, heavy_count)
    stats_config = config.get("statistics", {})
    proxy_config = config.get("proxies", {})
    fiber_stats = [summarize_image(source_path(row, source_roots), row, stats_config) for row in fiber_subset]
    blank_stats = [summarize_image(source_path(row, source_roots), row, stats_config) for row in blank_subset]
    proxy_rows = []
    for row in fiber_subset + blank_subset:
        image, _, _ = read_image(source_path(row, source_roots))
        proxy = proxy_measurements(image, proxy_config)
        proxy.update(
            {
                "stable_image_id": row["stable_image_id"],
                "source_kind": row["source_kind"],
                "relative_path": row["relative_path"],
                "culture_id": row.get("culture_id", "unknown"),
                "disease": row.get("disease", "unknown"),
                "tau_isoform": row.get("tau_isoform", "unknown"),
                "experimental_condition": row.get("experimental_condition", "unknown"),
                "div": row.get("div", "unknown"),
                "div_token": row.get("div_token", "unknown"),
                "experimental_group_id": row.get("experimental_group_id", "unknown"),
            }
        )
        proxy_rows.append(proxy)
    decomp_rows = []
    sigma_values = [float(v) for v in config.get("background_decomposition", {}).get("sigma_px_values", [8, 32, 64])]
    for row in blank_subset:
        image, _, _ = read_image(source_path(row, source_roots))
        for drow in decomposition_stats(
            image,
            sigma_values,
            int(stats_config.get("spectrum_size_px", 256)),
            int(stats_config.get("spectrum_bins", 32)),
        ):
            drow.update(
                {
                    "stable_image_id": row["stable_image_id"],
                    "source_kind": row["source_kind"],
                    "relative_path": row["relative_path"],
                    "culture_id": row.get("culture_id", "unknown"),
                    "disease": row.get("disease", "unknown"),
                    "tau_isoform": row.get("tau_isoform", "unknown"),
                    "experimental_condition": row.get("experimental_condition", "unknown"),
                    "div": row.get("div", "unknown"),
                    "div_token": row.get("div_token", "unknown"),
                    "experimental_group_id": row.get("experimental_group_id", "unknown"),
                }
            )
            decomp_rows.append(drow)
    write_csv(out / "real_fiber_stats.csv", fiber_stats)
    write_csv(out / "blank_stats.csv", blank_stats)
    write_csv(out / "proxy_stats.csv", proxy_rows)
    write_csv(out / "background_decomposition_stats.csv", decomp_rows)
    metadata = artifact_metadata(
        inventory_dir=args.inventory_dir,
        splits_path=args.splits,
        config=config,
        source_image_ids=[r["stable_image_id"] for r in fiber_subset + blank_subset],
        calibration_data_status=status,
        artifact_kind="appearance_characterization",
    )
    summary = {
        "metadata": metadata,
        "groups": {
            "real_fiber": aggregate_numeric(fiber_stats, comparison_fields()),
            "blank": aggregate_numeric(blank_stats, comparison_fields()),
        },
        "proxy_metadata": {
            "warning": "proxy estimates are not ground truth",
            "estimators": {
                "foreground_occupancy_proxy": "MAD/percentile threshold occupancy",
                "ridge_response_distribution": "central-difference gradient magnitude",
                "orientation_distribution": "gradient-normal orientation on high ridge proxy pixels",
                "apparent_width_proxy": "thresholded component minor bounding-box dimension",
                "component_length_proxy": "thresholded component major bounding-box dimension",
                "endpoint_crossing_density": "thresholded connected-component proxy counts",
            },
            "configuration": proxy_config,
        },
        "representative_selection": {
            "method": "deterministic stable_image_id sorted linspace subset",
            "heavy_analysis_max_images_per_kind": heavy_count,
            "real_fiber_ids": [r["stable_image_id"] for r in fiber_subset],
            "blank_ids": [r["stable_image_id"] for r in blank_subset],
        },
        "calibration_parameter_classes": config.get("calibration_parameter_classes", {}),
        "biological_independence_warning": "Current splits remain provisional; no biological independence or final held-out performance claim is made.",
    }
    write_json(out / "appearance_summary.json", summary)
    return 0


def comparison_fields() -> list[str]:
    return [
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


if __name__ == "__main__":
    raise SystemExit(main())
