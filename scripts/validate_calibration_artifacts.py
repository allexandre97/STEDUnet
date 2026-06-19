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
    parser.add_argument("--synthetic-dir", type=Path)
    parser.add_argument("--composite-dir", type=Path)
    parser.add_argument("--pure-blank-dir", type=Path)
    parser.add_argument("--sweep-dir", type=Path)
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
            for field in [
                "source_commit_sha",
                "working_tree_dirty",
                "generation_config_sha256",
                "inventory_manifest_sha256",
                "split_manifest_sha256",
                "blank_pool_manifest_sha256",
            ]:
                if field not in summary["metadata"]:
                    errors.append(f"missing calibration provenance field: {field}")
    if args.composite_dir:
        errors.extend(validate_composites(args.composite_dir))
    if args.pure_blank_dir:
        errors.extend(validate_pure_blank_qa(args.pure_blank_dir))
    if args.sweep_dir:
        errors.extend(validate_foreground_sweep(args.sweep_dir))
    if args.synthetic_dir or args.composite_dir or args.pure_blank_dir:
        errors.extend(validate_global_ids(args.synthetic_dir, args.composite_dir, args.pure_blank_dir))
    if errors:
        print("Calibration artifact validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Calibration artifact validation passed.")
    return 0


def validate_foreground_sweep(sweep_dir: Path) -> list[str]:
    import csv

    errors: list[str] = []
    summary_path = sweep_dir / "foreground_intensity_sweep_summary.json"
    per_sample_path = sweep_dir / "foreground_intensity_sweep_per_sample.csv"
    table_path = sweep_dir / "foreground_intensity_sweep_summary.csv"
    for path in [summary_path, per_sample_path, table_path]:
        if not path.exists():
            errors.append(f"missing foreground sweep artifact: {path.name}")
    if errors:
        return errors
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["metadata"].get("calibration_status") != "exploratory_unpartitioned":
        errors.append("foreground sweep must remain exploratory_unpartitioned")
    with per_sample_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_setting: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_setting.setdefault(row["setting_id"], []).append(row)
    reference: set[tuple[str, str, str, str, str, str]] | None = None
    for setting, selected in by_setting.items():
        identities = {
            (
                row["sample_id"],
                row["parent_synthetic_sample_id"],
                row["source_blank_id"],
                row["geometry_seed"],
                row["rendering_seed"],
                row["compositing_seed"],
            )
            for row in selected
        }
        if reference is None:
            reference = identities
        elif identities != reference:
            errors.append(
                f"{setting}: geometry, blank, parent, or seed identities differ "
                "from other sweep settings"
            )
    return errors


def validate_pure_blank_qa(dataset_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest = dataset_dir / "dataset_manifest.csv"
    if not manifest.exists():
        return [f"{manifest}: missing"]
    import csv
    import numpy as np

    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        with np.load(dataset_dir / row["npz_path"], allow_pickle=False) as data:
            required = ["semantic_mask", "centerline_mask", "endpoint_map", "junction_map", "crossing_map", "render_uint8"]
            for name in required:
                if name not in data.files:
                    errors.append(f"{row['sample_id']}: missing {name}")
            for name in required[:-1]:
                if name in data.files and int(np.sum(data[name])) != 0:
                    errors.append(f"{row['sample_id']}: pure blank target {name} must be zero")
    return errors


def validate_global_ids(*dirs: Path | None) -> list[str]:
    errors: list[str] = []
    seen: dict[str, Path] = {}
    synthetic_hashes: dict[str, str] = {}
    composite_parents: list[tuple[str, str, str]] = []
    import csv

    for dataset_dir in [d for d in dirs if d]:
        manifest = dataset_dir / "dataset_manifest.csv"
        if not manifest.exists():
            continue
        with manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            sid = row["sample_id"]
            if sid in seen:
                errors.append(f"sample_id collision: {sid} appears in {seen[sid]} and {dataset_dir}")
            seen[sid] = dataset_dir
            json_path = dataset_dir / row["json_path"]
            if json_path.exists():
                meta = json.loads(json_path.read_text(encoding="utf-8"))
                if meta.get("source_blank_provenance") == "not_applicable":
                    synthetic_hashes[sid] = row.get("npz_sha256", "")
                parent = meta.get("parent_synthetic_sample_id")
                if parent and parent != "generated_in_memory":
                    composite_parents.append((sid, parent, meta.get("source_synthetic_artifact_hash", "")))
    for sid, parent, expected_hash in composite_parents:
        if parent not in synthetic_hashes:
            errors.append(f"{sid}: parent_synthetic_sample_id {parent} does not resolve to supplied synthetic manifest")
        elif expected_hash and synthetic_hashes[parent] != expected_hash:
            errors.append(f"{sid}: parent synthetic hash mismatch")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
