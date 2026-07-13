#!/usr/bin/env python
"""Batch-evaluate real annotated STED images with the frozen first baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fibras.annotations import build_real_annotation_sample
from fibras.training.real_crops import read_real_annotation_manifest
from scripts.analyze_real_pilot_errors import (
    THRESHOLDS,
    image_intensity_stats,
    semantic_error_analysis,
    skeleton_error_analysis,
)
from scripts.evaluate_real_pilot_baseline import (
    add_uncertainty_probability_metrics,
    compute_real_pilot_metrics,
    import_torch,
    load_model,
    run_tiled_inference,
    safe_stem,
    write_outputs,
)


RECOMMENDATIONS = {
    "acceptable": "transfer looks acceptable for first baseline",
    "more_annotations": "more real annotations needed before deciding",
    "appearance": "synthetic appearance calibration likely needs improvement",
    "skeleton": "skeleton target/loss likely needs improvement",
    "fine_tune": "fine-tuning is justified",
    "clump_unknown": "clump behavior cannot be assessed because no real clump labels are present",
}

SUMMARY_COLUMNS = [
    "sample_id",
    "expected_category",
    "notes",
    "fibrous_dice",
    "fibrous_precision",
    "fibrous_recall",
    "predicted_to_target_fibrous_area_ratio",
    "false_positive_fibrous_background_pixels",
    "false_negative_fibrous_pixels",
    "clump_dice",
    "clump_target_pixels",
    "fibrous_inside_target_clump_fraction",
    "fibrous_inside_uncertain_ignore_fraction",
    "clump_inside_target_fibrous_fraction",
    "skeleton_inside_target_clump_fraction_0.75",
    "skeleton_inside_uncertain_ignore_fraction_0.75",
    "missed_thick_component_count",
    "missed_faint_component_count",
    "skeleton_dice_0.5",
    "skeleton_precision_0.5",
    "skeleton_recall_0.5",
    "skeleton_dice_0.75",
    "skeleton_precision_0.75",
    "skeleton_recall_0.75",
    "skeleton_dice_0.85",
    "skeleton_precision_0.85",
    "skeleton_recall_0.85",
    "target_skeleton_recovered_within_2px_0.5",
    "predicted_skeleton_within_2px_of_target_0.5",
    "target_skeleton_recovered_within_2px_0.75",
    "predicted_skeleton_within_2px_of_target_0.75",
    "target_skeleton_recovered_within_2px_0.85",
    "predicted_skeleton_within_2px_of_target_0.85",
    "predicted_skeleton_inside_predicted_fibrous_fraction",
    "predicted_skeleton_inside_target_fibrous_fraction",
    "intensity_mean",
    "intensity_std",
    "intensity_p50",
    "intensity_p95",
    "intensity_p99",
    "intensity_zero_fraction",
    "intensity_local_variance_mean_7x7",
    "intensity_local_variance_p95_7x7",
]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.annotation_dir is None) == (args.manifest is None):
        print("error: provide exactly one of --annotation-dir or --manifest", file=sys.stderr)
        return 2
    if args.checkpoint is None and args.run_dir is None:
        print("error: provide --checkpoint or --run-dir", file=sys.stderr)
        return 2

    try:
        records = records_from_args(args)
        torch = import_torch()
        model, checkpoint_path, device = load_model(args.checkpoint, args.run_dir, args.device, torch)
        from fibras.training.schema08_baseline import device_report
        args.device_report = device_report(device)
        print(f"device: {args.device_report}")
        rows = [
            evaluate_record(record, args.out, model, device, torch, checkpoint_path, args)
            for record in records
        ]
        write_aggregate_outputs(args.out, rows, checkpoint_path, args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"evaluated {len(records)} sample(s)")
    print(f"wrote {args.out / 'summary.csv'}")
    print(f"wrote {args.out / 'summary.json'}")
    print(f"wrote {args.out / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--partition", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/first_baseline_schema08_v0_50ep_wandb"))
    parser.add_argument("--out", type=Path, default=Path("reports/real_annotation_batch_eval"))
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skeleton-threshold", type=float, default=0.75)
    parser.add_argument("--overlap", "--tile-overlap", dest="tile_overlap", type=int, default=32)
    parser.add_argument(
        "--uncertainty-gating",
        choices=["none", "suppress_fibrous", "suppress_skeleton", "suppress_both"],
        default="none",
    )
    parser.add_argument("--uncertainty-threshold", type=float, default=0.5)
    parser.add_argument(
        "--include-image",
        action="append",
        default=[],
        help="Evaluate only matching sample_id values. May be repeated or comma-separated.",
    )
    return parser


def records_from_args(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.annotation_dir is not None:
        records = discover_annotation_triplets(args.annotation_dir)
    else:
        assert args.manifest is not None
        records = read_annotation_manifest(
            args.manifest,
            image_root=args.image_root,
            annotation_root=args.annotation_root,
        )
    if (args.fold_manifest is None) != (args.outer_fold is None):
        raise ValueError("--fold-manifest and --outer-fold must be provided together")
    if args.fold_manifest is not None:
        records = filter_records(records, fold_sample_ids(args.fold_manifest, args.outer_fold, args.partition))
    return filter_records(records, parse_include_images(args.include_image))


def fold_sample_ids(path: Path, outer_fold: int, partition: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected = [
        row["sample_id"] for row in rows
        if int(row["outer_fold"]) == outer_fold
        and row["partition"] == partition
        and row.get("validation_status", "valid") == "valid"
    ]
    if not selected:
        raise ValueError(f"fold {outer_fold} has no valid {partition} images")
    return selected


def parse_include_images(values: list[str]) -> list[str]:
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def filter_records(records: list[dict[str, str]], include_images: list[str]) -> list[dict[str, str]]:
    if not include_images:
        return records
    wanted = set(include_images)
    selected = [record for record in records if record["sample_id"] in wanted]
    missing = sorted(wanted - {record["sample_id"] for record in selected})
    if missing:
        available = ", ".join(record["sample_id"] for record in records)
        raise ValueError(f"include-image sample_id(s) not found: {missing}. Available sample_id values: {available}")
    return selected


def discover_annotation_triplets(annotation_dir: Path) -> list[dict[str, str]]:
    if not annotation_dir.exists():
        raise FileNotFoundError(f"annotation directory not found: {annotation_dir}")
    images = [
        path for path in sorted(annotation_dir.iterdir())
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in {".tif", ".tiff", ".png"}
    ]
    records = []
    for image in images:
        if image.name.endswith(("_labels.png", "_label.png", "_mask.png", "_labels.tif", "_label.tif", "_mask.tif")):
            continue
        snakes = image.with_suffix(".txt")
        labels = matching_label_path(image)
        if snakes.exists() and labels is not None:
            records.append(
                {
                    "sample_id": safe_stem(image),
                    "image_path": str(image),
                    "snakes_path": str(snakes),
                    "labels_path": str(labels),
                    "notes": "",
                    "expected_category": "",
                }
            )
    if not records:
        raise FileNotFoundError(f"no complete image/snakes/labels triplets found in {annotation_dir}")
    return records


def matching_label_path(image: Path) -> Path | None:
    candidates = [
        image.with_suffix(".labeling"),
        image.with_name(f"{image.stem}_labels.png"),
        image.with_name(f"{image.stem}_label.png"),
        image.with_name(f"{image.stem}_mask.png"),
        image.with_name(f"{image.stem}_labels.tif"),
        image.with_name(f"{image.stem}_label.tif"),
        image.with_name(f"{image.stem}_mask.tif"),
    ]
    return next((path for path in candidates if path.exists()), None)


def read_annotation_manifest(
    path: Path,
    *,
    image_root: Path | None = None,
    annotation_root: Path | None = None,
) -> list[dict[str, Any]]:
    records = read_real_annotation_manifest(
        path,
        image_root=image_root,
        annotation_root=annotation_root,
    )
    return [
        {
            "sample_id": record.sample_id,
            "image_path": str(record.image_path),
            "snakes_path": str(record.snakes_path) if record.snakes_path else None,
            "labels_path": str(record.labels_path),
            "notes": record.notes,
            "expected_category": (record.metadata or {}).get("expected_category")
            or (record.metadata or {}).get("disease", ""),
        }
        for record in records
    ]


def evaluate_record(
    record: dict[str, str],
    out_root: Path,
    model: Any,
    device: Any,
    torch: Any,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample = build_real_annotation_sample(record["image_path"], record["snakes_path"], record["labels_path"])
    image = sample["image_uint8"] if sample["image_uint8"] is not None else sample["image_float"]
    predictions = run_tiled_inference(
        model,
        image,
        device,
        torch,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        overlap=args.tile_overlap,
        uncertainty_gating=args.uncertainty_gating,
        uncertainty_threshold=args.uncertainty_threshold,
    )
    metrics = compute_real_pilot_metrics(
        sample["real_semantic_mask"],
        sample["real_skeleton_mask"],
        predictions["semantic_class_map"],
        predictions["skeleton_probability"],
        skeleton_threshold=args.skeleton_threshold,
    )
    if "uncertainty_probability" in predictions:
        add_uncertainty_probability_metrics(metrics, sample["real_semantic_mask"], predictions["uncertainty_probability"])
    sample_dir = out_root / record["sample_id"]
    metrics["metadata"] = {
        "sample_id": record["sample_id"],
        "expected_category": record.get("expected_category", ""),
        "notes": record.get("notes", ""),
        "checkpoint": str(checkpoint_path),
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "image": record["image_path"],
        "snakes": record["snakes_path"],
        "labels": record["labels_path"],
        "patch_size": int(args.patch_size),
        "batch_size": int(args.batch_size),
        "tile_overlap": int(args.tile_overlap),
        "device": str(device),
        "uncertainty_gating": args.uncertainty_gating,
        "uncertainty_threshold": float(args.uncertainty_threshold),
    }
    write_outputs(sample_dir, Path(record["image_path"]), sample, predictions, metrics)
    standardize_sample_outputs(sample_dir, Path(record["image_path"]))
    row = aggregate_sample(record, sample, predictions, metrics, sample_dir)
    write_sample_metrics(sample_dir / "metrics.json", metrics, row)
    return row


def standardize_sample_outputs(sample_dir: Path, image_path: Path) -> None:
    stem = safe_stem(image_path)
    renames = {
        sample_dir / f"{stem}_metrics.json": sample_dir / "metrics.json",
        sample_dir / f"{stem}_predictions.npz": sample_dir / "predictions.npz",
        sample_dir / f"{stem}_qa_panel.png": sample_dir / "qa_panel.png",
    }
    for src, dst in renames.items():
        if src.exists() and src != dst:
            src.replace(dst)


def write_sample_metrics(path: Path, evaluator_metrics: dict[str, Any], row: dict[str, Any]) -> None:
    payload = {
        **evaluator_metrics,
        "batch_metrics": {key: row.get(key) for key in SUMMARY_COLUMNS if key in row},
        "semantic_error_analysis": row["semantic_analysis"],
        "skeleton_threshold_analysis": row["skeleton_analysis"],
        "intensity": row["intensity"],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate_sample(
    record: dict[str, str],
    sample: dict[str, Any],
    predictions: dict[str, np.ndarray],
    metrics: dict[str, Any],
    sample_dir: Path,
) -> dict[str, Any]:
    raw = sample["image_uint8"] if sample["image_uint8"] is not None else sample["image_float"]
    semantic = semantic_error_analysis(sample["real_semantic_mask"], predictions["semantic_class_map"], raw)
    skeleton = skeleton_error_analysis(
        sample["real_semantic_mask"],
        sample["real_skeleton_mask"],
        predictions["skeleton_probability"],
    )
    intensity = image_intensity_stats(raw)
    target_fibrous = sample["real_semantic_mask"] == 1
    target_clump = sample["real_semantic_mask"] == 3
    target_uncertain = sample["real_semantic_mask"] == 255
    pred_fibrous = predictions["semantic_class_map"] == 1
    pred_clump = predictions["semantic_class_map"] == 3
    pred_skeleton = predictions["skeleton_probability"] > 0.75
    row: dict[str, Any] = {
        "sample_id": record["sample_id"],
        "expected_category": record.get("expected_category", ""),
        "notes": record.get("notes", ""),
        "sample_dir": str(sample_dir),
        "fibrous_dice": metrics["fibrous_dice"],
        "fibrous_precision": metrics["fibrous_precision"],
        "fibrous_recall": metrics["fibrous_recall"],
        "predicted_to_target_fibrous_area_ratio": semantic["predicted_to_target_fibrous_area_ratio"],
        "false_positive_fibrous_background_pixels": semantic["false_positive_fibrous_pixels_in_background"],
        "false_negative_fibrous_pixels": semantic["false_negative_fibrous_pixels"],
        "clump_dice": metrics["clump_dice"],
        "clump_target_pixels": metrics["clump_target_pixels"],
        "fibrous_inside_target_clump_fraction": mask_fraction(pred_fibrous, target_clump),
        "fibrous_inside_uncertain_ignore_fraction": mask_fraction(pred_fibrous, target_uncertain),
        "clump_inside_target_fibrous_fraction": mask_fraction(pred_clump, target_fibrous),
        "skeleton_inside_target_clump_fraction_0.75": mask_fraction(pred_skeleton, target_clump),
        "skeleton_inside_uncertain_ignore_fraction_0.75": mask_fraction(pred_skeleton, target_uncertain),
        "missed_thick_component_count": semantic["missed_thick_fiber_or_bundle_component_count"],
        "missed_faint_component_count": semantic["missed_faint_fiber_component_count"],
        "predicted_skeleton_inside_predicted_fibrous_fraction": metrics[
            "predicted_skeleton_inside_predicted_fibrous_fraction"
        ],
        "predicted_skeleton_inside_target_fibrous_fraction": metrics[
            "predicted_skeleton_inside_target_fibrous_fraction"
        ],
        "intensity_mean": intensity["mean"],
        "intensity_std": intensity["std"],
        "intensity_p50": intensity["p50"],
        "intensity_p95": intensity["p95"],
        "intensity_p99": intensity["p99"],
        "intensity_zero_fraction": intensity["zero_fraction"],
        "intensity_local_variance_mean_7x7": intensity["local_variance_mean_7x7"],
        "intensity_local_variance_p95_7x7": intensity["local_variance_p95_7x7"],
        "semantic_analysis": semantic,
        "skeleton_analysis": skeleton,
        "intensity": intensity,
    }
    for threshold, values in skeleton["thresholds"].items():
        row[f"skeleton_dice_{threshold}"] = values["dice"]
        row[f"skeleton_precision_{threshold}"] = values["precision"]
        row[f"skeleton_recall_{threshold}"] = values["recall"]
        row[f"target_skeleton_recovered_within_2px_{threshold}"] = values[
            "target_skeleton_recovered_within_2px_fraction"
        ]
        row[f"predicted_skeleton_within_2px_of_target_{threshold}"] = values[
            "predicted_skeleton_within_2px_of_target_fraction"
        ]
    return row


def mask_fraction(prediction: np.ndarray, target_region: np.ndarray) -> float | str:
    count = int(target_region.sum())
    return float((prediction & target_region).sum() / count) if count else "not_applicable"


def write_aggregate_outputs(out_root: Path, rows: list[dict[str, Any]], checkpoint_path: Path, args: argparse.Namespace) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    write_summary_csv(out_root / "summary.csv", rows)
    summary = {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "sample_count": len(rows),
        "thresholds": list(THRESHOLDS),
        "device_report": getattr(args, "device_report", None),
        "rows": rows,
        "aggregates": aggregate_rows(rows),
        "interpretation": batch_interpretation(rows),
        "threshold_note": "For real images, threshold 0.5 may be preferable for skeleton recall, while 0.75 may be more conservative.",
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_root / "report.md").write_text(markdown_report(summary), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "fibrous_dice",
        "fibrous_precision",
        "fibrous_recall",
        "predicted_to_target_fibrous_area_ratio",
        "clump_dice",
        "clump_target_pixels",
        "fibrous_inside_target_clump_fraction",
        "fibrous_inside_uncertain_ignore_fraction",
        "clump_inside_target_fibrous_fraction",
        "skeleton_inside_target_clump_fraction_0.75",
        "skeleton_inside_uncertain_ignore_fraction_0.75",
        "missed_thick_component_count",
        "missed_faint_component_count",
        "intensity_mean",
        "intensity_p95",
        "intensity_zero_fraction",
    ]
    for threshold in THRESHOLDS:
        text = str(threshold)
        keys.extend(
            [
                f"skeleton_dice_{text}",
                f"skeleton_precision_{text}",
                f"skeleton_recall_{text}",
                f"target_skeleton_recovered_within_2px_{text}",
                f"predicted_skeleton_within_2px_of_target_{text}",
            ]
        )
    return {key: mean_metric([row.get(key) for row in rows]) for key in keys}


def mean_metric(values: list[Any]) -> float | str:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return "not_applicable"
    return float(np.mean(numeric))


def batch_interpretation(rows: list[dict[str, Any]]) -> str:
    return RECOMMENDATIONS["more_annotations"]


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Real Annotation Batch Evaluation",
        "",
        f"Samples evaluated: {summary['sample_count']}",
        "",
        "This batch evaluation uses the frozen synthetic-only first baseline. It does not train, fine-tune, alter model architecture, alter synthetic generation, or change annotation protocol.",
        "",
        "Skeleton metrics are reported at thresholds 0.5, 0.75, and 0.85. For real images, 0.5 may be preferable for skeleton recall, while 0.75 may be more conservative; no final threshold is selected here.",
        "",
        "## Per-Sample Summary",
        "",
        "| Sample | Category | Fib Dice | Fib P | Fib R | Area ratio | Clump px | Skel Dice 0.5 | Skel Dice 0.75 | Skel Dice 0.85 | Target skel <=2px 0.75 | Pred skel <=2px 0.75 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['sample_id']} | {row.get('expected_category', '')} | {fmt(row['fibrous_dice'])} | "
            f"{fmt(row['fibrous_precision'])} | {fmt(row['fibrous_recall'])} | "
            f"{fmt(row['predicted_to_target_fibrous_area_ratio'])} | {row['clump_target_pixels']} | "
            f"{fmt(row['skeleton_dice_0.5'])} | {fmt(row['skeleton_dice_0.75'])} | "
            f"{fmt(row['skeleton_dice_0.85'])} | "
            f"{fmt(row['target_skeleton_recovered_within_2px_0.75'])} | "
            f"{fmt(row['predicted_skeleton_within_2px_of_target_0.75'])} |"
        )
    lines += [
        "",
        "## Batch Means",
        "",
        "| Metric | Mean |",
        "| --- | ---: |",
    ]
    for key, value in summary["aggregates"].items():
        lines.append(f"| {key} | {fmt(value)} |")
    lines += [
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
    ]
    if summary["rows"] and all(int(row.get("clump_target_pixels", 0)) == 0 for row in summary["rows"]):
        lines += [
            "Clump caveat: clump behavior cannot be assessed because no real clump labels are present.",
            "",
        ]
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
