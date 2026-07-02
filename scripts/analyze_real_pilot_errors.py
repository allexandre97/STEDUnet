#!/usr/bin/env python
"""Build a focused error-analysis report for real-pilot baseline outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    find_objects,
    label,
    uniform_filter,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fibras.annotations import build_real_annotation_sample
from scripts.evaluate_real_pilot_baseline import (
    NOT_APPLICABLE,
    dice,
    precision,
    recall,
    safe_stem,
    target_positive_dice,
    target_region_masks,
)


THRESHOLDS = (0.5, 0.75, 0.85)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.batch:
            summary = analyze_batch_outputs(
                args.eval_dir, args.out, args.synthetic_manifest, args.max_synthetic_samples
            )
        else:
            summary = analyze_outputs(args.eval_dir, args.synthetic_manifest, args.max_synthetic_samples)
            write_report(args.out, summary)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out / 'report.md'}")
    print(f"wrote {args.out / 'summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=Path("reports/real_pilot_baseline_eval"))
    parser.add_argument("--out", type=Path, default=Path("reports/real_pilot_baseline_eval/error_analysis"))
    parser.add_argument("--synthetic-manifest", type=Path, default=Path("data_manifests/training_v0_schema08.csv"))
    parser.add_argument("--max-synthetic-samples", type=int, default=64)
    parser.add_argument("--batch", action="store_true", help="Analyze each immediate sample subdirectory under --eval-dir.")
    return parser


def analyze_outputs(eval_dir: Path, synthetic_manifest: Path, max_synthetic_samples: int = 64) -> dict[str, Any]:
    pilots = []
    for metrics_path, predictions_path, output_dir in discover_eval_outputs(eval_dir):
        pilots.append(analyze_pilot(metrics_path, predictions_path, output_dir))
    if not pilots:
        raise FileNotFoundError(f"no metrics/predictions outputs found in {eval_dir}")
    return {
        "pilots": pilots,
        "domain_gap": {
            "real": {pilot["pilot_id"]: pilot["intensity"] for pilot in pilots},
            "synthetic_validation": synthetic_intensity_summary(synthetic_manifest, max_synthetic_samples),
        },
        "recommendation": "collect more real annotations before deciding",
    }


def analyze_batch_outputs(
    eval_dir: Path,
    out_dir: Path,
    synthetic_manifest: Path,
    max_synthetic_samples: int = 64,
) -> dict[str, Any]:
    children = [
        child
        for child in sorted(eval_dir.iterdir())
        if child.is_dir() and discover_eval_outputs(child)
    ]
    if not children:
        raise FileNotFoundError(f"no sample subdirectories with metrics/predictions found in {eval_dir}")
    pilots = []
    for child in children:
        sample_summary = analyze_outputs(child, synthetic_manifest, max_synthetic_samples)
        sample_out = out_dir / child.name
        write_report(sample_out, sample_summary)
        pilots.extend(sample_summary["pilots"])
    summary = {
        "pilots": pilots,
        "domain_gap": {
            "real": {pilot["pilot_id"]: pilot["intensity"] for pilot in pilots},
            "synthetic_validation": synthetic_intensity_summary(synthetic_manifest, max_synthetic_samples),
        },
        "recommendation": "collect more real annotations before deciding",
    }
    write_report(out_dir, summary)
    return summary


def discover_eval_outputs(eval_dir: Path) -> list[tuple[Path, Path, Path]]:
    outputs: list[tuple[Path, Path, Path]] = []
    canonical_metrics = eval_dir / "metrics.json"
    canonical_predictions = eval_dir / "predictions.npz"
    if canonical_metrics.exists() and canonical_predictions.exists():
        return [(canonical_metrics, canonical_predictions, eval_dir)]
    for metrics_path in sorted(eval_dir.glob("*_metrics.json")):
        stem = metrics_path.name.removesuffix("_metrics.json")
        pred_path = eval_dir / f"{stem}_predictions.npz"
        if pred_path.exists():
            outputs.append((metrics_path, pred_path, eval_dir))
    return outputs


def analyze_pilot(metrics_path: Path, pred_path: Path, output_dir: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = metrics["metadata"]
    if not pred_path.exists():
        raise FileNotFoundError(f"missing prediction arrays for {metrics_path}: {pred_path}")
    with np.load(pred_path, allow_pickle=False) as data:
        pred = {name: data[name].copy() for name in data.files}

    sample = build_real_annotation_sample(metadata["image"], metadata["snakes"], metadata["labels"])
    raw = sample["image_uint8"] if sample["image_uint8"] is not None else normalize_uint8(sample["image_float"])
    valid = sample["real_semantic_mask"] != 255
    semantic = semantic_error_analysis(
        sample["real_semantic_mask"],
        pred["semantic_class_map"],
        raw,
    )
    skeleton = skeleton_error_analysis(
        sample["real_semantic_mask"],
        sample["real_skeleton_mask"],
        pred["skeleton_probability"],
        pred["semantic_class_map"],
    )
    leakage = leakage_diagnostics(sample["real_semantic_mask"], pred)
    pilot = {
        "pilot_id": pilot_id(metadata["image"]),
        "image": metadata["image"],
        "metrics_path": str(metrics_path),
        "predictions_path": str(pred_path),
        "semantic": semantic,
        "skeleton": skeleton,
        "leakage": leakage,
        "intensity": image_intensity_stats(raw),
        "valid_pixel_count": int(valid.sum()),
    }
    stem = output_stem(metrics_path, metadata)
    write_error_panel(output_dir / "error_analysis" / f"{stem}_error_panel.png", sample, pred, semantic, skeleton)
    write_leakage_panel(output_dir / "error_analysis" / f"{stem}_leakage_panel.png", sample, pred)
    return pilot


def output_stem(metrics_path: Path, metadata: dict[str, Any]) -> str:
    if metrics_path.name == "metrics.json":
        return str(metadata.get("sample_id") or metrics_path.parent.name)
    return metrics_path.name.removesuffix("_metrics.json")


def pilot_id(image_path: str) -> str:
    name = Path(image_path).name
    return name.split(" ")[0]


def semantic_error_analysis(target_semantic: np.ndarray, pred_semantic: np.ndarray, image: np.ndarray) -> dict[str, Any]:
    valid = target_semantic != 255
    target = (target_semantic == 1) & valid
    pred = (pred_semantic == 1) & valid
    background = (target_semantic == 0) & valid
    fp = pred & background
    fn = target & ~pred
    target_fib_intensity = image[target]
    faint_cutoff = float(np.percentile(target_fib_intensity, 50)) if target_fib_intensity.size else None
    fp_components = component_summary(fp, image)
    fn_components = component_summary(fn, image)
    return {
        "target_fibrous_area_fraction": fraction(target.sum(), valid.sum()),
        "predicted_fibrous_area_fraction": fraction(pred.sum(), valid.sum()),
        "predicted_to_target_fibrous_area_ratio": ratio_or_na(pred.sum(), target.sum()),
        "false_positive_fibrous_pixels_in_background": int(fp.sum()),
        "false_negative_fibrous_pixels": int(fn.sum()),
        "false_positive_components": fp_components,
        "false_negative_components": fn_components,
        "large_false_positive_count": int(sum(area >= 100 for area in fp_components["areas"])),
        "tiny_speckle_false_positive_count": int(sum(area <= 8 for area in fp_components["areas"])),
        "missed_faint_fiber_component_count": count_low_intensity_components(fn_components, faint_cutoff),
        "missed_thick_fiber_or_bundle_component_count": int(sum(area >= 100 for area in fn_components["areas"])),
        "boundary_disagreement": {
            "false_positive_fraction_within_2px_of_target_fibrous": within_mask_distance_fraction(fp, target, 2.0),
            "false_negative_fraction_within_2px_of_predicted_fibrous": within_mask_distance_fraction(fn, pred, 2.0),
        },
    }


def skeleton_error_analysis(
    target_semantic: np.ndarray,
    target_skeleton: np.ndarray,
    skeleton_probability: np.ndarray,
    pred_semantic: np.ndarray | None = None,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict[str, Any]:
    valid = target_semantic != 255
    target_fibrous = (target_semantic == 1) & valid
    target = (target_skeleton > 0) & valid
    return {
        "thresholds": {
            str(threshold): skeleton_threshold_metrics(target, target_fibrous, skeleton_probability, valid, threshold)
            for threshold in thresholds
        }
    }


def skeleton_threshold_metrics(
    target_skeleton: np.ndarray,
    target_fibrous: np.ndarray,
    skeleton_probability: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    pred = (skeleton_probability > threshold) & valid
    target = target_skeleton & valid
    component = skeleton_component_summary(target, pred)
    inside = int((pred & target_fibrous).sum())
    outside = int((pred & ~target_fibrous & valid).sum())
    return {
        "dice": target_positive_dice(pred, target),
        "precision": precision(pred, target),
        "recall": recall(pred, target),
        "target_skeleton_recovered_within_2px_fraction": within_mask_distance_fraction(target, pred, 2.0),
        "predicted_skeleton_within_2px_of_target_fraction": within_mask_distance_fraction(pred, target, 2.0),
        "predicted_skeleton_pixels_inside_target_fibrous": inside,
        "predicted_skeleton_pixels_outside_target_fibrous": outside,
        "target_snake_components_with_prediction_within_2px": component["target_components_with_prediction_within_2px"],
        "target_snake_components_without_prediction_nearby": component["target_components_without_prediction_nearby"],
        "predicted_skeleton_components_without_target_nearby": component["predicted_components_without_target_nearby"],
        "target_skeleton_component_count": component["target_component_count"],
        "predicted_skeleton_component_count": component["predicted_component_count"],
    }


def leakage_diagnostics(target_semantic: np.ndarray, pred: dict[str, np.ndarray]) -> dict[str, Any]:
    pred_semantic = pred["semantic_class_map"]
    return {
        "semantic": semantic_leakage_analysis(target_semantic, pred_semantic),
        "skeleton": {
            str(threshold): skeleton_leakage_analysis(
                target_semantic,
                pred_semantic,
                pred["skeleton_probability"] > threshold,
            )
            for threshold in THRESHOLDS
        },
        "probability_summaries": probability_summaries_by_target_region(
            target_semantic,
            {
                "fibrous_probability": pred["fibrous_probability"],
                "clump_probability": pred["clump_probability"],
                "skeleton_probability": pred["skeleton_probability"],
            },
        ),
        "interpretation_flags": leakage_interpretation_flags(target_semantic, pred),
    }


def semantic_leakage_analysis(target_semantic: np.ndarray, pred_semantic: np.ndarray) -> dict[str, Any]:
    regions = target_region_masks(target_semantic)
    return {
        "target_clump_pixels": int(regions["target_clump"].sum()),
        "target_uncertain_ignore_pixels": int(regions["target_uncertain_ignore"].sum()),
        "predicted_fibrous": counts_and_fractions_by_region(pred_semantic == 1, regions),
        "predicted_clump": counts_by_region(pred_semantic == 3, regions),
    }


def skeleton_leakage_analysis(
    target_semantic: np.ndarray,
    pred_semantic: np.ndarray,
    pred_skeleton: np.ndarray,
) -> dict[str, Any]:
    regions = target_region_masks(target_semantic)
    pred_skeleton = pred_skeleton.astype(bool, copy=False)
    out = counts_and_fractions_by_region(pred_skeleton, regions)
    out.update(
        {
            "inside_predicted_fibrous_pixels": int((pred_skeleton & (pred_semantic == 1)).sum()),
            "inside_predicted_clump_pixels": int((pred_skeleton & (pred_semantic == 3)).sum()),
            "outside_predicted_foreground_pixels": int((pred_skeleton & ~((pred_semantic == 1) | (pred_semantic == 3))).sum()),
        }
    )
    return out


def counts_by_region(mask: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, int]:
    return {f"inside_{name}_pixels": int((mask & region).sum()) for name, region in regions.items()}


def counts_and_fractions_by_region(mask: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, Any]:
    total = int(mask.sum())
    out: dict[str, Any] = {"total_pixels": total}
    for name, region in regions.items():
        count = int((mask & region).sum())
        out[f"inside_{name}_pixels"] = count
        out[f"inside_{name}_fraction"] = fraction(count, total)
    return out


def probability_summaries_by_target_region(
    target_semantic: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, Any]]]:
    regions = target_region_masks(target_semantic)
    return {
        region_name: {
            prob_name: probability_summary(probability[mask])
            for prob_name, probability in probabilities.items()
        }
        for region_name, mask in regions.items()
    }


def probability_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"mean": NOT_APPLICABLE, "p95": NOT_APPLICABLE}
    return {"mean": float(values.mean()), "p95": float(np.percentile(values, 95))}


def leakage_interpretation_flags(target_semantic: np.ndarray, pred: dict[str, np.ndarray]) -> list[str]:
    semantic = semantic_leakage_analysis(target_semantic, pred["semantic_class_map"])
    skeleton = skeleton_leakage_analysis(
        target_semantic,
        pred["semantic_class_map"],
        pred["skeleton_probability"] > 0.75,
    )
    flags = []
    fib = semantic["predicted_fibrous"]
    pred_clump = semantic["predicted_clump"]
    if fib["inside_target_clump_pixels"]:
        flags.append("fibrous leakage into expert clump")
    if fib["inside_target_uncertain_ignore_pixels"]:
        flags.append("fibrous leakage into expert uncertain_ignore")
    if fib["inside_target_background_pixels"]:
        flags.append("diffuse background false positives")
    if pred_clump["inside_target_fibrous_pixels"]:
        flags.append("clump-vs-fibrous confusion")
    if skeleton["inside_target_clump_pixels"]:
        flags.append("skeleton leakage into clump")
    if skeleton["inside_target_uncertain_ignore_pixels"]:
        flags.append("skeleton leakage into uncertain_ignore")
    return flags


def skeleton_component_summary(target: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    target_labels, target_count = label(target)
    pred_labels, pred_count = label(pred)
    distance_to_pred = distance_transform_edt(~pred) if pred.any() else np.full(target.shape, np.inf)
    distance_to_target = distance_transform_edt(~target) if target.any() else np.full(target.shape, np.inf)
    recovered = components_touching_distance(target_labels, target_count, distance_to_pred, 2.0)
    unmatched_pred = pred_count - components_touching_distance(pred_labels, pred_count, distance_to_target, 2.0)
    return {
        "target_component_count": int(target_count),
        "predicted_component_count": int(pred_count),
        "target_components_with_prediction_within_2px": int(recovered),
        "target_components_without_prediction_nearby": int(target_count - recovered),
        "predicted_components_without_target_nearby": int(unmatched_pred),
    }


def components_touching_distance(labels: np.ndarray, count: int, distances: np.ndarray, max_distance: float) -> int:
    touched = 0
    for component_id in range(1, count + 1):
        component = labels == component_id
        if component.any() and np.any(distances[component] <= max_distance):
            touched += 1
    return touched


def component_summary(mask: np.ndarray, image: np.ndarray | None = None) -> dict[str, Any]:
    labels, count = label(mask)
    areas: list[int] = []
    means: list[float] = []
    boxes: list[list[int]] = []
    for component_id, slc in enumerate(find_objects(labels), start=1):
        if slc is None:
            continue
        component = labels[slc] == component_id
        area = int(component.sum())
        areas.append(area)
        y_slice, x_slice = slc
        boxes.append([int(y_slice.start), int(x_slice.start), int(y_slice.stop), int(x_slice.stop)])
        if image is not None and area:
            means.append(float(np.asarray(image)[slc][component].mean()))
    return {
        "count": int(count),
        "areas": areas,
        "area_distribution": distribution(areas),
        "mean_intensities": means,
        "mean_intensity_distribution": distribution(means),
        "largest_box_yx": largest_box(areas, boxes),
    }


def largest_box(areas: list[int], boxes: list[list[int]]) -> list[int] | None:
    if not areas:
        return None
    return boxes[int(np.argmax(areas))]


def distribution(values: list[float] | list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": NOT_APPLICABLE, "p50": NOT_APPLICABLE, "p95": NOT_APPLICABLE, "max": NOT_APPLICABLE}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def count_low_intensity_components(summary: dict[str, Any], cutoff: float | None) -> int | str:
    if cutoff is None:
        return NOT_APPLICABLE
    return int(sum(mean <= cutoff for mean in summary["mean_intensities"]))


def within_mask_distance_fraction(source: np.ndarray, target: np.ndarray, max_distance: float) -> float | str:
    source = source.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    source_count = int(source.sum())
    if source_count == 0:
        return NOT_APPLICABLE
    if not target.any():
        return 0.0
    distances = distance_transform_edt(~target)
    return float(np.count_nonzero(distances[source] <= max_distance) / source_count)


def image_intensity_stats(image: np.ndarray) -> dict[str, float]:
    arr = np.asarray(image, dtype=np.float32)
    local_mean = uniform_filter(arr, size=7)
    local_sq_mean = uniform_filter(arr * arr, size=7)
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "zero_fraction": float(np.count_nonzero(arr == 0) / arr.size),
        "local_variance_mean_7x7": float(local_var.mean()),
        "local_variance_p95_7x7": float(np.percentile(local_var, 95)),
    }


def synthetic_intensity_summary(manifest: Path, max_samples: int) -> dict[str, Any]:
    if not manifest.exists():
        return {"status": "not_available", "reason": f"manifest not found: {manifest}"}
    rows = read_manifest_rows(manifest)
    values = []
    local_vars = []
    sample_count = 0
    missing = 0
    for row in rows:
        if row.get("split") != "validation":
            continue
        path = resolve_manifest_path(row["npz_path"], manifest)
        if not path.exists():
            missing += 1
            continue
        with np.load(path, allow_pickle=False) as data:
            image = data["render_uint8"].astype(np.float32)
        values.append(image.reshape(-1))
        local_mean = uniform_filter(image, size=7)
        local_sq_mean = uniform_filter(image * image, size=7)
        local_vars.append(np.maximum(local_sq_mean - local_mean * local_mean, 0).reshape(-1))
        sample_count += 1
        if sample_count >= max_samples:
            break
    if not values:
        return {"status": "not_available", "reason": "no readable validation render_uint8 arrays", "missing_files": missing}
    arr = np.concatenate(values)
    lv = np.concatenate(local_vars)
    return {
        "status": "available",
        "sample_count": sample_count,
        "missing_files": missing,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "zero_fraction": float(np.count_nonzero(arr == 0) / arr.size),
        "local_variance_mean_7x7": float(lv.mean()),
        "local_variance_p95_7x7": float(np.percentile(lv, 95)),
    }


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_manifest_path(raw: str, manifest: Path) -> Path:
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest.parent / path
    return candidate if candidate.exists() else path


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(markdown_report(summary), encoding="utf-8")


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Real-Pilot First-Baseline Error Analysis",
        "",
        "This report analyzes existing synthetic-only baseline predictions on the two real expert-annotated pilot images. It does not change training, architecture, synthetic generation, schema, or annotation protocol.",
        "",
        "## Semantic Errors",
        "",
        "| Pilot | Target fibrous frac | Pred fibrous frac | Pred/target | FP background px | FN fibrous px | FP comps | Large FP | Tiny FP | FN comps | Missed faint | Missed thick |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pilot in summary["pilots"]:
        sem = pilot["semantic"]
        lines.append(
            f"| {pilot['pilot_id']} | {fmt(sem['target_fibrous_area_fraction'])} | "
            f"{fmt(sem['predicted_fibrous_area_fraction'])} | {fmt(sem['predicted_to_target_fibrous_area_ratio'])} | "
            f"{sem['false_positive_fibrous_pixels_in_background']} | {sem['false_negative_fibrous_pixels']} | "
            f"{sem['false_positive_components']['count']} | {sem['large_false_positive_count']} | "
            f"{sem['tiny_speckle_false_positive_count']} | {sem['false_negative_components']['count']} | "
            f"{sem['missed_faint_fiber_component_count']} | {sem['missed_thick_fiber_or_bundle_component_count']} |"
        )
    lines += [
        "",
        "Boundary-disagreement fractions estimate how much FP/FN area lies within 2 px of the opposite fibrous mask.",
        "",
        "| Pilot | FP within 2 px of target | FN within 2 px of prediction | FP area p50/p95/max | FN area p50/p95/max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for pilot in summary["pilots"]:
        sem = pilot["semantic"]
        bd = sem["boundary_disagreement"]
        fp = sem["false_positive_components"]["area_distribution"]
        fn = sem["false_negative_components"]["area_distribution"]
        lines.append(
            f"| {pilot['pilot_id']} | {fmt(bd['false_positive_fraction_within_2px_of_target_fibrous'])} | "
            f"{fmt(bd['false_negative_fraction_within_2px_of_predicted_fibrous'])} | "
            f"{fmt(fp['p50'])}/{fmt(fp['p95'])}/{fmt(fp['max'])} | "
            f"{fmt(fn['p50'])}/{fmt(fn['p95'])}/{fmt(fn['max'])} |"
        )
    lines += [
        "",
        "## Skeleton Threshold Sensitivity",
        "",
        "| Pilot | Threshold | Dice | Precision | Recall | Target recovered <=2 px | Pred within <=2 px | Pred px inside target fibrous | Pred px outside target fibrous | Missed target comps | Unmatched pred comps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pilot in summary["pilots"]:
        for threshold, metrics in pilot["skeleton"]["thresholds"].items():
            lines.append(
                f"| {pilot['pilot_id']} | {threshold} | {fmt(metrics['dice'])} | {fmt(metrics['precision'])} | "
                f"{fmt(metrics['recall'])} | {fmt(metrics['target_skeleton_recovered_within_2px_fraction'])} | "
                f"{fmt(metrics['predicted_skeleton_within_2px_of_target_fraction'])} | "
                f"{metrics['predicted_skeleton_pixels_inside_target_fibrous']} | "
                f"{metrics['predicted_skeleton_pixels_outside_target_fibrous']} | "
                f"{metrics['target_snake_components_without_prediction_nearby']} | "
                f"{metrics['predicted_skeleton_components_without_target_nearby']} |"
            )
    lines += [
        "",
        "## Clump And Ignore Leakage",
        "",
        "| Pilot | Pred fib in target fib | Pred fib in clump | Pred fib in ignore | Pred fib in background | Fib frac clump | Fib frac ignore | Pred clump in clump | Pred clump in fibrous | Pred clump in ignore | Pred clump in background |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pilot in summary["pilots"]:
        sem = pilot["leakage"]["semantic"]
        fib = sem["predicted_fibrous"]
        clump = sem["predicted_clump"]
        lines.append(
            f"| {pilot['pilot_id']} | {fib['inside_target_fibrous_pixels']} | "
            f"{fib['inside_target_clump_pixels']} | {fib['inside_target_uncertain_ignore_pixels']} | "
            f"{fib['inside_target_background_pixels']} | {fmt(fib['inside_target_clump_fraction'])} | "
            f"{fmt(fib['inside_target_uncertain_ignore_fraction'])} | {clump['inside_target_clump_pixels']} | "
            f"{clump['inside_target_fibrous_pixels']} | {clump['inside_target_uncertain_ignore_pixels']} | "
            f"{clump['inside_target_background_pixels']} |"
        )
    lines += [
        "",
        "| Pilot | Threshold | Skel in fibrous | Skel in clump | Skel in ignore | Skel in background | Frac clump | Frac ignore | Skel in pred fib | Skel in pred clump | Skel outside pred foreground |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pilot in summary["pilots"]:
        for threshold, skel in pilot["leakage"]["skeleton"].items():
            lines.append(
                f"| {pilot['pilot_id']} | {threshold} | {skel['inside_target_fibrous_pixels']} | "
                f"{skel['inside_target_clump_pixels']} | {skel['inside_target_uncertain_ignore_pixels']} | "
                f"{skel['inside_target_background_pixels']} | {fmt(skel['inside_target_clump_fraction'])} | "
                f"{fmt(skel['inside_target_uncertain_ignore_fraction'])} | {skel['inside_predicted_fibrous_pixels']} | "
                f"{skel['inside_predicted_clump_pixels']} | {skel['outside_predicted_foreground_pixels']} |"
            )
    lines += [
        "",
        "## Probability Summaries By Expert Region",
        "",
        "| Pilot | Region | Fib prob mean/p95 | Clump prob mean/p95 | Skeleton prob mean/p95 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for pilot in summary["pilots"]:
        for region, probs in pilot["leakage"]["probability_summaries"].items():
            fib = probs["fibrous_probability"]
            clump = probs["clump_probability"]
            skel = probs["skeleton_probability"]
            lines.append(
                f"| {pilot['pilot_id']} | {region} | {fmt(fib['mean'])}/{fmt(fib['p95'])} | "
                f"{fmt(clump['mean'])}/{fmt(clump['p95'])} | {fmt(skel['mean'])}/{fmt(skel['p95'])} |"
            )
    lines += [
        "",
        "## Intensity And Domain Gap",
        "",
        "| Source | Mean | Std | p50 | p95 | p99 | Zero frac | Local var mean | Local var p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pilot in summary["pilots"]:
        lines.append(intensity_row(pilot["pilot_id"], pilot["intensity"]))
    synthetic = summary["domain_gap"]["synthetic_validation"]
    if synthetic.get("status") == "available":
        lines.append(intensity_row(f"synthetic validation n={synthetic['sample_count']}", synthetic))
    else:
        lines.append(f"| synthetic validation | {synthetic.get('status')} | {synthetic.get('reason', '')} |  |  |  |  |  |  |")
    lines += [
        "",
        "## Main Visual Failure Modes",
        "",
        "- Semantic predictions over-segment fibrous area in both pilots, with predicted/target area ratios above 1.0 and many small false-positive components.",
        "- Boundary disagreement contributes to both false-positive and false-negative fibrous errors, but large missed regions remain visible in the component summaries.",
        "- Skeleton predictions are mostly constrained to target fibrous regions, but exact skeleton Dice remains low; 2 px tolerant recovery is substantially higher than exact-pixel Dice.",
        "- Raising the skeleton threshold to 0.85 improves precision in places but lowers recall and does not remove the need to inspect skeleton failure cases.",
        f"- Leakage flags: {leakage_flags_text(summary)}",
        "",
        "## Conclusion",
        "",
        f"{summary['recommendation']}.",
        "",
    ]
    return "\n".join(lines)


def leakage_flags_text(summary: dict[str, Any]) -> str:
    flags = sorted({flag for pilot in summary["pilots"] for flag in pilot["leakage"]["interpretation_flags"]})
    synthetic = summary["domain_gap"]["synthetic_validation"]
    if synthetic.get("status") == "available":
        synthetic_p95 = float(synthetic["p95"])
        if synthetic_p95 > 0 and any(float(pilot["intensity"]["p95"]) > 2 * synthetic_p95 for pilot in summary["pilots"]):
            flags.append("domain-gap likely")
    return ", ".join(flags) if flags else "none"


def intensity_row(name: str, stats: dict[str, Any]) -> str:
    return (
        f"| {name} | {fmt(stats['mean'])} | {fmt(stats['std'])} | {fmt(stats['p50'])} | "
        f"{fmt(stats['p95'])} | {fmt(stats['p99'])} | {fmt(stats['zero_fraction'])} | "
        f"{fmt(stats['local_variance_mean_7x7'])} | {fmt(stats['local_variance_p95_7x7'])} |"
    )


def write_error_panel(
    path: Path,
    sample: dict[str, Any],
    pred: dict[str, np.ndarray],
    semantic: dict[str, Any],
    skeleton: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = sample["image_uint8"] if sample["image_uint8"] is not None else normalize_uint8(sample["image_float"])
    target_sem = sample["real_semantic_mask"]
    valid = target_sem != 255
    target_fib = (target_sem == 1) & valid
    pred_fib = (pred["semantic_class_map"] == 1) & valid
    fp_fib = pred_fib & (target_sem == 0)
    fn_fib = target_fib & ~pred_fib
    target_skel = (sample["real_skeleton_mask"] > 0) & valid
    pred_skel = (pred["skeleton_probability"] > 0.75) & valid
    skel_fp = pred_skel & (distance_transform_edt(~target_skel) > 2)
    skel_fn = target_skel & (distance_transform_edt(~pred_skel) > 2)
    fp_box = semantic["false_positive_components"]["largest_box_yx"]
    fn_box = semantic["false_negative_components"]["largest_box_yx"]
    tiles = [
        draw_tile(outline_overlay(raw, target_fib, (0, 220, 80)), "expert fibrous outline"),
        draw_tile(outline_overlay(raw, pred_fib, (255, 220, 0)), "pred fibrous outline"),
        draw_tile(mask_overlay(raw, fp_fib, (255, 60, 40)), "fibrous false positives"),
        draw_tile(mask_overlay(raw, fn_fib, (40, 170, 255)), "fibrous false negatives"),
        draw_tile(mask_overlay(raw, target_skel, (0, 220, 255)), "expert snakes"),
        draw_tile(mask_overlay(raw, pred_skel, (255, 0, 255)), "pred skeleton 0.75"),
        draw_tile(two_error_overlay(raw, skel_fp, skel_fn), "skeleton fp/fn >2px"),
        draw_tile(crop_error_tile(raw, fp_fib, fp_box), "largest fibrous FP crop"),
        draw_tile(crop_error_tile(raw, fn_fib, fn_box), "largest fibrous FN crop"),
    ]
    tile_w, tile_h = tiles[0].size
    panel = Image.new("RGB", (3 * tile_w, 3 * tile_h), "white")
    for i, tile in enumerate(tiles):
        panel.paste(tile, ((i % 3) * tile_w, (i // 3) * tile_h))
    panel.save(path)


def write_leakage_panel(path: Path, sample: dict[str, Any], pred: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = sample["image_uint8"] if sample["image_uint8"] is not None else normalize_uint8(sample["image_float"])
    target_sem = sample["real_semantic_mask"]
    regions = target_region_masks(target_sem)
    pred_fib = pred["semantic_class_map"] == 1
    pred_skel = pred["skeleton_probability"] > 0.75
    tiles = [
        draw_tile(gray_rgb(raw), "raw image"),
        draw_tile(expert_regions_rgb(target_sem), "expert regions"),
        draw_tile(mask_overlay(raw, pred_fib & regions["target_fibrous"], (0, 220, 80)), "pred fib in fibrous"),
        draw_tile(mask_overlay(raw, pred_fib & regions["target_clump"], (255, 130, 0)), "pred fib in clump"),
        draw_tile(mask_overlay(raw, pred_fib & regions["target_uncertain_ignore"], (170, 90, 255)), "pred fib in ignore"),
        draw_tile(mask_overlay(raw, pred_skel & regions["target_fibrous"], (0, 220, 255)), "skel in fibrous"),
        draw_tile(mask_overlay(raw, pred_skel & regions["target_clump"], (255, 0, 255)), "skel in clump"),
        draw_tile(mask_overlay(raw, pred_skel & regions["target_uncertain_ignore"], (255, 80, 160)), "skel in ignore"),
        draw_tile(heatmap(pred["clump_probability"]), "clump probability"),
        draw_tile(heatmap(pred["fibrous_probability"]), "fibrous probability"),
    ]
    tile_w, tile_h = tiles[0].size
    panel = Image.new("RGB", (5 * tile_w, 2 * tile_h), "white")
    for i, tile in enumerate(tiles):
        panel.paste(tile, ((i % 5) * tile_w, (i // 5) * tile_h))
    panel.save(path)


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0 or float(arr.max()) == float(arr.min()):
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - arr.min()) / (arr.max() - arr.min()) * 255, 0, 255).astype(np.uint8)


def gray_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = normalize_uint8(image)
    return np.repeat(image[..., None], 3, axis=2)


def expert_regions_rgb(target_semantic: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*target_semantic.shape, 3), dtype=np.uint8)
    rgb[target_semantic == 1] = (0, 220, 80)
    rgb[target_semantic == 3] = (255, 130, 0)
    rgb[target_semantic == 255] = (170, 90, 255)
    return rgb


def heatmap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0, 1)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(255 * values, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(255 * (1 - np.abs(values - 0.5) * 2), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(255 * (1 - values), 0, 255).astype(np.uint8)
    return rgb


def outline_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = gray_rgb(image)
    edge = binary_dilation(mask) & ~binary_erosion(mask)
    rgb[edge] = color
    return rgb


def mask_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = gray_rgb(image).astype(np.float32)
    rgb[mask] = 0.35 * rgb[mask] + 0.65 * np.asarray(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def two_error_overlay(image: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    rgb = gray_rgb(image).astype(np.float32) * 0.6
    rgb[fp] = (255, 60, 40)
    rgb[fn] = (40, 170, 255)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def crop_error_tile(image: np.ndarray, mask: np.ndarray, box: list[int] | None, margin: int = 32) -> np.ndarray:
    if box is None:
        return gray_rgb(image)
    y0, x0, y1, x1 = box
    y0 = max(y0 - margin, 0)
    x0 = max(x0 - margin, 0)
    y1 = min(y1 + margin, image.shape[0])
    x1 = min(x1 + margin, image.shape[1])
    return mask_overlay(image[y0:y1, x0:x1], mask[y0:y1, x0:x1], (255, 60, 40))


def draw_tile(arr: np.ndarray, title: str, size: int = 256) -> Image.Image:
    im = Image.fromarray(arr.astype(np.uint8), "RGB")
    scale = min(size / max(im.size), 1.0)
    if scale < 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size + 18), "white")
    canvas.paste(im, ((size - im.width) // 2, 18 + (size - im.height) // 2))
    ImageDraw.Draw(canvas).text((4, 3), title, fill=(0, 0, 0))
    return canvas


def fraction(num: Any, denom: Any) -> float | str:
    denom_i = int(denom)
    if denom_i == 0:
        return NOT_APPLICABLE
    return float(int(num) / denom_i)


def ratio_or_na(num: Any, denom: Any) -> float | str:
    denom_i = int(denom)
    if denom_i == 0:
        return NOT_APPLICABLE
    return float(int(num) / denom_i)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
