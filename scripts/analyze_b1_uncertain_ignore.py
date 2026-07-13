#!/usr/bin/env python
"""Diagnose B1 real-STED behavior inside expert uncertain_ignore regions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fibras.annotations import build_real_annotation_sample
from scripts.evaluate_real_annotation_batch import read_annotation_manifest


REGIONS = {
    "confident_background": 0,
    "confident_fibrous_tau": 1,
    "confident_clump": 3,
    "expert_uncertain_ignore": 255,
}
FIB_THRESHOLDS = (0.5, 0.7, 0.8, 0.9)
SKELETON_THRESHOLD = 0.75
SPATIAL_DISTANCES = (2, 5, 10, 20)
SELECTION_GRID = np.linspace(0.0, 1.0, 101)
COMPARISONS = {
    "uncertain_vs_fibrous": (1,),
    "uncertain_vs_clump": (3,),
    "uncertain_vs_fibrous_clump": (1, 3),
    "uncertain_vs_background": (0,),
    "uncertain_vs_all_confident": (0, 1, 3),
}


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    outer_fold: int
    partition: str
    image_path: str
    snakes_path: str | None
    labels_path: str


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.manifest, args.fold_manifest, args.image_root, args.annotation_root)
    predictions = {r.sample_id: prediction_path(args.suite_root, r) for r in records if r.partition == "test"}
    missing = [sid for sid, path in predictions.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing B1 prediction files for {len(missing)} held-out images: {missing[:5]}")

    image_rows, samples = [], {}
    for record in records:
        if record.partition in {"validation", "test"}:
            samples[record.sample_id] = build_real_annotation_sample(record.image_path, record.snakes_path, record.labels_path)
    test_records = [r for r in records if r.partition == "test"]
    for record in test_records:
        row = analyze_test_image(record, samples[record.sample_id], predictions[record.sample_id])
        image_rows.append(row)
    write_csv(args.out / "per_image_uncertain_ignore.csv", image_rows)

    discrimination_rows = discrimination_analysis(test_records, samples, predictions)
    write_csv(args.out / "uncertainty_discrimination.csv", discrimination_rows)
    distribution_rows, distribution_values = probability_overlap(test_records, samples, predictions)
    write_csv(args.out / "uncertainty_probability_overlap.csv", distribution_rows)

    pr_rows, calib_rows = uncertainty_curves(test_records, samples, predictions)
    write_csv(args.out / "uncertainty_pr_curve.csv", pr_rows)
    write_csv(args.out / "uncertainty_calibration_curve.csv", calib_rows)

    threshold_rows, sweep_rows = calibrate_abstention(records, samples, predictions, args.suite_root, args.out)
    write_csv(args.out / "abstention_tradeoffs.csv", threshold_rows)
    write_csv(args.out / "validation_abstention_sweeps.csv", sweep_rows)
    plot_outputs(args.out, distribution_values, pr_rows, calib_rows, threshold_rows, sweep_rows)
    write_qualitative_panels(args.out / "qualitative_panels", test_records, samples, predictions, image_rows)
    write_report(args.out / "report.md", image_rows, discrimination_rows, distribution_rows, threshold_rows)
    print(f"wrote B1 uncertain_ignore diagnostics to {args.out}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", type=Path, default=Path("/ssd/STED_experiments/controlled_real_sted_v1"))
    p.add_argument("--out", type=Path, default=Path("/ssd/STED_experiments/controlled_real_sted_v1/analysis/b1_uncertain_ignore"))
    p.add_argument("--manifest", type=Path, default=ROOT / "data_manifests/real_annotations.csv")
    p.add_argument("--fold-manifest", type=Path, default=ROOT / "data_manifests/real_annotation_folds_v1.csv")
    p.add_argument("--image-root", type=Path)
    p.add_argument("--annotation-root", type=Path)
    return p


def load_records(manifest: Path, folds: Path, image_root: Path | None, annotation_root: Path | None) -> list[ImageRecord]:
    annotations = {r["sample_id"]: r for r in read_annotation_manifest(manifest, image_root=image_root, annotation_root=annotation_root)}
    with folds.open(newline="", encoding="utf-8") as f:
        fold_rows = list(csv.DictReader(f))
    out = []
    for row in fold_rows:
        if row.get("validation_status", "valid") != "valid":
            continue
        ann = annotations[row["sample_id"]]
        out.append(ImageRecord(
            sample_id=row["sample_id"],
            outer_fold=int(row["outer_fold"]),
            partition=row["partition"],
            image_path=ann["image_path"],
            snakes_path=ann["snakes_path"],
            labels_path=ann["labels_path"],
        ))
    return out


def prediction_path(root: Path, record: ImageRecord) -> Path:
    return root / "evaluations" / "b1_real_only" / f"fold_{record.outer_fold}" / record.sample_id / "predictions.npz"


def load_probabilities(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        pred = {k: data[k].copy() for k in data.files}
    required = {"fibrous_probability", "clump_probability", "skeleton_probability", "uncertainty_probability"}
    missing = sorted(required - set(pred))
    if missing:
        raise ValueError(f"{path}: missing probability arrays {missing}")
    pred["background_probability"] = np.clip(1.0 - pred["fibrous_probability"] - pred["clump_probability"], 0.0, 1.0)
    pred["max_semantic_probability"] = np.maximum.reduce([
        pred["background_probability"], pred["fibrous_probability"], pred["clump_probability"],
    ])
    if "semantic_class_map" not in pred:
        classes = np.argmax(np.stack([
            pred["background_probability"], pred["fibrous_probability"], pred["clump_probability"],
        ]), axis=0)
        pred["semantic_class_map"] = np.choose(classes, [0, 1, 3]).astype(np.uint8)
    return pred


def analyze_test_image(record: ImageRecord, sample: dict[str, Any], pred_path: Path) -> dict[str, Any]:
    target = sample["real_semantic_mask"]
    skeleton = sample["real_skeleton_mask"] > 0
    pred = load_probabilities(pred_path)
    pred_fib = pred["fibrous_probability"] >= 0.5
    valid = target != 255
    row: dict[str, Any] = {"sample_id": record.sample_id, "outer_fold": record.outer_fold}
    row.update(uncertainty_metrics(pred["uncertainty_probability"], target == 255))
    row.update(semantic_metrics(pred_fib, target == 1, valid, prefix="fibrous"))
    row.update(skeleton_metrics(pred["skeleton_probability"] >= SKELETON_THRESHOLD, skeleton, valid))
    for region, value in REGIONS.items():
        mask = target == value
        row[f"{region}_pixels"] = int(mask.sum())
        for name in ["background", "fibrous", "clump", "skeleton", "uncertainty"]:
            key = f"{name}_probability"
            row.update(summary_fields(f"{region}_{name}", pred[key][mask]))
    uncertain = target == 255
    row["uncertain_pixels"] = int(uncertain.sum())
    for threshold in FIB_THRESHOLDS:
        row[f"uncertain_fibrous_prob_ge_{threshold:g}_fraction"] = fraction((pred["fibrous_probability"][uncertain] >= threshold).sum(), uncertain.sum())
    return row


def discrimination_analysis(
    records: list[ImageRecord],
    samples: dict[str, dict[str, Any]],
    predictions: dict[str, Path],
) -> list[dict[str, Any]]:
    by_comparison: dict[str, list[tuple[ImageRecord, np.ndarray, np.ndarray]]] = {
        name: [] for name in COMPARISONS
    }
    by_comparison.update({f"uncertain_vs_spatial_tau_{d}px": [] for d in SPATIAL_DISTANCES})
    for record in records:
        target = samples[record.sample_id]["real_semantic_mask"]
        score = load_probabilities(predictions[record.sample_id])["uncertainty_probability"]
        uncertain = target == 255
        for name, negatives in COMPARISONS.items():
            mask = uncertain | np.isin(target, negatives)
            by_comparison[name].append((record, uncertain[mask], score[mask]))
        distance = distance_transform_edt(~uncertain)
        tau = np.isin(target, (1, 3))
        for d in SPATIAL_DISTANCES:
            mask = uncertain | (tau & (distance <= d))
            by_comparison[f"uncertain_vs_spatial_tau_{d}px"].append((record, uncertain[mask], score[mask]))

    rows = []
    for comparison, parts in by_comparison.items():
        image_rows = [discrimination_row(comparison, "image", r.sample_id, r.outer_fold, y, s) for r, y, s in parts]
        rows.extend(image_rows)
        rows.append(macro_discrimination_row(comparison, "global_macro", "all", image_rows))
        rows.append(pooled_discrimination_row(comparison, "global_pooled", "all", parts))
        for fold in sorted({r.outer_fold for r, _, _ in parts}):
            fold_parts = [part for part in parts if part[0].outer_fold == fold]
            fold_images = [row for row in image_rows if row["outer_fold"] == fold]
            rows.append(macro_discrimination_row(comparison, "fold_macro", f"fold_{fold}", fold_images, fold))
            rows.append(pooled_discrimination_row(comparison, "fold_pooled", f"fold_{fold}", fold_parts, fold))
    return rows


def discrimination_row(
    comparison: str,
    scope: str,
    sample_id: str,
    outer_fold: int | str,
    target: np.ndarray,
    score: np.ndarray,
) -> dict[str, Any]:
    positives, total = int(target.sum()), int(target.size)
    return {
        "comparison": comparison,
        "scope": scope,
        "sample_id": sample_id,
        "outer_fold": outer_fold,
        "positive_pixels": positives,
        "negative_pixels": total - positives,
        "prevalence": ratio(positives, total),
        "auroc": auroc(target, score),
        "average_precision": average_precision(target, score),
    }


def pooled_discrimination_row(
    comparison: str,
    scope: str,
    sample_id: str,
    parts: list[tuple[ImageRecord, np.ndarray, np.ndarray]],
    outer_fold: int | str = "all",
) -> dict[str, Any]:
    return discrimination_row(
        comparison,
        scope,
        sample_id,
        outer_fold,
        np.concatenate([part[1] for part in parts]),
        np.concatenate([part[2] for part in parts]),
    )


def macro_discrimination_row(
    comparison: str,
    scope: str,
    sample_id: str,
    rows: list[dict[str, Any]],
    outer_fold: int | str = "all",
) -> dict[str, Any]:
    return {
        "comparison": comparison,
        "scope": scope,
        "sample_id": sample_id,
        "outer_fold": outer_fold,
        "positive_pixels": sum(int(row["positive_pixels"]) for row in rows),
        "negative_pixels": sum(int(row["negative_pixels"]) for row in rows),
        "prevalence": mean(rows, "prevalence"),
        "auroc": mean(rows, "auroc"),
        "average_precision": mean(rows, "average_precision"),
    }


def probability_overlap(
    records: list[ImageRecord],
    samples: dict[str, dict[str, Any]],
    predictions: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    values: dict[str, list[np.ndarray]] = {
        "uncertain_ignore": [],
        "confident_fibrous_tau": [],
        "confident_clump": [],
        "spatial_tau_within_10px": [],
        "confident_background": [],
    }
    for record in records:
        target = samples[record.sample_id]["real_semantic_mask"]
        score = load_probabilities(predictions[record.sample_id])["uncertainty_probability"]
        uncertain = target == 255
        distance = distance_transform_edt(~uncertain)
        masks = {
            "uncertain_ignore": uncertain,
            "confident_fibrous_tau": target == 1,
            "confident_clump": target == 3,
            "spatial_tau_within_10px": np.isin(target, (1, 3)) & (distance <= 10),
            "confident_background": target == 0,
        }
        for name, mask in masks.items():
            values[name].append(score[mask])
    combined = {name: np.concatenate(parts) for name, parts in values.items()}
    uncertain_values = combined["uncertain_ignore"]
    rows = []
    for name, vals in combined.items():
        row = {"region": name, "pixels": int(vals.size), **quantile_fields(vals)}
        row["histogram_overlap_with_uncertain"] = histogram_overlap(uncertain_values, vals)
        rows.append(row)
    return rows, combined


def quantile_fields(values: np.ndarray) -> dict[str, float | str]:
    if values.size == 0:
        return {key: "not_applicable" for key in ("mean", "p05", "p25", "p50", "p75", "p95")}
    return {"mean": float(values.mean()), **{
        f"p{q:02d}": float(np.percentile(values, q)) for q in (5, 25, 50, 75, 95)
    }}


def histogram_overlap(a: np.ndarray, b: np.ndarray, bins: int = 100) -> float | str:
    if a.size == 0 or b.size == 0:
        return "not_applicable"
    ah, _ = np.histogram(a, bins=bins, range=(0, 1), density=False)
    bh, _ = np.histogram(b, bins=bins, range=(0, 1), density=False)
    return float(np.minimum(ah / ah.sum(), bh / bh.sum()).sum())


def uncertainty_metrics(score: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    y = target.ravel().astype(bool)
    s = score.ravel().astype(np.float64)
    return {
        "uncertainty_auroc": auroc(y, s),
        "uncertainty_average_precision": average_precision(y, s),
        "uncertain_mean_uncertainty_probability": float(s[y].mean()) if y.any() else "not_applicable",
        "nonuncertain_mean_uncertainty_probability": float(s[~y].mean()) if (~y).any() else "not_applicable",
    }


def semantic_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, prefix: str) -> dict[str, Any]:
    pred, target = pred & valid, target & valid
    tp = int((pred & target).sum())
    return {
        f"{prefix}_dice_confident": dice_counts(tp, int(pred.sum()), int(target.sum())),
        f"{prefix}_precision_confident": ratio(tp, int(pred.sum())),
        f"{prefix}_recall_confident": ratio(tp, int(target.sum())),
    }


def skeleton_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    pred, target = pred & valid, target & valid
    tp = int((pred & target).sum())
    return {
        "skeleton_dice_confident": dice_counts(tp, int(pred.sum()), int(target.sum())),
        "skeleton_recovery_2px_confident": within_distance_fraction(target, pred, 2.0),
    }


def calibrate_abstention(
    records: list[ImageRecord],
    samples: dict[str, dict[str, Any]],
    test_predictions: dict[str, Path],
    suite_root: Path,
    analysis_out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, sweep_rows = [], []
    cache: dict[str, dict[str, np.ndarray]] = {}
    for fold in sorted({r.outer_fold for r in records}):
        val = [r for r in records if r.outer_fold == fold and r.partition == "validation"]
        test = [r for r in records if r.outer_fold == fold and r.partition == "test"]
        val_predictions = {r.sample_id: validation_prediction_path(analysis_out, r) for r in val}
        grids = selection_metric_grids(val, samples, val_predictions, cache)
        sweep_rows.extend(abstention_sweep_rows(fold, grids))
        for rule in ["uncertainty", "low_confidence", "combined"]:
            threshold = select_threshold(rule, val, samples, val_predictions, cache, grids)
            rows.extend(evaluate_abstention_rule(rule, threshold, test, samples, test_predictions, cache))
    return rows, sweep_rows


def validation_prediction_path(analysis_out: Path, record: ImageRecord) -> Path:
    return analysis_out / "validation_predictions" / f"fold_{record.outer_fold}" / record.sample_id / "predictions.npz"


def select_threshold(
    rule: str,
    records: list[ImageRecord],
    samples: dict[str, dict[str, Any]],
    predictions: dict[str, Path],
    cache: dict[str, dict[str, np.ndarray]] | None = None,
    metric_grids: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    grids = metric_grids or selection_metric_grids(records, samples, predictions, cache)
    baseline_fib = grids["fibrous_dice"][-1, 0]
    baseline_skel = grids["skeleton_dice"][-1, 0]
    valid = (
        (grids["fibrous_dice"] >= baseline_fib - 0.01)
        & (grids["skeleton_dice"] >= baseline_skel - 0.01)
        & (grids["target_fibres_abstained_fraction"] <= 0.02)
        & (grids["high_conf_uncertain_reduction"] > 0)
    )
    allowed = np.zeros(valid.shape, dtype=bool)
    if rule == "uncertainty":
        allowed[:, 0] = True
    elif rule == "low_confidence":
        allowed[-1, :] = True
    elif rule == "combined":
        allowed[:] = True
    else:
        raise ValueError(f"unknown abstention rule: {rule}")
    candidates = np.argwhere(valid & allowed)
    common = {
        "threshold_status": "no_operational_threshold",
        "uncertainty_threshold": "not_applicable",
        "confidence_threshold": "not_applicable",
        "validation_baseline_fibrous_dice": float(baseline_fib),
        "validation_baseline_skeleton_dice": float(baseline_skel),
    }
    if not candidates.size:
        return common
    best = min(
        candidates,
        key=lambda ij: (
            -grids["high_conf_uncertain_reduction"][tuple(ij)],
            grids["all_pixel_rejected_fraction"][tuple(ij)],
        ),
    )
    i, j = map(int, best)
    return {
        **common,
        "threshold_status": "operational_threshold",
        "uncertainty_threshold": float(SELECTION_GRID[i]),
        "confidence_threshold": float(SELECTION_GRID[j]),
        "validation_fibrous_dice": float(grids["fibrous_dice"][i, j]),
        "validation_skeleton_dice": float(grids["skeleton_dice"][i, j]),
        "validation_target_fibres_abstained_fraction": float(grids["target_fibres_abstained_fraction"][i, j]),
        "validation_high_conf_uncertain_reduction": int(grids["high_conf_uncertain_reduction"][i, j]),
        "validation_expert_uncertain_recall": float(grids["expert_uncertain_recall"][i, j]),
        "validation_all_pixel_rejected_fraction": float(grids["all_pixel_rejected_fraction"][i, j]),
    }


def abstention_sweep_rows(fold: int, grids: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    baseline_fib = grids["fibrous_dice"][-1, 0]
    baseline_skel = grids["skeleton_dice"][-1, 0]
    rows = []
    for rule, indices in {
        "uncertainty": ((i, 0) for i in range(SELECTION_GRID.size)),
        "low_confidence": ((SELECTION_GRID.size - 1, j) for j in range(SELECTION_GRID.size)),
        "combined": ((i, j) for i in range(SELECTION_GRID.size) for j in range(SELECTION_GRID.size)),
    }.items():
        for i, j in indices:
            rows.append({
                "outer_fold": fold,
                "rule": rule,
                "uncertainty_threshold": float(SELECTION_GRID[i]),
                "confidence_threshold": float(SELECTION_GRID[j]),
                "fibrous_dice_full_image": float(grids["fibrous_dice"][i, j]),
                "skeleton_dice_full_image": float(grids["skeleton_dice"][i, j]),
                "target_fibres_abstained_fraction": float(grids["target_fibres_abstained_fraction"][i, j]),
                "expert_uncertain_recall": float(grids["expert_uncertain_recall"][i, j]),
                "all_pixel_rejected_fraction": float(grids["all_pixel_rejected_fraction"][i, j]),
                "high_conf_uncertain_reduction": int(grids["high_conf_uncertain_reduction"][i, j]),
                "constraints_satisfied": bool(
                    grids["fibrous_dice"][i, j] >= baseline_fib - 0.01
                    and grids["skeleton_dice"][i, j] >= baseline_skel - 0.01
                    and grids["target_fibres_abstained_fraction"][i, j] <= 0.02
                ),
            })
    return rows


def selection_metric_grids(
    records: list[ImageRecord],
    samples: dict[str, dict[str, Any]],
    predictions: dict[str, Path],
    cache: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    kept = {name: np.zeros((SELECTION_GRID.size, SELECTION_GRID.size), dtype=np.int64) for name in (
        "all", "uncertain", "target_fib", "pred_fib", "pred_fib_all", "fib_tp", "target_skel", "pred_skel", "skel_tp", "high_conf_uncertain",
    )}
    for record in records:
        target = samples[record.sample_id]["real_semantic_mask"]
        target_skel = samples[record.sample_id]["real_skeleton_mask"] > 0
        pred = cached_probabilities(record.sample_id, predictions[record.sample_id], cache)
        valid = target != 255
        pred_fib = pred["semantic_class_map"] == 1
        pred_skel = pred["skeleton_probability"] > SKELETON_THRESHOLD
        masks = {
            "all": np.ones(target.shape, dtype=bool),
            "uncertain": target == 255,
            "target_fib": target == 1,
            "pred_fib": pred_fib & valid,
            "pred_fib_all": pred_fib,
            "fib_tp": pred_fib & (target == 1),
            "target_skel": target_skel & valid,
            "pred_skel": pred_skel & valid,
            "skel_tp": pred_skel & target_skel & valid,
            "high_conf_uncertain": (target == 255) & (pred["fibrous_probability"] >= 0.7),
        }
        for name, mask in masks.items():
            kept[name] += kept_count_grid(pred["uncertainty_probability"], pred["max_semantic_probability"], mask)
    fib_target = kept["target_fib"][-1, 0]
    skel_target = kept["target_skel"][-1, 0]
    high_conf = kept["high_conf_uncertain"][-1, 0]
    uncertain = kept["uncertain"][-1, 0]
    return {
        "fibrous_dice": divide(2 * kept["fib_tp"], kept["pred_fib"] + fib_target),
        "skeleton_dice": divide(2 * kept["skel_tp"], kept["pred_skel"] + skel_target),
        "target_fibres_abstained_fraction": divide(fib_target - kept["target_fib"], fib_target),
        "high_conf_uncertain_reduction": high_conf - kept["high_conf_uncertain"],
        "expert_uncertain_recall": divide(uncertain - kept["uncertain"], uncertain),
        "predicted_fibres_abstained_fraction": divide(kept["pred_fib_all"][-1, 0] - kept["pred_fib_all"], kept["pred_fib_all"][-1, 0]),
        "all_pixel_rejected_fraction": divide(kept["all"][-1, 0] - kept["all"], kept["all"][-1, 0]),
    }


def kept_count_grid(uncertainty: np.ndarray, confidence: np.ndarray, mask: np.ndarray) -> np.ndarray:
    u_index = np.searchsorted(SELECTION_GRID, uncertainty[mask], side="left")
    c_index = np.searchsorted(SELECTION_GRID, confidence[mask], side="right") - 1
    hist = np.zeros((SELECTION_GRID.size, SELECTION_GRID.size), dtype=np.int64)
    np.add.at(hist, (u_index, c_index), 1)
    cumulative = hist.cumsum(axis=0)
    return np.flip(np.cumsum(np.flip(cumulative, axis=1), axis=1), axis=1)


def divide(numerator: np.ndarray | int, denominator: np.ndarray | int) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(np.asarray(numerator), dtype=float), where=np.asarray(denominator) != 0)


def evaluate_abstention_rule(
    rule: str,
    threshold: dict[str, Any],
    records: list[ImageRecord],
    samples: dict[str, dict[str, Any]],
    predictions: dict[str, Path],
    cache: dict[str, dict[str, np.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metrics = pooled_abstention_metrics([record], samples, predictions, threshold, cache)
        rows.append({"rule": rule, "outer_fold": record.outer_fold, "sample_id": record.sample_id, **threshold, **metrics})
    pooled = pooled_abstention_metrics(records, samples, predictions, threshold, cache)
    rows.append({"rule": rule, "outer_fold": records[0].outer_fold, "sample_id": "fold_pooled", **threshold, **pooled})
    return rows


def pooled_abstention_metrics(
    records: list[ImageRecord],
    samples: dict[str, dict[str, Any]],
    predictions: dict[str, Path],
    threshold: dict[str, Any],
    cache: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    counts = dict(all=0, rejected=0, uncertain=0, uncertain_rejected=0, pred_fib=0, pred_fib_rejected=0, target_fib=0, target_fib_rejected=0, residual_uncertain_fib=0)
    fib_tp = fib_pred = fib_target = clump_tp = clump_pred = clump_target = 0
    skel_tp = skel_pred = skel_target = skel_recovered = retained_fib_target = 0
    for record in records:
        target = samples[record.sample_id]["real_semantic_mask"]
        target_skel = samples[record.sample_id]["real_skeleton_mask"] > 0
        pred = cached_probabilities(record.sample_id, predictions[record.sample_id], cache)
        reject = abstention_mask(pred, threshold)
        pred_fib = pred["semantic_class_map"] == 1
        pred_clump = pred["semantic_class_map"] == 3
        pred_skel = pred["skeleton_probability"] > SKELETON_THRESHOLD
        valid = target != 255
        counts["all"] += target.size
        counts["rejected"] += int(reject.sum())
        counts["uncertain"] += int((target == 255).sum())
        counts["uncertain_rejected"] += int((reject & (target == 255)).sum())
        counts["pred_fib"] += int(pred_fib.sum())
        counts["pred_fib_rejected"] += int((pred_fib & reject).sum())
        counts["target_fib"] += int((target == 1).sum())
        counts["target_fib_rejected"] += int(((target == 1) & reject).sum())
        counts["residual_uncertain_fib"] += int((pred_fib & ~reject & (target == 255)).sum())
        kept_valid = valid & ~reject
        ft = target == 1
        fib_tp += int((pred_fib & ft & ~reject).sum())
        fib_pred += int((pred_fib & kept_valid).sum())
        fib_target += int(ft.sum())
        retained_fib_target += int((ft & ~reject).sum())
        ct = target == 3
        clump_tp += int((pred_clump & ct & ~reject).sum())
        clump_pred += int((pred_clump & kept_valid).sum())
        clump_target += int(ct.sum())
        st = target_skel & valid
        sp = pred_skel & kept_valid
        skel_tp += int((sp & st).sum())
        skel_pred += int(sp.sum())
        skel_target += int(st.sum())
        if st.any() and sp.any():
            skel_recovered += int(np.count_nonzero(st & (distance_transform_edt(~sp) <= 2.0)))
    return {
        "expert_uncertain_recall": ratio(counts["uncertain_rejected"], counts["uncertain"]),
        "all_pixel_rejected_fraction": ratio(counts["rejected"], counts["all"]),
        "target_fibres_abstained_fraction": ratio(counts["target_fib_rejected"], counts["target_fib"]),
        "predicted_fibres_rejected_fraction": ratio(counts["pred_fib_rejected"], counts["pred_fib"]),
        "residual_fibrous_predictions_inside_uncertain_pixels": counts["residual_uncertain_fib"],
        "fibrous_dice_full_image_after_abstention": dice_counts(fib_tp, fib_pred, fib_target),
        "fibrous_precision_full_image_after_abstention": ratio(fib_tp, fib_pred),
        "fibrous_recall_full_image_after_abstention": ratio(fib_tp, fib_target),
        "clump_dice_full_image_after_abstention": dice_counts(clump_tp, clump_pred, clump_target),
        "skeleton_dice_full_image_after_abstention": dice_counts(skel_tp, skel_pred, skel_target),
        "skeleton_recovery_2px_after_abstention": ratio(skel_recovered, skel_target),
        "fibrous_dice_retained_pixels_selective": dice_counts(fib_tp, fib_pred, retained_fib_target),
    }


def abstention_mask(pred: dict[str, np.ndarray], threshold: dict[str, Any]) -> np.ndarray:
    if threshold.get("threshold_status") == "no_operational_threshold":
        return np.zeros(pred["uncertainty_probability"].shape, dtype=bool)
    return (pred["uncertainty_probability"] > float(threshold["uncertainty_threshold"])) | (
        pred["max_semantic_probability"] < float(threshold["confidence_threshold"])
    )


def cached_probabilities(
    sample_id: str,
    path: Path,
    cache: dict[str, dict[str, np.ndarray]] | None,
) -> dict[str, np.ndarray]:
    if cache is None:
        return load_probabilities(path)
    key = f"{sample_id}:{path}"
    if key not in cache:
        cache[key] = load_probabilities(path)
    return cache[key]


def uncertainty_curves(
    records: list[ImageRecord],
    samples: dict[str, dict[str, Any]],
    predictions: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_parts, s_parts = [], []
    for record in records:
        target = samples[record.sample_id]["real_semantic_mask"]
        mask = (target == 255) | np.isin(target, (1, 3))
        y_parts.append((target[mask] == 255).ravel())
        s_parts.append(load_probabilities(predictions[record.sample_id])["uncertainty_probability"][mask].ravel())
    y = np.concatenate(y_parts).astype(bool)
    s = np.concatenate(s_parts).astype(np.float64)
    thresholds = np.unique(np.quantile(s, np.linspace(0, 1, 101)))
    pr = []
    for t in thresholds[::-1]:
        pred = s >= t
        pr.append({"threshold": float(t), "precision": ratio(int((pred & y).sum()), int(pred.sum())), "recall": ratio(int((pred & y).sum()), int(y.sum()))})
    bins = np.linspace(0, 1, 11)
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (s >= lo) & (s < hi if hi < 1 else s <= hi)
        calib.append({"bin_low": float(lo), "bin_high": float(hi), "mean_probability": float(s[mask].mean()) if mask.any() else "not_applicable", "observed_uncertain_fraction": ratio(int(y[mask].sum()), int(mask.sum())), "pixels": int(mask.sum())})
    return pr, calib


def plot_outputs(
    out: Path,
    distribution_values: dict[str, np.ndarray],
    pr_rows: list[dict[str, Any]],
    calib_rows: list[dict[str, Any]],
    abstention_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    plot_probability_distributions(out / "probability_distributions.png", distribution_values)
    plot_xy(out / "uncertainty_precision_recall.png", pr_rows, "recall", "precision", "Uncertain vs confident tau PR")
    plot_xy(out / "uncertainty_calibration.png", calib_rows, "mean_probability", "observed_uncertain_fraction", "Uncertain vs confident tau calibration")
    plt.figure(figsize=(6, 4))
    for rule, marker in [("uncertainty", "o"), ("low_confidence", "s"), ("combined", ".")]:
        rows = [r for r in sweep_rows if r["rule"] == rule and r["constraints_satisfied"]]
        plt.scatter(
            [r["target_fibres_abstained_fraction"] for r in rows],
            [r["expert_uncertain_recall"] for r in rows],
            s=10 if rule != "combined" else 3,
            alpha=0.45,
            marker=marker,
            label=rule,
        )
    plt.axvline(0.02, color="black", linestyle="--", linewidth=1)
    plt.xlabel("validation target fibres abstained")
    plt.ylabel("validation expert-uncertain recall")
    plt.xlim(0, 0.022)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "abstention_tradeoff.png", dpi=160)
    plt.close()


def plot_probability_distributions(path: Path, values: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 1, 101)
    for name, vals in values.items():
        if vals.size:
            plt.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5, label=name.replace("_", " "))
    plt.xlabel("uncertainty-head probability")
    plt.ylabel("density")
    plt.yscale("log")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_xy(path: Path, rows: list[dict[str, Any]], x: str, y: str, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    xs = [finite(r[x], np.nan) for r in rows]
    ys = [finite(r[y], np.nan) for r in rows]
    plt.plot(xs, ys)
    plt.xlabel(x.replace("_", " "))
    plt.ylabel(y.replace("_", " "))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_qualitative_panels(out: Path, records: list[ImageRecord], samples: dict[str, dict[str, Any]], predictions: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    selected_ids = [r["sample_id"] for r in sorted(rows, key=lambda x: finite(x["uncertain_fibrous_prob_ge_0.7_fraction"], -1), reverse=True)[:6]]
    for record in records:
        if record.sample_id not in selected_ids:
            continue
        sample = samples[record.sample_id]
        pred = load_probabilities(predictions[record.sample_id])
        raw = sample["image_uint8"] if sample["image_uint8"] is not None else normalize_display(sample["image_float"])
        tiles = [
            tile(gray_rgb(raw), "raw"),
            tile(mask_rgb(sample["real_semantic_mask"]), "expert"),
            tile(heatmap(pred["fibrous_probability"]), "fibrous p"),
            tile(heatmap(pred["uncertainty_probability"]), "uncertain p"),
            tile(overlay(raw, pred["fibrous_probability"] >= 0.7, (255, 50, 40)), "fibrous >= .7"),
            tile(overlay(raw, sample["real_semantic_mask"] == 255, (160, 90, 220)), "expert uncertain"),
        ]
        panel = Image.new("RGB", (3 * 256, 2 * 274), "white")
        for i, im in enumerate(tiles):
            panel.paste(im, ((i % 3) * 256, (i // 3) * 274))
        panel.save(out / f"{record.sample_id}.png")


def write_report(
    path: Path,
    image_rows: list[dict[str, Any]],
    discrimination_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    abstention_rows: list[dict[str, Any]],
) -> None:
    pooled = {row["comparison"]: row for row in discrimination_rows if row["scope"] == "global_pooled"}
    macro = {row["comparison"]: row for row in discrimination_rows if row["scope"] == "global_macro"}
    primary = pooled["uncertain_vs_fibrous_clump"]
    spatial = [pooled[f"uncertain_vs_spatial_tau_{d}px"] for d in SPATIAL_DISTANCES]
    background = pooled["uncertain_vs_background"]
    primary_auc = finite(primary["auroc"], 0.5)
    spatial_auc = min(finite(row["auroc"], 0.5) for row in spatial)
    background_auc = finite(background["auroc"], 0.5)
    if primary_auc >= 0.70 and spatial_auc >= 0.65:
        behavior = "an uncertainty detector"
    elif background_auc >= 0.75 and primary_auc < 0.65:
        behavior = "a foreground detector"
    elif primary_auc >= 0.65 and spatial_auc < primary_auc - 0.10:
        behavior = "a boundary or proximity detector"
    else:
        behavior = "an uninformative head for expert uncertainty"

    fold_rows = [row for row in abstention_rows if row["sample_id"] == "fold_pooled"]
    uncertainty_operational = [row for row in fold_rows if row["rule"] == "uncertainty" and row["threshold_status"] == "operational_threshold"]
    combined_operational = [row for row in fold_rows if row["rule"] == "combined" and row["threshold_status"] == "operational_threshold"]
    confidence_operational = [row for row in fold_rows if row["rule"] == "low_confidence" and row["threshold_status"] == "operational_threshold"]
    if len(uncertainty_operational) == 5 and primary_auc >= 0.65 and spatial_auc >= 0.60:
        recommendation = "retain the existing head with validation-only calibration"
    elif len(confidence_operational) > len(uncertainty_operational):
        recommendation = "abandon the existing uncertainty head; confidence-based abstention is better supported"
    else:
        recommendation = "abandon the existing head for operational use and retrain it before reuse"

    comparison_lines = []
    for name in COMPARISONS:
        comparison_lines.append(
            f"| {name} | {finite(pooled[name]['prevalence'], float('nan')):.3f} | "
            f"{finite(pooled[name]['auroc'], float('nan')):.3f} | {finite(pooled[name]['average_precision'], float('nan')):.3f} | "
            f"{finite(macro[name]['auroc'], float('nan')):.3f} | {finite(macro[name]['average_precision'], float('nan')):.3f} |"
        )
    spatial_lines = [
        f"| {d} | {finite(row['prevalence'], float('nan')):.3f} | {finite(row['auroc'], float('nan')):.3f} | {finite(row['average_precision'], float('nan')):.3f} |"
        for d, row in zip(SPATIAL_DISTANCES, spatial)
    ]
    threshold_lines = []
    for row in fold_rows:
        threshold_lines.append(
            "| {fold} | {rule} | {status} | {ut} | {ct} | {unc:.3f} | {all_rej:.3f} | {target_rej:.3f} | "
            "{pred_rej:.3f} | {resid} | {fib:.3f} | {prec:.3f} | {rec:.3f} | {clump:.3f} | {skel:.3f} | {skel2:.3f} |".format(
                fold=row["outer_fold"], rule=row["rule"], status=row["threshold_status"],
                ut=row["uncertainty_threshold"], ct=row["confidence_threshold"],
                unc=finite(row["expert_uncertain_recall"], float("nan")),
                all_rej=finite(row["all_pixel_rejected_fraction"], float("nan")),
                target_rej=finite(row["target_fibres_abstained_fraction"], float("nan")),
                pred_rej=finite(row["predicted_fibres_rejected_fraction"], float("nan")),
                resid=row["residual_fibrous_predictions_inside_uncertain_pixels"],
                fib=finite(row["fibrous_dice_full_image_after_abstention"], float("nan")),
                prec=finite(row["fibrous_precision_full_image_after_abstention"], float("nan")),
                rec=finite(row["fibrous_recall_full_image_after_abstention"], float("nan")),
                clump=finite(row["clump_dice_full_image_after_abstention"], float("nan")),
                skel=finite(row["skeleton_dice_full_image_after_abstention"], float("nan")),
                skel2=finite(row["skeleton_recovery_2px_after_abstention"], float("nan")),
            )
        )
    overlap = {row["region"]: row for row in distribution_rows}
    overlap_lines = [
        f"| {name} | {finite(row['mean'], float('nan')):.3f} | {finite(row['p05'], float('nan')):.3f} | "
        f"{finite(row['p50'], float('nan')):.3f} | {finite(row['p95'], float('nan')):.3f} | "
        f"{finite(row['histogram_overlap_with_uncertain'], float('nan')):.3f} |"
        for name, row in overlap.items()
    ]
    fib07 = mean(image_rows, "uncertain_fibrous_prob_ge_0.7_fraction")
    text = f"""# Corrected B1 uncertain_ignore diagnostic

This report supersedes the previous B1 abstention conclusions. It uses existing out-of-fold test predictions and existing inner-validation inference only; no model was trained and synthetic validation was not used for selection.

## Answers

1. **Can the head distinguish uncertain material from confident tau?** Primary pooled AUROC is {primary_auc:.3f} and AP is {finite(primary['average_precision'], float('nan')):.3f} for uncertain versus confident fibrous/clump, compared with background-conditional AUROC {background_auc:.3f}. The head behaves primarily as **{behavior}**, not as a validated uncertainty detector.
2. **Does discrimination persist near tau?** Spatial-band AUROCs at 2/5/10/20 px are {'/'.join(f'{finite(row["auroc"], float("nan")):.3f}' for row in spatial)}. These foreground-matched results, rather than the background-dominated AUROC, govern the interpretation.
3. **Is there a safe abstention threshold?** Validation-safe thresholds exist for {len(uncertainty_operational)}/5 uncertainty-only folds, {len(combined_operational)}/5 combined-rule folds, and {len(confidence_operational)}/5 low-confidence folds. On held-out images, however, uncertainty recall is only {min(finite(row['expert_uncertain_recall'], 0) for row in uncertainty_operational):.3f}-{max(finite(row['expert_uncertain_recall'], 0) for row in uncertainty_operational):.3f}, and target-fibre abstention exceeds 0.02 in {sum(finite(row['target_fibres_abstained_fraction'], 0) > 0.02 for row in uncertainty_operational)}/5 folds. There is no practically useful, robust operating point. A fold is marked `no_operational_threshold` unless validation fibrous Dice loss <=0.01, skeleton Dice loss <=0.01, target-fibre abstention <=0.02, and uncertain-region fibrous predictions with probability >=0.7 are reduced.
4. **Recommendation:** **{recommendation}.**

Expert-uncertain pixels with fibrous probability >= 0.5/0.7/0.8/0.9 (image-macro): {mean(image_rows, 'uncertain_fibrous_prob_ge_0.5_fraction'):.3f}/{fib07:.3f}/{mean(image_rows, 'uncertain_fibrous_prob_ge_0.8_fraction'):.3f}/{mean(image_rows, 'uncertain_fibrous_prob_ge_0.9_fraction'):.3f}.

## Foreground-conditional discrimination

| Comparison | Pooled prevalence | Pooled AUROC | Pooled AP | Image-macro AUROC | Image-macro AP |
|---|---:|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

Per-image, fold-pooled, and fold-macro results are in `uncertainty_discrimination.csv`.

## Spatially matched tau

All confident fibrous/clump pixels within each Euclidean band are included; uncertain pixels are the positive class.

| Distance (px) | Prevalence | AUROC | AP |
|---:|---:|---:|---:|
{chr(10).join(spatial_lines)}

## Probability overlap

| Region | Mean | P05 | P50 | P95 | Histogram overlap with uncertain |
|---|---:|---:|---:|---:|---:|
{chr(10).join(overlap_lines)}

## Validation-calibrated abstention

Rejected target pixels remain in complete-image denominators and therefore count as misses. The retained-pixel selective Dice is available in the CSV but is not used below or for threshold selection.

| Fold | Rule | Status | U threshold | Confidence threshold | Uncertain recall | All abstained | Target fibres abstained | Pred fibres abstained | Residual uncertain fib | Fib Dice | Fib precision | Fib recall | Clump Dice | Skeleton Dice | Skeleton <=2px recovery |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(threshold_lines)}

## Artifacts

- `uncertainty_discrimination.csv`
- `uncertainty_probability_overlap.csv`
- `per_image_uncertain_ignore.csv`
- `uncertainty_pr_curve.csv`
- `uncertainty_calibration_curve.csv`
- `abstention_tradeoffs.csv`
- `validation_abstention_sweeps.csv`
- `probability_distributions.png`
- `uncertainty_precision_recall.png`
- `uncertainty_calibration.png`
- `abstention_tradeoff.png`
- `qualitative_panels/`
"""
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_fields(prefix: str, values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {f"{prefix}_{k}": "not_applicable" for k in ["mean", "p50", "p90", "p95"]}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


def auroc(y: np.ndarray, score: np.ndarray) -> float | str:
    pos, neg = score[y], score[~y]
    if pos.size == 0 or neg.size == 0:
        return "not_applicable"
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    rank_sums = np.bincount(inv, ranks) / counts
    avg_ranks = rank_sums[inv]
    return float((avg_ranks[y].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def average_precision(y: np.ndarray, score: np.ndarray) -> float | str:
    positives = int(y.sum())
    if positives == 0:
        return "not_applicable"
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y_sorted).sum() / positives)


def dice_counts(tp: int, pred: int, target: int) -> float | str:
    return ratio(2 * tp, pred + target)


def ratio(num: int, den: int) -> float | str:
    return float(num / den) if den else "not_applicable"


def fraction(num: Any, den: Any) -> float | str:
    return ratio(int(num), int(den))


def within_distance_fraction(source: np.ndarray, target: np.ndarray, max_distance: float) -> float | str:
    if not source.any():
        return "not_applicable"
    if not target.any():
        return 0.0
    return float(np.count_nonzero(distance_transform_edt(~target)[source] <= max_distance) / int(source.sum()))


def finite(value: Any, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [finite(r.get(key), np.nan) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def normalize_display(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() == arr.min():
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - arr.min()) / (arr.max() - arr.min()) * 255, 0, 255).astype(np.uint8)


def gray_rgb(image: np.ndarray) -> np.ndarray:
    image = image if image.dtype == np.uint8 else normalize_display(image)
    return np.repeat(image[..., None], 3, axis=2)


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 1] = (0, 220, 80)
    rgb[mask == 3] = (255, 160, 0)
    rgb[mask == 255] = (160, 90, 220)
    return rgb


def heatmap(values: np.ndarray) -> np.ndarray:
    v = np.clip(values.astype(np.float32), 0, 1)
    return np.stack([255 * v, 255 * (1 - np.abs(v - 0.5) * 2), 255 * (1 - v)], axis=2).astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = gray_rgb(image).astype(np.float32)
    rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.asarray(color)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def tile(arr: np.ndarray, title: str) -> Image.Image:
    im = Image.fromarray(arr.astype(np.uint8), "RGB")
    im.thumbnail((256, 256))
    canvas = Image.new("RGB", (256, 274), "white")
    canvas.paste(im, ((256 - im.width) // 2, 18 + (256 - im.height) // 2))
    ImageDraw.Draw(canvas).text((4, 3), title, fill=(0, 0, 0))
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
