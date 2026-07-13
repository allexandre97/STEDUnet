#!/usr/bin/env python
"""Complete-image metrics and nested selection for uncertain-ignore experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from scripts.analyze_b1_uncertain_ignore import average_precision, auroc, dice_counts, ratio


SKELETON_THRESHOLDS = (0.5, 0.75, 0.85)
FIBROUS_THRESHOLDS = (0.5, 0.7, 0.8, 0.9)
BOUNDARY_BANDS = (("0_2", 0, 2), ("2_5", 2, 5), ("5_10", 5, 10), ("10_20", 10, 20), ("over_20", 20, math.inf))


def complete_image_metrics(
    target: np.ndarray,
    target_skeleton: np.ndarray,
    pred: dict[str, np.ndarray],
    rejection_threshold: float | None = None,
) -> dict[str, Any]:
    uncertain = target == 255
    valid = ~uncertain
    reject = (
        pred["uncertainty_probability"] > float(rejection_threshold)
        if rejection_threshold is not None else np.zeros(target.shape, dtype=bool)
    )
    pred_semantic = pred["semantic_class_map"]
    pred_fib_all = pred_semantic == 1
    pred_fib = pred_fib_all & valid & ~reject
    pred_clump = (pred_semantic == 3) & valid & ~reject
    target_fib = target == 1
    target_clump = target == 3
    fib_tp = int((pred_fib & target_fib).sum())
    clump_tp = int((pred_clump & target_clump).sum())
    metrics = {
        "fibrous_dice": dice_counts(fib_tp, int(pred_fib.sum()), int(target_fib.sum())),
        "fibrous_precision": ratio(fib_tp, int(pred_fib.sum())),
        "fibrous_recall": ratio(fib_tp, int(target_fib.sum())),
        "fibrous_area_ratio": ratio(int(pred_fib.sum()), int(target_fib.sum())),
        "clump_dice": (
            dice_counts(clump_tp, int(pred_clump.sum()), int(target_clump.sum()))
            if target_clump.any() else "not_applicable"
        ),
        "clump_applicable": bool(target_clump.any()),
        "fibrous_fraction_inside_uncertain_ignore": ratio(int((pred_fib_all & ~reject & uncertain).sum()), int(uncertain.sum())),
        "fibrous_leakage_into_clump": ratio(int((pred_fib & target_clump).sum()), int(target_clump.sum())),
        "rejected_all_fraction": ratio(int(reject.sum()), target.size),
        "rejected_target_fibres_fraction": ratio(int((reject & target_fib).sum()), int(target_fib.sum())),
        "rejected_predicted_fibres_fraction": ratio(int((reject & pred_fib_all).sum()), int(pred_fib_all.sum())),
        "expert_uncertain_precision": ratio(int((reject & uncertain).sum()), int(reject.sum())),
        "expert_uncertain_recall": ratio(int((reject & uncertain).sum()), int(uncertain.sum())),
        "expert_uncertain_auroc": auroc(uncertain.ravel(), pred["uncertainty_probability"].ravel()),
        "expert_uncertain_average_precision": average_precision(uncertain.ravel(), pred["uncertainty_probability"].ravel()),
    }
    for threshold in FIBROUS_THRESHOLDS:
        metrics[f"uncertain_fibrous_probability_ge_{threshold:g}_fraction"] = ratio(
            int((uncertain & ~reject & (pred["fibrous_probability"] >= threshold)).sum()), int(uncertain.sum())
        )
        metrics[f"uncertain_fibrous_probability_ge_{threshold:g}_fraction_raw"] = ratio(
            int((uncertain & (pred["fibrous_probability"] >= threshold)).sum()), int(uncertain.sum())
        )
    target_line = (target_skeleton > 0) & valid
    for threshold in SKELETON_THRESHOLDS:
        pred_line = (pred["skeleton_probability"] > threshold) & valid & ~reject
        tp = int((pred_line & target_line).sum())
        metrics[f"skeleton_dice_{threshold:g}"] = dice_counts(tp, int(pred_line.sum()), int(target_line.sum()))
        if threshold == 0.75:
            metrics["target_skeleton_recovered_2px"] = within_distance(target_line, pred_line, 2)
            metrics["predicted_skeleton_within_target_2px"] = within_distance(pred_line, target_line, 2)
    distance = distance_transform_edt(~uncertain) if uncertain.any() else np.full(target.shape, np.inf)
    for name, low, high in BOUNDARY_BANDS:
        band = target_fib & (distance > low if low else distance >= low) & (distance <= high)
        metrics[f"confident_fibrous_recall_{name}px"] = ratio(int((pred_fib & band).sum()), int(band.sum()))
    metrics.update({
        "_all_pixels": target.size,
        "_rejected_pixels": int(reject.sum()),
        "_target_fibres": int(target_fib.sum()),
        "_rejected_target_fibres": int((reject & target_fib).sum()),
        "_uncertain_pixels": int(uncertain.sum()),
        "_rejected_uncertain_pixels": int((reject & uncertain).sum()),
        "_clump_tp": clump_tp,
        "_predicted_clump": int(pred_clump.sum()),
        "_target_clump": int(target_clump.sum()),
    })
    return metrics


def summarize_images(rows: list[dict[str, Any]]) -> dict[str, Any]:
    public = sorted({key for row in rows for key in row if not key.startswith("_") and key not in {"sample_id", "outer_fold"}})
    summary = {key: numeric_mean(row.get(key) for row in rows) for key in public}
    summary.update({
        "image_count": len(rows),
        "rejected_all_fraction_pooled": pooled_ratio(rows, "_rejected_pixels", "_all_pixels"),
        "rejected_target_fibres_fraction_pooled": pooled_ratio(rows, "_rejected_target_fibres", "_target_fibres"),
        "expert_uncertain_recall_pooled": pooled_ratio(rows, "_rejected_uncertain_pixels", "_uncertain_pixels"),
        "clump_applicable_images": sum(bool(row["clump_applicable"]) for row in rows),
        "clump_dice_pooled": dice_counts(
            sum(row["_clump_tp"] for row in rows),
            sum(row["_predicted_clump"] for row in rows),
            sum(row["_target_clump"] for row in rows),
        ),
    })
    return summary


def selection_image_metrics(
    target: np.ndarray,
    target_skeleton: np.ndarray,
    pred: dict[str, np.ndarray],
    rejection_threshold: float | None = None,
) -> dict[str, Any]:
    uncertain = target == 255
    valid = ~uncertain
    reject = (
        pred["uncertainty_probability"] > float(rejection_threshold)
        if rejection_threshold is not None else np.zeros(target.shape, dtype=bool)
    )
    pred_fib = (pred["semantic_class_map"] == 1) & valid & ~reject
    target_fib = target == 1
    pred_skeleton = (pred["skeleton_probability"] > 0.75) & valid & ~reject
    target_line = (target_skeleton > 0) & valid
    return {
        "fibrous_dice": dice_counts(int((pred_fib & target_fib).sum()), int(pred_fib.sum()), int(target_fib.sum())),
        "skeleton_dice_0.75": dice_counts(
            int((pred_skeleton & target_line).sum()), int(pred_skeleton.sum()), int(target_line.sum())
        ),
        "uncertain_fibrous_probability_ge_0.7_fraction": ratio(
            int((uncertain & ~reject & (pred["fibrous_probability"] >= 0.7)).sum()), int(uncertain.sum())
        ),
        "uncertain_fibrous_probability_ge_0.9_fraction": ratio(
            int((uncertain & ~reject & (pred["fibrous_probability"] >= 0.9)).sum()), int(uncertain.sum())
        ),
        "_target_fibres": int(target_fib.sum()),
        "_rejected_target_fibres": int((reject & target_fib).sum()),
        "_uncertain_pixels": int(uncertain.sum()),
        "_rejected_uncertain_pixels": int((reject & uncertain).sum()),
    }


def summarize_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fibrous_dice": numeric_mean(row["fibrous_dice"] for row in rows),
        "skeleton_dice_0.75": numeric_mean(row["skeleton_dice_0.75"] for row in rows),
        "uncertain_fibrous_probability_ge_0.7_fraction": numeric_mean(
            row["uncertain_fibrous_probability_ge_0.7_fraction"] for row in rows
        ),
        "uncertain_fibrous_probability_ge_0.9_fraction": numeric_mean(
            row["uncertain_fibrous_probability_ge_0.9_fraction"] for row in rows
        ),
        "rejected_target_fibres_fraction_pooled": pooled_ratio(rows, "_rejected_target_fibres", "_target_fibres"),
        "expert_uncertain_recall_pooled": pooled_ratio(rows, "_rejected_uncertain_pixels", "_uncertain_pixels"),
    }


def select_s_candidate(baseline: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    acceptable = [row for row in candidates if (
        finite(row["fibrous_dice"]) >= finite(baseline["fibrous_dice"]) - 0.01
        and finite(row["skeleton_dice_0.75"]) >= finite(baseline["skeleton_dice_0.75"]) - 0.01
    )]
    if not acceptable:
        return {"selection": "U0", "reason": "no_acceptable_suppression_weight"}
    best = min(acceptable, key=lambda row: (
        finite(row["uncertain_fibrous_probability_ge_0.7_fraction"], math.inf),
        finite(row["uncertain_fibrous_probability_ge_0.9_fraction"], math.inf),
    ))
    if finite(best["uncertain_fibrous_probability_ge_0.7_fraction"]) >= finite(baseline["uncertain_fibrous_probability_ge_0.7_fraction"]):
        return {"selection": "U0", "reason": "no_validation_reduction"}
    return {"selection": "S", **best}


def select_h_operating_point(baseline: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    acceptable = [row for row in candidates if (
        finite(row["fibrous_dice"]) >= finite(baseline["fibrous_dice"]) - 0.01
        and finite(row["skeleton_dice_0.75"]) >= finite(baseline["skeleton_dice_0.75"]) - 0.01
        and finite(row["rejected_target_fibres_fraction_pooled"], math.inf) <= 0.02
        and finite(row["expert_uncertain_recall_pooled"]) > 0
    )]
    if not acceptable:
        return {"selection": "U0", "reason": "no_acceptable_rejection_operating_point"}
    best = max(acceptable, key=lambda row: (
        finite(row["expert_uncertain_recall_pooled"]),
        finite(baseline["uncertain_fibrous_probability_ge_0.7_fraction"])
        - finite(row["uncertain_fibrous_probability_ge_0.7_fraction"]),
    ))
    return {"selection": "H", **best}


def public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def within_distance(source: np.ndarray, target: np.ndarray, distance: float) -> float | str:
    if not source.any():
        return "not_applicable"
    if not target.any():
        return 0.0
    return float((distance_transform_edt(~target)[source] <= distance).mean())


def pooled_ratio(rows: list[dict[str, Any]], numerator: str, denominator: str) -> float | str:
    return ratio(sum(int(row[numerator]) for row in rows), sum(int(row[denominator]) for row in rows))


def numeric_mean(values: Any) -> float | str:
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return float(np.mean(numbers)) if numbers else "not_applicable"


def finite(value: Any, default: float = -math.inf) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else default
