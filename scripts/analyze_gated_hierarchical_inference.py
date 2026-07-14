#!/usr/bin/env python
"""Validation-calibrated hierarchical analysis of saved gated predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, maximum_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fibras.annotations import build_real_annotation_sample
from scripts.analyze_b1_uncertain_ignore import average_precision, auroc, load_probabilities, load_records
from scripts.evaluate_real_pilot_baseline import gray_rgb, real_mask_rgb
from scripts.uncertain_ignore_nested_metrics import BOUNDARY_BANDS, within_distance


SEMANTIC_GRID = np.linspace(0.0, 1.0, 101)
SKELETON_GRID = np.linspace(0.0, 1.0, 201)
CLASS_IDS = (0, 1, 3, 255)
SEMANTIC_METRICS = (
    "fibrous_dice", "fibrous_precision", "fibrous_recall", "fibrous_area_ratio", "clump_dice",
    "uncertain_precision", "uncertain_recall", "uncertain_dice",
    "uncertain_vs_confident_tau_auroc", "uncertain_vs_confident_tau_average_precision",
    "expert_uncertain_assigned_background", "expert_uncertain_assigned_fibre",
    "expert_uncertain_assigned_clump", "expert_uncertain_assigned_uncertain",
    "expert_confident_fibre_assigned_uncertain", "expert_background_assigned_uncertain",
    "fibrous_leakage_into_target_clump",
    *(f"confident_fibrous_recall_{name}px" for name, _, _ in BOUNDARY_BANDS),
)
SKELETON_METRICS = (
    "skeleton_dice", "skeleton_precision", "skeleton_recall", "target_skeleton_recovered_2px",
    "predicted_skeleton_within_target_2px", "skeleton_leakage_into_clump",
    "skeleton_leakage_into_uncertain",
)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    records = load_records(args.manifest, args.fold_manifest, args.image_root, args.annotation_root)
    selected_records = [record for record in records if record.partition in {"validation", "test"}]
    samples = {
        record.sample_id: build_real_annotation_sample(record.image_path, record.snakes_path, record.labels_path)
        for record in selected_records
    }
    gated = {(record.outer_fold, record.partition, record.sample_id): load_gated(gated_path(args, record))
             for record in selected_records}
    if args.panels_only:
        selections = json.loads((args.experiment_root / "selected_thresholds.json").read_text())
        semantics = {int(fold): value["semantic"] for fold, value in selections.items()}
        skeletons = {int(fold): value["skeleton"] for fold, value in selections.items()}
        write_panels(args.experiment_root / "hierarchical_qualitative_panels", records, samples, gated,
                     semantics, skeletons)
        return 0

    discrimination = gate_discrimination(selected_records, samples, gated)
    semantic_sweeps, semantic_selected = calibrate_semantics(args, records, samples, gated)
    skeleton_rows, skeleton_selected = calibrate_skeletons(
        records, samples, gated, semantic_selected
    )
    skeleton_rows.extend(evaluate_selected_skeletons(
        records, samples, gated, semantic_selected, skeleton_selected
    ))
    selections = {str(fold): {
        "semantic": semantic_selected[fold], "skeleton": skeleton_selected[fold],
        "selection_data": "inner_validation_only",
    } for fold in range(5)}
    outer_rows = outer_comparison(args, records, samples, gated, semantic_selected, skeleton_selected)
    macro = (
        [{"scope": "global_image_macro", **row} for row in summarize(outer_rows, ("variant",))]
        + [{"scope": "fold_image_macro", **row} for row in summarize(outer_rows, ("variant", "outer_fold"))]
    )
    paired = paired_comparisons(outer_rows)

    out = args.experiment_root
    write_csv(out / "gate_discrimination.csv", discrimination)
    write_csv(out / "validation_threshold_sweeps.csv", semantic_sweeps)
    write_json(out / "selected_thresholds.json", selections)
    write_csv(out / "hierarchical_outer_test_per_image.csv", outer_rows)
    write_json(out / "hierarchical_outer_test_macro.json", macro)
    write_csv(out / "skeleton_threshold_analysis.csv", skeleton_rows)
    write_csv(out / "paired_comparisons.csv", paired)
    write_panels(out / "hierarchical_qualitative_panels", records, samples, gated, semantic_selected, skeleton_selected)
    write_report(out / "hierarchical_gate_report.md", discrimination, semantic_selected,
                 skeleton_selected, skeleton_rows, macro)
    print(json.dumps({"semantic": semantic_selected, "skeleton": skeleton_selected}, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, default=Path("/ssd/STED_experiments/controlled_real_sted_v1"))
    parser.add_argument("--experiment-root", type=Path, default=Path(
        "/ssd/STED_experiments/controlled_real_sted_v1/gated_context_unet_v1"
    ))
    parser.add_argument("--manifest", type=Path, default=ROOT / "data_manifests/real_annotations.csv")
    parser.add_argument("--fold-manifest", type=Path, default=ROOT / "data_manifests/real_annotation_folds_v1.csv")
    parser.add_argument("--image-root", type=Path, default=Path("/ssd/STED_dataset/data"))
    parser.add_argument("--annotation-root", type=Path, default=Path("/cephfs/mhuang/STED_dataset/manual_annotations"))
    parser.add_argument("--panels-only", action="store_true")
    return parser


def gated_path(args: argparse.Namespace, record: Any) -> Path:
    return (args.experiment_root / "evaluations" / f"fold_{record.outer_fold}" / record.partition /
            record.sample_id / "predictions.npz")


def reference_path(args: argparse.Namespace, record: Any, variant: str) -> Path:
    if record.partition == "validation":
        if variant == "U0":
            base = args.suite_root / "analysis/b1_uncertain_ignore/validation_predictions"
        else:
            base = args.suite_root / "uncertain_ignore_nested_cv/evaluations/S_s0.05"
        return base / f"fold_{record.outer_fold}/validation" / record.sample_id / "predictions.npz" if variant == "S" else (
            base / f"fold_{record.outer_fold}" / record.sample_id / "predictions.npz"
        )
    if variant == "U0":
        return args.suite_root / "evaluations/b1_real_only" / f"fold_{record.outer_fold}" / record.sample_id / "predictions.npz"
    return (args.suite_root / "uncertain_ignore_nested_cv/evaluations/S_s0.05" /
            f"fold_{record.outer_fold}/test" / record.sample_id / "predictions.npz")


def load_gated(path: Path) -> dict[str, np.ndarray]:
    pred = load_probabilities(path)
    required = {
        "foreground_probability", "quantifiability_probability", "conditional_fibrous_probability",
        "conditional_clump_probability", "raw_centreline_probability", "final_skeleton_probability",
        "semantic_class_map",
    }
    missing = required - pred.keys()
    if missing:
        raise ValueError(f"{path}: missing gated arrays {sorted(missing)}")
    return pred


def hierarchical_classes(pred: dict[str, np.ndarray], t_foreground: float, t_gate: float) -> np.ndarray:
    """Apply q first, then g, then conditional morphology."""
    q = pred["foreground_probability"]
    g = pred["quantifiability_probability"]
    result = np.zeros(q.shape, dtype=np.uint8)
    foreground = q >= t_foreground
    result[foreground & (g < t_gate)] = 255
    confident = foreground & (g >= t_gate)
    fibre = pred["conditional_fibrous_probability"] >= pred["conditional_clump_probability"]
    result[confident & fibre] = 1
    result[confident & ~fibre] = 3
    return result


def raw_centreline_skeleton(
    semantic: np.ndarray, raw_probability: np.ndarray, threshold: float,
) -> np.ndarray:
    return (semantic == 1) & (raw_probability >= threshold)


def gate_discrimination(
    records: list[Any], samples: dict[str, dict[str, Any]],
    predictions: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for partition in ("validation", "test"):
        observations: dict[str, list[tuple[Any, np.ndarray, np.ndarray, float | str]]] = {}
        for record in (item for item in records if item.partition == partition):
            target = samples[record.sample_id]["real_semantic_mask"]
            pred = predictions[(record.outer_fold, partition, record.sample_id)]
            add_discrimination(observations, "q_background_vs_review_worthy", record,
                               target != 0, pred["foreground_probability"], np.ones(target.shape, bool))
            tau_or_uncertain = np.isin(target, (1, 3, 255))
            add_discrimination(observations, "one_minus_g_uncertain_vs_confident_tau", record,
                               target == 255, 1.0 - pred["quantifiability_probability"], tau_or_uncertain)
            uncertain = target == 255
            distance = distance_transform_edt(~uncertain) if uncertain.any() else np.full(target.shape, np.inf)
            confident_tau = np.isin(target, (1, 3))
            for radius in (2, 5, 10, 20):
                mask = uncertain | (confident_tau & (distance <= radius))
                add_discrimination(observations, f"one_minus_g_spatial_tau_{radius}px", record,
                                   uncertain, 1.0 - pred["quantifiability_probability"], mask)
            morphology = np.isin(target, (1, 3))
            add_discrimination(observations, "conditional_morphology_fibre_vs_clump", record,
                               target == 1, pred["conditional_fibrous_probability"], morphology,
                               accuracy=float(np.mean(
                                   (pred["conditional_fibrous_probability"][morphology] >=
                                    pred["conditional_clump_probability"][morphology]) ==
                                   (target[morphology] == 1)
                               )) if morphology.any() else "not_applicable")
        for comparison, parts in observations.items():
            image_rows = [discrimination_row(partition, comparison, "image", record.sample_id,
                                              record.outer_fold, y, score, accuracy)
                          for record, y, score, accuracy in parts]
            rows.extend(image_rows)
            rows.append(aggregate_discrimination(partition, comparison, "global_image_macro", "all", image_rows))
            rows.append(pooled_discrimination(partition, comparison, "global_pooled", "all", parts))
            for fold in range(5):
                fold_rows = [row for row in image_rows if row["outer_fold"] == fold]
                fold_parts = [part for part in parts if part[0].outer_fold == fold]
                rows.append(aggregate_discrimination(partition, comparison, "fold_image_macro", fold, fold_rows))
                rows.append(pooled_discrimination(partition, comparison, "fold_pooled", fold, fold_parts))
    return rows


def add_discrimination(
    groups: dict[str, list[tuple[Any, np.ndarray, np.ndarray, float | str]]], comparison: str,
    record: Any, target: np.ndarray, score: np.ndarray, mask: np.ndarray,
    accuracy: float | str = "not_applicable",
) -> None:
    groups.setdefault(comparison, []).append((record, target[mask].ravel(), score[mask].ravel(), accuracy))


def discrimination_row(
    partition: str, comparison: str, scope: str, sample_id: str, fold: int | str,
    target: np.ndarray, score: np.ndarray, accuracy: float | str,
) -> dict[str, Any]:
    return {
        "partition": partition, "comparison": comparison, "scope": scope, "sample_id": sample_id,
        "outer_fold": fold, "positive_pixels": int(target.sum()),
        "negative_pixels": int(target.size - target.sum()),
        "prevalence": ratio(int(target.sum()), int(target.size)), "auroc": auroc(target, score),
        "average_precision": average_precision(target, score), "accuracy": accuracy,
    }


def aggregate_discrimination(
    partition: str, comparison: str, scope: str, fold: int | str, rows: list[dict[str, Any]],
) -> dict[str, Any]:
    positive = sum(row["positive_pixels"] for row in rows)
    negative = sum(row["negative_pixels"] for row in rows)
    return {
        "partition": partition, "comparison": comparison, "scope": scope, "sample_id": "all",
        "outer_fold": fold, "positive_pixels": positive, "negative_pixels": negative,
        "prevalence": ratio(positive, positive + negative),
        "auroc": numeric_mean(row["auroc"] for row in rows),
        "average_precision": numeric_mean(row["average_precision"] for row in rows),
        "accuracy": numeric_mean(row["accuracy"] for row in rows),
    }


def pooled_discrimination(
    partition: str, comparison: str, scope: str, fold: int | str,
    parts: list[tuple[Any, np.ndarray, np.ndarray, float | str]],
) -> dict[str, Any]:
    if not parts:
        return discrimination_row(partition, comparison, scope, "all", fold,
                                  np.array([], bool), np.array([], float), "not_applicable")
    target = np.concatenate([part[1] for part in parts])
    score = np.concatenate([part[2] for part in parts])
    accurate = [(float(part[3]), part[1].size) for part in parts if is_number(part[3])]
    accuracy = (sum(value * pixels for value, pixels in accurate) / sum(pixels for _, pixels in accurate)
                if accurate else "not_applicable")
    return discrimination_row(partition, comparison, scope, "all", fold, target, score, accuracy)


def calibrate_semantics(
    args: argparse.Namespace, records: list[Any], samples: dict[str, dict[str, Any]],
    predictions: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    sweep_rows, selected = [], {}
    for fold in range(5):
        fold_records = validation_records(records, fold)
        matrices = [semantic_grid_metrics(
            samples[record.sample_id]["real_semantic_mask"],
            predictions[(fold, "validation", record.sample_id)],
        ) for record in fold_records]
        aggregate = {key: nanmean_stack(item[key] for item in matrices) for key in matrices[0]}
        baseline_rows = [complete_metrics(
            samples[record.sample_id]["real_semantic_mask"],
            samples[record.sample_id]["real_skeleton_mask"],
            load_probabilities(reference_path(args, record, "U0"))["semantic_class_map"],
        ) for record in fold_records]
        baseline = {key: numeric_mean(row[key] for row in baseline_rows) for key in (
            "fibrous_dice", "clump_dice", "confident_fibrous_recall_0_2px"
        )}
        constraints = semantic_constraints(aggregate, baseline)
        feasible = np.logical_and.reduce(list(constraints.values()))
        for i, tq in enumerate(SEMANTIC_GRID):
            for j, tg in enumerate(SEMANTIC_GRID):
                sweep_rows.append({
                    "outer_fold": fold, "t_foreground": float(tq), "t_gate": float(tg),
                    **{key: scalar(matrix[i, j]) for key, matrix in aggregate.items()},
                    "u0_validation_fibrous_dice": baseline["fibrous_dice"],
                    "u0_validation_clump_dice": baseline["clump_dice"],
                    "u0_validation_boundary_recall_0_2px": baseline["confident_fibrous_recall_0_2px"],
                    "constraints_satisfied": bool(feasible[i, j]),
                })
        selected[fold] = select_semantic_thresholds(aggregate, feasible, baseline, constraints)
    return sweep_rows, selected


def validation_records(records: Iterable[Any], fold: int) -> list[Any]:
    """Return the only records permitted to influence a fold's thresholds."""
    return [record for record in records if record.outer_fold == fold and record.partition == "validation"]


def semantic_constraints(
    metrics: dict[str, np.ndarray], baseline: dict[str, Any],
) -> dict[str, np.ndarray]:
    constraints = {
        "fibrous_dice": metrics["fibrous_dice"] >= float(baseline["fibrous_dice"]) - 0.01,
        "fibre_assigned_uncertain": metrics["fibre_assigned_uncertain"] <= 0.02,
        "background_assigned_uncertain": metrics["background_assigned_uncertain"] <= 0.01,
        "finite_uncertain_recall": np.isfinite(metrics["uncertain_recall"]),
    }
    if is_number(baseline["clump_dice"]):
        constraints["clump_dice"] = metrics["clump_dice"] >= float(baseline["clump_dice"]) - 0.01
    if is_number(baseline["confident_fibrous_recall_0_2px"]):
        constraints["boundary_recall_0_2px"] = (
            metrics["boundary_recall_0_2px"] >= float(baseline["confident_fibrous_recall_0_2px"]) - 0.01
        )
    return constraints


def select_semantic_thresholds(
    metrics: dict[str, np.ndarray], feasible: np.ndarray, baseline: dict[str, Any],
    constraints: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    candidates = np.argwhere(feasible)
    if not len(candidates):
        result = {"status": "no_valid_operating_point", "validation_u0": baseline,
                  "selected_from": "inner_validation"}
        if constraints:
            satisfied = sum(mask.astype(np.int16) for mask in constraints.values())
            best_count = int(satisfied.max())
            near = np.argwhere(satisfied == best_count)
            best = max(near, key=lambda ij: finite_or(metrics["uncertain_recall"][tuple(ij)]))
            i, j = map(int, best)
            result["constraint_diagnostics"] = {
                "total_constraints": len(constraints), "maximum_constraints_satisfied": best_count,
                "pairs_satisfying_each_constraint": {key: int(mask.sum()) for key, mask in constraints.items()},
                "diagnostic_nearest_pair_not_selected": {
                    "t_foreground": float(SEMANTIC_GRID[i]), "t_gate": float(SEMANTIC_GRID[j]),
                    "satisfied": [key for key, mask in constraints.items() if mask[i, j]],
                    "failed": [key for key, mask in constraints.items() if not mask[i, j]],
                    "metrics": {key: scalar(value[i, j]) for key, value in metrics.items()},
                },
            }
        return result
    best = max(candidates, key=lambda ij: (
        metrics["uncertain_recall"][tuple(ij)], metrics["fibrous_dice"][tuple(ij)],
        finite_or(metrics["clump_dice"][tuple(ij)]), -SEMANTIC_GRID[ij[0]], SEMANTIC_GRID[ij[1]],
    ))
    i, j = map(int, best)
    return {
        "status": "selected", "selected_from": "inner_validation",
        "t_foreground": float(SEMANTIC_GRID[i]), "t_gate": float(SEMANTIC_GRID[j]),
        "validation_u0": baseline,
        "validation_metrics": {key: scalar(value[i, j]) for key, value in metrics.items()},
    }


def semantic_grid_metrics(target: np.ndarray, pred: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    morphology_fibre = pred["conditional_fibrous_probability"] >= pred["conditional_clump_probability"]
    assignments = {}
    for class_id in CLASS_IDS:
        mask = target == class_id
        assignments[class_id] = grid_assignments(
            pred["foreground_probability"], pred["quantifiability_probability"], morphology_fibre, mask
        )
    valid_classes = (0, 1, 3)
    fib_tp = assignments[1][1]
    fib_pred = sum(assignments[class_id][1] for class_id in valid_classes)
    fib_target = int((target == 1).sum())
    clump_tp = assignments[3][3]
    clump_pred = sum(assignments[class_id][3] for class_id in valid_classes)
    clump_target = int((target == 3).sum())
    uncertain_target = int((target == 255).sum())
    boundary = boundary_mask(target, 0, 2)
    boundary_fibre = grid_assignments(
        pred["foreground_probability"], pred["quantifiability_probability"], morphology_fibre, boundary
    )[1]
    return {
        "fibrous_dice": divide(2 * fib_tp, fib_pred + fib_target),
        "fibrous_precision": divide(fib_tp, fib_pred), "fibrous_recall": divide(fib_tp, fib_target),
        "clump_dice": (divide(2 * clump_tp, clump_pred + clump_target) if clump_target
                       else np.full(fib_tp.shape, np.nan)),
        "uncertain_recall": divide(assignments[255][255], uncertain_target),
        "fibre_assigned_uncertain": divide(assignments[1][255], fib_target),
        "background_assigned_uncertain": divide(assignments[0][255], int((target == 0).sum())),
        "boundary_recall_0_2px": divide(boundary_fibre, int(boundary.sum())),
    }


def grid_assignments(
    q: np.ndarray, g: np.ndarray, morphology_fibre: np.ndarray, mask: np.ndarray,
) -> dict[int, np.ndarray]:
    hist_fibre = score_histogram(q, g, mask & morphology_fibre)
    hist_clump = score_histogram(q, g, mask & ~morphology_fibre)
    hist = hist_fibre + hist_clump
    background = np.broadcast_to(np.r_[0, np.cumsum(hist.sum(axis=1))[:-1]][:, None], hist.shape).copy()
    uncertain = q_ge_g_lt(hist)
    return {0: background, 255: uncertain, 1: q_ge_g_ge(hist_fibre), 3: q_ge_g_ge(hist_clump)}


def score_histogram(q: np.ndarray, g: np.ndarray, mask: np.ndarray) -> np.ndarray:
    q_bin = np.searchsorted(SEMANTIC_GRID, q[mask], side="right") - 1
    g_bin = np.searchsorted(SEMANTIC_GRID, g[mask], side="right") - 1
    hist = np.zeros((SEMANTIC_GRID.size, SEMANTIC_GRID.size), dtype=np.int64)
    np.add.at(hist, (np.clip(q_bin, 0, 100), np.clip(g_bin, 0, 100)), 1)
    return hist


def q_ge_g_lt(hist: np.ndarray) -> np.ndarray:
    q_reverse = np.cumsum(hist[::-1], axis=0)[::-1]
    result = np.zeros_like(hist)
    result[:, 1:] = np.cumsum(q_reverse, axis=1)[:, :-1]
    return result


def q_ge_g_ge(hist: np.ndarray) -> np.ndarray:
    return np.cumsum(np.cumsum(hist[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]


def calibrate_skeletons(
    records: list[Any], samples: dict[str, dict[str, Any]],
    predictions: dict[tuple[int, str, str], dict[str, np.ndarray]],
    semantics: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows, selected = [], {}
    for fold in range(5):
        fold_records = validation_records(records, fold)
        candidates = []
        for strategy in ("joint_final_probability", "hierarchical_raw_centreline"):
            if strategy.startswith("hierarchical") and semantics[fold]["status"] != "selected":
                continue
            image_curves = []
            for record in fold_records:
                target = samples[record.sample_id]["real_semantic_mask"]
                target_line = samples[record.sample_id]["real_skeleton_mask"] > 0
                pred = predictions[(fold, "validation", record.sample_id)]
                semantic = selected_semantic(pred, semantics[fold])
                probability = (pred["final_skeleton_probability"] if strategy.startswith("joint")
                               else pred["raw_centreline_probability"])
                mask = np.ones(target.shape, bool) if strategy.startswith("joint") else semantic == 1
                curve = skeleton_curve(target, target_line, probability, mask)
                image_curves.append(curve)
            macro = {key: nanmean_stack(curve[key] for curve in image_curves) for key in SKELETON_METRICS}
            for index, threshold in enumerate(SKELETON_GRID):
                rows.append({
                    "partition": "validation", "scope": "fold_image_macro", "outer_fold": fold,
                    "strategy": strategy, "threshold": float(threshold),
                    **{key: scalar(value[index]) for key, value in macro.items()},
                })
            valid = np.flatnonzero(np.isfinite(macro["skeleton_dice"]))
            best = max(valid, key=lambda index: (
                macro["skeleton_dice"][index], macro["target_skeleton_recovered_2px"][index],
                macro["skeleton_precision"][index], -SKELETON_GRID[index],
            ))
            candidates.append({
                "strategy": strategy, "threshold": float(SKELETON_GRID[best]),
                "validation_metrics": {key: scalar(value[best]) for key, value in macro.items()},
            })
        winner = max(candidates, key=lambda item: (
            item["validation_metrics"]["skeleton_dice"],
            item["validation_metrics"]["target_skeleton_recovered_2px"],
        ))
        selected[fold] = {"status": "selected", "selected_from": "inner_validation",
                          "candidates": candidates, **winner}
    return rows, selected


def skeleton_curve(
    target: np.ndarray, target_line: np.ndarray, probability: np.ndarray, semantic_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    true_line = target_line & (target == 1)
    values = np.where(semantic_mask, probability, -np.inf)
    predicted = count_at_least(values[semantic_mask], SKELETON_GRID)
    true_positive = count_at_least(values[true_line], SKELETON_GRID)
    target_count = int(true_line.sum())
    disk = np.fromfunction(lambda y, x: (y - 2) ** 2 + (x - 2) ** 2 <= 4, (5, 5), dtype=int)
    recovered_score = maximum_filter(values, footprint=disk, mode="constant", cval=-np.inf)
    recovered = count_at_least(recovered_score[true_line], SKELETON_GRID)
    near_target = distance_transform_edt(~true_line) <= 2 if true_line.any() else np.zeros(target.shape, bool)
    predicted_near = count_at_least(values[semantic_mask & near_target], SKELETON_GRID)
    clump = count_at_least(values[semantic_mask & (target == 3)], SKELETON_GRID)
    uncertain = count_at_least(values[semantic_mask & (target == 255)], SKELETON_GRID)
    return {
        "skeleton_dice": divide(2 * true_positive, predicted + target_count),
        "skeleton_precision": divide(true_positive, predicted),
        "skeleton_recall": divide(true_positive, target_count),
        "target_skeleton_recovered_2px": divide(recovered, target_count),
        "predicted_skeleton_within_target_2px": divide(predicted_near, predicted),
        "skeleton_leakage_into_clump": divide(clump, predicted),
        "skeleton_leakage_into_uncertain": divide(uncertain, predicted),
    }


def count_at_least(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=np.float32).ravel())
    return sorted_values.size - np.searchsorted(sorted_values, thresholds, side="left")


def selected_semantic(pred: dict[str, np.ndarray], selection: dict[str, Any]) -> np.ndarray | None:
    if selection["status"] != "selected":
        return None
    return hierarchical_classes(pred, selection["t_foreground"], selection["t_gate"])


def selected_skeleton(
    pred: dict[str, np.ndarray], semantic: np.ndarray, selection: dict[str, Any],
) -> np.ndarray:
    if selection["strategy"] == "joint_final_probability":
        return pred["final_skeleton_probability"] >= selection["threshold"]
    return raw_centreline_skeleton(semantic, pred["raw_centreline_probability"], selection["threshold"])


def evaluate_selected_skeletons(
    records: list[Any], samples: dict[str, dict[str, Any]],
    predictions: dict[tuple[int, str, str], dict[str, np.ndarray]],
    semantics: dict[int, dict[str, Any]], selections: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for fold in range(5):
        for candidate in selections[fold]["candidates"]:
            strategy, threshold = candidate["strategy"], candidate["threshold"]
            image_rows = []
            for record in (item for item in records if item.outer_fold == fold and item.partition == "test"):
                pred = predictions[(fold, "test", record.sample_id)]
                semantic = selected_semantic(pred, semantics[fold])
                if strategy.startswith("hierarchical") and semantic is None:
                    continue
                if strategy.startswith("joint"):
                    predicted = pred["final_skeleton_probability"] >= threshold
                else:
                    predicted = raw_centreline_skeleton(semantic, pred["raw_centreline_probability"], threshold)
                target = samples[record.sample_id]["real_semantic_mask"]
                target_line = (samples[record.sample_id]["real_skeleton_mask"] > 0) & (target == 1)
                metrics = skeleton_metrics(target, target_line, predicted)
                row = {
                    "partition": "test", "scope": "image", "outer_fold": fold,
                    "sample_id": record.sample_id, "strategy": strategy, "threshold": threshold,
                    "selected_strategy": strategy == selections[fold]["strategy"], **metrics,
                }
                rows.append(row)
                image_rows.append(row)
            if image_rows:
                rows.append({
                    "partition": "test", "scope": "fold_image_macro", "outer_fold": fold,
                    "sample_id": "all", "strategy": strategy, "threshold": threshold,
                    "selected_strategy": strategy == selections[fold]["strategy"],
                    **{key: numeric_mean(row[key] for row in image_rows) for key in SKELETON_METRICS},
                })
    for strategy in ("joint_final_probability", "hierarchical_raw_centreline"):
        selected_rows = [row for row in rows if row["scope"] == "image" and row["strategy"] == strategy]
        if selected_rows:
            rows.append({
                "partition": "test", "scope": "global_image_macro", "outer_fold": "all",
                "sample_id": "all", "strategy": strategy, "threshold": "fold_specific_validation_selected",
                "selected_strategy": all(row["selected_strategy"] for row in selected_rows),
                **{key: numeric_mean(row[key] for row in selected_rows) for key in SKELETON_METRICS},
            })
    return rows


def outer_comparison(
    args: argparse.Namespace, records: list[Any], samples: dict[str, dict[str, Any]],
    gated: dict[tuple[int, str, str], dict[str, np.ndarray]], semantics: dict[int, dict[str, Any]],
    skeletons: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for record in (item for item in records if item.partition == "test"):
        target = samples[record.sample_id]["real_semantic_mask"]
        target_line = samples[record.sample_id]["real_skeleton_mask"]
        for variant in ("U0", "S"):
            pred = load_probabilities(reference_path(args, record, variant))
            rows.append(metric_row(record, variant, target, target_line, pred["semantic_class_map"],
                                   pred["uncertainty_probability"], pred["skeleton_probability"] >= 0.75))
        pred = gated[(record.outer_fold, "test", record.sample_id)]
        gate_score = 1.0 - pred["quantifiability_probability"]
        rows.append(metric_row(record, "G_four_way_argmax", target, target_line, pred["semantic_class_map"],
                               gate_score, pred["final_skeleton_probability"] >= 0.75))
        semantic = selected_semantic(pred, semantics[record.outer_fold])
        if semantic is None:
            rows.extend(unavailable_rows(record, ("G_hierarchical", "G_hierarchical_calibrated_skeleton")))
            continue
        rows.append(metric_row(record, "G_hierarchical", target, target_line, semantic, gate_score,
                               pred["final_skeleton_probability"] >= 0.75))
        rows.append(metric_row(record, "G_hierarchical_calibrated_skeleton", target, target_line, semantic,
                               gate_score, selected_skeleton(pred, semantic, skeletons[record.outer_fold])))
    return rows


def metric_row(
    record: Any, variant: str, target: np.ndarray, target_line: np.ndarray, semantic: np.ndarray,
    uncertainty_score: np.ndarray, predicted_line: np.ndarray,
) -> dict[str, Any]:
    return {"variant": variant, "outer_fold": record.outer_fold, "sample_id": record.sample_id,
            "status": "available", **complete_metrics(target, target_line, semantic,
                                                        uncertainty_score, predicted_line)}


def unavailable_rows(record: Any, variants: Iterable[str]) -> list[dict[str, Any]]:
    return [{"variant": variant, "outer_fold": record.outer_fold, "sample_id": record.sample_id,
             "status": "no_valid_validation_operating_point"} for variant in variants]


def complete_metrics(
    target: np.ndarray, target_line: np.ndarray, semantic: np.ndarray,
    uncertainty_score: np.ndarray | None = None, predicted_line: np.ndarray | None = None,
) -> dict[str, Any]:
    valid = target != 255
    target_fibre, target_clump, target_uncertain = target == 1, target == 3, target == 255
    pred_fibre, pred_clump, pred_uncertain = semantic == 1, semantic == 3, semantic == 255
    fib_valid, clump_valid = pred_fibre & valid, pred_clump & valid
    fib_tp, clump_tp = int((fib_valid & target_fibre).sum()), int((clump_valid & target_clump).sum())
    uncertain_tp = int((pred_uncertain & target_uncertain).sum())
    metrics = {
        "fibrous_dice": dice(fib_tp, int(fib_valid.sum()), int(target_fibre.sum())),
        "fibrous_precision": ratio(fib_tp, int(fib_valid.sum())),
        "fibrous_recall": ratio(fib_tp, int(target_fibre.sum())),
        "fibrous_area_ratio": ratio(int(fib_valid.sum()), int(target_fibre.sum())),
        "clump_dice": (dice(clump_tp, int(clump_valid.sum()), int(target_clump.sum()))
                       if target_clump.any() else "not_applicable"),
        "uncertain_precision": ratio(uncertain_tp, int(pred_uncertain.sum())),
        "uncertain_recall": ratio(uncertain_tp, int(target_uncertain.sum())),
        "uncertain_dice": dice(uncertain_tp, int(pred_uncertain.sum()), int(target_uncertain.sum())),
        "expert_uncertain_assigned_background": assignment(semantic, target_uncertain, 0),
        "expert_uncertain_assigned_fibre": assignment(semantic, target_uncertain, 1),
        "expert_uncertain_assigned_clump": assignment(semantic, target_uncertain, 3),
        "expert_uncertain_assigned_uncertain": assignment(semantic, target_uncertain, 255),
        "expert_confident_fibre_assigned_uncertain": assignment(semantic, target_fibre, 255),
        "expert_background_assigned_uncertain": assignment(semantic, target == 0, 255),
        "fibrous_leakage_into_target_clump": ratio(
            int((pred_fibre & target_clump).sum()), int(target_clump.sum())
        ),
    }
    if uncertainty_score is not None:
        tau_mask = target_uncertain | target_fibre | target_clump
        metrics["uncertain_vs_confident_tau_auroc"] = auroc(target_uncertain[tau_mask], uncertainty_score[tau_mask])
        metrics["uncertain_vs_confident_tau_average_precision"] = average_precision(
            target_uncertain[tau_mask], uncertainty_score[tau_mask]
        )
    for name, low, high in BOUNDARY_BANDS:
        band = boundary_mask(target, low, high)
        metrics[f"confident_fibrous_recall_{name}px"] = ratio(int((pred_fibre & band).sum()), int(band.sum()))
    if predicted_line is not None:
        metrics.update(skeleton_metrics(target, (target_line > 0) & target_fibre, predicted_line))
    return metrics


def skeleton_metrics(target: np.ndarray, target_line: np.ndarray, predicted_line: np.ndarray) -> dict[str, Any]:
    tp = int((target_line & predicted_line).sum())
    predicted = int(predicted_line.sum())
    return {
        "skeleton_dice": dice(tp, predicted, int(target_line.sum())),
        "skeleton_precision": ratio(tp, predicted), "skeleton_recall": ratio(tp, int(target_line.sum())),
        "target_skeleton_recovered_2px": within_distance(target_line, predicted_line, 2),
        "predicted_skeleton_within_target_2px": within_distance(predicted_line, target_line, 2),
        "skeleton_leakage_into_clump": ratio(int((predicted_line & (target == 3)).sum()), predicted),
        "skeleton_leakage_into_uncertain": ratio(int((predicted_line & (target == 255)).sum()), predicted),
    }


def boundary_mask(target: np.ndarray, low: float, high: float) -> np.ndarray:
    uncertain = target == 255
    distance = distance_transform_edt(~uncertain) if uncertain.any() else np.full(target.shape, np.inf)
    return (target == 1) & (distance > low if low else distance >= low) & (distance <= high)


def assignment(semantic: np.ndarray, target_region: np.ndarray, class_id: int) -> float | str:
    return ratio(int(((semantic == class_id) & target_region).sum()), int(target_region.sum()))


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    metrics = (*SEMANTIC_METRICS, *SKELETON_METRICS)
    return [{**dict(zip(keys, group)), "image_count": len(selected),
             "available_images": sum(row["status"] == "available" for row in selected),
             **{metric: numeric_mean(row.get(metric) for row in selected) for metric in metrics}}
            for group, selected in sorted(groups.items())]


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["variant"], row["sample_id"]): row for row in rows}
    samples = sorted({row["sample_id"] for row in rows})
    output = []
    for variant in ("G_four_way_argmax", "G_hierarchical", "G_hierarchical_calibrated_skeleton"):
        for reference in ("U0", "S"):
            for metric in (*SEMANTIC_METRICS, *SKELETON_METRICS):
                differences = []
                for sample_id in samples:
                    value = lookup[(variant, sample_id)].get(metric)
                    baseline = lookup[(reference, sample_id)].get(metric)
                    if is_number(value) and is_number(baseline):
                        differences.append(float(value) - float(baseline))
                output.append({
                    "comparison": f"{variant}-minus-{reference}", "metric": metric,
                    "mean_paired_difference": float(np.mean(differences)) if differences else "not_applicable",
                    "applicable_images": len(differences),
                })
    return output


def write_panels(
    out: Path, records: list[Any], samples: dict[str, dict[str, Any]],
    predictions: dict[tuple[int, str, str], dict[str, np.ndarray]], semantics: dict[int, dict[str, Any]],
    skeletons: dict[int, dict[str, Any]],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for record in (item for item in records if item.partition == "test"):
        sample = samples[record.sample_id]
        pred = predictions[(record.outer_fold, "test", record.sample_id)]
        semantic = selected_semantic(pred, semantics[record.outer_fold])
        if semantic is None:
            semantic = pred["semantic_class_map"]
            line = selected_skeleton(pred, semantic, skeletons[record.outer_fold])
            note = "no valid hierarchical point"
        else:
            line = selected_skeleton(pred, semantic, skeletons[record.outer_fold])
            note = "hierarchical + calibrated skeleton"
        tiles = [
            labeled(gray_rgb(sample["image_uint8"]), "raw STED"),
            labeled(real_mask_rgb(sample["real_semantic_mask"]), "expert"),
            labeled(real_mask_rgb(pred["semantic_class_map"]), "four-way argmax"),
            labeled(real_mask_rgb(semantic), note),
            labeled(heatmap(pred["foreground_probability"]), "q"),
            labeled(heatmap(1 - pred["quantifiability_probability"]), "1-g"),
            labeled(heatmap(pred["conditional_fibrous_probability"]), "r fibre"),
            labeled(heatmap(pred["raw_centreline_probability"]), "raw centreline"),
            labeled(binary_rgb(line), "selected skeleton"),
        ]
        panel = Image.new("RGB", (3 * tiles[0].width, 3 * tiles[0].height), "white")
        for index, tile in enumerate(tiles):
            panel.paste(tile, ((index % 3) * tile.width, (index // 3) * tile.height))
        panel.save(out / f"fold_{record.outer_fold}_{record.sample_id}.png")


def heatmap(values: np.ndarray) -> Image.Image:
    scaled = np.clip(values, 0, 1)
    rgb = np.stack([scaled, np.sqrt(scaled) * (1 - scaled), 1 - scaled], axis=-1)
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255))


def binary_rgb(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(values) * 255).convert("RGB")


def labeled(image: Image.Image, text: str) -> Image.Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    image = image.convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 18), "white")
    canvas.paste(image, (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), text, fill="black")
    return canvas


def write_report(
    path: Path, discrimination: list[dict[str, Any]], semantics: dict[int, dict[str, Any]],
    skeletons: dict[int, dict[str, Any]], skeleton_rows: list[dict[str, Any]],
    macro: list[dict[str, Any]],
) -> None:
    disc = {(row["partition"], row["comparison"], row["scope"]): row for row in discrimination
            if row["outer_fold"] == "all"}
    values = {row["variant"]: row for row in macro if row["scope"] == "global_image_macro"}
    valid_folds = sum(selection["status"] == "selected" for selection in semantics.values())
    lines = [
        "# Hierarchical gate analysis", "", "## Validation-only selection", "",
        f"Valid hierarchical operating points: {valid_folds}/5 folds.", "",
        "| Fold | Semantic status | t foreground | t gate | Skeleton strategy | t skeleton |", "|---:|---|---:|---:|---|---:|",
    ]
    for fold in range(5):
        sem, skel = semantics[fold], skeletons[fold]
        lines.append(f"| {fold} | {sem['status']} | {format_number(sem.get('t_foreground'))} | "
                     f"{format_number(sem.get('t_gate'))} | {skel['strategy']} | {skel['threshold']:.3f} |")
    lines += ["", "## Gate discrimination", "",
              "| Partition | Comparison | AUROC pooled | AP pooled | AUROC macro | AP macro |", "|---|---|---:|---:|---:|---:|"]
    for partition in ("validation", "test"):
        for comparison in ("q_background_vs_review_worthy", "one_minus_g_uncertain_vs_confident_tau",
                           "conditional_morphology_fibre_vs_clump"):
            pooled = disc[(partition, comparison, "global_pooled")]
            image_macro = disc[(partition, comparison, "global_image_macro")]
            lines.append(f"| {partition} | {comparison} | {format_number(pooled['auroc'])} | "
                         f"{format_number(pooled['average_precision'])} | {format_number(image_macro['auroc'])} | "
                         f"{format_number(image_macro['average_precision'])} |")
    lines += ["", "## Outer-test macro comparison", "",
              "| Variant | Fibre Dice | Clump Dice | Uncertain recall | Skeleton Dice | Skeleton recovery 2px |",
              "|---|---:|---:|---:|---:|---:|"]
    for variant in ("U0", "S", "G_four_way_argmax", "G_hierarchical", "G_hierarchical_calibrated_skeleton"):
        row = values[variant]
        lines.append(f"| {variant} | {format_number(row['fibrous_dice'])} | {format_number(row['clump_dice'])} | "
                     f"{format_number(row['uncertain_recall'])} | {format_number(row['skeleton_dice'])} | "
                     f"{format_number(row['target_skeleton_recovered_2px'])} |")
    lines += ["", "## Skeleton calibration", "",
              "| Strategy | Outer-test Dice | Precision | Recall | Recovery 2px | Predicted within 2px |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in skeleton_rows:
        if row.get("partition") == "test" and row.get("scope") == "global_image_macro":
            lines.append(f"| {row['strategy']} | {format_number(row['skeleton_dice'])} | "
                         f"{format_number(row['skeleton_precision'])} | {format_number(row['skeleton_recall'])} | "
                         f"{format_number(row['target_skeleton_recovered_2px'])} | "
                         f"{format_number(row['predicted_skeleton_within_target_2px'])} |")
    lines += ["", "## Diagnosis", ""]
    if valid_folds == 5:
        lines.append("All folds have a validation-valid hierarchical operating point; outer-test metrics determine viability.")
    else:
        lines.append(
            f"Only {valid_folds}/5 folds have a validation-valid hierarchical operating point. "
            "Hierarchical threshold calibration therefore cannot rescue the existing checkpoints."
        )
        for fold in range(5):
            diagnostics = semantics[fold].get("constraint_diagnostics")
            if diagnostics:
                nearest = diagnostics["diagnostic_nearest_pair_not_selected"]
                lines.append(f"- Fold {fold}: nearest diagnostic pair failed {', '.join(nearest['failed'])}.")
        lines.append(
            "The existing architecture is not collapsed, so a single focused loss-balance recalibration is justified "
            "before considering abandonment."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def divide(numerator: np.ndarray | int, denominator: np.ndarray | int) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.full(np.broadcast(numerator, denominator).shape, np.nan),
                     where=np.asarray(denominator) != 0)


def dice(tp: int, predicted: int, target: int) -> float | str:
    return ratio(2 * tp, predicted + target)


def ratio(numerator: int | float, denominator: int | float) -> float | str:
    return float(numerator / denominator) if denominator else "not_applicable"


def nanmean_stack(values: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.stack(list(values))
    count = np.isfinite(stacked).sum(axis=0)
    return np.divide(np.nansum(stacked, axis=0), count,
                     out=np.full(stacked.shape[1:], np.nan), where=count != 0)


def numeric_mean(values: Iterable[Any]) -> float | str:
    finite = [float(value) for value in values if is_number(value)]
    return float(np.mean(finite)) if finite else "not_applicable"


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def scalar(value: Any) -> float | str:
    return float(value) if is_number(value) else "not_applicable"


def format_number(value: Any) -> str:
    return f"{float(value):.4f}" if is_number(value) else "n/a"


def finite_or(value: Any, default: float = -math.inf) -> float:
    return float(value) if is_number(value) else default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
