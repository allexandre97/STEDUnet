#!/usr/bin/env python
"""Analyze the controlled U0/S/G outer-test comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fibras.annotations import build_real_annotation_sample
from scripts.analyze_b1_uncertain_ignore import average_precision, auroc, load_probabilities, load_records
from scripts.evaluate_real_pilot_baseline import draw_tile, gray_rgb, heatmap, real_mask_rgb
from scripts.uncertain_ignore_nested_metrics import complete_image_metrics, numeric_mean

VARIANTS = ("U0", "S", "G")
PRIMARY = (
    "fibrous_dice", "fibrous_precision", "fibrous_recall", "fibrous_area_ratio", "clump_dice",
    "skeleton_dice_0.5", "skeleton_dice_0.75", "skeleton_dice_0.85",
    "target_skeleton_recovered_2px", "fibrous_leakage_into_clump",
)
PAIRED = PRIMARY + (
    "predicted_fibre_fraction_inside_expert_uncertain",
    "predicted_uncertain_fraction_inside_expert_uncertain",
    "expert_uncertain_precision", "expert_uncertain_recall", "expert_uncertain_dice",
    "confident_fibrous_recall_0_2px", "confident_fibrous_recall_2_5px",
    "confident_fibrous_recall_5_10px", "confident_fibrous_recall_10_20px",
    "confident_fibrous_recall_over_20px",
)
PROBABILITIES = (
    "foreground_probability", "quantifiability_probability",
    "conditional_fibrous_probability", "conditional_clump_probability",
    "final_fibrous_probability", "final_clump_probability", "final_uncertain_probability",
)
TARGETS = {"background": 0, "confident_fibre": 1, "confident_clump": 3, "expert_uncertain": 255}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    out = args.experiment_root
    records = [
        record for record in load_records(
            args.manifest, args.fold_manifest, args.image_root, args.annotation_root
        ) if record.partition == "test"
    ]
    if len(records) != 20 or len({record.sample_id for record in records}) != 20:
        raise ValueError("expected exactly 20 unique outer-test records")
    samples = {
        record.sample_id: build_real_annotation_sample(record.image_path, record.snakes_path, record.labels_path)
        for record in records
    }
    rows: list[dict[str, Any]] = []
    gated_predictions: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        sample = samples[record.sample_id]
        for variant in VARIANTS:
            pred = load_probabilities(prediction_path(args, record, variant))
            if variant == "G":
                require_gated_maps(prediction_path(args, record, variant), pred)
                gated_predictions[record.sample_id] = pred
            rows.append({
                "variant": variant, "outer_fold": record.outer_fold, "sample_id": record.sample_id,
                **image_metrics(sample["real_semantic_mask"], sample["real_skeleton_mask"], pred),
            })

    fold_rows = summarize_by(rows, ("variant", "outer_fold"))
    macro = summarize_by(rows, ("variant",))
    paired_rows, paired_summary = paired_comparisons(rows)
    decomposition = probability_decomposition(records, samples, gated_predictions)
    collapse = [
        collapse_diagnostics(record, gated_predictions[record.sample_id]) for record in records
    ]
    boundary = boundary_table(rows)
    provenance = checkpoint_provenance(args)
    decision = viability_decision(macro, paired_summary, collapse, decomposition)

    write_csv(out / "outer_test_per_image.csv", rows)
    write_json(out / "outer_test_per_image.json", rows)
    write_csv(out / "outer_test_per_fold.csv", fold_rows)
    write_json(out / "outer_test_macro.json", macro)
    write_csv(out / "paired_per_image.csv", paired_rows)
    write_csv(out / "paired_summary.csv", paired_summary)
    write_csv(out / "probability_decomposition.csv", decomposition)
    write_csv(out / "collapse_diagnostics.csv", collapse)
    write_json(out / "collapse_diagnostics.json", collapse)
    write_csv(out / "boundary_safety.csv", boundary)
    write_json(out / "checkpoint_provenance.json", provenance)
    write_json(out / "viability_decision.json", decision)
    write_qa_panels(out / "qualitative_qa", records, samples, gated_predictions)
    write_report(out / "controlled_evaluation_report.md", macro, paired_summary, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", type=Path, required=True)
    p.add_argument("--experiment-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fold-manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--annotation-root", type=Path, required=True)
    return p


def prediction_path(args: argparse.Namespace, record: Any, variant: str) -> Path:
    if variant == "U0":
        base = args.suite_root / "evaluations/b1_real_only"
        return base / f"fold_{record.outer_fold}" / record.sample_id / "predictions.npz"
    if variant == "S":
        base = args.suite_root / "uncertain_ignore_nested_cv/evaluations/S_s0.05"
        return base / f"fold_{record.outer_fold}/test" / record.sample_id / "predictions.npz"
    return args.experiment_root / "evaluations" / f"fold_{record.outer_fold}/test" / record.sample_id / "predictions.npz"


def require_gated_maps(path: Path, pred: dict[str, np.ndarray]) -> None:
    required = set(PROBABILITIES) | {
        "raw_centreline_probability", "final_background_probability", "final_skeleton_probability",
    }
    missing = sorted(required - set(pred))
    if missing:
        raise ValueError(f"{path}: missing gated maps {missing}")
    if set(np.unique(pred["semantic_class_map"])) - {0, 1, 3, 255}:
        raise ValueError(f"{path}: invalid gated semantic IDs")


def image_metrics(target: np.ndarray, skeleton: np.ndarray, pred: dict[str, np.ndarray]) -> dict[str, Any]:
    metrics = complete_image_metrics(target, skeleton, pred)
    uncertain = target == 255
    pred_uncertain = pred["semantic_class_map"] == 255
    tp = int((uncertain & pred_uncertain).sum())
    pred_count, target_count = int(pred_uncertain.sum()), int(uncertain.sum())
    metrics.update({
        "predicted_fibre_fraction_inside_expert_uncertain": ratio(
            int(((pred["semantic_class_map"] == 1) & uncertain).sum()), target_count
        ),
        "predicted_uncertain_fraction_inside_expert_uncertain": ratio(tp, target_count),
        "expert_uncertain_precision": ratio(tp, pred_count),
        "expert_uncertain_recall": ratio(tp, target_count),
        "expert_uncertain_dice": ratio(2 * tp, pred_count + target_count),
        "expert_uncertain_auroc": auroc(uncertain.ravel(), pred["uncertainty_probability"].ravel()),
        "expert_uncertain_average_precision": average_precision(
            uncertain.ravel(), pred["uncertainty_probability"].ravel()
        ),
        "expert_uncertain_assigned_background": assignment_fraction(pred, uncertain, 0),
        "expert_uncertain_assigned_fibre": assignment_fraction(pred, uncertain, 1),
        "expert_uncertain_assigned_clump": assignment_fraction(pred, uncertain, 3),
        "expert_uncertain_assigned_uncertain": assignment_fraction(pred, uncertain, 255),
        "expert_fibre_assigned_uncertain": assignment_fraction(pred, target == 1, 255),
        "expert_clump_assigned_uncertain": assignment_fraction(pred, target == 3, 255),
        "expert_background_assigned_uncertain": assignment_fraction(pred, target == 0, 255),
        "final_fibre_probability_inside_expert_uncertain": region_mean(
            pred["fibrous_probability"], uncertain
        ),
        "final_uncertain_probability_inside_expert_fibre": region_mean(
            pred["uncertainty_probability"], target == 1
        ),
    })
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def assignment_fraction(pred: dict[str, np.ndarray], region: np.ndarray, class_id: int) -> float | str:
    return ratio(int(((pred["semantic_class_map"] == class_id) & region).sum()), int(region.sum()))


def ratio(numerator: int | float, denominator: int | float) -> float | str:
    return float(numerator / denominator) if denominator else "not_applicable"


def region_mean(values: np.ndarray, mask: np.ndarray) -> float | str:
    return float(values[mask].mean()) if mask.any() else "not_applicable"


def summarize_by(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    metrics = sorted(set(rows[0]) - {"variant", "outer_fold", "sample_id", "clump_applicable"})
    return [
        {
            **dict(zip(keys, group)), "image_count": len(selected),
            **{metric: numeric_mean(row.get(metric) for row in selected) for metric in metrics},
            "clump_applicable_images": sum(bool(row["clump_applicable"]) for row in selected),
        }
        for group, selected in sorted(groups.items())
    ]


def paired_comparisons(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(row["variant"], row["sample_id"]): row for row in rows}
    samples = sorted({row["sample_id"] for row in rows})
    per_image, summary = [], []
    for reference in ("U0", "S"):
        for metric in PAIRED:
            values = []
            for sample in samples:
                g, baseline = lookup[("G", sample)][metric], lookup[(reference, sample)][metric]
                if is_number(g) and is_number(baseline):
                    difference = float(g) - float(baseline)
                    per_image.append({
                        "comparison": f"G-minus-{reference}", "sample_id": sample,
                        "outer_fold": lookup[("G", sample)]["outer_fold"], "metric": metric,
                        "gated": g, "reference": baseline, "difference": difference,
                    })
                    values.append(difference)
            summary.append({
                "comparison": f"G-minus-{reference}", "metric": metric,
                "mean_paired_difference": float(np.mean(values)) if values else "not_applicable",
                "sd_paired_difference": float(np.std(values, ddof=1)) if len(values) > 1 else "not_applicable",
                "applicable_images": len(values), "improved_images": sum(value > 0 for value in values),
                "worsened_images": sum(value < 0 for value in values),
            })
    return per_image, summary


def probability_decomposition(
    records: list[Any], samples: dict[str, dict[str, Any]], predictions: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    out = []
    scopes: list[tuple[str, int | str, list[Any]]] = [("all", "all", records)]
    scopes += [("fold", fold, [record for record in records if record.outer_fold == fold]) for fold in range(5)]
    for scope, fold, selected in scopes:
        for target_name, target_id in TARGETS.items():
            for probability in PROBABILITIES:
                values = np.concatenate([
                    predictions[record.sample_id][probability][
                        samples[record.sample_id]["real_semantic_mask"] == target_id
                    ].astype(np.float32, copy=False)
                    for record in selected
                ])
                quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
                out.append({
                    "scope": scope, "outer_fold": fold, "target_class": target_name,
                    "probability": probability, "pixels": values.size, "mean": float(values.mean()),
                    "std": float(values.std()), "p05": float(quantiles[0]), "p25": float(quantiles[1]),
                    "p50": float(quantiles[2]), "p75": float(quantiles[3]), "p95": float(quantiles[4]),
                })
    return out


def collapse_diagnostics(record: Any, pred: dict[str, np.ndarray]) -> dict[str, Any]:
    semantic, q, g = (
        pred["semantic_class_map"], pred["foreground_probability"], pred["quantifiability_probability"]
    )
    row: dict[str, Any] = {"sample_id": record.sample_id, "outer_fold": record.outer_fold}
    for class_id, name in ((0, "background"), (1, "fibre"), (3, "clump"), (255, "uncertain")):
        row[f"predicted_{name}_fraction"] = float(np.mean(semantic == class_id))
    for name, values in (("q", q), ("g", g)):
        quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        row.update({
            f"{name}_mean": float(values.mean()), f"{name}_std": float(values.std()),
            f"{name}_p05": float(quantiles[0]), f"{name}_p25": float(quantiles[1]),
            f"{name}_p50": float(quantiles[2]), f"{name}_p75": float(quantiles[3]),
            f"{name}_p95": float(quantiles[4]),
        })
    for threshold in (0.1, 0.25, 0.5, 0.75, 0.9):
        row[f"g_below_{threshold:g}_fraction"] = float(np.mean(g < threshold))
    row["foreground_probability_assigned_uncertain_fraction"] = float(
        pred["final_uncertain_probability"].sum() / max(float(q.sum()), 1e-7)
    )
    near_zero = []
    for name in (*PROBABILITIES, "raw_centreline_probability"):
        variance = float(pred[name].var())
        row[f"{name}_variance"] = variance
        if variance < 1e-8:
            near_zero.append(name)
    row["near_zero_variance_heads"] = ";".join(near_zero)
    row["collapsed"] = bool(
        max(row[f"predicted_{name}_fraction"] for name in ("background", "fibre", "clump", "uncertain")) > 0.99
        or row["q_mean"] < 0.01 or row["q_mean"] > 0.99 or row["g_mean"] < 0.01 or row["g_mean"] > 0.99
        or bool(near_zero)
    )
    return row


def boundary_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        for band in ("0_2px", "2_5px", "5_10px", "10_20px", "over_20px"):
            key = f"confident_fibrous_recall_{band}"
            values = [float(row[key]) for row in selected if is_number(row[key])]
            out.append({
                "variant": variant, "band": band, "macro_recall": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else "not_applicable",
                "applicable_images": len(values),
            })
    return out


def checkpoint_provenance(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    folds = {}
    for fold in range(5):
        run = args.experiment_root / "runs" / f"fold_{fold}"
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        summary = json.loads((run / "checkpoint_summary.json").read_text(encoding="utf-8"))
        epoch_metrics = []
        for path in sorted((run / "checkpoints").glob("model_epoch_*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            epoch_metrics.append({
                "path": str(path), "epoch": payload["epoch"],
                "validation_metrics": payload["validation_metrics"],
                "real_validation_metrics": payload["real_validation_metrics"],
            })
        folds[str(fold)] = {
            "configuration": metadata["command_line_args"],
            "initialization_report": metadata["initial_checkpoint"],
            "checkpoint_summary": summary, "per_epoch_validation_metrics": epoch_metrics,
        }
        write_json(run / "per_epoch_validation_metrics.json", epoch_metrics)
    return {"folds": folds}


def viability_decision(
    macro: list[dict[str, Any]], paired: list[dict[str, Any]], collapse: list[dict[str, Any]],
    decomposition: list[dict[str, Any]],
) -> dict[str, Any]:
    by_variant = {row["variant"]: row for row in macro}
    difference = {(row["comparison"], row["metric"]): row["mean_paired_difference"] for row in paired}
    g, u0, s = by_variant["G"], by_variant["U0"], by_variant["S"]
    uncertain_fibre = "predicted_fibre_fraction_inside_expert_uncertain"
    reductions = {
        "versus_u0": u0[uncertain_fibre] - g[uncertain_fibre],
        "versus_s": s[uncertain_fibre] - g[uncertain_fibre],
    }
    material_reduction = 0.02
    checks = {
        "materially_reduces_uncertain_region_fibre_vs_u0_and_s": all(
            value >= material_reduction for value in reductions.values()
        ),
        "expert_uncertain_recall_above_previous_h": g["expert_uncertain_recall"] > 0.115,
        "fibrous_dice_within_0.01_of_u0": difference[("G-minus-U0", "fibrous_dice")] >= -0.01,
        "skeleton_dice_075_within_0.01_of_u0": difference[("G-minus-U0", "skeleton_dice_0.75")] >= -0.01,
        "clump_dice_within_0.01_of_u0": difference[("G-minus-U0", "clump_dice")] >= -0.01,
        "boundary_0_2px_not_severe": difference[("G-minus-U0", "confident_fibrous_recall_0_2px")] >= -0.05,
        "no_collapse": not any(row["collapsed"] for row in collapse),
    }
    diagnosis = failure_diagnosis(checks, decomposition)
    viable = all(checks.values())
    return {
        "architecturally_viable": viable, "collapsed": not checks["no_collapse"],
        "criteria": checks, "failure_diagnosis": diagnosis,
        "uncertain_region_fibre_absolute_reduction": reductions,
        "material_reduction_threshold": material_reduction,
        "focused_weight_calibration_justified": bool(not viable and checks["no_collapse"]),
        "recommended_next_weights": {
            "lambda_foreground": 1.5, "lambda_gate": 1.0, "lambda_morphology": 1.0,
            "lambda_joint": 0.5, "lambda_skeleton": 1.0,
        },
    }


def failure_diagnosis(checks: dict[str, bool], rows: list[dict[str, Any]]) -> str:
    lookup = {
        (row["target_class"], row["probability"]): row["mean"]
        for row in rows if row["scope"] == "all"
    }
    if not checks["no_collapse"]:
        return "architectural output collapse"
    if lookup[("confident_fibre", "foreground_probability")] < 0.5:
        return "foreground detector q under-detects confident fibres"
    if lookup[("confident_fibre", "quantifiability_probability")] < 0.5:
        return "quantifiability gate g rejects too many confident fibres"
    if lookup[("confident_fibre", "conditional_fibrous_probability")] < 0.5:
        return "conditional morphology classifier r under-identifies fibres"
    uncertain_q = lookup[("expert_uncertain", "foreground_probability")]
    uncertain_g = lookup[("expert_uncertain", "quantifiability_probability")]
    fibre_g = lookup[("confident_fibre", "quantifiability_probability")]
    if uncertain_q < 0.5 and uncertain_g > 0.25:
        return (
            "q under-detects expert-uncertain material and g separation is too weak; "
            "confident-class r is learned, but coupling compresses final fibre and skeleton probabilities"
        )
    if fibre_g - uncertain_g < 0.35:
        return "quantifiability gate g insufficiently separates uncertain from confident foreground"
    if not checks["materially_reduces_uncertain_region_fibre_vs_u0_and_s"]:
        return "gate g or joint coupling does not materially divert expert-uncertain material from fibre"
    return "no dominant q/g/r failure identified; remaining deficits are coupled performance trade-offs"


def write_qa_panels(
    out: Path, records: list[Any], samples: dict[str, dict[str, Any]], predictions: dict[str, dict[str, np.ndarray]],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for record in records:
        sample, pred = samples[record.sample_id], predictions[record.sample_id]
        raw = sample["image_uint8"] if sample["image_uint8"] is not None else sample["image_float"]
        tiles = [
            draw_tile(gray_rgb(raw), "raw STED"), draw_tile(real_mask_rgb(sample["real_semantic_mask"]), "expert"),
            draw_tile(real_mask_rgb(pred["semantic_class_map"]), "G semantic"),
            draw_tile(heatmap(pred["foreground_probability"]), "q"),
            draw_tile(heatmap(pred["quantifiability_probability"]), "g"),
            draw_tile(heatmap(pred["conditional_fibrous_probability"]), "r fibre"),
            draw_tile(heatmap(pred["final_fibrous_probability"]), "final fibre"),
            draw_tile(heatmap(pred["final_uncertain_probability"]), "final uncertain"),
            draw_tile(heatmap(pred["final_skeleton_probability"]), "final skeleton"),
        ]
        panel = Image.new("RGB", (3 * tiles[0].width, 3 * tiles[0].height), "white")
        for index, tile in enumerate(tiles):
            panel.paste(tile, ((index % 3) * tile.width, (index // 3) * tile.height))
        panel.save(out / f"fold_{record.outer_fold}_{record.sample_id}.png")


def write_report(
    path: Path, macro: list[dict[str, Any]], paired: list[dict[str, Any]], decision: dict[str, Any],
) -> None:
    values = {row["variant"]: row for row in macro}
    diffs = {(row["comparison"], row["metric"]): row["mean_paired_difference"] for row in paired}
    lines = [
        "# Gated ContextUNet controlled evaluation", "", "## Decision", "",
        f"- Architecturally viable: {decision['architecturally_viable']}",
        f"- Collapsed: {decision['collapsed']}",
        f"- Diagnosis: {decision['failure_diagnosis']}", "", "## Primary macro metrics", "",
        "| Variant | Fibre Dice | Precision | Recall | Area ratio | Clump Dice | Skeleton .5/.75/.85 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = values[variant]
        lines.append(
            f"| {variant} | {row['fibrous_dice']:.4f} | {row['fibrous_precision']:.4f} | "
            f"{row['fibrous_recall']:.4f} | {row['fibrous_area_ratio']:.4f} | {row['clump_dice']:.4f} | "
            f"{row['skeleton_dice_0.5']:.4f}/{row['skeleton_dice_0.75']:.4f}/{row['skeleton_dice_0.85']:.4f} |"
        )
    lines += ["", "## Paired G differences", ""]
    for reference in ("U0", "S"):
        lines.append(
            f"- G−{reference}: fibre Dice {diffs[(f'G-minus-{reference}', 'fibrous_dice')]:+.4f}, "
            f"skeleton Dice .75 {diffs[(f'G-minus-{reference}', 'skeleton_dice_0.75')]:+.4f}, "
            f"clump Dice {diffs[(f'G-minus-{reference}', 'clump_dice')]:+.4f}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
