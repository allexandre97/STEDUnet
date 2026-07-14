#!/usr/bin/env python
"""Validation-only calibration audit of saved four-class predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, maximum_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fibras.annotations import build_real_annotation_sample
from scripts.analyze_b1_uncertain_ignore import load_probabilities, load_records
from scripts.analyze_four_class_context_evaluation import metric_row
import scripts.analyze_gated_hierarchical_inference as hi


CLASS_IDS = np.array([0, 1, 3, 255], dtype=np.uint8)
BIAS_GRID = np.round(np.arange(-6.0, 2.0001, 0.05), 2)
SKELETON_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
CALIBRATED = ("F_calibrated_raw_skeleton", "F_calibrated_operational_skeleton")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", type=Path, default=Path("/ssd/STED_experiments/controlled_real_sted_v1"))
    p.add_argument("--experiment-root", type=Path, default=Path("/ssd/STED_experiments/controlled_real_sted_v1/four_class_context_unet_v1"))
    p.add_argument("--manifest", type=Path, default=ROOT / "data_manifests/real_annotations.csv")
    p.add_argument("--fold-manifest", type=Path, default=ROOT / "data_manifests/real_annotation_folds_v1.csv")
    p.add_argument("--image-root", type=Path, default=Path("/ssd/STED_dataset/data"))
    p.add_argument("--annotation-root", type=Path, default=Path("/cephfs/mhuang/STED_dataset/manual_annotations"))
    return p


def prediction_path(root: Path, record: Any) -> Path:
    return root / "evaluations" / f"fold_{record.outer_fold}" / record.partition / record.sample_id / "predictions.npz"


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        pred = {key: data[key].copy() for key in data.files}
    if "semantic_logits" in pred:
        pred["calibration_logits"] = pred["semantic_logits"].astype(np.float64)
        pred["logit_source"] = np.array("saved_raw_logits")
    else:
        pred["calibration_logits"] = np.log(np.clip(pred["four_class_probabilities"], 1e-12, 1.0)).astype(np.float64)
        pred["logit_source"] = np.array("log_clamped_probability")
    return pred


def probabilities(logits: np.ndarray, delta: float) -> np.ndarray:
    adjusted = logits.copy()
    adjusted[3] += delta
    adjusted -= adjusted.max(axis=0, keepdims=True)
    exp = np.exp(adjusted)
    return exp / exp.sum(axis=0, keepdims=True)


def target_indices(target: np.ndarray) -> np.ndarray:
    out = np.empty(target.shape, dtype=np.int8)
    for index, class_id in enumerate(CLASS_IDS):
        out[target == class_id] = index
    return out


def nll(probs: np.ndarray, target: np.ndarray) -> float:
    selected = np.take_along_axis(probs, target_indices(target)[None], axis=0)[0]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def semantic_calibration_metrics(probs: np.ndarray, target: np.ndarray, bins: int = 15) -> dict[str, float]:
    indices = target_indices(target)
    one_hot = np.eye(4, dtype=np.float64)[indices].transpose(2, 0, 1)
    confidence, prediction = probs.max(axis=0), probs.argmax(axis=0)
    correct = prediction == indices
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > low) & (confidence <= high) if low else confidence <= high
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {"four_class_nll": nll(probs, target), "brier_score": float(np.square(probs - one_hot).sum(axis=0).mean()), "ece_15_bin": ece}


def select_semantic_bias(validation: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select from validation logits and labels; outer-test data is not accepted."""
    rows = []
    for delta in BIAS_GRID:
        values = [nll(probabilities(logits, float(delta)), target) for logits, target in validation]
        rows.append({"delta_uncertain": float(delta), "validation_nll": float(np.mean(values))})
    best = min(rows, key=lambda row: (row["validation_nll"], abs(row["delta_uncertain"])))
    return dict(best), rows


def semantic_map(probs: np.ndarray) -> np.ndarray:
    return CLASS_IDS[probs.argmax(axis=0)]


def skeleton_metrics(target: np.ndarray, target_line: np.ndarray, predicted: np.ndarray) -> dict[str, float | str]:
    return hi.skeleton_metrics(target, (target_line > 0) & (target == 1), predicted)


def clump_metrics(target: np.ndarray, semantic: np.ndarray) -> dict[str, float | str]:
    valid = target != 255
    predicted, expected = (semantic == 3) & valid, target == 3
    true_positive = int((predicted & expected).sum())
    return {
        "clump_precision": hi.ratio(true_positive, int(predicted.sum())),
        "clump_recall": hi.ratio(true_positive, int(expected.sum())),
        "clump_area_ratio": hi.ratio(int(predicted.sum()), int(expected.sum())),
    }


def select_skeleton_thresholds(
    validation: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select raw and operational thresholds using validation data only."""
    selected, rows = {}, []
    for output in ("raw", "operational"):
        candidates = []
        for threshold in SKELETON_GRID:
            metrics = []
            for target, target_line, centreline, semantic in validation:
                predicted = centreline >= threshold
                if output == "operational":
                    predicted &= semantic == 1
                metrics.append(skeleton_metrics(target, target_line, predicted))
            dice = hi.numeric_mean(row["skeleton_dice"] for row in metrics)
            precision = hi.numeric_mean(row["skeleton_precision"] for row in metrics)
            candidates.append({"output": output, "threshold": float(threshold), "validation_skeleton_dice": dice, "validation_skeleton_precision": precision})
        rows.extend(candidates)
        selected[output] = max(candidates, key=lambda row: (number(row["validation_skeleton_dice"]), number(row["validation_skeleton_precision"]), -row["threshold"]))
    return selected, rows


def number(value: Any) -> float:
    return float(value) if hi.is_number(value) else -np.inf


def validation_selections(records: list[Any], samples: dict[str, dict[str, Any]], root: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selections, semantic_rows, skeleton_rows = {}, [], []
    for fold in range(5):
        fold_records = [record for record in records if record.outer_fold == fold and record.partition == "validation"]
        loaded = [(record, load_prediction(prediction_path(root, record))) for record in fold_records]
        semantic_input = [(pred["calibration_logits"], samples[record.sample_id]["real_semantic_mask"]) for record, pred in loaded]
        semantic, sweep = select_semantic_bias(semantic_input)
        delta = semantic["delta_uncertain"]
        skeleton_input = []
        for record, pred in loaded:
            sample = samples[record.sample_id]
            calibrated = semantic_map(probabilities(pred["calibration_logits"], delta))
            skeleton_input.append((sample["real_semantic_mask"], sample["real_skeleton_mask"], pred["centreline_probability"], calibrated))
        skeleton, skeleton_sweep = select_skeleton_thresholds(skeleton_input)
        source = sorted({str(pred["logit_source"]) for _, pred in loaded})
        selections[fold] = {"delta_uncertain": delta, "validation_nll": semantic["validation_nll"], "logit_source": source, "skeleton": skeleton, "selection_partition": "validation_only", "validation_sample_ids": [record.sample_id for record, _ in loaded]}
        semantic_rows.extend({"outer_fold": fold, "partition": "validation", **row} for row in sweep)
        skeleton_rows.extend({"outer_fold": fold, "partition": "validation", **row} for row in skeleton_sweep)
    return selections, semantic_rows, skeleton_rows


def evaluate_test(records: list[Any], samples: dict[str, dict[str, Any]], root: Path, selections: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Load outer-test labels only after all fold parameters have been frozen."""
    rows = []
    for record in (record for record in records if record.partition == "test"):
        pred = load_prediction(prediction_path(root, record))
        sample, selection = samples[record.sample_id], selections[record.outer_fold]
        target, target_line = sample["real_semantic_mask"], sample["real_skeleton_mask"]
        raw_probs = pred["four_class_probabilities"].astype(np.float64)
        calibrated_probs = probabilities(pred["calibration_logits"], selection["delta_uncertain"])
        raw = metric_row(record, "F_raw_four_class", target, target_line, pred["semantic_class_map"], raw_probs[3], pred["centreline_probability"] >= 0.75)
        raw.update(clump_metrics(target, pred["semantic_class_map"]))
        raw.update(semantic_calibration_metrics(raw_probs, target))
        rows.append(raw)
        semantic = semantic_map(calibrated_probs)
        for output, variant in zip(("raw", "operational"), CALIBRATED):
            skeleton = pred["centreline_probability"] >= selection["skeleton"][output]["threshold"]
            if output == "operational":
                skeleton &= semantic == 1
            row = metric_row(record, variant, target, target_line, semantic, calibrated_probs[3], skeleton)
            row.update(clump_metrics(target, semantic))
            row.update(semantic_calibration_metrics(calibrated_probs, target))
            row["delta_uncertain"] = selection["delta_uncertain"]
            row["centreline_threshold"] = selection["skeleton"][output]["threshold"]
            rows.append(row)
    return rows


def reference_rows(experiment_root: Path) -> list[dict[str, Any]]:
    path = experiment_root / "outer_test_per_image.csv"
    wanted = {"U0", "S", "G2_four_way_calibrated_skeleton"}
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: parse(value) for key, value in row.items()} for row in csv.DictReader(handle) if row["variant"] in wanted]


def add_paired_differences(rows: list[dict[str, Any]], references: list[dict[str, Any]]) -> None:
    lookup = {(row["variant"], row["sample_id"]): row for row in rows + references}
    metrics = ("fibrous_dice", "clump_dice", "uncertain_dice", "skeleton_dice", "skeleton_precision", "skeleton_recall", "target_skeleton_recovered_2px", "predicted_skeleton_within_target_2px")
    names = {"F_raw_four_class": "raw_four_class", "U0": "u0", "S": "s", "G2_four_way_calibrated_skeleton": "g2"}
    for row in rows:
        if row["variant"] not in CALIBRATED:
            continue
        for variant, short in names.items():
            reference = lookup.get((variant, row["sample_id"]))
            if not reference:
                continue
            for metric in metrics:
                if hi.is_number(row.get(metric)) and hi.is_number(reference.get(metric)):
                    row[f"delta_{metric}_vs_{short}"] = float(row[metric]) - float(reference[metric])


def macro_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(set().union(*(row.keys() for row in rows)) - {"variant", "sample_id", "outer_fold", "status"})
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        output.append({"variant": variant, "scope": "outer_test_image_macro", "image_count": len(selected), **{key: hi.numeric_mean(row.get(key) for row in selected) for key in keys}})
    return output


def viability(macro: list[dict[str, Any]]) -> dict[str, Any]:
    values = {row["variant"]: row for row in macro}
    candidate, u0, s, g2 = values[CALIBRATED[1]], values["U0"], values["S"], values["G2_four_way_calibrated_skeleton"]
    criteria = {
        "fibre_dice_loss_vs_u0_le_0_01": candidate["fibrous_dice"] - u0["fibrous_dice"] >= -0.01,
        "clump_dice_loss_vs_u0_le_0_01": candidate["clump_dice"] - u0["clump_dice"] >= -0.01,
        "uncertain_recall_gt_g2": candidate["uncertain_recall"] > g2["uncertain_recall"],
        "fibre_inside_uncertainty_lt_s": candidate["expert_uncertain_assigned_fibre"] < s["expert_uncertain_assigned_fibre"],
        "near_uncertainty_recall_loss_vs_u0_le_0_03": candidate["confident_fibrous_recall_0_2px"] - u0["confident_fibrous_recall_0_2px"] >= -0.03,
        "skeleton_dice_loss_vs_u0_le_0_01": candidate["skeleton_dice"] - u0["skeleton_dice"] >= -0.01,
    }
    passed = sum(criteria.values())
    conclusion = "calibration_alone_restores_viability" if passed == len(criteria) else ("calibration_partially_recovers_performance" if passed else "calibration_fails")
    return {"conclusion": conclusion, "criteria": criteria, "passed": passed, "total": len(criteria)}


def plots(out: Path, semantic: list[dict[str, Any]], skeleton: list[dict[str, Any]], macro: list[dict[str, Any]]) -> None:
    for filename, rows, x, ys in (
        ("semantic_bias_tradeoff.png", semantic, "delta_uncertain", ("validation_nll",)),
        ("skeleton_threshold_tradeoff.png", skeleton, "threshold", ("validation_skeleton_dice", "validation_skeleton_precision")),
    ):
        series = []
        for fold, output in sorted({(row["outer_fold"], row.get("output", "semantic")) for row in rows}):
            selected = [row for row in rows if row["outer_fold"] == fold and row.get("output", "semantic") == output]
            for y in ys:
                series.append((f"fold {fold} {output} {y}", [(float(row[x]), number(row[y])) for row in selected]))
        line_plot(out / filename, x, series)
    metrics = ("fibrous_dice", "clump_dice", "uncertain_dice", "skeleton_dice")
    variants = ["F_raw_four_class", CALIBRATED[1], "U0", "S", "G2_four_way_calibrated_skeleton"]
    lookup = {row["variant"]: row for row in macro}
    bar_plot(out / "outer_test_tradeoffs.png", metrics, ["raw", "calibrated", "U0", "S", "G2"], [[number(lookup[v].get(metric)) for v in variants] for metric in metrics])


def line_plot(path: Path, x_label: str, series: list[tuple[str, list[tuple[float, float]]]]) -> None:
    image = Image.new("RGB", (1100, 700), "white")
    draw, box = ImageDraw.Draw(image), (80, 40, 850, 620)
    values = [(x, y) for _, points in series for x, y in points if np.isfinite(y)]
    xmin, xmax = min(x for x, _ in values), max(x for x, _ in values)
    ymin, ymax = min(y for _, y in values), max(y for _, y in values)
    draw.rectangle(box, outline="black"); draw.text((430, 660), x_label, fill="black")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999", "#F0E442", "#332288")
    for index, (label, points) in enumerate(series):
        xy = [(box[0] + (x - xmin) / max(xmax - xmin, 1e-12) * (box[2] - box[0]), box[3] - (y - ymin) / max(ymax - ymin, 1e-12) * (box[3] - box[1])) for x, y in points if np.isfinite(y)]
        if len(xy) > 1:
            draw.line(xy, fill=colors[index % len(colors)], width=2)
        draw.text((870, 45 + index * 20), label, fill=colors[index % len(colors)])
    image.save(path)


def bar_plot(path: Path, metrics: tuple[str, ...], labels: list[str], values: list[list[float]]) -> None:
    image, colors = Image.new("RGB", (1100, 760), "white"), ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    draw = ImageDraw.Draw(image)
    for panel, (metric, row) in enumerate(zip(metrics, values)):
        left, top = 50 + (panel % 2) * 540, 40 + (panel // 2) * 360
        draw.text((left, top), metric, fill="black"); draw.line((left, top + 280, left + 460, top + 280), fill="black")
        for index, (label, value) in enumerate(zip(labels, row)):
            height = max(0, value) * 250 if np.isfinite(value) else 0
            x = left + 20 + index * 90
            draw.rectangle((x, top + 280 - height, x + 55, top + 280), fill=colors[index])
            draw.text((x, top + 290), label, fill="black")
    image.save(path)


def report(path: Path, selections: dict[int, dict[str, Any]], macro: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    values = {row["variant"]: row for row in macro}
    lines = ["# Four-class calibration audit", "", "## Leakage control", "", "All semantic biases and centreline thresholds were selected independently per fold from that fold's validation images. The complete parameter file was written before outer-test evaluation loaded test labels.", "", "## Selected parameters", "", "| Fold | Uncertain bias | Validation NLL | Raw threshold | Operational threshold |", "|---:|---:|---:|---:|---:|"]
    for fold, value in selections.items():
        lines.append(f"| {fold} | {value['delta_uncertain']:.2f} | {value['validation_nll']:.4f} | {value['skeleton']['raw']['threshold']:.2f} | {value['skeleton']['operational']['threshold']:.2f} |")
    lines += ["", "## Outer-test image-macro results", "", "| Variant | Fibre D/P/R/area | Clump D/P/R/area | Uncertain D/P/R/fraction | NLL/Brier/ECE | Skeleton D/P/R/recovery/leakage |", "|---|---|---|---|---|---|"]
    for variant in ("F_raw_four_class", *CALIBRATED, "U0", "S", "G2_four_way_calibrated_skeleton"):
        row = values[variant]
        f = lambda key: f"{float(row[key]):.3f}" if hi.is_number(row.get(key)) else "n/a"
        lines.append(f"| {variant} | {f('fibrous_dice')}/{f('fibrous_precision')}/{f('fibrous_recall')}/{f('fibrous_area_ratio')} | {f('clump_dice')}/{f('clump_precision')}/{f('clump_recall')}/{f('clump_area_ratio')} | {f('uncertain_dice')}/{f('uncertain_precision')}/{f('uncertain_recall')}/{f('predicted_uncertain_fraction')} | {f('four_class_nll')}/{f('brier_score')}/{f('ece_15_bin')} | {f('skeleton_dice')}/{f('skeleton_precision')}/{f('skeleton_recall')}/{f('target_skeleton_recovered_2px')}/{f('skeleton_leakage_into_uncertain')} |")
    c = values[CALIBRATED[1]]
    lines += ["", "## Assignment trade-offs", "", f"Calibrated uncertain assignment was {c['expert_background_assigned_uncertain']:.3f} on expert background and {c['expert_confident_fibre_assigned_uncertain']:.3f} on confident fibre. Inside expert uncertainty, calibrated fibre/clump assignments were {c['expert_uncertain_assigned_fibre']:.3f}/{c['expert_uncertain_assigned_clump']:.3f}.", "", "## Conclusion", "", f"**{decision['conclusion'].replace('_', ' ')}.** The calibrated operational output passed {decision['passed']}/{decision['total']} prespecified viability criteria.", "", "Raw-threshold and confident-fibre-intersected skeleton outputs are reported separately. Paired per-image differences against raw four-class, U0, S, and G2 are in `outer_test_per_image.csv`."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    out = args.experiment_root / "calibration_audit"
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.manifest, args.fold_manifest, args.image_root, args.annotation_root)
    selected = [record for record in records if record.partition in {"validation", "test"}]
    samples = {record.sample_id: build_real_annotation_sample(record.image_path, record.snakes_path, record.labels_path) for record in selected}
    selections, semantic_sweep, skeleton_sweep = validation_selections(records, samples, args.experiment_root)
    write_json(out / "selected_parameters.json", {str(key): value for key, value in selections.items()})
    test_rows = evaluate_test(records, samples, args.experiment_root, selections)
    references = reference_rows(args.experiment_root)
    add_paired_differences(test_rows, references)
    all_rows = test_rows + references
    macro = macro_rows(all_rows)
    decision = viability(macro)
    write_csv(out / "semantic_bias_sweeps.csv", semantic_sweep)
    write_csv(out / "skeleton_threshold_sweeps.csv", skeleton_sweep)
    write_csv(out / "outer_test_per_image.csv", all_rows)
    write_json(out / "outer_test_macro.json", {"metrics": macro, "viability": decision})
    plots(out, semantic_sweep, skeleton_sweep, macro)
    report(out / "report.md", selections, macro, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def parse(value: str) -> Any:
    try:
        return float(value) if "." in value or "e" in value.lower() else int(value)
    except ValueError:
        return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
