#!/usr/bin/env python
"""Analyze the controlled four-class ContextUNet evaluation from saved maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fibras.annotations import build_real_annotation_sample
from scripts.analyze_b1_uncertain_ignore import average_precision, auroc, load_probabilities, load_records
import scripts.analyze_gated_hierarchical_inference as hi
from scripts.evaluate_real_pilot_baseline import gray_rgb, heatmap, real_mask_rgb


VARIANT = "F_four_class_argmax"
PROBABILITIES = (
    "background_probability", "fibrous_probability", "clump_probability",
    "uncertainty_probability", "centreline_probability",
)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    records = load_records(args.manifest, args.fold_manifest, args.image_root, args.annotation_root)
    selected = [record for record in records if record.partition in {"validation", "test"}]
    samples = {record.sample_id: build_real_annotation_sample(record.image_path, record.snakes_path, record.labels_path) for record in selected}
    predictions = load_four_class_predictions(args.experiment_root, selected)

    outer_rows = outer_test_rows(args, records, samples, predictions)
    macro = summarized_rows(outer_rows)
    paired = paired_rows(outer_rows)
    discrimination = discrimination_rows(records, samples, predictions)
    probability = probability_rows(selected, samples, predictions)
    collapse = collapse_rows(records, predictions)
    boundary = boundary_rows(outer_rows)
    bootstrap = bootstrap_rows(outer_rows, args.bootstrap_repeats, args.seed)
    provenance = checkpoint_provenance(args.experiment_root)
    decision = viability_decision(macro)

    out = args.experiment_root
    write_csv(out / "outer_test_per_image.csv", outer_rows)
    write_csv(out / "outer_test_per_fold.csv", [row for row in macro if row["scope"] == "fold_image_macro"])
    write_json(out / "outer_test_macro.json", [row for row in macro if row["scope"] == "global_image_macro"])
    write_csv(out / "paired_comparisons.csv", paired)
    write_csv(out / "uncertainty_discrimination.csv", discrimination)
    write_csv(out / "probability_decomposition.csv", probability)
    write_csv(out / "collapse_diagnostics.csv", collapse)
    write_csv(out / "boundary_safety.csv", boundary)
    write_csv(out / "bootstrap_confidence_intervals.csv", bootstrap)
    write_json(out / "checkpoint_provenance.json", provenance)
    write_json(out / "viability_decision.json", decision)
    write_panels(out / "qualitative_qa_panels", records, samples, predictions)
    write_report(out / "controlled_four_class_report.md", macro, paired, discrimination, collapse, bootstrap, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", type=Path, default=Path("/ssd/STED_experiments/controlled_real_sted_v1"))
    p.add_argument("--experiment-root", type=Path, default=Path("/ssd/STED_experiments/controlled_real_sted_v1/four_class_context_unet_v1"))
    p.add_argument("--manifest", type=Path, default=ROOT / "data_manifests/real_annotations.csv")
    p.add_argument("--fold-manifest", type=Path, default=ROOT / "data_manifests/real_annotation_folds_v1.csv")
    p.add_argument("--image-root", type=Path, default=Path("/ssd/STED_dataset/data"))
    p.add_argument("--annotation-root", type=Path, default=Path("/cephfs/mhuang/STED_dataset/manual_annotations"))
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    return p


def load_four_class_predictions(root: Path, records: list[Any]) -> dict[tuple[int, str, str], dict[str, np.ndarray]]:
    out = {}
    for record in records:
        path = root / "evaluations" / f"fold_{record.outer_fold}" / record.partition / record.sample_id / "predictions.npz"
        pred = load_probabilities(path)
        missing = {"semantic_class_map", "four_class_probabilities", "centreline_probability"} - pred.keys()
        if missing:
            raise ValueError(f"{path}: missing four-class arrays {sorted(missing)}")
        probs = pred["four_class_probabilities"]
        if probs.shape[0] != 4:
            raise ValueError(f"{path}: expected four probability channels, got {probs.shape}")
        pred.setdefault("background_probability", probs[0])
        pred.setdefault("fibrous_probability", probs[1])
        pred.setdefault("clump_probability", probs[2])
        pred.setdefault("uncertainty_probability", probs[3])
        pred.setdefault("skeleton_probability", pred["centreline_probability"])
        out[(record.outer_fold, record.partition, record.sample_id)] = pred
    return out


def outer_test_rows(
    args: argparse.Namespace, records: list[Any], samples: dict[str, dict[str, Any]],
    predictions: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for record in (item for item in records if item.partition == "test"):
        target = samples[record.sample_id]["real_semantic_mask"]
        target_line = samples[record.sample_id]["real_skeleton_mask"]
        for variant in ("U0", "S"):
            pred = load_probabilities(hi.reference_path(args, record, variant))
            rows.append(metric_row(record, variant, target, target_line, pred["semantic_class_map"], pred["uncertainty_probability"], pred["skeleton_probability"] >= 0.75))
        rows.extend(g2_rows(args, record, target, target_line))
        pred = predictions[(record.outer_fold, "test", record.sample_id)]
        rows.append(metric_row(record, VARIANT, target, target_line, pred["semantic_class_map"], pred["uncertainty_probability"], pred["centreline_probability"] >= 0.75))
    return rows


def g2_rows(args: argparse.Namespace, record: Any, target: np.ndarray, target_line: np.ndarray) -> list[dict[str, Any]]:
    path = args.suite_root / "gated_context_unet_g2" / "outer_test_per_image.csv"
    if not path.exists():
        return []
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row["sample_id"] == record.sample_id and row["variant"] in {"G2_four_way_argmax", "G2_four_way_calibrated_skeleton", "G2_hierarchical_calibrated_skeleton"}:
            parsed = {key: parse_value(value) for key, value in row.items()}
            parsed["variant"] = row["variant"]
            parsed["sample_id"] = row["sample_id"]
            parsed["outer_fold"] = int(row["outer_fold"])
            parsed["status"] = row.get("status", "available")
            rows.append(parsed)
    return rows


def metric_row(
    record: Any, variant: str, target: np.ndarray, target_line: np.ndarray, semantic: np.ndarray,
    uncertainty_score: np.ndarray, skeleton: np.ndarray,
) -> dict[str, Any]:
    row = {"variant": variant, "outer_fold": record.outer_fold, "sample_id": record.sample_id, "status": "available"}
    row.update(hi.complete_metrics(target, target_line, semantic, uncertainty_score, skeleton))
    row["predicted_fibre_fraction_inside_expert_uncertain"] = hi.assignment(semantic, target == 255, 1)
    row["predicted_clump_fraction_inside_expert_uncertain"] = hi.assignment(semantic, target == 255, 3)
    row["predicted_uncertain_fraction_inside_expert_uncertain"] = hi.assignment(semantic, target == 255, 255)
    row.update(class_fractions(semantic))
    return row


def class_fractions(semantic: np.ndarray) -> dict[str, float]:
    return {
        "predicted_background_fraction": float(np.mean(semantic == 0)),
        "predicted_fibre_fraction": float(np.mean(semantic == 1)),
        "predicted_clump_fraction": float(np.mean(semantic == 3)),
        "predicted_uncertain_fraction": float(np.mean(semantic == 255)),
    }


def discrimination_rows(
    records: list[Any], samples: dict[str, dict[str, Any]],
    predictions: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for partition in ("validation", "test"):
        parts = []
        for record in (item for item in records if item.partition == partition):
            target = samples[record.sample_id]["real_semantic_mask"]
            pred = predictions[(record.outer_fold, partition, record.sample_id)]
            mask = np.isin(target, (1, 3, 255))
            rows.append(discrimination_record(partition, "uncertain_vs_confident_tau", "image", record.outer_fold, record.sample_id, target[mask] == 255, pred["uncertainty_probability"][mask]))
            parts.append((record, target[mask] == 255, pred["uncertainty_probability"][mask]))
        rows.append(pooled_discrimination(partition, "uncertain_vs_confident_tau", "global_pooled", "all", parts))
        rows.append(macro_discrimination(partition, "uncertain_vs_confident_tau", "global_image_macro", "all", rows))
        for fold in range(5):
            fold_parts = [part for part in parts if part[0].outer_fold == fold]
            rows.append(pooled_discrimination(partition, "uncertain_vs_confident_tau", "fold_pooled", fold, fold_parts))
    return rows


def discrimination_record(partition: str, comparison: str, scope: str, fold: Any, sample_id: str, y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    return {
        "partition": partition, "comparison": comparison, "scope": scope, "outer_fold": fold,
        "sample_id": sample_id, "positive_pixels": int(y.sum()), "negative_pixels": int((~y).sum()),
        "prevalence": hi.ratio(int(y.sum()), int(y.size)), "auroc": auroc(y, score),
        "average_precision": average_precision(y, score),
    }


def pooled_discrimination(partition: str, comparison: str, scope: str, fold: Any, parts: list[tuple[Any, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    if not parts:
        return discrimination_record(partition, comparison, scope, fold, "all", np.array([], bool), np.array([], np.float32))
    return discrimination_record(partition, comparison, scope, fold, "all", np.concatenate([part[1] for part in parts]), np.concatenate([part[2] for part in parts]))


def macro_discrimination(partition: str, comparison: str, scope: str, fold: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["partition"] == partition and row["comparison"] == comparison and row["scope"] == "image"]
    return {
        "partition": partition, "comparison": comparison, "scope": scope, "outer_fold": fold, "sample_id": "all",
        "positive_pixels": sum(int(row["positive_pixels"]) for row in selected),
        "negative_pixels": sum(int(row["negative_pixels"]) for row in selected),
        "prevalence": hi.ratio(sum(int(row["positive_pixels"]) for row in selected), sum(int(row["positive_pixels"]) + int(row["negative_pixels"]) for row in selected)),
        "auroc": hi.numeric_mean(row["auroc"] for row in selected),
        "average_precision": hi.numeric_mean(row["average_precision"] for row in selected),
    }


def probability_rows(
    records: list[Any], samples: dict[str, dict[str, Any]],
    predictions: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    targets = {"background": 0, "fibre": 1, "clump": 3, "uncertain": 255}
    for partition in ("validation", "test"):
        for target_name, target_id in targets.items():
            for probability in PROBABILITIES:
                values = np.concatenate([
                    predictions[(record.outer_fold, partition, record.sample_id)][probability][samples[record.sample_id]["real_semantic_mask"] == target_id]
                    for record in records if record.partition == partition
                ])
                q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
                rows.append({
                    "partition": partition, "target_class": target_name, "probability": probability,
                    "pixels": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
                    "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4]),
                })
    return rows


def collapse_rows(records: list[Any], predictions: dict[tuple[int, str, str], dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for record in (item for item in records if item.partition == "test"):
        pred = predictions[(record.outer_fold, "test", record.sample_id)]
        row = {"variant": VARIANT, "outer_fold": record.outer_fold, "sample_id": record.sample_id, **class_fractions(pred["semantic_class_map"])}
        for name in PROBABILITIES:
            values = pred[name]
            quantiles = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
            row.update({f"{name}_mean": float(values.mean()), f"{name}_std": float(values.std()), f"{name}_p01": float(quantiles[0]), f"{name}_p25": float(quantiles[1]), f"{name}_p50": float(quantiles[2]), f"{name}_p75": float(quantiles[3]), f"{name}_p99": float(quantiles[4])})
        row["near_zero_variance_heads"] = ",".join(name for name in PROBABILITIES if row[f"{name}_std"] < 1e-5)
        row["collapsed"] = max(row["predicted_background_fraction"], row["predicted_fibre_fraction"], row["predicted_clump_fraction"], row["predicted_uncertain_fraction"]) > 0.99 or bool(row["near_zero_variance_heads"])
        rows.append(row)
    return rows


def summarized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize_groups(rows, ("variant",), "global_image_macro") + summarize_groups(rows, ("variant", "outer_fold"), "fold_image_macro")


def summarize_groups(rows: list[dict[str, Any]], keys: tuple[str, ...], scope: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    metrics = sorted(set().union(*(row.keys() for row in rows)) - {"variant", "outer_fold", "sample_id", "status"})
    return [{"scope": scope, **dict(zip(keys, group)), "image_count": len(selected), "available_images": sum(row.get("status") == "available" for row in selected), **{metric: hi.numeric_mean(row.get(metric) for row in selected) for metric in metrics}} for group, selected in sorted(groups.items())]


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["variant"], row["sample_id"]): row for row in rows}
    samples = sorted({row["sample_id"] for row in rows if (VARIANT, row["sample_id"]) in lookup})
    metrics = (*hi.SEMANTIC_METRICS, *hi.SKELETON_METRICS, "predicted_fibre_fraction_inside_expert_uncertain", "predicted_clump_fraction_inside_expert_uncertain")
    output = []
    for reference in ("U0", "S", "G2_four_way_argmax", "G2_four_way_calibrated_skeleton", "G2_hierarchical_calibrated_skeleton"):
        if not any((reference, sample) in lookup for sample in samples):
            continue
        for metric in metrics:
            differences = []
            for sample in samples:
                if (reference, sample) not in lookup:
                    continue
                value = lookup[(VARIANT, sample)].get(metric)
                baseline = lookup[(reference, sample)].get(metric)
                if hi.is_number(value) and hi.is_number(baseline):
                    difference = float(value) - float(baseline)
                    differences.append(difference)
                    output.append({"scope": "image", "comparison": f"{VARIANT}-minus-{reference}", "metric": metric, "sample_id": sample, "outer_fold": lookup[(VARIANT, sample)]["outer_fold"], "value": value, "reference": baseline, "difference": difference})
            output.append({"scope": "global_image_macro", "comparison": f"{VARIANT}-minus-{reference}", "metric": metric, "mean_paired_difference": float(np.mean(differences)) if differences else "not_applicable", "applicable_images": len(differences)})
    return output


def boundary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    variants = sorted({row["variant"] for row in rows})
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        for name, _, _ in hi.BOUNDARY_BANDS:
            metric = f"confident_fibrous_recall_{name}px"
            output.append({"variant": variant, "band": name, "macro_recall": hi.numeric_mean(row.get(metric) for row in selected)})
    return output


def bootstrap_rows(rows: list[dict[str, Any]], repeats: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant and row.get("status") == "available"]
        if not selected:
            continue
        for metric in ("fibrous_dice", "clump_dice", "uncertain_recall", "fibrous_leakage_into_target_clump", "skeleton_dice"):
            values = np.array([float(row[metric]) for row in selected if hi.is_number(row.get(metric))], dtype=np.float64)
            if not values.size:
                continue
            draws = np.array([rng.choice(values, size=values.size, replace=True).mean() for _ in range(repeats)])
            output.append({"variant": variant, "metric": metric, "mean": float(values.mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)), "images": int(values.size)})
    return output


def checkpoint_provenance(root: Path) -> dict[str, Any]:
    folds = {}
    for fold in range(5):
        run = root / f"runs/fold_{fold}"
        metadata = json.loads((run / "run_metadata.json").read_text()) if (run / "run_metadata.json").exists() else {}
        folds[str(fold)] = {
            "run_dir": str(run), "model_best": str(run / "model_best.pt"),
            "config": str(run / "config.json"), "metadata": metadata,
        }
    return {"experiment_root": str(root), "folds": folds}


def viability_decision(macro: list[dict[str, Any]]) -> dict[str, Any]:
    values = {row["variant"]: row for row in macro if row["scope"] == "global_image_macro"}
    f, u0, s = values.get(VARIANT, {}), values.get("U0", {}), values.get("S", {})
    g2 = values.get("G2_four_way_calibrated_skeleton") or values.get("G2_four_way_argmax", {})
    criteria = {
        "fibre_dice_loss_vs_u0_le_0_01": delta_ok(f, u0, "fibrous_dice", -0.01),
        "clump_dice_loss_vs_u0_le_0_01": delta_ok(f, u0, "clump_dice", -0.01),
        "uncertain_recall_gt_g2_8_31_percent": hi.is_number(f.get("uncertain_recall")) and float(f["uncertain_recall"]) > 0.0831,
        "fibre_leakage_inside_uncertainty_lt_s_25_30_percent": hi.is_number(f.get("expert_uncertain_assigned_fibre")) and float(f["expert_uncertain_assigned_fibre"]) < 0.253,
        "near_uncertainty_recall_loss_vs_u0_le_0_03": delta_ok(f, u0, "confident_fibrous_recall_0_2px", -0.03),
        "skeleton_dice_loss_vs_u0_le_0_01": delta_ok(f, u0, "skeleton_dice", -0.01),
        "not_worse_than_g2_uncertain_recall": not (hi.is_number(g2.get("uncertain_recall")) and hi.is_number(f.get("uncertain_recall"))) or float(f["uncertain_recall"]) >= float(g2["uncertain_recall"]),
    }
    return {
        "criteria": criteria, "decision": "viable" if all(criteria.values()) else "non_viable",
        "four_class": f, "u0": u0, "s": s, "g2_reference": g2,
    }


def delta_ok(candidate: dict[str, Any], baseline: dict[str, Any], metric: str, floor: float) -> bool:
    if not (hi.is_number(candidate.get(metric)) and hi.is_number(baseline.get(metric))):
        return False
    return float(candidate[metric]) - float(baseline[metric]) >= floor - 1e-12


def write_panels(out: Path, records: list[Any], samples: dict[str, dict[str, Any]], predictions: dict[tuple[int, str, str], dict[str, np.ndarray]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for record in (item for item in records if item.partition == "test"):
        sample = samples[record.sample_id]
        pred = predictions[(record.outer_fold, "test", record.sample_id)]
        tiles = [
            hi.labeled(gray_rgb(sample["image_uint8"]), "raw STED"),
            hi.labeled(real_mask_rgb(sample["real_semantic_mask"]), "expert"),
            hi.labeled(real_mask_rgb(pred["semantic_class_map"]), "four-class argmax"),
            hi.labeled(heatmap(pred["background_probability"]), "P background"),
            hi.labeled(heatmap(pred["fibrous_probability"]), "P fibre"),
            hi.labeled(heatmap(pred["clump_probability"]), "P clump"),
            hi.labeled(heatmap(pred["uncertainty_probability"]), "P uncertain"),
            hi.labeled(heatmap(pred["centreline_probability"]), "P centreline"),
            hi.labeled(hi.binary_rgb(pred["centreline_probability"] >= 0.75), "centreline >=0.75"),
        ]
        panel = Image.new("RGB", (3 * tiles[0].width, 3 * tiles[0].height), "white")
        for index, tile in enumerate(tiles):
            panel.paste(tile, ((index % 3) * tile.width, (index // 3) * tile.height))
        panel.save(out / f"fold_{record.outer_fold}_{record.sample_id}.png")


def write_report(
    path: Path, macro: list[dict[str, Any]], paired: list[dict[str, Any]],
    discrimination: list[dict[str, Any]], collapse: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]], decision: dict[str, Any],
) -> None:
    values = {row["variant"]: row for row in macro if row["scope"] == "global_image_macro"}
    lines = ["# Four-class ContextUNet controlled evaluation", "", "## Macro held-out metrics", "", "| Variant | Fibre Dice | Clump Dice | Uncertain recall | Fibre in uncertain | Skeleton Dice |", "|---|---:|---:|---:|---:|---:|"]
    for variant in ("U0", "S", "G2_four_way_argmax", "G2_four_way_calibrated_skeleton", VARIANT):
        if variant in values:
            row = values[variant]
            lines.append(f"| {variant} | {fmt(row.get('fibrous_dice'))} | {fmt(row.get('clump_dice'))} | {fmt(row.get('uncertain_recall'))} | {fmt(row.get('expert_uncertain_assigned_fibre'))} | {fmt(row.get('skeleton_dice'))} |")
    lines += ["", "## Decision Criteria", "", "| Criterion | Passed |", "|---|---:|"]
    for key, value in decision["criteria"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", f"Decision: `{decision['decision']}`.", "", "## Collapse", "", f"Collapsed images: {sum(bool(row['collapsed']) for row in collapse)}/{len(collapse)}.", "", "## Outputs", "", "- `outer_test_per_image.csv`", "- `outer_test_per_fold.csv`", "- `outer_test_macro.json`", "- `paired_comparisons.csv`", "- `uncertainty_discrimination.csv`", "- `probability_decomposition.csv`", "- `collapse_diagnostics.csv`", "- `bootstrap_confidence_intervals.csv`", "- `qualitative_qa_panels/`"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_value(value: str) -> Any:
    if value in {"", "not_applicable"}:
        return value or "not_applicable"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def fmt(value: Any) -> str:
    return f"{float(value):.4f}" if hi.is_number(value) else "n/a"


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
