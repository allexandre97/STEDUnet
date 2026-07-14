#!/usr/bin/env python
"""Analyze the confirmatory G2 loss-balance experiment from saved maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fibras.annotations import build_real_annotation_sample
from scripts.analyze_b1_uncertain_ignore import load_probabilities, load_records
import scripts.analyze_gated_hierarchical_inference as hi
from scripts.analyze_gated_context_evaluation import collapse_diagnostics


GATED_PROBABILITIES = (
    "foreground_probability", "quantifiability_probability", "one_minus_g_probability",
    "conditional_fibrous_probability", "conditional_clump_probability",
    "final_background_probability", "final_fibrous_probability", "final_clump_probability",
    "final_uncertain_probability", "raw_centreline_probability", "final_skeleton_probability",
)
TARGETS = {"background": 0, "confident_fibre": 1, "confident_clump": 3, "expert_uncertain": 255}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    records = load_records(args.manifest, args.fold_manifest, args.image_root, args.annotation_root)
    selected_records = [record for record in records if record.partition in {"validation", "test"}]
    samples = {record.sample_id: build_real_annotation_sample(
        record.image_path, record.snakes_path, record.labels_path
    ) for record in selected_records}
    g2 = load_experiment_predictions(args.experiment_root, selected_records)
    g1_root = args.suite_root / "gated_context_unet_v1"
    g1 = load_experiment_predictions(g1_root, selected_records)

    discrimination = tagged_discrimination("G1", selected_records, samples, g1)
    discrimination += tagged_discrimination("G2", selected_records, samples, g2)
    semantic_sweeps, semantic_selected = hi.calibrate_semantics(args, records, samples, g2)
    skeleton_rows, skeleton_selected = hi.calibrate_skeletons(records, samples, g2, semantic_selected)
    for fold, selection in skeleton_selected.items():
        joint = next(item for item in selection["candidates"] if item["strategy"] == "joint_final_probability")
        skeleton_selected[fold] = {"status": "selected", "selected_from": "inner_validation",
                                   "candidates": selection["candidates"], **joint}
    skeleton_rows.extend(hi.evaluate_selected_skeletons(
        records, samples, g2, semantic_selected, skeleton_selected
    ))
    skeleton_rows.extend(fixed_skeleton_diagnostics(records, samples, g2))

    outer_rows = g2_outer_rows(args, records, samples, g2, semantic_selected, skeleton_selected)
    outer_rows += g1_rows(args, records, samples, g1, g1_root)
    macro = summarized_rows(outer_rows)
    paired = paired_rows(outer_rows)
    probability = probability_comparison(selected_records, samples, g1, g2)
    probability += calibration_quality(selected_records, samples, g1, g2)
    collapse = collapse_comparison(selected_records, g1, g2)
    boundary = boundary_rows(outer_rows)
    provenance = checkpoint_provenance(args.experiment_root)
    decision = viability_decision(macro, semantic_selected, collapse)

    out = args.experiment_root
    hi.write_csv(out / "probability_calibration_comparison.csv", probability)
    hi.write_csv(out / "gate_discrimination.csv", discrimination)
    hi.write_csv(out / "spatial_gate_discrimination.csv", [
        row for row in discrimination if "spatial_tau" in row["comparison"]
    ])
    hi.write_csv(out / "validation_semantic_threshold_sweeps.csv", semantic_sweeps)
    hi.write_json(out / "selected_semantic_thresholds.json", semantic_selected)
    hi.write_csv(out / "skeleton_threshold_analysis.csv", skeleton_rows)
    hi.write_json(out / "selected_skeleton_thresholds.json", skeleton_selected)
    hi.write_csv(out / "outer_test_per_image.csv", outer_rows)
    hi.write_csv(out / "outer_test_per_fold.csv", [
        row for row in macro if row["scope"] == "fold_image_macro"
    ])
    hi.write_json(out / "outer_test_macro.json", [
        row for row in macro if row["scope"] == "global_image_macro"
    ])
    hi.write_csv(out / "paired_comparisons.csv", paired)
    hi.write_csv(out / "boundary_safety.csv", boundary)
    hi.write_csv(out / "collapse_diagnostics.csv", collapse)
    hi.write_json(out / "checkpoint_provenance.json", provenance)
    hi.write_json(out / "viability_decision.json", decision)
    hi.write_panels(out / "qualitative_qa_panels", records, samples, g2,
                    semantic_selected, skeleton_selected)
    write_report(out / "controlled_recalibration_report.md", macro, discrimination,
                 semantic_selected, skeleton_selected, probability, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    parser = hi.parser()
    parser.set_defaults(experiment_root=Path(
        "/ssd/STED_experiments/controlled_real_sted_v1/gated_context_unet_g2"
    ))
    return parser


def load_experiment_predictions(root: Path, records: list[Any]) -> dict[tuple[int, str, str], dict[str, np.ndarray]]:
    return {(record.outer_fold, record.partition, record.sample_id): hi.load_gated(
        root / "evaluations" / f"fold_{record.outer_fold}" / record.partition /
        record.sample_id / "predictions.npz"
    ) for record in records}


def tagged_discrimination(
    experiment: str, records: list[Any], samples: dict[str, dict[str, Any]],
    predictions: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    return [{"experiment": experiment, **row} for row in hi.gate_discrimination(records, samples, predictions)]


def g2_outer_rows(
    args: argparse.Namespace, records: list[Any], samples: dict[str, dict[str, Any]],
    g2: dict[tuple[int, str, str], dict[str, np.ndarray]], semantics: dict[int, dict[str, Any]],
    skeletons: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = hi.outer_comparison(args, records, samples, g2, semantics, skeletons)
    renames = {
        "G_four_way_argmax": "G2_four_way_argmax", "G_hierarchical": "G2_hierarchical",
        "G_hierarchical_calibrated_skeleton": "G2_hierarchical_calibrated_skeleton",
    }
    for row in rows:
        row["variant"] = renames.get(row["variant"], row["variant"])
        if row["status"] == "available":
            row["predicted_fibre_fraction_inside_expert_uncertain"] = row["expert_uncertain_assigned_fibre"]
            row["predicted_uncertain_fraction_inside_expert_uncertain"] = row["expert_uncertain_assigned_uncertain"]
        if row["variant"].startswith("G2") and row["status"] == "available":
            pred = g2[(row["outer_fold"], "test", row["sample_id"])]
            semantic = None
            if row["variant"].startswith("G2_hierarchical"):
                semantic = hi.selected_semantic(pred, semantics[row["outer_fold"]])
            add_probability_region_metrics(
                row, samples[row["sample_id"]]["real_semantic_mask"], pred, semantic
            )
    for record in (item for item in records if item.partition == "test"):
        pred = g2[(record.outer_fold, "test", record.sample_id)]
        semantic = pred["semantic_class_map"]
        line = hi.selected_skeleton(pred, semantic, skeletons[record.outer_fold])
        row = hi.metric_row(record, "G2_four_way_calibrated_skeleton",
                            samples[record.sample_id]["real_semantic_mask"],
                            samples[record.sample_id]["real_skeleton_mask"], semantic,
                            1 - pred["quantifiability_probability"], line)
        add_probability_region_metrics(row, samples[record.sample_id]["real_semantic_mask"], pred)
        rows.append(row)
    return rows


def g1_rows(
    args: argparse.Namespace, records: list[Any], samples: dict[str, dict[str, Any]],
    g1: dict[tuple[int, str, str], dict[str, np.ndarray]], root: Path,
) -> list[dict[str, Any]]:
    selections = json.loads((root / "selected_thresholds.json").read_text())
    skeletons = {int(fold): value["skeleton"] for fold, value in selections.items()}
    rows = []
    for record in (item for item in records if item.partition == "test"):
        pred = g1[(record.outer_fold, "test", record.sample_id)]
        target = samples[record.sample_id]["real_semantic_mask"]
        target_line = samples[record.sample_id]["real_skeleton_mask"]
        for variant, line in (
            ("G1_four_way_argmax", pred["final_skeleton_probability"] >= 0.75),
            ("G1_four_way_calibrated_skeleton",
             hi.selected_skeleton(pred, pred["semantic_class_map"], skeletons[record.outer_fold])),
        ):
            row = hi.metric_row(record, variant, target, target_line, pred["semantic_class_map"],
                                1 - pred["quantifiability_probability"], line)
            add_probability_region_metrics(row, target, pred)
            rows.append(row)
    return rows


def add_probability_region_metrics(
    row: dict[str, Any], target: np.ndarray, pred: dict[str, np.ndarray], semantic: np.ndarray | None = None,
) -> None:
    uncertain, fibre = target == 255, target == 1
    semantic = pred["semantic_class_map"] if semantic is None else semantic
    row.update({
        "predicted_fibre_fraction_inside_expert_uncertain": hi.assignment(semantic, uncertain, 1),
        "predicted_uncertain_fraction_inside_expert_uncertain": hi.assignment(semantic, uncertain, 255),
        "expert_confident_clump_assigned_uncertain": hi.assignment(semantic, target == 3, 255),
        "final_fibre_probability_inside_expert_uncertain": region_mean(pred["final_fibrous_probability"], uncertain),
        "final_uncertain_probability_inside_expert_fibre": region_mean(pred["final_uncertain_probability"], fibre),
    })
    for threshold in (0.5, 0.7, 0.8, 0.9):
        row[f"expert_uncertain_final_fibre_ge_{threshold:g}_fraction"] = hi.ratio(
            int((uncertain & (pred["final_fibrous_probability"] >= threshold)).sum()), int(uncertain.sum())
        )


def fixed_skeleton_diagnostics(
    records: list[Any], samples: dict[str, dict[str, Any]],
    predictions: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for threshold in (0.5, 0.75, 0.85):
        image_rows = []
        for record in (item for item in records if item.partition == "test"):
            target = samples[record.sample_id]["real_semantic_mask"]
            pred = predictions[(record.outer_fold, "test", record.sample_id)]
            metrics = hi.skeleton_metrics(
                target, (samples[record.sample_id]["real_skeleton_mask"] > 0) & (target == 1),
                pred["final_skeleton_probability"] >= threshold,
            )
            row = {"partition": "test", "scope": "image", "outer_fold": record.outer_fold,
                   "sample_id": record.sample_id, "strategy": "joint_fixed_diagnostic",
                   "threshold": threshold, "selected_strategy": False, **metrics}
            rows.append(row); image_rows.append(row)
        rows.append({"partition": "test", "scope": "global_image_macro", "outer_fold": "all",
                     "sample_id": "all", "strategy": "joint_fixed_diagnostic", "threshold": threshold,
                     "selected_strategy": False,
                     **{key: hi.numeric_mean(row[key] for row in image_rows) for key in hi.SKELETON_METRICS}})
    return rows


def probability_comparison(
    records: list[Any], samples: dict[str, dict[str, Any]],
    g1: dict[tuple[int, str, str], dict[str, np.ndarray]],
    g2: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for partition in ("validation", "test"):
        partition_records = [record for record in records if record.partition == partition]
        for experiment, predictions in (("G1", g1), ("G2", g2)):
            for scope, fold, chosen in [("global_pooled", "all", partition_records)] + [
                ("fold_pooled", value, [record for record in partition_records if record.outer_fold == value])
                for value in range(5)
            ]:
                for target_name, target_id in TARGETS.items():
                    for probability in GATED_PROBABILITIES:
                        values = np.concatenate([
                            probability_values(predictions[(record.outer_fold, partition, record.sample_id)], probability)[
                                samples[record.sample_id]["real_semantic_mask"] == target_id
                            ].astype(np.float32, copy=False) for record in chosen
                        ])
                        quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
                        rows.append({
                            "row_type": "distribution",
                            "experiment": experiment, "partition": partition, "scope": scope,
                            "outer_fold": fold, "target_class": target_name, "probability": probability,
                            "pixels": values.size, "mean": float(values.mean()), "std": float(values.std()),
                            "p05": float(quantiles[0]), "p25": float(quantiles[1]), "p50": float(quantiles[2]),
                            "p75": float(quantiles[3]), "p95": float(quantiles[4]),
                        })
    return rows


def calibration_quality(
    records: list[Any], samples: dict[str, dict[str, Any]],
    g1: dict[tuple[int, str, str], dict[str, np.ndarray]],
    g2: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for partition in ("validation", "test"):
        partition_records = [record for record in records if record.partition == partition]
        for experiment, predictions in (("G1", g1), ("G2", g2)):
            scopes = [("global_pooled", "all", partition_records)] + [
                ("fold_pooled", fold, [record for record in partition_records if record.outer_fold == fold])
                for fold in range(5)
            ]
            for scope, fold, chosen in scopes:
                probability_parts, target_parts = [], []
                for record in chosen:
                    pred = predictions[(record.outer_fold, partition, record.sample_id)]
                    probability_parts.append(np.stack([
                        pred["final_background_probability"], pred["final_fibrous_probability"],
                        pred["final_clump_probability"], pred["final_uncertain_probability"],
                    ], axis=-1).reshape(-1, 4))
                    target = samples[record.sample_id]["real_semantic_mask"]
                    target_parts.append(np.select(
                        [target == 0, target == 1, target == 3, target == 255], [0, 1, 2, 3]
                    ).ravel())
                probs, target = np.concatenate(probability_parts), np.concatenate(target_parts)
                true_probability = probs[np.arange(target.size), target]
                one_hot = np.eye(4, dtype=np.float32)[target]
                confidence, predicted = probs.max(axis=1), probs.argmax(axis=1)
                rows.append({
                    "row_type": "calibration", "experiment": experiment, "partition": partition,
                    "scope": scope, "outer_fold": fold, "target_class": "all",
                    "probability": "four_way_final", "pixels": target.size,
                    "multiclass_nll": float(-np.log(np.clip(true_probability, 1e-7, 1)).mean()),
                    "multiclass_brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
                    "expected_calibration_error": expected_calibration_error(confidence, predicted == target),
                    **{f"class_{name}_brier": float(np.square(probs[:, index] - one_hot[:, index]).mean())
                       for index, name in enumerate(("background", "fibre", "clump", "uncertain"))},
                })
    return rows


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray) -> float:
    total, result = confidence.size, 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return result


def probability_values(pred: dict[str, np.ndarray], name: str) -> np.ndarray:
    return 1 - pred["quantifiability_probability"] if name == "one_minus_g_probability" else pred[name]


def collapse_comparison(
    records: list[Any], g1: dict[tuple[int, str, str], dict[str, np.ndarray]],
    g2: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for experiment, predictions in (("G1", g1), ("G2", g2)):
        for record in (item for item in records if item.partition == "test"):
            row = collapse_diagnostics(record, predictions[(record.outer_fold, "test", record.sample_id)])
            rows.append({"experiment": experiment, **row})
    return rows


def summarized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return (summarize_groups(rows, ("variant",), "global_image_macro")
            + summarize_groups(rows, ("variant", "outer_fold"), "fold_image_macro"))


def summarize_groups(rows: list[dict[str, Any]], keys: tuple[str, ...], scope: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    excluded = {"variant", "outer_fold", "sample_id", "status"}
    metrics = sorted({key for row in rows for key in row} - excluded)
    return [{"scope": scope, **dict(zip(keys, group)), "image_count": len(selected),
             "available_images": sum(row["status"] == "available" for row in selected),
             **{metric: hi.numeric_mean(row.get(metric) for row in selected) for metric in metrics}}
            for group, selected in sorted(groups.items())]


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["variant"], row["sample_id"]): row for row in rows}
    samples = sorted({row["sample_id"] for row in rows})
    metrics = (*hi.SEMANTIC_METRICS, *hi.SKELETON_METRICS,
               "predicted_fibre_fraction_inside_expert_uncertain")
    output = []
    for reference in ("U0", "S", "G1_four_way_argmax", "G1_four_way_calibrated_skeleton"):
        for variant in ("G2_four_way_argmax", "G2_hierarchical", "G2_four_way_calibrated_skeleton"):
            for metric in metrics:
                differences = []
                comparison = f"{variant}-minus-{reference}"
                for sample in samples:
                    value, baseline = lookup[(variant, sample)].get(metric), lookup[(reference, sample)].get(metric)
                    if not (hi.is_number(value) and hi.is_number(baseline)):
                        continue
                    difference = float(value) - float(baseline)
                    differences.append(difference)
                    output.append({"scope": "image", "comparison": comparison, "metric": metric,
                                   "sample_id": sample, "outer_fold": lookup[(variant, sample)]["outer_fold"],
                                   "value": value, "reference": baseline, "difference": difference})
                output.append({"scope": "global_image_macro", "comparison": comparison, "metric": metric,
                               "mean_paired_difference": float(np.mean(differences)) if differences else "not_applicable",
                               "applicable_images": len(differences)})
    return output


def boundary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        chosen = [row for row in rows if row["variant"] == variant]
        for name, _, _ in hi.BOUNDARY_BANDS:
            metric = f"confident_fibrous_recall_{name}px"
            output.append({"variant": variant, "band": name,
                           "macro_recall": hi.numeric_mean(row.get(metric) for row in chosen)})
    return output


def checkpoint_provenance(root: Path) -> dict[str, Any]:
    folds = {}
    for fold in range(5):
        run = root / f"runs/fold_{fold}"
        metadata = json.loads((run / "run_metadata.json").read_text())
        summary = json.loads((run / "checkpoint_summary.json").read_text())
        folds[str(fold)] = {"configuration": metadata["command_line_args"],
                            "initialization_report": metadata["initial_checkpoint"],
                            "checkpoint_summary": summary}
    return {"folds": folds}


def viability_decision(
    macro: list[dict[str, Any]], semantics: dict[int, dict[str, Any]], collapse: list[dict[str, Any]],
) -> dict[str, Any]:
    values = {row["variant"]: row for row in macro if row["scope"] == "global_image_macro"}
    u0, s = values["U0"], values["S"]
    candidates = ["G2_four_way_calibrated_skeleton"]
    if all(value["status"] == "selected" for value in semantics.values()):
        candidates.append("G2_hierarchical_calibrated_skeleton")
    results = {}
    for variant in candidates:
        row = values[variant]
        checks = {
            "fibrous_dice": float(row["fibrous_dice"]) >= float(u0["fibrous_dice"]) - 0.01,
            "clump_dice": float(row["clump_dice"]) >= float(u0["clump_dice"]) - 0.01,
            "skeleton_dice": float(row["skeleton_dice"]) >= float(u0["skeleton_dice"]) - 0.01,
            "material_uncertain_fibre_reduction": (
                float(s["predicted_fibre_fraction_inside_expert_uncertain"])
                - float(row["predicted_fibre_fraction_inside_expert_uncertain"]) >= 0.02
            ),
            "uncertain_recall": float(row["uncertain_recall"]) > 0.115,
            "boundary_safety": float(row["confident_fibrous_recall_0_2px"]) >= float(u0["confident_fibrous_recall_0_2px"]) - 0.05,
            "no_collapse": not any(item["collapsed"] for item in collapse if item["experiment"] == "G2"),
        }
        results[variant] = {"viable": all(checks.values()), "checks": checks}
    viable = [variant for variant, result in results.items() if result["viable"]]
    return {"g2_viable": bool(viable), "valid_inference_rules": viable, "candidate_results": results,
            "hierarchical_valid_folds": sum(value["status"] == "selected" for value in semantics.values()),
            "strict_decision": ("G2 viable" if viable else
                                "G2 non-viable and gated factorisation should be abandoned")}


def write_report(
    path: Path, macro: list[dict[str, Any]], discrimination: list[dict[str, Any]],
    semantics: dict[int, dict[str, Any]], skeletons: dict[int, dict[str, Any]],
    probability: list[dict[str, Any]], decision: dict[str, Any],
) -> None:
    values = {row["variant"]: row for row in macro if row["scope"] == "global_image_macro"}
    lines = ["# G2 controlled recalibration", "", f"## Decision\n\n**{decision['strict_decision']}**", "",
             "## Outer-test macro", "",
             "| Variant | Fibre Dice | Clump Dice | Uncertain recall | Skeleton Dice |", "|---|---:|---:|---:|---:|"]
    for variant in ("U0", "S", "G1_four_way_argmax", "G1_four_way_calibrated_skeleton",
                    "G2_four_way_argmax", "G2_four_way_calibrated_skeleton", "G2_hierarchical"):
        row = values[variant]
        lines.append(f"| {variant} | {hi.format_number(row['fibrous_dice'])} | {hi.format_number(row['clump_dice'])} | "
                     f"{hi.format_number(row['uncertain_recall'])} | {hi.format_number(row['skeleton_dice'])} |")
    distributions = {(row["experiment"], row["target_class"], row["probability"]): row
                     for row in probability if row.get("row_type") == "distribution"
                     and row["partition"] == "test" and row["scope"] == "global_pooled"}
    lines += ["", "## Pooled probability changes", "",
              "| Target | Probability | G1 mean | G2 mean | G2−G1 |", "|---|---|---:|---:|---:|"]
    for target in TARGETS:
        for probability_name in ("foreground_probability", "quantifiability_probability"):
            g1 = distributions[("G1", target, probability_name)]["mean"]
            g2 = distributions[("G2", target, probability_name)]["mean"]
            lines.append(f"| {target} | {probability_name} | {g1:.4f} | {g2:.4f} | {g2-g1:+.4f} |")
    disc = {(row["experiment"], row["comparison"]): row for row in discrimination
            if row["partition"] == "test" and row["scope"] == "global_pooled"}
    lines += ["", "## Pooled outer-test discrimination", "",
              "| Experiment | Comparison | AUROC | AP |", "|---|---|---:|---:|"]
    for experiment in ("G1", "G2"):
        for comparison in ("q_background_vs_review_worthy", "one_minus_g_uncertain_vs_confident_tau",
                           "conditional_morphology_fibre_vs_clump"):
            row = disc[(experiment, comparison)]
            lines.append(f"| {experiment} | {comparison} | {hi.format_number(row['auroc'])} | "
                         f"{hi.format_number(row['average_precision'])} |")
    lines += ["", "## Validation selections", ""]
    for fold in range(5):
        semantic, skeleton = semantics[fold], skeletons[fold]
        lines.append(f"- Fold {fold}: semantic={semantic['status']}; skeleton={skeleton['threshold']:.3f}.")
    lines += ["", "## Decision checks", ""]
    for variant, result in decision["candidate_results"].items():
        lines.append(f"- {variant}: viable={result['viable']}; " + ", ".join(
            f"{name}={passed}" for name, passed in result["checks"].items()
        ))
    lines += ["", "## Interpretation", "",
              "G2 was judged only against validation-selected checkpoints and thresholds. "
              "If non-viable, no further gated loss combination should be run; the next model should be a directly supervised four-class semantic baseline."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def region_mean(values: np.ndarray, mask: np.ndarray) -> float | str:
    return float(values[mask].mean()) if mask.any() else "not_applicable"


if __name__ == "__main__":
    raise SystemExit(main())
