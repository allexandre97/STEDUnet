#!/usr/bin/env python
"""Audit completed uncertain-ignore nested-CV metrics without inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.analyze_b1_uncertain_ignore import average_precision, auroc, load_probabilities, load_records
from scripts.run_uncertain_ignore_nested_cv import load_samples, u0_prediction
from scripts.uncertain_ignore_nested_metrics import complete_image_metrics, public_metrics, summarize_images


COMPARISONS = {
    "uncertain_vs_fibrous": (1,), "uncertain_vs_clump": (3,),
    "uncertain_vs_fibrous_clump": (1, 3), "uncertain_vs_background": (0,),
    "uncertain_vs_all_confident": (0, 1, 3),
}
SPATIAL_DISTANCES = (2, 5, 10, 20)
BOUNDARY_KEYS = ("0_2px", "2_5px", "5_10px", "10_20px", "over_20px")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    out, root = args.experiment_root, args.reference_root
    records = [r for r in load_records(args.manifest, args.fold_manifest, args.image_root, args.annotation_root) if r.partition == "test"]
    samples, selections = load_samples(records), json.loads((out / "inner_validation_selections.json").read_text())
    paths = prediction_paths(root, out, records, selections)
    missing = [str(p) for variant in paths.values() for p in variant.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"saved probabilities missing; inference required: {missing[:5]}")

    rows = metric_rows(records, samples, paths, selections)
    preserve_originals(out)
    write_corrected_metrics(out, rows)
    discr, spatial = discrimination_rows(records, samples, paths)
    u0_primary = find_row(discr, "U0", "uncertain_vs_fibrous_clump", "global_pooled")
    if not 0.49 <= float(u0_primary["auroc"]) <= 0.54:
        raise RuntimeError(f"U0 foreground AUROC {u0_primary['auroc']} does not reproduce corrected B1 diagnostic")
    write_csv(out / "uncertainty_discrimination.csv", discr)
    write_csv(out / "uncertainty_spatial_discrimination.csv", spatial)
    audit = regression_audit(out, root, rows)
    write_csv(out / "u0_regression_audit.csv", audit)
    write_csv(out / "paired_boundary_effects.csv", boundary_effects(rows))
    write_csv(out / "metric_definitions.csv", metric_definitions(rows))
    write_report(out, rows, discr, spatial, audit)
    return 0


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-root", type=Path, default=Path("/ssd/STED_experiments/controlled_real_sted_v1/uncertain_ignore_nested_cv"))
    p.add_argument("--reference-root", type=Path, default=Path("/ssd/STED_experiments/controlled_real_sted_v1"))
    p.add_argument("--manifest", type=Path, default=root / "data_manifests/real_annotations.csv")
    p.add_argument("--fold-manifest", type=Path, default=root / "data_manifests/real_annotation_folds_v1.csv")
    p.add_argument("--image-root", type=Path, default=Path("/ssd/STED_dataset/data"))
    p.add_argument("--annotation-root", type=Path, default=Path("/cephfs/mhuang/STED_dataset/manual_annotations"))
    return p


def prediction_paths(root: Path, out: Path, records: list[Any], selections: dict[str, Any]) -> dict[str, dict[str, Path]]:
    result = {v: {} for v in ("U0", "S", "H", "SH")}
    for r in records:
        fold = str(r.outer_fold)
        result["U0"][r.sample_id] = u0_prediction(root, out, r)
        result["S"][r.sample_id] = out / "evaluations" / f"S_s{selections[fold]['S']['lambda_uncertain_fibrous']:.2f}" / f"fold_{fold}" / "test" / r.sample_id / "predictions.npz"
        result["H"][r.sample_id] = out / "evaluations" / f"H_h{selections[fold]['H']['lambda_rejection']:.2f}" / f"fold_{fold}" / "test" / r.sample_id / "predictions.npz"
        result["SH"][r.sample_id] = out / "evaluations" / f"SH_s{selections[fold]['S']['lambda_uncertain_fibrous']:.2f}_h{selections[fold]['H']['lambda_rejection']:.2f}" / f"fold_{fold}" / "test" / r.sample_id / "predictions.npz"
    return result


def metric_rows(records: list[Any], samples: dict[str, Any], paths: dict[str, Any], selections: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant, variant_paths in paths.items():
        for r in records:
            threshold = selections[str(r.outer_fold)]["H"]["rejection_threshold"] if variant in {"H", "SH"} else None
            sample = samples[r.sample_id]
            metrics = complete_image_metrics(sample["real_semantic_mask"], sample["real_skeleton_mask"], load_probabilities(variant_paths[r.sample_id]), threshold)
            rows.append({"variant": variant, "outer_fold": r.outer_fold, "sample_id": r.sample_id, "rejection_threshold": threshold if threshold is not None else "not_applicable", **metrics})
    return rows


def preserve_originals(out: Path) -> None:
    for name in ("outer_test_macro.json", "outer_test_per_fold.csv", "outer_test_per_image.csv"):
        original, backup = out / name, out / name.replace(".", ".pre_audit.", 1)
        if original.exists() and not backup.exists():
            shutil.copy2(original, backup)


def write_corrected_metrics(out: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out / "outer_test_per_image.csv", [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    folds = [{"variant": v, "outer_fold": f, **summarize_images([r for r in rows if r["variant"] == v and r["outer_fold"] == f])} for v in sorted({r["variant"] for r in rows}) for f in range(5)]
    write_csv(out / "outer_test_per_fold.csv", folds)
    macro = {v: summarize_images([r for r in rows if r["variant"] == v]) for v in sorted({r["variant"] for r in rows})}
    (out / "outer_test_macro.json").write_text(json.dumps(macro, indent=2, sort_keys=True) + "\n")


def discrimination_rows(records: list[Any], samples: dict[str, Any], paths: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    regular, spatial = [], []
    for variant in ("U0", "H", "SH"):
        parts = {name: [] for name in COMPARISONS}
        near = {d: [] for d in SPATIAL_DISTANCES}
        for r in records:
            target = samples[r.sample_id]["real_semantic_mask"]
            score = load_probabilities(paths[variant][r.sample_id])["uncertainty_probability"]
            uncertain = target == 255
            for name, classes in COMPARISONS.items():
                mask = uncertain | np.isin(target, classes)
                part = (r, uncertain[mask], score[mask])
                parts[name].append(part); regular.append(discrimination_row(variant, name, "image", r.sample_id, r.outer_fold, part[1], part[2]))
            distance = distance_transform_edt(~uncertain)
            for d in SPATIAL_DISTANCES:
                mask = uncertain | (np.isin(target, (1, 3)) & (distance <= d))
                part = (r, uncertain[mask], score[mask])
                near[d].append(part); spatial.append(discrimination_row(variant, f"within_{d}px", "image", r.sample_id, r.outer_fold, part[1], part[2]))
        for name, values in parts.items(): regular.extend(aggregate_discrimination(variant, name, values))
        for d, values in near.items(): spatial.extend(aggregate_discrimination(variant, f"within_{d}px", values))
    return regular, spatial


def discrimination_row(variant: str, comparison: str, scope: str, sample_id: str, fold: Any, y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    positive, total = int(y.sum()), int(y.size)
    return {"variant": variant, "comparison": comparison, "scope": scope, "sample_id": sample_id, "outer_fold": fold, "positive_pixels": positive, "negative_pixels": total-positive, "prevalence": positive/total if total else "not_applicable", "auroc": auroc(y, score), "average_precision": average_precision(y, score)}


def aggregate_discrimination(variant: str, comparison: str, parts: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for fold in range(5):
        selected = [p for p in parts if p[0].outer_fold == fold]
        rows.append(pooled_discrimination(variant, comparison, "fold_pooled", f"fold_{fold}", fold, selected))
        rows.append(macro_discrimination(variant, comparison, "fold_macro", f"fold_{fold}", fold, selected))
    rows.append(pooled_discrimination(variant, comparison, "global_pooled", "all", "all", parts))
    rows.append(macro_discrimination(variant, comparison, "global_macro", "all", "all", parts))
    return rows


def pooled_discrimination(variant: str, comparison: str, scope: str, sample: str, fold: Any, parts: list[Any]) -> dict[str, Any]:
    return discrimination_row(variant, comparison, scope, sample, fold, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]))


def macro_discrimination(variant: str, comparison: str, scope: str, sample: str, fold: Any, parts: list[Any]) -> dict[str, Any]:
    image = [discrimination_row(variant, comparison, "image", p[0].sample_id, p[0].outer_fold, p[1], p[2]) for p in parts]
    valid = lambda key: [float(r[key]) for r in image if isinstance(r[key], (float, int))]
    positive, negative = sum(r["positive_pixels"] for r in image), sum(r["negative_pixels"] for r in image)
    return {"variant": variant, "comparison": comparison, "scope": scope, "sample_id": sample, "outer_fold": fold, "positive_pixels": positive, "negative_pixels": negative, "prevalence": float(np.mean(valid("prevalence"))), "auroc": float(np.mean(valid("auroc"))), "average_precision": float(np.mean(valid("average_precision")))}


def regression_audit(out: Path, root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = list(csv.DictReader((root / "analysis/per_image_results.csv").open()))
    b1 = {r["sample_id"]: r for r in reference if r.get("experiment") == "B"}
    mapping = {"fibrous_dice":"fibrous_dice", "fibrous_precision":"fibrous_precision", "fibrous_recall":"fibrous_recall", "fibrous_area_ratio":"predicted_to_target_fibrous_area_ratio", "clump_dice":"clump_dice", "skeleton_dice_0.5":"skeleton_dice_0.5", "skeleton_dice_0.75":"skeleton_dice_0.75", "skeleton_dice_0.85":"skeleton_dice_0.85", "target_skeleton_recovered_2px":"target_skeleton_recovered_within_2px_0.75", "predicted_skeleton_within_target_2px":"predicted_skeleton_within_2px_of_target_0.75", "fibrous_fraction_inside_uncertain_ignore":"fibrous_inside_uncertain_ignore_fraction", "fibrous_leakage_into_clump":"fibrous_inside_target_clump_fraction"}
    audit = []
    for row in (r for r in rows if r["variant"] == "U0"):
        for metric, ref_metric in mapping.items():
            actual, expected = row[metric], b1[row["sample_id"]][ref_metric]
            comparable = isinstance(actual, (float, int)) and expected not in {"", "not_applicable"}
            delta = float(actual)-float(expected) if comparable else "not_applicable"
            match = abs(delta) <= 1e-12 if comparable else str(actual) == expected
            audit.append({"sample_id":row["sample_id"], "metric":metric, "nested_value":actual, "b1_value":expected, "absolute_difference":abs(delta) if comparable else "not_applicable", "match":match, "explanation":"exact reproduction" if match else "metric-definition discrepancy"})
    return audit


def boundary_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["variant"], r["sample_id"]): r for r in rows}
    result = []
    for variant in ("S", "H", "SH"):
        for row in (r for r in rows if r["variant"] == variant):
            for band in BOUNDARY_KEYS:
                key = f"confident_fibrous_recall_{band}"
                base, value = lookup[("U0", row["sample_id"])][key], row[key]
                result.append({"variant":variant, "outer_fold":row["outer_fold"], "sample_id":row["sample_id"], "band":band, "variant_recall":value, "u0_recall":base, "paired_difference":float(value)-float(base) if isinstance(value,(float,int)) and isinstance(base,(float,int)) else "not_applicable"})
    return result


def metric_definitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for key in sorted(k for k in rows[0] if not k.startswith("_") and k not in {"variant","outer_fold","sample_id"}):
        applicable = sum(isinstance(r[key], (float,int)) for r in rows if r["variant"] == "U0")
        aggregation = "macro_per_image"
        missing = "exclude non-applicable values"
        denominator = "applicable images"
        if key == "clump_dice": denominator = "images with target clump pixels"
        if key == "clump_dice_pooled": aggregation, denominator = "pooled", "sum predicted + target clump pixels"
        result.append({"metric":key, "headline_aggregation":aggregation, "applicable_u0_images":applicable, "denominator":denominator, "missing_handling":missing})
    return result


def write_report(out: Path, rows: list[dict[str, Any]], discr: list[dict[str, Any]], spatial: list[dict[str, Any]], audit: list[dict[str, Any]]) -> None:
    macro = json.loads((out/"outer_test_macro.json").read_text())
    primary = {v: find_row(discr,v,"uncertain_vs_fibrous_clump","global_pooled") for v in ("U0","H","SH")}
    spatial_u0 = [find_row(spatial,"U0",f"within_{d}px","global_pooled") for d in SPATIAL_DISTANCES]
    failures = [r for r in audit if not r["match"]]
    text = f"""# Nested-CV metric audit

The original metric tables are preserved as `*.pre_audit.*`; the canonical filenames now contain corrected metrics.

## Causes

- Clump Dice was returned as zero when the target contained no clump pixels. Those 11 non-applicable images were then included in the 20-image macro. Correct aggregation excludes them.
- `expert_uncertain_auroc` compared uncertain pixels with every other pixel and was background dominated. It is retained only as a legacy descriptive column; foreground-conditional tables supersede it for head assessment.
- Semantic IDs are consistent with B1: background 0, fibrous tau 1, clump 3, and uncertain-ignore 255. PSP images have no target clump pixels and are non-applicable, as are other images without clump annotation.

## Corrected checks

- U0 clump Dice: {macro['U0']['clump_dice']:.6f} over {macro['U0']['clump_applicable_images']} applicable images; pooled {macro['U0']['clump_dice_pooled']:.6f}.
- Foreground-conditional pooled AUROC/AP: U0 {primary['U0']['auroc']:.6f}/{primary['U0']['average_precision']:.6f}; H {primary['H']['auroc']:.6f}/{primary['H']['average_precision']:.6f}; SH {primary['SH']['auroc']:.6f}/{primary['SH']['average_precision']:.6f}.
- U0 spatial pooled AUROC at 2/5/10/20 px: {'/'.join(f"{float(r['auroc']):.6f}" for r in spatial_u0)}.
- U0/B1 regression mismatches: {len(failures)} of {len(audit)} comparisons.

## Rejection semantics

H and SH use the preserved fold thresholds 0.84/0.82/0.84/0.87/0.86. Rejected target fibres and skeleton pixels remain target positives, rejected predictions are removed, and complete-image denominators remain unchanged. `*_raw` probability metrics are computed before rejection; unsuffixed operational metrics are computed after rejection.

## Readiness

{'The corrected results are ready for final scientific analysis.' if not failures else 'Not ready: unresolved U0 regression mismatches remain; see `u0_regression_audit.csv`.'}
"""
    (out/"metric_audit_report.md").write_text(text)


def find_row(rows: list[dict[str, Any]], variant: str, comparison: str, scope: str) -> dict[str, Any]:
    return next(r for r in rows if r["variant"] == variant and r["comparison"] == comparison and r["scope"] == scope)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
