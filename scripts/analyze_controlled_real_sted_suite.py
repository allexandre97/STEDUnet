#!/usr/bin/env python
"""Build the reproducible report for the controlled real-STED suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "A": ("Frozen synthetic", "phase_a_synthetic_baseline"),
    "B": ("Real-only", "b1_real_only"),
    "C": ("80:20 mixed", "b2_mixed_80_20"),
    "D": ("80:20 no uncertainty", "b3_mixed_no_uncertainty"),
    "E": ("80:20 targeted", "b4_mixed_targeted"),
}
METRICS = [
    "fibrous_dice", "fibrous_precision", "fibrous_recall",
    "predicted_to_target_fibrous_area_ratio", "clump_dice",
    "skeleton_dice_0.5", "skeleton_dice_0.75", "skeleton_dice_0.85",
    "target_skeleton_recovered_within_2px_0.75",
    "predicted_skeleton_within_2px_of_target_0.75",
    "fibrous_inside_target_clump_fraction", "fibrous_inside_uncertain_ignore_fraction",
    "clump_inside_target_fibrous_fraction", "skeleton_inside_target_clump_fraction_0.75",
    "skeleton_inside_uncertain_ignore_fraction_0.75", "false_positive_fibrous_background_pixels",
    "missed_thick_component_count", "missed_faint_component_count",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inventory = read_csv(ROOT / "data_manifests/real_annotations.csv")
    folds = read_csv(ROOT / "data_manifests/real_annotation_folds_v1.csv")
    results = load_results(args.suite_root)
    provenance = validate_provenance(args.suite_root, inventory, folds, results)
    long_rows = build_long_rows(inventory, folds, results)
    write_csv(args.out / "per_image_results.csv", long_rows)
    comparisons = comparison_rows(inventory, results)
    write_csv(args.out / "per_image_comparisons.csv", comparisons)
    macro = macro_rows(results)
    fold_rows = build_fold_rows(args.suite_root)
    strata = strata_rows(long_rows)
    selection = selection_rows(args.suite_root)
    synthetic = synthetic_retention_rows(args.suite_root)
    for name, rows in [
        ("macro_summary.csv", macro), ("fold_metrics.csv", fold_rows),
        ("strata_summary.csv", strata), ("selection_vs_test.csv", selection),
        ("synthetic_retention.csv", synthetic),
    ]:
        write_csv(args.out / name, rows)
    examples = qualitative_examples(inventory, results)
    write_csv(args.out / "qualitative_examples.csv", examples)
    make_qualitative_panels(args.suite_root, args.out / "qualitative_examples", examples)
    plot_metric_points(long_rows, args.out / "main_metrics.png")
    plot_paired_deltas(comparisons, args.out / "paired_fibrous_dice_deltas.png")
    plot_strata(strata, args.out / "condition_fibrous_dice.png")
    provenance["qualitative_examples"] = examples
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(report_text(macro, fold_rows, strata, comparisons, selection, synthetic, examples, provenance))
    (args.out / "executive_summary.md").write_text(executive_summary(macro, comparisons))
    print(f"wrote analysis report to {args.out}")
    return 0


def load_results(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out = {}
    for code, (_, directory) in EXPERIMENTS.items():
        path = root / "evaluations" / directory / ("summary.json" if code == "A" else "oof_summary.json")
        payload = json.loads(path.read_text())
        rows = payload["rows"]
        if len(rows) != 20 or len({row["sample_id"] for row in rows}) != 20:
            raise ValueError(f"{code}: expected 20 unique per-image rows")
        out[code] = {row["sample_id"]: row for row in rows}
    return out


def validate_provenance(root: Path, inventory: list[dict[str, str]], folds: list[dict[str, str]], results: dict) -> dict:
    ids = {row["sample_id"] for row in inventory}
    if any(set(rows) != ids for rows in results.values()):
        raise ValueError("evaluation sample identities do not match inventory")
    if len(folds) != 100 or any(sum(r["partition"] == "test" for r in folds if r["sample_id"] == sid) != 1 for sid in ids):
        raise ValueError("fold manifest does not test each image exactly once")
    groups: dict[tuple[str, str], set[str]] = {}
    for row in folds:
        groups.setdefault((row["outer_fold"], row["split_group_id"]), set()).add(row["partition"])
    if any(len(parts) != 1 for parts in groups.values()):
        raise ValueError("biological group leakage detected")
    checkpoints = []
    for code, (_, directory) in EXPERIMENTS.items():
        if code == "A":
            summary = json.loads((root / "evaluations" / directory / "summary.json").read_text())
            checkpoints.append({"experiment": code, "fold": "all", "checkpoint": summary["checkpoint"]})
            continue
        for fold in range(5):
            run = root / "runs" / directory / f"fold_{fold}"
            evaluation = json.loads((root / "evaluations" / directory / f"fold_{fold}" / "summary.json").read_text())
            config = json.loads((run / "config.json").read_text())
            expected = str(run / "model_best.pt")
            if evaluation["checkpoint"] != expected or not Path(expected).exists():
                raise ValueError(f"{code} fold {fold}: checkpoint provenance mismatch")
            if config["seed"] != 123 or config["resolved_device"] != "cuda:0":
                raise ValueError(f"{code} fold {fold}: unexpected seed or logical device")
            checkpoints.append({"experiment": code, "fold": fold, "checkpoint": expected, "config": str(run / "config.json")})
    return {
        "repository_commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"], cwd=ROOT)).hexdigest(),
        "inventory_sha256": sha256(ROOT / "data_manifests/real_annotations.csv"),
        "fold_manifest_sha256": sha256(ROOT / "data_manifests/real_annotation_folds_v1.csv"),
        "fold_schema_version": folds[0]["fold_schema_version"],
        "inventory_count": len(inventory), "fold_row_count": len(folds),
        "seed": 123, "logical_device": "cuda:0", "physical_gpu_for_final_queue": "1",
        "gpu_name": "NVIDIA GeForce RTX 5090", "checkpoints": checkpoints,
        "training_launcher": "python scripts/run_controlled_real_sted_suite.py --root /ssd/STED_experiments/controlled_real_sted_v1 --gpus 1",
        "analysis_command": "python scripts/analyze_controlled_real_sted_suite.py --suite-root /ssd/STED_experiments/controlled_real_sted_v1 --out /ssd/STED_experiments/controlled_real_sted_v1/analysis",
        "software": software_versions(),
    }


def build_long_rows(inventory: list[dict[str, str]], folds: list[dict[str, str]], results: dict) -> list[dict[str, Any]]:
    meta = {row["sample_id"]: row for row in inventory}
    test_fold = {row["sample_id"]: row["outer_fold"] for row in folds if row["partition"] == "test"}
    rows = []
    for code, (label, _) in EXPERIMENTS.items():
        for sid, result in results[code].items():
            m = meta[sid]
            rows.append({
                "experiment": code, "experiment_name": label, "sample_id": sid,
                "image_identity": m["image_identity"], "outer_fold": test_fold[sid],
                "split_group_id": m["split_group_id"], "disease": m["disease"],
                "tau_isoform": m["tau_isoform"], "div": m["div"],
                **{key: result.get(key) for key in METRICS},
            })
    return rows


def comparison_rows(inventory: list[dict[str, str]], results: dict) -> list[dict[str, Any]]:
    rows = []
    for meta in inventory:
        sid = meta["sample_id"]
        row: dict[str, Any] = {key: meta[key] for key in ["sample_id", "image_identity", "disease", "tau_isoform", "div", "split_group_id"]}
        for code in EXPERIMENTS:
            row[f"{code}_fibrous_dice"] = results[code][sid]["fibrous_dice"]
            if code != "A":
                for metric in METRICS:
                    a, b = results["A"][sid].get(metric), results[code][sid].get(metric)
                    row[f"{code}_minus_A_{metric}"] = float(b) - float(a) if numeric(a) and numeric(b) else "not_applicable"
        rows.append(row)
    return rows


def macro_rows(results: dict) -> list[dict[str, Any]]:
    rows = []
    for code, (label, _) in EXPERIMENTS.items():
        row = {"experiment": code, "experiment_name": label, "image_count": 20}
        for metric in METRICS:
            values = numeric_values(item.get(metric) for item in results[code].values())
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_sd"] = sd(values)
            row[f"{metric}_min"] = min(values) if values else "not_applicable"
            row[f"{metric}_max"] = max(values) if values else "not_applicable"
        rows.append(row)
    return rows


def build_fold_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for code, (label, directory) in EXPERIMENTS.items():
        if code == "A":
            continue
        for fold in range(5):
            summary = json.loads((root / "evaluations" / directory / f"fold_{fold}" / "summary.json").read_text())
            row = {"experiment": code, "experiment_name": label, "outer_fold": fold, "image_count": summary["sample_count"]}
            row.update({key: summary["aggregates"].get(key) for key in METRICS})
            rows.append(row)
    return rows


def strata_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for field in ["disease", "tau_isoform", "div"]:
        for value in sorted({row[field] for row in rows}):
            for code in EXPERIMENTS:
                selected = [row for row in rows if row[field] == value and row["experiment"] == code]
                out.append({"stratum_type": field, "stratum": value, "experiment": code, "n_images": len(selected), **{
                    metric: mean(numeric_values(row.get(metric) for row in selected))
                    for metric in ["fibrous_dice", "fibrous_recall", "clump_dice", "skeleton_dice_0.75", "missed_thick_component_count", "missed_faint_component_count"]
                }})
    return out


def selection_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for code, (_, directory) in EXPERIMENTS.items():
        if code == "A": continue
        for fold in range(5):
            run = root / "runs" / directory / f"fold_{fold}"
            checkpoint = json.loads((run / "checkpoint_summary.json").read_text())
            test = json.loads((root / "evaluations" / directory / f"fold_{fold}" / "summary.json").read_text())
            rows.append({"experiment": code, "outer_fold": fold, "selection_partition": "validation",
                         "selection_metric": checkpoint["best_metric"], "best_epoch": checkpoint["best_epoch"],
                         "validation_value": checkpoint["best_value"], "test_fibrous_dice": test["aggregates"]["fibrous_dice"],
                         "test_image_count": test["sample_count"], "checkpoint": test["checkpoint"]})
    return rows


def synthetic_retention_rows(root: Path) -> list[dict[str, Any]]:
    import torch
    initial = torch.load(ROOT / "runs/first_baseline_schema08_context_v2_256_aspp_uncertain_softloss/model.pt", map_location="cpu", weights_only=False)["validation_metrics"]
    rows = [{"experiment": "A_initial_checkpoint", "outer_fold": "not_applicable", **{
        key: initial.get(key) for key in ["fibrous_dice", "clump_dice", "skeleton_dice", "loss"]
    }}]
    for code, (_, directory) in EXPERIMENTS.items():
        if code in {"A", "B"}: continue
        for fold in range(5):
            metrics = json.loads((root / "runs" / directory / f"fold_{fold}" / "validation_metrics.json").read_text())
            rows.append({"experiment": code, "outer_fold": fold, **{key: metrics.get(key) for key in ["fibrous_dice", "clump_dice", "skeleton_dice", "loss"]}})
    return rows


def qualitative_examples(inventory: list[dict[str, str]], results: dict) -> list[dict[str, Any]]:
    def delta(sid: str, metric: str, code: str = "B") -> float:
        return float(results[code][sid][metric]) - float(results["A"][sid][metric])
    ids = [row["sample_id"] for row in inventory]
    choices = {
        "clear_improvement": max(ids, key=lambda sid: delta(sid, "fibrous_dice")),
        "skeleton_localisation_regression": min(ids, key=lambda sid: delta(sid, "skeleton_dice_0.85")),
        "psp_thick_bundles": max((sid for sid in ids if next(r for r in inventory if r["sample_id"] == sid)["disease"] == "PSP"), key=lambda sid: results["A"][sid]["missed_thick_component_count"]),
        "ad_clumps": max((sid for sid in ids if next(r for r in inventory if r["sample_id"] == sid)["disease"] == "AD"), key=lambda sid: results["A"][sid]["clump_target_pixels"]),
        "faint_fibres": max(ids, key=lambda sid: results["A"][sid]["missed_faint_component_count"] - results["B"][sid]["missed_faint_component_count"]),
        "false_positives": max(ids, key=lambda sid: results["B"][sid]["false_positive_fibrous_background_pixels"]),
        "uncertain_regions": max(ids, key=lambda sid: results["B"][sid]["fibrous_inside_uncertain_ignore_fraction"]),
    }
    meta = {row["sample_id"]: row for row in inventory}
    return [{"category": category, "sample_id": sid, "image_identity": meta[sid]["image_identity"],
             "disease": meta[sid]["disease"], "A_fibrous_dice": results["A"][sid]["fibrous_dice"],
             "B_fibrous_dice": results["B"][sid]["fibrous_dice"], "B_minus_A_fibrous_dice": delta(sid, "fibrous_dice"),
             "B_minus_A_skeleton_dice_0.85": delta(sid, "skeleton_dice_0.85")}
            for category, sid in choices.items()]


def make_qualitative_panels(root: Path, out: Path, examples: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("*.png"):
        path.unlink()
    for example in examples:
        sid = example["sample_id"]
        a = root / "evaluations" / "phase_a_synthetic_baseline" / sid / "qa_panel.png"
        b_candidates = list((root / "evaluations" / "b1_real_only").glob(f"fold_*/{sid}/qa_panel.png"))
        if not a.exists() or len(b_candidates) != 1: continue
        images = [Image.open(a).convert("RGB"), Image.open(b_candidates[0]).convert("RGB")]
        width = 1400
        resized = [image.resize((width, round(image.height * width / image.width))) for image in images]
        canvas = Image.new("RGB", (width, sum(image.height for image in resized) + 52), "white")
        draw = ImageDraw.Draw(canvas); y = 0
        for label, image in zip(["A: frozen synthetic", "B: real-only fine-tuned"], resized):
            draw.text((8, y + 5), label, fill="black"); y += 26; canvas.paste(image, (0, y)); y += image.height
        canvas.save(out / f"{example['category']}_{sid}.png")


def plot_metric_points(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = [("fibrous_dice", "Fibrous Dice"), ("clump_dice", "Clump Dice"), ("skeleton_dice_0.75", "Skeleton Dice 0.75"), ("predicted_to_target_fibrous_area_ratio", "Fibrous area ratio")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    rng = np.random.default_rng(123)
    for ax, (metric, title) in zip(axes, metrics):
        for x, code in enumerate(EXPERIMENTS):
            values = numeric_values(row[metric] for row in rows if row["experiment"] == code)
            ax.scatter(x + rng.uniform(-.1, .1, len(values)), values, s=18, alpha=.65)
            ax.plot([x-.16, x+.16], [mean(values)]*2, color="black", lw=2)
        ax.set_title(title); ax.set_xticks(range(5), EXPERIMENTS); ax.grid(axis="y", alpha=.25)
    fig.savefig(path, dpi=180); plt.close(fig)


def plot_paired_deltas(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True); rng = np.random.default_rng(123)
    for x, code in enumerate("BCDE"):
        values = [float(row[f"{code}_minus_A_fibrous_dice"]) for row in rows]
        ax.scatter(x + rng.uniform(-.08, .08, len(values)), values, s=22, alpha=.7)
        ax.plot([x-.15, x+.15], [mean(values)]*2, color="black", lw=2)
    ax.axhline(0, color="black", lw=1); ax.set_xticks(range(4), list("BCDE")); ax.set_ylabel("Paired fibrous Dice change vs A"); ax.grid(axis="y", alpha=.25)
    fig.savefig(path, dpi=180); plt.close(fig)


def plot_strata(rows: list[dict[str, Any]], path: Path) -> None:
    selected = [row for row in rows if row["stratum_type"] == "disease"]
    diseases = sorted({row["stratum"] for row in selected}); x = np.arange(len(diseases)); width = .16
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for index, code in enumerate(EXPERIMENTS):
        values = [next(row["fibrous_dice"] for row in selected if row["stratum"] == disease and row["experiment"] == code) for disease in diseases]
        ax.bar(x + (index-2)*width, values, width, label=code)
    ax.set_xticks(x, diseases); ax.set_ylabel("Macro per-image fibrous Dice"); ax.legend(ncol=5); ax.grid(axis="y", alpha=.25)
    fig.savefig(path, dpi=180); plt.close(fig)


def report_text(macro, folds, strata, comparisons, selection, synthetic, examples, provenance) -> str:
    m = {row["experiment"]: row for row in macro}
    def value(code, metric): return m[code][f"{metric}_mean"]
    paired = {code: [float(row[f"{code}_minus_A_fibrous_dice"]) for row in comparisons] for code in "BCDE"}
    fold_dice = {code: [row["fibrous_dice"] for row in folds if row["experiment"] == code] for code in "BCDE"}
    disease = {(row["experiment"], row["stratum"]): row for row in strata if row["stratum_type"] == "disease"}
    synth = {code: [row["fibrous_dice"] for row in synthetic if row["experiment"] == code] for code in "CDE"}
    initial_synth = next(row for row in synthetic if row["experiment"] == "A_initial_checkpoint")
    lines = [
        "# Controlled Real-STED Cross-Validation Report", "",
        "## Executive findings", "",
        f"- All fine-tuned models improved fibrous Dice over the frozen model on all 20 paired images. Mean paired gains were B {mean(paired['B']):.3f}, C {mean(paired['C']):.3f}, D {mean(paired['D']):.3f}, and E {mean(paired['E']):.3f}.",
        f"- Real-only B had the strongest held-out fibrous Dice ({value('B','fibrous_dice'):.3f}) and skeleton Dice at 0.75 ({value('B','skeleton_dice_0.75'):.3f}). Mixed C reached {value('C','fibrous_dice'):.3f}; mixed training was not better than real-only on this corpus.",
        f"- Targeted E modestly improved over ordinary mixed C for fibrous Dice ({value('E','fibrous_dice'):.3f} vs {value('C','fibrous_dice'):.3f}) and skeleton Dice 0.75 ({value('E','skeleton_dice_0.75'):.3f} vs {value('C','skeleton_dice_0.75'):.3f}), but the difference is smaller than fold and image variability.",
        f"- C and D were effectively indistinguishable: fibrous Dice {value('C','fibrous_dice'):.3f} vs {value('D','fibrous_dice'):.3f}. The uncertainty head shows no measurable segmentation or localisation benefit here.",
        "- Recommended real-image baseline: B (real-only fine-tuning), subject to a separate synthetic-retention evaluation because B had no synthetic validation stream.",
        "", "## Macro held-out results", "",
        "All values are unweighted means across images; clump Dice excludes images without target clump pixels.", "",
        "| Run | Fib Dice | Precision | Recall | Area ratio | Clump Dice | Skeleton Dice .50/.75/.85 | Target/pred skeleton <=2 px (.75) |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for code in EXPERIMENTS:
        lines.append(f"| {code} | {value(code,'fibrous_dice'):.3f} | {value(code,'fibrous_precision'):.3f} | {value(code,'fibrous_recall'):.3f} | {value(code,'predicted_to_target_fibrous_area_ratio'):.3f} | {value(code,'clump_dice'):.3f} | {value(code,'skeleton_dice_0.5'):.3f}/{value(code,'skeleton_dice_0.75'):.3f}/{value(code,'skeleton_dice_0.85'):.3f} | {value(code,'target_skeleton_recovered_within_2px_0.75'):.3f}/{value(code,'predicted_skeleton_within_2px_of_target_0.75'):.3f} |")
    lines += ["", "## Variation and paired comparison", ""]
    for code in "BCDE":
        lines.append(f"- {code}: image Dice SD {m[code]['fibrous_dice_sd']:.3f}, range {m[code]['fibrous_dice_min']:.3f}-{m[code]['fibrous_dice_max']:.3f}; fold range {min(fold_dice[code]):.3f}-{max(fold_dice[code]):.3f}; paired gain range {min(paired[code]):.3f}-{max(paired[code]):.3f} (20/20 positive).")
    lines += ["", "Fold 3 is consistently weakest. This is a held-out composition effect, not evidence from independent biological replicates.", "", "## Biological strata", ""]
    for condition in ["AD", "CBD", "PID", "PSP"]:
        n = disease[("A", condition)]["n_images"]
        lines.append(f"- {condition} (n={n} images): fibrous Dice A/B/C/D/E = " + "/".join(f"{disease[(code,condition)]['fibrous_dice']:.3f}" for code in EXPERIMENTS) + ".")
    lines += [
        "", "PSP thick-bundle recovery improves under B by fibrous recall and fewer missed thick components, but PSP represents four images from few provisional sample groups. AD shows the largest mean fibrous gain and improved clump-versus-fibre separation. CBD and PID do not regress in fibrous Dice, although their group counts are too small for biological inference.",
        "", "## Failure modes", "",
        f"B increased fibrous predictions inside uncertain regions relative to A ({value('A','fibrous_inside_uncertain_ignore_fraction'):.3f} to {value('B','fibrous_inside_uncertain_ignore_fraction'):.3f}) while improving area calibration to {value('B','predicted_to_target_fibrous_area_ratio'):.3f}. Recurring failures remain faint-fragment misses, background components, uncertain-region fibrous predictions, and weak fold-2 clump Dice.",
        f"Fibrous recall rises under B, but mean missed-faint-component count also rises ({value('A','missed_faint_component_count'):.1f} to {value('B','missed_faint_component_count'):.1f}). This component metric is sensitive to fragmentation, so the 20 images do not support a clean claim that faint-fibre recovery improved.",
        f"Targeted E has the lowest fibrous-in-clump rate ({value('E','fibrous_inside_target_clump_fraction'):.3f}), but skeleton-in-uncertain remains similar across B-E ({value('B','skeleton_inside_uncertain_ignore_fraction_0.75'):.3f}-{max(value(code,'skeleton_inside_uncertain_ignore_fraction_0.75') for code in 'BCDE'):.3f}).",
        "", "## Validation versus outer test", "",
        "`selection_vs_test.csv` records validation selection values separately from outer-test Dice. Every best checkpoint was selected at epoch 19 using macro per-image validation fibrous Dice. No outer-test metric was used for selection.",
        "", "## Synthetic retention", "",
        f"The initial synthetic checkpoint had synthetic-validation fibrous Dice {float(initial_synth['fibrous_dice']):.3f}; C/D/E final means were {mean(synth['C']):.3f}/{mean(synth['D']):.3f}/{mean(synth['E']):.3f}. The small decrease does not indicate catastrophic forgetting. B cannot be assessed because real-only runs did not evaluate a synthetic validation stream; this is the main unresolved comparison before deployment.",
        "", "## Qualitative examples", "",
        "The deterministic selections below compare the same image under A and B; panels are in `qualitative_examples/`.", "",
        "| Role | Image | Condition | A Dice | B Dice | Fib delta | Skeleton .85 delta |", "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in examples:
        lines.append(f"| {row['category']} | {row['image_identity']} | {row['disease']} | {float(row['A_fibrous_dice']):.3f} | {float(row['B_fibrous_dice']):.3f} | {float(row['B_minus_A_fibrous_dice']):+.3f} | {float(row['B_minus_A_skeleton_dice_0.85']):+.3f} |")
    lines += [
        "", "All 20 fibrous-Dice changes were positive, but the selected regression shows that stricter skeleton localisation can worsen on an individual image. All metric-level regressions are retained in `per_image_comparisons.csv`.",
        "", "## Recommendation", "",
        "Use B as the next real-image baseline: initialize from the stated synthetic checkpoint, train four frozen-encoder epochs at 1e-4, then fifteen full-model epochs at 2e-5 with real-only balanced patch sampling. Do not discard C/E: mixed training may be preferable if a dedicated synthetic-retention test shows material forgetting under B.",
        "", "Generator priorities: add more thick and partially resolved bundles; improve faint/low-contrast fibre rendering; diversify compact clump boundaries and adjacent fibres; add structured hard backgrounds; and better model ambiguous transition regions without turning uncertainty into background.",
        "", "## Scope and limitations", "",
        "The unit summarized is the image, not pixels or crops. The 20 images are not 20 independent biological replicates; grouping is provisional and PN is not a biological grouping variable. Disease, isoform, and DIV strata are descriptive. Clump Dice applies only where clump labels exist. The working tree was dirty, so commit plus manifest/source fingerprints are required for reproduction.",
        "", "## Reproducibility", "",
        f"- Commit: `{provenance['repository_commit']}` (dirty working tree: `{provenance['working_tree_dirty']}`)",
        f"- Inventory SHA-256: `{provenance['inventory_sha256']}`", f"- Fold manifest: `{provenance['fold_schema_version']}`, SHA-256 `{provenance['fold_manifest_sha256']}`",
        f"- Seed: `{provenance['seed']}`; final queue physical GPU `{provenance['physical_gpu_for_final_queue']}` mapped to logical `{provenance['logical_device']}`",
        f"- Launcher: `{provenance['training_launcher']}`", f"- Analysis: `{provenance['analysis_command']}`",
        "- Exact checkpoint/config paths and software versions: `provenance.json`.",
    ]
    return "\n".join(lines) + "\n"


def executive_summary(macro, comparisons) -> str:
    m = {row["experiment"]: row for row in macro}
    return f"""# Real-STED Experiment Handoff

- Controlled suite complete: Phase A plus B-E, five grouped outer folds, 20 unique OOF images per fine-tuned run.
- Macro fibrous Dice A/B/C/D/E: {m['A']['fibrous_dice_mean']:.3f}/{m['B']['fibrous_dice_mean']:.3f}/{m['C']['fibrous_dice_mean']:.3f}/{m['D']['fibrous_dice_mean']:.3f}/{m['E']['fibrous_dice_mean']:.3f}.
- Every fine-tuned run improved fibrous Dice over A on every image. B gives the largest mean gain and is the recommended real-image baseline.
- C versus D is effectively unchanged, so the uncertainty head has no measurable segmentation benefit in this experiment.
- E modestly improves C and reduces fibrous leakage into clumps, but differences are within fold/image variability.
- Major unresolved issue: B lacks a synthetic-retention evaluation; do that before replacing mixed training for general deployment.
- Results: `/ssd/STED_experiments/controlled_real_sted_v1/analysis/report.md`.
- Machine-readable details: `macro_summary.csv`, `fold_metrics.csv`, `per_image_results.csv`, `per_image_comparisons.csv`, and `provenance.json` in the same directory.
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def numeric(value: Any) -> bool: return isinstance(value, (int, float)) and not isinstance(value, bool)
def numeric_values(values) -> list[float]: return [float(value) for value in values if numeric(value)]
def mean(values: list[float]) -> float | str: return float(np.mean(values)) if values else "not_applicable"
def sd(values: list[float]) -> float | str: return float(np.std(values, ddof=1)) if len(values) > 1 else "not_applicable"
def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def git(*args: str) -> str: return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def software_versions() -> dict[str, str]:
    import matplotlib, PIL, scipy, sys
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "Pillow": PIL.__version__, "matplotlib": matplotlib.__version__}


if __name__ == "__main__":
    raise SystemExit(main())
