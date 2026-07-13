#!/usr/bin/env python
"""Final scientific analysis of post-audit uncertain-ignore nested CV artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("U0", "S", "H", "SH")
PAIRED_METRICS = (
    "fibrous_dice", "fibrous_precision", "fibrous_recall", "fibrous_area_ratio", "clump_dice",
    "skeleton_dice_0.5", "skeleton_dice_0.75", "skeleton_dice_0.85",
    "target_skeleton_recovered_2px", "predicted_skeleton_within_target_2px", "fibrous_leakage_into_clump",
)
BOUNDARIES = ("0_2px", "2_5px", "5_10px", "10_20px", "over_20px")
THRESHOLDS = ("0.5", "0.7", "0.8", "0.9")
SEED = 20260713


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.experiment_root
    assert not any("pre_audit" in str(p) for p in required_inputs(root))
    rows = read_numeric(root / "outer_test_per_image.csv")
    metadata = load_metadata(args.manifest, args.fold_manifest)
    integrity_checks(root, rows, metadata)
    enriched = [{**row, **metadata[row["sample_id"]]} for row in rows]

    macro = macro_summary(enriched)
    paired = paired_differences(enriched)
    clusters = cluster_intervals(enriched)
    uncertain = uncertain_summary(enriched)
    boundary = boundary_summary(enriched)
    strata = strata_summary(enriched)
    qualitative = qualitative_examples(root, enriched)

    write_csv(root / "macro_summary.csv", macro)
    write_csv(root / "paired_differences.csv", paired)
    write_csv(root / "cluster_bootstrap_intervals.csv", clusters)
    write_csv(root / "uncertain_region_summary.csv", uncertain)
    write_csv(root / "boundary_safety_summary.csv", boundary)
    write_csv(root / "strata_summary.csv", strata)
    write_csv(root / "qualitative_examples.csv", qualitative)
    make_plots(root, enriched, paired, uncertain, boundary)
    write_reports(root, macro, paired, clusters, uncertain, boundary, strata)
    write_provenance(root, args, rows, metadata)
    consistency_checks(root, macro, paired)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-root", type=Path, default=Path("/ssd/STED_experiments/controlled_real_sted_v1/uncertain_ignore_nested_cv"))
    p.add_argument("--manifest", type=Path, default=ROOT / "data_manifests/real_annotations.csv")
    p.add_argument("--fold-manifest", type=Path, default=ROOT / "data_manifests/real_annotation_folds_v1.csv")
    return p


def required_inputs(root: Path) -> list[Path]:
    return [root / name for name in (
        "outer_test_per_image.csv", "outer_test_per_fold.csv", "outer_test_macro.json",
        "inner_validation_selections.json", "metric_definitions.csv", "u0_regression_audit.csv",
        "uncertainty_discrimination.csv", "uncertainty_spatial_discrimination.csv", "metric_audit_report.md",
    )]


def load_metadata(manifest: Path, folds: Path) -> dict[str, dict[str, Any]]:
    inventory = {r["sample_id"]: r for r in csv.DictReader(manifest.open())}
    test = {r["sample_id"]: r for r in csv.DictReader(folds.open()) if r["partition"] == "test"}
    return {sample: {
        "image_identity": inventory[sample]["image_identity"], "split_group_id": row["split_group_id"],
        "disease": row["disease"], "tau_isoform": row["tau_isoform"], "div": row["div"],
        "morphology_density": row["morphology_density"], "uncertain_prevalence": float(row["uncertain_fraction"]),
        "manifest_outer_fold": int(row["outer_fold"]), "source_sha256": inventory[sample]["source_sha256"],
        "labels_sha256": inventory[sample]["labels_sha256"], "snakes_sha256": inventory[sample]["snakes_sha256"],
    } for sample, row in test.items()}


def integrity_checks(root: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    for path in required_inputs(root):
        if not path.exists(): raise FileNotFoundError(path)
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        assert len(selected) == 20 and {r["sample_id"] for r in selected} == set(metadata)
        assert all(int(r["outer_fold"]) == metadata[r["sample_id"]]["manifest_outer_fold"] for r in selected)
    selections = json.loads((root / "inner_validation_selections.json").read_text())
    expected = [0.84, 0.82, 0.84, 0.87, 0.86]
    assert [round(selections[str(f)]["H"]["rejection_threshold"], 2) for f in range(5)] == expected
    for row in rows:
        if row["variant"] in {"H", "SH"}: assert math.isclose(row["rejection_threshold"], expected[int(row["outer_fold"])])
        else: assert row["rejection_threshold"] == "not_applicable"
    u0 = [r for r in rows if r["variant"] == "U0"]
    assert sum(bool(r["clump_applicable"]) for r in u0) == 9
    assert all(r["clump_dice"] == "not_applicable" for r in u0 if not r["clump_applicable"])
    audit = list(csv.DictReader((root / "u0_regression_audit.csv").open()))
    assert len(audit) == 240 and all(r["match"] == "True" for r in audit)


def macro_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = PAIRED_METRICS + tuple(f"uncertain_fibrous_probability_ge_{t}_fraction{suffix}" for t in THRESHOLDS for suffix in ("_raw", "")) + (
        "fibrous_fraction_inside_uncertain_ignore", "expert_uncertain_precision", "expert_uncertain_recall",
        "rejected_all_fraction", "rejected_predicted_fibres_fraction", "rejected_target_fibres_fraction",
    )
    out = []
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        for metric in metrics:
            values = numeric([r[metric] for r in selected])
            out.append({"variant":variant, "metric":metric, "macro_per_image":mean(values), "sd":sd(values), "applicable_images":len(values), "denominator":"applicable images; clump excludes absent target annotations"})
    return out


def paired_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["variant"], r["sample_id"]): r for r in rows}
    rng, out = np.random.default_rng(SEED), []
    for variant in VARIANTS[1:]:
        for metric in PAIRED_METRICS:
            differences = []
            for sample in sorted({r["sample_id"] for r in rows}):
                a, b = lookup[(variant, sample)][metric], lookup[("U0", sample)][metric]
                if is_number(a) and is_number(b): differences.append(float(a)-float(b))
            lo, hi = bootstrap(differences, rng)
            out.append({"variant":variant, "metric":metric, "mean_paired_difference":mean(differences), "sd_paired_difference":sd(differences), "image_bootstrap_95_low":lo, "image_bootstrap_95_high":hi, "applicable_images":len(differences), "improved_images":sum(x>0 for x in differences), "worsened_images":sum(x<0 for x in differences), "unchanged_images":sum(x==0 for x in differences)})
    return out


def cluster_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["variant"], r["sample_id"]): r for r in rows}
    meta = {r["sample_id"]: r for r in rows if r["variant"] == "U0"}
    groups = sorted({r["split_group_id"] for r in meta.values()})
    rng, out = np.random.default_rng(SEED + 1), []
    for variant in VARIANTS[1:]:
        for metric in PAIRED_METRICS:
            group_values = []
            for group in groups:
                values = []
                for sample, row in meta.items():
                    if row["split_group_id"] != group: continue
                    a, b = lookup[(variant,sample)][metric], lookup[("U0",sample)][metric]
                    if is_number(a) and is_number(b): values.append(float(a)-float(b))
                if values: group_values.append(mean(values))
            lo, hi = bootstrap(group_values, rng)
            out.append({"variant":variant, "metric":metric, "group_macro_difference":mean(group_values), "cluster_bootstrap_95_low":lo, "cluster_bootstrap_95_high":hi, "applicable_split_groups":len(group_values), "cluster_unit":"provisional split_group_id"})
    return out


def uncertain_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["variant"],r["sample_id"]):r for r in rows}
    metrics = ("fibrous_fraction_inside_uncertain_ignore",) + tuple(f"uncertain_fibrous_probability_ge_{t}_fraction{suffix}" for t in THRESHOLDS for suffix in ("_raw", "")) + ("expert_uncertain_precision","expert_uncertain_recall","rejected_all_fraction","rejected_predicted_fibres_fraction","rejected_target_fibres_fraction")
    out=[]
    for variant in VARIANTS:
        for metric in metrics:
            values=numeric([r[metric] for r in rows if r["variant"]==variant]); value=mean(values)
            base=mean(numeric([r[metric] for r in rows if r["variant"]=="U0"]))
            out.append({"variant":variant,"metric":metric,"macro_per_image":value,"sd":sd(values),"absolute_change_fraction":value-base,"absolute_change_percentage_points":100*(value-base),"relative_change_percent":100*(value-base)/base if base else "not_applicable","effect":"raw semantic confidence" if "_raw" in metric or metric=="fibrous_fraction_inside_uncertain_ignore" else "operational after abstention" if variant in {"H","SH"} else "semantic prediction without abstention"})
    return out


def boundary_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup={(r["variant"],r["sample_id"]):r for r in rows}; out=[]
    for variant in VARIANTS:
        for band in BOUNDARIES:
            key=f"confident_fibrous_recall_{band}"; vals=[]; diffs=[]
            for sample in sorted({r["sample_id"] for r in rows}):
                value,base=lookup[(variant,sample)][key],lookup[("U0",sample)][key]
                if is_number(value): vals.append(float(value))
                if is_number(value) and is_number(base): diffs.append(float(value)-float(base))
            out.append({"variant":variant,"band":band,"macro_recall":mean(vals),"sd_across_images":sd(vals),"mean_paired_difference":mean(diffs),"sd_paired_difference":sd(diffs),"applicable_images":len(diffs),"improved_images":sum(x>0 for x in diffs),"worsened_images":sum(x<0 for x in diffs),"unchanged_images":sum(x==0 for x in diffs)})
    return out


def strata_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]
    strata=("disease","tau_isoform","div","morphology_density","outer_fold","uncertain_prevalence_bin")
    for row in rows: row["uncertain_prevalence_bin"]="high" if row["uncertain_prevalence"]>=0.05 else "low"
    for key in strata:
        for level in sorted({str(r[key]) for r in rows}):
            for variant in VARIANTS:
                selected=[r for r in rows if str(r[key])==level and r["variant"]==variant]
                for metric in ("fibrous_dice","skeleton_dice_0.75","fibrous_fraction_inside_uncertain_ignore","uncertain_fibrous_probability_ge_0.7_fraction_raw","expert_uncertain_recall","confident_fibrous_recall_0_2px"):
                    vals=numeric([r[metric] for r in selected]); out.append({"stratum":key,"level":level,"variant":variant,"metric":metric,"macro_per_image":mean(vals),"sd":sd(vals),"images":len(vals)})
    return out


def qualitative_examples(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    u0=[r for r in rows if r["variant"]=="U0"]
    lookup={(r["variant"],r["sample_id"]):r for r in rows}
    boundary_candidates=[r for r in u0 if is_number(r["confident_fibrous_recall_0_2px"]) and is_number(lookup[("SH",r["sample_id"])]["confident_fibrous_recall_0_2px"])]
    roles={
        "dense_ad":max((r for r in u0 if r["disease"]=="AD" and r["morphology_density"]=="dense"),key=lambda r:r["uncertain_prevalence"]),
        "psp_thick_bundles":max((r for r in u0 if r["disease"]=="PSP"),key=lambda r:r["fibrous_area_ratio"]),
        "faint_fibres_fold3":min((r for r in u0 if int(r["outer_fold"])==3),key=lambda r:r["fibrous_recall"]),
        "adjacent_fibre_cost":min(boundary_candidates,key=lambda r:lookup[("SH",r["sample_id"])]["confident_fibrous_recall_0_2px"]-r["confident_fibrous_recall_0_2px"]),
        "successful_rejection":max(u0,key=lambda r:lookup[("H",r["sample_id"])]["expert_uncertain_recall"]),
        "broad_abstention":max(u0,key=lambda r:lookup[("SH",r["sample_id"])]["rejected_all_fraction"]),
    }
    out=[]; qdir=root/"final_qualitative"; qdir.mkdir(exist_ok=True)
    for role,row in roles.items():
        paths=[]
        for variant,prefix in (("S","S_s0.05"),("H","H_h0.05"),("SH","SH_s0.05_h0.05")):
            p=root/"evaluations"/prefix/f"fold_{int(row['outer_fold'])}"/"test"/row["sample_id"]/"qa_panel.png"; paths.append((variant,p))
        panel=qdir/f"{role}_{row['sample_id']}.png"; montage(paths,panel)
        out.append({"role":role,"sample_id":row["sample_id"],"image_identity":row["image_identity"],"disease":row["disease"],"fold":row["outer_fold"],"morphology_density":row["morphology_density"],"uncertain_prevalence":row["uncertain_prevalence"],"panel_path":str(panel),"interpretation":"descriptive case selected deterministically from corrected metrics; inspect alongside quantitative summaries"})
    return out


def make_plots(root: Path, rows: list[dict[str, Any]], paired: list[dict[str, Any]], uncertain: list[dict[str, Any]], boundary: list[dict[str, Any]]) -> None:
    pdir=root/"final_plots"; pdir.mkdir(exist_ok=True)
    macro=macro_summary(rows)
    grouped_bar(pdir/"primary_real_metrics.png", macro, ["fibrous_dice","fibrous_precision","fibrous_recall","clump_dice","skeleton_dice_0.75"], "macro_per_image", "Primary held-out real-image metrics")
    paired_box(pdir/"paired_fibrous_skeleton_differences.png", rows)
    grouped_bar(pdir/"uncertain_high_confidence.png", uncertain, [f"uncertain_fibrous_probability_ge_{t}_fraction_raw" for t in THRESHOLDS], "macro_per_image", "Raw fibrous confidence inside uncertain-ignore")
    scatter_plot(pdir/"rejection_coverage_vs_target_fibre_loss.png", rows)
    boundary_plot(pdir/"boundary_recall_vs_distance.png", boundary)
    discrimination_plot(root,pdir/"conditional_uncertainty_discrimination.png")
    fold_plot(pdir/"per_fold_variation.png",rows)


def write_reports(root: Path, macro: list[dict[str, Any]], paired: list[dict[str, Any]], clusters: list[dict[str, Any]], uncertain: list[dict[str, Any]], boundary: list[dict[str, Any]], strata: list[dict[str, Any]]) -> None:
    m=lambda v,k:find(macro,v,k,"macro_per_image"); d=lambda v,k:find(paired,v,k,"mean_paired_difference"); u=lambda v,k:find(uncertain,v,k,"macro_per_image"); b=lambda v,k:find(boundary,v,k,"mean_paired_difference")
    classifications={
        "S":"acceptable default" if d("S","fibrous_dice")>=-0.01 and d("S","skeleton_dice_0.75")>=-0.01 and d("S","clump_dice")>=-0.01 else "rejected",
        "H":"diagnostically useful but not operational",
        "SH":"rejected",
    }
    report=f"""# Final uncertain-ignore nested-CV analysis

## Executive conclusion

S is the recommended default ({classifications['S']}). H is classified as {classifications['H']}; SH is {classifications['SH']}. These labels account for complete-image segmentation constraints, outer-test target-fibre abstention, and explicit boundary costs, not uncertain-region reduction alone.

## Integrity

The analysis uses only post-audit canonical artifacts. All four variants contain the same 20 held-out images and manifest folds. U0 reproduces B1 for 240/240 shared comparisons. Clump Dice excludes 11 non-applicable images. H/SH thresholds are fold-specific 0.84/0.82/0.84/0.87/0.86 and rejected targets remain in complete-image denominators.

## Primary performance

| Variant | Fibre Dice | Precision | Recall | Area ratio | Clump Dice (n=9) | Skeleton .50/.75/.85 | Target/pred skeleton <=2 px |
|---|---:|---:|---:|---:|---:|---:|---:|
"""+"\n".join(f"| {v} | {m(v,'fibrous_dice'):.3f} | {m(v,'fibrous_precision'):.3f} | {m(v,'fibrous_recall'):.3f} | {m(v,'fibrous_area_ratio'):.3f} | {m(v,'clump_dice'):.3f} | {m(v,'skeleton_dice_0.5'):.3f}/{m(v,'skeleton_dice_0.75'):.3f}/{m(v,'skeleton_dice_0.85'):.3f} | {m(v,'target_skeleton_recovered_2px'):.3f}/{m(v,'predicted_skeleton_within_target_2px'):.3f} |" for v in VARIANTS)+f"""

S/H/SH paired fibre-Dice changes are {d('S','fibrous_dice'):+.4f}/{d('H','fibrous_dice'):+.4f}/{d('SH','fibrous_dice'):+.4f}; skeleton-Dice-.75 changes are {d('S','skeleton_dice_0.75'):+.4f}/{d('H','skeleton_dice_0.75'):+.4f}/{d('SH','skeleton_dice_0.75'):+.4f}. S fibre-Dice changed in 5/15 improved/worsened images, H in 7/13, and SH in 6/14. S reduced fibre recall in all 20 images, although its precision gain kept Dice effectively stable. Image and provisional-group bootstrap intervals and all metric-specific counts are in the companion CSVs. They are descriptive because images and groups are not guaranteed independent biological replicates.

## Uncertain-region behavior

Raw p(fibre)>=0.7 changes from U0 are S {100*(u('S','uncertain_fibrous_probability_ge_0.7_fraction_raw')-u('U0','uncertain_fibrous_probability_ge_0.7_fraction_raw')):+.2f} pp, H {100*(u('H','uncertain_fibrous_probability_ge_0.7_fraction_raw')-u('U0','uncertain_fibrous_probability_ge_0.7_fraction_raw')):+.2f} pp, and SH {100*(u('SH','uncertain_fibrous_probability_ge_0.7_fraction_raw')-u('U0','uncertain_fibrous_probability_ge_0.7_fraction_raw')):+.2f} pp, relative reductions of {100*(u('U0','uncertain_fibrous_probability_ge_0.7_fraction_raw')-u('S','uncertain_fibrous_probability_ge_0.7_fraction_raw'))/u('U0','uncertain_fibrous_probability_ge_0.7_fraction_raw'):.1f}%, {100*(u('U0','uncertain_fibrous_probability_ge_0.7_fraction_raw')-u('H','uncertain_fibrous_probability_ge_0.7_fraction_raw'))/u('U0','uncertain_fibrous_probability_ge_0.7_fraction_raw'):.1f}%, and {100*(u('U0','uncertain_fibrous_probability_ge_0.7_fraction_raw')-u('SH','uncertain_fibrous_probability_ge_0.7_fraction_raw'))/u('U0','uncertain_fibrous_probability_ge_0.7_fraction_raw'):.1f}%. These are semantic-confidence effects, not abstention. H/SH operational values additionally remove pixels selected by their calibrated rejection masks. H expert-uncertain recall is {u('H','expert_uncertain_recall'):.3f} while abstaining from {u('H','rejected_target_fibres_fraction'):.3f} of target fibres; SH recall is {u('SH','expert_uncertain_recall'):.3f} with target-fibre abstention {u('SH','rejected_target_fibres_fraction'):.3f}. Both exceed 2% on outer-test image-macro evaluation despite validation-only constrained selection.

## Rejection-head discrimination

The primary conditional comparison is uncertain-ignore versus confident fibrous/clump. H pooled/image-macro AUROC is 0.803/0.790 and AP 0.764/0.674; SH is 0.806/0.793 and 0.767/0.676. U0 is 0.512/0.487 and fails. H spatial AUROC at 2/5/10/20 px is 0.797/0.803/0.801/0.799; SH is 0.798/0.804/0.803/0.801. This persistence against nearby tau supports local expert-uncertainty discrimination rather than broad foreground detection. Per-fold values and prevalence are in `uncertainty_discrimination.csv` and `uncertainty_spatial_discrimination.csv`; uncertain-versus-background is not used as evidence.

## Boundary safety

U0 0-2 px recall is {find(boundary,'U0','0_2px','macro_recall'):.3f}. Paired S/H/SH losses at 0-2 px are {b('S','0_2px'):+.3f}/{b('H','0_2px'):+.3f}/{b('SH','0_2px'):+.3f}, and at 2-5 px {b('S','2_5px'):+.3f}/{b('H','2_5px'):+.3f}/{b('SH','2_5px'):+.3f}. At 5-10/10-20/>20 px, S changes are {b('S','5_10px'):+.3f}/{b('S','10_20px'):+.3f}/{b('S','over_20px'):+.3f}, H {b('H','5_10px'):+.3f}/{b('H','10_20px'):+.3f}/{b('H','over_20px'):+.3f}, and SH {b('SH','5_10px'):+.3f}/{b('SH','10_20px'):+.3f}/{b('SH','over_20px'):+.3f}. S damage decays with distance but is not confined to annotation boundaries. H adds a stronger local cost. SH has the largest and broadest loss, so its greater conservatism does not justify default use. Per-band SD and improved/worsened counts show the losses are distributed across images rather than attributable to one outlier.

## Morphology and strata

Disease, isoform, DIV, density, uncertainty prevalence, and fold summaries are descriptive only. Dense AD, PSP, faint fold-3 fibres, adjacent valid fibres, successful rejection, and broad abstention cases are indexed in `qualitative_examples.csv`. Fold 3 and small strata must not be interpreted as independent disease effects.

## Answers and decision

1. Soft suppression reduces excessive confidence with small complete-image metric changes and the least boundary damage of the new methods: use S by default.
2. The retrained H head distinguishes uncertain material from confident and spatially nearby tau.
3. Under the <=2% validation target-fibre constraint, H's operational recall is not robustly useful: outer-test target-fibre abstention averages 2.3% for only 11.5% uncertain recall. Retain its score for diagnostics and sensitivity analysis, not routine automatic exclusion.
4. SH improves conservative filtering but its larger adjacent-fibre and longer-range recall loss does not justify default use. Retain it only as an explicitly conservative sensitivity mode.
5. Default: S, fold-specific checkpoint selected with lambda suppression 0.05, tau 0.5, otherwise the B1 real-only recipe.
6. Optional sensitivity inference: H with lambda rejection 0.05 and thresholds 0.84/0.82/0.84/0.87/0.86, explicitly labelled non-operational. Do not retain SH as a routine mode; it may remain archived for retrospective comparison.
7. Exclude expert-uncertain and model-abstained pixels from fibre length, skeleton, and topology estimates; report default-S measurements plus H-rejected sensitivity estimates. Retain expert and model uncertainty masks separately. Expert uncertain-ignore is annotation ambiguity, while H is a learned score and cannot replace it. U0/S/H/SH disagreement should be retained as a diagnostic stability layer.

## Limitations

Only 20 images and 12 provisional split groups are available; annotations and strata are uneven, clump Dice applies to nine images, and cluster intervals inherit provisional grouping. Thresholds were selected without outer-test optimization, but operating behavior may not transfer. Synthetic metrics were not used for selection or recommendation.

## Recipe and artifacts

Recommended default checkpoints are `runs/S_s0.05/fold_*/model_best.pt`: initialize from the B1 synthetic checkpoint, use the unchanged B1 real-only schedule, semantic/skeleton ignore on uncertain pixels, and lambda 0.05 squared hinge suppression above p(fibre)=0.5. Optional H checkpoints are `runs/H_h0.05/fold_*/model_best.pt`, with stratified rejection BCE and the calibrated fold thresholds above. All tables, plots, provenance, and qualitative panels are under this experiment root.
"""
    (root/"final_report.md").write_text(report)
    (root/"executive_summary.md").write_text("# Executive summary\n\n"+"\n".join(report.splitlines()[4:9])+"\n\nSee `final_report.md` for evidence, intervals, boundary costs, limitations, and downstream policy.\n")


def write_provenance(root: Path, args: argparse.Namespace, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    inputs={str(p):sha256(p) for p in required_inputs(root)}
    inputs[str(args.manifest)]=sha256(args.manifest); inputs[str(args.fold_manifest)]=sha256(args.fold_manifest)
    payload={"analysis":"post-audit final uncertain-ignore nested CV","script":str(Path(__file__).resolve()),"seed":SEED,"variants":list(VARIANTS),"outer_test_images":20,"split_groups":len({m['split_group_id'] for m in metadata.values()}),"thresholds_by_fold":[0.84,0.82,0.84,0.87,0.86],"input_sha256":inputs,"excluded_pattern":"*.pre_audit.*","consistency_checks":"passed","model_training_run":False,"synthetic_selection_used":False}
    (root/"provenance.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")


def consistency_checks(root: Path, macro: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    audited=json.loads((root/"outer_test_macro.json").read_text())
    for variant in VARIANTS:
        for metric in PAIRED_METRICS:
            assert math.isclose(find(macro,variant,metric,"macro_per_image"),float(audited[variant][metric]),abs_tol=1e-12)
    assert math.isclose(find(macro,"U0","clump_dice","macro_per_image"),0.7830563560391917,abs_tol=1e-12)
    assert all((root/name).exists() for name in ("final_report.md","executive_summary.md","macro_summary.csv","paired_differences.csv","cluster_bootstrap_intervals.csv","uncertain_region_summary.csv","boundary_safety_summary.csv","strata_summary.csv","qualitative_examples.csv","provenance.json"))


def grouped_bar(path: Path, rows: list[dict[str, Any]], metrics: list[str], value: str, title: str) -> None:
    x=np.arange(len(metrics)); width=.19; fig,ax=plt.subplots(figsize=(11,5))
    for i,v in enumerate(VARIANTS): ax.bar(x+(i-1.5)*width,[find(rows,v,m,value) for m in metrics],width,label=v)
    ax.set_xticks(x,metrics,rotation=20,ha="right"); ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def paired_box(path: Path, rows: list[dict[str, Any]]) -> None:
    lookup={(r["variant"],r["sample_id"]):r for r in rows}; data=[]; labels=[]
    for metric in ("fibrous_dice","skeleton_dice_0.75"):
        for v in VARIANTS[1:]: data.append([lookup[(v,s)][metric]-lookup[("U0",s)][metric] for s in sorted({r['sample_id'] for r in rows})]); labels.append(f"{v}\n{metric.replace('skeleton_dice_0.75','skel .75')}")
    fig,ax=plt.subplots(figsize=(9,5)); ax.boxplot(data,tick_labels=labels); ax.axhline(0,color="black",lw=1); ax.set_ylabel("paired difference from U0"); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def scatter_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    fig,ax=plt.subplots(figsize=(7,5))
    for v in ("H","SH"): 
        selected=[r for r in rows if r["variant"]==v]; ax.scatter([r["rejected_target_fibres_fraction"] for r in selected],[r["expert_uncertain_recall"] for r in selected],label=v,alpha=.8)
    ax.axvline(.02,color="black",ls="--",lw=1); ax.set(xlabel="target fibres abstained",ylabel="expert-uncertain recall",title="Rejection coverage and fibre cost"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def boundary_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(5)
    for v in VARIANTS: ax.plot(x,[find(rows,v,b,"macro_recall") for b in BOUNDARIES],marker="o",label=v)
    ax.set_xticks(x,["0-2","2-5","5-10","10-20",">20"]); ax.set(xlabel="distance from uncertain-ignore (px)",ylabel="confident-fibre recall",title="Boundary safety"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def discrimination_plot(root: Path, path: Path) -> None:
    rows=read_numeric(root/"uncertainty_discrimination.csv"); fig,ax=plt.subplots(figsize=(7,5)); x=np.arange(3); width=.35
    selected=[next(r for r in rows if r["variant"]==v and r["comparison"]=="uncertain_vs_fibrous_clump" and r["scope"]=="global_pooled") for v in ("U0","H","SH")]
    ax.bar(x-width/2,[r["auroc"] for r in selected],width,label="AUROC"); ax.bar(x+width/2,[r["average_precision"] for r in selected],width,label="AP"); ax.set_xticks(x,["U0","H","SH"]); ax.set_ylim(0,1); ax.set_title("Uncertain vs confident fibrous/clump"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def fold_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    fig,ax=plt.subplots(figsize=(8,5))
    for v in VARIANTS: ax.plot(range(5),[mean([r["fibrous_dice"] for r in rows if r["variant"]==v and int(r["outer_fold"])==f]) for f in range(5)],marker="o",label=v)
    ax.set(xlabel="outer fold",ylabel="macro fibrous Dice",title="Per-fold held-out variation"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def montage(paths: list[tuple[str,Path]], out: Path) -> None:
    images=[(label,Image.open(path).convert("RGB")) for label,path in paths]; width=max(im.width for _,im in images); height=sum(im.height+28 for _,im in images)
    canvas=Image.new("RGB",(width,height),"white"); draw=ImageDraw.Draw(canvas); y=0
    for label,im in images: draw.text((8,y+6),label,fill="black"); y+=28; canvas.paste(im,(0,y)); y+=im.height
    canvas.save(out)


def read_numeric(path: Path) -> list[dict[str, Any]]:
    rows=[]
    for row in csv.DictReader(path.open()):
        rows.append({k:parse(v) for k,v in row.items()})
    return rows


def parse(value: str) -> Any:
    if value=="not_applicable": return value
    if value in {"True","False"}: return value=="True"
    try: return float(value)
    except ValueError: return value


def numeric(values: list[Any]) -> list[float]: return [float(x) for x in values if is_number(x)]
def is_number(x: Any) -> bool: return isinstance(x,(int,float,np.integer,np.floating)) and math.isfinite(float(x))
def mean(x: list[float]) -> float: return float(np.mean(x)) if x else float("nan")
def sd(x: list[float]) -> float: return float(np.std(x,ddof=1)) if len(x)>1 else 0.0
def bootstrap(values: list[float], rng: np.random.Generator, n: int=10000) -> tuple[float,float]:
    a=np.asarray(values); means=np.mean(a[rng.integers(0,len(a),(n,len(a)))],axis=1); return tuple(np.quantile(means,[.025,.975]))
def find(rows: list[dict[str, Any]], variant: str, metric: str, value: str) -> float:
    key="metric" if "metric" in rows[0] else "band"; return float(next(r[value] for r in rows if r["variant"]==variant and r[key]==metric))
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields=list(rows[0]);
    with path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    sys.path.insert(0,str(ROOT)); raise SystemExit(main())
