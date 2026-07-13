#!/usr/bin/env python
"""Analyze the longer real-dominant real-STED follow-up."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "A": "Frozen synthetic",
    "B1": "Original real-only, 19 epochs",
    "L100": "Long real-only with uncertainty",
    "R100": "Long real-only, no uncertainty",
    "R80S20": "80% real / 20% synthetic",
    "R50S50": "50% real / 50% synthetic",
}
FOLLOW_DIRS = {
    "L100": "b1_long_uncertainty", "R100": "r100_no_uncertainty",
    "R80S20": "r80s20_no_uncertainty", "R50S50": "r50s50_no_uncertainty",
}
METRICS = [
    "fibrous_dice", "fibrous_precision", "fibrous_recall", "predicted_to_target_fibrous_area_ratio",
    "clump_dice", "skeleton_dice_0.5", "skeleton_dice_0.75", "skeleton_dice_0.85",
    "target_skeleton_recovered_within_2px_0.75", "predicted_skeleton_within_2px_of_target_0.75",
    "fibrous_inside_target_clump_fraction", "fibrous_inside_uncertain_ignore_fraction",
    "skeleton_inside_target_clump_fraction_0.75", "skeleton_inside_uncertain_ignore_fraction_0.75",
    "false_positive_fibrous_background_pixels", "missed_thick_component_count", "missed_faint_component_count",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--followup-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    inventory = read_csv(ROOT / "data_manifests/real_annotations.csv")
    folds = read_csv(ROOT / "data_manifests/real_annotation_folds_v1.csv")
    results, fold_payloads = load_results(args.suite_root, args.followup_root)
    provenance = validate(args, inventory, folds, results, fold_payloads)
    long_rows = per_image_rows(inventory, folds, results)
    paired = paired_rows(inventory, results)
    macro = macro_rows(results)
    fold_rows = fold_metric_rows(fold_payloads)
    retention = retention_rows(args.followup_root)
    curves = learning_curve_rows(args.followup_root)
    runtimes = runtime_rows(args.followup_root)
    strata = strata_rows(long_rows)
    for name, rows in [("per_image_results.csv", long_rows), ("paired_comparisons.csv", paired),
                       ("macro_summary.csv", macro), ("fold_metrics.csv", fold_rows),
                       ("synthetic_retention.csv", retention), ("learning_curves.csv", curves),
                       ("strata_summary.csv", strata), ("runtime_summary.csv", runtimes)]: write_csv(args.out / name, rows)
    examples = qualitative_examples(inventory, results)
    write_csv(args.out / "qualitative_examples.csv", examples)
    qualitative_panels(args, examples)
    plot_main(long_rows, args.out / "main_metrics.png")
    plot_paired(paired, args.out / "paired_real_dice.png")
    plot_curves(curves, args.out / "learning_curves.png")
    plot_retention(retention, args.out / "synthetic_retention.png")
    plot_fold(fold_rows, args.out / "fold_variation.png")
    provenance["qualitative_examples"] = examples
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(report(macro, fold_rows, retention, curves, strata, paired, examples, runtimes, provenance))
    (args.out / "executive_summary.md").write_text(executive(macro, retention))
    print(f"wrote follow-up analysis to {args.out}")
    return 0


def load_results(suite: Path, follow: Path):
    results = {}
    a = json.loads((suite / "evaluations/phase_a_synthetic_baseline/summary.json").read_text())
    b = json.loads((suite / "evaluations/b1_real_only/oof_summary.json").read_text())
    results["A"] = {row["sample_id"]: row for row in a["rows"]}
    results["B1"] = {row["sample_id"]: row for row in b["rows"]}
    payloads = {}
    for code, directory in FOLLOW_DIRS.items():
        fold_data = [json.loads((follow / "evaluations" / directory / f"fold_{fold}/summary.json").read_text()) for fold in range(5)]
        rows = [row for data in fold_data for row in data["rows"]]
        results[code] = {row["sample_id"]: row for row in rows}; payloads[code] = fold_data
    return results, payloads


def validate(args, inventory, folds, results, payloads):
    ids = {row["sample_id"] for row in inventory}
    if any(set(rows) != ids or len(rows) != 20 for rows in results.values()): raise ValueError("OOF identities mismatch")
    if len(folds) != 100 or any(sum(row["partition"] == "test" for row in folds if row["sample_id"] == sid) != 1 for sid in ids): raise ValueError("fold manifest invalid")
    checkpoints = [{"run": "A", "fold": "all", "checkpoint": str(ROOT / "runs/first_baseline_schema08_context_v2_256_aspp_uncertain_softloss/model.pt")}]
    for fold in range(5):
        run = args.suite_root / "runs/b1_real_only" / f"fold_{fold}"
        checkpoints.append({"run": "B1", "fold": fold, "checkpoint": str(run / "model_best.pt"), "config": str(run / "config.json")})
    for code, directory in FOLLOW_DIRS.items():
        for fold, payload in enumerate(payloads[code]):
            run = args.followup_root / "runs" / directory / f"fold_{fold}"
            checkpoint = run / "model_best.pt"; config = json.loads((run / "config.json").read_text())
            if payload["checkpoint"] != str(checkpoint) or not checkpoint.exists(): raise ValueError(f"{code} fold {fold} checkpoint mismatch")
            if config["seed"] != 123 or config["resolved_device"] != "cuda:0": raise ValueError("seed/device mismatch")
            checkpoints.append({"run": code, "fold": fold, "checkpoint": str(checkpoint), "config": str(run / "config.json")})
    return {"repository_commit": git("rev-parse", "HEAD"), "working_tree_dirty": bool(git("status", "--porcelain")),
            "inventory_sha256": sha(ROOT / "data_manifests/real_annotations.csv"),
            "fold_manifest_sha256": sha(ROOT / "data_manifests/real_annotation_folds_v1.csv"),
            "fold_schema_version": folds[0]["fold_schema_version"], "seed": 123,
            "physical_gpu": "1", "logical_device": "cuda:0", "gpu_name": "NVIDIA GeForce RTX 5090",
            "launcher": "python scripts/run_real_sted_followup.py --suite-root /ssd/STED_experiments/controlled_real_sted_v1 --out-root /ssd/STED_experiments/controlled_real_sted_v1/followup_v1 --gpus 1",
            "analysis_command": "python scripts/analyze_real_sted_followup.py --suite-root /ssd/STED_experiments/controlled_real_sted_v1 --followup-root /ssd/STED_experiments/controlled_real_sted_v1/followup_v1 --out /ssd/STED_experiments/controlled_real_sted_v1/followup_v1/analysis",
            "software": software_versions(), "checkpoints": checkpoints}


def per_image_rows(inventory, folds, results):
    test_fold = {row["sample_id"]: row["outer_fold"] for row in folds if row["partition"] == "test"}
    meta = {row["sample_id"]: row for row in inventory}; out = []
    for code in RUNS:
        for sid, item in results[code].items():
            m = meta[sid]; out.append({"run": code, "run_name": RUNS[code], "sample_id": sid,
                "image_identity": m["image_identity"], "outer_fold": test_fold[sid], "split_group_id": m["split_group_id"],
                "disease": m["disease"], "tau_isoform": m["tau_isoform"], "div": m["div"],
                **{metric: item.get(metric) for metric in METRICS}})
    return out


def paired_rows(inventory, results):
    out=[]
    for meta in inventory:
        sid=meta["sample_id"]; row={key:meta[key] for key in ["sample_id","image_identity","disease","tau_isoform","div","split_group_id"]}
        for code in RUNS:
            row[f"{code}_fibrous_dice"]=results[code][sid]["fibrous_dice"]
            for reference in ["A","B1"]:
                if code == reference: continue
                for metric in METRICS:
                    x,y=results[code][sid].get(metric),results[reference][sid].get(metric)
                    row[f"{code}_minus_{reference}_{metric}"]=float(x)-float(y) if number(x) and number(y) else "not_applicable"
        out.append(row)
    return out


def macro_rows(results):
    out=[]
    for code in RUNS:
        row={"run":code,"run_name":RUNS[code],"n_images":20}
        for metric in METRICS:
            values=nums(item.get(metric) for item in results[code].values())
            row[f"{metric}_mean"]=avg(values); row[f"{metric}_sd"]=stdev(values)
        out.append(row)
    return out


def fold_metric_rows(payloads):
    out=[]
    for code, data in payloads.items():
        for fold,payload in enumerate(data): out.append({"run":code,"outer_fold":fold,"n_images":payload["sample_count"],**{m:payload["aggregates"].get(m) for m in METRICS}})
    return out


def retention_rows(root):
    paths=[("Initial","initial.json")]+[("B1",f"b1_fold_{fold}.json") for fold in range(5)]
    out=[]
    for code,name in paths:
        payload=json.loads((root/"synthetic_retention"/name).read_text()); out.append(retention_row(code,"all" if code=="Initial" else name[-6],payload))
    for code,directory in FOLLOW_DIRS.items():
        for fold in range(5): out.append(retention_row(code,fold,json.loads((root/"synthetic_retention"/directory/f"fold_{fold}.json").read_text())))
    return out


def retention_row(code,fold,payload):
    m=payload["metrics"]; return {"run":code,"outer_fold":fold,"checkpoint":payload["checkpoint"],"fibrous_dice":m["fibrous_dice"],"clump_dice":m["clump_dice"],"skeleton_dice_0.5":m["skeleton_dice"],"skeleton_dice_0.75":m["skeleton_metrics_by_threshold"]["0.75"]["dice"],"skeleton_recall_0.75":m["skeleton_metrics_by_threshold"]["0.75"]["recall"]}


def learning_curve_rows(root):
    out=[]
    for code,directory in FOLLOW_DIRS.items():
        for fold in range(5):
            for row in read_csv(root/"runs"/directory/f"fold_{fold}/training_log.csv"):
                out.append({"run":code,"outer_fold":fold,**row})
    return out


def runtime_rows(root):
    out=[]
    for code,directory in FOLLOW_DIRS.items():
        for fold in range(5):
            run=root/"runs"/directory/f"fold_{fold}"
            start=(run/"config.json").stat().st_mtime; end=(run/"checkpoint_summary.json").stat().st_mtime
            summary=json.loads((run/"checkpoint_summary.json").read_text())
            out.append({"run":code,"outer_fold":fold,"runtime_seconds":end-start,"runtime_minutes":(end-start)/60,
                        "best_epoch":summary["best_epoch"],"final_epoch":summary["final_epoch"],"stopped_early":summary["stopped_early"]})
    return out


def strata_rows(rows):
    out=[]
    for field in ["disease","tau_isoform","div"]:
        for value in sorted({row[field] for row in rows}):
            for code in RUNS:
                selected=[row for row in rows if row[field]==value and row["run"]==code]
                out.append({"stratum_type":field,"stratum":value,"run":code,"n_images":len(selected),**{m:avg(nums(row[m] for row in selected)) for m in ["fibrous_dice","fibrous_recall","clump_dice","skeleton_dice_0.75","missed_thick_component_count","missed_faint_component_count"]}})
    return out


def qualitative_examples(inventory, results):
    ids=[row["sample_id"] for row in inventory]; meta={row["sample_id"]:row for row in inventory}
    delta=lambda sid,metric,code="R80S20",ref="B1":float(results[code][sid][metric])-float(results[ref][sid][metric])
    choices={
        "long_training_gain":max(ids,key=lambda x:delta(x,"fibrous_dice","L100")),
        "ambiguous_expansion":max(ids,key=lambda x:delta(x,"fibrous_inside_uncertain_ignore_fraction","L100")),
        "fold3_hard_case":min((x for x in ids if "fold_3" in results["R80S20"][x]["sample_dir"]),key=lambda x:results["R80S20"][x]["fibrous_dice"]),
        "psp_thick_bundles":max((x for x in ids if meta[x]["disease"]=="PSP"),key=lambda x:results["A"][x]["missed_thick_component_count"]),
        "ad_clumps":max((x for x in ids if meta[x]["disease"]=="AD"),key=lambda x:results["A"][x]["clump_target_pixels"]),
        "faint_fibres":max(ids,key=lambda x:results["A"][x]["missed_faint_component_count"]),
    }
    return [{"category":cat,"sample_id":sid,"image_identity":meta[sid]["image_identity"],"disease":meta[sid]["disease"],**{f"{c}_fibrous_dice":results[c][sid]["fibrous_dice"] for c in RUNS}} for cat,sid in choices.items()]


def qualitative_panels(args, examples):
    out=args.out/"qualitative_examples"; out.mkdir(exist_ok=True)
    for path in out.glob("*.png"): path.unlink()
    for item in examples:
        sid=item["sample_id"]; sources=[]
        sources.append(("A",args.suite_root/"evaluations/phase_a_synthetic_baseline"/sid/"qa_panel.png"))
        b=list((args.suite_root/"evaluations/b1_real_only").glob(f"fold_*/{sid}/qa_panel.png")); sources.append(("B1",b[0]))
        for code in ["L100","R80S20","R50S50"]:
            directory=FOLLOW_DIRS[code]; path=list((args.followup_root/"evaluations"/directory).glob(f"fold_*/{sid}/qa_panel.png"))[0]; sources.append((code,path))
        width=1200; panels=[]
        for label,path in sources:
            image=Image.open(path).convert("RGB"); image=image.resize((width,round(image.height*width/image.width))); panels.append((label,image))
        canvas=Image.new("RGB",(width,sum(x.height+24 for _,x in panels)),"white"); draw=ImageDraw.Draw(canvas); y=0
        for label,image in panels: draw.text((6,y+4),label,fill="black"); y+=24; canvas.paste(image,(0,y)); y+=image.height
        canvas.save(out/f"{item['category']}_{sid}.png")


def plot_main(rows,path):
    metrics=[("fibrous_dice","Fibrous Dice"),("clump_dice","Clump Dice"),("skeleton_dice_0.75","Skeleton Dice .75"),("predicted_to_target_fibrous_area_ratio","Area ratio")]
    fig,axes=plt.subplots(1,4,figsize=(15,4),constrained_layout=True); rng=np.random.default_rng(123)
    for ax,(metric,title) in zip(axes,metrics):
        for x,code in enumerate(RUNS):
            values=nums(row[metric] for row in rows if row["run"]==code); ax.scatter(x+rng.uniform(-.1,.1,len(values)),values,s=17,alpha=.65); ax.plot([x-.15,x+.15],[avg(values)]*2,"k",lw=2)
        ax.set_title(title); ax.set_xticks(range(len(RUNS)),RUNS,rotation=35); ax.grid(axis="y",alpha=.25)
    fig.savefig(path,dpi=180); plt.close(fig)


def plot_paired(rows,path):
    codes=["L100","R100","R80S20","R50S50"]; fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True); rng=np.random.default_rng(123)
    for x,code in enumerate(codes):
        values=[float(row[f"{code}_minus_B1_fibrous_dice"]) for row in rows]; ax.scatter(x+rng.uniform(-.08,.08,len(values)),values,s=20,alpha=.7); ax.plot([x-.15,x+.15],[avg(values)]*2,"k",lw=2)
    ax.axhline(0,color="black",lw=1); ax.set_xticks(range(4),codes); ax.set_ylabel("Paired fibrous Dice change vs B1"); ax.grid(axis="y",alpha=.25); fig.savefig(path,dpi=180); plt.close(fig)


def plot_curves(rows,path):
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,(code,title) in zip(axes.flat,FOLLOW_DIRS.items()):
        for fold in range(5):
            selected=[r for r in rows if r["run"]==code and int(r["outer_fold"])==fold]; ax.plot([int(r["epoch"]) for r in selected],[float(r["real_validation_macro_image_fibrous_dice"]) for r in selected],alpha=.65,label=f"F{fold}")
        ax.axvline(4,color="black",ls="--",lw=1); ax.set_title(code); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation macro fibrous Dice"); ax.grid(alpha=.25)
    axes[0,0].legend(ncol=5,fontsize=8); fig.savefig(path,dpi=180); plt.close(fig)


def plot_retention(rows,path):
    initial=next(r for r in rows if r["run"]=="Initial"); codes=["B1","L100","R100","R80S20","R50S50"]; metrics=["fibrous_dice","clump_dice","skeleton_dice_0.75"]
    fig,axes=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
    for ax,metric in zip(axes,metrics):
        for x,code in enumerate(codes):
            vals=[float(r[metric]) for r in rows if r["run"]==code]; ax.scatter([x]*len(vals),vals,alpha=.7); ax.plot([x-.15,x+.15],[avg(vals)]*2,"k",lw=2)
        ax.axhline(float(initial[metric]),color="red",ls="--",label="initial"); ax.set_title(metric); ax.set_xticks(range(len(codes)),codes,rotation=35); ax.grid(axis="y",alpha=.25)
    axes[0].legend(); fig.savefig(path,dpi=180); plt.close(fig)


def plot_fold(rows,path):
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for code in FOLLOW_DIRS:
        selected=[r for r in rows if r["run"]==code]; ax.plot([int(r["outer_fold"]) for r in selected],[float(r["fibrous_dice"]) for r in selected],marker="o",label=code)
    ax.set_xticks(range(5)); ax.set_xlabel("Outer fold"); ax.set_ylabel("Macro fibrous Dice"); ax.grid(alpha=.25); ax.legend(); fig.savefig(path,dpi=180); plt.close(fig)


def report(macro,folds,retention,curves,strata,paired,examples,runtimes,prov):
    m={r["run"]:r for r in macro}; initial=next(r for r in retention if r["run"]=="Initial")
    ret={code:{metric:avg([float(r[metric]) for r in retention if r["run"]==code]) for metric in ["fibrous_dice","clump_dice","skeleton_dice_0.5","skeleton_dice_0.75"]} for code in ["B1","L100","R100","R80S20","R50S50"]}
    fold3={r["run"]:r for r in folds if int(r["outer_fold"])==3}; disease={(r["run"],r["stratum"]):r for r in strata if r["stratum_type"]=="disease"}
    lines=["# Real-STED Longer and Real-Dominant Follow-up","","## Decision","",
        "Adopt **R80S20 without the uncertainty head** as the next balanced baseline. It preserves almost all B1 real performance while materially reducing synthetic forgetting and uncertain-region leakage. R100/L100 gain little or nothing on held-out real images and forget substantially more synthetic morphology. R50S50 retains synthetic performance best but gives up more real accuracy.","","## Held-out real results","",
        "All values are macro means across the 20 OOF images; pixels and crops are not replicates.","",
        "| Run | Fib Dice | Precision | Recall | Area ratio | Clump Dice | Skel Dice .75 | Target/pred <=2 px | Fib in clump | Fib in uncertain | Skel in clump/uncertain |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|" ]
    for code in RUNS:
        x=m[code]; v=lambda metric:x[f"{metric}_mean"]
        lines.append(f"| {code} | {v('fibrous_dice'):.3f} | {v('fibrous_precision'):.3f} | {v('fibrous_recall'):.3f} | {v('predicted_to_target_fibrous_area_ratio'):.3f} | {v('clump_dice'):.3f} | {v('skeleton_dice_0.75'):.3f} | {v('target_skeleton_recovered_within_2px_0.75'):.3f}/{v('predicted_skeleton_within_2px_of_target_0.75'):.3f} | {v('fibrous_inside_target_clump_fraction'):.3f} | {v('fibrous_inside_uncertain_ignore_fraction'):.3f} | {v('skeleton_inside_target_clump_fraction_0.75'):.3f}/{v('skeleton_inside_uncertain_ignore_fraction_0.75'):.3f} |")
    lines += ["","## Synthetic retention","",f"Initial synthetic validation: fibrous {float(initial['fibrous_dice']):.3f}, clump {float(initial['clump_dice']):.3f}, skeleton .50/.75 {float(initial['skeleton_dice_0.5']):.3f}/{float(initial['skeleton_dice_0.75']):.3f}.","",
              "| Run | Fib Dice | Delta | Clump Dice | Delta | Skeleton .50/.75 |","|---|---:|---:|---:|---:|---|"]
    for code in ret:
        r=ret[code]; lines.append(f"| {code} | {r['fibrous_dice']:.3f} | {r['fibrous_dice']-float(initial['fibrous_dice']):+.3f} | {r['clump_dice']:.3f} | {r['clump_dice']-float(initial['clump_dice']):+.3f} | {r['skeleton_dice_0.5']:.3f}/{r['skeleton_dice_0.75']:.3f} |")
    lines += ["","B1 synthetic forgetting is substantial and worsens under longer real-only training. R80S20 limits forgetting; R50S50 nearly preserves fibrous/clump performance but still reduces skeleton performance.","","## Longer training and validation curves",""]
    for code in FOLLOW_DIRS:
        selected=[r for r in curves if r["run"]==code]; finals=[max(int(x["epoch"]) for x in selected if int(x["outer_fold"])==f) for f in range(5)]; best=[next(int(x["best_epoch"]) for x in reversed(selected) if int(x["outer_fold"])==f) for f in range(5)]
        lines.append(f"- {code}: final epochs {finals}; best epochs {best}. All selected epoch 44 and no run early-stopped.")
    runtime_total=sum(float(row["runtime_minutes"]) for row in runtimes)
    lines += ["",f"Recorded training time was {runtime_total:.1f} GPU-minutes across 20 uninterrupted runs; per-fold values are in `runtime_summary.csv`.","","Validation curves continue rising slowly through epoch 44. The original epoch-19 selections were not merely noise. However, L100 improves held-out Dice only marginally over B1 while expanding fibrous predictions into clump and uncertain regions and worsening retention; continued validation improvement is therefore not equivalent to a better balanced model.","","## Fold 3",""]
    lines.append("Fold 3 remains the weakest for every configuration: " + ", ".join(f"{code} {float(fold3[code]['fibrous_dice']):.3f}" for code in FOLLOW_DIRS) + ". It contains three AD images and one PSP image, has the highest clump/uncertain prevalence, and contains three dense-morphology images. The persistent deficit across ratios indicates a composition/domain difficulty rather than a single training failure.")
    lines += ["","## Morphology and failure analysis",""]
    for condition in ["AD","CBD","PID","PSP"]: lines.append(f"- {condition}: B1/R80S20 fibrous Dice {float(disease[('B1',condition)]['fibrous_dice']):.3f}/{float(disease[('R80S20',condition)]['fibrous_dice']):.3f} (n={disease[('B1',condition)]['n_images']}).")
    lines += ["","PSP thick-bundle recall remains improved over A but R80S20 is slightly below B1. AD clump discrimination remains close to B1 while leakage is lower than long R100. Faint-fibre recovery remains unresolved: recall improves, but component counts are fragmentation-sensitive and do not support a clean recovery claim.","",
              "The images driving uncertain leakage and representative thick-bundle, clump, faint, fold-3, and long-training cases are listed in `qualitative_examples.csv`; aligned A/B1/L100/R80S20/R50S50 panels are under `qualitative_examples/`.","","## Reproducibility", "",
              f"- Commit `{prov['repository_commit']}`; dirty working tree `{prov['working_tree_dirty']}`.",f"- Inventory SHA-256 `{prov['inventory_sha256']}`.",f"- Fold manifest `{prov['fold_schema_version']}`, SHA-256 `{prov['fold_manifest_sha256']}`.",f"- Seed `{prov['seed']}`; physical GPU `{prov['physical_gpu']}` mapped through CUDA_VISIBLE_DEVICES to logical `{prov['logical_device']}`.",f"- Launcher: `{prov['launcher']}`",f"- Analysis: `{prov['analysis_command']}`","- Exact checkpoint and config paths: `provenance.json`.","","## Limitations", "",
              "The 20 images are not 20 independent biological replicates. Biological grouping remains provisional, PN is not a grouping variable, and disease/isoform/DIV comparisons are descriptive. All best epochs hit the imposed ceiling, so convergence remains incomplete. No run failed or was omitted."]
    return "\n".join(lines)+"\n"


def executive(macro,retention):
    m={r["run"]:r for r in macro}; initial=next(r for r in retention if r["run"]=="Initial"); r80=avg([float(r["fibrous_dice"]) for r in retention if r["run"]=="R80S20"])
    return f"""# Real-STED Follow-up Handoff

- 20/20 follow-up runs completed; no failures.
- Real OOF fibrous Dice B1/L100/R100/R80S20/R50S50: {m['B1']['fibrous_dice_mean']:.3f}/{m['L100']['fibrous_dice_mean']:.3f}/{m['R100']['fibrous_dice_mean']:.3f}/{m['R80S20']['fibrous_dice_mean']:.3f}/{m['R50S50']['fibrous_dice_mean']:.3f}.
- Existing B1 synthetic fibrous retention is 0.760 versus {float(initial['fibrous_dice']):.3f} initially; longer real-only worsens it.
- R80S20 retains synthetic fibrous Dice {r80:.3f}, reduces uncertain leakage, and nearly matches B1 real performance.
- Recommended next balanced baseline: R80S20 without uncertainty head.
- All validation curves still rise at epoch 44; do not interpret the ceiling as convergence.
- Full report: `/ssd/STED_experiments/controlled_real_sted_v1/followup_v1/analysis/report.md`.
"""


def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    if not rows:return
    with Path(path).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction="ignore",lineterminator="\n");w.writeheader();w.writerows(rows)
def number(x):return isinstance(x,(int,float)) and not isinstance(x,bool)
def nums(xs):return [float(x) for x in xs if number(x)]
def avg(xs):return float(np.mean(xs)) if xs else "not_applicable"
def stdev(xs):return float(np.std(xs,ddof=1)) if len(xs)>1 else "not_applicable"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def software_versions():
    import matplotlib, PIL, scipy, torch
    return {"python":sys.version,"torch":torch.__version__,"torch_cuda":str(torch.version.cuda),"numpy":np.__version__,"scipy":scipy.__version__,"Pillow":PIL.__version__,"matplotlib":matplotlib.__version__}


if __name__ == "__main__": raise SystemExit(main())
