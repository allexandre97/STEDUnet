#!/usr/bin/env python
"""Run the single confirmatory G2 gated ContextUNet experiment on physical GPU 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_gated_context_evaluation import (
    INITIAL, LOSS_NAMES, PYTHON, SUITE, collapse_diagnostic, evaluation_command, gpu_name,
    latest_checkpoint, preflight, probability_summary, write_execution_report, write_json,
)

EXPERIMENT_NAME = "gated_context_unet_g2"
G1 = SUITE / "gated_context_unet_v1"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    require_gpu_one(args.gpus)
    out = args.out or args.suite_root / EXPERIMENT_NAME
    out.mkdir(parents=True, exist_ok=True)
    preflight(args)
    started, failures = time.time(), []
    if not args.skip_smoke:
        try:
            run_fold(0, args, out, epochs=2, smoke=True)
            report = smoke_report(args, out)
            write_json(out / "smoke/structural_report.json", report)
            write_smoke_markdown(out / "smoke_test_report.md", report)
            if not report["structurally_valid"]:
                failures.append("two-epoch smoke failed structural validity checks")
        except Exception as exc:
            failures.append(f"smoke/fold_0: {exc}")
        write_execution_report(out, started, failures, "smoke")
        if failures or args.smoke_only:
            return 1 if failures else 0
    for fold in range(5):
        try:
            run_fold(fold, args, out, epochs=19, smoke=False)
        except Exception as exc:
            failures.append(f"fold_{fold}: {exc}")
            break
    if not failures:
        combine_training_histories(out)
        command = [
            str(PYTHON), "scripts/analyze_gated_context_recalibration.py",
            "--suite-root", str(args.suite_root), "--experiment-root", str(out),
            "--manifest", str(args.manifest), "--fold-manifest", str(args.fold_manifest),
            "--image-root", str(args.image_root), "--annotation-root", str(args.annotation_root),
        ]
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"})
        if result.returncode:
            failures.append(f"analysis exited {result.returncode}")
    write_execution_report(out, started, failures, "complete")
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, default=SUITE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data_manifests/real_annotations.csv")
    parser.add_argument("--fold-manifest", type=Path, default=ROOT / "data_manifests/real_annotation_folds_v1.csv")
    parser.add_argument("--image-root", type=Path, default=Path("/ssd/STED_dataset/data"))
    parser.add_argument("--annotation-root", type=Path, default=Path("/cephfs/mhuang/STED_dataset/manual_annotations"))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def require_gpu_one(raw: str) -> str:
    if [value.strip() for value in raw.split(",") if value.strip()] != ["1"]:
        raise ValueError("G2 is restricted to exactly physical GPU 1; GPU 0 is prohibited")
    return "1"


def training_command(args: argparse.Namespace, run_dir: Path, fold: int, epochs: int) -> list[str]:
    return [
        str(PYTHON), "scripts/train_first_baseline.py",
        "--real-manifest", str(args.suite_root / "crops_256" / f"fold_{fold}/real_crop_manifest.csv"),
        "--synthetic-real-ratio", "0:100", "--init-checkpoint", str(INITIAL), "--out", str(run_dir),
        "--epochs", str(epochs), "--stage1-epochs", "4", "--stage1-learning-rate", "1e-4",
        "--learning-rate", "2e-5", "--early-stop-patience", "4", "--early-stop-warmup-epochs", "4",
        "--batch-size", "5", "--patch-size", "256", "--patches-per-sample", "4",
        "--validation-patches-per-sample", "1", "--patches-per-epoch", "256", "--num-workers", "0",
        "--seed", "123", "--device", "cuda:0", "--model-variant", "gated_context_unet",
        "--context-module", "aspp", "--aspp-dilations", "1,2,4,8",
        "--lambda-foreground", "1.5", "--lambda-gate", "1.0", "--lambda-morphology", "1.0",
        "--lambda-joint", "0.5", "--lambda-skeleton", "0.5",
        "--lambda-uncertainty", "0", "--lambda-uncertain-fibrous", "0",
        "--lambda-clump-anti-fibrous", "0", "--lambda-clump-anti-skeleton", "0",
        "--uncertain-skeleton-policy", "ignore", "--best-metric", "macro_image_fibrous_dice",
        "--best-mode", "max", "--save-best-checkpoint", "--save-checkpoint-every", "1",
        "--augmentation", "--cache-samples", "--qa-panel-count", "8",
    ]


def run_fold(fold: int, args: argparse.Namespace, out: Path, *, epochs: int, smoke: bool) -> None:
    base = out / "smoke" if smoke else out
    run_dir = base / "runs" / f"fold_{fold}"
    validation_dir = base / "evaluations" / f"fold_{fold}/validation"
    test_dir = base / "evaluations" / f"fold_{fold}/test"
    log_dir = base / "logs" / f"fold_{fold}"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not (run_dir / "checkpoint_summary.json").exists():
        command = training_command(args, run_dir, fold, epochs)
        resume = latest_checkpoint(run_dir)
        if resume:
            index = command.index("--init-checkpoint")
            del command[index:index + 2]
            command += ["--resume-checkpoint", str(resume)]
        execute(command, log_dir / "train.stdout.log", log_dir / "train.stderr.log", args.dry_run)
    checkpoint = run_dir / "model_best.pt"
    if not args.dry_run and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not (validation_dir / "summary.json").exists():
        execute(evaluation_command(args, checkpoint, validation_dir, fold, "validation"),
                log_dir / "validation.stdout.log", log_dir / "validation.stderr.log", args.dry_run)
    if not smoke and not (test_dir / "summary.json").exists():
        execute(evaluation_command(args, checkpoint, test_dir, fold, "test"),
                log_dir / "test.stdout.log", log_dir / "test.stderr.log", args.dry_run)
    write_json(log_dir / "job_status.json", {
        "status": "dry_run" if args.dry_run else "completed", "fold": fold,
        "physical_gpu": "1", "logical_device": "cuda:0", "cuda_visible_devices": "1",
        "gpu_name": gpu_name("1"), "runtime_seconds": time.time() - started,
        "checkpoint": str(checkpoint), "run_directory": str(run_dir),
        "validation_directory": str(validation_dir), "test_directory": None if smoke else str(test_dir),
    })


def execute(command: list[str], stdout: Path, stderr: Path, dry_run: bool) -> None:
    mapping = f"physical_gpu=1 CUDA_VISIBLE_DEVICES=1 logical_device=cuda:0 gpu_name={gpu_name('1')}"
    print(f"{mapping} {' '.join(command)}", flush=True)
    if dry_run:
        return
    with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
        out.write(mapping + "\n")
        out.flush()
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
                                stdout=out, stderr=err)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}; see {stdout} and {stderr}")


def smoke_report(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    run = out / "smoke/runs/fold_0"
    evaluation = out / "smoke/evaluations/fold_0/validation"
    rows = list(csv.DictReader((run / "training_log.csv").open(encoding="utf-8")))
    losses = {name: [float(row[f"train_{name}" if name != "loss" else "train_loss"]) for row in rows]
              for name in LOSS_NAMES}
    validation_losses = {name: [float(row[f"validation_{name}" if name != "loss" else "validation_loss"])
                                for row in rows] for name in LOSS_NAMES}
    finite = all(math.isfinite(value) for group in (losses, validation_losses)
                 for values in group.values() for value in values)
    plausible = len(rows) == 2 and losses["loss"][-1] <= losses["loss"][0] * 1.05
    diagnostics, summaries, comparisons = [], [], []
    for path in sorted(evaluation.glob("*/predictions.npz")):
        with np.load(path) as data:
            pred = {name: data[name] for name in data.files}
        diagnostics.append(collapse_diagnostic(path.parent.name, pred))
        g1_path = G1 / "smoke/evaluations/fold_0/validation" / path.parent.name / "predictions.npz"
        if not g1_path.exists():
            g1_path = G1 / "evaluations/fold_0/validation" / path.parent.name / "predictions.npz"
        with np.load(g1_path) as data:
            for name in ("foreground_probability", "quantifiability_probability"):
                comparisons.append({
                    "sample_id": path.parent.name, "probability": name,
                    "g1_mean": float(data[name].mean()), "g2_mean": float(pred[name].mean()),
                })
        for name in (
            "foreground_probability", "quantifiability_probability", "conditional_fibrous_probability",
            "conditional_clump_probability", "final_background_probability", "final_fibrous_probability",
            "final_clump_probability", "final_uncertain_probability", "raw_centreline_probability",
            "final_skeleton_probability",
        ):
            summaries.append(probability_summary(path.parent.name, name, pred[name]))
    collapsed = any(row["collapsed"] for row in diagnostics)
    panel = write_smoke_panel(args, evaluation, out / "smoke/validation_g2_qa_panel.png")
    init = json.loads((run / "run_metadata.json").read_text())["initial_checkpoint"]
    valid_init = not init["missing_backbone_keys"] and len(init["loaded_keys"]) == 34
    return {
        "structurally_valid": finite and plausible and not collapsed and valid_init,
        "finite_losses": finite, "plausible_loss_behavior": plausible, "train_losses": losses,
        "validation_losses": validation_losses, "collapse_diagnostics": diagnostics,
        "probability_summaries": summaries, "g1_g2_q_g_comparison": comparisons,
        "qa_panel": str(panel), "initialization_report": init,
    }


def write_smoke_panel(args: argparse.Namespace, evaluation: Path, path: Path) -> Path:
    from fibras.annotations import build_real_annotation_sample
    from scripts.analyze_b1_uncertain_ignore import load_records
    from scripts.evaluate_real_pilot_baseline import draw_tile, gray_rgb, heatmap, real_mask_rgb

    pred_path = sorted(evaluation.glob("*/predictions.npz"))[0]
    record = next(row for row in load_records(
        args.manifest, args.fold_manifest, args.image_root, args.annotation_root
    ) if row.sample_id == pred_path.parent.name)
    sample = build_real_annotation_sample(record.image_path, record.snakes_path, record.labels_path)
    with np.load(pred_path) as data:
        pred = {name: data[name] for name in data.files}
    raw = sample["image_uint8"] if sample["image_uint8"] is not None else sample["image_float"]
    tiles = [draw_tile(gray_rgb(raw), "raw STED"),
             draw_tile(real_mask_rgb(sample["real_semantic_mask"]), "expert labels")]
    for name, label in (
        ("foreground_probability", "q foreground"), ("quantifiability_probability", "g quantifiable"),
        (None, "1-g uncertainty"), ("conditional_fibrous_probability", "conditional fibre"),
        ("conditional_clump_probability", "conditional clump"),
        ("final_background_probability", "final background"), ("final_fibrous_probability", "final fibre"),
        ("final_clump_probability", "final clump"), ("final_uncertain_probability", "final uncertain"),
        ("final_skeleton_probability", "final skeleton"),
    ):
        values = 1 - pred["quantifiability_probability"] if name is None else pred[name]
        tiles.append(draw_tile(heatmap(values), label))
    tiles.append(draw_tile(real_mask_rgb(pred["semantic_class_map"]), "semantic map"))
    columns = 4
    panel = Image.new("RGB", (columns * tiles[0].width, math.ceil(len(tiles) / columns) * tiles[0].height), "white")
    for index, tile in enumerate(tiles):
        panel.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def write_smoke_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# G2 smoke test", "", f"- Structurally valid: {report['structurally_valid']}",
             f"- All losses finite: {report['finite_losses']}",
             f"- Plausible loss behavior: {report['plausible_loss_behavior']}",
             f"- QA panel: {report['qa_panel']}", "", "## G1/G2 q and g means", "",
             "| Image | Probability | G1 | G2 |", "|---|---|---:|---:|"]
    for row in report["g1_g2_q_g_comparison"]:
        lines.append(f"| {row['sample_id']} | {row['probability']} | {row['g1_mean']:.5f} | {row['g2_mean']:.5f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def combine_training_histories(out: Path) -> None:
    rows = []
    for fold in range(5):
        with (out / f"runs/fold_{fold}/training_log.csv").open(encoding="utf-8") as handle:
            rows.extend({"outer_fold": fold, **row} for row in csv.DictReader(handle))
    fields = sorted({key for row in rows for key in row})
    with (out / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
