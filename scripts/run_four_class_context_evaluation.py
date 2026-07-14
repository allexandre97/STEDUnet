#!/usr/bin/env python
"""Run the controlled five-fold four-class ContextUNet evaluation on physical GPU 1."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fibras.annotations import build_real_annotation_sample
from scripts.analyze_b1_uncertain_ignore import load_records
from scripts.run_gated_context_evaluation import INITIAL, PYTHON, SUITE, latest_checkpoint, write_json


EXPERIMENT_NAME = "four_class_context_unet_v1"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    require_gpu_one(args.gpus)
    out = args.out or args.suite_root / EXPERIMENT_NAME
    out.mkdir(parents=True, exist_ok=True)
    started, failures = time.time(), []
    try:
        preflight(args, out)
        for fold in range(5):
            run_fold(fold, args, out)
        combine_training_histories(out)
        command = [
            str(PYTHON), "scripts/analyze_four_class_context_evaluation.py",
            "--suite-root", str(args.suite_root), "--experiment-root", str(out),
            "--manifest", str(args.manifest), "--fold-manifest", str(args.fold_manifest),
            "--image-root", str(args.image_root), "--annotation-root", str(args.annotation_root),
        ]
        if not args.dry_run:
            result = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"})
            if result.returncode:
                failures.append(f"analysis exited {result.returncode}")
    except Exception as exc:
        failures.append(str(exc))
    write_execution_status(out, started, failures)
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", type=Path, default=SUITE)
    p.add_argument("--out", type=Path)
    p.add_argument("--gpus", default="1")
    p.add_argument("--manifest", type=Path, default=ROOT / "data_manifests/real_annotations.csv")
    p.add_argument("--fold-manifest", type=Path, default=ROOT / "data_manifests/real_annotation_folds_v1.csv")
    p.add_argument("--image-root", type=Path, default=Path("/ssd/STED_dataset/data"))
    p.add_argument("--annotation-root", type=Path, default=Path("/cephfs/mhuang/STED_dataset/manual_annotations"))
    p.add_argument("--dry-run", action="store_true")
    return p


def require_gpu_one(raw: str) -> str:
    if [value.strip() for value in raw.split(",") if value.strip()] != ["1"]:
        raise ValueError("four-class evaluation is restricted to exactly physical GPU 1; GPU 0 is prohibited")
    return "1"


def preflight(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    required = [INITIAL, args.manifest, args.fold_manifest]
    required += [args.suite_root / "crops_256" / f"fold_{fold}/real_crop_manifest.csv" for fold in range(5)]
    required += [args.suite_root / "runs/b1_real_only" / f"fold_{fold}/model_best.pt" for fold in range(5)]
    required += [args.suite_root / "uncertain_ignore_nested_cv/evaluations/S_s0.05" / f"fold_{fold}/test/summary.json" for fold in range(5)]
    required += [args.suite_root / "gated_context_unet_g2/evaluations" / f"fold_{fold}/test/summary.json" for fold in range(5)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing controlled-suite inputs: {missing}")

    records = load_records(args.manifest, args.fold_manifest, args.image_root, args.annotation_root)
    reconstruction = []
    for record in records:
        sample = build_real_annotation_sample(record.image_path, record.snakes_path, record.labels_path)
        semantic_uncertain = sample["real_semantic_mask"] == 255
        explicit_uncertain = sample["real_uncertain_ignore_mask"].astype(bool)
        mismatch = int(np.count_nonzero(semantic_uncertain ^ explicit_uncertain))
        reconstruction.append({"sample_id": record.sample_id, "uncertain_pixels": int(semantic_uncertain.sum()), "mismatched_pixels": mismatch})
        if mismatch:
            raise ValueError(f"{record.sample_id}: reconstructed uncertain mask mismatch ({mismatch} pixels)")

    fold_rows = list(csv.DictReader(args.fold_manifest.open(encoding="utf-8")))
    if len({row["sample_id"] for row in fold_rows if row["partition"] == "test"}) != 20:
        raise ValueError("fold manifest does not contain exactly 20 unique outer-test images")
    crop_counts = [fold_crop_counts(args.suite_root / "crops_256" / f"fold_{fold}/real_crop_manifest.csv", fold) for fold in range(5)]
    invalid = [row for row in crop_counts if not row["train_has_all_four_classes"]]
    if invalid:
        raise ValueError(f"training folds missing classes: {invalid}")
    report = {"uncertain_reconstruction": reconstruction, "fold_class_crop_counts": crop_counts}
    write_json(out / "preflight_report.json", report)
    return report


def fold_crop_counts(path: Path, fold: int) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    selected = [row for row in rows if row["split"] == "train"]
    counts = {"background_positive_crops": 0, "fibre_positive_crops": 0, "clump_positive_crops": 0, "uncertain_positive_crops": 0}
    pixels = {"background_pixels": 0, "fibre_pixels": 0, "clump_pixels": 0, "uncertain_pixels": 0}
    for row in selected:
        total = 256 * 256
        fibre = int(row["fibrous_tau_pixels"])
        clump = int(row["clump_pixels"])
        uncertain = int(row["uncertain_ignore_pixels"])
        background = total - fibre - clump - uncertain
        values = {"background": background, "fibre": fibre, "clump": clump, "uncertain": uncertain}
        for name, value in values.items():
            pixels[f"{name}_pixels"] += value
            if value > 0:
                counts[f"{name}_positive_crops"] += 1
    return {
        "outer_fold": fold, "train_crops": len(selected), **pixels, **counts,
        "train_has_all_four_classes": all(pixels[f"{name}_pixels"] > 0 for name in ("background", "fibre", "clump", "uncertain")),
    }


def training_command(args: argparse.Namespace, run_dir: Path, fold: int) -> list[str]:
    return [
        str(PYTHON), "scripts/train_first_baseline.py",
        "--real-manifest", str(args.suite_root / "crops_256" / f"fold_{fold}/real_crop_manifest.csv"),
        "--synthetic-real-ratio", "0:100", "--init-checkpoint", str(INITIAL), "--out", str(run_dir),
        "--epochs", "19", "--stage1-epochs", "4", "--stage1-learning-rate", "1e-4",
        "--learning-rate", "2e-5", "--early-stop-patience", "4", "--early-stop-warmup-epochs", "4",
        "--batch-size", "5", "--patch-size", "256", "--patches-per-sample", "4",
        "--validation-patches-per-sample", "1", "--patches-per-epoch", "256", "--num-workers", "0",
        "--seed", "123", "--device", "cuda:0", "--model-variant", "four_class_context_unet",
        "--context-module", "aspp", "--aspp-dilations", "1,2,4,8",
        "--lambda-semantic", "1.0", "--lambda-dice", "0.0", "--lambda-skeleton", "0.5",
        "--four-class-semantic-loss", "pixel_ce", "--four-class-uncertain-bias", "-6.0",
        "--four-class-skeleton-mask", "complete",
        "--lambda-uncertainty", "0", "--lambda-uncertain-fibrous", "0",
        "--lambda-clump-anti-fibrous", "0", "--lambda-clump-anti-skeleton", "0",
        "--uncertain-skeleton-policy", "ignore", "--best-metric", "macro_image_fibrous_dice",
        "--best-mode", "max", "--save-best-checkpoint", "--save-checkpoint-every", "1",
        "--augmentation", "--cache-samples", "--qa-panel-count", "8",
    ]


def evaluation_command(args: argparse.Namespace, checkpoint: Path, out: Path, fold: int, partition: str) -> list[str]:
    return [
        str(PYTHON), "scripts/evaluate_real_annotation_batch.py", "--manifest", str(args.manifest),
        "--image-root", str(args.image_root), "--annotation-root", str(args.annotation_root),
        "--fold-manifest", str(args.fold_manifest), "--outer-fold", str(fold), "--partition", partition,
        "--checkpoint", str(checkpoint), "--out", str(out), "--patch-size", "256", "--tile-overlap", "64",
        "--batch-size", "8", "--device", "cuda:0", "--uncertainty-gating", "none",
    ]


def run_fold(fold: int, args: argparse.Namespace, out: Path) -> None:
    run_dir = out / "runs" / f"fold_{fold}"
    validation_dir = out / "evaluations" / f"fold_{fold}/validation"
    test_dir = out / "evaluations" / f"fold_{fold}/test"
    log_dir = out / "logs" / f"fold_{fold}"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not (run_dir / "checkpoint_summary.json").exists():
        command = training_command(args, run_dir, fold)
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
        execute(evaluation_command(args, checkpoint, validation_dir, fold, "validation"), log_dir / "validation.stdout.log", log_dir / "validation.stderr.log", args.dry_run)
    if not (test_dir / "summary.json").exists():
        execute(evaluation_command(args, checkpoint, test_dir, fold, "test"), log_dir / "test.stdout.log", log_dir / "test.stderr.log", args.dry_run)
    write_json(log_dir / "job_status.json", {
        "status": "dry_run" if args.dry_run else "completed", "fold": fold,
        "physical_gpu": "1", "cuda_visible_devices": "1", "logical_device": "cuda:0",
        "runtime_seconds": time.time() - started, "checkpoint": str(checkpoint),
        "run_directory": str(run_dir), "validation_directory": str(validation_dir),
        "test_directory": str(test_dir),
    })


def execute(command: list[str], stdout: Path, stderr: Path, dry_run: bool) -> None:
    mapping = "physical_gpu=1 CUDA_VISIBLE_DEVICES=1 logical_device=cuda:0"
    print(f"{mapping} {' '.join(command)}", flush=True)
    if dry_run:
        return
    with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
        out.write(mapping + "\n")
        out.flush()
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"}, stdout=out, stderr=err)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}; see {stdout} and {stderr}")


def combine_training_histories(out: Path) -> None:
    rows = []
    for fold in range(5):
        path = out / f"runs/fold_{fold}/training_log.csv"
        if path.exists():
            rows.extend({"outer_fold": fold, **row} for row in csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with (out / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_execution_status(out: Path, started: float, failures: list[str]) -> None:
    write_json(out / "execution_status.json", {
        "status": "failed" if failures else "completed", "failures": failures,
        "runtime_seconds": time.time() - started, "physical_gpu": "1",
        "cuda_visible_devices": "1", "logical_device": "cuda:0",
        "runner": str(Path(__file__).resolve()),
    })


if __name__ == "__main__":
    raise SystemExit(main())
