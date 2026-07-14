#!/usr/bin/env python
"""Run the controlled five-fold gated ContextUNet evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, Thread
import time
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
PYTHON = Path("/lmb/home/alexandrebg/miniconda3/envs/fibras/bin/python")
SUITE = Path("/ssd/STED_experiments/controlled_real_sted_v1")
INITIAL = ROOT / "runs/first_baseline_schema08_context_v2_256_aspp_uncertain_softloss/model.pt"
LOSS_NAMES = ("loss", "foreground_loss", "gate_loss", "morphology_loss", "joint_loss", "skeleton_loss")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    out = args.out or args.suite_root / "gated_context_unet_v1"
    out.mkdir(parents=True, exist_ok=True)
    gpus = parse_gpus(args.gpus)
    preflight(args)
    started = time.time()
    failures: list[str] = []

    if not args.skip_smoke:
        try:
            run_fold(0, gpus[0], args, out, epochs=2, smoke=True)
            report = smoke_report(args, out)
            write_json(out / "smoke" / "structural_report.json", report)
            if not report["structurally_valid"]:
                failures.append("two-epoch smoke failed structural validity checks")
        except Exception as exc:
            failures.append(f"smoke/fold_0: {exc}")
        write_execution_report(out, started, failures, "smoke")
        if failures or args.smoke_only:
            return 1 if failures else 0

    failures += run_folds(gpus, args, out)
    if not failures:
        result = subprocess.run([
            str(PYTHON), "scripts/analyze_gated_context_evaluation.py",
            "--suite-root", str(args.suite_root), "--experiment-root", str(out),
            "--manifest", str(args.manifest), "--fold-manifest", str(args.fold_manifest),
            "--image-root", str(args.image_root), "--annotation-root", str(args.annotation_root),
        ], cwd=ROOT)
        if result.returncode:
            failures.append(f"analysis exited {result.returncode}")
    write_execution_report(out, started, failures, "complete")
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", type=Path, default=SUITE)
    p.add_argument("--out", type=Path)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--manifest", type=Path, default=ROOT / "data_manifests/real_annotations.csv")
    p.add_argument("--fold-manifest", type=Path, default=ROOT / "data_manifests/real_annotation_folds_v1.csv")
    p.add_argument("--image-root", type=Path, default=Path("/ssd/STED_dataset/data"))
    p.add_argument("--annotation-root", type=Path, default=Path("/cephfs/mhuang/STED_dataset/manual_annotations"))
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def parse_gpus(raw: str) -> list[str]:
    gpus = [value.strip() for value in raw.split(",") if value.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain unique physical GPU IDs")
    return gpus


def preflight(args: argparse.Namespace) -> None:
    required = [INITIAL, args.manifest, args.fold_manifest]
    required += [args.suite_root / "crops_256" / f"fold_{fold}/real_crop_manifest.csv" for fold in range(5)]
    required += [args.suite_root / "runs/b1_real_only" / f"fold_{fold}/model_best.pt" for fold in range(5)]
    required += [
        args.suite_root / "uncertain_ignore_nested_cv/evaluations/S_s0.05" / f"fold_{fold}/test/summary.json"
        for fold in range(5)
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing controlled-suite inputs: {missing}")
    rows = list(csv.DictReader(args.fold_manifest.open(encoding="utf-8")))
    if len({row["sample_id"] for row in rows if row["partition"] == "test"}) != 20:
        raise ValueError("fold manifest does not contain exactly 20 unique outer-test images")


def run_folds(gpus: list[str], args: argparse.Namespace, out: Path) -> list[str]:
    failures: list[str] = []
    lock = Lock()
    queues = [list(range(5))[index::len(gpus)] for index in range(len(gpus))]

    def worker(gpu: str, folds: list[int]) -> None:
        for fold in folds:
            try:
                run_fold(fold, gpu, args, out, epochs=19, smoke=False)
            except Exception as exc:
                with lock:
                    failures.append(f"fold_{fold}: {exc}")

    threads = [Thread(target=worker, args=(gpu, queue)) for gpu, queue in zip(gpus, queues)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def run_fold(
    fold: int, gpu: str, args: argparse.Namespace, out: Path, *, epochs: int, smoke: bool,
) -> None:
    base = out / "smoke" if smoke else out
    run_dir = base / "runs" / f"fold_{fold}"
    validation_dir = base / "evaluations" / f"fold_{fold}" / "validation"
    test_dir = base / "evaluations" / f"fold_{fold}" / "test"
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
        execute(command, gpu, log_dir / "train.stdout.log", log_dir / "train.stderr.log", args.dry_run)
    checkpoint = run_dir / "model_best.pt"
    if not args.dry_run and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not (validation_dir / "summary.json").exists():
        execute(
            evaluation_command(args, checkpoint, validation_dir, fold, "validation"), gpu,
            log_dir / "validation.stdout.log", log_dir / "validation.stderr.log", args.dry_run,
        )
    if not smoke and not (test_dir / "summary.json").exists():
        execute(
            evaluation_command(args, checkpoint, test_dir, fold, "test"), gpu,
            log_dir / "test.stdout.log", log_dir / "test.stderr.log", args.dry_run,
        )
    write_json(log_dir / "job_status.json", {
        "status": "dry_run" if args.dry_run else "completed",
        "fold": fold, "physical_gpu": gpu, "logical_device": "cuda:0",
        "gpu_name": gpu_name(gpu), "runtime_seconds": time.time() - started,
        "checkpoint": str(checkpoint), "run_directory": str(run_dir),
        "validation_directory": str(validation_dir),
        "test_directory": None if smoke else str(test_dir),
    })


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
        "--lambda-foreground", "1.0", "--lambda-gate", "0.5", "--lambda-morphology", "1.0",
        "--lambda-joint", "1.0", "--lambda-skeleton", "0.5",
        "--lambda-uncertainty", "0", "--lambda-uncertain-fibrous", "0",
        "--lambda-clump-anti-fibrous", "0", "--lambda-clump-anti-skeleton", "0",
        "--uncertain-skeleton-policy", "ignore", "--best-metric", "macro_image_fibrous_dice",
        "--best-mode", "max", "--save-best-checkpoint", "--save-checkpoint-every", "1",
        "--augmentation", "--cache-samples", "--qa-panel-count", "8",
    ]


def evaluation_command(
    args: argparse.Namespace, checkpoint: Path, out: Path, fold: int, partition: str,
) -> list[str]:
    return [
        str(PYTHON), "scripts/evaluate_real_annotation_batch.py", "--manifest", str(args.manifest),
        "--image-root", str(args.image_root), "--annotation-root", str(args.annotation_root),
        "--fold-manifest", str(args.fold_manifest), "--outer-fold", str(fold), "--partition", partition,
        "--checkpoint", str(checkpoint), "--out", str(out), "--patch-size", "256", "--tile-overlap", "64",
        "--batch-size", "8", "--device", "cuda:0", "--uncertainty-gating", "none",
    ]


def smoke_report(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    run = out / "smoke/runs/fold_0"
    evaluation = out / "smoke/evaluations/fold_0/validation"
    rows = list(csv.DictReader((run / "training_log.csv").open(encoding="utf-8")))
    losses = {
        name: [float(row["train_" + name] if name != "loss" else row["train_loss"]) for row in rows]
        for name in LOSS_NAMES
    }
    finite = all(math.isfinite(value) for values in losses.values() for value in values)
    plausible = len(rows) == 2 and losses["loss"][-1] <= losses["loss"][0] * 1.05
    diagnostics, summaries = [], []
    for path in sorted(evaluation.glob("*/predictions.npz")):
        with np.load(path) as data:
            pred = {name: data[name] for name in data.files}
        diagnostics.append(collapse_diagnostic(path.parent.name, pred))
        for name in (
            "final_background_probability", "final_fibrous_probability", "final_clump_probability",
            "final_uncertain_probability", "final_skeleton_probability",
        ):
            summaries.append(probability_summary(path.parent.name, name, pred[name]))
    collapsed = any(row["collapsed"] for row in diagnostics)
    panel = write_smoke_panel(args, evaluation, out / "smoke/validation_gated_qa_panel.png")
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    init = metadata["initial_checkpoint"]
    valid_init = not init["missing_backbone_keys"] and len(init["loaded_keys"]) == 34
    return {
        "structurally_valid": finite and plausible and not collapsed and valid_init,
        "finite_losses": finite, "plausible_total_loss": plausible, "losses": losses,
        "collapse_diagnostics": diagnostics, "probability_summaries": summaries,
        "qa_panel": str(panel), "initialization_report": init,
    }


def collapse_diagnostic(sample_id: str, pred: dict[str, np.ndarray]) -> dict[str, Any]:
    semantic = pred["semantic_class_map"]
    fractions = {str(value): float(np.mean(semantic == value)) for value in (0, 1, 3, 255)}
    q, g = pred["foreground_probability"], pred["quantifiability_probability"]
    collapsed = (
        max(fractions.values()) > 0.99 or float(q.mean()) < 0.01 or float(q.mean()) > 0.99
        or float(g.mean()) < 0.01 or float(g.mean()) > 0.99
    )
    return {
        "sample_id": sample_id, "class_fractions": fractions,
        "q_mean": float(q.mean()), "g_mean": float(g.mean()),
        "q_variance": float(q.var()), "g_variance": float(g.var()), "collapsed": collapsed,
    }


def probability_summary(sample_id: str, name: str, values: np.ndarray) -> dict[str, Any]:
    q = np.quantile(values, [0, 0.25, 0.5, 0.75, 0.95, 1])
    return {
        "sample_id": sample_id, "probability": name, "mean": float(values.mean()),
        "min": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]), "max": float(q[5]),
    }


def write_smoke_panel(args: argparse.Namespace, evaluation: Path, path: Path) -> Path:
    from fibras.annotations import build_real_annotation_sample
    from scripts.analyze_b1_uncertain_ignore import load_records
    from scripts.evaluate_real_pilot_baseline import draw_tile, gray_rgb, heatmap, real_mask_rgb

    pred_path = sorted(evaluation.glob("*/predictions.npz"))[0]
    record = next(
        row for row in load_records(args.manifest, args.fold_manifest, args.image_root, args.annotation_root)
        if row.sample_id == pred_path.parent.name
    )
    sample = build_real_annotation_sample(record.image_path, record.snakes_path, record.labels_path)
    raw = sample["image_uint8"] if sample["image_uint8"] is not None else sample["image_float"]
    with np.load(pred_path) as data:
        pred = {name: data[name] for name in data.files}
    tiles = [
        draw_tile(gray_rgb(raw), "raw STED"),
        draw_tile(real_mask_rgb(sample["real_semantic_mask"]), "expert labels"),
        draw_tile(real_mask_rgb(pred["semantic_class_map"]), "final semantic"),
        draw_tile(heatmap(pred["foreground_probability"]), "foreground q"),
        draw_tile(heatmap(pred["quantifiability_probability"]), "quantifiability g"),
        draw_tile(heatmap(pred["conditional_fibrous_probability"]), "conditional fibre"),
        draw_tile(heatmap(pred["final_fibrous_probability"]), "final fibre"),
        draw_tile(heatmap(pred["final_uncertain_probability"]), "final uncertain"),
        draw_tile(heatmap(pred["final_skeleton_probability"]), "final skeleton"),
    ]
    panel = Image.new("RGB", (3 * tiles[0].width, 3 * tiles[0].height), "white")
    for index, tile in enumerate(tiles):
        panel.paste(tile, ((index % 3) * tile.width, (index // 3) * tile.height))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def execute(command: list[str], gpu: str, stdout: Path, stderr: Path, dry_run: bool) -> None:
    mapping = f"physical_gpu={gpu} CUDA_VISIBLE_DEVICES={gpu} logical_device=cuda:0 gpu_name={gpu_name(gpu)}"
    print(f"{mapping} {' '.join(command)}", flush=True)
    if dry_run:
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
        out.write(mapping + "\n"); out.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}; see {stdout} and {stderr}")


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted((run_dir / "checkpoints").glob("model_epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def gpu_name(gpu: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", "-i", gpu, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_execution_report(out: Path, started: float, failures: list[str], phase: str) -> None:
    statuses = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(out.glob("**/job_status.json"))
    ]
    write_json(out / "execution_status.json", {
        "phase": phase, "runtime_seconds": time.time() - started,
        "failures": failures, "jobs": statuses,
    })
    (out / "execution_report.md").write_text(
        "# Gated ContextUNet controlled evaluation\n\n"
        f"- Phase: {phase}\n- Runtime: {time.time() - started:.1f} seconds\n"
        f"- Recorded jobs: {len(statuses)}\n- Failures: {len(failures)}\n\n"
        "## Failures\n\n" + ("\n".join(f"- {failure}" for failure in failures) or "- None") + "\n",
        encoding="utf-8",
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
