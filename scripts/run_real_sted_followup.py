#!/usr/bin/env python
"""Run the longer real-dominant real-STED follow-up queue."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/lmb/home/alexandrebg/miniconda3/envs/fibras/bin/python")
SYNTHETIC = Path("data_manifests/training_v2_uncertain_schema08.csv")
INITIAL = Path("runs/first_baseline_schema08_context_v2_256_aspp_uncertain_softloss/model.pt")
EXPERIMENTS = {
    "b1_long_uncertainty": {"ratio": "0:100", "uncertainty": True, "batch_size": 5, "batches": 256},
    "r100_no_uncertainty": {"ratio": "0:100", "uncertainty": False, "batch_size": 10, "batches": 128},
    "r80s20_no_uncertainty": {"ratio": "20:80", "uncertainty": False, "batch_size": 10, "batches": 128},
    "r50s50_no_uncertainty": {"ratio": "50:50", "uncertainty": False, "batch_size": 10, "batches": 128},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must name at least one physical GPU")
    jobs = [(name, fold) for name in EXPERIMENTS for fold in range(5)]
    queues = [jobs[index::len(gpus)] for index in range(len(gpus))]
    failures: list[str] = []
    threads = [Thread(target=worker, args=(gpu, queue, args, failures), daemon=False) for gpu, queue in zip(gpus, queues)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    return 1 if failures else 0


def worker(gpu: str, jobs: list[tuple[str, int]], args: argparse.Namespace, failures: list[str]) -> None:
    for experiment, fold in jobs:
        try:
            run_job(gpu, experiment, fold, args)
        except Exception as exc:
            message = f"FAILED {experiment}/fold_{fold} on physical GPU {gpu}: {exc}"
            failures.append(message); print(message, file=sys.stderr, flush=True)


def run_job(gpu: str, experiment: str, fold: int, args: argparse.Namespace) -> None:
    spec = EXPERIMENTS[experiment]
    run = args.out_root / "runs" / experiment / f"fold_{fold}"
    evaluation = args.out_root / "evaluations" / experiment / f"fold_{fold}"
    retention = args.out_root / "synthetic_retention" / experiment / f"fold_{fold}.json"
    logs = args.out_root / "logs" / experiment / f"fold_{fold}"
    if not args.dry_run: logs.mkdir(parents=True, exist_ok=True)
    train = training_command(args.suite_root, run, fold, spec)
    if not (run / "checkpoint_summary.json").exists():
        resume = latest_checkpoint(run)
        if resume:
            index = train.index("--init-checkpoint"); del train[index:index + 2]
            train += ["--resume-checkpoint", str(resume)]
        execute(train, gpu, logs / "train.stdout.log", logs / "train.stderr.log", args.dry_run)
    best = run / "model_best.pt"
    if not args.dry_run and not best.exists(): raise FileNotFoundError(best)
    if not (evaluation / "summary.json").exists():
        execute(real_evaluation_command(best, evaluation, fold), gpu, logs / "real_eval.stdout.log", logs / "real_eval.stderr.log", args.dry_run)
    if not retention.exists():
        execute(retention_command(best, retention), gpu, logs / "retention.stdout.log", logs / "retention.stderr.log", args.dry_run)


def training_command(suite: Path, run: Path, fold: int, spec: dict) -> list[str]:
    command = [str(PYTHON), "scripts/train_first_baseline.py"]
    if spec["ratio"] != "0:100": command += ["--manifest", str(SYNTHETIC)]
    command += [
        "--real-manifest", str(suite / "crops_256" / f"fold_{fold}" / "real_crop_manifest.csv"),
        "--synthetic-real-ratio", spec["ratio"], "--init-checkpoint", str(INITIAL), "--out", str(run),
        "--epochs", "44", "--stage1-epochs", "4", "--stage1-learning-rate", "1e-4", "--learning-rate", "2e-5",
        "--early-stop-patience", "6", "--early-stop-warmup-epochs", "8", "--batch-size", str(spec["batch_size"]),
        "--patch-size", "256", "--patches-per-sample", "4", "--validation-patches-per-sample", "1",
        "--patches-per-epoch", str(spec["batches"]), "--num-workers", "0", "--seed", "123", "--device", "cuda:0",
        "--model-variant", "context_unet", "--context-module", "aspp", "--aspp-dilations", "1,2,4,8",
        "--lambda-uncertainty", "0.05" if spec["uncertainty"] else "0", "--uncertainty-loss", "bce",
        "--uncertain-skeleton-policy", "ignore", "--lambda-clump-anti-fibrous", "0.05",
        "--lambda-clump-anti-skeleton", "0.05", "--best-metric", "macro_image_fibrous_dice", "--best-mode", "max",
        "--save-best-checkpoint", "--save-checkpoint-every", "1", "--augmentation", "--cache-samples", "--qa-panel-count", "8",
    ]
    if spec["uncertainty"]: command.append("--enable-uncertainty-head")
    return command


def real_evaluation_command(checkpoint: Path, out: Path, fold: int) -> list[str]:
    return [str(PYTHON), "scripts/evaluate_real_annotation_batch.py", "--manifest", "data_manifests/real_annotations.csv",
            "--image-root", "/ssd/STED_dataset/data", "--annotation-root", "/cephfs/mhuang/STED_dataset/manual_annotations",
            "--fold-manifest", "data_manifests/real_annotation_folds_v1.csv", "--outer-fold", str(fold), "--partition", "test",
            "--checkpoint", str(checkpoint), "--out", str(out), "--patch-size", "256", "--tile-overlap", "64",
            "--batch-size", "8", "--device", "cuda:0", "--uncertainty-gating", "none"]


def retention_command(checkpoint: Path, out: Path) -> list[str]:
    return [str(PYTHON), "scripts/evaluate_synthetic_checkpoint.py", "--checkpoint", str(checkpoint),
            "--manifest", str(SYNTHETIC), "--out", str(out), "--device", "cuda:0", "--batch-size", "8"]


def execute(command: list[str], gpu: str, stdout: Path, stderr: Path, dry_run: bool) -> None:
    mapping = f"physical_gpu={gpu} CUDA_VISIBLE_DEVICES={gpu} logical_device=cuda:0"
    print(f"{mapping} {' '.join(command)}", flush=True)
    if dry_run: return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
        out.write(mapping + "\n"); out.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
    if result.returncode: raise RuntimeError(f"exit {result.returncode}; see {stdout} and {stderr}")


def latest_checkpoint(run: Path) -> Path | None:
    paths = sorted((run / "checkpoints").glob("model_epoch_*.pt"))
    return paths[-1] if paths else None


if __name__ == "__main__":
    raise SystemExit(main())
