#!/usr/bin/env python
"""Run the fixed two-GPU real-STED cross-validation queue."""

from __future__ import annotations

import argparse
import csv
import json
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
    "b1_real_only": ("0:100", True, None),
    "b2_mixed_80_20": ("80:20", True, None),
    "b3_mixed_no_uncertainty": ("80:20", False, None),
    "b4_mixed_targeted": (
        "80:20",
        True,
        "fibrous_positive=2,clump_positive=3,uncertain_positive=2,"
        "dense_or_fibrous_clump_boundary=4,background_hard_negative=1,uniform_random=0",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must name at least one physical GPU")
    jobs = [(name, fold) for name in EXPERIMENTS for fold in range(5)]
    queues = [jobs[index::len(gpus)] for index in range(len(gpus))]
    threads = [Thread(target=worker, args=(gpu, queue, args), daemon=False) for gpu, queue in zip(gpus, queues)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = aggregate(args.root)
    return 1 if failures else 0


def worker(gpu: str, jobs: list[tuple[str, int]], args: argparse.Namespace) -> None:
    for experiment, fold in jobs:
        try:
            run_job(args.root, gpu, experiment, fold, args.dry_run)
        except Exception as exc:
            print(f"FAILED {experiment}/fold_{fold} on physical GPU {gpu}: {exc}", file=sys.stderr, flush=True)


def run_job(root: Path, gpu: str, experiment: str, fold: int, dry_run: bool) -> None:
    ratio, uncertainty, weights = EXPERIMENTS[experiment]
    run_dir = root / "runs" / experiment / f"fold_{fold}"
    eval_dir = root / "evaluations" / experiment / f"fold_{fold}"
    log_dir = root / "logs" / experiment / f"fold_{fold}"
    if not dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    train = training_command(root, run_dir, experiment, fold, ratio, uncertainty, weights)
    if not (run_dir / "checkpoint_summary.json").exists():
        resume = latest_checkpoint(run_dir)
        if resume:
            init_index = train.index("--init-checkpoint")
            del train[init_index:init_index + 2]
            train += ["--resume-checkpoint", str(resume)]
        execute(train, gpu, log_dir / "train.stdout.log", log_dir / "train.stderr.log", dry_run)
    best = run_dir / "model_best.pt"
    if not dry_run and not best.exists():
        raise FileNotFoundError(f"missing best checkpoint: {best}")
    if not (eval_dir / "summary.json").exists():
        evaluate = evaluation_command(root, best, eval_dir, fold)
        execute(evaluate, gpu, log_dir / "eval.stdout.log", log_dir / "eval.stderr.log", dry_run)


def training_command(
    root: Path, run_dir: Path, experiment: str, fold: int, ratio: str,
    uncertainty: bool, weights: str | None,
) -> list[str]:
    command = [str(PYTHON), "scripts/train_first_baseline.py"]
    if ratio != "0:100":
        command += ["--manifest", str(SYNTHETIC)]
    command += [
        "--real-manifest", str(root / "crops_256" / f"fold_{fold}" / "real_crop_manifest.csv"),
        "--synthetic-real-ratio", ratio, "--init-checkpoint", str(INITIAL), "--out", str(run_dir),
        "--epochs", "19", "--stage1-epochs", "4", "--stage1-learning-rate", "1e-4",
        "--learning-rate", "2e-5", "--early-stop-patience", "4", "--early-stop-warmup-epochs", "4",
        "--batch-size", "5", "--patch-size", "256", "--patches-per-sample", "4",
        "--validation-patches-per-sample", "1", "--patches-per-epoch", "256", "--num-workers", "0",
        "--seed", "123", "--device", "cuda:0", "--model-variant", "context_unet",
        "--context-module", "aspp", "--aspp-dilations", "1,2,4,8", "--lambda-uncertainty", "0.05" if uncertainty else "0",
        "--uncertainty-loss", "bce", "--uncertain-skeleton-policy", "ignore",
        "--lambda-clump-anti-fibrous", "0.05", "--lambda-clump-anti-skeleton", "0.05",
        "--best-metric", "macro_image_fibrous_dice", "--best-mode", "max", "--save-best-checkpoint",
        "--save-checkpoint-every", "1", "--augmentation", "--cache-samples", "--qa-panel-count", "8",
    ]
    if uncertainty:
        command.append("--enable-uncertainty-head")
    if weights:
        command += ["--real-sampling-weights", weights]
    return command


def evaluation_command(root: Path, checkpoint: Path, out: Path, fold: int) -> list[str]:
    return [
        str(PYTHON), "scripts/evaluate_real_annotation_batch.py", "--manifest", "data_manifests/real_annotations.csv",
        "--image-root", "/ssd/STED_dataset/data", "--annotation-root", "/cephfs/mhuang/STED_dataset/manual_annotations",
        "--fold-manifest", "data_manifests/real_annotation_folds_v1.csv", "--outer-fold", str(fold),
        "--partition", "test", "--checkpoint", str(checkpoint), "--out", str(out),
        "--patch-size", "256", "--tile-overlap", "64", "--batch-size", "8", "--device", "cuda:0",
        "--uncertainty-gating", "none",
    ]


def execute(command: list[str], gpu: str, stdout: Path, stderr: Path, dry_run: bool) -> None:
    print(f"CUDA_VISIBLE_DEVICES={gpu} {' '.join(command)}", flush=True)
    if dry_run:
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}; see {stdout} and {stderr}")


def latest_checkpoint(run_dir: Path) -> Path | None:
    paths = sorted((run_dir / "checkpoints").glob("model_epoch_*.pt"))
    return paths[-1] if paths else None


def aggregate(root: Path) -> int:
    failures = 0
    for experiment in EXPERIMENTS:
        summaries = []
        for fold in range(5):
            path = root / "evaluations" / experiment / f"fold_{fold}" / "summary.json"
            if path.exists():
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                failures += 1
        if len(summaries) != 5:
            continue
        rows = [row for summary in summaries for row in summary["rows"]]
        numeric = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
        macro = {key: sum(float(row[key]) for row in rows if isinstance(row.get(key), (int, float))) /
                 sum(isinstance(row.get(key), (int, float)) for row in rows) for key in numeric}
        out = root / "evaluations" / experiment
        (out / "oof_summary.json").write_text(json.dumps({"rows": rows, "macro": macro}, indent=2, sort_keys=True) + "\n")
        with (out / "oof_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
