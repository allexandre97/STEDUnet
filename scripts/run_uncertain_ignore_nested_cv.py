#!/usr/bin/env python
"""Run nested-CV suppression and stratified rejection experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, Thread
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fibras.annotations import build_real_annotation_sample
from scripts.analyze_b1_uncertain_ignore import ImageRecord, load_probabilities, load_records
from scripts.uncertain_ignore_nested_metrics import (
    complete_image_metrics,
    public_metrics,
    select_h_operating_point,
    select_s_candidate,
    selection_image_metrics,
    summarize_images,
    summarize_selection,
)


PYTHON = Path("/lmb/home/alexandrebg/miniconda3/envs/fibras/bin/python")
INITIAL = Path("runs/first_baseline_schema08_context_v2_256_aspp_uncertain_softloss/model.pt")
S_WEIGHTS = (0.01, 0.03, 0.05)
H_WEIGHTS = (0.02, 0.05)
THRESHOLDS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class Job:
    family: str
    fold: int
    suppression: float = 0.0
    rejection: float = 0.05

    @property
    def name(self) -> str:
        weights = [f"s{self.suppression:.2f}"] if self.suppression else []
        weights += [f"h{self.rejection:.2f}"] if self.family in {"H", "SH"} else []
        return f"{self.family}_{'_'.join(weights)}"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = time.time()
    gpus = parse_gpus(args.gpus)
    out = args.root / "uncertain_ignore_nested_cv"
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.manifest, args.fold_manifest, args.image_root, args.annotation_root)
    samples = load_samples(records)
    preflight(args.root, out, records)

    base_jobs = [Job("S", fold, suppression=weight) for fold in range(5) for weight in S_WEIGHTS]
    base_jobs += [Job("H", fold, rejection=weight) for fold in range(5) for weight in H_WEIGHTS]
    failures = run_jobs(base_jobs, gpus, args, out)
    if args.dry_run:
        write_execution_report(out, started, failures, {}, [])
        return 1 if failures else 0
    if failures:
        write_execution_report(out, started, failures, {}, [])
        return 1

    selections = select_folds(args.root, out, records, samples)
    write_json(out / "inner_validation_selections.json", selections)
    combined_jobs = [
        Job("SH", fold, selections[str(fold)]["S"]["lambda_uncertain_fibrous"], selections[str(fold)]["H"]["lambda_rejection"])
        for fold in range(5)
        if selections[str(fold)]["S"]["selection"] == "S" and selections[str(fold)]["H"]["selection"] == "H"
    ]
    failures += run_jobs(combined_jobs, gpus, args, out)
    if failures:
        write_execution_report(out, started, failures, selections, combined_jobs)
        return 1

    test_evaluations = selected_test_evaluations(selections, combined_jobs, out)
    failures += run_evaluations(test_evaluations, gpus, args, out)
    if failures:
        write_execution_report(out, started, failures, selections, combined_jobs)
        return 1
    metric_rows = final_metrics(args.root, out, records, samples, selections, combined_jobs)
    write_metric_artifacts(out, metric_rows)
    write_execution_report(out, started, failures, selections, combined_jobs)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/ssd/STED_experiments/controlled_real_sted_v1"))
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--manifest", type=Path, default=ROOT / "data_manifests/real_annotations.csv")
    p.add_argument("--fold-manifest", type=Path, default=ROOT / "data_manifests/real_annotation_folds_v1.csv")
    p.add_argument("--image-root", type=Path, default=Path("/ssd/STED_dataset/data"))
    p.add_argument("--annotation-root", type=Path, default=Path("/cephfs/mhuang/STED_dataset/manual_annotations"))
    p.add_argument("--dry-run", action="store_true")
    return p


def parse_gpus(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must name at least one physical GPU")
    return gpus


def load_samples(records: list[ImageRecord]) -> dict[str, dict[str, Any]]:
    samples = {}
    for record in records:
        if record.sample_id not in samples and record.partition in {"validation", "test"}:
            samples[record.sample_id] = build_real_annotation_sample(
                record.image_path, record.snakes_path, record.labels_path
            )
    return samples


def preflight(root: Path, out: Path, records: list[ImageRecord]) -> None:
    required = [root / "runs/b1_real_only" / f"fold_{fold}/model_best.pt" for fold in range(5)]
    required += [INITIAL]
    required += [u0_prediction(root, out, record) for record in records if record.partition in {"validation", "test"}]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required completed-suite artifacts: {missing[:10]}")


def run_jobs(jobs: list[Job], gpus: list[str], args: argparse.Namespace, out: Path) -> list[str]:
    failures: list[str] = []
    lock = Lock()
    queues = [jobs[index::len(gpus)] for index in range(len(gpus))]

    def worker(gpu: str, queue: list[Job]) -> None:
        for job in queue:
            try:
                run_job(job, gpu, args, out)
            except Exception as exc:
                with lock:
                    failures.append(f"{job.name}/fold_{job.fold}: {exc}")

    threads = [Thread(target=worker, args=(gpu, queue)) for gpu, queue in zip(gpus, queues)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def run_job(job: Job, gpu: str, args: argparse.Namespace, out: Path) -> None:
    run_dir = job_run_dir(out, job)
    eval_dir = job_eval_dir(out, job, "validation")
    log_dir = out / "logs" / job.name / f"fold_{job.fold}"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not (run_dir / "checkpoint_summary.json").exists():
        command = training_command(args.root, run_dir, job)
        resume = latest_checkpoint(run_dir)
        if resume:
            index = command.index("--init-checkpoint")
            del command[index:index + 2]
            command += ["--resume-checkpoint", str(resume)]
        execute(command, gpu, log_dir / "train.stdout.log", log_dir / "train.stderr.log", args.dry_run)
    checkpoint = run_dir / "model_best.pt"
    if not args.dry_run and not checkpoint.exists():
        raise FileNotFoundError(f"missing {checkpoint}")
    if not (eval_dir / "summary.json").exists():
        execute(
            evaluation_command(args, checkpoint, eval_dir, job.fold, "validation"), gpu,
            log_dir / "validation.stdout.log", log_dir / "validation.stderr.log", args.dry_run,
        )
    write_json(log_dir / "job_status.json", {
        "status": "dry_run" if args.dry_run else "completed",
        "family": job.family,
        "fold": job.fold,
        "suppression": job.suppression,
        "rejection": job.rejection,
        "physical_gpu": gpu,
        "logical_device": "cuda:0",
        "gpu_name": gpu_name(gpu),
        "runtime_seconds": time.time() - started,
        "checkpoint": str(checkpoint),
        "validation_artifacts": str(eval_dir),
    })


def training_command(root: Path, run_dir: Path, job: Job) -> list[str]:
    command = [
        str(PYTHON), "scripts/train_first_baseline.py",
        "--real-manifest", str(root / "crops_256" / f"fold_{job.fold}" / "real_crop_manifest.csv"),
        "--synthetic-real-ratio", "0:100", "--init-checkpoint", str(INITIAL), "--out", str(run_dir),
        "--epochs", "19", "--stage1-epochs", "4", "--stage1-learning-rate", "1e-4",
        "--learning-rate", "2e-5", "--early-stop-patience", "4", "--early-stop-warmup-epochs", "4",
        "--batch-size", "5", "--patch-size", "256", "--patches-per-sample", "4",
        "--validation-patches-per-sample", "1", "--patches-per-epoch", "256", "--num-workers", "0",
        "--seed", "123", "--device", "cuda:0", "--model-variant", "context_unet",
        "--context-module", "aspp", "--aspp-dilations", "1,2,4,8", "--enable-uncertainty-head",
        "--uncertain-skeleton-policy", "ignore", "--lambda-clump-anti-fibrous", "0.05",
        "--lambda-clump-anti-skeleton", "0.05", "--best-metric", "macro_image_fibrous_dice",
        "--best-mode", "max", "--save-best-checkpoint", "--save-checkpoint-every", "1",
        "--augmentation", "--cache-samples", "--qa-panel-count", "8",
        "--lambda-uncertain-fibrous", str(job.suppression), "--uncertain-fibrous-tau", "0.5",
    ]
    if job.family == "S":
        command += ["--lambda-uncertainty", "0.05", "--uncertainty-loss", "bce"]
    else:
        command += [
            "--lambda-uncertainty", str(job.rejection), "--uncertainty-loss", "stratified_bce",
            "--rejection-near-distance", "20",
        ]
    return command


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


def execute(command: list[str], gpu: str, stdout: Path, stderr: Path, dry_run: bool) -> None:
    print(f"CUDA_VISIBLE_DEVICES={gpu} {' '.join(command)}", flush=True)
    if dry_run:
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}; see {stdout} and {stderr}")


def select_folds(
    root: Path, out: Path, records: list[ImageRecord], samples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selections = {}
    for fold in range(5):
        validation = fold_records(records, fold, "validation")
        baseline = selection_summary_for(validation, samples, {
            record.sample_id: u0_prediction(root, out, record) for record in validation
        })
        s_rows = []
        for weight in S_WEIGHTS:
            job = Job("S", fold, suppression=weight)
            summary = selection_summary_for(validation, samples, prediction_paths(job_eval_dir(out, job, "validation"), validation))
            s_rows.append({"lambda_uncertain_fibrous": weight, **summary, "checkpoint": str(job_run_dir(out, job) / "model_best.pt")})
        selected_s = select_s_candidate(baseline, s_rows)
        h_rows = []
        for weight in H_WEIGHTS:
            job = Job("H", fold, rejection=weight)
            paths = prediction_paths(job_eval_dir(out, job, "validation"), validation)
            loaded = {sample_id: load_probabilities(path) for sample_id, path in paths.items()}
            for threshold in THRESHOLDS:
                summary = selection_summary_for(validation, samples, loaded, float(threshold))
                h_rows.append({
                    "lambda_rejection": weight, "rejection_threshold": float(threshold), **summary,
                    "checkpoint": str(job_run_dir(out, job) / "model_best.pt"),
                })
        selected_h = select_h_operating_point(baseline, h_rows)
        selections[str(fold)] = {"U0_validation": baseline, "S": selected_s, "H": selected_h}
    return selections


def selection_summary_for(
    records: list[ImageRecord], samples: dict[str, dict[str, Any]],
    predictions: dict[str, Path | dict[str, np.ndarray]], threshold: float | None = None,
) -> dict[str, Any]:
    rows = []
    for record in records:
        pred = predictions[record.sample_id]
        pred = load_probabilities(pred) if isinstance(pred, Path) else pred
        sample = samples[record.sample_id]
        rows.append(selection_image_metrics(
            sample["real_semantic_mask"], sample["real_skeleton_mask"], pred, threshold
        ))
    return summarize_selection(rows)


def selected_test_evaluations(selections: dict[str, Any], combined: list[Job], out: Path) -> list[tuple[Path, Path, int, str]]:
    evaluations = []
    combined_folds = {job.fold: job for job in combined}
    for fold in range(5):
        for family in ("S", "H"):
            selected = selections[str(fold)][family]
            if selected["selection"] == family:
                job = selected_job(family, fold, selected)
                evaluations.append((job_run_dir(out, job) / "model_best.pt", job_eval_dir(out, job, "test"), fold, family))
        if fold in combined_folds:
            job = combined_folds[fold]
            evaluations.append((job_run_dir(out, job) / "model_best.pt", job_eval_dir(out, job, "test"), fold, "SH"))
    return evaluations


def run_evaluations(
    evaluations: list[tuple[Path, Path, int, str]], gpus: list[str], args: argparse.Namespace, out: Path,
) -> list[str]:
    failures: list[str] = []
    queues = [evaluations[index::len(gpus)] for index in range(len(gpus))]
    lock = Lock()

    def worker(gpu: str, queue: list[tuple[Path, Path, int, str]]) -> None:
        for checkpoint, eval_dir, fold, family in queue:
            if (eval_dir / "summary.json").exists():
                continue
            log_dir = out / "logs" / "selected_test" / family / f"fold_{fold}"
            log_dir.mkdir(parents=True, exist_ok=True)
            try:
                execute(evaluation_command(args, checkpoint, eval_dir, fold, "test"), gpu,
                        log_dir / "stdout.log", log_dir / "stderr.log", args.dry_run)
            except Exception as exc:
                with lock:
                    failures.append(f"test {family}/fold_{fold}: {exc}")

    threads = [Thread(target=worker, args=(gpu, queue)) for gpu, queue in zip(gpus, queues)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def final_metrics(
    root: Path, out: Path, records: list[ImageRecord], samples: dict[str, dict[str, Any]],
    selections: dict[str, Any], combined: list[Job],
) -> list[dict[str, Any]]:
    rows = []
    combined_by_fold = {job.fold: job for job in combined}
    for fold in range(5):
        test = fold_records(records, fold, "test")
        variants: list[tuple[str, dict[str, Path], float | None]] = [
            ("U0", {record.sample_id: u0_prediction(root, out, record) for record in test}, None),
        ]
        for family in ("S", "H"):
            selected = selections[str(fold)][family]
            if selected["selection"] == family:
                job = selected_job(family, fold, selected)
                threshold = selected.get("rejection_threshold") if family == "H" else None
                variants.append((family, prediction_paths(job_eval_dir(out, job, "test"), test), threshold))
            else:
                variants.append((family, variants[0][1], None))
        if fold in combined_by_fold:
            job = combined_by_fold[fold]
            variants.append(("SH", prediction_paths(job_eval_dir(out, job, "test"), test), selections[str(fold)]["H"]["rejection_threshold"]))
        for variant, paths, threshold in variants:
            for record in test:
                sample = samples[record.sample_id]
                metrics = complete_image_metrics(
                    sample["real_semantic_mask"], sample["real_skeleton_mask"],
                    load_probabilities(paths[record.sample_id]), threshold,
                )
                rows.append({"variant": variant, "outer_fold": fold, "sample_id": record.sample_id, **metrics})
    return rows


def write_metric_artifacts(out: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(out / "outer_test_per_image.csv", [
        {"variant": row["variant"], "outer_fold": row["outer_fold"], "sample_id": row["sample_id"], **public_metrics(row)}
        for row in rows
    ])
    fold_rows = []
    for variant in sorted({row["variant"] for row in rows}):
        for fold in range(5):
            selected = [row for row in rows if row["variant"] == variant and row["outer_fold"] == fold]
            if selected:
                fold_rows.append({"variant": variant, "outer_fold": fold, **summarize_images(selected)})
    write_csv(out / "outer_test_per_fold.csv", fold_rows)
    macro = {
        variant: summarize_images([row for row in rows if row["variant"] == variant])
        for variant in sorted({row["variant"] for row in rows})
    }
    write_json(out / "outer_test_macro.json", macro)


def write_execution_report(
    out: Path, started: float, failures: list[str], selections: dict[str, Any], combined: list[Job],
) -> None:
    statuses = []
    for path in sorted((out / "logs").glob("*/*/job_status.json")):
        statuses.append(json.loads(path.read_text(encoding="utf-8")))
    selection_lines = [
        f"| {fold} | {row['S'].get('selection')} | {row['S'].get('lambda_uncertain_fibrous', 'n/a')} | "
        f"{row['H'].get('selection')} | {row['H'].get('lambda_rejection', 'n/a')} | {row['H'].get('rejection_threshold', 'n/a')} |"
        for fold, row in sorted(selections.items())
    ]
    text = f"""# Uncertain-ignore nested-CV execution report

This is an execution and provenance report, not a scientific ranking or recommendation.

## Status

- Runtime: {time.time() - started:.1f} seconds
- Completed training/validation jobs: {len(statuses)}
- Failed jobs: {len(failures)}
- Combined SH folds run: {','.join(str(job.fold) for job in combined) or 'none'}

## Inner-validation selections

| Fold | S selection | Suppression weight | H selection | Rejection weight | Threshold |
|---:|---|---:|---|---:|---:|
{chr(10).join(selection_lines)}

## Failures

{chr(10).join(f'- {failure}' for failure in failures) or '- None'}

## Artifacts

- `inner_validation_selections.json`
- `outer_test_per_image.csv`
- `outer_test_per_fold.csv`
- `outer_test_macro.json`
- `runs/`, `evaluations/`, and `logs/`

GPU assignments, logical devices, GPU names, runtimes, checkpoint paths, and validation artifact paths are recorded in each `job_status.json`. Outer-test metrics are reported without model ranking.
"""
    (out / "execution_report.md").write_text(text, encoding="utf-8")
    write_json(out / "execution_status.json", {"failures": failures, "jobs": statuses, "runtime_seconds": time.time() - started})


def selected_job(family: str, fold: int, selected: dict[str, Any]) -> Job:
    return Job(
        family, fold,
        suppression=float(selected.get("lambda_uncertain_fibrous", 0.0)),
        rejection=float(selected.get("lambda_rejection", 0.05)),
    )


def job_run_dir(out: Path, job: Job) -> Path:
    return out / "runs" / job.name / f"fold_{job.fold}"


def job_eval_dir(out: Path, job: Job, partition: str) -> Path:
    return out / "evaluations" / job.name / f"fold_{job.fold}" / partition


def prediction_paths(directory: Path, records: list[ImageRecord]) -> dict[str, Path]:
    return {record.sample_id: directory / record.sample_id / "predictions.npz" for record in records}


def u0_prediction(root: Path, out: Path, record: ImageRecord) -> Path:
    if record.partition == "test":
        return root / "evaluations/b1_real_only" / f"fold_{record.outer_fold}" / record.sample_id / "predictions.npz"
    return root / "analysis/b1_uncertain_ignore/validation_predictions" / f"fold_{record.outer_fold}" / record.sample_id / "predictions.npz"


def fold_records(records: list[ImageRecord], fold: int, partition: str) -> list[ImageRecord]:
    return [record for record in records if record.outer_fold == fold and record.partition == partition]


def latest_checkpoint(run_dir: Path) -> Path | None:
    paths = sorted((run_dir / "checkpoints").glob("model_epoch_*.pt"))
    return paths[-1] if paths else None


def gpu_name(gpu: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", "-i", gpu, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
