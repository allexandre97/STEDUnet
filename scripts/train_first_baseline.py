#!/usr/bin/env python
"""Train the first minimal schema-0.8 real-compatible STED baseline."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.training.schema08_baseline import (
    BaselineConfig,
    DeviceSelectionError,
    train_baseline,
    wandb_config_payload,
    wandb_epoch_metrics,
)


class WandbTracker:
    def __init__(self, wandb: Any, args: argparse.Namespace, mode: str) -> None:
        self.wandb = wandb
        self.args = args
        self.mode = mode
        self.run = None

    def start(self, config: BaselineConfig, metadata: dict[str, Any]) -> None:
        self.run = self.wandb.init(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            name=self.args.wandb_run_name,
            group=self.args.wandb_group,
            tags=parse_wandb_tags(self.args.wandb_tags),
            mode=self.mode,
            config=wandb_config_payload(config, metadata),
            dir=str(Path(config.out).resolve()),
        )

    def log_epoch(
        self,
        row: dict[str, float | int],
        val_metrics: dict[str, Any],
        real_val_metrics: dict[str, Any] | None = None,
    ) -> None:
        self.wandb.log(wandb_epoch_metrics(row, val_metrics, real_val_metrics), step=int(row["epoch"]))

    def log_qa_panels(self, panel_paths: list[Path]) -> None:
        if not self.args.wandb_log_panels or not panel_paths:
            return
        images = [self.wandb.Image(str(path), caption=path.name) for path in panel_paths[:8]]
        self.wandb.log({"validation/qa_panels": images})

    def log_checkpoint(self, checkpoint_path: Path) -> None:
        if not self.args.wandb_log_checkpoints:
            return
        artifact = self.wandb.Artifact("first_baseline_schema08_model", type="model")
        artifact.add_file(str(checkpoint_path))
        self.wandb.log_artifact(artifact)

    def finish(self) -> None:
        if self.run is not None:
            self.wandb.finish()


def parse_wandb_tags(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def wandb_mode(args: argparse.Namespace) -> str:
    if args.wandb_mode is not None:
        return args.wandb_mode
    return "online" if args.wandb else "disabled"


def create_wandb_tracker(args: argparse.Namespace) -> WandbTracker | None:
    mode = wandb_mode(args)
    if not args.wandb or mode == "disabled":
        return None
    prepare_wandb_dirs(args, mode)
    try:
        wandb = importlib.import_module("wandb")
    except ImportError as exc:
        raise RuntimeError(
            "--wandb was requested, but the wandb package is not installed. "
            "Install it in the training environment with `conda install -c conda-forge wandb` "
            "or `pip install wandb`."
        ) from exc
    return WandbTracker(wandb, args, mode)


def prepare_wandb_dirs(args: argparse.Namespace, mode: str) -> None:
    out = Path(args.out).resolve()
    os.environ.setdefault("WANDB_DIR", str(out / "wandb"))
    if mode == "offline":
        os.environ.setdefault("WANDB_CACHE_DIR", str(out / ".wandb_cache"))
        os.environ.setdefault("WANDB_CONFIG_DIR", str(out / ".wandb_config"))
        os.environ.setdefault("WANDB_DATA_DIR", str(out / ".wandb_data"))
    for name in ["WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR"]:
        if os.environ.get(name):
            Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--patch-size", "--patch_size", dest="patch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lambda-skeleton", "--lambda_skeleton", dest="lambda_skeleton", type=float, default=0.5)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--patches-per-sample", "--patches_per_sample", dest="patches_per_sample", type=int, default=4)
    parser.add_argument(
        "--validation-patches-per-sample",
        "--validation_patches_per_sample",
        dest="validation_patches_per_sample",
        type=int,
        default=2,
    )
    parser.add_argument("--foreground-fraction", "--foreground_fraction", dest="foreground_fraction", type=float, default=0.75)
    parser.add_argument("--skeleton-pos-weight", "--skeleton_pos_weight", dest="skeleton_pos_weight", type=float, default=8.0)
    parser.add_argument("--skeleton-threshold", "--skeleton_threshold", dest="skeleton_threshold", type=float, default=0.5)
    parser.add_argument("--cache-samples", "--cache_samples", dest="cache_samples", action="store_true")
    parser.add_argument("--limit-train-samples", "--limit_train_samples", dest="limit_train_samples", type=int)
    parser.add_argument("--limit-val-samples", "--limit_val_samples", dest="limit_val_samples", type=int)
    parser.add_argument("--patches-per-epoch", "--patches_per_epoch", dest="patches_per_epoch", type=int)
    parser.add_argument("--real-manifest", "--real_manifest", dest="real_manifest")
    parser.add_argument("--model-variant", "--model_variant", dest="model_variant", choices=["small_unet", "context_unet"], default="small_unet")
    parser.add_argument("--context-module", "--context_module", dest="context_module", choices=["none", "aspp"], default="none")
    parser.add_argument("--aspp-dilations", "--aspp_dilations", dest="aspp_dilations", default="1,2,4,8")
    parser.add_argument(
        "--init-checkpoint",
        "--init_checkpoint",
        dest="init_checkpoint",
        help="Optional checkpoint to warm-start from before training.",
    )
    parser.add_argument(
        "--synthetic-real-ratio",
        "--synthetic_real_ratio",
        dest="synthetic_real_ratio",
        help="Optional per-batch synthetic:real patch ratio, for example 80:20.",
    )
    parser.add_argument(
        "--enable-uncertainty-head",
        "--enable_uncertainty_head",
        dest="enable_uncertainty_head",
        action="store_true",
    )
    parser.add_argument("--lambda-uncertainty", "--lambda_uncertainty", dest="lambda_uncertainty", type=float, default=0.0)
    parser.add_argument(
        "--uncertainty-loss",
        "--uncertainty_loss",
        dest="uncertainty_loss",
        choices=["bce", "balanced_bce", "focal_bce"],
        default="bce",
    )
    parser.add_argument(
        "--uncertainty-focal-gamma",
        "--uncertainty_focal_gamma",
        dest="uncertainty_focal_gamma",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--uncertainty-pos-weight",
        "--uncertainty_pos_weight",
        dest="uncertainty_pos_weight",
        default="auto",
        help="Uncertainty positive weight: auto, none, or a numeric value.",
    )
    parser.add_argument(
        "--uncertain-skeleton-policy",
        "--uncertain_skeleton_policy",
        dest="uncertain_skeleton_policy",
        choices=["ignore", "suppress"],
        default="ignore",
    )
    parser.add_argument(
        "--lambda-clump-anti-fibrous",
        "--lambda_clump_anti_fibrous",
        dest="lambda_clump_anti_fibrous",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--lambda-clump-anti-skeleton",
        "--lambda_clump_anti_skeleton",
        dest="lambda_clump_anti_skeleton",
        type=float,
        default=0.0,
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--save-best-checkpoint", "--save_best_checkpoint", dest="save_best_checkpoint", action="store_true")
    parser.add_argument("--best-metric", "--best_metric", dest="best_metric", default="loss")
    parser.add_argument("--best-mode", "--best_mode", dest="best_mode", choices=["min", "max", "auto"], default="auto")
    parser.add_argument("--early-stop-patience", "--early_stop_patience", dest="early_stop_patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", "--early_stop_min_delta", dest="early_stop_min_delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-warmup-epochs",
        "--early_stop_warmup_epochs",
        dest="early_stop_warmup_epochs",
        type=int,
        default=0,
    )
    parser.add_argument("--save-checkpoint-every", "--save_checkpoint_every", dest="save_checkpoint_every", type=int, default=0)
    parser.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="sted-unet")
    parser.add_argument("--wandb-entity", "--wandb_entity", dest="wandb_entity")
    parser.add_argument("--wandb-run-name", "--wandb_run_name", dest="wandb_run_name")
    parser.add_argument("--wandb-group", "--wandb_group", dest="wandb_group")
    parser.add_argument("--wandb-tags", "--wandb_tags", dest="wandb_tags")
    parser.add_argument("--wandb-mode", "--wandb_mode", dest="wandb_mode", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-panels", "--wandb_log_panels", dest="wandb_log_panels", action="store_true")
    parser.add_argument(
        "--wandb-log-checkpoints",
        "--wandb_log_checkpoints",
        dest="wandb_log_checkpoints",
        action="store_true",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> BaselineConfig:
    return BaselineConfig(
        manifest=args.manifest,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        lambda_skeleton=args.lambda_skeleton,
        learning_rate=args.learning_rate,
        patches_per_sample=args.patches_per_sample,
        validation_patches_per_sample=args.validation_patches_per_sample,
        foreground_fraction=args.foreground_fraction,
        skeleton_pos_weight=args.skeleton_pos_weight,
        skeleton_threshold=args.skeleton_threshold,
        cache_samples=args.cache_samples,
        limit_train_samples=args.limit_train_samples,
        limit_val_samples=args.limit_val_samples,
        patches_per_epoch=args.patches_per_epoch,
        real_manifest=args.real_manifest,
        init_checkpoint=args.init_checkpoint,
        synthetic_real_ratio=args.synthetic_real_ratio,
        model_variant=args.model_variant,
        context_module=args.context_module,
        aspp_dilations=args.aspp_dilations,
        save_best_checkpoint=args.save_best_checkpoint,
        best_metric=args.best_metric,
        best_mode=args.best_mode,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_warmup_epochs=args.early_stop_warmup_epochs,
        save_checkpoint_every=args.save_checkpoint_every,
        enable_uncertainty_head=args.enable_uncertainty_head,
        lambda_uncertainty=args.lambda_uncertainty,
        uncertainty_loss=args.uncertainty_loss,
        uncertainty_focal_gamma=args.uncertainty_focal_gamma,
        uncertainty_pos_weight=args.uncertainty_pos_weight,
        uncertain_skeleton_policy=args.uncertain_skeleton_policy,
        lambda_clump_anti_fibrous=args.lambda_clump_anti_fibrous,
        lambda_clump_anti_skeleton=args.lambda_clump_anti_skeleton,
        command_line_args=vars(args),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        tracker = create_wandb_tracker(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        result = train_baseline(
            config_from_args(args),
            tracker=tracker,
        )
    except DeviceSelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"device: {result['device']}")
    print(f"validation metrics: {result['validation_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
