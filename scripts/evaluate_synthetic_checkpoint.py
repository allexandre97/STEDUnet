#!/usr/bin/env python
"""Evaluate a schema-0.8 checkpoint on the fixed synthetic validation split."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.training.schema08_baseline import (
    BaselineConfig,
    Schema08PatchDataset,
    build_model,
    choose_device,
    device_report,
    evaluate_model,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-patches-per-sample", type=int, default=1)
    args = parser.parse_args()
    device = choose_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    raw_config = payload.get("config", {})
    allowed = {field.name for field in fields(BaselineConfig)}
    config = BaselineConfig(**{key: value for key, value in raw_config.items() if key in allowed})
    dataset = Schema08PatchDataset(
        args.manifest, "validation", config.patch_size,
        args.validation_patches_per_sample, config.foreground_fraction,
        config.seed + 100_000, cache_samples=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = build_model(config).to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    metrics = evaluate_model(
        model, loader, device, config.lambda_skeleton, config.skeleton_pos_weight,
        config.skeleton_threshold, config.lambda_uncertainty, config.uncertainty_loss,
        config.uncertainty_focal_gamma, config.uncertainty_pos_weight,
        config.uncertain_skeleton_policy, config.lambda_clump_anti_fibrous,
        config.lambda_clump_anti_skeleton,
    )
    result = {
        "checkpoint": str(args.checkpoint), "manifest": str(args.manifest),
        "split": "validation", "sample_count": len(dataset.rows),
        "patch_count": len(dataset), "validation_patches_per_sample": args.validation_patches_per_sample,
        "seed": config.seed, "dataset_seed": config.seed + 100_000,
        "patch_size": config.patch_size, "device_report": device_report(device),
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
