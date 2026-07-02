"""Minimal PyTorch baseline for schema-0.8 real-compatible STED targets."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


IGNORE_INDEX = -100


class DeviceSelectionError(RuntimeError):
    """Raised when a requested training device cannot be used."""


@dataclass
class BaselineConfig:
    manifest: str
    out: str
    epochs: int = 2
    batch_size: int = 4
    patch_size: int = 128
    device: str = "auto"
    num_workers: int = 0
    seed: int = 123
    lambda_skeleton: float = 0.5
    learning_rate: float = 1e-3
    patches_per_sample: int = 4
    validation_patches_per_sample: int = 2
    foreground_fraction: float = 0.75
    skeleton_pos_weight: float = 8.0
    skeleton_threshold: float = 0.5
    cache_samples: bool = False
    limit_train_samples: int | None = None
    limit_val_samples: int | None = None
    patches_per_epoch: int | None = None
    qa_panel_count: int = 8
    command_line_args: dict[str, Any] | None = None


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_path(path: str, manifest_path: Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    candidate = manifest_path.parent / p
    return candidate if candidate.exists() else p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Schema08PatchDataset(Dataset):
    """Deterministic 128x128 patch loader for real-compatible schema-0.8 arrays."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        patch_size: int = 128,
        patches_per_sample: int = 4,
        foreground_fraction: float = 0.75,
        seed: int = 123,
        cache_samples: bool = False,
        limit_samples: int | None = None,
        patches_per_epoch: int | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.rows = [row for row in read_manifest(manifest_path) if row["split"] == split]
        if limit_samples is not None:
            self.rows = self.rows[: int(limit_samples)]
        if not self.rows:
            raise ValueError(f"no rows for split {split} in {manifest_path}")
        self.split = split
        self.patch_size = int(patch_size)
        self.patches_per_sample = int(patches_per_sample)
        self.patches_per_epoch = int(patches_per_epoch) if patches_per_epoch is not None else None
        self.foreground_fraction = float(foreground_fraction)
        self.seed = int(seed)
        self.cache_samples = bool(cache_samples)
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        if self.cache_samples:
            for row in self.rows:
                self._load_arrays(row)

    def __len__(self) -> int:
        if self.patches_per_epoch is not None:
            return self.patches_per_epoch
        return len(self.rows) * self.patches_per_sample

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_index = index % len(self.rows)
        patch_index = index // len(self.rows)
        row = self.rows[sample_index]
        arrays = self._load_arrays(row)
        image = arrays["render_uint8"].astype(np.float32) / 255.0
        semantic_raw = arrays["real_compatible_semantic_mask"]
        skeleton = arrays["real_compatible_skeleton_mask"].astype(np.float32)
        uncertain = arrays["real_compatible_uncertain_ignore_mask"].astype(bool) | (semantic_raw == 255)
        semantic = np.zeros_like(semantic_raw, dtype=np.int64)
        semantic[semantic_raw == 1] = 1
        semantic[semantic_raw == 3] = 2
        semantic[uncertain] = IGNORE_INDEX
        y0, x0 = self._patch_origin(row["sample_id"], patch_index, semantic_raw, uncertain)
        ps = self.patch_size
        valid = ~uncertain[y0 : y0 + ps, x0 : x0 + ps]
        return {
            "image": torch.from_numpy(image[y0 : y0 + ps, x0 : x0 + ps][None, ...].copy()),
            "semantic": torch.from_numpy(semantic[y0 : y0 + ps, x0 : x0 + ps].copy()),
            "skeleton": torch.from_numpy(skeleton[y0 : y0 + ps, x0 : x0 + ps][None, ...].copy()),
            "valid": torch.from_numpy(valid[None, ...].astype(np.float32).copy()),
            "sample_id": row["sample_id"],
        }

    def _load_arrays(self, row: dict[str, str]) -> dict[str, np.ndarray]:
        path = resolve_path(row["npz_path"], self.manifest_path)
        cache_key = path.as_posix()
        if cache_key in self._cache:
            return self._cache[cache_key]
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        if self.cache_samples:
            self._cache[cache_key] = arrays
        return arrays

    def _patch_origin(
        self,
        sample_id: str,
        patch_index: int,
        semantic_raw: np.ndarray,
        uncertain: np.ndarray,
    ) -> tuple[int, int]:
        rng = np.random.default_rng(self._item_seed(sample_id, patch_index))
        ps = self.patch_size
        h, w = semantic_raw.shape
        if h < ps or w < ps:
            raise ValueError(f"patch_size {ps} exceeds image shape {(h, w)}")
        foreground = ((semantic_raw == 1) | (semantic_raw == 3)) & ~uncertain
        if foreground.any() and rng.random() < self.foreground_fraction:
            ys, xs = np.nonzero(foreground)
            choice = int(rng.integers(0, len(ys)))
            y = int(np.clip(ys[choice] - ps // 2, 0, h - ps))
            x = int(np.clip(xs[choice] - ps // 2, 0, w - ps))
            return y, x
        return int(rng.integers(0, h - ps + 1)), int(rng.integers(0, w - ps + 1))

    def _item_seed(self, sample_id: str, patch_index: int) -> int:
        digest = hashlib.sha256(f"{self.seed}:{self.split}:{sample_id}:{patch_index}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little", signed=False)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.semantic_head = nn.Conv2d(c, 3, 1)
        self.skeleton_head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return {
            "semantic_logits": self.semantic_head(d1),
            "skeleton_logits": self.skeleton_head(d1),
        }


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_skeleton: float = 0.5,
    skeleton_pos_weight: float = 8.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    semantic_loss = F.cross_entropy(outputs["semantic_logits"], batch["semantic"], ignore_index=IGNORE_INDEX)
    pos_weight = torch.tensor([skeleton_pos_weight], device=outputs["skeleton_logits"].device)
    skeleton_loss_map = F.binary_cross_entropy_with_logits(
        outputs["skeleton_logits"],
        batch["skeleton"],
        pos_weight=pos_weight.view(1, 1, 1, 1),
        reduction="none",
    )
    valid = batch["valid"]
    skeleton_loss = (skeleton_loss_map * valid).sum() / valid.sum().clamp_min(1.0)
    total = semantic_loss + float(lambda_skeleton) * skeleton_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "semantic_loss": float(semantic_loss.detach().cpu()),
        "skeleton_loss": float(skeleton_loss.detach().cpu()),
    }


def dice_score(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float | None:
    pred = pred & valid
    target = target & valid
    denom = int(pred.sum() + target.sum())
    if denom == 0:
        return None
    return float(2 * int((pred & target).sum()) / denom)


def target_positive_dice(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float | None:
    if int((target & valid).sum()) == 0:
        return None
    return dice_score(pred, target, valid)


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    lambda_skeleton: float,
    skeleton_pos_weight: float,
    skeleton_threshold: float = 0.5,
) -> dict[str, float | str]:
    model.eval()
    losses: list[dict[str, float]] = []
    fibrous_dice: list[float] = []
    clump_dice: list[float] = []
    skeleton_dice: list[float] = []
    skeleton_precision: list[float] = []
    skeleton_recall: list[float] = []
    pred_inside_pred_fibrous: list[float] = []
    pred_inside_true_fibrous: list[float] = []
    threshold_metrics = {threshold: {"dice": [], "precision": [], "recall": []} for threshold in [0.25, 0.5, 0.75]}
    clump_target_pixels = 0
    with torch.no_grad():
        for batch in loader:
            batch = tensor_batch_to_device(batch, device)
            outputs = model(batch["image"])
            _, loss_parts = compute_loss(outputs, batch, lambda_skeleton, skeleton_pos_weight)
            losses.append(loss_parts)
            pred_sem = outputs["semantic_logits"].argmax(dim=1)
            target_sem = batch["semantic"]
            valid = target_sem != IGNORE_INDEX
            skeleton_prob = torch.sigmoid(outputs["skeleton_logits"][:, 0])
            pred_skel = skeleton_prob > skeleton_threshold
            true_skel = batch["skeleton"][:, 0] > 0.5
            for b in range(pred_sem.shape[0]):
                fib = target_positive_dice(pred_sem[b] == 1, target_sem[b] == 1, valid[b])
                if fib is not None:
                    fibrous_dice.append(fib)
                clump_target_pixels += int(((target_sem[b] == 2) & valid[b]).sum())
                clump = target_positive_dice(pred_sem[b] == 2, target_sem[b] == 2, valid[b])
                if clump is not None:
                    clump_dice.append(clump)
                skel = target_positive_dice(pred_skel[b], true_skel[b], valid[b])
                if skel is not None:
                    skeleton_dice.append(skel)
                pred_count = int((pred_skel[b] & valid[b]).sum())
                true_count = int((true_skel[b] & valid[b]).sum())
                tp = int((pred_skel[b] & true_skel[b] & valid[b]).sum())
                if pred_count:
                    skeleton_precision.append(float(tp / pred_count))
                    pred_inside_pred_fibrous.append(float(((pred_skel[b] & (pred_sem[b] == 1) & valid[b]).sum() / pred_count).cpu()))
                    pred_inside_true_fibrous.append(float(((pred_skel[b] & (target_sem[b] == 1) & valid[b]).sum() / pred_count).cpu()))
                if true_count:
                    skeleton_recall.append(float(tp / true_count))
                for threshold, metrics in threshold_metrics.items():
                    pred_at_threshold = skeleton_prob[b] > threshold
                    threshold_skel = target_positive_dice(pred_at_threshold, true_skel[b], valid[b])
                    if threshold_skel is not None:
                        metrics["dice"].append(threshold_skel)
                    pred_at_count = int((pred_at_threshold & valid[b]).sum())
                    if pred_at_count:
                        metrics["precision"].append(float(int((pred_at_threshold & true_skel[b] & valid[b]).sum()) / pred_at_count))
                    if true_count:
                        metrics["recall"].append(float(int((pred_at_threshold & true_skel[b] & valid[b]).sum()) / true_count))
    return {
        "loss": mean_metric(losses, "loss"),
        "semantic_loss": mean_metric(losses, "semantic_loss"),
        "skeleton_loss": mean_metric(losses, "skeleton_loss"),
        "fibrous_dice": mean_or_na(fibrous_dice),
        "clump_dice": mean_or_na(clump_dice),
        "clump_target_pixels": clump_target_pixels,
        "skeleton_dice": mean_or_na(skeleton_dice),
        "skeleton_precision": mean_or_na(skeleton_precision),
        "skeleton_recall": mean_or_na(skeleton_recall),
        "skeleton_threshold": float(skeleton_threshold),
        "skeleton_metrics_by_threshold": {
            str(threshold): {name: mean_or_na(values) for name, values in metrics.items()}
            for threshold, metrics in threshold_metrics.items()
        },
        "predicted_skeleton_inside_predicted_fibrous_fraction": mean_or_na(pred_inside_pred_fibrous),
        "predicted_skeleton_inside_true_fibrous_fraction": mean_or_na(pred_inside_true_fibrous),
    }


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else float("nan")


def mean_or_na(values: list[float]) -> float | str:
    return float(np.mean(values)) if values else "not_applicable"


def tensor_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(v) else v for k, v in batch.items()}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda":
        validate_cuda_device(device)
    return device


def validate_cuda_device(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise DeviceSelectionError(
            "CUDA was requested but is not available to PyTorch. "
            f"torch.version.cuda={torch.version.cuda!r}; device_count={torch.cuda.device_count()}. "
            "Check the NVIDIA driver, CUDA_VISIBLE_DEVICES, and that the conda environment has a CUDA-enabled PyTorch build."
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise DeviceSelectionError(
            f"CUDA device index {index} was requested, but only {torch.cuda.device_count()} device(s) are visible."
        )
    torch.cuda.set_device(index)
    try:
        torch.empty(1, device=torch.device("cuda", index))
        torch.cuda.synchronize(index)
    except RuntimeError as exc:
        raise DeviceSelectionError(f"CUDA device cuda:{index} is visible but failed a test allocation: {exc}") from exc


def device_report(device: torch.device) -> dict[str, Any]:
    report = {
        "selected_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_devices": [],
    }
    if torch.cuda.is_available():
        report["cuda_current_device"] = torch.cuda.current_device()
        report["cuda_devices"] = [
            {"index": i, "name": torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())
        ]
    return report


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log_status(message: str) -> None:
    print(message, flush=True)


def train_baseline(config: BaselineConfig, tracker: Any | None = None) -> dict[str, Any]:
    seed_everything(config.seed)
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    log_status(f"training output: {out}")
    log_status(f"using device: {device}")
    report = device_report(device)
    if report["cuda_available"]:
        log_status(f"CUDA devices visible: {report['cuda_devices']}")
    else:
        log_status("CUDA is not available to PyTorch; using CPU")
    save_json(out / "config.json", {**asdict(config), "resolved_device": str(device)})
    log_status("loading training split")
    train_ds = Schema08PatchDataset(
        Path(config.manifest),
        "train",
        config.patch_size,
        config.patches_per_sample,
        config.foreground_fraction,
        config.seed,
        config.cache_samples,
        config.limit_train_samples,
        config.patches_per_epoch,
    )
    log_status(f"training samples: {len(train_ds.rows)}; patches per epoch: {len(train_ds)}")
    log_status("loading validation split")
    val_ds = Schema08PatchDataset(
        Path(config.manifest),
        "validation",
        config.patch_size,
        config.validation_patches_per_sample,
        config.foreground_fraction,
        config.seed + 100_000,
        config.cache_samples,
        config.limit_val_samples,
    )
    log_status(f"validation samples: {len(val_ds.rows)}; validation patches: {len(val_ds)}")
    log_status("collecting run metadata and target pixel counts")
    metadata = run_metadata(config, train_ds, val_ds, device)
    save_json(out / "run_metadata.json", metadata)
    log_status(f"wrote metadata: {out / 'run_metadata.json'}")
    if tracker is not None:
        log_status("initializing W&B tracking")
        tracker.start(config, metadata)
        seed_everything(config.seed)
        log_status("W&B tracking initialized")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    log_rows: list[dict[str, float | int]] = []
    try:
        for epoch in range(1, config.epochs + 1):
            log_status(f"epoch {epoch}/{config.epochs}: training")
            model.train()
            running: list[dict[str, float]] = []
            for batch in train_loader:
                batch = tensor_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch["image"])
                loss, parts = compute_loss(outputs, batch, config.lambda_skeleton, config.skeleton_pos_weight)
                loss.backward()
                optimizer.step()
                running.append(parts)
            log_status(f"epoch {epoch}/{config.epochs}: validating")
            val_metrics = evaluate_model(
                model,
                val_loader,
                device,
                config.lambda_skeleton,
                config.skeleton_pos_weight,
                config.skeleton_threshold,
            )
            row = {
                "epoch": epoch,
                "train_loss": mean_metric(running, "loss"),
                "train_semantic_loss": mean_metric(running, "semantic_loss"),
                "train_skeleton_loss": mean_metric(running, "skeleton_loss"),
                "validation_loss": float(val_metrics["loss"]),
                "validation_semantic_loss": float(val_metrics["semantic_loss"]),
                "validation_skeleton_loss": float(val_metrics["skeleton_loss"]),
            }
            log_rows.append(row)
            if tracker is not None:
                tracker.log_epoch(row, val_metrics)
            print(
                f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
                f"validation_loss={row['validation_loss']:.4f}",
                flush=True,
            )
        write_log_csv(out / "training_log.csv", log_rows)
        log_status(f"wrote training log: {out / 'training_log.csv'}")
        log_status("running final validation")
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            config.lambda_skeleton,
            config.skeleton_pos_weight,
            config.skeleton_threshold,
        )
        save_json(out / "validation_metrics.json", val_metrics)
        log_status(f"wrote validation metrics: {out / 'validation_metrics.json'}")
        checkpoint_path = out / "model.pt"
        torch.save(
            {"model_state_dict": model.state_dict(), "config": asdict(config), "validation_metrics": val_metrics},
            checkpoint_path,
        )
        log_status(f"wrote checkpoint: {checkpoint_path}")
        if tracker is not None:
            tracker.log_checkpoint(checkpoint_path)
        panel_paths = write_qa_panels(model, val_ds, device, out / "qa_panels", config.qa_panel_count, config.skeleton_threshold)
        log_status(f"wrote QA panels: {out / 'qa_panels'}")
        if tracker is not None:
            tracker.log_qa_panels(panel_paths)
        return {"log": log_rows, "validation_metrics": val_metrics, "device": str(device)}
    finally:
        if tracker is not None:
            tracker.finish()


def run_metadata(
    config: BaselineConfig,
    train_ds: Schema08PatchDataset,
    val_ds: Schema08PatchDataset,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_report": device_report(device),
        "manifest": config.manifest,
        "output_directory": config.out,
        "command_line_args": config.command_line_args or asdict(config),
        "git_commit": git_commit(),
        "split_counts": {"train": len(train_ds.rows), "validation": len(val_ds.rows)},
        "schema_versions": sorted({row.get("schema_version", "unknown") or "unknown" for row in train_ds.rows + val_ds.rows}),
        "generator_versions": sorted({row.get("generator_version", "unknown") or "unknown" for row in train_ds.rows + val_ds.rows}),
        "dataset_paths": dataset_path_info(train_ds.rows + val_ds.rows),
        "target_class_pixel_counts": {
            "train": target_class_pixel_counts(train_ds),
            "validation": target_class_pixel_counts(val_ds),
        },
    }


def dataset_path_info(rows: list[dict[str, str]]) -> dict[str, Any]:
    npz_paths = [row.get("npz_path", "") for row in rows if row.get("npz_path")]
    metadata_paths = [row.get("metadata_path", row.get("json_path", "")) for row in rows if row.get("metadata_path") or row.get("json_path")]
    return {
        "npz_count": len(npz_paths),
        "metadata_count": len(metadata_paths),
        "npz_roots": sorted({Path(path).parts[0] for path in npz_paths if Path(path).parts}),
        "metadata_roots": sorted({Path(path).parts[0] for path in metadata_paths if Path(path).parts}),
        "source_synthetic_splits": sorted({row.get("source_synthetic_split", "unknown") or "unknown" for row in rows}),
    }


def model_config_payload() -> dict[str, Any]:
    return {
        "architecture": "SmallUNet",
        "in_channels": 1,
        "base_channels": 16,
        "semantic_classes": 3,
        "heads": ["semantic", "skeleton"],
    }


def wandb_config_payload(config: BaselineConfig, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "training": asdict(config),
        "metadata": metadata,
        "model": model_config_payload(),
    }


def wandb_epoch_metrics(row: dict[str, float | int], val_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "epoch": row["epoch"],
        "train/loss": row["train_loss"],
        "train/semantic_loss": row["train_semantic_loss"],
        "train/skeleton_loss": row["train_skeleton_loss"],
        "validation/loss": loggable_metric(val_metrics["loss"]),
        "validation/semantic_loss": loggable_metric(val_metrics["semantic_loss"]),
        "validation/skeleton_loss": loggable_metric(val_metrics["skeleton_loss"]),
        "validation/fibrous_dice": loggable_metric(val_metrics["fibrous_dice"]),
        "validation/clump_dice": loggable_metric(val_metrics["clump_dice"]),
        "validation/clump_target_pixels": loggable_metric(val_metrics["clump_target_pixels"]),
        "validation/skeleton_dice": loggable_metric(val_metrics["skeleton_dice"]),
        "validation/skeleton_precision": loggable_metric(val_metrics["skeleton_precision"]),
        "validation/skeleton_recall": loggable_metric(val_metrics["skeleton_recall"]),
        "validation/predicted_skeleton_inside_predicted_fibrous_fraction": loggable_metric(
            val_metrics["predicted_skeleton_inside_predicted_fibrous_fraction"]
        ),
        "validation/predicted_skeleton_inside_true_fibrous_fraction": loggable_metric(
            val_metrics["predicted_skeleton_inside_true_fibrous_fraction"]
        ),
    }
    for threshold, threshold_metrics in val_metrics.get("skeleton_metrics_by_threshold", {}).items():
        for name, value in threshold_metrics.items():
            metrics[f"validation/skeleton_threshold_{threshold}/{name}"] = loggable_metric(value)
    return metrics


def loggable_metric(value: Any) -> Any:
    return None if value == "not_applicable" else value


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "not_available"
    return result.stdout.strip()


def target_class_pixel_counts(dataset: Schema08PatchDataset) -> dict[str, int]:
    counts = {"background": 0, "fibrous_tau": 0, "clump": 0, "uncertain_ignore": 0, "skeleton": 0}
    for row in dataset.rows:
        arrays = dataset._load_arrays(row)
        semantic = arrays["real_compatible_semantic_mask"]
        counts["background"] += int(np.count_nonzero(semantic == 0))
        counts["fibrous_tau"] += int(np.count_nonzero(semantic == 1))
        counts["clump"] += int(np.count_nonzero(semantic == 3))
        counts["uncertain_ignore"] += int(np.count_nonzero((semantic == 255) | (arrays["real_compatible_uncertain_ignore_mask"] > 0)))
        counts["skeleton"] += int(np.count_nonzero(arrays["real_compatible_skeleton_mask"]))
    return counts


def write_log_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 1] = (0, 220, 80)
    rgb[mask == 2] = (255, 160, 0)
    rgb[mask == IGNORE_INDEX] = (140, 140, 140)
    return rgb


def gray_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(arr[..., None], 3, axis=2)


def draw_tile(arr: np.ndarray, title: str) -> Image.Image:
    im = Image.fromarray(arr.astype(np.uint8), "RGB")
    canvas = Image.new("RGB", (arr.shape[1], arr.shape[0] + 18), "white")
    canvas.paste(im, (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), title, fill=(0, 0, 0))
    return canvas


def write_qa_panels(
    model: nn.Module,
    dataset: Schema08PatchDataset,
    device: torch.device,
    out_dir: Path,
    count: int,
    skeleton_threshold: float = 0.5,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    paths: list[Path] = []
    for i in range(min(count, len(dataset))):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
        pred_sem = outputs["semantic_logits"].argmax(dim=1)[0].cpu().numpy().astype(np.int64)
        pred_skel = (torch.sigmoid(outputs["skeleton_logits"][0, 0]).cpu().numpy() > skeleton_threshold)
        true_sem = sample["semantic"].numpy().astype(np.int64)
        true_skel = sample["skeleton"][0].numpy() > 0.5
        raw = sample["image"][0].numpy()
        overlay = gray_rgb(raw).astype(np.float32)
        overlay[pred_sem == 1] = overlay[pred_sem == 1] * 0.45 + np.array([0, 220, 80]) * 0.55
        overlay[pred_sem == 2] = overlay[pred_sem == 2] * 0.45 + np.array([255, 160, 0]) * 0.55
        overlay[pred_skel] = np.array([255, 0, 255])
        tiles = [
            draw_tile(gray_rgb(raw), "input"),
            draw_tile(mask_rgb(true_sem), "true semantic"),
            draw_tile(mask_rgb(pred_sem), "pred semantic"),
            draw_tile(np.repeat((true_skel * 255).astype(np.uint8)[..., None], 3, axis=2), "true skeleton"),
            draw_tile(np.repeat((pred_skel * 255).astype(np.uint8)[..., None], 3, axis=2), "pred skeleton"),
            draw_tile(np.clip(overlay, 0, 255).astype(np.uint8), "pred overlay"),
        ]
        panel = Image.new("RGB", (3 * tiles[0].width, 2 * tiles[0].height), "white")
        for j, tile in enumerate(tiles):
            panel.paste(tile, ((j % 3) * tile.width, (j // 3) * tile.height))
        path = out_dir / f"validation_prediction_{i:02d}.png"
        panel.save(path)
        paths.append(path)
    return paths
