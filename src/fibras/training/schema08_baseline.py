"""Minimal PyTorch baseline for schema-0.8 real-compatible STED targets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
BEST_MODE_AUTO = {
    "loss": "min",
    "semantic_loss": "min",
    "skeleton_loss": "min",
    "uncertainty_loss": "min",
    "fibrous_dice": "max",
    "clump_dice": "max",
    "skeleton_dice": "max",
    "skeleton_dice_0.75": "max",
    "uncertain_dice": "max",
}


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
    real_manifest: str | None = None
    synthetic_real_ratio: str | None = None
    lambda_uncertainty: float = 0.0
    enable_uncertainty_head: bool = False
    uncertainty_loss: str = "bce"
    uncertainty_focal_gamma: float = 2.0
    uncertainty_pos_weight: str = "auto"
    uncertain_skeleton_policy: str = "ignore"
    lambda_clump_anti_fibrous: float = 0.0
    lambda_clump_anti_skeleton: float = 0.0
    model_variant: str = "small_unet"
    context_module: str = "none"
    aspp_dilations: str = "1,2,4,8"
    init_checkpoint: str | None = None
    save_best_checkpoint: bool = False
    best_metric: str = "loss"
    best_mode: str = "auto"
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    early_stop_warmup_epochs: int = 0
    save_checkpoint_every: int = 0
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
            "uncertain": torch.from_numpy(uncertain[y0 : y0 + ps, x0 : x0 + ps][None, ...].astype(np.float32).copy()),
            "valid": torch.from_numpy(valid[None, ...].astype(np.float32).copy()),
            "sample_id": row["sample_id"],
            "source_kind": row.get("source_kind", "synthetic"),
        }

    def _load_arrays(self, row: dict[str, str]) -> dict[str, np.ndarray]:
        path = resolve_path(row["npz_path"], self.manifest_path)
        cache_key = path.as_posix()
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not path.exists():
            raise FileNotFoundError(
                f"{self.manifest_path}: sample {row.get('sample_id', '<unknown>')} "
                f"npz_path {row.get('npz_path')!r} resolved to missing file {path}"
            )
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


class FixedRatioBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Batch sampler for explicit synthetic:real patch ratios."""

    def __init__(
        self,
        synthetic_len: int,
        real_len: int,
        batch_size: int,
        synthetic_real_ratio: str,
        seed: int = 123,
        batches_per_epoch: int | None = None,
    ) -> None:
        if synthetic_len <= 0 or real_len <= 0:
            raise ValueError("mixed training requires non-empty synthetic and real datasets")
        self.synthetic_len = int(synthetic_len)
        self.real_len = int(real_len)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.synthetic_per_batch, self.real_per_batch = batch_counts(batch_size, synthetic_real_ratio)
        default_batches = max(
            int(np.ceil(self.synthetic_len / self.synthetic_per_batch)),
            int(np.ceil(self.real_len / self.real_per_batch)),
        )
        self.batches_per_epoch = int(batches_per_epoch) if batches_per_epoch is not None else default_batches

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        synth_order = shuffled_cycle(self.synthetic_len, rng)
        real_order = shuffled_cycle(self.real_len, rng)
        real_offset = self.synthetic_len
        for _ in range(self.batches_per_epoch):
            batch = [next(synth_order) for _ in range(self.synthetic_per_batch)]
            batch += [real_offset + next(real_order) for _ in range(self.real_per_batch)]
            rng.shuffle(batch)
            yield batch


def shuffled_cycle(length: int, rng: np.random.Generator):
    while True:
        order = np.arange(length)
        rng.shuffle(order)
        for value in order:
            yield int(value)


def batch_counts(batch_size: int, synthetic_real_ratio: str) -> tuple[int, int]:
    parts = synthetic_real_ratio.split(":")
    if len(parts) != 2:
        raise ValueError("synthetic_real_ratio must look like '80:20' or '90:10'")
    synth, real = (float(part) for part in parts)
    if synth <= 0 or real <= 0:
        raise ValueError("synthetic_real_ratio requires positive synthetic and real parts")
    real_count = int(round(batch_size * real / (synth + real)))
    real_count = min(max(real_count, 1), batch_size - 1)
    return batch_size - real_count, real_count


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
    def __init__(self, in_channels: int = 1, base_channels: int = 16, uncertainty_head: bool = False) -> None:
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
        self.uncertainty_head = nn.Conv2d(c, 1, 1) if uncertainty_head else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        outputs = {
            "semantic_logits": self.semantic_head(d1),
            "skeleton_logits": self.skeleton_head(d1),
        }
        if self.uncertainty_head is not None:
            outputs["uncertainty_logits"] = self.uncertainty_head(d1)
        return outputs


class ASPPBlock(nn.Module):
    """Small dilated-convolution context block that preserves HxW."""

    def __init__(self, channels: int, dilations: list[int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
                    nn.ReLU(inplace=True),
                )
                for dilation in dilations
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels * len(dilations), channels, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class ContextUNet(nn.Module):
    """Small U-Net with optional bottleneck context for larger crops."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        uncertainty_head: bool = False,
        context_module: str = "aspp",
        aspp_dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        if context_module == "none":
            self.context = nn.Identity()
        elif context_module == "aspp":
            self.context = ASPPBlock(c * 4, aspp_dilations or [1, 2, 4, 8])
        else:
            raise ValueError(f"unsupported context_module {context_module!r}")
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.semantic_head = nn.Conv2d(c, 3, 1)
        self.skeleton_head = nn.Conv2d(c, 1, 1)
        self.uncertainty_head = nn.Conv2d(c, 1, 1) if uncertainty_head else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.context(self.enc3(self.pool(e2)))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        outputs = {
            "semantic_logits": self.semantic_head(d1),
            "skeleton_logits": self.skeleton_head(d1),
        }
        if self.uncertainty_head is not None:
            outputs["uncertainty_logits"] = self.uncertainty_head(d1)
        return outputs


def parse_dilations(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("aspp dilations must be positive integers")
    return values


def build_model(config: BaselineConfig) -> nn.Module:
    if config.model_variant == "small_unet":
        return SmallUNet(uncertainty_head=config.enable_uncertainty_head)
    if config.model_variant == "context_unet":
        return ContextUNet(
            uncertainty_head=config.enable_uncertainty_head,
            context_module=config.context_module,
            aspp_dilations=parse_dilations(config.aspp_dilations),
        )
    raise ValueError(f"unsupported model_variant {config.model_variant!r}")


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_skeleton: float = 0.5,
    skeleton_pos_weight: float = 8.0,
    lambda_uncertainty: float = 0.0,
    uncertainty_loss_name: str = "bce",
    uncertainty_focal_gamma: float = 2.0,
    uncertainty_pos_weight: str = "auto",
    uncertain_skeleton_policy: str = "ignore",
    lambda_clump_anti_fibrous: float = 0.0,
    lambda_clump_anti_skeleton: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    semantic_loss = F.cross_entropy(outputs["semantic_logits"], batch["semantic"], ignore_index=IGNORE_INDEX)
    pos_weight = torch.tensor([skeleton_pos_weight], device=outputs["skeleton_logits"].device)
    skeleton_valid = skeleton_valid_mask(batch, uncertain_skeleton_policy)
    skeleton_target = skeleton_target_for_policy(batch, uncertain_skeleton_policy)
    skeleton_loss_map = F.binary_cross_entropy_with_logits(
        outputs["skeleton_logits"],
        skeleton_target,
        pos_weight=pos_weight.view(1, 1, 1, 1),
        reduction="none",
    )
    skeleton_loss = (skeleton_loss_map * skeleton_valid).sum() / skeleton_valid.sum().clamp_min(1.0)
    uncertainty_loss = None
    total = semantic_loss + float(lambda_skeleton) * skeleton_loss
    if "uncertainty_logits" in outputs and lambda_uncertainty > 0:
        uncertainty_loss = uncertainty_loss_value(
            outputs["uncertainty_logits"],
            batch["uncertain"],
            uncertainty_loss_name,
            uncertainty_focal_gamma,
            uncertainty_pos_weight,
        )
        total = total + float(lambda_uncertainty) * uncertainty_loss
    clump_mask = ((batch["semantic"] == 2).unsqueeze(1) & (batch["valid"] > 0))
    clump_anti_fibrous_loss = masked_mean(
        semantic_prob(outputs["semantic_logits"])[:, 1:2],
        clump_mask,
    )
    clump_anti_skeleton_loss = masked_mean(torch.sigmoid(outputs["skeleton_logits"]), clump_mask)
    if lambda_clump_anti_fibrous > 0:
        total = total + float(lambda_clump_anti_fibrous) * clump_anti_fibrous_loss
    if lambda_clump_anti_skeleton > 0:
        total = total + float(lambda_clump_anti_skeleton) * clump_anti_skeleton_loss
    parts = {
        "loss": float(total.detach().cpu()),
        "semantic_loss": float(semantic_loss.detach().cpu()),
        "skeleton_loss": float(skeleton_loss.detach().cpu()),
        "clump_anti_fibrous_loss": float(clump_anti_fibrous_loss.detach().cpu()),
        "clump_anti_skeleton_loss": float(clump_anti_skeleton_loss.detach().cpu()),
    }
    if uncertainty_loss is not None:
        parts["uncertainty_loss"] = float(uncertainty_loss.detach().cpu())
    return total, parts


def skeleton_valid_mask(batch: dict[str, torch.Tensor], policy: str) -> torch.Tensor:
    if policy == "ignore":
        return batch["valid"]
    if policy == "suppress":
        return torch.ones_like(batch["valid"])
    raise ValueError(f"unsupported uncertain_skeleton_policy {policy!r}")


def skeleton_target_for_policy(batch: dict[str, torch.Tensor], policy: str) -> torch.Tensor:
    if policy == "ignore":
        return batch["skeleton"]
    if policy == "suppress":
        uncertain = batch.get("uncertain", torch.zeros_like(batch["skeleton"])) > 0.5
        return torch.where(uncertain, torch.zeros_like(batch["skeleton"]), batch["skeleton"])
    raise ValueError(f"unsupported uncertain_skeleton_policy {policy!r}")


def semantic_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values * mask.to(values.dtype)
    return selected.sum() / mask.to(values.dtype).sum().clamp_min(1.0)


def uncertainty_loss_value(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    focal_gamma: float,
    pos_weight: str,
) -> torch.Tensor:
    weight = None if loss_name == "bce" else uncertainty_pos_weight_tensor(target, pos_weight)
    if weight is not None:
        weight = weight.to(logits.device)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=weight.view(1, 1, 1, 1) if weight is not None else None,
        reduction="none",
    )
    if loss_name in {"bce", "balanced_bce"}:
        return bce.mean()
    if loss_name == "focal_bce":
        pt = torch.exp(-bce)
        return ((1.0 - pt).pow(float(focal_gamma)) * bce).mean()
    raise ValueError(f"unsupported uncertainty_loss {loss_name!r}")


def uncertainty_pos_weight_tensor(target: torch.Tensor, pos_weight: str) -> torch.Tensor | None:
    if pos_weight == "none" or pos_weight == "":
        return None
    if pos_weight == "auto":
        pos = target.sum().clamp_min(1.0)
        neg = (target.numel() - target.sum()).clamp_min(1.0)
        return neg / pos
    return torch.tensor(float(pos_weight), device=target.device)


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
    lambda_uncertainty: float = 0.0,
    uncertainty_loss: str = "bce",
    uncertainty_focal_gamma: float = 2.0,
    uncertainty_pos_weight: str = "auto",
    uncertain_skeleton_policy: str = "ignore",
    lambda_clump_anti_fibrous: float = 0.0,
    lambda_clump_anti_skeleton: float = 0.0,
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
    uncertain_dice: list[float] = []
    uncertain_precision: list[float] = []
    uncertain_recall: list[float] = []
    uncertainty_probs = {
        "target_fibrous": [],
        "target_clump": [],
        "target_uncertain_ignore": [],
        "target_background": [],
    }
    threshold_metrics = {threshold: {"dice": [], "precision": [], "recall": []} for threshold in [0.25, 0.5, 0.75, 0.85]}
    clump_target_pixels = 0
    uncertain_target_pixels = 0
    with torch.no_grad():
        for batch in loader:
            batch = tensor_batch_to_device(batch, device)
            outputs = model(batch["image"])
            _, loss_parts = compute_loss(
                outputs,
                batch,
                lambda_skeleton,
                skeleton_pos_weight,
                lambda_uncertainty,
                uncertainty_loss,
                uncertainty_focal_gamma,
                uncertainty_pos_weight,
                uncertain_skeleton_policy,
                lambda_clump_anti_fibrous,
                lambda_clump_anti_skeleton,
            )
            losses.append(loss_parts)
            pred_sem = outputs["semantic_logits"].argmax(dim=1)
            target_sem = batch["semantic"]
            valid = target_sem != IGNORE_INDEX
            skeleton_prob = torch.sigmoid(outputs["skeleton_logits"][:, 0])
            pred_skel = skeleton_prob > skeleton_threshold
            true_skel = batch["skeleton"][:, 0] > 0.5
            uncertain_target = batch.get("uncertain", torch.zeros_like(batch["skeleton"]))
            true_uncertain = uncertain_target[:, 0] > 0.5
            pred_uncertain = (
                torch.sigmoid(outputs["uncertainty_logits"][:, 0]) > 0.5
                if "uncertainty_logits" in outputs
                else None
            )
            uncertainty_prob = (
                torch.sigmoid(outputs["uncertainty_logits"][:, 0])
                if "uncertainty_logits" in outputs
                else None
            )
            for b in range(pred_sem.shape[0]):
                fib = target_positive_dice(pred_sem[b] == 1, target_sem[b] == 1, valid[b])
                if fib is not None:
                    fibrous_dice.append(fib)
                clump_target_pixels += int(((target_sem[b] == 2) & valid[b]).sum())
                clump = target_positive_dice(pred_sem[b] == 2, target_sem[b] == 2, valid[b])
                if clump is not None:
                    clump_dice.append(clump)
                uncertain_target_pixels += int(true_uncertain[b].sum())
                if pred_uncertain is not None:
                    all_pixels = torch.ones_like(valid[b], dtype=torch.bool)
                    uncertainty = target_positive_dice(pred_uncertain[b], true_uncertain[b], all_pixels)
                    if uncertainty is not None:
                        uncertain_dice.append(uncertainty)
                    pred_uncertain_count = int(pred_uncertain[b].sum())
                    true_uncertain_count = int(true_uncertain[b].sum())
                    uncertain_tp = int((pred_uncertain[b] & true_uncertain[b]).sum())
                    if pred_uncertain_count:
                        uncertain_precision.append(float(uncertain_tp / pred_uncertain_count))
                    if true_uncertain_count:
                        uncertain_recall.append(float(uncertain_tp / true_uncertain_count))
                if uncertainty_prob is not None:
                    append_region_probabilities(uncertainty_probs, uncertainty_prob[b], target_sem[b], true_uncertain[b])
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
    result = {
        "loss": mean_metric(losses, "loss"),
        "semantic_loss": mean_metric(losses, "semantic_loss"),
        "skeleton_loss": mean_metric(losses, "skeleton_loss"),
        "clump_anti_fibrous_loss": mean_metric(losses, "clump_anti_fibrous_loss"),
        "clump_anti_skeleton_loss": mean_metric(losses, "clump_anti_skeleton_loss"),
        "fibrous_dice": mean_or_na(fibrous_dice),
        "clump_dice": mean_or_na(clump_dice),
        "clump_target_pixels": clump_target_pixels,
        "uncertain_dice": mean_or_na(uncertain_dice),
        "uncertain_precision": mean_or_na(uncertain_precision),
        "uncertain_recall": mean_or_na(uncertain_recall),
        "uncertain_target_pixels": uncertain_target_pixels,
        "uncertainty_probability_by_target_region": {
            name: probability_summary(values) for name, values in uncertainty_probs.items()
        },
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
    if any("uncertainty_loss" in row for row in losses):
        result["uncertainty_loss"] = mean_metric(losses, "uncertainty_loss")
    return result


def append_region_probabilities(
    out: dict[str, list[float]],
    prob: torch.Tensor,
    target_sem: torch.Tensor,
    true_uncertain: torch.Tensor,
) -> None:
    regions = {
        "target_fibrous": target_sem == 1,
        "target_clump": target_sem == 2,
        "target_uncertain_ignore": true_uncertain,
        "target_background": target_sem == 0,
    }
    for name, mask in regions.items():
        values = prob[mask]
        if values.numel():
            out[name].extend(values.detach().cpu().float().tolist())


def probability_summary(values: list[float]) -> dict[str, float | str]:
    if not values:
        return {"mean": "not_applicable", "p95": "not_applicable"}
    arr = np.asarray(values, dtype=np.float32)
    return {"mean": float(np.mean(arr)), "p95": float(np.percentile(arr, 95))}


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if key in row]
    return float(np.mean(values)) if values else float("nan")


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


def resolve_best_mode(metric: str, mode: str) -> str:
    if mode in {"min", "max"}:
        return mode
    if mode != "auto":
        raise ValueError(f"unsupported best_mode {mode!r}; expected min, max, or auto")
    if metric not in BEST_MODE_AUTO:
        supported = ", ".join(sorted(BEST_MODE_AUTO))
        raise ValueError(f"best_mode auto does not know metric {metric!r}; supported metrics: {supported}")
    return BEST_MODE_AUTO[metric]


def selected_validation_metric(metrics: dict[str, Any], metric: str) -> float:
    value: Any
    if metric.startswith("skeleton_dice_"):
        threshold = metric.removeprefix("skeleton_dice_")
        value = metrics.get("skeleton_metrics_by_threshold", {}).get(threshold, {}).get("dice")
    else:
        value = metrics.get(metric)
    if value is None:
        available = sorted(flat_validation_metric_names(metrics))
        raise ValueError(f"validation metric {metric!r} is unavailable; available metrics: {available}")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"validation metric {metric!r} is not numeric for this run: {value!r}")
    return float(value)


def flat_validation_metric_names(metrics: dict[str, Any]) -> set[str]:
    names = {key for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    for threshold, values in metrics.get("skeleton_metrics_by_threshold", {}).items():
        if isinstance(values, dict) and isinstance(values.get("dice"), (int, float)):
            names.add(f"skeleton_dice_{threshold}")
    return names


def best_metric_improved(current: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    improvement = best - current if mode == "min" else current - best
    return improvement > float(min_delta)


def checkpoint_payload(
    model: nn.Module,
    config: BaselineConfig,
    epoch: int,
    val_metrics: dict[str, Any],
    real_val_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "epoch": int(epoch),
        "validation_metrics": val_metrics,
        "real_validation_metrics": real_val_metrics,
    }


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    config: BaselineConfig,
    epoch: int,
    val_metrics: dict[str, Any],
    real_val_metrics: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, config, epoch, val_metrics, real_val_metrics), path)


def validation_metric_values(metrics: dict[str, Any]) -> dict[str, float]:
    values = {}
    for name in flat_validation_metric_names(metrics):
        try:
            values[name] = selected_validation_metric(metrics, name)
        except ValueError:
            pass
    return dict(sorted(values.items()))


def checkpoint_summary(
    config: BaselineConfig,
    final_epoch: int,
    stopped_early: bool,
    early_stop_reason: str | None,
    best_mode: str,
    best_epoch: int | None,
    best_value: float | None,
    final_value: float | None,
    best_checkpoint_path: Path | None,
    final_checkpoint_path: Path,
    periodic_checkpoint_paths: list[str],
    validation_values_by_epoch: dict[str, dict[str, float]],
) -> dict[str, Any]:
    summary = {
        "final_epoch": int(final_epoch),
        "stopped_early": bool(stopped_early),
        "best_metric": config.best_metric,
        "best_mode": best_mode,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "final_value": final_value,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        "final_checkpoint_path": str(final_checkpoint_path),
        "periodic_checkpoint_paths": periodic_checkpoint_paths,
        "validation_metric_values_by_epoch": validation_values_by_epoch,
        "command_line_args": config.command_line_args,
    }
    if early_stop_reason is not None:
        summary["early_stop_reason"] = early_stop_reason
    return summary


def train_baseline(config: BaselineConfig, tracker: Any | None = None) -> dict[str, Any]:
    seed_everything(config.seed)
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    best_mode = resolve_best_mode(config.best_metric, config.best_mode)
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
    real_train_ds = None
    real_val_ds = None
    if config.real_manifest is not None:
        log_status("loading real annotation crop splits")
        real_train_ds = Schema08PatchDataset(
            Path(config.real_manifest),
            "train",
            config.patch_size,
            config.patches_per_sample,
            config.foreground_fraction,
            config.seed + 200_000,
            config.cache_samples,
        )
        real_val_ds = Schema08PatchDataset(
            Path(config.real_manifest),
            "validation",
            config.patch_size,
            config.validation_patches_per_sample,
            config.foreground_fraction,
            config.seed + 300_000,
            config.cache_samples,
            config.limit_val_samples,
        )
        log_status(f"real training crops: {len(real_train_ds.rows)}; real validation crops: {len(real_val_ds.rows)}")
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
    metadata = run_metadata(config, train_ds, val_ds, device, real_train_ds, real_val_ds)
    save_json(out / "run_metadata.json", metadata)
    log_status(f"wrote metadata: {out / 'run_metadata.json'}")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = build_train_loader(config, train_ds, real_train_ds, generator, device)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    real_val_loader = None
    if real_val_ds is not None:
        real_val_loader = torch.utils.data.DataLoader(
            real_val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
    model = build_model(config).to(device)
    checkpoint_report = load_initial_checkpoint(model, config.init_checkpoint, device)
    if checkpoint_report is not None:
        log_status(
            "loaded initial checkpoint: "
            f"{checkpoint_report['path']} "
            f"(missing={len(checkpoint_report['missing_keys'])}, "
            f"unexpected={len(checkpoint_report['unexpected_keys'])})"
        )
        metadata["initial_checkpoint"] = checkpoint_report
        save_json(out / "run_metadata.json", metadata)
    if tracker is not None:
        log_status("initializing W&B tracking")
        tracker.start(config, metadata)
        seed_everything(config.seed)
        log_status("W&B tracking initialized")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    log_rows: list[dict[str, float | int]] = []
    validation_values_by_epoch: dict[str, dict[str, float]] = {}
    best_value: float | None = None
    best_epoch: int | None = None
    patience_counter = 0
    stopped_early = False
    early_stop_reason: str | None = None
    periodic_checkpoint_paths: list[str] = []
    final_epoch = 0
    final_metric_value: float | None = None
    try:
        for epoch in range(1, config.epochs + 1):
            log_status(f"epoch {epoch}/{config.epochs}: training")
            model.train()
            running: list[dict[str, float]] = []
            for batch in train_loader:
                batch = tensor_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch["image"])
                loss, parts = compute_loss(
                    outputs,
                    batch,
                    config.lambda_skeleton,
                    config.skeleton_pos_weight,
                    config.lambda_uncertainty,
                    config.uncertainty_loss,
                    config.uncertainty_focal_gamma,
                    config.uncertainty_pos_weight,
                    config.uncertain_skeleton_policy,
                    config.lambda_clump_anti_fibrous,
                    config.lambda_clump_anti_skeleton,
                )
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
                config.lambda_uncertainty,
                config.uncertainty_loss,
                config.uncertainty_focal_gamma,
                config.uncertainty_pos_weight,
                config.uncertain_skeleton_policy,
                config.lambda_clump_anti_fibrous,
                config.lambda_clump_anti_skeleton,
            )
            real_val_metrics = (
                evaluate_model(
                    model,
                    real_val_loader,
                    device,
                    config.lambda_skeleton,
                    config.skeleton_pos_weight,
                    config.skeleton_threshold,
                    config.lambda_uncertainty,
                    config.uncertainty_loss,
                    config.uncertainty_focal_gamma,
                    config.uncertainty_pos_weight,
                    config.uncertain_skeleton_policy,
                    config.lambda_clump_anti_fibrous,
                    config.lambda_clump_anti_skeleton,
                )
                if real_val_loader is not None
                else None
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
            if "uncertainty_loss" in running[0]:
                row["train_uncertainty_loss"] = mean_metric(running, "uncertainty_loss")
            row["train_clump_anti_fibrous_loss"] = mean_metric(running, "clump_anti_fibrous_loss")
            row["train_clump_anti_skeleton_loss"] = mean_metric(running, "clump_anti_skeleton_loss")
            current_value = selected_validation_metric(val_metrics, config.best_metric)
            final_epoch = epoch
            final_metric_value = current_value
            validation_values_by_epoch[str(epoch)] = validation_metric_values(val_metrics)
            improved = best_metric_improved(current_value, best_value, best_mode, config.early_stop_min_delta)
            if improved:
                best_value = current_value
                best_epoch = epoch
                patience_counter = 0
                if config.save_best_checkpoint:
                    best_checkpoint_path = out / "model_best.pt"
                    save_training_checkpoint(best_checkpoint_path, model, config, epoch, val_metrics, real_val_metrics)
                    log_status(
                        f"New best validation {config.best_metric}: {current_value:.4f} "
                        f"at epoch {epoch}; saved model_best.pt"
                    )
            else:
                patience_counter += 1
            if config.save_checkpoint_every > 0 and epoch % config.save_checkpoint_every == 0:
                periodic_path = out / "checkpoints" / f"model_epoch_{epoch:04d}.pt"
                save_training_checkpoint(periodic_path, model, config, epoch, val_metrics, real_val_metrics)
                periodic_checkpoint_paths.append(str(periodic_path))
                log_status(f"wrote periodic checkpoint: {periodic_path}")
            row["best_metric_value"] = current_value
            row["best_metric_best_value"] = best_value if best_value is not None else current_value
            row["best_epoch"] = best_epoch if best_epoch is not None else epoch
            row["early_stop_patience_counter"] = patience_counter
            row["stopped_early"] = 0
            log_rows.append(row)
            print(
                f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
                f"validation_loss={row['validation_loss']:.4f}",
                flush=True,
            )
            warmup_complete = epoch >= int(config.early_stop_warmup_epochs)
            if config.early_stop_patience > 0 and warmup_complete and patience_counter >= config.early_stop_patience:
                stopped_early = True
                row["stopped_early"] = 1
                early_stop_reason = (
                    f"no improvement in validation {config.best_metric} for "
                    f"{config.early_stop_patience} validation epochs"
                )
                log_status(
                    f"Early stopping triggered at epoch {epoch}: {early_stop_reason}. "
                    f"Best epoch: {best_epoch}."
                )
                if tracker is not None:
                    tracker.log_epoch(row, val_metrics, real_val_metrics)
                break
            if tracker is not None:
                tracker.log_epoch(row, val_metrics, real_val_metrics)
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
            config.lambda_uncertainty,
            config.uncertainty_loss,
            config.uncertainty_focal_gamma,
            config.uncertainty_pos_weight,
            config.uncertain_skeleton_policy,
            config.lambda_clump_anti_fibrous,
            config.lambda_clump_anti_skeleton,
        )
        real_val_metrics = (
            evaluate_model(
                model,
                real_val_loader,
                device,
                config.lambda_skeleton,
                config.skeleton_pos_weight,
                config.skeleton_threshold,
                config.lambda_uncertainty,
                config.uncertainty_loss,
                config.uncertainty_focal_gamma,
                config.uncertainty_pos_weight,
                config.uncertain_skeleton_policy,
                config.lambda_clump_anti_fibrous,
                config.lambda_clump_anti_skeleton,
            )
            if real_val_loader is not None
            else None
        )
        if final_epoch > 0:
            final_metric_value = selected_validation_metric(val_metrics, config.best_metric)
        save_json(out / "validation_metrics.json", val_metrics)
        log_status(f"wrote validation metrics: {out / 'validation_metrics.json'}")
        if real_val_metrics is not None:
            save_json(out / "real_validation_metrics.json", real_val_metrics)
            log_status(f"wrote real validation metrics: {out / 'real_validation_metrics.json'}")
        checkpoint_path = out / "model.pt"
        save_training_checkpoint(checkpoint_path, model, config, final_epoch, val_metrics, real_val_metrics)
        log_status(f"wrote checkpoint: {checkpoint_path}")
        save_json(
            out / "checkpoint_summary.json",
            checkpoint_summary(
                config,
                final_epoch,
                stopped_early,
                early_stop_reason,
                best_mode,
                best_epoch,
                best_value,
                final_metric_value,
                out / "model_best.pt" if config.save_best_checkpoint else None,
                checkpoint_path,
                periodic_checkpoint_paths,
                validation_values_by_epoch,
            ),
        )
        log_status(f"wrote checkpoint summary: {out / 'checkpoint_summary.json'}")
        if tracker is not None:
            tracker.log_checkpoint(checkpoint_path)
        panel_paths = write_qa_panels(model, val_ds, device, out / "qa_panels", config.qa_panel_count, config.skeleton_threshold)
        log_status(f"wrote QA panels: {out / 'qa_panels'}")
        if tracker is not None:
            tracker.log_qa_panels(panel_paths)
        return {
            "log": log_rows,
            "validation_metrics": val_metrics,
            "real_validation_metrics": real_val_metrics,
            "device": str(device),
        }
    finally:
        if tracker is not None:
            tracker.finish()


def run_metadata(
    config: BaselineConfig,
    train_ds: Schema08PatchDataset,
    val_ds: Schema08PatchDataset,
    device: torch.device,
    real_train_ds: Schema08PatchDataset | None = None,
    real_val_ds: Schema08PatchDataset | None = None,
) -> dict[str, Any]:
    split_counts = {"train": len(train_ds.rows), "validation": len(val_ds.rows)}
    if real_train_ds is not None and real_val_ds is not None:
        split_counts["real_train"] = len(real_train_ds.rows)
        split_counts["real_validation"] = len(real_val_ds.rows)
    rows = train_ds.rows + val_ds.rows
    if real_train_ds is not None:
        rows += real_train_ds.rows
    if real_val_ds is not None:
        rows += real_val_ds.rows
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_report": device_report(device),
        "manifest": config.manifest,
        "output_directory": config.out,
        "command_line_args": config.command_line_args or asdict(config),
        "initial_checkpoint": None,
        "git_commit": git_commit(),
        "split_counts": split_counts,
        "schema_versions": sorted({row.get("schema_version", "unknown") or "unknown" for row in rows}),
        "generator_versions": sorted({row.get("generator_version", "unknown") or "unknown" for row in rows}),
        "dataset_paths": dataset_path_info(rows),
        "target_class_pixel_counts": {
            "train": target_class_pixel_counts(train_ds),
            "validation": target_class_pixel_counts(val_ds),
            **({"real_train": target_class_pixel_counts(real_train_ds)} if real_train_ds is not None else {}),
            **({"real_validation": target_class_pixel_counts(real_val_ds)} if real_val_ds is not None else {}),
        },
    }


def load_initial_checkpoint(
    model: nn.Module,
    checkpoint_path: str | None,
    device: torch.device,
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"initial checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"{path}: checkpoint does not contain a model state dict")
    state = strip_module_prefix(state)
    incompatible = model.load_state_dict(state, strict=False)
    return {
        "path": str(path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "checkpoint_config": checkpoint.get("config") if isinstance(checkpoint, dict) else None,
    }


def strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if not state or not all(key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


def build_train_loader(
    config: BaselineConfig,
    synthetic_ds: Schema08PatchDataset,
    real_ds: Schema08PatchDataset | None,
    generator: torch.Generator,
    device: torch.device,
) -> torch.utils.data.DataLoader:
    if real_ds is None:
        return torch.utils.data.DataLoader(
            synthetic_ds,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
    if config.synthetic_real_ratio is None:
        raise ValueError("--real-manifest requires --synthetic-real-ratio, for example 80:20")
    mixed = torch.utils.data.ConcatDataset([synthetic_ds, real_ds])
    sampler = FixedRatioBatchSampler(
        len(synthetic_ds),
        len(real_ds),
        config.batch_size,
        config.synthetic_real_ratio,
        seed=config.seed,
        batches_per_epoch=config.patches_per_epoch,
    )
    return torch.utils.data.DataLoader(
        mixed,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )


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


def model_config_payload(config: BaselineConfig | None = None) -> dict[str, Any]:
    return {
        "architecture": config.model_variant if config is not None else "SmallUNet",
        "in_channels": 1,
        "base_channels": 16,
        "semantic_classes": 3,
        "heads": ["semantic", "skeleton", "optional_uncertainty"],
        "context_module": config.context_module if config is not None else "none",
        "aspp_dilations": config.aspp_dilations if config is not None else "",
    }


def wandb_config_payload(config: BaselineConfig, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "training": asdict(config),
        "metadata": metadata,
        "model": model_config_payload(config),
    }


def wandb_epoch_metrics(
    row: dict[str, float | int],
    val_metrics: dict[str, Any],
    real_val_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = {
        "epoch": row["epoch"],
        "train/loss": row["train_loss"],
        "train/semantic_loss": row["train_semantic_loss"],
        "train/skeleton_loss": row["train_skeleton_loss"],
        "train/uncertainty_loss": loggable_metric(row.get("train_uncertainty_loss")),
        "train/clump_anti_fibrous_loss": loggable_metric(row.get("train_clump_anti_fibrous_loss")),
        "train/clump_anti_skeleton_loss": loggable_metric(row.get("train_clump_anti_skeleton_loss")),
        "best_metric/current": loggable_metric(row.get("best_metric_value")),
        "best_metric/best": loggable_metric(row.get("best_metric_best_value")),
        "best_metric/best_epoch": loggable_metric(row.get("best_epoch")),
        "best_metric/patience_counter": loggable_metric(row.get("early_stop_patience_counter")),
        "best_metric/stopped_early": bool(row.get("stopped_early", 0)),
        "validation/loss": loggable_metric(val_metrics["loss"]),
        "validation/semantic_loss": loggable_metric(val_metrics["semantic_loss"]),
        "validation/skeleton_loss": loggable_metric(val_metrics["skeleton_loss"]),
        "validation/uncertainty_loss": loggable_metric(val_metrics.get("uncertainty_loss")),
        "validation/clump_anti_fibrous_loss": loggable_metric(val_metrics.get("clump_anti_fibrous_loss")),
        "validation/clump_anti_skeleton_loss": loggable_metric(val_metrics.get("clump_anti_skeleton_loss")),
        "validation/fibrous_dice": loggable_metric(val_metrics["fibrous_dice"]),
        "validation/clump_dice": loggable_metric(val_metrics["clump_dice"]),
        "validation/clump_target_pixels": loggable_metric(val_metrics["clump_target_pixels"]),
        "validation/uncertain_dice": loggable_metric(val_metrics.get("uncertain_dice")),
        "validation/uncertain_precision": loggable_metric(val_metrics.get("uncertain_precision")),
        "validation/uncertain_recall": loggable_metric(val_metrics.get("uncertain_recall")),
        "validation/uncertain_target_pixels": loggable_metric(val_metrics.get("uncertain_target_pixels")),
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
    for region, values in val_metrics.get("uncertainty_probability_by_target_region", {}).items():
        for name, value in values.items():
            metrics[f"validation/uncertainty_probability/{region}/{name}"] = loggable_metric(value)
    if real_val_metrics is not None:
        metrics.update(prefixed_wandb_metrics("real_validation", real_val_metrics))
    return metrics


def prefixed_wandb_metrics(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    out = {
        f"{prefix}/loss": loggable_metric(values["loss"]),
        f"{prefix}/semantic_loss": loggable_metric(values["semantic_loss"]),
        f"{prefix}/skeleton_loss": loggable_metric(values["skeleton_loss"]),
        f"{prefix}/uncertainty_loss": loggable_metric(values.get("uncertainty_loss")),
        f"{prefix}/clump_anti_fibrous_loss": loggable_metric(values.get("clump_anti_fibrous_loss")),
        f"{prefix}/clump_anti_skeleton_loss": loggable_metric(values.get("clump_anti_skeleton_loss")),
        f"{prefix}/fibrous_dice": loggable_metric(values["fibrous_dice"]),
        f"{prefix}/clump_dice": loggable_metric(values["clump_dice"]),
        f"{prefix}/clump_target_pixels": loggable_metric(values["clump_target_pixels"]),
        f"{prefix}/uncertain_dice": loggable_metric(values.get("uncertain_dice")),
        f"{prefix}/uncertain_precision": loggable_metric(values.get("uncertain_precision")),
        f"{prefix}/uncertain_recall": loggable_metric(values.get("uncertain_recall")),
        f"{prefix}/uncertain_target_pixels": loggable_metric(values.get("uncertain_target_pixels")),
        f"{prefix}/skeleton_dice": loggable_metric(values["skeleton_dice"]),
        f"{prefix}/skeleton_precision": loggable_metric(values["skeleton_precision"]),
        f"{prefix}/skeleton_recall": loggable_metric(values["skeleton_recall"]),
        f"{prefix}/predicted_skeleton_inside_predicted_fibrous_fraction": loggable_metric(
            values["predicted_skeleton_inside_predicted_fibrous_fraction"]
        ),
        f"{prefix}/predicted_skeleton_inside_true_fibrous_fraction": loggable_metric(
            values["predicted_skeleton_inside_true_fibrous_fraction"]
        ),
    }
    for threshold, threshold_metrics in values.get("skeleton_metrics_by_threshold", {}).items():
        for name, value in threshold_metrics.items():
            out[f"{prefix}/skeleton_threshold_{threshold}/{name}"] = loggable_metric(value)
    for region, region_values in values.get("uncertainty_probability_by_target_region", {}).items():
        for name, value in region_values.items():
            out[f"{prefix}/uncertainty_probability/{region}/{name}"] = loggable_metric(value)
    return out


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
