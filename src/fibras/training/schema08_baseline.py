"""Minimal PyTorch baseline for schema-0.8 real-compatible STED targets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt


IGNORE_INDEX = -100
BEST_MODE_AUTO = {
    "loss": "min",
    "semantic_loss": "min",
    "skeleton_loss": "min",
    "uncertainty_loss": "min",
    "foreground_loss": "min",
    "gate_loss": "min",
    "morphology_loss": "min",
    "joint_loss": "min",
    "fibrous_dice": "max",
    "clump_dice": "max",
    "skeleton_dice": "max",
    "skeleton_dice_0.75": "max",
    "uncertain_dice": "max",
    "macro_image_fibrous_dice": "max",
    "macro_image_fibrous_area_ratio": "max",
    "macro_image_clump_leakage": "min",
    "macro_image_skeleton_recovery_2px": "max",
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
    lambda_foreground: float = 1.0
    lambda_gate: float = 0.5
    lambda_morphology: float = 1.0
    lambda_joint: float = 1.0
    lambda_semantic: float = 1.0
    lambda_dice: float = 0.0
    four_class_semantic_loss: str = "pixel_ce"
    four_class_uncertain_bias: float = -6.0
    four_class_skeleton_mask: str = "complete"
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
    lambda_uncertain_fibrous: float = 0.0
    uncertain_fibrous_tau: float = 0.5
    rejection_near_distance: float = 20.0
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
    resume_checkpoint: str | None = None
    stage1_epochs: int = 0
    stage1_learning_rate: float = 1e-3
    real_sampling_weights: str = "fibrous_positive=1,clump_positive=1,uncertain_positive=1,dense_or_fibrous_clump_boundary=1,background_hard_negative=1,uniform_random=1"
    augmentation: bool = False
    command_line_args: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.model_variant in {"gated_context_unet", "four_class_context_unet"}:
            self.context_module = "aspp"


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
        augmentation: bool = False,
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
        self.augmentation = bool(augmentation)
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
        uncertain_distance = (
            distance_transform_edt(~uncertain).astype(np.float32)
            if uncertain.any() else np.full(uncertain.shape, np.inf, dtype=np.float32)
        )
        skeleton_valid = arrays.get(
            "real_compatible_skeleton_valid_mask",
            np.ones_like(semantic_raw, dtype=np.uint8),
        ).astype(bool) & ~uncertain
        semantic = np.zeros_like(semantic_raw, dtype=np.int64)
        semantic[semantic_raw == 1] = 1
        semantic[semantic_raw == 3] = 2
        semantic[uncertain] = IGNORE_INDEX
        y0, x0 = self._patch_origin(row["sample_id"], patch_index, semantic_raw, uncertain)
        ps = self.patch_size
        valid = ~uncertain[y0 : y0 + ps, x0 : x0 + ps]
        sample = {
            "image": torch.from_numpy(image[y0 : y0 + ps, x0 : x0 + ps][None, ...].copy()),
            "semantic": torch.from_numpy(semantic[y0 : y0 + ps, x0 : x0 + ps].copy()),
            "skeleton": torch.from_numpy(skeleton[y0 : y0 + ps, x0 : x0 + ps][None, ...].copy()),
            "uncertain": torch.from_numpy(uncertain[y0 : y0 + ps, x0 : x0 + ps][None, ...].astype(np.float32).copy()),
            "uncertain_distance": torch.from_numpy(
                uncertain_distance[y0 : y0 + ps, x0 : x0 + ps][None, ...].copy()
            ),
            "valid": torch.from_numpy(valid[None, ...].astype(np.float32).copy()),
            "skeleton_valid": torch.from_numpy(
                skeleton_valid[y0 : y0 + ps, x0 : x0 + ps][None, ...].astype(np.float32).copy()
            ),
            "sample_id": row["sample_id"],
            "source_image_id": row.get("source_image_id", row.get("parent_image_id", row["sample_id"])),
            "source_kind": row.get("source_kind", "synthetic"),
        }
        return augment_sample(sample, self._item_seed(row["sample_id"], patch_index)) if self.augmentation else sample

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


def augment_sample(sample: dict[str, torch.Tensor | str], seed: int) -> dict[str, torch.Tensor | str]:
    rng = np.random.default_rng(seed)
    dims = (-2, -1)
    k = int(rng.integers(0, 4))
    flip_y, flip_x = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
    out = sample.copy()
    for name, value in sample.items():
        if not torch.is_tensor(value) or value.ndim < 2:
            continue
        transformed = torch.rot90(value, k, dims)
        if flip_y:
            transformed = torch.flip(transformed, [-2])
        if flip_x:
            transformed = torch.flip(transformed, [-1])
        out[name] = transformed
    return out


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
        real_rows: list[dict[str, str]] | None = None,
        real_sampling_weights: str | None = None,
    ) -> None:
        if synthetic_len < 0 or real_len < 0 or synthetic_len + real_len <= 0:
            raise ValueError("training requires at least one sample")
        self.synthetic_len = int(synthetic_len)
        self.real_len = int(real_len)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.synthetic_per_batch, self.real_per_batch = batch_counts(batch_size, synthetic_real_ratio)
        synth_batches = int(np.ceil(self.synthetic_len / self.synthetic_per_batch)) if self.synthetic_per_batch else 0
        real_batches = int(np.ceil(self.real_len / self.real_per_batch)) if self.real_per_batch else 0
        default_batches = max(synth_batches, real_batches)
        self.batches_per_epoch = int(batches_per_epoch) if batches_per_epoch is not None else default_batches
        self.real_rows = real_rows
        self.real_sampling_weights = real_sampling_weights

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        synth_order = shuffled_cycle(self.synthetic_len, rng) if self.synthetic_per_batch else None
        real_order = (
            balanced_real_index_stream(self.real_rows, rng, self.real_sampling_weights)
            if self.real_per_batch and self.real_rows is not None
            else shuffled_cycle(self.real_len, rng) if self.real_per_batch else None
        )
        real_offset = self.synthetic_len
        for _ in range(self.batches_per_epoch):
            batch = [next(synth_order) for _ in range(self.synthetic_per_batch)] if synth_order else []
            batch += [real_offset + next(real_order) for _ in range(self.real_per_batch)] if real_order else []
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
    if synth < 0 or real < 0 or synth + real <= 0:
        raise ValueError("synthetic_real_ratio requires non-negative parts with a positive total")
    if synth == 0:
        return 0, batch_size
    if real == 0:
        return batch_size, 0
    real_count = int(round(batch_size * real / (synth + real)))
    real_count = min(max(real_count, 1), batch_size - 1)
    return batch_size - real_count, real_count


REAL_PATCH_CATEGORIES = (
    "fibrous_positive",
    "clump_positive",
    "uncertain_positive",
    "dense_or_fibrous_clump_boundary",
    "background_hard_negative",
    "uniform_random",
)


def parse_sampling_weights(raw: str | None) -> dict[str, float]:
    if raw is None:
        return {name: 1.0 for name in REAL_PATCH_CATEGORIES}
    weights = {name: 0.0 for name in REAL_PATCH_CATEGORIES}
    for item in raw.split(","):
        name, separator, value = item.strip().partition("=")
        if not separator or name not in weights:
            raise ValueError(f"invalid real sampling weight {item!r}")
        weights[name] = float(value)
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("real sampling weights must be non-negative with a positive total")
    return weights


def balanced_real_index_stream(
    rows: list[dict[str, str]],
    rng: np.random.Generator,
    raw_weights: str | None,
):
    weights = parse_sampling_weights(raw_weights)
    parent_categories: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        parent = row.get("source_image_id", row.get("parent_image_id", row["sample_id"]))
        categories = set(row.get("sampling_categories", "uniform_random").split(";"))
        categories.add("uniform_random")
        for category in categories & set(weights):
            parent_categories[parent][category].append(index)
    parents = sorted(
        parent for parent, categories in parent_categories.items()
        if any(weights[category] > 0 for category in categories)
    )
    if not parents:
        raise ValueError("no real crops match the configured sampling categories")
    while True:
        parent = parents[int(rng.integers(0, len(parents)))]
        categories = [name for name in sorted(parent_categories[parent]) if weights[name] > 0]
        probabilities = np.asarray([weights[name] for name in categories], dtype=np.float64)
        probabilities /= probabilities.sum()
        category = categories[int(rng.choice(len(categories), p=probabilities))]
        indices = parent_categories[parent][category]
        yield indices[int(rng.integers(0, len(indices)))]


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

    def decode_features(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.context(self.enc3(self.pool(e2)))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        return self.dec1(torch.cat([self.up1(d2), e1], dim=1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        d1 = self.decode_features(x)
        outputs = {
            "semantic_logits": self.semantic_head(d1),
            "skeleton_logits": self.skeleton_head(d1),
        }
        if self.uncertainty_head is not None:
            outputs["uncertainty_logits"] = self.uncertainty_head(d1)
        return outputs


class GatedContextUNet(ContextUNet):
    """Context U-Net with coupled foreground, morphology, and quantifiability heads."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        context_module: str = "aspp",
        aspp_dilations: list[int] | None = None,
    ) -> None:
        super().__init__(in_channels, base_channels, False, context_module, aspp_dilations)
        del self.semantic_head, self.skeleton_head, self.uncertainty_head
        c = base_channels
        self.foreground_head = nn.Conv2d(c, 1, 1)
        self.morphology_head = nn.Conv2d(c, 2, 1)
        self.quantifiability_head = nn.Conv2d(c, 1, 1)
        self.centreline_head = nn.Conv2d(c, 1, 1)
        self.initialize_gated_heads()

    def initialize_gated_heads(self) -> None:
        for head in (
            self.foreground_head, self.morphology_head,
            self.quantifiability_head, self.centreline_head,
        ):
            head.reset_parameters()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.decode_features(x)
        foreground_logits = self.foreground_head(features)
        morphology_logits = self.morphology_head(features)
        quantifiability_logits = self.quantifiability_head(features)
        centreline_logits = self.centreline_head(features)
        probabilities = gated_probabilities(
            foreground_logits, morphology_logits, quantifiability_logits, centreline_logits
        )
        return {
            "foreground_logits": foreground_logits,
            "morphology_logits": morphology_logits,
            "quantifiability_logits": quantifiability_logits,
            "raw_centreline_logits": centreline_logits,
            **probabilities,
        }


class FourClassContextUNet(ContextUNet):
    """Context U-Net with direct background, fibre, clump, and uncertain logits."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        context_module: str = "aspp",
        aspp_dilations: list[int] | None = None,
    ) -> None:
        super().__init__(in_channels, base_channels, False, context_module, aspp_dilations)
        del self.semantic_head, self.skeleton_head, self.uncertainty_head
        self.semantic_head = nn.Conv2d(base_channels, 4, 1)
        self.centreline_head = nn.Conv2d(base_channels, 1, 1)
        self.initialize_output_heads()

    def initialize_output_heads(self) -> None:
        self.semantic_head.reset_parameters()
        self.centreline_head.reset_parameters()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.decode_features(x)
        semantic_logits = self.semantic_head(features)
        centreline_logits = self.centreline_head(features)
        return {
            "semantic_logits": semantic_logits,
            "raw_centreline_logits": centreline_logits,
            "semantic_probabilities": torch.softmax(semantic_logits, dim=1),
            "centreline_probability": torch.sigmoid(centreline_logits),
        }


def gated_probabilities(
    foreground_logits: torch.Tensor,
    morphology_logits: torch.Tensor,
    quantifiability_logits: torch.Tensor,
    raw_centreline_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    foreground = torch.sigmoid(foreground_logits)
    morphology = torch.softmax(morphology_logits, dim=1)
    quantifiability = torch.sigmoid(quantifiability_logits)
    raw_centreline = torch.sigmoid(raw_centreline_logits)
    background = 1.0 - foreground
    uncertain = foreground * (1.0 - quantifiability)
    fibrous = foreground * quantifiability * morphology[:, 0:1]
    clump = foreground * quantifiability * morphology[:, 1:2]
    skeleton = fibrous * raw_centreline
    return {
        "foreground_probability": foreground,
        "quantifiability_probability": quantifiability,
        "conditional_fibrous_probability": morphology[:, 0:1],
        "conditional_clump_probability": morphology[:, 1:2],
        "raw_centreline_probability": raw_centreline,
        "final_background_probability": background,
        "final_fibrous_probability": fibrous,
        "final_clump_probability": clump,
        "final_uncertain_probability": uncertain,
        "final_skeleton_probability": skeleton,
    }


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
    if config.model_variant == "gated_context_unet":
        return GatedContextUNet(
            context_module="aspp",
            aspp_dilations=parse_dilations(config.aspp_dilations),
        )
    if config.model_variant == "four_class_context_unet":
        return FourClassContextUNet(
            context_module="aspp",
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
    lambda_uncertain_fibrous: float = 0.0,
    uncertain_fibrous_tau: float = 0.5,
    rejection_near_distance: float = 20.0,
    lambda_clump_anti_fibrous: float = 0.0,
    lambda_clump_anti_skeleton: float = 0.0,
    lambda_foreground: float = 1.0,
    lambda_gate: float = 0.5,
    lambda_morphology: float = 1.0,
    lambda_joint: float = 1.0,
    lambda_semantic: float = 1.0,
    lambda_dice: float = 0.0,
    four_class_semantic_loss: str = "pixel_ce",
    four_class_skeleton_mask: str = "complete",
) -> tuple[torch.Tensor, dict[str, float]]:
    if "foreground_logits" in outputs:
        return compute_gated_loss(
            outputs, batch, lambda_foreground, lambda_gate, lambda_morphology,
            lambda_joint, lambda_skeleton, skeleton_pos_weight, rejection_near_distance,
        )
    if outputs["semantic_logits"].shape[1] == 4:
        return compute_four_class_loss(
            outputs, batch, lambda_semantic, lambda_dice, lambda_skeleton,
            skeleton_pos_weight, four_class_semantic_loss, four_class_skeleton_mask,
        )
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
    rejection_parts: dict[str, float] = {}
    if "uncertainty_logits" in outputs and lambda_uncertainty > 0:
        if uncertainty_loss_name == "stratified_bce":
            uncertainty_loss, rejection_parts = stratified_rejection_loss(
                outputs["uncertainty_logits"], batch, rejection_near_distance
            )
        else:
            uncertainty_loss = uncertainty_loss_value(
                outputs["uncertainty_logits"],
                batch["uncertain"],
                uncertainty_loss_name,
                uncertainty_focal_gamma,
                uncertainty_pos_weight,
            )
        total = total + float(lambda_uncertainty) * uncertainty_loss
    uncertain_fibrous_loss = uncertain_fibrous_suppression_loss(
        outputs["semantic_logits"], batch.get("uncertain", torch.zeros_like(batch["skeleton"])), uncertain_fibrous_tau
    )
    if lambda_uncertain_fibrous > 0:
        total = total + float(lambda_uncertain_fibrous) * uncertain_fibrous_loss
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
        "uncertain_fibrous_loss": float(uncertain_fibrous_loss.detach().cpu()),
        "clump_anti_fibrous_loss": float(clump_anti_fibrous_loss.detach().cpu()),
        "clump_anti_skeleton_loss": float(clump_anti_skeleton_loss.detach().cpu()),
    }
    if uncertainty_loss is not None:
        parts["uncertainty_loss"] = float(uncertainty_loss.detach().cpu())
        parts.update(rejection_parts)
    return total, parts


def four_class_target(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the legacy internal targets to background/fibre/clump/uncertain indices 0..3."""
    target = batch["semantic"].clone()
    uncertain = batch.get("uncertain", torch.zeros_like(batch["skeleton"]))[:, 0] > 0.5
    target[uncertain] = 3
    annotated = (batch["semantic"] != IGNORE_INDEX) | uncertain
    return target, annotated


def class_balanced_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, annotated: torch.Tensor,
) -> torch.Tensor:
    safe_target = torch.where(annotated, target, torch.zeros_like(target))
    loss_map = F.cross_entropy(logits, safe_target, reduction="none")
    means = [loss_map[(target == class_index) & annotated].mean()
             for class_index in range(logits.shape[1])
             if bool(((target == class_index) & annotated).any())]
    return torch.stack(means).mean() if means else logits.sum() * 0.0


def pixel_cross_entropy(logits: torch.Tensor, target: torch.Tensor, annotated: torch.Tensor) -> torch.Tensor:
    safe_target = torch.where(annotated, target, torch.zeros_like(target))
    loss_map = F.cross_entropy(logits, safe_target, reduction="none")
    return masked_mean(loss_map, annotated)


def multiclass_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, annotated: torch.Tensor, epsilon: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    losses = []
    for class_index in range(logits.shape[1]):
        target_class = (target == class_index) & annotated
        if not bool(target_class.any()):
            continue
        probability = probabilities[:, class_index] * annotated.to(probabilities.dtype)
        numerator = 2.0 * (probability * target_class).sum() + epsilon
        denominator = probability.sum() + target_class.sum() + epsilon
        losses.append(1.0 - numerator / denominator)
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def compute_four_class_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_semantic: float = 1.0,
    lambda_dice: float = 0.0,
    lambda_skeleton: float = 0.5,
    skeleton_pos_weight: float = 8.0,
    semantic_loss_name: str = "pixel_ce",
    skeleton_mask_name: str = "complete",
) -> tuple[torch.Tensor, dict[str, float]]:
    target, annotated = four_class_target(batch)
    if semantic_loss_name == "pixel_ce":
        semantic_loss = pixel_cross_entropy(outputs["semantic_logits"], target, annotated)
    elif semantic_loss_name == "class_balanced_ce":
        semantic_loss = class_balanced_cross_entropy(outputs["semantic_logits"], target, annotated)
    else:
        raise ValueError(f"unsupported four_class_semantic_loss {semantic_loss_name!r}")
    dice_loss = multiclass_dice_loss(outputs["semantic_logits"], target, annotated)
    if skeleton_mask_name == "complete":
        skeleton_valid = batch.get("skeleton_valid", batch["valid"]) > 0.5
    elif skeleton_mask_name == "legacy_fibre_only":
        skeleton_valid = (batch["semantic"] == 1).unsqueeze(1)
        skeleton_valid &= batch.get("skeleton_valid", batch["valid"]) > 0.5
    else:
        raise ValueError(f"unsupported four_class_skeleton_mask {skeleton_mask_name!r}")
    skeleton_map = F.binary_cross_entropy_with_logits(
        outputs["raw_centreline_logits"], batch["skeleton"],
        pos_weight=torch.tensor([skeleton_pos_weight], device=target.device).view(1, 1, 1, 1),
        reduction="none",
    )
    skeleton_loss = masked_mean(skeleton_map, skeleton_valid)
    semantic_parts = semantic_diagnostics(outputs["semantic_logits"], target, annotated)
    skeleton_parts = skeleton_diagnostics(batch["skeleton"] > 0.5, skeleton_valid)
    total = (
        float(lambda_semantic) * semantic_loss
        + float(lambda_dice) * dice_loss
        + float(lambda_skeleton) * skeleton_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "semantic_loss": float(semantic_loss.detach().cpu()),
        "dice_loss": float(dice_loss.detach().cpu()),
        "skeleton_loss": float(skeleton_loss.detach().cpu()),
        "uncertain_fibrous_loss": 0.0,
        "clump_anti_fibrous_loss": 0.0,
        "clump_anti_skeleton_loss": 0.0,
        **semantic_parts,
        **skeleton_parts,
    }


def semantic_diagnostics(
    logits: torch.Tensor, target: torch.Tensor, annotated: torch.Tensor,
) -> dict[str, float]:
    names = ("background", "fibre", "clump", "uncertain")
    probabilities = torch.softmax(logits.detach(), dim=1)
    pred = probabilities.argmax(dim=1)
    total = annotated.sum().clamp_min(1)
    parts = {}
    safe_target = torch.where(annotated, target, torch.zeros_like(target))
    loss_map = F.cross_entropy(logits.detach(), safe_target, reduction="none")
    for class_index, name in enumerate(names):
        target_mask = (target == class_index) & annotated
        parts[f"semantic_{name}_target_fraction"] = float(target_mask.sum().cpu() / total.cpu())
        parts[f"semantic_{name}_predicted_fraction"] = float(((pred == class_index) & annotated).sum().cpu() / total.cpu())
        parts[f"semantic_{name}_loss"] = (
            float(loss_map[target_mask].mean().cpu()) if bool(target_mask.any()) else float("nan")
        )
    return parts


def skeleton_diagnostics(target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    positive = target & valid
    negative = (~target) & valid
    ignored = ~valid
    return {
        "skeleton_positive_pixels": float(positive.sum().detach().cpu()),
        "skeleton_negative_pixels": float(negative.sum().detach().cpu()),
        "skeleton_ignored_pixels": float(ignored.sum().detach().cpu()),
    }


def compute_gated_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_foreground: float = 1.0,
    lambda_gate: float = 0.5,
    lambda_morphology: float = 1.0,
    lambda_joint: float = 1.0,
    lambda_skeleton: float = 0.5,
    skeleton_pos_weight: float = 8.0,
    near_distance: float = 20.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    semantic = batch["semantic"]
    uncertain = batch.get("uncertain", torch.zeros_like(batch["skeleton"]))[:, 0] > 0.5
    fibrous = semantic == 1
    clump = semantic == 2
    confident = fibrous | clump
    foreground = confident | uncertain

    foreground_loss = F.binary_cross_entropy_with_logits(
        outputs["foreground_logits"][:, 0], foreground.to(outputs["foreground_logits"].dtype)
    )
    gate_loss, gate_parts = stratified_gate_loss(
        outputs["quantifiability_logits"][:, 0], semantic, uncertain,
        batch.get("uncertain_distance"), near_distance,
    )
    morphology_map = F.cross_entropy(
        outputs["morphology_logits"], (semantic == 2).long(), reduction="none"
    )
    morphology_loss = masked_mean(morphology_map, confident)

    final = torch.cat([
        outputs["final_background_probability"],
        outputs["final_fibrous_probability"],
        outputs["final_clump_probability"],
        outputs["final_uncertain_probability"],
    ], dim=1).clamp(min=1e-7, max=1.0)
    joint_target = torch.zeros_like(semantic)
    joint_target[fibrous] = 1
    joint_target[clump] = 2
    joint_target[uncertain] = 3
    joint_loss = -torch.log(final.gather(1, joint_target.unsqueeze(1))[:, 0]).mean()

    skeleton_mask = fibrous.unsqueeze(1) & (batch.get("skeleton_valid", batch["valid"]) > 0.5)
    skeleton_map = F.binary_cross_entropy_with_logits(
        outputs["raw_centreline_logits"], batch["skeleton"],
        pos_weight=torch.tensor([skeleton_pos_weight], device=semantic.device).view(1, 1, 1, 1),
        reduction="none",
    )
    skeleton_loss = masked_mean(skeleton_map, skeleton_mask)
    total = (
        float(lambda_foreground) * foreground_loss
        + float(lambda_gate) * gate_loss
        + float(lambda_morphology) * morphology_loss
        + float(lambda_joint) * joint_loss
        + float(lambda_skeleton) * skeleton_loss
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "foreground_loss": float(foreground_loss.detach().cpu()),
        "gate_loss": float(gate_loss.detach().cpu()),
        "morphology_loss": float(morphology_loss.detach().cpu()),
        "joint_loss": float(joint_loss.detach().cpu()),
        "skeleton_loss": float(skeleton_loss.detach().cpu()),
        # Compatibility summary for existing logs and selection code.
        "semantic_loss": float(joint_loss.detach().cpu()),
        "uncertain_fibrous_loss": 0.0,
        "clump_anti_fibrous_loss": 0.0,
        "clump_anti_skeleton_loss": 0.0,
        **gate_parts,
    }
    return total, parts


def stratified_gate_loss(
    logits: torch.Tensor,
    semantic: torch.Tensor,
    uncertain: torch.Tensor,
    uncertain_distance: torch.Tensor | None = None,
    near_distance: float = 20.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average uncertainty and confident-foreground strata without background."""
    target = ((semantic == 1) | (semantic == 2)).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if not bool((((semantic == 1) | (semantic == 2)) | uncertain).any()):
        return logits.sum() * 0.0, {}
    near = (
        uncertain_distance[:, 0] <= float(near_distance)
        if uncertain_distance is not None else torch.ones_like(uncertain)
    )
    strata = {
        "uncertain": uncertain,
        "near_fibrous": (semantic == 1) & near,
        "near_clump": (semantic == 2) & near,
        "far_fibrous": (semantic == 1) & ~near,
        "far_clump": (semantic == 2) & ~near,
    }
    means = {name: masked_mean_or_none(bce, mask) for name, mask in strata.items()}
    negative = means["uncertain"]
    confident_values = [means[name] for name in strata if name != "uncertain" and means[name] is not None]
    confident_loss = sum(confident_values) / len(confident_values) if confident_values else None
    loss = weighted_present_mean(
        {"uncertain": negative, "confident": confident_loss},
        {"uncertain": 0.5, "confident": 0.5},
    )
    parts = {
        f"gate_{name}_loss": float(value.detach().cpu())
        for name, value in means.items() if value is not None
    }
    return loss, parts


def uncertain_fibrous_suppression_loss(
    semantic_logits: torch.Tensor,
    uncertain: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    excess = torch.relu(semantic_prob(semantic_logits)[:, 1:2] - float(tau)).square()
    return masked_mean(excess, uncertain > 0.5)


def stratified_rejection_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    near_distance: float = 20.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["uncertain"]
    semantic = batch["semantic"].unsqueeze(1)
    distance = batch["uncertain_distance"]
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    near = distance <= float(near_distance)
    masks = {
        "positive": target > 0.5,
        "near_fibrous": (semantic == 1) & near,
        "near_clump": (semantic == 2) & near,
        "far_tau": ((semantic == 1) | (semantic == 2)) & ~near,
        "background": semantic == 0,
    }
    means = {name: masked_mean_or_none(bce, mask) for name, mask in masks.items()}
    negative = weighted_present_mean(means, {
        "near_fibrous": 0.30,
        "near_clump": 0.20,
        "far_tau": 0.25,
        "background": 0.25,
    })
    loss = weighted_present_mean({"positive": means["positive"], "negative": negative}, {
        "positive": 0.5, "negative": 0.5,
    })
    parts = {
        f"rejection_{name}_loss": float(value.detach().cpu())
        for name, value in means.items() if value is not None
    }
    return loss, parts


def masked_mean_or_none(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    return values[mask].mean() if bool(mask.any()) else None


def weighted_present_mean(
    values: dict[str, torch.Tensor | None],
    weights: dict[str, float],
) -> torch.Tensor:
    present = [(values[name], weight) for name, weight in weights.items() if values.get(name) is not None]
    if not present:
        reference = next((value for value in values.values() if value is not None), None)
        if reference is None:
            raise ValueError("stratified rejection loss has no populated strata")
        return reference * 0.0
    total_weight = sum(weight for _, weight in present)
    return sum(value * weight for value, weight in present if value is not None) / total_weight


def skeleton_valid_mask(batch: dict[str, torch.Tensor], policy: str) -> torch.Tensor:
    if "skeleton_valid" in batch:
        return batch["skeleton_valid"] * batch["valid"]
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
    lambda_uncertain_fibrous: float = 0.0,
    uncertain_fibrous_tau: float = 0.5,
    rejection_near_distance: float = 20.0,
    lambda_clump_anti_fibrous: float = 0.0,
    lambda_clump_anti_skeleton: float = 0.0,
    lambda_foreground: float = 1.0,
    lambda_gate: float = 0.5,
    lambda_morphology: float = 1.0,
    lambda_joint: float = 1.0,
    lambda_semantic: float = 1.0,
    lambda_dice: float = 0.0,
    four_class_semantic_loss: str = "pixel_ce",
    four_class_skeleton_mask: str = "complete",
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
    gated_moments: dict[str, dict[str, dict[str, float]]] = {}
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
                lambda_uncertain_fibrous,
                uncertain_fibrous_tau,
                rejection_near_distance,
                lambda_clump_anti_fibrous,
                lambda_clump_anti_skeleton,
                lambda_foreground,
                lambda_gate,
                lambda_morphology,
                lambda_joint,
                lambda_semantic,
                lambda_dice,
                four_class_semantic_loss,
                four_class_skeleton_mask,
            )
            losses.append(loss_parts)
            gated = "foreground_logits" in outputs
            four_class = not gated and outputs["semantic_logits"].shape[1] == 4
            if gated:
                final_semantic = torch.cat([
                    outputs["final_background_probability"],
                    outputs["final_fibrous_probability"],
                    outputs["final_clump_probability"],
                    outputs["final_uncertain_probability"],
                ], dim=1)
                pred_sem = final_semantic.argmax(dim=1)
            else:
                pred_sem = outputs["semantic_logits"].argmax(dim=1)
            target_sem = batch["semantic"]
            valid = target_sem != IGNORE_INDEX
            skeleton_valid = batch.get("skeleton_valid", batch["valid"])[:, 0] > 0.5
            skeleton_prob = (
                outputs["final_skeleton_probability"][:, 0]
                if gated else outputs["centreline_probability"][:, 0]
                if four_class else torch.sigmoid(outputs["skeleton_logits"][:, 0])
            )
            pred_skel = skeleton_prob > skeleton_threshold
            true_skel = batch["skeleton"][:, 0] > 0.5
            uncertain_target = batch.get("uncertain", torch.zeros_like(batch["skeleton"]))
            true_uncertain = uncertain_target[:, 0] > 0.5
            pred_uncertain = (
                pred_sem == 3 if gated or four_class else
                torch.sigmoid(outputs["uncertainty_logits"][:, 0]) > 0.5
                if "uncertainty_logits" in outputs else None
            )
            uncertainty_prob = (
                outputs["final_uncertain_probability"][:, 0] if gated else
                outputs["semantic_probabilities"][:, 3] if four_class else
                torch.sigmoid(outputs["uncertainty_logits"][:, 0])
                if "uncertainty_logits" in outputs else None
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
                if gated:
                    append_gated_probability_moments(
                        gated_moments, outputs, b, target_sem[b], true_uncertain[b]
                    )
                skel = target_positive_dice(pred_skel[b], true_skel[b], skeleton_valid[b])
                if skel is not None:
                    skeleton_dice.append(skel)
                pred_count = int((pred_skel[b] & skeleton_valid[b]).sum())
                true_count = int((true_skel[b] & skeleton_valid[b]).sum())
                tp = int((pred_skel[b] & true_skel[b] & skeleton_valid[b]).sum())
                if pred_count:
                    skeleton_precision.append(float(tp / pred_count))
                    pred_inside_pred_fibrous.append(float(((pred_skel[b] & (pred_sem[b] == 1) & skeleton_valid[b]).sum() / pred_count).cpu()))
                    pred_inside_true_fibrous.append(float(((pred_skel[b] & (target_sem[b] == 1) & skeleton_valid[b]).sum() / pred_count).cpu()))
                if true_count:
                    skeleton_recall.append(float(tp / true_count))
                for threshold, metrics in threshold_metrics.items():
                    pred_at_threshold = skeleton_prob[b] > threshold
                    threshold_skel = target_positive_dice(pred_at_threshold, true_skel[b], skeleton_valid[b])
                    if threshold_skel is not None:
                        metrics["dice"].append(threshold_skel)
                    pred_at_count = int((pred_at_threshold & skeleton_valid[b]).sum())
                    if pred_at_count:
                        metrics["precision"].append(float(int((pred_at_threshold & true_skel[b] & skeleton_valid[b]).sum()) / pred_at_count))
                    if true_count:
                        metrics["recall"].append(
                            float(int((pred_at_threshold & true_skel[b] & skeleton_valid[b]).sum()) / true_count)
                        )
    result = {
        "loss": mean_metric(losses, "loss"),
        "semantic_loss": mean_metric(losses, "semantic_loss"),
        "skeleton_loss": mean_metric(losses, "skeleton_loss"),
        "uncertain_fibrous_loss": mean_metric(losses, "uncertain_fibrous_loss"),
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
        "gated_probability_by_target_region": summarize_probability_moments(gated_moments),
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
    if any("dice_loss" in row for row in losses):
        result["dice_loss"] = mean_metric(losses, "dice_loss")
    if any("uncertainty_loss" in row for row in losses):
        result["uncertainty_loss"] = mean_metric(losses, "uncertainty_loss")
    for name in ("foreground_loss", "gate_loss", "morphology_loss", "joint_loss"):
        if any(name in row for row in losses):
            result[name] = mean_metric(losses, name)
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


def append_gated_probability_moments(
    out: dict[str, dict[str, dict[str, float]]], outputs: dict[str, torch.Tensor], index: int,
    target_semantic: torch.Tensor, uncertain: torch.Tensor,
) -> None:
    regions = {
        "target_background": target_semantic == 0,
        "target_fibrous": target_semantic == 1,
        "target_clump": target_semantic == 2,
        "target_uncertain_ignore": uncertain,
    }
    names = (
        "foreground_probability", "quantifiability_probability", "final_background_probability",
        "final_fibrous_probability", "final_clump_probability", "final_uncertain_probability",
    )
    for region, mask in regions.items():
        if not mask.any():
            continue
        for name in names:
            values = outputs[name][index, 0][mask].detach().double()
            moments = out.setdefault(region, {}).setdefault(
                name, {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "min": 1.0, "max": 0.0}
            )
            moments["count"] += values.numel()
            moments["sum"] += float(values.sum())
            moments["sum_sq"] += float(values.square().sum())
            moments["min"] = min(moments["min"], float(values.min()))
            moments["max"] = max(moments["max"], float(values.max()))


def summarize_probability_moments(
    moments: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for region, probabilities in moments.items():
        result[region] = {}
        for name, values in probabilities.items():
            count = values["count"]
            mean = values["sum"] / count
            variance = max(values["sum_sq"] / count - mean * mean, 0.0)
            result[region][name] = {
                "count": int(count), "mean": mean, "std": math.sqrt(variance),
                "min": values["min"], "max": values["max"],
            }
    return result


def evaluate_real_per_image(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    skeleton_threshold: float = 0.5,
) -> dict[str, Any]:
    """Macro validation metrics accumulated per parent image, never pooled globally."""
    model.eval()
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with torch.no_grad():
        for batch in loader:
            parent_ids = list(batch["source_image_id"])
            device_batch = tensor_batch_to_device(batch, device)
            outputs = model(device_batch["image"])
            gated = "foreground_logits" in outputs
            four_class = not gated and outputs["semantic_logits"].shape[1] == 4
            if gated:
                probabilities = torch.cat([
                    outputs["final_background_probability"], outputs["final_fibrous_probability"],
                    outputs["final_clump_probability"], outputs["final_uncertain_probability"],
                ], dim=1)
                pred_semantic = probabilities.argmax(dim=1).detach().cpu().numpy()
                pred_skeleton = (outputs["final_skeleton_probability"][:, 0] > skeleton_threshold).cpu().numpy()
            else:
                pred_semantic = outputs["semantic_logits"].argmax(dim=1).detach().cpu().numpy()
                pred_skeleton = (
                    (outputs["centreline_probability"][:, 0] if four_class else
                     torch.sigmoid(outputs["skeleton_logits"][:, 0])) > skeleton_threshold
                ).detach().cpu().numpy()
            target_semantic = batch["semantic"].numpy()
            target_skeleton = batch["skeleton"][:, 0].numpy() > 0.5
            valid = batch["valid"][:, 0].numpy() > 0.5
            skeleton_valid = batch.get("skeleton_valid", batch["valid"])[:, 0].numpy() > 0.5
            for index, parent in enumerate(parent_ids):
                target_fibrous = (target_semantic[index] == 1) & valid[index]
                predicted_fibrous = (pred_semantic[index] == 1) & valid[index]
                predicted_clump = (pred_semantic[index] == 2) & valid[index]
                target_clump = (target_semantic[index] == 2) & valid[index]
                values = counts[parent]
                values["fibrous_tp"] += int(np.count_nonzero(predicted_fibrous & target_fibrous))
                values["fibrous_pred"] += int(np.count_nonzero(predicted_fibrous))
                values["fibrous_target"] += int(np.count_nonzero(target_fibrous))
                values["clump_pred"] += int(np.count_nonzero(predicted_clump))
                values["clump_leak"] += int(np.count_nonzero(predicted_clump & ~target_clump))
                target_line = target_skeleton[index] & skeleton_valid[index]
                predicted_line = pred_skeleton[index] & skeleton_valid[index]
                values["skeleton_target"] += int(np.count_nonzero(target_line))
                if target_line.any() and predicted_line.any():
                    values["skeleton_recovered_2px"] += int(
                        np.count_nonzero(target_line & (distance_transform_edt(~predicted_line) <= 2.0))
                    )
    per_image = {parent: per_image_metrics(values) for parent, values in sorted(counts.items())}
    return {
        "selection_scope": "macro_per_parent_image",
        "image_count": len(per_image),
        "macro_image_fibrous_dice": macro_metric(per_image, "fibrous_dice"),
        "macro_image_fibrous_area_ratio": macro_metric(per_image, "fibrous_area_ratio"),
        "macro_image_clump_leakage": macro_metric(per_image, "clump_leakage"),
        "macro_image_skeleton_recovery_2px": macro_metric(per_image, "skeleton_recovery_2px"),
        "per_image": per_image,
    }


def per_image_metrics(values: dict[str, float]) -> dict[str, float | str]:
    fibrous_denom = values["fibrous_pred"] + values["fibrous_target"]
    return {
        "fibrous_dice": 2.0 * values["fibrous_tp"] / fibrous_denom if fibrous_denom else "not_applicable",
        "fibrous_area_ratio": values["fibrous_pred"] / values["fibrous_target"] if values["fibrous_target"] else "not_applicable",
        "clump_leakage": values["clump_leak"] / values["clump_pred"] if values["clump_pred"] else 0.0,
        "skeleton_recovery_2px": (
            values["skeleton_recovered_2px"] / values["skeleton_target"]
            if values["skeleton_target"] else "not_applicable"
        ),
    }


def macro_metric(per_image: dict[str, dict[str, float | str]], name: str) -> float | str:
    values = [float(metrics[name]) for metrics in per_image.values() if isinstance(metrics[name], (int, float))]
    return float(np.mean(values)) if values else "not_applicable"


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
    try:
        device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if requested == "auto" else torch.device(requested)
        )
    except (RuntimeError, ValueError) as exc:
        raise DeviceSelectionError(f"invalid device {requested!r}: {exc}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise DeviceSelectionError(f"unsupported device type {device.type!r}; use cpu or cuda:N")
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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "selected_logical_device": str(device),
        "selected_gpu_name": None,
    }
    if torch.cuda.is_available():
        report["cuda_current_device"] = torch.cuda.current_device()
        report["cuda_devices"] = [
            {"index": i, "name": torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())
        ]
        if device.type == "cuda":
            index = torch.cuda.current_device() if device.index is None else device.index
            report["selected_gpu_name"] = torch.cuda.get_device_name(index)
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
    optimizer: torch.optim.Optimizer | None = None,
    training_state: dict[str, Any] | None = None,
    checkpoint_kind: str = "epoch",
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "epoch": int(epoch),
        "validation_metrics": val_metrics,
        "real_validation_metrics": real_val_metrics,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "training_state": training_state or {},
        "checkpoint_metadata": {
            "kind": checkpoint_kind,
            "model_variant": config.model_variant,
            "uncertainty_head": config.enable_uncertainty_head,
            "selection_metric": config.best_metric,
            "selection_scope": "macro_per_parent_image" if config.real_manifest else "validation_dataset",
        },
    }


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    config: BaselineConfig,
    epoch: int,
    val_metrics: dict[str, Any],
    real_val_metrics: dict[str, Any] | None,
    optimizer: torch.optim.Optimizer | None = None,
    training_state: dict[str, Any] | None = None,
    checkpoint_kind: str = "epoch",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model, config, epoch, val_metrics, real_val_metrics,
            optimizer, training_state, checkpoint_kind,
        ),
        path,
    )


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
    if config.init_checkpoint and config.resume_checkpoint:
        raise ValueError("init_checkpoint and resume_checkpoint are mutually exclusive")
    if config.real_manifest and not config.best_metric.startswith("macro_image_"):
        raise ValueError("real fine-tuning checkpoint selection requires a macro_image_* best_metric")
    if config.real_manifest and config.synthetic_real_ratio is None:
        raise ValueError("real fine-tuning requires synthetic_real_ratio, including 0:100 for real only")
    if config.stage1_epochs > 0 and config.learning_rate >= config.stage1_learning_rate:
        raise ValueError("stage 2 learning_rate must be lower than stage1_learning_rate")
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
    real_only = bool(
        config.real_manifest and config.synthetic_real_ratio
        and batch_counts(config.batch_size, config.synthetic_real_ratio)[0] == 0
    )
    train_ds = None
    val_ds = None
    if not real_only:
        log_status("loading synthetic training split")
        train_ds = Schema08PatchDataset(
            Path(config.manifest), "train", config.patch_size, config.patches_per_sample,
            config.foreground_fraction, config.seed, config.cache_samples,
            config.limit_train_samples, config.patches_per_epoch, config.augmentation,
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
            augmentation=config.augmentation,
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
    if not real_only:
        log_status("loading synthetic validation split")
        val_ds = Schema08PatchDataset(
            Path(config.manifest), "validation", config.patch_size,
            config.validation_patches_per_sample, config.foreground_fraction,
            config.seed + 100_000, config.cache_samples, config.limit_val_samples,
        )
        log_status(f"validation samples: {len(val_ds.rows)}; validation patches: {len(val_ds)}")
    log_status("collecting run metadata and target pixel counts")
    metadata = run_metadata(config, train_ds, val_ds, device, real_train_ds, real_val_ds)
    save_json(out / "run_metadata.json", metadata)
    log_status(f"wrote metadata: {out / 'run_metadata.json'}")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = build_train_loader(config, train_ds, real_train_ds, generator, device)
    val_loader = (
        torch.utils.data.DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=device.type == "cuda",
        ) if val_ds is not None else None
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
    resume_payload = load_resume_checkpoint(model, config.resume_checkpoint, device) if config.resume_checkpoint else None
    checkpoint_report = (
        load_initial_checkpoint(model, config.init_checkpoint, device, config.four_class_uncertain_bias)
        if resume_payload is None else None
    )
    if checkpoint_report is not None:
        log_status(
            "loaded initial checkpoint: "
            f"{checkpoint_report['path']} "
            f"(loaded={len(checkpoint_report['loaded_keys'])}, "
            f"initialized={len(checkpoint_report.get('initialized_keys', []))}, "
            f"rejected={len(checkpoint_report.get('rejected_parameters', {}))}, "
            f"missing_backbone={len(checkpoint_report['missing_backbone_keys'])})"
        )
        metadata["initial_checkpoint"] = checkpoint_report
        save_json(out / "run_metadata.json", metadata)
    if resume_payload is not None:
        metadata["resume_checkpoint"] = {
            "path": str(config.resume_checkpoint),
            "epoch": int(resume_payload.get("epoch", 0)),
        }
        save_json(out / "run_metadata.json", metadata)
    if tracker is not None:
        log_status("initializing W&B tracking")
        tracker.start(config, metadata)
        seed_everything(config.seed)
        log_status("W&B tracking initialized")
    start_epoch = int(resume_payload.get("epoch", 0)) + 1 if resume_payload else 1
    stage = fine_tuning_stage(start_epoch, config.stage1_epochs)
    optimizer = optimizer_for_stage(model, config, stage)
    resumed_stage = resume_payload.get("training_state", {}).get("fine_tuning_stage") if resume_payload else None
    if resume_payload and resume_payload.get("optimizer_state_dict") and resumed_stage == stage:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    log_rows: list[dict[str, float | int | str]] = []
    existing_log = out / "training_log.csv"
    if resume_payload is not None and existing_log.exists():
        with existing_log.open(newline="", encoding="utf-8") as handle:
            log_rows.extend(csv.DictReader(handle))
    validation_values_by_epoch: dict[str, dict[str, float]] = {}
    resumed_state = resume_payload.get("training_state", {}) if resume_payload else {}
    best_value: float | None = resumed_state.get("best_value")
    best_epoch: int | None = resumed_state.get("best_epoch")
    patience_counter = int(resumed_state.get("patience_counter", 0))
    validation_values_by_epoch.update(resumed_state.get("validation_metric_values_by_epoch", {}))
    stopped_early = False
    early_stop_reason: str | None = None
    periodic_checkpoint_paths: list[str] = []
    final_epoch = 0
    final_metric_value: float | None = None
    try:
        for epoch in range(start_epoch, config.epochs + 1):
            expected_stage = fine_tuning_stage(epoch, config.stage1_epochs)
            if expected_stage != stage:
                stage = expected_stage
                optimizer = optimizer_for_stage(model, config, stage)
                log_status(f"switched to {stage} at learning_rate={optimizer.param_groups[0]['lr']}")
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
                    config.lambda_uncertain_fibrous,
                    config.uncertain_fibrous_tau,
                    config.rejection_near_distance,
                    config.lambda_clump_anti_fibrous,
                    config.lambda_clump_anti_skeleton,
                    config.lambda_foreground, config.lambda_gate,
                    config.lambda_morphology, config.lambda_joint,
                    config.lambda_semantic, config.lambda_dice,
                    config.four_class_semantic_loss, config.four_class_skeleton_mask,
                )
                loss.backward()
                optimizer.step()
                running.append(parts)
            log_status(f"epoch {epoch}/{config.epochs}: validating")
            validation_loader = val_loader if val_loader is not None else real_val_loader
            assert validation_loader is not None
            val_metrics = evaluate_model(
                model,
                validation_loader,
                device,
                config.lambda_skeleton,
                config.skeleton_pos_weight,
                config.skeleton_threshold,
                config.lambda_uncertainty,
                config.uncertainty_loss,
                config.uncertainty_focal_gamma,
                config.uncertainty_pos_weight,
                config.uncertain_skeleton_policy,
                config.lambda_uncertain_fibrous,
                config.uncertain_fibrous_tau,
                config.rejection_near_distance,
                config.lambda_clump_anti_fibrous,
                config.lambda_clump_anti_skeleton,
                config.lambda_foreground, config.lambda_gate,
                config.lambda_morphology, config.lambda_joint,
                config.lambda_semantic, config.lambda_dice,
                config.four_class_semantic_loss, config.four_class_skeleton_mask,
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
                    config.lambda_uncertain_fibrous,
                    config.uncertain_fibrous_tau,
                    config.rejection_near_distance,
                    config.lambda_clump_anti_fibrous,
                    config.lambda_clump_anti_skeleton,
                    config.lambda_foreground, config.lambda_gate,
                    config.lambda_morphology, config.lambda_joint,
                    config.lambda_semantic, config.lambda_dice,
                    config.four_class_semantic_loss, config.four_class_skeleton_mask,
                )
                if real_val_loader is not None and val_loader is not None
                else val_metrics if real_val_loader is not None else None
            )
            real_image_metrics = (
                evaluate_real_per_image(model, real_val_loader, device, config.skeleton_threshold)
                if real_val_loader is not None else None
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
            if "dice_loss" in running[0]:
                row["train_dice_loss"] = mean_metric(running, "dice_loss")
                row["validation_dice_loss"] = float(val_metrics["dice_loss"])
            if "uncertainty_loss" in running[0]:
                row["train_uncertainty_loss"] = mean_metric(running, "uncertainty_loss")
            for name in ("foreground_loss", "gate_loss", "morphology_loss", "joint_loss"):
                if name in running[0]:
                    row[f"train_{name}"] = mean_metric(running, name)
                    row[f"validation_{name}"] = float(val_metrics[name])
            for region, probabilities in val_metrics.get("gated_probability_by_target_region", {}).items():
                for probability, summary in probabilities.items():
                    for statistic in ("mean", "std", "min", "max"):
                        row[f"validation_{region}_{probability}_{statistic}"] = summary[statistic]
            row["train_clump_anti_fibrous_loss"] = mean_metric(running, "clump_anti_fibrous_loss")
            row["train_clump_anti_skeleton_loss"] = mean_metric(running, "clump_anti_skeleton_loss")
            for name in running[0]:
                if name.startswith(("semantic_", "skeleton_positive_", "skeleton_negative_", "skeleton_ignored_")):
                    row[f"train_{name}"] = mean_metric(running, name)
                    if name in val_metrics:
                        row[f"validation_{name}"] = float(val_metrics[name])
            selection_metrics = real_image_metrics if real_image_metrics is not None else val_metrics
            current_value = selected_validation_metric(selection_metrics, config.best_metric)
            final_epoch = epoch
            final_metric_value = current_value
            validation_values_by_epoch[str(epoch)] = validation_metric_values(selection_metrics)
            improved = best_metric_improved(current_value, best_value, best_mode, config.early_stop_min_delta)
            if improved:
                best_value = current_value
                best_epoch = epoch
                patience_counter = 0
                if config.save_best_checkpoint:
                    best_checkpoint_path = out / "model_best.pt"
                    checkpoint_real_metrics = {
                        "patch_metrics": real_val_metrics,
                        "per_image_metrics": real_image_metrics,
                    } if real_val_metrics is not None else None
                    save_training_checkpoint(
                        best_checkpoint_path, model, config, epoch, val_metrics,
                        checkpoint_real_metrics, optimizer,
                        training_state_payload(best_value, best_epoch, patience_counter, validation_values_by_epoch, stage),
                        "best",
                    )
                    log_status(
                        f"New best validation {config.best_metric}: {current_value:.4f} "
                        f"at epoch {epoch}; saved model_best.pt"
                    )
            else:
                patience_counter += 1
            if config.save_checkpoint_every > 0 and epoch % config.save_checkpoint_every == 0:
                periodic_path = out / "checkpoints" / f"model_epoch_{epoch:04d}.pt"
                checkpoint_real_metrics = {
                    "patch_metrics": real_val_metrics,
                    "per_image_metrics": real_image_metrics,
                } if real_val_metrics is not None else None
                save_training_checkpoint(
                    periodic_path, model, config, epoch, val_metrics,
                    checkpoint_real_metrics, optimizer,
                    training_state_payload(best_value, best_epoch, patience_counter, validation_values_by_epoch, stage),
                    "periodic",
                )
                periodic_checkpoint_paths.append(str(periodic_path))
                log_status(f"wrote periodic checkpoint: {periodic_path}")
            row["best_metric_value"] = current_value
            row["best_metric_best_value"] = best_value if best_value is not None else current_value
            row["best_epoch"] = best_epoch if best_epoch is not None else epoch
            row["early_stop_patience_counter"] = patience_counter
            row["stopped_early"] = 0
            row["fine_tuning_stage"] = stage
            row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            if real_image_metrics is not None:
                for name in (
                    "macro_image_fibrous_dice", "macro_image_fibrous_area_ratio",
                    "macro_image_clump_leakage", "macro_image_skeleton_recovery_2px",
                ):
                    if isinstance(real_image_metrics[name], (int, float)):
                        row[f"real_validation_{name}"] = float(real_image_metrics[name])
            log_rows.append(row)
            write_log_csv(existing_log, log_rows)
            print(
                f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
                f"validation_loss={row['validation_loss']:.4f}",
                flush=True,
            )
            warmup_complete = epoch >= int(config.early_stop_warmup_epochs)
            if config.early_stop_patience > 0 and warmup_complete and patience_counter >= config.early_stop_patience:
                stopped_early = True
                row["stopped_early"] = 1
                write_log_csv(existing_log, log_rows)
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
        validation_loader = val_loader if val_loader is not None else real_val_loader
        assert validation_loader is not None
        val_metrics = evaluate_model(
            model,
            validation_loader,
            device,
            config.lambda_skeleton,
            config.skeleton_pos_weight,
            config.skeleton_threshold,
            config.lambda_uncertainty,
            config.uncertainty_loss,
            config.uncertainty_focal_gamma,
            config.uncertainty_pos_weight,
            config.uncertain_skeleton_policy,
            config.lambda_uncertain_fibrous,
            config.uncertain_fibrous_tau,
            config.rejection_near_distance,
            config.lambda_clump_anti_fibrous,
            config.lambda_clump_anti_skeleton,
            config.lambda_foreground, config.lambda_gate,
            config.lambda_morphology, config.lambda_joint,
            config.lambda_semantic, config.lambda_dice,
            config.four_class_semantic_loss, config.four_class_skeleton_mask,
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
                config.lambda_uncertain_fibrous,
                config.uncertain_fibrous_tau,
                config.rejection_near_distance,
                config.lambda_clump_anti_fibrous,
                config.lambda_clump_anti_skeleton,
                config.lambda_foreground, config.lambda_gate,
                config.lambda_morphology, config.lambda_joint,
                config.lambda_semantic, config.lambda_dice,
                config.four_class_semantic_loss, config.four_class_skeleton_mask,
            )
            if real_val_loader is not None and val_loader is not None
            else val_metrics if real_val_loader is not None else None
        )
        real_image_metrics = (
            evaluate_real_per_image(model, real_val_loader, device, config.skeleton_threshold)
            if real_val_loader is not None else None
        )
        if final_epoch > 0:
            final_metric_value = selected_validation_metric(
                real_image_metrics if real_image_metrics is not None else val_metrics,
                config.best_metric,
            )
        save_json(out / "validation_metrics.json", val_metrics)
        log_status(f"wrote validation metrics: {out / 'validation_metrics.json'}")
        if real_val_metrics is not None:
            save_json(out / "real_validation_metrics.json", real_val_metrics)
            log_status(f"wrote real validation metrics: {out / 'real_validation_metrics.json'}")
        if real_image_metrics is not None:
            save_json(out / "real_validation_per_image_metrics.json", real_image_metrics)
            log_status(f"wrote real per-image metrics: {out / 'real_validation_per_image_metrics.json'}")
        checkpoint_path = out / "model.pt"
        checkpoint_real_metrics = {
            "patch_metrics": real_val_metrics,
            "per_image_metrics": real_image_metrics,
        } if real_val_metrics is not None else None
        save_training_checkpoint(
            checkpoint_path, model, config, final_epoch, val_metrics,
            checkpoint_real_metrics, optimizer,
            training_state_payload(best_value, best_epoch, patience_counter, validation_values_by_epoch, stage),
            "final",
        )
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
        panel_dataset = val_ds if val_ds is not None else real_val_ds
        assert panel_dataset is not None
        panel_paths = write_qa_panels(model, panel_dataset, device, out / "qa_panels", config.qa_panel_count, config.skeleton_threshold)
        log_status(f"wrote QA panels: {out / 'qa_panels'}")
        if tracker is not None:
            tracker.log_qa_panels(panel_paths)
        return {
            "log": log_rows,
            "validation_metrics": val_metrics,
            "real_validation_metrics": real_val_metrics,
            "real_validation_per_image_metrics": real_image_metrics,
            "device": str(device),
        }
    finally:
        if tracker is not None:
            tracker.finish()


def run_metadata(
    config: BaselineConfig,
    train_ds: Schema08PatchDataset | None,
    val_ds: Schema08PatchDataset | None,
    device: torch.device,
    real_train_ds: Schema08PatchDataset | None = None,
    real_val_ds: Schema08PatchDataset | None = None,
) -> dict[str, Any]:
    split_counts = {
        "train": len(train_ds.rows) if train_ds is not None else 0,
        "validation": len(val_ds.rows) if val_ds is not None else 0,
    }
    if real_train_ds is not None and real_val_ds is not None:
        split_counts["real_train"] = len(real_train_ds.rows)
        split_counts["real_validation"] = len(real_val_ds.rows)
    rows = (train_ds.rows if train_ds is not None else []) + (val_ds.rows if val_ds is not None else [])
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
            **({"train": target_class_pixel_counts(train_ds)} if train_ds is not None else {}),
            **({"validation": target_class_pixel_counts(val_ds)} if val_ds is not None else {}),
            **({"real_train": target_class_pixel_counts(real_train_ds)} if real_train_ds is not None else {}),
            **({"real_validation": target_class_pixel_counts(real_val_ds)} if real_val_ds is not None else {}),
        },
    }


def load_initial_checkpoint(
    model: nn.Module,
    checkpoint_path: str | None,
    device: torch.device,
    four_class_uncertain_bias: float = -6.0,
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
    target = model.state_dict()
    if isinstance(model, FourClassContextUNet):
        return load_four_class_initial_checkpoint(
            model, state, checkpoint, path, float(four_class_uncertain_bias)
        )
    partial_backbone = isinstance(model, (GatedContextUNet, FourClassContextUNet))
    if not partial_backbone:
        incompatible = model.load_state_dict(state, strict=False)
        return {
            "path": str(path),
            "loaded_keys": sorted(key for key in state if key in target),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "incompatible_parameters": {},
            "initialized_keys": list(incompatible.missing_keys),
            "rejected_parameters": {},
            "missing_backbone_keys": [],
            "checkpoint_config": checkpoint.get("config") if isinstance(checkpoint, dict) else None,
        }
    compatible = {key: value for key, value in state.items() if key in target and target[key].shape == value.shape}
    incompatible_parameters = {
        key: {
            "checkpoint_shape": list(value.shape),
            "model_shape": list(target[key].shape) if key in target else None,
        }
        for key, value in state.items()
        if key not in target or target[key].shape != value.shape
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    backbone_prefixes = ("enc1.", "enc2.", "enc3.", "context.", "up2.", "dec2.", "up1.", "dec1.")
    missing_backbone = [key for key in incompatible.missing_keys if key.startswith(backbone_prefixes)]
    if missing_backbone:
        raise ValueError(f"{path}: partial initialization is missing backbone parameters: {missing_backbone}")
    initialized = [key for key in incompatible.missing_keys if not key.startswith(backbone_prefixes)]
    return {
        "path": str(path),
        "loaded_keys": sorted(compatible),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": [key for key in state if key not in target],
        "incompatible_parameters": incompatible_parameters,
        "initialized_keys": initialized,
        "rejected_parameters": incompatible_parameters,
        "missing_backbone_keys": missing_backbone,
        "checkpoint_config": checkpoint.get("config") if isinstance(checkpoint, dict) else None,
    }


def load_four_class_initial_checkpoint(
    model: FourClassContextUNet,
    state: dict[str, torch.Tensor],
    checkpoint: Any,
    path: Path,
    uncertain_bias: float,
) -> dict[str, Any]:
    target = model.state_dict()
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    native_missing = [key for key in target if key not in state or state[key].shape != target[key].shape]
    if not native_missing:
        model.load_state_dict({key: state[key] for key in target}, strict=True)
        return {
            "path": str(path),
            "loaded_keys": sorted(target),
            "copied_keys": sorted(target),
            "initialized_keys": [],
            "missing_keys": [],
            "unexpected_keys": [key for key in state if key not in target],
            "incompatible_parameters": incompatible_parameters(state, target),
            "rejected_parameters": incompatible_parameters(state, target),
            "missing_backbone_keys": [],
            "missing_inherited_keys": [],
            "checkpoint_config": checkpoint_config,
            "warm_start_kind": "native_four_class",
        }

    backbone_prefixes = ("enc1.", "enc2.", "enc3.", "context.", "up2.", "dec2.", "up1.", "dec1.")
    required = [key for key in target if key.startswith(backbone_prefixes)]
    required += ["semantic_head.weight", "semantic_head.bias", "skeleton_head.weight", "skeleton_head.bias"]
    problems = {
        key: {
            "checkpoint_shape": list(state[key].shape) if key in state else None,
            "expected_shape": legacy_four_class_expected_shape(key, target),
        }
        for key in required
        if key not in state or list(state[key].shape) != legacy_four_class_expected_shape(key, target)
    }
    if problems:
        raise ValueError(f"{path}: missing or incompatible inherited four-class warm-start tensors: {problems}")

    new_state = {key: value.detach().clone() for key, value in target.items()}
    copied = []
    for key in target:
        if key.startswith(backbone_prefixes):
            new_state[key] = state[key].detach().clone()
            copied.append(key)
    new_state["semantic_head.weight"][:3] = state["semantic_head.weight"].detach()
    new_state["semantic_head.bias"][:3] = state["semantic_head.bias"].detach()
    new_state["centreline_head.weight"] = state["skeleton_head.weight"].detach().clone()
    new_state["centreline_head.bias"] = state["skeleton_head.bias"].detach().clone()
    new_state["semantic_head.weight"][3].zero_()
    new_state["semantic_head.bias"][3].fill_(uncertain_bias)
    copied += [
        "semantic_head.weight[0:3]", "semantic_head.bias[0:3]",
        "skeleton_head.weight->centreline_head.weight",
        "skeleton_head.bias->centreline_head.bias",
    ]
    initialized = ["semantic_head.weight[3]", "semantic_head.bias[3]"]
    model.load_state_dict(new_state, strict=True)
    rejected = {
        key: {"checkpoint_shape": list(value.shape), "model_shape": list(target[key].shape) if key in target else None}
        for key, value in state.items()
        if key not in required
        and not (key in target and key.startswith(backbone_prefixes))
    }
    return {
        "path": str(path),
        "loaded_keys": sorted(key for key in state if key in target and key.startswith(backbone_prefixes)),
        "copied_keys": sorted(copied),
        "initialized_keys": initialized,
        "missing_keys": [],
        "unexpected_keys": [key for key in state if key not in target],
        "incompatible_parameters": incompatible_parameters(state, target),
        "rejected_parameters": rejected,
        "missing_backbone_keys": [],
        "missing_inherited_keys": [],
        "checkpoint_config": checkpoint_config,
        "warm_start_kind": "legacy_context_unet_to_four_class",
        "four_class_uncertain_bias": uncertain_bias,
    }


def legacy_four_class_expected_shape(key: str, target: dict[str, torch.Tensor]) -> list[int]:
    if key == "semantic_head.weight":
        return [3, *list(target[key].shape[1:])]
    if key == "semantic_head.bias":
        return [3]
    if key == "skeleton_head.weight":
        return list(target["centreline_head.weight"].shape)
    if key == "skeleton_head.bias":
        return list(target["centreline_head.bias"].shape)
    return list(target[key].shape)


def incompatible_parameters(
    state: dict[str, torch.Tensor], target: dict[str, torch.Tensor],
) -> dict[str, dict[str, list[int] | None]]:
    return {
        key: {
            "checkpoint_shape": list(value.shape),
            "model_shape": list(target[key].shape) if key in target else None,
        }
        for key, value in state.items()
        if key not in target or target[key].shape != value.shape
    }


def load_resume_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path}: resume requires a complete training checkpoint")
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]))
    return checkpoint


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    for name in ("enc1", "enc2", "enc3", "context"):
        module = getattr(model, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = trainable


def fine_tuning_stage(epoch: int, stage1_epochs: int) -> str:
    return "stage1_frozen_encoder" if stage1_epochs > 0 and epoch <= stage1_epochs else "stage2_full_model"


def optimizer_for_stage(model: nn.Module, config: BaselineConfig, stage: str) -> torch.optim.Optimizer:
    frozen = stage == "stage1_frozen_encoder"
    set_encoder_trainable(model, not frozen)
    learning_rate = config.stage1_learning_rate if frozen else config.learning_rate
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"{stage}: no trainable model parameters")
    return torch.optim.Adam(parameters, lr=learning_rate)


def training_state_payload(
    best_value: float | None,
    best_epoch: int | None,
    patience_counter: int,
    validation_values_by_epoch: dict[str, dict[str, float]],
    stage: str,
) -> dict[str, Any]:
    return {
        "best_value": best_value,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "validation_metric_values_by_epoch": validation_values_by_epoch,
        "fine_tuning_stage": stage,
    }


def strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if not state or not all(key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


def build_train_loader(
    config: BaselineConfig,
    synthetic_ds: Schema08PatchDataset | None,
    real_ds: Schema08PatchDataset | None,
    generator: torch.Generator,
    device: torch.device,
) -> torch.utils.data.DataLoader:
    if real_ds is None:
        if synthetic_ds is None:
            raise ValueError("no training dataset configured")
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
    synthetic_len = len(synthetic_ds) if synthetic_ds is not None else 0
    mixed = torch.utils.data.ConcatDataset([synthetic_ds, real_ds]) if synthetic_ds is not None else real_ds
    sampler = FixedRatioBatchSampler(
        synthetic_len,
        len(real_ds),
        config.batch_size,
        config.synthetic_real_ratio,
        seed=config.seed,
        batches_per_epoch=config.patches_per_epoch,
        real_rows=real_ds.rows,
        real_sampling_weights=config.real_sampling_weights,
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
    gated = config is not None and config.model_variant == "gated_context_unet"
    four_class = config is not None and config.model_variant == "four_class_context_unet"
    return {
        "architecture": config.model_variant if config is not None else "SmallUNet",
        "in_channels": 1,
        "base_channels": 16,
        "semantic_classes": 4 if gated or four_class else 3,
        "heads": (
            ["foreground", "conditional_morphology", "quantifiability", "raw_centreline"]
            if gated else ["four_class_semantic", "centreline"]
            if four_class else ["semantic", "skeleton", "optional_uncertainty"]
        ),
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
        "train/dice_loss": loggable_metric(row.get("train_dice_loss")),
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
        "validation/dice_loss": loggable_metric(val_metrics.get("dice_loss")),
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
    for name in ("foreground_loss", "gate_loss", "morphology_loss", "joint_loss"):
        metrics[f"train/{name}"] = loggable_metric(row.get(f"train_{name}"))
        metrics[f"validation/{name}"] = loggable_metric(val_metrics.get(name))
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
        f"{prefix}/dice_loss": loggable_metric(values.get("dice_loss")),
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
    for name in ("foreground_loss", "gate_loss", "morphology_loss", "joint_loss"):
        out[f"{prefix}/{name}"] = loggable_metric(values.get(name))
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
        if "foreground_logits" in outputs:
            final = torch.cat([
                outputs["final_background_probability"], outputs["final_fibrous_probability"],
                outputs["final_clump_probability"], outputs["final_uncertain_probability"],
            ], dim=1)
            pred_sem = final.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
            pred_sem[pred_sem == 3] = IGNORE_INDEX
            pred_skel = outputs["final_skeleton_probability"][0, 0].cpu().numpy() > skeleton_threshold
        else:
            pred_sem = outputs["semantic_logits"].argmax(dim=1)[0].cpu().numpy().astype(np.int64)
            four_class = outputs["semantic_logits"].shape[1] == 4
            if four_class:
                pred_sem[pred_sem == 3] = IGNORE_INDEX
            skeleton_probability = (
                outputs["centreline_probability"][0, 0] if four_class
                else torch.sigmoid(outputs["skeleton_logits"][0, 0])
            )
            pred_skel = skeleton_probability.cpu().numpy() > skeleton_threshold
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
