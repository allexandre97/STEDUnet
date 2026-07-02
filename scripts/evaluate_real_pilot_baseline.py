#!/usr/bin/env python
"""Evaluate the first schema-0.8 synthetic-only baseline on real pilot annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fibras.annotations import build_real_annotation_sample


NOT_APPLICABLE = "not_applicable"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint is None and args.run_dir is None:
        print("error: provide --checkpoint or --run-dir", file=sys.stderr)
        return 2
    try:
        torch = import_torch()
        model, checkpoint_path, device = load_model(args.checkpoint, args.run_dir, args.device, torch)
        sample = build_real_annotation_sample(args.image, args.snakes, args.labels)
        predictions = run_tiled_inference(
            model,
            sample["image_uint8"] if sample["image_uint8"] is not None else sample["image_float"],
            device,
            torch,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            overlap=args.tile_overlap,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    metrics = compute_real_pilot_metrics(
        sample["real_semantic_mask"],
        sample["real_skeleton_mask"],
        predictions["semantic_class_map"],
        predictions["skeleton_probability"],
        skeleton_threshold=args.skeleton_threshold,
    )
    metrics["metadata"] = {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "image": str(args.image),
        "snakes": str(args.snakes),
        "labels": str(args.labels),
        "patch_size": int(args.patch_size),
        "batch_size": int(args.batch_size),
        "tile_overlap": int(args.tile_overlap),
        "device": str(args.device),
        "semantic_threshold": args.semantic_threshold,
    }
    write_outputs(args.out, args.image, sample, predictions, metrics)
    print(f"wrote outputs under {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--snakes", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skeleton-threshold", type=float, default=0.75)
    parser.add_argument("--semantic-threshold", type=float)
    parser.add_argument("--overlap", "--tile-overlap", dest="tile_overlap", type=int, default=32)
    return parser


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint loading and inference") from exc
    return torch


def load_model(checkpoint: Path | None, run_dir: Path | None, device_name: str, torch: Any) -> tuple[Any, Path, Any]:
    from fibras.training.schema08_baseline import SmallUNet, choose_device

    device = choose_device(device_name)
    checkpoint_path = resolve_checkpoint_path(checkpoint, run_dir)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model = SmallUNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint_path, device


def resolve_checkpoint_path(checkpoint: Path | None, run_dir: Path | None) -> Path:
    if checkpoint is not None:
        path = checkpoint
        if path.is_dir():
            path = path / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    assert run_dir is not None
    candidates = [run_dir / "model.pt", *sorted(run_dir.glob("*.pt"))]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no checkpoint found in run directory: {run_dir}")


def normalize_like_training(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0, 255) / 255.0


def run_tiled_inference(
    model: Any,
    image: np.ndarray,
    device: Any,
    torch: Any,
    *,
    patch_size: int = 128,
    batch_size: int = 4,
    overlap: int = 32,
) -> dict[str, np.ndarray]:
    image_f = normalize_like_training(image)
    h, w = image_f.shape
    pad_h = max(h, patch_size)
    pad_w = max(w, patch_size)
    padded = np.pad(image_f, ((0, pad_h - h), (0, pad_w - w)), mode="edge")
    starts = list(tile_origins(padded.shape, patch_size, overlap))
    semantic_logits = np.zeros((3, *padded.shape), dtype=np.float32)
    skeleton_logits = np.zeros(padded.shape, dtype=np.float32)
    weights = np.zeros(padded.shape, dtype=np.float32)
    window = blend_window(patch_size)

    model.eval()
    with torch.no_grad():
        for batch_starts in batched(starts, batch_size):
            batch = np.stack([padded[y : y + patch_size, x : x + patch_size] for y, x in batch_starts])
            tensor = torch.from_numpy(batch[:, None].astype(np.float32)).to(device)
            outputs = model(tensor)
            sem = outputs["semantic_logits"].detach().cpu().numpy()
            skel = outputs["skeleton_logits"][:, 0].detach().cpu().numpy()
            for i, (y, x) in enumerate(batch_starts):
                semantic_logits[:, y : y + patch_size, x : x + patch_size] += sem[i] * window
                skeleton_logits[y : y + patch_size, x : x + patch_size] += skel[i] * window
                weights[y : y + patch_size, x : x + patch_size] += window

    weights = np.maximum(weights, 1e-6)
    semantic_logits = semantic_logits[:, :h, :w] / weights[:h, :w]
    skeleton_logits = skeleton_logits[:h, :w] / weights[:h, :w]
    semantic_prob = softmax_channel_first(semantic_logits)
    class_index = semantic_prob.argmax(axis=0).astype(np.uint8)
    semantic_class_map = np.zeros((h, w), dtype=np.uint8)
    semantic_class_map[class_index == 1] = 1
    semantic_class_map[class_index == 2] = 3
    skeleton_probability = sigmoid(skeleton_logits).astype(np.float32)
    return {
        "semantic_class_map": semantic_class_map,
        "fibrous_probability": semantic_prob[1].astype(np.float32),
        "clump_probability": semantic_prob[2].astype(np.float32),
        "skeleton_probability": skeleton_probability,
        "skeleton_mask_0_5": (skeleton_probability > 0.5).astype(np.uint8),
        "skeleton_mask_0_75": (skeleton_probability > 0.75).astype(np.uint8),
    }


def tile_origins(shape: tuple[int, int], patch_size: int, overlap: int) -> Iterable[tuple[int, int]]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must be >= 0 and smaller than patch_size")
    h, w = shape
    ys = axis_starts(h, patch_size, patch_size - overlap)
    xs = axis_starts(w, patch_size, patch_size - overlap)
    for y in ys:
        for x in xs:
            yield y, x


def axis_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_window(patch_size: int) -> np.ndarray:
    if patch_size <= 2:
        return np.ones((patch_size, patch_size), dtype=np.float32)
    one = np.hanning(patch_size).astype(np.float32)
    one = np.maximum(one, 0.05)
    return np.outer(one, one).astype(np.float32)


def batched(items: list[tuple[int, int]], size: int) -> Iterable[list[tuple[int, int]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def softmax_channel_first(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=0, keepdims=True)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_real_pilot_metrics(
    target_semantic: np.ndarray,
    target_skeleton: np.ndarray,
    pred_semantic: np.ndarray,
    skeleton_probability: np.ndarray,
    *,
    skeleton_threshold: float = 0.75,
) -> dict[str, Any]:
    valid = target_semantic != 255
    target_fibrous = (target_semantic == 1) & valid
    pred_fibrous = (pred_semantic == 1) & valid
    target_clump = (target_semantic == 3) & valid
    pred_clump = (pred_semantic == 3) & valid
    background = (target_semantic == 0) & valid
    target_skel = (target_skeleton > 0) & valid
    pred_skel = (skeleton_probability > skeleton_threshold) & valid

    metrics: dict[str, Any] = {
        "fibrous_dice": dice(pred_fibrous, target_fibrous),
        "fibrous_precision": precision(pred_fibrous, target_fibrous),
        "fibrous_recall": recall(pred_fibrous, target_fibrous),
        "clump_dice": target_positive_dice(pred_clump, target_clump),
        "clump_target_pixels": int(target_clump.sum()),
        "false_positive_fibrous_area_pixels": int((pred_fibrous & background).sum()),
        "false_positive_fibrous_area_fraction_of_background": fraction((pred_fibrous & background).sum(), background.sum()),
        "predicted_fibrous_area_fraction": fraction(pred_fibrous.sum(), valid.sum()),
        "target_fibrous_area_fraction": fraction(target_fibrous.sum(), valid.sum()),
        "predicted_to_target_fibrous_area_fraction_ratio": ratio_or_na(pred_fibrous.sum(), target_fibrous.sum()),
        "skeleton_threshold": float(skeleton_threshold),
        "predicted_skeleton_inside_predicted_fibrous_fraction": fraction((pred_skel & pred_fibrous).sum(), pred_skel.sum()),
        "predicted_skeleton_inside_target_fibrous_fraction": fraction((pred_skel & target_fibrous).sum(), pred_skel.sum()),
        "target_skeleton_pixels": int(target_skel.sum()),
        "predicted_skeleton_pixels": int(pred_skel.sum()),
        "target_skeleton_recovered_within_2px_fraction": within_distance_fraction(target_skel, pred_skel, 2.0),
        "predicted_skeleton_within_2px_of_target_fraction": within_distance_fraction(pred_skel, target_skel, 2.0),
        "semantic_leakage": semantic_leakage_counts(target_semantic, pred_semantic),
    }
    metrics["skeleton_metrics_by_threshold"] = {
        str(threshold): {
            "dice": target_positive_dice((skeleton_probability > threshold) & valid, target_skel),
            "precision": precision((skeleton_probability > threshold) & valid, target_skel),
            "recall": recall((skeleton_probability > threshold) & valid, target_skel),
            "leakage": skeleton_leakage_counts(
                target_semantic,
                pred_semantic,
                skeleton_probability > threshold,
            ),
        }
        for threshold in (0.5, 0.75, 0.85)
    }
    return metrics


def target_region_masks(target_semantic: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "target_fibrous": target_semantic == 1,
        "target_clump": target_semantic == 3,
        "target_uncertain_ignore": target_semantic == 255,
        "target_background": target_semantic == 0,
    }


def counts_and_fractions(mask: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, Any]:
    total = int(mask.sum())
    out: dict[str, Any] = {"total": total}
    for name, region in regions.items():
        count = int((mask & region).sum())
        out[f"inside_{name}_pixels"] = count
        out[f"inside_{name}_fraction"] = fraction(count, total)
    return out


def semantic_leakage_counts(target_semantic: np.ndarray, pred_semantic: np.ndarray) -> dict[str, Any]:
    regions = target_region_masks(target_semantic)
    return {
        "target_clump_pixels": int(regions["target_clump"].sum()),
        "target_uncertain_ignore_pixels": int(regions["target_uncertain_ignore"].sum()),
        "predicted_fibrous": counts_and_fractions(pred_semantic == 1, regions),
        "predicted_clump": counts_and_fractions(pred_semantic == 3, regions),
    }


def skeleton_leakage_counts(
    target_semantic: np.ndarray,
    pred_semantic: np.ndarray,
    pred_skeleton: np.ndarray,
) -> dict[str, Any]:
    regions = target_region_masks(target_semantic)
    leakage = counts_and_fractions(pred_skeleton.astype(bool, copy=False), regions)
    leakage.update(
        {
            "inside_predicted_fibrous_pixels": int((pred_skeleton & (pred_semantic == 1)).sum()),
            "inside_predicted_clump_pixels": int((pred_skeleton & (pred_semantic == 3)).sum()),
            "outside_predicted_foreground_pixels": int((pred_skeleton & ~((pred_semantic == 1) | (pred_semantic == 3))).sum()),
        }
    )
    return leakage


def dice(pred: np.ndarray, target: np.ndarray) -> float | str:
    denom = int(pred.sum() + target.sum())
    if denom == 0:
        return NOT_APPLICABLE
    return float(2 * int((pred & target).sum()) / denom)


def target_positive_dice(pred: np.ndarray, target: np.ndarray) -> float | str:
    if int(target.sum()) == 0:
        return NOT_APPLICABLE
    return dice(pred, target)


def precision(pred: np.ndarray, target: np.ndarray) -> float | str:
    pred_count = int(pred.sum())
    if pred_count == 0:
        return NOT_APPLICABLE
    return float(int((pred & target).sum()) / pred_count)


def recall(pred: np.ndarray, target: np.ndarray) -> float | str:
    target_count = int(target.sum())
    if target_count == 0:
        return NOT_APPLICABLE
    return float(int((pred & target).sum()) / target_count)


def fraction(num: Any, denom: Any) -> float | str:
    denom_i = int(denom)
    if denom_i == 0:
        return NOT_APPLICABLE
    return float(int(num) / denom_i)


def ratio_or_na(num: Any, denom: Any) -> float | str:
    denom_i = int(denom)
    if denom_i == 0:
        return NOT_APPLICABLE
    return float(int(num) / denom_i)


def within_distance_fraction(source: np.ndarray, target: np.ndarray, max_distance: float) -> float | str:
    source = source.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    source_count = int(source.sum())
    if source_count == 0:
        return NOT_APPLICABLE
    if not target.any():
        return 0.0
    distances = distance_transform_edt(~target)
    return float(np.count_nonzero(distances[source] <= max_distance) / source_count)


def write_outputs(
    out_dir: Path,
    image_path: Path,
    sample: dict[str, Any],
    predictions: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(image_path)
    np.savez_compressed(
        out_dir / f"{stem}_predictions.npz",
        semantic_class_map=predictions["semantic_class_map"],
        fibrous_probability=predictions["fibrous_probability"],
        clump_probability=predictions["clump_probability"],
        skeleton_probability=predictions["skeleton_probability"],
        skeleton_mask_0_5=predictions["skeleton_mask_0_5"],
        skeleton_mask_0_75=predictions["skeleton_mask_0_75"],
    )
    (out_dir / f"{stem}_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_qa_panel(out_dir / f"{stem}_qa_panel.png", sample, predictions)


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in path.stem).strip("_")


def write_qa_panel(path: Path, sample: dict[str, Any], predictions: dict[str, np.ndarray]) -> None:
    raw = sample["image_uint8"] if sample["image_uint8"] is not None else normalize_display(sample["image_float"])
    target_sem = sample["real_semantic_mask"]
    pred_sem = predictions["semantic_class_map"]
    target_fib = target_sem == 1
    pred_fib = pred_sem == 1
    target_skel = sample["real_skeleton_mask"] > 0
    pred_skel = predictions["skeleton_mask_0_75"] > 0
    tiles = [
        draw_tile(gray_rgb(raw), "raw input"),
        draw_tile(real_mask_rgb(target_sem), "expert semantic"),
        draw_tile(real_mask_rgb(pred_sem), "pred semantic"),
        draw_tile(overlay_mask(raw, target_fib, (0, 220, 80)), "expert fibrous"),
        draw_tile(overlay_mask(raw, pred_fib, (0, 220, 80)), "pred fibrous"),
        draw_tile(overlay_mask(raw, target_skel, (0, 220, 255)), "expert skeleton"),
        draw_tile(overlay_mask(raw, pred_skel, (255, 0, 255)), "pred skeleton"),
        draw_tile(heatmap(predictions["skeleton_probability"]), "skeleton prob"),
        draw_tile(error_rgb(raw, pred_fib, target_fib), "fibrous fp/fn"),
        draw_tile(error_rgb(raw, pred_skel, target_skel), "skeleton fp/fn"),
    ]
    tile_w, tile_h = tiles[0].size
    panel = Image.new("RGB", (5 * tile_w, 2 * tile_h), "white")
    for i, tile in enumerate(tiles):
        panel.paste(tile, ((i % 5) * tile_w, (i // 5) * tile_h))
    panel.save(path)


def normalize_display(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0 or float(arr.max()) == float(arr.min()):
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - float(arr.min())) / (float(arr.max()) - float(arr.min()))
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def gray_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = normalize_display(image)
    return np.repeat(image[..., None], 3, axis=2)


def real_mask_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 1] = (0, 220, 80)
    rgb[mask == 3] = (255, 160, 0)
    rgb[mask == 255] = (140, 90, 220)
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = gray_rgb(image).astype(np.float32)
    rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.asarray(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def heatmap(values: np.ndarray) -> np.ndarray:
    v = np.clip(values.astype(np.float32), 0, 1)
    rgb = np.zeros((*v.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(255 * v, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(255 * (1 - np.abs(v - 0.5) * 2), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(255 * (1 - v), 0, 255).astype(np.uint8)
    return rgb


def error_rgb(image: np.ndarray, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    rgb = gray_rgb(image).astype(np.float32) * 0.55
    tp = pred & target
    fp = pred & ~target
    fn = ~pred & target
    rgb[tp] = (0, 220, 80)
    rgb[fp] = (255, 60, 40)
    rgb[fn] = (40, 170, 255)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def draw_tile(arr: np.ndarray, title: str, max_side: int = 256) -> Image.Image:
    im = Image.fromarray(arr.astype(np.uint8), "RGB")
    scale = min(max_side / max(im.size), 1.0)
    if scale < 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (max_side, max_side + 18), "white")
    canvas.paste(im, ((max_side - im.width) // 2, 18 + (max_side - im.height) // 2))
    ImageDraw.Draw(canvas).text((4, 3), title, fill=(0, 0, 0))
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
