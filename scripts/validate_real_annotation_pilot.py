#!/usr/bin/env python
"""Validate one pilot real annotation bundle and write a small visual report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.annotations import build_real_annotation_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--snakes", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    try:
        sample = build_real_annotation_sample(args.image, args.snakes, args.labels)
    except Exception as exc:
        print(f"Warning: {exc}", file=sys.stderr)
        print("Real annotation pilot validation failed.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    summary = build_summary(sample)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    make_overlay(sample).save(args.out / "overlay.png")
    print(f"Wrote {args.out / 'summary.json'}")
    print(f"Wrote {args.out / 'overlay.png'}")
    return 0


def build_summary(sample: dict) -> dict:
    semantic = sample["real_semantic_mask"]
    flags = sample["metadata"]["snake_quality_flags"]
    return {
        "image_shape": sample["metadata"]["image_shape"],
        "label_names_found": sample["metadata"]["label_names_found"],
        "canonical_label_names": sample["metadata"]["canonical_label_names"],
        "pixel_counts_per_class": {
            "background": int(np.count_nonzero(semantic == 0)),
            "fibrous_tau": int(np.count_nonzero(semantic == 1)),
            "clump": int(np.count_nonzero(semantic == 3)),
            "uncertain_ignore": int(np.count_nonzero(semantic == 255)),
        },
        "snake_count": int(len(sample["real_snake_point_offsets"]) - 1),
        "snake_point_count": int(len(sample["real_snake_points_xy"])),
        "snake_overlap_class_fractions": {
            str(flag["snake_id"]): flag["overlap_class_fractions"] for flag in flags
        },
        "usable_fibrous_snake_count": int(
            sum(flag["usable_fibrous_skeleton"] for flag in flags)
        ),
        "flagged_snake_count": int(
            sum(not flag["usable_fibrous_skeleton"] for flag in flags)
        ),
        "warnings": sample["metadata"]["warnings"]
        + [flag["status"] for flag in flags if not flag["usable_fibrous_skeleton"]],
    }


def make_overlay(sample: dict) -> Image.Image:
    image = sample["image_uint8"]
    if image is None:
        image = normalize_uint8(sample["image_float"])
    rgb = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
    blend(rgb, sample["real_fibrous_mask"].astype(bool), (0, 255, 90), 0.35)
    blend(rgb, sample["real_clump_mask"].astype(bool), (255, 80, 0), 0.4)
    blend(rgb, sample["real_uncertain_ignore_mask"].astype(bool), (160, 80, 255), 0.4)
    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    draw_snakes(out, sample)
    return out


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0 or float(arr.max()) == float(arr.min()):
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - float(arr.min())) / (float(arr.max()) - float(arr.min()))
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def blend(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.asarray(color, dtype=np.float32)


def draw_snakes(image: Image.Image, sample: dict) -> None:
    draw = ImageDraw.Draw(image)
    points = sample["real_snake_points_xy"]
    offsets = sample["real_snake_point_offsets"]
    flags = {flag["snake_id"]: flag for flag in sample["metadata"]["snake_quality_flags"]}
    for snake_id, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        xy = [tuple(map(float, point)) for point in points[start:stop]]
        if not xy:
            continue
        usable = flags.get(snake_id, {}).get("usable_fibrous_skeleton", False)
        color = (0, 220, 255) if usable else (255, 40, 40)
        if len(xy) == 1:
            x, y = xy[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color, width=2)
        else:
            draw.line(xy, fill=color, width=2)


if __name__ == "__main__":
    raise SystemExit(main())
