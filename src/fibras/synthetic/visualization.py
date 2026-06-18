"""Pillow-based visual checks for MVP synthetic samples."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def visualize_dataset(dataset_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dataset_dir / "dataset_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index_lines = ["# Synthetic MVP visualizations", ""]
    for row in rows:
        npz_path = dataset_dir / row["npz_path"]
        json_path = dataset_dir / row["json_path"]
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        out_path = out_dir / f"{row['sample_id']}_overlay.png"
        draw_overlay(arrays, out_path)
        index_lines.append(f"- `{row['sample_id']}`: `{out_path.name}`; metadata `{json_path}`")
        metadata["visualization"] = str(out_path)
    (out_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def draw_overlay(arrays: dict[str, np.ndarray], out_path: Path) -> None:
    base = arrays["render_uint8"]
    rgb = np.stack([base, base, base], axis=2).astype(np.float32)
    overlay(rgb, arrays["semantic_mask"], (0, 110, 255), 0.25)
    overlay(rgb, arrays["centerline_mask"], (255, 255, 0), 0.8)
    overlay(rgb, arrays["endpoint_map"], (0, 255, 0), 0.9)
    overlay(rgb, arrays["junction_map"], (255, 0, 0), 0.9)
    overlay(rgb, arrays["crossing_map"], (255, 0, 255), 0.9)
    overlap_mask = arrays["overlap_count"] > 1
    overlay(rgb, overlap_mask.astype(np.uint8), (0, 255, 255), 0.7)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    points = arrays["fiber_points_xy"]
    offsets = arrays["fiber_point_offsets"]
    for start, end in zip(offsets[:-1], offsets[1:]):
        xy = [tuple(map(float, p)) for p in points[start:end]]
        if len(xy) > 1:
            draw.line(xy, fill=(255, 180, 0), width=1)
    image.save(out_path)


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    idx = mask.astype(bool)
    if not np.any(idx):
        return
    rgb[idx] = (1 - alpha) * rgb[idx] + alpha * np.asarray(color, dtype=np.float32)

