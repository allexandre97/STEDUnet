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
        if "fiber_points_xyz" in arrays:
            review_path = out_dir / f"{row['sample_id']}_3d_review.png"
            draw_3d_review(arrays, metadata, review_path)
            index_lines.append(f"- `{row['sample_id']}`: `{out_path.name}`, `{review_path.name}`; metadata `{json_path}`")
        else:
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
    overlap = arrays.get(
        "supervised_overlap_count", arrays.get("overlap_count")
    )
    overlap_mask = overlap > 1
    overlay(rgb, overlap_mask.astype(np.uint8), (0, 255, 255), 0.7)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    points = arrays.get("trace_points_xy", arrays["fiber_points_xy"])
    offsets = arrays.get("trace_point_offsets", arrays["fiber_point_offsets"])
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


def draw_3d_review(arrays: dict[str, np.ndarray], metadata: dict, out_path: Path) -> None:
    zero = np.zeros_like(arrays["render_uint8"], dtype=np.uint8)
    panels = [
        ("render uint8", rgb_gray(arrays["render_uint8"])),
        ("total clean signal", rgb_scaled(arrays["total_clean_signal"])),
        ("in-focus signal", rgb_scaled(arrays["in_focus_signal"])),
        ("out-of-focus signal", rgb_scaled(arrays["out_of_focus_signal"])),
        ("projection mask", rgb_mask(arrays["projection_mask"], (0, 160, 255))),
        ("in-focus mask", rgb_mask(arrays["in_focus_mask"], (0, 230, 80))),
        ("visible mask", rgb_mask(arrays["visible_signal_mask"], (255, 180, 0))),
        ("weighted depth", rgb_depth(arrays["weighted_mean_depth_map"], metadata)),
        ("semantic mask", rgb_mask(arrays["semantic_mask"], (0, 110, 255))),
        ("centerline mask", rgb_mask(arrays["centerline_mask"], (255, 255, 0))),
        ("crossing map", rgb_mask(arrays.get("projected_crossing_map", zero), (255, 0, 255))),
        ("ignore mask", rgb_mask(arrays.get("ignore_mask", zero), (255, 80, 0))),
    ]
    if "background_distance_to_semantic_foreground" in arrays:
        panels.append(("background distance", rgb_scaled(arrays["background_distance_to_semantic_foreground"])))
    if "orientation_valid_mask" in arrays:
        panels.append(("orientation valid", rgb_mask(arrays["orientation_valid_mask"], (100, 255, 100))))
    thumb = 256
    rows = int(np.ceil(len(panels) / 4))
    canvas = Image.new("RGB", (thumb * 4, thumb * rows + 74), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, img) in enumerate(panels):
        x = (i % 4) * thumb
        y = (i // 4) * thumb + 32
        canvas.paste(Image.fromarray(img).resize((thumb, thumb), Image.Resampling.BILINEAR), (x, y))
        draw.text((x + 6, y + 6), title, fill=(255, 255, 255))
    draw.text((8, 8), f"{metadata['sample_id']} | {metadata.get('generator_mode')} | schema {metadata['dataset_schema_version']}", fill="black")
    footer_y = thumb * rows + 42
    draw_depth_projection(draw, arrays, offset=(8, footer_y), width=450, height=26)
    width = metadata.get("width_calibration", {}).get("measured_fwhm_px")
    if width is not None:
        draw.text((480, footer_y), f"FWHM calibration: {float(width):.3f} px", fill="black")
    else:
        draw.text((480, footer_y), f"schema: {metadata.get('dataset_schema_version', 'not_reported')}", fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def rgb_gray(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.uint8)
    return np.stack([a, a, a], axis=2)


def rgb_scaled(arr: np.ndarray) -> np.ndarray:
    vals = arr.astype(np.float32)
    hi = float(np.percentile(vals, 99.5))
    lo = float(np.percentile(vals, 1.0))
    scaled = np.zeros_like(vals, dtype=np.uint8) if hi <= lo else np.clip((vals - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return np.stack([scaled, scaled, scaled], axis=2)


def rgb_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return rgb


def rgb_depth(depth: np.ndarray, metadata: dict) -> np.ndarray:
    zmax = max(float(metadata.get("volume_depth_px", 1.0)), 1.0)
    valid = depth >= 0
    scaled = np.zeros_like(depth, dtype=np.float32)
    scaled[valid] = np.clip(depth[valid] / zmax, 0, 1)
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.uint8(scaled * 255)
    rgb[..., 2] = np.uint8((1 - scaled) * 255) * valid.astype(np.uint8)
    rgb[..., 1] = np.uint8(valid) * 60
    return rgb


def draw_depth_projection(draw: ImageDraw.ImageDraw, arrays: dict[str, np.ndarray], offset: tuple[int, int], width: int, height: int) -> None:
    points = arrays["fiber_points_xyz"]
    offsets = arrays["fiber_point_offsets"]
    if points.size == 0:
        return
    zmin, zmax = float(np.min(points[:, 2])), float(np.max(points[:, 2]))
    scale = max(zmax - zmin, 1e-6)
    x0, y0 = offset
    draw.rectangle((x0, y0, x0 + width, y0 + height), outline="black")
    for start, end in zip(offsets[:-1], offsets[1:]):
        pts = points[start:end]
        if len(pts) < 2:
            continue
        for a, b in zip(pts[:-1], pts[1:]):
            color = (int(255 * (a[2] - zmin) / scale), 80, int(255 * (zmax - a[2]) / scale))
            draw.line((x0 + int(a[0] / 1024 * width), y0 + int(a[1] / 1024 * height), x0 + int(b[0] / 1024 * width), y0 + int(b[1] / 1024 * height)), fill=color)
