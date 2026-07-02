"""Convert pilot real annotations into real-compatible target arrays."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fibras.synthetic.real_compatible import collapse_synthetic_semantic_mask
from fibras.synthetic.targets import draw_disk, rasterize_polyline

from .jfilament import SnakePoint, parse_jfilament_snakes
from .labkit import LabkitLabels, load_real_labels


REAL_CLASS_NAMES = {
    0: "background",
    1: "fibrous_tau",
    3: "clump",
    255: "uncertain_ignore",
}
REAL_LABEL_REMAP = {
    0: 0,
    1: 1,
    2: 255,
    3: 3,
}
SYNTHETIC_TO_REAL_COMPATIBLE = {
    0: 0,
    1: 1,
    2: 1,
    3: 3,
    255: 255,
}


def build_real_annotation_sample(
    image_path: str | Path,
    snakes_path: str | Path,
    labels_path: str | Path,
    *,
    skeleton_radius_px: float = 0.75,
    usable_threshold: float = 0.6,
) -> dict[str, Any]:
    image = read_image_array(image_path)
    labels = load_real_labels(labels_path)
    if image.shape[:2] != labels.image_shape:
        raise ValueError(
            f"image shape {image.shape[:2]} does not match label shape "
            f"{labels.image_shape}"
        )

    points = parse_jfilament_snakes(snakes_path)
    grouped = group_snake_points(points)
    skeleton = np.zeros(labels.image_shape, dtype=np.uint8)
    all_snakes = np.zeros(labels.image_shape, dtype=np.uint8)
    flags: list[dict[str, Any]] = []

    for snake_id, snake_points in grouped.items():
        xy = np.asarray([(point.x, point.y) for point in snake_points], dtype=np.float32)
        mask = rasterize_snake(xy, labels.image_shape, skeleton_radius_px)
        all_snakes[mask] = 1
        quality = classify_snake_mask(snake_id, mask, labels.semantic_mask, usable_threshold)
        flags.append(quality)
        if quality["usable_fibrous_skeleton"]:
            skeleton[mask] = 1

    xy, offsets, ids = snake_arrays(grouped)
    return {
        "image_uint8": image.astype(np.uint8, copy=False) if image.dtype == np.uint8 else None,
        "image_float": image.astype(np.float32, copy=False) if image.dtype != np.uint8 else None,
        "real_semantic_mask": labels.semantic_mask,
        "real_fibrous_mask": (labels.semantic_mask == 1).astype(np.uint8),
        "real_clump_mask": (labels.semantic_mask == 3).astype(np.uint8),
        "real_uncertain_ignore_mask": (labels.semantic_mask == 255).astype(np.uint8),
        "real_skeleton_mask": skeleton,
        "real_all_snakes_mask": all_snakes,
        "real_snake_points_xy": xy,
        "real_snake_point_offsets": offsets,
        "real_snake_ids": ids,
        "metadata": {
            "image_path": str(image_path),
            "snakes_path": str(snakes_path),
            "labels_path": str(labels_path),
            "image_shape": list(labels.image_shape),
            "label_names_found": labels.label_names,
            "canonical_label_names": labels.canonical_label_names,
            "snake_quality_flags": flags,
            "warnings": labels.warnings,
        },
    }


def read_image_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        arr = np.asarray(Image.open(path))
    except Exception as exc:
        raise ValueError(
            f"{path}: could not read image. Use a plain TIFF, OME-TIFF readable "
            "by the installed libraries, or PNG."
        ) from exc
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        arr = arr[..., :3].mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(
            f"{path}: expected a 2D grayscale image. Use a plain TIFF, "
            "OME-TIFF readable by the installed libraries, or PNG."
        )
    return arr


def collapse_synthetic_semantic_to_real(mask: np.ndarray) -> np.ndarray:
    return collapse_synthetic_semantic_mask(mask)


def group_snake_points(points: list[SnakePoint]) -> dict[int, list[SnakePoint]]:
    grouped: dict[int, list[SnakePoint]] = defaultdict(list)
    for point in points:
        grouped[point.snake_id].append(point)
    return {sid: sorted(pts, key=lambda p: p.point_index) for sid, pts in grouped.items()}


def snake_arrays(grouped: dict[int, list[SnakePoint]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    ids: list[int] = []
    offsets = [0]
    for snake_id in sorted(grouped):
        for point in grouped[snake_id]:
            points.append((point.x, point.y))
            ids.append(snake_id)
        offsets.append(len(points))
    return (
        np.asarray(points, dtype=np.float32).reshape((-1, 2)),
        np.asarray(offsets, dtype=np.int32),
        np.asarray(ids, dtype=np.int32),
    )


def rasterize_snake(points_xy: np.ndarray, shape: tuple[int, int], radius: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if points_xy.size == 0:
        return mask
    if len(points_xy) == 1:
        draw_disk(mask.view(np.uint8), points_xy[0], radius)
        return mask
    return rasterize_polyline(points_xy, shape, radius)


def classify_snake_mask(
    snake_id: int,
    mask: np.ndarray,
    semantic: np.ndarray,
    usable_threshold: float,
) -> dict[str, Any]:
    values = semantic[mask]
    if values.size == 0:
        return _snake_quality(snake_id, "flagged_out_of_bounds", {}, False)

    fractions = {
        REAL_CLASS_NAMES[value]: float(np.count_nonzero(values == value) / values.size)
        for value in REAL_CLASS_NAMES
    }
    dominant_value = max(REAL_CLASS_NAMES, key=lambda value: fractions[REAL_CLASS_NAMES[value]])
    dominant_fraction = fractions[REAL_CLASS_NAMES[dominant_value]]
    usable = dominant_value == 1 and dominant_fraction >= usable_threshold
    if usable:
        status = "usable_fibrous_skeleton"
    elif dominant_fraction < usable_threshold:
        status = "flagged_mixed_overlap"
    else:
        status = f"flagged_{REAL_CLASS_NAMES[dominant_value]}"
    return _snake_quality(snake_id, status, fractions, usable)


def _snake_quality(
    snake_id: int,
    status: str,
    fractions: dict[str, float],
    usable: bool,
) -> dict[str, Any]:
    return {
        "snake_id": int(snake_id),
        "status": status,
        "usable_fibrous_skeleton": bool(usable),
        "overlap_class_fractions": fractions,
    }
