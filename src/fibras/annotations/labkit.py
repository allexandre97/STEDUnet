"""Minimal Labkit label loading for the real annotation pilot."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REAL_LABEL_VALUES = {
    "background": 0,
    "fibers": 1,
    "uncertain_ignore": 255,
    "clump": 3,
}

LABKIT_LABEL_ALIASES = {
    "fiber": "fibers",
    "fibres": "fibers",
    "filament": "fibers",
    "filaments": "fibers",
    "uncertain": "uncertain_ignore",
    "ignore": "uncertain_ignore",
    "unknown": "uncertain_ignore",
}


class LabkitLabelError(ValueError):
    pass


@dataclass(frozen=True)
class LabkitLabels:
    image_shape: tuple[int, int]
    label_names: list[str]
    canonical_label_names: list[str]
    masks: dict[str, np.ndarray]
    semantic_mask: np.ndarray
    warnings: list[str]


def load_real_labels(path: str | Path) -> LabkitLabels:
    path = Path(path)
    if path.suffix.lower() == ".labeling":
        return parse_labkit_labeling(path)
    return load_exported_integer_mask(path)


def parse_labkit_labeling(path: str | Path) -> LabkitLabels:
    """Load simple JSON `.labeling` files with per-label pixel coordinates."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LabkitLabelError(_export_mask_message(path)) from exc
    if not isinstance(data, dict) or not isinstance(data.get("labels"), dict):
        raise LabkitLabelError(_export_mask_message(path))

    shape = _interval_shape(data.get("interval"), path)
    masks = {name: np.zeros(shape, dtype=bool) for name in REAL_LABEL_VALUES}
    semantic = np.zeros(shape, dtype=np.uint8)
    label_names = list(data["labels"])
    canonical_names: list[str] = []
    warnings: list[str] = []

    for raw_name, coords in data["labels"].items():
        canonical = canonical_label_name(raw_name)
        canonical_names.append(canonical)
        if canonical not in {"fibers", "uncertain_ignore", "clump"}:
            warnings.append(f"ignored unknown Labkit label: {raw_name}")
            continue
        _paint_coordinates(masks[canonical], coords, path, raw_name)
        semantic[masks[canonical]] = REAL_LABEL_VALUES[canonical]

    return LabkitLabels(
        image_shape=shape,
        label_names=label_names,
        canonical_label_names=canonical_names,
        masks=masks,
        semantic_mask=semantic,
        warnings=warnings,
    )


def load_exported_integer_mask(path: str | Path) -> LabkitLabels:
    """Load a same-size integer mask with real labels 0, 1, 2, and 3."""
    path = Path(path)
    try:
        arr = np.asarray(Image.open(path))
    except Exception as exc:
        raise LabkitLabelError(
            f"{path}: could not read label mask. Provide a Labkit .labeling "
            "JSON file or a same-size exported integer mask."
        ) from exc
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise LabkitLabelError(f"{path}: exported integer mask must be 2D")

    raw = arr.astype(np.int32, copy=False)
    semantic = np.zeros(raw.shape, dtype=np.uint8)
    mapping = {0: 0, 1: 1, 2: 255, 3: 3, 255: 255}
    unknown = sorted(set(np.unique(raw).tolist()) - set(mapping))
    if unknown:
        raise LabkitLabelError(f"{path}: unsupported integer label values: {unknown}")
    for src, dst in mapping.items():
        semantic[raw == src] = dst
    masks = {
        "background": semantic == 0,
        "fibers": semantic == 1,
        "uncertain_ignore": semantic == 255,
        "clump": semantic == 3,
    }
    names = [name for name in ("fibers", "uncertain_ignore", "clump") if masks[name].any()]
    return LabkitLabels(raw.shape, names, names, masks, semantic, [])


def canonical_label_name(name: str) -> str:
    cleaned = name.strip().lower().replace(" ", "_").replace("-", "_")
    return LABKIT_LABEL_ALIASES.get(cleaned, cleaned)


def _interval_shape(interval: Any, path: Path) -> tuple[int, int]:
    if not isinstance(interval, dict):
        raise LabkitLabelError(_export_mask_message(path))
    mins = interval.get("min")
    maxes = interval.get("max")
    if not (
        isinstance(mins, list)
        and isinstance(maxes, list)
        and len(mins) == len(maxes) == 2
    ):
        raise LabkitLabelError(
            f"{path}: only 2D Labkit .labeling coordinate exports are supported. "
            "Export a same-size integer mask for other Labkit layouts."
        )
    width = int(maxes[0]) - int(mins[0]) + 1
    height = int(maxes[1]) - int(mins[1]) + 1
    if width <= 0 or height <= 0:
        raise LabkitLabelError(f"{path}: invalid Labkit interval")
    return (height, width)


def _paint_coordinates(mask: np.ndarray, coords: Any, path: Path, label: str) -> None:
    arr = np.asarray(coords, dtype=np.int64)
    if arr.size == 0:
        return
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise LabkitLabelError(
            f"{path}: label {label!r} is not a 2D coordinate list. "
            "Export a same-size integer mask instead."
        )
    x = arr[:, 0]
    y = arr[:, 1]
    in_bounds = (0 <= x) & (x < mask.shape[1]) & (0 <= y) & (y < mask.shape[0])
    if not np.all(in_bounds):
        raise LabkitLabelError(f"{path}: label {label!r} contains out-of-bounds pixels")
    mask[y, x] = True


def _export_mask_message(path: Path) -> str:
    return (
        f"{path}: unsupported Labkit .labeling layout. Export a same-size "
        "integer mask with labels 0 background, 1 fibers, 2 uncertain_ignore, "
        "and 3 clump."
    )
