"""Diagnostics for thick coherent synthetic bundle targets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

def bundle_diagnostics_from_arrays(
    arrays: dict[str, np.ndarray],
    pred_semantic: np.ndarray | None = None,
) -> dict[str, Any]:
    bundle = bundle_mask(arrays)
    axis = arrays.get("bundle_axis_mask", np.zeros_like(bundle, dtype=np.uint8)).astype(bool)
    widths = bundle_widths(bundle, axis)
    coherence = bundle_orientation_coherence(bundle)
    out = {
        "bundle_pixel_count": int(bundle.sum()),
        "bundle_component_count": int(ndimage.label(bundle)[1]),
        "bundle_width_px_p25": percentile_or_na(widths, 25),
        "bundle_width_px_p50": percentile_or_na(widths, 50),
        "bundle_width_px_p75": percentile_or_na(widths, 75),
        "bundle_directionality_coherence_mean": mean_or_na(coherence),
        "bundle_directionality_coherence_p50": percentile_or_na(coherence, 50),
        "bundle_pixels_predicted_clump_fraction": "not_available",
        "bundle_pixels_predicted_clump_count": "not_available",
    }
    if pred_semantic is not None and bundle.shape == pred_semantic.shape and bundle.any():
        clump = pred_semantic == 2
        out["bundle_pixels_predicted_clump_count"] = int((bundle & clump).sum())
        out["bundle_pixels_predicted_clump_fraction"] = float((bundle & clump).sum() / bundle.sum())
    return out


def diagnostics_for_manifest(manifest_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_manifest(manifest_path)
    out = []
    for row in rows[:limit]:
        npz_path = resolve_path(row["npz_path"], manifest_path)
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        out.append({"sample_id": row["sample_id"], **bundle_diagnostics_from_arrays(arrays)})
    return out


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_path(path: str, manifest_path: Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    candidate = manifest_path.parent / p
    return candidate if candidate.exists() else p


def write_bundle_diagnostics(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "aggregates": aggregate(rows)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def bundle_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "bundle_mask" in arrays:
        return arrays["bundle_mask"].astype(bool)
    if "semantic_class_mask" in arrays:
        return arrays["semantic_class_mask"] == 2
    return np.zeros_like(arrays["real_compatible_semantic_mask"], dtype=bool)


def bundle_widths(bundle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    if not bundle.any():
        return np.zeros(0, dtype=np.float32)
    distance = ndimage.distance_transform_edt(bundle)
    samples = distance[axis & bundle] if axis.any() else distance[bundle]
    return (2.0 * samples).astype(np.float32)


def bundle_orientation_coherence(bundle: np.ndarray) -> np.ndarray:
    if not bundle.any():
        return np.zeros(0, dtype=np.float32)
    labels, count = ndimage.label(bundle)
    values = []
    for component_id in range(1, count + 1):
        ys, xs = np.nonzero(labels == component_id)
        if len(xs) < 3:
            continue
        coords = np.column_stack([xs, ys]).astype(np.float32)
        cov = np.cov(coords, rowvar=False)
        eig = np.linalg.eigvalsh(cov)
        denom = float(eig.sum())
        values.append(float((eig[-1] - eig[0]) / denom) if denom > 0 else 0.0)
    return np.asarray(values, dtype=np.float32)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "bundle_pixel_count",
        "bundle_component_count",
        "bundle_width_px_p50",
        "bundle_directionality_coherence_mean",
        "bundle_pixels_predicted_clump_fraction",
    ]
    return {key: mean_numeric([row.get(key) for row in rows]) for key in keys}


def percentile_or_na(values: np.ndarray, q: float) -> float | str:
    return float(np.percentile(values, q)) if values.size else "not_available"


def mean_or_na(values: np.ndarray) -> float | str:
    return float(np.mean(values)) if values.size else "not_available"


def mean_numeric(values: list[Any]) -> float | str:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return float(np.mean(numeric)) if numeric else "not_available"
