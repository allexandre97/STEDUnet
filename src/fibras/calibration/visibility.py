"""Display-only composite review helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .schema import git_provenance, sha256_file


RAW_DISPLAY_RANGE = (0.0, 255.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scale_for_display(
    image: np.ndarray, vmin: float, vmax: float
) -> np.ndarray:
    values = image.astype(np.float32)
    if vmax <= vmin:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - vmin) / (vmax - vmin) * 255, 0, 255).astype(
        np.uint8
    )


def robust_display(
    image: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5)
) -> tuple[np.ndarray, tuple[float, float]]:
    values = image.astype(np.float32)
    lo, hi = map(float, np.percentile(values, percentiles))
    return scale_for_display(values, lo, hi), (lo, hi)


def global_uint8_percentiles(
    paths: list[Path], percentiles: tuple[float, float] = (0.5, 99.5)
) -> tuple[float, float]:
    histogram = np.zeros(256, dtype=np.int64)
    for path in paths:
        values = np.asarray(Image.open(path), dtype=np.uint8)
        histogram += np.bincount(values.ravel(), minlength=256)
    if histogram.sum() == 0:
        raise ValueError("cannot derive display limits from an empty population")
    cumulative = np.cumsum(histogram)
    total = int(cumulative[-1])
    limits = []
    for percentile in percentiles:
        rank = percentile / 100.0 * max(total - 1, 0)
        limits.append(float(np.searchsorted(cumulative, rank + 1, side="left")))
    return limits[0], limits[1]


def real_fiber_display_range(
    artifact_dir: Path,
    source_root: Path,
    percentiles: tuple[float, float] = (0.5, 99.5),
) -> tuple[float, float]:
    rows = read_csv(artifact_dir / "real_fiber_stats.csv")
    return global_uint8_percentiles(
        [source_root / row["relative_path"] for row in rows], percentiles
    )


def composite_visibility_diagnostics(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> dict[str, str]:
    blank = arrays["blank_float"].astype(np.float32)
    composite = arrays["render_uint8"].astype(np.float32)
    signal = arrays["synthetic_signal_float"].astype(np.float32)
    blank_p50, blank_p95, blank_p99 = np.percentile(blank, [50, 95, 99])
    comp_p50, comp_p95, comp_p99 = np.percentile(composite, [50, 95, 99])
    threshold = float(
        metadata.get("rendering_report", {}).get(
            "visible_signal_threshold", 0.0
        )
    )
    return {
        "sample_id": metadata["sample_id"],
        "parent_synthetic_sample_id": metadata.get(
            "parent_synthetic_sample_id", "not_recorded"
        ),
        "source_blank_id": metadata["source_blank_provenance"][
            "blank_stable_image_id"
        ],
        "blank_p50": f"{blank_p50:.6g}",
        "blank_p95": f"{blank_p95:.6g}",
        "blank_p99": f"{blank_p99:.6g}",
        "composite_p50": f"{comp_p50:.6g}",
        "composite_p95": f"{comp_p95:.6g}",
        "composite_p99": f"{comp_p99:.6g}",
        "synthetic_foreground_max": f"{float(signal.max()):.6g}",
        "synthetic_foreground_integrated_signal": f"{float(signal.sum()):.6g}",
        "composite_minus_blank_p95": f"{comp_p95 - blank_p95:.6g}",
        "composite_minus_blank_p99": f"{comp_p99 - blank_p99:.6g}",
        "clipping_fraction": f"{float(metadata.get('clipping_fraction', 0)):.6g}",
        "saturation_fraction": f"{float(metadata.get('saturation_fraction', 0)):.6g}",
        "foreground_occupancy": f"{float(np.mean(signal > threshold)):.6g}",
        "foreground_occupancy_definition": (
            "fraction of pixels where synthetic_signal_float exceeds the "
            "configured visible_signal_threshold"
        ),
    }


def build_composite_visibility_review(
    composite_dir: Path,
    artifact_dir: Path,
    source_root: Path,
    out_dir: Path,
    *,
    shared_percentiles: tuple[float, float] = (0.5, 99.5),
    image_percentiles: tuple[float, float] = (0.5, 99.5),
) -> tuple[tuple[float, float], list[dict[str, str]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shared_range = real_fiber_display_range(
        artifact_dir, source_root, shared_percentiles
    )
    rows = read_csv(composite_dir / "dataset_manifest.csv")
    diagnostics = []
    index = [
        "# Composite visibility review",
        "",
        f"- Shared real-STED display range: `{shared_range[0]:.6g}` to `{shared_range[1]:.6g}`.",
        f"- Shared-range population percentiles: `p{shared_percentiles[0]}` to `p{shared_percentiles[1]}` over all pixels in the real-fiber calibration population.",
        f"- Independent per-image stretch: `p{image_percentiles[0]}` to `p{image_percentiles[1]}`; display only.",
        "- Raw panel range: `0` to `255`.",
        "",
    ]
    for row in rows:
        metadata = json.loads(
            (composite_dir / row["json_path"]).read_text(encoding="utf-8")
        )
        with np.load(
            composite_dir / row["npz_path"], allow_pickle=False
        ) as data:
            arrays = {name: data[name].copy() for name in data.files}
        record = composite_visibility_diagnostics(arrays, metadata)
        diagnostics.append(record)
        output = out_dir / f"{row['sample_id']}_visibility.png"
        draw_composite_visibility(
            arrays,
            metadata,
            output,
            shared_range=shared_range,
            image_percentiles=image_percentiles,
        )
        index.append(
            f"- `{row['sample_id']}`: [{output.name}]({output.name}); "
            f"parent `{record['parent_synthetic_sample_id']}`; blank "
            f"`{record['source_blank_id']}`."
        )
    write_csv(out_dir / "composite_visibility_diagnostics.csv", diagnostics)
    commit, dirty = git_provenance()
    metadata = {
        "source_commit_sha": commit,
        "working_tree_dirty": dirty,
        "source_composite_manifest_sha256": sha256_file(
            composite_dir / "dataset_manifest.csv"
        ),
        "raw_display_range": list(RAW_DISPLAY_RANGE),
        "shared_review_range": list(shared_range),
        "shared_review_percentiles": list(shared_percentiles),
        "shared_review_population": "all pixels from real_fiber_stats.csv source images",
        "per_image_stretch_percentiles": list(image_percentiles),
        "stored_arrays_modified": False,
    }
    (out_dir / "visualization_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    write_visibility_report(
        out_dir / "report.md",
        shared_range,
        shared_percentiles,
        image_percentiles,
        diagnostics,
    )
    return shared_range, diagnostics


def draw_composite_visibility(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    path: Path,
    *,
    shared_range: tuple[float, float],
    image_percentiles: tuple[float, float] = (0.5, 99.5),
) -> None:
    blank = arrays["blank_float"]
    signal = arrays["synthetic_signal_float"]
    composite = arrays["render_uint8"]
    blank_shared = scale_for_display(blank, *shared_range)
    foreground, foreground_range = robust_display(signal, image_percentiles)
    raw = scale_for_display(composite, *RAW_DISPLAY_RANGE)
    shared = scale_for_display(composite, *shared_range)
    stretched, stretched_range = robust_display(composite, image_percentiles)
    overlay = np.stack([shared, shared, shared], axis=2).astype(np.float32)
    blend_mask(overlay, arrays["semantic_mask"], (0, 120, 255), 0.3)
    blend_mask(overlay, arrays["centerline_mask"], (255, 255, 0), 0.9)
    panels = [
        ("source blank | shared scale", gray_rgb(blank_shared)),
        (
            f"synthetic foreground | robust {foreground_range[0]:.3g}–{foreground_range[1]:.3g}",
            gray_rgb(foreground),
        ),
        ("raw composite | fixed 0–255", gray_rgb(raw)),
        (
            f"shared STED scale | {shared_range[0]:.3g}–{shared_range[1]:.3g}",
            gray_rgb(shared),
        ),
        (
            f"independently contrast-stretched | {stretched_range[0]:.3g}–{stretched_range[1]:.3g}",
            gray_rgb(stretched),
        ),
        ("semantic cyan + centerline yellow", overlay.astype(np.uint8)),
    ]
    thumb = 256
    canvas = Image.new("RGB", (thumb * 3, thumb * 2 + 92), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        x = index % 3 * thumb
        y = index // 3 * thumb + 42
        canvas.paste(
            Image.fromarray(image).resize(
                (thumb, thumb), Image.Resampling.BILINEAR
            ),
            (x, y),
        )
        draw.rectangle((x, y, x + thumb, y + 20), fill=(0, 0, 0))
        draw.text((x + 4, y + 4), title, fill="white")
    draw.text(
        (8, 8),
        f"{metadata['sample_id']} | parent "
        f"{metadata.get('parent_synthetic_sample_id', 'not_recorded')} | blank "
        f"{metadata['source_blank_provenance']['blank_stable_image_id']}",
        fill="black",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def gray_rgb(image: np.ndarray) -> np.ndarray:
    return np.repeat(image[..., None], 3, axis=2).astype(np.uint8)


def blend_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    selected = mask.astype(bool)
    rgb[selected] = (1 - alpha) * rgb[selected] + alpha * np.asarray(color)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_visibility_report(
    path: Path,
    shared_range: tuple[float, float],
    shared_percentiles: tuple[float, float],
    image_percentiles: tuple[float, float],
    diagnostics: list[dict[str, str]],
) -> None:
    fields = [
        "blank_p50",
        "blank_p95",
        "blank_p99",
        "composite_p50",
        "composite_p95",
        "composite_p99",
        "synthetic_foreground_max",
        "synthetic_foreground_integrated_signal",
        "composite_minus_blank_p95",
        "composite_minus_blank_p99",
        "clipping_fraction",
        "saturation_fraction",
        "foreground_occupancy",
    ]
    medians = {
        field: float(np.median([float(row[field]) for row in diagnostics]))
        for field in fields
    }
    lines = [
        "# Composite visibility review",
        "",
        "- Display changes do not modify stored arrays.",
        "- Raw composite panels use fixed `0–255` limits.",
        f"- Shared real-STED range: `{shared_range[0]:.6g}` to `{shared_range[1]:.6g}`, derived from global real-fiber `p{shared_percentiles[0]}` and `p{shared_percentiles[1]}` over all pixels.",
        f"- Independent structural-inspection stretch: per-image `p{image_percentiles[0]}` to `p{image_percentiles[1]}`.",
        "- Foreground-only panels use a separately labelled robust stretch.",
        "",
        "## Current default median diagnostics",
        "",
    ]
    lines.extend(f"- `{field}`: `{value:.6g}`" for field, value in medians.items())
    lines.extend(
        [
            "",
            "## Samples",
            "",
            "- [Per-sample index](index.md)",
            "- [Diagnostics CSV](composite_visibility_diagnostics.csv)",
            "- [Visualization metadata](visualization_metadata.json)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
