"""Condition-blind spatial heterogeneity diagnostics and review panels."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from fibras.sted_inventory import read_image
from fibras.synthetic.visualization import rgb_depth

from .proxies import gradients
from .statistics import aggregate_numeric
from .visibility import (
    gray_rgb,
    read_csv,
    real_fiber_display_range,
    robust_display,
    scale_for_display,
    write_csv,
)


DIAGNOSTIC_FIELDS = [
    "foreground_occupancy",
    "individual_filament_area_fraction",
    "bundle_area_fraction",
    "clump_area_fraction",
    "ignore_area_fraction",
    "tile_occupancy_mean",
    "tile_occupancy_variance",
    "empty_tile_fraction",
    "spatial_concentration_index",
    "occupied_domain_count",
    "occupied_domain_area_p50_tiles",
    "orientation_coherence_p50",
    "orientation_coherence_p90",
    "filament_length_represented_px",
    "foreground_signal_p50",
    "foreground_signal_p95",
    "foreground_signal_p99",
    "clipping_fraction",
    "saturation_fraction",
]


def generated_scene_diagnostics(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    tile_size: int = 64,
) -> dict[str, str]:
    if "semantic_class_mask" in arrays:
        classes = arrays["semantic_class_mask"]
        foreground = np.isin(classes, [1, 2, 3])
        class_fractions = {
            "individual_filament_area_fraction": np.mean(classes == 1),
            "bundle_area_fraction": np.mean(classes == 2),
            "clump_area_fraction": np.mean(classes == 3),
            "ignore_area_fraction": np.mean(classes == 255),
        }
    else:
        foreground = arrays["semantic_mask"].astype(bool)
        class_fractions = {
            "individual_filament_area_fraction": np.mean(foreground),
            "bundle_area_fraction": 0.0,
            "clump_area_fraction": 0.0,
            "ignore_area_fraction": np.mean(
                arrays.get("ignore_mask", np.zeros_like(foreground))
            ),
        }
    tile = tile_occupancy(foreground, tile_size)
    domains = occupied_domains(tile > 0.01)
    coherence = generated_orientation_coherence(arrays, tile_size)
    signal = arrays.get(
        "synthetic_signal_float", arrays.get("total_clean_signal")
    ).astype(np.float32)
    positive = signal[signal > 0]
    return {
        "sample_id": metadata["sample_id"],
        "source_kind": "morphology_scene"
        if "semantic_class_mask" in arrays
        else "previous_uniform_synthetic",
        "scene_mode": metadata.get("scenario", "not_reported"),
        "foreground_occupancy": numeric(np.mean(foreground)),
        **{key: numeric(value) for key, value in class_fractions.items()},
        "tile_occupancy_mean": numeric(np.mean(tile)),
        "tile_occupancy_variance": numeric(np.var(tile)),
        "empty_tile_fraction": numeric(np.mean(tile < 0.005)),
        "spatial_concentration_index": numeric(concentration_index(tile)),
        "occupied_domain_count": str(len(domains)),
        "occupied_domain_area_p50_tiles": numeric(
            np.median(domains) if domains else 0
        ),
        "orientation_coherence_p50": numeric(
            np.percentile(coherence, 50) if coherence.size else 0
        ),
        "orientation_coherence_p90": numeric(
            np.percentile(coherence, 90) if coherence.size else 0
        ),
        "filament_length_represented_px": numeric(
            represented_filament_length(arrays)
        ),
        "foreground_signal_p50": numeric(
            np.percentile(positive, 50) if positive.size else 0
        ),
        "foreground_signal_p95": numeric(
            np.percentile(positive, 95) if positive.size else 0
        ),
        "foreground_signal_p99": numeric(
            np.percentile(positive, 99) if positive.size else 0
        ),
        "clipping_fraction": numeric(
            metadata.get("clipping_fraction", metadata.get("rendering_report", {}).get("clipping_fraction", 0))
        ),
        "saturation_fraction": numeric(
            metadata.get("saturation_fraction", metadata.get("rendering_report", {}).get("saturation_fraction", 0))
        ),
    }


def real_scene_diagnostics(
    image: np.ndarray,
    sample_id: str,
    source_kind: str,
    tile_size: int = 64,
) -> dict[str, str]:
    arr = image.astype(np.float32)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    threshold = max(
        float(np.percentile(arr, 99)), median + 6 * 1.4826 * mad
    )
    foreground = (arr >= threshold) & (arr > median)
    tile = tile_occupancy(foreground, tile_size)
    domains = occupied_domains(tile > 0.01)
    coherence = proxy_orientation_coherence(arr, foreground, tile_size)
    positive = arr[foreground]
    return {
        "sample_id": sample_id,
        "source_kind": source_kind,
        "scene_mode": "condition_blind_proxy",
        "foreground_occupancy": numeric(np.mean(foreground)),
        "individual_filament_area_fraction": "not_available",
        "bundle_area_fraction": "not_available",
        "clump_area_fraction": "not_available",
        "ignore_area_fraction": "not_available",
        "tile_occupancy_mean": numeric(np.mean(tile)),
        "tile_occupancy_variance": numeric(np.var(tile)),
        "empty_tile_fraction": numeric(np.mean(tile < 0.005)),
        "spatial_concentration_index": numeric(concentration_index(tile)),
        "occupied_domain_count": str(len(domains)),
        "occupied_domain_area_p50_tiles": numeric(
            np.median(domains) if domains else 0
        ),
        "orientation_coherence_p50": numeric(
            np.percentile(coherence, 50) if coherence.size else 0
        ),
        "orientation_coherence_p90": numeric(
            np.percentile(coherence, 90) if coherence.size else 0
        ),
        "filament_length_represented_px": "not_available",
        "foreground_signal_p50": numeric(
            np.percentile(positive, 50) if positive.size else 0
        ),
        "foreground_signal_p95": numeric(
            np.percentile(positive, 95) if positive.size else 0
        ),
        "foreground_signal_p99": numeric(
            np.percentile(positive, 99) if positive.size else 0
        ),
        "clipping_fraction": numeric(np.mean(arr > 255)),
        "saturation_fraction": numeric(np.mean(arr == 255)),
    }


def build_morphology_review(
    config: dict[str, Any],
    synthetic_dir: Path,
    composite_dir: Path,
    previous_dir: Path,
    artifact_dir: Path,
    out_dir: Path,
    diagnostics_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    full_dir = out_dir / "full_samples"
    full_dir.mkdir(exist_ok=True)
    shared_range = real_fiber_display_range(
        artifact_dir, Path(config["source_roots"]["sted_fiber_data"])
    )
    morphology_rows = dataset_diagnostics(composite_dir)
    previous_rows = dataset_diagnostics(previous_dir)
    real_rows = source_diagnostics(
        artifact_dir / "real_fiber_stats.csv",
        Path(config["source_roots"]["sted_fiber_data"]),
        "real_fiber_proxy",
    )
    blank_rows = source_diagnostics(
        artifact_dir / "blank_stats.csv",
        Path(config["source_roots"]["sted_blank_data"]),
        "real_blank_proxy",
    )
    all_rows = morphology_rows + previous_rows + real_rows + blank_rows
    write_csv(diagnostics_dir / "spatial_heterogeneity_diagnostics.csv", all_rows)
    summaries = {
        kind: aggregate_numeric(
            [row for row in all_rows if row["source_kind"] == kind],
            DIAGNOSTIC_FIELDS,
        )
        for kind in sorted({row["source_kind"] for row in all_rows})
    }
    mode_summaries = {
        mode: aggregate_numeric(
            [row for row in morphology_rows if row["scene_mode"] == mode],
            DIAGNOSTIC_FIELDS,
        )
        for mode in sorted({row["scene_mode"] for row in morphology_rows})
    }
    (diagnostics_dir / "spatial_heterogeneity_summary.json").write_text(
        json.dumps(
            {
                "calibration_status": "exploratory_unpartitioned",
                "condition_blind": True,
                "groups": summaries,
                "morphology_modes": mode_summaries,
                "warning": (
                    "Real-image quantities are threshold-sensitive proxies and "
                    "are not hard morphology targets."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = read_csv(composite_dir / "dataset_manifest.csv")
    sample_records = []
    for row in manifest:
        metadata = json.loads(
            (composite_dir / row["json_path"]).read_text(encoding="utf-8")
        )
        with np.load(
            composite_dir / row["npz_path"], allow_pickle=False
        ) as data:
            arrays = {name: data[name].copy() for name in data.files}
        output = full_dir / f"{row['sample_id']}_morphology.png"
        draw_morphology_panels(arrays, metadata, output, shared_range)
        sample_records.append((row["sample_id"], metadata["scenario"], arrays))
    draw_mode_contact_sheet(
        sample_records, out_dir / "morphology_modes_contact_sheet.png", shared_range
    )
    draw_zoom_contact_sheet(
        sample_records, out_dir / "morphology_zoom_regions.png", shared_range
    )
    draw_diagnostic_summary(
        summaries, out_dir / "spatial_heterogeneity_summary.png"
    )
    write_morphology_report(
        out_dir / "report.md",
        summaries,
        mode_summaries,
        shared_range,
        sample_records,
    )


def dataset_diagnostics(dataset_dir: Path) -> list[dict[str, str]]:
    rows = []
    for row in read_csv(dataset_dir / "dataset_manifest.csv"):
        metadata = json.loads(
            (dataset_dir / row["json_path"]).read_text(encoding="utf-8")
        )
        with np.load(
            dataset_dir / row["npz_path"], allow_pickle=False
        ) as data:
            arrays = {name: data[name].copy() for name in data.files}
        rows.append(generated_scene_diagnostics(arrays, metadata))
    return rows


def source_diagnostics(
    stats_path: Path, root: Path, source_kind: str
) -> list[dict[str, str]]:
    rows = []
    for row in read_csv(stats_path):
        image, _, _ = read_image(root / row["relative_path"])
        rows.append(
            real_scene_diagnostics(
                image, row["stable_image_id"], source_kind
            )
        )
    return rows


def tile_occupancy(mask: np.ndarray, tile_size: int) -> np.ndarray:
    height = mask.shape[0] // tile_size * tile_size
    width = mask.shape[1] // tile_size * tile_size
    blocks = mask[:height, :width].reshape(
        height // tile_size, tile_size, width // tile_size, tile_size
    )
    return blocks.mean(axis=(1, 3))


def concentration_index(tile: np.ndarray) -> float:
    values = np.sort(tile.ravel())[::-1]
    total = float(values.sum())
    if total <= 0:
        return 0.0
    count = max(1, int(math.ceil(0.1 * len(values))))
    return float(values[:count].sum() / total)


def occupied_domains(mask: np.ndarray) -> list[int]:
    labels, count = ndimage.label(mask)
    return [
        int(np.sum(labels == label)) for label in range(1, count + 1)
    ]


def generated_orientation_coherence(
    arrays: dict[str, np.ndarray], tile_size: int
) -> np.ndarray:
    if "orientation_valid_mask" not in arrays:
        return np.zeros(0, dtype=np.float32)
    return orientation_tiles(
        arrays["orientation_cos2theta"],
        arrays["orientation_sin2theta"],
        arrays["orientation_valid_mask"].astype(bool),
        tile_size,
    )


def proxy_orientation_coherence(
    image: np.ndarray, foreground: np.ndarray, tile_size: int
) -> np.ndarray:
    gx, gy = gradients(image)
    theta = np.arctan2(gy, gx) + np.pi / 2
    return orientation_tiles(
        np.cos(2 * theta),
        np.sin(2 * theta),
        foreground,
        tile_size,
    )


def orientation_tiles(
    cos2: np.ndarray,
    sin2: np.ndarray,
    valid: np.ndarray,
    tile_size: int,
) -> np.ndarray:
    values = []
    for y in range(0, valid.shape[0] - tile_size + 1, tile_size):
        for x in range(0, valid.shape[1] - tile_size + 1, tile_size):
            selected = valid[y : y + tile_size, x : x + tile_size]
            if np.count_nonzero(selected) < 3:
                continue
            c = cos2[y : y + tile_size, x : x + tile_size][selected]
            s = sin2[y : y + tile_size, x : x + tile_size][selected]
            values.append(float(np.sqrt(np.mean(c) ** 2 + np.mean(s) ** 2)))
    return np.asarray(values, dtype=np.float32)


def represented_filament_length(arrays: dict[str, np.ndarray]) -> float:
    weights = arrays.get("sample_arc_length_weight")
    if weights is None:
        return 0.0
    supervised = arrays.get(
        "fiber_supervised_centerline_sample",
        np.ones(weights.shape, dtype=np.uint8),
    ).astype(bool)
    return float(weights[supervised].sum())


def draw_morphology_panels(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    path: Path,
    shared_range: tuple[float, float],
) -> None:
    composite = arrays["render_uint8"]
    raw = scale_for_display(composite, 0, 255)
    shared = scale_for_display(composite, *shared_range)
    stretched, _ = robust_display(composite, (0.5, 99.5))
    signal, _ = robust_display(arrays["synthetic_signal_float"], (0.5, 99.5))
    blank = scale_for_display(arrays["blank_float"], *shared_range)
    panels = [
        ("raw composite 0–255", gray_rgb(raw)),
        ("shared STED scale", gray_rgb(shared)),
        ("independent p0.5–p99.5", gray_rgb(stretched)),
        ("semantic classes", class_overlay(shared, arrays)),
        ("filament centerlines", mask_overlay(shared, arrays["filament_centerline_mask"], (255, 255, 0))),
        ("bundle axes", mask_overlay(shared, arrays["bundle_axis_mask"], (255, 120, 0))),
        ("instances", instance_overlay(arrays)),
        ("depth projection", rgb_depth(arrays["weighted_mean_depth_map"], metadata)),
        ("foreground signal", gray_rgb(signal)),
        ("source blank", gray_rgb(blank)),
    ]
    tile = 224
    canvas = Image.new("RGB", (tile * 5, tile * 2 + 44), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (6, 5),
        f"{metadata['sample_id']} | {metadata['scenario']} | foreground scale "
        f"{metadata.get('applied_compositor_foreground_scale', 1):.3f}",
        fill="black",
    )
    for index, (title, panel) in enumerate(panels):
        x = index % 5 * tile
        y = index // 5 * tile + 28
        canvas.paste(
            Image.fromarray(panel).resize((tile, tile), Image.Resampling.BILINEAR),
            (x, y),
        )
        draw.rectangle((x, y, x + tile, y + 18), fill="black")
        draw.text((x + 3, y + 3), title, fill="white")
    canvas.save(path, optimize=True)


def class_overlay(base: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    rgb = gray_rgb(base).astype(np.float32)
    for name, color in [
        ("individual_filament_mask", (0, 220, 255)),
        ("bundle_mask", (255, 140, 0)),
        ("clump_mask", (255, 0, 100)),
        ("uncertain_ignore_mask", (170, 170, 170)),
    ]:
        mask = arrays[name].astype(bool)
        rgb[mask] = 0.35 * rgb[mask] + 0.65 * np.asarray(color)
    return rgb.astype(np.uint8)


def mask_overlay(
    base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    rgb = gray_rgb(base).astype(np.float32)
    selected = mask.astype(bool)
    rgb[selected] = 0.2 * rgb[selected] + 0.8 * np.asarray(color)
    return rgb.astype(np.uint8)


def instance_overlay(arrays: dict[str, np.ndarray]) -> np.ndarray:
    shape = arrays["semantic_class_mask"].shape
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    for prefix in ["individual_filament", "bundle", "clump"]:
        y = arrays[f"{prefix}_membership_y"]
        x = arrays[f"{prefix}_membership_x"]
        ids = arrays[f"{prefix}_membership_instance_id"]
        for instance_id in np.unique(ids):
            selected = ids == instance_id
            color = instance_color(int(instance_id), prefix)
            rgb[y[selected], x[selected]] = color
    return rgb


def instance_color(instance_id: int, prefix: str) -> tuple[int, int, int]:
    offset = {"individual_filament": 17, "bundle": 83, "clump": 149}[prefix]
    rng = np.random.default_rng(instance_id * 1009 + offset)
    return tuple(map(int, rng.integers(70, 256, 3)))


def draw_mode_contact_sheet(
    records: list[tuple[str, str, dict[str, np.ndarray]]],
    path: Path,
    shared_range: tuple[float, float],
) -> None:
    modes = [
        "isolated_filaments",
        "clustered_filament_network",
        "bundle_dominated",
        "clump_dominated",
        "mixed_morphology",
    ]
    selected = []
    for mode in modes:
        selected.extend([record for record in records if record[1] == mode][:2])
    tile = 192
    canvas = Image.new("RGB", (tile * 5, tile * 2 + 25), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), f"Shared STED scale {shared_range[0]:.3g}–{shared_range[1]:.3g}", fill="black")
    for index, (sample_id, mode, arrays) in enumerate(selected):
        panel = class_overlay(
            scale_for_display(arrays["render_uint8"], *shared_range), arrays
        )
        x, y = index % 5 * tile, index // 5 * tile + 22
        canvas.paste(Image.fromarray(panel).resize((tile, tile)), (x, y))
        draw.rectangle((x, y, x + tile, y + 18), fill="black")
        draw.text((x + 3, y + 3), f"{mode} | {sample_id[-4:]}", fill="white")
    canvas.save(path, optimize=True)


def draw_zoom_contact_sheet(
    records: list[tuple[str, str, dict[str, np.ndarray]]],
    path: Path,
    shared_range: tuple[float, float],
) -> None:
    targets = [
        ("bundle", "bundle_mask"),
        ("clump", "clump_mask"),
        ("filament→bundle", "bundle_transition_mask"),
        ("filament→clump", "clump_transition_mask"),
        ("dense crossing", "projected_crossing_map"),
    ]
    crops = []
    for label, mask_name in targets:
        for sample_id, _, arrays in records:
            if np.any(arrays[mask_name]):
                crops.append(
                    (
                        label,
                        sample_id,
                        crop_around_mask(
                            class_overlay(
                                scale_for_display(
                                    arrays["render_uint8"], *shared_range
                                ),
                                arrays,
                            ),
                            arrays[mask_name],
                            192,
                        ),
                    )
                )
                break
    tile = 220
    canvas = Image.new("RGB", (tile * len(crops), tile + 25), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), "Deterministic morphology zooms", fill="black")
    for index, (label, sample_id, crop) in enumerate(crops):
        x = index * tile
        canvas.paste(Image.fromarray(crop).resize((tile, tile)), (x, 22))
        draw.rectangle((x, 22, x + tile, 40), fill="black")
        draw.text((x + 3, 25), f"{label} | {sample_id[-4:]}", fill="white")
    canvas.save(path, optimize=True)


def crop_around_mask(
    image: np.ndarray, mask: np.ndarray, size: int
) -> np.ndarray:
    y, x = np.nonzero(mask)
    cy, cx = int(np.median(y)), int(np.median(x))
    half = size // 2
    y0 = int(np.clip(cy - half, 0, image.shape[0] - size))
    x0 = int(np.clip(cx - half, 0, image.shape[1] - size))
    return image[y0 : y0 + size, x0 : x0 + size]


def draw_diagnostic_summary(
    summaries: dict[str, dict[str, dict[str, float]]], path: Path
) -> None:
    groups = [
        "morphology_scene",
        "previous_uniform_synthetic",
        "real_fiber_proxy",
        "real_blank_proxy",
    ]
    metrics = [
        "tile_occupancy_variance",
        "empty_tile_fraction",
        "spatial_concentration_index",
        "orientation_coherence_p50",
    ]
    canvas = Image.new("RGB", (900, 430), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Spatial heterogeneity diagnostic medians", fill="black")
    colors = [(30, 120, 210), (70, 170, 80), (190, 70, 80), (120, 120, 120)]
    for metric_index, metric in enumerate(metrics):
        values = [
            summaries.get(group, {}).get(metric, {}).get("median", 0)
            for group in groups
        ]
        maximum = max(values) or 1
        x0 = 30 + metric_index * 215
        draw.text((x0, 40), metric, fill="black")
        for index, value in enumerate(values):
            height = int(260 * value / maximum)
            x = x0 + index * 42
            draw.rectangle((x, 335 - height, x + 28, 335), fill=colors[index])
    for index, group in enumerate(groups):
        draw.text((15 + index * 220, 385), group, fill=colors[index])
    canvas.save(path, optimize=True)


def write_morphology_report(
    path: Path,
    summaries: dict[str, dict[str, dict[str, float]]],
    mode_summaries: dict[str, dict[str, dict[str, float]]],
    shared_range: tuple[float, float],
    records: list[tuple[str, str, dict[str, np.ndarray]]],
) -> None:
    lines = [
        "# Synthetic STED morphology heterogeneity review",
        "",
        "- calibration_status: `exploratory_unpartitioned`",
        "- schema: `synthetic_sted_3d_morphology_0.6.0`",
        "- Morphology generation is condition-blind and broadly randomized.",
        "- Real-image measurements are coarse threshold-sensitive diagnostics, not fitted biological targets.",
        "- Composite foreground scale is sampled deterministically from `1.0–2.0`.",
        f"- Shared display range: `{shared_range[0]:.6g}` to `{shared_range[1]:.6g}`.",
        f"- Review samples: `{len(records)}`.",
        "",
        "## Median spatial diagnostics",
        "",
        "| Group | Foreground occupancy | Tile variance | Empty tiles | Concentration | Orientation coherence | Bundle fraction | Clump fraction |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for group in [
        "morphology_scene",
        "previous_uniform_synthetic",
        "real_fiber_proxy",
        "real_blank_proxy",
    ]:
        summary = summaries.get(group, {})
        value = lambda key: report_value(summary, key)
        lines.append(
            f"| `{group}` | {value('foreground_occupancy')} | "
            f"{value('tile_occupancy_variance')} | "
            f"{value('empty_tile_fraction')} | "
            f"{value('spatial_concentration_index')} | "
            f"{value('orientation_coherence_p50')} | "
            f"{value('bundle_area_fraction')} | "
            f"{value('clump_area_fraction')} |"
        )
    lines.extend(
        [
            "",
            "## Morphology-mode medians",
            "",
            "| Mode | Foreground occupancy | Tile variance | Empty tiles | Bundle fraction | Clump fraction |",
            "|:--|--:|--:|--:|--:|--:|",
        ]
    )
    for mode, summary in mode_summaries.items():
        value = lambda key: report_value(summary, key)
        lines.append(
            f"| `{mode}` | {value('foreground_occupancy')} | "
            f"{value('tile_occupancy_variance')} | "
            f"{value('empty_tile_fraction')} | "
            f"{value('bundle_area_fraction')} | "
            f"{value('clump_area_fraction')} |"
        )
    lines.extend(
        [
            "",
            "The heterogeneous generator is evaluated for increased spatial variation only. No table entry is a biological matching claim.",
            "",
            "## Review assets",
            "",
            "- [Morphology mode contact sheet](morphology_modes_contact_sheet.png)",
            "- [Morphology transition and structure zooms](morphology_zoom_regions.png)",
            "- [Spatial diagnostic summary](spatial_heterogeneity_summary.png)",
            "- Full local per-sample panels are generated under `full_samples/` and remain ignored by Git.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_value(summary: dict[str, dict[str, float]], key: str) -> str:
    value = summary.get(key, {}).get("median")
    return "n/a" if value is None else f"{value:.4g}"


def numeric(value: Any) -> str:
    return f"{float(value):.6g}"
