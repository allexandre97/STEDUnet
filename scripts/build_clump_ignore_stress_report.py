#!/usr/bin/env python
"""Build QA metrics and panels for schema-0.8 clump/ignore stress samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
try:
    from scipy import ndimage as scipy_ndimage
except Exception:  # pragma: no cover - report still works without scipy.
    scipy_ndimage = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", required=True, type=Path)
    parser.add_argument("--composite-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--panel-count", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    synthetic = dataset_summary(args.synthetic_dir)
    composite = dataset_summary(args.composite_dir)
    write_panels(args.composite_dir, args.out / "panels", args.panel_count)
    summary = {"synthetic": synthetic, "composite": composite}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    print(f"wrote {args.out / 'report.md'}")
    print(f"wrote {args.out / 'summary.json'}")
    return 0


def dataset_summary(dataset_dir: Path) -> dict[str, Any]:
    rows = read_manifest(dataset_dir)
    metrics = []
    for row in rows:
        metadata = json.loads((dataset_dir / row["json_path"]).read_text(encoding="utf-8"))
        with np.load(dataset_dir / row["npz_path"], allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        metrics.append(sample_metrics(arrays, metadata))
    return {
        "dataset_dir": str(dataset_dir),
        "sample_count": len(metrics),
        "scenario_counts": count_values(metrics, "scenario"),
        "scenario_category_counts": count_values(metrics, "scenario_category"),
        "sample_variant_counts": count_values(metrics, "sample_variant"),
        "mean": aggregate(metrics),
        "distribution": distributions(metrics),
        "max_leakage": max_leakage(metrics),
    }


def sample_metrics(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    metadata = metadata or {}
    semantic = arrays["real_compatible_semantic_mask"]
    fibrous = arrays["real_compatible_fibrous_mask"].astype(bool)
    clump = arrays["real_compatible_clump_mask"].astype(bool)
    uncertain = arrays["real_compatible_uncertain_ignore_mask"].astype(bool)
    skeleton = arrays["real_compatible_skeleton_mask"].astype(bool)
    render = arrays["render_uint8"].astype(np.float32)
    total = semantic.size
    high_signal = render >= np.percentile(render, 95)
    patch = patch_metrics(clump, uncertain, skeleton, 128)
    return {
        "scenario": str(metadata.get("scenario", "not_reported")),
        "scenario_category": str(
            metadata.get(
                "scenario_category",
                metadata.get("rendering_report", {}).get(
                    "scenario_category", "not_reported"
                ),
            )
        ),
        "sample_variant": str(
            metadata.get("generation_config", {}).get(
                "sample_variant", "not_reported"
            )
        ),
        "clump_area_fraction": float(clump.sum() / total),
        "largest_clump_component_fraction": largest_component_fraction(clump),
        "uncertain_ignore_fraction": float(uncertain.sum() / total),
        "skeleton_pixels_inside_clump": int((skeleton & clump).sum()),
        "skeleton_pixels_inside_uncertain_ignore": int((skeleton & uncertain).sum()),
        "fibrous_tau_pixels_inside_clump": int((fibrous & clump).sum()),
        "fibrous_tau_pixels_inside_uncertain_ignore": int((fibrous & uncertain).sum()),
        "clump_high_signal_fraction": fraction((clump & high_signal).sum(), clump.sum()),
        "uncertain_ignore_high_signal_fraction": fraction((uncertain & high_signal).sum(), uncertain.sum()),
        "source_support_consistency": {
            "fibrous_overlaps_clump": int((fibrous & clump).sum()),
            "fibrous_overlaps_uncertain_ignore": int((fibrous & uncertain).sum()),
            "skeleton_overlaps_clump": int((skeleton & clump).sum()),
            "skeleton_overlaps_uncertain_ignore": int((skeleton & uncertain).sum()),
        },
        "intensity_p50": float(np.percentile(render, 50)),
        "intensity_p95": float(np.percentile(render, 95)),
        "intensity_p99": float(np.percentile(render, 99)),
        "clump_region_intensity": percentiles(render[clump]),
        "uncertain_ignore_region_intensity": percentiles(render[uncertain]),
        "background_region_intensity": percentiles(render[semantic == 0]),
        **patch,
    }


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "clump_area_fraction",
        "uncertain_ignore_fraction",
        "clump_high_signal_fraction",
        "uncertain_ignore_high_signal_fraction",
        "intensity_p50",
        "intensity_p95",
        "intensity_p99",
        "largest_clump_component_fraction",
        "max_patch_clump_fraction",
        "patch_fraction_clump_gt_25",
        "patch_fraction_clump_gt_50",
        "patch_fraction_uncertain_gt_10",
        "patch_skeleton_pixels_inside_clump",
        "patch_skeleton_pixels_inside_uncertain_ignore",
    ]
    count_keys = [
        "skeleton_pixels_inside_clump",
        "skeleton_pixels_inside_uncertain_ignore",
        "fibrous_tau_pixels_inside_clump",
        "fibrous_tau_pixels_inside_uncertain_ignore",
    ]
    out = {key: float(np.mean([m[key] for m in metrics])) if metrics else 0.0 for key in keys}
    out.update({key: int(sum(int(m[key]) for m in metrics)) for key in count_keys})
    for prefix in ["clump_region_intensity", "uncertain_ignore_region_intensity", "background_region_intensity"]:
        for percentile in ["p50", "p95", "p99"]:
            key = f"{prefix}_{percentile}"
            out[key] = float(np.mean([m[prefix][percentile] for m in metrics])) if metrics else 0.0
    return out


def distributions(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "clump_area_fraction": summary_values(metrics, "clump_area_fraction"),
        "largest_clump_component_fraction": summary_values(metrics, "largest_clump_component_fraction"),
        "uncertain_ignore_fraction": summary_values(metrics, "uncertain_ignore_fraction"),
        "largest_clump_gt_5_fraction": fraction(
            sum(m["largest_clump_component_fraction"] > 0.05 for m in metrics),
            len(metrics),
        ),
        "largest_clump_gt_10_fraction": fraction(
            sum(m["largest_clump_component_fraction"] > 0.10 for m in metrics),
            len(metrics),
        ),
        "no_clump_sample_fraction": fraction(
            sum(m["clump_area_fraction"] == 0 for m in metrics), len(metrics)
        ),
        "clump_sample_fraction": fraction(
            sum(m["clump_area_fraction"] > 0 for m in metrics), len(metrics)
        ),
        "clump_ignore_stress_sample_fraction": fraction(
            sum(m["scenario_category"] == "clump_ignore_stress" for m in metrics),
            len(metrics),
        ),
    }


def count_values(metrics: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for metric in metrics:
        value = str(metric.get(key, "not_reported"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def summary_values(metrics: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([m[key] for m in metrics], dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def largest_component_fraction(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    if scipy_ndimage is None:
        return float(mask.sum() / mask.size)
    labels, count = scipy_ndimage.label(mask)
    if count == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max() / mask.size)


def patch_metrics(
    clump: np.ndarray,
    uncertain: np.ndarray,
    skeleton: np.ndarray,
    patch_size: int,
) -> dict[str, Any]:
    clump_fracs = window_fractions(clump, patch_size)
    uncertain_fracs = window_fractions(uncertain, patch_size)
    return {
        "max_patch_clump_fraction": float(clump_fracs.max(initial=0.0)),
        "patch_count": int(clump_fracs.size),
        "patch_fraction_clump_gt_25": fraction(np.count_nonzero(clump_fracs > 0.25), clump_fracs.size),
        "patch_fraction_clump_gt_50": fraction(np.count_nonzero(clump_fracs > 0.50), clump_fracs.size),
        "patch_fraction_uncertain_gt_10": fraction(np.count_nonzero(uncertain_fracs > 0.10), uncertain_fracs.size),
        "patch_skeleton_pixels_inside_clump": int((skeleton & clump).sum()),
        "patch_skeleton_pixels_inside_uncertain_ignore": int((skeleton & uncertain).sum()),
    }


def window_fractions(mask: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = mask.shape
    ys = list(range(0, max(h - patch_size + 1, 1), patch_size))
    xs = list(range(0, max(w - patch_size + 1, 1), patch_size))
    if ys[-1] != h - patch_size:
        ys.append(max(h - patch_size, 0))
    if xs[-1] != w - patch_size:
        xs.append(max(w - patch_size, 0))
    area = patch_size * patch_size
    return np.asarray(
        [mask[y : y + patch_size, x : x + patch_size].sum() / area for y in ys for x in xs],
        dtype=np.float32,
    )


def max_leakage(metrics: list[dict[str, Any]]) -> dict[str, int]:
    keys = [
        "skeleton_pixels_inside_clump",
        "skeleton_pixels_inside_uncertain_ignore",
        "fibrous_tau_pixels_inside_clump",
        "fibrous_tau_pixels_inside_uncertain_ignore",
    ]
    return {key: max([int(m[key]) for m in metrics], default=0) for key in keys}


def write_panels(dataset_dir: Path, out_dir: Path, count: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in read_manifest(dataset_dir):
        metadata = json.loads((dataset_dir / row["json_path"]).read_text(encoding="utf-8"))
        with np.load(dataset_dir / row["npz_path"], allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        rows.append((row, arrays, sample_metrics(arrays, metadata)))
    for row, arrays, _ in representative_rows(rows, count):
        panel = sample_panel(arrays)
        panel.save(out_dir / f"{row['sample_id']}_clump_ignore_qa.png")


def representative_rows(rows: list[tuple[dict[str, str], dict[str, np.ndarray], dict[str, Any]]], count: int) -> list[tuple[dict[str, str], dict[str, np.ndarray], dict[str, Any]]]:
    selected: list[tuple[dict[str, str], dict[str, np.ndarray], dict[str, Any]]] = []
    selectors = [
        lambda item: item[2]["clump_area_fraction"] == 0,
        lambda item: 0 < item[2]["clump_area_fraction"] < 0.02,
        lambda item: item[2]["clump_area_fraction"] >= 0.05,
        lambda item: item[2]["max_patch_clump_fraction"] >= 0.5,
        lambda item: item[2]["patch_fraction_uncertain_gt_10"] > 0,
        lambda item: item[2]["scenario_category"] == "clump_ignore_stress",
    ]
    for selector in selectors:
        for item in rows:
            if selector(item) and item not in selected:
                selected.append(item)
                break
    for item in sorted(rows, key=lambda x: x[2]["clump_area_fraction"], reverse=True):
        if len(selected) >= count:
            break
        if item not in selected:
            selected.append(item)
    return selected[:count]


def sample_panel(arrays: dict[str, np.ndarray]) -> Image.Image:
    render = arrays["render_uint8"]
    panels = [
        ("raw/composite", gray(render)),
        ("semantic", semantic_rgb(arrays["real_compatible_semantic_mask"])),
        ("clump", mask_rgb(arrays["real_compatible_clump_mask"], (255, 130, 0))),
        ("uncertain", mask_rgb(arrays["real_compatible_uncertain_ignore_mask"], (170, 90, 255))),
        ("skeleton", mask_rgb(arrays["real_compatible_skeleton_mask"], (255, 0, 255))),
        ("clump fragments", latent_fragment_rgb(arrays)),
        ("high clump crop", high_fraction_crop(render, arrays["real_compatible_clump_mask"])),
        ("transition crop", high_fraction_crop(render, arrays["real_compatible_uncertain_ignore_mask"])),
        ("source support", source_support_rgb(arrays)),
    ]
    tile = 192
    canvas = Image.new("RGB", (4 * tile, 2 * (tile + 18)), "white")
    for i, (title, arr) in enumerate(panels):
        img = Image.fromarray(arr.astype(np.uint8), "RGB").resize((tile, tile), Image.Resampling.BILINEAR)
        x = (i % 4) * tile
        y = (i // 4) * (tile + 18)
        ImageDraw.Draw(canvas).text((x + 4, y + 3), title, fill=(0, 0, 0))
        canvas.paste(img, (x, y + 18))
    return canvas


def gray(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.uint8)
    return np.repeat(a[..., None], 3, axis=2)


def semantic_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 1] = (0, 220, 80)
    rgb[mask == 3] = (255, 130, 0)
    rgb[mask == 255] = (170, 90, 255)
    return rgb


def mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask.astype(bool)] = color
    return rgb


def clump_ignore_overlay(render: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    rgb = gray(render).astype(np.float32)
    blend(rgb, arrays["real_compatible_clump_mask"].astype(bool), (255, 130, 0), 0.5)
    blend(rgb, arrays["real_compatible_uncertain_ignore_mask"].astype(bool), (170, 90, 255), 0.5)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def source_support_rgb(arrays: dict[str, np.ndarray]) -> np.ndarray:
    shape = arrays["render_uint8"].shape
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    for name, color in [
        ("individual_filament_source_support_mask", (0, 220, 80)),
        ("bundle_source_support_mask", (0, 170, 255)),
        ("clump_source_support_mask", (255, 130, 0)),
    ]:
        if name in arrays:
            rgb[arrays[name].astype(bool)] = color
    return rgb


def latent_fragment_rgb(arrays: dict[str, np.ndarray]) -> np.ndarray:
    rgb = gray(arrays["render_uint8"]).astype(np.float32)
    ids = set(map(int, arrays.get("clump_fragment_fiber_ids", [])))
    if not ids:
        return rgb.astype(np.uint8)
    fiber_ids = arrays["visible_membership_instance_id"]
    y = arrays["visible_membership_y"]
    x = arrays["visible_membership_x"]
    mask = np.isin(fiber_ids, list(ids))
    rgb[y[mask], x[mask]] = (255, 170, 0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def high_fraction_crop(render: np.ndarray, mask: np.ndarray, patch_size: int = 128) -> np.ndarray:
    h, w = mask.shape
    best = (0, 0, -1)
    for y in range(0, max(h - patch_size + 1, 1), patch_size):
        for x in range(0, max(w - patch_size + 1, 1), patch_size):
            score = int(mask[y : y + patch_size, x : x + patch_size].sum())
            if score > best[2]:
                best = (y, x, score)
    y, x, _ = best
    crop = render[y : y + patch_size, x : x + patch_size]
    return gray(crop)


def blend(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.asarray(color, dtype=np.float32)


def fraction(num: Any, denom: Any) -> float:
    denom_i = int(denom)
    return 0.0 if denom_i == 0 else float(int(num) / denom_i)


def read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    with (dataset_dir / "dataset_manifest.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def markdown_report(summary: dict[str, Any]) -> str:
    lines = ["# Synthetic Clump/Ignore Stress QA", ""]
    for key in ["synthetic", "composite"]:
        mean = summary[key]["mean"]
        dist = summary[key]["distribution"]
        lines += [
            f"## {key.title()}",
            "",
            f"- Samples: {summary[key]['sample_count']}",
            f"- Scenario counts: `{json.dumps(summary[key]['scenario_counts'], sort_keys=True)}`",
            f"- Scenario category counts: `{json.dumps(summary[key]['scenario_category_counts'], sort_keys=True)}`",
            f"- Sample variant counts: `{json.dumps(summary[key]['sample_variant_counts'], sort_keys=True)}`",
            f"- Mean clump area fraction: {mean['clump_area_fraction']:.4f}",
            f"- Clump area fraction mean/p50/p95/max: {dist['clump_area_fraction']['mean']:.4f}/{dist['clump_area_fraction']['p50']:.4f}/{dist['clump_area_fraction']['p95']:.4f}/{dist['clump_area_fraction']['max']:.4f}",
            f"- Largest clump component fraction p50/p95/max: {dist['largest_clump_component_fraction']['p50']:.4f}/{dist['largest_clump_component_fraction']['p95']:.4f}/{dist['largest_clump_component_fraction']['max']:.4f}",
            f"- Samples with largest clump >5%/>10% image area: {dist['largest_clump_gt_5_fraction']:.3f}/{dist['largest_clump_gt_10_fraction']:.3f}",
            f"- Uncertain_ignore fraction mean/p50/p95/max: {dist['uncertain_ignore_fraction']['mean']:.4f}/{dist['uncertain_ignore_fraction']['p50']:.4f}/{dist['uncertain_ignore_fraction']['p95']:.4f}/{dist['uncertain_ignore_fraction']['max']:.4f}",
            f"- Samples with no clumps/with clumps/stress category: {dist['no_clump_sample_fraction']:.3f}/{dist['clump_sample_fraction']:.3f}/{dist['clump_ignore_stress_sample_fraction']:.3f}",
            f"- Skeleton pixels inside clump: {mean['skeleton_pixels_inside_clump']}",
            f"- Skeleton pixels inside uncertain_ignore: {mean['skeleton_pixels_inside_uncertain_ignore']}",
            f"- Fibrous_tau pixels inside clump: {mean['fibrous_tau_pixels_inside_clump']}",
            f"- Fibrous_tau pixels inside uncertain_ignore: {mean['fibrous_tau_pixels_inside_uncertain_ignore']}",
            f"- Intensity p50/p95/p99: {mean['intensity_p50']:.2f}/{mean['intensity_p95']:.2f}/{mean['intensity_p99']:.2f}",
            f"- Clump-region intensity p50/p95/p99: {mean['clump_region_intensity_p50']:.2f}/{mean['clump_region_intensity_p95']:.2f}/{mean['clump_region_intensity_p99']:.2f}",
            f"- Ignore-region intensity p50/p95/p99: {mean['uncertain_ignore_region_intensity_p50']:.2f}/{mean['uncertain_ignore_region_intensity_p95']:.2f}/{mean['uncertain_ignore_region_intensity_p99']:.2f}",
            f"- 128x128 max clump fraction and crop fractions >25%/>50%: {mean['max_patch_clump_fraction']:.3f}/{mean['patch_fraction_clump_gt_25']:.3f}/{mean['patch_fraction_clump_gt_50']:.3f}",
            "",
        ]
    lines += ["QA panels are in `panels/`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
