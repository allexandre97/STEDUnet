"""Directly observable exploratory appearance statistics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from fibras.sted_inventory import read_image
from fibras.synthetic.rendering import gaussian_blur

from .spectra import autocorrelation_radial, directional_power_ratio, normalized_radial_power_spectrum, radial_power_spectrum


PERCENTILES = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
PERCENTILE_NAMES = ["p0", "p0_1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99_9", "p100"]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def source_path(row: dict[str, str], source_roots: dict[str, str]) -> Path:
    return Path(source_roots[row["source_root_id"]]) / row["relative_path"]


def deterministic_subset(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or len(rows) <= count:
        return list(rows)
    ordered = sorted(rows, key=lambda r: r["stable_image_id"])
    idx = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return [ordered[int(i)] for i in idx]


def summarize_image(path: Path, row: dict[str, str], config: dict[str, Any]) -> dict[str, str]:
    image, _, _ = read_image(path)
    record = summarize_array(image, row, config)
    record.update(
        {
            "stable_image_id": row["stable_image_id"],
            "source_kind": row["source_kind"],
            "relative_path": row["relative_path"],
            "culture_id": row.get("culture_id", row.get("deprecated_inferred_pn", "unknown")),
            "disease": row.get("disease", row.get("deprecated_inferred_condition", "unknown")),
            "tau_isoform": row.get("tau_isoform", row.get("deprecated_inferred_round", "unknown")),
            "experimental_condition": row.get("experimental_condition", "unknown"),
            "div": row.get("div", "unknown"),
            "div_token": row.get("div_token", row.get("deprecated_inferred_div", "unknown")),
            "experimental_group_id": row.get("experimental_group_id", row.get("deprecated_biological_group_candidate", "unknown")),
            "series_index": row.get("series_index", "unknown"),
        }
    )
    return record


def summarize_array(image: np.ndarray, row: dict[str, str], config: dict[str, Any]) -> dict[str, str]:
    arr = image.astype(np.float32)
    vals = arr.ravel()
    pcts = dict(zip(PERCENTILE_NAMES, np.percentile(vals, PERCENTILES)))
    local = block_stats(arr, int(config.get("local_block_size_px", 32)))
    low_sigma = float(config.get("low_frequency_sigma_px", 32.0))
    low = gaussian_blur(arr, low_sigma)
    residual = arr - low
    std = float(arr.std()) or 1.0
    power = radial_power_spectrum(arr, int(config.get("spectrum_size_px", 256)), int(config.get("spectrum_bins", 32)))
    norm_power = normalized_radial_power_spectrum(arr, int(config.get("spectrum_size_px", 256)), int(config.get("spectrum_bins", 32)))
    ac = autocorrelation_radial(arr, int(config.get("spectrum_size_px", 256)), int(config.get("spectrum_bins", 32)))
    record = {
        "stable_image_id": row.get("stable_image_id", row.get("sample_id", "not_available")),
        "source_kind": row.get("source_kind", "generated"),
        "relative_path": row.get("relative_path", "not_applicable"),
        "culture_id": row.get("culture_id", "not_applicable"),
        "disease": row.get("disease", "not_applicable"),
        "tau_isoform": row.get("tau_isoform", "not_applicable"),
        "experimental_condition": row.get("experimental_condition", "not_applicable"),
        "div": row.get("div", "not_applicable"),
        "div_token": row.get("div_token", "not_applicable"),
        "experimental_group_id": row.get("experimental_group_id", "not_applicable"),
        "series_index": row.get("series_index", "not_applicable"),
        "mean": f"{float(vals.mean()):.6g}",
        "std": f"{float(vals.std()):.6g}",
        "zero_fraction": f"{float(np.mean(vals == 0)):.6g}",
        "saturation_fraction": f"{float(np.mean(vals == 255)):.6g}",
        "local_mean_p25": f"{local['mean_p25']:.6g}",
        "local_mean_p50": f"{local['mean_p50']:.6g}",
        "local_mean_p75": f"{local['mean_p75']:.6g}",
        "local_variance_p25": f"{local['variance_p25']:.6g}",
        "local_variance_p50": f"{local['variance_p50']:.6g}",
        "local_variance_p75": f"{local['variance_p75']:.6g}",
        "row_variation": f"{float(np.std(arr.mean(axis=1)) / std):.6g}",
        "column_variation": f"{float(np.std(arr.mean(axis=0)) / std):.6g}",
        "low_frequency_gradient_p95_minus_p5": f"{float(np.percentile(low, 95) - np.percentile(low, 5)):.6g}",
        "high_frequency_residual_std": f"{float(residual.std()):.6g}",
        "high_frequency_residual_p99_abs": f"{float(np.percentile(np.abs(residual), 99)):.6g}",
        "directional_power_ratio": f"{directional_power_ratio(arr, int(config.get('spectrum_size_px', 256))):.6g}",
        "radial_power_first_bin": f"{float(power[0]):.6g}",
        "radial_power_tail_median": f"{float(np.median(power[len(power)//2:])):.6g}",
        "normalized_radial_power_tail_median": f"{float(np.median(norm_power[len(norm_power)//2:])):.6g}",
        "normalized_radial_power_low_band_fraction": f"{float(np.sum(norm_power[1:max(2, len(norm_power)//4)])):.6g}",
        "normalized_radial_power_high_band_fraction": f"{float(np.sum(norm_power[len(norm_power)//2:])):.6g}",
        "autocorrelation_first_bin": f"{float(ac[0]):.6g}",
        "autocorrelation_tail_median": f"{float(np.median(ac[len(ac)//2:])):.6g}",
    }
    record.update({k: f"{float(v):.6g}" for k, v in pcts.items()})
    return record


def block_stats(arr: np.ndarray, block_size: int) -> dict[str, float]:
    h = arr.shape[0] // block_size * block_size
    w = arr.shape[1] // block_size * block_size
    cropped = arr[:h, :w]
    blocks = cropped.reshape(h // block_size, block_size, w // block_size, block_size)
    means = blocks.mean(axis=(1, 3)).ravel()
    variances = blocks.var(axis=(1, 3)).ravel()
    return {
        "mean_p25": float(np.percentile(means, 25)),
        "mean_p50": float(np.percentile(means, 50)),
        "mean_p75": float(np.percentile(means, 75)),
        "variance_p25": float(np.percentile(variances, 25)),
        "variance_p50": float(np.percentile(variances, 50)),
        "variance_p75": float(np.percentile(variances, 75)),
    }


def aggregate_numeric(rows: list[dict[str, str]], fields: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for field in fields:
        vals = np.asarray([float(r[field]) for r in rows if r.get(field) not in {"", "not_available"}], dtype=np.float64)
        if vals.size == 0:
            continue
        out[field] = {
            "median": float(np.median(vals)),
            "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
        }
    return out
