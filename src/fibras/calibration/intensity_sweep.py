"""Bounded exploratory foreground-intensity sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from fibras.synthetic.rendering import map_to_uint8

from .proxies import proxy_measurements
from .schema import artifact_metadata
from .statistics import aggregate_numeric, summarize_array
from .visibility import (
    gray_rgb,
    real_fiber_display_range,
    read_csv,
    scale_for_display,
    write_csv,
)


STAT_FIELDS = [
    "p50",
    "p95",
    "p99",
    "std",
    "local_variance_p50",
    "normalized_radial_power_low_band_fraction",
    "normalized_radial_power_high_band_fraction",
    "clipping_fraction",
    "saturation_fraction",
]
PROXY_FIELDS = [
    "foreground_occupancy_proxy",
    "ridge_response_p50",
    "ridge_response_p95",
]
DELTA_FIELDS = [
    "delta_p50",
    "delta_p95",
    "delta_p99",
    "delta_mean",
    "delta_variance",
    "delta_local_variance",
    "added_integrated_signal",
]


def effective_foreground_scale(
    density_multiplier: float, compositor_scale: float
) -> float:
    return float(density_multiplier) * float(compositor_scale)


def compose_scaled_foreground(
    blank: np.ndarray,
    base_signal: np.ndarray,
    effective_scale: float,
    output_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    signal = base_signal.astype(np.float32) * float(effective_scale)
    composite_float = blank.astype(np.float32) + signal
    composite_uint8, mapping = map_to_uint8(composite_float, output_config)
    mapping["clipping_fraction"] = (
        int(mapping["clipped_low_count"]) + int(mapping["clipped_high_count"])
    ) / composite_uint8.size
    mapping["saturation_fraction"] = (
        int(mapping["saturation_count"]) / composite_uint8.size
    )
    return signal, composite_uint8, mapping


def run_foreground_intensity_sweep(
    base_config: dict[str, Any],
    sweep_config: dict[str, Any],
    artifact_dir: Path,
    composite_dir: Path,
    inventory_dir: Path,
    splits_path: Path,
    out_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    if sweep_config.get("calibration_status") != "exploratory_unpartitioned":
        raise ValueError("foreground sweep must remain exploratory_unpartitioned")
    perturbation = float(
        base_config.get("real_blank_rendering", {}).get(
            "signal_dependent_perturbation_scale", 0.0
        )
    )
    if perturbation != 0:
        raise ValueError(
            "bounded one-dimensional sweep requires disabled signal perturbation "
            "so density and compositor scale remain mathematically redundant"
        )
    density_multiplier = float(
        sweep_config.get("fluorophore_density_multiplier", 1.0)
    )
    scales = [
        float(value)
        for value in sweep_config["compositor_foreground_scales"]
    ]
    manifest = read_csv(composite_dir / "dataset_manifest.csv")
    count = min(int(sweep_config.get("sample_count", 8)), len(manifest))
    manifest = manifest[:count]
    real_reference = real_reference_metrics(artifact_dir)
    stats_config = base_config.get("statistics", {})
    proxy_config = base_config.get("proxies", {})
    output_config = {
        **base_config.get("output_mapping", {}),
        **base_config.get("real_blank_rendering", {}),
    }
    per_sample: list[dict[str, str]] = []
    generated: dict[str, list[tuple[str, np.ndarray]]] = {}
    for compositor_scale in scales:
        effective = effective_foreground_scale(
            density_multiplier, compositor_scale
        )
        setting_id = f"scale_{compositor_scale:g}".replace(".", "p")
        generated[setting_id] = []
        for row in manifest:
            metadata = json.loads(
                (composite_dir / row["json_path"]).read_text(encoding="utf-8")
            )
            category = metadata.get(
                "scenario_category",
                metadata.get("rendering_report", {}).get(
                    "scenario_category", "not_reported"
                ),
            )
            require_realism_category(row["sample_id"], category)
            with np.load(
                composite_dir / row["npz_path"], allow_pickle=False
            ) as data:
                blank = data["blank_float"].copy()
                base_signal = data["total_clean_signal"].copy()
            signal, composite, mapping = compose_scaled_foreground(
                blank, base_signal, effective, output_config
            )
            stats = summarize_array(
                composite,
                {
                    "sample_id": row["sample_id"],
                    "source_kind": "foreground_intensity_sweep",
                },
                stats_config,
            )
            proxy = proxy_measurements(composite, proxy_config)
            blank_stats = summarize_array(
                blank,
                {
                    "sample_id": row["sample_id"],
                    "source_kind": "matched_blank",
                },
                stats_config,
            )
            record = {
                "setting_id": setting_id,
                "fluorophore_density_multiplier": f"{density_multiplier:.6g}",
                "compositor_foreground_scale": f"{compositor_scale:.6g}",
                "effective_foreground_scale": f"{effective:.6g}",
                "sample_id": row["sample_id"],
                "parent_synthetic_sample_id": metadata.get(
                    "parent_synthetic_sample_id", "not_recorded"
                ),
                "source_blank_id": metadata["source_blank_provenance"][
                    "blank_stable_image_id"
                ],
                "geometry_seed": str(metadata.get("geometry_seed")),
                "rendering_seed": str(metadata.get("rendering_seed")),
                "compositing_seed": str(metadata.get("compositing_seed")),
                **{field: stats[field] for field in STAT_FIELDS[:-2]},
                **{field: proxy[field] for field in PROXY_FIELDS},
                "clipping_fraction": f"{mapping['clipping_fraction']:.6g}",
                "saturation_fraction": f"{mapping['saturation_fraction']:.6g}",
                "delta_p50": difference(stats, blank_stats, "p50"),
                "delta_p95": difference(stats, blank_stats, "p95"),
                "delta_p99": difference(stats, blank_stats, "p99"),
                "delta_mean": difference(stats, blank_stats, "mean"),
                "delta_variance": f"{float(stats['std']) ** 2 - float(blank_stats['std']) ** 2:.6g}",
                "delta_local_variance": difference(
                    stats, blank_stats, "local_variance_p50"
                ),
                "added_integrated_signal": f"{float(signal.sum()):.6g}",
            }
            per_sample.append(record)
            generated[setting_id].append((row["sample_id"], composite))
    summary_rows = summarize_settings(
        per_sample, real_reference, sweep_config.get("ranking", {})
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "foreground_intensity_sweep_per_sample.csv", per_sample)
    write_csv(out_dir / "foreground_intensity_sweep_summary.csv", summary_rows)
    provenance = artifact_metadata(
        inventory_dir=inventory_dir,
        splits_path=splits_path,
        config={"base": base_config, "sweep": sweep_config},
        source_image_ids=sorted({row["source_blank_id"] for row in per_sample}),
        calibration_data_status="exploratory_unpartitioned",
        artifact_kind="foreground_intensity_sweep",
    )
    result = {
        "metadata": provenance,
        "redundancy_statement": (
            "With signal-dependent perturbation disabled, line-density and "
            "compositor scales enter only through their product. Line density "
            "was held at multiplier 1.0 and compositor scale was swept."
        ),
        "real_reference_medians": real_reference,
        "settings": summary_rows,
    }
    (out_dir / "foreground_intensity_sweep_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shared_range = real_fiber_display_range(
        artifact_dir,
        Path(base_config["source_roots"]["sted_fiber_data"]),
    )
    build_sweep_report(
        report_dir,
        summary_rows,
        generated,
        sweep_config,
        real_reference,
        shared_range,
    )
    return result


def require_realism_category(sample_id: str, category: str) -> None:
    if category != "realism_calibration":
        raise ValueError(
            f"{sample_id}: QA category {category} cannot enter the "
            "foreground-intensity sweep"
        )


def real_reference_metrics(artifact_dir: Path) -> dict[str, float]:
    real_stats = read_csv(artifact_dir / "real_fiber_stats.csv")
    real_proxies = [
        row
        for row in read_csv(artifact_dir / "proxy_stats.csv")
        if row.get("source_kind") == "fiber_image"
    ]
    aggregated = aggregate_numeric(
        real_stats,
        [
            "p50",
            "p95",
            "p99",
            "std",
            "local_variance_p50",
            "normalized_radial_power_low_band_fraction",
            "normalized_radial_power_high_band_fraction",
        ],
    )
    proxy_aggregated = aggregate_numeric(real_proxies, PROXY_FIELDS)
    return {
        field: values["median"]
        for field, values in {**aggregated, **proxy_aggregated}.items()
    }


def summarize_settings(
    rows: list[dict[str, str]],
    real_reference: dict[str, float],
    ranking_config: dict[str, Any],
) -> list[dict[str, str]]:
    settings = sorted({row["setting_id"] for row in rows})
    output = []
    for setting in settings:
        selected = [row for row in rows if row["setting_id"] == setting]
        aggregate = aggregate_numeric(
            selected, STAT_FIELDS + PROXY_FIELDS + DELTA_FIELDS
        )
        medians = {
            field: aggregate[field]["median"]
            for field in aggregate
            if "median" in aggregate[field]
        }
        intensity_mismatch = sum(
            abs(medians[field] - real_reference[field])
            for field in ["p50", "p95", "p99", "std"]
        )
        local_variance_mismatch = abs(
            medians["local_variance_p50"]
            - real_reference["local_variance_p50"]
        )
        structural_mismatch = sum(
            relative_difference(medians[field], real_reference[field])
            for field in PROXY_FIELDS
        )
        spectral_mismatch = sum(
            abs(medians[field] - real_reference[field])
            for field in [
                "normalized_radial_power_low_band_fraction",
                "normalized_radial_power_high_band_fraction",
            ]
        )
        clipping_limit = float(
            ranking_config.get("clipping_fraction_limit", 1e-4)
        )
        saturation_limit = float(
            ranking_config.get("saturation_fraction_limit", 1e-4)
        )
        max_clipping = max(float(row["clipping_fraction"]) for row in selected)
        max_saturation = max(
            float(row["saturation_fraction"]) for row in selected
        )
        output.append(
            {
                "setting_id": setting,
                "fluorophore_density_multiplier": selected[0][
                    "fluorophore_density_multiplier"
                ],
                "compositor_foreground_scale": selected[0][
                    "compositor_foreground_scale"
                ],
                "effective_foreground_scale": selected[0][
                    "effective_foreground_scale"
                ],
                **{
                    f"median_{field}": f"{value:.6g}"
                    for field, value in medians.items()
                },
                "intensity_mismatch_l1": f"{intensity_mismatch:.6g}",
                "local_variance_abs_mismatch": f"{local_variance_mismatch:.6g}",
                "structural_ridge_relative_mismatch_sum": f"{structural_mismatch:.6g}",
                "normalized_spectral_band_abs_mismatch_sum": f"{spectral_mismatch:.6g}",
                "max_clipping_fraction": f"{max_clipping:.6g}",
                "max_saturation_fraction": f"{max_saturation:.6g}",
                "clipping_or_saturation_violation": str(
                    max_clipping > clipping_limit
                    or max_saturation > saturation_limit
                ).lower(),
                "calibration_status": "exploratory_unpartitioned",
            }
        )
    add_rank(output, "intensity_mismatch_l1", "intensity_rank")
    add_rank(
        output, "local_variance_abs_mismatch", "local_variance_rank"
    )
    add_rank(
        output,
        "structural_ridge_relative_mismatch_sum",
        "structural_ridge_rank",
    )
    add_rank(
        output,
        "normalized_spectral_band_abs_mismatch_sum",
        "spectral_shape_rank",
    )
    return sorted(output, key=lambda row: float(row["effective_foreground_scale"]))


def difference(
    first: dict[str, str], second: dict[str, str], key: str
) -> str:
    return f"{float(first[key]) - float(second[key]):.6g}"


def relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-9)


def add_rank(
    rows: list[dict[str, str]], value_field: str, rank_field: str
) -> None:
    ordered = sorted(rows, key=lambda row: float(row[value_field]))
    for rank, row in enumerate(ordered, start=1):
        row[rank_field] = str(rank)


def build_sweep_report(
    report_dir: Path,
    summary_rows: list[dict[str, str]],
    generated: dict[str, list[tuple[str, np.ndarray]]],
    config: dict[str, Any],
    real_reference: dict[str, float],
    shared_range: tuple[float, float],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    draw_sweep_comparison(summary_rows, report_dir / "sweep_metric_summary.png")
    representative_count = int(
        config.get("visualization", {}).get(
            "representative_samples_per_setting", 3
        )
    )
    draw_sweep_contact_sheet(
        generated,
        report_dir / "foreground_scale_contact_sheet.png",
        representative_count,
        shared_range,
    )
    current_default = float(config.get("current_default_scale", 1.0))
    candidates = [
        row
        for row in sorted(
            summary_rows, key=lambda row: int(row["intensity_rank"])
        )
        if row["clipping_or_saturation_violation"] == "false"
        and float(row["effective_foreground_scale"]) > current_default
    ][:2]
    rejected = [
        row
        for row in sorted(
            summary_rows, key=lambda row: int(row["intensity_rank"])
        )
        if row["clipping_or_saturation_violation"] == "true"
    ]
    lines = [
        "# Exploratory foreground-intensity sweep",
        "",
        "- calibration_status: `exploratory_unpartitioned`",
        "- No setting is promoted to `empirically_matched`.",
        "- Main production configuration was not modified.",
        "- Geometry, parent samples, source blanks, PSF, seeds, and output mapping were held fixed.",
        "- Line-density multiplier was held at `1.0`; compositor scale was varied because the two factors are mathematically redundant with perturbation disabled.",
        f"- Comparison contact-sheet display range: shared real-STED `{shared_range[0]:.6g}` to `{shared_range[1]:.6g}`.",
        f"- Current production default effective scale: `{current_default}`.",
        "",
        "## Real-fiber reference medians",
        "",
    ]
    lines.extend(
        f"- `{key}`: `{value:.6g}`" for key, value in real_reference.items()
    )
    lines.extend(
        [
            "",
            "## Setting comparison",
            "",
            "| Effective scale | p95 | p99 | Std | Local variance | Occupancy proxy | Ridge p95 | Intensity rank | Variance rank | Structural rank | Max clipped | Max saturated | Violation |",
            "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['effective_foreground_scale']} | "
            f"{row['median_p95']} | {row['median_p99']} | {row['median_std']} | "
            f"{row['median_local_variance_p50']} | "
            f"{row['median_foreground_occupancy_proxy']} | "
            f"{row['median_ridge_response_p95']} | {row['intensity_rank']} | "
            f"{row['local_variance_rank']} | {row['structural_ridge_rank']} | "
            f"{row['max_clipping_fraction']} | "
            f"{row['max_saturation_fraction']} | "
            f"{row['clipping_or_saturation_violation']} |"
        )
    lines.extend(["", "## Promising candidates for later review", ""])
    if candidates:
        for row in candidates:
            lines.append(
                f"- Effective scale `{row['effective_foreground_scale']}`: "
                f"intensity rank `{row['intensity_rank']}`, variance rank "
                f"`{row['local_variance_rank']}`, structural/ridge rank "
                f"`{row['structural_ridge_rank']}`. This is exploratory, not a "
                "production selection."
            )
    else:
        lines.append(
            "- No brighter-than-default setting passed clipping/saturation limits."
        )
    lines.extend(["", "## Rejected settings", ""])
    if rejected:
        for row in rejected:
            lines.append(
                f"- Effective scale `{row['effective_foreground_scale']}`: "
                f"maximum clipped fraction `{row['max_clipping_fraction']}` and "
                f"maximum saturated fraction `{row['max_saturation_fraction']}` "
                "exceeded the configured sweep limit."
            )
    else:
        lines.append("- No setting exceeded the clipping/saturation limits.")
    lines.extend(
        [
            "",
            "## Ranking definitions",
            "",
            "- Intensity mismatch: sum of absolute median differences for p50, p95, p99, and standard deviation.",
            "- Local-variance mismatch: absolute median local-variance difference.",
            "- Structural/ridge mismatch: sum of relative median differences for occupancy and ridge p50/p95 proxies.",
            "- Spectral mismatch: sum of absolute differences in normalized low- and high-band fractions.",
            "- Rankings are separate diagnostics; they are not combined into a realism score.",
            "",
            "## Figures",
            "",
            "- [Foreground-scale contact sheet](foreground_scale_contact_sheet.png)",
            "- [Sweep metric summary](sweep_metric_summary.png)",
            "- Machine-readable summaries: `calibration_artifacts/foreground_intensity_sweep/foreground_intensity_sweep_summary.csv` and `foreground_intensity_sweep_per_sample.csv`.",
        ]
    )
    (report_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def draw_sweep_contact_sheet(
    generated: dict[str, list[tuple[str, np.ndarray]]],
    path: Path,
    representative_count: int,
    shared_range: tuple[float, float],
) -> None:
    settings = list(generated)
    tile = 160
    canvas = Image.new(
        "RGB", (tile * representative_count, tile * len(settings) + 28), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (6, 6),
        f"Shared real-STED display range {shared_range[0]:.3g}–{shared_range[1]:.3g}",
        fill="black",
    )
    for row_index, setting in enumerate(settings):
        samples = generated[setting]
        indices = np.linspace(
            0, len(samples) - 1, min(representative_count, len(samples)), dtype=int
        )
        for column_index, sample_index in enumerate(indices):
            sample_id, image = samples[int(sample_index)]
            display = scale_for_display(image, *shared_range)
            x = column_index * tile
            y = row_index * tile + 28
            canvas.paste(
                Image.fromarray(gray_rgb(display)).resize((tile, tile)),
                (x, y),
            )
            draw.rectangle((x, y, x + tile, y + 18), fill="black")
            draw.text(
                (x + 3, y + 3),
                f"{setting} | {sample_id[-4:]}",
                fill="white",
            )
    canvas.save(path, optimize=True)


def draw_sweep_comparison(
    rows: list[dict[str, str]], path: Path
) -> None:
    metrics = [
        ("median_p99", "p99"),
        ("median_std", "std"),
        ("median_local_variance_p50", "local variance"),
        ("median_ridge_response_p95", "ridge p95"),
    ]
    image = Image.new("RGB", (900, 440), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "Foreground-intensity sweep medians", fill="black")
    colors = [(0, 90, 220), (220, 90, 0), (60, 160, 60), (160, 60, 160)]
    for metric_index, (field, label) in enumerate(metrics):
        values = [float(row[field]) for row in rows]
        maximum = max(values) or 1.0
        x0 = 40 + metric_index * 215
        draw.text((x0, 42), label, fill="black")
        for index, value in enumerate(values):
            height = int(280 * value / maximum)
            x = x0 + index * 27
            draw.rectangle(
                (x, 350 - height, x + 18, 350), fill=colors[metric_index]
            )
            draw.text(
                (x, 358), rows[index]["effective_foreground_scale"], fill="black"
            )
    image.save(path, optimize=True)
