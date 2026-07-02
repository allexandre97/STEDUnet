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


GROUND_TRUTH_FIELDS = [
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
    "supervised_endpoint_count",
    "valid_endpoint_trace_count",
    "boundary_truncation_trace_count",
    "terminates_in_bundle_trace_count",
    "terminates_in_clump_trace_count",
    "ambiguous_termination_trace_count",
    "individual_filament_instance_count",
    "bundle_instance_count",
    "clump_instance_count",
    "supervised_membership_count",
    "latent_geometry_membership_count",
    "supervised_edge_count",
    "latent_edge_count",
    "bundle_width_p50_px",
    "clump_hole_fill_ratio_p50",
    "clump_component_count",
]

IMAGE_PROXY_FIELDS = [
    "foreground_occupancy",
    "tile_occupancy_mean",
    "tile_occupancy_variance",
    "empty_tile_fraction",
    "spatial_concentration_index",
    "occupied_domain_count",
    "occupied_domain_area_p50_tiles",
    "orientation_coherence_p50",
    "orientation_coherence_p90",
    "foreground_signal_p50",
    "foreground_signal_p95",
    "foreground_signal_p99",
    "clipping_fraction",
    "saturation_fraction",
]

ALIGNMENT_FIELDS = [
    "class_area_px",
    "fraction_of_class_mask_above_visible_threshold",
    "fraction_of_class_mask_with_nonzero_signal",
    "fraction_of_visible_class_signal_outside_class_mask",
    "signal_p50_inside_class",
    "signal_p95_inside_class",
    "signal_p99_inside_class",
    "ridge_response_p50_inside_class",
    "ridge_response_p95_inside_class",
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
        **ground_truth_counts(arrays),
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


def ground_truth_counts(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    fields = [
        "supervised_endpoint_count",
        "valid_endpoint_trace_count",
        "boundary_truncation_trace_count",
        "terminates_in_bundle_trace_count",
        "terminates_in_clump_trace_count",
        "ambiguous_termination_trace_count",
        "individual_filament_instance_count",
        "bundle_instance_count",
        "clump_instance_count",
        "supervised_membership_count",
        "latent_geometry_membership_count",
        "supervised_edge_count",
        "latent_edge_count",
        "bundle_width_p50_px",
        "clump_hole_fill_ratio_p50",
        "clump_component_count",
    ]
    if "semantic_class_mask" not in arrays:
        return {field: "not_available" for field in fields}
    status = np.concatenate(
        [
            arrays.get("trace_start_status", np.zeros(0, dtype=np.int16)),
            arrays.get("trace_end_status", np.zeros(0, dtype=np.int16)),
        ]
    )
    status_names = {
        1: "valid_endpoint_trace_count",
        2: "boundary_truncation_trace_count",
        3: "terminates_in_bundle_trace_count",
        4: "terminates_in_clump_trace_count",
        5: "ambiguous_termination_trace_count",
    }
    out = {
        name: str(int(np.count_nonzero(status == code)))
        for code, name in status_names.items()
    }
    out.update(
        {
            "supervised_endpoint_count": str(
                int(
                    np.count_nonzero(
                        arrays.get("node_supervised", np.zeros(0))
                    )
                )
            ),
            "individual_filament_instance_count": str(
                len(
                    np.unique(
                        arrays.get(
                            "individual_filament_membership_instance_id",
                            np.zeros(0),
                        )
                    )
                )
            ),
            "bundle_instance_count": str(
                len(np.unique(arrays.get("bundle_ids", np.zeros(0))))
            ),
            "clump_instance_count": str(
                len(np.unique(arrays.get("clump_ids", np.zeros(0))))
            ),
            "supervised_membership_count": str(
                int(arrays.get("supervised_membership_y", np.zeros(0)).size)
            ),
            "latent_geometry_membership_count": str(
                int(
                    arrays.get(
                        "latent_geometry_membership_y", np.zeros(0)
                    ).size
                )
            ),
            "supervised_edge_count": str(
                int(np.count_nonzero(arrays.get("edge_supervised", np.zeros(0))))
            ),
            "latent_edge_count": str(
                int(
                    np.count_nonzero(
                        ~arrays.get(
                            "edge_supervised", np.ones(0, dtype=np.uint8)
                        ).astype(bool)
                    )
                )
            ),
            "bundle_width_p50_px": numeric(bundle_width_p50(arrays)),
            "clump_hole_fill_ratio_p50": numeric(clump_hole_fill_ratio_p50(arrays)),
            "clump_component_count": str(
                int(ndimage.label(arrays.get("clump_mask", np.zeros((1, 1))))[1])
            ),
        }
    )
    return out


def bundle_width_p50(arrays: dict[str, np.ndarray]) -> float:
    mask = arrays.get("bundle_mask")
    axis = arrays.get("bundle_axis_mask")
    if mask is None or axis is None or not np.any(axis):
        return 0.0
    distance = ndimage.distance_transform_edt(mask > 0)
    return 2.0 * float(np.median(distance[axis.astype(bool)]))


def clump_hole_fill_ratio_p50(arrays: dict[str, np.ndarray]) -> float:
    ids = arrays.get("clump_ids", np.zeros(0))
    values = []
    for clump_id in ids:
        selected = (
            arrays["clump_membership_instance_id"] == int(clump_id)
        )
        mask = np.zeros_like(arrays["clump_mask"], dtype=bool)
        mask[
            arrays["clump_membership_y"][selected],
            arrays["clump_membership_x"][selected],
        ] = True
        values.append(
            float(mask.sum() / max(int(ndimage.binary_fill_holes(mask).sum()), 1))
        )
    return float(np.median(values)) if values else 0.0


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
    morphology_ground_truth = dataset_diagnostics(composite_dir)
    previous_ground_truth = dataset_diagnostics(previous_dir)
    morphology_proxy_rows = dataset_proxy_diagnostics(
        composite_dir, "new_morphology_composite_proxy"
    )
    previous_proxy_rows = dataset_proxy_diagnostics(
        previous_dir, "previous_composite_proxy"
    )
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
    ground_truth_rows = morphology_ground_truth + previous_ground_truth
    proxy_rows = morphology_proxy_rows + previous_proxy_rows + real_rows + blank_rows
    write_csv(
        diagnostics_dir / "synthetic_ground_truth_diagnostics.csv",
        ground_truth_rows,
    )
    write_csv(
        diagnostics_dir / "matched_image_proxy_diagnostics.csv", proxy_rows
    )
    alignment_rows = write_class_signal_alignment(
        composite_dir,
        diagnostics_dir / "class_signal_alignment.csv",
    )
    alignment_summaries = {
        class_name: aggregate_numeric(
            [row for row in alignment_rows if row["class_name"] == class_name],
            ALIGNMENT_FIELDS,
        )
        for class_name in sorted({row["class_name"] for row in alignment_rows})
    }
    ground_truth_summaries = {
        kind: aggregate_numeric(
            [row for row in ground_truth_rows if row["source_kind"] == kind],
            GROUND_TRUTH_FIELDS,
        )
        for kind in sorted({row["source_kind"] for row in ground_truth_rows})
    }
    proxy_summaries = {
        kind: aggregate_numeric(
            [row for row in proxy_rows if row["source_kind"] == kind],
            IMAGE_PROXY_FIELDS,
        )
        for kind in sorted({row["source_kind"] for row in proxy_rows})
    }
    mode_summaries = {
        mode: aggregate_numeric(
            [
                row
                for row in morphology_ground_truth
                if row["scene_mode"] == mode
            ],
            GROUND_TRUTH_FIELDS,
        )
        for mode in sorted(
            {row["scene_mode"] for row in morphology_ground_truth}
        )
    }
    (diagnostics_dir / "morphology_semantic_summary.json").write_text(
        json.dumps(
            {
                "calibration_status": "exploratory_unpartitioned",
                "condition_blind": True,
                "synthetic_ground_truth": ground_truth_summaries,
                "matched_image_proxies": proxy_summaries,
                "morphology_modes": mode_summaries,
                "class_signal_alignment": alignment_summaries,
                "warning": (
                    "Image proxies are threshold_sensitive, condition_blind, "
                    "and not_biological_ground_truth. Exact synthetic targets "
                    "are reported separately."
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
    draw_supervision_contact_sheet(
        sample_records,
        out_dir / "supervised_vs_latent_geometry.png",
        shared_range,
    )
    draw_signal_alignment_contact_sheet(
        sample_records,
        out_dir / "class_mask_signal_alignment.png",
        shared_range,
    )
    draw_endpoint_crossing_contact_sheet(
        sample_records,
        out_dir / "endpoint_and_resolved_crossing_semantics.png",
        shared_range,
    )
    draw_diagnostic_summary(
        proxy_summaries, out_dir / "matched_image_proxy_summary.png"
    )
    write_morphology_report(
        out_dir / "report.md",
        ground_truth_summaries,
        proxy_summaries,
        mode_summaries,
        alignment_summaries,
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


def dataset_proxy_diagnostics(
    dataset_dir: Path, source_kind: str
) -> list[dict[str, str]]:
    rows = []
    for row in read_csv(dataset_dir / "dataset_manifest.csv"):
        metadata = json.loads(
            (dataset_dir / row["json_path"]).read_text(encoding="utf-8")
        )
        with np.load(dataset_dir / row["npz_path"], allow_pickle=False) as data:
            image = data["render_uint8"].copy()
        rows.append(real_scene_diagnostics(image, metadata["sample_id"], source_kind))
    return rows


def write_class_signal_alignment(
    dataset_dir: Path, path: Path
) -> list[dict[str, str]]:
    rows = []
    for manifest_row in read_csv(dataset_dir / "dataset_manifest.csv"):
        metadata = json.loads(
            (dataset_dir / manifest_row["json_path"]).read_text(
                encoding="utf-8"
            )
        )
        for class_name, metrics in metadata.get("rendering_report", {}).get(
            "class_signal_alignment", {}
        ).items():
            rows.append(
                {
                    "sample_id": metadata["sample_id"],
                    "class_name": class_name,
                    **{
                        key: "not_available" if value is None else str(value)
                        for key, value in metrics.items()
                    },
                }
            )
    write_csv(path, rows)
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


def draw_supervision_contact_sheet(
    records: list[tuple[str, str, dict[str, np.ndarray]]],
    path: Path,
    shared_range: tuple[float, float],
) -> None:
    selected = records[:2] + [
        record
        for record in records
        if record[1] in {"bundle_dominated", "mixed_morphology"}
    ][:2]
    tile = 224
    canvas = Image.new("RGB", (tile * 2, tile * len(selected) + 24), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), "Supervised targets versus latent geometry provenance", fill="black")
    for row, (sample_id, _, arrays) in enumerate(selected):
        base = scale_for_display(arrays["render_uint8"], *shared_range)
        supervised = class_overlay(base, arrays)
        latent = gray_rgb(base).astype(np.float32)
        latent_mask = arrays["latent_geometry_overlap_count"] > 0
        supervised_mask = arrays["supervised_overlap_count"] > 0
        latent[latent_mask] = 0.3 * latent[latent_mask] + 0.7 * np.asarray(
            (255, 40, 40)
        )
        latent[supervised_mask] = 0.2 * latent[supervised_mask] + 0.8 * np.asarray(
            (0, 220, 255)
        )
        for col, (label, panel) in enumerate(
            [("supervised classes", supervised), ("latent red / supervised cyan", latent.astype(np.uint8))]
        ):
            x, y = col * tile, row * tile + 22
            canvas.paste(Image.fromarray(panel).resize((tile, tile)), (x, y))
            draw.rectangle((x, y, x + tile, y + 18), fill="black")
            draw.text((x + 3, y + 3), f"{label} | {sample_id[-4:]}", fill="white")
    canvas.save(path, optimize=True)


def draw_signal_alignment_contact_sheet(
    records: list[tuple[str, str, dict[str, np.ndarray]]],
    path: Path,
    shared_range: tuple[float, float],
) -> None:
    selected = [
        record
        for record in records
        if record[1] in {"bundle_dominated", "clump_dominated", "mixed_morphology"}
    ][:3]
    tile = 210
    canvas = Image.new("RGB", (tile * 4, tile * len(selected) + 24), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), "Class masks and attributed clean optical signal", fill="black")
    for row, (sample_id, _, arrays) in enumerate(selected):
        base = scale_for_display(arrays["render_uint8"], *shared_range)
        panels = [("class overlay", class_overlay(base, arrays))]
        for class_name, mask_name in [
            ("filament signal", "individual_filament"),
            ("bundle signal", "bundle"),
            ("clump signal", "clump"),
        ]:
            signal, _ = robust_display(
                arrays[f"{mask_name}_signal"], (0.5, 99.5)
            )
            panels.append(
                (
                    class_name,
                    mask_overlay(
                        signal,
                        arrays[f"{mask_name}_mask"],
                        (255, 140, 0),
                    ),
                )
            )
        for col, (label, panel) in enumerate(panels):
            x, y = col * tile, row * tile + 22
            canvas.paste(Image.fromarray(panel).resize((tile, tile)), (x, y))
            draw.rectangle((x, y, x + tile, y + 18), fill="black")
            draw.text((x + 3, y + 3), f"{label} | {sample_id[-4:]}", fill="white")
    canvas.save(path, optimize=True)


def draw_endpoint_crossing_contact_sheet(
    records: list[tuple[str, str, dict[str, np.ndarray]]],
    path: Path,
    shared_range: tuple[float, float],
) -> None:
    boundary_record = next(
        (
            record
            for record in records
            if np.any(record[2].get("node_boundary_code", np.zeros(0)))
        ),
        records[0],
    )
    crossing_record = next(
        (
            record
            for record in records
            if resolved_bundle_crossing_mask(record[2]).any()
        ),
        records[0],
    )
    panels = []
    for label, record in [
        ("endpoint / boundary", boundary_record),
        ("resolved bundle-child crossing", crossing_record),
    ]:
        sample_id, _, arrays = record
        base = gray_rgb(
            scale_for_display(arrays["render_uint8"], *shared_range)
        ).astype(np.float32)
        endpoint = arrays["endpoint_map"].astype(bool)
        boundary = point_mask(
            arrays["node_xy"],
            arrays.get("node_boundary_code", np.zeros(0)) > 0,
            endpoint.shape,
        )
        crossing = resolved_bundle_crossing_mask(arrays)
        base[endpoint] = (0, 255, 80)
        base[boundary] = (255, 40, 40)
        base[crossing] = (255, 0, 255)
        focus = boundary | crossing | endpoint
        panel = (
            crop_around_mask(base.astype(np.uint8), focus, 256)
            if np.any(focus)
            else base.astype(np.uint8)
        )
        panels.append((label, sample_id, panel))
    tile = 300
    canvas = Image.new("RGB", (tile * 2, tile + 24), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (5, 4),
        "Green valid endpoint; red boundary truncation; magenta resolved crossing",
        fill="black",
    )
    for index, (label, sample_id, panel) in enumerate(panels):
        x = index * tile
        canvas.paste(Image.fromarray(panel).resize((tile, tile)), (x, 22))
        draw.rectangle((x, 22, x + tile, 40), fill="black")
        draw.text((x + 3, 25), f"{label} | {sample_id[-4:]}", fill="white")
    canvas.save(path, optimize=True)


def point_mask(
    points_xy: np.ndarray,
    selected: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for point in points_xy[selected]:
        x = int(np.clip(round(float(point[0])), 0, shape[1] - 1))
        y = int(np.clip(round(float(point[1])), 0, shape[0] - 1))
        mask[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = True
    return mask


def resolved_bundle_crossing_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    mask = np.zeros_like(arrays["semantic_mask"], dtype=bool)
    fiber_types = dict(
        zip(map(int, arrays["fiber_ids"]), map(int, arrays["fiber_structure_type"]))
    )
    for point, pair in zip(
        arrays["projected_crossing_points_xy"],
        arrays["projected_crossing_fiber_ids"],
    ):
        if any(fiber_types[int(fiber_id)] == 2 for fiber_id in pair):
            x = int(np.clip(round(float(point[0])), 0, mask.shape[1] - 1))
            y = int(np.clip(round(float(point[1])), 0, mask.shape[0] - 1))
            mask[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4] = True
    return mask


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
        "new_morphology_composite_proxy",
        "previous_composite_proxy",
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
    draw.text(
        (10, 10),
        "Matched image-proxy medians: threshold-sensitive, condition-blind",
        fill="black",
    )
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
    ground_truth_summaries: dict[str, dict[str, dict[str, float]]],
    proxy_summaries: dict[str, dict[str, dict[str, float]]],
    mode_summaries: dict[str, dict[str, dict[str, float]]],
    alignment_summaries: dict[str, dict[str, dict[str, float]]],
    shared_range: tuple[float, float],
    records: list[tuple[str, str, dict[str, np.ndarray]]],
) -> None:
    lines = [
        "# Synthetic STED morphology heterogeneity review",
        "",
        "- calibration_status: `exploratory_unpartitioned`",
        "- schema: `synthetic_sted_3d_morphology_0.8.0`",
        "- Morphology generation is condition-blind and broadly randomized.",
        "- Real-image measurements are coarse threshold-sensitive diagnostics, not fitted biological targets.",
        "- Composite foreground scale is sampled deterministically from `1.0–2.0`.",
        f"- Shared display range: `{shared_range[0]:.6g}` to `{shared_range[1]:.6g}`.",
        f"- Review samples: `{len(records)}`.",
        "",
        "## A. Synthetic ground-truth morphology diagnostics",
        "",
        "| Group | Foreground occupancy | Tile variance | Endpoints | Supervised memberships | Latent memberships | Bundle width | Clump hole-fill ratio |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for group in [
        "morphology_scene",
        "previous_uniform_synthetic",
    ]:
        summary = ground_truth_summaries.get(group, {})
        value = lambda key: report_value(summary, key)
        lines.append(
            f"| `{group}` | {value('foreground_occupancy')} | "
            f"{value('tile_occupancy_variance')} | "
            f"{value('supervised_endpoint_count')} | "
            f"{value('supervised_membership_count')} | "
            f"{value('latent_geometry_membership_count')} | "
            f"{value('bundle_width_p50_px')} | "
            f"{value('clump_hole_fill_ratio_p50')} |"
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
            "## Class-mask and rendered-signal alignment",
            "",
            "| Class | Visible signal outside apparent mask | Visible signal inside apparent mask | Apparent/source area ratio | Apparent mask visible fraction |",
            "|:--|--:|--:|--:|--:|",
        ]
    )
    for class_name in [
        "individual_filament",
        "bundle",
        "clump",
        "uncertain_transition",
    ]:
        summary = alignment_summaries.get(class_name, {})
        value = lambda key: report_value(summary, key)
        lines.append(
            f"| `{class_name}` | "
            f"{value('fraction_of_visible_class_signal_outside_class_mask')} | "
            f"{value('fraction_of_visible_class_signal_inside_apparent_mask')} | "
            f"{value('apparent_to_source_area_ratio')} | "
            f"{value('fraction_of_apparent_mask_with_visible_class_signal')} |"
        )
    lines.extend(
        [
            "",
            "The uncertain-transition row is not evaluated as scene-wide signal spill.",
        ]
    )
    lines.extend(
        [
            "",
            "## B. Matched image-proxy diagnostics",
            "",
            "All rows below use identical thresholding, ridge/orientation processing, tile size, and connected-component settings. Metrics are `threshold_sensitive`, `condition_blind`, and `not_biological_ground_truth`.",
            "",
            "| Group | Occupancy proxy | Tile variance | Empty tiles | Concentration | Orientation coherence | Signal p95 |",
            "|:--|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for group in [
        "new_morphology_composite_proxy",
        "previous_composite_proxy",
        "real_fiber_proxy",
        "real_blank_proxy",
    ]:
        summary = proxy_summaries.get(group, {})
        value = lambda key: report_value(summary, key)
        lines.append(
            f"| `{group}` | {value('foreground_occupancy')} | "
            f"{value('tile_occupancy_variance')} | "
            f"{value('empty_tile_fraction')} | "
            f"{value('spatial_concentration_index')} | "
            f"{value('orientation_coherence_p50')} | "
            f"{value('foreground_signal_p95')} |"
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
            "- [Supervised versus latent geometry](supervised_vs_latent_geometry.png)",
            "- [Class masks versus signal](class_mask_signal_alignment.png)",
            "- [Endpoint and resolved-crossing semantics](endpoint_and_resolved_crossing_semantics.png)",
            "- [Matched image-proxy summary](matched_image_proxy_summary.png)",
            "- Full local per-sample panels are generated under `full_samples/` and remain ignored by Git.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_value(summary: dict[str, dict[str, float]], key: str) -> str:
    value = summary.get(key, {}).get("median")
    return "n/a" if value is None else f"{value:.4g}"


def numeric(value: Any) -> str:
    return f"{float(value):.6g}"
