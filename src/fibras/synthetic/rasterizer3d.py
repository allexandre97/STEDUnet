"""Depth-dependent 2D rasterization for 3D synthetic fibers."""

from __future__ import annotations

import math
import time
from functools import lru_cache
import json
from typing import Any

import numpy as np
try:
    from scipy import ndimage as scipy_ndimage
except Exception:  # pragma: no cover - exercised only if scipy is absent.
    scipy_ndimage = None

from .geometry3d import generate_persistent_chain_geometry
from .rendering import map_to_uint8
from .schema import NODE_TYPES
from .targets import draw_disk, segment_intersection


REALISM_SCENARIOS = {"random_persistent_chain", "sparse_near_planar", "strongly_3d", "dense_local_geometry", "weak_intermittent"}
STRUCTURAL_QA_SCENARIOS = {"projected_depth_crossing", "near_coplanar_crossing", "true_junction_fixture"}
OPTICAL_QA_SCENARIOS = {"straight_width_calibration", "in_focus_defocused_crossing"}


def rasterize_3d_sample(
    geometry: dict[str, Any],
    targets_config: dict[str, Any],
    optical_config: dict[str, Any],
    output_config: dict[str, Any],
    rendering_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    height, width = geometry["image_shape"]
    focal_z = float(geometry["focal_plane_z_px"])
    focal_depth = float(optical_config.get("focal_depth_range_px", 6.0))
    visible_threshold = float(optical_config.get("visible_signal_threshold", 3.0))
    foreground_scale = float(output_config.get("foreground_scale", optical_config.get("foreground_scale", 1.0)))

    target_arrays, target_report = geometry_targets_3d(geometry, targets_config, focal_depth)
    signal_arrays, signal_report = splat_geometry_signal(geometry, optical_config, visible_threshold, foreground_scale)
    arrays = {**target_arrays, **signal_arrays}
    arrays["semantic_mask"] = select_semantic_mask(arrays, targets_config)
    arrays["ignore_mask"] = build_ignore_mask(arrays, targets_config, visible_threshold)
    optional_report = apply_optional_targets(arrays, targets_config)

    render = arrays["total_clean_signal"].astype(np.float32) + float(output_config.get("background_level", 4.0))
    noise_std = float(output_config.get("background_noise_std", 0.0))
    if noise_std > 0:
        rng = np.random.default_rng(rendering_seed)
        render = render + rng.normal(0, noise_std, render.shape).astype(np.float32)
    render = np.maximum(render, 0).astype(np.float32)
    uint8, mapping = map_to_uint8(render, output_config)
    arrays["render_float"] = render
    arrays["render_uint8"] = uint8
    return arrays, {
        **signal_report,
        **target_report,
        **optional_report,
        **mapping,
        "float_to_uint8_mapping": mapping,
        "semantic_mask_source": targets_config.get("semantic_mask_source", "visible_signal_mask"),
        "ignore_mask_rule": targets_config.get("ignore_mask_rule", "none"),
        "scenario": geometry["parameters"].get("scenario", "random_persistent_chain"),
        "scenario_category": scenario_category(geometry["parameters"].get("scenario", "random_persistent_chain"), targets_config),
        "image_shape": [height, width],
        "foreground_scale": foreground_scale,
        "render_float_min": float(render.min()),
        "render_float_max": float(render.max()),
        "clipping_count": int(mapping["clipped_low_count"]) + int(mapping["clipped_high_count"]),
        "clipping_fraction": float((int(mapping["clipped_low_count"]) + int(mapping["clipped_high_count"])) / render.size),
        "saturation_fraction": float(int(mapping["saturation_count"]) / render.size),
    }


def geometry_targets_3d(geometry: dict[str, Any], config: dict[str, Any], focal_depth: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    height, width = geometry["image_shape"]
    centerline_radius = float(config.get("centerline_radius_px", 0.75))
    endpoint_radius = float(config.get("endpoint_radius_px", 3.0))
    junction_radius = float(config.get("junction_radius_px", 3.0))
    crossing_radius = float(config.get("crossing_radius_px", 3.0))
    near_coplanar_depth_px = float(config.get("near_coplanar_depth_px", 8.0))
    focal_z = float(geometry["focal_plane_z_px"])

    line_source = np.zeros((height, width), dtype=np.float32)
    support_preview = np.zeros((height, width), dtype=np.float32)
    projection = np.zeros((height, width), dtype=np.uint8)
    in_focus = np.zeros((height, width), dtype=np.uint8)
    overlap = np.zeros((height, width), dtype=np.uint16)
    centerline = np.zeros((height, width), dtype=np.uint8)
    orientation_cos_sum = np.zeros((height, width), dtype=np.float32)
    orientation_sin_sum = np.zeros((height, width), dtype=np.float32)
    orientation_instance_count = np.zeros((height, width), dtype=np.uint16)
    membership_y: list[np.ndarray] = []
    membership_x: list[np.ndarray] = []
    membership_i: list[np.ndarray] = []
    arc_weights: list[np.ndarray] = []
    densities: list[np.ndarray] = []
    trace_points: list[np.ndarray] = []
    trace_offsets = [0]
    trace_ids: list[int] = []

    for fiber in geometry["fibers"]:
        fid = int(fiber["fiber_id"])
        points = fiber["points_xyz"]
        radii = fiber["sample_radius"]
        density = fiber["sample_amplitude"]
        ds = sample_arc_length_weights(points)
        support = rasterize_variable_disks(points[:, :2], radii, (height, width))
        projection |= support.astype(np.uint8)
        overlap += support.astype(np.uint16)
        line = rasterize_variable_disks(points[:, :2], np.full(points.shape[0], centerline_radius, dtype=np.float32), (height, width))
        centerline |= line.astype(np.uint8)
        focus_mask = np.abs(points[:, 2] - focal_z) <= focal_depth
        if np.any(focus_mask):
            in_focus |= rasterize_variable_disks(points[focus_mask, :2], radii[focus_mask], (height, width)).astype(np.uint8)
        add_line_source(line_source, points[:, :2], density * ds)
        add_source_disks(support_preview, points[:, :2], radii, density * ds)
        add_orientation_contribution(orientation_cos_sum, orientation_sin_sum, orientation_instance_count, points[:, :2], centerline_radius)
        y, x = np.nonzero(support)
        membership_y.append(y.astype(np.int32))
        membership_x.append(x.astype(np.int32))
        membership_i.append(np.full(y.shape, fid, dtype=np.int32))
        arc_weights.append(ds.astype(np.float32))
        densities.append(density.astype(np.float32))
        trace_points.append(points[:, :2].astype(np.float32))
        trace_offsets.append(trace_offsets[-1] + points.shape[0])
        trace_ids.append(fid)

    endpoint_map = np.zeros((height, width), dtype=np.uint8)
    junction_map = np.zeros((height, width), dtype=np.uint8)
    for node in geometry["nodes"]:
        code = NODE_TYPES[node["type"]]
        draw_disk(junction_map if code == NODE_TYPES["true_junction"] else endpoint_map, node["xyz"][:2], junction_radius if code == NODE_TYPES["true_junction"] else endpoint_radius)

    crossing_start = time.perf_counter()
    crossings, crossing_diagnostics = projected_crossings(geometry, return_diagnostics=True)
    crossing_elapsed = time.perf_counter() - crossing_start
    crossing_map = np.zeros((height, width), dtype=np.uint8)
    near_map = np.zeros((height, width), dtype=np.uint8)
    near_points = []
    for xy, z_pair in crossings:
        draw_disk(crossing_map, xy, crossing_radius)
        if abs(float(z_pair[0]) - float(z_pair[1])) <= near_coplanar_depth_px:
            draw_disk(near_map, xy, crossing_radius)
            near_points.append(xy)

    orientation_arrays = finalize_orientation_field(orientation_cos_sum, orientation_sin_sum, orientation_instance_count, config)

    return {
        "line_source_float": line_source,
        "geometric_support_preview": support_preview,
        "projection_mask": projection,
        "in_focus_mask": in_focus,
        "centerline_mask": centerline,
        "endpoint_map": endpoint_map,
        "junction_map": junction_map,
        "crossing_map": crossing_map,
        "projected_crossing_map": crossing_map.copy(),
        "near_coplanar_crossing_map": near_map,
        "overlap_count": overlap,
        "membership_y": np.concatenate(membership_y).astype(np.int32) if membership_y else np.zeros(0, dtype=np.int32),
        "membership_x": np.concatenate(membership_x).astype(np.int32) if membership_x else np.zeros(0, dtype=np.int32),
        "membership_instance_id": np.concatenate(membership_i).astype(np.int32) if membership_i else np.zeros(0, dtype=np.int32),
        "geometric_crossing_points_xy": np.asarray([c[0] for c in crossings], dtype=np.float32).reshape((-1, 2)),
        "projected_crossing_points_xy": np.asarray([c[0] for c in crossings], dtype=np.float32).reshape((-1, 2)),
        "projected_crossing_depths": np.asarray([c[1] for c in crossings], dtype=np.float32).reshape((-1, 2)),
        "near_coplanar_crossing_points_xy": np.asarray(near_points, dtype=np.float32).reshape((-1, 2)),
        "sample_arc_length_weight": np.concatenate(arc_weights).astype(np.float32) if arc_weights else np.zeros(0, dtype=np.float32),
        "fluorophore_density_per_length": np.concatenate(densities).astype(np.float32) if densities else np.zeros(0, dtype=np.float32),
        "trace_points_xy": np.vstack(trace_points).astype(np.float32) if trace_points else np.zeros((0, 2), dtype=np.float32),
        "trace_point_offsets": np.asarray(trace_offsets, dtype=np.int32),
        "trace_ids": np.asarray(trace_ids, dtype=np.int32),
        "trace_status": np.ones(len(trace_ids), dtype=np.int16),
        "trace_source": np.ones(len(trace_ids), dtype=np.int16),
        **orientation_arrays,
    }, {
        "source_raster_semantics": {
            "line_source_float": "nearest-pixel empirical line-source raster; integral equals sum(lambda_k * delta_s_k) for in-frame curve samples",
            "geometric_support_preview": "finite-radius visualization/support preview; integral has no source-energy interpretation",
        },
        "line_source_integrated_signal": float(np.sum(line_source)),
        "geometric_support_preview_integral": float(np.sum(support_preview)),
        "crossing_detection": {
            **crossing_diagnostics,
            "elapsed_seconds": crossing_elapsed,
            "method": "uniform_grid_candidate_pruning",
        },
    }


def splat_geometry_signal(
    geometry: dict[str, Any],
    config: dict[str, Any],
    visible_threshold: float,
    foreground_scale: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    height, width = geometry["image_shape"]
    total_in = np.zeros((height, width), dtype=np.float32)
    total_out = np.zeros((height, width), dtype=np.float32)
    total_core = np.zeros((height, width), dtype=np.float32)
    total_halo = np.zeros((height, width), dtype=np.float32)
    weighted_depth = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    nearest_abs = np.full((height, width), np.inf, dtype=np.float32)
    nearest_depth = np.full((height, width), -1.0, dtype=np.float32)
    visible_overlap = np.zeros((height, width), dtype=np.uint16)
    contributing_overlap = np.zeros((height, width), dtype=np.uint16)
    visible_y: list[np.ndarray] = []
    visible_x: list[np.ndarray] = []
    visible_i: list[np.ndarray] = []
    contributing_y: list[np.ndarray] = []
    contributing_x: list[np.ndarray] = []
    contributing_i: list[np.ndarray] = []
    kernel_stats = new_kernel_stats()
    total_source_signal = 0.0
    contributing_threshold = float(config.get("contributing_signal_threshold", 1e-6))

    for fiber in geometry["fibers"]:
        ds = sample_arc_length_weights(fiber["points_xyz"])
        total_source_signal += float(np.sum(fiber["sample_amplitude"] * ds))
        fiber_in, fiber_out, fiber_core, fiber_halo, contrib_depth, contrib_weight, contrib_nearest_abs, contrib_nearest_depth, stats = splat_points(
            fiber["points_xyz"],
            fiber["sample_amplitude"],
            ds,
            float(geometry["focal_plane_z_px"]),
            config,
            (height, width),
        )
        merge_kernel_stats(kernel_stats, stats)
        total_in += fiber_in
        total_out += fiber_out
        total_core += fiber_core
        total_halo += fiber_halo
        weighted_depth += contrib_depth
        weight_sum += contrib_weight
        closer = contrib_nearest_abs < nearest_abs
        nearest_abs[closer] = contrib_nearest_abs[closer]
        nearest_depth[closer] = contrib_nearest_depth[closer]
        fiber_signal = (fiber_in + fiber_out) * foreground_scale
        visible = fiber_signal > visible_threshold
        contributing = fiber_signal > contributing_threshold
        visible_overlap += visible.astype(np.uint16)
        contributing_overlap += contributing.astype(np.uint16)
        y, x = np.nonzero(visible)
        visible_y.append(y.astype(np.int32))
        visible_x.append(x.astype(np.int32))
        visible_i.append(np.full(y.shape, int(fiber["fiber_id"]), dtype=np.int32))
        cy, cx = np.nonzero(contributing)
        contributing_y.append(cy.astype(np.int32))
        contributing_x.append(cx.astype(np.int32))
        contributing_i.append(np.full(cy.shape, int(fiber["fiber_id"]), dtype=np.int32))

    optical_total = total_in + total_out
    total = optical_total * foreground_scale
    total_in_scaled = total_in * foreground_scale
    total_out_scaled = total_out * foreground_scale
    total_core_scaled = total_core * foreground_scale
    total_halo_scaled = total_halo * foreground_scale
    mean_depth = np.full((height, width), -1.0, dtype=np.float32)
    mask = weight_sum > 0
    mean_depth[mask] = (weighted_depth[mask] / weight_sum[mask]).astype(np.float32)
    nearest_depth[~mask] = -1.0
    visible_signal = (total > visible_threshold).astype(np.uint8)
    combined_only_visible = ((visible_signal > 0) & (visible_overlap == 0)).astype(np.uint8)
    return {
        "in_focus_signal": total_in_scaled.astype(np.float32),
        "out_of_focus_signal": total_out_scaled.astype(np.float32),
        "total_clean_signal": total.astype(np.float32),
        "total_optical_signal": optical_total.astype(np.float32),
        "core_signal": total_core_scaled.astype(np.float32),
        "halo_signal": total_halo_scaled.astype(np.float32),
        "visible_signal_mask": visible_signal,
        "visible_overlap_count": visible_overlap,
        "visible_membership_y": np.concatenate(visible_y).astype(np.int32) if visible_y else np.zeros(0, dtype=np.int32),
        "visible_membership_x": np.concatenate(visible_x).astype(np.int32) if visible_x else np.zeros(0, dtype=np.int32),
        "visible_membership_instance_id": np.concatenate(visible_i).astype(np.int32) if visible_i else np.zeros(0, dtype=np.int32),
        "visible_instance_membership_y": np.concatenate(visible_y).astype(np.int32) if visible_y else np.zeros(0, dtype=np.int32),
        "visible_instance_membership_x": np.concatenate(visible_x).astype(np.int32) if visible_x else np.zeros(0, dtype=np.int32),
        "visible_instance_membership_id": np.concatenate(visible_i).astype(np.int32) if visible_i else np.zeros(0, dtype=np.int32),
        "contributing_membership_y": np.concatenate(contributing_y).astype(np.int32) if contributing_y else np.zeros(0, dtype=np.int32),
        "contributing_membership_x": np.concatenate(contributing_x).astype(np.int32) if contributing_x else np.zeros(0, dtype=np.int32),
        "contributing_membership_instance_id": np.concatenate(contributing_i).astype(np.int32) if contributing_i else np.zeros(0, dtype=np.int32),
        "contributing_overlap_count": contributing_overlap,
        "combined_only_visible_mask": combined_only_visible,
        "visible_unassigned_mask": combined_only_visible.copy(),
        "nearest_depth_map": nearest_depth.astype(np.float32),
        "weighted_mean_depth_map": mean_depth.astype(np.float32),
    }, {
        "psf_mode": config.get("psf_mode", "single_gaussian"),
        "psf_normalization": "unit_integral",
        "sigma_xy_0_px": float(config.get("sigma_xy_0_px", 2.123)),
        "sigma_z_px": float(config.get("sigma_z_px", 24.0)),
        "defocus_broadening": float(config.get("defocus_broadening", 0.025)),
        "kernel_truncation_sigma": float(config.get("kernel_truncation_radius", 3.0)),
        "kernel_truncation_radius": float(config.get("kernel_truncation_radius", 3.0)),
        "focal_plane_z_px": float(geometry["focal_plane_z_px"]),
        "sampling_density": "one splat per arc-length-resampled curve point",
        "sample_arc_length_weighting": True,
        "splatting_convention": "arc-length-weighted unit-integral finite-support 2D Gaussian at pixel centers",
        "visible_signal_threshold": visible_threshold,
        "contributing_signal_threshold": contributing_threshold,
        "visible_instance_membership_semantics": "instances whose individual scaled clean signal exceeds visible_signal_threshold",
        "contributing_instance_membership_semantics": "instances whose individual scaled clean signal exceeds contributing_signal_threshold",
        "combined_only_visible_mask_semantics": "pixels where summed clean signal exceeds visible threshold but no individual instance does",
        "line_density_units": "empirical_signal_units_per_pixel_equivalent_contour_length",
        "total_represented_contour_length_px": float(sum(np.sum(sample_arc_length_weights(f["points_xyz"])) for f in geometry["fibers"])),
        "total_emitted_source_signal": total_source_signal,
        "total_projected_optical_signal": float(np.sum(optical_total)),
        "integrated_foreground_signal": float(np.sum(total)),
        "core_integrated_signal": float(np.sum(total_core_scaled)),
        "halo_integrated_signal": float(np.sum(total_halo_scaled)),
        **finalize_kernel_stats(kernel_stats),
    }


def splat_points(
    points_xyz: np.ndarray,
    densities: np.ndarray,
    arc_length_weights: np.ndarray | None,
    focal_z: float,
    config: dict[str, Any],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = shape
    in_focus = np.zeros((height, width), dtype=np.float32)
    out_focus = np.zeros((height, width), dtype=np.float32)
    core = np.zeros((height, width), dtype=np.float32)
    halo = np.zeros((height, width), dtype=np.float32)
    weighted_depth = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    nearest_abs = np.full((height, width), np.inf, dtype=np.float32)
    nearest_depth = np.full((height, width), -1.0, dtype=np.float32)
    focal_depth = float(config.get("focal_depth_range_px", 6.0))
    stats = new_kernel_stats()
    if arc_length_weights is None:
        arc_length_weights = sample_arc_length_weights(points_xyz)

    for point, density, ds in zip(points_xyz, densities, arc_length_weights):
        dz = float(point[2] - focal_z)
        target = in_focus if abs(dz) <= focal_depth else out_focus
        for component in psf_components(config, dz):
            comp_target = core if component["name"] == "core" else halo if component["name"] == "halo" else core
            _splat_component(target, comp_target, weighted_depth, weight_sum, nearest_abs, nearest_depth, point, float(density) * float(ds), dz, component, stats)
    return in_focus, out_focus, core, halo, weighted_depth, weight_sum, nearest_abs, nearest_depth, stats


def psf_components(config: dict[str, Any], dz: float) -> list[dict[str, float | str]]:
    mode = config.get("psf_mode", "single_gaussian")
    if mode in {"core_halo", "core_plus_halo"}:
        specs = [
            ("core", float(config.get("core_weight", 0.85)), float(config.get("core_sigma_xy_0_px", config.get("sigma_xy_0_px", 2.123))), float(config.get("core_sigma_z_px", config.get("sigma_z_px", 24.0))), float(config.get("core_defocus_broadening", config.get("defocus_broadening", 0.025)))),
            ("halo", float(config.get("halo_weight", 0.15)), float(config.get("halo_sigma_xy_0_px", 6.0)), float(config.get("halo_sigma_z_px", 45.0)), float(config.get("halo_defocus_broadening", 0.055))),
        ]
    else:
        specs = [("single", 1.0, float(config.get("sigma_xy_0_px", 2.123)), float(config.get("sigma_z_px", 24.0)), float(config.get("defocus_broadening", 0.025)))]
    out = []
    for _, weight, sigma0, sigma_z, broadening in specs:
        axial = math.exp(-(dz * dz) / (2.0 * sigma_z * sigma_z)) if sigma_z > 0 else 1.0
        sigma_xy = math.sqrt(sigma0 * sigma0 + (broadening * dz) ** 2)
        out.append({"name": _, "weight": weight, "axial": axial, "sigma_xy": sigma_xy, "truncation": float(config.get("kernel_truncation_radius", 3.0))})
    return out


def _splat_component(
    image: np.ndarray,
    component_image: np.ndarray,
    weighted_depth: np.ndarray,
    weight_sum: np.ndarray,
    nearest_abs: np.ndarray,
    nearest_depth: np.ndarray,
    point: np.ndarray,
    source_signal: float,
    dz: float,
    component: dict[str, float | str],
    stats: dict[str, Any],
) -> None:
    height, width = image.shape
    sigma = max(float(component["sigma_xy"]), 1e-3)
    radius = float(component["truncation"]) * sigma
    x, y, z = map(float, point)
    full_min_x = int(math.floor(x - radius - 1))
    full_max_x = int(math.ceil(x + radius + 1))
    full_min_y = int(math.floor(y - radius - 1))
    full_max_y = int(math.ceil(y + radius + 1))
    if full_max_x < full_min_x or full_max_y < full_min_y:
        return
    yy, xx = np.mgrid[full_min_y : full_max_y + 1, full_min_x : full_max_x + 1]
    dist2 = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2
    inside = dist2 <= radius * radius
    kernel = np.where(inside, np.exp(-dist2 / (2.0 * sigma * sigma)), 0.0).astype(np.float64)
    sum_before = float(kernel.sum())
    if not np.isfinite(sum_before) or sum_before <= 0:
        return
    kernel /= sum_before
    sum_after = float(kernel.sum())
    min_x = max(0, full_min_x)
    max_x = min(width - 1, full_max_x)
    min_y = max(0, full_min_y)
    max_y = min(height - 1, full_max_y)
    if max_x < min_x or max_y < min_y:
        record_kernel_stats(stats, sum_before, sum_after, 0.0, True)
        return
    ky0 = min_y - full_min_y
    ky1 = ky0 + (max_y - min_y + 1)
    kx0 = min_x - full_min_x
    kx1 = kx0 + (max_x - min_x + 1)
    kernel_frame = kernel[ky0:ky1, kx0:kx1]
    inside_frame = inside[ky0:ky1, kx0:kx1]
    in_frame_sum = float(kernel_frame.sum())
    clipped = full_min_x < 0 or full_min_y < 0 or full_max_x >= width or full_max_y >= height
    record_kernel_stats(stats, sum_before, sum_after, in_frame_sum, clipped)
    patch = (source_signal * float(component["weight"]) * float(component["axial"]) * kernel_frame).astype(np.float32)
    target = image[min_y : max_y + 1, min_x : max_x + 1]
    target += patch
    component_target = component_image[min_y : max_y + 1, min_x : max_x + 1]
    component_target += patch
    w_patch = weight_sum[min_y : max_y + 1, min_x : max_x + 1]
    d_patch = weighted_depth[min_y : max_y + 1, min_x : max_x + 1]
    w_patch += patch
    d_patch += patch * z
    n_abs = nearest_abs[min_y : max_y + 1, min_x : max_x + 1]
    n_depth = nearest_depth[min_y : max_y + 1, min_x : max_x + 1]
    closer = inside_frame & (abs(dz) < n_abs)
    n_abs[closer] = abs(dz)
    n_depth[closer] = z


def sample_arc_length_weights(points_xyz: np.ndarray) -> np.ndarray:
    n = int(points_xyz.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)
    seg = np.linalg.norm(np.diff(points_xyz.astype(np.float64), axis=0), axis=1)
    weights = np.zeros(n, dtype=np.float64)
    weights[0] = 0.5 * seg[0]
    weights[-1] = 0.5 * seg[-1]
    if n > 2:
        weights[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    return weights.astype(np.float32)


def new_kernel_stats() -> dict[str, Any]:
    return {
        "kernel_count": 0,
        "kernel_sum_before_min": math.inf,
        "kernel_sum_before_max": 0.0,
        "kernel_sum_after_min": math.inf,
        "kernel_sum_after_max": 0.0,
        "kernel_normalization_error_max": 0.0,
        "kernel_in_frame_sum_min": math.inf,
        "kernel_in_frame_sum_max": 0.0,
        "kernel_out_of_frame_fraction_max": 0.0,
        "kernel_boundary_clipped_count": 0,
    }


def record_kernel_stats(stats: dict[str, Any], before: float, after: float, in_frame_sum: float, boundary_clipped: bool) -> None:
    stats["kernel_count"] += 1
    stats["kernel_sum_before_min"] = min(float(stats["kernel_sum_before_min"]), before)
    stats["kernel_sum_before_max"] = max(float(stats["kernel_sum_before_max"]), before)
    stats["kernel_sum_after_min"] = min(float(stats["kernel_sum_after_min"]), after)
    stats["kernel_sum_after_max"] = max(float(stats["kernel_sum_after_max"]), after)
    stats["kernel_normalization_error_max"] = max(float(stats["kernel_normalization_error_max"]), abs(after - 1.0))
    stats["kernel_in_frame_sum_min"] = min(float(stats["kernel_in_frame_sum_min"]), in_frame_sum)
    stats["kernel_in_frame_sum_max"] = max(float(stats["kernel_in_frame_sum_max"]), in_frame_sum)
    stats["kernel_out_of_frame_fraction_max"] = max(float(stats["kernel_out_of_frame_fraction_max"]), max(0.0, 1.0 - in_frame_sum))
    stats["kernel_boundary_clipped_count"] += int(boundary_clipped)


def merge_kernel_stats(total: dict[str, Any], part: dict[str, Any]) -> None:
    if int(part.get("kernel_count", 0)) == 0:
        return
    total["kernel_count"] += int(part["kernel_count"])
    total["kernel_sum_before_min"] = min(float(total["kernel_sum_before_min"]), float(part["kernel_sum_before_min"]))
    total["kernel_sum_before_max"] = max(float(total["kernel_sum_before_max"]), float(part["kernel_sum_before_max"]))
    total["kernel_sum_after_min"] = min(float(total["kernel_sum_after_min"]), float(part["kernel_sum_after_min"]))
    total["kernel_sum_after_max"] = max(float(total["kernel_sum_after_max"]), float(part["kernel_sum_after_max"]))
    total["kernel_normalization_error_max"] = max(float(total["kernel_normalization_error_max"]), float(part["kernel_normalization_error_max"]))
    total["kernel_in_frame_sum_min"] = min(float(total["kernel_in_frame_sum_min"]), float(part["kernel_in_frame_sum_min"]))
    total["kernel_in_frame_sum_max"] = max(float(total["kernel_in_frame_sum_max"]), float(part["kernel_in_frame_sum_max"]))
    total["kernel_out_of_frame_fraction_max"] = max(float(total["kernel_out_of_frame_fraction_max"]), float(part["kernel_out_of_frame_fraction_max"]))
    total["kernel_boundary_clipped_count"] += int(part["kernel_boundary_clipped_count"])


def finalize_kernel_stats(stats: dict[str, Any]) -> dict[str, Any]:
    if int(stats["kernel_count"]) == 0:
        return {
            "kernel_count": 0,
            "kernel_sum_before_normalization": "not_reported",
            "kernel_sum_after_normalization": "not_reported",
            "kernel_normalization_error": "not_reported",
            "kernel_in_frame_retained_sum": "not_reported",
            "kernel_out_of_frame_fraction_max": "not_reported",
            "kernel_boundary_clipped_count": 0,
        }
    return {
        "kernel_count": int(stats["kernel_count"]),
        "kernel_sum_before_normalization": {
            "min": float(stats["kernel_sum_before_min"]),
            "max": float(stats["kernel_sum_before_max"]),
        },
        "kernel_sum_after_normalization": {
            "min": float(stats["kernel_sum_after_min"]),
            "max": float(stats["kernel_sum_after_max"]),
        },
        "kernel_normalization_error": float(stats["kernel_normalization_error_max"]),
        "kernel_in_frame_retained_sum": {
            "min": float(stats["kernel_in_frame_sum_min"]),
            "max": float(stats["kernel_in_frame_sum_max"]),
        },
        "kernel_out_of_frame_fraction_max": float(stats["kernel_out_of_frame_fraction_max"]),
        "kernel_boundary_clipped_count": int(stats["kernel_boundary_clipped_count"]),
        "image_boundary_convention": "full finite-support discrete kernel is unit-normalized before image clipping; out-of-frame signal is lost, never renormalized",
    }


def rasterize_variable_disks(points_xy: np.ndarray, radii: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for xy, radius in zip(points_xy, radii):
        add_disk_bool(mask, xy, float(radius))
    return mask


def add_source_disks(source: np.ndarray, points_xy: np.ndarray, radii: np.ndarray, amplitudes: np.ndarray) -> None:
    for xy, radius, amp in zip(points_xy, radii, amplitudes):
        add_disk_value(source, xy, float(radius), float(amp))


def add_line_source(source: np.ndarray, points_xy: np.ndarray, signal: np.ndarray) -> None:
    height, width = source.shape
    for xy, value in zip(points_xy, signal):
        x = int(math.floor(float(xy[0])))
        y = int(math.floor(float(xy[1])))
        if 0 <= x < width and 0 <= y < height:
            source[y, x] += float(value)


def add_orientation_contribution(cos_sum: np.ndarray, sin_sum: np.ndarray, instance_count: np.ndarray, points_xy: np.ndarray, radius: float) -> None:
    if points_xy.shape[0] < 2:
        return
    fiber_cos = np.zeros_like(cos_sum, dtype=np.float32)
    fiber_sin = np.zeros_like(sin_sum, dtype=np.float32)
    fiber_count = np.zeros_like(instance_count, dtype=np.uint16)
    tangents = np.zeros_like(points_xy, dtype=np.float32)
    tangents[0] = points_xy[1] - points_xy[0]
    tangents[-1] = points_xy[-1] - points_xy[-2]
    if points_xy.shape[0] > 2:
        tangents[1:-1] = points_xy[2:] - points_xy[:-2]
    for xy, tangent in zip(points_xy, tangents):
        if float(np.dot(tangent, tangent)) <= 1e-8:
            continue
        theta = math.atan2(float(tangent[1]), float(tangent[0]))
        y_slice, x_slice, inside = disk_patch(cos_sum.shape, xy, radius)
        if inside.size:
            fiber_cos[y_slice, x_slice][inside] += math.cos(2.0 * theta)
            fiber_sin[y_slice, x_slice][inside] += math.sin(2.0 * theta)
            fiber_count[y_slice, x_slice][inside] += 1
    mask = fiber_count > 0
    if not np.any(mask):
        return
    mean_cos = np.zeros_like(fiber_cos)
    mean_sin = np.zeros_like(fiber_sin)
    mean_cos[mask] = fiber_cos[mask] / fiber_count[mask]
    mean_sin[mask] = fiber_sin[mask] / fiber_count[mask]
    mag = np.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)
    valid = mask & (mag > 1e-6)
    mean_cos[valid] /= mag[valid]
    mean_sin[valid] /= mag[valid]
    cos_sum[valid] += mean_cos[valid]
    sin_sum[valid] += mean_sin[valid]
    instance_count[valid] += 1


def finalize_orientation_field(cos_sum: np.ndarray, sin_sum: np.ndarray, instance_count: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    consensus = float(config.get("orientation_consensus_threshold", 0.95))
    cos2 = np.zeros_like(cos_sum, dtype=np.float32)
    sin2 = np.zeros_like(sin_sum, dtype=np.float32)
    valid_mask = np.zeros_like(instance_count, dtype=np.uint8)
    mask = instance_count > 0
    mag = np.zeros_like(cos_sum, dtype=np.float32)
    mag[mask] = np.sqrt(cos_sum[mask] ** 2 + sin_sum[mask] ** 2) / np.maximum(instance_count[mask], 1)
    valid = mask & ((instance_count == 1) | (mag >= consensus))
    norm = np.sqrt(cos_sum[valid] ** 2 + sin_sum[valid] ** 2)
    cos2[valid] = cos_sum[valid] / np.maximum(norm, 1e-6)
    sin2[valid] = sin_sum[valid] / np.maximum(norm, 1e-6)
    valid_mask[valid] = 1
    return {
        "orientation_cos2theta": cos2,
        "orientation_sin2theta": sin2,
        "orientation_valid_mask": valid_mask,
        "orientation_instance_count": instance_count.astype(np.uint16),
    }


def add_disk_bool(mask: np.ndarray, xy: np.ndarray, radius: float) -> None:
    y_slice, x_slice, inside = disk_patch(mask.shape, xy, radius)
    if inside.size:
        mask[y_slice, x_slice] |= inside


def add_disk_value(image: np.ndarray, xy: np.ndarray, radius: float, value: float) -> None:
    y_slice, x_slice, inside = disk_patch(image.shape, xy, radius)
    if inside.size:
        image[y_slice, x_slice] += np.where(inside, value, 0.0).astype(np.float32)


def disk_patch(shape: tuple[int, int], xy: np.ndarray, radius: float) -> tuple[slice, slice, np.ndarray]:
    height, width = shape
    x, y = float(xy[0]), float(xy[1])
    min_x = max(0, int(math.floor(x - radius - 1)))
    max_x = min(width - 1, int(math.ceil(x + radius + 1)))
    min_y = max(0, int(math.floor(y - radius - 1)))
    max_y = min(height - 1, int(math.ceil(y + radius + 1)))
    if max_x < min_x or max_y < min_y:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0), dtype=bool)
    yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    inside = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= radius * radius
    return slice(min_y, max_y + 1), slice(min_x, max_x + 1), inside


def projected_crossings(geometry: dict[str, Any], return_diagnostics: bool = False) -> Any:
    crossings, diagnostics = projected_crossings_grid(geometry)
    if return_diagnostics:
        return crossings, diagnostics
    return crossings


def projected_crossings_grid(geometry: dict[str, Any], cell_size: float = 32.0) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    shared = {int(edge["fiber_id"]): set(map(int, edge["node_indices"])) for edge in geometry["edges"]}
    segments = []
    for fiber in geometry["fibers"]:
        fid = int(fiber["fiber_id"])
        for seg_index, (p0, p1) in enumerate(zip(fiber["points_xyz"][:-1], fiber["points_xyz"][1:])):
            segments.append((fid, seg_index, p0, p1))
    brute_force_pairs = 0
    for i, a in enumerate(segments):
        for b in segments[i + 1 :]:
            if a[0] != b[0]:
                brute_force_pairs += 1
    bins: dict[tuple[int, int], list[int]] = {}
    for index, (_, _, p0, p1) in enumerate(segments):
        min_x = int(math.floor(min(float(p0[0]), float(p1[0])) / cell_size))
        max_x = int(math.floor(max(float(p0[0]), float(p1[0])) / cell_size))
        min_y = int(math.floor(min(float(p0[1]), float(p1[1])) / cell_size))
        max_y = int(math.floor(max(float(p0[1]), float(p1[1])) / cell_size))
        for gx in range(min_x, max_x + 1):
            for gy in range(min_y, max_y + 1):
                bins.setdefault((gx, gy), []).append(index)
    candidate_pairs: set[tuple[int, int]] = set()
    for indices in bins.values():
        ordered = sorted(set(indices))
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                if segments[a][0] != segments[b][0]:
                    candidate_pairs.add((a, b))
    crossings: list[tuple[np.ndarray, np.ndarray]] = []
    for ia, ib in sorted(candidate_pairs):
        fa, _, a0, a1 = segments[ia]
        fb, _, b0, b1 = segments[ib]
        if shared.get(fa, set()) & shared.get(fb, set()):
            continue
        hit = segment_intersection(a0[:2], a1[:2], b0[:2], b1[:2])
        if hit is not None and not _near_crossing(crossings, hit):
            za = _interp_z_at_hit(a0, a1, hit)
            zb = _interp_z_at_hit(b0, b1, hit)
            crossings.append((hit.astype(np.float32), np.asarray([za, zb], dtype=np.float32)))
    diagnostics = {
        "segment_count": len(segments),
        "brute_force_pair_count": brute_force_pairs,
        "candidate_pair_count": len(candidate_pairs),
        "candidate_pair_reduction": brute_force_pairs - len(candidate_pairs),
    }
    return crossings, diagnostics


def projected_crossings_bruteforce(geometry: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    shared = {int(edge["fiber_id"]): set(map(int, edge["node_indices"])) for edge in geometry["edges"]}
    crossings: list[tuple[np.ndarray, np.ndarray]] = []
    fibers = geometry["fibers"]
    for i, fa in enumerate(fibers):
        for fb in fibers[i + 1 :]:
            if shared.get(int(fa["fiber_id"]), set()) & shared.get(int(fb["fiber_id"]), set()):
                continue
            for a0, a1 in zip(fa["points_xyz"][:-1], fa["points_xyz"][1:]):
                for b0, b1 in zip(fb["points_xyz"][:-1], fb["points_xyz"][1:]):
                    hit = segment_intersection(a0[:2], a1[:2], b0[:2], b1[:2])
                    if hit is not None and not _near_crossing(crossings, hit):
                        za = _interp_z_at_hit(a0, a1, hit)
                        zb = _interp_z_at_hit(b0, b1, hit)
                        crossings.append((hit.astype(np.float32), np.asarray([za, zb], dtype=np.float32)))
    return crossings


def _interp_z_at_hit(p0: np.ndarray, p1: np.ndarray, hit_xy: np.ndarray) -> float:
    seg = p1[:2] - p0[:2]
    denom = float(np.dot(seg, seg))
    t = 0.0 if denom <= 1e-8 else float(np.clip(np.dot(hit_xy - p0[:2], seg) / denom, 0, 1))
    return float(p0[2] + t * (p1[2] - p0[2]))


def _near_crossing(crossings: list[tuple[np.ndarray, np.ndarray]], xy: np.ndarray, tol: float = 2.0) -> bool:
    return any(float(np.sum((xy - c[0]) ** 2)) <= tol * tol for c in crossings)


def select_semantic_mask(arrays: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    source = config.get("semantic_mask_source", "visible_signal_mask")
    if source not in {"projection_mask", "in_focus_mask", "visible_signal_mask"}:
        raise ValueError(f"invalid semantic_mask_source: {source}")
    return arrays[source].astype(np.uint8)


def scenario_category(scenario: str, config: dict[str, Any] | None = None) -> str:
    if config and config.get("scenario_category"):
        value = str(config["scenario_category"])
        if value not in {"structural_qa", "optical_qa", "realism_calibration"}:
            raise ValueError(f"invalid scenario_category: {value}")
        return value
    if scenario in STRUCTURAL_QA_SCENARIOS:
        return "structural_qa"
    if scenario in OPTICAL_QA_SCENARIOS:
        return "optical_qa"
    return "realism_calibration"


def build_ignore_mask(arrays: dict[str, np.ndarray], config: dict[str, Any], visible_threshold: float) -> np.ndarray:
    rule = config.get("ignore_mask_rule", "none")
    if rule in {"none", None, False}:
        return np.zeros_like(arrays["semantic_mask"], dtype=np.uint8)
    signal = arrays["total_clean_signal"]
    if rule == "near_visibility_threshold":
        lo = float(config.get("ignore_visibility_low_factor", 0.75)) * visible_threshold
        hi = float(config.get("ignore_visibility_high_factor", 1.25)) * visible_threshold
        return ((signal >= lo) & (signal <= hi)).astype(np.uint8)
    if rule == "defocused_only":
        return ((arrays["out_of_focus_signal"] > visible_threshold) & (arrays["in_focus_signal"] <= visible_threshold)).astype(np.uint8)
    if rule == "optical_overlap":
        return (arrays["visible_overlap_count"] > 1).astype(np.uint8)
    raise ValueError(f"invalid ignore_mask_rule: {rule}")


def apply_optional_targets(arrays: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    target_available = {
        "semantic_mask": True,
        "centerline_mask": True,
        "ignore_mask": True,
        "background_distance_to_semantic_foreground": bool(config.get("distance_transform_enabled", True)),
        "endpoint_map": True,
        "junction_map": True,
        "crossing_map": True,
        "orientation": bool(config.get("orientation_field_enabled", True)),
        "instance_membership": True,
        "visible_instance_membership": True,
        "contributing_instance_membership": True,
        "depth_maps": True,
        "optical_signal_decomposition": True,
    }
    if target_available["background_distance_to_semantic_foreground"]:
        start = time.perf_counter()
        arrays["background_distance_to_semantic_foreground"] = background_distance_to_foreground(arrays["semantic_mask"])
        distance_seconds = time.perf_counter() - start
    else:
        distance_seconds = 0.0
    if not target_available["orientation"]:
        for name in ["orientation_cos2theta", "orientation_sin2theta", "orientation_valid_mask", "orientation_instance_count"]:
            arrays.pop(name, None)
    return {
        "target_available": target_available,
        "distance_target_semantics": {
            "array": "background_distance_to_semantic_foreground" if target_available["background_distance_to_semantic_foreground"] else "not_generated",
            "source_mask": "semantic_mask",
            "zero_convention": "zero inside semantic foreground; Euclidean pixel distance outside to nearest semantic foreground pixel",
            "units": "pixels",
        },
        "distance_transform_elapsed_seconds": distance_seconds,
        "orientation_target_semantics": {
            "encoding": "cos(2 theta), sin(2 theta)",
            "valid_mask": "orientation_valid_mask marks single-instance or consensus-compatible orientation pixels",
            "invalid_value": "orientation components are zero where invalid or disabled",
            "consensus_threshold": float(config.get("orientation_consensus_threshold", 0.95)),
        },
    }


def background_distance_to_foreground(mask: np.ndarray) -> np.ndarray:
    foreground = mask.astype(bool)
    if scipy_ndimage is not None:
        return scipy_ndimage.distance_transform_edt(~foreground).astype(np.float32)
    return chamfer_distance(foreground).astype(np.float32)


def chamfer_distance(foreground: np.ndarray) -> np.ndarray:
    """Fallback approximate background distance if scipy is unavailable."""

    inf = 1_000_000.0
    dist = np.where(foreground, 0.0, inf).astype(np.float32)
    h, w = dist.shape
    for y in range(h):
        for x in range(w):
            best = dist[y, x]
            if y > 0:
                best = min(best, dist[y - 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y - 1, x - 1] + math.sqrt(2.0))
                if x + 1 < w:
                    best = min(best, dist[y - 1, x + 1] + math.sqrt(2.0))
            if x > 0:
                best = min(best, dist[y, x - 1] + 1.0)
            dist[y, x] = best
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            best = dist[y, x]
            if y + 1 < h:
                best = min(best, dist[y + 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y + 1, x - 1] + math.sqrt(2.0))
                if x + 1 < w:
                    best = min(best, dist[y + 1, x + 1] + math.sqrt(2.0))
            if x + 1 < w:
                best = min(best, dist[y, x + 1] + 1.0)
            dist[y, x] = best
    return dist


def measure_isolated_fiber_fwhm(config: dict[str, Any], angle_degrees: float | None = None, subpixel_shift: tuple[float, float] = (0.0, 0.0)) -> dict[str, float]:
    key = json.dumps(
        {
            "geometry": config.get("geometry", {}),
            "targets": config.get("targets", {}),
            "optical_model": config.get("optical_model", {}),
            "output_mapping": config.get("output_mapping", {}),
            "width_calibration": config.get("width_calibration", {}),
            "angle_degrees": angle_degrees,
            "subpixel_shift": list(subpixel_shift),
        },
        sort_keys=True,
        default=str,
    )
    return dict(_measure_isolated_fiber_fwhm_cached(key))


@lru_cache(maxsize=64)
def _measure_isolated_fiber_fwhm_cached(config_key: str) -> tuple[tuple[str, float], ...]:
    payload = json.loads(config_key)
    config = {
        "geometry": payload["geometry"],
        "targets": payload["targets"],
        "optical_model": payload["optical_model"],
        "output_mapping": payload["output_mapping"],
        "width_calibration": payload["width_calibration"],
    }
    result = _measure_isolated_fiber_fwhm_uncached(config, payload["angle_degrees"], tuple(payload["subpixel_shift"]))
    return tuple(sorted(result.items()))


def _measure_isolated_fiber_fwhm_uncached(config: dict[str, Any], angle_degrees: float | None = None, subpixel_shift: tuple[float, float] = (0.0, 0.0)) -> dict[str, float]:
    geometry_config = dict(config.get("geometry", {}))
    width_cfg = config.get("width_calibration", {})
    geometry_config["scenario"] = "straight_width_calibration"
    if angle_degrees is not None:
        geometry_config.setdefault("width_calibration", dict(width_cfg))
        geometry_config["width_calibration"]["angle_degrees"] = angle_degrees
    geometry = generate_persistent_chain_geometry(geometry_config, 0)
    if subpixel_shift != (0.0, 0.0):
        shift = np.asarray([subpixel_shift[0], subpixel_shift[1], 0.0], dtype=np.float32)
        for fiber in geometry["fibers"]:
            fiber["points_xyz"] += shift
            fiber["points_xy"] = fiber["points_xyz"][:, :2]
            fiber["raw_vertices_xyz"] += shift
        for node in geometry["nodes"]:
            node["xyz"] += shift
    arrays, _ = rasterize_3d_sample(geometry, config.get("targets", {}), config.get("optical_model", {}), config.get("output_mapping", {}), 0)
    angle = math.radians(float(geometry_config.get("width_calibration", width_cfg).get("angle_degrees", 0.0) if angle_degrees is None else angle_degrees))
    center = geometry["fibers"][0]["points_xyz"][len(geometry["fibers"][0]["points_xyz"]) // 2, :2]
    profile_x, profile = sample_transverse_profile(arrays["total_clean_signal"], center, angle)
    return {
        "measured_fwhm_px": fwhm(profile_x, profile),
        "target_fwhm_px": float(width_cfg.get("target_fwhm_px", 5.0)),
        "min_allowed_fwhm_px": float(width_cfg.get("min_allowed_fwhm_px", 4.5)),
        "max_allowed_fwhm_px": float(width_cfg.get("max_allowed_fwhm_px", 5.5)),
        "integrated_transverse_signal": float(np.trapezoid(profile, profile_x)),
        "peak_transverse_signal": float(np.max(profile)),
        "line_density_amplitude": float(np.mean(geometry["fibers"][0]["sample_amplitude"])),
        "sampling_interval_px": float(geometry_config.get("arc_length_sampling_interval_px", 1.0)),
    }


def sample_transverse_profile(image: np.ndarray, center_xy: np.ndarray, tangent_angle: float, half_width: int = 24) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray([-math.sin(tangent_angle), math.cos(tangent_angle)], dtype=np.float32)
    offsets = np.linspace(-half_width, half_width, 2 * half_width + 1, dtype=np.float32)
    values = np.asarray([bilinear(image, center_xy + normal * off) for off in offsets], dtype=np.float32)
    return offsets, values


def bilinear(image: np.ndarray, xy: np.ndarray) -> float:
    x, y = float(xy[0]), float(xy[1])
    h, w = image.shape
    if x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return 0.0
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    return float((1 - dx) * (1 - dy) * image[y0, x0] + dx * (1 - dy) * image[y0, x1] + (1 - dx) * dy * image[y1, x0] + dx * dy * image[y1, x1])


def fwhm(x: np.ndarray, y: np.ndarray) -> float:
    baseline = float(np.min(y))
    peak = float(np.max(y))
    half = baseline + 0.5 * (peak - baseline)
    above = np.flatnonzero(y >= half)
    if above.size < 2:
        return 0.0
    left = _crossing(x, y, half, above[0] - 1, above[0])
    right = _crossing(x, y, half, above[-1], above[-1] + 1)
    return float(right - left)


def _crossing(x: np.ndarray, y: np.ndarray, half: float, i0: int, i1: int) -> float:
    i0 = int(np.clip(i0, 0, len(x) - 1))
    i1 = int(np.clip(i1, 0, len(x) - 1))
    if i0 == i1 or y[i1] == y[i0]:
        return float(x[i1])
    t = (half - y[i0]) / (y[i1] - y[i0])
    return float(x[i0] + t * (x[i1] - x[i0]))
