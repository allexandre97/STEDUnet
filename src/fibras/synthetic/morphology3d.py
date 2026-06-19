"""Condition-blind heterogeneous 3D morphology scenes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geometry3d import (
    _handle_boundary,
    _persistent_tangent,
    _random_unit,
    _uniform,
    fluorophore_samples,
    resample_by_arc_length,
    smooth_catmull_rom,
)
from .schema import (
    FIBER_STRUCTURE_TYPE_CODES,
    NODE_TYPES,
    TRACE_TERMINATION_STATUS_CODES,
)


SCENE_MODES = {
    "isolated_filaments",
    "clustered_filament_network",
    "aligned_filament_domain",
    "dense_tangle",
    "bundle_dominated",
    "clump_dominated",
    "mixed_morphology",
}

MODE_COUNTS = {
    "isolated_filaments": ((4, 7), (0, 0), (0, 0)),
    "clustered_filament_network": ((10, 16), (0, 1), (0, 0)),
    "aligned_filament_domain": ((9, 14), (0, 1), (0, 0)),
    "dense_tangle": ((18, 26), (0, 1), (0, 1)),
    "bundle_dominated": ((3, 7), (2, 4), (0, 1)),
    "clump_dominated": ((3, 7), (0, 1), (2, 4)),
    "mixed_morphology": ((7, 13), (1, 3), (1, 3)),
}


def generate_morphology_geometry(
    config: dict[str, Any], sample_index: int
) -> dict[str, Any]:
    geometry_config = config["geometry"]
    scene_config = config.get("scene_morphology", {})
    rng = np.random.default_rng(
        int(geometry_config.get("base_seed", 81001)) + sample_index
    )
    height, width = map(int, geometry_config.get("image_shape", [1024, 1024]))
    depth = float(geometry_config.get("volume_depth_px", 96.0))
    focal_z = float(geometry_config.get("focal_plane_z_px", depth / 2))
    mode = scene_mode(scene_config, sample_index)
    domains = sample_domains(
        scene_config, mode, rng, width, height, depth, focal_z
    )
    fibers: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    clumps: list[dict[str, Any]] = []
    individual_range, bundle_range, clump_range = MODE_COUNTS[mode]
    individual_count = _integer_range(
        rng,
        _mode_value(
            config.get("individual_filaments", {}),
            "count_range",
            mode,
            individual_range,
        ),
    )
    bundle_count = (
        _integer_range(
            rng,
            _mode_value(
                config.get("bundles", {}),
                "count_range",
                mode,
                bundle_range,
            ),
        )
        if config.get("bundles", {}).get("enabled", True)
        else 0
    )
    clump_count = (
        _integer_range(
            rng,
            _mode_value(
                config.get("clumps", {}),
                "count_range",
                mode,
                clump_range,
            ),
        )
        if config.get("clumps", {}).get("enabled", True)
        else 0
    )
    for _ in range(individual_count):
        start, tangent = sample_start_tangent(
            domains, scene_config, mode, rng, width, height, depth, focal_z
        )
        raw = persistent_vertices_from(
            geometry_config,
            config.get("individual_filaments", {}),
            rng,
            start,
            tangent,
            width,
            height,
            depth,
        )
        append_fiber(
            fibers,
            nodes,
            edges,
            raw,
            geometry_config,
            config.get("intensity_variation", {}),
            rng,
            structure_type="individual_filament",
        )
    for bundle_id in range(1, bundle_count + 1):
        bundles.append(
            add_bundle(
                bundle_id,
                fibers,
                nodes,
                edges,
                domains,
                config,
                mode,
                rng,
                width,
                height,
                depth,
                focal_z,
            )
        )
    for clump_id in range(1, clump_count + 1):
        clumps.append(
            add_clump(
                clump_id,
                fibers,
                nodes,
                edges,
                domains,
                config,
                mode,
                rng,
                width,
                height,
                depth,
                focal_z,
            )
        )
    if mode == "mixed_morphology" and clumps:
        add_filament_terminating_in_clump(
            fibers,
            nodes,
            edges,
            clumps[0],
            config,
            rng,
            width,
            height,
            depth,
        )
    return {
        "image_shape": (height, width),
        "volume_depth_px": depth,
        "focal_plane_z_px": focal_z,
        "fibers": fibers,
        "nodes": nodes,
        "edges": edges,
        "bundles": bundles,
        "clumps": clumps,
        "scene_domains": domains,
        "parameters": {
            "generator_mode": "morphology_scene_3d",
            "scenario": mode,
            "scene_morphology_mode": mode,
            "scenario_category": "realism_calibration",
            "condition_blind": True,
            "spatial_units": "pixel_equivalent",
            "image_shape": [height, width],
            "volume_depth_px": depth,
            "focal_plane_z_px": focal_z,
            "individual_filament_count": individual_count,
            "bundle_count": bundle_count,
            "clump_count": clump_count,
            "domain_count": len(domains),
        },
    }


def scene_mode(config: dict[str, Any], sample_index: int) -> str:
    modes = config.get("review_modes", [])
    mode = str(
        modes[sample_index % len(modes)]
        if modes
        else config.get("mode", "mixed_morphology")
    )
    if mode not in SCENE_MODES:
        raise ValueError(f"unsupported scene morphology mode: {mode}")
    return mode


def sample_domains(
    config: dict[str, Any],
    mode: str,
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
) -> list[dict[str, Any]]:
    requested = config.get("cluster_count_range", [1, 4])
    count = 0 if mode == "isolated_filaments" else _integer_range(rng, requested)
    if mode in {"clustered_filament_network", "dense_tangle"}:
        count = max(2, count)
    domains = []
    for domain_id in range(1, count + 1):
        radius = _uniform(rng, config.get("cluster_radius_range_px", [90, 260]))
        anisotropy = _uniform(
            rng, config.get("cluster_anisotropy_range", [0.45, 1.0])
        )
        angle = rng.uniform(0, math.pi)
        alignment = _uniform(
            rng, config.get("alignment_strength_range", [0.0, 0.9])
        )
        if mode == "aligned_filament_domain":
            alignment = max(alignment, 0.8)
        domains.append(
            {
                "domain_id": domain_id,
                "center_xyz": np.asarray(
                    [
                        rng.uniform(0.15 * width, 0.85 * width),
                        rng.uniform(0.15 * height, 0.85 * height),
                        np.clip(rng.normal(focal_z, depth / 5), 0, depth),
                    ],
                    dtype=np.float32,
                ),
                "radius_xy": np.asarray(
                    [radius, radius * anisotropy], dtype=np.float32
                ),
                "angle_rad": float(angle),
                "preferred_orientation_rad": float(rng.uniform(0, math.pi)),
                "alignment_strength": float(alignment),
                "orientation_domain_size_px": _uniform(
                    rng,
                    config.get(
                        "orientation_domain_size_range_px", [100.0, 320.0]
                    ),
                ),
                "density_multiplier": _uniform(
                    rng,
                    config.get("local_density_multiplier_range", [1.2, 3.5]),
                ),
            }
        )
    return domains


def sample_start_tangent(
    domains: list[dict[str, Any]],
    config: dict[str, Any],
    mode: str,
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    clustered_probability = (
        0.15 if mode == "isolated_filaments" else 0.8 if domains else 0.0
    )
    if domains and rng.random() < clustered_probability:
        weights = np.asarray(
            [domain["density_multiplier"] for domain in domains],
            dtype=np.float64,
        )
        weights /= weights.sum()
        domain = domains[int(rng.choice(len(domains), p=weights))]
        local = rng.normal(0, 0.45, 2) * domain["radius_xy"]
        c, s = math.cos(domain["angle_rad"]), math.sin(domain["angle_rad"])
        xy = domain["center_xyz"][:2] + np.asarray(
            [c * local[0] - s * local[1], s * local[0] + c * local[1]]
        )
        distance = float(np.linalg.norm(xy - domain["center_xyz"][:2]))
        local_alignment = domain["alignment_strength"] * math.exp(
            -distance / max(domain["orientation_domain_size_px"], 1.0)
        )
        angle = (
            domain["preferred_orientation_rad"]
            + rng.normal(0, max(0.05, 1 - local_alignment))
        )
        tangent = np.asarray(
            [math.cos(angle), math.sin(angle), rng.normal(0, 0.2)],
            dtype=np.float64,
        )
    else:
        margin_fraction = _uniform(
            rng, config.get("empty_area_fraction_range", [0.05, 0.3])
        )
        margin_x = margin_fraction * width * 0.5
        margin_y = margin_fraction * height * 0.5
        xy = np.asarray(
            [
                rng.uniform(margin_x, width - margin_x),
                rng.uniform(margin_y, height - margin_y),
            ]
        )
        tangent = _random_unit(rng)
    start = np.asarray(
        [
            np.clip(xy[0], 0, width - 1),
            np.clip(xy[1], 0, height - 1),
            np.clip(rng.normal(focal_z, depth / 5), 0, depth),
        ],
        dtype=np.float64,
    )
    tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
    return start, tangent


def persistent_vertices_from(
    geometry_config: dict[str, Any],
    filament_config: dict[str, Any],
    rng: np.random.Generator,
    start: np.ndarray,
    tangent: np.ndarray,
    width: int,
    height: int,
    depth: float,
) -> np.ndarray:
    step = float(geometry_config.get("step_length_px", 6.0))
    length = _uniform(
        rng,
        filament_config.get(
            "contour_length_range_px",
            geometry_config.get("contour_length_range_px", [120, 620]),
        ),
    )
    persistence = _uniform(
        rng,
        filament_config.get(
            "persistence_length_range_px",
            geometry_config.get("persistence_length_range_px", [55, 320]),
        ),
    )
    corr = float(np.exp(-step / max(persistence, 1e-6)))
    vertices = [start.copy()]
    position = start.copy()
    direction = tangent.copy()
    for _ in range(max(2, int(math.ceil(length / step)))):
        direction = _persistent_tangent(direction, corr, rng)
        candidate = position + direction * step
        position, direction, stop = _handle_boundary(
            candidate,
            direction,
            width,
            height,
            depth,
            geometry_config.get("boundary_mode", "reflect"),
        )
        vertices.append(position.copy())
        if stop:
            break
    return np.asarray(vertices, dtype=np.float32)


def append_fiber(
    fibers: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    raw: np.ndarray,
    geometry_config: dict[str, Any],
    intensity_config: dict[str, Any],
    rng: np.random.Generator,
    *,
    structure_type: str,
    parent_bundle_id: int = 0,
    parent_clump_id: int = 0,
    supervised_samples: np.ndarray | None = None,
    start_status: str = "valid_endpoint",
    end_status: str = "valid_endpoint",
) -> int:
    smooth = smooth_catmull_rom(
        raw, int(geometry_config.get("spline_samples_per_segment", 6))
    )
    smooth = np.clip(
        smooth,
        [0, 0, 0],
        [
            geometry_config["image_shape"][1] - 1,
            geometry_config["image_shape"][0] - 1,
            geometry_config.get("volume_depth_px", 96),
        ],
    )
    points = resample_by_arc_length(
        smooth,
        float(geometry_config.get("arc_length_sampling_interval_px", 1.0)),
    )
    fluorophore_config = {
        **geometry_config.get("fluorophore", {}),
        **intensity_config,
    }
    amplitude, radius = fluorophore_samples(
        len(points), fluorophore_config, rng
    )
    if supervised_samples is None:
        supervised_samples = np.ones(len(points), dtype=bool)
    elif len(supervised_samples) != len(points):
        x = np.linspace(0, 1, len(supervised_samples))
        supervised_samples = (
            np.interp(np.linspace(0, 1, len(points)), x, supervised_samples) >= 0.5
        )
    fid = len(fibers) + 1
    node_start = len(nodes)
    nodes.extend(
        [
            {
                "xyz": points[0].astype(np.float32),
                "type": "truncated_endpoint"
                if start_status == "boundary_truncation"
                else "endpoint",
                "supervised_endpoint": start_status == "valid_endpoint",
                "termination_status": start_status,
            },
            {
                "xyz": points[-1].astype(np.float32),
                "type": "truncated_endpoint"
                if end_status == "boundary_truncation"
                else "endpoint",
                "supervised_endpoint": end_status == "valid_endpoint",
                "termination_status": end_status,
            },
        ]
    )
    fibers.append(
        {
            "fiber_id": fid,
            "raw_vertices_xyz": raw.astype(np.float32),
            "points_xyz": points.astype(np.float32),
            "points_xy": points[:, :2].astype(np.float32),
            "sample_amplitude": amplitude,
            "sample_radius": radius,
            "truncated_start": start_status == "boundary_truncation",
            "truncated_end": end_status == "boundary_truncation",
            "structure_type": structure_type,
            "parent_bundle_id": parent_bundle_id,
            "parent_clump_id": parent_clump_id,
            "supervised_centerline_sample": supervised_samples.astype(np.uint8),
            "trace_start_status": start_status,
            "trace_end_status": end_status,
            "crossing_eligible": structure_type == "individual_filament",
        }
    )
    edges.append(
        {
            "fiber_id": fid,
            "node_indices": (node_start, node_start + 1),
            "truncated_start": start_status == "boundary_truncation",
            "truncated_end": end_status == "boundary_truncation",
            "latent": structure_type != "individual_filament",
        }
    )
    return fid


def add_bundle(
    bundle_id: int,
    fibers: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    domains: list[dict[str, Any]],
    config: dict[str, Any],
    mode: str,
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
) -> dict[str, Any]:
    bundle_config = config.get("bundles", {})
    start, tangent = sample_start_tangent(
        domains,
        config.get("scene_morphology", {}),
        mode,
        rng,
        width,
        height,
        depth,
        focal_z,
    )
    axis_raw = persistent_vertices_from(
        config["geometry"],
        {
            "contour_length_range_px": bundle_config.get(
                "bundle_length_range_px", [180, 520]
            ),
            "persistence_length_range_px": bundle_config.get(
                "persistence_length_range_px", [180, 520]
            ),
        },
        rng,
        start,
        tangent,
        width,
        height,
        depth,
    )
    axis = resample_by_arc_length(
        smooth_catmull_rom(
            axis_raw,
            int(config["geometry"].get("spline_samples_per_segment", 6)),
        ),
        float(config["geometry"].get("arc_length_sampling_interval_px", 1.0)),
    )
    child_count = _integer_range(
        rng, bundle_config.get("child_filament_count_range", [3, 7])
    )
    radius_base = _uniform(
        rng, bundle_config.get("bundle_radius_range_px", [4.0, 10.0])
    )
    radius_profile = radius_base * np.clip(
        1
        + 0.2
        * np.sin(
            np.linspace(0, rng.uniform(1, 3) * math.pi, len(axis))
            + rng.uniform(0, 2 * math.pi)
        ),
        0.55,
        1.45,
    )
    partial = rng.random() < float(
        bundle_config.get("partial_resolution_probability", 0.7)
    )
    transition_fraction = (
        rng.uniform(0.12, 0.28) if partial else 0.0
    )
    supervised_template = np.zeros(len(axis), dtype=bool)
    transition_samples = max(1, int(len(axis) * transition_fraction))
    if partial:
        supervised_template[:transition_samples] = True
        supervised_template[-transition_samples:] = True
    twist_rate = _uniform(
        rng, bundle_config.get("twist_rate_range", [-0.03, 0.03])
    )
    converging = rng.random() < float(
        bundle_config.get("convergence_probability", 0.35)
    )
    diverging = rng.random() < float(
        bundle_config.get("divergence_probability", 0.35)
    )
    envelope = np.ones(len(axis), dtype=np.float32)
    if converging:
        envelope *= np.linspace(1.25, 0.55, len(axis), dtype=np.float32)
    if diverging:
        envelope *= np.linspace(0.55, 1.25, len(axis), dtype=np.float32)
    offsets = np.linspace(-0.75, 0.75, child_count) * radius_base
    child_ids = []
    for child_index, base_offset in enumerate(offsets):
        child = offset_bundle_child(
            axis,
            base_offset,
            twist_rate,
            child_index,
            child_count,
            rng,
            bundle_config,
            envelope,
        )
        child_ids.append(
            append_fiber(
                fibers,
                nodes,
                edges,
                child,
                config["geometry"],
                config.get("intensity_variation", {}),
                rng,
                structure_type="bundle_child",
                parent_bundle_id=bundle_id,
                supervised_samples=supervised_template,
                start_status="valid_endpoint"
                if partial
                else "terminates_in_bundle",
                end_status="valid_endpoint"
                if partial
                else "terminates_in_bundle",
            )
        )
    unresolved = np.ones(len(axis), dtype=np.uint8)
    if partial:
        unresolved[:transition_samples] = 0
        unresolved[-transition_samples:] = 0
    return {
        "bundle_id": bundle_id,
        "axis_points_xyz": axis.astype(np.float32),
        "axis_radius_px": radius_profile.astype(np.float32),
        "child_fiber_ids": np.asarray(child_ids, dtype=np.int32),
        "unresolved_sample": unresolved,
        "transition_sample_count": transition_samples if partial else 0,
        "partial_resolution": partial,
        "twist_rate": twist_rate,
        "converging": converging,
        "diverging": diverging,
    }


def offset_bundle_child(
    axis: np.ndarray,
    base_offset: float,
    twist_rate: float,
    child_index: int,
    child_count: int,
    rng: np.random.Generator,
    config: dict[str, Any],
    envelope: np.ndarray,
) -> np.ndarray:
    tangent = np.gradient(axis.astype(np.float64), axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
    normal_xy = np.column_stack([-tangent[:, 1], tangent[:, 0], np.zeros(len(axis))])
    normal_xy /= np.maximum(
        np.linalg.norm(normal_xy, axis=1, keepdims=True), 1e-8
    )
    axial_normal = np.cross(tangent, normal_xy)
    phase = 2 * math.pi * child_index / max(child_count, 1)
    arc = np.arange(len(axis), dtype=np.float64)
    angle = phase + twist_rate * arc
    correlation = float(config.get("child_offset_correlation_length_px", 30))
    controls = max(3, int(len(axis) / max(correlation, 1)) + 2)
    smooth_noise = np.interp(
        np.arange(len(axis)),
        np.linspace(0, len(axis) - 1, controls),
        rng.normal(0, 0.12, controls),
    )
    magnitude = base_offset * (1 + smooth_noise) * envelope
    offset = (
        normal_xy * (magnitude * np.cos(angle))[:, None]
        + axial_normal * (magnitude * np.sin(angle))[:, None]
    )
    return (axis + offset).astype(np.float32)


def add_clump(
    clump_id: int,
    fibers: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    domains: list[dict[str, Any]],
    config: dict[str, Any],
    mode: str,
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
) -> dict[str, Any]:
    clump_config = config.get("clumps", {})
    center, _ = sample_start_tangent(
        domains,
        config.get("scene_morphology", {}),
        mode,
        rng,
        width,
        height,
        depth,
        focal_z,
    )
    radius = _uniform(rng, clump_config.get("radius_range_px", [18, 55]))
    aspect = _uniform(rng, clump_config.get("aspect_ratio_range", [0.55, 1.0]))
    depth_extent = _uniform(
        rng, clump_config.get("depth_extent_range", [8, 28])
    )
    radii = np.asarray([radius, radius * aspect, depth_extent], dtype=np.float32)
    internal_density = _uniform(
        rng, clump_config.get("internal_density_range", [0.5, 1.0])
    )
    irregularity = _uniform(
        rng, clump_config.get("irregularity_range", [0.2, 0.7])
    )
    fragment_count = max(
        4,
        int(
            round(
                _integer_range(
                    rng, clump_config.get("fragment_count_range", [12, 28])
                )
                * internal_density
            )
        ),
    )
    fragment_ids = []
    for _ in range(fragment_count):
        local = _sample_inside_ellipsoid(rng, radii * 0.65)
        start = center + local
        tangent = _random_unit(rng)
        fragment_length = _uniform(
            rng, clump_config.get("fragment_length_range_px", [10, 42])
        )
        fragment_length *= rng.uniform(
            1 - 0.3 * irregularity, 1 + 0.3 * irregularity
        )
        vertices = [start]
        position = start.copy()
        for _ in range(max(2, int(fragment_length / 4))):
            tangent = _persistent_tangent(
                tangent,
                float(np.clip(0.75 - 0.4 * irregularity, 0.3, 0.7)),
                rng,
            )
            position = position + tangent * 4
            relative = position - center
            norm = np.sum((relative / np.maximum(radii, 1e-6)) ** 2)
            if norm > 1:
                position = center + relative / math.sqrt(norm)
                tangent *= -1
            vertices.append(position.copy())
        fragment_ids.append(
            append_fiber(
                fibers,
                nodes,
                edges,
                np.asarray(vertices, dtype=np.float32),
                config["geometry"],
                config.get("intensity_variation", {}),
                rng,
                structure_type="clump_fragment",
                parent_clump_id=clump_id,
                supervised_samples=np.zeros(len(vertices), dtype=bool),
                start_status="terminates_in_clump",
                end_status="terminates_in_clump",
            )
        )
    return {
        "clump_id": clump_id,
        "center_xyz": center.astype(np.float32),
        "radius_xyz": radii,
        "fragment_fiber_ids": np.asarray(fragment_ids, dtype=np.int32),
        "internal_density": internal_density,
        "irregularity": irregularity,
        "edge_diffuseness": _uniform(
            rng, clump_config.get("edge_diffuseness_range", [0.8, 1.5])
        ),
    }


def add_filament_terminating_in_clump(
    fibers: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    clump: dict[str, Any],
    config: dict[str, Any],
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
) -> None:
    center = clump["center_xyz"].astype(np.float64)
    direction = _random_unit(rng)
    direction[2] *= 0.25
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    start = center - direction * min(width, height) * 0.18
    end = center - direction * float(clump["radius_xyz"][0]) * 0.4
    raw = np.linspace(start, end, 14, dtype=np.float32)
    append_fiber(
        fibers,
        nodes,
        edges,
        raw,
        config["geometry"],
        config.get("intensity_variation", {}),
        rng,
        structure_type="individual_filament",
        start_status="valid_endpoint",
        end_status="terminates_in_clump",
    )


def morphology_geometry_to_arrays(
    geometry: dict[str, Any], base_arrays: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    fibers = geometry["fibers"]
    base_arrays.update(
        {
            "fiber_structure_type": np.asarray(
                [
                    FIBER_STRUCTURE_TYPE_CODES[fiber["structure_type"]]
                    for fiber in fibers
                ],
                dtype=np.int16,
            ),
            "fiber_parent_bundle_id": np.asarray(
                [fiber.get("parent_bundle_id", 0) for fiber in fibers],
                dtype=np.int32,
            ),
            "fiber_parent_clump_id": np.asarray(
                [fiber.get("parent_clump_id", 0) for fiber in fibers],
                dtype=np.int32,
            ),
            "fiber_supervised_centerline_sample": np.concatenate(
                [fiber["supervised_centerline_sample"] for fiber in fibers]
            ).astype(np.uint8),
        }
    )
    bundles = geometry["bundles"]
    bundle_axes = [bundle["axis_points_xyz"] for bundle in bundles]
    bundle_radii = [bundle["axis_radius_px"] for bundle in bundles]
    bundle_unresolved = [bundle["unresolved_sample"] for bundle in bundles]
    bundle_axis_offsets = [0]
    child_ids = []
    child_offsets = [0]
    for bundle in bundles:
        bundle_axis_offsets.append(
            bundle_axis_offsets[-1] + len(bundle["axis_points_xyz"])
        )
        child_ids.extend(bundle["child_fiber_ids"].tolist())
        child_offsets.append(child_offsets[-1] + len(bundle["child_fiber_ids"]))
    clumps = geometry["clumps"]
    fragment_ids = []
    fragment_offsets = [0]
    for clump in clumps:
        fragment_ids.extend(clump["fragment_fiber_ids"].tolist())
        fragment_offsets.append(
            fragment_offsets[-1] + len(clump["fragment_fiber_ids"])
        )
    base_arrays.update(
        {
            "bundle_axis_points_xyz": np.vstack(bundle_axes).astype(np.float32)
            if bundle_axes
            else np.zeros((0, 3), dtype=np.float32),
            "bundle_axis_points_xy": np.vstack(bundle_axes)[:, :2].astype(
                np.float32
            )
            if bundle_axes
            else np.zeros((0, 2), dtype=np.float32),
            "bundle_axis_point_offsets": np.asarray(
                bundle_axis_offsets, dtype=np.int32
            ),
            "bundle_axis_radius_px": np.concatenate(bundle_radii).astype(
                np.float32
            )
            if bundle_radii
            else np.zeros(0, dtype=np.float32),
            "bundle_axis_unresolved_sample": np.concatenate(
                bundle_unresolved
            ).astype(np.uint8)
            if bundle_unresolved
            else np.zeros(0, dtype=np.uint8),
            "bundle_ids": np.asarray(
                [bundle["bundle_id"] for bundle in bundles], dtype=np.int32
            ),
            "bundle_partial_resolution": np.asarray(
                [bundle["partial_resolution"] for bundle in bundles],
                dtype=np.uint8,
            ),
            "bundle_twist_rate": np.asarray(
                [bundle["twist_rate"] for bundle in bundles],
                dtype=np.float32,
            ),
            "bundle_converging": np.asarray(
                [bundle["converging"] for bundle in bundles], dtype=np.uint8
            ),
            "bundle_diverging": np.asarray(
                [bundle["diverging"] for bundle in bundles], dtype=np.uint8
            ),
            "bundle_child_fiber_ids": np.asarray(child_ids, dtype=np.int32),
            "bundle_child_fiber_offsets": np.asarray(
                child_offsets, dtype=np.int32
            ),
            "clump_ids": np.asarray(
                [clump["clump_id"] for clump in clumps], dtype=np.int32
            ),
            "clump_center_xyz": np.asarray(
                [clump["center_xyz"] for clump in clumps], dtype=np.float32
            ).reshape((-1, 3)),
            "clump_radius_xyz": np.asarray(
                [clump["radius_xyz"] for clump in clumps], dtype=np.float32
            ).reshape((-1, 3)),
            "clump_internal_density": np.asarray(
                [clump["internal_density"] for clump in clumps],
                dtype=np.float32,
            ),
            "clump_irregularity": np.asarray(
                [clump["irregularity"] for clump in clumps],
                dtype=np.float32,
            ),
            "clump_edge_diffuseness": np.asarray(
                [clump["edge_diffuseness"] for clump in clumps],
                dtype=np.float32,
            ),
            "clump_fragment_fiber_ids": np.asarray(
                fragment_ids, dtype=np.int32
            ),
            "clump_fragment_fiber_offsets": np.asarray(
                fragment_offsets, dtype=np.int32
            ),
        }
    )
    return base_arrays


def _sample_inside_ellipsoid(
    rng: np.random.Generator, radii: np.ndarray
) -> np.ndarray:
    direction = _random_unit(rng)
    return direction * radii * rng.random() ** (1 / 3)


def _integer_range(rng: np.random.Generator, value: Any) -> int:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(rng.integers(int(value[0]), int(value[1]) + 1))
    return int(value)


def _mode_value(
    section: dict[str, Any], key: str, mode: str, default: Any
) -> Any:
    return section.get(f"{key}_by_mode", {}).get(
        mode, section.get(key, default)
    )
