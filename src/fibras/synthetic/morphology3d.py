"""Condition-blind heterogeneous 3D morphology scenes."""

from __future__ import annotations

import math
import hashlib
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
    BOUNDARY_CODES,
FIBER_STRUCTURE_TYPE_CODES,
    NODE_TYPES,
    TRACE_TERMINATION_STATUS_CODES,
)

UNCERTAIN_FAMILY_CODES = {
    "faint_fragments": 1,
    "defocused_streaks": 2,
    "low_snr_anisotropic_fragments": 3,
    "merged_boundary_filaments": 4,
    "dense_overlapping_filaments": 5,
    "bundle_clump_transition": 6,
    "clump_halo_texture": 7,
    "short_discontinuous_fragments": 8,
    "weak_directional_texture": 9,
    "ambiguous_thick_bundle_edges": 10,
    "filamentous_fluff": 11,
}

UNCERTAIN_ADJACENCY_CODES = {
    "isolated": 1,
    "fibrous_adjacent": 2,
    "clump_adjacent": 3,
    "bundle_adjacent": 4,
}

UNCERTAIN_TARGET_ROLE_CODE = 255


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


class InvalidGeometryCandidate(ValueError):
    def __init__(self, sample_index: int, mode: str, object_type: str, object_index: int, attempts: int, message: str) -> None:
        super().__init__(
            f"{message}; sample_index={sample_index}, scene_mode={mode}, "
            f"object_type={object_type}, object_index={object_index}, attempts={attempts}"
        )
        self.sample_index = sample_index
        self.mode = mode
        self.object_type = object_type
        self.object_index = object_index
        self.attempts = attempts


def generate_morphology_geometry(
    config: dict[str, Any], sample_index: int
) -> dict[str, Any]:
    geometry_config = config["geometry"]
    scene_config = config.get("scene_morphology", {})
    base_seed = int(geometry_config.get("base_seed", 81001))
    rng = np.random.default_rng(base_seed + sample_index)
    max_attempts = int(geometry_config.get("invalid_geometry_max_attempts", 16))
    diagnostics = invalid_geometry_diagnostics()
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
    for individual_index in range(individual_count):
        for attempt in range(max_attempts):
            attempt_rng = candidate_rng(base_seed, sample_index, "individual_filament", individual_index, attempt)
            start, tangent = sample_start_tangent(
                domains, scene_config, mode, attempt_rng, width, height, depth, focal_z
            )
            raw = persistent_vertices_from(
                geometry_config,
                config.get("individual_filaments", {}),
                attempt_rng,
                start,
                tangent,
                width,
                height,
                depth,
            )
            try:
                append_fiber(
                    fibers,
                    nodes,
                    edges,
                    raw,
                    geometry_config,
                    config.get("intensity_variation", {}),
                    attempt_rng,
                    structure_type="individual_filament",
                )
                diagnostics["resample_attempt_count"] += attempt
                break
            except ValueError as exc:
                if "no valid in-volume segment" not in str(exc):
                    raise
                diagnostics["invalid_geometry_candidate_count"] += 1
                diagnostics["invalid_filament_count"] += 1
        else:
            raise InvalidGeometryCandidate(sample_index, mode, "individual_filament", individual_index, max_attempts, "individual filament has no valid in-volume segment")
    for bundle_id in range(1, bundle_count + 1):
        for attempt in range(max_attempts):
            attempt_rng = candidate_rng(base_seed, sample_index, "bundle", bundle_id, attempt)
            try:
                bundles.append(
                    add_bundle(
                        bundle_id,
                        fibers,
                        nodes,
                        edges,
                        domains,
                        config,
                        mode,
                        attempt_rng,
                        width,
                        height,
                        depth,
                        focal_z,
                        sample_index=sample_index,
                        diagnostics=diagnostics,
                        max_attempts=max_attempts,
                    )
                )
                diagnostics["resample_attempt_count"] += attempt
                break
            except ValueError as exc:
                if "no valid in-volume segment" not in str(exc):
                    raise
                diagnostics["invalid_geometry_candidate_count"] += 1
                diagnostics["invalid_bundle_count"] += 1
        else:
            raise InvalidGeometryCandidate(sample_index, mode, "bundle", bundle_id, max_attempts, "bundle axis has no valid in-volume segment")
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
                sample_index=sample_index,
                diagnostics=diagnostics,
                max_attempts=max_attempts,
                base_seed=base_seed,
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
            sample_index=sample_index,
            mode=mode,
            base_seed=base_seed,
            diagnostics=diagnostics,
            max_attempts=max_attempts,
        )
    uncertain_count, uncertain_family_counts = add_uncertain_ignore_components(
        fibers,
        nodes,
        edges,
        bundles,
        clumps,
        domains,
        config,
        mode,
        width,
        height,
        depth,
        focal_z,
        sample_index,
        base_seed,
        max_attempts,
        diagnostics,
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
            "uncertain_ignore_component_count": uncertain_count,
            "uncertain_ignore_family_counts": uncertain_family_counts,
            "domain_count": len(domains),
            "invalid_geometry_candidate_count": diagnostics["invalid_geometry_candidate_count"],
            "invalid_bundle_child_count": diagnostics["invalid_bundle_child_count"],
            "invalid_bundle_count": diagnostics["invalid_bundle_count"],
            "invalid_filament_count": diagnostics["invalid_filament_count"],
            "invalid_clump_fragment_count": diagnostics["invalid_clump_fragment_count"],
            "resample_attempt_count": diagnostics["resample_attempt_count"],
            "skipped_candidate_count": diagnostics["skipped_candidate_count"],
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


def invalid_geometry_diagnostics() -> dict[str, int]:
    return {
        "invalid_geometry_candidate_count": 0,
        "invalid_bundle_child_count": 0,
        "invalid_bundle_count": 0,
        "invalid_filament_count": 0,
        "invalid_clump_fragment_count": 0,
        "resample_attempt_count": 0,
        "skipped_candidate_count": 0,
    }


def candidate_rng(
    base_seed: int,
    sample_index: int,
    object_type: str,
    object_index: int,
    attempt_index: int,
) -> np.random.Generator:
    payload = f"{base_seed}:{sample_index}:{object_type}:{object_index}:{attempt_index}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)
    return np.random.default_rng(seed)


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
    extra: dict[str, Any] | None = None,
) -> int:
    smooth = smooth_catmull_rom(
        raw, int(geometry_config.get("spline_samples_per_segment", 6))
    )
    limits = np.asarray(
        [
            geometry_config["image_shape"][1] - 1,
            geometry_config["image_shape"][0] - 1,
            geometry_config.get("volume_depth_px", 96),
        ],
        dtype=np.float64,
    )
    original_start_strictly_inside = bool(
        np.all((smooth[0] > 0) & (smooth[0] < limits))
    )
    original_end_strictly_inside = bool(
        np.all((smooth[-1] > 0) & (smooth[-1] < limits))
    )
    smooth, start_boundary, end_boundary = clip_curve_to_volume(
        smooth,
        limits,
    )
    points = resample_by_arc_length(
        smooth,
        float(geometry_config.get("arc_length_sampling_interval_px", 1.0)),
    )
    start_status = corrected_endpoint_status(
        start_status, start_boundary, original_start_strictly_inside
    )
    end_status = corrected_endpoint_status(
        end_status, end_boundary, original_end_strictly_inside
    )
    if start_status != "boundary_truncation":
        start_boundary = ()
    if end_status != "boundary_truncation":
        end_boundary = ()
    fluorophore_config = {
        **geometry_config.get("fluorophore", {}),
        **intensity_config,
    }
    amplitude, radius = fluorophore_samples(
        len(points), fluorophore_config, rng
    )
    radius_range = fluorophore_config.get("radius_range_px", [0.6, 1.2])
    radius_variation = float(
        fluorophore_config.get("radius_variation_amplitude", 0.15)
    )
    radius = np.clip(
        radius,
        float(radius_range[0]) * 0.3,
        float(radius_range[1]) * (1.0 + 3.0 * radius_variation),
    ).astype(np.float32)
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
                "supervised_endpoint": (
                    structure_type == "individual_filament"
                    and start_status == "valid_endpoint"
                ),
                "termination_status": start_status,
                "boundary_code": boundary_code(start_boundary),
            },
            {
                "xyz": points[-1].astype(np.float32),
                "type": "truncated_endpoint"
                if end_status == "boundary_truncation"
                else "endpoint",
                "supervised_endpoint": (
                    structure_type == "individual_filament"
                    and end_status == "valid_endpoint"
                ),
                "termination_status": end_status,
                "boundary_code": boundary_code(end_boundary),
            },
        ]
    )
    record = {
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
            "start_boundary_code": boundary_code(start_boundary),
            "end_boundary_code": boundary_code(end_boundary),
        }
    if extra:
        record.update(extra)
    fibers.append(record)
    edges.append(
        {
            "fiber_id": fid,
            "node_indices": (node_start, node_start + 1),
            "truncated_start": start_status == "boundary_truncation",
            "truncated_end": end_status == "boundary_truncation",
            "latent": structure_type != "individual_filament",
            "supervised": structure_type == "individual_filament",
            "structure_type": structure_type,
        }
    )
    return fid


def add_uncertain_ignore_components(
    fibers: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    clumps: list[dict[str, Any]],
    domains: list[dict[str, Any]],
    config: dict[str, Any],
    mode: str,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
    sample_index: int,
    base_seed: int,
    max_attempts: int,
    diagnostics: dict[str, int],
) -> tuple[int, dict[str, int]]:
    uncertain_config = config.get("uncertain_ignore", {})
    if not uncertain_config.get("enabled", False):
        return 0, {}
    families = uncertain_config.get("families", {})
    total = 0
    counts: dict[str, int] = {}
    for family in sorted(families):
        family_config = families[family]
        if family not in UNCERTAIN_FAMILY_CODES or not family_config.get("enabled", True):
            continue
        count = _integer_range(
            np.random.default_rng(base_seed + sample_index + UNCERTAIN_FAMILY_CODES[family]),
            _mode_value(family_config, "count_range", mode, family_config.get("count_range", [0, 0])),
        )
        counts[family] = count
        if family == "filamentous_fluff":
            total += add_uncertain_fluff_patches(
                count,
                fibers,
                nodes,
                edges,
                bundles,
                clumps,
                domains,
                family_config,
                config,
                mode,
                width,
                height,
                depth,
                focal_z,
                sample_index,
                base_seed,
                diagnostics,
            )
            continue
        for index in range(count):
            for attempt in range(max_attempts):
                rng = candidate_rng(base_seed, sample_index, f"uncertain_{family}", index, attempt)
                raw, adjacency, source_id = uncertain_fragment_vertices(
                    family,
                    family_config,
                    domains,
                    bundles,
                    clumps,
                    config,
                    mode,
                    rng,
                    width,
                    height,
                    depth,
                    focal_z,
                )
                try:
                    length = polyline_length(raw)
                    intensity = uncertain_intensity_config(family_config, config.get("intensity_variation", {}), rng)
                    append_fiber(
                        fibers,
                        nodes,
                        edges,
                        raw,
                        config["geometry"],
                        intensity,
                        rng,
                        structure_type="uncertain_fragment",
                        supervised_samples=np.zeros(len(raw), dtype=bool),
                        start_status="ambiguous_termination",
                        end_status="ambiguous_termination",
                        extra={
                            "uncertain_family": family,
                            "uncertain_family_code": UNCERTAIN_FAMILY_CODES[family],
                            "uncertain_source_object_id": source_id,
                            "uncertain_target_role_code": UNCERTAIN_TARGET_ROLE_CODE,
                            "uncertain_intensity_multiplier": float(intensity["_uncertain_intensity_multiplier"]),
                            "uncertain_radius_multiplier": float(intensity["_uncertain_radius_multiplier"]),
                            "uncertain_blur_sigma_px": float(family_config.get("blur_sigma_px", 0.0)),
                            "uncertain_support_radius_multiplier": float(family_config.get("support_radius_multiplier", 1.0)),
                            "uncertain_adjacency": adjacency,
                            "uncertain_adjacency_code": UNCERTAIN_ADJACENCY_CODES[adjacency],
                            "uncertain_fragment_length_px": float(length),
                        },
                    )
                    diagnostics["resample_attempt_count"] += attempt
                    total += 1
                    break
                except ValueError as exc:
                    if "no valid in-volume segment" not in str(exc):
                        raise
                    diagnostics["invalid_geometry_candidate_count"] += 1
                    diagnostics["invalid_filament_count"] += 1
            else:
                diagnostics["skipped_candidate_count"] += 1
    return total, counts


def add_uncertain_fluff_patches(
    patch_count: int,
    fibers: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    clumps: list[dict[str, Any]],
    domains: list[dict[str, Any]],
    family_config: dict[str, Any],
    config: dict[str, Any],
    mode: str,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
    sample_index: int,
    base_seed: int,
    diagnostics: dict[str, int],
) -> int:
    total = 0
    for patch_index in range(patch_count):
        patch_rng = candidate_rng(base_seed, sample_index, "uncertain_filamentous_fluff_patch", patch_index, 0)
        adjacency = str(family_config.get("adjacency", "mixed"))
        if adjacency == "mixed":
            adjacency = str(patch_rng.choice(["isolated", "fibrous_adjacent", "clump_adjacent", "bundle_adjacent"]))
        center, tangent, source_id = uncertain_anchor(
            adjacency, domains, bundles, clumps, config, mode, patch_rng, width, height, depth, focal_z
        )
        radius = np.asarray(
            [
                _uniform(patch_rng, family_config.get("radius_x_range_px", [42.0, 96.0])),
                _uniform(patch_rng, family_config.get("radius_y_range_px", [24.0, 72.0])),
            ],
            dtype=np.float64,
        )
        angle = float(math.atan2(float(tangent[1]), float(tangent[0]))) if np.linalg.norm(tangent[:2]) > 1e-8 else float(patch_rng.uniform(0, math.pi))
        fragment_count = _integer_range(patch_rng, family_config.get("fragment_count_range", [20, 60]))
        alignment = float(family_config.get("alignment_strength", 0.35))
        for fragment_index in range(fragment_count):
            rng = candidate_rng(base_seed, sample_index, f"uncertain_filamentous_fluff_{patch_index}", fragment_index, 0)
            local = sample_fluff_local(rng, radius, angle)
            start = center.astype(np.float64) + np.asarray([local[0], local[1], rng.normal(0, float(family_config.get("z_jitter_px", 5.0)))])
            start = np.clip(start, [0, 0, 0], [width - 1, height - 1, depth])
            theta = angle + rng.normal(0, float(family_config.get("orientation_jitter_rad", 0.9)))
            aligned = np.asarray([math.cos(theta), math.sin(theta), rng.normal(0, 0.08)], dtype=np.float64)
            direction = (1 - alignment) * _random_unit(rng) + alignment * aligned
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            raw, length = fluff_fragment_vertices(start, direction, family_config, config, rng, width, height, depth)
            try:
                intensity = uncertain_intensity_config(family_config, config.get("intensity_variation", {}), rng)
                append_fiber(
                    fibers,
                    nodes,
                    edges,
                    raw,
                    config["geometry"],
                    intensity,
                    rng,
                    structure_type="uncertain_fragment",
                    supervised_samples=np.zeros(len(raw), dtype=bool),
                    start_status="ambiguous_termination",
                    end_status="ambiguous_termination",
                    extra={
                        "uncertain_family": "filamentous_fluff",
                        "uncertain_family_code": UNCERTAIN_FAMILY_CODES["filamentous_fluff"],
                        "uncertain_source_object_id": patch_index + 1,
                        "uncertain_target_role_code": UNCERTAIN_TARGET_ROLE_CODE,
                        "uncertain_intensity_multiplier": float(intensity["_uncertain_intensity_multiplier"]),
                        "uncertain_radius_multiplier": float(intensity["_uncertain_radius_multiplier"]),
                        "uncertain_blur_sigma_px": float(family_config.get("blur_sigma_px", 0.0)),
                        "uncertain_support_radius_multiplier": float(family_config.get("support_radius_multiplier", 3.0)),
                        "uncertain_adjacency": adjacency,
                        "uncertain_adjacency_code": UNCERTAIN_ADJACENCY_CODES.get(adjacency, UNCERTAIN_ADJACENCY_CODES["isolated"]),
                        "uncertain_fragment_length_px": float(length),
                    },
                )
                total += 1
            except ValueError as exc:
                if "no valid in-volume segment" not in str(exc):
                    raise
                diagnostics["invalid_geometry_candidate_count"] += 1
                diagnostics["invalid_filament_count"] += 1
    return total


def sample_fluff_local(rng: np.random.Generator, radius: np.ndarray, angle: float) -> np.ndarray:
    r = math.sqrt(float(rng.random()))
    theta = float(rng.uniform(0, 2 * math.pi))
    local = r * np.asarray([math.cos(theta) * radius[0], math.sin(theta) * radius[1]])
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([c * local[0] - s * local[1], s * local[0] + c * local[1]])


def fluff_fragment_vertices(
    start: np.ndarray,
    direction: np.ndarray,
    family_config: dict[str, Any],
    config: dict[str, Any],
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
) -> tuple[np.ndarray, float]:
    length = _uniform(rng, family_config.get("length_range_px", [8.0, 36.0]))
    step = float(family_config.get("step_length_px", 4.0))
    persistence = _uniform(rng, family_config.get("persistence_length_range_px", [10.0, 55.0]))
    corr = float(np.exp(-step / max(persistence, 1e-6)))
    points = [start.astype(np.float64)]
    for _ in range(max(2, int(math.ceil(length / step)))):
        direction = _persistent_tangent(direction, corr, rng)
        candidate = points[-1] + direction * step
        candidate, direction, stop = _handle_boundary(
            candidate,
            direction,
            width,
            height,
            depth,
            config["geometry"].get("boundary_mode", "reflect"),
        )
        points.append(candidate.copy())
        if stop:
            break
    raw = np.asarray(points, dtype=np.float32)
    return raw, polyline_length(raw)


def uncertain_fragment_vertices(
    family: str,
    family_config: dict[str, Any],
    domains: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    clumps: list[dict[str, Any]],
    config: dict[str, Any],
    mode: str,
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
) -> tuple[np.ndarray, str, int]:
    adjacency = str(family_config.get("adjacency", default_uncertain_adjacency(family)))
    anchor, tangent, source_id = uncertain_anchor(
        adjacency, domains, bundles, clumps, config, mode, rng, width, height, depth, focal_z
    )
    if family in {"defocused_streaks", "ambiguous_thick_bundle_edges"}:
        anchor[2] = np.clip(focal_z + rng.choice([-1, 1]) * _uniform(rng, family_config.get("defocus_offset_range_px", [12, 28])), 0, depth)
    length = _uniform(rng, family_config.get("length_range_px", [18, 80]))
    step = float(family_config.get("step_length_px", config["geometry"].get("step_length_px", 6.0)))
    persistence = _uniform(rng, family_config.get("persistence_length_range_px", [18, 120]))
    corr = float(np.exp(-step / max(persistence, 1e-6)))
    points = [anchor.astype(np.float64)]
    direction = tangent.astype(np.float64) / max(float(np.linalg.norm(tangent)), 1e-8)
    incoherence = float(family_config.get("direction_jitter", 0.25))
    for i in range(max(2, int(math.ceil(length / step)))):
        if family in {"low_snr_anisotropic_fragments", "weak_directional_texture"} and i % 3 == 0:
            direction = (1 - incoherence) * direction + incoherence * _random_unit(rng)
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
        else:
            direction = _persistent_tangent(direction, corr, rng)
        candidate = points[-1] + direction * step
        candidate, direction, stop = _handle_boundary(
            candidate,
            direction,
            width,
            height,
            depth,
            config["geometry"].get("boundary_mode", "reflect"),
        )
        points.append(candidate.copy())
        if stop:
            break
    return np.asarray(points, dtype=np.float32), adjacency, source_id


def default_uncertain_adjacency(family: str) -> str:
    if family in {"merged_boundary_filaments", "bundle_clump_transition", "clump_halo_texture"}:
        return "clump_adjacent"
    if family == "ambiguous_thick_bundle_edges":
        return "bundle_adjacent"
    if family in {"defocused_streaks", "dense_overlapping_filaments"}:
        return "fibrous_adjacent"
    return "isolated"


def uncertain_anchor(
    adjacency: str,
    domains: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    clumps: list[dict[str, Any]],
    config: dict[str, Any],
    mode: str,
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if adjacency == "clump_adjacent" and clumps:
        clump = clumps[int(rng.integers(0, len(clumps)))]
        angle = rng.uniform(0, 2 * math.pi)
        radius = clump["radius_xyz"].astype(np.float64)
        normal = np.asarray([math.cos(angle), math.sin(angle), rng.normal(0, 0.08)])
        normal /= max(float(np.linalg.norm(normal)), 1e-8)
        center = clump["center_xyz"].astype(np.float64)
        anchor = center + normal * radius * rng.uniform(0.85, 1.35)
        tangent = np.asarray([-normal[1], normal[0], rng.normal(0, 0.08)])
        return anchor, tangent, int(clump["clump_id"])
    if adjacency == "bundle_adjacent" and bundles:
        bundle = bundles[int(rng.integers(0, len(bundles)))]
        axis = bundle["axis_points_xyz"]
        i = int(rng.integers(0, len(axis)))
        tangent = axis[min(i + 1, len(axis) - 1)] - axis[max(i - 1, 0)]
        normal = np.asarray([-tangent[1], tangent[0], 0.0], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-8)
        anchor = axis[i].astype(np.float64) + normal * float(bundle["axis_radius_px"][i]) * rng.uniform(0.8, 1.4)
        return anchor, tangent, int(bundle["bundle_id"])
    start, tangent = sample_start_tangent(domains, config.get("scene_morphology", {}), mode, rng, width, height, depth, focal_z)
    return start, tangent, 0


def uncertain_intensity_config(
    family_config: dict[str, Any],
    base_config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    out = dict(base_config)
    amp = _uniform(rng, family_config.get("intensity_multiplier_range", [0.15, 0.75]))
    radius = _uniform(rng, family_config.get("radius_multiplier_range", [0.8, 2.4]))
    out["_uncertain_intensity_multiplier"] = amp
    out["_uncertain_radius_multiplier"] = radius
    if "base_amplitude_range" in out:
        out["base_amplitude_range"] = [float(v) * amp for v in out["base_amplitude_range"]]
    if "radius_range_px" in out:
        out["radius_range_px"] = [float(v) * radius for v in out["radius_range_px"]]
    for key in [
        "gap_probability",
        "gap_length_range_px",
        "gap_residual_fraction",
        "max_gaps_per_fiber",
        "punctate_probability",
        "variation_amplitude",
        "radius_variation_amplitude",
    ]:
        if key in family_config:
            out[key] = family_config[key]
    return out


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points.astype(np.float64), axis=0), axis=1).sum())


def corrected_endpoint_status(
    status: str,
    boundary: tuple[str, ...],
    intended_endpoint_inside: bool,
) -> str:
    if not boundary:
        return status
    if (
        intended_endpoint_inside
        and status in {"terminates_in_bundle", "terminates_in_clump"}
    ):
        return status
    return "boundary_truncation"


def boundary_code(names: tuple[str, ...]) -> int:
    return int(sum(BOUNDARY_CODES[name] for name in names))


def boundary_names(
    point: np.ndarray, limits: np.ndarray, tolerance: float = 1e-5
) -> tuple[str, ...]:
    names = []
    for axis, (name, high) in enumerate(zip(("x", "y", "z"), limits)):
        if abs(float(point[axis])) <= tolerance:
            names.append(f"{name}_min")
        if abs(float(point[axis]) - float(high)) <= tolerance:
            names.append(f"{name}_max")
    return tuple(names)


def clip_curve_to_volume(
    points: np.ndarray, limits: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Keep the first contiguous in-volume polyline section.

    Boundary intersections are inserted analytically. The curve is never
    flattened onto an edge and terminates at its first exit.
    """

    points = np.asarray(points, dtype=np.float64)
    output: list[np.ndarray] = []
    start_clipped = False
    end_clipped = False
    for p0, p1 in zip(points[:-1], points[1:]):
        interval = segment_box_interval(p0, p1, limits)
        if interval is None:
            if output:
                break
            continue
        enter, exit_ = interval
        direction = p1 - p0
        q0 = p0 + enter * direction
        q1 = p0 + exit_ * direction
        if not output:
            output.append(q0)
            start_clipped = enter > 1e-9 or np.any(
                (p0 < 0) | (p0 > limits)
            )
        if np.linalg.norm(q1 - output[-1]) > 1e-8:
            output.append(q1)
        if exit_ < 1 - 1e-9 or np.any((p1 < 0) | (p1 > limits)):
            end_clipped = True
            break
    if len(output) < 2:
        raise ValueError("fiber has no valid in-volume segment")
    clipped = np.asarray(output, dtype=np.float32)
    start_boundary = boundary_names(clipped[0], limits)
    end_boundary = boundary_names(clipped[-1], limits)
    if not start_clipped and not start_boundary:
        start_boundary = ()
    if not end_clipped and not end_boundary:
        end_boundary = ()
    return clipped, start_boundary, end_boundary


def segment_box_interval(
    p0: np.ndarray, p1: np.ndarray, limits: np.ndarray
) -> tuple[float, float] | None:
    enter, exit_ = 0.0, 1.0
    direction = p1 - p0
    for axis, limit in enumerate(limits):
        if abs(float(direction[axis])) <= 1e-12:
            if p0[axis] < 0 or p0[axis] > limit:
                return None
            continue
        t0 = (0.0 - p0[axis]) / direction[axis]
        t1 = (limit - p0[axis]) / direction[axis]
        lo, hi = sorted((float(t0), float(t1)))
        enter = max(enter, lo)
        exit_ = min(exit_, hi)
        if enter > exit_:
            return None
    return max(0.0, enter), min(1.0, exit_)


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
    *,
    sample_index: int = 0,
    diagnostics: dict[str, int] | None = None,
    max_attempts: int = 16,
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
    axis_smooth = smooth_catmull_rom(
        axis_raw,
        int(config["geometry"].get("spline_samples_per_segment", 6)),
    )
    axis_smooth, _, _ = clip_curve_to_volume(
        axis_smooth,
        np.asarray([width - 1, height - 1, depth], dtype=np.float64),
    )
    axis = resample_by_arc_length(
        axis_smooth,
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
        for attempt in range(max_attempts):
            child_rng = candidate_rng(
                int(config["geometry"].get("base_seed", 81001)),
                sample_index,
                f"bundle_child_{bundle_id}",
                child_index,
                attempt,
            )
            child = offset_bundle_child(
                axis,
                base_offset,
                twist_rate,
                child_index,
                child_count,
                child_rng,
                bundle_config,
                envelope,
            )
            try:
                child_ids.append(
                    append_fiber(
                        fibers,
                        nodes,
                        edges,
                        child,
                        config["geometry"],
                        config.get("intensity_variation", {}),
                        child_rng,
                        structure_type="bundle_child",
                        parent_bundle_id=bundle_id,
                        supervised_samples=supervised_template,
                        start_status="ambiguous_termination"
                        if partial
                        else "terminates_in_bundle",
                        end_status="ambiguous_termination"
                        if partial
                        else "terminates_in_bundle",
                    )
                )
                if diagnostics is not None:
                    diagnostics["resample_attempt_count"] += attempt
                break
            except ValueError as exc:
                if "no valid in-volume segment" not in str(exc):
                    raise
                if diagnostics is not None:
                    diagnostics["invalid_geometry_candidate_count"] += 1
                    diagnostics["invalid_bundle_child_count"] += 1
        else:
            if diagnostics is not None:
                diagnostics["skipped_candidate_count"] += 1
    if not child_ids:
        raise InvalidGeometryCandidate(
            sample_index,
            mode,
            "bundle_child",
            bundle_id,
            max_attempts,
            "bundle has no valid child fibers",
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
    *,
    sample_index: int = 0,
    diagnostics: dict[str, int] | None = None,
    max_attempts: int = 16,
    base_seed: int = 81001,
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
    _, radius_range = sample_clump_size_class(clump_config, rng)
    radius = _uniform(rng, radius_range)
    aspect = _uniform(rng, clump_config.get("aspect_ratio_range", [0.55, 1.0]))
    depth_extent = _uniform(
        rng, clump_config.get("depth_extent_range", [8, 28])
    )
    radii = np.asarray([radius, radius * aspect, depth_extent], dtype=np.float32)
    limits = np.asarray([width - 1, height - 1, depth], dtype=np.float32)
    margin = np.minimum(radii, limits * 0.45)
    center = np.clip(center, margin, limits - margin)
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
    if "fragment_count_per_radius_px" in clump_config:
        per_radius = _uniform(rng, clump_config["fragment_count_per_radius_px"])
        fragment_count += int(round(radius * per_radius))
    fragment_ids = []
    fragment_intensity = clump_fragment_intensity_config(
        config.get("intensity_variation", {}),
        clump_config,
        rng,
    )
    for fragment_index in range(fragment_count):
        for attempt in range(max_attempts):
            fragment_rng = candidate_rng(
                base_seed,
                sample_index,
                f"clump_fragment_{clump_id}",
                fragment_index,
                attempt,
            )
            local = _sample_inside_ellipsoid(fragment_rng, radii * 0.72)
            start = center + local
            tangent = clump_fragment_tangent(
                fragment_rng, clump_config, clump_id, fragment_index
            )
            fragment_length = _uniform(
                fragment_rng, clump_config.get("fragment_length_range_px", [10, 42])
            )
            fragment_length *= fragment_rng.uniform(
                1 - 0.3 * irregularity, 1 + 0.3 * irregularity
            )
            vertices = [start]
            position = start.copy()
            for _ in range(max(2, int(fragment_length / 4))):
                tangent = _persistent_tangent(
                    tangent,
                    float(np.clip(0.75 - 0.4 * irregularity, 0.3, 0.7)),
                    fragment_rng,
                )
                position = position + tangent * 4
                relative = position - center
                norm = np.sum((relative / np.maximum(radii, 1e-6)) ** 2)
                if norm > 1:
                    position = center + relative / math.sqrt(norm)
                    tangent *= -1
                vertices.append(position.copy())
            try:
                fragment_ids.append(
                    append_fiber(
                        fibers,
                        nodes,
                        edges,
                        np.asarray(vertices, dtype=np.float32),
                        config["geometry"],
                        fragment_intensity,
                        fragment_rng,
                        structure_type="clump_fragment",
                        parent_clump_id=clump_id,
                        supervised_samples=np.zeros(len(vertices), dtype=bool),
                        start_status="terminates_in_clump",
                        end_status="terminates_in_clump",
                    )
                )
                if diagnostics is not None:
                    diagnostics["resample_attempt_count"] += attempt
                break
            except ValueError as exc:
                if "no valid in-volume segment" not in str(exc):
                    raise
                if diagnostics is not None:
                    diagnostics["invalid_geometry_candidate_count"] += 1
                    diagnostics["invalid_clump_fragment_count"] += 1
        else:
            if diagnostics is not None:
                diagnostics["skipped_candidate_count"] += 1
    if not fragment_ids:
        raise InvalidGeometryCandidate(
            sample_index,
            mode,
            "clump_fragment",
            clump_id,
            max_attempts,
            "clump has no valid fragments",
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


def sample_clump_size_class(
    config: dict[str, Any], rng: np.random.Generator
) -> tuple[str, list[float]]:
    mixture = config.get("size_mixture")
    if not mixture:
        return "default", config.get("radius_range_px", [18, 55])
    names = sorted(mixture)
    weights = np.asarray(
        [float(mixture[name].get("weight", 1.0)) for name in names],
        dtype=np.float64,
    )
    weights /= weights.sum()
    name = names[int(rng.choice(len(names), p=weights))]
    return name, mixture[name].get("radius_range_px", config.get("radius_range_px", [18, 55]))


def clump_fragment_tangent(
    rng: np.random.Generator,
    config: dict[str, Any],
    clump_id: int,
    fragment_index: int,
) -> np.ndarray:
    tangent = _random_unit(rng)
    align_probability = float(config.get("fragment_alignment_probability", 0.0))
    if rng.random() >= align_probability:
        return tangent
    bundle_count = max(1, int(config.get("fragment_alignment_bundle_count", 3)))
    angle = (
        2 * math.pi * ((clump_id + fragment_index) % bundle_count) / bundle_count
        + rng.normal(0, float(config.get("fragment_alignment_jitter_rad", 0.25)))
    )
    aligned = np.asarray(
        [math.cos(angle), math.sin(angle), rng.normal(0, 0.08)], dtype=np.float64
    )
    strength = float(config.get("fragment_alignment_strength", 0.75))
    tangent = (1 - strength) * tangent + strength * aligned
    return tangent / max(float(np.linalg.norm(tangent)), 1e-8)


def clump_fragment_intensity_config(
    base_config: dict[str, Any],
    clump_config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    config = dict(base_config)
    amplitude_multiplier = _uniform(
        rng, clump_config.get("fragment_amplitude_multiplier_range", [1.0, 1.0])
    )
    radius_multiplier = _uniform(
        rng, clump_config.get("fragment_radius_multiplier_range", [1.0, 1.0])
    )
    if "base_amplitude_range" in config:
        config["base_amplitude_range"] = [
            float(v) * amplitude_multiplier for v in config["base_amplitude_range"]
        ]
    if "radius_range_px" in config:
        config["radius_range_px"] = [
            float(v) * radius_multiplier for v in config["radius_range_px"]
        ]
    for key in [
        "gap_probability",
        "punctate_probability",
        "variation_amplitude",
        "radius_variation_amplitude",
    ]:
        override = f"fragment_{key}"
        if override in clump_config:
            config[key] = clump_config[override]
    return config


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
    *,
    sample_index: int = 0,
    mode: str = "mixed_morphology",
    base_seed: int = 81001,
    diagnostics: dict[str, int] | None = None,
    max_attempts: int = 16,
) -> None:
    center = clump["center_xyz"].astype(np.float64)
    for attempt in range(max_attempts):
        attempt_rng = candidate_rng(base_seed, sample_index, "filament_terminating_in_clump", int(clump["clump_id"]), attempt)
        direction = _random_unit(attempt_rng)
        direction[2] *= 0.25
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        start = center - direction * min(width, height) * 0.18
        end = center - direction * float(clump["radius_xyz"][0]) * 0.4
        raw = np.linspace(start, end, 14, dtype=np.float32)
        try:
            append_fiber(
                fibers,
                nodes,
                edges,
                raw,
                config["geometry"],
                config.get("intensity_variation", {}),
                attempt_rng,
                structure_type="individual_filament",
                start_status="valid_endpoint",
                end_status="terminates_in_clump",
            )
            if diagnostics is not None:
                diagnostics["resample_attempt_count"] += attempt
            return
        except ValueError as exc:
            if "no valid in-volume segment" not in str(exc):
                raise
            if diagnostics is not None:
                diagnostics["invalid_geometry_candidate_count"] += 1
                diagnostics["invalid_filament_count"] += 1
    raise InvalidGeometryCandidate(
        sample_index,
        mode,
        "filament_terminating_in_clump",
        int(clump["clump_id"]),
        max_attempts,
        "filament terminating in clump has no valid in-volume segment",
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
            "fiber_start_boundary_code": np.asarray(
                [fiber.get("start_boundary_code", 0) for fiber in fibers],
                dtype=np.uint8,
            ),
            "fiber_end_boundary_code": np.asarray(
                [fiber.get("end_boundary_code", 0) for fiber in fibers],
                dtype=np.uint8,
            ),
            "node_supervised": np.asarray(
                [node.get("supervised_endpoint", False) for node in geometry["nodes"]],
                dtype=np.uint8,
            ),
            "node_termination_status": np.asarray(
                [
                    TRACE_TERMINATION_STATUS_CODES.get(
                        node.get("termination_status", "ambiguous_termination"),
                        TRACE_TERMINATION_STATUS_CODES["ambiguous_termination"],
                    )
                    for node in geometry["nodes"]
                ],
                dtype=np.int16,
            ),
            "node_boundary_code": np.asarray(
                [node.get("boundary_code", 0) for node in geometry["nodes"]],
                dtype=np.uint8,
            ),
            "edge_supervised": np.asarray(
                [edge.get("supervised", False) for edge in geometry["edges"]],
                dtype=np.uint8,
            ),
            "edge_structure_type": np.asarray(
                [
                    FIBER_STRUCTURE_TYPE_CODES[
                        edge.get("structure_type", "individual_filament")
                    ]
                    for edge in geometry["edges"]
                ],
                dtype=np.int16,
            ),
            "uncertain_family_code": np.asarray(
                [fiber.get("uncertain_family_code", 0) for fiber in fibers],
                dtype=np.uint8,
            ),
            "uncertain_source_object_id": np.asarray(
                [fiber.get("uncertain_source_object_id", 0) for fiber in fibers],
                dtype=np.int32,
            ),
            "uncertain_target_role_code": np.asarray(
                [fiber.get("uncertain_target_role_code", 0) for fiber in fibers],
                dtype=np.uint8,
            ),
            "uncertain_intensity_multiplier": np.asarray(
                [fiber.get("uncertain_intensity_multiplier", 0.0) for fiber in fibers],
                dtype=np.float32,
            ),
            "uncertain_radius_multiplier": np.asarray(
                [fiber.get("uncertain_radius_multiplier", 0.0) for fiber in fibers],
                dtype=np.float32,
            ),
            "uncertain_blur_sigma_px": np.asarray(
                [fiber.get("uncertain_blur_sigma_px", 0.0) for fiber in fibers],
                dtype=np.float32,
            ),
            "uncertain_adjacency_code": np.asarray(
                [fiber.get("uncertain_adjacency_code", 0) for fiber in fibers],
                dtype=np.uint8,
            ),
            "uncertain_support_radius_multiplier": np.asarray(
                [fiber.get("uncertain_support_radius_multiplier", 0.0) for fiber in fibers],
                dtype=np.float32,
            ),
            "uncertain_fragment_length_px": np.asarray(
                [fiber.get("uncertain_fragment_length_px", 0.0) for fiber in fibers],
                dtype=np.float32,
            ),
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
