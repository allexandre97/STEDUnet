"""Persistent-chain 3D fiber geometry for synthetic STED rendering."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .schema import NODE_TYPES


def generate_persistent_chain_geometry(config: dict[str, Any], sample_index: int) -> dict[str, Any]:
    """Generate smooth 3D fibers in pixel-equivalent coordinates.

    Tangent correlation follows the worm-like-chain approximation
    ``E[t(s) dot t(s + ds)] = exp(-ds / persistence_length)``.  The discrete
    update mixes the previous tangent with a random perpendicular direction
    using that correlation coefficient, so larger persistence lengths yield
    straighter fibers.
    """

    rng = np.random.default_rng(int(config.get("base_seed", 51001)) + sample_index)
    height, width = _shape(config)
    depth = float(config.get("volume_depth_px", 96.0))
    focal_z = float(config.get("focal_plane_z_px", depth / 2.0))
    scenario = _scenario(config, sample_index)
    fibers: list[dict[str, Any]]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    if scenario:
        fibers, nodes, edges = _scenario_geometry(scenario, config, rng, width, height, depth, focal_z)
    else:
        fibers, nodes, edges = _random_geometry(config, rng, width, height, depth, focal_z)

    return {
        "image_shape": (height, width),
        "volume_depth_px": depth,
        "focal_plane_z_px": focal_z,
        "fibers": fibers,
        "nodes": nodes,
        "edges": edges,
        "parameters": {
            "generator_mode": "persistent_chain_3d",
            "spatial_units": "pixel_equivalent",
            "axis_order": "x_y_z",
            "origin": "image upper-left at x=0, y=0; z=0 at near axial bound",
            "pixel_center": "pixel (x, y) is centered at (x + 0.5, y + 0.5)",
            "image_shape": [height, width],
            "volume_depth_px": depth,
            "focal_plane_z_px": focal_z,
            "fiber_count_range": config.get("fiber_count_range", [4, 8]),
            "contour_length_range_px": config.get("contour_length_range_px", [180.0, 520.0]),
            "step_length_px": float(config.get("step_length_px", 6.0)),
            "persistence_length_range_px": config.get("persistence_length_range_px", [80.0, 220.0]),
            "arc_length_sampling_interval_px": float(config.get("arc_length_sampling_interval_px", 1.0)),
            "boundary_mode": config.get("boundary_mode", "reflect"),
            "branching_enabled": bool(config.get("branching_enabled", False)),
            "bundling_enabled": bool(config.get("bundling_enabled", False)),
            "scenario": scenario or "random_persistent_chain",
        },
    }


def geometry3d_to_arrays(geometry: dict[str, Any]) -> dict[str, np.ndarray]:
    points, raw, offsets, raw_offsets = [], [], [0], [0]
    ids, widths, intensities, amplitudes, radii = [], [], [], [], []
    for fiber in geometry["fibers"]:
        pts = fiber["points_xyz"].astype(np.float32)
        verts = fiber["raw_vertices_xyz"].astype(np.float32)
        points.append(pts)
        raw.append(verts)
        offsets.append(offsets[-1] + len(pts))
        raw_offsets.append(raw_offsets[-1] + len(verts))
        ids.append(fiber["fiber_id"])
        amplitudes.append(fiber["sample_amplitude"].astype(np.float32))
        radii.append(fiber["sample_radius"].astype(np.float32))
        widths.append(float(2.0 * np.mean(fiber["sample_radius"])))
        intensities.append(float(np.mean(fiber["sample_amplitude"])))
    node_xyz = np.asarray([node["xyz"] for node in geometry["nodes"]], dtype=np.float32)
    return {
        "fiber_points_xyz": np.vstack(points).astype(np.float32),
        "fiber_points_xy": np.vstack(points).astype(np.float32)[:, :2],
        "fiber_vertex_xyz": np.vstack(raw).astype(np.float32),
        "fiber_point_offsets": np.asarray(offsets, dtype=np.int32),
        "fiber_vertex_offsets": np.asarray(raw_offsets, dtype=np.int32),
        "fiber_ids": np.asarray(ids, dtype=np.int32),
        "fiber_width": np.asarray(widths, dtype=np.float32),
        "fiber_intensity": np.asarray(intensities, dtype=np.float32),
        "fiber_sample_amplitude": np.concatenate(amplitudes).astype(np.float32),
        "fiber_sample_radius": np.concatenate(radii).astype(np.float32),
        "node_xyz": node_xyz,
        "node_xy": node_xyz[:, :2].astype(np.float32),
        "node_type": np.asarray([NODE_TYPES[node["type"]] for node in geometry["nodes"]], dtype=np.int16),
        "edge_node_indices": np.asarray([edge["node_indices"] for edge in geometry["edges"]], dtype=np.int32),
        "edge_fiber_id": np.asarray([edge["fiber_id"] for edge in geometry["edges"]], dtype=np.int32),
        "edge_truncated_start": np.asarray([edge["truncated_start"] for edge in geometry["edges"]], dtype=np.uint8),
        "edge_truncated_end": np.asarray([edge["truncated_end"] for edge in geometry["edges"]], dtype=np.uint8),
    }


def _shape(config: dict[str, Any]) -> tuple[int, int]:
    shape = tuple(config.get("image_shape", [1024, 1024]))
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("image_shape must contain two positive integers")
    return int(shape[0]), int(shape[1])


def _scenario(config: dict[str, Any], sample_index: int) -> str | None:
    value = config.get("scenario")
    if value:
        return str(value)
    scenarios = config.get("sample_scenarios", [])
    if scenarios:
        return str(scenarios[sample_index % len(scenarios)])
    return None


def _random_geometry(
    config: dict[str, Any],
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    count = int(rng.integers(int(config.get("fiber_count_range", [4, 8])[0]), int(config.get("fiber_count_range", [4, 8])[1]) + 1))
    fibers, nodes, edges = [], [], []
    for _ in range(count):
        raw = persistent_vertices(config, rng, width, height, depth, focal_z)
        _add_fiber_from_vertices(fibers, nodes, edges, raw, config, rng)
    if bool(config.get("branching_enabled", False)):
        _add_true_junction(fibers, nodes, edges, config, rng, width, height, depth, focal_z)
    return fibers, nodes, edges


def persistent_vertices(
    config: dict[str, Any],
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
) -> np.ndarray:
    step = float(config.get("step_length_px", 6.0))
    length = _uniform(rng, config.get("contour_length_range_px", [180.0, 520.0]))
    persistence = _uniform(rng, config.get("persistence_length_range_px", [80.0, 220.0]))
    corr = float(np.exp(-step / max(persistence, 1e-6)))
    n_steps = max(2, int(math.ceil(length / step)))
    pos = np.asarray(
        [
            rng.uniform(0.08 * width, 0.92 * width),
            rng.uniform(0.08 * height, 0.92 * height),
            _sample_z(config, rng, depth, focal_z),
        ],
        dtype=np.float64,
    )
    tangent = _random_unit(rng)
    vertices = [pos.copy()]
    for _ in range(n_steps):
        tangent = _persistent_tangent(tangent, corr, rng)
        candidate = pos + step * tangent
        pos, tangent, stop = _handle_boundary(candidate, tangent, width, height, depth, config.get("boundary_mode", "reflect"))
        vertices.append(pos.copy())
        if stop:
            break
    return np.asarray(vertices, dtype=np.float32)


def _add_fiber_from_vertices(
    fibers: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    raw: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    fid = len(fibers) + 1
    smooth = smooth_catmull_rom(raw, int(config.get("spline_samples_per_segment", 6)))
    smooth = _clip_to_volume(smooth, config)
    points = resample_by_arc_length(smooth, float(config.get("arc_length_sampling_interval_px", 1.0)))
    amp, radius = fluorophore_samples(points.shape[0], config.get("fluorophore", config), rng)
    n0 = len(nodes)
    nodes.append({"xyz": points[0].astype(np.float32), "type": _endpoint_type(points[0], config)})
    nodes.append({"xyz": points[-1].astype(np.float32), "type": _endpoint_type(points[-1], config)})
    fibers.append(
        {
            "fiber_id": fid,
            "raw_vertices_xyz": raw.astype(np.float32),
            "points_xyz": points.astype(np.float32),
            "points_xy": points[:, :2].astype(np.float32),
            "sample_amplitude": amp,
            "sample_radius": radius,
            "truncated_start": nodes[n0]["type"] == "truncated_endpoint",
            "truncated_end": nodes[n0 + 1]["type"] == "truncated_endpoint",
        }
    )
    edges.append({"fiber_id": fid, "node_indices": (n0, n0 + 1), "truncated_start": fibers[-1]["truncated_start"], "truncated_end": fibers[-1]["truncated_end"]})


def _scenario_geometry(
    scenario: str,
    config: dict[str, Any],
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fibers, nodes, edges = [], [], []
    if scenario == "straight_width_calibration":
        angle = math.radians(float(config.get("width_calibration", {}).get("angle_degrees", 0.0)))
        length = float(config.get("width_calibration", {}).get("length_px", min(width, height) * 0.72))
        center = np.asarray([width / 2, height / 2, focal_z], dtype=np.float32)
        raw = straight_vertices(center, angle, length)
        _add_fiber_from_vertices(fibers, nodes, edges, raw, config, rng)
        return fibers, nodes, edges
    if scenario in {"projected_depth_crossing", "in_focus_defocused_crossing", "near_coplanar_crossing"}:
        dz = 4.0 if scenario == "near_coplanar_crossing" else 28.0
        z1 = focal_z
        z2 = focal_z + dz
        centers = [np.asarray([width / 2, height / 2, z1], dtype=np.float32), np.asarray([width / 2, height / 2, z2], dtype=np.float32)]
        angles = [0.0, math.pi / 2.0]
        for center, angle in zip(centers, angles):
            _add_fiber_from_vertices(fibers, nodes, edges, straight_vertices(center, angle, min(width, height) * 0.70), config, rng)
        return fibers, nodes, edges
    if scenario == "true_junction_fixture":
        _add_true_junction(fibers, nodes, edges, config, rng, width, height, depth, focal_z)
        return fibers, nodes, edges
    local = dict(config)
    if scenario == "sparse_near_planar":
        local.update({"fiber_count_range": [3, 4], "depth_distribution_sigma_px": 3.0})
    elif scenario == "strongly_3d":
        local.update({"fiber_count_range": [5, 7], "depth_distribution_sigma_px": depth / 3.0})
    elif scenario == "dense_local_geometry":
        local.update({"fiber_count_range": [10, 12], "contour_length_range_px": [160.0, 360.0]})
    elif scenario == "weak_intermittent":
        local.setdefault("fluorophore", dict(config.get("fluorophore", {})))
        local["fluorophore"].update({"gap_probability": 0.8, "gap_length_range_px": [12.0, 42.0], "variation_amplitude": 0.55})
    return _random_geometry(local, rng, width, height, depth, focal_z)


def _add_true_junction(
    fibers: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    config: dict[str, Any],
    rng: np.random.Generator,
    width: int,
    height: int,
    depth: float,
    focal_z: float,
) -> None:
    center = np.asarray([0.38 * width, 0.38 * height, focal_z], dtype=np.float32)
    junction_index = len(nodes)
    nodes.append({"xyz": center, "type": "true_junction"})
    for angle in (math.radians(210), math.radians(330), math.radians(80)):
        end = center + np.asarray([math.cos(angle) * width * 0.16, math.sin(angle) * height * 0.16, rng.uniform(-4, 4)], dtype=np.float32)
        raw = np.linspace(center, end, 8, dtype=np.float32)
        fid = len(fibers) + 1
        smooth = resample_by_arc_length(raw, float(config.get("arc_length_sampling_interval_px", 1.0)))
        amp, radius = fluorophore_samples(smooth.shape[0], config.get("fluorophore", config), rng)
        end_index = len(nodes)
        nodes.append({"xyz": smooth[-1], "type": "endpoint"})
        fibers.append({"fiber_id": fid, "raw_vertices_xyz": raw, "points_xyz": smooth, "points_xy": smooth[:, :2], "sample_amplitude": amp, "sample_radius": radius, "truncated_start": False, "truncated_end": False})
        edges.append({"fiber_id": fid, "node_indices": (junction_index, end_index), "truncated_start": False, "truncated_end": False})


def straight_vertices(center: np.ndarray, angle: float, length: float) -> np.ndarray:
    direction = np.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=np.float32)
    offsets = np.linspace(-0.5, 0.5, 7, dtype=np.float32)[:, None] * length * direction
    return (center[None, :] + offsets).astype(np.float32)


def smooth_catmull_rom(points: np.ndarray, samples_per_segment: int) -> np.ndarray:
    if len(points) < 3:
        return points.astype(np.float32)
    pts = points.astype(np.float64)
    padded = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        ts = np.linspace(0.0, 1.0, max(2, samples_per_segment), endpoint=False)
        for t in ts:
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    out.append(pts[-1])
    return np.asarray(out, dtype=np.float32)


def resample_by_arc_length(points: np.ndarray, interval: float) -> np.ndarray:
    pts = points.astype(np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    pts = pts[keep]
    if len(pts) < 2:
        raise ValueError("fiber has fewer than two unique points")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    samples = np.arange(0.0, total, max(interval, 1e-3))
    if samples.size == 0 or samples[-1] < total:
        samples = np.append(samples, total)
    out = np.column_stack([np.interp(samples, cumulative, pts[:, axis]) for axis in range(3)])
    return out.astype(np.float32)


def fluorophore_samples(n: int, config: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    base = _uniform(rng, config.get("base_amplitude_range", [80.0, 150.0]))
    variation = _correlated_values(n, float(config.get("variation_scale_samples", 36.0)), rng)
    amp = base * np.clip(1.0 + float(config.get("variation_amplitude", 0.25)) * variation, 0.05, None)
    radius_base = _uniform(rng, config.get("radius_range_px", [0.6, 1.2]))
    radius_var = _correlated_values(n, float(config.get("radius_variation_scale_samples", 64.0)), rng)
    radius = radius_base * np.clip(1.0 + float(config.get("radius_variation_amplitude", 0.15)) * radius_var, 0.3, None)
    _apply_gaps(amp, config, rng)
    _apply_puncta(amp, config, rng)
    return amp.astype(np.float32), radius.astype(np.float32)


def _correlated_values(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    controls = max(3, int(math.ceil(n / max(scale, 1.0))) + 2)
    x = np.linspace(0, n - 1, controls)
    values = rng.normal(0, 1, controls)
    interp = np.interp(np.arange(n), x, values)
    std = float(np.std(interp)) or 1.0
    return interp / std


def _apply_gaps(amp: np.ndarray, config: dict[str, Any], rng: np.random.Generator) -> None:
    if rng.random() >= float(config.get("gap_probability", 0.15)):
        return
    lo, hi = config.get("gap_length_range_px", [8.0, 28.0])
    n_gaps = int(rng.integers(1, int(config.get("max_gaps_per_fiber", 3)) + 1))
    for _ in range(n_gaps):
        length = max(1, int(rng.uniform(float(lo), float(hi))))
        start = int(rng.integers(0, max(1, len(amp) - length)))
        amp[start : start + length] *= float(config.get("gap_residual_fraction", 0.05))


def _apply_puncta(amp: np.ndarray, config: dict[str, Any], rng: np.random.Generator) -> None:
    if rng.random() >= float(config.get("punctate_probability", 0.0)):
        return
    count = int(rng.integers(1, int(config.get("punctate_max_count", 4)) + 1))
    for _ in range(count):
        center = int(rng.integers(0, len(amp)))
        sigma = max(1.0, float(config.get("punctate_sigma_samples", 3.0)))
        gain = float(config.get("punctate_gain", 1.5))
        x = np.arange(len(amp))
        amp += amp.mean() * gain * np.exp(-((x - center) ** 2) / (2 * sigma * sigma))


def _persistent_tangent(tangent: np.ndarray, corr: float, rng: np.random.Generator) -> np.ndarray:
    random = _random_unit(rng)
    perp = random - tangent * float(np.dot(random, tangent))
    norm = float(np.linalg.norm(perp))
    if norm < 1e-8:
        perp = np.asarray([tangent[1], -tangent[0], 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(perp))
    perp /= norm
    out = corr * tangent + math.sqrt(max(0.0, 1.0 - corr * corr)) * perp
    return out / max(float(np.linalg.norm(out)), 1e-8)


def _handle_boundary(pos: np.ndarray, tangent: np.ndarray, width: int, height: int, depth: float, mode: str) -> tuple[np.ndarray, np.ndarray, bool]:
    limits = np.asarray([width - 1.0, height - 1.0, depth], dtype=np.float64)
    if np.all((pos >= 0) & (pos <= limits)):
        return pos, tangent, False
    if mode == "terminate":
        return np.clip(pos, 0, limits), tangent, True
    if mode == "clamp":
        return np.clip(pos, 0, limits), tangent, False
    if mode != "reflect":
        raise ValueError(f"unsupported boundary_mode: {mode}")
    out, tan = pos.copy(), tangent.copy()
    for axis, limit in enumerate(limits):
        if out[axis] < 0:
            out[axis] = -out[axis]
            tan[axis] *= -1
        if out[axis] > limit:
            out[axis] = 2 * limit - out[axis]
            tan[axis] *= -1
    return np.clip(out, 0, limits), tan / max(float(np.linalg.norm(tan)), 1e-8), False


def _clip_to_volume(points: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    height, width = _shape(config)
    depth = float(config.get("volume_depth_px", 96.0))
    return np.clip(points, [0, 0, 0], [width - 1, height - 1, depth]).astype(np.float32)


def _sample_z(config: dict[str, Any], rng: np.random.Generator, depth: float, focal_z: float) -> float:
    if "depth_range_px" in config:
        return float(np.clip(_uniform(rng, config["depth_range_px"]), 0, depth))
    sigma = float(config.get("depth_distribution_sigma_px", depth / 5.0))
    return float(np.clip(rng.normal(focal_z, sigma), 0, depth))


def _endpoint_type(point: np.ndarray, config: dict[str, Any]) -> str:
    height, width = _shape(config)
    depth = float(config.get("volume_depth_px", 96.0))
    near = point[0] <= 0.5 or point[1] <= 0.5 or point[2] <= 0.5 or point[0] >= width - 1.5 or point[1] >= height - 1.5 or point[2] >= depth - 0.5
    return "truncated_endpoint" if near else "endpoint"


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0, 1, 3)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def _uniform(rng: np.random.Generator, value: Any) -> float:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(rng.uniform(float(value[0]), float(value[1])))
    return float(value)
