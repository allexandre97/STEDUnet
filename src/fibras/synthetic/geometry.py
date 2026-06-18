"""Deterministic continuous fiber geometry generation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .schema import NODE_TYPES


def sample_quadratic(start: np.ndarray, control: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return ((1 - t) ** 2) * start + 2 * (1 - t) * t * control + (t**2) * end


def line_points(start: tuple[float, float], end: tuple[float, float], n: int) -> np.ndarray:
    return np.linspace(np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32), n, dtype=np.float32)


def generate_geometry(config: dict[str, Any], sample_index: int) -> dict[str, Any]:
    shape = tuple(config.get("image_shape", [1024, 1024]))
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("image_shape must contain two positive integers")
    height, width = int(shape[0]), int(shape[1])
    count_min, count_max = config.get("fiber_count_range", [6, 10])
    if count_min < 0 or count_max < count_min:
        raise ValueError("invalid fiber_count_range")
    base_seed = int(config.get("base_seed", 1234))
    rng = np.random.default_rng(base_seed + sample_index)
    n_points = int(config.get("points_per_fiber", 96))
    radius = float(config.get("fiber_radius_px", 2.0))
    intensity = float(config.get("fiber_intensity", 120.0))

    fibers: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def add_node(xy: tuple[float, float], node_type: str) -> int:
        nodes.append({"xy": np.asarray(xy, dtype=np.float32), "type": node_type})
        return len(nodes) - 1

    def add_fiber(points: np.ndarray, start_node: int, end_node: int, trunc_start: bool = False, trunc_end: bool = False) -> None:
        fid = len(fibers) + 1
        fibers.append(
            {
                "fiber_id": fid,
                "points_xy": points.astype(np.float32),
                "width": radius * 2.0,
                "intensity": intensity * float(rng.uniform(0.75, 1.25)),
                "truncated_start": bool(trunc_start),
                "truncated_end": bool(trunc_end),
            }
        )
        edges.append(
            {
                "fiber_id": fid,
                "node_indices": (start_node, end_node),
                "truncated_start": bool(trunc_start),
                "truncated_end": bool(trunc_end),
            }
        )

    # Deliberate apparent crossing: two disconnected fibers crossing near image center.
    if config.get("include_crossings", True):
        y = height * 0.48
        x = width * 0.52
        a0 = add_node((width * 0.18, y), "endpoint")
        a1 = add_node((width * 0.84, y + height * 0.03), "endpoint")
        add_fiber(line_points((width * 0.18, y), (width * 0.84, y + height * 0.03), n_points), a0, a1)
        b0 = add_node((x, height * 0.18), "endpoint")
        b1 = add_node((x - width * 0.05, height * 0.84), "endpoint")
        add_fiber(line_points((x, height * 0.18), (x - width * 0.05, height * 0.84), n_points), b0, b1)

    # Deliberate true junction: three connected segments sharing one graph node.
    if config.get("include_true_junctions", True):
        center = (width * 0.32, height * 0.32)
        j = add_node(center, "true_junction")
        for endpoint in [(width * 0.18, height * 0.18), (width * 0.48, height * 0.18), (width * 0.34, height * 0.52)]:
            n = add_node(endpoint, "endpoint")
            control = np.asarray(((center[0] + endpoint[0]) / 2, (center[1] + endpoint[1]) / 2), dtype=np.float32)
            pts = sample_quadratic(np.asarray(center, dtype=np.float32), control, np.asarray(endpoint, dtype=np.float32), n_points)
            add_fiber(pts, j, n)

    if config.get("include_truncated", True):
        y = height * 0.76
        n0 = add_node((0.0, y), "truncated_endpoint")
        n1 = add_node((width * 0.24, y + height * 0.08), "endpoint")
        add_fiber(line_points((0.0, y), (width * 0.24, y + height * 0.08), n_points), n0, n1, trunc_start=True)

    target_count = int(rng.integers(count_min, count_max + 1))
    while len(fibers) < target_count:
        length = float(rng.uniform(width * 0.12, width * 0.45))
        angle = float(rng.uniform(0, 2 * math.pi))
        center = np.asarray([rng.uniform(width * 0.15, width * 0.85), rng.uniform(height * 0.15, height * 0.85)], dtype=np.float32)
        delta = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32) * (length / 2)
        normal = np.asarray([-delta[1], delta[0]], dtype=np.float32)
        normal /= max(float(np.linalg.norm(normal)), 1e-6)
        curve = float(rng.uniform(-0.25, 0.25)) * length
        start = np.clip(center - delta, [0, 0], [width - 1, height - 1])
        end = np.clip(center + delta, [0, 0], [width - 1, height - 1])
        control = np.clip(center + normal * curve, [0, 0], [width - 1, height - 1])
        n0 = add_node((float(start[0]), float(start[1])), "endpoint")
        n1 = add_node((float(end[0]), float(end[1])), "endpoint")
        add_fiber(sample_quadratic(start, control, end, n_points), n0, n1)

    return {
        "image_shape": (height, width),
        "fibers": fibers,
        "nodes": nodes,
        "edges": edges,
        "parameters": {
            "fiber_count_range": [count_min, count_max],
            "fiber_radius_px": radius,
            "fiber_intensity": intensity,
            "points_per_fiber": n_points,
            "include_crossings": bool(config.get("include_crossings", True)),
            "include_true_junctions": bool(config.get("include_true_junctions", True)),
            "include_truncated": bool(config.get("include_truncated", True)),
            "spatial_units": "pixels",
        },
    }


def geometry_to_arrays(geometry: dict[str, Any]) -> dict[str, np.ndarray]:
    fibers = geometry["fibers"]
    points = []
    offsets = [0]
    fiber_ids = []
    widths = []
    intensities = []
    for fiber in fibers:
        pts = fiber["points_xy"].astype(np.float32)
        points.append(pts)
        offsets.append(offsets[-1] + len(pts))
        fiber_ids.append(fiber["fiber_id"])
        widths.append(fiber["width"])
        intensities.append(fiber["intensity"])
    node_xy = np.asarray([node["xy"] for node in geometry["nodes"]], dtype=np.float32)
    node_type = np.asarray([NODE_TYPES[node["type"]] for node in geometry["nodes"]], dtype=np.int16)
    edge_node_indices = np.asarray([edge["node_indices"] for edge in geometry["edges"]], dtype=np.int32)
    return {
        "fiber_points_xy": np.vstack(points).astype(np.float32),
        "fiber_point_offsets": np.asarray(offsets, dtype=np.int32),
        "fiber_ids": np.asarray(fiber_ids, dtype=np.int32),
        "fiber_width": np.asarray(widths, dtype=np.float32),
        "fiber_intensity": np.asarray(intensities, dtype=np.float32),
        "node_xy": node_xy,
        "node_type": node_type,
        "edge_node_indices": edge_node_indices,
        "edge_fiber_id": np.asarray([edge["fiber_id"] for edge in geometry["edges"]], dtype=np.int32),
        "edge_truncated_start": np.asarray([edge["truncated_start"] for edge in geometry["edges"]], dtype=np.uint8),
        "edge_truncated_end": np.asarray([edge["truncated_end"] for edge in geometry["edges"]], dtype=np.uint8),
    }

