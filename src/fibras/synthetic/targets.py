"""Raster target generation for synthetic STED fibers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .schema import NODE_TYPES


def rasterize_targets(geometry: dict[str, Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    height, width = geometry["image_shape"]
    centerline_radius = float(config.get("centerline_radius_px", 0.75))
    endpoint_radius = float(config.get("endpoint_radius_px", 3.0))
    junction_radius = float(config.get("junction_radius_px", 3.0))
    crossing_radius = float(config.get("crossing_radius_px", 3.0))

    source = np.zeros((height, width), dtype=np.float32)
    overlap = np.zeros((height, width), dtype=np.uint16)
    centerline = np.zeros((height, width), dtype=np.uint8)
    membership_y: list[np.ndarray] = []
    membership_x: list[np.ndarray] = []
    membership_i: list[np.ndarray] = []

    for fiber in geometry["fibers"]:
        fid = int(fiber["fiber_id"])
        radius = float(fiber["width"]) / 2.0
        mask = rasterize_polyline(fiber["points_xy"], (height, width), radius)
        line = rasterize_polyline(fiber["points_xy"], (height, width), centerline_radius)
        source[mask] += float(fiber["intensity"])
        overlap[mask] += 1
        centerline[line] = 1
        y, x = np.nonzero(mask)
        membership_y.append(y.astype(np.int32))
        membership_x.append(x.astype(np.int32))
        membership_i.append(np.full(y.shape, fid, dtype=np.int32))

    endpoint_map = np.zeros((height, width), dtype=np.uint8)
    junction_map = np.zeros((height, width), dtype=np.uint8)
    for node in geometry["nodes"]:
        code = NODE_TYPES[node["type"]]
        if code == NODE_TYPES["true_junction"]:
            draw_disk(junction_map, node["xy"], junction_radius)
        else:
            draw_disk(endpoint_map, node["xy"], endpoint_radius)

    crossings = crossing_points(geometry)
    crossing_map = np.zeros((height, width), dtype=np.uint8)
    for xy in crossings:
        draw_disk(crossing_map, xy, crossing_radius)

    return {
        "source_float": source,
        "semantic_mask": (overlap > 0).astype(np.uint8),
        "centerline_mask": centerline,
        "endpoint_map": endpoint_map,
        "junction_map": junction_map,
        "crossing_map": crossing_map,
        "overlap_count": overlap,
        "membership_y": np.concatenate(membership_y).astype(np.int32) if membership_y else np.zeros(0, dtype=np.int32),
        "membership_x": np.concatenate(membership_x).astype(np.int32) if membership_x else np.zeros(0, dtype=np.int32),
        "membership_instance_id": np.concatenate(membership_i).astype(np.int32) if membership_i else np.zeros(0, dtype=np.int32),
        "geometric_crossing_points_xy": np.asarray(crossings, dtype=np.float32).reshape((-1, 2)),
    }


def rasterize_polyline(points_xy: np.ndarray, shape: tuple[int, int], radius: float) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    points = points_xy.astype(np.float32)
    for a, b in zip(points[:-1], points[1:]):
        add_segment(mask, a, b, radius)
    return mask


def add_segment(mask: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> None:
    height, width = mask.shape
    min_x = max(int(np.floor(min(a[0], b[0]) - radius - 1)), 0)
    max_x = min(int(np.ceil(max(a[0], b[0]) + radius + 1)), width - 1)
    min_y = max(int(np.floor(min(a[1], b[1]) - radius - 1)), 0)
    max_y = min(int(np.ceil(max(a[1], b[1]) + radius + 1)), height - 1)
    if max_x < min_x or max_y < min_y:
        return
    yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    px = xx.astype(np.float32) + 0.5
    py = yy.astype(np.float32) + 0.5
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-6:
        dist2 = (px - a[0]) ** 2 + (py - a[1]) ** 2
    else:
        t = np.clip(((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / denom, 0, 1)
        qx = a[0] + t * ab[0]
        qy = a[1] + t * ab[1]
        dist2 = (px - qx) ** 2 + (py - qy) ** 2
    mask[min_y : max_y + 1, min_x : max_x + 1] |= dist2 <= radius * radius


def draw_disk(arr: np.ndarray, xy: np.ndarray | tuple[float, float], radius: float) -> None:
    x, y = float(xy[0]), float(xy[1])
    height, width = arr.shape
    min_x = max(int(np.floor(x - radius - 1)), 0)
    max_x = min(int(np.ceil(x + radius + 1)), width - 1)
    min_y = max(int(np.floor(y - radius - 1)), 0)
    max_y = min(int(np.ceil(y + radius + 1)), height - 1)
    yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    dist2 = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2
    arr[min_y : max_y + 1, min_x : max_x + 1][dist2 <= radius * radius] = 1


def crossing_points(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    shared = shared_nodes_by_fiber(geometry)
    crossings: list[tuple[float, float]] = []
    fibers = geometry["fibers"]
    for i, fa in enumerate(fibers):
        for fb in fibers[i + 1 :]:
            if shared[int(fa["fiber_id"])] & shared[int(fb["fiber_id"])]:
                continue
            for a0, a1 in zip(fa["points_xy"][:-1], fa["points_xy"][1:]):
                for b0, b1 in zip(fb["points_xy"][:-1], fb["points_xy"][1:]):
                    hit = segment_intersection(a0, a1, b0, b1)
                    if hit is not None and not near_existing(crossings, hit):
                        crossings.append((float(hit[0]), float(hit[1])))
    return crossings


def shared_nodes_by_fiber(geometry: dict[str, Any]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for edge in geometry["edges"]:
        out[int(edge["fiber_id"])] = set(map(int, edge["node_indices"]))
    return out


def segment_intersection(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray | None:
    da = a1 - a0
    db = b1 - b0
    denom = da[0] * db[1] - da[1] * db[0]
    if abs(float(denom)) < 1e-6:
        return None
    delta = b0 - a0
    t = (delta[0] * db[1] - delta[1] * db[0]) / denom
    u = (delta[0] * da[1] - delta[1] * da[0]) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return a0 + t * da
    return None


def near_existing(points: list[tuple[float, float]], candidate: np.ndarray, tol: float = 2.0) -> bool:
    for x, y in points:
        if (float(candidate[0]) - x) ** 2 + (float(candidate[1]) - y) ** 2 <= tol * tol:
            return True
    return False

