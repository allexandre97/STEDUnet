"""Schema constants and config helpers for synthetic STED samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DATASET_SCHEMA_VERSION = "synthetic_sted_mvp_0.1.0"
GENERATOR_VERSION = "fibras_synthetic_mvp_0.1.0"

NODE_TYPES = {
    "endpoint": 1,
    "true_junction": 2,
    "truncated_endpoint": 3,
}

CALIBRATION_STATUSES = {
    "procedural_unmatched",
    "empirically_matched",
    "approximately_physical",
    "physically_calibrated",
}

REQUIRED_ARRAYS = {
    "fiber_points_xy",
    "fiber_point_offsets",
    "fiber_ids",
    "fiber_width",
    "fiber_intensity",
    "node_xy",
    "node_type",
    "edge_node_indices",
    "edge_fiber_id",
    "edge_truncated_start",
    "edge_truncated_end",
    "source_float",
    "render_float",
    "render_uint8",
    "semantic_mask",
    "centerline_mask",
    "endpoint_map",
    "junction_map",
    "crossing_map",
    "overlap_count",
    "membership_y",
    "membership_x",
    "membership_instance_id",
    "geometric_crossing_points_xy",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    return data


def require_config(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"missing required config key: {key}")
    return config[key]

