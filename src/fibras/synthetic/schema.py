"""Schema constants and config helpers for synthetic STED samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DATASET_SCHEMA_VERSION = "synthetic_sted_mvp_0.1.0"
GENERATOR_VERSION = "fibras_synthetic_mvp_0.1.0"
DATASET_SCHEMA_VERSION_3D_LEGACY = "synthetic_sted_3d_rasterizer_0.2.0"
DATASET_SCHEMA_VERSION_3D_NORMALIZED = "synthetic_sted_3d_rasterizer_0.3.0"
DATASET_SCHEMA_VERSION_3D = "synthetic_sted_3d_rasterizer_0.4.0"
GENERATOR_VERSION_3D = "fibras_persistent_chain_3d_0.4.0"

GENERATOR_MODES = {
    "structural_test_2d",
    "legacy_2d",
    "persistent_chain_3d",
}

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

REQUIRED_ARRAYS_3D = REQUIRED_ARRAYS | {
    "fiber_points_xyz",
    "fiber_vertex_xyz",
    "fiber_vertex_offsets",
    "fiber_sample_amplitude",
    "fiber_sample_radius",
    "node_xyz",
    "projection_mask",
    "in_focus_mask",
    "visible_signal_mask",
    "visible_overlap_count",
    "visible_membership_y",
    "visible_membership_x",
    "visible_membership_instance_id",
    "nearest_depth_map",
    "weighted_mean_depth_map",
    "in_focus_signal",
    "out_of_focus_signal",
    "total_clean_signal",
    "projected_crossing_map",
    "near_coplanar_crossing_map",
    "projected_crossing_points_xy",
    "projected_crossing_depths",
    "near_coplanar_crossing_points_xy",
}

REQUIRED_ARRAYS_3D_LEGACY = set(REQUIRED_ARRAYS_3D)
REQUIRED_ARRAYS_3D_NORMALIZED = REQUIRED_ARRAYS_3D_LEGACY | {
    "sample_arc_length_weight",
    "fluorophore_density_per_length",
    "total_optical_signal",
    "core_signal",
    "halo_signal",
    "trace_points_xy",
    "trace_point_offsets",
    "trace_ids",
    "trace_status",
    "trace_source",
    "ignore_mask",
    "distance_transform",
    "orientation_cos2",
    "orientation_sin2",
}
REQUIRED_ARRAYS_3D = (REQUIRED_ARRAYS_3D_LEGACY - {"source_float"}) | {
    "sample_arc_length_weight",
    "fluorophore_density_per_length",
    "total_optical_signal",
    "core_signal",
    "halo_signal",
    "trace_points_xy",
    "trace_point_offsets",
    "trace_ids",
    "trace_status",
    "trace_source",
    "ignore_mask",
    "line_source_float",
    "geometric_support_preview",
    "visible_instance_membership_y",
    "visible_instance_membership_x",
    "visible_instance_membership_id",
    "contributing_membership_y",
    "contributing_membership_x",
    "contributing_membership_instance_id",
    "contributing_overlap_count",
    "combined_only_visible_mask",
    "visible_unassigned_mask",
}

REQUIRED_ARRAYS_BY_SCHEMA = {
    DATASET_SCHEMA_VERSION: REQUIRED_ARRAYS,
    DATASET_SCHEMA_VERSION_3D_LEGACY: REQUIRED_ARRAYS_3D_LEGACY,
    DATASET_SCHEMA_VERSION_3D_NORMALIZED: REQUIRED_ARRAYS_3D_NORMALIZED,
    DATASET_SCHEMA_VERSION_3D: REQUIRED_ARRAYS_3D,
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
