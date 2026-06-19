"""Schema constants and config helpers for synthetic STED samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DATASET_SCHEMA_VERSION = "synthetic_sted_mvp_0.1.0"
GENERATOR_VERSION = "fibras_synthetic_mvp_0.1.0"
DATASET_SCHEMA_VERSION_3D_LEGACY = "synthetic_sted_3d_rasterizer_0.2.0"
DATASET_SCHEMA_VERSION_3D_NORMALIZED = "synthetic_sted_3d_rasterizer_0.3.0"
DATASET_SCHEMA_VERSION_3D_HARDENED = "synthetic_sted_3d_rasterizer_0.4.0"
DATASET_SCHEMA_VERSION_3D = "synthetic_sted_3d_rasterizer_0.5.0"
GENERATOR_VERSION_3D = "fibras_persistent_chain_3d_0.5.0"
DATASET_SCHEMA_VERSION_3D_MORPHOLOGY = "synthetic_sted_3d_morphology_0.6.0"
GENERATOR_VERSION_3D_MORPHOLOGY = "fibras_morphology_scene_3d_0.6.0"

GENERATOR_MODES = {
    "structural_test_2d",
    "legacy_2d",
    "persistent_chain_3d",
    "morphology_scene_3d",
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

REAL_SEMANTIC_CLASSES = {
    0: "background",
    1: "individual_filament",
    2: "bundle",
    3: "clump",
    255: "uncertain_ignore",
}

TRACE_TERMINATION_STATUSES = {
    "valid_endpoint",
    "boundary_truncation",
    "terminates_in_bundle",
    "terminates_in_clump",
    "ambiguous_termination",
}

TRACE_TERMINATION_STATUS_CODES = {
    "valid_endpoint": 1,
    "boundary_truncation": 2,
    "terminates_in_bundle": 3,
    "terminates_in_clump": 4,
    "ambiguous_termination": 5,
}

FIBER_STRUCTURE_TYPE_CODES = {
    "individual_filament": 1,
    "bundle_child": 2,
    "clump_fragment": 3,
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
REQUIRED_ARRAYS_3D_HARDENED = (REQUIRED_ARRAYS_3D_LEGACY - {"source_float"}) | {
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
REQUIRED_ARRAYS_3D = REQUIRED_ARRAYS_3D_HARDENED | {
    "projected_crossing_fiber_ids",
    "projected_crossing_segment_indices",
}
REQUIRED_ARRAYS_3D_MORPHOLOGY = REQUIRED_ARRAYS_3D | {
    "semantic_class_mask",
    "individual_filament_mask",
    "bundle_mask",
    "clump_mask",
    "uncertain_ignore_mask",
    "filament_centerline_mask",
    "bundle_axis_mask",
    "bundle_transition_mask",
    "clump_transition_mask",
    "individual_filament_membership_y",
    "individual_filament_membership_x",
    "individual_filament_membership_instance_id",
    "bundle_membership_y",
    "bundle_membership_x",
    "bundle_membership_instance_id",
    "clump_membership_y",
    "clump_membership_x",
    "clump_membership_instance_id",
    "bundle_axis_points_xyz",
    "bundle_axis_points_xy",
    "bundle_axis_point_offsets",
    "bundle_axis_radius_px",
    "bundle_axis_unresolved_sample",
    "bundle_ids",
    "bundle_partial_resolution",
    "bundle_twist_rate",
    "bundle_converging",
    "bundle_diverging",
    "bundle_child_fiber_ids",
    "bundle_child_fiber_offsets",
    "clump_ids",
    "clump_center_xyz",
    "clump_radius_xyz",
    "clump_internal_density",
    "clump_irregularity",
    "clump_edge_diffuseness",
    "clump_fragment_fiber_ids",
    "clump_fragment_fiber_offsets",
    "fiber_structure_type",
    "fiber_parent_bundle_id",
    "fiber_parent_clump_id",
    "fiber_supervised_centerline_sample",
    "trace_start_status",
    "trace_end_status",
    "trace_fiber_ids",
}

REQUIRED_ARRAYS_BY_SCHEMA = {
    DATASET_SCHEMA_VERSION: REQUIRED_ARRAYS,
    DATASET_SCHEMA_VERSION_3D_LEGACY: REQUIRED_ARRAYS_3D_LEGACY,
    DATASET_SCHEMA_VERSION_3D_NORMALIZED: REQUIRED_ARRAYS_3D_NORMALIZED,
    DATASET_SCHEMA_VERSION_3D_HARDENED: REQUIRED_ARRAYS_3D_HARDENED,
    DATASET_SCHEMA_VERSION_3D: REQUIRED_ARRAYS_3D,
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY: REQUIRED_ARRAYS_3D_MORPHOLOGY,
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
