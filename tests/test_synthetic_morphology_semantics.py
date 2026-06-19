import copy

import numpy as np
import pytest

from fibras.calibration.morphology import (
    generated_scene_diagnostics,
    real_scene_diagnostics,
)
from fibras.synthetic.geometry3d import smooth_catmull_rom
from fibras.synthetic.morphology3d import (
    append_fiber,
    clip_curve_to_volume,
)
from fibras.synthetic.rasterizer3d import (
    projected_crossings_bruteforce,
    projected_crossings_grid,
    rasterize_3d_sample,
)
from fibras.synthetic.schema import (
    BOUNDARY_CODES,
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY,
    GENERATOR_VERSION_3D_MORPHOLOGY_LEGACY,
    TRACE_TERMINATION_STATUS_CODES,
    load_yaml,
)
from fibras.synthetic.storage import build_sample, validate_sample_arrays


def morphology_config(mode="mixed_morphology"):
    config = load_yaml(
        "configs/synthetic_sted/morphology_heterogeneity_review.yaml"
    )
    config["dataset_name"] = "semantic_test"
    config["sample_count"] = 8
    config["geometry"]["image_shape"] = [192, 192]
    config["geometry"]["width_calibration"]["length_px"] = 140
    config["scene_morphology"].pop("review_modes", None)
    config["scene_morphology"]["mode"] = mode
    config["scene_morphology"]["cluster_radius_range_px"] = [30, 60]
    config["individual_filaments"]["contour_length_range_px"] = [45, 110]
    config["bundles"]["bundle_length_range_px"] = [60, 130]
    config["clumps"]["radius_range_px"] = [12, 22]
    config["clumps"]["fragment_count_range"] = [8, 12]
    config["targets"]["distance_transform_enabled"] = False
    return config


def test_supervised_and_latent_memberships_are_explicitly_separated():
    _, arrays, metadata = build_sample(morphology_config(), 0)
    assert "membership_y" not in arrays
    assert "overlap_count" not in arrays
    assert arrays["supervised_membership_y"].size > 0
    assert arrays["latent_geometry_membership_y"].size > 0
    assert set(np.unique(arrays["supervised_membership_class_id"])) <= {1, 2, 3}
    assert metadata["target_available"]["instance_membership"] is False
    assert metadata["target_available"]["supervised_instance_membership"] is True
    assert not (
        set(metadata["target_roles"]["supervised"])
        & set(metadata["target_roles"]["latent_synthetic_provenance"])
    )
    assert validate_sample_arrays(metadata["sample_id"], arrays, metadata) == []


def test_latent_bundle_and_clump_curves_are_not_supervised_edges_or_filaments():
    _, arrays, _ = build_sample(morphology_config(), 1)
    fiber_type = dict(zip(arrays["fiber_ids"], arrays["fiber_structure_type"]))
    filament_ids = set(arrays["individual_filament_membership_instance_id"])
    assert all(fiber_type[int(fid)] != 3 for fid in filament_ids)
    assert np.all(
        arrays["edge_supervised"][arrays["edge_structure_type"] != 1] == 0
    )
    assert np.all(
        arrays["node_supervised"][
            arrays["node_termination_status"]
            != TRACE_TERMINATION_STATUS_CODES["valid_endpoint"]
        ]
        == 0
    )
    latent_edge_indices = np.flatnonzero(arrays["edge_structure_type"] != 1)
    latent_node_indices = arrays["edge_node_indices"][latent_edge_indices].ravel()
    assert np.all(arrays["node_supervised"][latent_node_indices] == 0)


def test_morphology_radius_variation_remains_within_documented_bound():
    config = morphology_config("clump_dominated")
    _, arrays, _ = build_sample(config, 5)
    upper = (
        config["geometry"]["fluorophore"]["radius_range_px"][1]
        * (
            1
            + 3
            * config["geometry"]["fluorophore"][
                "radius_variation_amplitude"
            ]
        )
    )
    assert float(arrays["fiber_sample_radius"].max()) <= upper + 1e-6


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ([-5, 20, 20], BOUNDARY_CODES["x_min"]),
        ([80, 20, 20], BOUNDARY_CODES["x_max"]),
        ([20, -5, 20], BOUNDARY_CODES["y_min"]),
        ([20, 80, 20], BOUNDARY_CODES["y_max"]),
        ([20, 20, -5], BOUNDARY_CODES["z_min"]),
        ([20, 20, 60], BOUNDARY_CODES["z_max"]),
        ([-5, -5, 20], BOUNDARY_CODES["x_min"] | BOUNDARY_CODES["y_min"]),
    ],
)
def test_boundary_intersections_terminate_without_edge_flattening(
    endpoint, expected
):
    config = morphology_config()["geometry"]
    config["image_shape"] = [64, 64]
    config["volume_depth_px"] = 48
    fibers, nodes, edges = [], [], []
    append_fiber(
        fibers,
        nodes,
        edges,
        np.asarray([[20, 20, 20], endpoint], dtype=np.float32),
        config,
        {},
        np.random.default_rng(2),
        structure_type="individual_filament",
    )
    assert fibers[0]["trace_end_status"] == "boundary_truncation"
    assert fibers[0]["end_boundary_code"] & expected == expected
    assert not nodes[-1]["supervised_endpoint"]
    assert edges[0]["truncated_end"]
    boundary_points = np.count_nonzero(
        np.any(
            np.isclose(
                fibers[0]["points_xyz"],
                np.asarray([0, 0, 0], dtype=np.float32),
            ),
            axis=1,
        )
    )
    assert boundary_points <= 1


def test_spline_overshoot_is_clipped_at_first_intersection():
    raw = np.asarray(
        [[2, 2, 5], [0.1, 2, 5], [0.1, 8, 5], [2, 8, 5]],
        dtype=np.float32,
    )
    smooth = smooth_catmull_rom(raw, 8)
    assert smooth[:, 0].min() < 0
    clipped, start_boundary, end_boundary = clip_curve_to_volume(
        smooth, np.asarray([10, 10, 10], dtype=np.float64)
    )
    assert np.all(clipped >= 0)
    assert np.all(clipped <= 10)
    assert end_boundary or start_boundary
    assert np.count_nonzero(np.isclose(clipped[:, 0], 0)) <= 1


def test_bundle_child_offset_outside_volume_is_boundary_truncated():
    config = morphology_config()["geometry"]
    config["image_shape"] = [64, 64]
    fibers, nodes, edges = [], [], []
    append_fiber(
        fibers,
        nodes,
        edges,
        np.asarray([[30, 30, 20], [80, 30, 20]], dtype=np.float32),
        config,
        {},
        np.random.default_rng(9),
        structure_type="bundle_child",
        supervised_samples=np.ones(2, dtype=np.uint8),
        start_status="ambiguous_termination",
        end_status="ambiguous_termination",
    )
    assert fibers[0]["trace_end_status"] == "boundary_truncation"
    assert fibers[0]["end_boundary_code"] == BOUNDARY_CODES["x_max"]
    assert not nodes[-1]["supervised_endpoint"]


def test_filament_to_clump_start_outside_becomes_boundary_truncation():
    config = morphology_config()["geometry"]
    config["image_shape"] = [64, 64]
    fibers, nodes, edges = [], [], []
    append_fiber(
        fibers,
        nodes,
        edges,
        np.asarray([[-10, 30, 20], [30, 30, 20]], dtype=np.float32),
        config,
        {},
        np.random.default_rng(3),
        structure_type="individual_filament",
        end_status="terminates_in_clump",
    )
    assert fibers[0]["trace_start_status"] == "boundary_truncation"
    assert fibers[0]["trace_end_status"] == "terminates_in_clump"


def test_boundary_truncation_is_absent_from_supervised_endpoint_map():
    config = morphology_config()
    fibers, nodes, edges = [], [], []
    append_fiber(
        fibers,
        nodes,
        edges,
        np.asarray([[20, 20, 20], [220, 20, 20]], dtype=np.float32),
        config["geometry"],
        {},
        np.random.default_rng(4),
        structure_type="individual_filament",
    )
    geometry = {
        "image_shape": (192, 192),
        "volume_depth_px": 96.0,
        "focal_plane_z_px": 48.0,
        "fibers": fibers,
        "nodes": nodes,
        "edges": edges,
        "bundles": [],
        "clumps": [],
        "parameters": {"scenario": "boundary_fixture"},
    }
    arrays, _ = rasterize_3d_sample(
        geometry,
        config["targets"],
        config["optical_model"],
        config["output_mapping"],
        7,
    )
    assert arrays["endpoint_map"][20, 191] == 0
    assert arrays["endpoint_map"][20, 20] == 1


def crossing_geometry(first_supervised, second_structure="individual_filament"):
    def fiber(fid, points, structure, supervised):
        return {
            "fiber_id": fid,
            "points_xyz": np.asarray(points, dtype=np.float32),
            "structure_type": structure,
            "supervised_centerline_sample": np.asarray(supervised, dtype=np.uint8),
        }

    return {
        "fibers": [
            fiber(
                1,
                [[2, 5, 10], [8, 5, 10]],
                "bundle_child",
                first_supervised,
            ),
            fiber(
                2,
                [[5, 2, 12], [5, 8, 12]],
                second_structure,
                [1, 1] if second_structure != "clump_fragment" else [0, 0],
            ),
        ],
        "edges": [
            {"fiber_id": 1, "node_indices": (0, 1)},
            {"fiber_id": 2, "node_indices": (2, 3)},
        ],
    }


def test_crossings_use_only_resolved_bundle_child_segments():
    resolved = crossing_geometry([1, 1])
    unresolved = crossing_geometry([0, 0])
    transition = crossing_geometry([1, 0])
    assert len(projected_crossings_grid(resolved)[0]) == 1
    assert projected_crossings_grid(unresolved)[0] == []
    assert projected_crossings_grid(transition)[0] == []


def test_resolved_bundle_crossings_preserve_identity_and_order_invariance():
    geometry = crossing_geometry([1, 1], "bundle_child")
    forward = projected_crossings_grid(geometry)[0]
    reverse_geometry = copy.deepcopy(geometry)
    reverse_geometry["fibers"].reverse()
    reverse = projected_crossings_grid(reverse_geometry)[0]
    assert len(forward) == len(reverse) == 1
    assert np.array_equal(forward[0]["fiber_ids"], [1, 2])
    assert np.array_equal(forward[0]["fiber_ids"], reverse[0]["fiber_ids"])
    assert np.allclose(forward[0]["xy"], reverse[0]["xy"])
    assert len(projected_crossings_bruteforce(geometry)) == 1


def test_resolved_child_crossing_clump_fragment_is_not_a_filament_crossing():
    geometry = crossing_geometry([1, 1], "clump_fragment")
    assert projected_crossings_grid(geometry)[0] == []


def test_class_signal_alignment_is_reported_and_validator_enforces_limits():
    _, arrays, metadata = build_sample(morphology_config(), 3)
    alignment = metadata["rendering_report"]["class_signal_alignment"]
    for class_name in [
        "individual_filament",
        "bundle",
        "clump",
        "uncertain_transition",
    ]:
        assert class_name in alignment
        assert "fraction_of_class_mask_with_nonzero_signal" in alignment[class_name]
    malformed = copy.deepcopy(metadata)
    malformed["rendering_report"]["class_signal_alignment"][
        "individual_filament"
    ]["fraction_of_class_mask_with_nonzero_signal"] = 0.0
    errors = validate_sample_arrays(malformed["sample_id"], arrays, malformed)
    assert any("insufficient rendered signal" in error for error in errors)


def test_schema_06_remains_explicitly_validated_without_reinterpretation():
    _, arrays, metadata = build_sample(morphology_config(), 2)
    legacy = dict(arrays)
    legacy["membership_y"] = legacy.pop("latent_geometry_membership_y")
    legacy["membership_x"] = legacy.pop("latent_geometry_membership_x")
    legacy["membership_instance_id"] = legacy.pop(
        "latent_geometry_membership_instance_id"
    )
    legacy["overlap_count"] = legacy.pop("latent_geometry_overlap_count")
    for name in [
        "supervised_membership_y",
        "supervised_membership_x",
        "supervised_membership_instance_id",
        "supervised_membership_class_id",
        "supervised_overlap_count",
        "node_supervised",
        "node_termination_status",
        "node_boundary_code",
        "edge_supervised",
        "edge_structure_type",
        "fiber_start_boundary_code",
        "fiber_end_boundary_code",
        "individual_filament_signal",
        "bundle_signal",
        "clump_signal",
    ]:
        legacy.pop(name, None)
    legacy_metadata = copy.deepcopy(metadata)
    legacy_metadata["dataset_schema_version"] = (
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY
    )
    legacy_metadata["generator_version"] = GENERATOR_VERSION_3D_MORPHOLOGY_LEGACY
    legacy_metadata["array_names"] = sorted(legacy)
    legacy_metadata["dtypes"] = {
        name: str(value.dtype) for name, value in legacy.items()
    }
    legacy_metadata["shapes"] = {
        name: list(value.shape) for name, value in legacy.items()
    }
    assert validate_sample_arrays(
        legacy_metadata["sample_id"], legacy, legacy_metadata
    ) == []


def test_ground_truth_and_image_proxy_diagnostics_remain_separate():
    _, arrays, metadata = build_sample(morphology_config(), 4)
    ground_truth = generated_scene_diagnostics(arrays, metadata, tile_size=32)
    proxy = real_scene_diagnostics(
        arrays["render_uint8"], "proxy", "synthetic_proxy", tile_size=32
    )
    assert ground_truth["supervised_membership_count"] != "not_available"
    assert ground_truth["bundle_area_fraction"] != "not_available"
    assert proxy["bundle_area_fraction"] == "not_available"
    assert proxy["scene_mode"] == "condition_blind_proxy"
