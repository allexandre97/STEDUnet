import copy

import numpy as np

from fibras.calibration.morphology import (
    generated_scene_diagnostics,
    tile_occupancy,
)
from fibras.synthetic.morphology3d import generate_morphology_geometry
from fibras.synthetic.schema import (
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
    TRACE_TERMINATION_STATUS_CODES,
    load_yaml,
)
from fibras.synthetic.storage import build_sample, validate_sample_arrays


def config(mode="mixed_morphology"):
    cfg = load_yaml(
        "configs/synthetic_sted/morphology_heterogeneity_review.yaml"
    )
    cfg["dataset_name"] = "morphology_test"
    cfg["sample_count"] = 8
    cfg["geometry"]["image_shape"] = [256, 256]
    cfg["geometry"]["width_calibration"]["length_px"] = 160
    cfg["scene_morphology"].pop("review_modes", None)
    cfg["scene_morphology"]["mode"] = mode
    cfg["scene_morphology"]["cluster_radius_range_px"] = [35, 70]
    cfg["individual_filaments"]["contour_length_range_px"] = [50, 150]
    cfg["bundles"]["bundle_length_range_px"] = [70, 150]
    cfg["bundles"]["bundle_radius_range_px"] = [4, 8]
    cfg["clumps"]["radius_range_px"] = [12, 24]
    cfg["clumps"]["fragment_count_range"] = [7, 12]
    cfg["targets"]["distance_transform_enabled"] = False
    return cfg


def test_fixed_seed_reproduces_identical_morphology_scene():
    first = build_sample(config(), 0)
    second = build_sample(config(), 0)
    assert first[2]["dataset_schema_version"] == (
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY
    )
    assert set(first[1]) == set(second[1])
    for name in first[1]:
        assert np.array_equal(first[1][name], second[1][name]), name


def test_clustered_scene_has_more_tile_variance_than_isolated_baseline():
    _, isolated, isolated_meta = build_sample(config("isolated_filaments"), 0)
    _, clustered, clustered_meta = build_sample(
        config("clustered_filament_network"), 0
    )
    isolated_diag = generated_scene_diagnostics(
        isolated, isolated_meta, tile_size=32
    )
    clustered_diag = generated_scene_diagnostics(
        clustered, clustered_meta, tile_size=32
    )
    assert float(clustered_diag["tile_occupancy_variance"]) > float(
        isolated_diag["tile_occupancy_variance"]
    )
    assert float(isolated_diag["empty_tile_fraction"]) > 0.4


def test_local_alignment_increases_orientation_coherence():
    _, aligned, aligned_meta = build_sample(
        config("aligned_filament_domain"), 1
    )
    _, tangled, tangled_meta = build_sample(config("dense_tangle"), 1)
    aligned_diag = generated_scene_diagnostics(
        aligned, aligned_meta, tile_size=32
    )
    tangled_diag = generated_scene_diagnostics(
        tangled, tangled_meta, tile_size=32
    )
    assert float(aligned_diag["orientation_coherence_p50"]) >= float(
        tangled_diag["orientation_coherence_p50"]
    )


def test_bundle_children_parent_width_and_hidden_centerline_semantics():
    cfg = config("bundle_dominated")
    geometry = generate_morphology_geometry(cfg, 0)
    assert geometry["bundles"]
    for bundle in geometry["bundles"]:
        children = set(map(int, bundle["child_fiber_ids"]))
        assert children
        for fiber in geometry["fibers"]:
            if int(fiber["fiber_id"]) in children:
                assert fiber["parent_bundle_id"] == bundle["bundle_id"]
        assert float(np.median(bundle["axis_radius_px"]) * 2) > 2.2
    _, arrays, metadata = build_sample(cfg, 0)
    assert not np.any(
        arrays["filament_centerline_mask"] & arrays["bundle_mask"]
    )
    assert not np.array_equal(
        arrays["bundle_axis_mask"], arrays["filament_centerline_mask"]
    )
    assert TRACE_TERMINATION_STATUS_CODES["terminates_in_bundle"] in set(
        arrays["trace_start_status"]
    ) | set(arrays["trace_end_status"])
    assert arrays["bundle_axis_radius_px"].shape[0] == arrays[
        "bundle_axis_points_xyz"
    ].shape[0]
    assert arrays["bundle_partial_resolution"].shape == arrays[
        "bundle_ids"
    ].shape
    assert validate_sample_arrays(metadata["sample_id"], arrays, metadata) == []


def test_clumps_are_compact_nonempty_and_have_no_supervised_skeleton():
    cfg = config("clump_dominated")
    _, arrays, metadata = build_sample(cfg, 0)
    assert arrays["clump_mask"].sum() > 0
    assert not np.any(
        arrays["clump_mask"] & arrays["filament_centerline_mask"]
    )
    assert not np.any(arrays["clump_mask"] & arrays["endpoint_map"])
    assert len(np.unique(arrays["clump_membership_instance_id"])) >= 2
    assert arrays["clump_internal_density"].shape == arrays["clump_ids"].shape
    assert arrays["clump_irregularity"].shape == arrays["clump_ids"].shape
    y, x = np.nonzero(arrays["clump_mask"])
    bounding_area = (y.max() - y.min() + 1) * (x.max() - x.min() + 1)
    assert arrays["clump_mask"].sum() / bounding_area > 0.01
    assert validate_sample_arrays(metadata["sample_id"], arrays, metadata) == []


def test_multiclass_contract_and_binary_compatibility():
    _, arrays, metadata = build_sample(config(), 2)
    assert set(map(int, np.unique(arrays["semantic_class_mask"]))) <= {
        0,
        1,
        2,
        3,
        255,
    }
    for class_id, name in [
        (1, "individual_filament_mask"),
        (2, "bundle_mask"),
        (3, "clump_mask"),
        (255, "uncertain_ignore_mask"),
    ]:
        assert np.array_equal(
            arrays[name],
            (arrays["semantic_class_mask"] == class_id).astype(np.uint8),
        )
    assert np.array_equal(
        arrays["semantic_mask"],
        np.isin(arrays["semantic_class_mask"], [1, 2, 3]).astype(np.uint8),
    )
    assert metadata["multiclass_target_contract"]["class_ids"][2] == "bundle"


def test_foreground_scale_does_not_change_geometry_or_class_targets():
    base = config()
    bright = copy.deepcopy(base)
    bright["output_mapping"]["foreground_scale"] = 2.0
    _, base_arrays, _ = build_sample(base, 3)
    _, bright_arrays, _ = build_sample(bright, 3)
    structural = [
        "semantic_class_mask",
        "semantic_mask",
        "individual_filament_mask",
        "bundle_mask",
        "clump_mask",
        "uncertain_ignore_mask",
        "filament_centerline_mask",
        "bundle_axis_mask",
        "endpoint_map",
        "trace_points_xy",
        "trace_start_status",
        "trace_end_status",
    ]
    for name in structural:
        assert np.array_equal(base_arrays[name], bright_arrays[name]), name
    assert not np.array_equal(
        base_arrays["total_clean_signal"], bright_arrays["total_clean_signal"]
    )


def test_scientific_metadata_labels_do_not_select_morphology_parameters():
    base = config()
    labelled = copy.deepcopy(base)
    labelled.update(
        {
            "culture_id": "PN999",
            "disease": "placeholder",
            "tau_isoform": "placeholder",
            "div": 99,
        }
    )
    first = generate_morphology_geometry(base, 4)
    second = generate_morphology_geometry(labelled, 4)
    assert first["parameters"] == second["parameters"]
    for a, b in zip(first["fibers"], second["fibers"]):
        assert np.array_equal(a["points_xyz"], b["points_xyz"])


def test_tile_occupancy_reports_empty_tiles():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:16, :16] = 1
    occupancy = tile_occupancy(mask, 16)
    assert occupancy[0, 0] == 1
    assert np.count_nonzero(occupancy == 0) == 15
