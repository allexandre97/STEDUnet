import copy

import numpy as np

from fibras.calibration.morphology import (
    generated_scene_diagnostics,
    tile_occupancy,
)
import fibras.synthetic.morphology3d as morphology3d
from scripts.build_clump_ignore_stress_report import sample_metrics
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


def test_invalid_in_volume_candidate_is_retried_deterministically(monkeypatch):
    cfg = config("isolated_filaments")
    cfg["individual_filaments"]["count_range_by_mode"] = {
        "isolated_filaments": [4, 4]
    }
    cfg["geometry"]["invalid_geometry_max_attempts"] = 4
    original_clip = morphology3d.clip_curve_to_volume

    def fail_first_clip_once():
        state = {"calls": 0}

        def wrapped(points, limits):
            state["calls"] += 1
            if state["calls"] == 1:
                raise ValueError("fiber has no valid in-volume segment")
            return original_clip(points, limits)

        return wrapped

    monkeypatch.setattr(morphology3d, "clip_curve_to_volume", fail_first_clip_once())
    first = build_sample(cfg, 3)
    monkeypatch.setattr(morphology3d, "clip_curve_to_volume", fail_first_clip_once())
    second = build_sample(cfg, 3)

    assert first[2]["geometry_parameters"]["invalid_geometry_candidate_count"] == 1
    assert first[2]["geometry_parameters"]["invalid_filament_count"] == 1
    assert first[2]["geometry_parameters"]["resample_attempt_count"] >= 1
    assert validate_sample_arrays(first[0], first[1], first[2]) == []
    for name in [
        "render_uint8",
        "semantic_class_mask",
        "real_compatible_semantic_mask",
        "real_compatible_skeleton_mask",
    ]:
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
        "individual_filament_source_support_mask",
        "bundle_source_support_mask",
        "clump_source_support_mask",
        "trace_points_xy",
        "trace_start_status",
        "trace_end_status",
    ]
    for name in structural:
        assert np.array_equal(base_arrays[name], bright_arrays[name]), name
    assert not np.array_equal(
        base_arrays["semantic_class_mask"], bright_arrays["semantic_class_mask"]
    )
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


def test_clump_ignore_stress_targets_remain_disjoint_and_schema_valid():
    cfg = load_yaml("configs/synthetic_sted/clump_ignore_stress_schema08.yaml")
    cfg["dataset_name"] = "clump_ignore_stress_test"
    cfg["sample_count"] = 2
    cfg["geometry"]["image_shape"] = [256, 256]
    cfg["geometry"]["width_calibration"]["length_px"] = 160
    cfg["targets"]["distance_transform_enabled"] = False
    _, arrays, metadata = build_sample(cfg, 0)
    assert arrays["real_compatible_clump_mask"].sum() > 0
    assert arrays["real_compatible_uncertain_ignore_mask"].sum() > 0
    assert not np.any(
        arrays["real_compatible_fibrous_mask"]
        & arrays["real_compatible_clump_mask"]
    )
    assert not np.any(
        arrays["real_compatible_fibrous_mask"]
        & arrays["real_compatible_uncertain_ignore_mask"]
    )
    assert not np.any(
        arrays["real_compatible_skeleton_mask"]
        & arrays["real_compatible_clump_mask"]
    )
    assert not np.any(
        arrays["real_compatible_skeleton_mask"]
        & arrays["real_compatible_uncertain_ignore_mask"]
    )
    assert validate_sample_arrays(metadata["sample_id"], arrays, metadata) == []


def test_clump_hard_negative_fragments_remain_latent_not_skeleton_targets():
    cfg = load_yaml("configs/synthetic_sted/clump_ignore_stress_schema08.yaml")
    cfg["dataset_name"] = "clump_ignore_stress_test"
    cfg["geometry"]["image_shape"] = [256, 256]
    cfg["geometry"]["width_calibration"]["length_px"] = 160
    cfg["targets"]["distance_transform_enabled"] = False
    _, arrays, metadata = build_sample(cfg, 1)
    fragment_ids = set(map(int, arrays["clump_fragment_fiber_ids"]))
    assert fragment_ids
    for fiber_id, start, stop in zip(
        arrays["fiber_ids"],
        arrays["fiber_point_offsets"][:-1],
        arrays["fiber_point_offsets"][1:],
    ):
        if int(fiber_id) in fragment_ids:
            assert not np.any(arrays["fiber_supervised_centerline_sample"][start:stop])
    assert not np.any(arrays["filament_centerline_mask"] & arrays["clump_mask"])
    assert "clump_fragment_fiber_ids" in metadata["target_roles"]["latent_synthetic_provenance"]


def test_clump_ignore_stress_generation_is_condition_blind():
    base = load_yaml("configs/synthetic_sted/clump_ignore_stress_schema08.yaml")
    labelled = copy.deepcopy(base)
    labelled.update(
        {
            "culture_id": "PN148",
            "disease": "AD",
            "tau_isoform": "4R",
            "div": 3,
            "seed_class": "forbidden_metadata",
        }
    )
    first = generate_morphology_geometry(base, 2)
    second = generate_morphology_geometry(labelled, 2)
    assert first["parameters"] == second["parameters"]
    for a, b in zip(first["clumps"], second["clumps"]):
        assert np.array_equal(a["center_xyz"], b["center_xyz"])
        assert np.array_equal(a["radius_xyz"], b["radius_xyz"])


def test_clump_ignore_stress_v2_large_clumps_schema_valid_and_safe():
    cfg = load_yaml("configs/synthetic_sted/clump_ignore_stress_v2_schema08.yaml")
    cfg["dataset_name"] = "clump_ignore_stress_v2_test"
    cfg["sample_count"] = 1
    cfg["geometry"]["image_shape"] = [256, 256]
    cfg["geometry"]["width_calibration"]["length_px"] = 160
    cfg["targets"]["distance_transform_enabled"] = False
    cfg["clumps"]["count_range_by_mode"]["clump_dominated"] = [1, 1]
    cfg["clumps"]["size_mixture"] = {
        "very_large": {"weight": 1.0, "radius_range_px": [90, 105]}
    }
    _, arrays, metadata = build_sample(cfg, 0)
    assert arrays["clump_mask"].sum() / arrays["clump_mask"].size > 0.05
    assert arrays["real_compatible_clump_mask"].sum() > 0
    assert not np.any(arrays["real_compatible_fibrous_mask"] & arrays["real_compatible_clump_mask"])
    assert not np.any(arrays["real_compatible_skeleton_mask"] & arrays["real_compatible_clump_mask"])
    assert validate_sample_arrays(metadata["sample_id"], arrays, metadata) == []


def test_clump_ignore_stress_v2_dense_fragments_remain_latent():
    cfg = load_yaml("configs/synthetic_sted/clump_ignore_stress_v2_schema08.yaml")
    cfg["dataset_name"] = "clump_ignore_stress_v2_test"
    cfg["geometry"]["image_shape"] = [256, 256]
    cfg["geometry"]["width_calibration"]["length_px"] = 160
    cfg["targets"]["distance_transform_enabled"] = False
    _, arrays, metadata = build_sample(cfg, 1)
    fragment_ids = set(map(int, arrays["clump_fragment_fiber_ids"]))
    assert len(fragment_ids) >= 20
    assert "clump_fragment_fiber_ids" in metadata["target_roles"]["latent_synthetic_provenance"]
    assert "clump_fragment_fiber_ids" not in metadata["target_roles"]["supervised"]
    assert not np.any(arrays["filament_centerline_mask"] & arrays["clump_mask"])
    assert not np.any(arrays["real_compatible_skeleton_mask"] & arrays["real_compatible_uncertain_ignore_mask"])


def test_clump_patch_diagnostics_report_dominated_128px_crops():
    shape = (256, 256)
    arrays = {
        "real_compatible_semantic_mask": np.zeros(shape, dtype=np.uint8),
        "real_compatible_fibrous_mask": np.zeros(shape, dtype=np.uint8),
        "real_compatible_clump_mask": np.zeros(shape, dtype=np.uint8),
        "real_compatible_uncertain_ignore_mask": np.zeros(shape, dtype=np.uint8),
        "real_compatible_skeleton_mask": np.zeros(shape, dtype=np.uint8),
        "render_uint8": np.zeros(shape, dtype=np.uint8),
    }
    arrays["real_compatible_clump_mask"][:128, :128] = 1
    arrays["real_compatible_semantic_mask"][:128, :128] = 3
    arrays["real_compatible_uncertain_ignore_mask"][128:, :64] = 1
    arrays["real_compatible_semantic_mask"][128:, :64] = 255
    metrics = sample_metrics(arrays)
    assert metrics["max_patch_clump_fraction"] == 1.0
    assert metrics["patch_fraction_clump_gt_50"] == 0.25
    assert metrics["patch_fraction_uncertain_gt_10"] == 0.25
    assert metrics["patch_skeleton_pixels_inside_clump"] == 0


def test_clump_ignore_stress_v2_generation_is_condition_blind():
    base = load_yaml("configs/synthetic_sted/clump_ignore_stress_v2_schema08.yaml")
    labelled = copy.deepcopy(base)
    labelled.update(
        {
            "culture_id": "PN148",
            "disease": "AD",
            "tau_isoform": "4R",
            "div": 3,
            "seed_class": "forbidden_metadata",
        }
    )
    first = generate_morphology_geometry(base, 5)
    second = generate_morphology_geometry(labelled, 5)
    assert first["parameters"] == second["parameters"]
    for a, b in zip(first["clumps"], second["clumps"]):
        assert np.array_equal(a["center_xyz"], b["center_xyz"])
        assert np.array_equal(a["radius_xyz"], b["radius_xyz"])


def uncertain_config(family: str):
    cfg = load_yaml("configs/synthetic_sted/uncertain_ignore_stress_schema08.yaml")
    cfg["dataset_name"] = "uncertain_ignore_test"
    cfg["sample_count"] = 1
    cfg["geometry"]["image_shape"] = [256, 256]
    cfg["geometry"]["width_calibration"]["length_px"] = 160
    cfg["targets"]["distance_transform_enabled"] = False
    cfg["scene_morphology"].pop("review_modes", None)
    cfg["scene_morphology"]["mode"] = "mixed_morphology"
    for name, section in cfg["uncertain_ignore"]["families"].items():
        section["enabled"] = name == family
        section["count_range"] = [3, 3]
    return cfg


def test_faint_uncertain_fragments_have_no_skeleton_target():
    _, arrays, metadata = build_sample(uncertain_config("faint_fragments"), 0)
    assert arrays["real_compatible_uncertain_ignore_mask"].sum() > 0
    assert not np.any(
        arrays["real_compatible_skeleton_mask"]
        & arrays["real_compatible_uncertain_ignore_mask"]
    )
    assert validate_sample_arrays(metadata["sample_id"], arrays, metadata) == []


def test_defocused_uncertain_streaks_do_not_become_fibrous_tau():
    _, arrays, metadata = build_sample(uncertain_config("defocused_streaks"), 0)
    assert arrays["real_compatible_uncertain_ignore_mask"].sum() > 0
    assert not np.any(
        arrays["real_compatible_fibrous_mask"]
        & arrays["real_compatible_uncertain_ignore_mask"]
    )
    assert validate_sample_arrays(metadata["sample_id"], arrays, metadata) == []


def test_clump_boundary_uncertain_halo_does_not_leak_to_skeleton():
    cfg = uncertain_config("merged_boundary_filaments")
    cfg["scene_morphology"]["mode"] = "clump_dominated"
    cfg["clumps"]["count_range_by_mode"]["clump_dominated"] = [1, 1]
    _, arrays, metadata = build_sample(cfg, 0)
    assert arrays["real_compatible_clump_mask"].sum() > 0
    assert arrays["real_compatible_uncertain_ignore_mask"].sum() > 0
    assert not np.any(
        arrays["real_compatible_skeleton_mask"]
        & arrays["real_compatible_uncertain_ignore_mask"]
    )
    assert validate_sample_arrays(metadata["sample_id"], arrays, metadata) == []


def test_bundle_to_clump_transition_keeps_roles_disjoint():
    cfg = uncertain_config("bundle_clump_transition")
    cfg["scene_morphology"]["mode"] = "mixed_morphology"
    cfg["bundles"]["count_range_by_mode"]["mixed_morphology"] = [1, 1]
    cfg["clumps"]["count_range_by_mode"]["mixed_morphology"] = [1, 1]
    _, arrays, metadata = build_sample(cfg, 0)
    assert arrays["real_compatible_fibrous_mask"].sum() > 0
    assert arrays["real_compatible_uncertain_ignore_mask"].sum() > 0
    assert arrays["real_compatible_clump_mask"].sum() > 0
    assert not np.any(
        arrays["real_compatible_skeleton_mask"]
        & (
            arrays["real_compatible_uncertain_ignore_mask"]
            | arrays["real_compatible_clump_mask"]
        )
    )
    assert validate_sample_arrays(metadata["sample_id"], arrays, metadata) == []


def test_uncertain_ignore_generation_is_condition_blind():
    base = uncertain_config("weak_directional_texture")
    labelled = copy.deepcopy(base)
    labelled.update({"culture_id": "PN148", "disease": "AD", "tau_isoform": "4R", "div": 3})
    first = generate_morphology_geometry(base, 3)
    second = generate_morphology_geometry(labelled, 3)
    assert first["parameters"] == second["parameters"]
    for a, b in zip(first["fibers"], second["fibers"]):
        assert np.array_equal(a["points_xyz"], b["points_xyz"])


def test_uncertain_report_metrics_include_patch_and_intensity_diagnostics():
    shape = (256, 256)
    arrays = {
        "real_compatible_semantic_mask": np.zeros(shape, dtype=np.uint8),
        "real_compatible_fibrous_mask": np.zeros(shape, dtype=np.uint8),
        "real_compatible_clump_mask": np.zeros(shape, dtype=np.uint8),
        "real_compatible_uncertain_ignore_mask": np.zeros(shape, dtype=np.uint8),
        "real_compatible_skeleton_mask": np.zeros(shape, dtype=np.uint8),
        "render_uint8": np.full(shape, 5, dtype=np.uint8),
        "uncertain_family_code": np.asarray([1, 2], dtype=np.uint8),
        "uncertain_fragment_length_px": np.asarray([12.0, 24.0], dtype=np.float32),
    }
    arrays["real_compatible_uncertain_ignore_mask"][:128, :128] = 1
    arrays["real_compatible_semantic_mask"][:128, :128] = 255
    arrays["render_uint8"][:128, :128] = 40
    metrics = sample_metrics(arrays)
    assert metrics["has_uncertain_ignore"] is True
    assert metrics["patch_fraction_uncertain_gt_25"] == 0.25
    assert metrics["uncertain_ignore_region_intensity"]["p50"] == 40.0
    assert metrics["uncertain_fragment_length_mean"] == 18.0


def filamentous_fluff_config():
    cfg = load_yaml("configs/synthetic_sted/uncertain_ignore_stress_v2_schema08.yaml")
    cfg["dataset_name"] = "filamentous_fluff_test"
    cfg["sample_count"] = 1
    cfg["geometry"]["image_shape"] = [256, 256]
    cfg["geometry"]["width_calibration"]["length_px"] = 160
    cfg["targets"]["distance_transform_enabled"] = False
    cfg.pop("sample_variants", None)
    cfg["scene_morphology"].pop("review_modes", None)
    cfg["scene_morphology"]["mode"] = "mixed_morphology"
    for name, section in cfg["uncertain_ignore"]["families"].items():
        section["enabled"] = name == "filamentous_fluff"
        section["count_range"] = [3, 3] if name == "filamentous_fluff" else [0, 0]
    fluff = cfg["uncertain_ignore"]["families"]["filamentous_fluff"]
    fluff["radius_x_range_px"] = [80.0, 110.0]
    fluff["radius_y_range_px"] = [55.0, 90.0]
    fluff["fragment_count_range"] = [220, 280]
    fluff["radius_multiplier_range"] = [1.2, 2.2]
    fluff["support_radius_multiplier"] = 4.0
    return cfg


def test_filamentous_fluff_creates_uncertain_area_above_minimum():
    _, arrays, metadata = build_sample(filamentous_fluff_config(), 0)
    frac = arrays["real_compatible_uncertain_ignore_mask"].mean()
    assert frac > 0.02
    assert validate_sample_arrays(metadata["sample_id"], arrays, metadata) == []


def test_filamentous_fluff_can_create_uncertain_heavy_128px_crop():
    _, arrays, _ = build_sample(filamentous_fluff_config(), 0)
    metrics = sample_metrics(arrays)
    assert metrics["max_patch_uncertain_fraction"] > 0.25


def test_filamentous_fluff_has_no_target_leakage():
    _, arrays, _ = build_sample(filamentous_fluff_config(), 0)
    uncertain = arrays["real_compatible_uncertain_ignore_mask"]
    assert not np.any(arrays["real_compatible_fibrous_mask"] & uncertain)
    assert not np.any(arrays["real_compatible_clump_mask"] & uncertain)
    assert not np.any(arrays["real_compatible_skeleton_mask"] & uncertain)


def test_uncertain_ignore_stress_v2_config_validates():
    cfg = load_yaml("configs/synthetic_sted/uncertain_ignore_stress_v2_schema08.yaml")
    cfg["sample_count"] = 4
    cfg["geometry"]["image_shape"] = [256, 256]
    cfg["geometry"]["width_calibration"]["length_px"] = 160
    cfg["targets"]["distance_transform_enabled"] = False
    for index in range(4):
        sample_id, arrays, metadata = build_sample(cfg, index)
        assert validate_sample_arrays(sample_id, arrays, metadata) == []


def test_training_v2_uncertain_has_limited_uncertain_heavy_frequency():
    cfg = load_yaml("configs/synthetic_sted/training_v2_uncertain_schema08.yaml")
    seq = cfg["sample_variants"]["sequence"]
    assert "uncertain_heavy" in seq
    frac = seq.count("uncertain_heavy") / len(seq)
    assert 0.10 <= frac <= 0.20
    assert cfg["uncertain_ignore"]["families"]["filamentous_fluff"]["enabled"] is False
    assert cfg["sample_variants"]["definitions"]["uncertain_heavy"]["uncertain_ignore"]["families"]["filamentous_fluff"]["enabled"] is True


def test_filamentous_fluff_generation_is_condition_blind():
    base = filamentous_fluff_config()
    labelled = copy.deepcopy(base)
    labelled.update({"culture_id": "PN148", "disease": "AD", "tau_isoform": "4R", "div": 3})
    first = generate_morphology_geometry(base, 2)
    second = generate_morphology_geometry(labelled, 2)
    assert first["parameters"] == second["parameters"]
    for a, b in zip(first["fibers"], second["fibers"]):
        assert np.array_equal(a["points_xyz"], b["points_xyz"])


def test_sample_variants_cycle_without_changing_sample_namespace():
    cfg = load_yaml("configs/synthetic_sted/training_v1_schema08.yaml")
    cfg["sample_count"] = 4
    cfg["geometry"]["image_shape"] = [256, 256]
    cfg["geometry"]["width_calibration"]["length_px"] = 160
    cfg["targets"]["distance_transform_enabled"] = False
    sample_id, _, normal = build_sample(cfg, 0)
    _, arrays, stress = build_sample(cfg, 3)
    assert sample_id == "synthetic_sted_training_v1_schema08_0000"
    assert normal["generation_config"]["sample_variant"] == "normal"
    assert stress["generation_config"]["sample_variant"] == "clump_ignore_hard_negative"
    assert stress["rendering_report"]["scenario_category"] == "clump_ignore_stress"
    assert arrays["real_compatible_clump_mask"].sum() > 0


def test_tile_occupancy_reports_empty_tiles():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:16, :16] = 1
    occupancy = tile_occupancy(mask, 16)
    assert occupancy[0, 0] == 1
    assert np.count_nonzero(occupancy == 0) == 15
