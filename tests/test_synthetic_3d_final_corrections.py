import copy

import numpy as np
import pytest

from fibras.synthetic import rasterizer3d
from fibras.synthetic.rasterizer3d import (
    background_distance_to_foreground,
    clear_width_calibration_cache,
    geometry_targets_3d,
    measure_isolated_fiber_fwhm,
    projected_crossings,
    projected_crossings_bruteforce,
    rasterize_3d_sample,
    validate_psf_config,
    width_calibration_cache_info,
)
from fibras.synthetic.schema import (
    DATASET_SCHEMA_VERSION_3D_NORMALIZED,
    REQUIRED_ARRAYS_3D_NORMALIZED,
)
from fibras.synthetic.storage import (
    build_sample,
    compare_complete_artifact,
    validate_sample_arrays,
)


def config():
    return {
        "dataset_name": "final_correction",
        "generator_mode": "persistent_chain_3d",
        "sample_count": 8,
        "calibration_status": "procedural_unmatched",
        "geometry": {
            "image_shape": [96, 96],
            "volume_depth_px": 40.0,
            "focal_plane_z_px": 20.0,
            "base_seed": 111,
            "scenario": "projected_depth_crossing",
            "arc_length_sampling_interval_px": 1.0,
            "width_calibration": {"angle_degrees": 0.0, "length_px": 64.0},
            "fluorophore": {
                "base_amplitude_range": [120, 120],
                "variation_amplitude": 0.0,
                "gap_probability": 0.0,
                "radius_range_px": [0.8, 0.8],
            },
        },
        "targets": {
            "semantic_mask_source": "visible_signal_mask",
            "ignore_mask_rule": "none",
            "orientation_consensus_threshold": 0.95,
            "orientation_field_enabled": True,
            "distance_transform_enabled": True,
        },
        "optical_model": {
            "psf_mode": "core_plus_halo",
            "core_weight": 0.92,
            "halo_weight": 0.08,
            "core_sigma_xy_0_px": 2.123,
            "core_sigma_z_px": 12.0,
            "core_defocus_broadening": 0.0,
            "halo_sigma_xy_0_px": 6.0,
            "halo_sigma_z_px": 24.0,
            "halo_defocus_broadening": 0.0,
            "kernel_truncation_radius": 3.0,
            "focal_depth_range_px": 4.0,
            "visible_signal_threshold": 2.0,
        },
        "output_mapping": {
            "background_level": 0.0,
            "foreground_scale": 1.0,
            "uint8_min": 0,
            "uint8_max": 255,
        },
        "width_calibration": {
            "target_fwhm_px": 5.0,
            "min_allowed_fwhm_px": 4.5,
            "max_allowed_fwhm_px": 5.5,
        },
    }


def test_scipy_edt_is_exact_and_reported():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1
    distance = background_distance_to_foreground(mask)
    assert distance.dtype == np.float32
    assert distance[2, 2] == 0
    assert distance[2, 3] == 1
    assert np.isclose(distance[0, 0], np.sqrt(8))
    _, arrays, metadata = build_sample(config(), 0)
    assert metadata["rendering_report"]["distance_transform_backend"] == (
        "scipy_ndimage_distance_transform_edt"
    )
    assert arrays["background_distance_to_semantic_foreground"].shape == (96, 96)


def test_missing_scipy_fails_only_when_distance_is_enabled(monkeypatch):
    monkeypatch.setattr(rasterizer3d, "scipy_ndimage", None)
    with pytest.raises(RuntimeError, match="SciPy is required"):
        background_distance_to_foreground(np.zeros((3, 3), dtype=np.uint8))
    cfg = config()
    cfg["targets"]["distance_transform_enabled"] = False
    _, arrays, metadata = build_sample(cfg, 0)
    assert "background_distance_to_semantic_foreground" not in arrays
    assert metadata["rendering_report"]["distance_transform_backend"] == "disabled"


def test_width_calibration_is_optical_only_and_cache_sensitive(monkeypatch):
    clear_width_calibration_cache()
    monkeypatch.setattr(
        rasterizer3d,
        "geometry_targets_3d",
        lambda *_args, **_kwargs: pytest.fail("structural targets were computed"),
    )
    first = measure_isolated_fiber_fwhm(config())
    info_after_first = width_calibration_cache_info()
    second = measure_isolated_fiber_fwhm(config())
    assert second == first
    assert width_calibration_cache_info().hits == info_after_first.hits + 1
    changed = config()
    changed["optical_model"]["core_sigma_xy_0_px"] = 2.3
    third = measure_isolated_fiber_fwhm(changed)
    assert width_calibration_cache_info().misses == info_after_first.misses + 1
    assert third["measured_fwhm_px"] != first["measured_fwhm_px"]
    assert 4.5 <= first["measured_fwhm_px"] <= 5.5
    assert first["optical_only_elapsed_seconds"] >= 0


@pytest.mark.parametrize(
    "weights,error",
    [
        ((-0.1, 1.1), "nonnegative"),
        ((np.nan, 0.0), "finite"),
        ((np.inf, 0.0), "finite"),
        ((0.0, 0.0), "positive"),
        ((0.8, 0.1), "sum to one"),
    ],
)
def test_invalid_core_halo_weights_fail(weights, error):
    optical = config()["optical_model"]
    optical["core_weight"], optical["halo_weight"] = weights
    with pytest.raises(ValueError, match=error):
        validate_psf_config(optical)


def test_valid_psf_weight_metadata_and_visibility_migration():
    cfg = config()
    _, _, metadata = build_sample(cfg, 0)
    report = metadata["rendering_report"]
    assert report["psf_component_weight_convention"] == "weights_sum_to_one"
    assert report["psf_component_weight_sum"] == 1
    cfg["targets"]["visible_signal_threshold"] = 2.0
    build_sample(cfg, 0)
    cfg["targets"]["visible_signal_threshold"] = 3.0
    with pytest.raises(ValueError, match="deprecated"):
        build_sample(cfg, 0)


def test_self_crossing_orientation_is_invalid_and_order_independent():
    points = np.asarray(
        [
            [10, 30, 20],
            [20, 30, 20],
            [30, 30, 20],
            [40, 30, 20],
            [45, 35, 20],
            [40, 20, 20],
            [30, 20, 20],
            [30, 30, 20],
            [30, 40, 20],
            [30, 50, 20],
        ],
        dtype=np.float32,
    )
    geometry = one_fiber_geometry(points)
    arrays, _ = geometry_targets_3d(
        geometry, config()["targets"], focal_depth=4.0
    )
    assert arrays["orientation_valid_mask"][30, 30] == 0
    reversed_geometry = one_fiber_geometry(points[::-1].copy())
    reversed_arrays, _ = geometry_targets_3d(
        reversed_geometry, config()["targets"], focal_depth=4.0
    )
    assert np.array_equal(
        arrays["orientation_valid_mask"],
        reversed_arrays["orientation_valid_mask"],
    )
    assert np.allclose(
        arrays["orientation_cos2theta"],
        reversed_arrays["orientation_cos2theta"],
    )


def test_orientation_consensus_handles_straight_parallel_antiparallel_and_loop():
    straight = np.asarray(
        [[10, 30, 20], [20, 30, 20], [30, 30, 20], [40, 30, 20]],
        dtype=np.float32,
    )
    arrays, _ = geometry_targets_3d(
        one_fiber_geometry(straight), config()["targets"], focal_depth=4.0
    )
    assert arrays["orientation_valid_mask"][30, 30] == 1
    for second in [straight.copy(), straight[::-1].copy()]:
        geometry = multi_fiber_geometry([straight, second])
        arrays, _ = geometry_targets_3d(
            geometry, config()["targets"], focal_depth=4.0
        )
        assert arrays["orientation_valid_mask"][30, 30] == 1
    angles = np.linspace(0, 2 * np.pi, 17)
    loop = np.column_stack(
        [
            30 + 0.4 * np.cos(angles),
            30 + 0.4 * np.sin(angles),
            np.full(angles.shape, 20),
        ]
    ).astype(np.float32)
    loop_arrays, _ = geometry_targets_3d(
        one_fiber_geometry(loop), config()["targets"], focal_depth=4.0
    )
    assert loop_arrays["orientation_valid_mask"][30, 30] == 0


def test_crossing_identity_deduplication_preserves_nearby_pairs():
    geometry = crossing_geometry()
    fast = projected_crossings(geometry)
    brute = projected_crossings_bruteforce(geometry)
    fast_signature = crossing_signature(fast)
    brute_signature = crossing_signature(brute)
    assert fast_signature == brute_signature
    pairs = [tuple(item["fiber_ids"]) for item in fast]
    assert (1, 2) in pairs
    assert (3, 4) in pairs
    assert len([pair for pair in pairs if pair == (1, 2)]) == 1


def test_complete_deterministic_comparison_checks_optional_arrays_and_metadata():
    sample_id, arrays, metadata = build_sample(config(), 0)
    _, rebuilt, rebuilt_metadata = build_sample(config(), 0)
    assert compare_complete_artifact(
        sample_id, arrays, metadata, rebuilt, rebuilt_metadata
    ) == []
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["orientation_cos2theta"][0, 0] = 1
    assert any(
        "orientation_cos2theta" in error
        for error in compare_complete_artifact(
            sample_id, changed, metadata, rebuilt, rebuilt_metadata
        )
    )
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["unexpected"] = np.zeros(1, dtype=np.uint8)
    assert any(
        "array-name mismatch" in error
        for error in compare_complete_artifact(
            sample_id, changed, metadata, rebuilt, rebuilt_metadata
        )
    )
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["ignore_mask"] = changed["ignore_mask"].astype(np.int16)
    assert any(
        "dtype mismatch" in error
        for error in compare_complete_artifact(
            sample_id, changed, metadata, rebuilt, rebuilt_metadata
        )
    )
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["ignore_mask"] = changed["ignore_mask"][:-1]
    assert any(
        "shape mismatch" in error
        for error in compare_complete_artifact(
            sample_id, changed, metadata, rebuilt, rebuilt_metadata
        )
    )
    bad_metadata = copy.deepcopy(metadata)
    bad_metadata["target_available"]["orientation"] = False
    assert any(
        "contradicts" in error
        for error in validate_sample_arrays(sample_id, arrays, bad_metadata)
    )


def test_schema_03_receives_full_normalized_validation():
    sample_id, current, metadata = build_sample(config(), 0)
    arrays = legacy_03_arrays(current)
    legacy = copy.deepcopy(metadata)
    legacy["dataset_schema_version"] = DATASET_SCHEMA_VERSION_3D_NORMALIZED
    legacy["generator_version"] = "fibras_persistent_chain_3d_0.3.0"
    legacy["array_names"] = sorted(arrays)
    legacy["dtypes"] = {name: str(array.dtype) for name, array in arrays.items()}
    legacy["shapes"] = {name: list(array.shape) for name, array in arrays.items()}
    assert validate_sample_arrays(sample_id, arrays, legacy) == []
    malformed = {name: value.copy() for name, value in arrays.items()}
    malformed["out_of_focus_signal"][0, 0] += 10
    assert any(
        "signal mismatch" in error
        for error in validate_sample_arrays(sample_id, malformed, legacy)
    )
    malformed = {name: value.copy() for name, value in arrays.items()}
    malformed["trace_point_offsets"][-1] -= 1
    assert any(
        "trace" in error
        for error in validate_sample_arrays(sample_id, malformed, legacy)
    )
    missing = dict(arrays)
    missing.pop("distance_transform")
    assert any(
        "missing arrays" in error
        for error in validate_sample_arrays(sample_id, missing, legacy)
    )


def one_fiber_geometry(points):
    return {
        "image_shape": (64, 64),
        "volume_depth_px": 40.0,
        "focal_plane_z_px": 20.0,
        "fibers": [
            {
                "fiber_id": 1,
                "points_xyz": points,
                "points_xy": points[:, :2],
                "raw_vertices_xyz": points,
                "sample_amplitude": np.ones(len(points), dtype=np.float32),
                "sample_radius": np.full(len(points), 0.8, dtype=np.float32),
            }
        ],
        "nodes": [
            {"xyz": points[0], "type": "endpoint"},
            {"xyz": points[-1], "type": "endpoint"},
        ],
        "edges": [
            {
                "fiber_id": 1,
                "node_indices": (0, 1),
                "truncated_start": False,
                "truncated_end": False,
            }
        ],
        "parameters": {"scenario": "self_crossing_fixture"},
    }


def crossing_geometry():
    lines = [
        np.asarray([[10, 30, 20], [30, 30, 20], [50, 30, 20]], np.float32),
        np.asarray([[30, 10, 20], [30, 30, 20], [30, 50, 20]], np.float32),
        np.asarray([[10, 31, 22], [31, 31, 22], [50, 31, 22]], np.float32),
        np.asarray([[31, 10, 22], [31, 31, 22], [31, 50, 22]], np.float32),
    ]
    fibers = []
    nodes = []
    edges = []
    for index, points in enumerate(lines, start=1):
        node_start = len(nodes)
        nodes.extend(
            [
                {"xyz": points[0], "type": "endpoint"},
                {"xyz": points[-1], "type": "endpoint"},
            ]
        )
        fibers.append(
            {
                "fiber_id": index,
                "points_xyz": points,
                "points_xy": points[:, :2],
                "raw_vertices_xyz": points,
                "sample_amplitude": np.ones(len(points), dtype=np.float32),
                "sample_radius": np.ones(len(points), dtype=np.float32),
            }
        )
        edges.append(
            {
                "fiber_id": index,
                "node_indices": (node_start, node_start + 1),
                "truncated_start": False,
                "truncated_end": False,
            }
        )
    return {
        "image_shape": (64, 64),
        "volume_depth_px": 40.0,
        "focal_plane_z_px": 20.0,
        "fibers": fibers,
        "nodes": nodes,
        "edges": edges,
        "parameters": {"scenario": "nearby_crossings"},
    }


def multi_fiber_geometry(lines):
    fibers = []
    nodes = []
    edges = []
    for index, points in enumerate(lines, start=1):
        start = len(nodes)
        nodes.extend(
            [
                {"xyz": points[0], "type": "endpoint"},
                {"xyz": points[-1], "type": "endpoint"},
            ]
        )
        fibers.append(
            {
                "fiber_id": index,
                "points_xyz": points,
                "points_xy": points[:, :2],
                "raw_vertices_xyz": points,
                "sample_amplitude": np.ones(len(points), dtype=np.float32),
                "sample_radius": np.ones(len(points), dtype=np.float32),
            }
        )
        edges.append(
            {
                "fiber_id": index,
                "node_indices": (start, start + 1),
                "truncated_start": False,
                "truncated_end": False,
            }
        )
    return {
        "image_shape": (64, 64),
        "volume_depth_px": 40.0,
        "focal_plane_z_px": 20.0,
        "fibers": fibers,
        "nodes": nodes,
        "edges": edges,
        "parameters": {"scenario": "orientation_fixture"},
    }


def crossing_signature(crossings):
    return [
        (
            tuple(map(int, item["fiber_ids"])),
            tuple(np.round(item["xy"], 5)),
            tuple(map(int, item["segment_indices"])),
        )
        for item in crossings
    ]


def legacy_03_arrays(current):
    arrays = {
        name: value.copy()
        for name, value in current.items()
        if name in REQUIRED_ARRAYS_3D_NORMALIZED
    }
    arrays["source_float"] = current["geometric_support_preview"].copy()
    arrays["distance_transform"] = current[
        "background_distance_to_semantic_foreground"
    ].copy()
    arrays["orientation_cos2"] = current["orientation_cos2theta"].copy()
    arrays["orientation_sin2"] = current["orientation_sin2theta"].copy()
    return arrays
