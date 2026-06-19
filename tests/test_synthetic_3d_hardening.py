import copy

import numpy as np

from fibras.synthetic.geometry3d import generate_persistent_chain_geometry
from fibras.synthetic.rasterizer3d import (
    background_distance_to_foreground,
    projected_crossings,
    projected_crossings_bruteforce,
    rasterize_3d_sample,
    splat_points,
)


def base_config():
    return {
        "geometry": {
            "image_shape": [96, 96],
            "volume_depth_px": 40.0,
            "focal_plane_z_px": 20.0,
            "base_seed": 111,
            "scenario": "projected_depth_crossing",
            "arc_length_sampling_interval_px": 1.0,
            "width_calibration": {"length_px": 64.0},
            "fluorophore": {"base_amplitude_range": [120, 120], "variation_amplitude": 0.0, "gap_probability": 0.0, "radius_range_px": [0.8, 0.8]},
        },
        "targets": {"semantic_mask_source": "visible_signal_mask", "ignore_mask_rule": "none", "orientation_field_enabled": True, "distance_transform_enabled": True},
        "optical_model": {
            "psf_mode": "single_gaussian",
            "sigma_xy_0_px": 2.123,
            "sigma_z_px": 12.0,
            "defocus_broadening": 0.0,
            "kernel_truncation_radius": 3.0,
            "focal_depth_range_px": 4.0,
            "visible_signal_threshold": 2.0,
            "contributing_signal_threshold": 1e-6,
        },
        "output_mapping": {"background_level": 0.0, "foreground_scale": 1.0, "uint8_min": 0, "uint8_max": 255},
    }


def test_psf_boundary_clipping_loses_energy_without_peak_boost():
    cfg = base_config()["optical_model"]
    amp = np.asarray([100.0], dtype=np.float32)
    ds = np.asarray([1.0], dtype=np.float32)
    center, *_rest, center_stats = splat_points(np.asarray([[32.0, 32.0, 20.0]], dtype=np.float32), amp, ds, 20.0, cfg, (64, 64))
    edge, *_rest, edge_stats = splat_points(np.asarray([[0.0, 32.0, 20.0]], dtype=np.float32), amp, ds, 20.0, cfg, (64, 64))
    corner, *_rest, corner_stats = splat_points(np.asarray([[0.0, 0.0, 20.0]], dtype=np.float32), amp, ds, 20.0, cfg, (64, 64))
    assert np.isclose(center.sum(), 100.0, rtol=1e-6)
    assert edge.sum() < center.sum()
    assert corner.sum() < edge.sum()
    assert edge.max() <= center.max() * 1.01
    assert corner.max() <= center.max() * 1.01
    assert center_stats["kernel_boundary_clipped_count"] == 0
    assert edge_stats["kernel_boundary_clipped_count"] == 1
    assert corner_stats["kernel_out_of_frame_fraction_max"] > edge_stats["kernel_out_of_frame_fraction_max"] > 0


def test_optimized_crossings_match_bruteforce():
    cfg = base_config()
    geom = generate_persistent_chain_geometry(cfg["geometry"], 0)
    fast = projected_crossings(geom)
    brute = projected_crossings_bruteforce(geom)
    assert len(fast) == len(brute)
    assert np.allclose(np.asarray([c[0] for c in fast]), np.asarray([c[0] for c in brute]))


def test_distance_target_semantics_are_background_to_semantic_foreground():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1
    dist = background_distance_to_foreground(mask)
    assert dist[2, 2] == 0
    assert np.isclose(dist[2, 3], 1.0)
    assert np.isclose(dist[0, 2], 2.0)


def test_optional_target_flags_control_arrays_and_availability():
    cfg = base_config()
    cfg["targets"]["orientation_field_enabled"] = False
    cfg["targets"]["distance_transform_enabled"] = False
    geom = generate_persistent_chain_geometry(cfg["geometry"], 0)
    arrays, report = rasterize_3d_sample(geom, cfg["targets"], cfg["optical_model"], cfg["output_mapping"], 0)
    assert report["target_available"]["orientation"] is False
    assert report["target_available"]["background_distance_to_semantic_foreground"] is False
    assert "orientation_valid_mask" not in arrays
    assert "background_distance_to_semantic_foreground" not in arrays


def test_crossing_orientation_is_invalid_and_order_independent():
    cfg = base_config()
    geom = generate_persistent_chain_geometry(cfg["geometry"], 0)
    arrays, _ = rasterize_3d_sample(geom, cfg["targets"], cfg["optical_model"], cfg["output_mapping"], 0)
    hit = arrays["projected_crossing_points_xy"][0]
    y, x = int(np.floor(hit[1])), int(np.floor(hit[0]))
    assert arrays["orientation_valid_mask"][y, x] == 0
    reversed_geom = copy.deepcopy(geom)
    reversed_geom["fibers"] = list(reversed(reversed_geom["fibers"]))
    reversed_arrays, _ = rasterize_3d_sample(reversed_geom, cfg["targets"], cfg["optical_model"], cfg["output_mapping"], 0)
    assert np.array_equal(arrays["orientation_valid_mask"], reversed_arrays["orientation_valid_mask"])
    assert np.allclose(arrays["orientation_cos2theta"], reversed_arrays["orientation_cos2theta"])


def test_line_source_integral_matches_emitted_signal():
    cfg = base_config()
    geom = generate_persistent_chain_geometry(cfg["geometry"], 0)
    arrays, report = rasterize_3d_sample(geom, cfg["targets"], cfg["optical_model"], cfg["output_mapping"], 0)
    assert "source_float" not in arrays
    assert np.isclose(arrays["line_source_float"].sum(), report["total_emitted_source_signal"], rtol=1e-5)


def test_combined_only_visible_pixels_are_explicitly_classified():
    cfg = base_config()
    cfg["optical_model"]["visible_signal_threshold"] = 10.0
    geom = overlapping_two_fiber_geometry()
    arrays, _ = rasterize_3d_sample(geom, cfg["targets"], cfg["optical_model"], cfg["output_mapping"], 0)
    assert arrays["combined_only_visible_mask"].sum() > 0
    classified = ((arrays["visible_overlap_count"] > 0) | (arrays["combined_only_visible_mask"] > 0)).astype(np.uint8)
    assert np.array_equal(classified, arrays["visible_signal_mask"])


def overlapping_two_fiber_geometry():
    points = np.asarray([[24.0, 48.0, 20.0], [48.0, 48.0, 20.0], [72.0, 48.0, 20.0]], dtype=np.float32)
    fibers = []
    for fid in [1, 2]:
        fibers.append(
            {
                "fiber_id": fid,
                "points_xyz": points.copy(),
                "points_xy": points[:, :2].copy(),
                "raw_vertices_xyz": points.copy(),
                "sample_amplitude": np.full(points.shape[0], 7.0, dtype=np.float32),
                "sample_radius": np.full(points.shape[0], 0.8, dtype=np.float32),
            }
        )
    return {
        "image_shape": (96, 96),
        "volume_depth_px": 40.0,
        "focal_plane_z_px": 20.0,
        "fibers": fibers,
        "nodes": [
            {"xyz": points[0], "type": "endpoint"},
            {"xyz": points[-1], "type": "endpoint"},
            {"xyz": points[0], "type": "endpoint"},
            {"xyz": points[-1], "type": "endpoint"},
        ],
        "edges": [
            {"fiber_id": 1, "node_indices": (0, 1), "truncated_start": False, "truncated_end": False},
            {"fiber_id": 2, "node_indices": (2, 3), "truncated_start": False, "truncated_end": False},
        ],
        "parameters": {"scenario": "combined_only_fixture"},
    }
