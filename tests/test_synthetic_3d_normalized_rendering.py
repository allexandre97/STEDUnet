import copy

import numpy as np

from fibras.synthetic.geometry3d import generate_persistent_chain_geometry
from fibras.synthetic.rasterizer3d import measure_isolated_fiber_fwhm, rasterize_3d_sample


def cfg(interval=1.0, length=82.0, amplitude=120.0, angle=0.0):
    return {
        "dataset_name": "normalized_test",
        "generator_mode": "persistent_chain_3d",
        "sample_count": 8,
        "calibration_status": "procedural_unmatched",
        "geometry": {
            "image_shape": [128, 128],
            "volume_depth_px": 48.0,
            "focal_plane_z_px": 24.0,
            "base_seed": 42,
            "scenario": "straight_width_calibration",
            "arc_length_sampling_interval_px": interval,
            "width_calibration": {"angle_degrees": angle, "length_px": length},
            "fluorophore": {
                "base_amplitude_range": [amplitude, amplitude],
                "variation_amplitude": 0.0,
                "gap_probability": 0.0,
                "punctate_probability": 0.0,
                "radius_range_px": [0.7, 0.7],
            },
        },
        "targets": {"semantic_mask_source": "visible_signal_mask", "ignore_mask_rule": "none"},
        "optical_model": {
            "psf_mode": "core_plus_halo",
            "core_weight": 0.92,
            "halo_weight": 0.08,
            "core_sigma_xy_0_px": 2.123,
            "core_sigma_z_px": 12.0,
            "core_defocus_broadening": 0.03,
            "halo_sigma_xy_0_px": 6.0,
            "halo_sigma_z_px": 24.0,
            "halo_defocus_broadening": 0.06,
            "kernel_truncation_radius": 3.0,
            "focal_depth_range_px": 4.0,
            "visible_signal_threshold": 2.0,
        },
        "output_mapping": {"background_level": 0.0, "background_noise_std": 0.0, "foreground_scale": 1.0, "uint8_min": 0, "uint8_max": 255},
        "width_calibration": {"target_fwhm_px": 5.0, "min_allowed_fwhm_px": 4.5, "max_allowed_fwhm_px": 5.5},
    }


def render(config):
    geom = generate_persistent_chain_geometry(config["geometry"], 0)
    arrays, report = rasterize_3d_sample(geom, config["targets"], config["optical_model"], config["output_mapping"], 0)
    return arrays, report


def test_sampling_density_invariance_for_straight_fiber():
    totals = []
    fwhms = []
    peaks = []
    for interval in [2.0, 1.0, 0.5, 0.25]:
        c = cfg(interval=interval)
        arrays, _ = render(c)
        totals.append(float(arrays["total_clean_signal"].sum()))
        fwhms.append(measure_isolated_fiber_fwhm(c)["measured_fwhm_px"])
        peaks.append(float(arrays["total_clean_signal"].max()))
    assert (max(totals) - min(totals)) / np.mean(totals) < 0.02
    assert max(fwhms) - min(fwhms) < 0.2
    assert peaks[-1] <= peaks[0] * 1.05


def test_five_pixel_width_is_orientation_and_subpixel_stable():
    fwhms = [measure_isolated_fiber_fwhm(cfg(angle=angle))["measured_fwhm_px"] for angle in [0, 15, 30, 45, 60, 75, 90]]
    shifted = [
        measure_isolated_fiber_fwhm(cfg(), subpixel_shift=shift)["measured_fwhm_px"]
        for shift in [(0.25, 0.25), (0.5, 0.1), (0.1, 0.5)]
    ]
    assert all(4.5 <= value <= 5.5 for value in fwhms + shifted)
    assert max(fwhms) - min(fwhms) < 0.2
    assert max(shifted) - min(shifted) < 0.2


def test_signal_is_proportional_to_contour_length_and_density():
    base_arrays, base_report = render(cfg(length=60.0, amplitude=100.0))
    long_arrays, _ = render(cfg(length=100.0, amplitude=100.0))
    bright_arrays, _ = render(cfg(length=60.0, amplitude=200.0))
    assert np.isclose(long_arrays["total_clean_signal"].sum() / base_arrays["total_clean_signal"].sum(), 100.0 / 60.0, rtol=0.06)
    assert np.isclose(bright_arrays["total_clean_signal"].sum() / base_arrays["total_clean_signal"].sum(), 2.0, rtol=0.03)
    assert base_report["psf_normalization"] == "unit_integral"
    assert base_report["kernel_normalization_error"] < 1e-5


def test_core_plus_halo_components_reconstruct_total_signal():
    arrays, report = render(cfg())
    assert np.allclose(arrays["core_signal"] + arrays["halo_signal"], arrays["total_clean_signal"], atol=1e-4)
    assert report["core_integrated_signal"] > report["halo_integrated_signal"] > 0
    assert report["kernel_count"] > 0


def test_structural_targets_are_independent_of_amplitude_only_changes():
    c = cfg()
    geom = generate_persistent_chain_geometry(c["geometry"], 0)
    base, _ = rasterize_3d_sample(geom, c["targets"], c["optical_model"], c["output_mapping"], 0)
    bright_geom = copy.deepcopy(geom)
    for fiber in bright_geom["fibers"]:
        fiber["sample_amplitude"] *= 3.0
    bright, _ = rasterize_3d_sample(bright_geom, c["targets"], c["optical_model"], c["output_mapping"], 0)
    structural = ["projection_mask", "in_focus_mask", "centerline_mask", "endpoint_map", "junction_map", "crossing_map", "overlap_count"]
    for name in structural:
        assert np.array_equal(base[name], bright[name])
    assert np.array_equal(base["trace_points_xy"], bright["trace_points_xy"])
    assert np.array_equal(base["trace_point_offsets"], bright["trace_point_offsets"])


def test_annotation_ready_targets_are_present():
    c = cfg()
    geom = generate_persistent_chain_geometry(c["geometry"], 0)
    arrays, _ = rasterize_3d_sample(geom, c["targets"], c["optical_model"], c["output_mapping"], 0)
    for name in [
        "trace_points_xy",
        "trace_point_offsets",
        "trace_ids",
        "trace_status",
        "trace_source",
        "ignore_mask",
        "background_distance_to_semantic_foreground",
        "orientation_cos2theta",
        "orientation_sin2theta",
        "orientation_valid_mask",
    ]:
        assert name in arrays
    expected_xy = np.vstack([fiber["points_xyz"][:, :2] for fiber in geom["fibers"]]).astype(np.float32)
    assert np.array_equal(arrays["trace_points_xy"], expected_xy)
    assert arrays["ignore_mask"].shape == arrays["semantic_mask"].shape
