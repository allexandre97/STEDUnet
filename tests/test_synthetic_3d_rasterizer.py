import numpy as np

from fibras.synthetic.geometry3d import generate_persistent_chain_geometry
from fibras.synthetic.rasterizer3d import (
    measure_isolated_fiber_fwhm,
    psf_components,
    rasterize_3d_sample,
    splat_points,
)


def config():
    return {
        "dataset_name": "test_3d",
        "generator_mode": "persistent_chain_3d",
        "sample_count": 8,
        "calibration_status": "procedural_unmatched",
        "geometry": {
            "image_shape": [128, 128],
            "volume_depth_px": 48.0,
            "focal_plane_z_px": 24.0,
            "base_seed": 42,
            "scenario": "straight_width_calibration",
            "fiber_count_range": [1, 1],
            "contour_length_range_px": [80.0, 80.0],
            "step_length_px": 4.0,
            "persistence_length_range_px": [100.0, 100.0],
            "arc_length_sampling_interval_px": 1.0,
            "width_calibration": {"angle_degrees": 0.0, "length_px": 82.0},
            "fluorophore": {"base_amplitude_range": [100, 100], "radius_range_px": [0.7, 0.7], "gap_probability": 0.0},
        },
        "targets": {"semantic_mask_source": "visible_signal_mask", "near_coplanar_depth_px": 6.0},
        "optical_model": {
            "psf_mode": "single_gaussian",
            "sigma_xy_0_px": 2.123,
            "sigma_z_px": 12.0,
            "defocus_broadening": 0.08,
            "kernel_truncation_radius": 3.0,
            "focal_depth_range_px": 4.0,
            "visible_signal_threshold": 2.0,
        },
        "output_mapping": {"background_level": 4.0, "background_noise_std": 0.0, "uint8_min": 0, "uint8_max": 255},
        "width_calibration": {"target_fwhm_px": 5.0, "min_allowed_fwhm_px": 4.5, "max_allowed_fwhm_px": 5.5},
    }


def test_depth_dependent_splatting_brightens_and_sharpens_in_focus_points():
    cfg = config()["optical_model"]
    focal = 24.0
    pts = np.asarray([[40.0, 40.0, focal]], dtype=np.float32)
    far = np.asarray([[40.0, 40.0, focal + 24.0]], dtype=np.float32)
    amp = np.asarray([100.0], dtype=np.float32)
    near_in, _, *_ = splat_points(pts, amp, None, focal, cfg, (96, 96))
    _, far_out, *_ = splat_points(far, amp, None, focal, cfg, (96, 96))
    assert float(near_in.max()) > float(far_out.max())
    assert np.count_nonzero(far_out > far_out.max() * 0.2) > np.count_nonzero(near_in > near_in.max() * 0.2)


def test_axial_attenuation_is_symmetric_and_kernel_support_is_finite():
    cfg = config()["optical_model"]
    plus = psf_components(cfg, 9.0)[0]
    minus = psf_components(cfg, -9.0)[0]
    assert plus == minus
    pts = np.asarray([[32.0, 32.0, 24.0]], dtype=np.float32)
    amp = np.asarray([100.0], dtype=np.float32)
    rendered, _, *_ = splat_points(pts, amp, None, 24.0, cfg | {"kernel_truncation_radius": 2.0}, (64, 64))
    y, x = np.nonzero(rendered)
    dist = np.sqrt((x + 0.5 - 32.0) ** 2 + (y + 0.5 - 32.0) ** 2)
    assert float(dist.max()) <= 2.0 * cfg["sigma_xy_0_px"] + 1.0


def test_core_halo_mode_is_finite_and_reproducible():
    cfg = config()
    cfg["optical_model"].update({"psf_mode": "core_plus_halo", "core_weight": 0.9, "halo_weight": 0.1, "halo_sigma_xy_0_px": 5.0})
    geom = generate_persistent_chain_geometry(cfg["geometry"], 0)
    a, _ = rasterize_3d_sample(geom, cfg["targets"], cfg["optical_model"], cfg["output_mapping"], 0)
    b, _ = rasterize_3d_sample(geom, cfg["targets"], cfg["optical_model"], cfg["output_mapping"], 0)
    assert np.isfinite(a["total_clean_signal"]).all()
    assert np.array_equal(a["total_clean_signal"], b["total_clean_signal"])


def test_width_calibration_is_near_five_pixels_and_stable():
    cfg = config()
    base = measure_isolated_fiber_fwhm(cfg)
    rotated = measure_isolated_fiber_fwhm(cfg, angle_degrees=35.0)
    shifted = measure_isolated_fiber_fwhm(cfg, subpixel_shift=(0.37, 0.21))
    assert 4.5 <= base["measured_fwhm_px"] <= 5.5
    assert abs(base["measured_fwhm_px"] - rotated["measured_fwhm_px"]) < 0.5
    assert abs(base["measured_fwhm_px"] - shifted["measured_fwhm_px"]) < 0.5
