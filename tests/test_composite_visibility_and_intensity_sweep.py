import copy

import numpy as np
import pytest
from PIL import Image

from fibras.calibration.intensity_sweep import (
    compose_scaled_foreground,
    effective_foreground_scale,
    require_realism_category,
    summarize_settings,
)
from fibras.calibration.visibility import (
    RAW_DISPLAY_RANGE,
    composite_visibility_diagnostics,
    global_uint8_percentiles,
    robust_display,
    scale_for_display,
)


def test_raw_panel_uses_fixed_zero_to_255_scaling():
    image = np.asarray([[0, 64, 128, 255]], dtype=np.uint8)
    assert RAW_DISPLAY_RANGE == (0.0, 255.0)
    assert np.array_equal(scale_for_display(image, *RAW_DISPLAY_RANGE), image)


def test_shared_limits_are_identical_and_per_image_stretch_is_display_only():
    first = np.asarray([[0, 10], [20, 30]], dtype=np.uint8)
    second = np.asarray([[5, 50], [100, 200]], dtype=np.uint8)
    first_before = first.copy()
    second_before = second.copy()
    limits = (1.0, 40.0)
    assert scale_for_display(first, *limits).shape == first.shape
    assert scale_for_display(second, *limits).shape == second.shape
    robust_display(first, (0.5, 99.5))
    robust_display(second, (0.5, 99.5))
    assert np.array_equal(first, first_before)
    assert np.array_equal(second, second_before)


def test_foreground_display_handles_zero_and_weak_signal():
    zero, zero_limits = robust_display(np.zeros((4, 4), dtype=np.float32))
    weak, weak_limits = robust_display(
        np.linspace(0, 1e-4, 16, dtype=np.float32).reshape(4, 4)
    )
    assert not zero.any()
    assert zero_limits == (0.0, 0.0)
    assert weak.max() == 255
    assert weak_limits[1] > weak_limits[0]


def test_global_shared_percentiles_use_all_pixels(tmp_path):
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    Image.fromarray(np.asarray([[0, 0], [10, 10]], dtype=np.uint8)).save(first)
    Image.fromarray(np.asarray([[20, 20], [100, 100]], dtype=np.uint8)).save(second)
    assert global_uint8_percentiles([first, second], (0, 100)) == (0, 100)


def test_sweep_reuses_blank_and_signal_and_is_deterministic():
    blank = np.full((4, 4), 5, dtype=np.float32)
    signal = np.arange(16, dtype=np.float32).reshape(4, 4) / 10
    blank_before = blank.copy()
    signal_before = signal.copy()
    config = {"uint8_min": 0, "uint8_max": 255}
    first = compose_scaled_foreground(blank, signal, 2.0, config)
    second = compose_scaled_foreground(blank, signal, 2.0, config)
    assert all(np.array_equal(a, b) for a, b in zip(first[:2], second[:2]))
    assert first[2] == second[2]
    assert np.array_equal(blank, blank_before)
    assert np.array_equal(signal, signal_before)
    assert np.array_equal(first[0], signal * 2)
    assert effective_foreground_scale(2, 4) == 8


def test_clipping_and_saturation_are_reported():
    blank = np.full((2, 2), 250, dtype=np.float32)
    signal = np.full((2, 2), 20, dtype=np.float32)
    _, composite, report = compose_scaled_foreground(
        blank, signal, 1.0, {"uint8_min": 0, "uint8_max": 255}
    )
    assert np.all(composite == 255)
    assert report["clipping_fraction"] == 1
    assert report["saturation_fraction"] == 1


def test_visibility_diagnostics_include_required_provenance_and_metrics():
    arrays = {
        "blank_float": np.full((2, 2), 4, dtype=np.float32),
        "render_uint8": np.asarray([[4, 5], [6, 7]], dtype=np.uint8),
        "synthetic_signal_float": np.asarray(
            [[0, 1], [2, 3]], dtype=np.float32
        ),
    }
    metadata = {
        "sample_id": "composite_0000",
        "parent_synthetic_sample_id": "parent_0000",
        "source_blank_provenance": {"blank_stable_image_id": "blank_0000"},
        "rendering_report": {"visible_signal_threshold": 1.5},
        "clipping_fraction": 0.0,
        "saturation_fraction": 0.0,
    }
    diagnostics = composite_visibility_diagnostics(arrays, metadata)
    assert diagnostics["sample_id"] == "composite_0000"
    assert diagnostics["parent_synthetic_sample_id"] == "parent_0000"
    assert diagnostics["source_blank_id"] == "blank_0000"
    assert float(diagnostics["synthetic_foreground_integrated_signal"]) == 6
    assert float(diagnostics["foreground_occupancy"]) == 0.5


def test_qa_categories_are_excluded_from_sweep():
    require_realism_category("sample", "realism_calibration")
    with pytest.raises(ValueError, match="QA category"):
        require_realism_category("fixture", "structural_qa")


def test_setting_summary_is_deterministic_and_separately_ranked():
    base = {
        "fluorophore_density_multiplier": "1",
        "p50": "4",
        "p95": "10",
        "p99": "20",
        "std": "3",
        "local_variance_p50": "2",
        "foreground_occupancy_proxy": "0.1",
        "ridge_response_p50": "1",
        "ridge_response_p95": "4",
        "normalized_radial_power_low_band_fraction": "0.5",
        "normalized_radial_power_high_band_fraction": "0.1",
        "clipping_fraction": "0",
        "saturation_fraction": "0",
        "delta_p50": "0",
        "delta_p95": "1",
        "delta_p99": "2",
        "delta_mean": "1",
        "delta_variance": "2",
        "delta_local_variance": "1",
        "added_integrated_signal": "100",
    }
    rows = [
        {
            **copy.deepcopy(base),
            "setting_id": "scale_1",
            "compositor_foreground_scale": "1",
            "effective_foreground_scale": "1",
        },
        {
            **copy.deepcopy(base),
            "setting_id": "scale_2",
            "compositor_foreground_scale": "2",
            "effective_foreground_scale": "2",
            "p99": "25",
        },
    ]
    reference = {
        "p50": 4,
        "p95": 10,
        "p99": 20,
        "std": 3,
        "local_variance_p50": 2,
        "foreground_occupancy_proxy": 0.1,
        "ridge_response_p50": 1,
        "ridge_response_p95": 4,
        "normalized_radial_power_low_band_fraction": 0.5,
        "normalized_radial_power_high_band_fraction": 0.1,
    }
    first = summarize_settings(rows, reference, {})
    second = summarize_settings(rows, reference, {})
    assert first == second
    assert first[0]["intensity_rank"] == "1"
    assert first[0]["calibration_status"] == "exploratory_unpartitioned"
