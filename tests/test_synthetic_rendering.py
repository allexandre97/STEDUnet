import numpy as np

from fibras.synthetic.rendering import render_image


def test_rendering_is_deterministic_and_reports_clipping():
    source = np.zeros((32, 32), dtype=np.float32)
    source[10:20, 10:20] = 300
    config = {"background_level": 4, "background_noise_std": 0, "psf_sigma_px": 1.0, "uint8_min": 0, "uint8_max": 255}
    a_float, a_uint8, a_stats = render_image(source, config, 5)
    b_float, b_uint8, b_stats = render_image(source, config, 5)
    assert np.array_equal(a_float, b_float)
    assert np.array_equal(a_uint8, b_uint8)
    assert a_stats == b_stats
    assert a_stats["saturation_count"] > 0

