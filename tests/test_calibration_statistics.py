import numpy as np

from fibras.calibration.statistics import block_stats, summarize_array
from fibras.calibration.spectra import autocorrelation_radial, radial_power_spectrum
from fibras.calibration.proxies import proxy_measurements


def test_direct_statistics_and_spectra_are_finite():
    image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    stats = summarize_array(image, {"sample_id": "fixture", "source_kind": "fixture"}, {"local_block_size_px": 16, "spectrum_size_px": 64})
    assert float(stats["p99"]) > float(stats["p50"])
    assert float(stats["local_variance_p50"]) > 0
    assert np.all(np.isfinite(radial_power_spectrum(image, size=64, bins=8)))
    assert np.all(np.isfinite(autocorrelation_radial(image, size=64, bins=8)))


def test_proxy_measurements_are_labelled_as_not_ground_truth():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[20:24, 8:56] = 80
    proxy = proxy_measurements(image, {"threshold_mad_k": 4, "ridge_percentile": 90})
    assert proxy["proxy_warning"].startswith("proxy estimate only")
    assert float(proxy["foreground_occupancy_proxy"]) > 0
    assert proxy["proxy_estimator_name"]

