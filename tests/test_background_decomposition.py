import numpy as np

from fibras.calibration.backgrounds import decompose_background, decomposition_stats


def test_background_decomposition_reconstructs_original():
    image = np.tile(np.linspace(0, 20, 64, dtype=np.float32), (64, 1))
    low, residual = decompose_background(image, 4.0)
    assert low.shape == image.shape
    assert residual.shape == image.shape
    assert np.allclose(low + residual, image, atol=1e-5)
    rows = decomposition_stats(image, [2.0, 4.0], 64, 8)
    assert len(rows) == 2
    assert rows[0]["physical_uniqueness_warning"].startswith("not physically unique")

