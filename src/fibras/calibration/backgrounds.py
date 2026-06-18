"""Background decomposition for expert-validated blanks."""

from __future__ import annotations

from typing import Any

import numpy as np

from fibras.synthetic.rendering import gaussian_blur

from .spectra import radial_power_spectrum


def decompose_background(image: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    arr = image.astype(np.float32)
    low = gaussian_blur(arr, sigma_px)
    residual = arr - low
    return low.astype(np.float32), residual.astype(np.float32)


def decomposition_stats(image: np.ndarray, sigma_values: list[float], spectrum_size: int = 256, spectrum_bins: int = 32) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sigma in sigma_values:
        low, residual = decompose_background(image, sigma)
        residual_power = radial_power_spectrum(residual, spectrum_size, spectrum_bins)
        rows.append(
            {
                "decomposition_method": "gaussian_lowpass",
                "sigma_px": f"{sigma:.6g}",
                "residual_variance": f"{float(np.var(residual)):.6g}",
                "residual_power_tail_median": f"{float(np.median(residual_power[len(residual_power)//2:])):.6g}",
                "low_frequency_gradient_p95_minus_p5": f"{float(np.percentile(low, 95) - np.percentile(low, 5)):.6g}",
                "physical_uniqueness_warning": "not physically unique; deterministic analysis decomposition only",
            }
        )
    return rows

