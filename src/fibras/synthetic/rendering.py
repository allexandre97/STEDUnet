"""Simple artificial-background rendering for MVP synthetic STED samples."""

from __future__ import annotations

from typing import Any

import numpy as np


def render_image(source_float: np.ndarray, config: dict[str, Any], rendering_seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    rng = np.random.default_rng(rendering_seed)
    background = float(config.get("background_level", 4.0))
    noise_std = float(config.get("background_noise_std", 0.0))
    sigma = float(config.get("psf_sigma_px", 1.0))
    render = source_float.astype(np.float32)
    if sigma > 0:
        render = gaussian_blur(render, sigma)
    render = render + background
    if noise_std > 0:
        render = render + rng.normal(0, noise_std, render.shape).astype(np.float32)
    render = np.maximum(render, 0).astype(np.float32)
    uint8, stats = map_to_uint8(render, config)
    return render, uint8, stats


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3 * sigma)))
    radius = min(radius, max(1, min(image.shape) // 2 - 1))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    temp = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=image)
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=temp)
    return out.astype(np.float32)


def map_to_uint8(image: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, int | float]]:
    lo = float(config.get("uint8_min", 0.0))
    hi = float(config.get("uint8_max", 255.0))
    if hi <= lo:
        raise ValueError("uint8_max must be greater than uint8_min")
    below = int(np.sum(image < lo))
    above = int(np.sum(image > hi))
    scaled = np.clip((image - lo) / (hi - lo), 0, 1)
    uint8 = np.rint(scaled * 255).astype(np.uint8)
    return uint8, {
        "uint8_min": lo,
        "uint8_max": hi,
        "clipped_low_count": below,
        "clipped_high_count": above,
        "saturation_count": int(np.sum(uint8 == 255)),
    }
