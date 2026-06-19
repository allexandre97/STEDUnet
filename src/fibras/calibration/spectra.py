"""Power-spectrum and autocorrelation helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    scaled = np.zeros_like(arr, dtype=np.uint8) if hi <= lo else np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    resized = Image.fromarray(scaled).resize((size, size), Image.Resampling.BILINEAR)
    out = np.asarray(resized, dtype=np.float32)
    return out / 255.0 * (hi - lo) + lo


def radial_power_spectrum(image: np.ndarray, size: int = 256, bins: int = 32) -> np.ndarray:
    arr = resize_square(image, size)
    arr = arr - float(arr.mean())
    fft = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(fft) ** 2
    return radial_average(power, bins).astype(np.float32)


def normalized_radial_power_spectrum(image: np.ndarray, size: int = 256, bins: int = 32) -> np.ndarray:
    """Radial spectral shape with DC removed and non-DC power normalized."""

    power = radial_power_spectrum(image, size, bins).astype(np.float64)
    power[0] = 0.0
    total = float(power.sum())
    if total <= 0 or not np.isfinite(total):
        return np.zeros_like(power, dtype=np.float32)
    return (power / total).astype(np.float32)


def autocorrelation_radial(image: np.ndarray, size: int = 256, bins: int = 32) -> np.ndarray:
    arr = resize_square(image, size)
    arr = arr - float(arr.mean())
    fft = np.fft.fft2(arr)
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(fft) ** 2).real)
    center = ac[size // 2, size // 2]
    if center:
        ac = ac / center
    return radial_average(ac, bins).astype(np.float32)


def directional_power_ratio(image: np.ndarray, size: int = 256) -> float:
    arr = resize_square(image, size)
    arr = arr - float(arr.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
    c = size // 2
    horizontal = float(np.mean(power[c - 2 : c + 3, :]))
    vertical = float(np.mean(power[:, c - 2 : c + 3]))
    return horizontal / max(vertical, 1e-9)


def radial_average(values: np.ndarray, bins: int) -> np.ndarray:
    h, w = values.shape
    y, x = np.indices(values.shape)
    r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    edges = np.linspace(0, r.max(), bins + 1)
    out = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        out[i] = float(values[mask].mean()) if np.any(mask) else 0.0
    return out
