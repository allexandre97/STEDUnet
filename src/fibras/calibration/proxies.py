"""Proxy-derived, non-ground-truth structure estimates."""

from __future__ import annotations

from typing import Any

import numpy as np


PROXY_WARNING = "proxy estimate only; not ground truth"


def proxy_measurements(image: np.ndarray, config: dict[str, Any]) -> dict[str, str]:
    arr = image.astype(np.float32)
    threshold_k = float(config.get("threshold_mad_k", 6.0))
    vals = arr.ravel()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    threshold = max(float(np.percentile(vals, 99)), med + threshold_k * 1.4826 * mad)
    mask = (arr >= threshold) & (arr > med)
    components = connected_components(mask)
    lengths = [max(c["height"], c["width"]) for c in components]
    widths = [min(c["height"], c["width"]) for c in components if c["area"] >= int(config.get("min_component_area_px", 8))]
    gx, gy = gradients(arr)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    ridge = grad_mag
    strong = ridge > np.percentile(ridge, float(config.get("ridge_percentile", 95)))
    orientations = np.mod(np.arctan2(gy[strong], gx[strong]) + np.pi / 2.0, np.pi)
    endpoint_candidates = sum(1 for c in components if c["area"] >= int(config.get("min_component_area_px", 8)))
    crossing_candidates = sum(1 for c in components if c["area"] >= int(config.get("crossing_component_area_px", 20)) and min(c["height"], c["width"]) >= 3)
    return {
        "proxy_estimator_name": "mad_threshold_components_gradient_orientation",
        "proxy_warning": PROXY_WARNING,
        "proxy_threshold": f"{threshold:.6g}",
        "foreground_occupancy_proxy": f"{float(mask.mean()):.6g}",
        "ridge_response_p50": f"{float(np.percentile(ridge, 50)):.6g}",
        "ridge_response_p95": f"{float(np.percentile(ridge, 95)):.6g}",
        "orientation_mean_proxy": f"{float(np.mean(orientations)):.6g}" if orientations.size else "not_available",
        "orientation_resultant_length_proxy": f"{orientation_resultant(orientations):.6g}" if orientations.size else "not_available",
        "apparent_width_proxy_p50": f"{float(np.median(widths)):.6g}" if widths else "not_available",
        "component_length_proxy_p50": f"{float(np.median(lengths)):.6g}" if lengths else "not_available",
        "local_snr_proxy": local_snr_proxy(arr, mask),
        "endpoint_candidate_density": f"{endpoint_candidates / arr.size:.6g}",
        "crossing_candidate_density": f"{crossing_candidates / arr.size:.6g}",
        "proxy_sensitivity_range": f"threshold_mad_k={config.get('sensitivity_threshold_mad_k', [4, 8])}",
        "proxy_uncertainty": "broad; threshold-sensitive and unverified without labels",
    }


def gradients(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(arr, dtype=np.float32)
    gy = np.zeros_like(arr, dtype=np.float32)
    gx[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / 2.0
    gy[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / 2.0
    return gx, gy


def orientation_resultant(orientations: np.ndarray) -> float:
    if orientations.size == 0:
        return 0.0
    doubled = 2.0 * orientations
    return float(np.sqrt(np.mean(np.cos(doubled)) ** 2 + np.mean(np.sin(doubled)) ** 2))


def local_snr_proxy(arr: np.ndarray, mask: np.ndarray) -> str:
    fg = arr[mask]
    bg = arr[~mask]
    if fg.size == 0 or bg.size == 0:
        return "not_available"
    return f"{float((np.median(fg) - np.median(bg)) / max(np.std(bg), 1e-6)):.6g}"


def connected_components(mask: np.ndarray) -> list[dict[str, int]]:
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[dict[str, int]] = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        comp_y: list[int] = []
        comp_x: list[int] = []
        while stack:
            y, x = stack.pop()
            comp_y.append(y)
            comp_x.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        components.append(
            {
                "area": len(comp_y),
                "height": max(comp_y) - min(comp_y) + 1,
                "width": max(comp_x) - min(comp_x) + 1,
            }
        )
    return components
