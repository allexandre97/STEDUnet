"""Real-compatible target views for rich synthetic morphology samples."""

from __future__ import annotations

from typing import Any

import numpy as np


REAL_COMPATIBLE_SUPERVISED_TARGETS = [
    "real_compatible_semantic_mask",
    "real_compatible_fibrous_mask",
    "real_compatible_clump_mask",
    "real_compatible_uncertain_ignore_mask",
    "real_compatible_skeleton_mask",
]

SYNTHETIC_ONLY_NOT_REAL_SUPERVISED = [
    "endpoint_map",
    "projected_crossing_map",
    "junction_map",
    "latent_geometry_membership_y",
    "latent_geometry_membership_x",
    "latent_geometry_membership_instance_id",
    "latent_geometry_overlap_count",
    "clump_fragment_fiber_ids",
    "fiber_structure_type",
]


def build_real_compatible_targets(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Collapse rich synthetic morphology targets to the expert real vocabulary."""
    individual, bundle, clump, uncertain = synthetic_class_masks(arrays)
    semantic = np.zeros(individual.shape, dtype=np.uint8)
    semantic[(individual | bundle) > 0] = 1
    semantic[clump > 0] = 3
    semantic[uncertain > 0] = 255

    skeleton = np.zeros_like(semantic, dtype=np.uint8)
    for name in ["filament_centerline_mask", "bundle_axis_mask"]:
        if name in arrays:
            skeleton |= arrays[name].astype(np.uint8)
    skeleton &= semantic == 1

    return {
        "real_compatible_semantic_mask": semantic,
        "real_compatible_fibrous_mask": (semantic == 1).astype(np.uint8),
        "real_compatible_clump_mask": (semantic == 3).astype(np.uint8),
        "real_compatible_uncertain_ignore_mask": (semantic == 255).astype(np.uint8),
        "real_compatible_skeleton_mask": skeleton.astype(np.uint8),
    }


def collapse_synthetic_semantic_mask(mask: np.ndarray) -> np.ndarray:
    """Map synthetic classes 0/1/2/3/255 to real-compatible 0/1/3/255."""
    out = np.zeros(mask.shape, dtype=np.uint8)
    values = set(map(int, np.unique(mask)))
    unknown = values - {0, 1, 2, 3, 255}
    if unknown:
        raise ValueError(f"unsupported synthetic semantic class values: {sorted(unknown)}")
    out[(mask == 1) | (mask == 2)] = 1
    out[mask == 3] = 3
    out[mask == 255] = 255
    return out


def synthetic_class_masks(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if all(
        name in arrays
        for name in [
            "individual_filament_mask",
            "bundle_mask",
            "clump_mask",
            "uncertain_ignore_mask",
        ]
    ):
        return (
            arrays["individual_filament_mask"].astype(np.uint8),
            arrays["bundle_mask"].astype(np.uint8),
            arrays["clump_mask"].astype(np.uint8),
            arrays["uncertain_ignore_mask"].astype(np.uint8),
        )
    if "semantic_class_mask" not in arrays:
        raise ValueError("synthetic morphology arrays require class masks or semantic_class_mask")
    classes = arrays["semantic_class_mask"]
    return (
        (classes == 1).astype(np.uint8),
        (classes == 2).astype(np.uint8),
        (classes == 3).astype(np.uint8),
        (classes == 255).astype(np.uint8),
    )


def validate_real_compatible_targets(
    sample_id: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    expected = build_real_compatible_targets(arrays)
    errors: list[str] = []
    missing = set(REAL_COMPATIBLE_SUPERVISED_TARGETS) - set(arrays)
    if missing:
        errors.append(f"{sample_id}: missing real-compatible arrays {sorted(missing)}")
        return errors

    semantic = arrays["real_compatible_semantic_mask"]
    present = set(map(int, np.unique(semantic)))
    if not present <= {0, 1, 3, 255}:
        errors.append(
            f"{sample_id}: invalid real-compatible class IDs {sorted(present - {0, 1, 3, 255})}"
        )
    for name, expected_array in expected.items():
        if not np.array_equal(arrays[name], expected_array):
            errors.append(f"{sample_id}: {name} disagrees with synthetic source masks")

    if np.any(arrays["real_compatible_skeleton_mask"] & arrays["real_compatible_clump_mask"]):
        errors.append(f"{sample_id}: real-compatible skeleton overlaps clump")
    if np.any(arrays["real_compatible_skeleton_mask"] & arrays["real_compatible_uncertain_ignore_mask"]):
        errors.append(f"{sample_id}: real-compatible skeleton overlaps uncertain_ignore")
    if metadata is not None:
        roles = metadata.get("target_roles", {})
        real_targets = set(roles.get("real_compatible_supervised", []))
        forbidden = real_targets & {
            "endpoint_map",
            "projected_crossing_map",
            "crossing_map",
            "junction_map",
            "latent_geometry_membership_y",
            "latent_geometry_membership_x",
            "latent_geometry_membership_instance_id",
            "latent_geometry_overlap_count",
        }
        if forbidden:
            errors.append(
                f"{sample_id}: synthetic-only arrays listed as real-compatible targets {sorted(forbidden)}"
            )
    return errors
