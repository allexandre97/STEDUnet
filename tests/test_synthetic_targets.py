import numpy as np

from fibras.synthetic.geometry import generate_geometry
from fibras.synthetic.targets import rasterize_targets


def test_targets_preserve_memberships_and_topology():
    geometry = generate_geometry(
        {
            "image_shape": [128, 128],
            "base_seed": 2,
            "fiber_count_range": [7, 7],
            "points_per_fiber": 32,
            "fiber_radius_px": 2.0,
            "fiber_intensity": 100.0,
            "include_crossings": True,
            "include_true_junctions": True,
            "include_truncated": True,
        },
        0,
    )
    targets = rasterize_targets(geometry, {})
    expected = np.zeros_like(targets["overlap_count"])
    np.add.at(expected, (targets["membership_y"], targets["membership_x"]), 1)
    assert np.array_equal(targets["overlap_count"], expected)
    assert np.array_equal(targets["semantic_mask"], (expected > 0).astype(np.uint8))
    assert targets["crossing_map"].sum() > 0
    assert targets["junction_map"].sum() > 0
    assert targets["endpoint_map"].sum() > 0

