from fibras.synthetic.geometry import generate_geometry
from fibras.synthetic.targets import crossing_points


def base_config():
    return {
        "image_shape": [128, 128],
        "base_seed": 1,
        "fiber_count_range": [7, 7],
        "points_per_fiber": 32,
        "fiber_radius_px": 2.0,
        "fiber_intensity": 100.0,
        "include_crossings": True,
        "include_true_junctions": True,
        "include_truncated": True,
    }


def test_geometry_is_deterministic_and_contains_required_structures():
    a = generate_geometry(base_config(), 0)
    b = generate_geometry(base_config(), 0)
    assert len(a["fibers"]) == len(b["fibers"])
    assert (a["fibers"][0]["points_xy"] == b["fibers"][0]["points_xy"]).all()
    assert any(node["type"] == "true_junction" for node in a["nodes"])
    assert any(edge["truncated_start"] or edge["truncated_end"] for edge in a["edges"])
    assert crossing_points(a)

