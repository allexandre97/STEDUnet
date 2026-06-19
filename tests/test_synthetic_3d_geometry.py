import numpy as np

from fibras.synthetic.geometry3d import generate_persistent_chain_geometry, persistent_vertices, resample_by_arc_length


def cfg(**overrides):
    base = {
        "image_shape": [96, 96],
        "volume_depth_px": 40.0,
        "focal_plane_z_px": 20.0,
        "base_seed": 10,
        "fiber_count_range": [3, 3],
        "contour_length_range_px": [80.0, 90.0],
        "step_length_px": 4.0,
        "persistence_length_range_px": [80.0, 80.0],
        "arc_length_sampling_interval_px": 1.0,
        "boundary_mode": "reflect",
        "fluorophore": {"base_amplitude_range": [100, 100], "radius_range_px": [0.7, 0.7], "gap_probability": 0.0},
    }
    base.update(overrides)
    return base


def mean_tangent_correlation(points):
    tangents = np.diff(points, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-8)
    return float(np.mean(np.sum(tangents[:-1] * tangents[1:], axis=1)))


def test_persistent_chain_geometry_is_deterministic_and_valid():
    a = generate_persistent_chain_geometry(cfg(), 0)
    b = generate_persistent_chain_geometry(cfg(), 0)
    assert len(a["fibers"]) == 3
    for fa, fb in zip(a["fibers"], b["fibers"]):
        assert np.array_equal(fa["raw_vertices_xyz"], fb["raw_vertices_xyz"])
        assert np.array_equal(fa["points_xyz"], fb["points_xyz"])
        assert np.all(np.isfinite(fa["points_xyz"]))
        assert np.all(fa["points_xyz"] >= 0)
        assert np.all(fa["points_xyz"][:, 0] <= 95)
        assert np.all(fa["points_xyz"][:, 1] <= 95)
        assert np.all(fa["points_xyz"][:, 2] <= 40)


def test_persistence_length_increases_tangent_correlation():
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    low = persistent_vertices(cfg(persistence_length_range_px=[8.0, 8.0]), rng1, 96, 96, 40, 20)
    high = persistent_vertices(cfg(persistence_length_range_px=[200.0, 200.0]), rng2, 96, 96, 40, 20)
    assert mean_tangent_correlation(high) > mean_tangent_correlation(low)


def test_arc_length_resampling_is_uniform_and_preserves_length():
    points = np.asarray([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=np.float32)
    sampled = resample_by_arc_length(points, 1.0)
    lengths = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
    assert abs(float(lengths.sum()) - 20.0) < 1e-3
    assert np.percentile(lengths, 95) - np.percentile(lengths, 5) < 1e-3


def test_terminate_boundary_stops_without_invalid_coordinates():
    g = generate_persistent_chain_geometry(
        cfg(image_shape=[32, 32], contour_length_range_px=[500, 500], persistence_length_range_px=[1000, 1000], boundary_mode="terminate"),
        0,
    )
    for fiber in g["fibers"]:
        assert np.all(np.isfinite(fiber["points_xyz"]))
        assert np.all(fiber["points_xyz"][:, 0] <= 31)
        assert np.all(fiber["points_xyz"][:, 1] <= 31)
