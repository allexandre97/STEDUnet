import copy
import json

import numpy as np

from fibras.synthetic.geometry3d import generate_persistent_chain_geometry
from fibras.synthetic.rasterizer3d import rasterize_3d_sample
from fibras.synthetic.storage import save_dataset, validate_dataset, validate_sample_arrays


def base_config():
    return {
        "dataset_name": "small_3d",
        "generator_mode": "persistent_chain_3d",
        "sample_count": 8,
        "split": "training",
        "calibration_status": "procedural_unmatched",
        "geometry": {
            "image_shape": [96, 96],
            "volume_depth_px": 40.0,
            "focal_plane_z_px": 20.0,
            "base_seed": 70,
            "scenario": "projected_depth_crossing",
            "fiber_count_range": [2, 2],
            "contour_length_range_px": [70.0, 80.0],
            "step_length_px": 4.0,
            "persistence_length_range_px": [80.0, 80.0],
            "arc_length_sampling_interval_px": 1.0,
            "width_calibration": {"angle_degrees": 0.0, "length_px": 70.0},
            "fluorophore": {"base_amplitude_range": [100, 100], "radius_range_px": [0.8, 0.8], "gap_probability": 0.0},
        },
        "targets": {"semantic_mask_source": "visible_signal_mask", "near_coplanar_depth_px": 8.0},
        "optical_model": {
            "psf_mode": "single_gaussian",
            "sigma_xy_0_px": 2.123,
            "sigma_z_px": 12.0,
            "defocus_broadening": 0.05,
            "kernel_truncation_radius": 3.0,
            "focal_depth_range_px": 4.0,
            "visible_signal_threshold": 2.0,
        },
        "output_mapping": {"background_level": 4.0, "background_noise_std": 0.0, "uint8_min": 0, "uint8_max": 255},
        "width_calibration": {"target_fwhm_px": 5.0, "min_allowed_fwhm_px": 4.5, "max_allowed_fwhm_px": 5.5},
    }


def test_3d_targets_preserve_mask_semantics_and_depth_maps():
    cfg = base_config()
    geom = generate_persistent_chain_geometry(cfg["geometry"], 0)
    arrays, _ = rasterize_3d_sample(geom, cfg["targets"], cfg["optical_model"], cfg["output_mapping"], 1)
    assert np.array_equal(arrays["projection_mask"] | arrays["in_focus_mask"], arrays["projection_mask"])
    assert np.array_equal(arrays["visible_signal_mask"], (arrays["total_clean_signal"] > 2.0).astype(np.uint8))
    assert np.allclose(arrays["in_focus_signal"] + arrays["out_of_focus_signal"], arrays["total_clean_signal"], atol=1e-4)
    assert arrays["projected_crossing_points_xy"].shape[0] >= 1
    assert arrays["near_coplanar_crossing_points_xy"].shape[0] == 0
    expected = np.zeros_like(arrays["overlap_count"])
    np.add.at(expected, (arrays["membership_y"], arrays["membership_x"]), 1)
    assert np.array_equal(arrays["overlap_count"], expected)
    contributing = arrays["total_clean_signal"] > 0
    assert np.all(arrays["nearest_depth_map"][contributing] >= 0)
    assert np.all(arrays["weighted_mean_depth_map"][contributing] >= 0)


def test_depth_validation_allows_tiny_negative_roundoff(tmp_path):
    cfg = dict(base_config(), sample_count=1)
    save_dataset(cfg, tmp_path)
    metadata = json.loads((tmp_path / "small_3d_0000.json").read_text(encoding="utf-8"))
    with np.load(tmp_path / "small_3d_0000.npz", allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    y, x = np.argwhere(arrays["contributing_overlap_count"] > 0)[0]

    rounded = copy.deepcopy(arrays)
    rounded["nearest_depth_map"][y, x] = -1e-8
    rounded["weighted_mean_depth_map"][y, x] = -1e-8
    assert "depth maps missing" not in "\n".join(
        validate_sample_arrays(metadata["sample_id"], rounded, metadata)
    )

    missing = copy.deepcopy(arrays)
    missing["nearest_depth_map"][y, x] = -1.0
    assert "depth maps missing" in "\n".join(
        validate_sample_arrays(metadata["sample_id"], missing, metadata)
    )


def test_true_junction_is_connected_in_3d_graph():
    cfg = base_config()
    cfg["geometry"].pop("scenario")
    cfg["geometry"]["branching_enabled"] = True
    geom = generate_persistent_chain_geometry(cfg["geometry"], 0)
    junction_nodes = [i for i, node in enumerate(geom["nodes"]) if node["type"] == "true_junction"]
    assert junction_nodes
    connected_edges = [edge for edge in geom["edges"] if junction_nodes[0] in edge["node_indices"]]
    assert len(connected_edges) == 3
    arrays, _ = rasterize_3d_sample(geom, cfg["targets"], cfg["optical_model"], cfg["output_mapping"], 1)
    assert arrays["junction_map"].sum() > 0


def test_3d_storage_validates_and_contains_no_object_arrays(tmp_path):
    cfg = base_config()
    cfg["geometry"]["image_shape"] = [80, 80]
    cfg["geometry"]["width_calibration"]["length_px"] = 54.0
    save_dataset(cfg, tmp_path)
    assert validate_dataset(tmp_path) == []
    with np.load(tmp_path / "small_3d_0000.npz", allow_pickle=False) as data:
        assert "fiber_points_xyz" in data.files
        assert all(data[name].dtype != object for name in data.files)
