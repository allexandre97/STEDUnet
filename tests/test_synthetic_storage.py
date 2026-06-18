from fibras.synthetic.storage import save_dataset, validate_dataset


def test_npz_json_storage_validates(tmp_path):
    config = {
        "dataset_name": "test_sted",
        "sample_count": 8,
        "split": "training",
        "calibration_status": "procedural_unmatched",
        "geometry": {"image_shape": [96, 96], "fiber_count_range": [7, 7], "base_seed": 10},
        "targets": {},
        "rendering": {"base_seed": 20, "psf_sigma_px": 0.5, "background_level": 4, "uint8_min": 0, "uint8_max": 255},
    }
    save_dataset(config, tmp_path)
    assert (tmp_path / "dataset_manifest.csv").exists()
    assert validate_dataset(tmp_path) == []

