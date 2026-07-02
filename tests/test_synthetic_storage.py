import csv
import json

import numpy as np
import pytest

from fibras.synthetic.storage import save_dataset, validate_dataset


def small_config() -> dict:
    return {
        "dataset_name": "test_sted",
        "sample_count": 2,
        "split": "training",
        "calibration_status": "procedural_unmatched",
        "geometry": {"image_shape": [96, 96], "fiber_count_range": [7, 7], "base_seed": 10},
        "targets": {},
        "rendering": {"base_seed": 20, "psf_sigma_px": 0.5, "background_level": 4, "uint8_min": 0, "uint8_max": 255},
    }


def test_npz_json_storage_validates(tmp_path):
    save_dataset(dict(small_config(), sample_count=8), tmp_path)
    assert (tmp_path / "dataset_manifest.csv").exists()
    assert validate_dataset(tmp_path) == []
    assert validate_dataset(tmp_path, num_workers=2) == []


def test_parallel_synthetic_generation_matches_serial(tmp_path):
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    config = small_config()
    save_dataset(config, serial, num_workers=1)
    save_dataset(config, parallel, num_workers=2)
    with (serial / "dataset_manifest.csv").open(newline="", encoding="utf-8") as f:
        serial_rows = list(csv.DictReader(f))
    with (parallel / "dataset_manifest.csv").open(newline="", encoding="utf-8") as f:
        parallel_rows = list(csv.DictReader(f))
    assert [r["sample_id"] for r in serial_rows] == [r["sample_id"] for r in parallel_rows]
    for row in serial_rows:
        sid = row["sample_id"]
        with np.load(serial / f"{sid}.npz", allow_pickle=False) as serial_npz, np.load(parallel / f"{sid}.npz", allow_pickle=False) as parallel_npz:
            for name in ["render_uint8", "semantic_mask", "overlap_count"]:
                assert np.array_equal(serial_npz[name], parallel_npz[name])
        serial_meta = json.loads((serial / f"{sid}.json").read_text(encoding="utf-8"))
        parallel_meta = json.loads((parallel / f"{sid}.json").read_text(encoding="utf-8"))
        for key in ["sample_id", "geometry_seed", "rendering_seed", "dataset_schema_version", "generator_version"]:
            assert serial_meta[key] == parallel_meta[key]


def test_parallel_validation_reports_errors_in_manifest_order(tmp_path):
    save_dataset(dict(small_config(), sample_count=4), tmp_path)
    for index in [1, 3]:
        path = tmp_path / f"test_sted_{index:04d}.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["sample_id"] = "corrupt"
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    errors = validate_dataset(tmp_path, deterministic=False, num_workers=2)
    checksum_errors = [error for error in errors if "JSON checksum mismatch" in error]
    assert checksum_errors == [
        "test_sted_0001: JSON checksum mismatch",
        "test_sted_0003: JSON checksum mismatch",
    ]


def test_parallel_validation_reports_worker_exception(tmp_path):
    save_dataset(dict(small_config(), sample_count=2), tmp_path)
    (tmp_path / "test_sted_0001.npz").unlink()
    errors = validate_dataset(tmp_path, num_workers=2)
    assert len(errors) == 1
    assert errors[0].startswith("test_sted_0001: validator worker failed: FileNotFoundError:")


def test_skip_existing_reuses_valid_samples_and_plain_existing_fails(tmp_path):
    config = small_config()
    save_dataset(config, tmp_path)
    with pytest.raises(FileExistsError):
        save_dataset(config, tmp_path)
    save_dataset(config, tmp_path, skip_existing=True)
    assert validate_dataset(tmp_path) == []
