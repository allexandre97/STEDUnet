from fibras.synthetic.schema import CALIBRATION_STATUSES, REQUIRED_ARRAYS
from fibras.synthetic.storage import build_sample


def test_sample_schema_has_required_arrays():
    config = {
        "dataset_name": "test",
        "sample_count": 8,
        "split": "training",
        "calibration_status": "procedural_unmatched",
        "geometry": {"image_shape": [128, 128], "fiber_count_range": [7, 7], "base_seed": 3},
        "targets": {},
        "rendering": {"base_seed": 4, "psf_sigma_px": 0, "background_level": 4, "uint8_min": 0, "uint8_max": 255},
    }
    _, arrays, metadata = build_sample(config, 0)
    assert REQUIRED_ARRAYS <= set(arrays)
    assert metadata["calibration_status"] in CALIBRATION_STATUSES
    assert metadata["source_blank_provenance"] == "not_applicable"
    assert all(arr.dtype != object for arr in arrays.values())

