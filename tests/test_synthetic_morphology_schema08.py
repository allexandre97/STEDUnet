import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from fibras.synthetic.rasterizer3d import finalize_morphology_apparent_targets
from fibras.synthetic.real_compatible import build_real_compatible_targets
from fibras.synthetic.schema import (
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_0_7,
    GENERATOR_VERSION_3D_MORPHOLOGY,
    load_yaml,
)
from fibras.synthetic.storage import build_sample, validate_sample_arrays


def morphology_config():
    config = load_yaml("configs/synthetic_sted/morphology_heterogeneity_review.yaml")
    config["dataset_name"] = "schema08_test"
    config["sample_count"] = 8
    config["geometry"]["image_shape"] = [96, 96]
    config["scene_morphology"].pop("review_modes", None)
    config["targets"]["distance_transform_enabled"] = False
    config["targets"]["min_bundle_width_px"] = 4.0
    return config


def tiny_apparent_arrays():
    shape = (3, 4)
    zeros = np.zeros(shape, dtype=np.uint8)
    ones = np.ones(shape, dtype=np.uint8)
    arrays = {
        "semantic_class_mask": zeros.copy(),
        "individual_filament_mask": zeros.copy(),
        "bundle_mask": zeros.copy(),
        "clump_mask": zeros.copy(),
        "uncertain_ignore_mask": zeros.copy(),
        "individual_filament_source_support_mask": ones.copy(),
        "bundle_source_support_mask": ones.copy(),
        "clump_source_support_mask": ones.copy(),
        "individual_filament_signal": np.zeros(shape, dtype=np.float32),
        "bundle_signal": np.zeros(shape, dtype=np.float32),
        "clump_signal": np.zeros(shape, dtype=np.float32),
        "filament_centerline_mask": zeros.copy(),
        "centerline_mask": zeros.copy(),
        "bundle_axis_mask": zeros.copy(),
        "endpoint_map": ones.copy(),
        "bundle_transition_mask": zeros.copy(),
        "clump_transition_mask": zeros.copy(),
    }
    for prefix in ["individual_filament", "bundle", "clump"]:
        arrays[f"{prefix}_membership_y"] = np.zeros(0, dtype=np.int32)
        arrays[f"{prefix}_membership_x"] = np.zeros(0, dtype=np.int32)
        arrays[f"{prefix}_membership_instance_id"] = np.zeros(0, dtype=np.int32)
    return arrays


def test_schema08_is_emitted_for_new_morphology_samples():
    sample_id, arrays, metadata = build_sample(morphology_config(), 0)

    assert metadata["dataset_schema_version"] == DATASET_SCHEMA_VERSION_3D_MORPHOLOGY
    assert metadata["generator_version"] == GENERATOR_VERSION_3D_MORPHOLOGY
    assert "individual_filament_source_support_mask" in arrays
    assert validate_sample_arrays(sample_id, arrays, metadata) == []


def test_schema07_examples_remain_readable():
    dataset = Path("examples/sted_blank_composites_morphology_review")
    if not (dataset / "dataset_manifest.csv").exists():
        pytest.skip("optional local schema-0.7 morphology example fixture is absent")
    with (dataset / "dataset_manifest.csv").open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    npz_name = row["npz_path"]
    json_name = row["json_path"]
    metadata = json.loads((dataset / json_name).read_text(encoding="utf-8"))
    with np.load(dataset / npz_name, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}

    assert metadata["dataset_schema_version"] == DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_0_7
    assert validate_sample_arrays(metadata["sample_id"], arrays, metadata) == []


def test_apparent_masks_use_threshold_band_and_class_priority():
    arrays = tiny_apparent_arrays()
    arrays["individual_filament_signal"][0, 0] = 10
    arrays["bundle_signal"][0, 1] = 10
    arrays["clump_signal"][0, 2] = 10
    arrays["individual_filament_signal"][1, 0] = 8
    arrays["individual_filament_signal"][1, 1] = 10
    arrays["bundle_signal"][1, 1] = 10
    arrays["clump_signal"][1, 1] = 10
    arrays["bundle_transition_mask"][2, 0] = 1

    finalize_morphology_apparent_targets(
        arrays,
        10,
        {
            "apparent_mask_visibility_low_factor": 0.75,
            "apparent_mask_visibility_high_factor": 1.0,
            "apparent_mask_max_source_distance_px": 10,
        },
    )

    assert arrays["semantic_class_mask"].tolist() == [
        [1, 2, 3, 0],
        [255, 3, 0, 0],
        [255, 0, 0, 0],
    ]
    assert set(np.unique(arrays["semantic_class_mask"])) == {0, 1, 2, 3, 255}
    assert np.array_equal(
        arrays["semantic_mask"],
        np.isin(arrays["semantic_class_mask"], [1, 2, 3]).astype(np.uint8),
    )


def test_real_compatible_view_uses_apparent_masks_and_clips_skeleton():
    arrays = tiny_apparent_arrays()
    arrays["individual_filament_signal"][0, 0] = 10
    arrays["bundle_signal"][0, 1] = 10
    arrays["clump_signal"][0, 2] = 10
    arrays["individual_filament_signal"][1, 0] = 8
    arrays["filament_centerline_mask"][0, 0] = 1
    arrays["filament_centerline_mask"][0, 2] = 1
    arrays["bundle_axis_mask"][0, 1] = 1
    arrays["bundle_axis_mask"][1, 0] = 1
    finalize_morphology_apparent_targets(
        arrays,
        10,
        {
            "apparent_mask_visibility_low_factor": 0.75,
            "apparent_mask_visibility_high_factor": 1.0,
            "apparent_mask_max_source_distance_px": 10,
        },
    )

    real = build_real_compatible_targets(arrays)

    assert real["real_compatible_semantic_mask"].tolist() == [
        [1, 1, 3, 0],
        [255, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert real["real_compatible_skeleton_mask"].tolist() == [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert set(np.unique(real["real_compatible_semantic_mask"])) == {0, 1, 3, 255}


def test_source_support_is_invariant_under_foreground_scale_and_roles_are_disjoint():
    base = morphology_config()
    bright = copy.deepcopy(base)
    bright["output_mapping"] = dict(base["output_mapping"], foreground_scale=2.0)
    _, base_arrays, metadata = build_sample(base, 1)
    _, bright_arrays, _ = build_sample(bright, 1)

    for name in [
        "individual_filament_source_support_mask",
        "bundle_source_support_mask",
        "clump_source_support_mask",
    ]:
        assert np.array_equal(base_arrays[name], bright_arrays[name])
        assert name in metadata["target_roles"]["latent_synthetic_provenance"]
        assert name not in metadata["target_roles"]["supervised"]

    roles = [
        set(metadata["target_roles"][name])
        for name in [
            "supervised",
            "latent_synthetic_provenance",
            "diagnostic_only",
            "real_compatible_supervised",
        ]
    ]
    for i, left in enumerate(roles):
        for right in roles[i + 1 :]:
            assert left.isdisjoint(right)


def test_clump_hole_fill_ratio_is_not_called_solidity():
    text = Path("src/fibras/calibration/morphology.py").read_text(encoding="utf-8")

    assert "clump_hole_fill_ratio" in text
    assert "clump_solidity" not in text
