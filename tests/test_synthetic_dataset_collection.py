import json

import numpy as np

from fibras.synthetic.collection import validate_dataset_collection
from fibras.synthetic.schema import DATASET_SCHEMA_VERSION_3D
from fibras.synthetic.storage import sha256_file, write_dataset_manifest


def test_collection_validates_global_ids_parent_hash_and_inheritance(tmp_path):
    parent_dir = tmp_path / "parent"
    composite_dir = tmp_path / "composite"
    write_sample(
        parent_dir,
        "synthetic_3d_realism_0000",
        {
            "projection_mask": np.ones((2, 2), dtype=np.uint8),
            "trace_points_xy": np.asarray([[0, 0], [1, 1]], dtype=np.float32),
            "render_float": np.ones((2, 2), dtype=np.float32),
            "render_uint8": np.ones((2, 2), dtype=np.uint8),
        },
        {
            "generation_config": {"dataset_name": "synthetic_3d_realism"},
            "scenario_category": "realism_calibration",
            "dataset_schema_version": DATASET_SCHEMA_VERSION_3D,
        },
    )
    parent_row = manifest_row(parent_dir)
    write_sample(
        composite_dir,
        "blank_composite_3d_realism_0000",
        {
            "projection_mask": np.ones((2, 2), dtype=np.uint8),
            "trace_points_xy": np.asarray([[0, 0], [1, 1]], dtype=np.float32),
            "render_float": np.full((2, 2), 2, dtype=np.float32),
            "render_uint8": np.full((2, 2), 2, dtype=np.uint8),
            "blank_float": np.ones((2, 2), dtype=np.float32),
        },
        {
            "configuration": {
                "compositing": {
                    "composite_dataset_name": "blank_composite_3d_realism"
                }
            },
            "dataset_schema_version": DATASET_SCHEMA_VERSION_3D,
            "parent_synthetic_sample_id": "synthetic_3d_realism_0000",
            "source_synthetic_artifact_hash": parent_row["npz_sha256"],
        },
    )
    assert validate_dataset_collection([parent_dir, composite_dir]) == []
    assert validate_dataset_collection([parent_dir, composite_dir], num_workers=2) == []
    metadata_path = composite_dir / "blank_composite_3d_realism_0000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_synthetic_artifact_hash"] = "bad"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert any(
        "SHA-256 mismatch" in error
        for error in validate_dataset_collection([parent_dir, composite_dir])
    )
    metadata["source_synthetic_artifact_hash"] = parent_row["npz_sha256"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    composite_npz = composite_dir / "blank_composite_3d_realism_0000.npz"
    with np.load(composite_npz, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["projection_mask"][0, 0] = 0
    np.savez_compressed(composite_npz, **arrays)
    assert any(
        "projection_mask was modified" in error
        for error in validate_dataset_collection([parent_dir, composite_dir])
    )


def test_collection_rejects_global_id_collisions_and_modified_structures(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    metadata = {
        "generation_config": {"dataset_name": "one"},
        "scenario_category": "realism_calibration",
        "dataset_schema_version": DATASET_SCHEMA_VERSION_3D,
    }
    write_sample(
        first,
        "collision_0000",
        {"projection_mask": np.zeros((2, 2), dtype=np.uint8)},
        metadata,
    )
    write_sample(
        second,
        "collision_0000",
        {"projection_mask": np.ones((2, 2), dtype=np.uint8)},
        {**metadata, "generation_config": {"dataset_name": "two"}},
    )
    assert any(
        "global sample_id collision" in error
        for error in validate_dataset_collection([first, second])
    )


def test_collection_allows_clump_ignore_stress_composite_parent(tmp_path):
    parent_dir = tmp_path / "parent"
    composite_dir = tmp_path / "composite"
    write_sample(
        parent_dir,
        "synthetic_sted_clump_ignore_stress_schema08_0000",
        {
            "projection_mask": np.ones((2, 2), dtype=np.uint8),
            "render_float": np.ones((2, 2), dtype=np.float32),
            "render_uint8": np.ones((2, 2), dtype=np.uint8),
        },
        {
            "generation_config": {"dataset_name": "synthetic_sted_clump_ignore_stress_schema08"},
            "scenario_category": "clump_ignore_stress",
            "dataset_schema_version": DATASET_SCHEMA_VERSION_3D,
        },
    )
    parent_row = manifest_row(parent_dir)
    write_sample(
        composite_dir,
        "sted_blank_composites_clump_ignore_stress_schema08_0000",
        {
            "projection_mask": np.ones((2, 2), dtype=np.uint8),
            "render_float": np.full((2, 2), 2, dtype=np.float32),
            "render_uint8": np.full((2, 2), 2, dtype=np.uint8),
            "blank_float": np.ones((2, 2), dtype=np.float32),
        },
        {
            "configuration": {
                "compositing": {
                    "composite_dataset_name": "sted_blank_composites_clump_ignore_stress_schema08"
                }
            },
            "dataset_schema_version": DATASET_SCHEMA_VERSION_3D,
            "parent_synthetic_sample_id": "synthetic_sted_clump_ignore_stress_schema08_0000",
            "source_synthetic_artifact_hash": parent_row["npz_sha256"],
        },
    )
    assert validate_dataset_collection([parent_dir, composite_dir]) == []


def write_sample(directory, sample_id, arrays, metadata):
    directory.mkdir()
    npz = directory / f"{sample_id}.npz"
    js = directory / f"{sample_id}.json"
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps({"sample_id": sample_id, **metadata}), encoding="utf-8")
    write_dataset_manifest(
        directory / "dataset_manifest.csv",
        [
            {
                "sample_id": sample_id,
                "npz_path": npz.name,
                "npz_sha256": sha256_file(npz),
                "json_path": js.name,
                "json_sha256": sha256_file(js),
                "schema_version": metadata["dataset_schema_version"],
                "generator_version": "fixture",
            }
        ],
    )


def manifest_row(directory):
    import csv

    with (directory / "dataset_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return next(csv.DictReader(handle))
