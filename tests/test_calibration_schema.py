from pathlib import Path

from fibras.calibration.schema import CALIBRATION_DATA_STATUSES, artifact_metadata


def test_calibration_metadata_records_exploratory_status(tmp_path):
    for name in ["sted_images.csv", "sted_blanks.csv", "acquisition_groups.csv", "sted_splits.csv"]:
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    metadata = artifact_metadata(
        inventory_dir=tmp_path,
        splits_path=tmp_path / "sted_splits.csv",
        config={"a": 1},
        source_image_ids=["img1"],
        calibration_data_status="exploratory_unpartitioned",
        artifact_kind="fixture",
    )
    assert metadata["calibration_status"] in CALIBRATION_DATA_STATUSES
    assert metadata["result_status"] == "exploratory"
    assert len(metadata["source_commit_sha"]) == 40
    assert isinstance(metadata["working_tree_dirty"], bool)
    assert len(metadata["generation_config_sha256"]) == 64
    assert len(metadata["inventory_manifest_sha256"]) == 64
    assert len(metadata["split_manifest_sha256"]) == 64
