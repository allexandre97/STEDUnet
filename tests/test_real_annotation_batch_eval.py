import csv
import json

import numpy as np
from PIL import Image

from scripts.evaluate_real_annotation_batch import (
    aggregate_rows,
    aggregate_sample,
    discover_annotation_triplets,
    filter_records,
    parse_include_images,
    read_annotation_manifest,
)


def test_annotation_triplet_discovery_from_directory(tmp_path):
    image = tmp_path / "PN001_test.tif"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image)
    (tmp_path / "PN001_test.txt").write_text("#\n", encoding="utf-8")
    (tmp_path / "PN001_test.labeling").write_text("{}", encoding="utf-8")
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / "PN001_test_labels.png")

    records = discover_annotation_triplets(tmp_path)

    assert len(records) == 1
    assert records[0]["sample_id"] == "PN001_test"
    assert records[0]["image_path"].endswith("PN001_test.tif")
    assert records[0]["snakes_path"].endswith("PN001_test.txt")
    assert records[0]["labels_path"].endswith("PN001_test.labeling")


def test_manifest_parsing_resolves_relative_paths(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "image_path", "snakes_path", "labels_path", "notes", "expected_category"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample_a",
                "image_path": "image.tif",
                "snakes_path": "snakes.txt",
                "labels_path": "labels.labeling",
                "notes": "fixture",
                "expected_category": "early_fibrous",
            }
        )

    records = read_annotation_manifest(manifest)

    assert records == [
        {
            "sample_id": "sample_a",
            "image_path": str(tmp_path / "image.tif"),
            "snakes_path": str(tmp_path / "snakes.txt"),
            "labels_path": str(tmp_path / "labels.labeling"),
            "notes": "fixture",
            "expected_category": "early_fibrous",
        }
    ]


def test_include_image_filter_selects_requested_sample_ids():
    records = [
        {"sample_id": "PN001", "image_path": "a"},
        {"sample_id": "PN002", "image_path": "b"},
        {"sample_id": "PN003", "image_path": "c"},
    ]

    include = parse_include_images(["PN001,PN003"])
    selected = filter_records(records, include)

    assert [record["sample_id"] for record in selected] == ["PN001", "PN003"]


def test_include_image_filter_reports_available_ids():
    records = [{"sample_id": "PN001", "image_path": "a"}]

    try:
        filter_records(records, ["missing"])
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing include-image should fail")

    assert "include-image sample_id" in message
    assert "PN001" in message


def test_aggregate_sample_records_metrics_thresholds_and_clump_na(tmp_path):
    target_semantic = np.zeros((6, 6), dtype=np.uint8)
    target_semantic[1:4, 1:4] = 1
    target_skeleton = np.zeros((6, 6), dtype=np.uint8)
    target_skeleton[2, 1:4] = 1
    pred_semantic = np.zeros((6, 6), dtype=np.uint8)
    pred_semantic[1:4, 1:4] = 1
    skeleton_probability = np.zeros((6, 6), dtype=np.float32)
    skeleton_probability[2, 1:4] = 0.8
    sample = {
        "image_uint8": np.arange(36, dtype=np.uint8).reshape(6, 6),
        "image_float": None,
        "real_semantic_mask": target_semantic,
        "real_skeleton_mask": target_skeleton,
    }
    predictions = {
        "semantic_class_map": pred_semantic,
        "skeleton_probability": skeleton_probability,
    }
    metrics = {
        "fibrous_dice": 1.0,
        "fibrous_precision": 1.0,
        "fibrous_recall": 1.0,
        "clump_dice": "not_applicable",
        "clump_target_pixels": 0,
        "predicted_skeleton_inside_predicted_fibrous_fraction": 1.0,
        "predicted_skeleton_inside_target_fibrous_fraction": 1.0,
    }

    row = aggregate_sample(
        {"sample_id": "sample_a", "notes": "", "expected_category": ""},
        sample,
        predictions,
        metrics,
        tmp_path / "sample_a",
    )

    assert row["clump_dice"] == "not_applicable"
    assert row["clump_target_pixels"] == 0
    assert row["skeleton_dice_0.5"] == 1.0
    assert row["skeleton_dice_0.75"] == 1.0
    assert row["skeleton_dice_0.85"] == 0.0
    assert set(row["skeleton_analysis"]["thresholds"]) == {"0.5", "0.75", "0.85"}


def test_aggregate_rows_means_numeric_metrics():
    rows = [
        {"fibrous_dice": 0.5, "clump_target_pixels": 0, "skeleton_dice_0.75": 0.25},
        {"fibrous_dice": 1.0, "clump_target_pixels": 4, "skeleton_dice_0.75": "not_applicable"},
    ]

    aggregates = aggregate_rows(rows)

    assert aggregates["fibrous_dice"] == 0.75
    assert aggregates["clump_target_pixels"] == 2.0
    assert aggregates["skeleton_dice_0.75"] == 0.25


def test_batch_metrics_do_not_emit_topology_metrics(tmp_path):
    target_semantic = np.ones((4, 4), dtype=np.uint8)
    sample = {
        "image_uint8": np.zeros((4, 4), dtype=np.uint8),
        "image_float": None,
        "real_semantic_mask": target_semantic,
        "real_skeleton_mask": np.zeros((4, 4), dtype=np.uint8),
    }
    row = aggregate_sample(
        {"sample_id": "sample_a", "notes": "", "expected_category": ""},
        sample,
        {"semantic_class_map": target_semantic, "skeleton_probability": np.zeros((4, 4), dtype=np.float32)},
        {
            "fibrous_dice": 1.0,
            "fibrous_precision": 1.0,
            "fibrous_recall": 1.0,
            "clump_dice": "not_applicable",
            "clump_target_pixels": 0,
            "predicted_skeleton_inside_predicted_fibrous_fraction": "not_applicable",
            "predicted_skeleton_inside_target_fibrous_fraction": "not_applicable",
        },
        tmp_path / "sample_a",
    )
    keys = json.dumps(row)

    forbidden = ("endpoint", "crossing", "junction", "branch", "merge")
    assert not any(word in keys for word in forbidden)


def test_explicit_checkpoint_path_wins_over_run_dir(tmp_path):
    from scripts.evaluate_real_pilot_baseline import resolve_checkpoint_path

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    final = run_dir / "model.pt"
    best = run_dir / "model_best.pt"
    final.write_bytes(b"final")
    best.write_bytes(b"best")

    assert resolve_checkpoint_path(best, run_dir) == best
