import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fibras.annotations.inventory import build_real_annotation_inventory


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_bundle(root: Path, image_root: Path, name: str, labels: dict[str, list[list[int]]]) -> str:
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    Image.fromarray(image).save(root / f"{name}.tif")
    Image.fromarray(image).save(image_root / f"{name}.tif")
    (root / f"{name}.labeling").write_text(
        json.dumps({
            "interval": {"min": [0, 0], "max": [7, 7], "n": 2},
            "labels": labels,
            "colors": {},
        }),
        encoding="utf-8",
    )
    return hashlib.sha256((image_root / f"{name}.tif").read_bytes()).hexdigest()


def source_row(name: str, checksum: str, sample_id: str, group: str) -> dict[str, str]:
    return {
        "relative_path": f"{name}.tif",
        "stable_image_id": sample_id,
        "source_root_id": "sted_fiber_data",
        "source_sha256": checksum,
        "shape_y": "8",
        "shape_x": "8",
        "culture_id": "PN001",
        "disease": "AD",
        "tau_isoform": "3R",
        "experimental_condition": "AD_3R",
        "div": "5",
        "div_token": "DIV05",
        "series_index": "0",
        "acquisition_group": group,
        "experimental_group_id": group,
    }


def test_inventory_resolves_overlap_as_uncertain_and_includes_its_crops(tmp_path):
    annotations = tmp_path / "annotations"
    images = tmp_path / "images"
    annotations.mkdir()
    images.mkdir()
    valid_name = "PN001_3R_AD_DIV05 (Series 0) [1]"
    overlap_name = "PN001_3R_AD_DIV05 (Series 1) [1]"
    valid_hash = write_bundle(annotations, images, valid_name, {"fibers": [[1, 1]], "clump": [[5, 5]]})
    overlap_hash = write_bundle(
        annotations,
        images,
        overlap_name,
        {"fibers": [[1, 1]], "uncertain_ignore": [[1, 1]]},
    )
    image_manifest = tmp_path / "sted_images.csv"
    split_manifest = tmp_path / "sted_splits.csv"
    write_csv(image_manifest, [
        source_row(valid_name, valid_hash, "img_valid", "prep_1"),
        source_row(overlap_name, overlap_hash, "img_overlap", "prep_1"),
    ])
    write_csv(split_manifest, [
        {"stable_image_id": "img_valid", "eligibility": "training"},
        {"stable_image_id": "img_overlap", "eligibility": "training"},
    ])

    rows, crops, audit = build_real_annotation_inventory(
        annotations, images, image_manifest, split_manifest, crop_size=4, stride=4
    )

    assert len(rows) == 2
    assert {row["sample_id"]: row["validation_status"] for row in rows} == {
        "img_valid": "valid",
        "img_overlap": "valid",
    }
    assert {crop["parent_image_id"] for crop in crops} == {"img_valid", "img_overlap"}
    assert {crop["split"] for crop in crops} == {"train"}
    assert audit["annotated_image_count"] == 2
    assert audit["validation_problems"] == []


def test_inventory_rejects_confident_fibers_clump_overlap(tmp_path):
    annotations = tmp_path / "annotations"
    images = tmp_path / "images"
    annotations.mkdir()
    images.mkdir()
    name = "PN001_3R_AD_DIV05 (Series 0) [1]"
    checksum = write_bundle(annotations, images, name, {"fibers": [[1, 1]], "clump": [[1, 1]]})
    image_manifest = tmp_path / "sted_images.csv"
    split_manifest = tmp_path / "sted_splits.csv"
    write_csv(image_manifest, [source_row(name, checksum, "img_overlap", "prep_1")])
    write_csv(split_manifest, [{"stable_image_id": "img_overlap", "eligibility": "training"}])

    rows, crops, audit = build_real_annotation_inventory(
        annotations, images, image_manifest, split_manifest, crop_size=4, stride=4
    )

    assert rows[0]["validation_status"] == "invalid"
    assert not crops
    assert audit["validation_problems"] == [
        f"{name}: fibers/clump label overlap outside uncertain_ignore"
    ]


def test_inventory_reports_missing_annotation_image(tmp_path):
    annotations = tmp_path / "annotations"
    images = tmp_path / "images"
    annotations.mkdir()
    images.mkdir()
    name = "PN001_3R_AD_DIV05 (Series 0) [1]"
    (annotations / f"{name}.labeling").write_text(
        json.dumps({"interval": {"min": [0, 0], "max": [7, 7]}, "labels": {}}),
        encoding="utf-8",
    )
    image_manifest = tmp_path / "sted_images.csv"
    split_manifest = tmp_path / "sted_splits.csv"
    write_csv(image_manifest, [source_row(name, "missing", "img_1", "prep_1")])
    write_csv(split_manifest, [{"stable_image_id": "img_1", "eligibility": "training"}])

    _, crops, audit = build_real_annotation_inventory(
        annotations, images, image_manifest, split_manifest, crop_size=4
    )

    assert not crops
    assert any("missing .tif" in problem for problem in audit["validation_problems"])
