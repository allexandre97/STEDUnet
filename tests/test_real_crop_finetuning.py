import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from fibras.training.bundle_diagnostics import bundle_diagnostics_from_arrays
from fibras.training.real_crops import (
    RealAnnotationRecord,
    apply_fold_assignments,
    build_real_crop_dataset,
    discover_real_annotation_triplets,
    image_splits,
    real_patch_categories,
)


def write_labeling(path: Path, shape: tuple[int, int], labels: dict[str, list[list[int]]]) -> None:
    path.write_text(
        json.dumps(
            {
                "interval": {"min": [0, 0], "max": [shape[1] - 1, shape[0] - 1], "n": 2},
                "labels": labels,
                "colors": {},
            }
        ),
        encoding="utf-8",
    )


def write_snakes(path: Path, y: int) -> None:
    path.write_text("\n".join(["stretch\t100.0", "#", "0", f"1\t0\t2.5\t{y + 0.5}\t0", f"1\t1\t13.5\t{y + 0.5}\t0"]), encoding="utf-8")


def real_record(root: Path, sample_id: str, y: int) -> RealAnnotationRecord:
    image = root / f"{sample_id}.png"
    labels = root / f"{sample_id}.labeling"
    snakes = root / f"{sample_id}.txt"
    Image.fromarray(np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)).save(image)
    write_labeling(
        labels,
        (16, 16),
        {
            "fibers": [[x, y] for x in range(2, 14)],
            "uncertain_ignore": [[x, y + 2] for x in range(2, 8)],
            "clump": [[x, y + 4] for x in range(2, 8)],
        },
    )
    write_snakes(snakes, y)
    return RealAnnotationRecord(sample_id, image, labels, snakes)


def test_real_crop_dataset_keeps_splits_at_whole_image_level(tmp_path):
    records = [real_record(tmp_path, "img_a", 2), real_record(tmp_path, "img_b", 3)]
    out = tmp_path / "crops"

    rows = build_real_crop_dataset(records, out, crop_size=8, stride=8, leave_one_out_image="img_b")

    assert {row["split"] for row in rows if row["source_image_id"] == "img_b"} == {"validation"}
    assert {row["split"] for row in rows if row["source_image_id"] == "img_a"} == {"train"}
    with np.load(out / rows[0]["npz_path"], allow_pickle=False) as data:
        assert set(data.files) >= {
            "render_uint8",
            "real_compatible_semantic_mask",
            "real_compatible_uncertain_ignore_mask",
            "real_compatible_skeleton_mask",
            "real_compatible_skeleton_valid_mask",
        }
        semantic = data["real_compatible_semantic_mask"]
        assert 255 in set(np.unique(semantic))
    manifest_rows = list(csv.DictReader((out / "real_crop_manifest.csv").open(newline="", encoding="utf-8")))
    assert len(manifest_rows) == len(rows)


def test_leave_one_image_out_split_rejects_unknown_image(tmp_path):
    records = [real_record(tmp_path, "img_a", 2)]
    with pytest.raises(ValueError, match="leave-one-out image"):
        image_splits(records, 1, 0.2, 0.0, "missing")


def test_grouped_images_cannot_cross_real_crop_splits(tmp_path):
    records = [
        RealAnnotationRecord("img_a", tmp_path / "a", tmp_path / "a", None, split_group_id="prep_1"),
        RealAnnotationRecord("img_b", tmp_path / "b", tmp_path / "b", None, split_group_id="prep_1"),
        RealAnnotationRecord("img_c", tmp_path / "c", tmp_path / "c", None, split_group_id="prep_2"),
    ]

    splits = image_splits(records, 1, 0.5, 0.0, None)

    assert splits["img_a"] == splits["img_b"]


def test_annotation_dir_discovery_matches_image_label_snake_triplets(tmp_path):
    record = real_record(tmp_path, "img a", 2)
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(tmp_path / "img a_labels.png")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(tmp_path / "orphan.png")

    records = discover_real_annotation_triplets(tmp_path)

    assert records == [RealAnnotationRecord("img_a", record.image_path, record.labels_path, record.snakes_path)]


def test_build_real_annotation_crops_accepts_annotation_dir(tmp_path):
    from scripts.build_real_annotation_crops import main

    real_record(tmp_path, "img_a", 2)
    out = tmp_path / "out"

    assert main(["--annotation-dir", str(tmp_path), "--out", str(out), "--crop-size", "8", "--leave-one-out-image", "img_a"]) == 0
    rows = list(csv.DictReader((out / "real_crop_manifest.csv").open(newline="", encoding="utf-8")))
    assert rows
    assert {row["split"] for row in rows} == {"validation"}


def test_build_real_annotation_crops_can_write_output_manifest(tmp_path):
    from scripts.build_real_annotation_crops import main

    real_record(tmp_path, "img_a", 2)
    out = tmp_path / "out"
    output_manifest = tmp_path / "manifests" / "loo_img_a.csv"

    assert (
        main(
            [
                "--annotation-dir",
                str(tmp_path),
                "--out",
                str(out),
                "--output-manifest",
                str(output_manifest),
                "--crop-size",
                "8",
                "--leave-one-out-image",
                "img_a",
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(output_manifest.open(newline="", encoding="utf-8")))
    assert rows
    assert Path(output_manifest.parent / rows[0]["npz_path"]).exists()
    assert rows[0]["npz_path"].startswith("../out/")


def test_build_real_annotation_crops_requires_one_input(tmp_path, capsys):
    from scripts.build_real_annotation_crops import main

    assert main(["--out", str(tmp_path / "out")]) == 2
    assert "provide exactly one" in capsys.readouterr().err


def test_mixed_batch_sampler_enforces_ratio():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import FixedRatioBatchSampler

    sampler = FixedRatioBatchSampler(10, 10, batch_size=10, synthetic_real_ratio="80:20", seed=1, batches_per_epoch=3)
    for batch in sampler:
        assert sum(index < 10 for index in batch) == 8
        assert sum(index >= 10 for index in batch) == 2


@pytest.mark.parametrize("ratio,expected", [("90:10", (9, 1)), ("80:20", (8, 2)), ("50:50", (5, 5)), ("0:100", (0, 10))])
def test_sampler_ratios_control_batch_composition(ratio, expected):
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import batch_counts

    assert batch_counts(10, ratio) == expected


def test_real_sampler_balances_parent_images_despite_crop_count():
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import balanced_real_index_stream

    rows = [
        {"sample_id": f"a_{i}", "source_image_id": "a", "sampling_categories": "uniform_random"}
        for i in range(50)
    ] + [{"sample_id": "b_0", "source_image_id": "b", "sampling_categories": "uniform_random"}]
    stream = balanced_real_index_stream(
        rows,
        np.random.default_rng(4),
        "fibrous_positive=0,clump_positive=0,uncertain_positive=0,dense_or_fibrous_clump_boundary=0,background_hard_negative=0,uniform_random=1",
    )
    parents = [rows[next(stream)]["source_image_id"] for _ in range(1000)]

    assert 400 < parents.count("a") < 600


def test_real_patch_categories_cover_requested_sampling_roles():
    semantic = np.zeros((8, 8), dtype=np.uint8)
    semantic[1:5, 1:5] = 1
    semantic[4:7, 4:7] = 3
    semantic[0, 0] = 255

    categories = real_patch_categories(semantic)

    assert set(categories) >= {
        "fibrous_positive", "clump_positive", "uncertain_positive",
        "dense_or_fibrous_clump_boundary", "uniform_random",
    }


def test_fold_assignment_is_inherited_by_parent_records(tmp_path):
    records = [
        RealAnnotationRecord("img_a", tmp_path / "a", tmp_path / "a", None, split_group_id="prep_a"),
        RealAnnotationRecord("img_b", tmp_path / "b", tmp_path / "b", None, split_group_id="prep_b"),
    ]
    manifest = tmp_path / "folds.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["outer_fold", "sample_id", "partition"])
        writer.writeheader()
        writer.writerows([
            {"outer_fold": "0", "sample_id": "img_a", "partition": "train"},
            {"outer_fold": "0", "sample_id": "img_b", "partition": "test"},
        ])

    assigned = apply_fold_assignments(records, manifest, 0)

    assert {record.sample_id: record.split for record in assigned} == {"img_a": "train", "img_b": "test"}


def test_uncertainty_head_adds_optional_loss():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import SmallUNet, compute_loss

    model = SmallUNet(base_channels=4, uncertainty_head=True)
    batch = {
        "image": torch.zeros(1, 1, 16, 16),
        "semantic": torch.zeros(1, 16, 16, dtype=torch.long),
        "skeleton": torch.zeros(1, 1, 16, 16),
        "uncertain": torch.zeros(1, 1, 16, 16),
        "valid": torch.ones(1, 1, 16, 16),
    }
    outputs = model(batch["image"])
    _, parts = compute_loss(outputs, batch, lambda_uncertainty=0.25)

    assert "uncertainty_logits" in outputs
    assert "uncertainty_loss" in parts


def test_bundle_diagnostics_report_width_coherence_and_clump_confusion():
    bundle = np.zeros((16, 16), dtype=np.uint8)
    bundle[4:8, 2:14] = 1
    axis = np.zeros_like(bundle)
    axis[5:7, 2:14] = 1
    pred = np.zeros((16, 16), dtype=np.uint8)
    pred[4:8, 8:14] = 2

    metrics = bundle_diagnostics_from_arrays({"bundle_mask": bundle, "bundle_axis_mask": axis}, pred)

    assert metrics["bundle_width_px_p50"] > 0
    assert metrics["bundle_directionality_coherence_mean"] > 0
    assert metrics["bundle_pixels_predicted_clump_fraction"] == 0.5
