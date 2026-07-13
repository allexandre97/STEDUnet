import json

import numpy as np
from PIL import Image

from fibras.annotations import (
    build_real_annotation_sample,
    collapse_synthetic_semantic_to_real,
    load_real_labels,
    parse_jfilament_snakes,
)


def write_snakes(path, snake_points):
    lines = ["stretch\t100.0"]
    for points in snake_points:
        lines.extend(["#", "0"])
        for i, (x, y) in enumerate(points):
            lines.append(f"1\t{i}\t{x}\t{y}\t0")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_labeling(path, shape, labels):
    data = {
        "interval": {"min": [0, 0], "max": [shape[1] - 1, shape[0] - 1], "n": 2},
        "labels": labels,
        "colors": {},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_jfilament_parser_reads_multiple_hash_separated_snakes(tmp_path):
    path = tmp_path / "snakes.txt"
    write_snakes(path, [[(1, 2), (3, 4)], [(5, 6), (7, 8), (9, 10)]])

    points = parse_jfilament_snakes(path)

    assert [(p.snake_id, p.point_index, p.x, p.y, p.z_or_slice) for p in points] == [
        (0, 0, 1.0, 2.0, 0.0),
        (0, 1, 3.0, 4.0, 0.0),
        (1, 0, 5.0, 6.0, 0.0),
        (1, 1, 7.0, 8.0, 0.0),
        (1, 2, 9.0, 10.0, 0.0),
    ]


def test_labkit_labeling_maps_names_and_leaves_unannotated_background(tmp_path):
    path = tmp_path / "labels.labeling"
    write_labeling(
        path,
        (4, 5),
        {
            "fibers": [[1, 1], [2, 1]],
            "uncertain_ignore": [[3, 1]],
            "clump": [[4, 3]],
        },
    )

    labels = load_real_labels(path)

    assert labels.image_shape == (4, 5)
    assert labels.label_names == ["fibers", "uncertain_ignore", "clump"]
    assert labels.semantic_mask[0, 0] == 0
    assert labels.semantic_mask[1, 1] == 1
    assert labels.semantic_mask[1, 3] == 255
    assert labels.semantic_mask[3, 4] == 3


def test_uncertain_ignore_precedence_is_independent_of_label_order(tmp_path):
    path = tmp_path / "labels.labeling"
    write_labeling(
        path,
        (2, 2),
        {
            "uncertain_ignore": [[1, 1]],
            "clump": [[1, 1]],
            "fibers": [[1, 1]],
        },
    )

    labels = load_real_labels(path)

    assert labels.masks["fibers"][1, 1]
    assert labels.masks["clump"][1, 1]
    assert labels.masks["uncertain_ignore"][1, 1]
    assert labels.semantic_mask[1, 1] == 255


def test_exported_integer_mask_remaps_uncertain_to_255(tmp_path):
    path = tmp_path / "labels.png"
    Image.fromarray(np.asarray([[0, 1, 2, 3]], dtype=np.uint8)).save(path)

    labels = load_real_labels(path)

    assert labels.semantic_mask.tolist() == [[0, 1, 255, 3]]


def test_real_conversion_flags_non_fibrous_snakes_without_topology_targets(tmp_path):
    image_path = tmp_path / "image.png"
    labels_path = tmp_path / "labels.labeling"
    snakes_path = tmp_path / "snakes.txt"
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(image_path)
    write_labeling(
        labels_path,
        (10, 10),
        {
            "fibers": [[x, 1] for x in range(1, 8)],
            "clump": [[x, 3] for x in range(1, 8)],
            "uncertain_ignore": [[x, 5] for x in range(1, 8)],
        },
    )
    write_snakes(
        snakes_path,
        [
            [(1.5, 1.5), (7.5, 1.5)],
            [(1.5, 3.5), (7.5, 3.5)],
            [(1.5, 5.5), (7.5, 5.5)],
            [(1.5, 8.5), (7.5, 8.5)],
        ],
    )

    sample = build_real_annotation_sample(
        image_path,
        snakes_path,
        labels_path,
        skeleton_radius_px=0.5,
        usable_threshold=0.6,
    )

    statuses = [flag["status"] for flag in sample["metadata"]["snake_quality_flags"]]
    assert statuses == [
        "usable_fibrous_skeleton",
        "flagged_clump",
        "flagged_uncertain_ignore",
        "flagged_background",
    ]
    assert sample["real_skeleton_mask"][1].sum() > 0
    assert sample["real_skeleton_mask"][3].sum() == 0
    assert not np.any(sample["real_skeleton_mask"] & (sample["real_semantic_mask"] != 1))
    assert not np.any(sample["real_skeleton_valid_mask"] & (sample["real_semantic_mask"] == 255))
    forbidden = ("endpoint", "crossing", "junction", "branch", "merge")
    assert not any(any(word in key for word in forbidden) for key in sample)


def test_synthetic_rich_classes_collapse_to_real_compatible_view():
    synthetic = np.asarray([[0, 1, 2, 3, 255]], dtype=np.uint8)

    real = collapse_synthetic_semantic_to_real(synthetic)

    assert real.tolist() == [[0, 1, 1, 3, 255]]


def test_real_conversion_without_snakes_disables_skeleton_supervision(tmp_path):
    image = tmp_path / "image.png"
    labels = tmp_path / "labels.labeling"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image)
    write_labeling(labels, (4, 4), {"fibers": [[1, 1]]})

    sample = build_real_annotation_sample(image, None, labels)

    assert not sample["metadata"]["skeleton_annotation_available"]
    assert not sample["real_skeleton_mask"].any()
    assert not sample["real_skeleton_valid_mask"].any()
