from collections import Counter, defaultdict

from fibras.training.cross_validation import (
    FOLD_SCHEMA_VERSION,
    build_grouped_folds,
    validate_grouped_folds,
)


def inventory_rows():
    rows = []
    groups = [
        ("prep_a", "AD", "3R", "5", 2),
        ("prep_b", "AD", "4R", "7", 2),
        ("prep_c", "PID", "3R", "5", 2),
        ("prep_d", "PSP", "4R", "10", 2),
        ("prep_e", "CBD", "4R", "5", 2),
    ]
    for group, disease, isoform, div, count in groups:
        for index in range(count):
            rows.append({
                "sample_id": f"{group}_{index}",
                "image_identity": f"{group}_{index}.tif",
                "split_group_id": group,
                "validation_status": "valid",
                "disease": disease,
                "tau_isoform": isoform,
                "div": div,
                "shape_y": "10",
                "shape_x": "10",
                "fibrous_tau_pixels": str(5 + index),
                "clump_pixels": str(index),
                "uncertain_ignore_pixels": str(2 * index),
            })
    return rows


def test_fixed_folds_have_no_group_or_image_leakage():
    inventory = inventory_rows()
    folds = build_grouped_folds(inventory, fold_count=5, seed=7)

    assert not validate_grouped_folds(folds, inventory, 5)
    assert {row["fold_schema_version"] for row in folds} == {FOLD_SCHEMA_VERSION}
    assert Counter(row["sample_id"] for row in folds if row["partition"] == "test") == Counter(
        {row["sample_id"]: 1 for row in inventory}
    )
    for fold in range(5):
        partitions = defaultdict(set)
        for row in folds:
            if int(row["outer_fold"]) == fold:
                partitions[row["split_group_id"]].add(row["partition"])
        assert all(len(value) == 1 for value in partitions.values())


def test_fold_generation_is_deterministic():
    first = build_grouped_folds(inventory_rows(), fold_count=5, seed=7)
    second = build_grouped_folds(inventory_rows(), fold_count=5, seed=7)
    assert first == second
