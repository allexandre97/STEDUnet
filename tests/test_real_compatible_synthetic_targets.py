import numpy as np

from fibras.synthetic.real_compatible import (
    build_real_compatible_targets,
    validate_real_compatible_targets,
)


def tiny_arrays():
    individual = np.zeros((3, 5), dtype=np.uint8)
    bundle = np.zeros_like(individual)
    clump = np.zeros_like(individual)
    uncertain = np.zeros_like(individual)
    individual[0, 1] = 1
    bundle[0, 2] = 1
    individual[1, 1] = 1
    clump[1, 1] = 1
    bundle[1, 2] = 1
    uncertain[1, 2] = 1
    clump[2, 3] = 1
    return {
        "individual_filament_mask": individual,
        "bundle_mask": bundle,
        "clump_mask": clump,
        "uncertain_ignore_mask": uncertain,
        "filament_centerline_mask": np.asarray(
            [
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
            ],
            dtype=np.uint8,
        ),
        "bundle_axis_mask": np.asarray(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        "endpoint_map": np.ones((3, 5), dtype=np.uint8),
        "projected_crossing_map": np.ones((3, 5), dtype=np.uint8),
        "junction_map": np.ones((3, 5), dtype=np.uint8),
    }


def test_real_compatible_semantic_priority_and_values():
    targets = build_real_compatible_targets(tiny_arrays())

    assert targets["real_compatible_semantic_mask"].tolist() == [
        [0, 1, 1, 0, 0],
        [0, 3, 255, 0, 0],
        [0, 0, 0, 3, 0],
    ]
    assert targets["real_compatible_fibrous_mask"].sum() == 2
    assert targets["real_compatible_clump_mask"].sum() == 2
    assert targets["real_compatible_uncertain_ignore_mask"].sum() == 1
    assert set(np.unique(targets["real_compatible_semantic_mask"])) == {0, 1, 3, 255}


def test_real_compatible_skeleton_uses_only_filament_and_bundle_axes():
    arrays = tiny_arrays()
    targets = build_real_compatible_targets(arrays)

    assert targets["real_compatible_skeleton_mask"].tolist() == [
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert not np.any(
        targets["real_compatible_skeleton_mask"]
        & targets["real_compatible_clump_mask"]
    )
    assert not np.any(
        targets["real_compatible_skeleton_mask"]
        & targets["real_compatible_uncertain_ignore_mask"]
    )


def test_real_compatible_validator_rejects_skeleton_uncertain_overlap():
    arrays = tiny_arrays()
    arrays.update(build_real_compatible_targets(arrays))
    arrays["real_compatible_skeleton_mask"][1, 2] = 1

    errors = validate_real_compatible_targets("sample", arrays, None)

    assert any("skeleton overlaps uncertain_ignore" in error for error in errors)


def test_real_compatible_validator_rejects_synthetic_only_real_roles():
    arrays = tiny_arrays()
    arrays.update(build_real_compatible_targets(arrays))

    errors = validate_real_compatible_targets(
        "sample",
        arrays,
        {"target_roles": {"real_compatible_supervised": ["endpoint_map"]}},
    )

    assert any("synthetic-only arrays" in error for error in errors)
