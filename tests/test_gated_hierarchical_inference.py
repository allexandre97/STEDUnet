from argparse import Namespace

import numpy as np

from scripts.analyze_gated_hierarchical_inference import (
    SEMANTIC_GRID,
    complete_metrics,
    hierarchical_classes,
    raw_centreline_skeleton,
    select_semantic_thresholds,
    validation_records,
)


def test_hierarchical_class_assignment_respects_decision_order():
    pred = {
        "foreground_probability": np.array([[0.2, 0.8, 0.8, 0.8]]),
        "quantifiability_probability": np.array([[0.9, 0.2, 0.8, 0.8]]),
        "conditional_fibrous_probability": np.array([[0.9, 0.9, 0.6, 0.4]]),
        "conditional_clump_probability": np.array([[0.1, 0.1, 0.4, 0.6]]),
    }
    result = hierarchical_classes(pred, 0.5, 0.5)
    np.testing.assert_array_equal(result, [[0, 255, 1, 3]])


def test_threshold_selection_inputs_are_validation_only():
    records = [
        Namespace(outer_fold=2, partition="validation", sample_id="validation"),
        Namespace(outer_fold=2, partition="test", sample_id="outer_test"),
        Namespace(outer_fold=1, partition="validation", sample_id="other_fold"),
    ]
    assert [record.sample_id for record in validation_records(records, 2)] == ["validation"]


def test_no_valid_operating_point_is_explicit():
    shape = (SEMANTIC_GRID.size, SEMANTIC_GRID.size)
    metrics = {
        "uncertain_recall": np.zeros(shape), "fibrous_dice": np.zeros(shape),
        "clump_dice": np.zeros(shape),
    }
    selected = select_semantic_thresholds(
        metrics, np.zeros(shape, dtype=bool),
        {"fibrous_dice": 0.8, "clump_dice": 0.7, "confident_fibrous_recall_0_2px": 0.6},
    )
    assert selected["status"] == "no_valid_operating_point"
    assert selected["selected_from"] == "inner_validation"


def test_raw_centreline_is_masked_by_hierarchical_confident_fibre():
    semantic = np.array([[1, 255, 3, 0]], dtype=np.uint8)
    raw = np.array([[0.8, 0.9, 0.9, 0.9]])
    np.testing.assert_array_equal(raw_centreline_skeleton(semantic, raw, 0.75), [[True, False, False, False]])


def test_complete_image_assignment_denominators_include_every_predicted_uncertain_pixel():
    target = np.array([[255, 0], [1, 3]], dtype=np.uint8)
    semantic = np.full((2, 2), 255, dtype=np.uint8)
    metrics = complete_metrics(target, np.zeros_like(target), semantic)
    assert metrics["uncertain_precision"] == 0.25
    assert metrics["uncertain_recall"] == 1.0
    assert metrics["expert_confident_fibre_assigned_uncertain"] == 1.0
    assert metrics["expert_background_assigned_uncertain"] == 1.0
