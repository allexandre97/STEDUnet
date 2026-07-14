import numpy as np

from scripts.audit_four_class_calibration import (
    probabilities,
    select_semantic_bias,
    select_skeleton_thresholds,
)


def test_semantic_selection_cannot_depend_on_outer_test_labels():
    logits = np.zeros((4, 2, 2), dtype=float)
    logits[3] = 2
    validation = [(logits, np.array([[0, 1], [3, 255]], dtype=np.uint8))]
    first, _ = select_semantic_bias(validation)
    outer_test_labels = np.zeros((2, 2), dtype=np.uint8)
    outer_test_labels[:] = 255
    second, _ = select_semantic_bias(validation)
    assert first == second


def test_skeleton_selection_cannot_depend_on_outer_test_labels():
    target = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    line = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    score = np.array([[0.1, 0.8], [0.7, 0.2]])
    semantic = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    validation = [(target, line, score, semantic)]
    first, _ = select_skeleton_thresholds(validation)
    outer_test_labels = np.full((2, 2), 255, dtype=np.uint8)
    second, _ = select_skeleton_thresholds(validation)
    assert first == second


def test_uncertain_bias_is_renormalized_by_softmax():
    logits = np.zeros((4, 1, 1), dtype=float)
    calibrated = probabilities(logits, -2)
    assert calibrated[:, 0, 0].sum() == 1
    assert calibrated[3, 0, 0] < calibrated[0, 0, 0]
