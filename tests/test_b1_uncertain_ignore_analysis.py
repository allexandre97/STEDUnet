from __future__ import annotations

import numpy as np

from scripts.analyze_b1_uncertain_ignore import (
    ImageRecord,
    auroc,
    average_precision,
    discrimination_analysis,
    load_probabilities,
    pooled_abstention_metrics,
    select_threshold,
)


def test_uncertainty_rank_metrics_are_discriminative() -> None:
    y = np.array([False, False, True, True])
    score = np.array([0.1, 0.2, 0.8, 0.9])

    assert auroc(y, score) == 1.0
    assert average_precision(y, score) == 1.0


def test_load_probabilities_derives_background_probability(tmp_path) -> None:
    path = tmp_path / "predictions.npz"
    np.savez(
        path,
        fibrous_probability=np.array([[0.2, 0.7]], dtype=np.float32),
        clump_probability=np.array([[0.3, 0.1]], dtype=np.float32),
        skeleton_probability=np.array([[0.4, 0.8]], dtype=np.float32),
        uncertainty_probability=np.array([[0.1, 0.9]], dtype=np.float32),
    )

    pred = load_probabilities(path)

    np.testing.assert_allclose(pred["background_probability"], [[0.5, 0.2]], atol=1e-6)
    np.testing.assert_allclose(pred["max_semantic_probability"], [[0.5, 0.7]], atol=1e-6)


def write_predictions(path, fibrous, skeleton, uncertainty, clump=None) -> None:
    fibrous = np.asarray(fibrous, dtype=np.float32)
    np.savez(
        path,
        fibrous_probability=fibrous,
        clump_probability=np.zeros_like(fibrous) if clump is None else np.asarray(clump, dtype=np.float32),
        skeleton_probability=np.asarray(skeleton, dtype=np.float32),
        uncertainty_probability=np.asarray(uncertainty, dtype=np.float32),
    )


def test_rejected_targets_remain_complete_image_false_negatives(tmp_path) -> None:
    path = tmp_path / "predictions.npz"
    write_predictions(path, [[0.9, 0.1, 0.1, 0.9, 0.1]], [[0.9, 0.1, 0.1, 0.9, 0.1]], [[0.9, 0.1, 0.1, 0.1, 0.1]])
    record = ImageRecord("img", 0, "test", "unused", None, "unused")
    samples = {
        "img": {
            "real_semantic_mask": np.array([[1, 0, 0, 1, 0]], dtype=np.uint8),
            "real_skeleton_mask": np.array([[1, 0, 0, 1, 0]], dtype=np.uint8),
        }
    }

    metrics = pooled_abstention_metrics(
        [record],
        samples,
        {"img": path},
        {"uncertainty_threshold": 0.5, "confidence_threshold": 0.0},
    )

    assert metrics["all_pixel_rejected_fraction"] == 0.2
    assert metrics["target_fibres_abstained_fraction"] == 0.5
    assert metrics["predicted_fibres_rejected_fraction"] == 0.5
    assert metrics["fibrous_recall_full_image_after_abstention"] == 0.5
    assert metrics["fibrous_dice_full_image_after_abstention"] == 2 / 3
    assert metrics["fibrous_dice_retained_pixels_selective"] == 1.0
    assert metrics["skeleton_dice_full_image_after_abstention"] == 2 / 3
    assert metrics["skeleton_recovery_2px_after_abstention"] == 0.5


def test_threshold_selection_obeys_validation_constraints(tmp_path) -> None:
    path = tmp_path / "predictions.npz"
    write_predictions(path, [[0.9, 0.9, 0.1]], [[0.9, 0.1, 0.1]], [[0.1, 0.8, 0.1]])
    record = ImageRecord("img", 0, "validation", "unused", None, "unused")
    samples = {"img": {
        "real_semantic_mask": np.array([[1, 255, 0]], dtype=np.uint8),
        "real_skeleton_mask": np.array([[1, 0, 0]], dtype=np.uint8),
    }}

    selected = select_threshold("uncertainty", [record], samples, {"img": path})

    assert selected["threshold_status"] == "operational_threshold"
    assert selected["validation_target_fibres_abstained_fraction"] == 0.0
    assert selected["validation_fibrous_dice"] == selected["validation_baseline_fibrous_dice"]
    assert selected["validation_skeleton_dice"] == selected["validation_baseline_skeleton_dice"]
    assert selected["validation_high_conf_uncertain_reduction"] == 1


def test_threshold_selection_records_no_valid_threshold(tmp_path) -> None:
    path = tmp_path / "predictions.npz"
    write_predictions(path, [[0.9, 0.9]], [[0.9, 0.1]], [[0.8, 0.8]])
    record = ImageRecord("img", 0, "validation", "unused", None, "unused")
    samples = {"img": {
        "real_semantic_mask": np.array([[1, 255]], dtype=np.uint8),
        "real_skeleton_mask": np.array([[1, 0]], dtype=np.uint8),
    }}

    selected = select_threshold("uncertainty", [record], samples, {"img": path})

    assert selected["threshold_status"] == "no_operational_threshold"
    assert selected["uncertainty_threshold"] == "not_applicable"


def test_foreground_conditional_discrimination_excludes_easy_background(tmp_path) -> None:
    path = tmp_path / "predictions.npz"
    write_predictions(path, [[0.1, 0.7, 0.1, 0.7]], [[0, 0, 0, 0]], [[0.0, 0.1, 0.2, 0.15]], clump=[[0.1, 0.1, 0.7, 0.1]])
    record = ImageRecord("img", 2, "test", "unused", None, "unused")
    samples = {"img": {"real_semantic_mask": np.array([[0, 1, 3, 255]], dtype=np.uint8)}}

    rows = discrimination_analysis([record], samples, {"img": path})
    pooled = {row["comparison"]: row for row in rows if row["scope"] == "global_pooled"}

    assert pooled["uncertain_vs_fibrous_clump"]["auroc"] == 0.5
    assert pooled["uncertain_vs_background"]["auroc"] == 1.0
    assert pooled["uncertain_vs_fibrous_clump"]["prevalence"] == 1 / 3
