from __future__ import annotations

import numpy as np
import torch

from scripts.uncertain_ignore_nested_metrics import (
    complete_image_metrics,
    select_h_operating_point,
    select_s_candidate,
    summarize_images,
)
from scripts.audit_uncertain_ignore_nested_cv import discrimination_row, macro_discrimination
from fibras.training.schema08_baseline import (
    compute_loss,
    stratified_rejection_loss,
    uncertain_fibrous_suppression_loss,
)


def test_suppression_loss_is_zero_below_tau() -> None:
    logits = torch.tensor([[[[0.0]], [[-0.4]], [[0.0]]]])
    uncertain = torch.ones((1, 1, 1, 1))

    loss = uncertain_fibrous_suppression_loss(logits, uncertain, tau=0.5)

    assert loss.item() == 0.0


def test_uncertain_pixels_remain_excluded_from_semantic_and_skeleton_losses() -> None:
    batch = {
        "semantic": torch.tensor([[[1, -100]]]),
        "skeleton": torch.tensor([[[[1.0, 1.0]]]]),
        "uncertain": torch.tensor([[[[0.0, 1.0]]]]),
        "valid": torch.tensor([[[[1.0, 0.0]]]]),
        "skeleton_valid": torch.tensor([[[[1.0, 0.0]]]]),
    }
    first = {
        "semantic_logits": torch.tensor([[[[0.0, 20.0]], [[2.0, -20.0]], [[0.0, 0.0]]]]),
        "skeleton_logits": torch.tensor([[[[2.0, -20.0]]]]),
    }
    second = {
        "semantic_logits": torch.tensor([[[[0.0, -20.0]], [[2.0, 20.0]], [[0.0, 0.0]]]]),
        "skeleton_logits": torch.tensor([[[[2.0, 20.0]]]]),
    }

    _, first_parts = compute_loss(first, batch)
    _, second_parts = compute_loss(second, batch)

    assert first_parts["semantic_loss"] == second_parts["semantic_loss"]
    assert first_parts["skeleton_loss"] == second_parts["skeleton_loss"]


def test_stratified_loss_uses_stratum_means_not_background_pixel_counts() -> None:
    base_semantic = torch.tensor([[[-100, 1, 2, 1, 0]]])
    base_target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0, 0.0]]]])
    base_distance = torch.tensor([[[[0.0, 2.0, 3.0, 30.0, 30.0]]]])
    base_logits = torch.tensor([[[[1.0, -1.0, -0.5, 0.5, 2.0]]]])
    batch = {"semantic": base_semantic, "uncertain": base_target, "uncertain_distance": base_distance}
    loss, parts = stratified_rejection_loss(base_logits, batch)

    repeated_batch = {
        "semantic": torch.cat([base_semantic, torch.zeros((1, 1, 20), dtype=torch.long)], dim=2),
        "uncertain": torch.cat([base_target, torch.zeros((1, 1, 1, 20))], dim=3),
        "uncertain_distance": torch.cat([base_distance, torch.full((1, 1, 1, 20), 30.0)], dim=3),
    }
    repeated_logits = torch.cat([base_logits, torch.full((1, 1, 1, 20), 2.0)], dim=3)
    repeated_loss, _ = stratified_rejection_loss(repeated_logits, repeated_batch)

    assert set(parts) == {
        "rejection_positive_loss", "rejection_near_fibrous_loss", "rejection_near_clump_loss",
        "rejection_far_tau_loss", "rejection_background_loss",
    }
    torch.testing.assert_close(loss, repeated_loss)


def test_nested_selection_uses_only_acceptable_validation_candidates() -> None:
    baseline = {
        "fibrous_dice": 0.80, "skeleton_dice_0.75": 0.60,
        "uncertain_fibrous_probability_ge_0.7_fraction": 0.20,
        "uncertain_fibrous_probability_ge_0.9_fraction": 0.10,
    }
    candidates = [
        {"lambda_uncertain_fibrous": 0.01, "fibrous_dice": 0.795, "skeleton_dice_0.75": 0.595,
         "uncertain_fibrous_probability_ge_0.7_fraction": 0.15, "uncertain_fibrous_probability_ge_0.9_fraction": 0.08},
        {"lambda_uncertain_fibrous": 0.05, "fibrous_dice": 0.78, "skeleton_dice_0.75": 0.60,
         "uncertain_fibrous_probability_ge_0.7_fraction": 0.05, "uncertain_fibrous_probability_ge_0.9_fraction": 0.01},
    ]

    selected = select_s_candidate(baseline, candidates)

    assert selected["selection"] == "S"
    assert selected["lambda_uncertain_fibrous"] == 0.01


def test_complete_image_abstention_keeps_rejected_targets_in_denominators() -> None:
    target = np.array([[1, 0, 0, 1, 0]], dtype=np.uint8)
    skeleton = np.array([[1, 0, 0, 1, 0]], dtype=np.uint8)
    pred = {
        "semantic_class_map": np.array([[1, 0, 0, 1, 0]], dtype=np.uint8),
        "fibrous_probability": np.array([[0.9, 0.1, 0.1, 0.9, 0.1]]),
        "skeleton_probability": np.array([[0.9, 0.1, 0.1, 0.9, 0.1]]),
        "uncertainty_probability": np.array([[0.9, 0.1, 0.1, 0.1, 0.1]]),
    }

    metrics = complete_image_metrics(target, skeleton, pred, rejection_threshold=0.5)

    assert metrics["fibrous_recall"] == 0.5
    assert metrics["fibrous_dice"] == 2 / 3
    assert metrics["skeleton_dice_0.75"] == 2 / 3
    assert metrics["target_skeleton_recovered_2px"] == 0.5
    assert metrics["rejected_target_fibres_fraction"] == 0.5


def test_boundary_bands_measure_local_fibre_recall() -> None:
    target = np.zeros((1, 27), dtype=np.uint8)
    target[0, 0] = 255
    target[0, [1, 3, 7, 15, 25]] = 1
    semantic = np.zeros_like(target)
    semantic[0, [1, 7, 25]] = 1
    pred = {
        "semantic_class_map": semantic,
        "fibrous_probability": (semantic == 1).astype(float),
        "skeleton_probability": np.zeros_like(target, dtype=float),
        "uncertainty_probability": np.zeros_like(target, dtype=float),
    }

    metrics = complete_image_metrics(target, np.zeros_like(target), pred)

    assert metrics["confident_fibrous_recall_0_2px"] == 1.0
    assert metrics["confident_fibrous_recall_2_5px"] == 0.0
    assert metrics["confident_fibrous_recall_5_10px"] == 1.0
    assert metrics["confident_fibrous_recall_10_20px"] == 0.0
    assert metrics["confident_fibrous_recall_over_20px"] == 1.0


def test_no_acceptable_candidates_fall_back_to_u0() -> None:
    baseline = {
        "fibrous_dice": 0.8, "skeleton_dice_0.75": 0.6,
        "uncertain_fibrous_probability_ge_0.7_fraction": 0.2,
        "uncertain_fibrous_probability_ge_0.9_fraction": 0.1,
    }
    bad_s = [{
        "fibrous_dice": 0.78, "skeleton_dice_0.75": 0.6,
        "uncertain_fibrous_probability_ge_0.7_fraction": 0.1,
        "uncertain_fibrous_probability_ge_0.9_fraction": 0.05,
    }]
    bad_h = [{
        "fibrous_dice": 0.8, "skeleton_dice_0.75": 0.6,
        "rejected_target_fibres_fraction_pooled": 0.03,
        "expert_uncertain_recall_pooled": 0.5,
        "uncertain_fibrous_probability_ge_0.7_fraction": 0.1,
    }]

    assert select_s_candidate(baseline, bad_s)["selection"] == "U0"
    assert select_h_operating_point(baseline, bad_h)["selection"] == "U0"


def test_clump_dice_excludes_non_applicable_images() -> None:
    pred = {
        "semantic_class_map": np.zeros((1, 2), dtype=np.uint8),
        "fibrous_probability": np.zeros((1, 2)), "skeleton_probability": np.zeros((1, 2)),
        "uncertainty_probability": np.zeros((1, 2)),
    }
    empty = complete_image_metrics(np.zeros((1, 2), dtype=np.uint8), np.zeros((1, 2)), pred)
    target = np.array([[3, 0]], dtype=np.uint8)
    clump_pred = {**pred, "semantic_class_map": target.copy()}
    applicable = complete_image_metrics(target, np.zeros((1, 2)), clump_pred)

    summary = summarize_images([empty, applicable])

    assert empty["clump_dice"] == "not_applicable"
    assert summary["clump_dice"] == 1.0
    assert summary["clump_applicable_images"] == 1


def test_macro_aggregation_weights_images_not_unequal_fold_means() -> None:
    rows = [{
        "fibrous_dice": value, "clump_applicable": False,
        "_rejected_pixels": 0, "_all_pixels": 1, "_rejected_target_fibres": 0,
        "_target_fibres": 1, "_rejected_uncertain_pixels": 0, "_uncertain_pixels": 1,
        "_clump_tp": 0, "_predicted_clump": 0, "_target_clump": 0,
    } for value in (0.0, 1.0, 1.0)]
    assert summarize_images(rows)["fibrous_dice"] == 2 / 3


def test_foreground_conditional_and_spatial_discrimination() -> None:
    y = np.array([1, 1, 0, 0], dtype=bool)
    score = np.array([0.9, 0.8, 0.2, 0.1])
    row = discrimination_row("H", "uncertain_vs_fibrous_clump", "image", "x", 0, y, score)
    part = [(type("R", (), {"outer_fold": 0, "sample_id": "x"})(), y, score)]
    spatial = macro_discrimination("H", "within_20px", "global_macro", "all", "all", part)

    assert row["auroc"] == 1.0
    assert row["average_precision"] == 1.0
    assert spatial["auroc"] == 1.0
    assert spatial["prevalence"] == 0.5
