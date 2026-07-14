from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_four_class_context_evaluation import (
    VARIANT,
    metric_row,
    paired_rows,
    viability_decision,
)
from scripts.run_four_class_context_evaluation import require_gpu_one, training_command


def test_four_class_runner_allows_only_physical_gpu_one():
    assert require_gpu_one("1") == "1"
    for value in ("0", "0,1", "1,0", "2", ""):
        with pytest.raises(ValueError, match="GPU 1"):
            require_gpu_one(value)


def test_four_class_training_command_uses_direct_four_class_configuration():
    command = training_command(Namespace(suite_root=Path("/suite")), Path("/out"), 2)
    joined = " ".join(command)
    assert "--model-variant four_class_context_unet" in joined
    assert "--lambda-semantic 1.0" in joined
    assert "--lambda-dice 0.0" in joined
    assert "--lambda-skeleton 0.5" in joined
    assert "--device cuda:0" in joined
    assert "--seed 123" in joined
    assert "--save-checkpoint-every 1" in joined
    assert "--lambda-foreground" not in command
    assert "--lambda-gate" not in command


def test_metric_row_uses_complete_image_uncertainty_denominators_and_independent_skeleton():
    target = np.array([[0, 1], [3, 255]], dtype=np.uint8)
    semantic = np.array([[0, 1], [255, 255]], dtype=np.uint8)
    skeleton_target = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    skeleton_pred = np.array([[0, 1], [1, 1]], dtype=bool)
    score = np.array([[0.0, 0.1], [0.8, 0.9]], dtype=np.float32)
    row = metric_row(type("R", (), {"outer_fold": 0, "sample_id": "x"})(), VARIANT, target, skeleton_target, semantic, score, skeleton_pred)
    assert row["uncertain_recall"] == 1.0
    assert row["uncertain_precision"] == pytest.approx(1 / 2)
    assert row["predicted_fibre_fraction_inside_expert_uncertain"] == 0.0
    assert row["skeleton_recall"] == 1.0
    assert row["skeleton_leakage_into_uncertain"] == pytest.approx(1 / 3)


def test_paired_rows_compare_four_class_against_controls_per_image():
    rows = [
        {"variant": VARIANT, "sample_id": "a", "outer_fold": 0, "status": "available", "fibrous_dice": 0.9},
        {"variant": "U0", "sample_id": "a", "outer_fold": 0, "status": "available", "fibrous_dice": 0.8},
    ]
    paired = paired_rows(rows)
    image = next(row for row in paired if row["scope"] == "image" and row["metric"] == "fibrous_dice")
    macro = next(row for row in paired if row["scope"] == "global_image_macro" and row["metric"] == "fibrous_dice")
    assert image["difference"] == pytest.approx(0.1)
    assert macro["mean_paired_difference"] == pytest.approx(0.1)


def test_viability_decision_requires_requested_thresholds():
    macro = [
        {"scope": "global_image_macro", "variant": "U0", "fibrous_dice": 0.8, "clump_dice": 0.7, "confident_fibrous_recall_0_2px": 0.5, "skeleton_dice": 0.6},
        {"scope": "global_image_macro", "variant": "S", "expert_uncertain_assigned_fibre": 0.253},
        {"scope": "global_image_macro", "variant": "G2_four_way_calibrated_skeleton", "uncertain_recall": 0.0831},
        {"scope": "global_image_macro", "variant": VARIANT, "fibrous_dice": 0.79, "clump_dice": 0.69, "confident_fibrous_recall_0_2px": 0.47, "skeleton_dice": 0.59, "uncertain_recall": 0.1, "expert_uncertain_assigned_fibre": 0.2},
    ]
    assert viability_decision(macro)["decision"] == "viable"
    macro[-1]["uncertain_recall"] = 0.08
    assert viability_decision(macro)["decision"] == "non_viable"
