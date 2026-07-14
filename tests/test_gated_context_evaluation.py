from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_gated_context_evaluation import (
    collapse_diagnostics,
    image_metrics,
    require_gated_maps,
)
from scripts.run_gated_context_evaluation import parse_gpus, training_command


def test_gated_training_command_reuses_b1_schedule_without_legacy_losses():
    args = Namespace(suite_root=Path("/suite"))
    command = training_command(args, Path("/out"), fold=2, epochs=19)
    joined = " ".join(command)
    assert "--model-variant gated_context_unet" in joined
    assert "--epochs 19 --stage1-epochs 4" in joined
    assert "--patches-per-epoch 256" in joined
    assert "--seed 123" in joined
    assert "--lambda-foreground 1.0" in joined
    assert "--lambda-gate 0.5" in joined
    assert "--lambda-morphology 1.0" in joined
    assert "--lambda-joint 1.0" in joined
    assert "--lambda-skeleton 0.5" in joined
    assert "--lambda-uncertainty 0" in joined
    assert "--lambda-uncertain-fibrous 0" in joined
    assert "--lambda-clump-anti-fibrous 0" in joined
    assert "--lambda-clump-anti-skeleton 0" in joined
    assert "--enable-uncertainty-head" not in command


def test_gpu_parser_requires_unique_physical_ids():
    assert parse_gpus("2,5") == ["2", "5"]
    with pytest.raises(ValueError):
        parse_gpus("1,1")


def test_gated_metrics_use_categorical_uncertain_assignment():
    target = np.array([[0, 1], [3, 255]], dtype=np.uint8)
    semantic = np.array([[0, 255], [3, 255]], dtype=np.uint8)
    pred = {
        "semantic_class_map": semantic,
        "fibrous_probability": np.full((2, 2), 0.1, dtype=np.float32),
        "clump_probability": np.full((2, 2), 0.1, dtype=np.float32),
        "skeleton_probability": np.zeros((2, 2), dtype=np.float32),
        "uncertainty_probability": np.array([[0.1, 0.8], [0.2, 0.9]], dtype=np.float32),
    }
    metrics = image_metrics(target, np.zeros_like(target), pred)
    assert metrics["expert_uncertain_recall"] == 1.0
    assert metrics["expert_uncertain_precision"] == 0.5
    assert metrics["expert_fibre_assigned_uncertain"] == 1.0


def test_collapse_diagnostic_reports_class_and_gate_statistics():
    shape = (4, 4)
    pred = {
        "semantic_class_map": np.tile(np.array([0, 1, 3, 255], dtype=np.uint8), (4, 1)),
        "foreground_probability": np.linspace(0.1, 0.9, 16, dtype=np.float32).reshape(shape),
        "quantifiability_probability": np.linspace(0.2, 0.8, 16, dtype=np.float32).reshape(shape),
        "final_uncertain_probability": np.linspace(0.1, 0.3, 16, dtype=np.float32).reshape(shape),
        "raw_centreline_probability": np.linspace(0, 1, 16, dtype=np.float32).reshape(shape),
    }
    for name in (
        "conditional_fibrous_probability", "conditional_clump_probability",
        "final_fibrous_probability", "final_clump_probability",
    ):
        pred[name] = np.linspace(0, 1, 16, dtype=np.float32).reshape(shape)
    row = collapse_diagnostics(Namespace(sample_id="x", outer_fold=0), pred)
    assert row["predicted_uncertain_fraction"] == 0.25
    assert row["collapsed"] is False
    assert row["near_zero_variance_heads"] == ""


def test_gated_probability_inventory_is_required():
    with pytest.raises(ValueError, match="missing gated maps"):
        require_gated_maps(Path("predictions.npz"), {"semantic_class_map": np.zeros((1, 1))})
