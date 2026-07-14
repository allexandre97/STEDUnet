from argparse import Namespace
from pathlib import Path

import pytest
import numpy as np
import torch

from scripts.analyze_gated_context_recalibration import probability_values, summarize_groups
from scripts.run_gated_context_recalibration import require_gpu_one, training_command
from fibras.training.schema08_baseline import append_gated_probability_moments, summarize_probability_moments


def test_g2_configuration_propagates_only_confirmatory_weights():
    command = training_command(Namespace(suite_root=Path("/suite")), Path("/out"), 3, 19)
    joined = " ".join(command)
    assert "--lambda-foreground 1.5" in joined
    assert "--lambda-gate 1.0" in joined
    assert "--lambda-morphology 1.0" in joined
    assert "--lambda-joint 0.5" in joined
    assert "--lambda-skeleton 0.5" in joined
    assert "--device cuda:0" in joined
    assert "--seed 123" in joined
    assert "--save-checkpoint-every 1" in joined


def test_g2_allows_only_physical_gpu_one():
    assert require_gpu_one("1") == "1"
    for invalid in ("0", "0,1", "1,0", "2", ""):
        with pytest.raises(ValueError, match="GPU 1"):
            require_gpu_one(invalid)


def test_g2_command_is_resumable_and_has_no_automatic_device_selection():
    command = training_command(Namespace(suite_root=Path("/suite")), Path("/out"), 0, 2)
    assert command[command.index("--device") + 1] == "cuda:0"
    assert "auto" not in command
    assert "--init-checkpoint" in command


def test_g1_g2_probability_comparison_uses_one_minus_gate_consistently():
    pred = {"quantifiability_probability": np.array([[0.2, 0.8]], dtype=np.float32)}
    np.testing.assert_allclose(probability_values(pred, "one_minus_g_probability"), [[0.8, 0.2]])


def test_macro_summary_keeps_complete_image_denominators_per_image():
    rows = [
        {"variant": "G2", "outer_fold": 0, "sample_id": "a", "status": "available", "uncertain_precision": 0.5},
        {"variant": "G2", "outer_fold": 0, "sample_id": "b", "status": "available", "uncertain_precision": 0.25},
    ]
    summary = summarize_groups(rows, ("variant",), "global_image_macro")[0]
    assert summary["uncertain_precision"] == 0.375


def test_gated_probability_moments_are_recorded_without_storing_pixel_lists():
    outputs = {name: torch.tensor([[[[0.2, 0.8]]]]) for name in (
        "foreground_probability", "quantifiability_probability", "final_background_probability",
        "final_fibrous_probability", "final_clump_probability", "final_uncertain_probability",
    )}
    moments = {}
    append_gated_probability_moments(
        moments, outputs, 0, torch.tensor([[0, 1]]), torch.tensor([[False, False]])
    )
    summary = summarize_probability_moments(moments)
    assert summary["target_background"]["foreground_probability"]["mean"] == pytest.approx(0.2)
    assert summary["target_fibrous"]["foreground_probability"]["mean"] == pytest.approx(0.8)
