import numpy as np
import pytest

from scripts.evaluate_real_pilot_baseline import (
    compute_real_pilot_metrics,
    load_model,
    run_tiled_inference,
)


def test_tiled_inference_preserves_image_size():
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def forward(self, image):
            fibrous = image[:, 0]
            background = 1.0 - fibrous
            clump = torch.zeros_like(fibrous)
            return {
                "semantic_logits": torch.stack([background, fibrous, clump], dim=1),
                "skeleton_logits": image,
            }

    image = np.zeros((17, 19), dtype=np.uint8)
    image[4:12, 5:15] = 255

    out = run_tiled_inference(
        TinyModel(),
        image,
        torch.device("cpu"),
        torch,
        patch_size=8,
        batch_size=3,
        overlap=2,
    )

    assert out["semantic_class_map"].shape == image.shape
    assert out["fibrous_probability"].shape == image.shape
    assert out["clump_probability"].shape == image.shape
    assert out["skeleton_probability"].shape == image.shape
    assert out["skeleton_mask_0_75"].shape == image.shape


def test_load_model_accepts_uncertainty_head_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import SmallUNet

    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": SmallUNet(uncertainty_head=True).state_dict(),
            "config": {"enable_uncertainty_head": True},
        },
        checkpoint,
    )

    model, checkpoint_path, device = load_model(checkpoint, None, "cpu", torch)

    assert checkpoint_path == checkpoint
    assert str(device) == "cpu"
    assert model.uncertainty_head is not None


def test_load_model_accepts_context_unet_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import ContextUNet

    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": ContextUNet(context_module="aspp", uncertainty_head=True).state_dict(),
            "config": {
                "model_variant": "context_unet",
                "context_module": "aspp",
                "aspp_dilations": "1,2,4,8",
                "enable_uncertainty_head": True,
            },
        },
        checkpoint,
    )

    model, _, _ = load_model(checkpoint, None, "cpu", torch)

    assert model.__class__.__name__ == "ContextUNet"
    assert model.uncertainty_head is not None


def test_uncertainty_gating_can_suppress_fibrous_and_skeleton():
    torch = pytest.importorskip("torch")

    class UncertainFibrousModel(torch.nn.Module):
        def forward(self, image):
            b, _, h, w = image.shape
            semantic = torch.zeros(b, 3, h, w, device=image.device)
            semantic[:, 1] = 10.0
            return {
                "semantic_logits": semantic,
                "skeleton_logits": torch.full((b, 1, h, w), 10.0, device=image.device),
                "uncertainty_logits": torch.full((b, 1, h, w), 10.0, device=image.device),
            }

    out = run_tiled_inference(
        UncertainFibrousModel(),
        np.zeros((8, 8), dtype=np.uint8),
        torch.device("cpu"),
        torch,
        patch_size=8,
        overlap=2,
        uncertainty_gating="suppress_both",
        uncertainty_threshold=0.5,
    )

    assert not np.any(out["semantic_class_map"] == 1)
    assert float(out["skeleton_probability"].max()) == 0.0
    assert float(out["uncertainty_probability"].min()) > 0.5


def test_uncertain_ignore_pixels_are_ignored_in_real_pilot_metrics():
    target = np.zeros((4, 4), dtype=np.uint8)
    target[0, 0] = 255
    target[1, 1] = 1
    pred = np.zeros_like(target)
    pred[0, 0] = 1
    pred[1, 1] = 1
    skeleton = np.zeros_like(target)
    skeleton[1, 1] = 1
    skeleton_probability = np.zeros((4, 4), dtype=np.float32)
    skeleton_probability[0, 0] = 1.0
    skeleton_probability[1, 1] = 1.0

    metrics = compute_real_pilot_metrics(target, skeleton, pred, skeleton_probability)

    assert metrics["fibrous_dice"] == 1.0
    assert metrics["fibrous_precision"] == 1.0
    assert metrics["fibrous_recall"] == 1.0
    assert metrics["skeleton_metrics_by_threshold"]["0.75"]["dice"] == 1.0
    assert metrics["predicted_skeleton_pixels"] == 1


def test_clump_dice_is_not_applicable_without_target_pixels():
    target = np.zeros((4, 4), dtype=np.uint8)
    pred = np.zeros_like(target)
    pred[1, 1] = 3

    metrics = compute_real_pilot_metrics(target, np.zeros_like(target), pred, np.zeros((4, 4), dtype=np.float32))

    assert metrics["clump_dice"] == "not_applicable"
    assert metrics["clump_target_pixels"] == 0


def test_skeleton_threshold_075_is_accepted_and_recorded():
    target = np.zeros((3, 3), dtype=np.uint8)
    target[1, 1] = 1
    skeleton = np.zeros_like(target)
    skeleton[1, 1] = 1
    probability = np.zeros((3, 3), dtype=np.float32)
    probability[1, 1] = 0.8

    metrics = compute_real_pilot_metrics(target, skeleton, target, probability, skeleton_threshold=0.75)

    assert metrics["skeleton_threshold"] == 0.75
    assert set(metrics["skeleton_metrics_by_threshold"]) == {"0.5", "0.75", "0.85"}
    assert metrics["skeleton_metrics_by_threshold"]["0.75"]["recall"] == 1.0


def test_real_pilot_metrics_do_not_emit_topology_metrics():
    target = np.zeros((3, 3), dtype=np.uint8)
    metrics = compute_real_pilot_metrics(target, target, target, np.zeros((3, 3), dtype=np.float32))
    keys = list(flat_keys(metrics))

    forbidden = ("endpoint", "crossing", "junction", "branch", "merge")
    assert not any(any(word in key for word in forbidden) for key in keys)


def flat_keys(value, prefix=""):
    if isinstance(value, dict):
        for key, nested in value.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            yield from flat_keys(nested, full)
    else:
        yield prefix
