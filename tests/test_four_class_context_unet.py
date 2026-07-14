import numpy as np
import pytest


torch = pytest.importorskip("torch")

from fibras.training.schema08_baseline import (
    ContextUNet,
    FourClassContextUNet,
    class_balanced_cross_entropy,
    compute_four_class_loss,
    four_class_target,
    load_initial_checkpoint,
    pixel_cross_entropy,
)
from scripts.evaluate_real_pilot_baseline import load_model, run_tiled_inference, write_outputs


def training_batch():
    return {
        "semantic": torch.tensor([[[0, 1, 2, -100]]]),
        "skeleton": torch.tensor([[[[0.0, 1.0, 1.0, 1.0]]]]),
        "uncertain": torch.tensor([[[[0.0, 0.0, 0.0, 1.0]]]]),
        "valid": torch.tensor([[[[1.0, 1.0, 1.0, 0.0]]]]),
        "skeleton_valid": torch.tensor([[[[1.0, 1.0, 1.0, 0.0]]]]),
    }


def model_outputs(logits=None, centreline=None):
    semantic = torch.zeros(1, 4, 1, 4) if logits is None else logits
    line = torch.zeros(1, 1, 1, 4) if centreline is None else centreline
    return {
        "semantic_logits": semantic,
        "raw_centreline_logits": line,
        "semantic_probabilities": torch.softmax(semantic, dim=1),
        "centreline_probability": torch.sigmoid(line),
    }


def test_probabilities_normalize_and_are_bounded():
    output = FourClassContextUNet(base_channels=4, aspp_dilations=[1, 2])(
        torch.randn(2, 1, 32, 32)
    )
    probabilities = output["semantic_probabilities"]
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones_like(probabilities[:, 0]))
    assert bool(((probabilities >= 0) & (probabilities <= 1)).all())
    assert bool(((output["centreline_probability"] >= 0) &
                 (output["centreline_probability"] <= 1)).all())


def test_uncertain_target_maps_to_fourth_class():
    target, annotated = four_class_target(training_batch())
    assert target.tolist() == [[[0, 1, 2, 3]]]
    assert bool(annotated.all())


def test_class_balanced_loss_is_not_changed_by_background_abundance():
    logits = torch.tensor([[[[0.0, 0.0]], [[0.0, 2.0]], [[0.0, 0.0]], [[0.0, 0.0]]]])
    target = torch.tensor([[[0, 1]]])
    annotated = torch.ones_like(target, dtype=torch.bool)
    loss = class_balanced_cross_entropy(logits, target, annotated)
    expanded_logits = torch.cat([logits, logits[:, :, :, :1].expand(-1, -1, -1, 100)], dim=3)
    expanded_target = torch.cat([target, torch.zeros(1, 1, 100, dtype=torch.long)], dim=2)
    torch.testing.assert_close(
        loss, class_balanced_cross_entropy(expanded_logits, expanded_target, torch.ones_like(expanded_target, dtype=torch.bool))
    )


def test_class_balanced_loss_skips_absent_classes():
    logits = torch.tensor([[[[2.0]], [[0.0]], [[0.0]], [[0.0]]]])
    target = torch.zeros(1, 1, 1, dtype=torch.long)
    actual = class_balanced_cross_entropy(logits, target, torch.ones_like(target, dtype=torch.bool))
    expected = torch.nn.functional.cross_entropy(logits, target)
    torch.testing.assert_close(actual, expected)


def test_complete_centreline_loss_uses_background_and_clump_negatives():
    before = compute_four_class_loss(model_outputs(), training_batch())[1]["skeleton_loss"]
    changed = torch.tensor([[[[100.0, 0.0, 100.0, 0.0]]]])
    after = compute_four_class_loss(model_outputs(centreline=changed), training_batch())[1]["skeleton_loss"]
    assert after > before


def test_centreline_loss_still_excludes_uncertain_pixels():
    before = compute_four_class_loss(model_outputs(), training_batch())[1]["skeleton_loss"]
    changed = torch.tensor([[[[0.0, 0.0, 0.0, 100.0]]]])
    after = compute_four_class_loss(model_outputs(centreline=changed), training_batch())[1]["skeleton_loss"]
    assert after == before


def test_legacy_fibre_only_centreline_mask_is_explicit():
    before = compute_four_class_loss(
        model_outputs(), training_batch(), skeleton_mask_name="legacy_fibre_only"
    )[1]["skeleton_loss"]
    changed = torch.tensor([[[[100.0, 0.0, 100.0, 0.0]]]])
    after = compute_four_class_loss(
        model_outputs(centreline=changed), training_batch(), skeleton_mask_name="legacy_fibre_only"
    )[1]["skeleton_loss"]
    assert after == before


def test_pixel_cross_entropy_matches_manual_calculation():
    logits = torch.tensor([[[[2.0, 0.0]], [[0.0, 3.0]], [[-1.0, 1.0]], [[0.5, -2.0]]]])
    target = torch.tensor([[[0, 1]]])
    annotated = torch.ones_like(target, dtype=torch.bool)
    actual = pixel_cross_entropy(logits, target, annotated)
    expected = -(torch.log_softmax(logits, dim=1)[0, 0, 0, 0] + torch.log_softmax(logits, dim=1)[0, 1, 0, 1]) / 2
    torch.testing.assert_close(actual, expected)


def test_cpu_one_batch_forward_backward_has_finite_gradients():
    model = FourClassContextUNet(base_channels=4, aspp_dilations=[1, 2])
    batch = {
        "semantic": torch.zeros(1, 16, 16, dtype=torch.long),
        "skeleton": torch.zeros(1, 1, 16, 16),
        "uncertain": torch.zeros(1, 1, 16, 16),
        "valid": torch.ones(1, 1, 16, 16),
        "skeleton_valid": torch.ones(1, 1, 16, 16),
    }
    batch["semantic"][:, 2:8, 2:8] = 1
    batch["semantic"][:, 8:14, 2:8] = 2
    batch["semantic"][:, 2:8, 8:14] = -100
    batch["uncertain"][:, :, 2:8, 8:14] = 1
    outputs = model(torch.randn(1, 1, 16, 16))
    loss, parts = compute_four_class_loss(outputs, batch, lambda_dice=0.2)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in parts.values())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_context_checkpoint_copies_semantic_channels_and_skeleton_head(tmp_path):
    source = ContextUNet(base_channels=4, context_module="aspp", aspp_dilations=[1, 2])
    path = tmp_path / "context.pt"
    torch.save({"model_state_dict": source.state_dict()}, path)
    target = FourClassContextUNet(base_channels=4, aspp_dilations=[1, 2])
    report = load_initial_checkpoint(target, str(path), torch.device("cpu"))
    for name, value in source.state_dict().items():
        if name.startswith(("enc", "context.", "up", "dec")):
            torch.testing.assert_close(target.state_dict()[name], value)
    torch.testing.assert_close(target.semantic_head.weight[:3], source.semantic_head.weight)
    torch.testing.assert_close(target.semantic_head.bias[:3], source.semantic_head.bias)
    torch.testing.assert_close(target.centreline_head.weight, source.skeleton_head.weight)
    torch.testing.assert_close(target.centreline_head.bias, source.skeleton_head.bias)
    assert report["warm_start_kind"] == "legacy_context_unet_to_four_class"
    assert report["missing_inherited_keys"] == []


def test_context_checkpoint_initializes_uncertain_channel_conservatively(tmp_path):
    source = ContextUNet(base_channels=4, context_module="aspp", aspp_dilations=[1, 2])
    path = tmp_path / "context.pt"
    torch.save({"model_state_dict": source.state_dict()}, path)
    target = FourClassContextUNet(base_channels=4, aspp_dilations=[1, 2])
    report = load_initial_checkpoint(target, str(path), torch.device("cpu"), four_class_uncertain_bias=-7.5)
    torch.testing.assert_close(target.semantic_head.weight[3], torch.zeros_like(target.semantic_head.weight[3]))
    torch.testing.assert_close(target.semantic_head.bias[3], torch.tensor(-7.5))
    assert set(report["initialized_keys"]) == {"semantic_head.weight[3]", "semantic_head.bias[3]"}


def test_partial_initialization_fails_when_backbone_is_missing(tmp_path):
    source = ContextUNet(base_channels=4, context_module="aspp", aspp_dilations=[1, 2])
    state = source.state_dict()
    del state["enc1.net.0.weight"]
    path = tmp_path / "broken.pt"
    torch.save({"model_state_dict": state}, path)
    with pytest.raises(ValueError, match="missing or incompatible inherited"):
        load_initial_checkpoint(
            FourClassContextUNet(base_channels=4, aspp_dilations=[1, 2]), str(path), torch.device("cpu")
        )


def test_partial_initialization_fails_when_semantic_head_is_incompatible(tmp_path):
    source = ContextUNet(base_channels=4, context_module="aspp", aspp_dilations=[1, 2])
    state = source.state_dict()
    state["semantic_head.weight"] = state["semantic_head.weight"][:2]
    path = tmp_path / "broken.pt"
    torch.save({"model_state_dict": state}, path)
    with pytest.raises(ValueError, match="semantic_head.weight"):
        load_initial_checkpoint(
            FourClassContextUNet(base_channels=4, aspp_dilations=[1, 2]), str(path), torch.device("cpu")
        )


def test_exact_four_class_checkpoint_reload(tmp_path):
    source = FourClassContextUNet(aspp_dilations=[1, 2])
    path = tmp_path / "four_class.pt"
    torch.save({
        "model_state_dict": source.state_dict(),
        "config": {"model_variant": "four_class_context_unet", "aspp_dilations": "1,2"},
    }, path)
    loaded, _, _ = load_model(path, None, "cpu", torch)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_native_four_class_initial_checkpoint_loads_exactly(tmp_path):
    source = FourClassContextUNet(base_channels=4, aspp_dilations=[1, 2])
    path = tmp_path / "four_class.pt"
    torch.save({"model_state_dict": source.state_dict()}, path)
    target = FourClassContextUNet(base_channels=4, aspp_dilations=[1, 2])
    report = load_initial_checkpoint(target, str(path), torch.device("cpu"))
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value)
    assert report["warm_start_kind"] == "native_four_class"


def test_tiled_inference_blends_logits_and_preserves_all_maps(tmp_path):
    class TinyFourClass(torch.nn.Module):
        def forward(self, image):
            b, _, h, w = image.shape
            logits = torch.zeros(b, 4, h, w, device=image.device)
            logits[:, 0, :, 0] = 8
            logits[:, 1, :, 1] = 8
            logits[:, 2, :, 2] = 8
            logits[:, 3, :, 3:] = 8
            line = torch.full((b, 1, h, w), 2.0, device=image.device)
            return {"semantic_logits": logits, "raw_centreline_logits": line}

    prediction = run_tiled_inference(
        TinyFourClass(), np.zeros((4, 4), dtype=np.uint8), torch.device("cpu"), torch,
        patch_size=4, overlap=0,
    )
    required = {
        "semantic_logits", "raw_centreline_logits", "four_class_probabilities",
        "background_probability", "fibrous_probability", "clump_probability",
        "uncertainty_probability", "centreline_probability", "semantic_class_map",
    }
    assert required <= set(prediction)
    np.testing.assert_allclose(prediction["four_class_probabilities"].sum(axis=0), 1.0, atol=1e-6)
    assert set(np.unique(prediction["semantic_class_map"])) == {0, 1, 3, 255}
    np.testing.assert_allclose(prediction["centreline_probability"], torch.sigmoid(torch.tensor(2.0)).item())
    sample = {
        "image_uint8": np.zeros((4, 4), dtype=np.uint8), "image_float": None,
        "real_semantic_mask": np.zeros((4, 4), dtype=np.uint8),
        "real_skeleton_mask": np.zeros((4, 4), dtype=np.uint8),
    }
    write_outputs(tmp_path, tmp_path / "image.tif", sample, prediction, {})
    with np.load(tmp_path / "image_predictions.npz") as saved:
        assert required <= set(saved.files)


def test_training_cli_exposes_four_class_weights_and_forces_aspp():
    from scripts.train_first_baseline import build_parser, config_from_args

    args = build_parser().parse_args([
        "--manifest", "manifest.csv", "--out", "run",
        "--model-variant", "four_class_context_unet",
        "--lambda-semantic", "1.2", "--lambda-dice", "0.3", "--lambda-skeleton", "0.4",
        "--four-class-semantic-loss", "pixel_ce",
    ])
    config = config_from_args(args)
    assert (config.lambda_semantic, config.lambda_dice, config.lambda_skeleton) == (1.2, 0.3, 0.4)
    assert config.four_class_semantic_loss == "pixel_ce"
    assert config.context_module == "aspp"
