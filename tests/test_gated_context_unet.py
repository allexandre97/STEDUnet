import numpy as np
import pytest


torch = pytest.importorskip("torch")

from fibras.training.schema08_baseline import (
    GatedContextUNet,
    compute_gated_loss,
    gated_probabilities,
    load_initial_checkpoint,
    stratified_gate_loss,
)
from scripts.evaluate_real_pilot_baseline import load_model, run_tiled_inference, write_outputs


PROBABILITY_KEYS = {
    "foreground_probability",
    "quantifiability_probability",
    "conditional_fibrous_probability",
    "conditional_clump_probability",
    "raw_centreline_probability",
    "final_background_probability",
    "final_fibrous_probability",
    "final_clump_probability",
    "final_uncertain_probability",
    "final_skeleton_probability",
}


def probabilities(foreground=0.0, morphology=(0.0, 0.0), gate=0.0, centreline=0.0):
    return gated_probabilities(
        torch.tensor([[[[foreground]]]], dtype=torch.float32),
        torch.tensor([[[[morphology[0]]], [[morphology[1]]]]], dtype=torch.float32),
        torch.tensor([[[[gate]]]], dtype=torch.float32),
        torch.tensor([[[[centreline]]]], dtype=torch.float32),
    )


def gated_outputs(shape=(1, 1, 1, 4), requires_grad=False):
    b, _, h, w = shape
    logits = {
        "foreground_logits": torch.zeros(b, 1, h, w, requires_grad=requires_grad),
        "morphology_logits": torch.zeros(b, 2, h, w, requires_grad=requires_grad),
        "quantifiability_logits": torch.zeros(b, 1, h, w, requires_grad=requires_grad),
        "raw_centreline_logits": torch.zeros(b, 1, h, w, requires_grad=requires_grad),
    }
    return {**logits, **gated_probabilities(*logits.values())}


def batch():
    # background, fibrous, clump, uncertain
    return {
        "semantic": torch.tensor([[[0, 1, 2, -100]]]),
        "skeleton": torch.tensor([[[[0.0, 1.0, 0.0, 1.0]]]]),
        "uncertain": torch.tensor([[[[0.0, 0.0, 0.0, 1.0]]]]),
        "uncertain_distance": torch.tensor([[[[3.0, 1.0, 1.0, 0.0]]]]),
        "valid": torch.tensor([[[[1.0, 1.0, 1.0, 0.0]]]]),
        "skeleton_valid": torch.tensor([[[[1.0, 1.0, 1.0, 0.0]]]]),
    }


def test_final_probabilities_are_bounded_and_sum_to_one():
    out = GatedContextUNet(base_channels=4, aspp_dilations=[1, 2])(torch.randn(2, 1, 32, 32))
    final = torch.cat([
        out["final_background_probability"], out["final_fibrous_probability"],
        out["final_clump_probability"], out["final_uncertain_probability"],
    ], dim=1)
    torch.testing.assert_close(final.sum(dim=1), torch.ones_like(final[:, 0]))
    assert all(bool(((out[key] >= 0) & (out[key] <= 1)).all()) for key in PROBABILITY_KEYS)


def test_lowering_gate_lowers_final_morphology_probabilities():
    high = probabilities(foreground=2.0, morphology=(1.0, -1.0), gate=3.0)
    low = probabilities(foreground=2.0, morphology=(1.0, -1.0), gate=-3.0)
    assert low["final_fibrous_probability"] < high["final_fibrous_probability"]
    assert low["final_clump_probability"] < high["final_clump_probability"]


def test_final_class_assignment_cannot_be_fibrous_and_uncertain():
    out = probabilities(foreground=4.0, morphology=(2.0, -2.0), gate=0.0)
    final = torch.cat([
        out["final_background_probability"], out["final_fibrous_probability"],
        out["final_clump_probability"], out["final_uncertain_probability"],
    ], dim=1)
    class_index = final.argmax(dim=1)
    assert not bool(((class_index == 1) & (class_index == 3)).any())


def test_uncertain_target_is_foreground_and_nonquantifiable():
    outputs = gated_outputs(requires_grad=True)
    loss, _ = compute_gated_loss(outputs, batch())
    loss.backward()
    assert outputs["foreground_logits"].grad[0, 0, 0, 3] < 0
    assert outputs["quantifiability_logits"].grad[0, 0, 0, 3] > 0


def test_morphology_loss_ignores_background_and_uncertain():
    outputs = gated_outputs()
    _, before = compute_gated_loss(outputs, batch())
    outputs["morphology_logits"][:, :, :, [0, 3]] = torch.tensor([[[[100.0, -100.0]], [[-100.0, 100.0]]]])
    outputs.update(gated_probabilities(
        outputs["foreground_logits"], outputs["morphology_logits"],
        outputs["quantifiability_logits"], outputs["raw_centreline_logits"],
    ))
    _, after = compute_gated_loss(outputs, batch())
    assert after["morphology_loss"] == before["morphology_loss"]


def test_skeleton_loss_ignores_clump_and_uncertain():
    outputs = gated_outputs()
    _, before = compute_gated_loss(outputs, batch())
    outputs["raw_centreline_logits"][:, :, :, [2, 3]] = 100.0
    outputs.update(gated_probabilities(
        outputs["foreground_logits"], outputs["morphology_logits"],
        outputs["quantifiability_logits"], outputs["raw_centreline_logits"],
    ))
    _, after = compute_gated_loss(outputs, batch())
    assert after["skeleton_loss"] == before["skeleton_loss"]


def test_final_skeleton_is_gated_by_confident_fibrous_probability():
    out = probabilities(foreground=1.0, morphology=(2.0, -2.0), gate=-1.0, centreline=2.0)
    torch.testing.assert_close(
        out["final_skeleton_probability"],
        out["final_fibrous_probability"] * out["raw_centreline_probability"],
    )
    assert out["final_skeleton_probability"] <= out["final_fibrous_probability"]


def test_gate_stratum_average_is_unchanged_by_background_abundance():
    semantic = torch.tensor([[[1, -100]]])
    uncertain = torch.tensor([[[False, True]]])
    logits = torch.tensor([[[2.0, -2.0]]])
    loss, _ = stratified_gate_loss(logits, semantic, uncertain)
    expanded_loss, _ = stratified_gate_loss(
        torch.cat([logits, torch.full((1, 1, 100), 20.0)], dim=2),
        torch.cat([semantic, torch.zeros(1, 1, 100, dtype=torch.long)], dim=2),
        torch.cat([uncertain, torch.zeros(1, 1, 100, dtype=torch.bool)], dim=2),
    )
    torch.testing.assert_close(loss, expanded_loss)


def test_context_checkpoint_initializes_only_compatible_gated_parameters(tmp_path):
    from fibras.training.schema08_baseline import ContextUNet

    source = ContextUNet(base_channels=4, context_module="aspp", aspp_dilations=[1, 2])
    path = tmp_path / "context.pt"
    torch.save({"model_state_dict": source.state_dict()}, path)
    target = GatedContextUNet(base_channels=4, context_module="aspp", aspp_dilations=[1, 2])
    report = load_initial_checkpoint(target, str(path), torch.device("cpu"))
    for name, value in source.state_dict().items():
        if name.startswith(("enc", "context.", "up", "dec")):
            torch.testing.assert_close(target.state_dict()[name], value)
    assert not report["missing_backbone_keys"]
    assert {"semantic_head.weight", "semantic_head.bias", "skeleton_head.weight", "skeleton_head.bias"} <= set(report["incompatible_parameters"])
    assert all(key.split(".")[0].endswith("head") for key in report["missing_keys"])


def test_gated_checkpoint_loads_exactly(tmp_path):
    source = GatedContextUNet(base_channels=16, context_module="aspp", aspp_dilations=[1, 2])
    path = tmp_path / "gated.pt"
    torch.save({
        "model_state_dict": source.state_dict(),
        "config": {
            "model_variant": "gated_context_unet", "context_module": "aspp",
            "aspp_dilations": "1,2",
        },
    }, path)
    loaded, _, _ = load_model(path, None, "cpu", torch)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_tiled_inference_preserves_all_gated_maps_and_semantic_ids(tmp_path):
    class TinyGated(torch.nn.Module):
        def forward(self, image):
            b, _, h, w = image.shape
            zeros = torch.zeros(b, 1, h, w, device=image.device)
            return {
                "foreground_logits": zeros + 2.0,
                "morphology_logits": torch.cat([zeros + 1.0, zeros - 1.0], dim=1),
                "quantifiability_logits": zeros,
                "raw_centreline_logits": zeros + 1.0,
            }

    predictions = run_tiled_inference(
        TinyGated(), np.zeros((9, 11), dtype=np.uint8), torch.device("cpu"), torch,
        patch_size=8, overlap=2,
    )
    assert PROBABILITY_KEYS <= set(predictions)
    assert set(np.unique(predictions["semantic_class_map"])) <= {0, 1, 3, 255}
    sample = {
        "image_uint8": np.zeros((9, 11), dtype=np.uint8),
        "image_float": None,
        "real_semantic_mask": np.zeros((9, 11), dtype=np.uint8),
        "real_skeleton_mask": np.zeros((9, 11), dtype=np.uint8),
    }
    write_outputs(tmp_path, tmp_path / "image.tif", sample, predictions, {})
    with np.load(tmp_path / "image_predictions.npz") as saved:
        assert PROBABILITY_KEYS <= set(saved.files)


def test_gated_argmax_maps_all_four_output_ids():
    class FourClasses(torch.nn.Module):
        def forward(self, image):
            b, _, h, w = image.shape
            foreground = torch.full((b, 1, h, w), 10.0, device=image.device)
            foreground[:, :, :, 0] = -10.0
            gate = torch.full_like(foreground, 10.0)
            gate[:, :, :, 3] = -10.0
            morphology = torch.full((b, 2, h, w), -10.0, device=image.device)
            morphology[:, 0, :, 1] = 10.0
            morphology[:, 1, :, 2] = 10.0
            return {
                "foreground_logits": foreground, "morphology_logits": morphology,
                "quantifiability_logits": gate, "raw_centreline_logits": foreground,
            }

    predictions = run_tiled_inference(
        FourClasses(), np.zeros((4, 4), dtype=np.uint8), torch.device("cpu"), torch,
        patch_size=4, overlap=0,
    )
    assert set(np.unique(predictions["semantic_class_map"])) == {0, 1, 3, 255}


def test_cpu_forward_backward_has_finite_gradients():
    model = GatedContextUNet(base_channels=4, aspp_dilations=[1, 2])
    outputs = model(torch.randn(1, 1, 16, 16))
    training_batch = {
        "semantic": torch.zeros(1, 16, 16, dtype=torch.long),
        "skeleton": torch.zeros(1, 1, 16, 16),
        "uncertain": torch.zeros(1, 1, 16, 16),
        "valid": torch.ones(1, 1, 16, 16),
    }
    training_batch["semantic"][:, 4:12, 4:12] = 1
    loss, _ = compute_gated_loss(outputs, training_batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_training_cli_exposes_gated_loss_weights_and_uses_aspp():
    from scripts.train_first_baseline import build_parser, config_from_args

    args = build_parser().parse_args([
        "--manifest", "manifest.csv", "--out", "run",
        "--model-variant", "gated_context_unet",
        "--lambda-foreground", "0.9", "--lambda-gate", "0.4",
        "--lambda-morphology", "0.8", "--lambda-joint", "0.7",
        "--lambda-skeleton", "0.3",
    ])
    config = config_from_args(args)
    assert (
        config.lambda_foreground, config.lambda_gate, config.lambda_morphology,
        config.lambda_joint, config.lambda_skeleton,
    ) == (0.9, 0.4, 0.8, 0.7, 0.3)
    assert config.context_module == "aspp"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_backward_accepts_explicit_device():
    from fibras.training.schema08_baseline import choose_device

    device = choose_device("cuda:0")
    model = GatedContextUNet(base_channels=4, aspp_dilations=[1, 2]).to(device)
    outputs = model(torch.randn(1, 1, 16, 16, device=device))
    loss = sum(outputs[key].mean() for key in PROBABILITY_KEYS)
    loss.backward()
    assert torch.isfinite(loss)
