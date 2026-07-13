import csv
import json
from pathlib import Path
import importlib

import numpy as np
import pytest

from fibras.training.schema08_index import build_training_manifest


def write_dataset(root: Path, count: int = 5, include_clump: bool = True) -> None:
    rows = []
    for i in range(count):
        sample_id = f"sample_{i:04d}"
        image = np.full((32, 32), i + 1, dtype=np.uint8)
        semantic = np.zeros((32, 32), dtype=np.uint8)
        semantic[4:18, 4:18] = 1
        if include_clump:
            semantic[18:24, 18:24] = 3
        semantic[0:3, 0:3] = 255
        skeleton = np.zeros((32, 32), dtype=np.uint8)
        skeleton[4:18, 10] = 1
        np.savez_compressed(
            root / f"{sample_id}.npz",
            render_uint8=image,
            real_compatible_semantic_mask=semantic,
            real_compatible_fibrous_mask=(semantic == 1).astype(np.uint8),
            real_compatible_clump_mask=(semantic == 3).astype(np.uint8),
            real_compatible_uncertain_ignore_mask=(semantic == 255).astype(np.uint8),
            real_compatible_skeleton_mask=skeleton,
        )
        metadata = {
            "dataset_schema_version": "synthetic_sted_3d_morphology_0.8.0",
            "generator_version": "fixture",
            "parent_synthetic_sample_id": f"parent_{i:04d}",
            "source_blank_provenance": {"blank_stable_image_id": f"blank_{i:04d}"},
            "synthetic_split": "synthetic_background_train",
        }
        (root / f"{sample_id}.json").write_text(json.dumps(metadata), encoding="utf-8")
        rows.append(
            {
                "sample_id": sample_id,
                "npz_path": f"{sample_id}.npz",
                "npz_sha256": "not_checked",
                "json_path": f"{sample_id}.json",
                "json_sha256": "not_checked",
                "schema_version": "synthetic_sted_3d_morphology_0.8.0",
                "generator_version": "fixture",
            }
        )
    with (root / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_schema08_training_manifest_indexes_required_fields(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_dataset(dataset, 5)
    out = tmp_path / "training_manifest.csv"
    rows = build_training_manifest(dataset, out, root=tmp_path)
    parallel_rows = build_training_manifest(
        dataset,
        tmp_path / "training_manifest_parallel.csv",
        root=tmp_path,
        num_workers=2,
    )
    assert parallel_rows == rows
    assert [row["split"] for row in rows].count("train") == 4
    assert [row["split"] for row in rows].count("validation") == 1
    assert rows[0]["parent_synthetic_sample_id"] == "parent_0000"
    assert rows[0]["source_blank_id"] == "blank_0000"
    assert rows[0]["source_synthetic_split"] == "synthetic_background_train"
    assert out.exists()


def test_schema08_patch_dataset_remaps_targets_and_ignore(tmp_path):
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import IGNORE_INDEX, Schema08PatchDataset

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_dataset(dataset, 5)
    manifest = tmp_path / "training_manifest.csv"
    build_training_manifest(dataset, manifest, root=tmp_path)
    ds = Schema08PatchDataset(
        manifest,
        "train",
        patch_size=16,
        patches_per_sample=1,
        foreground_fraction=1.0,
        seed=1,
    )
    item = ds[0]
    assert item["image"].shape == (1, 16, 16)
    assert item["semantic"].shape == (16, 16)
    assert set(torch.unique(item["semantic"]).tolist()) <= {IGNORE_INDEX, 0, 1, 2}
    assert item["skeleton"].shape == (1, 16, 16)
    assert item["valid"].shape == (1, 16, 16)


def test_schema08_patch_dataset_cache_matches_uncached_loading(tmp_path):
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import Schema08PatchDataset

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_dataset(dataset, 5)
    manifest = tmp_path / "training_manifest.csv"
    build_training_manifest(dataset, manifest, root=tmp_path)
    kwargs = dict(
        manifest_path=manifest,
        split="train",
        patch_size=16,
        patches_per_sample=2,
        foreground_fraction=1.0,
        seed=7,
    )
    uncached = Schema08PatchDataset(**kwargs, cache_samples=False)
    cached = Schema08PatchDataset(**kwargs, cache_samples=True)
    assert len(cached._cache) == len(cached.rows)
    for key in ["image", "semantic", "skeleton", "valid"]:
        assert torch.equal(uncached[3][key], cached[3][key])


def test_schema08_patch_dataset_limits_samples_and_epoch_patches(tmp_path):
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import Schema08PatchDataset

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_dataset(dataset, 6)
    manifest = tmp_path / "training_manifest.csv"
    build_training_manifest(dataset, manifest, root=tmp_path)
    ds = Schema08PatchDataset(
        manifest,
        "train",
        patch_size=16,
        patches_per_sample=4,
        limit_samples=2,
        patches_per_epoch=7,
    )
    assert len(ds.rows) == 2
    assert len(ds) == 7


def test_small_unet_outputs_semantic_and_skeleton_heads():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import SmallUNet

    model = SmallUNet(base_channels=4)
    outputs = model(torch.zeros(2, 1, 32, 32))
    assert outputs["semantic_logits"].shape == (2, 3, 32, 32)
    assert outputs["skeleton_logits"].shape == (2, 1, 32, 32)


def test_small_unet_forward_supports_128():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import SmallUNet

    outputs = SmallUNet(base_channels=4)(torch.zeros(1, 1, 128, 128))
    assert outputs["semantic_logits"].shape == (1, 3, 128, 128)
    assert outputs["skeleton_logits"].shape == (1, 1, 128, 128)


def test_context_unet_forward_supports_128_and_256():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import ContextUNet

    model = ContextUNet(base_channels=4, context_module="aspp", aspp_dilations=[1, 2, 4, 8])
    for size in (128, 256):
        outputs = model(torch.zeros(1, 1, size, size))
        assert outputs["semantic_logits"].shape == (1, 3, size, size)
        assert outputs["skeleton_logits"].shape == (1, 1, size, size)
        assert "uncertainty_logits" not in outputs


def test_context_unet_context_module_none_and_aspp_work():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import ContextUNet

    for context_module in ("none", "aspp"):
        model = ContextUNet(base_channels=4, context_module=context_module, aspp_dilations=[1, 2])
        outputs = model(torch.zeros(1, 1, 128, 128))
        assert outputs["semantic_logits"].shape == (1, 3, 128, 128)
        assert outputs["skeleton_logits"].shape == (1, 1, 128, 128)


def test_aspp_block_preserves_spatial_dimensions():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import ASPPBlock

    block = ASPPBlock(8, [1, 2, 4, 8])
    out = block(torch.zeros(2, 8, 17, 19))
    assert out.shape == (2, 8, 17, 19)


def test_context_unet_uncertainty_head_is_optional():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import ContextUNet

    without_head = ContextUNet(base_channels=4, context_module="none")
    with_head = ContextUNet(base_channels=4, context_module="none", uncertainty_head=True)

    assert "uncertainty_logits" not in without_head(torch.zeros(1, 1, 32, 32))
    assert with_head(torch.zeros(1, 1, 32, 32))["uncertainty_logits"].shape == (1, 1, 32, 32)


def test_init_checkpoint_can_warm_start_uncertainty_head_model(tmp_path):
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import SmallUNet, load_initial_checkpoint

    source = SmallUNet(base_channels=4)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": source.state_dict(), "config": {"fixture": True}}, checkpoint)
    target = SmallUNet(base_channels=4, uncertainty_head=True)

    report = load_initial_checkpoint(target, str(checkpoint), torch.device("cpu"))

    assert report["path"] == str(checkpoint)
    assert report["checkpoint_config"] == {"fixture": True}
    assert set(report["missing_keys"]) == {"uncertainty_head.weight", "uncertainty_head.bias"}
    assert report["unexpected_keys"] == []


def test_old_small_unet_checkpoint_can_warm_start_context_unet(tmp_path):
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import ContextUNet, SmallUNet, load_initial_checkpoint

    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": SmallUNet(base_channels=4).state_dict()}, checkpoint)
    target = ContextUNet(base_channels=4, context_module="aspp", aspp_dilations=[1, 2])

    report = load_initial_checkpoint(target, str(checkpoint), torch.device("cpu"))

    assert report["unexpected_keys"] == []
    assert all(key.startswith("context.") for key in report["missing_keys"])


def test_training_script_accepts_init_checkpoint_argument():
    pytest.importorskip("torch")
    import scripts.train_first_baseline as train_script

    args = train_script.build_parser().parse_args(
        [
            "--manifest",
            "manifest.csv",
            "--out",
            "run",
            "--init-checkpoint",
            "model.pt",
        ]
    )
    config = train_script.config_from_args(args)

    assert config.init_checkpoint == "model.pt"
    assert config.command_line_args["init_checkpoint"] == "model.pt"


def test_training_script_accepts_context_v2_arguments():
    pytest.importorskip("torch")
    import scripts.train_first_baseline as train_script

    args = train_script.build_parser().parse_args(
        [
            "--manifest",
            "manifest.csv",
            "--out",
            "run",
            "--model-variant",
            "context_unet",
            "--context-module",
            "aspp",
            "--aspp-dilations",
            "1,2,4,8",
            "--enable-uncertainty-head",
            "--lambda-uncertainty",
            "0.1",
            "--uncertainty-loss",
            "balanced_bce",
            "--uncertain-skeleton-policy",
            "suppress",
            "--lambda-clump-anti-fibrous",
            "0.25",
            "--lambda-clump-anti-skeleton",
            "0.5",
        ]
    )
    config = train_script.config_from_args(args)

    assert config.model_variant == "context_unet"
    assert config.context_module == "aspp"
    assert config.enable_uncertainty_head is True
    assert config.uncertainty_loss == "balanced_bce"
    assert config.uncertain_skeleton_policy == "suppress"
    assert config.lambda_clump_anti_fibrous == 0.25
    assert config.lambda_clump_anti_skeleton == 0.5


def test_training_script_accepts_checkpoint_bookkeeping_arguments():
    pytest.importorskip("torch")
    import scripts.train_first_baseline as train_script

    args = train_script.build_parser().parse_args(
        [
            "--manifest",
            "manifest.csv",
            "--out",
            "run",
            "--save-best-checkpoint",
            "--best-metric",
            "fibrous_dice",
            "--best-mode",
            "max",
            "--early-stop-patience",
            "10",
            "--early-stop-min-delta",
            "0.001",
            "--early-stop-warmup-epochs",
            "15",
            "--save-checkpoint-every",
            "5",
        ]
    )
    config = train_script.config_from_args(args)

    assert config.save_best_checkpoint is True
    assert config.best_metric == "fibrous_dice"
    assert config.best_mode == "max"
    assert config.early_stop_patience == 10
    assert config.early_stop_min_delta == 0.001
    assert config.early_stop_warmup_epochs == 15
    assert config.save_checkpoint_every == 5


def test_best_mode_auto_maps_loss_to_min():
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import resolve_best_mode

    assert resolve_best_mode("loss", "auto") == "min"


@pytest.mark.parametrize("metric", ["fibrous_dice", "clump_dice", "skeleton_dice", "skeleton_dice_0.75", "uncertain_dice"])
def test_best_mode_auto_maps_dice_metrics_to_max(metric):
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import resolve_best_mode

    assert resolve_best_mode(metric, "auto") == "max"


def test_unavailable_best_metric_fails_clearly():
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import selected_validation_metric

    with pytest.raises(ValueError, match="validation metric 'clump_dice' is not numeric"):
        selected_validation_metric({"loss": 0.1, "clump_dice": "not_applicable"}, "clump_dice")


def test_threshold_best_metric_reads_nested_validation_metric():
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import selected_validation_metric

    metrics = {"skeleton_metrics_by_threshold": {"0.75": {"dice": 0.7}}}

    assert selected_validation_metric(metrics, "skeleton_dice_0.75") == 0.7


def test_metrics_mark_clump_not_applicable_without_target_pixels():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import evaluate_model

    class ClumpEverywhere(torch.nn.Module):
        def forward(self, image):
            b, _, h, w = image.shape
            semantic = torch.zeros(b, 3, h, w, device=image.device)
            semantic[:, 2] = 10.0
            return {
                "semantic_logits": semantic,
                "skeleton_logits": torch.zeros(b, 1, h, w, device=image.device),
            }

    batch = {
        "image": torch.zeros(1, 1, 8, 8),
        "semantic": torch.zeros(1, 8, 8, dtype=torch.long),
        "skeleton": torch.zeros(1, 1, 8, 8),
        "valid": torch.ones(1, 1, 8, 8),
    }
    metrics = evaluate_model(ClumpEverywhere(), [batch], torch.device("cpu"), 0.5, 8.0, 0.5)
    assert metrics["clump_dice"] == "not_applicable"
    assert metrics["clump_target_pixels"] == 0


def test_metrics_ignore_uncertain_pixels_for_skeleton_and_semantic():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import IGNORE_INDEX, evaluate_model

    class PredictIgnoredOnly(torch.nn.Module):
        def forward(self, image):
            b, _, h, w = image.shape
            semantic = torch.zeros(b, 3, h, w, device=image.device)
            semantic[:, 2, :2, :2] = 10.0
            skeleton = torch.full((b, 1, h, w), -10.0, device=image.device)
            skeleton[:, :, :2, :2] = 10.0
            return {"semantic_logits": semantic, "skeleton_logits": skeleton}

    semantic = torch.zeros(1, 8, 8, dtype=torch.long)
    semantic[:, :2, :2] = IGNORE_INDEX
    skeleton = torch.zeros(1, 1, 8, 8)
    skeleton[:, :, :2, :2] = 1.0
    valid = (semantic != IGNORE_INDEX).unsqueeze(1).float()
    batch = {
        "image": torch.zeros(1, 1, 8, 8),
        "semantic": semantic,
        "skeleton": skeleton,
        "valid": valid,
    }
    metrics = evaluate_model(PredictIgnoredOnly(), [batch], torch.device("cpu"), 0.5, 8.0, 0.5)
    assert metrics["clump_dice"] == "not_applicable"
    assert metrics["skeleton_dice"] == "not_applicable"
    assert metrics["skeleton_precision"] == "not_applicable"
    assert metrics["skeleton_recall"] == "not_applicable"


def test_skeleton_threshold_is_recorded_in_metrics():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import evaluate_model

    class EmptyModel(torch.nn.Module):
        def forward(self, image):
            b, _, h, w = image.shape
            return {
                "semantic_logits": torch.zeros(b, 3, h, w, device=image.device),
                "skeleton_logits": torch.zeros(b, 1, h, w, device=image.device),
            }

    batch = {
        "image": torch.zeros(1, 1, 8, 8),
        "semantic": torch.zeros(1, 8, 8, dtype=torch.long),
        "skeleton": torch.zeros(1, 1, 8, 8),
        "valid": torch.ones(1, 1, 8, 8),
    }
    metrics = evaluate_model(EmptyModel(), [batch], torch.device("cpu"), 0.5, 8.0, 0.25)
    assert metrics["skeleton_threshold"] == 0.25
    assert set(metrics["skeleton_metrics_by_threshold"]) == {"0.25", "0.5", "0.75", "0.85"}


def test_uncertainty_loss_supervises_uncertain_ignore_without_semantic_background():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import IGNORE_INDEX, compute_loss

    outputs = {
        "semantic_logits": torch.zeros(1, 3, 2, 2),
        "skeleton_logits": torch.zeros(1, 1, 2, 2),
        "uncertainty_logits": torch.zeros(1, 1, 2, 2),
    }
    batch = {
        "semantic": torch.tensor([[[IGNORE_INDEX, 0], [1, 2]]]),
        "skeleton": torch.zeros(1, 1, 2, 2),
        "uncertain": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        "valid": torch.tensor([[[[0.0, 1.0], [1.0, 1.0]]]]),
    }

    _, parts = compute_loss(outputs, batch, lambda_uncertainty=1.0)

    assert "uncertainty_loss" in parts
    assert batch["semantic"][0, 0, 0].item() == IGNORE_INDEX


@pytest.mark.parametrize("loss_name", ["bce", "balanced_bce", "focal_bce"])
def test_uncertainty_loss_modes_are_supported(loss_name):
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import uncertainty_loss_value

    logits = torch.zeros(1, 1, 2, 2)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

    loss = uncertainty_loss_value(logits, target, loss_name, 2.0, "auto")

    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_explicit_skeleton_valid_mask_always_ignores_uncertain_pixels():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import IGNORE_INDEX, compute_loss

    outputs = {
        "semantic_logits": torch.zeros(1, 3, 1, 2),
        "skeleton_logits": torch.tensor([[[[10.0, -10.0]]]]),
    }
    batch = {
        "semantic": torch.tensor([[[IGNORE_INDEX, 0]]]),
        "skeleton": torch.zeros(1, 1, 1, 2),
        "uncertain": torch.tensor([[[[1.0, 0.0]]]]),
        "valid": torch.tensor([[[[0.0, 1.0]]]]),
        "skeleton_valid": torch.tensor([[[[0.0, 1.0]]]]),
    }

    _, ignored = compute_loss(outputs, batch, uncertain_skeleton_policy="ignore")
    _, suppressed = compute_loss(outputs, batch, uncertain_skeleton_policy="suppress")

    assert ignored["skeleton_loss"] < 1e-3
    assert suppressed["skeleton_loss"] == ignored["skeleton_loss"]


def test_explicit_skeleton_valid_mask_excludes_uncertain_positive_target():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import IGNORE_INDEX, compute_loss

    outputs = {
        "semantic_logits": torch.zeros(1, 3, 1, 2),
        "skeleton_logits": torch.tensor([[[[10.0, -10.0]]]]),
    }
    batch = {
        "semantic": torch.tensor([[[IGNORE_INDEX, 0]]]),
        "skeleton": torch.tensor([[[[1.0, 0.0]]]]),
        "uncertain": torch.tensor([[[[1.0, 0.0]]]]),
        "valid": torch.tensor([[[[0.0, 1.0]]]]),
        "skeleton_valid": torch.tensor([[[[0.0, 1.0]]]]),
    }

    _, ignored = compute_loss(outputs, batch, uncertain_skeleton_policy="ignore")
    _, suppressed = compute_loss(outputs, batch, uncertain_skeleton_policy="suppress")

    assert ignored["skeleton_loss"] < 1e-3
    assert suppressed["skeleton_loss"] == ignored["skeleton_loss"]


def test_clump_protection_losses_apply_only_inside_target_clump():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import compute_loss

    outputs = {
        "semantic_logits": torch.tensor([[[[0.0, 0.0]], [[5.0, 5.0]], [[0.0, 0.0]]]]),
        "skeleton_logits": torch.full((1, 1, 1, 2), 5.0),
    }
    clump_batch = {
        "semantic": torch.tensor([[[2, 0]]]),
        "skeleton": torch.zeros(1, 1, 1, 2),
        "uncertain": torch.zeros(1, 1, 1, 2),
        "valid": torch.ones(1, 1, 1, 2),
    }
    background_batch = {**clump_batch, "semantic": torch.zeros(1, 1, 2, dtype=torch.long)}

    _, clump_parts = compute_loss(outputs, clump_batch)
    _, background_parts = compute_loss(outputs, background_batch)

    assert clump_parts["clump_anti_fibrous_loss"] > 0
    assert clump_parts["clump_anti_skeleton_loss"] > 0
    assert background_parts["clump_anti_fibrous_loss"] == 0
    assert background_parts["clump_anti_skeleton_loss"] == 0


def test_validation_metrics_include_enabled_auxiliary_losses():
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import evaluate_model

    class EmptyUncertaintyModel(torch.nn.Module):
        def forward(self, image):
            b, _, h, w = image.shape
            return {
                "semantic_logits": torch.zeros(b, 3, h, w, device=image.device),
                "skeleton_logits": torch.zeros(b, 1, h, w, device=image.device),
                "uncertainty_logits": torch.zeros(b, 1, h, w, device=image.device),
            }

    batch = {
        "image": torch.zeros(1, 1, 4, 4),
        "semantic": torch.zeros(1, 4, 4, dtype=torch.long),
        "skeleton": torch.zeros(1, 1, 4, 4),
        "uncertain": torch.zeros(1, 1, 4, 4),
        "valid": torch.ones(1, 1, 4, 4),
    }
    metrics = evaluate_model(
        EmptyUncertaintyModel(),
        [batch],
        torch.device("cpu"),
        0.5,
        8.0,
        lambda_uncertainty=0.25,
        lambda_clump_anti_fibrous=0.25,
        lambda_clump_anti_skeleton=0.25,
    )
    assert "uncertainty_loss" in metrics
    assert "clump_anti_fibrous_loss" in metrics
    assert "clump_anti_skeleton_loss" in metrics


def test_cuda_request_fails_clearly_when_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import choose_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="CUDA was requested but is not available"):
        choose_device("cuda:0")


def test_auto_device_uses_cuda_when_available(monkeypatch):
    torch = pytest.importorskip("torch")
    import fibras.training.schema08_baseline as baseline

    validated = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(baseline, "validate_cuda_device", lambda device: validated.append(str(device)))
    device = baseline.choose_device("auto")
    assert str(device) == "cuda:0"
    assert validated == ["cuda:0"]


def test_invalid_device_string_fails_clearly():
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import DeviceSelectionError, choose_device

    with pytest.raises(DeviceSelectionError, match="invalid device"):
        choose_device("cuda:not-an-index")


def test_macro_per_image_metric_is_used_for_checkpoint_selection():
    from fibras.training.schema08_baseline import macro_metric, selected_validation_metric

    per_image = {
        "easy": {"fibrous_dice": 1.0},
        "hard": {"fibrous_dice": 0.0},
    }
    metrics = {"macro_image_fibrous_dice": macro_metric(per_image, "fibrous_dice")}

    assert selected_validation_metric(metrics, "macro_image_fibrous_dice") == 0.5


def test_stage_one_freezes_encoder_and_stage_two_unfreezes_it():
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import BaselineConfig, SmallUNet, optimizer_for_stage

    model = SmallUNet(base_channels=4)
    config = BaselineConfig("manifest.csv", "run", stage1_learning_rate=1e-3, learning_rate=1e-4)
    stage1 = optimizer_for_stage(model, config, "stage1_frozen_encoder")
    assert not any(parameter.requires_grad for parameter in model.enc1.parameters())
    assert stage1.param_groups[0]["lr"] == 1e-3

    stage2 = optimizer_for_stage(model, config, "stage2_full_model")
    assert all(parameter.requires_grad for parameter in model.enc1.parameters())
    assert stage2.param_groups[0]["lr"] == 1e-4


def test_training_script_does_not_import_wandb_when_disabled(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import scripts.train_first_baseline as train_script

    real_import_module = importlib.import_module

    def fail_on_wandb(name):
        if name == "wandb":
            raise AssertionError("wandb should not be imported")
        return real_import_module(name)

    def fake_train(config, tracker=None):
        assert tracker is None
        assert config.command_line_args["wandb"] is False
        return {"device": "cpu", "validation_metrics": {}, "log": []}

    monkeypatch.setattr(train_script.importlib, "import_module", fail_on_wandb)
    monkeypatch.setattr(train_script, "train_baseline", fake_train)
    assert train_script.main(["--manifest", "manifest.csv", "--out", str(tmp_path / "run")]) == 0


def test_training_script_reports_missing_wandb(monkeypatch, tmp_path, capsys):
    pytest.importorskip("torch")
    import scripts.train_first_baseline as train_script

    real_import_module = importlib.import_module

    def missing_wandb(name):
        if name == "wandb":
            raise ImportError("missing wandb")
        return real_import_module(name)

    monkeypatch.setattr(train_script.importlib, "import_module", missing_wandb)
    code = train_script.main(["--manifest", "manifest.csv", "--out", str(tmp_path / "run"), "--wandb"])
    stderr = capsys.readouterr().err
    assert code == 2
    assert "--wandb was requested" in stderr
    assert "conda install -c conda-forge wandb" in stderr


def test_wandb_config_payload_contains_training_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import BaselineConfig, Schema08PatchDataset, run_metadata, wandb_config_payload

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_dataset(dataset, 5)
    manifest = tmp_path / "training_manifest.csv"
    build_training_manifest(dataset, manifest, root=tmp_path)
    config = BaselineConfig(
        manifest=str(manifest),
        out=str(tmp_path / "run"),
        epochs=3,
        batch_size=4,
        patch_size=16,
        learning_rate=3e-4,
        lambda_skeleton=0.25,
        skeleton_pos_weight=4.0,
        skeleton_threshold=0.4,
        cache_samples=True,
        limit_train_samples=2,
        limit_val_samples=1,
        patches_per_epoch=7,
        command_line_args={"wandb": True, "wandb_project": "sted-unet"},
    )
    train_ds = Schema08PatchDataset(manifest, "train", patch_size=16, limit_samples=2)
    val_ds = Schema08PatchDataset(manifest, "validation", patch_size=16, limit_samples=1)
    payload = wandb_config_payload(config, run_metadata(config, train_ds, val_ds, torch.device("cpu")))
    assert payload["training"]["epochs"] == 3
    assert payload["training"]["batch_size"] == 4
    assert payload["training"]["learning_rate"] == 3e-4
    assert payload["training"]["lambda_skeleton"] == 0.25
    assert payload["training"]["skeleton_pos_weight"] == 4.0
    assert payload["training"]["skeleton_threshold"] == 0.4
    assert payload["training"]["cache_samples"] is True
    assert payload["training"]["limit_train_samples"] == 2
    assert payload["training"]["limit_val_samples"] == 1
    assert payload["metadata"]["split_counts"] == {"train": 2, "validation": 1}
    assert payload["metadata"]["schema_versions"] == ["synthetic_sted_3d_morphology_0.8.0"]
    assert payload["metadata"]["generator_versions"] == ["fixture"]
    assert payload["model"]["architecture"] == "small_unet"


def test_wandb_epoch_metrics_are_flattened_and_loggable():
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import wandb_epoch_metrics

    row = {
        "epoch": 2,
        "train_loss": 0.4,
        "train_semantic_loss": 0.2,
        "train_skeleton_loss": 0.3,
        "validation_loss": 0.5,
        "validation_semantic_loss": 0.25,
        "validation_skeleton_loss": 0.5,
    }
    metrics = wandb_epoch_metrics(
        row,
        {
            "loss": 0.5,
            "semantic_loss": 0.25,
            "skeleton_loss": 0.5,
            "fibrous_dice": 0.8,
            "clump_dice": "not_applicable",
            "clump_target_pixels": 0,
            "skeleton_dice": 0.3,
            "skeleton_precision": 0.2,
            "skeleton_recall": 0.7,
            "predicted_skeleton_inside_predicted_fibrous_fraction": 1.0,
            "predicted_skeleton_inside_true_fibrous_fraction": 0.9,
            "skeleton_metrics_by_threshold": {
                "0.25": {"dice": 0.2, "precision": 0.1, "recall": 0.9},
                "0.5": {"dice": 0.3, "precision": 0.2, "recall": 0.7},
                "0.75": {"dice": "not_applicable", "precision": "not_applicable", "recall": 0.1},
            },
        },
    )
    assert metrics["train/loss"] == 0.4
    assert metrics["validation/clump_dice"] is None
    assert metrics["validation/skeleton_threshold_0.25/dice"] == 0.2
    assert metrics["validation/skeleton_threshold_0.75/precision"] is None


def test_training_is_deterministic_with_wandb_disabled(tmp_path):
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import BaselineConfig, train_baseline

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_dataset(dataset, 5)
    manifest = tmp_path / "training_manifest.csv"
    build_training_manifest(dataset, manifest, root=tmp_path)
    common = dict(
        manifest=str(manifest),
        epochs=1,
        batch_size=2,
        patch_size=16,
        patches_per_sample=1,
        validation_patches_per_sample=1,
        patches_per_epoch=4,
        limit_train_samples=3,
        limit_val_samples=1,
        seed=99,
        device="cpu",
    )
    first = train_baseline(BaselineConfig(out=str(tmp_path / "run_a"), **common))
    second = train_baseline(BaselineConfig(out=str(tmp_path / "run_b"), **common))
    assert first["log"] == second["log"]
    assert first["validation_metrics"] == second["validation_metrics"]


def fake_validation_metric_sequence(monkeypatch, values, metric="loss"):
    import fibras.training.schema08_baseline as baseline

    values = iter(values)

    def fake_evaluate_model(*args, **kwargs):
        value = float(next(values))
        metrics = {
            "loss": value if metric == "loss" else 1.0,
            "semantic_loss": 0.1,
            "skeleton_loss": 0.2,
            "clump_anti_fibrous_loss": 0.0,
            "clump_anti_skeleton_loss": 0.0,
            "fibrous_dice": value if metric == "fibrous_dice" else 0.5,
            "clump_dice": 0.4,
            "clump_target_pixels": 1,
            "uncertain_dice": 0.3,
            "uncertain_precision": 0.3,
            "uncertain_recall": 0.3,
            "uncertain_target_pixels": 1,
            "uncertainty_probability_by_target_region": {},
            "skeleton_dice": 0.2,
            "skeleton_precision": 0.2,
            "skeleton_recall": 0.2,
            "skeleton_threshold": 0.5,
            "skeleton_metrics_by_threshold": {"0.75": {"dice": value if metric == "skeleton_dice_0.75" else 0.2}},
            "predicted_skeleton_inside_predicted_fibrous_fraction": 0.2,
            "predicted_skeleton_inside_true_fibrous_fraction": 0.2,
        }
        return metrics

    monkeypatch.setattr(baseline, "evaluate_model", fake_evaluate_model)


def tiny_baseline_config(tmp_path, **overrides):
    from fibras.training.schema08_baseline import BaselineConfig

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_dataset(dataset, 5)
    manifest = tmp_path / "training_manifest.csv"
    build_training_manifest(dataset, manifest, root=tmp_path)
    params = dict(
        manifest=str(manifest),
        out=str(tmp_path / "run"),
        epochs=3,
        batch_size=2,
        patch_size=16,
        patches_per_sample=1,
        validation_patches_per_sample=1,
        patches_per_epoch=2,
        limit_train_samples=3,
        limit_val_samples=1,
        seed=99,
        device="cpu",
    )
    params.update(overrides)
    return BaselineConfig(**params)


def test_best_checkpoint_is_saved_and_matches_best_epoch(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import train_baseline

    fake_validation_metric_sequence(monkeypatch, [0.5, 0.4, 0.45, 0.45])
    config = tiny_baseline_config(tmp_path, save_best_checkpoint=True, epochs=3)

    train_baseline(config)

    best = tmp_path / "run" / "model_best.pt"
    final = tmp_path / "run" / "model.pt"
    assert best.exists()
    assert final.exists()
    assert torch.load(best, map_location="cpu")["epoch"] == 2
    assert torch.load(final, map_location="cpu")["epoch"] == 3


def test_checkpoint_summary_records_best_and_final_paths(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import train_baseline

    fake_validation_metric_sequence(monkeypatch, [0.5, 0.4, 0.45, 0.45])
    train_baseline(tiny_baseline_config(tmp_path, save_best_checkpoint=True, epochs=3))

    summary = json.loads((tmp_path / "run" / "checkpoint_summary.json").read_text(encoding="utf-8"))
    assert summary["best_epoch"] == 2
    assert summary["best_value"] == 0.4
    assert summary["final_epoch"] == 3
    assert summary["stopped_early"] is False
    assert summary["best_checkpoint_path"].endswith("model_best.pt")
    assert summary["final_checkpoint_path"].endswith("model.pt")


def test_early_stopping_triggers_after_patience(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import train_baseline

    fake_validation_metric_sequence(monkeypatch, [0.5, 0.6, 0.7, 0.7])
    train_baseline(tiny_baseline_config(tmp_path, epochs=5, early_stop_patience=2))

    summary = json.loads((tmp_path / "run" / "checkpoint_summary.json").read_text(encoding="utf-8"))
    assert summary["stopped_early"] is True
    assert summary["final_epoch"] == 3
    assert "no improvement in validation loss for 2 validation epochs" in summary["early_stop_reason"]


def test_early_stopping_respects_warmup_epochs(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import train_baseline

    fake_validation_metric_sequence(monkeypatch, [0.5, 0.6, 0.7, 0.8])
    train_baseline(tiny_baseline_config(tmp_path, epochs=5, early_stop_patience=1, early_stop_warmup_epochs=3))

    summary = json.loads((tmp_path / "run" / "checkpoint_summary.json").read_text(encoding="utf-8"))
    assert summary["stopped_early"] is True
    assert summary["final_epoch"] == 3


def test_periodic_checkpoints_are_saved_when_enabled(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import train_baseline

    fake_validation_metric_sequence(monkeypatch, [0.5, 0.4, 0.3, 0.3])
    train_baseline(tiny_baseline_config(tmp_path, epochs=3, save_checkpoint_every=2))

    periodic = tmp_path / "run" / "checkpoints" / "model_epoch_0002.pt"
    summary = json.loads((tmp_path / "run" / "checkpoint_summary.json").read_text(encoding="utf-8"))
    assert periodic.exists()
    assert summary["periodic_checkpoint_paths"] == [str(periodic)]


def test_default_training_writes_existing_final_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from fibras.training.schema08_baseline import train_baseline

    fake_validation_metric_sequence(monkeypatch, [0.5, 0.4])
    train_baseline(tiny_baseline_config(tmp_path, epochs=1))

    assert (tmp_path / "run" / "model.pt").exists()
    assert not (tmp_path / "run" / "model_best.pt").exists()


def test_training_can_resume_complete_checkpoint(tmp_path):
    from dataclasses import replace
    torch = pytest.importorskip("torch")
    from fibras.training.schema08_baseline import BaselineConfig, train_baseline

    first = tiny_baseline_config(tmp_path, epochs=1)
    train_baseline(first)
    checkpoint = tmp_path / "run" / "model.pt"
    resumed = replace(
        first,
        out=str(tmp_path / "resumed"),
        epochs=2,
        resume_checkpoint=str(checkpoint),
    )

    train_baseline(resumed)

    payload = torch.load(tmp_path / "resumed" / "model.pt", map_location="cpu")
    assert payload["epoch"] == 2
    assert payload["optimizer_state_dict"] is not None
    assert payload["checkpoint_metadata"]["kind"] == "final"
