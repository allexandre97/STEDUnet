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
    assert set(metrics["skeleton_metrics_by_threshold"]) == {"0.25", "0.5", "0.75"}


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
    assert payload["model"]["architecture"] == "SmallUNet"


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
