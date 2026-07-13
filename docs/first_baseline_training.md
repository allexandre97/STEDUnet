# First Schema-0.8 Baseline Training

This is a minimal smoke/overfit pipeline for the schema-0.8 real-compatible synthetic STED dataset. It is not the final model architecture or training setup.

## Environment

The lightweight `environment.yml` intentionally remains suitable for knowledge-base and synthetic-data QA work without PyTorch.

Training requires PyTorch. The training environment also includes optional Weights & Biases support:

```bash
conda env create -f environment-training.yml
```

For an existing environment, update it with:

```bash
conda env update -f environment-training.yml
```

The local development environment used for the first smoke pass was `fibras`, which already had PyTorch and CUDA available.

W&B is optional and disabled by default. Normal training does not import or require `wandb`.

## CUDA Selection

The default `--device auto` uses `cuda:0` when PyTorch can see CUDA and falls back to CPU otherwise. To require a GPU, pass an explicit CUDA device:

```bash
conda run --no-capture-output -n fibras python scripts/train_first_baseline.py \
  --manifest data_manifests/training_v0_schema08.csv \
  --out runs/first_baseline_schema08_cuda_check \
  --epochs 1 \
  --batch-size 2 \
  --patch-size 128 \
  --patches-per-epoch 8 \
  --limit-train-samples 2 \
  --limit-val-samples 2 \
  --cache-samples \
  --seed 123 \
  --device cuda:0
```

If CUDA is requested but unavailable, the script exits before training with the PyTorch CUDA version, visible device count, and a driver/environment hint. For a quick environment check:

```bash
conda run --no-capture-output -n fibras python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

## Normal Training Without W&B

```bash
conda run --no-capture-output -n fibras python scripts/train_first_baseline.py \
  --manifest data_manifests/training_v0_schema08.csv \
  --out runs/first_baseline_schema08_v0_50ep \
  --epochs 50 \
  --batch-size 4 \
  --patch-size 128 \
  --patches-per-epoch 1024 \
  --cache-samples \
  --seed 123
```

## W&B Online Tracking

Use `--wandb` to enable W&B. The default W&B mode is `online` only when `--wandb` is set.

```bash
conda run --no-capture-output -n fibras python scripts/train_first_baseline.py \
  --manifest data_manifests/training_v0_schema08.csv \
  --out runs/first_baseline_schema08_v0_50ep_wandb \
  --epochs 50 \
  --batch-size 4 \
  --patch-size 128 \
  --patches-per-epoch 1024 \
  --cache-samples \
  --seed 123 \
  --wandb \
  --wandb-project sted-unet \
  --wandb-run-name first_baseline_schema08_v0_50ep
```

Panel and checkpoint uploads are off by default. Add `--wandb-log-panels` to upload the final validation QA panels, and `--wandb-log-checkpoints` to upload the final `model.pt` checkpoint.

## Optional Real-Crop Fine-Tuning Hooks

Synthetic-only training remains the default. To prepare expert-annotated real crops from image, label, and JFilament snake triplets, first build a real crop manifest:

```bash
python scripts/build_real_annotation_crops.py \
  --manifest data_manifests/real_annotations.csv \
  --image-root /ssd/STED_dataset/data \
  --annotation-root /cephfs/mhuang/STED_dataset/manual_annotations \
  --out runs/real_annotation_crops \
  --output-manifest data_manifests/real_crops.csv \
  --crop-size 128 \
  --stride 128
```

Inventory-backed crops inherit their parent image and `experimental_group_id` split. Legacy `--leave-one-out-image SAMPLE_ID` mode holds out the selected image's entire split group, not only that image. Crops are never assigned independently.

Mixed synthetic and real training is opt-in:

For leakage-safe grouped cross-validation and two-stage fine-tuning, use the fixed `real_annotation_folds_v1.csv` workflow in `docs/real_finetuning_cross_validation.md`. Real validation checkpoint selection is macro-averaged per parent image; outer test rows are not used during training.

```bash
python scripts/train_first_baseline.py \
  --manifest data_manifests/training_v0_schema08.csv \
  --real-manifest runs/real_annotation_crops/real_crop_manifest.csv \
  --synthetic-real-ratio 80:20 \
  --init-checkpoint runs/first_baseline_schema08_v1b_60ep_wandb/model.pt \
  --out runs/first_baseline_mixed_smoke
```

Add `--enable-uncertainty-head --lambda-uncertainty 0.1` to supervise `uncertain_ignore` as an auxiliary target. Without those flags, `uncertain_ignore` pixels remain excluded from semantic and skeleton losses and the model stays on the original 3-class semantic plus skeleton path.

Synthetic bundle diagnostics can be generated without training:

```bash
python scripts/build_synthetic_bundle_diagnostics.py \
  --manifest data_manifests/training_v1b_schema08.csv \
  --out reports/synthetic_bundle_diagnostics
```

## W&B Offline Tracking

Offline mode records a local W&B run without requiring login or network access during training:

```bash
conda run --no-capture-output -n fibras python scripts/train_first_baseline.py \
  --manifest data_manifests/training_v0_schema08.csv \
  --out runs/first_baseline_schema08_v0_50ep_wandb_offline \
  --epochs 50 \
  --batch-size 4 \
  --patch-size 128 \
  --patches-per-epoch 1024 \
  --cache-samples \
  --seed 123 \
  --wandb \
  --wandb-mode offline \
  --wandb-project sted-unet \
  --wandb-run-name first_baseline_schema08_v0_50ep_offline
```

Sync an offline run later with:

```bash
conda run --no-capture-output -n fibras wandb sync runs/first_baseline_schema08_v0_50ep_wandb_offline/wandb/offline-run-*
```

## W&B Sweeps

A small example sweep config is available at `configs/training/first_baseline_schema08_wandb_sweep.yaml`.

Start a sweep with:

```bash
conda run --no-capture-output -n fibras wandb sweep configs/training/first_baseline_schema08_wandb_sweep.yaml
```

Then run the agent command printed by W&B. The training script accepts both hyphenated CLI names, such as `--lambda-skeleton`, and underscore names, such as `--lambda_skeleton`, so W&B sweep parameters can override the same options cleanly.

## Cached Smoke Run

`--cache-samples` preloads the selected NPZ samples into process memory. For the 64-image candidate dataset this avoids reopening compressed full-size NPZ files for every patch.

```bash
conda run --no-capture-output -n fibras python scripts/train_first_baseline.py \
  --manifest data_manifests/training_candidate_schema08.csv \
  --out runs/first_baseline_schema08_cache_smoke \
  --epochs 2 \
  --batch-size 4 \
  --patch-size 128 \
  --cache-samples \
  --seed 123
```

## Tiny Overfit Debug

Use this to verify that the loader, losses, and heads can overfit a tiny subset before running a longer baseline:

```bash
conda run --no-capture-output -n fibras python scripts/train_first_baseline.py \
  --manifest data_manifests/training_candidate_schema08.csv \
  --out runs/first_baseline_schema08_overfit_debug \
  --epochs 20 \
  --batch-size 4 \
  --patch-size 128 \
  --limit-train-samples 2 \
  --limit-val-samples 2 \
  --patches-per-epoch 64 \
  --cache-samples \
  --seed 123
```

For quick CI/manual checks, reduce `--epochs`.
