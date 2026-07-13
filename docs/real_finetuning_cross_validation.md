# Real-Image Fine-Tuning And Grouped Cross-Validation

## Fixed Fold Contract

`data_manifests/real_annotation_folds_v1.csv` is the fixed fold assignment. Experiments must consume it rather than regenerate folds. It contains five outer folds and a grouped validation partition inside each outer development set.

The indivisible unit is the inventory `split_group_id`, currently equal to provisional `experimental_group_id`. `PN###` remains `culture_id`, not a replicate identifier. Every image appears in outer test exactly once, and every crop inherits its parent image partition.

Five folds are feasible across 12 groups, but perfect stratification is impossible. CBD has one four-image group, PID has only two groups, and several disease/isoform/DIV strata occur once. Consequently, one test fold is CBD-only and one is PID-only. Reducing the fold count does not create additional independent CBD or PID preparations, so five folds retain more test reuse without claiming balanced biological replication.

Two images have unresolved semantic overlaps. They remain assigned in the fold manifest to preserve complete inventory accounting, but `validation_status=invalid` excludes them from crop materialization, training, validation, and test evaluation.

Regenerate only to verify the committed version:

```bash
python scripts/build_real_annotation_folds.py --check
```

## Fold-Specific Crops

Materialize private crops for one fold:

```bash
python scripts/build_real_annotation_crops.py \
  --manifest data_manifests/real_annotations.csv \
  --image-root /ssd/STED_dataset/data \
  --annotation-root /cephfs/mhuang/STED_dataset/manual_annotations \
  --fold-manifest data_manifests/real_annotation_folds_v1.csv \
  --outer-fold 0 \
  --out runs/real_annotation_crops/fold_0 \
  --crop-size 128 \
  --stride 128
```

The crop manifest records parent image, split group, inherited partition, annotation pixel counts, and sampling categories. Target arrays preserve class 255 and an explicit skeleton-valid mask.

## Sampling

`--synthetic-real-ratio` controls each batch, not concatenated dataset size:

- `0:100`: real only;
- `90:10`: 90% synthetic, 10% real;
- `80:20`: 80% synthetic, 20% real;
- `50:50`: equal synthetic and real.

Choose a batch size that represents the requested ratio exactly, such as 10 for `90:10` and 5 for `80:20`. Real sampling first chooses an eligible parent image uniformly, then a weighted category, then one crop. This prevents dense images or images with more crops from dominating.

Configured categories are `fibrous_positive`, `clump_positive`, `uncertain_positive`, `dense_or_fibrous_clump_boundary`, `background_hard_negative`, and `uniform_random`. Set weights with `--real-sampling-weights NAME=WEIGHT,...`.

## Two-Stage Fine-Tuning

`--init-checkpoint` loads an existing synthetic checkpoint. During `--stage1-epochs`, encoder and context parameters are frozen while decoder and output heads train at `--stage1-learning-rate`. Stage 2 unfreezes the complete model and uses the lower `--learning-rate`.

`--resume-checkpoint` resumes model, optimizer, epoch, best metric, patience, validation history, and fine-tuning stage from a complete checkpoint. It is mutually exclusive with `--init-checkpoint`.

With a real manifest, checkpoint selection must use a `macro_image_*` metric and defaults to `macro_image_fibrous_dice`. The trainer also records macro per-image fibrous area ratio, clump leakage, and two-pixel-tolerant skeleton recovery. Pixels are accumulated within each parent image, then metrics are macro-averaged; outer test crops are never loaded for checkpoint selection.

## GPU Selection

Training and evaluation accept `--device cpu`, `--device cuda:0`, or `--device cuda:1`. Runs log the logical device, visible GPU names, selected GPU name, CUDA version, and `CUDA_VISIBLE_DEVICES`. No script enables multi-GPU training.

Logical CUDA indices are relative to visibility. For example:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_first_baseline.py ... --device cuda:0
```

Here logical `cuda:0` maps to physical GPU 1. Run separate folds concurrently rather than assigning both GPUs to one process.

## Example Concurrent Folds

GPU 0, fold 0, mixed `80:20`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_first_baseline.py \
  --manifest data_manifests/training_v2_uncertain_schema08.csv \
  --real-manifest runs/real_annotation_crops/fold_0/real_crop_manifest.csv \
  --synthetic-real-ratio 80:20 \
  --init-checkpoint runs/first_baseline_schema08_v2_uncertain_60ep_wandb/model.pt \
  --out runs/real_finetune/fold_0 \
  --stage1-epochs 5 --stage1-learning-rate 1e-3 --learning-rate 1e-4 \
  --best-metric macro_image_fibrous_dice --save-best-checkpoint \
  --uncertain-skeleton-policy ignore --device cuda:0
```

GPU 1, fold 1, mixed `50:50`:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_first_baseline.py \
  --manifest data_manifests/training_v2_uncertain_schema08.csv \
  --real-manifest runs/real_annotation_crops/fold_1/real_crop_manifest.csv \
  --synthetic-real-ratio 50:50 \
  --init-checkpoint runs/first_baseline_schema08_v2_uncertain_60ep_wandb/model.pt \
  --out runs/real_finetune/fold_1 \
  --stage1-epochs 5 --stage1-learning-rate 1e-3 --learning-rate 1e-4 \
  --best-metric macro_image_fibrous_dice --save-best-checkpoint \
  --uncertain-skeleton-policy ignore --device cuda:0
```

Evaluate only the fold's valid outer test images:

```bash
python scripts/evaluate_real_annotation_batch.py \
  --manifest data_manifests/real_annotations.csv \
  --image-root /ssd/STED_dataset/data \
  --annotation-root /cephfs/mhuang/STED_dataset/manual_annotations \
  --fold-manifest data_manifests/real_annotation_folds_v1.csv \
  --outer-fold 0 --partition test \
  --checkpoint runs/real_finetune/fold_0/model_best.pt \
  --out reports/real_finetune/fold_0_test --device cuda:0
```
