# Real Annotation Inventory Workflow

The authoritative machine-readable inventory is `data_manifests/real_annotations.csv`. It joins private expert annotations to stable identities and checksums in `sted_images.csv`. `real_annotation_crops.csv` is a deterministic crop plan; it does not contain private image or annotation data.

## Regenerate And Validate

The expert-adjudicated overlap rule assigns overlapping pixels to
`uncertain_ignore`. Regenerate the inventory and audit with strict validation:

```bash
/lmb/home/alexandrebg/miniconda3/envs/fibras-kb/bin/python scripts/build_real_annotation_inventory.py \
  --annotation-root /cephfs/mhuang/STED_dataset/manual_annotations \
  --image-root /ssd/STED_dataset/data
```

Check that committed artifacts are deterministic:

```bash
/lmb/home/alexandrebg/miniconda3/envs/fibras-kb/bin/python scripts/build_real_annotation_inventory.py \
  --annotation-root /cephfs/mhuang/STED_dataset/manual_annotations \
  --image-root /ssd/STED_dataset/data \
  --check
```

Use `--allow-invalid` only while auditing a newly discovered unresolved problem.

## Build Private Crop Arrays

```bash
/lmb/home/alexandrebg/miniconda3/envs/fibras-kb/bin/python scripts/build_real_annotation_crops.py \
  --manifest data_manifests/real_annotations.csv \
  --image-root /ssd/STED_dataset/data \
  --annotation-root /cephfs/mhuang/STED_dataset/manual_annotations \
  --out runs/real_annotation_crops \
  --output-manifest runs/real_annotation_crops/manifest.csv \
  --crop-size 128 \
  --stride 128
```

Only inventory rows with `validation_status=valid` are materialized. Generated targets preserve class 255, clip skeletons to class 1, and include `real_compatible_skeleton_valid_mask`.

## Evaluate Full Images

```bash
/lmb/home/alexandrebg/miniconda3/envs/fibras-kb/bin/python scripts/evaluate_real_annotation_batch.py \
  --manifest data_manifests/real_annotations.csv \
  --image-root /ssd/STED_dataset/data \
  --annotation-root /cephfs/mhuang/STED_dataset/manual_annotations \
  --run-dir runs/first_baseline_schema08_v0_50ep_wandb \
  --out reports/real_annotation_batch_eval
```

Invalid inventory rows are excluded. Do not use evaluation output as evidence of biological replicate independence until preparation or replicate metadata are resolved.
