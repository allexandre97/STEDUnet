# Synthetic STED Fiber MVP Pipeline

This pipeline implements the bounded MVP described in `docs/synthetic_sted_mvp_plan.md`.

It provides:

- reproducible real-data and expert-validated blank inventories;
- provisional manifest-driven splits;
- deterministic continuous fiber geometry;
- semantic, centerline, endpoint, true-junction, apparent-crossing, overlap-count, and sparse membership targets;
- artificial-background rendering with a simple pixel-space PSF;
- NPZ plus JSON sample storage;
- validation and visualization utilities.

It intentionally does not implement model code, neural-network training, real blank compositing, physical detector simulation, or claims of empirical realism.

## Portable tests

```bash
python -m pytest -q
```

Portable tests use generated TIFF fixtures and do not require `/ssd/STED_dataset`.

## Local real-data inventory

```bash
python scripts/inventory_sted_data.py \
  --fiber-root-id sted_fiber_data \
  --fiber-dir /ssd/STED_dataset/data \
  --blank-root-id sted_blank_data \
  --blank-dir /ssd/STED_dataset/bigblank \
  --out data_manifests \
  --report reports/data_inventory.md
```

## Local validation and examples

```bash
python scripts/validate_sted_inventory.py \
  --fiber-dir /ssd/STED_dataset/data \
  --blank-dir /ssd/STED_dataset/bigblank \
  --manifests data_manifests

python scripts/create_sted_splits.py \
  --inventory-dir data_manifests \
  --strategy pn_holdout \
  --out data_manifests/sted_splits.csv \
  --report reports/sted_split_report.md

python scripts/validate_sted_splits.py \
  --inventory-dir data_manifests \
  --splits data_manifests/sted_splits.csv

python scripts/generate_synthetic_examples.py \
  --config configs/synthetic_sted/mvp_rendering.yaml \
  --out examples/synthetic_sted_rendered

python scripts/validate_synthetic_samples.py examples/synthetic_sted_rendered

python scripts/visualize_synthetic_samples.py \
  examples/synthetic_sted_rendered \
  --out reports/synthetic_mvp_visuals
```

## Split policy

`PN###` identifies a biological sample and is the indivisible primary partitioning unit. All rows sharing a PN must have exactly one primary role among `calibration`, `training`, `validation`, and `held_out_test`.

Condition and DIV are retained for stratification reports but do not define grouping boundaries. Round and series are metadata and reporting variables. The primary held-out test consists only of PNs absent from calibration, training, and validation. Splits remain `human_approved=false` until the PN allocation and condition/DIV stratum counts are reviewed.

Secondary image-level evaluation roles may be recorded separately in `secondary_image_eval_role`, but they must not be mixed with the primary PN-held-out metric.

## Manual MVP review

- Review status: manually reviewed
- Reviewer: project owner
- Result: approved for continued development
- Review date: 2026-06-18
- Scope: geometry, crossings versus junctions, semantic masks, instance memberships, centerlines, endpoint/junction/crossing maps, overlap maps, artificial rendering, and overlays

Do not modify the approved geometry or structural-target conventions unless a concrete defect is discovered.
