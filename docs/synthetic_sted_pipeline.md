# Synthetic STED Fiber MVP Pipeline

This pipeline implements the bounded MVP described in `docs/synthetic_sted_mvp_plan.md`.

It provides:

- reproducible real-data and expert-validated blank inventories;
- provisional manifest-driven splits;
- deterministic continuous fiber geometry;
- semantic, centerline, endpoint, true-junction, apparent-crossing, overlap-count, and sparse membership targets;
- artificial-background rendering with a normalized empirical pixel-space PSF;
- exploratory real-blank compositing using expert-validated blank backgrounds;
- NPZ plus JSON sample storage;
- validation and visualization utilities.

It intentionally does not implement model code, neural-network training, physical detector simulation, disease-specific renderers, or claims of empirical realism.

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
  --strategy experimental_group_holdout \
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

python scripts/create_sted_blank_pools.py \
  --inventory-dir data_manifests \
  --out data_manifests/sted_blank_pools.csv \
  --report reports/sted_blank_pool_report.md
```

## Split policy

`PN###` is recorded as `culture_id`. `3R` and `4R` are tau isoforms. Disease labels such as `AD`, `PID`, `PSP`, and `CBD`, tau isoform, DIV time point, and series are preserved as separate metadata fields.

Primary development splits use `experimental_group_id = culture_id + disease + tau_isoform + div` as the indivisible grouping unit. All series from one experimental group remain in one role, with condition and DIV used for stratification reports when enough groups exist. Splits remain `human_approved=false` until reviewed.

A separate `culture_held_out` split strategy is supported for future culture-batch sensitivity analysis. Its results must not be mixed with ordinary experimental-group-held-out metrics.

## Manual MVP review

- Review status: manually reviewed
- Reviewer: project owner
- Result: approved for continued development
- Review date: 2026-06-18
- Scope: geometry, crossings versus junctions, semantic masks, instance memberships, centerlines, endpoint/junction/crossing maps, overlap maps, artificial rendering, and overlays

Do not modify the approved geometry or structural-target conventions unless a concrete defect is discovered.

## 3D persistent-chain rasterizer

The approved 2D structural generator remains available as `structural_test_2d` / `legacy_2d` for exact crossing, Y-junction, clipping, and regression tests. The realistic morphology path is separate and explicit:

```bash
python scripts/generate_synthetic_examples.py \
  --config configs/synthetic_sted/mvp_3d_rasterizer.yaml \
  --out examples/synthetic_sted_3d_rasterizer

python scripts/validate_synthetic_samples.py examples/synthetic_sted_3d_rasterizer

python scripts/visualize_synthetic_samples.py \
  examples/synthetic_sted_3d_rasterizer \
  --out reports/synthetic_3d_rasterizer_visuals
```

The 3D path uses pixel-equivalent coordinates `(x, y, z)`, a focal plane in the same units, persistent-chain curves, arc-length resampling, correlated fluorophore line-density amplitudes, and direct depth-dependent 2D Gaussian splatting. It is an empirical effective optical model, not a calibrated physical STED PSF.

The current normalized 3D sample schema is `synthetic_sted_3d_rasterizer_0.5.0`. It preserves the 0.4 conventions and adds crossing fiber and segment identities. Existing 0.2, 0.3, and 0.4 artifacts are validated under their original field names and are not silently reinterpreted.

PSF boundary convention: each full finite-support discrete kernel is normalized before image clipping. Signal outside the field of view is lost and is not renormalized into edge pixels.

Distance target convention: `background_distance_to_semantic_foreground` is zero inside `semantic_mask` and exact Euclidean pixel distance outside to the nearest semantic foreground pixel. Schema 0.4+ generation requires SciPy's `distance_transform_edt`; it never silently falls back to an approximate chamfer transform.

Source raster convention: `line_source_float` is the canonical pre-optical line-source raster whose integral matches emitted empirical line signal. `geometric_support_preview` is only a finite-radius visualization/support preview and has no source-energy interpretation.

Orientation convention: doubled-angle orientation arrays are valid only where `orientation_valid_mask == 1`. Tangent consensus is required within each fiber and across fibers, so self-crossings, tight loops, degenerate tangents, and incompatible multi-instance crossings are invalid. Parallel and anti-parallel traces remain compatible.

The authoritative visibility threshold is `optical_model.visible_signal_threshold`. The deprecated `targets.visible_signal_threshold` alias is accepted only when it is equal; conflicting values fail validation.

PSF component weights use `weights_sum_to_one`. Core-plus-halo weights must be finite, nonnegative, have positive total, and sum to one without implicit normalization.

The shared real-annotation vocabulary is defined in `docs/real_annotation_contract.md`. The normalized production path remains schema 0.5 and binary. The separate exploratory morphology path uses schema `synthetic_sted_3d_morphology_0.8.0` with apparent visible background, individual-filament, bundle, clump, and uncertain/ignore masks plus separate latent source-support masks.

Generate the normalized 3D review set with:

```bash
python scripts/generate_synthetic_examples.py \
  --config configs/synthetic_sted/mvp_3d_normalized_rasterizer.yaml \
  --out examples/synthetic_sted_3d_realism

python scripts/validate_synthetic_samples.py examples/synthetic_sted_3d_realism

python scripts/visualize_synthetic_samples.py \
  examples/synthetic_sted_3d_realism \
  --out reports/synthetic_3d_realism_visuals
```

Structural and optical QA fixtures are separated from realism-calibration examples:

```bash
python scripts/generate_synthetic_examples.py \
  --config configs/synthetic_sted/rasterizer_structural_qa.yaml \
  --out examples/rasterizer_structural_qa

python scripts/generate_synthetic_examples.py \
  --config configs/synthetic_sted/rasterizer_optical_qa.yaml \
  --out examples/rasterizer_optical_qa
```

Only `scenario_category: realism_calibration` samples should enter real-versus-synthetic appearance statistics.

## Morphology heterogeneity review

The condition-blind `morphology_scene_3d` mode adds spatial domains, isolated and clustered filaments, explicit correlated bundles, compact clumps, and mixed transition scenes. It does not read culture, disease, tau isoform, or DIV metadata. Real-image morphology measurements remain coarse diagnostics rather than generator-fitting targets.

```bash
python scripts/generate_synthetic_examples.py \
  --config configs/synthetic_sted/morphology_heterogeneity_review.yaml \
  --out examples/synthetic_sted_morphology_review

python scripts/generate_blank_composite_examples.py \
  --config configs/synthetic_sted/morphology_heterogeneity_review.yaml \
  --inventory-dir data_manifests \
  --splits data_manifests/sted_splits.csv \
  --out examples/sted_blank_composites_morphology_review

python scripts/build_morphology_review.py \
  --config configs/synthetic_sted/morphology_heterogeneity_review.yaml \
  --synthetic-dir examples/synthetic_sted_morphology_review \
  --composite-dir examples/sted_blank_composites_morphology_review \
  --previous-composite-dir examples/sted_blank_composites_3d_realism \
  --artifact-dir calibration_artifacts/exploratory \
  --out reports/synthetic_morphology_semantic_review \
  --diagnostics-out calibration_artifacts/morphology_semantic_review
```

The multiclass `semantic_class_mask` uses values `0`, `1`, `2`, `3`, and `255`. `semantic_mask` is the compatibility union of valid foreground classes 1–3. Schema 0.8 exposes apparent visible supervised masks, separate `*_source_support_mask` latent provenance masks, `supervised_membership_*` and `latent_geometry_membership_*` separately, plus numeric graph supervision and boundary flags. Bundle axes are bundle-level targets, not filament centerlines. Hidden bundle children and clump fragments remain synthetic-only latent geometry.

Boundary handling inserts the first analytic volume intersection and terminates the curve; it never flattens a sequence of points onto an edge. Only `valid_endpoint` trace ends enter `endpoint_map`. Resolved bundle-child segments may contribute filament crossings, while unresolved and transition segments remain crossing-ineligible.

See `docs/synthetic_morphology_schema_migration.md` for the explicit 0.6-to-0.7 field migration.

## Real-blank compositing and blank QA

Expert-validated blanks are assigned to dedicated provisional roles: `synthetic_background_train`, `synthetic_background_validation`, `synthetic_background_test`, and `pure_blank_qa`. Complete blank source images and acquisition groups stay together where possible, and all assignments remain `human_approved=false` until reviewed.

```bash
python scripts/generate_blank_composite_examples.py \
  --config configs/synthetic_sted/mvp_3d_normalized_rasterizer.yaml \
  --inventory-dir data_manifests \
  --splits data_manifests/sted_splits.csv \
  --out examples/sted_blank_composites_3d_realism

python scripts/generate_pure_blank_qa_examples.py \
  --config configs/synthetic_sted/mvp_3d_normalized_rasterizer.yaml \
  --inventory-dir data_manifests \
  --pools data_manifests/sted_blank_pools.csv \
  --out examples/pure_blank_qa \
  --count 8
```

Composites use globally unique IDs such as `blank_composite_3d_realism_0000` and record `parent_synthetic_sample_id` plus the parent synthetic NPZ hash.
