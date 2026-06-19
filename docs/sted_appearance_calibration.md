# STED Appearance Calibration and Real-Blank Compositing

## Status

This phase is exploratory. It characterizes real fiber-containing STED images and expert-validated blank STED images, then generates a small set of real-blank composite examples.

All outputs remain `exploratory_unpartitioned` or `exploratory_provisional_split`. Do not label any synthetic image `empirically_matched` until biological grouping, an approved calibration subset, quantitative comparisons, and visual comparisons have been reviewed.

## Manual review record for geometry MVP

- Review status: manually reviewed
- Reviewer: project owner
- Result: approved for continued development
- Review date: 2026-06-18
- Scope: geometry, crossings versus junctions, semantic masks, instance memberships, centerlines, endpoint/junction/crossing maps, overlap maps, artificial rendering, and overlays

The approved geometry and structural-target conventions must not be modified unless a concrete defect is discovered.

## Metadata model

`PN###` is the original culture identifier (`culture_id`). `3R` and `4R` are tau isoforms. `AD`, `PID`, `PSP`, and `CBD` are disease conditions. `DIV` is days in vitro and is stored as both an integer time point and the original token. `Series N` identifies an imaging series or field of view.

Therefore:

- all current real-image split assignments remain provisional;
- all split rows remain `human_approved=false`;
- no current split is claimed to be biologically independent;
- culture, disease, tau isoform, DIV, experimental condition, experimental group, and series remain separate metadata fields;
- no model training is part of this phase.

## Commands

```bash
python scripts/characterize_sted_appearance.py \
  --config configs/synthetic_sted/mvp_3d_normalized_rasterizer.yaml \
  --inventory-dir data_manifests \
  --splits data_manifests/sted_splits.csv \
  --out calibration_artifacts/exploratory

python scripts/create_sted_blank_pools.py \
  --inventory-dir data_manifests \
  --out data_manifests/sted_blank_pools.csv \
  --report reports/sted_blank_pool_report.md

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

python scripts/validate_calibration_artifacts.py \
  --artifact-dir calibration_artifacts/exploratory \
  --synthetic-dir examples/synthetic_sted_3d_realism \
  --composite-dir examples/sted_blank_composites_3d_realism \
  --pure-blank-dir examples/pure_blank_qa

python scripts/build_sted_calibration_report.py \
  --config configs/synthetic_sted/mvp_3d_normalized_rasterizer.yaml \
  --artifact-dir calibration_artifacts/exploratory \
  --artificial-dir examples/synthetic_sted_3d_realism \
  --composite-dir examples/sted_blank_composites_3d_realism \
  --pure-blank-dir examples/pure_blank_qa \
  --out reports/sted_appearance_calibration.md

python scripts/build_composite_visibility_review.py \
  --composite-dir examples/sted_blank_composites_3d_realism \
  --artifact-dir calibration_artifacts/exploratory \
  --real-fiber-root /ssd/STED_dataset/data \
  --out reports/composite_visibility_review

python scripts/run_foreground_intensity_sweep.py \
  --base-config configs/synthetic_sted/mvp_3d_normalized_rasterizer.yaml \
  --sweep-config configs/synthetic_sted/foreground_intensity_sweep.yaml \
  --artifact-dir calibration_artifacts/exploratory \
  --composite-dir examples/sted_blank_composites_3d_realism \
  --inventory-dir data_manifests \
  --splits data_manifests/sted_splits.csv \
  --out calibration_artifacts/foreground_intensity_sweep \
  --report-dir reports/foreground_intensity_sweep
```

## Measurement classes

Direct measurements include intensity percentiles, zero and saturation fractions, local mean/variance distributions, row/column variation, low-frequency gradients, high-frequency residual statistics, power spectra, autocorrelation, and image-to-image variability.

Proxy-derived measurements include foreground occupancy, ridge response, orientation, apparent width, local contrast/SNR, connected-component length, endpoint-candidate density, and crossing-candidate density. They are not ground truth and must not be silently converted into generator parameters.

## Real-blank compositing

The compositor uses the observed blank image as the authoritative background. It renders the normalized 3D synthetic foreground separately, adds it to the observed blank, optionally adds a small empirical signal-dependent perturbation to the added signal only, and applies one final explicit float-to-uint8 mapping.

It does not infer photon counts, detector gain, or Poisson calibration, and it does not alter structural targets during compositing.

Composite review panels separate stored intensity from display contrast:

- raw panels always use fixed limits `0–255`;
- shared panels use one global real-fiber p0.5–p99.5 range for real, blank, artificial, and composite images;
- independent per-image and foreground-only stretches are labelled display aids and are not quantitative transformations.

The bounded foreground-intensity sweep holds geometry, PSF, blanks, seeds, and output mapping fixed. With signal-dependent perturbation disabled, fluorophore-density and compositor scales are mathematically redundant, so the exploratory sweep varies their effective product without changing the production configuration.

## Normalized foreground convention

Synthetic foreground signal uses an empirical line-density convention:

```text
signal = fluorophore_density_per_unit_length × represented_arc_length × unit_integral_PSF
```

Each full finite-support discrete PSF kernel is normalized to unit integral before image-boundary clipping. Signal outside the field of view is lost, not renormalized into edge pixels. The default review configuration uses `core_plus_halo`; `single_gaussian` remains available for controlled comparisons. Apparent in-focus fiber width is measured from the floating-point transverse profile before uint8 clipping.

Only samples with `scenario_category: realism_calibration` enter appearance statistics. Structural fixtures and optical QA fixtures are generated and visualized separately and must not be averaged into p99, occupancy, spectrum, or morphology summaries.

## Annotation-ready targets

Synthetic samples store projected trace arrays compatible with future JFilament conversion:

- `trace_points_xy`
- `trace_point_offsets`
- `trace_ids`
- `trace_status`
- `trace_source`

They also record `target_available`, `semantic_mask_source`, `ignore_mask`, centerline targets, sparse instance memberships, optical/depth targets, and full blank provenance for composites. Future real annotations can use the same target-availability contract while leaving synthetic-only depth and optical decomposition targets unavailable.

Distance and source fields are explicitly named:

- `background_distance_to_semantic_foreground`: zero inside `semantic_mask`, Euclidean pixel distance outside.
- `line_source_float`: canonical empirical line-source raster with integral equal to emitted source signal.
- `geometric_support_preview`: finite-radius geometry visualization only; not an energy-preserving source raster.

Visible instance provenance is split into individual visible memberships, all contributing memberships, and `combined_only_visible_mask` for summed-subthreshold visibility.

The reserved multiclass real-annotation and trace-termination vocabulary is defined in `docs/real_annotation_contract.md`. Current synthetic masks remain binary; bundle and clump simulation is intentionally deferred.

Calibration artifacts record the full source commit SHA, dirty-worktree state, configuration SHA-256, and inventory, split, and blank-pool manifest SHA-256 values separately. A dirty generation is reported as dirty rather than presented as a clean revision.

## Blank pools and pure blank QA

Blank backgrounds are assigned to provisional roles in `data_manifests/sted_blank_pools.csv`. The roles are independent of disease/DIV split logic:

- `synthetic_background_train`
- `synthetic_background_validation`
- `synthetic_background_test`
- `pure_blank_qa`

The `pure_blank_qa` artifacts contain zero foreground/skeleton/junction targets and preserve blank provenance for future false-positive and spurious-skeleton evaluation. No model QA is run in this phase.
