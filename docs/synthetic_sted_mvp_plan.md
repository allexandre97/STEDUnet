# Synthetic STED Fiber Data Pipeline MVP Plan

Note: this document records the original bounded MVP plan. Later approved phases added the 3D persistent-chain rasterizer, normalized arc-length foreground signal, unit-integral effective PSF kernels, exploratory real-blank compositing, blank-pool roles, and annotation-ready trace/target metadata. Use `docs/synthetic_sted_pipeline.md` and `docs/sted_appearance_calibration.md` for current operational commands.

## Status

This document defines the approved implementation plan for a bounded synthetic-data MVP.

It authorizes implementation of:

- reproducible STED data inventories;
- authoritative source and split manifests;
- continuous synthetic fiber geometry;
- exact structural targets;
- artificial-background rendering;
- a simple configurable pixel-space PSF;
- deterministic example generation;
- validation and visualization.

It does **not** authorize:

- neural-network implementation;
- model training;
- large-scale synthetic dataset generation;
- compositing onto real blank images;
- physical detector or photon-count simulation;
- learned appearance transfer;
- claims that the synthetic distribution is realistic before quantitative and visual calibration.

The implementation must remain within the scope defined here unless the plan is explicitly revised.

---

## 1. Literature evidence and project rationale

The reviewed knowledge base supports synthetic data as a useful source of exact structural supervision, but not as a substitute for real-data evaluation.

Relevant evidence:

- `Liu18` supports 2D microscopy filament segmentation and semi-automatic annotation, but only for confocal semantic-mask evidence.
- `Shi20b` supports testing Dice plus soft-clDice for connectivity preservation, but does not solve crossing or instance continuity and is not validated on this STED distribution.
- `Tet18` supports synthetic pretraining and auxiliary centerline or bifurcation supervision, but its evidence comes from 3D angiography.
- `Liu20` supports synthetic endpoint and junction targets with graph-oriented postprocessing, but the real-image validation is limited.
- `Liu19` supports crossing-aware synthetic supervision and orientation-aware reasoning, while also demonstrating that synthetic assumptions can introduce orientation and pairing biases.
- `Xu15` and `Togo26` support synthetic topology benchmarks and classical tracing, but not automatic ground-truth generation for real STED images.
- `Ozd21` identifies annotation burden, density, low signal-to-noise ratio, intersections, and topology-aware evaluation as recurring risks.

The project therefore uses synthetic data for:

- exact geometry;
- exact instance identities;
- exact centerlines;
- exact crossings and true junctions;
- controlled density and topology;
- future pretraining and ablation experiments.

Real STED images remain necessary for:

- appearance calibration;
- generator validation;
- manual annotation;
- fine-tuning;
- held-out evaluation.

The success criterion is improvement on held-out, manually reviewed real STED data, not visual plausibility of synthetic images alone.

---

## 2. Current real-data inventory baseline

The following values come from an exploratory exact all-pixel pass and must be reproduced by committed inventory code.

### 2.1 Inventory methodology

- TIFF reader: `PIL.Image` / Pillow.
- Array conversion: `numpy.asarray(Image.open(path))`.
- Reported statistics: exact over all pixels and all files; no sampling.
- Full-file duplicate identity: SHA-256 of source file bytes.
- Decoded-pixel duplicate identity: SHA-256 of the decoded pixel array plus shape and dtype.
- Thumbnail duplicate identity:
  - robust-normalize each image using per-image p1 and p99;
  - resize deterministically to a configured thumbnail size;
  - hash the thumbnail bytes.
- Near-duplicate candidates:
  - must use a similarity metric rather than exact thumbnail hash equality;
  - acceptable options include perceptual hash distance, normalized cross-correlation, SSIM, or a documented combination;
  - candidate generation and thresholds must be deterministic and recorded.

The current elongated-component quality-control screen is:

- threshold:
  - `max(p99.9, median + 8 * 1.4826 * MAD, 20)`;
- connected components on `image > threshold`;
- flag components with:
  - area `>= 12 px`;
  - elongation `>= 2.5`.

This screen is only for quality control and acquisition-artifact reporting. It is not used to determine whether blank images contain fibers.

### 2.2 `/ssd/STED_dataset/data`

- 438 `.tif` files.
- 438 readable.
- All `1024 x 1024`.
- Single-frame.
- Single-channel grayscale.
- PIL mode `L`.
- Dtype `uint8`.
- Bit depth 8.
- Total pixels: 459,276,288.
- Exact global percentiles:
  - p0 = 0
  - p0.1 = 0
  - p1 = 0
  - p5 = 0
  - p25 = 1
  - p50 = 4
  - p75 = 9
  - p95 = 24
  - p99 = 61
  - p99.9 = 145
  - p100 = 255
- Per-image minimum range: 0–0.
- Per-image maximum range: 29–255.
- Pixels equal to 255: 9,683.
- No unreadable, constant, unusually sized, full-file duplicate, decoded-pixel duplicate, or exact-thumbnail duplicate images were detected in the exploratory pass.
- Any future near-duplicate claim must be based on the explicit similarity method described above.

### 2.3 `/ssd/STED_dataset/bigblank`

- 94 `.tif` files.
- 94 readable.
- All `1024 x 1024`.
- Single-frame.
- Single-channel grayscale.
- PIL mode `L`.
- Dtype `uint8`.
- Bit depth 8.
- Total pixels: 98,566,144.
- Exact global percentiles:
  - p0 = 0
  - p0.1 = 0
  - p1 = 0
  - p5 = 0
  - p25 = 1
  - p50 = 4
  - p75 = 8
  - p95 = 16
  - p99 = 23
  - p99.9 = 34
  - p100 = 169
- Per-image minimum range: 0–0.
- Per-image maximum range: 31–169.
- Pixels equal to 255: 0.
- No unreadable, constant, unusually sized, full-file duplicate, decoded-pixel duplicate, or exact-thumbnail duplicate images were detected in the exploratory pass.
- Any future near-duplicate claim must be based on an explicit similarity metric.

The blank images are expert-validated as fiber-free. Their blank status is not inferred from image processing.

---

## 3. Reproducible inventory artifacts

Create:

- `data_manifests/sted_images.csv`
- `data_manifests/sted_blanks.csv`
- `data_manifests/acquisition_groups.csv`
- `data_manifests/sted_splits.csv`
- `reports/data_inventory.md`

Do not rename source files.

### 3.1 Portable source identity

Do not use absolute paths as the authoritative source identity.

Persist:

- `source_root_id`
- `relative_path`
- `source_sha256`
- `pixel_sha256`
- `stable_image_id`

Example source root IDs:

- `sted_fiber_data`
- `sted_blank_data`

Local configuration maps source root IDs to filesystem locations:

```yaml
source_roots:
  sted_fiber_data: /ssd/STED_dataset/data
  sted_blank_data: /ssd/STED_dataset/bigblank
```

A resolved absolute path may appear in a local report, but it must not define deterministic identity.

### 3.2 Image-manifest fields

Each source-image record must include:

- `source_root_id`
- `relative_path`
- `source_kind`
  - `fiber_image`
  - `blank_background`
- `source_sha256`
- `pixel_sha256`
- `stable_image_id`
- `shape_y`
- `shape_x`
- `frames`
- `channels`
- `dtype`
- `bit_depth`
- `min`
- `max`
- `mean`
- `std`
- `p0`
- `p0_1`
- `p1`
- `p5`
- `p25`
- `p50`
- `p75`
- `p95`
- `p99`
- `p99_9`
- `p100`
- `saturation_value`
- `saturation_count`
- `zero_count`
- `culture_id`
- `disease`
- `tau_isoform`
- `experimental_condition`
- `div`
- `div_token`
- `series_index`
- `filename_prefix`
- `acquisition_group`
- `experimental_group_id`
- `parse_status`
- deprecated migration fields such as `deprecated_inferred_round`, if needed
- `file_duplicate_group`
- `pixel_duplicate_group`
- `thumbnail_duplicate_group`
- `near_duplicate_candidate_group`
- `near_duplicate_score`
- `near_duplicate_method`
- `validation_flags`
- `intended_split_status`

### 3.3 Blank provenance fields

Blank records additionally include:

- `blank_status: expert_validated`
- `validation_source: human_expert_review`
- `validator_role: STED expert`
- `validation_date: not_recorded`, unless known
- `notes`

Automated checks may still flag:

- unreadable or corrupted files;
- unexpected shape, dtype, channels, or frames;
- saturation;
- extreme intensity outliers;
- duplicate or near-duplicate candidates;
- unusual row or column artifacts;
- unusual background distributions.

These are quality-control flags only. Do not reinterpret blank validity.

No image may be silently excluded.

---

## 4. Duplicate and near-duplicate handling

Use distinct categories.

### 4.1 `file_duplicate`

Two files are file duplicates when their complete file SHA-256 values are identical.

### 4.2 `pixel_duplicate`

Two images are pixel duplicates when their decoded pixel arrays, shapes, and dtypes hash identically, even if TIFF metadata differs.

### 4.3 `thumbnail_duplicate`

Two images are thumbnail duplicates when a deterministic normalized thumbnail representation hashes identically.

This is not a general near-duplicate test.

### 4.4 `near_duplicate_candidate`

A near-duplicate candidate is a pair or group exceeding a documented similarity threshold.

The implementation must record:

- normalization procedure;
- thumbnail size;
- similarity metric;
- threshold;
- candidate IDs;
- score;
- review status.

A staged deterministic search is acceptable:

1. bucket by shape and rough intensity statistics;
2. use perceptual hash or thumbnail descriptors for candidate generation;
3. compute SSIM or normalized cross-correlation for candidate confirmation.

Exhaustive all-pairs comparison is not required when unnecessary.

Unit tests must verify that:

- metadata-only differences produce pixel duplicates;
- identical thumbnails are not mislabelled as general near-duplicates;
- deliberately perturbed fixtures can be detected as near-duplicate candidates under the configured method.

---

## 5. Leakage-safe split strategy

Reserve held-out real data before generator calibration.

### 5.1 Real fiber-image eligibility classes

- `calibration`
- `training`
- `validation`
- `held_out_test`

### 5.2 Experimental-group primary grouping

Use the clarified experimental hierarchy:

- `PN###` is `culture_id`, a culture-record/batch identifier, not the disease condition and not an automatic partitioning unit;
- `3R` and `4R` are `tau_isoform`, not acquisition rounds;
- `AD`, `PID`, `PSP`, and `CBD` are disease conditions;
- `DIV` is days in vitro and is retained as an integer time point plus original token;
- `experimental_condition = disease + '_' + tau_isoform`;
- `experimental_group_id = culture_id + disease + tau_isoform + div`;
- all series belonging to the same `experimental_group_id` remain in exactly one primary role.

Rules:

- never split fields of view or series from the same experimental group across incompatible primary roles;
- use experimental condition and DIV to report stratum coverage and warn when too few experimental groups exist;
- store `human_approved=false` until the proposed allocation and stratum counts are reviewed;
- support a separate `culture_held_out` strategy for culture-batch sensitivity analysis, without mixing it with primary experimental-group-held-out metrics.

The final held-out groups must not influence:

- generator calibration;
- generator parameter fitting;
- visual tuning;
- threshold selection;
- preprocessing selection;
- model training;
- model selection;
- hyperparameter selection.

### 5.3 Split-manifest fields

- `stable_image_id`
- `source_kind`
- `culture_id`
- `disease`
- `tau_isoform`
- `experimental_condition`
- `div`
- `div_token`
- `series_index`
- `experimental_group_id`
- `acquisition_group`
- `eligibility`
- `primary_metric_role`
- `secondary_image_eval_role`
- `assignment_reason`
- `grouping_rule`
- `human_approved`
- `synthetic_split`
- `override_status`
- `override_reason`

The split manifest is authoritative. Downstream code must not re-parse filenames independently.

---

## 6. Blank-background split strategy

Real blank compositing is deferred, but the split schema must be defined now.

Rules:

- assign each complete blank source image to exactly one primary split by experimental group by default;
- keep all blank images belonging to the same experimental group in one split;
- retain blank acquisition group, culture, disease, tau isoform, DIV, and series as metadata;
- do not split crops from one blank image across train, validation, and test;
- do not allow overlapping crops from one blank image to cross incompatible splits;
- when PN grouping cannot be inferred, fail validation or require an explicit reviewed override;
- any cross-split reuse requires an explicit configuration override;
- validation must fail on unapproved blank leakage.

Record:

- blank stable ID;
- blank PN;
- blank acquisition group;
- assigned synthetic split;
- assignment strategy;
- random seed, if applicable;
- override status;
- override reason.

The same blank image or acquisition group must not appear in incompatible synthetic splits under default behavior.

---

## 7. Expert-validated blank provenance

The images in `/ssd/STED_dataset/bigblank` were selected and reviewed by an STED expert and are confirmed to contain no fibers.

Treat them as validated negative backgrounds.

Do not require additional manual validation before using them later.

Synthetic samples that eventually use blanks must record:

- source root ID;
- relative path;
- stable blank ID;
- source SHA-256;
- crop coordinates;
- blank acquisition group;
- synthetic split;
- geometry seed;
- rendering seed;
- renderer configuration;
- dataset schema version;
- generator version.

---

## 8. Canonical geometry model

Represent each fiber as a continuous subpixel curve.

The MVP may use:

- cubic splines;
- piecewise cubic Bézier curves;
- another documented continuously sampled curve model.

Curves must use arc-length-aware or sufficiently dense parameterization so rasterization quality does not depend strongly on control-point spacing.

### 8.1 Sampled geometry parameters

Support configurable distributions over:

- fiber count;
- fiber length;
- curvature;
- tortuosity;
- width;
- width variation;
- intensity;
- intensity variation;
- endpoints;
- boundary clipping;
- gaps or weak segments;
- sparse density;
- moderate density;
- near-parallel fibers;
- short fragments;
- overlaps;
- apparent crossings;
- true junctions.

All spatial parameters are expressed in pixels in the MVP.

Do not report physical units until pixel-size metadata are available.

### 8.2 True junctions and apparent crossings

A true junction is a graph node shared by two or more structurally connected fibers or fiber segments.

An apparent crossing is a geometric intersection or rendered overlap between disconnected fiber identities.

These must remain separate in:

- vector geometry;
- graph connectivity;
- raster targets;
- metadata;
- tests.

---

## 9. Canonical overlapping-instance representation

Do not use an arbitrary primary instance label as the canonical ground truth.

### 9.1 Authoritative representation

Preserve:

- vector centerlines per fiber;
- graph geometry;
- one sparse or ragged binary mask per instance;
- sparse pixel-to-instance memberships;
- overlap count;
- semantic union mask.

An optional flattened single-label instance map may be generated only as a derived compatibility artifact.

### 9.2 Definitions

- `geometric_crossing_point`: subpixel intersection of disconnected centerlines.
- `geometric_crossing_region`: raster tolerance region around a geometric crossing.
- `true_junction_node`: graph node connecting fiber topology.
- `rendered_overlap_region`: pixels covered by two or more clean rendered fiber supports before PSF blur.
- `multi_instance_pixel_membership`: complete set of contributing instance IDs for a pixel.
- `finite_width_ambiguity`: ambiguity caused by intersecting fiber supports.
- `psf_blur_ambiguity`: ambiguity caused by image formation outside the clean support.

### 9.3 Tolerance conventions

Store in metadata:

- centerline raster tolerance;
- crossing-map radius;
- junction-map radius;
- endpoint-map radius;
- PSF ambiguity threshold;
- rasterization convention.

Defaults may be derived from rendered fiber radius, but all values must be explicit and configurable.

---

## 10. Generated targets

Each MVP sample emits:

- vector fiber geometry;
- graph geometry;
- clean floating-point source image;
- floating-point rendered image;
- mapped uint8 image;
- semantic foreground mask;
- sparse per-instance representation;
- centerline raster;
- endpoint map;
- true-junction map;
- apparent-crossing map;
- overlap-count map;
- sparse pixel-to-instance memberships;
- generation parameters;
- rendering parameters;
- geometry seed;
- rendering seed;
- schema version;
- generator version;
- calibration status.

Potential future targets such as distance transforms, tangent orientation, complete graph-learning encodings, and ambiguity maps may be included only when they fit without expanding the MVP materially. The implementation must not broaden scope merely because the schema can support more fields.

---

## 11. NPZ and JSON sample representation

The MVP canonical persisted format is:

- one compressed `.npz` per sample;
- one adjacent `.json` metadata file;
- one dataset-level manifest containing file hashes and sample IDs.

No object arrays.

No `allow_pickle=True`.

No arbitrary code execution required for loading.

### 11.1 Vector centerlines

Use:

- `fiber_points_xy`: concatenated `float32`, shape `(N_points, 2)`;
- `fiber_point_offsets`: integer offsets, shape `(N_fibers + 1,)`;
- `fiber_ids`: integer array, shape `(N_fibers,)`.

### 11.2 Graph

Use:

- `node_xy`: `float32`, shape `(N_nodes, 2)`;
- `node_type`: integer-coded array;
- `edge_node_indices`: integer, shape `(N_edges, 2)`;
- `edge_fiber_id`: integer array;
- optional concatenated edge points plus offsets.

### 11.3 Sparse memberships

Use a coordinate representation such as:

- `membership_y`
- `membership_x`
- `membership_instance_id`

All arrays must use fixed numeric dtypes.

### 11.4 Raster arrays

At minimum:

- `source_float`
- `render_float`
- `render_uint8`
- `semantic_mask`
- `centerline_mask`
- `endpoint_map`
- `junction_map`
- `crossing_map`
- `overlap_count`

Optional derived compatibility output:

- `flattened_instance_map`

### 11.5 Metadata requirements

The JSON metadata defines:

- sample ID;
- dataset schema version;
- generator version;
- geometry seed;
- rendering seed;
- geometry parameters;
- appearance parameters;
- split;
- source blank provenance, when applicable;
- image shape;
- calibration status;
- array names;
- dtypes;
- shapes;
- enum mappings;
- coordinate conventions;
- indexing convention;
- pixel-center convention;
- units;
- tolerance radii;
- rasterization rules.

Use zero-based indexing.

Define coordinates in pixel units.

Use a documented pixel-center convention, for example pixel `(x, y)` centered at `(x + 0.5, y + 0.5)`.

### 11.6 Integrity checksums

Do not embed a self-referential checksum inside a file being hashed.

Use a dataset-level manifest recording:

- sample ID;
- NPZ relative path;
- NPZ SHA-256;
- JSON relative path;
- JSON SHA-256;
- schema version;
- generator version.

---

## 12. Intensity and noise limitations

The TIFF images are uint8 with limited metadata.

Do not assume values are:

- linear in photon count;
- proportional to fluorophore count;
- raw detector measurements;
- physically calibrated.

MVP rendering rules:

- render internally in floating point;
- use explicit configurable mapping to uint8;
- report clipping and saturation introduced by the mapping;
- retain floating-point renders for the small audit example set;
- treat added noise as empirical unless physical calibration becomes available;
- do not infer detector gain;
- do not claim Poisson calibration.

Calibration status values:

- `procedural_unmatched`
- `empirically_matched`
- `approximately_physical`
- `physically_calibrated`

MVP output starts as `procedural_unmatched`.

It may only be promoted to `empirically_matched` after a quantitative and visual calibration report is reviewed.

---

## 13. MVP rendering model

The MVP renderer is intentionally simple.

Implement:

1. supersampled or anti-aliased subpixel geometry rendering;
2. configurable fiber width in pixels;
3. configurable intensity in arbitrary floating-point units;
4. optional along-fiber intensity variation;
5. simple artificial background;
6. simple configurable PSF approximation in pixel units;
7. explicit float-to-uint8 mapping;
8. clipping and saturation reporting.

The PSF may initially be:

- isotropic Gaussian;
- optionally anisotropic Gaussian.

Configuration fields must state units in pixels.

Do not implement:

- real blank compositing;
- photon-count simulation;
- detector-gain estimation;
- spatially varying PSFs;
- bleaching;
- stripe artifacts;
- learned appearance transfer.

---

## 14. Calibration uncertainty handling

Separate calibration quantities into four classes.

### 14.1 Directly observable without segmentation

Examples:

- intensity percentiles;
- saturation frequency;
- spatial power spectrum;
- autocorrelation;
- row variation;
- column variation;
- illumination gradients;
- background distributions.

### 14.2 Proxy-derived

Examples:

- foreground occupancy;
- apparent width;
- local SNR;
- component length;
- curvature;
- orientation;
- endpoint density;
- crossing-candidate density.

Every proxy-derived quantity must record:

- estimator;
- configuration;
- thresholds;
- sensitivity sweep;
- uncertainty or plausible range;
- warning that the estimate is not ground truth.

### 14.3 Manually sampled

Examples:

- fiber-width distributions;
- true versus false crossing examples;
- weak-fiber appearance;
- bundle behavior.

### 14.4 Unavailable until formal annotation

Examples:

- true graph topology;
- true instance continuity;
- true-junction frequency;
- segmentation accuracy;
- centerline accuracy.

When the data do not justify a fitted distribution, use broad bounded parameter distributions rather than false precision.

---

## 15. Authoritative parser, manifests, and split logic

Create one filename parser module and test it.

- inventory code parses filenames;
- inventory manifests store parsed fields;
- split code reads manifests;
- generation code reads split manifests;
- validation code reads manifests;
- no downstream component independently re-parses raw filenames.

Manual split edits remain authoritative input.

Manifest schema versions must be recorded.

Validation must detect:

- unknown source IDs;
- duplicate stable IDs;
- incompatible split leakage;
- missing checksums;
- missing expert blank provenance;
- invalid grouping fields;
- source files absent from the manifest.

---

## 16. Portable testing requirements

Normal unit tests and CI must not require:

- `/ssd/STED_dataset/data`;
- `/ssd/STED_dataset/bigblank`;
- any machine-specific absolute path.

### 16.1 Unit tests

Use generated TIFF fixtures in temporary directories for:

- filename parsing;
- source-root mapping;
- inventory fields;
- percentile calculations;
- hashes;
- file duplicates;
- pixel duplicates;
- thumbnail duplicates;
- near-duplicate candidates;
- blank provenance;
- corrupted files;
- unusual shapes or dtypes;
- split validation.

### 16.2 Integration tests

Use small repository fixtures only when useful.

### 16.3 Local-data validation

Commands accessing `/ssd/STED_dataset` are optional local validation commands and must run only when paths are explicitly provided.

Normal `pytest` must pass without the real dataset.

Document separately:

- portable test commands;
- local real-data inventory commands;
- local real-data validation commands.

---

## 17. Bounded MVP scope

Implement only:

1. reproducible real and blank inventories;
2. portable source identities;
3. authoritative acquisition-group and split manifests;
4. expert blank provenance;
5. continuous synthetic fiber geometry;
6. sparse and moderately dense fields;
7. true junctions;
8. apparent crossings;
9. semantic masks;
10. sparse per-instance memberships;
11. centerlines;
12. endpoint maps;
13. junction maps;
14. crossing maps;
15. overlap-count maps;
16. deterministic artificial-background rendering;
17. simple pixel-space PSF;
18. deterministic regeneration;
19. integrity validation;
20. visualization;
21. 8–16 generated examples.

Deferred:

- real blank compositing;
- detailed detector noise;
- physical photon simulation;
- spatially varying PSF;
- bleaching;
- scan artifacts;
- learned appearance transfer;
- large-scale generation;
- neural-network implementation;
- model training;
- self-supervised learning;
- full graph-learning targets.

---

## 18. MVP milestones, commands, and acceptance criteria

### Milestone 1: Inventory implementation

Command:

```bash
python scripts/inventory_sted_data.py \
  --fiber-root-id sted_fiber_data \
  --fiber-dir /ssd/STED_dataset/data \
  --blank-root-id sted_blank_data \
  --blank-dir /ssd/STED_dataset/bigblank \
  --out data_manifests \
  --report reports/data_inventory.md
```

Expected artifacts:

- `data_manifests/sted_images.csv`
- `data_manifests/sted_blanks.csv`
- `data_manifests/acquisition_groups.csv`
- `reports/data_inventory.md`

Validation:

```bash
python scripts/validate_sted_inventory.py \
  --fiber-dir /ssd/STED_dataset/data \
  --blank-dir /ssd/STED_dataset/bigblank \
  --manifests data_manifests
```

Pass conditions:

- every source file represented exactly once;
- checksums present;
- source root IDs and relative paths used;
- shape and dtype recorded;
- expert blank provenance present;
- no silent exclusions;
- duplicate and near-duplicate categories reported correctly;
- exact statistics match the committed methodology.

### Milestone 2: Provisional split infrastructure

Command:

```bash
python scripts/create_sted_splits.py \
  --inventory-dir data_manifests \
  --strategy experimental_group_holdout \
  --out data_manifests/sted_splits.csv \
  --report reports/sted_split_report.md
```

Validation:

```bash
python scripts/validate_sted_splits.py \
  --inventory-dir data_manifests \
  --splits data_manifests/sted_splits.csv
```

Pass conditions:

- no PN crosses incompatible primary roles;
- held-out PNs are absent from calibration, training, and validation;
- condition and DIV counts are reported by split with low-PN warnings;
- blank images and blank groups have deterministic split assignments;
- no blank source crosses incompatible synthetic splits;
- `human_approved=false` until the PN allocation and stratum counts are reviewed;
- split rules are manifest-driven.

### Milestone 3: Geometry MVP

Command:

```bash
python scripts/generate_synthetic_examples.py \
  --config configs/synthetic_sted/mvp_geometry.yaml \
  --out examples/synthetic_sted_mvp
```

Expected artifacts:

- 8–16 samples;
- NPZ plus JSON per sample;
- dataset-level hash manifest.

Validation:

```bash
python scripts/validate_synthetic_samples.py \
  examples/synthetic_sted_mvp
```

Pass conditions:

- deterministic regeneration;
- valid curve geometry;
- valid true-junction topology;
- valid disconnected crossings;
- valid sparse memberships;
- no object arrays;
- no pickle requirement;
- invalid parameter combinations fail clearly.

### Milestone 4: Target integrity

Command:

```bash
python -m pytest -q \
  tests/test_synthetic_geometry.py \
  tests/test_synthetic_targets.py \
  tests/test_synthetic_schema.py
```

Pass conditions:

- semantic mask equals the union of instance support;
- overlap count matches sparse memberships;
- crossings do not create graph connectivity;
- true junctions do create graph connectivity;
- endpoints agree with graph topology;
- truncated fibers retain explicit metadata;
- raster targets agree with vector geometry within configured tolerances.

### Milestone 5: Artificial rendering

Command:

```bash
python scripts/generate_synthetic_examples.py \
  --config configs/synthetic_sted/mvp_rendering.yaml \
  --out examples/synthetic_sted_rendered
```

Validation:

```bash
python scripts/validate_synthetic_samples.py \
  examples/synthetic_sted_rendered
```

Pass conditions:

- float and uint8 renders are reproducible;
- PSF parameters are stored in pixel units;
- float-to-uint8 mapping is explicit;
- clipping and saturation are reported;
- calibration status is `procedural_unmatched`;
- no real blank compositing occurs.

### Milestone 6: Visualization

Command:

```bash
python scripts/visualize_synthetic_samples.py \
  examples/synthetic_sted_rendered \
  --out reports/synthetic_mvp_visuals
```

Expected overlays:

- rendered image;
- semantic mask;
- per-instance geometry;
- centerline;
- endpoints;
- true junctions;
- apparent crossings;
- overlap count.

Pass conditions:

- visualization files exist;
- each visualization links to sample metadata;
- target conventions are visually inspectable;
- crossings and true junctions are clearly distinguishable.

### Milestone 7: Full MVP validation

Portable command:

```bash
python -m pytest -q
```

Local validation command:

```bash
python scripts/validate_sted_inventory.py \
  --fiber-dir /ssd/STED_dataset/data \
  --blank-dir /ssd/STED_dataset/bigblank \
  --manifests data_manifests \
&& python scripts/validate_sted_splits.py \
  --inventory-dir data_manifests \
  --splits data_manifests/sted_splits.csv \
&& python scripts/validate_synthetic_samples.py \
  examples/synthetic_sted_rendered \
&& git diff --check
```

Pass conditions:

- portable tests pass without `/ssd`;
- local validations pass when real paths are supplied;
- generated examples validate;
- no whitespace errors;
- no undocumented scope expansion.

---

## 19. Files to create

### Configuration

- `configs/synthetic_sted/source_roots.example.yaml`
- `configs/synthetic_sted/mvp_geometry.yaml`
- `configs/synthetic_sted/mvp_rendering.yaml`

### Source modules

- `src/fibras/sted_inventory.py`
- `src/fibras/sted_filename_parser.py`
- `src/fibras/sted_splits.py`
- `src/fibras/synthetic/geometry.py`
- `src/fibras/synthetic/targets.py`
- `src/fibras/synthetic/rendering.py`
- `src/fibras/synthetic/schema.py`
- `src/fibras/synthetic/storage.py`
- `src/fibras/synthetic/visualization.py`

### Scripts

- `scripts/inventory_sted_data.py`
- `scripts/validate_sted_inventory.py`
- `scripts/create_sted_splits.py`
- `scripts/validate_sted_splits.py`
- `scripts/generate_synthetic_examples.py`
- `scripts/validate_synthetic_samples.py`
- `scripts/visualize_synthetic_samples.py`

### Tests

- `tests/test_sted_filename_parser.py`
- `tests/test_sted_inventory.py`
- `tests/test_sted_duplicates.py`
- `tests/test_sted_splits.py`
- `tests/test_synthetic_geometry.py`
- `tests/test_synthetic_targets.py`
- `tests/test_synthetic_rendering.py`
- `tests/test_synthetic_schema.py`
- `tests/test_synthetic_storage.py`

### Documentation and manifests

- `data_manifests/README.md`
- `reports/README.md`
- `docs/synthetic_sted_pipeline.md`

Generated locally:

- `data_manifests/sted_images.csv`
- `data_manifests/sted_blanks.csv`
- `data_manifests/acquisition_groups.csv`
- `data_manifests/sted_splits.csv`
- `reports/data_inventory.md`
- `examples/synthetic_sted_mvp/`
- `examples/synthetic_sted_rendered/`
- `reports/synthetic_mvp_visuals/`

Do not commit machine-specific source-root configuration containing absolute local paths unless the repository policy explicitly allows it.

---

## 20. Files that may be modified

Only when necessary:

- `environment.yml`
- `.gitignore`
- `AGENTS.md`

`AGENTS.md` may receive a short durable rule stating that:

- STED inventory and split manifests are authoritative;
- downstream code must not independently infer grouping;
- normal tests must not require local `/ssd` paths.

Do not duplicate this full plan inside `AGENTS.md`.

---

## 21. Dependencies

Prefer minimal dependencies.

Expected:

- Python 3.10 or newer, as allowed by the repository;
- NumPy;
- Pillow;
- PyYAML;
- pytest.

Potential optional dependencies:

- SciPy for geometry, connected components, or filtering;
- scikit-image for SSIM, morphology, or rasterization.

Only add a dependency when the implementation uses it materially.

Document optional versus required dependencies.

Do not add machine-specific package exports.

---

## 22. Compute and storage estimate

### Inventory

- approximately 558 million pixels;
- CPU-only;
- minutes-scale;
- CSV and Markdown outputs likely below 10 MB.

### MVP examples

- 8–16 samples at `1024 x 1024`;
- compressed NPZ plus JSON;
- likely 5–30 MB per sample depending on sparse target density and retained float arrays;
- expected total approximately 100–500 MB maximum.

No GPU is required.

Avoid storing redundant arrays when they can be regenerated from:

- vector geometry;
- seeds;
- renderer configuration;
- schema version.

---

## 23. Human decisions still required

Before authoritative annotation protocol or approving the proposed PN allocation:

- meaning of `3R` and `4R`;
- meaning of `AD`, `PID`, `PSP`, and `CBD`;
- meaning of `DIV`;
- meaning of series indices;
- which real groups should be reserved for final held-out evaluation;
- known pixel size;
- known physical calibration;
- whether acquisition metadata exists outside the TIFF files;
- desired initial annotation level:
  - mask-only;
  - mask plus centerline;
  - full instance and graph;
- whether true biological branches are expected;
- whether bundles represent separate instances or compound structures;
- how weak or ambiguous fibers should be labelled;
- whether out-of-focus structures are foreground, background, or ignored.

Blank validity is resolved: the blank images are expert-validated as fiber-free.

The MVP may implement provisional split infrastructure before these decisions, but must not mark provisional real-data splits as human approved.

---

## 24. Implementation authorization and reporting

After reading this plan, proceed with the bounded MVP.

Do not request another broad planning pass unless a blocking contradiction is found.

At completion, report:

1. files created or modified;
2. commands run;
3. test results;
4. local real-data validation results;
5. generated example locations and sizes;
6. unresolved human decisions;
7. deviations from this plan;
8. intentionally deferred features.

Do not:

- implement a neural network;
- train a model;
- composite onto real blanks;
- create a large synthetic dataset;
- add physical detector simulation;
- claim empirical realism;
- silently change the sample schema;
- expand scope without explicit approval.
