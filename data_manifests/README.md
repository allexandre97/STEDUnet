# STED data manifests

This directory is for generated, machine-readable STED source manifests.

Authoritative generated files:

- `sted_images.csv`
- `sted_blanks.csv`
- `acquisition_groups.csv`
- `sted_splits.csv`
- `real_annotations.csv`
- `real_annotation_crops.csv`
- `real_annotation_audit.json`
- `real_annotation_folds_v1.csv`
- `real_annotation_folds_v1_summary.json`

Source identity is portable: use source root IDs, relative paths, source SHA-256, pixel SHA-256, and stable image IDs. Do not rename original STED files.

`real_annotations.csv` inventories every discovered expert annotation, including invalid rows that require manual resolution. `real_annotation_crops.csv` is a crop plan, not a binary dataset: it includes only valid parent images, and every crop inherits its parent's `split` and `split_group_id`. Private labels, traces, real-image arrays, and generated crop NPZ files remain outside the repository.

`real_annotation_folds_v1.csv` is the versioned grouped cross-validation assignment. Do not generate experiment-specific folds; materialized private crop manifests inherit one committed outer fold's train, validation, and test partitions.
