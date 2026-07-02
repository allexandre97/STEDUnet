# Synthetic Morphology Schema Migration

Schema `synthetic_sted_3d_morphology_0.7.0` corrects supervision semantics without reinterpreting schema 0.6 artifacts.

## Memberships

Schema 0.6 `membership_*` and `overlap_count` include every generated curve. In schema 0.7 they are replaced by:

- `supervised_membership_*` and `supervised_overlap_count` for loss-eligible class instances;
- `supervised_membership_class_id` for class attribution;
- `latent_geometry_membership_*` and `latent_geometry_overlap_count` for synthetic provenance.

Class-specific memberships remain authoritative. Loaders must consult metadata `target_roles`; latent and diagnostic arrays are never supervised defaults.

## Graph and boundaries

Schema 0.7 adds `node_supervised`, `node_termination_status`, `node_boundary_code`, `edge_supervised`, `edge_structure_type`, and per-fiber boundary bit fields. Boundary bits are listed in metadata `enum_mappings.boundary_bit`.

Curves terminate at their first volume intersection. Bundle/clump transition statuses remain distinct from valid biological endpoints, and latent bundle-child or clump-fragment nodes are not supervised endpoints.

## Compatibility

Validators continue to accept schema 0.6 using its original generic-membership contract. Existing artifacts are not rewritten automatically. Regenerate from the stored configuration and seed to obtain schema 0.7.

## Schema 0.8 apparent masks

Schema `synthetic_sted_3d_morphology_0.8.0` keeps schema 0.7 readable and adds an explicit split between source geometry and supervised apparent masks.

- `individual_filament_source_support_mask`, `bundle_source_support_mask`, and `clump_source_support_mask` are latent simulator/source-support provenance.
- `individual_filament_mask`, `bundle_mask`, `clump_mask`, `semantic_class_mask`, and `semantic_mask` are supervised apparent visible STED footprints.
- Apparent class pixels are derived from class-attributed optical signals using `class_signal >= visible_signal_threshold * apparent_mask_visibility_high_factor`.
- Pixels between `apparent_mask_visibility_low_factor` and the high threshold become `uncertain_ignore`.
- Source-support masks, latent memberships, hidden bundle-child geometry, and clump fragments remain non-supervised provenance.

Use the `real_compatible_*` arrays for first mixed synthetic/real training targets.
