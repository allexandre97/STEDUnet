# Real STED Annotation Contract

## Study and training roles

Most real fiber-containing STED images form the scientific study corpus. Only a small selected subset will be manually annotated for real-world fine-tuning and locked QA. Most supervised train, validation, and controlled test data will remain synthetic.

Real images may be used condition-blind for image-formation calibration. Synthetic biological geometry must remain broadly randomized rather than fitted tightly to disease-, isoform-, or DIV-specific morphology.

## Expert real annotation classes

The STED expert pilot uses a deliberately small Labkit vocabulary:

| Real label | Class | Annotation rule |
|---:|:--|:--|
| 0 | `background` / unannotated | No labelled structure. Unannotated pixels are background. |
| 1 | `fibers` | Fibrous tau signal. Real annotations do not distinguish individual filaments from bundles. |
| 2 | `uncertain_ignore` | Class, presence, or boundary is uncertain; exclude from applicable supervised losses. |
| 3 | `clump` | Compact or amorphous structure without reliably traceable filament morphology. |

These real labels are remapped to the internal real-compatible class vocabulary:

| Real label | Real class | Internal value | Internal class |
|---:|:--|---:|:--|
| 0 | `background` | 0 | `background` |
| 1 | `fibers` | 1 | `fibrous_tau` |
| 2 | `uncertain_ignore` | 255 | `uncertain_ignore` |
| 3 | `clump` | 3 | `clump` |

JFilament exports one `.txt` file containing all snakes for an image. Real snakes are fibrous skeletons or bundle axes; they are not guaranteed individual-filament centerlines. They may be used as real-compatible fibrous skeleton supervision only when their rasterized support overlaps mostly with the `fibrous_tau` semantic mask. Snakes mostly in `clump`, `uncertain_ignore`, or `background`, or with mixed overlap, are flagged rather than used as confident skeleton supervision.

Crossing points, merge points, branch points, endpoints, and bundle-entry labels are not supervised real labels. Any topology beyond the expert snakes is synthetic-only or post-processing-derived.

## Rich synthetic morphology classes

Synthetic samples keep the richer morphology vocabulary:

| Value | Class | Rule |
|---:|:--|:--|
| 0 | `background` | No labelled structure. |
| 1 | `individual_filament` | Narrow, individually traceable fiber. |
| 2 | `bundle` | Elongated, wider structure containing unresolved or tightly associated filaments. |
| 3 | `clump` | Compact or amorphous structure without reliably traceable filament morphology. Do not assign an arbitrary filament skeleton. |
| 255 | `uncertain_ignore` | Class, presence, or boundary is uncertain; exclude from applicable supervised losses. |

Schema `synthetic_sted_3d_morphology_0.8.0` emits apparent visible multiclass masks for training and separate `*_source_support_mask` arrays for latent geometry/source provenance. Schema 0.7 remains readable under its source-support class-mask semantics, and schema 0.6 remains readable under its original ambiguous generic-membership semantics; use the documented migration before treating memberships as supervised.

Synthetic bundles are correlated child filaments around a bundle axis. Their deliberately unresolved child centerlines are latent simulator provenance, not ordinary filament-centerline targets. Synthetic clumps are compact aggregates of short latent fragments and receive no filament centerline or endpoint target. These procedural structures broaden training-domain coverage; they are not claimed to reproduce real bundle or clump distributions.

Schema 0.8 loss code may consume only arrays listed under metadata `target_roles.supervised` or `target_roles.real_compatible_supervised`, depending on the training view. Arrays under `target_roles.latent_synthetic_provenance` or `diagnostic_only` must not be used as labels. Bundle-child and clump-fragment graph edges are latent; their graph endpoints are not supervised biological endpoints.

For real-compatible training views, collapse the richer synthetic classes as follows:

| Synthetic class | Real-compatible class |
|:--|:--|
| `background` | `background` |
| `individual_filament` | `fibrous_tau` |
| `bundle` | `fibrous_tau` |
| `clump` | `clump` |
| `uncertain_ignore` | `uncertain_ignore` |

Synthetic morphology samples expose this collapsed view as `real_compatible_semantic_mask`, class-specific real-compatible masks, and `real_compatible_skeleton_mask`, derived from apparent supervised masks rather than source-support masks. The skeleton view is `filament_centerline_mask OR bundle_axis_mask` clipped to `fibrous_tau`; endpoint, crossing, junction, latent bundle-child, and clump-fragment arrays remain synthetic-only or diagnostic by default.

## Trace termination statuses

Reserve these values for annotation conversion and future trace storage:

- `valid_endpoint`
- `boundary_truncation`
- `terminates_in_bundle`
- `terminates_in_clump`
- `ambiguous_termination`

Centerline losses apply only where an individual filament is traceable. Clumps receive no filament centerline target. Patch- or image-boundary clipping is a boundary truncation, not a biological endpoint.

At a clear filament-to-bundle or filament-to-clump transition, the trace uses `terminates_in_bundle` or `terminates_in_clump`; it is not a biological endpoint. Configured uncertain transition pixels use class 255 and are excluded from the binary foreground compatibility mask.
