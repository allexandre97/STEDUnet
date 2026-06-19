# Real STED Annotation Contract

## Study and training roles

Most real fiber-containing STED images form the scientific study corpus. Only a small selected subset will be manually annotated for real-world fine-tuning and locked QA. Most supervised train, validation, and controlled test data will remain synthetic.

Real images may be used condition-blind for image-formation calibration. Synthetic biological geometry must remain broadly randomized rather than fitted tightly to disease-, isoform-, or DIV-specific morphology.

## Reserved semantic classes

| Value | Class | Annotation rule |
|---:|:--|:--|
| 0 | `background` | No labelled structure. |
| 1 | `individual_filament` | Narrow, individually traceable fiber; eligible for filament centerline supervision. |
| 2 | `bundle` | Elongated, wider structure containing unresolved or tightly associated filaments. An optional bundle axis is not an individual-filament centerline. |
| 3 | `clump` | Compact or amorphous structure without reliably traceable filament morphology. Do not assign an arbitrary filament skeleton. |
| 255 | `uncertain_ignore` | Class, presence, or boundary is uncertain; exclude from applicable supervised losses. |

Schema `synthetic_sted_3d_morphology_0.6.0` emits this multiclass target for exploratory morphology review while retaining a binary compatibility mask equal to the union of classes 1–3. Earlier 3D schemas remain binary and are not reinterpreted.

Synthetic bundles are correlated child filaments around a bundle axis. Their deliberately unresolved child centerlines are latent simulator provenance, not ordinary filament-centerline targets. Synthetic clumps are compact aggregates of short latent fragments and receive no filament centerline or endpoint target. These procedural structures broaden training-domain coverage; they are not claimed to reproduce real bundle or clump distributions.

## Trace termination statuses

Reserve these values for annotation conversion and future trace storage:

- `valid_endpoint`
- `boundary_truncation`
- `terminates_in_bundle`
- `terminates_in_clump`
- `ambiguous_termination`

Centerline losses apply only where an individual filament is traceable. Clumps receive no filament centerline target. Patch- or image-boundary clipping is a boundary truncation, not a biological endpoint.

At a clear filament-to-bundle or filament-to-clump transition, the trace uses `terminates_in_bundle` or `terminates_in_clump`; it is not a biological endpoint. Configured uncertain transition pixels use class 255 and are excluded from the binary foreground compatibility mask.
