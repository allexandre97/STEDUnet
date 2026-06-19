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

The current synthetic generator still emits a binary semantic mask. These class values are reserved for future real annotations and compatible loaders; the simulator does not claim to generate realistic bundles or clumps.

## Trace termination statuses

Reserve these values for annotation conversion and future trace storage:

- `valid_endpoint`
- `boundary_truncation`
- `terminates_in_bundle`
- `terminates_in_clump`
- `ambiguous_termination`

Centerline losses apply only where an individual filament is traceable. Clumps receive no filament centerline target. Patch- or image-boundary clipping is a boundary truncation, not a biological endpoint.
