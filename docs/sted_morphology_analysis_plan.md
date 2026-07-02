# STED Morphology Analysis Plan

This project is a STED tau segmentation-to-measurement pipeline, not a pathology classifier. The goal is to produce auditable morphology measurements from STED tau images.

## Project Objective

The model should support:

- segmentation of tau-positive fibrous signal and clumps;
- extraction of skeletons or axes where traceable;
- conversion of predictions into object-level and graph-level morphology features;
- downstream replicate-aware comparison across tauopathy seed class, DIV, culture or batch, field of view, and cell.

Do not frame the primary model as a disease classifier.

## Confirmed Real Annotation Protocol

Expert-confirmed Labkit labels:

| Label | Class |
|---:|:--|
| 0 | `background` / unannotated |
| 1 | `fibers` |
| 2 | `uncertain_ignore` |
| 3 | `clump` |

Internal remap:

| Real label | Internal value | Internal class |
|---:|---:|:--|
| 0 `background` | 0 | `background` |
| 1 `fibers` | 1 | `fibrous_tau` |
| 2 `uncertain_ignore` | 255 | `uncertain_ignore` |
| 3 `clump` | 3 | `clump` |

One JFilament `.txt` file may contain all snakes. Real snakes are fibrous skeletons or bundle axes; they are not guaranteed individual-filament centerlines.

Real annotations do not provide reliable labels for endpoints, crossings, branch points, merge points, bundle-entry points, or individual filament versus bundle class separation. These may be synthetic-only, post-processing-derived, or future optional targets, but they are not required real-supervised targets.

## First Model Scope

First real-compatible target set:

| Head | Targets |
|:--|:--|
| Semantic | `fibrous_tau`, `clump`, ignore mask from `uncertain_ignore` |
| Skeleton / axis | fibrous skeleton from JFilament snakes |
| Optional auxiliary | local orientation; apparent width or distance map |

Endpoint, junction, and crossing heads should not be trained as real-supervised heads unless a future annotation protocol supports them.

Synthetic data may still contain richer labels, but training views must collapse to the real-compatible vocabulary when mixing with expert annotations:

| Synthetic class | Real-compatible class |
|:--|:--|
| `individual_filament` + `bundle` | `fibrous_tau` |
| `clump` | `clump` |
| `uncertain_ignore` | `uncertain_ignore` |
| `background` | `background` |

Use the synthetic `real_compatible_*` arrays for the first mixed synthetic/real training pass. In schema 0.8 these derive from apparent visible supervised masks, while `*_source_support_mask` arrays remain latent simulator provenance. Keep endpoint, crossing, junction, latent bundle-child, and clump-fragment arrays out of real-supervised target sets unless a future real annotation protocol supports them.

## Morphology Feature Table

| Measurement axis | Features |
|:--|:--|
| Aggregate / fibrous burden | area fraction; object count; integrated intensity |
| Clumping | clump area; clump count; compactness; largest-object fraction |
| Filamentization | skeleton length density; edge or segment length distribution; fragmentation |
| Bundling / thickness | apparent width distribution; thick-fiber fraction; width heterogeneity |
| Topology / orientation proxies | connected components; contact or crossing proxies; endpoint-like skeleton termini from post-processing; orientation anisotropy |

Treat 2D STED apparent crossings conservatively as image-plane contacts or proxies, not definitive 3D physical branches.

## Statistical Principle

Seed class, disease, tau isoform, DIV, culture, and field metadata are downstream predictors for analysis, not inputs to the segmentation model.

Final statistics should be replicate-aware and avoid treating fields or cells as independent biological replicates when they are nested within culture or seed preparation.

## Implementation Order

1. Import and validate real pilot annotations.
2. Finish synthetic target alignment to the real-compatible vocabulary.
3. Generate the first synthetic training dataset.
4. Train a semantic + skeleton baseline.
5. Validate or fine-tune on the small real pilot set.
6. Expand annotations only after inspecting real failure modes.
7. Freeze morphology feature definitions before disease/DIV analysis.
