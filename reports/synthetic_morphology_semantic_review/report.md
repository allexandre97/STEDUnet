# Synthetic STED morphology heterogeneity review

- calibration_status: `exploratory_unpartitioned`
- schema: `synthetic_sted_3d_morphology_0.8.0`
- Morphology generation is condition-blind and broadly randomized.
- Real-image measurements are coarse threshold-sensitive diagnostics, not fitted biological targets.
- Composite foreground scale is sampled deterministically from `1.0–2.0`.
- Shared display range: `0` to `96`.
- Review samples: `24`.

## A. Synthetic ground-truth morphology diagnostics

| Group | Foreground occupancy | Tile variance | Endpoints | Supervised memberships | Latent memberships | Bundle width | Clump hole-fill ratio |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `morphology_scene` | 0.009413 | 0.00126 | 15.5 | 1.004e+04 | 7557 | 10.1 | 0.9418 |
| `previous_uniform_synthetic` | 0.00915 | 0.00085 | n/a | n/a | n/a | n/a | n/a |

## Morphology-mode medians

| Mode | Foreground occupancy | Tile variance | Empty tiles | Bundle fraction | Clump fraction |
|:--|--:|--:|--:|--:|--:|
| `bundle_dominated` | 0.01147 | 0.001565 | 0.8691 | 0.007988 | 0 |
| `clump_dominated` | 0.006959 | 0.001026 | 0.8828 | 0.002317 | 0.002178 |
| `clustered_filament_network` | 0.005272 | 0.0001904 | 0.8027 | 0 | 0 |
| `isolated_filaments` | 0.001526 | 4.137e-05 | 0.9336 | 0 | 0 |
| `mixed_morphology` | 0.01463 | 0.002103 | 0.7852 | 0.005743 | 0.002213 |

## Class-mask and rendered-signal alignment

Schema 0.8 reports apparent supervised masks separately from latent source-support masks. Median in-memory validation over the 24 small review samples produced zero target-validation errors.

| Class | Visible signal outside apparent mask | Visible signal inside apparent mask | Apparent/source area ratio | Apparent mask visible fraction |
|:--|--:|--:|--:|--:|
| `individual_filament` | 0.032 | 0.9622 | 3.84 | 1 |
| `bundle` | 0.0005 | 0.9568 | 1.0045 | 1 |
| `clump` | 0 | 0.9879 | 1.3658 | 1 |
| `uncertain_transition` | not_applicable | not_applicable | not_applicable | not_applicable |

The filament spill outside the supervised apparent mask is much lower than the previous source-support mask diagnostic. The uncertain-transition row is not evaluated as scene-wide signal spill.

## B. Matched image-proxy diagnostics

All rows below use identical thresholding, ridge/orientation processing, tile size, and connected-component settings. Metrics are `threshold_sensitive`, `condition_blind`, and `not_biological_ground_truth`.

| Group | Occupancy proxy | Tile variance | Empty tiles | Concentration | Orientation coherence | Signal p95 |
|:--|--:|--:|--:|--:|--:|--:|
| `new_morphology_composite_proxy` | 0.01005 | 0.0007892 | 0.8457 | 0.9046 | 0.37 | 106 |
| `previous_composite_proxy` | 0.00693 | 0.000223 | 0.7871 | 0.7283 | 0.3174 | 45 |
| `real_fiber_proxy` | 0.004877 | 0.0003159 | 0.9121 | 0.9053 | 0.1714 | 62.95 |
| `real_blank_proxy` | 0.0002818 | 8.292e-07 | 0.9961 | 0.7758 | 0.2938 | 39.1 |

The heterogeneous generator is evaluated for increased spatial variation only. No table entry is a biological matching claim.

## Review assets

- [Morphology mode contact sheet](morphology_modes_contact_sheet.png)
- [Morphology transition and structure zooms](morphology_zoom_regions.png)
- [Supervised versus latent geometry](supervised_vs_latent_geometry.png)
- [Class masks versus signal](class_mask_signal_alignment.png)
- [Endpoint and resolved-crossing semantics](endpoint_and_resolved_crossing_semantics.png)
- [Matched image-proxy summary](matched_image_proxy_summary.png)
- Full local per-sample panels are generated under `full_samples/` and remain ignored by Git.
