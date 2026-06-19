# Synthetic STED morphology heterogeneity review

- calibration_status: `exploratory_unpartitioned`
- schema: `synthetic_sted_3d_morphology_0.6.0`
- Morphology generation is condition-blind and broadly randomized.
- Real-image measurements are coarse threshold-sensitive diagnostics, not fitted biological targets.
- Composite foreground scale is sampled deterministically from `1.0–2.0`.
- Shared display range: `0` to `96`.
- Review samples: `24`.

## Median spatial diagnostics

| Group | Foreground occupancy | Tile variance | Empty tiles | Concentration | Orientation coherence | Bundle fraction | Clump fraction |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `morphology_scene` | 0.009501 | 0.001175 | 0.8496 | 0.9088 | 0.8844 | 0.002996 | 0.0008674 |
| `previous_uniform_synthetic` | 0.00915 | 0.00085 | 0.8574 | 0.9265 | 0.8897 | 0 | 0 |
| `real_fiber_proxy` | 0.004877 | 0.0003159 | 0.9121 | 0.9053 | 0.1714 | n/a | n/a |
| `real_blank_proxy` | 0.0002818 | 8.292e-07 | 0.9961 | 0.7758 | 0.2938 | n/a | n/a |

## Morphology-mode medians

| Mode | Foreground occupancy | Tile variance | Empty tiles | Bundle fraction | Clump fraction |
|:--|--:|--:|--:|--:|--:|
| `bundle_dominated` | 0.01296 | 0.002672 | 0.8672 | 0.00979 | 0 |
| `clump_dominated` | 0.006659 | 0.0007803 | 0.8828 | 0.002408 | 0.001985 |
| `clustered_filament_network` | 0.005256 | 0.0001654 | 0.7871 | 0 | 0 |
| `isolated_filaments` | 0.001502 | 3.97e-05 | 0.9355 | 0 | 0 |
| `mixed_morphology` | 0.01653 | 0.002537 | 0.7637 | 0.007742 | 0.002312 |

The heterogeneous generator is evaluated for increased spatial variation only. No table entry is a biological matching claim.

## Review assets

- [Morphology mode contact sheet](morphology_modes_contact_sheet.png)
- [Morphology transition and structure zooms](morphology_zoom_regions.png)
- [Spatial diagnostic summary](spatial_heterogeneity_summary.png)
- Full local per-sample panels are generated under `full_samples/` and remain ignored by Git.
