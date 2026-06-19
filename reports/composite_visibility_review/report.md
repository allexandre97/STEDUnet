# Composite visibility review

- Display changes do not modify stored arrays.
- Raw composite panels use fixed `0–255` limits.
- Shared real-STED range: `0` to `96`, derived from global real-fiber `p0.5` and `p99.5` over all pixels.
- Independent structural-inspection stretch: per-image `p0.5` to `p99.5`.
- Foreground-only panels use a separately labelled robust stretch.

## Current default median diagnostics

- `blank_p50`: `2`
- `blank_p95`: `8.5`
- `blank_p99`: `13.5`
- `composite_p50`: `2`
- `composite_p95`: `9`
- `composite_p99`: `14`
- `synthetic_foreground_max`: `107.697`
- `synthetic_foreground_integrated_signal`: `134474`
- `composite_minus_blank_p95`: `0`
- `composite_minus_blank_p99`: `0`
- `clipping_fraction`: `0`
- `saturation_fraction`: `0`
- `foreground_occupancy`: `0.00914955`

## Samples

- [Per-sample index](index.md)
- [Diagnostics CSV](composite_visibility_diagnostics.csv)
- [Visualization metadata](visualization_metadata.json)
