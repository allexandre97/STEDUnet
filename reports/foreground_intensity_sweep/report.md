# Exploratory foreground-intensity sweep

- calibration_status: `exploratory_unpartitioned`
- No setting is promoted to `empirically_matched`.
- Main production configuration was not modified.
- Geometry, parent samples, source blanks, PSF, seeds, and output mapping were held fixed.
- Line-density multiplier was held at `1.0`; compositor scale was varied because the two factors are mathematically redundant with perturbation disabled.
- Comparison contact-sheet display range: shared real-STED `0` to `96`.
- Current production default effective scale: `1.0`.

## Real-fiber reference medians

- `p50`: `4`
- `p95`: `17.5`
- `p99`: `27`
- `std`: `6.11361`
- `local_variance_p50`: `11.235`
- `normalized_radial_power_low_band_fraction`: `0.92777`
- `normalized_radial_power_high_band_fraction`: `0.0102763`
- `foreground_occupancy_proxy`: `0.00487661`
- `ridge_response_p50`: `2.5`
- `ridge_response_p95`: `6.57647`

## Setting comparison

| Effective scale | p95 | p99 | Std | Local variance | Occupancy proxy | Ridge p95 | Intensity rank | Variance rank | Structural rank | Max clipped | Max saturated | Violation |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.5 | 9 | 13.5 | 3.21171 | 5.6255 | 0.00558951 | 4.60778 | 5 | 6 | 1 | 0 | 0 | false |
| 1 | 9 | 14 | 3.71982 | 5.68973 | 0.00692985 | 4.72487 | 4 | 5 | 2 | 0 | 0 | false |
| 2 | 10 | 17 | 4.691 | 5.77601 | 0.00849392 | 4.80489 | 3 | 4 | 3 | 3.8147e-05 | 3.8147e-05 | false |
| 4 | 10 | 21.5 | 7.52805 | 5.83016 | 0.010118 | 4.87096 | 1 | 3 | 6 | 0.000228882 | 0.000230789 | true |
| 8 | 10 | 31 | 12.423 | 5.91451 | 0.0101185 | 5.02439 | 2 | 2 | 5 | 0.00120831 | 0.00121498 | true |
| 12 | 10 | 37.5 | 15.4365 | 5.95312 | 0.0100742 | 5.04951 | 6 | 1 | 4 | 0.00331879 | 0.00332546 | true |

## Promising candidates for later review

- Effective scale `2`: intensity rank `3`, variance rank `4`, structural/ridge rank `3`. This is exploratory, not a production selection.

## Rejected settings

- Effective scale `4`: maximum clipped fraction `0.000228882` and maximum saturated fraction `0.000230789` exceeded the configured sweep limit.
- Effective scale `8`: maximum clipped fraction `0.00120831` and maximum saturated fraction `0.00121498` exceeded the configured sweep limit.
- Effective scale `12`: maximum clipped fraction `0.00331879` and maximum saturated fraction `0.00332546` exceeded the configured sweep limit.

## Ranking definitions

- Intensity mismatch: sum of absolute median differences for p50, p95, p99, and standard deviation.
- Local-variance mismatch: absolute median local-variance difference.
- Structural/ridge mismatch: sum of relative median differences for occupancy and ridge p50/p95 proxies.
- Spectral mismatch: sum of absolute differences in normalized low- and high-band fractions.
- Rankings are separate diagnostics; they are not combined into a realism score.

## Figures

- [Foreground-scale contact sheet](foreground_scale_contact_sheet.png)
- [Sweep metric summary](sweep_metric_summary.png)
- Machine-readable summaries: `calibration_artifacts/foreground_intensity_sweep/foreground_intensity_sweep_summary.csv` and `foreground_intensity_sweep_per_sample.csv`.
