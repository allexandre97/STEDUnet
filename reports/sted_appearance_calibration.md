# Exploratory STED appearance calibration report

## Status

- calibration_status: `exploratory_unpartitioned`
- Result is exploratory, not approved calibration.
- Current split assignments are provisional and must not be interpreted as biologically independent.
- Synthetic images are not labelled `empirically_matched`.

## Artifact metadata

- Code version: `812bacb+dirty`
- Date generated: `2026-06-18`
- Fiber images characterized: 48
- Blank images characterized: 48

## Quantitative comparison summary

| Quantity | Real fiber median | Blank median | Artificial synthetic median | Real-blank composite median |
|:--|--:|--:|--:|--:|
| `p50` | 4 | 4 | 4 | 6 |
| `p99` | 27 | 18 | 91 | 101.5 |
| `std` | 6.114 | 4.278 | 13.02 | 15.62 |
| `zero_fraction` | 0.09958 | 0.1045 | 0.001701 | 0.06494 |
| `saturation_fraction` | 0 | 0 | 1.478e-05 | 3.195e-05 |
| `local_variance_p50` | 11.24 | 10.02 | 1.531 | 12.59 |
| `row_variation` | 0.3487 | 0.3317 | 0.2074 | 0.2443 |
| `column_variation` | 0.3289 | 0.3217 | 0.1793 | 0.2043 |

## Proxy-estimate warning

Foreground occupancy, ridge response, orientation, apparent width, component length, SNR, endpoint density, and crossing density are proxy estimates only. They are threshold-sensitive and are not ground truth.

## Representative plots

- Intensity histogram comparison: `intensity_histogram_comparison.png`
- Power/autocorrelation proxy comparison: `power_proxy_comparison.png`
- Proxy distribution comparison: `proxy_comparison.png`
- Deterministic real-fiber representatives: `representative_real_fiber_images.png`; source IDs: img_00273a310a71b5db79f4, img_b39e45524852b6ff530d, img_ffe8f9679077448a6547
- Deterministic blank representatives: `representative_blank_images.png`; source IDs: img_014d9d77d1d61fed189b, img_6fcc906a6259240a019e, img_351f2b19d31066686b88
- Composite transverse profile proxy: `transverse_profile_proxy.png`

## Example sets

- Artificial-background synthetic examples: `examples/synthetic_sted_rendered`
- Real-blank composite examples: `examples/sted_blank_composites_exploratory`

Representative overlays for composites are generated separately by the visualization script and remain labelled exploratory.

## Mismatches and unresolved uncertainties

- `p99`: real-blank composite median minus real-fiber median = `74.5`.
- `p99`: artificial synthetic median minus real-fiber median = `64`.
- `std`: real-blank composite median minus real-fiber median = `9.506`.
- `std`: artificial synthetic median minus real-fiber median = `6.902`.
- `local_variance_p50`: real-blank composite median minus real-fiber median = `1.359`.
- `local_variance_p50`: artificial synthetic median minus real-fiber median = `-9.704`.
- `radial_power_tail_median`: real-blank composite median minus real-fiber median = `9.959e+04`.
- `radial_power_tail_median`: artificial synthetic median minus real-fiber median = `7.663e+04`.
- `zero_fraction`: real-blank composite median minus real-fiber median = `-0.03465`.
- `zero_fraction`: artificial synthetic median minus real-fiber median = `-0.09788`.
- Label-dependent structure quantities remain uncertain until formal annotation exists.
- Current calibration is exploratory because biological grouping and approved calibration subsets are unresolved.
