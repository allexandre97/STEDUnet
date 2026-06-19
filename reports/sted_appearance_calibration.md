# Exploratory STED appearance calibration report

## Status

- calibration_status: `exploratory_unpartitioned`
- Result is exploratory, not approved calibration.
- Current split assignments are provisional and must not be interpreted as biologically independent.
- Synthetic images are not labelled `empirically_matched`.

## Artifact metadata

- Code version: `1aec4ed+dirty`
- Date generated: `2026-06-19`
- Fiber images characterized: 48
- Blank images characterized: 48

## Quantitative comparison summary

| Quantity | Real fiber median | Blank median | Artificial synthetic median | Real-blank composite median |
|:--|--:|--:|--:|--:|
| `p50` | 4 | 4 | 4 | 2 |
| `p99` | 27 | 18 | 6.5 | 14 |
| `std` | 6.114 | 4.278 | 1.723 | 3.72 |
| `zero_fraction` | 0.09958 | 0.1045 | 0 | 0.2335 |
| `saturation_fraction` | 0 | 0 | 0 | 0 |
| `local_variance_p50` | 11.24 | 10.02 | 0 | 5.69 |
| `row_variation` | 0.3487 | 0.3317 | 0.1394 | 0.2373 |
| `column_variation` | 0.3289 | 0.3217 | 0.1282 | 0.1783 |
| `normalized_radial_power_tail_median` | 0.0004305 | 0.0009131 | 0.0001298 | 0.0006181 |

## Proxy-estimate warning

Foreground occupancy, ridge response, orientation, apparent width, component length, SNR, endpoint density, and crossing density are proxy estimates only. They are threshold-sensitive and are not ground truth.

## Representative plots

- Sorted per-image median intensity summary: `intensity_percentile_summary.png`
- DC-removed normalized spectral-shape and autocorrelation comparison: `normalized_spectral_shape_comparison.png`
- Proxy distribution comparison: `proxy_comparison.png`
- Deterministic real-fiber representatives: `representative_real_fiber_images.png`; source IDs: img_00273a310a71b5db79f4, img_b39e45524852b6ff530d, img_ffe8f9679077448a6547
- Deterministic blank representatives: `representative_blank_images.png`; source IDs: img_014d9d77d1d61fed189b, img_6fcc906a6259240a019e, img_351f2b19d31066686b88
- Composite transverse profile proxy: `transverse_profile_proxy.png`

## Example sets

- Artificial-background synthetic examples: `examples/synthetic_sted_3d_realism`
- Real-blank composite examples: `examples/sted_blank_composites_3d_realism`
- Pure blank QA examples: `examples/pure_blank_qa`

## Width and PSF checks

- Apparent in-focus FWHM: `5.212856292724609` px; target `5.0` px.
- PSF mode: `core_plus_halo`; normalization `unit_integral`.
- Core integrated signal: `48176.38671875`; halo integrated signal: `5318.74365234375`.
- PSF remains an empirical effective model, not a physically calibrated STED PSF.

Representative overlays for composites are generated separately by the visualization script and remain labelled exploratory.

## Mismatches and unresolved uncertainties

- `p99`: real-blank composite median minus real-fiber median = `-13`.
- `p99`: artificial synthetic median minus real-fiber median = `-20.5`.
- `std`: real-blank composite median minus real-fiber median = `-2.394`.
- `std`: artificial synthetic median minus real-fiber median = `-4.39`.
- `local_variance_p50`: real-blank composite median minus real-fiber median = `-5.545`.
- `local_variance_p50`: artificial synthetic median minus real-fiber median = `-11.24`.
- `normalized_radial_power_tail_median`: real-blank composite median minus real-fiber median = `0.0001876`.
- `normalized_radial_power_tail_median`: artificial synthetic median minus real-fiber median = `-0.0003007`.
- `zero_fraction`: real-blank composite median minus real-fiber median = `0.1339`.
- `zero_fraction`: artificial synthetic median minus real-fiber median = `-0.09958`.
- Label-dependent structure quantities remain uncertain until formal annotation exists.
- Current calibration is exploratory because biological grouping and approved calibration subsets are unresolved.

## Matched source-blank deltas

| Quantity | Median matched delta |
|:--|--:|
| `delta_p50` | 0 |
| `delta_p95` | 0 |
| `delta_p99` | 0 |
| `delta_mean` | 0.1248 |
| `delta_variance` | 3.259 |
| `delta_zero_fraction` | -0.003266 |
| `delta_local_variance_p50` | 0.3613 |
| `foreground_added_integrated_signal` | 1.345e+05 |

Matched deltas compare each normalized composite directly with its own source blank; positive values indicate the composite exceeded the blank.
