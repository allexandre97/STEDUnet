# Exploratory STED appearance calibration report

## Status

- calibration_status: `exploratory_unpartitioned`
- Result is exploratory, not approved calibration.
- Current split assignments are provisional and must not be interpreted as biologically independent.
- Synthetic images are not labelled `empirically_matched`.

## Artifact metadata

- Source commit: `9a5aa79b3f680814ad279ec17a3d4543d1624d9f`
- Working tree dirty: `True`
- Generation config SHA-256: `7002bebd6ad5525101cffc5c88edfce0d3d106a087a73b8d0473fddf6cae6a47`
- Inventory manifest SHA-256: `afd9a3a5dddd80b55803d5d19bc4bd388d733a9cfa121e5f51b8c6094133088a`
- Split manifest SHA-256: `9489719f104bbf190c59919440db7327f9729b7f074efad3d8360355c015c1cd`
- Blank-pool manifest SHA-256: `0cbe198c5c1d7da52bbd70963d4db13a48e40240778fc946085c45d940e3953a`
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

- Full local analysis assets are reproducible but intentionally ignored by Git; the committed curated plots are linked below.

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

## Curated review package

- [Realism samples](review_package/realism_samples_contact_sheet.png)
- [Real-blank composites](review_package/real_blank_composites_contact_sheet.png)
- [Structural QA](review_package/structural_qa_contact_sheet.png)
- [Optical QA](review_package/optical_qa_contact_sheet.png)
- [Real-versus-synthetic intensity summary](review_package/real_vs_synthetic_intensity_summary.png)
- [Normalized spectral summary](review_package/normalized_spectral_summary.png)
- [Matched blank-delta summary](review_package/matched_blank_delta_summary.png)
- [Review-package metadata](review_package/README.md)
