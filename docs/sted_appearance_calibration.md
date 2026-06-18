# STED Appearance Calibration and Real-Blank Compositing

## Status

This phase is exploratory. It characterizes real fiber-containing STED images and expert-validated blank STED images, then generates a small set of real-blank composite examples.

All outputs remain `exploratory_unpartitioned` or `exploratory_provisional_split`. Do not label any synthetic image `empirically_matched` until biological grouping, an approved calibration subset, quantitative comparisons, and visual comparisons have been reviewed.

## Manual review record for geometry MVP

- Review status: manually reviewed
- Reviewer: project owner
- Result: approved for continued development
- Review date: 2026-06-18
- Scope: geometry, crossings versus junctions, semantic masks, instance memberships, centerlines, endpoint/junction/crossing maps, overlap maps, artificial rendering, and overlays

The approved geometry and structural-target conventions must not be modified unless a concrete defect is discovered.

## Metadata constraint

The meanings of `PN###` and `3R`/`4R` are not yet confirmed. `DIV` likely means days in vitro, but it is not yet known whether different DIV values correspond to independent biological samples.

Therefore:

- all real fiber-image splits remain provisional;
- all split rows remain `human_approved=false`;
- no current split is claimed to be biologically independent;
- PN, round, condition, DIV, and series remain separate metadata fields;
- no model training is part of this phase.

## Commands

```bash
python scripts/characterize_sted_appearance.py \
  --config configs/synthetic_sted/exploratory_calibration.yaml \
  --inventory-dir data_manifests \
  --splits data_manifests/sted_splits.csv \
  --out calibration_artifacts/exploratory

python scripts/generate_blank_composite_examples.py \
  --config configs/synthetic_sted/exploratory_calibration.yaml \
  --inventory-dir data_manifests \
  --splits data_manifests/sted_splits.csv \
  --out examples/sted_blank_composites_exploratory

python scripts/validate_calibration_artifacts.py \
  --artifact-dir calibration_artifacts/exploratory \
  --composite-dir examples/sted_blank_composites_exploratory

python scripts/build_sted_calibration_report.py \
  --config configs/synthetic_sted/exploratory_calibration.yaml \
  --artifact-dir calibration_artifacts/exploratory \
  --artificial-dir examples/synthetic_sted_rendered \
  --composite-dir examples/sted_blank_composites_exploratory \
  --out reports/sted_appearance_calibration.md
```

## Measurement classes

Direct measurements include intensity percentiles, zero and saturation fractions, local mean/variance distributions, row/column variation, low-frequency gradients, high-frequency residual statistics, power spectra, autocorrelation, and image-to-image variability.

Proxy-derived measurements include foreground occupancy, ridge response, orientation, apparent width, local contrast/SNR, connected-component length, endpoint-candidate density, and crossing-candidate density. They are not ground truth and must not be silently converted into generator parameters.

## Real-blank compositing

The compositor uses the observed blank image as the authoritative background. It renders the synthetic fiber signal separately, applies a configurable pixel-space PSF to the signal, adds it to the observed blank, optionally adds a small empirical signal-dependent perturbation to the added signal only, and applies one final explicit float-to-uint8 mapping.

It does not infer photon counts, detector gain, or Poisson calibration, and it does not alter structural targets during compositing.

