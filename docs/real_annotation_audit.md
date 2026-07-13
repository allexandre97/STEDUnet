# Real Annotation Audit

Annotated images: 20

## Metadata Distribution

| Field | Counts |
|:--|:--|
| `validation_status` | `valid`: 20 |
| `disease` | `AD`: 8, `CBD`: 4, `PID`: 4, `PSP`: 4 |
| `tau_isoform` | `3R`: 8, `4R`: 12 |
| `div` | `10`: 3, `3`: 2, `5`: 11, `7`: 4 |
| `split` | `calibration`: 3, `test`: 4, `train`: 2, `validation`: 11 |
| `experimental_group_id` | `PN148_AD_3R_DIV05`: 2, `PN148_AD_3R_DIV07`: 1, `PN148_AD_3R_DIV10`: 1, `PN148_AD_4R_DIV03`: 2, `PN148_AD_4R_DIV05`: 1, `PN148_AD_4R_DIV07`: 1, `PN148_PID_3R_DIV05`: 3, `PN148_PID_3R_DIV07`: 1, `PN148_PSP_4R_DIV05`: 1, `PN148_PSP_4R_DIV07`: 1, `PN148_PSP_4R_DIV10`: 2, `PN151_CBD_4R_DIV05`: 4 |

## Class Statistics

| Class | Pixels | Components |
|:--|--:|--:|
| `background` | 18742862 | 2313 |
| `fibrous_tau` | 837980 | 1041 |
| `clump` | 360549 | 63 |
| `uncertain_ignore` | 1030129 | 2627 |

## Skeleton Annotations

Available for 20 images. 1584 of 1620 snakes are usable; 36 are flagged. Target construction clipped 11448 rasterized snake pixels outside confident fibrous regions.

## Ambiguous Or Missing Metadata

- Biological preparation identity is not available beyond the existing provisional experimental_group_id. PN is retained only as culture_id.

## Manual Resolution

- Confirm biological preparation/replicate identifiers before final train/evaluation assignment.
- Review flagged snake overlap in 5 image(s): PN148_3R_AD_DIV07 (Series 0) [1], PN148_3R_PID_DIV05 (Series 2) [1], PN148_4R_AD_DIV03 (Series 11) [1], PN148_4R_PSP_DIV5 (Series 5) [1], PN151_4R_CBD_DIV05 (Series 3) [1].
- Resolve 0 validation problem(s) before using affected images for training or evaluation.

## Validation Problems

- None.
