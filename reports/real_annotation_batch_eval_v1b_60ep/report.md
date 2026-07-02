# Real Annotation Batch Evaluation

Samples evaluated: 4

This batch evaluation uses the frozen synthetic-only first baseline. It does not train, fine-tune, alter model architecture, alter synthetic generation, or change annotation protocol.

Skeleton metrics are reported at thresholds 0.5, 0.75, and 0.85. For real images, 0.5 may be preferable for skeleton recall, while 0.75 may be more conservative; no final threshold is selected here.

## Per-Sample Summary

| Sample | Category | Fib Dice | Fib P | Fib R | Area ratio | Clump px | Skel Dice 0.5 | Skel Dice 0.75 | Skel Dice 0.85 | Target skel <=2px 0.75 | Pred skel <=2px 0.75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05__Series_2___1 |  | 0.6092 | 0.4816 | 0.8288 | 1.7210 | 0 | 0.3261 | 0.2936 | 0.2165 | 0.7277 | 0.5759 |
| PN148_4R_AD_DIV03__Series_11___1 |  | 0.1516 | 0.0858 | 0.6521 | 7.5997 | 109300 | 0.0722 | 0.1662 | 0.2690 | 0.7218 | 0.2066 |
| PN148_4R_PSP_DIV5__Series_5___1 |  | 0.4186 | 0.4071 | 0.4308 | 1.0581 | 0 | 0.2408 | 0.2448 | 0.2027 | 0.5131 | 0.5464 |
| PN151_4R_CBD_DIV05__Series_3___1 |  | 0.7385 | 0.6777 | 0.8114 | 1.1974 | 0 | 0.3833 | 0.3384 | 0.2991 | 0.7562 | 0.7883 |

## Batch Means

| Metric | Mean |
| --- | ---: |
| fibrous_dice | 0.4795 |
| fibrous_precision | 0.4130 |
| fibrous_recall | 0.6808 |
| predicted_to_target_fibrous_area_ratio | 2.8940 |
| clump_target_pixels | 27325.0000 |
| intensity_mean | 19.3542 |
| intensity_p95 | 66.7500 |
| intensity_zero_fraction | 0.0453 |
| skeleton_dice_0.5 | 0.2556 |
| skeleton_precision_0.5 | 0.2022 |
| skeleton_recall_0.5 | 0.4878 |
| target_skeleton_recovered_within_2px_0.5 | 0.7870 |
| predicted_skeleton_within_2px_of_target_0.5 | 0.4395 |
| skeleton_dice_0.75 | 0.2608 |
| skeleton_precision_0.75 | 0.2523 |
| skeleton_recall_0.75 | 0.3181 |
| target_skeleton_recovered_within_2px_0.75 | 0.6797 |
| predicted_skeleton_within_2px_of_target_0.75 | 0.5293 |
| skeleton_dice_0.85 | 0.2468 |
| skeleton_precision_0.85 | 0.3116 |
| skeleton_recall_0.85 | 0.2159 |
| target_skeleton_recovered_within_2px_0.85 | 0.5815 |
| predicted_skeleton_within_2px_of_target_0.85 | 0.6263 |

## Interpretation

more real annotations needed before deciding
