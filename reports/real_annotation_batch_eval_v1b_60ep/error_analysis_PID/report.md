# Real-Pilot First-Baseline Error Analysis

This report analyzes existing synthetic-only baseline predictions on the two real expert-annotated pilot images. It does not change training, architecture, synthetic generation, schema, or annotation protocol.

## Semantic Errors

| Pilot | Target fibrous frac | Pred fibrous frac | Pred/target | FP background px | FN fibrous px | FP comps | Large FP | Tiny FP | FN comps | Missed faint | Missed thick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 0.0332 | 0.0571 | 1.7210 | 30999 | 5949 | 535 | 65 | 343 | 324 | 287 | 10 |

Boundary-disagreement fractions estimate how much FP/FN area lies within 2 px of the opposite fibrous mask.

| Pilot | FP within 2 px of target | FN within 2 px of prediction | FP area p50/p95/max | FN area p50/p95/max |
| --- | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 0.3729 | 0.5673 | 2.0000/278.9000/2389.0000 | 2.0000/79.0000/777.0000 |

## Skeleton Threshold Sensitivity

| Pilot | Threshold | Dice | Precision | Recall | Target recovered <=2 px | Pred within <=2 px | Pred px inside target fibrous | Pred px outside target fibrous | Missed target comps | Unmatched pred comps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 0.5 | 0.3261 | 0.2278 | 0.5737 | 0.8587 | 0.5168 | 18971 | 11395 | 0 | 147 |
| PN148_3R_PID_DIV05 | 0.75 | 0.2936 | 0.2613 | 0.3351 | 0.7277 | 0.5759 | 10951 | 4509 | 0 | 179 |
| PN148_3R_PID_DIV05 | 0.85 | 0.2165 | 0.2761 | 0.1781 | 0.5794 | 0.6061 | 5747 | 2030 | 0 | 265 |

## Clump And Ignore Leakage

| Pilot | Pred fib in target fib | Pred fib in clump | Pred fib in ignore | Pred fib in background | Fib frac clump | Fib frac ignore | Pred clump in clump | Pred clump in fibrous | Pred clump in ignore | Pred clump in background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 28795 | 0 | 368 | 30999 | 0.0000 | 0.0061 | 0 | 3055 | 144 | 2086 |

| Pilot | Threshold | Skel in fibrous | Skel in clump | Skel in ignore | Skel in background | Frac clump | Frac ignore | Skel in pred fib | Skel in pred clump | Skel outside pred foreground |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 0.5 | 18971 | 0 | 248 | 11395 | 0.0000 | 0.0081 | 29344 | 624 | 646 |
| PN148_3R_PID_DIV05 | 0.75 | 10951 | 0 | 134 | 4509 | 0.0000 | 0.0086 | 15551 | 34 | 9 |
| PN148_3R_PID_DIV05 | 0.85 | 5747 | 0 | 62 | 2030 | 0.0000 | 0.0079 | 7838 | 1 | 0 |

## Probability Summaries By Expert Region

| Pilot | Region | Fib prob mean/p95 | Clump prob mean/p95 | Skeleton prob mean/p95 |
| --- | --- | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | target_fibrous | 0.7971/0.9976 | 0.1032/0.6761 | 0.5002/0.9248 |
| PN148_3R_PID_DIV05 | target_clump | not_applicable/not_applicable | not_applicable/not_applicable | not_applicable/not_applicable |
| PN148_3R_PID_DIV05 | target_uncertain_ignore | 0.6073/0.9728 | 0.2987/0.8715 | 0.4154/0.9180 |
| PN148_3R_PID_DIV05 | target_background | 0.0313/0.0850 | 0.0030/0.0008 | 0.0148/0.0293 |

## Intensity And Domain Gap

| Source | Mean | Std | p50 | p95 | p99 | Zero frac | Local var mean | Local var p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 12.8460 | 12.3172 | 9.0000 | 36.0000 | 61.0000 | 0.0481 | 28.3781 | 85.9633 |
| synthetic validation | not_available | no readable validation render_uint8 arrays |  |  |  |  |  |  |

## Main Visual Failure Modes

- Semantic predictions over-segment fibrous area in both pilots, with predicted/target area ratios above 1.0 and many small false-positive components.
- Boundary disagreement contributes to both false-positive and false-negative fibrous errors, but large missed regions remain visible in the component summaries.
- Skeleton predictions are mostly constrained to target fibrous regions, but exact skeleton Dice remains low; 2 px tolerant recovery is substantially higher than exact-pixel Dice.
- Raising the skeleton threshold to 0.85 improves precision in places but lowers recall and does not remove the need to inspect skeleton failure cases.
- Leakage flags: clump-vs-fibrous confusion, diffuse background false positives, fibrous leakage into expert uncertain_ignore, skeleton leakage into uncertain_ignore

## Conclusion

collect more real annotations before deciding.
