# Real-Pilot First-Baseline Error Analysis

This report analyzes existing synthetic-only baseline predictions on the two real expert-annotated pilot images. It does not change training, architecture, synthetic generation, schema, or annotation protocol.

## Semantic Errors

| Pilot | Target fibrous frac | Pred fibrous frac | Pred/target | FP background px | FN fibrous px | FP comps | Large FP | Tiny FP | FN comps | Missed faint | Missed thick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_4R_AD_DIV03 | 0.0159 | 0.1205 | 7.5997 | 72697 | 5065 | 1575 | 86 | 1063 | 286 | 227 | 12 |

Boundary-disagreement fractions estimate how much FP/FN area lies within 2 px of the opposite fibrous mask.

| Pilot | FP within 2 px of target | FN within 2 px of prediction | FP area p50/p95/max | FN area p50/p95/max |
| --- | ---: | ---: | ---: | ---: |
| PN148_4R_AD_DIV03 | 0.0559 | 0.3828 | 3.0000/111.3000/7903.0000 | 2.0000/92.7500/231.0000 |

## Skeleton Threshold Sensitivity

| Pilot | Threshold | Dice | Precision | Recall | Target recovered <=2 px | Pred within <=2 px | Pred px inside target fibrous | Pred px outside target fibrous | Missed target comps | Unmatched pred comps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_4R_AD_DIV03 | 0.5 | 0.0722 | 0.0385 | 0.5668 | 0.8277 | 0.0841 | 5273 | 51502 | 0 | 1168 |
| PN148_4R_AD_DIV03 | 0.75 | 0.1662 | 0.1050 | 0.3987 | 0.7218 | 0.2066 | 3296 | 11368 | 1 | 1289 |
| PN148_4R_AD_DIV03 | 0.85 | 0.2690 | 0.2429 | 0.3013 | 0.6446 | 0.4347 | 2246 | 2541 | 7 | 644 |

## Clump And Ignore Leakage

| Pilot | Pred fib in target fib | Pred fib in clump | Pred fib in ignore | Pred fib in background | Fib frac clump | Fib frac ignore | Pred clump in clump | Pred clump in fibrous | Pred clump in ignore | Pred clump in background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_4R_AD_DIV03 | 9492 | 28440 | 69601 | 72697 | 0.1578 | 0.3862 | 80593 | 2927 | 45596 | 21260 |

| Pilot | Threshold | Skel in fibrous | Skel in clump | Skel in ignore | Skel in background | Frac clump | Frac ignore | Skel in pred fib | Skel in pred clump | Skel outside pred foreground |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_4R_AD_DIV03 | 0.5 | 5273 | 21260 | 42297 | 30242 | 0.2146 | 0.4269 | 93998 | 4204 | 870 |
| PN148_4R_AD_DIV03 | 0.75 | 3296 | 5070 | 14198 | 6298 | 0.1757 | 0.4919 | 28653 | 180 | 29 |
| PN148_4R_AD_DIV03 | 0.85 | 2246 | 920 | 5012 | 1621 | 0.0939 | 0.5115 | 9788 | 11 | 0 |

## Probability Summaries By Expert Region

| Pilot | Region | Fib prob mean/p95 | Clump prob mean/p95 | Skeleton prob mean/p95 |
| --- | --- | ---: | ---: | ---: |
| PN148_4R_AD_DIV03 | target_fibrous | 0.6237/0.9968 | 0.2163/0.9099 | 0.3747/0.9556 |
| PN148_4R_AD_DIV03 | target_clump | 0.2683/0.9935 | 0.7263/1.0000 | 0.1982/0.7430 |
| PN148_4R_AD_DIV03 | target_uncertain_ignore | 0.5163/0.9873 | 0.3535/0.9775 | 0.3398/0.8312 |
| PN148_4R_AD_DIV03 | target_background | 0.0905/0.8051 | 0.0304/0.1576 | 0.0481/0.4119 |

## Intensity And Domain Gap

| Source | Mean | Std | p50 | p95 | p99 | Zero frac | Local var mean | Local var p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_4R_AD_DIV03 | 32.7143 | 40.7152 | 19.0000 | 124.0000 | 204.0000 | 0.0267 | 79.8775 | 317.0664 |
| synthetic validation | not_available | no readable validation render_uint8 arrays |  |  |  |  |  |  |

## Main Visual Failure Modes

- Semantic predictions over-segment fibrous area in both pilots, with predicted/target area ratios above 1.0 and many small false-positive components.
- Boundary disagreement contributes to both false-positive and false-negative fibrous errors, but large missed regions remain visible in the component summaries.
- Skeleton predictions are mostly constrained to target fibrous regions, but exact skeleton Dice remains low; 2 px tolerant recovery is substantially higher than exact-pixel Dice.
- Raising the skeleton threshold to 0.85 improves precision in places but lowers recall and does not remove the need to inspect skeleton failure cases.
- Leakage flags: clump-vs-fibrous confusion, diffuse background false positives, fibrous leakage into expert clump, fibrous leakage into expert uncertain_ignore, skeleton leakage into clump, skeleton leakage into uncertain_ignore

## Conclusion

collect more real annotations before deciding.
