# Real-Pilot First-Baseline Error Analysis

This report analyzes existing synthetic-only baseline predictions on the two real expert-annotated pilot images. It does not change training, architecture, synthetic generation, schema, or annotation protocol.

## Semantic Errors

| Pilot | Target fibrous frac | Pred fibrous frac | Pred/target | FP background px | FN fibrous px | FP comps | Large FP | Tiny FP | FN comps | Missed faint | Missed thick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_4R_PSP_DIV5 | 0.1200 | 0.1270 | 1.0581 | 74656 | 67739 | 1670 | 151 | 1093 | 748 | 666 | 82 |

Boundary-disagreement fractions estimate how much FP/FN area lies within 2 px of the opposite fibrous mask.

| Pilot | FP within 2 px of target | FN within 2 px of prediction | FP area p50/p95/max | FN area p50/p95/max |
| --- | ---: | ---: | ---: | ---: |
| PN148_4R_PSP_DIV5 | 0.2861 | 0.1774 | 2.0000/185.1000/3009.0000 | 2.0000/229.6500/22770.0000 |

## Skeleton Threshold Sensitivity

| Pilot | Threshold | Dice | Precision | Recall | Target recovered <=2 px | Pred within <=2 px | Pred px inside target fibrous | Pred px outside target fibrous | Missed target comps | Unmatched pred comps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_4R_PSP_DIV5 | 0.5 | 0.2408 | 0.1694 | 0.4163 | 0.6393 | 0.4026 | 32938 | 26432 | 14 | 655 |
| PN148_4R_PSP_DIV5 | 0.75 | 0.2448 | 0.2461 | 0.2436 | 0.5131 | 0.5464 | 16920 | 6997 | 19 | 759 |
| PN148_4R_PSP_DIV5 | 0.85 | 0.2027 | 0.3150 | 0.1494 | 0.3963 | 0.6536 | 9012 | 2444 | 25 | 520 |

## Clump And Ignore Leakage

| Pilot | Pred fib in target fib | Pred fib in clump | Pred fib in ignore | Pred fib in background | Fib frac clump | Fib frac ignore | Pred clump in clump | Pred clump in fibrous | Pred clump in ignore | Pred clump in background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_4R_PSP_DIV5 | 51269 | 0 | 18070 | 74656 | 0.0000 | 0.1255 | 0 | 61618 | 36016 | 42785 |

| Pilot | Threshold | Skel in fibrous | Skel in clump | Skel in ignore | Skel in background | Frac clump | Frac ignore | Skel in pred fib | Skel in pred clump | Skel outside pred foreground |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_4R_PSP_DIV5 | 0.5 | 32938 | 0 | 11900 | 26432 | 0.0000 | 0.1670 | 66223 | 4221 | 826 |
| PN148_4R_PSP_DIV5 | 0.75 | 16920 | 0 | 3946 | 6997 | 0.0000 | 0.1416 | 27628 | 197 | 38 |
| PN148_4R_PSP_DIV5 | 0.85 | 9012 | 0 | 1125 | 2444 | 0.0000 | 0.0894 | 12571 | 9 | 1 |

## Probability Summaries By Expert Region

| Pilot | Region | Fib prob mean/p95 | Clump prob mean/p95 | Skeleton prob mean/p95 |
| --- | --- | ---: | ---: | ---: |
| PN148_4R_PSP_DIV5 | target_fibrous | 0.4295/0.9965 | 0.5124/0.9993 | 0.2896/0.8926 |
| PN148_4R_PSP_DIV5 | target_clump | not_applicable/not_applicable | not_applicable/not_applicable | not_applicable/not_applicable |
| PN148_4R_PSP_DIV5 | target_uncertain_ignore | 0.3350/0.9724 | 0.6097/0.9991 | 0.2489/0.7844 |
| PN148_4R_PSP_DIV5 | target_background | 0.0853/0.7858 | 0.0502/0.4510 | 0.0403/0.3247 |

## Intensity And Domain Gap

| Source | Mean | Std | p50 | p95 | p99 | Zero frac | Local var mean | Local var p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_4R_PSP_DIV5 | 24.5244 | 28.0057 | 14.0000 | 84.0000 | 134.0000 | 0.0239 | 76.3070 | 311.1226 |
| synthetic validation | not_available | no readable validation render_uint8 arrays |  |  |  |  |  |  |

## Main Visual Failure Modes

- Semantic predictions over-segment fibrous area in both pilots, with predicted/target area ratios above 1.0 and many small false-positive components.
- Boundary disagreement contributes to both false-positive and false-negative fibrous errors, but large missed regions remain visible in the component summaries.
- Skeleton predictions are mostly constrained to target fibrous regions, but exact skeleton Dice remains low; 2 px tolerant recovery is substantially higher than exact-pixel Dice.
- Raising the skeleton threshold to 0.85 improves precision in places but lowers recall and does not remove the need to inspect skeleton failure cases.
- Leakage flags: clump-vs-fibrous confusion, diffuse background false positives, fibrous leakage into expert uncertain_ignore, skeleton leakage into uncertain_ignore

## Conclusion

collect more real annotations before deciding.
