# Real-Pilot First-Baseline Error Analysis

This report analyzes existing synthetic-only baseline predictions on the two real expert-annotated pilot images. It does not change training, architecture, synthetic generation, schema, or annotation protocol.

## Semantic Errors

| Pilot | Target fibrous frac | Pred fibrous frac | Pred/target | FP background px | FN fibrous px | FP comps | Large FP | Tiny FP | FN comps | Missed faint | Missed thick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN151_4R_CBD_DIV05 | 0.0504 | 0.0603 | 1.1974 | 20326 | 9932 | 1386 | 43 | 1019 | 830 | 774 | 18 |

Boundary-disagreement fractions estimate how much FP/FN area lies within 2 px of the opposite fibrous mask.

| Pilot | FP within 2 px of target | FN within 2 px of prediction | FP area p50/p95/max | FN area p50/p95/max |
| --- | ---: | ---: | ---: | ---: |
| PN151_4R_CBD_DIV05 | 0.7160 | 0.5278 | 2.0000/69.0000/372.0000 | 1.0000/57.0000/633.0000 |

## Skeleton Threshold Sensitivity

| Pilot | Threshold | Dice | Precision | Recall | Target recovered <=2 px | Pred within <=2 px | Pred px inside target fibrous | Pred px outside target fibrous | Missed target comps | Unmatched pred comps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN151_4R_CBD_DIV05 | 0.5 | 0.3833 | 0.3731 | 0.3941 | 0.8222 | 0.7544 | 17736 | 2883 | 1 | 43 |
| PN151_4R_CBD_DIV05 | 0.75 | 0.3384 | 0.3970 | 0.2949 | 0.7562 | 0.7883 | 12951 | 1547 | 1 | 75 |
| PN151_4R_CBD_DIV05 | 0.85 | 0.2991 | 0.4125 | 0.2347 | 0.7059 | 0.8108 | 10152 | 951 | 1 | 109 |

## Clump And Ignore Leakage

| Pilot | Pred fib in target fib | Pred fib in clump | Pred fib in ignore | Pred fib in background | Fib frac clump | Fib frac ignore | Pred clump in clump | Pred clump in fibrous | Pred clump in ignore | Pred clump in background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN151_4R_CBD_DIV05 | 42731 | 0 | 703 | 20326 | 0.0000 | 0.0110 | 0 | 4828 | 1057 | 2586 |

| Pilot | Threshold | Skel in fibrous | Skel in clump | Skel in ignore | Skel in background | Frac clump | Frac ignore | Skel in pred fib | Skel in pred clump | Skel outside pred foreground |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN151_4R_CBD_DIV05 | 0.5 | 17736 | 0 | 450 | 2883 | 0.0000 | 0.0214 | 19702 | 688 | 679 |
| PN151_4R_CBD_DIV05 | 0.75 | 12951 | 0 | 233 | 1547 | 0.0000 | 0.0158 | 14482 | 185 | 64 |
| PN151_4R_CBD_DIV05 | 0.85 | 10152 | 0 | 133 | 951 | 0.0000 | 0.0118 | 11171 | 61 | 4 |

## Probability Summaries By Expert Region

| Pilot | Region | Fib prob mean/p95 | Clump prob mean/p95 | Skeleton prob mean/p95 |
| --- | --- | ---: | ---: | ---: |
| PN151_4R_CBD_DIV05 | target_fibrous | 0.7893/0.9997 | 0.1058/0.7513 | 0.3416/0.9888 |
| PN151_4R_CBD_DIV05 | target_clump | not_applicable/not_applicable | not_applicable/not_applicable | not_applicable/not_applicable |
| PN151_4R_CBD_DIV05 | target_uncertain_ignore | 0.2466/0.9500 | 0.3232/0.9452 | 0.1693/0.8224 |
| PN151_4R_CBD_DIV05 | target_background | 0.0208/0.0049 | 0.0033/0.0002 | 0.0041/0.0012 |

## Intensity And Domain Gap

| Source | Mean | Std | p50 | p95 | p99 | Zero frac | Local var mean | Local var p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN151_4R_CBD_DIV05 | 7.3321 | 8.7658 | 4.0000 | 23.0000 | 46.0000 | 0.0824 | 23.7902 | 117.4505 |
| synthetic validation | not_available | no readable validation render_uint8 arrays |  |  |  |  |  |  |

## Main Visual Failure Modes

- Semantic predictions over-segment fibrous area in both pilots, with predicted/target area ratios above 1.0 and many small false-positive components.
- Boundary disagreement contributes to both false-positive and false-negative fibrous errors, but large missed regions remain visible in the component summaries.
- Skeleton predictions are mostly constrained to target fibrous regions, but exact skeleton Dice remains low; 2 px tolerant recovery is substantially higher than exact-pixel Dice.
- Raising the skeleton threshold to 0.85 improves precision in places but lowers recall and does not remove the need to inspect skeleton failure cases.
- Leakage flags: clump-vs-fibrous confusion, diffuse background false positives, fibrous leakage into expert uncertain_ignore, skeleton leakage into uncertain_ignore

## Conclusion

collect more real annotations before deciding.
