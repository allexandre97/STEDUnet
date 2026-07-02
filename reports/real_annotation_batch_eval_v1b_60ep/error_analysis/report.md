# Real-Pilot First-Baseline Error Analysis

This report analyzes existing synthetic-only baseline predictions on the two real expert-annotated pilot images. It does not change training, architecture, synthetic generation, schema, or annotation protocol.

## Semantic Errors

| Pilot | Target fibrous frac | Pred fibrous frac | Pred/target | FP background px | FN fibrous px | FP comps | Large FP | Tiny FP | FN comps | Missed faint | Missed thick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 0.0332 | 0.0571 | 1.7210 | 30999 | 5949 | 535 | 65 | 343 | 324 | 287 | 10 |
| PN148_4R_AD_DIV03 | 0.0159 | 0.1205 | 7.5997 | 72697 | 5065 | 1575 | 86 | 1063 | 286 | 227 | 12 |
| PN148_4R_PSP_DIV5 | 0.1200 | 0.1270 | 1.0581 | 74656 | 67739 | 1670 | 151 | 1093 | 748 | 666 | 82 |
| PN151_4R_CBD_DIV05 | 0.0504 | 0.0603 | 1.1974 | 20326 | 9932 | 1386 | 43 | 1019 | 830 | 774 | 18 |

Boundary-disagreement fractions estimate how much FP/FN area lies within 2 px of the opposite fibrous mask.

| Pilot | FP within 2 px of target | FN within 2 px of prediction | FP area p50/p95/max | FN area p50/p95/max |
| --- | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 0.3729 | 0.5673 | 2.0000/278.9000/2389.0000 | 2.0000/79.0000/777.0000 |
| PN148_4R_AD_DIV03 | 0.0559 | 0.3828 | 3.0000/111.3000/7903.0000 | 2.0000/92.7500/231.0000 |
| PN148_4R_PSP_DIV5 | 0.2861 | 0.1774 | 2.0000/185.1000/3009.0000 | 2.0000/229.6500/22770.0000 |
| PN151_4R_CBD_DIV05 | 0.7160 | 0.5278 | 2.0000/69.0000/372.0000 | 1.0000/57.0000/633.0000 |

## Skeleton Threshold Sensitivity

| Pilot | Threshold | Dice | Precision | Recall | Target recovered <=2 px | Pred within <=2 px | Pred px inside target fibrous | Pred px outside target fibrous | Missed target comps | Unmatched pred comps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 0.5 | 0.3261 | 0.2278 | 0.5737 | 0.8587 | 0.5168 | 18971 | 11395 | 0 | 147 |
| PN148_3R_PID_DIV05 | 0.75 | 0.2936 | 0.2613 | 0.3351 | 0.7277 | 0.5759 | 10951 | 4509 | 0 | 179 |
| PN148_3R_PID_DIV05 | 0.85 | 0.2165 | 0.2761 | 0.1781 | 0.5794 | 0.6061 | 5747 | 2030 | 0 | 265 |
| PN148_4R_AD_DIV03 | 0.5 | 0.0722 | 0.0385 | 0.5668 | 0.8277 | 0.0841 | 5273 | 51502 | 0 | 1168 |
| PN148_4R_AD_DIV03 | 0.75 | 0.1662 | 0.1050 | 0.3987 | 0.7218 | 0.2066 | 3296 | 11368 | 1 | 1289 |
| PN148_4R_AD_DIV03 | 0.85 | 0.2690 | 0.2429 | 0.3013 | 0.6446 | 0.4347 | 2246 | 2541 | 7 | 644 |
| PN148_4R_PSP_DIV5 | 0.5 | 0.2408 | 0.1694 | 0.4163 | 0.6393 | 0.4026 | 32938 | 26432 | 14 | 655 |
| PN148_4R_PSP_DIV5 | 0.75 | 0.2448 | 0.2461 | 0.2436 | 0.5131 | 0.5464 | 16920 | 6997 | 19 | 759 |
| PN148_4R_PSP_DIV5 | 0.85 | 0.2027 | 0.3150 | 0.1494 | 0.3963 | 0.6536 | 9012 | 2444 | 25 | 520 |
| PN151_4R_CBD_DIV05 | 0.5 | 0.3833 | 0.3731 | 0.3941 | 0.8222 | 0.7544 | 17736 | 2883 | 1 | 43 |
| PN151_4R_CBD_DIV05 | 0.75 | 0.3384 | 0.3970 | 0.2949 | 0.7562 | 0.7883 | 12951 | 1547 | 1 | 75 |
| PN151_4R_CBD_DIV05 | 0.85 | 0.2991 | 0.4125 | 0.2347 | 0.7059 | 0.8108 | 10152 | 951 | 1 | 109 |

## Clump And Ignore Leakage

| Pilot | Pred fib in target fib | Pred fib in clump | Pred fib in ignore | Pred fib in background | Fib frac clump | Fib frac ignore | Pred clump in clump | Pred clump in fibrous | Pred clump in ignore | Pred clump in background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 28795 | 0 | 368 | 30999 | 0.0000 | 0.0061 | 0 | 3055 | 144 | 2086 |
| PN148_4R_AD_DIV03 | 9492 | 28440 | 69601 | 72697 | 0.1578 | 0.3862 | 80593 | 2927 | 45596 | 21260 |
| PN148_4R_PSP_DIV5 | 51269 | 0 | 18070 | 74656 | 0.0000 | 0.1255 | 0 | 61618 | 36016 | 42785 |
| PN151_4R_CBD_DIV05 | 42731 | 0 | 703 | 20326 | 0.0000 | 0.0110 | 0 | 4828 | 1057 | 2586 |

| Pilot | Threshold | Skel in fibrous | Skel in clump | Skel in ignore | Skel in background | Frac clump | Frac ignore | Skel in pred fib | Skel in pred clump | Skel outside pred foreground |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 0.5 | 18971 | 0 | 248 | 11395 | 0.0000 | 0.0081 | 29344 | 624 | 646 |
| PN148_3R_PID_DIV05 | 0.75 | 10951 | 0 | 134 | 4509 | 0.0000 | 0.0086 | 15551 | 34 | 9 |
| PN148_3R_PID_DIV05 | 0.85 | 5747 | 0 | 62 | 2030 | 0.0000 | 0.0079 | 7838 | 1 | 0 |
| PN148_4R_AD_DIV03 | 0.5 | 5273 | 21260 | 42297 | 30242 | 0.2146 | 0.4269 | 93998 | 4204 | 870 |
| PN148_4R_AD_DIV03 | 0.75 | 3296 | 5070 | 14198 | 6298 | 0.1757 | 0.4919 | 28653 | 180 | 29 |
| PN148_4R_AD_DIV03 | 0.85 | 2246 | 920 | 5012 | 1621 | 0.0939 | 0.5115 | 9788 | 11 | 0 |
| PN148_4R_PSP_DIV5 | 0.5 | 32938 | 0 | 11900 | 26432 | 0.0000 | 0.1670 | 66223 | 4221 | 826 |
| PN148_4R_PSP_DIV5 | 0.75 | 16920 | 0 | 3946 | 6997 | 0.0000 | 0.1416 | 27628 | 197 | 38 |
| PN148_4R_PSP_DIV5 | 0.85 | 9012 | 0 | 1125 | 2444 | 0.0000 | 0.0894 | 12571 | 9 | 1 |
| PN151_4R_CBD_DIV05 | 0.5 | 17736 | 0 | 450 | 2883 | 0.0000 | 0.0214 | 19702 | 688 | 679 |
| PN151_4R_CBD_DIV05 | 0.75 | 12951 | 0 | 233 | 1547 | 0.0000 | 0.0158 | 14482 | 185 | 64 |
| PN151_4R_CBD_DIV05 | 0.85 | 10152 | 0 | 133 | 951 | 0.0000 | 0.0118 | 11171 | 61 | 4 |

## Probability Summaries By Expert Region

| Pilot | Region | Fib prob mean/p95 | Clump prob mean/p95 | Skeleton prob mean/p95 |
| --- | --- | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | target_fibrous | 0.7971/0.9976 | 0.1032/0.6761 | 0.5002/0.9248 |
| PN148_3R_PID_DIV05 | target_clump | not_applicable/not_applicable | not_applicable/not_applicable | not_applicable/not_applicable |
| PN148_3R_PID_DIV05 | target_uncertain_ignore | 0.6073/0.9728 | 0.2987/0.8715 | 0.4154/0.9180 |
| PN148_3R_PID_DIV05 | target_background | 0.0313/0.0850 | 0.0030/0.0008 | 0.0148/0.0293 |
| PN148_4R_AD_DIV03 | target_fibrous | 0.6237/0.9968 | 0.2163/0.9099 | 0.3747/0.9556 |
| PN148_4R_AD_DIV03 | target_clump | 0.2683/0.9935 | 0.7263/1.0000 | 0.1982/0.7430 |
| PN148_4R_AD_DIV03 | target_uncertain_ignore | 0.5163/0.9873 | 0.3535/0.9775 | 0.3398/0.8312 |
| PN148_4R_AD_DIV03 | target_background | 0.0905/0.8051 | 0.0304/0.1576 | 0.0481/0.4119 |
| PN148_4R_PSP_DIV5 | target_fibrous | 0.4295/0.9965 | 0.5124/0.9993 | 0.2896/0.8926 |
| PN148_4R_PSP_DIV5 | target_clump | not_applicable/not_applicable | not_applicable/not_applicable | not_applicable/not_applicable |
| PN148_4R_PSP_DIV5 | target_uncertain_ignore | 0.3350/0.9724 | 0.6097/0.9991 | 0.2489/0.7844 |
| PN148_4R_PSP_DIV5 | target_background | 0.0853/0.7858 | 0.0502/0.4510 | 0.0403/0.3247 |
| PN151_4R_CBD_DIV05 | target_fibrous | 0.7893/0.9997 | 0.1058/0.7513 | 0.3416/0.9888 |
| PN151_4R_CBD_DIV05 | target_clump | not_applicable/not_applicable | not_applicable/not_applicable | not_applicable/not_applicable |
| PN151_4R_CBD_DIV05 | target_uncertain_ignore | 0.2466/0.9500 | 0.3232/0.9452 | 0.1693/0.8224 |
| PN151_4R_CBD_DIV05 | target_background | 0.0208/0.0049 | 0.0033/0.0002 | 0.0041/0.0012 |

## Intensity And Domain Gap

| Source | Mean | Std | p50 | p95 | p99 | Zero frac | Local var mean | Local var p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PN148_3R_PID_DIV05 | 12.8460 | 12.3172 | 9.0000 | 36.0000 | 61.0000 | 0.0481 | 28.3781 | 85.9633 |
| PN148_4R_AD_DIV03 | 32.7143 | 40.7152 | 19.0000 | 124.0000 | 204.0000 | 0.0267 | 79.8775 | 317.0664 |
| PN148_4R_PSP_DIV5 | 24.5244 | 28.0057 | 14.0000 | 84.0000 | 134.0000 | 0.0239 | 76.3070 | 311.1226 |
| PN151_4R_CBD_DIV05 | 7.3321 | 8.7658 | 4.0000 | 23.0000 | 46.0000 | 0.0824 | 23.7902 | 117.4505 |
| synthetic validation | not_available | no readable validation render_uint8 arrays |  |  |  |  |  |  |

## Main Visual Failure Modes

- Semantic predictions over-segment fibrous area in both pilots, with predicted/target area ratios above 1.0 and many small false-positive components.
- Boundary disagreement contributes to both false-positive and false-negative fibrous errors, but large missed regions remain visible in the component summaries.
- Skeleton predictions are mostly constrained to target fibrous regions, but exact skeleton Dice remains low; 2 px tolerant recovery is substantially higher than exact-pixel Dice.
- Raising the skeleton threshold to 0.85 improves precision in places but lowers recall and does not remove the need to inspect skeleton failure cases.
- Leakage flags: clump-vs-fibrous confusion, diffuse background false positives, fibrous leakage into expert clump, fibrous leakage into expert uncertain_ignore, skeleton leakage into clump, skeleton leakage into uncertain_ignore

## Conclusion

collect more real annotations before deciding.
