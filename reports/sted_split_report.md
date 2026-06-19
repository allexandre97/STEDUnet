# STED experimental-group split report

## Policy

- `PN###` is recorded as `culture_id`, not as the disease condition and not as an automatic partitioning unit.
- `3R` and `4R` are tau isoforms, not acquisition rounds.
- Disease, tau isoform, DIV, culture, and series are preserved as separate metadata fields.
- Primary development splits use `experimental_group_id = culture_id + disease + tau_isoform + div` as the indivisible grouping unit.
- Experimental condition is `disease + '_' + tau_isoform`.
- Current assignments are provisional and keep `human_approved=false`; no biological independence claim is made.
- Optional `culture_held_out` splitting is supported separately and must not be mixed with primary experimental-group-held-out metrics.

## Allocation warning

- 29 condition/DIV stratum or strata have fewer than 3 independent experimental groups.
- These strata are too small for meaningful stratified train/validation/test allocation without project-owner review.
- Split assignments remain provisional.

## Experimental-group allocation

| Experimental group | Split | Culture | Condition | DIV | Images | Source kinds | Series |
|:--|:--|:--|:--|:--|--:|:--|:--|
| `PN148_AD_3R_DIV03` | `training` | `PN148` | `AD_3R` | `3` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_3R_DIV05` | `validation` | `PN148` | `AD_3R` | `5` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_3R_DIV07` | `held_out_test` | `PN148` | `AD_3R` | `7` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_3R_DIV10` | `calibration` | `PN148` | `AD_3R` | `10` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_3R_DIV11` | `training` | `PN148` | `AD_3R` | `11` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_3R_DIV13` | `training` | `PN148` | `AD_3R` | `13` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_3R_DIV15` | `training` | `PN148` | `AD_3R` | `15` | 30 | `blank_background`, `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_4R_DIV03` | `training` | `PN148` | `AD_4R` | `3` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_4R_DIV05` | `validation` | `PN148` | `AD_4R` | `5` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_4R_DIV07` | `held_out_test` | `PN148` | `AD_4R` | `7` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_4R_DIV10` | `calibration` | `PN148` | `AD_4R` | `10` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_4R_DIV11` | `training` | `PN148` | `AD_4R` | `11` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_AD_4R_DIV13` | `training` | `PN148` | `AD_4R` | `13` | 30 | `blank_background`, `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_PID_3R_DIV03` | `training` | `PN148` | `PID_3R` | `3` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_PID_3R_DIV05` | `validation` | `PN148` | `PID_3R` | `5` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_PID_3R_DIV07` | `held_out_test` | `PN148` | `PID_3R` | `7` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_PID_3R_DIV10` | `calibration` | `PN148` | `PID_3R` | `10` | 30 | `blank_background`, `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_PSP_4R_DIV03` | `training` | `PN148` | `PSP_4R` | `3` | 16 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14`, `15` |
| `PN148_PSP_4R_DIV05` | `validation` | `PN148` | `PSP_4R` | `5` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_PSP_4R_DIV07` | `held_out_test` | `PN148` | `PSP_4R` | `7` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_PSP_4R_DIV10` | `calibration` | `PN148` | `PSP_4R` | `10` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_PSP_4R_DIV11` | `training` | `PN148` | `PSP_4R` | `11` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_PSP_4R_DIV13` | `training` | `PN148` | `PSP_4R` | `13` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN148_PSP_4R_DIV15` | `training` | `PN148` | `PSP_4R` | `15` | 30 | `blank_background`, `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN151_CBD_4R_DIV03` | `training` | `PN151` | `CBD_4R` | `3` | 16 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14`, `15` |
| `PN151_CBD_4R_DIV05` | `validation` | `PN151` | `CBD_4R` | `5` | 16 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14`, `15` |
| `PN151_CBD_4R_DIV07` | `held_out_test` | `PN151` | `CBD_4R` | `7` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN151_CBD_4R_DIV09` | `calibration` | `PN151` | `CBD_4R` | `9` | 15 | `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14` |
| `PN151_CBD_4R_DIV11` | `training` | `PN151` | `CBD_4R` | `11` | 49 | `blank_background`, `fiber_image` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14`, `15`, `16`, `17`, `18`, `19`, `20`, `21`, `22`, `23`, `24`, `25`, `26`, `27`, `28`, `29`, `30`, `31`, `32`, `33` |

## Split summary

| Split | Experimental groups | Cultures | Images | Fiber images | Blank images |
|:--|--:|--:|--:|--:|--:|
| `calibration` | 5 | 2 | 90 | 75 | 15 |
| `held_out_test` | 5 | 2 | 75 | 75 | 0 |
| `training` | 14 | 2 | 291 | 212 | 79 |
| `validation` | 5 | 2 | 76 | 76 | 0 |

## Condition and DIV strata

| Experimental condition | DIV | Experimental groups | Images | Splits represented | Warning |
|:--|:--|--:|--:|:--|:--|
| `AD_3R` | `10` | 1 | 15 | calibration | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_3R` | `11` | 1 | 15 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_3R` | `13` | 1 | 15 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_3R` | `15` | 1 | 30 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_3R` | `3` | 1 | 15 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_3R` | `5` | 1 | 15 | validation | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_3R` | `7` | 1 | 15 | held_out_test | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_4R` | `10` | 1 | 15 | calibration | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_4R` | `11` | 1 | 15 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_4R` | `13` | 1 | 30 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_4R` | `3` | 1 | 15 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_4R` | `5` | 1 | 15 | validation | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `AD_4R` | `7` | 1 | 15 | held_out_test | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `CBD_4R` | `11` | 1 | 49 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `CBD_4R` | `3` | 1 | 16 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `CBD_4R` | `5` | 1 | 16 | validation | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `CBD_4R` | `7` | 1 | 15 | held_out_test | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `CBD_4R` | `9` | 1 | 15 | calibration | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `PID_3R` | `10` | 1 | 30 | calibration | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `PID_3R` | `3` | 1 | 15 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `PID_3R` | `5` | 1 | 15 | validation | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `PID_3R` | `7` | 1 | 15 | held_out_test | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `PSP_4R` | `10` | 1 | 15 | calibration | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `PSP_4R` | `11` | 1 | 15 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `PSP_4R` | `13` | 1 | 15 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `PSP_4R` | `15` | 1 | 30 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `PSP_4R` | `3` | 1 | 16 | training | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `PSP_4R` | `5` | 1 | 15 | validation | too_few_experimental_groups_for_safe_distribution_lt_3 |
| `PSP_4R` | `7` | 1 | 15 | held_out_test | too_few_experimental_groups_for_safe_distribution_lt_3 |

## Split x condition x DIV counts

### `calibration`

| Experimental condition | DIV | Experimental groups | Images |
|:--|:--|--:|--:|
| `AD_3R` | `10` | 1 | 15 |
| `AD_4R` | `10` | 1 | 15 |
| `CBD_4R` | `9` | 1 | 15 |
| `PID_3R` | `10` | 1 | 30 |
| `PSP_4R` | `10` | 1 | 15 |

### `held_out_test`

| Experimental condition | DIV | Experimental groups | Images |
|:--|:--|--:|--:|
| `AD_3R` | `7` | 1 | 15 |
| `AD_4R` | `7` | 1 | 15 |
| `CBD_4R` | `7` | 1 | 15 |
| `PID_3R` | `7` | 1 | 15 |
| `PSP_4R` | `7` | 1 | 15 |

### `training`

| Experimental condition | DIV | Experimental groups | Images |
|:--|:--|--:|--:|
| `AD_3R` | `11` | 1 | 15 |
| `AD_3R` | `13` | 1 | 15 |
| `AD_3R` | `15` | 1 | 30 |
| `AD_3R` | `3` | 1 | 15 |
| `AD_4R` | `11` | 1 | 15 |
| `AD_4R` | `13` | 1 | 30 |
| `AD_4R` | `3` | 1 | 15 |
| `CBD_4R` | `11` | 1 | 49 |
| `CBD_4R` | `3` | 1 | 16 |
| `PID_3R` | `3` | 1 | 15 |
| `PSP_4R` | `11` | 1 | 15 |
| `PSP_4R` | `13` | 1 | 15 |
| `PSP_4R` | `15` | 1 | 30 |
| `PSP_4R` | `3` | 1 | 16 |

### `validation`

| Experimental condition | DIV | Experimental groups | Images |
|:--|:--|--:|--:|
| `AD_3R` | `5` | 1 | 15 |
| `AD_4R` | `5` | 1 | 15 |
| `CBD_4R` | `5` | 1 | 16 |
| `PID_3R` | `5` | 1 | 15 |
| `PSP_4R` | `5` | 1 | 15 |
