# STED PN-level split report

## Policy

- `PN###` is treated as a biological sample and the indivisible primary partitioning unit.
- Condition and DIV are stratification/reporting variables, not grouping boundaries.
- Round and series are retained as metadata and reporting variables.
- The primary held-out test contains only PNs absent from calibration, training, and validation.
- `human_approved=false` is retained until this allocation and the stratum counts are reviewed.
- Secondary image-level evaluation roles, if used later, must not be mixed with the primary PN-held-out metric.

## Allocation warning

- Only 2 independent PN group(s) are available in the manifest.
- Under PN-level partitioning, this is insufficient to safely populate calibration, training, validation, and held-out test roles with independent biological samples.
- Condition/DIV coverage cannot be balanced across train, validation, and test until more independent PNs or an approved allocation are available.

## PN allocation

| PN | Split | Source kinds | Images | Conditions | DIVs |
|:--|:--|:--|--:|:--|:--|
| `PN148` | `calibration` | `blank_background`, `fiber_image` | 421 | `AD`, `PID`, `PSP` | `DIV03`, `DIV05`, `DIV07`, `DIV10`, `DIV11`, `DIV13`, `DIV15`, `DIV3`, `DIV5`, `DIV7` |
| `PN151` | `held_out_test` | `blank_background`, `fiber_image` | 111 | `CBD` | `DIV03`, `DIV05`, `DIV07`, `DIV09`, `DIV11` |

## Split summary

| Split | PNs | Images | Fiber images | Blank images |
|:--|--:|--:|--:|--:|
| `calibration` | 1 | 421 | 361 | 60 |
| `held_out_test` | 1 | 111 | 77 | 34 |

## Condition and DIV strata

| Condition | DIV | PNs | Images | Splits represented | Warning |
|:--|:--|--:|--:|:--|:--|
| `AD` | `DIV03` | 1 | 30 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `AD` | `DIV05` | 1 | 30 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `AD` | `DIV07` | 1 | 30 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `AD` | `DIV10` | 1 | 30 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `AD` | `DIV11` | 1 | 30 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `AD` | `DIV13` | 1 | 45 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `AD` | `DIV15` | 1 | 30 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `CBD` | `DIV03` | 1 | 16 | held_out_test | too_few_independent_pns_for_safe_distribution_lt_3 |
| `CBD` | `DIV05` | 1 | 16 | held_out_test | too_few_independent_pns_for_safe_distribution_lt_3 |
| `CBD` | `DIV07` | 1 | 15 | held_out_test | too_few_independent_pns_for_safe_distribution_lt_3 |
| `CBD` | `DIV09` | 1 | 15 | held_out_test | too_few_independent_pns_for_safe_distribution_lt_3 |
| `CBD` | `DIV11` | 1 | 49 | held_out_test | too_few_independent_pns_for_safe_distribution_lt_3 |
| `PID` | `DIV05` | 1 | 15 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `PID` | `DIV10` | 1 | 30 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `PID` | `DIV3` | 1 | 15 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `PID` | `DIV7` | 1 | 15 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `PSP` | `DIV10` | 1 | 15 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `PSP` | `DIV11` | 1 | 15 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `PSP` | `DIV13` | 1 | 15 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `PSP` | `DIV15` | 1 | 30 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `PSP` | `DIV3` | 1 | 16 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `PSP` | `DIV5` | 1 | 15 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |
| `PSP` | `DIV7` | 1 | 15 | calibration | too_few_independent_pns_for_safe_distribution_lt_3 |

## Split x condition x DIV counts

### `calibration`

| Condition | DIV | PNs | Images |
|:--|:--|--:|--:|
| `AD` | `DIV03` | 1 | 30 |
| `AD` | `DIV05` | 1 | 30 |
| `AD` | `DIV07` | 1 | 30 |
| `AD` | `DIV10` | 1 | 30 |
| `AD` | `DIV11` | 1 | 30 |
| `AD` | `DIV13` | 1 | 45 |
| `AD` | `DIV15` | 1 | 30 |
| `PID` | `DIV05` | 1 | 15 |
| `PID` | `DIV10` | 1 | 30 |
| `PID` | `DIV3` | 1 | 15 |
| `PID` | `DIV7` | 1 | 15 |
| `PSP` | `DIV10` | 1 | 15 |
| `PSP` | `DIV11` | 1 | 15 |
| `PSP` | `DIV13` | 1 | 15 |
| `PSP` | `DIV15` | 1 | 30 |
| `PSP` | `DIV3` | 1 | 16 |
| `PSP` | `DIV5` | 1 | 15 |
| `PSP` | `DIV7` | 1 | 15 |

### `held_out_test`

| Condition | DIV | PNs | Images |
|:--|:--|--:|--:|
| `CBD` | `DIV03` | 1 | 16 |
| `CBD` | `DIV05` | 1 | 16 |
| `CBD` | `DIV07` | 1 | 15 |
| `CBD` | `DIV09` | 1 | 15 |
| `CBD` | `DIV11` | 1 | 49 |
