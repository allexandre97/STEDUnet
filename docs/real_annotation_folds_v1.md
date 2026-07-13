# Real Annotation Grouped Cross-Validation Folds v1

Five fixed outer folds use `experimental_group_id` as the provisional indivisible sample group. Each outer development set has a grouped validation partition used only for checkpoint selection.

Perfect stratification is impossible: CBD has one four-image group, PID has two groups, and multiple disease/isoform/DIV strata occur in only one group. Five folds remain defensible because 12 groups can populate five nonempty test partitions with four images each or close to it.

Two images with semantic overlaps remain assigned so every inventory image appears in outer test exactly once, but `validation_status=invalid` excludes them from training and evaluation until corrected.

| Fold | Train images/groups | Validation images/groups | Test images/groups | Test diseases | Test density | Invalid test |
|--:|:--|:--|:--|:--|:--|--:|
| 0 | 13/8 | 3/3 | 4/1 | CBD:4 | dense:1, sparse:3 | 0 |
| 1 | 12/6 | 3/3 | 5/3 | AD:3, PSP:2 | dense:2, sparse:3 | 0 |
| 2 | 14/8 | 3/3 | 3/1 | PID:3 | dense:2, sparse:1 | 0 |
| 3 | 13/6 | 3/3 | 4/3 | AD:3, PSP:1 | dense:3, sparse:1 | 0 |
| 4 | 13/6 | 3/2 | 4/4 | AD:2, PID:1, PSP:1 | dense:2, sparse:2 | 0 |

## Leakage Contract

- Every image occurs in the outer test partition exactly once.
- Every `split_group_id` is wholly train, validation, or test within a fold.
- Crop manifests must inherit the parent image partition from this file.
- Outer test rows are never used for checkpoint selection.
- Fold generation is deterministic and checked byte-for-byte; experiments consume the committed manifest rather than regenerating folds.
