# Real Annotation Pilot Validation

The expert real annotation protocol validated on two pilot STED examples from the local workstation annotation directory:

```text
/cephfs/mhuang/STED_dataset/manual_annotations
```

Validated file types:

- original image TIFF;
- JFilament `.txt` snakes, with all snakes for an image in one file;
- Labkit `.labeling`.

Agreed label vocabulary:

- `fibers`
- `uncertain_ignore`
- `clump`

Internal remap:

| Real label | Internal value | Internal class |
|---:|---:|:--|
| 0 `background` | 0 | `background` |
| 1 `fibers` | 1 | `fibrous_tau` |
| 2 `uncertain_ignore` | 255 | `uncertain_ignore` |
| 3 `clump` | 3 | `clump` |

## Validation Summary

| Example | Image | Snakes | Labeling | Shape | Labels | Background pixels | Fibrous pixels | Uncertain pixels | Clump pixels | Snakes | Snake points | Usable snakes | Flagged snakes | Flagged statuses |
|:--|:--|:--|:--|:--|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| `PN148_3R_PID_DIV05_Series2` | read | read | read | 1024 x 1024 | `fibers`, `uncertain_ignore`, `clump` | 1013278 | 34744 | 554 | 0 | 39 | 9363 | 35 | 4 | 3 background; 1 mixed overlap |
| `PN151_4R_CBD_DIV05_Series3` | read | read | read | 1024 x 1024 | `fibers`, `uncertain_ignore`, `clump` | 992771 | 52663 | 3142 | 0 | 110 | 16715 | 102 | 8 | 3 background; 5 mixed overlap |

Generated local outputs are under `reports/real_annotation_pilot_check/`. They are real-image-derived QA artifacts and must remain uncommitted.

## Interpretation

The pilot protocol is usable for real-compatible `fibrous_tau` semantic supervision and JFilament-derived fibrous skeleton or axis supervision.

Clump labels are not yet validated because both pilot examples contain zero `clump` pixels.

Flagged snakes should be inspected visually, especially those classified as background or mixed-overlap. These flags are not blocking for the pilot protocol because flagged snakes are excluded from confident skeleton supervision.
