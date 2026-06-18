# STED data inventory

## Methodology

- Reader: Pillow (`PIL.Image`).
- Statistics in manifests are per image and exact over all decoded pixels.
- Global percentiles in this report are exact over all readable decoded pixels.
- Source identity uses source root ID, relative path, source SHA-256, pixel SHA-256, and stable image ID.
- File duplicates use full-file SHA-256 equality.
- Pixel duplicates use decoded array shape, dtype, and bytes.
- Thumbnail duplicates use deterministic p1/p99-normalized 32x32 thumbnail hashes.
- Near-duplicate candidates use normalized thumbnail NCC with threshold 0.995, excluding exact thumbnail duplicates.
- Elongated-component QC threshold is `max(p99.9, median + 8 * 1.4826 * MAD, 20)` with area >=12 px and elongation >=2.5.

## Fiber images

- Files: 438
- Readable: 438
- Global percentiles: {"p0": "0", "p0_1": "0", "p1": "0", "p100": "255", "p25": "1", "p5": "0", "p50": "4", "p75": "9", "p95": "24", "p99": "61", "p99_9": "145"}
- Validation flags: {"elongated_high_intensity_component": 100, "none": 304, "saturation": 38, "strong_row_or_column_variation": 3}

## Expert-validated blanks

- Files: 94
- Readable: 94
- Blank status: expert_validated
- Validation source: human_expert_review
- Global percentiles: {"p0": "0", "p0_1": "0", "p1": "0", "p100": "169", "p25": "1", "p5": "0", "p50": "4", "p75": "8", "p95": "16", "p99": "23", "p99_9": "34"}
- Validation flags: {"none": 93, "strong_row_or_column_variation": 1}
