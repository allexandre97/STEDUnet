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
- Filename metadata: `PN###` is recorded as `culture_id`; `3R`/`4R` are `tau_isoform`; `AD`, `PID`, `PSP`, and `CBD` are disease labels; `DIV` is parsed as an integer time point plus original token.
- `experimental_condition = disease + '_' + tau_isoform`; `experimental_group_id = culture_id + experimental_condition + canonical DIV`.
- Deprecated inferred columns are retained only for migration compatibility and are not used for new grouping logic.

## Fiber images

- Files: 438
- Readable: 438
- Global percentiles: {"p0": "0", "p0_1": "0", "p1": "0", "p100": "255", "p25": "1", "p5": "0", "p50": "4", "p75": "9", "p95": "24", "p99": "61", "p99_9": "145"}
- Validation flags: {"elongated_high_intensity_component": 100, "none": 304, "saturation": 38, "strong_row_or_column_variation": 3}
- Cultures: {"PN148": 361, "PN151": 77}
- Diseases: {"AD": 195, "CBD": 77, "PID": 60, "PSP": 106}
- Tau isoforms: {"3R": 165, "4R": 273}
- DIVs: {"10": 60, "11": 60, "13": 45, "15": 30, "3": 77, "5": 76, "7": 75, "9": 15}
- Experimental conditions: {"AD_3R": 105, "AD_4R": 90, "CBD_4R": 77, "PID_3R": 60, "PSP_4R": 106}

## Expert-validated blanks

- Files: 94
- Readable: 94
- Blank status: expert_validated
- Validation source: human_expert_review
- Global percentiles: {"p0": "0", "p0_1": "0", "p1": "0", "p100": "169", "p25": "1", "p5": "0", "p50": "4", "p75": "8", "p95": "16", "p99": "23", "p99_9": "34"}
- Validation flags: {"none": 93, "strong_row_or_column_variation": 1}
- Cultures: {"PN148": 60, "PN151": 34}
- Diseases: {"AD": 30, "CBD": 34, "PID": 15, "PSP": 15}
- Tau isoforms: {"3R": 30, "4R": 64}
- DIVs: {"10": 15, "11": 34, "13": 15, "15": 30}
- Experimental conditions: {"AD_3R": 15, "AD_4R": 15, "CBD_4R": 34, "PID_3R": 15, "PSP_4R": 15}
