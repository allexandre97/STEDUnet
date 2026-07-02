# Synthetic Clump/Ignore Stress QA

## Synthetic

- Samples: 640
- Scenario counts: `{"bundle_dominated": 81, "clump_dominated": 81, "clustered_filament_network": 81, "isolated_filaments": 81, "mixed_morphology": 316}`
- Scenario category counts: `{"clump_ignore_stress": 160, "realism_calibration": 480}`
- Sample variant counts: `{"clump_ignore_hard_negative": 160, "normal": 480}`
- Mean clump area fraction: 0.0060
- Clump area fraction mean/p50/p95/max: 0.0060/0.0023/0.0277/0.0510
- Largest clump component fraction p50/p95/max: 0.0013/0.0147/0.0337
- Samples with largest clump >5%/>10% image area: 0.000/0.000
- Uncertain_ignore fraction mean/p50/p95/max: 0.0036/0.0022/0.0132/0.0277
- Samples with no clumps/with clumps/stress category: 0.311/0.689/0.250
- Skeleton pixels inside clump: 0
- Skeleton pixels inside uncertain_ignore: 0
- Fibrous_tau pixels inside clump: 0
- Fibrous_tau pixels inside uncertain_ignore: 0
- Intensity p50/p95/p99: 4.00/4.85/27.11
- Clump-region intensity p50/p95/p99: 26.42/96.35/117.55
- Ignore-region intensity p50/p95/p99: 9.05/43.02/88.40
- 128x128 max clump fraction and crop fractions >25%/>50%: 0.146/0.007/0.001

## Composite

- Samples: 640
- Scenario counts: `{"bundle_dominated": 81, "clump_dominated": 81, "clustered_filament_network": 81, "isolated_filaments": 81, "mixed_morphology": 316}`
- Scenario category counts: `{"clump_ignore_stress": 160, "realism_calibration": 480}`
- Sample variant counts: `{"not_reported": 640}`
- Mean clump area fraction: 0.0060
- Clump area fraction mean/p50/p95/max: 0.0060/0.0023/0.0277/0.0510
- Largest clump component fraction p50/p95/max: 0.0013/0.0147/0.0337
- Samples with largest clump >5%/>10% image area: 0.000/0.000
- Uncertain_ignore fraction mean/p50/p95/max: 0.0036/0.0022/0.0132/0.0277
- Samples with no clumps/with clumps/stress category: 0.311/0.689/0.250
- Skeleton pixels inside clump: 0
- Skeleton pixels inside uncertain_ignore: 0
- Fibrous_tau pixels inside clump: 0
- Fibrous_tau pixels inside uncertain_ignore: 0
- Intensity p50/p95/p99: 3.27/12.75/44.85
- Clump-region intensity p50/p95/p99: 41.57/119.29/142.88
- Ignore-region intensity p50/p95/p99: 12.36/68.56/114.64
- 128x128 max clump fraction and crop fractions >25%/>50%: 0.146/0.007/0.001

QA panels are in `panels/`.
