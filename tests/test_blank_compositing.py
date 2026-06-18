import csv
from pathlib import Path

import numpy as np
from PIL import Image

from fibras.calibration.compositing import generate_composites, validate_composites


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_blank_compositing_preserves_targets_and_provenance(tmp_path):
    blank_root = tmp_path / "blank_root"
    blank_root.mkdir()
    rows = []
    for i in range(8):
        name = f"PN001_1R_AD_DIV01 (Series {i}) [1].tif"
        arr = np.full((64, 64), 4 + i, dtype=np.uint8)
        Image.fromarray(arr).save(blank_root / name)
        rows.append(
            {
                "source_root_id": "sted_blank_data",
                "relative_path": name,
                "source_kind": "blank_background",
                "source_sha256": f"sha{i}",
                "pixel_sha256": f"pix{i}",
                "stable_image_id": f"blank{i}",
                "inferred_pn": "PN001",
                "inferred_round": "1R",
                "inferred_condition": "AD",
                "inferred_div": "DIV01",
                "series_index": str(i),
                "acquisition_group": "PN001_1R_AD_DIV01",
                "biological_group_candidate": "PN001",
                "blank_status": "expert_validated",
            }
        )
    inventory_dir = tmp_path / "manifests"
    inventory_dir.mkdir()
    write_csv(inventory_dir / "sted_blanks.csv", rows)
    write_csv(inventory_dir / "sted_images.csv", [dict(rows[0], source_kind="fiber_image", stable_image_id="fiber0")])
    write_csv(inventory_dir / "acquisition_groups.csv", [{"source_kind": "blank_background", "biological_group_candidate": "PN001", "acquisition_group": "PN001_1R_AD_DIV01"}])
    split_rows = [
        {
            "stable_image_id": row["stable_image_id"],
            "source_kind": "blank_background",
            "biological_group_candidate": "PN001",
            "acquisition_group": row["acquisition_group"],
            "inferred_condition": "AD",
            "inferred_div": "DIV01",
            "inferred_round": "1R",
            "series_index": row["series_index"],
            "eligibility": "calibration",
            "primary_metric_role": "calibration",
            "secondary_image_eval_role": "not_assigned",
            "assignment_reason": "fixture",
            "grouping_rule": "pn_holdout_primary",
            "human_approved": "false",
            "synthetic_split": "calibration",
            "override_status": "none",
            "override_reason": "none",
        }
        for row in rows
    ]
    splits = tmp_path / "splits.csv"
    write_csv(splits, split_rows)
    config = {
        "dataset_name": "fixture_composite",
        "calibration_data_status": "exploratory_unpartitioned",
        "source_roots": {"sted_blank_data": str(blank_root)},
        "geometry": {"image_shape": [64, 64], "base_seed": 1, "fiber_count_range": [7, 7], "points_per_fiber": 24},
        "targets": {},
        "real_blank_rendering": {"base_seed": 2, "source_signal_scale": 1.0, "psf_sigma_px": 0.5, "uint8_min": 0, "uint8_max": 255},
        "compositing": {"base_seed": 3, "sample_count": 8, "synthetic_split": "calibration", "blank_scale": 1.0},
    }
    out = tmp_path / "out"
    generate_composites(config, inventory_dir, splits, out)
    assert validate_composites(out) == []

