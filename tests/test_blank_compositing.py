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
        name = f"PN001_3R_AD_DIV01 (Series {i}) [1].tif"
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
                "culture_id": "PN001",
                "disease": "AD",
                "tau_isoform": "3R",
                "experimental_condition": "AD_3R",
                "div": "1",
                "div_token": "DIV01",
                "series_index": str(i),
                "experimental_group_id": "PN001_AD_3R_DIV01",
                "acquisition_group": "PN001_3R_AD_DIV01",
                "blank_status": "expert_validated",
            }
        )
    inventory_dir = tmp_path / "manifests"
    inventory_dir.mkdir()
    write_csv(inventory_dir / "sted_blanks.csv", rows)
    write_csv(inventory_dir / "sted_images.csv", [dict(rows[0], source_kind="fiber_image", stable_image_id="fiber0")])
    write_csv(inventory_dir / "acquisition_groups.csv", [{"source_kind": "blank_background", "culture_id": "PN001", "experimental_group_id": "PN001_AD_3R_DIV01", "acquisition_group": "PN001_3R_AD_DIV01"}])
    split_rows = [
        {
            "stable_image_id": row["stable_image_id"],
            "source_kind": "blank_background",
            "culture_id": "PN001",
            "disease": "AD",
            "tau_isoform": "3R",
            "experimental_condition": "AD_3R",
            "div": "1",
            "div_token": "DIV01",
            "experimental_group_id": "PN001_AD_3R_DIV01",
            "acquisition_group": row["acquisition_group"],
            "series_index": row["series_index"],
            "eligibility": "calibration",
            "primary_metric_role": "calibration",
            "secondary_image_eval_role": "not_assigned",
            "assignment_reason": "fixture",
            "grouping_rule": "experimental_group_holdout_primary",
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


def test_blank_pool_reuse_is_opt_in_within_split(tmp_path):
    blank_root = tmp_path / "blank_root"
    blank_root.mkdir()
    rows = []
    for i in range(4):
        name = f"PN001_3R_AD_DIV01 (Series {i}) [1].tif"
        Image.fromarray(np.full((64, 64), 4 + i, dtype=np.uint8)).save(blank_root / name)
        rows.append(
            {
                "source_root_id": "sted_blank_data",
                "relative_path": name,
                "source_kind": "blank_background",
                "source_sha256": f"sha{i}",
                "pixel_sha256": f"pix{i}",
                "stable_image_id": f"blank{i}",
                "culture_id": "PN001",
                "disease": "AD",
                "tau_isoform": "3R",
                "experimental_condition": "AD_3R",
                "div": "1",
                "div_token": "DIV01",
                "series_index": str(i),
                "experimental_group_id": "PN001_AD_3R_DIV01",
                "acquisition_group": f"PN001_3R_AD_DIV01_{i}",
                "blank_status": "expert_validated",
            }
        )
    inventory_dir = tmp_path / "manifests"
    inventory_dir.mkdir()
    write_csv(inventory_dir / "sted_blanks.csv", rows)
    write_csv(
        inventory_dir / "blank_pools.csv",
        [
            dict(
                row,
                blank_pool_role="synthetic_background_train",
                assignment_reason="fixture",
                human_approved="false",
                validation_source="human_expert_review",
                validator_role="STED expert",
                validation_date="not_recorded",
                notes="",
            )
            for row in rows
        ],
    )
    splits = tmp_path / "splits.csv"
    write_csv(
        splits,
        [
            dict(
                row,
                eligibility="training",
                primary_metric_role="training",
                secondary_image_eval_role="not_assigned",
                assignment_reason="fixture",
                grouping_rule="fixture",
                human_approved="false",
                synthetic_split="synthetic_background_train",
                override_status="none",
                override_reason="none",
            )
            for row in rows
        ],
    )
    config = {
        "dataset_name": "fixture_reused_composite",
        "calibration_data_status": "exploratory_unpartitioned",
        "source_roots": {"sted_blank_data": str(blank_root)},
        "geometry": {"image_shape": [64, 64], "base_seed": 1, "fiber_count_range": [7, 7], "points_per_fiber": 24},
        "targets": {},
        "real_blank_rendering": {"base_seed": 2, "source_signal_scale": 1.0, "psf_sigma_px": 0.5, "uint8_min": 0, "uint8_max": 255},
        "compositing": {
            "base_seed": 3,
            "sample_count": 8,
            "synthetic_split": "synthetic_background_train",
            "blank_pool_role": "synthetic_background_train",
            "blank_pool_manifest": str(inventory_dir / "blank_pools.csv"),
            "allow_blank_reuse_within_pool": True,
            "blank_scale": 1.0,
        },
    }
    out = tmp_path / "out_reuse"
    generate_composites(config, inventory_dir, splits, out)
    assert validate_composites(out) == []
    with (out / "dataset_manifest.csv").open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 8


def test_3d_blank_compositing_preserves_normalized_targets(tmp_path):
    blank_root = tmp_path / "blank_root"
    blank_root.mkdir()
    rows = []
    for i in range(8):
        name = f"PN001_3R_AD_DIV01 (Series {i}) [1].tif"
        Image.fromarray(np.full((64, 64), 5 + i, dtype=np.uint8)).save(blank_root / name)
        rows.append(
            {
                "source_root_id": "sted_blank_data",
                "relative_path": name,
                "source_kind": "blank_background",
                "source_sha256": f"sha{i}",
                "pixel_sha256": f"pix{i}",
                "stable_image_id": f"blank{i}",
                "culture_id": "PN001",
                "disease": "AD",
                "tau_isoform": "3R",
                "experimental_condition": "AD_3R",
                "div": "1",
                "div_token": "DIV01",
                "series_index": str(i),
                "experimental_group_id": "PN001_AD_3R_DIV01",
                "acquisition_group": f"PN001_3R_AD_DIV01_{i}",
                "blank_status": "expert_validated",
            }
        )
    inventory_dir = tmp_path / "manifests"
    inventory_dir.mkdir()
    write_csv(inventory_dir / "sted_blanks.csv", rows)
    split_rows = [dict(row, eligibility="calibration", primary_metric_role="calibration", secondary_image_eval_role="not_assigned", assignment_reason="fixture", grouping_rule="fixture", human_approved="false", synthetic_split="calibration", override_status="none", override_reason="none") for row in rows]
    splits = tmp_path / "splits.csv"
    write_csv(splits, split_rows)
    config = {
        "dataset_name": "fixture_3d_composite",
        "generator_mode": "persistent_chain_3d",
        "calibration_data_status": "exploratory_unpartitioned",
        "source_roots": {"sted_blank_data": str(blank_root)},
        "geometry": {
            "image_shape": [64, 64],
            "volume_depth_px": 32.0,
            "focal_plane_z_px": 16.0,
            "base_seed": 10,
            "scenario": "projected_depth_crossing",
            "arc_length_sampling_interval_px": 1.0,
            "width_calibration": {"length_px": 42.0},
            "fluorophore": {"base_amplitude_range": [120, 120], "variation_amplitude": 0.0, "gap_probability": 0.0, "radius_range_px": [0.7, 0.7]},
        },
        "targets": {"semantic_mask_source": "visible_signal_mask", "ignore_mask_rule": "none"},
        "optical_model": {
            "psf_mode": "core_plus_halo",
            "core_weight": 0.9,
            "halo_weight": 0.1,
            "core_sigma_xy_0_px": 2.123,
            "core_sigma_z_px": 12.0,
            "halo_sigma_xy_0_px": 5.0,
            "halo_sigma_z_px": 24.0,
            "kernel_truncation_radius": 3.0,
            "focal_depth_range_px": 4.0,
            "visible_signal_threshold": 2.0,
        },
        "output_mapping": {"base_seed": 2, "foreground_scale": 1.0, "background_level": 0.0, "uint8_min": 0, "uint8_max": 255},
        "real_blank_rendering": {"uint8_min": 0, "uint8_max": 255, "signal_dependent_perturbation_scale": 0.0},
        "compositing": {"base_seed": 3, "sample_count": 8, "synthetic_split": "calibration", "blank_scale": 1.0, "foreground_scale": 1.0},
    }
    out = tmp_path / "out3d"
    generate_composites(config, inventory_dir, splits, out)
    assert validate_composites(out) == []
    with np.load(out / "fixture_3d_composite_0000.npz", allow_pickle=False) as data:
        assert "trace_points_xy" in data.files
        assert "ignore_mask" in data.files
        assert np.allclose(data["in_focus_signal"] + data["out_of_focus_signal"], data["total_clean_signal"], atol=1e-4)
