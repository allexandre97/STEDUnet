"""Synthetic fiber compositing onto expert-validated real blank backgrounds."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from fibras.sted_inventory import read_image
from fibras.synthetic.geometry import generate_geometry, geometry_to_arrays
from fibras.synthetic.rendering import gaussian_blur, map_to_uint8
from fibras.synthetic.schema import DATASET_SCHEMA_VERSION, GENERATOR_VERSION, NODE_TYPES
from fibras.synthetic.storage import assert_no_object_arrays, sha256_file, write_dataset_manifest
from fibras.synthetic.targets import rasterize_targets

from .schema import CALIBRATION_DATA_STATUSES, CALIBRATION_SCHEMA_VERSION, artifact_metadata


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_blank_lookup(inventory_dir: Path) -> dict[str, dict[str, str]]:
    return {row["stable_image_id"]: row for row in read_csv(inventory_dir / "sted_blanks.csv")}


def select_blank_rows(inventory_dir: Path, splits_path: Path, split: str, count: int) -> list[dict[str, str]]:
    blanks = build_blank_lookup(inventory_dir)
    split_rows = [r for r in read_csv(splits_path) if r["source_kind"] == "blank_background" and r["synthetic_split"] == split]
    selected = [blanks[r["stable_image_id"]] for r in sorted(split_rows, key=lambda r: r["stable_image_id"])]
    if len(selected) < count:
        raise ValueError(f"not enough blank images for split {split}: need {count}, found {len(selected)}")
    return selected[:count]


def generate_composites(config: dict[str, Any], inventory_dir: Path, splits_path: Path, out_dir: Path) -> None:
    status = config.get("calibration_data_status", "exploratory_unpartitioned")
    if status not in CALIBRATION_DATA_STATUSES or not status.startswith("exploratory"):
        raise ValueError("this exploratory compositor requires exploratory calibration_data_status")
    sample_count = int(config.get("compositing", {}).get("sample_count", 8))
    if not (8 <= sample_count <= 16):
        raise ValueError("sample_count must be 8-16 for bounded output")
    split = config.get("compositing", {}).get("synthetic_split", "calibration")
    source_roots = config["source_roots"]
    blank_rows = select_blank_rows(inventory_dir, splits_path, split, sample_count)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for index, blank_row in enumerate(blank_rows):
        sample_id, arrays, metadata = build_composite_sample(config, blank_row, source_roots, index, inventory_dir, splits_path)
        npz_path = out_dir / f"{sample_id}.npz"
        json_path = out_dir / f"{sample_id}.json"
        assert_no_object_arrays(arrays)
        np.savez_compressed(npz_path, **arrays)
        metadata["integrity_checksum"] = {"npz_sha256": sha256_file(npz_path)}
        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "npz_path": npz_path.name,
                "npz_sha256": sha256_file(npz_path),
                "json_path": json_path.name,
                "json_sha256": sha256_file(json_path),
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION,
            }
        )
    write_dataset_manifest(out_dir / "dataset_manifest.csv", manifest_rows)


def build_composite_sample(
    config: dict[str, Any],
    blank_row: dict[str, str],
    source_roots: dict[str, str],
    sample_index: int,
    inventory_dir: Path,
    splits_path: Path,
) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    compositor_config = config.get("compositing", {})
    geometry_config = config["geometry"].copy()
    rendering_config = config["real_blank_rendering"].copy()
    sample_id = f"{config.get('dataset_name', 'sted_blank_composite')}_{sample_index:04d}"
    blank_path = Path(source_roots[blank_row["source_root_id"]]) / blank_row["relative_path"]
    blank, _, _ = read_image(blank_path)
    blank_float = blank.astype(np.float32)
    geometry_config["image_shape"] = [int(blank_float.shape[0]), int(blank_float.shape[1])]
    geometry_seed = int(geometry_config.get("base_seed", 41001)) + sample_index
    rendering_seed = int(rendering_config.get("base_seed", 42001)) + sample_index
    compositing_seed = int(compositor_config.get("base_seed", 43001)) + sample_index
    geometry = generate_geometry(geometry_config, sample_index)
    arrays: dict[str, np.ndarray] = {}
    arrays.update(geometry_to_arrays(geometry))
    targets = rasterize_targets(geometry, config.get("targets", {}))
    arrays.update(targets)
    signal = targets["source_float"].astype(np.float32) * float(rendering_config.get("source_signal_scale", 1.0))
    sigma = float(rendering_config.get("psf_sigma_px", 1.2))
    if sigma > 0:
        signal = gaussian_blur(signal, sigma)
    perturb_scale = float(rendering_config.get("signal_dependent_perturbation_scale", 0.0))
    if perturb_scale > 0:
        rng = np.random.default_rng(compositing_seed)
        signal = signal + rng.normal(0, perturb_scale * np.sqrt(np.maximum(signal, 0)), signal.shape).astype(np.float32)
    signal = np.maximum(signal, 0).astype(np.float32)
    blank_scale = float(compositor_config.get("blank_scale", 1.0))
    composite = blank_float * blank_scale + signal
    composite_uint8, mapping_stats = map_to_uint8(composite, rendering_config)
    arrays["blank_float"] = blank_float.astype(np.float32)
    arrays["synthetic_signal_float"] = signal.astype(np.float32)
    arrays["composite_float"] = composite.astype(np.float32)
    arrays["render_float"] = composite.astype(np.float32)
    arrays["render_uint8"] = composite_uint8
    metadata = composite_metadata(
        sample_id,
        config,
        blank_row,
        geometry,
        arrays,
        geometry_seed,
        rendering_seed,
        compositing_seed,
        mapping_stats,
        inventory_dir,
        splits_path,
    )
    return sample_id, arrays, metadata


def composite_metadata(
    sample_id: str,
    config: dict[str, Any],
    blank_row: dict[str, str],
    geometry: dict[str, Any],
    arrays: dict[str, np.ndarray],
    geometry_seed: int,
    rendering_seed: int,
    compositing_seed: int,
    mapping_stats: dict[str, Any],
    inventory_dir: Path,
    splits_path: Path,
) -> dict[str, Any]:
    status = config.get("calibration_data_status", "exploratory_unpartitioned")
    metadata = artifact_metadata(
        inventory_dir=inventory_dir,
        splits_path=splits_path,
        config=config,
        source_image_ids=[blank_row["stable_image_id"]],
        calibration_data_status=status,
        artifact_kind="real_blank_composite_sample",
    )
    metadata.update(
        {
            "sample_id": sample_id,
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "geometry_seed": geometry_seed,
            "rendering_seed": rendering_seed,
            "compositing_seed": compositing_seed,
            "geometry_parameters": geometry["parameters"],
            "renderer_configuration": config.get("real_blank_rendering", {}),
            "compositor_configuration": config.get("compositing", {}),
            "source_blank_provenance": {
                "source_root_id": blank_row["source_root_id"],
                "blank_stable_image_id": blank_row["stable_image_id"],
                "blank_relative_path": blank_row["relative_path"],
                "blank_sha256": blank_row["source_sha256"],
                "blank_acquisition_group": blank_row["acquisition_group"],
                "crop_coordinates": [0, 0, int(arrays["blank_float"].shape[1]), int(arrays["blank_float"].shape[0])],
                "blank_status": blank_row.get("blank_status", "expert_validated"),
            },
            "synthetic_split": config.get("compositing", {}).get("synthetic_split", "calibration"),
            "float_to_uint8_mapping": mapping_stats,
            "clipping_count": int(mapping_stats["clipped_low_count"]) + int(mapping_stats["clipped_high_count"]),
            "saturation_count": int(mapping_stats["saturation_count"]),
            "array_names": sorted(arrays),
            "dtypes": {name: str(arr.dtype) for name, arr in arrays.items()},
            "shapes": {name: list(arr.shape) for name, arr in arrays.items()},
            "enum_mappings": {"node_type": NODE_TYPES},
            "alignment_statement": "structural targets are unchanged during real-blank compositing",
        }
    )
    return metadata


def validate_composites(dataset_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest = dataset_dir / "dataset_manifest.csv"
    if not manifest.exists():
        return [f"{manifest}: missing"]
    rows = read_csv(manifest)
    blank_by_split: dict[str, set[str]] = {}
    for row in rows:
        npz_path = dataset_dir / row["npz_path"]
        json_path = dataset_dir / row["json_path"]
        if sha256_file(npz_path) != row["npz_sha256"]:
            errors.append(f"{row['sample_id']}: NPZ checksum mismatch")
        if sha256_file(json_path) != row["json_sha256"]:
            errors.append(f"{row['sample_id']}: JSON checksum mismatch")
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if meta.get("calibration_status") not in {"exploratory_unpartitioned", "exploratory_provisional_split"}:
            errors.append(f"{row['sample_id']}: calibration_status must be exploratory")
        if meta.get("source_blank_provenance") == "not_applicable":
            errors.append(f"{row['sample_id']}: missing blank provenance")
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        for name, arr in arrays.items():
            if arr.dtype == object:
                errors.append(f"{row['sample_id']}: object dtype prohibited for {name}")
        if not np.array_equal(arrays["semantic_mask"], (arrays["overlap_count"] > 0).astype(np.uint8)):
            errors.append(f"{row['sample_id']}: target alignment failure")
        blank_id = meta["source_blank_provenance"]["blank_stable_image_id"]
        split = meta["synthetic_split"]
        for other_split, ids in blank_by_split.items():
            if other_split != split and blank_id in ids:
                errors.append(f"{blank_id}: blank reused across splits without override")
        blank_by_split.setdefault(split, set()).add(blank_id)
    return errors

