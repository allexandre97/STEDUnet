"""NPZ/JSON storage for MVP synthetic STED samples."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import generate_geometry, geometry_to_arrays
from .rendering import render_image
from .schema import CALIBRATION_STATUSES, DATASET_SCHEMA_VERSION, GENERATOR_VERSION, NODE_TYPES, REQUIRED_ARRAYS
from .targets import rasterize_targets


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_sample(config: dict[str, Any], sample_index: int) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    sample_id = f"{config.get('dataset_name', 'synthetic_sted')}_{sample_index:04d}"
    geometry_config = config.get("geometry", {})
    target_config = config.get("targets", {})
    rendering_config = config.get("rendering", {})
    geometry_seed = int(geometry_config.get("base_seed", 1234)) + sample_index
    rendering_seed = int(rendering_config.get("base_seed", 5678)) + sample_index
    geometry = generate_geometry(geometry_config, sample_index)
    arrays = {}
    arrays.update(geometry_to_arrays(geometry))
    targets = rasterize_targets(geometry, target_config)
    render_float, render_uint8, clip_stats = render_image(targets["source_float"], rendering_config, rendering_seed)
    arrays.update(targets)
    arrays["render_float"] = render_float.astype(np.float32)
    arrays["render_uint8"] = render_uint8
    metadata = metadata_for_sample(sample_id, config, sample_index, geometry, arrays, geometry_seed, rendering_seed, clip_stats)
    return sample_id, arrays, metadata


def metadata_for_sample(
    sample_id: str,
    config: dict[str, Any],
    sample_index: int,
    geometry: dict[str, Any],
    arrays: dict[str, np.ndarray],
    geometry_seed: int,
    rendering_seed: int,
    clip_stats: dict[str, Any],
) -> dict[str, Any]:
    calibration_status = config.get("calibration_status", "procedural_unmatched")
    if calibration_status not in CALIBRATION_STATUSES:
        raise ValueError(f"invalid calibration_status: {calibration_status}")
    return {
        "sample_id": sample_id,
        "sample_index": sample_index,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "geometry_seed": geometry_seed,
        "rendering_seed": rendering_seed,
        "generation_config": config,
        "geometry_parameters": geometry["parameters"],
        "appearance_parameters": config.get("rendering", {}),
        "split": config.get("split", "training"),
        "source_blank_provenance": "not_applicable",
        "image_shape": list(geometry["image_shape"]),
        "calibration_status": calibration_status,
        "array_names": sorted(arrays),
        "dtypes": {name: str(arr.dtype) for name, arr in arrays.items()},
        "shapes": {name: list(arr.shape) for name, arr in arrays.items()},
        "enum_mappings": {"node_type": NODE_TYPES},
        "coordinate_conventions": {
            "indexing": "zero_based",
            "units": "pixels",
            "xy_order": "x_y",
            "pixel_center": "pixel (x, y) is centered at (x + 0.5, y + 0.5)",
        },
        "tolerance_radii": {
            "centerline_radius_px": config.get("targets", {}).get("centerline_radius_px", 0.75),
            "endpoint_radius_px": config.get("targets", {}).get("endpoint_radius_px", 3.0),
            "junction_radius_px": config.get("targets", {}).get("junction_radius_px", 3.0),
            "crossing_radius_px": config.get("targets", {}).get("crossing_radius_px", 3.0),
            "psf_ambiguity_threshold": "not_implemented_in_mvp",
        },
        "rasterization_rules": {
            "semantic_mask": "union of all clean finite-width instance supports before PSF blur",
            "overlap_count": "number of instance memberships per pixel before PSF blur",
            "crossing_map": "raster disks around disconnected centerline intersections",
            "junction_map": "raster disks around true graph junction nodes",
            "canonical_instance_membership": "sparse pixel-to-instance coordinate table",
        },
        "rendering_report": clip_stats,
        "integrity_checksum": "npz_sha256_recorded_after_save",
    }


def save_dataset(config: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = int(config.get("sample_count", 8))
    if not (8 <= count <= 16):
        raise ValueError("MVP sample_count must be between 8 and 16")
    rows: list[dict[str, str]] = []
    for index in range(count):
        sample_id, arrays, metadata = build_sample(config, index)
        npz_path = out_dir / f"{sample_id}.npz"
        json_path = out_dir / f"{sample_id}.json"
        assert_no_object_arrays(arrays)
        np.savez_compressed(npz_path, **arrays)
        metadata["integrity_checksum"] = {"npz_sha256": sha256_file(npz_path)}
        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "sample_id": sample_id,
                "npz_path": npz_path.name,
                "npz_sha256": sha256_file(npz_path),
                "json_path": json_path.name,
                "json_sha256": sha256_file(json_path),
                "schema_version": DATASET_SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION,
            }
        )
    write_dataset_manifest(out_dir / "dataset_manifest.csv", rows)


def assert_no_object_arrays(arrays: dict[str, np.ndarray]) -> None:
    missing = REQUIRED_ARRAYS - set(arrays)
    if missing:
        raise ValueError(f"missing required arrays: {sorted(missing)}")
    for name, arr in arrays.items():
        if arr.dtype == object:
            raise ValueError(f"{name}: object arrays are prohibited")


def write_dataset_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["sample_id", "npz_path", "npz_sha256", "json_path", "json_sha256", "schema_version", "generator_version"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_dataset(out_dir: Path, deterministic: bool = True) -> list[str]:
    manifest = out_dir / "dataset_manifest.csv"
    if not manifest.exists():
        return [f"{manifest}: missing dataset manifest"]
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    errors: list[str] = []
    for row in rows:
        npz_path = out_dir / row["npz_path"]
        json_path = out_dir / row["json_path"]
        if sha256_file(npz_path) != row["npz_sha256"]:
            errors.append(f"{row['sample_id']}: NPZ checksum mismatch")
        if sha256_file(json_path) != row["json_sha256"]:
            errors.append(f"{row['sample_id']}: JSON checksum mismatch")
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        errors.extend(validate_sample_arrays(row["sample_id"], arrays, metadata))
        if deterministic:
            _, rebuilt, _ = build_sample(metadata["generation_config"], int(metadata["sample_index"]))
            for name in REQUIRED_ARRAYS:
                if not np.array_equal(arrays[name], rebuilt[name]):
                    errors.append(f"{row['sample_id']}: deterministic regeneration mismatch for {name}")
                    break
    return errors


def validate_sample_arrays(sample_id: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_ARRAYS - set(arrays)
    if missing:
        errors.append(f"{sample_id}: missing arrays {sorted(missing)}")
    for name, arr in arrays.items():
        if arr.dtype == object:
            errors.append(f"{sample_id}: {name} has object dtype")
    if metadata.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        errors.append(f"{sample_id}: invalid schema version")
    if metadata.get("generator_version") != GENERATOR_VERSION:
        errors.append(f"{sample_id}: invalid generator version")
    if metadata.get("calibration_status") != "procedural_unmatched":
        errors.append(f"{sample_id}: MVP examples must be procedural_unmatched")
    if metadata.get("source_blank_provenance") != "not_applicable":
        errors.append(f"{sample_id}: real blank compositing is deferred")
    if not missing:
        overlap = arrays["overlap_count"]
        expected = np.zeros_like(overlap, dtype=np.uint16)
        y = arrays["membership_y"]
        x = arrays["membership_x"]
        np.add.at(expected, (y, x), 1)
        if not np.array_equal(overlap, expected):
            errors.append(f"{sample_id}: overlap_count does not match sparse memberships")
        if not np.array_equal(arrays["semantic_mask"], (overlap > 0).astype(np.uint8)):
            errors.append(f"{sample_id}: semantic mask is not membership union")
        if arrays["geometric_crossing_points_xy"].shape[0] == 0:
            errors.append(f"{sample_id}: no apparent crossing points")
        if int(np.sum(arrays["junction_map"])) == 0:
            errors.append(f"{sample_id}: no true junction map pixels")
        if int(np.sum(arrays["endpoint_map"])) == 0:
            errors.append(f"{sample_id}: no endpoint map pixels")
        if int(np.sum(arrays["edge_truncated_start"]) + np.sum(arrays["edge_truncated_end"])) == 0:
            errors.append(f"{sample_id}: no truncated endpoint metadata")
    return errors

