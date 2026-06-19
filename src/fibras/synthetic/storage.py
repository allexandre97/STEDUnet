"""NPZ/JSON storage for MVP synthetic STED samples."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import generate_geometry, geometry_to_arrays
from .geometry3d import generate_persistent_chain_geometry, geometry3d_to_arrays
from .rendering import render_image
from .rasterizer3d import measure_isolated_fiber_fwhm, rasterize_3d_sample
from .schema import (
    CALIBRATION_STATUSES,
    DATASET_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION_3D,
    DATASET_SCHEMA_VERSION_3D_HARDENED,
    DATASET_SCHEMA_VERSION_3D_LEGACY,
    DATASET_SCHEMA_VERSION_3D_NORMALIZED,
    GENERATOR_MODES,
    GENERATOR_VERSION,
    GENERATOR_VERSION_3D,
    NODE_TYPES,
    REAL_SEMANTIC_CLASSES,
    REQUIRED_ARRAYS,
    REQUIRED_ARRAYS_BY_SCHEMA,
    TRACE_TERMINATION_STATUSES,
)
from .targets import rasterize_targets


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_sample(config: dict[str, Any], sample_index: int) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    sample_id = f"{config.get('dataset_name', 'synthetic_sted')}_{sample_index:04d}"
    mode = generator_mode(config)
    if mode == "persistent_chain_3d":
        return build_sample_3d(config, sample_index, sample_id)
    if mode not in {"structural_test_2d", "legacy_2d"}:
        raise ValueError(f"unsupported generator_mode: {mode}")
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


def generator_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("generator_mode", "structural_test_2d"))
    if mode not in GENERATOR_MODES:
        raise ValueError(f"invalid generator_mode: {mode}")
    return mode


def build_sample_3d(config: dict[str, Any], sample_index: int, sample_id: str) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    geometry_config = config.get("geometry", {})
    rendering_seed = int(config.get("output_mapping", {}).get("base_seed", 62001)) + sample_index
    geometry_seed = int(geometry_config.get("base_seed", 61001)) + sample_index
    geometry = generate_persistent_chain_geometry(geometry_config, sample_index)
    arrays = {}
    arrays.update(geometry3d_to_arrays(geometry))
    raster_arrays, render_report = rasterize_3d_sample(
        geometry,
        config.get("targets", {}),
        config.get("optical_model", {}),
        config.get("output_mapping", {}),
        rendering_seed,
    )
    arrays.update(raster_arrays)
    metadata = metadata_for_sample_3d(sample_id, config, sample_index, geometry, arrays, geometry_seed, rendering_seed, render_report)
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


def metadata_for_sample_3d(
    sample_id: str,
    config: dict[str, Any],
    sample_index: int,
    geometry: dict[str, Any],
    arrays: dict[str, np.ndarray],
    geometry_seed: int,
    rendering_seed: int,
    render_report: dict[str, Any],
) -> dict[str, Any]:
    calibration_status = config.get("calibration_status", "procedural_unmatched")
    if calibration_status not in CALIBRATION_STATUSES:
        raise ValueError(f"invalid calibration_status: {calibration_status}")
    width_report = measure_isolated_fiber_fwhm(config)
    targets_config = config.get("targets", {})
    return {
        "sample_id": sample_id,
        "sample_index": sample_index,
        "dataset_schema_version": DATASET_SCHEMA_VERSION_3D,
        "schema_migration": "0.5.0 adds projected_crossing_fiber_ids and projected_crossing_segment_indices; 0.4.0 source, distance, orientation, membership, and PSF conventions are unchanged",
        "generator_version": GENERATOR_VERSION_3D,
        "generator_mode": "persistent_chain_3d",
        "geometry_seed": geometry_seed,
        "rendering_seed": rendering_seed,
        "generation_config": config,
        "geometry_parameters": geometry["parameters"],
        "scenario": render_report.get("scenario", geometry["parameters"].get("scenario", "random_persistent_chain")),
        "scenario_category": render_report.get("scenario_category", "realism_calibration"),
        "appearance_parameters": {
            "optical_model": config.get("optical_model", {}),
            "output_mapping": config.get("output_mapping", {}),
            "fluorophore_density_per_length": "stored in fluorophore_density_per_length and legacy fiber_sample_amplitude arrays",
        },
        "split": config.get("split", "training"),
        "source_blank_provenance": "not_applicable",
        "image_shape": list(geometry["image_shape"]),
        "volume_depth_px": float(geometry["volume_depth_px"]),
        "focal_plane_z_px": float(geometry["focal_plane_z_px"]),
        "calibration_status": calibration_status,
        "array_names": sorted(arrays),
        "dtypes": {name: str(arr.dtype) for name, arr in arrays.items()},
        "shapes": {name: list(arr.shape) for name, arr in arrays.items()},
        "enum_mappings": {"node_type": NODE_TYPES},
        "coordinate_conventions": {
            "indexing": "zero_based",
            "units": "pixel_equivalent",
            "axis_order": "x_y_z",
            "xy_order": "x_y",
            "origin": "x/y origin at upper-left image corner; z=0 is the near axial bound",
            "pixel_center": "pixel (x, y) is centered at (x + 0.5, y + 0.5)",
        },
        "mask_semantics": {
            "projection_mask": "finite-width projected fiber support from all depths before optical PSF",
            "in_focus_mask": "finite-width support from curve samples within configured focal_depth_range_px",
            "visible_signal_mask": "clean rendered signal above visible_signal_threshold after depth-dependent splatting",
            "semantic_mask": f"copy of {render_report['semantic_mask_source']} for this configuration",
            "centerline_mask": "projected centerline support compatible with 2D vector-trace annotations",
            "ignore_mask": f"simulator-defined ambiguity mask from rule `{render_report.get('ignore_mask_rule', 'none')}`; not equivalent to human uncertainty",
            "background_distance_to_semantic_foreground": render_report.get("distance_target_semantics", {}),
        },
        "depth_target_semantics": {
            "nearest_depth_map": "z of contributing curve sample with smallest absolute distance to focal plane; -1 means no contribution",
            "weighted_mean_depth_map": "absolute z depth, signal-weighted across clean optical contributions; -1 means no contribution",
        },
        "target_available": render_report.get("target_available", {}),
        "target_provenance": {
            "semantic_mask_source": render_report["semantic_mask_source"],
            "centerline_source": "projected_3d_ground_truth",
            "trace_source": "projected_3d_ground_truth",
            "ignore_mask_rule": render_report.get("ignore_mask_rule", "none"),
        },
        "annotation_contract": {
            "compatible_real_annotation_sources": ["JFilament_vector_centerlines", "Labkit_or_brush_semantic_masks"],
            "trace_arrays": ["trace_points_xy", "trace_point_offsets", "trace_ids", "trace_status", "trace_source"],
            "trace_status_codes": {"valid": 1, "boundary_truncated": 2},
            "trace_source_codes": {"projected_3d_ground_truth": 1},
            "coordinate_convention": "zero-based x_y pixel-equivalent coordinates with pixel-center convention matching synthetic arrays",
            "synthetic_only_targets": ["fiber_points_xyz", "nearest_depth_map", "weighted_mean_depth_map", "in_focus_signal", "out_of_focus_signal", "total_optical_signal"],
            "reserved_real_semantic_classes": REAL_SEMANTIC_CLASSES,
            "reserved_trace_termination_statuses": sorted(TRACE_TERMINATION_STATUSES),
            "current_binary_semantic_mask": True,
            "bundle_and_clump_generation": "not_implemented",
        },
        "patch_provenance_schema": {
            "parent_sample_id": "not_applicable_for_full_image",
            "crop_origin_xy": "not_applicable_for_full_image",
            "crop_shape": "not_applicable_for_full_image",
            "clipped_trace_ids": "not_applicable_for_full_image",
            "boundary_truncated_endpoint_rule": "patch-clipped trace endpoints are not biological endpoints",
        },
        "tolerance_radii": {
            "centerline_radius_px": config.get("targets", {}).get("centerline_radius_px", 0.75),
            "endpoint_radius_px": config.get("targets", {}).get("endpoint_radius_px", 3.0),
            "junction_radius_px": config.get("targets", {}).get("junction_radius_px", 3.0),
            "crossing_radius_px": config.get("targets", {}).get("crossing_radius_px", 3.0),
            "near_coplanar_depth_px": config.get("targets", {}).get("near_coplanar_depth_px", 8.0),
        },
        "rasterization_rules": {
            "geometric_overlap": "overlap_count counts projected finite-width instance memberships before PSF blur",
            "visible_optical_overlap": "visible_overlap_count counts instances whose clean rendered signal exceeds threshold at each pixel",
            "combined_only_visible": "combined_only_visible_mask marks pixels whose summed signal is visible while all individual instance signals are subthreshold",
            "contributing_instance_membership": "sparse table of all instances with individual signal above contributing_signal_threshold",
            "projected_crossing": "2D projected intersection of disconnected 3D centerlines",
            "near_coplanar_crossing": "projected crossing whose interpolated z separation is within near_coplanar_depth_px",
            "true_junction": "3D graph node shared by connected edges; never inferred from 2D projection",
        },
        "source_raster_semantics": render_report.get("source_raster_semantics", {}),
        "rendering_report": render_report,
        "width_calibration": {
            **width_report,
            "measurement_method": "bilinear transverse profile through isolated straight in-focus fiber; FWHM measured at half peak above local baseline",
            "geometric_fiber_radius_px": config.get("fluorophore", config.get("geometry", {}).get("fluorophore", {})).get("radius_range_px", "configured_range"),
            "in_focus_psf_sigma_px": config.get("optical_model", {}).get("sigma_xy_0_px", config.get("optical_model", {}).get("core_sigma_xy_0_px", "not_reported")),
            "foreground_signal_used": "total_clean_signal before uint8 clipping",
        },
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
                "schema_version": metadata["dataset_schema_version"],
                "generator_version": metadata["generator_version"],
            }
        )
    write_dataset_manifest(out_dir / "dataset_manifest.csv", rows)


def assert_no_object_arrays(arrays: dict[str, np.ndarray]) -> None:
    for name, arr in arrays.items():
        if arr.dtype == object:
            raise ValueError(f"{name}: object arrays are prohibited")


def write_dataset_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["sample_id", "npz_path", "npz_sha256", "json_path", "json_sha256", "schema_version", "generator_version"]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate sample_id values are prohibited")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
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
        if deterministic and metadata.get("dataset_schema_version") == DATASET_SCHEMA_VERSION_3D:
            _, rebuilt, rebuilt_metadata = build_sample(
                metadata["generation_config"], int(metadata["sample_index"])
            )
            errors.extend(
                compare_complete_artifact(
                    row["sample_id"], arrays, metadata, rebuilt, rebuilt_metadata
                )
            )
    return errors


def compare_complete_artifact(
    sample_id: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    rebuilt: dict[str, np.ndarray],
    rebuilt_metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    stored_names = set(arrays)
    rebuilt_names = set(rebuilt)
    if stored_names != rebuilt_names:
        missing = sorted(rebuilt_names - stored_names)
        unexpected = sorted(stored_names - rebuilt_names)
        errors.append(
            f"{sample_id}: deterministic array-name mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(stored_names & rebuilt_names):
        stored = arrays[name]
        expected = rebuilt[name]
        if stored.dtype != expected.dtype:
            errors.append(
                f"{sample_id}: deterministic dtype mismatch for {name}: "
                f"{stored.dtype} != {expected.dtype}"
            )
        elif stored.shape != expected.shape:
            errors.append(
                f"{sample_id}: deterministic shape mismatch for {name}: "
                f"{stored.shape} != {expected.shape}"
            )
        elif not np.array_equal(stored, expected):
            errors.append(f"{sample_id}: deterministic value mismatch for {name}")
    if set(metadata.get("array_names", [])) != stored_names:
        errors.append(f"{sample_id}: metadata array_names do not match stored arrays")
    expected_dtypes = {name: str(array.dtype) for name, array in arrays.items()}
    expected_shapes = {name: list(array.shape) for name, array in arrays.items()}
    if metadata.get("dtypes") != expected_dtypes:
        errors.append(f"{sample_id}: metadata dtypes do not match stored arrays")
    if metadata.get("shapes") != expected_shapes:
        errors.append(f"{sample_id}: metadata shapes do not match stored arrays")
    for key in ["target_available", "scenario", "scenario_category"]:
        if metadata.get(key) != rebuilt_metadata.get(key):
            errors.append(f"{sample_id}: deterministic metadata mismatch for {key}")
    for key in [
        "semantic_mask_source",
        "ignore_mask_rule",
        "psf_normalization",
        "psf_component_weight_convention",
        "distance_transform_backend",
    ]:
        stored_value = metadata.get("rendering_report", {}).get(key)
        rebuilt_value = rebuilt_metadata.get("rendering_report", {}).get(key)
        if stored_value != rebuilt_value:
            errors.append(
                f"{sample_id}: deterministic rendering metadata mismatch for {key}"
            )
    return errors


def validate_sample_arrays(sample_id: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    schema_version = metadata.get("dataset_schema_version")
    required = REQUIRED_ARRAYS_BY_SCHEMA.get(schema_version)
    if required is None:
        return [f"{sample_id}: unsupported schema version {schema_version}; explicit migration required"]
    missing = required - set(arrays)
    if missing:
        errors.append(f"{sample_id}: missing arrays {sorted(missing)}")
    for name, arr in arrays.items():
        if arr.dtype == object:
            errors.append(f"{sample_id}: {name} has object dtype")
    if schema_version == DATASET_SCHEMA_VERSION and metadata.get("generator_version") != GENERATOR_VERSION:
        errors.append(f"{sample_id}: invalid generator version")
    if schema_version == DATASET_SCHEMA_VERSION_3D and metadata.get("generator_version") != GENERATOR_VERSION_3D:
        errors.append(f"{sample_id}: invalid 3D generator version")
    if metadata.get("calibration_status") != "procedural_unmatched":
        errors.append(f"{sample_id}: MVP examples must be procedural_unmatched")
    if metadata.get("source_blank_provenance") != "not_applicable":
        errors.append(f"{sample_id}: real blank compositing is deferred")
    if not missing and schema_version == DATASET_SCHEMA_VERSION:
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
    if not missing and schema_version in {
        DATASET_SCHEMA_VERSION_3D,
        DATASET_SCHEMA_VERSION_3D_HARDENED,
        DATASET_SCHEMA_VERSION_3D_NORMALIZED,
        DATASET_SCHEMA_VERSION_3D_LEGACY,
    }:
        errors.extend(validate_3d_arrays(sample_id, arrays, metadata))
    return errors


def validate_3d_arrays(sample_id: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    schema_version = metadata.get("dataset_schema_version")
    is_normalized_schema = schema_version in {
        DATASET_SCHEMA_VERSION_3D,
        DATASET_SCHEMA_VERSION_3D_HARDENED,
        DATASET_SCHEMA_VERSION_3D_NORMALIZED,
    }
    is_hardened_schema = schema_version in {
        DATASET_SCHEMA_VERSION_3D,
        DATASET_SCHEMA_VERSION_3D_HARDENED,
    }
    if arrays["fiber_points_xyz"].shape[1] != 3:
        errors.append(f"{sample_id}: fiber_points_xyz must have xyz columns")
    if arrays["fiber_points_xy"].shape[0] != arrays["fiber_points_xyz"].shape[0]:
        errors.append(f"{sample_id}: xy and xyz point counts differ")
    if not np.allclose(arrays["fiber_points_xy"], arrays["fiber_points_xyz"][:, :2]):
        errors.append(f"{sample_id}: fiber_points_xy is not the projection of fiber_points_xyz")
    if arrays["fiber_sample_amplitude"].shape[0] != arrays["fiber_points_xyz"].shape[0]:
        errors.append(f"{sample_id}: sample amplitude count mismatch")
    if arrays["fiber_sample_radius"].shape[0] != arrays["fiber_points_xyz"].shape[0]:
        errors.append(f"{sample_id}: sample radius count mismatch")
    if np.any(~np.isfinite(arrays["fiber_points_xyz"])):
        errors.append(f"{sample_id}: invalid 3D coordinates")
    if not np.array_equal(arrays["projection_mask"] | arrays["in_focus_mask"], arrays["projection_mask"]):
        errors.append(f"{sample_id}: projection_mask must contain in_focus_mask")
    if not np.array_equal(arrays["visible_signal_mask"], (arrays["total_clean_signal"] > metadata["rendering_report"]["visible_signal_threshold"]).astype(np.uint8)):
        errors.append(f"{sample_id}: visible_signal_mask threshold mismatch")
    if not np.allclose(arrays["in_focus_signal"] + arrays["out_of_focus_signal"], arrays["total_clean_signal"], atol=1e-4):
        errors.append(f"{sample_id}: in-focus plus out-of-focus signal mismatch")
    expected = np.zeros_like(arrays["overlap_count"], dtype=np.uint16)
    np.add.at(expected, (arrays["membership_y"], arrays["membership_x"]), 1)
    if not np.array_equal(arrays["overlap_count"], expected):
        errors.append(f"{sample_id}: overlap_count does not match sparse memberships")
    semantic_source = metadata["rendering_report"]["semantic_mask_source"]
    if not np.array_equal(arrays["semantic_mask"], arrays[semantic_source].astype(np.uint8)):
        errors.append(f"{sample_id}: semantic_mask source mismatch")
    mask = arrays["total_clean_signal"] > 0
    if np.any(arrays["nearest_depth_map"][mask] < 0) or np.any(arrays["weighted_mean_depth_map"][mask] < 0):
        errors.append(f"{sample_id}: depth maps missing for contributing pixels")
    fwhm = metadata["width_calibration"]["measured_fwhm_px"]
    lo = metadata["width_calibration"]["min_allowed_fwhm_px"]
    hi = metadata["width_calibration"]["max_allowed_fwhm_px"]
    if not (lo <= fwhm <= hi):
        errors.append(f"{sample_id}: width calibration FWHM {fwhm:.3f} outside [{lo}, {hi}]")
    if is_normalized_schema:
        if not np.array_equal(arrays["trace_points_xy"], arrays["fiber_points_xy"]):
            errors.append(f"{sample_id}: trace_points_xy must match projected fiber_points_xy")
        if not np.array_equal(arrays["trace_point_offsets"], arrays["fiber_point_offsets"]):
            errors.append(f"{sample_id}: trace offsets must match fiber point offsets")
        if arrays["trace_point_offsets"].ndim != 1:
            errors.append(f"{sample_id}: trace_point_offsets must be one-dimensional")
        elif (
            arrays["trace_point_offsets"].size == 0
            or int(arrays["trace_point_offsets"][0]) != 0
            or np.any(np.diff(arrays["trace_point_offsets"]) < 0)
            or int(arrays["trace_point_offsets"][-1])
            != int(arrays["trace_points_xy"].shape[0])
        ):
            errors.append(f"{sample_id}: malformed trace_point_offsets")
        if arrays["trace_ids"].shape[0] != arrays["trace_point_offsets"].shape[0] - 1:
            errors.append(f"{sample_id}: trace_ids count does not match trace offsets")
        if arrays["trace_status"].shape != arrays["trace_ids"].shape:
            errors.append(f"{sample_id}: trace_status shape mismatch")
        if arrays["trace_source"].shape != arrays["trace_ids"].shape:
            errors.append(f"{sample_id}: trace_source shape mismatch")
        if arrays["sample_arc_length_weight"].shape[0] != arrays["fiber_points_xyz"].shape[0]:
            errors.append(f"{sample_id}: sample_arc_length_weight count mismatch")
        if arrays["fluorophore_density_per_length"].shape[0] != arrays["fiber_points_xyz"].shape[0]:
            errors.append(f"{sample_id}: fluorophore density count mismatch")
        if np.any(arrays["sample_arc_length_weight"] < 0):
            errors.append(f"{sample_id}: negative arc-length weights")
        if not np.allclose(arrays["fiber_sample_amplitude"], arrays["fluorophore_density_per_length"]):
            errors.append(f"{sample_id}: legacy fiber_sample_amplitude must match density-per-length alias")
        if metadata["rendering_report"].get("psf_normalization") != "unit_integral":
            errors.append(f"{sample_id}: PSF normalization must be unit_integral")
        if metadata["rendering_report"].get("sample_arc_length_weighting") is not True:
            errors.append(f"{sample_id}: sample arc-length weighting metadata missing")
        err = metadata["rendering_report"].get("kernel_normalization_error")
        if isinstance(err, (int, float)) and float(err) > 1e-5:
            errors.append(f"{sample_id}: kernel normalization error {err} exceeds tolerance")
        available = metadata.get("target_available", {})
        if arrays["ignore_mask"].shape != arrays["semantic_mask"].shape:
            errors.append(f"{sample_id}: ignore_mask shape mismatch")
        if not np.allclose(arrays["core_signal"] + arrays["halo_signal"], arrays["total_clean_signal"], atol=1e-4):
            errors.append(f"{sample_id}: core plus halo signal mismatch")
        if is_hardened_schema:
            required_targets = {
                "semantic_mask",
                "centerline_mask",
                "ignore_mask",
                "background_distance_to_semantic_foreground",
                "endpoint_map",
                "junction_map",
                "crossing_map",
                "orientation",
                "instance_membership",
                "visible_instance_membership",
                "contributing_instance_membership",
                "depth_maps",
                "optical_signal_decomposition",
            }
            missing_targets = required_targets - set(available)
            if missing_targets:
                errors.append(f"{sample_id}: missing target_available keys {sorted(missing_targets)}")
            if available.get("orientation"):
                for name in ["orientation_cos2theta", "orientation_sin2theta", "orientation_valid_mask", "orientation_instance_count"]:
                    if name not in arrays:
                        errors.append(f"{sample_id}: target_available orientation=true but {name} missing")
                if "orientation_valid_mask" in arrays and arrays["orientation_valid_mask"].shape != arrays["semantic_mask"].shape:
                    errors.append(f"{sample_id}: orientation_valid_mask shape mismatch")
            else:
                forbidden = {"orientation_cos2theta", "orientation_sin2theta", "orientation_valid_mask", "orientation_instance_count"} & set(arrays)
                if forbidden:
                    errors.append(f"{sample_id}: orientation arrays present while orientation target is disabled: {sorted(forbidden)}")
            dist_name = "background_distance_to_semantic_foreground"
            if available.get(dist_name):
                if dist_name not in arrays:
                    errors.append(f"{sample_id}: distance target enabled but {dist_name} missing")
                else:
                    if not np.all(arrays[dist_name][arrays["semantic_mask"].astype(bool)] == 0):
                        errors.append(f"{sample_id}: distance target must be zero inside semantic foreground")
            elif dist_name in arrays:
                errors.append(f"{sample_id}: distance target present while disabled")
            if "source_float" in arrays:
                errors.append(f"{sample_id}: source_float is obsolete for 0.4.0; use line_source_float and geometric_support_preview")
            source_integral = float(np.sum(arrays["line_source_float"]))
            emitted = float(metadata["rendering_report"]["total_emitted_source_signal"])
            if not np.isclose(source_integral, emitted, rtol=1e-5, atol=1e-3):
                errors.append(f"{sample_id}: line_source_float integral {source_integral:.6g} does not match emitted signal {emitted:.6g}")
            visible_expected = np.zeros_like(arrays["visible_overlap_count"], dtype=np.uint16)
            np.add.at(visible_expected, (arrays["visible_instance_membership_y"], arrays["visible_instance_membership_x"]), 1)
            if not np.array_equal(visible_expected, arrays["visible_overlap_count"]):
                errors.append(f"{sample_id}: visible instance membership does not match visible_overlap_count")
            contributing_expected = np.zeros_like(arrays["contributing_overlap_count"], dtype=np.uint16)
            np.add.at(contributing_expected, (arrays["contributing_membership_y"], arrays["contributing_membership_x"]), 1)
            if not np.array_equal(contributing_expected, arrays["contributing_overlap_count"]):
                errors.append(f"{sample_id}: contributing membership does not match contributing_overlap_count")
            visible_classified = (arrays["visible_overlap_count"] > 0) | (arrays["combined_only_visible_mask"] > 0)
            if not np.array_equal(visible_classified.astype(np.uint8), arrays["visible_signal_mask"]):
                errors.append(f"{sample_id}: visible pixels are not fully classified by instance or combined-only masks")
            retained = metadata["rendering_report"].get("kernel_in_frame_retained_sum", {})
            if isinstance(retained, dict) and float(retained.get("max", 0.0)) > 1.000001:
                errors.append(f"{sample_id}: retained kernel sum cannot exceed one")
            if metadata["rendering_report"].get("scenario_category") not in {"structural_qa", "optical_qa", "realism_calibration"}:
                errors.append(f"{sample_id}: invalid scenario_category")
            expected_optional = {
                "orientation": {
                    "orientation_cos2theta",
                    "orientation_sin2theta",
                    "orientation_valid_mask",
                    "orientation_instance_count",
                },
                "background_distance_to_semantic_foreground": {
                    "background_distance_to_semantic_foreground"
                },
            }
            for target, names in expected_optional.items():
                present = names <= set(arrays)
                if bool(available.get(target)) != present:
                    errors.append(
                        f"{sample_id}: target_available[{target}] contradicts stored arrays"
                    )
        if schema_version == DATASET_SCHEMA_VERSION_3D_NORMALIZED:
            for old_name in [
                "source_float",
                "distance_transform",
                "orientation_cos2",
                "orientation_sin2",
                "visible_membership_y",
                "visible_membership_x",
                "visible_membership_instance_id",
            ]:
                if old_name not in arrays:
                    errors.append(
                        f"{sample_id}: schema 0.3 requires legacy field {old_name}; "
                        "use an explicit migration rather than reinterpretation"
                    )
        if schema_version == DATASET_SCHEMA_VERSION_3D:
            report = metadata["rendering_report"]
            if report.get("distance_transform_backend") not in {
                "scipy_ndimage_distance_transform_edt",
                "disabled",
            }:
                errors.append(f"{sample_id}: invalid distance_transform_backend")
            if report.get("psf_component_weight_convention") != "weights_sum_to_one":
                errors.append(f"{sample_id}: missing PSF component-weight convention")
            weight_sum = report.get("psf_component_weight_sum")
            if not isinstance(weight_sum, (int, float)) or not np.isclose(
                float(weight_sum), 1.0, rtol=0.0, atol=1e-9
            ):
                errors.append(f"{sample_id}: invalid PSF component weight sum")
            crossing_count = arrays["projected_crossing_points_xy"].shape[0]
            if arrays["projected_crossing_fiber_ids"].shape != (crossing_count, 2):
                errors.append(f"{sample_id}: projected crossing fiber-ID shape mismatch")
            if arrays["projected_crossing_segment_indices"].shape != (
                crossing_count,
                2,
            ):
                errors.append(
                    f"{sample_id}: projected crossing segment-index shape mismatch"
                )
    return errors
