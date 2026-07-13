"""NPZ/JSON storage for MVP synthetic STED samples."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from .geometry import generate_geometry, geometry_to_arrays
from .geometry3d import generate_persistent_chain_geometry, geometry3d_to_arrays
from .morphology3d import (
    generate_morphology_geometry,
    morphology_geometry_to_arrays,
)
from .rendering import render_image
from .rasterizer3d import measure_isolated_fiber_fwhm, rasterize_3d_sample
from .real_compatible import (
    REAL_COMPATIBLE_SUPERVISED_TARGETS,
    SYNTHETIC_ONLY_NOT_REAL_SUPERVISED,
    build_real_compatible_targets,
    validate_real_compatible_targets,
)
from .schema import (
    BOUNDARY_CODES,
    CALIBRATION_STATUSES,
    DATASET_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION_3D,
    DATASET_SCHEMA_VERSION_3D_HARDENED,
    DATASET_SCHEMA_VERSION_3D_LEGACY,
    DATASET_SCHEMA_VERSION_3D_NORMALIZED,
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_0_7,
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY,
    FIBER_STRUCTURE_TYPE_CODES,
    GENERATOR_MODES,
    GENERATOR_VERSION,
    GENERATOR_VERSION_3D,
    GENERATOR_VERSION_3D_MORPHOLOGY,
    GENERATOR_VERSION_3D_MORPHOLOGY_0_7,
    GENERATOR_VERSION_3D_MORPHOLOGY_LEGACY,
    NODE_TYPES,
    REAL_SEMANTIC_CLASSES,
    REQUIRED_ARRAYS,
    REQUIRED_ARRAYS_BY_SCHEMA,
    TRACE_TERMINATION_STATUSES,
    TRACE_TERMINATION_STATUS_CODES,
)
from .targets import draw_disk, rasterize_targets


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def set_worker_thread_limits() -> None:
    for name in THREAD_ENV_VARS:
        os.environ.setdefault(name, "1")


def build_sample(config: dict[str, Any], sample_index: int) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    sample_id = f"{config.get('dataset_name', 'synthetic_sted')}_{sample_index:04d}"
    config = sample_config(config, sample_index)
    config["dataset_name"] = sample_id.rsplit("_", 1)[0]
    mode = generator_mode(config)
    if mode == "persistent_chain_3d":
        return build_sample_3d(config, sample_index, sample_id)
    if mode == "morphology_scene_3d":
        return build_sample_morphology(config, sample_index, sample_id)
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


def sample_config(config: dict[str, Any], sample_index: int) -> dict[str, Any]:
    variants = config.get("sample_variants")
    if not variants:
        return dict(config)
    sequence = variants.get("sequence", [])
    definitions = variants.get("definitions", {})
    if not sequence:
        raise ValueError("sample_variants.sequence must not be empty")
    name = str(sequence[sample_index % len(sequence)])
    if name not in definitions:
        raise ValueError(f"sample_variants definition not found: {name}")
    merged = deep_merge_dicts(
        {key: value for key, value in config.items() if key != "sample_variants"},
        definitions[name],
    )
    merged["sample_variant"] = name
    merged["sample_variant_sequence_period"] = len(sequence)
    return merged


def deep_merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


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


def build_sample_morphology(
    config: dict[str, Any], sample_index: int, sample_id: str
) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    geometry_config = config["geometry"]
    rendering_seed = (
        int(config.get("output_mapping", {}).get("base_seed", 82001))
        + sample_index
    )
    geometry_seed = int(geometry_config.get("base_seed", 81001)) + sample_index
    geometry = generate_morphology_geometry(config, sample_index)
    arrays = morphology_geometry_to_arrays(
        geometry, geometry3d_to_arrays(geometry)
    )
    raster_arrays, render_report = rasterize_3d_sample(
        geometry,
        config.get("targets", {}),
        config.get("optical_model", {}),
        config.get("output_mapping", {}),
        rendering_seed,
    )
    arrays.update(raster_arrays)
    arrays.update(build_real_compatible_targets(arrays))
    metadata = metadata_for_sample_3d(
        sample_id,
        config,
        sample_index,
        geometry,
        arrays,
        geometry_seed,
        rendering_seed,
        render_report,
        schema_version=DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
        generator_version=GENERATOR_VERSION_3D_MORPHOLOGY,
        mode="morphology_scene_3d",
    )
    metadata["schema_migration"] = (
        "0.8.0 separates latent source-support masks from apparent visible "
        "supervised semantic masks derived from class-attributed optical signal; "
        "0.7.0 remains readable under its original source-support class-mask "
        "semantics and is not silently reinterpreted. 0.7.0 replaces ambiguous "
        "schema-0.6 generic memberships with explicit "
        "supervised and latent geometry arrays, adds graph/boundary supervision "
        "flags and class-attributed optical signals; schema 0.6 remains "
        "supported under its original field meanings and is not reinterpreted"
    )
    metadata["annotation_contract"].update(
        {
            "current_binary_semantic_mask": False,
            "multiclass_semantic_mask": "semantic_class_mask",
            "bundle_and_clump_generation": "implemented_as_exploratory_synthetic_morphology",
            "centerline_supervision": "individual_filament class only; real-compatible skeleton additionally exposes bundle axes",
            "clump_centerline_target": "not_available",
            "trace_status_codes": TRACE_TERMINATION_STATUS_CODES,
            "loss_eligible_arrays": [
                "semantic_class_mask",
                "semantic_mask",
                "individual_filament_mask",
                "bundle_mask",
                "clump_mask",
                "uncertain_ignore_mask",
                "filament_centerline_mask",
                "bundle_axis_mask",
                "supervised_membership_y",
                "supervised_membership_x",
                "supervised_membership_instance_id",
                "supervised_membership_class_id",
                "supervised_overlap_count",
            ],
            "latent_provenance_arrays_are_not_loss_targets": True,
            "real_compatible_training_view": {
                "semantic": "individual_filament and bundle collapse to fibrous_tau; clump stays clump; uncertain_ignore stays 255",
                "skeleton": "filament_centerline_mask OR bundle_axis_mask, clipped to fibrous_tau",
                "recommended_first_training_targets": REAL_COMPATIBLE_SUPERVISED_TARGETS,
            },
        }
    )
    metadata["multiclass_target_contract"] = {
        "class_ids": REAL_SEMANTIC_CLASSES,
        "class_priority": render_report["class_priority"],
        "binary_semantic_mask": render_report["binary_semantic_mask_source"],
        "supervised_targets": [
            "semantic_class_mask",
            "semantic_mask",
            "individual_filament_mask",
            "bundle_mask",
            "clump_mask",
            "uncertain_ignore_mask",
            "filament_centerline_mask",
            "bundle_axis_mask",
            "supervised_membership_y",
            "supervised_membership_x",
            "supervised_membership_instance_id",
            "supervised_membership_class_id",
            "supervised_overlap_count",
        ],
        "latent_synthetic_provenance": [
            "individual_filament_source_support_mask",
            "bundle_source_support_mask",
            "clump_source_support_mask",
            "uncertain_ignore_source_support_mask",
            "fiber_structure_type",
            "fiber_parent_bundle_id",
            "fiber_parent_clump_id",
            "fiber_supervised_centerline_sample",
            "bundle_child_fiber_ids",
            "clump_fragment_fiber_ids",
            "uncertain_family_code",
            "uncertain_source_object_id",
            "uncertain_target_role_code",
            "uncertain_intensity_multiplier",
            "uncertain_radius_multiplier",
            "uncertain_blur_sigma_px",
            "uncertain_adjacency_code",
            "uncertain_support_radius_multiplier",
            "uncertain_fragment_length_px",
            "latent_geometry_membership_y",
            "latent_geometry_membership_x",
            "latent_geometry_membership_instance_id",
            "latent_geometry_overlap_count",
        ],
        "real_compatible_supervised": REAL_COMPATIBLE_SUPERVISED_TARGETS,
        "synthetic_only_not_real_supervised": SYNTHETIC_ONLY_NOT_REAL_SUPERVISED,
    }
    metadata["graph_supervision"] = {
        "node_supervised": "1 only for supervised biological endpoint nodes",
        "edge_supervised": (
            "1 only when the complete graph edge is an individual-filament "
            "supervision target; partially resolved bundle children remain 0"
        ),
        "edge_structure_type": FIBER_STRUCTURE_TYPE_CODES,
        "latent_nodes_and_edges_retained": True,
    }
    metadata["target_roles"] = {
        "supervised": metadata["multiclass_target_contract"][
            "supervised_targets"
        ],
        "real_compatible_supervised": REAL_COMPATIBLE_SUPERVISED_TARGETS,
        "latent_synthetic_provenance": metadata[
            "multiclass_target_contract"
        ]["latent_synthetic_provenance"],
        "diagnostic_only": [
            "individual_filament_signal",
            "bundle_signal",
            "clump_signal",
            "uncertain_ignore_signal",
            "total_clean_signal",
            "total_optical_signal",
            "core_signal",
            "halo_signal",
            "in_focus_signal",
            "out_of_focus_signal",
            "endpoint_map",
            "projected_crossing_map",
            "crossing_map",
            "junction_map",
            "near_coplanar_crossing_map",
            "visible_instance_membership_y",
            "visible_instance_membership_x",
            "visible_instance_membership_id",
            "contributing_membership_y",
            "contributing_membership_x",
            "contributing_membership_instance_id",
        ],
    }
    metadata["target_available"].update(
        {name: True for name in REAL_COMPATIBLE_SUPERVISED_TARGETS}
    )
    metadata["target_available"].update(
        {
            "individual_filament_source_support_mask": True,
            "bundle_source_support_mask": True,
            "clump_source_support_mask": True,
            "uncertain_ignore_source_support_mask": True,
        }
    )
    metadata["enum_mappings"].update(
        {
            "trace_termination_status": TRACE_TERMINATION_STATUS_CODES,
            "fiber_structure_type": FIBER_STRUCTURE_TYPE_CODES,
            "uncertain_ignore_family": {
                "faint_fragments": 1,
                "defocused_streaks": 2,
                "low_snr_anisotropic_fragments": 3,
                "merged_boundary_filaments": 4,
                "dense_overlapping_filaments": 5,
                "bundle_clump_transition": 6,
                "clump_halo_texture": 7,
                "short_discontinuous_fragments": 8,
                "weak_directional_texture": 9,
                "ambiguous_thick_bundle_edges": 10,
                "filamentous_fluff": 11,
            },
            "uncertain_ignore_adjacency": {
                "isolated": 1,
                "fibrous_adjacent": 2,
                "clump_adjacent": 3,
                "bundle_adjacent": 4,
            },
            "boundary_bit": BOUNDARY_CODES,
        }
    )
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
    *,
    schema_version: str = DATASET_SCHEMA_VERSION_3D,
    generator_version: str = GENERATOR_VERSION_3D,
    mode: str = "persistent_chain_3d",
) -> dict[str, Any]:
    calibration_status = config.get("calibration_status", "procedural_unmatched")
    if calibration_status not in CALIBRATION_STATUSES:
        raise ValueError(f"invalid calibration_status: {calibration_status}")
    width_report = measure_isolated_fiber_fwhm(config)
    targets_config = config.get("targets", {})
    return {
        "sample_id": sample_id,
        "sample_index": sample_index,
        "dataset_schema_version": schema_version,
        "schema_migration": "0.5.0 adds projected_crossing_fiber_ids and projected_crossing_segment_indices; 0.4.0 source, distance, orientation, membership, and PSF conventions are unchanged",
        "generator_version": generator_version,
        "generator_mode": mode,
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


def save_dataset(
    config: dict[str, Any],
    out_dir: Path,
    num_workers: int = 1,
    skip_existing: bool = False,
    overwrite: bool = False,
    sample_count_override: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and overwrite:
        raise ValueError("--skip-existing and --overwrite are mutually exclusive")
    count = int(sample_count_override if sample_count_override is not None else config.get("sample_count", 8))
    mode = generator_mode(config)
    if count < 1:
        raise ValueError(f"{mode} sample_count must be at least 1")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    config = dict(config)
    config["sample_count"] = count
    indexed_rows: dict[int, dict[str, str]] = {}
    pending: list[int] = []
    for index in range(count):
        sample_id = expected_sample_id(config, index)
        existing = existing_sample_row(out_dir, sample_id)
        if existing is not None:
            if overwrite:
                pending.append(index)
            elif skip_existing:
                indexed_rows[index] = existing
            else:
                raise FileExistsError(f"{sample_id}: output exists; use --skip-existing or --overwrite")
        elif sample_outputs_exist(out_dir, sample_id):
            if overwrite:
                pending.append(index)
            else:
                raise FileExistsError(f"{sample_id}: partial or unreadable output exists; use --overwrite")
        else:
            pending.append(index)
    print(f"synthetic generation: total={count} skipped={len(indexed_rows)} pending={len(pending)} workers={num_workers}")
    if num_workers == 1:
        for done, index in enumerate(pending, start=1):
            indexed_rows[index] = write_synthetic_sample(config, out_dir, index)
            print(f"completed {done}/{len(pending)} sample_index={index}")
    else:
        set_worker_thread_limits()
        with ProcessPoolExecutor(max_workers=num_workers, initializer=set_worker_thread_limits) as pool:
            futures = {pool.submit(write_synthetic_sample, config, out_dir, index): index for index in pending}
            for done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                try:
                    indexed_rows[index] = future.result()
                except Exception as exc:
                    raise RuntimeError(f"sample_index={index} failed") from exc
                print(f"completed {done}/{len(pending)} sample_index={index}")
    rows = [indexed_rows[index] for index in range(count)]
    write_dataset_manifest(out_dir / "dataset_manifest.csv", rows)


def expected_sample_id(config: dict[str, Any], sample_index: int) -> str:
    return f"{config.get('dataset_name', 'synthetic_sted')}_{sample_index:04d}"


def sample_outputs_exist(out_dir: Path, sample_id: str) -> bool:
    return (out_dir / f"{sample_id}.npz").exists() or (out_dir / f"{sample_id}.json").exists()


def existing_sample_row(out_dir: Path, sample_id: str) -> dict[str, str] | None:
    npz_path = out_dir / f"{sample_id}.npz"
    json_path = out_dir / f"{sample_id}.json"
    if not npz_path.exists() and not json_path.exists():
        return None
    if not npz_path.exists() or not json_path.exists():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            _ = data.files
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if metadata.get("sample_id") != sample_id:
        return None
    return manifest_row(sample_id, npz_path, json_path, metadata)


def write_synthetic_sample(config: dict[str, Any], out_dir: Path, index: int) -> dict[str, str]:
    sample_id, arrays, metadata = build_sample(config, index)
    return write_sample_atomic(out_dir, sample_id, arrays, metadata)


def write_sample_atomic(
    out_dir: Path,
    sample_id: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, str]:
    npz_path = out_dir / f"{sample_id}.npz"
    json_path = out_dir / f"{sample_id}.json"
    tmp_npz = out_dir / f"{sample_id}.npz.tmp.{os.getpid()}"
    tmp_json = out_dir / f"{sample_id}.json.tmp.{os.getpid()}"
    try:
        assert_no_object_arrays(arrays)
        with tmp_npz.open("wb") as f:
            np.savez_compressed(f, **arrays)
        metadata["integrity_checksum"] = {"npz_sha256": sha256_file(tmp_npz)}
        tmp_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_npz, npz_path)
        os.replace(tmp_json, json_path)
        return manifest_row(sample_id, npz_path, json_path, metadata)
    except Exception:
        tmp_npz.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
        raise


def manifest_row(sample_id: str, npz_path: Path, json_path: Path, metadata: dict[str, Any]) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "npz_path": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "json_path": json_path.name,
        "json_sha256": sha256_file(json_path),
        "schema_version": metadata["dataset_schema_version"],
        "generator_version": metadata["generator_version"],
    }


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


def validate_dataset(
    out_dir: Path, deterministic: bool = True, num_workers: int = 1
) -> list[str]:
    manifest = out_dir / "dataset_manifest.csv"
    if not manifest.exists():
        return [f"{manifest}: missing dataset manifest"]
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if num_workers > 1:
        set_worker_thread_limits()
        errors_by_index: dict[int, list[str]] = {}
        with ProcessPoolExecutor(
            max_workers=num_workers, initializer=set_worker_thread_limits
        ) as pool:
            futures = {
                pool.submit(validate_dataset_row, out_dir, row, deterministic): index
                for index, row in enumerate(rows)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    errors_by_index[index] = future.result()
                except Exception as exc:
                    sample_id = rows[index].get("sample_id", f"row_{index}")
                    errors_by_index[index] = [
                        f"{sample_id}: validator worker failed: {type(exc).__name__}: {exc}"
                    ]
        return [
            error
            for index in range(len(rows))
            for error in errors_by_index.get(index, [])
        ]
    errors: list[str] = []
    for row in rows:
        errors.extend(validate_dataset_row(out_dir, row, deterministic))
    return errors


def validate_dataset_row(
    out_dir: Path, row: dict[str, str], deterministic: bool
) -> list[str]:
    errors: list[str] = []
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
    is_composite = metadata.get("source_blank_provenance") != "not_applicable"
    if (
        deterministic
        and not is_composite
        and metadata.get("dataset_schema_version")
        in {
            DATASET_SCHEMA_VERSION_3D,
            DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
        }
    ):
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
    if (
        schema_version == DATASET_SCHEMA_VERSION_3D_MORPHOLOGY
        and metadata.get("generator_version")
        != GENERATOR_VERSION_3D_MORPHOLOGY
    ):
        errors.append(f"{sample_id}: invalid morphology generator version")
    if (
        schema_version == DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_0_7
        and metadata.get("generator_version")
        != GENERATOR_VERSION_3D_MORPHOLOGY_0_7
    ):
        errors.append(f"{sample_id}: invalid schema 0.7 morphology generator version")
    if (
        schema_version == DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY
        and metadata.get("generator_version")
        != GENERATOR_VERSION_3D_MORPHOLOGY_LEGACY
    ):
        errors.append(f"{sample_id}: invalid legacy morphology generator version")
    is_composite = metadata.get("source_blank_provenance") != "not_applicable"
    if is_composite:
        if metadata.get("calibration_status") not in {
            "exploratory_unpartitioned",
            "exploratory_provisional_split",
        }:
            errors.append(
                f"{sample_id}: composite calibration status must be exploratory"
            )
    elif metadata.get("calibration_status") != "procedural_unmatched":
        errors.append(f"{sample_id}: MVP examples must be procedural_unmatched")
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
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_0_7,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
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
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_0_7,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
    }
    is_hardened_schema = schema_version in {
        DATASET_SCHEMA_VERSION_3D,
        DATASET_SCHEMA_VERSION_3D_HARDENED,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_0_7,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
    }
    is_morphology_schema = schema_version in {
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_0_7,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
    }
    is_corrected_morphology_schema = schema_version in {
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_0_7,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
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
    if is_corrected_morphology_schema:
        expected = np.zeros_like(
            arrays["latent_geometry_overlap_count"], dtype=np.uint16
        )
        np.add.at(
            expected,
            (
                arrays["latent_geometry_membership_y"],
                arrays["latent_geometry_membership_x"],
            ),
            1,
        )
        if not np.array_equal(
            arrays["latent_geometry_overlap_count"], expected
        ):
            errors.append(
                f"{sample_id}: latent geometry overlap does not match memberships"
            )
    else:
        expected = np.zeros_like(arrays["overlap_count"], dtype=np.uint16)
        np.add.at(
            expected, (arrays["membership_y"], arrays["membership_x"]), 1
        )
        if not np.array_equal(arrays["overlap_count"], expected):
            errors.append(
                f"{sample_id}: overlap_count does not match sparse memberships"
            )
    semantic_source = metadata["rendering_report"]["semantic_mask_source"]
    if semantic_source == "semantic_class_union":
        expected_semantic = np.isin(
            arrays["semantic_class_mask"], [1, 2, 3]
        ).astype(np.uint8)
    else:
        expected_semantic = arrays[semantic_source].astype(np.uint8)
    if not np.array_equal(arrays["semantic_mask"], expected_semantic):
        errors.append(f"{sample_id}: semantic_mask source mismatch")
    mask = (
        arrays["contributing_overlap_count"] > 0
        if "contributing_overlap_count" in arrays
        else arrays["total_clean_signal"] > 0
    )
    if np.any(arrays["nearest_depth_map"][mask] <= -0.5) or np.any(arrays["weighted_mean_depth_map"][mask] <= -0.5):
        errors.append(f"{sample_id}: depth maps missing for contributing pixels")
    fwhm = metadata["width_calibration"]["measured_fwhm_px"]
    lo = metadata["width_calibration"]["min_allowed_fwhm_px"]
    hi = metadata["width_calibration"]["max_allowed_fwhm_px"]
    if not (lo <= fwhm <= hi):
        errors.append(f"{sample_id}: width calibration FWHM {fwhm:.3f} outside [{lo}, {hi}]")
    if is_normalized_schema:
        if (
            not is_morphology_schema
            and not np.array_equal(
                arrays["trace_points_xy"], arrays["fiber_points_xy"]
            )
        ):
            errors.append(f"{sample_id}: trace_points_xy must match projected fiber_points_xy")
        if (
            not is_morphology_schema
            and not np.array_equal(
                arrays["trace_point_offsets"], arrays["fiber_point_offsets"]
            )
        ):
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
            emitted = float(
                metadata["rendering_report"].get(
                    "line_source_integrated_signal",
                    metadata["rendering_report"]["total_emitted_source_signal"],
                )
            )
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
            if metadata["rendering_report"].get("scenario_category") not in {
                "structural_qa",
                "optical_qa",
                "realism_calibration",
                "clump_ignore_stress",
                "uncertain_ignore_stress",
            }:
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
        if schema_version in {
            DATASET_SCHEMA_VERSION_3D,
            DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY,
            DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_0_7,
            DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
        }:
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
        if is_morphology_schema:
            errors.extend(
                validate_morphology_arrays(
                    sample_id,
                    arrays,
                    metadata,
                    corrected=is_corrected_morphology_schema,
                )
            )
    return errors


def validate_morphology_arrays(
    sample_id: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    corrected: bool,
) -> list[str]:
    errors: list[str] = []
    classes = arrays["semantic_class_mask"]
    allowed = {0, 1, 2, 3, 255}
    present = set(map(int, np.unique(classes)))
    if not present <= allowed:
        errors.append(
            f"{sample_id}: invalid semantic class IDs {sorted(present - allowed)}"
        )
    masks = {
        1: arrays["individual_filament_mask"],
        2: arrays["bundle_mask"],
        3: arrays["clump_mask"],
        255: arrays["uncertain_ignore_mask"],
    }
    for class_id, mask in masks.items():
        if not np.array_equal(mask, (classes == class_id).astype(np.uint8)):
            errors.append(
                f"{sample_id}: class mask {class_id} disagrees with semantic_class_mask"
            )
    expected_binary = np.isin(classes, [1, 2, 3]).astype(np.uint8)
    if not np.array_equal(arrays["semantic_mask"], expected_binary):
        errors.append(
            f"{sample_id}: binary semantic mask must equal classes 1/2/3"
        )
    if np.any(
        arrays["filament_centerline_mask"]
        & ~arrays["individual_filament_mask"]
    ):
        errors.append(
            f"{sample_id}: filament centerline lies outside individual-filament mask"
        )
    if np.any(arrays["bundle_axis_mask"] & ~arrays["bundle_mask"]):
        errors.append(f"{sample_id}: bundle axis lies outside bundle mask")
    if np.any(arrays["endpoint_map"] & arrays["clump_mask"]):
        errors.append(f"{sample_id}: clumps contain supervised endpoints")
    if np.any(arrays["filament_centerline_mask"] & arrays["clump_mask"]):
        errors.append(f"{sample_id}: clumps contain filament centerlines")
    for prefix, mask_name in [
        ("individual_filament", "individual_filament_mask"),
        ("bundle", "bundle_mask"),
        ("clump", "clump_mask"),
    ]:
        y = arrays[f"{prefix}_membership_y"]
        x = arrays[f"{prefix}_membership_x"]
        if y.shape != x.shape or y.shape != arrays[
            f"{prefix}_membership_instance_id"
        ].shape:
            errors.append(f"{sample_id}: malformed {prefix} memberships")
        elif np.any(arrays[mask_name][y, x] == 0):
            errors.append(
                f"{sample_id}: {prefix} membership lies outside class mask"
            )
        else:
            covered = np.zeros_like(arrays[mask_name], dtype=np.uint8)
            covered[y, x] = 1
            if not np.array_equal(covered, arrays[mask_name]):
                errors.append(
                    f"{sample_id}: {prefix} memberships do not exactly cover class mask"
                )
    trace_count = arrays["trace_ids"].shape[0]
    for name in [
        "trace_fiber_ids",
        "trace_status",
        "trace_start_status",
        "trace_end_status",
        "trace_source",
    ]:
        if arrays[name].shape != (trace_count,):
            errors.append(f"{sample_id}: {name} trace-count mismatch")
    valid_status = set(TRACE_TERMINATION_STATUS_CODES.values())
    for name in ["trace_start_status", "trace_end_status"]:
        if not set(map(int, np.unique(arrays[name]))) <= valid_status:
            errors.append(f"{sample_id}: invalid {name} code")
    if arrays["fiber_structure_type"].shape != arrays["fiber_ids"].shape:
        errors.append(f"{sample_id}: fiber structure-type count mismatch")
    if arrays["fiber_supervised_centerline_sample"].shape[0] != arrays[
        "fiber_points_xyz"
    ].shape[0]:
        errors.append(
            f"{sample_id}: supervised centerline sample count mismatch"
        )
    axis_count = arrays["bundle_axis_points_xyz"].shape[0]
    for name in ["bundle_axis_points_xy", "bundle_axis_radius_px", "bundle_axis_unresolved_sample"]:
        if arrays[name].shape[0] != axis_count:
            errors.append(f"{sample_id}: {name} bundle-axis count mismatch")
    bundle_count = arrays["bundle_ids"].shape[0]
    for name in [
        "bundle_partial_resolution",
        "bundle_twist_rate",
        "bundle_converging",
        "bundle_diverging",
    ]:
        if arrays[name].shape != (bundle_count,):
            errors.append(f"{sample_id}: {name} bundle-count mismatch")
    clump_count = arrays["clump_ids"].shape[0]
    for name in [
        "clump_internal_density",
        "clump_irregularity",
        "clump_edge_diffuseness",
    ]:
        if arrays[name].shape != (clump_count,):
            errors.append(f"{sample_id}: {name} clump-count mismatch")
    for name in ["clump_center_xyz", "clump_radius_xyz"]:
        if arrays[name].shape != (clump_count, 3):
            errors.append(f"{sample_id}: {name} clump-count mismatch")
    target_available = metadata.get("target_available", {})
    for name in [
        "semantic_class_mask",
        "individual_filament_mask",
        "bundle_mask",
        "clump_mask",
        "uncertain_ignore_mask",
        "filament_centerline_mask",
        "bundle_axis_mask",
    ]:
        if target_available.get(name) is not True:
            errors.append(f"{sample_id}: target_available missing {name}")
    if corrected:
        errors.extend(
            validate_corrected_morphology_semantics(
                sample_id, arrays, metadata
            )
        )
        has_real_compatible_view = any(
            name in arrays for name in REAL_COMPATIBLE_SUPERVISED_TARGETS
        ) or bool(
            metadata.get("target_roles", {}).get(
                "real_compatible_supervised"
            )
        )
        if has_real_compatible_view:
            errors.extend(
                validate_real_compatible_targets(sample_id, arrays, metadata)
            )
    if metadata.get("dataset_schema_version") == DATASET_SCHEMA_VERSION_3D_MORPHOLOGY:
        errors.extend(validate_schema08_apparent_masks(sample_id, arrays, metadata))
    return errors


def validate_schema08_apparent_masks(
    sample_id: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for name in [
        "individual_filament_source_support_mask",
        "bundle_source_support_mask",
        "clump_source_support_mask",
        "uncertain_ignore_source_support_mask",
    ]:
        if name not in arrays:
            errors.append(f"{sample_id}: missing source-support mask {name}")
    roles = metadata.get("target_roles", {})
    role_names = [
        "supervised",
        "latent_synthetic_provenance",
        "diagnostic_only",
        "real_compatible_supervised",
    ]
    role_sets = {name: set(roles.get(name, [])) for name in role_names}
    for i, left in enumerate(role_names):
        for right in role_names[i + 1 :]:
            overlap = role_sets[left] & role_sets[right]
            if overlap:
                errors.append(
                    f"{sample_id}: target role lists overlap for {left}/{right}: {sorted(overlap)}"
                )
    supervised = role_sets["supervised"]
    latent = role_sets["latent_synthetic_provenance"]
    for name in [
        "individual_filament_source_support_mask",
        "bundle_source_support_mask",
        "clump_source_support_mask",
        "uncertain_ignore_source_support_mask",
    ]:
        if name in supervised or name not in latent:
            errors.append(f"{sample_id}: {name} must be latent, not supervised")
    if np.any(arrays["uncertain_ignore_mask"] & arrays["individual_filament_mask"]):
        errors.append(f"{sample_id}: uncertain_ignore overlaps individual_filament")
    if np.any(arrays["uncertain_ignore_mask"] & arrays["bundle_mask"]):
        errors.append(f"{sample_id}: uncertain_ignore overlaps bundle")
    if np.any(arrays["uncertain_ignore_mask"] & arrays["clump_mask"]):
        errors.append(f"{sample_id}: uncertain_ignore overlaps clump")
    if np.any(arrays["filament_centerline_mask"] & arrays["uncertain_ignore_mask"]):
        errors.append(f"{sample_id}: filament centerline overlaps uncertain_ignore")
    if np.any(arrays["bundle_axis_mask"] & arrays["uncertain_ignore_mask"]):
        errors.append(f"{sample_id}: bundle axis overlaps uncertain_ignore")

    report = metadata.get("rendering_report", {})
    rule = report.get("apparent_mask_rule", {})
    visible = float(report["visible_signal_threshold"])
    low = visible * float(rule.get("low_factor", 0.75))
    high = visible * float(rule.get("high_factor", 1.0))
    max_distance = float(rule.get("max_source_distance_px", 4.0))
    expected = expected_apparent_semantic(arrays, low, high, max_distance)
    if not np.array_equal(arrays["semantic_class_mask"], expected):
        errors.append(f"{sample_id}: apparent semantic mask does not match threshold rule")
    if not np.array_equal(
        arrays["semantic_mask"], np.isin(expected, [1, 2, 3]).astype(np.uint8)
    ):
        errors.append(f"{sample_id}: semantic_mask must exclude uncertain_ignore")
    return errors


def expected_apparent_semantic(
    arrays: dict[str, np.ndarray],
    low_threshold: float,
    high_threshold: float,
    max_distance: float,
) -> np.ndarray:
    def region(mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return mask.astype(bool)
        try:
            from scipy import ndimage
        except Exception:
            return mask.astype(bool)
        return ndimage.distance_transform_edt(~mask.astype(bool)) <= max_distance

    sources = {
        "individual_filament": arrays["individual_filament_source_support_mask"].astype(bool),
        "bundle": arrays["bundle_source_support_mask"].astype(bool),
        "clump": arrays["clump_source_support_mask"].astype(bool),
        "uncertain_ignore": arrays.get(
            "uncertain_ignore_source_support_mask",
            np.zeros_like(arrays["semantic_class_mask"]),
        ).astype(bool),
    }
    signals = {
        "individual_filament": arrays["individual_filament_signal"],
        "bundle": arrays["bundle_signal"],
        "clump": arrays["clump_signal"],
        "uncertain_ignore": arrays.get(
            "uncertain_ignore_signal",
            np.zeros_like(arrays["semantic_class_mask"], dtype=np.float32),
        ),
    }
    high = {
        name: (signals[name] >= high_threshold) & region(mask)
        for name, mask in sources.items()
        if name != "uncertain_ignore"
    }
    low = {
        name: (signals[name] >= low_threshold)
        & (signals[name] < high_threshold)
        & region(mask)
        for name, mask in sources.items()
        if name != "uncertain_ignore"
    }
    semantic = np.zeros_like(arrays["semantic_class_mask"], dtype=np.uint8)
    semantic[high["individual_filament"]] = 1
    semantic[high["bundle"]] = 2
    semantic[high["clump"]] = 3
    uncertain = (low["individual_filament"] | low["bundle"] | low["clump"]) & (semantic == 0)
    source_union = (
        arrays["individual_filament_source_support_mask"].astype(bool)
        | arrays["bundle_source_support_mask"].astype(bool)
        | arrays["clump_source_support_mask"].astype(bool)
    )
    transition = (
        arrays.get("bundle_transition_mask", 0)
        | arrays.get("clump_transition_mask", 0)
    ) & source_union
    uncertain |= np.asarray(transition).astype(bool)
    uncertain |= sources["uncertain_ignore"] & (semantic == 0)
    semantic[uncertain] = 255
    return semantic


def validate_corrected_morphology_semantics(
    sample_id: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    forbidden = {
        "membership_y",
        "membership_x",
        "membership_instance_id",
        "overlap_count",
    } & set(arrays)
    if forbidden:
        errors.append(
            f"{sample_id}: schema 0.7 prohibits ambiguous generic memberships "
            f"{sorted(forbidden)}"
        )
    expected_y = np.concatenate(
        [
            arrays["individual_filament_membership_y"],
            arrays["bundle_membership_y"],
            arrays["clump_membership_y"],
        ]
    )
    expected_x = np.concatenate(
        [
            arrays["individual_filament_membership_x"],
            arrays["bundle_membership_x"],
            arrays["clump_membership_x"],
        ]
    )
    expected_ids = np.concatenate(
        [
            arrays["individual_filament_membership_instance_id"],
            arrays["bundle_membership_instance_id"],
            arrays["clump_membership_instance_id"],
        ]
    )
    expected_classes = np.concatenate(
        [
            np.full(
                arrays["individual_filament_membership_y"].shape,
                1,
                dtype=np.uint8,
            ),
            np.full(
                arrays["bundle_membership_y"].shape, 2, dtype=np.uint8
            ),
            np.full(
                arrays["clump_membership_y"].shape, 3, dtype=np.uint8
            ),
        ]
    )
    for name, expected in [
        ("supervised_membership_y", expected_y),
        ("supervised_membership_x", expected_x),
        ("supervised_membership_instance_id", expected_ids),
        ("supervised_membership_class_id", expected_classes),
    ]:
        if not np.array_equal(arrays[name], expected):
            errors.append(
                f"{sample_id}: {name} disagrees with class-specific memberships"
            )
    supervised_overlap = np.zeros_like(
        arrays["supervised_overlap_count"], dtype=np.uint16
    )
    np.add.at(supervised_overlap, (expected_y, expected_x), 1)
    if not np.array_equal(
        supervised_overlap, arrays["supervised_overlap_count"]
    ):
        errors.append(
            f"{sample_id}: supervised overlap does not match memberships"
        )
    if expected_y.size and np.any(
        arrays["semantic_class_mask"][expected_y, expected_x]
        != expected_classes
    ):
        errors.append(
            f"{sample_id}: supervised membership has an invalid class"
        )
    fiber_type = dict(
        zip(
            map(int, arrays["fiber_ids"]),
            map(int, arrays["fiber_structure_type"]),
        )
    )
    supervised_filament_ids = set(
        map(int, arrays["individual_filament_membership_instance_id"])
    )
    if any(fiber_type.get(fid) == FIBER_STRUCTURE_TYPE_CODES["clump_fragment"] for fid in supervised_filament_ids):
        errors.append(
            f"{sample_id}: clump fragments leaked into filament memberships"
        )
    target_roles = metadata.get("target_roles", {})
    supervised_targets = set(target_roles.get("supervised", []))
    latent_targets = set(target_roles.get("latent_synthetic_provenance", []))
    if not latent_targets or supervised_targets & latent_targets:
        errors.append(
            f"{sample_id}: supervised and latent target roles are ambiguous"
        )
    if metadata.get("target_available", {}).get("instance_membership") is not False:
        errors.append(
            f"{sample_id}: ambiguous generic instance membership must be unavailable"
        )
    if metadata.get("target_available", {}).get(
        "supervised_instance_membership"
    ) is not True:
        errors.append(
            f"{sample_id}: supervised instance membership availability missing"
        )
    valid = TRACE_TERMINATION_STATUS_CODES["valid_endpoint"]
    boundary = TRACE_TERMINATION_STATUS_CODES["boundary_truncation"]
    node_supervised = arrays["node_supervised"].astype(bool)
    node_status = arrays["node_termination_status"]
    node_boundary = arrays["node_boundary_code"]
    node_count = arrays["node_xyz"].shape[0]
    for name in ["node_supervised", "node_termination_status", "node_boundary_code"]:
        if arrays[name].shape != (node_count,):
            errors.append(f"{sample_id}: {name} node-count mismatch")
    edge_count = arrays["edge_node_indices"].shape[0]
    for name in ["edge_supervised", "edge_structure_type"]:
        if arrays[name].shape != (edge_count,):
            errors.append(f"{sample_id}: {name} edge-count mismatch")
    fiber_count = arrays["fiber_ids"].shape[0]
    for name in ["fiber_start_boundary_code", "fiber_end_boundary_code"]:
        if arrays[name].shape != (fiber_count,):
            errors.append(f"{sample_id}: {name} fiber-count mismatch")
    if np.any(node_supervised & (node_status != valid)):
        errors.append(
            f"{sample_id}: non-valid graph node marked supervised"
        )
    if np.any((node_boundary > 0) & (node_status != boundary)):
        errors.append(
            f"{sample_id}: boundary graph node lacks boundary status"
        )
    if np.any((node_status == boundary) & node_supervised):
        errors.append(
            f"{sample_id}: boundary graph node marked supervised"
        )
    truncated_node = arrays["node_type"] == NODE_TYPES["truncated_endpoint"]
    if not np.array_equal(truncated_node, node_status == boundary):
        errors.append(
            f"{sample_id}: truncated node type disagrees with termination status"
        )
    if edge_count:
        start_nodes = arrays["edge_node_indices"][:, 0]
        end_nodes = arrays["edge_node_indices"][:, 1]
        if not np.array_equal(
            arrays["edge_truncated_start"].astype(bool),
            node_status[start_nodes] == boundary,
        ):
            errors.append(
                f"{sample_id}: edge start truncation disagrees with node status"
            )
        if not np.array_equal(
            arrays["edge_truncated_end"].astype(bool),
            node_status[end_nodes] == boundary,
        ):
            errors.append(
                f"{sample_id}: edge end truncation disagrees with node status"
            )
    if np.any(
        arrays["edge_supervised"].astype(bool)
        & (
            arrays["edge_structure_type"]
            != FIBER_STRUCTURE_TYPE_CODES["individual_filament"]
        )
    ):
        errors.append(
            f"{sample_id}: latent bundle/clump edge marked supervised"
        )
    if metadata.get("dataset_schema_version") != DATASET_SCHEMA_VERSION_3D_MORPHOLOGY:
        expected_endpoint = np.zeros_like(arrays["endpoint_map"], dtype=np.uint8)
        radius = float(
            metadata.get("generation_config", {})
            .get("targets", {})
            .get("endpoint_radius_px", 3.0)
        )
        offsets = arrays["trace_point_offsets"]
        for index, (start_status, end_status) in enumerate(
            zip(arrays["trace_start_status"], arrays["trace_end_status"])
        ):
            start, end = int(offsets[index]), int(offsets[index + 1])
            if end <= start:
                continue
            if int(start_status) == valid:
                draw_disk(expected_endpoint, arrays["trace_points_xy"][start], radius)
            if int(end_status) == valid:
                draw_disk(expected_endpoint, arrays["trace_points_xy"][end - 1], radius)
        expected_endpoint &= arrays["individual_filament_mask"]
        if not np.array_equal(expected_endpoint, arrays["endpoint_map"]):
            errors.append(
                f"{sample_id}: endpoint map contains non-valid or missing endpoints"
            )
    if not np.allclose(
        arrays["individual_filament_signal"]
        + arrays["bundle_signal"]
        + arrays["clump_signal"]
        + arrays.get(
            "uncertain_ignore_signal",
            np.zeros_like(arrays["clump_signal"], dtype=np.float32),
        ),
        arrays["total_clean_signal"],
        atol=1e-4,
    ):
        errors.append(
            f"{sample_id}: class-attributed signals do not reconstruct total signal"
        )
    errors.extend(validate_signal_alignment(sample_id, metadata, arrays))
    errors.extend(validate_compact_class_geometry(sample_id, arrays, metadata))
    return errors


def validate_signal_alignment(
    sample_id: str,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray] | None = None,
) -> list[str]:
    errors = []
    report = metadata.get("rendering_report", {}).get(
        "class_signal_alignment", {}
    )
    limits = metadata.get("generation_config", {}).get("targets", {}).get(
        "signal_alignment_acceptance", {}
    )
    min_nonzero = float(limits.get("min_nonzero_fraction", 0.05))
    max_outside = float(limits.get("max_visible_signal_outside_fraction", 0.95))
    for class_name in ["individual_filament", "bundle", "clump"]:
        metrics = report.get(class_name)
        if not metrics:
            errors.append(
                f"{sample_id}: missing signal alignment for {class_name}"
            )
            continue
        if int(metrics["class_area_px"]) == 0:
            continue
        if float(metrics["fraction_of_class_mask_with_nonzero_signal"]) < min_nonzero:
            errors.append(
                f"{sample_id}: {class_name} mask has insufficient rendered signal"
            )
        outside = unsupported_visible_signal_fraction(class_name, arrays, metadata)
        if outside is None:
            outside = float(
                metrics["fraction_of_visible_class_signal_outside_class_mask"]
            )
        if outside > max_outside:
            errors.append(
                f"{sample_id}: {class_name} signal spill exceeds acceptance limit"
            )
    return errors


def unsupported_visible_signal_fraction(
    class_name: str,
    arrays: dict[str, np.ndarray] | None,
    metadata: dict[str, Any],
) -> float | None:
    if arrays is None:
        return None
    signal_name = f"{class_name}_signal"
    mask_name = f"{class_name}_mask"
    required = [
        signal_name,
        mask_name,
        "individual_filament_mask",
        "bundle_mask",
        "clump_mask",
        "uncertain_ignore_mask",
    ]
    if not all(name in arrays for name in required):
        return None
    signal = arrays[signal_name]
    threshold = float(
        metadata.get("rendering_report", {}).get("visible_signal_threshold", 3.0)
    )
    visible = signal > threshold
    visible_energy = float(signal[visible].sum())
    if visible_energy <= 0:
        return 0.0
    compatible = (
        arrays["individual_filament_mask"].astype(bool)
        | arrays["bundle_mask"].astype(bool)
        | arrays["clump_mask"].astype(bool)
        | arrays["uncertain_ignore_mask"].astype(bool)
    )
    outside_energy = float(signal[visible & ~compatible].sum())
    return outside_energy / visible_energy


def validate_compact_class_geometry(
    sample_id: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> list[str]:
    errors = []
    try:
        from scipy import ndimage
    except Exception:
        return [f"{sample_id}: SciPy required for morphology validation"]
    config = metadata.get("generation_config", {}).get("targets", {})
    if np.any(arrays["bundle_mask"]):
        axis_distance = ndimage.distance_transform_edt(
            arrays["bundle_mask"] > 0
        )[arrays["bundle_axis_mask"].astype(bool)]
        if not axis_distance.size:
            errors.append(f"{sample_id}: bundle axis has no supervised pixels")
            width = 0.0
        else:
            width = 2.0 * float(np.percentile(axis_distance, 50))
        min_width = float(
            config.get(
                "min_bundle_width_px",
                metadata["width_calibration"]["min_allowed_fwhm_px"],
            )
        )
        tolerance = float(config.get("bundle_width_tolerance_px", 0.05))
        if width + tolerance <= min_width:
            errors.append(
                f"{sample_id}: unresolved bundle width {width:.3f} "
                f"does not exceed filament width {min_width:.3f}"
            )
    if np.any(arrays["clump_mask"]):
        dark_fraction = float(
            np.mean(
                arrays["clump_signal"][arrays["clump_mask"].astype(bool)]
                <= 0
            )
        )
        if dark_fraction > float(config.get("max_clump_dark_fraction", 0.2)):
            errors.append(
                f"{sample_id}: clump dark-hole fraction {dark_fraction:.3f} "
                "exceeds acceptance limit"
            )
        if metadata.get("dataset_schema_version") == DATASET_SCHEMA_VERSION_3D_MORPHOLOGY:
            return errors
        for clump_id in arrays["clump_ids"]:
            y = arrays["clump_membership_y"][
                arrays["clump_membership_instance_id"] == clump_id
            ]
            x = arrays["clump_membership_x"][
                arrays["clump_membership_instance_id"] == clump_id
            ]
            if y.size < int(config.get("min_clump_area_px", 16)):
                errors.append(
                    f"{sample_id}: clump {int(clump_id)} is too small"
                )
                continue
            local = np.zeros_like(arrays["clump_mask"], dtype=bool)
            local[y, x] = True
            component_count = ndimage.label(local)[1]
            if component_count > int(
                config.get("max_clump_components_per_instance", 4)
            ):
                errors.append(
                    f"{sample_id}: clump {int(clump_id)} has "
                    f"{component_count} disconnected components"
                )
            filled = ndimage.binary_fill_holes(local)
            hole_fill_ratio = float(local.sum() / max(int(filled.sum()), 1))
            if hole_fill_ratio < float(config.get("min_clump_hole_fill_ratio", 0.2)):
                errors.append(
                    f"{sample_id}: clump {int(clump_id)} hole-fill ratio too low"
                )
    return errors
