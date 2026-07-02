"""Synthetic fiber compositing onto expert-validated real blank backgrounds."""

from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from fibras.sted_inventory import read_image
from fibras.synthetic.geometry import generate_geometry, geometry_to_arrays
from fibras.synthetic.geometry3d import generate_persistent_chain_geometry, geometry3d_to_arrays
from fibras.synthetic.morphology3d import (
    generate_morphology_geometry,
    morphology_geometry_to_arrays,
)
from fibras.synthetic.rasterizer3d import rasterize_3d_sample
from fibras.synthetic.rendering import gaussian_blur, map_to_uint8
from fibras.synthetic.schema import (
    DATASET_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION_3D,
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
    GENERATOR_VERSION,
    GENERATOR_VERSION_3D,
    GENERATOR_VERSION_3D_MORPHOLOGY,
    NODE_TYPES,
    REAL_SEMANTIC_CLASSES,
    TRACE_TERMINATION_STATUSES,
)
from fibras.synthetic.storage import (
    assert_no_object_arrays,
    existing_sample_row,
    sample_outputs_exist,
    set_worker_thread_limits,
    sha256_file,
    write_dataset_manifest,
    write_sample_atomic,
)
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


def repeat_to_count(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    return [rows[i % len(rows)] for i in range(count)]


def select_blank_pool_rows(
    inventory_dir: Path,
    pool_path: Path,
    role: str,
    count: int,
    allow_reuse: bool = False,
) -> list[dict[str, str]]:
    blanks = build_blank_lookup(inventory_dir)
    rows = [r for r in read_csv(pool_path) if r.get("blank_pool_role") == role]
    selected = [blanks[r["stable_image_id"]] for r in sorted(rows, key=lambda r: r["stable_image_id"]) if r["stable_image_id"] in blanks]
    if allow_reuse and selected:
        return repeat_to_count(selected, count)
    if len(selected) < count:
        raise ValueError(f"not enough blank images for role {role}: need {count}, found {len(selected)}")
    return selected[:count]


def generate_composites(
    config: dict[str, Any],
    inventory_dir: Path,
    splits_path: Path,
    out_dir: Path,
    num_workers: int = 1,
    skip_existing: bool = False,
    overwrite: bool = False,
    sample_count_override: int | None = None,
) -> None:
    status = config.get("calibration_data_status", "exploratory_unpartitioned")
    if status not in CALIBRATION_DATA_STATUSES or not status.startswith("exploratory"):
        raise ValueError("this exploratory compositor requires exploratory calibration_data_status")
    if skip_existing and overwrite:
        raise ValueError("--skip-existing and --overwrite are mutually exclusive")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    sample_count = int(sample_count_override if sample_count_override is not None else config.get("compositing", {}).get("sample_count", 8))
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    config = dict(config)
    config["compositing"] = dict(config.get("compositing", {}))
    config["compositing"]["sample_count"] = sample_count
    split = config.get("compositing", {}).get("synthetic_split", "calibration")
    blank_pool_role = config.get("compositing", {}).get("blank_pool_role", split)
    source_roots = config["source_roots"]
    pool_manifest = config.get("compositing", {}).get("blank_pool_manifest")
    if pool_manifest:
        blank_rows = select_blank_pool_rows(
            inventory_dir,
            Path(pool_manifest),
            blank_pool_role,
            sample_count,
            bool(config.get("compositing", {}).get("allow_blank_reuse_within_pool", False)),
        )
    else:
        blank_rows = select_blank_rows(inventory_dir, splits_path, split, sample_count)
    out_dir.mkdir(parents=True, exist_ok=True)
    indexed_rows: dict[int, dict[str, str]] = {}
    pending: list[int] = []
    for index in range(sample_count):
        sample_id = expected_composite_sample_id(config, index)
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
    print(f"blank-composite generation: total={sample_count} skipped={len(indexed_rows)} pending={len(pending)} workers={num_workers}")
    if num_workers == 1:
        for done, index in enumerate(pending, start=1):
            indexed_rows[index] = write_composite_sample(config, blank_rows[index], source_roots, index, inventory_dir, splits_path, out_dir)
            print(f"completed {done}/{len(pending)} sample_index={index}")
    else:
        set_worker_thread_limits()
        with ProcessPoolExecutor(max_workers=num_workers, initializer=set_worker_thread_limits) as pool:
            futures = {
                pool.submit(write_composite_sample, config, blank_rows[index], source_roots, index, inventory_dir, splits_path, out_dir): index
                for index in pending
            }
            for done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                try:
                    indexed_rows[index] = future.result()
                except Exception as exc:
                    raise RuntimeError(f"sample_index={index} failed") from exc
                print(f"completed {done}/{len(pending)} sample_index={index}")
    manifest_rows = [indexed_rows[index] for index in range(sample_count)]
    write_dataset_manifest(out_dir / "dataset_manifest.csv", manifest_rows)


def expected_composite_sample_id(config: dict[str, Any], sample_index: int) -> str:
    if config.get("generator_mode") in {"persistent_chain_3d", "morphology_scene_3d"}:
        base = config.get("compositing", {}).get("composite_dataset_name", config.get("dataset_name", "sted_blank_composite"))
    else:
        base = config.get("dataset_name", "sted_blank_composite")
    return f"{base}_{sample_index:04d}"


def write_composite_sample(
    config: dict[str, Any],
    blank_row: dict[str, str],
    source_roots: dict[str, str],
    sample_index: int,
    inventory_dir: Path,
    splits_path: Path,
    out_dir: Path,
) -> dict[str, str]:
    sample_id, arrays, metadata = build_composite_sample(config, blank_row, source_roots, sample_index, inventory_dir, splits_path)
    return write_sample_atomic(out_dir, sample_id, arrays, metadata)


def load_parent_synthetic(config: dict[str, Any], sample_index: int) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, str]] | None:
    parent_dir = config.get("compositing", {}).get("parent_synthetic_dir")
    if not parent_dir:
        return None
    parent_path = Path(parent_dir)
    rows = read_csv(parent_path / "dataset_manifest.csv")
    if sample_index >= len(rows):
        raise ValueError(f"parent_synthetic_dir has {len(rows)} rows, cannot resolve sample index {sample_index}")
    row = rows[sample_index]
    metadata = json.loads((parent_path / row["json_path"]).read_text(encoding="utf-8"))
    with np.load(parent_path / row["npz_path"], allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    return arrays, metadata, row


def build_composite_sample(
    config: dict[str, Any],
    blank_row: dict[str, str],
    source_roots: dict[str, str],
    sample_index: int,
    inventory_dir: Path,
    splits_path: Path,
) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    if config.get("generator_mode") in {
        "persistent_chain_3d",
        "morphology_scene_3d",
    }:
        return build_composite_sample_3d(config, blank_row, source_roots, sample_index, inventory_dir, splits_path)
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
        None,
        {},
        {},
    )
    return sample_id, arrays, metadata


def build_composite_sample_3d(
    config: dict[str, Any],
    blank_row: dict[str, str],
    source_roots: dict[str, str],
    sample_index: int,
    inventory_dir: Path,
    splits_path: Path,
) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    compositor_config = config.get("compositing", {})
    parent = load_parent_synthetic(config, sample_index)
    geometry_config = dict(config["geometry"])
    output_config = dict(config.get("output_mapping", {}))
    mapping_config = {**output_config, **config.get("real_blank_rendering", {})}
    sample_id = f"{compositor_config.get('composite_dataset_name', config.get('dataset_name', 'sted_blank_composite'))}_{sample_index:04d}"
    blank_path = Path(source_roots[blank_row["source_root_id"]]) / blank_row["relative_path"]
    blank, _, _ = read_image(blank_path)
    blank_float = blank.astype(np.float32)
    if parent:
        arrays, parent_metadata, parent_row = parent
        geometry_seed = int(parent_metadata["geometry_seed"])
        rendering_seed = int(parent_metadata["rendering_seed"])
        geometry = {"parameters": parent_metadata["geometry_parameters"]}
        local_config = dict(parent_metadata["generation_config"])
        render_report = dict(parent_metadata["rendering_report"])
        if list(arrays["render_uint8"].shape) != list(blank_float.shape):
            raise ValueError("parent synthetic sample and blank image shape differ")
    else:
        geometry_config["image_shape"] = [int(blank_float.shape[0]), int(blank_float.shape[1])]
        geometry_seed = int(geometry_config.get("base_seed", 61001)) + sample_index
        rendering_seed = int(output_config.get("base_seed", 62001)) + sample_index
        local_config = dict(config)
        local_config["geometry"] = geometry_config
        if config.get("generator_mode") == "morphology_scene_3d":
            local_config = dict(config)
            local_config["geometry"] = geometry_config
            geometry = generate_morphology_geometry(local_config, sample_index)
            arrays = morphology_geometry_to_arrays(
                geometry, geometry3d_to_arrays(geometry)
            )
        else:
            geometry = generate_persistent_chain_geometry(
                geometry_config, sample_index
            )
            arrays = geometry3d_to_arrays(geometry)
        raster_output = dict(output_config)
        raster_output["background_level"] = 0.0
        raster_output["background_noise_std"] = 0.0
        raster_arrays, render_report = rasterize_3d_sample(
            geometry,
            config.get("targets", {}),
            config.get("optical_model", {}),
            raster_output,
            rendering_seed,
        )
        arrays.update(raster_arrays)
        parent_metadata = {}
        parent_row = {}
    compositing_seed = int(compositor_config.get("base_seed", 63001)) + sample_index
    foreground_scale = composite_foreground_scale(
        compositor_config, compositing_seed
    )
    signal = arrays["total_clean_signal"].astype(np.float32) * foreground_scale
    perturb_scale = float(config.get("real_blank_rendering", {}).get("signal_dependent_perturbation_scale", 0.0))
    if perturb_scale > 0:
        rng = np.random.default_rng(compositing_seed)
        signal = signal + rng.normal(0, perturb_scale * np.sqrt(np.maximum(signal, 0)), signal.shape).astype(np.float32)
    signal = np.maximum(signal, 0).astype(np.float32)
    blank_scale = float(compositor_config.get("blank_scale", 1.0))
    composite = blank_float * blank_scale + signal
    composite_uint8, mapping_stats = map_to_uint8(composite, mapping_config)
    arrays["blank_float"] = blank_float.astype(np.float32)
    arrays["synthetic_signal_float"] = signal.astype(np.float32)
    arrays["composite_float"] = composite.astype(np.float32)
    arrays["render_float"] = composite.astype(np.float32)
    arrays["render_uint8"] = composite_uint8
    metadata = composite_metadata(
        sample_id,
        local_config,
        blank_row,
        geometry,
        arrays,
        geometry_seed,
        rendering_seed,
        compositing_seed,
        mapping_stats,
        inventory_dir,
        splits_path,
        render_report,
        parent_metadata,
        parent_row,
    )
    return sample_id, arrays, metadata


def composite_foreground_scale(
    config: dict[str, Any], compositing_seed: int
) -> float:
    value = config.get("foreground_scale_range")
    if value is None:
        return float(config.get("foreground_scale", 1.0))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("foreground_scale_range must contain [min, max]")
    rng = np.random.default_rng(compositing_seed)
    return float(rng.uniform(float(value[0]), float(value[1])))


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
    render_report: dict[str, Any] | None,
    parent_metadata: dict[str, Any] | None = None,
    parent_row: dict[str, str] | None = None,
) -> dict[str, Any]:
    status = config.get("calibration_data_status", "exploratory_unpartitioned")
    mode = config.get("generator_mode")
    is_3d = mode in {"persistent_chain_3d", "morphology_scene_3d"}
    is_morphology = mode == "morphology_scene_3d"
    schema_version = (
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY
        if is_morphology
        else DATASET_SCHEMA_VERSION_3D
        if is_3d
        else DATASET_SCHEMA_VERSION
    )
    generator_version = (
        GENERATOR_VERSION_3D_MORPHOLOGY
        if is_morphology
        else GENERATOR_VERSION_3D
        if is_3d
        else GENERATOR_VERSION
    )
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
            "dataset_schema_version": schema_version,
            "calibration_artifact_schema_version": CALIBRATION_SCHEMA_VERSION,
            "generator_version": generator_version,
            "generator_mode": config.get("generator_mode", "legacy_2d"),
            "geometry_seed": geometry_seed,
            "rendering_seed": rendering_seed,
            "compositing_seed": compositing_seed,
            "geometry_parameters": geometry["parameters"],
            "scenario": (parent_metadata or {}).get(
                "scenario", geometry["parameters"].get("scenario", "not_reported")
            ),
            "scenario_category": (parent_metadata or {}).get(
                "scenario_category",
                (render_report or {}).get(
                    "scenario_category", "realism_calibration"
                ),
            ),
            "renderer_configuration": config.get("optical_model", config.get("real_blank_rendering", {})),
            "output_mapping_configuration": config.get("output_mapping", config.get("real_blank_rendering", {})),
            "compositor_configuration": config.get("compositing", {}),
            "applied_compositor_foreground_scale": float(
                arrays["synthetic_signal_float"].sum()
                / max(float(arrays.get("total_clean_signal", arrays["synthetic_signal_float"]).sum()), 1e-12)
            ),
            "source_blank_provenance": {
                "source_root_id": blank_row["source_root_id"],
                "blank_stable_image_id": blank_row["stable_image_id"],
                "blank_relative_path": blank_row["relative_path"],
                "blank_sha256": blank_row["source_sha256"],
                "blank_acquisition_group": blank_row["acquisition_group"],
                "culture_id": blank_row.get("culture_id", "unknown"),
                "disease": blank_row.get("disease", "unknown"),
                "tau_isoform": blank_row.get("tau_isoform", "unknown"),
                "experimental_condition": blank_row.get("experimental_condition", "unknown"),
                "div": blank_row.get("div", "unknown"),
                "div_token": blank_row.get("div_token", "unknown"),
                "experimental_group_id": blank_row.get("experimental_group_id", "unknown"),
                "crop_coordinates": [0, 0, int(arrays["blank_float"].shape[1]), int(arrays["blank_float"].shape[0])],
                "blank_status": blank_row.get("blank_status", "expert_validated"),
                "blank_pool_role": config.get("compositing", {}).get("blank_pool_role", config.get("compositing", {}).get("synthetic_split", "calibration")),
            },
            "synthetic_split": config.get("compositing", {}).get("synthetic_split", "calibration"),
            "float_to_uint8_mapping": mapping_stats,
            "clipping_count": int(mapping_stats["clipped_low_count"]) + int(mapping_stats["clipped_high_count"]),
            "clipping_fraction": float((int(mapping_stats["clipped_low_count"]) + int(mapping_stats["clipped_high_count"])) / arrays["render_uint8"].size),
            "saturation_count": int(mapping_stats["saturation_count"]),
            "saturation_fraction": float(int(mapping_stats["saturation_count"]) / arrays["render_uint8"].size),
            "array_names": sorted(arrays),
            "dtypes": {name: str(arr.dtype) for name, arr in arrays.items()},
            "shapes": {name: list(arr.shape) for name, arr in arrays.items()},
            "enum_mappings": {"node_type": NODE_TYPES},
            "alignment_statement": "structural targets are unchanged during real-blank compositing",
            "parent_synthetic_sample_id": (parent_metadata or {}).get("sample_id", "generated_in_memory"),
            "source_synthetic_artifact_hash": (parent_row or {}).get("npz_sha256", "not_recorded"),
            "composite_schema_version": CALIBRATION_SCHEMA_VERSION,
            "width_calibration": (parent_metadata or {}).get(
                "width_calibration", {}
            ),
        }
    )
    if render_report is not None:
        metadata.update(
            {
                "rendering_report": render_report,
                "target_available": render_report.get("target_available", {}),
                "target_provenance": {
                    "semantic_mask_source": render_report["semantic_mask_source"],
                    "centerline_source": "projected_3d_ground_truth",
                    "trace_source": "projected_3d_ground_truth",
                    "ignore_mask_rule": render_report.get("ignore_mask_rule", "none"),
                },
                "annotation_contract": {
                    "trace_arrays": ["trace_points_xy", "trace_point_offsets", "trace_ids", "trace_status", "trace_source"],
                    "coordinate_convention": "zero-based x_y pixel-equivalent coordinates matching planned JFilament conversion",
                    "synthetic_only_targets_optional_for_real_samples": ["fiber_points_xyz", "nearest_depth_map", "weighted_mean_depth_map", "in_focus_signal", "out_of_focus_signal"],
                    "reserved_real_semantic_classes": REAL_SEMANTIC_CLASSES,
                    "reserved_trace_termination_statuses": sorted(TRACE_TERMINATION_STATUSES),
                    "current_binary_semantic_mask": not is_morphology,
                    "multiclass_semantic_mask": is_morphology,
                    "binary_compatibility_mask": (
                        "union of semantic classes 1, 2, and 3"
                        if is_morphology
                        else "semantic_mask"
                    ),
                },
                "foreground_signal_convention": "arc-length weighted empirical line density convolved with unit-integral discrete PSF before real-blank addition",
            }
        )
    if is_morphology:
        for key in [
            "multiclass_target_contract",
            "graph_supervision",
            "target_roles",
        ]:
            if key in (parent_metadata or {}):
                metadata[key] = parent_metadata[key]
        if "enum_mappings" in (parent_metadata or {}):
            metadata["enum_mappings"] = parent_metadata["enum_mappings"]
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
        if meta.get("parent_synthetic_sample_id") == row["sample_id"]:
            errors.append(f"{row['sample_id']}: composite sample_id must differ from parent synthetic sample_id")
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        for name, arr in arrays.items():
            if arr.dtype == object:
                errors.append(f"{row['sample_id']}: object dtype prohibited for {name}")
        semantic_source = meta.get("target_provenance", {}).get("semantic_mask_source")
        if semantic_source == "semantic_class_union":
            expected_semantic = np.isin(
                arrays["semantic_class_mask"], [1, 2, 3]
            ).astype(np.uint8)
        elif semantic_source and semantic_source in arrays:
            expected_semantic = arrays[semantic_source].astype(np.uint8)
        else:
            expected_semantic = (arrays["overlap_count"] > 0).astype(np.uint8)
        if not np.array_equal(arrays["semantic_mask"], expected_semantic):
            errors.append(f"{row['sample_id']}: target alignment failure")
        if "in_focus_signal" in arrays and not np.allclose(arrays["in_focus_signal"] + arrays["out_of_focus_signal"], arrays["total_clean_signal"], atol=1e-4):
            errors.append(f"{row['sample_id']}: optical signal decomposition failure")
        blank_id = meta["source_blank_provenance"]["blank_stable_image_id"]
        split = meta["synthetic_split"]
        for other_split, ids in blank_by_split.items():
            if other_split != split and blank_id in ids:
                errors.append(f"{blank_id}: blank reused across splits without override")
        blank_by_split.setdefault(split, set()).add(blank_id)
    return errors
