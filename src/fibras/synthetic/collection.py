"""Cross-dataset validation for synthetic and blank-composite artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from .schema import (
    DATASET_SCHEMA_VERSION_3D,
    DATASET_SCHEMA_VERSION_3D_HARDENED,
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY,
)
from .storage import set_worker_thread_limits


COMPOSITE_PARENT_SCENARIO_CATEGORIES = {
    "realism_calibration",
    "clump_ignore_stress",
}


def read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    path = dataset_dir / "dataset_manifest.csv"
    if not path.exists():
        raise ValueError(f"{path}: missing dataset manifest")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_dataset_collection(dataset_dirs: list[Path], num_workers: int = 1) -> list[str]:
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    errors: list[str] = []
    records: dict[str, list[dict[str, Any]]] = {}
    namespaces: dict[str, Path] = {}
    for dataset_dir in dataset_dirs:
        try:
            rows = read_manifest(dataset_dir)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for row in rows:
            metadata_path = dataset_dir / row["json_path"]
            if not metadata_path.exists():
                errors.append(f"{metadata_path}: missing metadata")
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            record = {
                "dir": dataset_dir,
                "row": row,
                "metadata": metadata,
                "npz": dataset_dir / row["npz_path"],
            }
            records.setdefault(row["sample_id"], []).append(record)
            namespace = dataset_namespace(metadata, row["sample_id"])
            owner = namespaces.get(namespace)
            if owner is not None and owner != dataset_dir:
                errors.append(
                    f"dataset namespace collision: {namespace} is used by "
                    f"{owner} and {dataset_dir}"
                )
            namespaces[namespace] = dataset_dir
    for sample_id, matches in records.items():
        if len(matches) != 1:
            locations = ", ".join(str(match["dir"]) for match in matches)
            errors.append(
                f"global sample_id collision: {sample_id} appears in {locations}"
            )
    unique = {
        sample_id: matches[0]
        for sample_id, matches in records.items()
        if len(matches) == 1
    }
    parent_checks: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for sample_id, record in unique.items():
        metadata = record["metadata"]
        parent_id = metadata.get("parent_synthetic_sample_id")
        if not parent_id or parent_id == "generated_in_memory":
            continue
        if parent_id == sample_id:
            errors.append(f"{sample_id}: composite cannot be its own parent")
            continue
        parent = unique.get(parent_id)
        if parent is None:
            errors.append(f"{sample_id}: unresolved parent {parent_id}")
            continue
        parent_checks.append((sample_id, record, parent))
    if num_workers == 1:
        for sample_id, record, parent in parent_checks:
            errors.extend(validate_parent(sample_id, record, parent))
    else:
        set_worker_thread_limits()
        errors_by_index: dict[int, list[str]] = {}
        with ProcessPoolExecutor(
            max_workers=num_workers, initializer=set_worker_thread_limits
        ) as pool:
            futures = {
                pool.submit(validate_parent, sample_id, record, parent): index
                for index, (sample_id, record, parent) in enumerate(parent_checks)
            }
            for future in as_completed(futures):
                index = futures[future]
                sample_id = parent_checks[index][0]
                try:
                    errors_by_index[index] = future.result()
                except Exception as exc:
                    errors_by_index[index] = [
                        f"{sample_id}: validator worker failed: {type(exc).__name__}: {exc}"
                    ]
        for index in range(len(parent_checks)):
            errors.extend(errors_by_index.get(index, []))
    return errors


def dataset_namespace(metadata: dict[str, Any], sample_id: str) -> str:
    config = metadata.get("generation_config", metadata.get("configuration", {}))
    composite_name = (
        config.get("compositing", {}).get("composite_dataset_name")
        if metadata.get("parent_synthetic_sample_id")
        else None
    )
    return str(composite_name or config.get("dataset_name") or sample_id.rsplit("_", 1)[0])


def validate_parent(
    sample_id: str,
    composite: dict[str, Any],
    parent: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    composite_meta = composite["metadata"]
    parent_meta = parent["metadata"]
    if parent_meta.get("parent_synthetic_sample_id"):
        errors.append(f"{sample_id}: parent must not itself be a composite")
    expected_hash = composite_meta.get("source_synthetic_artifact_hash")
    actual_hash = parent["row"].get("npz_sha256")
    if not expected_hash or expected_hash != actual_hash:
        errors.append(f"{sample_id}: parent NPZ SHA-256 mismatch")
    compatible = {
        DATASET_SCHEMA_VERSION_3D_HARDENED,
        DATASET_SCHEMA_VERSION_3D,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY,
        DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
    }
    parent_schema = parent_meta.get("dataset_schema_version")
    composite_schema = composite_meta.get("dataset_schema_version")
    if parent_schema not in compatible or composite_schema != parent_schema:
        errors.append(
            f"{sample_id}: incompatible parent/composite schemas "
            f"{parent_schema!r} and {composite_schema!r}"
        )
    category = parent_meta.get(
        "scenario_category",
        parent_meta.get("rendering_report", {}).get("scenario_category"),
    )
    if category not in COMPOSITE_PARENT_SCENARIO_CATEGORIES:
        errors.append(
            f"{sample_id}: composite parent category must be one of "
            f"{sorted(COMPOSITE_PARENT_SCENARIO_CATEGORIES)}, "
            f"not {category!r}"
        )
    if not parent["npz"].exists() or not composite["npz"].exists():
        errors.append(f"{sample_id}: parent or composite NPZ is missing")
        return errors
    excluded = {"render_float", "render_uint8"}
    with np.load(parent["npz"], allow_pickle=False) as parent_data, np.load(
        composite["npz"], allow_pickle=False
    ) as composite_data:
        for name in sorted(set(parent_data.files) - excluded):
            if name not in composite_data.files:
                errors.append(f"{sample_id}: inherited parent array {name} is missing")
            elif not np.array_equal(parent_data[name], composite_data[name]):
                errors.append(
                    f"{sample_id}: inherited parent array {name} was modified"
                )
    return errors
