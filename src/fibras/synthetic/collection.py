"""Cross-dataset validation for synthetic and blank-composite artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import (
    DATASET_SCHEMA_VERSION_3D,
    DATASET_SCHEMA_VERSION_3D_HARDENED,
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY,
    DATASET_SCHEMA_VERSION_3D_MORPHOLOGY_LEGACY,
)


def read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    path = dataset_dir / "dataset_manifest.csv"
    if not path.exists():
        raise ValueError(f"{path}: missing dataset manifest")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_dataset_collection(dataset_dirs: list[Path]) -> list[str]:
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
        errors.extend(validate_parent(sample_id, record, parent))
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
    if category != "realism_calibration":
        errors.append(
            f"{sample_id}: composite parent category must be realism_calibration, "
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
