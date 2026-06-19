"""Schema and provenance helpers for exploratory calibration artifacts."""

from __future__ import annotations

from datetime import date
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


CALIBRATION_SCHEMA_VERSION = "sted_appearance_calibration_0.1.0"
CALIBRATION_DATA_STATUSES = {
    "exploratory_unpartitioned",
    "exploratory_provisional_split",
    "approved_calibration_subset",
    "held_out_excluded",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_provenance() -> tuple[str, bool | str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        return commit, bool(status)
    except Exception:
        return "not_available", "not_available"


def code_version() -> str:
    commit, dirty = git_provenance()
    return f"{commit}+dirty" if dirty is True else commit


def sha256_json(data: dict[str, Any]) -> str:
    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    existing = [path for path in paths if path.exists()]
    if not existing:
        return "missing"
    for path in existing:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def manifest_versions(inventory_dir: Path, splits_path: Path) -> dict[str, str]:
    paths = {
        "sted_images_csv_sha256": inventory_dir / "sted_images.csv",
        "sted_blanks_csv_sha256": inventory_dir / "sted_blanks.csv",
        "acquisition_groups_csv_sha256": inventory_dir / "acquisition_groups.csv",
        "sted_splits_csv_sha256": splits_path,
    }
    return {name: sha256_file(path) if path.exists() else "missing" for name, path in paths.items()}


def artifact_metadata(
    *,
    inventory_dir: Path,
    splits_path: Path,
    config: dict[str, Any],
    source_image_ids: list[str],
    calibration_data_status: str,
    artifact_kind: str,
) -> dict[str, Any]:
    if calibration_data_status not in CALIBRATION_DATA_STATUSES:
        raise ValueError(f"invalid calibration_data_status: {calibration_data_status}")
    commit, dirty = git_provenance()
    manifests = manifest_versions(inventory_dir, splits_path)
    blank_pool_path = Path(
        config.get("compositing", {}).get(
            "blank_pool_manifest", inventory_dir / "sted_blank_pools.csv"
        )
    )
    return {
        "artifact_schema_version": CALIBRATION_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "source_manifest_version": manifests,
        "split_manifest_version": manifests["sted_splits_csv_sha256"],
        "source_image_ids": source_image_ids,
        "calibration_data_status": calibration_data_status,
        "calibration_status": calibration_data_status,
        "configuration": config,
        "code_version": code_version(),
        "source_commit_sha": commit,
        "working_tree_dirty": dirty,
        "generation_config_sha256": sha256_json(config),
        "inventory_manifest_sha256": sha256_paths(
            [
                inventory_dir / "sted_images.csv",
                inventory_dir / "sted_blanks.csv",
                inventory_dir / "acquisition_groups.csv",
            ]
        ),
        "split_manifest_sha256": sha256_file(splits_path)
        if splits_path.exists()
        else "missing",
        "blank_pool_manifest_sha256": sha256_file(blank_pool_path)
        if blank_pool_path.exists()
        else "missing",
        "date_generated": date.today().isoformat(),
        "result_status": "exploratory" if calibration_data_status.startswith("exploratory") else "approved",
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
