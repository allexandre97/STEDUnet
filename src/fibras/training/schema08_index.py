"""Index schema-0.8 blank-composited samples for first-pass training."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from fibras.synthetic.storage import set_worker_thread_limits


FIELDS = [
    "sample_id",
    "npz_path",
    "metadata_path",
    "schema_version",
    "generator_version",
    "parent_synthetic_sample_id",
    "source_blank_id",
    "source_synthetic_split",
    "split",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def repo_relative(path: Path, root: Path | None = None) -> str:
    root = (root or Path.cwd()).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_training_manifest(
    dataset_dir: Path,
    out_path: Path,
    train_fraction: float = 0.8,
    root: Path | None = None,
    num_workers: int = 1,
) -> list[dict[str, str]]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    dataset_manifest = dataset_dir / "dataset_manifest.csv"
    rows = sorted(read_csv(dataset_manifest), key=lambda r: r["sample_id"])
    train_count = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
    if num_workers == 1:
        indexed = [
            index_manifest_row(dataset_dir, row, i, train_count, root)
            for i, row in enumerate(rows)
        ]
    else:
        set_worker_thread_limits()
        rows_by_index: dict[int, dict[str, str]] = {}
        with ProcessPoolExecutor(
            max_workers=num_workers, initializer=set_worker_thread_limits
        ) as pool:
            futures = {
                pool.submit(
                    index_manifest_row, dataset_dir, row, i, train_count, root
                ): i
                for i, row in enumerate(rows)
            }
            for future in as_completed(futures):
                rows_by_index[futures[future]] = future.result()
        indexed = [rows_by_index[i] for i in range(len(rows))]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(indexed)
    return indexed


def index_manifest_row(
    dataset_dir: Path,
    row: dict[str, str],
    index: int,
    train_count: int,
    root: Path | None,
) -> dict[str, str]:
    metadata_path = dataset_dir / row["json_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    blank = metadata.get("source_blank_provenance", {})
    return {
        "sample_id": row["sample_id"],
        "npz_path": repo_relative(dataset_dir / row["npz_path"], root),
        "metadata_path": repo_relative(metadata_path, root),
        "schema_version": row.get("schema_version") or metadata.get("dataset_schema_version", "unknown"),
        "generator_version": row.get("generator_version") or metadata.get("generator_version", "unknown"),
        "parent_synthetic_sample_id": metadata.get("parent_synthetic_sample_id", "not_recorded"),
        "source_blank_id": blank.get("blank_stable_image_id", "not_recorded"),
        "source_synthetic_split": metadata.get("synthetic_split", "not_recorded"),
        "split": "train" if index < train_count else "validation",
    }
