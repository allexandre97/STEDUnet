#!/usr/bin/env python
"""Check real-compatible target views in synthetic morphology samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.synthetic.storage import set_worker_thread_limits
from fibras.synthetic.real_compatible import (
    build_real_compatible_targets,
    validate_real_compatible_targets,
)


def validate_row(dataset_dir: Path, row: dict[str, str]) -> list[str]:
    sample_id = row["sample_id"]
    with np.load(dataset_dir / row["npz_path"], allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    metadata = json.loads((dataset_dir / row["json_path"]).read_text(encoding="utf-8"))
    if "semantic_class_mask" not in arrays:
        return []
    check_arrays = dict(arrays)
    check_arrays.update(build_real_compatible_targets(arrays))
    return validate_real_compatible_targets(sample_id, check_arrays, metadata)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")

    manifest = args.dataset_dir / "dataset_manifest.csv"
    if not manifest.exists():
        print(f"{manifest}: missing dataset manifest", file=sys.stderr)
        return 1

    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    errors_by_index: dict[int, list[str]] = {}
    if args.num_workers == 1:
        errors_by_index = {
            index: validate_row(args.dataset_dir, row) for index, row in enumerate(rows)
        }
    else:
        set_worker_thread_limits()
        with ProcessPoolExecutor(
            max_workers=args.num_workers, initializer=set_worker_thread_limits
        ) as pool:
            futures = {
                pool.submit(validate_row, args.dataset_dir, row): index
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
    errors = [
        error
        for index in range(len(rows))
        for error in errors_by_index.get(index, [])
    ]

    if errors:
        print("Real-compatible synthetic target validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Real-compatible synthetic target validation passed for {len(rows)} samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
