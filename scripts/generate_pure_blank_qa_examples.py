#!/usr/bin/env python
"""Generate pure-blank QA artifacts from expert-validated blank images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.sted_blank_pools import read_csv
from fibras.sted_inventory import read_image
from fibras.synthetic.schema import load_yaml
from fibras.synthetic.storage import sha256_file, write_dataset_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--pools", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    config = load_yaml(args.config)
    roots = config["source_roots"]
    blanks = {r["stable_image_id"]: r for r in read_csv(args.inventory_dir / "sted_blanks.csv")}
    pool_rows = [r for r in read_csv(args.pools) if r["blank_pool_role"] == "pure_blank_qa"]
    selected = [blanks[r["stable_image_id"]] for r in sorted(pool_rows, key=lambda r: r["stable_image_id"])[: args.count]]
    if not selected:
        raise SystemExit("no pure_blank_qa rows available")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected):
        sample_id = f"pure_blank_qa_{index:04d}"
        image, _, _ = read_image(Path(roots[row["source_root_id"]]) / row["relative_path"])
        arrays = blank_arrays(image)
        npz_path = args.out / f"{sample_id}.npz"
        json_path = args.out / f"{sample_id}.json"
        np.savez_compressed(npz_path, **arrays)
        metadata = {
            "sample_id": sample_id,
            "dataset_schema_version": "pure_blank_qa_0.1.0",
            "generator_version": "not_applicable",
            "calibration_status": config.get("calibration_data_status", "exploratory_unpartitioned"),
            "source_blank_provenance": {
                "source_root_id": row["source_root_id"],
                "blank_stable_image_id": row["stable_image_id"],
                "blank_relative_path": row["relative_path"],
                "blank_sha256": row["source_sha256"],
                "blank_acquisition_group": row["acquisition_group"],
                "crop_coordinates": [0, 0, int(image.shape[1]), int(image.shape[0])],
                "blank_status": row.get("blank_status", "expert_validated"),
                "blank_pool_role": "pure_blank_qa",
            },
            "target_available": {
                "semantic_mask": True,
                "centerline_mask": True,
                "ignore_mask": True,
                "distance_transform": False,
                "endpoint_map": True,
                "junction_map": True,
                "crossing_map": True,
                "orientation": False,
                "instance_membership": False,
                "depth_maps": False,
                "optical_signal_decomposition": False,
            },
            "expected_model_qa_targets": {
                "false_positive_foreground_fraction": "future_model_output_compared_to_zero_semantic_mask",
                "spurious_skeleton_length": "future_model_output_compared_to_zero_centerline_mask",
                "false_endpoints": "future_model_output_compared_to_zero_endpoint_map",
                "false_junctions": "future_model_output_compared_to_zero_junction_map",
            },
            "array_names": sorted(arrays),
            "dtypes": {name: str(arr.dtype) for name, arr in arrays.items()},
            "shapes": {name: list(arr.shape) for name, arr in arrays.items()},
            "integrity_checksum": {"npz_sha256": sha256_file(npz_path)},
        }
        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_rows.append(
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
    write_dataset_manifest(args.out / "dataset_manifest.csv", manifest_rows)
    return 0


def blank_arrays(image: np.ndarray) -> dict[str, np.ndarray]:
    render = image.astype(np.uint8)
    zeros = np.zeros(render.shape, dtype=np.uint8)
    return {
        "blank_float": image.astype(np.float32),
        "render_float": image.astype(np.float32),
        "render_uint8": render,
        "semantic_mask": zeros.copy(),
        "centerline_mask": zeros.copy(),
        "ignore_mask": zeros.copy(),
        "endpoint_map": zeros.copy(),
        "junction_map": zeros.copy(),
        "crossing_map": zeros.copy(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
