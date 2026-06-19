#!/usr/bin/env python
"""Benchmark bounded 3D synthetic STED generation cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.synthetic.rasterizer3d import (
    background_distance_to_foreground,
    clear_width_calibration_cache,
    measure_isolated_fiber_fwhm,
    rasterize_3d_sample,
)
from fibras.synthetic.geometry3d import generate_persistent_chain_geometry
from fibras.synthetic.schema import load_yaml
from fibras.synthetic.storage import build_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    base = load_yaml(args.config)
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[256:768, 256:768] = 1
    start = time.perf_counter()
    background_distance_to_foreground(mask)
    edt_seconds = time.perf_counter() - start
    clear_width_calibration_cache()
    start = time.perf_counter()
    width = measure_isolated_fiber_fwhm(base)
    width_seconds = time.perf_counter() - start
    width_config = base.get("width_calibration", {})
    width_by_orientation = {
        str(angle): measure_isolated_fiber_fwhm(base, angle_degrees=float(angle))[
            "measured_fwhm_px"
        ]
        for angle in width_config.get("test_orientations_degrees", [])
    }
    width_by_subpixel_offset = {
        ",".join(map(str, offset)): measure_isolated_fiber_fwhm(
            base, subpixel_shift=tuple(offset)
        )["measured_fwhm_px"]
        for offset in width_config.get("test_subpixel_offsets", [])
    }
    full_width_config = json.loads(json.dumps(base))
    full_width_config["geometry"]["scenario"] = "straight_width_calibration"
    full_width_config["geometry"].pop("sample_scenarios", None)
    geometry = generate_persistent_chain_geometry(full_width_config["geometry"], 0)
    start = time.perf_counter()
    rasterize_3d_sample(
        geometry,
        full_width_config.get("targets", {}),
        full_width_config.get("optical_model", {}),
        full_width_config.get("output_mapping", {}),
        0,
    )
    full_width_seconds = time.perf_counter() - start
    cases = {
        "sparse_1024": {"scenario": "sparse_near_planar"},
        "dense_1024": {"scenario": "dense_local_geometry"},
        "structural_qa_1024": {"scenario": "projected_depth_crossing", "scenario_category": "structural_qa"},
    }
    rows = []
    for name, overrides in cases.items():
        cfg = json.loads(json.dumps(base))
        cfg["dataset_name"] = f"benchmark_{name}"
        cfg["sample_count"] = 8
        cfg["geometry"]["scenario"] = overrides["scenario"]
        cfg["geometry"].pop("sample_scenarios", None)
        cfg["targets"]["scenario_category"] = overrides.get("scenario_category", "realism_calibration")
        tracemalloc.start()
        start = time.perf_counter()
        _, _, metadata = build_sample(cfg, 0)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        render = metadata["rendering_report"]
        rows.append(
            {
                "case": name,
                "wall_time_seconds": elapsed,
                "peak_memory_mb": peak / (1024 * 1024),
                "crossing_detection_seconds": render.get("crossing_detection", {}).get("elapsed_seconds", 0.0),
                "distance_transform_seconds": render.get("distance_transform_elapsed_seconds", 0.0),
                "kernel_count": render.get("kernel_count", 0),
                "candidate_pairs": render.get("crossing_detection", {}).get("candidate_pair_count", 0),
                "brute_force_pairs": render.get("crossing_detection", {}).get("brute_force_pair_count", 0),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "distance_transform_1024_seconds": edt_seconds,
        "distance_transform_backend": "scipy_ndimage_distance_transform_edt",
        "optical_only_width_calibration_seconds": width_seconds,
        "full_pipeline_width_fixture_seconds": full_width_seconds,
        "width_calibration": width,
        "width_fwhm_by_orientation_degrees": width_by_orientation,
        "width_fwhm_by_subpixel_offset": width_by_subpixel_offset,
        "benchmarks": rows,
    }
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
