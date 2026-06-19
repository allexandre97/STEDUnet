#!/usr/bin/env python
"""Benchmark bounded 3D synthetic STED generation cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.synthetic.schema import load_yaml
from fibras.synthetic.storage import build_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    base = load_yaml(args.config)
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
    args.out.write_text(json.dumps({"benchmarks": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"benchmarks": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
