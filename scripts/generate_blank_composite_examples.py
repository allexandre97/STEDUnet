#!/usr/bin/env python
"""Generate bounded exploratory real-blank composite examples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.calibration.compositing import generate_composites
from fibras.synthetic.schema import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-count-override", type=int)
    parser.add_argument("--parent-synthetic-dir-override", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.parent_synthetic_dir_override is not None:
        config = dict(config)
        config["compositing"] = dict(config.get("compositing", {}))
        config["compositing"]["parent_synthetic_dir"] = str(args.parent_synthetic_dir_override)
    generate_composites(
        config,
        args.inventory_dir,
        args.splits,
        args.out,
        num_workers=args.num_workers,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
        sample_count_override=args.sample_count_override,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
