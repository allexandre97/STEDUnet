#!/usr/bin/env python
"""Build the local schema-0.8 first-training manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.training.schema08_index import build_training_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        "--input",
        dest="dataset_dir",
        type=Path,
        default=Path("examples/sted_blank_composites_training_candidate_schema08"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data_manifests/training_candidate_schema08.csv"),
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    rows = build_training_manifest(
        args.dataset_dir,
        args.out,
        args.train_fraction,
        num_workers=args.num_workers,
    )
    train = sum(row["split"] == "train" for row in rows)
    validation = sum(row["split"] == "validation" for row in rows)
    print(f"Wrote {args.out}: train={train}, validation={validation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
