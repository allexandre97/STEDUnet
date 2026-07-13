#!/usr/bin/env python
"""Build baseline-compatible real annotation crop datasets."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fibras.training.real_crops import (
    apply_fold_assignments,
    build_real_crop_dataset,
    discover_real_annotation_triplets,
    read_real_annotation_manifest,
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.annotation_dir is None) == (args.manifest is None):
        print("error: provide exactly one of --annotation-dir or --manifest", file=sys.stderr)
        return 2
    try:
        records = (
            discover_real_annotation_triplets(args.annotation_dir)
            if args.annotation_dir is not None
            else read_real_annotation_manifest(
                args.manifest,
                image_root=args.image_root,
                annotation_root=args.annotation_root,
            )
        )
        if (args.fold_manifest is None) != (args.outer_fold is None):
            raise ValueError("--fold-manifest and --outer-fold must be provided together")
        if args.fold_manifest is not None:
            records = apply_fold_assignments(records, args.fold_manifest, args.outer_fold)
        rows = build_real_crop_dataset(
            records,
            args.out,
            crop_size=args.crop_size,
            stride=args.stride,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            leave_one_out_image=args.leave_one_out_image,
            min_labeled_fraction=args.min_labeled_fraction,
        )
        crop_manifest = args.out / "real_crop_manifest.csv"
        if args.output_manifest is not None:
            write_output_manifest(rows, args.out, args.output_manifest)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} crops")
    print(f"wrote {args.out / 'real_crop_manifest.csv'}")
    if args.output_manifest is not None:
        print(f"wrote {args.output_manifest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--leave-one-out-image")
    parser.add_argument("--min-labeled-fraction", type=float, default=0.0)
    return parser


def write_output_manifest(rows: list[dict[str, str]], crop_root: Path, output_manifest: Path) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    rewritten = []
    for row in rows:
        out_row = row.copy()
        for key in ["npz_path", "json_path"]:
            source_path = crop_root / row[key]
            out_row[key] = relative_path_for_manifest(source_path, output_manifest)
        rewritten.append(out_row)
    with output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rewritten[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rewritten)


def relative_path_for_manifest(path: Path, manifest_path: Path) -> str:
    return os.path.relpath(path, manifest_path.parent).replace(os.sep, "/")


if __name__ == "__main__":
    raise SystemExit(main())
