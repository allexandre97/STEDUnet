#!/usr/bin/env python
"""Validate STED inventory manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.sted_inventory import BLANK_EXTRA_FIELDS, IMAGE_FIELDS, sha256_file


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_manifest(path: Path, expected_fields: list[str]) -> list[str]:
    errors: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [field for field in expected_fields if field not in (reader.fieldnames or [])]
        if missing:
            errors.append(f"{path}: missing fields {missing}")
        rows = list(reader)
    ids = [r["stable_image_id"] for r in rows if "stable_image_id" in r]
    if len(ids) != len(set(ids)):
        errors.append(f"{path}: duplicate stable_image_id")
    for row in rows:
        if not row.get("source_sha256"):
            errors.append(f"{path}: missing source_sha256 for {row.get('relative_path')}")
        if row.get("source_kind") == "blank_background":
            for field, value in [
                ("blank_status", "expert_validated"),
                ("validation_source", "human_expert_review"),
                ("validator_role", "STED expert"),
                ("validation_date", "not_recorded"),
            ]:
                if row.get(field) != value:
                    errors.append(f"{row.get('stable_image_id')}: invalid blank provenance {field}={row.get(field)}")
    return errors


def validate_paths(root: Path, rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    seen = set()
    for row in rows:
        rel = row["relative_path"]
        if rel in seen:
            errors.append(f"duplicate relative_path: {rel}")
        seen.add(rel)
        path = root / rel
        if not path.exists():
            errors.append(f"missing source file: {path}")
            continue
        if sha256_file(path) != row["source_sha256"]:
            errors.append(f"checksum mismatch: {path}")
    source_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    missing_rows = sorted(source_files - seen)
    if missing_rows:
        errors.append(f"source files absent from manifest: {missing_rows[:5]}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fiber-dir", type=Path)
    parser.add_argument("--blank-dir", type=Path)
    parser.add_argument("--manifests", required=True, type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    images_path = args.manifests / "sted_images.csv"
    blanks_path = args.manifests / "sted_blanks.csv"
    groups_path = args.manifests / "acquisition_groups.csv"
    for path in [images_path, blanks_path, groups_path]:
        if not path.exists():
            errors.append(f"missing required manifest: {path}")
    if not errors:
        images = read_csv(images_path)
        blanks = read_csv(blanks_path)
        errors.extend(validate_manifest(images_path, IMAGE_FIELDS))
        errors.extend(validate_manifest(blanks_path, IMAGE_FIELDS + BLANK_EXTRA_FIELDS))
        if args.fiber_dir:
            errors.extend(validate_paths(args.fiber_dir, images))
        if args.blank_dir:
            errors.extend(validate_paths(args.blank_dir, blanks))
    if errors:
        print("STED inventory validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("STED inventory validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

