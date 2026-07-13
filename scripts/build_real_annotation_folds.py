#!/usr/bin/env python
"""Build fixed grouped cross-validation folds for real annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fibras.training.cross_validation import (
    build_grouped_folds,
    fold_summary,
    read_csv,
    write_folds,
)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        rows = build_grouped_folds(
            read_csv(args.inventory),
            fold_count=args.fold_count,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
        )
        summary = fold_summary(rows)
        fold_bytes, json_bytes, markdown_bytes = render(rows, summary)
        outputs = (
            (args.out, fold_bytes),
            (args.summary_json, json_bytes),
            (args.summary_markdown, markdown_bytes),
        )
        if args.check:
            stale = [str(path) for path, content in outputs if not path.exists() or path.read_bytes() != content]
            if stale:
                raise ValueError(f"fold artifacts are stale or missing: {stale}")
        else:
            for path, content in outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{'validated' if args.check else 'wrote'} {args.fold_count} fixed grouped folds ({len(rows)} rows)")
    return 0


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    out.add_argument("--inventory", type=Path, default=Path("data_manifests/real_annotations.csv"))
    out.add_argument("--out", type=Path, default=Path("data_manifests/real_annotation_folds_v1.csv"))
    out.add_argument("--summary-json", type=Path, default=Path("data_manifests/real_annotation_folds_v1_summary.json"))
    out.add_argument("--summary-markdown", type=Path, default=Path("docs/real_annotation_folds_v1.md"))
    out.add_argument("--fold-count", type=int, default=5)
    out.add_argument("--seed", type=int, default=20260713)
    out.add_argument("--validation-fraction", type=float, default=0.2)
    out.add_argument("--check", action="store_true")
    return out


def render(rows: list[dict[str, str]], summary: dict) -> tuple[bytes, bytes, bytes]:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "folds.csv"
        write_folds(csv_path, rows)
        return (
            csv_path.read_bytes(),
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
            markdown_summary(summary).encode(),
        )


def markdown_summary(summary: dict) -> str:
    lines = [
        "# Real Annotation Grouped Cross-Validation Folds v1",
        "",
        "Five fixed outer folds use `experimental_group_id` as the provisional indivisible sample group. Each outer development set has a grouped validation partition used only for checkpoint selection.",
        "",
        "Perfect stratification is impossible: CBD has one four-image group, PID has two groups, and multiple disease/isoform/DIV strata occur in only one group. Five folds remain defensible because 12 groups can populate five nonempty test partitions with four images each or close to it.",
        "",
        "Two images with semantic overlaps remain assigned so every inventory image appears in outer test exactly once, but `validation_status=invalid` excludes them from training and evaluation until corrected.",
        "",
        "| Fold | Train images/groups | Validation images/groups | Test images/groups | Test diseases | Test density | Invalid test |",
        "|--:|:--|:--|:--|:--|:--|--:|",
    ]
    for fold, partitions in summary["folds"].items():
        train, validation, test = (partitions[name] for name in ("train", "validation", "test"))
        diseases = ", ".join(f"{name}:{count}" for name, count in test["disease"].items())
        density = ", ".join(f"{name}:{count}" for name, count in test["morphology_density"].items())
        lines.append(
            f"| {fold} | {train['images']}/{train['groups']} | {validation['images']}/{validation['groups']} | "
            f"{test['images']}/{test['groups']} | {diseases} | {density} | {test['invalid_images']} |"
        )
    lines += [
        "", "## Leakage Contract", "",
        "- Every image occurs in the outer test partition exactly once.",
        "- Every `split_group_id` is wholly train, validation, or test within a fold.",
        "- Crop manifests must inherit the parent image partition from this file.",
        "- Outer test rows are never used for checkpoint selection.",
        "- Fold generation is deterministic and checked byte-for-byte; experiments consume the committed manifest rather than regenerating folds.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
