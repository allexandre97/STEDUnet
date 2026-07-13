#!/usr/bin/env python
"""Report synthetic bundle width, coherence, and clump-confusion diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fibras.training.bundle_diagnostics import diagnostics_for_manifest, write_bundle_diagnostics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = diagnostics_for_manifest(args.manifest, args.limit)
        write_bundle_diagnostics(rows, args.out)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out / 'summary.csv'}")
    print(f"wrote {args.out / 'summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
