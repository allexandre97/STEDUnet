#!/usr/bin/env python
"""Create a provisional blank-background pool manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibras.sted_blank_pools import assign_blank_pools, read_csv, validate_blank_pools, write_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rows = assign_blank_pools(read_csv(args.inventory_dir / "sted_blanks.csv"))
    errors = validate_blank_pools(rows)
    if errors:
        print("Blank-pool manifest validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    write_csv(args.out, rows)
    if args.report:
        write_report(args.report, rows)
    return 0


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    by_role: dict[str, int] = {}
    by_group: dict[str, str] = {}
    for row in rows:
        by_role[row["blank_pool_role"]] = by_role.get(row["blank_pool_role"], 0) + 1
        by_group[row["acquisition_group"]] = row["blank_pool_role"]
    lines = [
        "# STED blank-background pool report",
        "",
        "- Status: provisional; `human_approved=false`.",
        "- Blank status is expert-validated; pool assignment is for background-leakage control only.",
        "",
        "| Role | Images |",
        "|:--|--:|",
    ]
    for role, count in sorted(by_role.items()):
        lines.append(f"| `{role}` | {count} |")
    lines.extend(["", "| Acquisition group | Role |", "|:--|:--|"])
    for group, role in sorted(by_group.items()):
        lines.append(f"| `{group}` | `{role}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
