"""Dedicated blank-background pool assignment for synthetic rendering."""

from __future__ import annotations

import csv
from pathlib import Path


BLANK_POOL_ROLES = [
    "synthetic_background_train",
    "synthetic_background_validation",
    "synthetic_background_test",
    "pure_blank_qa",
]


FIELDS = [
    "stable_image_id",
    "source_root_id",
    "relative_path",
    "source_sha256",
    "acquisition_group",
    "blank_pool_role",
    "assignment_reason",
    "human_approved",
    "blank_status",
    "validation_source",
    "validator_role",
    "validation_date",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def assign_blank_pools(blank_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in blank_rows:
        group = row.get("acquisition_group") or row["stable_image_id"]
        groups.setdefault(group, []).append(row)
    assignments: list[dict[str, str]] = []
    for i, group in enumerate(sorted(groups)):
        role = BLANK_POOL_ROLES[i] if i < len(BLANK_POOL_ROLES) else "synthetic_background_train"
        for row in sorted(groups[group], key=lambda r: r["stable_image_id"]):
            assignments.append(
                {
                    "stable_image_id": row["stable_image_id"],
                    "source_root_id": row["source_root_id"],
                    "relative_path": row["relative_path"],
                    "source_sha256": row["source_sha256"],
                    "acquisition_group": group,
                    "blank_pool_role": role,
                    "assignment_reason": "deterministic acquisition-group allocation for synthetic background and pure-blank QA pools",
                    "human_approved": "false",
                    "blank_status": row.get("blank_status", "expert_validated"),
                    "validation_source": row.get("validation_source", "human_expert_review"),
                    "validator_role": row.get("validator_role", "STED expert"),
                    "validation_date": row.get("validation_date", "not_recorded"),
                    "notes": row.get("notes", ""),
                }
            )
    return assignments


def validate_blank_pools(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    by_id: dict[str, str] = {}
    by_group: dict[str, str] = {}
    for row in rows:
        stable_id = row["stable_image_id"]
        role = row["blank_pool_role"]
        group = row["acquisition_group"]
        if role not in BLANK_POOL_ROLES:
            errors.append(f"{stable_id}: invalid blank_pool_role {role}")
        if stable_id in by_id and by_id[stable_id] != role:
            errors.append(f"{stable_id}: assigned to multiple blank-pool roles")
        by_id[stable_id] = role
        if group in by_group and by_group[group] != role:
            errors.append(f"{group}: acquisition group split across blank-pool roles")
        by_group[group] = role
        if row.get("blank_status") != "expert_validated":
            errors.append(f"{stable_id}: blank_status must preserve expert_validated provenance")
        if row.get("human_approved") != "false":
            errors.append(f"{stable_id}: blank-pool allocation must remain human_approved=false")
    if len(set(by_group)) >= len(BLANK_POOL_ROLES):
        missing = sorted(set(BLANK_POOL_ROLES) - set(by_id.values()))
        if missing:
            errors.append(f"blank-pool roles unexpectedly empty: {missing}")
    return errors
