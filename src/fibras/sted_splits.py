"""Manifest-driven STED split assignment and validation."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path


SPLIT_FIELDS = [
    "stable_image_id",
    "source_kind",
    "biological_group_candidate",
    "acquisition_group",
    "inferred_condition",
    "inferred_div",
    "inferred_round",
    "series_index",
    "eligibility",
    "primary_metric_role",
    "secondary_image_eval_role",
    "assignment_reason",
    "grouping_rule",
    "human_approved",
    "synthetic_split",
    "override_status",
    "override_reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SPLIT_FIELDS})


def load_inventory(inventory_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return read_csv(inventory_dir / "sted_images.csv"), read_csv(inventory_dir / "sted_blanks.csv")


def create_splits(inventory_dir: Path, strategy: str = "pn_holdout") -> list[dict[str, str]]:
    if strategy != "pn_holdout":
        raise ValueError(f"unsupported strategy: {strategy}")
    images, blanks = load_inventory(inventory_dir)
    records = images + blanks
    role_by_pn = allocate_pn_roles(records)
    rows = [
        split_row(
            rec,
            role_by_pn[rec["biological_group_candidate"]],
            assignment_reason(role_by_pn[rec["biological_group_candidate"]]),
            "pn_holdout_primary",
            synthetic_split=role_by_pn[rec["biological_group_candidate"]] if rec["source_kind"] == "blank_background" else "not_applicable",
        )
        for rec in records
    ]
    return sorted(rows, key=lambda r: (r["biological_group_candidate"], r["source_kind"], r["acquisition_group"], r["stable_image_id"]))


def allocate_pn_roles(records: list[dict[str, str]]) -> dict[str, str]:
    """Assign one primary role per PN.

    The allocation is intentionally conservative and deterministic. With too few
    independent PNs, not every role is populated; the report warns about this.
    """

    pns = sorted({r["biological_group_candidate"] for r in records})
    if not pns:
        return {}
    if len(pns) == 1:
        return {pns[0]: "calibration"}
    if len(pns) == 2:
        return {pns[0]: "calibration", pns[1]: "held_out_test"}
    if len(pns) == 3:
        return {pns[0]: "training", pns[1]: "validation", pns[2]: "held_out_test"}
    roles = ["training", "validation", "held_out_test", "calibration"]
    allocation: dict[str, str] = {}
    for idx, pn in enumerate(pns):
        allocation[pn] = roles[idx] if idx < len(roles) else "training"
    return allocation


def split_row(rec: dict[str, str], eligibility: str, reason: str, rule: str, synthetic_split: str) -> dict[str, str]:
    return {
        "stable_image_id": rec["stable_image_id"],
        "source_kind": rec["source_kind"],
        "biological_group_candidate": rec["biological_group_candidate"],
        "acquisition_group": rec["acquisition_group"],
        "inferred_condition": rec.get("inferred_condition", "unknown"),
        "inferred_div": rec.get("inferred_div", "unknown"),
        "inferred_round": rec.get("inferred_round", "unknown"),
        "series_index": rec.get("series_index", "unknown"),
        "eligibility": eligibility,
        "primary_metric_role": eligibility,
        "secondary_image_eval_role": "not_assigned",
        "assignment_reason": reason,
        "grouping_rule": rule,
        "human_approved": "false",
        "synthetic_split": synthetic_split,
        "override_status": "none",
        "override_reason": "none",
    }


def assignment_reason(role: str) -> str:
    if role == "held_out_test":
        return "provisional PN-level primary held-out test allocation"
    if role == "calibration":
        return "provisional PN-level calibration allocation"
    if role == "training":
        return "provisional PN-level training allocation"
    if role == "validation":
        return "provisional PN-level validation allocation"
    return "provisional PN-level allocation"


def validate_splits(inventory_dir: Path, splits_path: Path) -> list[str]:
    images, blanks = load_inventory(inventory_dir)
    inventory = {r["stable_image_id"]: r for r in images + blanks}
    rows = read_csv(splits_path)
    errors: list[str] = []
    split_ids = [r["stable_image_id"] for r in rows]
    if len(split_ids) != len(set(split_ids)):
        errors.append("duplicate stable_image_id in split manifest")
    missing = sorted(set(inventory) - set(split_ids))
    extra = sorted(set(split_ids) - set(inventory))
    if missing:
        errors.append(f"missing split rows: {missing[:5]}")
    if extra:
        errors.append(f"unknown split rows: {extra[:5]}")
    for row in rows:
        if row.get("human_approved") != "false":
            errors.append(f"{row['stable_image_id']}: MVP splits must remain human_approved=false")
        if row.get("primary_metric_role") != row.get("eligibility"):
            errors.append(f"{row['stable_image_id']}: primary_metric_role must equal eligibility")
        if row.get("grouping_rule") != "pn_holdout_primary":
            errors.append(f"{row['stable_image_id']}: grouping_rule must be pn_holdout_primary")
    errors.extend(check_pn_leakage(rows))
    errors.extend(check_group_leakage(rows, "fiber_image", "acquisition_group"))
    errors.extend(check_group_leakage(rows, "blank_background", "acquisition_group"))
    for row in rows:
        if (
            row["source_kind"] == "fiber_image"
            and row["eligibility"] == "held_out_test"
            and "holdout" not in row["assignment_reason"]
            and "held-out" not in row["assignment_reason"]
        ):
            errors.append(f"{row['stable_image_id']}: held_out_test lacks holdout reason")
        if row["source_kind"] == "blank_background" and row["synthetic_split"] != row["eligibility"]:
            errors.append(f"{row['stable_image_id']}: blank synthetic_split does not match eligibility")
        if row.get("secondary_image_eval_role", "not_assigned") not in {
            "not_assigned",
            "secondary_training",
            "secondary_validation",
            "secondary_test",
        }:
            errors.append(f"{row['stable_image_id']}: invalid secondary_image_eval_role")
    return errors


def check_pn_leakage(rows: list[dict[str, str]]) -> list[str]:
    by_pn: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_pn[row["biological_group_candidate"]].add(row["eligibility"])
    errors = []
    for pn, roles in by_pn.items():
        if len(roles) != 1:
            errors.append(f"PN {pn} crosses primary roles: {sorted(roles)}")
    return errors


def check_group_leakage(rows: list[dict[str, str]], source_kind: str, group_field: str) -> list[str]:
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["source_kind"] == source_kind:
            by_group[row[group_field]].add(row["eligibility"])
    errors = []
    for group, roles in by_group.items():
        if len(roles) > 1:
            errors.append(f"{source_kind} {group_field}={group} crosses roles: {sorted(roles)}")
    return errors


def write_split_report(splits_path: Path, report_path: Path, min_pns_per_stratum: int = 3) -> None:
    rows = read_csv(splits_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# STED PN-level split report",
        "",
        "## Policy",
        "",
        "- `PN###` is treated as a biological sample and the indivisible primary partitioning unit.",
        "- Condition and DIV are stratification/reporting variables, not grouping boundaries.",
        "- Round and series are retained as metadata and reporting variables.",
        "- The primary held-out test contains only PNs absent from calibration, training, and validation.",
        "- `human_approved=false` is retained until this allocation and the stratum counts are reviewed.",
        "- Secondary image-level evaluation roles, if used later, must not be mixed with the primary PN-held-out metric.",
        "",
    ]
    pns_all = {r["biological_group_candidate"] for r in rows}
    roles_all = {r["eligibility"] for r in rows}
    if len(pns_all) < 4 or not {"training", "validation", "held_out_test"} <= roles_all:
        lines.extend(
            [
                "## Allocation warning",
                "",
                f"- Only {len(pns_all)} independent PN group(s) are available in the manifest.",
                "- Under PN-level partitioning, this is insufficient to safely populate calibration, training, validation, and held-out test roles with independent biological samples.",
                "- Condition/DIV coverage cannot be balanced across train, validation, and test until more independent PNs or an approved allocation are available.",
                "",
            ]
        )
    lines.extend(
        [
        "## PN allocation",
        "",
        "| PN | Split | Source kinds | Images | Conditions | DIVs |",
        "|:--|:--|:--|--:|:--|:--|",
        ]
    )
    for pn, group_rows in sorted(group_by(rows, "biological_group_candidate").items()):
        roles = ", ".join(f"`{role}`" for role in sorted({r["eligibility"] for r in group_rows}))
        lines.append(
            f"| `{pn}` | {roles} | {join_values(group_rows, 'source_kind')} | {len(group_rows)} | "
            f"{join_values(group_rows, 'inferred_condition')} | {join_values(group_rows, 'inferred_div')} |"
        )
    lines.extend(["", "## Split summary", "", "| Split | PNs | Images | Fiber images | Blank images |", "|:--|--:|--:|--:|--:|"])
    for split, group_rows in sorted(group_by(rows, "eligibility").items()):
        pns = {r["biological_group_candidate"] for r in group_rows}
        fiber_count = sum(1 for r in group_rows if r["source_kind"] == "fiber_image")
        blank_count = sum(1 for r in group_rows if r["source_kind"] == "blank_background")
        lines.append(f"| `{split}` | {len(pns)} | {len(group_rows)} | {fiber_count} | {blank_count} |")
    lines.extend(
        [
            "",
            "## Condition and DIV strata",
            "",
            "| Condition | DIV | PNs | Images | Splits represented | Warning |",
            "|:--|:--|--:|--:|:--|:--|",
        ]
    )
    for (condition, div), group_rows in sorted(group_by_two(rows, "inferred_condition", "inferred_div").items()):
        pns = {r["biological_group_candidate"] for r in group_rows}
        splits = {r["eligibility"] for r in group_rows}
        warning = "none"
        if len(pns) < min_pns_per_stratum:
            warning = f"too_few_independent_pns_for_safe_distribution_lt_{min_pns_per_stratum}"
        lines.append(f"| `{condition}` | `{div}` | {len(pns)} | {len(group_rows)} | {', '.join(sorted(splits))} | {warning} |")
    lines.extend(["", "## Split x condition x DIV counts", ""])
    for split, split_rows in sorted(group_by(rows, "eligibility").items()):
        lines.extend([f"### `{split}`", "", "| Condition | DIV | PNs | Images |", "|:--|:--|--:|--:|"])
        for (condition, div), group_rows in sorted(group_by_two(split_rows, "inferred_condition", "inferred_div").items()):
            pns = {r["biological_group_candidate"] for r in group_rows}
            lines.append(f"| `{condition}` | `{div}` | {len(pns)} | {len(group_rows)} |")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def group_by(rows: list[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[row[field]].append(row)
    return dict(out)


def group_by_two(rows: list[dict[str, str]], left: str, right: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[(row[left], row[right])].append(row)
    return dict(out)


def join_values(rows: list[dict[str, str]], field: str) -> str:
    return ", ".join(f"`{value}`" for value in sorted({r[field] for r in rows}))
