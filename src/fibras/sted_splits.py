"""Manifest-driven STED split assignment and validation."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path


ALLOWED_TAU_ISOFORMS = {"3R", "4R", "unknown"}
PRIMARY_GROUPING_RULE = "experimental_group_holdout_primary"
CULTURE_HELD_OUT_RULE = "culture_held_out"

SPLIT_FIELDS = [
    "stable_image_id",
    "source_kind",
    "culture_id",
    "disease",
    "tau_isoform",
    "experimental_condition",
    "div",
    "div_token",
    "series_index",
    "experimental_group_id",
    "acquisition_group",
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
        writer = csv.DictWriter(f, fieldnames=SPLIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SPLIT_FIELDS})


def load_inventory(inventory_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return read_csv(inventory_dir / "sted_images.csv"), read_csv(inventory_dir / "sted_blanks.csv")


def create_splits(inventory_dir: Path, strategy: str = PRIMARY_GROUPING_RULE) -> list[dict[str, str]]:
    if strategy in {"pn_holdout", "experimental_group_holdout"}:
        strategy = PRIMARY_GROUPING_RULE
    if strategy not in {PRIMARY_GROUPING_RULE, CULTURE_HELD_OUT_RULE}:
        raise ValueError(f"unsupported strategy: {strategy}")
    images, blanks = load_inventory(inventory_dir)
    records = images + blanks
    if strategy == CULTURE_HELD_OUT_RULE:
        role_by_group = allocate_culture_held_out_roles(records)
        group_field = "culture_id"
        rule = CULTURE_HELD_OUT_RULE
    else:
        role_by_group = allocate_experimental_group_roles(records)
        group_field = "experimental_group_id"
        rule = PRIMARY_GROUPING_RULE
    rows = [
        split_row(
            rec,
            role_by_group[rec[group_field]],
            assignment_reason(role_by_group[rec[group_field]], rule),
            rule,
            synthetic_split=role_by_group[rec[group_field]] if rec["source_kind"] == "blank_background" else "not_applicable",
        )
        for rec in records
    ]
    return sorted(rows, key=lambda r: (r["experimental_group_id"], r["source_kind"], sort_key(r["series_index"]), r["stable_image_id"]))


def allocate_experimental_group_roles(records: list[dict[str, str]]) -> dict[str, str]:
    """Assign one provisional role per experimental_group_id.

    Allocation is deterministic and balances roles within experimental
    condition when possible. Condition x DIV strata are still reported
    separately; with one group per DIV, they are too small for safe stratified
    allocation and require project-owner review.
    """

    by_condition: dict[str, set[str]] = defaultdict(set)
    div_by_group: dict[str, str] = {}
    for rec in records:
        group = rec["experimental_group_id"]
        by_condition[rec.get("experimental_condition", "unknown")].add(group)
        div_by_group[group] = rec.get("div", "unknown")
    allocation: dict[str, str] = {}
    for _, groups in sorted(by_condition.items()):
        ordered = sorted(groups, key=lambda group: (sort_key(div_by_group.get(group, "unknown")), group))
        if len(ordered) == 1:
            allocation[ordered[0]] = "calibration"
        elif len(ordered) == 2:
            allocation[ordered[0]] = "training"
            allocation[ordered[1]] = "validation"
        else:
            pattern = ["training", "validation", "held_out_test", "calibration"]
            for idx, group in enumerate(ordered):
                allocation[group] = pattern[idx] if idx < len(pattern) else "training"
    return allocation


def allocate_culture_held_out_roles(records: list[dict[str, str]]) -> dict[str, str]:
    cultures = sorted({r["culture_id"] for r in records})
    if not cultures:
        return {}
    if len(cultures) == 1:
        return {cultures[0]: "calibration"}
    allocation = {culture: "training" for culture in cultures[:-1]}
    allocation[cultures[-1]] = "culture_held_out"
    return allocation


def split_row(rec: dict[str, str], eligibility: str, reason: str, rule: str, synthetic_split: str) -> dict[str, str]:
    return {
        "stable_image_id": rec["stable_image_id"],
        "source_kind": rec["source_kind"],
        "culture_id": rec.get("culture_id", "unknown"),
        "disease": rec.get("disease", "unknown"),
        "tau_isoform": rec.get("tau_isoform", "unknown"),
        "experimental_condition": rec.get("experimental_condition", "unknown"),
        "div": rec.get("div", "unknown"),
        "div_token": rec.get("div_token", "unknown"),
        "series_index": rec.get("series_index", "unknown"),
        "experimental_group_id": rec.get("experimental_group_id", "unknown"),
        "acquisition_group": rec.get("acquisition_group", "unknown"),
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


def assignment_reason(role: str, rule: str) -> str:
    if rule == CULTURE_HELD_OUT_RULE:
        return f"provisional culture-held-out allocation: {role}"
    return f"provisional experimental-group allocation: {role}"


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
        rec = inventory.get(row["stable_image_id"])
        if not rec:
            continue
        errors.extend(validate_row_metadata(row, rec))
        if row.get("human_approved") != "false":
            errors.append(f"{row['stable_image_id']}: splits must remain human_approved=false")
        if row.get("primary_metric_role") != row.get("eligibility"):
            errors.append(f"{row['stable_image_id']}: primary_metric_role must equal eligibility")
        if row.get("grouping_rule") not in {PRIMARY_GROUPING_RULE, CULTURE_HELD_OUT_RULE}:
            errors.append(f"{row['stable_image_id']}: invalid grouping_rule {row.get('grouping_rule')}")
        if row["source_kind"] == "blank_background" and row["synthetic_split"] != row["eligibility"]:
            errors.append(f"{row['stable_image_id']}: blank synthetic_split does not match eligibility")
        if row.get("secondary_image_eval_role", "not_assigned") not in {
            "not_assigned",
            "secondary_training",
            "secondary_validation",
            "secondary_test",
            "culture_held_out",
        }:
            errors.append(f"{row['stable_image_id']}: invalid secondary_image_eval_role")
    rules = {r.get("grouping_rule") for r in rows}
    if len(rules) == 1 and PRIMARY_GROUPING_RULE in rules:
        errors.extend(check_group_leakage(rows, "experimental_group_id", "experimental group"))
    if len(rules) == 1 and CULTURE_HELD_OUT_RULE in rules:
        errors.extend(check_group_leakage(rows, "culture_id", "culture"))
    errors.extend(check_group_leakage(rows, "acquisition_group", "acquisition group"))
    return errors


def validate_row_metadata(row: dict[str, str], rec: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for field in ["source_kind", "culture_id", "disease", "tau_isoform", "experimental_condition", "div", "div_token", "series_index", "experimental_group_id", "acquisition_group"]:
        if row.get(field) != rec.get(field):
            errors.append(f"{row['stable_image_id']}: split {field}={row.get(field)} does not match inventory {rec.get(field)}")
    if row.get("tau_isoform") not in ALLOWED_TAU_ISOFORMS:
        errors.append(f"{row['stable_image_id']}: invalid tau_isoform {row.get('tau_isoform')}")
    expected_condition = expected_experimental_condition(row)
    if row.get("experimental_condition") != expected_condition:
        errors.append(f"{row['stable_image_id']}: inconsistent experimental_condition")
    expected_group = expected_experimental_group_id(row)
    if row.get("experimental_group_id") != expected_group:
        errors.append(f"{row['stable_image_id']}: inconsistent experimental_group_id")
    return errors


def expected_experimental_condition(row: dict[str, str]) -> str:
    if "unknown" in {row.get("disease"), row.get("tau_isoform")}:
        return "unknown"
    return f"{row['disease']}_{row['tau_isoform']}"


def expected_experimental_group_id(row: dict[str, str]) -> str:
    condition = expected_experimental_condition(row)
    if "unknown" in {row.get("culture_id"), condition, row.get("div")}:
        return "unknown"
    return f"{row['culture_id']}_{condition}_DIV{int(row['div']):02d}"


def check_group_leakage(rows: list[dict[str, str]], group_field: str, label: str) -> list[str]:
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_group[row[group_field]].add(row["eligibility"])
    errors = []
    for group, roles in by_group.items():
        if len(roles) > 1:
            errors.append(f"{label} {group} crosses roles: {sorted(roles)}")
    return errors


def write_split_report(splits_path: Path, report_path: Path, min_groups_per_stratum: int = 3) -> None:
    rows = read_csv(splits_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# STED experimental-group split report",
        "",
        "## Policy",
        "",
        "- `PN###` is recorded as `culture_id`, not as the disease condition and not as an automatic partitioning unit.",
        "- `3R` and `4R` are tau isoforms, not acquisition rounds.",
        "- Disease, tau isoform, DIV, culture, and series are preserved as separate metadata fields.",
        "- Primary development splits use `experimental_group_id = culture_id + disease + tau_isoform + div` as the indivisible grouping unit.",
        "- Experimental condition is `disease + '_' + tau_isoform`.",
        "- Current assignments are provisional and keep `human_approved=false`; no biological independence claim is made.",
        "- Optional `culture_held_out` splitting is supported separately and must not be mixed with primary experimental-group-held-out metrics.",
        "",
    ]
    strata = group_by_two(rows, "experimental_condition", "div")
    small = [(k, v) for k, v in strata.items() if len({r["experimental_group_id"] for r in v}) < min_groups_per_stratum]
    if small:
        lines.extend(
            [
                "## Allocation warning",
                "",
                f"- {len(small)} condition/DIV stratum or strata have fewer than {min_groups_per_stratum} independent experimental groups.",
                "- These strata are too small for meaningful stratified train/validation/test allocation without project-owner review.",
                "- Split assignments remain provisional.",
                "",
            ]
        )
    lines.extend(["## Experimental-group allocation", "", "| Experimental group | Split | Culture | Condition | DIV | Images | Source kinds | Series |", "|:--|:--|:--|:--|:--|--:|:--|:--|"])
    for group, group_rows in sorted(group_by(rows, "experimental_group_id").items()):
        roles = join_values(group_rows, "eligibility")
        lines.append(
            f"| `{group}` | {roles} | {join_values(group_rows, 'culture_id')} | {join_values(group_rows, 'experimental_condition')} | "
            f"{join_values(group_rows, 'div')} | {len(group_rows)} | {join_values(group_rows, 'source_kind')} | {join_values(group_rows, 'series_index')} |"
        )
    lines.extend(["", "## Split summary", "", "| Split | Experimental groups | Cultures | Images | Fiber images | Blank images |", "|:--|--:|--:|--:|--:|--:|"])
    for split, group_rows in sorted(group_by(rows, "eligibility").items()):
        groups = {r["experimental_group_id"] for r in group_rows}
        cultures = {r["culture_id"] for r in group_rows}
        fiber_count = sum(1 for r in group_rows if r["source_kind"] == "fiber_image")
        blank_count = sum(1 for r in group_rows if r["source_kind"] == "blank_background")
        lines.append(f"| `{split}` | {len(groups)} | {len(cultures)} | {len(group_rows)} | {fiber_count} | {blank_count} |")
    lines.extend(["", "## Condition and DIV strata", "", "| Experimental condition | DIV | Experimental groups | Images | Splits represented | Warning |", "|:--|:--|--:|--:|:--|:--|"])
    for (condition, div), group_rows in sorted(strata.items()):
        groups = {r["experimental_group_id"] for r in group_rows}
        splits = {r["eligibility"] for r in group_rows}
        warning = "none"
        if len(groups) < min_groups_per_stratum:
            warning = f"too_few_experimental_groups_for_safe_distribution_lt_{min_groups_per_stratum}"
        lines.append(f"| `{condition}` | `{div}` | {len(groups)} | {len(group_rows)} | {', '.join(sorted(splits))} | {warning} |")
    lines.extend(["", "## Split x condition x DIV counts", ""])
    for split, split_rows in sorted(group_by(rows, "eligibility").items()):
        lines.extend([f"### `{split}`", "", "| Experimental condition | DIV | Experimental groups | Images |", "|:--|:--|--:|--:|"])
        for (condition, div), group_rows in sorted(group_by_two(split_rows, "experimental_condition", "div").items()):
            groups = {r["experimental_group_id"] for r in group_rows}
            lines.append(f"| `{condition}` | `{div}` | {len(groups)} | {len(group_rows)} |")
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
    return ", ".join(f"`{value}`" for value in sorted({r[field] for r in rows}, key=sort_key))


def sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):08d}") if value.isdigit() else (1, value)
