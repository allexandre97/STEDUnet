"""Fixed grouped cross-validation assignments for real annotations."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import itertools
from pathlib import Path
from typing import Any


FOLD_SCHEMA_VERSION = "real_annotation_grouped_cv_1.0.0"
FOLD_FIELDS = [
    "fold_schema_version", "outer_fold", "sample_id", "image_identity",
    "split_group_id", "partition", "validation_status", "disease",
    "tau_isoform", "div", "fibrous_fraction", "clump_fraction",
    "uncertain_fraction", "morphology_density",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_grouped_folds(
    inventory_rows: list[dict[str, str]],
    *,
    fold_count: int = 5,
    seed: int = 20260713,
    validation_fraction: float = 0.2,
) -> list[dict[str, str]]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    groups = grouped_records(inventory_rows)
    if fold_count > len(groups):
        raise ValueError(f"cannot create {fold_count} folds from {len(groups)} indivisible groups")
    enrich_prevalence(inventory_rows)
    features = feature_names(inventory_rows)
    vectors = {group: feature_vector(rows, features) for group, rows in groups.items()}
    outer = greedy_group_assignment(vectors, fold_count, seed)
    fold_rows: list[dict[str, str]] = []
    for fold in range(fold_count):
        test_groups = {group for group, assigned in outer.items() if assigned == fold}
        development = {group: vector for group, vector in vectors.items() if group not in test_groups}
        validation_groups = choose_validation_groups(development, validation_fraction)
        for row in inventory_rows:
            group = row["split_group_id"]
            partition = "test" if group in test_groups else "validation" if group in validation_groups else "train"
            fold_rows.append(fold_row(row, fold, partition))
    problems = validate_grouped_folds(fold_rows, inventory_rows, fold_count)
    if problems:
        raise ValueError("invalid grouped folds:\n- " + "\n- ".join(problems))
    return fold_rows


def grouped_records(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group = row.get("split_group_id", "")
        if not group or group == "unknown":
            raise ValueError(f"{row.get('sample_id', '<unknown>')}: missing split_group_id")
        groups[group].append(row)
    return dict(groups)


def enrich_prevalence(rows: list[dict[str, str]]) -> None:
    foreground = []
    for row in rows:
        pixels = int(row["shape_y"]) * int(row["shape_x"])
        row["fibrous_fraction"] = f"{int(row['fibrous_tau_pixels']) / pixels:.8f}"
        row["clump_fraction"] = f"{int(row['clump_pixels']) / pixels:.8f}"
        row["uncertain_fraction"] = f"{int(row['uncertain_ignore_pixels']) / pixels:.8f}"
        foreground.append((int(row["fibrous_tau_pixels"]) + int(row["clump_pixels"])) / pixels)
    median = sorted(foreground)[len(foreground) // 2]
    for row, value in zip(rows, foreground):
        row["morphology_density"] = "dense" if value >= median else "sparse"


def feature_names(rows: list[dict[str, str]]) -> list[str]:
    categorical = []
    for field in ("disease", "tau_isoform", "div", "morphology_density"):
        categorical.extend(f"{field}={value}" for value in sorted({row[field] for row in rows}))
    return ["image_count", "fibrous_fraction", "clump_fraction", "uncertain_fraction", *categorical]


def feature_vector(rows: list[dict[str, str]], names: list[str]) -> dict[str, float]:
    vector = {name: 0.0 for name in names}
    vector["image_count"] = float(len(rows))
    for row in rows:
        for name in ("fibrous_fraction", "clump_fraction", "uncertain_fraction"):
            vector[name] += float(row[name])
        for field in ("disease", "tau_isoform", "div", "morphology_density"):
            vector[f"{field}={row[field]}"] += 1.0
    return vector


def greedy_group_assignment(
    vectors: dict[str, dict[str, float]],
    fold_count: int,
    seed: int,
) -> dict[str, int]:
    totals = sum_vectors(vectors.values())
    targets = {name: value / fold_count for name, value in totals.items()}
    folds = [{name: 0.0 for name in totals} for _ in range(fold_count)]
    assignment: dict[str, int] = {}
    ordered = sorted(
        vectors,
        key=lambda group: (
            -vectors[group]["image_count"],
            -rarity_score(vectors[group], totals),
            seeded_key(group, seed),
        ),
    )
    for index, group in enumerate(ordered):
        candidates = range(fold_count) if index >= fold_count else [index]
        fold = min(candidates, key=lambda item: (fold_cost(add_vectors(folds[item], vectors[group]), targets), item))
        assignment[group] = fold
        folds[fold] = add_vectors(folds[fold], vectors[group])
    return improve_assignment(assignment, vectors, fold_count, targets)


def improve_assignment(
    assignment: dict[str, int],
    vectors: dict[str, dict[str, float]],
    fold_count: int,
    targets: dict[str, float],
) -> dict[str, int]:
    def cost(value: dict[str, int]) -> float:
        fold_vectors = [sum_vectors(vectors[group] for group, fold in value.items() if fold == index) for index in range(fold_count)]
        return sum(fold_cost(vector, targets) for vector in fold_vectors)

    best = cost(assignment)
    changed = True
    while changed:
        changed = False
        for left, right in itertools.combinations(sorted(assignment), 2):
            if assignment[left] == assignment[right]:
                continue
            trial = assignment.copy()
            trial[left], trial[right] = trial[right], trial[left]
            trial_cost = cost(trial)
            if trial_cost + 1e-12 < best:
                assignment, best, changed = trial, trial_cost, True
                break
        if changed:
            continue
    return assignment


def choose_validation_groups(
    development: dict[str, dict[str, float]],
    fraction: float,
) -> set[str]:
    names = sorted(development)
    totals = sum_vectors(development.values())
    target = {name: value * fraction for name, value in totals.items()}
    target_images = max(1, round(totals["image_count"] * fraction))
    best: tuple[float, tuple[str, ...]] | None = None
    for size in range(1, len(names)):
        for subset in itertools.combinations(names, size):
            vector = sum_vectors(development[group] for group in subset)
            image_penalty = 8.0 * ((vector["image_count"] - target_images) / max(target_images, 1)) ** 2
            score = fold_cost(vector, target) + image_penalty
            candidate = (score, subset)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("development set cannot provide grouped validation")
    return set(best[1])


def fold_cost(vector: dict[str, float], target: dict[str, float]) -> float:
    cost = 0.0
    for name, expected in target.items():
        weight = 10.0 if name == "image_count" else 2.0 if name in {
            "fibrous_fraction", "clump_fraction", "uncertain_fraction"
        } else 1.0
        cost += weight * ((vector.get(name, 0.0) - expected) / max(expected, 0.25)) ** 2
    return cost


def rarity_score(vector: dict[str, float], totals: dict[str, float]) -> float:
    return sum(value / max(totals[name], 1.0) for name, value in vector.items() if "=" in name)


def seeded_key(value: str, seed: int) -> int:
    return sum((index + seed) * ord(char) for index, char in enumerate(value))


def sum_vectors(vectors) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for vector in vectors:
        for name, value in vector.items():
            out[name] += value
    return dict(out)


def add_vectors(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {name: left.get(name, 0.0) + right.get(name, 0.0) for name in set(left) | set(right)}


def fold_row(row: dict[str, str], fold: int, partition: str) -> dict[str, str]:
    return {
        "fold_schema_version": FOLD_SCHEMA_VERSION,
        "outer_fold": str(fold),
        "sample_id": row["sample_id"],
        "image_identity": row["image_identity"],
        "split_group_id": row["split_group_id"],
        "partition": partition,
        "validation_status": row["validation_status"],
        "disease": row["disease"],
        "tau_isoform": row["tau_isoform"],
        "div": row["div"],
        "fibrous_fraction": row["fibrous_fraction"],
        "clump_fraction": row["clump_fraction"],
        "uncertain_fraction": row["uncertain_fraction"],
        "morphology_density": row["morphology_density"],
    }


def validate_grouped_folds(
    rows: list[dict[str, str]],
    inventory: list[dict[str, str]],
    fold_count: int,
) -> list[str]:
    problems: list[str] = []
    inventory_ids = {row["sample_id"] for row in inventory}
    for fold in range(fold_count):
        fold_rows = [row for row in rows if int(row["outer_fold"]) == fold]
        if {row["sample_id"] for row in fold_rows} != inventory_ids:
            problems.append(f"fold {fold}: image membership differs from inventory")
        if {row["partition"] for row in fold_rows} != {"train", "validation", "test"}:
            problems.append(f"fold {fold}: train/validation/test partitions are required")
        group_partitions: dict[str, set[str]] = defaultdict(set)
        for row in fold_rows:
            group_partitions[row["split_group_id"]].add(row["partition"])
        for group, partitions in group_partitions.items():
            if len(partitions) > 1:
                problems.append(f"fold {fold}: group {group} crosses partitions")
    test_counts = Counter(row["sample_id"] for row in rows if row["partition"] == "test")
    if test_counts != Counter({sample_id: 1 for sample_id in inventory_ids}):
        problems.append("each image must appear in outer test exactly once")
    return problems


def write_folds(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FOLD_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fold_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary = {}
    for fold in sorted({row["outer_fold"] for row in rows}, key=int):
        fold_rows = [row for row in rows if row["outer_fold"] == fold]
        partitions = {}
        for partition in ("train", "validation", "test"):
            selected = [row for row in fold_rows if row["partition"] == partition]
            partitions[partition] = {
                "images": len(selected),
                "groups": len({row["split_group_id"] for row in selected}),
                "disease": dict(sorted(Counter(row["disease"] for row in selected).items())),
                "isoform": dict(sorted(Counter(row["tau_isoform"] for row in selected).items())),
                "div": dict(sorted(Counter(row["div"] for row in selected).items())),
                "morphology_density": dict(sorted(Counter(row["morphology_density"] for row in selected).items())),
                "annotation_prevalence_mean": {
                    name: float(sum(float(row[f"{name}_fraction"]) for row in selected) / len(selected)) if selected else 0.0
                    for name in ("fibrous", "clump", "uncertain")
                },
                "invalid_images": sum(row["validation_status"] != "valid" for row in selected),
            }
        summary[fold] = partitions
    return {"fold_schema_version": FOLD_SCHEMA_VERSION, "folds": summary}
