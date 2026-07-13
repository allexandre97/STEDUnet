"""Manifest-driven inventory and validation for expert real annotations."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label as connected_components

from .conversion import build_real_annotation_sample
from .labkit import load_real_labels


CLASS_VALUES = {
    "background": 0,
    "fibrous_tau": 1,
    "clump": 3,
    "uncertain_ignore": 255,
}

INVENTORY_FIELDS = [
    "sample_id", "image_identity", "source_root_id", "image_relative_path",
    "source_sha256", "annotation_root_id", "annotation_image_relative_path",
    "annotation_image_sha256", "labels_relative_path", "labels_sha256",
    "snakes_relative_path", "snakes_sha256", "skeleton_annotation_available",
    "shape_y", "shape_x", "culture_id", "disease", "tau_isoform",
    "experimental_condition", "div", "div_token", "series_index",
    "acquisition_group", "experimental_group_id", "split_group_id", "split",
    "biological_grouping_status", "background_pixels", "fibrous_tau_pixels",
    "clump_pixels", "uncertain_ignore_pixels", "background_components",
    "fibrous_tau_components", "clump_components", "uncertain_ignore_components",
    "snake_count", "usable_snake_count", "flagged_snake_count",
    "skeleton_pixels", "clipped_nonfibrous_snake_pixels", "validation_status",
    "problems",
]

CROP_FIELDS = [
    "crop_id", "parent_image_id", "source_image_id", "split_group_id", "split",
    "crop_origin_y", "crop_origin_x", "crop_height", "crop_width",
    "fibrous_tau_pixels", "clump_pixels", "uncertain_ignore_pixels",
    "skeleton_pixels", "skeleton_annotation_available",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_real_annotation_inventory(
    annotation_root: Path,
    image_root: Path,
    image_manifest: Path,
    split_manifest: Path,
    *,
    crop_size: int = 128,
    stride: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    images = {row["relative_path"]: row for row in read_csv(image_manifest)}
    splits = {row["stable_image_id"]: row for row in read_csv(split_manifest)}
    bundles, discovery_problems = discover_bundles(annotation_root)
    rows: list[dict[str, str]] = []
    samples: dict[str, dict[str, Any]] = {}
    validation_problems = list(discovery_problems)

    for identity, paths in sorted(bundles.items()):
        row, sample, problems = inventory_row(
            identity, paths, annotation_root, image_root, images, splits
        )
        rows.append(row)
        validation_problems.extend(f"{identity}: {problem}" for problem in problems)
        if sample is not None:
            samples[row["sample_id"]] = sample

    ids = [row["sample_id"] for row in rows]
    for sample_id, count in Counter(ids).items():
        if count > 1:
            problem = f"duplicate image/annotation pair for {sample_id}"
            validation_problems.append(problem)
            mark_invalid(rows, lambda row: row["sample_id"] == sample_id, problem)
    for problem, group in group_split_problems(rows):
        validation_problems.append(problem)
        mark_invalid(rows, lambda row: row["split_group_id"] == group, problem)

    valid_rows = [row for row in rows if row["validation_status"] == "valid"]
    crops = build_crop_plan(valid_rows, samples, crop_size, stride or crop_size)
    crop_problems = validate_crop_splits(rows, crops)
    if crop_problems:
        raise ValueError("real annotation crop validation failed:\n- " + "\n- ".join(crop_problems))
    return rows, crops, build_audit(rows, validation_problems)


def discover_bundles(annotation_root: Path) -> tuple[dict[str, dict[str, Path | None]], list[str]]:
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"annotation directory not found: {annotation_root}")
    substantive = [p for p in annotation_root.iterdir() if p.is_file() and not p.name.startswith("._")]
    by_suffix: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in substantive:
        if path.suffix.lower() in {".tif", ".tiff", ".labeling", ".txt"}:
            by_suffix[path.suffix.lower()][path.stem] = path
    identities = set(by_suffix[".labeling"])
    problems: list[str] = []
    for suffix in (".tif", ".labeling"):
        missing = sorted(identities - set(by_suffix[suffix]))
        extra = sorted(set(by_suffix[suffix]) - identities)
        problems.extend(f"missing {suffix} for {name}" for name in missing)
        problems.extend(f"orphan {suffix} for {name}" for name in extra)
    bundles = {
        name: {
            "annotation_image": by_suffix[".tif"].get(name),
            "labels": by_suffix[".labeling"].get(name),
            "snakes": by_suffix[".txt"].get(name),
        }
        for name in identities
    }
    return bundles, problems


def inventory_row(
    identity: str,
    paths: dict[str, Path | None],
    annotation_root: Path,
    image_root: Path,
    images: dict[str, dict[str, str]],
    splits: dict[str, dict[str, str]],
) -> tuple[dict[str, str], dict[str, Any] | None, list[str]]:
    filename = f"{identity}.tif"
    source = images.get(filename)
    annotation_image = paths["annotation_image"]
    labels_path = paths["labels"]
    snakes_path = paths["snakes"]
    problems: list[str] = []
    if source is None:
        return failed_row(identity, paths, annotation_root, "missing source manifest row"), None, ["missing source manifest row"]
    source_path = image_root / source["relative_path"]
    split = splits.get(source["stable_image_id"])
    if split is None:
        problems.append("missing split manifest row")
    for label, path in (("source image", source_path), ("annotation image", annotation_image), ("labels", labels_path)):
        if path is None or not path.is_file():
            problems.append(f"missing {label}")
    if problems:
        return failed_row(identity, paths, annotation_root, ";".join(problems), source), None, problems

    assert annotation_image is not None and labels_path is not None
    source_hash = sha256(source_path)
    annotation_hash = sha256(annotation_image)
    if source_hash != source["source_sha256"]:
        problems.append("source image checksum differs from sted_images.csv")
    if annotation_hash != source_hash:
        problems.append("annotation TIFF is not byte-identical to paired STED image")

    try:
        labels = load_real_labels(labels_path)
        sample = build_real_annotation_sample(source_path, snakes_path, labels_path)
    except Exception as exc:
        problems.append(str(exc))
        return failed_row(identity, paths, annotation_root, ";".join(problems), source), None, problems

    semantic = sample["real_semantic_mask"]
    expected_shape = (int(source["shape_y"]), int(source["shape_x"]))
    if semantic.shape != expected_shape:
        problems.append(f"annotation dimensions {semantic.shape} do not match manifest {expected_shape}")
    if set(np.unique(semantic).tolist()) - set(CLASS_VALUES.values()):
        problems.append(f"unexpected internal label values: {sorted(np.unique(semantic).tolist())}")
    confident_overlap = labels.masks["fibers"] & labels.masks["clump"] & ~labels.masks["uncertain_ignore"]
    if np.any(confident_overlap):
        problems.append("fibers/clump label overlap outside uncertain_ignore")
    if labels.warnings:
        problems.extend(labels.warnings)
    if np.any(sample["real_skeleton_mask"].astype(bool) & (semantic != 1)):
        problems.append("skeleton pixels outside confident fibrous regions")
    if np.any(sample["real_fibrous_mask"].astype(bool) & sample["real_clump_mask"].astype(bool)):
        problems.append("fibrous/clump target overlap")
    if np.any(sample["real_uncertain_ignore_mask"].astype(bool) & (semantic == 0)):
        problems.append("uncertain_ignore converted to background")

    flags = sample["metadata"]["snake_quality_flags"]
    counts = class_statistics(semantic)
    row = {
        "sample_id": source["stable_image_id"],
        "image_identity": identity,
        "source_root_id": source["source_root_id"],
        "image_relative_path": source["relative_path"],
        "source_sha256": source_hash,
        "annotation_root_id": "real_annotations",
        "annotation_image_relative_path": annotation_image.relative_to(annotation_root).as_posix(),
        "annotation_image_sha256": annotation_hash,
        "labels_relative_path": labels_path.relative_to(annotation_root).as_posix(),
        "labels_sha256": sha256(labels_path),
        "snakes_relative_path": snakes_path.relative_to(annotation_root).as_posix() if snakes_path else "not_available",
        "snakes_sha256": sha256(snakes_path) if snakes_path else "not_available",
        "skeleton_annotation_available": text_bool(sample["metadata"]["skeleton_annotation_available"]),
        "shape_y": str(semantic.shape[0]),
        "shape_x": str(semantic.shape[1]),
        **{field: source[field] for field in (
            "culture_id", "disease", "tau_isoform", "experimental_condition", "div",
            "div_token", "series_index", "acquisition_group", "experimental_group_id"
        )},
        "split_group_id": source["experimental_group_id"],
        "split": canonical_split(split["eligibility"]) if split else "unknown",
        "biological_grouping_status": "provisional_experimental_group_from_existing_manifest",
        **counts,
        "snake_count": str(len(flags)),
        "usable_snake_count": str(sum(flag["usable_fibrous_skeleton"] for flag in flags)),
        "flagged_snake_count": str(sum(not flag["usable_fibrous_skeleton"] for flag in flags)),
        "skeleton_pixels": str(int(np.count_nonzero(sample["real_skeleton_mask"]))),
        "clipped_nonfibrous_snake_pixels": str(sum(flag["clipped_nonfibrous_pixels"] for flag in flags)),
        "validation_status": "valid" if not problems else "invalid",
        "problems": ";".join(problems) if problems else "none",
    }
    return row, sample, problems


def failed_row(
    identity: str,
    paths: dict[str, Path | None],
    annotation_root: Path,
    problem: str,
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    row = {field: "unknown" for field in INVENTORY_FIELDS}
    row.update({
        "sample_id": source["stable_image_id"] if source else identity,
        "image_identity": identity,
        "annotation_root_id": "real_annotations",
        "labels_relative_path": relative_or_missing(paths.get("labels"), annotation_root),
        "snakes_relative_path": relative_or_missing(paths.get("snakes"), annotation_root),
        "validation_status": "invalid",
        "problems": problem,
    })
    return row


def relative_or_missing(path: Path | None, root: Path) -> str:
    return path.relative_to(root).as_posix() if path else "not_available"


def class_statistics(semantic: np.ndarray) -> dict[str, str]:
    stats: dict[str, str] = {}
    for name, value in CLASS_VALUES.items():
        mask = semantic == value
        stats[f"{name}_pixels"] = str(int(mask.sum()))
        stats[f"{name}_components"] = str(int(connected_components(mask)[1]))
    return stats


def build_crop_plan(
    rows: list[dict[str, str]],
    samples: dict[str, dict[str, Any]],
    crop_size: int,
    stride: int,
) -> list[dict[str, str]]:
    crops: list[dict[str, str]] = []
    for row in rows:
        semantic = samples[row["sample_id"]]["real_semantic_mask"]
        skeleton = samples[row["sample_id"]]["real_skeleton_mask"]
        ys = axis_origins(semantic.shape[0], crop_size, stride)
        xs = axis_origins(semantic.shape[1], crop_size, stride)
        for index, (y, x) in enumerate((y, x) for y in ys for x in xs):
            target = semantic[y:y + crop_size, x:x + crop_size]
            crops.append({
                "crop_id": f"{row['sample_id']}__crop_{index:04d}",
                "parent_image_id": row["sample_id"],
                "source_image_id": row["sample_id"],
                "split_group_id": row["split_group_id"],
                "split": row["split"],
                "crop_origin_y": str(y),
                "crop_origin_x": str(x),
                "crop_height": str(crop_size),
                "crop_width": str(crop_size),
                "fibrous_tau_pixels": str(int(np.count_nonzero(target == 1))),
                "clump_pixels": str(int(np.count_nonzero(target == 3))),
                "uncertain_ignore_pixels": str(int(np.count_nonzero(target == 255))),
                "skeleton_pixels": str(int(np.count_nonzero(skeleton[y:y + crop_size, x:x + crop_size]))),
                "skeleton_annotation_available": row["skeleton_annotation_available"],
            })
    return crops


def axis_origins(length: int, crop_size: int, stride: int) -> list[int]:
    if crop_size <= 0 or stride <= 0 or length < crop_size:
        raise ValueError(f"invalid crop_size/stride {crop_size}/{stride} for length {length}")
    starts = list(range(0, length - crop_size + 1, stride))
    if starts[-1] != length - crop_size:
        starts.append(length - crop_size)
    return starts


def group_split_problems(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_group[row["split_group_id"]].add(row["split"])
    return [
        (f"split group {group} crosses splits: {sorted(splits)}", group)
        for group, splits in by_group.items() if len(splits) > 1
    ]


def mark_invalid(rows: list[dict[str, str]], selected, problem: str) -> None:
    for row in rows:
        if selected(row):
            row["validation_status"] = "invalid"
            row["problems"] = problem if row["problems"] == "none" else f"{row['problems']};{problem}"


def validate_crop_splits(rows: list[dict[str, str]], crops: list[dict[str, str]]) -> list[str]:
    parents = {row["sample_id"]: row for row in rows}
    problems: list[str] = []
    seen: set[str] = set()
    for crop in crops:
        if crop["crop_id"] in seen:
            problems.append(f"duplicate crop_id {crop['crop_id']}")
        seen.add(crop["crop_id"])
        parent = parents.get(crop["parent_image_id"])
        if parent is None:
            problems.append(f"{crop['crop_id']}: missing parent image")
        elif (crop["split"], crop["split_group_id"]) != (parent["split"], parent["split_group_id"]):
            problems.append(f"{crop['crop_id']}: crop split differs from parent image")
    return problems


def build_audit(rows: list[dict[str, str]], validation_problems: list[str]) -> dict[str, Any]:
    fields = (
        "validation_status", "disease", "tau_isoform", "div", "split",
        "experimental_group_id",
    )
    classes = tuple(CLASS_VALUES)
    flagged = [row["image_identity"] for row in rows if integer(row["flagged_snake_count"]) > 0]
    return {
        "annotated_image_count": len(rows),
        "metadata_distribution": {
            field: dict(sorted(Counter(row[field] for row in rows).items())) for field in fields
        },
        "class_statistics": {
            name: {
                "pixels": sum(integer(row[f"{name}_pixels"]) for row in rows),
                "components": sum(integer(row[f"{name}_components"]) for row in rows),
            }
            for name in classes
        },
        "skeleton_annotations": {
            "images_available": sum(row["skeleton_annotation_available"] == "true" for row in rows),
            "total_snakes": sum(integer(row["snake_count"]) for row in rows),
            "usable_snakes": sum(integer(row["usable_snake_count"]) for row in rows),
            "flagged_snakes": sum(integer(row["flagged_snake_count"]) for row in rows),
            "skeleton_pixels": sum(integer(row["skeleton_pixels"]) for row in rows),
            "nonfibrous_pixels_clipped": sum(integer(row["clipped_nonfibrous_snake_pixels"]) for row in rows),
        },
        "ambiguous_or_missing_metadata": [
            "Biological preparation identity is not available beyond the existing provisional experimental_group_id. PN is retained only as culture_id."
        ],
        "manual_resolution_required": [
            "Confirm biological preparation/replicate identifiers before final train/evaluation assignment.",
            f"Review flagged snake overlap in {len(flagged)} image(s): {', '.join(flagged) if flagged else 'none'}.",
            f"Resolve {len(validation_problems)} validation problem(s) before using affected images for training or evaluation.",
        ],
        "validation_problems": validation_problems,
    }


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_bool(value: bool) -> str:
    return "true" if value else "false"


def integer(value: str) -> int:
    return int(value) if value.isdigit() else 0


def canonical_split(value: str) -> str:
    return {"training": "train", "held_out_test": "test"}.get(value, value)
