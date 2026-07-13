"""Real annotation crop manifests for optional fine-tuning."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from fibras.annotations import build_real_annotation_sample


@dataclass(frozen=True)
class RealAnnotationRecord:
    sample_id: str
    image_path: Path
    labels_path: Path
    snakes_path: Path | None
    notes: str = ""
    split: str | None = None
    split_group_id: str | None = None
    metadata: dict[str, str] | None = None


def read_real_annotation_manifest(
    path: Path,
    *,
    image_root: Path | None = None,
    annotation_root: Path | None = None,
) -> list[RealAnnotationRecord]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if row.get("validation_status", "valid") == "valid"]
    fields = set(rows[0] if rows else [])
    legacy = {"image_path", "labels_path"} <= fields
    portable = {"image_relative_path", "labels_relative_path"} <= fields
    if "sample_id" not in fields or not (legacy or portable):
        raise ValueError(f"{path}: expected sample_id and legacy path or portable relative-path columns")
    if portable and (image_root is None or annotation_root is None):
        raise ValueError(f"{path}: portable inventory requires image_root and annotation_root")
    return [
        RealAnnotationRecord(
            sample_id=row["sample_id"],
            image_path=(image_root / row["image_relative_path"] if portable else resolve_path(row["image_path"], path.parent)),
            labels_path=(annotation_root / row["labels_relative_path"] if portable else resolve_path(row["labels_path"], path.parent)),
            snakes_path=manifest_snakes_path(row, path, annotation_root, portable),
            notes=row.get("notes", ""),
            split=row.get("split") or None,
            split_group_id=row.get("split_group_id") or row["sample_id"],
            metadata={key: row.get(key, "unknown") for key in (
                "image_identity", "culture_id", "disease", "tau_isoform", "div",
                "series_index", "acquisition_group", "experimental_group_id",
                "biological_grouping_status", "expected_category",
            )},
        )
        for row in rows
    ]


def apply_fold_assignments(
    records: list[RealAnnotationRecord],
    fold_manifest: Path,
    outer_fold: int,
) -> list[RealAnnotationRecord]:
    rows = [
        row for row in read_csv_rows(fold_manifest)
        if int(row["outer_fold"]) == outer_fold
    ]
    assignments = {row["sample_id"]: row["partition"] for row in rows}
    missing = sorted({record.sample_id for record in records} - set(assignments))
    if missing:
        raise ValueError(f"fold {outer_fold} is missing real images: {missing}")
    return [replace(record, split=assignments[record.sample_id]) for record in records]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_snakes_path(
    row: dict[str, str],
    manifest_path: Path,
    annotation_root: Path | None,
    portable: bool,
) -> Path | None:
    key = "snakes_relative_path" if portable else "snakes_path"
    raw = row.get(key, "")
    if raw in {"", "not_available", "none"}:
        return None
    return annotation_root / raw if portable else resolve_path(raw, manifest_path.parent)


def discover_real_annotation_triplets(annotation_dir: Path) -> list[RealAnnotationRecord]:
    if not annotation_dir.exists():
        raise FileNotFoundError(f"annotation directory not found: {annotation_dir}")
    images = [
        path
        for path in sorted(annotation_dir.iterdir())
        if path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower() in {".tif", ".tiff", ".png"}
        and not is_label_image(path)
    ]
    records = [
        RealAnnotationRecord(
            safe_stem(image), image, labels,
            image.with_suffix(".txt") if image.with_suffix(".txt").exists() else None,
        )
        for image in images
        for labels in [matching_label_path(image)]
        if labels is not None
    ]
    if not records:
        raise FileNotFoundError(f"no complete image/snakes/labels triplets found in {annotation_dir}")
    return records


def is_label_image(path: Path) -> bool:
    return path.name.endswith(("_labels.png", "_label.png", "_mask.png", "_labels.tif", "_label.tif", "_mask.tif"))


def matching_label_path(image: Path) -> Path | None:
    candidates = [
        image.with_suffix(".labeling"),
        image.with_name(f"{image.stem}_labels.png"),
        image.with_name(f"{image.stem}_label.png"),
        image.with_name(f"{image.stem}_mask.png"),
        image.with_name(f"{image.stem}_labels.tif"),
        image.with_name(f"{image.stem}_label.tif"),
        image.with_name(f"{image.stem}_mask.tif"),
    ]
    return next((path for path in candidates if path.exists()), None)


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in path.stem).strip("_")


def resolve_path(raw: str, root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def build_real_crop_dataset(
    records: list[RealAnnotationRecord],
    out_dir: Path,
    *,
    crop_size: int = 128,
    stride: int | None = None,
    seed: int = 123,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    leave_one_out_image: str | None = None,
    min_labeled_fraction: float = 0.0,
) -> list[dict[str, str]]:
    """Write real-compatible crop NPZs and a training manifest.

    Splits are assigned per source image before crops are generated, so crops
    from the same expert-annotated image cannot cross train/validation/test.
    """
    if not records:
        raise ValueError("no real annotation records provided")
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = image_splits(records, seed, validation_fraction, test_fraction, leave_one_out_image)
    rows: list[dict[str, str]] = []
    for record in records:
        sample = build_real_annotation_sample(record.image_path, record.snakes_path, record.labels_path)
        image = as_uint8(sample["image_uint8"], sample["image_float"])
        semantic = sample["real_semantic_mask"]
        skeleton = sample["real_skeleton_mask"]
        split = record.split or splits[record.sample_id]
        for crop_index, (y0, x0) in enumerate(crop_origins(semantic.shape, crop_size, stride or crop_size)):
            crop_semantic = semantic[y0 : y0 + crop_size, x0 : x0 + crop_size]
            labeled = np.isin(crop_semantic, [1, 3, 255])
            if float(labeled.mean()) < min_labeled_fraction:
                continue
            crop_id = f"{record.sample_id}__crop_{crop_index:04d}"
            arrays = {
                "render_uint8": image[y0 : y0 + crop_size, x0 : x0 + crop_size],
                "real_compatible_semantic_mask": crop_semantic.astype(np.uint8, copy=False),
                "real_compatible_fibrous_mask": (crop_semantic == 1).astype(np.uint8),
                "real_compatible_clump_mask": (crop_semantic == 3).astype(np.uint8),
                "real_compatible_uncertain_ignore_mask": (crop_semantic == 255).astype(np.uint8),
                "real_compatible_skeleton_mask": skeleton[y0 : y0 + crop_size, x0 : x0 + crop_size].astype(np.uint8),
                "real_compatible_skeleton_valid_mask": sample["real_skeleton_valid_mask"][
                    y0 : y0 + crop_size, x0 : x0 + crop_size
                ].astype(np.uint8),
            }
            validate_real_target_arrays(arrays, crop_id)
            npz_path = out_dir / f"{crop_id}.npz"
            json_path = out_dir / f"{crop_id}.json"
            np.savez_compressed(npz_path, **arrays)
            metadata = crop_metadata(record, sample, crop_id, split, y0, x0, crop_size)
            json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rows.append(manifest_row(
                crop_id, split, record.sample_id, record.split_group_id or record.sample_id,
                npz_path, json_path, out_dir, crop_semantic,
            ))
    validate_generated_crop_splits(records, rows)
    write_manifest(out_dir / "real_crop_manifest.csv", rows)
    return rows


def image_splits(
    records: list[RealAnnotationRecord],
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    leave_one_out_image: str | None,
) -> dict[str, str]:
    ids = [record.sample_id for record in records]
    group_by_id = {record.sample_id: record.split_group_id or record.sample_id for record in records}
    if leave_one_out_image is not None:
        if leave_one_out_image not in ids:
            raise ValueError(f"leave-one-out image {leave_one_out_image!r} is not in the manifest")
        held_group = group_by_id[leave_one_out_image]
        return {sample_id: ("validation" if group_by_id[sample_id] == held_group else "train") for sample_id in ids}
    rng = np.random.default_rng(seed)
    shuffled = sorted(set(group_by_id.values()))
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * validation_fraction))
    if n > 1 and n_val == 0 and validation_fraction > 0:
        n_val = 1
    test_ids = set(shuffled[:n_test])
    val_ids = set(shuffled[n_test : n_test + n_val])
    group_splits = {
        group: "test" if group in test_ids else "validation" if group in val_ids else "train"
        for group in shuffled
    }
    return {sample_id: group_splits[group_by_id[sample_id]] for sample_id in ids}


def crop_origins(shape: tuple[int, int], crop_size: int, stride: int) -> list[tuple[int, int]]:
    h, w = shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"crop_size {crop_size} exceeds image shape {shape}")
    ys = axis_origins(h, crop_size, stride)
    xs = axis_origins(w, crop_size, stride)
    return [(y, x) for y in ys for x in xs]


def axis_origins(length: int, crop_size: int, stride: int) -> list[int]:
    starts = list(range(0, length - crop_size + 1, stride))
    if starts[-1] != length - crop_size:
        starts.append(length - crop_size)
    return starts


def as_uint8(image_uint8: np.ndarray | None, image_float: np.ndarray | None) -> np.ndarray:
    if image_uint8 is not None:
        return image_uint8.astype(np.uint8, copy=False)
    assert image_float is not None
    arr = image_float.astype(np.float32, copy=False)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def crop_metadata(
    record: RealAnnotationRecord,
    sample: dict[str, Any],
    crop_id: str,
    split: str,
    y0: int,
    x0: int,
    crop_size: int,
) -> dict[str, Any]:
    return {
        "dataset_schema_version": "real_annotation_crop_0.1.0",
        "source_kind": "real_annotation",
        "sample_id": crop_id,
        "source_image_id": record.sample_id,
        "split": split,
        "crop_origin_yx": [int(y0), int(x0)],
        "crop_size": int(crop_size),
        "image_path": str(record.image_path),
        "labels_path": str(record.labels_path),
        "snakes_path": str(record.snakes_path),
        "notes": record.notes,
        "split_group_id": record.split_group_id or record.sample_id,
        "source_metadata": record.metadata or {},
        "annotation_metadata": sample["metadata"],
    }


def manifest_row(
    crop_id: str,
    split: str,
    source_image_id: str,
    split_group_id: str,
    npz_path: Path,
    json_path: Path,
    root: Path,
    semantic: np.ndarray,
) -> dict[str, str]:
    categories = real_patch_categories(semantic)
    return {
        "sample_id": crop_id,
        "split": split,
        "source_kind": "real",
        "source_image_id": source_image_id,
        "split_group_id": split_group_id,
        "sampling_categories": ";".join(categories),
        "fibrous_tau_pixels": str(int(np.count_nonzero(semantic == 1))),
        "clump_pixels": str(int(np.count_nonzero(semantic == 3))),
        "uncertain_ignore_pixels": str(int(np.count_nonzero(semantic == 255))),
        "npz_path": npz_path.relative_to(root).as_posix(),
        "npz_sha256": sha256(npz_path),
        "json_path": json_path.relative_to(root).as_posix(),
        "json_sha256": sha256(json_path),
        "schema_version": "real_annotation_crop_0.1.0",
        "generator_version": "real_crop_builder_0.1.0",
    }


def real_patch_categories(semantic: np.ndarray, dense_fraction: float = 0.05) -> list[str]:
    fibrous = semantic == 1
    clump = semantic == 3
    uncertain = semantic == 255
    categories = []
    if fibrous.any():
        categories.append("fibrous_positive")
    if clump.any():
        categories.append("clump_positive")
    if uncertain.any():
        categories.append("uncertain_positive")
    if float(fibrous.mean()) >= dense_fraction or masks_touch(fibrous, clump):
        categories.append("dense_or_fibrous_clump_boundary")
    if not fibrous.any() and not clump.any():
        categories.append("background_hard_negative")
    categories.append("uniform_random")
    return categories


def masks_touch(left: np.ndarray, right: np.ndarray) -> bool:
    if not left.any() or not right.any():
        return False
    padded = np.pad(left, 1)
    neighbours = np.zeros_like(left)
    for y in range(3):
        for x in range(3):
            neighbours |= padded[y:y + left.shape[0], x:x + left.shape[1]]
    return bool(np.any(neighbours & right))


def validate_generated_crop_splits(
    records: list[RealAnnotationRecord],
    rows: list[dict[str, str]],
) -> None:
    parent_split = {record.sample_id: record.split for record in records if record.split}
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        expected = parent_split.get(row["source_image_id"])
        if expected is not None and row["split"] != expected:
            raise ValueError(f"{row['sample_id']}: crop split differs from parent image")
        group_splits.setdefault(row["split_group_id"], set()).add(row["split"])
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"split groups cross crop splits: {leaked}")


def validate_real_target_arrays(arrays: dict[str, np.ndarray], sample_id: str) -> None:
    semantic = arrays["real_compatible_semantic_mask"]
    unexpected = sorted(set(np.unique(semantic).tolist()) - {0, 1, 3, 255})
    if unexpected:
        raise ValueError(f"{sample_id}: unexpected label values {unexpected}")
    expected = {
        "real_compatible_fibrous_mask": semantic == 1,
        "real_compatible_clump_mask": semantic == 3,
        "real_compatible_uncertain_ignore_mask": semantic == 255,
    }
    for name, target in expected.items():
        if not np.array_equal(arrays[name].astype(bool), target):
            raise ValueError(f"{sample_id}: {name} leaks or disagrees with semantic target")
    skeleton = arrays["real_compatible_skeleton_mask"].astype(bool)
    if np.any(skeleton & (semantic != 1)):
        raise ValueError(f"{sample_id}: skeleton pixels outside confident fibrous regions")
    valid = arrays["real_compatible_skeleton_valid_mask"].astype(bool)
    if np.any(valid & (semantic == 255)):
        raise ValueError(f"{sample_id}: skeleton loss includes uncertain_ignore")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("real crop builder produced no crops")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
