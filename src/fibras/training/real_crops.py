"""Real annotation crop manifests for optional fine-tuning."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fibras.annotations import build_real_annotation_sample


@dataclass(frozen=True)
class RealAnnotationRecord:
    sample_id: str
    image_path: Path
    labels_path: Path
    snakes_path: Path
    notes: str = ""
    split: str | None = None


def read_real_annotation_manifest(path: Path) -> list[RealAnnotationRecord]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"sample_id", "image_path", "labels_path", "snakes_path"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
    return [
        RealAnnotationRecord(
            sample_id=row["sample_id"],
            image_path=resolve_path(row["image_path"], path.parent),
            labels_path=resolve_path(row["labels_path"], path.parent),
            snakes_path=resolve_path(row["snakes_path"], path.parent),
            notes=row.get("notes", ""),
            split=row.get("split") or None,
        )
        for row in rows
    ]


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
        RealAnnotationRecord(safe_stem(image), image, labels, image.with_suffix(".txt"))
        for image in images
        for labels in [matching_label_path(image)]
        if labels is not None and image.with_suffix(".txt").exists()
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
            }
            npz_path = out_dir / f"{crop_id}.npz"
            json_path = out_dir / f"{crop_id}.json"
            np.savez_compressed(npz_path, **arrays)
            metadata = crop_metadata(record, sample, crop_id, split, y0, x0, crop_size)
            json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rows.append(manifest_row(crop_id, split, record.sample_id, npz_path, json_path, out_dir))
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
    if leave_one_out_image is not None:
        if leave_one_out_image not in ids:
            raise ValueError(f"leave-one-out image {leave_one_out_image!r} is not in the manifest")
        return {sample_id: ("validation" if sample_id == leave_one_out_image else "train") for sample_id in ids}
    rng = np.random.default_rng(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * validation_fraction))
    if n > 1 and n_val == 0 and validation_fraction > 0:
        n_val = 1
    test_ids = set(shuffled[:n_test])
    val_ids = set(shuffled[n_test : n_test + n_val])
    return {
        sample_id: "test" if sample_id in test_ids else "validation" if sample_id in val_ids else "train"
        for sample_id in ids
    }


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
        "annotation_metadata": sample["metadata"],
    }


def manifest_row(
    crop_id: str,
    split: str,
    source_image_id: str,
    npz_path: Path,
    json_path: Path,
    root: Path,
) -> dict[str, str]:
    return {
        "sample_id": crop_id,
        "split": split,
        "source_kind": "real",
        "source_image_id": source_image_id,
        "npz_path": npz_path.relative_to(root).as_posix(),
        "npz_sha256": sha256(npz_path),
        "json_path": json_path.relative_to(root).as_posix(),
        "json_sha256": sha256(json_path),
        "schema_version": "real_annotation_crop_0.1.0",
        "generator_version": "real_crop_builder_0.1.0",
    }


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
