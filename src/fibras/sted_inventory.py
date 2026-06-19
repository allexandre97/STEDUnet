"""Reproducible STED source-image inventories."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageSequence

from .sted_filename_parser import parse_sted_filename


IMAGE_FIELDS = [
    "source_root_id",
    "relative_path",
    "source_kind",
    "source_sha256",
    "pixel_sha256",
    "stable_image_id",
    "shape_y",
    "shape_x",
    "frames",
    "channels",
    "dtype",
    "bit_depth",
    "min",
    "max",
    "mean",
    "std",
    "p0",
    "p0_1",
    "p1",
    "p5",
    "p25",
    "p50",
    "p75",
    "p95",
    "p99",
    "p99_9",
    "p100",
    "saturation_value",
    "saturation_count",
    "zero_count",
    "culture_id",
    "disease",
    "tau_isoform",
    "experimental_condition",
    "div",
    "div_token",
    "series_index",
    "filename_prefix",
    "acquisition_group",
    "experimental_group_id",
    "parse_status",
    "deprecated_inferred_pn",
    "deprecated_inferred_round",
    "deprecated_inferred_condition",
    "deprecated_inferred_div",
    "deprecated_biological_group_candidate",
    "file_duplicate_group",
    "pixel_duplicate_group",
    "thumbnail_duplicate_group",
    "near_duplicate_candidate_group",
    "near_duplicate_score",
    "near_duplicate_method",
    "validation_flags",
    "intended_split_status",
]

BLANK_EXTRA_FIELDS = [
    "blank_status",
    "validation_source",
    "validator_role",
    "validation_date",
    "notes",
]

PERCENTILES = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
PERCENTILE_NAMES = ["p0", "p0_1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99_9", "p100"]
NEAR_DUPLICATE_METHOD = "normalized_thumbnail_ncc_threshold_0.995"


@dataclass(frozen=True)
class SourceRoot:
    root_id: str
    path: Path
    source_kind: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_sha256(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def stable_image_id(source_root_id: str, relative_path: str, source_hash: str) -> str:
    raw = f"{source_root_id}:{relative_path}:{source_hash}".encode("utf-8")
    return "img_" + hashlib.sha256(raw).hexdigest()[:20]


def read_image(path: Path) -> tuple[np.ndarray, int, str]:
    with Image.open(path) as im:
        frames = [np.asarray(frame).copy() for frame in ImageSequence.Iterator(im)]
        mode = im.mode
    if not frames:
        raise ValueError("no frames found")
    return frames[0], len(frames), mode


def channel_count(arr: np.ndarray, mode: str) -> int:
    if arr.ndim == 2:
        return 1
    if arr.ndim == 3:
        return int(arr.shape[-1])
    if mode == "L":
        return 1
    return 1


def thumbnail(arr: np.ndarray, size: int = 32) -> np.ndarray:
    gray = arr
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    gray = gray.astype(np.float32)
    lo, hi = np.percentile(gray, [1, 99])
    scaled = np.zeros_like(gray, dtype=np.float32) if hi <= lo else np.clip((gray - lo) / (hi - lo), 0, 1)
    img = Image.fromarray((scaled * 255).astype(np.uint8))
    return np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def thumbnail_hash(arr: np.ndarray) -> str:
    thumb = thumbnail(arr)
    h = hashlib.sha256()
    h.update(str(thumb.shape).encode("utf-8"))
    h.update(thumb.tobytes())
    return h.hexdigest()


def normalized_thumbnail_vector(arr: np.ndarray) -> np.ndarray:
    vec = thumbnail(arr).astype(np.float32).ravel()
    vec -= float(vec.mean())
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size:
        return 0.0
    return float(np.dot(a, b))


def quality_flags(arr: np.ndarray, expected_shape: tuple[int, int] | None = None) -> list[str]:
    flags: list[str] = []
    if expected_shape and tuple(arr.shape[:2]) != expected_shape:
        flags.append("unexpected_shape")
    if arr.ndim > 3:
        flags.append("unexpected_dimensions")
    if arr.size == 0:
        flags.append("empty_image")
        return flags
    if np.min(arr) == np.max(arr):
        flags.append("constant_image")
    if np.issubdtype(arr.dtype, np.integer):
        max_dtype = np.iinfo(arr.dtype).max
        if int(np.sum(arr == max_dtype)):
            flags.append("saturation")
    vals = arr.astype(np.float32).ravel()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    p999 = float(np.percentile(vals, 99.9))
    threshold = max(p999, med + 8 * 1.4826 * mad, 20.0)
    high = arr > threshold
    if high.mean() < 0.02 and elongated_component_count(high) > 0:
        flags.append("elongated_high_intensity_component")
    std = float(np.std(vals)) or 1.0
    if arr.ndim == 2:
        row_var = float(np.std(arr.mean(axis=1)) / std)
        col_var = float(np.std(arr.mean(axis=0)) / std)
        if row_var > 0.65 or col_var > 0.65:
            flags.append("strong_row_or_column_variation")
    return flags


def elongated_component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        comp_y: list[int] = []
        comp_x: list[int] = []
        while stack:
            y, x = stack.pop()
            comp_y.append(y)
            comp_x.append(x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        area = len(comp_y)
        if area < 12:
            continue
        height = max(comp_y) - min(comp_y) + 1
        width = max(comp_x) - min(comp_x) + 1
        elongation = max(height, width) / max(1, min(height, width))
        if elongation >= 2.5:
            count += 1
    return count


def collect_records(root: SourceRoot) -> list[dict[str, str]]:
    files = sorted(p for p in root.path.rglob("*") if p.is_file())
    records: list[dict[str, str]] = []
    expected_shape: tuple[int, int] | None = None
    for path in files:
        rel = path.relative_to(root.path).as_posix()
        parsed = parse_sted_filename(rel)
        base = {
            "source_root_id": root.root_id,
            "relative_path": rel,
            "source_kind": root.source_kind,
            "intended_split_status": "unassigned",
        }
        try:
            source_hash = sha256_file(path)
            arr, frames, mode = read_image(path)
            if expected_shape is None and arr.ndim >= 2:
                expected_shape = tuple(arr.shape[:2])
            pix_hash = pixel_sha256(arr)
            vals = arr.astype(np.float64).ravel()
            pcts = dict(zip(PERCENTILE_NAMES, np.percentile(vals, PERCENTILES)))
            sat_value = int(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else float(np.max(vals))
            flags = quality_flags(arr, expected_shape)
            record = {
                **base,
                "source_sha256": source_hash,
                "pixel_sha256": pix_hash,
                "stable_image_id": stable_image_id(root.root_id, rel, source_hash),
                "shape_y": str(arr.shape[0]) if arr.ndim >= 2 else "not_applicable",
                "shape_x": str(arr.shape[1]) if arr.ndim >= 2 else "not_applicable",
                "frames": str(frames),
                "channels": str(channel_count(arr, mode)),
                "dtype": str(arr.dtype),
                "bit_depth": str(arr.dtype.itemsize * 8),
                "min": format_float(float(np.min(vals))),
                "max": format_float(float(np.max(vals))),
                "mean": format_float(float(np.mean(vals))),
                "std": format_float(float(np.std(vals))),
                **{k: format_float(float(v)) for k, v in pcts.items()},
                "saturation_value": str(sat_value),
                "saturation_count": str(int(np.sum(arr == sat_value))),
                "zero_count": str(int(np.sum(arr == 0))),
                **parsed_manifest_fields(parsed),
                "file_duplicate_group": "none",
                "pixel_duplicate_group": "none",
                "thumbnail_duplicate_group": "none",
                "near_duplicate_candidate_group": "none",
                "near_duplicate_score": "not_applicable",
                "near_duplicate_method": NEAR_DUPLICATE_METHOD,
                "validation_flags": ";".join(flags) if flags else "none",
                "_thumbnail_hash": thumbnail_hash(arr),
                "_thumbnail_vec": normalized_thumbnail_vector(arr),
            }
        except Exception as exc:  # include unreadable files instead of silently excluding them
            source_hash = sha256_file(path) if path.exists() else "not_available"
            record = {
                **base,
                "source_sha256": source_hash,
                "pixel_sha256": "not_available",
                "stable_image_id": stable_image_id(root.root_id, rel, source_hash),
                "shape_y": "not_readable",
                "shape_x": "not_readable",
                "frames": "not_readable",
                "channels": "not_readable",
                "dtype": "not_readable",
                "bit_depth": "not_readable",
                "min": "not_readable",
                "max": "not_readable",
                "mean": "not_readable",
                "std": "not_readable",
                **{k: "not_readable" for k in PERCENTILE_NAMES},
                "saturation_value": "not_readable",
                "saturation_count": "not_readable",
                "zero_count": "not_readable",
                **parsed_manifest_fields(parsed),
                "file_duplicate_group": "none",
                "pixel_duplicate_group": "none",
                "thumbnail_duplicate_group": "none",
                "near_duplicate_candidate_group": "none",
                "near_duplicate_score": "not_applicable",
                "near_duplicate_method": NEAR_DUPLICATE_METHOD,
                "validation_flags": f"unreadable:{type(exc).__name__}",
                "_thumbnail_hash": "",
                "_thumbnail_vec": np.zeros(32 * 32, dtype=np.float32),
            }
        if root.source_kind == "blank_background":
            record.update(
                {
                    "blank_status": "expert_validated",
                    "validation_source": "human_expert_review",
                    "validator_role": "STED expert",
                    "validation_date": "not_recorded",
                    "notes": "Expert-validated fiber-free blank; automated flags are QC only.",
                }
            )
        records.append(record)
    annotate_duplicates(records)
    return records


def parsed_manifest_fields(parsed) -> dict[str, str]:
    return {
        "culture_id": parsed.culture_id,
        "disease": parsed.disease,
        "tau_isoform": parsed.tau_isoform,
        "experimental_condition": parsed.experimental_condition,
        "div": parsed.div,
        "div_token": parsed.div_token,
        "series_index": parsed.series_index,
        "filename_prefix": parsed.prefix,
        "acquisition_group": parsed.acquisition_group,
        "experimental_group_id": parsed.experimental_group_id,
        "parse_status": parsed.parse_status,
        "deprecated_inferred_pn": parsed.culture_id,
        "deprecated_inferred_round": parsed.tau_isoform,
        "deprecated_inferred_condition": parsed.disease,
        "deprecated_inferred_div": parsed.div_token,
        "deprecated_biological_group_candidate": parsed.experimental_group_id,
    }


def format_float(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.6g}"


def annotate_duplicates(records: list[dict[str, str]]) -> None:
    for key, field in [
        ("source_sha256", "file_duplicate_group"),
        ("pixel_sha256", "pixel_duplicate_group"),
        ("_thumbnail_hash", "thumbnail_duplicate_group"),
    ]:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for rec in records:
            value = rec.get(key, "")
            if value and value not in {"not_available", "not_readable"}:
                groups[value].append(rec)
        idx = 1
        for group in groups.values():
            if len(group) > 1:
                gid = f"{field}_{idx:03d}"
                for rec in group:
                    rec[field] = gid
                idx += 1

    near_idx = 1
    readable = [r for r in records if r.get("pixel_sha256") not in {"not_available", "not_readable"}]
    for i, left in enumerate(readable):
        for right in readable[i + 1 :]:
            if left["thumbnail_duplicate_group"] != "none" and left["thumbnail_duplicate_group"] == right["thumbnail_duplicate_group"]:
                continue
            if left.get("shape_y") != right.get("shape_y") or left.get("shape_x") != right.get("shape_x"):
                continue
            score = ncc(left["_thumbnail_vec"], right["_thumbnail_vec"])
            if score >= 0.995:
                gid = left.get("near_duplicate_candidate_group")
                if gid == "none":
                    gid = right.get("near_duplicate_candidate_group")
                if gid == "none":
                    gid = f"near_duplicate_candidate_{near_idx:03d}"
                    near_idx += 1
                for rec in (left, right):
                    rec["near_duplicate_candidate_group"] = gid
                    rec["near_duplicate_score"] = format_float(score)


def public_record(record: dict[str, str], blank: bool = False) -> dict[str, str]:
    fields = IMAGE_FIELDS + (BLANK_EXTRA_FIELDS if blank else [])
    return {field: str(record.get(field, "")) for field in fields}


def write_csv(path: Path, records: Iterable[dict[str, str]], blank: bool = False) -> None:
    fields = IMAGE_FIELDS + (BLANK_EXTRA_FIELDS if blank else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(public_record(record, blank=blank))


def write_acquisition_groups(path: Path, records: list[dict[str, str]]) -> None:
    fields = [
        "source_kind",
        "culture_id",
        "disease",
        "tau_isoform",
        "experimental_condition",
        "div",
        "div_token",
        "experimental_group_id",
        "acquisition_group",
        "image_count",
        "root_ids",
        "series_indices",
    ]
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for rec in records:
        groups[(rec["source_kind"], rec["experimental_group_id"], rec["acquisition_group"])].append(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for (kind, group_id, acquisition), rows in sorted(groups.items()):
            first = rows[0]
            writer.writerow(
                {
                    "source_kind": kind,
                    "culture_id": first["culture_id"],
                    "disease": first["disease"],
                    "tau_isoform": first["tau_isoform"],
                    "experimental_condition": first["experimental_condition"],
                    "div": first["div"],
                    "div_token": ";".join(sorted({r["div_token"] for r in rows})),
                    "experimental_group_id": group_id,
                    "acquisition_group": acquisition,
                    "image_count": len(rows),
                    "root_ids": ";".join(sorted({r["source_root_id"] for r in rows})),
                    "series_indices": ";".join(sorted({r["series_index"] for r in rows}, key=sort_key)),
                }
            )


def sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):08d}") if value.isdigit() else (1, value)


def summarize(records: list[dict[str, str]]) -> dict[str, object]:
    readable = [r for r in records if r["dtype"] != "not_readable"]
    total_pixels = sum(int(r["shape_y"]) * int(r["shape_x"]) for r in readable)
    return {
        "files": len(records),
        "readable": len(readable),
        "total_pixels": total_pixels,
        "shapes": sorted({f"{r['shape_y']}x{r['shape_x']}" for r in readable}),
        "dtypes": sorted({r["dtype"] for r in readable}),
        "global_percentiles": exact_global_percentiles(records),
        "saturation_count": sum(int(r["saturation_count"]) for r in readable if r["saturation_count"].isdigit()),
        "validation_flag_counts": flag_counts(records),
    }


def exact_global_percentiles(records: list[dict[str, str]]) -> dict[str, str]:
    # Inventory records hold per-image statistics. Full global percentiles are
    # computed by the CLI before writing records, where source paths are known.
    return {}


def flag_counts(records: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        for flag in rec["validation_flags"].split(";"):
            counts[flag or "none"] += 1
    return dict(sorted(counts.items()))


def compute_global_percentiles(root: SourceRoot) -> dict[str, str]:
    arrays = []
    for path in sorted(p for p in root.path.rglob("*") if p.is_file()):
        try:
            arr, _, _ = read_image(path)
        except Exception:
            continue
        arrays.append(arr.ravel())
    if not arrays:
        return {name: "not_available" for name in PERCENTILE_NAMES}
    vals = np.concatenate(arrays).astype(np.float64)
    return {name: format_float(float(v)) for name, v in zip(PERCENTILE_NAMES, np.percentile(vals, PERCENTILES))}


def write_report(path: Path, fiber_records: list[dict[str, str]], blank_records: list[dict[str, str]], fiber_root: SourceRoot, blank_root: SourceRoot) -> None:
    fiber_global = compute_global_percentiles(fiber_root)
    blank_global = compute_global_percentiles(blank_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# STED data inventory",
        "",
        "## Methodology",
        "",
        "- Reader: Pillow (`PIL.Image`).",
        "- Statistics in manifests are per image and exact over all decoded pixels.",
        "- Global percentiles in this report are exact over all readable decoded pixels.",
        "- Source identity uses source root ID, relative path, source SHA-256, pixel SHA-256, and stable image ID.",
        "- File duplicates use full-file SHA-256 equality.",
        "- Pixel duplicates use decoded array shape, dtype, and bytes.",
        "- Thumbnail duplicates use deterministic p1/p99-normalized 32x32 thumbnail hashes.",
        "- Near-duplicate candidates use normalized thumbnail NCC with threshold 0.995, excluding exact thumbnail duplicates.",
        "- Elongated-component QC threshold is `max(p99.9, median + 8 * 1.4826 * MAD, 20)` with area >=12 px and elongation >=2.5.",
        "- Filename metadata: `PN###` is recorded as `culture_id`; `3R`/`4R` are `tau_isoform`; `AD`, `PID`, `PSP`, and `CBD` are disease labels; `DIV` is parsed as an integer time point plus original token.",
        "- `experimental_condition = disease + '_' + tau_isoform`; `experimental_group_id = culture_id + experimental_condition + canonical DIV`.",
        "- Deprecated inferred columns are retained only for migration compatibility and are not used for new grouping logic.",
        "",
        "## Fiber images",
        "",
        f"- Files: {len(fiber_records)}",
        f"- Readable: {sum(r['dtype'] != 'not_readable' for r in fiber_records)}",
        f"- Global percentiles: {json.dumps(fiber_global, sort_keys=True)}",
        f"- Validation flags: {json.dumps(flag_counts(fiber_records), sort_keys=True)}",
        f"- Cultures: {json.dumps(count_values(fiber_records, 'culture_id'), sort_keys=True)}",
        f"- Diseases: {json.dumps(count_values(fiber_records, 'disease'), sort_keys=True)}",
        f"- Tau isoforms: {json.dumps(count_values(fiber_records, 'tau_isoform'), sort_keys=True)}",
        f"- DIVs: {json.dumps(count_values(fiber_records, 'div'), sort_keys=True)}",
        f"- Experimental conditions: {json.dumps(count_values(fiber_records, 'experimental_condition'), sort_keys=True)}",
        "",
        "## Expert-validated blanks",
        "",
        f"- Files: {len(blank_records)}",
        f"- Readable: {sum(r['dtype'] != 'not_readable' for r in blank_records)}",
        f"- Blank status: expert_validated",
        f"- Validation source: human_expert_review",
        f"- Global percentiles: {json.dumps(blank_global, sort_keys=True)}",
        f"- Validation flags: {json.dumps(flag_counts(blank_records), sort_keys=True)}",
        f"- Cultures: {json.dumps(count_values(blank_records, 'culture_id'), sort_keys=True)}",
        f"- Diseases: {json.dumps(count_values(blank_records, 'disease'), sort_keys=True)}",
        f"- Tau isoforms: {json.dumps(count_values(blank_records, 'tau_isoform'), sort_keys=True)}",
        f"- DIVs: {json.dumps(count_values(blank_records, 'div'), sort_keys=True)}",
        f"- Experimental conditions: {json.dumps(count_values(blank_records, 'experimental_condition'), sort_keys=True)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def count_values(records: list[dict[str, str]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[rec.get(field, "unknown")] += 1
    return dict(sorted(counts.items()))


def run_inventory(
    fiber_root_id: str,
    fiber_dir: Path,
    blank_root_id: str,
    blank_dir: Path,
    out_dir: Path,
    report_path: Path,
) -> None:
    fiber_root = SourceRoot(fiber_root_id, fiber_dir, "fiber_image")
    blank_root = SourceRoot(blank_root_id, blank_dir, "blank_background")
    fiber_records = collect_records(fiber_root)
    blank_records = collect_records(blank_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "sted_images.csv", fiber_records, blank=False)
    write_csv(out_dir / "sted_blanks.csv", blank_records, blank=True)
    write_acquisition_groups(out_dir / "acquisition_groups.csv", fiber_records + blank_records)
    write_report(report_path, fiber_records, blank_records, fiber_root, blank_root)
