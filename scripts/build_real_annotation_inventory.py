#!/usr/bin/env python
"""Build and validate the expert real-annotation inventory and crop plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fibras.annotations.inventory import (
    CROP_FIELDS,
    INVENTORY_FIELDS,
    build_real_annotation_inventory,
    write_csv,
)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        rows, crops, audit = build_real_annotation_inventory(
            args.annotation_root,
            args.image_root,
            args.image_manifest,
            args.split_manifest,
            crop_size=args.crop_size,
            stride=args.stride,
        )
        if args.expected_count is not None and len(rows) != args.expected_count:
            raise ValueError(f"expected {args.expected_count} annotated images, found {len(rows)}")
        outputs = render_outputs(rows, crops, audit)
        if audit["validation_problems"] and not args.allow_invalid:
            raise ValueError(
                "real annotation validation failed:\n- "
                + "\n- ".join(audit["validation_problems"])
                + "\nUse --allow-invalid only to write an audit that excludes invalid images from crops."
            )
        if args.check:
            mismatches = [str(path) for path, content in output_paths(args, outputs) if not path.exists() or path.read_bytes() != content]
            if mismatches:
                raise ValueError(f"generated artifacts are stale or missing: {mismatches}")
        else:
            for path, content in output_paths(args, outputs):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    action = "validated" if args.check else "wrote"
    print(f"{action} {len(rows)} full-image records and {len(crops)} crop records")
    return 0


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    out.add_argument("--annotation-root", type=Path, default=Path("/cephfs/mhuang/STED_dataset/manual_annotations"))
    out.add_argument("--image-root", type=Path, default=Path("/ssd/STED_dataset/data"))
    out.add_argument("--image-manifest", type=Path, default=Path("data_manifests/sted_images.csv"))
    out.add_argument("--split-manifest", type=Path, default=Path("data_manifests/sted_splits.csv"))
    out.add_argument("--inventory-out", type=Path, default=Path("data_manifests/real_annotations.csv"))
    out.add_argument("--crops-out", type=Path, default=Path("data_manifests/real_annotation_crops.csv"))
    out.add_argument("--audit-json-out", type=Path, default=Path("data_manifests/real_annotation_audit.json"))
    out.add_argument("--audit-markdown-out", type=Path, default=Path("docs/real_annotation_audit.md"))
    out.add_argument("--crop-size", type=int, default=128)
    out.add_argument("--stride", type=int, default=128)
    out.add_argument("--expected-count", type=int, default=20)
    out.add_argument("--check", action="store_true")
    out.add_argument("--allow-invalid", action="store_true")
    return out


def render_outputs(
    rows: list[dict[str, str]],
    crops: list[dict[str, str]],
    audit: dict,
) -> tuple[bytes, bytes, bytes, bytes]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inventory = root / "inventory.csv"
        crop_manifest = root / "crops.csv"
        write_csv(inventory, rows, INVENTORY_FIELDS)
        write_csv(crop_manifest, crops, CROP_FIELDS)
        return (
            inventory.read_bytes(),
            crop_manifest.read_bytes(),
            (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode(),
            audit_markdown(audit).encode(),
        )


def output_paths(args: argparse.Namespace, outputs: tuple[bytes, ...]):
    paths = (args.inventory_out, args.crops_out, args.audit_json_out, args.audit_markdown_out)
    return zip(paths, outputs)


def audit_markdown(audit: dict) -> str:
    lines = [
        "# Real Annotation Audit",
        "",
        f"Annotated images: {audit['annotated_image_count']}",
        "",
        "## Metadata Distribution",
        "",
        "| Field | Counts |",
        "|:--|:--|",
    ]
    for field, counts in audit["metadata_distribution"].items():
        values = ", ".join(f"`{key}`: {value}" for key, value in counts.items())
        lines.append(f"| `{field}` | {values} |")
    lines += ["", "## Class Statistics", "", "| Class | Pixels | Components |", "|:--|--:|--:|"]
    for name, stats in audit["class_statistics"].items():
        lines.append(f"| `{name}` | {stats['pixels']} | {stats['components']} |")
    skeleton = audit["skeleton_annotations"]
    lines += [
        "", "## Skeleton Annotations", "",
        f"Available for {skeleton['images_available']} images. {skeleton['usable_snakes']} of "
        f"{skeleton['total_snakes']} snakes are usable; {skeleton['flagged_snakes']} are flagged. "
        f"Target construction clipped {skeleton['nonfibrous_pixels_clipped']} rasterized snake pixels "
        "outside confident fibrous regions.",
        "", "## Ambiguous Or Missing Metadata", "",
    ]
    lines.extend(f"- {item}" for item in audit["ambiguous_or_missing_metadata"])
    lines += ["", "## Manual Resolution", ""]
    lines.extend(f"- {item}" for item in audit["manual_resolution_required"])
    lines += ["", "## Validation Problems", ""]
    lines.extend(f"- {item}" for item in audit["validation_problems"] or ["None."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
