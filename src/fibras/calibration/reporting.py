"""Exploratory calibration report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_report(
    artifact_dir: Path,
    synthetic_dir: Path,
    composite_dir: Path,
    out_path: Path,
    config: dict[str, Any] | None = None,
    pure_blank_dir: Path | None = None,
    structural_qa_dir: Path | None = None,
    optical_qa_dir: Path | None = None,
    review_package_dir: Path | None = None,
) -> None:
    assets = out_path.parent / (out_path.stem + "_assets")
    assets.mkdir(parents=True, exist_ok=True)
    for stale in assets.glob("*.png"):
        stale.unlink()
    summary = json.loads((artifact_dir / "appearance_summary.json").read_text(encoding="utf-8"))
    real_rows = read_csv(artifact_dir / "real_fiber_stats.csv")
    blank_rows = read_csv(artifact_dir / "blank_stats.csv")
    proxy_rows = read_csv(artifact_dir / "proxy_stats.csv")
    lines = [
        "# Exploratory STED appearance calibration report",
        "",
        "## Status",
        "",
        "- calibration_status: `exploratory_unpartitioned`",
        "- Result is exploratory, not approved calibration.",
        "- Current split assignments are provisional and must not be interpreted as biologically independent.",
        "- Synthetic images are not labelled `empirically_matched`.",
        "",
        "## Artifact metadata",
        "",
        f"- Source commit: `{summary['metadata'].get('source_commit_sha', summary['metadata'].get('code_version', 'not_available'))}`",
        f"- Working tree dirty: `{summary['metadata'].get('working_tree_dirty', 'not_recorded')}`",
        f"- Generation config SHA-256: `{summary['metadata'].get('generation_config_sha256', 'not_recorded')}`",
        f"- Inventory manifest SHA-256: `{summary['metadata'].get('inventory_manifest_sha256', 'not_recorded')}`",
        f"- Split manifest SHA-256: `{summary['metadata'].get('split_manifest_sha256', 'not_recorded')}`",
        f"- Blank-pool manifest SHA-256: `{summary['metadata'].get('blank_pool_manifest_sha256', 'not_recorded')}`",
        f"- Date generated: `{summary['metadata']['date_generated']}`",
        f"- Fiber images characterized: {len(real_rows)}",
        f"- Blank images characterized: {len(blank_rows)}",
        "",
        "## Quantitative comparison summary",
        "",
    ]
    lines.extend(summary_table(summary))
    lines.extend(
        [
            "",
            "## Proxy-estimate warning",
            "",
            "Foreground occupancy, ridge response, orientation, apparent width, component length, SNR, endpoint density, and crossing density are proxy estimates only. They are threshold-sensitive and are not ground truth.",
            "",
            "## Representative plots",
            "",
        ]
    )
    histogram_path = assets / "intensity_percentile_summary.png"
    draw_histogram_comparison(real_rows, blank_rows, histogram_path)
    power_path = assets / "normalized_spectral_shape_comparison.png"
    draw_bar_comparison(summary, power_path)
    proxy_path = assets / "proxy_comparison.png"
    draw_proxy_plot(proxy_rows, proxy_path)
    if config and "source_roots" in config:
        real_contact = assets / "representative_real_fiber_images.png"
        blank_contact = assets / "representative_blank_images.png"
        draw_source_contact_sheet(
            real_rows, config["source_roots"], real_contact, "real fiber"
        )
        draw_source_contact_sheet(
            blank_rows, config["source_roots"], blank_contact, "expert blank"
        )
    profile_path = assets / "transverse_profile_proxy.png"
    draw_transverse_profile(composite_dir, profile_path)
    lines.append(
        "- Full local analysis assets are reproducible but intentionally ignored by Git; "
        "the committed curated plots are linked below."
    )
    lines.extend(["", "## Example sets", ""])
    lines.append(f"- Artificial-background synthetic examples: `{synthetic_dir}`")
    lines.append(f"- Real-blank composite examples: `{composite_dir}`")
    if pure_blank_dir:
        lines.append(f"- Pure blank QA examples: `{pure_blank_dir}`")
    width_lines = width_calibration_lines(synthetic_dir)
    if width_lines:
        lines.extend(["", "## Width and PSF checks", ""])
        lines.extend(width_lines)
    lines.append("")
    lines.append("Representative overlays for composites are generated separately by the visualization script and remain labelled exploratory.")
    lines.extend(["", "## Mismatches and unresolved uncertainties", ""])
    lines.extend(mismatch_lines(summary))
    matched_path = artifact_dir / "real_blank_composite_matched_blank_delta_stats.csv"
    if matched_path.exists():
        lines.extend(["", "## Matched source-blank deltas", ""])
        lines.extend(matched_delta_lines(read_csv(matched_path)))
    package_dir = review_package_dir or out_path.parent / "review_package"
    package = build_review_package(
        package_dir,
        summary,
        synthetic_dir,
        composite_dir,
        structural_qa_dir,
        optical_qa_dir,
        read_csv(matched_path) if matched_path.exists() else [],
    )
    if package:
        lines.extend(["", "## Curated review package", ""])
        for label, path in package:
            relative = path.relative_to(out_path.parent)
            lines.append(f"- [{label}]({relative.as_posix()})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def width_calibration_lines(synthetic_dir: Path) -> list[str]:
    manifest = synthetic_dir / "dataset_manifest.csv"
    if not manifest.exists():
        return []
    rows = read_csv(manifest)
    if not rows:
        return []
    meta_path = synthetic_dir / rows[0]["json_path"]
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    width = meta.get("width_calibration", {})
    render = meta.get("rendering_report", {})
    return [
        f"- Apparent in-focus FWHM: `{width.get('measured_fwhm_px', 'n/a')}` px; target `{width.get('target_fwhm_px', 'n/a')}` px.",
        f"- PSF mode: `{render.get('psf_mode', 'n/a')}`; normalization `{render.get('psf_normalization', 'n/a')}`.",
        f"- Core integrated signal: `{render.get('core_integrated_signal', 'n/a')}`; halo integrated signal: `{render.get('halo_integrated_signal', 'n/a')}`.",
        "- PSF remains an empirical effective model, not a physically calibrated STED PSF.",
    ]


def summary_table(summary: dict[str, Any]) -> list[str]:
    lines = ["| Quantity | Real fiber median | Blank median | Artificial synthetic median | Real-blank composite median |", "|:--|--:|--:|--:|--:|"]
    for key in ["p50", "p99", "std", "zero_fraction", "saturation_fraction", "local_variance_p50", "row_variation", "column_variation", "normalized_radial_power_tail_median"]:
        lines.append(
            f"| `{key}` | {fmt(summary, 'real_fiber', key)} | {fmt(summary, 'blank', key)} | {fmt(summary, 'artificial_synthetic', key)} | {fmt(summary, 'real_blank_composite', key)} |"
        )
    return lines


def draw_source_contact_sheet(rows: list[dict[str, str]], source_roots: dict[str, str], path: Path, label: str) -> list[str]:
    selected = deterministic_representatives(rows)
    tile = 192
    img = Image.new("RGB", (tile * len(selected), tile + 40), "white")
    draw = ImageDraw.Draw(img)
    ids: list[str] = []
    for i, row in enumerate(selected):
        src = Path(source_roots["sted_fiber_data" if row["source_kind"] == "fiber_image" else "sted_blank_data"]) / row["relative_path"]
        arr = np.asarray(Image.open(src), dtype=np.float32)
        preview = normalize_preview(arr, tile)
        img.paste(preview, (i * tile, 20))
        text = f"{label}\n{row['stable_image_id'][:10]}\np99={row.get('p99','?')}"
        draw.multiline_text((i * tile + 4, tile + 22), text, fill="black")
        ids.append(row["stable_image_id"])
    img.save(path)
    return ids


def deterministic_representatives(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: r["stable_image_id"])
    by_p50 = sorted(rows, key=lambda r: float(r.get("p50", 0)))
    by_p99 = sorted(rows, key=lambda r: float(r.get("p99", 0)))
    selected = [ordered[0], by_p50[len(by_p50) // 2], by_p99[-1]]
    unique: list[dict[str, str]] = []
    seen = set()
    for row in selected:
        if row["stable_image_id"] not in seen:
            unique.append(row)
            seen.add(row["stable_image_id"])
    return unique


def normalize_preview(arr: np.ndarray, size: int) -> Image.Image:
    lo, hi = np.percentile(arr, [1, 99])
    scaled = np.zeros_like(arr, dtype=np.uint8) if hi <= lo else np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(scaled).resize((size, size), Image.Resampling.BILINEAR).convert("RGB")


def draw_transverse_profile(composite_dir: Path, path: Path) -> bool:
    manifest_path = composite_dir / "dataset_manifest.csv"
    if not manifest_path.exists():
        return False
    with manifest_path.open(newline="", encoding="utf-8") as f:
        first = next(csv.DictReader(f))
    with np.load(composite_dir / first["npz_path"], allow_pickle=False) as data:
        render = data["render_uint8"].astype(np.float32)
        signal = data["synthetic_signal_float"].astype(np.float32)
    y = render.shape[0] // 2
    width, height = 720, 320
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 10), f"Transverse profile proxy: center row of {first['sample_id']}", fill="black")
    draw_profile(draw, render[y], (20, 280), 650, 220, (0, 80, 220), "composite uint8")
    draw_profile(draw, signal[y], (20, 280), 650, 220, (220, 80, 0), "synthetic signal")
    img.save(path)
    return True


def draw_profile(draw: ImageDraw.ImageDraw, values: np.ndarray, origin: tuple[int, int], width: int, height: int, color: tuple[int, int, int], label: str) -> None:
    vals = values.astype(np.float32)
    if vals.size == 0:
        return
    sample_idx = np.linspace(0, vals.size - 1, min(width, vals.size), dtype=int)
    sampled = vals[sample_idx]
    vmin = float(np.min(sampled))
    vmax = float(np.max(sampled))
    if vmax <= vmin:
        vmax = vmin + 1.0
    x0, y0 = origin
    points = []
    for i, value in enumerate(sampled):
        x = x0 + int(i * width / max(1, len(sampled) - 1))
        y = y0 - int((float(value) - vmin) / (vmax - vmin) * height)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=color, width=2)
    draw.text((x0, y0 + (15 if color[0] > 100 else 0)), label, fill=color)


def fmt(summary: dict[str, Any], group: str, key: str) -> str:
    try:
        return f"{summary['groups'][group][key]['median']:.4g}"
    except Exception:
        return "n/a"


def draw_histogram_comparison(real_rows: list[dict[str, str]], blank_rows: list[dict[str, str]], path: Path) -> None:
    width, height = 720, 360
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 10), "Median intensity by image: real fiber vs blank", fill="black")
    draw_series(draw, [float(r["p50"]) for r in real_rows], (20, 300), 300, (0, 90, 220), "real")
    draw_series(draw, [float(r["p50"]) for r in blank_rows], (380, 300), 300, (220, 90, 0), "blank")
    img.save(path)


def draw_bar_comparison(summary: dict[str, Any], path: Path) -> None:
    keys = ["normalized_radial_power_tail_median", "normalized_radial_power_high_band_fraction", "autocorrelation_tail_median"]
    groups = ["real_fiber", "blank", "artificial_synthetic", "real_blank_composite"]
    img = Image.new("RGB", (820, 420), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 10), "Normalized spectral-shape/autocorrelation proxy medians", fill="black")
    x = 40
    for key in keys:
        draw.text((x, 40), key, fill="black")
        vals = [summary["groups"].get(g, {}).get(key, {}).get("median", 0.0) for g in groups]
        scale = max(vals) or 1.0
        for i, value in enumerate(vals):
            h = int(250 * value / scale)
            draw.rectangle((x + i * 45, 330 - h, x + i * 45 + 28, 330), fill=[(0, 90, 220), (220, 90, 0), (60, 160, 60), (160, 60, 160)][i])
        x += 260
    img.save(path)


def draw_proxy_plot(proxy_rows: list[dict[str, str]], path: Path) -> None:
    img = Image.new("RGB", (720, 360), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 10), "Foreground occupancy proxy by source kind", fill="black")
    by_kind: dict[str, list[float]] = {}
    for row in proxy_rows:
        by_kind.setdefault(row["source_kind"], []).append(float(row["foreground_occupancy_proxy"]))
    x = 40
    for kind, values in sorted(by_kind.items()):
        draw_series(draw, values, (x, 300), 260, (40, 120, 180), kind)
        x += 330
    img.save(path)


def draw_series(draw: ImageDraw.ImageDraw, values: list[float], origin: tuple[int, int], width: int, color: tuple[int, int, int], label: str) -> None:
    if not values:
        return
    vals = sorted(values)
    maxv = max(vals) or 1.0
    x0, y0 = origin
    for i, value in enumerate(vals):
        x = x0 + int(i * max(1, width / max(1, len(vals))))
        h = int(220 * value / maxv)
        draw.line((x, y0, x, y0 - h), fill=color)
    draw.text((x0, y0 + 10), label, fill="black")


def mismatch_lines(summary: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    real = summary["groups"].get("real_fiber", {})
    comp = summary["groups"].get("real_blank_composite", {})
    synth = summary["groups"].get("artificial_synthetic", {})
    for key in ["p99", "std", "local_variance_p50", "normalized_radial_power_tail_median", "zero_fraction"]:
        if key in real and key in comp:
            diff = comp[key]["median"] - real[key]["median"]
            lines.append(f"- `{key}`: real-blank composite median minus real-fiber median = `{diff:.4g}`.")
        if key in real and key in synth:
            diff = synth[key]["median"] - real[key]["median"]
            lines.append(f"- `{key}`: artificial synthetic median minus real-fiber median = `{diff:.4g}`.")
    lines.append("- Label-dependent structure quantities remain uncertain until formal annotation exists.")
    lines.append("- Current calibration is exploratory because biological grouping and approved calibration subsets are unresolved.")
    return lines


def matched_delta_lines(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["- No matched blank-relative rows were available."]
    keys = ["delta_p50", "delta_p95", "delta_p99", "delta_mean", "delta_variance", "delta_zero_fraction", "delta_local_variance_p50", "foreground_added_integrated_signal"]
    lines = ["| Quantity | Median matched delta |", "|:--|--:|"]
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key)]
        if values:
            lines.append(f"| `{key}` | {float(np.median(values)):.4g} |")
    lines.append("")
    lines.append("Matched deltas compare each normalized composite directly with its own source blank; positive values indicate the composite exceeded the blank.")
    return lines


def build_review_package(
    out_dir: Path,
    summary: dict[str, Any],
    synthetic_dir: Path,
    composite_dir: Path,
    structural_qa_dir: Path | None,
    optical_qa_dir: Path | None,
    matched_rows: list[dict[str, str]],
) -> list[tuple[str, Path]]:
    datasets = [
        ("Realism samples", synthetic_dir, "realism_samples_contact_sheet.png"),
        (
            "Real-blank composites",
            composite_dir,
            "real_blank_composites_contact_sheet.png",
        ),
        (
            "Structural QA",
            structural_qa_dir,
            "structural_qa_contact_sheet.png",
        ),
        ("Optical QA", optical_qa_dir, "optical_qa_contact_sheet.png"),
    ]
    if not any(
        dataset_dir and (dataset_dir / "dataset_manifest.csv").exists()
        for _, dataset_dir, _ in datasets
    ):
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[str, Path]] = []
    selected_ids: dict[str, list[str]] = {}
    for label, dataset_dir, filename in datasets:
        if dataset_dir and (dataset_dir / "dataset_manifest.csv").exists():
            path = out_dir / filename
            selected_ids[label] = draw_generated_contact_sheet(
                dataset_dir, path, label
            )
            outputs.append((label, path))
    intensity = out_dir / "real_vs_synthetic_intensity_summary.png"
    draw_intensity_group_summary(summary, intensity)
    outputs.append(("Real-versus-synthetic intensity summary", intensity))
    spectra = out_dir / "normalized_spectral_summary.png"
    draw_bar_comparison(summary, spectra)
    outputs.append(("Normalized spectral summary", spectra))
    matched = out_dir / "matched_blank_delta_summary.png"
    draw_matched_delta_summary(matched_rows, matched)
    outputs.append(("Matched blank-delta summary", matched))
    metadata = summary.get("metadata", {})
    readme = [
        "# Curated STED calibration review package",
        "",
        f"- Source commit: `{metadata.get('source_commit_sha', metadata.get('code_version', 'not_available'))}`",
        f"- Working tree dirty: `{metadata.get('working_tree_dirty', 'not_recorded')}`",
        f"- Schema: `synthetic_sted_3d_rasterizer_0.5.0`",
        f"- Generation config SHA-256: `{metadata.get('generation_config_sha256', 'not_recorded')}`",
        "- Selection: deterministic low/median/high p99 cases per generated dataset.",
        "- Contents are lightweight PNG derivatives; no raw STED TIFF or NPZ data are included.",
        "",
        "## Selected samples",
        "",
    ]
    for label, ids in selected_ids.items():
        readme.append(f"- {label}: {', '.join(f'`{item}`' for item in ids)}")
    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(readme) + "\n", encoding="utf-8")
    outputs.append(("Review-package metadata", readme_path))
    return outputs


def draw_generated_contact_sheet(
    dataset_dir: Path, path: Path, title: str
) -> list[str]:
    rows = read_csv(dataset_dir / "dataset_manifest.csv")
    previews: list[tuple[dict[str, str], np.ndarray, float, str]] = []
    for row in rows:
        metadata = json.loads(
            (dataset_dir / row["json_path"]).read_text(encoding="utf-8")
        )
        with np.load(dataset_dir / row["npz_path"], allow_pickle=False) as data:
            image = data["render_uint8"].copy()
        previews.append(
            (
                row,
                image,
                float(np.percentile(image, 99)),
                str(metadata.get("dataset_schema_version", row.get("schema_version"))),
            )
        )
    if not previews:
        return []
    ordered = sorted(previews, key=lambda item: (item[2], item[0]["sample_id"]))
    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    unique: list[tuple[dict[str, str], np.ndarray, float, str]] = []
    seen: set[str] = set()
    for item in selected:
        if item[0]["sample_id"] not in seen:
            unique.append(item)
            seen.add(item[0]["sample_id"])
    tile = 256
    image = Image.new("RGB", (tile * len(unique), tile + 62), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 4), title, fill="black")
    for index, (row, array, p99, schema) in enumerate(unique):
        image.paste(normalize_preview(array.astype(np.float32), tile), (index * tile, 22))
        draw.text(
            (index * tile + 4, tile + 25),
            f"{row['sample_id']}\np99={p99:.3g} | {schema.rsplit('_', 1)[-1]}",
            fill="black",
        )
    image.save(path)
    return [item[0]["sample_id"] for item in unique]


def draw_intensity_group_summary(summary: dict[str, Any], path: Path) -> None:
    groups = [
        "real_fiber",
        "blank",
        "artificial_synthetic",
        "real_blank_composite",
    ]
    keys = ["p50", "p95", "p99"]
    image = Image.new("RGB", (760, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), "Per-group median intensity percentiles", fill="black")
    colors = [(0, 90, 220), (220, 90, 0), (60, 160, 60), (160, 60, 160)]
    for key_index, key in enumerate(keys):
        values = [
            float(summary.get("groups", {}).get(group, {}).get(key, {}).get("median", 0))
            for group in groups
        ]
        scale = max(values) or 1.0
        x0 = 40 + key_index * 235
        draw.text((x0, 45), key, fill="black")
        for group_index, value in enumerate(values):
            height = int(250 * value / scale)
            x = x0 + group_index * 45
            draw.rectangle((x, 330 - height, x + 28, 330), fill=colors[group_index])
    for index, group in enumerate(groups):
        draw.text((20 + index * 180, 375), group, fill=colors[index])
    image.save(path)


def draw_matched_delta_summary(
    rows: list[dict[str, str]], path: Path
) -> None:
    keys = [
        "delta_mean",
        "delta_variance",
        "delta_local_variance_p50",
        "delta_zero_fraction",
    ]
    medians = [
        float(np.median([float(row[key]) for row in rows if row.get(key)]))
        if any(row.get(key) for row in rows)
        else 0.0
        for key in keys
    ]
    image = Image.new("RGB", (760, 380), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), "Median composite minus matched source blank", fill="black")
    max_abs = max([abs(value) for value in medians] + [1.0])
    baseline = 210
    for index, (key, value) in enumerate(zip(keys, medians)):
        x = 55 + index * 175
        height = int(130 * abs(value) / max_abs)
        top, bottom = (
            (baseline - height, baseline)
            if value >= 0
            else (baseline, baseline + height)
        )
        draw.rectangle((x, top, x + 70, bottom), fill=(60, 130, 190))
        draw.text((x, 310), key.replace("delta_", ""), fill="black")
        draw.text((x, 285), f"{value:.4g}", fill="black")
    draw.line((30, baseline, 730, baseline), fill="black")
    image.save(path)
