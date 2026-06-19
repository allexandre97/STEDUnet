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


def build_report(artifact_dir: Path, synthetic_dir: Path, composite_dir: Path, out_path: Path, config: dict[str, Any] | None = None, pure_blank_dir: Path | None = None) -> None:
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
        f"- Code version: `{summary['metadata']['code_version']}`",
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
    lines.append(f"- Sorted per-image median intensity summary: `{histogram_path.name}`")
    power_path = assets / "normalized_spectral_shape_comparison.png"
    draw_bar_comparison(summary, power_path)
    lines.append(f"- DC-removed normalized spectral-shape and autocorrelation comparison: `{power_path.name}`")
    proxy_path = assets / "proxy_comparison.png"
    draw_proxy_plot(proxy_rows, proxy_path)
    lines.append(f"- Proxy distribution comparison: `{proxy_path.name}`")
    if config and "source_roots" in config:
        real_contact = assets / "representative_real_fiber_images.png"
        blank_contact = assets / "representative_blank_images.png"
        real_ids = draw_source_contact_sheet(real_rows, config["source_roots"], real_contact, "real fiber")
        blank_ids = draw_source_contact_sheet(blank_rows, config["source_roots"], blank_contact, "expert blank")
        lines.append(f"- Deterministic real-fiber representatives: `{real_contact.name}`; source IDs: {', '.join(real_ids)}")
        lines.append(f"- Deterministic blank representatives: `{blank_contact.name}`; source IDs: {', '.join(blank_ids)}")
    profile_path = assets / "transverse_profile_proxy.png"
    if draw_transverse_profile(composite_dir, profile_path):
        lines.append(f"- Composite transverse profile proxy: `{profile_path.name}`")
    else:
        lines.append("- Composite transverse profile proxy: not available; composite dataset manifest missing.")
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
