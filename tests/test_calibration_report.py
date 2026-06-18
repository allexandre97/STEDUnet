import csv
import json

from fibras.calibration.reporting import build_report


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_report_generation(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    base = {
        "stable_image_id": "id",
        "source_kind": "kind",
        "p50": "4",
        "p99": "20",
        "std": "3",
        "zero_fraction": "0.1",
        "saturation_fraction": "0",
        "local_variance_p50": "2",
        "row_variation": "0.2",
        "column_variation": "0.2",
        "radial_power_tail_median": "1",
        "autocorrelation_tail_median": "0.1",
        "directional_power_ratio": "1",
        "foreground_occupancy_proxy": "0.1",
    }
    write_csv(artifact / "real_fiber_stats.csv", [dict(base, source_kind="fiber_image")])
    write_csv(artifact / "blank_stats.csv", [dict(base, source_kind="blank_background")])
    write_csv(artifact / "proxy_stats.csv", [dict(base, source_kind="fiber_image", proxy_warning="proxy")])
    summary = {
        "metadata": {"code_version": "test", "date_generated": "2026-06-18"},
        "groups": {
            "real_fiber": {"p50": {"median": 4}, "p99": {"median": 20}, "std": {"median": 3}, "zero_fraction": {"median": 0.1}, "saturation_fraction": {"median": 0}, "local_variance_p50": {"median": 2}, "row_variation": {"median": 0.2}, "column_variation": {"median": 0.2}},
            "blank": {"p50": {"median": 4}, "p99": {"median": 20}, "std": {"median": 3}, "zero_fraction": {"median": 0.1}, "saturation_fraction": {"median": 0}, "local_variance_p50": {"median": 2}, "row_variation": {"median": 0.2}, "column_variation": {"median": 0.2}},
            "artificial_synthetic": {},
            "real_blank_composite": {},
        },
    }
    (artifact / "appearance_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "report.md"
    build_report(artifact, tmp_path / "synthetic", tmp_path / "composite", out)
    assert "Exploratory STED appearance calibration report" in out.read_text()

