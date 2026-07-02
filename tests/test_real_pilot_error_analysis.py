import numpy as np

from scripts.analyze_real_pilot_errors import (
    discover_eval_outputs,
    leakage_diagnostics,
    probability_summaries_by_target_region,
    semantic_error_analysis,
    semantic_leakage_analysis,
    skeleton_error_analysis,
    skeleton_leakage_analysis,
    within_mask_distance_fraction,
)


def test_semantic_error_analysis_ignores_uncertain_pixels():
    target = np.zeros((5, 5), dtype=np.uint8)
    target[0, 0] = 255
    target[2, 2] = 1
    pred = np.zeros_like(target)
    pred[0, 0] = 1
    pred[2, 2] = 1
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)

    metrics = semantic_error_analysis(target, pred, image)

    assert metrics["false_positive_fibrous_pixels_in_background"] == 0
    assert metrics["false_negative_fibrous_pixels"] == 0
    assert metrics["predicted_to_target_fibrous_area_ratio"] == 1.0


def test_discover_eval_outputs_supports_canonical_and_legacy_layouts(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "metrics.json").write_text("{}", encoding="utf-8")
    (sample / "predictions.npz").write_bytes(b"npz")
    assert discover_eval_outputs(sample) == [
        (sample / "metrics.json", sample / "predictions.npz", sample)
    ]

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "image_metrics.json").write_text("{}", encoding="utf-8")
    (legacy / "image_predictions.npz").write_bytes(b"npz")
    assert discover_eval_outputs(legacy) == [
        (legacy / "image_metrics.json", legacy / "image_predictions.npz", legacy)
    ]


def test_distance_tolerant_skeleton_recovery_counts_nearby_pixels():
    target = np.zeros((9, 9), dtype=bool)
    pred = np.zeros_like(target)
    target[4, 4] = True
    pred[4, 6] = True

    assert within_mask_distance_fraction(target, pred, 2.0) == 1.0
    assert within_mask_distance_fraction(pred, target, 2.0) == 1.0


def test_skeleton_threshold_sweep_records_required_thresholds():
    target_semantic = np.ones((5, 5), dtype=np.uint8)
    target_skeleton = np.zeros((5, 5), dtype=np.uint8)
    target_skeleton[2, 2] = 1
    probability = np.zeros((5, 5), dtype=np.float32)
    probability[2, 2] = 0.8

    metrics = skeleton_error_analysis(target_semantic, target_skeleton, probability)

    assert set(metrics["thresholds"]) == {"0.5", "0.75", "0.85"}
    assert metrics["thresholds"]["0.75"]["recall"] == 1.0
    assert metrics["thresholds"]["0.85"]["recall"] == 0.0


def test_error_analysis_does_not_emit_topology_metrics():
    target_semantic = np.ones((5, 5), dtype=np.uint8)
    target_skeleton = np.zeros((5, 5), dtype=np.uint8)
    probability = np.zeros((5, 5), dtype=np.float32)

    metrics = {
        "semantic": semantic_error_analysis(target_semantic, target_semantic, np.zeros((5, 5), dtype=np.uint8)),
        "skeleton": skeleton_error_analysis(target_semantic, target_skeleton, probability),
    }
    keys = list(flat_keys(metrics))

    forbidden = ("endpoint", "crossing", "junction", "branch", "merge")
    assert not any(any(word in key for word in forbidden) for key in keys)


def test_semantic_leakage_accounting_sums_regions_separately():
    target = np.asarray([[1, 3], [255, 0]], dtype=np.uint8)
    pred = np.asarray([[1, 1], [1, 3]], dtype=np.uint8)

    leakage = semantic_leakage_analysis(target, pred)

    fib = leakage["predicted_fibrous"]
    assert fib["total_pixels"] == 3
    assert fib["inside_target_fibrous_pixels"] == 1
    assert fib["inside_target_clump_pixels"] == 1
    assert fib["inside_target_uncertain_ignore_pixels"] == 1
    assert fib["inside_target_background_pixels"] == 0
    assert fib["inside_target_clump_fraction"] == 1 / 3
    clump = leakage["predicted_clump"]
    assert clump["inside_target_background_pixels"] == 1


def test_skeleton_leakage_accounting_at_multiple_thresholds_tracks_ignore_and_clump():
    target = np.asarray([[1, 3], [255, 0]], dtype=np.uint8)
    pred_semantic = np.asarray([[1, 3], [0, 0]], dtype=np.uint8)
    probability = np.asarray([[0.9, 0.8], [0.6, 0.4]], dtype=np.float32)
    pred = {
        "semantic_class_map": pred_semantic,
        "fibrous_probability": np.zeros((2, 2), dtype=np.float32),
        "clump_probability": np.zeros((2, 2), dtype=np.float32),
        "skeleton_probability": probability,
    }

    leakage = leakage_diagnostics(target, pred)["skeleton"]

    assert set(leakage) == {"0.5", "0.75", "0.85"}
    assert leakage["0.5"]["inside_target_clump_pixels"] == 1
    assert leakage["0.5"]["inside_target_uncertain_ignore_pixels"] == 1
    assert leakage["0.75"]["inside_target_uncertain_ignore_pixels"] == 0
    assert leakage["0.85"]["inside_target_clump_pixels"] == 0
    assert leakage["0.5"]["inside_predicted_clump_pixels"] == 1
    assert leakage["0.5"]["outside_predicted_foreground_pixels"] == 1


def test_probability_summaries_are_reported_per_target_region():
    target = np.asarray([[1, 3], [255, 0]], dtype=np.uint8)
    probs = {
        "fibrous_probability": np.asarray([[0.9, 0.2], [0.4, 0.1]], dtype=np.float32),
        "clump_probability": np.asarray([[0.1, 0.8], [0.3, 0.2]], dtype=np.float32),
        "skeleton_probability": np.asarray([[0.7, 0.6], [0.5, 0.4]], dtype=np.float32),
    }

    summaries = probability_summaries_by_target_region(target, probs)

    assert np.isclose(summaries["target_fibrous"]["fibrous_probability"]["mean"], 0.9)
    assert np.isclose(summaries["target_clump"]["clump_probability"]["mean"], 0.8)
    assert np.isclose(summaries["target_uncertain_ignore"]["skeleton_probability"]["mean"], 0.5)
    assert np.isclose(summaries["target_background"]["fibrous_probability"]["p95"], 0.1)


def test_direct_skeleton_leakage_tracks_clump_separately_from_background():
    target = np.asarray([[3, 0], [3, 0]], dtype=np.uint8)
    pred_semantic = np.zeros((2, 2), dtype=np.uint8)
    pred_skeleton = np.ones((2, 2), dtype=bool)

    leakage = skeleton_leakage_analysis(target, pred_semantic, pred_skeleton)

    assert leakage["inside_target_clump_pixels"] == 2
    assert leakage["inside_target_background_pixels"] == 2
    assert leakage["inside_target_uncertain_ignore_pixels"] == 0


def flat_keys(value, prefix=""):
    if isinstance(value, dict):
        for key, nested in value.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            yield from flat_keys(nested, full)
    else:
        yield prefix
