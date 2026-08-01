"""Tests for v0.5 probability calibration primitives."""

from __future__ import annotations

import numpy as np

from sentinelgraph.modeling.calibration import (
    IsotonicCalibrator,
    ScoreStackCalibrator,
    SigmoidCalibrator,
    calibration_metrics,
    expected_calibration_error,
)


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(
        [0.02, 0.08, 0.15, 0.30, 0.55, 0.70, 0.88, 0.97],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8)
    return scores, labels


def test_sigmoid_and_isotonic_return_bounded_probabilities() -> None:
    scores, labels = _fixture()
    for calibrator in (SigmoidCalibrator(), IsotonicCalibrator()):
        calibrated = calibrator.fit(scores, labels).predict(scores)
        assert calibrated.shape == scores.shape
        assert np.all(calibrated > 0.0)
        assert np.all(calibrated < 1.0)


def test_score_stack_uses_all_named_components() -> None:
    scores, labels = _fixture()
    components = {
        "behavioural_probability": scores,
        "anomaly_score": np.sqrt(scores),
        "graph_probability": np.clip(scores * 0.8 + 0.1, 0.0, 1.0),
    }
    calibrator = ScoreStackCalibrator().fit(components, labels)
    calibrated = calibrator.predict(components)

    assert calibrated.shape == scores.shape
    assert np.all(np.isfinite(calibrated))


def test_calibration_metrics_report_reliability_bins() -> None:
    scores, labels = _fixture()
    metrics = calibration_metrics(labels, scores, bins=4)
    ece, maximum_error, bins = expected_calibration_error(
        labels,
        scores,
        bins=4,
    )

    assert metrics["rows"] == 8
    assert metrics["fraud_rows"] == 4
    assert metrics["expected_calibration_error"] == ece
    assert metrics["maximum_calibration_error"] == maximum_error
    assert metrics["calibration_bins"] == bins
    assert 0.0 <= ece <= 1.0


def test_calibrator_rejects_single_class_labels() -> None:
    scores = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    labels = np.zeros(3, dtype=np.uint8)

    try:
        SigmoidCalibrator().fit(scores, labels)
    except ValueError as error:
        assert "both legitimate and fraud" in str(error)
    else:
        raise AssertionError("single-class calibration should fail")
