"""Tests for fraud-oriented model evaluation."""

import numpy as np

from sentinelgraph.modeling.metrics import (
    evaluate_scores,
    threshold_at_fpr,
)


def test_threshold_respects_false_positive_budget() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9], dtype=np.float64)

    threshold = threshold_at_fpr(labels, scores, maximum_fpr=0.25)
    metrics = evaluate_scores(
        labels,
        scores,
        np.ones(6, dtype=np.float64),
        threshold=threshold,
    )

    assert metrics["false_positive_rate"] <= 0.25
    assert metrics["recall"] == 1.0
    assert metrics["true_positive"] == 2


def test_constant_scores_choose_no_positive_threshold() -> None:
    labels = np.array([0, 0, 1], dtype=np.uint8)
    scores = np.full(3, 0.1, dtype=np.float64)

    threshold = threshold_at_fpr(labels, scores, maximum_fpr=0.01)

    assert np.isfinite(threshold)
    assert threshold > scores.max()
