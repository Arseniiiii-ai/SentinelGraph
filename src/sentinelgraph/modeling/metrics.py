"""Fraud-oriented threshold selection and evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def positive_scores(model: Any, features: NDArray[np.float32]) -> NDArray[np.float64]:
    """Return positive-class probabilities from a fitted estimator."""
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("predict_proba must return two class columns")
    return probabilities[:, 1]


def threshold_at_fpr(
    labels: NDArray[np.uint8],
    scores: NDArray[np.float64],
    *,
    maximum_fpr: float,
) -> float:
    """Select the highest-recall validation threshold within an FPR budget."""
    if not 0.0 < maximum_fpr < 1.0:
        raise ValueError("maximum_fpr must be between zero and one")
    false_positive_rates, true_positive_rates, thresholds = roc_curve(
        labels,
        scores,
        drop_intermediate=False,
    )
    valid = np.flatnonzero(false_positive_rates <= maximum_fpr)
    if valid.size == 0:
        return float(np.nextafter(scores.max(), np.inf))
    valid_recalls = true_positive_rates[valid]
    best_recall = valid_recalls.max()
    best = valid[valid_recalls == best_recall]
    selected = float(thresholds[best[0]])
    if not np.isfinite(selected):
        return float(np.nextafter(scores.max(), np.inf))
    return selected


def evaluate_scores(
    labels: NDArray[np.uint8],
    scores: NDArray[np.float64],
    amounts: NDArray[np.float64],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Calculate ranking, calibration, classification, and value metrics."""
    predictions = (scores >= threshold).astype(np.uint8)
    confusion = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in confusion.ravel()
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    legitimate_rows = true_negative + false_positive
    fraud_rows = true_positive + false_negative
    total_fraud_amount = float(amounts[labels == 1].sum())
    captured_fraud_amount = float(amounts[(labels == 1) & (predictions == 1)].sum())

    return {
        "rows": int(labels.size),
        "fraud_rows": fraud_rows,
        "fraud_rate": float(labels.mean()),
        "threshold": threshold,
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": (
            false_positive / legitimate_rows if legitimate_rows else 0.0
        ),
        "false_positives_per_10k_legitimate": (
            false_positive * 10_000 / legitimate_rows if legitimate_rows else 0.0
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "total_fraud_amount": total_fraud_amount,
        "captured_fraud_amount": captured_fraud_amount,
        "captured_fraud_amount_rate": (
            captured_fraud_amount / total_fraud_amount if total_fraud_amount else 0.0
        ),
    }
