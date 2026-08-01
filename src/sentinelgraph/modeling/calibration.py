"""Probability calibration primitives for the SentinelGraph risk engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sentinelgraph.modeling.models import RANDOM_SEED

PROBABILITY_EPSILON = 1e-6
DEFAULT_CALIBRATION_BINS = 15
RISK_COMPONENT_NAMES = (
    "behavioural_probability",
    "anomaly_score",
    "graph_probability",
)


def _validated_labels(labels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    values = np.asarray(labels, dtype=np.uint8)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("labels must be binary")
    if np.unique(values).size != 2:
        raise ValueError("calibration requires both legitimate and fraud rows")
    return values


def clip_probabilities(
    scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return finite probabilities inside the open unit interval."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("scores must be between zero and one")
    return np.clip(values, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def logit_scores(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform bounded scores into numerically stable log odds."""
    probabilities = clip_probabilities(scores)
    return np.log(probabilities / (1.0 - probabilities))


def expected_calibration_error(
    labels: NDArray[np.uint8],
    probabilities: NDArray[np.float64],
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Return weighted ECE, maximum calibration error, and bin statistics."""
    values = _validated_labels(labels)
    risks = clip_probabilities(probabilities)
    if values.size != risks.size:
        raise ValueError("labels and probabilities must have equal length")
    if bins < 2:
        raise ValueError("bins must be at least two")

    bin_indexes = np.minimum((risks * bins).astype(np.int32), bins - 1)
    rows: list[dict[str, Any]] = []
    weighted_error = 0.0
    maximum_error = 0.0
    for index in range(bins):
        mask = bin_indexes == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_risk = float(risks[mask].mean())
        fraud_rate = float(values[mask].mean())
        absolute_error = abs(mean_risk - fraud_rate)
        weighted_error += count * absolute_error / values.size
        maximum_error = max(maximum_error, absolute_error)
        rows.append(
            {
                "bin": index,
                "lower_bound": index / bins,
                "upper_bound": (index + 1) / bins,
                "rows": count,
                "mean_predicted_risk": mean_risk,
                "observed_fraud_rate": fraud_rate,
                "absolute_error": absolute_error,
            }
        )
    return float(weighted_error), float(maximum_error), rows


def calibration_metrics(
    labels: NDArray[np.uint8],
    probabilities: NDArray[np.float64],
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, Any]:
    """Calculate ranking and probability-quality metrics."""
    values = _validated_labels(labels)
    risks = clip_probabilities(probabilities)
    if values.size != risks.size:
        raise ValueError("labels and probabilities must have equal length")
    ece, maximum_error, calibration_bins = expected_calibration_error(
        values,
        risks,
        bins=bins,
    )
    return {
        "rows": int(values.size),
        "fraud_rows": int(values.sum()),
        "fraud_rate": float(values.mean()),
        "mean_predicted_risk": float(risks.mean()),
        "average_precision": float(average_precision_score(values, risks)),
        "roc_auc": float(roc_auc_score(values, risks)),
        "brier_score": float(brier_score_loss(values, risks)),
        "log_loss": float(log_loss(values, risks, labels=[0, 1])),
        "expected_calibration_error": ece,
        "maximum_calibration_error": maximum_error,
        "calibration_bins": calibration_bins,
    }


class SigmoidCalibrator:
    """Platt-style calibration over the champion model's log odds."""

    def __init__(self) -> None:
        self.estimator_: LogisticRegression | None = None

    def fit(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.uint8],
    ) -> "SigmoidCalibrator":
        """Fit an unweighted logistic mapping on a later temporal window."""
        values = _validated_labels(labels)
        transformed = logit_scores(scores)
        if transformed.size != values.size:
            raise ValueError("scores and labels must have equal length")
        estimator = LogisticRegression(
            C=1_000.0,
            max_iter=500,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        estimator.fit(transformed.reshape(-1, 1), values)
        self.estimator_ = estimator
        return self

    def predict(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return calibrated fraud probabilities."""
        if self.estimator_ is None:
            raise RuntimeError("SigmoidCalibrator must be fitted first")
        transformed = logit_scores(scores)
        probabilities = self.estimator_.predict_proba(
            transformed.reshape(-1, 1)
        )[:, 1]
        return clip_probabilities(np.asarray(probabilities, dtype=np.float64))


class IsotonicCalibrator:
    """Monotonic non-parametric probability calibrator."""

    def __init__(self) -> None:
        self.estimator_: IsotonicRegression | None = None

    def fit(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.uint8],
    ) -> "IsotonicCalibrator":
        """Fit a monotonic mapping on a later temporal window."""
        values = _validated_labels(labels)
        risks = clip_probabilities(scores)
        if risks.size != values.size:
            raise ValueError("scores and labels must have equal length")
        estimator = IsotonicRegression(
            out_of_bounds="clip",
            y_min=PROBABILITY_EPSILON,
            y_max=1.0 - PROBABILITY_EPSILON,
        )
        estimator.fit(risks, values)
        self.estimator_ = estimator
        return self

    def predict(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return calibrated fraud probabilities."""
        if self.estimator_ is None:
            raise RuntimeError("IsotonicCalibrator must be fitted first")
        probabilities = self.estimator_.predict(clip_probabilities(scores))
        return clip_probabilities(np.asarray(probabilities, dtype=np.float64))


def _stack_matrix(
    component_scores: Mapping[str, NDArray[np.float64]],
    component_names: Sequence[str],
) -> NDArray[np.float64]:
    missing = [name for name in component_names if name not in component_scores]
    if missing:
        raise ValueError(f"missing risk component: {missing[0]}")
    columns = [logit_scores(component_scores[name]) for name in component_names]
    row_counts = {column.size for column in columns}
    if len(row_counts) != 1:
        raise ValueError("risk components must have equal length")
    return np.column_stack(columns).astype(np.float64, copy=False)


class ScoreStackCalibrator:
    """Calibrate and combine supervised, anomaly, and graph evidence."""

    def __init__(
        self,
        component_names: Sequence[str] = RISK_COMPONENT_NAMES,
    ) -> None:
        self.component_names = tuple(component_names)
        self.estimator_: Pipeline | None = None

    def fit(
        self,
        component_scores: Mapping[str, NDArray[np.float64]],
        labels: NDArray[np.uint8],
    ) -> "ScoreStackCalibrator":
        """Fit an unweighted logistic stack on temporally later scores."""
        values = _validated_labels(labels)
        matrix = _stack_matrix(component_scores, self.component_names)
        if matrix.shape[0] != values.size:
            raise ValueError("risk components and labels must have equal length")
        estimator = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=500,
                        random_state=RANDOM_SEED,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        estimator.fit(matrix, values)
        self.estimator_ = estimator
        return self

    def predict(
        self,
        component_scores: Mapping[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Return calibrated probabilities from all evidence components."""
        if self.estimator_ is None:
            raise RuntimeError("ScoreStackCalibrator must be fitted first")
        matrix = _stack_matrix(component_scores, self.component_names)
        probabilities = self.estimator_.predict_proba(matrix)[:, 1]
        return clip_probabilities(np.asarray(probabilities, dtype=np.float64))
