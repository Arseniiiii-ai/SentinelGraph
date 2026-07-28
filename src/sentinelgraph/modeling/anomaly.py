"""Isolation-Forest scoring and anomaly-augmented classification for v0.3."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from sentinelgraph.modeling.models import RANDOM_SEED


def build_behavioural_gradient_boosting() -> HistGradientBoostingClassifier:
    """Return the supervised v0.3 challenger."""
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        early_stopping=True,
        l2_regularization=2.0,
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        n_iter_no_change=15,
        random_state=RANDOM_SEED,
        validation_fraction=0.1,
    )


class IsolationForestDetector:
    """Expose unsupervised anomaly scores through a classifier-like interface."""

    def __init__(
        self,
        *,
        n_estimators: int = 150,
        max_samples: int = 50_000,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimator_: IsolationForest | None = None
        self.legitimate_fit_rows_: int | None = None

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.uint8],
    ) -> "IsolationForestDetector":
        """Fit only on legitimate development behaviour."""
        legitimate = features[labels == 0]
        if legitimate.shape[0] == 0:
            raise ValueError("Isolation Forest requires legitimate training rows")
        self.estimator_ = IsolationForest(
            contamination="auto",
            max_samples=min(self.max_samples, legitimate.shape[0]),
            n_estimators=self.n_estimators,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.estimator_.fit(legitimate)
        self.legitimate_fit_rows_ = int(legitimate.shape[0])
        return self

    def anomaly_score(
        self,
        features: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Return a bounded score where larger values mean more anomalous."""
        if self.estimator_ is None:
            raise RuntimeError("IsolationForestDetector must be fitted first")
        raw = -np.asarray(
            self.estimator_.decision_function(features),
            dtype=np.float64,
        )
        clipped = np.clip(5.0 * raw, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def predict_proba(
        self,
        features: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Return anomaly scores in the positive-class probability column."""
        scores = self.anomaly_score(features)
        return np.column_stack((1.0 - scores, scores))


class AnomalyAugmentedClassifier:
    """Fit a supervised model with a legitimate-only anomaly score feature."""

    def __init__(self) -> None:
        self.anomaly_detector = IsolationForestDetector()
        self.classifier = build_behavioural_gradient_boosting()
        self.n_features_in_: int | None = None

    @staticmethod
    def _augment(
        features: NDArray[np.float32],
        anomaly_scores: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        return np.column_stack((features, anomaly_scores)).astype(
            np.float32,
            copy=False,
        )

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.uint8],
    ) -> "AnomalyAugmentedClassifier":
        """Fit the anomaly detector first, then the supervised classifier."""
        self.anomaly_detector.fit(features, labels)
        anomaly_scores = self.anomaly_detector.anomaly_score(features)
        self.classifier.fit(self._augment(features, anomaly_scores), labels)
        self.n_features_in_ = int(features.shape[1])
        return self

    def predict_proba(
        self,
        features: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Return positive-class probabilities from the augmented classifier."""
        if self.n_features_in_ is None:
            raise RuntimeError("AnomalyAugmentedClassifier must be fitted first")
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, received "
                f"{features.shape[1]}"
            )
        anomaly_scores = self.anomaly_detector.anomaly_score(features)
        probabilities: Any = self.classifier.predict_proba(
            self._augment(features, anomaly_scores)
        )
        return np.asarray(probabilities, dtype=np.float64)
