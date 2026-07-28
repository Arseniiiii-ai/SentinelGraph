"""Tests for v0.3 anomaly and augmented models."""

import numpy as np

from sentinelgraph.modeling.anomaly import (
    AnomalyAugmentedClassifier,
    IsolationForestDetector,
    build_behavioural_gradient_boosting,
)
from sentinelgraph.modeling.behaviour import BEHAVIOURAL_FEATURE_NAMES
from sentinelgraph.modeling.metrics import positive_scores


def _dataset() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(42)
    legitimate = generator.normal(
        0.0,
        1.0,
        size=(240, len(BEHAVIOURAL_FEATURE_NAMES)),
    )
    fraud = generator.normal(
        4.0,
        1.0,
        size=(40, len(BEHAVIOURAL_FEATURE_NAMES)),
    )
    features = np.vstack((legitimate, fraud)).astype(np.float32)
    labels = np.concatenate(
        (
            np.zeros(legitimate.shape[0], dtype=np.uint8),
            np.ones(fraud.shape[0], dtype=np.uint8),
        )
    )
    return features, labels


def test_isolation_forest_fits_only_legitimate_rows() -> None:
    features, labels = _dataset()
    model = IsolationForestDetector(n_estimators=20, max_samples=100)

    model.fit(features, labels)
    scores = positive_scores(model, features)

    assert model.legitimate_fit_rows_ == 240
    assert scores.shape == (280,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert scores[labels == 1].mean() > scores[labels == 0].mean()


def test_behavioural_gradient_returns_probabilities() -> None:
    features, labels = _dataset()
    model = build_behavioural_gradient_boosting()

    model.fit(features, labels)
    scores = positive_scores(model, features)

    assert scores.shape == (280,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_anomaly_augmented_classifier_round_trip() -> None:
    features, labels = _dataset()
    model = AnomalyAugmentedClassifier()
    model.anomaly_detector.n_estimators = 20
    model.anomaly_detector.max_samples = 100

    model.fit(features, labels)
    scores = positive_scores(model, features)

    assert model.n_features_in_ == len(BEHAVIOURAL_FEATURE_NAMES)
    assert scores.shape == (280,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
