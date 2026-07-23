"""Tests for the v0.2 baseline estimators."""

import numpy as np
import pytest

from sentinelgraph.modeling.features import FEATURE_NAMES
from sentinelgraph.modeling.metrics import positive_scores
from sentinelgraph.modeling.models import (
    build_dummy_baseline,
    build_gradient_boosting_baseline,
    build_logistic_baseline,
)
from sentinelgraph.modeling.rules import RuleBaseline


def _small_dataset() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(42)
    features = generator.normal(size=(200, len(FEATURE_NAMES))).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.uint8)
    return features, labels


@pytest.mark.parametrize(
    "builder",
    [
        build_dummy_baseline,
        build_logistic_baseline,
        build_gradient_boosting_baseline,
    ],
)
def test_learned_baseline_returns_probabilities(
    builder: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    features, labels = _small_dataset()
    model = builder()
    model.fit(features, labels)

    scores = positive_scores(model, features)

    assert scores.shape == (features.shape[0],)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_large_transfer_rule_uses_only_transfer_and_amount() -> None:
    features = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    amount_index = FEATURE_NAMES.index("amount")
    transfer_index = FEATURE_NAMES.index("type_transfer")
    cash_out_index = FEATURE_NAMES.index("type_cash_out")
    features[:, amount_index] = [250001.0, 250001.0, 100.0]
    features[0, transfer_index] = 1.0
    features[1, cash_out_index] = 1.0
    features[2, transfer_index] = 1.0

    scores = RuleBaseline().predict_proba(features)[:, 1]

    assert scores.tolist() == [1.0, 0.0, 0.0]
