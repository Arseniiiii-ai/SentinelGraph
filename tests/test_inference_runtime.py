"""Tests for loading and validating a serving risk bundle."""

from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any

import joblib
import numpy as np
import pytest
from numpy.typing import NDArray

from sentinelgraph.api.inference import RiskBundleError, RiskScorer
from sentinelgraph.modeling.behaviour import BEHAVIOURAL_FEATURE_NAMES
from sentinelgraph.modeling.decision import DecisionPolicy
from sentinelgraph.modeling.graph import (
    GRAPH_FEATURE_NAMES,
    GRAPH_ONLY_MODEL_FEATURE_NAMES,
)


class ConstantModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict_proba(self, features: NDArray[np.float32]) -> NDArray[np.float64]:
        scores = np.full(features.shape[0], self.score, dtype=np.float64)
        return np.column_stack((1.0 - scores, scores))


class StackCalibrator:
    def predict(
        self, components: dict[str, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        return np.asarray(components["behavioural_probability"], dtype=np.float64)


class AmountSensitiveModel:
    def predict_proba(self, features: NDArray[np.float32]) -> NDArray[np.float64]:
        scores = np.clip(0.1 + 0.001 * features[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - scores, scores)).astype(np.float64)


def valid_bundle() -> dict[str, Any]:
    return {
        "release": "v0.5",
        "feature_names": list(GRAPH_FEATURE_NAMES),
        "behavioural_feature_names": list(BEHAVIOURAL_FEATURE_NAMES),
        "graph_feature_names": list(GRAPH_ONLY_MODEL_FEATURE_NAMES),
        "models": {
            "behavioural_probability": ConstantModel(0.7),
            "anomaly_score": ConstantModel(0.2),
            "graph_probability": ConstantModel(0.4),
        },
        "calibrator_name": "logistic_score_stack",
        "calibrator": StackCalibrator(),
        "policy": DecisionPolicy(
            review_threshold=0.3,
            decline_threshold=0.8,
            maximum_review_rate=0.01,
            maximum_decline_rate=0.001,
            minimum_decline_precision=0.8,
        ),
        "explanation_baseline": np.zeros(len(GRAPH_FEATURE_NAMES), dtype=np.float32),
    }


def write_bundle(path: Path, bundle: dict[str, Any]) -> None:
    joblib.dump(bundle, path)


def test_risk_scorer_validates_and_scores_bundle(tmp_path: Path) -> None:
    path = tmp_path / "risk.joblib"
    write_bundle(path, valid_bundle())
    scorer = RiskScorer(path)
    result = scorer.score(
        np.zeros((1, len(GRAPH_FEATURE_NAMES)), dtype=np.float32)
    )
    assert scorer.ready
    assert scorer.model_version == "v0.5"
    assert result.risk_probability == pytest.approx(0.7)
    assert result.risk_points == 700
    assert result.decision == "review"
    assert result.component_scores == {
        "behavioural_probability": pytest.approx(0.7),
        "anomaly_score": pytest.approx(0.2),
        "graph_probability": pytest.approx(0.4),
    }


def test_risk_scorer_rejects_feature_contract_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "invalid.joblib"
    bundle = valid_bundle()
    bundle["feature_names"] = list(GRAPH_FEATURE_NAMES[:-1])
    write_bundle(path, bundle)
    with pytest.raises(RiskBundleError, match="feature_names contract mismatch"):
        RiskScorer(path)


def test_risk_scorer_requires_explanation_baseline(tmp_path: Path) -> None:
    path = tmp_path / "invalid.joblib"
    bundle = valid_bundle()
    del bundle["explanation_baseline"]
    write_bundle(path, bundle)
    with pytest.raises(RiskBundleError, match="missing key"):
        RiskScorer(path)


def test_risk_scorer_checks_digest_before_deserialisation(tmp_path: Path) -> None:
    path = tmp_path / "risk.joblib"
    write_bundle(path, valid_bundle())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert RiskScorer(path, expected_sha256=digest).ready
    with pytest.raises(RiskBundleError, match="SHA-256 mismatch"):
        RiskScorer(path, expected_sha256="0" * 64)


def test_vectorized_occlusion_returns_exact_amount_reason(tmp_path: Path) -> None:
    path = tmp_path / "risk.joblib"
    bundle = valid_bundle()
    bundle["models"]["behavioural_probability"] = AmountSensitiveModel()
    write_bundle(path, bundle)
    features = np.zeros((1, len(GRAPH_FEATURE_NAMES)), dtype=np.float32)
    features[0, GRAPH_FEATURE_NAMES.index("amount")] = 100.0
    result = RiskScorer(path).score(features)
    assert result.risk_probability == pytest.approx(0.2)
    assert result.reason_codes[0]["code"] == "HIGH_TRANSACTION_AMOUNT"
    assert result.reason_codes[0]["feature"] == "amount"
    assert result.reason_codes[0]["contribution"] == pytest.approx(0.1)
