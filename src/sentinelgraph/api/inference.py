"""Validated, process-local inference over the released v0.5 risk bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import numpy as np
from numpy.typing import NDArray

from sentinelgraph.modeling.decision import (
    DecisionPolicy,
    FEATURE_REASON_CODES,
    REASON_CODE_DESCRIPTIONS,
    apply_decision_policy,
    probability_to_risk_points,
)
from sentinelgraph.modeling.graph import (
    GRAPH_FEATURE_NAMES,
    GRAPH_ONLY_MODEL_FEATURE_NAMES,
)
from sentinelgraph.modeling.behaviour import BEHAVIOURAL_FEATURE_NAMES
from sentinelgraph.modeling.metrics import positive_scores


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """One calibrated prediction with evidence and policy output."""

    risk_probability: float
    risk_points: int
    decision: str
    reason_codes: list[dict[str, Any]]
    component_scores: dict[str, float]
    model_version: str
    policy_version: str


class ScoringEngine(Protocol):
    """Minimal interface used by the application service and test doubles."""

    @property
    def ready(self) -> bool: ...

    @property
    def model_version(self) -> str: ...

    def score(self, features: NDArray[np.float32]) -> InferenceResult: ...


class RiskBundleError(RuntimeError):
    """Raised when an artifact cannot satisfy the serving contract."""


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RiskBundleError(f"risk bundle {name} must be a mapping")
    return cast(Mapping[str, Any], value)


class RiskScorer:
    """Load once, validate aggressively, and score without mutable state."""

    def __init__(
        self,
        bundle_path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        if not bundle_path.is_file():
            raise RiskBundleError(f"risk bundle is missing: {bundle_path}")
        if expected_sha256 is not None:
            digest = hashlib.sha256()
            with bundle_path.open("rb") as artifact:
                for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise RiskBundleError("risk bundle SHA-256 mismatch")
        raw = joblib.load(bundle_path)
        bundle = _required_mapping(raw, "root")
        self._validate_bundle(bundle)
        self._models = _required_mapping(bundle["models"], "models")
        self._calibrator = bundle["calibrator"]
        self._calibrator_name = str(bundle["calibrator_name"])
        self._policy = cast(DecisionPolicy, bundle["policy"])
        self._baseline = np.asarray(
            bundle["explanation_baseline"], dtype=np.float32
        )
        self._model_version = str(bundle["release"])
        self._policy_version = f"{self._model_version}-policy"
        self._behavioural_indexes = tuple(
            GRAPH_FEATURE_NAMES.index(name) for name in BEHAVIOURAL_FEATURE_NAMES
        )
        self._graph_indexes = tuple(
            GRAPH_FEATURE_NAMES.index(name)
            for name in GRAPH_ONLY_MODEL_FEATURE_NAMES
        )

    @staticmethod
    def _validate_bundle(bundle: Mapping[str, Any]) -> None:
        required = {
            "release",
            "feature_names",
            "behavioural_feature_names",
            "graph_feature_names",
            "models",
            "calibrator_name",
            "calibrator",
            "policy",
            "explanation_baseline",
        }
        missing = sorted(required - set(bundle))
        if missing:
            raise RiskBundleError(f"risk bundle missing key: {missing[0]}")
        contracts: tuple[tuple[str, Sequence[str]], ...] = (
            ("feature_names", GRAPH_FEATURE_NAMES),
            ("behavioural_feature_names", BEHAVIOURAL_FEATURE_NAMES),
            ("graph_feature_names", GRAPH_ONLY_MODEL_FEATURE_NAMES),
        )
        for key, expected in contracts:
            if list(bundle[key]) != list(expected):
                raise RiskBundleError(f"risk bundle {key} contract mismatch")
        models = _required_mapping(bundle["models"], "models")
        expected_models = {
            "behavioural_probability",
            "anomaly_score",
            "graph_probability",
        }
        if set(models) != expected_models:
            raise RiskBundleError("risk bundle component model contract mismatch")
        baseline = np.asarray(bundle["explanation_baseline"])
        if baseline.shape != (len(GRAPH_FEATURE_NAMES),):
            raise RiskBundleError("risk bundle explanation baseline shape mismatch")
        if not np.all(np.isfinite(baseline)):
            raise RiskBundleError("risk bundle explanation baseline is non-finite")
        if not isinstance(bundle["policy"], DecisionPolicy):
            raise RiskBundleError("risk bundle policy type mismatch")
        if str(bundle["calibrator_name"]) not in {
            "sigmoid_behavioural",
            "isotonic_behavioural",
            "logistic_score_stack",
        }:
            raise RiskBundleError("risk bundle calibrator is unsupported")

    @property
    def ready(self) -> bool:
        return True

    @property
    def model_version(self) -> str:
        return self._model_version

    def _predict_probabilities(
        self,
        features: NDArray[np.float32],
    ) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(GRAPH_FEATURE_NAMES):
            raise ValueError(
                f"expected a two-dimensional {len(GRAPH_FEATURE_NAMES)}-feature matrix"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("inference matrix contains non-finite values")
        behavioural = matrix[:, self._behavioural_indexes]
        graph = matrix[:, self._graph_indexes]
        components = {
            "behavioural_probability": positive_scores(
                self._models["behavioural_probability"], behavioural
            ),
            "anomaly_score": positive_scores(
                self._models["anomaly_score"], behavioural
            ),
            "graph_probability": positive_scores(
                self._models["graph_probability"], graph
            ),
        }
        if self._calibrator_name == "logistic_score_stack":
            raw_probabilities = self._calibrator.predict(components)
        else:
            raw_probabilities = self._calibrator.predict(
                components["behavioural_probability"]
            )
        probabilities = np.asarray(raw_probabilities, dtype=np.float64)
        return probabilities, components

    def score(self, features: NDArray[np.float32]) -> InferenceResult:
        """Score one request and calculate bounded, end-to-end reason codes."""
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.shape[0] != 1:
            raise ValueError("score accepts exactly one feature row")
        probabilities, components = self._predict_probabilities(matrix)
        decisions = apply_decision_policy(probabilities, self._policy)
        points = probability_to_risk_points(probabilities)
        occluded = np.repeat(matrix, len(GRAPH_FEATURE_NAMES), axis=0)
        indexes = np.arange(len(GRAPH_FEATURE_NAMES))
        occluded[indexes, indexes] = self._baseline
        occluded_probabilities, _ = self._predict_probabilities(occluded)
        contributions = np.maximum(
            probabilities[0] - occluded_probabilities,
            0.0,
        )
        reasons: list[dict[str, Any]] = []
        for index in np.argsort(-contributions, kind="stable"):
            contribution = float(contributions[int(index)])
            if contribution <= 0.0:
                continue
            feature_name = GRAPH_FEATURE_NAMES[int(index)]
            code = FEATURE_REASON_CODES.get(
                feature_name,
                "MODEL_FEATURE_ELEVATED",
            )
            reasons.append(
                {
                    "code": code,
                    "description": REASON_CODE_DESCRIPTIONS[code],
                    "feature": feature_name,
                    "contribution": contribution,
                }
            )
            if len(reasons) == 3:
                break
        return InferenceResult(
            risk_probability=float(probabilities[0]),
            risk_points=int(points[0]),
            decision=str(decisions[0]),
            reason_codes=reasons,
            component_scores={
                name: float(values[0]) for name, values in components.items()
            },
            model_version=self._model_version,
            policy_version=self._policy_version,
        )
