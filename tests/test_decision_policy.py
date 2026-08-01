"""Tests for v0.5 decision thresholds, cost simulation, and reasons."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from sentinelgraph.modeling.decision import (
    APPROVE,
    DECLINE,
    REVIEW,
    CostAssumptions,
    DecisionPolicy,
    apply_decision_policy,
    local_occlusion_explanations,
    policy_backtest,
    probability_to_risk_points,
    select_decision_policy,
)


def test_policy_selection_respects_capacity_and_decline_precision() -> None:
    risks = np.linspace(0.001, 0.999, 1_000, dtype=np.float64)
    labels = np.zeros(1_000, dtype=np.uint8)
    labels[-20:] = 1
    amounts = np.full(1_000, 100.0, dtype=np.float64)

    policy = select_decision_policy(
        labels,
        risks,
        amounts,
        maximum_review_rate=0.05,
        maximum_decline_rate=0.01,
        minimum_decline_precision=0.80,
    )
    backtest = policy_backtest(labels, risks, amounts, policy)

    assert policy.review_threshold <= policy.decline_threshold
    assert backtest["review_queue_rate"] <= 0.05
    assert backtest["decline_rate"] <= 0.01
    assert backtest["decisions"][DECLINE]["precision"] >= 0.80


def test_policy_boundaries_and_backtest_costs_are_explicit() -> None:
    risks = np.asarray([0.05, 0.30, 0.79, 0.80, 0.95], dtype=np.float64)
    labels = np.asarray([0, 1, 0, 1, 1], dtype=np.uint8)
    amounts = np.asarray([10.0, 100.0, 20.0, 200.0, 300.0], dtype=np.float64)
    policy = DecisionPolicy(
        review_threshold=0.25,
        decline_threshold=0.80,
        maximum_review_rate=1.0,
        maximum_decline_rate=1.0,
        minimum_decline_precision=0.0,
    )

    decisions = apply_decision_policy(risks, policy)
    result = policy_backtest(
        labels,
        risks,
        amounts,
        policy,
        costs=CostAssumptions(
            review_cost=5.0,
            false_decline_cost=100.0,
            review_fraud_recovery_rate=0.50,
            decline_fraud_recovery_rate=1.0,
        ),
    )

    assert decisions.tolist() == [APPROVE, REVIEW, REVIEW, DECLINE, DECLINE]
    assert result["review_queue_rows"] == 2
    assert result["decisions"][DECLINE]["fraud_rows"] == 2
    assert result["recovered_fraud_amount"] == 550.0
    assert result["missed_fraud_loss"] == 50.0
    assert result["review_operations_cost"] == 10.0


def test_probability_to_points_has_stable_boundaries() -> None:
    risks = np.asarray([0.0, 0.1236, 0.9999, 1.0], dtype=np.float64)
    assert probability_to_risk_points(risks).tolist() == [0, 124, 1_000, 1_000]


def test_local_occlusion_returns_human_reason_codes() -> None:
    training = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.2, 1.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    model = LogisticRegression(random_state=42).fit(training, labels)
    explanations = local_occlusion_explanations(
        model,
        np.asarray([[1.4, 1.0]], dtype=np.float32),
        ("amount", "type_transfer"),
        np.asarray([0.0, 0.0], dtype=np.float32),
        top_k=2,
    )

    reasons = explanations["local_reasons"][0]
    assert reasons
    assert {reason["code"] for reason in reasons} <= {
        "HIGH_TRANSACTION_AMOUNT",
        "TRANSFER_PATTERN",
    }
