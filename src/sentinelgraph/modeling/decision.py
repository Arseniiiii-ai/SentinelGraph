"""Capacity-aware decisions, backtesting, and local risk explanations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sentinelgraph.modeling.metrics import positive_scores

APPROVE = "approve"
REVIEW = "review"
DECLINE = "decline"
DECISION_NAMES = (APPROVE, REVIEW, DECLINE)


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """Thresholds and capacity constraints for three-way risk decisions."""

    review_threshold: float
    decline_threshold: float
    maximum_review_rate: float
    maximum_decline_rate: float
    minimum_decline_precision: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.review_threshold):
            raise ValueError("review_threshold must be finite")
        if not np.isfinite(self.decline_threshold):
            raise ValueError("decline_threshold must be finite")
        if self.review_threshold < 0.0:
            raise ValueError("review_threshold must not be negative")
        if self.review_threshold > self.decline_threshold:
            raise ValueError("review_threshold must not exceed decline_threshold")
        for name, value in (
            ("maximum_review_rate", self.maximum_review_rate),
            ("maximum_decline_rate", self.maximum_decline_rate),
            ("minimum_decline_precision", self.minimum_decline_precision),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-compatible policy representation."""
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class CostAssumptions:
    """Transparent proxy costs used by the offline policy simulation."""

    review_cost: float = 5.0
    false_decline_cost: float = 100.0
    review_fraud_recovery_rate: float = 0.80
    decline_fraud_recovery_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.review_cost < 0.0 or self.false_decline_cost < 0.0:
            raise ValueError("operational costs must not be negative")
        for name, value in (
            ("review_fraud_recovery_rate", self.review_fraud_recovery_rate),
            ("decline_fraud_recovery_rate", self.decline_fraud_recovery_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")

    def to_dict(self) -> dict[str, float]:
        """Return JSON-compatible cost assumptions."""
        return {key: float(value) for key, value in asdict(self).items()}


def _validate_policy_arrays(
    labels: NDArray[np.uint8],
    risks: NDArray[np.float64],
    amounts: NDArray[np.float64],
) -> tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(risks, dtype=np.float64)
    transaction_amounts = np.asarray(amounts, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if probabilities.shape != values.shape or transaction_amounts.shape != values.shape:
        raise ValueError("labels, risks, and amounts must have equal shapes")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("labels must be binary")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("risks must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("risks must be between zero and one")
    if not np.all(np.isfinite(transaction_amounts)) or np.any(
        transaction_amounts < 0.0
    ):
        raise ValueError("amounts must be finite and non-negative")
    return values, probabilities, transaction_amounts


def _group_end_indexes(sorted_risks: NDArray[np.float64]) -> NDArray[np.int64]:
    changes = np.flatnonzero(sorted_risks[1:] != sorted_risks[:-1])
    return np.concatenate(
        (changes.astype(np.int64), np.asarray([sorted_risks.size - 1]))
    )


def select_decision_policy(
    labels: NDArray[np.uint8],
    risks: NDArray[np.float64],
    amounts: NDArray[np.float64],
    *,
    maximum_review_rate: float = 0.01,
    maximum_decline_rate: float = 0.001,
    minimum_decline_precision: float = 0.80,
) -> DecisionPolicy:
    """Select temporal policy thresholds under queue and precision constraints."""
    values, probabilities, transaction_amounts = _validate_policy_arrays(
        labels,
        risks,
        amounts,
    )
    for name, value in (
        ("maximum_review_rate", maximum_review_rate),
        ("maximum_decline_rate", maximum_decline_rate),
        ("minimum_decline_precision", minimum_decline_precision),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")

    row_count = values.size
    disabled_threshold = float(np.nextafter(1.0, np.inf))
    decline_threshold = disabled_threshold
    maximum_declines = int(np.floor(row_count * maximum_decline_rate))
    if maximum_declines > 0:
        order = np.argsort(-probabilities, kind="stable")
        sorted_risks = probabilities[order]
        sorted_labels = values[order]
        sorted_fraud_amounts = transaction_amounts[order] * sorted_labels
        ends = _group_end_indexes(sorted_risks)
        counts = ends + 1
        cumulative_fraud = np.cumsum(sorted_labels)[ends]
        cumulative_amount = np.cumsum(sorted_fraud_amounts)[ends]
        precision = cumulative_fraud / counts
        valid = (counts <= maximum_declines) & (
            precision >= minimum_decline_precision
        )
        if np.any(valid):
            valid_indexes = np.flatnonzero(valid)
            best_amount = cumulative_amount[valid_indexes].max()
            best_indexes = valid_indexes[
                cumulative_amount[valid_indexes] == best_amount
            ]
            selected = int(best_indexes[-1])
            decline_threshold = float(sorted_risks[ends[selected]])

    decline_mask = probabilities >= decline_threshold
    review_eligible = ~decline_mask
    maximum_reviews = int(np.floor(row_count * maximum_review_rate))
    review_threshold = decline_threshold
    eligible_risks = probabilities[review_eligible]
    if maximum_reviews > 0 and eligible_risks.size:
        sorted_review_risks = np.sort(eligible_risks)[::-1]
        ends = _group_end_indexes(sorted_review_risks)
        valid = ends + 1 <= maximum_reviews
        if np.any(valid):
            selected = int(np.flatnonzero(valid)[-1])
            review_threshold = float(sorted_review_risks[ends[selected]])

    return DecisionPolicy(
        review_threshold=review_threshold,
        decline_threshold=decline_threshold,
        maximum_review_rate=maximum_review_rate,
        maximum_decline_rate=maximum_decline_rate,
        minimum_decline_precision=minimum_decline_precision,
    )


def apply_decision_policy(
    risks: NDArray[np.float64],
    policy: DecisionPolicy,
) -> NDArray[np.str_]:
    """Map calibrated probabilities to approve, review, or decline."""
    probabilities = np.asarray(risks, dtype=np.float64)
    if probabilities.ndim != 1:
        raise ValueError("risks must be one-dimensional")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("risks must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("risks must be between zero and one")
    decisions = np.full(probabilities.shape, APPROVE, dtype="<U7")
    review_mask = probabilities >= policy.review_threshold
    decline_mask = probabilities >= policy.decline_threshold
    decisions[review_mask] = REVIEW
    decisions[decline_mask] = DECLINE
    return decisions


def policy_backtest(
    labels: NDArray[np.uint8],
    risks: NDArray[np.float64],
    amounts: NDArray[np.float64],
    policy: DecisionPolicy,
    *,
    costs: CostAssumptions = CostAssumptions(),
) -> dict[str, Any]:
    """Measure capacity, fraud capture, and transparent proxy economics."""
    values, probabilities, transaction_amounts = _validate_policy_arrays(
        labels,
        risks,
        amounts,
    )
    decisions = apply_decision_policy(probabilities, policy)
    masks = {name: decisions == name for name in DECISION_NAMES}
    fraud_mask = values == 1
    legitimate_mask = ~fraud_mask
    total_fraud = int(fraud_mask.sum())
    total_fraud_amount = float(transaction_amounts[fraud_mask].sum())

    outcome: dict[str, Any] = {}
    for name, mask in masks.items():
        fraud_rows = int(np.count_nonzero(mask & fraud_mask))
        legitimate_rows = int(np.count_nonzero(mask & legitimate_mask))
        fraud_amount = float(transaction_amounts[mask & fraud_mask].sum())
        outcome[name] = {
            "rows": int(mask.sum()),
            "rate": float(mask.mean()),
            "fraud_rows": fraud_rows,
            "legitimate_rows": legitimate_rows,
            "precision": fraud_rows / int(mask.sum()) if np.any(mask) else 0.0,
            "fraud_amount": fraud_amount,
            "fraud_amount_rate": (
                fraud_amount / total_fraud_amount if total_fraud_amount else 0.0
            ),
        }

    review_fraud_amount = float(outcome[REVIEW]["fraud_amount"])
    decline_fraud_amount = float(outcome[DECLINE]["fraud_amount"])
    recovered_fraud_amount = (
        review_fraud_amount * costs.review_fraud_recovery_rate
        + decline_fraud_amount * costs.decline_fraud_recovery_rate
    )
    missed_fraud_loss = total_fraud_amount - recovered_fraud_amount
    review_operations_cost = float(outcome[REVIEW]["rows"]) * costs.review_cost
    false_decline_cost = (
        float(outcome[DECLINE]["legitimate_rows"]) * costs.false_decline_cost
    )
    total_policy_cost = (
        missed_fraud_loss + review_operations_cost + false_decline_cost
    )
    approve_all_cost = total_fraud_amount

    return {
        "rows": int(values.size),
        "fraud_rows": total_fraud,
        "fraud_rate": float(values.mean()),
        "total_fraud_amount": total_fraud_amount,
        "policy": policy.to_dict(),
        "cost_assumptions": costs.to_dict(),
        "decisions": outcome,
        "review_queue_rows": int(outcome[REVIEW]["rows"]),
        "review_queue_rate": float(outcome[REVIEW]["rate"]),
        "decline_rate": float(outcome[DECLINE]["rate"]),
        "flagged_fraud_recall": (
            (
                int(outcome[REVIEW]["fraud_rows"])
                + int(outcome[DECLINE]["fraud_rows"])
            )
            / total_fraud
            if total_fraud
            else 0.0
        ),
        "recovered_fraud_amount": recovered_fraud_amount,
        "recovered_fraud_amount_rate": (
            recovered_fraud_amount / total_fraud_amount
            if total_fraud_amount
            else 0.0
        ),
        "missed_fraud_loss": missed_fraud_loss,
        "review_operations_cost": review_operations_cost,
        "false_decline_cost": false_decline_cost,
        "total_policy_cost": total_policy_cost,
        "approve_all_cost": approve_all_cost,
        "simulated_net_savings": approve_all_cost - total_policy_cost,
        "simulated_cost_reduction_rate": (
            (approve_all_cost - total_policy_cost) / approve_all_cost
            if approve_all_cost
            else 0.0
        ),
    }


def probability_to_risk_points(
    risks: NDArray[np.float64],
) -> NDArray[np.int32]:
    """Convert calibrated probabilities into stable 0–1000 risk points."""
    probabilities = np.asarray(risks, dtype=np.float64)
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("risks must be between zero and one")
    return np.rint(probabilities * 1_000.0).astype(np.int32)


FEATURE_REASON_CODES = {
    "amount": "HIGH_TRANSACTION_AMOUNT",
    "log_amount": "HIGH_TRANSACTION_AMOUNT",
    "type_transfer": "TRANSFER_PATTERN",
    "type_cash_out": "CASH_OUT_PATTERN",
    "origin_is_new": "NEW_ORIGIN_ACCOUNT",
    "destination_is_new": "NEW_DESTINATION_ACCOUNT",
    "origin_log_amount_deviation": "ORIGIN_AMOUNT_DEVIATION",
    "destination_log_amount_deviation": "DESTINATION_AMOUNT_DEVIATION",
    "origin_hours_since_last": "ORIGIN_TIMING_PATTERN",
    "destination_hours_since_last": "DESTINATION_TIMING_PATTERN",
    "origin_log_tx_count_24h": "ORIGIN_24H_VELOCITY",
    "origin_log_amount_sum_24h": "ORIGIN_24H_VALUE_VELOCITY",
    "destination_log_tx_count_24h": "DESTINATION_24H_VELOCITY",
    "destination_log_amount_sum_24h": "DESTINATION_24H_VALUE_VELOCITY",
    "origin_log_unique_destinations": "ORIGIN_COUNTERPARTY_DIVERSITY",
    "destination_log_unique_origins": "DESTINATION_COUNTERPARTY_DIVERSITY",
    "origin_log_graph_in_tx_count": "ORIGIN_GRAPH_ACTIVITY",
    "origin_log_graph_out_tx_count": "ORIGIN_GRAPH_ACTIVITY",
    "origin_log_graph_in_degree": "ORIGIN_GRAPH_ACTIVITY",
    "origin_log_graph_out_degree": "ORIGIN_GRAPH_ACTIVITY",
    "origin_log_graph_total_degree": "ORIGIN_GRAPH_ACTIVITY",
    "origin_graph_in_out_tx_log_ratio": "ORIGIN_GRAPH_FLOW_PATTERN",
    "origin_log_graph_received_amount": "ORIGIN_GRAPH_FLOW_PATTERN",
    "origin_log_graph_sent_amount": "ORIGIN_GRAPH_FLOW_PATTERN",
    "origin_graph_flow_log_ratio": "ORIGIN_GRAPH_FLOW_PATTERN",
    "origin_graph_prior_role_count": "ORIGIN_GRAPH_ACTIVITY",
    "destination_log_graph_in_tx_count": "DESTINATION_GRAPH_ACTIVITY",
    "destination_log_graph_out_tx_count": "DESTINATION_GRAPH_ACTIVITY",
    "destination_log_graph_in_degree": "DESTINATION_GRAPH_ACTIVITY",
    "destination_log_graph_out_degree": "DESTINATION_GRAPH_ACTIVITY",
    "destination_log_graph_total_degree": "DESTINATION_GRAPH_ACTIVITY",
    "destination_graph_in_out_tx_log_ratio": "DESTINATION_GRAPH_FLOW_PATTERN",
    "destination_log_graph_received_amount": "DESTINATION_GRAPH_FLOW_PATTERN",
    "destination_log_graph_sent_amount": "DESTINATION_GRAPH_FLOW_PATTERN",
    "destination_graph_flow_log_ratio": "DESTINATION_GRAPH_FLOW_PATTERN",
    "destination_graph_prior_role_count": "DESTINATION_GRAPH_ACTIVITY",
    "origin_log_component_size": "GRAPH_COMPONENT_PATTERN",
    "destination_log_component_size": "GRAPH_COMPONENT_PATTERN",
    "endpoints_same_component_prior": "GRAPH_COMPONENT_PATTERN",
    "log_combined_component_size": "GRAPH_COMPONENT_PATTERN",
    "component_size_log_ratio": "GRAPH_COMPONENT_PATTERN",
    "origin_component_is_isolated": "GRAPH_COMPONENT_PATTERN",
    "destination_component_is_isolated": "GRAPH_COMPONENT_PATTERN",
    "both_components_established": "GRAPH_COMPONENT_PATTERN",
}

REASON_CODE_DESCRIPTIONS = {
    "HIGH_TRANSACTION_AMOUNT": "Transaction amount materially increased risk.",
    "TRANSFER_PATTERN": "Transfer-type behaviour materially increased risk.",
    "CASH_OUT_PATTERN": "Cash-out behaviour materially increased risk.",
    "NEW_ORIGIN_ACCOUNT": "The origin account has no prior observed activity.",
    "NEW_DESTINATION_ACCOUNT": (
        "The destination account has no prior observed activity."
    ),
    "ORIGIN_AMOUNT_DEVIATION": (
        "Amount differs materially from the origin account's prior behaviour."
    ),
    "DESTINATION_AMOUNT_DEVIATION": (
        "Amount differs materially from the destination's prior history."
    ),
    "ORIGIN_TIMING_PATTERN": "Origin timing behaviour increased model risk.",
    "DESTINATION_TIMING_PATTERN": (
        "Destination timing behaviour increased model risk."
    ),
    "ORIGIN_24H_VELOCITY": "Origin transaction velocity increased risk.",
    "ORIGIN_24H_VALUE_VELOCITY": "Origin value velocity increased risk.",
    "DESTINATION_24H_VELOCITY": (
        "Destination transaction velocity increased risk."
    ),
    "DESTINATION_24H_VALUE_VELOCITY": (
        "Destination value velocity increased risk."
    ),
    "ORIGIN_COUNTERPARTY_DIVERSITY": (
        "Origin counterparty diversity increased risk."
    ),
    "DESTINATION_COUNTERPARTY_DIVERSITY": (
        "Destination counterparty diversity increased risk."
    ),
    "ORIGIN_GRAPH_ACTIVITY": "Origin graph activity increased risk.",
    "ORIGIN_GRAPH_FLOW_PATTERN": "Origin graph flow pattern increased risk.",
    "DESTINATION_GRAPH_ACTIVITY": "Destination graph activity increased risk.",
    "DESTINATION_GRAPH_FLOW_PATTERN": (
        "Destination graph flow pattern increased risk."
    ),
    "GRAPH_COMPONENT_PATTERN": "Prior graph-component structure increased risk.",
    "MODEL_FEATURE_ELEVATED": "A model feature materially increased risk.",
}


def local_occlusion_explanations(
    model: Any,
    features: NDArray[np.float32],
    feature_names: Sequence[str],
    baseline_values: NDArray[np.float32],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """Explain risk through deterministic local baseline occlusion.

    Each feature is replaced by its legitimate-development median. The drop in
    model probability is treated as a local contribution. This is a bounded,
    model-agnostic SHAP alternative for the first decision-engine release.
    """
    matrix = np.asarray(features, dtype=np.float32)
    baselines = np.asarray(baseline_values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names must align with feature columns")
    if baselines.shape != (matrix.shape[1],):
        raise ValueError("baseline_values must align with feature columns")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    original_scores = positive_scores(model, matrix)
    contributions = np.empty(matrix.shape, dtype=np.float64)
    for index in range(matrix.shape[1]):
        occluded = matrix.copy()
        occluded[:, index] = baselines[index]
        contributions[:, index] = original_scores - positive_scores(
            model,
            occluded,
        )

    positive_contributions = np.maximum(contributions, 0.0)
    global_order = np.argsort(
        -positive_contributions.mean(axis=0),
        kind="stable",
    )
    global_importance = [
        {
            "feature": feature_names[int(index)],
            "reason_code": FEATURE_REASON_CODES.get(
                feature_names[int(index)],
                "MODEL_FEATURE_ELEVATED",
            ),
            "mean_positive_contribution": float(
                positive_contributions[:, int(index)].mean()
            ),
        }
        for index in global_order
        if positive_contributions[:, int(index)].mean() > 0.0
    ]

    local: list[list[dict[str, Any]]] = []
    for row in positive_contributions:
        order = np.argsort(-row, kind="stable")
        reasons = []
        for index in order:
            contribution = float(row[int(index)])
            if contribution <= 0.0:
                continue
            feature_name = feature_names[int(index)]
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
            if len(reasons) == top_k:
                break
        local.append(reasons)
    return {
        "method": "local_legitimate_median_occlusion",
        "baseline": "per-feature median over legitimate development rows",
        "top_k": top_k,
        "global_importance": global_importance,
        "local_reasons": local,
    }
