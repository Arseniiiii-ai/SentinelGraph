"""Render the v0.3 behavioural and anomaly-detection report."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{value * 100:.3f}%"


def _metric_rows(results: dict[str, Any], dataset_name: str) -> str:
    rows = []
    for model_name, payload in results["models"].items():
        metrics = payload["evaluation"][dataset_name]
        rows.append(
            f"| `{model_name}` | {metrics['average_precision']:.5f} | "
            f"{metrics['roc_auc']:.5f} | {_percent(metrics['recall'])} | "
            f"{_percent(metrics['precision'])} | "
            f"{metrics['false_positives_per_10k_legitimate']:.2f} | "
            f"{_percent(metrics['captured_fraud_amount_rate'])} |"
        )
    return "\n".join(rows)


def _slice_rows(results: dict[str, Any]) -> str:
    selected = results["selected_model"]
    slices = results["models"][selected]["future_slices"]
    rows = []
    for name, metrics in slices.items():
        rows.append(
            f"| `{name}` | {metrics['rows']:,} | {metrics['fraud_rows']:,} | "
            f"{metrics['average_precision']:.5f} | "
            f"{_percent(metrics['recall'])} | "
            f"{metrics['false_positives_per_10k_legitimate']:.2f} |"
        )
    return "\n".join(rows)


def render_behavioural_report(results: dict[str, Any]) -> str:
    """Return the generated v0.3 experiment report."""
    split = results["development_split"]
    feature_store = results["feature_store"]
    history = feature_store["history_contract"]
    comparison = results["comparison_to_v0_2"]
    selected = results["selected_model"]
    selection = results["selection_evidence"]
    promotion = results["promotion_decision"]
    feature_names = results["feature_names"]
    behavioural_names = results["behavioural_feature_names"]

    return f"""# SentinelGraph v0.3 Behavioural and Anomaly Report

## Scope

v0.3 adds point-in-time account velocity, amount-deviation, transaction-type,
and counterparty-diversity features plus a legitimate-only Isolation Forest.
It compares a supervised behavioural model, standalone anomaly detection, and
an anomaly-augmented supervised model. Graph features, probability calibration,
and decision orchestration remain out of scope.

The model matrix contains **{len(feature_names)}** numeric features, including
**{len(behavioural_names)}** new behavioural features. Raw account identifiers
are used only as aggregation keys and are never emitted as model inputs.

## Point-in-time contract

- Every history window ends at step
  **{history["window_upper_bound_steps"]}** relative to the current event.
- Transactions in the same PaySim hour are excluded because their within-hour
  ordering is not guaranteed.
- Future events and fraud labels are never used in feature computation.
- New accounts receive explicit cold-start flags and bounded sentinel values.

## Temporal experiment

- Model-development window: steps **{split["development_min_step"]}–{split["development_max_step"]}**
- Threshold-validation window: steps **{split["validation_min_step"]}–{split["validation_max_step"]}**
- Final future holdout: steps **{split["future_min_step"]}–{split["future_max_step"]}**
- Maximum validation FPR: **{_percent(results["target_maximum_fpr"])}**
- Deterministic legitimate-row cap: **{results["maximum_legitimate_rows"]:,}**

The selected model is **`{selected}`**. The anomaly-augmented challenger added
only **{selection["anomaly_validation_average_precision_gain"]:+.5f}**
validation PR-AUC against a required **+{selection["minimum_required_gain"]:.3f}**
complexity hurdle, so the simpler supervised behavioural model is retained.
This decision uses validation results only.

## Temporal validation

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{_metric_rows(results, "validation")}

## Future-time holdout

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{_metric_rows(results, "future_time_holdout")}

## Future new-account holdout

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{_metric_rows(results, "new_account_holdout")}

## Comparison with v0.2

The v0.2 histogram-gradient-boosting benchmark had future PR-AUC
**{comparison["baseline_average_precision"]:.5f}**, recall
**{_percent(comparison["baseline_recall"])}**, and
**{comparison["baseline_false_positives_per_10k"]:.2f}** false positives per
10,000 legitimate transactions.

The selected v0.3 model changes:

- PR-AUC by **{comparison["average_precision_delta"]:+.5f}**;
- recall by **{_percent(comparison["recall_delta"])}**;
- false positives per 10,000 legitimate by
  **{comparison["false_positives_per_10k_delta"]:+.2f}**;
- captured fraud amount by
  **{_percent(comparison["captured_fraud_amount_rate_delta"])}**.

Promotion decision: **{"PASS" if promotion["passed"] else "FAIL"}**. The
challenger must improve future PR-AUC and keep future FPR within the configured
capacity budget.

## Selected-model error slices

| Slice | Rows | Fraud rows | PR-AUC | Recall | FP / 10k legitimate |
| --- | ---: | ---: | ---: | ---: | ---: |
{_slice_rows(results)}

## Interpretation guardrails

1. PaySim is synthetic; results demonstrate engineering quality, not expected
   production fraud performance.
2. The Isolation Forest learns only legitimate development behaviour, but an
   unusual event is not automatically fraudulent.
3. The anomaly score is intentionally uncalibrated. Probability calibration is
   reserved for v0.5.
4. The future test is not used for feature fitting, model fitting, threshold
   selection, or candidate selection.
5. Origin accounts in PaySim are overwhelmingly cold-start entities, so
   destination-history and explicit new-account handling are especially
   important.
"""


def write_behavioural_report(path: Path, results: dict[str, Any]) -> None:
    """Write the generated behavioural report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_behavioural_report(results), encoding="utf-8")
