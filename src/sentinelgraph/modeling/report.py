"""Render the v0.2 baseline comparison report."""

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


def _training_rows(results: dict[str, Any]) -> str:
    rows = []
    for model_name, payload in results["models"].items():
        rows.append(
            f"| `{model_name}` | {payload['fit_rows']:,} | "
            f"{payload['fit_fraud_rows']:,} | "
            f"`{payload['threshold_strategy']}` | "
            f"{payload['threshold']:.8g} |"
        )
    return "\n".join(rows)


def render_baseline_report(results: dict[str, Any]) -> str:
    """Return a Markdown report for the v0.2 baseline run."""
    split = results["development_split"]
    target_fpr = results["target_maximum_fpr"]
    target_fp_per_10k = target_fpr * 10_000
    feature_list = ", ".join(f"`{name}`" for name in results["feature_names"])

    return f"""# SentinelGraph v0.2 Baseline Model Report

## Scope

This release compares a label-prior dummy model, a transparent transaction
rule, balanced logistic regression, and histogram gradient boosting. It does
not add behavioural, anomaly, graph, calibration, or decision-engine features.

Approved model inputs are {feature_list}. Account IDs, balance fields,
`isFlaggedFraud`, `isFraud`, and `source_row_number` are prohibited features.

## Temporal development design

- Model-development window: steps **{split["development_min_step"]}–{split["development_max_step"]}**
- Threshold-validation window: steps **{split["validation_min_step"]}–{split["validation_max_step"]}**
- Final future holdout: steps **{split["future_min_step"]}–{split["future_max_step"]}**
- Threshold objective: maximum validation recall with FPR no higher than
  **{_percent(target_fpr)}** ({target_fp_per_10k:.0f} false positives per
  10,000 legitimate transactions)

No final-holdout row is used for fitting or threshold selection.

## Training

| Model | Fit rows | Fit fraud rows | Threshold strategy | Threshold |
| --- | ---: | ---: | --- | ---: |
{_training_rows(results)}

The gradient-boosting baseline retains every development fraud row and uses a
deterministic cap on legitimate rows. Logistic regression uses the complete
development window. The rule uses its fixed documented policy threshold and
may exceed the common FPR budget; learned models select their threshold only on
the temporal validation window.

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

## Interpretation guardrails

1. PaySim is synthetic, so these metrics are engineering baselines rather than
   estimates of production fraud performance.
2. The future label rate differs materially from the development period.
3. The new-account holdout is a subset of the future-time holdout and is not an
   independent third test set.
4. Brier score is recorded in the machine-readable metrics, but probability
   calibration is deliberately deferred to v0.5.
5. Model selection must prioritise PR-AUC, recall under the FPR budget, and
   captured fraud amount—not accuracy.
"""


def write_baseline_report(path: Path, results: dict[str, Any]) -> None:
    """Write the generated baseline report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_baseline_report(results), encoding="utf-8")
