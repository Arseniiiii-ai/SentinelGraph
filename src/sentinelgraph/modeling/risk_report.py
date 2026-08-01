"""Render the SentinelGraph v0.5 calibrated decision-engine report."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{value * 100:.3f}%"


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _calibration_candidate_rows(results: dict[str, Any]) -> str:
    selection = results["calibration_selection"]
    entries = {
        "raw_behavioural_reference": selection["raw_behavioural_reference"],
        **selection["candidates"],
    }
    rows = []
    for name, metrics in entries.items():
        marker = " ✓" if name == results["selected_calibrator"] else ""
        rows.append(
            f"| `{name}`{marker} | {metrics['average_precision']:.5f} | "
            f"{metrics['brier_score']:.6f} | {metrics['log_loss']:.6f} | "
            f"{metrics['expected_calibration_error']:.6f} | "
            f"{_percent(metrics['mean_predicted_risk'])} |"
        )
    return "\n".join(rows)


def _evaluation_rows(results: dict[str, Any]) -> str:
    rows = []
    for name, evaluation in results["evaluations"].items():
        calibration = evaluation["calibration"]
        backtest = evaluation["policy_backtest"]
        rows.append(
            f"| `{name}` | {calibration['average_precision']:.5f} | "
            f"{calibration['brier_score']:.6f} | "
            f"{calibration['expected_calibration_error']:.6f} | "
            f"{_percent(backtest['review_queue_rate'])} | "
            f"{_percent(backtest['decline_rate'])} | "
            f"{_percent(backtest['flagged_fraud_recall'])} | "
            f"{_percent(backtest['recovered_fraud_amount_rate'])} |"
        )
    return "\n".join(rows)


def _decision_rows(backtest: dict[str, Any]) -> str:
    rows = []
    for name in ("approve", "review", "decline"):
        decision = backtest["decisions"][name]
        rows.append(
            f"| `{name}` | {decision['rows']:,} | "
            f"{_percent(decision['rate'])} | {decision['fraud_rows']:,} | "
            f"{decision['legitimate_rows']:,} | "
            f"{_percent(decision['precision'])} | "
            f"{_percent(decision['fraud_amount_rate'])} |"
        )
    return "\n".join(rows)


def _importance_rows(results: dict[str, Any]) -> str:
    rows = []
    for item in results["explanations"]["global_importance"][:10]:
        rows.append(
            f"| `{item['reason_code']}` | `{item['feature']}` | "
            f"{item['mean_positive_contribution']:.6f} |"
        )
    return "\n".join(rows) or "| — | — | — |"


def render_risk_report(results: dict[str, Any]) -> str:
    """Return the generated v0.5 risk-engine report."""
    temporal = results["temporal_contract"]
    policy = results["decision_policy"]
    costs = results["cost_assumptions"]
    future = results["evaluations"]["future_time_holdout"]
    future_backtest = future["policy_backtest"]
    selection = results["calibration_selection"]

    return f"""# SentinelGraph v0.5 Calibrated Risk Engine Report

## Scope

v0.5 converts model evidence into calibrated fraud probabilities, 0–1000 risk
points, and auditable `approve`, `review`, or simulated `decline` decisions.
The engine combines the v0.3 behavioural champion, the legitimate-only anomaly
detector, and the v0.4 graph challenger as calibration candidates. API serving,
automatic customer actions, and investigator storage remain out of scope.

## Leakage-safe governance timeline

| Window | Steps | Purpose |
| --- | ---: | --- |
| Development | {temporal['development']['minimum_step']}–{temporal['development']['maximum_step']} | Fit base component models |
| Calibration fit | {temporal['calibration_fit']['minimum_step']}–{temporal['calibration_fit']['maximum_step']} | Fit calibration candidates |
| Calibration selection | {temporal['calibration_selection']['minimum_step']}–{temporal['calibration_selection']['maximum_step']} | Select calibration method |
| Policy selection | {temporal['policy_selection']['minimum_step']}–{temporal['policy_selection']['maximum_step']} | Select decision thresholds |
| Future holdout | {temporal['future_time_holdout']['minimum_step']}–{temporal['future_time_holdout']['maximum_step']} | One-time final evaluation |

No future row is used to fit a model, calibrator, decision threshold, cost
assumption, or explanation baseline.

## Calibration selection

| Candidate | PR-AUC | Brier | Log loss | ECE | Mean risk |
| --- | ---: | ---: | ---: | ---: | ---: |
{_calibration_candidate_rows(results)}

Selected calibrator: **`{results['selected_calibrator']}`**.

The selected method improves selection-window Brier score by
**{selection['selected_brier_improvement']:+.6f}** against the raw behavioural
probability while retaining at least
**{_percent(selection['minimum_ranking_retention'])}** of its PR-AUC.

## Decision policy

| Parameter | Value |
| --- | ---: |
| Review threshold | {policy['review_threshold']:.8f} |
| Decline threshold | {policy['decline_threshold']:.8f} |
| Maximum review rate | {_percent(policy['maximum_review_rate'])} |
| Maximum decline rate | {_percent(policy['maximum_decline_rate'])} |
| Minimum decline precision | {_percent(policy['minimum_decline_precision'])} |

`Decline` is an offline recommendation only. SentinelGraph v0.5 never performs
an automatic customer-impacting action.

## Holdout evaluation

| Dataset | PR-AUC | Brier | ECE | Review rate | Decline rate | Flagged fraud recall | Recovered fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{_evaluation_rows(results)}

## Future decision distribution

| Decision | Rows | Rate | Fraud | Legitimate | Precision | Fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{_decision_rows(future_backtest)}

## Operational value simulation

The simulation uses explicit proxy assumptions rather than claiming real bank
economics:

- investigation cost: **{_money(costs['review_cost'])}** per review;
- false-decline friction cost: **{_money(costs['false_decline_cost'])}**;
- reviewed-fraud recovery: **{_percent(costs['review_fraud_recovery_rate'])}**;
- declined-fraud recovery: **{_percent(costs['decline_fraud_recovery_rate'])}**.

On the untouched future holdout:

- approve-all loss proxy: **{_money(future_backtest['approve_all_cost'])}**;
- calibrated-policy cost proxy: **{_money(future_backtest['total_policy_cost'])}**;
- simulated net savings: **{_money(future_backtest['simulated_net_savings'])}**;
- simulated cost reduction: **{_percent(future_backtest['simulated_cost_reduction_rate'])}**.

These values are scenario outputs on synthetic PaySim data, not production
financial forecasts.

## Explanations and reason codes

The engine uses **local legitimate-median occlusion**: one approved feature at
a time is replaced by its median among legitimate development rows, all three
evidence components and the selected calibrator are rescored, and the drop in
final calibrated risk is recorded as a local contribution. This is a bounded,
end-to-end model-agnostic SHAP alternative for v0.5.

| Reason code | Feature | Mean positive contribution |
| --- | --- | ---: |
{_importance_rows(results)}

Tracked, de-identified examples are stored in
`reports/v0.5/reason_code_examples.json`. Raw account identifiers are never
included in explanation artifacts.

## Artifact reproducibility policy

The joblib risk bundle and its checksum manifest remain ignored under
`models/v0.5/`. Joblib byte streams are not treated as canonical evidence
because serialization bytes can change even when predictions and semantic
metrics are identical. Tracked JSON metrics, temporal contracts, thresholds,
reason codes, and reports are the release evidence.

## Limitations

1. PaySim is synthetic and has a substantial temporal fraud-rate shift.
2. Cost values are transparent proxies, not estimates of a real institution.
3. `Decline` remains simulation-only until human, legal, and policy review.
4. Occlusion contributions describe model sensitivity, not causal effects.
5. Calibration must be monitored and refreshed when the observed base rate
   changes.
"""


def write_risk_report(path: Path, results: dict[str, Any]) -> None:
    """Write the generated calibrated risk-engine report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_risk_report(results), encoding="utf-8")
