"""Render the SentinelGraph v0.4 transaction-graph report."""

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


def render_graph_report(results: dict[str, Any]) -> str:
    """Return the generated v0.4 graph experiment report."""
    split = results["development_split"]
    topology = results["feature_store"]["topology"]
    components = results["feature_store"]["components"]
    selection = results["selection_evidence"]
    decision = results["graph_keep_decision"]
    comparison = results["comparison_to_v0_3"]
    gnn = results["graphsage_decision"]

    return f"""# SentinelGraph v0.4 Transaction-Graph Intelligence Report

## Scope

v0.4 builds a strict point-in-time account graph and compares graph-only and
behavioural-plus-graph challengers with the v0.3 behavioural champion. Graph
features contain no account IDs, fraud labels, future edges, or same-step
edges. Calibration and decision orchestration remain out of scope.

## PaySim graph topology

| Property | Value |
| --- | ---: |
| Accounts | {topology["account_count"]:,} |
| Directed transaction edges | {topology["edge_count"]:,} |
| Unique directed pairs | {topology["unique_directed_pair_count"]:,} |
| Repeated directed pairs | {topology["repeated_directed_pair_count"]:,} |
| Reciprocal pairs | {topology["reciprocal_pair_count"]:,} |
| Accounts observed in both roles | {topology["cross_role_account_count"]:,} |
| Final weak components | {components["component_count"]:,} |
| Largest weak component | {components["largest_component_size"]:,} |
| Undirected cycle rank | {components["cycle_rank"]:,} |

Every PaySim transaction forms a unique directed pair. The graph is therefore
dominated by one-time origin leaves attached to repeated destinations. This
limits the information available to pair-history and message-passing models.

## Point-in-time contract

- Node degree, flow, and component state use edges from steps strictly smaller
  than the current transaction step.
- All edges from the current PaySim hour are applied only after every
  transaction in that hour has been scored.
- Raw account IDs are grouping keys only and never enter the model matrix.
- Fraud-neighbour propagation is excluded because PaySim has no timestamp for
  when a fraud label became known.

## Features and temporal design

- Behavioural features: **{results["behavioural_feature_count"]}**
- New graph features: **{results["graph_only_feature_count"]}**
- Combined model features: **{results["combined_feature_count"]}**
- Development: steps **{split["development_min_step"]}–{split["development_max_step"]}**
- Threshold validation: steps **{split["validation_min_step"]}–{split["validation_max_step"]}**
- Final future holdout: steps **{split["future_min_step"]}–{split["future_max_step"]}**
- Maximum validation FPR: **{_percent(results["target_maximum_fpr"])}**

## Temporal validation

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{_metric_rows(results, "validation")}

The best graph challenger, **`{selection["best_graph_validation_candidate"]}`**,
changed validation PR-AUC by
**{selection["graph_validation_average_precision_gain"]:+.5f}** against a
required **+{selection["minimum_required_gain"]:.3f}** complexity hurdle.
The validation-selected candidate is
**`{selection["validation_selected_model"]}`**.

## Future-time holdout

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{_metric_rows(results, "future_time_holdout")}

## Future new-account holdout

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{_metric_rows(results, "new_account_holdout")}

## Keep/remove decision

Graph feature promotion: **{"KEEP" if decision["kept"] else "REMOVE"}**.

- Validation hurdle passed: **{decision["validation_hurdle_passed"]}**
- Future PR-AUC confirmation passed:
  **{decision["future_average_precision_improved"]}**
- Future FPR capacity passed: **{decision["future_fpr_within_budget"]}**
- Release champion: **`{results["release_champion"]}`**

Against the tracked v0.3 champion, the release champion changes:

- PR-AUC by **{comparison["average_precision_delta"]:+.5f}**;
- recall by **{comparison["recall_delta"] * 100:+.3f} percentage points**;
- false positives per 10,000 legitimate by
  **{comparison["false_positives_per_10k_delta"]:+.2f}**;
- captured fraud amount by
  **{comparison["captured_fraud_amount_rate_delta"] * 100:+.3f} percentage points**.

## GraphSAGE decision

GraphSAGE status: **{gnn["status"]}**.

{gnn["rationale"]}

## Interpretation guardrails

1. PaySim is synthetic and has unusually sparse account reuse.
2. Weak component features are exact for prior steps, not approximations built
   from the final graph.
3. Absence of graph lift on PaySim would not imply that graph intelligence is
   useless on a real payment network with repeated entities.
4. Graph labels are never propagated without an availability timestamp.
5. The future holdout confirms a validation decision once; it is not used for
   iterative feature selection.
"""


def write_graph_report(path: Path, results: dict[str, Any]) -> None:
    """Write the generated graph report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_graph_report(results), encoding="utf-8")
