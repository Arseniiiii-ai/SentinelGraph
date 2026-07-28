# SentinelGraph v0.4 Transaction-Graph Intelligence Report

## Scope

v0.4 builds a strict point-in-time account graph and compares graph-only and
behavioural-plus-graph challengers with the v0.3 behavioural champion. Graph
features contain no account IDs, fraud labels, future edges, or same-step
edges. Calibration and decision orchestration remain out of scope.

## PaySim graph topology

| Property | Value |
| --- | ---: |
| Accounts | 9,073,900 |
| Directed transaction edges | 6,362,620 |
| Unique directed pairs | 6,362,620 |
| Repeated directed pairs | 0 |
| Reciprocal pairs | 0 |
| Accounts observed in both roles | 1,769 |
| Final weak components | 2,711,280 |
| Largest weak component | 121 |
| Undirected cycle rank | 0 |

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

- Behavioural features: **32**
- New graph features: **28**
- Combined model features: **60**
- Development: steps **1–416**
- Threshold validation: steps **417–520**
- Final future holdout: steps **521–743**
- Maximum validation FPR: **1.000%**

## Temporal validation

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `behavioural_reference` | 0.63768 | 0.97646 | 69.646% | 45.526% | 98.91 | 92.143% |
| `graph_only_hist_gradient_boosting` | 0.63888 | 0.97625 | 67.132% | 44.810% | 98.14 | 90.924% |
| `behavioural_graph_hist_gradient_boosting` | 0.61971 | 0.97583 | 69.460% | 45.322% | 99.46 | 92.030% |

The best graph challenger, **`graph_only_hist_gradient_boosting`**,
changed validation PR-AUC by
**+0.00120** against a
required **+0.005** complexity hurdle.
The validation-selected candidate is
**`behavioural_reference`**.

## Future-time holdout

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `behavioural_reference` | 0.60967 | 0.97183 | 69.038% | 43.307% | 79.01 | 92.610% |
| `graph_only_hist_gradient_boosting` | 0.61776 | 0.97427 | 67.229% | 46.095% | 68.73 | 91.851% |
| `behavioural_graph_hist_gradient_boosting` | 0.60478 | 0.97254 | 68.380% | 43.228% | 78.51 | 92.460% |

## Future new-account holdout

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `behavioural_reference` | 0.60990 | 0.97187 | 69.015% | 43.338% | 78.94 | 92.612% |
| `graph_only_hist_gradient_boosting` | 0.61783 | 0.97431 | 67.244% | 46.102% | 68.78 | 91.857% |
| `behavioural_graph_hist_gradient_boosting` | 0.60493 | 0.97255 | 68.397% | 43.240% | 78.55 | 92.465% |

## Keep/remove decision

Graph feature promotion: **REMOVE**.

- Validation hurdle passed: **False**
- Future PR-AUC confirmation passed:
  **True**
- Future FPR capacity passed: **True**
- Release champion: **`behavioural_reference`**

Against the tracked v0.3 champion, the release champion changes:

- PR-AUC by **+0.00000**;
- recall by **+0.000 percentage points**;
- false positives per 10,000 legitimate by
  **+0.00**;
- captured fraud amount by
  **+0.000 percentage points**.

## GraphSAGE decision

GraphSAGE status: **not retained**.

GraphSAGE was not added: every directed account pair occurs once, there are no reciprocal pairs, and the non-GNN graph experiment must first demonstrate material incremental value. A GNN would add cost without a defensible message-passing advantage.

## Interpretation guardrails

1. PaySim is synthetic and has unusually sparse account reuse.
2. Weak component features are exact for prior steps, not approximations built
   from the final graph.
3. Absence of graph lift on PaySim would not imply that graph intelligence is
   useless on a real payment network with repeated entities.
4. Graph labels are never propagated without an availability timestamp.
5. The future holdout confirms a validation decision once; it is not used for
   iterative feature selection.
