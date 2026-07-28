# SentinelGraph v0.3 Behavioural and Anomaly Report

## Scope

v0.3 adds point-in-time account velocity, amount-deviation, transaction-type,
and counterparty-diversity features plus a legitimate-only Isolation Forest.
It compares a supervised behavioural model, standalone anomaly detection, and
an anomaly-augmented supervised model. Graph features, probability calibration,
and decision orchestration remain out of scope.

The model matrix contains **32** numeric features, including
**23** new behavioural features. Raw account identifiers
are used only as aggregation keys and are never emitted as model inputs.

## Point-in-time contract

- Every history window ends at step
  **-1** relative to the current event.
- Transactions in the same PaySim hour are excluded because their within-hour
  ordering is not guaranteed.
- Future events and fraud labels are never used in feature computation.
- New accounts receive explicit cold-start flags and bounded sentinel values.

## Temporal experiment

- Model-development window: steps **1–416**
- Threshold-validation window: steps **417–520**
- Final future holdout: steps **521–743**
- Maximum validation FPR: **1.000%**
- Deterministic legitimate-row cap: **999,999**

The selected model is **`behavioural_hist_gradient_boosting`**. The anomaly-augmented challenger added
only **+0.00013**
validation PR-AUC against a required **+0.005**
complexity hurdle, so the simpler supervised behavioural model is retained.
This decision uses validation results only.

## Temporal validation

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `isolation_forest` | 0.10689 | 0.88268 | 20.764% | 19.858% | 99.46 | 50.999% |
| `behavioural_hist_gradient_boosting` | 0.63768 | 0.97646 | 69.646% | 45.526% | 98.91 | 92.143% |
| `behavioural_anomaly_hist_gradient_boosting` | 0.63781 | 0.97505 | 70.857% | 45.761% | 99.68 | 92.360% |

## Future-time holdout

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `isolation_forest` | 0.04205 | 0.79474 | 33.059% | 4.347% | 635.95 | 74.698% |
| `behavioural_hist_gradient_boosting` | 0.60967 | 0.97183 | 69.038% | 43.307% | 79.01 | 92.610% |
| `behavioural_anomaly_hist_gradient_boosting` | 0.60539 | 0.96961 | 71.094% | 37.295% | 104.50 | 92.907% |

## Future new-account holdout

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `isolation_forest` | 0.05915 | 0.79650 | 32.921% | 4.507% | 610.27 | 74.686% |
| `behavioural_hist_gradient_boosting` | 0.60990 | 0.97187 | 69.015% | 43.338% | 78.94 | 92.612% |
| `behavioural_anomaly_hist_gradient_boosting` | 0.60632 | 0.96959 | 71.158% | 37.405% | 104.17 | 92.920% |

## Comparison with v0.2

The v0.2 histogram-gradient-boosting benchmark had future PR-AUC
**0.40481**, recall
**47.451%**, and
**56.08** false positives per
10,000 legitimate transactions.

The selected v0.3 model changes:

- PR-AUC by **+0.20487**;
- recall by **21.587%**;
- false positives per 10,000 legitimate by
  **+22.93**;
- captured fraud amount by
  **12.496%**.

Promotion decision: **PASS**. The
challenger must improve future PR-AUC and keep future FPR within the configured
capacity budget.

## Selected-model error slices

| Slice | Rows | Fraud rows | PR-AUC | Recall | FP / 10k legitimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cash_out` | 90,363 | 1,216 | 0.53566 | 55.674% | 76.17 |
| `transfer` | 26,317 | 1,216 | 0.65589 | 82.401% | 605.16 |
| `origin_new` | 279,830 | 2,427 | 0.60990 | 69.015% | 78.95 |
| `origin_returning` | 783 | 5 | 0.57163 | 80.000% | 102.83 |
| `destination_new` | 125,070 | 1,539 | 0.66928 | 77.778% | 125.96 |
| `destination_returning` | 155,543 | 893 | 0.49344 | 53.975% | 41.51 |
| `amount_at_least_200k` | 70,318 | 1,618 | 0.73052 | 72.373% | 131.73 |

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
