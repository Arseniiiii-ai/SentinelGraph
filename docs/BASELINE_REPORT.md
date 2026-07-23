# SentinelGraph v0.2 Baseline Model Report

## Scope

This release compares a label-prior dummy model, a transparent transaction
rule, balanced logistic regression, and histogram gradient boosting. It does
not add behavioural, anomaly, graph, calibration, or decision-engine features.

Approved model inputs are `amount`, `log_amount`, `hour_sin`, `hour_cos`, `type_cash_in`, `type_cash_out`, `type_debit`, `type_payment`, `type_transfer`. Account IDs, balance fields,
`isFlaggedFraud`, `isFraud`, and `source_row_number` are prohibited features.

## Temporal development design

- Model-development window: steps **1–416**
- Threshold-validation window: steps **417–520**
- Final future holdout: steps **521–743**
- Threshold objective: maximum validation recall with FPR no higher than
  **1.000%** (100 false positives per
  10,000 legitimate transactions)

No final-holdout row is used for fitting or threshold selection.

## Training

| Model | Fit rows | Fit fraud rows | Threshold strategy | Threshold |
| --- | ---: | ---: | --- | ---: |
| `dummy_prior` | 5,990,447 | 4,707 | `validation_fpr_budget` | 0.00078575105 |
| `large_transfer_rule` | 5,990,447 | 4,707 | `fixed_policy_threshold` | 0.5 |
| `logistic_regression` | 5,990,447 | 4,707 | `validation_fpr_budget` | 0.94857776 |
| `hist_gradient_boosting` | 1,004,707 | 4,707 | `validation_fpr_budget` | 0.95151736 |

The gradient-boosting baseline retains every development fraud row and uses a
deterministic cap on legitimate rows. Logistic regression uses the complete
development window. The rule uses its fixed documented policy threshold and
may exceed the common FPR budget; learned models select their threshold only on
the temporal validation window.

## Temporal validation

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dummy_prior` | 0.01173 | 0.50000 | 0.000% | 0.000% | 0.00 | 0.000% |
| `large_transfer_rule` | 0.02834 | 0.64322 | 35.382% | 5.867% | 673.81 | 49.812% |
| `logistic_regression` | 0.24088 | 0.92798 | 31.285% | 27.162% | 99.57 | 73.628% |
| `hist_gradient_boosting` | 0.40765 | 0.94829 | 49.162% | 36.949% | 99.57 | 76.537% |

## Future-time holdout

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dummy_prior` | 0.00867 | 0.50000 | 0.000% | 0.000% | 0.00 | 0.000% |
| `large_transfer_rule` | 0.02019 | 0.63485 | 33.470% | 4.308% | 649.97 | 49.725% |
| `logistic_regression` | 0.21612 | 0.93280 | 29.564% | 30.740% | 58.24 | 74.451% |
| `hist_gradient_boosting` | 0.40481 | 0.94625 | 47.451% | 42.520% | 56.08 | 80.114% |

## Future new-account holdout

| Model | PR-AUC | ROC-AUC | Recall | Precision | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dummy_prior` | 0.00867 | 0.50000 | 0.000% | 0.000% | 0.00 | 0.000% |
| `large_transfer_rule` | 0.02024 | 0.63518 | 33.539% | 4.317% | 650.39 | 49.749% |
| `logistic_regression` | 0.21635 | 0.93275 | 29.625% | 30.766% | 58.32 | 74.487% |
| `hist_gradient_boosting` | 0.40463 | 0.94630 | 47.425% | 42.488% | 56.16 | 80.114% |

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
