# SentinelGraph v0.5 Calibrated Risk Engine Report

## Scope

v0.5 converts model evidence into calibrated fraud probabilities, 0–1000 risk
points, and auditable `approve`, `review`, or simulated `decline` decisions.
The engine combines the v0.3 behavioural champion, the legitimate-only anomaly
detector, and the v0.4 graph challenger as calibration candidates. API serving,
automatic customer actions, and investigator storage remain out of scope.

## Leakage-safe governance timeline

| Window | Steps | Purpose |
| --- | ---: | --- |
| Development | 1–416 | Fit base component models |
| Calibration fit | 417–450 | Fit calibration candidates |
| Calibration selection | 451–484 | Select calibration method |
| Policy selection | 485–520 | Select decision thresholds |
| Future holdout | 521–743 | One-time final evaluation |

No future row is used to fit a model, calibrator, decision threshold, cost
assumption, or explanation baseline.

## Calibration selection

| Candidate | PR-AUC | Brier | Log loss | ECE | Mean risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw_behavioural_reference` | 0.68602 | 0.044398 | 0.162220 | 0.094003 | 10.740% |
| `sigmoid_behavioural` | 0.68602 | 0.007151 | 0.026928 | 0.004679 | 1.517% |
| `isotonic_behavioural` | 0.60973 | 0.006919 | 0.026729 | 0.002928 | 1.505% |
| `logistic_score_stack` ✓ | 0.69193 | 0.006748 | 0.026249 | 0.003778 | 1.571% |

Selected calibrator: **`logistic_score_stack`**.

The selected method improves selection-window Brier score by
**+0.037650** against the raw behavioural
probability while retaining at least
**99.000%** of its PR-AUC.

## Decision policy

| Parameter | Value |
| --- | ---: |
| Review threshold | 0.35011340 |
| Decline threshold | 0.85539184 |
| Maximum review rate | 1.000% |
| Maximum decline rate | 0.100% |
| Minimum decline precision | 80.000% |

`Decline` is an offline recommendation only. SentinelGraph v0.5 never performs
an automatic customer-impacting action.

## Holdout evaluation

| Dataset | PR-AUC | Brier | ECE | Review rate | Decline rate | Flagged fraud recall | Recovered fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `policy_selection` | 0.61645 | 0.005461 | 0.003978 | 0.997% | 0.100% | 62.113% | 79.277% |
| `future_time_holdout` | 0.61382 | 0.005467 | 0.006535 | 0.979% | 0.134% | 61.472% | 83.930% |
| `new_account_holdout` | 0.61709 | 0.005438 | 0.006452 | 0.974% | 0.133% | 61.434% | 83.930% |

## Future decision distribution

| Decision | Rows | Rate | Fraud | Legitimate | Precision | Fraud amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `approve` | 277,490 | 98.887% | 937 | 276,553 | 0.338% | 8.782% |
| `review` | 2,747 | 0.979% | 1,123 | 1,624 | 40.881% | 36.436% |
| `decline` | 376 | 0.134% | 372 | 4 | 98.936% | 54.781% |

## Operational value simulation

The simulation uses explicit proxy assumptions rather than claiming real bank
economics:

- investigation cost: **5.00** per review;
- false-decline friction cost: **100.00**;
- reviewed-fraud recovery: **80.000%**;
- declined-fraud recovery: **100.000%**.

On the untouched future holdout:

- approve-all loss proxy: **3,627,405,635.40**;
- calibrated-policy cost proxy: **582,926,193.84**;
- simulated net savings: **3,044,479,441.56**;
- simulated cost reduction: **83.930%**.

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
| `HIGH_TRANSACTION_AMOUNT` | `amount` | 0.404505 |
| `CASH_OUT_PATTERN` | `type_cash_out` | 0.315558 |
| `MODEL_FEATURE_ELEVATED` | `hour_cos` | 0.074761 |
| `DESTINATION_GRAPH_ACTIVITY` | `destination_log_graph_in_tx_count` | 0.055449 |
| `DESTINATION_TIMING_PATTERN` | `destination_hours_since_last` | 0.026378 |
| `NEW_DESTINATION_ACCOUNT` | `destination_is_new` | 0.012864 |
| `MODEL_FEATURE_ELEVATED` | `destination_log_amount_sum_168h` | 0.011910 |
| `HIGH_TRANSACTION_AMOUNT` | `log_amount` | 0.011493 |
| `MODEL_FEATURE_ELEVATED` | `hour_sin` | 0.009150 |
| `GRAPH_COMPONENT_PATTERN` | `component_size_log_ratio` | 0.007793 |

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
