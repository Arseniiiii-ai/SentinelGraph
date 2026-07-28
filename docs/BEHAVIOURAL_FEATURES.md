# SentinelGraph v0.3 Behavioural Feature Dictionary

## Point-in-time policy

PaySim `step` is an hourly bucket and does not guarantee ordering inside an
hour. Every historical feature therefore uses only transactions from steps
strictly smaller than the current step. Events from the current step and all
future steps are excluded.

`nameOrig` and `nameDest` are aggregation keys only. Their raw values are not
written to the feature store and never enter a model matrix. Fraud labels and
balance-after fields are also excluded.

New accounts have `*_is_new = 1`, zero historical aggregates, and
`hours_since_last = 744`, paired with the explicit new-account flag so the
sentinel cannot be mistaken for observed history.

## Base transaction features

| Feature | Definition | Availability |
| --- | --- | --- |
| `amount` | Current transaction amount | Current event |
| `log_amount` | `ln(1 + amount)` | Current event |
| `hour_sin`, `hour_cos` | Cyclical encoding of PaySim hour | Current event |
| `type_*` | One-hot transaction type | Current event |
| `destination_is_merchant` | Whether the destination ID has PaySim's merchant prefix | Current event |

## Origin-account history

| Feature | Definition |
| --- | --- |
| `origin_is_new` | No origin transaction in a strictly earlier step |
| `origin_log_prior_tx_count` | `ln(1 + lifetime prior transaction count)` |
| `origin_log_prior_amount_mean` | `ln(1 + lifetime prior mean amount)` |
| `origin_log_amount_deviation` | Current log amount minus prior mean log amount |
| `origin_hours_since_last` | Steps since the most recent prior origin event |
| `origin_log_tx_count_24h` | Log count during the preceding 24 steps |
| `origin_log_amount_sum_24h` | Log amount sum during the preceding 24 steps |
| `origin_log_tx_count_168h` | Log count during the preceding 168 steps |
| `origin_log_amount_sum_168h` | Log amount sum during the preceding 168 steps |
| `origin_same_type_share` | Share of prior origin events with the current type |
| `origin_log_unique_destinations` | Log number of distinct prior destinations |

## Destination-account history

| Feature | Definition |
| --- | --- |
| `destination_is_new` | No received transaction in a strictly earlier step |
| `destination_log_prior_tx_count` | `ln(1 + lifetime prior received count)` |
| `destination_log_prior_amount_mean` | `ln(1 + lifetime prior received mean amount)` |
| `destination_log_amount_deviation` | Current log amount minus prior received mean log amount |
| `destination_hours_since_last` | Steps since the most recent prior received event |
| `destination_log_tx_count_24h` | Log received count during the preceding 24 steps |
| `destination_log_amount_sum_24h` | Log received sum during the preceding 24 steps |
| `destination_log_tx_count_168h` | Log received count during the preceding 168 steps |
| `destination_log_amount_sum_168h` | Log received sum during the preceding 168 steps |
| `destination_same_type_share` | Share of prior received events with the current type |
| `destination_log_unique_origins` | Log number of distinct prior origin counterparties |

## Known limitations

- PaySim origin IDs rarely repeat, so origin-history features are frequently in
  cold-start state.
- The features are calculated at hourly rather than event-time resolution.
- Merchant status is specific to PaySim's identifier convention.
- Behavioural aggregates do not use graph neighbourhoods; those belong to v0.4.
- Scores remain uncalibrated until v0.5.
