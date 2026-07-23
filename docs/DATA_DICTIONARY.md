# PaySim Data Dictionary and Point-in-Time Availability

The table distinguishes source meaning from whether a value is permitted in a
leakage-safe model. “Available” means known when a transaction scoring request
arrives. “Restricted” means the source may be used for grouping, slicing, or
analysis but not passed directly to a baseline model.

| Field | Type | Source meaning | Null policy | Scoring-time status | v0.2 policy |
| --- | --- | --- | --- | --- | --- |
| `step` | integer | Simulation hour; one step represents one hour. | Required | Available as event time surrogate | Restricted to splitting and time derivation |
| `type` | enum | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, or `TRANSFER` | Required | Available | Feature |
| `amount` | non-negative float | Transaction amount in the simulated local currency | Required | Available | Feature; zero-value slice retained |
| `nameOrig` | string | Originating customer account | Required | Available | Grouping key only; never raw categorical feature |
| `oldbalanceOrg` | non-negative float | Origin balance recorded before the transaction | Required | Conceptually pre-event | Excluded for PaySim because publisher warns balance fields reflect fraud cancellation |
| `newbalanceOrig` | non-negative float | Origin balance after the transaction | Required | Post-event | Prohibited |
| `nameDest` | string | Destination customer (`C…`) or merchant (`M…`) | Required | Available | Grouping key only; never raw categorical feature |
| `oldbalanceDest` | non-negative float | Destination balance recorded before the transaction; merchant entries use zero where balance information is unavailable | Required | Conceptually pre-event but semantically missing for merchants | Excluded for PaySim |
| `newbalanceDest` | non-negative float | Destination balance after the transaction; merchant entries use zero where balance information is unavailable | Required | Post-event | Prohibited |
| `isFraud` | binary integer | Injected fraud ground-truth label | Required | Outcome/label | Target only |
| `isFlaggedFraud` | binary integer | Existing policy flag for large transfer attempts | Required | Policy output | Excluded from learned-model features; rules-baseline analysis only |
| `source_row_number` | positive integer | SentinelGraph-derived one-based source row identifier in DuckDB/Parquet outputs | Generated | Not a business feature | Traceability only |

## Semantic missingness

The CSV has no syntactically null cells. However, the publisher states that
destination balances are not available for merchant IDs beginning with `M`.
Their numeric zeroes are therefore **semantic missingness**, not proof of a
zero merchant balance. Data-contract tests preserve this distinction and do not
impute or reinterpret the raw values.

## Leakage controls

1. Fit preprocessing and models only on `train.parquet`.
2. Do not use any future-time row when producing historical account features.
3. Exclude all four PaySim balance fields from v0.2 learned baselines.
4. Never pass `isFraud`, `isFlaggedFraud`, or `source_row_number` as features.
5. Use account IDs only to calculate strictly prior-history features.
6. Report metrics on the complete future holdout and on its unseen-origin
   subset.
