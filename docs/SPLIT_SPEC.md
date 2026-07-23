# Leakage-Safe Split Specification

## Objective

The split must estimate performance on later transactions and separately expose
cold-start performance for origin accounts absent from training. Random row
splits are prohibited.

## Definitions

Let `s_min` and `s_max` be the minimum and maximum observed PaySim steps. The
training cutoff is calculated from the first 70% of the inclusive observed time
span:

```text
train_steps = floor((s_max - s_min + 1) * 0.70)
train_end = s_min + train_steps - 1
```

- `train`: `step <= train_end`
- `future_time_holdout`: `step > train_end`
- `new_account_holdout`: future-time rows where `nameOrig` never appears at or
  before `train_end`

The new-account set is deliberately a **subset** of the future-time set. These
are two evaluation views, not three mutually exclusive model-fitting
partitions.

## Invariants

- No training row occurs after the cutoff.
- No future-time row occurs at or before the cutoff.
- Every new-account row is also a future-time row.
- No `new_account_holdout.nameOrig` appears in training.
- Preprocessing and historical features are fit or calculated from training
  only.
- The row-level `source_row_number` is retained for auditability but excluded
  from modelling.

## Artifacts

The pipeline materialises typed, Zstandard-compressed Parquet files:

```text
data/processed/train.parquet
data/processed/future_time_holdout.parquet
data/processed/new_account_holdout.parquet
```

Exact sizes, SHA-256 checksums, time ranges, row counts, fraud counts, and
origin counts are stored in `data/metadata/split_manifest.json`.
