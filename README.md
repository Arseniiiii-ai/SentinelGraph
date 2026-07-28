# SentinelGraph

Production-grade fraud, account-takeover, and risk intelligence platform.

SentinelGraph turns transaction events into calibrated risk scores and
investigation cases. It combines:

- supervised fraud classification;
- unsupervised anomaly detection;
- customer and account behavioural features;
- transaction-graph features;
- explainable, cost-aware decision thresholds;
- real-time and batch scoring;
- model, data, and service monitoring;
- a grounded, read-only investigator copilot.

The project is designed as a portfolio-grade ML product rather than a
notebook-only model comparison.

## Business problem

Fraud teams must identify a small number of harmful transactions among a very
large number of legitimate events. A useful system must detect fraud while
controlling false positives, explain its decisions, respond quickly, and remain
reliable as customer behaviour changes.

SentinelGraph will answer:

1. How risky is this transaction?
2. Why was it scored as risky?
3. Is the behaviour unusual for this account?
4. Is the account connected to previously suspicious entities?
5. Should the transaction be approved, reviewed, or declined?
6. What evidence should an investigator examine next?

## Initial dataset

The first release uses PaySim transaction data because it contains account
identifiers, transaction types, balances, amounts, time steps, and fraud labels.
That structure supports tabular, behavioural, temporal, and graph features.

The dataset is synthetic, so the final report must clearly state that measured
performance does not represent production fraud performance.

The canonical Kaggle version is pinned to `ealaxi/paysim1`, version 2, under
CC BY-SA 4.0. Acquisition verifies the extracted CSV against SHA-256
`16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.

## Product releases

| Release | Outcome |
| --- | --- |
| v0.1 | Reproducible data validation and leakage-safe split |
| v0.2 | Rules, logistic regression, and gradient-boosting baselines |
| v0.3 | Behavioural and anomaly detection features |
| v0.4 | Transaction-graph features and graph experiments |
| v0.5 | Calibrated decision engine and explainability |
| v0.6 | FastAPI scoring, PostgreSQL case storage, and analyst feedback |
| v0.7 | MLflow, tests, Docker, CI/CD, and monitoring |
| v0.8 | Grounded investigator copilot |
| v1.0 | Cloud deployment, benchmark report, demo, and model card |

## Primary evaluation metrics

- PR-AUC;
- recall at a fixed false-positive rate;
- false positives per 10,000 legitimate transactions;
- expected financial value or prevented-loss proxy;
- calibration and Brier score;
- recall on new accounts and future time windows;
- p50 and p95 inference latency;
- model and data drift.

Accuracy is not a primary metric because fraud is highly imbalanced.

## Planned architecture

```text
Transaction events
    -> validation
    -> online/offline feature pipeline
    -> supervised model
    -> anomaly detector
    -> graph risk features
    -> calibrated decision engine
    -> approve / review / decline
    -> case database and investigator dashboard
    -> monitoring and feedback
```

## Repository structure

```text
sentinelgraph/
├── data/
│   ├── metadata/          # tracked provenance, contract, profile, split JSON
│   ├── raw/               # ignored archive and canonical CSV
│   ├── interim/           # ignored DuckDB analytical cache
│   └── processed/         # ignored leakage-safe Parquet splits
├── models/
│   ├── v0.2/              # ignored fitted baseline binaries
│   └── v0.3/              # ignored behavioural/anomaly binaries
├── reports/
│   ├── v0.2/baseline_metrics.json
│   └── v0.3/behavioural_metrics.json
├── docs/
│   ├── BASELINE_REPORT.md
│   ├── BEHAVIOURAL_FEATURES.md
│   ├── BEHAVIOURAL_REPORT.md
│   ├── DATA_DICTIONARY.md
│   ├── DATA_QUALITY_REPORT.md
│   ├── DATA_SOURCE.md
│   ├── PROJECT_CHARTER.md
│   └── SPLIT_SPEC.md
├── src/
│   └── sentinelgraph/
│       ├── data/
│       │   ├── contract.py
│       │   ├── pipeline.py
│       │   ├── provenance.py
│       │   ├── quality.py
│       │   ├── report.py
│       │   ├── schema.py
│       │   └── splits.py
│       ├── modeling/
│       │   ├── advanced.py
│       │   ├── advanced_report.py
│       │   ├── anomaly.py
│       │   ├── behaviour.py
│       │   ├── features.py
│       │   ├── metrics.py
│       │   ├── models.py
│       │   ├── report.py
│       │   ├── rules.py
│       │   └── train.py
│       └── __init__.py
├── tests/
│   ├── test_data_contract.py
│   ├── test_anomaly_models.py
│   ├── test_behavioural_features.py
│   ├── test_model_features.py
│   ├── test_model_metrics.py
│   ├── test_models.py
│   ├── test_provenance.py
│   ├── test_schema.py
│   ├── test_splits.py
│   ├── test_v01_artifacts.py
│   ├── test_v02_artifacts.py
│   └── test_v03_artifacts.py
├── .gitignore
├── pyproject.toml
├── README.md
└── ROADMAP.md
```

## Current status

**v0.3 behavioural ML stage complete:** strict prior-step account histories,
velocity, amount deviation, transaction-type behaviour, counterparty diversity,
and legitimate-only anomaly detection now extend the v0.2 baselines.

Observed v0.1 facts:

- 6,362,620 transactions across steps 1–743;
- 8,213 fraud rows (0.1291%);
- no syntactically missing cells or exact duplicate rows;
- 16 zero-amount fraud `CASH_OUT` rows retained as an explicit warning;
- training ends at step 520 and future evaluation begins at step 521.

See [the full data-quality report](docs/DATA_QUALITY_REPORT.md).

Observed v0.2 future-time results:

| Model | PR-AUC | Recall | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: |
| Large-transfer rule | 0.02019 | 33.47% | 649.97 | 49.72% |
| Logistic regression | 0.21612 | 29.56% | 58.24 | 74.45% |
| Histogram gradient boosting | 0.40481 | 47.45% | 56.08 | 80.11% |

Learned-model thresholds are selected only on steps 417–520 under a 1% FPR
budget. Final future steps 521–743 remain untouched until evaluation. See the
[baseline report](docs/BASELINE_REPORT.md) for validation, future-time, and
new-account results.

Observed v0.3 future-time results:

| Model | PR-AUC | Recall | FP / 10k legitimate | Captured fraud amount |
| --- | ---: | ---: | ---: | ---: |
| Isolation Forest | 0.04205 | 33.06% | 635.95 | 74.70% |
| Behavioural gradient boosting | 0.60967 | 69.04% | 79.01 | 92.61% |
| Behavioural + anomaly boosting | 0.60539 | 71.09% | 104.50 | 92.91% |

The behavioural gradient model is selected. Anomaly augmentation did not clear
its validation-only complexity hurdle and remains a documented negative result.
See the [feature dictionary](docs/BEHAVIOURAL_FEATURES.md) and
[v0.3 report](docs/BEHAVIOURAL_REPORT.md).

## Reproduce v0.1 through v0.3

Python 3.11 or newer and `uv` are recommended:

```bash
uv sync --extra dev
uv run sentinelgraph-data all
uv run sentinelgraph-baselines
uv run sentinelgraph-behaviour
uv run ruff check .
uv run mypy src
uv run pytest -q
```

`sentinelgraph-data all` downloads about 178 MiB and creates approximately
1 GiB of ignored raw, analytical, and processed artifacts.
`sentinelgraph-baselines` trains all v0.2 models, writes ignored model binaries,
and regenerates the tracked metrics and report. `sentinelgraph-behaviour`
builds the ignored point-in-time feature store, trains v0.3 challengers, and
regenerates tracked v0.3 metrics and documentation. Graph intelligence remains
for v0.4 and probability calibration remains for v0.5.

## Development principles

- Split data before fitting transformations.
- Never use future information in behavioural features.
- Compare every advanced model with a simple baseline.
- Optimize decisions and business cost, not accuracy alone.
- Treat probability calibration and thresholds as product components.
- Keep investigator actions human-approved.
- Report limitations and negative results.
- Add CV claims only after measuring them.
