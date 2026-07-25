# SentinelGraph Roadmap

## Phase 0 — Product and data contract

- [x] Define product and users
- [x] Define decisions and non-goals
- [x] Create initial transaction schema
- [x] Acquire PaySim data
- [x] Record dataset source, license, checksum, and size
- [x] Create a data dictionary
- [x] Profile missing values, duplicates, labels, entities, and time steps
- [x] Write a point-in-time feature availability table
- [x] Specify temporal and new-entity holdouts

Exit criterion: a reviewed data card and split specification. **Met in v0.1.**

## Phase 1 — Reproducible baseline

- [x] Build raw-to-processed data pipeline
- [x] Add automated data-contract checks
- [x] Build a rule baseline
- [x] Build DummyClassifier baseline
- [x] Build logistic regression pipeline
- [x] Build histogram gradient-boosting baseline
- [x] Add PR-AUC, recall-at-FPR, calibration, and cost metrics
- [x] Track baseline configuration and metrics as JSON artifacts
- [x] Publish baseline report

Exit criterion: one command reproduces the baseline metrics. **Met in v0.2.**

## Phase 2 — Behavioural ML

- [ ] Create point-in-time account velocity features
- [ ] Add counterparty diversity and amount deviation
- [ ] Add historical transaction-type behaviour
- [ ] Train LightGBM or CatBoost
- [ ] Train Isolation Forest on legitimate behaviour
- [ ] Calibrate probabilities
- [ ] Optimize review threshold under capacity constraints
- [ ] Complete sliced error analysis

Exit criterion: an advanced model beats the baseline on the future-time test
without an unacceptable increase in false positives.

## Phase 3 — Graph intelligence

- [ ] Build account transaction graph
- [ ] Calculate degree, fan-in/fan-out, velocity, and component features
- [ ] Add suspicious-neighbour exposure without target leakage
- [ ] Compare graph features against the behavioural model
- [ ] Test GraphSAGE only after a strong non-GNN baseline
- [ ] Document whether the GNN adds enough value to keep

Exit criterion: measured incremental value and a clear keep/remove decision.

## Phase 4 — Decision engine and explanations

- [ ] Combine classifier, anomaly, and graph scores
- [ ] Add probability calibration
- [ ] Implement approve/review/decline simulation
- [ ] Add reason codes
- [ ] Add SHAP or equivalent explanations
- [ ] Backtest investigator queue size and captured fraud amount
- [ ] Add policy regression tests

Exit criterion: decision thresholds have quantified operational consequences.

## Phase 5 — Product API and case workflow

- [ ] Create FastAPI single-score endpoint
- [ ] Create batch-score endpoint
- [ ] Add health/readiness endpoints
- [ ] Store predictions and feature versions in PostgreSQL
- [ ] Add investigator case and feedback tables
- [ ] Build investigation dashboard
- [ ] Add authentication and input limits
- [ ] Run load and failure tests

Exit criterion: a reviewer can submit a transaction and investigate the case.

## Phase 6 — MLOps and monitoring

- [ ] Add MLflow model registry
- [ ] Containerize API, database, dashboard, and tracking
- [ ] Add unit, data, model, integration, and API tests
- [ ] Add Ruff, mypy, pytest, and image scanning to CI
- [ ] Monitor service health, latency, and errors
- [ ] Monitor feature, score, and decision drift
- [ ] Simulate drift and demonstrate an alert
- [ ] Add champion/challenger comparison and rollback

Exit criterion: a failed quality or security check blocks release.

## Phase 7 — Investigator copilot

- [ ] Define a case-evidence schema
- [ ] Build read-only evidence retrieval
- [ ] Add timeline and graph-summary tools
- [ ] Require citations to case records
- [ ] Restrict tool and case access
- [ ] Create at least 100 evaluation cases
- [ ] Measure grounding, task completion, tool accuracy, latency, and cost
- [ ] Test prompt injection and unauthorized action requests

Exit criterion: the copilot is useful, grounded, and unable to take sensitive
actions.

## Phase 8 — Cloud, release, and job package

- [ ] Store artifacts in S3
- [ ] Publish images to ECR
- [ ] Deploy API to ECS
- [ ] Deploy PostgreSQL to RDS or document a lower-cost equivalent
- [ ] Configure CloudWatch logs and alarms
- [ ] Add budget controls
- [ ] Publish architecture, data card, model card, threat model, and runbook
- [ ] Record a 3-minute product demo
- [ ] Record a 15-minute technical walkthrough
- [ ] Create measured CV bullets
- [ ] Publish v1.0 release

Exit criterion: public portfolio package is complete and every claim is
reproducible.
