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

- [x] Create point-in-time account velocity features
- [x] Add counterparty diversity and amount deviation
- [x] Add historical transaction-type behaviour
- [x] Train an advanced histogram-gradient-boosting challenger
- [x] Train Isolation Forest on legitimate behaviour
- [x] Evaluate anomaly-score augmentation with a complexity hurdle
- [x] Optimize review threshold under capacity constraints
- [x] Complete sliced error analysis

Exit criterion: an advanced model beats the baseline on the future-time test
without an unacceptable increase in false positives. **Met in v0.3.**

Probability calibration remains deliberately scheduled for v0.5 so v0.3 can
measure ranking and anomaly value independently.

## Phase 3 — Graph intelligence

- [x] Build account transaction graph
- [x] Calculate degree, fan-in/fan-out, flow, and component features
- [x] Define and enforce a leakage-safe neighbour-label policy
- [x] Compare graph features against the behavioural model
- [x] Evaluate GraphSAGE eligibility after a strong non-GNN baseline
- [x] Document whether graph and GNN complexity add enough value to keep

Exit criterion: measured incremental value and a clear keep/remove decision.
**Met in v0.4:** graph features and GraphSAGE are not promoted on PaySim; the
v0.3 behavioural model remains champion.

## Phase 4 — Decision engine and explanations

- [x] Combine classifier, anomaly, and graph scores
- [x] Add probability calibration
- [x] Implement approve/review/decline simulation
- [x] Add reason codes
- [x] Add SHAP or equivalent explanations
- [x] Backtest investigator queue size and captured fraud amount
- [x] Add policy regression tests

Exit criterion: decision thresholds have quantified operational consequences.
**Met in v0.5:** a temporally governed score stack is calibrated, converted to
three-way decisions, explained, and backtested under explicit queue and cost
assumptions.

## Phase 5 — Product API and case workflow

- [x] Create FastAPI single-score endpoint
- [x] Create batch-score endpoint
- [x] Add health/readiness endpoints
- [x] Store predictions and feature versions in PostgreSQL
- [x] Add investigator case and feedback tables
- [x] Build investigation dashboard
- [x] Add authentication and input limits
- [x] Run local latency and failure-path tests

Exit criterion: a reviewer can submit a transaction and investigate the case.
**Implemented on the v0.6 development branch:** production PostgreSQL migration,
single/batch scoring, durable audit history, investigator feedback, and a web
console are covered. Container and distributed load testing remain v0.7 scope.

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
