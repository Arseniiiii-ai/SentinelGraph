# SentinelGraph v0.6 service report

## Outcome

v0.6 wraps the calibrated v0.5 risk engine in a versioned FastAPI contract, persists predictions and human-review cases in PostgreSQL, and provides a browser investigator console.

## Serving contract

- API operations: 8
- PostgreSQL tables: 6
- Model features: 60
- Event-derived features: 10
- Strictly-prior snapshot features: 50
- Feature contract: `v0.6.0`

The service rejects incomplete, extra, non-finite, same-step, or wrong-version feature snapshots. Account identifiers are salted and hashed before persistence; they are never model inputs.

## Local sequential inference benchmark

- Model: `v0.5`
- Rows: 25
- Median: 10.674 ms
- p95: 12.643 ms
- p99: 13.237 ms
- Throughput: 90.91 rows/s

These are workstation observations for sequential model inference including local occlusion explanations, not an infrastructure SLA. Database, network, concurrency, and cold-start time are excluded.

## Controls

- API-key authentication with constant-time comparison.
- Required idempotency keys and payload-fingerprint conflicts.
- Request and batch-size limits.
- Model checksum verification before Joblib deserialisation in production.
- Atomic score/case writes and optimistic case locking.
- Liveness and dependency-aware readiness probes.
- Simulation-only decline recommendations; no automated customer action.

## Scope boundary

The online feature store, global edge rate limiting, Docker, CI/CD, MLflow, and production monitoring belong to later deployment/MLOps milestones. v0.6 defines and validates their integration contracts.
