# SentinelGraph v0.6 API and operations guide

## Purpose

The v0.6 API serves the released v0.5 calibrated risk engine and creates
durable investigator cases. It never performs an automatic customer decline;
`decline` is a high-risk recommendation that still enters human review.

## Start the service

Python 3.11+, the local v0.5 bundle, and PostgreSQL are required.

```bash
uv sync --extra dev
export SENTINELGRAPH_DATABASE_URL='postgresql+psycopg://sentinelgraph:sentinelgraph@localhost:5432/sentinelgraph'
export SENTINELGRAPH_API_KEY='replace-with-a-long-secret'
export SENTINELGRAPH_ACCOUNT_HASH_SALT='replace-with-an-independent-long-salt'
uv run alembic upgrade head
uv run sentinelgraph-api --host 127.0.0.1 --port 8000
```

Development uses explicit environment variables but permits a local default
key and salt. `SENTINELGRAPH_ENVIRONMENT=production` fails closed unless the
database is PostgreSQL, secrets are replaced, and
`SENTINELGRAPH_MODEL_SHA256` contains the expected lowercase SHA-256 digest.

## Operations

| Method and path | Authentication | Purpose |
| --- | --- | --- |
| `GET /health/live` | none | Process liveness |
| `GET /health/ready` | none | Model and database readiness |
| `POST /v1/score` | API key | Score one transaction atomically |
| `POST /v1/score/batch` | API key | Score a bounded all-or-nothing batch |
| `GET /v1/cases` | API key | Filtered investigator queue |
| `GET /v1/cases/{case_id}` | API key | Case evidence and audit history |
| `POST /v1/cases/{case_id}/decision` | API key | Record assignment, status, and feedback |
| `GET /dashboard` | none | UI shell; all case data calls require the key |

Protected calls use `X-API-Key`. Scoring calls also require
`Idempotency-Key`. Reusing an external ID or idempotency key with an identical
payload returns the stored result with `replayed=true`; changing the payload
returns HTTP 409.

## Online feature contract

A score request contains the current transaction and `historical_features`:

- `version` must equal `v0.6.0`;
- `as_of_step` must be strictly less than the transaction step;
- `values` must contain exactly all 50 historical behavioural/graph features;
- missing, unexpected, or non-finite values are rejected.

The API derives the ten event-known fields itself: amount, log amount, cyclic
hour, five transaction-type indicators, and destination-merchant indicator.
Together they reproduce the exact 60-column training order. The OpenAPI schema
at `/docs` is the canonical HTTP contract; Python clients can import
`HISTORICAL_FEATURE_NAMES` to generate or validate snapshots.

## Response contract

Every prediction returns:

- durable transaction and prediction IDs;
- calibrated probability and 0–1000 risk points;
- `approve`, `review`, or simulation-only `decline`;
- bounded reason codes and component scores;
- model, policy, and feature versions;
- request ID, model-call latency, creation time, and replay status.

Review and decline outputs create cases in the same database transaction.
Approved predictions remain auditable but do not create investigator work.

## Investigator concurrency

Case decisions include `expected_version`. If another investigator updates the
case first, the stale request receives HTTP 409 and must refresh. Closing a case
requires a disposition; feedback and the case event are written atomically.
Closed cases are immutable, and the decision endpoint cannot return work to
`open`.

## Privacy and security controls

- Account IDs are never model features.
- Only salted SHA-256 account pseudonyms are stored.
- API keys are compared in constant time.
- Request bytes and batch rows are bounded.
- Model SHA-256 is checked before unsafe Joblib deserialisation in production.
- Raw secrets are neither logged nor returned.
- The dashboard stores its key only in browser `sessionStorage`.

v0.6 API-key auth is intentionally narrow. Identity-aware RBAC, secret
rotation, gateway-wide rate limiting, TLS termination, and centralized audit
export are deployment controls for later milestones.

## Database changes

Alembic is the only production schema mechanism:

```bash
uv run alembic current
uv run alembic upgrade head
uv run alembic history
```

The API does not auto-create or auto-migrate tables during startup. This avoids
concurrent schema mutation when multiple replicas start together.
