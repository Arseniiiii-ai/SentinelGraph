"""Generate reproducible v0.6 service-contract evidence and local latency data."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sentinelgraph.api.config import project_root
from sentinelgraph.api.inference import RiskScorer
from sentinelgraph.api.schemas import (
    EVENT_DERIVED_FEATURE_NAMES,
    FEATURE_VERSION,
    HISTORICAL_FEATURE_NAMES,
)
from sentinelgraph.modeling.graph import GRAPH_FEATURE_NAMES, load_graph_matrix

API_ENDPOINTS = (
    "GET /health/live",
    "GET /health/ready",
    "POST /v1/score",
    "POST /v1/score/batch",
    "GET /v1/cases",
    "GET /v1/cases/{case_id}",
    "POST /v1/cases/{case_id}/decision",
    "GET /dashboard",
)
DATABASE_TABLES = (
    "transactions",
    "predictions",
    "cases",
    "case_events",
    "investigator_feedback",
    "model_versions",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    latency = metrics["local_sequential_inference_benchmark"]
    contract = metrics["service_contract"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "# SentinelGraph v0.6 service report",
                "",
                "## Outcome",
                "",
                "v0.6 wraps the calibrated v0.5 risk engine in a versioned FastAPI "
                "contract, persists predictions and human-review cases in PostgreSQL, "
                "and provides a browser investigator console.",
                "",
                "## Serving contract",
                "",
                f"- API operations: {len(contract['api_endpoints'])}",
                f"- PostgreSQL tables: {len(contract['database_tables'])}",
                f"- Model features: {contract['model_feature_count']}",
                f"- Event-derived features: {contract['event_derived_feature_count']}",
                f"- Strictly-prior snapshot features: "
                f"{contract['historical_feature_count']}",
                f"- Feature contract: `{contract['feature_version']}`",
                "",
                "The service rejects incomplete, extra, non-finite, same-step, or "
                "wrong-version feature snapshots. Account identifiers are salted and "
                "hashed before persistence; they are never model inputs.",
                "",
                "## Local sequential inference benchmark",
                "",
                f"- Model: `{latency['model_version']}`",
                f"- Rows: {latency['rows']}",
                f"- Median: {latency['latency_ms']['p50']:.3f} ms",
                f"- p95: {latency['latency_ms']['p95']:.3f} ms",
                f"- p99: {latency['latency_ms']['p99']:.3f} ms",
                f"- Throughput: {latency['rows_per_second']:.2f} rows/s",
                "",
                "These are workstation observations for sequential model inference "
                "including local occlusion explanations, not an infrastructure SLA. "
                "Database, network, concurrency, and cold-start time are excluded.",
                "",
                "## Controls",
                "",
                "- API-key authentication with constant-time comparison.",
                "- Required idempotency keys and payload-fingerprint conflicts.",
                "- Request and batch-size limits.",
                "- Model checksum verification before Joblib deserialisation in production.",
                "- Atomic score/case writes and optimistic case locking.",
                "- Liveness and dependency-aware readiness probes.",
                "- Simulation-only decline recommendations; no automated customer action.",
                "",
                "## Scope boundary",
                "",
                "The online feature store, global edge rate limiting, Docker, CI/CD, "
                "MLflow, and production monitoring belong to later deployment/MLOps "
                "milestones. v0.6 defines and validates their integration contracts.",
                "",
            )
        ),
        encoding="utf-8",
    )


def benchmark_service(root: Path, *, rows: int = 25) -> dict[str, Any]:
    """Measure local end-to-end model calls over a deterministic matrix slice."""
    if rows <= 0:
        raise ValueError("rows must be positive")
    graph_path = root / "data/processed/graph_features.parquet"
    component_path = root / "data/processed/graph_components.npz"
    bundle_path = root / "models/v0.5/risk_bundle.joblib"
    dataset = load_graph_matrix(
        graph_path,
        component_path,
        where_sql=f"source_row_number <= {rows}",
    )
    if dataset.rows != rows:
        raise ValueError(f"requested {rows} benchmark rows, found {dataset.rows}")
    scorer = RiskScorer(bundle_path)
    scorer.score(dataset.features[0:1])
    latencies: list[float] = []
    started_all = time.perf_counter()
    for index in range(dataset.rows):
        started = time.perf_counter()
        scorer.score(dataset.features[index : index + 1])
        latencies.append((time.perf_counter() - started) * 1_000.0)
    elapsed = time.perf_counter() - started_all
    values = np.asarray(latencies, dtype=np.float64)
    metrics: dict[str, Any] = {
        "release": "v0.6",
        "service_contract": {
            "api_endpoints": list(API_ENDPOINTS),
            "database_tables": list(DATABASE_TABLES),
            "feature_version": FEATURE_VERSION,
            "model_feature_count": len(GRAPH_FEATURE_NAMES),
            "event_derived_feature_count": len(EVENT_DERIVED_FEATURE_NAMES),
            "historical_feature_count": len(HISTORICAL_FEATURE_NAMES),
            "same_step_history_allowed": False,
        },
        "local_sequential_inference_benchmark": {
            "model_version": scorer.model_version,
            "rows": dataset.rows,
            "includes_local_occlusion_explanations": True,
            "latency_ms": {
                "minimum": float(values.min()),
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "maximum": float(values.max()),
            },
            "rows_per_second": float(dataset.rows / elapsed),
            "scope": "local sequential warm model inference; database and HTTP excluded",
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": sys.platform,
                "machine": platform.machine(),
            },
        },
        "guardrails": {
            "authentication": "constant-time API-key comparison",
            "idempotency": "required key plus canonical request fingerprint",
            "account_storage": "salted SHA-256 pseudonyms only",
            "model_integrity": "production requires pre-deserialisation SHA-256",
            "decline_semantics": "simulation-only recommendation",
        },
    }
    _write_json(root / "reports/v0.6/service_metrics.json", metrics)
    _write_report(root / "docs/SERVICE_REPORT.md", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark and report the SentinelGraph v0.6 service"
    )
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--rows", type=int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metrics = benchmark_service(args.project_root, rows=args.rows)
    benchmark = metrics["local_sequential_inference_benchmark"]
    print(
        "SentinelGraph v0.6 report written: "
        f"p95={benchmark['latency_ms']['p95']:.3f} ms"
    )


if __name__ == "__main__":
    main()
