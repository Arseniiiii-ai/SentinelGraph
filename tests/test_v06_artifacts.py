"""Regression checks for tracked v0.6 service evidence and contracts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v06_release_artifacts_exist() -> None:
    expected = (
        "alembic.ini",
        "migrations/versions/0001_v06_initial_schema.py",
        "docs/API.md",
        "docs/ARCHITECTURE_V06.md",
        "docs/INTERVIEW_V06.md",
        "docs/SERVICE_REPORT.md",
        "docs/releases/v0.6.md",
        "reports/v0.6/service_metrics.json",
    )
    for relative_path in expected:
        assert (ROOT / relative_path).is_file(), relative_path


def test_service_metrics_describe_exact_contract() -> None:
    metrics = json.loads(
        (ROOT / "reports/v0.6/service_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["release"] == "v0.6"
    contract = metrics["service_contract"]
    assert contract["feature_version"] == "v0.6.0"
    assert contract["model_feature_count"] == 60
    assert contract["event_derived_feature_count"] == 10
    assert contract["historical_feature_count"] == 50
    assert contract["same_step_history_allowed"] is False
    assert len(contract["api_endpoints"]) == 8
    assert len(contract["database_tables"]) == 6
    benchmark = metrics["local_sequential_inference_benchmark"]
    assert benchmark["model_version"] == "v0.5"
    assert benchmark["rows"] == 25
    assert benchmark["includes_local_occlusion_explanations"] is True
    assert 0.0 < benchmark["latency_ms"]["p50"] <= benchmark["latency_ms"]["p95"]
    assert benchmark["latency_ms"]["p95"] < 100.0
    assert benchmark["rows_per_second"] > 10.0


def test_migration_contains_auditable_schema_and_model_registry() -> None:
    migration = (
        ROOT / "migrations/versions/0001_v06_initial_schema.py"
    ).read_text(encoding="utf-8")
    for table in (
        "transactions",
        "predictions",
        "cases",
        "case_events",
        "investigator_feedback",
        "model_versions",
    ):
        assert f'"{table}"' in migration
    assert "fk_predictions_model_version" in migration
    assert "JSONB" in migration


def test_versions_and_interview_bank_are_current() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package = (ROOT / "src/sentinelgraph/__init__.py").read_text(
        encoding="utf-8"
    )
    interview = (ROOT / "docs/INTERVIEW_V06.md").read_text(encoding="utf-8")
    assert 'version = "0.6.0.dev0"' in pyproject
    assert '__version__ = "0.6.0.dev0"' in package
    assert "70. **" in interview
