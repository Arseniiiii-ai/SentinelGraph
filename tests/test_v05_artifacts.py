"""Integration assertions over generated v0.5 risk-engine artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinelgraph.data.provenance import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports" / "v0.5" / "risk_metrics.json"
EXAMPLES_PATH = (
    PROJECT_ROOT / "reports" / "v0.5" / "reason_code_examples.json"
)


def _load(path: Path) -> dict[str, object]:
    payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return payload


@pytest.mark.integration
def test_v05_temporal_governance_windows_do_not_overlap() -> None:
    metrics = _load(METRICS_PATH)
    contract = metrics["temporal_contract"]
    assert isinstance(contract, dict)
    ordered = [
        contract["development"],
        contract["calibration_fit"],
        contract["calibration_selection"],
        contract["policy_selection"],
        contract["future_time_holdout"],
    ]
    for first, second in zip(ordered, ordered[1:], strict=False):
        assert first["maximum_step"] < second["minimum_step"]
    assert ordered[-1]["minimum_step"] == 521


@pytest.mark.integration
def test_v05_calibration_improves_brier_without_losing_ranking() -> None:
    metrics = _load(METRICS_PATH)
    selection = metrics["calibration_selection"]
    assert isinstance(selection, dict)
    selected_name = selection["selected_calibrator"]
    candidates = selection["candidates"]
    raw = selection["raw_behavioural_reference"]
    selected = candidates[selected_name]

    assert selected_name in selection["eligible_candidates"]
    assert selected["brier_score"] < raw["brier_score"]
    assert selected["average_precision"] >= (
        raw["average_precision"] * selection["minimum_ranking_retention"]
    )


@pytest.mark.integration
def test_v05_policy_selection_respects_declared_capacity() -> None:
    metrics = _load(METRICS_PATH)
    policy = metrics["decision_policy"]
    evaluation = metrics["evaluations"]["policy_selection"][
        "policy_backtest"
    ]
    assert policy["review_threshold"] <= policy["decline_threshold"]
    assert evaluation["review_queue_rate"] <= policy["maximum_review_rate"]
    assert evaluation["decline_rate"] <= policy["maximum_decline_rate"]
    assert (
        evaluation["decisions"]["decline"]["precision"]
        >= policy["minimum_decline_precision"]
    )


@pytest.mark.integration
def test_v05_future_decisions_cover_every_row() -> None:
    metrics = _load(METRICS_PATH)
    future = metrics["evaluations"]["future_time_holdout"]["policy_backtest"]
    decisions = future["decisions"]
    assert set(decisions) == {"approve", "review", "decline"}
    assert sum(payload["rows"] for payload in decisions.values()) == future["rows"]
    assert future["review_queue_rows"] == decisions["review"]["rows"]


@pytest.mark.integration
def test_v05_reason_examples_are_deidentified_and_bounded() -> None:
    examples = _load(EXAMPLES_PATH)
    rows = examples["examples"]
    assert isinstance(rows, list)
    assert 0 < len(rows) <= 25
    serialized = json.dumps(examples)
    assert "nameOrig" not in serialized
    assert "nameDest" not in serialized
    for row in rows:
        assert 0 <= row["risk_points"] <= 1_000
        assert row["decision"] in {"review", "decline"}
        assert 1 <= len(row["reason_codes"]) <= 3


@pytest.mark.integration
def test_v05_local_bundle_matches_ignored_manifest_when_present() -> None:
    manifest_path = PROJECT_ROOT / "models" / "v0.5" / "artifact_manifest.json"
    if not manifest_path.exists():
        pytest.skip("risk bundle is generated locally")
    manifest = _load(manifest_path)
    artifact = manifest["risk_bundle"]
    assert isinstance(artifact, dict)
    path = PROJECT_ROOT / str(artifact["path"])
    if not path.exists():
        pytest.skip("risk bundle is generated locally")
    assert path.stat().st_size == artifact["size_bytes"]
    assert sha256_file(path) == artifact["sha256"]
