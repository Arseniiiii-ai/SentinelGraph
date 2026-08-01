"""Local artifact parity from one offline row through the public HTTP API."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from sentinelgraph.api.app import create_app
from sentinelgraph.api.config import Settings
from sentinelgraph.api.database import Database
from sentinelgraph.api.inference import RiskScorer
from sentinelgraph.api.models import ModelVersionRecord
from sentinelgraph.api.online_features import build_feature_vector
from sentinelgraph.api.schemas import HISTORICAL_FEATURE_NAMES, ScoreRequest
from sentinelgraph.modeling.graph import GRAPH_FEATURE_NAMES, load_graph_matrix


@pytest.mark.integration
def test_real_bundle_http_and_direct_inference_are_equal(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle_path = root / "models/v0.5/risk_bundle.joblib"
    graph_path = root / "data/processed/graph_features.parquet"
    component_path = root / "data/processed/graph_components.npz"
    train_path = root / "data/processed/train.parquet"
    required = (bundle_path, graph_path, component_path, train_path)
    if not all(path.exists() for path in required):
        pytest.skip("local model and processed feature artifacts are required")

    matrix = load_graph_matrix(
        graph_path,
        component_path,
        where_sql="source_row_number = 1",
    )
    by_name = dict(zip(GRAPH_FEATURE_NAMES, matrix.features[0], strict=True))
    connection = duckdb.connect()
    try:
        source = connection.execute(
            "SELECT step, type, amount, nameOrig, nameDest "
            "FROM read_parquet(?) WHERE source_row_number = 1",
            [str(train_path)],
        ).fetchone()
    finally:
        connection.close()
    assert source is not None
    payload = {
        "external_id": "real-parity-row-1",
        "step": int(source[0]),
        "transaction_type": str(source[1]),
        "amount": float(source[2]),
        "origin_account": str(source[3]),
        "destination_account": str(source[4]),
        "historical_features": {
            "version": "v0.6.0",
            "as_of_step": max(0, int(source[0]) - 1),
            "values": {
                name: float(by_name[name]) for name in HISTORICAL_FEATURE_NAMES
            },
        },
    }
    score_request = ScoreRequest.model_validate(payload)
    scorer = RiskScorer(bundle_path)
    expected = scorer.score(build_feature_vector(score_request))

    database = Database(f"sqlite:///{tmp_path / 'parity.db'}")
    database.create_schema_for_tests()
    with database.session() as session:
        session.add(
            ModelVersionRecord(
                version="v0.5",
                feature_version="v0.6.0",
                policy_version="v0.5-policy",
                artifact_path=str(bundle_path),
                active=True,
            )
        )
        session.commit()
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'parity.db'}",
        api_key="real-parity-api-key-long",
        account_hash_salt="real-parity-account-salt-long",
        model_bundle_path=bundle_path,
    )
    app = create_app(settings, database=database, scorer=scorer)
    with TestClient(app) as client:
        response = client.post(
            "/v1/score",
            json=payload,
            headers={
                "X-API-Key": settings.api_key,
                "Idempotency-Key": "real-parity-idempotency",
            },
        )
    assert response.status_code == 201, response.text
    actual = response.json()
    assert actual["risk_probability"] == pytest.approx(
        expected.risk_probability, abs=1e-12
    )
    assert actual["risk_points"] == expected.risk_points
    assert actual["decision"] == expected.decision
    assert [item["code"] for item in actual["reason_codes"]] == [
        item["code"] for item in expected.reason_codes
    ]
    assert [item["contribution"] for item in actual["reason_codes"]] == pytest.approx(
        [item["contribution"] for item in expected.reason_codes], abs=1e-12
    )
