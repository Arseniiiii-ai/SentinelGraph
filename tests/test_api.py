"""End-to-end HTTP contract tests with an isolated SQL database."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from sqlalchemy import select

from sentinelgraph.api.app import create_app
from sentinelgraph.api.config import Settings
from sentinelgraph.api.database import Database
from sentinelgraph.api.inference import InferenceResult
from sentinelgraph.api.models import ModelVersionRecord, TransactionRecord
from sentinelgraph.api.schemas import HISTORICAL_FEATURE_NAMES

API_KEY = "test-api-key-long-enough"


class FakeScorer:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def ready(self) -> bool:
        return True

    @property
    def model_version(self) -> str:
        return "test-model"

    def score(self, features: NDArray[np.float32]) -> InferenceResult:
        self.calls += 1
        amount = float(features[0, 0])
        risk = 0.91 if amount >= 100.0 else 0.01
        decision = "review" if amount >= 100.0 else "approve"
        return InferenceResult(
            risk_probability=risk,
            risk_points=round(risk * 1_000),
            decision=decision,
            reason_codes=(
                [
                    {
                        "code": "HIGH_TRANSACTION_AMOUNT",
                        "description": "Transaction amount materially increased risk.",
                        "feature": "amount",
                        "contribution": 0.4,
                    }
                ]
                if decision == "review"
                else []
            ),
            component_scores={"behavioural_probability": risk},
            model_version="test-model",
            policy_version="test-policy",
        )


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[TestClient, Database, FakeScorer]]:
    database = Database(f"sqlite:///{tmp_path / 'test.db'}")
    database.create_schema_for_tests()
    with database.session() as session:
        session.add(
            ModelVersionRecord(
                version="test-model",
                feature_version="v0.6.0",
                policy_version="test-policy",
                artifact_path="test-double",
                active=True,
            )
        )
        session.commit()
    scorer = FakeScorer()
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        api_key=API_KEY,
        account_hash_salt="test-account-salt-long-enough",
        model_bundle_path=tmp_path / "unused.joblib",
        maximum_batch_size=2,
    )
    with TestClient(
        create_app(settings, database=database, scorer=scorer)
    ) as client:
        yield client, database, scorer


def score_payload(external_id: str = "tx-100", amount: float = 500.0) -> dict[str, object]:
    return {
        "external_id": external_id,
        "step": 10,
        "transaction_type": "TRANSFER",
        "amount": amount,
        "origin_account": "C-RAW-ORIGIN",
        "destination_account": "C-RAW-DESTINATION",
        "historical_features": {
            "version": "v0.6.0",
            "as_of_step": 9,
            "values": {name: 0.0 for name in HISTORICAL_FEATURE_NAMES},
        },
    }


def auth_headers(idempotency_key: str = "idem-1") -> dict[str, str]:
    return {"X-API-Key": API_KEY, "Idempotency-Key": idempotency_key}


def test_health_and_dashboard(api: tuple[TestClient, Database, FakeScorer]) -> None:
    client, _, _ = api
    assert client.get("/health/live").json()["status"] == "ok"
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"model": "ok", "database": "ok"}
    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "Investigator Console" in dashboard.text


def test_scoring_requires_authentication(
    api: tuple[TestClient, Database, FakeScorer],
) -> None:
    client, _, _ = api
    response = client.post(
        "/v1/score",
        json=score_payload(),
        headers={"Idempotency-Key": "idem"},
    )
    assert response.status_code == 401


def test_oversized_body_is_rejected_before_parsing(
    api: tuple[TestClient, Database, FakeScorer],
) -> None:
    client, _, scorer = api
    response = client.post(
        "/v1/score",
        content=b"{}",
        headers={
            **auth_headers(),
            "Content-Length": "3000000",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 413
    assert scorer.calls == 0


def test_chunked_body_limit_does_not_trust_content_length(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'body-limit.db'}")
    scorer = FakeScorer()
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'body-limit.db'}",
        api_key=API_KEY,
        account_hash_salt="test-account-salt-long-enough",
        model_bundle_path=tmp_path / "unused.joblib",
        maximum_request_bytes=16,
    )
    with TestClient(
        create_app(settings, database=database, scorer=scorer)
    ) as client:
        response = client.post(
            "/v1/score",
            content=iter((b"1234567890", b"abcdefghij")),
            headers={
                **auth_headers(),
                "Content-Type": "application/json",
                "Transfer-Encoding": "chunked",
            },
        )
    assert response.status_code == 413
    assert scorer.calls == 0


def test_score_is_idempotent_and_pseudonymises_accounts(
    api: tuple[TestClient, Database, FakeScorer],
) -> None:
    client, database, scorer = api
    first = client.post(
        "/v1/score", json=score_payload(), headers=auth_headers()
    )
    assert first.status_code == 201, first.text
    assert first.json()["decision"] == "review"
    assert first.json()["replayed"] is False
    second = client.post(
        "/v1/score", json=score_payload(), headers=auth_headers()
    )
    assert second.status_code == 201
    assert second.json()["prediction_id"] == first.json()["prediction_id"]
    assert second.json()["replayed"] is True
    assert scorer.calls == 1

    with database.session() as session:
        transaction = session.scalar(select(TransactionRecord))
        assert transaction is not None
        assert transaction.origin_account_hash != "C-RAW-ORIGIN"
        assert transaction.destination_account_hash != "C-RAW-DESTINATION"
        assert "C-RAW" not in str(transaction.feature_snapshot)


def test_idempotency_conflict_is_rejected(
    api: tuple[TestClient, Database, FakeScorer],
) -> None:
    client, _, _ = api
    assert client.post(
        "/v1/score", json=score_payload(), headers=auth_headers()
    ).status_code == 201
    changed = score_payload(amount=600.0)
    conflict = client.post(
        "/v1/score", json=changed, headers=auth_headers()
    )
    assert conflict.status_code == 409


def test_crossed_external_id_and_idempotency_key_are_rejected(
    api: tuple[TestClient, Database, FakeScorer],
) -> None:
    client, _, _ = api
    first_payload = score_payload(external_id="tx-first", amount=500.0)
    second_payload = score_payload(external_id="tx-second", amount=600.0)
    assert client.post(
        "/v1/score", json=first_payload, headers=auth_headers("idem-first")
    ).status_code == 201
    assert client.post(
        "/v1/score", json=second_payload, headers=auth_headers("idem-second")
    ).status_code == 201

    crossed = client.post(
        "/v1/score", json=first_payload, headers=auth_headers("idem-second")
    )
    assert crossed.status_code == 409
    assert "different transactions" in crossed.json()["detail"]


def test_review_case_can_be_closed_with_feedback(
    api: tuple[TestClient, Database, FakeScorer],
) -> None:
    client, _, _ = api
    prediction = client.post(
        "/v1/score", json=score_payload(), headers=auth_headers()
    )
    assert prediction.status_code == 201
    queue = client.get("/v1/cases", headers={"X-API-Key": API_KEY})
    assert queue.status_code == 200
    case = queue.json()["cases"][0]
    case_id = case["case_id"]
    detail = client.get(
        f"/v1/cases/{case_id}", headers={"X-API-Key": API_KEY}
    )
    assert detail.json()["reason_codes"][0]["code"] == "HIGH_TRANSACTION_AMOUNT"

    decision_payload = {
        "status": "closed",
        "investigator": "analyst@example.test",
        "expected_version": 1,
        "assigned_to": "analyst@example.test",
        "disposition": "confirmed_fraud",
        "fraud_confirmed": True,
        "notes": "Confirmed against independent evidence.",
    }
    closed = client.post(
        f"/v1/cases/{case_id}/decision",
        json=decision_payload,
        headers={"X-API-Key": API_KEY},
    )
    assert closed.status_code == 200, closed.text
    assert closed.json()["status"] == "closed"
    assert closed.json()["version"] == 2
    assert closed.json()["feedback"][0]["fraud_confirmed"] is True

    stale = client.post(
        f"/v1/cases/{case_id}/decision",
        json=decision_payload,
        headers={"X-API-Key": API_KEY},
    )
    assert stale.status_code == 409

    immutable = dict(decision_payload)
    immutable["expected_version"] = 2
    immutable_case = client.post(
        f"/v1/cases/{case_id}/decision",
        json=immutable,
        headers={"X-API-Key": API_KEY},
    )
    assert immutable_case.status_code == 409


def test_case_decision_cannot_return_to_open(
    api: tuple[TestClient, Database, FakeScorer],
) -> None:
    client, _, _ = api
    prediction = client.post(
        "/v1/score", json=score_payload(), headers=auth_headers()
    )
    case_id = client.get(
        "/v1/cases", headers={"X-API-Key": API_KEY}
    ).json()["cases"][0]["case_id"]
    response = client.post(
        f"/v1/cases/{case_id}/decision",
        json={
            "status": "open",
            "investigator": "analyst@example.test",
            "expected_version": 1,
        },
        headers={"X-API-Key": API_KEY},
    )
    assert prediction.status_code == 201
    assert response.status_code == 422


def test_approve_does_not_create_case(
    api: tuple[TestClient, Database, FakeScorer],
) -> None:
    client, _, _ = api
    response = client.post(
        "/v1/score",
        json=score_payload(external_id="tx-low", amount=10.0),
        headers=auth_headers("idem-low"),
    )
    assert response.json()["decision"] == "approve"
    queue = client.get("/v1/cases", headers={"X-API-Key": API_KEY})
    assert queue.json()["count"] == 0


def test_case_count_is_total_not_page_length(
    api: tuple[TestClient, Database, FakeScorer],
) -> None:
    client, _, _ = api
    for index in range(2):
        response = client.post(
            "/v1/score",
            json=score_payload(external_id=f"tx-page-{index}"),
            headers=auth_headers(f"idem-page-{index}"),
        )
        assert response.status_code == 201
    queue = client.get(
        "/v1/cases?limit=1", headers={"X-API-Key": API_KEY}
    ).json()
    assert queue["count"] == 2
    assert len(queue["cases"]) == 1


def test_batch_limit_and_duplicate_ids(
    api: tuple[TestClient, Database, FakeScorer],
) -> None:
    client, _, _ = api
    headers = auth_headers("batch-1")
    oversized = client.post(
        "/v1/score/batch",
        json={
            "transactions": [
                score_payload(f"tx-{index}", 10.0) for index in range(3)
            ]
        },
        headers=headers,
    )
    assert oversized.status_code == 413
    duplicate = client.post(
        "/v1/score/batch",
        json={"transactions": [score_payload("same"), score_payload("same")]},
        headers=headers,
    )
    assert duplicate.status_code == 422
