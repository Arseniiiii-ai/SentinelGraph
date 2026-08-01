"""Tests for the versioned point-in-time online feature contract."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from sentinelgraph.api.online_features import build_feature_vector
from sentinelgraph.api.schemas import (
    HISTORICAL_FEATURE_NAMES,
    HistoricalFeatureSnapshot,
    ScoreRequest,
    TransactionType,
)
from sentinelgraph.modeling.graph import GRAPH_FEATURE_NAMES


def historical_values(value: float = 0.0) -> dict[str, float]:
    return {name: value for name in HISTORICAL_FEATURE_NAMES}


def request_payload() -> dict[str, object]:
    return {
        "external_id": "tx-online-1",
        "step": 25,
        "transaction_type": "TRANSFER",
        "amount": 99.0,
        "origin_account": "C100",
        "destination_account": "M200",
        "historical_features": {
            "version": "v0.6.0",
            "as_of_step": 24,
            "values": historical_values(),
        },
    }


def test_online_vector_matches_training_feature_order() -> None:
    request = ScoreRequest.model_validate(request_payload())
    vector = build_feature_vector(request)
    assert vector.shape == (1, len(GRAPH_FEATURE_NAMES))
    by_name = dict(zip(GRAPH_FEATURE_NAMES, vector[0], strict=True))
    assert by_name["amount"] == pytest.approx(99.0)
    assert by_name["log_amount"] == pytest.approx(math.log1p(99.0))
    assert by_name["hour_sin"] == pytest.approx(0.0, abs=1e-7)
    assert by_name["hour_cos"] == pytest.approx(1.0)
    assert by_name["type_transfer"] == 1.0
    assert by_name["type_cash_out"] == 0.0
    assert by_name["destination_is_merchant"] == 1.0
    assert np.isfinite(vector).all()


def test_snapshot_requires_exact_feature_set() -> None:
    values = historical_values()
    values.pop(HISTORICAL_FEATURE_NAMES[0])
    with pytest.raises(ValidationError, match="contract mismatch"):
        HistoricalFeatureSnapshot(
            version="v0.6.0",
            as_of_step=1,
            values=values,
        )


def test_snapshot_rejects_non_finite_values() -> None:
    values = historical_values()
    values[HISTORICAL_FEATURE_NAMES[0]] = float("nan")
    with pytest.raises(ValidationError, match="non-finite"):
        HistoricalFeatureSnapshot(
            version="v0.6.0",
            as_of_step=1,
            values=values,
        )


def test_score_request_rejects_same_step_history() -> None:
    payload = request_payload()
    historical = cast(dict[str, object], payload["historical_features"]).copy()
    historical["as_of_step"] = 25
    payload["historical_features"] = historical
    with pytest.raises(ValidationError, match="strictly before"):
        ScoreRequest.model_validate(payload)


def test_transaction_type_is_closed_enum() -> None:
    assert TransactionType.TRANSFER.value == "TRANSFER"
    payload = request_payload()
    payload["transaction_type"] = "WIRE"
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate(payload)
