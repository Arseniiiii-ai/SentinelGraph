"""Construct the exact v0.5 matrix from a versioned online snapshot."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from numpy.typing import NDArray

from sentinelgraph.api.schemas import ScoreRequest, TransactionType
from sentinelgraph.modeling.graph import GRAPH_FEATURE_NAMES


def build_feature_vector(request: ScoreRequest) -> NDArray[np.float32]:
    """Build one model row while keeping all historical state explicit."""
    hour = (request.step - 1) % 24
    angle = 2.0 * math.pi * hour / 24.0
    event_features = {
        "amount": request.amount,
        "log_amount": math.log1p(request.amount),
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
        "type_cash_in": float(request.transaction_type is TransactionType.CASH_IN),
        "type_cash_out": float(request.transaction_type is TransactionType.CASH_OUT),
        "type_debit": float(request.transaction_type is TransactionType.DEBIT),
        "type_payment": float(request.transaction_type is TransactionType.PAYMENT),
        "type_transfer": float(request.transaction_type is TransactionType.TRANSFER),
        "destination_is_merchant": float(
            request.destination_account.startswith("M")
        ),
    }
    values = {**request.historical_features.values, **event_features}
    vector = np.asarray(
        [[values[name] for name in GRAPH_FEATURE_NAMES]],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(vector)):
        raise ValueError("online feature vector contains non-finite values")
    return vector


def request_fingerprint(request: ScoreRequest) -> str:
    """Return a canonical payload hash for idempotency conflict detection."""
    payload = request.model_dump(mode="json")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_account(account_id: str, salt: str) -> str:
    """Pseudonymise an account before durable persistence."""
    return hashlib.sha256(f"{salt}:{account_id}".encode()).hexdigest()
