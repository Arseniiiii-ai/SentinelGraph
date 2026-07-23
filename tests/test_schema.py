"""Tests for the initial transaction contract."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from sentinelgraph.data.schema import TransactionEvent, TransactionType


def test_example_transaction_is_valid() -> None:
    event = TransactionEvent.example()
    assert event.amount == 1250.0
    assert event.transaction_type is TransactionType.TRANSFER


def test_negative_amount_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            transaction_id="txn-negative",
            occurred_at=datetime.now(timezone.utc),
            transaction_type=TransactionType.PAYMENT,
            amount=-1.0,
            origin_account_id="origin",
            destination_account_id="destination",
            origin_balance_before=10.0,
            destination_balance_before=20.0,
        )


def test_same_origin_and_destination_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            transaction_id="txn-self",
            occurred_at=datetime.now(timezone.utc),
            transaction_type=TransactionType.TRANSFER,
            amount=10.0,
            origin_account_id="same-account",
            destination_account_id="same-account",
            origin_balance_before=100.0,
            destination_balance_before=100.0,
        )


def test_timezone_is_required() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            transaction_id="txn-no-timezone",
            occurred_at=datetime(2026, 7, 23, 12, 0),
            transaction_type=TransactionType.CASH_OUT,
            amount=10.0,
            origin_account_id="origin",
            destination_account_id="destination",
            origin_balance_before=100.0,
            destination_balance_before=100.0,
        )
