"""Input data contracts for transaction events."""

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TransactionType(StrEnum):
    """Supported PaySim-style transaction types."""

    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class TransactionEvent(BaseModel):
    """A transaction as it should be known at scoring time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str = Field(min_length=1, max_length=128)
    occurred_at: datetime
    transaction_type: TransactionType
    amount: float = Field(gt=0)
    origin_account_id: str = Field(min_length=1, max_length=128)
    destination_account_id: str = Field(min_length=1, max_length=128)
    origin_balance_before: float = Field(ge=0)
    destination_balance_before: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_transaction(self) -> "TransactionEvent":
        """Reject impossible or leakage-prone transaction inputs."""
        if self.origin_account_id == self.destination_account_id:
            raise ValueError("origin and destination accounts must be different")
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")
        return self

    @classmethod
    def example(cls) -> "TransactionEvent":
        """Return a stable example used by documentation and smoke tests."""
        return cls(
            transaction_id="txn-0001",
            occurred_at=datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc),
            transaction_type=TransactionType.TRANSFER,
            amount=1250.0,
            origin_account_id="acct-origin",
            destination_account_id="acct-destination",
            origin_balance_before=5000.0,
            destination_balance_before=300.0,
        )
