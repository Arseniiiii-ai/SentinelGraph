"""Versioned HTTP and online-feature contracts for SentinelGraph v0.6."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from sentinelgraph.modeling.features import FEATURE_NAMES
from sentinelgraph.modeling.graph import GRAPH_FEATURE_NAMES

FEATURE_VERSION = "v0.6.0"
EVENT_DERIVED_FEATURE_NAMES = FEATURE_NAMES + ("destination_is_merchant",)
HISTORICAL_FEATURE_NAMES = tuple(
    name for name in GRAPH_FEATURE_NAMES if name not in EVENT_DERIVED_FEATURE_NAMES
)
HISTORICAL_FEATURE_SET = frozenset(HISTORICAL_FEATURE_NAMES)

ExternalId = Annotated[str, Field(min_length=1, max_length=128)]
AccountId = Annotated[str, Field(min_length=1, max_length=128)]


class StrictModel(BaseModel):
    """Base contract that rejects silently ignored client fields."""

    model_config = ConfigDict(extra="forbid")


class TransactionType(StrEnum):
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class DecisionName(StrEnum):
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"


class CaseStatus(StrEnum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    CLOSED = "closed"


class CaseDisposition(StrEnum):
    CONFIRMED_FRAUD = "confirmed_fraud"
    LEGITIMATE = "legitimate"
    INCONCLUSIVE = "inconclusive"


class HistoricalFeatureSnapshot(StrictModel):
    """Strictly-prior feature state supplied by the online feature provider."""

    version: Literal["v0.6.0"] = "v0.6.0"
    as_of_step: int = Field(ge=0)
    values: dict[str, float]

    @model_validator(mode="after")
    def validate_feature_set(self) -> "HistoricalFeatureSnapshot":
        supplied = set(self.values)
        missing = sorted(HISTORICAL_FEATURE_SET - supplied)
        unexpected = sorted(supplied - HISTORICAL_FEATURE_SET)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={','.join(missing)}")
            if unexpected:
                details.append(f"unexpected={','.join(unexpected)}")
            raise ValueError("historical feature contract mismatch: " + "; ".join(details))
        non_finite = [
            name for name, value in self.values.items() if not math.isfinite(value)
        ]
        if non_finite:
            raise ValueError(f"non-finite historical feature: {sorted(non_finite)[0]}")
        return self


class ScoreRequest(StrictModel):
    """One transaction plus a feature snapshot built before its time step."""

    external_id: ExternalId
    step: int = Field(ge=1)
    transaction_type: TransactionType
    amount: float = Field(ge=0.0, le=1e15, allow_inf_nan=False)
    origin_account: AccountId
    destination_account: AccountId
    occurred_at: AwareDatetime | None = None
    historical_features: HistoricalFeatureSnapshot

    @model_validator(mode="after")
    def validate_point_in_time(self) -> "ScoreRequest":
        if self.historical_features.as_of_step >= self.step:
            raise ValueError(
                "historical_features.as_of_step must be strictly before step"
            )
        return self


class BatchScoreRequest(StrictModel):
    transactions: list[ScoreRequest] = Field(min_length=1, max_length=1_000)


class ReasonCode(StrictModel):
    code: str
    description: str
    feature: str
    contribution: float


class ScoreResponse(StrictModel):
    prediction_id: str
    transaction_id: str
    external_id: str
    risk_probability: float
    risk_points: int
    decision: DecisionName
    reason_codes: list[ReasonCode]
    component_scores: dict[str, float]
    model_version: str
    policy_version: str
    feature_version: str
    request_id: str
    latency_ms: float
    created_at: datetime
    replayed: bool = False


class BatchScoreResponse(StrictModel):
    predictions: list[ScoreResponse]
    count: int


class CaseSummary(StrictModel):
    case_id: str
    prediction_id: str
    external_id: str
    risk_probability: float
    risk_points: int
    decision: DecisionName
    status: CaseStatus
    priority: str
    assigned_to: str | None
    version: int
    created_at: datetime
    updated_at: datetime


class CaseDetail(CaseSummary):
    transaction: dict[str, Any]
    reason_codes: list[dict[str, Any]]
    component_scores: dict[str, float]
    model_version: str
    policy_version: str
    feature_version: str
    events: list[dict[str, Any]]
    feedback: list[dict[str, Any]]


class CaseListResponse(StrictModel):
    cases: list[CaseSummary]
    count: int
    limit: int
    offset: int


class CaseDecisionRequest(StrictModel):
    status: CaseStatus
    investigator: str = Field(min_length=1, max_length=128)
    expected_version: int = Field(ge=1)
    assigned_to: str | None = Field(default=None, max_length=128)
    disposition: CaseDisposition | None = None
    fraud_confirmed: bool | None = None
    notes: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def validate_closed_case(self) -> "CaseDecisionRequest":
        if self.status is CaseStatus.OPEN:
            raise ValueError("case decisions cannot return a case to open")
        if self.status is CaseStatus.CLOSED and self.disposition is None:
            raise ValueError("closed cases require a disposition")
        if self.status is not CaseStatus.CLOSED and self.disposition is not None:
            raise ValueError("disposition is only accepted when closing a case")
        if self.status is not CaseStatus.CLOSED and self.fraud_confirmed is not None:
            raise ValueError("fraud_confirmed is only accepted when closing a case")
        if (
            self.disposition is CaseDisposition.CONFIRMED_FRAUD
            and self.fraud_confirmed is not True
        ):
            raise ValueError("confirmed_fraud requires fraud_confirmed=true")
        if (
            self.disposition is CaseDisposition.LEGITIMATE
            and self.fraud_confirmed is not False
        ):
            raise ValueError("legitimate requires fraud_confirmed=false")
        if (
            self.disposition is CaseDisposition.INCONCLUSIVE
            and self.fraud_confirmed is not None
        ):
            raise ValueError("inconclusive requires fraud_confirmed=null")
        return self


class HealthResponse(StrictModel):
    status: str
    release: str
    checks: dict[str, str] = Field(default_factory=dict)
