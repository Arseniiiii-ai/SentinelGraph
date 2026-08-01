"""Transactional scoring, idempotency, and investigator case workflow."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import Select, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from sentinelgraph.api.inference import ScoringEngine
from sentinelgraph.api.models import (
    CaseEventRecord,
    CaseRecord,
    InvestigatorFeedbackRecord,
    PredictionRecord,
    TransactionRecord,
)
from sentinelgraph.api.online_features import (
    build_feature_vector,
    hash_account,
    request_fingerprint,
)
from sentinelgraph.api.schemas import (
    FEATURE_VERSION,
    CaseDecisionRequest,
    CaseDetail,
    CaseListResponse,
    CaseStatus,
    CaseSummary,
    DecisionName,
    ReasonCode,
    ScoreRequest,
    ScoreResponse,
)


class ServiceError(RuntimeError):
    pass


class NotFoundError(ServiceError):
    pass


class ConflictError(ServiceError):
    pass


def _new_id() -> str:
    return str(uuid.uuid4())


def batch_item_idempotency_key(batch_key: str, external_id: str) -> str:
    """Derive a bounded, deterministic key for every item in a batch."""
    return hashlib.sha256(f"{batch_key}:{external_id}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class _CaseRow:
    case: CaseRecord
    prediction: PredictionRecord
    transaction: TransactionRecord


class ScoringService:
    """Application service that keeps scoring and audit writes atomic."""

    def __init__(self, scorer: ScoringEngine, account_hash_salt: str) -> None:
        self.scorer = scorer
        self.account_hash_salt = account_hash_salt

    @staticmethod
    def _prediction_for_transaction(
        session: Session, transaction_id: str
    ) -> PredictionRecord:
        prediction = session.scalar(
            select(PredictionRecord).where(
                PredictionRecord.transaction_id == transaction_id
            )
        )
        if prediction is None:
            raise RuntimeError("stored transaction has no prediction")
        return prediction

    @staticmethod
    def _response(
        transaction: TransactionRecord,
        prediction: PredictionRecord,
        *,
        replayed: bool,
    ) -> ScoreResponse:
        return ScoreResponse(
            prediction_id=prediction.id,
            transaction_id=transaction.id,
            external_id=transaction.external_id,
            risk_probability=prediction.risk_probability,
            risk_points=prediction.risk_points,
            decision=DecisionName(prediction.decision),
            reason_codes=[
                ReasonCode.model_validate(item) for item in prediction.reason_codes
            ],
            component_scores=prediction.component_scores,
            model_version=prediction.model_version,
            policy_version=prediction.policy_version,
            feature_version=prediction.feature_version,
            request_id=prediction.request_id,
            latency_ms=prediction.latency_ms,
            created_at=prediction.created_at,
            replayed=replayed,
        )

    def score(
        self,
        session: Session,
        request: ScoreRequest,
        *,
        idempotency_key: str,
        request_id: str,
    ) -> ScoreResponse:
        """Return a replay or stage one new transaction/prediction/case write."""
        fingerprint = request_fingerprint(request)
        existing = session.scalar(
            select(TransactionRecord).where(
                (TransactionRecord.external_id == request.external_id)
                | (TransactionRecord.idempotency_key == idempotency_key)
            )
        )
        if existing is not None:
            if existing.request_hash != fingerprint:
                raise ConflictError(
                    "external_id or Idempotency-Key was reused with a different payload"
                )
            prediction = self._prediction_for_transaction(session, existing.id)
            return self._response(existing, prediction, replayed=True)

        features = build_feature_vector(request)
        started = time.perf_counter()
        result = self.scorer.score(features)
        latency_ms = (time.perf_counter() - started) * 1_000.0

        transaction = TransactionRecord(
            id=_new_id(),
            external_id=request.external_id,
            idempotency_key=idempotency_key,
            request_hash=fingerprint,
            step=request.step,
            transaction_type=request.transaction_type.value,
            amount=request.amount,
            origin_account_hash=hash_account(
                request.origin_account, self.account_hash_salt
            ),
            destination_account_hash=hash_account(
                request.destination_account, self.account_hash_salt
            ),
            feature_version=FEATURE_VERSION,
            feature_snapshot=request.historical_features.model_dump(mode="json"),
            occurred_at=request.occurred_at,
        )
        prediction = PredictionRecord(
            id=_new_id(),
            transaction_id=transaction.id,
            risk_probability=result.risk_probability,
            risk_points=result.risk_points,
            decision=result.decision,
            reason_codes=result.reason_codes,
            component_scores=result.component_scores,
            model_version=result.model_version,
            policy_version=result.policy_version,
            feature_version=FEATURE_VERSION,
            request_id=request_id,
            latency_ms=latency_ms,
        )
        session.add(transaction)
        session.flush()
        session.add(prediction)
        session.flush()
        if result.decision != DecisionName.APPROVE.value:
            case = CaseRecord(
                id=_new_id(),
                prediction_id=prediction.id,
                status=CaseStatus.OPEN.value,
                priority=(
                    "critical"
                    if result.decision == DecisionName.DECLINE.value
                    else "high"
                ),
            )
            event = CaseEventRecord(
                id=_new_id(),
                case_id=case.id,
                event_type="case_created",
                actor="sentinelgraph-system",
                details={
                    "decision": result.decision,
                    "risk_points": result.risk_points,
                    "automated_action": False,
                },
            )
            session.add(case)
            session.flush()
            session.add(event)
        session.flush()
        return self._response(transaction, prediction, replayed=False)

    @staticmethod
    def _case_select() -> Select[tuple[CaseRecord, PredictionRecord, TransactionRecord]]:
        return (
            select(CaseRecord, PredictionRecord, TransactionRecord)
            .join(PredictionRecord, CaseRecord.prediction_id == PredictionRecord.id)
            .join(
                TransactionRecord,
                PredictionRecord.transaction_id == TransactionRecord.id,
            )
        )

    @staticmethod
    def _summary(row: _CaseRow) -> CaseSummary:
        return CaseSummary(
            case_id=row.case.id,
            prediction_id=row.prediction.id,
            external_id=row.transaction.external_id,
            risk_probability=row.prediction.risk_probability,
            risk_points=row.prediction.risk_points,
            decision=DecisionName(row.prediction.decision),
            status=CaseStatus(row.case.status),
            priority=row.case.priority,
            assigned_to=row.case.assigned_to,
            version=row.case.version,
            created_at=row.case.created_at,
            updated_at=row.case.updated_at,
        )

    def list_cases(
        self,
        session: Session,
        *,
        status: CaseStatus | None,
        decision: DecisionName | None,
        limit: int,
        offset: int,
    ) -> CaseListResponse:
        statement = self._case_select()
        count_statement = (
            select(func.count())
            .select_from(CaseRecord)
            .join(PredictionRecord, CaseRecord.prediction_id == PredictionRecord.id)
        )
        if status is not None:
            statement = statement.where(CaseRecord.status == status.value)
            count_statement = count_statement.where(
                CaseRecord.status == status.value
            )
        if decision is not None:
            statement = statement.where(PredictionRecord.decision == decision.value)
            count_statement = count_statement.where(
                PredictionRecord.decision == decision.value
            )
        statement = statement.order_by(
            PredictionRecord.risk_points.desc(), CaseRecord.created_at.asc()
        ).limit(limit).offset(offset)
        rows = [
            _CaseRow(case=row[0], prediction=row[1], transaction=row[2])
            for row in session.execute(statement).all()
        ]
        return CaseListResponse(
            cases=[self._summary(row) for row in rows],
            count=int(session.scalar(count_statement) or 0),
            limit=limit,
            offset=offset,
        )

    def _case_row(self, session: Session, case_id: str) -> _CaseRow:
        row = session.execute(
            self._case_select().where(CaseRecord.id == case_id)
        ).one_or_none()
        if row is None:
            raise NotFoundError(f"case not found: {case_id}")
        return _CaseRow(case=row[0], prediction=row[1], transaction=row[2])

    def get_case(self, session: Session, case_id: str) -> CaseDetail:
        row = self._case_row(session, case_id)
        events = session.scalars(
            select(CaseEventRecord)
            .where(CaseEventRecord.case_id == case_id)
            .order_by(CaseEventRecord.created_at.asc())
        ).all()
        feedback = session.scalars(
            select(InvestigatorFeedbackRecord)
            .where(InvestigatorFeedbackRecord.case_id == case_id)
            .order_by(InvestigatorFeedbackRecord.created_at.asc())
        ).all()
        summary = self._summary(row)
        return CaseDetail(
            **summary.model_dump(),
            transaction={
                "step": row.transaction.step,
                "transaction_type": row.transaction.transaction_type,
                "amount": row.transaction.amount,
                "occurred_at": row.transaction.occurred_at,
                "origin_account_hash": row.transaction.origin_account_hash,
                "destination_account_hash": row.transaction.destination_account_hash,
                "received_at": row.transaction.received_at,
            },
            reason_codes=row.prediction.reason_codes,
            component_scores=row.prediction.component_scores,
            model_version=row.prediction.model_version,
            policy_version=row.prediction.policy_version,
            feature_version=row.prediction.feature_version,
            events=[
                {
                    "event_type": event.event_type,
                    "actor": event.actor,
                    "details": event.details,
                    "created_at": event.created_at,
                }
                for event in events
            ],
            feedback=[
                {
                    "disposition": item.disposition,
                    "fraud_confirmed": item.fraud_confirmed,
                    "notes": item.notes,
                    "investigator": item.investigator,
                    "created_at": item.created_at,
                }
                for item in feedback
            ],
        )

    def update_case(
        self,
        session: Session,
        case_id: str,
        request: CaseDecisionRequest,
    ) -> CaseDetail:
        current = self._case_row(session, case_id)
        if current.case.status == CaseStatus.CLOSED.value:
            raise ConflictError("closed cases are immutable")
        values: dict[str, Any] = {
            "status": request.status.value,
            "version": current.case.version + 1,
        }
        if request.assigned_to is not None:
            values["assigned_to"] = request.assigned_to
        if request.disposition is not None:
            values["resolution"] = request.disposition.value
        outcome = cast(
            CursorResult[Any],
            session.execute(
                update(CaseRecord)
                .where(
                    CaseRecord.id == case_id,
                    CaseRecord.version == request.expected_version,
                )
                .values(**values)
            ),
        )
        if outcome.rowcount != 1:
            raise ConflictError(
                "case changed after it was loaded; refresh and retry with its new version"
            )
        session.add(
            CaseEventRecord(
                id=_new_id(),
                case_id=case_id,
                event_type="case_updated",
                actor=request.investigator,
                details={
                    "previous_status": current.case.status,
                    "new_status": request.status.value,
                    "assigned_to": request.assigned_to,
                    "disposition": (
                        request.disposition.value
                        if request.disposition is not None
                        else None
                    ),
                    "notes": request.notes,
                },
            )
        )
        if request.disposition is not None:
            session.add(
                InvestigatorFeedbackRecord(
                    id=_new_id(),
                    case_id=case_id,
                    disposition=request.disposition.value,
                    fraud_confirmed=request.fraud_confirmed,
                    notes=request.notes,
                    investigator=request.investigator,
                )
            )
        session.flush()
        session.expire_all()
        return self.get_case(session, case_id)
