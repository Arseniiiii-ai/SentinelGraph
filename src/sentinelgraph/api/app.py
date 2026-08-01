"""FastAPI application factory and HTTP boundary for SentinelGraph v0.6."""

from __future__ import annotations

import hmac
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.responses import Response

from sentinelgraph.api.config import Settings
from sentinelgraph.api.database import Database
from sentinelgraph.api.inference import RiskScorer, ScoringEngine
from sentinelgraph.api.middleware import RequestBodyLimitMiddleware
from sentinelgraph.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    CaseDecisionRequest,
    CaseDetail,
    CaseListResponse,
    CaseStatus,
    DecisionName,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from sentinelgraph.api.service import (
    ConflictError,
    NotFoundError,
    ScoringService,
    batch_item_idempotency_key,
)

RELEASE = "v0.6"


def create_app(
    settings: Settings | None = None,
    *,
    database: Database | None = None,
    scorer: ScoringEngine | None = None,
) -> FastAPI:
    """Build an independently testable application with injectable adapters."""
    runtime_settings = settings or Settings.from_env()
    runtime_database = database or Database(runtime_settings.database_url)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        active_scorer = scorer or RiskScorer(
            runtime_settings.model_bundle_path,
            expected_sha256=runtime_settings.model_sha256,
        )
        application.state.scorer = active_scorer
        application.state.service = ScoringService(
            active_scorer,
            runtime_settings.account_hash_salt,
        )
        yield
        runtime_database.dispose()

    application = FastAPI(
        title="SentinelGraph Risk and Investigation API",
        version="0.6.0",
        description=(
            "Calibrated fraud-risk scoring with durable, human-in-the-loop cases. "
            "Decline outputs are simulation-only recommendations."
        ),
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.database = runtime_database
    application.state.scorer = scorer
    application.state.service = None
    application.add_middleware(
        RequestBodyLimitMiddleware,
        maximum_bytes=runtime_settings.maximum_request_bytes,
    )

    package_dir = Path(__file__).resolve().parent
    application.mount(
        "/static",
        StaticFiles(directory=package_dir / "static"),
        name="static",
    )
    templates = Jinja2Templates(directory=package_dir / "templates")

    @application.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        raw_request_id = request.headers.get("X-Request-ID", "")
        request_id = (
            raw_request_id
            if raw_request_id and len(raw_request_id) <= 64
            else str(uuid.uuid4())
        )
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = (
            f"{(time.perf_counter() - started) * 1_000.0:.3f}"
        )
        return response

    def session_dependency() -> Iterator[Session]:
        yield from runtime_database.session_dependency()

    def authenticated(
        x_api_key: str = Header(default="", alias="X-API-Key"),
    ) -> None:
        if not hmac.compare_digest(x_api_key, runtime_settings.api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    def service(request: Request) -> ScoringService:
        active = request.app.state.service
        if not isinstance(active, ScoringService):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="scoring service is not ready",
            )
        return active

    def commit_or_conflict(session: Session) -> None:
        try:
            session.commit()
        except IntegrityError as error:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="concurrent idempotency or uniqueness conflict",
            ) from error

    @application.get(
        "/health/live",
        response_model=HealthResponse,
        tags=["health"],
    )
    def live() -> HealthResponse:
        return HealthResponse(status="ok", release=RELEASE)

    @application.get(
        "/health/ready",
        response_model=HealthResponse,
        tags=["health"],
    )
    def ready(request: Request) -> HealthResponse:
        checks: dict[str, str] = {}
        active_scorer = request.app.state.scorer
        checks["model"] = (
            "ok"
            if active_scorer is not None and active_scorer.ready
            else "unavailable"
        )
        try:
            runtime_database.ping()
            checks["database"] = "ok"
        except Exception:
            checks["database"] = "unavailable"
        if set(checks.values()) != {"ok"}:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"status": "not_ready", "release": RELEASE, "checks": checks},
            )
        return HealthResponse(status="ready", release=RELEASE, checks=checks)

    @application.post(
        "/v1/score",
        response_model=ScoreResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["scoring"],
        dependencies=[Depends(authenticated)],
    )
    def score_transaction(
        payload: ScoreRequest,
        request: Request,
        idempotency_key: str = Header(
            min_length=1, max_length=128, alias="Idempotency-Key"
        ),
        session: Session = Depends(session_dependency),
        scoring_service: ScoringService = Depends(service),
    ) -> ScoreResponse:
        try:
            response = scoring_service.score(
                session,
                payload,
                idempotency_key=idempotency_key,
                request_id=request.state.request_id,
            )
        except ConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except IntegrityError as error:
            session.rollback()
            raise HTTPException(
                status_code=409,
                detail="concurrent idempotency or uniqueness conflict",
            ) from error
        commit_or_conflict(session)
        return response

    @application.post(
        "/v1/score/batch",
        response_model=BatchScoreResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["scoring"],
        dependencies=[Depends(authenticated)],
    )
    def score_batch(
        payload: BatchScoreRequest,
        request: Request,
        idempotency_key: str = Header(
            min_length=1, max_length=128, alias="Idempotency-Key"
        ),
        session: Session = Depends(session_dependency),
        scoring_service: ScoringService = Depends(service),
    ) -> BatchScoreResponse:
        if len(payload.transactions) > runtime_settings.maximum_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    "batch exceeds configured limit of "
                    f"{runtime_settings.maximum_batch_size}"
                ),
            )
        external_ids = [item.external_id for item in payload.transactions]
        if len(set(external_ids)) != len(external_ids):
            raise HTTPException(
                status_code=422,
                detail="external_id values must be unique within a batch",
            )
        try:
            predictions = [
                scoring_service.score(
                    session,
                    item,
                    idempotency_key=batch_item_idempotency_key(
                        idempotency_key, item.external_id
                    ),
                    request_id=request.state.request_id,
                )
                for item in payload.transactions
            ]
        except ConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except IntegrityError as error:
            session.rollback()
            raise HTTPException(
                status_code=409,
                detail="concurrent idempotency or uniqueness conflict",
            ) from error
        commit_or_conflict(session)
        return BatchScoreResponse(
            predictions=predictions,
            count=len(predictions),
        )

    @application.get(
        "/v1/cases",
        response_model=CaseListResponse,
        tags=["investigation"],
        dependencies=[Depends(authenticated)],
    )
    def list_cases(
        case_status: CaseStatus | None = Query(default=None, alias="status"),
        decision: DecisionName | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        session: Session = Depends(session_dependency),
        scoring_service: ScoringService = Depends(service),
    ) -> CaseListResponse:
        return scoring_service.list_cases(
            session,
            status=case_status,
            decision=decision,
            limit=limit,
            offset=offset,
        )

    @application.get(
        "/v1/cases/{case_id}",
        response_model=CaseDetail,
        tags=["investigation"],
        dependencies=[Depends(authenticated)],
    )
    def get_case(
        case_id: str,
        session: Session = Depends(session_dependency),
        scoring_service: ScoringService = Depends(service),
    ) -> CaseDetail:
        try:
            return scoring_service.get_case(session, case_id)
        except NotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.post(
        "/v1/cases/{case_id}/decision",
        response_model=CaseDetail,
        tags=["investigation"],
        dependencies=[Depends(authenticated)],
    )
    def decide_case(
        case_id: str,
        payload: CaseDecisionRequest,
        session: Session = Depends(session_dependency),
        scoring_service: ScoringService = Depends(service),
    ) -> CaseDetail:
        try:
            response = scoring_service.update_case(session, case_id, payload)
        except NotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        commit_or_conflict(session)
        return response

    @application.get(
        "/dashboard",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    def dashboard(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={"release": RELEASE},
        )

    return application


app = create_app()
