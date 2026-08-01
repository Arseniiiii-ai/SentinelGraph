"""SQLAlchemy engine and request-scoped unit-of-work construction."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from sentinelgraph.api.models import Base


class Database:
    """Own the process-wide engine and create isolated request sessions."""

    def __init__(self, database_url: str) -> None:
        connect_args = (
            {"check_same_thread": False}
            if database_url.startswith("sqlite")
            else {}
        )
        self.engine: Engine = create_engine(
            database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        if database_url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def enable_sqlite_foreign_keys(
                dbapi_connection: Any, connection_record: Any
            ) -> None:
                del connection_record
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        self._session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )

    def session(self) -> Session:
        """Create one unit of work; callers own commit or rollback."""
        return self._session_factory()

    def session_dependency(self) -> Iterator[Session]:
        """Yield a request session and guarantee rollback/close on failure."""
        session = self.session()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def ping(self) -> None:
        """Verify that a connection can execute a trivial statement."""
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def create_schema_for_tests(self) -> None:
        """Create tables only for isolated tests; production uses Alembic."""
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()
