"""Environment-backed service configuration with safe production guards."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://sentinelgraph:sentinelgraph@localhost:5432/sentinelgraph"
)
DEFAULT_DEVELOPMENT_API_KEY = "local-development-only"


def project_root() -> Path:
    """Resolve the repository root from the installed source tree."""
    return Path(__file__).resolve().parents[3]


def _positive_int(name: str, raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration. Secrets are intentionally never serialised."""

    database_url: str = DEFAULT_DATABASE_URL
    api_key: str = DEFAULT_DEVELOPMENT_API_KEY
    account_hash_salt: str = "local-development-salt"
    model_bundle_path: Path = project_root() / "models/v0.5/risk_bundle.joblib"
    model_sha256: str | None = None
    environment: str = "development"
    maximum_batch_size: int = 100
    maximum_request_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        if not self.database_url:
            raise ValueError("database_url must not be empty")
        if len(self.api_key) < 16:
            raise ValueError("api_key must contain at least 16 characters")
        if len(self.account_hash_salt) < 16:
            raise ValueError("account_hash_salt must contain at least 16 characters")
        if self.maximum_batch_size <= 0 or self.maximum_batch_size > 1_000:
            raise ValueError("maximum_batch_size must be between 1 and 1000")
        if self.maximum_request_bytes <= 0:
            raise ValueError("maximum_request_bytes must be positive")
        if self.model_sha256 is not None:
            if len(self.model_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in self.model_sha256
            ):
                raise ValueError("model_sha256 must be a lowercase SHA-256 digest")
        if self.environment == "production":
            if self.api_key == DEFAULT_DEVELOPMENT_API_KEY:
                raise ValueError("production requires SENTINELGRAPH_API_KEY")
            if self.account_hash_salt == "local-development-salt":
                raise ValueError("production requires SENTINELGRAPH_ACCOUNT_HASH_SALT")
            if not self.database_url.startswith("postgresql+"):
                raise ValueError("production requires a PostgreSQL database URL")
            if self.model_sha256 is None:
                raise ValueError("production requires SENTINELGRAPH_MODEL_SHA256")

    @classmethod
    def from_env(cls) -> "Settings":
        """Read settings once at application construction time."""
        return cls(
            database_url=os.getenv(
                "SENTINELGRAPH_DATABASE_URL", DEFAULT_DATABASE_URL
            ),
            api_key=os.getenv(
                "SENTINELGRAPH_API_KEY", DEFAULT_DEVELOPMENT_API_KEY
            ),
            account_hash_salt=os.getenv(
                "SENTINELGRAPH_ACCOUNT_HASH_SALT", "local-development-salt"
            ),
            model_bundle_path=Path(
                os.getenv(
                    "SENTINELGRAPH_MODEL_BUNDLE",
                    str(project_root() / "models/v0.5/risk_bundle.joblib"),
                )
            ),
            model_sha256=os.getenv("SENTINELGRAPH_MODEL_SHA256") or None,
            environment=os.getenv(
                "SENTINELGRAPH_ENVIRONMENT", "development"
            ).lower(),
            maximum_batch_size=_positive_int(
                "SENTINELGRAPH_MAXIMUM_BATCH_SIZE",
                os.getenv("SENTINELGRAPH_MAXIMUM_BATCH_SIZE", "100"),
            ),
            maximum_request_bytes=_positive_int(
                "SENTINELGRAPH_MAXIMUM_REQUEST_BYTES",
                os.getenv("SENTINELGRAPH_MAXIMUM_REQUEST_BYTES", "2000000"),
            ),
        )
