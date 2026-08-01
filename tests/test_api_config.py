"""Fail-closed configuration checks for production serving."""

from __future__ import annotations

from typing import Any

import pytest

from sentinelgraph.api.config import Settings


def production_settings(**overrides: object) -> dict[str, Any]:
    values: dict[str, Any] = {
        "database_url": "postgresql+psycopg://service@db/sentinelgraph",
        "api_key": "production-api-key-long-enough",
        "account_hash_salt": "production-account-salt-long-enough",
        "environment": "production",
        "model_sha256": "a" * 64,
    }
    values.update(overrides)
    return values


def test_production_requires_model_checksum() -> None:
    with pytest.raises(ValueError, match="MODEL_SHA256"):
        Settings(**production_settings(model_sha256=None))


def test_production_requires_postgresql() -> None:
    with pytest.raises(ValueError, match="PostgreSQL"):
        Settings(**production_settings(database_url="sqlite:///unsafe.db"))


def test_valid_production_settings_are_accepted() -> None:
    settings = Settings(**production_settings())
    assert settings.environment == "production"
    assert settings.model_sha256 == "a" * 64
