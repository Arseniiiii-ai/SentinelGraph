"""Tests for v0.2 feature policy and matrix construction."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from sentinelgraph.modeling.features import (
    ALLOWED_SOURCE_FIELDS,
    FEATURE_NAMES,
    PROHIBITED_MODEL_FIELDS,
    deterministic_class_cap,
    load_matrix,
)


def _write_fixture_parquet(path: Path) -> None:
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE fixture (
                source_row_number BIGINT,
                step INTEGER,
                type VARCHAR,
                amount DOUBLE,
                isFraud UTINYINT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO fixture VALUES
                (1, 1, 'PAYMENT', 10.0, 0),
                (2, 2, 'TRANSFER', 250000.0, 1),
                (3, 25, 'CASH_OUT', 100.0, 0)
            """
        )
        escaped = str(path).replace("'", "''")
        connection.execute(f"COPY fixture TO '{escaped}' (FORMAT PARQUET)")
    finally:
        connection.close()


def test_feature_policy_excludes_known_leakage() -> None:
    assert ALLOWED_SOURCE_FIELDS == {"type", "amount", "step"}
    assert "isFraud" in PROHIBITED_MODEL_FIELDS
    assert "isFlaggedFraud" in PROHIBITED_MODEL_FIELDS
    assert "newbalanceOrig" in PROHIBITED_MODEL_FIELDS
    assert "nameOrig" in PROHIBITED_MODEL_FIELDS
    assert not ALLOWED_SOURCE_FIELDS & PROHIBITED_MODEL_FIELDS


def test_loader_builds_numeric_point_in_time_matrix(tmp_path: Path) -> None:
    parquet_path = tmp_path / "fixture.parquet"
    _write_fixture_parquet(parquet_path)

    dataset = load_matrix(parquet_path)

    assert dataset.features.shape == (3, len(FEATURE_NAMES))
    assert dataset.features.dtype == np.float32
    assert dataset.labels.tolist() == [0, 1, 0]
    assert dataset.amounts.tolist() == [10.0, 250000.0, 100.0]
    assert np.isclose(dataset.features[0, 1], np.log1p(10.0))
    assert dataset.features[1, FEATURE_NAMES.index("type_transfer")] == 1.0


def test_class_cap_keeps_all_fraud() -> None:
    features = np.arange(60, dtype=np.float32).reshape(6, 10)
    labels = np.array([0, 0, 0, 0, 1, 1], dtype=np.uint8)
    amounts = np.arange(6, dtype=np.float64)
    from sentinelgraph.modeling.features import MatrixDataset

    capped = deterministic_class_cap(
        MatrixDataset(features, labels, amounts),
        max_legitimate_rows=2,
        random_seed=42,
    )

    assert capped.rows == 4
    assert int(capped.labels.sum()) == 2
