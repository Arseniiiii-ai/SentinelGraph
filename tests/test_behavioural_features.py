"""Tests for strict point-in-time v0.3 behavioural features."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from sentinelgraph.modeling.behaviour import (
    BEHAVIOURAL_FEATURE_NAMES,
    BEHAVIOURAL_GROUPING_FIELDS,
    load_behavioural_matrix,
    materialize_behavioural_features,
)


def _write_transactions(path: Path) -> None:
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE fixture (
                source_row_number BIGINT,
                step INTEGER,
                type VARCHAR,
                amount DOUBLE,
                nameOrig VARCHAR,
                nameDest VARCHAR,
                isFraud UTINYINT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO fixture VALUES
                (1, 1, 'PAYMENT', 10.0, 'A', 'X', 0),
                (2, 2, 'TRANSFER', 20.0, 'A', 'X', 0),
                (3, 2, 'PAYMENT', 30.0, 'A', 'X', 1),
                (4, 3, 'PAYMENT', 40.0, 'A', 'Y', 0),
                (5, 4, 'CASH_OUT', 50.0, 'B', 'X', 1)
            """
        )
        escaped = str(path).replace("'", "''")
        connection.execute(f"COPY fixture TO '{escaped}' (FORMAT PARQUET)")
    finally:
        connection.close()


def _materialized(tmp_path: Path) -> Path:
    source = tmp_path / "source.parquet"
    destination = tmp_path / "features.parquet"
    _write_transactions(source)
    materialize_behavioural_features((source,), destination)
    return destination


def test_same_step_rows_are_excluded_from_history(tmp_path: Path) -> None:
    feature_path = _materialized(tmp_path)
    dataset = load_behavioural_matrix(feature_path)
    indexes = {name: index for index, name in enumerate(BEHAVIOURAL_FEATURE_NAMES)}

    row_two = dataset.features[1]
    row_three = dataset.features[2]
    row_four = dataset.features[3]

    assert np.isclose(row_two[indexes["origin_log_prior_tx_count"]], np.log(2))
    assert np.isclose(row_three[indexes["origin_log_prior_tx_count"]], np.log(2))
    assert row_two[indexes["origin_same_type_share"]] == 0.0
    assert row_three[indexes["origin_same_type_share"]] == 1.0
    assert np.isclose(row_four[indexes["origin_log_prior_tx_count"]], np.log(4))


def test_identifiers_are_grouping_keys_not_features(tmp_path: Path) -> None:
    feature_path = _materialized(tmp_path)
    connection = duckdb.connect()
    try:
        columns = {
            row[0]
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{feature_path}')"
            ).fetchall()
        }
    finally:
        connection.close()

    assert BEHAVIOURAL_GROUPING_FIELDS == {"nameOrig", "nameDest"}
    assert not BEHAVIOURAL_GROUPING_FIELDS & columns
    assert not BEHAVIOURAL_GROUPING_FIELDS & set(BEHAVIOURAL_FEATURE_NAMES)
    assert "target" in columns


def test_new_account_and_counterparty_features(tmp_path: Path) -> None:
    feature_path = _materialized(tmp_path)
    dataset = load_behavioural_matrix(feature_path)
    indexes = {name: index for index, name in enumerate(BEHAVIOURAL_FEATURE_NAMES)}

    first = dataset.features[0]
    returning = dataset.features[3]
    new_origin_known_destination = dataset.features[4]

    assert first[indexes["origin_is_new"]] == 1.0
    assert first[indexes["destination_is_new"]] == 1.0
    assert returning[indexes["origin_is_new"]] == 0.0
    assert np.isclose(
        returning[indexes["origin_log_unique_destinations"]],
        np.log(2),
    )
    assert new_origin_known_destination[indexes["origin_is_new"]] == 1.0
    assert new_origin_known_destination[indexes["destination_is_new"]] == 0.0


def test_sql_cap_is_deterministic_and_keeps_all_fraud(tmp_path: Path) -> None:
    feature_path = _materialized(tmp_path)

    first = load_behavioural_matrix(
        feature_path,
        max_legitimate_rows=2,
        random_seed=42,
    )
    second = load_behavioural_matrix(
        feature_path,
        max_legitimate_rows=2,
        random_seed=42,
    )

    assert first.rows == 4
    assert int(first.labels.sum()) == 2
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.labels, second.labels)
