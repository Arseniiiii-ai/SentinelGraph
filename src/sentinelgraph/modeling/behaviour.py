"""Leakage-safe point-in-time behavioural feature construction for v0.3."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from sentinelgraph.data.provenance import file_record
from sentinelgraph.modeling.features import FEATURE_NAMES, MatrixDataset

BEHAVIOURAL_SOURCE_FIELDS = frozenset(
    {"source_row_number", "step", "type", "amount", "nameOrig", "nameDest"}
)
BEHAVIOURAL_GROUPING_FIELDS = frozenset({"nameOrig", "nameDest"})
STRICT_PAST_WINDOW_END = -1
NEW_ACCOUNT_HOURS_SENTINEL = 744.0

BEHAVIOURAL_ONLY_FEATURE_NAMES = (
    "destination_is_merchant",
    "origin_is_new",
    "origin_log_prior_tx_count",
    "origin_log_prior_amount_mean",
    "origin_log_amount_deviation",
    "origin_hours_since_last",
    "origin_log_tx_count_24h",
    "origin_log_amount_sum_24h",
    "origin_log_tx_count_168h",
    "origin_log_amount_sum_168h",
    "origin_same_type_share",
    "origin_log_unique_destinations",
    "destination_is_new",
    "destination_log_prior_tx_count",
    "destination_log_prior_amount_mean",
    "destination_log_amount_deviation",
    "destination_hours_since_last",
    "destination_log_tx_count_24h",
    "destination_log_amount_sum_24h",
    "destination_log_tx_count_168h",
    "destination_log_amount_sum_168h",
    "destination_same_type_share",
    "destination_log_unique_origins",
)
BEHAVIOURAL_FEATURE_NAMES = FEATURE_NAMES + BEHAVIOURAL_ONLY_FEATURE_NAMES


def _escaped(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _source_sql(source_paths: Sequence[Path]) -> str:
    if not source_paths:
        raise ValueError("at least one source Parquet path is required")
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing source Parquet: {missing[0]}")
    paths = ", ".join(f"'{_escaped(path)}'" for path in source_paths)
    return f"read_parquet([{paths}])"


def _feature_query(source_paths: Sequence[Path]) -> str:
    source = _source_sql(source_paths)
    return f"""
        WITH base AS (
            SELECT
                source_row_number::BIGINT AS source_row_number,
                step::INTEGER AS step,
                type::VARCHAR AS type,
                amount::DOUBLE AS amount,
                nameOrig::VARCHAR AS nameOrig,
                nameDest::VARCHAR AS nameDest,
                isFraud::UTINYINT AS target
            FROM {source}
        ),
        history AS (
            SELECT
                *,
                count(*) OVER (
                    PARTITION BY nameOrig ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS origin_prior_count,
                avg(amount) OVER (
                    PARTITION BY nameOrig ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS origin_prior_amount_mean,
                max(step) OVER (
                    PARTITION BY nameOrig ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS origin_previous_step,
                count(*) OVER (
                    PARTITION BY nameOrig ORDER BY step
                    RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
                ) AS origin_count_24h,
                sum(amount) OVER (
                    PARTITION BY nameOrig ORDER BY step
                    RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
                ) AS origin_amount_24h,
                count(*) OVER (
                    PARTITION BY nameOrig ORDER BY step
                    RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
                ) AS origin_count_168h,
                sum(amount) OVER (
                    PARTITION BY nameOrig ORDER BY step
                    RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
                ) AS origin_amount_168h,
                count(*) OVER (
                    PARTITION BY nameOrig, type ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS origin_same_type_count,
                count(DISTINCT nameDest) OVER (
                    PARTITION BY nameOrig ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS origin_unique_destinations,
                count(*) OVER (
                    PARTITION BY nameDest ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS destination_prior_count,
                avg(amount) OVER (
                    PARTITION BY nameDest ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS destination_prior_amount_mean,
                max(step) OVER (
                    PARTITION BY nameDest ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS destination_previous_step,
                count(*) OVER (
                    PARTITION BY nameDest ORDER BY step
                    RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
                ) AS destination_count_24h,
                sum(amount) OVER (
                    PARTITION BY nameDest ORDER BY step
                    RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
                ) AS destination_amount_24h,
                count(*) OVER (
                    PARTITION BY nameDest ORDER BY step
                    RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
                ) AS destination_count_168h,
                sum(amount) OVER (
                    PARTITION BY nameDest ORDER BY step
                    RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
                ) AS destination_amount_168h,
                count(*) OVER (
                    PARTITION BY nameDest, type ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS destination_same_type_count,
                count(DISTINCT nameOrig) OVER (
                    PARTITION BY nameDest ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS destination_unique_origins
            FROM base
        )
        SELECT
            source_row_number,
            step,
            amount::FLOAT AS amount,
            ln(1 + amount)::FLOAT AS log_amount,
            sin(2 * pi() * ((step - 1) % 24) / 24)::FLOAT AS hour_sin,
            cos(2 * pi() * ((step - 1) % 24) / 24)::FLOAT AS hour_cos,
            (type = 'CASH_IN')::UTINYINT::FLOAT AS type_cash_in,
            (type = 'CASH_OUT')::UTINYINT::FLOAT AS type_cash_out,
            (type = 'DEBIT')::UTINYINT::FLOAT AS type_debit,
            (type = 'PAYMENT')::UTINYINT::FLOAT AS type_payment,
            (type = 'TRANSFER')::UTINYINT::FLOAT AS type_transfer,
            starts_with(nameDest, 'M')::UTINYINT::FLOAT
                AS destination_is_merchant,
            (origin_prior_count = 0)::UTINYINT::FLOAT AS origin_is_new,
            ln(1 + origin_prior_count)::FLOAT AS origin_log_prior_tx_count,
            ln(1 + coalesce(origin_prior_amount_mean, 0))::FLOAT
                AS origin_log_prior_amount_mean,
            CASE
                WHEN origin_prior_count = 0 THEN 0
                ELSE ln(1 + amount) - ln(1 + origin_prior_amount_mean)
            END::FLOAT AS origin_log_amount_deviation,
            coalesce(
                step - origin_previous_step,
                {NEW_ACCOUNT_HOURS_SENTINEL}
            )::FLOAT AS origin_hours_since_last,
            ln(1 + origin_count_24h)::FLOAT AS origin_log_tx_count_24h,
            ln(1 + coalesce(origin_amount_24h, 0))::FLOAT
                AS origin_log_amount_sum_24h,
            ln(1 + origin_count_168h)::FLOAT AS origin_log_tx_count_168h,
            ln(1 + coalesce(origin_amount_168h, 0))::FLOAT
                AS origin_log_amount_sum_168h,
            CASE
                WHEN origin_prior_count = 0 THEN 0
                ELSE origin_same_type_count::DOUBLE / origin_prior_count
            END::FLOAT AS origin_same_type_share,
            ln(1 + origin_unique_destinations)::FLOAT
                AS origin_log_unique_destinations,
            (destination_prior_count = 0)::UTINYINT::FLOAT
                AS destination_is_new,
            ln(1 + destination_prior_count)::FLOAT
                AS destination_log_prior_tx_count,
            ln(1 + coalesce(destination_prior_amount_mean, 0))::FLOAT
                AS destination_log_prior_amount_mean,
            CASE
                WHEN destination_prior_count = 0 THEN 0
                ELSE ln(1 + amount) - ln(1 + destination_prior_amount_mean)
            END::FLOAT AS destination_log_amount_deviation,
            coalesce(
                step - destination_previous_step,
                {NEW_ACCOUNT_HOURS_SENTINEL}
            )::FLOAT AS destination_hours_since_last,
            ln(1 + destination_count_24h)::FLOAT
                AS destination_log_tx_count_24h,
            ln(1 + coalesce(destination_amount_24h, 0))::FLOAT
                AS destination_log_amount_sum_24h,
            ln(1 + destination_count_168h)::FLOAT
                AS destination_log_tx_count_168h,
            ln(1 + coalesce(destination_amount_168h, 0))::FLOAT
                AS destination_log_amount_sum_168h,
            CASE
                WHEN destination_prior_count = 0 THEN 0
                ELSE destination_same_type_count::DOUBLE
                    / destination_prior_count
            END::FLOAT AS destination_same_type_share,
            ln(1 + destination_unique_origins)::FLOAT
                AS destination_log_unique_origins,
            target,
            amount::DOUBLE AS evaluation_amount
        FROM history
    """


def materialize_behavioural_features(
    source_paths: Sequence[Path],
    destination: Path,
    *,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic feature store using only strictly earlier steps."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    escaped_destination = _escaped(destination)
    connection = duckdb.connect()
    try:
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(
            f"""
            COPY (
                {_feature_query(source_paths)}
                ORDER BY source_row_number
            )
            TO '{escaped_destination}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
    finally:
        connection.close()
    record_root = relative_to if relative_to is not None else destination.parent
    return inspect_behavioural_feature_store(
        destination,
        relative_to=record_root,
    )


def inspect_behavioural_feature_store(
    feature_path: Path,
    *,
    relative_to: Path,
) -> dict[str, Any]:
    """Return the reproducibility manifest for an existing feature store."""
    if not feature_path.exists():
        raise FileNotFoundError(f"missing behavioural feature store: {feature_path}")
    connection = duckdb.connect()
    try:
        row = connection.execute(
            f"""
            SELECT
                count(*)::BIGINT,
                min(step)::INTEGER,
                max(step)::INTEGER,
                count_if(target = 1)::BIGINT
            FROM read_parquet('{_escaped(feature_path)}')
            """
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("behavioural feature statistics query returned no row")
    return {
        "artifact": file_record(feature_path, relative_to=relative_to),
        "rows": int(row[0]),
        "minimum_step": int(row[1]),
        "maximum_step": int(row[2]),
        "fraud_rows": int(row[3]),
        "feature_names": list(BEHAVIOURAL_FEATURE_NAMES),
        "feature_count": len(BEHAVIOURAL_FEATURE_NAMES),
        "history_contract": {
            "window_upper_bound_steps": STRICT_PAST_WINDOW_END,
            "same_step_events_excluded": True,
            "identifiers_used_only_as_grouping_keys": sorted(
                BEHAVIOURAL_GROUPING_FIELDS
            ),
            "identifiers_emitted_as_features": False,
        },
    }


def load_behavioural_matrix(
    feature_path: Path,
    *,
    where_sql: str = "TRUE",
    max_legitimate_rows: int | None = None,
    random_seed: int = 42,
) -> MatrixDataset:
    """Load v0.3 features, optionally with a deterministic legitimate-row cap."""
    if not feature_path.exists():
        raise FileNotFoundError(f"missing behavioural feature store: {feature_path}")
    escaped_path = _escaped(feature_path)
    feature_columns = ",\n".join(BEHAVIOURAL_FEATURE_NAMES)
    if max_legitimate_rows is not None and max_legitimate_rows <= 0:
        raise ValueError("max_legitimate_rows must be positive")

    if max_legitimate_rows is None:
        selected_sql = "SELECT * FROM filtered"
    else:
        selected_sql = f"""
            SELECT * FROM filtered WHERE target = 1
            UNION ALL
            SELECT * EXCLUDE (sample_rank)
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        ORDER BY hash(source_row_number, {random_seed})
                    ) AS sample_rank
                FROM filtered
                WHERE target = 0
            )
            WHERE sample_rank <= {max_legitimate_rows}
        """

    connection = duckdb.connect()
    try:
        columns = connection.execute(
            f"""
            WITH filtered AS (
                SELECT *
                FROM read_parquet('{escaped_path}')
                WHERE {where_sql}
            ),
            selected AS (
                {selected_sql}
            )
            SELECT
                {feature_columns},
                target::UTINYINT AS target,
                evaluation_amount::DOUBLE AS evaluation_amount
            FROM selected
            ORDER BY source_row_number
            """
        ).fetchnumpy()
    finally:
        connection.close()

    feature_matrix = np.column_stack(
        [
            np.asarray(columns[name], dtype=np.float32)
            for name in BEHAVIOURAL_FEATURE_NAMES
        ]
    )
    labels = np.asarray(columns["target"], dtype=np.uint8)
    amounts = np.asarray(columns["evaluation_amount"], dtype=np.float64)
    return MatrixDataset(feature_matrix, labels, amounts)
