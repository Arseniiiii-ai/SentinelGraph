"""Exact data-quality profiling queries for the PaySim release."""

from __future__ import annotations

from typing import Any

import duckdb

RAW_COLUMNS_SQL = """
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
"""


def _rows_as_dicts(
    connection: duckdb.DuckDBPyConnection,
    query: str,
) -> list[dict[str, Any]]:
    cursor = connection.execute(query)
    fields = [description[0] for description in cursor.description]
    return [dict(zip(fields, row, strict=True)) for row in cursor.fetchall()]


def _one_row(
    connection: duckdb.DuckDBPyConnection,
    query: str,
) -> dict[str, Any]:
    rows = _rows_as_dicts(connection, query)
    if len(rows) != 1:
        raise RuntimeError(f"expected one profiling row, received {len(rows)}")
    return rows[0]


def profile_transactions(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Profile labels, missingness, duplicates, activity, types, and time."""
    overview = _one_row(
        connection,
        """
        SELECT
            count(*)::BIGINT AS rows,
            min(step)::INTEGER AS min_step,
            max(step)::INTEGER AS max_step,
            min(amount)::DOUBLE AS min_amount,
            max(amount)::DOUBLE AS max_amount,
            sum(amount)::DOUBLE AS total_amount,
            sum(isFraud)::BIGINT AS fraud_rows,
            sum(isFlaggedFraud)::BIGINT AS flagged_rows,
            sum(CASE WHEN isFraud = 1 THEN amount ELSE 0 END)::DOUBLE
                AS fraud_amount,
            count(DISTINCT nameOrig)::BIGINT AS unique_origins,
            count(DISTINCT nameDest)::BIGINT AS unique_destinations
        FROM transactions
        """,
    )
    overview["fraud_rate"] = overview["fraud_rows"] / overview["rows"]
    overview["flagged_rate"] = overview["flagged_rows"] / overview["rows"]

    missing = _one_row(
        connection,
        """
        SELECT
            count_if(step IS NULL)::BIGINT AS step,
            count_if(type IS NULL)::BIGINT AS type,
            count_if(amount IS NULL)::BIGINT AS amount,
            count_if(nameOrig IS NULL OR nameOrig = '')::BIGINT AS nameOrig,
            count_if(oldbalanceOrg IS NULL)::BIGINT AS oldbalanceOrg,
            count_if(newbalanceOrig IS NULL)::BIGINT AS newbalanceOrig,
            count_if(nameDest IS NULL OR nameDest = '')::BIGINT AS nameDest,
            count_if(oldbalanceDest IS NULL)::BIGINT AS oldbalanceDest,
            count_if(newbalanceDest IS NULL)::BIGINT AS newbalanceDest,
            count_if(isFraud IS NULL)::BIGINT AS isFraud,
            count_if(isFlaggedFraud IS NULL)::BIGINT AS isFlaggedFraud
        FROM transactions
        """,
    )

    duplicates = _one_row(
        connection,
        f"""
        SELECT
            count(*)::BIGINT AS duplicate_groups,
            coalesce(sum(group_size - 1), 0)::BIGINT AS duplicate_rows
        FROM (
            SELECT count(*)::BIGINT AS group_size
            FROM transactions
            GROUP BY {RAW_COLUMNS_SQL}
            HAVING count(*) > 1
        )
        """,
    )

    label_relationships = _one_row(
        connection,
        """
        SELECT
            count_if(amount = 0)::BIGINT AS zero_amount_rows,
            count_if(amount = 0 AND isFraud = 1)::BIGINT
                AS zero_amount_fraud_rows,
            count_if(isFlaggedFraud = 1 AND isFraud = 1)::BIGINT
                AS flagged_true_fraud_rows,
            count_if(isFlaggedFraud = 1 AND isFraud = 0)::BIGINT
                AS flagged_non_fraud_rows
        FROM transactions
        """,
    )

    semantic_missingness = _one_row(
        connection,
        """
        SELECT
            count_if(nameDest LIKE 'M%')::BIGINT
                AS merchant_destination_rows,
            count_if(
                nameDest LIKE 'M%'
                AND oldbalanceDest = 0
                AND newbalanceDest = 0
            )::BIGINT AS merchant_zero_balance_placeholders,
            count_if(
                nameDest LIKE 'C%'
                AND oldbalanceDest = 0
                AND newbalanceDest = 0
            )::BIGINT AS customer_zero_destination_balances
        FROM transactions
        """,
    )

    types = _rows_as_dicts(
        connection,
        """
        SELECT
            type,
            count(*)::BIGINT AS rows,
            sum(isFraud)::BIGINT AS fraud_rows,
            sum(amount)::DOUBLE AS total_amount,
            sum(CASE WHEN isFraud = 1 THEN amount ELSE 0 END)::DOUBLE
                AS fraud_amount
        FROM transactions
        GROUP BY type
        ORDER BY rows DESC
        """,
    )

    account_summary = _one_row(
        connection,
        """
        WITH account_events AS (
            SELECT
                nameOrig AS account_id,
                count(*)::BIGINT AS originated,
                0::BIGINT AS received,
                sum(isFraud)::BIGINT AS fraud_events
            FROM transactions
            GROUP BY nameOrig
            UNION ALL
            SELECT
                nameDest AS account_id,
                0::BIGINT AS originated,
                count(*)::BIGINT AS received,
                sum(isFraud)::BIGINT AS fraud_events
            FROM transactions
            GROUP BY nameDest
        ),
        activity AS (
            SELECT
                account_id,
                sum(originated)::BIGINT AS originated,
                sum(received)::BIGINT AS received,
                sum(originated + received)::BIGINT AS total_events,
                sum(fraud_events)::BIGINT AS fraud_events
            FROM account_events
            GROUP BY account_id
        )
        SELECT
            count(*)::BIGINT AS unique_accounts,
            count_if(account_id LIKE 'M%')::BIGINT AS merchant_accounts,
            count_if(originated > 1)::BIGINT AS repeat_originators,
            median(total_events)::DOUBLE AS median_events_per_account,
            quantile_cont(total_events, 0.95)::DOUBLE AS p95_events_per_account,
            max(total_events)::BIGINT AS max_events_per_account,
            count_if(fraud_events > 0)::BIGINT AS accounts_touching_fraud
        FROM activity
        """,
    )

    top_origins = _rows_as_dicts(
        connection,
        """
        SELECT
            nameOrig AS account_id,
            count(*)::BIGINT AS transactions,
            sum(isFraud)::BIGINT AS fraud_transactions,
            sum(amount)::DOUBLE AS total_amount
        FROM transactions
        GROUP BY nameOrig
        ORDER BY transactions DESC, total_amount DESC, account_id
        LIMIT 10
        """,
    )
    top_destinations = _rows_as_dicts(
        connection,
        """
        SELECT
            nameDest AS account_id,
            count(*)::BIGINT AS transactions,
            sum(isFraud)::BIGINT AS fraud_transactions,
            sum(amount)::DOUBLE AS total_amount
        FROM transactions
        GROUP BY nameDest
        ORDER BY transactions DESC, total_amount DESC, account_id
        LIMIT 10
        """,
    )
    temporal = _one_row(
        connection,
        """
        WITH hourly AS (
            SELECT
                step,
                count(*)::BIGINT AS rows,
                sum(isFraud)::BIGINT AS fraud_rows
            FROM transactions
            GROUP BY step
        )
        SELECT
            count(*)::BIGINT AS observed_steps,
            min(rows)::BIGINT AS min_rows_per_step,
            median(rows)::DOUBLE AS median_rows_per_step,
            max(rows)::BIGINT AS max_rows_per_step,
            count_if(fraud_rows > 0)::BIGINT AS steps_with_fraud
        FROM hourly
        """,
    )

    return {
        "overview": overview,
        "missing_values": missing,
        "semantic_missingness": semantic_missingness,
        "duplicates": duplicates,
        "label_relationships": label_relationships,
        "transaction_types": types,
        "account_activity": account_summary,
        "top_origin_accounts": top_origins,
        "top_destination_accounts": top_destinations,
        "temporal_activity": temporal,
    }
