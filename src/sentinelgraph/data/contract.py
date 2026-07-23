"""Raw PaySim data contract and DuckDB validation queries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import duckdb

PAYSIM_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)

PAYSIM_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

COLUMN_SQL = """{
    'step': 'INTEGER',
    'type': 'VARCHAR',
    'amount': 'DOUBLE',
    'nameOrig': 'VARCHAR',
    'oldbalanceOrg': 'DOUBLE',
    'newbalanceOrig': 'DOUBLE',
    'nameDest': 'VARCHAR',
    'oldbalanceDest': 'DOUBLE',
    'newbalanceDest': 'DOUBLE',
    'isFraud': 'UTINYINT',
    'isFlaggedFraud': 'UTINYINT'
}"""


def read_header(csv_path: Path) -> tuple[str, ...]:
    """Read the CSV header without loading any transaction rows."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(next(csv.reader(handle)))


def csv_scan_sql(csv_path: Path) -> str:
    """Return a strict, explicitly typed DuckDB CSV scan expression."""
    escaped_path = str(csv_path.resolve()).replace("'", "''")
    return (
        f"read_csv('{escaped_path}', header = true, columns = {COLUMN_SQL}, "
        "nullstr = '', strict_mode = true, ignore_errors = false)"
    )


def create_transaction_table(
    connection: duckdb.DuckDBPyConnection,
    csv_path: Path,
) -> None:
    """Load the canonical CSV into a typed table with a stable source row id."""
    actual_header = read_header(csv_path)
    if actual_header != PAYSIM_COLUMNS:
        raise ValueError(
            f"unexpected PaySim header: {actual_header!r}; expected {PAYSIM_COLUMNS!r}"
        )

    connection.execute("DROP TABLE IF EXISTS transactions")
    connection.execute(
        f"""
        CREATE TABLE transactions AS
        SELECT
            row_number() OVER ()::BIGINT AS source_row_number,
            *
        FROM {csv_scan_sql(csv_path)}
        """
    )


def validate_transaction_table(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Run release-blocking checks and non-blocking anomaly observations."""
    aggregate = connection.execute(
        """
        SELECT
            count(*)::BIGINT AS row_count,
            count_if(step IS NULL)::BIGINT AS null_step,
            count_if(type IS NULL)::BIGINT AS null_type,
            count_if(amount IS NULL)::BIGINT AS null_amount,
            count_if(nameOrig IS NULL OR nameOrig = '')::BIGINT AS null_nameOrig,
            count_if(oldbalanceOrg IS NULL)::BIGINT AS null_oldbalanceOrg,
            count_if(newbalanceOrig IS NULL)::BIGINT AS null_newbalanceOrig,
            count_if(nameDest IS NULL OR nameDest = '')::BIGINT AS null_nameDest,
            count_if(oldbalanceDest IS NULL)::BIGINT AS null_oldbalanceDest,
            count_if(newbalanceDest IS NULL)::BIGINT AS null_newbalanceDest,
            count_if(isFraud IS NULL)::BIGINT AS null_isFraud,
            count_if(isFlaggedFraud IS NULL)::BIGINT AS null_isFlaggedFraud,
            count_if(step < 1)::BIGINT AS invalid_step,
            count_if(type NOT IN (
                'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'
            ))::BIGINT AS invalid_type,
            count_if(amount < 0)::BIGINT AS negative_amount,
            count_if(
                oldbalanceOrg < 0 OR newbalanceOrig < 0
                OR oldbalanceDest < 0 OR newbalanceDest < 0
            )::BIGINT AS negative_balance,
            count_if(NOT regexp_full_match(nameOrig, 'C[0-9]+'))::BIGINT
                AS invalid_origin_id,
            count_if(NOT regexp_full_match(nameDest, '[CM][0-9]+'))::BIGINT
                AS invalid_destination_id,
            count_if(isFraud NOT IN (0, 1))::BIGINT AS invalid_fraud_label,
            count_if(isFlaggedFraud NOT IN (0, 1))::BIGINT
                AS invalid_flag_label,
            count_if(nameOrig = nameDest)::BIGINT AS self_transfer,
            count_if(isFraud = 1 AND type NOT IN ('TRANSFER', 'CASH_OUT'))::BIGINT
                AS fraud_outside_expected_types,
            count_if(isFlaggedFraud = 1 AND type <> 'TRANSFER')::BIGINT
                AS flag_outside_transfer,
            count_if(amount = 0)::BIGINT AS zero_amount,
            count_if(
                newbalanceOrig > oldbalanceOrg AND type <> 'CASH_IN'
            )::BIGINT AS unexpected_origin_increase
        FROM transactions
        """
    ).fetchone()
    if aggregate is None:
        raise RuntimeError("validation query returned no result")

    fields = [description[0] for description in connection.description]
    values = dict(zip(fields, aggregate, strict=True))

    error_names = (
        "null_step",
        "null_type",
        "null_amount",
        "null_nameOrig",
        "null_oldbalanceOrg",
        "null_newbalanceOrig",
        "null_nameDest",
        "null_oldbalanceDest",
        "null_newbalanceDest",
        "null_isFraud",
        "null_isFlaggedFraud",
        "invalid_step",
        "invalid_type",
        "negative_amount",
        "negative_balance",
        "invalid_origin_id",
        "invalid_destination_id",
        "invalid_fraud_label",
        "invalid_flag_label",
        "self_transfer",
        "fraud_outside_expected_types",
        "flag_outside_transfer",
    )
    warning_names = ("zero_amount", "unexpected_origin_increase")

    errors = {name: int(values[name]) for name in error_names}
    warnings = {name: int(values[name]) for name in warning_names}
    return {
        "passed": all(count == 0 for count in errors.values()),
        "row_count": int(values["row_count"]),
        "errors": errors,
        "warnings": warnings,
    }
