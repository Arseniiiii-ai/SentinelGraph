"""Tests for the raw PaySim release contract."""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pytest

from sentinelgraph.data.contract import (
    PAYSIM_COLUMNS,
    create_transaction_table,
    read_header,
    validate_transaction_table,
)


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PAYSIM_COLUMNS)
        writer.writerows(rows)


def test_valid_raw_rows_pass_contract(tmp_path: Path) -> None:
    csv_path = tmp_path / "paysim.csv"
    _write_rows(
        csv_path,
        [
            [
                1,
                "PAYMENT",
                10.5,
                "C1",
                20.5,
                10.0,
                "M2",
                0.0,
                0.0,
                0,
                0,
            ],
            [
                2,
                "TRANSFER",
                0.0,
                "C3",
                0.0,
                0.0,
                "C4",
                0.0,
                0.0,
                1,
                0,
            ],
        ],
    )
    connection = duckdb.connect()
    try:
        create_transaction_table(connection, csv_path)
        result = validate_transaction_table(connection)
    finally:
        connection.close()

    assert result["passed"] is True
    assert result["row_count"] == 2
    assert result["warnings"]["zero_amount"] == 1
    assert all(count == 0 for count in result["errors"].values())


def test_negative_amount_is_release_blocking(tmp_path: Path) -> None:
    csv_path = tmp_path / "paysim.csv"
    _write_rows(
        csv_path,
        [
            [
                1,
                "PAYMENT",
                -1.0,
                "C1",
                20.0,
                21.0,
                "M2",
                0.0,
                0.0,
                0,
                0,
            ]
        ],
    )
    connection = duckdb.connect()
    try:
        create_transaction_table(connection, csv_path)
        result = validate_transaction_table(connection)
    finally:
        connection.close()

    assert result["passed"] is False
    assert result["errors"]["negative_amount"] == 1


def test_unexpected_header_fails_before_loading(tmp_path: Path) -> None:
    csv_path = tmp_path / "wrong.csv"
    csv_path.write_text("step,type,amount\n1,PAYMENT,10\n", encoding="utf-8")
    assert read_header(csv_path) == ("step", "type", "amount")

    connection = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="unexpected PaySim header"):
            create_transaction_table(connection, csv_path)
    finally:
        connection.close()
