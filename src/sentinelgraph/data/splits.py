"""Leakage-safe chronological and new-account evaluation splits."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb

from sentinelgraph.data.provenance import file_record

TRAIN_FRACTION_BY_TIME = 0.70


def temporal_cutoff(
    minimum_step: int,
    maximum_step: int,
    train_fraction: float = TRAIN_FRACTION_BY_TIME,
) -> int:
    """Return the last training step from a fraction of the observed time span."""
    if minimum_step > maximum_step:
        raise ValueError("minimum_step must not exceed maximum_step")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    observed_span = maximum_step - minimum_step + 1
    train_steps = max(1, math.floor(observed_span * train_fraction))
    return minimum_step + train_steps - 1


def _split_stats(
    connection: duckdb.DuckDBPyConnection,
    where_sql: str,
) -> dict[str, Any]:
    result = connection.execute(
        f"""
        SELECT
            count(*)::BIGINT AS rows,
            min(step)::INTEGER AS min_step,
            max(step)::INTEGER AS max_step,
            sum(isFraud)::BIGINT AS fraud_rows,
            sum(CASE WHEN isFraud = 1 THEN amount ELSE 0 END)::DOUBLE
                AS fraud_amount,
            count(DISTINCT nameOrig)::BIGINT AS unique_origins
        FROM transactions t
        WHERE {where_sql}
        """
    ).fetchone()
    if result is None:
        raise RuntimeError("split statistics query returned no result")
    fields = [description[0] for description in connection.description]
    stats = dict(zip(fields, result, strict=True))
    stats["fraud_rate"] = stats["fraud_rows"] / stats["rows"] if stats["rows"] else None
    return stats


def _copy_parquet(
    connection: duckdb.DuckDBPyConnection,
    destination: Path,
    where_sql: str,
) -> None:
    if destination.exists():
        destination.unlink()
    escaped = str(destination.resolve()).replace("'", "''")
    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM transactions t
            WHERE {where_sql}
            ORDER BY source_row_number
        )
        TO '{escaped}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def build_splits(
    connection: duckdb.DuckDBPyConnection,
    processed_dir: Path,
    *,
    train_fraction: float = TRAIN_FRACTION_BY_TIME,
) -> dict[str, Any]:
    """Build train, future-time, and future new-origin Parquet artifacts.

    The new-account holdout is intentionally a subset of the future-time
    holdout. It tests cold-start performance without introducing a random-time
    split or contaminating training with future events.
    """
    limits = connection.execute(
        "SELECT min(step)::INTEGER, max(step)::INTEGER FROM transactions"
    ).fetchone()
    if limits is None or limits[0] is None or limits[1] is None:
        raise RuntimeError("cannot split an empty dataset")
    minimum_step, maximum_step = int(limits[0]), int(limits[1])
    cutoff = temporal_cutoff(minimum_step, maximum_step, train_fraction)

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE training_origins AS
        SELECT DISTINCT nameOrig
        FROM transactions
        WHERE step <= {cutoff}
        """
    )

    predicates = {
        "train": f"t.step <= {cutoff}",
        "future_time_holdout": f"t.step > {cutoff}",
        "new_account_holdout": (
            f"t.step > {cutoff} AND NOT EXISTS ("
            "SELECT 1 FROM training_origins known "
            "WHERE known.nameOrig = t.nameOrig)"
        ),
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for name, predicate in predicates.items():
        destination = processed_dir / f"{name}.parquet"
        _copy_parquet(connection, destination, predicate)
        outputs[name] = {
            "definition": predicate,
            "statistics": _split_stats(connection, predicate),
            "artifact": file_record(
                destination,
                relative_to=processed_dir.parents[1],
            ),
        }

    overlap = connection.execute(
        f"""
        SELECT count(*)::BIGINT
        FROM transactions t
        WHERE t.step <= {cutoff}
          AND NOT EXISTS (
              SELECT 1 FROM training_origins known
              WHERE known.nameOrig = t.nameOrig
          )
        """
    ).fetchone()
    if overlap is None:
        raise RuntimeError("split leakage query returned no result")

    return {
        "strategy": "chronological_time_span_then_future_new-origin_slice",
        "train_fraction_by_observed_time_span": train_fraction,
        "minimum_step": minimum_step,
        "maximum_step": maximum_step,
        "train_end_step_inclusive": cutoff,
        "future_start_step_inclusive": cutoff + 1,
        "new_account_definition": (
            "origin account nameOrig has no transaction at or before the "
            "training cutoff"
        ),
        "relationship": (
            "new_account_holdout is a subset of future_time_holdout; both are "
            "disjoint from train"
        ),
        "training_new_origin_violations": int(overlap[0]),
        "splits": outputs,
    }
