"""Point-in-time feature policy and efficient Parquet matrix loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES = (
    "amount",
    "log_amount",
    "hour_sin",
    "hour_cos",
    "type_cash_in",
    "type_cash_out",
    "type_debit",
    "type_payment",
    "type_transfer",
)

ALLOWED_SOURCE_FIELDS = frozenset({"type", "amount", "step"})
PROHIBITED_MODEL_FIELDS = frozenset(
    {
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
        "source_row_number",
    }
)


@dataclass(frozen=True, slots=True)
class MatrixDataset:
    """Numeric model matrix plus labels and evaluation-only amounts."""

    features: NDArray[np.float32]
    labels: NDArray[np.uint8]
    amounts: NDArray[np.float64]

    @property
    def rows(self) -> int:
        """Return the number of transactions."""
        return int(self.labels.shape[0])


def _escaped(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def step_range(parquet_path: Path) -> tuple[int, int]:
    """Return minimum and maximum steps without materialising the dataset."""
    connection = duckdb.connect()
    try:
        result = connection.execute(
            f"""
            SELECT min(step)::INTEGER, max(step)::INTEGER
            FROM read_parquet('{_escaped(parquet_path)}')
            """
        ).fetchone()
    finally:
        connection.close()
    if result is None or result[0] is None or result[1] is None:
        raise ValueError(f"empty Parquet dataset: {parquet_path}")
    return int(result[0]), int(result[1])


def load_matrix(
    parquet_path: Path,
    *,
    where_sql: str = "TRUE",
) -> MatrixDataset:
    """Load the approved feature matrix with DuckDB and bounded dtypes."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} is missing; run sentinelgraph-data all first"
        )
    connection = duckdb.connect()
    try:
        columns = connection.execute(
            f"""
            SELECT
                amount::FLOAT AS amount,
                ln(1 + amount)::FLOAT AS log_amount,
                sin(2 * pi() * ((step - 1) % 24) / 24)::FLOAT AS hour_sin,
                cos(2 * pi() * ((step - 1) % 24) / 24)::FLOAT AS hour_cos,
                (type = 'CASH_IN')::UTINYINT::FLOAT AS type_cash_in,
                (type = 'CASH_OUT')::UTINYINT::FLOAT AS type_cash_out,
                (type = 'DEBIT')::UTINYINT::FLOAT AS type_debit,
                (type = 'PAYMENT')::UTINYINT::FLOAT AS type_payment,
                (type = 'TRANSFER')::UTINYINT::FLOAT AS type_transfer,
                isFraud::UTINYINT AS target,
                amount::DOUBLE AS evaluation_amount
            FROM read_parquet('{_escaped(parquet_path)}')
            WHERE {where_sql}
            ORDER BY source_row_number
            """
        ).fetchnumpy()
    finally:
        connection.close()

    feature_matrix = np.column_stack(
        [np.asarray(columns[name], dtype=np.float32) for name in FEATURE_NAMES]
    )
    labels = np.asarray(columns["target"], dtype=np.uint8)
    amounts = np.asarray(columns["evaluation_amount"], dtype=np.float64)
    return MatrixDataset(feature_matrix, labels, amounts)


def deterministic_class_cap(
    dataset: MatrixDataset,
    *,
    max_legitimate_rows: int,
    random_seed: int,
) -> MatrixDataset:
    """Keep all fraud and a deterministic sample of legitimate rows."""
    if max_legitimate_rows <= 0:
        raise ValueError("max_legitimate_rows must be positive")
    fraud_indices = np.flatnonzero(dataset.labels == 1)
    legitimate_indices = np.flatnonzero(dataset.labels == 0)
    if legitimate_indices.size <= max_legitimate_rows:
        return dataset

    generator = np.random.default_rng(random_seed)
    sampled_legitimate = generator.choice(
        legitimate_indices,
        size=max_legitimate_rows,
        replace=False,
    )
    selected = np.concatenate((fraud_indices, sampled_legitimate))
    generator.shuffle(selected)
    return MatrixDataset(
        dataset.features[selected],
        dataset.labels[selected],
        dataset.amounts[selected],
    )
