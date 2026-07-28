"""Strict point-in-time transaction-graph features for SentinelGraph v0.4."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from numpy.typing import NDArray

from sentinelgraph.data.provenance import file_record
from sentinelgraph.modeling.behaviour import (
    BEHAVIOURAL_FEATURE_NAMES,
    materialize_behavioural_features,
)
from sentinelgraph.modeling.features import FEATURE_NAMES, MatrixDataset

GRAPH_GROUPING_FIELDS = frozenset({"nameOrig", "nameDest"})
GRAPH_HISTORY_WINDOW_END = -1

NODE_GRAPH_FEATURE_NAMES = (
    "origin_log_graph_in_tx_count",
    "origin_log_graph_out_tx_count",
    "origin_log_graph_in_degree",
    "origin_log_graph_out_degree",
    "origin_log_graph_total_degree",
    "origin_graph_in_out_tx_log_ratio",
    "origin_log_graph_received_amount",
    "origin_log_graph_sent_amount",
    "origin_graph_flow_log_ratio",
    "origin_graph_prior_role_count",
    "destination_log_graph_in_tx_count",
    "destination_log_graph_out_tx_count",
    "destination_log_graph_in_degree",
    "destination_log_graph_out_degree",
    "destination_log_graph_total_degree",
    "destination_graph_in_out_tx_log_ratio",
    "destination_log_graph_received_amount",
    "destination_log_graph_sent_amount",
    "destination_graph_flow_log_ratio",
    "destination_graph_prior_role_count",
)

COMPONENT_GRAPH_FEATURE_NAMES = (
    "origin_log_component_size",
    "destination_log_component_size",
    "endpoints_same_component_prior",
    "log_combined_component_size",
    "component_size_log_ratio",
    "origin_component_is_isolated",
    "destination_component_is_isolated",
    "both_components_established",
)

GRAPH_ONLY_FEATURE_NAMES = (
    NODE_GRAPH_FEATURE_NAMES + COMPONENT_GRAPH_FEATURE_NAMES
)
GRAPH_FEATURE_NAMES = BEHAVIOURAL_FEATURE_NAMES + GRAPH_ONLY_FEATURE_NAMES
GRAPH_ONLY_MODEL_FEATURE_NAMES = FEATURE_NAMES + GRAPH_ONLY_FEATURE_NAMES


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


def _graph_topology(
    connection: duckdb.DuckDBPyConnection,
    source: str,
) -> dict[str, int]:
    account_row = connection.execute(
        f"""
        WITH account_roles AS (
            SELECT nameOrig AS account_id, 1 AS is_origin, 0 AS is_destination
            FROM {source}
            UNION ALL
            SELECT nameDest AS account_id, 0 AS is_origin, 1 AS is_destination
            FROM {source}
        ),
        accounts AS (
            SELECT
                account_id,
                max(is_origin) AS is_origin,
                max(is_destination) AS is_destination
            FROM account_roles
            GROUP BY account_id
        )
        SELECT
            count(*)::BIGINT,
            count_if(is_origin = 1)::BIGINT,
            count_if(is_destination = 1)::BIGINT,
            count_if(is_origin = 1 AND is_destination = 1)::BIGINT,
            count(DISTINCT hash(account_id))::BIGINT
        FROM accounts
        """
    ).fetchone()
    edge_row = connection.execute(
        f"""
        WITH pairs AS (
            SELECT nameOrig, nameDest, count(*)::BIGINT AS edge_count
            FROM {source}
            GROUP BY nameOrig, nameDest
        )
        SELECT
            sum(edge_count)::BIGINT,
            count(*)::BIGINT,
            count_if(edge_count > 1)::BIGINT,
            count_if(nameOrig = nameDest)::BIGINT
        FROM pairs
        """
    ).fetchone()
    reciprocal_row = connection.execute(
        f"""
        WITH pairs AS (
            SELECT DISTINCT nameOrig, nameDest
            FROM {source}
        )
        SELECT count(*)::BIGINT
        FROM pairs forward
        INNER JOIN pairs reverse
          ON forward.nameOrig = reverse.nameDest
         AND forward.nameDest = reverse.nameOrig
        WHERE forward.nameOrig < forward.nameDest
        """
    ).fetchone()
    if account_row is None or edge_row is None or reciprocal_row is None:
        raise RuntimeError("graph topology query returned no row")
    account_count = int(account_row[0])
    hash_count = int(account_row[4])
    if account_count != hash_count:
        raise RuntimeError("account hash collision detected")
    return {
        "account_count": account_count,
        "origin_account_count": int(account_row[1]),
        "destination_account_count": int(account_row[2]),
        "cross_role_account_count": int(account_row[3]),
        "account_hash_count": hash_count,
        "edge_count": int(edge_row[0]),
        "unique_directed_pair_count": int(edge_row[1]),
        "repeated_directed_pair_count": int(edge_row[2]),
        "self_loop_count": int(edge_row[3]),
        "reciprocal_pair_count": int(reciprocal_row[0]),
    }


def _materialize_dense_edges(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    destination: Path,
) -> tuple[int, int]:
    escaped_destination = _escaped(destination)
    if destination.exists():
        destination.unlink()
    connection.execute(
        f"""
        COPY (
            WITH base AS (
                SELECT
                    source_row_number::BIGINT AS source_row_number,
                    step::INTEGER AS step,
                    hash(nameOrig)::UBIGINT AS origin_hash,
                    hash(nameDest)::UBIGINT AS destination_hash
                FROM {source}
            ),
            node_hashes AS (
                SELECT origin_hash AS account_hash FROM base
                UNION
                SELECT destination_hash AS account_hash FROM base
            ),
            nodes AS (
                SELECT
                    account_hash,
                    (
                        row_number() OVER (ORDER BY account_hash) - 1
                    )::UINTEGER AS node_id
                FROM node_hashes
            )
            SELECT
                base.source_row_number,
                base.step,
                origin.node_id AS origin_node_id,
                destination.node_id AS destination_node_id
            FROM base
            INNER JOIN nodes origin
                ON base.origin_hash = origin.account_hash
            INNER JOIN nodes destination
                ON base.destination_hash = destination.account_hash
            ORDER BY base.source_row_number
        )
        TO '{escaped_destination}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    row = connection.execute(
        f"""
        SELECT
            count(*)::BIGINT,
            1 + greatest(
                max(origin_node_id)::BIGINT,
                max(destination_node_id)::BIGINT
            )
        FROM read_parquet('{escaped_destination}')
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("dense graph edge query returned no row")
    return int(row[0]), int(row[1])


def _find_root(parent: NDArray[np.int32], node: int) -> int:
    while int(parent[node]) != node:
        parent[node] = parent[int(parent[node])]
        node = int(parent[node])
    return node


def _component_features(
    edge_path: Path,
    component_path: Path,
    *,
    node_count: int,
) -> dict[str, Any]:
    connection = duckdb.connect()
    try:
        columns = connection.execute(
            f"""
            SELECT
                source_row_number,
                step,
                origin_node_id,
                destination_node_id
            FROM read_parquet('{_escaped(edge_path)}')
            ORDER BY source_row_number
            """
        ).fetchnumpy()
    finally:
        connection.close()

    source_rows = np.asarray(columns["source_row_number"], dtype=np.int64)
    expected_rows = np.arange(1, source_rows.size + 1, dtype=np.int64)
    if not np.array_equal(source_rows, expected_rows):
        raise ValueError("source_row_number must be contiguous and one-based")
    steps = np.asarray(columns["step"], dtype=np.int32)
    if np.any(steps[1:] < steps[:-1]):
        raise ValueError("graph edges must be ordered chronologically")
    origins = np.asarray(columns["origin_node_id"], dtype=np.int32)
    destinations = np.asarray(columns["destination_node_id"], dtype=np.int32)

    parent = np.arange(node_count, dtype=np.int32)
    sizes = np.ones(node_count, dtype=np.int32)
    origin_sizes = np.empty(source_rows.size, dtype=np.int32)
    destination_sizes = np.empty(source_rows.size, dtype=np.int32)
    same_component = np.empty(source_rows.size, dtype=np.uint8)
    combined_sizes = np.empty(source_rows.size, dtype=np.int32)

    start = 0
    while start < source_rows.size:
        end = start + 1
        current_step = int(steps[start])
        while end < source_rows.size and int(steps[end]) == current_step:
            end += 1

        for index in range(start, end):
            origin_root = _find_root(parent, int(origins[index]))
            destination_root = _find_root(parent, int(destinations[index]))
            origin_size = int(sizes[origin_root])
            destination_size = int(sizes[destination_root])
            is_same = origin_root == destination_root
            origin_sizes[index] = origin_size
            destination_sizes[index] = destination_size
            same_component[index] = int(is_same)
            combined_sizes[index] = (
                origin_size if is_same else origin_size + destination_size
            )

        for index in range(start, end):
            origin_root = _find_root(parent, int(origins[index]))
            destination_root = _find_root(parent, int(destinations[index]))
            if origin_root == destination_root:
                continue
            if int(sizes[origin_root]) < int(sizes[destination_root]):
                origin_root, destination_root = destination_root, origin_root
            parent[destination_root] = origin_root
            sizes[origin_root] += sizes[destination_root]
        start = end

    component_count = int(np.count_nonzero(parent == np.arange(node_count)))
    largest_component_size = int(sizes.max(initial=0))
    cycle_rank = int(source_rows.size - node_count + component_count)
    component_path.parent.mkdir(parents=True, exist_ok=True)
    if component_path.exists():
        component_path.unlink()
    np.savez_compressed(
        component_path,
        origin_component_size=origin_sizes,
        destination_component_size=destination_sizes,
        endpoints_same_component_prior=same_component,
        combined_component_size=combined_sizes,
    )
    return {
        "node_count": node_count,
        "component_count": component_count,
        "largest_component_size": largest_component_size,
        "cycle_rank": cycle_rank,
        "strict_past_cycle_closure_rows": int(same_component.sum()),
    }


def _node_feature_query(source_paths: Sequence[Path]) -> str:
    source = _source_sql(source_paths)
    return f"""
        WITH base AS (
            SELECT
                source_row_number::BIGINT AS source_row_number,
                step::INTEGER AS step,
                amount::DOUBLE AS amount,
                nameOrig::VARCHAR AS nameOrig,
                nameDest::VARCHAR AS nameDest
            FROM {source}
        ),
        node_events AS (
            SELECT
                source_row_number,
                step,
                nameOrig AS account_id,
                nameDest AS counterparty_id,
                amount,
                1::UTINYINT AS endpoint_is_origin,
                1::UTINYINT AS direction_is_out
            FROM base
            UNION ALL
            SELECT
                source_row_number,
                step,
                nameDest AS account_id,
                nameOrig AS counterparty_id,
                amount,
                0::UTINYINT AS endpoint_is_origin,
                0::UTINYINT AS direction_is_out
            FROM base
        ),
        history AS (
            SELECT
                *,
                count(*) FILTER (WHERE direction_is_out = 0) OVER (
                    PARTITION BY account_id ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS prior_in_count,
                count(*) FILTER (WHERE direction_is_out = 1) OVER (
                    PARTITION BY account_id ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS prior_out_count,
                count(DISTINCT counterparty_id)
                    FILTER (WHERE direction_is_out = 0) OVER (
                        PARTITION BY account_id ORDER BY step
                        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    ) AS prior_in_degree,
                count(DISTINCT counterparty_id)
                    FILTER (WHERE direction_is_out = 1) OVER (
                        PARTITION BY account_id ORDER BY step
                        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    ) AS prior_out_degree,
                sum(amount) FILTER (WHERE direction_is_out = 0) OVER (
                    PARTITION BY account_id ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS prior_received_amount,
                sum(amount) FILTER (WHERE direction_is_out = 1) OVER (
                    PARTITION BY account_id ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS prior_sent_amount
            FROM node_events
        ),
        endpoint_history AS (
            SELECT
                source_row_number,
                max(prior_in_count) FILTER (
                    WHERE endpoint_is_origin = 1
                ) AS origin_in_count,
                max(prior_out_count) FILTER (
                    WHERE endpoint_is_origin = 1
                ) AS origin_out_count,
                max(prior_in_degree) FILTER (
                    WHERE endpoint_is_origin = 1
                ) AS origin_in_degree,
                max(prior_out_degree) FILTER (
                    WHERE endpoint_is_origin = 1
                ) AS origin_out_degree,
                max(prior_received_amount) FILTER (
                    WHERE endpoint_is_origin = 1
                ) AS origin_received_amount,
                max(prior_sent_amount) FILTER (
                    WHERE endpoint_is_origin = 1
                ) AS origin_sent_amount,
                max(prior_in_count) FILTER (
                    WHERE endpoint_is_origin = 0
                ) AS destination_in_count,
                max(prior_out_count) FILTER (
                    WHERE endpoint_is_origin = 0
                ) AS destination_out_count,
                max(prior_in_degree) FILTER (
                    WHERE endpoint_is_origin = 0
                ) AS destination_in_degree,
                max(prior_out_degree) FILTER (
                    WHERE endpoint_is_origin = 0
                ) AS destination_out_degree,
                max(prior_received_amount) FILTER (
                    WHERE endpoint_is_origin = 0
                ) AS destination_received_amount,
                max(prior_sent_amount) FILTER (
                    WHERE endpoint_is_origin = 0
                ) AS destination_sent_amount
            FROM history
            GROUP BY source_row_number
        )
        SELECT
            source_row_number,
            ln(1 + origin_in_count)::FLOAT
                AS origin_log_graph_in_tx_count,
            ln(1 + origin_out_count)::FLOAT
                AS origin_log_graph_out_tx_count,
            ln(1 + origin_in_degree)::FLOAT
                AS origin_log_graph_in_degree,
            ln(1 + origin_out_degree)::FLOAT
                AS origin_log_graph_out_degree,
            ln(1 + origin_in_degree + origin_out_degree)::FLOAT
                AS origin_log_graph_total_degree,
            ln((1 + origin_in_count)::DOUBLE / (1 + origin_out_count))::FLOAT
                AS origin_graph_in_out_tx_log_ratio,
            ln(1 + coalesce(origin_received_amount, 0))::FLOAT
                AS origin_log_graph_received_amount,
            ln(1 + coalesce(origin_sent_amount, 0))::FLOAT
                AS origin_log_graph_sent_amount,
            ln(
                (1 + coalesce(origin_received_amount, 0))
                / (1 + coalesce(origin_sent_amount, 0))
            )::FLOAT AS origin_graph_flow_log_ratio,
            (
                (origin_in_count > 0)::UTINYINT
                + (origin_out_count > 0)::UTINYINT
            )::FLOAT AS origin_graph_prior_role_count,
            ln(1 + destination_in_count)::FLOAT
                AS destination_log_graph_in_tx_count,
            ln(1 + destination_out_count)::FLOAT
                AS destination_log_graph_out_tx_count,
            ln(1 + destination_in_degree)::FLOAT
                AS destination_log_graph_in_degree,
            ln(1 + destination_out_degree)::FLOAT
                AS destination_log_graph_out_degree,
            ln(1 + destination_in_degree + destination_out_degree)::FLOAT
                AS destination_log_graph_total_degree,
            ln(
                (1 + destination_in_count)::DOUBLE
                / (1 + destination_out_count)
            )::FLOAT AS destination_graph_in_out_tx_log_ratio,
            ln(1 + coalesce(destination_received_amount, 0))::FLOAT
                AS destination_log_graph_received_amount,
            ln(1 + coalesce(destination_sent_amount, 0))::FLOAT
                AS destination_log_graph_sent_amount,
            ln(
                (1 + coalesce(destination_received_amount, 0))
                / (1 + coalesce(destination_sent_amount, 0))
            )::FLOAT AS destination_graph_flow_log_ratio,
            (
                (destination_in_count > 0)::UTINYINT
                + (destination_out_count > 0)::UTINYINT
            )::FLOAT AS destination_graph_prior_role_count
        FROM endpoint_history
    """


def _materialize_node_features(
    connection: duckdb.DuckDBPyConnection,
    source_paths: Sequence[Path],
    behavioural_path: Path,
    destination: Path,
) -> None:
    if not behavioural_path.exists():
        materialize_behavioural_features(
            source_paths,
            behavioural_path,
            relative_to=behavioural_path.parents[2],
        )
    if destination.exists():
        destination.unlink()
    escaped_behavioural = _escaped(behavioural_path)
    escaped_destination = _escaped(destination)
    node_columns = ",\n".join(
        f"node.{name}" for name in NODE_GRAPH_FEATURE_NAMES
    )
    connection.execute(
        f"""
        COPY (
            WITH node AS (
                {_node_feature_query(source_paths)}
            )
            SELECT
                behavioural.*,
                {node_columns}
            FROM read_parquet('{escaped_behavioural}') behavioural
            INNER JOIN node USING (source_row_number)
            ORDER BY behavioural.source_row_number
        )
        TO '{escaped_destination}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def inspect_graph_feature_store(
    graph_path: Path,
    component_path: Path,
    *,
    relative_to: Path,
    topology: dict[str, int],
    component_statistics: dict[str, Any],
) -> dict[str, Any]:
    """Return a reproducibility manifest for v0.4 graph artifacts."""
    connection = duckdb.connect()
    try:
        row = connection.execute(
            f"""
            SELECT
                count(*)::BIGINT,
                min(step)::INTEGER,
                max(step)::INTEGER,
                count_if(target = 1)::BIGINT
            FROM read_parquet('{_escaped(graph_path)}')
            """
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("graph feature statistics query returned no row")
    return {
        "node_feature_artifact": file_record(
            graph_path,
            relative_to=relative_to,
        ),
        "component_feature_artifact": file_record(
            component_path,
            relative_to=relative_to,
        ),
        "rows": int(row[0]),
        "minimum_step": int(row[1]),
        "maximum_step": int(row[2]),
        "fraud_rows": int(row[3]),
        "feature_count": len(GRAPH_FEATURE_NAMES),
        "feature_names": list(GRAPH_FEATURE_NAMES),
        "graph_only_feature_names": list(GRAPH_ONLY_FEATURE_NAMES),
        "history_contract": {
            "window_upper_bound_steps": GRAPH_HISTORY_WINDOW_END,
            "same_step_edges_excluded": True,
            "labels_used_in_graph_features": False,
            "identifiers_used_only_as_grouping_keys": sorted(
                GRAPH_GROUPING_FIELDS
            ),
            "identifiers_emitted_as_features": False,
        },
        "topology": topology,
        "components": component_statistics,
        "label_exposure_policy": (
            "suspicious-neighbour label propagation is excluded because "
            "PaySim has no fraud-confirmation timestamp"
        ),
    }


def materialize_graph_features(
    source_paths: Sequence[Path],
    behavioural_path: Path,
    graph_path: Path,
    component_path: Path,
    edge_path: Path,
    *,
    relative_to: Path,
) -> dict[str, Any]:
    """Build graph node and component features using only prior steps."""
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    source = _source_sql(source_paths)
    connection = duckdb.connect()
    try:
        connection.execute("SET preserve_insertion_order = false")
        topology = _graph_topology(connection, source)
        edge_count, node_count = _materialize_dense_edges(
            connection,
            source,
            edge_path,
        )
        if edge_count != topology["edge_count"]:
            raise RuntimeError("dense edge artifact lost graph rows")
        _materialize_node_features(
            connection,
            source_paths,
            behavioural_path,
            graph_path,
        )
    finally:
        connection.close()

    component_statistics = _component_features(
        edge_path,
        component_path,
        node_count=node_count,
    )
    return inspect_graph_feature_store(
        graph_path,
        component_path,
        relative_to=relative_to,
        topology=topology,
        component_statistics=component_statistics,
    )


def _component_columns(
    component_path: Path,
    source_rows: NDArray[np.int64],
) -> dict[str, NDArray[np.float32]]:
    indexes = source_rows - 1
    if np.any(indexes < 0):
        raise ValueError("source_row_number must be positive")
    with np.load(component_path) as arrays:
        origin_all = np.asarray(arrays["origin_component_size"], dtype=np.int32)
        destination_all = np.asarray(
            arrays["destination_component_size"],
            dtype=np.int32,
        )
        same_all = np.asarray(
            arrays["endpoints_same_component_prior"],
            dtype=np.uint8,
        )
        combined_all = np.asarray(
            arrays["combined_component_size"],
            dtype=np.int32,
        )
        if indexes.size and int(indexes.max()) >= origin_all.size:
            raise ValueError("component artifact is not aligned to source rows")
        origin = origin_all[indexes]
        destination = destination_all[indexes]
        same = same_all[indexes]
        combined = combined_all[indexes]

    origin_log = np.log1p(origin).astype(np.float32)
    destination_log = np.log1p(destination).astype(np.float32)
    return {
        "origin_log_component_size": origin_log,
        "destination_log_component_size": destination_log,
        "endpoints_same_component_prior": same.astype(np.float32),
        "log_combined_component_size": np.log1p(combined).astype(np.float32),
        "component_size_log_ratio": (origin_log - destination_log).astype(
            np.float32
        ),
        "origin_component_is_isolated": (origin == 1).astype(np.float32),
        "destination_component_is_isolated": (destination == 1).astype(
            np.float32
        ),
        "both_components_established": (
            (origin > 1) & (destination > 1)
        ).astype(np.float32),
    }


def load_graph_matrix(
    graph_path: Path,
    component_path: Path,
    *,
    feature_names: Sequence[str] = GRAPH_FEATURE_NAMES,
    where_sql: str = "TRUE",
    max_legitimate_rows: int | None = None,
    random_seed: int = 42,
) -> MatrixDataset:
    """Load an aligned behavioural/graph matrix with optional class capping."""
    if not graph_path.exists() or not component_path.exists():
        raise FileNotFoundError("graph feature artifacts are missing")
    unknown = set(feature_names) - set(GRAPH_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown graph feature: {sorted(unknown)[0]}")
    if max_legitimate_rows is not None and max_legitimate_rows <= 0:
        raise ValueError("max_legitimate_rows must be positive")

    sql_feature_names = [
        name for name in feature_names if name not in COMPONENT_GRAPH_FEATURE_NAMES
    ]
    selected_columns = ",\n".join(sql_feature_names)
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
                FROM read_parquet('{_escaped(graph_path)}')
                WHERE {where_sql}
            ),
            selected AS (
                {selected_sql}
            )
            SELECT
                source_row_number,
                {selected_columns},
                target::UTINYINT AS target,
                evaluation_amount::DOUBLE AS evaluation_amount
            FROM selected
            ORDER BY source_row_number
            """
        ).fetchnumpy()
    finally:
        connection.close()

    source_rows = np.asarray(columns["source_row_number"], dtype=np.int64)
    component_columns = _component_columns(component_path, source_rows)
    matrix_columns = []
    for name in feature_names:
        if name in component_columns:
            matrix_columns.append(component_columns[name])
        else:
            matrix_columns.append(np.asarray(columns[name], dtype=np.float32))
    feature_matrix = np.column_stack(matrix_columns).astype(
        np.float32,
        copy=False,
    )
    labels = np.asarray(columns["target"], dtype=np.uint8)
    amounts = np.asarray(columns["evaluation_amount"], dtype=np.float64)
    return MatrixDataset(feature_matrix, labels, amounts)
