"""Tests for strict point-in-time v0.4 graph features."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from sentinelgraph.modeling.behaviour import materialize_behavioural_features
from sentinelgraph.modeling.graph import (
    GRAPH_FEATURE_NAMES,
    load_graph_matrix,
    materialize_graph_features,
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
                (2, 2, 'TRANSFER', 20.0, 'B', 'X', 0),
                (3, 2, 'PAYMENT', 30.0, 'X', 'C', 1),
                (4, 3, 'TRANSFER', 40.0, 'X', 'D', 0),
                (5, 4, 'CASH_OUT', 50.0, 'D', 'A', 1)
            """
        )
        escaped = str(path).replace("'", "''")
        connection.execute(f"COPY fixture TO '{escaped}' (FORMAT PARQUET)")
    finally:
        connection.close()


def _artifacts(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    source = tmp_path / "source.parquet"
    behavioural = tmp_path / "behavioural.parquet"
    graph = tmp_path / "graph.parquet"
    components = tmp_path / "components.npz"
    edges = tmp_path / "edges.parquet"
    _write_transactions(source)
    materialize_behavioural_features((source,), behavioural)
    manifest = materialize_graph_features(
        (source,),
        behavioural,
        graph,
        components,
        edges,
        relative_to=tmp_path,
    )
    return graph, components, manifest


def test_graph_history_excludes_same_step_edges(tmp_path: Path) -> None:
    graph_path, component_path, _ = _artifacts(tmp_path)
    dataset = load_graph_matrix(graph_path, component_path)
    indexes = {name: index for index, name in enumerate(GRAPH_FEATURE_NAMES)}

    same_step_destination = dataset.features[1]
    same_step_origin = dataset.features[2]
    later_origin = dataset.features[3]

    assert np.isclose(
        same_step_destination[
            indexes["destination_log_graph_in_tx_count"]
        ],
        np.log(2),
    )
    assert np.isclose(
        same_step_origin[indexes["origin_log_graph_in_tx_count"]],
        np.log(2),
    )
    assert same_step_origin[indexes["origin_log_graph_out_tx_count"]] == 0.0
    assert np.isclose(
        later_origin[indexes["origin_log_graph_in_tx_count"]],
        np.log(3),
    )
    assert np.isclose(
        later_origin[indexes["origin_log_graph_out_tx_count"]],
        np.log(2),
    )


def test_components_are_strictly_prior_to_current_step(tmp_path: Path) -> None:
    graph_path, component_path, manifest = _artifacts(tmp_path)
    dataset = load_graph_matrix(graph_path, component_path)
    indexes = {name: index for index, name in enumerate(GRAPH_FEATURE_NAMES)}

    second = dataset.features[1]
    third = dataset.features[2]
    fourth = dataset.features[3]
    cycle_closure = dataset.features[4]

    assert np.isclose(
        second[indexes["destination_log_component_size"]],
        np.log(3),
    )
    assert np.isclose(
        third[indexes["origin_log_component_size"]],
        np.log(3),
    )
    assert np.isclose(
        fourth[indexes["origin_log_component_size"]],
        np.log(5),
    )
    assert cycle_closure[indexes["endpoints_same_component_prior"]] == 1.0

    topology = manifest["topology"]
    components = manifest["components"]
    assert isinstance(topology, dict)
    assert isinstance(components, dict)
    assert topology["account_count"] == 5
    assert topology["repeated_directed_pair_count"] == 0
    assert components["component_count"] == 1
    assert components["cycle_rank"] == 1
    assert components["strict_past_cycle_closure_rows"] == 1


def test_graph_store_never_emits_account_identifiers(tmp_path: Path) -> None:
    graph_path, _, manifest = _artifacts(tmp_path)
    connection = duckdb.connect()
    try:
        columns = {
            row[0]
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{graph_path}')"
            ).fetchall()
        }
    finally:
        connection.close()

    assert "nameOrig" not in columns
    assert "nameDest" not in columns
    assert not {"nameOrig", "nameDest"} & set(GRAPH_FEATURE_NAMES)
    history = manifest["history_contract"]
    assert isinstance(history, dict)
    assert history["labels_used_in_graph_features"] is False


def test_graph_loader_cap_is_deterministic_and_keeps_fraud(
    tmp_path: Path,
) -> None:
    graph_path, component_path, _ = _artifacts(tmp_path)

    first = load_graph_matrix(
        graph_path,
        component_path,
        max_legitimate_rows=2,
        random_seed=42,
    )
    second = load_graph_matrix(
        graph_path,
        component_path,
        max_legitimate_rows=2,
        random_seed=42,
    )

    assert first.rows == 4
    assert int(first.labels.sum()) == 2
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.labels, second.labels)
