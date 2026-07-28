"""Train and evaluate SentinelGraph v0.4 graph challengers."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import joblib

from sentinelgraph.data.provenance import file_record
from sentinelgraph.data.splits import temporal_cutoff
from sentinelgraph.modeling.anomaly import build_behavioural_gradient_boosting
from sentinelgraph.modeling.behaviour import BEHAVIOURAL_FEATURE_NAMES
from sentinelgraph.modeling.features import (
    PROHIBITED_MODEL_FIELDS,
    MatrixDataset,
    step_range,
)
from sentinelgraph.modeling.graph import (
    GRAPH_FEATURE_NAMES,
    GRAPH_ONLY_FEATURE_NAMES,
    GRAPH_ONLY_MODEL_FEATURE_NAMES,
    load_graph_matrix,
    materialize_graph_features,
)
from sentinelgraph.modeling.graph_report import write_graph_report
from sentinelgraph.modeling.metrics import (
    evaluate_scores,
    positive_scores,
    threshold_at_fpr,
)
from sentinelgraph.modeling.models import RANDOM_SEED

VALIDATION_FRACTION_BY_TIME = 0.20
TARGET_MAXIMUM_FPR = 0.01
MAXIMUM_LEGITIMATE_ROWS = 999_999
MINIMUM_GRAPH_VALIDATION_AP_GAIN = 0.005


def default_project_root() -> Path:
    """Resolve the repository root from this installed source tree."""
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fit_model(model: Any, dataset: MatrixDataset) -> tuple[Any, float]:
    started = time.perf_counter()
    model.fit(dataset.features, dataset.labels)
    return model, time.perf_counter() - started


def _subset(
    dataset: MatrixDataset,
    feature_names: Sequence[str],
) -> MatrixDataset:
    indexes = [GRAPH_FEATURE_NAMES.index(name) for name in feature_names]
    return MatrixDataset(
        dataset.features[:, indexes],
        dataset.labels,
        dataset.amounts,
    )


def _load_v03_benchmark(project_root: Path) -> dict[str, Any]:
    path = project_root / "reports" / "v0.3" / "behavioural_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing v0.3 benchmark metrics: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    selected = payload["selected_model"]
    benchmark = payload["models"][selected]["evaluation"]["future_time_holdout"]
    return cast(dict[str, Any], benchmark)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            "graph manifest is missing; rebuild the graph feature store"
        )
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def train_graph_models(
    project_root: Path,
    *,
    maximum_fpr: float = TARGET_MAXIMUM_FPR,
    maximum_legitimate_rows: int = MAXIMUM_LEGITIMATE_ROWS,
    rebuild_features: bool = True,
) -> dict[str, Any]:
    """Build v0.4 graph features and compare graph challengers."""
    processed_dir = project_root / "data" / "processed"
    interim_dir = project_root / "data" / "interim"
    train_path = processed_dir / "train.parquet"
    future_path = processed_dir / "future_time_holdout.parquet"
    new_account_path = processed_dir / "new_account_holdout.parquet"
    source_paths = (train_path, future_path)
    for path in (*source_paths, new_account_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run sentinelgraph-data all first"
            )

    behavioural_path = processed_dir / "behavioural_features.parquet"
    graph_path = processed_dir / "graph_features.parquet"
    component_path = processed_dir / "graph_components.npz"
    edge_path = interim_dir / "graph_edges.parquet"
    manifest_path = (
        project_root / "data" / "metadata" / "graph_feature_manifest.json"
    )

    print("[1/6] Building strict point-in-time graph features", flush=True)
    artifacts_exist = all(
        path.exists() for path in (graph_path, component_path, edge_path)
    )
    if rebuild_features or not artifacts_exist:
        feature_manifest = materialize_graph_features(
            source_paths,
            behavioural_path,
            graph_path,
            component_path,
            edge_path,
            relative_to=project_root,
        )
        _write_json(manifest_path, feature_manifest)
    else:
        feature_manifest = _load_manifest(manifest_path)

    train_min_step, train_max_step = step_range(train_path)
    development_end_step = temporal_cutoff(
        train_min_step,
        train_max_step,
        1.0 - VALIDATION_FRACTION_BY_TIME,
    )
    print("[2/6] Loading aligned behavioural and graph matrices", flush=True)
    development = load_graph_matrix(
        graph_path,
        component_path,
        where_sql=f"step <= {development_end_step}",
        max_legitimate_rows=maximum_legitimate_rows,
        random_seed=RANDOM_SEED,
    )
    validation = load_graph_matrix(
        graph_path,
        component_path,
        where_sql=(
            f"step > {development_end_step} AND step <= {train_max_step}"
        ),
    )

    feature_sets: dict[str, Sequence[str]] = {
        "behavioural_reference": BEHAVIOURAL_FEATURE_NAMES,
        "graph_only_hist_gradient_boosting": GRAPH_ONLY_MODEL_FEATURE_NAMES,
        "behavioural_graph_hist_gradient_boosting": GRAPH_FEATURE_NAMES,
    }
    models_dir = project_root / "models" / "v0.4"
    models_dir.mkdir(parents=True, exist_ok=True)
    fitted: dict[str, Any] = {}
    model_results: dict[str, Any] = {}

    print("[3/6] Fitting reference, graph-only, and combined models", flush=True)
    for name, names in feature_sets.items():
        fit_dataset = _subset(development, names)
        validation_dataset = _subset(validation, names)
        fitted_model, fit_seconds = _fit_model(
            build_behavioural_gradient_boosting(),
            fit_dataset,
        )
        print(f"  - {name}: {fit_seconds:.2f} s", flush=True)
        validation_scores = positive_scores(
            fitted_model,
            validation_dataset.features,
        )
        threshold = threshold_at_fpr(
            validation_dataset.labels,
            validation_scores,
            maximum_fpr=maximum_fpr,
        )
        artifact_path = models_dir / f"{name}.joblib"
        joblib.dump(fitted_model, artifact_path, compress=3)
        fitted[name] = fitted_model
        model_results[name] = {
            "feature_count": len(names),
            "feature_names": list(names),
            "fit_rows": fit_dataset.rows,
            "fit_fraud_rows": int(fit_dataset.labels.sum()),
            "threshold": threshold,
            "threshold_strategy": "validation_fpr_budget",
            "artifact": file_record(artifact_path, relative_to=project_root),
            "evaluation": {
                "validation": evaluate_scores(
                    validation_dataset.labels,
                    validation_scores,
                    validation_dataset.amounts,
                    threshold=threshold,
                )
            },
        }

    reference_name = "behavioural_reference"
    graph_candidate_names = (
        "graph_only_hist_gradient_boosting",
        "behavioural_graph_hist_gradient_boosting",
    )
    graph_name = max(
        graph_candidate_names,
        key=lambda name: model_results[name]["evaluation"]["validation"][
            "average_precision"
        ],
    )
    reference_validation_ap = model_results[reference_name]["evaluation"][
        "validation"
    ]["average_precision"]
    graph_validation_ap = model_results[graph_name]["evaluation"]["validation"][
        "average_precision"
    ]
    graph_validation_gain = graph_validation_ap - reference_validation_ap
    validation_hurdle_passed = (
        graph_validation_gain >= MINIMUM_GRAPH_VALIDATION_AP_GAIN
    )
    validation_selected_model = (
        graph_name if validation_hurdle_passed else reference_name
    )

    print("[4/6] Loading untouched future and new-account views", flush=True)
    escaped_new_account = str(new_account_path.resolve()).replace("'", "''")
    future = load_graph_matrix(
        graph_path,
        component_path,
        where_sql=f"step > {train_max_step}",
    )
    new_account = load_graph_matrix(
        graph_path,
        component_path,
        where_sql=(
            "source_row_number IN ("
            "SELECT source_row_number FROM read_parquet("
            f"'{escaped_new_account}'))"
        ),
    )
    evaluation_sets = {
        "future_time_holdout": future,
        "new_account_holdout": new_account,
    }
    print("[5/6] Evaluating final graph holdouts", flush=True)
    for name, model in fitted.items():
        threshold = float(model_results[name]["threshold"])
        names = feature_sets[name]
        for dataset_name, dataset in evaluation_sets.items():
            selected_dataset = _subset(dataset, names)
            scores = positive_scores(model, selected_dataset.features)
            model_results[name]["evaluation"][dataset_name] = evaluate_scores(
                selected_dataset.labels,
                scores,
                selected_dataset.amounts,
                threshold=threshold,
            )

    reference_future = model_results[reference_name]["evaluation"][
        "future_time_holdout"
    ]
    graph_future = model_results[graph_name]["evaluation"][
        "future_time_holdout"
    ]
    future_average_precision_improved = (
        graph_future["average_precision"] > reference_future["average_precision"]
    )
    future_fpr_within_budget = (
        graph_future["false_positive_rate"] <= maximum_fpr
    )
    graph_kept = (
        validation_hurdle_passed
        and future_average_precision_improved
        and future_fpr_within_budget
    )
    release_champion = graph_name if graph_kept else reference_name

    benchmark = _load_v03_benchmark(project_root)
    champion_future = model_results[release_champion]["evaluation"][
        "future_time_holdout"
    ]
    comparison = {
        "baseline_release": "v0.3",
        "baseline_model": "behavioural_hist_gradient_boosting",
        "baseline_average_precision": benchmark["average_precision"],
        "baseline_recall": benchmark["recall"],
        "baseline_false_positives_per_10k": benchmark[
            "false_positives_per_10k_legitimate"
        ],
        "baseline_captured_fraud_amount_rate": benchmark[
            "captured_fraud_amount_rate"
        ],
        "average_precision_delta": (
            champion_future["average_precision"]
            - benchmark["average_precision"]
        ),
        "recall_delta": champion_future["recall"] - benchmark["recall"],
        "false_positives_per_10k_delta": (
            champion_future["false_positives_per_10k_legitimate"]
            - benchmark["false_positives_per_10k_legitimate"]
        ),
        "captured_fraud_amount_rate_delta": (
            champion_future["captured_fraud_amount_rate"]
            - benchmark["captured_fraud_amount_rate"]
        ),
    }

    topology = feature_manifest["topology"]
    graph_sparse = (
        topology["repeated_directed_pair_count"] == 0
        and topology["reciprocal_pair_count"] == 0
    )
    if graph_kept and not graph_sparse:
        graphsage_status = "eligible for a later controlled experiment"
        graphsage_rationale = (
            "The non-GNN graph challenger passed its promotion gate and the "
            "graph has repeated relational structure."
        )
    else:
        graphsage_status = "not retained"
        graphsage_rationale = (
            "GraphSAGE was not added: every directed account pair occurs once, "
            "there are no reciprocal pairs, and the non-GNN graph experiment "
            "must first demonstrate material incremental value. A GNN would "
            "add cost without a defensible message-passing advantage."
        )

    future_min_step, future_max_step = step_range(future_path)
    results = {
        "release": "v0.4",
        "random_seed": RANDOM_SEED,
        "feature_store": feature_manifest,
        "feature_names": list(GRAPH_FEATURE_NAMES),
        "behavioural_feature_count": len(BEHAVIOURAL_FEATURE_NAMES),
        "graph_only_feature_count": len(GRAPH_ONLY_FEATURE_NAMES),
        "combined_feature_count": len(GRAPH_FEATURE_NAMES),
        "prohibited_model_fields": sorted(PROHIBITED_MODEL_FIELDS),
        "target_maximum_fpr": maximum_fpr,
        "maximum_legitimate_rows": maximum_legitimate_rows,
        "development_split": {
            "development_min_step": train_min_step,
            "development_max_step": development_end_step,
            "validation_min_step": development_end_step + 1,
            "validation_max_step": train_max_step,
            "future_min_step": future_min_step,
            "future_max_step": future_max_step,
        },
        "selection_policy": (
            "graph augmentation must improve temporal-validation average "
            f"precision by at least {MINIMUM_GRAPH_VALIDATION_AP_GAIN:.3f}, "
            "then improve future average precision within the FPR budget"
        ),
        "selection_evidence": {
            "behavioural_validation_average_precision": (
                reference_validation_ap
            ),
            "graph_validation_average_precision": graph_validation_ap,
            "graph_validation_average_precision_gain": graph_validation_gain,
            "best_graph_validation_candidate": graph_name,
            "minimum_required_gain": MINIMUM_GRAPH_VALIDATION_AP_GAIN,
            "validation_selected_model": validation_selected_model,
        },
        "models": model_results,
        "graph_keep_decision": {
            "kept": graph_kept,
            "validation_hurdle_passed": validation_hurdle_passed,
            "future_average_precision_improved": (
                future_average_precision_improved
            ),
            "future_fpr_within_budget": future_fpr_within_budget,
        },
        "release_champion": release_champion,
        "comparison_to_v0_3": comparison,
        "graphsage_decision": {
            "status": graphsage_status,
            "rationale": graphsage_rationale,
        },
    }
    metrics_path = project_root / "reports" / "v0.4" / "graph_metrics.json"
    _write_json(metrics_path, results)
    print("[6/6] Writing graph metrics and report", flush=True)
    write_graph_report(
        project_root / "docs" / "GRAPH_REPORT.md",
        results,
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the v0.4 graph experiment CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train SentinelGraph v0.4 graph challengers"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root(),
    )
    parser.add_argument(
        "--maximum-fpr",
        type=float,
        default=TARGET_MAXIMUM_FPR,
    )
    parser.add_argument(
        "--maximum-legitimate-rows",
        type=int,
        default=MAXIMUM_LEGITIMATE_ROWS,
    )
    parser.add_argument(
        "--reuse-feature-store",
        action="store_true",
        help="reuse ignored graph artifacts and their tracked manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v0.4 graph experiment from the command line."""
    args = build_parser().parse_args(argv)
    train_graph_models(
        args.project_root.resolve(),
        maximum_fpr=args.maximum_fpr,
        maximum_legitimate_rows=args.maximum_legitimate_rows,
        rebuild_features=not args.reuse_feature_store,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
