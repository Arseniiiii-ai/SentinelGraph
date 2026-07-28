"""Train and evaluate SentinelGraph v0.3 behavioural challengers."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from numpy.typing import NDArray

from sentinelgraph.data.provenance import file_record
from sentinelgraph.data.splits import temporal_cutoff
from sentinelgraph.modeling.advanced_report import write_behavioural_report
from sentinelgraph.modeling.anomaly import (
    AnomalyAugmentedClassifier,
    IsolationForestDetector,
    build_behavioural_gradient_boosting,
)
from sentinelgraph.modeling.behaviour import (
    BEHAVIOURAL_FEATURE_NAMES,
    BEHAVIOURAL_ONLY_FEATURE_NAMES,
    inspect_behavioural_feature_store,
    load_behavioural_matrix,
    materialize_behavioural_features,
)
from sentinelgraph.modeling.features import (
    PROHIBITED_MODEL_FIELDS,
    MatrixDataset,
    step_range,
)
from sentinelgraph.modeling.metrics import (
    evaluate_scores,
    positive_scores,
    threshold_at_fpr,
)
from sentinelgraph.modeling.models import RANDOM_SEED

VALIDATION_FRACTION_BY_TIME = 0.20
TARGET_MAXIMUM_FPR = 0.01
# DuckDB's windowed Top-N implementation requires n to be strictly below 1M.
MAXIMUM_LEGITIMATE_ROWS = 999_999
V02_BENCHMARK_MODEL = "hist_gradient_boosting"
MINIMUM_ANOMALY_VALIDATION_AP_GAIN = 0.005


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


def _escaped(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _metrics_for_mask(
    dataset: MatrixDataset,
    scores: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    threshold: float,
) -> dict[str, Any] | None:
    if not np.any(mask):
        return None
    labels = dataset.labels[mask]
    if np.unique(labels).size < 2:
        return None
    return evaluate_scores(
        labels,
        scores[mask],
        dataset.amounts[mask],
        threshold=threshold,
    )


def _future_slices(
    dataset: MatrixDataset,
    scores: NDArray[np.float64],
    *,
    threshold: float,
) -> dict[str, Any]:
    indexes = {name: index for index, name in enumerate(BEHAVIOURAL_FEATURE_NAMES)}
    origin_new = dataset.features[:, indexes["origin_is_new"]] == 1.0
    destination_new = (
        dataset.features[:, indexes["destination_is_new"]] == 1.0
    )
    slices: dict[str, NDArray[np.bool_]] = {
        "cash_out": dataset.features[:, indexes["type_cash_out"]] == 1.0,
        "transfer": dataset.features[:, indexes["type_transfer"]] == 1.0,
        "origin_new": origin_new,
        "origin_returning": ~origin_new,
        "destination_new": destination_new,
        "destination_returning": ~destination_new,
        "amount_at_least_200k": dataset.amounts >= 200_000.0,
    }
    output: dict[str, Any] = {}
    for name, mask in slices.items():
        metrics = _metrics_for_mask(
            dataset,
            scores,
            mask,
            threshold=threshold,
        )
        if metrics is not None:
            output[name] = metrics
    return output


def _load_v02_benchmark(project_root: Path) -> dict[str, Any]:
    path = project_root / "reports" / "v0.2" / "baseline_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing v0.2 benchmark metrics: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    benchmark = payload["models"][V02_BENCHMARK_MODEL]["evaluation"][
        "future_time_holdout"
    ]
    return cast(dict[str, Any], benchmark)


def train_behavioural_models(
    project_root: Path,
    *,
    maximum_fpr: float = TARGET_MAXIMUM_FPR,
    maximum_legitimate_rows: int = MAXIMUM_LEGITIMATE_ROWS,
    rebuild_features: bool = True,
) -> dict[str, Any]:
    """Build v0.3 features, train challengers, and evaluate temporal holdouts."""
    processed_dir = project_root / "data" / "processed"
    train_path = processed_dir / "train.parquet"
    future_path = processed_dir / "future_time_holdout.parquet"
    new_account_path = processed_dir / "new_account_holdout.parquet"
    for path in (train_path, future_path, new_account_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run sentinelgraph-data all first"
            )

    feature_path = processed_dir / "behavioural_features.parquet"
    print("[1/6] Building strict point-in-time behavioural features", flush=True)
    if rebuild_features or not feature_path.exists():
        feature_manifest = materialize_behavioural_features(
            (train_path, future_path),
            feature_path,
            relative_to=project_root,
        )
    else:
        feature_manifest = inspect_behavioural_feature_store(
            feature_path,
            relative_to=project_root,
        )
    _write_json(
        project_root / "data" / "metadata" / "behavioural_feature_manifest.json",
        feature_manifest,
    )

    train_min_step, train_max_step = step_range(train_path)
    development_end_step = temporal_cutoff(
        train_min_step,
        train_max_step,
        1.0 - VALIDATION_FRACTION_BY_TIME,
    )
    print("[2/6] Loading capped development and temporal validation", flush=True)
    development = load_behavioural_matrix(
        feature_path,
        where_sql=f"step <= {development_end_step}",
        max_legitimate_rows=maximum_legitimate_rows,
        random_seed=RANDOM_SEED,
    )
    validation = load_behavioural_matrix(
        feature_path,
        where_sql=(
            f"step > {development_end_step} AND step <= {train_max_step}"
        ),
    )

    model_specs = (
        ("isolation_forest", IsolationForestDetector()),
        (
            "behavioural_hist_gradient_boosting",
            build_behavioural_gradient_boosting(),
        ),
        (
            "behavioural_anomaly_hist_gradient_boosting",
            AnomalyAugmentedClassifier(),
        ),
    )
    models_dir = project_root / "models" / "v0.3"
    models_dir.mkdir(parents=True, exist_ok=True)
    fitted: dict[str, Any] = {}
    model_results: dict[str, Any] = {}
    scores_by_model: dict[str, dict[str, NDArray[np.float64]]] = {}

    print("[3/6] Fitting behavioural and anomaly challengers", flush=True)
    for name, model in model_specs:
        fitted_model, fit_seconds = _fit_model(model, development)
        print(f"  - {name}: {fit_seconds:.2f} s", flush=True)
        validation_scores = positive_scores(fitted_model, validation.features)
        threshold = threshold_at_fpr(
            validation.labels,
            validation_scores,
            maximum_fpr=maximum_fpr,
        )
        artifact_path = models_dir / f"{name}.joblib"
        joblib.dump(fitted_model, artifact_path, compress=3)
        fitted[name] = fitted_model
        scores_by_model[name] = {"validation": validation_scores}
        model_results[name] = {
            "fit_rows": development.rows,
            "fit_fraud_rows": int(development.labels.sum()),
            "threshold": threshold,
            "threshold_strategy": "validation_fpr_budget",
            "artifact": file_record(artifact_path, relative_to=project_root),
            "evaluation": {
                "validation": evaluate_scores(
                    validation.labels,
                    validation_scores,
                    validation.amounts,
                    threshold=threshold,
                )
            },
        }

    behavioural_name = "behavioural_hist_gradient_boosting"
    augmented_name = "behavioural_anomaly_hist_gradient_boosting"
    behavioural_validation_ap = model_results[behavioural_name]["evaluation"][
        "validation"
    ]["average_precision"]
    augmented_validation_ap = model_results[augmented_name]["evaluation"][
        "validation"
    ]["average_precision"]
    anomaly_validation_gain = (
        augmented_validation_ap - behavioural_validation_ap
    )
    selected_model = (
        augmented_name
        if anomaly_validation_gain >= MINIMUM_ANOMALY_VALIDATION_AP_GAIN
        else behavioural_name
    )

    print("[4/6] Loading untouched future and cold-start views", flush=True)
    future = load_behavioural_matrix(
        feature_path,
        where_sql=f"step > {train_max_step}",
    )
    new_account = load_behavioural_matrix(
        feature_path,
        where_sql=(
            "source_row_number IN ("
            "SELECT source_row_number FROM read_parquet("
            f"'{_escaped(new_account_path)}'))"
        ),
    )
    evaluation_sets = {
        "future_time_holdout": future,
        "new_account_holdout": new_account,
    }
    print("[5/6] Evaluating final holdouts and operational slices", flush=True)
    for name, model in fitted.items():
        threshold = float(model_results[name]["threshold"])
        for dataset_name, dataset in evaluation_sets.items():
            scores = positive_scores(model, dataset.features)
            scores_by_model[name][dataset_name] = scores
            model_results[name]["evaluation"][dataset_name] = evaluate_scores(
                dataset.labels,
                scores,
                dataset.amounts,
                threshold=threshold,
            )

    selected_scores = scores_by_model[selected_model]["future_time_holdout"]
    model_results[selected_model]["future_slices"] = _future_slices(
        future,
        selected_scores,
        threshold=float(model_results[selected_model]["threshold"]),
    )

    baseline = _load_v02_benchmark(project_root)
    selected_future = model_results[selected_model]["evaluation"][
        "future_time_holdout"
    ]
    comparison = {
        "baseline_release": "v0.2",
        "baseline_model": V02_BENCHMARK_MODEL,
        "baseline_average_precision": baseline["average_precision"],
        "baseline_recall": baseline["recall"],
        "baseline_false_positives_per_10k": baseline[
            "false_positives_per_10k_legitimate"
        ],
        "baseline_captured_fraud_amount_rate": baseline[
            "captured_fraud_amount_rate"
        ],
        "average_precision_delta": (
            selected_future["average_precision"] - baseline["average_precision"]
        ),
        "recall_delta": selected_future["recall"] - baseline["recall"],
        "false_positives_per_10k_delta": (
            selected_future["false_positives_per_10k_legitimate"]
            - baseline["false_positives_per_10k_legitimate"]
        ),
        "captured_fraud_amount_rate_delta": (
            selected_future["captured_fraud_amount_rate"]
            - baseline["captured_fraud_amount_rate"]
        ),
    }
    promotion_passed = (
        comparison["average_precision_delta"] > 0
        and selected_future["false_positive_rate"] <= maximum_fpr
    )
    future_min_step, future_max_step = step_range(future_path)
    results = {
        "release": "v0.3",
        "random_seed": RANDOM_SEED,
        "feature_names": list(BEHAVIOURAL_FEATURE_NAMES),
        "behavioural_feature_names": list(BEHAVIOURAL_ONLY_FEATURE_NAMES),
        "prohibited_model_fields": sorted(PROHIBITED_MODEL_FIELDS),
        "feature_store": feature_manifest,
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
            "select the anomaly-augmented supervised model only when it "
            "improves temporal-validation average precision by at least "
            f"{MINIMUM_ANOMALY_VALIDATION_AP_GAIN:.3f}; otherwise prefer "
            "the simpler behavioural supervised model"
        ),
        "selection_evidence": {
            "behavioural_validation_average_precision": (
                behavioural_validation_ap
            ),
            "anomaly_augmented_validation_average_precision": (
                augmented_validation_ap
            ),
            "anomaly_validation_average_precision_gain": (
                anomaly_validation_gain
            ),
            "minimum_required_gain": MINIMUM_ANOMALY_VALIDATION_AP_GAIN,
        },
        "selected_model": selected_model,
        "models": model_results,
        "comparison_to_v0_2": comparison,
        "promotion_decision": {
            "passed": promotion_passed,
            "criteria": (
                "future PR-AUC exceeds v0.2 and future FPR does not exceed "
                "the configured capacity budget"
            ),
        },
    }
    metrics_path = (
        project_root / "reports" / "v0.3" / "behavioural_metrics.json"
    )
    _write_json(metrics_path, results)
    print("[6/6] Writing metrics and behavioural report", flush=True)
    write_behavioural_report(
        project_root / "docs" / "BEHAVIOURAL_REPORT.md",
        results,
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the v0.3 experiment CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train SentinelGraph v0.3 behavioural and anomaly models"
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
        help="reuse the ignored point-in-time Parquet artifact when it exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v0.3 experiment from the command line."""
    args = build_parser().parse_args(argv)
    train_behavioural_models(
        args.project_root.resolve(),
        maximum_fpr=args.maximum_fpr,
        maximum_legitimate_rows=args.maximum_legitimate_rows,
        rebuild_features=not args.reuse_feature_store,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
