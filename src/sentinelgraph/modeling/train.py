"""Train and evaluate the leakage-safe v0.2 baseline models."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib

from sentinelgraph.data.provenance import file_record
from sentinelgraph.data.splits import temporal_cutoff
from sentinelgraph.modeling.features import (
    FEATURE_NAMES,
    PROHIBITED_MODEL_FIELDS,
    MatrixDataset,
    deterministic_class_cap,
    load_matrix,
    step_range,
)
from sentinelgraph.modeling.metrics import (
    evaluate_scores,
    positive_scores,
    threshold_at_fpr,
)
from sentinelgraph.modeling.models import (
    RANDOM_SEED,
    build_dummy_baseline,
    build_gradient_boosting_baseline,
    build_logistic_baseline,
)
from sentinelgraph.modeling.report import write_baseline_report
from sentinelgraph.modeling.rules import RuleBaseline

VALIDATION_FRACTION_BY_TIME = 0.20
TARGET_MAXIMUM_FPR = 0.01
GRADIENT_MAX_LEGITIMATE_ROWS = 1_000_000


def default_project_root() -> Path:
    """Resolve the repository root from this installed source tree."""
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fit_model(
    model: Any,
    dataset: MatrixDataset,
) -> tuple[Any, float]:
    started = time.perf_counter()
    model.fit(dataset.features, dataset.labels)
    return model, time.perf_counter() - started


def _evaluate_model(
    model: Any,
    dataset: MatrixDataset,
    *,
    threshold: float,
) -> dict[str, Any]:
    scores = positive_scores(model, dataset.features)
    return evaluate_scores(
        dataset.labels,
        scores,
        dataset.amounts,
        threshold=threshold,
    )


def train_baselines(
    project_root: Path,
    *,
    maximum_fpr: float = TARGET_MAXIMUM_FPR,
    gradient_max_legitimate_rows: int = GRADIENT_MAX_LEGITIMATE_ROWS,
) -> dict[str, Any]:
    """Train all v0.2 baselines and evaluate untouched holdouts."""
    processed_dir = project_root / "data" / "processed"
    train_path = processed_dir / "train.parquet"
    future_path = processed_dir / "future_time_holdout.parquet"
    new_account_path = processed_dir / "new_account_holdout.parquet"
    for path in (train_path, future_path, new_account_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run sentinelgraph-data all first"
            )

    train_min_step, train_max_step = step_range(train_path)
    development_fraction = 1.0 - VALIDATION_FRACTION_BY_TIME
    development_end_step = temporal_cutoff(
        train_min_step,
        train_max_step,
        development_fraction,
    )

    print(
        "[1/5] Loading development and temporal validation matrices",
        flush=True,
    )
    development = load_matrix(
        train_path,
        where_sql=f"step <= {development_end_step}",
    )
    validation = load_matrix(
        train_path,
        where_sql=f"step > {development_end_step}",
    )
    gradient_development = deterministic_class_cap(
        development,
        max_legitimate_rows=gradient_max_legitimate_rows,
        random_seed=RANDOM_SEED,
    )

    model_specs = (
        ("dummy_prior", build_dummy_baseline(), development),
        ("large_transfer_rule", RuleBaseline(), development),
        ("logistic_regression", build_logistic_baseline(), development),
        (
            "hist_gradient_boosting",
            build_gradient_boosting_baseline(),
            gradient_development,
        ),
    )
    fitted: dict[str, Any] = {}
    model_results: dict[str, Any] = {}
    models_dir = project_root / "models" / "v0.2"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("[2/5] Fitting dummy, rule, logistic, and gradient models", flush=True)
    for name, model, fit_dataset in model_specs:
        fitted_model, fit_seconds = _fit_model(model, fit_dataset)
        print(f"  - {name}: {fit_seconds:.2f} s", flush=True)
        fitted[name] = fitted_model
        validation_scores = positive_scores(
            fitted_model,
            validation.features,
        )
        if name == "large_transfer_rule":
            threshold = 0.5
            threshold_strategy = "fixed_policy_threshold"
        else:
            threshold = threshold_at_fpr(
                validation.labels,
                validation_scores,
                maximum_fpr=maximum_fpr,
            )
            threshold_strategy = "validation_fpr_budget"
        artifact_path = models_dir / f"{name}.joblib"
        joblib.dump(fitted_model, artifact_path, compress=3)
        model_results[name] = {
            "fit_rows": fit_dataset.rows,
            "fit_fraud_rows": int(fit_dataset.labels.sum()),
            "threshold": threshold,
            "threshold_strategy": threshold_strategy,
            "artifact": file_record(
                artifact_path,
                relative_to=project_root,
            ),
            "evaluation": {
                "validation": evaluate_scores(
                    validation.labels,
                    validation_scores,
                    validation.amounts,
                    threshold=threshold,
                )
            },
        }

    print("[3/5] Loading untouched future evaluation matrices", flush=True)
    future = load_matrix(future_path)
    new_account = load_matrix(new_account_path)
    evaluation_sets = {
        "future_time_holdout": future,
        "new_account_holdout": new_account,
    }
    print("[4/5] Evaluating both final holdout views", flush=True)
    for name, model in fitted.items():
        threshold = float(model_results[name]["threshold"])
        for dataset_name, dataset in evaluation_sets.items():
            model_results[name]["evaluation"][dataset_name] = _evaluate_model(
                model,
                dataset,
                threshold=threshold,
            )

    future_min_step, future_max_step = step_range(future_path)
    results = {
        "release": "v0.2",
        "random_seed": RANDOM_SEED,
        "feature_names": list(FEATURE_NAMES),
        "prohibited_model_fields": sorted(PROHIBITED_MODEL_FIELDS),
        "target_maximum_fpr": maximum_fpr,
        "gradient_max_legitimate_rows": gradient_max_legitimate_rows,
        "development_split": {
            "development_min_step": train_min_step,
            "development_max_step": development_end_step,
            "validation_min_step": development_end_step + 1,
            "validation_max_step": train_max_step,
            "future_min_step": future_min_step,
            "future_max_step": future_max_step,
        },
        "models": model_results,
    }
    metrics_path = project_root / "reports" / "v0.2" / "baseline_metrics.json"
    _write_json(metrics_path, results)
    print("[5/5] Writing machine-readable metrics and baseline report", flush=True)
    write_baseline_report(
        project_root / "docs" / "BASELINE_REPORT.md",
        results,
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the baseline training CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train SentinelGraph v0.2 leakage-safe baselines"
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
        "--gradient-max-legitimate-rows",
        type=int,
        default=GRADIENT_MAX_LEGITIMATE_ROWS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run baseline training from the command line."""
    args = build_parser().parse_args(argv)
    train_baselines(
        args.project_root.resolve(),
        maximum_fpr=args.maximum_fpr,
        gradient_max_legitimate_rows=args.gradient_max_legitimate_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
