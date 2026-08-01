"""Train the SentinelGraph v0.5 calibrated risk and decision engine."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import joblib
import numpy as np
from numpy.typing import NDArray

from sentinelgraph.data.provenance import file_record
from sentinelgraph.data.splits import temporal_cutoff
from sentinelgraph.modeling.anomaly import (
    IsolationForestDetector,
    build_behavioural_gradient_boosting,
)
from sentinelgraph.modeling.behaviour import BEHAVIOURAL_FEATURE_NAMES
from sentinelgraph.modeling.calibration import (
    IsotonicCalibrator,
    ScoreStackCalibrator,
    SigmoidCalibrator,
    calibration_metrics,
)
from sentinelgraph.modeling.decision import (
    APPROVE,
    CostAssumptions,
    DecisionPolicy,
    apply_decision_policy,
    local_occlusion_explanations,
    policy_backtest,
    probability_to_risk_points,
    select_decision_policy,
)
from sentinelgraph.modeling.features import (
    PROHIBITED_MODEL_FIELDS,
    MatrixDataset,
    step_range,
)
from sentinelgraph.modeling.graph import (
    GRAPH_FEATURE_NAMES,
    GRAPH_ONLY_MODEL_FEATURE_NAMES,
    load_graph_matrix,
    materialize_graph_features,
)
from sentinelgraph.modeling.metrics import evaluate_scores, positive_scores
from sentinelgraph.modeling.models import RANDOM_SEED
from sentinelgraph.modeling.risk_report import write_risk_report

DEVELOPMENT_FRACTION_BY_TIME = 0.80
CALIBRATION_FIT_END_STEP = 450
CALIBRATION_SELECTION_END_STEP = 484
TARGET_MAXIMUM_REVIEW_RATE = 0.01
TARGET_MAXIMUM_DECLINE_RATE = 0.001
TARGET_MINIMUM_DECLINE_PRECISION = 0.80
MINIMUM_RANKING_RETENTION = 0.99
MAXIMUM_LEGITIMATE_ROWS = 999_999
EXPLANATION_SAMPLE_ROWS = 100
TRACKED_REASON_EXAMPLES = 25

Calibrator: TypeAlias = (
    SigmoidCalibrator | IsotonicCalibrator | ScoreStackCalibrator
)


def default_project_root() -> Path:
    """Resolve the repository root from this installed source tree."""
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _fit_model(model: Any, dataset: MatrixDataset) -> Any:
    started = time.perf_counter()
    model.fit(dataset.features, dataset.labels)
    print(f"    fitted in {time.perf_counter() - started:.2f} s", flush=True)
    return model


def _fit_component_models(development: MatrixDataset) -> dict[str, Any]:
    behavioural = _subset(development, BEHAVIOURAL_FEATURE_NAMES)
    graph = _subset(development, GRAPH_ONLY_MODEL_FEATURE_NAMES)
    print("  - behavioural classifier", flush=True)
    behavioural_model = _fit_model(
        build_behavioural_gradient_boosting(),
        behavioural,
    )
    print("  - legitimate-only anomaly detector", flush=True)
    anomaly_model = _fit_model(IsolationForestDetector(), behavioural)
    print("  - graph challenger", flush=True)
    graph_model = _fit_model(build_behavioural_gradient_boosting(), graph)
    return {
        "behavioural_probability": behavioural_model,
        "anomaly_score": anomaly_model,
        "graph_probability": graph_model,
    }


def _component_scores(
    models: Mapping[str, Any],
    dataset: MatrixDataset,
) -> dict[str, NDArray[np.float64]]:
    behavioural = _subset(dataset, BEHAVIOURAL_FEATURE_NAMES)
    graph = _subset(dataset, GRAPH_ONLY_MODEL_FEATURE_NAMES)
    return {
        "behavioural_probability": positive_scores(
            models["behavioural_probability"],
            behavioural.features,
        ),
        "anomaly_score": positive_scores(
            models["anomaly_score"],
            behavioural.features,
        ),
        "graph_probability": positive_scores(
            models["graph_probability"],
            graph.features,
        ),
    }


def _concatenate_component_scores(
    first: Mapping[str, NDArray[np.float64]],
    second: Mapping[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    if set(first) != set(second):
        raise ValueError("component score sets must have the same names")
    return {
        name: np.concatenate((first[name], second[name]))
        for name in first
    }


def _fit_calibrator(
    name: str,
    components: Mapping[str, NDArray[np.float64]],
    labels: NDArray[np.uint8],
) -> Calibrator:
    if name == "sigmoid_behavioural":
        return SigmoidCalibrator().fit(
            components["behavioural_probability"],
            labels,
        )
    if name == "isotonic_behavioural":
        return IsotonicCalibrator().fit(
            components["behavioural_probability"],
            labels,
        )
    if name == "logistic_score_stack":
        return ScoreStackCalibrator().fit(components, labels)
    raise ValueError(f"unknown calibrator: {name}")


def _calibrated_scores(
    calibrator: Calibrator,
    components: Mapping[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    if isinstance(calibrator, ScoreStackCalibrator):
        return calibrator.predict(components)
    return calibrator.predict(components["behavioural_probability"])


class _CalibratedRiskAdapter:
    """Expose end-to-end risk probabilities for model-agnostic occlusion."""

    def __init__(
        self,
        models: Mapping[str, Any],
        calibrator: Calibrator,
    ) -> None:
        self.models = models
        self.calibrator = calibrator
        self.behavioural_indexes = tuple(
            GRAPH_FEATURE_NAMES.index(name)
            for name in BEHAVIOURAL_FEATURE_NAMES
        )
        self.graph_indexes = tuple(
            GRAPH_FEATURE_NAMES.index(name)
            for name in GRAPH_ONLY_MODEL_FEATURE_NAMES
        )

    def predict_proba(
        self,
        features: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Return calibrated risk while recomputing every evidence component."""
        matrix = np.asarray(features, dtype=np.float32)
        behavioural = matrix[:, self.behavioural_indexes]
        graph = matrix[:, self.graph_indexes]
        components = {
            "behavioural_probability": positive_scores(
                self.models["behavioural_probability"],
                behavioural,
            ),
            "anomaly_score": positive_scores(
                self.models["anomaly_score"],
                behavioural,
            ),
            "graph_probability": positive_scores(
                self.models["graph_probability"],
                graph,
            ),
        }
        risks = _calibrated_scores(self.calibrator, components)
        return np.column_stack((1.0 - risks, risks))


def _select_calibrator(
    calibration_fit_components: Mapping[str, NDArray[np.float64]],
    calibration_fit_labels: NDArray[np.uint8],
    selection_components: Mapping[str, NDArray[np.float64]],
    selection_labels: NDArray[np.uint8],
) -> tuple[str, dict[str, Any]]:
    raw_metrics = calibration_metrics(
        selection_labels,
        selection_components["behavioural_probability"],
    )
    candidate_results: dict[str, Any] = {
        "raw_behavioural_reference": raw_metrics
    }
    candidate_names = (
        "sigmoid_behavioural",
        "isotonic_behavioural",
        "logistic_score_stack",
    )
    for name in candidate_names:
        calibrator = _fit_calibrator(
            name,
            calibration_fit_components,
            calibration_fit_labels,
        )
        probabilities = _calibrated_scores(calibrator, selection_components)
        candidate_results[name] = calibration_metrics(
            selection_labels,
            probabilities,
        )

    minimum_average_precision = (
        float(raw_metrics["average_precision"]) * MINIMUM_RANKING_RETENTION
    )
    eligible = [
        name
        for name in candidate_names
        if float(candidate_results[name]["average_precision"])
        >= minimum_average_precision
    ]
    if not eligible:
        eligible = ["sigmoid_behavioural"]
    complexity_order = {
        "sigmoid_behavioural": 0,
        "isotonic_behavioural": 1,
        "logistic_score_stack": 2,
    }
    selected = min(
        eligible,
        key=lambda name: (
            float(candidate_results[name]["brier_score"]),
            float(candidate_results[name]["log_loss"]),
            complexity_order[name],
        ),
    )
    evidence = {
        "raw_behavioural_reference": raw_metrics,
        "candidates": {
            name: candidate_results[name] for name in candidate_names
        },
        "minimum_ranking_retention": MINIMUM_RANKING_RETENTION,
        "minimum_eligible_average_precision": minimum_average_precision,
        "eligible_candidates": eligible,
        "selected_calibrator": selected,
        "selection_metric": (
            "minimum Brier score, then log loss, subject to retaining at least "
            "99% of raw behavioural average precision"
        ),
        "selected_brier_improvement": (
            float(raw_metrics["brier_score"])
            - float(candidate_results[selected]["brier_score"])
        ),
    }
    return selected, evidence


def _dataset_evaluation(
    dataset: MatrixDataset,
    risks: NDArray[np.float64],
    *,
    policy: DecisionPolicy,
    costs: CostAssumptions,
) -> dict[str, Any]:
    return {
        "calibration": calibration_metrics(dataset.labels, risks),
        "binary_review_or_decline": evaluate_scores(
            dataset.labels,
            risks,
            dataset.amounts,
            threshold=float(policy.review_threshold),
        ),
        "policy_backtest": policy_backtest(
            dataset.labels,
            risks,
            dataset.amounts,
            policy,
            costs=costs,
        ),
    }


def _reason_examples(
    models: Mapping[str, Any],
    calibrator: Calibrator,
    development: MatrixDataset,
    future: MatrixDataset,
    future_components: Mapping[str, NDArray[np.float64]],
    risks: NDArray[np.float64],
    policy: DecisionPolicy,
) -> dict[str, Any]:
    decisions = apply_decision_policy(risks, policy)
    flagged = np.flatnonzero(decisions != APPROVE)
    if flagged.size == 0:
        return {
            "method": "local_legitimate_median_occlusion",
            "examples": [],
            "global_importance": [],
        }
    ordered = flagged[
        np.argsort(-risks[flagged], kind="stable")
    ][:EXPLANATION_SAMPLE_ROWS]
    legitimate_baseline = np.median(
        development.features[development.labels == 0],
        axis=0,
    ).astype(np.float32)
    risk_adapter = _CalibratedRiskAdapter(models, calibrator)
    explanation = local_occlusion_explanations(
        risk_adapter,
        future.features[ordered],
        GRAPH_FEATURE_NAMES,
        legitimate_baseline,
        top_k=3,
    )
    risk_points = probability_to_risk_points(risks[ordered])
    local_reasons = explanation["local_reasons"]
    examples = []
    for rank, index in enumerate(ordered[:TRACKED_REASON_EXAMPLES], start=1):
        position = rank - 1
        examples.append(
            {
                "risk_rank": rank,
                "calibrated_probability": float(risks[index]),
                "risk_points": int(risk_points[position]),
                "decision": str(decisions[index]),
                "transaction_amount": float(future.amounts[index]),
                "actual_fraud_label": int(future.labels[index]),
                "component_evidence": {
                    name: float(scores[index])
                    for name, scores in future_components.items()
                },
                "reason_codes": local_reasons[position],
            }
        )
    return {
        "method": explanation["method"],
        "baseline": explanation["baseline"],
        "coverage": "review and decline decisions only",
        "sample_rows": int(ordered.size),
        "tracked_example_rows": len(examples),
        "global_importance": explanation["global_importance"][:15],
        "examples": examples,
    }


def train_risk_engine(
    project_root: Path,
    *,
    maximum_legitimate_rows: int = MAXIMUM_LEGITIMATE_ROWS,
    maximum_review_rate: float = TARGET_MAXIMUM_REVIEW_RATE,
    maximum_decline_rate: float = TARGET_MAXIMUM_DECLINE_RATE,
    minimum_decline_precision: float = TARGET_MINIMUM_DECLINE_PRECISION,
    rebuild_features: bool = False,
) -> dict[str, Any]:
    """Train, calibrate, explain, and backtest the v0.5 risk engine."""
    processed_dir = project_root / "data" / "processed"
    interim_dir = project_root / "data" / "interim"
    train_path = processed_dir / "train.parquet"
    future_path = processed_dir / "future_time_holdout.parquet"
    new_account_path = processed_dir / "new_account_holdout.parquet"
    for path in (train_path, future_path, new_account_path):
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
    source_paths = (train_path, future_path)
    feature_artifacts_exist = all(
        path.exists()
        for path in (behavioural_path, graph_path, component_path, edge_path)
    )
    print("[1/8] Preparing point-in-time feature stores", flush=True)
    if rebuild_features or not feature_artifacts_exist:
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
        if not manifest_path.exists():
            raise FileNotFoundError(
                "graph feature manifest is missing; rebuild the feature store"
            )
        feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_min_step, train_max_step = step_range(train_path)
    development_end_step = temporal_cutoff(
        train_min_step,
        train_max_step,
        DEVELOPMENT_FRACTION_BY_TIME,
    )
    if development_end_step >= CALIBRATION_FIT_END_STEP:
        raise ValueError("development and calibration windows overlap")
    print("[2/8] Loading temporal development and governance windows", flush=True)
    development = load_graph_matrix(
        graph_path,
        component_path,
        where_sql=f"step <= {development_end_step}",
        max_legitimate_rows=maximum_legitimate_rows,
        random_seed=RANDOM_SEED,
    )
    calibration_fit = load_graph_matrix(
        graph_path,
        component_path,
        where_sql=(
            f"step > {development_end_step} "
            f"AND step <= {CALIBRATION_FIT_END_STEP}"
        ),
    )
    calibration_selection = load_graph_matrix(
        graph_path,
        component_path,
        where_sql=(
            f"step > {CALIBRATION_FIT_END_STEP} "
            f"AND step <= {CALIBRATION_SELECTION_END_STEP}"
        ),
    )
    policy_selection = load_graph_matrix(
        graph_path,
        component_path,
        where_sql=(
            f"step > {CALIBRATION_SELECTION_END_STEP} "
            f"AND step <= {train_max_step}"
        ),
    )

    print("[3/8] Fitting classifier, anomaly, and graph components", flush=True)
    models = _fit_component_models(development)
    fit_components = _component_scores(models, calibration_fit)
    selection_components = _component_scores(models, calibration_selection)
    print("[4/8] Selecting and refitting probability calibration", flush=True)
    selected_calibrator_name, calibration_selection_evidence = (
        _select_calibrator(
            fit_components,
            calibration_fit.labels,
            selection_components,
            calibration_selection.labels,
        )
    )
    combined_components = _concatenate_component_scores(
        fit_components,
        selection_components,
    )
    combined_labels = np.concatenate(
        (calibration_fit.labels, calibration_selection.labels)
    )
    calibrator = _fit_calibrator(
        selected_calibrator_name,
        combined_components,
        combined_labels,
    )

    print("[5/8] Selecting approve/review/decline thresholds", flush=True)
    policy_components = _component_scores(models, policy_selection)
    policy_risks = _calibrated_scores(calibrator, policy_components)
    policy = select_decision_policy(
        policy_selection.labels,
        policy_risks,
        policy_selection.amounts,
        maximum_review_rate=maximum_review_rate,
        maximum_decline_rate=maximum_decline_rate,
        minimum_decline_precision=minimum_decline_precision,
    )
    costs = CostAssumptions()

    print("[6/8] Evaluating untouched future and new-account holdouts", flush=True)
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
    future_components = _component_scores(models, future)
    new_account_components = _component_scores(models, new_account)
    future_risks = _calibrated_scores(calibrator, future_components)
    new_account_risks = _calibrated_scores(calibrator, new_account_components)
    evaluations = {
        "policy_selection": _dataset_evaluation(
            policy_selection,
            policy_risks,
            policy=policy,
            costs=costs,
        ),
        "future_time_holdout": _dataset_evaluation(
            future,
            future_risks,
            policy=policy,
            costs=costs,
        ),
        "new_account_holdout": _dataset_evaluation(
            new_account,
            new_account_risks,
            policy=policy,
            costs=costs,
        ),
    }

    print("[7/8] Generating bounded local explanations", flush=True)
    explanations = _reason_examples(
        models,
        calibrator,
        development,
        future,
        future_components,
        future_risks,
        policy,
    )
    explanation_path = (
        project_root / "reports" / "v0.5" / "reason_code_examples.json"
    )
    _write_json(explanation_path, explanations)

    models_dir = project_root / "models" / "v0.5"
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = models_dir / "risk_bundle.joblib"
    bundle = {
        "release": "v0.5",
        "feature_names": list(GRAPH_FEATURE_NAMES),
        "behavioural_feature_names": list(BEHAVIOURAL_FEATURE_NAMES),
        "graph_feature_names": list(GRAPH_ONLY_MODEL_FEATURE_NAMES),
        "models": models,
        "calibrator_name": selected_calibrator_name,
        "calibrator": calibrator,
        "policy": policy,
        "cost_assumptions": costs,
    }
    joblib.dump(bundle, bundle_path, compress=3)
    _write_json(
        models_dir / "artifact_manifest.json",
        {
            "risk_bundle": file_record(bundle_path, relative_to=project_root),
            "tracked": False,
            "rationale": (
                "joblib byte streams are not treated as canonical tracked "
                "evidence; semantic metrics and configuration are tracked"
            ),
        },
    )

    future_min_step, future_max_step = step_range(future_path)
    results = {
        "release": "v0.5",
        "random_seed": RANDOM_SEED,
        "feature_store": feature_manifest,
        "prohibited_model_fields": sorted(PROHIBITED_MODEL_FIELDS),
        "temporal_contract": {
            "development": {
                "minimum_step": train_min_step,
                "maximum_step": development_end_step,
                "purpose": "fit base classifier, anomaly, and graph models",
            },
            "calibration_fit": {
                "minimum_step": development_end_step + 1,
                "maximum_step": CALIBRATION_FIT_END_STEP,
                "purpose": "fit calibration candidates",
            },
            "calibration_selection": {
                "minimum_step": CALIBRATION_FIT_END_STEP + 1,
                "maximum_step": CALIBRATION_SELECTION_END_STEP,
                "purpose": "select calibration method without future data",
            },
            "policy_selection": {
                "minimum_step": CALIBRATION_SELECTION_END_STEP + 1,
                "maximum_step": train_max_step,
                "purpose": "select approve/review/decline thresholds",
            },
            "future_time_holdout": {
                "minimum_step": future_min_step,
                "maximum_step": future_max_step,
                "purpose": "one-time final temporal evaluation",
            },
        },
        "component_models": {
            "behavioural_probability": {
                "role": "v0.3 release champion and primary ranking signal",
                "feature_count": len(BEHAVIOURAL_FEATURE_NAMES),
            },
            "anomaly_score": {
                "role": "legitimate-only unsupervised evidence signal",
                "feature_count": len(BEHAVIOURAL_FEATURE_NAMES),
            },
            "graph_probability": {
                "role": "v0.4 graph challenger evidence signal",
                "feature_count": len(GRAPH_ONLY_MODEL_FEATURE_NAMES),
            },
        },
        "calibration_selection": calibration_selection_evidence,
        "selected_calibrator": selected_calibrator_name,
        "decision_policy": policy.to_dict(),
        "decision_policy_semantics": {
            "approve": "risk is below the investigator review threshold",
            "review": "human investigation is required",
            "decline": (
                "offline high-risk recommendation only; no automated customer "
                "action is performed"
            ),
        },
        "cost_assumptions": costs.to_dict(),
        "evaluations": evaluations,
        "explanations": {
            key: value for key, value in explanations.items() if key != "examples"
        },
        "reason_code_examples": {
            "path": "reports/v0.5/reason_code_examples.json",
            "rows": len(explanations["examples"]),
        },
        "model_artifact": {
            "path": "models/v0.5/risk_bundle.joblib",
            "local_manifest": "models/v0.5/artifact_manifest.json",
            "tracked": False,
            "checksum_policy": (
                "local-only because non-canonical joblib byte serialization "
                "can change while predictions and semantic metrics remain equal"
            ),
        },
    }
    metrics_path = project_root / "reports" / "v0.5" / "risk_metrics.json"
    _write_json(metrics_path, results)
    print("[8/8] Writing risk metrics and decision report", flush=True)
    write_risk_report(project_root / "docs" / "RISK_ENGINE_REPORT.md", results)
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the v0.5 risk-engine CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train SentinelGraph v0.5 calibrated risk engine"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root(),
    )
    parser.add_argument(
        "--maximum-legitimate-rows",
        type=int,
        default=MAXIMUM_LEGITIMATE_ROWS,
    )
    parser.add_argument(
        "--maximum-review-rate",
        type=float,
        default=TARGET_MAXIMUM_REVIEW_RATE,
    )
    parser.add_argument(
        "--maximum-decline-rate",
        type=float,
        default=TARGET_MAXIMUM_DECLINE_RATE,
    )
    parser.add_argument(
        "--minimum-decline-precision",
        type=float,
        default=TARGET_MINIMUM_DECLINE_PRECISION,
    )
    parser.add_argument(
        "--rebuild-feature-store",
        action="store_true",
        help="rebuild v0.3/v0.4 point-in-time feature artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v0.5 calibrated decision-engine experiment."""
    args = build_parser().parse_args(argv)
    train_risk_engine(
        args.project_root.resolve(),
        maximum_legitimate_rows=args.maximum_legitimate_rows,
        maximum_review_rate=args.maximum_review_rate,
        maximum_decline_rate=args.maximum_decline_rate,
        minimum_decline_precision=args.minimum_decline_precision,
        rebuild_features=args.rebuild_feature_store,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
