"""Factories for the v0.2 learned and sanity-check baselines."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def build_dummy_baseline() -> DummyClassifier:
    """Return a label-prior sanity baseline."""
    return DummyClassifier(strategy="prior", random_state=RANDOM_SEED)


def build_logistic_baseline() -> Pipeline:
    """Return a balanced, scaled logistic-regression baseline."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=250,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_gradient_boosting_baseline() -> HistGradientBoostingClassifier:
    """Return a regularised histogram gradient-boosting baseline."""
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        early_stopping=True,
        l2_regularization=1.0,
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        n_iter_no_change=12,
        random_state=RANDOM_SEED,
        validation_fraction=0.1,
    )
