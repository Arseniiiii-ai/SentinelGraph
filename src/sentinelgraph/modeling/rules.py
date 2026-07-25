"""Transparent transaction-rule baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sentinelgraph.modeling.features import FEATURE_NAMES

TRANSFER_INDEX = FEATURE_NAMES.index("type_transfer")
AMOUNT_INDEX = FEATURE_NAMES.index("amount")


class RuleBaseline:
    """Recompute PaySim's documented large-transfer policy from safe inputs."""

    def __init__(self, transfer_threshold: float = 200_000.0) -> None:
        self.transfer_threshold = transfer_threshold

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.uint8],
    ) -> "RuleBaseline":
        """Return the stateless rule for estimator API compatibility."""
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have the same row count")
        return self

    def predict_proba(
        self,
        features: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Return a binary rule score as a two-class probability matrix."""
        amount = features[:, AMOUNT_INDEX].astype(np.float64)
        is_transfer = features[:, TRANSFER_INDEX] > 0.5
        positive = is_transfer & (amount > self.transfer_threshold)
        score = positive.astype(np.float64)
        return np.column_stack((1.0 - score, score))

    def get_params(self, deep: bool = True) -> dict[str, float]:
        """Return serialisable estimator parameters."""
        del deep
        return {"transfer_threshold": self.transfer_threshold}
