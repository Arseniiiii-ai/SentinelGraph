"""Tests for pinned PaySim provenance."""

from sentinelgraph.data.provenance import (
    DATASET_LICENSE,
    DATASET_SLUG,
    DATASET_VERSION,
    EXPECTED_CSV_SHA256,
)


def test_dataset_distribution_is_pinned() -> None:
    assert DATASET_SLUG == "ealaxi/paysim1"
    assert DATASET_VERSION == 2
    assert DATASET_LICENSE == "CC BY-SA 4.0"
    assert len(EXPECTED_CSV_SHA256) == 64
    int(EXPECTED_CSV_SHA256, 16)
