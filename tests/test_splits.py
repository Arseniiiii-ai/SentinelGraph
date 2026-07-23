"""Tests for leakage-safe split definitions."""

import pytest

from sentinelgraph.data.splits import temporal_cutoff


def test_paysim_cutoff_uses_first_seventy_percent_of_time() -> None:
    assert temporal_cutoff(1, 743, 0.70) == 520


def test_cutoff_supports_non_one_minimum_step() -> None:
    assert temporal_cutoff(10, 19, 0.70) == 16


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        temporal_cutoff(1, 100, fraction)


def test_inverted_step_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        temporal_cutoff(2, 1)
