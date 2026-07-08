from __future__ import annotations

import pytest

from colab.run_stage5_support6_seed_replication import canonical_frontier_from_score, summarize_results
from colab.stage5_frontier_metrics import (
    bar_crossing_frontier,
    deepest_passing_selection_frontier,
)


def test_deepest_passing_selection_frontier_is_deprecated_diagnostic() -> None:
    selection = {
        "7": {"correct": 52, "pass": True},
        "8": {"correct": 20, "pass": True},
        "9": {"correct": 13, "pass": False},
        "10": {"correct": 30, "pass": True},
    }

    assert deepest_passing_selection_frontier(selection) == 10


def test_bar_crossing_frontier_interpolates_first_drop_below_bar() -> None:
    diag = {"7": 93 / 128, "8": 48 / 128, "9": 25 / 128, "10": 13 / 128}

    assert bar_crossing_frontier(diag, bar=0.71) == pytest.approx(7.047, abs=0.001)


def test_canonical_frontier_uses_diagonal_counts_not_selection_pass_flags() -> None:
    score = {
        "diagonal_counts": {
            "7": {"correct": 93, "total": 128, "accuracy": 93 / 128},
            "8": {"correct": 48, "total": 128, "accuracy": 48 / 128},
            "9": {"correct": 25, "total": 128, "accuracy": 25 / 128},
            "10": {"correct": 13, "total": 128, "accuracy": 13 / 128},
        },
        "selection": {
            "7": {"correct": 93, "pass": True},
            "8": {"correct": 48, "pass": True},
            "9": {"correct": 25, "pass": True},
            "10": {"correct": 13, "pass": False},
        },
    }

    assert canonical_frontier_from_score(score) == pytest.approx(7.047, abs=0.001)


def test_seed_replication_summary_requires_band_around_target() -> None:
    passing = summarize_results([{"canonical_frontier": 8.0}, {"canonical_frontier": 10.0}])
    failing = summarize_results([{"canonical_frontier": 7.9}, {"canonical_frontier": 10.0}])

    assert passing["status"] == "replication_pass"
    assert passing["within_plus_minus_one"] is True
    assert failing["status"] == "replication_needs_review"
