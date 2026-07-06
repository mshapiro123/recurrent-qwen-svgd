from __future__ import annotations

from colab.run_stage5_support6_seed_replication import frontier_from_selection, summarize_results


def test_frontier_from_selection_uses_depth_specific_pass_flags() -> None:
    selection = {
        "7": {"correct": 52, "pass": True},
        "8": {"correct": 20, "pass": True},
        "9": {"correct": 13, "pass": False},
        "10": {"correct": 30, "pass": True},
    }

    assert frontier_from_selection(selection) == 10


def test_seed_replication_summary_requires_band_around_target() -> None:
    passing = summarize_results([{"frontier": 8}, {"frontier": 10}])
    failing = summarize_results([{"frontier": 7}, {"frontier": 10}])

    assert passing["status"] == "replication_pass"
    assert passing["within_plus_minus_one"] is True
    assert failing["status"] == "replication_needs_review"
