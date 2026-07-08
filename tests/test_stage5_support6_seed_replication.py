from __future__ import annotations

import pytest

from colab.run_stage5_support6_dosed_seed_resolution import (
    failed_replicate_runs,
    seed_from_label,
    summarize_dosed_results,
)
from colab.run_stage5_support6_seed26_plateau import plateau_outcome
from colab.run_stage5_support6_seed_replication import canonical_frontier_from_score, summarize_results
from colab.run_stage5_synthetic_release_receipts import release_status
from colab.run_stage5_synthetic_release_receipts import compact_dosed
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


def test_dosed_seed_resolution_selects_only_failed_replicates() -> None:
    receipt = {
        "runs": [
            {"label": "original", "canonical_frontier_pass": True},
            {"label": "seed_20260716", "canonical_frontier_pass": False},
            {"label": "seed_20260726", "canonical_frontier_pass": True},
        ]
    }

    failed = failed_replicate_runs(receipt)

    assert [item["label"] for item in failed] == ["seed_20260716"]
    assert seed_from_label("seed_20260716") == "20260716"


def test_dosed_seed_resolution_requires_post_dose_frontier_band() -> None:
    passing = summarize_dosed_results(
        [
            {
                "pre_dose": {"canonical_frontier": 7.0},
                "post_dose": {"canonical_frontier": 8.1},
            },
            {
                "pre_dose": {"canonical_frontier": 6.9},
                "post_dose": {"canonical_frontier": 9.2},
            },
        ]
    )
    failing = summarize_dosed_results(
        [
            {
                "pre_dose": {"canonical_frontier": 7.0},
                "post_dose": {"canonical_frontier": 7.9},
            }
        ]
    )

    assert passing["status"] == "dosed_seed_resolution_pass"
    assert passing["all_dosed_frontiers_improved"] is True
    one_completed_result = [
        {
            "pre_dose": {"canonical_frontier": 7.0},
            "post_dose": {"canonical_frontier": 8.1},
        }
    ]
    partial = summarize_dosed_results(one_completed_result, expected_count=2)
    assert partial["status"] == "dosed_seed_resolution_running"
    assert failing["status"] == "dosed_seed_resolution_needs_review"


def test_seed26_plateau_classifier_uses_locked_rules() -> None:
    assert plateau_outcome(pre_frontier=7.77, post_frontier=8.01)["status"] == "seed26_unified"
    assert plateau_outcome(pre_frontier=7.77, post_frontier=7.95)["status"] == "seed26_plateau"
    ambiguous = plateau_outcome(pre_frontier=7.40, post_frontier=7.80)
    assert ambiguous["status"] == "seed26_ambiguous"
    assert ambiguous["gain"] == pytest.approx(0.4)


def test_release_receipt_status_prioritizes_blockers_then_pending() -> None:
    assert release_status({"blockers": ["x"], "pending_followups": []}) == "release_receipts_blocked"
    assert release_status({"blockers": [], "pending_followups": ["x"]}) == "release_receipts_need_followup"
    assert release_status({"blockers": [], "pending_followups": []}) == "release_receipts_complete"


def test_release_receipts_reclassify_partial_dosed_parent_as_running() -> None:
    compact = compact_dosed(
        {
            "status": "dosed_seed_resolution_pass",
            "failed_replicates": [{"label": "seed_a"}, {"label": "seed_b"}],
            "results": [{"label": "seed_a"}],
        },
        None,
    )

    assert compact["status"] == "dosed_seed_resolution_running"
    assert compact["all_expected_completed"] is False
