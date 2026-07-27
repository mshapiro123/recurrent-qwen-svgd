from __future__ import annotations

from eval.eval_speculative_depth_router_feasibility import (
    binary_auroc,
    budget_policy_curve,
    classify_depth_pair,
    deterministic_group_split,
    oracle_compute_frontier,
    oracle_depth_profile,
    router_verdict,
    summarize_oracle_router,
)


def test_classify_depth_pair_distinguishes_recovery_harm_and_ties() -> None:
    assert classify_depth_pair(False, True) == "recovered_at_2"
    assert classify_depth_pair(True, False) == "harmed_at_2"
    assert classify_depth_pair(True, True) == "both_correct"
    assert classify_depth_pair(False, False) == "both_wrong"


def test_group_split_never_splits_a_source_row() -> None:
    rows = [
        {"row_index": row_index, "stratum": stratum}
        for stratum in ("general", "code")
        for row_index in range(40)
    ]
    mapping = deterministic_group_split(rows, seed=20260727)
    assert set(mapping.values()) == {"train", "validation", "test"}
    assert len(mapping) == 80
    assert mapping[(7, "general")] in {"train", "validation", "test"}


def test_oracle_depth_profile_uses_first_correct_loop_and_saves_compute_on_misses() -> None:
    assert oracle_depth_profile([False, False, True, True]) == {
        "any_correct": True,
        "first_correct_depth": 3,
        "correct_depths": [3, 4],
        "selected_depth": 3,
    }


def test_oracle_frontier_prioritizes_cheapest_recoveries() -> None:
    matches = [
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [False, False, False],
    ]
    point = oracle_compute_frontier(matches, budgets=(1.5,))[0]
    assert point["realized_mean_loops"] == 1.25
    assert point["correct"] == 2
    assert point["accuracy"] == 0.5


def test_binary_auroc_is_tie_aware() -> None:
    assert binary_auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert binary_auroc([1.0, 1.0], [False, True]) == 0.5


def test_oracle_summary_reports_teacher_signal_as_non_deployable() -> None:
    base = {
        "row_index": 0,
        "local_position": 0,
        "stratum": "general",
        "kl": 0.5,
        "rank": 2,
        "run_length": 1,
        "teacher_entropy": 1.0,
        "negative_drafter_logprob_under_teacher": 2.0,
        "predictions": [1, 2, 3, 4, 5, 6],
        "matches_teacher_14b": [False, False, False, False, False, False],
    }
    rows = [
        {**base, "local_position": 0, "matches_teacher_7b": [False, True, False, False, False, False]},
        {**base, "local_position": 1, "kl": 1.5, "matches_teacher_7b": [True, False, False, False, False, False]},
    ]
    summary = summarize_oracle_router(rows, teacher="7b")
    assert summary["best_fixed_depth"] in {1, 2}
    assert summary["oracle_any_depth"]["accuracy"] == 1.0
    assert summary["teacher_signal_predictability"]["teacher_dependent_not_deployable"] is True
    assert oracle_depth_profile([False, False]) == {
        "any_correct": False,
        "first_correct_depth": None,
        "correct_depths": [],
        "selected_depth": 1,
    }


def test_budget_curve_reports_random_and_oracle_comparators() -> None:
    rows = [
        {"loop1_correct": False, "loop2_correct": True, "score": 0.9},
        {"loop1_correct": True, "loop2_correct": False, "score": 0.1},
        {"loop1_correct": False, "loop2_correct": True, "score": 0.8},
        {"loop1_correct": False, "loop2_correct": False, "score": 0.2},
    ]
    curve = budget_policy_curve(rows, score_field="score", fractions=(0.5,))
    point = curve[0]
    assert point["selected_for_loop2"] == 2
    assert point["model_correct"] == 3
    assert point["random_expected_correct"] == 1.5
    assert point["oracle_correct"] == 3
    assert point["mean_loops"] == 1.5


def test_router_verdict_requires_predictive_and_utility_evidence() -> None:
    positive = router_verdict(
        auroc=0.66,
        budget_points=[
            {"fraction": 0.25, "uplift_vs_random": 0.012, "bootstrap_low": 0.002},
            {"fraction": 0.50, "uplift_vs_random": 0.015, "bootstrap_low": 0.004},
            {"fraction": 0.75, "uplift_vs_random": 0.003, "bootstrap_low": -0.001},
        ],
    )
    assert positive == "viable_deployable_signal"
    assert router_verdict(auroc=0.59, budget_points=[]) == "no_deployable_signal"
