from __future__ import annotations

from eval.eval_coconut_composite_numerics import (
    HORIZONTAL_STEPS,
    adjacent_finite_difference_pass,
    select_precision_policy,
)


def test_rg11_covers_the_full_locked_k_range_without_exceeding_it() -> None:
    assert HORIZONTAL_STEPS == (1, 2, 3)


def test_rg4_requires_two_adjacent_original_criterion_passes() -> None:
    rows = [
        {"epsilon": 0.1, "passes_original_criterion": False},
        {"epsilon": 0.03, "passes_original_criterion": True},
        {"epsilon": 0.01, "passes_original_criterion": True},
        {"epsilon": 0.003, "passes_original_criterion": False},
    ]
    receipt = adjacent_finite_difference_pass(rows)
    assert receipt["passed"] is True
    assert receipt["adjacent_epsilons"] == [0.03, 0.01]


def test_rg4_rejects_isolated_passing_epsilon() -> None:
    rows = [
        {"epsilon": 0.1, "passes_original_criterion": True},
        {"epsilon": 0.03, "passes_original_criterion": False},
        {"epsilon": 0.01, "passes_original_criterion": True},
    ]
    assert adjacent_finite_difference_pass(rows)["passed"] is False


def test_precision_policy_prefers_autocast_then_fp32_and_never_full_bf16_on_failure() -> None:
    policies = {
        "full_bf16": {"all_examples_pass": False},
        "fp32_master_bf16_autocast": {"all_examples_pass": True},
        "full_fp32": {"all_examples_pass": True},
    }
    assert select_precision_policy(policies) == "fp32_master_bf16_autocast"
    policies["fp32_master_bf16_autocast"]["all_examples_pass"] = False
    assert select_precision_policy(policies) == "full_fp32"
