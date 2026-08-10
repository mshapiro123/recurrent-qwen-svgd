from __future__ import annotations

from eval.eval_paper2_phase3_guardrail_recalibration import (
    paired_discordance,
    simulate_rule,
)


def test_empirical_discordance_uses_floor_dev_only() -> None:
    rows = [
        {
            "partition": "dev",
            "battery_role": "floor_retention_only",
            "battery": "tier1",
            "base_correct": True,
            "teacher_14b_correct": False,
        },
        {
            "partition": "dev",
            "battery_role": "floor_retention_only",
            "battery": "tier1",
            "base_correct": True,
            "teacher_14b_correct": True,
        },
        {
            "partition": "dev",
            "battery_role": "target_primary",
            "battery": "gsm8k",
            "base_correct": False,
            "teacher_14b_correct": True,
        },
    ]
    result = paired_discordance(rows)
    assert result["rows"] == 2
    assert result["pooled_base_teacher_discordance"] == 0.5


def test_sequential_simulator_reports_action_count_and_probability() -> None:
    result = simulate_rule(
        rows=64,
        looks=6,
        discordance=0.2,
        correlation=0.5,
        alpha=0.1,
        decision_margin=-0.03,
        true_difference=-0.05,
        campaigns=300,
        seed=20260810,
        batch_campaigns=50,
    )
    assert 0.0 <= result["estimated_action_probability"] <= 1.0
    assert result["expected_actions_per_campaign"] >= result["estimated_action_probability"]
    assert result["consecutive_looks"] == 2
