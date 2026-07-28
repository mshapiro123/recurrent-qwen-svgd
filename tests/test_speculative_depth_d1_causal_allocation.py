from __future__ import annotations

from pathlib import Path

import torch

from eval.eval_speculative_depth_d1_causal_allocation import (
    accepted_loss_decomposition,
    d1_label_balance,
    deployed_policy_frontier,
    deterministic_sample_rows,
    recurrent_states_token_loop,
    oracle_frontier,
    policy_confusion,
    replay_equivalence,
    source_fold,
    transition_labels,
)


ROOT = Path(__file__).resolve().parents[1]


def test_recurrent_states_are_normalized_to_token_loop_hidden() -> None:
    batch = 1
    sequence = 344
    hidden = 12
    loops = 4
    captured = [
        torch.full((batch, sequence, hidden), float(loop_index))
        for loop_index in range(loops)
    ]

    states = recurrent_states_token_loop(
        captured,
        expected_loops=loops,
        expected_tokens=sequence - 1,
    )

    assert states.shape == (sequence - 1, loops, hidden)
    for loop_index in range(loops):
        assert torch.all(states[:, loop_index] == float(loop_index))


def test_utility_labels_continue_only_for_causal_help() -> None:
    assert transition_labels([False, True, True, False]) == ["helps", "neutral", "hurts"]
    assert transition_labels([False, False, False, False]) == ["neutral", "neutral", "neutral"]


def test_oracle_frontier_prices_harm_and_prefers_shallow_ties() -> None:
    matches = torch.tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, False, True],
        ]
    )
    zero, expensive = oracle_frontier(matches, penalties=(0.0, 1.0))
    assert zero["correct"] == 3
    assert zero["selected_depth_counts"] == {"1": 1, "2": 1, "3": 0, "4": 1}
    assert expensive["selected_depth_counts"] == {"1": 3, "2": 0, "3": 0, "4": 0}


def test_source_fold_keeps_all_positions_from_one_source_together() -> None:
    assert source_fold(17, "general") == source_fold(17, "general")
    assert 0 <= source_fold(18, "code") < 5


def test_deterministic_label_train_sample_is_repeatable_and_position_capped() -> None:
    rows = [
        {"row_id": f"row-{index}", "input_ids": list(range(101))}
        for index in range(20)
    ]
    first, positions = deterministic_sample_rows(rows, max_positions=550)
    second, repeated = deterministic_sample_rows(rows, max_positions=550)
    assert first == second
    assert positions == repeated == 600
    assert len(first) == 6


def test_policy_confusion_reports_forced_and_reachable_tables() -> None:
    matches = torch.tensor([[False, True, True, True], [True, False, False, False]])
    controls = torch.tensor([[0, 1, 1, 1], [1, 0, 0, 0]])
    result = policy_confusion(matches, controls)
    assert result["1"]["all_forced_positions"]["continue_true_continue"] == 1
    assert result["1"]["all_forced_positions"]["stop_true_stop"] == 1
    assert result["2"]["deployed_reachable_positions"]["positions"] == 1


def test_d1_label_balance_and_deployed_frontier_share_one_objective() -> None:
    matches = torch.tensor([[False, True, True, True], [True, False, False, False]])
    controls = torch.tensor([[0, 1, 1, 1], [1, 0, 0, 0]])
    balance = d1_label_balance(matches)
    frontier = deployed_policy_frontier(matches, controls, penalties=(0.0, 0.5))
    assert balance["continue"] == 1
    assert balance["total_transition_labels"] == 6
    assert frontier[0]["accuracy"] == 1.0
    assert frontier[0]["mean_loops"] == 1.5
    assert frontier[1]["net_utility"] == 0.75


def test_accepted_loss_decomposition_separates_weight_and_policy_damage() -> None:
    matches = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, True, True, True],
        ]
    )
    controls = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
    metadata = [
        {"plain_accepted": True, "stratum": "general", "severity_bin": "q1"},
        {"plain_accepted": True, "stratum": "code", "severity_bin": "q2"},
        {"plain_accepted": True, "stratum": "general", "severity_bin": "q1"},
    ]
    result = accepted_loss_decomposition(matches, controls, metadata)
    assert result["accepted_positions_lost"] == 2
    assert result["loop1_weight_regression"] == 1
    assert result["post_loop_policy_losses"] == 1
    assert result["preventable_fraction"] == 0.5


def test_audit_contract_forbids_training_and_discloses_missing_margin() -> None:
    evaluator = (ROOT / "eval/eval_speculative_depth_d1_causal_allocation.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "colab/run_stage5_paper2_d1_causal_allocation_audit.py").read_text(
        encoding="utf-8"
    )
    spec = (ROOT / "docs/PAPER2_D1_CAUSAL_ALLOCATION_AUDIT_SPEC_20260727.md").read_text(
        encoding="utf-8"
    )
    assert "teacher_top1_top2_margin_available" in evaluator
    assert "teacher top-1/top-2 margin" in spec
    assert "STAGE5_PAPER2_D1_ALLOW_TRAINING" in runner
    assert ".backward(" not in evaluator
    assert "torch.optim" not in evaluator
    assert "training_started\": False" in evaluator
    assert "optimizer_steps\": 0" in evaluator
    assert "replay_equivalence" in evaluator
