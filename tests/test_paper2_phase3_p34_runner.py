from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from training.paper2_phase3_p34 import (
    P34_FLOW_LOOPS,
    initial_annealing_state,
    slot_supervision_loss,
)
from training.run_paper2_phase3_p34 import _advance_sequential_rule, _task_guardrail


ROOT = Path(__file__).resolve().parents[1]


def test_task_guardrail_uses_paired_task_differences() -> None:
    lock = {"guardrails": {
        "tier_s_one_sided_alpha": 0.1,
        "tier_s_decision_margin": -0.03,
        "tier_w_drop_class": -0.03,
    }}
    rows = [
        {"base_correct": True, "augmented_correct": False} for _ in range(60)
    ] + [
        {"base_correct": True, "augmented_correct": True} for _ in range(40)
    ]
    read = _task_guardrail(rows, lock)
    assert read["mean_augmented_minus_base"] == -0.6
    assert read["tier_s_condition"]
    assert read["tier_w_condition"]


def test_runner_controller_initialization_accepts_locked_chi_vector() -> None:
    lock = json.loads(
        (ROOT / "training/paper2_phase3_p34_preregistration.json").read_text()
    )
    state = initial_annealing_state(
        chi_max_by_rung=tuple(lock["guardrails"]["chi_max_by_rung"])
    )
    assert state.rung == lock["controller"]["initial_rung"]
    assert list(state.chi_max_by_rung) == lock["guardrails"]["chi_max_by_rung"]


def test_runner_is_sealed_partition_blind_and_resumable() -> None:
    source = (ROOT / "training/run_paper2_phase3_p34.py").read_text(encoding="utf-8")
    assert "task_rows_look_" in source
    assert "checkpoint_step_" in source
    assert 'saved.get("lock_sha256") != sha256_file(args.lock)' in source
    assert source.count("module.bridge.set_gate_ceiling(controller.gate_ceiling)") == 3
    assert "step % SHARE_WINDOW_STEPS == 0" in source
    assert '"overlap_with_prior_window_steps": 0' in source
    assert '"reason": "loss_share_contract_demote"' in source
    assert "share_transition is not None or stop_reason is not None" in source
    assert '"sampled_depth_mixture_solve": sampled_depth_solve' in source
    assert '"optimizer_constructed": False' in source
    assert '"confirm_scored": False' in source
    assert '"eval_e_scored": False' in source
    lock = json.loads((ROOT / "training/paper2_phase3_p34_preregistration.json").read_text())
    assert lock["guardrails"]["look_count"] == 20


def test_colab_target_wires_the_approved_three_arm_campaign() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_PAPER2_PHASE3_P34_CELL.py").read_text(encoding="utf-8")
    launcher = (ROOT / "colab/run_stage5_paper2_phase3_p34.py").read_text(encoding="utf-8")
    assert '"paper2_phase3_p34"' in bootstrap
    assert "paper2_phase3_p34_campaign_v1" in cell
    assert 'parser.add_argument("--arm"' in launcher
    assert "I1_ID" in launcher


class _IdentityLift(torch.nn.Module):
    def forward(self, state: torch.Tensor, tied_weight: torch.Tensor) -> torch.Tensor:
        return state @ tied_weight.T


def test_slot_supervision_accepts_the_registered_sampled_depths() -> None:
    tied_weight = torch.eye(4)
    teacher_tokens = torch.zeros((2, 4), dtype=torch.long)
    teacher_mask = torch.ones((2, 4), dtype=torch.bool)
    for depth in range(1, P34_FLOW_LOOPS + 1):
        states = [torch.zeros((2, 4, 4)) for _ in range(depth + 1)]
        loss, metrics = slot_supervision_loss(
            lift=_IdentityLift(),
            flow_states=states,
            tied_weight=tied_weight,
            teacher_tokens=teacher_tokens,
            teacher_mask=teacher_mask,
        )
        assert torch.isfinite(loss)
        assert metrics["executed_loops"] == depth
        assert metrics["deep_supervision_weights"] == pytest.approx(
            [index / P34_FLOW_LOOPS for index in range(1, depth + 1)]
        )


def test_main_and_slot_use_the_same_depth_rng_schedule() -> None:
    source = (ROOT / "training/run_paper2_phase3_p34.py").read_text(encoding="utf-8")
    assert "manual_seed(20260813 + args.seed)" in source
    assert "depth = sampled_depth(generator=generator)" in source
    assert "if slot_lift is not None else sampled_depth" not in source


def test_sequential_rule_emits_non_overlapping_nested_events() -> None:
    tier_w_streak = 0
    tier_s_streak = 0
    tier_w_events = []
    tier_s_events = []
    for look in range(1, 9):
        tier_w_streak, tier_w_event = _advance_sequential_rule(
            condition=True, prior_streak=tier_w_streak, required_looks=2
        )
        tier_s_streak, tier_s_event = _advance_sequential_rule(
            condition=True, prior_streak=tier_s_streak, required_looks=4
        )
        if tier_w_event:
            tier_w_events.append(look)
        if tier_s_event:
            tier_s_events.append(look)
    assert tier_w_events == [2, 4, 6, 8]
    assert tier_s_events == [4, 8]
    assert set(tier_s_events).issubset(tier_w_events)


def test_sequential_rule_resets_after_recovery() -> None:
    streak = 0
    events = []
    for condition in (True, True, False, True, True, True, True):
        streak, event = _advance_sequential_rule(
            condition=condition, prior_streak=streak, required_looks=4
        )
        events.append(event)
    assert events == [False, False, False, False, False, False, True]
