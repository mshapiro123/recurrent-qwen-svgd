from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from training.paper2_phase3_p34 import (
    LossShareBounds,
    P34_CHI_MAX_BY_RUNG,
    SlotSupervisionLift,
    TaskInferenceContract,
    classify_loss_shares,
    controller_transition,
    gap_closed,
    initial_annealing_state,
    postclip_loss_gradient_norms,
    share_targets,
    slot_supervision_loss,
    solve_static_loss_weights,
    weighted_p34_total,
)


def test_task_contract_is_exact_and_rejects_drift() -> None:
    TaskInferenceContract().validate()
    with pytest.raises(RuntimeError, match="contract changed"):
        TaskInferenceContract(flow_loops=3).validate()


def test_controller_opens_at_rung_one_and_moves_once() -> None:
    state = initial_annealing_state()
    assert state.rung == 1
    assert state.chi_max == pytest.approx(0.0005)
    advanced, receipt = controller_transition(
        state, pi_dep=0.26, chi=0.0004, tier_w_event=False
    )
    assert advanced.rung == 2
    assert receipt["reason"] == "advance"
    demoted, receipt = controller_transition(
        advanced, pi_dep=1.0, chi=0.0, tier_w_event=True
    )
    assert demoted.rung == 1
    assert receipt["reason"] == "tier_w_demote"


def test_controller_uses_rung_specific_collateral_limits() -> None:
    assert P34_CHI_MAX_BY_RUNG == (0.0005, 0.0005, 0.0005, 0.0010)
    state = initial_annealing_state()
    held, receipt = controller_transition(
        state, pi_dep=0.50, chi=0.0006, tier_w_event=False
    )
    assert held.rung == 1
    assert receipt["chi_max_before"] == pytest.approx(0.0005)


def test_loss_contract_observes_warns_then_stops_per_named_loss() -> None:
    bounds = LossShareBounds()
    shares = {"kl": 0.34, "aim": 0.20, "ce": 0.10, "gate": 0.03, "preserve": 0.20}
    first = classify_loss_shares(shares, bounds=bounds)
    second = classify_loss_shares(shares, bounds=bounds, prior_consecutive_misses=1)
    third = classify_loss_shares(shares, bounds=bounds, prior_consecutive_misses=2)
    fourth = classify_loss_shares(shares, bounds=bounds, prior_consecutive_misses=3)
    assert first["classification"] == "breach_observed"
    assert first["failed_contracts"] == ["kl"]
    assert second["classification"] == "warn"
    assert third["classification"] == "warn"
    assert fourth["classification"] == "stop"


def test_slot_arm_loss_contract_binds_slot_independently() -> None:
    shares = {
        "kl": 0.40,
        "aim": 0.16,
        "ce": 0.11,
        "gate": 0.04,
        "slot": 0.09,
        "preserve": 0.20,
    }
    result = classify_loss_shares(shares)
    assert result["classification"] == "breach_observed"
    assert result["failed_contracts"] == ["slot"]


def test_share_solver_meets_every_arm_floor() -> None:
    for slot_arm in (False, True):
        targets = share_targets(slot_arm=slot_arm)
        solved = solve_static_loss_weights(
            {name: float(index + 1) for index, name in enumerate(targets)},
            slot_arm=slot_arm,
        )
        assert sum(solved["solved_shares"].values()) == pytest.approx(1.0)
        assert solved["preservation_at_or_below_25_percent"]
        assert solved["solved_shares"]["kl"] >= 0.35
        assert solved["solved_shares"]["aim"] >= 0.15
        assert solved["solved_shares"]["ce"] >= 0.10
        assert solved["solved_shares"]["gate"] >= 0.03
        if slot_arm:
            assert solved["solved_shares"]["slot"] >= 0.10


def test_postclip_attribution_includes_slot_lift_in_heads_group() -> None:
    class ToyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bridge = torch.nn.Linear(1, 1, bias=False)
            self.control = torch.nn.Linear(1, 1, bias=False)

    module = ToyModule()
    lift = SlotSupervisionLift(latent_dim=1, hidden_size=1)
    parameters = [*module.parameters(), *lift.parameters()]
    values = [module.bridge.weight, module.control.weight, lift.lift.weight]
    losses = {
        "kl": values[0].sum(),
        "aim": 2.0 * values[0].sum(),
        "ce": values[1].sum(),
        "gate": 2.0 * values[1].sum(),
        "slot": values[2].sum(),
        "preserve": 3.0 * values[0].sum(),
    }
    result = postclip_loss_gradient_norms(
        losses=losses,
        module=module,
        parameters=parameters,
        slot_lift=lift,
    )
    assert set(result["postclip_gradient_norms"]) == set(losses)
    assert result["group_clip_ceilings"] == {"bridge": 0.5, "heads": 1.0}
    assert sum(result["unit_weight_shares"].values()) == pytest.approx(1.0)


def test_slot_lift_is_zero_init_and_only_lift_receives_gradients() -> None:
    torch.manual_seed(0)
    lift = SlotSupervisionLift(latent_dim=3, hidden_size=5)
    assert torch.count_nonzero(lift.lift.weight) == 0
    flow_states = tuple(torch.randn(2, 4, 3) for _ in range(5))
    tied_weight = torch.randn(7, 5, requires_grad=True)
    teacher_tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    teacher_mask = torch.ones_like(teacher_tokens, dtype=torch.bool)
    loss, telemetry = slot_supervision_loss(
        lift=lift,
        flow_states=flow_states,
        tied_weight=tied_weight,
        teacher_tokens=teacher_tokens,
        teacher_mask=teacher_mask,
    )
    loss.backward()
    assert lift.lift.weight.grad is not None
    assert tied_weight.grad is None
    assert telemetry["future_slot_indices"] == [0, 1, 2, 3]
    assert telemetry["tied_head_frozen"]


def test_weighted_total_accepts_main_and_slot_arms() -> None:
    main_losses = {name: torch.tensor(1.0) for name in share_targets(slot_arm=False)}
    slot_losses = {name: torch.tensor(1.0) for name in share_targets(slot_arm=True)}
    assert weighted_p34_total(main_losses, {name: 1.0 for name in main_losses}) == 5.0
    assert weighted_p34_total(slot_losses, {name: 1.0 for name in slot_losses}) == 6.0


def test_gap_closed_keeps_raw_delta_and_handles_nonpositive_gap() -> None:
    result = gap_closed(augmented=0.60, base=0.50, teacher=0.70)
    assert result["raw_delta"] == pytest.approx(0.10)
    assert result["gap_closed"] == pytest.approx(0.50)
    assert not gap_closed(augmented=0.60, base=0.70, teacher=0.60)["defined"]


def test_pending_lock_cannot_authorize_training() -> None:
    path = Path(__file__).resolve().parents[1] / "training/paper2_phase3_p34_preregistration.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    assert lock["status"] == "prerequisites_pending"
    assert not lock["locked_before_training"]
    assert not lock["training_authorized"]
    assert lock["unresolved_lock_fields"]
    assert not lock["boundaries"]["p34_training_runner_present"]
    assert lock["authority"]["sha256"] == (
        "80cb1b13eb48ffff064ff7cc6c0d02de773dfec80924c1c50736115821c97ce4"
    )
