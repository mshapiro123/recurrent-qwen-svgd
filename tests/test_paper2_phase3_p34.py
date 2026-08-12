from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.paper2_phase3_p34 import (
    LossShareBounds,
    TaskInferenceContract,
    classify_loss_shares,
    controller_transition,
    gap_closed,
    initial_annealing_state,
)


def test_task_contract_is_exact_and_rejects_drift() -> None:
    TaskInferenceContract().validate()
    with pytest.raises(RuntimeError, match="contract changed"):
        TaskInferenceContract(flow_loops=3).validate()


def test_controller_opens_at_rung_one_and_moves_once() -> None:
    state = initial_annealing_state(chi_max=0.02)
    assert state.rung == 1
    advanced, receipt = controller_transition(
        state, pi_dep=0.26, chi=0.01, tier_w_event=False
    )
    assert advanced.rung == 2
    assert receipt["reason"] == "advance"
    demoted, receipt = controller_transition(
        advanced, pi_dep=1.0, chi=0.0, tier_w_event=True
    )
    assert demoted.rung == 1
    assert receipt["reason"] == "tier_w_demote"


def test_loss_contract_warns_then_stops_per_named_loss() -> None:
    bounds = LossShareBounds()
    shares = {"kl": 0.34, "aim": 0.20, "ce": 0.10, "gate": 0.03, "preserve": 0.20}
    first = classify_loss_shares(shares, bounds=bounds)
    second = classify_loss_shares(
        shares, bounds=bounds, prior_consecutive_misses=first["consecutive_misses"]
    )
    assert first["classification"] == "warn"
    assert first["failed_contracts"] == ["kl"]
    assert second["classification"] == "stop"


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
