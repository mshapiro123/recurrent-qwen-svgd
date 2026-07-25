import pytest

from training.internal_think_token_t1 import (
    causal_override_schedule,
    ordinary_least_squares_slope,
    phase_t1_gate_verdict,
    phase_t1_seed_1_trigger,
    stage_boundary_liveness_verdict,
)
from training.run_internal_think_token_t1_lite import (
    DeviceEMA,
    verify_stage_checkpoint_manifest,
    write_stage_checkpoint_bundle,
)

import torch


def _selection(correct: int = 128):
    return {str(depth): {"correct": correct, "total": 128} for depth in range(1, 9)}


def test_stage_liveness_aborts_only_on_joint_flat_and_zero_stop():
    aborted = stage_boundary_liveness_verdict(
        [(100, 0.7), (200, 0.7)], stop_correct=0, stop_total=16
    )
    assert aborted["abort_for_diagnosis"] is True
    assert aborted["registered_attempt_consumed"] is False

    learning = stage_boundary_liveness_verdict(
        [(100, 0.7), (200, 0.5)], stop_correct=0, stop_total=16
    )
    assert learning["control_loss_flat"] is False
    assert learning["abort_for_diagnosis"] is False

    live = stage_boundary_liveness_verdict(
        [(100, 0.7), (200, 0.7)], stop_correct=1, stop_total=16
    )
    assert live["abort_for_diagnosis"] is False


def test_ols_rejects_nonidentifying_points():
    assert ordinary_least_squares_slope([(0, 1.0), (100, 0.5)]) == pytest.approx(-0.005)
    with pytest.raises(ValueError, match="distinct"):
        ordinary_least_squares_slope([(1, 1.0), (1, 0.5)])


def test_causal_override_schedule_has_exact_registered_counts():
    total_stops = 0
    total_continues = 0
    for depth in range(1, 9):
        schedule = causal_override_schedule(depth)
        assert len(schedule["forced_stops"]) == depth
        assert schedule["forced_stops"][-1]["expected_executed_loops"] == depth
        assert schedule["forced_continue"]["expected_executed_loops"] == depth + 1
        total_stops += depth * 128
        total_continues += 128
    assert total_stops == 4608
    assert total_continues == 1024


def test_all_four_gates_are_jointly_required():
    passed = phase_t1_gate_verdict(
        forced_correct=1000,
        forced_total=1024,
        self_halted_correct=980,
        self_halted_total=1024,
        selection_by_depth=_selection(120),
        causal_exact=5632,
        causal_total=5632,
    )
    assert passed["all_four_passed"] is True

    failed = phase_t1_gate_verdict(
        forced_correct=1000,
        forced_total=1024,
        self_halted_correct=980,
        self_halted_total=1024,
        selection_by_depth=_selection(120),
        causal_exact=5631,
        causal_total=5632,
    )
    assert failed["all_four_passed"] is False
    assert failed["verdict"] == "registered_negative"


def test_seed_1_trigger_covers_pass_and_locked_near_threshold_cases():
    passed = phase_t1_gate_verdict(
        forced_correct=1000,
        forced_total=1024,
        self_halted_correct=980,
        self_halted_total=1024,
        selection_by_depth=_selection(120),
        causal_exact=5632,
        causal_total=5632,
    )
    assert phase_t1_seed_1_trigger(passed)["triggered"] is True
    assert phase_t1_seed_1_trigger(passed)["reasons"] == ["full_pass"]

    near_gate3 = phase_t1_gate_verdict(
        forced_correct=1000,
        forced_total=1024,
        self_halted_correct=980,
        self_halted_total=1024,
        selection_by_depth={
            str(depth): {"correct": 112 if depth == 8 else 113, "total": 128}
            for depth in range(1, 9)
        },
        causal_exact=5632,
        causal_total=5632,
    )
    trigger = phase_t1_seed_1_trigger(near_gate3)
    assert trigger["triggered"] is True
    assert trigger["near_threshold"] is True
    assert "near_gate3" in trigger["reasons"]


def test_stage_reset_ema_restarts_from_current_raw_weights() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    ema = DeviceEMA([("weight", parameter)], decay=0.9)
    with torch.no_grad():
        parameter.fill_(3.0)
    ema.update([("weight", parameter)])
    assert torch.allclose(ema.shadow["weight"], torch.tensor([1.2]))
    ema.reset_from_parameters([("weight", parameter)])
    assert torch.equal(ema.shadow["weight"], torch.tensor([3.0]))


def test_stage_checkpoint_bundle_is_atomic_hashed_and_complete(tmp_path) -> None:
    local = tmp_path / "local"
    backup = tmp_path / "backup"
    receipt = write_stage_checkpoint_bundle(
        local_dir=local,
        backup_dir=backup,
        step=500,
        raw_state={"weight": torch.tensor([1.0])},
        continuous_ema_state={"weight": torch.tensor([2.0])},
        stage_reset_ema_state={"weight": torch.tensor([3.0])},
    )
    assert receipt["step"] == 500
    assert set(receipt["states"]) == {"raw", "continuous_ema", "stage_reset_ema"}
    for state in receipt["states"].values():
        assert state["local_sha256"] == state["backup_sha256"]
        assert not state["local_path"].endswith(".tmp")

    manifest = verify_stage_checkpoint_manifest(
        receipts=[receipt],
        required_steps=[500],
        require_backup=True,
    )
    assert manifest["complete"] is True
