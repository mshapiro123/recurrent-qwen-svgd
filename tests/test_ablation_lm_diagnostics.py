from __future__ import annotations

import json

import pytest
import torch

from models.ablation_lm.diagnostics import (
    RouterMomentSnapshot,
    SYM_COLLAPSE_WINDOW,
    SymmetryCollapseBlocked,
    SymmetryCollapseTracker,
    router_calibration_stability,
)


def test_router_calibration_cannot_freeze_at_step_zero_or_one_window() -> None:
    step_zero = [RouterMomentSnapshot(step=0, mean=0.0, std=1.0)] * 1
    one_window = [
        RouterMomentSnapshot(step=step, mean=0.0, std=1.0)
        for step in range(4)
    ]

    assert not router_calibration_stability(step_zero, window=2).ready
    assert not router_calibration_stability(one_window, window=4).ready


def test_router_calibration_requires_stable_nonzero_trajectory_windows() -> None:
    stable = [
        RouterMomentSnapshot(step=step, mean=0.5 + 1e-4 * step, std=1.0)
        for step in range(1, 17)
    ]
    drifting = [
        RouterMomentSnapshot(step=step, mean=float(step), std=1.0 + step / 10)
        for step in range(1, 17)
    ]

    stable_decision = router_calibration_stability(stable, window=8, minimum_step=16)
    drifting_decision = router_calibration_stability(drifting, window=8, minimum_step=16)

    assert stable_decision.ready
    assert stable_decision.reason == "stable_trajectory_windows"
    assert not drifting_decision.ready
    assert drifting_decision.reason == "router_moments_still_drifting"


def test_router_calibration_rejects_step_zero_gaps_and_nonfinite_tolerances() -> None:
    gapped = [
        RouterMomentSnapshot(step=step, mean=0.5, std=1.0)
        for step in (0, 1, 2, 16)
    ]

    decision = router_calibration_stability(gapped, window=2, minimum_step=16)

    assert not decision.ready
    assert decision.reason == "nonadjacent_or_step_zero_windows"
    with pytest.raises(ValueError, match="finite and positive"):
        router_calibration_stability(gapped, relative_tolerance=float("nan"))


def test_sym_collapse_trips_on_exact_thousandth_consecutive_eligible_step() -> None:
    tracker = SymmetryCollapseTracker(
        {"core.0.attention.q": 0.2, "core.0.ffn.up": torch.tensor(0.4)}
    )

    for step in range(SYM_COLLAPSE_WINDOW - 1):
        receipt = tracker.observe(
            step,
            {"core.0.attention.q": 0.1, "core.0.ffn.up": 0.5},
        )

    assert receipt.step == 998
    assert dict(receipt.delta_ratio) == {
        "core.0.attention.q": 0.1,
        "core.0.ffn.up": 0.5,
    }
    assert dict(receipt.below_initial_consecutive_steps) == {
        "core.0.attention.q": 999,
        "core.0.ffn.up": 0,
    }
    assert receipt.sym_collapse_window == 1_000

    with pytest.raises(SymmetryCollapseBlocked, match="SYM-COLLAPSE") as caught:
        tracker.observe(
            SYM_COLLAPSE_WINDOW - 1,
            {"core.0.attention.q": 0.1, "core.0.ffn.up": 0.5},
        )
    assert caught.value.step == 999
    assert caught.value.matrix_name == "core.0.attention.q"
    assert caught.value.consecutive_steps == 1_000
    assert caught.value.receipt.step == 999
    assert dict(caught.value.receipt.below_initial_consecutive_steps)[
        "core.0.attention.q"
    ] == 1_000


def test_sym_collapse_reset_and_fail_closed_step_and_matrix_contract() -> None:
    tracker = SymmetryCollapseTracker({"paired": 0.2})
    tracker.observe(10, {"paired": 0.1})
    reset = tracker.observe(11, {"paired": 0.2})
    assert reset.below_initial_consecutive_steps == (("paired", 0),)
    with pytest.raises(ValueError, match="strict consecutive order"):
        tracker.observe(13, {"paired": 0.1})

    fresh = SymmetryCollapseTracker({"paired": 0.2})
    with pytest.raises(ValueError, match="exactly match the T7"):
        fresh.observe(0, {"different": 0.1})
    with pytest.raises(ValueError, match="strictly positive"):
        SymmetryCollapseTracker({"paired": 0.0})


def test_sym_collapse_tracker_resume_preserves_partial_streak_exactly() -> None:
    tracker = SymmetryCollapseTracker({"paired": 0.2})
    for step in range(17):
        tracker.observe(step, {"paired": 0.1})

    durable_state = json.loads(json.dumps(tracker.state_dict()))
    restored = SymmetryCollapseTracker.from_state_dict(durable_state)
    receipt = restored.observe(17, {"paired": 0.1})

    assert receipt.below_initial_consecutive_steps == (("paired", 18),)
    assert restored.last_step == 17
    malformed = dict(durable_state)
    malformed["sym_collapse_window"] = 999
    with pytest.raises(ValueError, match="exactly 1000"):
        SymmetryCollapseTracker.from_state_dict(malformed)
