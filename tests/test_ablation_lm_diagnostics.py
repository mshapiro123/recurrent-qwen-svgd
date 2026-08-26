from __future__ import annotations

import pytest

from models.ablation_lm.diagnostics import (
    RouterMomentSnapshot,
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
