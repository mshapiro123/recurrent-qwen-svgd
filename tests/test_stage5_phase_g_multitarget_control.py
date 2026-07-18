from __future__ import annotations

import pytest

from training.phase_g_multitarget_spec import required_posterior_control_thresholds


def environment() -> dict[str, str]:
    return {
        "STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS": "32",
        "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_TARGET_RATE": "0.60",
        "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_TARGET_LIFT": "0.15",
        "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS": "24",
        "STAGE5_PHASE_G_MULTITARGET_MAX_TEACHER_PRIOR_TARGET_LIFT_PVALUE": "0.05",
    }


def test_multitarget_control_refuses_to_train_without_locked_thresholds() -> None:
    with pytest.raises(RuntimeError, match="thresholds must be locked"):
        required_posterior_control_thresholds({})


def test_multitarget_control_accepts_explicit_prelocked_thresholds() -> None:
    thresholds = required_posterior_control_thresholds(environment())

    assert thresholds["STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS"] == 32
    assert thresholds["STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_TARGET_RATE"] == 0.60


def test_multitarget_control_uses_the_strategy_locked_a0_margins() -> None:
    thresholds = required_posterior_control_thresholds(environment())

    assert thresholds == {
        "STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS": 32,
        "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_TARGET_RATE": 0.60,
        "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_TARGET_LIFT": 0.15,
        "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS": 24,
        "STAGE5_PHASE_G_MULTITARGET_MAX_TEACHER_PRIOR_TARGET_LIFT_PVALUE": 0.05,
    }
