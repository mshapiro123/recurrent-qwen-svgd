from __future__ import annotations

import pytest

from training.weft1_gtok_confirmation_v2 import (
    ConfirmationBurstGateEvidenceV2,
    ConfirmationBurstGateViolationV2,
    ConfirmationBurstFlopReceiptV2,
    confirmation_flops_within_target_v2,
    confirmation_physical_burst_evidence_sha256_v2,
    confirmation_retry_steps_v2,
    floor_arm_mean_flops_v2,
    precompute_byte_checkpoint_steps_v2,
    prelaunch_confirmation_steps_v2,
)
from training.weft1_gtok_v2_contract import ArmCalibrationProjectionV2, GTokV2Stop


def test_prelaunch_formula_is_exact_above_float_integer_precision() -> None:
    seed0 = 2**63 + 101
    seed1 = 2**63 + 104
    arm_mean = floor_arm_mean_flops_v2(seed0, seed1)
    assert arm_mean == (seed0 + seed1) // 2

    target = arm_mean - 12_345
    base_steps = 7_777
    expected = (target * base_steps) // arm_mean
    assert prelaunch_confirmation_steps_v2(
        target_flops=target,
        arm_mean_flops=arm_mean,
        byte_matched_optimizer_steps=base_steps,
    ) == expected


def test_q4_strict_range_over_even_sample_median_boundary() -> None:
    at_boundary = (1_000,) * 99 + (1_010,)
    receipt = ConfirmationBurstFlopReceiptV2(
        ordered_step_flops=at_boundary,
        prelaunch_arm_mean_flops=100_000,
        byte_matched_optimizer_steps=100,
    )
    assert receipt.measured_flops == sum(at_boundary)

    with pytest.raises(GTokV2Stop, match="range/median"):
        ConfirmationBurstFlopReceiptV2(
            ordered_step_flops=(1_000,) * 99 + (1_011,),
            prelaunch_arm_mean_flops=100_000,
            byte_matched_optimizer_steps=100,
        )


def test_q5_requires_burst_mean_within_prelaunch_one_percent() -> None:
    with pytest.raises(GTokV2Stop, match="pre-launch f_step"):
        ConfirmationBurstFlopReceiptV2(
            ordered_step_flops=(1_000,) * 100,
            prelaunch_arm_mean_flops=99_000,
            byte_matched_optimizer_steps=100,
        )

    # Individual steps may straddle the inherited value while the governed
    # mean and the independent range/median statistic both remain in band.
    receipt = ConfirmationBurstFlopReceiptV2(
        ordered_step_flops=(995, 1_005) * 50,
        prelaunch_arm_mean_flops=100_000,
        byte_matched_optimizer_steps=100,
    )
    assert receipt.measured_flops == 100_000


def test_q4_failure_retains_ordered_evidence_and_physical_attempt_identity() -> None:
    evidence = ConfirmationBurstGateEvidenceV2(
        ordered_step_flops=(1_000,) * 99 + (1_011,),
        prelaunch_arm_mean_flops=100_000,
        byte_matched_optimizer_steps=100,
    )
    assert evidence.status == "STOP_RANGE_OVER_MEDIAN_ABOVE_ONE_PERCENT"
    with pytest.raises(ConfirmationBurstGateViolationV2) as caught:
        raise ConfirmationBurstGateViolationV2(evidence)
    assert caught.value.evidence.receipt_sha256 == evidence.receipt_sha256

    green = ConfirmationBurstFlopReceiptV2(
        ordered_step_flops=(1_000,) * 100,
        prelaunch_arm_mean_flops=100_000,
        byte_matched_optimizer_steps=100,
    )
    first = confirmation_physical_burst_evidence_sha256_v2(
        compute_attempt_id="attempt-0",
        execution_plan_sha256="a" * 64,
        burst=green,
    )
    second = confirmation_physical_burst_evidence_sha256_v2(
        compute_attempt_id="attempt-1",
        execution_plan_sha256="a" * 64,
        burst=green,
    )
    assert first != second


def test_end_band_and_retry_use_exact_integer_inequalities() -> None:
    target = 10_000
    assert confirmation_flops_within_target_v2(
        realized_flops=9_900, target_flops=target
    )
    assert confirmation_flops_within_target_v2(
        realized_flops=10_100, target_flops=target
    )
    assert not confirmation_flops_within_target_v2(
        realized_flops=9_899, target_flops=target
    )
    assert not confirmation_flops_within_target_v2(
        realized_flops=10_101, target_flops=target
    )
    assert confirmation_retry_steps_v2(
        target_flops=target,
        realized_flops=10_101,
        optimizer_steps=1_000,
    ) == (target * 1_000) // 10_101


def test_l6_checkpoints_are_first_exact_byte_fraction_crossings() -> None:
    cumulative = tuple(step * 10 for step in range(1, 101))
    assert precompute_byte_checkpoint_steps_v2(cumulative) == (25, 50, 100)

    overshooting = list(cumulative)
    overshooting[23] = 240
    overshooting[24] = 260
    assert precompute_byte_checkpoint_steps_v2(tuple(overshooting))[0] == 25


def test_confirmation_preflight_can_inherit_base_calibration_at_zero_spend() -> None:
    source_sha = "a" * 64
    projection = ArmCalibrationProjectionV2(
        scope="confirmation",
        vocab_size=32_768,
        calibration_attempt_id="inherited-base-calibration-v32768",
        calibration_steps=100,
        measured_tokens=80,
        measured_a100_microseconds=40,
        planned_tokens_per_run=160,
        projected_run_a100_microseconds=83,
        charged_calibration_a100_microseconds=0,
        measured_heldout_evaluation_a100_microseconds=1,
        heldout_evaluations_per_full_run=3,
        measured_output_surface_a100_microseconds=0,
        output_surface_benchmarks_per_full_run=0,
        projection_source="completed_base_calibration",
        projection_source_receipt_sha256=source_sha,
    )
    assert projection.charged_a100_microseconds == 0
    assert projection.projection_source_receipt_sha256 == source_sha
