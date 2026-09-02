from __future__ import annotations

import math

import pytest

from analysis.weft1_preflight_c1 import (
    ATTENTION_SCALE_AUTHORITY,
    ATTENTION_SCALE_AUTHORITY_BYTES,
    ATTENTION_SCALE_AUTHORITY_DRIVE_ID,
    ATTENTION_SCALE_AUTHORITY_SHA256,
    C1CoordinateCatch,
    C1_CPU_WIDTHS,
    C1_DEFERRED_GPU_WIDTHS,
    C1_TRAINING_STEPS,
    C1_WIDTH_DRIFT_LIMIT,
    PREFLIGHT_PROGRAM_SHA256,
    PREFLIGHT_RATIFICATION_SHA256,
    run_preflight_c1,
)


def test_c1_cpu_width_coordinate_check_returns_numbered_catch_without_patch() -> None:
    receipt = run_preflight_c1()

    assert receipt.program_sha256 == PREFLIGHT_PROGRAM_SHA256
    assert receipt.ratification_sha256 == PREFLIGHT_RATIFICATION_SHA256
    assert receipt.attention_scale_authority == ATTENTION_SCALE_AUTHORITY
    assert receipt.attention_scale_authority.endswith("#8.1")
    assert receipt.attention_scale_authority_bytes == ATTENTION_SCALE_AUTHORITY_BYTES == 61_329
    assert receipt.attention_scale_authority_sha256 == ATTENTION_SCALE_AUTHORITY_SHA256
    assert receipt.attention_scale_authority_drive_id == ATTENTION_SCALE_AUTHORITY_DRIVE_ID
    assert receipt.cpu_widths == C1_CPU_WIDTHS == (64, 128, 256)
    assert receipt.deferred_gpu_widths == C1_DEFERRED_GPU_WIDTHS == (512,)
    assert receipt.training_steps == C1_TRAINING_STEPS == 10
    assert receipt.width_drift_limit == C1_WIDTH_DRIFT_LIMIT == 2.0
    assert receipt.a100_hours == 0.0
    assert receipt.not_materialized_integrations == (
        "integrated_rotor_carrier",
        "per_band_callosum",
        "sidecar",
    )
    assert all(run.unique_decoder_blocks == 10 for run in receipt.width_runs)
    assert all(run.executed_decoder_block_passes == 16 for run in receipt.width_runs)

    assert {
        item.surface for item in receipt.attention_scale_evidence
    } == {
        "integrated_grouped_query_attention",
        "standalone_bicameral_transformer_block",
    }
    assert tuple(item.head_dim for item in receipt.attention_scale_evidence) == (
        8,
        8,
        16,
        16,
        32,
        32,
    )
    assert all(
        item.sdpa_matches == "ordinary_inverse_sqrt_head_dim"
        and item.math_matches == "ordinary_inverse_sqrt_head_dim"
        and not item.passed
        for item in receipt.attention_scale_evidence
    )
    for item in receipt.attention_scale_evidence:
        assert item.ratified_scale == 1.0 / item.head_dim
        assert item.ordinary_scale == 1.0 / math.sqrt(item.head_dim)
        assert item.ordinary_to_ratified_ratio == pytest.approx(math.sqrt(item.head_dim))
        assert item.sdpa_error_to_ordinary < 1e-5
        assert item.math_error_to_ordinary < 1e-5
        assert item.sdpa_error_to_ratified > 1e-6
        assert item.math_error_to_ratified > 1e-6

    coordinates = {
        (coordinate.phase, coordinate.module_name): coordinate
        for coordinate in receipt.activation_coordinates
    }
    assert coordinates[("init", "prelude.0.feed_forward")].maximum_to_minimum_ratio > 7
    assert not coordinates[("init", "prelude.0.feed_forward")].passed
    assert coordinates[("init", "prelude.0.attention")].maximum_to_minimum_ratio > 3
    assert not coordinates[("init", "prelude.0.attention")].passed
    assert coordinates[("after_steps", "core.0.block_output")].maximum_to_minimum_ratio > 4
    assert not coordinates[("after_steps", "core.0.block_output")].passed

    assert receipt.attention_scale_passed is False
    assert receipt.activation_coordinate_passed is False
    assert receipt.passed is False
    assert receipt.catch_number == 28
    assert receipt.disposition == "catch_28_return_to_strategy_no_model_patch"
    with pytest.raises(C1CoordinateCatch, match=r"CATCH #28"):
        receipt.require_passed()


def test_c1_runner_rejects_unregistered_or_gpu_widths_in_cpu_slice() -> None:
    with pytest.raises(ValueError, match="non-empty unique"):
        run_preflight_c1(widths=())
    with pytest.raises(ValueError, match="CPU-scoped"):
        run_preflight_c1(widths=(512,), training_steps=1)
    with pytest.raises(ValueError, match="positive integer"):
        run_preflight_c1(widths=(64,), training_steps=0)
