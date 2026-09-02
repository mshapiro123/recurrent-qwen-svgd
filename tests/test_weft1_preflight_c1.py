from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from analysis.weft1_preflight_c1 import (
    BUILD_HANDOFF_AUTHORITY_BYTES,
    BUILD_HANDOFF_AUTHORITY_FILE,
    BUILD_HANDOFF_AUTHORITY_SHA256,
    C1AuthorityIntegrityError,
    C1CoordinateCatch,
    C1_BATCH_SIZE,
    C1_CATCH_NUMBER,
    C1_CPU_WIDTHS,
    C1_DEFERRED_GPU_WIDTHS,
    C1_EXECUTION_STATUS,
    C1_HEAD_DIM,
    C1_SEQUENCE_LENGTH,
    C1_TRAINING_STEPS,
    C1_WIDTH_DRIFT_LIMIT,
    PF2_AUTHORITY_BYTES,
    PF2_AUTHORITY_FILE,
    PF2_AUTHORITY_SHA256,
    S81_AUTHORITY_BYTES,
    S81_AUTHORITY_FILE,
    S81_AUTHORITY_SHA256,
    S81_FUTURE_HEAD_DIM_POLICY,
    run_preflight_c1,
    verify_c1_authorities,
)
from models.ablation_lm.config import MUP_D_HEAD_BASE


EXPECTED_UNBOUND_MUP_COMPONENTS = (
    "model_base_width_d_base",
    "internal_base_initialization_sigma_base",
    "complete_per_tensor_initialization_map",
    "internal_base_learning_rate_eta_base",
    "complete_per_tensor_learning_rate_map",
    "residual_branch_alpha",
    "embedding_multiplier",
    "residual_multiplier",
)


def test_pf2_s81_and_build_handoff_authorities_are_byte_exact() -> None:
    pf2, s81, build_handoff = verify_c1_authorities()

    assert pf2.filename == PF2_AUTHORITY_FILE
    assert pf2.expected_bytes == pf2.actual_bytes == PF2_AUTHORITY_BYTES == 13_097
    assert pf2.expected_sha256 == pf2.actual_sha256 == PF2_AUTHORITY_SHA256
    assert pf2.verified is True
    assert s81.filename == S81_AUTHORITY_FILE
    assert s81.expected_bytes == s81.actual_bytes == S81_AUTHORITY_BYTES == 3_403
    assert s81.expected_sha256 == s81.actual_sha256 == S81_AUTHORITY_SHA256
    assert s81.verified is True
    assert build_handoff.filename == BUILD_HANDOFF_AUTHORITY_FILE
    assert (
        build_handoff.expected_bytes
        == build_handoff.actual_bytes
        == BUILD_HANDOFF_AUTHORITY_BYTES
        == 61_329
    )
    assert (
        build_handoff.expected_sha256
        == build_handoff.actual_sha256
        == BUILD_HANDOFF_AUTHORITY_SHA256
    )
    assert build_handoff.verified is True


def test_authority_verification_fails_closed_when_files_are_absent(
    tmp_path: Path,
) -> None:
    with pytest.raises(C1AuthorityIntegrityError, match="missing C1 authority"):
        verify_c1_authorities(tmp_path)


def test_pf21_topology_is_bound_on_the_real_width_axis() -> None:
    receipt = run_preflight_c1()

    assert receipt.cpu_widths == C1_CPU_WIDTHS == (128, 256, 512)
    assert receipt.deferred_gpu_widths == C1_DEFERRED_GPU_WIDTHS == (1_024,)
    assert receipt.batch_size == C1_BATCH_SIZE == 2
    assert receipt.sequence_length == C1_SEQUENCE_LENGTH == 64
    assert receipt.training_steps == C1_TRAINING_STEPS == 10
    assert receipt.width_drift_limit == C1_WIDTH_DRIFT_LIMIT == 2.0
    assert receipt.authority_verified is True
    assert receipt.topology_verified is True
    assert receipt.attention_contract_bound is True
    assert receipt.future_head_dim_policy == S81_FUTURE_HEAD_DIM_POLICY
    assert receipt.future_head_dim_policy == (
        "future_WEFT_d_head_not_64_requires_explicit_base_shape_implementation"
    )
    assert BUILD_HANDOFF_AUTHORITY_BYTES == 61_329
    assert BUILD_HANDOFF_AUTHORITY_SHA256 == (
        "498f34b5966f0879c7f0a15ca8be02a603558781c35f59f03fb29cc9edd3eb02"
    )
    assert MUP_D_HEAD_BASE == C1_HEAD_DIM == 64

    for topology in receipt.width_topologies:
        width = topology.width
        assert topology.head_dim == 64
        assert topology.q_heads == width // 64
        assert topology.kv_heads == width // 128
        assert topology.q_heads == 2 * topology.kv_heads
        assert topology.d_ff == 11 * width // 4
        assert topology.scratch_lanes == 2
        assert topology.scratch_width_per_lane == width // 4
        assert topology.total_scratch_width == width // 2
        assert (
            topology.n_prelude_layers,
            topology.n_core_blocks,
            topology.n_coda_layers,
        ) == (4, 2, 4)
        assert topology.recurrent_steps == 4
        assert topology.unique_decoder_blocks == 10
        assert topology.executed_decoder_block_passes == 16
        assert topology.attention_logit_scale == 0.125
        assert topology.is_pf2_bound()


def test_pf21_returns_catch_33_before_model_or_optimizer_construction() -> None:
    receipt = run_preflight_c1()
    bindings = {binding.component: binding for binding in receipt.mup_bindings}

    assert bindings["attention_logit_scale"].status == "bound"
    assert bindings["attention_logit_scale"].missing_requirement is None
    assert receipt.unbound_mup_components == EXPECTED_UNBOUND_MUP_COMPONENTS
    assert all(
        bindings[name].status == "unbound"
        for name in EXPECTED_UNBOUND_MUP_COMPONENTS
    )
    assert all(
        bindings[name].missing_requirement
        for name in EXPECTED_UNBOUND_MUP_COMPONENTS
    )
    assert receipt.mup_protocol_complete is False
    assert receipt.execution_status == C1_EXECUTION_STATUS
    assert receipt.model_initialized is False
    assert receipt.optimizer_constructed is False
    assert receipt.training_performed is False
    assert receipt.activation_coordinate_passed is None
    assert receipt.a100_hours == 0.0
    assert receipt.passed is False
    assert receipt.catch_number == C1_CATCH_NUMBER == 33
    assert receipt.disposition == "catch_33_return_to_strategy_mup_protocol_unbound"
    with pytest.raises(
        C1CoordinateCatch,
        match=r"CATCH #33: PF-2\.1 stopped before model initialization",
    ):
        receipt.require_passed()


@pytest.mark.parametrize(
    "updates",
    (
        {"passed": True},
        {"mup_protocol_complete": True},
        {"unbound_mup_components": ()},
        {"catch_number": 0},
        {"authority_verified": False},
        {"topology_verified": False},
    ),
)
def test_c1_receipt_rejects_forged_promotion_or_derived_state(
    updates: dict[str, object],
) -> None:
    receipt = run_preflight_c1()

    with pytest.raises(ValueError, match="C1"):
        replace(receipt, **updates)


def test_pf21_harness_rejects_discretionary_width_or_step_overrides() -> None:
    with pytest.raises(ValueError, match="complete PF-2.1 CPU axis"):
        run_preflight_c1(widths=())
    with pytest.raises(ValueError, match="complete PF-2.1 CPU axis"):
        run_preflight_c1(widths=(128, 256))
    with pytest.raises(ValueError, match="literal 10"):
        run_preflight_c1(training_steps=9)
