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
    PF3_AUTHORITY_BYTES,
    PF3_AUTHORITY_FILE,
    PF3_AUTHORITY_SHA256,
    PREFLIGHT_PROGRAM_BYTES,
    PREFLIGHT_PROGRAM_FILE,
    PREFLIGHT_PROGRAM_SHA256,
    PREFLIGHT_RATIFICATION_BYTES,
    PREFLIGHT_RATIFICATION_FILE,
    PREFLIGHT_RATIFICATION_SHA256,
    S81_AUTHORITY_BYTES,
    S81_AUTHORITY_FILE,
    S81_AUTHORITY_SHA256,
    run_preflight_c1,
    verify_c1_authorities,
)
from models.ablation_lm.config import MUP_D_HEAD_BASE
from models.ablation_lm.mup import MuPParameterClass


@pytest.fixture(scope="module")
def receipt():
    return run_preflight_c1()


def test_pf3_chain_and_build_handoff_are_byte_exact() -> None:
    verified = {item.filename: item for item in verify_c1_authorities()}
    expected = {
        PREFLIGHT_PROGRAM_FILE: (PREFLIGHT_PROGRAM_BYTES, PREFLIGHT_PROGRAM_SHA256),
        PREFLIGHT_RATIFICATION_FILE: (
            PREFLIGHT_RATIFICATION_BYTES,
            PREFLIGHT_RATIFICATION_SHA256,
        ),
        BUILD_HANDOFF_AUTHORITY_FILE: (BUILD_HANDOFF_AUTHORITY_BYTES, BUILD_HANDOFF_AUTHORITY_SHA256),
        PF2_AUTHORITY_FILE: (PF2_AUTHORITY_BYTES, PF2_AUTHORITY_SHA256),
        S81_AUTHORITY_FILE: (S81_AUTHORITY_BYTES, S81_AUTHORITY_SHA256),
        PF3_AUTHORITY_FILE: (PF3_AUTHORITY_BYTES, PF3_AUTHORITY_SHA256),
    }
    assert expected[BUILD_HANDOFF_AUTHORITY_FILE][0] == 61_329
    assert expected[PREFLIGHT_PROGRAM_FILE][0] == 15_575
    assert expected[PREFLIGHT_RATIFICATION_FILE][0] == 2_233
    assert expected[PF2_AUTHORITY_FILE][0] == 13_097
    assert expected[S81_AUTHORITY_FILE][0] == 3_403
    assert expected[PF3_AUTHORITY_FILE][0] == 14_632
    assert set(verified) == set(expected)
    for filename, (size, digest) in expected.items():
        row = verified[filename]
        assert row.expected_bytes == row.actual_bytes == size
        assert row.expected_sha256 == row.actual_sha256 == digest
        assert row.verified is True


def test_authority_verification_fails_closed_when_files_are_absent(tmp_path: Path) -> None:
    with pytest.raises(C1AuthorityIntegrityError, match="missing C1 authority"):
        verify_c1_authorities(tmp_path)


def test_pf31_topology_is_bound_on_the_real_width_axis(receipt) -> None:
    assert receipt.cpu_widths == C1_CPU_WIDTHS == (128, 256, 512)
    assert receipt.deferred_gpu_widths == C1_DEFERRED_GPU_WIDTHS == (1_024,)
    assert receipt.batch_size == C1_BATCH_SIZE == 2
    assert receipt.sequence_length == C1_SEQUENCE_LENGTH == 64
    assert receipt.training_steps == C1_TRAINING_STEPS == 10
    assert receipt.width_drift_limit == C1_WIDTH_DRIFT_LIMIT == 2.0
    assert receipt.authority_verified is True
    assert receipt.topology_verified is True
    assert MUP_D_HEAD_BASE == C1_HEAD_DIM == 64
    for row in receipt.width_classifications:
        topology = row.topology
        assert topology.head_dim == 64
        assert topology.q_heads == row.width // 64
        assert topology.kv_heads == row.width // 128
        assert topology.q_heads == 2 * topology.kv_heads
        assert topology.d_ff == 11 * row.width // 4
        assert topology.scratch_lanes == 2
        assert topology.scratch_width_per_lane == row.width // 4
        assert topology.total_scratch_width == row.width // 2
        assert topology.unique_decoder_blocks == 10
        assert topology.executed_decoder_block_passes == 16
        assert topology.attention_logit_scale == 0.125
        assert topology.is_bound()


def test_pf31_echoes_every_tensor_and_only_router_is_unclassified(receipt) -> None:
    assert receipt.unclassified_tensor_shapes == (
        ("front_hadamard.router.weight", (8, 128)),
        ("front_hadamard.router.weight", (8, 256)),
        ("front_hadamard.router.weight", (8, 512)),
    )
    for row in receipt.width_classifications:
        assert row.unique_trainable_tensors == 148
        assert len(row.classified_tensors) == 147
        assert len(row.unclassified_tensors) == 1
        assert row.unclassified_tensors[0].canonical_name == "front_hadamard.router.weight"
        assert row.unclassified_tensors[0].shape == (8, row.width)
        assert "unclassifiable trainable tensor" in row.unclassified_tensors[0].reason
        assert len({item.canonical_name for item in row.classified_tensors}) == 147
        assert len(row.classified_map_sha256) == 64


def test_ltm_and_engram_classes_prove_actual_scaling_dimensions(receipt) -> None:
    for row in receipt.width_classifications:
        items = {item.canonical_name: item for item in row.classified_tensors}
        assert items["long_term_memory.query.weight"].shape == (row.width // 4, row.width)
        assert items["long_term_memory.output.weight"].shape == (row.width, row.width // 4)
        assert items["long_term_memory.query.weight"].parameter_class is MuPParameterClass.HIDDEN
        assert items["long_term_memory.output.weight"].parameter_class is MuPParameterClass.HIDDEN
        assert items["engram.query_proj.weight"].shape == (64, row.width)
        assert items["engram.query_proj.weight"].parameter_class is MuPParameterClass.HIDDEN
        value = items["engram.value_proj.weight"]
        assert value.shape == (row.width, 64)
        assert value.parameter_class is MuPParameterClass.INPUT
        assert value.learning_rate == 3.0e-4
        assert value.weight_decay == 0.0
        assert "engram.key_proj.weight" not in items
        for name in ("engram.query_norm.weight", "engram.key_norm.weight"):
            gain = items[name]
            assert gain.shape == (64,)
            assert gain.parameter_class is MuPParameterClass.VECTOR
            assert gain.initialization_rule == "module_design_state_preserved"
            assert gain.learning_rate == 3.0e-4
            assert gain.weight_decay == 0.0
        gains = items["front_hadamard.expert_gains"]
        assert gains.shape == (8, row.width)
        assert gains.parameter_class is MuPParameterClass.VECTOR
        assert gains.initialization_rule == "module_design_state_preserved"


def test_pf31_catch_35_prevents_initialization_optimizer_forward_and_rms(receipt) -> None:
    assert receipt.tensor_class_map_complete is False
    assert receipt.mup_protocol_complete is False
    assert receipt.execution_status == C1_EXECUTION_STATUS
    assert receipt.model_constructed_for_inventory is True
    assert receipt.pf3_initialization_applied is False
    assert receipt.optimizer_constructed is False
    assert receipt.forward_executed is False
    assert receipt.training_performed is False
    assert receipt.activation_rms_measured is False
    assert receipt.activation_coordinate_passed is None
    assert receipt.passed is False
    assert receipt.catch_number == C1_CATCH_NUMBER == 35
    assert receipt.disposition == "catch_35_hadamard_router_mup_class_unbound"
    assert receipt.a100_hours == 0.0
    with pytest.raises(C1CoordinateCatch, match=r"CATCH #35: front_hadamard\.router\.weight"):
        receipt.require_passed()


@pytest.mark.parametrize(
    "updates",
    (
        {"passed": True},
        {"tensor_class_map_complete": True},
        {"mup_protocol_complete": True},
        {"unclassified_tensor_shapes": ()},
        {"pf3_initialization_applied": True},
        {"optimizer_constructed": True},
        {"training_performed": True},
        {"activation_rms_measured": True},
        {"catch_number": 0},
        {"authority_verified": False},
        {"topology_verified": False},
        {"program_sha256": "0" * 64},
        {"ratification_sha256": "0" * 64},
        {"build_handoff_authority": "forged"},
        {"build_handoff_authority_drive_id": "forged"},
        {"deferred_gpu_widths": ()},
        {"batch_size": 1},
        {"sequence_length": 63},
        {"training_steps": 11},
        {"width_drift_limit": 3.0},
        {"provisional_base_shape": "forged"},
        {"provisional_numerics": "forged"},
        {"decay_implementation": "forged"},
        {"disposition": "forged"},
    ),
)
def test_c1_receipt_rejects_forged_promotion_or_derived_state(receipt, updates) -> None:
    with pytest.raises(ValueError, match="C1"):
        replace(receipt, **updates)


def test_pf31_harness_rejects_discretionary_width_or_step_overrides() -> None:
    with pytest.raises(ValueError, match="complete PF-3.1 CPU axis"):
        run_preflight_c1(widths=(128, 256))
    with pytest.raises(ValueError, match="literal 10"):
        run_preflight_c1(training_steps=9)
