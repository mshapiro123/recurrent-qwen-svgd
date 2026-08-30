from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path

import pytest
import torch

from models.ablation_lm import AblationLM, AblationLMConfig
from training.weft1_gtok_contract import canonical_sha256
import training.weft1_gtok_determinism_v2 as determinism
from training.weft1_gtok_training_v2 import build_flat_a1_adamw_v2
from training.weft1_gtok_training_v2 import TrainingPlanV2


def _hash(label: str) -> str:
    return canonical_sha256({"label": label})


@dataclass(frozen=True)
class _TinyPlan:
    optimizer_steps: int = 2
    compute_token_slots: int = 48  # one 4x8 full batch, then one 2x8 tail
    receipt_sha256: str = _hash("tiny-plan")


def _tiny_binding() -> determinism.DeterminismReplayPlanBindingV2:
    return determinism.DeterminismReplayPlanBindingV2(
        vocab_size=32,
        terminal_rows=2,
        representative_training_seed=202,
        representative_initialization_seed=101,
        representative_plan_sha256=_TinyPlan().receipt_sha256,
        equivalent_training_seeds=(202,),
        equivalent_plan_receipt_sha256s=(_TinyPlan().receipt_sha256,),
    )


def _tiny_model() -> AblationLM:
    return AblationLM(
        AblationLMConfig(
            vocab_size=32,
            d_model=8,
            n_heads=2,
            n_kv_heads=1,
            d_ff=16,
            n_prelude_layers=1,
            n_core_blocks=1,
            n_coda_layers=0,
            max_sequence_length=8,
            recurrent_steps=1,
            max_recurrent_steps=2,
            use_recurrence=False,
            use_front_hadamard_experts=False,
            use_reentry_bridge=False,
            use_scratch=False,
            use_lane_carrier=False,
            use_engram=False,
            use_long_term_memory=False,
            initialization_seed=101,
            run_seed=202,
        )
    )


def _backend_state() -> dict[str, object]:
    return {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "debug": torch.get_deterministic_debug_mode(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
    }


def _restore_backend_state(state: dict[str, object]) -> None:
    torch.use_deterministic_algorithms(
        bool(state["deterministic"]),
        warn_only=bool(state["warn_only"]),
    )
    torch.set_deterministic_debug_mode(int(state["debug"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(state["matmul_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(state["cudnn_tf32"])
    torch.backends.cudnn.benchmark = bool(state["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(state["cudnn_deterministic"])
    torch.backends.cudnn.enabled = bool(state["cudnn_enabled"])
    torch.backends.cuda.enable_flash_sdp(bool(state["flash"]))
    torch.backends.cuda.enable_mem_efficient_sdp(bool(state["efficient"]))
    torch.backends.cuda.enable_math_sdp(bool(state["math"]))
    torch.backends.cuda.enable_cudnn_sdp(bool(state["cudnn_sdp"]))


def _cpu_attestation() -> determinism.CudaDeterminismAttestationV2:
    return determinism.CudaDeterminismAttestationV2(
        policy=determinism.CUDA_DETERMINISM_POLICY_V2,
        torch_version=str(torch.__version__),
        device_type="cpu",
        authority_status="NON_AUTHORITATIVE_CPU_TEST",
    )


def _replica(index: int) -> determinism.DeterminismReplayReplicaV2:
    return determinism._execute_replay_replica_v2(
        replica_index=index,
        model_factory=_tiny_model,
        optimizer_factory=build_flat_a1_adamw_v2,
        plan=_TinyPlan(),
        vocab_size=32,
        initialization_seed=101,
        run_seed=202,
        device=torch.device("cpu"),
        microbatch_sequences=2,
        policy_receipt_sha256=(
            determinism.CUDA_DETERMINISM_POLICY_V2.receipt_sha256
        ),
        replay_plan_binding_sha256=_tiny_binding().receipt_sha256,
        gpu_uuid_provenance=None,
        global_batch_sequences=4,
        sequence_length=8,
        evaluation_tokens=4,
        require_fused_flash=False,
    )


def test_cuda_policy_binds_every_flag_and_fails_closed_on_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = _backend_state()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(
        determinism.GTokDeterminismV2Stop,
        match="CUDA_POLICY_ENVIRONMENT_MISMATCH",
    ):
        determinism.apply_and_attest_cuda_determinism_policy_v2(
            device=torch.device("cpu"),
            allow_nonproduction_cpu=True,
        )
    monkeypatch.setenv(
        "CUBLAS_WORKSPACE_CONFIG",
        determinism.CUBLAS_WORKSPACE_CONFIG_V2,
    )
    try:
        attestation = determinism.apply_and_attest_cuda_determinism_policy_v2(
            device=torch.device("cpu"),
            allow_nonproduction_cpu=True,
        )
        assert attestation.policy == determinism.CUDA_DETERMINISM_POLICY_V2
        assert attestation.policy.receipt_sha256 == (
            determinism.CUDA_DETERMINISM_POLICY_SHA256_V2
        )
        assert attestation.authority_status == "NON_AUTHORITATIVE_CPU_TEST"
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_deterministic_debug_mode() == 2
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.enabled is True
        assert torch.backends.cuda.flash_sdp_enabled() is True
        assert torch.backends.cuda.mem_efficient_sdp_enabled() is False
        assert torch.backends.cuda.math_sdp_enabled() is False
        assert torch.backends.cuda.cudnn_sdp_enabled() is False
    finally:
        _restore_backend_state(prior)


def test_two_fresh_optimizer_eval_replays_match_exactly_across_both_shapes() -> None:
    first = _replica(0)
    second = _replica(1)
    dry = determinism.validate_determinism_replay_pair_v2(first, second)

    assert tuple(row.label for row in first.fingerprint.shapes) == (
        "full",
        "terminal_partial",
    )
    assert tuple(row.batch_rows for row in first.fingerprint.shapes) == (4, 2)
    assert first.fingerprint.model_state_sha256_by_shape == (
        second.fingerprint.model_state_sha256_by_shape
    )
    assert first.fingerprint.optimizer_state_sha256_by_shape == (
        second.fingerprint.optimizer_state_sha256_by_shape
    )
    assert first.fingerprint.evaluation_output_sha256_by_shape == (
        second.fingerprint.evaluation_output_sha256_by_shape
    )
    assert first.fingerprint.receipt_sha256 == second.fingerprint.receipt_sha256
    assert dry.fingerprint_receipt_sha256 == first.fingerprint.receipt_sha256
    assert dry.measured_device_microseconds == (
        first.charged_device_microseconds + second.charged_device_microseconds
    )


def test_pair_projection_prices_outer_lifecycle_not_inner_device_timer() -> None:
    first = replace(_replica(0), charged_device_microseconds=10)
    projection = determinism.project_second_determinism_replay_replica_v2(
        first,
        replay_plan_binding=_tiny_binding(),
        first_lifecycle_a100_microseconds=1_000,
        prior_campaign_a100_microseconds=7,
    )

    assert projection.first_replica_a100_microseconds == 1_000
    assert projection.projected_second_replica_a100_microseconds == 1_000
    assert projection.second_replica_watchdog_a100_microseconds == 2_000
    assert projection.projected_campaign_a100_microseconds == 2_007
    with pytest.raises(ValueError, match="include its inner device timer"):
        determinism.project_second_determinism_replay_replica_v2(
            first,
            replay_plan_binding=_tiny_binding(),
            first_lifecycle_a100_microseconds=9,
            prior_campaign_a100_microseconds=0,
        )


def test_replay_meter_starts_before_shape_and_batch_preamble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((0, 20_000_000))
    monkeypatch.setattr(determinism.time, "perf_counter_ns", lambda: next(clock))
    model_built = False

    def model_factory() -> torch.nn.Module:
        nonlocal model_built
        model_built = True
        return _tiny_model()

    with pytest.raises(
        determinism.GTokDeterminismV2Stop,
        match="DETERMINISM_REPLAY_WATCHDOG",
    ):
        determinism._execute_replay_replica_v2(
            replica_index=0,
            model_factory=model_factory,
            optimizer_factory=build_flat_a1_adamw_v2,
            plan=_TinyPlan(),
            vocab_size=32,
            initialization_seed=101,
            run_seed=202,
            device=torch.device("cpu"),
            microbatch_sequences=2,
            policy_receipt_sha256=(
                determinism.CUDA_DETERMINISM_POLICY_V2.receipt_sha256
            ),
            replay_plan_binding_sha256=_tiny_binding().receipt_sha256,
            gpu_uuid_provenance=None,
            global_batch_sequences=4,
            sequence_length=8,
            evaluation_tokens=4,
            require_fused_flash=False,
            watchdog_limit_device_microseconds=10_000,
        )
    assert not model_built


def test_canonical_replay_traversal_deduplicates_only_vocab_and_tail_rows() -> None:
    full_slots = 256 * 2_048

    def plan(label: str, terminal_rows: int) -> TrainingPlanV2:
        return TrainingPlanV2(
            optimizer_steps=2,
            compute_token_slots=full_slots + terminal_rows * 2_048,
            valid_prediction_count=1,
            realized_raw_bytes=1,
            document_count=1,
            packed_stream_sha256=_hash(label),
        )

    plans = {
        (16_384, 1): plan("16-a", 10),
        (16_384, 2): plan("16-b", 10),
        (24_576, 1): plan("24-a", 12),
        (24_576, 2): plan("24-b", 13),
    }
    rows = determinism.canonical_determinism_replay_plan_bindings_v2(
        plans,
        vocabularies=(16_384, 24_576),
        training_seeds=(1, 2),
        initialization_seeds=(11, 22),
    )
    assert tuple((row.vocab_size, row.terminal_rows) for row in rows) == (
        (16_384, 10),
        (24_576, 12),
        (24_576, 13),
    )
    assert rows[0].representative_training_seed == 1
    assert rows[0].representative_initialization_seed == 11
    assert rows[0].equivalent_training_seeds == (1, 2)
    assert rows[0].equivalent_plan_receipt_sha256s == (
        plans[(16_384, 1)].receipt_sha256,
        plans[(16_384, 2)].receipt_sha256,
    )


def test_any_exact_state_output_or_receipt_difference_is_a_governed_stop() -> None:
    first = _replica(0)
    second = _replica(1)
    rows = list(second.fingerprint.evaluation_output_sha256_by_shape)
    rows[1] = ("terminal_partial", _hash("tampered-output"))
    bad_fingerprint = replace(
        second.fingerprint,
        evaluation_output_sha256_by_shape=tuple(rows),
    )
    bad_second = replace(second, fingerprint=bad_fingerprint)
    with pytest.raises(
        determinism.GTokDeterminismV2Stop,
        match="DETERMINISM_REPLAY_MISMATCH",
    ):
        determinism.validate_determinism_replay_pair_v2(first, bad_second)


def test_flash_backend_trace_rejects_failure_math_or_backend_substitution() -> None:
    flash = tuple(
        sorted(
            (
                "aten::_scaled_dot_product_flash_attention",
                "aten::_scaled_dot_product_flash_attention_backward",
            )
        )
    )
    determinism._validate_flash_backend_trace_v2(flash)
    with pytest.raises(
        determinism.GTokDeterminismV2Stop,
        match="FUSED_SDPA_BACKEND_FAILED",
    ):
        determinism._validate_flash_backend_trace_v2(
            ("aten::scaled_dot_product_attention",)
        )
    with pytest.raises(
        determinism.GTokDeterminismV2Stop,
        match="FUSED_SDPA_BACKEND_DRIFT",
    ):
        determinism._validate_flash_backend_trace_v2(
            tuple(sorted((flash[0], "aten::_scaled_dot_product_attention_math")))
        )
    with pytest.raises(
        determinism.GTokDeterminismV2Stop,
        match="FUSED_SDPA_BACKEND_DRIFT",
    ):
        determinism._validate_flash_backend_trace_v2(
            tuple(sorted((flash[0], "aten::_scaled_dot_product_efficient_attention")))
        )


def test_injected_or_cpu_replays_have_no_physical_mint_surface() -> None:
    signature = __import__("inspect").signature(
        determinism.validate_determinism_replay_pair_v2
    )
    assert "authoritative" not in signature.parameters
    first = _replica(0)
    second = _replica(1)
    dry = determinism.validate_determinism_replay_pair_v2(first, second)
    assert isinstance(dry, determinism.DryRunDeterminismReplayV2)
    assert not hasattr(dry, "charged_a100_microseconds")
    assert _cpu_attestation().authority_status == "NON_AUTHORITATIVE_CPU_TEST"


def test_terminal_full_shape_or_invalid_tail_stops_before_replay() -> None:
    full_tail = _TinyPlan(compute_token_slots=64)
    with pytest.raises(
        determinism.GTokDeterminismV2Stop,
        match="TERMINAL_PARTIAL_SHAPE_UNAVAILABLE",
    ):
        determinism._shapes_from_plan_v2(
            full_tail,
            global_batch_sequences=4,
            sequence_length=8,
        )
    invalid_tail = _TinyPlan(compute_token_slots=49)
    with pytest.raises(
        determinism.GTokDeterminismV2Stop,
        match="TERMINAL_PARTIAL_SHAPE_INVALID",
    ):
        determinism._shapes_from_plan_v2(
            invalid_tail,
            global_batch_sequences=4,
            sequence_length=8,
        )


def test_physical_receipt_requires_both_attempt_charges_in_its_meter(
    tmp_path: Path,
) -> None:
    first = _replica(0)
    fingerprint = replace(
        first.fingerprint,
        fused_backend_operator_names=tuple(
            sorted(
                (
                    "aten::_scaled_dot_product_flash_attention",
                    "aten::_scaled_dot_product_flash_attention_backward",
                )
            )
        ),
    )
    binding = _tiny_binding()
    projection = determinism.DeterminismReplayPairProjectionV2(
        replay_plan_binding_sha256=binding.receipt_sha256,
        first_replica_a100_microseconds=10,
        projected_second_replica_a100_microseconds=10,
        second_replica_watchdog_a100_microseconds=20,
        prior_campaign_a100_microseconds=0,
        projected_campaign_a100_microseconds=20,
    )
    with pytest.raises(ValueError, match="include both fresh attempts"):
        determinism.PrecalibrationDeterminismReplayReceiptV2(
            policy_attestation_receipt_sha256=_hash("policy-attestation"),
            training_runtime_receipt_sha256=_hash("runtime"),
            code_closure_receipt_sha256=_hash("code"),
            replay_plan_binding=binding,
            pair_projection=projection,
            fingerprint=fingerprint,
            replay_fingerprint_sha256s=(
                fingerprint.receipt_sha256,
                fingerprint.receipt_sha256,
            ),
            attempts=(
                (0, 10, "GPU-aaaaaaaa"),
                (1, 20, "GPU-aaaaaaaa"),
            ),
            inner_device_microseconds=(1, 1),
            charged_a100_microseconds=29,
        )
    receipt = determinism.PrecalibrationDeterminismReplayReceiptV2(
        policy_attestation_receipt_sha256=_hash("policy-attestation"),
        training_runtime_receipt_sha256=_hash("runtime"),
        code_closure_receipt_sha256=_hash("code"),
        replay_plan_binding=binding,
        pair_projection=projection,
        fingerprint=fingerprint,
        replay_fingerprint_sha256s=(
            fingerprint.receipt_sha256,
            fingerprint.receipt_sha256,
        ),
        attempts=(
            (0, 10, "GPU-aaaaaaaa"),
            (1, 20, "GPU-aaaaaaaa"),
        ),
        inner_device_microseconds=(1, 1),
        charged_a100_microseconds=30,
    )
    assert receipt.charged_a100_microseconds == 30
    assert len(receipt.receipt_sha256) == 64
    path = tmp_path / "precalibration-replay.json"
    assert determinism.write_precalibration_determinism_replay_receipt_v2(
        path, receipt
    ) == receipt.receipt_sha256
    assert determinism.load_precalibration_determinism_replay_receipt_v2(path) == receipt
    assert determinism.write_precalibration_determinism_replay_receipt_v2(
        path, receipt
    ) == receipt.receipt_sha256
