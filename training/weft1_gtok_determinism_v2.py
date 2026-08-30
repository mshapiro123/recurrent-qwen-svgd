"""Fail-closed deterministic CUDA policy and physical replay gate for G-TOK.

The policy is deliberately narrower than ``torch.use_deterministic_algorithms``:
it binds the process-level cuBLAS workspace contract, disables TF32 and cuDNN
autotuning, and admits only Flash SDPA.  The replay gate then executes two fresh
model/optimizer replicas through one full logical optimizer batch and the actual
terminal-partial logical batch shape.  Exact fingerprints are compared after
both steps; a backend fallback or any state/output mismatch is a governed STOP.

This module does not mutate the campaign ledger.  The campaign integration must
run it before calibration, persist its receipt, and add both physical replay
charges to the cumulative 12-hour meter.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import torch

from training.weft1_gtok_contract import canonical_json_bytes
from training.weft1_gtok_training_v2 import (
    PACKING_BINDING_V2,
    PackedBatchV2,
    TrainingPlanV2,
    _autocast,
    _execute_optimizer_step_v2,
    build_flat_a1_adamw_v2,
    build_gtok_proxy_model_v2,
    require_production_a100_v2,
)
from training.weft1_gtok_v2_contract import (
    GTOK_TRIPWIRE_A100_MICROSECONDS,
    GTokV2Stop,
    gtok_v2_bound_sha256,
)
from training.weft1_strict_io import assert_no_symlink_ancestors, load_canonical_json_snapshot


CUBLAS_WORKSPACE_CONFIG_V2 = ":4096:8"
FLASH_SDPA_BACKEND_V2 = "FLASH_ATTENTION_ONLY"
DETERMINISM_POLICY_SCHEMA_V2 = "weft1_gtok_v2_cuda_determinism_policy"
DETERMINISM_REPLAY_SCHEMA_V2 = "weft1_gtok_v2_precalibration_replay_gate"
DETERMINISM_REPLAY_EVALUATION_TOKENS_V2 = 32
_HEX = frozenset("0123456789abcdef")


class GTokDeterminismV2Error(RuntimeError):
    """The deterministic-policy or replay evidence is malformed."""


class GTokDeterminismV2Stop(GTokV2Stop):
    """A registered deterministic CUDA/replay stop condition fired."""

    def __init__(self, reason: str, message: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("determinism STOP reason must be nonempty")
        self.reason = reason
        super().__init__(f"{reason}: {message}")


def _require_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class CudaDeterminismPolicyV2:
    """The literal process and PyTorch backend state required by G-TOK."""

    cublas_workspace_config: str = CUBLAS_WORKSPACE_CONFIG_V2
    deterministic_algorithms: bool = True
    deterministic_warn_only: bool = False
    deterministic_debug_mode: int = 2
    cuda_matmul_allow_tf32: bool = False
    cudnn_allow_tf32: bool = False
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    cudnn_enabled: bool = True
    sdpa_flash_enabled: bool = True
    sdpa_mem_efficient_enabled: bool = False
    sdpa_math_enabled: bool = False
    sdpa_cudnn_enabled: bool = False
    fused_sdpa_backend: str = FLASH_SDPA_BACKEND_V2
    schema: str = DETERMINISM_POLICY_SCHEMA_V2

    def __post_init__(self) -> None:
        expected = {
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG_V2,
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "deterministic_debug_mode": 2,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_enabled": True,
            "sdpa_flash_enabled": True,
            "sdpa_mem_efficient_enabled": False,
            "sdpa_math_enabled": False,
            "sdpa_cudnn_enabled": False,
            "fused_sdpa_backend": FLASH_SDPA_BACKEND_V2,
            "schema": DETERMINISM_POLICY_SCHEMA_V2,
        }
        if asdict(self) != expected:
            raise ValueError("deterministic CUDA policy differs from its literal binding")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256(DETERMINISM_POLICY_SCHEMA_V2, self)


CUDA_DETERMINISM_POLICY_V2 = CudaDeterminismPolicyV2()
CUDA_DETERMINISM_POLICY_SHA256_V2 = CUDA_DETERMINISM_POLICY_V2.receipt_sha256


@dataclass(frozen=True)
class CudaDeterminismAttestationV2:
    policy: CudaDeterminismPolicyV2
    torch_version: str
    device_type: str
    authority_status: str

    def __post_init__(self) -> None:
        if self.policy != CUDA_DETERMINISM_POLICY_V2:
            raise ValueError("CUDA attestation policy differs from the fixed binding")
        if not isinstance(self.torch_version, str) or not self.torch_version:
            raise ValueError("CUDA attestation requires a PyTorch version")
        if self.device_type not in ("cuda", "cpu"):
            raise ValueError("CUDA attestation device type is invalid")
        expected = (
            "AUTHORITATIVE_A100_FLASH_ONLY"
            if self.device_type == "cuda"
            else "NON_AUTHORITATIVE_CPU_TEST"
        )
        if self.authority_status != expected:
            raise ValueError("CUDA attestation authority status drifted")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_cuda_determinism_attestation", self)


def _backend_api(name: str) -> Callable[..., Any]:
    value = getattr(torch.backends.cuda, name, None)
    if not callable(value):
        raise GTokDeterminismV2Stop(
            "CUDA_POLICY_API_UNAVAILABLE",
            f"the bound PyTorch runtime lacks torch.backends.cuda.{name}",
        )
    return value


def _observed_cuda_policy_v2() -> CudaDeterminismPolicyV2:
    """Read every mutable flag back instead of trusting setter success."""

    try:
        return CudaDeterminismPolicyV2(
            cublas_workspace_config=str(os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")),
            deterministic_algorithms=bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            deterministic_warn_only=bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            deterministic_debug_mode=int(torch.get_deterministic_debug_mode()),
            cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
            cudnn_enabled=bool(torch.backends.cudnn.enabled),
            sdpa_flash_enabled=bool(_backend_api("flash_sdp_enabled")()),
            sdpa_mem_efficient_enabled=bool(
                _backend_api("mem_efficient_sdp_enabled")()
            ),
            sdpa_math_enabled=bool(_backend_api("math_sdp_enabled")()),
            sdpa_cudnn_enabled=bool(_backend_api("cudnn_sdp_enabled")()),
            fused_sdpa_backend=FLASH_SDPA_BACKEND_V2,
        )
    except GTokDeterminismV2Stop:
        raise
    except Exception as error:
        raise GTokDeterminismV2Stop(
            "CUDA_POLICY_ATTESTATION_FAILED",
            "the fixed deterministic backend state could not be read",
        ) from error


def apply_and_attest_cuda_determinism_policy_v2(
    *,
    device: torch.device,
    allow_nonproduction_cpu: bool = False,
) -> CudaDeterminismAttestationV2:
    """Apply the fixed flags before model construction and attest them exactly.

    ``CUBLAS_WORKSPACE_CONFIG`` is intentionally verified, not set here: setting
    it after CUDA initialization can create a false determinism claim.  The
    authoritative launcher must place it in the child environment.
    """

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG_V2:
        raise GTokDeterminismV2Stop(
            "CUDA_POLICY_ENVIRONMENT_MISMATCH",
            "CUBLAS_WORKSPACE_CONFIG was not fixed by the parent launcher",
        )
    if device.type != "cuda" and not allow_nonproduction_cpu:
        raise GTokDeterminismV2Stop(
            "CUDA_POLICY_A100_REQUIRED",
            "authoritative replay requires an NVIDIA A100",
        )
    if device.type == "cuda":
        try:
            require_production_a100_v2(device)
        except Exception as error:
            raise GTokDeterminismV2Stop(
                "CUDA_POLICY_A100_REQUIRED",
                "authoritative replay requires the bound A100 device",
            ) from error
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_deterministic_debug_mode("error")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        _backend_api("enable_flash_sdp")(True)
        _backend_api("enable_mem_efficient_sdp")(False)
        _backend_api("enable_math_sdp")(False)
        _backend_api("enable_cudnn_sdp")(False)
    except GTokDeterminismV2Stop:
        raise
    except Exception as error:
        raise GTokDeterminismV2Stop(
            "CUDA_POLICY_APPLICATION_FAILED",
            "the fixed deterministic backend state could not be applied",
        ) from error
    observed = _observed_cuda_policy_v2()
    if observed != CUDA_DETERMINISM_POLICY_V2:
        raise GTokDeterminismV2Stop(
            "CUDA_POLICY_ATTESTATION_MISMATCH",
            "observed PyTorch backend flags differ from the fixed policy",
        )
    return CudaDeterminismAttestationV2(
        policy=observed,
        torch_version=str(torch.__version__),
        device_type=device.type,
        authority_status=(
            "AUTHORITATIVE_A100_FLASH_ONLY"
            if device.type == "cuda"
            else "NON_AUTHORITATIVE_CPU_TEST"
        ),
    )


@dataclass(frozen=True)
class ReplayBatchShapeV2:
    label: str
    batch_rows: int
    sequence_length: int
    valid_prediction_count: int
    batch_sha256: str

    def __post_init__(self) -> None:
        if self.label not in ("full", "terminal_partial"):
            raise ValueError("replay batch shape label is invalid")
        for name in ("batch_rows", "sequence_length", "valid_prediction_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        _require_sha256(self.batch_sha256, "batch_sha256")


@dataclass(frozen=True)
class DeterminismReplayPlanBindingV2:
    """One canonical distinct ``(vocabulary, terminal rows)`` replay cell."""

    vocab_size: int
    terminal_rows: int
    representative_training_seed: int
    representative_initialization_seed: int
    representative_plan_sha256: str
    equivalent_training_seeds: tuple[int, ...]
    equivalent_plan_receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("vocab_size", "terminal_rows"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        for name in (
            "representative_training_seed",
            "representative_initialization_seed",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        _require_sha256(self.representative_plan_sha256, "representative_plan_sha256")
        if (
            not self.equivalent_training_seeds
            or len(set(self.equivalent_training_seeds))
            != len(self.equivalent_training_seeds)
            or self.equivalent_training_seeds[0]
            != self.representative_training_seed
        ):
            raise ValueError("replay binding seeds must be unique and representative-first")
        if len(self.equivalent_plan_receipt_sha256s) != len(
            self.equivalent_training_seeds
        ):
            raise ValueError("replay binding must name one plan receipt per equivalent seed")
        for value in self.equivalent_plan_receipt_sha256s:
            _require_sha256(value, "equivalent_plan_receipt_sha256s")
        if self.equivalent_plan_receipt_sha256s[0] != self.representative_plan_sha256:
            raise ValueError("representative replay plan must be the first bound plan")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_determinism_replay_plan_binding", self)


def canonical_determinism_replay_plan_bindings_v2(
    plans: Mapping[tuple[int, int], TrainingPlanV2],
    *,
    vocabularies: tuple[int, ...],
    training_seeds: tuple[int, ...],
    initialization_seeds: tuple[int, ...],
) -> tuple[DeterminismReplayPlanBindingV2, ...]:
    """Deduplicate only identical ``(V, terminal_rows)`` keys in canonical order."""

    if (
        not vocabularies
        or not training_seeds
        or len(training_seeds) != len(initialization_seeds)
        or len(set(vocabularies)) != len(vocabularies)
        or len(set(training_seeds)) != len(training_seeds)
    ):
        raise ValueError("determinism replay traversal inputs are incomplete or duplicated")
    expected = {
        (vocab_size, training_seed)
        for vocab_size in vocabularies
        for training_seed in training_seeds
    }
    if set(plans) != expected or any(
        not isinstance(plan, TrainingPlanV2) for plan in plans.values()
    ):
        raise ValueError("determinism replay traversal requires the complete typed plan matrix")
    sequence_length = int(PACKING_BINDING_V2["sequence_length"])
    global_rows = int(PACKING_BINDING_V2["batch_sequences"])
    groups: dict[tuple[int, int], list[tuple[int, int, str]]] = {}
    initialization_by_seed = dict(zip(training_seeds, initialization_seeds, strict=True))
    for vocab_size in vocabularies:
        for training_seed in training_seeds:
            plan = plans[(vocab_size, training_seed)]
            _, terminal_rows = _shapes_from_plan_v2(
                plan,
                global_batch_sequences=global_rows,
                sequence_length=sequence_length,
            )
            groups.setdefault((vocab_size, terminal_rows), []).append(
                (
                    training_seed,
                    initialization_by_seed[training_seed],
                    plan.receipt_sha256,
                )
            )
    rows: list[DeterminismReplayPlanBindingV2] = []
    for vocab_size, terminal_rows in sorted(groups):
        equivalents = groups[(vocab_size, terminal_rows)]
        representative_seed, representative_initialization, representative_plan = (
            equivalents[0]
        )
        rows.append(
            DeterminismReplayPlanBindingV2(
                vocab_size=vocab_size,
                terminal_rows=terminal_rows,
                representative_training_seed=representative_seed,
                representative_initialization_seed=representative_initialization,
                representative_plan_sha256=representative_plan,
                equivalent_training_seeds=tuple(row[0] for row in equivalents),
                equivalent_plan_receipt_sha256s=tuple(row[2] for row in equivalents),
            )
        )
    if tuple((row.vocab_size, row.terminal_rows) for row in rows) != tuple(
        sorted(set((row.vocab_size, row.terminal_rows) for row in rows))
    ):
        raise AssertionError("determinism replay traversal is not canonical")
    return tuple(rows)


@dataclass(frozen=True)
class DeterminismReplayFingerprintV2:
    policy_receipt_sha256: str
    replay_plan_binding_sha256: str
    training_plan_sha256: str
    vocab_size: int
    initialization_seed: int
    run_seed: int
    microbatch_sequences: int
    shapes: tuple[ReplayBatchShapeV2, ReplayBatchShapeV2]
    model_state_sha256_by_shape: tuple[tuple[str, str], tuple[str, str]]
    optimizer_state_sha256_by_shape: tuple[tuple[str, str], tuple[str, str]]
    evaluation_output_sha256_by_shape: tuple[tuple[str, str], tuple[str, str]]
    fused_backend_operator_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.policy_receipt_sha256, "policy_receipt_sha256")
        _require_sha256(
            self.replay_plan_binding_sha256, "replay_plan_binding_sha256"
        )
        _require_sha256(self.training_plan_sha256, "training_plan_sha256")
        for name in ("vocab_size", "microbatch_sequences"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        for name in ("initialization_seed", "run_seed"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if tuple(row.label for row in self.shapes) != ("full", "terminal_partial"):
            raise ValueError("replay must cover full then terminal-partial shapes")
        for field_name in (
            "model_state_sha256_by_shape",
            "optimizer_state_sha256_by_shape",
            "evaluation_output_sha256_by_shape",
        ):
            rows = getattr(self, field_name)
            if tuple(label for label, _ in rows) != ("full", "terminal_partial"):
                raise ValueError(f"{field_name} shape order drifted")
            for _, value in rows:
                _require_sha256(value, field_name)
        if tuple(sorted(set(self.fused_backend_operator_names))) != (
            self.fused_backend_operator_names
        ):
            raise ValueError("fused backend operator names must be sorted and unique")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_determinism_replay_fingerprint", self)


@dataclass(frozen=True)
class DeterminismReplayReplicaV2:
    replica_index: int
    fingerprint: DeterminismReplayFingerprintV2
    charged_device_microseconds: int
    gpu_uuid_provenance: str | None

    def __post_init__(self) -> None:
        if self.replica_index not in (0, 1):
            raise ValueError("replay replica index must be zero or one")
        if not isinstance(self.fingerprint, DeterminismReplayFingerprintV2):
            raise TypeError("replay replica requires a deterministic fingerprint")
        if type(self.charged_device_microseconds) is not int or self.charged_device_microseconds < 1:
            raise ValueError("replay charge must be a positive exact integer")
        if self.gpu_uuid_provenance is not None and (
            not isinstance(self.gpu_uuid_provenance, str)
            or not self.gpu_uuid_provenance.startswith("GPU-")
            or len(self.gpu_uuid_provenance) <= 4
        ):
            raise ValueError("replay GPU provenance must be one NVIDIA GPU UUID")


@dataclass(frozen=True)
class DeterminismReplayPairProjectionV2:
    """First replica is the burst; project and watchdog the second before launch."""

    replay_plan_binding_sha256: str
    first_replica_a100_microseconds: int
    projected_second_replica_a100_microseconds: int
    second_replica_watchdog_a100_microseconds: int
    prior_campaign_a100_microseconds: int
    projected_campaign_a100_microseconds: int
    tripwire_a100_microseconds: int = GTOK_TRIPWIRE_A100_MICROSECONDS

    def __post_init__(self) -> None:
        _require_sha256(
            self.replay_plan_binding_sha256, "replay_plan_binding_sha256"
        )
        for name in (
            "first_replica_a100_microseconds",
            "projected_second_replica_a100_microseconds",
            "second_replica_watchdog_a100_microseconds",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if (
            type(self.prior_campaign_a100_microseconds) is not int
            or self.prior_campaign_a100_microseconds < 0
        ):
            raise ValueError("prior campaign meter must be a non-negative exact integer")
        if (
            self.projected_second_replica_a100_microseconds
            != self.first_replica_a100_microseconds
        ):
            raise ValueError("the first physical replica must project the matched second")
        if self.second_replica_watchdog_a100_microseconds != (
            2 * self.projected_second_replica_a100_microseconds
        ):
            raise ValueError("the second replay watchdog must be exactly 2x projection")
        expected = (
            self.prior_campaign_a100_microseconds
            + self.first_replica_a100_microseconds
            + self.projected_second_replica_a100_microseconds
        )
        if self.projected_campaign_a100_microseconds != expected:
            raise ValueError("replay projection must include prior, burst, and second replica")
        if self.tripwire_a100_microseconds != GTOK_TRIPWIRE_A100_MICROSECONDS:
            raise ValueError("replay projection must preserve the 12 A100-hour tripwire")
        if self.projected_campaign_a100_microseconds > self.tripwire_a100_microseconds:
            raise GTokDeterminismV2Stop(
                "DETERMINISM_REPLAY_PROJECTED_TRIPWIRE",
                "projected second replica would cross the cumulative 12-hour meter",
            )

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256(
            "weft1_gtok_v2_determinism_replay_pair_projection", self
        )


def project_second_determinism_replay_replica_v2(
    first: DeterminismReplayReplicaV2,
    *,
    replay_plan_binding: DeterminismReplayPlanBindingV2,
    first_lifecycle_a100_microseconds: int,
    prior_campaign_a100_microseconds: int,
) -> DeterminismReplayPairProjectionV2:
    if first.replica_index != 0:
        raise ValueError("only the first fresh replica may project the second")
    if (
        first.fingerprint.replay_plan_binding_sha256
        != replay_plan_binding.receipt_sha256
    ):
        raise ValueError("first replay differs from its canonical plan binding")
    if (
        type(first_lifecycle_a100_microseconds) is not int
        or first_lifecycle_a100_microseconds < first.charged_device_microseconds
    ):
        raise ValueError(
            "first replay lifecycle charge must include its inner device timer"
        )
    return DeterminismReplayPairProjectionV2(
        replay_plan_binding_sha256=replay_plan_binding.receipt_sha256,
        first_replica_a100_microseconds=first_lifecycle_a100_microseconds,
        projected_second_replica_a100_microseconds=(
            first_lifecycle_a100_microseconds
        ),
        second_replica_watchdog_a100_microseconds=(
            2 * first_lifecycle_a100_microseconds
        ),
        prior_campaign_a100_microseconds=prior_campaign_a100_microseconds,
        projected_campaign_a100_microseconds=(
            prior_campaign_a100_microseconds
            + 2 * first_lifecycle_a100_microseconds
        ),
    )


@dataclass(frozen=True)
class PrecalibrationDeterminismReplayReceiptV2:
    policy_attestation_receipt_sha256: str
    training_runtime_receipt_sha256: str
    code_closure_receipt_sha256: str
    replay_plan_binding: DeterminismReplayPlanBindingV2
    pair_projection: DeterminismReplayPairProjectionV2
    fingerprint: DeterminismReplayFingerprintV2
    replay_fingerprint_sha256s: tuple[str, str]
    attempts: tuple[tuple[int, int, str], tuple[int, int, str]]
    inner_device_microseconds: tuple[int, int]
    charged_a100_microseconds: int
    status: str = "GREEN_EXACT_A100_FLASH_REPLAY"

    def __post_init__(self) -> None:
        for name in (
            "policy_attestation_receipt_sha256",
            "training_runtime_receipt_sha256",
            "code_closure_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.fingerprint, DeterminismReplayFingerprintV2):
            raise TypeError("physical replay receipt requires its exact fingerprint")
        if not isinstance(
            self.replay_plan_binding, DeterminismReplayPlanBindingV2
        ) or not isinstance(
            self.pair_projection, DeterminismReplayPairProjectionV2
        ):
            raise TypeError("physical replay receipt requires plan and projection bindings")
        if (
            self.fingerprint.replay_plan_binding_sha256
            != self.replay_plan_binding.receipt_sha256
            or self.pair_projection.replay_plan_binding_sha256
            != self.replay_plan_binding.receipt_sha256
            or self.fingerprint.training_plan_sha256
            != self.replay_plan_binding.representative_plan_sha256
        ):
            raise ValueError("physical replay receipt plan joins drifted")
        if self.replay_fingerprint_sha256s != (
            self.fingerprint.receipt_sha256,
            self.fingerprint.receipt_sha256,
        ):
            raise ValueError("the two physical replay receipts are not exactly equal")
        if tuple(index for index, _, _ in self.attempts) != (0, 1):
            raise ValueError("physical replay attempts must be the two fresh replicas")
        if any(charge < 1 for _, charge, _ in self.attempts):
            raise ValueError("physical replay attempts must both be charged")
        if (
            len(self.inner_device_microseconds) != 2
            or any(
                type(charge) is not int or charge < 1
                for charge in self.inner_device_microseconds
            )
            or any(
                inner > outer
                for inner, (_, outer, _) in zip(
                    self.inner_device_microseconds,
                    self.attempts,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "inner replay device timers must be positive diagnostics within lifecycle charges"
            )
        if any(
            not gpu.startswith("GPU-") or len(gpu) <= 4
            for _, _, gpu in self.attempts
        ):
            raise ValueError("physical replay attempts require GPU provenance")
        if self.charged_a100_microseconds != sum(
            charge for _, charge, _ in self.attempts
        ):
            raise ValueError("replay A100 meter must include both fresh attempts")
        if (
            self.attempts[0][1]
            != self.pair_projection.first_replica_a100_microseconds
            or self.attempts[1][1]
            > self.pair_projection.second_replica_watchdog_a100_microseconds
            or self.pair_projection.prior_campaign_a100_microseconds
            + self.charged_a100_microseconds
            > self.pair_projection.tripwire_a100_microseconds
        ):
            raise GTokDeterminismV2Stop(
                "DETERMINISM_REPLAY_RUNTIME_METER",
                "physical replay charge crossed its watchdog or cumulative tripwire",
            )
        if self.status != "GREEN_EXACT_A100_FLASH_REPLAY":
            raise ValueError("physical replay receipt is not GREEN")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256(DETERMINISM_REPLAY_SCHEMA_V2, self)


@dataclass(frozen=True)
class DryRunDeterminismReplayV2:
    fingerprint_receipt_sha256: str
    measured_device_microseconds: int
    authority_status: str = "NON_AUTHORITATIVE_CPU_OR_INJECTED_REPLAY"

    def __post_init__(self) -> None:
        _require_sha256(self.fingerprint_receipt_sha256, "fingerprint_receipt_sha256")
        if type(self.measured_device_microseconds) is not int or self.measured_device_microseconds < 2:
            raise ValueError("dry replay must include both measured replicas")
        if self.authority_status != "NON_AUTHORITATIVE_CPU_OR_INJECTED_REPLAY":
            raise ValueError("dry replay authority status drifted")


def _tensor_descriptor_v2(tensor: torch.Tensor) -> Mapping[str, Any]:
    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype),
        "shape": tuple(int(item) for item in value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _normalized_state_v2(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "value": _tensor_descriptor_v2(value)}
    if isinstance(value, Mapping):
        rows = [
            (_normalized_state_v2(key), _normalized_state_v2(item))
            for key, item in value.items()
        ]
        rows.sort(key=lambda row: canonical_json_bytes(row[0]))
        return {"kind": "mapping", "rows": tuple(rows)}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": tuple(_normalized_state_v2(item) for item in value)}
    if isinstance(value, list):
        return {"kind": "list", "items": tuple(_normalized_state_v2(item) for item in value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise GTokDeterminismV2Error("state contains a non-finite scalar")
        return {"kind": type(value).__name__, "value": value}
    raise GTokDeterminismV2Error(
        f"state contains unsupported value type {type(value).__name__}"
    )


def _state_sha256_v2(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(_normalized_state_v2(value))).hexdigest()


def _batch_sha256_v2(batch: PackedBatchV2) -> str:
    return _state_sha256_v2(
        {
            "attention_mask": batch.attention_mask,
            "document_ids": batch.document_ids,
            "input_ids": batch.input_ids,
            "target_ids": batch.target_ids,
            "valid_prediction_count": batch.valid_prediction_count,
        }
    )


def _replay_batch_v2(
    *,
    label: str,
    rows: int,
    sequence_length: int,
    vocab_size: int,
    run_seed: int,
) -> tuple[PackedBatchV2, ReplayBatchShapeV2]:
    if label not in ("full", "terminal_partial"):
        raise ValueError("replay batch label is invalid")
    if rows < 1 or sequence_length < 4 or vocab_size < 8:
        raise ValueError("replay batch dimensions are invalid")
    positions = torch.arange(sequence_length, dtype=torch.int64).view(1, -1)
    row_ids = torch.arange(rows, dtype=torch.int64).view(-1, 1)
    token_modulus = vocab_size - 4
    input_ids = ((positions * 131 + row_ids * 977 + abs(run_seed)) % token_modulus) + 4
    attention_mask = torch.ones((rows, sequence_length), dtype=torch.bool)
    midpoint = sequence_length // 2
    document_ids = row_ids.expand(-1, sequence_length).clone() * 2
    document_ids[:, midpoint:] += 1
    input_ids[:, midpoint] = 3
    if label == "terminal_partial":
        valid_until = max(midpoint + 1, sequence_length - max(2, sequence_length // 8))
        attention_mask[-1, valid_until:] = False
        document_ids[-1, valid_until:] = -1
        input_ids[-1, valid_until:] = 0
    target_ids = torch.full_like(input_ids, -100)
    target_ids[:, :-1] = input_ids[:, 1:]
    target_ids[:, midpoint - 1] = -100
    target_ids.masked_fill_(~attention_mask, -100)
    target_ids[:, -1] = -100
    valid_predictions = int(target_ids.ne(-100).sum().item())
    batch = PackedBatchV2(
        input_ids=input_ids,
        target_ids=target_ids,
        document_ids=document_ids,
        attention_mask=attention_mask,
        completed_raw_bytes=0,
        completed_document_count=0,
        valid_prediction_count=valid_predictions,
    )
    shape = ReplayBatchShapeV2(
        label=label,
        batch_rows=rows,
        sequence_length=sequence_length,
        valid_prediction_count=valid_predictions,
        batch_sha256=_batch_sha256_v2(batch),
    )
    return batch, shape


def _shapes_from_plan_v2(
    plan: Any,
    *,
    global_batch_sequences: int,
    sequence_length: int,
) -> tuple[int, int]:
    if type(plan.optimizer_steps) is not int or plan.optimizer_steps < 2:
        raise ValueError("determinism replay plan requires at least two optimizer steps")
    full_slots = global_batch_sequences * sequence_length
    final_slots = plan.compute_token_slots - (plan.optimizer_steps - 1) * full_slots
    if final_slots < sequence_length or final_slots % sequence_length:
        raise GTokDeterminismV2Stop(
            "TERMINAL_PARTIAL_SHAPE_INVALID",
            "training plan does not encode a whole-row terminal batch",
        )
    terminal_rows = final_slots // sequence_length
    if terminal_rows >= global_batch_sequences:
        raise GTokDeterminismV2Stop(
            "TERMINAL_PARTIAL_SHAPE_UNAVAILABLE",
            "training plan has no smaller terminal logical batch",
        )
    return global_batch_sequences, terminal_rows


def _evaluation_output_sha256_v2(
    model: torch.nn.Module,
    *,
    batch: PackedBatchV2,
    device: torch.device,
    microbatch_sequences: int,
    evaluation_tokens: int,
) -> str:
    rows = min(int(batch.input_ids.shape[0]), microbatch_sequences, 2)
    length = int(batch.input_ids.shape[1])
    width = min(evaluation_tokens, length)
    start = max(0, length // 2 - width // 2)
    stop = start + width
    ids = batch.input_ids[-rows:, start:stop].to(device)
    documents = batch.document_ids[-rows:, start:stop].to(device)
    mask = batch.attention_mask[-rows:, start:stop].to(device)
    model.eval()
    with torch.no_grad(), _autocast(device):
        output = model(
            ids,
            attention_mask=mask,
            document_ids=documents,
            labels=None,
        )
    value = output.logits.detach()
    result = _state_sha256_v2(value)
    model.train()
    return result


def _profiler_context_v2(device: torch.device, *, enabled: bool):
    if not enabled:
        return nullcontext(None)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(activities=activities, record_shapes=False)


def _backend_operator_names_v2(profiler: Any) -> tuple[str, ...]:
    if profiler is None:
        return ()
    try:
        return tuple(
            sorted(
                {
                    str(event.key)
                    for event in profiler.key_averages()
                    if "scaled_dot_product" in str(event.key)
                    or "flash_attention" in str(event.key)
                    or "efficient_attention" in str(event.key)
                    or "cudnn_attention" in str(event.key)
                }
            )
        )
    except Exception as error:
        raise GTokDeterminismV2Stop(
            "FUSED_SDPA_BACKEND_EVIDENCE_FAILED",
            "the physical replay profiler emitted no readable backend trace",
        ) from error


def _validate_flash_backend_trace_v2(operator_names: tuple[str, ...]) -> None:
    lowered = tuple(name.lower() for name in operator_names)
    flash = tuple(name for name in lowered if "flash_attention" in name)
    rejected = tuple(
        name
        for name in lowered
        if "efficient_attention" in name
        or "cudnn_attention" in name
        or "attention_math" in name
    )
    if rejected:
        raise GTokDeterminismV2Stop(
            "FUSED_SDPA_BACKEND_DRIFT",
            "physical replay selected a non-Flash SDPA backend",
        )
    if not flash or not any("backward" in name for name in flash):
        raise GTokDeterminismV2Stop(
            "FUSED_SDPA_BACKEND_FAILED",
            "Flash SDPA forward/backward was not observed; math fallback is prohibited",
        )


def _elapsed_device_microseconds_v2(start_ns: int, device: torch.device) -> int:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return max(1, math.ceil((time.perf_counter_ns() - start_ns) / 1_000))


def _execute_replay_replica_v2(
    *,
    replica_index: int,
    model_factory: Callable[[], torch.nn.Module],
    optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
    plan: Any,
    vocab_size: int,
    initialization_seed: int,
    run_seed: int,
    device: torch.device,
    microbatch_sequences: int,
    policy_receipt_sha256: str,
    replay_plan_binding_sha256: str,
    gpu_uuid_provenance: str | None,
    global_batch_sequences: int,
    sequence_length: int,
    evaluation_tokens: int,
    require_fused_flash: bool,
    watchdog_limit_device_microseconds: int | None = None,
    prior_campaign_device_microseconds: int = 0,
    campaign_tripwire_device_microseconds: int | None = None,
) -> DeterminismReplayReplicaV2:
    """Execute one fresh replica; callers may inject only for non-authoritative tests."""

    start_ns = time.perf_counter_ns()

    def guard_meter() -> int:
        elapsed = _elapsed_device_microseconds_v2(start_ns, device)
        if (
            watchdog_limit_device_microseconds is not None
            and elapsed > watchdog_limit_device_microseconds
        ):
            raise GTokDeterminismV2Stop(
                "DETERMINISM_REPLAY_WATCHDOG",
                "physical replay exceeded its registered 2x watchdog",
            )
        if (
            campaign_tripwire_device_microseconds is not None
            and prior_campaign_device_microseconds + elapsed
            > campaign_tripwire_device_microseconds
        ):
            raise GTokDeterminismV2Stop(
                "DETERMINISM_REPLAY_RUNTIME_TRIPWIRE",
                "physical replay crossed the cumulative 12-hour meter",
            )
        return elapsed

    full_rows, terminal_rows = _shapes_from_plan_v2(
        plan,
        global_batch_sequences=global_batch_sequences,
        sequence_length=sequence_length,
    )
    full_batch, full_shape = _replay_batch_v2(
        label="full",
        rows=full_rows,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        run_seed=run_seed,
    )
    terminal_batch, terminal_shape = _replay_batch_v2(
        label="terminal_partial",
        rows=terminal_rows,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        run_seed=run_seed,
    )
    guard_meter()
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        model = model_factory()
        optimizer = optimizer_factory(model)
        model.to(device)
        model.train()
        model_hashes: list[tuple[str, str]] = []
        optimizer_hashes: list[tuple[str, str]] = []
        output_hashes: list[tuple[str, str]] = []
        profiler_enabled = require_fused_flash

        with _profiler_context_v2(device, enabled=profiler_enabled) as profiler:
            for label, batch, step in (
                ("full", full_batch, 1),
                ("terminal_partial", terminal_batch, plan.optimizer_steps),
            ):
                _execute_optimizer_step_v2(
                    model,
                    optimizer,  # type: ignore[arg-type]
                    batch=batch,
                    step=step,
                    plan=plan,
                    device=device,
                    microbatch_sequences=microbatch_sequences,
                )
                output_hashes.append(
                    (
                        label,
                        _evaluation_output_sha256_v2(
                            model,
                            batch=batch,
                            device=device,
                            microbatch_sequences=microbatch_sequences,
                            evaluation_tokens=evaluation_tokens,
                        ),
                    )
                )
                model_hashes.append((label, _state_sha256_v2(model.state_dict())))
                optimizer_hashes.append(
                    (label, _state_sha256_v2(optimizer.state_dict()))
                )
                guard_meter()
        operators = _backend_operator_names_v2(profiler)
        if require_fused_flash:
            _validate_flash_backend_trace_v2(operators)
        fingerprint = DeterminismReplayFingerprintV2(
            policy_receipt_sha256=policy_receipt_sha256,
            replay_plan_binding_sha256=replay_plan_binding_sha256,
            training_plan_sha256=_require_sha256(
                str(plan.receipt_sha256), "training_plan_sha256"
            ),
            vocab_size=vocab_size,
            initialization_seed=initialization_seed,
            run_seed=run_seed,
            microbatch_sequences=microbatch_sequences,
            shapes=(full_shape, terminal_shape),
            model_state_sha256_by_shape=tuple(model_hashes),  # type: ignore[arg-type]
            optimizer_state_sha256_by_shape=tuple(optimizer_hashes),  # type: ignore[arg-type]
            evaluation_output_sha256_by_shape=tuple(output_hashes),  # type: ignore[arg-type]
            fused_backend_operator_names=operators,
        )
        charged = guard_meter()
        return DeterminismReplayReplicaV2(
            replica_index=replica_index,
            fingerprint=fingerprint,
            charged_device_microseconds=charged,
            gpu_uuid_provenance=gpu_uuid_provenance,
        )
    except GTokDeterminismV2Stop:
        raise
    except RuntimeError as error:
        raise GTokDeterminismV2Stop(
            "DETERMINISM_REPLAY_EXECUTION_FAILED",
            "the fixed fused backend or deterministic operator policy rejected the replay",
        ) from error
    finally:
        del optimizer
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _assert_matching_replay_pair_v2(
    first: DeterminismReplayReplicaV2,
    second: DeterminismReplayReplicaV2,
) -> None:
    if not isinstance(first, DeterminismReplayReplicaV2) or not isinstance(
        second, DeterminismReplayReplicaV2
    ):
        raise TypeError("determinism gate requires two typed replay replicas")
    if (first.replica_index, second.replica_index) != (0, 1):
        raise ValueError("determinism replay pair order drifted")
    if (
        first.fingerprint != second.fingerprint
        or first.fingerprint.receipt_sha256 != second.fingerprint.receipt_sha256
    ):
        raise GTokDeterminismV2Stop(
            "DETERMINISM_REPLAY_MISMATCH",
            "fresh replicas differ in model state, optimizer state, output, or receipt",
        )


def validate_determinism_replay_pair_v2(
    first: DeterminismReplayReplicaV2,
    second: DeterminismReplayReplicaV2,
) -> DryRunDeterminismReplayV2:
    """Validate injected/CPU replicas without exposing a physical mint surface."""

    _assert_matching_replay_pair_v2(first, second)
    return DryRunDeterminismReplayV2(
        fingerprint_receipt_sha256=first.fingerprint.receipt_sha256,
        measured_device_microseconds=(
            first.charged_device_microseconds + second.charged_device_microseconds
        ),
    )


def _mint_precalibration_determinism_replay_receipt_v2(
    first: DeterminismReplayReplicaV2,
    second: DeterminismReplayReplicaV2,
    *,
    policy_attestation: CudaDeterminismAttestationV2,
    replay_plan_binding: DeterminismReplayPlanBindingV2,
    pair_projection: DeterminismReplayPairProjectionV2,
    first_lifecycle_a100_microseconds: int,
    second_lifecycle_a100_microseconds: int,
    training_runtime_receipt_sha256: str,
    code_closure_receipt_sha256: str,
) -> PrecalibrationDeterminismReplayReceiptV2:
    """Private mint reached only by the non-injectable physical runner."""

    _assert_matching_replay_pair_v2(first, second)
    if policy_attestation.authority_status != "AUTHORITATIVE_A100_FLASH_ONLY":
        raise GTokDeterminismV2Stop(
            "CUDA_POLICY_A100_REQUIRED",
            "a CPU/injected replay cannot mint the physical gate",
        )
    _validate_flash_backend_trace_v2(first.fingerprint.fused_backend_operator_names)
    if first.fingerprint.policy_receipt_sha256 != policy_attestation.policy.receipt_sha256:
        raise GTokDeterminismV2Stop(
            "CUDA_POLICY_REPLAY_JOIN_FAILED",
            "replay fingerprint differs from its policy attestation",
        )
    for replica in (first, second):
        if replica.gpu_uuid_provenance is None:
            raise GTokDeterminismV2Stop(
                "CUDA_POLICY_GPU_PROVENANCE_MISSING",
                "physical replay lacks per-attempt GPU provenance",
            )
    for lifecycle_charge, replica in (
        (first_lifecycle_a100_microseconds, first),
        (second_lifecycle_a100_microseconds, second),
    ):
        if (
            type(lifecycle_charge) is not int
            or lifecycle_charge < replica.charged_device_microseconds
        ):
            raise GTokDeterminismV2Stop(
                "DETERMINISM_REPLAY_RUNTIME_METER",
                "lifecycle replay charge does not include its inner device timer",
            )
    return PrecalibrationDeterminismReplayReceiptV2(
        policy_attestation_receipt_sha256=policy_attestation.receipt_sha256,
        training_runtime_receipt_sha256=_require_sha256(
            training_runtime_receipt_sha256, "training_runtime_receipt_sha256"
        ),
        code_closure_receipt_sha256=_require_sha256(
            code_closure_receipt_sha256, "code_closure_receipt_sha256"
        ),
        replay_plan_binding=replay_plan_binding,
        pair_projection=pair_projection,
        fingerprint=first.fingerprint,
        replay_fingerprint_sha256s=(
            first.fingerprint.receipt_sha256,
            second.fingerprint.receipt_sha256,
        ),
        attempts=(
            (
                first.replica_index,
                first_lifecycle_a100_microseconds,
                str(first.gpu_uuid_provenance),
            ),
            (
                second.replica_index,
                second_lifecycle_a100_microseconds,
                str(second.gpu_uuid_provenance),
            ),
        ),
        inner_device_microseconds=(
            first.charged_device_microseconds,
            second.charged_device_microseconds,
        ),
        charged_a100_microseconds=(
            first_lifecycle_a100_microseconds + second_lifecycle_a100_microseconds
        ),
    )


def execute_precalibration_determinism_replay_replica_v2(
    *,
    replica_index: int,
    policy_attestation: CudaDeterminismAttestationV2,
    replay_plan_binding: DeterminismReplayPlanBindingV2,
    plan: TrainingPlanV2,
    device: torch.device,
    microbatch_sequences: int,
    gpu_uuid_provenance: str,
    watchdog_limit_a100_microseconds: int | None,
    prior_campaign_a100_microseconds: int,
) -> DeterminismReplayReplicaV2:
    """Execute one non-injectable physical replica under campaign-owned lifecycle."""

    if (
        not isinstance(plan, TrainingPlanV2)
        or not isinstance(replay_plan_binding, DeterminismReplayPlanBindingV2)
        or plan.receipt_sha256
        != replay_plan_binding.representative_plan_sha256
    ):
        raise TypeError("physical determinism replay requires its representative typed plan")
    if policy_attestation.authority_status != "AUTHORITATIVE_A100_FLASH_ONLY":
        raise GTokDeterminismV2Stop(
            "CUDA_POLICY_A100_REQUIRED",
            "physical replay requires the authoritative CUDA policy attestation",
        )
    sequence_length = int(PACKING_BINDING_V2["sequence_length"])
    global_batch_sequences = int(PACKING_BINDING_V2["batch_sequences"])
    _, observed_terminal_rows = _shapes_from_plan_v2(
        plan,
        global_batch_sequences=global_batch_sequences,
        sequence_length=sequence_length,
    )
    if observed_terminal_rows != replay_plan_binding.terminal_rows:
        raise ValueError("representative plan terminal shape differs from its binding")

    def model_factory() -> torch.nn.Module:
        return build_gtok_proxy_model_v2(
            vocab_size=replay_plan_binding.vocab_size,
            initialization_seed=(
                replay_plan_binding.representative_initialization_seed
            ),
            run_seed=replay_plan_binding.representative_training_seed,
        )

    return _execute_replay_replica_v2(
        replica_index=replica_index,
        model_factory=model_factory,
        optimizer_factory=build_flat_a1_adamw_v2,
        plan=plan,
        vocab_size=replay_plan_binding.vocab_size,
        initialization_seed=replay_plan_binding.representative_initialization_seed,
        run_seed=replay_plan_binding.representative_training_seed,
        device=device,
        microbatch_sequences=microbatch_sequences,
        policy_receipt_sha256=policy_attestation.policy.receipt_sha256,
        replay_plan_binding_sha256=replay_plan_binding.receipt_sha256,
        gpu_uuid_provenance=gpu_uuid_provenance,
        global_batch_sequences=global_batch_sequences,
        sequence_length=sequence_length,
        evaluation_tokens=DETERMINISM_REPLAY_EVALUATION_TOKENS_V2,
        require_fused_flash=True,
        watchdog_limit_device_microseconds=watchdog_limit_a100_microseconds,
        prior_campaign_device_microseconds=prior_campaign_a100_microseconds,
        campaign_tripwire_device_microseconds=GTOK_TRIPWIRE_A100_MICROSECONDS,
    )


def write_precalibration_determinism_replay_receipt_v2(
    path: Path,
    receipt: PrecalibrationDeterminismReplayReceiptV2,
) -> str:
    """Persist one immutable green replay receipt; never overwrite on mismatch."""

    if not isinstance(path, Path) or not isinstance(
        receipt, PrecalibrationDeterminismReplayReceiptV2
    ):
        raise TypeError("replay receipt writer requires pathlib.Path and typed receipt")
    target = assert_no_symlink_ancestors(path)
    envelope = {
        "payload": asdict(receipt),
        "receipt_sha256": receipt.receipt_sha256,
        "schema": DETERMINISM_REPLAY_SCHEMA_V2,
    }
    raw = canonical_json_bytes(envelope) + b"\n"
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
            raise GTokDeterminismV2Error(
                "stored determinism replay receipt differs on resume"
            )
        return receipt.receipt_sha256
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return receipt.receipt_sha256


def load_precalibration_determinism_replay_receipt_v2(
    path: Path,
) -> PrecalibrationDeterminismReplayReceiptV2:
    """Strictly load a completed receipt for receipt-boundary resume."""

    raw, envelope = load_canonical_json_snapshot(assert_no_symlink_ancestors(path))
    if (
        raw != canonical_json_bytes(envelope) + b"\n"
        or set(envelope) != {"payload", "receipt_sha256", "schema"}
        or envelope.get("schema") != DETERMINISM_REPLAY_SCHEMA_V2
        or not isinstance(envelope.get("payload"), Mapping)
    ):
        raise GTokDeterminismV2Error("determinism replay receipt envelope drifted")
    payload = dict(envelope["payload"])
    try:
        binding_payload = dict(payload["replay_plan_binding"])
        binding_payload["equivalent_training_seeds"] = tuple(
            binding_payload["equivalent_training_seeds"]
        )
        binding_payload["equivalent_plan_receipt_sha256s"] = tuple(
            binding_payload["equivalent_plan_receipt_sha256s"]
        )
        payload["replay_plan_binding"] = DeterminismReplayPlanBindingV2(
            **binding_payload
        )
        payload["pair_projection"] = DeterminismReplayPairProjectionV2(
            **dict(payload["pair_projection"])
        )
        fingerprint_payload = dict(payload["fingerprint"])
        shapes = tuple(ReplayBatchShapeV2(**dict(row)) for row in fingerprint_payload["shapes"])
        fingerprint_payload["shapes"] = shapes
        for name in (
            "model_state_sha256_by_shape",
            "optimizer_state_sha256_by_shape",
            "evaluation_output_sha256_by_shape",
            "fused_backend_operator_names",
        ):
            fingerprint_payload[name] = tuple(
                tuple(row) if isinstance(row, list) else row
                for row in fingerprint_payload[name]
            )
        fingerprint = DeterminismReplayFingerprintV2(**fingerprint_payload)
        payload["fingerprint"] = fingerprint
        payload["replay_fingerprint_sha256s"] = tuple(
            payload["replay_fingerprint_sha256s"]
        )
        payload["attempts"] = tuple(tuple(row) for row in payload["attempts"])
        payload["inner_device_microseconds"] = tuple(
            payload["inner_device_microseconds"]
        )
        receipt = PrecalibrationDeterminismReplayReceiptV2(**payload)
    except (KeyError, TypeError, ValueError) as error:
        raise GTokDeterminismV2Error("determinism replay receipt payload is invalid") from error
    if receipt.receipt_sha256 != envelope.get("receipt_sha256"):
        raise GTokDeterminismV2Error("determinism replay receipt identity drifted")
    return receipt


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG_V2",
    "CUDA_DETERMINISM_POLICY_SHA256_V2",
    "CUDA_DETERMINISM_POLICY_V2",
    "CudaDeterminismAttestationV2",
    "CudaDeterminismPolicyV2",
    "DETERMINISM_REPLAY_SCHEMA_V2",
    "DeterminismReplayFingerprintV2",
    "DeterminismReplayPairProjectionV2",
    "DeterminismReplayPlanBindingV2",
    "DeterminismReplayReplicaV2",
    "DryRunDeterminismReplayV2",
    "GTokDeterminismV2Error",
    "GTokDeterminismV2Stop",
    "PrecalibrationDeterminismReplayReceiptV2",
    "ReplayBatchShapeV2",
    "apply_and_attest_cuda_determinism_policy_v2",
    "canonical_determinism_replay_plan_bindings_v2",
    "execute_precalibration_determinism_replay_replica_v2",
    "load_precalibration_determinism_replay_receipt_v2",
    "project_second_determinism_replay_replica_v2",
    "validate_determinism_replay_pair_v2",
    "write_precalibration_determinism_replay_receipt_v2",
]
