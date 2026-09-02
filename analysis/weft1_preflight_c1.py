"""PF-3.1 fail-closed inventory for the WEFT-1 C1 width coordinate.

PF-3 closes Catch #33 only if every tensor can be assigned before the
registered ten-step run.  The current Hadamard router has shape ``E x d``:
fan-in scales while fan-out is fixed.  It is therefore neither hidden, input,
vector/scalar, nor the tied readout class.  This receipt inventories all three
registered CPU widths and returns Catch #35 before PF-3 initialization,
optimizer construction, forward execution, or activation-RMS measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import torch

from models.ablation_lm.config import AblationLMConfig, MUP_D_HEAD_BASE
from models.ablation_lm.memory import ReadOnlyLatentMemory
from models.ablation_lm.model import AblationLM
from models.ablation_lm.mup import (
    MUP_ALPHA_EMBEDDING,
    MUP_ALPHA_OUTPUT,
    MUP_BASE_HEAD_DIM,
    MUP_BASE_VOCAB_SIZE,
    MUP_BETAS,
    MUP_DECAY_IMPLEMENTATION,
    MUP_EPSILON,
    MUP_ETA_BASE,
    MUP_NUMERICS_STATUS,
    MUP_RESIDUAL_MULTIPLIER,
    MUP_SIGMA_BASE,
    MUP_SIGMA_EMBEDDING,
    MUP_WEIGHT_DECAY,
    MuPClassificationIssue,
    MuPParameterAssignment,
    audit_mup_parameters,
)
from models.ablation_lm.rng import derive_module_seed


PREFLIGHT_PROGRAM_FILE = "STRATEGY_PREFLIGHT_PROGRAM_20260902.md"
PREFLIGHT_PROGRAM_BYTES = 15_575
PREFLIGHT_PROGRAM_SHA256 = "ceaa5338830307d3783296b8a4aef7bb87962eb35535d392f4c6d217dff88a5b"
PREFLIGHT_RATIFICATION_FILE = "STRATEGY_PREFLIGHT_RATIFICATION_20260902.md"
PREFLIGHT_RATIFICATION_BYTES = 2_233
PREFLIGHT_RATIFICATION_SHA256 = "4a13054d38c68e5e9476330528649d445ff845e639e0a36bb01641b54ef66965"
PF2_AUTHORITY_FILE = "STRATEGY_PREFLIGHT_AMENDMENT_PF2_20260902.md"
PF2_AUTHORITY_BYTES = 13_097
PF2_AUTHORITY_SHA256 = "be11390c28ae36210a1571f7c6d358ee54e977d239f5344f7e6402212448eb05"
S81_AUTHORITY_FILE = "STRATEGY_HANDOFF_S81_AMENDMENT_20260902.md"
S81_AUTHORITY_BYTES = 3_403
S81_AUTHORITY_SHA256 = "dd79aaa6fd9bab15bf02aaef28f99f47c745d63eed6db2c9b929b4bb1cfbb418"
PF3_AUTHORITY_FILE = "STRATEGY_PREFLIGHT_AMENDMENT_PF3_20260902.md"
PF3_AUTHORITY_BYTES = 14_632
PF3_AUTHORITY_SHA256 = "7f0081e504366ce98f8bf183b7e14c0bed47647aa381196a9d5e9540b5334cef"
BUILD_HANDOFF_AUTHORITY = "STRATEGY_TO_CODING_AGENT_LOOM1_HANDOFF_20260826.md#8"
BUILD_HANDOFF_AUTHORITY_FILE = "STRATEGY_TO_CODING_AGENT_LOOM1_HANDOFF_20260826.md"
BUILD_HANDOFF_AUTHORITY_BYTES = 61_329
BUILD_HANDOFF_AUTHORITY_SHA256 = "498f34b5966f0879c7f0a15ca8be02a603558781c35f59f03fb29cc9edd3eb02"
BUILD_HANDOFF_AUTHORITY_DRIVE_ID = "1XaE81mfqTOYEYGFMa-ZJwpLW-KQtMMC_"

C1_CPU_WIDTHS = (128, 256, 512)
C1_DEFERRED_GPU_WIDTHS = (1_024,)
C1_HEAD_DIM = 64
C1_BATCH_SIZE = 2
C1_SEQUENCE_LENGTH = 64
C1_TRAINING_STEPS = 10
C1_WIDTH_DRIFT_LIMIT = 2.0
C1_CATCH_NUMBER = 35
C1_SEED = 20_260_902
C1_EXECUTION_STATUS = "blocked_before_pf3_initialization"
C1_PROVISIONAL_BASE_SHAPE = "d512_head64_q8_kv4_ff1408_lanes2x128_vocab32768"
C1_CATCH_DISPOSITION = "catch_35_hadamard_router_mup_class_unbound"
S81_FUTURE_HEAD_DIM_POLICY = "future_WEFT_d_head_not_64_requires_explicit_base_shape_implementation"


class C1CoordinateCatch(RuntimeError):
    """Raised when a caller attempts to promote the blocked receipt."""


class C1AuthorityIntegrityError(RuntimeError):
    """Raised when a local C1 authority is absent or is not byte-exact."""


@dataclass(frozen=True)
class AuthorityVerification:
    filename: str
    expected_bytes: int
    actual_bytes: int
    expected_sha256: str
    actual_sha256: str
    verified: bool

    def __post_init__(self) -> None:
        if not self.filename:
            raise ValueError("C1 authority filename must be non-empty")
        if type(self.expected_bytes) is not int or self.expected_bytes < 1:
            raise ValueError("C1 expected authority bytes must be positive")
        if type(self.actual_bytes) is not int or self.actual_bytes < 0:
            raise ValueError("C1 actual authority bytes must be non-negative")
        for digest in (self.expected_sha256, self.actual_sha256):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("C1 authority SHA-256 must be lowercase hexadecimal")
        expected_verified = (
            self.actual_bytes == self.expected_bytes
            and self.actual_sha256 == self.expected_sha256
        )
        if self.verified is not expected_verified:
            raise ValueError("C1 authority verification flag contradicts the bytes or hash")


@dataclass(frozen=True)
class C1WidthTopology:
    width: int
    head_dim: int
    q_heads: int
    kv_heads: int
    d_ff: int
    scratch_lanes: int
    scratch_width_per_lane: int
    n_prelude_layers: int
    n_core_blocks: int
    n_coda_layers: int
    recurrent_steps: int
    attention_logit_scale: float

    @property
    def total_scratch_width(self) -> int:
        return self.scratch_lanes * self.scratch_width_per_lane

    @property
    def unique_decoder_blocks(self) -> int:
        return self.n_prelude_layers + self.n_core_blocks + self.n_coda_layers

    @property
    def executed_decoder_block_passes(self) -> int:
        return self.n_prelude_layers + self.recurrent_steps * self.n_core_blocks + self.n_coda_layers

    def is_bound(self) -> bool:
        return (
            self.width in C1_CPU_WIDTHS
            and self.head_dim == C1_HEAD_DIM == MUP_D_HEAD_BASE == MUP_BASE_HEAD_DIM
            and self.q_heads == self.width // 64
            and self.kv_heads == self.width // 128
            and self.q_heads == 2 * self.kv_heads
            and self.d_ff == 11 * self.width // 4
            and self.scratch_lanes == 2
            and self.scratch_width_per_lane == self.width // 4
            and (self.n_prelude_layers, self.n_core_blocks, self.n_coda_layers) == (4, 2, 4)
            and self.recurrent_steps == 4
            and self.attention_logit_scale == math.sqrt(MUP_D_HEAD_BASE) / self.head_dim
        )


@dataclass(frozen=True)
class WidthClassificationReceipt:
    width: int
    width_multiplier: float
    topology: C1WidthTopology
    unique_trainable_tensors: int
    classified_tensors: tuple[MuPParameterAssignment, ...]
    unclassified_tensors: tuple[MuPClassificationIssue, ...]
    classified_map_sha256: str

    def __post_init__(self) -> None:
        if self.width != self.topology.width or self.width not in C1_CPU_WIDTHS:
            raise ValueError("C1 width receipt does not match a registered topology")
        if self.width_multiplier != self.width / 512:
            raise ValueError("C1 width multiplier is inconsistent")
        if not self.topology.is_bound():
            raise ValueError("C1 width topology is not bound")
        if self.unique_trainable_tensors != (
            len(self.classified_tensors) + len(self.unclassified_tensors)
        ):
            raise ValueError("C1 tensor inventory count is inconsistent")
        classified_names = tuple(item.canonical_name for item in self.classified_tensors)
        unclassified_names = tuple(item.canonical_name for item in self.unclassified_tensors)
        if len(set(classified_names)) != len(classified_names):
            raise ValueError("C1 classified tensor names are not unique")
        if len(set(unclassified_names)) != len(unclassified_names):
            raise ValueError("C1 unclassified tensor names are not unique")
        if set(classified_names) & set(unclassified_names):
            raise ValueError("C1 classified and unclassified inventories overlap")
        if self.classified_map_sha256 != _map_sha(self.classified_tensors):
            raise ValueError("C1 classified tensor map hash is inconsistent")


@dataclass(frozen=True)
class C1PreflightReceipt:
    program_sha256: str
    ratification_sha256: str
    authorities: tuple[AuthorityVerification, ...]
    build_handoff_authority: str
    build_handoff_authority_drive_id: str
    cpu_widths: tuple[int, ...]
    deferred_gpu_widths: tuple[int, ...]
    batch_size: int
    sequence_length: int
    training_steps: int
    width_drift_limit: float
    provisional_base_shape: str
    provisional_numerics: str
    decay_implementation: str
    width_classifications: tuple[WidthClassificationReceipt, ...]
    unclassified_tensor_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    authority_verified: bool
    topology_verified: bool
    tensor_class_map_complete: bool
    mup_protocol_complete: bool
    execution_status: str
    model_constructed_for_inventory: bool
    pf3_initialization_applied: bool
    optimizer_constructed: bool
    forward_executed: bool
    training_performed: bool
    activation_rms_measured: bool
    activation_coordinate_passed: bool | None
    passed: bool
    catch_number: int
    disposition: str
    a100_hours: float = 0.0

    def __post_init__(self) -> None:
        expected_numerics = (
            f"{MUP_NUMERICS_STATUS};sigma_base={MUP_SIGMA_BASE};sigma_emb={MUP_SIGMA_EMBEDDING};"
            f"eta_base={MUP_ETA_BASE};alpha_out={MUP_ALPHA_OUTPUT};alpha_emb={MUP_ALPHA_EMBEDDING};"
            f"residual_multiplier={MUP_RESIDUAL_MULTIPLIER};wd={MUP_WEIGHT_DECAY};"
            f"betas={MUP_BETAS};eps={MUP_EPSILON}"
        )
        literal_state = (
            self.program_sha256 == PREFLIGHT_PROGRAM_SHA256
            and self.ratification_sha256 == PREFLIGHT_RATIFICATION_SHA256
            and self.build_handoff_authority == BUILD_HANDOFF_AUTHORITY
            and self.build_handoff_authority_drive_id == BUILD_HANDOFF_AUTHORITY_DRIVE_ID
            and self.cpu_widths == C1_CPU_WIDTHS
            and self.deferred_gpu_widths == C1_DEFERRED_GPU_WIDTHS
            and self.batch_size == C1_BATCH_SIZE
            and self.sequence_length == C1_SEQUENCE_LENGTH
            and self.training_steps == C1_TRAINING_STEPS
            and self.width_drift_limit == C1_WIDTH_DRIFT_LIMIT
            and self.provisional_base_shape == C1_PROVISIONAL_BASE_SHAPE
            and self.provisional_numerics == expected_numerics
            and self.decay_implementation == MUP_DECAY_IMPLEMENTATION
        )
        if not literal_state:
            raise ValueError("C1 authority or protocol literals are inconsistent")
        expected_authorities = (
            (PREFLIGHT_PROGRAM_FILE, PREFLIGHT_PROGRAM_BYTES, PREFLIGHT_PROGRAM_SHA256),
            (
                PREFLIGHT_RATIFICATION_FILE,
                PREFLIGHT_RATIFICATION_BYTES,
                PREFLIGHT_RATIFICATION_SHA256,
            ),
            (BUILD_HANDOFF_AUTHORITY_FILE, BUILD_HANDOFF_AUTHORITY_BYTES, BUILD_HANDOFF_AUTHORITY_SHA256),
            (PF2_AUTHORITY_FILE, PF2_AUTHORITY_BYTES, PF2_AUTHORITY_SHA256),
            (S81_AUTHORITY_FILE, S81_AUTHORITY_BYTES, S81_AUTHORITY_SHA256),
            (PF3_AUTHORITY_FILE, PF3_AUTHORITY_BYTES, PF3_AUTHORITY_SHA256),
        )
        observed_authorities = tuple(
            (item.filename, item.expected_bytes, item.expected_sha256)
            for item in self.authorities
        )
        if observed_authorities != expected_authorities:
            raise ValueError("C1 authority identities are inconsistent")
        authority_verified = bool(self.authorities) and all(item.verified for item in self.authorities)
        if self.authority_verified is not authority_verified:
            raise ValueError("C1 authority state is inconsistent")
        topology_verified = (
            tuple(row.width for row in self.width_classifications) == self.cpu_widths == C1_CPU_WIDTHS
            and all(row.topology.is_bound() for row in self.width_classifications)
        )
        if self.topology_verified is not topology_verified:
            raise ValueError("C1 topology state is inconsistent")
        for row in self.width_classifications:
            row.__post_init__()
        complete = all(not row.unclassified_tensors for row in self.width_classifications)
        if self.tensor_class_map_complete is not complete or self.mup_protocol_complete is not complete:
            raise ValueError("C1 tensor-class completion state is inconsistent")
        derived_shapes = tuple(
            (issue.canonical_name, issue.shape)
            for row in self.width_classifications
            for issue in row.unclassified_tensors
        )
        if self.unclassified_tensor_shapes != derived_shapes:
            raise ValueError("C1 unclassified tensor inventory is inconsistent")
        expected = (
            ("front_hadamard.router.weight", (8, 128)),
            ("front_hadamard.router.weight", (8, 256)),
            ("front_hadamard.router.weight", (8, 512)),
        )
        if derived_shapes != expected:
            raise ValueError("C1 Catch #35 evidence differs from the exact observed router shapes")
        blocked = (
            not complete
            and self.execution_status == C1_EXECUTION_STATUS
            and self.model_constructed_for_inventory
            and not self.pf3_initialization_applied
            and not self.optimizer_constructed
            and not self.forward_executed
            and not self.training_performed
            and not self.activation_rms_measured
            and self.activation_coordinate_passed is None
            and not self.passed
            and self.catch_number == C1_CATCH_NUMBER
            and self.disposition == C1_CATCH_DISPOSITION
            and self.a100_hours == 0.0
        )
        if not blocked:
            raise ValueError("C1 fail-closed Catch #35 state is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def require_passed(self) -> None:
        self.__post_init__()
        raise C1CoordinateCatch(
            "CATCH #35: front_hadamard.router.weight has shape E x d with "
            "scaling fan-in and fixed fan-out; PF-3.1 assigns no muP class"
        )


def _verify_authority_file(docs_dir: Path, *, filename: str, expected_bytes: int, expected_sha256: str) -> AuthorityVerification:
    path = docs_dir / filename
    try:
        payload = path.read_bytes()
    except FileNotFoundError as error:
        raise C1AuthorityIntegrityError(f"missing C1 authority: {path}") from error
    actual_sha256 = sha256(payload).hexdigest()
    verified = len(payload) == expected_bytes and actual_sha256 == expected_sha256
    receipt = AuthorityVerification(filename, expected_bytes, len(payload), expected_sha256, actual_sha256, verified)
    if not verified:
        raise C1AuthorityIntegrityError(
            f"C1 authority mismatch for {filename}: bytes={len(payload)}/{expected_bytes}, sha256={actual_sha256}/{expected_sha256}"
        )
    return receipt


def verify_c1_authorities(docs_dir: Path | None = None) -> tuple[AuthorityVerification, ...]:
    authority_dir = Path(__file__).resolve().parents[1] / "docs" if docs_dir is None else Path(docs_dir)
    specs = (
        (PREFLIGHT_PROGRAM_FILE, PREFLIGHT_PROGRAM_BYTES, PREFLIGHT_PROGRAM_SHA256),
        (
            PREFLIGHT_RATIFICATION_FILE,
            PREFLIGHT_RATIFICATION_BYTES,
            PREFLIGHT_RATIFICATION_SHA256,
        ),
        (BUILD_HANDOFF_AUTHORITY_FILE, BUILD_HANDOFF_AUTHORITY_BYTES, BUILD_HANDOFF_AUTHORITY_SHA256),
        (PF2_AUTHORITY_FILE, PF2_AUTHORITY_BYTES, PF2_AUTHORITY_SHA256),
        (S81_AUTHORITY_FILE, S81_AUTHORITY_BYTES, S81_AUTHORITY_SHA256),
        (PF3_AUTHORITY_FILE, PF3_AUTHORITY_BYTES, PF3_AUTHORITY_SHA256),
    )
    return tuple(_verify_authority_file(authority_dir, filename=name, expected_bytes=size, expected_sha256=digest) for name, size, digest in specs)


def _topology(width: int) -> C1WidthTopology:
    result = C1WidthTopology(width, 64, width // 64, width // 128, 11 * width // 4, 2, width // 4, 4, 2, 4, 4, math.sqrt(MUP_D_HEAD_BASE) / 64)
    if not result.is_bound():
        raise ValueError(f"width {width} does not realize the PF-3.1 topology")
    return result


def _config(width: int) -> AblationLMConfig:
    topology = _topology(width)
    return AblationLMConfig(
        vocab_size=MUP_BASE_VOCAB_SIZE,
        d_model=width,
        n_heads=topology.q_heads,
        n_kv_heads=topology.kv_heads,
        d_ff=topology.d_ff,
        n_prelude_layers=4,
        n_core_blocks=2,
        n_coda_layers=4,
        use_recurrence=True,
        recurrent_steps=4,
        max_recurrent_steps=8,
        use_static_kv_core=True,
        max_sequence_length=C1_SEQUENCE_LENGTH,
        use_front_hadamard_experts=True,
        hadamard_experts=8,
        use_reentry_bridge=True,
        use_scratch=True,
        use_lane_carrier=True,
        scratch_width=width // 4,
        use_engram=True,
        use_long_term_memory=True,
        long_term_memory_width=width // 4,
        initialization_seed=C1_SEED,
        run_seed=C1_SEED,
        hadamard_seed=C1_SEED,
        engram_hash_seed=C1_SEED,
        jet_plane_probe_seed=C1_SEED,
    )


def _memory(config: AblationLMConfig) -> ReadOnlyLatentMemory:
    def draw(key: str) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(derive_module_seed(C1_SEED, key, 0))
        return torch.randn(config.long_term_memory_slots, config.long_term_memory_width, generator=generator)
    return ReadOnlyLatentMemory(
        config.d_model,
        keys=draw("preflight.c1.memory.keys"),
        values=draw("preflight.c1.memory.values"),
        provenance_ids=torch.arange(config.long_term_memory_slots),
        layer_scale=config.long_term_memory_layer_scale,
        norm_eps=config.norm_eps,
        initialization_seed=config.initialization_seed,
    )


def _map_sha(assignments: tuple[MuPParameterAssignment, ...]) -> str:
    payload = [
        {**asdict(item), "parameter_class": item.parameter_class.value}
        for item in assignments
    ]
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _audit_width(width: int) -> WidthClassificationReceipt:
    config = _config(width)
    model = AblationLM(config, long_term_memory=_memory(config))
    audit = audit_mup_parameters(model, width=width)
    unique = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    if len(audit.assignments) + len(audit.issues) != unique:
        raise RuntimeError("C1 audit did not account for every unique trainable tensor")
    return WidthClassificationReceipt(
        width=width,
        width_multiplier=audit.width_multiplier,
        topology=_topology(width),
        unique_trainable_tensors=unique,
        classified_tensors=audit.assignments,
        unclassified_tensors=audit.issues,
        classified_map_sha256=_map_sha(audit.assignments),
    )


def run_preflight_c1(*, widths: tuple[int, ...] = C1_CPU_WIDTHS, training_steps: int = C1_TRAINING_STEPS) -> C1PreflightReceipt:
    if widths != C1_CPU_WIDTHS:
        raise ValueError("widths must equal the complete PF-3.1 CPU axis")
    if type(training_steps) is not int or training_steps != C1_TRAINING_STEPS:
        raise ValueError("training_steps must equal the PF-3.1 literal 10")
    authorities = verify_c1_authorities()
    classifications = tuple(_audit_width(width) for width in widths)
    issues = tuple(
        (issue.canonical_name, issue.shape)
        for row in classifications
        for issue in row.unclassified_tensors
    )
    return C1PreflightReceipt(
        program_sha256=PREFLIGHT_PROGRAM_SHA256,
        ratification_sha256=PREFLIGHT_RATIFICATION_SHA256,
        authorities=authorities,
        build_handoff_authority=BUILD_HANDOFF_AUTHORITY,
        build_handoff_authority_drive_id=BUILD_HANDOFF_AUTHORITY_DRIVE_ID,
        cpu_widths=widths,
        deferred_gpu_widths=C1_DEFERRED_GPU_WIDTHS,
        batch_size=C1_BATCH_SIZE,
        sequence_length=C1_SEQUENCE_LENGTH,
        training_steps=training_steps,
        width_drift_limit=C1_WIDTH_DRIFT_LIMIT,
        provisional_base_shape=C1_PROVISIONAL_BASE_SHAPE,
        provisional_numerics=(
            f"{MUP_NUMERICS_STATUS};sigma_base={MUP_SIGMA_BASE};sigma_emb={MUP_SIGMA_EMBEDDING};"
            f"eta_base={MUP_ETA_BASE};alpha_out={MUP_ALPHA_OUTPUT};alpha_emb={MUP_ALPHA_EMBEDDING};"
            f"residual_multiplier={MUP_RESIDUAL_MULTIPLIER};wd={MUP_WEIGHT_DECAY};"
            f"betas={MUP_BETAS};eps={MUP_EPSILON}"
        ),
        decay_implementation=MUP_DECAY_IMPLEMENTATION,
        width_classifications=classifications,
        unclassified_tensor_shapes=issues,
        authority_verified=True,
        topology_verified=True,
        tensor_class_map_complete=False,
        mup_protocol_complete=False,
        execution_status=C1_EXECUTION_STATUS,
        model_constructed_for_inventory=True,
        pf3_initialization_applied=False,
        optimizer_constructed=False,
        forward_executed=False,
        training_performed=False,
        activation_rms_measured=False,
        activation_coordinate_passed=None,
        passed=False,
        catch_number=C1_CATCH_NUMBER,
        disposition=C1_CATCH_DISPOSITION,
    )


def main() -> None:
    print(json.dumps(run_preflight_c1().to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "AuthorityVerification",
    "BUILD_HANDOFF_AUTHORITY",
    "BUILD_HANDOFF_AUTHORITY_BYTES",
    "BUILD_HANDOFF_AUTHORITY_FILE",
    "BUILD_HANDOFF_AUTHORITY_DRIVE_ID",
    "BUILD_HANDOFF_AUTHORITY_SHA256",
    "C1AuthorityIntegrityError",
    "C1CoordinateCatch",
    "C1PreflightReceipt",
    "C1WidthTopology",
    "C1_BATCH_SIZE",
    "C1_CATCH_NUMBER",
    "C1_CPU_WIDTHS",
    "C1_DEFERRED_GPU_WIDTHS",
    "C1_EXECUTION_STATUS",
    "C1_HEAD_DIM",
    "C1_SEQUENCE_LENGTH",
    "C1_TRAINING_STEPS",
    "C1_WIDTH_DRIFT_LIMIT",
    "PF2_AUTHORITY_BYTES",
    "PF2_AUTHORITY_FILE",
    "PF2_AUTHORITY_SHA256",
    "PF3_AUTHORITY_BYTES",
    "PF3_AUTHORITY_FILE",
    "PF3_AUTHORITY_SHA256",
    "PREFLIGHT_PROGRAM_BYTES",
    "PREFLIGHT_PROGRAM_FILE",
    "PREFLIGHT_PROGRAM_SHA256",
    "PREFLIGHT_RATIFICATION_BYTES",
    "PREFLIGHT_RATIFICATION_FILE",
    "PREFLIGHT_RATIFICATION_SHA256",
    "S81_AUTHORITY_BYTES",
    "S81_AUTHORITY_FILE",
    "S81_AUTHORITY_SHA256",
    "S81_FUTURE_HEAD_DIM_POLICY",
    "WidthClassificationReceipt",
    "run_preflight_c1",
    "verify_c1_authorities",
]
