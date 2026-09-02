"""Fail-closed PF-2.1 binding gate for the WEFT-1 C1 width check.

PF-2.1 replaces the original discretionary C1 diagnostic with a fixed width
axis and chassis. It also requires ten AdamW steps to use exactly the muP
initialization, multiplier, and per-tensor learning-rate rules in build-handoff
section 8. Section 8 does not bind that complete protocol. Consequently this
module verifies the two local amendment authorities and the bound topology,
then returns Catch #33 before any model, optimizer, batch, or activation is
materialized. It deliberately contains no fallback initializer or optimizer
recipe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

from models.ablation_lm.config import MUP_D_HEAD_BASE


PREFLIGHT_PROGRAM_SHA256 = (
    "ceaa5338830307d3783296b8a4aef7bb87962eb35535d392f4c6d217dff88a5b"
)
PREFLIGHT_RATIFICATION_SHA256 = (
    "4a13054d38c68e5e9476330528649d445ff845e639e0a36bb01641b54ef66965"
)

PF2_AUTHORITY_FILE = "STRATEGY_PREFLIGHT_AMENDMENT_PF2_20260902.md"
PF2_AUTHORITY_BYTES = 13_097
PF2_AUTHORITY_SHA256 = (
    "be11390c28ae36210a1571f7c6d358ee54e977d239f5344f7e6402212448eb05"
)
S81_AUTHORITY_FILE = "STRATEGY_HANDOFF_S81_AMENDMENT_20260902.md"
S81_AUTHORITY_BYTES = 3_403
S81_AUTHORITY_SHA256 = (
    "dd79aaa6fd9bab15bf02aaef28f99f47c745d63eed6db2c9b929b4bb1cfbb418"
)
BUILD_HANDOFF_AUTHORITY = "STRATEGY_TO_CODING_AGENT_LOOM1_HANDOFF_20260826.md#8.1"
BUILD_HANDOFF_AUTHORITY_FILE = "STRATEGY_TO_CODING_AGENT_LOOM1_HANDOFF_20260826.md"
BUILD_HANDOFF_AUTHORITY_BYTES = 61_329
BUILD_HANDOFF_AUTHORITY_SHA256 = (
    "498f34b5966f0879c7f0a15ca8be02a603558781c35f59f03fb29cc9edd3eb02"
)
BUILD_HANDOFF_AUTHORITY_DRIVE_ID = "1XaE81mfqTOYEYGFMa-ZJwpLW-KQtMMC_"

C1_CPU_WIDTHS = (128, 256, 512)
C1_DEFERRED_GPU_WIDTHS = (1_024,)
C1_HEAD_DIM = 64
C1_BATCH_SIZE = 2
C1_SEQUENCE_LENGTH = 64
C1_TRAINING_STEPS = 10
C1_WIDTH_DRIFT_LIMIT = 2.0
C1_CATCH_NUMBER = 33
C1_EXECUTION_STATUS = "blocked_before_model_initialization"
S81_FUTURE_HEAD_DIM_POLICY = (
    "future_WEFT_d_head_not_64_requires_explicit_base_shape_implementation"
)


class C1CoordinateCatch(RuntimeError):
    """Raised when a caller attempts to promote a blocked C1 receipt."""


class C1AuthorityIntegrityError(RuntimeError):
    """Raised when a local PF-2 authority is absent or is not byte-exact."""


@dataclass(frozen=True)
class AuthorityVerification:
    filename: str
    expected_bytes: int
    actual_bytes: int
    expected_sha256: str
    actual_sha256: str
    verified: bool


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
        return (
            self.n_prelude_layers
            + self.recurrent_steps * self.n_core_blocks
            + self.n_coda_layers
        )

    def is_pf2_bound(self) -> bool:
        return (
            self.width in C1_CPU_WIDTHS
            and self.head_dim == C1_HEAD_DIM == MUP_D_HEAD_BASE
            and self.q_heads == self.width // 64
            and self.kv_heads == self.width // 128
            and self.q_heads == 2 * self.kv_heads
            and self.d_ff == 11 * self.width // 4
            and self.scratch_lanes == 2
            and self.scratch_width_per_lane == self.width // 4
            and (self.n_prelude_layers, self.n_core_blocks, self.n_coda_layers)
            == (4, 2, 4)
            and self.recurrent_steps == 4
            and self.attention_logit_scale
            == math.sqrt(MUP_D_HEAD_BASE) / self.head_dim
        )


@dataclass(frozen=True)
class MUPBinding:
    component: str
    status: str
    authority_rule: str
    missing_requirement: str | None


@dataclass(frozen=True)
class C1PreflightReceipt:
    program_sha256: str
    ratification_sha256: str
    pf2_authority: AuthorityVerification
    s81_authority: AuthorityVerification
    build_handoff_verification: AuthorityVerification
    build_handoff_authority: str
    build_handoff_authority_bytes: int
    build_handoff_authority_sha256: str
    build_handoff_authority_drive_id: str
    cpu_widths: tuple[int, ...]
    deferred_gpu_widths: tuple[int, ...]
    batch_size: int
    sequence_length: int
    training_steps: int
    width_drift_limit: float
    width_topologies: tuple[C1WidthTopology, ...]
    mup_bindings: tuple[MUPBinding, ...]
    unbound_mup_components: tuple[str, ...]
    authority_verified: bool
    topology_verified: bool
    attention_contract_bound: bool
    future_head_dim_policy: str
    mup_protocol_complete: bool
    execution_status: str
    model_initialized: bool
    optimizer_constructed: bool
    training_performed: bool
    activation_coordinate_passed: bool | None
    passed: bool
    catch_number: int
    disposition: str
    a100_hours: float = 0.0

    def __post_init__(self) -> None:
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        verified = (
            self.pf2_authority.verified
            and self.s81_authority.verified
            and self.build_handoff_verification.verified
        )
        if self.authority_verified is not verified:
            raise ValueError("C1 authority_verified is inconsistent with byte verification")
        if (
            self.build_handoff_verification.filename != BUILD_HANDOFF_AUTHORITY_FILE
            or self.build_handoff_verification.expected_bytes
            != self.build_handoff_authority_bytes
            or self.build_handoff_verification.expected_sha256
            != self.build_handoff_authority_sha256
        ):
            raise ValueError("C1 build-handoff verification metadata is inconsistent")

        topology_verified = (
            self.cpu_widths == C1_CPU_WIDTHS
            and len(self.width_topologies) == len(C1_CPU_WIDTHS)
            and tuple(item.width for item in self.width_topologies) == C1_CPU_WIDTHS
            and all(item.is_pf2_bound() for item in self.width_topologies)
        )
        if self.topology_verified is not topology_verified:
            raise ValueError("C1 topology_verified is inconsistent with the bound topology")
        attention_bound = (
            MUP_D_HEAD_BASE == C1_HEAD_DIM
            and all(item.attention_logit_scale == 0.125 for item in self.width_topologies)
        )
        if self.attention_contract_bound is not attention_bound:
            raise ValueError("C1 attention_contract_bound is inconsistent")

        derived_unbound = tuple(
            binding.component for binding in self.mup_bindings if binding.status != "bound"
        )
        if self.unbound_mup_components != derived_unbound:
            raise ValueError("C1 unbound muP component list is inconsistent")
        protocol_complete = not derived_unbound
        if self.mup_protocol_complete is not protocol_complete:
            raise ValueError("C1 mup_protocol_complete is inconsistent")

        if not protocol_complete:
            blocked_state = (
                self.execution_status == C1_EXECUTION_STATUS
                and not self.model_initialized
                and not self.optimizer_constructed
                and not self.training_performed
                and self.activation_coordinate_passed is None
                and not self.passed
                and self.catch_number == C1_CATCH_NUMBER
                and self.disposition
                == "catch_33_return_to_strategy_mup_protocol_unbound"
                and self.a100_hours == 0.0
            )
            if not blocked_state:
                raise ValueError("C1 blocked Catch #33 state is inconsistent")
        elif self.passed and not (
            self.model_initialized
            and self.optimizer_constructed
            and self.training_performed
            and self.activation_coordinate_passed is True
        ):
            raise ValueError("C1 cannot pass without a complete executed width check")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def require_passed(self) -> None:
        self._validate_invariants()
        if self.passed:
            return
        missing = ", ".join(self.unbound_mup_components)
        raise C1CoordinateCatch(
            f"CATCH #{self.catch_number}: PF-2.1 stopped before model "
            f"initialization; unbound section 8 muP components: {missing}"
        )


def _verify_authority_file(
    docs_dir: Path,
    *,
    filename: str,
    expected_bytes: int,
    expected_sha256: str,
) -> AuthorityVerification:
    path = docs_dir / filename
    try:
        payload = path.read_bytes()
    except FileNotFoundError as error:
        raise C1AuthorityIntegrityError(f"missing C1 authority: {path}") from error
    actual_sha256 = sha256(payload).hexdigest()
    verification = AuthorityVerification(
        filename=filename,
        expected_bytes=expected_bytes,
        actual_bytes=len(payload),
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        verified=(
            len(payload) == expected_bytes
            and actual_sha256 == expected_sha256
        ),
    )
    if not verification.verified:
        raise C1AuthorityIntegrityError(
            f"C1 authority mismatch for {filename}: "
            f"bytes={verification.actual_bytes}/{expected_bytes}, "
            f"sha256={actual_sha256}/{expected_sha256}"
        )
    return verification


def verify_c1_authorities(
    docs_dir: Path | None = None,
) -> tuple[AuthorityVerification, AuthorityVerification, AuthorityVerification]:
    """Verify the byte-exact PF-2 and S81 amendments before interpreting them."""

    authority_dir = (
        Path(__file__).resolve().parents[1] / "docs"
        if docs_dir is None
        else Path(docs_dir)
    )
    pf2 = _verify_authority_file(
        authority_dir,
        filename=PF2_AUTHORITY_FILE,
        expected_bytes=PF2_AUTHORITY_BYTES,
        expected_sha256=PF2_AUTHORITY_SHA256,
    )
    s81 = _verify_authority_file(
        authority_dir,
        filename=S81_AUTHORITY_FILE,
        expected_bytes=S81_AUTHORITY_BYTES,
        expected_sha256=S81_AUTHORITY_SHA256,
    )
    build_handoff = _verify_authority_file(
        authority_dir,
        filename=BUILD_HANDOFF_AUTHORITY_FILE,
        expected_bytes=BUILD_HANDOFF_AUTHORITY_BYTES,
        expected_sha256=BUILD_HANDOFF_AUTHORITY_SHA256,
    )
    return pf2, s81, build_handoff


def _bound_width_topology(width: int) -> C1WidthTopology:
    if type(width) is not int or width not in C1_CPU_WIDTHS:
        raise ValueError("width must be one of the PF-2.1 CPU widths")
    topology = C1WidthTopology(
        width=width,
        head_dim=C1_HEAD_DIM,
        q_heads=width // 64,
        kv_heads=width // 128,
        d_ff=11 * width // 4,
        scratch_lanes=2,
        scratch_width_per_lane=width // 4,
        n_prelude_layers=4,
        n_core_blocks=2,
        n_coda_layers=4,
        recurrent_steps=4,
        attention_logit_scale=math.sqrt(MUP_D_HEAD_BASE) / C1_HEAD_DIM,
    )
    if not topology.is_pf2_bound():
        raise RuntimeError(f"PF-2.1 topology construction failed at d={width}")
    return topology


def _section8_mup_bindings() -> tuple[MUPBinding, ...]:
    """Return only literal section 8/S81 bindings; never fill a missing value locally."""

    return (
        MUPBinding(
            component="attention_logit_scale",
            status="bound",
            authority_rule="sqrt(d_head_base) / d_head with d_head_base=64",
            missing_requirement=None,
        ),
        MUPBinding(
            component="model_base_width_d_base",
            status="unbound",
            authority_rule="m_width = d_model / d_base",
            missing_requirement="numeric d_base",
        ),
        MUPBinding(
            component="internal_base_initialization_sigma_base",
            status="unbound",
            authority_rule="sigma_internal = sigma_base / sqrt(m_width)",
            missing_requirement="numeric sigma_base",
        ),
        MUPBinding(
            component="complete_per_tensor_initialization_map",
            status="unbound",
            authority_rule="verify input/output taxonomy; do not assume",
            missing_requirement=(
                "exhaustive tensor taxonomy and initialization rule for every "
                "trainable tensor"
            ),
        ),
        MUPBinding(
            component="internal_base_learning_rate_eta_base",
            status="unbound",
            authority_rule="eta_internal = eta_base / m_width",
            missing_requirement="numeric eta_base",
        ),
        MUPBinding(
            component="complete_per_tensor_learning_rate_map",
            status="unbound",
            authority_rule="per-tensor learning-rate rules exactly as section 8",
            missing_requirement=(
                "exhaustive tensor-to-learning-rate or multiplier mapping"
            ),
        ),
        MUPBinding(
            component="residual_branch_alpha",
            status="unbound",
            authority_rule="h <- h + alpha F_theta(h); alpha found on proxy",
            missing_requirement="numeric alpha or a ratified selection receipt",
        ),
        MUPBinding(
            component="embedding_multiplier",
            status="unbound",
            authority_rule="embedding multiplier is first-class",
            missing_requirement="numeric embedding multiplier",
        ),
        MUPBinding(
            component="residual_multiplier",
            status="unbound",
            authority_rule="residual multiplier is first-class",
            missing_requirement="numeric residual multiplier",
        ),
    )


def run_preflight_c1(
    *,
    widths: tuple[int, ...] = C1_CPU_WIDTHS,
    training_steps: int = C1_TRAINING_STEPS,
) -> C1PreflightReceipt:
    """Audit PF-2.1 bindings and stop before executing an underbound C1 run."""

    if widths != C1_CPU_WIDTHS:
        raise ValueError("widths must equal the complete PF-2.1 CPU axis (128,256,512)")
    if type(training_steps) is not int or training_steps != C1_TRAINING_STEPS:
        raise ValueError("training_steps must equal the PF-2.1 literal 10")

    pf2, s81, build_handoff = verify_c1_authorities()
    topologies = tuple(_bound_width_topology(width) for width in widths)
    bindings = _section8_mup_bindings()
    unbound = tuple(
        binding.component for binding in bindings if binding.status != "bound"
    )
    if not unbound:
        raise RuntimeError(
            "PF-2.1 binding audit unexpectedly became complete; implement and "
            "review the model/optimizer execution path before running it"
        )

    return C1PreflightReceipt(
        program_sha256=PREFLIGHT_PROGRAM_SHA256,
        ratification_sha256=PREFLIGHT_RATIFICATION_SHA256,
        pf2_authority=pf2,
        s81_authority=s81,
        build_handoff_verification=build_handoff,
        build_handoff_authority=BUILD_HANDOFF_AUTHORITY,
        build_handoff_authority_bytes=BUILD_HANDOFF_AUTHORITY_BYTES,
        build_handoff_authority_sha256=BUILD_HANDOFF_AUTHORITY_SHA256,
        build_handoff_authority_drive_id=BUILD_HANDOFF_AUTHORITY_DRIVE_ID,
        cpu_widths=widths,
        deferred_gpu_widths=C1_DEFERRED_GPU_WIDTHS,
        batch_size=C1_BATCH_SIZE,
        sequence_length=C1_SEQUENCE_LENGTH,
        training_steps=training_steps,
        width_drift_limit=C1_WIDTH_DRIFT_LIMIT,
        width_topologies=topologies,
        mup_bindings=bindings,
        unbound_mup_components=unbound,
        authority_verified=pf2.verified and s81.verified and build_handoff.verified,
        topology_verified=all(topology.is_pf2_bound() for topology in topologies),
        attention_contract_bound=(
            MUP_D_HEAD_BASE == C1_HEAD_DIM
            and all(
                topology.attention_logit_scale == 0.125
                for topology in topologies
            )
        ),
        future_head_dim_policy=S81_FUTURE_HEAD_DIM_POLICY,
        mup_protocol_complete=False,
        execution_status=C1_EXECUTION_STATUS,
        model_initialized=False,
        optimizer_constructed=False,
        training_performed=False,
        activation_coordinate_passed=None,
        passed=False,
        catch_number=C1_CATCH_NUMBER,
        disposition="catch_33_return_to_strategy_mup_protocol_unbound",
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
    "MUPBinding",
    "PF2_AUTHORITY_BYTES",
    "PF2_AUTHORITY_FILE",
    "PF2_AUTHORITY_SHA256",
    "PREFLIGHT_PROGRAM_SHA256",
    "PREFLIGHT_RATIFICATION_SHA256",
    "S81_AUTHORITY_BYTES",
    "S81_AUTHORITY_FILE",
    "S81_AUTHORITY_SHA256",
    "S81_FUTURE_HEAD_DIM_POLICY",
    "run_preflight_c1",
    "verify_c1_authorities",
]
