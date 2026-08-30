"""Forward-only execution and selection contracts for the WEFT-1 G-TOK screen.

This module does not construct a model, optimizer, trainer, or checkpoint.  It
validates already-produced evidence under the corpus handoff as amended by
A1/A2/A3 and the release-close record.  The legacy G-TOK contract remains
unchanged; every receipt here uses a new authority chain and schema domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import NoReturn

from training.weft1_gtok_a1_contract import ScreenCorpusReceiptV2
from training.weft1_gtok_contract import (
    FlatAdamWRecipe,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V2,
    GTOK_HELDOUT_BYTE_TARGET,
    GTOK_PROXY_TOPOLOGY_SHA256,
    GTOK_SEED_COUNT,
    GTOK_STRATA,
    GTOK_TRAINING_BYTE_BUDGET,
    GTOK_VOCABULARY_ARMS,
    StratumNllReceipt,
    a1_flat_adamw_recipe,
    canonical_json_bytes,
    canonical_sha256,
)


GTOK_AMENDMENT_A2_SHA256 = (
    "f7a2655b30f6c699035ec4ffdccee8c03068eeab8da94894be8e5818f955ce02"
)
GTOK_AMENDMENT_A3_SHA256 = (
    "4e7b18ec676c6d613c7a0f85ece4c7b8fcc1daab48d5ce0b8cd11bc06875b6c0"
)
GTOK_RELEASE_CLOSE_SHA256 = (
    "d8c4f3bf8829bbe48e2464bf758ec3594ef730a0f952712099b45d183ca2ab3e"
)
GTOK_A2_BINDINGS_SHA256 = (
    "ee10e69a3ccd55f7960949f4c318daa4db1197c779f5e88fb67cec82ab7f263b"
)
GTOK_V2_AUTHORITY_CHAIN = (
    *GTOK_EXECUTION_AUTHORITY_CHAIN_V2,
    GTOK_AMENDMENT_A2_SHA256,
    GTOK_AMENDMENT_A3_SHA256,
    GTOK_RELEASE_CLOSE_SHA256,
    GTOK_A2_BINDINGS_SHA256,
)

GTOK_FIRST_BOUNDARY_BYTES = 1_000_000_000
GTOK_SECOND_BOUNDARY_BYTES = 2_000_000_000
GTOK_MILESTONE_LABELS = ("after_1b", "after_2b", "terminal_realized_T")
GTOK_TERMINAL_METRIC = "terminal_pooled_bpb"
GTOK_TERMINAL_BUDGET = "terminal_realized_T"
GTOK_VOCABULARY_FRACTION_CAP = Fraction(1, 5)
GTOK_TRIPWIRE_A100_MICROSECONDS = 12 * 60 * 60 * 1_000_000
GTOK_CALIBRATION_MAX_STEPS = 100
GTOK_PER_RUN_WATCHDOG_MULTIPLIER = 2
GTOK_COMPUTE_SCOPES = ("base_screen", "confirmation")
GTOK_ATTEMPT_STATUSES = (
    "aborted_watchdog",
    "cancelled",
    "completed",
    "failed",
    "pending",
    "preempted",
    "running",
)

GTOK_SELECTOR_LITERAL_BINDING_V2 = {
    "admissibility_cap": GTOK_VOCABULARY_FRACTION_CAP,
    "comparison_budget": GTOK_TERMINAL_BUDGET,
    "comparison_metric": GTOK_TERMINAL_METRIC,
    "confirmation_pair": "first_two_arms_in_agreed_strict_terminal_order",
    "displacement": "challenger_delta_strictly_greater_than_3_s_hat",
    "pairwise_s_hat": "sqrt(((n_a-1)*s_a^2+(n_b-1)*s_b^2)/(n_a+n_b-2))",
    "scan": "smallest_admissible_then_larger_vocabularies_ascending",
    "seed_rule": "identical_strict_seed_specific_total_orders_or_stop",
    "tie_diagnostic": "abs(delta)_strictly_less_than_2_s_hat",
}
GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2 = canonical_sha256(
    GTOK_SELECTOR_LITERAL_BINDING_V2
)
_A1_OPTIMIZER = a1_flat_adamw_recipe()
_A1_OPTIMIZER_BYTES = canonical_json_bytes(_A1_OPTIMIZER)
_SHA256_CHARS = frozenset("0123456789abcdef")
_FACTORY_SENTINEL = object()


class GTokV2Stop(RuntimeError):
    """A registered G-TOK stop condition fired."""


def _require_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _require_exact_int(value: int, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _require_finite(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def gtok_v2_bound_sha256(schema: str, value: object) -> str:
    """Hash an additive v2 receipt without re-keying legacy receipts."""

    if not isinstance(schema, str) or not schema.startswith("weft1_gtok_v2_"):
        raise ValueError("G-TOK v2 schemas require the weft1_gtok_v2_ prefix")
    return canonical_sha256(
        {
            "authority_chain": GTOK_V2_AUTHORITY_CHAIN,
            "payload": value,
            "schema": schema,
            "selector_literal_binding_sha256": (
                GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2
            ),
        }
    )


@dataclass(frozen=True)
class FrozenScreenCorpusV2:
    """P-B-frozen parent corpus plus A1's independent T/H floors."""

    full_corpus_manifest_sha256: str
    screen_submanifest_sha256: str
    corpus_freeze_receipt_sha256: str
    d1_d6_gate_bundle_sha256: str
    decontamination_receipt_sha256: str
    floors: ScreenCorpusReceiptV2

    def __post_init__(self) -> None:
        for name in (
            "full_corpus_manifest_sha256",
            "screen_submanifest_sha256",
            "corpus_freeze_receipt_sha256",
            "d1_d6_gate_bundle_sha256",
            "decontamination_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.floors, ScreenCorpusReceiptV2):
            raise TypeError("floors must be an A1 ScreenCorpusReceiptV2")
        if self.floors.training_target_bytes != GTOK_TRAINING_BYTE_BUDGET:
            raise ValueError("T target must remain exactly 4,000,000,000 bytes")
        if self.floors.heldout_target_bytes != GTOK_HELDOUT_BYTE_TARGET:
            raise ValueError("H target must remain exactly 80,000,000 bytes")

    @property
    def training_target_bytes(self) -> int:
        return self.floors.training_target_bytes

    @property
    def training_realized_bytes(self) -> int:
        return self.floors.training_realized_bytes

    @property
    def heldout_target_bytes(self) -> int:
        return self.floors.heldout_target_bytes

    @property
    def heldout_realized_bytes(self) -> int:
        return self.floors.heldout_realized_bytes

    @property
    def training_stream_sha256(self) -> str:
        return self.floors.training_stream_sha256

    @property
    def heldout_stream_sha256(self) -> str:
        return self.floors.heldout_stream_sha256

    @property
    def heldout_denominator_signature(self) -> tuple[tuple[str, int], ...]:
        return tuple((row.stratum, row.realized_bytes) for row in self.floors.heldout)

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_frozen_screen_corpus", self)


@dataclass(frozen=True)
class TokenizerArmReceiptV2:
    """One deterministic tokenizer identity and its append-only basis inputs."""

    vocab_size: int
    tokenizer_json_sha256: str
    merges_sha256: str
    token_inventory_sha256: str
    reserved_inventory_sha256: str
    pretokenizer_regex_sha256: str
    fit_stream_sha256: str
    full_corpus_manifest_sha256: str
    double_fit_receipt_sha256: str
    byte_round_trip_receipt_sha256: str
    tokenizer_version: str = "0.22.2"
    token_inventory_count: int = 0
    byte_atom_count: int = 256
    reachable_unk: bool = False
    stochastic_segmentation: bool = False
    irreversible_normalization: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("tokenizer uses an unregistered G-TOK vocabulary arm")
        for name in (
            "tokenizer_json_sha256",
            "merges_sha256",
            "token_inventory_sha256",
            "reserved_inventory_sha256",
            "pretokenizer_regex_sha256",
            "fit_stream_sha256",
            "full_corpus_manifest_sha256",
            "double_fit_receipt_sha256",
            "byte_round_trip_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.tokenizer_version != "0.22.2":
            raise ValueError("G-TOK v2 requires tokenizers==0.22.2")
        if (
            type(self.token_inventory_count) is not int
            or self.token_inventory_count != self.vocab_size
        ):
            raise ValueError("token inventory count must equal vocabulary size")
        if type(self.byte_atom_count) is not int or self.byte_atom_count != 256:
            raise ValueError("byte-level BPE requires all 256 byte atoms")
        for name in (
            "reachable_unk",
            "stochastic_segmentation",
            "irreversible_normalization",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.reachable_unk:
            raise ValueError("the tokenizer may not expose a reachable UNK path")
        if self.stochastic_segmentation:
            raise ValueError("stochastic segmentation is prohibited")
        if self.irreversible_normalization:
            raise ValueError("irreversible normalization is prohibited")

    @property
    def algorithm_signature(self) -> tuple[object, ...]:
        return (
            self.reserved_inventory_sha256,
            self.pretokenizer_regex_sha256,
            self.fit_stream_sha256,
            self.full_corpus_manifest_sha256,
            self.tokenizer_version,
            self.byte_atom_count,
            self.reachable_unk,
            self.stochastic_segmentation,
            self.irreversible_normalization,
        )

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_tokenizer_arm", self)


@dataclass(frozen=True)
class BpbMilestoneReceiptV2:
    """BPB at the first crossing batch or at terminal realized T."""

    label: str
    optimizer_step: int
    previous_training_raw_bytes: int
    training_raw_bytes: int
    heldout_stream_sha256: str
    strata: tuple[StratumNllReceipt, ...]

    def __post_init__(self) -> None:
        if self.label not in GTOK_MILESTONE_LABELS:
            raise ValueError("unknown G-TOK v2 milestone label")
        _require_exact_int(self.optimizer_step, "optimizer_step", minimum=1)
        _require_exact_int(
            self.previous_training_raw_bytes,
            "previous_training_raw_bytes",
        )
        _require_exact_int(self.training_raw_bytes, "training_raw_bytes", minimum=1)
        if self.previous_training_raw_bytes >= self.training_raw_bytes:
            raise ValueError("milestone bytes must advance at a batch boundary")
        threshold = {
            "after_1b": GTOK_FIRST_BOUNDARY_BYTES,
            "after_2b": GTOK_SECOND_BOUNDARY_BYTES,
        }.get(self.label)
        if threshold is not None and not (
            self.previous_training_raw_bytes < threshold <= self.training_raw_bytes
        ):
            raise ValueError(
                f"{self.label} must be the first batch boundary at or after its threshold"
            )
        _require_sha256(self.heldout_stream_sha256, "heldout_stream_sha256")
        if not isinstance(self.strata, tuple) or any(
            not isinstance(item, StratumNllReceipt) for item in self.strata
        ):
            raise TypeError("strata must contain StratumNllReceipt values")
        if tuple(item.stratum for item in self.strata) != GTOK_STRATA:
            raise ValueError("milestone requires one canonically ordered NLL row per stratum")
        if not math.isfinite(self.pooled_nll_nats):
            raise ValueError("pooled held-out NLL must remain finite")

    @property
    def pooled_nll_nats(self) -> float:
        return math.fsum(item.nll_nats for item in self.strata)

    @property
    def pooled_raw_byte_count(self) -> int:
        return sum(item.raw_byte_count for item in self.strata)

    @property
    def pooled_bpb(self) -> float:
        return self.pooled_nll_nats / (
            math.log(2.0) * self.pooled_raw_byte_count
        )

    @property
    def denominator_signature(self) -> tuple[tuple[str, int], ...]:
        return tuple((item.stratum, item.raw_byte_count) for item in self.strata)

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_bpb_milestone", self)


@dataclass(frozen=True)
class GTokRunReceiptV2:
    """One completed throwaway G-TOK arm/seed run."""

    vocab_size: int
    seed: int
    frozen_screen_corpus_sha256: str
    tokenizer_receipt_sha256: str
    initialization_recipe_sha256: str
    initialization_seed: int
    shared_initial_state_sha256: str
    data_order_seed: int
    data_order_sha256: str
    compute_attempt_id: str
    measured_a100_microseconds: int
    measured_flops: int
    optimizer: FlatAdamWRecipe
    observations: tuple[BpbMilestoneReceiptV2, ...]
    model_topology_sha256: str = GTOK_PROXY_TOPOLOGY_SHA256
    executing_block_count: int = 10
    checkpoint_retained: bool = False
    sealed_data_consumed: bool = False
    byte_round_trip_passed: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("run uses an unregistered G-TOK vocabulary arm")
        if type(self.seed) is not int:
            raise TypeError("seed must be an exact integer")
        for name in (
            "frozen_screen_corpus_sha256",
            "tokenizer_receipt_sha256",
            "initialization_recipe_sha256",
            "shared_initial_state_sha256",
            "data_order_sha256",
            "model_topology_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.model_topology_sha256 != GTOK_PROXY_TOPOLOGY_SHA256:
            raise ValueError("run topology must equal the corrected ten-block S0 receipt")
        if self.executing_block_count != 10:
            raise ValueError("G-TOK structural-OFF must execute 4+2+4 = ten blocks")
        for name in ("initialization_seed", "data_order_seed"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if not isinstance(self.compute_attempt_id, str) or not self.compute_attempt_id:
            raise ValueError("compute_attempt_id must be nonempty")
        _require_exact_int(
            self.measured_a100_microseconds,
            "measured_a100_microseconds",
            minimum=1,
        )
        _require_exact_int(self.measured_flops, "measured_flops", minimum=1)
        if not isinstance(self.optimizer, FlatAdamWRecipe):
            raise TypeError("optimizer must be a FlatAdamWRecipe")
        if canonical_json_bytes(self.optimizer) != _A1_OPTIMIZER_BYTES:
            raise ValueError("run must use the exact flat A1 AdamW recipe")
        if self.optimizer.muon_enabled:
            raise ValueError("Muon is prohibited in G-TOK")
        for name in (
            "checkpoint_retained",
            "sealed_data_consumed",
            "byte_round_trip_passed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.checkpoint_retained:
            raise ValueError("G-TOK may not retain model checkpoints")
        if self.sealed_data_consumed:
            raise ValueError("G-TOK runs may not consume sealed data")
        if not self.byte_round_trip_passed:
            raise ValueError("G-TOK requires a green exact byte round-trip")
        if not isinstance(self.observations, tuple) or tuple(
            item.label for item in self.observations
        ) != GTOK_MILESTONE_LABELS:
            raise ValueError(
                "run requires after-1B, after-2B, and terminal-realized-T observations"
            )
        if any(not isinstance(item, BpbMilestoneReceiptV2) for item in self.observations):
            raise TypeError("observations must contain BpbMilestoneReceiptV2 values")
        steps = tuple(item.optimizer_step for item in self.observations)
        if not steps[0] < steps[1] <= steps[2]:
            raise ValueError("milestone optimizer steps must preserve execution order")
        observed_bytes = tuple(item.training_raw_bytes for item in self.observations)
        if not observed_bytes[0] < observed_bytes[1] <= observed_bytes[2]:
            raise ValueError("milestone raw-byte boundaries must preserve execution order")

    @property
    def terminal(self) -> BpbMilestoneReceiptV2:
        return self.observations[-1]

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_run", self)


@dataclass(frozen=True)
class ArmCalibrationProjectionV2:
    """Measured <=100-step calibration and exact projection for one arm."""

    scope: str
    vocab_size: int
    calibration_attempt_id: str
    calibration_steps: int
    measured_tokens: int
    measured_a100_microseconds: int
    planned_tokens_per_run: int
    projected_run_a100_microseconds: int
    full_run_count: int = GTOK_SEED_COUNT
    full_run_attempt_count_at_projection: int = 0

    def __post_init__(self) -> None:
        if self.scope not in GTOK_COMPUTE_SCOPES:
            raise ValueError("calibration uses an unregistered compute scope")
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("calibration uses an unregistered vocabulary arm")
        if not isinstance(self.calibration_attempt_id, str) or not (
            self.calibration_attempt_id
        ):
            raise ValueError("calibration_attempt_id must be nonempty")
        _require_exact_int(self.calibration_steps, "calibration_steps", minimum=1)
        if self.calibration_steps > GTOK_CALIBRATION_MAX_STEPS:
            raise ValueError("an A2-R6 calibration burst may not exceed 100 steps")
        for name in (
            "measured_tokens",
            "measured_a100_microseconds",
            "planned_tokens_per_run",
            "projected_run_a100_microseconds",
        ):
            _require_exact_int(getattr(self, name), name, minimum=1)
        expected_projection = _ceil_div(
            self.measured_a100_microseconds * self.planned_tokens_per_run,
            self.measured_tokens,
        )
        if self.projected_run_a100_microseconds != expected_projection:
            raise ValueError("per-arm projection must be computed from measured tokens/sec")
        if self.full_run_count != GTOK_SEED_COUNT:
            raise ValueError("each projected arm must price the registered two-seed panel")
        if self.full_run_attempt_count_at_projection != 0:
            raise GTokV2Stop("preflight projection must precede every full-run launch")

    @property
    def projected_scope_a100_microseconds(self) -> int:
        return self.measured_a100_microseconds + (
            self.full_run_count * self.projected_run_a100_microseconds
        )

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_arm_calibration_projection", self)


@dataclass(frozen=True)
class PreflightProjectionReceiptV2:
    """Phase projection proven before any full run in that phase launches."""

    scope: str
    prior_campaign_a100_microseconds: int
    prior_event_ledger_sha256: str | None
    calibrations: tuple[ArmCalibrationProjectionV2, ...]
    projected_campaign_a100_microseconds: int
    full_run_launch_count_at_projection: int = 0
    tripwire_a100_microseconds: int = GTOK_TRIPWIRE_A100_MICROSECONDS

    def __post_init__(self) -> None:
        if self.scope not in GTOK_COMPUTE_SCOPES:
            raise ValueError("preflight uses an unregistered compute scope")
        _require_exact_int(
            self.prior_campaign_a100_microseconds,
            "prior_campaign_a100_microseconds",
        )
        if self.scope == "base_screen":
            if self.prior_campaign_a100_microseconds != 0:
                raise ValueError("base-screen preflight must begin from a zero campaign meter")
            if self.prior_event_ledger_sha256 is not None:
                raise ValueError("base-screen preflight may not name a predecessor ledger")
        else:
            if self.prior_campaign_a100_microseconds < 1:
                raise ValueError("confirmation preflight must include prior campaign compute")
            if self.prior_event_ledger_sha256 is None:
                raise ValueError("confirmation preflight must join the base event ledger")
            _require_sha256(
                self.prior_event_ledger_sha256,
                "prior_event_ledger_sha256",
            )
        if not isinstance(self.calibrations, tuple) or not self.calibrations:
            raise ValueError("preflight requires per-arm calibration receipts")
        if any(
            not isinstance(item, ArmCalibrationProjectionV2)
            for item in self.calibrations
        ):
            raise TypeError("calibrations must contain ArmCalibrationProjectionV2 values")
        if any(item.scope != self.scope for item in self.calibrations):
            raise ValueError("calibration scope differs from its preflight")
        vocabularies = tuple(item.vocab_size for item in self.calibrations)
        if vocabularies != tuple(sorted(set(vocabularies))):
            raise ValueError("preflight calibrations must use unique ascending arms")
        attempt_ids = tuple(item.calibration_attempt_id for item in self.calibrations)
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("calibration attempt IDs must be unique")
        _require_exact_int(
            self.projected_campaign_a100_microseconds,
            "projected_campaign_a100_microseconds",
            minimum=1,
        )
        expected = self.prior_campaign_a100_microseconds + sum(
            item.projected_scope_a100_microseconds for item in self.calibrations
        )
        if self.projected_campaign_a100_microseconds != expected:
            raise ValueError("campaign projection must use measured per-arm throughput")
        if self.full_run_launch_count_at_projection != 0:
            raise GTokV2Stop("projection was not completed before full-run launch")
        if self.tripwire_a100_microseconds != GTOK_TRIPWIRE_A100_MICROSECONDS:
            raise ValueError("the projected tripwire must remain 12 A100-hours")
        if self.projected_campaign_a100_microseconds > self.tripwire_a100_microseconds:
            raise GTokV2Stop(
                "projected campaign exceeds 12 A100-hours; halt before full launch"
            )

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_preflight_projection", self)


@dataclass(frozen=True)
class ComputeAttemptReceiptV2:
    """One calibration, selected run, or retry in the cumulative ledger."""

    attempt_id: str
    scope: str
    kind: str
    vocab_size: int
    seed: int | None
    consumed_a100_microseconds: int
    status: str
    calibration_projection_sha256: str
    projected_run_a100_microseconds: int
    watchdog_limit_a100_microseconds: int
    hard_abort_issued: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("compute attempt_id must be nonempty")
        if self.scope not in GTOK_COMPUTE_SCOPES:
            raise ValueError("compute attempt uses an unregistered scope")
        if self.kind not in ("calibration", "full_run"):
            raise ValueError("compute attempt kind must be calibration or full_run")
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("compute attempt uses an unregistered vocabulary")
        if self.kind == "calibration":
            if self.seed is not None:
                raise ValueError("a per-arm calibration may not bind one run seed")
        elif type(self.seed) is not int:
            raise TypeError("a full-run attempt requires an exact integer seed")
        _require_exact_int(
            self.consumed_a100_microseconds,
            "consumed_a100_microseconds",
            minimum=1,
        )
        if self.status not in GTOK_ATTEMPT_STATUSES:
            raise ValueError("compute attempt has an unregistered status")
        _require_sha256(
            self.calibration_projection_sha256,
            "calibration_projection_sha256",
        )
        _require_exact_int(
            self.projected_run_a100_microseconds,
            "projected_run_a100_microseconds",
            minimum=1,
        )
        expected_limit = (
            GTOK_PER_RUN_WATCHDOG_MULTIPLIER
            * self.projected_run_a100_microseconds
        )
        if self.watchdog_limit_a100_microseconds != expected_limit:
            raise ValueError("per-run watchdog must remain exactly 2x its arm projection")
        if type(self.hard_abort_issued) is not bool:
            raise TypeError("hard_abort_issued must be an exact bool")
        exceeded = self.consumed_a100_microseconds > expected_limit
        if self.kind == "calibration" and exceeded:
            raise ValueError("calibration compute cannot exceed its projected run watchdog")
        if self.kind == "full_run" and exceeded:
            if self.status != "aborted_watchdog" or not self.hard_abort_issued:
                raise GTokV2Stop(
                    "a run above 2x its arm projection requires a hard watchdog abort"
                )
        if self.status == "aborted_watchdog" and (
            self.kind != "full_run" or not exceeded or not self.hard_abort_issued
        ):
            raise ValueError("watchdog-aborted status requires a strict >2x hard abort")
        if self.hard_abort_issued and self.status != "aborted_watchdog":
            raise ValueError("per-run hard abort may only attest a watchdog-aborted run")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_compute_attempt", self)


@dataclass(frozen=True)
class RuntimeTripwireSnapshotV2:
    """Literal pending/running hard-abort decision at one cumulative reading."""

    event_ledger_sha256: str
    cumulative_a100_microseconds: int
    pending_attempt_ids: tuple[str, ...]
    running_attempt_ids: tuple[str, ...]
    hard_abort_attempt_ids: tuple[str, ...]
    hard_abort_and_report: bool
    return_to_strategy: bool
    tripwire_a100_microseconds: int = GTOK_TRIPWIRE_A100_MICROSECONDS

    def __post_init__(self) -> None:
        _require_sha256(self.event_ledger_sha256, "event_ledger_sha256")
        _require_exact_int(
            self.cumulative_a100_microseconds,
            "cumulative_a100_microseconds",
        )
        for name in (
            "pending_attempt_ids",
            "running_attempt_ids",
            "hard_abort_attempt_ids",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(item, str) or not item for item in values
            ):
                raise TypeError(f"{name} must be a tuple of nonempty attempt IDs")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} may not contain duplicates")
        active = set(self.pending_attempt_ids) | set(self.running_attempt_ids)
        if set(self.pending_attempt_ids) & set(self.running_attempt_ids):
            raise ValueError("an attempt cannot be both pending and running")
        if self.tripwire_a100_microseconds != GTOK_TRIPWIRE_A100_MICROSECONDS:
            raise ValueError("the runtime tripwire must remain 12 A100-hours")
        for name in ("hard_abort_and_report", "return_to_strategy"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        crossed = self.cumulative_a100_microseconds > self.tripwire_a100_microseconds
        if crossed:
            if set(self.hard_abort_attempt_ids) != active:
                raise ValueError("crossing must hard-abort every pending and running attempt")
            if not self.hard_abort_and_report or not self.return_to_strategy:
                raise ValueError("crossing requires hard abort + report + strategy return")
        elif self.hard_abort_attempt_ids or self.hard_abort_and_report or self.return_to_strategy:
            raise ValueError("an under-tripwire snapshot may not claim a tripwire abort")

    @property
    def crossed(self) -> bool:
        return self.cumulative_a100_microseconds > self.tripwire_a100_microseconds

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_runtime_tripwire_snapshot", self)


def enforce_runtime_tripwire_v2(snapshot: RuntimeTripwireSnapshotV2) -> None:
    """Hard-stop a valid crossed snapshot after its abort obligations are bound."""

    if not isinstance(snapshot, RuntimeTripwireSnapshotV2):
        raise TypeError("runtime meter requires a RuntimeTripwireSnapshotV2")
    if snapshot.crossed:
        raise GTokV2Stop(
            "cumulative meter crossed 12 A100-hours; pending/running work hard-aborted"
        )


@dataclass(frozen=True)
class CampaignComputeReceiptV2:
    """Append-only, projected, all-attempt meter for one G-TOK phase."""

    scope: str
    predecessor_campaign_sha256: str | None
    preflight: PreflightProjectionReceiptV2
    attempts: tuple[ComputeAttemptReceiptV2, ...]
    event_ledger_sha256: str
    consumed_a100_microseconds: int
    selected_run_a100_microseconds: int
    runtime_snapshot: RuntimeTripwireSnapshotV2
    all_attempts_accounted: bool
    pending_or_running_attempt_count: int = 0
    tripwire_a100_microseconds: int = GTOK_TRIPWIRE_A100_MICROSECONDS

    def __post_init__(self) -> None:
        if self.scope not in GTOK_COMPUTE_SCOPES:
            raise ValueError("campaign uses an unregistered compute scope")
        if not isinstance(self.preflight, PreflightProjectionReceiptV2):
            raise TypeError("campaign requires a PreflightProjectionReceiptV2")
        if self.preflight.scope != self.scope:
            raise ValueError("campaign scope differs from its preflight")
        if self.scope == "base_screen":
            if self.predecessor_campaign_sha256 is not None:
                raise ValueError("base-screen campaign may not have a predecessor")
        else:
            if self.predecessor_campaign_sha256 is None:
                raise ValueError("confirmation campaign requires the base predecessor")
            _require_sha256(
                self.predecessor_campaign_sha256,
                "predecessor_campaign_sha256",
            )
        if not isinstance(self.attempts, tuple) or not self.attempts:
            raise ValueError("campaign requires a nonempty all-attempt ledger")
        if any(not isinstance(item, ComputeAttemptReceiptV2) for item in self.attempts):
            raise TypeError("attempts must contain ComputeAttemptReceiptV2 values")
        if any(item.scope != self.scope for item in self.attempts):
            raise ValueError("compute attempt scope differs from its campaign")
        attempt_ids = tuple(item.attempt_id for item in self.attempts)
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("campaign compute attempt IDs must be unique")
        projections = {item.vocab_size: item for item in self.preflight.calibrations}
        if any(item.vocab_size not in projections for item in self.attempts):
            raise ValueError("compute attempt lacks a per-arm calibration projection")
        for attempt in self.attempts:
            projection = projections[attempt.vocab_size]
            if (
                attempt.calibration_projection_sha256 != projection.receipt_sha256
                or attempt.projected_run_a100_microseconds
                != projection.projected_run_a100_microseconds
            ):
                raise ValueError("compute attempt does not join its arm projection")
        calibration_attempts = tuple(
            item for item in self.attempts if item.kind == "calibration"
        )
        if len(calibration_attempts) != len(self.preflight.calibrations):
            raise ValueError("every per-arm calibration must be charged exactly once")
        calibration_by_id = {item.attempt_id: item for item in calibration_attempts}
        for projection in self.preflight.calibrations:
            attempt = calibration_by_id.get(projection.calibration_attempt_id)
            if (
                attempt is None
                or attempt.vocab_size != projection.vocab_size
                or attempt.consumed_a100_microseconds
                != projection.measured_a100_microseconds
                or attempt.status != "completed"
            ):
                raise ValueError("calibration attempt is absent from the cumulative ledger")
        _require_sha256(self.event_ledger_sha256, "event_ledger_sha256")
        _require_exact_int(
            self.consumed_a100_microseconds,
            "consumed_a100_microseconds",
            minimum=1,
        )
        expected_consumed = self.preflight.prior_campaign_a100_microseconds + sum(
            item.consumed_a100_microseconds for item in self.attempts
        )
        if self.consumed_a100_microseconds != expected_consumed:
            raise ValueError("cumulative meter must include calibration, retries, and all attempts")
        _require_exact_int(
            self.selected_run_a100_microseconds,
            "selected_run_a100_microseconds",
            minimum=1,
        )
        if self.selected_run_a100_microseconds > sum(
            item.consumed_a100_microseconds
            for item in self.attempts
            if item.kind == "full_run"
        ):
            raise ValueError("selected-run compute cannot exceed full-run attempt compute")
        if self.tripwire_a100_microseconds != GTOK_TRIPWIRE_A100_MICROSECONDS:
            raise ValueError("the cumulative tripwire must remain 12 A100-hours")
        if self.all_attempts_accounted is not True:
            raise ValueError("every failed, retried, and completed attempt must be metered")
        live_count = sum(
            item.status in ("pending", "running") for item in self.attempts
        )
        if self.pending_or_running_attempt_count != live_count:
            raise ValueError("pending/running count differs from the attempt ledger")
        if live_count:
            raise ValueError("a green campaign receipt may not leave an attempt live")
        if not isinstance(self.runtime_snapshot, RuntimeTripwireSnapshotV2):
            raise TypeError("campaign requires a RuntimeTripwireSnapshotV2")
        if (
            self.runtime_snapshot.event_ledger_sha256 != self.event_ledger_sha256
            or self.runtime_snapshot.cumulative_a100_microseconds
            != self.consumed_a100_microseconds
        ):
            raise ValueError("runtime snapshot does not equal the cumulative event ledger")
        enforce_runtime_tripwire_v2(self.runtime_snapshot)

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_compute_campaign", self)


@dataclass(frozen=True, init=False)
class ValidatedGTokMatrixV2:
    """Factory-only, complete four-arm/two-seed evidence matrix."""

    schema: str
    authority_chain: tuple[str, ...]
    selector_literal_binding_sha256: str
    corpus: FrozenScreenCorpusV2
    tokenizers: tuple[TokenizerArmReceiptV2, ...]
    runs: tuple[GTokRunReceiptV2, ...]
    compute: CampaignComputeReceiptV2
    seeds: tuple[int, ...]
    status: str

    def __new__(cls) -> "ValidatedGTokMatrixV2":
        raise TypeError("ValidatedGTokMatrixV2 is factory-minted after validation")

    @classmethod
    def _validated(
        cls,
        *,
        corpus: FrozenScreenCorpusV2,
        tokenizers: tuple[TokenizerArmReceiptV2, ...],
        runs: tuple[GTokRunReceiptV2, ...],
        compute: CampaignComputeReceiptV2,
        seeds: tuple[int, ...],
        sentinel: object,
    ) -> "ValidatedGTokMatrixV2":
        if sentinel is not _FACTORY_SENTINEL:
            raise PermissionError("validated G-TOK matrices are factory-only")
        value = object.__new__(cls)
        payload = {
            "schema": "weft1_gtok_v2_complete_matrix",
            "authority_chain": GTOK_V2_AUTHORITY_CHAIN,
            "selector_literal_binding_sha256": (
                GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2
            ),
            "corpus": corpus,
            "tokenizers": tokenizers,
            "runs": runs,
            "compute": compute,
            "seeds": seeds,
            "status": "GREEN_COMPLETE_EVIDENCE",
        }
        for name, item in payload.items():
            object.__setattr__(value, name, item)
        return value

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256(self.schema, self)


def validate_complete_gtok_matrix_v2(
    runs: tuple[GTokRunReceiptV2, ...],
    *,
    corpus: FrozenScreenCorpusV2,
    tokenizers: tuple[TokenizerArmReceiptV2, ...],
    compute: CampaignComputeReceiptV2,
) -> ValidatedGTokMatrixV2:
    """Join the exact eight runs without selecting a vocabulary."""

    if not isinstance(corpus, FrozenScreenCorpusV2):
        raise TypeError("corpus must be a FrozenScreenCorpusV2")
    if not isinstance(compute, CampaignComputeReceiptV2):
        raise TypeError("compute must be a CampaignComputeReceiptV2")
    if compute.scope != "base_screen":
        raise ValueError("base G-TOK matrix requires a base_screen compute campaign")
    if tuple(item.vocab_size for item in compute.preflight.calibrations) != (
        GTOK_VOCABULARY_ARMS
    ):
        raise ValueError("base preflight requires one calibration per vocabulary arm")
    expected_run_count = len(GTOK_VOCABULARY_ARMS) * GTOK_SEED_COUNT
    if not isinstance(runs, tuple) or len(runs) != expected_run_count:
        raise ValueError(f"G-TOK v2 requires exactly {expected_run_count} completed runs")
    if any(not isinstance(run, GTokRunReceiptV2) for run in runs):
        raise TypeError("runs must contain GTokRunReceiptV2 values")
    keys = tuple((run.vocab_size, run.seed) for run in runs)
    if len(set(keys)) != expected_run_count:
        raise ValueError("G-TOK v2 run keys must be unique")
    seeds = tuple(sorted({run.seed for run in runs}))
    if len(seeds) != GTOK_SEED_COUNT or set(keys) != {
        (vocab_size, seed) for vocab_size in GTOK_VOCABULARY_ARMS for seed in seeds
    }:
        raise ValueError("each vocabulary arm requires the identical two-seed panel")

    if not isinstance(tokenizers, tuple) or tuple(
        item.vocab_size for item in tokenizers
    ) != GTOK_VOCABULARY_ARMS:
        raise ValueError("tokenizers require one canonically ordered receipt per arm")
    if any(not isinstance(item, TokenizerArmReceiptV2) for item in tokenizers):
        raise TypeError("tokenizers must contain TokenizerArmReceiptV2 values")
    if len({item.algorithm_signature for item in tokenizers}) != 1:
        raise ValueError("tokenizer arms may differ only in vocabulary-dependent artifacts")
    for field_name in (
        "tokenizer_json_sha256",
        "merges_sha256",
        "token_inventory_sha256",
        "double_fit_receipt_sha256",
    ):
        if len({getattr(item, field_name) for item in tokenizers}) != len(tokenizers):
            raise ValueError(f"each arm requires a distinct {field_name}")
    tokenizer_by_vocab = {item.vocab_size: item for item in tokenizers}
    if any(
        item.fit_stream_sha256 != corpus.training_stream_sha256
        or item.full_corpus_manifest_sha256 != corpus.full_corpus_manifest_sha256
        for item in tokenizers
    ):
        raise ValueError("tokenizers must fit T and join the frozen parent corpus")

    frozen_hashes = {run.frozen_screen_corpus_sha256 for run in runs}
    if frozen_hashes != {corpus.receipt_sha256}:
        raise ValueError("every run must join the same frozen screen corpus")
    if {run.model_topology_sha256 for run in runs} != {GTOK_PROXY_TOPOLOGY_SHA256}:
        raise ValueError("every run must use the corrected ten-block topology")
    if len({run.initialization_recipe_sha256 for run in runs}) != 1:
        raise ValueError("every run must use one initialization recipe")
    if any(
        run.tokenizer_receipt_sha256
        != tokenizer_by_vocab[run.vocab_size].receipt_sha256
        for run in runs
    ):
        raise ValueError("run tokenizer receipt is not joined to its vocabulary arm")

    for seed in seeds:
        seed_runs = tuple(run for run in runs if run.seed == seed)
        if len({run.data_order_seed for run in seed_runs}) != 1 or len(
            {run.data_order_sha256 for run in seed_runs}
        ) != 1:
            raise ValueError("arms must share an identical data order within each seed")
        if len({run.initialization_seed for run in seed_runs}) != 1 or len(
            {run.shared_initial_state_sha256 for run in seed_runs}
        ) != 1:
            raise ValueError(
                "arms must share identical non-vocabulary initialization within each seed"
            )
    if len({run.data_order_sha256 for run in runs}) != GTOK_SEED_COUNT:
        raise ValueError("the two seeds must use distinct data orders")
    if len({run.data_order_seed for run in runs}) != GTOK_SEED_COUNT:
        raise ValueError("the two seeds must use distinct data-order seeds")
    if len({run.shared_initial_state_sha256 for run in runs}) != GTOK_SEED_COUNT:
        raise ValueError("the two seeds must use distinct initial states")
    if len({run.initialization_seed for run in runs}) != GTOK_SEED_COUNT:
        raise ValueError("the two seeds must use distinct initialization seeds")
    attempt_ids = tuple(run.compute_attempt_id for run in runs)
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("every selected base run requires a distinct compute attempt")
    compute_by_id = {item.attempt_id: item for item in compute.attempts}
    for run in runs:
        attempt = compute_by_id.get(run.compute_attempt_id)
        if (
            attempt is None
            or attempt.kind != "full_run"
            or attempt.vocab_size != run.vocab_size
            or attempt.seed != run.seed
            or attempt.status != "completed"
            or attempt.consumed_a100_microseconds
            != run.measured_a100_microseconds
            or attempt.hard_abort_issued
        ):
            raise ValueError("selected base run is absent from the all-attempt ledger")

    denominator = corpus.heldout_denominator_signature
    for run in runs:
        if run.terminal.training_raw_bytes != corpus.training_realized_bytes:
            raise ValueError("terminal milestone must equal realized T, not its target")
        if run.observations[1].training_raw_bytes > run.terminal.training_raw_bytes:
            raise ValueError("the 2B crossing cannot occur after terminal realized T")
        for observation in run.observations:
            if observation.heldout_stream_sha256 != corpus.heldout_stream_sha256:
                raise ValueError("every BPB point must use the frozen H stream")
            if observation.denominator_signature != denominator:
                raise ValueError("every BPB denominator must equal manifested realized H")

    selected_compute = sum(run.measured_a100_microseconds for run in runs)
    if selected_compute != compute.selected_run_a100_microseconds:
        raise ValueError("selected-run compute differs from the cumulative meter")
    ordered_runs = tuple(sorted(runs, key=lambda item: (item.vocab_size, item.seed)))
    return ValidatedGTokMatrixV2._validated(
        corpus=corpus,
        tokenizers=tokenizers,
        runs=ordered_runs,
        compute=compute,
        seeds=seeds,
        sentinel=_FACTORY_SENTINEL,
    )


@dataclass(frozen=True)
class VocabularyAdmissibilityReceiptV2:
    """Exact rung-B vocabulary parameter share for one arm."""

    vocab_size: int
    vocabulary_parameter_count: int
    target_parameter_count: int
    target_rung: str = "B"
    cap: Fraction = GTOK_VOCABULARY_FRACTION_CAP

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("admissibility uses an unregistered vocabulary arm")
        _require_exact_int(
            self.vocabulary_parameter_count,
            "vocabulary_parameter_count",
            minimum=1,
        )
        _require_exact_int(
            self.target_parameter_count,
            "target_parameter_count",
            minimum=1,
        )
        if self.vocabulary_parameter_count >= self.target_parameter_count:
            raise ValueError("vocabulary parameters must be a proper target subset")
        if self.target_rung != "B":
            raise ValueError("G-TOK selection admissibility is priced on target rung B")
        if self.cap != GTOK_VOCABULARY_FRACTION_CAP:
            raise ValueError("rung-B vocabulary share cap must remain 20 percent")

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.vocabulary_parameter_count, self.target_parameter_count)

    @property
    def admissible(self) -> bool:
        return self.fraction <= self.cap

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_vocabulary_admissibility", self)


@dataclass(frozen=True)
class ArmTerminalStatisticsV2:
    vocab_size: int
    seeds: tuple[int, int]
    seed_bpbs: tuple[float, float]
    mean_bpb: float
    sample_sd: float

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("statistics use an unregistered vocabulary arm")
        if not isinstance(self.seeds, tuple) or len(self.seeds) != 2:
            raise ValueError("statistics require exactly two seeds")
        if any(type(seed) is not int for seed in self.seeds) or len(set(self.seeds)) != 2:
            raise ValueError("statistics seeds must be two distinct exact integers")
        if not isinstance(self.seed_bpbs, tuple) or len(self.seed_bpbs) != 2:
            raise ValueError("statistics require exactly two BPB values")
        values = tuple(_require_finite(item, "seed BPB", positive=True) for item in self.seed_bpbs)
        expected_mean = math.fsum(values) / 2.0
        expected_sd = math.sqrt(math.fsum((item - expected_mean) ** 2 for item in values))
        if self.mean_bpb != expected_mean or self.sample_sd != expected_sd:
            raise ValueError("arm mean/SD do not equal the two-seed sample statistics")


@dataclass(frozen=True)
class SelectionComparisonV2:
    comparison_index: int
    incumbent_vocab_before: int
    challenger_vocab: int
    incumbent_mean_bpb: float
    challenger_mean_bpb: float
    incumbent_sample_sd: float
    challenger_sample_sd: float
    s_hat: float
    delta_bpb: float
    two_s_hat: float
    three_s_hat: float
    tie_diagnostic: bool
    displaced: bool
    incumbent_vocab_after: int
    metric: str = GTOK_TERMINAL_METRIC
    budget: str = GTOK_TERMINAL_BUDGET
    displacement_operator: str = ">"

    def __post_init__(self) -> None:
        _require_exact_int(self.comparison_index, "comparison_index")
        if (
            self.incumbent_vocab_before not in GTOK_VOCABULARY_ARMS
            or self.challenger_vocab not in GTOK_VOCABULARY_ARMS
        ):
            raise ValueError("comparison arms must be registered")
        if self.challenger_vocab <= self.incumbent_vocab_before:
            raise ValueError("selector traversal must challenge in ascending vocabulary order")
        for name in (
            "incumbent_mean_bpb",
            "challenger_mean_bpb",
            "incumbent_sample_sd",
            "challenger_sample_sd",
            "s_hat",
            "delta_bpb",
            "two_s_hat",
            "three_s_hat",
        ):
            _require_finite(getattr(self, name), name)
        for name in (
            "incumbent_sample_sd",
            "challenger_sample_sd",
            "s_hat",
            "two_s_hat",
            "three_s_hat",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} may not be negative")
        expected_s_hat = math.sqrt(
            (self.incumbent_sample_sd**2 + self.challenger_sample_sd**2) / 2.0
        )
        expected_delta = self.incumbent_mean_bpb - self.challenger_mean_bpb
        if self.s_hat != expected_s_hat:
            raise ValueError("s_hat must use the pairwise equal-n pooled sample SD")
        if self.delta_bpb != expected_delta:
            raise ValueError("comparison delta uses the wrong arm pair or metric")
        if self.two_s_hat != 2.0 * self.s_hat or self.three_s_hat != 3.0 * self.s_hat:
            raise ValueError("comparison thresholds must be exact multiples of s_hat")
        if self.tie_diagnostic is not (abs(self.delta_bpb) < self.two_s_hat):
            raise ValueError("2*s_hat tie diagnostic must use strict less-than")
        if self.displaced is not (self.delta_bpb > self.three_s_hat):
            raise ValueError("larger V may displace only for delta strictly > 3*s_hat")
        expected_after = self.challenger_vocab if self.displaced else self.incumbent_vocab_before
        if self.incumbent_vocab_after != expected_after:
            raise ValueError("comparison incumbent transition is inconsistent")
        if self.metric != GTOK_TERMINAL_METRIC or self.budget != GTOK_TERMINAL_BUDGET:
            raise ValueError("selector comparison must use terminal pooled BPB at realized T")
        if self.displacement_operator != ">":
            raise ValueError("selector displacement operator must remain strict >")


@dataclass(frozen=True)
class GTokSelectionReceiptV2:
    matrix_receipt_sha256: str
    selector_literal_binding_sha256: str
    seed_specific_orders: tuple[tuple[int, tuple[int, ...]], ...]
    agreed_strict_terminal_order: tuple[int, ...]
    arm_statistics: tuple[ArmTerminalStatisticsV2, ...]
    admissibility: tuple[VocabularyAdmissibilityReceiptV2, ...]
    comparisons: tuple[SelectionComparisonV2, ...]
    selected_vocab_size: int
    compute_confirmation_pair: tuple[int, int]
    status: str = "GREEN_PENDING_COMPUTE_CONFIRMATION"

    def __post_init__(self) -> None:
        _require_sha256(self.matrix_receipt_sha256, "matrix_receipt_sha256")
        if self.selector_literal_binding_sha256 != GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2:
            raise ValueError("selection uses a different A2-R7 literal binding")
        if self.status != "GREEN_PENDING_COMPUTE_CONFIRMATION":
            raise ValueError("selection status drifted")
        if self.selected_vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("selection result is not a registered vocabulary")
        if len(self.compute_confirmation_pair) != 2:
            raise ValueError("compute confirmation requires exactly two arms")
        if tuple(row.comparison_index for row in self.comparisons) != tuple(
            range(len(self.comparisons))
        ):
            raise ValueError("selection comparisons must preserve traversal order")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_selection", self)


def _arm_terminal_statistics(
    matrix: ValidatedGTokMatrixV2,
) -> tuple[ArmTerminalStatisticsV2, ...]:
    rows: list[ArmTerminalStatisticsV2] = []
    for vocab_size in GTOK_VOCABULARY_ARMS:
        arm_runs = tuple(run for run in matrix.runs if run.vocab_size == vocab_size)
        arm_runs = tuple(sorted(arm_runs, key=lambda item: item.seed))
        values = tuple(run.terminal.pooled_bpb for run in arm_runs)
        mean = math.fsum(values) / 2.0
        sample_sd = math.sqrt(math.fsum((item - mean) ** 2 for item in values))
        rows.append(
            ArmTerminalStatisticsV2(
                vocab_size=vocab_size,
                seeds=(arm_runs[0].seed, arm_runs[1].seed),
                seed_bpbs=(values[0], values[1]),
                mean_bpb=mean,
                sample_sd=sample_sd,
            )
        )
    return tuple(rows)


def _strict_seed_orders(
    matrix: ValidatedGTokMatrixV2,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    rows: list[tuple[int, tuple[int, ...]]] = []
    for seed in matrix.seeds:
        seed_runs = tuple(run for run in matrix.runs if run.seed == seed)
        values = tuple(run.terminal.pooled_bpb for run in seed_runs)
        if len(set(values)) != len(values):
            raise GTokV2Stop("seed-specific terminal BPB order is not strict")
        order = tuple(
            run.vocab_size
            for run in sorted(seed_runs, key=lambda item: item.terminal.pooled_bpb)
        )
        rows.append((seed, order))
    if len({order for _, order in rows}) != 1:
        raise GTokV2Stop(
            "seed-specific terminal pooled-BPB total orders disagree; return to strategy"
        )
    return tuple(rows)


def _build_selection_v2(
    matrix: ValidatedGTokMatrixV2,
    admissibility: tuple[VocabularyAdmissibilityReceiptV2, ...],
) -> GTokSelectionReceiptV2:
    if not isinstance(matrix, ValidatedGTokMatrixV2):
        raise TypeError("selector requires a factory-validated G-TOK matrix")
    if not isinstance(admissibility, tuple) or tuple(
        row.vocab_size for row in admissibility
    ) != GTOK_VOCABULARY_ARMS:
        raise ValueError("admissibility requires one canonical rung-B row per arm")
    if any(not isinstance(row, VocabularyAdmissibilityReceiptV2) for row in admissibility):
        raise TypeError("admissibility rows must be VocabularyAdmissibilityReceiptV2")

    seed_orders = _strict_seed_orders(matrix)
    agreed_order = seed_orders[0][1]
    allowed = tuple(row.vocab_size for row in admissibility if row.admissible)
    if not allowed:
        raise GTokV2Stop("no vocabulary arm passes the rung-B 20-percent guard")
    if any(vocab_size not in allowed for vocab_size in agreed_order[:2]):
        raise GTokV2Stop("a compute-confirmation arm fails the rung-B admissibility guard")

    statistics = _arm_terminal_statistics(matrix)
    by_vocab = {row.vocab_size: row for row in statistics}
    incumbent = min(allowed)
    comparisons: list[SelectionComparisonV2] = []
    for challenger in tuple(item for item in GTOK_VOCABULARY_ARMS if item in allowed)[1:]:
        incumbent_stats = by_vocab[incumbent]
        challenger_stats = by_vocab[challenger]
        s_hat = math.sqrt(
            (incumbent_stats.sample_sd**2 + challenger_stats.sample_sd**2) / 2.0
        )
        delta = incumbent_stats.mean_bpb - challenger_stats.mean_bpb
        displaced = delta > 3.0 * s_hat
        next_incumbent = challenger if displaced else incumbent
        comparisons.append(
            SelectionComparisonV2(
                comparison_index=len(comparisons),
                incumbent_vocab_before=incumbent,
                challenger_vocab=challenger,
                incumbent_mean_bpb=incumbent_stats.mean_bpb,
                challenger_mean_bpb=challenger_stats.mean_bpb,
                incumbent_sample_sd=incumbent_stats.sample_sd,
                challenger_sample_sd=challenger_stats.sample_sd,
                s_hat=s_hat,
                delta_bpb=delta,
                two_s_hat=2.0 * s_hat,
                three_s_hat=3.0 * s_hat,
                tie_diagnostic=abs(delta) < 2.0 * s_hat,
                displaced=displaced,
                incumbent_vocab_after=next_incumbent,
            )
        )
        incumbent = next_incumbent
    return GTokSelectionReceiptV2(
        matrix_receipt_sha256=matrix.receipt_sha256,
        selector_literal_binding_sha256=GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2,
        seed_specific_orders=seed_orders,
        agreed_strict_terminal_order=agreed_order,
        arm_statistics=statistics,
        admissibility=admissibility,
        comparisons=tuple(comparisons),
        selected_vocab_size=incumbent,
        compute_confirmation_pair=(agreed_order[0], agreed_order[1]),
    )


def select_vocabulary_v2(
    matrix: ValidatedGTokMatrixV2,
    *,
    admissibility: tuple[VocabularyAdmissibilityReceiptV2, ...],
) -> GTokSelectionReceiptV2:
    """Apply the complete deterministic A2-R7 traversal."""

    return _build_selection_v2(matrix, admissibility)


def validate_selection_receipt_v2(
    selection: GTokSelectionReceiptV2,
    *,
    matrix: ValidatedGTokMatrixV2,
) -> GTokSelectionReceiptV2:
    """Recompute every selection field; caller-authored summaries are rejected."""

    if not isinstance(selection, GTokSelectionReceiptV2):
        raise TypeError("selection must be a GTokSelectionReceiptV2")
    expected = _build_selection_v2(matrix, selection.admissibility)
    if canonical_json_bytes(selection) != canonical_json_bytes(expected):
        raise ValueError("selection receipt differs from deterministic recomputation")
    return selection


@dataclass(frozen=True)
class ComputeConfirmationRunV2:
    vocab_size: int
    seed: int
    base_run_receipt_sha256: str
    compute_attempt_id: str
    common_flop_budget: int
    measured_flops: int
    heldout_stream_sha256: str
    strata: tuple[StratumNllReceipt, ...]
    measured_a100_microseconds: int
    checkpoint_retained: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("confirmation uses an unregistered vocabulary")
        if type(self.seed) is not int:
            raise TypeError("confirmation seed must be an exact integer")
        _require_sha256(self.base_run_receipt_sha256, "base_run_receipt_sha256")
        if not isinstance(self.compute_attempt_id, str) or not self.compute_attempt_id:
            raise ValueError("confirmation compute_attempt_id must be nonempty")
        _require_exact_int(self.common_flop_budget, "common_flop_budget", minimum=1)
        _require_exact_int(self.measured_flops, "measured_flops", minimum=1)
        if self.measured_flops != self.common_flop_budget:
            raise ValueError("compute confirmation must stop at the exact common FLOP budget")
        _require_sha256(self.heldout_stream_sha256, "heldout_stream_sha256")
        if not isinstance(self.strata, tuple) or any(
            not isinstance(item, StratumNllReceipt) for item in self.strata
        ):
            raise TypeError("confirmation strata must contain StratumNllReceipt values")
        if tuple(item.stratum for item in self.strata) != GTOK_STRATA:
            raise ValueError("confirmation requires canonical held-out strata")
        _require_exact_int(
            self.measured_a100_microseconds,
            "measured_a100_microseconds",
            minimum=1,
        )
        if type(self.checkpoint_retained) is not bool:
            raise TypeError("checkpoint_retained must be an exact bool")
        if self.checkpoint_retained:
            raise ValueError("compute confirmation may not retain checkpoints")
        if not math.isfinite(self.pooled_bpb):
            raise ValueError("confirmation pooled BPB must remain finite")

    @property
    def pooled_bpb(self) -> float:
        nll = math.fsum(item.nll_nats for item in self.strata)
        raw_bytes = sum(item.raw_byte_count for item in self.strata)
        return nll / (math.log(2.0) * raw_bytes)

    @property
    def denominator_signature(self) -> tuple[tuple[str, int], ...]:
        return tuple((item.stratum, item.raw_byte_count) for item in self.strata)

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_compute_confirmation_run", self)


@dataclass(frozen=True, init=False)
class ValidatedComputeConfirmationV2:
    selection_receipt_sha256: str
    matrix_receipt_sha256: str
    compute_campaign_receipt_sha256: str
    pair: tuple[int, int]
    common_flop_budget: int
    runs: tuple[ComputeConfirmationRunV2, ...]
    cumulative_campaign_a100_microseconds: int
    status: str

    def __new__(cls) -> "ValidatedComputeConfirmationV2":
        raise TypeError(
            "ValidatedComputeConfirmationV2 is factory-minted after validation"
        )

    @classmethod
    def _validated(
        cls,
        *,
        selection: GTokSelectionReceiptV2,
        matrix: ValidatedGTokMatrixV2,
        compute: CampaignComputeReceiptV2,
        common_flop_budget: int,
        runs: tuple[ComputeConfirmationRunV2, ...],
        sentinel: object,
    ) -> "ValidatedComputeConfirmationV2":
        if sentinel is not _FACTORY_SENTINEL:
            raise PermissionError("compute confirmations are factory-only")
        value = object.__new__(cls)
        payload = {
            "selection_receipt_sha256": selection.receipt_sha256,
            "matrix_receipt_sha256": matrix.receipt_sha256,
            "compute_campaign_receipt_sha256": compute.receipt_sha256,
            "pair": selection.compute_confirmation_pair,
            "common_flop_budget": common_flop_budget,
            "runs": runs,
            "cumulative_campaign_a100_microseconds": (
                compute.consumed_a100_microseconds
            ),
            "status": "GREEN_NO_REVERSAL",
        }
        for name, item in payload.items():
            object.__setattr__(value, name, item)
        return value

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_compute_confirmation", self)


def validate_compute_confirmation_v2(
    runs: tuple[ComputeConfirmationRunV2, ...],
    *,
    matrix: ValidatedGTokMatrixV2,
    selection: GTokSelectionReceiptV2,
    compute: CampaignComputeReceiptV2,
) -> ValidatedComputeConfirmationV2:
    """Validate the preregistered top-two equal-FLOP confirmation."""

    validate_selection_receipt_v2(selection, matrix=matrix)
    if not isinstance(compute, CampaignComputeReceiptV2):
        raise TypeError("confirmation requires a structured compute campaign receipt")
    if compute.scope != "confirmation":
        raise ValueError("confirmation requires a confirmation compute scope")
    if compute.predecessor_campaign_sha256 != matrix.compute.receipt_sha256:
        raise ValueError("confirmation compute does not extend the base campaign receipt")
    if (
        compute.preflight.prior_campaign_a100_microseconds
        != matrix.compute.consumed_a100_microseconds
        or compute.preflight.prior_event_ledger_sha256
        != matrix.compute.event_ledger_sha256
    ):
        raise ValueError("confirmation preflight does not begin at the base campaign meter")
    pair = selection.compute_confirmation_pair
    if tuple(item.vocab_size for item in compute.preflight.calibrations) != tuple(
        sorted(pair)
    ):
        raise ValueError("confirmation preflight requires one calibration per selected arm")
    expected_keys = {(vocab, seed) for vocab in pair for seed in matrix.seeds}
    if not isinstance(runs, tuple) or len(runs) != len(expected_keys):
        raise ValueError("compute confirmation requires the top two arms at both seeds")
    if any(not isinstance(run, ComputeConfirmationRunV2) for run in runs):
        raise TypeError("confirmation runs must be ComputeConfirmationRunV2 values")
    keys = {(run.vocab_size, run.seed) for run in runs}
    if keys != expected_keys or len(keys) != len(runs):
        raise ValueError("compute confirmation run keys differ from the registered pair")
    flop_budgets = {run.common_flop_budget for run in runs}
    if len(flop_budgets) != 1:
        raise ValueError("confirmation runs must share one exact FLOP budget")
    denominator = matrix.corpus.heldout_denominator_signature
    if any(
        run.heldout_stream_sha256 != matrix.corpus.heldout_stream_sha256
        or run.denominator_signature != denominator
        for run in runs
    ):
        raise ValueError("confirmation must use the same frozen H denominator")
    by_key = {(run.vocab_size, run.seed): run for run in runs}
    base_by_key = {(run.vocab_size, run.seed): run for run in matrix.runs}
    if any(
        run.base_run_receipt_sha256
        != base_by_key[(run.vocab_size, run.seed)].receipt_sha256
        for run in runs
    ):
        raise ValueError("confirmation run is not joined to its base arm/seed receipt")
    attempt_ids = tuple(run.compute_attempt_id for run in runs)
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("confirmation selected-run attempt IDs must be unique")
    compute_by_id = {item.attempt_id: item for item in compute.attempts}
    for run in runs:
        attempt = compute_by_id.get(run.compute_attempt_id)
        if (
            attempt is None
            or attempt.kind != "full_run"
            or attempt.vocab_size != run.vocab_size
            or attempt.seed != run.seed
            or attempt.status != "completed"
            or attempt.consumed_a100_microseconds
            != run.measured_a100_microseconds
            or attempt.hard_abort_issued
        ):
            raise ValueError("confirmation run is absent from the all-attempt ledger")
    for seed in matrix.seeds:
        first = by_key[(pair[0], seed)].pooled_bpb
        second = by_key[(pair[1], seed)].pooled_bpb
        if not first < second:
            raise GTokV2Stop(
                "compute-matched confirmation reversed or tied; return to strategy"
            )
    selected_compute = sum(run.measured_a100_microseconds for run in runs)
    if compute.selected_run_a100_microseconds != selected_compute:
        raise ValueError("confirmation selected-run compute differs from its meter")
    ordered = tuple(sorted(runs, key=lambda item: (item.vocab_size, item.seed)))
    return ValidatedComputeConfirmationV2._validated(
        selection=selection,
        matrix=matrix,
        compute=compute,
        common_flop_budget=next(iter(flop_budgets)),
        runs=ordered,
        sentinel=_FACTORY_SENTINEL,
    )


@dataclass(frozen=True)
class VocabExtBasisV2:
    """Immutable append-only continuation basis for the selected tokenizer."""

    vocab_size: int
    tokenizer_json_sha256: str
    merges_sha256: str
    token_inventory_sha256: str
    reserved_inventory_sha256: str
    pretokenizer_regex_sha256: str
    full_corpus_manifest_sha256: str
    screen_submanifest_sha256: str
    existing_token_ids_never_renumbered: bool = True
    extension_mode: str = "append_only_merge_continuation"

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("VOCAB-EXT basis uses an unregistered vocabulary")
        for name in (
            "tokenizer_json_sha256",
            "merges_sha256",
            "token_inventory_sha256",
            "reserved_inventory_sha256",
            "pretokenizer_regex_sha256",
            "full_corpus_manifest_sha256",
            "screen_submanifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.existing_token_ids_never_renumbered is not True:
            raise ValueError("VOCAB-EXT may never renumber an existing token ID")
        if self.extension_mode != "append_only_merge_continuation":
            raise ValueError("VOCAB-EXT must remain append-only merge continuation")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_vocab_ext_basis", self)


@dataclass(frozen=True, init=False)
class VocabularyFreezeArtifactV2:
    """Factory-only immutable V and VOCAB-EXT freeze artifact."""

    status: str
    selected_vocab_size: int
    matrix_receipt_sha256: str
    selection_receipt_sha256: str
    confirmation_receipt_sha256: str
    full_corpus_manifest_sha256: str
    screen_submanifest_sha256: str
    tokenizer_json_sha256: str
    vocab_ext_basis: VocabExtBasisV2
    selector_literal_binding_sha256: str

    def __new__(cls) -> "VocabularyFreezeArtifactV2":
        raise TypeError("VocabularyFreezeArtifactV2 is factory-minted after validation")

    @classmethod
    def _minted(
        cls,
        *,
        matrix: ValidatedGTokMatrixV2,
        selection: GTokSelectionReceiptV2,
        confirmation: ValidatedComputeConfirmationV2,
        basis: VocabExtBasisV2,
        sentinel: object,
    ) -> "VocabularyFreezeArtifactV2":
        if sentinel is not _FACTORY_SENTINEL:
            raise PermissionError("vocabulary freeze artifacts are factory-only")
        value = object.__new__(cls)
        payload = {
            "status": "FROZEN_GREEN",
            "selected_vocab_size": selection.selected_vocab_size,
            "matrix_receipt_sha256": matrix.receipt_sha256,
            "selection_receipt_sha256": selection.receipt_sha256,
            "confirmation_receipt_sha256": confirmation.receipt_sha256,
            "full_corpus_manifest_sha256": matrix.corpus.full_corpus_manifest_sha256,
            "screen_submanifest_sha256": matrix.corpus.screen_submanifest_sha256,
            "tokenizer_json_sha256": basis.tokenizer_json_sha256,
            "vocab_ext_basis": basis,
            "selector_literal_binding_sha256": (
                GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2
            ),
        }
        for name, item in payload.items():
            object.__setattr__(value, name, item)
        return value

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_vocabulary_freeze", self)


def mint_vocabulary_freeze_v2(
    *,
    matrix: ValidatedGTokMatrixV2,
    selection: GTokSelectionReceiptV2,
    confirmation: ValidatedComputeConfirmationV2,
    basis: VocabExtBasisV2,
) -> VocabularyFreezeArtifactV2:
    """Mint only after green base evidence and non-reversing confirmation."""

    if not isinstance(matrix, ValidatedGTokMatrixV2):
        raise TypeError("freeze requires a factory-validated green G-TOK matrix")
    validate_selection_receipt_v2(selection, matrix=matrix)
    if not isinstance(confirmation, ValidatedComputeConfirmationV2):
        raise TypeError("freeze requires a factory-validated compute confirmation")
    if (
        confirmation.matrix_receipt_sha256 != matrix.receipt_sha256
        or confirmation.selection_receipt_sha256 != selection.receipt_sha256
        or confirmation.pair != selection.compute_confirmation_pair
        or confirmation.status != "GREEN_NO_REVERSAL"
    ):
        raise ValueError("compute confirmation does not compose with the selection")
    if not isinstance(basis, VocabExtBasisV2):
        raise TypeError("freeze requires a VocabExtBasisV2")
    tokenizer = next(
        item for item in matrix.tokenizers if item.vocab_size == selection.selected_vocab_size
    )
    expected_basis = VocabExtBasisV2(
        vocab_size=tokenizer.vocab_size,
        tokenizer_json_sha256=tokenizer.tokenizer_json_sha256,
        merges_sha256=tokenizer.merges_sha256,
        token_inventory_sha256=tokenizer.token_inventory_sha256,
        reserved_inventory_sha256=tokenizer.reserved_inventory_sha256,
        pretokenizer_regex_sha256=tokenizer.pretokenizer_regex_sha256,
        full_corpus_manifest_sha256=matrix.corpus.full_corpus_manifest_sha256,
        screen_submanifest_sha256=matrix.corpus.screen_submanifest_sha256,
    )
    if canonical_json_bytes(basis) != canonical_json_bytes(expected_basis):
        raise ValueError("VOCAB-EXT basis does not equal the selected tokenizer basis")
    return VocabularyFreezeArtifactV2._minted(
        matrix=matrix,
        selection=selection,
        confirmation=confirmation,
        basis=basis,
        sentinel=_FACTORY_SENTINEL,
    )


def refuse_unvalidated_freeze(action: str) -> NoReturn:
    """Explicit fail-closed surface for callers lacking validated receipts."""

    if not isinstance(action, str) or not action.strip():
        raise ValueError("blocked action must be named")
    raise GTokV2Stop(
        f"{action} requires a green v2 matrix, deterministic selection, "
        "compute confirmation, and exact VOCAB-EXT basis"
    )


__all__ = [
    "ArmCalibrationProjectionV2",
    "ArmTerminalStatisticsV2",
    "BpbMilestoneReceiptV2",
    "CampaignComputeReceiptV2",
    "ComputeAttemptReceiptV2",
    "ComputeConfirmationRunV2",
    "FrozenScreenCorpusV2",
    "GTOK_A2_BINDINGS_SHA256",
    "GTOK_AMENDMENT_A2_SHA256",
    "GTOK_AMENDMENT_A3_SHA256",
    "GTOK_CALIBRATION_MAX_STEPS",
    "GTOK_COMPUTE_SCOPES",
    "GTOK_FIRST_BOUNDARY_BYTES",
    "GTOK_MILESTONE_LABELS",
    "GTOK_PER_RUN_WATCHDOG_MULTIPLIER",
    "GTOK_RELEASE_CLOSE_SHA256",
    "GTOK_SECOND_BOUNDARY_BYTES",
    "GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2",
    "GTOK_SELECTOR_LITERAL_BINDING_V2",
    "GTOK_TERMINAL_BUDGET",
    "GTOK_TERMINAL_METRIC",
    "GTOK_TRIPWIRE_A100_MICROSECONDS",
    "GTOK_V2_AUTHORITY_CHAIN",
    "GTokRunReceiptV2",
    "GTokSelectionReceiptV2",
    "GTokV2Stop",
    "PreflightProjectionReceiptV2",
    "RuntimeTripwireSnapshotV2",
    "SelectionComparisonV2",
    "TokenizerArmReceiptV2",
    "ValidatedComputeConfirmationV2",
    "ValidatedGTokMatrixV2",
    "VocabExtBasisV2",
    "VocabularyAdmissibilityReceiptV2",
    "VocabularyFreezeArtifactV2",
    "enforce_runtime_tripwire_v2",
    "gtok_v2_bound_sha256",
    "mint_vocabulary_freeze_v2",
    "refuse_unvalidated_freeze",
    "select_vocabulary_v2",
    "validate_complete_gtok_matrix_v2",
    "validate_compute_confirmation_v2",
    "validate_selection_receipt_v2",
]
