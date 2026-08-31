"""Forward-only execution and selection contracts for the WEFT-1 G-TOK screen.

This module does not construct a model, optimizer, trainer, or checkpoint.  It
validates already-produced evidence under the corpus handoff as amended by
A1/A2/A3 and the release-close record.  The legacy G-TOK contract remains
unchanged; every receipt here uses a new authority chain and schema domain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import math
from typing import Any, Mapping, NoReturn

from training.weft1_corpus_a2 import A2_CAMPAIGN_ROOT_SEED
from training.weft1_corpus_a3 import execution_authority_v4_bound_sha256
from training.weft1_gtok_contract import (
    FlatAdamWRecipe,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V2,
    GTOK_HELDOUT_BYTE_TARGET,
    GTOK_PROXY_TOPOLOGY_SHA256,
    GTOK_SCREEN_HELDOUT_STRATUM_TARGETS,
    GTOK_SCREEN_TRAIN_STRATUM_TARGETS,
    GTOK_SEED_COUNT,
    GTOK_STRATA,
    GTOK_STRATUM_TOLERANCE,
    GTOK_TRAINING_BYTE_BUDGET,
    GTOK_VOCABULARY_ARMS,
    StratumNllReceipt,
    a1_flat_adamw_recipe,
    canonical_json_bytes,
    canonical_sha256,
)
from training.weft1_seed import derive_module_seed


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
GTOK_CONFIRMATION_SEMANTICS_SHA256 = (
    "2e42664d0062a119c9fadcb76bf227a91134914920116627f9244f650defe72d"
)
GTOK_SEMANTICS_AMENDMENT_S1_SHA256 = (
    "c37c4be064fe447e01182acc11b1713239c761ddd50583a8299972b4b340bd2a"
)
GTOK_SEMANTICS_AMENDMENT_S2_SHA256 = (
    "5420a4e57c080d09f5f924acc859a5579edd1ca1939c8bbdaf727e5afd55ac5e"
)
GTOK_SELECTION_CONFIRMATION_AUTHORITY_CHAIN = (
    *GTOK_V2_AUTHORITY_CHAIN,
    GTOK_CONFIRMATION_SEMANTICS_SHA256,
    GTOK_SEMANTICS_AMENDMENT_S1_SHA256,
    GTOK_SEMANTICS_AMENDMENT_S2_SHA256,
)
GTOK_SELECTION_CONFIRMATION_AUTHORITY_SHA256 = canonical_sha256(
    {
        "authority_chain": GTOK_SELECTION_CONFIRMATION_AUTHORITY_CHAIN,
        "schema": "weft1_gtok_selection_confirmation_authority_v2",
    }
)
GTOK_RHO_BPB_DECIMAL_PLACES = 6
GTOK_RHO_BPB_SCALE = 10**GTOK_RHO_BPB_DECIMAL_PLACES
GTOK_PHYSICAL_FLOP_BINDING_SHA256_V2 = (
    "8e47324f079a52da68f28878dbc5f1fdd279b2733352e787c21966683712a61a"
)
GTOK_SELECTION_CONFIRMATION_LITERAL_BINDING_V2 = {
    **GTOK_SELECTOR_LITERAL_BINDING_V2,
    "confirmation_pair": "D-C-2_winner_W_then_raw_rho_runner_up_U",
    "rho_accumulation": "float64_mean_of_two_float64_pooled_BPB_values",
    "rho_comparison": "integer_micros_after_half_even_6_decimal_rounding",
    "rho_reporting": "integer_BPB_micros_for_all_four_arms",
    "rho_tie_break": "equal_integer_micros_breaks_toward_smaller_V",
}
GTOK_SELECTION_CONFIRMATION_LITERAL_BINDING_SHA256_V2 = canonical_sha256(
    GTOK_SELECTION_CONFIRMATION_LITERAL_BINDING_V2
)
_A1_OPTIMIZER = a1_flat_adamw_recipe()
_A1_OPTIMIZER_BYTES = canonical_json_bytes(_A1_OPTIMIZER)
_SHA256_CHARS = frozenset("0123456789abcdef")
_FACTORY_SENTINEL = object()
_RHO_BPB_QUANTUM = Decimal("0.000001")
_RHO_BPB_SCALE_DECIMAL = Decimal(GTOK_RHO_BPB_SCALE)


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


def _rho_bpb_micros(value: float) -> int:
    """Round one already-computed binary64 rho half-even to six BPB decimals."""

    rho = _require_finite(value, "rho BPB", positive=True)
    with localcontext() as context:
        context.prec = 50
        rounded = Decimal.from_float(rho).quantize(
            _RHO_BPB_QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )
        micros = rounded * _RHO_BPB_SCALE_DECIMAL
    return int(micros)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def compute_event_ledger_sha256_v2(attempts: tuple[Any, ...]) -> str:
    """Canonical ordered attempt-ledger identity used by every v2 mint."""

    return hashlib.sha256(
        canonical_json_bytes(
            tuple(
                {
                    "attempt_id": attempt.attempt_id,
                    "receipt_sha256": attempt.receipt_sha256,
                }
                for attempt in attempts
            )
        )
    ).hexdigest()


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
class A2FirstFitGroupReceiptV2:
    """One A2 first-fit-continuation group, without A1 prefix-floor fields."""

    stream: str
    stratum: str
    target_bytes: int
    realized_bytes: int
    deficit_bytes: int
    document_count: int
    ordered_raw_content_ids_sha256: str

    def __post_init__(self) -> None:
        if self.stream not in ("T", "H"):
            raise ValueError("first-fit stream must be T or H")
        if self.stratum not in GTOK_STRATA:
            raise ValueError("first-fit group has an unregistered stratum")
        targets = dict(
            GTOK_SCREEN_TRAIN_STRATUM_TARGETS
            if self.stream == "T"
            else GTOK_SCREEN_HELDOUT_STRATUM_TARGETS
        )
        for name, minimum in (
            ("target_bytes", 1),
            ("realized_bytes", 1),
            ("deficit_bytes", 0),
            ("document_count", 1),
        ):
            _require_exact_int(getattr(self, name), name, minimum=minimum)
        if self.target_bytes != targets[self.stratum]:
            raise ValueError("first-fit target differs from the governed target")
        if self.target_bytes - self.realized_bytes != self.deficit_bytes:
            raise ValueError("first-fit deficit must equal target minus realized")
        if Fraction(self.deficit_bytes, self.target_bytes) > GTOK_STRATUM_TOLERANCE:
            raise ValueError("first-fit group exceeds the 0.5 percent tolerance")
        _require_sha256(
            self.ordered_raw_content_ids_sha256,
            "ordered_raw_content_ids_sha256",
        )


@dataclass(frozen=True)
class A2FirstFitScreenReceiptV2:
    """The eight canonical A2 first-fit groups and physical T/H identities."""

    groups: tuple[A2FirstFitGroupReceiptV2, ...]
    training_framed_stream_sha256: str
    heldout_framed_stream_sha256: str
    document_overlap_count: int
    cluster_overlap_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.groups, tuple) or any(
            not isinstance(row, A2FirstFitGroupReceiptV2) for row in self.groups
        ):
            raise TypeError("first-fit groups must be a tuple of group receipts")
        expected = tuple(
            (stream, stratum)
            for stream in ("T", "H")
            for stratum in GTOK_STRATA
        )
        observed = tuple((row.stream, row.stratum) for row in self.groups)
        if observed != expected:
            raise ValueError("first-fit groups must be the exact canonical eight")
        _require_sha256(
            self.training_framed_stream_sha256,
            "training_framed_stream_sha256",
        )
        _require_sha256(
            self.heldout_framed_stream_sha256,
            "heldout_framed_stream_sha256",
        )
        for name in ("document_overlap_count", "cluster_overlap_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) != 0:
                raise ValueError(f"{name} must be exact zero")

    @property
    def training(self) -> tuple[A2FirstFitGroupReceiptV2, ...]:
        return tuple(row for row in self.groups if row.stream == "T")

    @property
    def heldout(self) -> tuple[A2FirstFitGroupReceiptV2, ...]:
        return tuple(row for row in self.groups if row.stream == "H")

    @property
    def training_target_bytes(self) -> int:
        return sum(row.target_bytes for row in self.training)

    @property
    def training_realized_bytes(self) -> int:
        return sum(row.realized_bytes for row in self.training)

    @property
    def heldout_target_bytes(self) -> int:
        return sum(row.target_bytes for row in self.heldout)

    @property
    def heldout_realized_bytes(self) -> int:
        return sum(row.realized_bytes for row in self.heldout)

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_a2_first_fit_screen", self)


@dataclass(frozen=True)
class FrozenScreenCorpusV2:
    """P-B-frozen corpus and the revalidated A2 first-fit physical view."""

    full_corpus_manifest_sha256: str
    screen_submanifest_sha256: str
    d6_physical_evidence_sha256: str
    corpus_freeze_receipt_sha256: str
    d1_d6_gate_bundle_sha256: str
    decontamination_receipt_sha256: str
    first_fit: A2FirstFitScreenReceiptV2

    def __post_init__(self) -> None:
        for name in (
            "full_corpus_manifest_sha256",
            "screen_submanifest_sha256",
            "d6_physical_evidence_sha256",
            "corpus_freeze_receipt_sha256",
            "d1_d6_gate_bundle_sha256",
            "decontamination_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.first_fit, A2FirstFitScreenReceiptV2):
            raise TypeError("first_fit must be an A2FirstFitScreenReceiptV2")
        if self.first_fit.training_target_bytes != GTOK_TRAINING_BYTE_BUDGET:
            raise ValueError("T target must remain exactly 4,000,000,000 bytes")
        if self.first_fit.heldout_target_bytes != GTOK_HELDOUT_BYTE_TARGET:
            raise ValueError("H target must remain exactly 80,000,000 bytes")

    @property
    def training_target_bytes(self) -> int:
        return self.first_fit.training_target_bytes

    @property
    def training_realized_bytes(self) -> int:
        return self.first_fit.training_realized_bytes

    @property
    def heldout_target_bytes(self) -> int:
        return self.first_fit.heldout_target_bytes

    @property
    def heldout_realized_bytes(self) -> int:
        return self.first_fit.heldout_realized_bytes

    @property
    def training_stream_sha256(self) -> str:
        return self.first_fit.training_framed_stream_sha256

    @property
    def heldout_stream_sha256(self) -> str:
        return self.first_fit.heldout_framed_stream_sha256

    @property
    def heldout_denominator_signature(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (row.stratum, row.realized_bytes) for row in self.first_fit.heldout
        )

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
    training_runtime_receipt_sha256: str
    code_closure_receipt_sha256: str
    compute_attempt_id: str
    measured_a100_microseconds: int
    measured_flops: int
    optimizer: FlatAdamWRecipe
    observations: tuple[BpbMilestoneReceiptV2, ...]
    gpu_uuid_provenance: str | None = None
    model_topology_sha256: str = GTOK_PROXY_TOPOLOGY_SHA256
    executing_block_count: int = 10
    checkpoint_retained: bool = False
    sealed_data_consumed: bool = False
    byte_round_trip_passed: bool = True
    stream_bytes: int | None = None
    stream_docs: int | None = None
    stream_tokens: int | None = None
    trained_tokens: int | None = None
    dropped_tokens: int | None = None
    trained_bytes: int | None = None
    dropped_bytes: int | None = None
    trained_docs_full: int | None = None
    boundary_doc_id: str | None = None
    boundary_doc_consumed_tokens: int | None = None
    dropped_docs: int | None = None

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
            "training_runtime_receipt_sha256",
            "code_closure_receipt_sha256",
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
        if self.gpu_uuid_provenance is not None and (
            not isinstance(self.gpu_uuid_provenance, str)
            or not self.gpu_uuid_provenance.startswith("GPU-")
            or len(self.gpu_uuid_provenance) <= 4
        ):
            raise ValueError("GPU UUID provenance must be an NVIDIA GPU UUID")
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
        if not steps[0] < steps[1] < steps[2]:
            raise ValueError("milestone optimizer steps must preserve execution order")
        observed_bytes = tuple(item.training_raw_bytes for item in self.observations)
        if not observed_bytes[0] < observed_bytes[1] < observed_bytes[2]:
            raise ValueError("milestone raw-byte boundaries must preserve execution order")
        accounting_names = (
            "stream_bytes",
            "stream_docs",
            "stream_tokens",
            "trained_tokens",
            "dropped_tokens",
            "trained_bytes",
            "dropped_bytes",
            "trained_docs_full",
            "dropped_docs",
        )
        accounting_values = tuple(getattr(self, name) for name in accounting_names)
        exact_accounting = any(value is not None for value in accounting_values) or any(
            value is not None
            for value in (self.boundary_doc_id, self.boundary_doc_consumed_tokens)
        )
        if exact_accounting:
            if any(value is None for value in accounting_values):
                raise ValueError("run stream accounting is incomplete")
            for name in accounting_names:
                value = getattr(self, name)
                if type(value) is not int or value < 0:
                    raise ValueError(f"{name} must be a non-negative exact integer")
            assert self.stream_bytes is not None
            assert self.stream_docs is not None
            assert self.stream_tokens is not None
            assert self.trained_tokens is not None
            assert self.dropped_tokens is not None
            assert self.trained_bytes is not None
            assert self.dropped_bytes is not None
            assert self.trained_docs_full is not None
            assert self.dropped_docs is not None
            if self.trained_tokens < 1 or self.trained_tokens % (256 * 2_048):
                raise ValueError("trained tokens must contain only complete global batches")
            if self.stream_tokens != self.trained_tokens + self.dropped_tokens:
                raise ValueError("run stream token accounting does not close")
            if not 0 <= self.dropped_tokens < 256 * 2_048:
                raise ValueError("run dropped tokens must be one partial global-batch suffix")
            if self.stream_bytes != self.trained_bytes + self.dropped_bytes:
                raise ValueError("run stream byte accounting does not close")
            if self.terminal.training_raw_bytes != self.trained_bytes:
                raise ValueError("terminal BPB training bytes must equal the trained prefix")
            quarter, half, terminal = self.observations
            if not (
                4 * quarter.previous_training_raw_bytes
                < self.trained_bytes
                <= 4 * quarter.training_raw_bytes
            ):
                raise ValueError(
                    "quarter BPB must be the first batch crossing one quarter of trained bytes"
                )
            if not (
                2 * half.previous_training_raw_bytes
                < self.trained_bytes
                <= 2 * half.training_raw_bytes
            ):
                raise ValueError(
                    "half BPB must be the first batch crossing one half of trained bytes"
                )
            if terminal.optimizer_step != self.trained_tokens // (256 * 2_048):
                raise ValueError("terminal BPB step must equal the final executed full batch")
            has_boundary = self.boundary_doc_id is not None
            if has_boundary is not (self.boundary_doc_consumed_tokens is not None):
                raise ValueError("run boundary identity and token count must appear together")
            if has_boundary:
                assert self.boundary_doc_id is not None
                assert self.boundary_doc_consumed_tokens is not None
                if (
                    len(self.boundary_doc_id) != 40
                    or any(character not in "0123456789abcdef" for character in self.boundary_doc_id)
                ):
                    raise ValueError("run boundary document ID must be lowercase SHA-1")
                if self.boundary_doc_consumed_tokens < 1:
                    raise ValueError("run boundary document must consume at least one token")
            if self.stream_docs != (
                self.trained_docs_full + self.dropped_docs + int(has_boundary)
            ):
                raise ValueError("run stream document accounting does not close")

    @property
    def terminal(self) -> BpbMilestoneReceiptV2:
        return self.observations[-1]

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_run", self)

    @property
    def has_exact_stream_accounting(self) -> bool:
        return self.stream_tokens is not None

    @property
    def dropped_token_fraction(self) -> float | None:
        if self.stream_tokens is None or self.dropped_tokens is None:
            return None
        return self.dropped_tokens / self.stream_tokens


@dataclass(frozen=True)
class ArmCalibrationProjectionV2:
    """Literal 20-warmup/80-measured calibration and exact arm projection."""

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
    charged_calibration_a100_microseconds: int | None = None
    measured_heldout_evaluation_a100_microseconds: int = 0
    heldout_evaluations_per_full_run: int = 0
    measured_output_surface_a100_microseconds: int = 0
    output_surface_benchmarks_per_full_run: int = 0
    projection_source: str = "standalone_calibration"
    projection_source_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.scope not in GTOK_COMPUTE_SCOPES:
            raise ValueError("calibration uses an unregistered compute scope")
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("calibration uses an unregistered vocabulary arm")
        if not isinstance(self.calibration_attempt_id, str) or not (
            self.calibration_attempt_id
        ):
            raise ValueError("calibration_attempt_id must be nonempty")
        if self.projection_source not in {
            "standalone_calibration",
            "completed_base_calibration",
        }:
            raise ValueError("arm projection uses an unregistered evidence source")
        inherited = self.projection_source == "completed_base_calibration"
        if inherited:
            if self.scope != "confirmation":
                raise ValueError("only confirmation may inherit completed base calibration")
            _require_sha256(
                str(self.projection_source_receipt_sha256),
                "projection_source_receipt_sha256",
            )
        elif self.projection_source_receipt_sha256 is not None:
            raise ValueError("standalone calibration may not claim an inherited source")
        _require_exact_int(self.calibration_steps, "calibration_steps", minimum=1)
        if self.calibration_steps > GTOK_CALIBRATION_MAX_STEPS:
            raise ValueError("an A2-R6 calibration burst may not exceed 100 steps")
        if self.calibration_steps != GTOK_CALIBRATION_MAX_STEPS:
            raise ValueError("A2 calibration must execute exactly 20 warmup + 80 measured steps")
        for name in (
            "measured_tokens",
            "measured_a100_microseconds",
            "planned_tokens_per_run",
            "projected_run_a100_microseconds",
        ):
            _require_exact_int(getattr(self, name), name, minimum=1)
        training_projection = _ceil_div(
            self.measured_a100_microseconds * self.planned_tokens_per_run,
            self.measured_tokens,
        )
        _require_exact_int(
            self.measured_heldout_evaluation_a100_microseconds,
            "measured_heldout_evaluation_a100_microseconds",
        )
        _require_exact_int(
            self.heldout_evaluations_per_full_run,
            "heldout_evaluations_per_full_run",
        )
        if self.heldout_evaluations_per_full_run not in (0, 3):
            raise ValueError("full-run projection must bind all three H evaluations")
        if bool(self.measured_heldout_evaluation_a100_microseconds) != bool(
            self.heldout_evaluations_per_full_run
        ):
            raise ValueError("H-evaluation projection evidence is incomplete")
        _require_exact_int(
            self.measured_output_surface_a100_microseconds,
            "measured_output_surface_a100_microseconds",
        )
        _require_exact_int(
            self.output_surface_benchmarks_per_full_run,
            "output_surface_benchmarks_per_full_run",
        )
        if self.output_surface_benchmarks_per_full_run not in (0, 1):
            raise ValueError("full-run projection may price one output-surface panel")
        if bool(self.measured_output_surface_a100_microseconds) != bool(
            self.output_surface_benchmarks_per_full_run
        ):
            raise ValueError("output-surface projection evidence is incomplete")
        expected_projection = training_projection + (
            self.measured_heldout_evaluation_a100_microseconds
            * self.heldout_evaluations_per_full_run
        ) + (
            self.measured_output_surface_a100_microseconds
            * self.output_surface_benchmarks_per_full_run
        )
        if self.projected_run_a100_microseconds != expected_projection:
            raise ValueError("per-arm projection must be computed from measured tokens/sec")
        if self.full_run_count != GTOK_SEED_COUNT:
            raise ValueError("each projected arm must price the registered two-seed panel")
        if self.full_run_attempt_count_at_projection != 0:
            raise GTokV2Stop("preflight projection must precede every full-run launch")
        if self.charged_calibration_a100_microseconds is not None:
            _require_exact_int(
                self.charged_calibration_a100_microseconds,
                "charged_calibration_a100_microseconds",
            )
            if inherited and self.charged_calibration_a100_microseconds != 0:
                raise ValueError("inherited base calibration adds no confirmation spend")
            if not inherited and self.charged_calibration_a100_microseconds < self.measured_a100_microseconds:
                raise ValueError("charged calibration time cannot omit the warmup window")
        elif inherited:
            raise ValueError("inherited base calibration must record zero added spend")

    @property
    def charged_a100_microseconds(self) -> int:
        """All 100 steps are charged; legacy fixtures default to their measured value."""

        return (
            self.measured_a100_microseconds
            if self.charged_calibration_a100_microseconds is None
            else self.charged_calibration_a100_microseconds
        )

    @property
    def projected_scope_a100_microseconds(self) -> int:
        return self.charged_a100_microseconds + (
            self.full_run_count * self.projected_run_a100_microseconds
        )

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_arm_calibration_projection", self)


@dataclass(frozen=True)
class PrecalibrationReplayAttemptReceiptV2:
    """One durably metered physical replica in the pre-calibration replay gate.

    Replay attempts deliberately have their own receipt shape.  A first replica
    is the measurement burst which creates the pair projection, so pretending it
    already had an arm-calibration projection would be circular.  A second
    replica must, by contrast, join the measured pair projection and its exact
    2x watchdog.  Failed and hard-killed retries remain in this ledger.
    """

    attempt_id: str
    scope: str
    kind: str
    vocab_size: int
    terminal_rows: int
    representative_seed: int
    replica_index: int
    consumed_a100_microseconds: int
    status: str
    replay_plan_binding_sha256: str
    replay_pair_projection_sha256: str | None = None
    projected_replica_a100_microseconds: int | None = None
    watchdog_limit_a100_microseconds: int | None = None
    hard_abort_issued: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("replay compute attempt_id must be nonempty")
        if self.scope != "base_screen" or self.kind != "determinism_replay":
            raise ValueError("determinism replay attempts belong to the base preflight")
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("replay attempt uses an unregistered vocabulary")
        _require_exact_int(self.terminal_rows, "terminal_rows", minimum=1)
        if type(self.representative_seed) is not int:
            raise TypeError("representative_seed must be an exact integer")
        if self.replica_index not in (0, 1):
            raise ValueError("replay attempt replica index must be zero or one")
        _require_exact_int(
            self.consumed_a100_microseconds,
            "consumed_a100_microseconds",
            minimum=1,
        )
        if self.status not in ("completed", "failed", "preempted", "aborted_watchdog"):
            raise ValueError("durable replay attempt has an unregistered terminal status")
        _require_sha256(
            self.replay_plan_binding_sha256,
            "replay_plan_binding_sha256",
        )
        projection_values = (
            self.replay_pair_projection_sha256,
            self.projected_replica_a100_microseconds,
            self.watchdog_limit_a100_microseconds,
        )
        if self.replica_index == 1:
            if any(value is None for value in projection_values):
                raise ValueError("second replay replica must join its measured projection")
            _require_sha256(
                str(self.replay_pair_projection_sha256),
                "replay_pair_projection_sha256",
            )
            _require_exact_int(
                int(self.projected_replica_a100_microseconds),
                "projected_replica_a100_microseconds",
                minimum=1,
            )
            if self.watchdog_limit_a100_microseconds != (
                GTOK_PER_RUN_WATCHDOG_MULTIPLIER
                * int(self.projected_replica_a100_microseconds)
            ):
                raise ValueError("second replay watchdog must be exactly 2x projection")
        elif any(value is not None for value in projection_values):
            raise ValueError("first replay burst may not claim a prior pair projection")
        if type(self.hard_abort_issued) is not bool:
            raise TypeError("hard_abort_issued must be an exact bool")
        if self.status == "aborted_watchdog":
            if self.replica_index != 1 or not self.hard_abort_issued:
                raise ValueError("only a second replay replica may hard-abort at its watchdog")
        elif self.hard_abort_issued:
            raise ValueError("replay hard abort requires watchdog-aborted status")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256(
            "weft1_gtok_v2_precalibration_replay_compute_attempt",
            self,
        )


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
    recovered_attempt_a100_microseconds: int = 0
    precalibration_replay_attempts: tuple[PrecalibrationReplayAttemptReceiptV2, ...] = ()
    precalibration_replay_plan_set_sha256: str | None = None
    precalibration_replay_receipt_sha256s: tuple[str, ...] = ()
    precalibration_replay_authority_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.scope not in GTOK_COMPUTE_SCOPES:
            raise ValueError("preflight uses an unregistered compute scope")
        _require_exact_int(
            self.prior_campaign_a100_microseconds,
            "prior_campaign_a100_microseconds",
        )
        _require_exact_int(
            self.recovered_attempt_a100_microseconds,
            "recovered_attempt_a100_microseconds",
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
        replay_bound = bool(self.precalibration_replay_attempts)
        if replay_bound:
            if self.scope != "base_screen" or any(
                not isinstance(item, PrecalibrationReplayAttemptReceiptV2)
                for item in self.precalibration_replay_attempts
            ):
                raise TypeError("base preflight replay attempts require typed receipts")
            for name, value in (
                (
                    "precalibration_replay_plan_set_sha256",
                    self.precalibration_replay_plan_set_sha256,
                ),
                (
                    "precalibration_replay_authority_sha256",
                    self.precalibration_replay_authority_sha256,
                ),
            ):
                _require_sha256(str(value), name)
            if not self.precalibration_replay_receipt_sha256s:
                raise ValueError("base preflight must bind every green replay pair receipt")
            for value in self.precalibration_replay_receipt_sha256s:
                _require_sha256(value, "precalibration_replay_receipt_sha256s")
            if len(set(self.precalibration_replay_receipt_sha256s)) != len(
                self.precalibration_replay_receipt_sha256s
            ):
                raise ValueError("pre-calibration replay receipt identities must be unique")
        elif any(
            value
            for value in (
                self.precalibration_replay_plan_set_sha256,
                self.precalibration_replay_receipt_sha256s,
                self.precalibration_replay_authority_sha256,
            )
        ):
            raise ValueError("pre-calibration replay binding is incomplete")
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
        expected = (
            self.prior_campaign_a100_microseconds
            + self.recovered_attempt_a100_microseconds
            + sum(
                item.consumed_a100_microseconds
                for item in self.precalibration_replay_attempts
            )
            + sum(
            item.projected_scope_a100_microseconds for item in self.calibrations
            )
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
    execution_plan_sha256: str | None = None
    planned_compute_token_slots: int | None = None
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
        confirmation_full_run = self.scope == "confirmation" and self.kind == "full_run"
        if confirmation_full_run:
            _require_sha256(str(self.execution_plan_sha256), "execution_plan_sha256")
            _require_exact_int(
                self.planned_compute_token_slots,
                "planned_compute_token_slots",
                minimum=1,
            )
        elif (
            self.execution_plan_sha256 is not None
            or self.planned_compute_token_slots is not None
        ):
            raise ValueError(
                "only a confirmation full-run attempt may bind an execution plan"
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
    attempts: tuple[
        ComputeAttemptReceiptV2 | PrecalibrationReplayAttemptReceiptV2, ...
    ]
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
        if any(
            not isinstance(
                item,
                (ComputeAttemptReceiptV2, PrecalibrationReplayAttemptReceiptV2),
            )
            for item in self.attempts
        ):
            raise TypeError("attempts must contain registered compute-attempt receipts")
        if any(item.scope != self.scope for item in self.attempts):
            raise ValueError("compute attempt scope differs from its campaign")
        attempt_ids = tuple(item.attempt_id for item in self.attempts)
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("campaign compute attempt IDs must be unique")
        projections = {item.vocab_size: item for item in self.preflight.calibrations}
        ordinary_attempts = tuple(
            item for item in self.attempts if isinstance(item, ComputeAttemptReceiptV2)
        )
        replay_attempts = tuple(
            item
            for item in self.attempts
            if isinstance(item, PrecalibrationReplayAttemptReceiptV2)
        )
        if replay_attempts != self.preflight.precalibration_replay_attempts:
            raise ValueError("campaign replay meter differs from its preflight authority")
        if any(item.vocab_size not in projections for item in ordinary_attempts):
            raise ValueError("compute attempt lacks a per-arm calibration projection")
        for attempt in ordinary_attempts:
            projection = projections[attempt.vocab_size]
            if attempt.calibration_projection_sha256 != projection.receipt_sha256:
                raise ValueError("compute attempt does not join its arm projection")
            if self.scope == "confirmation" and attempt.kind == "full_run":
                assert attempt.planned_compute_token_slots is not None
                expected_projected_run = _ceil_div(
                    projection.measured_a100_microseconds
                    * attempt.planned_compute_token_slots,
                    projection.measured_tokens,
                ) + (
                    projection.measured_heldout_evaluation_a100_microseconds
                    * projection.heldout_evaluations_per_full_run
                ) + (
                    projection.measured_output_surface_a100_microseconds
                    * projection.output_surface_benchmarks_per_full_run
                )
            else:
                expected_projected_run = projection.projected_run_a100_microseconds
            if attempt.projected_run_a100_microseconds != expected_projected_run:
                raise ValueError(
                    "compute attempt projection differs from its exact planned horizon"
                )
        calibration_attempts = tuple(
            item for item in ordinary_attempts if item.kind == "calibration"
        )
        calibration_by_id = {item.attempt_id: item for item in calibration_attempts}
        for projection in self.preflight.calibrations:
            if projection.projection_source == "completed_base_calibration":
                if any(
                    item.vocab_size == projection.vocab_size
                    for item in calibration_attempts
                ):
                    raise ValueError(
                        "inherited confirmation projection may not add calibration attempts"
                    )
                continue
            attempt = calibration_by_id.get(projection.calibration_attempt_id)
            if (
                attempt is None
                or attempt.vocab_size != projection.vocab_size
                or attempt.consumed_a100_microseconds
                != projection.charged_a100_microseconds
                or attempt.status != "completed"
            ):
                raise ValueError("calibration attempt is absent from the cumulative ledger")
        for projection in self.preflight.calibrations:
            completed_for_arm = tuple(
                item
                for item in calibration_attempts
                if item.vocab_size == projection.vocab_size
                and item.status == "completed"
            )
            expected_completed = (
                0
                if projection.projection_source == "completed_base_calibration"
                else 1
            )
            if len(completed_for_arm) != expected_completed:
                raise ValueError(
                    "arm calibration attempts differ from their projection source"
                )
        _require_sha256(self.event_ledger_sha256, "event_ledger_sha256")
        if self.event_ledger_sha256 != compute_event_ledger_sha256_v2(
            self.attempts
        ):
            raise ValueError("compute event ledger differs from ordered attempt receipts")
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
            for item in ordinary_attempts
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
    training_runtime_receipt_sha256: str
    code_closure_receipt_sha256: str
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
            "training_runtime_receipt_sha256": runs[0].training_runtime_receipt_sha256,
            "code_closure_receipt_sha256": runs[0].code_closure_receipt_sha256,
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
    if any(getattr(attempt, "status", None) == "aborted_watchdog" for attempt in compute.attempts):
        raise GTokV2Stop("base G-TOK evidence contains a watchdog abort")
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
    if len({run.training_runtime_receipt_sha256 for run in runs}) != 1:
        raise ValueError("every base run must use one exact training runtime")
    if len({run.code_closure_receipt_sha256 for run in runs}) != 1:
        raise ValueError("every base run must use one exact behavior-bearing code closure")
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
        if not run.has_exact_stream_accounting:
            raise ValueError("authoritative matrix requires exact Q2 stream accounting")
        if run.stream_bytes != corpus.training_realized_bytes:
            raise ValueError("declared run stream must equal manifested realized T")
        if run.terminal.training_raw_bytes != run.trained_bytes:
            raise ValueError("terminal milestone must equal the trained byte prefix")
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
    rho_bpb_micros: int
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
        if self.rho_bpb_micros != _rho_bpb_micros(expected_mean):
            raise ValueError("arm rho does not equal the rounded float64 two-seed mean")
        if self.sample_sd != expected_sd:
            raise ValueError("arm SD does not equal the unrounded two-seed sample SD")

    @property
    def rho_bpb(self) -> float:
        """Return the authoritative reported rho in BPB units."""

        return self.rho_bpb_micros / GTOK_RHO_BPB_SCALE


@dataclass(frozen=True)
class SelectionComparisonV2:
    comparison_index: int
    incumbent_vocab_before: int
    challenger_vocab: int
    incumbent_rho_bpb_micros: int
    challenger_rho_bpb_micros: int
    incumbent_sample_sd: float
    challenger_sample_sd: float
    s_hat: float
    delta_bpb_micros: int
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
            "incumbent_rho_bpb_micros",
            "challenger_rho_bpb_micros",
        ):
            _require_exact_int(getattr(self, name), name, minimum=1)
        if type(self.delta_bpb_micros) is not int:
            raise ValueError("delta_bpb_micros must be an exact integer")
        for name in (
            "incumbent_sample_sd",
            "challenger_sample_sd",
            "s_hat",
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
        expected_delta = (
            self.incumbent_rho_bpb_micros - self.challenger_rho_bpb_micros
        )
        if self.s_hat != expected_s_hat:
            raise ValueError("s_hat must use the pairwise equal-n pooled sample SD")
        if self.delta_bpb_micros != expected_delta:
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

    @property
    def incumbent_rho_bpb(self) -> float:
        return self.incumbent_rho_bpb_micros / GTOK_RHO_BPB_SCALE

    @property
    def challenger_rho_bpb(self) -> float:
        return self.challenger_rho_bpb_micros / GTOK_RHO_BPB_SCALE

    @property
    def delta_bpb(self) -> float:
        return self.delta_bpb_micros / GTOK_RHO_BPB_SCALE


@dataclass(frozen=True)
class GTokSelectionReceiptV2:
    matrix_receipt_sha256: str
    selection_confirmation_authority_sha256: str
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
        if (
            self.selection_confirmation_authority_sha256
            != GTOK_SELECTION_CONFIRMATION_AUTHORITY_SHA256
        ):
            raise ValueError("selection uses a different forward semantics authority")
        if (
            self.selector_literal_binding_sha256
            != GTOK_SELECTION_CONFIRMATION_LITERAL_BINDING_SHA256_V2
        ):
            raise ValueError("selection uses a different forward literal binding")
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
                rho_bpb_micros=_rho_bpb_micros(mean),
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
        delta_micros = (
            incumbent_stats.rho_bpb_micros - challenger_stats.rho_bpb_micros
        )
        delta = delta_micros / GTOK_RHO_BPB_SCALE
        displaced = delta > 3.0 * s_hat
        next_incumbent = challenger if displaced else incumbent
        comparisons.append(
            SelectionComparisonV2(
                comparison_index=len(comparisons),
                incumbent_vocab_before=incumbent,
                challenger_vocab=challenger,
                incumbent_rho_bpb_micros=incumbent_stats.rho_bpb_micros,
                challenger_rho_bpb_micros=challenger_stats.rho_bpb_micros,
                incumbent_sample_sd=incumbent_stats.sample_sd,
                challenger_sample_sd=challenger_stats.sample_sd,
                s_hat=s_hat,
                delta_bpb_micros=delta_micros,
                two_s_hat=2.0 * s_hat,
                three_s_hat=3.0 * s_hat,
                tie_diagnostic=abs(delta) < 2.0 * s_hat,
                displaced=displaced,
                incumbent_vocab_after=next_incumbent,
            )
        )
        incumbent = next_incumbent
    runner_up = min(
        (row for row in statistics if row.vocab_size != incumbent),
        key=lambda row: (row.rho_bpb_micros, row.vocab_size),
    ).vocab_size
    confirmation_pair = (incumbent, runner_up)
    if any(vocab_size not in allowed for vocab_size in confirmation_pair):
        raise GTokV2Stop("a compute-confirmation arm fails the rung-B admissibility guard")
    return GTokSelectionReceiptV2(
        matrix_receipt_sha256=matrix.receipt_sha256,
        selection_confirmation_authority_sha256=(
            GTOK_SELECTION_CONFIRMATION_AUTHORITY_SHA256
        ),
        selector_literal_binding_sha256=(
            GTOK_SELECTION_CONFIRMATION_LITERAL_BINDING_SHA256_V2
        ),
        seed_specific_orders=seed_orders,
        agreed_strict_terminal_order=agreed_order,
        arm_statistics=statistics,
        admissibility=admissibility,
        comparisons=tuple(comparisons),
        selected_vocab_size=incumbent,
        compute_confirmation_pair=confirmation_pair,
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
    """One fresh max-FLOP confirmation run, paired to a base seed slot."""

    vocab_size: int
    seed_slot: int
    registry_key: str
    seed: int
    initialization_seed: int
    data_order_seed: int
    data_order_sha256: str
    confirmation_order_receipt_sha256: str
    physical_d6_evidence_sha256: str
    document_multiset_sha256: str
    framed_payload_sha256: str
    execution_plan_sha256: str
    training_plan_sha256: str
    base_run_receipt_sha256: str
    compute_attempt_id: str
    common_flop_budget: int
    measured_flops: int
    heldout_stream_sha256: str
    observations: tuple[BpbMilestoneReceiptV2, ...]
    measured_a100_microseconds: int
    training_runtime_receipt_sha256: str
    code_closure_receipt_sha256: str
    stream_bytes: int
    stream_docs: int
    stream_tokens: int
    trained_tokens: int
    dropped_tokens: int
    trained_bytes: int
    dropped_bytes: int
    trained_docs_full: int
    boundary_doc_id: str | None
    boundary_doc_consumed_tokens: int | None
    dropped_docs: int
    gpu_uuid_provenance: str | None = None
    checkpoint_retained: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("confirmation uses an unregistered vocabulary")
        if self.seed_slot not in (0, 1):
            raise ValueError("confirmation seed slot must be exactly 0 or 1")
        expected_key = f"gtok.confirm.{self.vocab_size}.{self.seed_slot}"
        if self.registry_key != expected_key:
            raise ValueError("confirmation seed registry key drifted")
        expected_seed = int.from_bytes(
            hashlib.sha256(expected_key.encode("ascii")).digest()[:8],
            byteorder="big",
        )
        if self.seed != expected_seed:
            raise ValueError("confirmation seed differs from its direct SHA-256 root")
        if self.initialization_seed != derive_module_seed(
            A2_CAMPAIGN_ROOT_SEED,
            f"gtok.init.shared.{self.seed}",
        ):
            raise ValueError("confirmation initialization seed left the A2 role tree")
        if self.data_order_seed != derive_module_seed(
            A2_CAMPAIGN_ROOT_SEED,
            f"gtok.data.shared.{self.seed}",
        ):
            raise ValueError("confirmation data-order seed left the A2 role tree")
        _require_sha256(self.data_order_sha256, "data_order_sha256")
        for name in (
            "confirmation_order_receipt_sha256",
            "physical_d6_evidence_sha256",
            "document_multiset_sha256",
            "framed_payload_sha256",
            "execution_plan_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_sha256(self.training_plan_sha256, "training_plan_sha256")
        _require_sha256(self.base_run_receipt_sha256, "base_run_receipt_sha256")
        if not isinstance(self.compute_attempt_id, str) or not self.compute_attempt_id:
            raise ValueError("confirmation compute_attempt_id must be nonempty")
        _require_exact_int(self.common_flop_budget, "common_flop_budget", minimum=1)
        _require_exact_int(self.measured_flops, "measured_flops", minimum=1)
        if 100 * abs(self.measured_flops - self.common_flop_budget) > self.common_flop_budget:
            raise ValueError("compute confirmation must remain within one percent of F_star")
        _require_sha256(self.heldout_stream_sha256, "heldout_stream_sha256")
        if not isinstance(self.observations, tuple) or tuple(
            item.label for item in self.observations
        ) != GTOK_MILESTONE_LABELS:
            raise ValueError("confirmation requires quarter, half, and terminal observations")
        if any(not isinstance(item, BpbMilestoneReceiptV2) for item in self.observations):
            raise TypeError("confirmation observations must be BPB milestone receipts")
        if any(
            item.heldout_stream_sha256 != self.heldout_stream_sha256
            for item in self.observations
        ):
            raise ValueError("confirmation observation held-out identity drifted")
        steps = tuple(item.optimizer_step for item in self.observations)
        if not steps[0] < steps[1] < steps[2]:
            raise ValueError("confirmation BPB checkpoint steps must be strictly increasing")
        observed_bytes = tuple(item.training_raw_bytes for item in self.observations)
        if not observed_bytes[0] < observed_bytes[1] < observed_bytes[2]:
            raise ValueError("confirmation BPB checkpoint bytes must be strictly increasing")
        _require_exact_int(
            self.measured_a100_microseconds,
            "measured_a100_microseconds",
            minimum=1,
        )
        _require_sha256(
            self.training_runtime_receipt_sha256,
            "training_runtime_receipt_sha256",
        )
        _require_sha256(
            self.code_closure_receipt_sha256,
            "code_closure_receipt_sha256",
        )
        if self.gpu_uuid_provenance is not None and (
            not isinstance(self.gpu_uuid_provenance, str)
            or not self.gpu_uuid_provenance.startswith("GPU-")
            or len(self.gpu_uuid_provenance) <= 4
        ):
            raise ValueError("confirmation GPU provenance must be an NVIDIA GPU UUID")
        if type(self.checkpoint_retained) is not bool:
            raise TypeError("checkpoint_retained must be an exact bool")
        if self.checkpoint_retained:
            raise ValueError("compute confirmation may not retain checkpoints")
        accounting_names = (
            "stream_bytes",
            "stream_docs",
            "stream_tokens",
            "trained_tokens",
            "dropped_tokens",
            "trained_bytes",
            "dropped_bytes",
            "trained_docs_full",
            "dropped_docs",
        )
        for name in accounting_names:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.stream_bytes < 1 or self.stream_docs < 1 or self.stream_tokens < 1:
            raise ValueError("confirmation stream accounting must be positive")
        if self.trained_tokens < 1 or self.trained_tokens % (256 * 2_048):
            raise ValueError("confirmation trains only complete global batches")
        if self.stream_tokens != self.trained_tokens + self.dropped_tokens:
            raise ValueError("confirmation stream token accounting does not close")
        if self.stream_bytes != self.trained_bytes + self.dropped_bytes:
            raise ValueError("confirmation stream byte accounting does not close")
        if self.terminal.training_raw_bytes != self.trained_bytes:
            raise ValueError("confirmation terminal BPB bytes differ from trained bytes")
        if self.terminal.optimizer_step != self.trained_tokens // (256 * 2_048):
            raise ValueError("confirmation terminal BPB step differs from trained tokens")
        quarter, half, _terminal = self.observations
        if not (
            4 * quarter.previous_training_raw_bytes
            < self.trained_bytes
            <= 4 * quarter.training_raw_bytes
        ):
            raise ValueError("confirmation quarter checkpoint is not the first byte crossing")
        if not (
            2 * half.previous_training_raw_bytes
            < self.trained_bytes
            <= 2 * half.training_raw_bytes
        ):
            raise ValueError("confirmation half checkpoint is not the first byte crossing")
        has_boundary = self.boundary_doc_id is not None
        if has_boundary is not (self.boundary_doc_consumed_tokens is not None):
            raise ValueError("confirmation boundary identity and token count must appear together")
        if has_boundary:
            assert self.boundary_doc_id is not None
            assert self.boundary_doc_consumed_tokens is not None
            if (
                len(self.boundary_doc_id) != 40
                or any(character not in "0123456789abcdef" for character in self.boundary_doc_id)
            ):
                raise ValueError("confirmation boundary document ID must be lowercase SHA-1")
            if self.boundary_doc_consumed_tokens < 1:
                raise ValueError("confirmation boundary document must consume at least one token")
        if self.stream_docs != (
            self.trained_docs_full + self.dropped_docs + int(has_boundary)
        ):
            raise ValueError("confirmation stream document accounting does not close")
        if not math.isfinite(self.pooled_bpb):
            raise ValueError("confirmation pooled BPB must remain finite")

    @property
    def terminal(self) -> BpbMilestoneReceiptV2:
        return self.observations[-1]

    @property
    def pooled_bpb(self) -> float:
        return self.terminal.pooled_bpb

    @property
    def denominator_signature(self) -> tuple[tuple[str, int], ...]:
        return self.terminal.denominator_signature

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_compute_confirmation_run", self)


@dataclass(frozen=True)
class ConfirmationFreshEvidenceJoinV2:
    """One successful fresh slot joined through Q3, execution, and lifecycle."""

    vocab_size: int
    seed_slot: int
    fresh_run_receipt_sha256: str
    confirmation_order_receipt_sha256: str
    physical_d6_evidence_sha256: str
    document_multiset_sha256: str
    ordered_raw_content_ids_sha256: str
    framed_payload_sha256: str
    order_document_count: int
    order_retained_text_bytes: int
    execution_plan_sha256: str
    training_plan_sha256: str
    compute_attempt_id: str
    terminal_lifecycle_event_sha256: str
    burst_receipt_sha256: str
    physical_flop_ledger_sha256: str

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS or self.seed_slot not in (0, 1):
            raise ValueError("confirmation evidence slot identity is invalid")
        for name in (
            "fresh_run_receipt_sha256",
            "confirmation_order_receipt_sha256",
            "physical_d6_evidence_sha256",
            "document_multiset_sha256",
            "ordered_raw_content_ids_sha256",
            "framed_payload_sha256",
            "execution_plan_sha256",
            "training_plan_sha256",
            "terminal_lifecycle_event_sha256",
            "burst_receipt_sha256",
            "physical_flop_ledger_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.compute_attempt_id, str) or not self.compute_attempt_id:
            raise ValueError("confirmation evidence requires its successful attempt ID")
        _require_exact_int(self.order_document_count, "order_document_count", minimum=1)
        _require_exact_int(
            self.order_retained_text_bytes,
            "order_retained_text_bytes",
            minimum=1,
        )

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_confirmation_fresh_join", self)


@dataclass(frozen=True)
class ConfirmationRetryEvidenceJoinV2:
    """One invalid-band attempt and the exact fresh successor it authorizes."""

    vocab_size: int
    seed_slot: int
    correction_ordinal: int
    failed_attempt_id: str
    failed_attempt_receipt_sha256: str
    failed_execution_plan_sha256: str
    failed_terminal_lifecycle_event_sha256: str
    invalid_physical_flop_ledger_sha256: str
    realized_flops: int
    target_flops: int
    prior_optimizer_steps: int
    retry_optimizer_steps: int
    retry_execution_plan_sha256: str
    retry_artifact_physical_sha256: str

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS or self.seed_slot not in (0, 1):
            raise ValueError("confirmation retry evidence slot identity is invalid")
        _require_exact_int(
            self.correction_ordinal,
            "correction_ordinal",
        )
        if not isinstance(self.failed_attempt_id, str) or not self.failed_attempt_id:
            raise ValueError("confirmation retry join requires its failed attempt ID")
        for name in (
            "failed_attempt_receipt_sha256",
            "failed_execution_plan_sha256",
            "failed_terminal_lifecycle_event_sha256",
            "invalid_physical_flop_ledger_sha256",
            "retry_execution_plan_sha256",
            "retry_artifact_physical_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in (
            "realized_flops",
            "target_flops",
            "prior_optimizer_steps",
            "retry_optimizer_steps",
        ):
            _require_exact_int(getattr(self, name), name, minimum=1)
        if self.retry_optimizer_steps != (
            self.target_flops * self.prior_optimizer_steps
        ) // self.realized_flops:
            raise ValueError("confirmation retry horizon differs from exact recomputation")
        if self.failed_execution_plan_sha256 == self.retry_execution_plan_sha256:
            raise ValueError("confirmation retry must change the execution plan identity")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_confirmation_retry_join", self)


@dataclass(frozen=True)
class ConfirmationLifecycleEventEvidenceV2:
    """Canonical lifecycle-event preimage carried into decision validation."""

    logical_attempt_id: str
    attempt_id: str
    scope: str
    kind: str
    phase: str
    charged_a100_microseconds: int
    terminal_status: str | None
    completion_payload: Mapping[str, Any] | None = None
    gpu_uuid_provenance: str | None = None
    offline_network_launch_receipt_sha256: str | None = None
    heartbeat_interval_a100_microseconds: int = 30_000_000
    checkpoint_retained: bool = False

    def __post_init__(self) -> None:
        for name in ("logical_attempt_id", "attempt_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.scope != "confirmation" or self.kind != "full_run":
            raise ValueError("confirmation closure lifecycle scope/kind drifted")
        if self.phase not in ("START", "HEARTBEAT", "TERMINAL"):
            raise ValueError("confirmation closure lifecycle phase is invalid")
        _require_exact_int(
            self.charged_a100_microseconds,
            "charged_a100_microseconds",
            minimum=1,
        )
        if self.phase == "TERMINAL":
            if self.terminal_status not in (
                "completed",
                "failed",
                "preempted",
                "aborted_watchdog",
            ):
                raise ValueError("confirmation terminal lifecycle status is invalid")
            if (self.terminal_status == "completed") != isinstance(
                self.completion_payload,
                Mapping,
            ):
                raise ValueError("confirmation completion payload/status mismatch")
        elif self.terminal_status is not None or self.completion_payload is not None:
            raise ValueError("nonterminal confirmation lifecycle evidence claims completion")
        if self.heartbeat_interval_a100_microseconds != 30_000_000:
            raise ValueError("confirmation lifecycle heartbeat cadence drifted")
        if self.checkpoint_retained is not False:
            raise ValueError("confirmation lifecycle evidence retained a checkpoint")
        if self.gpu_uuid_provenance is not None and (
            not isinstance(self.gpu_uuid_provenance, str)
            or not self.gpu_uuid_provenance.startswith("GPU-")
        ):
            raise ValueError("confirmation lifecycle GPU provenance is invalid")
        if self.offline_network_launch_receipt_sha256 is not None:
            _require_sha256(
                self.offline_network_launch_receipt_sha256,
                "offline_network_launch_receipt_sha256",
            )

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_lifecycle_event", self)


_CONFIRMATION_BINDING_SHA256_V2 = (
    "1b2657cb0ad399bc7eaede8c5daa565edf3c0a02615ceedfd135f4c9ef8317d4"
)
_CONFIRMATION_EXECUTION_PLAN_FIELDS_V2 = frozenset(
    {
        "vocab_size",
        "seed_slot",
        "registry_key",
        "seed",
        "initialization_seed",
        "data_order_seed",
        "data_order_sha256",
        "confirmation_order_receipt_sha256",
        "physical_d6_evidence_sha256",
        "document_multiset_sha256",
        "framed_payload_sha256",
        "order_document_count",
        "order_retained_text_bytes",
        "target_flops",
        "arm_mean_flops",
        "byte_matched_optimizer_steps",
        "arm_flop_plan_sha256",
        "training_plan",
        "retry_of_realized_flops",
        "retry_of_optimizer_steps",
        "heldout_evaluation_steps",
    }
)
_CONFIRMATION_TRAINING_PLAN_FIELDS_V2 = frozenset(
    {
        "confirmation_order_receipt_sha256",
        "optimizer_steps",
        "global_batch_sequences",
        "sequence_length",
        "compute_token_slots",
        "valid_prediction_count",
        "trained_bytes",
        "trained_tokens",
        "trained_docs_full",
        "boundary_doc_id",
        "boundary_doc_consumed_tokens",
        "stream_bytes",
        "stream_tokens",
        "stream_docs",
        "dropped_bytes",
        "dropped_tokens",
        "dropped_docs",
        "packed_stream_sha256",
        "calibration_prefix_compute_token_slots",
        "calibration_prefix_valid_prediction_count",
        "calibration_prefix_realized_raw_bytes",
        "calibration_prefix_document_count",
        "calibration_prefix_packed_stream_sha256",
        "bpb_checkpoint_steps",
        "packing_binding_sha256",
        "calibration_prefix_steps",
    }
)
_CONFIRMATION_RETRY_ARTIFACT_FIELDS_V2 = frozenset(
    {
        "attempt",
        "attempt_receipt_sha256",
        "binding_sha256",
        "correction_ordinal",
        "failed_execution_plan_sha256",
        "failed_optimizer_steps",
        "failed_projected_run_a100_microseconds",
        "failed_terminal_lifecycle_event_sha256",
        "invalid_physical_flop_ledger",
        "invalid_physical_flop_ledger_sha256",
        "invalid_flop_ledger_receipt_sha256",
        "passed_burst_flop_receipt",
        "passed_burst_receipt_sha256",
        "passed_physical_burst_evidence_sha256",
        "realized_flops",
        "retry_execution_plan",
        "retry_execution_plan_sha256",
        "retry_projected_run_a100_microseconds",
        "retry_steps",
        "schema",
        "target_flops",
    }
)
_CONFIRMATION_ORDER_FIELDS_V4 = frozenset(
    {
        "confirmation_run_seed",
        "data_order_seed",
        "physical_d6_evidence_sha256",
        "document_multiset_sha256",
        "ordered_raw_content_ids_sha256",
        "framed_payload_sha256",
        "document_count",
        "retained_text_bytes",
        "order_key_domain",
        "schema",
    }
)
_CONFIRMATION_ORDER_SCHEMA_V4 = "weft1_gtok_confirmation_consumer_order_v4"
_CONFIRMATION_ORDER_KEY_DOMAIN_V4 = "WEFT-1/gtok-training-order/raw-content-id/v4"
_CONFIRMATION_BASE_FLOP_EVIDENCE_FIELDS_V2 = frozenset(
    {
        "vocab_size",
        "seed",
        "base_run_receipt_sha256",
        "base_compute_attempt_id",
        "flop_ledger_sha256",
        "steps",
        "measured_flops",
    }
)
_CONFIRMATION_BASE_STEP_FLOP_FIELDS_V2 = frozenset(
    {
        "optimizer_step",
        "batch_rows",
        "sequence_length",
        "optimizer_phase",
        "measured_flops",
    }
)
_CONFIRMATION_ARM_FLOP_PLAN_FIELDS_V2 = frozenset(
    {
        "vocab_size",
        "seeds",
        "base_flops",
        "base_flop_evidence_sha256s",
        "byte_matched_optimizer_steps",
        "arm_mean_flops",
        "target_flops",
        "planned_optimizer_steps",
    }
)
_CONFIRMATION_ATTEMPT_LAUNCH_FIELDS_V2 = frozenset(
    {
        "attempt_id",
        "binding_sha256",
        "calibration_projection_sha256",
        "execution_plan_sha256",
        "logical_attempt_id",
        "planned_compute_token_slots",
        "projected_run_a100_microseconds",
        "schema",
        "seed",
        "vocab_size",
        "watchdog_limit_a100_microseconds",
    }
)


@dataclass(frozen=True)
class ConfirmationOrderEnvelopeV2:
    """Canonical V4 frozen-T order preimage for one fresh confirmation slot."""

    payload: Mapping[str, Any]
    receipt_sha256: str
    physical_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping) or set(self.payload) != (
            _CONFIRMATION_ORDER_FIELDS_V4
        ):
            raise ValueError("confirmation order preimage fields drifted")
        if (
            self.payload.get("schema") != _CONFIRMATION_ORDER_SCHEMA_V4
            or self.payload.get("order_key_domain")
            != _CONFIRMATION_ORDER_KEY_DOMAIN_V4
        ):
            raise ValueError("confirmation order preimage authority drifted")
        _require_sha256(self.receipt_sha256, "receipt_sha256")
        expected = execution_authority_v4_bound_sha256(
            _CONFIRMATION_ORDER_SCHEMA_V4,
            self.payload,
        )
        if self.receipt_sha256 != expected:
            raise ValueError("confirmation order SHA differs from its V4 preimage")
        _require_sha256(self.physical_sha256, "physical_sha256")
        artifact = {
            "binding_sha256": _CONFIRMATION_BINDING_SHA256_V2,
            "payload": self.payload,
            "receipt_sha256": self.receipt_sha256,
            "schema": _CONFIRMATION_ORDER_SCHEMA_V4,
        }
        expected_physical = hashlib.sha256(
            canonical_json_bytes(artifact) + b"\n"
        ).hexdigest()
        if self.physical_sha256 != expected_physical:
            raise ValueError("confirmation order physical SHA differs from durable bytes")


@dataclass(frozen=True)
class ConfirmationBaseRunFlopSourceEnvelopeV2:
    """Raw base profiler ledger joined to its reconstructed per-step evidence."""

    flop_ledger_payload: Mapping[str, Any]
    flop_ledger_receipt_sha256: str
    base_flop_evidence_payload: Mapping[str, Any]
    base_flop_evidence_receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.flop_ledger_payload, Mapping):
            raise TypeError("base FLOP source requires a raw ledger mapping")
        if not isinstance(self.base_flop_evidence_payload, Mapping) or set(
            self.base_flop_evidence_payload
        ) != _CONFIRMATION_BASE_FLOP_EVIDENCE_FIELDS_V2:
            raise ValueError("base FLOP evidence preimage fields drifted")
        _require_sha256(self.flop_ledger_receipt_sha256, "flop_ledger_receipt_sha256")
        _require_sha256(
            self.base_flop_evidence_receipt_sha256,
            "base_flop_evidence_receipt_sha256",
        )
        if self.flop_ledger_receipt_sha256 != canonical_sha256(
            self.flop_ledger_payload
        ):
            raise ValueError("base raw FLOP ledger SHA differs from its preimage")
        if self.base_flop_evidence_receipt_sha256 != gtok_v2_bound_sha256(
            "weft1_gtok_v2_base_run_flop_evidence",
            self.base_flop_evidence_payload,
        ):
            raise ValueError("base FLOP evidence SHA differs from its preimage")
        if (
            self.base_flop_evidence_payload.get("flop_ledger_sha256")
            != self.flop_ledger_receipt_sha256
        ):
            raise ValueError("base FLOP evidence points to a different raw ledger")


@dataclass(frozen=True)
class ConfirmationArmFlopSourceEnvelopeV2:
    """One pair arm's exact S2-Q5 plan and both physical base sources."""

    arm_plan_payload: Mapping[str, Any]
    arm_plan_receipt_sha256: str
    base_runs: tuple[
        ConfirmationBaseRunFlopSourceEnvelopeV2,
        ConfirmationBaseRunFlopSourceEnvelopeV2,
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.arm_plan_payload, Mapping) or set(
            self.arm_plan_payload
        ) != _CONFIRMATION_ARM_FLOP_PLAN_FIELDS_V2:
            raise ValueError("confirmation arm FLOP plan preimage fields drifted")
        _require_sha256(self.arm_plan_receipt_sha256, "arm_plan_receipt_sha256")
        if self.arm_plan_receipt_sha256 != gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_arm_flop_plan",
            self.arm_plan_payload,
        ):
            raise ValueError("confirmation arm FLOP plan SHA differs from its preimage")
        if (
            not isinstance(self.base_runs, tuple)
            or len(self.base_runs) != GTOK_SEED_COUNT
            or any(
                not isinstance(row, ConfirmationBaseRunFlopSourceEnvelopeV2)
                for row in self.base_runs
            )
        ):
            raise TypeError("confirmation arm FLOP source requires two typed base runs")


@dataclass(frozen=True)
class ConfirmationAttemptLaunchEnvelopeV2:
    """Physical pre-START binding from one attempt ID to its priced plan."""

    payload: Mapping[str, Any]
    physical_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping) or set(self.payload) != (
            _CONFIRMATION_ATTEMPT_LAUNCH_FIELDS_V2
        ):
            raise ValueError("confirmation attempt-launch fields drifted")
        if (
            self.payload.get("binding_sha256") != _CONFIRMATION_BINDING_SHA256_V2
            or self.payload.get("schema")
            != "weft1_gtok_v2_confirmation_attempt_launch"
        ):
            raise ValueError("confirmation attempt-launch authority drifted")
        for name in ("calibration_projection_sha256", "execution_plan_sha256"):
            _require_sha256(str(self.payload.get(name)), name)
        for name in (
            "planned_compute_token_slots",
            "projected_run_a100_microseconds",
            "watchdog_limit_a100_microseconds",
        ):
            _require_exact_int(self.payload.get(name), name, minimum=1)
        for name in ("attempt_id", "logical_attempt_id"):
            if not isinstance(self.payload.get(name), str) or not self.payload.get(name):
                raise ValueError(f"confirmation attempt-launch {name} is empty")
        _require_sha256(self.physical_sha256, "physical_sha256")
        expected = hashlib.sha256(
            canonical_json_bytes(self.payload) + b"\n"
        ).hexdigest()
        if self.physical_sha256 != expected:
            raise ValueError("confirmation attempt-launch SHA differs from durable bytes")


@dataclass(frozen=True)
class ConfirmationExecutionPlanEnvelopeV2:
    """Canonical plan preimage used to price every physical attempt."""

    payload: Mapping[str, Any]
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("confirmation execution-plan payload must be a mapping")
        if set(self.payload) != _CONFIRMATION_EXECUTION_PLAN_FIELDS_V2:
            raise ValueError("confirmation execution-plan fields drifted")
        training_plan = self.payload.get("training_plan")
        if not isinstance(training_plan, Mapping) or set(training_plan) != (
            _CONFIRMATION_TRAINING_PLAN_FIELDS_V2
        ):
            raise ValueError("confirmation execution-plan training fields drifted")
        _require_sha256(self.receipt_sha256, "receipt_sha256")
        expected = gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_execution_plan",
            self.payload,
        )
        if self.receipt_sha256 != expected:
            raise ValueError("confirmation execution-plan SHA differs from its preimage")


@dataclass(frozen=True)
class ConfirmationRetryArtifactEnvelopeV2:
    """Canonical on-disk invalid-band artifact and its physical byte hash."""

    payload: Mapping[str, Any]
    physical_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("confirmation retry artifact payload must be a mapping")
        if set(self.payload) != _CONFIRMATION_RETRY_ARTIFACT_FIELDS_V2:
            raise ValueError("confirmation retry artifact fields drifted")
        if (
            self.payload.get("binding_sha256") != _CONFIRMATION_BINDING_SHA256_V2
            or self.payload.get("schema")
            != "weft1_gtok_v2_invalid_confirmation_flop_band"
        ):
            raise ValueError("confirmation retry artifact authority drifted")
        _require_sha256(self.physical_sha256, "physical_sha256")
        expected = hashlib.sha256(canonical_json_bytes(self.payload) + b"\n").hexdigest()
        if self.physical_sha256 != expected:
            raise ValueError("confirmation retry artifact physical SHA differs from bytes")


@dataclass(frozen=True)
class ConfirmationEvidenceClosureV2:
    """Complete evidence envelope required before a confirmation decision can mint."""

    compute_event_ledger_sha256: str
    lifecycle_ledger_sha256: str
    lifecycle_events: tuple[ConfirmationLifecycleEventEvidenceV2, ...]
    execution_plans: tuple[ConfirmationExecutionPlanEnvelopeV2, ...]
    confirmation_orders: tuple[ConfirmationOrderEnvelopeV2, ...]
    attempt_launches: tuple[ConfirmationAttemptLaunchEnvelopeV2, ...]
    base_flop_sources: tuple[ConfirmationArmFlopSourceEnvelopeV2, ...]
    fresh_joins: tuple[ConfirmationFreshEvidenceJoinV2, ...]
    retry_joins: tuple[ConfirmationRetryEvidenceJoinV2, ...] = ()
    retry_artifacts: tuple[ConfirmationRetryArtifactEnvelopeV2, ...] = ()

    def __post_init__(self) -> None:
        _require_sha256(self.compute_event_ledger_sha256, "compute_event_ledger_sha256")
        _require_sha256(self.lifecycle_ledger_sha256, "lifecycle_ledger_sha256")
        if not isinstance(self.lifecycle_events, tuple) or not self.lifecycle_events or any(
            not isinstance(row, ConfirmationLifecycleEventEvidenceV2)
            for row in self.lifecycle_events
        ):
            raise TypeError("confirmation closure requires typed lifecycle preimages")
        expected_lifecycle_sha256 = gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_lifecycle_ledger",
            self.lifecycle_events,
        )
        if self.lifecycle_ledger_sha256 != expected_lifecycle_sha256:
            raise ValueError("confirmation lifecycle ledger SHA differs from its preimages")
        prior_by_attempt: dict[str, ConfirmationLifecycleEventEvidenceV2] = {}
        for event in self.lifecycle_events:
            previous = prior_by_attempt.get(event.attempt_id)
            if previous is None:
                if event.phase != "START":
                    raise ValueError("confirmation lifecycle attempt lacks START")
            elif (
                previous.phase == "TERMINAL"
                or event.phase == "START"
                or event.charged_a100_microseconds
                < previous.charged_a100_microseconds
            ):
                raise ValueError("confirmation lifecycle transition is invalid")
            elif any(
                getattr(event, name) != getattr(previous, name)
                for name in (
                    "logical_attempt_id",
                    "scope",
                    "kind",
                    "gpu_uuid_provenance",
                    "offline_network_launch_receipt_sha256",
                    "heartbeat_interval_a100_microseconds",
                    "checkpoint_retained",
                )
            ):
                raise ValueError("confirmation lifecycle attempt identity drifted")
            prior_by_attempt[event.attempt_id] = event
        if any(event.phase != "TERMINAL" for event in prior_by_attempt.values()):
            raise ValueError("confirmation lifecycle attempt lacks TERMINAL")
        if (
            not isinstance(self.execution_plans, tuple)
            or not self.execution_plans
            or any(
                not isinstance(row, ConfirmationExecutionPlanEnvelopeV2)
                for row in self.execution_plans
            )
        ):
            raise TypeError("confirmation closure requires typed execution-plan preimages")
        plan_sha256s = tuple(row.receipt_sha256 for row in self.execution_plans)
        if len(set(plan_sha256s)) != len(plan_sha256s):
            raise ValueError("confirmation closure repeats an execution-plan preimage")
        if (
            not isinstance(self.confirmation_orders, tuple)
            or len(self.confirmation_orders) != GTOK_SEED_COUNT
            or any(
                not isinstance(row, ConfirmationOrderEnvelopeV2)
                for row in self.confirmation_orders
            )
        ):
            raise TypeError("confirmation closure requires two typed V4 order preimages")
        if len({row.receipt_sha256 for row in self.confirmation_orders}) != GTOK_SEED_COUNT:
            raise ValueError("confirmation closure order preimages collided")
        if len({row.physical_sha256 for row in self.confirmation_orders}) != GTOK_SEED_COUNT:
            raise ValueError("confirmation closure physical order artifacts collided")
        if (
            not isinstance(self.attempt_launches, tuple)
            or not self.attempt_launches
            or any(
                not isinstance(row, ConfirmationAttemptLaunchEnvelopeV2)
                for row in self.attempt_launches
            )
        ):
            raise TypeError("confirmation closure requires typed attempt-launch preimages")
        launch_ids = tuple(str(row.payload["attempt_id"]) for row in self.attempt_launches)
        if len(set(launch_ids)) != len(launch_ids) or len(
            {row.physical_sha256 for row in self.attempt_launches}
        ) != len(self.attempt_launches):
            raise ValueError("confirmation closure attempt-launch preimages collided")
        if (
            not isinstance(self.base_flop_sources, tuple)
            or len(self.base_flop_sources) != 2
            or any(
                not isinstance(row, ConfirmationArmFlopSourceEnvelopeV2)
                for row in self.base_flop_sources
            )
        ):
            raise TypeError("confirmation closure requires both typed pair-arm FLOP sources")
        source_vocabs = tuple(
            row.arm_plan_payload["vocab_size"] for row in self.base_flop_sources
        )
        if source_vocabs != tuple(sorted(source_vocabs)) or len(set(source_vocabs)) != 2:
            raise ValueError("confirmation pair-arm FLOP sources are not canonical")
        if (
            not isinstance(self.fresh_joins, tuple)
            or len(self.fresh_joins) != GTOK_SEED_COUNT
            or any(
                not isinstance(row, ConfirmationFreshEvidenceJoinV2)
                for row in self.fresh_joins
            )
        ):
            raise TypeError("confirmation closure requires exactly two typed fresh joins")
        if tuple(row.seed_slot for row in self.fresh_joins) != (0, 1):
            raise ValueError("confirmation closure fresh joins must be ordered by seed slot")
        if len({row.receipt_sha256 for row in self.fresh_joins}) != GTOK_SEED_COUNT:
            raise ValueError("confirmation closure fresh joins must be distinct")
        for name in (
            "terminal_lifecycle_event_sha256",
            "burst_receipt_sha256",
            "physical_flop_ledger_sha256",
        ):
            if len({getattr(row, name) for row in self.fresh_joins}) != GTOK_SEED_COUNT:
                raise ValueError(f"confirmation fresh-slot {name} identities collided")
        if not isinstance(self.retry_joins, tuple) or any(
            not isinstance(row, ConfirmationRetryEvidenceJoinV2)
            for row in self.retry_joins
        ):
            raise TypeError("confirmation retry joins must be a typed tuple")
        failed_ids = tuple(row.failed_attempt_id for row in self.retry_joins)
        if len(set(failed_ids)) != len(failed_ids):
            raise ValueError("confirmation retry joins repeat a failed attempt")
        if not isinstance(self.retry_artifacts, tuple) or any(
            not isinstance(row, ConfirmationRetryArtifactEnvelopeV2)
            for row in self.retry_artifacts
        ):
            raise TypeError("confirmation retry artifacts must be typed envelopes")
        if len(self.retry_artifacts) != len(self.retry_joins) or len(
            {row.physical_sha256 for row in self.retry_artifacts}
        ) != len(self.retry_artifacts) or {row.physical_sha256 for row in self.retry_artifacts} != {
            row.retry_artifact_physical_sha256 for row in self.retry_joins
        }:
            raise ValueError("confirmation retry joins and artifact preimages differ")
        ordered_retry_keys = tuple(
            (row.seed_slot, row.correction_ordinal) for row in self.retry_joins
        )
        if ordered_retry_keys != tuple(sorted(ordered_retry_keys)):
            raise ValueError("confirmation retry joins must use canonical slot/ordinal order")
        for seed_slot in (0, 1):
            ordinals = tuple(
                row.correction_ordinal
                for row in self.retry_joins
                if row.seed_slot == seed_slot
            )
            if ordinals != tuple(range(len(ordinals))):
                raise ValueError("confirmation correction ordinals must be contiguous per slot")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_confirmation_evidence_closure", self)


def _confirmation_physical_burst_evidence_sha256_v2(
    *,
    compute_attempt_id: str,
    execution_plan_sha256: str,
    burst_receipt_sha256: str,
) -> str:
    return gtok_v2_bound_sha256(
        "weft1_gtok_v2_confirmation_physical_burst_evidence",
        {
            "burst_receipt_sha256": burst_receipt_sha256,
            "compute_attempt_id": compute_attempt_id,
            "execution_plan_sha256": execution_plan_sha256,
        },
    )


def _confirmation_physical_flop_evidence_sha256_v2(
    *,
    compute_attempt_id: str,
    execution_plan_sha256: str,
    flop_ledger_receipt_sha256: str,
) -> str:
    return gtok_v2_bound_sha256(
        "weft1_gtok_v2_confirmation_physical_flop_ledger_evidence",
        {
            "compute_attempt_id": compute_attempt_id,
            "execution_plan_sha256": execution_plan_sha256,
            "flop_ledger_receipt_sha256": flop_ledger_receipt_sha256,
        },
    )


def _validate_confirmation_flop_ledger_payload_v2(
    payload: Mapping[str, Any],
) -> tuple[str, int, int, int]:
    """Validate a CompleteFlopLedgerV2 canonical preimage without an import cycle."""

    required = {
        "shapes",
        "optimizer_steps",
        "compute_token_slots",
        "profiler_with_flops",
        "flop_binding_sha256",
    }
    if set(payload) != required or payload.get("profiler_with_flops") is not True:
        raise ValueError("confirmation FLOP ledger preimage fields drifted")
    optimizer_steps = payload["optimizer_steps"]
    compute_token_slots = payload["compute_token_slots"]
    _require_exact_int(optimizer_steps, "flop optimizer_steps", minimum=1)
    _require_exact_int(compute_token_slots, "flop compute_token_slots", minimum=1)
    _require_sha256(str(payload["flop_binding_sha256"]), "flop_binding_sha256")
    if payload["flop_binding_sha256"] != GTOK_PHYSICAL_FLOP_BINDING_SHA256_V2:
        raise ValueError("confirmation FLOP ledger binding drifted")
    raw_shapes = payload["shapes"]
    if not isinstance(raw_shapes, (list, tuple)) or not raw_shapes:
        raise ValueError("confirmation FLOP ledger requires physical shapes")
    total_occurrences = 0
    total_slots = 0
    measured_flops = 0
    initial_occurrences = 0
    for raw_shape in raw_shapes:
        if not isinstance(raw_shape, Mapping) or set(raw_shape) != {
            "batch_rows",
            "sequence_length",
            "optimizer_phase",
            "occurrences",
            "profiler_rows",
            "unsupported_rows",
            "zero_flop_profiler_operators",
        }:
            raise ValueError("confirmation FLOP shape preimage fields drifted")
        batch_rows = raw_shape["batch_rows"]
        sequence_length = raw_shape["sequence_length"]
        occurrences = raw_shape["occurrences"]
        for name, value in (
            ("batch_rows", batch_rows),
            ("sequence_length", sequence_length),
            ("occurrences", occurrences),
        ):
            _require_exact_int(value, name, minimum=1)
        phase = raw_shape["optimizer_phase"]
        if phase not in ("initial", "steady"):
            raise ValueError("confirmation FLOP optimizer phase drifted")
        if phase == "initial":
            initial_occurrences += occurrences
        shape_flops = 0
        for rows_name in ("profiler_rows", "unsupported_rows"):
            raw_rows = raw_shape[rows_name]
            if not isinstance(raw_rows, (list, tuple)) or not raw_rows:
                raise ValueError("confirmation FLOP operator evidence is absent")
            for raw_row in raw_rows:
                expected_row_fields = (
                    {"operator", "flops_per_occurrence"}
                    if rows_name == "profiler_rows"
                    else {"family", "flops_per_occurrence", "derivation"}
                )
                if not isinstance(raw_row, Mapping) or set(raw_row) != expected_row_fields:
                    raise ValueError("confirmation FLOP operator row is invalid")
                flops = raw_row.get("flops_per_occurrence")
                _require_exact_int(flops, "flops_per_occurrence", minimum=1)
                shape_flops += flops
                if rows_name == "unsupported_rows" and "=" not in str(
                    raw_row.get("derivation", "")
                ):
                    raise ValueError("unsupported FLOP row lacks its derivation")
        zero_rows = raw_shape["zero_flop_profiler_operators"]
        if not isinstance(zero_rows, (list, tuple)) or tuple(
            sorted(set(zero_rows))
        ) != tuple(zero_rows):
            raise ValueError("zero-FLOP operator inventory drifted")
        total_occurrences += occurrences
        total_slots += batch_rows * sequence_length * occurrences
        measured_flops += shape_flops * occurrences
    if (
        total_occurrences != optimizer_steps
        or total_slots != compute_token_slots
        or initial_occurrences != 1
    ):
        raise ValueError("confirmation FLOP ledger totals do not close")
    return (
        canonical_sha256(payload),
        measured_flops,
        optimizer_steps,
        compute_token_slots,
    )


def _expand_confirmation_base_flop_steps_v2(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Reconstruct the governed initial/steady base-step sequence from a raw ledger."""

    _validate_confirmation_flop_ledger_payload_v2(payload)
    shapes = tuple(payload["shapes"])
    initial = tuple(row for row in shapes if row["optimizer_phase"] == "initial")
    steady = tuple(row for row in shapes if row["optimizer_phase"] == "steady")
    if len(initial) != 1 or initial[0]["occurrences"] != 1 or not steady:
        raise ValueError("base FLOP ledger cannot reconstruct physical step order")
    if len(steady) > 2:
        raise ValueError("base FLOP ledger has an unregistered shape schedule")
    if len(steady) == 2:
        ordered_steady = tuple(
            sorted(
                steady,
                key=lambda row: (row["batch_rows"], row["sequence_length"]),
                reverse=True,
            )
        )
        if ordered_steady[-1]["occurrences"] != 1:
            raise ValueError("only the terminal base batch may be partial")
    else:
        ordered_steady = steady
    expanded: list[Mapping[str, Any]] = []

    def append_shape(shape: Mapping[str, Any], count: int) -> None:
        per_step = sum(
            row["flops_per_occurrence"]
            for name in ("profiler_rows", "unsupported_rows")
            for row in shape[name]
        )
        for _ in range(count):
            expanded.append(
                {
                    "optimizer_step": len(expanded) + 1,
                    "batch_rows": shape["batch_rows"],
                    "sequence_length": shape["sequence_length"],
                    "optimizer_phase": "initial" if not expanded else "steady",
                    "measured_flops": per_step,
                }
            )

    append_shape(initial[0], 1)
    for shape in ordered_steady:
        append_shape(shape, shape["occurrences"])
    return tuple(expanded)


@dataclass(frozen=True)
class ConfirmationResultSlotV2:
    """Uniform slot reference over one reused base or one fresh result."""

    vocab_size: int
    seed_slot: int
    source: str
    run_seed: int
    registry_key: str
    source_run_receipt_sha256: str
    paired_base_run_receipt_sha256: str
    compute_attempt_id: str | None
    observations: tuple[BpbMilestoneReceiptV2, ...]
    stream_bytes: int
    stream_docs: int
    stream_tokens: int
    trained_tokens: int
    dropped_tokens: int
    trained_bytes: int
    dropped_bytes: int
    trained_docs_full: int
    boundary_doc_id: str | None
    boundary_doc_consumed_tokens: int | None
    dropped_docs: int

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS or self.seed_slot not in (0, 1):
            raise ValueError("confirmation result slot identity is invalid")
        if self.source not in ("reused_byte_matched", "fresh_confirmation"):
            raise ValueError("confirmation result slot source is unregistered")
        if type(self.run_seed) is not int:
            raise TypeError("confirmation result run seed must be an exact integer")
        _require_sha256(self.source_run_receipt_sha256, "source_run_receipt_sha256")
        _require_sha256(
            self.paired_base_run_receipt_sha256,
            "paired_base_run_receipt_sha256",
        )
        if self.source == "fresh_confirmation":
            expected_key = f"gtok.confirm.{self.vocab_size}.{self.seed_slot}"
            if self.registry_key != expected_key or not self.compute_attempt_id:
                raise ValueError("fresh confirmation result lacks its registry/attempt join")
        elif self.registry_key or self.compute_attempt_id is not None:
            raise ValueError("reused confirmation result may not consume a confirm root")
        if tuple(item.label for item in self.observations) != GTOK_MILESTONE_LABELS:
            raise ValueError("confirmation result slot requires all three curve points")
        for name in (
            "stream_bytes",
            "stream_docs",
            "stream_tokens",
            "trained_tokens",
            "dropped_tokens",
            "trained_bytes",
            "dropped_bytes",
            "trained_docs_full",
            "dropped_docs",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"confirmation slot {name} is invalid")

    @property
    def terminal_bpb(self) -> float:
        return self.observations[-1].pooled_bpb


@dataclass(frozen=True, init=False)
class ValidatedComputeConfirmationV2:
    selection_receipt_sha256: str
    selection_confirmation_authority_sha256: str
    matrix_receipt_sha256: str
    compute_campaign_receipt_sha256: str
    pair: tuple[int, int]
    winner_vocab_size: int
    runner_up_vocab_size: int
    reused_vocab_size: int
    fresh_vocab_size: int
    common_flop_budget: int
    runs: tuple[ComputeConfirmationRunV2, ...]
    evidence_closure: ConfirmationEvidenceClosureV2
    evidence_closure_sha256: str
    result_slots: tuple[ConfirmationResultSlotV2, ...]
    rho_bpb_micros: tuple[tuple[int, int], tuple[int, int]]
    sample_sd_bpb: tuple[tuple[int, float], tuple[int, float]]
    s_hat_c_bpb: float
    delta_bpb_micros: int
    threshold_multiplier: int
    threshold_bpb: float
    slot_delta_bpb: tuple[float, float]
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
        evidence_closure: ConfirmationEvidenceClosureV2,
        reused_vocab_size: int,
        fresh_vocab_size: int,
        result_slots: tuple[ConfirmationResultSlotV2, ...],
        rho_bpb_micros: tuple[tuple[int, int], tuple[int, int]],
        sample_sd_bpb: tuple[tuple[int, float], tuple[int, float]],
        s_hat_c_bpb: float,
        delta_bpb_micros: int,
        threshold_multiplier: int,
        threshold_bpb: float,
        slot_delta_bpb: tuple[float, float],
        status: str,
        sentinel: object,
    ) -> "ValidatedComputeConfirmationV2":
        if sentinel is not _FACTORY_SENTINEL:
            raise PermissionError("compute confirmations are factory-only")
        value = object.__new__(cls)
        payload = {
            "selection_receipt_sha256": selection.receipt_sha256,
            "selection_confirmation_authority_sha256": (
                selection.selection_confirmation_authority_sha256
            ),
            "matrix_receipt_sha256": matrix.receipt_sha256,
            "compute_campaign_receipt_sha256": compute.receipt_sha256,
            "pair": selection.compute_confirmation_pair,
            "winner_vocab_size": selection.compute_confirmation_pair[0],
            "runner_up_vocab_size": selection.compute_confirmation_pair[1],
            "reused_vocab_size": reused_vocab_size,
            "fresh_vocab_size": fresh_vocab_size,
            "common_flop_budget": common_flop_budget,
            "runs": runs,
            "evidence_closure": evidence_closure,
            "evidence_closure_sha256": evidence_closure.receipt_sha256,
            "result_slots": result_slots,
            "rho_bpb_micros": rho_bpb_micros,
            "sample_sd_bpb": sample_sd_bpb,
            "s_hat_c_bpb": s_hat_c_bpb,
            "delta_bpb_micros": delta_bpb_micros,
            "threshold_multiplier": threshold_multiplier,
            "threshold_bpb": threshold_bpb,
            "slot_delta_bpb": slot_delta_bpb,
            "cumulative_campaign_a100_microseconds": (
                compute.consumed_a100_microseconds
            ),
            "status": status,
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
    evidence_closure: ConfirmationEvidenceClosureV2,
) -> ValidatedComputeConfirmationV2:
    """Validate the preregistered top-two equal-FLOP confirmation."""

    validate_selection_receipt_v2(selection, matrix=matrix)
    if not isinstance(compute, CampaignComputeReceiptV2):
        raise TypeError("confirmation requires a structured compute campaign receipt")
    if not isinstance(evidence_closure, ConfirmationEvidenceClosureV2):
        raise TypeError("confirmation requires a typed evidence closure")
    # Frozen dataclasses do not make nested Mapping objects immutable.  Re-run
    # every nested envelope constructor at the mint boundary so a post-build
    # payload mutation cannot survive on a stale receipt SHA.
    for row in evidence_closure.execution_plans:
        ConfirmationExecutionPlanEnvelopeV2(
            payload=row.payload,
            receipt_sha256=row.receipt_sha256,
        )
    for row in evidence_closure.confirmation_orders:
        ConfirmationOrderEnvelopeV2(
            payload=row.payload,
            receipt_sha256=row.receipt_sha256,
            physical_sha256=row.physical_sha256,
        )
    for row in evidence_closure.attempt_launches:
        ConfirmationAttemptLaunchEnvelopeV2(
            payload=row.payload,
            physical_sha256=row.physical_sha256,
        )
    for arm in evidence_closure.base_flop_sources:
        reconstructed_base_runs = tuple(
            ConfirmationBaseRunFlopSourceEnvelopeV2(
                flop_ledger_payload=row.flop_ledger_payload,
                flop_ledger_receipt_sha256=row.flop_ledger_receipt_sha256,
                base_flop_evidence_payload=row.base_flop_evidence_payload,
                base_flop_evidence_receipt_sha256=(
                    row.base_flop_evidence_receipt_sha256
                ),
            )
            for row in arm.base_runs
        )
        assert len(reconstructed_base_runs) == GTOK_SEED_COUNT
        ConfirmationArmFlopSourceEnvelopeV2(
            arm_plan_payload=arm.arm_plan_payload,
            arm_plan_receipt_sha256=arm.arm_plan_receipt_sha256,
            base_runs=(reconstructed_base_runs[0], reconstructed_base_runs[1]),
        )
    for row in evidence_closure.retry_artifacts:
        ConfirmationRetryArtifactEnvelopeV2(
            payload=row.payload,
            physical_sha256=row.physical_sha256,
        )
    if compute.scope != "confirmation":
        raise ValueError("confirmation requires a confirmation compute scope")
    expected_event_ledger_sha256 = compute_event_ledger_sha256_v2(
        compute.attempts
    )
    if compute.event_ledger_sha256 != expected_event_ledger_sha256:
        raise ValueError("confirmation compute event ledger differs from ordered attempts")
    if any(
        isinstance(attempt, ComputeAttemptReceiptV2)
        and attempt.status == "aborted_watchdog"
        for attempt in compute.attempts
    ):
        raise GTokV2Stop("confirmation contains a watchdog abort; return to strategy")
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
    base_by_key = {(run.vocab_size, run.seed): run for run in matrix.runs}
    if tuple(
        row.arm_plan_payload["vocab_size"]
        for row in evidence_closure.base_flop_sources
    ) != tuple(sorted(pair)):
        raise ValueError("confirmation base FLOP sources differ from the selected pair")
    arm_source_by_vocab: dict[int, ConfirmationArmFlopSourceEnvelopeV2] = {}
    for arm_source in evidence_closure.base_flop_sources:
        arm_plan = arm_source.arm_plan_payload
        vocab_size = arm_plan["vocab_size"]
        arm_source_by_vocab[vocab_size] = arm_source
        if tuple(arm_plan["seeds"]) != matrix.seeds:
            raise ValueError("confirmation arm FLOP source seed order drifted")
        evidence_sha256s: list[str] = []
        base_flops: list[int] = []
        base_steps: list[int] = []
        for seed, source in zip(matrix.seeds, arm_source.base_runs, strict=True):
            evidence = source.base_flop_evidence_payload
            ledger_sha256, ledger_flops, ledger_steps, ledger_slots = (
                _validate_confirmation_flop_ledger_payload_v2(
                    source.flop_ledger_payload
                )
            )
            expanded_steps = _expand_confirmation_base_flop_steps_v2(
                source.flop_ledger_payload
            )
            raw_steps = evidence.get("steps")
            if not isinstance(raw_steps, (list, tuple)) or any(
                not isinstance(row, Mapping)
                or set(row) != _CONFIRMATION_BASE_STEP_FLOP_FIELDS_V2
                for row in raw_steps
            ):
                raise ValueError("base FLOP source step preimages drifted")
            base_run = base_by_key.get((vocab_size, seed))
            if (
                ledger_sha256 != source.flop_ledger_receipt_sha256
                or canonical_json_bytes(raw_steps)
                != canonical_json_bytes(expanded_steps)
                or evidence.get("vocab_size") != vocab_size
                or evidence.get("seed") != seed
                or base_run is None
                or evidence.get("base_run_receipt_sha256")
                != base_run.receipt_sha256
                or evidence.get("base_compute_attempt_id")
                != base_run.compute_attempt_id
                or evidence.get("measured_flops") != ledger_flops
                or ledger_flops != base_run.measured_flops
                or ledger_steps != base_run.terminal.optimizer_step
                or ledger_slots != base_run.trained_tokens
            ):
                raise ValueError("base FLOP source does not close to its physical matrix run")
            evidence_sha256s.append(source.base_flop_evidence_receipt_sha256)
            base_flops.append(ledger_flops)
            base_steps.append(ledger_steps)
        if len(set(base_steps)) != 1:
            raise ValueError("confirmation arm base FLOP horizons disagree")
        arm_mean = sum(base_flops) // GTOK_SEED_COUNT
        if (
            tuple(arm_plan["base_flop_evidence_sha256s"])
            != tuple(evidence_sha256s)
            or tuple(arm_plan["base_flops"]) != tuple(base_flops)
            or arm_plan["byte_matched_optimizer_steps"] != base_steps[0]
            or arm_plan["arm_mean_flops"] != arm_mean
            or arm_plan["planned_optimizer_steps"]
            != (
                arm_plan["target_flops"]
                * arm_plan["byte_matched_optimizer_steps"]
            )
            // arm_mean
        ):
            raise ValueError("confirmation arm FLOP plan differs from raw base evidence")
    source_target_flops = min(
        row.arm_plan_payload["arm_mean_flops"]
        for row in evidence_closure.base_flop_sources
    )
    if any(
        row.arm_plan_payload["target_flops"] != source_target_flops
        for row in evidence_closure.base_flop_sources
    ):
        raise ValueError("confirmation F_star differs from the raw pair-arm evidence")
    arm_mean_flops = {
        vocab_size: sum(base_by_key[(vocab_size, seed)].measured_flops for seed in matrix.seeds)
        // GTOK_SEED_COUNT
        for vocab_size in pair
    }
    if len(set(arm_mean_flops.values())) != 2:
        raise GTokV2Stop("confirmation pair has no unique min-FLOP arm; return to strategy")
    reused_vocab_size = min(pair, key=lambda value: arm_mean_flops[value])
    fresh_vocab_size = next(value for value in pair if value != reused_vocab_size)
    common_flop_budget = arm_mean_flops[reused_vocab_size]
    if common_flop_budget != source_target_flops or any(
        arm_source_by_vocab[vocab_size].arm_plan_payload["arm_mean_flops"]
        != arm_mean_flops[vocab_size]
        for vocab_size in pair
    ):
        raise ValueError("confirmation raw FLOP sources differ from matrix totals")
    if tuple(item.vocab_size for item in compute.preflight.calibrations) != (
        fresh_vocab_size,
    ):
        raise ValueError("confirmation preflight requires only the fresh arm projection")
    projection = compute.preflight.calibrations[0]
    base_projection = next(
        item
        for item in matrix.compute.preflight.calibrations
        if item.vocab_size == fresh_vocab_size
    )
    if (
        projection.projection_source != "completed_base_calibration"
        or projection.full_run_count != GTOK_SEED_COUNT
        or projection.projection_source_receipt_sha256
        != base_projection.receipt_sha256
    ):
        raise ValueError("confirmation fresh-arm projection must reuse completed base evidence")
    inherited_fields = (
        "calibration_steps",
        "measured_tokens",
        "measured_a100_microseconds",
        "measured_heldout_evaluation_a100_microseconds",
        "heldout_evaluations_per_full_run",
        "measured_output_surface_a100_microseconds",
        "output_surface_benchmarks_per_full_run",
    )
    if any(
        getattr(projection, name) != getattr(base_projection, name)
        for name in inherited_fields
    ):
        raise ValueError("confirmation projection changed inherited base measurements")
    if not isinstance(runs, tuple) or len(runs) != GTOK_SEED_COUNT:
        raise ValueError("compute confirmation requires exactly two fresh seed-slot runs")
    if any(not isinstance(run, ComputeConfirmationRunV2) for run in runs):
        raise TypeError("confirmation runs must be ComputeConfirmationRunV2 values")
    if {run.vocab_size for run in runs} != {fresh_vocab_size} or {
        run.seed_slot for run in runs
    } != {0, 1}:
        raise ValueError("compute confirmation runs differ from the fresh arm/slot registry")
    flop_budgets = {run.common_flop_budget for run in runs}
    if flop_budgets != {common_flop_budget}:
        raise ValueError("fresh confirmation runs must target exact F_star")
    denominator = matrix.corpus.heldout_denominator_signature
    if any(
        run.heldout_stream_sha256 != matrix.corpus.heldout_stream_sha256
        or any(
            observation.denominator_signature != denominator
            for observation in run.observations
        )
        for run in runs
    ):
        raise ValueError("every confirmation H pass must use the frozen denominator")
    if any(
        run.physical_d6_evidence_sha256
        != matrix.corpus.d6_physical_evidence_sha256
        for run in runs
    ):
        raise ValueError("fresh confirmation Q3 evidence differs from frozen physical D6")
    if len({run.document_multiset_sha256 for run in runs}) != 1:
        raise ValueError("fresh confirmation slots name different frozen T multisets")
    for name in (
        "confirmation_order_receipt_sha256",
        "data_order_sha256",
        "framed_payload_sha256",
        "execution_plan_sha256",
        "training_plan_sha256",
    ):
        if len({getattr(run, name) for run in runs}) != GTOK_SEED_COUNT:
            raise ValueError(f"fresh confirmation {name} identities collided")
    by_slot = {run.seed_slot: run for run in runs}
    plan_by_sha256 = {
        row.receipt_sha256: row.payload for row in evidence_closure.execution_plans
    }
    order_by_receipt_sha256 = {
        row.receipt_sha256: row.payload for row in evidence_closure.confirmation_orders
    }
    referenced_order_sha256s: set[str] = set()
    byte_matched_steps = {
        base_by_key[(fresh_vocab_size, seed)].terminal.optimizer_step
        for seed in matrix.seeds
    }
    if len(byte_matched_steps) != 1:
        raise ValueError("fresh base runs disagree on their byte-matched horizon")
    governed_byte_matched_steps = next(iter(byte_matched_steps))
    plan_training_sha256: dict[str, str] = {}
    initial_plan_slots: list[int] = []
    initial_plan_token_counts: list[int] = []
    for plan_sha256, plan in plan_by_sha256.items():
        training_plan = plan["training_plan"]
        assert isinstance(training_plan, Mapping)
        seed_slot = plan["seed_slot"]
        _require_exact_int(seed_slot, "execution-plan seed_slot", minimum=0)
        if seed_slot not in by_slot:
            raise ValueError("confirmation execution plan has an unregistered seed slot")
        slot_run = by_slot[seed_slot]
        expected_registry_key = f"gtok.confirm.{fresh_vocab_size}.{seed_slot}"
        expected_run_seed = int.from_bytes(
            hashlib.sha256(expected_registry_key.encode("ascii")).digest()[:8],
            byteorder="big",
        )
        expected_initialization_seed = derive_module_seed(
            A2_CAMPAIGN_ROOT_SEED,
            f"gtok.init.shared.{expected_run_seed}",
        )
        expected_data_order_seed = derive_module_seed(
            A2_CAMPAIGN_ROOT_SEED,
            f"gtok.data.shared.{expected_run_seed}",
        )
        expected_plan_identity = {
            "vocab_size": fresh_vocab_size,
            "registry_key": slot_run.registry_key,
            "seed": slot_run.seed,
            "initialization_seed": slot_run.initialization_seed,
            "data_order_seed": slot_run.data_order_seed,
            "data_order_sha256": slot_run.data_order_sha256,
            "confirmation_order_receipt_sha256": (
                slot_run.confirmation_order_receipt_sha256
            ),
            "physical_d6_evidence_sha256": slot_run.physical_d6_evidence_sha256,
            "document_multiset_sha256": slot_run.document_multiset_sha256,
            "framed_payload_sha256": slot_run.framed_payload_sha256,
            "order_document_count": slot_run.stream_docs,
            "order_retained_text_bytes": slot_run.stream_bytes,
            "target_flops": common_flop_budget,
            "arm_mean_flops": arm_mean_flops[fresh_vocab_size],
            "byte_matched_optimizer_steps": governed_byte_matched_steps,
        }
        if (
            plan.get("registry_key") != expected_registry_key
            or plan.get("seed") != expected_run_seed
            or plan.get("initialization_seed") != expected_initialization_seed
            or plan.get("data_order_seed") != expected_data_order_seed
            or any(
                plan.get(name) != value
                for name, value in expected_plan_identity.items()
            )
        ):
            raise ValueError("confirmation execution-plan authority fields drifted")
        if plan.get("arm_flop_plan_sha256") != arm_source_by_vocab[
            fresh_vocab_size
        ].arm_plan_receipt_sha256:
            raise ValueError("confirmation execution plan lacks its raw arm-FLOP source")
        order = order_by_receipt_sha256.get(
            str(plan["confirmation_order_receipt_sha256"])
        )
        if order is None or any(
            order.get(name) != value
            for name, value in {
                "confirmation_run_seed": plan["seed"],
                "data_order_seed": plan["data_order_seed"],
                "physical_d6_evidence_sha256": plan["physical_d6_evidence_sha256"],
                "document_multiset_sha256": plan["document_multiset_sha256"],
                "ordered_raw_content_ids_sha256": plan["data_order_sha256"],
                "framed_payload_sha256": plan["framed_payload_sha256"],
                "document_count": plan["order_document_count"],
                "retained_text_bytes": plan["order_retained_text_bytes"],
            }.items()
        ):
            raise ValueError("confirmation execution plan lacks its physical Q3 order preimage")
        referenced_order_sha256s.add(str(plan["confirmation_order_receipt_sha256"]))
        for name in (
            "arm_flop_plan_sha256",
            "packed_stream_sha256",
            "calibration_prefix_packed_stream_sha256",
            "packing_binding_sha256",
        ):
            _require_sha256(
                plan[name] if name == "arm_flop_plan_sha256" else training_plan[name],
                name,
            )
        for name in (
            "optimizer_steps",
            "global_batch_sequences",
            "sequence_length",
            "compute_token_slots",
            "valid_prediction_count",
            "trained_bytes",
            "trained_tokens",
            "stream_bytes",
            "stream_tokens",
            "stream_docs",
            "calibration_prefix_compute_token_slots",
            "calibration_prefix_valid_prediction_count",
        ):
            _require_exact_int(training_plan[name], name, minimum=1)
        for name in (
            "trained_docs_full",
            "dropped_bytes",
            "dropped_tokens",
            "dropped_docs",
            "calibration_prefix_realized_raw_bytes",
            "calibration_prefix_document_count",
        ):
            _require_exact_int(training_plan[name], name, minimum=0)
        if (
            training_plan["confirmation_order_receipt_sha256"]
            != plan["confirmation_order_receipt_sha256"]
            or training_plan["stream_bytes"] != plan["order_retained_text_bytes"]
            or training_plan["stream_docs"] != plan["order_document_count"]
            or training_plan["compute_token_slots"]
            != training_plan["optimizer_steps"]
            * training_plan["global_batch_sequences"]
            * training_plan["sequence_length"]
            or training_plan["trained_tokens"]
            != training_plan["compute_token_slots"]
            or training_plan["stream_tokens"]
            != training_plan["trained_tokens"] + training_plan["dropped_tokens"]
            or training_plan["stream_bytes"]
            != training_plan["trained_bytes"] + training_plan["dropped_bytes"]
            or training_plan["calibration_prefix_steps"] != 100
            or training_plan["optimizer_steps"] < 100
            or training_plan["global_batch_sequences"] != 256
            or training_plan["sequence_length"] != 2_048
            or training_plan["calibration_prefix_compute_token_slots"]
            != 100
            * training_plan["global_batch_sequences"]
            * training_plan["sequence_length"]
        ):
            raise ValueError("confirmation execution-plan accounting does not close")
        has_boundary = training_plan["boundary_doc_id"] is not None
        if has_boundary != (training_plan["boundary_doc_consumed_tokens"] is not None):
            raise ValueError("confirmation execution-plan boundary fields drifted")
        if training_plan["stream_docs"] != (
            training_plan["trained_docs_full"]
            + training_plan["dropped_docs"]
            + int(has_boundary)
        ):
            raise ValueError("confirmation execution-plan document accounting does not close")
        retry_realized = plan["retry_of_realized_flops"]
        retry_prior_steps = plan["retry_of_optimizer_steps"]
        if (retry_realized is None) != (retry_prior_steps is None):
            raise ValueError("confirmation execution-plan retry fields drifted")
        if retry_realized is None:
            expected_optimizer_steps = (
                common_flop_budget * governed_byte_matched_steps
            ) // arm_mean_flops[fresh_vocab_size]
            initial_plan_slots.append(seed_slot)
            initial_plan_token_counts.append(training_plan["compute_token_slots"])
        else:
            _require_exact_int(retry_realized, "retry_of_realized_flops", minimum=1)
            _require_exact_int(retry_prior_steps, "retry_of_optimizer_steps", minimum=1)
            expected_optimizer_steps = (
                common_flop_budget * retry_prior_steps
            ) // retry_realized
        if training_plan["optimizer_steps"] != expected_optimizer_steps:
            raise ValueError("confirmation execution-plan FLOP horizon drifted")
        checkpoints = tuple(plan["heldout_evaluation_steps"])
        if (
            checkpoints != tuple(training_plan["bpb_checkpoint_steps"])
            or len(checkpoints) != 3
            or tuple(sorted(set(checkpoints))) != checkpoints
            or checkpoints[-1] != training_plan["optimizer_steps"]
        ):
            raise ValueError("confirmation execution-plan checkpoints drifted")
        plan_training_sha256[plan_sha256] = canonical_sha256(training_plan)
    if sorted(initial_plan_slots) != [0, 1]:
        raise ValueError("confirmation closure requires one initial plan per seed slot")
    if (
        len(set(initial_plan_token_counts)) != 1
        or projection.planned_tokens_per_run != initial_plan_token_counts[0]
    ):
        raise ValueError("confirmation preflight is not priced from initial plan horizons")
    expected_initial_projection = _ceil_div(
        projection.measured_a100_microseconds * initial_plan_token_counts[0],
        projection.measured_tokens,
    ) + (
        projection.measured_heldout_evaluation_a100_microseconds
        * projection.heldout_evaluations_per_full_run
    ) + (
        projection.measured_output_surface_a100_microseconds
        * projection.output_surface_benchmarks_per_full_run
    )
    if projection.projected_run_a100_microseconds != expected_initial_projection:
        raise ValueError("confirmation preflight run projection is underpriced")
    if referenced_order_sha256s != set(order_by_receipt_sha256):
        raise ValueError("confirmation closure carries an unreferenced Q3 order preimage")
    if any(
        run.base_run_receipt_sha256
        != base_by_key[(fresh_vocab_size, matrix.seeds[run.seed_slot])].receipt_sha256
        for run in runs
    ):
        raise ValueError("fresh run is not joined to its paired base seed-slot receipt")
    if any(
        (
            run.stream_bytes,
            run.stream_docs,
            run.stream_tokens,
        )
        != (
            base_by_key[(fresh_vocab_size, matrix.seeds[run.seed_slot])].stream_bytes,
            base_by_key[(fresh_vocab_size, matrix.seeds[run.seed_slot])].stream_docs,
            base_by_key[(fresh_vocab_size, matrix.seeds[run.seed_slot])].stream_tokens,
        )
        for run in runs
    ):
        raise ValueError("fresh confirmation stream contents differ from the byte-matched arm")
    base_order_identities = {
        run.data_order_sha256
        for run in matrix.runs
        if run.vocab_size == fresh_vocab_size
    }
    if (
        len({run.data_order_sha256 for run in runs}) != GTOK_SEED_COUNT
        or any(run.data_order_sha256 in base_order_identities for run in runs)
    ):
        raise ValueError("fresh confirmation data orders were replayed or collided")
    if any(
        run.training_runtime_receipt_sha256
        != matrix.training_runtime_receipt_sha256
        for run in runs
    ):
        raise ValueError("confirmation runtime differs from the base campaign runtime")
    if any(
        run.code_closure_receipt_sha256 != matrix.code_closure_receipt_sha256
        for run in runs
    ):
        raise ValueError("confirmation code closure differs from the base campaign")
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
            or attempt.execution_plan_sha256 != run.execution_plan_sha256
            or attempt.planned_compute_token_slots != run.trained_tokens
            or attempt.hard_abort_issued
        ):
            raise ValueError("confirmation run is absent from the all-attempt ledger")
    completed_attempt_ids = {
        item.attempt_id
        for item in compute.attempts
        if isinstance(item, ComputeAttemptReceiptV2)
        and item.kind == "full_run"
        and item.status == "completed"
    }
    if completed_attempt_ids != {run.compute_attempt_id for run in runs}:
        raise ValueError("confirmation ledger has missing or duplicate selected completions")
    if any(
        not isinstance(item, ComputeAttemptReceiptV2)
        or item.kind != "full_run"
        or item.vocab_size != fresh_vocab_size
        for item in compute.attempts
    ):
        raise ValueError("confirmation ledger may charge only fresh-arm full-run attempts")
    selected_compute = sum(run.measured_a100_microseconds for run in runs)
    if compute.selected_run_a100_microseconds != selected_compute:
        raise ValueError("confirmation selected-run compute differs from its meter")
    if evidence_closure.compute_event_ledger_sha256 != compute.event_ledger_sha256:
        raise ValueError("confirmation closure differs from the compute event ledger")
    launch_by_attempt_id = {
        str(row.payload["attempt_id"]): row
        for row in evidence_closure.attempt_launches
    }
    if set(launch_by_attempt_id) != {attempt.attempt_id for attempt in compute.attempts}:
        raise ValueError("confirmation attempt-launch evidence does not cover the meter")
    referenced_plan_sha256s: set[str] = set()
    for attempt in compute.attempts:
        plan = plan_by_sha256.get(str(attempt.execution_plan_sha256))
        if plan is None:
            raise ValueError("confirmation attempt lacks its execution-plan preimage")
        training_plan = plan["training_plan"]
        assert isinstance(training_plan, Mapping)
        projected = _ceil_div(
            projection.measured_a100_microseconds
            * training_plan["compute_token_slots"],
            projection.measured_tokens,
        ) + (
            projection.measured_heldout_evaluation_a100_microseconds
            * projection.heldout_evaluations_per_full_run
        ) + (
            projection.measured_output_surface_a100_microseconds
            * projection.output_surface_benchmarks_per_full_run
        )
        launch = launch_by_attempt_id[attempt.attempt_id].payload
        if (
            attempt.vocab_size != plan["vocab_size"]
            or attempt.seed != plan["seed"]
            or attempt.planned_compute_token_slots
            != training_plan["compute_token_slots"]
            or attempt.projected_run_a100_microseconds != projected
            or attempt.watchdog_limit_a100_microseconds
            != GTOK_PER_RUN_WATCHDOG_MULTIPLIER * projected
            or launch["logical_attempt_id"] == ""
            or launch["binding_sha256"] != _CONFIRMATION_BINDING_SHA256_V2
            or launch["calibration_projection_sha256"]
            != attempt.calibration_projection_sha256
            or launch["execution_plan_sha256"] != attempt.execution_plan_sha256
            or launch["planned_compute_token_slots"]
            != attempt.planned_compute_token_slots
            or launch["projected_run_a100_microseconds"]
            != attempt.projected_run_a100_microseconds
            or launch["watchdog_limit_a100_microseconds"]
            != attempt.watchdog_limit_a100_microseconds
            or launch["vocab_size"] != attempt.vocab_size
            or launch["seed"] != attempt.seed
        ):
            raise ValueError("confirmation attempt differs from its priced execution plan")
        referenced_plan_sha256s.add(str(attempt.execution_plan_sha256))
    if referenced_plan_sha256s != set(plan_by_sha256):
        raise ValueError("confirmation closure carries an unattempted execution plan")
    lifecycle_terminal = tuple(
        event
        for event in evidence_closure.lifecycle_events
        if event.phase == "TERMINAL"
    )
    if (
        len(lifecycle_terminal) != len(compute.attempts)
        or len({event.attempt_id for event in lifecycle_terminal})
        != len(lifecycle_terminal)
    ):
        raise ValueError("confirmation lifecycle terminals do not cover every attempt once")
    terminal_by_attempt = {event.attempt_id: event for event in lifecycle_terminal}
    terminal_by_sha256 = {
        event.receipt_sha256: event for event in lifecycle_terminal
    }
    for attempt in compute.attempts:
        terminal = terminal_by_attempt.get(attempt.attempt_id)
        if (
            terminal is None
            or terminal.terminal_status != attempt.status
            or terminal.charged_a100_microseconds
            != attempt.consumed_a100_microseconds
            or terminal.logical_attempt_id
            != launch_by_attempt_id[attempt.attempt_id].payload["logical_attempt_id"]
        ):
            raise ValueError("confirmation lifecycle terminal differs from compute meter")
    failed_attempts = {
        item.attempt_id: item
        for item in compute.attempts
        if isinstance(item, ComputeAttemptReceiptV2) and item.status == "failed"
    }
    retry_by_failed_id = {
        row.failed_attempt_id: row for row in evidence_closure.retry_joins
    }
    retry_artifact_by_sha256 = {
        row.physical_sha256: row for row in evidence_closure.retry_artifacts
    }
    if set(failed_attempts) != set(retry_by_failed_id):
        raise ValueError("every failed confirmation attempt requires one retry evidence join")
    attempt_position = {
        item.attempt_id: index for index, item in enumerate(compute.attempts)
    }
    for failed_id, failed_attempt in failed_attempts.items():
        retry = retry_by_failed_id[failed_id]
        slot_run = by_slot[retry.seed_slot]
        if (
            retry.vocab_size != fresh_vocab_size
            or failed_attempt.vocab_size != retry.vocab_size
            or failed_attempt.seed != slot_run.seed
            or retry.failed_attempt_receipt_sha256
            != failed_attempt.receipt_sha256
            or retry.failed_execution_plan_sha256
            != failed_attempt.execution_plan_sha256
            or retry.target_flops != common_flop_budget
        ):
            raise ValueError("confirmation retry join differs from its charged failed attempt")
        successor_attempts = tuple(
            item
            for index, item in enumerate(compute.attempts)
            if index > attempt_position[failed_id]
            and isinstance(item, ComputeAttemptReceiptV2)
            and item.kind == "full_run"
            and item.vocab_size == retry.vocab_size
            and item.seed == failed_attempt.seed
            and item.execution_plan_sha256 == retry.retry_execution_plan_sha256
        )
        if not successor_attempts:
            raise ValueError("confirmation retry plan was never attempted after correction")
        failed_terminal = terminal_by_sha256.get(
            retry.failed_terminal_lifecycle_event_sha256
        )
        artifact = retry_artifact_by_sha256.get(
            retry.retry_artifact_physical_sha256
        )
        if (
            failed_terminal is None
            or failed_terminal.attempt_id != failed_id
            or failed_terminal.terminal_status != "failed"
            or failed_terminal.completion_payload is not None
            or artifact is None
        ):
            raise ValueError("confirmation retry lacks failed lifecycle/artifact preimages")
        retry_payload = artifact.payload
        raw_attempt = retry_payload.get("attempt")
        raw_invalid_ledger = retry_payload.get("invalid_physical_flop_ledger")
        raw_passed_burst = retry_payload.get("passed_burst_flop_receipt")
        raw_retry_plan = retry_payload.get("retry_execution_plan")
        if (
            not isinstance(raw_attempt, Mapping)
            or canonical_json_bytes(raw_attempt)
            != canonical_json_bytes(asdict(failed_attempt))
            or not isinstance(raw_invalid_ledger, Mapping)
            or not isinstance(raw_passed_burst, Mapping)
            or not isinstance(raw_retry_plan, Mapping)
        ):
            raise ValueError("confirmation retry artifact typed preimages drifted")
        (
            invalid_ledger_sha256,
            invalid_measured_flops,
            invalid_optimizer_steps,
            invalid_compute_token_slots,
        ) = _validate_confirmation_flop_ledger_payload_v2(raw_invalid_ledger)
        invalid_physical_sha256 = (
            _confirmation_physical_flop_evidence_sha256_v2(
                compute_attempt_id=failed_id,
                execution_plan_sha256=str(failed_attempt.execution_plan_sha256),
                flop_ledger_receipt_sha256=invalid_ledger_sha256,
            )
        )
        if set(raw_passed_burst) != {
            "ordered_step_flops",
            "prelaunch_arm_mean_flops",
            "byte_matched_optimizer_steps",
        }:
            raise ValueError("confirmation retry burst preimage fields drifted")
        retry_burst_flops = raw_passed_burst["ordered_step_flops"]
        retry_arm_mean = raw_passed_burst["prelaunch_arm_mean_flops"]
        retry_byte_steps = raw_passed_burst["byte_matched_optimizer_steps"]
        failed_plan = plan_by_sha256[str(failed_attempt.execution_plan_sha256)]
        if (
            not isinstance(retry_burst_flops, (list, tuple))
            or len(retry_burst_flops) != 100
            or any(type(value) is not int or value < 1 for value in retry_burst_flops)
        ):
            raise ValueError("confirmation retry burst preimage is invalid")
        _require_exact_int(retry_arm_mean, "retry burst arm mean", minimum=1)
        _require_exact_int(retry_byte_steps, "retry burst byte steps", minimum=1)
        ordered_retry_burst = tuple(sorted(retry_burst_flops))
        if (
            200 * (ordered_retry_burst[-1] - ordered_retry_burst[0])
            > ordered_retry_burst[49] + ordered_retry_burst[50]
            or abs(sum(retry_burst_flops) * retry_byte_steps - 100 * retry_arm_mean)
            > retry_arm_mean
            or retry_arm_mean != failed_plan["arm_mean_flops"]
            or retry_byte_steps != failed_plan["byte_matched_optimizer_steps"]
        ):
            raise ValueError("confirmation retry artifact carries a failed burst")
        retry_burst_receipt_sha256 = gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_burst_flops",
            raw_passed_burst,
        )
        retry_physical_burst_sha256 = (
            _confirmation_physical_burst_evidence_sha256_v2(
                compute_attempt_id=failed_id,
                execution_plan_sha256=str(failed_attempt.execution_plan_sha256),
                burst_receipt_sha256=retry_burst_receipt_sha256,
            )
        )
        retry_plan_sha256 = gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_execution_plan",
            raw_retry_plan,
        )
        raw_retry_training_plan = raw_retry_plan.get("training_plan")
        if not isinstance(raw_retry_training_plan, Mapping):
            raise ValueError("confirmation retry plan omits its training-plan preimage")
        retry_slots = raw_retry_training_plan.get("compute_token_slots")
        _require_exact_int(retry_slots, "retry compute_token_slots", minimum=1)
        retry_projected = _ceil_div(
            projection.measured_a100_microseconds * retry_slots,
            projection.measured_tokens,
        ) + (
            projection.measured_heldout_evaluation_a100_microseconds
            * projection.heldout_evaluations_per_full_run
        ) + (
            projection.measured_output_surface_a100_microseconds
            * projection.output_surface_benchmarks_per_full_run
        )
        if (
            retry_payload.get("schema")
            != "weft1_gtok_v2_invalid_confirmation_flop_band"
            or retry_payload.get("correction_ordinal")
            != retry.correction_ordinal
            or retry_payload.get("attempt_receipt_sha256")
            != failed_attempt.receipt_sha256
            or retry_payload.get("failed_execution_plan_sha256")
            != failed_attempt.execution_plan_sha256
            or retry_payload.get("failed_optimizer_steps")
            != invalid_optimizer_steps
            or retry_payload.get("failed_projected_run_a100_microseconds")
            != failed_attempt.projected_run_a100_microseconds
            or retry_payload.get("failed_terminal_lifecycle_event_sha256")
            != failed_terminal.receipt_sha256
            or retry_payload.get("invalid_flop_ledger_receipt_sha256")
            != invalid_ledger_sha256
            or retry_payload.get("invalid_physical_flop_ledger_sha256")
            != invalid_physical_sha256
            or retry_payload.get("passed_burst_receipt_sha256")
            != retry_burst_receipt_sha256
            or retry_payload.get("passed_physical_burst_evidence_sha256")
            != retry_physical_burst_sha256
            or retry.invalid_physical_flop_ledger_sha256
            != invalid_physical_sha256
            or invalid_measured_flops != retry.realized_flops
            or invalid_compute_token_slots
            != failed_attempt.planned_compute_token_slots
            or retry_payload.get("realized_flops") != retry.realized_flops
            or retry_payload.get("target_flops") != retry.target_flops
            or retry_payload.get("retry_steps") != retry.retry_optimizer_steps
            or retry_payload.get("retry_execution_plan_sha256")
            != retry_plan_sha256
            or retry.retry_execution_plan_sha256 != retry_plan_sha256
            or raw_retry_plan.get("retry_of_realized_flops")
            != retry.realized_flops
            or raw_retry_plan.get("retry_of_optimizer_steps")
            != retry.prior_optimizer_steps
            or raw_retry_training_plan.get("optimizer_steps")
            != retry.retry_optimizer_steps
            or retry_payload.get("retry_projected_run_a100_microseconds")
            != retry_projected
        ):
            raise ValueError("confirmation retry artifact evidence does not close")
    for seed_slot, slot_run in by_slot.items():
        slot_retries = tuple(
            row
            for row in evidence_closure.retry_joins
            if row.seed_slot == seed_slot
        )
        if slot_retries:
            for prior, successor in zip(slot_retries, slot_retries[1:]):
                if (
                    successor.failed_execution_plan_sha256
                    != prior.retry_execution_plan_sha256
                ):
                    raise ValueError("confirmation correction plan chain is discontinuous")
            if slot_retries[-1].retry_execution_plan_sha256 != slot_run.execution_plan_sha256:
                raise ValueError("confirmation selected run did not use the final correction plan")
    joins_by_slot = {row.seed_slot: row for row in evidence_closure.fresh_joins}
    for run in runs:
        join = joins_by_slot.get(run.seed_slot)
        if join is None:
            raise ValueError("confirmation closure omits a successful fresh slot")
        expected = {
            "vocab_size": run.vocab_size,
            "fresh_run_receipt_sha256": run.receipt_sha256,
            "confirmation_order_receipt_sha256": (
                run.confirmation_order_receipt_sha256
            ),
            "physical_d6_evidence_sha256": run.physical_d6_evidence_sha256,
            "document_multiset_sha256": run.document_multiset_sha256,
            "ordered_raw_content_ids_sha256": run.data_order_sha256,
            "framed_payload_sha256": run.framed_payload_sha256,
            "order_document_count": run.stream_docs,
            "order_retained_text_bytes": run.stream_bytes,
            "execution_plan_sha256": run.execution_plan_sha256,
            "training_plan_sha256": run.training_plan_sha256,
            "compute_attempt_id": run.compute_attempt_id,
        }
        if any(getattr(join, name) != value for name, value in expected.items()):
            raise ValueError("confirmation closure does not join Q3, plan, run, and attempt")
        terminal = terminal_by_sha256.get(join.terminal_lifecycle_event_sha256)
        if (
            terminal is None
            or terminal.attempt_id != run.compute_attempt_id
            or terminal.terminal_status != "completed"
            or terminal.gpu_uuid_provenance != run.gpu_uuid_provenance
            or terminal.charged_a100_microseconds
            != run.measured_a100_microseconds
            or not isinstance(terminal.completion_payload, Mapping)
        ):
            raise ValueError("confirmation fresh join lacks its completed lifecycle preimage")
        completion = terminal.completion_payload
        successful_plan = plan_by_sha256[run.execution_plan_sha256]
        successful_training_plan = successful_plan["training_plan"]
        assert isinstance(successful_training_plan, Mapping)
        q2_fields = (
            "stream_bytes",
            "stream_tokens",
            "stream_docs",
            "trained_bytes",
            "trained_tokens",
            "trained_docs_full",
            "dropped_bytes",
            "dropped_tokens",
            "dropped_docs",
            "boundary_doc_id",
            "boundary_doc_consumed_tokens",
        )
        if any(
            successful_training_plan[name] != getattr(run, name)
            for name in q2_fields
        ):
            raise ValueError("confirmation Q2 execution plan differs from realized run")
        required_completion_fields = {
            "run",
            "flop_ledger",
            "execution_plan_sha256",
            "base_flop_evidence_sha256",
            "training_plan_sha256",
            "heldout_evaluation_steps",
            "burst_flop_receipt",
            "physical_flop_ledger_sha256",
            "physical_optimizer_steps",
            "training_runtime_receipt_sha256",
            "code_closure_receipt_sha256",
            "checkpoint_retained",
        }
        if set(completion) != required_completion_fields:
            raise ValueError("confirmation completion evidence fields drifted")
        raw_run = completion["run"]
        raw_burst = completion["burst_flop_receipt"]
        raw_ledger = completion["flop_ledger"]
        if (
            not isinstance(raw_run, Mapping)
            or canonical_json_bytes(raw_run) != canonical_json_bytes(asdict(run))
            or not isinstance(raw_burst, Mapping)
            or not isinstance(raw_ledger, Mapping)
        ):
            raise ValueError("confirmation completion preimages differ from the fresh run")
        if set(raw_burst) != {
            "ordered_step_flops",
            "prelaunch_arm_mean_flops",
            "byte_matched_optimizer_steps",
        }:
            raise ValueError("confirmation burst preimage fields drifted")
        burst_flops = raw_burst["ordered_step_flops"]
        arm_mean_flops = raw_burst["prelaunch_arm_mean_flops"]
        byte_matched_steps = raw_burst["byte_matched_optimizer_steps"]
        if (
            not isinstance(burst_flops, (list, tuple))
            or len(burst_flops) != 100
            or any(type(value) is not int or value < 1 for value in burst_flops)
        ):
            raise ValueError("confirmation burst preimage requires 100 FLOP values")
        _require_exact_int(arm_mean_flops, "prelaunch_arm_mean_flops", minimum=1)
        _require_exact_int(byte_matched_steps, "byte_matched_optimizer_steps", minimum=1)
        ordered_burst = tuple(sorted(burst_flops))
        if (
            200 * (ordered_burst[-1] - ordered_burst[0])
            > ordered_burst[49] + ordered_burst[50]
            or abs(sum(burst_flops) * byte_matched_steps - 100 * arm_mean_flops)
            > arm_mean_flops
        ):
            raise ValueError("confirmation burst preimage fails its governed gates")
        raw_burst_sha256 = gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_burst_flops",
            raw_burst,
        )
        expected_burst_evidence_sha256 = (
            _confirmation_physical_burst_evidence_sha256_v2(
                compute_attempt_id=run.compute_attempt_id,
                execution_plan_sha256=run.execution_plan_sha256,
                burst_receipt_sha256=raw_burst_sha256,
            )
        )
        (
            raw_ledger_sha256,
            ledger_measured_flops,
            ledger_optimizer_steps,
            ledger_compute_token_slots,
        ) = _validate_confirmation_flop_ledger_payload_v2(raw_ledger)
        expected_flop_evidence_sha256 = (
            _confirmation_physical_flop_evidence_sha256_v2(
                compute_attempt_id=run.compute_attempt_id,
                execution_plan_sha256=run.execution_plan_sha256,
                flop_ledger_receipt_sha256=raw_ledger_sha256,
            )
        )
        if (
            join.burst_receipt_sha256 != expected_burst_evidence_sha256
            or join.physical_flop_ledger_sha256
            != expected_flop_evidence_sha256
            or completion["physical_flop_ledger_sha256"]
            != expected_flop_evidence_sha256
            or ledger_measured_flops != run.measured_flops
            or ledger_optimizer_steps != run.terminal.optimizer_step
            or ledger_compute_token_slots != run.trained_tokens
            or completion["physical_optimizer_steps"] != ledger_optimizer_steps
            or completion["execution_plan_sha256"] != run.execution_plan_sha256
            or completion["base_flop_evidence_sha256"]
            != successful_plan["arm_flop_plan_sha256"]
            or completion["training_plan_sha256"] != run.training_plan_sha256
            or completion["training_plan_sha256"]
            != plan_training_sha256[run.execution_plan_sha256]
            or arm_mean_flops != successful_plan["arm_mean_flops"]
            or byte_matched_steps
            != successful_plan["byte_matched_optimizer_steps"]
            or tuple(completion["heldout_evaluation_steps"])
            != tuple(item.optimizer_step for item in run.observations)
            or completion["training_runtime_receipt_sha256"]
            != run.training_runtime_receipt_sha256
            or completion["code_closure_receipt_sha256"]
            != run.code_closure_receipt_sha256
            or completion["checkpoint_retained"] is not False
        ):
            raise ValueError("confirmation physical burst/FLOP evidence does not close")
    ordered = tuple(sorted(runs, key=lambda item: item.seed_slot))

    slots: list[ConfirmationResultSlotV2] = []
    for vocab_size in pair:
        for seed_slot, base_seed in enumerate(matrix.seeds):
            base_run = base_by_key[(vocab_size, base_seed)]
            if vocab_size == reused_vocab_size:
                source = base_run
                slots.append(
                    ConfirmationResultSlotV2(
                        vocab_size=vocab_size,
                        seed_slot=seed_slot,
                        source="reused_byte_matched",
                        run_seed=base_seed,
                        registry_key="",
                        source_run_receipt_sha256=base_run.receipt_sha256,
                        paired_base_run_receipt_sha256=base_run.receipt_sha256,
                        compute_attempt_id=None,
                        observations=base_run.observations,
                        stream_bytes=int(base_run.stream_bytes),
                        stream_docs=int(base_run.stream_docs),
                        stream_tokens=int(base_run.stream_tokens),
                        trained_tokens=int(base_run.trained_tokens),
                        dropped_tokens=int(base_run.dropped_tokens),
                        trained_bytes=int(base_run.trained_bytes),
                        dropped_bytes=int(base_run.dropped_bytes),
                        trained_docs_full=int(base_run.trained_docs_full),
                        boundary_doc_id=base_run.boundary_doc_id,
                        boundary_doc_consumed_tokens=(
                            base_run.boundary_doc_consumed_tokens
                        ),
                        dropped_docs=int(base_run.dropped_docs),
                    )
                )
            else:
                source = by_slot[seed_slot]
                slots.append(
                    ConfirmationResultSlotV2(
                        vocab_size=vocab_size,
                        seed_slot=seed_slot,
                        source="fresh_confirmation",
                        run_seed=source.seed,
                        registry_key=source.registry_key,
                        source_run_receipt_sha256=source.receipt_sha256,
                        paired_base_run_receipt_sha256=(
                            source.base_run_receipt_sha256
                        ),
                        compute_attempt_id=source.compute_attempt_id,
                        observations=source.observations,
                        stream_bytes=source.stream_bytes,
                        stream_docs=source.stream_docs,
                        stream_tokens=source.stream_tokens,
                        trained_tokens=source.trained_tokens,
                        dropped_tokens=source.dropped_tokens,
                        trained_bytes=source.trained_bytes,
                        dropped_bytes=source.dropped_bytes,
                        trained_docs_full=source.trained_docs_full,
                        boundary_doc_id=source.boundary_doc_id,
                        boundary_doc_consumed_tokens=(
                            source.boundary_doc_consumed_tokens
                        ),
                        dropped_docs=source.dropped_docs,
                    )
                )
    result_slots = tuple(sorted(slots, key=lambda item: (item.vocab_size, item.seed_slot)))
    if any(
        observation.heldout_stream_sha256 != matrix.corpus.heldout_stream_sha256
        or observation.denominator_signature != denominator
        for slot in result_slots
        for observation in slot.observations
    ):
        raise ValueError("confirmation result curves do not share the frozen H denominator")
    values_by_vocab = {
        vocab_size: tuple(
            next(
                item.terminal_bpb
                for item in result_slots
                if item.vocab_size == vocab_size and item.seed_slot == seed_slot
            )
            for seed_slot in (0, 1)
        )
        for vocab_size in pair
    }
    raw_means = {
        vocab_size: math.fsum(values) / GTOK_SEED_COUNT
        for vocab_size, values in values_by_vocab.items()
    }
    rho = {vocab_size: _rho_bpb_micros(value) for vocab_size, value in raw_means.items()}
    sample_sd = {
        vocab_size: math.sqrt(
            math.fsum((value - raw_means[vocab_size]) ** 2 for value in values)
        )
        for vocab_size, values in values_by_vocab.items()
    }
    winner, runner_up = pair
    s_hat_c = math.sqrt(
        (sample_sd[winner] ** 2 + sample_sd[runner_up] ** 2) / 2.0
    )
    delta_micros = rho[winner] - rho[runner_up]
    multiplier = 3 if runner_up > winner else 2
    threshold = multiplier * s_hat_c
    slot_delta = tuple(
        values_by_vocab[winner][seed_slot] - values_by_vocab[runner_up][seed_slot]
        for seed_slot in (0, 1)
    )
    if slot_delta[0] * slot_delta[1] < 0:
        status = "ESCALATE_SEED_SPLIT"
    elif (
        slot_delta[0] > 0
        and slot_delta[1] > 0
        and delta_micros > threshold * GTOK_RHO_BPB_SCALE
    ):
        status = "ESCALATE_REVERSAL"
    else:
        status = "GREEN_NO_REVERSAL"
    return ValidatedComputeConfirmationV2._validated(
        selection=selection,
        matrix=matrix,
        compute=compute,
        common_flop_budget=common_flop_budget,
        runs=ordered,
        evidence_closure=evidence_closure,
        reused_vocab_size=reused_vocab_size,
        fresh_vocab_size=fresh_vocab_size,
        result_slots=result_slots,
        rho_bpb_micros=((winner, rho[winner]), (runner_up, rho[runner_up])),
        sample_sd_bpb=(
            (winner, sample_sd[winner]),
            (runner_up, sample_sd[runner_up]),
        ),
        s_hat_c_bpb=s_hat_c,
        delta_bpb_micros=delta_micros,
        threshold_multiplier=multiplier,
        threshold_bpb=threshold,
        slot_delta_bpb=(slot_delta[0], slot_delta[1]),
        status=status,
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
                selection.selector_literal_binding_sha256
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
    "A2FirstFitGroupReceiptV2",
    "A2FirstFitScreenReceiptV2",
    "ArmCalibrationProjectionV2",
    "ArmTerminalStatisticsV2",
    "BpbMilestoneReceiptV2",
    "CampaignComputeReceiptV2",
    "ComputeAttemptReceiptV2",
    "ComputeConfirmationRunV2",
    "ConfirmationArmFlopSourceEnvelopeV2",
    "ConfirmationAttemptLaunchEnvelopeV2",
    "ConfirmationBaseRunFlopSourceEnvelopeV2",
    "ConfirmationEvidenceClosureV2",
    "ConfirmationExecutionPlanEnvelopeV2",
    "ConfirmationFreshEvidenceJoinV2",
    "ConfirmationLifecycleEventEvidenceV2",
    "ConfirmationOrderEnvelopeV2",
    "ConfirmationRetryArtifactEnvelopeV2",
    "ConfirmationRetryEvidenceJoinV2",
    "ConfirmationResultSlotV2",
    "FrozenScreenCorpusV2",
    "GTOK_A2_BINDINGS_SHA256",
    "GTOK_AMENDMENT_A2_SHA256",
    "GTOK_AMENDMENT_A3_SHA256",
    "GTOK_CALIBRATION_MAX_STEPS",
    "GTOK_COMPUTE_SCOPES",
    "GTOK_CONFIRMATION_SEMANTICS_SHA256",
    "GTOK_FIRST_BOUNDARY_BYTES",
    "GTOK_MILESTONE_LABELS",
    "GTOK_PER_RUN_WATCHDOG_MULTIPLIER",
    "GTOK_RELEASE_CLOSE_SHA256",
    "GTOK_RHO_BPB_DECIMAL_PLACES",
    "GTOK_RHO_BPB_SCALE",
    "GTOK_SECOND_BOUNDARY_BYTES",
    "GTOK_SELECTION_CONFIRMATION_AUTHORITY_CHAIN",
    "GTOK_SELECTION_CONFIRMATION_AUTHORITY_SHA256",
    "GTOK_SELECTION_CONFIRMATION_LITERAL_BINDING_SHA256_V2",
    "GTOK_SELECTION_CONFIRMATION_LITERAL_BINDING_V2",
    "GTOK_SEMANTICS_AMENDMENT_S1_SHA256",
    "GTOK_SEMANTICS_AMENDMENT_S2_SHA256",
    "GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2",
    "GTOK_SELECTOR_LITERAL_BINDING_V2",
    "GTOK_TERMINAL_BUDGET",
    "GTOK_TERMINAL_METRIC",
    "GTOK_TRIPWIRE_A100_MICROSECONDS",
    "GTOK_V2_AUTHORITY_CHAIN",
    "GTokRunReceiptV2",
    "GTokSelectionReceiptV2",
    "GTokV2Stop",
    "PrecalibrationReplayAttemptReceiptV2",
    "PreflightProjectionReceiptV2",
    "RuntimeTripwireSnapshotV2",
    "SelectionComparisonV2",
    "TokenizerArmReceiptV2",
    "ValidatedComputeConfirmationV2",
    "ValidatedGTokMatrixV2",
    "VocabExtBasisV2",
    "VocabularyAdmissibilityReceiptV2",
    "VocabularyFreezeArtifactV2",
    "compute_event_ledger_sha256_v2",
    "enforce_runtime_tripwire_v2",
    "gtok_v2_bound_sha256",
    "mint_vocabulary_freeze_v2",
    "refuse_unvalidated_freeze",
    "select_vocabulary_v2",
    "validate_complete_gtok_matrix_v2",
    "validate_compute_confirmation_v2",
    "validate_selection_receipt_v2",
]
