"""Choice-independent contracts for the WEFT-1 G-TOK screen.

This module is deliberately contract-only.  It does not fit a tokenizer,
construct an optimizer, train a model, read a corpus, freeze an artifact, or
select a vocabulary.  Amendment A1 authorizes the bounded run axis and settles
the block count, byte targets, optimizer values, high-level tokenizer recipe,
dedup dimensions, and RNG namespaces.  Execution still fails closed wherever
the amendment does not identify a reproducible literal implementation.

Append-only vocabulary continuation preserves the byte meaning of every old
token ID and the old merge list as an exact prefix.  It does *not* imply that
old text has segmentation invariance: appended merges may change how an old
byte string is segmented while all pre-existing token meanings remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, NoReturn

from models.ablation_lm.config import AblationLMConfig


GTOK_TRAINING_BYTE_BUDGET = 4_000_000_000
GTOK_TRAINING_BYTE_CEILING = 4_000_000_000
GTOK_A100_HOUR_TRIPWIRE = 12.0
GTOK_VOCABULARY_ARMS = (16_384, 24_576, 32_768, 49_152)
GTOK_SEED_COUNT = 2
GTOK_HELDOUT_BYTE_TARGET = 80_000_000
GTOK_PROJECTED_A100_HOURS = 6.5
GTOK_HANDOFF_SHA256 = "498f34b5966f0879c7f0a15ca8be02a603558781c35f59f03fb29cc9edd3eb02"
GTOK_RULINGS_SHA256 = "167fc17da1ac71a8263f5e190dc07dcd681ed82c64123a3394dc2b47e42cf0d2"
GTOK_ENGLISH_SCOPE_SHA256 = (
    "19399342cb6233258ac2ba411b6dc1feaab101c3f3986d751b6debe20dee02d3"
)
GTOK_CURRICULUM_AMENDMENT_SHA256 = (
    "0221545d62f7ed189898abf56f1ca65be6683de4d8a396d80bae4a4a094065b5"
)
GTOK_RATIFICATION_SHA256 = (
    "c5df74297594e75697ffb71d8d05d75efcf94f7857d55ddd357043200efb6d3a"
)
# Later authorities are kept separate from ``GTOK_AUTHORITY_CHAIN`` so the
# hashes of already-banked design receipts do not change retroactively.
GTOK_CURRICULUM_DATA_SHA256 = (
    "14f0ba5d32898d69413839b8e342cc74b858eef90e65079175e37968052dea22"
)
GTOK_QWEN_ADJUDICATION_SHA256 = (
    "6c2568d5ba7f8295c65493b863d0530e71ee78e2290455307b00bdcdee480a1f"
)
GTOK_CURRICULUM_DECISIONS_SHA256 = (
    "61fc7727e456d822f43613db602c0251344b64ea92c7b256af5f1fe560cd8b6d"
)
GTOK_EXECUTION_HANDOFF_SHA256 = (
    "2aecb64711a2bf2776c8d1940350bc5d42b335f60eb774ac1e941f470b9cf74c"
)
GTOK_AMENDMENT_A1_SHA256 = (
    "e996f89fee81871a6432d90fabbaa0dc470b8f7643bc65756966e27883af3267"
)
GTOK_AUTHORITY_CHAIN = (
    GTOK_HANDOFF_SHA256,
    GTOK_RULINGS_SHA256,
    GTOK_ENGLISH_SCOPE_SHA256,
    GTOK_CURRICULUM_AMENDMENT_SHA256,
    GTOK_RATIFICATION_SHA256,
)
GTOK_EXECUTION_AUTHORITY_CHAIN = (
    GTOK_HANDOFF_SHA256,
    GTOK_RATIFICATION_SHA256,
    GTOK_RULINGS_SHA256,
    GTOK_ENGLISH_SCOPE_SHA256,
    GTOK_CURRICULUM_AMENDMENT_SHA256,
    GTOK_CURRICULUM_DATA_SHA256,
    GTOK_QWEN_ADJUDICATION_SHA256,
    GTOK_CURRICULUM_DECISIONS_SHA256,
    GTOK_EXECUTION_HANDOFF_SHA256,
)
# A1 explicitly amends forward without re-keying any banked execution receipt.
# New v2 artifacts use this chain and a separate hash domain; the v1 chain and
# ``execution_authority_bound_sha256`` above remain byte-for-byte stable.
GTOK_EXECUTION_AUTHORITY_CHAIN_V2 = (
    *GTOK_EXECUTION_AUTHORITY_CHAIN,
    GTOK_AMENDMENT_A1_SHA256,
)
GTOK_BPB_MILESTONE_FRACTIONS = (
    Fraction(1, 4),
    Fraction(1, 2),
    Fraction(1, 1),
)
GTOK_BPB_MILESTONE_BYTES = {
    fraction: GTOK_TRAINING_BYTE_BUDGET * fraction.numerator // fraction.denominator
    for fraction in GTOK_BPB_MILESTONE_FRACTIONS
}
GTOK_STRATA = ("general", "code", "mathematics", "science_technical")
GTOK_STRATUM_SHARES = (
    Fraction(45, 100),
    Fraction(25, 100),
    Fraction(15, 100),
    Fraction(15, 100),
)
GTOK_SCREEN_TRAIN_STRATUM_TARGETS = tuple(
    (name, GTOK_TRAINING_BYTE_BUDGET * share.numerator // share.denominator)
    for name, share in zip(GTOK_STRATA, GTOK_STRATUM_SHARES, strict=True)
)
GTOK_SCREEN_HELDOUT_STRATUM_TARGETS = tuple(
    (name, GTOK_HELDOUT_BYTE_TARGET * share.numerator // share.denominator)
    for name, share in zip(GTOK_STRATA, GTOK_STRATUM_SHARES, strict=True)
)
GTOK_STRATUM_TOLERANCE = Fraction(5, 1_000)
GTOK_PRETOKENIZER_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| "
    r"?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
GTOK_TOKENIZER_LIBRARY = "tokenizers"
GTOK_TOKENIZER_FAMILY = "byte_level_bpe"
GTOK_TOKENIZER_MIN_FREQUENCY = 2
GTOK_PIPELINE_RNG_NAMES = (
    "corpus.dedup",
    "corpus.shuffle",
    "corpus.split",
    "corpus.topup",
    "gtok.bpe",
)
GTOK_RUN_RNG_NAME_TEMPLATES = (
    "gtok.data.{arm}.{seed}",
    "gtok.init.{arm}.{seed}",
)
GTOK_ROUND_TRIP_CATEGORIES = (
    "accented_latin",
    "cjk",
    "greek",
    "mixed_indentation",
    "right_to_left",
    "tabs",
    "typographic_punctuation",
)
GTOK_COMPUTE_SCOPES = ("base_screen", "confirmation", "infrastructure", "pilot")
GTOK_COMPUTE_STATUSES = ("aborted", "cancelled", "completed", "failed", "preempted")
UNRESOLVED_GTOK_DECISIONS = (
    "per-family source-route admissibility and quality-selection tie-breaks",
    "run termination and BPB milestone semantics when document-floor T is below four billion bytes",
    "language-ID package/model/version, exact threshold, and boundary tie behavior",
    "literal NFC whitespace-collapse algorithm and Unicode version",
    "production MinHash hash family/seed/framing, short-document rule, and candidate ordering",
    "shard serialization/compression/framing and manifest self-hash exclusion rules",
    "tokenizers version, ordered reserved-token strings/IDs, decoder/post-processor, and BPE tie behavior",
    "the two run-seed values and a realizable role for gtok.bpe in a deterministic trainer with no seed argument",
    "undertrained-row norm threshold, packing/final-batch/scheduler-step semantics, and measurement/runtime receipt schemas",
    "pre-dispatch and in-flight enforcement of the cumulative 12 A100-hour tripwire",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GTokExecutionBlocked(RuntimeError):
    """Raised when choice-dependent G-TOK work is requested prematurely."""


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Fraction):
        return {
            "denominator": value.denominator,
            "numerator": value.numerator,
        }
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical mappings require string keys")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical payloads may not contain non-finite floats")
        return value
    raise TypeError(f"unsupported canonical payload type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a receipt deterministically for hashing and comparison."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def authority_bound_sha256(schema: str, value: Any) -> str:
    """Hash a domain-separated receipt under the exact governing authority chain."""

    if not isinstance(schema, str) or not schema.strip():
        raise ValueError("receipt schema must be a nonempty string")
    return canonical_sha256(
        {
            "authority_chain": GTOK_AUTHORITY_CHAIN,
            "payload": value,
            "schema": schema,
        }
    )


def execution_authority_bound_sha256(schema: str, value: Any) -> str:
    """Hash a new execution receipt without rewriting banked design receipts."""

    if not isinstance(schema, str) or not schema.strip():
        raise ValueError("receipt schema must be a nonempty string")
    return canonical_sha256(
        {
            "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN,
            "payload": value,
            "schema": schema,
        }
    )


def execution_authority_v2_bound_sha256(schema: str, value: Any) -> str:
    """Hash an A1-era v2 receipt without changing any banked v1 digest."""

    if not isinstance(schema, str) or not schema.strip():
        raise ValueError("receipt schema must be a nonempty string")
    if not schema.endswith("_v2"):
        raise ValueError("A1 execution receipts require an explicit v2 schema")
    return canonical_sha256(
        {
            "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V2,
            "payload": value,
            "schema": schema,
        }
    )


@dataclass(frozen=True)
class GTokProxyTopologyReceipt:
    """Exact S0 body shared by every G-TOK vocabulary arm.

    Vocabulary size and seed identities are bound separately by
    :class:`GTokRunReceipt`.  The two middle blocks execute once as ordinary
    dense blocks, while the recurrent loop and every current optional module
    are structurally absent.
    """

    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1_408
    n_prelude_layers: int = 4
    n_core_blocks: int = 2
    n_coda_layers: int = 4
    max_sequence_length: int = 2_048
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    tie_embeddings: bool = True
    recurrent_steps: int = 1
    max_recurrent_steps: int = 8
    recurrence_coefficient: float = 1.0
    recurrence_exponent: float = 1.0
    use_recurrence: bool = False
    use_static_kv_core: bool = False
    static_kv_midpoint_refresh: bool = False
    use_front_hadamard_experts: bool = False
    use_reentry_bridge: bool = False
    use_scratch: bool = False
    use_lane_carrier: bool = False
    use_engram: bool = False
    use_long_term_memory: bool = False
    z_loss_coefficient: float = 0.0
    vocabulary_binding: str = "gtok_run_arm"

    def __post_init__(self) -> None:
        expected = {
            "d_model": 512,
            "n_heads": 8,
            "n_kv_heads": 4,
            "d_ff": 1_408,
            "n_prelude_layers": 4,
            "n_core_blocks": 2,
            "n_coda_layers": 4,
            "max_sequence_length": 2_048,
            "rope_theta": 500_000.0,
            "norm_eps": 1e-5,
            "attention_dropout": 0.0,
            "tie_embeddings": True,
            "recurrent_steps": 1,
            "max_recurrent_steps": 8,
            "recurrence_coefficient": 1.0,
            "recurrence_exponent": 1.0,
            "use_recurrence": False,
            "use_static_kv_core": False,
            "static_kv_midpoint_refresh": False,
            "use_front_hadamard_experts": False,
            "use_reentry_bridge": False,
            "use_scratch": False,
            "use_lane_carrier": False,
            "use_engram": False,
            "use_long_term_memory": False,
            "z_loss_coefficient": 0.0,
            "vocabulary_binding": "gtok_run_arm",
        }
        for name, expected_value in expected.items():
            actual = getattr(self, name)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(
                    "G-TOK proxy topology must equal the exact ratified 4/2/4 S0 graph"
                )

    @classmethod
    def from_config(cls, config: AblationLMConfig) -> "GTokProxyTopologyReceipt":
        """Validate one vocabulary-arm config and return its shared body receipt."""

        if not isinstance(config, AblationLMConfig):
            raise TypeError("G-TOK proxy topology requires an AblationLMConfig")
        if config.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("G-TOK proxy config uses an unregistered vocabulary arm")
        return cls(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            d_ff=config.d_ff,
            n_prelude_layers=config.n_prelude_layers,
            n_core_blocks=config.n_core_blocks,
            n_coda_layers=config.n_coda_layers,
            max_sequence_length=config.max_sequence_length,
            rope_theta=float(config.rope_theta),
            norm_eps=float(config.norm_eps),
            attention_dropout=float(config.attention_dropout),
            tie_embeddings=config.tie_embeddings,
            recurrent_steps=config.recurrent_steps,
            max_recurrent_steps=config.max_recurrent_steps,
            recurrence_coefficient=float(config.recurrence_coefficient),
            recurrence_exponent=float(config.recurrence_exponent),
            use_recurrence=config.use_recurrence,
            use_static_kv_core=config.use_static_kv_core,
            static_kv_midpoint_refresh=config.static_kv_midpoint_refresh,
            use_front_hadamard_experts=config.use_front_hadamard_experts,
            use_reentry_bridge=config.use_reentry_bridge,
            use_scratch=config.use_scratch,
            use_lane_carrier=config.use_lane_carrier,
            use_engram=config.use_engram,
            use_long_term_memory=config.use_long_term_memory,
            z_loss_coefficient=float(config.z_loss_coefficient),
        )

    @property
    def receipt_sha256(self) -> str:
        return authority_bound_sha256("weft1_gtok_proxy_topology_v1", self)

    @property
    def executing_block_count(self) -> int:
        """Count dense blocks; structural-OFF never removes the core blocks."""

        return self.n_prelude_layers + self.n_core_blocks + self.n_coda_layers


GTOK_PROXY_TOPOLOGY = GTokProxyTopologyReceipt()
if GTOK_PROXY_TOPOLOGY.executing_block_count != 10:  # pragma: no cover - import invariant
    raise RuntimeError("the ratified G-TOK 4/2/4 proxy must execute ten dense blocks")
GTOK_PROXY_TOPOLOGY_SHA256 = GTOK_PROXY_TOPOLOGY.receipt_sha256


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes without text decoding or normalization."""

    if not isinstance(value, bytes):
        raise TypeError("sha256_bytes requires bytes")
    return hashlib.sha256(value).hexdigest()


def bits_per_byte(nll_nats: float, raw_byte_count: int) -> float:
    """Compute BPB as summed NLL nats divided by ``ln(2) * raw bytes``."""

    if isinstance(nll_nats, bool) or not isinstance(nll_nats, Real):
        raise TypeError("nll_nats must be numeric")
    value = float(nll_nats)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("nll_nats must be finite and non-negative")
    if type(raw_byte_count) is not int or raw_byte_count < 1:
        raise ValueError("raw_byte_count must be a positive integer")
    return value / (math.log(2.0) * raw_byte_count)


def validate_a100_hour_tripwire(a100_hours: float) -> float:
    """Return measured screen hours, stopping at the binding 12-hour tripwire."""

    if isinstance(a100_hours, bool) or not isinstance(a100_hours, Real):
        raise TypeError("a100_hours must be numeric")
    value = float(a100_hours)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("a100_hours must be finite and non-negative")
    if value > GTOK_A100_HOUR_TRIPWIRE:
        raise RuntimeError("G-TOK exceeded the 12 A100-hour tripwire")
    return value


@dataclass(frozen=True)
class CorpusStratumByteStats:
    """C1/C3 byte statistics for one frozen-corpus stratum."""

    name: str
    raw_byte_count: int
    non_ascii_byte_count: int
    non_ascii_fraction: Fraction

    def __post_init__(self) -> None:
        if self.name not in GTOK_STRATA:
            raise ValueError(f"unknown G-TOK stratum: {self.name!r}")
        if type(self.raw_byte_count) is not int or self.raw_byte_count < 1:
            raise ValueError("stratum raw_byte_count must be a positive integer")
        if (
            type(self.non_ascii_byte_count) is not int
            or not 0 <= self.non_ascii_byte_count <= self.raw_byte_count
        ):
            raise ValueError("non_ascii_byte_count must lie within the stratum")
        if not isinstance(self.non_ascii_fraction, Fraction):
            raise TypeError("non_ascii_fraction must be an exact Fraction")
        expected = Fraction(self.non_ascii_byte_count, self.raw_byte_count)
        if self.non_ascii_fraction != expected:
            raise ValueError("recorded non-ASCII fraction disagrees with byte counts")


@dataclass(frozen=True)
class ByteRoundTripFixtureReceipt:
    """C2 evidence for one deliberately adversarial byte fixture."""

    fixture_id: str
    categories: tuple[str, ...]
    original_bytes_sha256: str
    round_trip_bytes_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.fixture_id, str) or not self.fixture_id.strip():
            raise ValueError("fixture_id must be nonempty")
        if not isinstance(self.categories, tuple) or not self.categories:
            raise ValueError("round-trip categories must be a nonempty tuple")
        if any(not isinstance(category, str) for category in self.categories):
            raise TypeError("round-trip categories must be strings")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("round-trip categories must be unique")
        if tuple(sorted(self.categories)) != self.categories:
            raise ValueError("round-trip categories must use canonical sorted order")
        unknown = set(self.categories) - set(GTOK_ROUND_TRIP_CATEGORIES)
        if unknown:
            raise ValueError(f"unknown round-trip categories: {sorted(unknown)!r}")
        _require_sha256(self.original_bytes_sha256, "original_bytes_sha256")
        _require_sha256(self.round_trip_bytes_sha256, "round_trip_bytes_sha256")
        if self.original_bytes_sha256 != self.round_trip_bytes_sha256:
            raise ValueError("C2 exact byte round-trip hash differs")


@dataclass(frozen=True)
class CorpusAuditManifestReceipt:
    """Choice-independent C1-C3 manifest evidence; not a corpus-freeze grant."""

    source_file_manifest_sha256: str
    document_split_manifest_sha256: str
    training_corpus_sha256: str
    heldout_corpus_sha256: str
    strata: tuple[CorpusStratumByteStats, ...]
    heldout_stratum_raw_byte_counts: tuple[tuple[str, int], ...]
    round_trip_fixtures: tuple[ByteRoundTripFixtureReceipt, ...]
    training_raw_byte_count: int
    heldout_raw_byte_count: int
    heldout_fraction: Fraction
    document_overlap_count: int
    language_filter_method: str
    language_filter_threshold: Fraction
    language_filter_audit_sha256: str
    language_filtered_strata: tuple[str, ...] = ("general",)
    byte_filtering_applied: bool = False

    def __post_init__(self) -> None:
        _require_sha256(self.source_file_manifest_sha256, "source_file_manifest_sha256")
        _require_sha256(
            self.document_split_manifest_sha256,
            "document_split_manifest_sha256",
        )
        _require_sha256(self.training_corpus_sha256, "training_corpus_sha256")
        _require_sha256(self.heldout_corpus_sha256, "heldout_corpus_sha256")
        _require_sha256(
            self.language_filter_audit_sha256,
            "language_filter_audit_sha256",
        )
        if self.training_corpus_sha256 == self.heldout_corpus_sha256:
            raise ValueError("training and held-out corpus hashes must differ")
        if not isinstance(self.strata, tuple):
            raise TypeError("strata must be a tuple")
        if any(not isinstance(item, CorpusStratumByteStats) for item in self.strata):
            raise TypeError("strata must contain CorpusStratumByteStats receipts")
        if tuple(item.name for item in self.strata) != GTOK_STRATA:
            raise ValueError("C3 requires one canonically ordered row per G-TOK stratum")
        total_raw_bytes = sum(item.raw_byte_count for item in self.strata)
        if type(self.training_raw_byte_count) is not int or self.training_raw_byte_count < 1:
            raise ValueError("training_raw_byte_count must be a positive integer")
        if type(self.heldout_raw_byte_count) is not int or self.heldout_raw_byte_count < 1:
            raise ValueError("heldout_raw_byte_count must be a positive integer")
        if self.training_raw_byte_count + self.heldout_raw_byte_count != total_raw_bytes:
            raise ValueError("training and held-out bytes must equal the stratum total")
        if not isinstance(self.heldout_fraction, Fraction):
            raise TypeError("heldout_fraction must be an exact Fraction")
        if self.heldout_fraction != Fraction(1, 50):
            raise ValueError("the frozen held-out slice must be exactly two percent")
        if Fraction(self.heldout_raw_byte_count, total_raw_bytes) != self.heldout_fraction:
            raise ValueError("held-out byte counts disagree with the frozen fraction")
        if not isinstance(self.heldout_stratum_raw_byte_counts, tuple):
            raise TypeError("heldout_stratum_raw_byte_counts must be a tuple")
        heldout_names: list[str] = []
        heldout_counts: list[int] = []
        for row in self.heldout_stratum_raw_byte_counts:
            if not isinstance(row, tuple) or len(row) != 2:
                raise TypeError("held-out stratum rows must be (name, raw_bytes) tuples")
            name, raw_bytes = row
            if name not in GTOK_STRATA:
                raise ValueError(f"unknown held-out stratum: {name!r}")
            if type(raw_bytes) is not int or raw_bytes < 1:
                raise ValueError("held-out stratum bytes must be positive integers")
            heldout_names.append(name)
            heldout_counts.append(raw_bytes)
        if tuple(heldout_names) != GTOK_STRATA:
            raise ValueError("held-out strata must use canonical G-TOK order")
        if sum(heldout_counts) != self.heldout_raw_byte_count:
            raise ValueError("held-out stratum bytes must equal the held-out total")
        for item, expected_share in zip(
            self.strata,
            GTOK_STRATUM_SHARES,
            strict=True,
        ):
            if Fraction(item.raw_byte_count, total_raw_bytes) != expected_share:
                raise ValueError("corpus strata must use the frozen 45/25/15/15 byte mix")
        for item, heldout_count in zip(self.strata, heldout_counts, strict=True):
            if Fraction(heldout_count, item.raw_byte_count) != self.heldout_fraction:
                raise ValueError("the two-percent holdout must be stratified by raw bytes")
        if type(self.document_overlap_count) is not int or self.document_overlap_count != 0:
            raise ValueError("training and held-out document sets must be disjoint")
        if not isinstance(self.language_filter_method, str) or not self.language_filter_method:
            raise ValueError("document-level language filter method must be recorded")
        if self.language_filter_method.strip().casefold() in {
            "disabled",
            "none",
            "not_applied",
        }:
            raise ValueError("the general stratum requires a document-level language filter")
        if not isinstance(self.language_filter_threshold, Fraction):
            raise TypeError("language_filter_threshold must be an exact Fraction")
        if not 0 < self.language_filter_threshold <= 1:
            raise ValueError("language filter threshold must lie in (0, 1]")
        by_name = {item.name: item for item in self.strata}
        for required in ("code", "mathematics"):
            if by_name[required].non_ascii_byte_count == 0:
                raise ValueError(f"C1 requires non-ASCII bytes in {required}")
        if self.language_filtered_strata != ("general",):
            raise ValueError("language filtering is permitted only on general documents")
        if type(self.byte_filtering_applied) is not bool:
            raise TypeError("byte_filtering_applied must be boolean")
        if self.byte_filtering_applied:
            raise ValueError("C1-C3 prohibit byte-class filtering")
        if not isinstance(self.round_trip_fixtures, tuple) or not self.round_trip_fixtures:
            raise ValueError("C2 requires byte round-trip fixtures")
        if any(
            not isinstance(item, ByteRoundTripFixtureReceipt)
            for item in self.round_trip_fixtures
        ):
            raise TypeError("C2 fixtures must be ByteRoundTripFixtureReceipt values")
        covered = {
            category
            for fixture in self.round_trip_fixtures
            for category in fixture.categories
        }
        missing = set(GTOK_ROUND_TRIP_CATEGORIES) - covered
        if missing:
            raise ValueError(f"C2 fixture coverage is incomplete: {sorted(missing)!r}")

    @property
    def manifest_sha256(self) -> str:
        return authority_bound_sha256("weft1_gtok_corpus_audit_v1", self)


def _validate_named_values(
    rows: tuple[tuple[str, Any], ...],
    name: str,
    *,
    require_nonempty: bool,
) -> None:
    if not isinstance(rows, tuple):
        raise TypeError(f"{name} must be a tuple")
    if require_nonempty and not rows:
        raise ValueError(f"{name} must not be empty")
    keys: list[str] = []
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            raise TypeError(f"{name} rows must be (name, value) tuples")
        key, value = row
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be nonempty strings")
        keys.append(key)
        _canonical_value(value)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} keys must be unique")
    if tuple(sorted(keys)) != tuple(keys):
        raise ValueError(f"{name} keys must use canonical sorted order")


@dataclass(frozen=True)
class FlatAdamWRecipe:
    """Caller-supplied AdamW settings with no vocabulary-size correction.

    The contract fixes equality and optimizer family, not numerical values.
    Values remain external until they are separately bound.
    """

    hyperparameters: tuple[tuple[str, Any], ...]
    schedule: tuple[tuple[str, Any], ...]
    optimizer_family: str = "AdamW"
    parameter_partition: str = "flat_all_trainables"
    vocabulary_size_correction: None = None
    muon_enabled: bool = False

    def __post_init__(self) -> None:
        if self.optimizer_family != "AdamW":
            raise ValueError("G-TOK requires AdamW")
        if self.parameter_partition != "flat_all_trainables":
            raise ValueError("G-TOK requires one flat all-trainables partition")
        if self.vocabulary_size_correction is not None:
            raise ValueError("G-TOK prohibits every per-V learning-rate correction")
        if self.muon_enabled is not False:
            raise ValueError("Muon is prohibited in the G-TOK screen")
        _validate_named_values(
            self.hyperparameters,
            "AdamW hyperparameters",
            require_nonempty=True,
        )
        _validate_named_values(self.schedule, "AdamW schedule", require_nonempty=True)

    @property
    def recipe_sha256(self) -> str:
        return authority_bound_sha256("weft1_gtok_flat_adamw_v1", self)


def a1_flat_adamw_recipe() -> FlatAdamWRecipe:
    """Return Amendment A1's exact flat, screen-only optimizer recipe."""

    return FlatAdamWRecipe(
        hyperparameters=(
            ("betas", (0.9, 0.95)),
            ("eps", 1e-8),
            ("gradient_clip_norm", 1.0),
            ("learning_rate", 3e-4),
            ("weight_decay", 0.1),
        ),
        schedule=(
            ("batch_sequence_length", 2_048),
            ("batch_sequences", 256),
            ("compute_dtype", "bfloat16"),
            ("decay", "cosine"),
            ("final_learning_rate_fraction", Fraction(1, 10)),
            ("loss_reduction_dtype", "float32"),
            ("master_weight_dtype", "float32"),
            ("warmup_fraction", Fraction(1, 100)),
        ),
    )


def assert_identical_flat_adamw(recipes: tuple[FlatAdamWRecipe, ...]) -> FlatAdamWRecipe:
    """Require one byte-identical flat AdamW recipe across every run."""

    if not isinstance(recipes, tuple) or not recipes:
        raise ValueError("at least one AdamW recipe is required")
    if any(not isinstance(recipe, FlatAdamWRecipe) for recipe in recipes):
        raise TypeError("recipes must contain FlatAdamWRecipe values")
    first = recipes[0]
    first_bytes = canonical_json_bytes(first)
    if any(canonical_json_bytes(recipe) != first_bytes for recipe in recipes[1:]):
        raise ValueError("all G-TOK arms and seeds require an identical AdamW recipe")
    return first


@dataclass(frozen=True)
class StratumNllReceipt:
    """Held-out NLL and its raw-byte denominator for one stratum."""

    stratum: str
    nll_nats: float
    raw_byte_count: int

    def __post_init__(self) -> None:
        if self.stratum not in GTOK_STRATA:
            raise ValueError(f"unknown G-TOK stratum: {self.stratum!r}")
        if isinstance(self.nll_nats, bool) or not isinstance(self.nll_nats, Real):
            raise TypeError("nll_nats must be numeric")
        value = float(self.nll_nats)
        bits_per_byte(value, self.raw_byte_count)
        object.__setattr__(self, "nll_nats", value)

    @property
    def bpb(self) -> float:
        return bits_per_byte(self.nll_nats, self.raw_byte_count)


@dataclass(frozen=True)
class BpbObservationReceipt:
    """Pooled and per-stratum BPB evidence at one exact byte milestone."""

    milestone_fraction: Fraction
    training_raw_bytes: int
    heldout_stream_sha256: str
    strata: tuple[StratumNllReceipt, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.milestone_fraction, Fraction):
            raise TypeError("milestone_fraction must be an exact Fraction")
        if self.milestone_fraction not in GTOK_BPB_MILESTONE_FRACTIONS:
            raise ValueError("observation is not at a registered BPB milestone")
        expected = GTOK_BPB_MILESTONE_BYTES[self.milestone_fraction]
        if type(self.training_raw_bytes) is not int:
            raise TypeError("training_raw_bytes must be an exact integer")
        if self.training_raw_bytes != expected:
            raise ValueError("training_raw_bytes disagrees with the exact milestone")
        _require_sha256(self.heldout_stream_sha256, "heldout_stream_sha256")
        if not isinstance(self.strata, tuple):
            raise TypeError("strata must be a tuple")
        if any(not isinstance(item, StratumNllReceipt) for item in self.strata):
            raise TypeError("strata must contain StratumNllReceipt values")
        if tuple(item.stratum for item in self.strata) != GTOK_STRATA:
            raise ValueError("BPB requires one canonically ordered receipt per stratum")
        try:
            pooled_nll = math.fsum(item.nll_nats for item in self.strata)
        except OverflowError as error:
            raise ValueError("pooled held-out NLL must remain finite") from error
        if not math.isfinite(pooled_nll):
            raise ValueError("pooled held-out NLL must remain finite")

    @property
    def pooled_nll_nats(self) -> float:
        return math.fsum(item.nll_nats for item in self.strata)

    @property
    def pooled_raw_byte_count(self) -> int:
        return sum(item.raw_byte_count for item in self.strata)

    @property
    def pooled_bpb(self) -> float:
        return bits_per_byte(self.pooled_nll_nats, self.pooled_raw_byte_count)

    @property
    def bpb_by_stratum(self) -> dict[str, float]:
        return {item.stratum: item.bpb for item in self.strata}

    @property
    def denominator_signature(self) -> tuple[tuple[str, int], ...]:
        return tuple((item.stratum, item.raw_byte_count) for item in self.strata)


@dataclass(frozen=True)
class BaseTokenizerContractReceipt:
    """Choice-independent tokenizer evidence for one vocabulary arm."""

    vocab_size: int
    artifact_sha256: str
    corpus_manifest_sha256: str
    fit_corpus_sha256: str
    tokenizer_library: str
    tokenizer_version: str
    pretokenizer_regex_sha256: str
    reserved_inventory_sha256: str
    token_inventory_sha256: str
    token_inventory_count: int
    byte_atom_count: int = 256
    reachable_unk: bool = False
    bpe_dropout: float = 0.0
    stochastic_segmentation: bool = False
    irreversible_normalization: bool = False

    def __post_init__(self) -> None:
        if type(self.vocab_size) is not int:
            raise TypeError("tokenizer vocabulary size must be an exact integer")
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("tokenizer receipt uses an unregistered vocabulary arm")
        for name in (
            "artifact_sha256",
            "corpus_manifest_sha256",
            "fit_corpus_sha256",
            "pretokenizer_regex_sha256",
            "reserved_inventory_sha256",
            "token_inventory_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.tokenizer_library, str) or not self.tokenizer_library:
            raise ValueError("tokenizer library must be recorded")
        if not isinstance(self.tokenizer_version, str) or not self.tokenizer_version:
            raise ValueError("tokenizer version must be recorded")
        if type(self.token_inventory_count) is not int:
            raise TypeError("token_inventory_count must be an exact integer")
        if self.token_inventory_count != self.vocab_size:
            raise ValueError("token inventory count must equal the registered vocabulary size")
        if type(self.byte_atom_count) is not int:
            raise TypeError("byte_atom_count must be an exact integer")
        if self.byte_atom_count != 256:
            raise ValueError("byte-level BPE requires all 256 byte atoms")
        for name in (
            "reachable_unk",
            "stochastic_segmentation",
            "irreversible_normalization",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.reachable_unk:
            raise ValueError("the tokenizer may not have a reachable UNK path")
        if self.stochastic_segmentation:
            raise ValueError("stochastic segmentation is prohibited in G-TOK")
        if self.irreversible_normalization:
            raise ValueError("irreversible normalization is prohibited")
        if type(self.bpe_dropout) not in (int, float):
            raise TypeError("bpe_dropout must be numeric")
        if float(self.bpe_dropout) != 0.0:
            raise ValueError("BPE dropout must be exactly zero")
        object.__setattr__(self, "bpe_dropout", 0.0)

    @property
    def contract_sha256(self) -> str:
        return authority_bound_sha256("weft1_gtok_base_tokenizer_v1", self)


@dataclass(frozen=True)
class GTokComputeEventReceipt:
    """One ordered scheduler event, including failed or retried attempts."""

    attempt_id: str
    event_index: int
    scope: str
    status: str
    measured_a100_hours: float
    vocab_size: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("attempt_id", "scope", "status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty exact string")
            if value != value.strip():
                raise ValueError(f"{name} may not contain surrounding whitespace")
        if self.scope not in GTOK_COMPUTE_SCOPES:
            raise ValueError(f"compute event scope must be one of {GTOK_COMPUTE_SCOPES!r}")
        if self.status not in GTOK_COMPUTE_STATUSES:
            raise ValueError(
                f"compute event status must be one of {GTOK_COMPUTE_STATUSES!r}"
            )
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("event_index must be a non-negative exact integer")
        if self.vocab_size is not None:
            if type(self.vocab_size) is not int:
                raise TypeError("compute event vocabulary size must be an exact integer")
            if self.vocab_size not in GTOK_VOCABULARY_ARMS:
                raise ValueError("compute event uses an unregistered vocabulary arm")
        if self.seed is not None and type(self.seed) is not int:
            raise TypeError("compute event seed must be an exact integer or None")
        if (self.vocab_size is None) != (self.seed is None):
            raise ValueError("compute event arm and seed must be both present or both absent")
        if self.scope == "base_screen" and self.vocab_size is None:
            raise ValueError("base-screen compute events require an arm and seed")
        object.__setattr__(
            self,
            "measured_a100_hours",
            validate_a100_hour_tripwire(self.measured_a100_hours),
        )

    @property
    def receipt_sha256(self) -> str:
        return authority_bound_sha256("weft1_gtok_compute_event_v1", self)


@dataclass(frozen=True)
class GTokComputeReceipt:
    """Complete ordered all-attempt snapshot under the cumulative tripwire."""

    source_event_log_sha256: str
    events: tuple[GTokComputeEventReceipt, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.source_event_log_sha256, "source_event_log_sha256")
        if not isinstance(self.events, tuple) or not self.events:
            raise ValueError("compute events must be a nonempty tuple")
        if any(not isinstance(event, GTokComputeEventReceipt) for event in self.events):
            raise TypeError("events must contain GTokComputeEventReceipt values")
        attempt_ids = tuple(event.attempt_id for event in self.events)
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("compute receipt contains a duplicate attempt ID")
        if tuple(event.event_index for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("compute events must preserve contiguous scheduler order")
        validate_a100_hour_tripwire(self.total_a100_hours)

    @property
    def total_a100_hours(self) -> float:
        return math.fsum(event.measured_a100_hours for event in self.events)

    @property
    def receipt_sha256(self) -> str:
        return authority_bound_sha256("weft1_gtok_compute_ledger_v1", self)


@dataclass(frozen=True)
class GTokRunReceipt:
    """One arm/seed receipt.  Validation is evidence QA, not run authorization."""

    vocab_size: int
    seed: int
    corpus_manifest_sha256: str
    training_corpus_sha256: str
    tokenizer_artifact_sha256: str
    model_topology_sha256: str
    initialization_recipe_sha256: str
    initialization_seed: int
    shared_initial_state_sha256: str
    data_order_seed: int
    data_order_sha256: str
    compute_attempt_id: str
    measured_a100_hours: float
    optimizer: FlatAdamWRecipe
    observations: tuple[BpbObservationReceipt, ...]
    training_raw_byte_budget: int = GTOK_TRAINING_BYTE_BUDGET
    checkpoint_retained: bool = False

    def __post_init__(self) -> None:
        if type(self.vocab_size) is not int:
            raise TypeError("run vocabulary size must be an exact integer")
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("run uses an unregistered G-TOK vocabulary arm")
        if type(self.seed) is not int:
            raise TypeError("seed must be an exact integer")
        _require_sha256(self.corpus_manifest_sha256, "corpus_manifest_sha256")
        _require_sha256(self.training_corpus_sha256, "training_corpus_sha256")
        _require_sha256(self.tokenizer_artifact_sha256, "tokenizer_artifact_sha256")
        _require_sha256(self.model_topology_sha256, "model_topology_sha256")
        if self.model_topology_sha256 != GTOK_PROXY_TOPOLOGY_SHA256:
            raise ValueError(
                "run topology must equal the exact ratified G-TOK 4/2/4 S0 receipt"
            )
        _require_sha256(
            self.initialization_recipe_sha256,
            "initialization_recipe_sha256",
        )
        if type(self.initialization_seed) is not int:
            raise TypeError("initialization_seed must be an exact integer")
        _require_sha256(
            self.shared_initial_state_sha256,
            "shared_initial_state_sha256",
        )
        if type(self.data_order_seed) is not int:
            raise TypeError("data_order_seed must be an exact integer")
        _require_sha256(self.data_order_sha256, "data_order_sha256")
        if not isinstance(self.compute_attempt_id, str) or not self.compute_attempt_id:
            raise ValueError("compute_attempt_id must be a nonempty exact string")
        object.__setattr__(
            self,
            "measured_a100_hours",
            validate_a100_hour_tripwire(self.measured_a100_hours),
        )
        if not isinstance(self.optimizer, FlatAdamWRecipe):
            raise TypeError("optimizer must be a FlatAdamWRecipe")
        if type(self.training_raw_byte_budget) is not int:
            raise TypeError("training_raw_byte_budget must be an exact integer")
        if self.training_raw_byte_budget != GTOK_TRAINING_BYTE_BUDGET:
            raise ValueError("every completed G-TOK run must use the byte-matched budget")
        if self.training_raw_byte_budget > GTOK_TRAINING_BYTE_CEILING:
            raise ValueError("G-TOK run exceeds its training-byte ceiling")
        if type(self.checkpoint_retained) is not bool:
            raise TypeError("checkpoint_retained must be boolean")
        if self.checkpoint_retained:
            raise ValueError("G-TOK may not retain model checkpoints")
        if not isinstance(self.observations, tuple):
            raise TypeError("observations must be a tuple")
        if any(not isinstance(item, BpbObservationReceipt) for item in self.observations):
            raise TypeError("observations must contain BpbObservationReceipt values")
        fractions = tuple(item.milestone_fraction for item in self.observations)
        if fractions != GTOK_BPB_MILESTONE_FRACTIONS:
            raise ValueError("run requires exactly the 0.25/0.5/1.0 BPB curve")

    @property
    def receipt_sha256(self) -> str:
        return authority_bound_sha256("weft1_gtok_run_v1", self)


@dataclass(frozen=True)
class ValidatedGTokBpbMatrix:
    """Joined, non-selecting 4-arm, 2-seed, 3-milestone BPB evidence only."""

    schema: str
    authority_chain: tuple[str, ...]
    vocab_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    runs: tuple[GTokRunReceipt, ...]
    corpus_manifest_sha256: str
    training_corpus_sha256: str
    heldout_stream_sha256: str
    heldout_denominator_signature: tuple[tuple[str, int], ...]
    optimizer_recipe_sha256: str
    tokenizer_contract_sha256_by_vocab: tuple[tuple[int, str], ...]
    model_topology_sha256: str
    initialization_recipe_sha256: str
    compute_receipt_sha256: str
    total_measured_a100_hours: float

    @property
    def receipt_sha256(self) -> str:
        return authority_bound_sha256(self.schema, self)


def validate_complete_gtok_bpb_receipts(
    runs: tuple[GTokRunReceipt, ...],
    *,
    corpus_manifest: CorpusAuditManifestReceipt,
    tokenizers: tuple[BaseTokenizerContractReceipt, ...],
    compute: GTokComputeReceipt,
) -> ValidatedGTokBpbMatrix:
    """Validate the joined BPB matrix without fitting or choosing a winner.

    Compression, undertrained-row, throughput, FLOP, and compute-confirmation
    decision receipts remain separately blocked and are not implied here.
    """

    if not isinstance(runs, tuple):
        raise TypeError("runs must be a tuple")
    if any(not isinstance(run, GTokRunReceipt) for run in runs):
        raise TypeError("runs must contain GTokRunReceipt values")
    expected_count = len(GTOK_VOCABULARY_ARMS) * GTOK_SEED_COUNT
    if len(runs) != expected_count:
        raise ValueError(f"G-TOK requires exactly {expected_count} arm/seed runs")
    keys = tuple((run.vocab_size, run.seed) for run in runs)
    if len(set(keys)) != len(keys):
        raise ValueError("G-TOK contains a duplicate arm/seed run")
    seeds = tuple(sorted({run.seed for run in runs}))
    if len(seeds) != GTOK_SEED_COUNT:
        raise ValueError(f"G-TOK requires exactly {GTOK_SEED_COUNT} seeds")
    for vocab_size in GTOK_VOCABULARY_ARMS:
        observed = tuple(sorted(run.seed for run in runs if run.vocab_size == vocab_size))
        if observed != seeds:
            raise ValueError("every vocabulary arm must use the same two seeds")

    if not isinstance(corpus_manifest, CorpusAuditManifestReceipt):
        raise TypeError("corpus_manifest must be a CorpusAuditManifestReceipt")
    if not isinstance(tokenizers, tuple) or any(
        not isinstance(item, BaseTokenizerContractReceipt) for item in tokenizers
    ):
        raise TypeError("tokenizers must contain BaseTokenizerContractReceipt values")
    tokenizer_by_vocab = {item.vocab_size: item for item in tokenizers}
    if len(tokenizers) != len(GTOK_VOCABULARY_ARMS) or tuple(
        sorted(tokenizer_by_vocab)
    ) != GTOK_VOCABULARY_ARMS:
        raise ValueError("one tokenizer contract is required for every vocabulary arm")
    if not isinstance(compute, GTokComputeReceipt):
        raise TypeError("compute must be a GTokComputeReceipt")

    tokenizer_artifact_hashes = {item.artifact_sha256 for item in tokenizers}
    tokenizer_inventory_hashes = {
        item.token_inventory_sha256 for item in tokenizers
    }
    if len(tokenizer_artifact_hashes) != len(GTOK_VOCABULARY_ARMS):
        raise ValueError("every vocabulary arm requires a distinct tokenizer artifact")
    if len(tokenizer_inventory_hashes) != len(GTOK_VOCABULARY_ARMS):
        raise ValueError("every vocabulary arm requires a distinct token inventory")
    non_v_fields = (
        "corpus_manifest_sha256",
        "fit_corpus_sha256",
        "tokenizer_library",
        "tokenizer_version",
        "pretokenizer_regex_sha256",
        "reserved_inventory_sha256",
        "byte_atom_count",
        "reachable_unk",
        "bpe_dropout",
        "stochastic_segmentation",
        "irreversible_normalization",
    )
    for field_name in non_v_fields:
        if len({getattr(item, field_name) for item in tokenizers}) != 1:
            raise ValueError("tokenizer arms may differ only in vocabulary size and inventory")

    manifest_hashes = {run.corpus_manifest_sha256 for run in runs}
    training_hashes = {run.training_corpus_sha256 for run in runs}
    if len(manifest_hashes) != 1 or len(training_hashes) != 1:
        raise ValueError("every arm and seed must use the same frozen corpus")
    if manifest_hashes != {corpus_manifest.manifest_sha256}:
        raise ValueError("run corpus hashes are not joined to the audit manifest")
    if training_hashes != {corpus_manifest.training_corpus_sha256}:
        raise ValueError("run training hashes are not joined to the audit manifest")
    for seed in seeds:
        seed_runs = tuple(run for run in runs if run.seed == seed)
        order_hashes = {run.data_order_sha256 for run in seed_runs}
        if len(order_hashes) != 1:
            raise ValueError("arms must share an identical data order within each seed")
        if len({run.data_order_seed for run in seed_runs}) != 1:
            raise ValueError("arms must share the same data-order seed within each seed")
        if len({run.initialization_seed for run in seed_runs}) != 1:
            raise ValueError("arms must share the same initialization seed within each seed")
        if len({run.shared_initial_state_sha256 for run in seed_runs}) != 1:
            raise ValueError("arms must share the same non-vocabulary initialization state")
    if len({run.data_order_sha256 for run in runs}) != GTOK_SEED_COUNT:
        raise ValueError("the two seeds must use distinct data orders")
    if len({run.data_order_seed for run in runs}) != GTOK_SEED_COUNT:
        raise ValueError("the two seeds must use distinct data-order seeds")
    if len({run.initialization_seed for run in runs}) != GTOK_SEED_COUNT:
        raise ValueError("the two seeds must use distinct initialization seeds")
    if len({run.shared_initial_state_sha256 for run in runs}) != GTOK_SEED_COUNT:
        raise ValueError("the two seeds must use distinct non-vocabulary initial states")
    initialization_recipe_hashes = {
        run.initialization_recipe_sha256 for run in runs
    }
    if len(initialization_recipe_hashes) != 1:
        raise ValueError("every G-TOK run must use one initialization recipe")

    topology_hashes = {run.model_topology_sha256 for run in runs}
    if len(topology_hashes) != 1:
        raise ValueError("every G-TOK run must use the same model topology")
    for run in runs:
        tokenizer = tokenizer_by_vocab[run.vocab_size]
        if run.tokenizer_artifact_sha256 != tokenizer.artifact_sha256:
            raise ValueError("run tokenizer hash is not joined to its arm contract")
        if tokenizer.corpus_manifest_sha256 != corpus_manifest.manifest_sha256:
            raise ValueError("tokenizer contract is not joined to the audit corpus")
        if tokenizer.fit_corpus_sha256 != corpus_manifest.training_corpus_sha256:
            raise ValueError("tokenizer fit corpus must exclude the held-out bytes")

    optimizer = assert_identical_flat_adamw(tuple(run.optimizer for run in runs))
    observations = tuple(
        observation
        for run in runs
        for observation in run.observations
    )
    heldout_hashes = {item.heldout_stream_sha256 for item in observations}
    denominator_signatures = {item.denominator_signature for item in observations}
    if len(heldout_hashes) != 1:
        raise ValueError("every BPB point must use the identical held-out byte stream")
    if len(denominator_signatures) != 1:
        raise ValueError("every BPB point must use identical raw-byte denominators")
    if heldout_hashes != {corpus_manifest.heldout_corpus_sha256}:
        raise ValueError("BPB held-out stream is not joined to the audit manifest")
    heldout_denominator_signature = next(iter(denominator_signatures))
    if heldout_denominator_signature != corpus_manifest.heldout_stratum_raw_byte_counts:
        raise ValueError("BPB denominators do not match the manifested held-out strata")

    compute_by_attempt = {event.attempt_id: event for event in compute.events}
    completed_base_events = tuple(
        event
        for event in compute.events
        if event.scope == "base_screen" and event.status == "completed"
    )
    completed_base_keys = tuple(
        (event.vocab_size, event.seed) for event in completed_base_events
    )
    if (
        len(completed_base_keys) != expected_count
        or len(set(completed_base_keys)) != expected_count
        or set(completed_base_keys) != set(keys)
    ):
        raise ValueError("compute ledger requires exactly one completed attempt per arm/seed")
    selected_attempt_ids = tuple(run.compute_attempt_id for run in runs)
    if len(set(selected_attempt_ids)) != len(selected_attempt_ids):
        raise ValueError("every BPB run requires a distinct compute attempt")
    for run in runs:
        event = compute_by_attempt.get(run.compute_attempt_id)
        if event is None:
            raise ValueError("compute receipt must contain every selected BPB attempt")
        if event.scope != "base_screen" or event.status != "completed":
            raise ValueError("selected BPB attempts must be completed base-screen events")
        if (event.vocab_size, event.seed) != (run.vocab_size, run.seed):
            raise ValueError("selected compute attempt has the wrong arm/seed key")
        if event.measured_a100_hours != run.measured_a100_hours:
            raise ValueError("run compute differs from the cumulative compute receipt")

    ordered_runs = tuple(sorted(runs, key=lambda run: (run.vocab_size, run.seed)))
    return ValidatedGTokBpbMatrix(
        schema="weft1_gtok_bpb_matrix_v1",
        authority_chain=GTOK_AUTHORITY_CHAIN,
        vocab_sizes=GTOK_VOCABULARY_ARMS,
        seeds=seeds,
        runs=ordered_runs,
        corpus_manifest_sha256=corpus_manifest.manifest_sha256,
        training_corpus_sha256=corpus_manifest.training_corpus_sha256,
        heldout_stream_sha256=corpus_manifest.heldout_corpus_sha256,
        heldout_denominator_signature=heldout_denominator_signature,
        optimizer_recipe_sha256=optimizer.recipe_sha256,
        tokenizer_contract_sha256_by_vocab=tuple(
            (vocab_size, tokenizer_by_vocab[vocab_size].contract_sha256)
            for vocab_size in GTOK_VOCABULARY_ARMS
        ),
        model_topology_sha256=next(iter(topology_hashes)),
        initialization_recipe_sha256=next(iter(initialization_recipe_hashes)),
        compute_receipt_sha256=compute.receipt_sha256,
        total_measured_a100_hours=compute.total_a100_hours,
    )


def require_gtok_execution_authority(action: str) -> NoReturn:
    """Fail closed until every literal binding and authority conflict is resolved."""

    if not isinstance(action, str) or not action.strip():
        raise ValueError("blocked action must be named")
    unresolved = "; ".join(UNRESOLVED_GTOK_DECISIONS)
    raise GTokExecutionBlocked(
        f"{action} is inside the authorized run-axis envelope but blocked by "
        f"unresolved G-TOK bindings or authority conflicts: {unresolved}"
    )


@dataclass(frozen=True)
class TokenizerArtifactSnapshot:
    """Library-independent evidence needed to audit append-only continuation."""

    artifact_sha256: str
    corpus_manifest_sha256: str
    tokenizer_library: str
    tokenizer_version: str
    pretokenizer_regex: str
    pretokenizer_regex_sha256: str
    normalization: str
    reserved_token_ids: tuple[int, ...]
    id_to_token_bytes_sha256: tuple[str, ...]
    id_to_token_metadata_sha256: tuple[str, ...]
    special_role_to_id: tuple[tuple[str, int], ...]
    merge_entries: tuple[str, ...]
    merge_table_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        _require_sha256(self.corpus_manifest_sha256, "corpus_manifest_sha256")
        if not isinstance(self.tokenizer_library, str) or not self.tokenizer_library:
            raise ValueError("tokenizer_library must be an exact nonempty string")
        if not isinstance(self.tokenizer_version, str) or not self.tokenizer_version:
            raise ValueError("tokenizer_version must be an exact nonempty string")
        if not isinstance(self.pretokenizer_regex, str) or not self.pretokenizer_regex:
            raise ValueError("pretokenizer_regex must be an exact nonempty string")
        expected_regex_hash = sha256_bytes(self.pretokenizer_regex.encode("utf-8"))
        _require_sha256(self.pretokenizer_regex_sha256, "pretokenizer_regex_sha256")
        if self.pretokenizer_regex_sha256 != expected_regex_hash:
            raise ValueError("pretokenizer regex SHA disagrees with its exact text")
        if not isinstance(self.normalization, str) or not self.normalization:
            raise ValueError("normalization contract must be an exact nonempty string")
        if (
            not isinstance(self.id_to_token_bytes_sha256, tuple)
            or not self.id_to_token_bytes_sha256
        ):
            raise ValueError("token ID meaning manifest must be a nonempty tuple")
        for token_hash in self.id_to_token_bytes_sha256:
            _require_sha256(token_hash, "token byte meaning hash")
        if len(set(self.id_to_token_bytes_sha256)) != len(
            self.id_to_token_bytes_sha256
        ):
            raise ValueError("token IDs may not alias an existing byte meaning")
        if not isinstance(self.id_to_token_metadata_sha256, tuple):
            raise TypeError("token metadata manifest must be a tuple")
        if len(self.id_to_token_metadata_sha256) != len(
            self.id_to_token_bytes_sha256
        ):
            raise ValueError("every token ID requires an exact metadata/flag hash")
        for metadata_hash in self.id_to_token_metadata_sha256:
            _require_sha256(metadata_hash, "token metadata meaning hash")
        if not isinstance(self.reserved_token_ids, tuple):
            raise TypeError("reserved_token_ids must be a tuple")
        if any(type(token_id) is not int for token_id in self.reserved_token_ids):
            raise TypeError("reserved token IDs must be exact integers")
        if tuple(sorted(set(self.reserved_token_ids))) != self.reserved_token_ids:
            raise ValueError("reserved token IDs must be unique and sorted")
        if any(
            token_id < 0 or token_id >= len(self.id_to_token_bytes_sha256)
            for token_id in self.reserved_token_ids
        ):
            raise ValueError("reserved token ID lies outside the vocabulary")
        if not isinstance(self.special_role_to_id, tuple) or not self.special_role_to_id:
            raise ValueError("special token role map must be a nonempty tuple")
        roles: list[str] = []
        role_ids: list[int] = []
        for row in self.special_role_to_id:
            if not isinstance(row, tuple) or len(row) != 2:
                raise TypeError("special role rows must be (role, token_id) tuples")
            role, token_id = row
            if not isinstance(role, str) or not role:
                raise ValueError("special token roles must be nonempty strings")
            if type(token_id) is not int:
                raise TypeError("special role token IDs must be exact integers")
            roles.append(role)
            role_ids.append(token_id)
        if tuple(sorted(roles)) != tuple(roles) or len(set(roles)) != len(roles):
            raise ValueError("special token roles must be unique and canonically sorted")
        if not set(role_ids).issubset(self.reserved_token_ids):
            raise ValueError("special token roles must point to reserved token IDs")
        if not isinstance(self.merge_entries, tuple) or not self.merge_entries:
            raise ValueError("merge_entries must be a nonempty tuple")
        if any(not isinstance(entry, str) or not entry for entry in self.merge_entries):
            raise ValueError("merge entries must be exact nonempty strings")
        if len(set(self.merge_entries)) != len(self.merge_entries):
            raise ValueError("merge entries must be unique")
        _require_sha256(self.merge_table_sha256, "merge_table_sha256")
        if self.merge_table_sha256 != canonical_sha256(self.merge_entries):
            raise ValueError("merge table SHA disagrees with its canonical entries")

    @property
    def token_id_manifest_sha256(self) -> str:
        return authority_bound_sha256(
            "weft1_token_id_meanings_v1",
            {
                "bytes": self.id_to_token_bytes_sha256,
                "metadata": self.id_to_token_metadata_sha256,
                "special_roles": self.special_role_to_id,
            },
        )


@dataclass(frozen=True)
class AppendOnlyExtensionBasisReceipt:
    """Hashes proving which frozen artifact is the continuation basis."""

    parent_artifact_sha256: str
    parent_merge_table_sha256: str
    parent_corpus_manifest_sha256: str
    parent_pretokenizer_regex_sha256: str
    parent_token_id_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "parent_artifact_sha256",
            "parent_merge_table_sha256",
            "parent_corpus_manifest_sha256",
            "parent_pretokenizer_regex_sha256",
            "parent_token_id_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)


@dataclass(frozen=True)
class AppendOnlyCorpusExtensionReceipt:
    """Manifest relationship for continuation on a parent-plus-added corpus."""

    parent_corpus_manifest_sha256: str
    added_corpus_manifest_sha256: str
    combined_corpus_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "parent_corpus_manifest_sha256",
            "added_corpus_manifest_sha256",
            "combined_corpus_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.added_corpus_manifest_sha256 == self.parent_corpus_manifest_sha256:
            raise ValueError("extension corpus must add a distinct manifested source")
        if self.combined_corpus_manifest_sha256 in {
            self.parent_corpus_manifest_sha256,
            self.added_corpus_manifest_sha256,
        }:
            raise ValueError("combined corpus must be a strict parent-plus-addition manifest")

    @property
    def receipt_sha256(self) -> str:
        return authority_bound_sha256("weft1_append_only_corpus_extension_v1", self)


@dataclass(frozen=True)
class AppendOnlyExtensionValidation:
    preserved_token_ids: int
    appended_token_ids: int
    preserved_merges: int
    appended_merges: int

    @property
    def segmentation_invariant(self) -> bool:
        return False


def validate_append_only_tokenizer_extension(
    parent: TokenizerArtifactSnapshot,
    child: TokenizerArtifactSnapshot,
    basis: AppendOnlyExtensionBasisReceipt,
    corpus_extension: AppendOnlyCorpusExtensionReceipt,
) -> AppendOnlyExtensionValidation:
    """Validate ID/merge append-only continuation, not segmentation identity.

    New merges may alter the segmentation of old byte strings.  The guarantee
    here is narrower: old IDs keep their byte meaning, the old merge list stays
    an exact prefix, and every newly defined ID is appended.
    """

    if not isinstance(parent, TokenizerArtifactSnapshot):
        raise TypeError("parent must be a TokenizerArtifactSnapshot")
    if not isinstance(child, TokenizerArtifactSnapshot):
        raise TypeError("child must be a TokenizerArtifactSnapshot")
    if not isinstance(basis, AppendOnlyExtensionBasisReceipt):
        raise TypeError("basis must be an AppendOnlyExtensionBasisReceipt")
    if not isinstance(corpus_extension, AppendOnlyCorpusExtensionReceipt):
        raise TypeError("corpus_extension must be an AppendOnlyCorpusExtensionReceipt")
    expected_basis = AppendOnlyExtensionBasisReceipt(
        parent_artifact_sha256=parent.artifact_sha256,
        parent_merge_table_sha256=parent.merge_table_sha256,
        parent_corpus_manifest_sha256=parent.corpus_manifest_sha256,
        parent_pretokenizer_regex_sha256=parent.pretokenizer_regex_sha256,
        parent_token_id_manifest_sha256=parent.token_id_manifest_sha256,
    )
    if basis != expected_basis:
        raise ValueError("extension basis hashes do not identify the parent artifact")
    if child.artifact_sha256 == parent.artifact_sha256:
        raise ValueError("append-only extension requires a distinct child artifact")
    if corpus_extension.parent_corpus_manifest_sha256 != parent.corpus_manifest_sha256:
        raise ValueError("extension corpus receipt does not identify the parent corpus")
    if corpus_extension.combined_corpus_manifest_sha256 != child.corpus_manifest_sha256:
        raise ValueError("child artifact is not joined to the combined extension corpus")
    if child.tokenizer_library != parent.tokenizer_library:
        raise ValueError("append-only extension may not change tokenizer library")
    if child.tokenizer_version != parent.tokenizer_version:
        raise ValueError("append-only extension may not change tokenizer version")
    if child.pretokenizer_regex != parent.pretokenizer_regex:
        raise ValueError("append-only extension may not change the pre-tokenizer regex")
    if child.pretokenizer_regex_sha256 != parent.pretokenizer_regex_sha256:
        raise ValueError("append-only extension changed the pre-tokenizer regex SHA")
    if child.normalization != parent.normalization:
        raise ValueError("append-only extension may not change normalization")
    if child.reserved_token_ids != parent.reserved_token_ids:
        raise ValueError("append-only extension may not change reserved token IDs")
    if child.special_role_to_id != parent.special_role_to_id:
        raise ValueError("append-only extension may not change special token roles")
    parent_token_count = len(parent.id_to_token_bytes_sha256)
    if len(child.id_to_token_bytes_sha256) <= parent_token_count:
        raise ValueError("append-only extension must append at least one token ID")
    if child.id_to_token_bytes_sha256[:parent_token_count] != parent.id_to_token_bytes_sha256:
        raise ValueError("an existing token ID was renumbered or redefined")
    if (
        child.id_to_token_metadata_sha256[:parent_token_count]
        != parent.id_to_token_metadata_sha256
    ):
        raise ValueError("an existing token ID changed kind or AddedToken flags")
    parent_merge_count = len(parent.merge_entries)
    if len(child.merge_entries) <= parent_merge_count:
        raise ValueError("BPE continuation must append at least one merge")
    if child.merge_entries[:parent_merge_count] != parent.merge_entries:
        raise ValueError("the parent merge list is not an exact child prefix")
    appended_token_count = len(child.id_to_token_bytes_sha256) - parent_token_count
    appended_merge_count = len(child.merge_entries) - parent_merge_count
    if appended_token_count != appended_merge_count:
        raise ValueError("BPE continuation requires one appended token ID per merge")
    return AppendOnlyExtensionValidation(
        preserved_token_ids=parent_token_count,
        appended_token_ids=appended_token_count,
        preserved_merges=parent_merge_count,
        appended_merges=appended_merge_count,
    )


__all__ = [
    "AppendOnlyCorpusExtensionReceipt",
    "AppendOnlyExtensionBasisReceipt",
    "AppendOnlyExtensionValidation",
    "BaseTokenizerContractReceipt",
    "BpbObservationReceipt",
    "ByteRoundTripFixtureReceipt",
    "CorpusAuditManifestReceipt",
    "CorpusStratumByteStats",
    "FlatAdamWRecipe",
    "GTOK_AMENDMENT_A1_SHA256",
    "GTOK_A100_HOUR_TRIPWIRE",
    "GTOK_BPB_MILESTONE_BYTES",
    "GTOK_BPB_MILESTONE_FRACTIONS",
    "GTOK_AUTHORITY_CHAIN",
    "GTOK_CURRICULUM_AMENDMENT_SHA256",
    "GTOK_CURRICULUM_DATA_SHA256",
    "GTOK_CURRICULUM_DECISIONS_SHA256",
    "GTOK_COMPUTE_SCOPES",
    "GTOK_COMPUTE_STATUSES",
    "GTOK_ENGLISH_SCOPE_SHA256",
    "GTOK_EXECUTION_AUTHORITY_CHAIN",
    "GTOK_EXECUTION_AUTHORITY_CHAIN_V2",
    "GTOK_EXECUTION_HANDOFF_SHA256",
    "GTOK_HANDOFF_SHA256",
    "GTOK_HELDOUT_BYTE_TARGET",
    "GTOK_PIPELINE_RNG_NAMES",
    "GTOK_PRETOKENIZER_REGEX",
    "GTOK_PROJECTED_A100_HOURS",
    "GTOK_PROXY_TOPOLOGY",
    "GTOK_PROXY_TOPOLOGY_SHA256",
    "GTOK_QWEN_ADJUDICATION_SHA256",
    "GTOK_ROUND_TRIP_CATEGORIES",
    "GTOK_RUN_RNG_NAME_TEMPLATES",
    "GTOK_RULINGS_SHA256",
    "GTOK_SEED_COUNT",
    "GTOK_SCREEN_HELDOUT_STRATUM_TARGETS",
    "GTOK_SCREEN_TRAIN_STRATUM_TARGETS",
    "GTOK_STRATA",
    "GTOK_STRATUM_SHARES",
    "GTOK_STRATUM_TOLERANCE",
    "GTOK_TOKENIZER_FAMILY",
    "GTOK_TOKENIZER_LIBRARY",
    "GTOK_TOKENIZER_MIN_FREQUENCY",
    "GTOK_TRAINING_BYTE_BUDGET",
    "GTOK_TRAINING_BYTE_CEILING",
    "GTOK_VOCABULARY_ARMS",
    "GTokComputeEventReceipt",
    "GTokComputeReceipt",
    "GTokExecutionBlocked",
    "GTokProxyTopologyReceipt",
    "GTokRunReceipt",
    "StratumNllReceipt",
    "TokenizerArtifactSnapshot",
    "UNRESOLVED_GTOK_DECISIONS",
    "ValidatedGTokBpbMatrix",
    "assert_identical_flat_adamw",
    "a1_flat_adamw_recipe",
    "authority_bound_sha256",
    "bits_per_byte",
    "canonical_json_bytes",
    "canonical_sha256",
    "execution_authority_bound_sha256",
    "execution_authority_v2_bound_sha256",
    "require_gtok_execution_authority",
    "sha256_bytes",
    "validate_a100_hour_tripwire",
    "validate_append_only_tokenizer_extension",
    "validate_complete_gtok_bpb_receipts",
]
