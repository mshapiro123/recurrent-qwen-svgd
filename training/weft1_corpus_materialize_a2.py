"""Offline, disk-bounded WEFT-1 corpus P-A materialization.

This module turns already enumerated, already cached, typed source streams into
the deterministic full-corpus selection and the G-TOK train/held-out shards.
It deliberately has no downloader and no gate minter.  A production run must
bring an authoritative upstream-enumeration receipt and a byte-verified local
cache receipt; bounded fixtures are explicitly non-authoritative.

The materializer keeps document text in a SQLite spool, streams selection and
shard writes, and emits content-only manifests without timestamps, host names,
PIDs, or absolute paths.  Consequently two fresh work roots can be compared by
rehashing the emitted tree, which is the artifact shape required by D1.  The
actual D1 gate remains the responsibility of the independent parent verifier.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import socket
import tempfile
from typing import Protocol

import zstandard

from training.weft1_corpus_a2 import (
    A2_DEDUP_SEED,
    A2_LANGUAGE_ID_BINDING,
    A2_MATCH_NORMALIZATION_BINDING,
    A2_MINHASH_BINDING,
    A2_ZSTD_CODEC_BINDING,
    FIRST_FIT_TOLERANCE,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
    MINHASH_RECALL_JACCARD_LEVELS,
    LanguageIdDecisionV3,
    MinHashRecallAuditV3,
    MinHashSyntheticRecallCellV3,
    JsonlZstdShardIdentityV3,
    StableDocumentV3,
    NEAR_DUPLICATE_THRESHOLD,
    byte_shingles_v3,
    exact_jaccard_v3,
    execution_authority_v3_bound_sha256,
    lsh_band_keys_v3,
    minhash_signature_v3,
    normalize_match_text,
    normalized_match_bytes,
    pipeline_seed,
)
from training.weft1_corpus_enumeration_a2 import (
    AUTHORITATIVE_MODE,
    UpstreamEnumerationReceiptV3,
)
from training.weft1_corpus_pa import (
    DEFAULT_SHARD_TARGET_BYTES,
    sha256_file,
    write_jsonl_zstd_shards_v3,
)
from training.weft1_corpus_sources_a2 import (
    QUALITY_GATED_SOURCE_FAMILIES,
    SCORED_SOURCE_FAMILIES,
    VerifiedLocalCacheManifestV3,
)
from training.weft1_corpus_source_io_a2 import (
    DROP_EMPTY_TEXT,
    DROP_INVALID_UTF8,
    DROP_QUALITY_LT3,
    PRODUCTION_PARSER_BINDINGS_V3,
    RETAIN,
    SourceCacheDownloadReceiptV3,
    iter_source_asset_events_v3,
    plan_source_cache_assets_v3,
)
from training.weft1_corpus_replay_a2 import DEDUP_LEDGER_IDENTITY_DOMAIN_V3
from training.weft1_corpus_replay_a2 import (
    CHILD_RECEIPT_FILENAME,
    CHILD_RECEIPT_SCHEMA_V3,
    NETWORK_PROBE_RESULT,
)
from training.weft1_strict_io import (
    assert_no_symlink_ancestors,
    load_canonical_json_object,
)
from training.weft1_gtok_a1_contract import (
    SOURCE_FAMILIES,
    SOURCE_FAMILY_TARGET_BYTES,
)
from training.weft1_gtok_contract import (
    GTOK_HELDOUT_BYTE_TARGET,
    GTOK_SCREEN_HELDOUT_STRATUM_TARGETS,
    GTOK_SCREEN_TRAIN_STRATUM_TARGETS,
    GTOK_STRATA,
    GTOK_TRAINING_BYTE_BUDGET,
    GTOK_VOCABULARY_ARMS,
    canonical_json_bytes,
)
from models.ablation_lm.rng import derive_module_seed


PRODUCTION_MODE = "PRODUCTION"
FIXTURE_MODE = "NONAUTHORITATIVE_FIXTURE"
MATERIALIZATION_MODES = (PRODUCTION_MODE, FIXTURE_MODE)

FULL_POOL_ORDER = (
    "wikipedia_wikibooks",
    "dolma_web",
    "fineweb_edu",
    "stackedu",
    "finemath_3plus",
    "science_technical_combined",
)
PRODUCTION_FULL_POOL_TARGETS = (
    ("wikipedia_wikibooks", SOURCE_FAMILY_TARGET_BYTES["wikipedia_wikibooks"]),
    ("dolma_web", SOURCE_FAMILY_TARGET_BYTES["dolma_web"]),
    ("fineweb_edu", SOURCE_FAMILY_TARGET_BYTES["fineweb_edu"]),
    ("stackedu", SOURCE_FAMILY_TARGET_BYTES["stackedu"]),
    ("finemath_3plus", SOURCE_FAMILY_TARGET_BYTES["finemath_3plus"]),
    ("science_technical_combined", SOURCE_FAMILY_TARGET_BYTES["arxiv"]),
)
SOURCE_TO_STRATUM = {
    "dolma_web": "general",
    "wikipedia_wikibooks": "general",
    "stackedu": "code",
    "finemath_3plus": "mathematics",
    "arxiv": "science_technical",
    "olmocr": "science_technical",
    "fineweb_edu": "general",
}
SCREEN_ORDER_DOMAIN = b"WEFT-1/corpus-screen-order/v1"
MATERIALIZER_SCHEMA = "weft1_corpus_pa_materialization_v3"
MATERIALIZER_ALGORITHM_VERSION = 2
A2_SCREEN_ORDER_SEED = pipeline_seed("corpus.shuffle")
REAL_RECALL_SAMPLE_PER_SOURCE = 64
SYNTHETIC_RECALL_PAIRS_PER_CELL = 64
RECALL_SAMPLE_DOMAIN = b"WEFT-1/minhash-real-sample/v1"
SYNTHETIC_RECALL_DOMAIN = b"WEFT-1/minhash-synthetic/v1"
_PRODUCTION_WORKER_RECEIPT_SENTINEL = object()
GTOK_TRAINING_SEEDS = tuple(
    derive_module_seed(int(GTOK_EXECUTION_AUTHORITY_CHAIN_V3[-1][:16], 16), "gtok.seed", replica)
    for replica in range(2)
)

_SHA256_CHARS = frozenset("0123456789abcdef")


class CorpusMaterializationError(RuntimeError):
    """Fail-closed P-A orchestration error."""


def _require_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _rows_mapping(
    rows: Sequence[tuple[str, int]], expected_keys: Sequence[str], name: str
) -> dict[str, int]:
    if not isinstance(rows, tuple):
        raise TypeError(f"{name} must be a tuple")
    if tuple(key for key, _ in rows) != tuple(expected_keys):
        raise ValueError(f"{name} must use the canonical key order")
    if any(type(value) is not int or value < 1 for _, value in rows):
        raise ValueError(f"{name} values must be positive exact integers")
    return dict(rows)


@dataclass(frozen=True)
class MaterializationPlanV3:
    """Exact full-pool and screen byte targets for one P-A execution."""

    mode: str
    full_pool_target_bytes: tuple[tuple[str, int], ...]
    training_stratum_target_bytes: tuple[tuple[str, int], ...]
    heldout_stratum_target_bytes: tuple[tuple[str, int], ...]
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES

    def __post_init__(self) -> None:
        if self.mode not in MATERIALIZATION_MODES:
            raise ValueError("materialization mode must be explicit")
        full = _rows_mapping(
            self.full_pool_target_bytes, FULL_POOL_ORDER, "full-pool targets"
        )
        training = _rows_mapping(
            self.training_stratum_target_bytes,
            GTOK_STRATA,
            "training stratum targets",
        )
        heldout = _rows_mapping(
            self.heldout_stratum_target_bytes,
            GTOK_STRATA,
            "held-out stratum targets",
        )
        if type(self.shard_target_bytes) is not int or self.shard_target_bytes < 1:
            raise ValueError("shard_target_bytes must be a positive exact integer")
        full_by_stratum = {
            "general": full["wikipedia_wikibooks"]
            + full["dolma_web"]
            + full["fineweb_edu"],
            "code": full["stackedu"],
            "mathematics": full["finemath_3plus"],
            "science_technical": full["science_technical_combined"],
        }
        for stratum in GTOK_STRATA:
            if training[stratum] + heldout[stratum] > full_by_stratum[stratum]:
                raise ValueError("screen targets exceed their full-corpus stratum")
        if self.mode == PRODUCTION_MODE:
            if self.full_pool_target_bytes != PRODUCTION_FULL_POOL_TARGETS:
                raise ValueError("production full-pool targets differ from A2")
            if self.training_stratum_target_bytes != (
                GTOK_SCREEN_TRAIN_STRATUM_TARGETS
            ):
                raise ValueError("production training targets differ from A2")
            if self.heldout_stratum_target_bytes != (
                GTOK_SCREEN_HELDOUT_STRATUM_TARGETS
            ):
                raise ValueError("production held-out targets differ from A2")
            if sum(training.values()) != GTOK_TRAINING_BYTE_BUDGET:
                raise ValueError("production training target is not four billion bytes")
            if sum(heldout.values()) != GTOK_HELDOUT_BYTE_TARGET:
                raise ValueError("production held-out target is not eighty million bytes")

    @classmethod
    def production(cls) -> "MaterializationPlanV3":
        return cls(
            mode=PRODUCTION_MODE,
            full_pool_target_bytes=PRODUCTION_FULL_POOL_TARGETS,
            training_stratum_target_bytes=GTOK_SCREEN_TRAIN_STRATUM_TARGETS,
            heldout_stratum_target_bytes=GTOK_SCREEN_HELDOUT_STRATUM_TARGETS,
        )

    @property
    def identity_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_materialization_plan_v3", self
        )


@dataclass(frozen=True)
class MaterializerSourceRecordV3:
    """One typed parser output in the source family's canonical stream order."""

    source_family: str
    stratum: str
    source_order_ordinal: int
    stable_source_record_id: str
    source_asset_identity_sha256: str
    text: str | bytes
    declared_retained_byte_count: int
    int_score: int | None = None

    def __post_init__(self) -> None:
        if self.source_family not in SOURCE_FAMILIES:
            raise ValueError("source record uses an unknown family")
        if self.stratum != SOURCE_TO_STRATUM[self.source_family]:
            raise ValueError("source record appears in the wrong stratum")
        if type(self.source_order_ordinal) is not int or self.source_order_ordinal < 0:
            raise ValueError("source_order_ordinal must be non-negative")
        _require_sha256(self.stable_source_record_id, "stable source record ID")
        _require_sha256(self.source_asset_identity_sha256, "source asset identity")
        if not isinstance(self.text, (str, bytes)):
            raise TypeError("source text must be exact str or bytes")
        if (
            type(self.declared_retained_byte_count) is not int
            or self.declared_retained_byte_count < 1
        ):
            raise ValueError("declared retained byte count must be positive")
        if self.source_family in QUALITY_GATED_SOURCE_FAMILIES:
            if type(self.int_score) is not int:
                raise ValueError("quality-gated sources require an integer score")
        elif self.int_score is not None:
            raise ValueError("ungated sources may not carry an integer score")


@dataclass(frozen=True)
class InjectedSourceStreamV3:
    source_family: str
    parser_binding_sha256: str
    parse_event_ledger_sha256: str
    records: Iterable[MaterializerSourceRecordV3]
    invalid_utf8_drop_count: int = 0
    empty_text_drop_count: int = 0
    quality_drop_count: int = 0

    def __post_init__(self) -> None:
        if self.source_family not in SOURCE_FAMILIES:
            raise ValueError("source stream uses an unknown family")
        _require_sha256(self.parser_binding_sha256, "parser binding SHA-256")
        _require_sha256(
            self.parse_event_ledger_sha256, "parse-event ledger SHA-256"
        )
        if not isinstance(self.records, Iterable):
            raise TypeError("source stream records must be iterable")
        for name in (
            "invalid_utf8_drop_count",
            "empty_text_drop_count",
            "quality_drop_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")


def injected_source_stream_from_parsed_records_v3(
    *,
    source_family: str,
    parser_binding_sha256: str,
    parse_event_ledger_sha256: str,
    records: Iterable[object],
    invalid_utf8_drop_count: int = 0,
    empty_text_drop_count: int = 0,
    quality_drop_count: int = 0,
) -> InjectedSourceStreamV3:
    """Adapt source-I/O parser outputs after any required canonical disk sort.

    FineMath and FineWeb callers must provide their globally score-ranked
    streams here; this adapter intentionally does not hide an in-memory sort.
    The materializer rechecks that ordering while consuming the lazy iterator.
    """

    if source_family not in SOURCE_FAMILIES:
        raise ValueError("parsed source adapter uses an unknown family")
    _require_sha256(parser_binding_sha256, "parser binding SHA-256")
    _require_sha256(parse_event_ledger_sha256, "parse-event ledger SHA-256")

    def adapted() -> Iterator[MaterializerSourceRecordV3]:
        from training.weft1_corpus_source_io_a2 import ParsedSourceRecordV3

        for global_ordinal, parsed in enumerate(records):
            if not isinstance(parsed, ParsedSourceRecordV3):
                raise TypeError("parsed source adapter received an untyped record")
            canonical = parsed.canonical_record
            raw = parsed.raw_document
            if canonical.source_family != source_family or raw.source != source_family:
                raise CorpusMaterializationError(
                    "parsed source adapter crossed family boundaries"
                )
            if parsed.parser_binding_sha256 != parser_binding_sha256:
                raise CorpusMaterializationError(
                    "parsed record uses a different parser binding"
                )
            yield MaterializerSourceRecordV3(
                source_family=source_family,
                stratum=raw.stratum,
                source_order_ordinal=global_ordinal,
                stable_source_record_id=raw.stable_source_record_id,
                source_asset_identity_sha256=(
                    canonical.asset.asset_identity_sha256
                ),
                text=raw.text,
                declared_retained_byte_count=canonical.retained_byte_count,
                int_score=canonical.int_score,
            )

    return InjectedSourceStreamV3(
        source_family=source_family,
        parser_binding_sha256=parser_binding_sha256,
        parse_event_ledger_sha256=parse_event_ledger_sha256,
        records=adapted(),
        invalid_utf8_drop_count=invalid_utf8_drop_count,
        empty_text_drop_count=empty_text_drop_count,
        quality_drop_count=quality_drop_count,
    )


@dataclass(frozen=True)
class MaterializationInputV3:
    """Network-free source boundary for one P-A materialization.

    Public stream injection is deliberately fixture-only.  Production accepts
    only a factory-minted download receipt, its independently re-read cache
    receipt, and the cache root whose bytes will be re-read again by the exact
    source-I/O parsers.  Thus caller-supplied text or caller-supplied parser
    hashes can never enter an authoritative run.
    """

    mode: str
    streams: tuple[InjectedSourceStreamV3, ...] = ()
    upstream_enumeration: UpstreamEnumerationReceiptV3 | None = None
    verified_cache: VerifiedLocalCacheManifestV3 | None = None
    source_cache_download_receipt: SourceCacheDownloadReceiptV3 | None = None
    cache_root: Path | None = None
    fixture_source_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MATERIALIZATION_MODES:
            raise ValueError("input mode must be explicit")
        if self.mode == PRODUCTION_MODE:
            if self.streams:
                raise CorpusMaterializationError(
                    "production P-A rejects injected source streams"
                )
            if (
                not isinstance(self.upstream_enumeration, UpstreamEnumerationReceiptV3)
                or not self.upstream_enumeration.authoritative
                or self.upstream_enumeration.mode != AUTHORITATIVE_MODE
            ):
                raise CorpusMaterializationError(
                    "production P-A requires an authoritative upstream enumeration"
                )
            if not isinstance(self.verified_cache, VerifiedLocalCacheManifestV3):
                raise CorpusMaterializationError(
                    "production P-A requires a byte-verified source cache"
                )
            if not isinstance(
                self.source_cache_download_receipt, SourceCacheDownloadReceiptV3
            ):
                raise CorpusMaterializationError(
                    "production P-A requires a factory-minted cache download receipt"
                )
            if not isinstance(self.cache_root, Path):
                raise CorpusMaterializationError(
                    "production P-A requires the verified cache root"
                )
            # Inspect the lexical chain before resolve/stat/open.  This rejects
            # both POSIX symlinks and Windows reparse points.
            lexical_cache_root = assert_no_symlink_ancestors(self.cache_root)
            resolved_cache_root = lexical_cache_root.resolve(strict=True)
            if not resolved_cache_root.is_dir():
                raise CorpusMaterializationError("production cache root is not a directory")
            if (resolved_cache_root.name or ".") != self.verified_cache.cache_root_label:
                raise CorpusMaterializationError(
                    "production cache root label differs from its verification receipt"
                )
            download = self.source_cache_download_receipt
            if (
                download.enumeration_mode != AUTHORITATIVE_MODE
                or download.enumeration_receipt_sha256
                != self.upstream_enumeration.receipt_sha256
                or download.source_manifest != self.verified_cache.source_manifest
                or download.verification_receipt_sha256
                != self.verified_cache.verification_receipt_sha256
            ):
                raise CorpusMaterializationError(
                    "download, enumeration, and verified-cache receipts do not compose"
                )
            if self.fixture_source_identity_sha256 is not None:
                raise ValueError("production input may not carry a fixture identity")
        else:
            if tuple(stream.source_family for stream in self.streams) != SOURCE_FAMILIES:
                raise ValueError(
                    "fixture streams must cover every family in canonical order"
                )
            if any(
                value is not None
                for value in (
                    self.upstream_enumeration,
                    self.verified_cache,
                    self.source_cache_download_receipt,
                    self.cache_root,
                )
            ):
                raise ValueError("fixture input may not carry production receipts")
            if self.fixture_source_identity_sha256 is None:
                raise ValueError("fixture input requires an explicit source identity")
            _require_sha256(
                self.fixture_source_identity_sha256, "fixture source identity"
            )
        if self.upstream_enumeration is not None and self.verified_cache is not None:
            enumerated = {
                (
                    asset.source_family,
                    asset.repository,
                    asset.config,
                    asset.revision,
                    asset.split,
                    asset.asset_locator,
                ): asset
                for family in self.upstream_enumeration.families
                for asset in family.assets
            }
            for verified in self.verified_cache.assets:
                expected = verified.expected
                key = (
                    expected.source_family,
                    expected.repository,
                    expected.config,
                    expected.revision,
                    expected.split,
                    expected.asset_locator,
                )
                upstream = enumerated.get(key)
                if upstream is None:
                    raise CorpusMaterializationError(
                        "verified cache asset is absent from upstream enumeration"
                    )
                if upstream.upstream_bytes != expected.bytes:
                    raise CorpusMaterializationError(
                        "verified cache bytes differ from upstream enumeration"
                    )
                if (
                    upstream.content_sha256 is not None
                    and upstream.content_sha256 != expected.sha256
                ):
                    raise CorpusMaterializationError(
                        "verified cache hash differs from upstream content identity"
                    )
            # Reconstruct the factory plan from the complete enumeration, not
            # from caller order, and bind the exact deterministic subset/order.
            selected_identities = {
                asset.expected.asset_identity_sha256
                for asset in self.verified_cache.assets
            }
            selected_upstream = tuple(
                asset
                for family in self.upstream_enumeration.families
                for asset in family.assets
                if asset.asset_identity_sha256 in selected_identities
            )
            if len(selected_upstream) != len(selected_identities):
                raise CorpusMaterializationError(
                    "verified cache subset is incomplete in the enumeration"
                )
            expected_plan = plan_source_cache_assets_v3(
                self.upstream_enumeration, selected_upstream
            )
            assert self.source_cache_download_receipt is not None
            if (
                self.source_cache_download_receipt.selection_plan_sha256
                != expected_plan.plan_sha256
            ):
                raise CorpusMaterializationError(
                    "download receipt selection plan/order differs from enumeration"
                )

    @property
    def source_identity_sha256(self) -> str:
        if self.mode == PRODUCTION_MODE:
            assert self.upstream_enumeration is not None
            assert self.verified_cache is not None
            assert self.source_cache_download_receipt is not None
            return execution_authority_v3_bound_sha256(
                "weft1_corpus_materialization_transport_input_v3",
                {
                    "download_receipt_sha256": (
                        self.source_cache_download_receipt.receipt_sha256
                    ),
                    "selection_plan_sha256": (
                        self.source_cache_download_receipt.selection_plan_sha256
                    ),
                    "upstream_enumeration": self.upstream_enumeration.receipt_sha256,
                    "verified_cache": self.verified_cache.verification_receipt_sha256,
                },
            )
        assert self.fixture_source_identity_sha256 is not None
        return self.fixture_source_identity_sha256


class LanguageClassifierV3(Protocol):
    def classify(self, document: StableDocumentV3) -> LanguageIdDecisionV3: ...


@dataclass(frozen=True)
class MaterializationResultV3:
    mode: str
    source_identity_sha256: str
    content_identity_sha256: str
    d1_ready_manifest_sha256: str
    output_root: Path
    work_root: Path


def screen_order_digest_v3(stratum: str, document_id: str) -> bytes:
    """Return A2-R7's exact per-stratum external-sort key."""

    if stratum not in GTOK_STRATA:
        raise ValueError("screen order uses an unknown stratum")
    _require_sha256(document_id, "screen document ID")
    encoded = stratum.encode("utf-8")
    return hashlib.sha256(
        SCREEN_ORDER_DOMAIN
        + b"\x00"
        + A2_SCREEN_ORDER_SEED.to_bytes(8, "big")
        + len(encoded).to_bytes(8, "big")
        + encoded
        + document_id.encode("ascii")
    ).digest()


def _write_canonical_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise CorpusMaterializationError(f"refusing to overwrite {path.name}")
    payload = canonical_json_bytes(value) + b"\n"
    partial.write_bytes(payload)
    partial.replace(path)


def framed_jsonl_identity_sha256_v3(path: Path, *, domain: bytes) -> str:
    """Hash a canonical JSONL ledger using the parent-verifier framing."""

    if not isinstance(path, Path) or not isinstance(domain, bytes) or not domain:
        raise TypeError("framed JSONL identity requires a path and nonempty bytes domain")
    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(8, "big"))
    digest.update(domain)
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                raise CorpusMaterializationError(
                    "semantic JSONL identity found an unterminated line"
                )
            digest.update(len(line).to_bytes(8, "big"))
            digest.update(line)
    return digest.hexdigest()


def _within_tolerance(remaining: int, target: int) -> bool:
    return Fraction(remaining, target) <= FIRST_FIT_TOLERANCE


def _fraction_payload(numerator: int, denominator: int) -> dict[str, int]:
    value = Fraction(numerator, denominator)
    return {"denominator": value.denominator, "numerator": value.numerator}


def _synthetic_shingle(
    level: Fraction, pair_ordinal: int, shingle_ordinal: int
) -> bytes:
    payload = (
        SYNTHETIC_RECALL_DOMAIN
        + level.numerator.to_bytes(8, "big")
        + level.denominator.to_bytes(8, "big")
        + pair_ordinal.to_bytes(8, "big")
        + shingle_ordinal.to_bytes(8, "big")
    )
    return hashlib.sha256(payload).digest()[:13]


def synthetic_recall_cells_v3() -> tuple[MinHashSyntheticRecallCellV3, ...]:
    """Run the registered six-cell deterministic, report-only qualification."""

    cells: list[MinHashSyntheticRecallCellV3] = []
    for level in MINHASH_RECALL_JACCARD_LEVELS:
        candidates = 0
        for pair_ordinal in range(SYNTHETIC_RECALL_PAIRS_PER_CELL):
            union = tuple(
                _synthetic_shingle(level, pair_ordinal, shingle_ordinal)
                for shingle_ordinal in range(level.denominator)
            )
            if len(set(union)) != len(union):
                raise CorpusMaterializationError(
                    "synthetic recall fixture had a shingle collision"
                )
            left = frozenset(union[: level.numerator])
            right = frozenset(union)
            if exact_jaccard_v3(left, right) != level:
                raise CorpusMaterializationError(
                    "synthetic recall fixture has the wrong exact Jaccard"
                )
            left_bands = lsh_band_keys_v3(minhash_signature_v3(left))
            right_bands = lsh_band_keys_v3(minhash_signature_v3(right))
            candidates += int(
                any(
                    left_band == right_band
                    for left_band, right_band in zip(
                        left_bands, right_bands, strict=True
                    )
                )
            )
        cells.append(
            MinHashSyntheticRecallCellV3(
                exact_jaccard=level,
                pair_count=SYNTHETIC_RECALL_PAIRS_PER_CELL,
                candidate_count=candidates,
            )
        )
    return tuple(cells)


class _Spool:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.event_ordinal = 0
        self.language_ordinal = 0
        self.full_ordinal = 0
        self.pending_mutations = 0
        self._setup()
        self.connection.execute("BEGIN IMMEDIATE")

    def _setup(self) -> None:
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA temp_store = FILE")
        self.connection.executescript(
            """
            CREATE TABLE seen_documents (
              document_id TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE selected_documents (
              document_id TEXT PRIMARY KEY,
              raw_content_id TEXT NOT NULL UNIQUE,
              raw_content_sha256 TEXT NOT NULL,
              stable_source_record_id TEXT NOT NULL,
              source TEXT NOT NULL,
              stratum TEXT NOT NULL,
              retained_bytes INTEGER NOT NULL,
              text BLOB NOT NULL,
              cluster_id TEXT NOT NULL,
              screen_key BLOB NOT NULL,
              full_ordinal INTEGER NOT NULL UNIQUE
            ) STRICT;
            CREATE INDEX selected_screen_order
              ON selected_documents(stratum, screen_key, document_id);
            CREATE TABLE near_cluster_nodes (
              document_id TEXT PRIMARY KEY,
              parent_document_id TEXT NOT NULL
            ) WITHOUT ROWID, STRICT;
            CREATE TABLE near_cluster_lsh (
              band_index INTEGER NOT NULL,
              band_key BLOB NOT NULL,
              document_id TEXT NOT NULL,
              PRIMARY KEY(band_index, band_key, document_id)
            ) WITHOUT ROWID, STRICT;
            CREATE INDEX near_cluster_lsh_lookup
              ON near_cluster_lsh(band_index, band_key, document_id);
            CREATE TABLE near_cluster_edges (
              left_document_id TEXT NOT NULL,
              right_document_id TEXT NOT NULL,
              jaccard_numerator INTEGER NOT NULL,
              jaccard_denominator INTEGER NOT NULL,
              PRIMARY KEY(left_document_id, right_document_id)
            ) WITHOUT ROWID, STRICT;
            CREATE TEMP TABLE near_cluster_candidates (
              document_id TEXT PRIMARY KEY
            ) WITHOUT ROWID, STRICT;
            CREATE TABLE selection_events (
              event_ordinal INTEGER PRIMARY KEY,
              payload BLOB NOT NULL
            ) STRICT;
            CREATE TABLE language_events (
              event_ordinal INTEGER PRIMARY KEY,
              payload BLOB NOT NULL
            ) STRICT;
            CREATE TABLE split_documents (
              stream TEXT NOT NULL,
              stratum TEXT NOT NULL,
              stream_ordinal INTEGER NOT NULL,
              document_id TEXT NOT NULL,
              PRIMARY KEY(stream, stratum, stream_ordinal),
              UNIQUE(stream, document_id),
              FOREIGN KEY(document_id) REFERENCES selected_documents(document_id)
            ) STRICT;
            CREATE TABLE split_decisions (
              decision_ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
              stream TEXT NOT NULL,
              stratum TEXT NOT NULL,
              document_id TEXT NOT NULL,
              disposition TEXT NOT NULL,
              retained_bytes INTEGER NOT NULL,
              remaining_before INTEGER NOT NULL,
              remaining_after INTEGER NOT NULL
            ) STRICT;
            CREATE TABLE heldout_candidates (
              stratum TEXT NOT NULL,
              candidate_ordinal INTEGER NOT NULL,
              document_id TEXT NOT NULL,
              PRIMARY KEY(stratum, candidate_ordinal),
              UNIQUE(stratum, document_id)
            ) STRICT;
            CREATE TABLE recall_samples (
              source TEXT NOT NULL,
              sample_key BLOB NOT NULL,
              document_id TEXT NOT NULL UNIQUE,
              stable_source_record_id TEXT NOT NULL,
              text BLOB NOT NULL,
              PRIMARY KEY(source, sample_key, document_id)
            ) STRICT;
            """
        )

    def close(self) -> None:
        if self.connection.in_transaction:
            self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()

    def mutated(self, count: int = 1) -> None:
        self.pending_mutations += count
        if self.pending_mutations >= 4096:
            self.connection.commit()
            self.connection.execute("BEGIN IMMEDIATE")
            self.pending_mutations = 0

    def flush(self) -> None:
        if self.connection.in_transaction:
            self.connection.commit()
        self.connection.execute("BEGIN IMMEDIATE")
        self.pending_mutations = 0

    def mark_seen(self, document_id: str) -> bool:
        try:
            self.connection.execute(
                "INSERT INTO seen_documents VALUES (?)", (document_id,)
            )
        except sqlite3.IntegrityError:
            return False
        self.mutated()
        return True

    def event(self, value: Mapping[str, object]) -> None:
        self.connection.execute(
            "INSERT INTO selection_events VALUES (?, ?)",
            (self.event_ordinal, canonical_json_bytes(dict(value))),
        )
        self.mutated()
        self.event_ordinal += 1

    def language(self, value: Mapping[str, object]) -> None:
        self.connection.execute(
            "INSERT INTO language_events VALUES (?, ?)",
            (self.language_ordinal, canonical_json_bytes(dict(value))),
        )
        self.mutated()
        self.language_ordinal += 1

    def selected_content_owner(self, document: StableDocumentV3) -> str | None:
        row = self.connection.execute(
            "SELECT source, retained_bytes, raw_content_sha256, text FROM "
            "selected_documents WHERE raw_content_id = ?",
            (document.shard_record_id,),
        ).fetchone()
        if row is None:
            return None
        if (
            int(row["retained_bytes"]) != document.retained_byte_count
            or row["raw_content_sha256"] != document.retained_sha256
            or bytes(row["text"]) != document.retained_bytes
        ):
            raise CorpusMaterializationError(
                "raw-content SHA-1 collision detected; refusing to deduplicate"
            )
        return str(row["source"])

    def select(self, document: StableDocumentV3) -> None:
        text = document.retained_bytes
        if self.selected_content_owner(document) is not None:
            raise CorpusMaterializationError(
                "selected corpus attempted to repeat a raw-content ID"
            )
        self.connection.execute(
            "INSERT INTO selected_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                document.document_id,
                document.shard_record_id,
                document.retained_sha256,
                document.stable_source_record_id,
                document.source,
                document.stratum,
                len(text),
                text,
                document.document_id,
                screen_order_digest_v3(document.stratum, document.document_id),
                self.full_ordinal,
            ),
        )
        self._add_near_cluster_document(document)
        self.mutated()
        self.full_ordinal += 1

    def _cluster_root(self, document_id: str) -> str:
        trail: list[str] = []
        current = document_id
        while True:
            row = self.connection.execute(
                "SELECT parent_document_id FROM near_cluster_nodes "
                "WHERE document_id = ?",
                (current,),
            ).fetchone()
            if row is None:
                raise CorpusMaterializationError("near-cluster node is absent")
            parent = str(row[0])
            if parent == current:
                break
            trail.append(current)
            current = parent
        for item in trail:
            self.connection.execute(
                "UPDATE near_cluster_nodes SET parent_document_id = ? "
                "WHERE document_id = ?",
                (current, item),
            )
        return current

    def _union_clusters(self, left: str, right: str) -> None:
        left_root = self._cluster_root(left)
        right_root = self._cluster_root(right)
        if left_root == right_root:
            return
        winner, loser = sorted((left_root, right_root))
        self.connection.execute(
            "UPDATE near_cluster_nodes SET parent_document_id = ? "
            "WHERE document_id = ?",
            (winner, loser),
        )

    def _add_near_cluster_document(self, document: StableDocumentV3) -> None:
        """Add one document to the deterministic registered LSH/Jaccard graph."""

        self.connection.execute(
            "INSERT INTO near_cluster_nodes VALUES (?, ?)",
            (document.document_id, document.document_id),
        )
        match_bytes = normalized_match_bytes(document.text)
        if not match_bytes:
            return
        shingles = byte_shingles_v3(match_bytes)
        bands = lsh_band_keys_v3(minhash_signature_v3(shingles))
        self.connection.execute("DELETE FROM near_cluster_candidates")
        for band_index, band_key in enumerate(bands):
            self.connection.execute(
                "INSERT OR IGNORE INTO near_cluster_candidates "
                "SELECT document_id FROM near_cluster_lsh WHERE "
                "band_index = ? AND band_key = ?",
                (band_index, band_key),
            )
        for row in self.connection.execute(
            "SELECT c.document_id, d.text FROM near_cluster_candidates AS c "
            "JOIN selected_documents AS d ON d.document_id = c.document_id "
            "ORDER BY c.document_id"
        ):
            candidate_id = str(row["document_id"])
            candidate_text = bytes(row["text"]).decode("utf-8", errors="strict")
            candidate_match = normalized_match_bytes(candidate_text)
            if not candidate_match:
                continue
            score = exact_jaccard_v3(
                shingles, byte_shingles_v3(candidate_match)
            )
            if score < NEAR_DUPLICATE_THRESHOLD:
                continue
            left, right = sorted((candidate_id, document.document_id))
            self.connection.execute(
                "INSERT OR IGNORE INTO near_cluster_edges VALUES (?, ?, ?, ?)",
                (left, right, score.numerator, score.denominator),
            )
            self._union_clusters(left, right)
        self.connection.executemany(
            "INSERT INTO near_cluster_lsh VALUES (?, ?, ?)",
            tuple(
                (band_index, band_key, document.document_id)
                for band_index, band_key in enumerate(bands)
            ),
        )

    def finalize_near_clusters(self) -> dict[str, object]:
        document_count = int(
            self.connection.execute(
                "SELECT count(*) FROM selected_documents"
            ).fetchone()[0]
        )
        nodes = tuple(
            str(row[0])
            for row in self.connection.execute(
                "SELECT document_id FROM near_cluster_nodes ORDER BY document_id"
            )
        )
        if len(nodes) != document_count:
            raise CorpusMaterializationError(
                "near-cluster graph does not cover the full corpus"
            )
        for document_id in nodes:
            root = self._cluster_root(document_id)
            self.connection.execute(
                "UPDATE selected_documents SET cluster_id = ? WHERE document_id = ?",
                (root, document_id),
            )
        edge_count = int(
            self.connection.execute(
                "SELECT count(*) FROM near_cluster_edges"
            ).fetchone()[0]
        )
        cluster_count = int(
            self.connection.execute(
                "SELECT count(DISTINCT cluster_id) FROM selected_documents"
            ).fetchone()[0]
        )
        self.mutated(len(nodes))
        return {
            "algorithm_identity_sha256": execution_authority_v3_bound_sha256(
                "weft1_corpus_near_cluster_algorithm_v3",
                {
                    "match_normalization_binding_sha256": (
                        A2_MATCH_NORMALIZATION_BINDING.receipt_sha256
                    ),
                    "minhash_binding_sha256": A2_MINHASH_BINDING.receipt_sha256,
                    "union_root_rule": "LEXICOGRAPHIC_MIN_DOCUMENT_ID",
                },
            ),
            "cluster_count": cluster_count,
            "document_count": document_count,
            "qualifying_edge_count": edge_count,
            "semantics": "REGISTERED_LSH_CANDIDATE_GRAPH_EXACT_JACCARD_COMPONENTS",
        }

    def consider_recall_sample(self, document: StableDocumentV3) -> None:
        if document.source not in {"dolma_web", "fineweb_edu"}:
            raise ValueError("real recall sampling accepts only dedup sources")
        if not normalize_match_text(document.text):
            return
        sample_key = hashlib.sha256(
            RECALL_SAMPLE_DOMAIN
            + A2_DEDUP_SEED.to_bytes(8, "big")
            + document.source.encode("ascii")
            + b"\x00"
            + document.document_id.encode("ascii")
        ).digest()
        count = int(
            self.connection.execute(
                "SELECT count(*) FROM recall_samples WHERE source = ?",
                (document.source,),
            ).fetchone()[0]
        )
        if count >= REAL_RECALL_SAMPLE_PER_SOURCE:
            largest = self.connection.execute(
                "SELECT sample_key, document_id FROM recall_samples WHERE source = ? "
                "ORDER BY sample_key DESC, document_id DESC LIMIT 1",
                (document.source,),
            ).fetchone()
            assert largest is not None
            if (sample_key, document.document_id) >= (
                bytes(largest["sample_key"]),
                largest["document_id"],
            ):
                return
            self.connection.execute(
                "DELETE FROM recall_samples WHERE source = ? AND sample_key = ? "
                "AND document_id = ?",
                (
                    document.source,
                    largest["sample_key"],
                    largest["document_id"],
                ),
            )
        self.connection.execute(
            "INSERT INTO recall_samples VALUES (?, ?, ?, ?, ?)",
            (
                document.source,
                sample_key,
                document.document_id,
                document.stable_source_record_id,
                document.retained_bytes,
            ),
        )
        self.mutated(2 if count >= REAL_RECALL_SAMPLE_PER_SOURCE else 1)

    def recall_sample(self, source: str) -> tuple[StableDocumentV3, ...]:
        if source not in {"dolma_web", "fineweb_edu"}:
            raise ValueError("unknown recall-sample source")
        rows = self.connection.execute(
            "SELECT stable_source_record_id, text FROM recall_samples "
            "WHERE source = ? ORDER BY sample_key, document_id",
            (source,),
        )
        return tuple(
            StableDocumentV3(
                source=source,
                stratum="general",
                stable_source_record_id=row["stable_source_record_id"],
                text=bytes(row["text"]).decode("utf-8", errors="strict"),
            )
            for row in rows
        )

    def export_blob_rows(self, query: str, path: Path) -> str:
        digest = hashlib.sha256()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            for row in self.connection.execute(query):
                payload = bytes(row[0]) + b"\n"
                handle.write(payload)
                digest.update(payload)
        return digest.hexdigest()

    def iter_split_documents(
        self, stream: str, stratum: str
    ) -> Iterator[StableDocumentV3]:
        cursor = self.connection.execute(
            "SELECT d.source, d.stratum, d.stable_source_record_id, d.text "
            "FROM split_documents AS s JOIN selected_documents AS d "
            "ON d.document_id = s.document_id WHERE s.stream = ? AND "
            "s.stratum = ? ORDER BY s.stream_ordinal",
            (stream, stratum),
        )
        for row in cursor:
            yield StableDocumentV3(
                source=row["source"],
                stratum=row["stratum"],
                stable_source_record_id=row["stable_source_record_id"],
                text=bytes(row["text"]).decode("utf-8", errors="strict"),
            )


class _Materializer:
    def __init__(
        self,
        *,
        inputs: MaterializationInputV3,
        plan: MaterializationPlanV3,
        language_classifier: LanguageClassifierV3,
        output_root: Path,
        work_root: Path,
    ) -> None:
        if inputs.mode != plan.mode:
            raise ValueError("input and plan modes differ")
        if not hasattr(language_classifier, "classify"):
            raise TypeError("language classifier must expose classify(document)")
        if inputs.mode == PRODUCTION_MODE:
            from training.weft1_corpus_pa import FastTextLanguageIdAdapterV3

            if type(language_classifier) is not FastTextLanguageIdAdapterV3:
                raise CorpusMaterializationError(
                    "production P-A requires the verified FastText language adapter"
                )
        self.inputs = inputs
        self.plan = plan
        self.classifier = language_classifier
        self.output_root = Path(output_root)
        self.work_root = Path(work_root)
        self.streams = {stream.source_family: stream for stream in inputs.streams}
        self._production_source_db: sqlite3.Connection | None = None
        self.source_parse_receipts: dict[str, dict[str, object]] = {
            stream.source_family: {
                "asset_order_identity_sha256": None,
                "empty_text_drop_count": stream.empty_text_drop_count,
                "invalid_utf8_drop_count": stream.invalid_utf8_drop_count,
                "parse_event_count": None,
                "parse_event_ledger_path": None,
                "parse_event_ledger_sha256": stream.parse_event_ledger_sha256,
                "parser_binding_sha256": stream.parser_binding_sha256,
                "quality_drop_count": stream.quality_drop_count,
                "source_family": stream.source_family,
                "stable_id_duplicate_count": 0,
            }
            for stream in inputs.streams
        }
        self.materialized_source_identity_sha256 = inputs.source_identity_sha256
        self.full_targets = dict(plan.full_pool_target_bytes)
        self.training_targets = dict(plan.training_stratum_target_bytes)
        self.heldout_targets = dict(plan.heldout_stratum_target_bytes)
        self.language_invocations = Counter({stratum: 0 for stratum in GTOK_STRATA})
        self.language_rejections = Counter({stratum: 0 for stratum in GTOK_STRATA})
        self.invalid_utf8_by_source = Counter({source: 0 for source in SOURCE_FAMILIES})
        self.source_parse_drop_counts = {
            source: {"empty_text": 0, "invalid_utf8": 0, "quality_lt3": 0}
            for source in SOURCE_FAMILIES
        }
        for stream in inputs.streams:
            self.invalid_utf8_by_source[stream.source_family] = (
                stream.invalid_utf8_drop_count
            )
            self.source_parse_drop_counts[stream.source_family] = {
                "empty_text": stream.empty_text_drop_count,
                "invalid_utf8": stream.invalid_utf8_drop_count,
                "quality_lt3": stream.quality_drop_count,
            }
        self.duplicate_occurrences_by_source = Counter(
            {source: 0 for source in SOURCE_FAMILIES}
        )
        self.empty_normalization_by_source = Counter(
            {source: 0 for source in SOURCE_FAMILIES}
        )
        self.global_exact_duplicates_by_source = Counter(
            {source: 0 for source in SOURCE_FAMILIES}
        )
        self.recall_population_counts = Counter(
            {"dolma_web": 0, "fineweb_edu": 0}
        )
        self._cache_asset_ids = (
            None
            if inputs.verified_cache is None
            else {
                asset.expected.asset_identity_sha256: asset.expected.source_family
                for asset in inputs.verified_cache.assets
            }
        )

    def _prepare_roots(self) -> None:
        for root, name in (
            (self.output_root, "output_root"),
            (self.work_root, "work_root"),
        ):
            # Lexical inspection must precede exists/resolve/mkdir so a junction
            # or symlink cannot be silently dereferenced at the authority edge.
            assert_no_symlink_ancestors(root)
            if root.exists():
                raise CorpusMaterializationError(f"{name} must be a fresh absent path")
            if root.resolve(strict=False) == root.parent.resolve(strict=False):
                raise CorpusMaterializationError(f"unsafe {name}")
        output_resolved = self.output_root.resolve(strict=False)
        work_resolved = self.work_root.resolve(strict=False)
        if (
            output_resolved == work_resolved
            or output_resolved in work_resolved.parents
            or work_resolved in output_resolved.parents
        ):
            raise CorpusMaterializationError(
                "output and work roots must be disjoint"
            )
        self.output_root.mkdir(parents=True)
        self.work_root.mkdir(parents=True)
        (self.output_root / "_INCOMPLETE").write_bytes(b"P-A incomplete\n")

    def _prepare_production_sources(self) -> None:
        """Re-read and parse the verified subset into a disk-backed canonical spool."""

        if self.inputs.mode != PRODUCTION_MODE:
            return
        assert self.inputs.verified_cache is not None
        assert self.inputs.cache_root is not None
        assert self.inputs.source_cache_download_receipt is not None
        parse_root = self.output_root / "source-parse"
        parse_root.mkdir()
        database_path = self.work_root / "production-source-records.sqlite"
        connection = sqlite3.connect(database_path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA temp_store = FILE")
        connection.executescript(
            """
            CREATE TABLE parsed_records (
              source TEXT NOT NULL,
              stable_source_record_id TEXT NOT NULL,
              source_asset_identity_sha256 TEXT NOT NULL,
              asset_order_ordinal INTEGER NOT NULL,
              asset_record_ordinal INTEGER NOT NULL,
              text BLOB NOT NULL,
              retained_bytes INTEGER NOT NULL,
              int_score INTEGER,
              PRIMARY KEY(source, stable_source_record_id)
            ) WITHOUT ROWID, STRICT;
            CREATE INDEX parsed_unscored_order ON parsed_records(
              source, asset_order_ordinal, asset_record_ordinal,
              stable_source_record_id
            );
            CREATE INDEX parsed_scored_order ON parsed_records(
              source, int_score DESC, stable_source_record_id
            );
            """
        )
        connection.execute("BEGIN IMMEDIATE")
        assets_by_source = {
            source: tuple(
                asset
                for asset in self.inputs.verified_cache.assets
                if asset.expected.source_family == source
            )
            for source in SOURCE_FAMILIES
        }
        try:
            for source in SOURCE_FAMILIES:
                binding = PRODUCTION_PARSER_BINDINGS_V3[source]
                ledger_path = parse_root / f"{source}.jsonl"
                ledger_digest = hashlib.sha256()
                disposition_counts = Counter()
                duplicate_count = 0
                event_ordinal = 0
                with ledger_path.open("xb") as ledger:
                    for asset_order_ordinal, verified_asset in enumerate(
                        assets_by_source[source]
                    ):
                        lexical_asset = self.inputs.cache_root.joinpath(
                            *PurePosixPath(
                                verified_asset.expected.relative_path
                            ).parts
                        )
                        assert_no_symlink_ancestors(lexical_asset)
                        for event in iter_source_asset_events_v3(
                            verified_asset,
                            self.inputs.cache_root,
                            binding=binding,
                        ):
                            payload = canonical_json_bytes(
                                {
                                    "asset_order_ordinal": asset_order_ordinal,
                                    "event_ordinal": event_ordinal,
                                    "event_sha256": event.event_sha256,
                                    "source_asset_identity_sha256": (
                                        verified_asset.expected.asset_identity_sha256
                                    ),
                                    "source_family": source,
                                    "source_record_ordinal": (
                                        event.source_record_ordinal
                                    ),
                                    "disposition": event.disposition,
                                }
                            ) + b"\n"
                            ledger.write(payload)
                            ledger_digest.update(payload)
                            event_ordinal += 1
                            disposition_counts[event.disposition] += 1
                            if event.disposition != RETAIN:
                                continue
                            assert event.record is not None
                            parsed = event.record
                            raw = parsed.raw_document
                            canonical = parsed.canonical_record
                            if parsed.parser_binding_sha256 != binding.binding_sha256:
                                raise CorpusMaterializationError(
                                    "production parser emitted a foreign binding"
                                )
                            if isinstance(raw.text, bytes):
                                text_bytes = raw.text
                            else:
                                text_bytes = raw.text.encode("utf-8", errors="strict")
                            try:
                                connection.execute(
                                    "INSERT INTO parsed_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                    (
                                        source,
                                        raw.stable_source_record_id,
                                        canonical.asset.asset_identity_sha256,
                                        asset_order_ordinal,
                                        event.source_record_ordinal,
                                        text_bytes,
                                        canonical.retained_byte_count,
                                        canonical.int_score,
                                    ),
                                )
                            except sqlite3.IntegrityError:
                                # Upsampled source mixes can repeat a stable
                                # document.  Canonical first asset/row wins,
                                # but only after byte/metadata equality proves
                                # this is multiplicity rather than an ID clash.
                                existing = connection.execute(
                                    "SELECT text, retained_bytes, int_score FROM "
                                    "parsed_records WHERE source = ? AND "
                                    "stable_source_record_id = ?",
                                    (source, raw.stable_source_record_id),
                                ).fetchone()
                                if existing is None or (
                                    bytes(existing["text"]) != text_bytes
                                    or int(existing["retained_bytes"])
                                    != canonical.retained_byte_count
                                    or existing["int_score"] != canonical.int_score
                                ):
                                    raise CorpusMaterializationError(
                                        "stable source record ID repeats with different bytes"
                                    )
                                duplicate_count += 1
                            if event_ordinal % 4096 == 0:
                                connection.commit()
                                connection.execute("BEGIN IMMEDIATE")
                asset_order_identity = execution_authority_v3_bound_sha256(
                    "weft1_corpus_source_asset_consumption_order_v3",
                    tuple(
                        asset.expected.asset_identity_sha256
                        for asset in assets_by_source[source]
                    ),
                )
                self.source_parse_drop_counts[source] = {
                    "empty_text": disposition_counts[DROP_EMPTY_TEXT],
                    "invalid_utf8": disposition_counts[DROP_INVALID_UTF8],
                    "quality_lt3": disposition_counts[DROP_QUALITY_LT3],
                }
                self.invalid_utf8_by_source[source] = disposition_counts[
                    DROP_INVALID_UTF8
                ]
                self.source_parse_receipts[source] = {
                    "asset_order_identity_sha256": asset_order_identity,
                    "empty_text_drop_count": disposition_counts[DROP_EMPTY_TEXT],
                    "invalid_utf8_drop_count": disposition_counts[DROP_INVALID_UTF8],
                    "parse_event_count": event_ordinal,
                    "parse_event_ledger_path": f"source-parse/{source}.jsonl",
                    "parse_event_ledger_sha256": ledger_digest.hexdigest(),
                    "parser_binding_sha256": binding.binding_sha256,
                    "quality_drop_count": disposition_counts[DROP_QUALITY_LT3],
                    "source_family": source,
                    "stable_id_duplicate_count": duplicate_count,
                }
            connection.commit()
            self._production_source_db = connection
            self.materialized_source_identity_sha256 = (
                execution_authority_v3_bound_sha256(
                    "weft1_corpus_materialization_source_input_v3",
                    {
                        "parser_receipts": tuple(
                            self.source_parse_receipts[source]
                            for source in SOURCE_FAMILIES
                        ),
                        "transport_identity_sha256": (
                            self.inputs.source_identity_sha256
                        ),
                    },
                )
            )
        except BaseException:
            connection.rollback()
            connection.close()
            raise

    def _ordered_records(
        self, source: str
    ) -> Iterator[MaterializerSourceRecordV3]:
        if self.inputs.mode == PRODUCTION_MODE:
            if self._production_source_db is None:
                raise CorpusMaterializationError(
                    "production source parser spool was not prepared"
                )
            if source in SCORED_SOURCE_FAMILIES:
                order = "int_score DESC, stable_source_record_id"
            else:
                order = (
                    "asset_order_ordinal, asset_record_ordinal, "
                    "stable_source_record_id"
                )
            cursor = self._production_source_db.execute(
                "SELECT stable_source_record_id, source_asset_identity_sha256, "
                "text, retained_bytes, int_score FROM parsed_records WHERE "
                f"source = ? ORDER BY {order}",
                (source,),
            )
            for ordinal, row in enumerate(cursor):
                yield MaterializerSourceRecordV3(
                    source_family=source,
                    stratum=SOURCE_TO_STRATUM[source],
                    source_order_ordinal=ordinal,
                    stable_source_record_id=row["stable_source_record_id"],
                    source_asset_identity_sha256=(
                        row["source_asset_identity_sha256"]
                    ),
                    text=bytes(row["text"]),
                    declared_retained_byte_count=int(row["retained_bytes"]),
                    int_score=(
                        None if row["int_score"] is None else int(row["int_score"])
                    ),
                )
            return
        previous_score_key: tuple[int, str] | None = None
        expected_ordinal = 0
        for record in self.streams[source].records:
            if not isinstance(record, MaterializerSourceRecordV3):
                raise TypeError("source stream emitted an untyped record")
            if record.source_family != source:
                raise CorpusMaterializationError("source stream crossed family boundaries")
            if record.source_order_ordinal != expected_ordinal:
                raise CorpusMaterializationError("source order ordinals are not contiguous")
            expected_ordinal += 1
            if self._cache_asset_ids is not None:
                cached_family = self._cache_asset_ids.get(
                    record.source_asset_identity_sha256
                )
                if cached_family is None:
                    raise CorpusMaterializationError(
                        "source record names an asset outside the verified cache"
                    )
                if cached_family != source:
                    raise CorpusMaterializationError(
                        "source record names a verified asset from another family"
                    )
            if source in SCORED_SOURCE_FAMILIES:
                assert record.int_score is not None
                key = (-record.int_score, record.stable_source_record_id)
                if previous_score_key is not None and key < previous_score_key:
                    raise CorpusMaterializationError(
                        "score-ranked source stream is not in canonical order"
                    )
                previous_score_key = key
            yield record

    def _eligible_documents(
        self, source: str, spool: _Spool
    ) -> Iterator[StableDocumentV3]:
        for record in self._ordered_records(source):
            if (
                source in QUALITY_GATED_SOURCE_FAMILIES
                and record.int_score is not None
                and record.int_score < 3
            ):
                spool.event(
                    {
                        "action": "DROP_QUALITY_SCORE_BELOW_3",
                        "source": source,
                        "source_order_ordinal": record.source_order_ordinal,
                        "stable_source_record_id": record.stable_source_record_id,
                    }
                )
                continue
            try:
                if isinstance(record.text, bytes):
                    text = record.text.decode("utf-8", errors="strict")
                else:
                    record.text.encode("utf-8", errors="strict")
                    text = record.text
            except UnicodeError:
                self.invalid_utf8_by_source[source] += 1
                spool.event(
                    {
                        "action": "DROP_INVALID_UTF8",
                        "source": source,
                        "source_order_ordinal": record.source_order_ordinal,
                        "stable_source_record_id": record.stable_source_record_id,
                    }
                )
                continue
            if not text:
                spool.event(
                    {
                        "action": "DROP_ZERO_RETAINED_BYTES",
                        "source": source,
                        "source_order_ordinal": record.source_order_ordinal,
                        "stable_source_record_id": record.stable_source_record_id,
                    }
                )
                continue
            if len(text.encode("utf-8")) != record.declared_retained_byte_count:
                raise CorpusMaterializationError(
                    "parser-declared retained byte count differs from text"
                )
            document = StableDocumentV3(
                source=source,
                stratum=SOURCE_TO_STRATUM[source],
                stable_source_record_id=record.stable_source_record_id,
                text=text,
            )
            if not spool.mark_seen(document.document_id):
                self.duplicate_occurrences_by_source[source] += 1
                spool.event(
                    {
                        "action": "DROP_DUPLICATE_OCCURRENCE",
                        "document_id": document.document_id,
                        "source": source,
                        "source_order_ordinal": record.source_order_ordinal,
                    }
                )
                continue
            if document.stratum == "general":
                self.language_invocations["general"] += 1
                decision = self.classifier.classify(document)
                if not isinstance(decision, LanguageIdDecisionV3):
                    raise TypeError("language classifier returned an untyped decision")
                if (
                    decision.document_id != document.document_id
                    or decision.stratum != "general"
                ):
                    raise CorpusMaterializationError(
                        "language decision is bound to a different document"
                    )
                spool.language(
                    {
                        "binding_sha256": decision.binding_sha256,
                        "document_id": decision.document_id,
                        "keep": decision.keep,
                        "label": decision.label,
                        "probability": decision.probability,
                        "receipt_sha256": decision.receipt_sha256,
                        "scoring_input_sha256": decision.scoring_input_sha256,
                        "source": source,
                    }
                )
                if not decision.keep:
                    self.language_rejections["general"] += 1
                    spool.event(
                        {
                            "action": "DROP_LANGUAGE_ID",
                            "document_id": document.document_id,
                            "source": source,
                        }
                    )
                    continue
            yield document

    def _select_simple_pool(
        self,
        *,
        source: str,
        pool: str,
        target: int,
        spool: _Spool,
    ) -> dict[str, object]:
        remaining = target
        considered = 0
        oversized = 0
        source_exhausted = False
        documents = iter(self._eligible_documents(source, spool))
        while not _within_tolerance(remaining, target):
            try:
                document = next(documents)
            except StopIteration:
                source_exhausted = True
                break
            considered += 1
            before = remaining
            owner = spool.selected_content_owner(document)
            if owner is not None:
                self.global_exact_duplicates_by_source[source] += 1
                action = "DROP_GLOBAL_EXACT_DUPLICATE"
            elif document.retained_byte_count > remaining:
                oversized += 1
                action = "SKIP_OVERSIZED_REMAINING_CAPACITY"
            else:
                spool.select(document)
                remaining -= document.retained_byte_count
                action = "SELECT_FULL_CORPUS"
            spool.event(
                {
                    "action": action,
                    "document_id": document.document_id,
                    "pool": pool,
                    "remaining_after": remaining,
                    "remaining_before": before,
                    "retained_byte_count": document.retained_byte_count,
                    "source": source,
                }
            )
        if not _within_tolerance(remaining, target):
            raise CorpusMaterializationError(
                f"{pool} exhausted {remaining} bytes above the 0.5% tolerance"
            )
        return {
            "considered_documents": considered,
            "deficit_bytes": remaining,
            "oversized_skips": oversized,
            "pool": pool,
            "realized_bytes": target - remaining,
            "source_exhausted": source_exhausted,
            "target_bytes": target,
        }

    def _select_science(self, spool: _Spool) -> dict[str, object]:
        target = self.full_targets["science_technical_combined"]
        remaining = target
        considered = 0
        oversized = 0
        per_source = Counter()
        for source in ("arxiv", "olmocr"):
            if _within_tolerance(remaining, target):
                break
            documents = iter(self._eligible_documents(source, spool))
            while not _within_tolerance(remaining, target):
                try:
                    document = next(documents)
                except StopIteration:
                    break
                considered += 1
                before = remaining
                owner = spool.selected_content_owner(document)
                if owner is not None:
                    self.global_exact_duplicates_by_source[source] += 1
                    action = "DROP_GLOBAL_EXACT_DUPLICATE"
                elif document.retained_byte_count > remaining:
                    oversized += 1
                    action = "SKIP_OVERSIZED_REMAINING_CAPACITY"
                else:
                    spool.select(document)
                    remaining -= document.retained_byte_count
                    per_source[source] += document.retained_byte_count
                    action = "SELECT_FULL_CORPUS"
                spool.event(
                    {
                        "action": action,
                        "document_id": document.document_id,
                        "pool": "science_technical_combined",
                        "remaining_after": remaining,
                        "remaining_before": before,
                        "retained_byte_count": document.retained_byte_count,
                        "source": source,
                    }
                )
        if not _within_tolerance(remaining, target):
            raise CorpusMaterializationError(
                "science sources exhausted above the 0.5% tolerance"
            )
        return {
            "considered_documents": considered,
            "deficit_bytes": remaining,
            "oversized_skips": oversized,
            "per_source_realized_bytes": tuple(sorted(per_source.items())),
            "pool": "science_technical_combined",
            "realized_bytes": target - remaining,
            "target_bytes": target,
        }

    def _split_one_stratum(
        self, spool: _Spool, *, stratum: str
    ) -> dict[str, object]:
        training_target = self.training_targets[stratum]
        heldout_target = self.heldout_targets[stratum]
        training_remaining = training_target
        heldout_candidate_ordinal = 0
        last_considered: tuple[bytes, str] | None = None
        training_ordinal = 0
        cursor = spool.connection.execute(
            "SELECT document_id, retained_bytes, screen_key FROM selected_documents "
            "WHERE stratum = ? ORDER BY screen_key, document_id",
            (stratum,),
        )
        for row in cursor:
            if _within_tolerance(training_remaining, training_target):
                break
            document_id = row["document_id"]
            retained = int(row["retained_bytes"])
            key = bytes(row["screen_key"])
            last_considered = (key, document_id)
            before = training_remaining
            if retained > training_remaining:
                disposition = "oversized_remaining_capacity"
                spool.connection.execute(
                    "INSERT INTO heldout_candidates VALUES (?, ?, ?)",
                    (stratum, heldout_candidate_ordinal, document_id),
                )
                spool.mutated()
                heldout_candidate_ordinal += 1
            else:
                disposition = "accepted"
                training_remaining -= retained
                spool.connection.execute(
                    "INSERT INTO split_documents VALUES ('T', ?, ?, ?)",
                    (stratum, training_ordinal, document_id),
                )
                spool.mutated()
                training_ordinal += 1
            spool.connection.execute(
                "INSERT INTO split_decisions(stream, stratum, document_id, "
                "disposition, retained_bytes, remaining_before, remaining_after) "
                "VALUES ('T', ?, ?, ?, ?, ?, ?)",
                (
                    stratum,
                    document_id,
                    disposition,
                    retained,
                    before,
                    training_remaining,
                ),
            )
            spool.mutated()
        if not _within_tolerance(training_remaining, training_target):
            raise CorpusMaterializationError(
                f"T/{stratum} exhausted above the 0.5% tolerance"
            )

        suffix_query = (
            "SELECT screen_key, document_id FROM selected_documents "
            "WHERE stratum = ? ORDER BY screen_key, document_id"
        )
        suffix_args: tuple[object, ...] = (stratum,)
        if last_considered is not None:
            suffix_query = (
                "SELECT screen_key, document_id FROM selected_documents WHERE "
                "stratum = ? AND (screen_key > ? OR (screen_key = ? AND "
                "document_id > ?)) ORDER BY screen_key, document_id"
            )
            suffix_args = (
                stratum,
                last_considered[0],
                last_considered[0],
                last_considered[1],
            )
        heldout_remaining = heldout_target
        heldout_ordinal = 0
        heldout_prefix = spool.connection.execute(
            "SELECT document_id FROM heldout_candidates WHERE stratum = ? "
            "ORDER BY candidate_ordinal",
            (stratum,),
        )
        suffix = spool.connection.execute(suffix_query, suffix_args)

        def heldout_ids() -> Iterator[str]:
            for row in heldout_prefix:
                yield row["document_id"]
            for row in suffix:
                yield row["document_id"]

        for document_id in heldout_ids():
            if _within_tolerance(heldout_remaining, heldout_target):
                break
            row = spool.connection.execute(
                "SELECT retained_bytes, cluster_id FROM selected_documents "
                "WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            assert row is not None
            retained = int(row["retained_bytes"])
            before = heldout_remaining
            train_cluster_member = spool.connection.execute(
                "SELECT 1 FROM split_documents AS t "
                "JOIN selected_documents AS td ON td.document_id = t.document_id "
                "WHERE t.stream = 'T' AND td.cluster_id = ? LIMIT 1",
                (row["cluster_id"],),
            ).fetchone()
            if train_cluster_member is not None:
                disposition = "training_cluster_excluded"
            elif retained > heldout_remaining:
                disposition = "oversized_remaining_capacity"
            else:
                disposition = "accepted"
                heldout_remaining -= retained
                spool.connection.execute(
                    "INSERT INTO split_documents VALUES ('H', ?, ?, ?)",
                    (stratum, heldout_ordinal, document_id),
                )
                spool.mutated()
                heldout_ordinal += 1
            spool.connection.execute(
                "INSERT INTO split_decisions(stream, stratum, document_id, "
                "disposition, retained_bytes, remaining_before, remaining_after) "
                "VALUES ('H', ?, ?, ?, ?, ?, ?)",
                (
                    stratum,
                    document_id,
                    disposition,
                    retained,
                    before,
                    heldout_remaining,
                ),
            )
            spool.mutated()
        if not _within_tolerance(heldout_remaining, heldout_target):
            raise CorpusMaterializationError(
                f"H/{stratum} exhausted above the 0.5% tolerance"
            )
        return {
            "heldout": {
                "deficit_bytes": heldout_remaining,
                "document_count": heldout_ordinal,
                "realized_bytes": heldout_target - heldout_remaining,
                "target_bytes": heldout_target,
            },
            "stratum": stratum,
            "training": {
                "deficit_bytes": training_remaining,
                "document_count": training_ordinal,
                "realized_bytes": training_target - training_remaining,
                "target_bytes": training_target,
            },
        }

    def _write_shards(self, spool: _Spool) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        shard_root = self.output_root / "shards"
        for stratum in GTOK_STRATA:
            for stream in ("T", "H"):
                result = write_jsonl_zstd_shards_v3(
                    spool.iter_split_documents(stream, stratum),
                    shard_root,
                    stream=stream,
                    stratum=stratum,
                    shard_target_bytes=self.plan.shard_target_bytes,
                )
                if result.invalid_utf8_total != 0:
                    raise CorpusMaterializationError(
                        "validated spool produced an invalid UTF-8 shard drop"
                    )
                rows.extend(
                    {
                        **asdict(identity),
                        "content_identity_sha256": identity.content_identity_sha256,
                        "identity_relative_path": identity.relative_path,
                        "relative_path": f"shards/{identity.relative_path}",
                        "stream": stream,
                        "stratum": stratum,
                    }
                    for identity in result.shards
                )
        return tuple(sorted(rows, key=lambda row: str(row["relative_path"])))

    def _consumer_order_receipts(
        self, spool: _Spool
    ) -> tuple[dict[str, object], ...]:
        """Prove per-seed T order while keeping arm order structurally shared."""

        multiset_digest = hashlib.sha256()
        document_count = 0
        retained_bytes = 0
        for row in spool.connection.execute(
            "SELECT d.document_id, d.retained_bytes FROM split_documents AS s "
            "JOIN selected_documents AS d ON d.document_id = s.document_id "
            "WHERE s.stream = 'T' ORDER BY d.document_id"
        ):
            encoded = str(row["document_id"]).encode("ascii")
            multiset_digest.update(len(encoded).to_bytes(8, "big"))
            multiset_digest.update(encoded)
            document_count += 1
            retained_bytes += int(row["retained_bytes"])
        if document_count < 2:
            raise CorpusMaterializationError(
                "D6 per-seed ordering requires at least two training documents"
            )
        multiset_identity = multiset_digest.hexdigest()
        spool.connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS consumer_order ("
            "order_key BLOB NOT NULL, document_id TEXT PRIMARY KEY) "
            "WITHOUT ROWID, STRICT"
        )
        receipts: list[dict[str, object]] = []
        for training_seed in GTOK_TRAINING_SEEDS:
            spool.connection.execute("DELETE FROM consumer_order")
            cursor = spool.connection.execute(
                "SELECT document_id FROM split_documents WHERE stream = 'T' "
                "ORDER BY document_id"
            )
            for row in cursor:
                document_id = str(row[0])
                order_key = hashlib.sha256(
                    b"WEFT-1/gtok-training-order/v1\x00"
                    + training_seed.to_bytes(8, "big")
                    + document_id.encode("ascii")
                ).digest()
                spool.connection.execute(
                    "INSERT INTO consumer_order VALUES (?, ?)",
                    (order_key, document_id),
                )
            ordered_ids = hashlib.sha256()
            framed_payload = hashlib.sha256()
            observed_count = 0
            observed_bytes = 0
            for row in spool.connection.execute(
                "SELECT d.document_id, d.text FROM consumer_order AS o "
                "JOIN selected_documents AS d ON d.document_id = o.document_id "
                "ORDER BY o.order_key, o.document_id"
            ):
                document_id_bytes = str(row["document_id"]).encode("ascii")
                payload = bytes(row["text"])
                ordered_ids.update(len(document_id_bytes).to_bytes(8, "big"))
                ordered_ids.update(document_id_bytes)
                framed_payload.update(len(document_id_bytes).to_bytes(8, "big"))
                framed_payload.update(document_id_bytes)
                framed_payload.update(len(payload).to_bytes(8, "big"))
                framed_payload.update(payload)
                observed_count += 1
                observed_bytes += len(payload)
            if observed_count != document_count or observed_bytes != retained_bytes:
                raise CorpusMaterializationError(
                    "D6 consumer-order spool changed the training multiset"
                )
            receipt = {
                "document_count": document_count,
                "document_multiset_sha256": multiset_identity,
                "framed_payload_sha256": framed_payload.hexdigest(),
                "ordered_document_ids_sha256": ordered_ids.hexdigest(),
                "order_key_domain": "WEFT-1/gtok-training-order/v1",
                "retained_text_bytes": retained_bytes,
                "training_seed": training_seed,
            }
            receipt["receipt_sha256"] = execution_authority_v3_bound_sha256(
                "weft1_corpus_consumer_order_receipt_v3", receipt
            )
            receipts.append(receipt)
        if len({row["document_multiset_sha256"] for row in receipts}) != 1:
            raise CorpusMaterializationError("D6 per-seed multisets differ")
        if len({row["ordered_document_ids_sha256"] for row in receipts}) != len(
            receipts
        ):
            raise CorpusMaterializationError("D6 training seeds did not change order")
        return tuple(receipts)

    def _tokenizer_fit_input_receipt(
        self,
        spool: _Spool,
        shard_rows: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Bind the only tokenizer-admissible input: T in manifest/JSONL order."""

        ordered_t_paths = tuple(
            str(row["relative_path"])
            for row in shard_rows
            if row["stream"] == "T"
        )
        if not ordered_t_paths or ordered_t_paths != tuple(sorted(ordered_t_paths)):
            raise CorpusMaterializationError(
                "tokenizer T shards are absent or noncanonical in the manifest"
            )
        text_digest = hashlib.sha256()
        id_digest = hashlib.sha256()
        document_count = 0
        retained_bytes = 0
        # Shard rows are path-sorted, hence stratum names are lexical here.
        # Each stratum's JSONL record order is split stream_ordinal.
        for stratum in sorted(GTOK_STRATA):
            for document in spool.iter_split_documents("T", stratum):
                text = document.retained_bytes
                document_id = document.document_id.encode("ascii")
                text_digest.update(len(text).to_bytes(8, "big"))
                text_digest.update(text)
                id_digest.update(len(document_id).to_bytes(8, "big"))
                id_digest.update(document_id)
                document_count += 1
                retained_bytes += len(text)
        receipt = {
            "allowed_stream": "T",
            "document_count": document_count,
            "fit_text_stream_sha256": text_digest.hexdigest(),
            "heldout_admissible": False,
            "ordered_document_ids_sha256": id_digest.hexdigest(),
            "ordered_shard_paths": ordered_t_paths,
            "ordering": "SHARD_MANIFEST_THEN_JSONL_RECORD_ORDER",
            "retained_text_bytes": retained_bytes,
            "schema": "weft1_gtok_tokenizer_fit_input_v3",
        }
        receipt["receipt_sha256"] = execution_authority_v3_bound_sha256(
            "weft1_gtok_tokenizer_fit_input_receipt_v3", receipt
        )
        return receipt

    def _build_recall_audit(self, spool: _Spool, dedup: object) -> MinHashRecallAuditV3:
        from training.weft1_corpus_streaming_a2 import StreamingDedupStoreV3

        if not isinstance(dedup, StreamingDedupStoreV3):
            raise TypeError("recall audit requires the streaming dedup store")
        dolma = spool.recall_sample("dolma_web")
        fineweb = spool.recall_sample("fineweb_edu")
        if not dolma or not fineweb:
            raise CorpusMaterializationError(
                "real MinHash recall audit requires both source samples"
            )
        sample_identity = execution_authority_v3_bound_sha256(
            "weft1_corpus_real_minhash_sample_v3",
            {
                "dedup_seed": A2_DEDUP_SEED,
                "dolma": tuple(
                    (document.document_id, document.retained_sha256)
                    for document in dolma
                ),
                "fineweb": tuple(
                    (document.document_id, document.retained_sha256)
                    for document in fineweb
                ),
                "population_counts": tuple(
                    sorted(self.recall_population_counts.items())
                ),
                "sample_limit_per_source": REAL_RECALL_SAMPLE_PER_SOURCE,
            },
        )
        pair_ordinal = 0
        for query in fineweb:
            for canonical in dolma:
                dedup.append_recall_pair(
                    sample_identity_sha256=sample_identity,
                    pair_ordinal=pair_ordinal,
                    query_document=query,
                    canonical_document_id=canonical.document_id,
                )
                pair_ordinal += 1
        accounting = dedup.recall_accounting()
        if accounting.pair_count != len(dolma) * len(fineweb):
            raise CorpusMaterializationError(
                "real MinHash recall pair accounting is incomplete"
            )
        return MinHashRecallAuditV3(
            seed=A2_DEDUP_SEED,
            synthetic_cells=synthetic_recall_cells_v3(),
            real_sample_identity_sha256=sample_identity,
            real_dolma_document_count=len(dolma),
            real_fineweb_document_count=len(fineweb),
            real_exact_pairs_at_or_above_threshold=(
                accounting.exact_pairs_at_or_above_threshold
            ),
            real_candidate_pairs_at_or_above_threshold=(
                accounting.candidate_pairs_at_or_above_threshold
            ),
        )

    def _d5_round_trip(self, shard_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        cases: list[dict[str, object]] = []
        for row in shard_rows:
            relative = str(row["relative_path"])
            path = self.output_root.joinpath(*relative.split("/"))
            observed_zstd_sha256 = sha256_file(path)
            if (
                observed_zstd_sha256 != row["zstd_sha256"]
                or path.stat().st_size != row["zstd_bytes"]
            ):
                raise CorpusMaterializationError("D5 zstd transport identity drifted")
            logical_sha256 = hashlib.sha256()
            logical_bytes = 0
            retained = 0
            record_count = 0
            with path.open("rb") as compressed_handle:
                with zstandard.ZstdDecompressor().stream_reader(
                    compressed_handle
                ) as reader:
                    for line in io.BufferedReader(reader):
                        logical_sha256.update(line)
                        logical_bytes += len(line)
                        if not line.endswith(b"\n"):
                            raise CorpusMaterializationError("D5 shard line lacks LF framing")
                        try:
                            payload = json.loads(line[:-1].decode("utf-8", errors="strict"))
                        except (UnicodeError, json.JSONDecodeError) as error:
                            raise CorpusMaterializationError("D5 shard JSON is invalid") from error
                        if not isinstance(payload, dict) or tuple(payload) != (
                            "id",
                            "source",
                            "stratum",
                            "text",
                        ):
                            raise CorpusMaterializationError("D5 shard record keys drifted")
                        document = StableDocumentV3(
                            source=payload["source"],
                            stratum=payload["stratum"],
                            stable_source_record_id=hashlib.sha256(
                                b"D5-reconstruction\x00" + str(record_count).encode() + line
                            ).hexdigest(),
                            text=payload["text"],
                        )
                        if document.shard_record_id != payload["id"]:
                            raise CorpusMaterializationError("D5 shard record ID drifted")
                        expected = {
                            "id": payload["id"],
                            "source": payload["source"],
                            "stratum": payload["stratum"],
                            "text": payload["text"],
                        }
                        canonical = (
                            json.dumps(
                                expected,
                                ensure_ascii=False,
                                allow_nan=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                            + b"\n"
                        )
                        if canonical != line:
                            raise CorpusMaterializationError("D5 shard JSON is noncanonical")
                        retained += document.retained_byte_count
                        record_count += 1
            if logical_sha256.hexdigest() != row["logical_jsonl_sha256"]:
                raise CorpusMaterializationError("D5 logical shard hash mismatch")
            if logical_bytes != row["logical_jsonl_bytes"]:
                raise CorpusMaterializationError("D5 logical shard byte mismatch")
            if record_count != row["record_count"] or retained != row["retained_text_bytes"]:
                raise CorpusMaterializationError("D5 shard record accounting drifted")
            cases.append(
                {
                    "logical_jsonl_sha256": row["logical_jsonl_sha256"],
                    "record_count": record_count,
                    "relative_path": relative,
                    "retained_text_bytes": retained,
                    "zstd_sha256": observed_zstd_sha256,
                }
            )
        return {
            "gate": "D5",
            "status": "CHECK_PASS_NO_GATE_MINT",
            "cases": tuple(cases),
        }

    def run(self) -> MaterializationResultV3:
        self._prepare_roots()
        self._prepare_production_sources()
        spool = _Spool(self.work_root / "materialization.sqlite")
        dedup_path = self.work_root / "cross-source-dedup.sqlite"
        from training.weft1_corpus_streaming_a2 import StreamingDedupStoreV3

        pool_receipts: list[dict[str, object]] = []
        fineweb_metrics: dict[str, int] = {}
        recall_audit: MinHashRecallAuditV3 | None = None
        try:
            with StreamingDedupStoreV3(dedup_path) as dedup:
                # Dolma is selected and indexed before any FineWeb query exists.
                dolma_target = self.full_targets["dolma_web"]
                dolma_remaining = dolma_target
                canonical_ordinal = 0
                considered = 0
                oversized = 0
                dolma_documents = iter(self._eligible_documents("dolma_web", spool))
                while not _within_tolerance(dolma_remaining, dolma_target):
                    try:
                        document = next(dolma_documents)
                    except StopIteration:
                        break
                    considered += 1
                    before = dolma_remaining
                    owner = spool.selected_content_owner(document)
                    if owner is not None:
                        self.global_exact_duplicates_by_source["dolma_web"] += 1
                        action = "DROP_GLOBAL_EXACT_DUPLICATE"
                    elif not normalize_match_text(document.text):
                        decision = dedup.append_canonical(
                            document, source_order_ordinal=canonical_ordinal
                        )
                        canonical_ordinal += 1
                        self.empty_normalization_by_source["dolma_web"] += 1
                        action = decision.action
                    elif document.retained_byte_count > dolma_remaining:
                        oversized += 1
                        action = "SKIP_OVERSIZED_REMAINING_CAPACITY"
                    else:
                        decision = dedup.append_canonical(
                            document, source_order_ordinal=canonical_ordinal
                        )
                        canonical_ordinal += 1
                        if decision.action != "KEEP_CANONICAL":
                            raise CorpusMaterializationError(
                                "nonempty Dolma document was not canonicalized"
                            )
                        spool.select(document)
                        spool.consider_recall_sample(document)
                        self.recall_population_counts["dolma_web"] += 1
                        dolma_remaining -= document.retained_byte_count
                        action = "SELECT_FULL_CORPUS_AND_CANONICALIZE"
                    spool.event(
                        {
                            "action": action,
                            "document_id": document.document_id,
                            "pool": "dolma_web",
                            "remaining_after": dolma_remaining,
                            "remaining_before": before,
                            "retained_byte_count": document.retained_byte_count,
                            "source": "dolma_web",
                        }
                    )
                if not _within_tolerance(dolma_remaining, dolma_target):
                    raise CorpusMaterializationError(
                        "Dolma exhausted above the 0.5% tolerance"
                    )
                dedup.seal_canonical_phase(expected_source_count=canonical_ordinal)
                pool_receipts.append(
                    {
                        "considered_documents": considered,
                        "deficit_bytes": dolma_remaining,
                        "oversized_skips": oversized,
                        "pool": "dolma_web",
                        "realized_bytes": dolma_target - dolma_remaining,
                        "target_bytes": dolma_target,
                    }
                )

                for source, pool in (
                    ("wikipedia_wikibooks", "wikipedia_wikibooks"),
                    ("stackedu", "stackedu"),
                    ("finemath_3plus", "finemath_3plus"),
                ):
                    pool_receipts.append(
                        self._select_simple_pool(
                            source=source,
                            pool=pool,
                            target=self.full_targets[pool],
                            spool=spool,
                        )
                    )
                pool_receipts.append(self._select_science(spool))

                # FineWeb first fills its nominal pool, then every top-up record
                # is classified and re-deduplicated against the sealed Dolma index.
                fineweb_target = self.full_targets["fineweb_edu"]
                initial_remaining = fineweb_target
                final_remaining = fineweb_target
                query_ordinal = 0
                dedup_dropped_bytes = 0
                topup_selected_bytes = 0
                considered = 0
                fineweb_documents = iter(
                    self._eligible_documents("fineweb_edu", spool)
                )
                while not (
                    _within_tolerance(initial_remaining, fineweb_target)
                    and _within_tolerance(final_remaining, fineweb_target)
                ):
                    try:
                        document = next(fineweb_documents)
                    except StopIteration:
                        break
                    considered += 1
                    phase = (
                        "INITIAL"
                        if not _within_tolerance(initial_remaining, fineweb_target)
                        else "TOPUP"
                    )
                    if phase == "INITIAL" and (
                        document.retained_byte_count > initial_remaining
                    ):
                        spool.event(
                            {
                                "action": "SKIP_INITIAL_OVERSIZED_REMAINING_CAPACITY",
                                "document_id": document.document_id,
                                "pool": "fineweb_edu",
                                "remaining_after": initial_remaining,
                                "remaining_before": initial_remaining,
                                "retained_byte_count": document.retained_byte_count,
                                "source": "fineweb_edu",
                            }
                        )
                        continue
                    if phase == "INITIAL":
                        initial_remaining -= document.retained_byte_count
                    owner = spool.selected_content_owner(document)
                    if owner is not None and owner != "dolma_web":
                        self.global_exact_duplicates_by_source["fineweb_edu"] += 1
                        spool.event(
                            {
                                "action": "DROP_GLOBAL_EXACT_DUPLICATE",
                                "document_id": document.document_id,
                                "duplicate_owner_source": owner,
                                "phase": phase,
                                "pool": "fineweb_edu",
                                "retained_byte_count": document.retained_byte_count,
                                "source": "fineweb_edu",
                            }
                        )
                        continue
                    decision = dedup.append_query(
                        document, source_order_ordinal=query_ordinal
                    )
                    query_ordinal += 1
                    if decision.action != "DROP_EMPTY":
                        spool.consider_recall_sample(document)
                        self.recall_population_counts["fineweb_edu"] += 1
                    if decision.action == "DROP_EMPTY":
                        self.empty_normalization_by_source["fineweb_edu"] += 1
                    if decision.action in {"DROP_EXACT", "DROP_NEAR"}:
                        dedup_dropped_bytes += document.retained_byte_count
                        action = decision.action
                    elif decision.action == "DROP_EMPTY":
                        action = decision.action
                    elif decision.action == "KEEP_FINEWEB":
                        if document.retained_byte_count > final_remaining:
                            action = "SKIP_FINAL_OVERSIZED_REMAINING_CAPACITY"
                        else:
                            spool.select(document)
                            final_remaining -= document.retained_byte_count
                            if phase == "TOPUP":
                                topup_selected_bytes += document.retained_byte_count
                            action = (
                                "SELECT_FINEWEB_INITIAL"
                                if phase == "INITIAL"
                                else "SELECT_FINEWEB_TOPUP"
                            )
                    else:  # pragma: no cover - typed decision closes this branch
                        raise CorpusMaterializationError("unknown dedup decision")
                    spool.event(
                        {
                            "action": action,
                            "dedup_action": decision.action,
                            "document_id": document.document_id,
                            "phase": phase,
                            "pool": "fineweb_edu",
                            "retained_byte_count": document.retained_byte_count,
                            "source": "fineweb_edu",
                        }
                    )
                if not _within_tolerance(initial_remaining, fineweb_target):
                    raise CorpusMaterializationError(
                        "FineWeb initial fill exhausted above tolerance"
                    )
                if not _within_tolerance(final_remaining, fineweb_target):
                    raise CorpusMaterializationError(
                        "FineWeb top-up exhausted above tolerance"
                    )
                dedup.seal_query_phase(expected_source_count=query_ordinal)
                recall_audit = self._build_recall_audit(spool, dedup)
                fineweb_metrics = {
                    "considered_documents": considered,
                    "dedup_dropped_bytes": dedup_dropped_bytes,
                    "deficit_bytes": final_remaining,
                    "initial_deficit_bytes": initial_remaining,
                    "query_decision_count": query_ordinal,
                    "realized_bytes": fineweb_target - final_remaining,
                    "target_bytes": fineweb_target,
                    "topup_selected_bytes": topup_selected_bytes,
                }
                pool_receipts.append({"pool": "fineweb_edu", **fineweb_metrics})

                # Freeze the actual registered LSH/Jaccard connected components
                # before allocating T/H.  H selection excludes every component
                # already represented in T.
                near_cluster_receipt = spool.finalize_near_clusters()

                artifact_root = self.output_root / "artifacts"
                artifact_root.mkdir()
                _write_canonical_json(
                    artifact_root / "minhash-recall-audit.json",
                    recall_audit,
                )
                dedup_ledger = artifact_root / "dedup-decisions.jsonl"
                with dedup_ledger.open("xb") as handle:
                    for payload in dedup.iter_decision_jsonl_bytes():
                        handle.write(payload)
                dedup_ledger_sha256 = sha256_file(dedup_ledger)
                if dedup_ledger_sha256 != dedup.decision_ledger_sha256():
                    raise CorpusMaterializationError(
                        "exported dedup ledger differs from the streaming store"
                    )
                dedup_counts = dedup.decision_counts()

            split_rows = tuple(
                self._split_one_stratum(spool, stratum=stratum)
                for stratum in GTOK_STRATA
            )
            spool.flush()

            selection_ledger_sha256 = spool.export_blob_rows(
                "SELECT payload FROM selection_events ORDER BY event_ordinal",
                self.output_root / "artifacts" / "selection-decisions.jsonl",
            )
            language_ledger_sha256 = spool.export_blob_rows(
                "SELECT payload FROM language_events ORDER BY event_ordinal",
                self.output_root / "artifacts" / "language-decisions.jsonl",
            )
            split_ledger_path = self.output_root / "artifacts" / "split-decisions.jsonl"
            split_digest = hashlib.sha256()
            with split_ledger_path.open("xb") as handle:
                for row in spool.connection.execute(
                    "SELECT stream, stratum, document_id, disposition, retained_bytes, "
                    "remaining_before, remaining_after FROM split_decisions "
                    "ORDER BY decision_ordinal"
                ):
                    payload = canonical_json_bytes(dict(row)) + b"\n"
                    handle.write(payload)
                    split_digest.update(payload)
            split_ledger_sha256 = split_digest.hexdigest()

            if recall_audit is None:
                raise CorpusMaterializationError("D2 recall audit was not produced")
            decision_count = sum(dedup_counts.values())
            exact_match_rate = _fraction_payload(
                dedup_counts["DROP_EXACT"], query_ordinal
            )
            near_match_rate = _fraction_payload(
                dedup_counts["DROP_NEAR"], query_ordinal
            )
            recall_path = self.output_root / "artifacts" / "minhash-recall-audit.json"
            recall_sha256 = sha256_file(recall_path)
            dedup_ledger_identity_sha256 = framed_jsonl_identity_sha256_v3(
                dedup_ledger,
                domain=DEDUP_LEDGER_IDENTITY_DOMAIN_V3,
            )
            semantic_projection = {
                "binding_identity_sha256": A2_MINHASH_BINDING.receipt_sha256,
                "decision_count": decision_count,
                "decision_ledger_path": "artifacts/dedup-decisions.jsonl",
                "decision_ledger_sha256": dedup_ledger_sha256,
                "dropped_bytes": fineweb_metrics["dedup_dropped_bytes"],
                "exact_match_rate": exact_match_rate,
                "near_match_rate": near_match_rate,
                "minhash_recall_audit_path": "artifacts/minhash-recall-audit.json",
                "minhash_recall_audit_receipt_sha256": recall_audit.receipt_sha256,
                "minhash_recall_audit_sha256": recall_sha256,
                "selection_ledger_path": "artifacts/selection-decisions.jsonl",
                "selection_ledger_sha256": selection_ledger_sha256,
                "topup_bytes": fineweb_metrics["topup_selected_bytes"],
            }
            parent_replay_metadata = {
                "binding_identity_sha256": A2_MINHASH_BINDING.receipt_sha256,
                "decision_count": decision_count,
                "decision_ledger_identity_sha256": (
                    dedup_ledger_identity_sha256
                ),
                "decision_ledger_path": "artifacts/dedup-decisions.jsonl",
                "decision_ledger_sha256": dedup_ledger_sha256,
                "dropped_bytes": fineweb_metrics["dedup_dropped_bytes"],
                "exact_match_rate": exact_match_rate,
                "minhash_recall_audit_path": (
                    "artifacts/minhash-recall-audit.json"
                ),
                "minhash_recall_audit_receipt_sha256": (
                    recall_audit.receipt_sha256
                ),
                "minhash_recall_audit_sha256": recall_sha256,
                "near_match_rate": near_match_rate,
                "schema": "weft1_corpus_parent_dedup_evidence_v3",
                "selection_ledger_path": "artifacts/selection-decisions.jsonl",
                "selection_ledger_sha256": selection_ledger_sha256,
                "topup_bytes": fineweb_metrics["topup_selected_bytes"],
            }
            d2_descriptor = {
                "gate_minted": False,
                "parent_replay_metadata": parent_replay_metadata,
                "report_only": True,
                "schema": "weft1_corpus_d2_evidence_descriptor_v3",
                "semantic_projection": semantic_projection,
                "semantic_projection_identity_sha256": (
                    execution_authority_v3_bound_sha256(
                        "weft1_corpus_d2_semantic_projection_v3",
                        semantic_projection,
                    )
                ),
            }
            d2_descriptor_path = (
                self.output_root / "artifacts" / "d2-evidence-descriptor.json"
            )
            _write_canonical_json(d2_descriptor_path, d2_descriptor)
            d2_descriptor_sha256 = sha256_file(d2_descriptor_path)

            shard_rows = self._write_shards(spool)
            tokenizer_fit_input = self._tokenizer_fit_input_receipt(
                spool, shard_rows
            )
            _write_canonical_json(
                self.output_root / "artifacts" / "tokenizer-fit-input.json",
                tokenizer_fit_input,
            )
            _write_canonical_json(
                self.output_root / "artifacts" / "shard-manifest.json",
                {
                    "codec_binding_sha256": A2_ZSTD_CODEC_BINDING.receipt_sha256,
                    "schema": "weft1_corpus_shard_manifest_v3",
                    "shards": shard_rows,
                    "tokenizer_fit_input_receipt_sha256": (
                        tokenizer_fit_input["receipt_sha256"]
                    ),
                },
            )

            composition = {
                row["source"]: int(row["retained"])
                for row in spool.connection.execute(
                    "SELECT source, sum(retained_bytes) AS retained FROM "
                    "selected_documents GROUP BY source ORDER BY source"
                )
            }
            strata = {
                stratum: int(
                    spool.connection.execute(
                        "SELECT coalesce(sum(retained_bytes), 0) FROM "
                        "selected_documents WHERE stratum = ?", (stratum,)
                    ).fetchone()[0]
                )
                for stratum in GTOK_STRATA
            }
            observed_pools = {
                "wikipedia_wikibooks": composition.get("wikipedia_wikibooks", 0),
                "dolma_web": composition.get("dolma_web", 0),
                "fineweb_edu": composition.get("fineweb_edu", 0),
                "stackedu": composition.get("stackedu", 0),
                "finemath_3plus": composition.get("finemath_3plus", 0),
                "science_technical_combined": composition.get("arxiv", 0)
                + composition.get("olmocr", 0),
            }
            d3_rows = tuple(
                {
                    "deficit_fraction": _fraction_payload(
                        self.full_targets[pool] - observed_pools[pool],
                        self.full_targets[pool],
                    ),
                    "observed_bytes": observed_pools[pool],
                    "pool": pool,
                    "target_bytes": self.full_targets[pool],
                }
                for pool in FULL_POOL_ORDER
            )
            if any(
                Fraction(
                    self.full_targets[row["pool"]] - row["observed_bytes"],
                    self.full_targets[row["pool"]],
                )
                > FIRST_FIT_TOLERANCE
                for row in d3_rows
            ):
                raise CorpusMaterializationError("D3 composition exceeds tolerance")
            d3 = {
                "gate": "D3",
                "status": "CHECK_PASS_NO_GATE_MINT",
                "full_pool_rows": d3_rows,
                "observed_stratum_bytes": tuple(
                    (stratum, strata[stratum]) for stratum in GTOK_STRATA
                ),
                "pool_receipts": tuple(
                    sorted(
                        pool_receipts,
                        key=lambda receipt: FULL_POOL_ORDER.index(
                            str(receipt["pool"])
                        ),
                    )
                ),
            }
            d4 = {
                "gate": "D4",
                "status": "CHECK_PASS_NO_GATE_MINT",
                "invocation_counts": tuple(
                    (stratum, self.language_invocations[stratum])
                    for stratum in GTOK_STRATA
                ),
                "rejection_counts": tuple(
                    (stratum, self.language_rejections[stratum])
                    for stratum in GTOK_STRATA
                ),
            }
            if any(
                self.language_invocations[stratum]
                or self.language_rejections[stratum]
                for stratum in GTOK_STRATA
                if stratum != "general"
            ):
                raise CorpusMaterializationError("D4 language scope escaped general")
            d5 = self._d5_round_trip(shard_rows)

            overlaps = spool.connection.execute(
                "SELECT count(*) FROM split_documents AS t JOIN split_documents AS h "
                "ON t.document_id = h.document_id WHERE t.stream = 'T' "
                "AND h.stream = 'H'"
            ).fetchone()[0]
            if overlaps:
                raise CorpusMaterializationError("D6 split documents overlap")
            cluster_overlaps = spool.connection.execute(
                "SELECT count(*) FROM split_documents AS t "
                "JOIN selected_documents AS td ON td.document_id = t.document_id "
                "JOIN split_documents AS h ON h.stream = 'H' "
                "JOIN selected_documents AS hd ON hd.document_id = h.document_id "
                "AND hd.cluster_id = td.cluster_id WHERE t.stream = 'T'"
            ).fetchone()[0]
            if cluster_overlaps:
                raise CorpusMaterializationError("D6 split clusters overlap")
            repeated_full_raw_ids = spool.connection.execute(
                "SELECT count(*) FROM (SELECT raw_content_id FROM "
                "selected_documents GROUP BY raw_content_id HAVING count(*) > 1)"
            ).fetchone()[0]
            repeated_split_raw_ids = spool.connection.execute(
                "SELECT count(*) FROM (SELECT d.raw_content_id FROM split_documents AS s "
                "JOIN selected_documents AS d ON d.document_id = s.document_id "
                "GROUP BY d.raw_content_id HAVING count(*) > 1)"
            ).fetchone()[0]
            if repeated_full_raw_ids or repeated_split_raw_ids:
                raise CorpusMaterializationError("D6 found a repeated raw-content ID")
            stream_identities: list[dict[str, object]] = []
            for stream in ("T", "H"):
                digest = hashlib.sha256()
                byte_count = 0
                document_count = 0
                for stratum in GTOK_STRATA:
                    for document in spool.iter_split_documents(stream, stratum):
                        payload = document.retained_bytes
                        digest.update(len(payload).to_bytes(8, "big"))
                        digest.update(payload)
                        byte_count += len(payload)
                        document_count += 1
                stream_identities.append(
                    {
                        "document_count": document_count,
                        "framed_retained_text_sha256": digest.hexdigest(),
                        "retained_text_bytes": byte_count,
                        "stream": stream,
                    }
                )
            consumer_order_receipts = self._consumer_order_receipts(spool)
            order_by_seed = {
                int(row["training_seed"]): row for row in consumer_order_receipts
            }
            heldout_identity = next(
                row["framed_retained_text_sha256"]
                for row in stream_identities
                if row["stream"] == "H"
            )
            consumer_bindings = tuple(
                {
                    "heldout_framed_retained_text_sha256": heldout_identity,
                    "training_document_multiset_sha256": order_by_seed[
                        training_seed
                    ]["document_multiset_sha256"],
                    "training_order_receipt_sha256": order_by_seed[
                        training_seed
                    ]["receipt_sha256"],
                    "training_ordered_document_ids_sha256": order_by_seed[
                        training_seed
                    ]["ordered_document_ids_sha256"],
                    "training_seed": training_seed,
                    "vocabulary_size": vocabulary_size,
                }
                for vocabulary_size in GTOK_VOCABULARY_ARMS
                for training_seed in GTOK_TRAINING_SEEDS
            )
            d6 = {
                "gate": "D6",
                "status": "CHECK_PASS_NO_GATE_MINT",
                "consumer_bindings": consumer_bindings,
                "consumer_order_receipts": consumer_order_receipts,
                "document_overlap_count": 0,
                "full_corpus_repeated_raw_content_id_count": int(
                    repeated_full_raw_ids
                ),
                "cluster_overlap_count": int(cluster_overlaps),
                "near_cluster_receipt": near_cluster_receipt,
                "screen_repeated_raw_content_id_count": int(repeated_split_raw_ids),
                "split_rows": split_rows,
                "stream_identities": tuple(stream_identities),
                "tokenizer_fit_contract": {
                    "allowed_stream": "T_ONLY",
                    "fit_input_receipt_sha256": (
                        tokenizer_fit_input["receipt_sha256"]
                    ),
                    "fit_text_stream_sha256": (
                        tokenizer_fit_input["fit_text_stream_sha256"]
                    ),
                    "heldout_admissible": False,
                    "input_order": "CANONICAL_T_SHARD_MANIFEST_ORDER",
                },
            }

            diagnostics = {"d3": d3, "d4": d4, "d5": d5, "d6": d6}
            for name, diagnostic in diagnostics.items():
                _write_canonical_json(
                    self.output_root / "diagnostics" / f"{name}.json", diagnostic
                )

            algorithm_identity = execution_authority_v3_bound_sha256(
                "weft1_corpus_materializer_algorithm_v3",
                {
                    "algorithm_version": MATERIALIZER_ALGORITHM_VERSION,
                    "language_id_binding_sha256": A2_LANGUAGE_ID_BINDING.receipt_sha256,
                    "match_normalization_binding_sha256": (
                        A2_MATCH_NORMALIZATION_BINDING.receipt_sha256
                    ),
                    "minhash_binding_sha256": A2_MINHASH_BINDING.receipt_sha256,
                    "plan_sha256": self.plan.identity_sha256,
                    "screen_order_domain": SCREEN_ORDER_DOMAIN.decode("ascii"),
                    "zstd_binding_sha256": A2_ZSTD_CODEC_BINDING.receipt_sha256,
                },
            )
            content_manifest = {
                "algorithm_identity_sha256": algorithm_identity,
                "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
                "authoritative_gate_receipts": (),
                "dedup_counts": tuple(sorted(dedup_counts.items())),
                "dedup_decision_ledger_identity_sha256": (
                    dedup_ledger_identity_sha256
                ),
                "dedup_ledger_sha256": dedup_ledger_sha256,
                "d2_evidence_descriptor_sha256": d2_descriptor_sha256,
                "minhash_recall_audit_receipt_sha256": (
                    recall_audit.receipt_sha256
                ),
                "minhash_recall_audit_sha256": recall_sha256,
                "diagnostic_sha256s": tuple(
                    (
                        name.upper(),
                        sha256_file(self.output_root / "diagnostics" / f"{name}.json"),
                    )
                    for name in diagnostics
                ),
                "empty_normalization_by_source": tuple(
                    (source, self.empty_normalization_by_source[source])
                    for source in SOURCE_FAMILIES
                ),
                "fineweb_topup": fineweb_metrics,
                "global_exact_duplicate_drops_by_source": tuple(
                    (source, self.global_exact_duplicates_by_source[source])
                    for source in SOURCE_FAMILIES
                ),
                "invalid_utf8_by_source": tuple(
                    (source, self.invalid_utf8_by_source[source])
                    for source in SOURCE_FAMILIES
                ),
                "language_ledger_sha256": language_ledger_sha256,
                "mode": self.plan.mode,
                "readiness": (
                    "AUTHORITATIVE_INPUTS_D1_READY_NO_GATE_MINT"
                    if self.plan.mode == PRODUCTION_MODE
                    else "NONAUTHORITATIVE_FIXTURE_D1_SHAPE_ONLY"
                ),
                "schema": MATERIALIZER_SCHEMA,
                "selection_ledger_sha256": selection_ledger_sha256,
                "source_parse_drop_counts": tuple(
                    {
                        "counts": self.source_parse_drop_counts[source],
                        **self.source_parse_receipts[source],
                        "source": source,
                    }
                    for source in SOURCE_FAMILIES
                ),
                "shard_manifest_sha256": sha256_file(
                    self.output_root / "artifacts" / "shard-manifest.json"
                ),
                "tokenizer_fit_input_receipt_sha256": (
                    tokenizer_fit_input["receipt_sha256"]
                ),
                "tokenizer_fit_input_sha256": sha256_file(
                    self.output_root / "artifacts" / "tokenizer-fit-input.json"
                ),
                "source_identity_sha256": self.materialized_source_identity_sha256,
                "transport_authority": (
                    None
                    if self.inputs.mode != PRODUCTION_MODE
                    else {
                        "cache_download_receipt_sha256": (
                            self.inputs.source_cache_download_receipt.receipt_sha256
                        ),
                        "cache_verification_receipt_sha256": (
                            self.inputs.verified_cache.verification_receipt_sha256
                        ),
                        "selection_plan_sha256": (
                            self.inputs.source_cache_download_receipt.selection_plan_sha256
                        ),
                        "upstream_enumeration_receipt_sha256": (
                            self.inputs.upstream_enumeration.receipt_sha256
                        ),
                    }
                ),
                "split_ledger_sha256": split_ledger_sha256,
            }
            content_identity_sha256 = execution_authority_v3_bound_sha256(
                "weft1_corpus_materialized_content_v3", content_manifest
            )
            content_manifest["content_identity_sha256"] = content_identity_sha256
            _write_canonical_json(
                self.output_root / "content-manifest.json", content_manifest
            )

            inventory = tuple(
                {
                    "bytes": path.stat().st_size,
                    "relative_path": path.relative_to(self.output_root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in sorted(
                    (
                        path
                        for path in self.output_root.rglob("*")
                        if path.is_file()
                        and path.name not in {"d1-ready-manifest.json", "_INCOMPLETE"}
                    ),
                    key=lambda path: path.relative_to(self.output_root).as_posix(),
                )
            )
            d1_payload = {
                "content_identity_sha256": content_identity_sha256,
                "file_inventory": inventory,
                "gate_minted": False,
                "mode": self.plan.mode,
                "readiness": content_manifest["readiness"],
                "schema": "weft1_corpus_d1_ready_manifest_v3",
                "source_identity_sha256": self.materialized_source_identity_sha256,
            }
            d1_identity = execution_authority_v3_bound_sha256(
                "weft1_corpus_d1_ready_inventory_v3", d1_payload
            )
            d1_payload["d1_ready_identity_sha256"] = d1_identity
            d1_path = self.output_root / "d1-ready-manifest.json"
            _write_canonical_json(d1_path, d1_payload)
            (self.output_root / "_INCOMPLETE").unlink()
            return MaterializationResultV3(
                mode=self.plan.mode,
                source_identity_sha256=self.materialized_source_identity_sha256,
                content_identity_sha256=content_identity_sha256,
                d1_ready_manifest_sha256=sha256_file(d1_path),
                output_root=self.output_root,
                work_root=self.work_root,
            )
        finally:
            spool.close()
            if self._production_source_db is not None:
                self._production_source_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._production_source_db.close()


def iter_materialized_tokenizer_fit_texts_v3(
    materialization_root: Path,
) -> Iterator[str]:
    """Yield only the receipt-bound T stream in canonical manifest order.

    This is the production tokenizer entry boundary.  It deliberately does
    not accept an arbitrary shard sequence, and it rejects H even if an H shard
    is otherwise byte-valid.
    """

    if not isinstance(materialization_root, Path):
        raise TypeError("materialization root must be a pathlib.Path")
    lexical_root = assert_no_symlink_ancestors(materialization_root)
    root = lexical_root.resolve(strict=True)
    if not root.is_dir():
        raise CorpusMaterializationError("materialization root is not a directory")
    manifest_path = root / "artifacts" / "shard-manifest.json"
    fit_path = root / "artifacts" / "tokenizer-fit-input.json"
    manifest = load_canonical_json_object(manifest_path)
    fit = load_canonical_json_object(fit_path)
    if set(manifest) != {
        "codec_binding_sha256",
        "schema",
        "shards",
        "tokenizer_fit_input_receipt_sha256",
    } or manifest.get("schema") != "weft1_corpus_shard_manifest_v3":
        raise CorpusMaterializationError("tokenizer shard manifest schema drifted")
    if set(fit) != {
        "allowed_stream",
        "document_count",
        "fit_text_stream_sha256",
        "heldout_admissible",
        "ordered_document_ids_sha256",
        "ordered_shard_paths",
        "ordering",
        "receipt_sha256",
        "retained_text_bytes",
        "schema",
    }:
        raise CorpusMaterializationError("tokenizer fit-input receipt fields drifted")
    fit_core = dict(fit)
    claimed_receipt = fit_core.pop("receipt_sha256")
    observed_receipt = execution_authority_v3_bound_sha256(
        "weft1_gtok_tokenizer_fit_input_receipt_v3", fit_core
    )
    if (
        claimed_receipt != observed_receipt
        or manifest.get("tokenizer_fit_input_receipt_sha256") != observed_receipt
        or fit.get("allowed_stream") != "T"
        or fit.get("heldout_admissible") is not False
        or fit.get("ordering") != "SHARD_MANIFEST_THEN_JSONL_RECORD_ORDER"
        or fit.get("schema") != "weft1_gtok_tokenizer_fit_input_v3"
    ):
        raise CorpusMaterializationError("tokenizer fit-input authority drifted")
    raw_rows = manifest.get("shards")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise CorpusMaterializationError("tokenizer shard manifest is empty")
    if any(not isinstance(row, Mapping) for row in raw_rows):
        raise CorpusMaterializationError("tokenizer shard manifest has an untyped row")
    rows = tuple(dict(row) for row in raw_rows)
    relative_paths = tuple(str(row.get("relative_path")) for row in rows)
    if relative_paths != tuple(sorted(relative_paths)) or len(relative_paths) != len(
        set(relative_paths)
    ):
        raise CorpusMaterializationError("tokenizer shard manifest order drifted")
    t_rows = tuple(row for row in rows if row.get("stream") == "T")
    if any(row.get("stream") not in {"T", "H"} for row in rows):
        raise CorpusMaterializationError("shard manifest uses an unknown stream")
    ordered_paths = tuple(str(row["relative_path"]) for row in t_rows)
    if list(ordered_paths) != fit.get("ordered_shard_paths"):
        raise CorpusMaterializationError(
            "tokenizer fit receipt does not select canonical T-only shard order"
        )
    identity_keys = {
        "codec_binding_sha256",
        "identity_relative_path",
        "logical_jsonl_bytes",
        "logical_jsonl_sha256",
        "record_count",
        "relative_path",
        "retained_text_bytes",
        "zstd_bytes",
        "zstd_sha256",
    }
    shards: list[JsonlZstdShardIdentityV3] = []
    for row in t_rows:
        if set(row) != identity_keys | {"content_identity_sha256", "stream", "stratum"}:
            raise CorpusMaterializationError("tokenizer shard identity fields drifted")
        shard = JsonlZstdShardIdentityV3(
            **{
                (
                    "relative_path"
                    if key == "identity_relative_path"
                    else key
                ): row[key]
                for key in identity_keys
                if key != "relative_path"
            }
        )
        if shard.content_identity_sha256 != row["content_identity_sha256"]:
            raise CorpusMaterializationError("tokenizer shard content identity drifted")
        assert_no_symlink_ancestors(
            root.joinpath(*PurePosixPath(str(row["relative_path"])).parts)
        )
        shards.append(shard)

    from training.weft1_gtok_tokenizer_a2 import iter_a2_shard_texts

    digest = hashlib.sha256()
    document_count = 0
    retained_bytes = 0
    for text_value in iter_a2_shard_texts(root / "shards", tuple(shards)):
        encoded = text_value.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        document_count += 1
        retained_bytes += len(encoded)
        yield text_value
    if (
        digest.hexdigest() != fit.get("fit_text_stream_sha256")
        or document_count != fit.get("document_count")
        or retained_bytes != fit.get("retained_text_bytes")
    ):
        raise CorpusMaterializationError(
            "tokenizer T-only text stream differs from its materialization receipt"
        )


def materialize_corpus_pa_v3(
    *,
    inputs: MaterializationInputV3,
    plan: MaterializationPlanV3,
    language_classifier: LanguageClassifierV3,
    output_root: Path,
    work_root: Path,
) -> MaterializationResultV3:
    """Materialize P-A offline; never freeze P-B or mint D1-D6/G-TOK gates."""

    if not isinstance(inputs, MaterializationInputV3):
        raise TypeError("inputs must be MaterializationInputV3")
    if not isinstance(plan, MaterializationPlanV3):
        raise TypeError("plan must be MaterializationPlanV3")
    return _Materializer(
        inputs=inputs,
        plan=plan,
        language_classifier=language_classifier,
        output_root=Path(output_root),
        work_root=Path(work_root),
    ).run()


def _probe_parent_network_guard_v3() -> str:
    probe = socket.socket()
    try:
        probe.connect(("127.0.0.1", 9))
    except RuntimeError as error:
        if str(error) != "WEFT-1 parent replay disables network access":
            raise CorpusMaterializationError(
                "worker network probe hit an unregistered runtime guard"
            ) from error
        return NETWORK_PROBE_RESULT
    except OSError as error:
        raise CorpusMaterializationError(
            "worker network probe reached the operating-system socket"
        ) from error
    finally:
        probe.close()
    raise CorpusMaterializationError("worker network probe was not blocked")


def _write_production_replay_child_receipt_v3(
    result: MaterializationResultV3,
    *,
    runtime_environment_identity_sha256: str,
    sentinel: object | None = None,
) -> str:
    """Emit the exact parent-verifier child receipt after production P-A.

    The function is intentionally production-only and environment-bound.  It
    never runs materialization itself, never downloads, and never mints a gate;
    the concrete worker below is the only exported composition of load, parse,
    materialize, and receipt emission.
    """

    if sentinel is not _PRODUCTION_WORKER_RECEIPT_SENTINEL:
        raise PermissionError(
            "production child receipts are emitted only by the concrete worker"
        )
    if not isinstance(result, MaterializationResultV3):
        raise TypeError("child receipt requires a materialization result")
    if result.mode != PRODUCTION_MODE:
        raise CorpusMaterializationError(
            "parent replay child receipts are production-materialization only"
        )
    environment_identity = _require_sha256(
        runtime_environment_identity_sha256,
        "runtime environment identity",
    )
    output_root = assert_no_symlink_ancestors(result.output_root).resolve(strict=True)
    if not output_root.is_dir() or (output_root / "_INCOMPLETE").exists():
        raise CorpusMaterializationError("materialization output is incomplete")
    content = load_canonical_json_object(output_root / "content-manifest.json")
    d1 = load_canonical_json_object(output_root / "d1-ready-manifest.json")
    descriptor = load_canonical_json_object(
        output_root / "artifacts" / "d2-evidence-descriptor.json"
    )
    if (
        content.get("mode") != PRODUCTION_MODE
        or content.get("content_identity_sha256") != result.content_identity_sha256
        or content.get("source_identity_sha256") != result.source_identity_sha256
        or d1.get("mode") != PRODUCTION_MODE
        or descriptor.get("gate_minted") is not False
        or descriptor.get("report_only") is not True
    ):
        raise CorpusMaterializationError(
            "production result and its governed manifests do not compose"
        )
    dedup_metadata = descriptor.get("parent_replay_metadata")
    if not isinstance(dedup_metadata, Mapping):
        raise CorpusMaterializationError("D2 parent metadata is absent")

    assigned_root = os.environ.get("WEFT1_REPLAY_OUTPUT_ROOT")
    receipt_assignment = os.environ.get("WEFT1_REPLAY_RECEIPT_PATH")
    run_id = os.environ.get("WEFT1_REPLAY_RUN_ID")
    if (
        not assigned_root
        or Path(assigned_root).resolve(strict=True) != output_root
        or not receipt_assignment
        or Path(receipt_assignment).resolve(strict=False)
        != output_root / CHILD_RECEIPT_FILENAME
        or not isinstance(run_id, str)
        or not run_id
    ):
        raise CorpusMaterializationError("parent replay assignments differ from output")
    if (
        os.environ.get("WEFT1_NETWORK_DISABLED") != "1"
        or os.environ.get("WEFT1_NETWORK_GUARD_ACTIVE") != "1"
    ):
        raise CorpusMaterializationError(
            "production worker requires the parent-injected offline guard"
        )
    network_probe = _probe_parent_network_guard_v3()
    input_identity = _require_sha256(
        os.environ.get("WEFT1_REPLAY_INPUT_IDENTITY_SHA256", ""),
        "parent replay input identity",
    )
    compatibility_identity = _require_sha256(
        os.environ.get("WEFT1_REPLAY_WORKER_COMPATIBILITY_SHA256", ""),
        "parent replay worker compatibility identity",
    )
    network_guard_sha256 = _require_sha256(
        os.environ.get("WEFT1_NETWORK_GUARD_SHA256", ""),
        "parent replay network guard identity",
    )
    dedup_paths = {
        str(dedup_metadata.get("decision_ledger_path")),
        str(dedup_metadata.get("selection_ledger_path")),
        str(dedup_metadata.get("minhash_recall_audit_path")),
    }
    if len(dedup_paths) != 3:
        raise CorpusMaterializationError("D2 evidence paths are incomplete")
    rows: list[dict[str, object]] = []
    for path in sorted(
        output_root.rglob("*"),
        key=lambda item: item.relative_to(output_root).as_posix(),
    ):
        assert_no_symlink_ancestors(path)
        if path.is_dir():
            continue
        if not path.is_file():
            raise CorpusMaterializationError(
                "production output contains a non-regular artifact"
            )
        relative = path.relative_to(output_root).as_posix()
        if relative == CHILD_RECEIPT_FILENAME:
            raise CorpusMaterializationError("child receipt already exists")
        rows.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "role": "dedup_evidence" if relative in dedup_paths else "content",
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise CorpusMaterializationError("production output inventory is empty")
    receipt = {
        "content_metadata": {
            "content_identity_sha256": result.content_identity_sha256,
            "d1_ready_manifest_sha256": result.d1_ready_manifest_sha256,
            "environment_identity_sha256": environment_identity,
            "materializer_algorithm_version": MATERIALIZER_ALGORITHM_VERSION,
            "source_identity_sha256": result.source_identity_sha256,
            "tokenizer_fit_input_receipt_sha256": content.get(
                "tokenizer_fit_input_receipt_sha256"
            ),
        },
        "dedup_evidence_complete": True,
        "dedup_metadata": dict(dedup_metadata),
        "files": rows,
        "input_identity_sha256": input_identity,
        "network_disabled": True,
        "network_guard_active": True,
        "network_guard_sha256": network_guard_sha256,
        "network_probe": network_probe,
        "output_root": str(output_root),
        "process_id": os.getpid(),
        "run_id": run_id,
        "schema": CHILD_RECEIPT_SCHEMA_V3,
        "worker_compatibility_sha256": compatibility_identity,
    }
    receipt_path = output_root / CHILD_RECEIPT_FILENAME
    payload = canonical_json_bytes(receipt) + b"\n"
    with receipt_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def run_production_materialization_worker_v3(
    *,
    enumeration_receipt_path: Path,
    cache_download_receipt_path: Path,
    source_manifest_path: Path,
    cache_root: Path,
    fasttext_model_path: Path,
    route_manifest_path: Path,
) -> str:
    """Offline production child: load receipts, reparse cache, materialize, report.

    This function has no downloader/open-upstream callback and accepts no source
    stream injection.  Its output location and run identity come only from the
    parent replay environment.
    """

    paths = (
        enumeration_receipt_path,
        cache_download_receipt_path,
        source_manifest_path,
        cache_root,
        fasttext_model_path,
        route_manifest_path,
    )
    if any(not isinstance(path, Path) for path in paths):
        raise TypeError("production worker inputs must be pathlib.Path values")
    for path in paths:
        assert_no_symlink_ancestors(path)
    if (
        os.environ.get("WEFT1_NETWORK_DISABLED") != "1"
        or os.environ.get("WEFT1_NETWORK_GUARD_ACTIVE") != "1"
    ):
        raise CorpusMaterializationError(
            "production materialization worker requires parent offline execution"
        )
    raw_output_root = os.environ.get("WEFT1_REPLAY_OUTPUT_ROOT")
    if not raw_output_root:
        raise CorpusMaterializationError("production worker lacks an assigned root")
    output_root = Path(raw_output_root)
    assert_no_symlink_ancestors(output_root)
    if output_root.exists():
        raise CorpusMaterializationError("production worker root must be fresh")

    # Lazy imports ensure loading the module itself never opens a model, cache,
    # or network resource.  The receipt loaders independently rehash every
    # local cache byte and reconstruct their factory-only typed objects.
    from training.weft1_corpus_enumeration_a2 import (
        load_upstream_enumeration_receipt_v3,
    )
    from training.weft1_corpus_source_io_a2 import (
        load_source_cache_download_receipt_v3,
    )
    from training.weft1_corpus_pa import (
        FastTextLanguageIdAdapterV3,
        attest_runtime_v3,
    )

    # This attestation deliberately precedes every receipt/cache load and model
    # open.  A child running under a different Python, dependency lock, Unicode,
    # SQLite, zstd, locale, or determinism environment may not observe corpus
    # bytes and later claim the parent's execution identity.
    runtime_attestation = attest_runtime_v3()

    enumeration = load_upstream_enumeration_receipt_v3(
        enumeration_receipt_path,
        route_manifest_path=route_manifest_path,
    )
    download_receipt, verified_cache = load_source_cache_download_receipt_v3(
        cache_download_receipt_path,
        enumeration=enumeration,
        source_manifest_path=source_manifest_path,
        cache_root=cache_root,
        route_manifest_path=route_manifest_path,
    )
    inputs = MaterializationInputV3(
        mode=PRODUCTION_MODE,
        upstream_enumeration=enumeration,
        verified_cache=verified_cache,
        source_cache_download_receipt=download_receipt,
        cache_root=cache_root,
    )
    classifier = FastTextLanguageIdAdapterV3(fasttext_model_path)
    output_parent = assert_no_symlink_ancestors(output_root.parent).resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}-work-",
        dir=output_parent,
    ) as raw_temporary:
        temporary = Path(raw_temporary)
        assert_no_symlink_ancestors(temporary)
        result = materialize_corpus_pa_v3(
            inputs=inputs,
            plan=MaterializationPlanV3.production(),
            language_classifier=classifier,
            output_root=output_root,
            work_root=temporary / "spool",
        )
        return _write_production_replay_child_receipt_v3(
            result,
            runtime_environment_identity_sha256=(
                runtime_attestation.environment_identity_sha256
            ),
            sentinel=_PRODUCTION_WORKER_RECEIPT_SENTINEL,
        )


__all__ = [
    "CorpusMaterializationError",
    "FIXTURE_MODE",
    "FULL_POOL_ORDER",
    "InjectedSourceStreamV3",
    "MaterializationInputV3",
    "MaterializationPlanV3",
    "MaterializationResultV3",
    "MaterializerSourceRecordV3",
    "PRODUCTION_FULL_POOL_TARGETS",
    "PRODUCTION_MODE",
    "materialize_corpus_pa_v3",
    "injected_source_stream_from_parsed_records_v3",
    "iter_materialized_tokenizer_fit_texts_v3",
    "run_production_materialization_worker_v3",
    "screen_order_digest_v3",
]
