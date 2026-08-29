"""Disk-backed streaming deduplication for WEFT-1 corpus Amendment A2.

The pure algorithm and receipt contracts live in :mod:`training.weft1_corpus_a2`.
This module is the production-scale state boundary for the cross-source pass:
Dolma-web documents are indexed first, that phase is explicitly sealed, and
FineWeb-Edu documents are then tested against the immutable Dolma index.  The
corpus and decision ledger remain in SQLite rather than being accumulated in
Python containers.

Every state mutation uses ``BEGIN IMMEDIATE``.  Corpus/index/decision/audit
tables are append-only (triggers reject UPDATE and DELETE); only the singleton
cursor table is mutable.  LSH remains candidate generation, never an exhaustive
claim.  Exact set-Jaccard always decides acceptance.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path
import sqlite3

from training.weft1_corpus_a2 import (
    A2_DEDUP_SEED,
    A2_MATCH_NORMALIZATION_BINDING,
    A2_MINHASH_BINDING,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
    NEAR_DUPLICATE_THRESHOLD,
    StableDocumentV3,
    byte_shingles_v3,
    exact_jaccard_v3,
    lsh_band_keys_v3,
    minhash_signature_v3,
    normalized_match_bytes,
    select_dedup_winner_v3,
)
from training.weft1_gtok_contract import canonical_json_bytes


SQLITE_RUNTIME_VERSION = "3.45.1"
STREAMING_SCHEMA = "weft1_corpus_streaming_dedup_v3"
STREAMING_SCHEMA_VERSION = 1
SERIAL_REFERENCE_MAX_DOCUMENTS = 64
RECALL_STATUS = "REPORT_ONLY_NO_RECALL_FLOOR"

_CANONICAL_SOURCE = "dolma_web"
_QUERY_SOURCE = "fineweb_edu"
_PHASE_CANONICAL = "DOLMA_CANONICAL"
_PHASE_QUERY = "FINEWEB_QUERY"
_PHASE_COMPLETE = "COMPLETE"
_ACTIONS = {
    "KEEP_CANONICAL",
    "DROP_EMPTY",
    "DROP_EXACT",
    "DROP_NEAR",
    "KEEP_FINEWEB",
}
_APPEND_ONLY_TABLES = (
    "metadata",
    "phase_events",
    "canonical_documents",
    "lsh_bands",
    "dedup_decisions",
    "recall_observations",
)


class StreamingDedupError(RuntimeError):
    """Fail-closed error at the A2 streaming-state boundary."""


def _require_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_ordinal(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact integer")
    return value


def _match_identity(value: bytes) -> tuple[bytes, int, bytes]:
    """Return the registered exact-candidate key and confirmation fields."""

    return (
        hashlib.sha1(value).digest(),  # noqa: S324 - registered candidate key
        len(value),
        hashlib.sha256(value).digest(),
    )


def _signature_blob(signature: Sequence[int]) -> bytes:
    if len(signature) != 128:
        raise ValueError("A2 MinHash signatures require 128 components")
    return b"".join(value.to_bytes(8, "little", signed=False) for value in signature)


@dataclass(frozen=True)
class StreamingDedupDecisionV3:
    """One append-only decision in deterministic source order."""

    decision_ordinal: int
    source: str
    source_order_ordinal: int
    document_id: str
    stable_source_record_id: str
    action: str
    canonical_document_id: str | None
    canonical_source_record_id: str | None
    exact_jaccard_numerator: int | None
    exact_jaccard_denominator: int | None
    retained_byte_count: int
    normalized_byte_count: int
    lsh_candidate_count: int

    def __post_init__(self) -> None:
        _require_ordinal(self.decision_ordinal, "decision_ordinal")
        _require_ordinal(self.source_order_ordinal, "source_order_ordinal")
        if self.source not in {_CANONICAL_SOURCE, _QUERY_SOURCE}:
            raise ValueError("streaming dedup decision uses an unregistered source")
        _require_sha256(self.document_id, "document_id")
        _require_sha256(self.stable_source_record_id, "stable_source_record_id")
        if self.action not in _ACTIONS:
            raise ValueError("streaming dedup decision uses an unknown action")
        for name in (
            "retained_byte_count",
            "normalized_byte_count",
            "lsh_candidate_count",
        ):
            _require_ordinal(getattr(self, name), name)
        has_winner = self.canonical_document_id is not None
        has_score = self.exact_jaccard_numerator is not None
        if has_winner != (self.canonical_source_record_id is not None):
            raise ValueError("canonical winner identity is incomplete")
        if has_score != (self.exact_jaccard_denominator is not None):
            raise ValueError("exact Jaccard fraction is incomplete")
        if has_winner:
            _require_sha256(self.canonical_document_id or "", "canonical_document_id")
            _require_sha256(
                self.canonical_source_record_id or "",
                "canonical_source_record_id",
            )
        if has_score:
            if self.exact_jaccard_denominator is None or (
                self.exact_jaccard_denominator < 1
            ):
                raise ValueError("exact Jaccard denominator must be positive")
            score = Fraction(
                self.exact_jaccard_numerator,
                self.exact_jaccard_denominator,
            )
            if not Fraction(0, 1) <= score <= Fraction(1, 1):
                raise ValueError("exact Jaccard lies outside [0, 1]")
        if self.action in {"DROP_EXACT", "DROP_NEAR"}:
            if not has_winner or not has_score:
                raise ValueError(
                    "a duplicate drop requires a canonical winner and score"
                )
        elif has_winner or has_score:
            raise ValueError("a non-duplicate decision may not name a winner or score")
        if self.action == "DROP_EXACT" and self.exact_jaccard != Fraction(1, 1):
            raise ValueError("an exact duplicate must have Jaccard 1")
        if self.action == "DROP_NEAR" and (
            self.exact_jaccard is None
            or not NEAR_DUPLICATE_THRESHOLD <= self.exact_jaccard < 1
        ):
            raise ValueError("a near duplicate must meet the registered threshold")
        if self.action == "DROP_EMPTY" and self.normalized_byte_count != 0:
            raise ValueError("an empty-normalization drop must retain zero match bytes")
        if self.source == _CANONICAL_SOURCE and self.action not in {
            "KEEP_CANONICAL",
            "DROP_EMPTY",
        }:
            raise ValueError("Dolma ingestion may only keep canonical or drop empty")
        if self.source == _QUERY_SOURCE and self.action == "KEEP_CANONICAL":
            raise ValueError("FineWeb may not become the canonical source")

    @property
    def exact_jaccard(self) -> Fraction | None:
        if self.exact_jaccard_numerator is None:
            return None
        return Fraction(
            self.exact_jaccard_numerator,
            self.exact_jaccard_denominator,
        )

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "action": self.action,
            "canonical_document_id": self.canonical_document_id,
            "canonical_source_record_id": self.canonical_source_record_id,
            "decision_ordinal": self.decision_ordinal,
            "document_id": self.document_id,
            "exact_jaccard_denominator": self.exact_jaccard_denominator,
            "exact_jaccard_numerator": self.exact_jaccard_numerator,
            "lsh_candidate_count": self.lsh_candidate_count,
            "normalized_byte_count": self.normalized_byte_count,
            "retained_byte_count": self.retained_byte_count,
            "source": self.source,
            "source_order_ordinal": self.source_order_ordinal,
            "stable_source_record_id": self.stable_source_record_id,
        }


@dataclass(frozen=True)
class StreamingStateV3:
    phase: str
    next_canonical_ordinal: int
    next_query_ordinal: int
    next_decision_ordinal: int
    canonical_document_count: int
    next_recall_pair_ordinal: int
    recall_sample_identity_sha256: str | None


@dataclass(frozen=True)
class RecallAccountingV3:
    """Report-only LSH candidate accounting; no recall floor is implied."""

    sample_identity_sha256: str
    pair_count: int
    exact_pairs_at_or_above_threshold: int
    candidate_pairs_at_or_above_threshold: int
    missed_pairs_at_or_above_threshold: int
    candidate_pairs_below_threshold: int
    status: str = RECALL_STATUS

    def __post_init__(self) -> None:
        _require_sha256(self.sample_identity_sha256, "recall sample identity")
        for name in (
            "pair_count",
            "exact_pairs_at_or_above_threshold",
            "candidate_pairs_at_or_above_threshold",
            "missed_pairs_at_or_above_threshold",
            "candidate_pairs_below_threshold",
        ):
            _require_ordinal(getattr(self, name), name)
        if self.candidate_pairs_at_or_above_threshold + (
            self.missed_pairs_at_or_above_threshold
        ) != self.exact_pairs_at_or_above_threshold:
            raise ValueError("recall qualifying-pair accounting does not reconcile")
        if self.status != RECALL_STATUS:
            raise ValueError("A2 LSH qualification remains report-only")

    @property
    def observed_candidate_fraction_at_threshold(self) -> Fraction:
        if self.exact_pairs_at_or_above_threshold == 0:
            return Fraction(0, 1)
        return Fraction(
            self.candidate_pairs_at_or_above_threshold,
            self.exact_pairs_at_or_above_threshold,
        )


@contextmanager
def _begin_immediate(connection: sqlite3.Connection) -> Iterator[None]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()


class StreamingDedupStoreV3:
    """External-state A2 dedup index and append-only decision ledger."""

    def __init__(self, database_path: Path, *, timeout_seconds: float = 60.0) -> None:
        if sqlite3.sqlite_version != SQLITE_RUNTIME_VERSION:
            raise StreamingDedupError(
                "SQLite runtime differs from A2: "
                f"expected {SQLITE_RUNTIME_VERSION}, observed {sqlite3.sqlite_version}"
            )
        path = Path(database_path)
        if path.name in {"", ".", ".."} or path.exists() and path.is_dir():
            raise ValueError("streaming dedup database must be a regular-file path")
        if path.is_symlink():
            raise StreamingDedupError("streaming dedup database may not be a symlink")
        if path.exists() and not path.is_file():
            raise StreamingDedupError(
                "streaming dedup database must be absent or a regular file"
            )
        if not path.parent.is_dir():
            raise StreamingDedupError("streaming dedup database parent is absent")
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("SQLite timeout_seconds must be positive")
        self._database_was_present = path.exists()
        self.database_path = path.resolve(strict=False)
        self._connection = sqlite3.connect(
            self.database_path,
            timeout=float(timeout_seconds),
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._configure_connection()
            self._initialize_or_verify_schema()
        except BaseException:
            self._connection.close()
            raise

    def __enter__(self) -> StreamingDedupStoreV3:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _configure_connection(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute("PRAGMA temp_store = FILE")
        self._connection.execute("PRAGMA wal_autocheckpoint = 1000")
        mode = self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).casefold() != "wal":
            raise StreamingDedupError("SQLite refused the registered WAL journal mode")
        self._connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS query_bands ("
            "band_index INTEGER NOT NULL, band_key BLOB NOT NULL, "
            "PRIMARY KEY (band_index, band_key)) WITHOUT ROWID"
        )

    @staticmethod
    def _schema_statements() -> tuple[str, ...]:
        statements = (
            "CREATE TABLE IF NOT EXISTS metadata ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT",
            "CREATE TABLE IF NOT EXISTS ingest_state ("
            "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
            "phase TEXT NOT NULL CHECK (phase IN "
            "('DOLMA_CANONICAL','FINEWEB_QUERY','COMPLETE')), "
            "next_canonical_ordinal INTEGER NOT NULL "
            "CHECK (next_canonical_ordinal >= 0), "
            "next_query_ordinal INTEGER NOT NULL CHECK (next_query_ordinal >= 0), "
            "next_decision_ordinal INTEGER NOT NULL "
            "CHECK (next_decision_ordinal >= 0), "
            "canonical_document_count INTEGER NOT NULL "
            "CHECK (canonical_document_count >= 0), "
            "next_recall_pair_ordinal INTEGER NOT NULL "
            "CHECK (next_recall_pair_ordinal >= 0), "
            "recall_sample_identity_sha256 TEXT) STRICT",
            "CREATE TABLE IF NOT EXISTS phase_events ("
            "event_ordinal INTEGER PRIMARY KEY, event TEXT NOT NULL, "
            "source_count INTEGER NOT NULL CHECK (source_count >= 0)) STRICT",
            "CREATE TABLE IF NOT EXISTS canonical_documents ("
            "canonical_ordinal INTEGER PRIMARY KEY, "
            "document_id TEXT NOT NULL UNIQUE, "
            "stable_source_record_id TEXT NOT NULL UNIQUE, "
            "match_sha1 BLOB NOT NULL, match_length INTEGER NOT NULL "
            "CHECK (match_length > 0), match_sha256 BLOB NOT NULL, "
            "match_bytes BLOB NOT NULL, signature BLOB NOT NULL, "
            "CHECK (length(match_sha1) = 20), "
            "CHECK (length(match_sha256) = 32), "
            "CHECK (length(match_bytes) = match_length), "
            "CHECK (length(signature) = 1024)) STRICT",
            "CREATE INDEX IF NOT EXISTS canonical_exact_candidates ON "
            "canonical_documents(match_sha1, match_length, match_sha256, document_id)",
            "CREATE TABLE IF NOT EXISTS lsh_bands ("
            "band_index INTEGER NOT NULL CHECK (band_index BETWEEN 0 AND 15), "
            "band_key BLOB NOT NULL CHECK (length(band_key) = 64), "
            "canonical_document_id TEXT NOT NULL, "
            "PRIMARY KEY (band_index, band_key, canonical_document_id), "
            "FOREIGN KEY (canonical_document_id) REFERENCES "
            "canonical_documents(document_id)) WITHOUT ROWID, STRICT",
            "CREATE TABLE IF NOT EXISTS dedup_decisions ("
            "decision_ordinal INTEGER PRIMARY KEY, source TEXT NOT NULL, "
            "source_order_ordinal INTEGER NOT NULL CHECK (source_order_ordinal >= 0), "
            "document_id TEXT NOT NULL, stable_source_record_id TEXT NOT NULL, "
            "action TEXT NOT NULL, canonical_document_id TEXT, "
            "canonical_source_record_id TEXT, exact_jaccard_numerator INTEGER, "
            "exact_jaccard_denominator INTEGER, "
            "retained_byte_count INTEGER NOT NULL CHECK (retained_byte_count >= 0), "
            "normalized_byte_count INTEGER NOT NULL "
            "CHECK (normalized_byte_count >= 0), "
            "lsh_candidate_count INTEGER NOT NULL CHECK (lsh_candidate_count >= 0), "
            "UNIQUE (source, source_order_ordinal), "
            "FOREIGN KEY (canonical_document_id) REFERENCES "
            "canonical_documents(document_id)) STRICT",
            "CREATE TABLE IF NOT EXISTS recall_observations ("
            "pair_ordinal INTEGER PRIMARY KEY, sample_identity_sha256 TEXT NOT NULL, "
            "query_document_id TEXT NOT NULL, canonical_document_id TEXT NOT NULL, "
            "exact_jaccard_numerator INTEGER NOT NULL, "
            "exact_jaccard_denominator INTEGER NOT NULL "
            "CHECK (exact_jaccard_denominator > 0), "
            "was_lsh_candidate INTEGER NOT NULL "
            "CHECK (was_lsh_candidate IN (0, 1)), "
            "UNIQUE (sample_identity_sha256, query_document_id, "
            "canonical_document_id), "
            "FOREIGN KEY (canonical_document_id) REFERENCES "
            "canonical_documents(document_id)) STRICT",
        )
        trigger_statements: list[str] = []
        for table in _APPEND_ONLY_TABLES:
            trigger_statements.extend(
                (
                    f"CREATE TRIGGER IF NOT EXISTS {table}_reject_update "
                    f"BEFORE UPDATE ON {table} BEGIN "
                    "SELECT RAISE(ABORT, 'append-only table'); END",
                    f"CREATE TRIGGER IF NOT EXISTS {table}_reject_delete "
                    f"BEFORE DELETE ON {table} BEGIN "
                    "SELECT RAISE(ABORT, 'append-only table'); END",
                )
            )
        return statements + tuple(trigger_statements)

    @staticmethod
    def _expected_metadata() -> dict[str, str]:
        return {
            "authority_chain": canonical_json_bytes(
                GTOK_EXECUTION_AUTHORITY_CHAIN_V3
            ).decode("utf-8"),
            "dedup_seed": str(A2_DEDUP_SEED),
            "match_normalization_binding_sha256": (
                A2_MATCH_NORMALIZATION_BINDING.receipt_sha256
            ),
            "minhash_binding_sha256": A2_MINHASH_BINDING.receipt_sha256,
            "schema": STREAMING_SCHEMA,
            "schema_version": str(STREAMING_SCHEMA_VERSION),
            "source_order": f"{_CANONICAL_SOURCE},{_QUERY_SOURCE}",
            "sqlite_version": SQLITE_RUNTIME_VERSION,
        }

    @classmethod
    def _expected_schema_catalog(cls) -> tuple[tuple[object, ...], ...]:
        """Build the exact persistent schema in an isolated reference database."""

        reference = sqlite3.connect(":memory:", isolation_level=None)
        try:
            for statement in cls._schema_statements():
                reference.execute(statement)
            return tuple(
                reference.execute(
                    "SELECT type, name, tbl_name, sql FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                )
            )
        finally:
            reference.close()

    def _observed_schema_catalog(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            tuple(row)
            for row in self._connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )
        )

    def _verify_resume_invariants(self) -> None:
        state_rows = tuple(
            self._connection.execute(
                "SELECT phase, next_canonical_ordinal, next_query_ordinal, "
                "next_decision_ordinal, canonical_document_count, "
                "next_recall_pair_ordinal, recall_sample_identity_sha256 "
                "FROM ingest_state WHERE singleton = 1"
            )
        )
        if len(state_rows) != 1:
            raise StreamingDedupError("streaming dedup cursor state is corrupt")
        state = StreamingStateV3(*tuple(state_rows[0]))

        def _count(table: str) -> int:
            return int(
                self._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[
                    0
                ]
            )

        decision_count = _count("dedup_decisions")
        canonical_count = _count("canonical_documents")
        recall_count = _count("recall_observations")
        lsh_count = _count("lsh_bands")
        source_counts = dict(
            self._connection.execute(
                "SELECT source, count(*) FROM dedup_decisions "
                "GROUP BY source ORDER BY source"
            )
        )
        if decision_count != state.next_decision_ordinal:
            raise StreamingDedupError("decision ledger count differs from its cursor")
        if canonical_count != state.canonical_document_count:
            raise StreamingDedupError("canonical index count differs from its cursor")
        if recall_count != state.next_recall_pair_ordinal:
            raise StreamingDedupError("recall ledger count differs from its cursor")
        if lsh_count != 16 * canonical_count:
            raise StreamingDedupError("LSH index cardinality differs from canonical state")
        if int(source_counts.get(_CANONICAL_SOURCE, 0)) != (
            state.next_canonical_ordinal
        ):
            raise StreamingDedupError("canonical decision count differs from its cursor")
        if int(source_counts.get(_QUERY_SOURCE, 0)) != state.next_query_ordinal:
            raise StreamingDedupError("query decision count differs from its cursor")

        expected_events: tuple[tuple[object, ...], ...]
        if state.phase == _PHASE_CANONICAL:
            expected_events = ((0, "BEGIN_DOLMA_CANONICAL", 0),)
        elif state.phase == _PHASE_QUERY:
            expected_events = (
                (0, "BEGIN_DOLMA_CANONICAL", 0),
                (1, "SEAL_DOLMA_BEGIN_FINEWEB", state.next_canonical_ordinal),
            )
        elif state.phase == _PHASE_COMPLETE:
            expected_events = (
                (0, "BEGIN_DOLMA_CANONICAL", 0),
                (1, "SEAL_DOLMA_BEGIN_FINEWEB", state.next_canonical_ordinal),
                (2, "SEAL_FINEWEB_COMPLETE", state.next_query_ordinal),
            )
        else:  # pragma: no cover - the STRICT CHECK also rejects this at write time
            raise StreamingDedupError("streaming dedup phase is invalid")
        observed_events = tuple(
            tuple(row)
            for row in self._connection.execute(
                "SELECT event_ordinal, event, source_count FROM phase_events "
                "ORDER BY event_ordinal"
            )
        )
        if observed_events != expected_events:
            raise StreamingDedupError("phase-event ledger differs from cursor state")
        if (state.recall_sample_identity_sha256 is None) != (recall_count == 0):
            raise StreamingDedupError("recall sample identity differs from recall state")
        foreign_key_errors = tuple(self._connection.execute("PRAGMA foreign_key_check"))
        if foreign_key_errors:
            raise StreamingDedupError("SQLite foreign-key check failed")

    def _initialize_or_verify_schema(self) -> None:
        with _begin_immediate(self._connection):
            user_version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if not self._database_was_present:
                if self._observed_schema_catalog():
                    raise StreamingDedupError("new streaming database was not empty")
                for statement in self._schema_statements():
                    self._connection.execute(statement)
                self._connection.execute(
                    f"PRAGMA user_version = {STREAMING_SCHEMA_VERSION}"
                )
            else:
                if user_version != STREAMING_SCHEMA_VERSION:
                    raise StreamingDedupError("streaming dedup schema version drifted")
                if self._observed_schema_catalog() != self._expected_schema_catalog():
                    raise StreamingDedupError(
                        "streaming dedup persistent schema differs from A2"
                    )

            user_version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if user_version != STREAMING_SCHEMA_VERSION:
                raise StreamingDedupError("streaming dedup schema version drifted")

            observed = dict(
                self._connection.execute("SELECT key, value FROM metadata")
            )
            expected = self._expected_metadata()
            if not observed:
                if self._database_was_present:
                    raise StreamingDedupError("streaming dedup metadata is absent")
                self._connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    tuple(sorted(expected.items())),
                )
                self._connection.execute(
                    "INSERT INTO ingest_state VALUES (1, ?, 0, 0, 0, 0, 0, NULL)",
                    (_PHASE_CANONICAL,),
                )
                self._connection.execute(
                    "INSERT INTO phase_events VALUES (0, 'BEGIN_DOLMA_CANONICAL', 0)"
                )
            elif observed != expected:
                raise StreamingDedupError("streaming dedup metadata differs from A2")
            self._verify_resume_invariants()
            integrity = self._connection.execute("PRAGMA quick_check").fetchone()[0]
            if integrity != "ok":
                raise StreamingDedupError(f"SQLite quick_check failed: {integrity}")

    def state(self) -> StreamingStateV3:
        row = self._connection.execute(
            "SELECT phase, next_canonical_ordinal, next_query_ordinal, "
            "next_decision_ordinal, canonical_document_count, "
            "next_recall_pair_ordinal, recall_sample_identity_sha256 "
            "FROM ingest_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise StreamingDedupError("streaming dedup cursor state is absent")
        return StreamingStateV3(*tuple(row))

    @staticmethod
    def _decision_from_row(row: sqlite3.Row) -> StreamingDedupDecisionV3:
        return StreamingDedupDecisionV3(**dict(row))

    def _insert_decision(self, decision: StreamingDedupDecisionV3) -> None:
        self._connection.execute(
            "INSERT INTO dedup_decisions VALUES "
            "(:decision_ordinal, :source, :source_order_ordinal, :document_id, "
            ":stable_source_record_id, :action, :canonical_document_id, "
            ":canonical_source_record_id, :exact_jaccard_numerator, "
            ":exact_jaccard_denominator, :retained_byte_count, "
            ":normalized_byte_count, :lsh_candidate_count)",
            decision.canonical_payload,
        )

    def _require_position(self, *, phase: str, ordinal: int) -> StreamingStateV3:
        state = self.state()
        if state.phase != phase:
            raise StreamingDedupError(
                f"source phase mismatch: expected {phase}, observed {state.phase}"
            )
        expected = (
            state.next_canonical_ordinal
            if phase == _PHASE_CANONICAL
            else state.next_query_ordinal
        )
        if ordinal != expected:
            raise StreamingDedupError(
                f"source order mismatch: expected {expected}, received {ordinal}"
            )
        return state

    def append_canonical(
        self,
        document: StableDocumentV3,
        *,
        source_order_ordinal: int,
    ) -> StreamingDedupDecisionV3:
        """Append one Dolma-web document in exact caller-declared source order."""

        if not isinstance(document, StableDocumentV3):
            raise TypeError("canonical ingestion requires a StableDocumentV3")
        if document.source != _CANONICAL_SOURCE:
            raise ValueError("canonical ingestion accepts only Dolma web")
        _require_ordinal(source_order_ordinal, "source_order_ordinal")
        match_bytes = normalized_match_bytes(document.text)
        signature: tuple[int, ...] | None = None
        bands: tuple[bytes, ...] = ()
        if match_bytes:
            signature = minhash_signature_v3(byte_shingles_v3(match_bytes))
            bands = lsh_band_keys_v3(signature)

        with _begin_immediate(self._connection):
            state = self._require_position(
                phase=_PHASE_CANONICAL,
                ordinal=source_order_ordinal,
            )
            action = "KEEP_CANONICAL" if match_bytes else "DROP_EMPTY"
            decision = StreamingDedupDecisionV3(
                decision_ordinal=state.next_decision_ordinal,
                source=document.source,
                source_order_ordinal=source_order_ordinal,
                document_id=document.document_id,
                stable_source_record_id=document.stable_source_record_id,
                action=action,
                canonical_document_id=None,
                canonical_source_record_id=None,
                exact_jaccard_numerator=None,
                exact_jaccard_denominator=None,
                retained_byte_count=document.retained_byte_count,
                normalized_byte_count=len(match_bytes),
                lsh_candidate_count=0,
            )
            canonical_count = state.canonical_document_count
            if match_bytes:
                sha1, length, sha256 = _match_identity(match_bytes)
                try:
                    self._connection.execute(
                        "INSERT INTO canonical_documents "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            canonical_count,
                            document.document_id,
                            document.stable_source_record_id,
                            sha1,
                            length,
                            sha256,
                            match_bytes,
                            _signature_blob(signature or ()),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise StreamingDedupError(
                        "canonical stable identity is not unique"
                    ) from error
                self._connection.executemany(
                    "INSERT INTO lsh_bands VALUES (?, ?, ?)",
                    tuple(
                        (band_index, band_key, document.document_id)
                        for band_index, band_key in enumerate(bands)
                    ),
                )
                canonical_count += 1
            self._insert_decision(decision)
            changed = self._connection.execute(
                "UPDATE ingest_state SET next_canonical_ordinal = ?, "
                "next_decision_ordinal = ?, canonical_document_count = ? "
                "WHERE singleton = 1 AND phase = ? AND next_canonical_ordinal = ?",
                (
                    source_order_ordinal + 1,
                    state.next_decision_ordinal + 1,
                    canonical_count,
                    _PHASE_CANONICAL,
                    source_order_ordinal,
                ),
            ).rowcount
            if changed != 1:
                raise StreamingDedupError("canonical cursor advance lost its lock")
        return decision

    def seal_canonical_phase(self, *, expected_source_count: int) -> None:
        """Freeze the Dolma index before the first FineWeb query is admitted."""

        _require_ordinal(expected_source_count, "expected_source_count")
        with _begin_immediate(self._connection):
            state = self.state()
            if state.phase != _PHASE_CANONICAL:
                raise StreamingDedupError("Dolma canonical phase is already sealed")
            if state.next_canonical_ordinal != expected_source_count:
                raise StreamingDedupError("Dolma source count differs at phase seal")
            if state.canonical_document_count < 1:
                raise StreamingDedupError("cannot seal an empty canonical index")
            self._connection.execute(
                "INSERT INTO phase_events VALUES (1, 'SEAL_DOLMA_BEGIN_FINEWEB', ?)",
                (expected_source_count,),
            )
            changed = self._connection.execute(
                "UPDATE ingest_state SET phase = ? "
                "WHERE singleton = 1 AND phase = ?",
                (_PHASE_QUERY, _PHASE_CANONICAL),
            ).rowcount
            if changed != 1:
                raise StreamingDedupError("phase seal lost its lock")

    def _exact_winner(
        self,
        *,
        match_sha1: bytes,
        match_length: int,
        match_sha256: bytes,
        match_bytes: bytes,
    ) -> tuple[str, str] | None:
        cursor = self._connection.execute(
            "SELECT document_id, stable_source_record_id, match_length, "
            "match_sha256, match_bytes FROM canonical_documents "
            "WHERE match_sha1 = ? ORDER BY document_id",
            (match_sha1,),
        )
        best: tuple[str, str] | None = None
        for row in cursor:
            if (
                row["match_length"] == match_length
                and row["match_sha256"] == match_sha256
                and row["match_bytes"] == match_bytes
            ):
                candidate = (
                    row["stable_source_record_id"],
                    row["document_id"],
                )
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return None
        source_record_id, document_id = best
        return document_id, source_record_id

    def _near_winner(
        self,
        *,
        match_bytes: bytes,
        band_keys: Sequence[bytes],
    ) -> tuple[tuple[str, str, Fraction] | None, int]:
        self._connection.execute("DELETE FROM query_bands")
        self._connection.executemany(
            "INSERT INTO query_bands VALUES (?, ?)",
            tuple(enumerate(band_keys)),
        )
        cursor = self._connection.execute(
            "SELECT DISTINCT d.document_id, d.stable_source_record_id, d.match_bytes "
            "FROM canonical_documents AS d "
            "JOIN lsh_bands AS b ON b.canonical_document_id = d.document_id "
            "JOIN query_bands AS q ON q.band_index = b.band_index "
            "AND q.band_key = b.band_key ORDER BY d.document_id"
        )
        query_shingles = byte_shingles_v3(match_bytes)
        best: tuple[Fraction, str, str] | None = None
        candidate_count = 0
        for row in cursor:
            candidate_count += 1
            score = exact_jaccard_v3(
                query_shingles,
                byte_shingles_v3(row["match_bytes"]),
            )
            if score < NEAR_DUPLICATE_THRESHOLD:
                continue
            candidate = (score, row["stable_source_record_id"], row["document_id"])
            if best is None or (-candidate[0], candidate[1]) < (-best[0], best[1]):
                best = candidate
        if best is None:
            return None, candidate_count
        score, source_record_id, document_id = best
        return (document_id, source_record_id, score), candidate_count

    def append_query(
        self,
        document: StableDocumentV3,
        *,
        source_order_ordinal: int,
    ) -> StreamingDedupDecisionV3:
        """Evaluate one FineWeb-Edu document against the frozen Dolma index."""

        if not isinstance(document, StableDocumentV3):
            raise TypeError("query ingestion requires a StableDocumentV3")
        if document.source != _QUERY_SOURCE:
            raise ValueError("query ingestion accepts only FineWeb-Edu")
        _require_ordinal(source_order_ordinal, "source_order_ordinal")
        match_bytes = normalized_match_bytes(document.text)

        with _begin_immediate(self._connection):
            state = self._require_position(
                phase=_PHASE_QUERY,
                ordinal=source_order_ordinal,
            )
            winner_document_id: str | None = None
            winner_source_record_id: str | None = None
            score: Fraction | None = None
            candidate_count = 0
            if not match_bytes:
                action = "DROP_EMPTY"
            else:
                sha1, length, sha256 = _match_identity(match_bytes)
                exact = self._exact_winner(
                    match_sha1=sha1,
                    match_length=length,
                    match_sha256=sha256,
                    match_bytes=match_bytes,
                )
                if exact is not None:
                    winner_document_id, winner_source_record_id = exact
                    score = Fraction(1, 1)
                    action = "DROP_EXACT"
                else:
                    signature = minhash_signature_v3(byte_shingles_v3(match_bytes))
                    near, candidate_count = self._near_winner(
                        match_bytes=match_bytes,
                        band_keys=lsh_band_keys_v3(signature),
                    )
                    if near is None:
                        action = "KEEP_FINEWEB"
                    else:
                        (
                            winner_document_id,
                            winner_source_record_id,
                            score,
                        ) = near
                        action = "DROP_NEAR"
            decision = StreamingDedupDecisionV3(
                decision_ordinal=state.next_decision_ordinal,
                source=document.source,
                source_order_ordinal=source_order_ordinal,
                document_id=document.document_id,
                stable_source_record_id=document.stable_source_record_id,
                action=action,
                canonical_document_id=winner_document_id,
                canonical_source_record_id=winner_source_record_id,
                exact_jaccard_numerator=None if score is None else score.numerator,
                exact_jaccard_denominator=None if score is None else score.denominator,
                retained_byte_count=document.retained_byte_count,
                normalized_byte_count=len(match_bytes),
                lsh_candidate_count=candidate_count,
            )
            self._insert_decision(decision)
            changed = self._connection.execute(
                "UPDATE ingest_state SET next_query_ordinal = ?, "
                "next_decision_ordinal = ? WHERE singleton = 1 "
                "AND phase = ? AND next_query_ordinal = ?",
                (
                    source_order_ordinal + 1,
                    state.next_decision_ordinal + 1,
                    _PHASE_QUERY,
                    source_order_ordinal,
                ),
            ).rowcount
            if changed != 1:
                raise StreamingDedupError("FineWeb cursor advance lost its lock")
        return decision

    def seal_query_phase(self, *, expected_source_count: int) -> None:
        _require_ordinal(expected_source_count, "expected_source_count")
        with _begin_immediate(self._connection):
            state = self.state()
            if state.phase != _PHASE_QUERY:
                raise StreamingDedupError("FineWeb query phase cannot be sealed now")
            if state.next_query_ordinal != expected_source_count:
                raise StreamingDedupError("FineWeb source count differs at phase seal")
            self._connection.execute(
                "INSERT INTO phase_events VALUES (2, 'SEAL_FINEWEB_COMPLETE', ?)",
                (expected_source_count,),
            )
            changed = self._connection.execute(
                "UPDATE ingest_state SET phase = ? WHERE singleton = 1 AND phase = ?",
                (_PHASE_COMPLETE, _PHASE_QUERY),
            ).rowcount
            if changed != 1:
                raise StreamingDedupError("query phase seal lost its lock")

    def iter_decisions(self) -> Iterator[StreamingDedupDecisionV3]:
        """Stream the canonical ledger without loading all decisions into RAM."""

        cursor = self._connection.execute(
            "SELECT decision_ordinal, source, source_order_ordinal, document_id, "
            "stable_source_record_id, action, canonical_document_id, "
            "canonical_source_record_id, exact_jaccard_numerator, "
            "exact_jaccard_denominator, retained_byte_count, normalized_byte_count, "
            "lsh_candidate_count FROM dedup_decisions ORDER BY decision_ordinal"
        )
        for row in cursor:
            yield self._decision_from_row(row)

    def decision_ledger_sha256(self) -> str:
        """Hash canonical JSONL by streaming ordered rows from external state."""

        digest = hashlib.sha256()
        for record in self.iter_decision_jsonl_bytes():
            digest.update(record)
        return digest.hexdigest()

    def iter_decision_jsonl_bytes(self) -> Iterator[bytes]:
        """Stream the bound canonical JSONL ledger without an in-memory ledger."""

        for decision in self.iter_decisions():
            yield canonical_json_bytes(decision.canonical_payload) + b"\n"

    def decision_counts(self) -> dict[str, int]:
        counts = dict(
            self._connection.execute(
                "SELECT action, count(*) FROM dedup_decisions "
                "GROUP BY action ORDER BY action"
            )
        )
        return {action: int(counts.get(action, 0)) for action in sorted(_ACTIONS)}

    def empty_drop_counts_by_source(self) -> dict[str, int]:
        counts = dict(
            self._connection.execute(
                "SELECT source, count(*) FROM dedup_decisions "
                "WHERE action = 'DROP_EMPTY' GROUP BY source ORDER BY source"
            )
        )
        return {
            _CANONICAL_SOURCE: int(counts.get(_CANONICAL_SOURCE, 0)),
            _QUERY_SOURCE: int(counts.get(_QUERY_SOURCE, 0)),
        }

    def append_recall_pair(
        self,
        *,
        sample_identity_sha256: str,
        pair_ordinal: int,
        query_document: StableDocumentV3,
        canonical_document_id: str,
    ) -> None:
        """Append one externally selected exhaustive-sample pair for reporting.

        The store recomputes exact Jaccard and whether the registered 16x8 index
        would generate the pair.  The caller owns exhaustive sample enumeration;
        this hook never upgrades the result into a recall guarantee.
        """

        _require_sha256(sample_identity_sha256, "sample_identity_sha256")
        _require_ordinal(pair_ordinal, "pair_ordinal")
        _require_sha256(canonical_document_id, "canonical_document_id")
        if not isinstance(query_document, StableDocumentV3):
            raise TypeError("recall pair query requires a StableDocumentV3")
        if query_document.source != _QUERY_SOURCE:
            raise ValueError("recall pair query must be FineWeb-Edu")
        query_match = normalized_match_bytes(query_document.text)
        if not query_match:
            raise ValueError("recall audit excludes empty-normalization documents")
        query_shingles = byte_shingles_v3(query_match)
        query_bands = lsh_band_keys_v3(minhash_signature_v3(query_shingles))

        with _begin_immediate(self._connection):
            state = self.state()
            if pair_ordinal != state.next_recall_pair_ordinal:
                raise StreamingDedupError("recall pair order is not contiguous")
            if state.recall_sample_identity_sha256 not in {
                None,
                sample_identity_sha256,
            }:
                raise StreamingDedupError("recall sample identity changed mid-audit")
            canonical = self._connection.execute(
                "SELECT match_bytes FROM canonical_documents WHERE document_id = ?",
                (canonical_document_id,),
            ).fetchone()
            if canonical is None:
                raise StreamingDedupError("recall canonical document is absent")
            score = exact_jaccard_v3(
                query_shingles,
                byte_shingles_v3(canonical["match_bytes"]),
            )
            was_candidate = any(
                self._connection.execute(
                    "SELECT 1 FROM lsh_bands WHERE band_index = ? "
                    "AND band_key = ? AND canonical_document_id = ? LIMIT 1",
                    (band_index, band_key, canonical_document_id),
                ).fetchone()
                is not None
                for band_index, band_key in enumerate(query_bands)
            )
            self._connection.execute(
                "INSERT INTO recall_observations VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    pair_ordinal,
                    sample_identity_sha256,
                    query_document.document_id,
                    canonical_document_id,
                    score.numerator,
                    score.denominator,
                    int(was_candidate),
                ),
            )
            changed = self._connection.execute(
                "UPDATE ingest_state SET next_recall_pair_ordinal = ?, "
                "recall_sample_identity_sha256 = ? WHERE singleton = 1 "
                "AND next_recall_pair_ordinal = ?",
                (pair_ordinal + 1, sample_identity_sha256, pair_ordinal),
            ).rowcount
            if changed != 1:
                raise StreamingDedupError("recall cursor advance lost its lock")

    def recall_accounting(self) -> RecallAccountingV3:
        state = self.state()
        if state.recall_sample_identity_sha256 is None:
            raise StreamingDedupError("no recall sample observations are recorded")
        row = self._connection.execute(
            "SELECT count(*) AS pair_count, "
            "sum(CASE WHEN exact_jaccard_numerator * 5 >= "
            "exact_jaccard_denominator * 4 THEN 1 ELSE 0 END) AS exact_qualifying, "
            "sum(CASE WHEN exact_jaccard_numerator * 5 >= "
            "exact_jaccard_denominator * 4 AND was_lsh_candidate = 1 "
            "THEN 1 ELSE 0 END) AS candidate_qualifying, "
            "sum(CASE WHEN exact_jaccard_numerator * 5 < "
            "exact_jaccard_denominator * 4 AND was_lsh_candidate = 1 "
            "THEN 1 ELSE 0 END) AS candidate_below "
            "FROM recall_observations WHERE sample_identity_sha256 = ?",
            (state.recall_sample_identity_sha256,),
        ).fetchone()
        exact_qualifying = int(row["exact_qualifying"] or 0)
        candidate_qualifying = int(row["candidate_qualifying"] or 0)
        return RecallAccountingV3(
            sample_identity_sha256=state.recall_sample_identity_sha256,
            pair_count=int(row["pair_count"]),
            exact_pairs_at_or_above_threshold=exact_qualifying,
            candidate_pairs_at_or_above_threshold=candidate_qualifying,
            missed_pairs_at_or_above_threshold=(
                exact_qualifying - candidate_qualifying
            ),
            candidate_pairs_below_threshold=int(row["candidate_below"] or 0),
        )


def serial_reference_decisions_v3(
    canonical_documents: Sequence[StableDocumentV3],
    query_documents: Sequence[StableDocumentV3],
) -> tuple[StreamingDedupDecisionV3, ...]:
    """Bounded in-memory oracle for fixtures, never a production implementation."""

    if not isinstance(canonical_documents, Sequence) or isinstance(
        canonical_documents, (str, bytes)
    ):
        raise TypeError("serial canonical fixture must be a typed sequence")
    if not isinstance(query_documents, Sequence) or isinstance(
        query_documents, (str, bytes)
    ):
        raise TypeError("serial query fixture must be a typed sequence")
    if len(canonical_documents) + len(query_documents) > SERIAL_REFERENCE_MAX_DOCUMENTS:
        raise ValueError("serial reference is limited to bounded fixtures")
    if any(
        not isinstance(document, StableDocumentV3)
        or document.source != _CANONICAL_SOURCE
        for document in canonical_documents
    ):
        raise ValueError("serial canonical fixture accepts only Dolma web")
    if any(
        not isinstance(document, StableDocumentV3)
        or document.source != _QUERY_SOURCE
        for document in query_documents
    ):
        raise ValueError("serial query fixture accepts only FineWeb-Edu")

    decisions: list[StreamingDedupDecisionV3] = []
    indexed: list[
        tuple[StableDocumentV3, bytes, bytes, int, bytes, tuple[bytes, ...]]
    ] = []
    for ordinal, document in enumerate(canonical_documents):
        match = normalized_match_bytes(document.text)
        decisions.append(
            StreamingDedupDecisionV3(
                decision_ordinal=len(decisions),
                source=document.source,
                source_order_ordinal=ordinal,
                document_id=document.document_id,
                stable_source_record_id=document.stable_source_record_id,
                action="KEEP_CANONICAL" if match else "DROP_EMPTY",
                canonical_document_id=None,
                canonical_source_record_id=None,
                exact_jaccard_numerator=None,
                exact_jaccard_denominator=None,
                retained_byte_count=document.retained_byte_count,
                normalized_byte_count=len(match),
                lsh_candidate_count=0,
            )
        )
        if match:
            sha1, length, sha256 = _match_identity(match)
            bands = lsh_band_keys_v3(
                minhash_signature_v3(byte_shingles_v3(match))
            )
            indexed.append((document, sha1, length, sha256, bands))

    for ordinal, document in enumerate(query_documents):
        match = normalized_match_bytes(document.text)
        winner_document_id: str | None = None
        winner_source_record_id: str | None = None
        score: Fraction | None = None
        candidate_count = 0
        if not match:
            action = "DROP_EMPTY"
        else:
            sha1, length, sha256 = _match_identity(match)
            exact_candidates = [
                candidate
                for candidate, candidate_sha1, candidate_length, candidate_sha256, _
                in indexed
                if candidate_sha1 == sha1
                and candidate_length == length
                and candidate_sha256 == sha256
                and normalized_match_bytes(candidate.text) == match
            ]
            if exact_candidates:
                winner = min(
                    exact_candidates,
                    key=lambda candidate: candidate.stable_source_record_id,
                )
                winner_document_id = winner.document_id
                winner_source_record_id = winner.stable_source_record_id
                score = Fraction(1, 1)
                action = "DROP_EXACT"
            else:
                query_bands = lsh_band_keys_v3(
                    minhash_signature_v3(byte_shingles_v3(match))
                )
                candidate_documents = sorted(
                    (
                        candidate
                        for candidate, _, _, _, bands in indexed
                        if any(
                            left == right
                            for left, right in zip(query_bands, bands, strict=True)
                        )
                    ),
                    key=lambda candidate: candidate.document_id,
                )
                candidate_count = len(candidate_documents)
                winner = select_dedup_winner_v3(document, candidate_documents)
                if winner is None:
                    action = "KEEP_FINEWEB"
                else:
                    winner_document_id = winner.canonical_document_id
                    winner_source_record_id = winner.canonical_source_record_id
                    score = winner.exact_jaccard
                    action = "DROP_NEAR"
        decisions.append(
            StreamingDedupDecisionV3(
                decision_ordinal=len(decisions),
                source=document.source,
                source_order_ordinal=ordinal,
                document_id=document.document_id,
                stable_source_record_id=document.stable_source_record_id,
                action=action,
                canonical_document_id=winner_document_id,
                canonical_source_record_id=winner_source_record_id,
                exact_jaccard_numerator=None if score is None else score.numerator,
                exact_jaccard_denominator=None if score is None else score.denominator,
                retained_byte_count=document.retained_byte_count,
                normalized_byte_count=len(match),
                lsh_candidate_count=candidate_count,
            )
        )
    return tuple(decisions)


__all__ = [
    "RECALL_STATUS",
    "SERIAL_REFERENCE_MAX_DOCUMENTS",
    "SQLITE_RUNTIME_VERSION",
    "STREAMING_SCHEMA",
    "RecallAccountingV3",
    "StreamingDedupDecisionV3",
    "StreamingDedupError",
    "StreamingDedupStoreV3",
    "StreamingStateV3",
    "serial_reference_decisions_v3",
]
