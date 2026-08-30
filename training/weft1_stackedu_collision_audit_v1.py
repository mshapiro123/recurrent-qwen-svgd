"""Hash-only, offline StackEdu native-ID collision audit.

This diagnostic deliberately does not alter corpus selection or materialization.
It replays the selected StackEdu assets from a V4 source-cache manifest, uses
the frozen production parser, and persists only hashes and integer metadata.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tempfile
from typing import BinaryIO, Mapping

from training.weft1_corpus_a3 import execution_authority_v4_bound_sha256
from training.weft1_corpus_materialize_a3 import load_source_manifest_artifact_v4
from training.weft1_corpus_source_io_a2 import (
    PARSE_DISPOSITIONS,
    PRODUCTION_PARSER_COMPOSITE_IDENTITIES_V3,
    RETAIN,
    iter_source_asset_events_v3,
    resolve_production_parser_binding_v3,
)
from training.weft1_corpus_sources_a2 import VerifiedLocalCacheAssetV3
from training.weft1_gtok_a1_contract import load_source_route_manifest
from training.weft1_gtok_contract import canonical_json_bytes, canonical_sha256
from training.weft1_strict_io import assert_no_symlink_ancestors


AUDIT_SCHEMA_V1 = "weft1_stackedu_native_id_collision_audit_v4"
EVIDENCE_SCHEMA_V1 = "weft1_stackedu_native_id_collision_evidence_v4"
RECEIPT_ARTIFACT_SCHEMA_V1 = "weft1_stackedu_collision_audit_artifact_v4"
LEDGER_NAME_V1 = "stackedu-native-id-collisions-v1.jsonl"
RECEIPT_NAME_V1 = "stackedu-native-id-collision-receipt-v1.json"

EXACT_REPEAT = "EXACT_REPEAT"
SCORE_ONLY_VARIANCE = "SCORE_ONLY_VARIANCE"
CONTENT_DIVERGENCE = "CONTENT_DIVERGENCE"
CLASSIFICATIONS = (EXACT_REPEAT, SCORE_ONLY_VARIANCE, CONTENT_DIVERGENCE)

GREEN_COMPLETE = "GREEN_COMPLETE"
STOP_CONTENT_DIVERGENCE = "STOP_CONTENT_DIVERGENCE"
STATUSES = (
    GREEN_COMPLETE,
    STOP_CONTENT_DIVERGENCE,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_NATIVE_ID_DOMAIN = b"weft1-stackedu-native-id-v1\0"


class StackEduCollisionAuditError(RuntimeError):
    """The diagnostic cannot produce governed collision evidence."""


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_exact_nonnegative(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact integer")
    return value


@dataclass(frozen=True)
class StackEduOccurrenceV1:
    """One hash-only occurrence; no source text or raw native ID is retained."""

    native_id_sha256: str
    native_id_utf8_bytes: int
    stable_source_record_id: str
    source_cache_asset_identity_sha256: str
    manifest_asset_order_ordinal: int
    source_record_ordinal: int
    text_sha256: str
    text_utf8_bytes: int
    int_score: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.native_id_sha256, "native ID SHA-256"),
            (self.stable_source_record_id, "stable source record ID"),
            (
                self.source_cache_asset_identity_sha256,
                "source-cache asset identity",
            ),
            (self.text_sha256, "text SHA-256"),
        ):
            _require_sha256(value, name)
        for value, name in (
            (self.native_id_utf8_bytes, "native ID bytes"),
            (self.manifest_asset_order_ordinal, "manifest asset ordinal"),
            (self.source_record_ordinal, "source record ordinal"),
            (self.text_utf8_bytes, "text bytes"),
        ):
            _require_exact_nonnegative(value, name)
        if self.native_id_utf8_bytes < 1 or self.text_utf8_bytes < 1:
            raise ValueError("native ID and text byte counts must be positive")
        if type(self.int_score) is not int or self.int_score < 3:
            raise ValueError("collision occurrences must be eligible StackEdu rows")


@dataclass(frozen=True)
class StackEduCollisionEvidenceV1:
    classification: str
    first: StackEduOccurrenceV1
    repeated: StackEduOccurrenceV1

    def __post_init__(self) -> None:
        if self.classification not in CLASSIFICATIONS:
            raise ValueError("unknown StackEdu collision classification")
        if not isinstance(self.first, StackEduOccurrenceV1) or not isinstance(
            self.repeated, StackEduOccurrenceV1
        ):
            raise TypeError("collision evidence requires typed occurrences")
        if (
            self.first.stable_source_record_id
            != self.repeated.stable_source_record_id
            or self.first.native_id_sha256 != self.repeated.native_id_sha256
            or self.first.native_id_utf8_bytes
            != self.repeated.native_id_utf8_bytes
        ):
            raise ValueError("collision evidence joins different native identities")
        same_content = (
            self.first.text_sha256 == self.repeated.text_sha256
            and self.first.text_utf8_bytes == self.repeated.text_utf8_bytes
        )
        same_score = self.first.int_score == self.repeated.int_score
        expected = (
            EXACT_REPEAT
            if same_content and same_score
            else SCORE_ONLY_VARIANCE
            if same_content
            else CONTENT_DIVERGENCE
        )
        if self.classification != expected:
            raise ValueError("collision classification disagrees with its evidence")

    @property
    def evidence_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(EVIDENCE_SCHEMA_V1, self)


@dataclass(frozen=True)
class StackEduCollisionAuditReceiptV1:
    schema: str
    status: str
    source_manifest_artifact_sha256: str
    source_manifest_receipt_sha256: str
    execution_binding_sha256: str
    effective_route_identity_sha256: str
    stackedu_effective_route_receipt_sha256: str
    parser_composite_identity_sha256: str
    parser_binding_asset_counts: tuple[tuple[str, int, int], ...]
    selected_asset_count: int
    selected_compressed_bytes: int
    selected_asset_sequence_sha256: str
    verified_asset_count: int
    verified_compressed_bytes: int
    verified_asset_sequence_sha256: str
    parse_event_count: int
    parse_disposition_counts: tuple[tuple[str, int], ...]
    eligible_record_count: int
    distinct_eligible_native_id_count: int
    exact_repeat_count: int
    score_only_variance_count: int
    content_divergence_count: int
    collision_ledger_rows: int
    collision_ledger_bytes: int
    collision_ledger_sha256: str
    terminal_evidence_sha256: str | None

    def __post_init__(self) -> None:
        if self.schema != AUDIT_SCHEMA_V1:
            raise ValueError("unexpected StackEdu collision-audit schema")
        if self.status not in STATUSES:
            raise ValueError("unexpected StackEdu collision-audit status")
        for value, name in (
            (self.source_manifest_artifact_sha256, "manifest artifact"),
            (self.source_manifest_receipt_sha256, "manifest receipt"),
            (self.execution_binding_sha256, "execution binding"),
            (self.effective_route_identity_sha256, "effective route identity"),
            (
                self.stackedu_effective_route_receipt_sha256,
                "StackEdu effective route receipt",
            ),
            (
                self.parser_composite_identity_sha256,
                "parser composite identity",
            ),
            (self.selected_asset_sequence_sha256, "selected asset sequence"),
            (self.verified_asset_sequence_sha256, "verified asset sequence"),
            (self.collision_ledger_sha256, "collision ledger"),
        ):
            _require_sha256(value, name)
        if self.terminal_evidence_sha256 is not None:
            _require_sha256(self.terminal_evidence_sha256, "terminal evidence")
        for field in (
            "selected_asset_count",
            "selected_compressed_bytes",
            "verified_asset_count",
            "verified_compressed_bytes",
            "parse_event_count",
            "eligible_record_count",
            "distinct_eligible_native_id_count",
            "exact_repeat_count",
            "score_only_variance_count",
            "content_divergence_count",
            "collision_ledger_rows",
            "collision_ledger_bytes",
        ):
            _require_exact_nonnegative(getattr(self, field), field)
        if self.selected_asset_count < 1:
            raise ValueError("collision audit requires selected StackEdu assets")
        if self.selected_compressed_bytes < 1:
            raise ValueError("selected compressed bytes must be positive")
        if not 0 < self.verified_asset_count <= self.selected_asset_count:
            raise ValueError("verified asset coverage is invalid")
        if not 0 < self.verified_compressed_bytes <= self.selected_compressed_bytes:
            raise ValueError("verified compressed-byte coverage is invalid")
        if not self.parser_binding_asset_counts:
            raise ValueError("collision audit lacks parser-binding coverage")
        parser_hashes: list[str] = []
        selected_parser_assets = 0
        verified_parser_assets = 0
        for row in self.parser_binding_asset_counts:
            if (
                not isinstance(row, tuple)
                or len(row) != 3
                or type(row[1]) is not int
                or type(row[2]) is not int
                or row[1] < 1
                or not 0 <= row[2] <= row[1]
            ):
                raise ValueError("parser-binding asset coverage is invalid")
            parser_hashes.append(_require_sha256(row[0], "parser binding"))
            selected_parser_assets += row[1]
            verified_parser_assets += row[2]
        if parser_hashes != sorted(set(parser_hashes)):
            raise ValueError("parser-binding coverage is noncanonical")
        if (
            selected_parser_assets != self.selected_asset_count
            or verified_parser_assets != self.verified_asset_count
        ):
            raise ValueError("parser-binding counts do not cover audited assets")
        expected_dispositions = tuple(
            (name, count)
            for name, count in self.parse_disposition_counts
        )
        if tuple(name for name, _ in expected_dispositions) != tuple(
            sorted(PARSE_DISPOSITIONS)
        ):
            raise ValueError("parse disposition counts are incomplete or noncanonical")
        if any(type(count) is not int or count < 0 for _, count in expected_dispositions):
            raise ValueError("parse disposition count is invalid")
        if sum(count for _, count in expected_dispositions) != self.parse_event_count:
            raise ValueError("parse disposition counts do not sum to parse events")
        if dict(expected_dispositions)[RETAIN] != self.eligible_record_count:
            raise ValueError("eligible records disagree with retained parse events")
        if not 0 <= self.distinct_eligible_native_id_count <= self.eligible_record_count:
            raise ValueError("distinct eligible native-ID count is invalid")
        if self.collision_ledger_rows != (
            self.exact_repeat_count
            + self.score_only_variance_count
            + self.content_divergence_count
        ):
            raise ValueError("collision ledger rows disagree with aggregate counts")
        if self.collision_ledger_rows != (
            self.eligible_record_count - self.distinct_eligible_native_id_count
        ):
            raise ValueError("collision ledger does not cover every repeated native ID")
        terminal_expected = self.status != GREEN_COMPLETE
        if terminal_expected != (self.terminal_evidence_sha256 is not None):
            raise ValueError("terminal evidence does not match audit status")
        if self.status == GREEN_COMPLETE:
            if (
                self.verified_asset_count != self.selected_asset_count
                or self.verified_compressed_bytes != self.selected_compressed_bytes
                or self.verified_asset_sequence_sha256
                != self.selected_asset_sequence_sha256
                or self.content_divergence_count
            ):
                raise ValueError("green audit does not prove complete clean coverage")
        elif self.content_divergence_count != 1:
            raise ValueError("content-divergence stop counts are invalid")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(AUDIT_SCHEMA_V1, self)


def _snapshot_file(source: Path, destination: Path) -> tuple[int, str]:
    """Copy and hash one exact source handle to close a manifest TOCTOU gap."""

    assert_no_symlink_ancestors(source)
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            writer.write(chunk)
            digest.update(chunk)
            count += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    return count, digest.hexdigest()


def _native_id_sha256(native_id: str) -> tuple[int, str]:
    encoded = native_id.encode("utf-8", errors="strict")
    return len(encoded), hashlib.sha256(_NATIVE_ID_DOMAIN + encoded).hexdigest()


def _occurrence(
    *,
    parsed: object,
    manifest_asset_order_ordinal: int,
) -> StackEduOccurrenceV1:
    canonical = parsed.canonical_record  # type: ignore[attr-defined]
    raw = parsed.raw_document  # type: ignore[attr-defined]
    native_id = canonical.native_record_id
    if not isinstance(native_id, str) or not native_id:
        raise StackEduCollisionAuditError("eligible StackEdu row lacks a native ID")
    if type(canonical.int_score) is not int or canonical.int_score < 3:
        raise StackEduCollisionAuditError("parser retained an ineligible StackEdu row")
    text = raw.text
    if isinstance(text, bytes):
        text_bytes = text
        text_bytes.decode("utf-8", errors="strict")
    elif isinstance(text, str):
        text_bytes = text.encode("utf-8", errors="strict")
    else:
        raise StackEduCollisionAuditError("StackEdu parser returned non-text content")
    native_bytes, native_sha = _native_id_sha256(native_id)
    return StackEduOccurrenceV1(
        native_id_sha256=native_sha,
        native_id_utf8_bytes=native_bytes,
        stable_source_record_id=raw.stable_source_record_id,
        source_cache_asset_identity_sha256=canonical.asset.asset_identity_sha256,
        manifest_asset_order_ordinal=manifest_asset_order_ordinal,
        source_record_ordinal=canonical.source_record_ordinal,
        text_sha256=hashlib.sha256(text_bytes).hexdigest(),
        text_utf8_bytes=len(text_bytes),
        int_score=canonical.int_score,
    )


def _stored_occurrence(row: sqlite3.Row) -> StackEduOccurrenceV1:
    return StackEduOccurrenceV1(
        native_id_sha256=str(row["native_id_sha256"]),
        native_id_utf8_bytes=int(row["native_id_utf8_bytes"]),
        stable_source_record_id=str(row["stable_source_record_id"]),
        source_cache_asset_identity_sha256=str(row["asset_identity_sha256"]),
        manifest_asset_order_ordinal=int(row["asset_ordinal"]),
        source_record_ordinal=int(row["record_ordinal"]),
        text_sha256=str(row["text_sha256"]),
        text_utf8_bytes=int(row["text_bytes"]),
        int_score=int(row["int_score"]),
    )


def _write_evidence(
    handle: BinaryIO,
    digest: object,
    evidence: StackEduCollisionEvidenceV1,
) -> tuple[int, str]:
    core = {
        "classification": evidence.classification,
        "first": asdict(evidence.first),
        "repeated": asdict(evidence.repeated),
        "schema": EVIDENCE_SCHEMA_V1,
    }
    evidence_sha = evidence.evidence_sha256
    payload = canonical_json_bytes({**core, "evidence_sha256": evidence_sha}) + b"\n"
    handle.write(payload)
    digest.update(payload)  # type: ignore[attr-defined]
    return len(payload), evidence_sha


def _asset_sequence_sha256(assets: tuple[object, ...]) -> str:
    return canonical_sha256(
        tuple(
            {
                "asset_identity_sha256": asset.asset_identity_sha256,
                "bytes": asset.bytes,
                "manifest_asset_order_ordinal": ordinal,
                "sha256": asset.sha256,
            }
            for ordinal, asset in enumerate(assets)
        )
    )


def _assert_stackedu_route(assets: tuple[object, ...]) -> str:
    routes = {
        route.source_family: route for route in load_source_route_manifest().routes
    }
    expected = routes["stackedu"]
    route_receipts: set[str] = set()
    for asset in assets:
        if (
            asset.repository,
            asset.config,
            asset.revision,
            asset.split,
        ) != (
            expected.repository,
            expected.config,
            expected.revision,
            expected.split,
        ):
            raise StackEduCollisionAuditError("StackEdu cache route drifted")
        if PurePosixPath(asset.relative_path).suffixes[-2:] != [".jsonl", ".zst"]:
            raise StackEduCollisionAuditError("StackEdu cache container drifted")
        route_receipts.add(asset.effective_route_receipt_sha256)
    if len(route_receipts) != 1:
        raise StackEduCollisionAuditError("StackEdu assets span effective routes")
    return next(iter(route_receipts))


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA trusted_schema = OFF;
        CREATE TABLE seen (
          stable_source_record_id TEXT PRIMARY KEY,
          native_id_sha256 TEXT NOT NULL,
          native_id_utf8_bytes INTEGER NOT NULL,
          asset_identity_sha256 TEXT NOT NULL,
          asset_ordinal INTEGER NOT NULL,
          record_ordinal INTEGER NOT NULL,
          text_sha256 TEXT NOT NULL,
          text_bytes INTEGER NOT NULL,
          int_score INTEGER NOT NULL
        ) WITHOUT ROWID, STRICT;
        """
    )
    connection.execute("BEGIN IMMEDIATE")


def _audit_to_temporary_artifacts(
    *,
    source_manifest_path: Path,
    cache_root: Path,
    work_directory: Path,
) -> tuple[StackEduCollisionAuditReceiptV1, Path]:
    manifest_snapshot = work_directory / "source-cache-manifest-v4.snapshot.json"
    manifest_bytes, manifest_artifact_sha256 = _snapshot_file(
        source_manifest_path, manifest_snapshot
    )
    if manifest_bytes < 1:
        raise StackEduCollisionAuditError("source manifest artifact is empty")
    manifest = load_source_manifest_artifact_v4(manifest_snapshot)
    assert_no_symlink_ancestors(cache_root)
    root = cache_root.resolve(strict=True)
    if not root.is_dir():
        raise StackEduCollisionAuditError("source cache root is not a directory")

    indexed_assets = tuple(
        (ordinal, asset)
        for ordinal, asset in enumerate(manifest.assets)
        if asset.source_family == "stackedu"
    )
    if not indexed_assets:
        raise StackEduCollisionAuditError("V4 manifest has no selected StackEdu assets")
    assets = tuple(asset for _, asset in indexed_assets)
    effective_route_receipt = _assert_stackedu_route(assets)
    selected_asset_sequence = _asset_sequence_sha256(assets)
    selected_bytes = sum(asset.bytes for asset in assets)
    parser_composite_identity = PRODUCTION_PARSER_COMPOSITE_IDENTITIES_V3[
        "stackedu"
    ]
    binding_by_asset = {}
    selected_binding_counts: Counter[str] = Counter()
    for asset in assets:
        verified = VerifiedLocalCacheAssetV3(
            expected=asset,
            observed_bytes=asset.bytes,
            observed_sha256=asset.sha256,
        )
        binding = resolve_production_parser_binding_v3(verified)
        binding_by_asset[asset.asset_identity_sha256] = binding
        selected_binding_counts[binding.binding_sha256] += 1

    ledger_path = work_directory / LEDGER_NAME_V1
    database_path = work_directory / "stackedu-native-id-seen-v1.sqlite3"
    disposition_counts = {name: 0 for name in PARSE_DISPOSITIONS}
    verified_asset_rows: list[object] = []
    verified_bytes = 0
    parse_events = 0
    eligible_records = 0
    distinct_ids = 0
    exact_repeats = 0
    score_variances = 0
    content_divergences = 0
    ledger_rows = 0
    ledger_bytes = 0
    ledger_digest = hashlib.sha256()
    verified_binding_counts: Counter[str] = Counter()
    status = GREEN_COMPLETE
    terminal_evidence_sha256: str | None = None

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        _initialize_database(connection)
        with ledger_path.open("xb") as ledger:
            stop = False
            for manifest_ordinal, asset in indexed_assets:
                verified = VerifiedLocalCacheAssetV3(
                    expected=asset,
                    observed_bytes=asset.bytes,
                    observed_sha256=asset.sha256,
                )
                binding = binding_by_asset[asset.asset_identity_sha256]
                # The frozen parser copies and hashes the exact cache handle into
                # an anonymous snapshot, validates the full zstd container, and
                # only then yields decompressed records.
                events = iter_source_asset_events_v3(
                    verified,
                    root,
                    binding=binding,
                )
                verified_asset_rows.append(asset)
                verified_bytes += asset.bytes
                verified_binding_counts[binding.binding_sha256] += 1
                for event in events:
                    parse_events += 1
                    disposition_counts[event.disposition] += 1
                    if event.disposition != RETAIN:
                        continue
                    if event.record is None:
                        raise StackEduCollisionAuditError(
                            "retained StackEdu event lacks a parsed record"
                        )
                    current = _occurrence(
                        parsed=event.record,
                        manifest_asset_order_ordinal=manifest_ordinal,
                    )
                    eligible_records += 1
                    existing = connection.execute(
                        "SELECT * FROM seen WHERE stable_source_record_id = ?",
                        (current.stable_source_record_id,),
                    ).fetchone()
                    if existing is None:
                        connection.execute(
                            "INSERT INTO seen VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                current.stable_source_record_id,
                                current.native_id_sha256,
                                current.native_id_utf8_bytes,
                                current.source_cache_asset_identity_sha256,
                                current.manifest_asset_order_ordinal,
                                current.source_record_ordinal,
                                current.text_sha256,
                                current.text_utf8_bytes,
                                current.int_score,
                            ),
                        )
                        distinct_ids += 1
                    else:
                        first = _stored_occurrence(existing)
                        if (
                            first.native_id_sha256 != current.native_id_sha256
                            or first.native_id_utf8_bytes
                            != current.native_id_utf8_bytes
                        ):
                            raise StackEduCollisionAuditError(
                                "stable source ID joins different native-ID hashes"
                            )
                        same_content = (
                            first.text_sha256 == current.text_sha256
                            and first.text_utf8_bytes == current.text_utf8_bytes
                        )
                        classification = (
                            EXACT_REPEAT
                            if same_content and first.int_score == current.int_score
                            else SCORE_ONLY_VARIANCE
                            if same_content
                            else CONTENT_DIVERGENCE
                        )
                        evidence = StackEduCollisionEvidenceV1(
                            classification=classification,
                            first=first,
                            repeated=current,
                        )
                        size, evidence_sha = _write_evidence(
                            ledger, ledger_digest, evidence
                        )
                        ledger_rows += 1
                        ledger_bytes += size
                        if classification == EXACT_REPEAT:
                            exact_repeats += 1
                        elif classification == SCORE_ONLY_VARIANCE:
                            # Production accepts this exact case: both scores
                            # already passed >=3, text bytes are identical, and
                            # the first occurrence remains the canonical row.
                            score_variances += 1
                        else:
                            content_divergences = 1
                            status = STOP_CONTENT_DIVERGENCE
                            terminal_evidence_sha256 = evidence_sha
                            stop = True
                            break
                    if eligible_records % 4096 == 0:
                        connection.commit()
                        connection.execute("BEGIN IMMEDIATE")
                if stop:
                    events.close()  # type: ignore[attr-defined]
                    break
            connection.commit()
            ledger.flush()
            os.fsync(ledger.fileno())
    finally:
        connection.close()

    verified_assets = tuple(verified_asset_rows)
    receipt = StackEduCollisionAuditReceiptV1(
        schema=AUDIT_SCHEMA_V1,
        status=status,
        source_manifest_artifact_sha256=manifest_artifact_sha256,
        source_manifest_receipt_sha256=manifest.receipt_sha256,
        execution_binding_sha256=manifest.execution_binding.receipt_sha256,
        effective_route_identity_sha256=manifest.effective_route_identity_sha256,
        stackedu_effective_route_receipt_sha256=effective_route_receipt,
        parser_composite_identity_sha256=parser_composite_identity,
        parser_binding_asset_counts=tuple(
            (
                binding_sha,
                selected_binding_counts[binding_sha],
                verified_binding_counts[binding_sha],
            )
            for binding_sha in sorted(selected_binding_counts)
        ),
        selected_asset_count=len(assets),
        selected_compressed_bytes=selected_bytes,
        selected_asset_sequence_sha256=selected_asset_sequence,
        verified_asset_count=len(verified_assets),
        verified_compressed_bytes=verified_bytes,
        verified_asset_sequence_sha256=_asset_sequence_sha256(verified_assets),
        parse_event_count=parse_events,
        parse_disposition_counts=tuple(
            (name, disposition_counts[name]) for name in sorted(PARSE_DISPOSITIONS)
        ),
        eligible_record_count=eligible_records,
        distinct_eligible_native_id_count=distinct_ids,
        exact_repeat_count=exact_repeats,
        score_only_variance_count=score_variances,
        content_divergence_count=content_divergences,
        collision_ledger_rows=ledger_rows,
        collision_ledger_bytes=ledger_bytes,
        collision_ledger_sha256=ledger_digest.hexdigest(),
        terminal_evidence_sha256=terminal_evidence_sha256,
    )
    return receipt, ledger_path


def _copy_exclusive(source: Path, destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            writer.write(chunk)
            digest.update(chunk)
            count += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    return count, digest.hexdigest()


def run_stackedu_collision_audit_v1(
    *,
    source_manifest_path: Path,
    cache_root: Path,
    work_root: Path,
    output_root: Path,
) -> StackEduCollisionAuditReceiptV1:
    """Run the bounded offline audit and exclusively mint its two artifacts."""

    for value, name in (
        (source_manifest_path, "source manifest"),
        (cache_root, "cache root"),
        (work_root, "work root"),
        (output_root, "output root"),
    ):
        if not isinstance(value, Path):
            raise TypeError(f"{name} must be pathlib.Path")
    assert_no_symlink_ancestors(work_root)
    work = work_root.resolve(strict=True)
    if not work.is_dir():
        raise StackEduCollisionAuditError("work root is not a directory")
    assert_no_symlink_ancestors(output_root)
    output = output_root.resolve(strict=False)
    if output.exists():
        raise StackEduCollisionAuditError("collision-audit output root must be fresh")
    if output == work or output in work.parents or work in output.parents:
        raise StackEduCollisionAuditError("audit work and output roots must be disjoint")

    with tempfile.TemporaryDirectory(prefix="weft1-stackedu-audit-", dir=work) as temporary:
        receipt, temporary_ledger = _audit_to_temporary_artifacts(
            source_manifest_path=source_manifest_path,
            cache_root=cache_root,
            work_directory=Path(temporary),
        )
        output.mkdir(parents=False)
        ledger_count, ledger_sha = _copy_exclusive(
            temporary_ledger, output / LEDGER_NAME_V1
        )
        if (
            ledger_count != receipt.collision_ledger_bytes
            or ledger_sha != receipt.collision_ledger_sha256
        ):
            raise StackEduCollisionAuditError("collision ledger changed during mint")
        payload = {
            "receipt": asdict(receipt),
            "receipt_sha256": receipt.receipt_sha256,
            "schema": RECEIPT_ARTIFACT_SCHEMA_V1,
        }
        raw = canonical_json_bytes(payload) + b"\n"
        with (output / RECEIPT_NAME_V1).open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    return receipt


def load_stackedu_collision_audit_v1(
    *, receipt_path: Path, collision_ledger_path: Path
) -> StackEduCollisionAuditReceiptV1:
    """Re-open and authenticate a minted hash-only audit and its ledger."""

    assert_no_symlink_ancestors(receipt_path)
    raw = receipt_path.read_bytes()
    try:
        envelope = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StackEduCollisionAuditError("collision receipt is not JSON") from error
    if raw != canonical_json_bytes(envelope) + b"\n" or not isinstance(
        envelope, Mapping
    ):
        raise StackEduCollisionAuditError("collision receipt is not canonical JSON")
    if set(envelope) != {"receipt", "receipt_sha256", "schema"} or envelope.get(
        "schema"
    ) != RECEIPT_ARTIFACT_SCHEMA_V1:
        raise StackEduCollisionAuditError("collision receipt envelope drifted")
    value = envelope.get("receipt")
    if not isinstance(value, Mapping) or set(value) != {
        field.name for field in fields(StackEduCollisionAuditReceiptV1)
    }:
        raise StackEduCollisionAuditError("collision receipt payload drifted")
    payload = dict(value)
    counts = payload.get("parse_disposition_counts")
    if not isinstance(counts, list) or any(
        not isinstance(row, list) or len(row) != 2 for row in counts
    ):
        raise StackEduCollisionAuditError("collision disposition counts drifted")
    payload["parse_disposition_counts"] = tuple(
        (str(row[0]), row[1]) for row in counts
    )
    binding_counts = payload.get("parser_binding_asset_counts")
    if not isinstance(binding_counts, list) or any(
        not isinstance(row, list) or len(row) != 3 for row in binding_counts
    ):
        raise StackEduCollisionAuditError("parser-binding counts drifted")
    payload["parser_binding_asset_counts"] = tuple(
        (str(row[0]), row[1], row[2]) for row in binding_counts
    )
    try:
        receipt = StackEduCollisionAuditReceiptV1(**payload)
    except (TypeError, ValueError) as error:
        raise StackEduCollisionAuditError("collision receipt is invalid") from error
    if envelope.get("receipt_sha256") != receipt.receipt_sha256:
        raise StackEduCollisionAuditError("collision receipt identity drifted")
    assert_no_symlink_ancestors(collision_ledger_path)
    ledger_digest = hashlib.sha256()
    ledger_bytes = 0
    row_count = 0
    classification_counts = {name: 0 for name in CLASSIFICATIONS}
    terminal_evidence_sha256: str | None = None
    with collision_ledger_path.open("rb") as handle:
        for line in handle:
            ledger_digest.update(line)
            ledger_bytes += len(line)
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise StackEduCollisionAuditError(
                    "collision ledger row is not JSON"
                ) from error
            if line != canonical_json_bytes(row) + b"\n":
                raise StackEduCollisionAuditError(
                    "collision ledger row is not canonical"
                )
            if not isinstance(row, Mapping) or set(row) != {
                "classification",
                "evidence_sha256",
                "first",
                "repeated",
                "schema",
            } or row.get("schema") != EVIDENCE_SCHEMA_V1:
                raise StackEduCollisionAuditError("collision ledger row schema drifted")
            occurrences: list[StackEduOccurrenceV1] = []
            occurrence_fields = {field.name for field in fields(StackEduOccurrenceV1)}
            for key in ("first", "repeated"):
                value = row.get(key)
                if not isinstance(value, Mapping) or set(value) != occurrence_fields:
                    raise StackEduCollisionAuditError(
                        "collision occurrence fields drifted"
                    )
                try:
                    occurrences.append(StackEduOccurrenceV1(**dict(value)))
                except (TypeError, ValueError) as error:
                    raise StackEduCollisionAuditError(
                        "collision occurrence is invalid"
                    ) from error
            try:
                evidence = StackEduCollisionEvidenceV1(
                    classification=str(row.get("classification")),
                    first=occurrences[0],
                    repeated=occurrences[1],
                )
            except (TypeError, ValueError) as error:
                raise StackEduCollisionAuditError(
                    "collision evidence is invalid"
                ) from error
            if row.get("evidence_sha256") != evidence.evidence_sha256:
                raise StackEduCollisionAuditError(
                    "collision evidence identity drifted"
                )
            classification_counts[evidence.classification] += 1
            if evidence.classification == CONTENT_DIVERGENCE:
                terminal_evidence_sha256 = evidence.evidence_sha256
            row_count += 1
    if (
        ledger_bytes != receipt.collision_ledger_bytes
        or ledger_digest.hexdigest() != receipt.collision_ledger_sha256
    ):
        raise StackEduCollisionAuditError("collision ledger identity drifted")
    if row_count != receipt.collision_ledger_rows:
        raise StackEduCollisionAuditError("collision ledger row count drifted")
    if (
        classification_counts[EXACT_REPEAT] != receipt.exact_repeat_count
        or classification_counts[SCORE_ONLY_VARIANCE]
        != receipt.score_only_variance_count
        or classification_counts[CONTENT_DIVERGENCE]
        != receipt.content_divergence_count
        or terminal_evidence_sha256 != receipt.terminal_evidence_sha256
    ):
        raise StackEduCollisionAuditError("collision ledger aggregates drifted")
    return receipt


__all__ = [
    "AUDIT_SCHEMA_V1",
    "CLASSIFICATIONS",
    "CONTENT_DIVERGENCE",
    "EXACT_REPEAT",
    "GREEN_COMPLETE",
    "LEDGER_NAME_V1",
    "RECEIPT_NAME_V1",
    "SCORE_ONLY_VARIANCE",
    "STATUSES",
    "STOP_CONTENT_DIVERGENCE",
    "StackEduCollisionAuditError",
    "StackEduCollisionAuditReceiptV1",
    "StackEduCollisionEvidenceV1",
    "StackEduOccurrenceV1",
    "load_stackedu_collision_audit_v1",
    "run_stackedu_collision_audit_v1",
]
