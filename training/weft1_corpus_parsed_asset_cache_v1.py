"""Immutable per-asset recovery segments for WEFT-1 source parsing.

This module integrates with the production materializer only at the explicit
per-asset recovery boundary.  It defines a fresh recovery domain for future
attempts and has no reader for the older r3 progress-only checkpoint format.

Each completed source asset is represented by one zstd-compressed canonical
JSONL segment plus one canonical receipt.  The JSONL rows are the existing
``_canonical_spool_event`` representation, so retained text, explicit drops,
record observations, and the exact source-event digest are preserved without
inventing a second parser-output schema.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Iterable, Iterator, Mapping
import uuid

import zstandard

from training.weft1_corpus_pa import RawDocumentV3
from training.weft1_corpus_source_io_a2 import (
    FINEWEB_SELECTED_SCHEMA_CENSUS_PHYSICAL_SHA256_V1,
    PARSE_DISPOSITIONS,
    PRODUCTION_PARSER_BINDINGS_V3,
    RETAIN,
    STACKEDU_PYTHON_PARSER_BINDING_V3,
    ParsedSourceRecordV3,
    SourceParseEventV3,
    SourceParserBindingV3,
    SourceRecordObservationV3,
    _canonical_spool_event,
)
from training.weft1_corpus_sources_a2 import (
    CanonicalSourceRecordV3,
    SourceCacheAssetV3,
    VerifiedLocalCacheAssetV3,
)
from training.weft1_gtok_contract import canonical_json_bytes, canonical_sha256
from training.weft1_strict_io import assert_no_symlink_ancestors


PARSED_ASSET_RECOVERY_DOMAIN_V1 = (
    "WEFT1_PARSED_ASSET_CACHE_V1_FRESH_ONLY_NO_R3_IMPORT"
)
PARSED_ASSET_SEGMENT_SCHEMA_V1 = "weft1_parsed_asset_segment_v1"
PARSED_ASSET_RECEIPT_SCHEMA_V1 = "weft1_parsed_asset_segment_receipt_v1"
PARSED_ASSET_RECEIPT_ARTIFACT_SCHEMA_V1 = (
    "weft1_parsed_asset_segment_receipt_artifact_v1"
)
PARSED_ASSET_RUNTIME_IDENTITY_SCHEMA_V1 = (
    "weft1_parsed_asset_parser_runtime_identity_v1"
)
PARSED_ASSET_COMPATIBILITY_POLICY_SCHEMA_V1 = (
    "weft1_parsed_asset_code_compatibility_policy_v1"
)
PARSED_ASSET_COMPATIBILITY_POLICY_ARTIFACT_SCHEMA_V1 = (
    "weft1_parsed_asset_code_compatibility_policy_artifact_v1"
)
PARSED_ASSET_INCIDENT_AUTHORITY_SCHEMA_V1 = (
    "weft1_pa_schema_remediation_incident_authority_v1"
)
PARSED_ASSET_INCIDENT_AUTHORITY_PATH_V1 = Path(__file__).with_name(
    "weft1_pa_schema_remediation_incident_authority_20260904.json"
)
PARSED_ASSET_INCIDENT_AUTHORITY_PHYSICAL_BYTES_V1 = 2_562
PARSED_ASSET_INCIDENT_AUTHORITY_PHYSICAL_SHA256_V1 = (
    "0dd25287938bca429d1ab60e0743dc2d8811f0a0add24abcf5a2d606f0be0e7b"
)
PARSED_ASSET_INCIDENT_AUTHORITY_RECEIPT_SHA256_V1 = (
    "c0a9a0730a6f1f42a59ed1b0096b4cc533cbfe8bc209e57d138e165bb12ce7b2"
)
PARSED_ASSET_INCIDENT_ELIGIBLE_RUN_ID_V1 = "production-v4-replay-a"
PARSED_ASSET_INCIDENT_PREDECESSOR_CODE_IDENTITY_SHA256_V1 = (
    "89a8b42dbe06edad2db7c67ae126c779a356612a3ed9e94587a98befb0d94657"
)
PARSED_ASSET_INCIDENT_PREDECESSOR_ASSET_COUNT_V1 = 394
PARSED_ASSET_INCIDENT_CURRENT_ASSET_COUNT_V1 = 3
PARSED_ASSET_COMPOSITE_BRIDGE_SCHEMA_V1 = (
    "weft1_parsed_asset_composite_bridge_v1"
)
PARSED_ASSET_COMPOSITE_BRIDGE_ARTIFACT_SCHEMA_V1 = (
    "weft1_parsed_asset_composite_bridge_artifact_v1"
)
PARSED_ASSET_COMPOSITE_BRIDGE_FILENAME_V1 = (
    "parsed-asset-composite-bridge-v1.json"
)
CURRENT_CONTEXT_RESOLUTION_V1 = "CURRENT_CONTEXT"
READ_ONLY_PREDECESSOR_RESOLUTION_V1 = "READ_ONLY_PREDECESSOR"
PARSED_ASSET_RESOLUTIONS_V1 = (
    CURRENT_CONTEXT_RESOLUTION_V1,
    READ_ONLY_PREDECESSOR_RESOLUTION_V1,
)
PARSED_ASSET_SOURCE_FAMILIES_V1 = frozenset(
    {
        "arxiv",
        "dolma_web",
        "finemath_3plus",
        "fineweb_edu",
        "olmocr",
        "stackedu",
        "wikipedia_wikibooks",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPOOL_ROW_KEYS = frozenset(
    {
        "asset_order_ordinal",
        "disposition",
        "event_ordinal",
        "event_sha256",
        "observation",
        "parser_binding_sha256",
        "reason",
        "retained",
        "source_asset_identity_sha256",
        "source_asset_sha256",
        "source_family",
        "source_record_ordinal",
    }
)
_RETAINED_KEYS = frozenset(
    {
        "canonical_record",
        "parser_binding_sha256",
        "raw_document",
        "text_utf8_bytes",
        "text_utf8_sha256",
    }
)
_RAW_DOCUMENT_KEYS = frozenset(
    {"source", "stable_source_record_id", "stratum", "text"}
)
_CANONICAL_RECORD_REQUIRED_KEYS = frozenset(
    {
        "asset",
        "int_score",
        "native_record_id",
        "retained_byte_count",
        "source_record_ordinal",
    }
)
_CANONICAL_RECORD_OPTIONAL_KEYS = frozenset({"native_record_namespace"})
_OBSERVATION_KEYS = frozenset(field.name for field in fields(SourceRecordObservationV3))


class ParsedAssetRecoveryError(RuntimeError):
    """A recovery segment or its identity envelope failed closed."""


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive exact integer")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    if frozenset(value) != expected:
        raise ParsedAssetRecoveryError(f"{name} fields are not exact")


def _json_no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ParsedAssetRecoveryError("canonical JSON repeats a key")
        value[key] = item
    return value


def _load_canonical_json(raw: bytes, *, name: str) -> Mapping[str, object]:
    if not raw.endswith(b"\n") or raw == b"\n":
        raise ParsedAssetRecoveryError(f"{name} has invalid newline framing")
    try:
        value = json.loads(
            raw[:-1].decode("utf-8", errors="strict"),
            object_pairs_hook=_json_no_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ParsedAssetRecoveryError(
                    f"{name} contains a non-finite JSON constant {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ParsedAssetRecoveryError(f"{name} is not strict JSON") from error
    if not isinstance(value, Mapping):
        raise ParsedAssetRecoveryError(f"{name} is not a JSON object")
    if canonical_json_bytes(value) + b"\n" != raw:
        raise ParsedAssetRecoveryError(f"{name} is not canonical JSON")
    return value


def _validate_parsed_asset_incident_authority_v1() -> Mapping[str, object]:
    assert_no_symlink_ancestors(PARSED_ASSET_INCIDENT_AUTHORITY_PATH_V1)
    try:
        raw = PARSED_ASSET_INCIDENT_AUTHORITY_PATH_V1.read_bytes()
    except OSError as error:
        raise ParsedAssetRecoveryError(
            "parsed-asset incident authority is unavailable"
        ) from error
    value = _load_canonical_json(raw, name="parsed-asset incident authority")
    if (
        len(raw) != PARSED_ASSET_INCIDENT_AUTHORITY_PHYSICAL_BYTES_V1
        or hashlib.sha256(raw).hexdigest()
        != PARSED_ASSET_INCIDENT_AUTHORITY_PHYSICAL_SHA256_V1
        or value.get("schema") != PARSED_ASSET_INCIDENT_AUTHORITY_SCHEMA_V1
        or value.get("receipt_sha256")
        != PARSED_ASSET_INCIDENT_AUTHORITY_RECEIPT_SHA256_V1
        or value.get("fineweb_schema_census_physical_sha256")
        != FINEWEB_SELECTED_SCHEMA_CENSUS_PHYSICAL_SHA256_V1
    ):
        raise ParsedAssetRecoveryError(
            "parsed-asset incident authority identity drifted"
        )
    receipt_payload = dict(value)
    receipt_payload.pop("receipt_sha256")
    if canonical_sha256(
        {
            "payload": receipt_payload,
            "schema": PARSED_ASSET_INCIDENT_AUTHORITY_SCHEMA_V1,
        }
    ) != PARSED_ASSET_INCIDENT_AUTHORITY_RECEIPT_SHA256_V1:
        raise ParsedAssetRecoveryError(
            "parsed-asset incident authority semantic identity drifted"
        )
    incident = value.get("incident")
    if (
        not isinstance(incident, Mapping)
        or incident.get("eligible_run_id")
        != PARSED_ASSET_INCIDENT_ELIGIBLE_RUN_ID_V1
        or incident.get("predecessor_code_identity_sha256")
        != PARSED_ASSET_INCIDENT_PREDECESSOR_CODE_IDENTITY_SHA256_V1
        or incident.get("predecessor_verified_segment_count")
        != PARSED_ASSET_INCIDENT_PREDECESSOR_ASSET_COUNT_V1
        or incident.get("fineweb_selected_asset_count")
        != PARSED_ASSET_INCIDENT_CURRENT_ASSET_COUNT_V1
        or value.get("expected_predecessor_asset_count")
        != PARSED_ASSET_INCIDENT_PREDECESSOR_ASSET_COUNT_V1
        or value.get("expected_current_asset_count")
        != PARSED_ASSET_INCIDENT_CURRENT_ASSET_COUNT_V1
    ):
        raise ParsedAssetRecoveryError(
            "parsed-asset incident scope literals drifted"
        )
    return value


PARSED_ASSET_INCIDENT_AUTHORITY_V1 = (
    _validate_parsed_asset_incident_authority_v1()
)


def _hash_open_file(handle: object) -> tuple[int, str]:
    if not hasattr(handle, "read") or not hasattr(handle, "seek"):
        raise TypeError("hashing requires a seekable binary handle")
    handle.seek(0)  # type: ignore[attr-defined]
    digest = hashlib.sha256()
    byte_count = 0
    while True:
        chunk = handle.read(8 * 1024 * 1024)  # type: ignore[attr-defined]
        if not chunk:
            break
        byte_count += len(chunk)
        digest.update(chunk)
    handle.seek(0)  # type: ignore[attr-defined]
    return byte_count, digest.hexdigest()


def _hash_path(path: Path) -> tuple[int, str]:
    assert_no_symlink_ancestors(path)
    with path.open("rb") as handle:
        return _hash_open_file(handle)


def _fsync_directory(path: Path) -> str:
    unsupported = {errno.EINVAL}
    if hasattr(errno, "ENOTSUP"):
        unsupported.add(errno.ENOTSUP)
    if hasattr(errno, "EOPNOTSUPP"):
        unsupported.add(errno.EOPNOTSUPP)
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(str(path), flags)
    except OSError as error:
        if error.errno in unsupported or (
            os.name == "nt" and error.errno == errno.EACCES
        ):
            return f"UNSUPPORTED_{os.name.upper()}_ERRNO_{error.errno}"
        raise ParsedAssetRecoveryError("directory open for fsync failed") from error
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno in unsupported:
                return f"UNSUPPORTED_{os.name.upper()}_ERRNO_{error.errno}"
            raise ParsedAssetRecoveryError("directory fsync failed") from error
    finally:
        os.close(descriptor)
    return "SUPPORTED"


@dataclass(frozen=True)
class ParsedAssetRecoveryContextV1:
    """Stable replay-lane identities that survive a legitimate backend retry."""

    run_id: str
    durable_marker_physical_sha256: str
    runtime_identity_sha256: str
    code_identity_sha256: str
    input_identity_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.run_id) is None
        ):
            raise ValueError("recovery run_id is not a canonical identifier")
        _require_sha256(
            self.durable_marker_physical_sha256,
            "durable marker physical identity",
        )
        _require_sha256(self.runtime_identity_sha256, "runtime identity")
        _require_sha256(self.code_identity_sha256, "code identity")
        _require_sha256(self.input_identity_sha256, "input identity")

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "payload": self,
                "schema": "weft1_parsed_asset_recovery_context_v1",
            }
        )


@dataclass(frozen=True)
class ParsedAssetCompatibilityPolicyV1:
    """One incident-scoped, non-transitive V1-to-V1 compatibility grant."""

    schema: str
    authority_sha256: str
    eligible_run_id: str
    predecessor_code_identity_sha256: str
    successor_code_identity_sha256: str
    compatible_parser_bindings: tuple[tuple[str, str], ...]
    excluded_source_families: tuple[str, ...]
    expected_predecessor_asset_count: int
    expected_current_asset_count: int

    def __post_init__(self) -> None:
        if self.schema != PARSED_ASSET_COMPATIBILITY_POLICY_SCHEMA_V1:
            raise ValueError("parsed-asset compatibility policy schema is unknown")
        for value, name in (
            (self.authority_sha256, "compatibility authority"),
            (self.predecessor_code_identity_sha256, "predecessor code identity"),
            (self.successor_code_identity_sha256, "successor code identity"),
        ):
            _require_sha256(value, name)
        if self.predecessor_code_identity_sha256 == self.successor_code_identity_sha256:
            raise ValueError("compatibility policy requires distinct code identities")
        if (
            self.authority_sha256
            != PARSED_ASSET_INCIDENT_AUTHORITY_PHYSICAL_SHA256_V1
        ):
            raise ValueError(
                "compatibility policy is not bound to the incident authority"
            )
        if (
            not isinstance(self.eligible_run_id, str)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.eligible_run_id
            )
            is None
        ):
            raise ValueError("compatibility policy run ID is invalid")
        if (
            self.eligible_run_id != PARSED_ASSET_INCIDENT_ELIGIBLE_RUN_ID_V1
            or self.predecessor_code_identity_sha256
            != PARSED_ASSET_INCIDENT_PREDECESSOR_CODE_IDENTITY_SHA256_V1
            or self.expected_predecessor_asset_count
            != PARSED_ASSET_INCIDENT_PREDECESSOR_ASSET_COUNT_V1
            or self.expected_current_asset_count
            != PARSED_ASSET_INCIDENT_CURRENT_ASSET_COUNT_V1
        ):
            raise ValueError(
                "compatibility policy differs from the exact incident scope"
            )
        if (
            tuple(sorted(self.compatible_parser_bindings))
            != self.compatible_parser_bindings
            or len(set(self.compatible_parser_bindings))
            != len(self.compatible_parser_bindings)
            or any(
                not isinstance(source, str)
                or not source
                or _SHA256.fullmatch(binding_sha256) is None
                for source, binding_sha256 in self.compatible_parser_bindings
            )
        ):
            raise ValueError("compatible parser bindings are not canonical")
        if (
            tuple(sorted(self.excluded_source_families))
            != self.excluded_source_families
            or len(set(self.excluded_source_families))
            != len(self.excluded_source_families)
            or any(not isinstance(source, str) or not source for source in self.excluded_source_families)
        ):
            raise ValueError("excluded source families are not canonical")
        compatible_sources = {source for source, unused in self.compatible_parser_bindings}
        if compatible_sources.intersection(self.excluded_source_families):
            raise ValueError("a source cannot be both compatible and excluded")
        if (
            compatible_sources.union(self.excluded_source_families)
            != PARSED_ASSET_SOURCE_FAMILIES_V1
            or self.excluded_source_families != ("fineweb_edu",)
        ):
            raise ValueError(
                "compatibility policy must cover every source and exclude only FineWeb"
            )
        expected_bindings = {
            (source, binding.binding_sha256)
            for source, binding in PRODUCTION_PARSER_BINDINGS_V3.items()
            if source != "fineweb_edu"
        }
        expected_bindings.add(
            ("stackedu", STACKEDU_PYTHON_PARSER_BINDING_V3.binding_sha256)
        )
        if set(self.compatible_parser_bindings) != expected_bindings:
            raise ValueError(
                "compatibility policy parser bindings differ from unchanged code"
            )
        _require_nonnegative_int(
            self.expected_predecessor_asset_count,
            "expected predecessor asset count",
        )
        _require_nonnegative_int(
            self.expected_current_asset_count,
            "expected current asset count",
        )

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {"payload": self, "schema": PARSED_ASSET_COMPATIBILITY_POLICY_SCHEMA_V1}
        )

    def permits(self, source_family: str, parser_binding_sha256: str) -> bool:
        if source_family in self.excluded_source_families:
            return False
        return (source_family, parser_binding_sha256) in self.compatible_parser_bindings

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ParsedAssetCompatibilityPolicyV1":
        expected = frozenset(field.name for field in fields(cls))
        _require_exact_keys(value, expected, "parsed-asset compatibility policy")
        bindings = value["compatible_parser_bindings"]
        excluded = value["excluded_source_families"]
        if not isinstance(bindings, list) or not isinstance(excluded, list):
            raise ParsedAssetRecoveryError("compatibility policy arrays are malformed")
        binding_rows: list[tuple[str, str]] = []
        for row in bindings:
            if (
                not isinstance(row, list)
                or len(row) != 2
                or not all(isinstance(item, str) for item in row)
            ):
                raise ParsedAssetRecoveryError("compatibility binding row is malformed")
            binding_rows.append((row[0], row[1]))
        if any(not isinstance(item, str) for item in excluded):
            raise ParsedAssetRecoveryError("compatibility exclusion row is malformed")
        payload = dict(value)
        payload["compatible_parser_bindings"] = tuple(binding_rows)
        payload["excluded_source_families"] = tuple(excluded)
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ParsedAssetRecoveryError("compatibility policy is invalid") from error


def validate_compatible_recovery_contexts_v1(
    *,
    current: ParsedAssetRecoveryContextV1,
    predecessor: ParsedAssetRecoveryContextV1,
    policy: ParsedAssetCompatibilityPolicyV1,
) -> None:
    """Require exact lane/input/runtime/storage identity across a code-only bridge."""

    if not all(
        isinstance(value, ParsedAssetRecoveryContextV1)
        for value in (current, predecessor)
    ) or not isinstance(policy, ParsedAssetCompatibilityPolicyV1):
        raise TypeError("compatible recovery contexts require typed values")
    if current.run_id != policy.eligible_run_id:
        raise ParsedAssetRecoveryError("compatibility policy cannot cross replay lanes")
    expected_predecessor = ParsedAssetRecoveryContextV1(
        run_id=current.run_id,
        durable_marker_physical_sha256=current.durable_marker_physical_sha256,
        runtime_identity_sha256=current.runtime_identity_sha256,
        code_identity_sha256=policy.predecessor_code_identity_sha256,
        input_identity_sha256=current.input_identity_sha256,
    )
    if (
        current.code_identity_sha256 != policy.successor_code_identity_sha256
        or predecessor != expected_predecessor
    ):
        raise ParsedAssetRecoveryError(
            "compatibility contexts differ by more than the registered code identity"
        )


@dataclass(frozen=True)
class ParsedAssetCompositeBridgeRowV1:
    source_family: str
    asset_order_ordinal: int
    source_asset_identity_sha256: str
    source_asset_sha256: str
    parser_binding_sha256: str
    first_event_ordinal: int
    next_event_ordinal: int
    resolution: str
    selected_context_identity_sha256: str
    selected_code_identity_sha256: str
    segment_relative_path: str
    segment_physical_bytes: int
    segment_physical_sha256: str
    segment_receipt_sha256: str
    segment_receipt_physical_bytes: int
    segment_receipt_physical_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_family, str) or not self.source_family:
            raise ValueError("composite bridge row requires a source family")
        _require_nonnegative_int(self.asset_order_ordinal, "asset order ordinal")
        for value, name in (
            (self.source_asset_identity_sha256, "source asset identity"),
            (self.source_asset_sha256, "source asset SHA-256"),
            (self.parser_binding_sha256, "parser binding"),
            (self.selected_context_identity_sha256, "selected context identity"),
            (self.selected_code_identity_sha256, "selected code identity"),
            (self.segment_physical_sha256, "segment physical identity"),
            (self.segment_receipt_sha256, "segment receipt identity"),
            (
                self.segment_receipt_physical_sha256,
                "segment receipt physical identity",
            ),
        ):
            _require_sha256(value, name)
        _require_nonnegative_int(self.first_event_ordinal, "first event ordinal")
        _require_nonnegative_int(self.next_event_ordinal, "next event ordinal")
        if self.next_event_ordinal < self.first_event_ordinal:
            raise ValueError("composite bridge event interval is reversed")
        if self.resolution not in PARSED_ASSET_RESOLUTIONS_V1:
            raise ValueError("composite bridge resolution is unknown")
        relative = PurePosixPath(self.segment_relative_path)
        if (
            "\\" in self.segment_relative_path
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.suffixes[-3:] != [".parsed", ".jsonl", ".zst"]
        ):
            raise ValueError("composite bridge segment path is not canonical")
        _require_positive_int(self.segment_physical_bytes, "segment physical bytes")
        _require_positive_int(
            self.segment_receipt_physical_bytes,
            "segment receipt physical bytes",
        )


@dataclass(frozen=True)
class ParsedAssetCompositeBridgeV1:
    schema: str
    recovery_domain: str
    current_context: ParsedAssetRecoveryContextV1
    predecessor_context: ParsedAssetRecoveryContextV1 | None
    compatibility_policy_sha256: str | None
    rows: tuple[ParsedAssetCompositeBridgeRowV1, ...]
    current_asset_count: int
    predecessor_asset_count: int

    def __post_init__(self) -> None:
        if self.schema != PARSED_ASSET_COMPOSITE_BRIDGE_SCHEMA_V1:
            raise ValueError("parsed-asset composite bridge schema is unknown")
        if self.recovery_domain != PARSED_ASSET_RECOVERY_DOMAIN_V1:
            raise ValueError("composite bridge is outside the fresh V1 domain")
        if not isinstance(self.current_context, ParsedAssetRecoveryContextV1):
            raise TypeError("composite bridge requires a current context")
        if (self.predecessor_context is None) != (
            self.compatibility_policy_sha256 is None
        ):
            raise ValueError("composite bridge predecessor and policy must be paired")
        if self.predecessor_context is not None and not isinstance(
            self.predecessor_context, ParsedAssetRecoveryContextV1
        ):
            raise TypeError("composite bridge predecessor context is untyped")
        if self.compatibility_policy_sha256 is not None:
            _require_sha256(
                self.compatibility_policy_sha256,
                "compatibility policy identity",
            )
        if type(self.rows) is not tuple or any(
            type(row) is not ParsedAssetCompositeBridgeRowV1 for row in self.rows
        ):
            raise TypeError("composite bridge rows must be an exact typed tuple")
        if not self.rows:
            raise ValueError("composite bridge requires at least one asset row")
        keys = tuple((row.source_family, row.asset_order_ordinal) for row in self.rows)
        if len(set(keys)) != len(keys):
            raise ValueError("composite bridge repeats an asset row")
        if keys != tuple(sorted(keys)):
            raise ValueError("composite bridge rows are not in canonical order")
        _require_nonnegative_int(self.current_asset_count, "current asset count")
        _require_nonnegative_int(
            self.predecessor_asset_count, "predecessor asset count"
        )
        if self.current_asset_count + self.predecessor_asset_count != len(self.rows):
            raise ValueError("composite bridge asset counts do not cover rows")
        if self.current_asset_count != sum(
            row.resolution == CURRENT_CONTEXT_RESOLUTION_V1 for row in self.rows
        ) or self.predecessor_asset_count != sum(
            row.resolution == READ_ONLY_PREDECESSOR_RESOLUTION_V1 for row in self.rows
        ):
            raise ValueError("composite bridge resolution counts drifted")
        if self.predecessor_context is None and self.predecessor_asset_count != 0:
            raise ValueError("ordinary bridge cannot select predecessor assets")
        if self.predecessor_context is not None and self.predecessor_asset_count < 1:
            raise ValueError("compatibility bridge must select predecessor assets")
        current_identity = self.current_context.identity_sha256
        predecessor_identity = (
            None
            if self.predecessor_context is None
            else self.predecessor_context.identity_sha256
        )
        for row in self.rows:
            expected_context = (
                current_identity
                if row.resolution == CURRENT_CONTEXT_RESOLUTION_V1
                else predecessor_identity
            )
            expected_code = (
                self.current_context.code_identity_sha256
                if row.resolution == CURRENT_CONTEXT_RESOLUTION_V1
                else (
                    None
                    if self.predecessor_context is None
                    else self.predecessor_context.code_identity_sha256
                )
            )
            if (
                row.selected_context_identity_sha256 != expected_context
                or row.selected_code_identity_sha256 != expected_code
            ):
                raise ValueError("composite bridge row selected a foreign context")

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(
            {"payload": self, "schema": PARSED_ASSET_COMPOSITE_BRIDGE_SCHEMA_V1}
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ParsedAssetCompositeBridgeV1":
        expected = frozenset(field.name for field in fields(cls))
        _require_exact_keys(value, expected, "parsed-asset composite bridge")
        current = value["current_context"]
        predecessor = value["predecessor_context"]
        rows = value["rows"]
        if not isinstance(current, Mapping) or not isinstance(rows, list):
            raise ParsedAssetRecoveryError("composite bridge fields are malformed")
        if predecessor is not None and not isinstance(predecessor, Mapping):
            raise ParsedAssetRecoveryError("composite predecessor context is malformed")
        context_keys = frozenset(field.name for field in fields(ParsedAssetRecoveryContextV1))
        _require_exact_keys(current, context_keys, "composite current context")
        if predecessor is not None:
            _require_exact_keys(predecessor, context_keys, "composite predecessor context")
        row_keys = frozenset(field.name for field in fields(ParsedAssetCompositeBridgeRowV1))
        typed_rows: list[ParsedAssetCompositeBridgeRowV1] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ParsedAssetRecoveryError("composite bridge row is not an object")
            _require_exact_keys(row, row_keys, "composite bridge row")
            try:
                typed_rows.append(ParsedAssetCompositeBridgeRowV1(**dict(row)))  # type: ignore[arg-type]
            except (TypeError, ValueError) as error:
                raise ParsedAssetRecoveryError("composite bridge row is invalid") from error
        payload = dict(value)
        payload["current_context"] = ParsedAssetRecoveryContextV1(**dict(current))
        payload["predecessor_context"] = (
            None
            if predecessor is None
            else ParsedAssetRecoveryContextV1(**dict(predecessor))
        )
        payload["rows"] = tuple(typed_rows)
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ParsedAssetRecoveryError("composite bridge is invalid") from error


@dataclass(frozen=True)
class ParsedAssetSegmentReceiptV1:
    schema: str
    recovery_domain: str
    segment_schema: str
    context: ParsedAssetRecoveryContextV1
    source_family: str
    asset_order_ordinal: int
    source_asset_identity_sha256: str
    source_asset_sha256: str
    parser_binding_sha256: str
    segment_relative_path: str
    segment_physical_bytes: int
    segment_physical_sha256: str
    logical_jsonl_bytes: int
    logical_jsonl_sha256: str
    event_count: int
    retained_record_count: int
    drop_counts: tuple[tuple[str, int], ...]
    observation_count: int
    first_event_ordinal: int
    next_event_ordinal: int
    first_source_record_ordinal: int
    next_source_record_ordinal: int
    directory_fsync: str

    def __post_init__(self) -> None:
        if self.schema != PARSED_ASSET_RECEIPT_SCHEMA_V1:
            raise ValueError("parsed-asset receipt uses an unknown schema")
        if self.recovery_domain != PARSED_ASSET_RECOVERY_DOMAIN_V1:
            raise ValueError("parsed-asset receipt is outside the fresh V1 domain")
        if self.segment_schema != PARSED_ASSET_SEGMENT_SCHEMA_V1:
            raise ValueError("parsed-asset receipt uses an unknown segment schema")
        if not isinstance(self.context, ParsedAssetRecoveryContextV1):
            raise TypeError("parsed-asset receipt requires a typed context")
        if not isinstance(self.source_family, str) or not self.source_family:
            raise ValueError("parsed-asset receipt requires a source family")
        _require_nonnegative_int(self.asset_order_ordinal, "asset order ordinal")
        for value, name in (
            (self.source_asset_identity_sha256, "source asset identity"),
            (self.source_asset_sha256, "source asset SHA-256"),
            (self.parser_binding_sha256, "parser binding"),
            (self.segment_physical_sha256, "segment physical SHA-256"),
            (self.logical_jsonl_sha256, "logical JSONL SHA-256"),
        ):
            _require_sha256(value, name)
        relative = PurePosixPath(self.segment_relative_path)
        if (
            "\\" in self.segment_relative_path
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.suffixes[-3:] != [".parsed", ".jsonl", ".zst"]
        ):
            raise ValueError("parsed-asset segment path is not canonical relative POSIX")
        _require_positive_int(self.segment_physical_bytes, "segment physical bytes")
        _require_nonnegative_int(self.logical_jsonl_bytes, "logical JSONL bytes")
        _require_nonnegative_int(self.event_count, "event count")
        _require_nonnegative_int(self.retained_record_count, "retained record count")
        _require_nonnegative_int(self.observation_count, "observation count")
        if self.observation_count > min(1, self.event_count):
            raise ValueError("observation count exceeds the parser's per-asset bound")
        expected_drop_names = tuple(
            disposition for disposition in PARSE_DISPOSITIONS if disposition != RETAIN
        )
        if tuple(name for name, unused in self.drop_counts) != expected_drop_names:
            raise ValueError("parsed-asset drop counts are not canonical")
        if any(type(count) is not int or count < 0 for unused, count in self.drop_counts):
            raise ValueError("parsed-asset drop count is invalid")
        if self.retained_record_count + sum(
            count for unused, count in self.drop_counts
        ) != self.event_count:
            raise ValueError("parsed-asset disposition counts do not cover events")
        _require_nonnegative_int(self.first_event_ordinal, "first event ordinal")
        _require_nonnegative_int(self.next_event_ordinal, "next event ordinal")
        if self.next_event_ordinal != self.first_event_ordinal + self.event_count:
            raise ValueError("parsed-asset event interval is inconsistent")
        if self.first_source_record_ordinal != 0:
            raise ValueError("parsed-asset source records must begin at zero")
        if self.next_source_record_ordinal != self.event_count:
            raise ValueError("parsed-asset source-record interval is inconsistent")
        if self.directory_fsync != "SUPPORTED" and re.fullmatch(
            r"UNSUPPORTED_[A-Z]+_ERRNO_[0-9]+", self.directory_fsync
        ) is None:
            raise ValueError("parsed-asset directory-fsync status is invalid")

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(
            {"payload": self, "schema": PARSED_ASSET_RECEIPT_SCHEMA_V1}
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ParsedAssetSegmentReceiptV1":
        expected = frozenset(field.name for field in fields(cls))
        _require_exact_keys(value, expected, "parsed-asset receipt")
        context = value["context"]
        if not isinstance(context, Mapping):
            raise ParsedAssetRecoveryError("parsed-asset context is not an object")
        context_keys = frozenset(field.name for field in fields(ParsedAssetRecoveryContextV1))
        _require_exact_keys(context, context_keys, "parsed-asset context")
        drops = value["drop_counts"]
        if not isinstance(drops, list):
            raise ParsedAssetRecoveryError("parsed-asset drop counts are not an array")
        drop_rows: list[tuple[str, int]] = []
        for row in drops:
            if (
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], str)
                or type(row[1]) is not int
            ):
                raise ParsedAssetRecoveryError("parsed-asset drop row is malformed")
            drop_rows.append((row[0], row[1]))
        payload = dict(value)
        payload["context"] = ParsedAssetRecoveryContextV1(**dict(context))
        payload["drop_counts"] = tuple(drop_rows)
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ParsedAssetRecoveryError("parsed-asset receipt is invalid") from error


@dataclass(frozen=True)
class PublishedParsedAssetSegmentV1:
    receipt: ParsedAssetSegmentReceiptV1
    segment_path: Path
    receipt_path: Path
    receipt_physical_bytes: int
    receipt_physical_sha256: str


@dataclass(frozen=True)
class RecoveredSourceParseEventV1:
    """One validated event with direct legacy-ledger and SQLite projections."""

    asset_order_ordinal: int
    event_ordinal: int
    source_asset_identity_sha256: str
    source_asset_sha256: str
    parser_binding_sha256: str
    event: SourceParseEventV3

    @property
    def ledger_payload(self) -> Mapping[str, object]:
        return {
            "asset_order_ordinal": self.asset_order_ordinal,
            "disposition": self.event.disposition,
            "event_ordinal": self.event_ordinal,
            "event_sha256": self.event.event_sha256,
            "source_asset_identity_sha256": self.source_asset_identity_sha256,
            "source_family": self.event.source_family,
            "source_record_ordinal": self.event.source_record_ordinal,
        }

    @property
    def sqlite_insert_fields(self) -> Mapping[str, object] | None:
        if self.event.disposition != RETAIN:
            return None
        assert self.event.record is not None
        parsed = self.event.record
        text = parsed.raw_document.text
        text_bytes = text if isinstance(text, bytes) else text.encode("utf-8", errors="strict")
        return {
            "source": self.event.source_family,
            "stable_source_record_id": parsed.raw_document.stable_source_record_id,
            "source_asset_identity_sha256": self.source_asset_identity_sha256,
            "asset_order_ordinal": self.asset_order_ordinal,
            "asset_record_ordinal": self.event.source_record_ordinal,
            "text_bytes": text_bytes,
            "retained_bytes": parsed.canonical_record.retained_byte_count,
            "int_score": parsed.canonical_record.int_score,
        }


def _segment_relative_path(
    *,
    context_identity_sha256: str,
    source_family: str,
    asset_order_ordinal: int,
    source_asset_identity_sha256: str,
) -> str:
    return (
        f"{context_identity_sha256[:24]}/{source_family}/{asset_order_ordinal:06d}-"
        f"{source_asset_identity_sha256}.parsed.jsonl.zst"
    )


def _paths_for_asset(
    root: Path,
    *,
    context: ParsedAssetRecoveryContextV1,
    verified_asset: VerifiedLocalCacheAssetV3,
    asset_order_ordinal: int,
) -> tuple[str, Path, Path]:
    relative = _segment_relative_path(
        context_identity_sha256=context.identity_sha256,
        source_family=verified_asset.expected.source_family,
        asset_order_ordinal=asset_order_ordinal,
        source_asset_identity_sha256=verified_asset.expected.asset_identity_sha256,
    )
    segment_path = root.joinpath(*PurePosixPath(relative).parts)
    receipt_path = segment_path.with_name(segment_path.name + ".receipt.json")
    return relative, segment_path, receipt_path


def _receipt_artifact_bytes(receipt: ParsedAssetSegmentReceiptV1) -> bytes:
    return canonical_json_bytes(
        {
            "receipt": asdict(receipt),
            "receipt_sha256": receipt.receipt_sha256,
            "schema": PARSED_ASSET_RECEIPT_ARTIFACT_SCHEMA_V1,
        }
    ) + b"\n"


def _load_receipt(path: Path) -> tuple[ParsedAssetSegmentReceiptV1, int, str]:
    assert_no_symlink_ancestors(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ParsedAssetRecoveryError("parsed-asset receipt is unavailable") from error
    value = _load_canonical_json(raw, name="parsed-asset receipt artifact")
    _require_exact_keys(
        value,
        frozenset({"receipt", "receipt_sha256", "schema"}),
        "parsed-asset receipt artifact",
    )
    if value["schema"] != PARSED_ASSET_RECEIPT_ARTIFACT_SCHEMA_V1:
        raise ParsedAssetRecoveryError("parsed-asset receipt artifact schema is foreign")
    payload = value["receipt"]
    if not isinstance(payload, Mapping):
        raise ParsedAssetRecoveryError("parsed-asset receipt payload is not an object")
    receipt = ParsedAssetSegmentReceiptV1.from_mapping(payload)
    if value["receipt_sha256"] != receipt.receipt_sha256:
        raise ParsedAssetRecoveryError("parsed-asset receipt identity drifted")
    return receipt, len(raw), hashlib.sha256(raw).hexdigest()


def parsed_asset_composite_bridge_path_v1(root: Path) -> Path:
    if not isinstance(root, Path):
        raise TypeError("parsed-asset cache root must be pathlib.Path")
    return root / PARSED_ASSET_COMPOSITE_BRIDGE_FILENAME_V1


def load_parsed_asset_compatibility_policy_v1(
    path: Path,
) -> tuple[ParsedAssetCompatibilityPolicyV1, int, str]:
    """Load one canonical incident authority without weakening code identity."""

    if not isinstance(path, Path):
        raise TypeError("compatibility authority path must be pathlib.Path")
    assert_no_symlink_ancestors(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ParsedAssetRecoveryError(
            "parsed-asset compatibility authority is unavailable"
        ) from error
    value = _load_canonical_json(raw, name="parsed-asset compatibility authority")
    _require_exact_keys(
        value,
        frozenset({"policy", "policy_sha256", "schema"}),
        "parsed-asset compatibility authority",
    )
    if value["schema"] != PARSED_ASSET_COMPATIBILITY_POLICY_ARTIFACT_SCHEMA_V1:
        raise ParsedAssetRecoveryError(
            "parsed-asset compatibility authority schema is foreign"
        )
    payload = value["policy"]
    if not isinstance(payload, Mapping):
        raise ParsedAssetRecoveryError(
            "parsed-asset compatibility policy is not an object"
        )
    policy = ParsedAssetCompatibilityPolicyV1.from_mapping(payload)
    if value["policy_sha256"] != policy.identity_sha256:
        raise ParsedAssetRecoveryError(
            "parsed-asset compatibility policy identity drifted"
        )
    return policy, len(raw), hashlib.sha256(raw).hexdigest()


def validate_parsed_asset_composite_bridge_policy_v1(
    bridge: ParsedAssetCompositeBridgeV1,
    policy: ParsedAssetCompatibilityPolicyV1,
) -> None:
    if not isinstance(bridge, ParsedAssetCompositeBridgeV1) or not isinstance(
        policy, ParsedAssetCompatibilityPolicyV1
    ):
        raise TypeError("composite bridge policy validation requires typed values")
    if (
        bridge.compatibility_policy_sha256 != policy.identity_sha256
        or bridge.predecessor_asset_count
        != policy.expected_predecessor_asset_count
        or bridge.current_asset_count != policy.expected_current_asset_count
    ):
        raise ParsedAssetRecoveryError(
            "composite bridge counts or policy identity differ from authority"
        )
    for row in bridge.rows:
        permitted = policy.permits(
            row.source_family,
            row.parser_binding_sha256,
        )
        if row.resolution == READ_ONLY_PREDECESSOR_RESOLUTION_V1:
            if not permitted:
                raise ParsedAssetRecoveryError(
                    "composite bridge selected an unauthorized predecessor row"
                )
        elif (
            permitted
            or row.source_family not in policy.excluded_source_families
        ):
            raise ParsedAssetRecoveryError(
                "composite bridge selected current code outside an excluded family"
            )


def _composite_bridge_artifact_bytes(
    bridge: ParsedAssetCompositeBridgeV1,
) -> bytes:
    return canonical_json_bytes(
        {
            "bridge": asdict(bridge),
            "bridge_sha256": bridge.receipt_sha256,
            "schema": PARSED_ASSET_COMPOSITE_BRIDGE_ARTIFACT_SCHEMA_V1,
        }
    ) + b"\n"


def load_parsed_asset_composite_bridge_v1(
    root: Path,
) -> tuple[ParsedAssetCompositeBridgeV1, int, str]:
    path = parsed_asset_composite_bridge_path_v1(root)
    assert_no_symlink_ancestors(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ParsedAssetRecoveryError("parsed-asset composite bridge is unavailable") from error
    value = _load_canonical_json(raw, name="parsed-asset composite bridge artifact")
    _require_exact_keys(
        value,
        frozenset({"bridge", "bridge_sha256", "schema"}),
        "parsed-asset composite bridge artifact",
    )
    if value["schema"] != PARSED_ASSET_COMPOSITE_BRIDGE_ARTIFACT_SCHEMA_V1:
        raise ParsedAssetRecoveryError("parsed-asset composite bridge schema is foreign")
    payload = value["bridge"]
    if not isinstance(payload, Mapping):
        raise ParsedAssetRecoveryError("parsed-asset composite bridge payload is not an object")
    bridge = ParsedAssetCompositeBridgeV1.from_mapping(payload)
    if value["bridge_sha256"] != bridge.receipt_sha256:
        raise ParsedAssetRecoveryError("parsed-asset composite bridge identity drifted")
    return bridge, len(raw), hashlib.sha256(raw).hexdigest()


def publish_parsed_asset_composite_bridge_v1(
    root: Path,
    bridge: ParsedAssetCompositeBridgeV1,
) -> tuple[int, str]:
    """Publish once, or verify that an already-published bridge is identical."""

    if not isinstance(root, Path) or not isinstance(bridge, ParsedAssetCompositeBridgeV1):
        raise TypeError("parsed-asset composite bridge publication requires typed values")
    assert_no_symlink_ancestors(root)
    root_resolved = root.resolve(strict=True)
    if not root_resolved.is_dir():
        raise ParsedAssetRecoveryError("parsed-asset cache root is not a directory")
    path = parsed_asset_composite_bridge_path_v1(root_resolved)
    assert_no_symlink_ancestors(path)
    raw = _composite_bridge_artifact_bytes(bridge)
    expected_sha256 = hashlib.sha256(raw).hexdigest()
    if path.exists():
        loaded, physical_bytes, physical_sha256 = load_parsed_asset_composite_bridge_v1(
            root_resolved
        )
        if loaded != bridge or physical_bytes != len(raw) or physical_sha256 != expected_sha256:
            raise ParsedAssetRecoveryError(
                "refusing to replace a different parsed-asset composite bridge"
            )
        return physical_bytes, physical_sha256
    try:
        # The Pharma Drive capability probe rejects hard links (EPERM) and
        # renameat2(RENAME_NOREPLACE) (EINVAL).  ``xb`` is therefore the only
        # demonstrated atomic no-clobber primitive on the durable backend.
        # Its scope is one tiny bridge receipt: an interrupted short write can
        # strand a malformed final, which every reader rejects and which then
        # needs an explicit, audited manual recovery.  Parsed segment payloads
        # are never modified by this publication.
        with path.open("xb") as handle:
            written = handle.write(raw)
            if written != len(raw):
                raise ParsedAssetRecoveryError(
                    "parsed-asset composite bridge publication was short"
                )
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(root_resolved)
        loaded, physical_bytes, physical_sha256 = load_parsed_asset_composite_bridge_v1(
            root_resolved
        )
        if loaded != bridge or physical_bytes != len(raw) or physical_sha256 != expected_sha256:
            raise ParsedAssetRecoveryError("published parsed-asset composite bridge changed")
        return physical_bytes, physical_sha256
    except FileExistsError:
        loaded, physical_bytes, physical_sha256 = load_parsed_asset_composite_bridge_v1(
            root_resolved
        )
        if loaded != bridge or physical_bytes != len(raw) or physical_sha256 != expected_sha256:
            raise ParsedAssetRecoveryError(
                "refusing to replace a racing parsed-asset composite bridge"
            )
        return physical_bytes, physical_sha256


def parsed_asset_composite_bridge_row_v1(
    *,
    root: Path,
    receipt: ParsedAssetSegmentReceiptV1,
    resolution: str,
) -> ParsedAssetCompositeBridgeRowV1:
    if not isinstance(root, Path):
        raise TypeError("composite bridge row root must be pathlib.Path")
    if not isinstance(receipt, ParsedAssetSegmentReceiptV1):
        raise TypeError("composite bridge row requires a typed segment receipt")
    if resolution not in PARSED_ASSET_RESOLUTIONS_V1:
        raise ValueError("composite bridge resolution is unknown")
    root_resolved = assert_no_symlink_ancestors(root).resolve(strict=True)
    segment_path = root_resolved.joinpath(
        *PurePosixPath(receipt.segment_relative_path).parts
    )
    receipt_path = segment_path.with_name(segment_path.name + ".receipt.json")
    assert_no_symlink_ancestors(segment_path)
    try:
        segment_stat = segment_path.stat()
    except OSError as error:
        raise ParsedAssetRecoveryError(
            "composite bridge segment is unavailable during selection"
        ) from error
    if (
        not segment_path.is_file()
        or segment_path.is_symlink()
        or segment_stat.st_size != receipt.segment_physical_bytes
    ):
        raise ParsedAssetRecoveryError(
            "composite bridge segment type or size changed during selection"
        )
    loaded, receipt_physical_bytes, receipt_physical_sha256 = _load_receipt(
        receipt_path
    )
    if loaded != receipt:
        raise ParsedAssetRecoveryError(
            "composite bridge segment receipt changed during selection"
        )
    return ParsedAssetCompositeBridgeRowV1(
        source_family=receipt.source_family,
        asset_order_ordinal=receipt.asset_order_ordinal,
        source_asset_identity_sha256=receipt.source_asset_identity_sha256,
        source_asset_sha256=receipt.source_asset_sha256,
        parser_binding_sha256=receipt.parser_binding_sha256,
        first_event_ordinal=receipt.first_event_ordinal,
        next_event_ordinal=receipt.next_event_ordinal,
        resolution=resolution,
        selected_context_identity_sha256=receipt.context.identity_sha256,
        selected_code_identity_sha256=receipt.context.code_identity_sha256,
        segment_relative_path=receipt.segment_relative_path,
        segment_physical_bytes=receipt.segment_physical_bytes,
        segment_physical_sha256=receipt.segment_physical_sha256,
        segment_receipt_sha256=receipt.receipt_sha256,
        segment_receipt_physical_bytes=receipt_physical_bytes,
        segment_receipt_physical_sha256=receipt_physical_sha256,
    )


def select_parsed_asset_bridge_row_v1(
    bridge: ParsedAssetCompositeBridgeV1,
    *,
    source_family: str,
    asset_order_ordinal: int,
) -> ParsedAssetCompositeBridgeRowV1:
    if not isinstance(bridge, ParsedAssetCompositeBridgeV1):
        raise TypeError("parsed-asset bridge selection requires a typed bridge")
    matches = tuple(
        row
        for row in bridge.rows
        if row.source_family == source_family
        and row.asset_order_ordinal == asset_order_ordinal
    )
    if len(matches) != 1:
        raise ParsedAssetRecoveryError(
            "parsed-asset composite bridge does not select exactly one asset"
        )
    return matches[0]


def validate_parsed_asset_composite_bridge_segments_v1(
    bridge: ParsedAssetCompositeBridgeV1,
    *,
    current_root: Path,
    predecessor_root: Path | None,
) -> None:
    """Rehash every selected receipt and segment before a parent can mint."""

    if not isinstance(bridge, ParsedAssetCompositeBridgeV1):
        raise TypeError("parsed-asset bridge transport validation requires a bridge")
    if not isinstance(current_root, Path):
        raise TypeError("parsed-asset current root must be pathlib.Path")
    if (bridge.predecessor_context is None) != (predecessor_root is None):
        raise ParsedAssetRecoveryError(
            "parsed-asset bridge predecessor root assignment is incomplete"
        )
    roots = {
        CURRENT_CONTEXT_RESOLUTION_V1: assert_no_symlink_ancestors(
            current_root
        ).resolve(strict=True)
    }
    if predecessor_root is not None:
        if not isinstance(predecessor_root, Path):
            raise TypeError("parsed-asset predecessor root must be pathlib.Path")
        roots[READ_ONLY_PREDECESSOR_RESOLUTION_V1] = assert_no_symlink_ancestors(
            predecessor_root
        ).resolve(strict=True)
    for row in bridge.rows:
        root = roots.get(row.resolution)
        if root is None:
            raise ParsedAssetRecoveryError(
                "parsed-asset bridge row has no assigned transport root"
            )
        segment_path = root.joinpath(*PurePosixPath(row.segment_relative_path).parts)
        receipt_path = segment_path.with_name(segment_path.name + ".receipt.json")
        receipt, receipt_bytes, receipt_sha256 = _load_receipt(receipt_path)
        expected_context = (
            bridge.current_context
            if row.resolution == CURRENT_CONTEXT_RESOLUTION_V1
            else bridge.predecessor_context
        )
        if expected_context is None:
            raise ParsedAssetRecoveryError(
                "parsed-asset predecessor context disappeared"
            )
        if (
            receipt.context != expected_context
            or receipt.source_family != row.source_family
            or receipt.asset_order_ordinal != row.asset_order_ordinal
            or receipt.source_asset_identity_sha256
            != row.source_asset_identity_sha256
            or receipt.source_asset_sha256 != row.source_asset_sha256
            or receipt.parser_binding_sha256 != row.parser_binding_sha256
            or receipt.first_event_ordinal != row.first_event_ordinal
            or receipt.next_event_ordinal != row.next_event_ordinal
            or receipt.segment_relative_path != row.segment_relative_path
            or receipt.segment_physical_bytes != row.segment_physical_bytes
            or receipt.segment_physical_sha256 != row.segment_physical_sha256
            or receipt.receipt_sha256 != row.segment_receipt_sha256
            or receipt_bytes != row.segment_receipt_physical_bytes
            or receipt_sha256 != row.segment_receipt_physical_sha256
        ):
            raise ParsedAssetRecoveryError(
                "parsed-asset bridge receipt transport differs from its row"
            )
        segment_bytes, segment_sha256 = _hash_path(segment_path)
        if (
            segment_bytes != row.segment_physical_bytes
            or segment_sha256 != row.segment_physical_sha256
        ):
            raise ParsedAssetRecoveryError(
                "parsed-asset bridge segment transport differs from its row"
            )


def _validate_expected_receipt(
    receipt: ParsedAssetSegmentReceiptV1,
    *,
    context: ParsedAssetRecoveryContextV1,
    verified_asset: VerifiedLocalCacheAssetV3,
    parser_binding: SourceParserBindingV3,
    asset_order_ordinal: int,
    expected_first_event_ordinal: int,
    expected_relative_path: str,
) -> None:
    expected = (
        context,
        verified_asset.expected.source_family,
        asset_order_ordinal,
        verified_asset.expected.asset_identity_sha256,
        verified_asset.observed_sha256,
        parser_binding.binding_sha256,
        expected_first_event_ordinal,
        expected_relative_path,
    )
    observed = (
        receipt.context,
        receipt.source_family,
        receipt.asset_order_ordinal,
        receipt.source_asset_identity_sha256,
        receipt.source_asset_sha256,
        receipt.parser_binding_sha256,
        receipt.first_event_ordinal,
        receipt.segment_relative_path,
    )
    if observed != expected:
        raise ParsedAssetRecoveryError("parsed-asset receipt does not match caller identity")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ParsedAssetRecoveryError(f"{name} is not an object")
    return value


def parsed_asset_runtime_identity_v1(
    environment_payload: Mapping[str, object],
) -> str:
    """Bind parser-relevant runtime bytes without binding a Colab host image.

    The full production environment receipt remains exact and host-bound.  A
    recovery segment, however, must survive a legitimate replacement VM when
    the hash-pinned interpreter, installed files, linkage bytes, versions, and
    determinism settings are unchanged.  Absolute installation/linkage paths
    and the host kernel release are therefore intentionally excluded.
    """

    if not isinstance(environment_payload, Mapping):
        raise TypeError("parsed-asset runtime payload must be an object")
    required = frozenset(
        {
            "byteorder",
            "cache_tag",
            "dependency_lock_sha256",
            "distributions",
            "environment",
            "filesystem_encoding",
            "implementation",
            "installed_distribution_inventory",
            "locale",
            "machine",
            "maxunicode",
            "platform_system",
            "preferred_encoding",
            "python_executable_sha256",
            "runtime_linkage",
            "runtime_versions",
        }
    )
    expected_top_level = required | {"platform_release"}
    if set(environment_payload) != expected_top_level:
        raise ParsedAssetRecoveryError(
            "parsed-asset runtime payload fields require an explicit projection"
        )
    inventory = _mapping(
        environment_payload["installed_distribution_inventory"],
        "installed distribution inventory",
    )
    _require_exact_keys(
        inventory,
        frozenset(
            {
                "bootstrap_distributions",
                "distributions",
                "files",
                "installation_prefix",
                "inventory_identity_sha256",
                "schema",
                "site_roots",
            }
        ),
        "installed distribution inventory",
    )
    linkage = _mapping(environment_payload["runtime_linkage"], "runtime linkage")
    _require_exact_keys(
        linkage,
        frozenset(
            {
                "executable",
                "libpython_library",
                "linkage_identity_sha256",
                "schema",
                "sqlite_extension",
                "sqlite_library",
            }
        ),
        "runtime linkage",
    )
    linkage_projection: dict[str, object] = {"schema": linkage["schema"]}
    for name in (
        "executable",
        "libpython_library",
        "sqlite_extension",
        "sqlite_library",
    ):
        row = _mapping(linkage[name], f"runtime linkage {name}")
        _require_exact_keys(
            row,
            frozenset({"bytes", "path", "sha256"}),
            f"runtime linkage {name}",
        )
        linkage_projection[name] = {
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
    projection = {
        key: environment_payload[key]
        for key in sorted(required - {"installed_distribution_inventory", "runtime_linkage"})
    }
    projection["installed_distribution_inventory"] = {
        key: inventory[key]
        for key in (
            "bootstrap_distributions",
            "distributions",
            "files",
            "schema",
            "site_roots",
        )
    }
    projection["runtime_linkage"] = linkage_projection
    return canonical_sha256(
        {
            "payload": projection,
            "schema": PARSED_ASSET_RUNTIME_IDENTITY_SCHEMA_V1,
        }
    )


def _recovered_event_from_row(
    row: Mapping[str, object],
    *,
    receipt: ParsedAssetSegmentReceiptV1,
    local_index: int,
    expected_asset: SourceCacheAssetV3,
) -> RecoveredSourceParseEventV1:
    _require_exact_keys(row, _SPOOL_ROW_KEYS, "parsed-asset event")
    expected_event_ordinal = receipt.first_event_ordinal + local_index
    expected_scalars = {
        "asset_order_ordinal": receipt.asset_order_ordinal,
        "event_ordinal": expected_event_ordinal,
        "parser_binding_sha256": receipt.parser_binding_sha256,
        "source_asset_identity_sha256": receipt.source_asset_identity_sha256,
        "source_asset_sha256": receipt.source_asset_sha256,
        "source_family": receipt.source_family,
        "source_record_ordinal": local_index,
    }
    if any(row[key] != value for key, value in expected_scalars.items()):
        raise ParsedAssetRecoveryError("parsed-asset event identity or order drifted")
    disposition = row["disposition"]
    if disposition not in PARSE_DISPOSITIONS:
        raise ParsedAssetRecoveryError("parsed-asset event disposition is unknown")

    observation: SourceRecordObservationV3 | None = None
    if row["observation"] is not None:
        observed = _mapping(row["observation"], "parsed-asset observation")
        _require_exact_keys(observed, _OBSERVATION_KEYS, "parsed-asset observation")
        try:
            observation = SourceRecordObservationV3(**dict(observed))  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ParsedAssetRecoveryError("parsed-asset observation is invalid") from error
        if (
            observation.source_cache_asset_identity_sha256
            != receipt.source_asset_identity_sha256
            or observation.source_asset_sha256 != receipt.source_asset_sha256
        ):
            raise ParsedAssetRecoveryError("parsed-asset observation identity drifted")

    record: ParsedSourceRecordV3 | None = None
    reason = row["reason"]
    if disposition == RETAIN:
        if reason is not None:
            raise ParsedAssetRecoveryError("retained parsed-asset event carries a reason")
        retained = _mapping(row["retained"], "parsed-asset retained record")
        _require_exact_keys(retained, _RETAINED_KEYS, "parsed-asset retained record")
        if retained["parser_binding_sha256"] != receipt.parser_binding_sha256:
            raise ParsedAssetRecoveryError("retained parser binding drifted")
        raw_value = _mapping(retained["raw_document"], "parsed-asset raw document")
        _require_exact_keys(raw_value, _RAW_DOCUMENT_KEYS, "parsed-asset raw document")
        if not isinstance(raw_value["text"], str):
            raise ParsedAssetRecoveryError("retained parsed-asset text is not UTF-8 text")
        text_bytes = raw_value["text"].encode("utf-8", errors="strict")
        if (
            retained["text_utf8_bytes"] != len(text_bytes)
            or retained["text_utf8_sha256"] != hashlib.sha256(text_bytes).hexdigest()
        ):
            raise ParsedAssetRecoveryError("retained parsed-asset text identity drifted")

        canonical_value = _mapping(
            retained["canonical_record"], "parsed-asset canonical record"
        )
        canonical_keys = frozenset(canonical_value)
        if canonical_keys not in {
            _CANONICAL_RECORD_REQUIRED_KEYS,
            _CANONICAL_RECORD_REQUIRED_KEYS | _CANONICAL_RECORD_OPTIONAL_KEYS,
        }:
            raise ParsedAssetRecoveryError("parsed-asset canonical-record fields are not exact")
        asset_value = _mapping(canonical_value["asset"], "parsed-asset source asset")
        try:
            source_asset = type(expected_asset).from_mapping(asset_value)
        except (TypeError, ValueError) as error:
            raise ParsedAssetRecoveryError("parsed-asset source asset is invalid") from error
        if source_asset != expected_asset:
            raise ParsedAssetRecoveryError("parsed-asset retained source asset drifted")
        try:
            canonical_record = CanonicalSourceRecordV3(
                asset=source_asset,
                source_record_ordinal=canonical_value["source_record_ordinal"],  # type: ignore[arg-type]
                retained_byte_count=canonical_value["retained_byte_count"],  # type: ignore[arg-type]
                native_record_id=canonical_value["native_record_id"],  # type: ignore[arg-type]
                int_score=canonical_value["int_score"],  # type: ignore[arg-type]
                native_record_namespace=canonical_value.get("native_record_namespace"),  # type: ignore[arg-type]
            )
            raw_document = RawDocumentV3(**dict(raw_value))  # type: ignore[arg-type]
            record = ParsedSourceRecordV3(
                canonical_record=canonical_record,
                raw_document=raw_document,
                parser_binding_sha256=receipt.parser_binding_sha256,
            )
        except (TypeError, ValueError) as error:
            raise ParsedAssetRecoveryError("parsed-asset retained record is invalid") from error
        if (
            canonical_record.source_record_ordinal != local_index
            or canonical_record.retained_byte_count != len(text_bytes)
        ):
            raise ParsedAssetRecoveryError("retained parsed-asset ordinals or bytes drifted")
    else:
        if row["retained"] is not None or not isinstance(reason, str) or not reason:
            raise ParsedAssetRecoveryError("dropped parsed-asset event is malformed")

    try:
        event = SourceParseEventV3(
            source_family=receipt.source_family,
            source_record_ordinal=local_index,
            disposition=disposition,  # type: ignore[arg-type]
            record=record,
            reason=reason,  # type: ignore[arg-type]
            observation=observation,
        )
    except (TypeError, ValueError) as error:
        raise ParsedAssetRecoveryError("parsed-asset event is invalid") from error
    if row["event_sha256"] != event.event_sha256:
        raise ParsedAssetRecoveryError("parsed-asset event SHA-256 drifted")
    return RecoveredSourceParseEventV1(
        asset_order_ordinal=receipt.asset_order_ordinal,
        event_ordinal=expected_event_ordinal,
        source_asset_identity_sha256=receipt.source_asset_identity_sha256,
        source_asset_sha256=receipt.source_asset_sha256,
        parser_binding_sha256=receipt.parser_binding_sha256,
        event=event,
    )


def _validate_segment_to_snapshot(
    segment_path: Path,
    receipt: ParsedAssetSegmentReceiptV1,
    *,
    expected_asset: SourceCacheAssetV3,
    snapshot: object,
) -> None:
    assert_no_symlink_ancestors(segment_path)
    logical_digest = hashlib.sha256()
    logical_bytes = 0
    retained_count = 0
    observation_count = 0
    dispositions: Counter[str] = Counter()
    row_count = 0
    try:
        with segment_path.open("rb") as compressed:
            physical_bytes, physical_sha256 = _hash_open_file(compressed)
            if (
                physical_bytes != receipt.segment_physical_bytes
                or physical_sha256 != receipt.segment_physical_sha256
            ):
                raise ParsedAssetRecoveryError("parsed-asset physical bytes drifted")
            with zstandard.ZstdDecompressor().stream_reader(
                compressed, read_across_frames=True, closefd=False
            ) as reader:
                buffered = io.BufferedReader(reader, buffer_size=8 * 1024 * 1024)
                for raw_line in buffered:
                    row = _load_canonical_json(raw_line, name="parsed-asset event row")
                    recovered = _recovered_event_from_row(
                        row,
                        receipt=receipt,
                        local_index=row_count,
                        expected_asset=expected_asset,
                    )
                    snapshot.write(raw_line)  # type: ignore[attr-defined]
                    logical_digest.update(raw_line)
                    logical_bytes += len(raw_line)
                    row_count += 1
                    dispositions[recovered.event.disposition] += 1
                    if recovered.event.disposition == RETAIN:
                        retained_count += 1
                    if recovered.event.observation is not None:
                        observation_count += 1
    except (OSError, zstandard.ZstdError) as error:
        raise ParsedAssetRecoveryError("parsed-asset segment cannot be read") from error
    if (
        row_count != receipt.event_count
        or logical_bytes != receipt.logical_jsonl_bytes
        or logical_digest.hexdigest() != receipt.logical_jsonl_sha256
        or retained_count != receipt.retained_record_count
        or observation_count != receipt.observation_count
        or tuple(
            (name, dispositions[name])
            for name in PARSE_DISPOSITIONS
            if name != RETAIN
        )
        != receipt.drop_counts
    ):
        raise ParsedAssetRecoveryError("parsed-asset logical receipt counts drifted")
    snapshot.flush()  # type: ignore[attr-defined]
    snapshot.seek(0)  # type: ignore[attr-defined]


def write_parsed_asset_segment_v1(
    root: Path,
    *,
    context: ParsedAssetRecoveryContextV1,
    verified_asset: VerifiedLocalCacheAssetV3,
    parser_binding: SourceParserBindingV3,
    asset_order_ordinal: int,
    first_event_ordinal: int,
    events: Iterable[SourceParseEventV3],
) -> PublishedParsedAssetSegmentV1:
    """Publish and re-open one immutable, complete source-asset segment."""

    if not isinstance(root, Path):
        raise TypeError("parsed-asset cache root must be pathlib.Path")
    if not isinstance(context, ParsedAssetRecoveryContextV1):
        raise TypeError("parsed-asset writer requires a typed context")
    if not isinstance(verified_asset, VerifiedLocalCacheAssetV3):
        raise TypeError("parsed-asset writer requires a verified source asset")
    if not isinstance(parser_binding, SourceParserBindingV3):
        raise TypeError("parsed-asset writer requires a typed parser binding")
    _require_nonnegative_int(asset_order_ordinal, "asset order ordinal")
    _require_nonnegative_int(first_event_ordinal, "first event ordinal")
    source_family = verified_asset.expected.source_family
    if parser_binding.source_family != source_family:
        raise ParsedAssetRecoveryError("parser binding and source asset disagree")
    relative, segment_path, receipt_path = _paths_for_asset(
        root,
        context=context,
        verified_asset=verified_asset,
        asset_order_ordinal=asset_order_ordinal,
    )
    assert_no_symlink_ancestors(root)
    if receipt_path.exists():
        raise ParsedAssetRecoveryError("refusing to overwrite parsed-asset cache state")
    assert_no_symlink_ancestors(segment_path)
    orphan_segment_exists = segment_path.exists()
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    partial_nonce = uuid.uuid4().hex
    segment_partial = segment_path.with_name(f".data-{partial_nonce}.partial")
    receipt_partial = receipt_path.with_name(f".receipt-{partial_nonce}.partial")
    assert_no_symlink_ancestors(segment_partial)
    assert_no_symlink_ancestors(receipt_partial)

    logical_digest = hashlib.sha256()
    logical_bytes = 0
    event_count = 0
    retained_count = 0
    observation_count = 0
    dispositions: Counter[str] = Counter()
    segment_published = False
    try:
        with segment_partial.open("xb") as raw_handle:
            compressor = zstandard.ZstdCompressor(
                level=3,
                write_checksum=True,
            )
            with compressor.stream_writer(raw_handle, closefd=False) as writer:
                for event in events:
                    if not isinstance(event, SourceParseEventV3):
                        raise TypeError("parsed-asset writer received an untyped event")
                    if (
                        event.source_family != source_family
                        or event.source_record_ordinal != event_count
                    ):
                        raise ParsedAssetRecoveryError(
                            "parsed-asset events are foreign or non-contiguous"
                        )
                    if event.record is not None:
                        if (
                            event.record.parser_binding_sha256
                            != parser_binding.binding_sha256
                            or event.record.canonical_record.asset
                            != verified_asset.expected
                        ):
                            raise ParsedAssetRecoveryError(
                                "parsed-asset retained record identity drifted"
                            )
                    row = _canonical_spool_event(
                        asset_order_ordinal=asset_order_ordinal,
                        verified_asset=verified_asset,
                        event_ordinal=first_event_ordinal + event_count,
                        event=event,
                        binding=parser_binding,
                    )
                    logical = canonical_json_bytes(row) + b"\n"
                    writer.write(logical)
                    logical_digest.update(logical)
                    logical_bytes += len(logical)
                    event_count += 1
                    dispositions[event.disposition] += 1
                    if event.disposition == RETAIN:
                        retained_count += 1
                    if event.observation is not None:
                        observation_count += 1
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        segment_physical_bytes, segment_physical_sha256 = _hash_path(segment_partial)
        if orphan_segment_exists:
            orphan_bytes, orphan_sha256 = _hash_path(segment_path)
            if (
                orphan_bytes != segment_physical_bytes
                or orphan_sha256 != segment_physical_sha256
            ):
                raise ParsedAssetRecoveryError(
                    "orphan parsed-asset segment differs from exact retry"
                )
            segment_partial.unlink()
        else:
            if segment_path.exists():
                raise ParsedAssetRecoveryError("parsed-asset segment appeared before publish")
            os.replace(segment_partial, segment_path)
        directory_fsync = _fsync_directory(segment_path.parent)
        segment_published = True

        receipt = ParsedAssetSegmentReceiptV1(
            schema=PARSED_ASSET_RECEIPT_SCHEMA_V1,
            recovery_domain=PARSED_ASSET_RECOVERY_DOMAIN_V1,
            segment_schema=PARSED_ASSET_SEGMENT_SCHEMA_V1,
            context=context,
            source_family=source_family,
            asset_order_ordinal=asset_order_ordinal,
            source_asset_identity_sha256=(
                verified_asset.expected.asset_identity_sha256
            ),
            source_asset_sha256=verified_asset.observed_sha256,
            parser_binding_sha256=parser_binding.binding_sha256,
            segment_relative_path=relative,
            segment_physical_bytes=segment_physical_bytes,
            segment_physical_sha256=segment_physical_sha256,
            logical_jsonl_bytes=logical_bytes,
            logical_jsonl_sha256=logical_digest.hexdigest(),
            event_count=event_count,
            retained_record_count=retained_count,
            drop_counts=tuple(
                (name, dispositions[name])
                for name in PARSE_DISPOSITIONS
                if name != RETAIN
            ),
            observation_count=observation_count,
            first_event_ordinal=first_event_ordinal,
            next_event_ordinal=first_event_ordinal + event_count,
            first_source_record_ordinal=0,
            next_source_record_ordinal=event_count,
            directory_fsync=directory_fsync,
        )
        receipt_raw = _receipt_artifact_bytes(receipt)
        with receipt_partial.open("xb") as handle:
            handle.write(receipt_raw)
            handle.flush()
            os.fsync(handle.fileno())
        if receipt_path.exists():
            raise ParsedAssetRecoveryError("parsed-asset receipt appeared before publish")
        os.replace(receipt_partial, receipt_path)
        receipt_directory_fsync = _fsync_directory(receipt_path.parent)
        if receipt_directory_fsync != directory_fsync:
            raise ParsedAssetRecoveryError("directory-fsync capability changed mid-publication")

        loaded, receipt_bytes, receipt_sha256 = _load_receipt(receipt_path)
        if loaded != receipt:
            raise ParsedAssetRecoveryError("published parsed-asset receipt changed")
        recovered_count = sum(
            1
            for unused in iter_parsed_asset_segment_v1(
                root,
                context=context,
                verified_asset=verified_asset,
                parser_binding=parser_binding,
                asset_order_ordinal=asset_order_ordinal,
                expected_first_event_ordinal=first_event_ordinal,
            )
        )
        if recovered_count != event_count:
            raise ParsedAssetRecoveryError("published parsed-asset segment changed")
        return PublishedParsedAssetSegmentV1(
            receipt=receipt,
            segment_path=segment_path,
            receipt_path=receipt_path,
            receipt_physical_bytes=receipt_bytes,
            receipt_physical_sha256=receipt_sha256,
        )
    except BaseException:
        segment_partial.unlink(missing_ok=True)
        receipt_partial.unlink(missing_ok=True)
        # Once a final object exists it is immutable forensic evidence.  Do not
        # silently remove or overwrite it after a later publication failure.
        if segment_published:
            _fsync_directory(segment_path.parent)
        raise


def iter_parsed_asset_segment_v1(
    root: Path,
    *,
    context: ParsedAssetRecoveryContextV1,
    verified_asset: VerifiedLocalCacheAssetV3,
    parser_binding: SourceParserBindingV3,
    asset_order_ordinal: int,
    expected_first_event_ordinal: int,
) -> Iterator[RecoveredSourceParseEventV1]:
    """Validate a whole segment before yielding any recovery event.

    Validation copies canonical logical rows to an anonymous temporary snapshot.
    No row is yielded until physical identity, logical identity, every event,
    and all receipt counts have passed.  Yielding therefore cannot expose a
    prefix of a corrupt durable segment.
    """

    if not isinstance(root, Path):
        raise TypeError("parsed-asset cache root must be pathlib.Path")
    if not isinstance(context, ParsedAssetRecoveryContextV1):
        raise TypeError("parsed-asset loader requires a typed context")
    if not isinstance(verified_asset, VerifiedLocalCacheAssetV3):
        raise TypeError("parsed-asset loader requires a verified source asset")
    if not isinstance(parser_binding, SourceParserBindingV3):
        raise TypeError("parsed-asset loader requires a typed parser binding")
    _require_nonnegative_int(asset_order_ordinal, "asset order ordinal")
    _require_nonnegative_int(expected_first_event_ordinal, "first event ordinal")
    relative, segment_path, receipt_path = _paths_for_asset(
        root,
        context=context,
        verified_asset=verified_asset,
        asset_order_ordinal=asset_order_ordinal,
    )
    receipt, _unused_bytes, _unused_sha256 = _load_receipt(receipt_path)
    _validate_expected_receipt(
        receipt,
        context=context,
        verified_asset=verified_asset,
        parser_binding=parser_binding,
        asset_order_ordinal=asset_order_ordinal,
        expected_first_event_ordinal=expected_first_event_ordinal,
        expected_relative_path=relative,
    )
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b") as snapshot:
        _validate_segment_to_snapshot(
            segment_path,
            receipt,
            expected_asset=verified_asset.expected,
            snapshot=snapshot,
        )
        for local_index, raw_line in enumerate(snapshot):
            row = _load_canonical_json(raw_line, name="validated parsed-asset event")
            yield _recovered_event_from_row(
                row,
                receipt=receipt,
                local_index=local_index,
                expected_asset=verified_asset.expected,
            )


def inspect_parsed_asset_segment_receipt_v1(
    root: Path,
    *,
    context: ParsedAssetRecoveryContextV1,
    verified_asset: VerifiedLocalCacheAssetV3,
    parser_binding: SourceParserBindingV3,
    asset_order_ordinal: int,
    expected_first_event_ordinal: int,
    allow_receiptless_orphan: bool = False,
) -> ParsedAssetSegmentReceiptV1 | None:
    """Return a validated commit receipt without reading the segment payload.

    The receipt is the publication commit marker.  This inspection is intended
    for a resumable cache-fill pass that must skip completed assets in O(asset
    count), not O(cached bytes).  Any consumer of event rows must still call
    :func:`iter_parsed_asset_segment_v1`, which validates the complete physical
    and logical segment before yielding its first row.
    """

    if type(allow_receiptless_orphan) is not bool:
        raise TypeError("allow_receiptless_orphan must be an exact boolean")
    relative, segment_path, receipt_path = _paths_for_asset(
        root,
        context=context,
        verified_asset=verified_asset,
        asset_order_ordinal=asset_order_ordinal,
    )
    segment_exists = segment_path.exists()
    receipt_exists = receipt_path.exists()
    if not segment_exists and not receipt_exists:
        return None
    if segment_exists and not receipt_exists:
        if allow_receiptless_orphan:
            return None
        raise ParsedAssetRecoveryError(
            "receiptless parsed-asset orphan requires opt-in"
        )
    if receipt_exists and not segment_exists:
        raise ParsedAssetRecoveryError(
            "parsed-asset receipt exists without its segment"
        )
    assert_no_symlink_ancestors(segment_path)
    try:
        segment_stat = segment_path.stat()
    except OSError as error:
        raise ParsedAssetRecoveryError(
            "parsed-asset segment metadata is unavailable"
        ) from error
    receipt, _receipt_bytes, _receipt_sha256 = _load_receipt(receipt_path)
    _validate_expected_receipt(
        receipt,
        context=context,
        verified_asset=verified_asset,
        parser_binding=parser_binding,
        asset_order_ordinal=asset_order_ordinal,
        expected_first_event_ordinal=expected_first_event_ordinal,
        expected_relative_path=relative,
    )
    if (
        not segment_path.is_file()
        or segment_path.is_symlink()
        or segment_stat.st_size != receipt.segment_physical_bytes
    ):
        raise ParsedAssetRecoveryError(
            "parsed-asset segment type or size differs from its receipt"
        )
    return receipt


def probe_parsed_asset_segment_v1(
    root: Path,
    *,
    context: ParsedAssetRecoveryContextV1,
    verified_asset: VerifiedLocalCacheAssetV3,
    parser_binding: SourceParserBindingV3,
    asset_order_ordinal: int,
    expected_first_event_ordinal: int,
    allow_receiptless_orphan: bool = False,
) -> str:
    """Return ``MISS`` or ``HIT`` without treating malformed state as absence.

    A receiptless final segment is non-authoritative and may be reported as a
    miss only when the caller explicitly opts into exact orphan adoption.
    Receipt-without-segment, foreign receipts, and physical drift always raise.
    Uncommitted uniquely named ``.partial`` files are intentionally ignored.
    """

    if type(allow_receiptless_orphan) is not bool:
        raise TypeError("allow_receiptless_orphan must be an exact boolean")
    receipt = inspect_parsed_asset_segment_receipt_v1(
        root,
        context=context,
        verified_asset=verified_asset,
        parser_binding=parser_binding,
        asset_order_ordinal=asset_order_ordinal,
        expected_first_event_ordinal=expected_first_event_ordinal,
        allow_receiptless_orphan=allow_receiptless_orphan,
    )
    if receipt is None:
        return "MISS"
    segment_path = root.joinpath(*PurePosixPath(receipt.segment_relative_path).parts)
    physical_bytes, physical_sha256 = _hash_path(segment_path)
    if (
        physical_bytes != receipt.segment_physical_bytes
        or physical_sha256 != receipt.segment_physical_sha256
    ):
        raise ParsedAssetRecoveryError("parsed-asset physical bytes drifted")
    return "HIT"


__all__ = [
    "CURRENT_CONTEXT_RESOLUTION_V1",
    "PARSED_ASSET_COMPATIBILITY_POLICY_ARTIFACT_SCHEMA_V1",
    "PARSED_ASSET_COMPATIBILITY_POLICY_SCHEMA_V1",
    "PARSED_ASSET_COMPOSITE_BRIDGE_ARTIFACT_SCHEMA_V1",
    "PARSED_ASSET_COMPOSITE_BRIDGE_FILENAME_V1",
    "PARSED_ASSET_COMPOSITE_BRIDGE_SCHEMA_V1",
    "PARSED_ASSET_RECEIPT_ARTIFACT_SCHEMA_V1",
    "PARSED_ASSET_RECEIPT_SCHEMA_V1",
    "PARSED_ASSET_RECOVERY_DOMAIN_V1",
    "PARSED_ASSET_RUNTIME_IDENTITY_SCHEMA_V1",
    "PARSED_ASSET_SEGMENT_SCHEMA_V1",
    "READ_ONLY_PREDECESSOR_RESOLUTION_V1",
    "ParsedAssetCompatibilityPolicyV1",
    "ParsedAssetCompositeBridgeRowV1",
    "ParsedAssetCompositeBridgeV1",
    "ParsedAssetRecoveryContextV1",
    "ParsedAssetRecoveryError",
    "ParsedAssetSegmentReceiptV1",
    "PublishedParsedAssetSegmentV1",
    "RecoveredSourceParseEventV1",
    "inspect_parsed_asset_segment_receipt_v1",
    "iter_parsed_asset_segment_v1",
    "load_parsed_asset_compatibility_policy_v1",
    "load_parsed_asset_composite_bridge_v1",
    "parsed_asset_composite_bridge_path_v1",
    "parsed_asset_composite_bridge_row_v1",
    "parsed_asset_runtime_identity_v1",
    "probe_parsed_asset_segment_v1",
    "publish_parsed_asset_composite_bridge_v1",
    "select_parsed_asset_bridge_row_v1",
    "validate_compatible_recovery_contexts_v1",
    "validate_parsed_asset_composite_bridge_policy_v1",
    "validate_parsed_asset_composite_bridge_segments_v1",
    "write_parsed_asset_segment_v1",
]
