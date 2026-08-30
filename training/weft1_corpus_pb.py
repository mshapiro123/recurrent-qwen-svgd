"""Forward-only P-B freeze/minter scaffold for WEFT-1.

The scaffold consumes a completed V4 P-A tree plus independently minted D1-D6
evidence.  Its production mint owns the parent launch of the network-isolated
DECON child and can only mint from that fresh aggregate receipt.  The parent
never parses sealed rows; an external receipt may be checked read-only but can
never cross the mint boundary.

No receipt is written until every P-A file has been re-read, C1-C3 have been
recomputed, D1-D6 are authoritative, and DECON is clean.  A DECON hit raises a
dedicated hard-stop exception before the output path is touched.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, fields
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tempfile
from typing import Any, Mapping, Sequence

from training.weft1_corpus_a2 import (
    A2_LANGUAGE_ID_BINDING,
    A2_ZSTD_CODEC_BINDING,
    FIRST_FIT_TOLERANCE,
    JsonlZstdShardIdentityV3,
    LanguageIdDecisionV3,
    StableDocumentV3,
    execution_authority_v3_bound_sha256,
)
from training.weft1_corpus_a3 import (
    GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
    execution_authority_v4_bound_sha256,
)
from training.weft1_corpus_contract import CORPUS_STRATUM_TARGETS
from training.weft1_corpus_materialize_a3 import (
    CONSUMER_ORDER_SCHEMA_V4,
    CorpusMaterializationV4Error,
    D1_READY_IDENTITY_SCHEMA_V4,
    D1_READY_SCHEMA_V4,
    D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4,
    D6_PHYSICAL_EVIDENCE_SCHEMA_V4,
    FULL_SHARD_MANIFEST_RELATIVE_PATH_V4,
    FULL_SHARD_MANIFEST_SCHEMA_V4,
    MATERIALIZED_CONTENT_SCHEMA_V4,
    MATERIALIZER_SCHEMA_V4,
    RELEASE_MANIFEST_SECTION_SCHEMA_V4,
    SCREEN_SUBMANIFEST_RELATIVE_PATH_V4,
    SCREEN_SUBMANIFEST_SCHEMA_V4,
    TOKENIZER_FIT_INPUT_SCHEMA_V4,
    V4_READINESS,
    recompute_physical_d6_evidence_v4,
)
from training.weft1_gtok_contract import (
    GTOK_ROUND_TRIP_CATEGORIES,
    GTOK_SCREEN_HELDOUT_STRATUM_TARGETS,
    GTOK_SCREEN_TRAIN_STRATUM_TARGETS,
    GTOK_SEED_COUNT,
    GTOK_STRATA,
    GTOK_VOCABULARY_ARMS,
    canonical_json_bytes,
)
from training.weft1_corpus_materialize_a2 import (
    SOURCE_TO_STRATUM,
)
from training.weft1_gtok_a1_contract import SOURCE_FAMILIES
from training.weft1_release import (
    RELEASE_AUTHORITY_SHA256,
    release_manifest_section,
    verify_release_authority_artifact,
)
from training.weft1_strict_io import (
    StrictJsonError,
    assert_no_symlink_ancestors,
    load_canonical_json_snapshot,
)
from training.weft1_corpus_replay_a3 import (
    PARENT_EVIDENCE_SCHEMA_V4,
    PARENT_REPLAY_SCHEMA_V4,
    ParentReplayVerificationV4,
)
from training import weft1_corpus_replay_a2 as replay_v3
from training.weft1_corpus_decon_contract import (
    DECON_BATTERIES,
    GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256,
    GOVERNED_CONFIRM_SEAL_SET_SHA256,
    GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256,
    GOVERNED_CONFIRM_SOURCE_ROWS_SHA256,
    GOVERNED_EVAL_E_ANONYMOUS_INDEX_SHA256,
    GOVERNED_EVAL_E_LOCK_SHA256,
    algorithm_profiles as decon_algorithm_profiles,
)
from training.weft1_corpus_decon import (
    DECON_CODE_RELATIVE_PATHS as _DECON_CODE_RELATIVE_PATHS,
    DECON_PARENT_WATCHDOG_SECONDS,
    DECON_RECEIPT_FILENAME,
    DeconError,
    launch_hermetic_decon,
)
from training import weft1_corpus_pa as corpus_pa
from training.weft1_gtok_tokenizer_a2 import iter_a2_shard_texts


PB_AUTHORITY_CHAIN_V5 = (
    *GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
    RELEASE_AUTHORITY_SHA256,
)
PB_GATE_BUNDLE_SCHEMA_V4 = "weft1_corpus_d1_d6_gate_bundle_v4"
PB_C2_EVIDENCE_SCHEMA_V5 = "weft1_corpus_c2_fixture_evidence_v5"
PB_DECON_SCHEMA_V5 = "weft1_corpus_hermetic_decon_receipt_v5"
PB_FREEZE_SCHEMA_V5 = "weft1_corpus_freeze_receipt_v5"

FULL_STRATUM_TARGETS_V4 = dict(CORPUS_STRATUM_TARGETS)
DECON_REQUIRED_BATTERIES = DECON_BATTERIES
_GATES = ("D1", "D2", "D3", "D4", "D5", "D6")
_GATE_VERIFIERS = {
    "D1": "independent_full_replay",
    "D2": "independent_dedup_replay",
    "D3": "independent_manifest_composition_reread",
    "D4": "independent_language_ledger_reread",
    "D5": "independent_fixture_and_shard_roundtrip",
    "D6": "independent_stream_reread",
}
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARD_KEYS = {
    "codec_binding_sha256",
    "content_identity_sha256",
    "identity_relative_path",
    "first_full_ordinal",
    "last_full_ordinal",
    "logical_jsonl_bytes",
    "logical_jsonl_sha256",
    "record_count",
    "relative_path",
    "retained_text_bytes",
    "source",
    "stream",
    "stratum",
    "zstd_bytes",
    "zstd_sha256",
}
_SCREEN_SHARD_KEYS = _SHARD_KEYS - {
    "first_full_ordinal",
    "last_full_ordinal",
    "source",
}
_C2_PAYLOADS = {
    "accented_latin": "café e\u0301".encode("utf-8"),
    "cjk": "漢字かなカナ".encode("utf-8"),
    "greek": "αλφάβητο".encode("utf-8"),
    "mixed_indentation": b"  alpha\n\tbeta\n    gamma",
    "right_to_left": "עברית العربية".encode("utf-8"),
    "tabs": b"\talpha\tbeta\t",
    "typographic_punctuation": "“quote”—‘dash’…".encode("utf-8"),
}


class PBFreezeError(RuntimeError):
    """A P-B prerequisite or independent re-read failed."""


class DecontaminationHit(PBFreezeError):
    """Hermetic DECON found at least one match; nothing may be minted."""


@dataclass(frozen=True)
class PAInspectionV4:
    root: Path
    content_manifest_physical_sha256: str
    content_identity_sha256: str
    d1_ready_manifest_physical_sha256: str
    d1_ready_identity_sha256: str
    full_shard_manifest_physical_sha256: str
    full_shard_manifest_relative_path: str
    full_shard_manifest_identity_sha256: str
    full_shard_rows: tuple[dict[str, object], ...]
    full_source_summaries: tuple[dict[str, object], ...]
    screen_submanifest_physical_sha256: str
    screen_submanifest_identity_sha256: str
    screen_submanifest_relative_path: str
    d6_physical_evidence_physical_sha256: str
    d6_physical_evidence_identity_sha256: str
    d6_physical_evidence_relative_path: str
    screen_groups: tuple[dict[str, object], ...]
    screen_shard_manifest_physical_sha256: str
    screen_shard_rows: tuple[dict[str, object], ...]
    diagnostic_sha256s: tuple[tuple[str, str], ...]
    d2_evidence_descriptor_sha256: str
    release_manifest_section_identity_sha256: str


def pb_authority_bound_sha256(schema: str, value: object) -> str:
    if not isinstance(schema, str) or not schema.endswith("_v5"):
        raise ValueError("P-B receipts require an explicit v5 schema")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "authority_chain": PB_AUTHORITY_CHAIN_V5,
                "payload": value,
                "schema": schema,
            }
        )
    ).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _full_shard_projection(pa: PAInspectionV4) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "relative_path": row["relative_path"],
            "zstd_sha256": row["zstd_sha256"],
        }
        for row in pa.full_shard_rows
    )


def _full_shard_set_commitment(pa: PAInspectionV4) -> str:
    return hashlib.sha256(
        canonical_json_bytes(_full_shard_projection(pa))
    ).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PBFreezeError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_mapping(value: object, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PBFreezeError(f"{name} fields drifted")
    return value


def _canonical_relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PBFreezeError(f"{name} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PBFreezeError(f"{name} must be a canonical relative POSIX path")
    return value


def _load_canonical(path: Path, name: str) -> tuple[bytes, Mapping[str, Any]]:
    assert_no_symlink_ancestors(path)
    raw, value = load_canonical_json_snapshot(path)
    if raw != canonical_json_bytes(value) + b"\n" or not isinstance(value, Mapping):
        raise PBFreezeError(f"{name} must be canonical JSON")
    return raw, value


def _verify_content_manifest(
    content: Mapping[str, Any], *, physical_sha256: str
) -> tuple[str, str]:
    if content.get("schema") != MATERIALIZER_SCHEMA_V4:
        raise PBFreezeError("P-A content manifest is not V4")
    if tuple(content.get("authority_chain", ())) != GTOK_EXECUTION_AUTHORITY_CHAIN_V4:
        raise PBFreezeError("P-A content authority chain drifted")
    if content.get("mode") != "PRODUCTION" or content.get("readiness") != V4_READINESS:
        raise PBFreezeError("P-A content is not production D1-ready")
    if content.get("authoritative_gate_receipts") not in ([], ()):
        raise PBFreezeError("P-A content unexpectedly claims gate receipts")
    payload = dict(content)
    claimed_identity = _require_sha256(
        payload.pop("content_identity_sha256", None), "P-A content identity"
    )
    expected_identity = execution_authority_v4_bound_sha256(
        MATERIALIZED_CONTENT_SCHEMA_V4, payload
    )
    if claimed_identity != expected_identity:
        raise PBFreezeError("P-A content identity failed independent recomputation")

    release = _exact_mapping(
        content.get("release"),
        {"authority_sha256", "manifest_section", "manifest_section_identity_sha256"},
        "P-A release section",
    )
    verify_release_authority_artifact()
    expected_section = release_manifest_section()
    expected_release_identity = execution_authority_v4_bound_sha256(
        RELEASE_MANIFEST_SECTION_SCHEMA_V4, expected_section
    )
    if (
        release.get("authority_sha256") != RELEASE_AUTHORITY_SHA256
        or release.get("manifest_section") != expected_section
        or release.get("manifest_section_identity_sha256") != expected_release_identity
    ):
        raise PBFreezeError("P-A release manifest section drifted")
    if physical_sha256 == claimed_identity:
        raise PBFreezeError("physical and typed content identities were conflated")
    return claimed_identity, expected_release_identity


def _verify_d1_inventory(root: Path, d1: Mapping[str, Any]) -> str:
    if (
        d1.get("schema") != D1_READY_SCHEMA_V4
        or d1.get("gate_minted") is not False
        or d1.get("mode") != "PRODUCTION"
        or d1.get("readiness") != V4_READINESS
    ):
        raise PBFreezeError("P-A D1-ready envelope is not a completed V4 artifact")
    payload = dict(d1)
    claimed = _require_sha256(
        payload.pop("d1_ready_identity_sha256", None), "D1-ready identity"
    )
    if claimed != execution_authority_v4_bound_sha256(
        D1_READY_IDENTITY_SCHEMA_V4, payload
    ):
        raise PBFreezeError("D1-ready identity failed independent recomputation")
    rows = d1.get("file_inventory")
    if not isinstance(rows, list) or not rows:
        raise PBFreezeError("D1-ready inventory is empty")
    observed_paths: list[str] = []
    for row in rows:
        item = _exact_mapping(
            row, {"bytes", "relative_path", "sha256"}, "D1 inventory row"
        )
        relative = _canonical_relative_path(
            item["relative_path"], "D1 inventory path"
        )
        path = root.joinpath(*PurePosixPath(relative).parts)
        assert_no_symlink_ancestors(path)
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise PBFreezeError("D1 inventory file size drifted")
        if _sha256_file(path) != item["sha256"]:
            raise PBFreezeError("D1 inventory file hash drifted")
        observed_paths.append(relative)
    if observed_paths != sorted(observed_paths) or len(set(observed_paths)) != len(
        observed_paths
    ):
        raise PBFreezeError("D1 inventory paths are not unique and sorted")
    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"d1-ready-manifest.json", "child-receipt.json"}
    )
    if actual_paths != observed_paths:
        raise PBFreezeError("P-A tree contains files outside the D1 inventory")
    return claimed


def _load_full_shard_manifest(
    root: Path, content: Mapping[str, Any]
) -> tuple[
    str,
    str,
    str,
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    str,
    str,
    str,
    str,
    str,
    str,
    tuple[dict[str, object], ...],
    str,
    tuple[dict[str, object], ...],
]:
    binding = _exact_mapping(
        content.get("v4_full_corpus"),
        {
            "d6_physical_evidence_identity_sha256",
            "d6_physical_evidence_path",
            "d6_physical_evidence_sha256",
            "document_count",
            "full_manifest_identity_sha256",
            "full_manifest_path",
            "full_manifest_sha256",
            "non_screen_full_document_count",
            "retained_text_bytes",
            "screen_submanifest_identity_sha256",
            "screen_submanifest_path",
            "screen_submanifest_sha256",
        },
        "P-A V4 full-corpus binding",
    )
    relative = binding.get("full_manifest_path")
    relative = _canonical_relative_path(relative, "full-corpus shard manifest path")
    if relative != FULL_SHARD_MANIFEST_RELATIVE_PATH_V4:
        raise PBFreezeError("P-A full manifest path differs from the V4 contract")
    path = root.joinpath(*PurePosixPath(relative).parts)
    raw, manifest = _load_canonical(path, "full-corpus shard manifest")
    physical = _sha256_bytes(raw)
    if binding.get("full_manifest_sha256") != physical:
        raise PBFreezeError("P-A content does not bind the full-corpus shard manifest")
    manifest = _exact_mapping(
        manifest,
        {
            "codec_binding_sha256",
            "document_order",
            "document_count",
            "manifest_identity_sha256",
            "ordered_raw_content_ids_sha256",
            "retained_text_bytes",
            "schema",
            "shard_target_uncompressed_jsonl_bytes",
            "shard_order",
            "shards",
            "sources",
        },
        "full-corpus shard manifest",
    )
    if (
        manifest.get("schema") != FULL_SHARD_MANIFEST_SCHEMA_V4
        or manifest.get("codec_binding_sha256")
        != A2_ZSTD_CODEC_BINDING.receipt_sha256
        or manifest.get("document_order")
        != "canonical_stratum_then_canonical_source_then_full_ordinal"
        or manifest.get("shard_order")
        != "canonical_stratum_then_canonical_source_then_shard_index"
    ):
        raise PBFreezeError("full-corpus shard manifest binding drifted")
    manifest_core = dict(manifest)
    manifest_identity = _require_sha256(
        manifest_core.pop("manifest_identity_sha256", None),
        "full manifest identity",
    )
    if manifest_identity != execution_authority_v4_bound_sha256(
        FULL_SHARD_MANIFEST_SCHEMA_V4, manifest_core
    ):
        raise PBFreezeError("full-corpus manifest identity drifted")
    if (
        binding.get("full_manifest_identity_sha256") != manifest_identity
        or type(manifest.get("document_count")) is not int
        or int(manifest["document_count"]) < 1
        or type(manifest.get("retained_text_bytes")) is not int
        or int(manifest["retained_text_bytes"]) < 1
        or type(manifest.get("shard_target_uncompressed_jsonl_bytes")) is not int
        or int(manifest["shard_target_uncompressed_jsonl_bytes"]) < 1
        or binding.get("document_count") != manifest.get("document_count")
        or binding.get("retained_text_bytes") != manifest.get("retained_text_bytes")
    ):
        raise PBFreezeError("full-corpus summary binding drifted")
    _require_sha256(
        manifest.get("ordered_raw_content_ids_sha256"),
        "full ordered raw-content IDs",
    )
    source_rows = manifest.get("sources")
    if (
        not isinstance(source_rows, list)
        or tuple(row.get("source") for row in source_rows) != SOURCE_FAMILIES
    ):
        raise PBFreezeError("full source summaries are incomplete or noncanonical")
    normalized_sources: list[dict[str, object]] = []
    source_strata: dict[str, str] = {}
    for raw_source in source_rows:
        source = dict(
            _exact_mapping(
                raw_source,
                {"document_count", "retained_text_bytes", "source", "stratum"},
                "full source summary",
            )
        )
        if (
            source.get("stratum") not in GTOK_STRATA
            or type(source.get("document_count")) is not int
            or int(source["document_count"]) < 1
            or type(source.get("retained_text_bytes")) is not int
            or int(source["retained_text_bytes"]) < 1
        ):
            raise PBFreezeError("full source summary accounting drifted")
        source_strata[str(source["source"])] = str(source["stratum"])
        normalized_sources.append(source)
    if (
        sum(int(row["document_count"]) for row in normalized_sources)
        != manifest["document_count"]
        or sum(int(row["retained_text_bytes"]) for row in normalized_sources)
        != manifest["retained_text_bytes"]
    ):
        raise PBFreezeError("full source summaries do not add to the corpus")
    rows = manifest.get("shards")
    if not isinstance(rows, list) or not rows:
        raise PBFreezeError("full-corpus shard manifest has no shards")
    normalized: list[dict[str, object]] = []
    paths: list[str] = []
    for raw_row in rows:
        row = dict(_exact_mapping(raw_row, _SHARD_KEYS, "full shard row"))
        relative_path = _canonical_relative_path(
            row["relative_path"], "full shard path"
        )
        if (
            row.get("stream") != "FULL"
            or row.get("stratum") not in GTOK_STRATA
            or row.get("source") not in SOURCE_FAMILIES
            or source_strata.get(str(row.get("source"))) != row.get("stratum")
            or row.get("codec_binding_sha256")
            != A2_ZSTD_CODEC_BINDING.receipt_sha256
        ):
            raise PBFreezeError("full shard stream, stratum, or codec drifted")
        for name in (
            "record_count",
            "retained_text_bytes",
            "logical_jsonl_bytes",
            "zstd_bytes",
        ):
            if type(row.get(name)) is not int or int(row[name]) < 1:
                raise PBFreezeError("full shard byte/count field drifted")
        for name in (
            "logical_jsonl_sha256",
            "zstd_sha256",
            "content_identity_sha256",
        ):
            _require_sha256(row.get(name), f"full shard {name}")
        for name in ("first_full_ordinal", "last_full_ordinal"):
            if type(row.get(name)) is not int or int(row[name]) < 0:
                raise PBFreezeError("full shard ordinal field drifted")
        if int(row["first_full_ordinal"]) > int(row["last_full_ordinal"]):
            raise PBFreezeError("full shard ordinal interval is reversed")
        identity_relative = _canonical_relative_path(
            row.get("identity_relative_path"), "full shard identity path"
        )
        identity = JsonlZstdShardIdentityV3(
            relative_path=identity_relative,
            record_count=int(row["record_count"]),
            retained_text_bytes=int(row["retained_text_bytes"]),
            logical_jsonl_sha256=str(row["logical_jsonl_sha256"]),
            logical_jsonl_bytes=int(row["logical_jsonl_bytes"]),
            zstd_sha256=str(row["zstd_sha256"]),
            zstd_bytes=int(row["zstd_bytes"]),
            codec_binding_sha256=str(row["codec_binding_sha256"]),
        )
        if (
            relative_path
            != f"full-shards/{row['source']}/{identity_relative}"
            or row.get("content_identity_sha256")
            != identity.content_identity_sha256
        ):
            raise PBFreezeError("full shard typed identity drifted")
        paths.append(relative_path)
        normalized.append(row)
    canonical_path_keys = tuple(
        (
            GTOK_STRATA.index(str(row["stratum"])),
            SOURCE_FAMILIES.index(str(row["source"])),
            str(row["relative_path"]),
        )
        for row in normalized
    )
    if canonical_path_keys != tuple(sorted(canonical_path_keys)) or len(paths) != len(
        set(paths)
    ):
        raise PBFreezeError("full shard paths are not unique and canonically ordered")
    if (
        sum(int(row["record_count"]) for row in normalized)
        != manifest["document_count"]
        or sum(int(row["retained_text_bytes"]) for row in normalized)
        != manifest["retained_text_bytes"]
    ):
        raise PBFreezeError("full shard rows do not add to the manifest summary")

    screen_relative = _canonical_relative_path(
        binding.get("screen_submanifest_path"), "screen submanifest path"
    )
    if screen_relative != SCREEN_SUBMANIFEST_RELATIVE_PATH_V4:
        raise PBFreezeError("P-A screen submanifest path differs from V4")
    screen_path = root.joinpath(*PurePosixPath(screen_relative).parts)
    screen_raw, screen = _load_canonical(screen_path, "screen submanifest")
    screen_physical = _sha256_bytes(screen_raw)
    if binding.get("screen_submanifest_sha256") != screen_physical:
        raise PBFreezeError("P-A content does not bind the screen submanifest")
    screen = _exact_mapping(
        screen,
        {
            "d6_physical_evidence_identity_sha256",
            "d6_physical_evidence_path",
            "d6_physical_evidence_sha256",
            "full_manifest_identity_sha256",
            "full_manifest_path",
            "full_manifest_sha256",
            "groups",
            "missing_full_document_count",
            "non_screen_full_document_count",
            "schema",
            "screen_document_count",
            "screen_shard_manifest_path",
            "screen_shard_manifest_sha256",
            "submanifest_identity_sha256",
        },
        "screen submanifest",
    )
    screen_core = dict(screen)
    screen_identity = _require_sha256(
        screen_core.pop("submanifest_identity_sha256", None),
        "screen submanifest identity",
    )
    if (
        screen.get("schema") != SCREEN_SUBMANIFEST_SCHEMA_V4
        or screen_identity
        != execution_authority_v4_bound_sha256(
            SCREEN_SUBMANIFEST_SCHEMA_V4, screen_core
        )
        or binding.get("screen_submanifest_identity_sha256") != screen_identity
        or screen.get("full_manifest_identity_sha256") != manifest_identity
        or screen.get("full_manifest_sha256") != physical
        or screen.get("full_manifest_path") != relative
        or screen.get("d6_physical_evidence_identity_sha256")
        != binding.get("d6_physical_evidence_identity_sha256")
        or screen.get("d6_physical_evidence_path")
        != binding.get("d6_physical_evidence_path")
        or screen.get("d6_physical_evidence_sha256")
        != binding.get("d6_physical_evidence_sha256")
        or screen.get("missing_full_document_count") != 0
        or screen.get("non_screen_full_document_count")
        != binding.get("non_screen_full_document_count")
    ):
        raise PBFreezeError("screen submanifest parent binding drifted")
    d6_evidence_relative = _canonical_relative_path(
        screen.get("d6_physical_evidence_path"),
        "physical D6 evidence path",
    )
    if d6_evidence_relative != D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4:
        raise PBFreezeError("physical D6 evidence path differs from V4")
    d6_evidence_physical = _require_sha256(
        screen.get("d6_physical_evidence_sha256"),
        "physical D6 evidence SHA-256",
    )
    d6_evidence_identity = _require_sha256(
        screen.get("d6_physical_evidence_identity_sha256"),
        "physical D6 evidence identity",
    )
    d6_evidence_path = root.joinpath(
        *PurePosixPath(d6_evidence_relative).parts
    )
    try:
        if _sha256_file(d6_evidence_path) != d6_evidence_physical:
            raise PBFreezeError("physical D6 evidence transport hash drifted")
    except OSError as error:
        raise PBFreezeError("physical D6 evidence artifact is absent") from error
    screen_count = screen.get("screen_document_count")
    non_screen_count = screen.get("non_screen_full_document_count")
    if (
        type(screen_count) is not int
        or screen_count < 1
        or type(non_screen_count) is not int
        or non_screen_count < 0
        or screen_count + non_screen_count != manifest["document_count"]
    ):
        raise PBFreezeError("screen/full membership count drifted")
    groups = screen.get("groups")
    expected_group_keys = {
        "document_count",
        "full_location_projection_sha256",
        "ordered_raw_content_ids_sha256",
        "retained_text_bytes",
        "stratum",
        "stream",
    }
    expected_group_order = tuple(
        (stream, stratum)
        for stream in ("T", "H")
        for stratum in GTOK_STRATA
    )
    if (
        not isinstance(groups, list)
        or tuple((row.get("stream"), row.get("stratum")) for row in groups)
        != expected_group_order
    ):
        raise PBFreezeError("screen membership groups are incomplete or noncanonical")
    normalized_groups: list[dict[str, object]] = []
    for raw_group in groups:
        group = dict(_exact_mapping(raw_group, expected_group_keys, "screen group"))
        if (
            type(group.get("document_count")) is not int
            or int(group["document_count"]) < 0
            or type(group.get("retained_text_bytes")) is not int
            or int(group["retained_text_bytes"]) < 0
        ):
            raise PBFreezeError("screen group accounting drifted")
        _require_sha256(
            group.get("ordered_raw_content_ids_sha256"), "screen raw-content IDs"
        )
        _require_sha256(
            group.get("full_location_projection_sha256"), "screen locations"
        )
        normalized_groups.append(group)
    if sum(int(group["document_count"]) for group in normalized_groups) != screen_count:
        raise PBFreezeError("screen groups do not add to the screen total")

    screen_shard_relative = _canonical_relative_path(
        screen.get("screen_shard_manifest_path"), "screen shard manifest path"
    )
    if screen_shard_relative != "artifacts/shard-manifest.json":
        raise PBFreezeError("screen shard manifest path differs from V4")
    screen_shard_path = root.joinpath(*PurePosixPath(screen_shard_relative).parts)
    screen_shard_raw, screen_shard_manifest = _load_canonical(
        screen_shard_path, "screen shard manifest"
    )
    screen_shard_physical = _sha256_bytes(screen_shard_raw)
    screen_shard_manifest = _exact_mapping(
        screen_shard_manifest,
        {
            "codec_binding_sha256",
            "schema",
            "shards",
            "tokenizer_fit_input_receipt_sha256",
        },
        "screen shard manifest",
    )
    if (
        screen.get("screen_shard_manifest_sha256") != screen_shard_physical
        or screen_shard_manifest.get("schema") != "weft1_corpus_shard_manifest_v3"
        or screen_shard_manifest.get("codec_binding_sha256")
        != A2_ZSTD_CODEC_BINDING.receipt_sha256
    ):
        raise PBFreezeError("screen shard manifest binding drifted")
    _require_sha256(
        screen_shard_manifest.get("tokenizer_fit_input_receipt_sha256"),
        "screen tokenizer-fit receipt",
    )
    screen_rows = screen_shard_manifest.get("shards")
    if not isinstance(screen_rows, list) or not screen_rows:
        raise PBFreezeError("screen shard manifest has no shards")
    normalized_screen_rows: list[dict[str, object]] = []
    screen_paths: list[str] = []
    for raw_row in screen_rows:
        row = dict(_exact_mapping(raw_row, _SCREEN_SHARD_KEYS, "screen shard row"))
        relative_path = _canonical_relative_path(
            row.get("relative_path"), "screen shard path"
        )
        identity_relative = _canonical_relative_path(
            row.get("identity_relative_path"), "screen shard identity path"
        )
        if (
            row.get("stream") not in {"T", "H"}
            or row.get("stratum") not in GTOK_STRATA
            or row.get("codec_binding_sha256")
            != A2_ZSTD_CODEC_BINDING.receipt_sha256
        ):
            raise PBFreezeError("screen shard stream, stratum, or codec drifted")
        for name in (
            "record_count",
            "retained_text_bytes",
            "logical_jsonl_bytes",
            "zstd_bytes",
        ):
            if type(row.get(name)) is not int or int(row[name]) < 1:
                raise PBFreezeError("screen shard byte/count field drifted")
        for name in (
            "logical_jsonl_sha256",
            "zstd_sha256",
            "content_identity_sha256",
        ):
            _require_sha256(row.get(name), f"screen shard {name}")
        identity = JsonlZstdShardIdentityV3(
            relative_path=identity_relative,
            record_count=int(row["record_count"]),
            retained_text_bytes=int(row["retained_text_bytes"]),
            logical_jsonl_sha256=str(row["logical_jsonl_sha256"]),
            logical_jsonl_bytes=int(row["logical_jsonl_bytes"]),
            zstd_sha256=str(row["zstd_sha256"]),
            zstd_bytes=int(row["zstd_bytes"]),
            codec_binding_sha256=str(row["codec_binding_sha256"]),
        )
        if (
            relative_path != f"shards/{identity_relative}"
            or row.get("content_identity_sha256")
            != identity.content_identity_sha256
        ):
            raise PBFreezeError("screen shard typed identity drifted")
        screen_paths.append(relative_path)
        normalized_screen_rows.append(row)
    if screen_paths != sorted(screen_paths) or len(screen_paths) != len(
        set(screen_paths)
    ):
        raise PBFreezeError("screen shard paths are not unique and sorted")
    return (
        relative,
        physical,
        manifest_identity,
        tuple(normalized),
        tuple(normalized_sources),
        screen_relative,
        screen_physical,
        screen_identity,
        d6_evidence_relative,
        d6_evidence_physical,
        d6_evidence_identity,
        tuple(normalized_groups),
        screen_shard_physical,
        tuple(normalized_screen_rows),
    )


def inspect_pa_v4(materialization_root: Path) -> PAInspectionV4:
    if not isinstance(materialization_root, Path):
        raise TypeError("materialization_root must be pathlib.Path")
    root = assert_no_symlink_ancestors(materialization_root).resolve(strict=True)
    if not root.is_dir() or (root / "_INCOMPLETE").exists():
        raise PBFreezeError("P-A root is absent or incomplete")
    content_raw, content = _load_canonical(
        root / "content-manifest.json", "P-A content manifest"
    )
    content_physical = _sha256_bytes(content_raw)
    content_identity, release_identity = _verify_content_manifest(
        content, physical_sha256=content_physical
    )
    d1_raw, d1 = _load_canonical(root / "d1-ready-manifest.json", "D1 envelope")
    if d1.get("content_identity_sha256") != content_identity:
        raise PBFreezeError("D1 envelope points at a different content identity")
    d1_identity = _verify_d1_inventory(root, d1)
    (
        relative,
        full_physical,
        full_identity,
        full_rows,
        full_sources,
        screen_relative,
        screen_physical,
        screen_identity,
        d6_evidence_relative,
        d6_evidence_physical,
        d6_evidence_identity,
        screen_groups,
        screen_shard_physical,
        screen_rows,
    ) = _load_full_shard_manifest(root, content)
    diagnostics = content.get("diagnostic_sha256s")
    if (
        not isinstance(diagnostics, list)
        or any(not isinstance(row, list) or len(row) != 2 for row in diagnostics)
        or tuple(row[0] for row in diagnostics) != ("D3", "D4", "D5", "D6")
    ):
        raise PBFreezeError("P-A diagnostic hash inventory drifted")
    normalized_diagnostics: list[tuple[str, str]] = []
    for gate, claimed in diagnostics:
        digest = _require_sha256(claimed, f"{gate} diagnostic hash")
        observed = _sha256_file(root / "diagnostics" / f"{gate.lower()}.json")
        if digest != observed:
            raise PBFreezeError(f"{gate} diagnostic changed after P-A")
        normalized_diagnostics.append((gate, digest))
    d2_sha = _require_sha256(
        content.get("d2_evidence_descriptor_sha256"), "D2 descriptor hash"
    )
    if _sha256_file(root / "artifacts" / "d2-evidence-descriptor.json") != d2_sha:
        raise PBFreezeError("D2 descriptor changed after P-A")
    return PAInspectionV4(
        root=root,
        content_manifest_physical_sha256=content_physical,
        content_identity_sha256=content_identity,
        d1_ready_manifest_physical_sha256=_sha256_bytes(d1_raw),
        d1_ready_identity_sha256=d1_identity,
        full_shard_manifest_physical_sha256=full_physical,
        full_shard_manifest_relative_path=relative,
        full_shard_manifest_identity_sha256=full_identity,
        full_shard_rows=full_rows,
        full_source_summaries=full_sources,
        screen_submanifest_physical_sha256=screen_physical,
        screen_submanifest_identity_sha256=screen_identity,
        screen_submanifest_relative_path=screen_relative,
        d6_physical_evidence_physical_sha256=d6_evidence_physical,
        d6_physical_evidence_identity_sha256=d6_evidence_identity,
        d6_physical_evidence_relative_path=d6_evidence_relative,
        screen_groups=screen_groups,
        screen_shard_manifest_physical_sha256=screen_shard_physical,
        screen_shard_rows=screen_rows,
        diagnostic_sha256s=tuple(normalized_diagnostics),
        d2_evidence_descriptor_sha256=d2_sha,
        release_manifest_section_identity_sha256=release_identity,
    )


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PBFreezeError(f"shard JSON repeats key {key!r}")
        value[key] = item
    return value


def _scan_full_shards(pa: PAInspectionV4) -> dict[str, object]:
    try:
        import zstandard
    except ImportError as error:  # pragma: no cover - exact runtime has the pin
        raise PBFreezeError("P-B requires the pinned zstandard runtime") from error

    raw_bytes = {stratum: 0 for stratum in GTOK_STRATA}
    non_ascii_bytes = {stratum: 0 for stratum in GTOK_STRATA}
    document_counts = {stratum: 0 for stratum in GTOK_STRATA}
    source_bytes = {source: 0 for source in SOURCE_FAMILIES}
    source_document_counts = {source: 0 for source in SOURCE_FAMILIES}
    stream_digest = hashlib.sha256()
    ordered_document_ids = hashlib.sha256()
    full_document_count = 0
    # An empty SQLite filename is a private, disk-backed temporary database
    # deleted on close.  Store only compact commitments and locations here;
    # retaining full corpus text (or a Python set of every ID) would exceed the
    # Colab runtime's memory at the ratified 38 GB corpus scale.
    membership = sqlite3.connect("")
    membership.execute("PRAGMA temp_store=FILE")
    membership.execute("PRAGMA journal_mode=OFF")
    membership.execute("PRAGMA synchronous=OFF")
    membership.execute(
        "CREATE TABLE full_membership ("
        "document_id TEXT PRIMARY KEY, full_shard_index INTEGER NOT NULL, "
        "shard_record_ordinal INTEGER NOT NULL, payload_sha256 TEXT NOT NULL, "
        "text_sha256 TEXT NOT NULL, text_bytes INTEGER NOT NULL, "
        "screen_seen INTEGER NOT NULL DEFAULT 0 CHECK(screen_seen IN (0, 1))"
        ") WITHOUT ROWID, STRICT"
    )
    shard_rehashes: list[dict[str, object]] = []
    full_rows_in_corpus_order = tuple(
        row
        for stratum in GTOK_STRATA
        for row in pa.full_shard_rows
        if row["stratum"] == stratum
    )
    full_paths_in_corpus_order = tuple(
        str(row["relative_path"]) for row in full_rows_in_corpus_order
    )
    membership.execute("BEGIN IMMEDIATE")
    for full_shard_index, row in enumerate(full_rows_in_corpus_order):
        relative = str(row["relative_path"])
        path = pa.root.joinpath(*PurePosixPath(relative).parts)
        assert_no_symlink_ancestors(path)
        if not path.is_file() or path.stat().st_size != row["zstd_bytes"]:
            raise PBFreezeError("full shard transport size drifted")
        observed_zstd = _sha256_file(path)
        if observed_zstd != row["zstd_sha256"]:
            raise PBFreezeError("full shard transport hash drifted")
        logical_digest = hashlib.sha256()
        logical_bytes = 0
        retained_bytes = 0
        record_count = 0
        stratum = str(row["stratum"])
        with path.open("rb") as compressed:
            try:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                    for line in io.BufferedReader(reader):
                        logical_digest.update(line)
                        logical_bytes += len(line)
                        if not line.endswith(b"\n"):
                            raise PBFreezeError("full shard record lacks LF framing")
                        try:
                            item = json.loads(
                                line[:-1].decode("utf-8", errors="strict"),
                                object_pairs_hook=_json_no_duplicates,
                                parse_constant=lambda value: (_ for _ in ()).throw(
                                    PBFreezeError(
                                        f"non-finite shard JSON value: {value}"
                                    )
                                ),
                            )
                        except (UnicodeError, json.JSONDecodeError) as error:
                            raise PBFreezeError(
                                "full shard is not strict UTF-8 JSONL"
                            ) from error
                        item = _exact_mapping(
                            item,
                            {"id", "source", "stratum", "text"},
                            "shard record",
                        )
                        if (
                            item["stratum"] != stratum
                            or item["source"] != row["source"]
                            or item["source"] not in SOURCE_FAMILIES
                            or not isinstance(item["source"], str)
                            or not item["source"]
                            or not isinstance(item["text"], str)
                        ):
                            raise PBFreezeError(
                                "full shard record stratum/text drifted"
                            )
                        text_bytes = item["text"].encode(
                            "utf-8", errors="strict"
                        )
                        record_id = item["id"]
                        if (
                            not isinstance(record_id, str)
                            or _SHA1.fullmatch(record_id) is None
                            or record_id
                            != hashlib.sha1(text_bytes).hexdigest()  # noqa: S324
                        ):
                            raise PBFreezeError("full shard record ID drifted")
                        encoded_id = record_id.encode("ascii")
                        ordered_document_ids.update(
                            len(encoded_id).to_bytes(8, "big")
                        )
                        ordered_document_ids.update(encoded_id)
                        inserted = membership.execute(
                            "INSERT OR IGNORE INTO full_membership "
                            "(document_id, full_shard_index, shard_record_ordinal, "
                            "payload_sha256, text_sha256, text_bytes) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                record_id,
                                full_shard_index,
                                record_count,
                                hashlib.sha256(
                                    canonical_json_bytes(
                                        {
                                            "source": item["source"],
                                            "stratum": item["stratum"],
                                            "text": item["text"],
                                        }
                                    )
                                ).hexdigest(),
                                hashlib.sha256(text_bytes).hexdigest(),
                                len(text_bytes),
                            ),
                        ).rowcount
                        if inserted != 1:
                            raise PBFreezeError(
                                "full corpus repeats a raw-content ID"
                            )
                        full_document_count += 1
                        retained_bytes += len(text_bytes)
                        raw_bytes[stratum] += len(text_bytes)
                        non_ascii_bytes[stratum] += sum(
                            byte >= 0x80 for byte in text_bytes
                        )
                        document_counts[stratum] += 1
                        source_bytes[str(item["source"])] += len(text_bytes)
                        source_document_counts[str(item["source"])] += 1
                        stream_digest.update(len(text_bytes).to_bytes(8, "big"))
                        stream_digest.update(text_bytes)
                        record_count += 1
            except zstandard.ZstdError as error:
                raise PBFreezeError("full shard zstd frame is invalid") from error
        if (
            logical_digest.hexdigest() != row["logical_jsonl_sha256"]
            or logical_bytes != row["logical_jsonl_bytes"]
            or retained_bytes != row["retained_text_bytes"]
            or record_count != row["record_count"]
        ):
            raise PBFreezeError("full shard logical accounting drifted")
        shard_rehashes.append(
            {
                "relative_path": relative,
                "zstd_sha256": observed_zstd,
            }
        )
    membership.commit()

    _, full_manifest = _load_canonical(
        pa.root.joinpath(
            *PurePosixPath(pa.full_shard_manifest_relative_path).parts
        ),
        "full-corpus shard manifest",
    )
    if (
        ordered_document_ids.hexdigest()
        != full_manifest.get("ordered_raw_content_ids_sha256")
        or full_document_count != full_manifest.get("document_count")
    ):
        raise PBFreezeError("full-corpus ordered document commitment drifted")
    for expected_source in pa.full_source_summaries:
        source = str(expected_source["source"])
        if (
            source_bytes[source] != expected_source["retained_text_bytes"]
            or source_document_counts[source] != expected_source["document_count"]
        ):
            raise PBFreezeError("full source summary differs from physical shards")

    membership_groups: dict[tuple[str, str], dict[str, object]] = {}
    for stream in ("T", "H"):
        for stratum in GTOK_STRATA:
            membership_groups[(stream, stratum)] = {
                "count": 0,
                "ids": hashlib.sha256(),
                "locations": hashlib.sha256(),
                "retained": 0,
            }
    screen_document_count = 0
    screen_stream_stats = {
        stream: {"bytes": 0, "count": 0, "digest": hashlib.sha256()}
        for stream in ("T", "H")
    }
    for row in pa.screen_shard_rows:
        relative = str(row["relative_path"])
        path = pa.root.joinpath(*PurePosixPath(relative).parts)
        assert_no_symlink_ancestors(path)
        if (
            not path.is_file()
            or path.stat().st_size != row["zstd_bytes"]
            or _sha256_file(path) != row["zstd_sha256"]
        ):
            raise PBFreezeError("screen shard transport identity drifted")
        logical_digest = hashlib.sha256()
        logical_bytes = 0
        retained = 0
        count = 0
        stream = str(row["stream"])
        stratum = str(row["stratum"])
        group = membership_groups[(stream, stratum)]
        with path.open("rb") as compressed:
            try:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                    for line in io.BufferedReader(reader):
                        logical_digest.update(line)
                        logical_bytes += len(line)
                        if not line.endswith(b"\n"):
                            raise PBFreezeError("screen shard record lacks LF framing")
                        try:
                            item = json.loads(
                                line[:-1].decode("utf-8", errors="strict"),
                                object_pairs_hook=_json_no_duplicates,
                            )
                        except (UnicodeError, json.JSONDecodeError) as error:
                            raise PBFreezeError(
                                "screen shard is not strict UTF-8 JSONL"
                            ) from error
                        item = _exact_mapping(
                            item,
                            {"id", "source", "stratum", "text"},
                            "screen shard record",
                        )
                        record_id = item.get("id")
                        if (
                            not isinstance(record_id, str)
                            or not isinstance(item.get("source"), str)
                            or not isinstance(item.get("stratum"), str)
                            or not isinstance(item.get("text"), str)
                        ):
                            raise PBFreezeError("screen document fields drifted")
                        full = membership.execute(
                            "SELECT full_shard_index, shard_record_ordinal, "
                            "payload_sha256, text_sha256, text_bytes, screen_seen "
                            "FROM full_membership WHERE document_id = ?",
                            (record_id,),
                        ).fetchone()
                        text_bytes = str(item["text"]).encode("utf-8")
                        payload_sha256 = hashlib.sha256(
                            canonical_json_bytes(
                                {
                                    "source": item["source"],
                                    "stratum": item["stratum"],
                                    "text": item["text"],
                                }
                            )
                        ).hexdigest()
                        if (
                            full is None
                            or full[2] != payload_sha256
                            or full[3] != hashlib.sha256(text_bytes).hexdigest()
                            or full[4] != len(text_bytes)
                            or full[5] != 0
                        ):
                            raise PBFreezeError(
                                "screen document is not an exact full-corpus member"
                            )
                        if item.get("stratum") != stratum:
                            raise PBFreezeError("screen shard stratum drifted")
                        changed = membership.execute(
                            "UPDATE full_membership SET screen_seen = 1 "
                            "WHERE document_id = ? AND screen_seen = 0",
                            (record_id,),
                        ).rowcount
                        if changed != 1:
                            raise PBFreezeError("screen membership marking drifted")
                        screen_document_count += 1
                        retained += len(text_bytes)
                        count += 1
                        screen_stream_stats[stream]["bytes"] += len(text_bytes)
                        screen_stream_stats[stream]["count"] += 1
                        screen_stream_stats[stream]["digest"].update(
                            len(text_bytes).to_bytes(8, "big")
                        )
                        screen_stream_stats[stream]["digest"].update(text_bytes)
                        encoded = record_id.encode("ascii")
                        group["ids"].update(len(encoded).to_bytes(8, "big"))
                        group["ids"].update(encoded)
                        location = canonical_json_bytes(
                            {
                                "raw_content_id": record_id,
                                "source": item["source"],
                                "full_shard_relative_path": (
                                    full_paths_in_corpus_order[int(full[0])]
                                ),
                                "shard_record_ordinal": int(full[1]),
                            }
                        )
                        group["locations"].update(len(location).to_bytes(8, "big"))
                        group["locations"].update(location)
                        group["count"] += 1
                        group["retained"] += len(text_bytes)
            except zstandard.ZstdError as error:
                raise PBFreezeError("screen shard zstd frame is invalid") from error
        if (
            logical_digest.hexdigest() != row["logical_jsonl_sha256"]
            or logical_bytes != row["logical_jsonl_bytes"]
            or retained != row["retained_text_bytes"]
            or count != row["record_count"]
        ):
            raise PBFreezeError("screen shard logical accounting drifted")
    for expected in pa.screen_groups:
        group = membership_groups[(str(expected["stream"]), str(expected["stratum"]))]
        if (
            group["count"] != expected["document_count"]
            or group["retained"] != expected["retained_text_bytes"]
            or group["ids"].hexdigest()
            != expected["ordered_raw_content_ids_sha256"]
            or group["locations"].hexdigest()
            != expected["full_location_projection_sha256"]
        ):
            raise PBFreezeError("screen/full membership projection drifted")
    if screen_document_count != sum(
        int(group["document_count"]) for group in pa.screen_groups
    ):
        raise PBFreezeError("screen membership count drifted")
    membership.commit()
    membership.close()

    d3_raw, d3 = _load_canonical(pa.root / "diagnostics" / "d3.json", "D3 diagnostic")
    expected_d3_sha = dict(pa.diagnostic_sha256s)["D3"]
    if _sha256_bytes(d3_raw) != expected_d3_sha:
        raise PBFreezeError("D3 diagnostic physical identity drifted")
    d3 = _exact_mapping(
        d3,
        {"full_pool_rows", "gate", "observed_stratum_bytes", "pool_receipts", "status"},
        "D3 diagnostic",
    )
    if d3.get("gate") != "D3" or d3.get("status") != "CHECK_PASS_NO_GATE_MINT":
        raise PBFreezeError("P-A D3 diagnostic claims an invalid status")
    observed_rows = d3.get("observed_stratum_bytes")
    if (
        not isinstance(observed_rows, list)
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or type(row[1]) is not int
            for row in observed_rows
        )
        or tuple(row[0] for row in observed_rows) != GTOK_STRATA
    ):
        raise PBFreezeError("D3 stratum rows drifted")
    d3_bytes = {stratum: count for stratum, count in observed_rows}
    if d3_bytes != raw_bytes:
        raise PBFreezeError(
            "full materialized shards do not equal the D3 full-corpus composition"
        )
    for stratum, target in FULL_STRATUM_TARGETS_V4.items():
        observed = raw_bytes[stratum]
        if observed > target or Fraction(target - observed, target) > FIRST_FIT_TOLERANCE:
            raise PBFreezeError("full corpus stratum lies outside A2 first-fit tolerance")
    for required in ("code", "mathematics"):
        if non_ascii_bytes[required] == 0:
            raise PBFreezeError(f"C1 failed: {required} contains zero non-ASCII bytes")

    strata = tuple(
        {
            "document_count": document_counts[stratum],
            "name": stratum,
            "non_ascii_byte_count": non_ascii_bytes[stratum],
            "non_ascii_fraction": {
                "denominator": raw_bytes[stratum],
                "numerator": non_ascii_bytes[stratum],
            },
            "raw_byte_count": raw_bytes[stratum],
        }
        for stratum in GTOK_STRATA
    )
    return {
        "c1_status": "PASS",
        "c3_status": "PASS",
        "full_text_stream_sha256": stream_digest.hexdigest(),
        "shard_rehashes": tuple(shard_rehashes),
        "screen_streams": tuple(
            {
                "document_count": screen_stream_stats[stream]["count"],
                "framed_retained_text_sha256": screen_stream_stats[stream][
                    "digest"
                ].hexdigest(),
                "retained_text_bytes": screen_stream_stats[stream]["bytes"],
                "stream": stream,
            }
            for stream in ("T", "H")
        ),
        "sources": tuple(
            {
                "document_count": source_document_counts[source],
                "name": source,
                "raw_byte_count": source_bytes[source],
            }
            for source in SOURCE_FAMILIES
        ),
        "strata": strata,
        "total_document_count": sum(document_counts.values()),
        "total_raw_bytes": sum(raw_bytes.values()),
    }


def build_c2_fixture_evidence() -> dict[str, object]:
    """Round-trip every C2 byte fixture through the production shard path."""

    if tuple(sorted(_C2_PAYLOADS)) != tuple(sorted(GTOK_ROUND_TRIP_CATEGORIES)):
        raise PBFreezeError("C2 fixture binding does not cover the registered categories")
    categories = tuple(sorted(_C2_PAYLOADS))
    documents = tuple(
        StableDocumentV3(
            source="wikipedia_wikibooks",
            stratum="general",
            stable_source_record_id=hashlib.sha256(
                b"WEFT-1/C2/fixture/v1\x00" + category.encode("ascii")
            ).hexdigest(),
            text=_C2_PAYLOADS[category].decode("utf-8", errors="strict"),
        )
        for category in categories
    )
    try:
        with tempfile.TemporaryDirectory(prefix="weft1-c2-") as directory:
            shard_root = Path(directory) / "shards"
            result = corpus_pa.write_jsonl_zstd_shards_v3(
                documents,
                shard_root,
                stream="T",
                stratum="general",
                shard_target_bytes=1024 * 1024,
            )
            restored_texts = tuple(
                iter_a2_shard_texts(shard_root, tuple(result.shards))
            )
    except (OSError, UnicodeError, ValueError) as error:
        raise PBFreezeError("C2 production shard round trip failed") from error
    if len(restored_texts) != len(categories):
        raise PBFreezeError("C2 production shard round trip changed fixture count")
    cases: list[dict[str, object]] = []
    for category, restored_text in zip(categories, restored_texts, strict=True):
        original = _C2_PAYLOADS[category]
        restored = restored_text.encode("utf-8", errors="strict")
        if restored != original:
            raise PBFreezeError("C2 production shard round trip changed fixture bytes")
        cases.append(
            {
                "category": category,
                "fixture_id": f"fixture-{category.replace('_', '-')}",
                "original_bytes_b64": base64.b64encode(original).decode("ascii"),
                "original_sha256": _sha256_bytes(original),
                "round_trip_bytes_b64": base64.b64encode(restored).decode("ascii"),
                "round_trip_sha256": _sha256_bytes(restored),
            }
        )
    payload: dict[str, object] = {
        "authority_chain": PB_AUTHORITY_CHAIN_V5,
        "cases": cases,
        "codec_binding_sha256": A2_ZSTD_CODEC_BINDING.receipt_sha256,
        "schema": PB_C2_EVIDENCE_SCHEMA_V5,
        "status": "CHECK_PASS_NO_FREEZE_MINT",
    }
    payload["suite_identity_sha256"] = pb_authority_bound_sha256(
        PB_C2_EVIDENCE_SCHEMA_V5, payload
    )
    return payload


def load_c2_fixture_evidence(path: Path) -> tuple[str, str]:
    raw, evidence = _load_canonical(path, "C2 fixture evidence")
    expected_keys = {
        "authority_chain",
        "cases",
        "codec_binding_sha256",
        "schema",
        "status",
        "suite_identity_sha256",
    }
    evidence = _exact_mapping(evidence, expected_keys, "C2 fixture evidence")
    if (
        evidence.get("schema") != PB_C2_EVIDENCE_SCHEMA_V5
        or tuple(evidence.get("authority_chain", ())) != PB_AUTHORITY_CHAIN_V5
        or evidence.get("status") != "CHECK_PASS_NO_FREEZE_MINT"
        or evidence.get("codec_binding_sha256")
        != A2_ZSTD_CODEC_BINDING.receipt_sha256
    ):
        raise PBFreezeError("C2 evidence authority or codec drifted")
    payload = dict(evidence)
    claimed = _require_sha256(
        payload.pop("suite_identity_sha256", None), "C2 suite identity"
    )
    if claimed != pb_authority_bound_sha256(PB_C2_EVIDENCE_SCHEMA_V5, payload):
        raise PBFreezeError("C2 suite identity drifted")
    cases = evidence.get("cases")
    if not isinstance(cases, list) or tuple(row.get("category") for row in cases) != tuple(
        sorted(GTOK_ROUND_TRIP_CATEGORIES)
    ):
        raise PBFreezeError("C2 fixture coverage is incomplete or noncanonical")
    for row in cases:
        item = _exact_mapping(
            row,
            {
                "category",
                "fixture_id",
                "original_bytes_b64",
                "original_sha256",
                "round_trip_bytes_b64",
                "round_trip_sha256",
            },
            "C2 fixture case",
        )
        try:
            original = base64.b64decode(item["original_bytes_b64"], validate=True)
            restored = base64.b64decode(item["round_trip_bytes_b64"], validate=True)
        except (TypeError, ValueError) as error:
            raise PBFreezeError("C2 fixture base64 is invalid") from error
        category = item.get("category")
        if (
            category not in _C2_PAYLOADS
            or item.get("fixture_id")
            != f"fixture-{str(category).replace('_', '-')}"
            or original != _C2_PAYLOADS[str(category)]
        ):
            raise PBFreezeError("C2 fixture payload differs from the registered case")
        if (
            base64.b64encode(original).decode("ascii") != item["original_bytes_b64"]
            or base64.b64encode(restored).decode("ascii")
            != item["round_trip_bytes_b64"]
            or _sha256_bytes(original) != item["original_sha256"]
            or _sha256_bytes(restored) != item["round_trip_sha256"]
            or original != restored
        ):
            raise PBFreezeError("C2 exact fixture round trip failed")
    return _sha256_bytes(raw), claimed


def load_parent_replay_verification_v4(
    path: Path, *, pa: PAInspectionV4
) -> tuple[str, str]:
    """Load only the factory-shaped V4 parent replay PASS receipt."""

    raw, receipt = _load_canonical(path, "V4 parent replay verification")
    field_names = {field.name for field in fields(ParentReplayVerificationV4)}
    receipt = _exact_mapping(
        receipt,
        {*field_names, "receipt_sha256"},
        "V4 parent replay verification",
    )
    body = {name: receipt[name] for name in field_names}
    try:
        typed = ParentReplayVerificationV4(**body)
    except (TypeError, ValueError) as error:
        raise PBFreezeError("V4 parent replay is not a factory PASS") from error
    claimed = _require_sha256(
        receipt.get("receipt_sha256"), "V4 parent replay receipt identity"
    )
    if claimed != typed.receipt_sha256:
        raise PBFreezeError("V4 parent replay receipt identity drifted")
    expected_evidence = execution_authority_v4_bound_sha256(
        PARENT_EVIDENCE_SCHEMA_V4,
        {
            "first_child_receipt_sha256": typed.first_child_receipt_sha256,
            "input_identity_sha256": typed.input_identity_sha256,
            "second_child_receipt_sha256": typed.second_child_receipt_sha256,
            "worker_compatibility_sha256": typed.worker_compatibility_sha256,
        },
    )
    if typed.evidence_sha256 != expected_evidence:
        raise PBFreezeError("V4 parent replay evidence identity drifted")
    try:
        first = assert_no_symlink_ancestors(Path(typed.first_output_root)).resolve(
            strict=True
        )
        second = assert_no_symlink_ancestors(Path(typed.second_output_root)).resolve(
            strict=True
        )
    except OSError as error:
        raise PBFreezeError("V4 replay outputs are no longer present") from error
    if not first.is_dir() or not second.is_dir() or first == second:
        raise PBFreezeError("V4 replay output roots drifted")
    if pa.root not in {first, second}:
        raise PBFreezeError("P-A tree is not one of the verified replay outputs")
    verified_children = []
    inspections: list[PAInspectionV4] = []
    for root, expected_child_sha in (
        (first, typed.first_child_receipt_sha256),
        (second, typed.second_child_receipt_sha256),
    ):
        try:
            _, child_receipt = _load_canonical(
                root / replay_v3.CHILD_RECEIPT_FILENAME,
                "V4 child replay receipt",
            )
            guard_sha = _require_sha256(
                child_receipt.get("network_guard_sha256"),
                "child network guard",
            )
            run_id = child_receipt.get("run_id")
            process_id = child_receipt.get("process_id")
            if not isinstance(run_id, str) or type(process_id) is not int:
                raise PBFreezeError("V4 child run identity drifted")
            child = replay_v3._validate_child_receipt(
                output_root=root,
                expected_run_id=run_id,
                actual_process_id=process_id,
                expected_input_identity_sha256=typed.input_identity_sha256,
                expected_worker_compatibility_sha256=(
                    typed.worker_compatibility_sha256
                ),
                expected_network_guard_sha256=guard_sha,
                stdout=b"",
                stderr=b"",
            )
            inspection = inspect_pa_v4(root)
        except (OSError, StrictJsonError, replay_v3.ParentReplayError) as error:
            raise PBFreezeError(
                "V4 child replay evidence failed independent revalidation"
            ) from error
        if (
            child.child_receipt_sha256 != expected_child_sha
            or child.content_metadata.get("content_identity_sha256")
            != inspection.content_identity_sha256
            or child.content_metadata.get("d1_ready_manifest_sha256")
            != inspection.d1_ready_manifest_physical_sha256
        ):
            raise PBFreezeError("V4 child replay/content binding drifted")
        verified_children.append(child)
        inspections.append(inspection)
    left, right = verified_children
    left_inspection, right_inspection = inspections
    if (
        left.actual_process_id == right.actual_process_id
        or left.output_file_rows != right.output_file_rows
        or left.output_file_projection_sha256
        != right.output_file_projection_sha256
        or left.content_projection_sha256 != right.content_projection_sha256
        or not left.dedup_evidence_complete
        or not right.dedup_evidence_complete
        or left.dedup_projection_sha256 is None
        or left.dedup_projection_sha256 != right.dedup_projection_sha256
        or left_inspection.content_identity_sha256
        != right_inspection.content_identity_sha256
        or left_inspection.d1_ready_identity_sha256
        != right_inspection.d1_ready_identity_sha256
        or left_inspection.full_shard_manifest_identity_sha256
        != right_inspection.full_shard_manifest_identity_sha256
        or left_inspection.screen_submanifest_identity_sha256
        != right_inspection.screen_submanifest_identity_sha256
        or left_inspection.d6_physical_evidence_identity_sha256
        != right_inspection.d6_physical_evidence_identity_sha256
        or left_inspection.d6_physical_evidence_physical_sha256
        != right_inspection.d6_physical_evidence_physical_sha256
        or left_inspection.release_manifest_section_identity_sha256
        != right_inspection.release_manifest_section_identity_sha256
    ):
        raise PBFreezeError("V4 child replay D1/D2 equivalence drifted")
    return _sha256_bytes(raw), claimed


def _ordered_count_pairs(value: object, name: str) -> tuple[tuple[str, int], ...]:
    if (
        not isinstance(value, list)
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or type(row[1]) is not int
            or row[1] < 0
            for row in value
        )
    ):
        raise PBFreezeError(f"{name} rows drifted")
    if tuple(row[0] for row in value) != GTOK_STRATA:
        raise PBFreezeError(f"{name} rows drifted")
    return tuple((str(row[0]), int(row[1])) for row in value)


def _reread_language_decisions(pa: PAInspectionV4) -> tuple[str, int, int]:
    """Recount D4 from the physical ledger, never from the D4 summary."""

    path = pa.root / "artifacts" / "language-decisions.jsonl"
    assert_no_symlink_ancestors(path)
    invocations = 0
    rejections = 0
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                if not line.endswith(b"\n"):
                    raise PBFreezeError("D4 language ledger lacks LF framing")
                try:
                    row = json.loads(
                        line[:-1].decode("utf-8", errors="strict"),
                        object_pairs_hook=_json_no_duplicates,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            PBFreezeError(
                                f"non-finite language decision: {value}"
                            )
                        ),
                    )
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise PBFreezeError(
                        "D4 language ledger is not strict UTF-8 JSONL"
                    ) from error
                if line != canonical_json_bytes(row) + b"\n":
                    raise PBFreezeError("D4 language ledger is not canonical JSONL")
                item = _exact_mapping(
                    row,
                    {
                        "binding_sha256",
                        "document_id",
                        "keep",
                        "label",
                        "probability",
                        "receipt_sha256",
                        "scoring_input_sha256",
                        "source",
                    },
                    "D4 language decision",
                )
                source = item.get("source")
                if source not in SOURCE_TO_STRATUM or SOURCE_TO_STRATUM[str(source)] != "general":
                    raise PBFreezeError("D4 language ledger escaped the general stratum")
                try:
                    decision = LanguageIdDecisionV3(
                        binding_sha256=str(item["binding_sha256"]),
                        document_id=str(item["document_id"]),
                        label=str(item["label"]),
                        probability=item["probability"],
                        scoring_input_sha256=str(item["scoring_input_sha256"]),
                        stratum="general",
                    )
                except (TypeError, ValueError) as error:
                    raise PBFreezeError(
                        "D4 language decision failed its registered binding"
                    ) from error
                if (
                    decision.binding_sha256 != A2_LANGUAGE_ID_BINDING.receipt_sha256
                    or item.get("keep") is not decision.keep
                    or item.get("receipt_sha256") != decision.receipt_sha256
                ):
                    raise PBFreezeError("D4 language decision receipt drifted")
                invocations += 1
                if not decision.keep:
                    rejections += 1
    except OSError as error:
        raise PBFreezeError("D4 language ledger is absent or unreadable") from error
    return digest.hexdigest(), invocations, rejections


def _recompute_physical_d6_evidence(
    pa: PAInspectionV4,
) -> tuple[str, Mapping[str, Any]]:
    """Require the published V4 evidence to equal a fresh physical reread."""

    path = pa.root.joinpath(
        *PurePosixPath(pa.d6_physical_evidence_relative_path).parts
    )
    try:
        raw, published = _load_canonical(path, "physical D6 evidence")
        with tempfile.TemporaryDirectory(prefix="weft1-pb-d6-") as directory:
            recomputed, physical = recompute_physical_d6_evidence_v4(
                root=pa.root,
                sqlite_path=Path(directory) / "physical-d6.sqlite",
            )
    except (OSError, CorpusMaterializationV4Error) as error:
        raise PBFreezeError("D6 physical evidence recomputation failed") from error
    observed_physical = _sha256_bytes(raw)
    if (
        observed_physical != pa.d6_physical_evidence_physical_sha256
        or physical != observed_physical
        or canonical_json_bytes(published) != canonical_json_bytes(recomputed)
        or published.get("schema") != D6_PHYSICAL_EVIDENCE_SCHEMA_V4
        or published.get("evidence_identity_sha256")
        != pa.d6_physical_evidence_identity_sha256
    ):
        raise PBFreezeError("D6 physical evidence differs from fresh recomputation")
    return observed_physical, published


def _validate_d4_d5_d6(
    pa: PAInspectionV4, *, independent_scan: Mapping[str, object]
) -> dict[str, str]:
    _, d3 = _load_canonical(pa.root / "diagnostics" / "d3.json", "D3 diagnostic")
    source_rows = independent_scan.get("sources")
    if not isinstance(source_rows, tuple):
        raise PBFreezeError("independent source composition is absent")
    source_bytes = {
        str(row["name"]): int(row["raw_byte_count"]) for row in source_rows
    }
    expected_pools = {
        "arxiv": source_bytes["arxiv"],
        "dolma_web": source_bytes["dolma_web"],
        "fineweb_edu": source_bytes["fineweb_edu"],
        "finemath_3plus": source_bytes["finemath_3plus"],
        "olmocr": source_bytes["olmocr"],
        "science_technical_combined": (
            source_bytes["arxiv"] + source_bytes["olmocr"]
        ),
        "stackedu": source_bytes["stackedu"],
        "wikipedia_wikibooks": source_bytes["wikipedia_wikibooks"],
    }
    pool_rows = d3.get("full_pool_rows")
    if not isinstance(pool_rows, list):
        raise PBFreezeError("D3 pool composition is absent")
    observed_pools: dict[str, int] = {}
    for raw_pool in pool_rows:
        pool = _exact_mapping(
            raw_pool,
            {
                "deficit_fraction",
                "observed_bytes",
                "pool",
                "target_bytes",
            },
            "D3 pool row",
        )
        name = pool.get("pool")
        observed = pool.get("observed_bytes")
        target = pool.get("target_bytes")
        fraction = _exact_mapping(
            pool.get("deficit_fraction"),
            {"denominator", "numerator"},
            "D3 deficit fraction",
        )
        if (
            name not in expected_pools
            or name in observed_pools
            or type(observed) is not int
            or type(target) is not int
            or target < 1
            or observed < 0
            or observed > target
            or type(fraction.get("numerator")) is not int
            or type(fraction.get("denominator")) is not int
            or fraction["denominator"] < 1
            or Fraction(fraction["numerator"], fraction["denominator"])
            != Fraction(target - observed, target)
            or Fraction(target - observed, target) > Fraction(1, 100)
            or observed != expected_pools[str(name)]
        ):
            raise PBFreezeError("D3 pool composition failed independent replay")
        observed_pools[str(name)] = int(observed)
    required_pool_names = {
        "wikipedia_wikibooks",
        "dolma_web",
        "fineweb_edu",
        "stackedu",
        "finemath_3plus",
        "science_technical_combined",
    }
    if set(observed_pools) != required_pool_names:
        raise PBFreezeError("D3 pool coverage is incomplete or noncanonical")
    required_general = {
        "wikipedia_wikibooks": Fraction(22, 100),
        "dolma_web": Fraction(39, 100),
        "fineweb_edu": Fraction(39, 100),
    }
    general_total = sum(source_bytes[name] for name in required_general)
    if general_total < 1 or any(
        abs(Fraction(source_bytes[name], general_total) - target_share)
        > Fraction(2, 100)
        for name, target_share in required_general.items()
    ):
        raise PBFreezeError("D3 general 22/39/39 split is outside 2 percent")

    _, d4 = _load_canonical(pa.root / "diagnostics" / "d4.json", "D4 diagnostic")
    d4 = _exact_mapping(
        d4,
        {"gate", "invocation_counts", "rejection_counts", "status"},
        "D4 diagnostic",
    )
    invocations = dict(
        _ordered_count_pairs(d4.get("invocation_counts"), "D4 invocation")
    )
    rejections = dict(
        _ordered_count_pairs(d4.get("rejection_counts"), "D4 rejection")
    )
    language_ledger_sha256, physical_invocations, physical_rejections = (
        _reread_language_decisions(pa)
    )
    if (
        d4.get("gate") != "D4"
        or d4.get("status") != "CHECK_PASS_NO_GATE_MINT"
        or invocations["general"] < 1
        or rejections["general"] > invocations["general"]
        or invocations["general"] != physical_invocations
        or rejections["general"] != physical_rejections
        or any(
            invocations[stratum] != 0 or rejections[stratum] != 0
            for stratum in GTOK_STRATA
            if stratum != "general"
        )
    ):
        raise PBFreezeError("D4 language-ID scope did not independently pass")

    _, d5 = _load_canonical(pa.root / "diagnostics" / "d5.json", "D5 diagnostic")
    d5 = _exact_mapping(d5, {"cases", "gate", "status"}, "D5 diagnostic")
    cases = d5.get("cases")
    if (
        d5.get("gate") != "D5"
        or d5.get("status") != "CHECK_PASS_NO_GATE_MINT"
        or not isinstance(cases, list)
        or len(cases) != len(pa.screen_shard_rows)
    ):
        raise PBFreezeError("D5 diagnostic coverage drifted")
    for case, shard in zip(cases, pa.screen_shard_rows, strict=True):
        case = _exact_mapping(
            case,
            {
                "logical_jsonl_sha256",
                "record_count",
                "relative_path",
                "retained_text_bytes",
                "zstd_sha256",
            },
            "D5 shard case",
        )
        if any(
            case.get(name) != shard[name]
            for name in (
                "logical_jsonl_sha256",
                "record_count",
                "relative_path",
                "retained_text_bytes",
                "zstd_sha256",
            )
        ):
            raise PBFreezeError("D5 diagnostic differs from physical screen shards")

    _, d6 = _load_canonical(pa.root / "diagnostics" / "d6.json", "D6 diagnostic")
    d6 = _exact_mapping(
        d6,
        {
            "cluster_overlap_count",
            "consumer_bindings",
            "consumer_order_receipts",
            "document_overlap_count",
            "full_corpus_repeated_raw_content_id_count",
            "gate",
            "near_cluster_receipt",
            "screen_repeated_raw_content_id_count",
            "split_rows",
            "status",
            "stream_identities",
            "tokenizer_fit_contract",
        },
        "D6 diagnostic",
    )
    physical_d6_sha256, physical_d6 = _recompute_physical_d6_evidence(pa)
    physical_stream_rows = physical_d6.get("stream_identities")
    if not isinstance(physical_stream_rows, list):
        raise PBFreezeError("physical D6 stream identities are absent")
    physical_stream_projection = tuple(
        {
            name: row[name]
            for name in (
                "document_count",
                "framed_retained_text_sha256",
                "retained_text_bytes",
                "stream",
            )
        }
        for row in physical_stream_rows
    )
    if (
        d6.get("gate") != "D6"
        or d6.get("status") != "CHECK_PASS_NO_GATE_MINT"
        or any(
            d6.get(name) != 0
            for name in (
                "cluster_overlap_count",
                "document_overlap_count",
                "full_corpus_repeated_raw_content_id_count",
                "screen_repeated_raw_content_id_count",
            )
        )
        or not isinstance(d6.get("stream_identities"), list)
        or tuple(d6["stream_identities"])
        != tuple(independent_scan.get("screen_streams", ()))
        or tuple(d6["stream_identities"]) != physical_stream_projection
        or physical_d6.get("document_overlap_count") != 0
        or physical_d6.get("repeated_raw_content_id_count") != 0
    ):
        raise PBFreezeError("D6 split/stream identities did not independently pass")

    split_rows = d6.get("split_rows")
    if (
        not isinstance(split_rows, list)
        or tuple(row.get("stratum") for row in split_rows) != GTOK_STRATA
    ):
        raise PBFreezeError("D6 split rows are incomplete or noncanonical")
    group_by_key = {
        (str(group["stream"]), str(group["stratum"])): group
        for group in pa.screen_groups
    }
    physical_groups = physical_d6.get("split_groups")
    if (
        not isinstance(physical_groups, list)
        or tuple((row.get("stream"), row.get("stratum")) for row in physical_groups)
        != tuple(
            (stream, stratum)
            for stream in ("T", "H")
            for stratum in GTOK_STRATA
        )
    ):
        raise PBFreezeError("physical D6 split groups are incomplete")
    for physical_group in physical_groups:
        governed_group = group_by_key[
            (str(physical_group["stream"]), str(physical_group["stratum"]))
        ]
        if any(
            physical_group.get(name) != governed_group[name]
            for name in (
                "document_count",
                "ordered_raw_content_ids_sha256",
                "retained_text_bytes",
                "stratum",
                "stream",
            )
        ):
            raise PBFreezeError("physical D6 groups differ from screen membership")
    stratum_shares = {
        "general": Fraction(45, 100),
        "code": Fraction(25, 100),
        "mathematics": Fraction(15, 100),
        "science_technical": Fraction(15, 100),
    }
    for stream in ("T", "H"):
        stream_total = sum(
            int(group_by_key[(stream, stratum)]["retained_text_bytes"])
            for stratum in GTOK_STRATA
        )
        if stream_total < 1 or any(
            abs(
                Fraction(
                    int(group_by_key[(stream, stratum)]["retained_text_bytes"]),
                    stream_total,
                )
                - stratum_shares[stratum]
            )
            > Fraction(1, 100)
            for stratum in GTOK_STRATA
        ):
            raise PBFreezeError("D6 stream stratification is outside 1 percent")
    for row in split_rows:
        row = _exact_mapping(row, {"heldout", "stratum", "training"}, "D6 split row")
        stratum = str(row["stratum"])
        for field, stream in (("training", "T"), ("heldout", "H")):
            split = _exact_mapping(
                row[field],
                {
                    "deficit_bytes",
                    "document_count",
                    "realized_bytes",
                    "target_bytes",
                },
                "D6 split accounting",
            )
            group = group_by_key[(stream, stratum)]
            governed_target = dict(
                GTOK_SCREEN_TRAIN_STRATUM_TARGETS
                if stream == "T"
                else GTOK_SCREEN_HELDOUT_STRATUM_TARGETS
            )[stratum]
            if (
                any(type(split.get(name)) is not int for name in split)
                or split["document_count"] != group["document_count"]
                or split["realized_bytes"] != group["retained_text_bytes"]
                or split["target_bytes"] != governed_target
                or split["target_bytes"] - split["realized_bytes"]
                != split["deficit_bytes"]
                or split["deficit_bytes"] < 0
                or split["target_bytes"] < 1
                or Fraction(split["deficit_bytes"], split["target_bytes"])
                > FIRST_FIT_TOLERANCE
            ):
                raise PBFreezeError("D6 split accounting differs from screen membership")

    near = _exact_mapping(
        d6.get("near_cluster_receipt"),
        {
            "algorithm_identity_sha256",
            "cluster_count",
            "document_count",
            "qualifying_edge_count",
            "semantics",
        },
        "D6 near-cluster receipt",
    )
    full_count = sum(int(row["document_count"]) for row in pa.full_source_summaries)
    if (
        near.get("semantics")
        != "REGISTERED_LSH_CANDIDATE_GRAPH_EXACT_JACCARD_COMPONENTS"
        or near.get("document_count") != full_count
        or type(near.get("cluster_count")) is not int
        or not 1 <= near["cluster_count"] <= full_count
        or type(near.get("qualifying_edge_count")) is not int
        or near["qualifying_edge_count"] < 0
    ):
        raise PBFreezeError("D6 near-cluster coverage drifted")
    _require_sha256(near.get("algorithm_identity_sha256"), "D6 near-cluster algorithm")

    # V3's consumer identities use an internal stable document ID that is not
    # present in physical JSONL.  They are historical diagnostics only.  The
    # gate authority is the independently recomputed V4 raw-content-ID object.
    order_rows = physical_d6.get("consumer_order_receipts")
    if not isinstance(order_rows, list) or len(order_rows) != GTOK_SEED_COUNT:
        raise PBFreezeError("physical D6 consumer order receipt count drifted")
    orders: dict[int, Mapping[str, Any]] = {}
    for raw_order in order_rows:
        order = _exact_mapping(
            raw_order,
            {
                "document_count",
                "document_multiset_sha256",
                "framed_payload_sha256",
                "order_key_domain",
                "ordered_raw_content_ids_sha256",
                "receipt_sha256",
                "retained_text_bytes",
                "schema",
                "training_seed",
            },
            "physical D6 consumer order receipt",
        )
        body = dict(order)
        claimed = _require_sha256(
            body.pop("receipt_sha256", None), "physical D6 order receipt identity"
        )
        if (
            order.get("schema") != CONSUMER_ORDER_SCHEMA_V4
            or order.get("order_key_domain")
            != "WEFT-1/gtok-training-order/raw-content-id/v4"
            or claimed
            != execution_authority_v4_bound_sha256(
                CONSUMER_ORDER_SCHEMA_V4, body
            )
            or type(order.get("training_seed")) is not int
            or int(order["training_seed"]) in orders
        ):
            raise PBFreezeError("physical D6 consumer order receipt drifted")
        orders[int(order["training_seed"])] = order
    training_stream = next(
        row for row in physical_stream_rows if row["stream"] == "T"
    )
    if (
        len({row["document_multiset_sha256"] for row in orders.values()}) != 1
        or len({row["ordered_raw_content_ids_sha256"] for row in orders.values()})
        != len(orders)
        or any(
            row["document_count"] != training_stream["document_count"]
            or row["retained_text_bytes"] != training_stream["retained_text_bytes"]
            for row in orders.values()
        )
    ):
        raise PBFreezeError("physical D6 consumer order pairing drifted")

    bindings = physical_d6.get("consumer_bindings")
    expected_pairs = {
        (vocabulary_size, seed)
        for vocabulary_size in GTOK_VOCABULARY_ARMS
        for seed in orders
    }
    if not isinstance(bindings, list) or len(bindings) != len(expected_pairs):
        raise PBFreezeError("physical D6 consumer binding count drifted")
    seen_pairs: set[tuple[int, int]] = set()
    heldout_sha = next(
        row["framed_retained_text_sha256"]
        for row in physical_stream_rows
        if row["stream"] == "H"
    )
    for raw_binding in bindings:
        binding = _exact_mapping(
            raw_binding,
            {
                "heldout_framed_retained_text_sha256",
                "training_document_multiset_sha256",
                "training_order_receipt_sha256",
                "training_ordered_raw_content_ids_sha256",
                "training_seed",
                "vocabulary_size",
            },
            "physical D6 consumer binding",
        )
        pair = (int(binding["vocabulary_size"]), int(binding["training_seed"]))
        order = orders.get(pair[1])
        if (
            pair not in expected_pairs
            or pair in seen_pairs
            or order is None
            or binding["heldout_framed_retained_text_sha256"] != heldout_sha
            or binding["training_document_multiset_sha256"]
            != order["document_multiset_sha256"]
            or binding["training_order_receipt_sha256"] != order["receipt_sha256"]
            or binding["training_ordered_raw_content_ids_sha256"]
            != order["ordered_raw_content_ids_sha256"]
        ):
            raise PBFreezeError("physical D6 consumer binding drifted")
        seen_pairs.add(pair)
    if seen_pairs != expected_pairs:
        raise PBFreezeError("physical D6 consumer binding matrix is incomplete")

    fit = _exact_mapping(
        d6.get("tokenizer_fit_contract"),
        {
            "allowed_stream",
            "fit_input_receipt_sha256",
            "fit_text_stream_sha256",
            "heldout_admissible",
            "input_order",
        },
        "D6 tokenizer fit contract",
    )
    if (
        fit.get("allowed_stream") != "T_ONLY"
        or fit.get("heldout_admissible") is not False
        or fit.get("input_order") != "CANONICAL_T_SHARD_MANIFEST_ORDER"
    ):
        raise PBFreezeError("D6 tokenizer fit boundary drifted")
    physical_fit = _exact_mapping(
        physical_d6.get("tokenizer_fit_input"),
        {
            "allowed_stream",
            "document_count",
            "fit_text_stream_sha256",
            "heldout_admissible",
            "ordered_raw_content_ids_sha256",
            "ordered_shard_paths",
            "ordering",
            "receipt_sha256",
            "retained_text_bytes",
            "schema",
        },
        "physical D6 tokenizer fit-input",
    )
    fit_body = dict(physical_fit)
    fit_receipt = _require_sha256(
        fit_body.pop("receipt_sha256", None),
        "physical D6 tokenizer fit receipt",
    )
    if (
        physical_fit.get("schema") != TOKENIZER_FIT_INPUT_SCHEMA_V4
        or physical_fit.get("allowed_stream") != "T"
        or physical_fit.get("heldout_admissible") is not False
        or physical_fit.get("ordering")
        != "PHYSICAL_SHARD_MANIFEST_THEN_JSONL_RECORD_ORDER"
        or fit_receipt
        != execution_authority_v4_bound_sha256(
            TOKENIZER_FIT_INPUT_SCHEMA_V4, fit_body
        )
        or physical_fit.get("document_count") != training_stream["document_count"]
        or physical_fit.get("retained_text_bytes")
        != training_stream["retained_text_bytes"]
        or physical_fit.get("fit_text_stream_sha256")
        != training_stream["framed_retained_text_sha256"]
        or tuple(physical_fit.get("ordered_shard_paths", ()))
        != tuple(
            row["relative_path"]
            for row in pa.screen_shard_rows
            if row["stream"] == "T"
        )
        or _require_sha256(fit.get("fit_text_stream_sha256"), "D6 fit text stream")
        != physical_fit.get("fit_text_stream_sha256")
    ):
        raise PBFreezeError("D6 tokenizer fit boundary differs from physical V4 T")
    return {
        pa.d6_physical_evidence_relative_path: physical_d6_sha256,
        "artifacts/language-decisions.jsonl": language_ledger_sha256,
    }


def _expected_gate_evidence(
    pa: PAInspectionV4,
    *,
    c2_evidence_sha256: str,
    parent_replay_receipt_sha256: str,
) -> dict[str, tuple[str, ...]]:
    diagnostics = dict(pa.diagnostic_sha256s)
    return {
        "D1": tuple(
            sorted(
                (
                    pa.d1_ready_manifest_physical_sha256,
                    parent_replay_receipt_sha256,
                )
            )
        ),
        "D2": tuple(
            sorted(
                (pa.d2_evidence_descriptor_sha256, parent_replay_receipt_sha256)
            )
        ),
        "D3": tuple(
            sorted((diagnostics["D3"], pa.full_shard_manifest_physical_sha256))
        ),
        "D4": (diagnostics["D4"],),
        "D5": tuple(
            sorted(
                (
                    diagnostics["D5"],
                    c2_evidence_sha256,
                    pa.screen_shard_manifest_physical_sha256,
                )
            )
        ),
        "D6": tuple(
            sorted(
                (
                    diagnostics["D6"],
                    pa.d6_physical_evidence_physical_sha256,
                    pa.screen_submanifest_physical_sha256,
                )
            )
        ),
    }


def load_d1_d6_gate_bundle(
    path: Path,
    *,
    pa: PAInspectionV4,
    c2_evidence_sha256: str,
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    raw, bundle = _load_canonical(path, "D1-D6 gate bundle")
    bundle = _exact_mapping(
        bundle,
        {
            "authority_chain",
            "bundle_identity_sha256",
            "corpus_content_identity_sha256",
            "d1_ready_identity_sha256",
            "gates",
            "parent_replay_receipt_identity_sha256",
            "parent_replay_receipt_sha256",
            "release_manifest_section_identity_sha256",
            "schema",
        },
        "D1-D6 gate bundle",
    )
    if (
        bundle.get("schema") != PB_GATE_BUNDLE_SCHEMA_V4
        or tuple(bundle.get("authority_chain", ()))
        != GTOK_EXECUTION_AUTHORITY_CHAIN_V4
        or bundle.get("corpus_content_identity_sha256")
        != pa.content_identity_sha256
        or bundle.get("d1_ready_identity_sha256")
        != pa.d1_ready_identity_sha256
        or bundle.get("release_manifest_section_identity_sha256")
        != pa.release_manifest_section_identity_sha256
    ):
        raise PBFreezeError("D1-D6 gate bundle does not bind this completed P-A tree")
    payload = dict(bundle)
    claimed_bundle = _require_sha256(
        payload.pop("bundle_identity_sha256", None), "gate bundle identity"
    )
    if claimed_bundle != execution_authority_v4_bound_sha256(
        PB_GATE_BUNDLE_SCHEMA_V4, payload
    ):
        raise PBFreezeError("D1-D6 gate bundle identity drifted")
    gates = bundle.get("gates")
    if not isinstance(gates, list) or tuple(row.get("gate") for row in gates) != _GATES:
        raise PBFreezeError("D1-D6 gate bundle is incomplete or noncanonical")
    parent_physical = _require_sha256(
        bundle.get("parent_replay_receipt_sha256"),
        "parent replay receipt physical identity",
    )
    _require_sha256(
        bundle.get("parent_replay_receipt_identity_sha256"),
        "parent replay receipt typed identity",
    )
    expected_evidence = _expected_gate_evidence(
        pa,
        c2_evidence_sha256=c2_evidence_sha256,
        parent_replay_receipt_sha256=parent_physical,
    )
    receipts: list[tuple[str, str]] = []
    for gate, raw_receipt in zip(_GATES, gates, strict=True):
        receipt = _exact_mapping(
            raw_receipt,
            {
                "authoritative",
                "corpus_content_identity_sha256",
                "d1_ready_identity_sha256",
                "evidence_artifact_sha256s",
                "gate",
                "receipt_sha256",
                "release_manifest_section_identity_sha256",
                "status",
                "verifier_kind",
            },
            f"{gate} gate receipt",
        )
        if (
            receipt.get("gate") != gate
            or receipt.get("status") != "PASS"
            or receipt.get("authoritative") is not True
            or receipt.get("verifier_kind") != _GATE_VERIFIERS[gate]
            or receipt.get("corpus_content_identity_sha256")
            != pa.content_identity_sha256
            or receipt.get("d1_ready_identity_sha256")
            != pa.d1_ready_identity_sha256
            or receipt.get("release_manifest_section_identity_sha256")
            != pa.release_manifest_section_identity_sha256
            or tuple(receipt.get("evidence_artifact_sha256s", ()))
            != expected_evidence[gate]
        ):
            raise PBFreezeError(f"{gate} is not authoritative for this P-A tree")
        body = dict(receipt)
        claimed_receipt = _require_sha256(
            body.pop("receipt_sha256", None), f"{gate} gate receipt identity"
        )
        if claimed_receipt != execution_authority_v4_bound_sha256(
            f"weft1_corpus_{gate.lower()}_gate_receipt_v4", body
        ):
            raise PBFreezeError(f"{gate} gate receipt identity drifted")
        receipts.append((gate, claimed_receipt))
    return _sha256_bytes(raw), claimed_bundle, tuple(receipts)


def _rehash_gate_mint_inputs(
    *,
    pa: PAInspectionV4,
    parent_replay_receipt_path: Path,
    parent_replay_receipt_sha256: str,
    c2_evidence_path: Path,
    c2_evidence_sha256: str,
    physical_artifact_sha256s: Mapping[str, str],
) -> None:
    fixed = (
        (pa.root / "content-manifest.json", pa.content_manifest_physical_sha256),
        (pa.root / "d1-ready-manifest.json", pa.d1_ready_manifest_physical_sha256),
        (
            pa.root.joinpath(
                *PurePosixPath(pa.full_shard_manifest_relative_path).parts
            ),
            pa.full_shard_manifest_physical_sha256,
        ),
        (
            pa.root.joinpath(
                *PurePosixPath(pa.screen_submanifest_relative_path).parts
            ),
            pa.screen_submanifest_physical_sha256,
        ),
        (
            pa.root / "artifacts" / "shard-manifest.json",
            pa.screen_shard_manifest_physical_sha256,
        ),
        (
            pa.root / "artifacts" / "d2-evidence-descriptor.json",
            pa.d2_evidence_descriptor_sha256,
        ),
        (parent_replay_receipt_path, parent_replay_receipt_sha256),
        (c2_evidence_path, c2_evidence_sha256),
        *tuple(
            (
                pa.root.joinpath(*PurePosixPath(relative).parts),
                digest,
            )
            for relative, digest in sorted(physical_artifact_sha256s.items())
        ),
        *tuple(
            (
                pa.root / "diagnostics" / f"{gate.lower()}.json",
                digest,
            )
            for gate, digest in pa.diagnostic_sha256s
        ),
    )
    if any(_sha256_file(path) != expected for path, expected in fixed):
        raise PBFreezeError("a D1-D6 mint input changed before the mint boundary")
    for row in (*pa.full_shard_rows, *pa.screen_shard_rows):
        path = pa.root.joinpath(*PurePosixPath(str(row["relative_path"])).parts)
        if _sha256_file(path) != row["zstd_sha256"]:
            raise PBFreezeError("a corpus shard changed before D1-D6 minting")


def build_d1_d6_gate_bundle(
    *,
    materialization_root: Path,
    parent_replay_receipt_path: Path,
    c2_evidence_path: Path,
) -> dict[str, object]:
    """Build PASS fields from evidence only; callers cannot supply a verdict."""

    pa = inspect_pa_v4(materialization_root)
    c2_physical, unused_c2_identity = load_c2_fixture_evidence(c2_evidence_path)
    parent_physical, parent_identity = load_parent_replay_verification_v4(
        parent_replay_receipt_path, pa=pa
    )
    independent_scan = _scan_full_shards(pa)
    physical_artifact_sha256s = _validate_d4_d5_d6(
        pa, independent_scan=independent_scan
    )
    _rehash_gate_mint_inputs(
        pa=pa,
        parent_replay_receipt_path=parent_replay_receipt_path,
        parent_replay_receipt_sha256=parent_physical,
        c2_evidence_path=c2_evidence_path,
        c2_evidence_sha256=c2_physical,
        physical_artifact_sha256s=physical_artifact_sha256s,
    )
    expected_evidence = _expected_gate_evidence(
        pa,
        c2_evidence_sha256=c2_physical,
        parent_replay_receipt_sha256=parent_physical,
    )
    gates: list[dict[str, object]] = []
    for gate in _GATES:
        receipt: dict[str, object] = {
            "authoritative": True,
            "corpus_content_identity_sha256": pa.content_identity_sha256,
            "d1_ready_identity_sha256": pa.d1_ready_identity_sha256,
            "evidence_artifact_sha256s": expected_evidence[gate],
            "gate": gate,
            "release_manifest_section_identity_sha256": (
                pa.release_manifest_section_identity_sha256
            ),
            "status": "PASS",
            "verifier_kind": _GATE_VERIFIERS[gate],
        }
        receipt["receipt_sha256"] = execution_authority_v4_bound_sha256(
            f"weft1_corpus_{gate.lower()}_gate_receipt_v4", receipt
        )
        gates.append(receipt)
    bundle: dict[str, object] = {
        "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
        "corpus_content_identity_sha256": pa.content_identity_sha256,
        "d1_ready_identity_sha256": pa.d1_ready_identity_sha256,
        "gates": gates,
        "parent_replay_receipt_identity_sha256": parent_identity,
        "parent_replay_receipt_sha256": parent_physical,
        "release_manifest_section_identity_sha256": (
            pa.release_manifest_section_identity_sha256
        ),
        "schema": PB_GATE_BUNDLE_SCHEMA_V4,
    }
    bundle["bundle_identity_sha256"] = execution_authority_v4_bound_sha256(
        PB_GATE_BUNDLE_SCHEMA_V4, bundle
    )
    return bundle


def mint_d1_d6_gate_bundle(
    *,
    materialization_root: Path,
    parent_replay_receipt_path: Path,
    c2_evidence_path: Path,
    output_path: Path,
) -> tuple[dict[str, object], str]:
    if output_path.exists() or output_path.is_symlink():
        raise PBFreezeError("output gate-bundle path already exists")
    bundle = build_d1_d6_gate_bundle(
        materialization_root=materialization_root,
        parent_replay_receipt_path=parent_replay_receipt_path,
        c2_evidence_path=c2_evidence_path,
    )
    physical = _exclusive_atomic_json(output_path, bundle)
    return bundle, physical


def load_hermetic_decon_receipt(
    path: Path, *, pa: PAInspectionV4
) -> tuple[str, str, str]:
    raw, receipt = _load_canonical(path, "hermetic DECON receipt")
    receipt = _exact_mapping(
        receipt,
        {
            "algorithm_profiles",
            "authority_chain",
            "battery_scope",
            "corpus_content_identity_sha256",
            "corpus_manifest_sha256",
            "exact_match_count",
            "full_shard_manifest_identity_sha256",
            "full_shard_manifest_sha256",
            "hermetic",
            "hit_action",
            "input_commitments",
            "near_match_count",
            "network_accessed",
            "plaintext_exported",
            "receipt_sha256",
            "registered_battery_count",
            "release_manifest_section_identity_sha256",
            "runtime_commitments",
            "salt_exported",
            "schema",
            "screen_code_commitments",
            "screen_code_set_commitment_sha256",
            "screened_full_shard_count",
            "screened_full_shards",
            "screened_full_shard_set_commitment_sha256",
            "screened_battery_count",
            "screened_battery_set_commitment_sha256",
            "screened_document_count",
            "screen_submanifest_identity_sha256",
            "screen_submanifest_sha256",
            "sealed_battery_registry_commitment_sha256",
            "sealed_identifiers_exported",
            "status",
            "total_match_count",
        },
        "hermetic DECON receipt",
    )
    if (
        receipt.get("schema") != PB_DECON_SCHEMA_V5
        or tuple(receipt.get("authority_chain", ())) != PB_AUTHORITY_CHAIN_V5
        or receipt.get("corpus_content_identity_sha256")
        != pa.content_identity_sha256
        or receipt.get("corpus_manifest_sha256")
        != pa.content_manifest_physical_sha256
        or receipt.get("full_shard_manifest_sha256")
        != pa.full_shard_manifest_physical_sha256
        or receipt.get("full_shard_manifest_identity_sha256")
        != pa.full_shard_manifest_identity_sha256
        or receipt.get("screen_submanifest_sha256")
        != pa.screen_submanifest_physical_sha256
        or receipt.get("screen_submanifest_identity_sha256")
        != pa.screen_submanifest_identity_sha256
        or receipt.get("release_manifest_section_identity_sha256")
        != pa.release_manifest_section_identity_sha256
        or tuple(receipt.get("battery_scope", ())) != DECON_REQUIRED_BATTERIES
        or receipt.get("algorithm_profiles") != decon_algorithm_profiles()
        or receipt.get("screened_full_shard_count") != len(pa.full_shard_rows)
        or receipt.get("screened_document_count")
        != sum(int(row["record_count"]) for row in pa.full_shard_rows)
        or receipt.get("screened_full_shards") != list(_full_shard_projection(pa))
        or receipt.get("screened_full_shard_set_commitment_sha256")
        != _full_shard_set_commitment(pa)
    ):
        raise PBFreezeError("DECON receipt does not bind this completed P-A tree")

    inputs = _exact_mapping(
        receipt.get("input_commitments"),
        {
            "confirm_complete_ledger_sha256",
            "confirm_private_rows_sha256",
            "confirm_seal_file_set_sha256",
            "confirm_seal_ledger_sha256",
            "confirm_source_manifest_sha256",
            "eval_e_anonymous_index_sha256",
            "eval_e_lock_sha256",
            "private_input_set_commitment_sha256",
        },
        "DECON private input commitments",
    )
    for name, value in inputs.items():
        _require_sha256(value, f"DECON {name}")
    if (
        inputs.get("confirm_complete_ledger_sha256")
        != GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256
        or inputs.get("confirm_private_rows_sha256")
        != GOVERNED_CONFIRM_SOURCE_ROWS_SHA256
        or inputs.get("confirm_seal_file_set_sha256")
        != GOVERNED_CONFIRM_SEAL_SET_SHA256
        or inputs.get("confirm_source_manifest_sha256")
        != GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256
        or inputs.get("eval_e_anonymous_index_sha256")
        != GOVERNED_EVAL_E_ANONYMOUS_INDEX_SHA256
        or inputs.get("eval_e_lock_sha256") != GOVERNED_EVAL_E_LOCK_SHA256
    ):
        raise PBFreezeError("DECON governed input identity drifted")
    input_core = {
        name: inputs[name]
        for name in sorted(inputs)
        if name != "private_input_set_commitment_sha256"
    }
    if inputs.get("private_input_set_commitment_sha256") != hashlib.sha256(
        canonical_json_bytes(input_core)
    ).hexdigest():
        raise PBFreezeError("DECON private input set commitment drifted")

    runtime = _exact_mapping(
        receipt.get("runtime_commitments"),
        {
            "global_execution_provenance_sha256",
            "network_guard_sha256",
            "python_executable_sha256",
            "runtime_build_receipt_sha256",
            "unshare_executable_sha256",
        },
        "DECON runtime commitments",
    )
    for name, value in runtime.items():
        _require_sha256(value, f"DECON {name}")
    try:
        governed_runtime = {
            "global_execution_provenance_sha256": _sha256_file(
                pa.root / replay_v3.GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3
            ),
            "runtime_build_receipt_sha256": _sha256_file(
                pa.root / replay_v3.RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1
            ),
        }
    except OSError as error:
        raise PBFreezeError("P-A governed runtime artifact is absent") from error
    if any(runtime.get(name) != digest for name, digest in governed_runtime.items()):
        raise PBFreezeError("DECON runtime differs from the P-A governed runtime")

    code_rows = receipt.get("screen_code_commitments")
    if not isinstance(code_rows, list) or not code_rows:
        raise PBFreezeError("DECON screen code commitment list is empty")
    normalized_code_rows: list[dict[str, str]] = []
    for raw_code in code_rows:
        code = _exact_mapping(
            raw_code, {"relative_path", "sha256"}, "DECON screen code row"
        )
        relative = _canonical_relative_path(
            code.get("relative_path"), "DECON screen code path"
        )
        digest = _require_sha256(code.get("sha256"), "DECON screen code SHA-256")
        normalized_code_rows.append({"relative_path": relative, "sha256": digest})
    if normalized_code_rows != sorted(
        normalized_code_rows, key=lambda row: row["relative_path"]
    ) or len({row["relative_path"] for row in normalized_code_rows}) != len(
        normalized_code_rows
    ):
        raise PBFreezeError("DECON screen code commitments are noncanonical")
    if tuple(row["relative_path"] for row in normalized_code_rows) != (
        _DECON_CODE_RELATIVE_PATHS
    ):
        raise PBFreezeError("DECON screen code inventory drifted")
    repository_root = Path(__file__).resolve().parents[1]
    try:
        for row in normalized_code_rows:
            code_path = repository_root.joinpath(
                *PurePosixPath(row["relative_path"]).parts
            )
            assert_no_symlink_ancestors(code_path)
            if (
                not code_path.is_file()
                or code_path.is_symlink()
                or _sha256_file(code_path) != row["sha256"]
            ):
                raise PBFreezeError("DECON screen code bytes drifted")
    except OSError as error:
        raise PBFreezeError("DECON screen code file is absent") from error
    code_set = _require_sha256(
        receipt.get("screen_code_set_commitment_sha256"),
        "DECON screen code set commitment",
    )
    if code_set != hashlib.sha256(
        canonical_json_bytes(normalized_code_rows)
    ).hexdigest():
        raise PBFreezeError("DECON screen code set commitment drifted")

    battery_set = _require_sha256(
        receipt.get("screened_battery_set_commitment_sha256"),
        "DECON battery set commitment",
    )
    registry = _require_sha256(
        receipt.get("sealed_battery_registry_commitment_sha256"),
        "DECON sealed registry commitment",
    )
    _require_sha256(
        receipt.get("screened_full_shard_set_commitment_sha256"),
        "DECON full-shard set commitment",
    )
    expected_battery_set = hashlib.sha256(
        canonical_json_bytes(DECON_REQUIRED_BATTERIES)
    ).hexdigest()
    expected_registry = hashlib.sha256(
        canonical_json_bytes(
            {
                "algorithm_profiles": decon_algorithm_profiles(),
                "input_commitments": dict(inputs),
                "registered_battery_count": len(DECON_REQUIRED_BATTERIES),
            }
        )
    ).hexdigest()
    if (
        receipt.get("hermetic") is not True
        or receipt.get("network_accessed") is not False
        or receipt.get("plaintext_exported") is not False
        or receipt.get("sealed_identifiers_exported") is not False
        or receipt.get("salt_exported") is not False
        or receipt.get("hit_action") != "HARD_STOP_NO_MINT"
    ):
        raise PBFreezeError("DECON hermetic or salted-hash posture drifted")
    registered = receipt.get("registered_battery_count")
    screened = receipt.get("screened_battery_count")
    if (
        type(registered) is not int
        or registered != len(DECON_REQUIRED_BATTERIES)
        or screened != registered
        or battery_set != expected_battery_set
        or registry != expected_registry
    ):
        raise PBFreezeError("DECON did not cover every sealed battery")
    counts = (
        receipt.get("exact_match_count"),
        receipt.get("near_match_count"),
        receipt.get("total_match_count"),
    )
    if any(type(value) is not int or value < 0 for value in counts) or (
        counts[0] + counts[1] != counts[2]
    ):
        raise PBFreezeError("DECON match accounting drifted")
    body = dict(receipt)
    claimed = _require_sha256(
        body.pop("receipt_sha256", None), "DECON receipt identity"
    )
    if claimed != pb_authority_bound_sha256(PB_DECON_SCHEMA_V5, body):
        raise PBFreezeError("DECON receipt identity drifted")
    status = receipt.get("status")
    if status == "HIT":
        if counts[2] < 1:
            raise PBFreezeError("DECON HIT status has no match")
    elif status == "CLEAN":
        if counts != (0, 0, 0):
            raise PBFreezeError("DECON CLEAN status carries a match")
    else:
        raise PBFreezeError("DECON status is neither CLEAN nor HIT")
    return _sha256_bytes(raw), claimed, str(status)


def _rehash_freeze_inputs(
    *,
    pa: PAInspectionV4,
    gate_bundle_path: Path,
    gate_bundle_sha256: str,
    c2_evidence_path: Path,
    c2_evidence_sha256: str,
    decon_receipt_path: Path,
    decon_receipt_sha256: str,
) -> None:
    fixed = (
        (pa.root / "content-manifest.json", pa.content_manifest_physical_sha256),
        (pa.root / "d1-ready-manifest.json", pa.d1_ready_manifest_physical_sha256),
        (
            pa.root.joinpath(
                *PurePosixPath(pa.full_shard_manifest_relative_path).parts
            ),
            pa.full_shard_manifest_physical_sha256,
        ),
        (
            pa.root.joinpath(
                *PurePosixPath(pa.screen_submanifest_relative_path).parts
            ),
            pa.screen_submanifest_physical_sha256,
        ),
        (
            pa.root / "artifacts" / "shard-manifest.json",
            pa.screen_shard_manifest_physical_sha256,
        ),
        (
            pa.root.joinpath(
                *PurePosixPath(pa.d6_physical_evidence_relative_path).parts
            ),
            pa.d6_physical_evidence_physical_sha256,
        ),
        (gate_bundle_path, gate_bundle_sha256),
        (c2_evidence_path, c2_evidence_sha256),
        (decon_receipt_path, decon_receipt_sha256),
    )
    if any(_sha256_file(path) != expected for path, expected in fixed):
        raise PBFreezeError("a P-B input changed before the mint boundary")
    for row in pa.full_shard_rows:
        path = pa.root.joinpath(*PurePosixPath(str(row["relative_path"])).parts)
        if _sha256_file(path) != row["zstd_sha256"]:
            raise PBFreezeError("a full shard changed before the mint boundary")
    for row in pa.screen_shard_rows:
        path = pa.root.joinpath(*PurePosixPath(str(row["relative_path"])).parts)
        if _sha256_file(path) != row["zstd_sha256"]:
            raise PBFreezeError("a screen shard changed before the mint boundary")


def build_freeze_receipt(
    *,
    materialization_root: Path,
    gate_bundle_path: Path,
    c2_evidence_path: Path,
    decon_receipt_path: Path,
) -> dict[str, object]:
    pa = inspect_pa_v4(materialization_root)
    c2_physical, c2_identity = load_c2_fixture_evidence(c2_evidence_path)
    gate_physical, gate_identity, gate_receipts = load_d1_d6_gate_bundle(
        gate_bundle_path,
        pa=pa,
        c2_evidence_sha256=c2_physical,
    )
    c1_c3 = _scan_full_shards(pa)
    decon_physical, decon_identity, decon_status = load_hermetic_decon_receipt(
        decon_receipt_path, pa=pa
    )
    if decon_status == "HIT":
        raise DecontaminationHit("DECON hit: hard stop; no freeze receipt may be minted")
    _rehash_freeze_inputs(
        pa=pa,
        gate_bundle_path=gate_bundle_path,
        gate_bundle_sha256=gate_physical,
        c2_evidence_path=c2_evidence_path,
        c2_evidence_sha256=c2_physical,
        decon_receipt_path=decon_receipt_path,
        decon_receipt_sha256=decon_physical,
    )
    receipt: dict[str, object] = {
        "authority_chain": PB_AUTHORITY_CHAIN_V5,
        "authoritative_d1_d6_receipts": gate_receipts,
        "c1_c3_independent_reread": c1_c3,
        "c2_fixture_evidence_identity_sha256": c2_identity,
        "c2_fixture_evidence_sha256": c2_physical,
        "corpus_content_identity_sha256": pa.content_identity_sha256,
        "corpus_manifest_sha256": pa.content_manifest_physical_sha256,
        "d1_d6_gate_bundle_identity_sha256": gate_identity,
        "d1_d6_gate_bundle_sha256": gate_physical,
        "d1_ready_identity_sha256": pa.d1_ready_identity_sha256,
        "d6_physical_evidence_identity_sha256": (
            pa.d6_physical_evidence_identity_sha256
        ),
        "d6_physical_evidence_sha256": (
            pa.d6_physical_evidence_physical_sha256
        ),
        "decon_receipt_identity_sha256": decon_identity,
        "decon_receipt_sha256": decon_physical,
        "full_shard_manifest_identity_sha256": (
            pa.full_shard_manifest_identity_sha256
        ),
        "full_shard_manifest_sha256": pa.full_shard_manifest_physical_sha256,
        "gate_sequence": (
            ("C1", "PASS"),
            ("C2", "PASS"),
            ("C3", "PASS"),
            ("DECON", "CLEAN"),
        ),
        "raw_text_shards_publication_authorized": False,
        "release_manifest_section_identity_sha256": (
            pa.release_manifest_section_identity_sha256
        ),
        "screen_shard_manifest_sha256": pa.screen_shard_manifest_physical_sha256,
        "screen_submanifest_identity_sha256": (
            pa.screen_submanifest_identity_sha256
        ),
        "screen_submanifest_sha256": pa.screen_submanifest_physical_sha256,
        "schema": PB_FREEZE_SCHEMA_V5,
        "sealed_data_accessed_by_minter": False,
        "status": "FROZEN",
    }
    receipt["freeze_receipt_identity_sha256"] = pb_authority_bound_sha256(
        PB_FREEZE_SCHEMA_V5, receipt
    )
    return receipt


def _exclusive_atomic_json(path: Path, payload: Mapping[str, object]) -> str:
    if not isinstance(path, Path):
        raise TypeError("output path must be pathlib.Path")
    parent = assert_no_symlink_ancestors(path.parent).resolve(strict=True)
    target = parent / path.name
    if target.exists() or target.is_symlink():
        raise PBFreezeError("output receipt path already exists")
    raw = canonical_json_bytes(payload) + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, target)
    except FileExistsError as error:
        raise PBFreezeError("output receipt path already exists") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    if _sha256_file(target) != _sha256_bytes(raw):
        raise PBFreezeError("minted receipt failed its post-write hash check")
    return _sha256_bytes(raw)


def mint_freeze_receipt(
    *,
    materialization_root: Path,
    gate_bundle_path: Path,
    c2_evidence_path: Path,
    confirm_seal_paths: Sequence[Path],
    confirm_seal_ledger_path: Path,
    confirm_private_rows_path: Path,
    eval_e_index_path: Path,
    eval_e_lock_path: Path,
    decon_output_root: Path,
    decon_local_work_parent: Path,
    output_path: Path,
    python_executable: Path | None = None,
    unshare_executable: Path | None = None,
    decon_timeout_seconds: int = DECON_PARENT_WATCHDOG_SECONDS,
) -> tuple[dict[str, object], str]:
    """Launch hermetic DECON and mint from that fresh receipt in one call."""

    if output_path.exists() or output_path.is_symlink():
        raise PBFreezeError("output receipt path already exists")
    if decon_output_root.exists() or decon_output_root.is_symlink():
        raise PBFreezeError("DECON output root must be fresh before mint")
    try:
        decon_physical, decon_identity, decon_status = launch_hermetic_decon(
            materialization_root=materialization_root,
            confirm_seal_paths=confirm_seal_paths,
            confirm_seal_ledger_path=confirm_seal_ledger_path,
            confirm_private_rows_path=confirm_private_rows_path,
            eval_e_index_path=eval_e_index_path,
            eval_e_lock_path=eval_e_lock_path,
            output_root=decon_output_root,
            local_work_parent=decon_local_work_parent,
            python_executable=python_executable,
            unshare_executable=unshare_executable,
            timeout_seconds=decon_timeout_seconds,
        )
    except DeconError as error:
        raise PBFreezeError("hermetic DECON launch blocked the P-B mint") from error
    if decon_status == "HIT":
        raise DecontaminationHit(
            "DECON hit: hard stop; no freeze receipt may be minted"
        )
    if decon_status != "CLEAN":
        raise PBFreezeError("hermetic DECON returned no mintable status")
    decon_receipt_path = decon_output_root / DECON_RECEIPT_FILENAME
    receipt = build_freeze_receipt(
        materialization_root=materialization_root,
        gate_bundle_path=gate_bundle_path,
        c2_evidence_path=c2_evidence_path,
        decon_receipt_path=decon_receipt_path,
    )
    if (
        receipt.get("decon_receipt_sha256") != decon_physical
        or receipt.get("decon_receipt_identity_sha256") != decon_identity
    ):
        raise PBFreezeError("fresh DECON result changed before the mint boundary")
    physical = _exclusive_atomic_json(output_path, receipt)
    return receipt, physical


def _add_check_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--gate-bundle", type=Path, required=True)
    parser.add_argument("--c2-evidence", type=Path, required=True)
    parser.add_argument("--decon-receipt", type=Path, required=True)


def _add_mint_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--gate-bundle", type=Path, required=True)
    parser.add_argument("--c2-evidence", type=Path, required=True)
    parser.add_argument("--confirm-seal", action="append", type=Path, required=True)
    parser.add_argument("--confirm-seal-ledger", type=Path, required=True)
    parser.add_argument("--confirm-private-rows", type=Path, required=True)
    parser.add_argument("--eval-e-index", type=Path, required=True)
    parser.add_argument("--eval-e-lock", type=Path, required=True)
    parser.add_argument("--decon-output-root", type=Path, required=True)
    parser.add_argument("--decon-local-work-parent", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--unshare-executable", type=Path)
    parser.add_argument(
        "--decon-timeout-seconds",
        type=int,
        default=DECON_PARENT_WATCHDOG_SECONDS,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed WEFT-1 P-B freeze scaffold"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixtures = subparsers.add_parser(
        "c2-fixtures", help="emit independent C2 fixture evidence"
    )
    fixtures.add_argument("--output", type=Path, required=True)
    gates = subparsers.add_parser(
        "gates", help="mint D1-D6 only from replay and independent evidence"
    )
    gates.add_argument("--materialization-root", type=Path, required=True)
    gates.add_argument("--parent-replay-receipt", type=Path, required=True)
    gates.add_argument("--c2-evidence", type=Path, required=True)
    gates.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser(
        "check", help="validate an external DECON receipt read-only"
    )
    _add_check_inputs(check)
    mint = subparsers.add_parser(
        "mint", help="launch hermetic DECON and mint after every gate passes"
    )
    _add_mint_inputs(mint)
    mint.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "c2-fixtures":
            evidence = build_c2_fixture_evidence()
            physical = _exclusive_atomic_json(arguments.output, evidence)
            print(
                json.dumps(
                    {
                        "physical_sha256": physical,
                        "status": "C2_EVIDENCE_WRITTEN_NO_FREEZE_MINT",
                        "suite_identity_sha256": evidence["suite_identity_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "gates":
            bundle, physical = mint_d1_d6_gate_bundle(
                materialization_root=arguments.materialization_root,
                parent_replay_receipt_path=arguments.parent_replay_receipt,
                c2_evidence_path=arguments.c2_evidence,
                output_path=arguments.output,
            )
            print(
                json.dumps(
                    {
                        "bundle_identity_sha256": bundle[
                            "bundle_identity_sha256"
                        ],
                        "physical_sha256": physical,
                        "status": "D1_D6_AUTHORITATIVE_PASS_BUNDLE_MINTED",
                    },
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "check":
            receipt = build_freeze_receipt(
                materialization_root=arguments.materialization_root,
                gate_bundle_path=arguments.gate_bundle,
                c2_evidence_path=arguments.c2_evidence,
                decon_receipt_path=arguments.decon_receipt,
            )
            print(
                json.dumps(
                    {
                        "corpus_manifest_sha256": receipt[
                            "corpus_manifest_sha256"
                        ],
                        "prospective_freeze_receipt_identity_sha256": receipt[
                            "freeze_receipt_identity_sha256"
                        ],
                        "status": (
                            "EXTERNAL_DECON_RECEIPT_VALIDATED_READ_ONLY_NO_MINT"
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 0
        receipt, physical = mint_freeze_receipt(
            materialization_root=arguments.materialization_root,
            gate_bundle_path=arguments.gate_bundle,
            c2_evidence_path=arguments.c2_evidence,
            confirm_seal_paths=tuple(arguments.confirm_seal),
            confirm_seal_ledger_path=arguments.confirm_seal_ledger,
            confirm_private_rows_path=arguments.confirm_private_rows,
            eval_e_index_path=arguments.eval_e_index,
            eval_e_lock_path=arguments.eval_e_lock,
            decon_output_root=arguments.decon_output_root,
            decon_local_work_parent=arguments.decon_local_work_parent,
            output_path=arguments.output,
            python_executable=arguments.python_executable,
            unshare_executable=arguments.unshare_executable,
            decon_timeout_seconds=arguments.decon_timeout_seconds,
        )
        print(
            json.dumps(
                {
                    "freeze_receipt_identity_sha256": receipt[
                        "freeze_receipt_identity_sha256"
                    ],
                    "physical_sha256": physical,
                    "status": "FROZEN",
                },
                sort_keys=True,
            )
        )
        return 0
    except DecontaminationHit as error:
        print(str(error), file=os.sys.stderr)
        return 23
    except (OSError, PBFreezeError, TypeError, ValueError) as error:
        print(f"P-B blocked: {error}", file=os.sys.stderr)
        return 2


__all__ = [
    "DecontaminationHit",
    "CONSUMER_ORDER_SCHEMA_V4",
    "DECON_REQUIRED_BATTERIES",
    "D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4",
    "D6_PHYSICAL_EVIDENCE_SCHEMA_V4",
    "FULL_SHARD_MANIFEST_RELATIVE_PATH_V4",
    "FULL_SHARD_MANIFEST_SCHEMA_V4",
    "PAInspectionV4",
    "PBFreezeError",
    "PB_AUTHORITY_CHAIN_V5",
    "PB_C2_EVIDENCE_SCHEMA_V5",
    "PB_DECON_SCHEMA_V5",
    "PB_FREEZE_SCHEMA_V5",
    "PB_GATE_BUNDLE_SCHEMA_V4",
    "PARENT_REPLAY_SCHEMA_V4",
    "TOKENIZER_FIT_INPUT_SCHEMA_V4",
    "build_d1_d6_gate_bundle",
    "build_c2_fixture_evidence",
    "build_freeze_receipt",
    "inspect_pa_v4",
    "load_c2_fixture_evidence",
    "load_d1_d6_gate_bundle",
    "load_hermetic_decon_receipt",
    "load_parent_replay_verification_v4",
    "main",
    "mint_freeze_receipt",
    "mint_d1_d6_gate_bundle",
    "pb_authority_bound_sha256",
    "recompute_physical_d6_evidence_v4",
]


if __name__ == "__main__":
    raise SystemExit(main())
