"""Additive parent replay verifier for A3/V4 WEFT-1 corpus materialization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

from training import weft1_corpus_replay_a2 as replay_v3
from training.weft1_corpus_a3 import (
    A3_AUTHORITY_PATH,
    A3_AUTHORITY_SHA256,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
    execution_authority_v4_bound_sha256,
)
from training.weft1_corpus_materialize_a3 import (
    BRIDGE_RELATIVE_PATH_V4,
    D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4,
    D1_READY_IDENTITY_SCHEMA_V4,
    D1_READY_SCHEMA_V4,
    FULL_SHARD_MANIFEST_RELATIVE_PATH_V4,
    FULL_SHARD_MANIFEST_SCHEMA_V4,
    MATERIALIZATION_BRIDGE_SCHEMA_V4,
    MATERIALIZED_CONTENT_SCHEMA_V4,
    MATERIALIZER_SCHEMA_V4,
    RELEASE_MANIFEST_SECTION_SCHEMA_V4,
    SCREEN_SUBMANIFEST_RELATIVE_PATH_V4,
    SCREEN_SUBMANIFEST_SCHEMA_V4,
    V4_READINESS,
    load_cache_download_artifact_v4,
    load_source_manifest_artifact_v4,
    load_upstream_enumeration_artifact_v4,
    validate_physical_d6_evidence_v4,
)
from training.weft1_corpus_pa import (
    DEFAULT_REQUIREMENTS_LOCK_SHA256,
    attest_runtime_v3,
)
from training.weft1_corpus_parsed_asset_cache_v1 import (
    ParsedAssetRecoveryContextV1,
    parsed_asset_runtime_identity_v1,
)
from training.weft1_corpus_replay_a2 import (
    GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3,
    ParentReplayError,
    RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1,
    RUNTIME_BUILD_RECEIPT_SCHEMA_V1,
)
from training.weft1_release import (
    RELEASE_AUTHORITY_PATH,
    RELEASE_AUTHORITY_SHA256,
    release_manifest_section,
    verify_release_authority_artifact,
)
from training.weft1_gtok_a1_contract import SOURCE_FAMILIES
from training.weft1_gtok_contract import GTOK_STRATA, canonical_json_bytes
from training.weft1_strict_io import assert_no_symlink_ancestors


REPOSITORY_ROOT_V4 = Path(__file__).resolve().parents[1]
PRODUCTION_WORKER_PATH_V4 = (
    REPOSITORY_ROOT_V4 / "scripts" / "run_weft1_corpus_materialize_a3.py"
)
PRODUCTION_DEPENDENCY_LOCK_PATH_V4 = (
    REPOSITORY_ROOT_V4 / "training" / "weft1_corpus_gtok_a2_requirements.lock"
)
RUNTIME_BUILDER_PATH_V4 = (
    REPOSITORY_ROOT_V4 / "scripts" / "build_weft1_pa_runtime.py"
)
PARENT_REPLAY_SCHEMA_V4 = "weft1_corpus_parent_replay_verification_v4"
PARENT_INPUT_SCHEMA_V4 = "weft1_corpus_parent_replay_inputs_v4"
WORKER_COMPATIBILITY_SCHEMA_V4 = "weft1_corpus_parent_worker_compatibility_v4"
PARENT_EVIDENCE_SCHEMA_V4 = "weft1_corpus_parent_replay_evidence_v4"
# The observed production projection is roughly seven days per replay.  Keep a
# finite parent-side sanity limit, but do not let our own watchdog duplicate
# Colab's shorter backend lifetime and kill an otherwise healthy worker first.
V4_DEFAULT_WORKER_TIMEOUT_SECONDS = 14 * 24 * 60 * 60
V4_PARENT_LANE_OPERATION_ORDER = (
    ("cache_fill", 0),
    ("cache_fill", 1),
    ("materialize", 0),
    ("materialize", 1),
)
PARSED_ASSET_CODE_IDENTITY_SCHEMA_V1 = (
    "weft1_parsed_asset_cache_code_identity_v1"
)
PARSED_ASSET_INPUT_IDENTITY_SCHEMA_V1 = (
    "weft1_parsed_asset_cache_input_identity_v1"
)
PARSED_ASSET_CODE_LOGICAL_NAMES_V1 = frozenset(
    {
        "production_io",
        "training/weft1_corpus_a2.py",
        "training/weft1_corpus_enumeration_a2.py",
        "training/weft1_corpus_materialize_a2.py",
        "training/weft1_corpus_materialize_a3.py",
        "training/weft1_corpus_parsed_asset_cache_v1.py",
        "training/weft1_corpus_source_io_a2.py",
        "training/weft1_corpus_sources_a2.py",
        "training/weft1_gtok_a1_contract.py",
        "training/weft1_gtok_contract.py",
        "training/weft1_strict_io.py",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compatibility_files_v4() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(
        (item for item in (REPOSITORY_ROOT_V4 / "training").iterdir() if item.is_file()),
        key=lambda item: item.name.encode("utf-8"),
    ):
        files[f"training/{path.name}"] = path
    for path in sorted(
        (REPOSITORY_ROOT_V4 / "models").rglob("*.py"),
        key=lambda item: item.relative_to(REPOSITORY_ROOT_V4).as_posix().encode("utf-8"),
    ):
        files[path.relative_to(REPOSITORY_ROOT_V4).as_posix()] = path
    for path in (A3_AUTHORITY_PATH, RELEASE_AUTHORITY_PATH):
        files[path.relative_to(REPOSITORY_ROOT_V4).as_posix()] = path
    # Logical names used by the frozen runtime receipt validator are kept exact.
    files.pop("training/weft1_corpus_pa.py", None)
    files["production_io"] = REPOSITORY_ROOT_V4 / "training" / "weft1_corpus_pa.py"
    files["runtime_builder"] = RUNTIME_BUILDER_PATH_V4
    files["worker"] = PRODUCTION_WORKER_PATH_V4
    return files


@dataclass(frozen=True)
class ExpectedTransportV4:
    execution_binding_sha256: str
    effective_route_identity_sha256: str
    enumeration_receipt_sha256: str
    download_receipt_sha256: str
    verification_receipt_sha256: str
    selection_plan_sha256: str
    source_strata: tuple[tuple[str, str], ...]


def _load_expected_transport_v4(
    enumeration_path: Path,
    download_path: Path,
    manifest_path: Path,
) -> ExpectedTransportV4:
    enumeration = load_upstream_enumeration_artifact_v4(enumeration_path)
    manifest = load_source_manifest_artifact_v4(manifest_path)
    download = load_cache_download_artifact_v4(download_path)
    if (
        not enumeration.authoritative
        or download.enumeration_receipt_sha256 != enumeration.receipt_sha256
        or download.source_manifest != manifest
        or enumeration.execution_binding != manifest.execution_binding
        or download.execution_binding != manifest.execution_binding
        or download.selection_plan_sha256 != manifest.selection_plan_sha256
    ):
        raise ParentReplayError("V4 parent input receipts do not compose")
    source_strata = tuple(
        (family.route.source_family, family.route.stratum)
        for family in enumeration.families
    )
    if tuple(source for source, _stratum in source_strata) != SOURCE_FAMILIES or any(
        stratum not in GTOK_STRATA for _source, stratum in source_strata
    ):
        raise ParentReplayError("V4 parent routes do not cover canonical sources/strata")
    return ExpectedTransportV4(
        execution_binding_sha256=enumeration.execution_binding.receipt_sha256,
        effective_route_identity_sha256=(
            enumeration.execution_binding.effective_route_identity_sha256
        ),
        enumeration_receipt_sha256=enumeration.receipt_sha256,
        download_receipt_sha256=download.receipt_sha256,
        verification_receipt_sha256=download.verification_receipt_sha256,
        selection_plan_sha256=download.selection_plan_sha256,
        source_strata=source_strata,
    )


def _read_json(path: Path) -> tuple[Mapping[str, object], str]:
    value, digest = replay_v3._read_canonical_json_object(path)
    return value, digest


_FULL_SHARD_ROW_KEYS_V4 = {
    "codec_binding_sha256",
    "content_identity_sha256",
    "first_full_ordinal",
    "identity_relative_path",
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


def _validate_full_corpus_manifest_structure_v4(
    full_manifest: Mapping[str, object],
    *,
    output_rows: Mapping[str, Mapping[str, object]],
    expected_source_strata: tuple[tuple[str, str], ...],
) -> tuple[int, int]:
    """Validate source-homogeneous full shards before a parent can mint."""

    expected_manifest_keys = {
        "codec_binding_sha256",
        "document_count",
        "document_order",
        "manifest_identity_sha256",
        "ordered_raw_content_ids_sha256",
        "retained_text_bytes",
        "schema",
        "shard_order",
        "shard_target_uncompressed_jsonl_bytes",
        "shards",
        "sources",
    }
    if set(full_manifest) != expected_manifest_keys or (
        full_manifest.get("document_order")
        != "canonical_stratum_then_canonical_source_then_full_ordinal"
        or full_manifest.get("shard_order")
        != "canonical_stratum_then_canonical_source_then_shard_index"
    ):
        raise ParentReplayError("V4 full-shard manifest fields/order drifted")
    if tuple(source for source, _stratum in expected_source_strata) != SOURCE_FAMILIES:
        raise ParentReplayError("V4 expected source order is not canonical")
    expected_by_source = dict(expected_source_strata)
    summaries = full_manifest.get("sources")
    if not isinstance(summaries, list) or len(summaries) != len(SOURCE_FAMILIES):
        raise ParentReplayError("V4 full-shard source summaries drifted")
    summary_counts: dict[str, int] = {}
    summary_bytes: dict[str, int] = {}
    for expected_source, expected_stratum, summary in zip(
        (row[0] for row in expected_source_strata),
        (row[1] for row in expected_source_strata),
        summaries,
        strict=True,
    ):
        if (
            not isinstance(summary, Mapping)
            or set(summary)
            != {"document_count", "retained_text_bytes", "source", "stratum"}
            or summary.get("source") != expected_source
            or summary.get("stratum") != expected_stratum
            or type(summary.get("document_count")) is not int
            or int(summary["document_count"]) < 0
            or type(summary.get("retained_text_bytes")) is not int
            or int(summary["retained_text_bytes"]) < 0
            or (int(summary["document_count"]) == 0)
            != (int(summary["retained_text_bytes"]) == 0)
        ):
            raise ParentReplayError("V4 full-shard source summary is invalid")
        summary_counts[expected_source] = int(summary["document_count"])
        summary_bytes[expected_source] = int(summary["retained_text_bytes"])

    shard_rows = full_manifest.get("shards")
    if not isinstance(shard_rows, list) or not shard_rows:
        raise ParentReplayError("V4 full-shard manifest is empty")
    observed_counts = {source: 0 for source in SOURCE_FAMILIES}
    observed_bytes = {source: 0 for source in SOURCE_FAMILIES}
    observed_shard_counts = {source: 0 for source in SOURCE_FAMILIES}
    seen_paths: set[str] = set()
    canonical_pair_order = {
        (stratum, source): ordinal
        for ordinal, (stratum, source) in enumerate(
            (stratum, source)
            for stratum in GTOK_STRATA
            for source in SOURCE_FAMILIES
            if expected_by_source[source] == stratum
        )
    }
    previous_pair_ordinal = -1
    for shard in shard_rows:
        if not isinstance(shard, Mapping) or set(shard) != _FULL_SHARD_ROW_KEYS_V4:
            raise ParentReplayError("V4 full-shard row fields drifted")
        source = shard.get("source")
        stratum = shard.get("stratum")
        identity_relative = shard.get("identity_relative_path")
        relative = shard.get("relative_path")
        expected_index = observed_shard_counts.get(str(source), -1)
        expected_identity_relative = f"{stratum}/full-{expected_index:05d}.jsonl.zst"
        pair_ordinal = canonical_pair_order.get((str(stratum), str(source)), -1)
        if (
            type(source) is not str
            or source not in expected_by_source
            or stratum != expected_by_source[source]
            or type(identity_relative) is not str
            or type(relative) is not str
            or relative != f"full-shards/{source}/{identity_relative}"
            or identity_relative != expected_identity_relative
            or pair_ordinal < previous_pair_ordinal
            or relative in seen_paths
            or shard.get("stream") != "FULL"
            or shard.get("codec_binding_sha256")
            != full_manifest.get("codec_binding_sha256")
        ):
            raise ParentReplayError("V4 full shard is not source-homogeneous/bound")
        previous_pair_ordinal = pair_ordinal
        seen_paths.add(relative)
        physical = output_rows.get(relative)
        if (
            physical is None
            or physical.get("role") != "content"
            or physical.get("sha256") != shard.get("zstd_sha256")
            or physical.get("bytes") != shard.get("zstd_bytes")
        ):
            raise ParentReplayError("V4 full shard differs from parent inventory")
        for name in (
            "record_count",
            "retained_text_bytes",
            "logical_jsonl_bytes",
            "zstd_bytes",
        ):
            if type(shard.get(name)) is not int or int(shard[name]) < 1:
                raise ParentReplayError("V4 full-shard counts are invalid")
        first = shard.get("first_full_ordinal")
        last = shard.get("last_full_ordinal")
        if (
            type(first) is not int
            or type(last) is not int
            or int(first) < 0
            or int(last) < int(first)
        ):
            raise ParentReplayError("V4 full-shard ordinal range is invalid")
        observed_counts[source] += int(shard["record_count"])
        observed_bytes[source] += int(shard["retained_text_bytes"])
        observed_shard_counts[source] += 1
    if observed_counts != summary_counts or observed_bytes != summary_bytes:
        raise ParentReplayError("V4 full-shard source summaries do not reconcile")
    return sum(observed_counts.values()), sum(observed_bytes.values())


def _validate_v4_content_profile(
    child: object,
    *,
    expected_environment_identity_sha256: str,
    expected_global_execution_provenance: Mapping[str, object],
    expected_transport: ExpectedTransportV4,
    expected_release_section: Mapping[str, object],
) -> None:
    metadata = child.content_metadata
    expected_provenance = replay_v3._validate_global_execution_provenance_v3(
        expected_global_execution_provenance
    )
    expected_provenance_sha = hashlib.sha256(
        canonical_json_bytes(expected_provenance) + b"\n"
    ).hexdigest()
    if (
        metadata.get("environment_identity_sha256")
        != expected_environment_identity_sha256
        or metadata.get("pipeline_code_identity_sha256")
        != expected_provenance.get("pipeline_code_identity_sha256")
        or metadata.get("global_execution_provenance_identity_sha256")
        != expected_provenance.get("provenance_identity_sha256")
        or metadata.get("global_execution_provenance_sha256")
        != expected_provenance_sha
        or metadata.get("runtime_build_receipt_identity_sha256")
        != expected_provenance.get("runtime_build_receipt_identity_sha256")
        or metadata.get("runtime_build_receipt_sha256")
        != expected_provenance.get("runtime_build_receipt_sha256")
        or metadata.get("materializer_algorithm_version") != 2
    ):
        raise ParentReplayError("V4 child runtime/provenance metadata drifted")

    root = Path(child.output_root)
    rows = {str(row["path"]): row for row in child.output_file_rows}
    required = {
        "content-manifest.json",
        "d1-ready-manifest.json",
        "artifacts/shard-manifest.json",
        BRIDGE_RELATIVE_PATH_V4,
        D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4,
        FULL_SHARD_MANIFEST_RELATIVE_PATH_V4,
        SCREEN_SUBMANIFEST_RELATIVE_PATH_V4,
        GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3,
        RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1,
    }
    if not required.issubset(rows) or any(rows[path]["role"] != "content" for path in required):
        raise ParentReplayError("V4 child inventory lacks a required governed artifact")
    content, content_sha = _read_json(root / "content-manifest.json")
    d1, d1_sha = _read_json(root / "d1-ready-manifest.json")
    bridge, bridge_sha = _read_json(root / BRIDGE_RELATIVE_PATH_V4)
    full_manifest, full_manifest_sha = _read_json(
        root / FULL_SHARD_MANIFEST_RELATIVE_PATH_V4
    )
    screen_manifest, screen_manifest_sha = _read_json(
        root / SCREEN_SUBMANIFEST_RELATIVE_PATH_V4
    )
    d6_physical, d6_physical_sha = _read_json(
        root / D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
    )
    provenance, provenance_sha = _read_json(
        root / GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3
    )
    runtime, runtime_sha = _read_json(root / RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1)
    if (
        rows["content-manifest.json"]["sha256"] != content_sha
        or rows["d1-ready-manifest.json"]["sha256"] != d1_sha
        or rows[BRIDGE_RELATIVE_PATH_V4]["sha256"] != bridge_sha
        or rows[FULL_SHARD_MANIFEST_RELATIVE_PATH_V4]["sha256"]
        != full_manifest_sha
        or rows[SCREEN_SUBMANIFEST_RELATIVE_PATH_V4]["sha256"]
        != screen_manifest_sha
        or rows[D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4]["sha256"]
        != d6_physical_sha
        or rows[GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3]["sha256"] != provenance_sha
        or rows[RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1]["sha256"] != runtime_sha
        or dict(provenance) != expected_provenance
        or provenance_sha != expected_provenance_sha
        or runtime.get("schema") != RUNTIME_BUILD_RECEIPT_SCHEMA_V1
        or runtime.get("status") != "PASS"
        or runtime.get("authoritative") is not True
        or runtime_sha != expected_provenance.get("runtime_build_receipt_sha256")
    ):
        raise ParentReplayError("V4 child governed file inventory drifted")

    with tempfile.TemporaryDirectory(prefix="weft1-v4-d6-reread-") as raw_d6:
        try:
            recomputed_d6, recomputed_d6_sha = validate_physical_d6_evidence_v4(
                root=root, sqlite_path=Path(raw_d6) / "physical-d6.sqlite"
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ParentReplayError(
                "V4 physical D6 evidence failed independent recomputation"
            ) from error
    if (
        canonical_json_bytes(d6_physical) != canonical_json_bytes(recomputed_d6)
        or d6_physical_sha != recomputed_d6_sha
    ):
        raise ParentReplayError("V4 physical D6 evidence changed after recomputation")

    content_payload = dict(content)
    claimed_content_identity = content_payload.pop("content_identity_sha256", None)
    recomputed_content_identity = execution_authority_v4_bound_sha256(
        MATERIALIZED_CONTENT_SCHEMA_V4, content_payload
    )
    expected_release_identity = execution_authority_v4_bound_sha256(
        RELEASE_MANIFEST_SECTION_SCHEMA_V4, expected_release_section
    )
    release = content.get("release")
    if not isinstance(release, Mapping) or set(release) != {
        "authority_sha256",
        "manifest_section",
        "manifest_section_identity_sha256",
    }:
        raise ParentReplayError("V4 content release section shape drifted")
    if (
        claimed_content_identity != recomputed_content_identity
        or claimed_content_identity != metadata.get("content_identity_sha256")
        or content.get("schema") != MATERIALIZER_SCHEMA_V4
        or content.get("authority_chain") != list(GTOK_EXECUTION_AUTHORITY_CHAIN_V4)
        or content.get("mode") != "PRODUCTION"
        or content.get("readiness") != V4_READINESS
        or content.get("source_identity_sha256")
        != metadata.get("source_identity_sha256")
        or release.get("authority_sha256") != RELEASE_AUTHORITY_SHA256
        or release.get("manifest_section") != expected_release_section
        or release.get("manifest_section_identity_sha256") != expected_release_identity
    ):
        raise ParentReplayError("V4 content identity/authority/release binding drifted")

    bridge_payload = dict(bridge)
    claimed_bridge_identity = bridge_payload.pop("bridge_identity_sha256", None)
    if (
        claimed_bridge_identity
        != execution_authority_v4_bound_sha256(
            MATERIALIZATION_BRIDGE_SCHEMA_V4, bridge_payload
        )
        or bridge.get("schema") != MATERIALIZATION_BRIDGE_SCHEMA_V4
        or bridge.get("authority_chain") != list(GTOK_EXECUTION_AUTHORITY_CHAIN_V4)
        or bridge.get("release_authority_sha256") != RELEASE_AUTHORITY_SHA256
        or bridge.get("release_manifest_section_identity_sha256")
        != expected_release_identity
        or bridge.get("execution_binding_sha256")
        != expected_transport.execution_binding_sha256
        or bridge.get("effective_route_identity_sha256")
        != expected_transport.effective_route_identity_sha256
        or bridge.get("upstream_enumeration_receipt_sha256")
        != expected_transport.enumeration_receipt_sha256
        or bridge.get("cache_download_receipt_sha256")
        != expected_transport.download_receipt_sha256
        or bridge.get("cache_verification_receipt_sha256")
        != expected_transport.verification_receipt_sha256
    ):
        raise ParentReplayError("V4 transport bridge receipt drifted")
    bridge_binding = content.get("v4_transport_bridge")
    if not isinstance(bridge_binding, Mapping) or (
        bridge_binding.get("path") != BRIDGE_RELATIVE_PATH_V4
        or bridge_binding.get("sha256") != bridge_sha
        or bridge_binding.get("bridge_identity_sha256") != claimed_bridge_identity
    ):
        raise ParentReplayError("V4 content manifest does not bind its bridge bytes")

    full_core = dict(full_manifest)
    claimed_full_identity = full_core.pop("manifest_identity_sha256", None)
    screen_core = dict(screen_manifest)
    claimed_screen_identity = screen_core.pop("submanifest_identity_sha256", None)
    full_binding = content.get("v4_full_corpus")
    if not isinstance(full_binding, Mapping):
        raise ParentReplayError("V4 content lacks its full-corpus binding")
    full_record_count, full_retained_bytes = (
        _validate_full_corpus_manifest_structure_v4(
            full_manifest,
            output_rows=rows,
            expected_source_strata=expected_transport.source_strata,
        )
    )
    groups = screen_manifest.get("groups")
    if not isinstance(groups, list) or len(groups) != 8:
        raise ParentReplayError("V4 screen submanifest lacks the 2x4 groups")
    expected_groups = [
        (stream, stratum) for stream in ("T", "H") for stratum in GTOK_STRATA
    ]
    screen_count = 0
    for group, (stream, stratum) in zip(groups, expected_groups, strict=True):
        if (
            not isinstance(group, Mapping)
            or set(group)
            != {
                "document_count",
                "full_location_projection_sha256",
                "ordered_raw_content_ids_sha256",
                "retained_text_bytes",
                "stratum",
                "stream",
            }
            or group.get("stream") != stream
            or group.get("stratum") != stratum
            or type(group.get("document_count")) is not int
            or int(group["document_count"]) < 0
            or type(group.get("retained_text_bytes")) is not int
            or int(group["retained_text_bytes"]) < 0
        ):
            raise ParentReplayError("V4 screen submanifest group drifted")
        screen_count += int(group["document_count"])
    if (
        claimed_full_identity
        != execution_authority_v4_bound_sha256(
            FULL_SHARD_MANIFEST_SCHEMA_V4, full_core
        )
        or claimed_screen_identity
        != execution_authority_v4_bound_sha256(
            SCREEN_SUBMANIFEST_SCHEMA_V4, screen_core
        )
        or full_manifest.get("schema") != FULL_SHARD_MANIFEST_SCHEMA_V4
        or full_manifest.get("document_count") != full_record_count
        or full_manifest.get("retained_text_bytes") != full_retained_bytes
        or screen_manifest.get("schema") != SCREEN_SUBMANIFEST_SCHEMA_V4
        or screen_manifest.get("missing_full_document_count") != 0
        or screen_manifest.get("screen_document_count") != screen_count
        or screen_manifest.get("full_manifest_identity_sha256")
        != claimed_full_identity
        or screen_manifest.get("full_manifest_sha256") != full_manifest_sha
        or screen_manifest.get("screen_shard_manifest_sha256")
        != rows["artifacts/shard-manifest.json"]["sha256"]
        or screen_manifest.get("d6_physical_evidence_path")
        != D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
        or screen_manifest.get("d6_physical_evidence_sha256") != d6_physical_sha
        or screen_manifest.get("d6_physical_evidence_identity_sha256")
        != d6_physical.get("evidence_identity_sha256")
        or screen_manifest.get("non_screen_full_document_count")
        != full_record_count - screen_count
        or full_binding.get("full_manifest_identity_sha256")
        != claimed_full_identity
        or full_binding.get("full_manifest_sha256") != full_manifest_sha
        or full_binding.get("screen_submanifest_identity_sha256")
        != claimed_screen_identity
        or full_binding.get("screen_submanifest_sha256") != screen_manifest_sha
        or full_binding.get("d6_physical_evidence_path")
        != D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
        or full_binding.get("d6_physical_evidence_sha256") != d6_physical_sha
        or full_binding.get("d6_physical_evidence_identity_sha256")
        != d6_physical.get("evidence_identity_sha256")
        or bridge.get("full_manifest_identity_sha256") != claimed_full_identity
        or bridge.get("d6_physical_evidence_identity_sha256")
        != d6_physical.get("evidence_identity_sha256")
        or bridge.get("screen_submanifest_identity_sha256")
        != claimed_screen_identity
    ):
        raise ParentReplayError("V4 full corpus/screen subset binding drifted")

    expected_d1_keys = {
        "content_identity_sha256",
        "d1_ready_identity_sha256",
        "file_inventory",
        "gate_minted",
        "mode",
        "readiness",
        "schema",
        "source_identity_sha256",
    }
    if set(d1) != expected_d1_keys:
        raise ParentReplayError("V4 D1-ready fields drifted")
    d1_payload = dict(d1)
    claimed_d1_identity = d1_payload.pop("d1_ready_identity_sha256", None)
    expected_inventory = [
        {
            "bytes": row["bytes"],
            "relative_path": row["path"],
            "sha256": row["sha256"],
        }
        for row in child.output_file_rows
        if row["path"] != "d1-ready-manifest.json"
    ]
    if (
        claimed_d1_identity
        != execution_authority_v4_bound_sha256(
            D1_READY_IDENTITY_SCHEMA_V4, d1_payload
        )
        or d1.get("schema") != D1_READY_SCHEMA_V4
        or d1.get("mode") != "PRODUCTION"
        or d1.get("readiness") != V4_READINESS
        or d1.get("gate_minted") is not False
        or d1.get("content_identity_sha256") != claimed_content_identity
        or d1.get("source_identity_sha256") != metadata.get("source_identity_sha256")
        or d1.get("file_inventory") != expected_inventory
        or d1_sha != metadata.get("d1_ready_manifest_sha256")
    ):
        raise ParentReplayError("V4 D1-ready manifest does not compose with inventory")


@dataclass(frozen=True)
class ParentReplayVerificationV4:
    status: str
    authoritative: bool
    d1_file_replay_verified: bool
    d2_dedup_replay_verified: bool
    v4_content_profile_verified: bool
    release_binding_verified: bool
    runtime_provenance_verified: bool
    os_network_isolation_verified: bool
    durable_post_write_rehash_verified: bool
    input_identity_sha256: str
    worker_compatibility_sha256: str
    first_child_receipt_sha256: str
    second_child_receipt_sha256: str
    first_output_root: str
    second_output_root: str
    durable_output_parent: str
    durable_parsed_asset_cache_parent: str
    first_parsed_asset_cache_context_sha256: str
    second_parsed_asset_cache_context_sha256: str
    local_work_parent: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.status != "PASS" or not all(
            (
                self.authoritative,
                self.d1_file_replay_verified,
                self.d2_dedup_replay_verified,
                self.v4_content_profile_verified,
                self.release_binding_verified,
                self.runtime_provenance_verified,
                self.os_network_isolation_verified,
                self.durable_post_write_rehash_verified,
            )
        ):
            raise ValueError("V4 parent replay receipt may only represent full PASS")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(PARENT_REPLAY_SCHEMA_V4, self)


def verify_production_materialization_replays_v4(
    *,
    python_executable: Path,
    enumeration_receipt_path: Path,
    cache_download_receipt_path: Path,
    source_manifest_path: Path,
    cache_root: Path,
    fasttext_model_path: Path,
    runtime_build_receipt_path: Path,
    durable_mount_root: Path,
    durable_storage_marker_path: Path,
    durable_output_parent: Path,
    durable_parsed_asset_cache_parent: Path,
    local_work_parent: Path,
    first_output_root: Path,
    second_output_root: Path,
    first_run_id: str = "production-v4-replay-a",
    second_run_id: str = "production-v4-replay-b",
    timeout_seconds: float = V4_DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> ParentReplayVerificationV4:
    """Run the V4 worker twice and mint only after all production gates pass."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
        or float(timeout_seconds) > V4_DEFAULT_WORKER_TIMEOUT_SECONDS
    ):
        raise ParentReplayError(
            "V4 timeout_seconds must be finite, positive, and no more than "
            "the 14-day per-worker watchdog"
        )

    executable = replay_v3._resolve_python_executable(python_executable)
    if executable != replay_v3._resolve_python_executable(Path(os.sys.executable)):
        raise ParentReplayError("V4 production replay must use the attested interpreter")
    roots = replay_v3._resolved_fresh_roots(first_output_root, second_output_root)
    durable_parent, local_parent, storage_identity = (
        replay_v3._validate_production_storage_roots_v3(
            roots=roots,
            durable_mount_root=durable_mount_root,
            durable_storage_marker_path=durable_storage_marker_path,
            durable_output_parent=durable_output_parent,
            local_work_parent=local_work_parent,
        )
    )
    parsed_asset_cache_parent = assert_no_symlink_ancestors(
        durable_parsed_asset_cache_parent
    ).resolve(strict=True)
    if not parsed_asset_cache_parent.is_dir():
        raise ParentReplayError(
            "V4 parsed-asset cache parent must be a real directory"
        )
    if (
        parsed_asset_cache_parent == durable_parent
        or parsed_asset_cache_parent in durable_parent.parents
        or durable_parent in parsed_asset_cache_parent.parents
        or parsed_asset_cache_parent == local_parent
        or parsed_asset_cache_parent in local_parent.parents
        or local_parent in parsed_asset_cache_parent.parents
    ):
        raise ParentReplayError(
            "V4 parsed-asset cache, output, and local-work parents must be disjoint"
        )
    if replay_v3.attest_production_storage_v3(
        durable_mount_root=durable_mount_root,
        durable_storage_marker_path=durable_storage_marker_path,
        durable_output_parent=parsed_asset_cache_parent,
        local_work_parent=local_parent,
    ) != storage_identity:
        raise ParentReplayError(
            "V4 parsed-asset cache is not on the registered durable storage"
        )
    if _sha256_file(A3_AUTHORITY_PATH) != A3_AUTHORITY_SHA256:
        raise ParentReplayError("A3 authority artifact drifted")
    if verify_release_authority_artifact() != RELEASE_AUTHORITY_SHA256:
        raise ParentReplayError("release authority artifact drifted")
    if _sha256_file(PRODUCTION_DEPENDENCY_LOCK_PATH_V4) != DEFAULT_REQUIREMENTS_LOCK_SHA256:
        raise ParentReplayError("V4 dependency lock drifted")

    from training.weft1_corpus_a2 import A2_LANGUAGE_ID_BINDING

    fasttext_stat = fasttext_model_path.stat()
    if (
        fasttext_stat.st_size != A2_LANGUAGE_ID_BINDING.model_bytes
        or _sha256_file(fasttext_model_path) != A2_LANGUAGE_ID_BINDING.model_sha256
    ):
        raise ParentReplayError("FastText model differs from the A2 binding")
    cache_resolved = assert_no_symlink_ancestors(cache_root).resolve(strict=True)
    if not cache_resolved.is_dir():
        raise ParentReplayError("V4 source cache must be a real directory")
    if (
        parsed_asset_cache_parent == cache_resolved
        or parsed_asset_cache_parent in cache_resolved.parents
        or cache_resolved in parsed_asset_cache_parent.parents
    ):
        raise ParentReplayError(
            "V4 parsed-asset recovery cache must be disjoint from the source cache"
        )

    compatibility_files = _compatibility_files_v4()
    compatibility_rows = replay_v3._logical_file_rows(
        compatibility_files, name="V4 production compatibility files"
    )
    parsed_asset_code_rows = tuple(
        row
        for row in compatibility_rows
        if row["logical_name"] in PARSED_ASSET_CODE_LOGICAL_NAMES_V1
    )
    if {
        str(row["logical_name"]) for row in parsed_asset_code_rows
    } != PARSED_ASSET_CODE_LOGICAL_NAMES_V1:
        raise ParentReplayError(
            "V4 parsed-asset parser/cache code closure is incomplete"
        )
    parsed_asset_code_identity = execution_authority_v4_bound_sha256(
        PARSED_ASSET_CODE_IDENTITY_SCHEMA_V1,
        parsed_asset_code_rows,
    )
    compatibility_hashes = {
        str(row["logical_name"]): str(row["sha256"]) for row in compatibility_rows
    }
    runtime_attestation = attest_runtime_v3(
        requirements_lock=PRODUCTION_DEPENDENCY_LOCK_PATH_V4,
        executable=executable,
    )
    unshare = replay_v3._resolve_unshare_executable(replay_v3.LINUX_UNSHARE_PATH_V1)
    replay_v3._verify_unshare_network_isolation(
        unshare_executable=unshare,
        python_executable=executable,
    )
    release_section = release_manifest_section()

    with tempfile.TemporaryDirectory(
        prefix="weft1-v4-production-inputs-", dir=local_parent
    ) as raw_snapshot:
        snapshot_root = Path(raw_snapshot).resolve(strict=True)
        code_root = snapshot_root / "code"
        code_files = replay_v3._snapshot_production_code_v3(
            compatibility_files, code_root=code_root
        )
        if replay_v3._logical_file_rows(
            code_files, name="V4 production compatibility snapshots"
        ) != compatibility_rows:
            raise ParentReplayError("V4 code snapshot identity drifted")
        snapshots = {
            "cache_download": replay_v3._snapshot_governed_file_v3(
                cache_download_receipt_path,
                snapshot_root / "cache-download-receipt-v4.json",
                name="V4 cache download receipt",
            ),
            "dependency_lock": replay_v3._snapshot_governed_file_v3(
                PRODUCTION_DEPENDENCY_LOCK_PATH_V4,
                snapshot_root / "requirements.lock",
                name="V4 dependency lock",
            ),
            "enumeration": replay_v3._snapshot_governed_file_v3(
                enumeration_receipt_path,
                snapshot_root / "enumeration-receipt-v4.json",
                name="V4 enumeration receipt",
            ),
            "fasttext_model": replay_v3._snapshot_governed_file_v3(
                fasttext_model_path,
                snapshot_root / "lid.176.bin",
                name="FastText model",
            ),
            "source_manifest": replay_v3._snapshot_governed_file_v3(
                source_manifest_path,
                snapshot_root / "source-manifest-v4.json",
                name="V4 source manifest",
            ),
            "runtime_build_receipt": replay_v3._snapshot_governed_file_v3(
                runtime_build_receipt_path,
                snapshot_root / "runtime-build-receipt.json",
                name="runtime build receipt",
            ),
        }
        expected_transport = _load_expected_transport_v4(
            snapshots["enumeration"],
            snapshots["cache_download"],
            snapshots["source_manifest"],
        )
        runtime_receipt, runtime_receipt_sha = replay_v3._load_runtime_build_receipt_v1(
            snapshots["runtime_build_receipt"],
            runtime_attestation=runtime_attestation,
            expected_builder_sha256=compatibility_hashes["runtime_builder"],
            expected_runtime_contract_sha256=compatibility_hashes["production_io"],
            expected_python_executable=executable,
        )
        global_provenance = replay_v3._build_global_execution_provenance_v3(
            environment_payload=runtime_attestation.environment_payload,
            environment_identity_sha256=runtime_attestation.environment_identity_sha256,
            python_executable_sha256=runtime_attestation.executable_sha256,
            dependency_lock_sha256=runtime_attestation.dependency_lock_sha256,
            pipeline_components=replay_v3._logical_file_rows(
                code_files, name="V4 production compatibility snapshots"
            ),
            runtime_build_receipt_identity_sha256=str(
                runtime_receipt["receipt_identity_sha256"]
            ),
            runtime_build_receipt_sha256=runtime_receipt_sha,
            selected_wheels=runtime_receipt["selected_wheels"],
            production_storage_identity=storage_identity,
        )
        snapshots["execution_provenance"] = snapshot_root / "global-execution-provenance.json"
        replay_v3._write_fresh_canonical_json_v3(
            snapshots["execution_provenance"], global_provenance
        )
        input_files = {
            "cache_download_receipt": snapshots["cache_download"],
            "dependency_lock": snapshots["dependency_lock"],
            "enumeration_receipt": snapshots["enumeration"],
            "fasttext_model": snapshots["fasttext_model"],
            "global_execution_provenance": snapshots["execution_provenance"],
            "runtime_build_receipt": snapshots["runtime_build_receipt"],
            "source_manifest": snapshots["source_manifest"],
        }
        parsed_asset_input_rows = replay_v3._logical_file_rows(
            {
                "cache_download_receipt": snapshots["cache_download"],
                "enumeration_receipt": snapshots["enumeration"],
                "source_manifest": snapshots["source_manifest"],
            },
            name="V4 parsed-asset recovery inputs",
        )
        parsed_asset_input_identity = execution_authority_v4_bound_sha256(
            PARSED_ASSET_INPUT_IDENTITY_SCHEMA_V1,
            parsed_asset_input_rows,
        )
        input_rows = replay_v3._logical_file_rows(input_files, name="V4 replay inputs")
        input_identity = execution_authority_v4_bound_sha256(
            PARENT_INPUT_SCHEMA_V4, input_rows
        )
        durable_marker_physical_sha256 = str(
            storage_identity["durable_marker_sha256"]
        )
        parsed_asset_contexts = tuple(
            ParsedAssetRecoveryContextV1(
                run_id=run_id,
                durable_marker_physical_sha256=(
                    durable_marker_physical_sha256
                ),
                runtime_identity_sha256=parsed_asset_runtime_identity_v1(
                    runtime_attestation.environment_payload
                ),
                code_identity_sha256=parsed_asset_code_identity,
                input_identity_sha256=parsed_asset_input_identity,
            )
            for run_id in (first_run_id, second_run_id)
        )
        parsed_asset_cache_roots = []
        for context in parsed_asset_contexts:
            lane_parent = parsed_asset_cache_parent / context.run_id
            assert_no_symlink_ancestors(lane_parent)
            lane_parent.mkdir(exist_ok=True)
            cache_lane_root = lane_parent / context.identity_sha256
            assert_no_symlink_ancestors(cache_lane_root)
            cache_lane_root.mkdir(exist_ok=True)
            parsed_asset_cache_roots.append(
                cache_lane_root.resolve(strict=True)
            )
        arguments = (
            str(code_files["worker"]),
            "--enumeration-receipt",
            str(snapshots["enumeration"]),
            "--cache-download-receipt",
            str(snapshots["cache_download"]),
            "--source-manifest",
            str(snapshots["source_manifest"]),
            "--cache-root",
            str(cache_resolved),
            "--fasttext-model",
            str(snapshots["fasttext_model"]),
            "--breakdown-root",
            str(code_root),
            "--execution-provenance",
            str(snapshots["execution_provenance"]),
            "--runtime-build-receipt",
            str(snapshots["runtime_build_receipt"]),
        )
        cache_fill_arguments = (*arguments, "--cache-fill-only")
        worker_compatibility = execution_authority_v4_bound_sha256(
            WORKER_COMPATIBILITY_SCHEMA_V4,
            {
                "cache_fill_arguments": cache_fill_arguments,
                "materialization_arguments": arguments,
                "compatibility_files": replay_v3._logical_file_rows(
                    code_files, name="V4 production compatibility snapshots"
                ),
                "network_isolation_executable_sha256": _sha256_file(unshare),
                "python_executable_sha256": _sha256_file(executable),
            },
        )
        guard_sha = hashlib.sha256(replay_v3._NETWORK_GUARD_SOURCE).hexdigest()
        children = []
        with tempfile.TemporaryDirectory(
            prefix="weft1-v4-network-guard-", dir=local_parent
        ) as raw_guard:
            guard_root = Path(raw_guard).resolve(strict=True)
            guard_path = guard_root / "sitecustomize.py"
            guard_path.write_bytes(replay_v3._NETWORK_GUARD_SOURCE)
            if _sha256_file(guard_path) != guard_sha:
                raise ParentReplayError("V4 network guard byte check failed")
            lanes = tuple(zip(
                (first_run_id, second_run_id),
                roots,
                parsed_asset_cache_roots,
                parsed_asset_contexts,
                strict=True,
            ))

            def lane_environment(
                run_id: str,
                output_root: Path,
                parsed_cache_root: Path,
                parsed_context: ParsedAssetRecoveryContextV1,
            ) -> dict[str, str]:
                replay_v3._validate_exact_code_snapshot_tree_v3(code_root, code_files)
                runtime_now = attest_runtime_v3(
                    requirements_lock=snapshots["dependency_lock"],
                    executable=executable,
                )
                if runtime_now.environment_identity_sha256 != runtime_attestation.environment_identity_sha256:
                    raise ParentReplayError("V4 runtime changed before worker launch")
                if replay_v3.attest_production_storage_v3(
                    durable_mount_root=durable_mount_root,
                    durable_storage_marker_path=durable_storage_marker_path,
                    durable_output_parent=durable_parent,
                    local_work_parent=local_parent,
                ) != storage_identity:
                    raise ParentReplayError("V4 durable storage changed before launch")
                return replay_v3._offline_environment(
                    guard_directory=guard_root,
                    guard_sha256=guard_sha,
                    run_id=run_id,
                    output_root=output_root,
                    local_work_parent=local_parent,
                    input_identity_sha256=input_identity,
                    worker_compatibility_sha256=worker_compatibility,
                    worker_import_root=code_root,
                    extra_environment=None,
                    parsed_asset_cache_root=parsed_cache_root,
                    parsed_asset_code_identity_sha256=(
                        parsed_context.code_identity_sha256
                    ),
                    parsed_asset_durable_marker_sha256=(
                        parsed_context.durable_marker_physical_sha256
                    ),
                    parsed_asset_input_identity_sha256=(
                        parsed_context.input_identity_sha256
                    ),
                )

            # Complete both independent parser lanes before either expensive
            # deterministic materialization.  On a replacement Colab backend,
            # committed receipts are scanned in O(asset count) and only the
            # active missing asset is reparsed.
            for operation, lane_index in V4_PARENT_LANE_OPERATION_ORDER:
                if operation != "cache_fill":
                    continue
                (
                    run_id,
                    output_root,
                    parsed_cache_root,
                    parsed_context,
                ) = lanes[lane_index]
                environment = lane_environment(
                    run_id,
                    output_root,
                    parsed_cache_root,
                    parsed_context,
                )
                replay_v3._run_worker(
                    command=(
                        str(unshare),
                        "--net",
                        "--",
                        str(executable),
                        "-I",
                        "-B",
                        "-c",
                        replay_v3._ISOLATED_WORKER_BOOTSTRAP_SOURCE_V3,
                        str(guard_path),
                        str(code_root),
                        *cache_fill_arguments,
                    ),
                    cwd=code_root,
                    environment=environment,
                    timeout_seconds=float(timeout_seconds),
                )
                if output_root.exists():
                    raise ParentReplayError(
                        "V4 cache-fill worker mutated a replay output root"
                    )

            for operation, lane_index in V4_PARENT_LANE_OPERATION_ORDER:
                if operation != "materialize":
                    continue
                (
                    run_id,
                    output_root,
                    parsed_cache_root,
                    parsed_context,
                ) = lanes[lane_index]
                environment = lane_environment(
                    run_id,
                    output_root,
                    parsed_cache_root,
                    parsed_context,
                )
                pid, stdout, stderr = replay_v3._run_worker(
                    command=(
                        str(unshare),
                        "--net",
                        "--",
                        str(executable),
                        "-I",
                        "-B",
                        "-c",
                        replay_v3._ISOLATED_WORKER_BOOTSTRAP_SOURCE_V3,
                        str(guard_path),
                        str(code_root),
                        *arguments,
                    ),
                    cwd=code_root,
                    environment=environment,
                    timeout_seconds=float(timeout_seconds),
                )
                child = replay_v3._validate_child_receipt(
                    output_root=output_root,
                    expected_run_id=run_id,
                    actual_process_id=pid,
                    expected_input_identity_sha256=input_identity,
                    expected_worker_compatibility_sha256=worker_compatibility,
                    expected_network_guard_sha256=guard_sha,
                    stdout=stdout,
                    stderr=stderr,
                )
                _validate_v4_content_profile(
                    child,
                    expected_environment_identity_sha256=(
                        runtime_attestation.environment_identity_sha256
                    ),
                    expected_global_execution_provenance=global_provenance,
                    expected_transport=expected_transport,
                    expected_release_section=release_section,
                )
                children.append(child)

        first, second = children
        if (
            first.actual_process_id == second.actual_process_id
            or first.output_file_rows != second.output_file_rows
            or first.output_file_projection_sha256 != second.output_file_projection_sha256
            or first.content_projection_sha256 != second.content_projection_sha256
            or not first.dedup_evidence_complete
            or not second.dedup_evidence_complete
            or first.dedup_projection_sha256 != second.dedup_projection_sha256
            or first.dedup_projection_sha256 is None
        ):
            raise ParentReplayError("V4 D1/D2 replay equivalence failed")
        if input_rows != replay_v3._logical_file_rows(input_files, name="V4 replay inputs"):
            raise ParentReplayError("V4 replay inputs changed during execution")
        if compatibility_rows != replay_v3._logical_file_rows(
            compatibility_files, name="V4 production compatibility files"
        ):
            raise ParentReplayError("V4 compatibility files changed during execution")
        replay_v3._validate_exact_code_snapshot_tree_v3(code_root, code_files)
        if attest_runtime_v3(
            requirements_lock=snapshots["dependency_lock"], executable=executable
        ).environment_identity_sha256 != runtime_attestation.environment_identity_sha256:
            raise ParentReplayError("V4 runtime changed before parent minting")
        if replay_v3.attest_production_storage_v3(
            durable_mount_root=durable_mount_root,
            durable_storage_marker_path=durable_storage_marker_path,
            durable_output_parent=durable_parent,
            local_work_parent=local_parent,
        ) != storage_identity:
            raise ParentReplayError("V4 durable storage changed before parent minting")
        if replay_v3.attest_production_storage_v3(
            durable_mount_root=durable_mount_root,
            durable_storage_marker_path=durable_storage_marker_path,
            durable_output_parent=parsed_asset_cache_parent,
            local_work_parent=local_parent,
        ) != storage_identity:
            raise ParentReplayError(
                "V4 parsed-asset cache storage changed before parent minting"
            )
        for child in children:
            final_rows = replay_v3._validate_file_inventory(
                output_root=Path(child.output_root),
                claimed_files=list(child.output_file_rows),
            )
            if final_rows != child.output_file_rows:
                raise ParentReplayError("V4 output changed before parent minting")
            receipt_path = Path(child.output_root) / replay_v3.CHILD_RECEIPT_FILENAME
            if _sha256_file(receipt_path) != child.child_receipt_sha256:
                raise ParentReplayError("V4 child receipt changed before parent minting")
        evidence_payload = {
            "first_child_receipt_sha256": first.child_receipt_sha256,
            "first_parsed_asset_cache_context_sha256": (
                parsed_asset_contexts[0].identity_sha256
            ),
            "input_identity_sha256": input_identity,
            "second_child_receipt_sha256": second.child_receipt_sha256,
            "second_parsed_asset_cache_context_sha256": (
                parsed_asset_contexts[1].identity_sha256
            ),
            "worker_compatibility_sha256": worker_compatibility,
        }
        evidence_sha = execution_authority_v4_bound_sha256(
            PARENT_EVIDENCE_SCHEMA_V4, evidence_payload
        )
        return ParentReplayVerificationV4(
            status="PASS",
            authoritative=True,
            d1_file_replay_verified=True,
            d2_dedup_replay_verified=True,
            v4_content_profile_verified=True,
            release_binding_verified=True,
            runtime_provenance_verified=True,
            os_network_isolation_verified=True,
            durable_post_write_rehash_verified=True,
            input_identity_sha256=input_identity,
            worker_compatibility_sha256=worker_compatibility,
            first_child_receipt_sha256=first.child_receipt_sha256,
            second_child_receipt_sha256=second.child_receipt_sha256,
            first_output_root=first.output_root,
            second_output_root=second.output_root,
            durable_output_parent=str(durable_parent),
            durable_parsed_asset_cache_parent=str(
                parsed_asset_cache_parent
            ),
            first_parsed_asset_cache_context_sha256=(
                parsed_asset_contexts[0].identity_sha256
            ),
            second_parsed_asset_cache_context_sha256=(
                parsed_asset_contexts[1].identity_sha256
            ),
            local_work_parent=str(local_parent),
            evidence_sha256=evidence_sha,
        )


__all__ = [
    "ParentReplayVerificationV4",
    "V4_DEFAULT_WORKER_TIMEOUT_SECONDS",
    "V4_PARENT_LANE_OPERATION_ORDER",
    "verify_production_materialization_replays_v4",
]
