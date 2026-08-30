from __future__ import annotations

import base64
from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import training.weft1_corpus_pb as pb
from training import weft1_corpus_pa as production_io
from training.weft1_corpus_a2 import StableDocumentV3
from training.weft1_corpus_a3 import (
    GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
    execution_authority_v4_bound_sha256,
)
from training.weft1_corpus_materialize_a3 import (
    FULL_SHARD_MANIFEST_RELATIVE_PATH_V4,
    FULL_SHARD_MANIFEST_SCHEMA_V4,
    SCREEN_SUBMANIFEST_RELATIVE_PATH_V4,
    SCREEN_SUBMANIFEST_SCHEMA_V4,
    MATERIALIZED_CONTENT_SCHEMA_V4,
    MATERIALIZER_SCHEMA_V4,
    RELEASE_MANIFEST_SECTION_SCHEMA_V4,
    V4_READINESS,
)
from training.weft1_gtok_a1_contract import SOURCE_FAMILIES
from training.weft1_gtok_contract import GTOK_STRATA, canonical_json_bytes


SOURCE_STRATA = {
    "dolma_web": "general",
    "wikipedia_wikibooks": "general",
    "stackedu": "code",
    "finemath_3plus": "mathematics",
    "arxiv": "science_technical",
    "olmocr": "science_technical",
    "fineweb_edu": "general",
}
SOURCE_TEXT = {
    "dolma_web": "general-dolma",
    "wikipedia_wikibooks": "general-wiki",
    "stackedu": "code-π",
    "finemath_3plus": "math-α",
    "arxiv": "science-arxiv",
    "olmocr": "science-olmocr",
    "fineweb_edu": "general-fineweb",
}


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _document(source: str) -> StableDocumentV3:
    return StableDocumentV3(
        source=source,
        stratum=SOURCE_STRATA[source],
        stable_source_record_id=hashlib.sha256(
            f"fixture:{source}".encode("utf-8")
        ).hexdigest(),
        text=SOURCE_TEXT[source],
    )


def _write_full_identity(
    output_root: Path, document: StableDocumentV3
) -> object:
    shard = production_io._open_shard(
        output_root,
        stream="FULL",
        stratum=document.stratum,
        index=0,
    )
    record = production_io.canonical_jsonl_record_bytes_v3(document)
    shard.zstd_handle.write(record)
    shard.logical_sha256.update(record)
    shard.logical_bytes += len(record)
    shard.retained_text_bytes += document.retained_byte_count
    shard.record_count += 1
    return production_io._close_shard(shard)


def _shard_row(
    identity: object,
    *,
    actual_prefix: str,
    source: str | None,
    stream: str,
    stratum: str,
    first_full_ordinal: int | None = None,
) -> dict[str, object]:
    row = {
        **asdict(identity),
        "content_identity_sha256": identity.content_identity_sha256,
        "identity_relative_path": identity.relative_path,
        "relative_path": f"{actual_prefix}/{identity.relative_path}",
        "stream": stream,
        "stratum": stratum,
    }
    if source is not None:
        row["source"] = source
    if first_full_ordinal is not None:
        row["first_full_ordinal"] = first_full_ordinal
        row["last_full_ordinal"] = first_full_ordinal
    return row


def _build_scan_fixture(tmp_path: Path) -> tuple[pb.PAInspectionV4, dict[str, int]]:
    root = tmp_path / "pa"
    root.mkdir()
    provenance_path = root / pb.replay_v3.GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3
    runtime_path = root / pb.replay_v3.RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_bytes(b"fixture-provenance\n")
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_bytes(b"fixture-runtime\n")
    documents = {source: _document(source) for source in SOURCE_FAMILIES}
    full_rows: list[dict[str, object]] = []
    full_locations: dict[str, tuple[str, int, str]] = {}
    ordered_ids = hashlib.sha256()
    ordinal = 0
    for stratum in GTOK_STRATA:
        for source in SOURCE_FAMILIES:
            if SOURCE_STRATA[source] != stratum:
                continue
            document = documents[source]
            identity = _write_full_identity(
                root / "full-shards" / source, document
            )
            row = _shard_row(
                identity,
                actual_prefix=f"full-shards/{source}",
                source=source,
                stream="FULL",
                stratum=stratum,
                first_full_ordinal=ordinal,
            )
            full_rows.append(row)
            full_locations[document.shard_record_id] = (
                str(row["relative_path"]),
                0,
                source,
            )
            encoded = document.shard_record_id.encode("ascii")
            ordered_ids.update(len(encoded).to_bytes(8, "big"))
            ordered_ids.update(encoded)
            ordinal += 1

    source_summaries = tuple(
        {
            "document_count": 1,
            "retained_text_bytes": documents[source].retained_byte_count,
            "source": source,
            "stratum": SOURCE_STRATA[source],
        }
        for source in SOURCE_FAMILIES
    )
    full_core = {
        "codec_binding_sha256": pb.A2_ZSTD_CODEC_BINDING.receipt_sha256,
        "document_order": "canonical_stratum_then_canonical_source_then_full_ordinal",
        "document_count": len(documents),
        "ordered_raw_content_ids_sha256": ordered_ids.hexdigest(),
        "retained_text_bytes": sum(
            document.retained_byte_count for document in documents.values()
        ),
        "schema": FULL_SHARD_MANIFEST_SCHEMA_V4,
        "shard_target_uncompressed_jsonl_bytes": production_io.DEFAULT_SHARD_TARGET_BYTES,
        "shard_order": "canonical_stratum_then_canonical_source_then_shard_index",
        "shards": tuple(full_rows),
        "sources": source_summaries,
    }
    full_identity = execution_authority_v4_bound_sha256(
        FULL_SHARD_MANIFEST_SCHEMA_V4, full_core
    )
    full_manifest = {**full_core, "manifest_identity_sha256": full_identity}
    full_physical = _write(
        root / FULL_SHARD_MANIFEST_RELATIVE_PATH_V4, full_manifest
    )

    screen_assignments = (
        ("T", documents["dolma_web"]),
        ("T", documents["stackedu"]),
        ("H", documents["wikipedia_wikibooks"]),
    )
    screen_rows: list[dict[str, object]] = []
    for stream, document in screen_assignments:
        result = production_io.write_jsonl_zstd_shards_v3(
            (document,),
            root / "shards",
            stream=stream,
            stratum=document.stratum,
            shard_target_bytes=1024,
        )
        screen_rows.append(
            _shard_row(
                result.shards[0],
                actual_prefix="shards",
                source=None,
                stream=stream,
                stratum=document.stratum,
            )
        )
    screen_rows.sort(key=lambda row: str(row["relative_path"]))
    screen_shard_manifest = {
        "codec_binding_sha256": pb.A2_ZSTD_CODEC_BINDING.receipt_sha256,
        "schema": "weft1_corpus_shard_manifest_v3",
        "shards": tuple(screen_rows),
        "tokenizer_fit_input_receipt_sha256": "a" * 64,
    }
    screen_shard_physical = _write(
        root / "artifacts" / "shard-manifest.json", screen_shard_manifest
    )
    d6_evidence, expected_d6_physical = pb.recompute_physical_d6_evidence_v4(
        root=root,
        sqlite_path=tmp_path / "physical-d6.sqlite",
    )
    d6_physical = _write(
        root / pb.D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4,
        d6_evidence,
    )
    assert d6_physical == expected_d6_physical

    groups: list[dict[str, object]] = []
    assignment_by_group = {
        (stream, document.stratum): document
        for stream, document in screen_assignments
    }
    for stream in ("T", "H"):
        for stratum in GTOK_STRATA:
            ids = hashlib.sha256()
            locations = hashlib.sha256()
            document = assignment_by_group.get((stream, stratum))
            retained = 0
            count = 0
            if document is not None:
                encoded = document.shard_record_id.encode("ascii")
                ids.update(len(encoded).to_bytes(8, "big"))
                ids.update(encoded)
                full_path, record_ordinal, source = full_locations[
                    document.shard_record_id
                ]
                location = canonical_json_bytes(
                    {
                        "full_shard_relative_path": full_path,
                        "raw_content_id": document.shard_record_id,
                        "shard_record_ordinal": record_ordinal,
                        "source": source,
                    }
                )
                locations.update(len(location).to_bytes(8, "big"))
                locations.update(location)
                retained = document.retained_byte_count
                count = 1
            groups.append(
                {
                    "document_count": count,
                    "full_location_projection_sha256": locations.hexdigest(),
                    "ordered_raw_content_ids_sha256": ids.hexdigest(),
                    "retained_text_bytes": retained,
                    "stratum": stratum,
                    "stream": stream,
                }
            )
    screen_core = {
        "d6_physical_evidence_identity_sha256": d6_evidence[
            "evidence_identity_sha256"
        ],
        "d6_physical_evidence_path": pb.D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4,
        "d6_physical_evidence_sha256": d6_physical,
        "full_manifest_identity_sha256": full_identity,
        "full_manifest_path": FULL_SHARD_MANIFEST_RELATIVE_PATH_V4,
        "full_manifest_sha256": full_physical,
        "groups": tuple(groups),
        "missing_full_document_count": 0,
        "non_screen_full_document_count": len(documents) - len(screen_assignments),
        "schema": SCREEN_SUBMANIFEST_SCHEMA_V4,
        "screen_document_count": len(screen_assignments),
        "screen_shard_manifest_path": "artifacts/shard-manifest.json",
        "screen_shard_manifest_sha256": screen_shard_physical,
    }
    screen_identity = execution_authority_v4_bound_sha256(
        SCREEN_SUBMANIFEST_SCHEMA_V4, screen_core
    )
    screen_manifest = {**screen_core, "submanifest_identity_sha256": screen_identity}
    screen_physical = _write(
        root / SCREEN_SUBMANIFEST_RELATIVE_PATH_V4, screen_manifest
    )

    stratum_bytes = {
        stratum: sum(
            document.retained_byte_count
            for document in documents.values()
            if document.stratum == stratum
        )
        for stratum in GTOK_STRATA
    }
    d3 = {
        "full_pool_rows": [],
        "gate": "D3",
        "observed_stratum_bytes": tuple(stratum_bytes.items()),
        "pool_receipts": [],
        "status": "CHECK_PASS_NO_GATE_MINT",
    }
    d3_sha = _write(root / "diagnostics" / "d3.json", d3)
    pa = pb.PAInspectionV4(
        root=root,
        content_manifest_physical_sha256="1" * 64,
        content_identity_sha256="2" * 64,
        d1_ready_manifest_physical_sha256="3" * 64,
        d1_ready_identity_sha256="4" * 64,
        full_shard_manifest_physical_sha256=full_physical,
        full_shard_manifest_relative_path=FULL_SHARD_MANIFEST_RELATIVE_PATH_V4,
        full_shard_manifest_identity_sha256=full_identity,
        full_shard_rows=tuple(full_rows),
        full_source_summaries=source_summaries,
        screen_submanifest_physical_sha256=screen_physical,
        screen_submanifest_identity_sha256=screen_identity,
        screen_submanifest_relative_path=SCREEN_SUBMANIFEST_RELATIVE_PATH_V4,
        d6_physical_evidence_physical_sha256=d6_physical,
        d6_physical_evidence_identity_sha256=str(
            d6_evidence["evidence_identity_sha256"]
        ),
        d6_physical_evidence_relative_path=pb.D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4,
        screen_groups=tuple(groups),
        screen_shard_manifest_physical_sha256=screen_shard_physical,
        screen_shard_rows=tuple(screen_rows),
        diagnostic_sha256s=(("D3", d3_sha), ("D4", "5" * 64), ("D5", "6" * 64), ("D6", "7" * 64)),
        d2_evidence_descriptor_sha256="8" * 64,
        release_manifest_section_identity_sha256="9" * 64,
    )
    return pa, stratum_bytes


def _write_c2(path: Path) -> tuple[str, str]:
    evidence = pb.build_c2_fixture_evidence()
    physical = _write(path, evidence)
    return physical, str(evidence["suite_identity_sha256"])


def _gate_bundle(
    pa: pb.PAInspectionV4, *, c2_physical_sha256: str
) -> dict[str, object]:
    parent_physical = "e" * 64
    expected = pb._expected_gate_evidence(
        pa,
        c2_evidence_sha256=c2_physical_sha256,
        parent_replay_receipt_sha256=parent_physical,
    )
    gates: list[dict[str, object]] = []
    for gate in ("D1", "D2", "D3", "D4", "D5", "D6"):
        receipt: dict[str, object] = {
            "authoritative": True,
            "corpus_content_identity_sha256": pa.content_identity_sha256,
            "d1_ready_identity_sha256": pa.d1_ready_identity_sha256,
            "evidence_artifact_sha256s": expected[gate],
            "gate": gate,
            "release_manifest_section_identity_sha256": (
                pa.release_manifest_section_identity_sha256
            ),
            "status": "PASS",
            "verifier_kind": pb._GATE_VERIFIERS[gate],
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
        "parent_replay_receipt_identity_sha256": "f" * 64,
        "parent_replay_receipt_sha256": parent_physical,
        "release_manifest_section_identity_sha256": pa.release_manifest_section_identity_sha256,
        "schema": pb.PB_GATE_BUNDLE_SCHEMA_V4,
    }
    bundle["bundle_identity_sha256"] = execution_authority_v4_bound_sha256(
        pb.PB_GATE_BUNDLE_SCHEMA_V4, bundle
    )
    return bundle


def _decon_payload(pa: pb.PAInspectionV4, *, status: str = "CLEAN") -> dict[str, object]:
    hit = status == "HIT"
    input_core = {
        "confirm_complete_ledger_sha256": (
            pb.GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256
        ),
        "confirm_private_rows_sha256": pb.GOVERNED_CONFIRM_SOURCE_ROWS_SHA256,
        "confirm_seal_file_set_sha256": pb.GOVERNED_CONFIRM_SEAL_SET_SHA256,
        "confirm_seal_ledger_sha256": "3" * 64,
        "confirm_source_manifest_sha256": (
            pb.GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256
        ),
        "eval_e_anonymous_index_sha256": (
            pb.GOVERNED_EVAL_E_ANONYMOUS_INDEX_SHA256
        ),
        "eval_e_lock_sha256": pb.GOVERNED_EVAL_E_LOCK_SHA256,
    }
    inputs = {
        **input_core,
        "private_input_set_commitment_sha256": hashlib.sha256(
            canonical_json_bytes(input_core)
        ).hexdigest(),
    }
    repository_root = Path(pb.__file__).resolve().parents[1]
    code_rows = [
        {
            "relative_path": relative,
            "sha256": hashlib.sha256(
                (repository_root / relative).read_bytes()
            ).hexdigest(),
        }
        for relative in pb._DECON_CODE_RELATIVE_PATHS
    ]
    profiles = pb.decon_algorithm_profiles()
    registry = hashlib.sha256(
        canonical_json_bytes(
            {
                "algorithm_profiles": profiles,
                "input_commitments": inputs,
                "registered_battery_count": len(pb.DECON_REQUIRED_BATTERIES),
            }
        )
    ).hexdigest()
    payload: dict[str, object] = {
        "algorithm_profiles": profiles,
        "authority_chain": pb.PB_AUTHORITY_CHAIN_V5,
        "battery_scope": pb.DECON_REQUIRED_BATTERIES,
        "corpus_content_identity_sha256": pa.content_identity_sha256,
        "corpus_manifest_sha256": pa.content_manifest_physical_sha256,
        "exact_match_count": 1 if hit else 0,
        "full_shard_manifest_identity_sha256": pa.full_shard_manifest_identity_sha256,
        "full_shard_manifest_sha256": pa.full_shard_manifest_physical_sha256,
        "hermetic": True,
        "hit_action": "HARD_STOP_NO_MINT",
        "input_commitments": inputs,
        "near_match_count": 0,
        "network_accessed": False,
        "plaintext_exported": False,
        "registered_battery_count": len(pb.DECON_REQUIRED_BATTERIES),
        "release_manifest_section_identity_sha256": pa.release_manifest_section_identity_sha256,
        "runtime_commitments": {
            "global_execution_provenance_sha256": hashlib.sha256(
                (pa.root / pb.replay_v3.GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3).read_bytes()
            ).hexdigest(),
            "network_guard_sha256": "7" * 64,
            "python_executable_sha256": "8" * 64,
            "runtime_build_receipt_sha256": hashlib.sha256(
                (pa.root / pb.replay_v3.RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1).read_bytes()
            ).hexdigest(),
            "unshare_executable_sha256": "9" * 64,
        },
        "salt_exported": False,
        "schema": pb.PB_DECON_SCHEMA_V5,
        "screen_code_commitments": code_rows,
        "screen_code_set_commitment_sha256": hashlib.sha256(
            canonical_json_bytes(code_rows)
        ).hexdigest(),
        "screened_battery_count": len(pb.DECON_REQUIRED_BATTERIES),
        "screened_battery_set_commitment_sha256": hashlib.sha256(
            canonical_json_bytes(pb.DECON_REQUIRED_BATTERIES)
        ).hexdigest(),
        "screened_document_count": sum(
            int(row["record_count"]) for row in pa.full_shard_rows
        ),
        "screened_full_shard_count": len(pa.full_shard_rows),
        "screened_full_shards": list(pb._full_shard_projection(pa)),
        "screened_full_shard_set_commitment_sha256": pb._full_shard_set_commitment(pa),
        "screen_submanifest_identity_sha256": pa.screen_submanifest_identity_sha256,
        "screen_submanifest_sha256": pa.screen_submanifest_physical_sha256,
        "sealed_battery_registry_commitment_sha256": registry,
        "sealed_identifiers_exported": False,
        "status": status,
        "total_match_count": 1 if hit else 0,
    }
    payload["receipt_sha256"] = pb.pb_authority_bound_sha256(
        pb.PB_DECON_SCHEMA_V5, payload
    )
    return payload


def _parent_replay_payload(pa: pb.PAInspectionV4, second_root: Path) -> dict[str, object]:
    second_root.mkdir(exist_ok=True)
    core = {
        "first_child_receipt_sha256": "1" * 64,
        "input_identity_sha256": "2" * 64,
        "second_child_receipt_sha256": "3" * 64,
        "worker_compatibility_sha256": "4" * 64,
    }
    typed = pb.ParentReplayVerificationV4(
        status="PASS",
        authoritative=True,
        d1_file_replay_verified=True,
        d2_dedup_replay_verified=True,
        v4_content_profile_verified=True,
        release_binding_verified=True,
        runtime_provenance_verified=True,
        os_network_isolation_verified=True,
        durable_post_write_rehash_verified=True,
        input_identity_sha256=core["input_identity_sha256"],
        worker_compatibility_sha256=core["worker_compatibility_sha256"],
        first_child_receipt_sha256=core["first_child_receipt_sha256"],
        second_child_receipt_sha256=core["second_child_receipt_sha256"],
        first_output_root=str(pa.root),
        second_output_root=str(second_root),
        durable_output_parent=str(pa.root.parent),
        local_work_parent=str(pa.root.parent / "work"),
        evidence_sha256=execution_authority_v4_bound_sha256(
            pb.PARENT_EVIDENCE_SCHEMA_V4, core
        ),
    )
    return {**asdict(typed), "receipt_sha256": typed.receipt_sha256}


def _balanced_independent_scan() -> dict[str, object]:
    source_bytes = {
        "dolma_web": 39,
        "wikipedia_wikibooks": 22,
        "stackedu": 25,
        "finemath_3plus": 15,
        "arxiv": 8,
        "olmocr": 7,
        "fineweb_edu": 39,
    }
    return {
        "sources": tuple(
            {
                "document_count": 1,
                "name": source,
                "raw_byte_count": source_bytes[source],
            }
            for source in SOURCE_FAMILIES
        ),
        "screen_streams": (
            {
                "document_count": 4,
                "framed_retained_text_sha256": "a" * 64,
                "retained_text_bytes": 100,
                "stream": "T",
            },
            {
                "document_count": 4,
                "framed_retained_text_sha256": "b" * 64,
                "retained_text_bytes": 100,
                "stream": "H",
            },
        ),
    }


def _write_balanced_d3(pa: pb.PAInspectionV4) -> None:
    scan = _balanced_independent_scan()
    source = {row["name"]: row["raw_byte_count"] for row in scan["sources"]}
    pools = (
        ("wikipedia_wikibooks", source["wikipedia_wikibooks"]),
        ("dolma_web", source["dolma_web"]),
        ("fineweb_edu", source["fineweb_edu"]),
        ("stackedu", source["stackedu"]),
        ("finemath_3plus", source["finemath_3plus"]),
        ("science_technical_combined", source["arxiv"] + source["olmocr"]),
    )
    _, old = pb._load_canonical(pa.root / "diagnostics" / "d3.json", "old D3")
    d3 = {
        **old,
        "full_pool_rows": tuple(
            {
                "deficit_fraction": {"denominator": 1, "numerator": 0},
                "observed_bytes": observed,
                "pool": pool,
                "target_bytes": observed,
            }
            for pool, observed in pools
        ),
    }
    _write(pa.root / "diagnostics" / "d3.json", d3)


def _write_valid_d4_d5(pa: pb.PAInspectionV4) -> None:
    _write(
        pa.root / "diagnostics" / "d4.json",
        {
            "gate": "D4",
            "invocation_counts": tuple(
                (stratum, 1 if stratum == "general" else 0)
                for stratum in GTOK_STRATA
            ),
            "rejection_counts": tuple((stratum, 0) for stratum in GTOK_STRATA),
            "status": "CHECK_PASS_NO_GATE_MINT",
        },
    )
    _write(
        pa.root / "diagnostics" / "d5.json",
        {
            "cases": tuple(
                {
                    name: row[name]
                    for name in (
                        "logical_jsonl_sha256",
                        "record_count",
                        "relative_path",
                        "retained_text_bytes",
                        "zstd_sha256",
                    )
                }
                for row in pa.screen_shard_rows
            ),
            "gate": "D5",
            "status": "CHECK_PASS_NO_GATE_MINT",
        },
    )


def _minimal_d6(
    scan: dict[str, object], split_rows: tuple[dict[str, object], ...]
) -> dict[str, object]:
    return {
        "cluster_overlap_count": 0,
        "consumer_bindings": [],
        "consumer_order_receipts": [],
        "document_overlap_count": 0,
        "full_corpus_repeated_raw_content_id_count": 0,
        "gate": "D6",
        "near_cluster_receipt": {},
        "screen_repeated_raw_content_id_count": 0,
        "split_rows": split_rows,
        "status": "CHECK_PASS_NO_GATE_MINT",
        "stream_identities": scan["screen_streams"],
        "tokenizer_fit_contract": {},
    }


def _physical_d6_prefix(
    pa: pb.PAInspectionV4, scan: dict[str, object]
) -> dict[str, object]:
    return {
        "document_overlap_count": 0,
        "repeated_raw_content_id_count": 0,
        "split_groups": [
            {
                name: group[name]
                for name in (
                    "document_count",
                    "ordered_raw_content_ids_sha256",
                    "retained_text_bytes",
                    "stratum",
                    "stream",
                )
            }
            for group in pa.screen_groups
        ],
        "stream_identities": [
            {**row, "ordered_raw_content_ids_sha256": "f" * 64}
            for row in scan["screen_streams"]
        ],
    }


def test_c2_fixture_evidence_is_complete_and_exact(tmp_path: Path) -> None:
    path = tmp_path / "c2.json"
    physical, identity = _write_c2(path)

    assert pb.load_c2_fixture_evidence(path) == (physical, identity)


def test_c2_rejects_a_self_consistent_substitute_payload(tmp_path: Path) -> None:
    evidence = pb.build_c2_fixture_evidence()
    changed = json.loads(json.dumps(evidence))
    case = changed["cases"][0]
    substitute = b"registered-name-wrong-payload"
    encoded = base64.b64encode(substitute).decode("ascii")
    digest = hashlib.sha256(substitute).hexdigest()
    case["original_bytes_b64"] = encoded
    case["original_sha256"] = digest
    case["round_trip_bytes_b64"] = encoded
    case["round_trip_sha256"] = digest
    changed.pop("suite_identity_sha256")
    changed["suite_identity_sha256"] = pb.pb_authority_bound_sha256(
        pb.PB_C2_EVIDENCE_SCHEMA_V5, changed
    )
    path = tmp_path / "c2-substitute.json"
    _write(path, changed)

    with pytest.raises(pb.PBFreezeError, match="registered case"):
        pb.load_c2_fixture_evidence(path)


def test_c2_detects_text_mutation_before_the_production_shard_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_writer = pb.corpus_pa.write_jsonl_zstd_shards_v3

    def mutating_writer(documents: object, *args: object, **kwargs: object) -> object:
        rows = list(documents)
        first = rows[0]
        rows[0] = StableDocumentV3(
            source=first.source,
            stratum=first.stratum,
            stable_source_record_id=first.stable_source_record_id,
            text=first.text + " mutation",
        )
        return original_writer(tuple(rows), *args, **kwargs)

    monkeypatch.setattr(
        pb.corpus_pa, "write_jsonl_zstd_shards_v3", mutating_writer
    )

    with pytest.raises(pb.PBFreezeError, match="changed fixture bytes"):
        pb.build_c2_fixture_evidence()


def test_d4_language_counts_are_reread_from_physical_ledger(tmp_path: Path) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    decision = pb.LanguageIdDecisionV3(
        document_id=_document("dolma_web").document_id,
        stratum="general",
        scoring_input_sha256="a" * 64,
        label=pb.A2_LANGUAGE_ID_BINDING.keep_label,
        probability=0.99,
    )
    row = {
        "binding_sha256": decision.binding_sha256,
        "document_id": decision.document_id,
        "keep": decision.keep,
        "label": decision.label,
        "probability": decision.probability,
        "receipt_sha256": decision.receipt_sha256,
        "scoring_input_sha256": decision.scoring_input_sha256,
        "source": "dolma_web",
    }
    ledger = pa.root / "artifacts" / "language-decisions.jsonl"
    ledger.write_bytes(canonical_json_bytes(row) + b"\n")

    digest, invocations, rejections = pb._reread_language_decisions(pa)

    assert digest == hashlib.sha256(ledger.read_bytes()).hexdigest()
    assert (invocations, rejections) == (1, 0)


def test_cli_emits_c2_evidence_once_and_fails_closed_on_reuse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "c2.json"

    assert pb.main(["c2-fixtures", "--output", str(output)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "C2_EVIDENCE_WRITTEN_NO_FREEZE_MINT"
    assert output.is_file()
    assert pb.main(["c2-fixtures", "--output", str(output)]) == 2
    assert "already exists" in capsys.readouterr().err


def test_v4_full_and_screen_manifests_are_parent_bound(tmp_path: Path) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    content = {
        "v4_full_corpus": {
            "d6_physical_evidence_identity_sha256": (
                pa.d6_physical_evidence_identity_sha256
            ),
            "d6_physical_evidence_path": pa.d6_physical_evidence_relative_path,
            "d6_physical_evidence_sha256": (
                pa.d6_physical_evidence_physical_sha256
            ),
            "document_count": sum(
                int(row["document_count"]) for row in pa.full_source_summaries
            ),
            "full_manifest_identity_sha256": pa.full_shard_manifest_identity_sha256,
            "full_manifest_path": pa.full_shard_manifest_relative_path,
            "full_manifest_sha256": pa.full_shard_manifest_physical_sha256,
            "non_screen_full_document_count": (
                sum(
                    int(row["document_count"])
                    for row in pa.full_source_summaries
                )
                - sum(int(group["document_count"]) for group in pa.screen_groups)
            ),
            "retained_text_bytes": sum(
                int(row["retained_text_bytes"]) for row in pa.full_source_summaries
            ),
            "screen_submanifest_identity_sha256": pa.screen_submanifest_identity_sha256,
            "screen_submanifest_path": pa.screen_submanifest_relative_path,
            "screen_submanifest_sha256": pa.screen_submanifest_physical_sha256,
        }
    }

    loaded = pb._load_full_shard_manifest(pa.root, content)

    assert loaded[0] == FULL_SHARD_MANIFEST_RELATIVE_PATH_V4
    assert loaded[2] == pa.full_shard_manifest_identity_sha256
    assert loaded[4] == pa.full_source_summaries
    assert loaded[7] == pa.screen_submanifest_identity_sha256
    assert loaded[9] == pa.d6_physical_evidence_physical_sha256


def test_scan_uses_disk_index_and_reconciles_strata_sources_and_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, stratum_bytes = _build_scan_fixture(tmp_path)
    monkeypatch.setattr(pb, "FULL_STRATUM_TARGETS_V4", stratum_bytes)

    evidence = pb._scan_full_shards(pa)

    assert evidence["c1_status"] == "PASS"
    assert evidence["c3_status"] == "PASS"
    assert tuple(row["name"] for row in evidence["sources"]) == SOURCE_FAMILIES
    source = inspect.getsource(pb._scan_full_shards)
    assert 'sqlite3.connect("")' in source
    assert "CREATE TABLE full_membership" in source
    assert "full_locations" not in source
    assert "seen_ids" not in source
    assert "text TEXT" not in source


def test_scan_rejects_source_summary_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, stratum_bytes = _build_scan_fixture(tmp_path)
    monkeypatch.setattr(pb, "FULL_STRATUM_TARGETS_V4", stratum_bytes)
    changed_sources = list(pa.full_source_summaries)
    changed_sources[0] = {**changed_sources[0], "retained_text_bytes": 999}

    with pytest.raises(pb.PBFreezeError, match="source summary"):
        pb._scan_full_shards(
            replace(pa, full_source_summaries=tuple(changed_sources))
        )


def test_scan_rejects_mixed_source_full_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, stratum_bytes = _build_scan_fixture(tmp_path)
    monkeypatch.setattr(pb, "FULL_STRATUM_TARGETS_V4", stratum_bytes)
    rows = list(pa.full_shard_rows)
    rows[0] = {**rows[0], "source": "wikipedia_wikibooks"}

    with pytest.raises(pb.PBFreezeError, match="record stratum/text drifted"):
        pb._scan_full_shards(replace(pa, full_shard_rows=tuple(rows)))


def test_physical_d6_is_freshly_recomputed_in_raw_content_id_domain(
    tmp_path: Path,
) -> None:
    pa, _ = _build_scan_fixture(tmp_path)

    physical, evidence = pb._recompute_physical_d6_evidence(pa)

    assert physical == pa.d6_physical_evidence_physical_sha256
    assert all(
        row["schema"] == pb.CONSUMER_ORDER_SCHEMA_V4
        and "ordered_raw_content_ids_sha256" in row
        and "ordered_document_ids_sha256" not in row
        for row in evidence["consumer_order_receipts"]
    )

    changed = json.loads(json.dumps(evidence))
    order = changed["consumer_order_receipts"][0]
    order["framed_payload_sha256"] = "0" * 64
    order.pop("receipt_sha256")
    order["receipt_sha256"] = execution_authority_v4_bound_sha256(
        pb.CONSUMER_ORDER_SCHEMA_V4, order
    )
    changed.pop("evidence_identity_sha256")
    changed["evidence_identity_sha256"] = execution_authority_v4_bound_sha256(
        pb.D6_PHYSICAL_EVIDENCE_SCHEMA_V4, changed
    )
    path = pa.root / pa.d6_physical_evidence_relative_path
    changed_physical = _write(path, changed)
    changed_pa = replace(
        pa,
        d6_physical_evidence_physical_sha256=changed_physical,
        d6_physical_evidence_identity_sha256=str(
            changed["evidence_identity_sha256"]
        ),
    )

    with pytest.raises(pb.PBFreezeError, match="fresh recomputation"):
        pb._recompute_physical_d6_evidence(changed_pa)


def test_physical_d6_consumer_contract_binds_distinct_governed_data_seeds(
    tmp_path: Path,
) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    _physical, evidence = pb._recompute_physical_d6_evidence(pa)

    pb._validate_physical_d6_consumers(evidence)
    assert tuple(
        (row["training_seed"], row["data_order_seed"])
        for row in evidence["consumer_order_receipts"]
    ) == tuple(pb.GTOK_DATA_ORDER_SEED_BY_TRAINING_SEED_V4.items())
    assert {
        (row["training_seed"], row["data_order_seed"])
        for row in evidence["consumer_bindings"]
    } == set(pb.GTOK_DATA_ORDER_SEED_BY_TRAINING_SEED_V4.items())


@pytest.mark.parametrize("surface", ("receipt", "binding"))
def test_physical_d6_consumer_contract_rejects_data_seed_substitution(
    tmp_path: Path,
    surface: str,
) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    _physical, evidence = pb._recompute_physical_d6_evidence(pa)
    changed = json.loads(json.dumps(evidence))
    if surface == "receipt":
        row = changed["consumer_order_receipts"][0]
        row["data_order_seed"] = row["training_seed"]
        row.pop("receipt_sha256")
        row["receipt_sha256"] = execution_authority_v4_bound_sha256(
            pb.CONSUMER_ORDER_SCHEMA_V4, row
        )
        match = "order receipt drifted"
    else:
        row = changed["consumer_bindings"][0]
        row["data_order_seed"] = row["training_seed"]
        match = "consumer binding drifted"

    with pytest.raises(pb.PBFreezeError, match=match):
        pb._validate_physical_d6_consumers(changed)


def test_d3_rejects_general_source_mix_outside_two_percent(tmp_path: Path) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    scan = _balanced_independent_scan()
    source_bytes = {
        "wikipedia_wikibooks": 10,
        "dolma_web": 45,
        "fineweb_edu": 45,
        "stackedu": 25,
        "finemath_3plus": 15,
        "arxiv": 8,
        "olmocr": 7,
    }
    scan["sources"] = tuple(
        {
            "document_count": 1,
            "name": source,
            "raw_byte_count": source_bytes[source],
        }
        for source in SOURCE_FAMILIES
    )
    pools = (
        ("wikipedia_wikibooks", source_bytes["wikipedia_wikibooks"]),
        ("dolma_web", source_bytes["dolma_web"]),
        ("fineweb_edu", source_bytes["fineweb_edu"]),
        ("stackedu", source_bytes["stackedu"]),
        ("finemath_3plus", source_bytes["finemath_3plus"]),
        (
            "science_technical_combined",
            source_bytes["arxiv"] + source_bytes["olmocr"],
        ),
    )
    _write(
        pa.root / "diagnostics" / "d3.json",
        {
            "full_pool_rows": tuple(
                {
                    "deficit_fraction": {"denominator": 1, "numerator": 0},
                    "observed_bytes": observed,
                    "pool": pool,
                    "target_bytes": observed,
                }
                for pool, observed in pools
            ),
            "gate": "D3",
            "observed_stratum_bytes": [],
            "pool_receipts": [],
            "status": "CHECK_PASS_NO_GATE_MINT",
        },
    )

    with pytest.raises(pb.PBFreezeError, match="22/39/39"):
        pb._validate_d4_d5_d6(pa, independent_scan=scan)


def test_malformed_d4_pair_rows_fail_as_pb_error(tmp_path: Path) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    _write_balanced_d3(pa)
    _write(
        pa.root / "diagnostics" / "d4.json",
        {
            "gate": "D4",
            "invocation_counts": [["general"]],
            "rejection_counts": tuple((stratum, 0) for stratum in GTOK_STRATA),
            "status": "CHECK_PASS_NO_GATE_MINT",
        },
    )

    with pytest.raises(pb.PBFreezeError, match="D4 invocation rows"):
        pb._validate_d4_d5_d6(
            pa, independent_scan=_balanced_independent_scan()
        )


def test_d6_rejects_stream_stratification_outside_one_percent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    scan = _balanced_independent_scan()
    _write_balanced_d3(pa)
    _write_valid_d4_d5(pa)
    split_rows = tuple(
        {"heldout": {}, "stratum": stratum, "training": {}}
        for stratum in GTOK_STRATA
    )
    _write(pa.root / "diagnostics" / "d6.json", _minimal_d6(scan, split_rows))
    monkeypatch.setattr(
        pb, "_reread_language_decisions", lambda unused: ("a" * 64, 1, 0)
    )
    monkeypatch.setattr(
        pb,
        "_recompute_physical_d6_evidence",
        lambda unused: ("b" * 64, _physical_d6_prefix(pa, scan)),
    )

    with pytest.raises(pb.PBFreezeError, match="stratification"):
        pb._validate_d4_d5_d6(pa, independent_scan=scan)


def test_d6_rejects_per_split_deficit_above_half_percent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    scan = _balanced_independent_scan()
    shares = {"general": 45, "code": 25, "mathematics": 15, "science_technical": 15}
    groups = tuple(
        {
            **group,
            "document_count": 1,
            "retained_text_bytes": shares[str(group["stratum"])],
        }
        for group in pa.screen_groups
    )
    pa = replace(pa, screen_groups=groups)
    _write_balanced_d3(pa)
    _write_valid_d4_d5(pa)
    split_rows = []
    for stratum in GTOK_STRATA:
        realized = shares[stratum]
        split_rows.append(
            {
                "heldout": {
                    "deficit_bytes": 0,
                    "document_count": 1,
                    "realized_bytes": realized,
                    "target_bytes": realized,
                },
                "stratum": stratum,
                "training": {
                    "deficit_bytes": 55 if stratum == "general" else 0,
                    "document_count": 1,
                    "realized_bytes": realized,
                    "target_bytes": 100 if stratum == "general" else realized,
                },
            }
        )
    _write(
        pa.root / "diagnostics" / "d6.json",
        _minimal_d6(scan, tuple(split_rows)),
    )
    monkeypatch.setattr(
        pb, "_reread_language_decisions", lambda unused: ("a" * 64, 1, 0)
    )
    monkeypatch.setattr(
        pb,
        "_recompute_physical_d6_evidence",
        lambda unused: ("b" * 64, _physical_d6_prefix(pa, scan)),
    )
    monkeypatch.setattr(
        pb,
        "GTOK_SCREEN_TRAIN_STRATUM_TARGETS",
        tuple(
            (stratum, 100 if stratum == "general" else shares[stratum])
            for stratum in GTOK_STRATA
        ),
    )
    monkeypatch.setattr(
        pb,
        "GTOK_SCREEN_HELDOUT_STRATUM_TARGETS",
        tuple((stratum, shares[stratum]) for stratum in GTOK_STRATA),
    )

    with pytest.raises(pb.PBFreezeError, match="split accounting"):
        pb._validate_d4_d5_d6(pa, independent_scan=scan)


def test_d1_d6_bundle_requires_every_authoritative_gate(tmp_path: Path) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    c2_path = tmp_path / "c2.json"
    c2_physical, _ = _write_c2(c2_path)
    path = tmp_path / "gates.json"
    bundle = _gate_bundle(pa, c2_physical_sha256=c2_physical)
    physical = _write(path, bundle)

    loaded = pb.load_d1_d6_gate_bundle(
        path, pa=pa, c2_evidence_sha256=c2_physical
    )
    assert loaded[0] == physical
    assert tuple(gate for gate, unused in loaded[2]) == (
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
    )

    changed = _gate_bundle(pa, c2_physical_sha256=c2_physical)
    changed["gates"][3]["authoritative"] = False
    changed["gates"][3].pop("receipt_sha256")
    changed["gates"][3]["receipt_sha256"] = execution_authority_v4_bound_sha256(
        "weft1_corpus_d4_gate_receipt_v4", changed["gates"][3]
    )
    changed.pop("bundle_identity_sha256")
    changed["bundle_identity_sha256"] = execution_authority_v4_bound_sha256(
        pb.PB_GATE_BUNDLE_SCHEMA_V4, changed
    )
    _write(path, changed)
    with pytest.raises(pb.PBFreezeError, match="D4 is not authoritative"):
        pb.load_d1_d6_gate_bundle(
            path, pa=pa, c2_evidence_sha256=c2_physical
        )


def test_gate_bundle_minter_consumes_factory_parent_and_is_fresh_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    c2_path = tmp_path / "c2.json"
    c2_physical, _ = _write_c2(c2_path)
    parent_path = tmp_path / "parent.json"
    parent_payload = _parent_replay_payload(pa, tmp_path / "replay-b")
    parent_physical = _write(parent_path, parent_payload)
    monkeypatch.setattr(pb, "inspect_pa_v4", lambda unused: pa)
    monkeypatch.setattr(
        pb,
        "load_parent_replay_verification_v4",
        lambda unused, *, pa: (
            parent_physical,
            parent_payload["receipt_sha256"],
        ),
    )
    monkeypatch.setattr(
        pb,
        "_scan_full_shards",
        lambda unused: {"c1_status": "PASS", "c3_status": "PASS"},
    )
    monkeypatch.setattr(pb, "_validate_d4_d5_d6", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pb, "_rehash_gate_mint_inputs", lambda *args, **kwargs: None
    )
    output = tmp_path / "gates-minted.json"

    assert (
        pb.main(
            [
                "gates",
                "--materialization-root",
                str(pa.root),
                "--parent-replay-receipt",
                str(parent_path),
                "--c2-evidence",
                str(c2_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "D1_D6_AUTHORITATIVE_PASS_BUNDLE_MINTED"
    physical, bundle = pb._load_canonical(output, "minted gate bundle")
    assert hashlib.sha256(physical).hexdigest() == summary["physical_sha256"]
    pb.load_d1_d6_gate_bundle(
        output, pa=pa, c2_evidence_sha256=c2_physical
    )
    assert all(row["status"] == "PASS" for row in bundle["gates"])
    assert (
        pb.main(
            [
                "gates",
                "--materialization-root",
                str(pa.root),
                "--parent-replay-receipt",
                str(parent_path),
                "--c2-evidence",
                str(c2_path),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "already exists" in capsys.readouterr().err


def test_parent_replay_mutation_blocks_gate_mint_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    c2_path = tmp_path / "c2.json"
    _write_c2(c2_path)
    parent = _parent_replay_payload(pa, tmp_path / "replay-b")
    parent["authoritative"] = False
    parent.pop("receipt_sha256")
    parent["receipt_sha256"] = execution_authority_v4_bound_sha256(
        pb.PARENT_REPLAY_SCHEMA_V4, parent
    )
    parent_path = tmp_path / "parent-mutated.json"
    _write(parent_path, parent)
    monkeypatch.setattr(pb, "inspect_pa_v4", lambda unused: pa)
    output = tmp_path / "gates.json"

    with pytest.raises(pb.PBFreezeError, match="not a factory PASS"):
        pb.mint_d1_d6_gate_bundle(
            materialization_root=pa.root,
            parent_replay_receipt_path=parent_path,
            c2_evidence_path=c2_path,
            output_path=output,
        )
    assert not output.exists()


def test_parent_replay_pass_fields_cannot_replace_child_tree_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    c2_path = tmp_path / "c2.json"
    _write_c2(c2_path)
    parent_path = tmp_path / "parent-pass-empty-children.json"
    _write(parent_path, _parent_replay_payload(pa, tmp_path / "empty-replay-b"))
    monkeypatch.setattr(pb, "inspect_pa_v4", lambda unused: pa)
    output = tmp_path / "gates.json"

    with pytest.raises(pb.PBFreezeError, match="child replay evidence"):
        pb.mint_d1_d6_gate_bundle(
            materialization_root=pa.root,
            parent_replay_receipt_path=parent_path,
            c2_evidence_path=c2_path,
            output_path=output,
        )
    assert not output.exists()


def test_content_manifest_recomputes_and_exactly_binds_release_section() -> None:
    release = pb.release_manifest_section()
    release_identity = execution_authority_v4_bound_sha256(
        RELEASE_MANIFEST_SECTION_SCHEMA_V4, release
    )
    payload: dict[str, object] = {
        "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
        "authoritative_gate_receipts": [],
        "mode": "PRODUCTION",
        "readiness": V4_READINESS,
        "release": {
            "authority_sha256": pb.RELEASE_AUTHORITY_SHA256,
            "manifest_section": release,
            "manifest_section_identity_sha256": release_identity,
        },
        "schema": MATERIALIZER_SCHEMA_V4,
    }
    payload["content_identity_sha256"] = execution_authority_v4_bound_sha256(
        MATERIALIZED_CONTENT_SCHEMA_V4, payload
    )
    assert pb._verify_content_manifest(payload, physical_sha256="f" * 64) == (
        payload["content_identity_sha256"],
        release_identity,
    )

    changed = json.loads(json.dumps(payload))
    changed["release"]["manifest_section"]["public_release"][
        "raw_text_shards_published"
    ] = True
    changed.pop("content_identity_sha256")
    changed["content_identity_sha256"] = execution_authority_v4_bound_sha256(
        MATERIALIZED_CONTENT_SCHEMA_V4, changed
    )
    with pytest.raises(pb.PBFreezeError, match="release manifest section drifted"):
        pb._verify_content_manifest(changed, physical_sha256="e" * 64)


def test_decon_requires_seven_batteries_and_every_full_shard(tmp_path: Path) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    clean = _decon_payload(pa)
    path = tmp_path / "decon.json"
    physical = _write(path, clean)

    assert pb.load_hermetic_decon_receipt(path, pa=pa) == (
        physical,
        clean["receipt_sha256"],
        "CLEAN",
    )
    assert (
        clean["screened_battery_set_commitment_sha256"]
        != clean["sealed_battery_registry_commitment_sha256"]
    )

    changed = dict(clean)
    changed["screened_full_shard_count"] = len(pa.full_shard_rows) - 1
    changed.pop("receipt_sha256")
    changed["receipt_sha256"] = pb.pb_authority_bound_sha256(
        pb.PB_DECON_SCHEMA_V5, changed
    )
    _write(path, changed)
    with pytest.raises(pb.PBFreezeError, match="does not bind"):
        pb.load_hermetic_decon_receipt(path, pa=pa)


def test_decon_rejects_self_consistent_code_substitution(tmp_path: Path) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    receipt = _decon_payload(pa)
    receipt["screen_code_commitments"][0]["sha256"] = "0" * 64
    receipt["screen_code_set_commitment_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt["screen_code_commitments"])
    ).hexdigest()
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = pb.pb_authority_bound_sha256(
        pb.PB_DECON_SCHEMA_V5, receipt
    )
    path = tmp_path / "decon-code-substitute.json"
    _write(path, receipt)

    with pytest.raises(pb.PBFreezeError, match="code bytes"):
        pb.load_hermetic_decon_receipt(path, pa=pa)


def test_decon_rejects_self_consistent_eval_e_substitution(tmp_path: Path) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    receipt = _decon_payload(pa)
    inputs = dict(receipt["input_commitments"])
    inputs["eval_e_anonymous_index_sha256"] = "4" * 64
    input_core = {
        name: inputs[name]
        for name in sorted(inputs)
        if name != "private_input_set_commitment_sha256"
    }
    inputs["private_input_set_commitment_sha256"] = hashlib.sha256(
        canonical_json_bytes(input_core)
    ).hexdigest()
    receipt["input_commitments"] = inputs
    receipt["sealed_battery_registry_commitment_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {
                "algorithm_profiles": receipt["algorithm_profiles"],
                "input_commitments": inputs,
                "registered_battery_count": receipt["registered_battery_count"],
            }
        )
    ).hexdigest()
    receipt["receipt_sha256"] = pb.pb_authority_bound_sha256(
        pb.PB_DECON_SCHEMA_V5, receipt
    )
    path = tmp_path / "decon-eval-e-substitute.json"
    _write(path, receipt)

    with pytest.raises(pb.PBFreezeError, match="governed input identity"):
        pb.load_hermetic_decon_receipt(path, pa=pa)


def test_decon_rejects_plaintext_field_even_if_receipt_is_rehashed(tmp_path: Path) -> None:
    pa, _ = _build_scan_fixture(tmp_path)
    receipt = _decon_payload(pa)
    receipt["plaintext"] = "sealed prompt"
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = pb.pb_authority_bound_sha256(
        pb.PB_DECON_SCHEMA_V5, receipt
    )
    path = tmp_path / "decon.json"
    _write(path, receipt)

    with pytest.raises(pb.PBFreezeError, match="fields drifted"):
        pb.load_hermetic_decon_receipt(path, pa=pa)


def test_hit_hard_stops_before_fresh_output_is_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pb,
        "launch_hermetic_decon",
        lambda **unused: ("a" * 64, "b" * 64, "HIT"),
    )
    output = tmp_path / "freeze.json"

    with pytest.raises(pb.DecontaminationHit):
        pb.mint_freeze_receipt(
            materialization_root=tmp_path / "pa",
            gate_bundle_path=tmp_path / "gates.json",
            c2_evidence_path=tmp_path / "c2.json",
            confirm_seal_paths=(tmp_path / "seal.json",),
            confirm_seal_ledger_path=tmp_path / "ledger.json",
            confirm_private_rows_path=tmp_path / "private.jsonl",
            eval_e_index_path=tmp_path / "eval-index.json",
            eval_e_lock_path=tmp_path / "eval-lock.json",
            decon_output_root=tmp_path / "decon-output",
            decon_local_work_parent=tmp_path / "local-work",
            output_path=output,
        )
    assert not output.exists()


def test_mint_is_exclusive_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {
        "decon_receipt_identity_sha256": "b" * 64,
        "decon_receipt_sha256": "a" * 64,
        "freeze_receipt_identity_sha256": "f" * 64,
        "schema": pb.PB_FREEZE_SCHEMA_V5,
        "status": "FROZEN",
    }
    monkeypatch.setattr(pb, "build_freeze_receipt", lambda **kwargs: receipt)
    monkeypatch.setattr(
        pb,
        "launch_hermetic_decon",
        lambda **unused: ("a" * 64, "b" * 64, "CLEAN"),
    )
    output = tmp_path / "freeze.json"
    kwargs = {
        "materialization_root": tmp_path,
        "gate_bundle_path": tmp_path / "gates.json",
        "c2_evidence_path": tmp_path / "c2.json",
        "confirm_seal_paths": (tmp_path / "seal.json",),
        "confirm_seal_ledger_path": tmp_path / "ledger.json",
        "confirm_private_rows_path": tmp_path / "private.jsonl",
        "eval_e_index_path": tmp_path / "eval-index.json",
        "eval_e_lock_path": tmp_path / "eval-lock.json",
        "decon_output_root": tmp_path / "decon-output",
        "decon_local_work_parent": tmp_path / "local-work",
        "output_path": output,
    }

    minted, physical = pb.mint_freeze_receipt(**kwargs)

    assert minted == receipt
    assert hashlib.sha256(output.read_bytes()).hexdigest() == physical
    with pytest.raises(pb.PBFreezeError, match="already exists"):
        pb.mint_freeze_receipt(**kwargs)


def test_external_hash_shaped_decon_receipt_cannot_cross_mint_boundary(
    tmp_path: Path,
) -> None:
    external = tmp_path / "self-authored-decon.json"
    external.write_text('{"status":"CLEAN"}\n', encoding="utf-8")
    output = tmp_path / "freeze.json"
    assert "decon_receipt_path" not in inspect.signature(
        pb.mint_freeze_receipt
    ).parameters
    with pytest.raises(TypeError, match="decon_receipt_path"):
        pb.mint_freeze_receipt(  # type: ignore[call-arg]
            materialization_root=tmp_path / "pa",
            gate_bundle_path=tmp_path / "gates.json",
            c2_evidence_path=tmp_path / "c2.json",
            decon_receipt_path=external,
            output_path=output,
        )
    assert not output.exists()
