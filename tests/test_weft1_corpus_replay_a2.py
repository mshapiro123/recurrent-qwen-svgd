from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import training.weft1_corpus_a2 as corpus_contracts
import training.weft1_corpus_pa as production_io
import training.weft1_corpus_replay_a2 as replay
import training.weft1_corpus_sources_a2 as source_routes

from training.weft1_corpus_replay_a2 import (
    CHILD_RECEIPT_SCHEMA_V3,
    PARENT_RECEIPT_SCHEMA_V3,
    ParentReplayError,
    ParentReplayVerificationV3,
    verify_parent_replays_v3,
    verify_production_materialization_replays_v3,
)


ROOT = Path(__file__).resolve().parents[1]


_WORKER_SOURCE = r'''from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, os.getcwd())
from training.weft1_corpus_a2 import (
    A2_DEDUP_SEED,
    A2_MINHASH_BINDING,
    MINHASH_RECALL_JACCARD_LEVELS,
    MinHashRecallAuditV3,
    MinHashSyntheticRecallCellV3,
)
from training.weft1_corpus_replay_a2 import DEDUP_LEDGER_IDENTITY_DOMAIN_V3

SCHEMA = "weft1_corpus_parent_replay_child_receipt_v3"

def canonical(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")

def sha256(value):
    return hashlib.sha256(value).hexdigest()

def row(root, relative, role):
    data = (root / relative).read_bytes()
    return {"bytes": len(data), "path": relative, "role": role,
            "sha256": sha256(data)}

mode = sys.argv[1]
run_id = os.environ["WEFT1_REPLAY_RUN_ID"]
root = Path(os.environ["WEFT1_REPLAY_OUTPUT_ROOT"])
receipt_path = Path(os.environ["WEFT1_REPLAY_RECEIPT_PATH"])
root.mkdir(parents=True, exist_ok=False)

probe = socket.socket()
try:
    probe.connect(("127.0.0.1", 9))
except RuntimeError:
    network_probe = "python_socket_connect_blocked"
else:
    raise SystemExit("network guard did not block the probe")
finally:
    probe.close()

content = b"stable corpus shard\n"
if mode == "divergent" and run_id == "replay-b":
    content = b"different second shard\n"
(root / "general").mkdir()
(root / "general" / "t-00000.jsonl.zst").write_bytes(content)
files = [row(root, "general/t-00000.jsonl.zst", "content")]

complete = mode != "incomplete"
if complete:
    (root / "evidence").mkdir()
    dolma_id = sha256(b"dolma-document")
    dolma_source_id = sha256(b"dolma-source-record")
    fineweb_exact_id = sha256(b"fineweb-exact-document")
    fineweb_exact_source_id = sha256(b"fineweb-exact-source-record")
    fineweb_keep_id = sha256(b"fineweb-keep-document")
    fineweb_keep_source_id = sha256(b"fineweb-keep-source-record")
    decisions = [
        {"action":"KEEP_CANONICAL","canonical_document_id":None,
         "canonical_source_record_id":None,"decision_ordinal":0,
         "document_id":dolma_id,"exact_jaccard_denominator":None,
         "exact_jaccard_numerator":None,"lsh_candidate_count":0,
         "normalized_byte_count":7,"retained_byte_count":7,
         "source":"dolma_web","source_order_ordinal":0,
         "stable_source_record_id":dolma_source_id},
        {"action":"DROP_EXACT","canonical_document_id":dolma_id,
         "canonical_source_record_id":dolma_source_id,"decision_ordinal":1,
         "document_id":fineweb_exact_id,"exact_jaccard_denominator":1,
         "exact_jaccard_numerator":1,"lsh_candidate_count":0,
         "normalized_byte_count":7,"retained_byte_count":7,
         "source":"fineweb_edu","source_order_ordinal":0,
         "stable_source_record_id":fineweb_exact_source_id},
        {"action":"KEEP_FINEWEB","canonical_document_id":None,
         "canonical_source_record_id":None,"decision_ordinal":2,
         "document_id":fineweb_keep_id,"exact_jaccard_denominator":None,
         "exact_jaccard_numerator":None,"lsh_candidate_count":0,
         "normalized_byte_count":11,"retained_byte_count":11,
         "source":"fineweb_edu","source_order_ordinal":1,
         "stable_source_record_id":fineweb_keep_source_id},
    ]
    decision_bytes = b"".join(canonical(item) for item in decisions)
    (root / "evidence" / "dedup-decisions.jsonl").write_bytes(decision_bytes)
    semantic = hashlib.sha256()
    semantic.update(len(DEDUP_LEDGER_IDENTITY_DOMAIN_V3).to_bytes(8, "big"))
    semantic.update(DEDUP_LEDGER_IDENTITY_DOMAIN_V3)
    for item in decisions:
        framed = canonical(item)
        semantic.update(len(framed).to_bytes(8, "big"))
        semantic.update(framed)

    selection = [
        {"action":"DROP_EXACT","dedup_action":"DROP_EXACT",
         "document_id":fineweb_exact_id,"phase":"INITIAL",
         "pool":"fineweb_edu","retained_byte_count":7,
         "source":"fineweb_edu"},
        {"action":"SELECT_FINEWEB_TOPUP","dedup_action":"KEEP_FINEWEB",
         "document_id":fineweb_keep_id,"phase":"TOPUP",
         "pool":"fineweb_edu","retained_byte_count":11,
         "source":"fineweb_edu"},
    ]
    (root / "evidence" / "selection-decisions.jsonl").write_bytes(
        b"".join(canonical(item) for item in selection)
    )

    audit = MinHashRecallAuditV3(
        seed=A2_DEDUP_SEED,
        synthetic_cells=tuple(
            MinHashSyntheticRecallCellV3(
                exact_jaccard=level,
                pair_count=100,
                candidate_count=95,
            )
            for level in MINHASH_RECALL_JACCARD_LEVELS
        ),
        real_sample_identity_sha256=sha256(b"real-recall-sample"),
        real_dolma_document_count=1,
        real_fineweb_document_count=2,
        real_exact_pairs_at_or_above_threshold=1,
        real_candidate_pairs_at_or_above_threshold=1,
    )
    recall_payload = {
        "real_candidate_pairs_at_or_above_threshold": 1,
        "real_dolma_document_count": 1,
        "real_exact_pairs_at_or_above_threshold": 1,
        "real_fineweb_document_count": 2,
        "real_sample_identity_sha256": audit.real_sample_identity_sha256,
        "seed": A2_DEDUP_SEED,
        "status": audit.status,
        "synthetic_cells": [
            {"candidate_count": cell.candidate_count,
             "exact_jaccard": {"denominator": cell.exact_jaccard.denominator,
                                "numerator": cell.exact_jaccard.numerator},
             "pair_count": cell.pair_count}
            for cell in audit.synthetic_cells
        ],
    }
    (root / "evidence" / "minhash-recall-audit.json").write_bytes(
        canonical(recall_payload)
    )
    for relative in (
        "evidence/dedup-decisions.jsonl",
        "evidence/minhash-recall-audit.json",
        "evidence/selection-decisions.jsonl",
    ):
        files.append(row(root, relative, "dedup_evidence"))
    evidence_rows = {item["path"]: item for item in files}
    dedup_metadata = {
        "binding_identity_sha256": A2_MINHASH_BINDING.receipt_sha256,
        "decision_count": 3,
        "decision_ledger_identity_sha256": semantic.hexdigest(),
        "decision_ledger_path": "evidence/dedup-decisions.jsonl",
        "decision_ledger_sha256": evidence_rows[
            "evidence/dedup-decisions.jsonl"
        ]["sha256"],
        "dropped_bytes": 7,
        "exact_match_rate": {"denominator": 2, "numerator": 1},
        "minhash_recall_audit_path": "evidence/minhash-recall-audit.json",
        "minhash_recall_audit_receipt_sha256": audit.receipt_sha256,
        "minhash_recall_audit_sha256": evidence_rows[
            "evidence/minhash-recall-audit.json"
        ]["sha256"],
        "near_match_rate": {"denominator": 1, "numerator": 0},
        "schema": "weft1_corpus_parent_dedup_evidence_v3",
        "selection_ledger_path": "evidence/selection-decisions.jsonl",
        "selection_ledger_sha256": evidence_rows[
            "evidence/selection-decisions.jsonl"
        ]["sha256"],
        "topup_bytes": 11,
    }
    if mode == "wrong-binding":
        dedup_metadata["binding_identity_sha256"] = sha256(b"wrong binding")
    if mode == "wrong-derived":
        dedup_metadata["decision_count"] = 4
else:
    dedup_metadata = None

files.sort(key=lambda item: item["path"])
receipt = {
    "content_metadata": {
        "fixture_sha256": sha256(b"fixture-input\n"),
        "shard_count": 1,
    },
    "dedup_evidence_complete": complete,
    "dedup_metadata": dedup_metadata,
    "files": files,
    "input_identity_sha256": os.environ["WEFT1_REPLAY_INPUT_IDENTITY_SHA256"],
    "network_disabled": True,
    "network_guard_active": os.environ.get("WEFT1_NETWORK_GUARD_ACTIVE") == "1",
    "network_guard_sha256": os.environ["WEFT1_NETWORK_GUARD_SHA256"],
    "network_probe": network_probe,
    "output_root": str(root) if mode != "root" else str(root.parent),
    "process_id": os.getpid() if mode != "pid" else os.getpid() + 1,
    "run_id": run_id,
    "schema": SCHEMA,
    "worker_compatibility_sha256": os.environ[
        "WEFT1_REPLAY_WORKER_COMPATIBILITY_SHA256"
    ],
}
if mode == "noncanonical":
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
else:
    receipt_path.write_bytes(canonical(receipt))
if mode == "tamper" and run_id == "replay-b":
    (root / "general" / "t-00000.jsonl.zst").write_bytes(b"post-receipt tamper\n")
if mode == "extra" and run_id == "replay-b":
    (root / "unclaimed.bin").write_bytes(b"not in inventory")
'''


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    fixture = tmp_path / "fixture.txt"
    fixture.write_bytes(b"fixture-input\n")
    worker = tmp_path / "worker.py"
    worker.write_text(_WORKER_SOURCE, encoding="utf-8", newline="\n")
    return fixture, worker


def _verify(tmp_path: Path, mode: str) -> ParentReplayVerificationV3:
    fixture, worker = _fixture(tmp_path)
    return verify_parent_replays_v3(
        python_executable=Path(sys.executable),
        worker_arguments=(str(worker), mode),
        first_output_root=tmp_path / "outputs" / "a",
        second_output_root=tmp_path / "outputs" / "b",
        input_files={"fixture": fixture},
        compatibility_files={"worker": worker},
        worker_cwd=ROOT,
        timeout_seconds=20,
    )


def test_parent_recomputes_complete_d1_d2_but_socket_guard_cannot_mint_pass(
    tmp_path: Path,
) -> None:
    receipt = _verify(tmp_path, "happy")
    assert receipt.schema == PARENT_RECEIPT_SCHEMA_V3
    assert receipt.status == "CHECK_PASS"
    assert receipt.authoritative is False
    assert receipt.d1_file_replay_verified is True
    assert receipt.d2_dedup_replay_verified is True
    assert receipt.network_isolation_kind == "python_socket_guard_only"
    assert receipt.network_isolation_authoritative is False
    assert receipt.network_isolation_executable_sha256 is None
    assert receipt.first_process_id != receipt.second_process_id
    assert Path(receipt.first_output_root).is_dir()
    assert Path(receipt.second_output_root).is_dir()
    assert len(receipt.output_file_projection_sha256) == 64
    assert len(receipt.content_projection_sha256) == 64
    assert len(receipt.dedup_projection_sha256 or "") == 64
    assert len(receipt.receipt_sha256) == 64
    assert receipt.first_child_receipt_sha256 != (
        receipt.second_child_receipt_sha256
    )
    assert hashlib.sha256(
        (Path(receipt.first_output_root) / "general/t-00000.jsonl.zst").read_bytes()
    ).hexdigest() == hashlib.sha256(b"stable corpus shard\n").hexdigest()
    with pytest.raises(TypeError, match="factory-minted"):
        ParentReplayVerificationV3()


def test_missing_full_dedup_evidence_stays_non_gate_even_when_d1_matches(
    tmp_path: Path,
) -> None:
    receipt = _verify(tmp_path, "incomplete")
    assert receipt.status == "CHECK_PASS"
    assert receipt.authoritative is False
    assert receipt.d1_file_replay_verified is True
    assert receipt.d2_dedup_replay_verified is False
    assert receipt.dedup_projection_sha256 is None


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("tamper", "parent rehash differs"),
        ("pid", "Popen PID"),
        ("root", "output root differs"),
        ("divergent", "D1 failed"),
        ("noncanonical", "not canonical"),
        ("extra", "does not cover every output file"),
        ("wrong-binding", "wrong A2 MinHash binding"),
        ("wrong-derived", "parent-recomputed evidence"),
    ],
)
def test_parent_fails_closed_on_child_or_output_tampering(
    tmp_path: Path, mode: str, message: str
) -> None:
    with pytest.raises(ParentReplayError, match=message):
        _verify(tmp_path, mode)


def test_roots_must_be_fresh_resolved_and_non_overlapping_before_launch(
    tmp_path: Path,
) -> None:
    fixture, worker = _fixture(tmp_path)
    common = {
        "python_executable": Path(sys.executable),
        "worker_arguments": (str(worker), "happy"),
        "input_files": {"fixture": fixture},
        "compatibility_files": {"worker": worker},
        "worker_cwd": ROOT,
        "timeout_seconds": 20,
    }
    with pytest.raises(ParentReplayError, match="non-overlapping"):
        verify_parent_replays_v3(
            **common,
            first_output_root=tmp_path / "nested",
            second_output_root=tmp_path / "nested" / "child",
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ParentReplayError, match="fresh and absent"):
        verify_parent_replays_v3(
            **common,
            first_output_root=existing,
            second_output_root=tmp_path / "other",
        )


def test_extra_environment_cannot_restore_network_proxy_variables(
    tmp_path: Path,
) -> None:
    fixture, worker = _fixture(tmp_path)
    with pytest.raises(ParentReplayError, match="may not override HTTPS_PROXY"):
        verify_parent_replays_v3(
            python_executable=Path(sys.executable),
            worker_arguments=(str(worker), "happy"),
            first_output_root=tmp_path / "a",
            second_output_root=tmp_path / "b",
            input_files={"fixture": fixture},
            compatibility_files={"worker": worker},
            worker_cwd=ROOT,
            timeout_seconds=20,
            extra_environment={"HTTPS_PROXY": "http://127.0.0.1:9"},
        )


def test_reserved_environment_and_guard_bypass_flags_are_rejected(
    tmp_path: Path,
) -> None:
    fixture, worker = _fixture(tmp_path)
    common = {
        "python_executable": Path(sys.executable),
        "first_output_root": tmp_path / "outputs" / "a",
        "second_output_root": tmp_path / "outputs" / "b",
        "input_files": {"fixture": fixture},
        "compatibility_files": {"worker": worker},
        "worker_cwd": ROOT,
        "timeout_seconds": 20,
    }
    with pytest.raises(ParentReplayError, match="bypass"):
        verify_parent_replays_v3(
            **common,
            worker_arguments=("-S", str(worker), "happy"),
        )
    with pytest.raises(ParentReplayError, match="may not override"):
        verify_parent_replays_v3(
            **common,
            worker_arguments=(str(worker), "happy"),
            extra_environment={"WEFT1_REPLAY_RUN_ID": "forged"},
        )


def test_public_child_schema_constant_is_exact() -> None:
    assert CHILD_RECEIPT_SCHEMA_V3 == (
        "weft1_corpus_parent_replay_child_receipt_v3"
    )


def test_production_child_metadata_composes_with_parent_rehashed_manifests(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "production-child"
    output_root.mkdir()
    source_identity = hashlib.sha256(b"source").hexdigest()
    tokenizer_identity = hashlib.sha256(b"tokenizer-fit").hexdigest()
    environment_identity = hashlib.sha256(b"environment").hexdigest()
    content = {
        "mode": "PRODUCTION",
        "readiness": replay.PRODUCTION_READINESS_V3,
        "schema": replay.PRODUCTION_MATERIALIZER_SCHEMA_V3,
        "source_identity_sha256": source_identity,
        "tokenizer_fit_input_receipt_sha256": tokenizer_identity,
    }
    content_identity = corpus_contracts.execution_authority_v3_bound_sha256(
        "weft1_corpus_materialized_content_v3", content
    )
    content["content_identity_sha256"] = content_identity
    content_path = output_root / "content-manifest.json"
    content_path.write_bytes(replay._canonical_json_line(content))
    content_row = {
        "bytes": content_path.stat().st_size,
        "path": "content-manifest.json",
        "role": "content",
        "sha256": hashlib.sha256(content_path.read_bytes()).hexdigest(),
    }
    d1 = {
        "content_identity_sha256": content_identity,
        "file_inventory": [
            {
                "bytes": content_row["bytes"],
                "relative_path": content_row["path"],
                "sha256": content_row["sha256"],
            }
        ],
        "gate_minted": False,
        "mode": "PRODUCTION",
        "readiness": replay.PRODUCTION_READINESS_V3,
        "schema": replay.PRODUCTION_D1_READY_SCHEMA_V3,
        "source_identity_sha256": source_identity,
    }
    d1["d1_ready_identity_sha256"] = (
        corpus_contracts.execution_authority_v3_bound_sha256(
            "weft1_corpus_d1_ready_inventory_v3", d1
        )
    )
    d1_path = output_root / "d1-ready-manifest.json"
    d1_path.write_bytes(replay._canonical_json_line(d1))
    d1_sha256 = hashlib.sha256(d1_path.read_bytes()).hexdigest()
    d1_row = {
        "bytes": d1_path.stat().st_size,
        "path": "d1-ready-manifest.json",
        "role": "content",
        "sha256": d1_sha256,
    }
    metadata = {
        "content_identity_sha256": content_identity,
        "d1_ready_manifest_sha256": d1_sha256,
        "environment_identity_sha256": environment_identity,
        "materializer_algorithm_version": (
            replay.PRODUCTION_MATERIALIZER_ALGORITHM_VERSION_V3
        ),
        "source_identity_sha256": source_identity,
        "tokenizer_fit_input_receipt_sha256": tokenizer_identity,
    }
    child = replay._VerifiedChildReplayV3(
        run_id="production-child",
        actual_process_id=1,
        output_root=str(output_root),
        child_receipt_sha256="1" * 64,
        stdout_sha256="2" * 64,
        stderr_sha256="3" * 64,
        output_file_rows=(content_row, d1_row),
        output_file_projection_sha256="4" * 64,
        content_projection_sha256="5" * 64,
        dedup_projection_sha256="6" * 64,
        dedup_evidence_complete=True,
        content_metadata=metadata,
    )
    replay._validate_production_child_profile_v3(
        child,
        expected_environment_identity_sha256=environment_identity,
    )
    with pytest.raises(ParentReplayError, match="runtime identity"):
        replay._validate_production_child_profile_v3(
            child,
            expected_environment_identity_sha256="f" * 64,
        )


def test_only_fixed_production_wrapper_can_assert_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "production_profile_verified" not in inspect.signature(
        verify_parent_replays_v3
    ).parameters
    files = {
        name: tmp_path / name
        for name in (
            "authority.md",
            "bindings.json",
            "download.json",
            "enumeration.json",
            "lock.txt",
            "model.bin",
            "routes.json",
            "source-manifest.json",
            "worker.py",
        )
    }
    payloads = {
        "authority.md": b"authority\n",
        "bindings.json": b"bindings\n",
        "download.json": b"{}\n",
        "enumeration.json": b"{}\n",
        "lock.txt": b"lock\n",
        "model.bin": b"model-bytes",
        "routes.json": b"routes\n",
        "source-manifest.json": b"{}\n",
        "worker.py": b"raise SystemExit(0)\n",
    }
    for name, path in files.items():
        path.write_bytes(payloads[name])
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    def sha(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    monkeypatch.setattr(replay, "PRODUCTION_BINDINGS_PATH_V3", files["bindings.json"])
    monkeypatch.setattr(
        replay,
        "PRODUCTION_BINDINGS_SHA256_V3",
        sha(payloads["bindings.json"]),
    )
    monkeypatch.setattr(replay, "PRODUCTION_DEPENDENCY_LOCK_PATH_V3", files["lock.txt"])
    monkeypatch.setattr(replay, "PRODUCTION_WORKER_PATH_V3", files["worker.py"])
    monkeypatch.setattr(
        replay, "PRODUCTION_AUTHORITY_SHA256_V3", sha(payloads["authority.md"])
    )
    monkeypatch.setattr(
        production_io,
        "DEFAULT_REQUIREMENTS_LOCK_SHA256",
        sha(payloads["lock.txt"]),
    )
    monkeypatch.setattr(source_routes, "SOURCE_ROUTE_MANIFEST_PATH", files["routes.json"])
    monkeypatch.setattr(
        source_routes,
        "SOURCE_ROUTE_MANIFEST_SHA256",
        sha(payloads["routes.json"]),
    )
    monkeypatch.setattr(
        corpus_contracts,
        "A2_LANGUAGE_ID_BINDING",
        SimpleNamespace(
            model_bytes=len(payloads["model.bin"]),
            model_sha256=sha(payloads["model.bin"]),
        ),
    )
    marker = object()
    captured: dict[str, object] = {}
    captured_snapshot_bytes: dict[str, bytes] = {}

    def fake_impl(**kwargs: object) -> object:
        captured.update(kwargs)
        input_files = kwargs["input_files"]
        assert isinstance(input_files, dict)
        captured_snapshot_bytes.update(
            {
                name: Path(path).read_bytes()
                for name, path in input_files.items()
            }
        )
        return marker

    monkeypatch.setattr(replay, "_verify_parent_replays_v3_impl", fake_impl)
    result = verify_production_materialization_replays_v3(
        python_executable=Path(sys.executable),
        authority_path=files["authority.md"],
        enumeration_receipt_path=files["enumeration.json"],
        cache_download_receipt_path=files["download.json"],
        source_manifest_path=files["source-manifest.json"],
        cache_root=cache_root,
        fasttext_model_path=files["model.bin"],
        first_output_root=tmp_path / "first",
        second_output_root=tmp_path / "second",
    )
    assert result is marker
    assert (
        captured["production_profile_sentinel"]
        is replay._PRODUCTION_PROFILE_SENTINEL
    )
    assert captured["network_namespace_executable"] == replay.LINUX_UNSHARE_PATH_V1
    arguments = captured["worker_arguments"]
    assert isinstance(arguments, tuple)
    assert arguments[0] == str(files["worker.py"].resolve())
    assert "--cache-root" in arguments
    assert captured["extra_environment"] is None
    assert captured_snapshot_bytes["a2_authority"] == payloads["authority.md"]
    assert captured_snapshot_bytes["enumeration_receipt"] == payloads["enumeration.json"]
    assert captured_snapshot_bytes["cache_download_receipt"] == payloads["download.json"]
    assert captured_snapshot_bytes["source_manifest"] == payloads["source-manifest.json"]
    assert captured_snapshot_bytes["fasttext_model"] == payloads["model.bin"]


def test_internal_reducer_rejects_arbitrary_production_authority_token(
    tmp_path: Path,
) -> None:
    with pytest.raises(ParentReplayError, match="authority token"):
        replay._verify_parent_replays_v3_impl(
            python_executable=Path(sys.executable),
            worker_arguments=("unused.py",),
            first_output_root=tmp_path / "first",
            second_output_root=tmp_path / "second",
            input_files={},
            compatibility_files={},
            worker_cwd=ROOT,
            production_profile_sentinel=object(),
        )
