from __future__ import annotations

import hashlib
import inspect
import json
import os
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


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _runtime_build_receipt() -> dict[str, object]:
    return {
        "authoritative": True,
        "evidence": {},
        "receipt_identity_sha256": _sha_text("runtime-receipt-identity"),
        "schema": replay.RUNTIME_BUILD_RECEIPT_SCHEMA_V1,
        "status": "PASS",
    }


def _storage_identity() -> dict[str, object]:
    durable_mount = {
        "filesystem_type": "fuse.drive",
        "major_minor": "0:99",
        "mount_id": 99,
        "mount_point": str(ROOT),
        "mount_root": "/",
        "mount_source": "drive",
        "parent_mount_id": 1,
        "st_dev": 99,
    }
    local_mount = {
        "filesystem_type": "overlay",
        "major_minor": "0:98",
        "mount_id": 98,
        "mount_point": str(ROOT.parent),
        "mount_root": "/",
        "mount_source": "overlay",
        "parent_mount_id": 1,
        "st_dev": 98,
    }
    core: dict[str, object] = {
        "durable_marker_sha256": _sha_text("marker"),
        "durable_mount": durable_mount,
        "durable_mount_root": str(ROOT),
        "durable_storage_root": str(ROOT),
        "local_mount": local_mount,
        "provider": "google_colab_drive_v1",
        "schema": replay.PRODUCTION_STORAGE_IDENTITY_SCHEMA_V3,
    }
    core["storage_identity_sha256"] = (
        corpus_contracts.execution_authority_v3_bound_sha256(
            replay.PRODUCTION_STORAGE_IDENTITY_SCHEMA_V3, core
        )
    )
    return core


def _installed_inventory(prefix: Path, *, record_sha256: str) -> dict[str, object]:
    record_path = "lib/python3.11/site-packages/alpha-1.0.dist-info/RECORD"
    normalized: dict[str, object] = {
        "bootstrap_distributions": [],
        "distributions": [
            {
                "distribution": "alpha",
                "file_count": 1,
                "record_path": record_path,
                "record_sha256": record_sha256,
                "source": "hash_locked_wheel",
                "version": "1.0",
            }
        ],
        "files": [
            {
                "bytes": 1,
                "owners": ["alpha"],
                "relative_path": record_path,
                "sha256": record_sha256,
            }
        ],
        "installation_prefix": str(prefix),
        "schema": production_io.INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA_V3,
        "site_roots": ["lib/python3.11/site-packages"],
    }
    normalized["inventory_identity_sha256"] = (
        production_io._installed_inventory_identity_sha256_v3(normalized)
    )
    return normalized


def _write_storage_marker(path: Path, storage_root: Path) -> None:
    core: dict[str, object] = {
        "durable_storage_root": str(storage_root.resolve()),
        "filesystem_type_prefix": "fuse.drive",
        "mount_source": "drive",
        "provider": "google_colab_drive_v1",
        "schema": "weft1_durable_storage_marker_v3",
    }
    core["marker_identity_sha256"] = (
        corpus_contracts.execution_authority_v3_bound_sha256(
            "weft1_durable_storage_marker_v3", core
        )
    )
    path.write_bytes(replay._canonical_json_line(core))


def _global_provenance() -> dict[str, object]:
    lock_sha256 = _sha_text("dependency-lock")
    executable_sha256 = _sha_text("python-executable")
    wheel_sha256 = _sha_text("alpha-wheel")
    linkage_core = {
        "executable": {
            "bytes": 1,
            "path": str((ROOT / "python3.11").resolve()),
            "sha256": executable_sha256,
        },
        "libpython_library": {
            "bytes": 1,
            "path": str((ROOT / "libpython3.11.so.1.0").resolve()),
            "sha256": _sha_text("libpython"),
        },
        "schema": production_io.RUNTIME_LINKAGE_SCHEMA_V3,
        "sqlite_extension": {
            "bytes": 1,
            "path": str((ROOT / "_sqlite3.so").resolve()),
            "sha256": _sha_text("sqlite-extension"),
        },
        "sqlite_library": {
            "bytes": 1,
            "path": str((ROOT / "libsqlite3.so.0.8.6").resolve()),
            "sha256": _sha_text("sqlite-library"),
        },
    }
    environment = {
        "dependency_lock_sha256": lock_sha256,
        "distributions": [
            {
                "artifact_sha256s": [wheel_sha256],
                "distribution": "alpha",
                "version": "1.0",
            }
        ],
        "python_executable_sha256": executable_sha256,
        "runtime_linkage": {
            **linkage_core,
            "linkage_identity_sha256": corpus_contracts.execution_authority_v3_bound_sha256(
                production_io.RUNTIME_LINKAGE_SCHEMA_V3, linkage_core
            ),
        },
        "runtime_versions": {
            "libzstd_version": "1.5.7",
            "python_version": "3.11.9",
            "sqlite_source_id": "sqlite-source-id",
            "sqlite_version": "3.45.1",
            "unicode_data_version": "14.0.0",
            "zstandard_package_version": "0.25.0",
        },
    }
    return replay._build_global_execution_provenance_v3(
        environment_payload=environment,
        environment_identity_sha256=(
            corpus_contracts.execution_authority_v3_bound_sha256(
                "weft1_corpus_execution_environment_v3", environment
            )
        ),
        python_executable_sha256=executable_sha256,
        dependency_lock_sha256=lock_sha256,
        pipeline_components=(
            {
                "bytes": 1,
                "logical_name": "materializer",
                "sha256": _sha_text("code"),
            },
        ),
        runtime_build_receipt_identity_sha256=_sha_text(
            "runtime-receipt-identity"
        ),
        runtime_build_receipt_sha256=hashlib.sha256(
            replay._canonical_json_line(_runtime_build_receipt())
        ).hexdigest(),
        selected_wheels=(
            {
                "bytes": 1,
                "filename": "alpha-1.0-py3-none-any.whl",
                "sha256": wheel_sha256,
            },
        ),
        production_storage_identity=_storage_identity(),
    )


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


def test_production_offline_environment_routes_all_temp_to_local_work(
    tmp_path: Path,
) -> None:
    local_work = tmp_path / "local-work"
    local_work.mkdir()
    environment = replay._offline_environment(
        guard_directory=tmp_path,
        guard_sha256="1" * 64,
        run_id="temp-routing",
        output_root=tmp_path / "durable-output",
        local_work_parent=local_work,
        input_identity_sha256="2" * 64,
        worker_compatibility_sha256="3" * 64,
        worker_import_root=tmp_path,
        extra_environment=None,
    )
    assert environment["WEFT1_REPLAY_LOCAL_WORK_PARENT"] == str(local_work)
    assert {
        environment[key] for key in ("SQLITE_TMPDIR", "TEMP", "TMP", "TMPDIR")
    } == {str(local_work)}
    assert environment["PYTHONPATH"] == str(tmp_path.resolve())
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "PYTHONUSERBASE" not in environment


def test_production_offline_environment_binds_parsed_asset_recovery_as_one_set(
    tmp_path: Path,
) -> None:
    local_work = tmp_path / "local-work"
    local_work.mkdir()
    parsed_cache = tmp_path / "parsed-cache"
    parsed_cache.mkdir()
    common = {
        "guard_directory": tmp_path,
        "guard_sha256": "1" * 64,
        "run_id": "production-v4-replay-a",
        "output_root": tmp_path / "durable-output",
        "local_work_parent": local_work,
        "input_identity_sha256": "2" * 64,
        "worker_compatibility_sha256": "3" * 64,
        "worker_import_root": tmp_path,
        "extra_environment": None,
    }
    environment = replay._offline_environment(
        **common,
        parsed_asset_cache_root=parsed_cache,
        parsed_asset_code_identity_sha256="4" * 64,
        parsed_asset_durable_marker_sha256="5" * 64,
        parsed_asset_input_identity_sha256="6" * 64,
    )
    assert environment["WEFT1_REPLAY_PARSED_ASSET_CACHE_ROOT"] == str(
        parsed_cache.resolve(strict=True)
    )
    assert environment["WEFT1_REPLAY_PARSED_ASSET_CODE_IDENTITY_SHA256"] == (
        "4" * 64
    )
    assert environment["WEFT1_REPLAY_PARSED_ASSET_DURABLE_MARKER_SHA256"] == (
        "5" * 64
    )
    assert environment["WEFT1_REPLAY_PARSED_ASSET_INPUT_IDENTITY_SHA256"] == (
        "6" * 64
    )
    with pytest.raises(ParentReplayError, match="one complete set"):
        replay._offline_environment(
            **common,
            parsed_asset_cache_root=parsed_cache,
        )


def test_isolated_worker_ignores_hostile_pythonpath_and_usercustomize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, worker = _fixture(tmp_path)
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    marker = tmp_path / "usercustomize-ran"
    (hostile / "usercustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(hostile))
    monkeypatch.setenv("PYTHONUSERBASE", str(hostile))
    result = verify_parent_replays_v3(
        python_executable=Path(sys.executable),
        worker_arguments=(str(worker), "happy"),
        first_output_root=tmp_path / "outputs" / "a",
        second_output_root=tmp_path / "outputs" / "b",
        input_files={"fixture": fixture},
        compatibility_files={"worker": worker},
        worker_cwd=ROOT,
        timeout_seconds=20,
    )
    assert result.status == "CHECK_PASS"
    assert not marker.exists()


def test_reserved_environment_names_are_case_insensitive(tmp_path: Path) -> None:
    fixture, worker = _fixture(tmp_path)
    with pytest.raises(ParentReplayError, match="may not override"):
        verify_parent_replays_v3(
            python_executable=Path(sys.executable),
            worker_arguments=(str(worker), "happy"),
            first_output_root=tmp_path / "outputs" / "a",
            second_output_root=tmp_path / "outputs" / "b",
            input_files={"fixture": fixture},
            compatibility_files={"worker": worker},
            worker_cwd=ROOT,
            timeout_seconds=20,
            extra_environment={"pythonpath": str(tmp_path)},
        )

    with pytest.raises(ParentReplayError, match="may not override"):
        verify_parent_replays_v3(
            python_executable=Path(sys.executable),
            worker_arguments=(str(worker), "happy"),
            first_output_root=tmp_path / "outputs" / "hashseed-a",
            second_output_root=tmp_path / "outputs" / "hashseed-b",
            input_files={"fixture": fixture},
            compatibility_files={"worker": worker},
            worker_cwd=ROOT,
            timeout_seconds=20,
            extra_environment={"pythonhashseed": "random"},
        )


def test_public_child_schema_constant_is_exact() -> None:
    assert CHILD_RECEIPT_SCHEMA_V3 == (
        "weft1_corpus_parent_replay_child_receipt_v3"
    )


def test_runtime_build_receipt_binds_exact_runtime_lock_and_selected_wheels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import build_weft1_pa_runtime as runtime_builder

    monkeypatch.setattr(
        runtime_builder,
        "verify_build_receipt_payload",
        lambda receipt: str(receipt["receipt_identity_sha256"]),
    )
    prefix = tmp_path / "runtime"
    executable = prefix / "bin" / "python3.11"
    sqlite_extension = prefix / "lib" / "_sqlite3.so"
    libpython_library = prefix / "lib" / "libpython3.11.so.1.0"
    sqlite_library = prefix / "lib" / "libsqlite3.so.0.8.6"
    executable.parent.mkdir(parents=True)
    sqlite_extension.parent.mkdir(parents=True)
    executable.write_bytes(b"python-runtime")
    sqlite_extension.write_bytes(b"sqlite-extension")
    libpython_library.write_bytes(b"libpython")
    sqlite_library.write_bytes(b"sqlite-library")
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    sqlite_extension_sha256 = hashlib.sha256(
        sqlite_extension.read_bytes()
    ).hexdigest()
    libpython_sha256 = hashlib.sha256(libpython_library.read_bytes()).hexdigest()
    sqlite_library_sha256 = hashlib.sha256(sqlite_library.read_bytes()).hexdigest()
    lock_sha256 = _sha_text("dependency-lock")
    wheel_sha256 = _sha_text("alpha-wheel")
    builder_sha256 = _sha_text("runtime-builder")
    runtime_contract_sha256 = _sha_text("runtime-contract")
    runtime_versions = {
        "libzstd_version": "1.5.7",
        "python_version": "3.11.9",
        "sqlite_source_id": "sqlite-source-id",
        "sqlite_version": "3.45.1",
        "unicode_data_version": "14.0.0",
        "zstandard_package_version": "0.25.0",
    }
    environment = {
        "dependency_lock_sha256": lock_sha256,
        "distributions": [
            {
                "artifact_sha256s": [wheel_sha256],
                "distribution": "alpha",
                "version": "1.0",
            }
        ],
        "python_executable_sha256": executable_sha256,
        "runtime_versions": runtime_versions,
    }
    installed_inventory = _installed_inventory(
        prefix, record_sha256=_sha_text("alpha-record")
    )
    environment["installed_distribution_inventory"] = installed_inventory
    linkage_core = {
        "executable": {
            "bytes": executable.stat().st_size,
            "path": str(executable.resolve()),
            "sha256": executable_sha256,
        },
        "libpython_library": {
            "bytes": libpython_library.stat().st_size,
            "path": str(libpython_library.resolve()),
            "sha256": libpython_sha256,
        },
        "schema": production_io.RUNTIME_LINKAGE_SCHEMA_V3,
        "sqlite_extension": {
            "bytes": sqlite_extension.stat().st_size,
            "path": str(sqlite_extension.resolve()),
            "sha256": sqlite_extension_sha256,
        },
        "sqlite_library": {
            "bytes": sqlite_library.stat().st_size,
            "path": str(sqlite_library.resolve()),
            "sha256": sqlite_library_sha256,
        },
    }
    environment["runtime_linkage"] = {
        **linkage_core,
        "linkage_identity_sha256": corpus_contracts.execution_authority_v3_bound_sha256(
            production_io.RUNTIME_LINKAGE_SCHEMA_V3, linkage_core
        ),
    }
    environment_identity = corpus_contracts.execution_authority_v3_bound_sha256(
        "weft1_corpus_execution_environment_v3", environment
    )
    runtime = SimpleNamespace(
        dependency_lock_sha256=lock_sha256,
        environment_identity_sha256=environment_identity,
        environment_payload=environment,
        executable_sha256=executable_sha256,
    )
    artifacts = {
        "build_log_sha256": _sha_text("build-log"),
        "cpython_executable_sha256": executable_sha256,
        "libpython_library_sha256": libpython_sha256,
        "sqlite3_extension_sha256": sqlite_extension_sha256,
        "sqlite3_library_sha256": sqlite_library_sha256,
    }
    probe = {
        "cpython_executable_ldd": [
            f"libpython3.11.so.1.0 => {libpython_library}"
        ],
        "cpython_executable_search_paths": [str(prefix / "lib")],
        "extra_distributions": [],
        "libpython_library_path": str(libpython_library.resolve()),
        "libpython_library_sha256": libpython_sha256,
        "libzstd_version": runtime_versions["libzstd_version"],
        "locked_distributions": [["alpha", "1.0"]],
        "missing_distributions": [],
        "python_version": runtime_versions["python_version"],
        "python_prefix": str(prefix),
        "sqlite3_extension_search_paths": [str(prefix / "lib")],
        "sqlite_extension_ldd": ["libsqlite3.so.0 => pinned/libsqlite3.so.0"],
        "sqlite_extension_path": str(sqlite_extension),
        "sqlite_extension_sha256": sqlite_extension_sha256,
        "sqlite_library_path": str(sqlite_library.resolve()),
        "sqlite_library_sha256": sqlite_library_sha256,
        "sqlite_source_id": runtime_versions["sqlite_source_id"],
        "sqlite_version": runtime_versions["sqlite_version"],
        "unicode_data_version": runtime_versions["unicode_data_version"],
        "wrong_distributions": [],
        "zstandard_package_version": runtime_versions[
            "zstandard_package_version"
        ],
    }
    repository_attestation = {
        "dependency_lock_sha256": lock_sha256,
        "environment_identity_sha256": environment_identity,
        "environment_payload": environment,
        "executable_sha256": executable_sha256,
    }
    evidence = {
        "artifacts": artifacts,
        "build_dependency_versions": {},
        "builder_sha256": builder_sha256,
        "cpython_site_packages_readme_removal": {},
        "host": {},
        "installed_distribution_inventory": installed_inventory,
        "jobs": 1,
        "locked_distributions": [["alpha", "1.0"]],
        "prefix": str(prefix),
        "recipe": {},
        "recipe_identity_sha256": _sha_text("recipe"),
        "requirements_lock": {
            "bytes": 1,
            "filename": "requirements.lock",
            "sha256": lock_sha256,
            "source_filename": "requirements.lock",
        },
        "repository_runtime_attestation": repository_attestation,
        "runtime_contract_sha256": runtime_contract_sha256,
        "runtime_probe": probe,
        "sources": [],
        "trusted_installer_chain": {},
        "wheelhouse": [
            {
                "bytes": 17,
                "filename": "alpha-1.0-py3-none-any.whl",
                "sha256": wheel_sha256,
            }
        ],
        "wheelhouse_identity_sha256": "",
    }
    evidence["recipe_identity_sha256"] = hashlib.sha256(
        replay._canonical_json_line(evidence["recipe"])
    ).hexdigest()
    evidence["wheelhouse_identity_sha256"] = hashlib.sha256(
        replay._canonical_json_line(evidence["wheelhouse"])
    ).hexdigest()

    def receipt_for(current_evidence: dict[str, object]) -> dict[str, object]:
        return {
            "authoritative": True,
            "evidence": current_evidence,
            "receipt_identity_sha256": hashlib.sha256(
                replay._canonical_json_line(
                    {
                        "domain": replay.RUNTIME_BUILD_RECEIPT_SCHEMA_V1,
                        "evidence": current_evidence,
                    }
                )
            ).hexdigest(),
            "schema": replay.RUNTIME_BUILD_RECEIPT_SCHEMA_V1,
            "status": "PASS",
        }

    receipt_path = tmp_path / "runtime-receipt.json"
    receipt_path.write_bytes(replay._canonical_json_line(receipt_for(evidence)))
    validated, physical_sha256 = replay._load_runtime_build_receipt_v1(
        receipt_path,
        runtime_attestation=runtime,
        expected_builder_sha256=builder_sha256,
        expected_runtime_contract_sha256=runtime_contract_sha256,
        expected_python_executable=executable,
    )
    assert validated["selected_wheels"] == evidence["wheelhouse"]
    assert physical_sha256 == hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    tampered_evidence = json.loads(json.dumps(evidence))
    tampered_evidence["wheelhouse"][0]["sha256"] = _sha_text("unlocked-wheel")
    tampered_path = tmp_path / "tampered-runtime-receipt.json"
    tampered_path.write_bytes(
        replay._canonical_json_line(receipt_for(tampered_evidence))
    )
    with pytest.raises(ParentReplayError, match="hash-locked distribution closure"):
        replay._load_runtime_build_receipt_v1(
            tampered_path,
            runtime_attestation=runtime,
            expected_builder_sha256=builder_sha256,
            expected_runtime_contract_sha256=runtime_contract_sha256,
            expected_python_executable=executable,
        )


def test_global_execution_provenance_recomputes_pipeline_and_receipt_bindings() -> None:
    provenance = _global_provenance()
    assert replay.validate_global_execution_provenance_v3(provenance) == provenance
    tampered = json.loads(json.dumps(provenance))
    tampered["pipeline_components"][0]["sha256"] = _sha_text("tampered-code")
    with pytest.raises(ParentReplayError, match="pipeline code identity"):
        replay.validate_global_execution_provenance_v3(tampered)


def test_production_child_metadata_composes_with_parent_rehashed_manifests(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "production-child"
    output_root.mkdir()
    source_identity = hashlib.sha256(b"source").hexdigest()
    tokenizer_identity = hashlib.sha256(b"tokenizer-fit").hexdigest()
    provenance = _global_provenance()
    environment_identity = str(provenance["environment_identity_sha256"])
    provenance_path = (
        output_root / replay.GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3
    )
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_bytes(replay._canonical_json_line(provenance))
    provenance_row = {
        "bytes": provenance_path.stat().st_size,
        "path": replay.GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3,
        "role": "content",
        "sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
    }
    runtime_receipt_path = output_root / replay.RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1
    runtime_receipt_path.write_bytes(
        replay._canonical_json_line(_runtime_build_receipt())
    )
    runtime_receipt_row = {
        "bytes": runtime_receipt_path.stat().st_size,
        "path": replay.RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1,
        "role": "content",
        "sha256": hashlib.sha256(runtime_receipt_path.read_bytes()).hexdigest(),
    }
    content = {
        "global": {
            "execution_provenance": provenance,
            "execution_provenance_path": (
                replay.GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3
            ),
            "execution_provenance_sha256": provenance_row["sha256"],
            "runtime_build_receipt_path": replay.RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1,
            "runtime_build_receipt_sha256": runtime_receipt_row["sha256"],
        },
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
                "bytes": provenance_row["bytes"],
                "relative_path": provenance_row["path"],
                "sha256": provenance_row["sha256"],
            },
            {
                "bytes": runtime_receipt_row["bytes"],
                "relative_path": runtime_receipt_row["path"],
                "sha256": runtime_receipt_row["sha256"],
            },
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
        "global_execution_provenance_identity_sha256": provenance[
            "provenance_identity_sha256"
        ],
        "global_execution_provenance_sha256": provenance_row["sha256"],
        "materializer_algorithm_version": (
            replay.PRODUCTION_MATERIALIZER_ALGORITHM_VERSION_V3
        ),
        "pipeline_code_identity_sha256": provenance[
            "pipeline_code_identity_sha256"
        ],
        "runtime_build_receipt_identity_sha256": provenance[
            "runtime_build_receipt_identity_sha256"
        ],
        "runtime_build_receipt_sha256": runtime_receipt_row["sha256"],
        "source_identity_sha256": source_identity,
        "tokenizer_fit_input_receipt_sha256": tokenizer_identity,
    }
    child = replay._VerifiedChildReplayV3(
        run_id="production-child",
        actual_process_id=1,
        output_root=str(output_root),
        child_receipt_bytes=1,
        child_receipt_sha256="1" * 64,
        stdout_sha256="2" * 64,
        stderr_sha256="3" * 64,
        output_file_rows=(provenance_row, runtime_receipt_row, content_row, d1_row),
        output_file_projection_sha256="4" * 64,
        content_projection_sha256="5" * 64,
        dedup_projection_sha256="6" * 64,
        dedup_evidence_complete=True,
        content_metadata=metadata,
    )
    replay._validate_production_child_profile_v3(
        child,
        expected_environment_identity_sha256=environment_identity,
        expected_global_execution_provenance=provenance,
    )
    with pytest.raises(ParentReplayError, match="runtime identity"):
        replay._validate_production_child_profile_v3(
            child,
            expected_environment_identity_sha256="f" * 64,
            expected_global_execution_provenance=provenance,
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
            "runtime-receipt.json",
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
        "runtime-receipt.json": b"{}\n",
        "source-manifest.json": b"{}\n",
        "worker.py": b"raise SystemExit(0)\n",
    }
    for name, path in files.items():
        path.write_bytes(payloads[name])
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    durable_output_parent = tmp_path / "durable"
    durable_output_parent.mkdir()
    local_work_parent = tmp_path / "local-work"
    local_work_parent.mkdir()
    durable_mount_root = tmp_path / "mount"
    durable_mount_root.mkdir()
    durable_storage_marker = durable_mount_root / "storage-marker.json"
    durable_storage_marker.write_text("{}\n", encoding="utf-8")

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
    provenance_template = _global_provenance()
    runtime_environment = dict(provenance_template["environment_payload"])
    runtime_environment["dependency_lock_sha256"] = sha(payloads["lock.txt"])
    runtime_attestation = SimpleNamespace(
        dependency_lock_sha256=sha(payloads["lock.txt"]),
        environment_identity_sha256=(
            corpus_contracts.execution_authority_v3_bound_sha256(
                "weft1_corpus_execution_environment_v3", runtime_environment
            )
        ),
        environment_payload=runtime_environment,
        executable_sha256=provenance_template["python_executable_sha256"],
    )
    monkeypatch.setattr(
        production_io,
        "attest_runtime_v3",
        lambda **_kwargs: runtime_attestation,
    )
    runtime_receipt_identity = _sha_text("runtime-receipt-identity")
    runtime_receipt_physical = hashlib.sha256(
        payloads["runtime-receipt.json"]
    ).hexdigest()
    monkeypatch.setattr(
        replay,
        "_load_runtime_build_receipt_v1",
        lambda *_args, **_kwargs: (
            {
                "receipt_identity_sha256": runtime_receipt_identity,
                "receipt_sha256": runtime_receipt_physical,
                "selected_wheels": provenance_template["selected_wheels"],
            },
            runtime_receipt_physical,
        ),
    )
    monkeypatch.setattr(
        replay,
        "_validate_production_storage_roots_v3",
        lambda **_kwargs: (
            durable_output_parent.resolve(),
            local_work_parent.resolve(),
            _storage_identity(),
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
        runtime_build_receipt_path=files["runtime-receipt.json"],
        durable_mount_root=durable_mount_root,
        durable_storage_marker_path=durable_storage_marker,
        durable_output_parent=durable_output_parent,
        local_work_parent=local_work_parent,
        first_output_root=durable_output_parent / "first",
        second_output_root=durable_output_parent / "second",
    )
    assert result is marker
    assert (
        captured["production_profile_sentinel"]
        is replay._PRODUCTION_PROFILE_SENTINEL
    )
    assert captured["network_namespace_executable"] == replay.LINUX_UNSHARE_PATH_V1
    arguments = captured["worker_arguments"]
    assert isinstance(arguments, tuple)
    assert str(arguments[0]).endswith(
        str(Path("scripts") / "run_weft1_corpus_materialize_a2.py")
    )
    assert arguments[0] != str(files["worker.py"].resolve())
    assert "--cache-root" in arguments
    assert "--execution-provenance" in arguments
    assert captured["extra_environment"] is None
    compatibility_files = captured["compatibility_files"]
    assert isinstance(compatibility_files, dict)
    assert str(compatibility_files["seed_derivation"]).endswith(
        str(Path("training") / "weft1_seed.py")
    )
    assert captured["worker_cwd"] != ROOT
    assert captured["durable_mount_root"] == durable_mount_root
    assert captured["durable_storage_marker_path"] == durable_storage_marker
    assert captured_snapshot_bytes["a2_authority"] == payloads["authority.md"]
    assert captured_snapshot_bytes["enumeration_receipt"] == payloads["enumeration.json"]
    assert captured_snapshot_bytes["cache_download_receipt"] == payloads["download.json"]
    assert captured_snapshot_bytes["source_manifest"] == payloads["source-manifest.json"]
    assert captured_snapshot_bytes["fasttext_model"] == payloads["model.bin"]
    assert captured_snapshot_bytes["runtime_build_receipt"] == payloads[
        "runtime-receipt.json"
    ]
    assert captured["durable_output_parent"] == durable_output_parent.resolve()
    assert captured["local_work_parent"] == local_work_parent.resolve()


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


def test_colab_drive_registration_and_storage_attestation_are_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount_root = tmp_path / "drive"
    storage_root = mount_root / "MyDrive" / "weft1"
    output_parent = storage_root / "outputs"
    local_root = tmp_path / "content-local"
    output_parent.mkdir(parents=True)
    local_root.mkdir()
    marker_path = storage_root / "durable-marker.json"
    durable_row = {
        "filesystem_type": "fuse.drive",
        "major_minor": "0:99",
        "mount_id": 99,
        "mount_point": str(mount_root.resolve()),
        "mount_root": "/",
        "mount_source": "drive",
        "parent_mount_id": 1,
        "st_dev": 99,
    }
    local_row = {
        "filesystem_type": "overlay",
        "major_minor": "0:98",
        "mount_id": 98,
        "mount_point": str(local_root.resolve()),
        "mount_root": "/",
        "mount_source": "overlay",
        "parent_mount_id": 1,
        "st_dev": 98,
    }

    def observe(path: Path) -> dict[str, object]:
        resolved = path.resolve()
        return dict(local_row if resolved == local_root.resolve() else durable_row)

    monkeypatch.setattr(replay, "_observed_mount_identity_v3", observe)
    monkeypatch.setattr(
        replay, "_fsync_directory_if_supported_v3", lambda _path: "supported"
    )
    registration = replay.register_colab_drive_storage_v3(
        durable_mount_root=mount_root,
        durable_storage_root=storage_root,
        marker_path=marker_path,
    )
    assert registration["provider"] == "google_colab_drive_v1"
    identity = replay.attest_production_storage_v3(
        durable_mount_root=mount_root,
        durable_storage_marker_path=marker_path,
        durable_output_parent=output_parent,
        local_work_parent=local_root,
    )
    assert replay.validate_production_storage_identity_v3(identity) == identity
    assert identity["durable_marker_sha256"] == registration["marker_sha256"]

    same_backing = dict(local_row)
    same_backing["st_dev"] = 99
    monkeypatch.setattr(
        replay,
        "_observed_mount_identity_v3",
        lambda path: dict(
            same_backing if path.resolve() == local_root.resolve() else durable_row
        ),
    )
    with pytest.raises(ParentReplayError, match="shares the durable backing"):
        replay.attest_production_storage_v3(
            durable_mount_root=mount_root,
            durable_storage_marker_path=marker_path,
            durable_output_parent=output_parent,
            local_work_parent=local_root,
        )


def test_production_storage_rejects_non_drive_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount_root = tmp_path / "mount"
    storage_root = mount_root / "storage"
    output_parent = storage_root / "outputs"
    local_root = tmp_path / "local"
    output_parent.mkdir(parents=True)
    local_root.mkdir()
    marker = storage_root / "marker.json"
    _write_storage_marker(marker, storage_root)
    local_row = {
        "filesystem_type": "overlay",
        "major_minor": "0:1",
        "mount_id": 1,
        "mount_point": str(tmp_path.resolve()),
        "mount_root": "/",
        "mount_source": "overlay",
        "parent_mount_id": 0,
        "st_dev": 1,
    }
    monkeypatch.setattr(
        replay, "_observed_mount_identity_v3", lambda _path: dict(local_row)
    )
    with pytest.raises(ParentReplayError, match="not the declared Colab Drive"):
        replay.attest_production_storage_v3(
            durable_mount_root=mount_root,
            durable_storage_marker_path=marker,
            durable_output_parent=output_parent,
            local_work_parent=local_root,
        )


def test_final_child_receipt_is_physically_reread_and_bound(tmp_path: Path) -> None:
    output_root = tmp_path / "child"
    output_root.mkdir()
    receipt_path = output_root / replay.CHILD_RECEIPT_FILENAME
    receipt_path.write_bytes(replay._canonical_json_line({"value": 1}))
    child = replay._VerifiedChildReplayV3(
        run_id="child",
        actual_process_id=1,
        output_root=str(output_root),
        child_receipt_bytes=receipt_path.stat().st_size,
        child_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        stdout_sha256="1" * 64,
        stderr_sha256="2" * 64,
        output_file_rows=(),
        output_file_projection_sha256="3" * 64,
        content_projection_sha256="4" * 64,
        dedup_projection_sha256=None,
        dedup_evidence_complete=False,
        content_metadata={"fixture": True},
    )
    row = replay._final_child_receipt_row_v3(child)
    assert row["role"] == "receipt"
    assert row["sha256"] == child.child_receipt_sha256
    receipt_path.write_bytes(replay._canonical_json_line({"value": 2}))
    with pytest.raises(ParentReplayError, match="changed before parent minting"):
        replay._final_child_receipt_row_v3(child)


def test_production_code_snapshot_is_exact_and_rejects_added_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    worker = repository / "scripts" / "run_weft1_corpus_materialize_a2.py"
    package_init = repository / "training" / "__init__.py"
    worker.parent.mkdir(parents=True)
    package_init.parent.mkdir(parents=True)
    worker.write_text("raise SystemExit(0)\n", encoding="utf-8")
    package_init.write_text("", encoding="utf-8")
    (repository / "training" / "rogue.py").write_text("raise RuntimeError\n")
    monkeypatch.setattr(replay, "REPOSITORY_ROOT_V3", repository)
    code_root = tmp_path / "snapshot"
    snapshots = replay._snapshot_production_code_v3(
        {"training_init": package_init, "worker": worker}, code_root=code_root
    )
    replay._validate_exact_code_snapshot_tree_v3(code_root, snapshots)
    assert not (code_root / "training" / "rogue.py").exists()
    (code_root / "training" / "rogue.py").write_text("raise RuntimeError\n")
    with pytest.raises(ParentReplayError, match="gained or lost files"):
        replay._validate_exact_code_snapshot_tree_v3(code_root, snapshots)
