from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import threading

import pytest

import training.weft1_corpus_materialize_a3 as materialize_v4
import training.weft1_corpus_replay_a2 as replay_v3
import training.weft1_corpus_replay_a3 as replay_v4
import scripts.run_weft1_corpus_pa_a3 as cli_v4
from tests import test_weft1_corpus_materialize_a2 as core_fixture
from tests import test_weft1_corpus_materialize_a3 as transport_fixture
from training.weft1_corpus_a2 import execution_authority_v3_bound_sha256
from training.weft1_corpus_materialize_a2 import (
    MATERIALIZER_SCHEMA,
    PRODUCTION_MODE,
    MaterializationResultV3,
    materialize_corpus_pa_v3,
)
from training.weft1_corpus_parsed_asset_cache_v1 import (
    CURRENT_CONTEXT_RESOLUTION_V1,
    PARSED_ASSET_COMPATIBILITY_POLICY_ARTIFACT_SCHEMA_V1,
    PARSED_ASSET_COMPATIBILITY_POLICY_SCHEMA_V1,
    PARSED_ASSET_COMPOSITE_BRIDGE_SCHEMA_V1,
    PARSED_ASSET_INCIDENT_AUTHORITY_PHYSICAL_SHA256_V1,
    PARSED_ASSET_INCIDENT_AUTHORITY_V1,
    ParsedAssetCompatibilityPolicyV1,
    ParsedAssetCompositeBridgeRowV1,
    ParsedAssetCompositeBridgeV1,
    ParsedAssetRecoveryContextV1,
    ParsedAssetRecoveryError,
    load_parsed_asset_compatibility_policy_v1,
    load_parsed_asset_composite_bridge_v1,
    parsed_asset_composite_bridge_path_v1,
    publish_parsed_asset_composite_bridge_v1,
)
from training.weft1_corpus_source_io_a2 import (
    PRODUCTION_PARSER_BINDINGS_V3,
    STACKEDU_PYTHON_PARSER_BINDING_V3,
)
from training.weft1_gtok_contract import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
INCIDENT_COMPATIBILITY_AUTHORITY = (
    ROOT
    / "training"
    / "weft1_pa_parsed_asset_compatibility_authority_20260904.json"
)


def _parsed_context(
    *,
    run_id: str = "production-v4-replay-a",
    code_identity_sha256: str,
) -> ParsedAssetRecoveryContextV1:
    return ParsedAssetRecoveryContextV1(
        run_id=run_id,
        durable_marker_physical_sha256="1" * 64,
        runtime_identity_sha256="2" * 64,
        code_identity_sha256=code_identity_sha256,
        input_identity_sha256="3" * 64,
    )


def _compatibility_authority(
    path: Path,
    *,
    current: ParsedAssetRecoveryContextV1,
    predecessor: ParsedAssetRecoveryContextV1,
) -> tuple[ParsedAssetCompatibilityPolicyV1, bytes, str]:
    bindings = {
        (source, binding.binding_sha256)
        for source, binding in PRODUCTION_PARSER_BINDINGS_V3.items()
        if source != "fineweb_edu"
    }
    bindings.add(("stackedu", STACKEDU_PYTHON_PARSER_BINDING_V3.binding_sha256))
    policy = ParsedAssetCompatibilityPolicyV1(
        schema=PARSED_ASSET_COMPATIBILITY_POLICY_SCHEMA_V1,
        authority_sha256=PARSED_ASSET_INCIDENT_AUTHORITY_PHYSICAL_SHA256_V1,
        eligible_run_id=current.run_id,
        predecessor_code_identity_sha256=predecessor.code_identity_sha256,
        successor_code_identity_sha256=current.code_identity_sha256,
        compatible_parser_bindings=tuple(sorted(bindings)),
        excluded_source_families=("fineweb_edu",),
        expected_predecessor_asset_count=394,
        expected_current_asset_count=3,
    )
    raw = canonical_json_bytes(
        {
            "policy": asdict(policy),
            "policy_sha256": policy.identity_sha256,
            "schema": PARSED_ASSET_COMPATIBILITY_POLICY_ARTIFACT_SCHEMA_V1,
        }
    ) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return policy, raw, hashlib.sha256(raw).hexdigest()


def test_checked_in_incident_policy_matches_current_parser_code_closure() -> None:
    policy, physical_bytes, physical_sha256 = (
        load_parsed_asset_compatibility_policy_v1(
            INCIDENT_COMPATIBILITY_AUTHORITY
        )
    )
    files = replay_v4._compatibility_files_v4()
    parsed_rows = replay_v4._parsed_asset_semantic_file_rows_v5(
        {
            logical_name: path
            for logical_name, path in files.items()
            if logical_name in replay_v4.PARSED_ASSET_CODE_LOGICAL_NAMES_V1
        },
        name="checked-in parsed-asset semantic code closure",
    )
    observed_code_identity = replay_v4.parsed_asset_code_identity_sha256_v5(
        parsed_rows
    )
    assert policy.eligible_run_id == "production-v4-replay-a"
    assert policy.predecessor_code_identity_sha256 == (
        "89a8b42dbe06edad2db7c67ae126c779a356612a3ed9e94587a98befb0d94657"
    )
    assert policy.successor_code_identity_sha256 == observed_code_identity
    assert policy.expected_predecessor_asset_count == 394
    assert policy.expected_current_asset_count == 3
    assert physical_bytes == 1_273
    assert physical_sha256 == (
        "c9fade99ee00d683500235e010b3807772e86c4738146ec470f4a979fc30f327"
    )


def test_checked_in_incident_authority_records_frozen_census_identity_failure() -> None:
    failure = PARSED_ASSET_INCIDENT_AUTHORITY_V1[
        "fineweb_census_identity_projection_failure"
    ]
    assert failure == {
        "attempt_count": 3,
        "durable_termination_receipt": {
            "bytes": 2_910,
            "path": (
                "launch-receipts/"
                "pa-v4-ce2f5577-r6-portable-v3-supervisor.termination-v1.json"
            ),
            "sha256": (
                "705c993b8b181410b25bf8709813884c9d0dc3ac05fc52d5f5036c8bff5d5c2d"
            ),
        },
        "gate_minted": False,
        "rejection": "FINEWEB_SELECTED_ASSET_IDENTITY_OR_ORDER_DIFFERS_FROM_CENSUS",
        "repository_commit": "ce2f55779d0573e8fd7974a6ca12a47ff1cc2607",
        "root_cause": (
            "FROZEN_V3_CENSUS_IDENTITY_COMPARED_TO_CURRENT_V4_ASSET_OVERRIDE_IDENTITY"
        ),
        "successor_gate_count": 0,
        "successor_receipt_count": 0,
        "successor_write_count": 0,
        "supervisor_disposition": "EXHAUSTED_AND_STOPPED_AT_ZERO",
    }


@pytest.mark.parametrize(
    "changes",
    (
        {"eligible_run_id": "production-v4-replay-b"},
        {"predecessor_code_identity_sha256": "f" * 64},
        {"expected_predecessor_asset_count": 393},
        {"expected_current_asset_count": 4},
    ),
)
def test_compatibility_policy_cannot_escape_exact_incident_scope(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    predecessor = _parsed_context(
        code_identity_sha256=(
            "89a8b42dbe06edad2db7c67ae126c779a356612a3ed9e94587a98befb0d94657"
        )
    )
    current = _parsed_context(code_identity_sha256="6" * 64)
    policy, unused_raw, unused_physical_sha = _compatibility_authority(
        tmp_path / "policy.json",
        current=current,
        predecessor=predecessor,
    )
    del unused_raw, unused_physical_sha
    with pytest.raises(ValueError, match="exact incident scope"):
        replace(policy, **changes)


_COPY_WORKER = r'''from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys

raw_parsed_cache_root = os.environ.get("WEFT1_REPLAY_PARSED_ASSET_CACHE_ROOT")
if "--cache-fill-only" in sys.argv:
    if raw_parsed_cache_root is None:
        raise SystemExit("cache-fill worker lacks its parsed cache assignment")
    parsed_cache_root = Path(raw_parsed_cache_root)
    (parsed_cache_root / "test-fill-complete").write_bytes(b"complete\n")
    raise SystemExit(0)
if raw_parsed_cache_root is not None:
    parsed_cache_root = Path(raw_parsed_cache_root)
    cache_parent = parsed_cache_root.parent.parent
    if len(tuple(cache_parent.glob("*/*/test-fill-complete"))) != 2:
        raise SystemExit("both cache-fill lanes must precede materialization")

template = Path(sys.argv[1]).resolve(strict=True)
root = Path(os.environ["WEFT1_REPLAY_OUTPUT_ROOT"])
receipt_path = Path(os.environ["WEFT1_REPLAY_RECEIPT_PATH"])
shutil.copytree(template, root)

probe = socket.socket()
try:
    probe.connect(("127.0.0.1", 9))
except RuntimeError:
    network_probe = "python_socket_connect_blocked"
else:
    raise SystemExit("network guard did not block the probe")
finally:
    probe.close()

def canonical(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")

def row(path, role):
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": path.relative_to(root).as_posix(),
        "role": role,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }

descriptor = json.loads(
    (root / "artifacts" / "d2-evidence-descriptor.json").read_text("utf-8")
)
metadata = descriptor["parent_replay_metadata"]
dedup_paths = {
    metadata["decision_ledger_path"],
    metadata["selection_ledger_path"],
    metadata["minhash_recall_audit_path"],
}
files = [
    row(path, "dedup_evidence" if path.relative_to(root).as_posix() in dedup_paths
        else "content")
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    if path.is_file()
]
content = json.loads((root / "content-manifest.json").read_text("utf-8"))
receipt = {
    "content_metadata": {
        "content_identity_sha256": content["content_identity_sha256"],
        "fixture_kind": "v4_full_corpus_forward_replay",
    },
    "dedup_evidence_complete": True,
    "dedup_metadata": metadata,
    "files": files,
    "input_identity_sha256": os.environ["WEFT1_REPLAY_INPUT_IDENTITY_SHA256"],
    "network_disabled": True,
    "network_guard_active": os.environ.get("WEFT1_NETWORK_GUARD_ACTIVE") == "1",
    "network_guard_sha256": os.environ["WEFT1_NETWORK_GUARD_SHA256"],
    "network_probe": network_probe,
    "output_root": str(root),
    "process_id": os.getpid(),
    "run_id": os.environ["WEFT1_REPLAY_RUN_ID"],
    "schema": "weft1_corpus_parent_replay_child_receipt_v3",
    "worker_compatibility_sha256": os.environ[
        "WEFT1_REPLAY_WORKER_COMPATIBILITY_SHA256"
    ],
}
receipt_path.write_bytes(canonical(receipt))
'''


def test_v4_cli_default_watchdog_exceeds_observed_replay_projection() -> None:
    arguments = cli_v4.build_parser().parse_args(
        [
            "full-pa-v4",
            "--enumeration-receipt",
            "enumeration.json",
            "--cache-download-receipt",
            "download.json",
            "--source-cache-manifest",
            "manifest.json",
            "--source-cache",
            "source-cache",
            "--fasttext-model",
            "lid.176.bin",
            "--runtime-build-receipt",
            "runtime.json",
            "--durable-mount-root",
            "drive",
            "--durable-storage-marker",
            "marker.json",
            "--durable-output-parent",
            "output",
            "--durable-parsed-asset-cache-parent",
            "parsed-cache",
            "--local-work-parent",
            "work",
            "--receipt-out",
            "output/receipt.json",
        ]
    )
    assert (
        arguments.timeout_seconds
        == replay_v4.V4_DEFAULT_WORKER_TIMEOUT_SECONDS
        == 14 * 24 * 60 * 60
    )
    assert arguments.incident_compatibility_authority_path is None


def test_v4_parent_fills_both_lanes_before_either_materialization() -> None:
    assert replay_v4.V4_PARENT_LANE_OPERATION_ORDER == (
        ("cache_fill", 0),
        ("cache_fill", 1),
        ("materialize", 0),
        ("materialize", 1),
    )
    assert (
        replay_v4.V4_WRITE_ENABLED_CHILD_POLICY
        == "SINGLE_PARENT_ONE_WRITE_ENABLED_CHILD"
    )
    assert replay_v4.V4_MAX_CONCURRENT_WRITE_ENABLED_CHILDREN == 1


def test_v4_write_enabled_child_guard_enforces_exact_synchronous_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[tuple[str, ...], Path]] = []

    def fake_run_worker(**kwargs: object) -> tuple[int, bytes, bytes]:
        observed.append((kwargs["command"], kwargs["cwd"]))  # type: ignore[arg-type]
        return 100 + len(observed), b"", b""

    monkeypatch.setattr(replay_v3, "_run_worker", fake_run_worker)
    guard = replay_v4._WriteEnabledChildGuardV4()
    for operation, lane_index in replay_v4.V4_PARENT_LANE_OPERATION_ORDER:
        guard.run(
            operation=operation,
            lane_index=lane_index,
            command=(operation, str(lane_index)),
            cwd=ROOT,
            environment={},
            timeout_seconds=1.0,
        )
    guard.assert_complete()
    assert guard.max_active_children == 1
    assert [row[0] for row in observed] == [
        (operation, str(lane_index))
        for operation, lane_index in replay_v4.V4_PARENT_LANE_OPERATION_ORDER
    ]


def test_v4_write_enabled_child_guard_rejects_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = replay_v4._WriteEnabledChildGuardV4()

    def overlapping_worker(**unused_kwargs: object) -> tuple[int, bytes, bytes]:
        with pytest.raises(replay_v3.ParentReplayError, match="overlap"):
            guard.run(
                operation="cache_fill",
                lane_index=0,
                command=("nested",),
                cwd=ROOT,
                environment={},
                timeout_seconds=1.0,
            )
        return 101, b"", b""

    monkeypatch.setattr(replay_v3, "_run_worker", overlapping_worker)
    guard.run(
        operation="cache_fill",
        lane_index=0,
        command=("outer",),
        cwd=ROOT,
        environment={},
        timeout_seconds=1.0,
    )
    assert guard.max_active_children == 1


def test_v4_parsed_asset_identity_schemas_are_authority_bound() -> None:
    payload = ({"logical_name": "component", "sha256": "a" * 64},)
    assert len(replay_v4.parsed_asset_code_identity_sha256_v5(payload)) == 64
    assert len(
        replay_v4.execution_authority_v4_bound_sha256(
            replay_v4.PARSED_ASSET_INPUT_IDENTITY_SCHEMA_V4, payload
        )
    ) == 64
    with pytest.raises(ValueError, match="explicit v4 schema"):
        replay_v4.execution_authority_v4_bound_sha256(
            replay_v4.PARSED_ASSET_CODE_IDENTITY_SCHEMA_V5, payload
        )
    assert replay_v4.PARSED_ASSET_CODE_IDENTITY_SCHEMA_V5.endswith("_v5")
    assert replay_v4.PARSED_ASSET_INPUT_IDENTITY_SCHEMA_V4.endswith("_v4")
    assert (
        "training/weft1_fineweb_selected_parquet_schema_census_20260904.json"
        in replay_v4.PARSED_ASSET_CODE_LOGICAL_NAMES_V1
    )


@pytest.mark.parametrize(
    ("lf_bytes", "alternate_bytes"),
    (
        (b"first\nsecond\n", b"first\r\nsecond\r\n"),
        (b"first\nsecond\n", b"first\rsecond\r"),
        (b"first\nsecond\nthird\n", b"first\r\nsecond\rthird\n"),
    ),
)
def test_v5_python_semantic_identity_is_newline_portable_without_weakening_physical_rows(
    tmp_path: Path, lf_bytes: bytes, alternate_bytes: bytes
) -> None:
    lf = tmp_path / "lf.py"
    alternate = tmp_path / "alternate.py"
    lf.write_bytes(lf_bytes)
    alternate.write_bytes(alternate_bytes)
    lf_files = {"component": lf}
    alternate_files = {"component": alternate}
    lf_rows = replay_v4._parsed_asset_semantic_file_rows_v5(
        lf_files, name="LF Python fixture"
    )
    alternate_rows = replay_v4._parsed_asset_semantic_file_rows_v5(
        alternate_files, name="alternate-newline Python fixture"
    )
    assert lf_rows == alternate_rows
    assert lf_rows[0]["normalization"] == (
        replay_v4.PARSED_ASSET_PYTHON_NORMALIZATION_V5
    )
    assert replay_v4.parsed_asset_code_identity_sha256_v5(
        lf_rows
    ) == replay_v4.parsed_asset_code_identity_sha256_v5(alternate_rows)
    lf_physical_rows = replay_v3._logical_file_rows(
        lf_files, name="physical LF Python fixture"
    )
    alternate_physical_rows = replay_v3._logical_file_rows(
        alternate_files, name="physical alternate-newline Python fixture"
    )
    assert lf_physical_rows != alternate_physical_rows
    assert replay_v4.execution_authority_v4_bound_sha256(
        replay_v4.WORKER_COMPATIBILITY_SCHEMA_V4, lf_physical_rows
    ) != replay_v4.execution_authority_v4_bound_sha256(
        replay_v4.WORKER_COMPATIBILITY_SCHEMA_V4,
        alternate_physical_rows,
    )


@pytest.mark.parametrize(
    "changed_bytes",
    (b"first", b"fIrst\n"),
)
def test_v5_python_semantic_identity_preserves_non_newline_changes(
    tmp_path: Path, changed_bytes: bytes
) -> None:
    baseline = tmp_path / "baseline.py"
    changed = tmp_path / "changed.py"
    baseline.write_bytes(b"first\n")
    changed.write_bytes(changed_bytes)
    baseline_rows = replay_v4._parsed_asset_semantic_file_rows_v5(
        {"component": baseline}, name="baseline Python fixture"
    )
    changed_rows = replay_v4._parsed_asset_semantic_file_rows_v5(
        {"component": changed}, name="changed Python fixture"
    )
    assert baseline_rows != changed_rows
    assert replay_v4.parsed_asset_code_identity_sha256_v5(
        baseline_rows
    ) != replay_v4.parsed_asset_code_identity_sha256_v5(changed_rows)


def test_v5_non_python_authority_bytes_remain_physically_exact(
    tmp_path: Path,
) -> None:
    lf = tmp_path / "authority-lf.json"
    crlf = tmp_path / "authority-crlf.json"
    lf.write_bytes(b'{"authority":true}\n')
    crlf.write_bytes(b'{"authority":true}\r\n')
    lf_rows = replay_v4._parsed_asset_semantic_file_rows_v5(
        {"authority": lf}, name="LF authority fixture"
    )
    crlf_rows = replay_v4._parsed_asset_semantic_file_rows_v5(
        {"authority": crlf}, name="CRLF authority fixture"
    )
    assert lf_rows != crlf_rows
    assert lf_rows[0]["normalization"] == (
        replay_v4.PARSED_ASSET_EXACT_NORMALIZATION_V5
    )
    assert replay_v4.parsed_asset_code_identity_sha256_v5(
        lf_rows
    ) != replay_v4.parsed_asset_code_identity_sha256_v5(crlf_rows)


def test_v4_parsed_asset_code_identity_covers_exact_local_semantic_closure() -> None:
    required = frozenset(
        {
            "production_io",
            "worker",
            "training/__init__.py",
            "training/weft1_corpus_a2.py",
            "training/weft1_corpus_a3.py",
            "training/weft1_corpus_breakdown_a3.py",
            "training/weft1_corpus_enumeration_a2.py",
            "training/weft1_corpus_fetch_a2.py",
            "training/weft1_corpus_fetch_a3.py",
            "training/weft1_corpus_materialize_a2.py",
            "training/weft1_corpus_materialize_a3.py",
            "training/weft1_corpus_parsed_asset_cache_v1.py",
            "training/weft1_corpus_replay_a2.py",
            "training/weft1_corpus_replay_a3.py",
            "training/weft1_corpus_semantic_evidence_a3.py",
            "training/weft1_corpus_source_io_a2.py",
            "training/weft1_corpus_sources_a2.py",
            "training/weft1_fineweb_selected_parquet_schema_census_20260904.json",
            "training/weft1_gtok_a1_contract.py",
            "training/weft1_gtok_contract.py",
            "training/weft1_pa_schema_remediation_incident_authority_20260904.json",
            "training/weft1_seed.py",
            "training/weft1_strict_io.py",
        }
    )
    assert replay_v4.PARSED_ASSET_CODE_LOGICAL_NAMES_V1 == required
    all_files = replay_v4._compatibility_files_v4()
    files = {
        logical_name: path
        for logical_name, path in all_files.items()
        if logical_name in required
    }
    rows = replay_v4._parsed_asset_semantic_file_rows_v5(
        files, name="parsed-cache semantic closure"
    )
    assert len(rows) == 23
    assert {str(row["logical_name"]) for row in rows} == required
    assert sum(
        row["normalization"] == replay_v4.PARSED_ASSET_PYTHON_NORMALIZATION_V5
        for row in rows
    ) == 21
    assert sum(
        row["normalization"] == replay_v4.PARSED_ASSET_EXACT_NORMALIZATION_V5
        for row in rows
    ) == 2
    assert {
        row["logical_name"]
        for row in rows
        if row["normalization"] == replay_v4.PARSED_ASSET_EXACT_NORMALIZATION_V5
    } == {
        "training/weft1_fineweb_selected_parquet_schema_census_20260904.json",
        "training/weft1_pa_schema_remediation_incident_authority_20260904.json",
    }
    baseline = replay_v4.parsed_asset_code_identity_sha256_v5(rows)
    for logical_name in required:
        mutated = tuple(
            {
                **row,
                "sha256": (
                    "0" * 64
                    if row["logical_name"] == logical_name
                    and row["sha256"] != "0" * 64
                    else "f" * 64
                    if row["logical_name"] == logical_name
                    else row["sha256"]
                ),
            }
            for row in rows
        )
        assert replay_v4.parsed_asset_code_identity_sha256_v5(mutated) != baseline


def test_v5_full_closure_is_portable_while_physical_custody_stays_exact(
    tmp_path: Path,
) -> None:
    all_files = replay_v4._compatibility_files_v4()
    closure = {
        logical_name: path
        for logical_name, path in all_files.items()
        if logical_name in replay_v4.PARSED_ASSET_CODE_LOGICAL_NAMES_V1
    }
    lf_files: dict[str, Path] = {}
    crlf_files: dict[str, Path] = {}
    for index, (logical_name, source_path) in enumerate(sorted(closure.items())):
        suffix = source_path.suffix
        lf_path = tmp_path / "lf" / f"{index:02d}{suffix}"
        crlf_path = tmp_path / "crlf" / f"{index:02d}{suffix}"
        lf_path.parent.mkdir(parents=True, exist_ok=True)
        crlf_path.parent.mkdir(parents=True, exist_ok=True)
        raw = source_path.read_bytes()
        if suffix == ".py":
            semantic = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            lf_path.write_bytes(semantic)
            crlf_path.write_bytes(semantic.replace(b"\n", b"\r\n"))
        else:
            lf_path.write_bytes(raw)
            crlf_path.write_bytes(raw)
        lf_files[logical_name] = lf_path
        crlf_files[logical_name] = crlf_path

    lf_semantic = replay_v4._parsed_asset_semantic_file_rows_v5(
        lf_files, name="full LF semantic closure"
    )
    crlf_semantic = replay_v4._parsed_asset_semantic_file_rows_v5(
        crlf_files, name="full CRLF semantic closure"
    )
    assert len(lf_semantic) == len(crlf_semantic) == 23
    assert sum(
        row["normalization"] == replay_v4.PARSED_ASSET_PYTHON_NORMALIZATION_V5
        for row in lf_semantic
    ) == 21
    assert sum(
        row["normalization"] == replay_v4.PARSED_ASSET_EXACT_NORMALIZATION_V5
        for row in lf_semantic
    ) == 2
    assert {
        row["logical_name"]
        for row in lf_semantic
        if row["normalization"] == replay_v4.PARSED_ASSET_EXACT_NORMALIZATION_V5
    } == {
        "training/weft1_fineweb_selected_parquet_schema_census_20260904.json",
        "training/weft1_pa_schema_remediation_incident_authority_20260904.json",
    }
    assert replay_v4.parsed_asset_code_identity_sha256_v5(
        lf_semantic
    ) == replay_v4.parsed_asset_code_identity_sha256_v5(crlf_semantic)

    lf_physical = replay_v3._logical_file_rows(
        lf_files, name="full LF physical closure"
    )
    crlf_physical = replay_v3._logical_file_rows(
        crlf_files, name="full CRLF physical closure"
    )
    assert lf_physical != crlf_physical
    assert replay_v4.execution_authority_v4_bound_sha256(
        replay_v4.WORKER_COMPATIBILITY_SCHEMA_V4,
        {"compatibility_files": lf_physical},
    ) != replay_v4.execution_authority_v4_bound_sha256(
        replay_v4.WORKER_COMPATIBILITY_SCHEMA_V4,
        {"compatibility_files": crlf_physical},
    )


def test_v4_worker_compatibility_assignment_is_complete_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = _parsed_context(code_identity_sha256="6" * 64)
    predecessor = _parsed_context(
        code_identity_sha256=(
            "89a8b42dbe06edad2db7c67ae126c779a356612a3ed9e94587a98befb0d94657"
        )
    )
    cache_lane_parent = tmp_path / "durable" / current.run_id
    current_root = cache_lane_parent / current.identity_sha256
    predecessor_root = cache_lane_parent / predecessor.identity_sha256
    for path in (
        current_root,
        predecessor_root,
        tmp_path / "output",
        tmp_path / "local",
        tmp_path / "source",
    ):
        path.mkdir(parents=True, exist_ok=True)
    authority_path = tmp_path / "local" / "snapshot" / "compatibility.json"
    policy, raw, physical_sha = _compatibility_authority(
        authority_path, current=current, predecessor=predecessor
    )
    assignment = {
        materialize_v4.PARSED_ASSET_COMPATIBILITY_AUTHORITY_PATH_ENV_V4: str(
            authority_path
        ),
        materialize_v4.PARSED_ASSET_COMPATIBILITY_POLICY_SHA256_ENV_V4: (
            policy.identity_sha256
        ),
        materialize_v4.PARSED_ASSET_COMPATIBILITY_PHYSICAL_BYTES_ENV_V4: str(
            len(raw)
        ),
        materialize_v4.PARSED_ASSET_COMPATIBILITY_PHYSICAL_SHA256_ENV_V4: (
            physical_sha
        ),
        materialize_v4.PARSED_ASSET_COMPATIBILITY_PREDECESSOR_CACHE_ROOT_ENV_V4: str(
            predecessor_root
        ),
        materialize_v4.PARSED_ASSET_COMPATIBILITY_PREDECESSOR_CONTEXT_SHA256_ENV_V4: (
            predecessor.identity_sha256
        ),
    }
    for key, value in assignment.items():
        monkeypatch.setenv(key, value)
    loaded_root, loaded_context, loaded_policy = (
        materialize_v4._load_parsed_asset_compatibility_assignment_v4(
            current_context=current,
            current_cache_root=current_root,
            output_parent=tmp_path / "output",
            local_work_parent=tmp_path / "local",
            source_cache_root=tmp_path / "source",
        )
    )
    assert loaded_root == predecessor_root.resolve(strict=True)
    assert loaded_context == predecessor
    assert loaded_policy == policy
    monkeypatch.setenv(
        materialize_v4.PARSED_ASSET_COMPATIBILITY_PHYSICAL_SHA256_ENV_V4,
        "f" * 64,
    )
    with pytest.raises(
        materialize_v4.CorpusMaterializationV4Error,
        match="changed after parent snapshot",
    ):
        materialize_v4._load_parsed_asset_compatibility_assignment_v4(
            current_context=current,
            current_cache_root=current_root,
            output_parent=tmp_path / "output",
            local_work_parent=tmp_path / "local",
            source_cache_root=tmp_path / "source",
        )


def test_v4_worker_fresh_lane_rejects_partial_compatibility_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = _parsed_context(
        run_id="production-v4-replay-b", code_identity_sha256="6" * 64
    )
    for key in materialize_v4.PARSED_ASSET_COMPATIBILITY_ENVIRONMENT_KEYS_V4:
        monkeypatch.delenv(key, raising=False)
    roots = {
        "current_cache_root": tmp_path / "durable" / current.run_id / current.identity_sha256,
        "output_parent": tmp_path / "output",
        "local_work_parent": tmp_path / "local",
        "source_cache_root": tmp_path / "source",
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    assert materialize_v4._load_parsed_asset_compatibility_assignment_v4(
        current_context=current, **roots
    ) == (None, None, None)
    monkeypatch.setenv(
        materialize_v4.PARSED_ASSET_COMPATIBILITY_AUTHORITY_PATH_ENV_V4,
        str(tmp_path / "missing.json"),
    )
    with pytest.raises(
        materialize_v4.CorpusMaterializationV4Error,
        match="one complete set",
    ):
        materialize_v4._load_parsed_asset_compatibility_assignment_v4(
            current_context=current, **roots
        )


def _single_current_bridge(
    current: ParsedAssetRecoveryContextV1,
    *,
    source_asset_sha256: str = "8" * 64,
) -> ParsedAssetCompositeBridgeV1:
    row = ParsedAssetCompositeBridgeRowV1(
        source_family="fixture",
        asset_order_ordinal=0,
        source_asset_identity_sha256="7" * 64,
        source_asset_sha256=source_asset_sha256,
        parser_binding_sha256="9" * 64,
        first_event_ordinal=0,
        next_event_ordinal=1,
        resolution=CURRENT_CONTEXT_RESOLUTION_V1,
        selected_context_identity_sha256=current.identity_sha256,
        selected_code_identity_sha256=current.code_identity_sha256,
        segment_relative_path="segments/fixture.parsed.jsonl.zst",
        segment_physical_bytes=1,
        segment_physical_sha256="a" * 64,
        segment_receipt_sha256="b" * 64,
        segment_receipt_physical_bytes=1,
        segment_receipt_physical_sha256="c" * 64,
    )
    return ParsedAssetCompositeBridgeV1(
        schema=PARSED_ASSET_COMPOSITE_BRIDGE_SCHEMA_V1,
        recovery_domain="WEFT1_PARSED_ASSET_CACHE_V1_FRESH_ONLY_NO_R3_IMPORT",
        current_context=current,
        predecessor_context=None,
        compatibility_policy_sha256=None,
        rows=(row,),
        current_asset_count=1,
        predecessor_asset_count=0,
    )


def test_v4_parent_bridge_snapshot_detects_physical_drift(tmp_path: Path) -> None:
    current = _parsed_context(code_identity_sha256="6" * 64)
    cache_root = tmp_path / "bridge"
    cache_root.mkdir()
    bridge = _single_current_bridge(current)
    publish_parsed_asset_composite_bridge_v1(cache_root, bridge)
    snapshot = replay_v4._load_expected_parsed_asset_bridge_v4(
        cache_root=cache_root,
        current_context=current,
        predecessor_context=None,
        compatibility_policy=None,
    )
    assert snapshot[0] == bridge
    bridge_path = parsed_asset_composite_bridge_path_v1(cache_root)
    bridge_path.write_bytes(bridge_path.read_bytes() + b" ")
    with pytest.raises(replay_v3.ParentReplayError, match="bridge failed"):
        replay_v4._load_expected_parsed_asset_bridge_v4(
            cache_root=cache_root,
            current_context=current,
            predecessor_context=None,
            compatibility_policy=None,
        )


def test_composite_bridge_rows_are_nonempty_exact_immutable_tuple() -> None:
    current = _parsed_context(code_identity_sha256="6" * 64)
    bridge = _single_current_bridge(current)
    row = bridge.rows[0]
    mutable_rows = [row]
    with pytest.raises(TypeError, match="exact typed tuple"):
        replace(bridge, rows=mutable_rows)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        replace(bridge, rows=())
    copied = replace(bridge, rows=tuple(mutable_rows))
    before = copied.receipt_sha256
    mutable_rows.append(row)
    assert copied.rows == (row,)
    assert copied.receipt_sha256 == before


def test_bridge_short_write_strands_only_fail_closed_bridge_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = _parsed_context(code_identity_sha256="6" * 64)
    cache_root = tmp_path / "bridge-short-write"
    segment = cache_root / "segments" / "fixture.parsed.jsonl.zst"
    receipt = segment.with_name(segment.name + ".receipt.json")
    segment.parent.mkdir(parents=True)
    segment.write_bytes(b"S")
    receipt.write_bytes(b"R")
    bridge = _single_current_bridge(current)
    bridge_path = parsed_asset_composite_bridge_path_v1(cache_root)
    original_open = Path.open

    class _ShortWriter:
        def __init__(self, handle: object) -> None:
            self.handle = handle

        def __enter__(self) -> "_ShortWriter":
            return self

        def __exit__(self, *args: object) -> None:
            self.handle.close()  # type: ignore[attr-defined]

        def write(self, value: bytes) -> int:
            return self.handle.write(value[:-1])  # type: ignore[attr-defined]

        def flush(self) -> None:
            self.handle.flush()  # type: ignore[attr-defined]

        def fileno(self) -> int:
            return self.handle.fileno()  # type: ignore[attr-defined,no-any-return]

    def short_open(
        self: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        handle = original_open(self, mode, *args, **kwargs)
        if self == bridge_path and mode == "xb":
            return _ShortWriter(handle)
        return handle

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "open", short_open)
        with pytest.raises(ParsedAssetRecoveryError, match="publication was short"):
            publish_parsed_asset_composite_bridge_v1(cache_root, bridge)
    assert bridge_path.exists()
    assert segment.read_bytes() == b"S"
    assert receipt.read_bytes() == b"R"
    with pytest.raises(ParsedAssetRecoveryError):
        publish_parsed_asset_composite_bridge_v1(cache_root, bridge)
    assert segment.read_bytes() == b"S"
    assert receipt.read_bytes() == b"R"


def test_bridge_two_publishers_never_clobber_different_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = _parsed_context(code_identity_sha256="6" * 64)
    first = _single_current_bridge(current, source_asset_sha256="8" * 64)
    second = _single_current_bridge(current, source_asset_sha256="d" * 64)
    cache_root = tmp_path / "bridge-race"
    cache_root.mkdir()
    bridge_path = parsed_asset_composite_bridge_path_v1(cache_root)
    barrier = threading.Barrier(2)
    original_open = Path.open

    def synchronized_open(
        self: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        if self == bridge_path and mode == "xb":
            barrier.wait(timeout=5)
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", synchronized_open)

    def publish(candidate: ParsedAssetCompositeBridgeV1) -> object:
        try:
            return publish_parsed_asset_composite_bridge_v1(cache_root, candidate)
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(publish, (first, second)))
    assert sum(isinstance(value, tuple) for value in outcomes) == 1
    assert sum(isinstance(value, ParsedAssetRecoveryError) for value in outcomes) == 1
    loaded, unused_bytes, unused_sha256 = load_parsed_asset_composite_bridge_v1(
        cache_root
    )
    del unused_bytes, unused_sha256
    assert loaded in {first, second}


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        True,
        0.0,
        float("inf"),
        float("nan"),
        replay_v4.V4_DEFAULT_WORKER_TIMEOUT_SECONDS + 1,
    ],
)
def test_v4_parent_rejects_invalid_worker_timeout_before_io(
    tmp_path: Path, timeout_seconds: object
) -> None:
    with pytest.raises(replay_v3.ParentReplayError, match="14-day per-worker"):
        replay_v4.verify_production_materialization_replays_v4(
            python_executable=Path(sys.executable),
            enumeration_receipt_path=tmp_path / "enumeration.json",
            cache_download_receipt_path=tmp_path / "download.json",
            source_manifest_path=tmp_path / "manifest.json",
            cache_root=tmp_path / "source-cache",
            fasttext_model_path=tmp_path / "lid.176.bin",
            runtime_build_receipt_path=tmp_path / "runtime.json",
            durable_mount_root=tmp_path / "drive",
            durable_storage_marker_path=tmp_path / "marker.json",
            durable_output_parent=tmp_path / "output",
            durable_parsed_asset_cache_parent=tmp_path / "parsed-cache",
            local_work_parent=tmp_path / "work",
            first_output_root=tmp_path / "output" / "a",
            second_output_root=tmp_path / "output" / "b",
            timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
        )


def _rewrite_fixture_as_verified_v3_core(
    result: MaterializationResultV3,
    *,
    source_identity_sha256: str,
) -> MaterializationResultV3:
    root = result.output_root
    content_path = root / "content-manifest.json"
    content = json.loads(content_path.read_text(encoding="utf-8"))
    content.pop("content_identity_sha256")
    content.update(
        {
            "mode": PRODUCTION_MODE,
            "readiness": "AUTHORITATIVE_INPUTS_D1_READY_NO_GATE_MINT",
            "schema": MATERIALIZER_SCHEMA,
            "source_identity_sha256": source_identity_sha256,
        }
    )
    content_identity = execution_authority_v3_bound_sha256(
        "weft1_corpus_materialized_content_v3", content
    )
    content["content_identity_sha256"] = content_identity
    materialize_v4._atomic_replace_json(content_path, content)

    inventory = tuple(
        {
            "bytes": path.stat().st_size,
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": materialize_v4._sha256_file(path),
        }
        for path in sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.name != "d1-ready-manifest.json"
            ),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    )
    d1 = {
        "content_identity_sha256": content_identity,
        "file_inventory": inventory,
        "gate_minted": False,
        "mode": PRODUCTION_MODE,
        "readiness": "AUTHORITATIVE_INPUTS_D1_READY_NO_GATE_MINT",
        "schema": "weft1_corpus_d1_ready_manifest_v3",
        "source_identity_sha256": source_identity_sha256,
    }
    d1["d1_ready_identity_sha256"] = execution_authority_v3_bound_sha256(
        "weft1_corpus_d1_ready_inventory_v3", d1
    )
    d1_sha = materialize_v4._atomic_replace_json(
        root / "d1-ready-manifest.json", d1
    )
    return MaterializationResultV3(
        mode=PRODUCTION_MODE,
        source_identity_sha256=source_identity_sha256,
        content_identity_sha256=content_identity,
        d1_ready_manifest_sha256=d1_sha,
        output_root=root,
        work_root=result.work_root,
    )


def _build_v4_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, materialize_v4.MaterializationInputV4]:
    inputs = transport_fixture._load_fixture(tmp_path / "transport", monkeypatch)
    core = materialize_corpus_pa_v3(
        inputs=core_fixture._fixture_inputs(),
        plan=core_fixture._fixture_plan(),
        language_classifier=core_fixture._EnglishOnlyClassifier(),
        output_root=tmp_path / "template",
        work_root=tmp_path / "work",
    )
    verified_core = _rewrite_fixture_as_verified_v3_core(
        core, source_identity_sha256=inputs.source_identity_sha256
    )
    finalized = materialize_v4.finalize_materialization_output_v4(
        verified_core, inputs
    )
    return finalized.output_root, inputs


def test_two_subprocess_v4_fixture_replays_preserve_full_and_d2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template, inputs = _build_v4_template(tmp_path, monkeypatch)
    worker = tmp_path / "copy-worker.py"
    worker.write_text(_COPY_WORKER, encoding="utf-8", newline="\n")
    input_files = {
        f"template/{path.relative_to(template).as_posix()}": path
        for path in sorted(
            (item for item in template.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(template).as_posix(),
        )
    }
    receipt = replay_v3.verify_parent_replays_v3(
        python_executable=Path(sys.executable),
        worker_arguments=(str(worker), str(template)),
        first_output_root=tmp_path / "outputs" / "a",
        second_output_root=tmp_path / "outputs" / "b",
        input_files=input_files,
        compatibility_files={"worker": worker},
        worker_cwd=ROOT,
        timeout_seconds=30,
    )
    assert receipt.d1_file_replay_verified is True
    assert receipt.d2_dedup_replay_verified is True
    assert receipt.first_process_id != receipt.second_process_id
    assert receipt.first_output_root != receipt.second_output_root

    expected_source_strata = tuple(
        (family.route.source_family, family.route.stratum)
        for family in inputs.upstream_enumeration.families
    )
    identities = []
    for raw_root in (receipt.first_output_root, receipt.second_output_root):
        root = Path(raw_root)
        full_path = root / materialize_v4.FULL_SHARD_MANIFEST_RELATIVE_PATH_V4
        full = json.loads(full_path.read_text(encoding="utf-8"))
        rows = {}
        for path in root.rglob("*"):
            if path.is_file() and path.name != replay_v3.CHILD_RECEIPT_FILENAME:
                relative = path.relative_to(root).as_posix()
                raw = path.read_bytes()
                rows[relative] = {
                    "bytes": len(raw),
                    "path": relative,
                    "role": "content",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
        full_count, _full_bytes = replay_v4._validate_full_corpus_manifest_structure_v4(
            full,
            output_rows=rows,
            expected_source_strata=expected_source_strata,
        )
        screen = json.loads(
            (root / materialize_v4.SCREEN_SUBMANIFEST_RELATIVE_PATH_V4).read_text(
                encoding="utf-8"
            )
        )
        assert full_count == full["document_count"]
        assert screen["screen_document_count"] < full_count
        identities.append(
            (
                full["manifest_identity_sha256"],
                screen["submanifest_identity_sha256"],
            )
        )
    assert identities[0] == identities[1]


def test_v4_worker_fails_before_opening_inputs_without_parent_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WEFT1_NETWORK_DISABLED", raising=False)
    monkeypatch.delenv("WEFT1_NETWORK_GUARD_ACTIVE", raising=False)
    missing = tmp_path / "missing"
    with pytest.raises(materialize_v4.CorpusMaterializationV4Error, match="parent offline"):
        materialize_v4.run_production_materialization_worker_v4(
            enumeration_receipt_path=missing / "enumeration.json",
            cache_download_receipt_path=missing / "download.json",
            source_manifest_path=missing / "manifest.json",
            cache_root=missing / "cache",
            fasttext_model_path=missing / "lid.176.bin",
            breakdown_root=missing / "breakdown",
            execution_provenance_path=missing / "provenance.json",
            runtime_build_receipt_path=missing / "runtime.json",
        )


def test_release_section_identity_is_v4_authority_bound() -> None:
    from training.weft1_release import release_manifest_section

    section = release_manifest_section()
    assert replay_v4.execution_authority_v4_bound_sha256(
        materialize_v4.RELEASE_MANIFEST_SECTION_SCHEMA_V4, section
    ) == replay_v4.execution_authority_v4_bound_sha256(
        materialize_v4.RELEASE_MANIFEST_SECTION_SCHEMA_V4,
        json.loads(canonical_json_bytes(section)),
    )


def test_full_pa_v4_cli_writes_one_canonical_parent_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local"
    local.mkdir()
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    durable = durable_root / "run"
    parsed_asset_cache_parent = durable_root / "parsed-asset-cache"
    source_cache = tmp_path / "cache"
    source_cache.mkdir()
    receipt_path = durable / "parent-replay-v4.json"
    expected = replay_v4.ParentReplayVerificationV4(
        status="PASS",
        authoritative=True,
        d1_file_replay_verified=True,
        d2_dedup_replay_verified=True,
        v4_content_profile_verified=True,
        release_binding_verified=True,
        runtime_provenance_verified=True,
        os_network_isolation_verified=True,
        durable_post_write_rehash_verified=True,
        write_enabled_child_policy=replay_v4.V4_WRITE_ENABLED_CHILD_POLICY,
        write_enabled_operation_order=replay_v4.V4_PARENT_LANE_OPERATION_ORDER,
        max_concurrent_write_enabled_children=1,
        input_identity_sha256="1" * 64,
        worker_compatibility_sha256="2" * 64,
        first_child_receipt_sha256="3" * 64,
        second_child_receipt_sha256="4" * 64,
        first_output_root=str(durable / "production-v4-replay-a"),
        second_output_root=str(durable / "production-v4-replay-b"),
        durable_output_parent=str(durable),
        durable_parsed_asset_cache_parent=str(parsed_asset_cache_parent),
        first_parsed_asset_cache_context_sha256="6" * 64,
        second_parsed_asset_cache_context_sha256="7" * 64,
        first_predecessor_parsed_asset_cache_context_sha256=None,
        incident_compatibility_policy_sha256=None,
        incident_compatibility_authority_physical_bytes=None,
        incident_compatibility_authority_physical_sha256=None,
        first_parsed_asset_bridge_receipt_sha256="8" * 64,
        first_parsed_asset_bridge_physical_bytes=101,
        first_parsed_asset_bridge_physical_sha256="9" * 64,
        second_parsed_asset_bridge_receipt_sha256="a" * 64,
        second_parsed_asset_bridge_physical_bytes=102,
        second_parsed_asset_bridge_physical_sha256="b" * 64,
        local_work_parent=str(local),
        evidence_sha256="5" * 64,
    )
    captured = {}
    monkeypatch.setattr(
        cli_v4,
        "attest_production_storage_v3",
        lambda **kwargs: {"storage": "fixture"},
    )

    def fake_verify(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        cli_v4, "verify_production_materialization_replays_v4", fake_verify
    )
    arguments = argparse.Namespace(
        command="full-pa-v4",
        enumeration_receipt=tmp_path / "enumeration.json",
        cache_download_receipt=tmp_path / "download.json",
        source_cache_manifest=tmp_path / "manifest.json",
        source_cache=source_cache,
        fasttext_model=tmp_path / "lid.176.bin",
        runtime_build_receipt=tmp_path / "runtime.json",
        durable_mount_root=tmp_path / "drive",
        durable_storage_marker=tmp_path / "marker.json",
        durable_output_parent=durable,
        durable_parsed_asset_cache_parent=parsed_asset_cache_parent,
        local_work_parent=local,
        receipt_out=receipt_path,
        incident_compatibility_authority_path=tmp_path / "compatibility.json",
        timeout_seconds=123.0,
    )
    payload = cli_v4._run(arguments)
    assert captured["first_output_root"] == durable / "production-v4-replay-a"
    assert captured["second_output_root"] == durable / "production-v4-replay-b"
    assert captured["durable_parsed_asset_cache_parent"] == (
        parsed_asset_cache_parent
    )
    assert captured["timeout_seconds"] == 123.0
    assert captured["incident_compatibility_authority_path"] == (
        tmp_path / "compatibility.json"
    )
    assert payload["receipt_sha256"] == expected.receipt_sha256
    assert receipt_path.read_bytes() == canonical_json_bytes(payload) + b"\n"
    assert not receipt_path.with_name(receipt_path.name + ".partial").exists()
