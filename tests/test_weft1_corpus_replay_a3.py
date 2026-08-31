from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

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
from training.weft1_gtok_contract import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]


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


def test_v4_parent_fills_both_lanes_before_either_materialization() -> None:
    assert replay_v4.V4_PARENT_LANE_OPERATION_ORDER == (
        ("cache_fill", 0),
        ("cache_fill", 1),
        ("materialize", 0),
        ("materialize", 1),
    )


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
        timeout_seconds=123.0,
    )
    payload = cli_v4._run(arguments)
    assert captured["first_output_root"] == durable / "production-v4-replay-a"
    assert captured["second_output_root"] == durable / "production-v4-replay-b"
    assert captured["durable_parsed_asset_cache_parent"] == (
        parsed_asset_cache_parent
    )
    assert captured["timeout_seconds"] == 123.0
    assert payload["receipt_sha256"] == expected.receipt_sha256
    assert receipt_path.read_bytes() == canonical_json_bytes(payload) + b"\n"
    assert not receipt_path.with_name(receipt_path.name + ".partial").exists()
