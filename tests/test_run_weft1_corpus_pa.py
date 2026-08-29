from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_weft1_corpus_pa as cli
import training.weft1_strict_io as strict_io
from training.weft1_corpus_sources_a2 import (
    SOURCE_CACHE_SCHEMA_V3,
    SOURCE_ROUTE_MANIFEST_SHA256,
    load_exact_source_routes_v3,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_canonical(path: Path, value: object) -> Path:
    path.write_bytes(cli.canonical_json_bytes(value))
    return path


def _cache_fixture(tmp_path: Path) -> tuple[Path, Path]:
    payload = b"already verified source bytes\n"
    cache_root = tmp_path / "cache"
    relative_path = "dolma_web/fixture.jsonl.zst"
    asset_path = cache_root / "dolma_web" / "fixture.jsonl.zst"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(payload)
    route = load_exact_source_routes_v3()[0]
    manifest = {
        "schema": SOURCE_CACHE_SCHEMA_V3,
        "source_route_manifest_sha256": SOURCE_ROUTE_MANIFEST_SHA256,
        "assets": [
            {
                "asset_locator": "data/common_crawl-en-0019/fixture.jsonl.zst",
                "bytes": len(payload),
                "config": route.config,
                "relative_path": relative_path,
                "repository": route.repository,
                "revision": route.revision,
                "sha256": _sha256(payload),
                "source_family": route.source_family,
                "split": route.split,
            }
        ],
    }
    manifest_path = _write_canonical(tmp_path / "source-cache.json", manifest)
    return manifest_path, cache_root


def _fixture(tmp_path: Path) -> Path:
    return _write_canonical(
        tmp_path / "fixture.json",
        {
            "documents": [
                {
                    "source": "dolma_web",
                    "stable_source_record_id": hashlib.sha256(
                        b"fixture-001"
                    ).hexdigest(),
                    "text": "A deterministic first document.",
                },
                {
                    "source": "wikipedia_wikibooks",
                    "stable_source_record_id": hashlib.sha256(
                        b"fixture-002"
                    ).hexdigest(),
                    "text": "A deterministic second document.",
                },
            ],
            "schema": "weft1_corpus_pa_fixture_v3",
            "shard_target_bytes": 256,
            "stratum": "general",
            "stream": "T",
        },
    )


def test_receipt_envelope_is_canonical_and_hashes_exact_payload() -> None:
    receipt = cli._receipt("unit", {"z": 1, "a": "two"})
    raw = cli.canonical_json_bytes(receipt)
    assert raw.endswith(b"\n")
    assert b'"a":"two","z":1' in raw
    assert receipt["receipt_payload_sha256"] == _sha256(
        cli.canonical_json_bytes(receipt["receipt"])
    )


def test_json_reader_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"a":1,"a":2}\n')
    with pytest.raises(cli.PreflightError, match="repeats key"):
        cli._read_json_object(duplicate)
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_bytes(b'{"a":NaN}\n')
    with pytest.raises(cli.PreflightError, match="non-finite"):
        cli._read_json_object(nonfinite)


def test_governed_cli_paths_reject_reparse_and_snapshot_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_canonical(tmp_path / "source.json", {"value": "original"})
    snapshot = tmp_path / "snapshot" / "input.json"
    observed_bytes, observed_sha256 = cli._snapshot_regular_file(
        source,
        snapshot,
        name="fixture",
    )
    source.write_bytes(cli.canonical_json_bytes({"value": "replaced"}))
    assert snapshot.read_bytes() == cli.canonical_json_bytes({"value": "original"})
    assert observed_bytes == len(snapshot.read_bytes())
    assert observed_sha256 == cli.sha256_file(snapshot)

    guarded = tmp_path / "guarded"
    guarded.mkdir()
    (guarded / "input.json").write_bytes(cli.canonical_json_bytes({"ok": True}))
    original_link_check = strict_io._is_link_or_reparse
    monkeypatch.setattr(
        strict_io,
        "_is_link_or_reparse",
        lambda candidate: candidate == guarded or original_link_check(candidate),
    )
    with pytest.raises(cli.PreflightError, match="symlinks/reparse"):
        cli._read_json_object(guarded / "input.json")
    with pytest.raises(cli.PreflightError, match="symlinks/reparse"):
        cli._emit({"status": "fixture"}, guarded / "receipt.json")


def test_contract_verifier_checks_authority_bindings_lock_and_route_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = tmp_path / "A2.md"
    authority.write_bytes(b"fixture authority bytes")
    authority_sha256 = cli.sha256_file(authority)
    monkeypatch.setattr(cli, "EXPECTED_A2_SHA256", authority_sha256)

    bindings = json.loads(cli.DEFAULT_BINDINGS.read_text(encoding="utf-8"))
    bindings["authority_sha256"] = authority_sha256
    bindings_path = _write_canonical(tmp_path / "bindings.json", bindings)
    evidence = cli.verify_contracts(
        authority_path=authority,
        bindings_path=bindings_path,
        dependency_lock_path=cli.DEFAULT_DEPENDENCY_LOCK,
        route_manifest_path=cli.DEFAULT_ROUTE_MANIFEST,
        expected_bindings_sha256=cli.sha256_file(bindings_path),
    )
    assert evidence["verified"] is True
    assert evidence["authority_sha256"] == authority_sha256
    assert evidence["dependency_lock_sha256"] == (
        "bccb8e5b58b5e8fa9eee367fe9c26f59053fff5b7fadf81f23f96b83d1531860"
    )
    assert evidence["route_manifest_receipt_sha256"] == (
        SOURCE_ROUTE_MANIFEST_SHA256
    )

    authority.write_bytes(b"drift")
    with pytest.raises(cli.PreflightError, match="authority byte hash"):
        cli.verify_contracts(
            authority_path=authority,
            bindings_path=bindings_path,
            dependency_lock_path=cli.DEFAULT_DEPENDENCY_LOCK,
            route_manifest_path=cli.DEFAULT_ROUTE_MANIFEST,
            expected_bindings_sha256=cli.sha256_file(bindings_path),
        )


def test_checked_in_bindings_pin_is_current() -> None:
    assert cli.sha256_file(cli.DEFAULT_BINDINGS) == cli.EXPECTED_BINDINGS_SHA256


def test_environment_receipt_observe_only_omits_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "do-not-disclose.json")
    monkeypatch.setenv("ACCESS_TOKEN", "secret-value")
    receipt = cli.environment_receipt(require_match=False)
    encoded = cli.canonical_json_bytes(receipt)
    assert receipt["credentials_inspected"] is False
    assert receipt["environment_variables_enumerated"] is False
    assert receipt["authoritative"] is False
    assert b"secret-value" not in encoded
    assert b"do-not-disclose" not in encoded


def test_observe_only_and_label_only_receipts_never_claim_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "environment_receipt",
        lambda *, require_match: {
            "authoritative": False,
            "matches_bound_environment": False,
        },
    )
    environment = cli._dispatch(
        cli.build_parser().parse_args(["environment-receipt", "--observe-only"])
    )
    assert environment["receipt"]["status"] == "OBSERVED"

    colab = cli._dispatch(
        cli.build_parser().parse_args(
            [
                "colab-preflight",
                "--workspace-label",
                cli.EXPECTED_COLAB_WORKSPACE_LABEL,
                "--subscription-label",
                cli.EXPECTED_COLAB_SUBSCRIPTION_LABEL,
                "--surface-label",
                cli.EXPECTED_COLAB_SURFACE_LABEL,
                "--runtime-label",
                "externally supplied",
            ]
        )
    )
    assert colab["receipt"]["status"] == "OBSERVED"
    assert colab["receipt"]["evidence"]["authoritative"] is False
    assert colab["receipt"]["evidence"]["verified"] is False


def test_source_cache_verification_delegates_to_exact_offline_verifier(
    tmp_path: Path,
) -> None:
    manifest, cache_root = _cache_fixture(tmp_path)
    receipt = cli.verify_source_cache_manifest(
        manifest_path=manifest,
        cache_root=cache_root,
    )
    assert receipt["schema"] == "weft1_verified_local_source_cache_v3"
    assert receipt["asset_count"] == 1
    assert receipt["network_used"] is False
    assert receipt["verified"] is True

    (cache_root / "dolma_web" / "fixture.jsonl.zst").write_bytes(b"tampered")
    with pytest.raises(cli.PreflightError, match="verification failed"):
        cli.verify_source_cache_manifest(
            manifest_path=manifest,
            cache_root=cache_root,
        )


def test_two_fixture_replays_use_parent_observed_processes_and_files(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = cli.run_two_fixture_replays(
        fixture_path=fixture,
        output_parent=tmp_path / "replays",
        use_builtin_test_backend=True,
    )
    assert evidence["network_disabled"] is False
    assert evidence["network_isolation_kind"] == "python_socket_guard_only"
    assert evidence["distinct_process_ids"] is True
    assert evidence["distinct_output_roots"] is True
    assert evidence["production_io_hook_used"] is False
    assert evidence["status"] == "CHECK_PASS"
    assert evidence["authoritative"] is False
    assert evidence["d1_file_replay_verified"] is True
    assert evidence["d2_dedup_replay_verified"] is False
    assert len(set(evidence["worker_process_ids"])) == 2
    assert evidence["output_tree"]
    assert len(evidence["output_tree_sha256"]) == 64
    parent_receipt = evidence["parent_replay_receipt"]
    assert parent_receipt["status"] == "CHECK_PASS"
    assert parent_receipt["authoritative"] is False
    assert parent_receipt["first_process_id"] == evidence["worker_process_ids"][0]
    assert parent_receipt["second_process_id"] == evidence["worker_process_ids"][1]
    assert len(evidence["parent_replay_receipt_sha256"]) == 64
    assert all(
        row["path"] != "child-receipt.json" for row in evidence["output_tree"]
    )
    assert (tmp_path / "replays" / "replay-a").is_dir()
    assert (tmp_path / "replays" / "replay-b").is_dir()
    for replay_name, expected_pid in zip(
        ("replay-a", "replay-b"), evidence["worker_process_ids"], strict=True
    ):
        receipt_path = (
            tmp_path / "replays" / replay_name / "child-receipt.json"
        )
        child_receipt = json.loads(receipt_path.read_bytes())
        assert receipt_path.read_bytes() == cli.canonical_json_bytes(child_receipt)
        assert child_receipt["process_id"] == expected_pid
        assert child_receipt["dedup_evidence_complete"] is False
        assert child_receipt["dedup_metadata"] is None
        assert child_receipt["network_guard_active"] is True
        assert child_receipt["network_probe"] == "python_socket_connect_blocked"
    with pytest.raises(cli.PreflightError, match="must not already exist"):
        cli.run_two_fixture_replays(
            fixture_path=fixture,
            output_parent=tmp_path / "replays",
            use_builtin_test_backend=True,
        )


def test_replay_cli_envelope_preserves_nonauthoritative_check_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(
        cli,
        "run_two_fixture_replays",
        lambda **_kwargs: {
            "authoritative": False,
            "status": "CHECK_PASS",
        },
    )
    args = cli.build_parser().parse_args(
        [
            "replay-fixture",
            "--fixture",
            str(fixture),
            "--output-parent",
            str(tmp_path / "replays"),
        ]
    )
    envelope = cli._dispatch(args)
    assert envelope["receipt"]["status"] == "CHECK_PASS"
    assert envelope["receipt"]["evidence"]["authoritative"] is False


def test_colab_preflight_uses_only_external_pharma_labels() -> None:
    evidence = cli.colab_preflight(
        workspace_label="Pharma Initiatives",
        subscription_label="Pro+",
        surface_label="in-app browser",
        runtime_label="runtime-ui-label-01",
    )
    assert evidence["verified"] is False
    assert evidence["authoritative"] is False
    assert evidence["runtime_label_source"] == "external_cli_argument"
    assert evidence["credentials_read"] is False
    assert evidence["accelerator_inspected"] is False
    assert evidence["gpu_requested"] is False
    with pytest.raises(cli.PreflightError, match="labels do not match"):
        cli.colab_preflight(
            workspace_label="personal",
            subscription_label="Pro+",
            surface_label="in-app browser",
            runtime_label="runtime-ui-label-01",
        )


def test_full_pa_guard_refuses_missing_inputs_and_never_executes(
    tmp_path: Path,
) -> None:
    with pytest.raises(cli.PreflightError, match="required inputs missing"):
        cli.full_pa_guard(
            source_cache=None,
            source_cache_manifest=None,
            route_manifest=None,
            output_path=None,
        )
    manifest, cache_root = _cache_fixture(tmp_path)
    evidence = cli.full_pa_guard(
        source_cache=cache_root,
        source_cache_manifest=manifest,
        route_manifest=cli.DEFAULT_ROUTE_MANIFEST,
        output_path=tmp_path / "fresh-output",
    )
    assert evidence["all_required_paths_supplied"] is True
    assert evidence["execution_enabled"] is False
    assert evidence["execution_status"] == "NOT_EXECUTED_INITIAL_READ_ONLY_REVISION"
    assert not (tmp_path / "fresh-output").exists()


def test_full_pa_executes_only_the_fixed_production_parent_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @dataclass(frozen=True)
    class Parent:
        authoritative: bool = True
        status: str = "PASS"
        d1_file_replay_verified: bool = True
        d2_dedup_replay_verified: bool = True
        network_isolation_kind: str = "linux_unshare_net_v1"
        production_profile_verified: bool = True
        receipt_sha256: str = "a" * 64

    captured: dict[str, object] = {}

    def fake_parent(**kwargs: object) -> Parent:
        captured.update(kwargs)
        return Parent()

    monkeypatch.setattr(
        cli, "verify_production_materialization_replays_v3", fake_parent
    )
    named = {
        name: tmp_path / name
        for name in (
            "authority.md",
            "enumeration.json",
            "download.json",
            "source-manifest.json",
            "cache",
            "model.bin",
        )
    }
    evidence = cli.run_full_pa_replays(
        authority_path=named["authority.md"],
        enumeration_receipt_path=named["enumeration.json"],
        cache_download_receipt_path=named["download.json"],
        source_cache_manifest_path=named["source-manifest.json"],
        source_cache=named["cache"],
        fasttext_model_path=named["model.bin"],
        output_parent=tmp_path / "production-output",
    )
    assert evidence["status"] == "PASS"
    assert evidence["authoritative"] is True
    assert evidence["production_profile_verified"] is True
    assert captured["python_executable"] == Path(sys.executable)
    assert captured["first_output_root"] == (
        tmp_path / "production-output" / "production-replay-a"
    )
    assert captured["second_output_root"] == (
        tmp_path / "production-output" / "production-replay-b"
    )


def test_full_pa_receipt_cannot_mutate_governed_replay_tree_after_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden_run(**_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {"status": "PASS"}

    monkeypatch.setattr(cli, "run_full_pa_replays", forbidden_run)
    output_parent = tmp_path / "production-output"
    args = cli.build_parser().parse_args(
        [
            "full-pa",
            "--authority",
            str(tmp_path / "authority.md"),
            "--enumeration-receipt",
            str(tmp_path / "enumeration.json"),
            "--cache-download-receipt",
            str(tmp_path / "download.json"),
            "--source-cache",
            str(tmp_path / "cache"),
            "--source-cache-manifest",
            str(tmp_path / "source-manifest.json"),
            "--fasttext-model",
            str(tmp_path / "model.bin"),
            "--output-parent",
            str(output_parent),
            "--receipt-out",
            str(output_parent / "parent-receipt.json"),
        ]
    )
    with pytest.raises(cli.PreflightError, match="outside the governed replay tree"):
        cli._dispatch(args)
    assert called is False


def test_cli_prints_one_canonical_receipt_and_never_reads_credentials(
    tmp_path: Path,
) -> None:
    script = Path(cli.__file__).resolve()
    environment = os.environ.copy()
    environment["GOOGLE_APPLICATION_CREDENTIALS"] = "must-not-appear.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "colab-preflight",
            "--workspace-label",
            "Pharma Initiatives",
            "--subscription-label",
            "Pro+",
            "--surface-label",
            "in-app browser",
            "--runtime-label",
            "externally-visible-runtime-label",
        ],
        cwd=cli.ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    parsed = json.loads(completed.stdout)
    assert completed.stdout == cli.canonical_json_bytes(parsed)
    assert completed.stderr == b""
    assert b"must-not-appear" not in completed.stdout
    assert parsed["receipt"]["status"] == "OBSERVED"
    assert parsed["receipt"]["evidence"]["authoritative"] is False


def test_cli_failure_is_canonical_and_fail_closed() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(cli.__file__).resolve()), "full-pa"],
        cwd=cli.ROOT,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""
    parsed = json.loads(completed.stderr)
    assert completed.stderr == cli.canonical_json_bytes(parsed)
    assert parsed["receipt"]["status"] == "FAIL"
    assert parsed["receipt"]["evidence"]["failed_closed"] is True
