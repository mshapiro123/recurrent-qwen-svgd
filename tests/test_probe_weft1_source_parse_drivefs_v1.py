from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import scripts.probe_weft1_source_parse_drivefs_v1 as probe


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_weft1_source_parse_drivefs_v1.py"
SOURCE_FAMILY = "wikipedia_wikibooks"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _event(
    ordinal: int,
    *,
    asset_order_ordinal: int,
    source_record_ordinal: int,
) -> bytes:
    return _canonical(
        {
            "asset_order_ordinal": asset_order_ordinal,
            "disposition": "RETAIN",
            "event_ordinal": ordinal,
            "event_sha256": _sha(f"probe-event:{ordinal}".encode()),
            "source_asset_identity_sha256": _sha(
                f"probe-asset:{asset_order_ordinal}".encode()
            ),
            "source_family": SOURCE_FAMILY,
            "source_record_ordinal": source_record_ordinal,
        }
    )


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPT), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60.0,
    )


def test_two_phase_local_dry_run_is_bound_nonreusable_and_fail_closed(
    tmp_path: Path,
) -> None:
    durable = tmp_path / "durable"
    local = tmp_path / "local"
    stage_path = tmp_path / "stage.json"
    final_path = tmp_path / "final.json"

    published = _run(
        "publish",
        "--durable-root",
        str(durable),
        "--local-root",
        str(local),
        "--stage-manifest-out",
        str(stage_path),
        "--local-dry-run",
    )
    assert published.returncode == 0, published.stderr
    publish_result = json.loads(published.stdout)
    assert _canonical(publish_result).decode() == published.stdout
    assert publish_result["authoritative"] is False
    assert publish_result["gate_minted"] is False
    assert publish_result["status"] == (
        "PUBLISHED_AWAITING_PROVIDER_REMOUNT_NO_GATE_MINT"
    )
    stage_raw = stage_path.read_bytes()
    assert _sha(stage_raw) == publish_result["publication"]["sha256"]
    stage_envelope = json.loads(stage_raw)
    assert _canonical(stage_envelope) == stage_raw
    stage = stage_envelope["stage_manifest"]
    assert stage["authoritative"] is False
    assert stage["gate_minted"] is False
    assert stage["probe_outputs_reusable"] is False
    assert stage["fresh_replay_required_for_corpus_work"] is True
    assert stage["storage_before_provider_remount"]["classification"] == (
        "LOCAL_TMP_SIMULATION_NO_DRIVE_CLAIM"
    )
    assert stage["storage_before_provider_remount"]["mountinfo_verified"] is False
    assert stage["runtime"]["authoritative_exact_runtime"] is False
    assert stage["killed_ledger"]["termination_method"] == "subprocess.Popen.kill"
    assert stage["killed_ledger"]["durable_max_event_ordinal"] == 1
    assert stage["killed_ledger"]["final_ledger_present"] is False
    assert stage["killed_ledger"]["fresh_replay_required"] is True
    assert stage["success_ledger"]["legacy_stream_exact_match"] is True

    killed_final = (
        durable
        / "killed-output"
        / "source-parse"
        / f"{SOURCE_FAMILY}.jsonl"
    )
    assert not killed_final.exists()
    assert (durable / "killed-output" / "_INCOMPLETE").read_bytes() == (
        b"P-A incomplete\n"
    )
    success_final = (
        durable
        / "success-output"
        / "source-parse"
        / f"{SOURCE_FAMILY}.jsonl"
    )
    expected_success = b"".join(
        _event(
            ordinal,
            asset_order_ordinal=0 if ordinal < 3 else 1,
            source_record_ordinal=ordinal if ordinal < 3 else ordinal - 3,
        )
        for ordinal in range(5)
    )
    assert success_final.read_bytes() == expected_success
    assert _sha(expected_success) == stage["success_ledger"]["final_sha256"]

    expected_files = {
        "killed-output/_INCOMPLETE",
        (
            "killed-output/source-parse/"
            f".{SOURCE_FAMILY}.jsonl.checkpoints/chunk-000000.jsonl"
        ),
        (
            "killed-output/source-parse/"
            f".{SOURCE_FAMILY}.jsonl.checkpoints/"
            "chunk-000000.receipt.json"
        ),
        f"success-output/source-parse/{SOURCE_FAMILY}.jsonl",
    }
    assert {
        row["path"] for row in stage["durable_tree_projection"]["files"]
    } == expected_files

    wrong_hash = "0" * 64
    rejected = _run(
        "verify",
        "--durable-root",
        str(durable),
        "--stage-manifest",
        str(stage_path),
        "--expected-stage-manifest-sha256",
        wrong_hash,
        "--final-receipt-out",
        str(final_path),
        "--local-dry-run",
    )
    assert rejected.returncode == 2
    failure = json.loads(rejected.stderr)
    assert failure["status"] == "FAIL_CLOSED_NO_GATE_MINT"
    assert failure["authoritative"] is False
    assert failure["gate_minted"] is False
    assert not final_path.exists()

    verified = _run(
        "verify",
        "--durable-root",
        str(durable),
        "--stage-manifest",
        str(stage_path),
        "--expected-stage-manifest-sha256",
        publish_result["publication"]["sha256"],
        "--final-receipt-out",
        str(final_path),
        "--local-dry-run",
    )
    assert verified.returncode == 0, verified.stderr
    verify_result = json.loads(verified.stdout)
    assert verify_result["authoritative"] is False
    assert verify_result["gate_minted"] is False
    assert verify_result["status"] == (
        "PROVIDER_REMOUNT_VERIFIED_NO_GATE_MINT_NONREUSABLE"
    )
    final_raw = final_path.read_bytes()
    assert _sha(final_raw) == verify_result["publication"]["sha256"]
    final_envelope = json.loads(final_raw)
    assert _canonical(final_envelope) == final_raw
    final = final_envelope["final_receipt"]
    assert final["authoritative"] is False
    assert final["gate_minted"] is False
    assert final["probe_outputs_reusable"] is False
    assert final["fresh_replay_required_for_corpus_work"] is True
    transition = final["mount_transition"]
    assert transition["kernel_mount_id_changed"] is True
    assert transition["provider_barrier_classification"] == "SIMULATED_LOCAL_ONLY"
    assert transition["prior_mount"]["mount_id"] != transition["current_mount"][
        "mount_id"
    ]
    assert final["stage_manifest"]["sha256"] == publish_result["publication"][
        "sha256"
    ]
    assert final["verification"]["durable_object_reopen_passes"] == 2
    assert final["verification"]["every_durable_object_rehashed"] is True

    final_sha256 = _sha(final_raw)
    duplicate = _run(
        "verify",
        "--durable-root",
        str(durable),
        "--stage-manifest",
        str(stage_path),
        "--expected-stage-manifest-sha256",
        publish_result["publication"]["sha256"],
        "--final-receipt-out",
        str(final_path),
        "--local-dry-run",
    )
    assert duplicate.returncode == 2
    assert _sha(final_path.read_bytes()) == final_sha256


def test_exact_runtime_evidence_is_json_normalized_before_stage_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested_attestation_payload = {
        "locked_distributions": (
            {
                "artifact_sha256s": ("a" * 64, "b" * 64),
                "distribution": "example",
            },
        ),
        "required_environment": (("TOKENIZERS_PARALLELISM", "false"),),
    }
    monkeypatch.setattr(
        probe,
        "attest_runtime_v3",
        lambda: SimpleNamespace(
            dependency_lock_sha256="c" * 64,
            environment_identity_sha256="d" * 64,
            environment_payload=nested_attestation_payload,
            executable_sha256="e" * 64,
        ),
    )

    current = probe._runtime_evidence(local_dry_run=False)
    stage_after_serialize_reload = json.loads(_canonical(current))

    assert current == stage_after_serialize_reload
    assert isinstance(current["environment_payload"]["locked_distributions"], list)
    assert isinstance(current["environment_payload"]["required_environment"], list)
    assert isinstance(
        current["environment_payload"]["required_environment"][0], list
    )

    drifted_stage = json.loads(_canonical(current))
    drifted_stage["environment_payload"]["required_environment"][0][1] = "true"
    assert current != drifted_stage
