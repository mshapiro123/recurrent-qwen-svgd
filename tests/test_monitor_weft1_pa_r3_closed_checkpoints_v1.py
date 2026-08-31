from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import monitor_weft1_pa_r3_closed_checkpoints_v1 as monitor
import training.weft1_corpus_materialize_a2 as materializer_v3
from training.weft1_corpus_materialize_a2 import _DurableSourceParseLedgerV3


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _event(
    event_ordinal: int,
    *,
    source_family: str,
    asset_order_ordinal: int,
    source_record_ordinal: int,
    asset_identity_sha256: str,
) -> bytes:
    return (
        json.dumps(
            {
                "asset_order_ordinal": asset_order_ordinal,
                "disposition": "RETAIN",
                "event_ordinal": event_ordinal,
                "event_sha256": _sha(f"event:{source_family}:{event_ordinal}"),
                "source_asset_identity_sha256": asset_identity_sha256,
                "source_family": source_family,
                "source_record_ordinal": source_record_ordinal,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _ledger(tmp_path: Path, source_family: str) -> _DurableSourceParseLedgerV3:
    parse_root = tmp_path / source_family / "output" / "source-parse"
    parse_root.mkdir(parents=True)
    local_root = tmp_path / source_family / "local"
    local_root.mkdir()
    return _DurableSourceParseLedgerV3(
        final_path=parse_root / f"{source_family}.jsonl",
        local_root=local_root,
        source_family=source_family,
        checkpoint_event_cadence=100,
    )


def _write(
    writer: _DurableSourceParseLedgerV3,
    event_ordinal: int,
    *,
    asset_order_ordinal: int,
    source_record_ordinal: int,
    asset_identity_sha256: str,
) -> None:
    writer.write(
        _event(
            event_ordinal,
            source_family=writer.source_family,
            asset_order_ordinal=asset_order_ordinal,
            source_record_ordinal=source_record_ordinal,
            asset_identity_sha256=asset_identity_sha256,
        ),
        event_ordinal=event_ordinal,
        asset_order_ordinal=asset_order_ordinal,
        source_record_ordinal=source_record_ordinal,
    )
    writer.commit_event(event_ordinal)


def test_wikipedia_condition_uses_closed_pair_and_never_reads_partial_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _sha("wikipedia-asset-0")
    writer = _ledger(tmp_path, monitor.WIKIPEDIA_FAMILY)
    with writer:
        for event_ordinal in range(3):
            _write(
                writer,
                event_ordinal,
                asset_order_ordinal=0,
                source_record_ordinal=event_ordinal,
                asset_identity_sha256=identity,
            )
        writer.seal_asset_boundary()

    poisoned_tail = writer.checkpoint_root / "chunk-000001.jsonl.partial"
    poisoned_tail.write_bytes(b"not-json-and-must-not-be-read")
    original_read = monitor._read_source_parse_child_once_v3

    def reject_partial_read(path: Path) -> bytes:
        assert not path.name.endswith(".partial")
        return original_read(path)

    monkeypatch.setattr(
        monitor,
        "_read_source_parse_child_once_v3",
        reject_partial_read,
    )
    monkeypatch.setattr(
        materializer_v3,
        "_read_source_parse_child_once_v3",
        reject_partial_read,
    )
    report = monitor._evaluate_wikipedia(
        writer.checkpoint_root,
        asset_identity_sha256=identity,
        target_event=2,
        target_asset=0,
        target_record=2,
    )

    assert report["status"] == "PASS_CLOSED_CHECKPOINT_REPORT_ONLY_NO_GATE_MINT"
    assert report["event"]["event_ordinal"] == 2
    assert report["checkpoint_chain"]["tail"] == {
        "acceptance_reads_from_tail": False,
        "orphan_chunk_names": [],
        "orphan_receipt_names": [],
        "partial_names": [poisoned_tail.name],
        "status": "UNPUBLISHED_TAIL",
        "unexpected_names": [],
    }


def test_stackedu_condition_requires_direct_asset_four_records_zero_and_one(
    tmp_path: Path,
) -> None:
    identities = tuple(_sha(f"stackedu-asset-{index}") for index in range(5))
    writer = _ledger(tmp_path, monitor.STACKEDU_FAMILY)
    event_ordinal = 0
    with writer:
        for asset_ordinal in range(4):
            _write(
                writer,
                event_ordinal,
                asset_order_ordinal=asset_ordinal,
                source_record_ordinal=0,
                asset_identity_sha256=identities[asset_ordinal],
            )
            event_ordinal += 1
            writer.seal_asset_boundary()
        for record_ordinal in (0, 1):
            _write(
                writer,
                event_ordinal,
                asset_order_ordinal=4,
                source_record_ordinal=record_ordinal,
                asset_identity_sha256=identities[4],
            )
            event_ordinal += 1
        writer.seal_asset_boundary()

    report = monitor._evaluate_stackedu(
        writer.checkpoint_root,
        asset_identity_sha256=identities[4],
    )

    assert report["status"] == "PASS_CLOSED_CHECKPOINT_REPORT_ONLY_NO_GATE_MINT"
    assert [row["source_record_ordinal"] for row in report["events"]] == [0, 1]
    assert [row["event_ordinal"] for row in report["events"]] == [4, 5]
    assert report["direct_python_parser_binding_sha256"] == (
        monitor.STACKEDU_DIRECT_PYTHON_BINDING_SHA256
    )
    with pytest.raises(monitor.CheckpointMonitorError, match="identity drifted"):
        monitor._evaluate_stackedu(
            writer.checkpoint_root,
            asset_identity_sha256=_sha("wrong-asset"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("unexpected_names", ("unexpected",)),
        ("orphan_receipt_names", ("chunk-000001.receipt.json",)),
        ("partial_names", ("chunk-000099.jsonl.partial",)),
    ),
)
def test_noncanonical_checkpoint_tail_fails_closed(field: str, value: object) -> None:
    fields = {
        "receipts": ({"chunk_index": 0},),
        "tail_status": "UNPUBLISHED_TAIL",
        "partial_names": (),
        "orphan_chunk_names": (),
        "orphan_receipt_names": (),
        "unexpected_names": (),
    }
    fields[field] = value

    with pytest.raises(monitor.CheckpointMonitorError):
        monitor._tail_evidence(SimpleNamespace(**fields))


def test_absent_child_root_is_pending_without_creating_it(tmp_path: Path) -> None:
    child_root = tmp_path / "absent-child"
    target = {"asset_identity_sha256": _sha("asset")}

    report = monitor._snapshot(
        child_root,
        wikipedia_target=target,
        stackedu_target=target,
        provenance={},
        captured_wikipedia=None,
    )

    assert report["status"] == "PENDING_REPORT_ONLY_NO_GATE_MINT"
    assert report["conditions"]["wikipedia"]["status"] == (
        "PENDING_CHILD_OUTPUT_ROOT_ABSENT"
    )
    assert not child_root.exists()


def test_receipt_publication_tail_is_reported_but_never_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _sha("wikipedia-publication-race")
    writer = _ledger(tmp_path, monitor.WIKIPEDIA_FAMILY)
    with writer:
        for event_ordinal in range(3):
            _write(
                writer,
                event_ordinal,
                asset_order_ordinal=0,
                source_record_ordinal=event_ordinal,
                asset_identity_sha256=identity,
            )
        writer.seal_asset_boundary()
    orphan_chunk = writer.checkpoint_root / "chunk-000001.jsonl"
    receipt_partial = writer.checkpoint_root / "chunk-000001.receipt.json.partial"
    orphan_chunk.write_bytes(b"open-publication-chunk-must-not-be-read")
    receipt_partial.write_bytes(b"open-receipt-must-not-be-read")
    original_read = monitor._read_source_parse_child_once_v3

    def reject_open_tail(path: Path) -> bytes:
        assert path not in {orphan_chunk, receipt_partial}
        return original_read(path)

    monkeypatch.setattr(
        monitor,
        "_read_source_parse_child_once_v3",
        reject_open_tail,
    )
    monkeypatch.setattr(
        materializer_v3,
        "_read_source_parse_child_once_v3",
        reject_open_tail,
    )

    report = monitor._evaluate_wikipedia(
        writer.checkpoint_root,
        asset_identity_sha256=identity,
        target_event=2,
        target_asset=0,
        target_record=2,
    )

    tail = report["checkpoint_chain"]["tail"]
    assert tail["acceptance_reads_from_tail"] is False
    assert tail["orphan_chunk_names"] == [orphan_chunk.name]
    assert tail["partial_names"] == [receipt_partial.name]


def test_checkpoint_root_replacement_is_retryable_not_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_root = tmp_path / "child"
    parse_root = child_root / "source-parse"
    parse_root.mkdir(parents=True)
    checkpoint_root = parse_root / f".{monitor.WIKIPEDIA_FAMILY}.jsonl.checkpoints"
    checkpoint_root.mkdir()
    identities = iter(((1, 101), (1, 202)))
    monkeypatch.setattr(
        monitor,
        "_checkpoint_root_identity",
        lambda _path: next(identities),
    )
    monkeypatch.setattr(monitor, "_wikipedia_hint", lambda _path: True)

    def replaced(_root: Path, *, asset_identity_sha256: str) -> object:
        del asset_identity_sha256
        raise monitor.CorpusMaterializationError("root changed during validation")

    monkeypatch.setattr(monitor, "_evaluate_wikipedia", replaced)

    report = monitor._source_observation(
        child_root,
        source_family=monitor.WIKIPEDIA_FAMILY,
        target_asset_identity_sha256=_sha("asset"),
    )

    assert report["status"] == (
        "TRANSIENT_CHECKPOINT_ROOT_REPLACEMENT_NO_ACCEPTANCE"
    )


def test_same_root_publication_io_race_retries_without_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_root = tmp_path / "child"
    parse_root = child_root / "source-parse"
    parse_root.mkdir(parents=True)
    checkpoint_root = parse_root / f".{monitor.WIKIPEDIA_FAMILY}.jsonl.checkpoints"
    checkpoint_root.mkdir()
    monkeypatch.setattr(
        monitor,
        "_checkpoint_root_identity",
        lambda _path: (1, 101),
    )
    monkeypatch.setattr(monitor, "_wikipedia_hint", lambda _path: True)

    def publication_race(_root: Path, *, asset_identity_sha256: str) -> object:
        del asset_identity_sha256
        raise FileNotFoundError("atomic publication view changed")

    monkeypatch.setattr(monitor, "_evaluate_wikipedia", publication_race)

    report = monitor._source_observation(
        child_root,
        source_family=monitor.WIKIPEDIA_FAMILY,
        target_asset_identity_sha256=_sha("asset"),
    )

    assert report["status"] == "TRANSIENT_CHECKPOINT_IO_RACE_NO_ACCEPTANCE"


def test_final_ledger_presence_is_metadata_only_and_never_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_root = tmp_path / "child"
    parse_root = child_root / "source-parse"
    parse_root.mkdir(parents=True)
    final_path = parse_root / f"{monitor.WIKIPEDIA_FAMILY}.jsonl"
    final_path.write_bytes(b"must-not-be-read")

    def reject_read(_path: Path) -> bytes:
        raise AssertionError("final ledger or another governed file was read")

    monkeypatch.setattr(monitor, "_read_regular_once", reject_read)
    monkeypatch.setattr(
        monitor,
        "_read_source_parse_child_once_v3",
        reject_read,
    )

    report = monitor._source_observation(
        child_root,
        source_family=monitor.WIKIPEDIA_FAMILY,
        target_asset_identity_sha256=_sha("asset"),
    )

    assert report["status"] == (
        "CHECKPOINT_WINDOW_MISSED_FINAL_PRESENT_NO_ACCEPTANCE"
    )


def test_captured_wikipedia_cannot_cross_child_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_root = tmp_path / "child"
    (child_root / "source-parse").mkdir(parents=True)
    (child_root / "_INCOMPLETE").write_bytes(monitor.INCOMPLETE_BYTES)
    target = {"asset_identity_sha256": _sha("asset")}
    guard = SimpleNamespace(
        assert_current=lambda: None,
        identity={"test": "new-run"},
        identity_sha256=_sha("new-run"),
    )
    captured = {
        "authoritative": False,
        "child_run_identity_sha256": _sha("old-run"),
        "status": "PASS_CLOSED_CHECKPOINT_REPORT_ONLY_NO_GATE_MINT",
    }
    monkeypatch.setattr(
        monitor,
        "_source_observation",
        lambda *_args, **_kwargs: monitor._pending_condition(
            "stackedu_direct_python_asset_4_records_0_1",
            "PENDING_NO_CLOSED_CHECKPOINT_ROOT",
        ),
    )

    with pytest.raises(
        monitor.ChildRunIdentityChangedError,
        match="different child run",
    ):
        monitor._snapshot(
            child_root,
            wikipedia_target=target,
            stackedu_target=target,
            provenance={},
            captured_wikipedia=captured,
            run_guard=guard,
        )


def test_manifest_target_decode_uses_the_single_hashed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source-manifest.json"
    original = b'{"original":true}\n'
    replacement = b'{"replacement":true}\n'
    path.write_bytes(original)
    decoded = SimpleNamespace(receipt_sha256=_sha("receipt"))
    read_count = 0
    original_reader = monitor._read_regular_once

    def counted_read(target: Path) -> bytes:
        nonlocal read_count
        read_count += 1
        return original_reader(target)

    def decode_snapshot(raw: bytes) -> object:
        assert raw == original
        path.write_bytes(replacement)
        return decoded

    monkeypatch.setattr(monitor, "_read_regular_once", counted_read)
    monkeypatch.setattr(monitor, "_decode_source_manifest_snapshot", decode_snapshot)

    raw, manifest = monitor._load_source_manifest_snapshot(
        path,
        expected_physical_sha256=hashlib.sha256(original).hexdigest(),
    )

    assert raw == original
    assert manifest is decoded
    assert path.read_bytes() == replacement
    assert read_count == 1


def test_deployment_closure_requires_clean_exact_monitor_only_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor_commit = "e" * 40

    def clean_git(*arguments: str) -> bytes:
        if arguments == ("rev-parse", "HEAD"):
            return (monitor_commit + "\n").encode("ascii")
        if arguments == ("rev-parse", "--show-toplevel"):
            return (str(monitor.ROOT.resolve()) + "\n").encode("utf-8")
        if arguments == ("rev-list", "--parents", "-n", "1", monitor_commit):
            return (
                monitor_commit + " " + monitor.TARGET_PRODUCTION_COMMIT + "\n"
            ).encode("ascii")
        if arguments[:4] == (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
        ):
            return b"\0".join(
                path.encode("utf-8") for path in sorted(monitor.MONITOR_COMMIT_PATHS)
            ) + b"\0"
        if arguments == (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ):
            return b""
        raise AssertionError(arguments)

    monkeypatch.setattr(monitor, "_git_bytes", clean_git)
    evidence = monitor._deployment_evidence(monitor_commit)
    assert evidence["worktree_clean"] is True
    assert set(evidence["changed_paths"]) == monitor.MONITOR_COMMIT_PATHS

    def dirty_git(*arguments: str) -> bytes:
        if arguments == (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ):
            return b"?? extra.py\0"
        return clean_git(*arguments)

    monkeypatch.setattr(monitor, "_git_bytes", dirty_git)
    with pytest.raises(monitor.CheckpointMonitorError, match="not exactly clean"):
        monitor._deployment_evidence(monitor_commit)

    def extra_path_git(*arguments: str) -> bytes:
        if arguments[:4] == (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
        ):
            return clean_git(*arguments) + b"training/foreign.py\0"
        return clean_git(*arguments)

    monkeypatch.setattr(monitor, "_git_bytes", extra_path_git)
    with pytest.raises(monitor.CheckpointMonitorError, match="exact deployment"):
        monitor._deployment_evidence(monitor_commit)


def test_code_and_runtime_evidence_bind_every_direct_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = {
        path: f"target:{path}\n".encode("utf-8")
        for path in monitor.TARGET_DEPENDENCY_PATHS
    }
    monkeypatch.setattr(
        monitor,
        "_deployment_evidence",
        lambda expected: {"monitor_commit": expected, "worktree_clean": True},
    )

    def target_blob(*arguments: str) -> bytes:
        assert arguments[0] == "show"
        relative = arguments[1].split(":", 1)[1]
        return payloads[relative]

    def governed_read(path: Path) -> bytes:
        try:
            relative = path.relative_to(monitor.ROOT).as_posix()
        except ValueError:
            return b"monitor"
        return payloads.get(relative, b"monitor")

    monkeypatch.setattr(monitor, "_git_bytes", target_blob)
    monkeypatch.setattr(monitor, "_read_regular_once", governed_read)
    code = monitor._code_evidence("e" * 40)
    assert [row["path"] for row in code["production_modules"]] == list(
        monitor.TARGET_DEPENDENCY_PATHS
    )

    attestation = SimpleNamespace(
        dependency_lock_sha256="a" * 64,
        environment_identity_sha256="b" * 64,
        executable_sha256="c" * 64,
    )
    monkeypatch.setattr(monitor, "attest_runtime_v3", lambda: attestation)
    assert monitor._runtime_evidence() == {
        "authoritative_exact_runtime": True,
        "dependency_lock_sha256": "a" * 64,
        "environment_identity_sha256": "b" * 64,
        "executable_sha256": "c" * 64,
    }
