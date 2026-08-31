from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import zstandard

import training.weft1_corpus_pa as production_io
import training.weft1_corpus_materialize_a2 as materializer
import training.weft1_corpus_replay_a2 as replay
import training.weft1_strict_io as strict_io
from training.weft1_corpus_a2 import (
    LanguageIdDecisionV3,
    StableDocumentV3,
    language_id_decision_v3,
)
from training.weft1_corpus_materialize_a2 import (
    CorpusMaterializationError,
    FIXTURE_MODE,
    FULL_POOL_ORDER,
    InjectedSourceStreamV3,
    MaterializationInputV3,
    MaterializationPlanV3,
    MaterializationResultV3,
    MaterializerSourceRecordV3,
    PRODUCTION_MODE,
    materialize_corpus_pa_v3,
    iter_materialized_tokenizer_fit_texts_v3,
    run_production_materialization_worker_v3,
    _EMPTY_STABLE_ID_SCORE_VARIANCE_DIGEST_SHA256_V3,
    _insert_parsed_record_v3,
    _PRODUCTION_WORKER_RECEIPT_SENTINEL,
    _write_production_replay_child_receipt_v3,
    screen_order_digest_v3,
    _Spool,
)
from training.weft1_corpus_replay_a2 import (
    _validate_child_receipt,
    _validate_complete_dedup_metadata,
)
from training.weft1_gtok_a1_contract import SOURCE_FAMILIES
from training.weft1_gtok_contract import GTOK_STRATA


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_event_payload(
    ordinal: int,
    *,
    asset_order_ordinal: int = 0,
    source_record_ordinal: int | None = None,
) -> bytes:
    source_record = ordinal if source_record_ordinal is None else source_record_ordinal
    return (
        json.dumps(
            {
                "asset_order_ordinal": asset_order_ordinal,
                "disposition": "RETAIN",
                "event_ordinal": ordinal,
                "event_sha256": _sha(f"event:{ordinal}"),
                "source_asset_identity_sha256": _sha(
                    f"asset:{asset_order_ordinal}"
                ),
                "source_family": "wikipedia_wikibooks",
                "source_record_ordinal": source_record,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def test_source_parse_checkpoint_hard_kill_preserves_only_closed_chunks(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "durable-output"
    work_root = tmp_path / "local-work"
    parse_root = output_root / "source-parse"
    parse_root.mkdir(parents=True)
    work_root.mkdir()
    (output_root / "_INCOMPLETE").write_bytes(b"P-A incomplete\n")
    final_path = parse_root / "wikipedia_wikibooks.jsonl"
    child = f'''\
import hashlib
import json
import os
from pathlib import Path
from training.weft1_corpus_materialize_a2 import _DurableSourceParseLedgerV3

writer = _DurableSourceParseLedgerV3(
    final_path=Path({str(final_path)!r}),
    local_root=Path({str(work_root)!r}),
    source_family="wikipedia_wikibooks",
    checkpoint_event_cadence=2,
    after_checkpoint=lambda receipt: (
        os._exit(77) if receipt["chunk_index"] == 1 else None
    ),
)
with writer:
    for ordinal in range(6):
        payload = (
            json.dumps(
                {{
                    "asset_order_ordinal": 0,
                    "disposition": "RETAIN",
                    "event_ordinal": ordinal,
                    "event_sha256": hashlib.sha256(
                        f"event:{{ordinal}}".encode("utf-8")
                    ).hexdigest(),
                    "source_asset_identity_sha256": hashlib.sha256(
                        b"asset:0"
                    ).hexdigest(),
                    "source_family": "wikipedia_wikibooks",
                    "source_record_ordinal": ordinal,
                }},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\\n"
        )
        writer.write(
            payload,
            event_ordinal=ordinal,
            asset_order_ordinal=0,
            source_record_ordinal=ordinal,
        )
        writer.commit_event(ordinal)
'''
    killed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    assert killed.returncode == 77

    checkpoint_root = materializer._source_parse_checkpoint_root_v3(final_path)
    recovery = materializer._validate_source_parse_checkpoint_chain_v3(
        checkpoint_root,
        source_family="wikipedia_wikibooks",
    )
    receipts = recovery.receipts
    assert [row["chunk_index"] for row in receipts] == [0, 1]
    assert [row["event_start_ordinal"] for row in receipts] == [0, 2]
    assert [row["event_end_ordinal_exclusive"] for row in receipts] == [2, 4]
    assert recovery.next_event_ordinal == 4
    assert recovery.tail_status == "CLEAN"
    assert all(
        row["progress_semantics"] == "PARSE_PROGRESS_ONLY_NO_RESUME"
        for row in receipts
    )
    assert [row["next_event_ordinal_required"] for row in receipts] == [2, 4]
    assert b"".join(
        (checkpoint_root / str(row["chunk_name"])).read_bytes()
        for row in receipts
    ) == b"".join(_parse_event_payload(ordinal) for ordinal in range(4))
    assert not final_path.exists()
    assert (output_root / "_INCOMPLETE").is_file()
    assert not (output_root / "content-manifest.json").exists()
    assert not (output_root / "d1-ready-manifest.json").exists()
    assert not (output_root / replay.CHILD_RECEIPT_FILENAME).exists()
    assert not tuple(checkpoint_root.glob("*.partial"))


def test_source_parse_checkpoint_completion_reconstructs_legacy_ledger(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "output" / "source-parse" / "wikipedia_wikibooks.jsonl"
    final_path.parent.mkdir(parents=True)
    work_root = tmp_path / "work"
    work_root.mkdir()
    payloads = tuple(
        _parse_event_payload(ordinal, asset_order_ordinal=ordinal // 3)
        for ordinal in range(5)
    )
    writer = materializer._DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=work_root,
        source_family="wikipedia_wikibooks",
        checkpoint_event_cadence=2,
    )
    with writer:
        for ordinal, payload in enumerate(payloads):
            writer.write(
                payload,
                event_ordinal=ordinal,
                asset_order_ordinal=ordinal // 3,
                source_record_ordinal=ordinal,
            )
            writer.commit_event(ordinal)
            if ordinal == 2:
                writer.seal_asset_boundary()
        observed_sha256 = writer.finish()

    expected = b"".join(payloads)
    assert final_path.read_bytes() == expected
    assert observed_sha256 == hashlib.sha256(expected).hexdigest()
    assert not materializer._source_parse_checkpoint_root_v3(final_path).exists()
    assert not tuple(final_path.parent.rglob("*.partial"))


def test_source_parse_checkpoint_rehash_mismatch_fails_before_receipt(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "output" / "source-parse" / "wikipedia_wikibooks.jsonl"
    final_path.parent.mkdir(parents=True)
    work_root = tmp_path / "work"
    work_root.mkdir()
    checkpoint_root = materializer._source_parse_checkpoint_root_v3(final_path)

    def corrupt_replaced_chunk(phase: str) -> None:
        if phase == "chunk_replaced":
            with (checkpoint_root / "chunk-000000.jsonl").open("ab") as handle:
                handle.write(b"corruption")

    writer = materializer._DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=work_root,
        source_family="wikipedia_wikibooks",
        checkpoint_event_cadence=1,
        publication_hook=corrupt_replaced_chunk,
    )
    with writer, pytest.raises(
        CorpusMaterializationError,
        match="failed close/reopen rehash",
    ):
        writer.write(
            _parse_event_payload(0),
            event_ordinal=0,
            asset_order_ordinal=0,
            source_record_ordinal=0,
        )
        writer.commit_event(0)
    assert not final_path.exists()
    assert not tuple(checkpoint_root.glob("*.receipt.json"))


def test_source_parse_checkpoint_replace_failure_never_publishes_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "output" / "source-parse" / "wikipedia_wikibooks.jsonl"
    final_path.parent.mkdir(parents=True)
    work_root = tmp_path / "work"
    work_root.mkdir()
    writer = materializer._DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=work_root,
        source_family="wikipedia_wikibooks",
        checkpoint_event_cadence=1,
    )
    checkpoint_root = materializer._source_parse_checkpoint_root_v3(final_path)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    original_replace = materializer.os.replace
    monkeypatch.setattr(materializer.os, "replace", fail_replace)
    with writer, pytest.raises(OSError, match="injected replace failure"):
        writer.write(
            _parse_event_payload(0),
            event_ordinal=0,
            asset_order_ordinal=0,
            source_record_ordinal=0,
        )
        writer.commit_event(0)
    assert not final_path.exists()
    assert not tuple(checkpoint_root.glob("chunk-*.jsonl"))
    assert not tuple(checkpoint_root.glob("*.receipt.json"))
    assert tuple(checkpoint_root.glob("*.partial"))
    monkeypatch.setattr(materializer.os, "replace", original_replace)
    with pytest.raises(CorpusMaterializationError, match="unpublished tail"):
        writer.finish()


@pytest.mark.parametrize(
    ("phase", "returncode", "expected_orphan_chunks"),
    [
        ("chunk_partial_written", 78, ()),
        ("receipt_partial_written", 79, ("chunk-000001.jsonl",)),
    ],
)
def test_source_parse_checkpoint_recovers_prefix_across_publication_kill(
    tmp_path: Path,
    phase: str,
    returncode: int,
    expected_orphan_chunks: tuple[str, ...],
) -> None:
    final_path = tmp_path / phase / "source-parse" / "wikipedia_wikibooks.jsonl"
    final_path.parent.mkdir(parents=True)
    work_root = tmp_path / f"work-{phase}"
    work_root.mkdir()
    child = f'''\
import hashlib
import json
import os
from pathlib import Path
from training.weft1_corpus_materialize_a2 import _DurableSourceParseLedgerV3

target_phase = {phase!r}
exit_code = {returncode}
target_hits = 0

def publication_hook(observed):
    global target_hits
    if observed == target_phase:
        target_hits += 1
        if target_hits == 2:
            os._exit(exit_code)

writer = _DurableSourceParseLedgerV3(
    final_path=Path({str(final_path)!r}),
    local_root=Path({str(work_root)!r}),
    source_family="wikipedia_wikibooks",
    checkpoint_event_cadence=1,
    publication_hook=publication_hook,
)
with writer:
    for ordinal in range(2):
        payload = (
            json.dumps(
                {{
                    "asset_order_ordinal": 0,
                    "disposition": "RETAIN",
                    "event_ordinal": ordinal,
                    "event_sha256": hashlib.sha256(
                        f"event:{{ordinal}}".encode("utf-8")
                    ).hexdigest(),
                    "source_asset_identity_sha256": hashlib.sha256(b"asset:0").hexdigest(),
                    "source_family": "wikipedia_wikibooks",
                    "source_record_ordinal": ordinal,
                }},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\\n"
        )
        writer.write(
            payload,
            event_ordinal=ordinal,
            asset_order_ordinal=0,
            source_record_ordinal=ordinal,
        )
        writer.commit_event(ordinal)
'''
    killed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    assert killed.returncode == returncode
    recovery = materializer._validate_source_parse_checkpoint_chain_v3(
        materializer._source_parse_checkpoint_root_v3(final_path),
        source_family="wikipedia_wikibooks",
    )
    assert [row["chunk_index"] for row in recovery.receipts] == [0]
    assert recovery.next_event_ordinal == 1
    assert recovery.tail_status == "UNPUBLISHED_TAIL"
    assert recovery.orphan_chunk_names == expected_orphan_chunks
    assert recovery.partial_names
    assert not final_path.exists()


def test_source_parse_checkpoint_cadence_and_asset_boundary_are_exact(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "output" / "source-parse" / "wikipedia_wikibooks.jsonl"
    final_path.parent.mkdir(parents=True)
    work_root = tmp_path / "work"
    work_root.mkdir()
    published: list[Mapping[str, object]] = []
    writer = materializer._DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=work_root,
        source_family="wikipedia_wikibooks",
        checkpoint_event_cadence=3,
        after_checkpoint=lambda receipt: published.append(dict(receipt)),
    )
    with writer:
        ordinal = 0
        for asset, count in ((0, 2), (1, 3), (2, 1)):
            for source_record in range(count):
                published_before_write = len(published)
                writer.write(
                    _parse_event_payload(
                        ordinal,
                        asset_order_ordinal=asset,
                        source_record_ordinal=source_record,
                    ),
                    event_ordinal=ordinal,
                    asset_order_ordinal=asset,
                    source_record_ordinal=source_record,
                )
                assert len(published) == published_before_write
                writer.commit_event(ordinal)
                ordinal += 1
            writer.seal_asset_boundary()
        writer.finish()
    assert [row["event_count"] for row in published] == [2, 3, 1]
    assert [row["first_asset_order_ordinal"] for row in published] == [0, 1, 2]
    assert [row["last_asset_order_ordinal"] for row in published] == [0, 1, 2]
    assert [row["next_event_ordinal_required"] for row in published] == [2, 5, 6]


def test_source_parse_empty_source_finishes_as_exact_empty_legacy_ledger(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "output" / "source-parse" / "wikipedia_wikibooks.jsonl"
    final_path.parent.mkdir(parents=True)
    work_root = tmp_path / "work"
    work_root.mkdir()
    writer = materializer._DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=work_root,
        source_family="wikipedia_wikibooks",
    )
    with writer:
        observed = writer.finish()
    assert final_path.read_bytes() == b""
    assert observed == hashlib.sha256(b"").hexdigest()
    assert not materializer._source_parse_checkpoint_root_v3(final_path).exists()


def test_source_parse_checkpoint_payload_source_is_bound_to_receipt(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "output" / "source-parse" / "wikipedia_wikibooks.jsonl"
    final_path.parent.mkdir(parents=True)
    work_root = tmp_path / "work"
    work_root.mkdir()

    def stop_after_checkpoint(_receipt: Mapping[str, object]) -> None:
        raise RuntimeError("stop after checkpoint")

    writer = materializer._DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=work_root,
        source_family="wikipedia_wikibooks",
        checkpoint_event_cadence=1,
        after_checkpoint=stop_after_checkpoint,
    )
    with writer, pytest.raises(RuntimeError, match="stop after checkpoint"):
        writer.write(
            _parse_event_payload(0),
            event_ordinal=0,
            asset_order_ordinal=0,
            source_record_ordinal=0,
        )
        writer.commit_event(0)
    checkpoint_root = materializer._source_parse_checkpoint_root_v3(final_path)
    chunk = checkpoint_root / "chunk-000000.jsonl"
    changed_payload = json.loads(chunk.read_text(encoding="utf-8"))
    changed_payload["source_family"] = "dolma_web"
    changed_bytes = (
        json.dumps(changed_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    chunk.write_bytes(changed_bytes)
    receipt_path = checkpoint_root / "chunk-000000.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["chunk_bytes"] = len(changed_bytes)
    receipt["chunk_sha256"] = hashlib.sha256(changed_bytes).hexdigest()
    receipt_path.write_bytes(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(CorpusMaterializationError, match="payload fields"):
        materializer._validate_source_parse_checkpoint_chain_v3(
            checkpoint_root,
            source_family="wikipedia_wikibooks",
        )


def test_source_parse_checkpoint_rejects_child_symlink_or_reparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "output" / "source-parse" / "wikipedia_wikibooks.jsonl"
    final_path.parent.mkdir(parents=True)
    work_root = tmp_path / "work"
    work_root.mkdir()
    writer = materializer._DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=work_root,
        source_family="wikipedia_wikibooks",
        checkpoint_event_cadence=1,
        after_checkpoint=lambda _receipt: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    with writer, pytest.raises(RuntimeError, match="stop"):
        writer.write(
            _parse_event_payload(0),
            event_ordinal=0,
            asset_order_ordinal=0,
            source_record_ordinal=0,
        )
        writer.commit_event(0)
    checkpoint_root = materializer._source_parse_checkpoint_root_v3(final_path)
    chunk = checkpoint_root / "chunk-000000.jsonl"
    original = strict_io._is_link_or_reparse
    monkeypatch.setattr(
        strict_io,
        "_is_link_or_reparse",
        lambda path: path == chunk or original(path),
    )
    with pytest.raises(strict_io.StrictPathError, match="symlink/reparse"):
        materializer._validate_source_parse_checkpoint_chain_v3(
            checkpoint_root,
            source_family="wikipedia_wikibooks",
        )


def test_source_parse_uncommitted_event_is_never_published(tmp_path: Path) -> None:
    final_path = tmp_path / "output" / "source-parse" / "wikipedia_wikibooks.jsonl"
    final_path.parent.mkdir(parents=True)
    work_root = tmp_path / "work"
    work_root.mkdir()
    phases: list[str] = []
    writer = materializer._DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=work_root,
        source_family="wikipedia_wikibooks",
        checkpoint_event_cadence=1,
        publication_hook=phases.append,
    )
    with writer:
        writer.write(
            _parse_event_payload(0),
            event_ordinal=0,
            asset_order_ordinal=0,
            source_record_ordinal=0,
        )
        checkpoint_root = materializer._source_parse_checkpoint_root_v3(final_path)
        assert tuple(checkpoint_root.iterdir()) == ()
        assert phases == []
        with pytest.raises(CorpusMaterializationError, match="uncommitted event"):
            writer.seal_asset_boundary()
        with pytest.raises(CorpusMaterializationError, match="uncommitted event"):
            writer.finish()
    assert tuple(checkpoint_root.iterdir()) == ()
    assert not final_path.exists()


def test_source_parse_existing_incomplete_chain_is_not_a_resume_input(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "output" / "source-parse" / "wikipedia_wikibooks.jsonl"
    final_path.parent.mkdir(parents=True)
    first_work_root = tmp_path / "first-work"
    first_work_root.mkdir()
    writer = materializer._DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=first_work_root,
        source_family="wikipedia_wikibooks",
        checkpoint_event_cadence=1,
    )
    with writer:
        writer.write(
            _parse_event_payload(0),
            event_ordinal=0,
            asset_order_ordinal=0,
            source_record_ordinal=0,
        )
        writer.commit_event(0)

    second_work_root = tmp_path / "second-work"
    second_work_root.mkdir()
    with pytest.raises(CorpusMaterializationError, match="must be fresh"):
        materializer._DurableSourceParseLedgerV3(
            final_path=final_path,
            local_root=second_work_root,
            source_family="wikipedia_wikibooks",
            checkpoint_event_cadence=1,
        )
    recovery = materializer._validate_source_parse_checkpoint_chain_v3(
        materializer._source_parse_checkpoint_root_v3(final_path),
        source_family="wikipedia_wikibooks",
    )
    assert recovery.next_event_ordinal == 1
    assert recovery.receipts[0]["resume_authorized"] is False
    assert not final_path.exists()


@pytest.mark.parametrize(
    "tamper",
    ("previous_receipt", "event_range", "unpaired_chunk"),
)
def test_source_parse_tampered_chain_or_unpaired_chunk_fails_closed(
    tmp_path: Path,
    tamper: str,
) -> None:
    final_path = tmp_path / tamper / "source-parse" / "wikipedia_wikibooks.jsonl"
    final_path.parent.mkdir(parents=True)
    work_root = tmp_path / f"work-{tamper}"
    work_root.mkdir()
    writer = materializer._DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=work_root,
        source_family="wikipedia_wikibooks",
        checkpoint_event_cadence=1,
    )
    with writer:
        for ordinal in range(2):
            writer.write(
                _parse_event_payload(ordinal),
                event_ordinal=ordinal,
                asset_order_ordinal=0,
                source_record_ordinal=ordinal,
            )
            writer.commit_event(ordinal)
        checkpoint_root = materializer._source_parse_checkpoint_root_v3(final_path)
        second_receipt_path = checkpoint_root / "chunk-000001.receipt.json"
        if tamper == "unpaired_chunk":
            second_receipt_path.unlink()
            recovery = materializer._validate_source_parse_checkpoint_chain_v3(
                checkpoint_root,
                source_family="wikipedia_wikibooks",
            )
            assert [row["chunk_index"] for row in recovery.receipts] == [0]
            assert recovery.orphan_chunk_names == ("chunk-000001.jsonl",)
            assert recovery.tail_status == "UNPUBLISHED_TAIL"
        else:
            receipt = json.loads(second_receipt_path.read_text(encoding="utf-8"))
            if tamper == "previous_receipt":
                receipt["previous_checkpoint_receipt_sha256"] = "0" * 64
            else:
                receipt["event_start_ordinal"] = 0
            second_receipt_path.write_bytes(
                json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
                + b"\n"
            )
            with pytest.raises(
                CorpusMaterializationError,
                match="receipt chain drifted",
            ):
                materializer._validate_source_parse_checkpoint_chain_v3(
                    checkpoint_root,
                    source_family="wikipedia_wikibooks",
                )
        with pytest.raises(CorpusMaterializationError):
            writer.finish()
    assert not final_path.exists()


def test_production_prepare_reconstructs_legacy_ledgers_and_removes_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "output"
    work_root = tmp_path / "work"
    cache_root = tmp_path / "cache"
    output_root.mkdir()
    work_root.mkdir()
    cache_root.mkdir()
    asset_identity = _sha("wikipedia:asset")
    expected_binding = materializer.PRODUCTION_PARSER_COMPOSITE_IDENTITIES_V3[
        "wikipedia_wikibooks"
    ]
    verified_asset = SimpleNamespace(
        expected=SimpleNamespace(
            source_family="wikipedia_wikibooks",
            relative_path="wikipedia/asset.jsonl",
            asset_identity_sha256=asset_identity,
        )
    )
    parsed = SimpleNamespace(
        parser_binding_sha256=expected_binding,
        raw_document=SimpleNamespace(
            stable_source_record_id=_sha("wikipedia:record"),
            text="retained production-path text",
        ),
        canonical_record=SimpleNamespace(
            asset=SimpleNamespace(asset_identity_sha256=asset_identity),
            retained_byte_count=len(b"retained production-path text"),
            int_score=None,
        ),
    )
    event = SimpleNamespace(
        disposition=materializer.RETAIN,
        event_sha256=_sha("wikipedia:event"),
        record=parsed,
        source_record_ordinal=0,
    )

    monkeypatch.setattr(
        materializer,
        "resolve_production_parser_binding_v3",
        lambda _asset: SimpleNamespace(binding_sha256=expected_binding),
    )
    monkeypatch.setattr(
        materializer,
        "iter_source_asset_events_v3",
        lambda asset, _root, *, binding: (
            (event,) if asset is verified_asset else ()
        ),
    )
    inputs = SimpleNamespace(
        mode=PRODUCTION_MODE,
        verified_cache=SimpleNamespace(assets=(verified_asset,)),
        cache_root=cache_root,
        source_cache_download_receipt=object(),
        source_identity_sha256=_sha("transport"),
    )
    instance = object.__new__(materializer._Materializer)
    instance.inputs = inputs
    instance.output_root = output_root
    instance.work_root = work_root
    instance.source_parse_drop_counts = {
        source: {"empty_text": 0, "invalid_utf8": 0, "quality_lt3": 0}
        for source in SOURCE_FAMILIES
    }
    instance.invalid_utf8_by_source = Counter(
        {source: 0 for source in SOURCE_FAMILIES}
    )
    instance.source_parse_receipts = {}
    instance._production_source_db = None

    instance._prepare_production_sources()
    try:
        parse_root = output_root / "source-parse"
        assert tuple(sorted(path.name for path in parse_root.iterdir())) == tuple(
            sorted(f"{source}.jsonl" for source in SOURCE_FAMILIES)
        )
        expected_payload = materializer.canonical_json_bytes(
            {
                "asset_order_ordinal": 0,
                "disposition": materializer.RETAIN,
                "event_ordinal": 0,
                "event_sha256": event.event_sha256,
                "source_asset_identity_sha256": asset_identity,
                "source_family": "wikipedia_wikibooks",
                "source_record_ordinal": 0,
            }
        ) + b"\n"
        assert (
            parse_root / "wikipedia_wikibooks.jsonl"
        ).read_bytes() == expected_payload
        assert all(
            (parse_root / f"{source}.jsonl").read_bytes() == b""
            for source in SOURCE_FAMILIES
            if source != "wikipedia_wikibooks"
        )
        assert not tuple(parse_root.glob(".*.checkpoints"))
        checkpoint_work_root = work_root / "source-parse-checkpoints"
        assert tuple(checkpoint_work_root.iterdir()) == ()
        assert instance.source_parse_receipts["wikipedia_wikibooks"][
            "parse_event_ledger_sha256"
        ] == hashlib.sha256(expected_payload).hexdigest()
    finally:
        assert instance._production_source_db is not None
        instance._production_source_db.close()


def _parsed_records_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE parsed_records (
          source TEXT NOT NULL,
          stable_source_record_id TEXT NOT NULL,
          source_asset_identity_sha256 TEXT NOT NULL,
          asset_order_ordinal INTEGER NOT NULL,
          asset_record_ordinal INTEGER NOT NULL,
          text BLOB NOT NULL,
          retained_bytes INTEGER NOT NULL,
          int_score INTEGER,
          PRIMARY KEY(source, stable_source_record_id)
        ) WITHOUT ROWID, STRICT
        """
    )
    return connection


def _runtime_build_receipt() -> dict[str, object]:
    return {
        "authoritative": True,
        "evidence": {},
        "receipt_identity_sha256": _sha("runtime-receipt-identity"),
        "schema": replay.RUNTIME_BUILD_RECEIPT_SCHEMA_V1,
        "status": "PASS",
    }


def _global_provenance() -> dict[str, object]:
    lock_sha256 = _sha("dependency-lock")
    executable_sha256 = _sha("python-executable")
    wheel_sha256 = _sha("alpha-wheel")
    root = Path(__file__).resolve().parents[1]
    linkage_core = {
        "executable": {
            "bytes": 1,
            "path": str((root / "python3.11").resolve()),
            "sha256": executable_sha256,
        },
        "libpython_library": {
            "bytes": 1,
            "path": str((root / "libpython3.11.so.1.0").resolve()),
            "sha256": _sha("libpython"),
        },
        "schema": production_io.RUNTIME_LINKAGE_SCHEMA_V3,
        "sqlite_extension": {
            "bytes": 1,
            "path": str((root / "_sqlite3.so").resolve()),
            "sha256": _sha("sqlite-extension"),
        },
        "sqlite_library": {
            "bytes": 1,
            "path": str((root / "libsqlite3.so.0.8.6").resolve()),
            "sha256": _sha("sqlite-library"),
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
            "linkage_identity_sha256": (
                replay.execution_authority_v3_bound_sha256(
                    production_io.RUNTIME_LINKAGE_SCHEMA_V3, linkage_core
                )
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
    storage_identity: dict[str, object] = {
        "durable_marker_sha256": _sha("marker"),
        "durable_mount": {
            "filesystem_type": "fuse.drive",
            "major_minor": "0:99",
            "mount_id": 99,
            "mount_point": str(root),
            "mount_root": "/",
            "mount_source": "drive",
            "parent_mount_id": 1,
            "st_dev": 99,
        },
        "durable_mount_root": str(root),
        "durable_storage_root": str(root),
        "local_mount": {
            "filesystem_type": "overlay",
            "major_minor": "0:98",
            "mount_id": 98,
            "mount_point": str(root.parent),
            "mount_root": "/",
            "mount_source": "overlay",
            "parent_mount_id": 1,
            "st_dev": 98,
        },
        "provider": "google_colab_drive_v1",
        "schema": replay.PRODUCTION_STORAGE_IDENTITY_SCHEMA_V3,
    }
    storage_identity["storage_identity_sha256"] = (
        replay.execution_authority_v3_bound_sha256(
            replay.PRODUCTION_STORAGE_IDENTITY_SCHEMA_V3, storage_identity
        )
    )
    return replay._build_global_execution_provenance_v3(
        environment_payload=environment,
        environment_identity_sha256=(
            replay.execution_authority_v3_bound_sha256(
                "weft1_corpus_execution_environment_v3", environment
            )
        ),
        python_executable_sha256=executable_sha256,
        dependency_lock_sha256=lock_sha256,
        pipeline_components=(
            {"bytes": 1, "logical_name": "materializer", "sha256": _sha("code")},
        ),
        runtime_build_receipt_identity_sha256=_sha("runtime-receipt-identity"),
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
        production_storage_identity=storage_identity,
    )


class _EnglishOnlyClassifier:
    def __init__(self) -> None:
        self.strata: list[str] = []

    def classify(self, document: StableDocumentV3) -> LanguageIdDecisionV3:
        self.strata.append(document.stratum)
        if document.stratum != "general":
            raise AssertionError("language ID escaped the general stratum")
        return language_id_decision_v3(
            document,
            label="__label__en",
            probability=0.99,
        )


def _record(
    source: str,
    ordinal: int,
    text: str,
    *,
    score: int | None = None,
) -> MaterializerSourceRecordV3:
    stratum = {
        "dolma_web": "general",
        "wikipedia_wikibooks": "general",
        "stackedu": "code",
        "finemath_3plus": "mathematics",
        "arxiv": "science_technical",
        "olmocr": "science_technical",
        "fineweb_edu": "general",
    }[source]
    return MaterializerSourceRecordV3(
        source_family=source,
        stratum=stratum,
        source_order_ordinal=ordinal,
        stable_source_record_id=_sha(f"{source}:{ordinal}"),
        source_asset_identity_sha256=_sha(f"asset:{source}"),
        text=text,
        declared_retained_byte_count=len(text.encode("utf-8")),
        int_score=score,
    )


def _fixture_inputs() -> MaterializationInputV3:
    dolma = tuple(_record("dolma_web", i, f"D{i:09d}") for i in range(4))
    records = {
        "dolma_web": dolma,
        "wikipedia_wikibooks": (
            _record("wikipedia_wikibooks", 0, dolma[1].text),
            *tuple(
                _record("wikipedia_wikibooks", i + 1, f"W{i:09d}")
                for i in range(4)
            ),
        ),
        "stackedu": tuple(
            _record("stackedu", i, f"C{i:09d}", score=3) for i in range(4)
        ),
        "finemath_3plus": tuple(
            _record("finemath_3plus", i, f"M{i:09d}", score=10 - i)
            for i in range(4)
        ),
        "arxiv": tuple(_record("arxiv", i, f"A{i:09d}") for i in range(2)),
        "olmocr": tuple(_record("olmocr", i, f"S{i:09d}") for i in range(2)),
        # The initial forty-byte fill contains an exact Dolma duplicate.  The
        # fifth record must therefore be re-deduplicated and selected as top-up.
        "fineweb_edu": (
            _record("fineweb_edu", 0, dolma[0].text, score=10),
            _record("fineweb_edu", 1, "F000000001", score=9),
            _record("fineweb_edu", 2, "F000000002", score=8),
            _record("fineweb_edu", 3, "F000000003", score=7),
            _record("fineweb_edu", 4, "F000000004", score=6),
        ),
    }
    return MaterializationInputV3(
        mode=FIXTURE_MODE,
        streams=tuple(
            InjectedSourceStreamV3(
                source_family=source,
                parser_binding_sha256=_sha(f"parser:{source}"),
                parse_event_ledger_sha256=_sha(f"events:{source}"),
                records=records[source],
            )
            for source in SOURCE_FAMILIES
        ),
        fixture_source_identity_sha256=_sha("fixture-source-v1"),
    )


def _fixture_plan() -> MaterializationPlanV3:
    return MaterializationPlanV3(
        mode=FIXTURE_MODE,
        full_pool_target_bytes=tuple((pool, 40) for pool in FULL_POOL_ORDER),
        training_stratum_target_bytes=(
            ("general", 80),
            ("code", 20),
            ("mathematics", 20),
            ("science_technical", 20),
        ),
        heldout_stratum_target_bytes=(
            ("general", 20),
            ("code", 10),
            ("mathematics", 10),
            ("science_technical", 10),
        ),
        shard_target_bytes=70,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_fixture_two_fresh_runs_are_byte_identical_and_d1_ready(tmp_path: Path) -> None:
    results = []
    classifiers = []
    for run in ("first", "second"):
        classifier = _EnglishOnlyClassifier()
        classifiers.append(classifier)
        results.append(
            materialize_corpus_pa_v3(
                inputs=_fixture_inputs(),
                plan=_fixture_plan(),
                language_classifier=classifier,
                output_root=tmp_path / f"{run}-out",
                work_root=tmp_path / f"{run}-work",
            )
        )

    first, second = results
    assert first.content_identity_sha256 == second.content_identity_sha256
    assert first.d1_ready_manifest_sha256 == second.d1_ready_manifest_sha256
    assert _tree_bytes(first.output_root) == _tree_bytes(second.output_root)
    assert all(set(classifier.strata) == {"general"} for classifier in classifiers)

    content = json.loads((first.output_root / "content-manifest.json").read_text())
    assert content["authoritative_gate_receipts"] == []
    assert content["readiness"] == "NONAUTHORITATIVE_FIXTURE_D1_SHAPE_ONLY"
    assert content["fineweb_topup"]["dedup_dropped_bytes"] == 10
    assert content["fineweb_topup"]["topup_selected_bytes"] == 10
    assert dict(content["dedup_counts"])["DROP_EXACT"] == 1
    assert dict(content["global_exact_duplicate_drops_by_source"])[
        "wikipedia_wikibooks"
    ] == 1
    for receipt in content["source_parse_drop_counts"]:
        assert receipt["stable_id_score_variance_count"] == 0
        assert receipt["stable_id_score_variance_digest_sha256"] == (
            _EMPTY_STABLE_ID_SCORE_VARIANCE_DIGEST_SHA256_V3
        )

    d3 = json.loads((first.output_root / "diagnostics" / "d3.json").read_text())
    d4 = json.loads((first.output_root / "diagnostics" / "d4.json").read_text())
    d5 = json.loads((first.output_root / "diagnostics" / "d5.json").read_text())
    d6 = json.loads((first.output_root / "diagnostics" / "d6.json").read_text())
    assert d3["status"] == "CHECK_PASS_NO_GATE_MINT"
    assert dict(d4["invocation_counts"])["code"] == 0
    assert dict(d4["invocation_counts"])["mathematics"] == 0
    assert dict(d4["invocation_counts"])["science_technical"] == 0
    assert d5["status"] == "CHECK_PASS_NO_GATE_MINT"
    assert d6["document_overlap_count"] == 0
    assert d6["full_corpus_repeated_raw_content_id_count"] == 0
    assert d6["screen_repeated_raw_content_id_count"] == 0
    assert len(d6["consumer_bindings"]) == 8
    assert len(d6["consumer_order_receipts"]) == 2
    assert len(
        {
            row["ordered_document_ids_sha256"]
            for row in d6["consumer_order_receipts"]
        }
    ) == 2
    assert len(
        {row["document_multiset_sha256"] for row in d6["consumer_order_receipts"]}
    ) == 1
    assert d6["near_cluster_receipt"]["qualifying_edge_count"] >= 0
    assert d6["tokenizer_fit_contract"]["allowed_stream"] == "T_ONLY"
    assert d6["tokenizer_fit_contract"]["heldout_admissible"] is False
    fit_input = json.loads(
        (first.output_root / "artifacts" / "tokenizer-fit-input.json").read_text()
    )
    assert fit_input["allowed_stream"] == "T"
    assert all("/t-" in path for path in fit_input["ordered_shard_paths"])
    fit_digest = hashlib.sha256()
    for relative in fit_input["ordered_shard_paths"]:
        with (first.output_root / relative).open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                for line in io.BufferedReader(reader):
                    text = json.loads(line)["text"].encode("utf-8")
                    fit_digest.update(len(text).to_bytes(8, "big"))
                    fit_digest.update(text)
    assert fit_digest.hexdigest() == fit_input["fit_text_stream_sha256"]
    fit_texts = tuple(iter_materialized_tokenizer_fit_texts_v3(first.output_root))
    assert len(fit_texts) == fit_input["document_count"]
    assert sum(len(value.encode("utf-8")) for value in fit_texts) == fit_input[
        "retained_text_bytes"
    ]

    recall = json.loads(
        (first.output_root / "artifacts" / "minhash-recall-audit.json").read_text()
    )
    assert set(recall) == {
        "real_candidate_pairs_at_or_above_threshold",
        "real_dolma_document_count",
        "real_exact_pairs_at_or_above_threshold",
        "real_fineweb_document_count",
        "real_sample_identity_sha256",
        "seed",
        "status",
        "synthetic_cells",
    }
    assert len(recall["synthetic_cells"]) == 6
    assert all(
        set(cell) == {"candidate_count", "exact_jaccard", "pair_count"}
        and set(cell["exact_jaccard"]) == {"denominator", "numerator"}
        for cell in recall["synthetic_cells"]
    )

    descriptor = json.loads(
        (first.output_root / "artifacts" / "d2-evidence-descriptor.json").read_text()
    )
    metadata = descriptor["parent_replay_metadata"]
    assert set(metadata) == {
        "binding_identity_sha256",
        "decision_count",
        "decision_ledger_identity_sha256",
        "decision_ledger_path",
        "decision_ledger_sha256",
        "dropped_bytes",
        "exact_match_rate",
        "minhash_recall_audit_path",
        "minhash_recall_audit_receipt_sha256",
        "minhash_recall_audit_sha256",
        "near_match_rate",
        "schema",
        "selection_ledger_path",
        "selection_ledger_sha256",
        "topup_bytes",
    }
    semantic = hashlib.sha256()
    domain = b"weft1_corpus_dedup_decision_ledger_v3"
    semantic.update(len(domain).to_bytes(8, "big"))
    semantic.update(domain)
    with (first.output_root / metadata["decision_ledger_path"]).open("rb") as handle:
        for line in handle:
            semantic.update(len(line).to_bytes(8, "big"))
            semantic.update(line)
    assert metadata["decision_ledger_identity_sha256"] == semantic.hexdigest()
    assert metadata["decision_count"] == 9
    assert metadata["exact_match_rate"] == {"denominator": 5, "numerator": 1}
    assert metadata["near_match_rate"] == {"denominator": 1, "numerator": 0}
    assert metadata["dropped_bytes"] == 10
    assert metadata["topup_bytes"] == 10
    assert descriptor["gate_minted"] is False
    evidence_rows = tuple(
        {
            "path": metadata[path_key],
            "sha256": metadata[sha_key],
        }
        for path_key, sha_key in (
            ("decision_ledger_path", "decision_ledger_sha256"),
            ("selection_ledger_path", "selection_ledger_sha256"),
            ("minhash_recall_audit_path", "minhash_recall_audit_sha256"),
        )
    )
    assert (
        _validate_complete_dedup_metadata(
            metadata,
            output_root=first.output_root,
            dedup_rows=evidence_rows,
        )
        == metadata
    )
    assert not (first.output_root / "_INCOMPLETE").exists()


def test_production_input_fails_without_authoritative_enumeration_and_cache() -> None:
    streams = tuple(
        InjectedSourceStreamV3(
            source_family=source,
            parser_binding_sha256=_sha(f"parser:{source}"),
            parse_event_ledger_sha256=_sha(f"events:{source}"),
            records=(),
        )
        for source in SOURCE_FAMILIES
    )
    with pytest.raises(CorpusMaterializationError, match="rejects injected"):
        MaterializationInputV3(mode=PRODUCTION_MODE, streams=streams)

    with pytest.raises(CorpusMaterializationError, match="authoritative upstream"):
        MaterializationInputV3(mode=PRODUCTION_MODE)


def test_parsed_native_id_exact_repeat_keeps_first_route_occurrence() -> None:
    connection = _parsed_records_connection()
    try:
        stable_id = _sha("stackedu:native-id")
        first_asset = _sha("asset:first")
        repeated_asset = _sha("asset:repeat")
        first = _insert_parsed_record_v3(
            connection,
            source="stackedu",
            stable_source_record_id=stable_id,
            source_asset_identity_sha256=first_asset,
            asset_order_ordinal=2,
            asset_record_ordinal=11,
            text_bytes=b"identical retained text",
            retained_bytes=len(b"identical retained text"),
            int_score=3,
        )
        repeated = _insert_parsed_record_v3(
            connection,
            source="stackedu",
            stable_source_record_id=stable_id,
            source_asset_identity_sha256=repeated_asset,
            asset_order_ordinal=7,
            asset_record_ordinal=19,
            text_bytes=b"identical retained text",
            retained_bytes=len(b"identical retained text"),
            int_score=3,
        )
        canonical = connection.execute(
            "SELECT source_asset_identity_sha256, asset_order_ordinal, "
            "asset_record_ordinal, int_score FROM parsed_records"
        ).fetchone()
        assert first == ("INSERTED", None)
        assert repeated == ("EXACT_REPEAT", None)
        assert tuple(canonical) == (first_asset, 2, 11, 3)
    finally:
        connection.close()


def test_stackedu_score_only_repeat_is_bound_and_first_score_wins() -> None:
    connection = _parsed_records_connection()
    try:
        stable_id = _sha("stackedu:native-score-variance")
        first_asset = _sha("asset:first-score")
        repeated_asset = _sha("asset:repeated-score")
        text = b"same retained bytes across an upsampled StackEdu row"
        assert _insert_parsed_record_v3(
            connection,
            source="stackedu",
            stable_source_record_id=stable_id,
            source_asset_identity_sha256=first_asset,
            asset_order_ordinal=0,
            asset_record_ordinal=4,
            text_bytes=text,
            retained_bytes=len(text),
            int_score=3,
        ) == ("INSERTED", None)
        classification, evidence = _insert_parsed_record_v3(
            connection,
            source="stackedu",
            stable_source_record_id=stable_id,
            source_asset_identity_sha256=repeated_asset,
            asset_order_ordinal=5,
            asset_record_ordinal=8,
            text_bytes=text,
            retained_bytes=len(text),
            int_score=4,
        )
        canonical = connection.execute(
            "SELECT source_asset_identity_sha256, asset_order_ordinal, "
            "asset_record_ordinal, int_score FROM parsed_records"
        ).fetchone()
        assert classification == "STACKEDU_SCORE_ONLY_VARIANCE"
        assert evidence is not None
        assert evidence["classification"] == classification
        assert evidence["first_retained_text_sha256"] == hashlib.sha256(
            text
        ).hexdigest()
        assert evidence["repeated_retained_text_sha256"] == hashlib.sha256(
            text
        ).hexdigest()
        assert evidence["first_int_score"] == 3
        assert evidence["repeated_int_score"] == 4
        assert tuple(canonical) == (first_asset, 0, 4, 3)
    finally:
        connection.close()


def test_parsed_native_id_content_divergence_fails_with_hash_only_evidence() -> None:
    connection = _parsed_records_connection()
    first_text = b"first private retained text"
    repeated_text = b"different private retained text"
    try:
        stable_id = _sha("stackedu:native-content-divergence")
        _insert_parsed_record_v3(
            connection,
            source="stackedu",
            stable_source_record_id=stable_id,
            source_asset_identity_sha256=_sha("asset:first-content"),
            asset_order_ordinal=1,
            asset_record_ordinal=2,
            text_bytes=first_text,
            retained_bytes=len(first_text),
            int_score=3,
        )
        with pytest.raises(
            CorpusMaterializationError,
            match="stable source record collision evidence=",
        ) as caught:
            _insert_parsed_record_v3(
                connection,
                source="stackedu",
                stable_source_record_id=stable_id,
                source_asset_identity_sha256=_sha("asset:repeated-content"),
                asset_order_ordinal=3,
                asset_record_ordinal=5,
                text_bytes=repeated_text,
                retained_bytes=len(repeated_text),
                int_score=4,
            )
        message = str(caught.value)
        evidence = json.loads(message.split("evidence=", 1)[1])
        assert evidence["classification"] == "CONTENT_DIVERGENCE"
        assert evidence["first_retained_text_sha256"] == hashlib.sha256(
            first_text
        ).hexdigest()
        assert evidence["repeated_retained_text_sha256"] == hashlib.sha256(
            repeated_text
        ).hexdigest()
        assert first_text.decode("ascii") not in message
        assert repeated_text.decode("ascii") not in message
        assert connection.execute(
            "SELECT COUNT(*) FROM parsed_records"
        ).fetchone()[0] == 1
    finally:
        connection.close()


def test_wikipedia_cross_provenance_succeeds_but_same_namespace_divergence_stops(
) -> None:
    connection = _parsed_records_connection()
    try:
        first_identity = _sha(
            "wiki:12:en_simple_wiki_v0-0001.json.gz:2755174"
        )
        cross_provenance_identity = _sha(
            "wiki:12:en_simple_wiki_v0-0000.json.gz:4"
        )
        assert _insert_parsed_record_v3(
            connection,
            source="wikipedia_wikibooks",
            stable_source_record_id=first_identity,
            source_asset_identity_sha256=_sha("wiki:asset:0"),
            asset_order_ordinal=0,
            asset_record_ordinal=2_755_173,
            text_bytes=b"first project page",
            retained_bytes=len(b"first project page"),
            int_score=None,
        ) == ("INSERTED", None)
        assert _insert_parsed_record_v3(
            connection,
            source="wikipedia_wikibooks",
            stable_source_record_id=cross_provenance_identity,
            source_asset_identity_sha256=_sha("wiki:asset:1"),
            asset_order_ordinal=1,
            asset_record_ordinal=3,
            text_bytes=b"second project page",
            retained_bytes=len(b"second project page"),
            int_score=None,
        ) == ("INSERTED", None)
        assert connection.execute(
            "SELECT COUNT(*) FROM parsed_records"
        ).fetchone()[0] == 2

        with pytest.raises(
            CorpusMaterializationError,
            match='"classification":"CONTENT_DIVERGENCE"',
        ):
            _insert_parsed_record_v3(
                connection,
                source="wikipedia_wikibooks",
                stable_source_record_id=first_identity,
                source_asset_identity_sha256=_sha("wiki:asset:repeat"),
                asset_order_ordinal=2,
                asset_record_ordinal=9,
                text_bytes=b"divergent text in the same provenance namespace",
                retained_bytes=len(
                    b"divergent text in the same provenance namespace"
                ),
                int_score=None,
            )
    finally:
        connection.close()


def test_plan_and_screen_order_are_exact_and_fail_closed() -> None:
    plan = _fixture_plan()
    assert tuple(name for name, _ in plan.training_stratum_target_bytes) == GTOK_STRATA
    document_id = _sha("document")
    assert screen_order_digest_v3("general", document_id).hex() == (
        screen_order_digest_v3("general", document_id).hex()
    )
    with pytest.raises(ValueError, match="canonical key order"):
        MaterializationPlanV3(
            mode=FIXTURE_MODE,
            full_pool_target_bytes=tuple(reversed(plan.full_pool_target_bytes)),
            training_stratum_target_bytes=plan.training_stratum_target_bytes,
            heldout_stratum_target_bytes=plan.heldout_stratum_target_bytes,
        )


def test_near_cluster_ids_are_real_registered_components(tmp_path: Path) -> None:
    spool = _Spool(tmp_path / "cluster.sqlite")
    try:
        base = "".join(chr(33 + (index % 90)) for index in range(180))
        documents = (
            StableDocumentV3(
                source="stackedu",
                stratum="code",
                stable_source_record_id=_sha("cluster:left"),
                text=base,
            ),
            StableDocumentV3(
                source="stackedu",
                stratum="code",
                stable_source_record_id=_sha("cluster:right"),
                text=base + "Z",
            ),
        )
        for document in documents:
            spool.select(document)
        receipt = spool.finalize_near_clusters()
        rows = tuple(
            spool.connection.execute(
                "SELECT document_id, cluster_id FROM selected_documents "
                "ORDER BY document_id"
            )
        )
        assert len(rows) == 2
        assert rows[0][1] == rows[1][1] == min(document.document_id for document in documents)
        assert receipt["qualifying_edge_count"] == 1
        assert receipt["cluster_count"] == 1
    finally:
        spool.close()


def test_production_worker_and_child_receipt_reject_fixture_or_no_parent_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_result = materialize_corpus_pa_v3(
        inputs=_fixture_inputs(),
        plan=_fixture_plan(),
        language_classifier=_EnglishOnlyClassifier(),
        output_root=tmp_path / "fixture-out",
        work_root=tmp_path / "fixture-work",
    )
    with pytest.raises(PermissionError, match="concrete worker"):
        _write_production_replay_child_receipt_v3(
            fixture_result,
            runtime_environment_identity_sha256=_sha("runtime-environment"),
        )

    monkeypatch.delenv("WEFT1_NETWORK_DISABLED", raising=False)
    monkeypatch.delenv("WEFT1_NETWORK_GUARD_ACTIVE", raising=False)
    with pytest.raises(CorpusMaterializationError, match="parent offline"):
        run_production_materialization_worker_v3(
            enumeration_receipt_path=tmp_path / "enumeration.json",
            cache_download_receipt_path=tmp_path / "download.json",
            source_manifest_path=tmp_path / "manifest.json",
            cache_root=tmp_path / "cache",
            fasttext_model_path=tmp_path / "lid.176.bin",
            route_manifest_path=tmp_path / "routes.json",
            execution_provenance_path=tmp_path / "provenance.json",
            runtime_build_receipt_path=tmp_path / "runtime-receipt.json",
        )


def test_factory_child_receipt_matches_parent_d1_d2_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = materialize_corpus_pa_v3(
        inputs=_fixture_inputs(),
        plan=_fixture_plan(),
        language_classifier=_EnglishOnlyClassifier(),
        output_root=tmp_path / "worker-out",
        work_root=tmp_path / "worker-work",
    )

    def rewrite(path: Path, value: object) -> None:
        path.write_bytes(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )

    # The test exercises only the private receipt factory shape.  Production
    # worker code is the sole caller of its sentinel and supplies real receipts.
    content_path = fixture.output_root / "content-manifest.json"
    content = json.loads(content_path.read_text(encoding="utf-8"))
    provenance = _global_provenance()
    provenance_path = (
        fixture.output_root / replay.GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_bytes(replay._canonical_json_line(provenance))
    runtime_receipt_path = (
        fixture.output_root / replay.RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1
    )
    runtime_receipt_path.write_bytes(
        replay._canonical_json_line(_runtime_build_receipt())
    )
    content["global"] = {
        "execution_provenance": provenance,
        "execution_provenance_path": (
            replay.GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3
        ),
        "execution_provenance_sha256": hashlib.sha256(
            provenance_path.read_bytes()
        ).hexdigest(),
        "runtime_build_receipt_path": replay.RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1,
        "runtime_build_receipt_sha256": hashlib.sha256(
            runtime_receipt_path.read_bytes()
        ).hexdigest(),
    }
    content["mode"] = PRODUCTION_MODE
    rewrite(content_path, content)
    d1_path = fixture.output_root / "d1-ready-manifest.json"
    d1 = json.loads(d1_path.read_text(encoding="utf-8"))
    d1["mode"] = PRODUCTION_MODE
    rewrite(d1_path, d1)
    production_shape = MaterializationResultV3(
        mode=PRODUCTION_MODE,
        source_identity_sha256=fixture.source_identity_sha256,
        content_identity_sha256=fixture.content_identity_sha256,
        d1_ready_manifest_sha256=hashlib.sha256(d1_path.read_bytes()).hexdigest(),
        output_root=fixture.output_root,
        work_root=fixture.work_root,
    )
    identities = {
        "input": _sha("parent-input"),
        "compatibility": _sha("worker-compatibility"),
        "guard": _sha("network-guard"),
    }
    monkeypatch.setenv("WEFT1_REPLAY_OUTPUT_ROOT", str(fixture.output_root.resolve()))
    monkeypatch.setenv(
        "WEFT1_REPLAY_RECEIPT_PATH",
        str((fixture.output_root / "child-receipt.json").resolve()),
    )
    monkeypatch.setenv("WEFT1_REPLAY_RUN_ID", "receipt-shape-test")
    monkeypatch.setenv("WEFT1_NETWORK_DISABLED", "1")
    monkeypatch.setenv("WEFT1_NETWORK_GUARD_ACTIVE", "1")
    monkeypatch.setenv("WEFT1_REPLAY_INPUT_IDENTITY_SHA256", identities["input"])
    monkeypatch.setenv(
        "WEFT1_REPLAY_WORKER_COMPATIBILITY_SHA256", identities["compatibility"]
    )
    monkeypatch.setenv("WEFT1_NETWORK_GUARD_SHA256", identities["guard"])

    def blocked_connect(_socket: object, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("WEFT-1 parent replay disables network access")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    _write_production_replay_child_receipt_v3(
        production_shape,
        runtime_environment_identity_sha256=str(
            provenance["environment_identity_sha256"]
        ),
        sentinel=_PRODUCTION_WORKER_RECEIPT_SENTINEL,
    )
    raw_receipt = json.loads(
        (fixture.output_root / "child-receipt.json").read_text(encoding="utf-8")
    )
    assert raw_receipt["content_metadata"]["environment_identity_sha256"] == (
        provenance["environment_identity_sha256"]
    )
    verified = _validate_child_receipt(
        output_root=fixture.output_root.resolve(),
        expected_run_id="receipt-shape-test",
        actual_process_id=os.getpid(),
        expected_input_identity_sha256=identities["input"],
        expected_worker_compatibility_sha256=identities["compatibility"],
        expected_network_guard_sha256=identities["guard"],
        stdout=b"",
        stderr=b"",
    )
    assert verified.dedup_evidence_complete is True
    assert verified.dedup_projection_sha256 is not None


def test_production_worker_attests_before_loading_receipts_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training import weft1_corpus_enumeration_a2 as enumeration_module
    from training import weft1_corpus_pa as pa_module

    paths = {
        "enumeration_receipt_path": tmp_path / "enumeration.json",
        "cache_download_receipt_path": tmp_path / "download.json",
        "source_manifest_path": tmp_path / "source-manifest.json",
        "cache_root": tmp_path / "cache",
        "fasttext_model_path": tmp_path / "lid.176.bin",
        "route_manifest_path": tmp_path / "routes.json",
        "execution_provenance_path": tmp_path / "provenance.json",
        "runtime_build_receipt_path": tmp_path / "runtime-receipt.json",
    }
    for path in paths.values():
        if path.suffix:
            path.write_bytes(b"fixture")
        else:
            path.mkdir()
    monkeypatch.setenv("WEFT1_NETWORK_DISABLED", "1")
    monkeypatch.setenv("WEFT1_NETWORK_GUARD_ACTIVE", "1")
    monkeypatch.setenv("WEFT1_REPLAY_OUTPUT_ROOT", str(tmp_path / "fresh-output"))
    calls: list[str] = []

    class Attestation:
        environment_identity_sha256 = _sha("runtime-environment")
        environment_payload = {"runtime": "test"}
        executable_sha256 = _sha("python-executable")
        dependency_lock_sha256 = _sha("dependency-lock")

    def attest() -> Attestation:
        calls.append("attest")
        return Attestation()

    def stop_at_first_receipt_load(*_args: object, **_kwargs: object) -> None:
        calls.append("load-enumeration")
        raise RuntimeError("ordering witness")

    class ModelMustNotOpen:
        def __init__(self, _path: Path) -> None:
            calls.append("open-model")
            raise AssertionError("model opened before receipt loading completed")

    monkeypatch.setattr(pa_module, "attest_runtime_v3", attest)
    monkeypatch.setattr(
        "training.weft1_corpus_materialize_a2.load_canonical_json_object",
        lambda _path: {"fixture": True},
    )
    monkeypatch.setattr(
        "training.weft1_corpus_materialize_a2.validate_global_execution_provenance_v3",
        lambda _value: {
            "dependency_lock_sha256": Attestation.dependency_lock_sha256,
            "environment_identity_sha256": Attestation.environment_identity_sha256,
            "environment_payload": Attestation.environment_payload,
            "python_executable_sha256": Attestation.executable_sha256,
        },
    )
    monkeypatch.setattr(
        "training.weft1_corpus_materialize_a2._validated_runtime_build_receipt_v1",
        lambda *_args, **_kwargs: _runtime_build_receipt(),
    )
    monkeypatch.setattr(
        pa_module,
        "FastTextLanguageIdAdapterV3",
        ModelMustNotOpen,
    )
    monkeypatch.setattr(
        enumeration_module,
        "load_upstream_enumeration_receipt_v3",
        stop_at_first_receipt_load,
    )
    with pytest.raises(RuntimeError, match="ordering witness"):
        run_production_materialization_worker_v3(**paths)
    assert calls == ["attest", "load-enumeration"]
