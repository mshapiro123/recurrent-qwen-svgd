from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
from pathlib import Path
import sqlite3

import pytest

from training.weft1_corpus_a2 import StableDocumentV3
from training.weft1_corpus_streaming_a2 import (
    RECALL_STATUS,
    SERIAL_REFERENCE_MAX_DOCUMENTS,
    SQLITE_RUNTIME_VERSION,
    StreamingDedupError,
    StreamingDedupStoreV3,
    serial_reference_decisions_v3,
)


def _document(source: str, ordinal: int, text: str) -> StableDocumentV3:
    return StableDocumentV3(
        source=source,
        stratum="general",
        stable_source_record_id=hashlib.sha256(
            f"{source}:{ordinal}".encode()
        ).hexdigest(),
        text=text,
    )


def _near_pair() -> tuple[str, str]:
    base = "".join(f"token-{index:03d};" for index in range(80))
    return base, base.replace("token-079", "toker-079")


def _build_store(
    path: Path,
    canonical: tuple[StableDocumentV3, ...],
) -> StreamingDedupStoreV3:
    store = StreamingDedupStoreV3(path)
    for ordinal, document in enumerate(canonical):
        store.append_canonical(document, source_order_ordinal=ordinal)
    store.seal_canonical_phase(expected_source_count=len(canonical))
    return store


def test_runtime_schema_wal_and_bound_metadata(tmp_path: Path) -> None:
    assert sqlite3.sqlite_version == SQLITE_RUNTIME_VERSION
    store = StreamingDedupStoreV3(tmp_path / "state.sqlite3")
    try:
        assert store.state().phase == "DOLMA_CANONICAL"
        assert store._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        metadata = dict(store._connection.execute("SELECT key, value FROM metadata"))
        assert metadata["sqlite_version"] == "3.45.1"
        assert metadata["source_order"] == "dolma_web,fineweb_edu"
    finally:
        store.close()


def test_normalized_exact_collision_confirmation_and_empty_counts(
    tmp_path: Path,
) -> None:
    canonical = (
        _document("dolma_web", 0, "Alpha\t beta"),
        _document("dolma_web", 1, " \n\t "),
    )
    with _build_store(tmp_path / "state.sqlite3", canonical) as store:
        exact = store.append_query(
            _document("fineweb_edu", 0, "Alpha beta"),
            source_order_ordinal=0,
        )
        empty = store.append_query(
            _document("fineweb_edu", 1, "\r\n\t"),
            source_order_ordinal=1,
        )
        assert exact.action == "DROP_EXACT"
        assert exact.exact_jaccard_numerator == 1
        assert exact.canonical_source_record_id == canonical[0].stable_source_record_id
        assert empty.action == "DROP_EMPTY"
        assert store.empty_drop_counts_by_source() == {
            "dolma_web": 1,
            "fineweb_edu": 1,
        }
        row = store._connection.execute(
            "SELECT length(match_sha1), match_length, length(match_sha256) "
            "FROM canonical_documents"
        ).fetchone()
        assert tuple(row) == (20, len(b"Alpha beta"), 32)


def test_near_candidate_is_confirmed_by_exact_jaccard(tmp_path: Path) -> None:
    base, near = _near_pair()
    canonical = (_document("dolma_web", 0, base),)
    with _build_store(tmp_path / "state.sqlite3", canonical) as store:
        decision = store.append_query(
            _document("fineweb_edu", 0, near),
            source_order_ordinal=0,
        )
        assert decision.action == "DROP_NEAR"
        assert decision.exact_jaccard is not None
        assert decision.exact_jaccard >= Fraction(4, 5)
        assert decision.lsh_candidate_count == 1


def test_no_lsh_candidate_keeps_fineweb(tmp_path: Path) -> None:
    canonical = (_document("dolma_web", 0, "canonical text " * 20),)
    with _build_store(tmp_path / "state.sqlite3", canonical) as store:
        decision = store.append_query(
            _document("fineweb_edu", 0, "entirely unrelated query " * 20),
            source_order_ordinal=0,
        )
        assert decision.action == "KEEP_FINEWEB"
        assert decision.canonical_document_id is None
        assert decision.exact_jaccard is None


def test_source_phase_and_contiguous_order_fail_closed(tmp_path: Path) -> None:
    store = StreamingDedupStoreV3(tmp_path / "state.sqlite3")
    try:
        with pytest.raises(StreamingDedupError, match="source order mismatch"):
            store.append_canonical(
                _document("dolma_web", 1, "one"),
                source_order_ordinal=1,
            )
        store.append_canonical(
            _document("dolma_web", 0, "zero"),
            source_order_ordinal=0,
        )
        with pytest.raises(StreamingDedupError, match="source phase mismatch"):
            store.append_query(
                _document("fineweb_edu", 0, "query"),
                source_order_ordinal=0,
            )
        with pytest.raises(StreamingDedupError, match="source count differs"):
            store.seal_canonical_phase(expected_source_count=2)
        store.seal_canonical_phase(expected_source_count=1)
        with pytest.raises(StreamingDedupError, match="source phase mismatch"):
            store.append_canonical(
                _document("dolma_web", 2, "late"),
                source_order_ordinal=1,
            )
    finally:
        store.close()


def test_append_only_tables_reject_update_and_delete(tmp_path: Path) -> None:
    canonical = (_document("dolma_web", 0, "canonical"),)
    with _build_store(tmp_path / "state.sqlite3", canonical) as store:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store._connection.execute(
                "UPDATE dedup_decisions SET action = 'DROP_EMPTY'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store._connection.execute("DELETE FROM canonical_documents")


def test_reopen_resumes_external_cursor_and_ledger_hash(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    canonical = (_document("dolma_web", 0, "canonical"),)
    store = _build_store(path, canonical)
    first_hash = store.decision_ledger_sha256()
    store.close()

    with StreamingDedupStoreV3(path) as reopened:
        assert reopened.state().next_query_ordinal == 0
        reopened.append_query(
            _document("fineweb_edu", 0, "unrelated"),
            source_order_ordinal=0,
        )
        assert reopened.state().next_query_ordinal == 1
        assert reopened.decision_ledger_sha256() != first_hash
        assert [item.decision_ordinal for item in reopened.iter_decisions()] == [0, 1]


def test_serial_reference_equivalence_fixture(tmp_path: Path) -> None:
    base, near = _near_pair()
    canonical = (
        _document("dolma_web", 0, base),
        _document("dolma_web", 1, "exact canonical"),
        _document("dolma_web", 2, " \t\n"),
    )
    queries = (
        _document("fineweb_edu", 0, near),
        _document("fineweb_edu", 1, "exact\tcanonical"),
        _document("fineweb_edu", 2, "unrelated retained text"),
        _document("fineweb_edu", 3, "\r\n"),
    )
    expected = serial_reference_decisions_v3(canonical, queries)
    with _build_store(tmp_path / "state.sqlite3", canonical) as store:
        for ordinal, document in enumerate(queries):
            store.append_query(document, source_order_ordinal=ordinal)
        observed = tuple(store.iter_decisions())
    assert observed == expected
    assert [item.action for item in observed] == [
        "KEEP_CANONICAL",
        "KEEP_CANONICAL",
        "DROP_EMPTY",
        "DROP_NEAR",
        "DROP_EXACT",
        "KEEP_FINEWEB",
        "DROP_EMPTY",
    ]


@pytest.mark.parametrize(
    "dropped_object",
    (
        "dedup_decisions",
        "lsh_bands",
        "canonical_documents",
        "dedup_decisions_reject_update",
    ),
)
def test_reopen_never_repairs_missing_persistent_schema(
    tmp_path: Path, dropped_object: str
) -> None:
    path = tmp_path / "state.sqlite3"
    canonical = (_document("dolma_web", 0, "canonical"),)
    store = _build_store(path, canonical)
    store.close()

    connection = sqlite3.connect(path)
    try:
        object_type = "TRIGGER" if dropped_object.endswith("reject_update") else "TABLE"
        connection.execute(f"DROP {object_type} {dropped_object}")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StreamingDedupError, match="persistent schema differs"):
        StreamingDedupStoreV3(path)


def test_reopen_rejects_cursor_ledger_and_index_cardinality_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    canonical = (_document("dolma_web", 0, "canonical"),)
    store = _build_store(path, canonical)
    store.close()

    connection = sqlite3.connect(path)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'dedup_decisions_reject_delete'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER dedup_decisions_reject_delete")
        connection.execute("DELETE FROM dedup_decisions")
        # Put the trigger back exactly so this test reaches the state invariant.
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StreamingDedupError, match="decision ledger count"):
        StreamingDedupStoreV3(path)


def test_serial_reference_rejects_unbounded_input() -> None:
    canonical = tuple(
        _document("dolma_web", index, str(index))
        for index in range(SERIAL_REFERENCE_MAX_DOCUMENTS + 1)
    )
    with pytest.raises(ValueError, match="bounded fixtures"):
        serial_reference_decisions_v3(canonical, ())


def test_report_only_recall_accounting_hook(tmp_path: Path) -> None:
    base, near = _near_pair()
    canonical = (_document("dolma_web", 0, base),)
    query = _document("fineweb_edu", 0, near)
    sample_identity = hashlib.sha256(b"bounded-recall-sample").hexdigest()
    with _build_store(tmp_path / "state.sqlite3", canonical) as store:
        store.append_recall_pair(
            sample_identity_sha256=sample_identity,
            pair_ordinal=0,
            query_document=query,
            canonical_document_id=canonical[0].document_id,
        )
        accounting = store.recall_accounting()
        assert accounting.status == RECALL_STATUS
        assert accounting.pair_count == 1
        assert accounting.exact_pairs_at_or_above_threshold == 1
        assert accounting.candidate_pairs_at_or_above_threshold == 1
        assert accounting.missed_pairs_at_or_above_threshold == 0
        with pytest.raises(StreamingDedupError, match="order is not contiguous"):
            store.append_recall_pair(
                sample_identity_sha256=sample_identity,
                pair_ordinal=2,
                query_document=query,
                canonical_document_id=canonical[0].document_id,
            )


def test_decision_validation_rejects_false_exact_label(tmp_path: Path) -> None:
    canonical = (_document("dolma_web", 0, "canonical"),)
    with _build_store(tmp_path / "state.sqlite3", canonical) as store:
        decision = store.append_query(
            _document("fineweb_edu", 0, "canonical"),
            source_order_ordinal=0,
        )
        with pytest.raises(ValueError, match="Jaccard 1"):
            replace(
                decision,
                exact_jaccard_numerator=9,
                exact_jaccard_denominator=10,
            )
