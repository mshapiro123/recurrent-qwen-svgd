from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
from pathlib import Path

import pytest

from training.weft1_corpus_a2 import (
    A2_CAMPAIGN_ROOT_SEED,
    A2_CALIBRATION_MEASURED_STEPS,
    A2_CALIBRATION_STEPS_MAXIMUM,
    A2_CALIBRATION_WARMUP_STEPS,
    A2_CHARGED_ATTEMPT_STATUSES,
    A2_DEDUP_SEED,
    A2_LANGUAGE_ID_BINDING,
    A2_MATCH_NORMALIZATION_BINDING,
    A2_MINHASH_BINDING,
    A2_PIPELINE_SEEDS,
    A2_STREAM_PRECEDENCE,
    A2_TRIPWIRE_A100_SECONDS,
    A2_ZSTD_CODEC_BINDING,
    CorpusContentManifestV3,
    CorpusGateReceiptV3,
    DocumentOccurrenceV3,
    EMPTY_MATCH_NORMALIZATION_DISPOSITION,
    GTOK_AMENDMENT_A2_SHA256,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
    JsonlZstdShardIdentityV3,
    MINHASH_RECALL_JACCARD_LEVELS,
    MinHashRecallAuditV3,
    MinHashSyntheticRecallCellV3,
    ProcessAttestationV3,
    ReplayRunReceiptV3,
    SelectionCandidateV3,
    StableDocumentV3,
    TripwireProjectionV3,
    byte_shingles_v3,
    canonical_jsonl_record_bytes_v3,
    dedup_match_input_v3,
    greedy_first_fit_v3,
    greedy_training_then_heldout_v3,
    ideal_lsh_candidate_probability_v3,
    is_exact_duplicate_v3,
    language_id_decision_from_predictions_v3,
    language_id_decision_v3,
    language_scoring_bytes_v3,
    lsh_band_keys_v3,
    minhash_signature_v3,
    normalize_match_text,
    normalized_match_bytes,
    pipeline_seed,
    select_dedup_winner_v3,
    tripwire_preflight_v3,
    tripwire_runtime_v3,
    validate_independent_replays_v3,
)
from training.weft1_gtok_a1_contract import a1_contract_snapshot_sha256
from training.weft1_gtok_contract import GTOK_EXECUTION_AUTHORITY_CHAIN_V2


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _document(
    record_id: str,
    text: str,
    *,
    source: str = "dolma_web",
    stratum: str = "general",
) -> StableDocumentV3:
    return StableDocumentV3(
        source=source,
        stratum=stratum,
        stable_source_record_id=_hash(record_id),
        text=text,
    )


def _candidate(
    record_id: str,
    text: str,
    *,
    asset: str | None = None,
    ordinal: int = 0,
    cluster: str | None = None,
) -> SelectionCandidateV3:
    occurrence = DocumentOccurrenceV3(
        document=_document(record_id, text),
        source_asset_sha256=_hash(asset or f"asset-{record_id}"),
        source_record_ordinal=ordinal,
    )
    return SelectionCandidateV3(
        occurrence=occurrence,
        cluster_id=_hash(cluster or f"cluster-{record_id}"),
    )


def _shard(*, logical: str = "logical", compressed: str = "compressed") -> JsonlZstdShardIdentityV3:
    return JsonlZstdShardIdentityV3(
        relative_path="general/h-00000.jsonl.zst",
        record_count=2,
        retained_text_bytes=11,
        logical_jsonl_sha256=_hash(logical),
        logical_jsonl_bytes=221,
        zstd_sha256=_hash(compressed),
        zstd_bytes=130,
    )


def _manifest(
    run_id: str,
    *,
    created: str,
    root: str,
    host: str = "test-host",
    pid: int = 101,
    shard: JsonlZstdShardIdentityV3 | None = None,
) -> CorpusContentManifestV3:
    return CorpusContentManifestV3(
        run_id=run_id,
        created_at_utc=created,
        host_name=host,
        process_id=pid,
        local_output_root=root,
        source_asset_manifest_sha256=_hash("sources"),
        language_manifest_sha256=_hash("language"),
        dedup_manifest_sha256=_hash("dedup"),
        selection_manifest_sha256=_hash("selection"),
        algorithm_manifest_sha256=_hash("algorithms"),
        shards=(_shard() if shard is None else shard,),
    )


def _process(*, pid: int, root: str) -> ProcessAttestationV3:
    root = str(Path(root).resolve(strict=False))
    return ProcessAttestationV3(
        executable_sha256=_hash("python-executable"),
        dependency_lock_sha256=_hash("dependency-lock"),
        environment_identity_sha256=_hash("environment"),
        process_id=pid,
        output_root=root,
    )


def _replay(
    run_id: str,
    *,
    pid: int,
    root: str,
    shard: JsonlZstdShardIdentityV3 | None = None,
) -> ReplayRunReceiptV3:
    root = str(Path(root).resolve(strict=False))
    return ReplayRunReceiptV3(
        run_id=run_id,
        process_attestation=_process(pid=pid, root=root),
        input_identity_sha256=_hash("input"),
        dedup_binding_identity_sha256=A2_MINHASH_BINDING.receipt_sha256,
        dedup_decision_ledger_identity_sha256=_hash("dedup-ledger"),
        dedup_exact_match_rate=Fraction(7, 100),
        dedup_near_match_rate=Fraction(3, 100),
        dedup_dropped_bytes=101,
        dedup_topup_bytes=103,
        minhash_recall_audit=_recall_audit(),
        content_manifest=_manifest(
            run_id,
            created="2026-08-28T20:00:00Z",
            root=root,
            host=f"host-{run_id}",
            pid=pid,
            shard=shard,
        ),
    )


def _recall_audit() -> MinHashRecallAuditV3:
    return MinHashRecallAuditV3(
        seed=A2_DEDUP_SEED,
        synthetic_cells=tuple(
            MinHashSyntheticRecallCellV3(
                exact_jaccard=level,
                pair_count=100,
                candidate_count=95,
            )
            for level in MINHASH_RECALL_JACCARD_LEVELS
        ),
        real_sample_identity_sha256=_hash("real-recall-sample"),
        real_dolma_document_count=10,
        real_fineweb_document_count=10,
        real_exact_pairs_at_or_above_threshold=7,
        real_candidate_pairs_at_or_above_threshold=6,
    )


def test_a2_appends_v3_without_rekeying_banked_v1_or_v2() -> None:
    assert GTOK_EXECUTION_AUTHORITY_CHAIN_V3[:-1] == GTOK_EXECUTION_AUTHORITY_CHAIN_V2
    assert GTOK_EXECUTION_AUTHORITY_CHAIN_V3[-1] == GTOK_AMENDMENT_A2_SHA256
    assert a1_contract_snapshot_sha256() == (
        "1027ad932de92f21fc15695f5dc2d591295f03c7e2929b58f6e13d1dda5e3d03"
    )
    assert A2_CAMPAIGN_ROOT_SEED == 17_843_936_115_933_234_841
    assert A2_DEDUP_SEED == 10_865_107_354_467_150_331
    assert dict(A2_PIPELINE_SEEDS) == {
        "corpus.dedup": 10_865_107_354_467_150_331,
        "corpus.shuffle": 4_184_065_941_793_491_150,
        "corpus.split": 11_339_526_499_384_240_813,
        "corpus.topup": 12_725_005_642_314_495_995,
        "gtok.bpe": 3_884_157_959_809_795_597,
    }
    assert pipeline_seed("corpus.dedup") == A2_DEDUP_SEED
    with pytest.raises(ValueError, match="unknown A2"):
        pipeline_seed("corpus.unregistered")


def test_stable_identity_is_distinct_from_upsampled_occurrence_identity() -> None:
    first_document = _document("stable-row", "same retained text")
    second_document = _document("stable-row", "same retained text")
    assert first_document.document_id == second_document.document_id

    first = DocumentOccurrenceV3(first_document, _hash("asset-a"), 3)
    second = DocumentOccurrenceV3(second_document, _hash("asset-b"), 91)
    assert first.occurrence_id != second.occurrence_id
    assert first.document.document_id == second.document.document_id
    assert replace(second_document, text="changed").document_id != first_document.document_id
    same_bytes_other_record = _document("other-row", "same retained text")
    assert same_bytes_other_record.document_id != first_document.document_id
    assert is_exact_duplicate_v3(first_document, same_bytes_other_record) is True
    assert is_exact_duplicate_v3(first_document, replace(first_document, text="changed")) is False


def test_match_normalization_is_nfc_maximal_isspace_collapse_strip_and_utf8() -> None:
    raw = "\t e\u0301 \n\u2003x\u00a0\r\n"
    normalized = normalize_match_text(raw)
    assert [ord(character) for character in normalized] == [0xE9, 0x20, 0x78]
    assert normalized_match_bytes(raw) == b"\xc3\xa9 x"
    whitespace_only = _document("whitespace", "\t \r\n\u2003")
    assert dedup_match_input_v3(whitespace_only) is None
    assert EMPTY_MATCH_NORMALIZATION_DISPOSITION == (
        "DROP_EMPTY_AFTER_MATCH_NORMALIZATION"
    )
    assert normalize_match_text(" \t\n\u2003 ") == ""
    assert A2_MATCH_NORMALIZATION_BINDING.retained_text_is_unchanged is True


def test_byte_13_grams_bind_short_document_one_shingle_rule() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        byte_shingles_v3(b"")
    assert byte_shingles_v3(b"short") == frozenset((b"short",))
    exactly = b"1234567890123"
    assert byte_shingles_v3(exactly) == frozenset((exactly,))
    assert byte_shingles_v3(exactly + b"x") == frozenset((exactly, b"234567890123x"))
    with pytest.raises(ValueError, match="13-gram"):
        byte_shingles_v3(b"abc", 12)


def test_numpy_pcg64_linear_minhash_and_little_endian_bands_have_golden_vectors() -> None:
    shingles = byte_shingles_v3(b"abcdefghijklmnopqrstuvwxyz")
    signature = minhash_signature_v3(shingles)
    assert len(signature) == 128
    assert signature[:8] == (
        422_937_182_425_879_919,
        650_747_644_288_341_512,
        1_010_275_692_557_857_300,
        4_183_204_463_763_076_488,
        2_980_329_761_610_051_473,
        477_643_196_862_122_526,
        278_553_328_096_163_988,
        463_486_953_420_547_563,
    )
    packed = b"".join(value.to_bytes(8, "little") for value in signature)
    assert hashlib.sha256(packed).hexdigest() == (
        "26367cd2af1cc365983775f09756a82bc3cd1075291f43fcf16c488d76815455"
    )
    bands = lsh_band_keys_v3(signature)
    assert len(bands) == 16
    assert all(len(key) == 64 for key in bands)
    assert bands[0].hex() == (
        "6f95cc682093de0508963ae088eb070914721fc46338050e"
        "88356dd54bba0d3a91af4ab220425c291e4a59f6f4eda006"
        "94b82cacc69edd03ebe5a8b0eda26e06"
    )
    assert hashlib.sha256(b"".join(bands)).hexdigest() == (
        "26367cd2af1cc365983775f09756a82bc3cd1075291f43fcf16c488d76815455"
    )
    assert A2_MINHASH_BINDING.component_count == 128
    assert A2_MINHASH_BINDING.bit_generator == "numpy.random.PCG64"


def test_exact_jaccard_winner_uses_score_then_lowest_dolma_record_id() -> None:
    query = _document(
        "fineweb-query",
        "the exact same retained content",
        source="fineweb_edu",
    )
    high_id = _document("z-record", query.text)
    low_id = _document("a-record", query.text)
    winner = select_dedup_winner_v3(query, (high_id, low_id))
    assert winner is not None
    assert winner.exact_jaccard == Fraction(1, 1)
    assert winner.canonical_source_record_id == min(
        _hash("a-record"), _hash("z-record")
    )
    expected_document = min(
        (low_id, high_id), key=lambda item: item.stable_source_record_id
    )
    assert winner.canonical_document_id == expected_document.document_id
    unrelated = _document("unrelated", "completely unrelated bytes")
    assert select_dedup_winner_v3(query, (unrelated,)) is None
    with pytest.raises(ValueError, match="FineWeb-Edu"):
        select_dedup_winner_v3(low_id, (high_id,))


def test_minhash_recall_audit_is_explicitly_report_only() -> None:
    audit = _recall_audit()
    assert audit.status == "REPORT_ONLY_NO_RECALL_FLOOR"
    assert len(audit.receipt_sha256) == 64
    at_threshold = ideal_lsh_candidate_probability_v3(Fraction(4, 5))
    assert float(at_threshold) == pytest.approx(0.9470487964)


def test_greedy_first_fit_skips_oversize_and_continues_without_backtracking() -> None:
    candidates = (
        _candidate("too-large", "abcdef"),
        _candidate("fits-two", "ab"),
        _candidate("fits-three", "xyz"),
        _candidate("unseen-after-exact", "q"),
    )
    receipt = greedy_first_fit_v3(
        candidates,
        stream="T",
        stratum="general",
        target_bytes=5,
    )
    assert tuple(item.disposition for item in receipt.decisions) == (
        "oversized_remaining_capacity",
        "accepted",
        "accepted",
    )
    assert receipt.realized_bytes == 5
    assert receipt.shortfall_bytes == 0
    assert receipt.source_exhausted is False
    assert receipt.termination_reason == "exact"
    assert receipt.unscanned_suffix_start == 3
    assert receipt.accepted_document_ids == (
        candidates[1].document_id,
        candidates[2].document_id,
    )


def test_first_fit_stops_inside_tolerance_and_fails_on_exhaustion_above_it() -> None:
    receipt = greedy_first_fit_v3(
        (_candidate("within", "x" * 996), _candidate("unscanned", "tail")),
        stream="T",
        stratum="general",
        target_bytes=1_000,
    )
    assert receipt.realized_bytes == 996
    assert receipt.deficit_bytes == 4
    assert receipt.unscanned_suffix_start == 1
    assert receipt.source_exhausted is False
    assert receipt.termination_reason == "within_tolerance"
    with pytest.raises(RuntimeError, match="above the A2 0.5% tolerance"):
        greedy_first_fit_v3(
            (_candidate("underfilled", "x" * 900),),
            stream="T",
            stratum="general",
            target_bytes=1_000,
        )


def test_t_then_h_uses_oversize_skips_then_suffix_with_t_exclusions() -> None:
    too_big_for_t = _candidate("too-big", "abcd", cluster="oversize")
    training_doc = _candidate("training", "abc", cluster="shared")
    same_cluster = _candidate("near-copy", "def", cluster="shared")
    heldout_doc = _candidate("heldout", "xyz", cluster="independent")
    pair = greedy_training_then_heldout_v3(
        (too_big_for_t, training_doc, same_cluster, heldout_doc),
        stratum="general",
        training_target_bytes=3,
        heldout_target_bytes=3,
    )
    assert A2_STREAM_PRECEDENCE == ("T", "H")
    assert pair.training.accepted_document_ids == (training_doc.document_id,)
    assert pair.training.unscanned_suffix_start == 2
    assert tuple(item.document_id for item in pair.heldout.decisions) == (
        too_big_for_t.document_id,
        same_cluster.document_id,
        heldout_doc.document_id,
    )
    assert tuple(item.disposition for item in pair.heldout.decisions) == (
        "oversized_remaining_capacity",
        "excluded_cluster",
        "accepted",
    )
    assert pair.heldout.accepted_document_ids == (heldout_doc.document_id,)
    assert not set(pair.heldout.accepted_cluster_ids) & set(pair.training.accepted_cluster_ids)


def test_language_id_prefix_is_utf8_safe_general_only_and_equality_keeps() -> None:
    document = _document("language", "a" * 65_535 + "\u00e9")
    scoring = language_scoring_bytes_v3(document)
    assert scoring == b"a" * 65_535
    at_threshold = language_id_decision_v3(
        document, label="__label__en", probability=0.9
    )
    assert at_threshold.keep is True
    assert A2_LANGUAGE_ID_BINDING.scope == "general_only"
    assert language_id_decision_v3(
        document, label="__label__en", probability=0.899
    ).keep is False
    with pytest.raises(ValueError, match="only for general"):
        language_scoring_bytes_v3(
            _document("code", "print(1)", source="stackedu", stratum="code")
        )


def test_language_id_selects_highest_score_with_ascii_label_tiebreak() -> None:
    document = _document("language-tie", "English text")
    decision = language_id_decision_from_predictions_v3(
        document,
        (
            ("__label__fr", 0.93),
            ("__label__en", 0.93),
            ("__label__de", 0.91),
        ),
    )
    assert decision.label == "__label__en"
    assert decision.probability == 0.93
    assert decision.keep is True
    with pytest.raises(ValueError, match="must be ASCII"):
        language_id_decision_from_predictions_v3(
            document,
            (("__label__é", 1.0),),
        )


def test_canonical_jsonl_has_exact_key_order_utf8_and_lf_golden() -> None:
    document = _document("row-1", "caf\u00e9\nline")
    assert document.document_id == (
        "89cbcbef22b3063f2efb573d1d4799e36e7e3e760ca35f6f98520c1855ba9b56"
    )
    encoded = canonical_jsonl_record_bytes_v3(document)
    assert document.shard_record_id == "3adef87bd8ed30b991eb19fdea579cfc76f7e30c"
    assert encoded == (
        b'{"id":"3adef87bd8ed30b991eb19fdea579cfc76f7e30c",'
        b'"source":"dolma_web","stratum":"general","text":"caf\xc3\xa9\\nline"}\n'
    )
    assert hashlib.sha256(encoded).hexdigest() == (
        "c83cbda5499d7c6477850db905c0c4a060ed8f8e49bc99d33f5e535059133cd9"
    )


def test_zstd_binding_is_literal_but_performs_no_compression() -> None:
    assert A2_ZSTD_CODEC_BINDING.python_package_version == "0.25.0"
    assert A2_ZSTD_CODEC_BINDING.libzstd_version == "1.5.7"
    assert A2_ZSTD_CODEC_BINDING.compression_level == 3
    assert A2_ZSTD_CODEC_BINDING.threads == 0
    assert A2_ZSTD_CODEC_BINDING.write_checksum is True
    assert A2_ZSTD_CODEC_BINDING.write_content_size is False
    assert A2_ZSTD_CODEC_BINDING.write_dict_id is False


def test_tripwire_projects_exact_steps_and_fails_closed_at_boundaries() -> None:
    projection = TripwireProjectionV3(
        warmup_steps=A2_CALIBRATION_WARMUP_STEPS,
        measured_steps=A2_CALIBRATION_MEASURED_STEPS,
        warmup_a100_microseconds=250_000,
        measured_nonpad_tokens=1_000,
        measured_a100_microseconds=1_000_000,
        remaining_step_nonpad_token_counts=(1_000,) * 10,
    )
    assert projection.measured_synchronized_tokens_per_second == Fraction(1_000, 1)
    assert projection.remaining_exact_step_count == 10
    assert projection.projected_remaining_a100_seconds == Fraction(10, 1)
    assert projection.projected_total_a100_seconds == Fraction(45, 4)
    assert projection.calibration_steps == A2_CALIBRATION_STEPS_MAXIMUM
    with pytest.raises(ValueError, match="exactly 20 warmup"):
        replace(projection, warmup_steps=19)
    with pytest.raises(ValueError, match="exactly 80 measured"):
        replace(projection, measured_steps=79)
    assert A2_TRIPWIRE_A100_SECONDS == 43_200
    assert A2_CHARGED_ATTEMPT_STATUSES == (
        "calibration",
        "completed",
        "failed",
        "aborted",
        "preempted",
        "retried",
    )
    assert tripwire_preflight_v3(
        consumed_a100_seconds=Fraction(43_190, 1),
        active_reservations_a100_seconds=Fraction(0, 1),
        projection=projection,
        calibration_charged_to_consumed=True,
    ).action == "ALLOW"
    assert tripwire_preflight_v3(
        consumed_a100_seconds=Fraction(43_191, 1),
        active_reservations_a100_seconds=Fraction(0, 1),
        projection=projection,
        calibration_charged_to_consumed=True,
    ).action == "REJECT_PREFLIGHT"
    with pytest.raises(ValueError, match="already be charged"):
        tripwire_preflight_v3(
            consumed_a100_seconds=Fraction(0, 1),
            active_reservations_a100_seconds=Fraction(0, 1),
            projection=projection,
            calibration_charged_to_consumed=False,
        )
    assert tripwire_runtime_v3(
        cumulative_charged_a100_seconds=Fraction(43_200, 1),
        run_charged_a100_seconds=Fraction(1, 1),
        calibration_projection_a100_seconds=Fraction(10, 1),
    ).action == "HARD_ABORT_ALL"
    assert tripwire_runtime_v3(
        cumulative_charged_a100_seconds=Fraction(100, 1),
        run_charged_a100_seconds=Fraction(21, 1),
        calibration_projection_a100_seconds=Fraction(10, 1),
    ).action == "HARD_ABORT_RUN"
    assert tripwire_runtime_v3(
        cumulative_charged_a100_seconds=Fraction(100, 1),
        run_charged_a100_seconds=Fraction(20, 1),
        calibration_projection_a100_seconds=Fraction(10, 1),
    ).action == "ALLOW"


def test_manifest_content_identity_excludes_run_time_host_pid_and_local_path() -> None:
    first = _manifest(
        "build-a",
        created="2026-08-28T20:00:00Z",
        root="/tmp/build-a",
        host="host-a",
        pid=101,
    )
    second = _manifest(
        "build-b",
        created="2026-08-29T01:02:03Z",
        root="C:/different/build-b",
        host="host-b",
        pid=202,
    )
    assert first.content_payload == second.content_payload
    assert first.content_identity_sha256 == second.content_identity_sha256
    assert first.audit_identity_sha256 != second.audit_identity_sha256

    drifted = _manifest(
        "build-b",
        created="2026-08-29T01:02:03Z",
        root="C:/different/build-b",
        host="host-b",
        pid=202,
        shard=_shard(logical="changed"),
    )
    assert drifted.content_identity_sha256 != first.content_identity_sha256


def test_independent_replay_claims_remain_nonauthoritative_without_parent_rehash() -> None:
    first = _replay("build-a", pid=101, root="/tmp/a")
    second = _replay("build-b", pid=202, root="/tmp/b")
    d1, d2 = validate_independent_replays_v3(first, second)
    assert (d1.gate, d2.gate) == ("D1", "D2")
    assert d1.status == d2.status == "CHECK_PASS"
    assert d1.authoritative is d2.authoritative is False
    assert len(d1.receipt_sha256) == len(d2.receipt_sha256) == 64
    with pytest.raises(TypeError, match="factory-minted"):
        CorpusGateReceiptV3()
    with pytest.raises(ValueError, match="wrong registered dedup binding"):
        replace(first, dedup_binding_identity_sha256=_hash("wrong-dedup-binding"))
    with pytest.raises(ValueError, match="registered A2 seed"):
        replace(_recall_audit(), seed=A2_DEDUP_SEED + 1)

    with pytest.raises(ValueError, match="process IDs"):
        validate_independent_replays_v3(
            first,
            _replay("build-b", pid=101, root="/tmp/b"),
        )
    with pytest.raises(ValueError, match="non-overlapping roots"):
        validate_independent_replays_v3(
            first,
            _replay("build-b", pid=202, root="/tmp/a"),
        )
    with pytest.raises(ValueError, match="executable/dependency/environment"):
        validate_independent_replays_v3(
            first,
            replace(
                second,
                process_attestation=replace(
                    second.process_attestation,
                    executable_sha256=_hash("different-python"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="D2 failed"):
        validate_independent_replays_v3(
            first,
            replace(second, dedup_near_match_rate=Fraction(4, 100)),
        )
    with pytest.raises(ValueError, match="D1 failed"):
        validate_independent_replays_v3(
            first,
            replace(
                second,
                content_manifest=_manifest(
                    "build-b",
                    created="2026-08-29T01:00:00Z",
                    root=str(Path("/tmp/b").resolve(strict=False)),
                    host="host-build-b",
                    pid=202,
                    shard=_shard(compressed="changed-zstd"),
                ),
            ),
        )
