from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib

import pytest

from training.weft1_corpus_contract import (
    AlgorithmSpec,
    BuildRunDiagnostic,
    BYTE_SHINGLE_WIDTH,
    CORPUS_STRATUM_TARGETS,
    CORPUS_TOTAL_TARGET_BYTES,
    DedupDecisionDiagnostic,
    DocumentRecord,
    DraftDiagnosticReceipt,
    GENERAL_SOURCE_TARGETS,
    GTOK_SCREEN_TARGET_BYTES,
    NEAR_DUPLICATE_JACCARD_THRESHOLD,
    NormalizedDocumentDiagnostic,
    ReferenceDedupBinding,
    RoundTripCaseDiagnostic,
    RoundTripFixture,
    RoundTripFixtureManifest,
    RunStreamDiagnostic,
    ShardDiagnostic,
    SourceAsset,
    SplitEntryDiagnostic,
    SplitManifestDiagnostic,
    StreamDiagnostic,
    build_reference_cross_source_dedup_diagnostic,
    byte_shingles,
    describe_d3_composition,
    diagnose_d1_reproduction,
    diagnose_d2_dedup_reproduction,
    diagnose_d4_language_filter_scope,
    diagnose_d5_round_trip_suite,
    diagnose_d6_streams,
    exact_set_jaccard,
    mint_authoritative_gate_receipt,
    reference_frame_payload,
    reference_minhash_signature,
)
from training.weft1_gtok_contract import (
    GTOK_ROUND_TRIP_CATEGORIES,
    GTOK_VOCABULARY_ARMS,
    GTokExecutionBlocked,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _algorithm(name: str) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        schema_version="fixture-v1",
        exact_spec_sha256=_hash(f"{name}-spec"),
        implementation_tree_sha256=_hash(f"{name}-implementation"),
        dependency_lock_sha256=_hash(f"{name}-dependencies"),
    )


def _source_asset(source: str) -> SourceAsset:
    repository = "allenai/dolma3" if source == "dolma_web" else "HuggingFaceFW/fineweb-edu"
    return SourceAsset(
        source_family=source,
        requested_repository=repository,
        resolved_repository=f"fixture/{source}",
        revision="1" * 40,
        config="default",
        split="train",
        locator_kind="repo_path",
        locator=f"data/{source}/part-00000.jsonl.zst",
        byte_size=123,
        sha256=_hash(f"{source}-asset"),
    )


def _document(source: str, record_id: str, retained: bytes) -> DocumentRecord:
    return DocumentRecord(
        source_asset=_source_asset(source),
        stratum="general",
        stable_source_record_id=record_id,
        retained_bytes=retained,
    )


def _normalized(
    source: str,
    record_id: str,
    retained: bytes,
    normalized: bytes | None = None,
    *,
    normalizer: AlgorithmSpec | None = None,
) -> NormalizedDocumentDiagnostic:
    return NormalizedDocumentDiagnostic(
        document=_document(source, record_id, retained),
        normalization_spec=_algorithm("normalizer") if normalizer is None else normalizer,
        normalized_bytes=retained if normalized is None else normalized,
    )


def _binding() -> ReferenceDedupBinding:
    return ReferenceDedupBinding(
        normalization=_algorithm("normalizer"),
        minhash=_algorithm("reference-minhash"),
        shingle_width=13,
        minhash_components=64,
        minhash_seed=20260828,
        lsh_bands=64,
        lsh_rows_per_band=1,
        jaccard_threshold=Fraction(4, 5),
    )


def _shard(path: str = "general/00000.bin") -> ShardDiagnostic:
    return ShardDiagnostic(
        relative_path=path,
        serializer_spec_draft_sha256=_hash("serializer"),
        logical_stream_sha256=_hash("logical-stream"),
        retained_byte_count=100,
        record_count=2,
        file_sha256=_hash("file"),
        file_size=120,
    )


def _build(run_id: str, *, shard: ShardDiagnostic | None = None) -> BuildRunDiagnostic:
    return BuildRunDiagnostic(
        run_id=run_id,
        source_asset_manifest_sha256=_hash("sources"),
        algorithm_binding_manifest_sha256=_hash("algorithms"),
        selection_spec_sha256=_hash("selection"),
        shards=(_shard() if shard is None else shard,),
    )


def test_declared_target_centers_are_exact_and_explicitly_not_tolerances() -> None:
    assert CORPUS_TOTAL_TARGET_BYTES == 38_000_000_000
    assert sum(value for _, value in CORPUS_STRATUM_TARGETS) == CORPUS_TOTAL_TARGET_BYTES
    assert sum(value for _, value in GENERAL_SOURCE_TARGETS) == 17_100_000_000
    assert GTOK_SCREEN_TARGET_BYTES == 4_000_000_000
    assert BYTE_SHINGLE_WIDTH == 13
    assert NEAR_DUPLICATE_JACCARD_THRESHOLD == Fraction(4, 5)


def test_source_asset_is_content_addressed_but_does_not_infer_route_authority() -> None:
    asset = _source_asset("dolma_web")
    assert len(asset.draft_sha256) == 64
    with pytest.raises(ValueError, match="revision"):
        replace(asset, revision="main")
    with pytest.raises(ValueError, match="noncanonical"):
        replace(asset, locator="data/../escape.bin")
    with pytest.raises(ValueError, match="canonical HTTPS"):
        replace(asset, locator_kind="https_uri", locator="https://user@example.com/x?q=1")
    https_asset = replace(
        asset,
        locator_kind="https_uri",
        locator="https://huggingface.co/datasets/fixture/data.bin",
    )
    assert https_asset.locator_kind == "https_uri"
    for unsafe_uri in (
        "https://:443/x",
        "https://Example.com/x",
        "https://example.com./x",
        "https://127.0.0.1/x",
        "https://example.com//x",
        "https://example.com/%2e%2e/x",
        "https://example.com\n/evil",
    ):
        with pytest.raises(ValueError):
            replace(asset, locator_kind="https_uri", locator=unsafe_uri)


def test_document_identity_binds_source_asset_and_retained_bytes() -> None:
    document = _document("dolma_web", "row-1", b"alpha\t\tbeta\n")
    changed_asset = replace(
        document,
        source_asset=replace(document.source_asset, revision="2" * 40),
    )
    assert document.doc_id != changed_asset.doc_id
    with pytest.raises(ValueError, match="at least one byte"):
        replace(document, retained_bytes=b"")


def test_normalization_output_is_labeled_diagnostic_and_binds_original_document() -> None:
    first = _normalized("dolma_web", "row-1", b"alpha\t\tbeta", b"alpha beta")
    second = replace(first, normalized_bytes=b"arbitrary")
    assert first.draft_sha256 != second.draft_sha256
    assert first.document.retained_bytes == b"alpha\t\tbeta"


def test_reference_shingles_jaccard_and_minhash_have_golden_vectors() -> None:
    short = byte_shingles(b"abc", 13)
    assert short == frozenset((b"abc",))
    left = byte_shingles(b"abcdefghijklmnopqrstuvwxyz", 13)
    right = byte_shingles(b"abcdefghijklmnopqrstuvwxyZ", 13)
    assert exact_set_jaccard(left, right) == Fraction(13, 15)
    signature = reference_minhash_signature(left, components=4, seed=7)
    assert signature == (
        5620341053833927800,
        2108293533298255676,
        241571056798219212,
        1007336631381635417,
    )
    assert signature != reference_minhash_signature(left, components=4, seed=8)
    with pytest.raises(ValueError, match="safe bound"):
        reference_minhash_signature(left, components=4097, seed=7)


def test_reference_dedup_confirms_exact_bytes_and_exact_jaccard() -> None:
    normalizer = _algorithm("normalizer")
    exact_dolma = _normalized(
        "dolma_web", "dolma-exact", b"alpha\t\tbeta", b"alpha beta", normalizer=normalizer
    )
    near_base = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    near_dolma = _normalized(
        "dolma_web", "dolma-near", near_base, normalizer=normalizer
    )
    fine_exact = _normalized(
        "fineweb_edu", "fine-exact", b"alpha  beta\n", b"alpha beta", normalizer=normalizer
    )
    fine_near = _normalized(
        "fineweb_edu", "fine-near", near_base[:-1] + b"!", normalizer=normalizer
    )
    ledger = build_reference_cross_source_dedup_diagnostic(
        (exact_dolma, near_dolma),
        (fine_exact, fine_near),
        _binding(),
        run_id="reference-a",
    )
    assert {item.route for item in ledger.decisions} == {"exact", "near"}
    assert all(
        Fraction(item.exact_jaccard_numerator, item.exact_jaccard_denominator)
        >= Fraction(4, 5)
        for item in ledger.decisions
    )
    with pytest.raises(ValueError, match=">= 4/5"):
        DedupDecisionDiagnostic(
            fineweb_doc_id=_hash("fine"),
            canonical_dolma_doc_id=_hash("dolma"),
            route="near",
            exact_jaccard_numerator=1,
            exact_jaccard_denominator=10,
        )
    with pytest.raises(ValueError, match="reduced form"):
        DedupDecisionDiagnostic(
            fineweb_doc_id=_hash("fine"),
            canonical_dolma_doc_id=_hash("dolma"),
            route="exact",
            exact_jaccard_numerator=2,
            exact_jaccard_denominator=2,
        )
    with pytest.raises(ValueError, match="reduced form"):
        DedupDecisionDiagnostic(
            fineweb_doc_id=_hash("fine"),
            canonical_dolma_doc_id=_hash("dolma"),
            route="near",
            exact_jaccard_numerator=8,
            exact_jaccard_denominator=10,
        )


def test_reference_dedup_rejects_a_mislabeled_normalizer() -> None:
    dolma = _normalized("dolma_web", "d", b"same")
    fineweb = _normalized(
        "fineweb_edu", "f", b"same", normalizer=_algorithm("different-normalizer")
    )
    with pytest.raises(ValueError, match="bound reference normalizer"):
        build_reference_cross_source_dedup_diagnostic(
            (dolma,), (fineweb,), _binding(), run_id="reference-a"
        )


def test_d1_requires_independent_run_ids_and_typed_shards() -> None:
    first = _build("build-a")
    second = _build("build-b")
    receipt = diagnose_d1_reproduction(first, second)
    assert receipt.status == "DRAFT_CONSISTENT"
    assert receipt.authoritative is False
    with pytest.raises(ValueError, match="distinct"):
        diagnose_d1_reproduction(first, first)
    with pytest.raises(ValueError, match="different shards"):
        diagnose_d1_reproduction(
            first,
            _build("build-b", shard=replace(_shard(), file_sha256=_hash("changed"))),
        )


@pytest.mark.parametrize(
    "path",
    (
        "../escape.bin",
        "A.bin",
        "nul.txt/x.bin",
        "ok/has space.bin",
        "ok/x.bin:stream",
        "ok/trailing.",
        "ok/\x00.bin",
    ),
)
def test_shard_paths_use_one_canonical_cross_platform_grammar(path: str) -> None:
    with pytest.raises(ValueError):
        _shard(path)


def test_d2_requires_independent_dedup_runs() -> None:
    normalizer = _algorithm("normalizer")
    dolma = (_normalized("dolma_web", "d", b"same", normalizer=normalizer),)
    fineweb = (_normalized("fineweb_edu", "f", b"same", normalizer=normalizer),)
    first = build_reference_cross_source_dedup_diagnostic(
        dolma, fineweb, _binding(), run_id="dedup-a"
    )
    second = build_reference_cross_source_dedup_diagnostic(
        dolma, fineweb, _binding(), run_id="dedup-b"
    )
    assert diagnose_d2_dedup_reproduction(first, second).authoritative is False
    with pytest.raises(ValueError, match="distinct"):
        diagnose_d2_dedup_reproduction(first, first)


def test_d3_reports_observed_targets_and_leaves_tolerance_unbound() -> None:
    diagnostic = describe_d3_composition(
        dict(CORPUS_STRATUM_TARGETS),
        dict(GENERAL_SOURCE_TARGETS),
        composition_input_sha256=_hash("composition"),
    )
    assert diagnostic.status == "DESCRIPTIVE_ONLY"
    assert dict(diagnostic.evidence)["tolerance_semantics"] == "UNBOUND"
    changed_strata = dict(CORPUS_STRATUM_TARGETS)
    changed_general = dict(GENERAL_SOURCE_TARGETS)
    changed_strata["general"] += 100
    changed_general["fineweb_edu"] += 100
    changed = describe_d3_composition(
        changed_strata,
        changed_general,
        composition_input_sha256=_hash("composition"),
    )
    assert changed.status == "DESCRIPTIVE_ONLY"


def test_d4_requires_zero_classifier_calls_outside_general() -> None:
    invocations = {"general": 10, "code": 0, "mathematics": 0, "science_technical": 0}
    rejections = {"general": 2, "code": 0, "mathematics": 0, "science_technical": 0}
    assert diagnose_d4_language_filter_scope(
        invocations,
        rejections,
        language_filter_input_sha256=_hash("language-filter"),
    ).authoritative is False
    with pytest.raises(ValueError, match="called outside"):
        diagnose_d4_language_filter_scope(
            {**invocations, "code": 1},
            rejections,
            language_filter_input_sha256=_hash("language-filter"),
        )
    with pytest.raises(ValueError, match="exceeds"):
        diagnose_d4_language_filter_scope(
            invocations,
            {**rejections, "general": 11},
            language_filter_input_sha256=_hash("language-filter"),
        )


def test_d5_requires_every_registered_case_and_exercises_a_callback() -> None:
    calls: list[bytes] = []

    def round_trip(value: bytes) -> bytes:
        calls.append(value)
        return bytes(value)

    payloads = {
        "accented_latin": "café e\u0301".encode("utf-8"),
        "cjk": "漢字かなカナ".encode("utf-8"),
        "greek": "αλφάβητο".encode("utf-8"),
        "mixed_indentation": b"  alpha\n\tbeta\n    gamma",
        "right_to_left": "עברית العربية".encode("utf-8"),
        "tabs": b"\talpha\tbeta\t",
        "typographic_punctuation": "“quote”—‘dash’…".encode("utf-8"),
    }
    fixtures = tuple(
        sorted(
            (
                RoundTripFixture(
                    fixture_id=f"fixture-{category.replace('_', '-')}",
                    category=category,
                    payload=payloads[category],
                )
                for category in GTOK_ROUND_TRIP_CATEGORIES
            ),
            key=lambda item: item.fixture_id,
        )
    )
    manifest = RoundTripFixtureManifest(fixtures)
    cases = tuple(RoundTripCaseDiagnostic.exercise(fixture, round_trip) for fixture in fixtures)
    diagnostic = diagnose_d5_round_trip_suite(
        cases,
        codec_spec=_algorithm("codec"),
        fixture_manifest=manifest,
    )
    assert diagnostic.authoritative is False
    assert len(calls) == len(GTOK_ROUND_TRIP_CATEGORIES)
    with pytest.raises(ValueError, match="typed fixture manifest"):
        diagnose_d5_round_trip_suite(
            cases[:-1],
            codec_spec=_algorithm("codec"),
            fixture_manifest=manifest,
        )
    with pytest.raises(ValueError, match="at least one byte"):
        RoundTripFixture("fixture-empty", "tabs", b"")


def test_reference_length_framing_is_unambiguous_but_not_a_gate_codec() -> None:
    assert reference_frame_payload(b"ab") + reference_frame_payload(b"c") != (
        reference_frame_payload(b"a") + reference_frame_payload(b"bc")
    )


def _split_and_runs() -> tuple[SplitManifestDiagnostic, tuple[RunStreamDiagnostic, ...]]:
    codec = _algorithm("stream-codec")
    first = _document("dolma_web", "one", b"one")
    second = _document("fineweb_edu", "two", b"two")
    heldout_doc = _document("dolma_web", "heldout", b"heldout")
    split = SplitManifestDiagnostic(
        split_spec_draft_sha256=_hash("split-spec"),
        cluster_spec_draft_sha256=_hash("cluster-spec"),
        training=tuple(
            sorted(
                (
                    SplitEntryDiagnostic(first.doc_id, _hash("train-cluster-1"), "general", 3),
                    SplitEntryDiagnostic(second.doc_id, _hash("train-cluster-2"), "general", 3),
                ),
                key=lambda item: item.doc_id,
            )
        ),
        heldout=(
            SplitEntryDiagnostic(heldout_doc.doc_id, _hash("heldout-cluster"), "general", 7),
        ),
    )
    heldout = StreamDiagnostic.from_documents(
        (heldout_doc,), permutation_seed=None, codec_spec=codec
    )
    orders = {101: (first, second), 202: (second, first)}
    runs = tuple(
        RunStreamDiagnostic(
            vocabulary_size=vocabulary,
            seed=seed,
            training=StreamDiagnostic.from_documents(
                orders[seed], permutation_seed=seed, codec_spec=codec
            ),
            heldout=heldout,
        )
        for vocabulary in GTOK_VOCABULARY_ARMS
        for seed in (101, 202)
    )
    return split, runs


def test_split_manifest_binds_cluster_aware_disjointness() -> None:
    split, unused_runs = _split_and_runs()
    with pytest.raises(ValueError, match="dedup cluster"):
        replace(
            split,
            heldout=(replace(split.heldout[0], cluster_id=split.training[0].cluster_id),),
        )


def test_d6_uses_typed_train_heldout_and_split_receipts_without_greening() -> None:
    split, runs = _split_and_runs()
    diagnostic = diagnose_d6_streams(runs, split)
    evidence = dict(diagnostic.evidence)
    assert diagnostic.status == "DESCRIPTIVE_ONLY"
    assert evidence["seed_identities_status"] == "UNBOUND"
    assert evidence["registered_screen_target_bytes"] == 4_000_000_000

    with pytest.raises(ValueError, match="seed disagrees"):
        replace(runs[-2], training=runs[-1].training)

    drifted = list(runs)
    wrong_order = replace(
        drifted[-2].training,
        ordered_document_ids_sha256=_hash("wrong-order"),
        framed_payload_stream_sha256=_hash("wrong-payload-order"),
    )
    drifted[-2] = replace(drifted[-2], training=wrong_order)
    with pytest.raises(ValueError, match="different order within a seed"):
        diagnose_d6_streams(tuple(drifted), split)

    wrong_heldout = replace(
        runs[-1].heldout,
        framed_payload_stream_sha256=_hash("different-heldout"),
    )
    with pytest.raises(ValueError, match="held-out stream differs"):
        diagnose_d6_streams((*runs[:-1], replace(runs[-1], heldout=wrong_heldout)), split)

    other_codec = _algorithm("other-stream-codec")
    codec_drift = list(runs)
    codec_drift[-1] = replace(
        codec_drift[-1],
        training=replace(
            codec_drift[-1].training,
            codec_spec_draft_sha256=other_codec.draft_sha256,
        ),
    )
    with pytest.raises(ValueError, match="different stream codecs"):
        diagnose_d6_streams(tuple(codec_drift), split)

    cross_seed_codec_drift = tuple(
        replace(
            run,
            training=replace(
                run.training,
                codec_spec_draft_sha256=other_codec.draft_sha256,
            ),
        )
        if run.seed == 202
        else run
        for run in runs
    )
    with pytest.raises(ValueError, match="one stream codec"):
        diagnose_d6_streams(cross_seed_codec_drift, split)

    with pytest.raises(ValueError, match="unregistered vocabulary"):
        replace(runs[0], vocabulary_size=float(runs[0].vocabulary_size))


def test_draft_diagnostics_cannot_mint_authoritative_gate_receipts() -> None:
    diagnostic = DraftDiagnosticReceipt(
        gate="D1",
        status="DRAFT_CONSISTENT",
        named_input_sha256s=(("input", _hash("input")),),
        evidence=(("fact", "draft-only"),),
    )
    with pytest.raises(GTokExecutionBlocked, match="unresolved"):
        mint_authoritative_gate_receipt("D1", diagnostic)


def test_draft_receipt_rejects_order_dependent_unnamed_input_sets() -> None:
    with pytest.raises(ValueError, match="unique and sorted"):
        DraftDiagnosticReceipt(
            gate="D1",
            status="DRAFT_CONSISTENT",
            named_input_sha256s=(("z", _hash("z")), ("a", _hash("a"))),
            evidence=(("fact", "draft-only"),),
        )
