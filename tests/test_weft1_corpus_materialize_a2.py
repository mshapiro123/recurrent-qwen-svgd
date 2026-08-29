from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import socket

import pytest
import zstandard

import training.weft1_corpus_pa as production_io
import training.weft1_corpus_replay_a2 as replay
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
