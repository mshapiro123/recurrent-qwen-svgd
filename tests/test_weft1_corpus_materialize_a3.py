from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
import zstandard

import training.weft1_corpus_materialize_a3 as bridge
import training.weft1_corpus_pa as production_io
import training.weft1_corpus_replay_a2 as replay_v3
import training.weft1_corpus_replay_a3 as replay_v4
from training.weft1_corpus_a2 import (
    A2_CAMPAIGN_ROOT_SEED,
    A2_ZSTD_CODEC_BINDING,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
    StableDocumentV3,
    execution_authority_v3_bound_sha256,
)
from training.weft1_seed import derive_module_seed
from training.weft1_corpus_a3 import (
    A3EffectiveRouteResolution,
    EffectiveSourceRouteA3,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
)
from training.weft1_corpus_fetch_a3 import (
    AUTHORITATIVE_MODE,
    DOWNLOAD_RECEIPT_ARTIFACT_SCHEMA_V4,
    DOWNLOAD_RECEIPT_SCHEMA_V4,
    SOURCE_CACHE_MANIFEST_SCHEMA_V4,
    UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V4,
    UPSTREAM_ENUMERATION_SCHEMA_V4,
    DownloadedAssetEvidenceV4,
    FamilyEnumerationV4,
    PAExecutionBindingV4,
    PASourceExecutionContextV4,
    SourceAssetDownloadPlanV4,
    SourceCacheAssetV4,
    SourceCacheDownloadReceiptV4,
    SourceCacheManifestV4,
    UpstreamAssetV4,
    UpstreamEnumerationReceiptV4,
    VerifiedLocalCacheV4,
)
from training.weft1_corpus_materialize_a2 import (
    MATERIALIZER_SCHEMA,
    PRODUCTION_MODE,
    MaterializationInputV3,
    MaterializationResultV3,
)
from training.weft1_corpus_sources_a2 import asset_order_digest_v3
from training.weft1_gtok_a1_contract import SOURCE_FAMILIES, load_source_route_manifest
from training.weft1_gtok_contract import GTOK_STRATA, canonical_json_bytes


def _fineweb_ids() -> tuple[str, ...]:
    return tuple(f"CC-MAIN-{2000 + index // 100:04d}-{index % 100:02d}" for index in range(110))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _fixture(tmp_path: Path) -> tuple[
    PASourceExecutionContextV4,
    UpstreamEnumerationReceiptV4,
    SourceCacheManifestV4,
    SourceCacheDownloadReceiptV4,
    Path,
    Path,
    Path,
    Path,
]:
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True)
    routes: list[EffectiveSourceRouteA3] = []
    raw_by_family: dict[str, bytes] = {}
    bases = {row.source_family: row for row in load_source_route_manifest().routes}
    for source in SOURCE_FAMILIES:
        base = bases[source]
        raw = f"{source}:v4\n".encode("utf-8")
        raw_by_family[source] = raw
        overlay = source in {"dolma_web", "fineweb_edu"}
        routes.append(
            EffectiveSourceRouteA3(
                source_family=source,
                stratum=base.stratum,
                role=base.role,
                repository=base.repository,
                config=base.config,
                revision=base.revision,
                split=base.split,
                asset_selector=base.asset_selector,
                selection_rule=base.selection_rule,
                declared_license=base.declared_license,
                card_url=base.card_url,
                card_sha256=base.card_sha256,
                asset_count=1,
                available_bytes=len(raw),
                available_bytes_basis=base.available_bytes_basis,
                required_bytes=1,
                lineage_evidence=base.lineage_evidence,
                parse_policy=base.parse_policy,
                external_locator_manifest_sha256=base.external_locator_manifest_sha256,
                base_route_receipt_sha256=base.receipt_sha256,
                route_resolution=("A3_TEST_OVERLAY" if overlay else "PASSTHROUGH_A1_UNCHANGED"),
                breakdown_artifact_receipt_sha256=("a" * 64 if overlay else None),
                breakdown_family_projection_sha256=(
                    ("b" if source == "dolma_web" else "c") * 64 if overlay else None
                ),
            )
        )
    resolution = A3EffectiveRouteResolution(
        routes=tuple(routes),
        effective_route_identity_sha256="d" * 64,
        breakdown_artifact_physical_sha256="e" * 64,
        breakdown_artifact_receipt_sha256="a" * 64,
        family_projection_sha256s=(("dolma_web", "b" * 64), ("fineweb_edu", "c" * 64)),
    )
    binding = PAExecutionBindingV4.from_resolution(resolution)
    context = PASourceExecutionContextV4(
        resolution=resolution,
        binding=binding,
        fineweb_cc_dump_ids=_fineweb_ids(),
        dolma_top_bucket_group_ids=("common_crawl-test-0019",),
    )

    families: list[FamilyEnumerationV4] = []
    cache_assets: list[SourceCacheAssetV4] = []
    upstream_by_key: dict[tuple[str, str], UpstreamAssetV4] = {}
    for route in routes:
        source = route.source_family
        raw = raw_by_family[source]
        digest = hashlib.sha256(raw).hexdigest()
        locator = f"data/{source}/asset.bin"
        upstream = UpstreamAssetV4(
            source_family=source,
            repository=route.repository,
            config=route.config,
            revision=route.revision,
            split=route.split,
            asset_locator=locator,
            upstream_bytes=len(raw),
            blob_identity_kind="content_sha256",
            blob_identity=digest,
            content_sha256=digest,
            effective_route_receipt_sha256=route.receipt_sha256,
            execution_binding_sha256=binding.receipt_sha256,
        )
        upstream_by_key[(source, locator)] = upstream
        external = (
            {
                "external_locator_manifest_sha256": route.external_locator_manifest_sha256,
                "external_locator_manifest_bytes": 1,
                "external_locator_listing_receipt_sha256": "f" * 64,
            }
            if source == "wikipedia_wikibooks"
            else {}
        )
        families.append(
            FamilyEnumerationV4(
                route=route,
                execution_binding_sha256=binding.receipt_sha256,
                observed_available_bytes=len(raw),
                assets=(upstream,),
                **external,
            )
        )
        relative = f"{source}/asset.bin"
        path = cache_root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        cache_assets.append(
            SourceCacheAssetV4(
                source_family=source,
                repository=route.repository,
                config=route.config,
                revision=route.revision,
                split=route.split,
                asset_locator=locator,
                relative_path=relative,
                bytes=len(raw),
                sha256=digest,
                effective_route_receipt_sha256=route.receipt_sha256,
                execution_binding_sha256=binding.receipt_sha256,
            )
        )
    enumeration = UpstreamEnumerationReceiptV4(
        schema=UPSTREAM_ENUMERATION_SCHEMA_V4,
        mode=AUTHORITATIVE_MODE,
        execution_binding=binding,
        enumerator_binding_sha256="1" * 64,
        replay_attestation_receipt_sha256="2" * 64,
        families=tuple(families),
    )
    selected = tuple(asset for family in enumeration.families for asset in family.assets)
    plan = SourceAssetDownloadPlanV4(
        execution_binding=binding,
        enumeration_receipt_sha256=enumeration.receipt_sha256,
        enumeration_mode=enumeration.mode,
        assets=selected,
    )
    ordered_cache = tuple(
        sorted(
            cache_assets,
            key=lambda asset: (
                asset_order_digest_v3(asset.asset_locator),
                asset.source_family.encode("utf-8"),
                asset.asset_locator.encode("utf-8"),
                asset.sha256,
            ),
        )
    )
    manifest = SourceCacheManifestV4(
        schema=SOURCE_CACHE_MANIFEST_SCHEMA_V4,
        execution_binding=binding,
        effective_route_identity_sha256=binding.effective_route_identity_sha256,
        selection_plan_sha256=plan.receipt_sha256,
        assets=ordered_cache,
    )
    evidence = tuple(
        DownloadedAssetEvidenceV4(
            execution_binding_sha256=binding.receipt_sha256,
            upstream_asset_identity_sha256=upstream_by_key[(asset.source_family, asset.asset_locator)].asset_identity_sha256,
            source_cache_asset_identity_sha256=asset.asset_identity_sha256,
            relative_path=asset.relative_path,
            observed_bytes=asset.bytes,
            observed_sha256=asset.sha256,
            upstream_identity_check="content_sha256",
        )
        for asset in ordered_cache
    )
    verified = VerifiedLocalCacheV4(
        execution_binding=binding,
        source_manifest=manifest,
        cache_root_label=cache_root.name,
        observations=tuple((asset.relative_path, asset.bytes, asset.sha256) for asset in ordered_cache),
    )
    download = SourceCacheDownloadReceiptV4(
        schema=DOWNLOAD_RECEIPT_SCHEMA_V4,
        execution_binding=binding,
        enumeration_receipt_sha256=enumeration.receipt_sha256,
        enumeration_mode=enumeration.mode,
        selection_plan_sha256=plan.receipt_sha256,
        source_manifest=manifest,
        evidence=evidence,
        verification_receipt_sha256=verified.receipt_sha256,
    )
    enumeration_path = tmp_path / "receipts" / "upstream-enumeration-v4.json"
    manifest_path = tmp_path / "receipts" / "source-cache-manifest-v4.json"
    download_path = tmp_path / "receipts" / "source-cache-download-receipt-v4.json"
    _write(
        enumeration_path,
        {
            "receipt": asdict(enumeration),
            "receipt_sha256": enumeration.receipt_sha256,
            "schema": UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V4,
        },
    )
    _write(
        manifest_path,
        {
            "manifest": asdict(manifest),
            "manifest_sha256": manifest.receipt_sha256,
            "schema": bridge.SOURCE_CACHE_MANIFEST_ARTIFACT_SCHEMA_V4,
        },
    )
    _write(
        download_path,
        {
            "receipt": asdict(download),
            "receipt_sha256": download.receipt_sha256,
            "schema": DOWNLOAD_RECEIPT_ARTIFACT_SCHEMA_V4,
        },
    )
    return (
        context,
        enumeration,
        manifest,
        download,
        enumeration_path,
        manifest_path,
        download_path,
        cache_root,
    )


def _load_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bridge.MaterializationInputV4:
    context, _, _, _, enumeration_path, manifest_path, download_path, cache_root = _fixture(tmp_path)
    monkeypatch.setattr(
        bridge,
        "load_pa_source_execution_context_v4",
        lambda *, breakdown_root: context,
    )
    return bridge.load_materialization_input_v4(
        enumeration_receipt_path=enumeration_path,
        cache_download_receipt_path=download_path,
        source_manifest_path=manifest_path,
        cache_root=cache_root,
        breakdown_root=tmp_path,
    )


def test_v4_bridge_loads_strict_receipts_rehashes_cache_and_is_v3_parser_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _load_fixture(tmp_path, monkeypatch)

    assert isinstance(inputs, MaterializationInputV3)
    assert inputs.mode == PRODUCTION_MODE
    assert len(inputs.verified_cache.assets) == len(SOURCE_FAMILIES)
    assert inputs.source_identity_sha256 == inputs.source_identity_sha256
    assert all(asset.expected.execution_binding_sha256 for asset in inputs.verified_cache.assets)


def test_cache_fill_loader_skips_global_rehash_but_preserves_receipt_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        context,
        _,
        _,
        _,
        enumeration_path,
        manifest_path,
        download_path,
        cache_root,
    ) = _fixture(tmp_path)
    monkeypatch.setattr(
        bridge,
        "load_pa_source_execution_context_v4",
        lambda *, breakdown_root: context,
    )

    def forbidden_global_rehash(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cache-fill loader invoked the whole-cache rehash")

    monkeypatch.setattr(bridge, "_verify_cache", forbidden_global_rehash)
    inputs = bridge.load_cache_fill_input_v4(
        enumeration_receipt_path=enumeration_path,
        cache_download_receipt_path=download_path,
        source_manifest_path=manifest_path,
        cache_root=cache_root,
        breakdown_root=tmp_path,
    )
    assert isinstance(inputs, MaterializationInputV3)
    assert len(inputs.verified_cache.assets) == len(SOURCE_FAMILIES)


def test_v4_bridge_fails_closed_on_cache_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, _, _, _, enumeration_path, manifest_path, download_path, cache_root = _fixture(tmp_path)
    monkeypatch.setattr(
        bridge,
        "load_pa_source_execution_context_v4",
        lambda *, breakdown_root: context,
    )
    first = next(path for path in cache_root.rglob("*") if path.is_file())
    first.write_bytes(first.read_bytes() + b"tamper")

    with pytest.raises(bridge.CorpusMaterializationV4Error, match="cache bytes"):
        bridge.load_materialization_input_v4(
            enumeration_receipt_path=enumeration_path,
            cache_download_receipt_path=download_path,
            source_manifest_path=manifest_path,
            cache_root=cache_root,
            breakdown_root=tmp_path,
        )


def test_v4_bridge_rejects_unknown_envelope_fields(tmp_path: Path) -> None:
    _, _, _, _, enumeration_path, _, _, _ = _fixture(tmp_path)
    _, payload = bridge.load_canonical_json_snapshot(enumeration_path)
    changed = dict(payload)
    changed["extra"] = "not allowed"
    _write(enumeration_path, changed)

    with pytest.raises(bridge.CorpusMaterializationV4Error, match="fields drifted"):
        bridge.load_upstream_enumeration_artifact_v4(enumeration_path)


def _run_and_assert_forward_finalizer_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    inputs = _load_fixture(tmp_path / "inputs", monkeypatch)
    output = tmp_path / "output"
    output.mkdir()
    (output / "artifacts").mkdir()
    (output / "artifacts" / "payload.bin").write_bytes(b"payload")
    route_strata = {
        family.route.source_family: family.route.stratum
        for family in inputs.upstream_enumeration.families
    }
    documents = [
        ("T", StableDocumentV3("dolma_web", "general", "1" * 64, "training")),
        ("H", StableDocumentV3("dolma_web", "general", "2" * 64, "heldout")),
        ("T", StableDocumentV3("dolma_web", "general", "4" * 64, "training-two")),
        ("T", StableDocumentV3("dolma_web", "general", "5" * 64, "training-three")),
        ("T", StableDocumentV3("dolma_web", "general", "6" * 64, "training-four")),
        (None, StableDocumentV3("dolma_web", "general", "3" * 64, "full-only")),
    ]
    documents.extend(
        (
            None,
            StableDocumentV3(
                source,
                route_strata[source],
                hashlib.sha256(f"fixture:{source}".encode("utf-8")).hexdigest(),
                f"full-only-{source}",
            ),
        )
        for source in SOURCE_FAMILIES
        if source != "dolma_web"
    )
    screen_shards = []
    screen_groups: dict[tuple[str, str], list[StableDocumentV3]] = {}
    for stream, document in (row for row in documents if row[0] is not None):
        assert stream is not None
        screen_groups.setdefault((stream, document.stratum), []).append(document)
    for (stream, stratum), group_documents in screen_groups.items():
        result = production_io.write_jsonl_zstd_shards_v3(
            group_documents,
            output / "shards",
            stream=stream,
            stratum=stratum,
            shard_target_bytes=1024,
        )
        screen_shards.extend(
            {
                **asdict(row),
                "content_identity_sha256": row.content_identity_sha256,
                "identity_relative_path": row.relative_path,
                "relative_path": f"shards/{row.relative_path}",
                "stream": stream,
                "stratum": stratum,
            }
            for row in result.shards
        )
    screen_shards.sort(key=lambda row: row["relative_path"])
    _write(
        output / "artifacts" / "shard-manifest.json",
        {
            "codec_binding_sha256": A2_ZSTD_CODEC_BINDING.receipt_sha256,
            "schema": "weft1_corpus_shard_manifest_v3",
            "shards": screen_shards,
            "tokenizer_fit_input_receipt_sha256": "a" * 64,
        },
    )
    parsed_source_identity = "9" * 64
    core = {
        "algorithm_identity_sha256": "8" * 64,
        "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
        "mode": PRODUCTION_MODE,
        "readiness": "AUTHORITATIVE_INPUTS_D1_READY_NO_GATE_MINT",
        "schema": MATERIALIZER_SCHEMA,
        "source_identity_sha256": parsed_source_identity,
    }
    core_identity = execution_authority_v3_bound_sha256(
        "weft1_corpus_materialized_content_v3", core
    )
    core["content_identity_sha256"] = core_identity
    _write(output / "content-manifest.json", core)
    core_inventory = tuple(
        {
            "bytes": path.stat().st_size,
            "relative_path": path.relative_to(output).as_posix(),
            "sha256": bridge._sha256_file(path),
        }
        for path in sorted(
            (path for path in output.rglob("*") if path.is_file()),
            key=lambda item: item.relative_to(output).as_posix(),
        )
    )
    core_d1 = {
        "content_identity_sha256": core_identity,
        "file_inventory": core_inventory,
        "gate_minted": False,
        "mode": PRODUCTION_MODE,
        "readiness": "AUTHORITATIVE_INPUTS_D1_READY_NO_GATE_MINT",
        "schema": "weft1_corpus_d1_ready_manifest_v3",
        "source_identity_sha256": parsed_source_identity,
    }
    core_d1["d1_ready_identity_sha256"] = execution_authority_v3_bound_sha256(
        "weft1_corpus_d1_ready_inventory_v3", core_d1
    )
    _write(output / "d1-ready-manifest.json", core_d1)
    work_root = tmp_path / "work"
    work_root.mkdir()
    connection = sqlite3.connect(work_root / "materialization.sqlite")
    connection.executescript(
        """
        CREATE TABLE selected_documents (
          document_id TEXT PRIMARY KEY, raw_content_id TEXT NOT NULL UNIQUE,
          source TEXT NOT NULL, stratum TEXT NOT NULL,
          stable_source_record_id TEXT NOT NULL, text BLOB NOT NULL,
          retained_bytes INTEGER NOT NULL, full_ordinal INTEGER NOT NULL UNIQUE
        ) STRICT;
        CREATE TABLE split_documents (
          stream TEXT NOT NULL, stratum TEXT NOT NULL, stream_ordinal INTEGER NOT NULL,
          document_id TEXT NOT NULL, PRIMARY KEY(stream, stratum, stream_ordinal)
        ) STRICT;
        """
    )
    split_ordinals: dict[tuple[str, str], int] = {}
    for ordinal, (stream, document) in enumerate(documents):
        connection.execute(
            "INSERT INTO selected_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                document.document_id,
                document.shard_record_id,
                document.source,
                document.stratum,
                document.stable_source_record_id,
                document.retained_bytes,
                document.retained_byte_count,
                ordinal,
            ),
        )
        if stream is not None:
            key = (stream, document.stratum)
            stream_ordinal = split_ordinals.get(key, 0)
            split_ordinals[key] = stream_ordinal + 1
            connection.execute(
                "INSERT INTO split_documents VALUES (?, ?, ?, ?)",
                (stream, document.stratum, stream_ordinal, document.document_id),
            )
    connection.commit()
    connection.close()
    result = MaterializationResultV3(
        mode=PRODUCTION_MODE,
        source_identity_sha256=parsed_source_identity,
        content_identity_sha256=core_identity,
        d1_ready_manifest_sha256=hashlib.sha256((output / "d1-ready-manifest.json").read_bytes()).hexdigest(),
        output_root=output,
        work_root=work_root,
    )

    upgraded = bridge.finalize_materialization_output_v4(result, inputs)
    _, content = bridge.load_canonical_json_snapshot(output / "content-manifest.json")
    _, d1 = bridge.load_canonical_json_snapshot(output / "d1-ready-manifest.json")

    assert upgraded.content_identity_sha256 == content["content_identity_sha256"]
    assert content["schema"] == bridge.MATERIALIZER_SCHEMA_V4
    assert content["authority_chain"] == list(GTOK_EXECUTION_AUTHORITY_CHAIN_V4)
    assert [row["source_family"] for row in content["release"]["manifest_section"]["attributions"]] == [
        "dolma3",
        "fineweb_edu",
        "stackedu",
    ]
    assert d1["schema"] == bridge.D1_READY_SCHEMA_V4
    assert content["v4_full_corpus"]["document_count"] == 12
    assert content["v4_full_corpus"]["non_screen_full_document_count"] == 7
    _, full_manifest = bridge.load_canonical_json_snapshot(
        output / bridge.FULL_SHARD_MANIFEST_RELATIVE_PATH_V4
    )
    assert [row["source"] for row in full_manifest["sources"]] == list(SOURCE_FAMILIES)
    assert {row["stratum"] for row in full_manifest["sources"]} == set(GTOK_STRATA)
    full_texts: list[str] = []
    full_ids = hashlib.sha256()
    full_locations: dict[str, dict[str, object]] = {}
    for row in full_manifest["shards"]:
        with (output / row["relative_path"]).open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                for record_ordinal, line in enumerate(reader.read().splitlines()):
                    record = json.loads(line)
                    assert record["source"] == row["source"]
                    assert record["stratum"] == row["stratum"]
                    encoded_id = record["id"].encode("ascii")
                    full_ids.update(len(encoded_id).to_bytes(8, "big"))
                    full_ids.update(encoded_id)
                    full_locations[record["id"]] = {
                        "full_shard_relative_path": row["relative_path"],
                        "shard_record_ordinal": record_ordinal,
                        "source": record["source"],
                    }
                    full_texts.append(record["text"])
    assert full_ids.hexdigest() == full_manifest["ordered_raw_content_ids_sha256"]
    assert set(full_texts) == {
        "training",
        "training-two",
        "training-three",
        "training-four",
        "heldout",
        "full-only",
        *(f"full-only-{source}" for source in SOURCE_FAMILIES if source != "dolma_web"),
    }
    physical_rows = {
        row["relative_path"]: {
            "bytes": row["bytes"],
            "path": row["relative_path"],
            "role": "content",
            "sha256": row["sha256"],
        }
        for row in d1["file_inventory"]
    }
    expected_source_strata = tuple(
        (family.route.source_family, family.route.stratum)
        for family in inputs.upstream_enumeration.families
    )
    assert replay_v4._validate_full_corpus_manifest_structure_v4(
        full_manifest,
        output_rows=physical_rows,
        expected_source_strata=expected_source_strata,
    ) == (12, sum(document.retained_byte_count for _stream, document in documents))
    mixed_source = json.loads(json.dumps(full_manifest))
    mixed_source["shards"][0]["source"] = ["dolma_web", "fineweb_edu"]
    with pytest.raises(replay_v3.ParentReplayError, match="source-homogeneous"):
        replay_v4._validate_full_corpus_manifest_structure_v4(
            mixed_source,
            output_rows=physical_rows,
            expected_source_strata=expected_source_strata,
        )
    _, screen_manifest = bridge.load_canonical_json_snapshot(
        output / bridge.SCREEN_SUBMANIFEST_RELATIVE_PATH_V4
    )
    assert screen_manifest["screen_document_count"] == 5
    assert screen_manifest["non_screen_full_document_count"] == 7
    assert screen_manifest["missing_full_document_count"] == 0
    _, d6_physical = bridge.load_canonical_json_snapshot(
        output / bridge.D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
    )
    d6_bytes_before = (
        output / bridge.D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
    ).read_bytes()
    recomputed_d6, recomputed_d6_sha = bridge.validate_physical_d6_evidence_v4(
        root=output, sqlite_path=work_root / "independent-d6.sqlite"
    )
    assert recomputed_d6 == d6_physical
    assert recomputed_d6_sha == hashlib.sha256(d6_bytes_before).hexdigest()
    assert (
        output / bridge.D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
    ).read_bytes() == d6_bytes_before
    assert not (work_root / "independent-d6.sqlite").exists()
    assert "document_id" not in d6_bytes_before.decode("utf-8")
    assert d6_physical["schema"] == bridge.D6_PHYSICAL_EVIDENCE_SCHEMA_V4
    assert d6_physical["status"] == "PHYSICAL_REREAD_PASS_NO_GATE_MINT"
    assert len(d6_physical["consumer_order_receipts"]) == 2
    assert tuple(
        (row["training_seed"], row["data_order_seed"])
        for row in d6_physical["consumer_order_receipts"]
    ) == bridge.GTOK_GOVERNED_DATA_ORDER_SEED_ROWS_V4
    assert tuple(
        (row["training_seed"], row["data_order_seed"])
        for row in d6_physical["consumer_bindings"][:2]
    ) == bridge.GTOK_GOVERNED_DATA_ORDER_SEED_ROWS_V4
    assert {
        row["ordered_raw_content_ids_sha256"]
        for row in d6_physical["consumer_order_receipts"]
    } == {
        row["training_ordered_raw_content_ids_sha256"]
        for row in d6_physical["consumer_bindings"]
    }
    assert (
        d6_physical["tokenizer_fit_input"]["schema"]
        == bridge.TOKENIZER_FIT_INPUT_SCHEMA_V4
    )
    assert "ordered_document_ids_sha256" not in d6_physical["tokenizer_fit_input"]
    fit_texts = tuple(bridge.iter_materialized_tokenizer_fit_texts_v4(output))
    assert fit_texts == (
        "training",
        "training-two",
        "training-three",
        "training-four",
    )
    order_receipt_by_seed = {
        row["training_seed"]: row
        for row in d6_physical["consumer_order_receipts"]
    }
    training_orders = tuple(
        tuple(
            bridge.iter_materialized_training_texts_v4(
                output,
                training_seed=seed,
                expected_physical_d6_evidence_sha256=recomputed_d6_sha,
                expected_consumer_order_receipt=(
                    seed,
                    order_receipt_by_seed[seed]["data_order_seed"],
                    order_receipt_by_seed[seed][
                        "ordered_raw_content_ids_sha256"
                    ],
                ),
            )
        )
        for seed in bridge.GTOK_TRAINING_SEEDS
    )
    first_seed = bridge.GTOK_TRAINING_SEEDS[0]
    first_receipt = order_receipt_by_seed[first_seed]
    with pytest.raises(
        bridge.CorpusMaterializationV4Error,
        match="differs from source-loaded identity",
    ):
        tuple(
            bridge.iter_materialized_training_texts_v4(
                output,
                training_seed=first_seed,
                expected_physical_d6_evidence_sha256="0" * 64,
                expected_consumer_order_receipt=(
                    first_seed,
                    first_receipt["data_order_seed"],
                    first_receipt["ordered_raw_content_ids_sha256"],
                ),
            )
        )
    with pytest.raises(
        bridge.CorpusMaterializationV4Error,
        match="differs from source-loaded receipt",
    ):
        tuple(
            bridge.iter_materialized_training_texts_v4(
                output,
                training_seed=first_seed,
                expected_physical_d6_evidence_sha256=recomputed_d6_sha,
                expected_consumer_order_receipt=(
                    first_seed,
                    first_receipt["data_order_seed"],
                    "0" * 64,
                ),
            )
        )
    def expected_order(order_seed: int) -> tuple[str, ...]:
        rows = []
        for text in fit_texts:
            raw_id = hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324
            key = hashlib.sha256(
                b"WEFT-1/gtok-training-order/raw-content-id/v4\x00"
                + order_seed.to_bytes(8, "big")
                + raw_id.encode("ascii")
            ).digest()
            rows.append((key, raw_id, text))
        return tuple(row[2] for row in sorted(rows))

    expected_data_orders = tuple(
        expected_order(data_order_seed)
        for _training_seed, data_order_seed in (
            bridge.GTOK_GOVERNED_DATA_ORDER_SEED_ROWS_V4
        )
    )
    legacy_training_seed_orders = tuple(
        expected_order(training_seed) for training_seed in bridge.GTOK_TRAINING_SEEDS
    )
    assert training_orders == expected_data_orders
    assert training_orders != legacy_training_seed_orders
    assert len(set(training_orders)) == len(bridge.GTOK_TRAINING_SEEDS)
    assert all(set(order) == set(fit_texts) for order in training_orders)
    assert screen_manifest["d6_physical_evidence_sha256"] == recomputed_d6_sha
    assert (
        content["v4_full_corpus"]["d6_physical_evidence_sha256"]
        == recomputed_d6_sha
    )
    assert [row["stratum"] for row in screen_manifest["groups"]] == [
        stratum for _stream in ("T", "H") for stratum in GTOK_STRATA
    ]
    for group in screen_manifest["groups"]:
        ids = hashlib.sha256()
        locations = hashlib.sha256()
        count = 0
        retained = 0
        prefix = f"shards/{group['stratum']}/{group['stream'].casefold()}-"
        for shard in sorted(
            (row for row in screen_shards if row["relative_path"].startswith(prefix)),
            key=lambda row: row["relative_path"],
        ):
            with (output / shard["relative_path"]).open("rb") as raw:
                with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                    for line in reader.read().splitlines():
                        record = json.loads(line)
                        encoded_id = record["id"].encode("ascii")
                        ids.update(len(encoded_id).to_bytes(8, "big"))
                        ids.update(encoded_id)
                        location = canonical_json_bytes(
                            {
                                "raw_content_id": record["id"],
                                **full_locations[record["id"]],
                            }
                        )
                        locations.update(len(location).to_bytes(8, "big"))
                        locations.update(location)
                        count += 1
                        retained += len(record["text"].encode("utf-8"))
        assert group["document_count"] == count
        assert group["retained_text_bytes"] == retained
        assert group["ordered_raw_content_ids_sha256"] == ids.hexdigest()
        assert group["full_location_projection_sha256"] == locations.hexdigest()
    assert any(
        row["relative_path"] == bridge.BRIDGE_RELATIVE_PATH_V4
        for row in d1["file_inventory"]
    )
    assert any(
        row["relative_path"] == bridge.D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
        for row in d1["file_inventory"]
    )
    assert not d1["gate_minted"]
    return output, work_root


def test_forward_finalizer_upgrades_authority_and_rebuilds_complete_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_and_assert_forward_finalizer_fixture(tmp_path, monkeypatch)


def test_v4_fixture_two_fresh_isolated_replays_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _run_and_assert_forward_finalizer_fixture(first, monkeypatch)
    _run_and_assert_forward_finalizer_fixture(second, monkeypatch)

    def tree_bytes(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(
                (item for item in root.rglob("*") if item.is_file()),
                key=lambda item: item.relative_to(root).as_posix(),
            )
        }

    first_output = first / "output"
    second_output = second / "output"
    assert first_output.resolve() != second_output.resolve()
    assert tree_bytes(first_output) == tree_bytes(second_output)
    for relative in (
        "content-manifest.json",
        "d1-ready-manifest.json",
        bridge.FULL_SHARD_MANIFEST_RELATIVE_PATH_V4,
        bridge.SCREEN_SUBMANIFEST_RELATIVE_PATH_V4,
        bridge.D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4,
        bridge.BRIDGE_RELATIVE_PATH_V4,
    ):
        assert bridge._sha256_file(first_output / relative) == bridge._sha256_file(
            second_output / relative
        )


def test_physical_d6_recompute_rejects_mutated_t_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, work = _run_and_assert_forward_finalizer_fixture(tmp_path, monkeypatch)
    _, manifest = bridge.load_canonical_json_snapshot(
        output / "artifacts" / "shard-manifest.json"
    )
    t_row = next(row for row in manifest["shards"] if row["stream"] == "T")
    with (output / t_row["relative_path"]).open("ab") as handle:
        handle.write(b"mutation")
    with pytest.raises(
        bridge.CorpusMaterializationV4Error, match="shard identity drifted"
    ):
        bridge.recompute_physical_d6_evidence_v4(
            root=output, sqlite_path=work / "mutation-d6.sqlite"
        )


def test_physical_d6_rejects_self_consistent_data_order_seed_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, work = _run_and_assert_forward_finalizer_fixture(tmp_path, monkeypatch)
    path = output / bridge.D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
    _, evidence = bridge.load_canonical_json_snapshot(path)
    changed = json.loads(json.dumps(evidence))
    receipt = changed["consumer_order_receipts"][0]
    governed_data_seed = receipt["data_order_seed"]
    assert governed_data_seed != receipt["training_seed"]
    receipt["data_order_seed"] = receipt["training_seed"]
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = bridge.execution_authority_v4_bound_sha256(
        bridge.CONSUMER_ORDER_SCHEMA_V4, receipt
    )
    for binding in changed["consumer_bindings"]:
        if binding["training_seed"] == receipt["training_seed"]:
            binding["data_order_seed"] = receipt["data_order_seed"]
    changed.pop("evidence_identity_sha256")
    changed["evidence_identity_sha256"] = bridge.execution_authority_v4_bound_sha256(
        bridge.D6_PHYSICAL_EVIDENCE_SCHEMA_V4, changed
    )
    _write(path, changed)

    with pytest.raises(
        bridge.CorpusMaterializationV4Error,
        match="stored physical D6 evidence differs",
    ):
        bridge.validate_physical_d6_evidence_v4(
            root=output, sqlite_path=work / "substituted-seed-d6.sqlite"
        )


def test_current_physical_d6_identity_rejoins_heldout_control_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _work = _run_and_assert_forward_finalizer_fixture(tmp_path, monkeypatch)
    d6_path = output / bridge.D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
    expected_d6_sha256 = hashlib.sha256(d6_path.read_bytes()).hexdigest()
    observed = bridge.assert_current_physical_d6_identity_v4(
        output, expected_physical_sha256=expected_d6_sha256
    )
    assert observed["screen_shard_manifest_sha256"] == hashlib.sha256(
        (output / "artifacts" / "shard-manifest.json").read_bytes()
    ).hexdigest()

    manifest_path = output / "artifacts" / "shard-manifest.json"
    _, manifest = bridge.load_canonical_json_snapshot(manifest_path)
    changed = json.loads(json.dumps(manifest))
    changed["tokenizer_fit_input_receipt_sha256"] = "0" * 64
    _write(manifest_path, changed)
    with pytest.raises(
        bridge.CorpusMaterializationV4Error,
        match="screen-shard manifest differs",
    ):
        bridge.assert_current_physical_d6_identity_v4(
            output, expected_physical_sha256=expected_d6_sha256
        )


def test_confirmation_orders_are_fresh_read_only_views_of_frozen_t(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _work = _run_and_assert_forward_finalizer_fixture(tmp_path, monkeypatch)
    d6_path = output / bridge.D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
    d6_before = d6_path.read_bytes()
    d6_sha256 = hashlib.sha256(d6_before).hexdigest()
    tree_before = {
        path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*")
        if path.is_file()
    }
    _, evidence = bridge.load_canonical_json_snapshot(d6_path)
    base_receipts = {
        row["training_seed"]: row for row in evidence["consumer_order_receipts"]
    }
    base_orders = tuple(
        tuple(
            bridge.iter_materialized_training_texts_v4(
                output,
                training_seed=seed,
                expected_physical_d6_evidence_sha256=d6_sha256,
                expected_consumer_order_receipt=(
                    seed,
                    base_receipts[seed]["data_order_seed"],
                    base_receipts[seed]["ordered_raw_content_ids_sha256"],
                ),
            )
        )
        for seed in bridge.GTOK_TRAINING_SEEDS
    )
    run_seeds = (
        9_884_118_125_684_999_954,
        7_190_589_679_906_404_951,
    )
    order_receipts = tuple(
        bridge.build_materialized_confirmation_order_v4(
            output,
            confirmation_run_seed=run_seed,
            data_order_seed=derive_module_seed(
                A2_CAMPAIGN_ROOT_SEED,
                f"gtok.data.shared.{run_seed}",
            ),
            expected_physical_d6_evidence_sha256=d6_sha256,
        )
        for run_seed in run_seeds
    )
    confirmation_orders = tuple(
        tuple(
            bridge.iter_materialized_confirmation_training_texts_v4(
                output,
                order_receipt=receipt,
            )
        )
        for receipt in order_receipts
    )

    frozen_multiset = set(base_orders[0])
    assert len(set(confirmation_orders)) == 2
    assert all(order not in base_orders for order in confirmation_orders)
    assert all(set(order) == frozen_multiset for order in confirmation_orders)
    assert {
        receipt.document_multiset_sha256 for receipt in order_receipts
    } == {
        row["document_multiset_sha256"]
        for row in evidence["consumer_order_receipts"]
    }
    assert all(
        receipt.order_key_domain == bridge.CONSUMER_ORDER_KEY_DOMAIN_V4
        and receipt.schema == bridge.CONFIRMATION_CONSUMER_ORDER_SCHEMA_V4
        and len(receipt.receipt_sha256) == 64
        for receipt in order_receipts
    )

    with pytest.raises(ValueError, match="run-harness tree"):
        bridge.build_materialized_confirmation_order_v4(
            output,
            confirmation_run_seed=run_seeds[0],
            data_order_seed=order_receipts[0].data_order_seed + 1,
            expected_physical_d6_evidence_sha256=d6_sha256,
        )
    tampered = replace(
        order_receipts[0],
        ordered_raw_content_ids_sha256="0" * 64,
    )
    with pytest.raises(
        bridge.CorpusMaterializationV4Error,
        match="differs from its read-only receipt",
    ):
        tuple(
            bridge.iter_materialized_confirmation_training_texts_v4(
                output,
                order_receipt=tampered,
            )
        )

    assert d6_path.read_bytes() == d6_before
    tree_after = {
        path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert tree_after == tree_before


def test_v4_runtime_builder_binding_is_the_existing_v3_builder() -> None:
    assert replay_v4.RUNTIME_BUILDER_PATH_V4 == replay_v3.RUNTIME_BUILDER_PATH_V1
    assert replay_v4.RUNTIME_BUILDER_PATH_V4.is_file()
    compatibility = replay_v4._compatibility_files_v4()
    assert compatibility["runtime_builder"] == replay_v4.RUNTIME_BUILDER_PATH_V4
    assert bridge._sha256_file(compatibility["runtime_builder"]) == hashlib.sha256(
        replay_v4.RUNTIME_BUILDER_PATH_V4.read_bytes()
    ).hexdigest()
