from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Iterable

import pytest

import training.weft1_corpus_fetch_a3 as fetch
from training.weft1_corpus_a2 import A2_CAMPAIGN_ROOT_SEED
from training.weft1_corpus_a3 import (
    A3_AUTHORITY_SHA256,
    A3BreakdownPending,
    A3EffectiveRouteResolution,
    EffectiveSourceRouteA3,
)
from training.weft1_corpus_enumeration_a2 import (
    ExternalLocatorAssetV3,
    ExternalLocatorListingV3,
)
from training.weft1_corpus_breakdown_a3 import (
    DOLMA_TOP_QUALITY_ASSERTION_A3_V1,
    FINEWEB_MAIN_DATA_ASSERTION_A3_V1,
    FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3,
    FIXTURE_OBSERVATION_MODE_A3,
    PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3,
    PRODUCTION_OBSERVATION_MODE_A3,
    PathMemberReceiptA3,
    PinnedSemanticEvidenceA3,
    PriorRouteDeclarationA3,
    build_dolma_path_breakdown_a3,
    build_fineweb_path_breakdown_a3,
    build_upstream_path_breakdown_a3,
)
from training.weft1_gtok_a1_contract import (
    load_source_route_manifest,
)


def _fineweb_ids() -> tuple[str, ...]:
    return tuple(
        f"CC-MAIN-{2000 + index // 100:04d}-{index % 100:02d}"
        for index in range(110)
    )


def _effective_routes() -> tuple[EffectiveSourceRouteA3, ...]:
    routes: list[EffectiveSourceRouteA3] = []
    for base in load_source_route_manifest().routes:
        overlay = base.source_family in {"dolma_web", "fineweb_edu"}
        selector = base.asset_selector
        asset_count = base.asset_count
        available_bytes = base.available_bytes
        resolution = "PASSTHROUGH_A1_UNCHANGED"
        if base.source_family == "dolma_web":
            selector = "data/common_crawl-*-0019/*.jsonl.zst"
            asset_count = 2
            available_bytes = 20
            resolution = "CONFIRM_TOP_BUCKET_SELECTOR_REMINT_DECLARATION"
        elif base.source_family == "fineweb_edu":
            selector = "semantic:fineweb_edu_configured_110_cc_main_dumps_all_parquet_a3_v1"
            asset_count = 110
            available_bytes = 110
            resolution = "WIDEN_TO_ALL_MAIN_DATA_CC_DUMPS"
        routes.append(
            EffectiveSourceRouteA3(
                source_family=base.source_family,
                stratum=base.stratum,
                role=base.role,
                repository=base.repository,
                config=base.config,
                revision=base.revision,
                split=base.split,
                asset_selector=selector,
                selection_rule=base.selection_rule,
                declared_license=base.declared_license,
                card_url=base.card_url,
                card_sha256=base.card_sha256,
                asset_count=asset_count,
                available_bytes=available_bytes,
                available_bytes_basis=base.available_bytes_basis,
                required_bytes=1,
                lineage_evidence=base.lineage_evidence,
                parse_policy=base.parse_policy,
                external_locator_manifest_sha256=(
                    base.external_locator_manifest_sha256
                ),
                base_route_receipt_sha256=base.receipt_sha256,
                route_resolution=resolution,
                breakdown_artifact_receipt_sha256=("a" * 64 if overlay else None),
                breakdown_family_projection_sha256=(
                    ("b" if base.source_family == "dolma_web" else "c") * 64
                    if overlay
                    else None
                ),
            )
        )
    return tuple(routes)


def _context() -> fetch.PASourceExecutionContextV4:
    resolution = A3EffectiveRouteResolution(
        routes=_effective_routes(),
        effective_route_identity_sha256="d" * 64,
        breakdown_artifact_physical_sha256="e" * 64,
        breakdown_artifact_receipt_sha256="a" * 64,
        family_projection_sha256s=(
            ("dolma_web", "b" * 64),
            ("fineweb_edu", "c" * 64),
        ),
    )
    return fetch.PASourceExecutionContextV4(
        resolution=resolution,
        binding=fetch.PAExecutionBindingV4.from_resolution(resolution),
        fineweb_cc_dump_ids=_fineweb_ids(),
        dolma_top_bucket_group_ids=(
            "common_crawl-art-0019",
            "common_crawl-science-0019",
        ),
    )


def _path_member(path: str, size: int) -> PathMemberReceiptA3:
    return PathMemberReceiptA3(
        path=path,
        upstream_bytes=size,
        blob_identity_kind="git_sha1",
        blob_identity=hashlib.sha1(path.encode("utf-8")).hexdigest(),
    )


def _governed_context() -> tuple[
    fetch.PASourceExecutionContextV4,
    tuple[PathMemberReceiptA3, ...],
    tuple[PathMemberReceiptA3, ...],
]:
    base = {row.source_family: row for row in load_source_route_manifest().routes}
    dolma_members = (
        _path_member("README.md", 1),
        _path_member("data/common_crawl-art-0019/00000.jsonl.zst", 10),
        _path_member("data/common_crawl-science-0019/00000.jsonl.zst", 10),
    )
    fineweb_members = tuple(
        [
            _path_member(
                f"data/{group}/000_00000.parquet",
                1,
            )
            for group in _fineweb_ids()
        ]
        + [_path_member("README.md", 1)]
    )
    dolma = build_dolma_path_breakdown_a3(
        dolma_members,
        repository=base["dolma_web"].repository,
        revision=base["dolma_web"].revision,
        prior_declaration=PriorRouteDeclarationA3(
            source_family="dolma_web",
            asset_selector=base["dolma_web"].asset_selector,
            asset_count=base["dolma_web"].asset_count,
            available_bytes=base["dolma_web"].available_bytes,
            declaration_receipt_sha256=base["dolma_web"].receipt_sha256,
        ),
        semantic_evidence=PinnedSemanticEvidenceA3(
            evidence_id="dolma-test",
            locator="https://example.invalid/dolma",
            pin="test",
            content_sha256="4" * 64,
            assertion=DOLMA_TOP_QUALITY_ASSERTION_A3_V1,
        ),
    )
    fineweb = build_fineweb_path_breakdown_a3(
        fineweb_members,
        repository=base["fineweb_edu"].repository,
        revision=base["fineweb_edu"].revision,
        prior_declaration=PriorRouteDeclarationA3(
            source_family="fineweb_edu",
            asset_selector=base["fineweb_edu"].asset_selector,
            asset_count=base["fineweb_edu"].asset_count,
            available_bytes=base["fineweb_edu"].available_bytes,
            declaration_receipt_sha256=base["fineweb_edu"].receipt_sha256,
        ),
        semantic_evidence=PinnedSemanticEvidenceA3(
            evidence_id="fineweb-test",
            locator="https://example.invalid/fineweb",
            pin="test",
            content_sha256="5" * 64,
            assertion=FINEWEB_MAIN_DATA_ASSERTION_A3_V1,
        ),
        configured_main_dump_ids=_fineweb_ids(),
    )
    breakdown = build_upstream_path_breakdown_a3(
        authority_sha256=A3_AUTHORITY_SHA256,
        observation_mode=PRODUCTION_OBSERVATION_MODE_A3,
        observation_client_identity=PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3,
        dolma=dolma,
        fineweb=fineweb,
    )
    routes = tuple(
        replace(
            route,
            breakdown_artifact_receipt_sha256=breakdown.receipt_sha256,
        )
        if route.source_family in {"dolma_web", "fineweb_edu"}
        else route
        for route in _effective_routes()
    )
    resolution = A3EffectiveRouteResolution(
        routes=routes,
        effective_route_identity_sha256="d" * 64,
        breakdown_artifact_physical_sha256="e" * 64,
        breakdown_artifact_receipt_sha256=breakdown.receipt_sha256,
        family_projection_sha256s=(
            ("dolma_web", "b" * 64),
            ("fineweb_edu", "c" * 64),
        ),
    )
    context = fetch.PASourceExecutionContextV4(
        resolution=resolution,
        binding=fetch.PAExecutionBindingV4.from_resolution(resolution),
        fineweb_cc_dump_ids=_fineweb_ids(),
        dolma_top_bucket_group_ids=(
            "common_crawl-art-0019",
            "common_crawl-science-0019",
        ),
        overlay_physical_sha256="6" * 64,
        overlay_identity_sha256="7" * 64,
        semantic_evidence_artifact_physical_sha256="9" * 64,
        semantic_evidence_artifact_receipt_sha256="a" * 64,
        semantic_evidence_family_receipt_sha256s=(
            ("dolma_web", dolma.semantic_evidence.receipt_sha256),
            ("fineweb_edu", fineweb.semantic_evidence.receipt_sha256),
        ),
        path_breakdown=breakdown,
    )
    return context, dolma_members, fineweb_members


def _route(context: fetch.PASourceExecutionContextV4, family: str) -> EffectiveSourceRouteA3:
    return next(item for item in context.routes if item.source_family == family)


def _entry(locator: str, size: int = 1) -> dict[str, object]:
    return {
        "blob_id": hashlib.sha1(locator.encode("utf-8")).hexdigest(),
        "path": locator,
        "size": size,
        "type": "file",
    }


def test_v4_binding_preserves_a2_seed_and_all_a3_authority_edges() -> None:
    context = _context()
    binding = context.binding

    assert binding.authority_sha256 == A3_AUTHORITY_SHA256
    assert binding.campaign_root_seed == A2_CAMPAIGN_ROOT_SEED
    assert binding.effective_route_identity_sha256 == "d" * 64
    assert binding.breakdown_artifact_physical_sha256 == "e" * 64
    assert binding.breakdown_artifact_receipt_sha256 == "a" * 64
    assert binding.receipt_sha256 == context.binding_sha256


def test_fineweb_selector_accepts_numeric_and_legacy_names_in_exact_110_only() -> None:
    context = _context()
    route = _route(context, "fineweb_edu")
    included = context.fineweb_cc_dump_ids[0]

    assert fetch.locator_matches_effective_route_v4(
        context, route, f"data/{included}/000_00000.parquet"
    )
    assert fetch.locator_matches_effective_route_v4(
        context, route, f"data/{included}/train-00000-of-00010.parquet"
    )
    assert not fetch.locator_matches_effective_route_v4(
        context, route, "data/CC-MAIN-2099-99/000_00000.parquet"
    )
    assert not fetch.locator_matches_effective_route_v4(
        context, route, "data/sample-10BT/000_00000.parquet"
    )
    assert not fetch.locator_matches_effective_route_v4(
        context, route, f"data/{included}/nested/000_00000.parquet"
    )


def test_dolma_selector_admits_only_observer_proven_bucket_0019_groups() -> None:
    context = _context()
    route = _route(context, "dolma_web")

    assert fetch.locator_matches_effective_route_v4(
        context, route, "data/common_crawl-art-0019/00000.jsonl.zst"
    )
    assert not fetch.locator_matches_effective_route_v4(
        context, route, "data/common_crawl-art-0018/00000.jsonl.zst"
    )
    assert not fetch.locator_matches_effective_route_v4(
        context, route, "data/common_crawl-mystery-0019/00000.jsonl.zst"
    )


def test_context_rejects_anything_other_than_exact_110_fineweb_dumps() -> None:
    context = _context()
    with pytest.raises(ValueError, match="exactly 110"):
        fetch.PASourceExecutionContextV4(
            resolution=context.resolution,
            binding=context.binding,
            fineweb_cc_dump_ids=context.fineweb_cc_dump_ids[:-1],
            dolma_top_bucket_group_ids=context.dolma_top_bucket_group_ids,
        )


def test_production_context_propagates_pending_a3_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def pending() -> object:
        raise A3BreakdownPending("still pending")

    monkeypatch.setattr(fetch, "load_effective_route_overlay_a3", pending)
    with pytest.raises(A3BreakdownPending, match="still pending"):
        fetch.load_pa_source_execution_context_v4(breakdown_root=fetch.Path("unused"))


class _OfflineTree:
    def __init__(self, rows: dict[tuple[str, str], tuple[object, ...]]) -> None:
        self.rows = rows

    def __call__(self, **kwargs: object) -> Iterable[object]:
        return self.rows[(str(kwargs["repo_id"]), str(kwargs["revision"]))]


def _fixture_enumeration() -> fetch.UpstreamEnumerationReceiptV4:
    context = _context()
    trees: dict[tuple[str, str], list[object]] = {}
    locators = {
        "dolma_web": "data/common_crawl-art-0019/00000.jsonl.zst",
        "stackedu": "data/stack_edu-unit/00000.jsonl.zst",
        "finemath_3plus": "finemath-3plus/train-00000.parquet",
        "arxiv": "data/rpj-proofpile-arxiv/00000.jsonl.zst",
        "olmocr": "data/olmocr_science_pdfs-unit/00000.jsonl.zst",
    }
    for route in context.routes:
        if route.source_family == "wikipedia_wikibooks":
            continue
        key = (route.repository, route.revision)
        if route.source_family == "fineweb_edu":
            trees.setdefault(key, []).extend(
                _entry(f"data/{group}/000_00000.parquet")
                for group in context.fineweb_cc_dump_ids
            )
        else:
            trees.setdefault(key, []).append(_entry(locators[route.source_family]))
    frozen = {key: tuple(rows) for key, rows in trees.items()}

    def external(**kwargs: object) -> ExternalLocatorListingV3:
        return ExternalLocatorListingV3.fixture(
            source_family="wikipedia_wikibooks",
            external_locator_manifest_sha256=str(kwargs["expected_manifest_sha256"]),
            available_bytes=10,
            available_bytes_basis="pinned repository card reported UTF-8 bytes",
            assets=(
                ExternalLocatorAssetV3(
                    locator="https://olmo-data.org/dolma-v1_7/wiki/wiki-0000.json.gz",
                    upstream_bytes=1,
                    content_sha256="1" * 64,
                ),
                ExternalLocatorAssetV3(
                    locator="https://olmo-data.org/dolma-v1_7/wiki/wiki-0001.json.gz",
                    upstream_bytes=1,
                    content_sha256="2" * 64,
                ),
            ),
        )

    return fetch.enumerate_upstream_assets_v4(
        context=context,
        list_repo_tree=_OfflineTree(frozen),
        enumerate_external_locators=external,
    )


def test_fixture_enumeration_includes_all_110_numeric_fineweb_assets() -> None:
    receipt = _fixture_enumeration()
    fineweb = next(row for row in receipt.families if row.source_family == "fineweb_edu")

    assert receipt.mode == fetch.FIXTURE_MODE
    assert receipt.authoritative is False
    assert len(fineweb.assets) == 110
    assert {
        asset.asset_locator.split("/")[1] for asset in fineweb.assets
    } == set(_fineweb_ids())
    assert all(asset.asset_locator.endswith("/000_00000.parquet") for asset in fineweb.assets)


def test_enumeration_selection_and_plan_repeat_the_same_a3_binding() -> None:
    enumeration = _fixture_enumeration()
    plan, selection = fetch.select_required_asset_prefixes_v4(enumeration)
    binding_sha = enumeration.execution_binding.receipt_sha256

    assert plan.execution_binding == enumeration.execution_binding
    assert selection.execution_binding == enumeration.execution_binding
    assert all(row.execution_binding_sha256 == binding_sha for row in selection.families)
    assert all(asset.execution_binding_sha256 == binding_sha for asset in plan.assets)
    assert selection.selection_plan_sha256 == plan.receipt_sha256


def test_v4_cache_asset_accepts_new_numeric_fineweb_name_without_v3_relabeling() -> None:
    context = _context()
    route = _route(context, "fineweb_edu")
    locator = f"data/{context.fineweb_cc_dump_ids[-1]}/000_00000.parquet"
    asset = fetch.SourceCacheAssetV4(
        source_family="fineweb_edu",
        repository=route.repository,
        config=route.config,
        revision=route.revision,
        split=route.split,
        asset_locator=locator,
        relative_path="assets/fineweb_edu/item.parquet",
        bytes=1,
        sha256="f" * 64,
        effective_route_receipt_sha256=route.receipt_sha256,
        execution_binding_sha256=context.binding_sha256,
    )

    assert asset.asset_locator == locator
    assert asset.execution_binding_sha256 == context.binding_sha256
    assert asset.asset_identity_sha256 != route.receipt_sha256


def _raw_member(member: PathMemberReceiptA3) -> dict[str, object]:
    return {
        "blob_id": member.blob_identity,
        "path": member.path,
        "size": member.upstream_bytes,
        "type": "file",
    }


def test_live_replay_rejects_same_count_same_bytes_swapped_member() -> None:
    context, dolma_members, fineweb_members = _governed_context()
    routes = {route.source_family: route for route in context.routes}
    exact_trees = {
        (routes["dolma_web"].repository, routes["dolma_web"].revision): tuple(
            _raw_member(item) for item in dolma_members
        ),
        (routes["fineweb_edu"].repository, routes["fineweb_edu"].revision): tuple(
            _raw_member(item) for item in fineweb_members
        ),
    }
    fetch._replay_changed_family_trees_v4(context, exact_trees)

    changed = list(fineweb_members)
    original = changed[0]
    group = original.path.split("/")[1]
    changed[0] = _path_member(f"data/{group}/999_99999.parquet", original.upstream_bytes)
    swapped_trees = dict(exact_trees)
    swapped_trees[
        (routes["fineweb_edu"].repository, routes["fineweb_edu"].revision)
    ] = tuple(_raw_member(item) for item in changed)

    assert len(changed) == len(fineweb_members)
    assert sum(item.upstream_bytes for item in changed) == sum(
        item.upstream_bytes for item in fineweb_members
    )
    with pytest.raises(ValueError, match="replay differs"):
        fetch._replay_changed_family_trees_v4(context, swapped_trees)


def _code_identity(context: fetch.PASourceExecutionContextV4) -> fetch.SourcePrepCodeIdentityV4:
    return fetch.SourcePrepCodeIdentityV4(
        mode=fetch.AUTHORITATIVE_MODE,
        execution_binding=context.binding,
        git_commit="8" * 40,
        files=tuple(
            fetch.SourcePrepImplementationFileV4(
                repo_path=path,
                bytes=1,
                sha256=hashlib.sha256(path.encode("utf-8")).hexdigest(),
                git_blob_sha1=hashlib.sha1(path.encode("utf-8")).hexdigest(),
            )
            for path in fetch.SOURCE_PREP_IMPLEMENTATION_REPO_PATHS_V4
        ),
    )


def _attestation(
    context: fetch.PASourceExecutionContextV4,
    code: fetch.SourcePrepCodeIdentityV4,
) -> fetch.A3ReplayAttestationV4:
    assert context.path_breakdown is not None
    assert context.overlay_physical_sha256 is not None
    assert context.overlay_identity_sha256 is not None
    return fetch.A3ReplayAttestationV4(
        schema=fetch.A3_REPLAY_ATTESTATION_SCHEMA_V4,
        status="ATTESTED_CLEAN_HEAD_LIVE_REPLAY_PASS",
        authorizes_downloads=True,
        git_commit=code.git_commit,
        git_status="CLEAN",
        authority_sha256=A3_AUTHORITY_SHA256,
        execution_binding_sha256=context.binding_sha256,
        source_prep_code_identity_sha256=code.receipt_sha256,
        semantic_evidence_artifact_physical_sha256=str(
            context.semantic_evidence_artifact_physical_sha256
        ),
        semantic_evidence_artifact_receipt_sha256=str(
            context.semantic_evidence_artifact_receipt_sha256
        ),
        semantic_evidence_family_receipt_sha256s=(
            context.semantic_evidence_family_receipt_sha256s
        ),
        breakdown_artifact_physical_sha256=(
            context.binding.breakdown_artifact_physical_sha256
        ),
        breakdown_artifact_receipt_sha256=(
            context.binding.breakdown_artifact_receipt_sha256
        ),
        overlay_artifact_physical_sha256=context.overlay_physical_sha256,
        overlay_identity_sha256=context.overlay_identity_sha256,
        effective_route_identity_sha256=(
            context.binding.effective_route_identity_sha256
        ),
        huggingface_hub_distribution="huggingface-hub",
        huggingface_hub_version="1.24.0",
        observation_mode=PRODUCTION_OBSERVATION_MODE_A3,
        observation_client_identity_sha256=(
            PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3.receipt_sha256
        ),
        live_replay_status="PASS_EXACT_BREAKDOWN_REPLAY",
        live_replay_receipt_sha256=context.path_breakdown.receipt_sha256,
    )


def test_external_clean_commit_attestation_is_required_before_online_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, unused_dolma, unused_fineweb = _governed_context()
    del unused_dolma, unused_fineweb
    code = _code_identity(context)
    valid = _attestation(context, code)
    fetch.validate_a3_replay_attestation_v4(
        valid,
        context=context,
        source_prep_code_identity=code,
    )
    wrong_evidence = replace(
        valid,
        semantic_evidence_artifact_physical_sha256="0" * 64,
    )
    with pytest.raises(fetch.SourceFetchV4Error, match="differs from code"):
        fetch.validate_a3_replay_attestation_v4(
            wrong_evidence,
            context=context,
            source_prep_code_identity=code,
        )
    bad = replace(valid, effective_route_identity_sha256="0" * 64)
    constructed = False

    def should_not_construct(*args: object, **kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("network cache constructed before authority validation")

    monkeypatch.setattr(fetch, "ExternalResourceCacheV3", should_not_construct)
    with pytest.raises(fetch.SourceFetchV4Error, match="differs from code"):
        fetch.prepare_pa_sources_online_v4(
            context=context,
            cache_root=fetch.Path("cache"),
            transport_cache_root=fetch.Path("transport"),
            receipt_root=fetch.Path("receipts"),
            source_prep_code_identity=code,
            replay_attestation=bad,
        )
    assert constructed is False


def test_private_authoritative_enumerator_checks_full_attestation_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, unused_dolma, unused_fineweb = _governed_context()
    del unused_dolma, unused_fineweb
    code = _code_identity(context)
    attestation = _attestation(context, code)
    wrong_code = replace(code, git_commit="9" * 40)
    runtime_touched = False

    def should_not_query_runtime(*args: object, **kwargs: object) -> str:
        del args, kwargs
        nonlocal runtime_touched
        runtime_touched = True
        raise AssertionError("runtime queried before full attestation validation")

    monkeypatch.setattr(fetch.metadata, "version", should_not_query_runtime)
    with pytest.raises(fetch.SourceFetchV4Error, match="differs from code"):
        fetch._enumerate_authoritative_upstream_assets_v4(
            context=context,
            open_resource=lambda _: pytest.fail("resource opened before validation"),
            replay_attestation=attestation,
            source_prep_code_identity=wrong_code,
        )
    assert runtime_touched is False


def test_authoritative_cache_path_is_private_to_online_wrapper(
    tmp_path: fetch.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enumeration = _fixture_enumeration()
    plan, unused_selection = fetch.select_required_asset_prefixes_v4(enumeration)
    del unused_selection
    monkeypatch.setattr(
        fetch.UpstreamEnumerationReceiptV4,
        "authoritative",
        property(lambda _: True),
    )

    with pytest.raises(fetch.SourceFetchV4Error, match="private to online"):
        fetch.materialize_source_cache_v4(
            enumeration,
            plan,
            tmp_path / "cache",
            open_upstream=lambda _: pytest.fail("asset opened outside wrapper"),
        )
    assert "enumerate_authoritative_upstream_assets_v4" not in fetch.__all__
    assert "materialize_source_cache_v4" not in fetch.__all__
    assert "finalize_source_cache_v4" not in fetch.__all__


def test_execution_context_rejects_fixture_observer_identity() -> None:
    context, unused_dolma, unused_fineweb = _governed_context()
    del unused_dolma, unused_fineweb
    assert context.path_breakdown is not None
    fixture_breakdown = replace(
        context.path_breakdown,
        observation_mode=FIXTURE_OBSERVATION_MODE_A3,
        observation_client_identity=FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3,
        observation_client_identity_sha256=(
            FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3.receipt_sha256
        ),
    )
    with pytest.raises(ValueError, match="rejects non-authoritative"):
        fetch.PASourceExecutionContextV4(
            resolution=context.resolution,
            binding=context.binding,
            fineweb_cc_dump_ids=context.fineweb_cc_dump_ids,
            dolma_top_bucket_group_ids=context.dolma_top_bucket_group_ids,
            overlay_physical_sha256=context.overlay_physical_sha256,
            overlay_identity_sha256=context.overlay_identity_sha256,
            semantic_evidence_artifact_physical_sha256=(
                context.semantic_evidence_artifact_physical_sha256
            ),
            semantic_evidence_artifact_receipt_sha256=(
                context.semantic_evidence_artifact_receipt_sha256
            ),
            semantic_evidence_family_receipt_sha256s=(
                context.semantic_evidence_family_receipt_sha256s
            ),
            path_breakdown=fixture_breakdown,
        )
