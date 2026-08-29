from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Iterable

import pytest

import scripts.run_weft1_corpus_fetch_a2 as fetch_cli
import training.weft1_corpus_fetch_a2 as fetch_module
from training.weft1_corpus_enumeration_a2 import (
    FIXTURE_MODE,
    ExternalLocatorAssetV3,
    ExternalLocatorListingV3,
    enumerate_upstream_assets_v3,
    load_upstream_enumeration_receipt_v3,
)
from training.weft1_corpus_fetch_a2 import (
    DOWNLOAD_ARTIFACT_NAME,
    ENUMERATION_ARTIFACT_NAME,
    EXTERNAL_TRANSPORT_ARTIFACT_NAME,
    SELECTION_ARTIFACT_NAME,
    SOURCE_MANIFEST_NAME,
    ExternalResourceCacheV3,
    SourceFetchError,
    build_external_transport_receipt_v1,
    load_external_transport_receipt_v1,
    prepare_selected_source_cache_v3,
    select_required_asset_prefixes_v3,
    write_external_transport_receipt_v1,
)
from training.weft1_corpus_source_io_a2 import (
    SourceTransportError,
    load_source_cache_download_receipt_v3,
)
from training.weft1_corpus_sources_a2 import load_exact_source_routes_v3
from training.weft1_gtok_a1_contract import (
    SOURCE_FAMILIES,
    SOURCE_ROUTE_MANIFEST_PATH,
    load_source_route_manifest,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture_code_identity() -> fetch_module.SourcePrepCodeIdentityV1:
    return fetch_module.SourcePrepCodeIdentityV1(
        schema=fetch_module.SOURCE_PREP_CODE_IDENTITY_SCHEMA_V1,
        mode=fetch_module.SOURCE_PREP_FIXTURE_MODE,
        git_commit="1" * 40,
        files=(
            fetch_module.SourcePrepImplementationFileV1(
                repo_path="fixture/source-prep.py",
                bytes=1,
                sha256=_sha(b"x"),
                git_blob_sha1="2" * 40,
            ),
        ),
    )


def _attested_code_identity(seed: str = "a") -> fetch_module.SourcePrepCodeIdentityV1:
    rows = tuple(
        fetch_module.SourcePrepImplementationFileV1(
            repo_path=repo_path,
            bytes=index + 1,
            sha256=_sha((seed * (index + 1)).encode("ascii")),
            git_blob_sha1=hashlib.sha1(
                f"{seed}:{repo_path}".encode("utf-8")
            ).hexdigest(),
        )
        for index, repo_path in enumerate(
            fetch_module.SOURCE_PREP_IMPLEMENTATION_REPO_PATHS_V1
        )
    )
    return fetch_module.SourcePrepCodeIdentityV1(
        schema=fetch_module.SOURCE_PREP_CODE_IDENTITY_SCHEMA_V1,
        mode=fetch_module.SOURCE_PREP_ATTESTED_HEAD_MODE,
        git_commit=hashlib.sha1(seed.encode("ascii")).hexdigest(),
        files=rows,
    )


def test_attested_code_identity_requires_the_exact_governed_inventory() -> None:
    identity = _attested_code_identity()

    def rebuild(files):
        return fetch_module.SourcePrepCodeIdentityV1(
            schema=identity.schema,
            mode=identity.mode,
            git_commit=identity.git_commit,
            files=tuple(files),
        )

    with pytest.raises(ValueError, match="exact file inventory"):
        rebuild(identity.files[:-1])
    extra = fetch_module.SourcePrepImplementationFileV1(
        repo_path="training/zz_extra.py",
        bytes=1,
        sha256=_sha(b"z"),
        git_blob_sha1="9" * 40,
    )
    with pytest.raises(ValueError, match="exact file inventory"):
        rebuild(identity.files + (extra,))
    with pytest.raises(ValueError, match="not ordered"):
        rebuild(tuple(reversed(identity.files)))


def test_cli_code_attestation_binds_clean_head_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(arguments, **kwargs):
        del kwargs
        if arguments[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="2" * 40 + "\n")
        if "status" in arguments:
            return SimpleNamespace(stdout=b"")
        if "rev-parse" in arguments:
            return SimpleNamespace(stdout="5" * 40 + "\n")
        assert "hash-object" in arguments
        return SimpleNamespace(stdout="5" * 40 + "\n")

    monkeypatch.setattr(fetch_cli.subprocess, "run", run)
    identity = fetch_cli._attest_clean_source_prep_code()
    assert identity.git_commit == "2" * 40
    assert identity.mode == fetch_module.SOURCE_PREP_ATTESTED_HEAD_MODE
    assert tuple(row.repo_path for row in identity.files) == (
        fetch_module.SOURCE_PREP_IMPLEMENTATION_REPO_PATHS_V1
    )
    assert identity.files[0].sha256 == _sha(
        fetch_cli.ROOT.joinpath(
            *identity.files[0].repo_path.split("/")
        ).read_bytes()
    )


def test_cli_code_attestation_rejects_dirty_or_head_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dirty(arguments, **kwargs):
        del kwargs
        if arguments[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="3" * 40 + "\n")
        assert "status" in arguments
        return SimpleNamespace(stdout=b" M training/weft1_corpus_fetch_a2.py\n")

    monkeypatch.setattr(fetch_cli.subprocess, "run", dirty)
    with pytest.raises(RuntimeError, match="completely clean"):
        fetch_cli._attest_clean_source_prep_code()

    def drift(arguments, **kwargs):
        del kwargs
        if arguments[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="4" * 40 + "\n")
        if "status" in arguments:
            return SimpleNamespace(stdout=b"")
        if "rev-parse" in arguments:
            return SimpleNamespace(stdout="6" * 40 + "\n")
        assert "hash-object" in arguments
        return SimpleNamespace(stdout="7" * 40 + "\n")

    monkeypatch.setattr(fetch_cli.subprocess, "run", drift)
    with pytest.raises(RuntimeError, match="differs from HEAD"):
        fetch_cli._attest_clean_source_prep_code()


class _RedirectedResponse(io.BytesIO):
    def __init__(self, payload: bytes, final_url: str) -> None:
        super().__init__(payload)
        self._final_url = final_url
        self.was_closed = False

    def geturl(self) -> str:
        return self._final_url

    def close(self) -> None:
        self.was_closed = True
        super().close()


class _RedirectingOpener:
    def __init__(
        self,
        handler,
        *,
        targets: tuple[str, ...],
        payload: bytes,
    ) -> None:
        self._handler = handler
        self._targets = targets
        self._payload = payload
        self.hop_responses: list[io.BytesIO] = []

    def open(self, request, timeout: int):
        assert timeout == 120
        current = request
        for target in self._targets:
            hop_response = io.BytesIO(b"redirect")
            self.hop_responses.append(hop_response)
            try:
                redirected = self._handler.redirect_request(
                    current,
                    hop_response,
                    307,
                    "Found",
                    {},
                    target,
                )
            finally:
                hop_response.close()
            assert redirected is not None
            current = redirected
        return _RedirectedResponse(self._payload, current.full_url)


def _accepted_hf_cache_target() -> str:
    return (
        "https://huggingface.co"
        f"{fetch_module._HF_DOLMA_MANIFEST_CACHE_PATH}"
        f"?{fetch_module._HF_DOLMA_MANIFEST_CACHE_QUERY}"
    )


def _locator(family: str, index: int) -> str:
    return {
        "dolma_web": f"data/common_crawl-fetch-0019/{index:05d}.jsonl.zst",
        "wikipedia_wikibooks": (
            f"https://olmo-data.org/dolma-v1_7/wiki/wiki-{index:04d}.json.gz"
        ),
        "stackedu": f"data/stack_edu-fetch/{index:05d}.jsonl.zst",
        "finemath_3plus": f"finemath-3plus/train-{index:05d}.parquet",
        "arxiv": f"data/rpj-proofpile-arxiv/{index:05d}.jsonl.zst",
        "olmocr": f"data/olmocr_science_pdfs-fetch/{index:05d}.jsonl.zst",
        "fineweb_edu": f"data/CC-MAIN-fetch/train-{index:05d}.parquet",
    }[family]


def _fixture_enumeration():
    routes = load_exact_source_routes_v3()
    declared = {
        row.source_family: row for row in load_source_route_manifest().routes
    }
    trees: dict[tuple[str, str], list[dict[str, object]]] = {}
    payloads_by_locator: dict[str, bytes] = {}
    for route in routes:
        count = 2 if route.source_family == "wikipedia_wikibooks" else 3
        for index in range(count):
            locator = _locator(route.source_family, index)
            payload = (
                f"fetch-fixture:{route.source_family}:{index}:" + "x" * (index + 1)
            ).encode("ascii")
            payloads_by_locator[locator] = payload
            if route.source_family != "wikipedia_wikibooks":
                trees.setdefault((route.repository, route.revision), []).append(
                    {
                        "blob_id": hashlib.sha1(locator.encode("utf-8")).hexdigest(),
                        "lfs": {"sha256": _sha(payload), "size": len(payload)},
                        "path": locator,
                        "size": len(payload),
                        "type": "file",
                    }
                )

    def tree(**kwargs: object) -> Iterable[object]:
        return tuple(trees[(str(kwargs["repo_id"]), str(kwargs["revision"]))])

    def external(**kwargs: object) -> ExternalLocatorListingV3:
        family = "wikipedia_wikibooks"
        assets = tuple(
            ExternalLocatorAssetV3(
                locator=_locator(family, index),
                upstream_bytes=len(payloads_by_locator[_locator(family, index)]),
                content_sha256=_sha(payloads_by_locator[_locator(family, index)]),
            )
            for index in range(2)
        )
        row = declared[family]
        return ExternalLocatorListingV3.fixture(
            source_family=family,
            external_locator_manifest_sha256=str(
                kwargs["expected_manifest_sha256"]
            ),
            available_bytes=sum(asset.upstream_bytes for asset in assets),
            available_bytes_basis=row.available_bytes_basis,
            assets=assets,
        )

    enumeration = enumerate_upstream_assets_v3(
        list_repo_tree=tree,
        enumerate_external_locators=external,
        mode=FIXTURE_MODE,
    )
    payloads = {
        asset.asset_identity_sha256: payloads_by_locator[asset.asset_locator]
        for family in enumeration.families
        for asset in family.assets
    }
    targets = {
        family.source_family: (
            1
            if family.source_family == "wikipedia_wikibooks"
            else sum(asset.upstream_bytes for asset in family.assets[:2])
        )
        for family in enumeration.families
    }
    return enumeration, payloads, targets


def test_prefix_selection_is_minimal_and_keeps_both_science_reserves() -> None:
    enumeration, unused, targets = _fixture_enumeration()
    del unused
    plan, receipt = select_required_asset_prefixes_v3(
        enumeration,
        required_bytes_by_family=targets,
    )

    rows = {row.source_family: row for row in receipt.families}
    assert rows["wikipedia_wikibooks"].selected_asset_count == 2
    assert rows["wikipedia_wikibooks"].selection_rule == (
        "complete_pinned_wikipedia_asset_set"
    )
    assert rows["arxiv"].selected_asset_count == 2
    assert rows["olmocr"].selected_asset_count == 2
    assert all(
        row.selected_upstream_bytes >= row.required_bytes
        for row in receipt.families
    )
    assert len(plan.assets) == 14
    again_plan, again_receipt = select_required_asset_prefixes_v3(
        enumeration,
        required_bytes_by_family=targets,
    )
    assert again_plan == plan
    assert again_receipt == receipt


def test_fetch_is_resumable_and_persists_replayable_receipts(tmp_path: Path) -> None:
    enumeration, payloads, targets = _fixture_enumeration()
    opened: list[str] = []

    def opener(asset):
        opened.append(asset.asset_identity_sha256)
        return io.BytesIO(payloads[asset.asset_identity_sha256])

    cache_root = tmp_path / "cache"
    receipt_root = tmp_path / "receipts"
    plan, unused_selection = select_required_asset_prefixes_v3(
        enumeration,
        required_bytes_by_family=targets,
    )
    del unused_selection
    first_upstream = plan.assets[0]
    suffix = next(
        value
        for value in (".jsonl.zst", ".json.gz", ".parquet")
        if first_upstream.asset_locator.endswith(value)
    )
    interrupted = (
        cache_root
        / "assets"
        / first_upstream.source_family
        / f"{first_upstream.asset_identity_sha256}{suffix}.partial"
    )
    interrupted.parent.mkdir(parents=True)
    interrupted.write_bytes(b"interrupted")
    first = prepare_selected_source_cache_v3(
        enumeration=enumeration,
        cache_root=cache_root,
        receipt_root=receipt_root,
        open_upstream=opener,
        required_bytes_by_family=targets,
        allow_nonauthoritative_fixture=True,
    )
    assert not interrupted.exists()
    assert len(opened) == len(first.plan.assets)
    governed_paths = (
        receipt_root / ENUMERATION_ARTIFACT_NAME,
        receipt_root / SELECTION_ARTIFACT_NAME,
        receipt_root / SOURCE_MANIFEST_NAME,
        receipt_root / DOWNLOAD_ARTIFACT_NAME,
        *tuple(
            cache_root.joinpath(*PurePosixPath(asset.relative_path).parts)
            for asset in first.download.source_manifest.assets
        ),
    )
    before = {path: (path.stat().st_mtime_ns, _sha(path.read_bytes())) for path in governed_paths}

    second = prepare_selected_source_cache_v3(
        enumeration=enumeration,
        cache_root=cache_root,
        receipt_root=receipt_root,
        open_upstream=opener,
        required_bytes_by_family=targets,
        allow_nonauthoritative_fixture=True,
    )
    assert second == first
    assert len(opened) == len(first.plan.assets)
    assert before == {
        path: (path.stat().st_mtime_ns, _sha(path.read_bytes())) for path in governed_paths
    }
    assert all(path.read_bytes().endswith(b"\n") for path in governed_paths[:4])
    assert all(not path.read_bytes().endswith(b"\n\n") for path in governed_paths[:4])

    replay_enumeration = load_upstream_enumeration_receipt_v3(
        receipt_root / ENUMERATION_ARTIFACT_NAME
    )
    replay_download, verified = load_source_cache_download_receipt_v3(
        receipt_root / DOWNLOAD_ARTIFACT_NAME,
        enumeration=replay_enumeration,
        source_manifest_path=receipt_root / SOURCE_MANIFEST_NAME,
        cache_root=cache_root,
        route_manifest_path=SOURCE_ROUTE_MANIFEST_PATH,
    )
    assert replay_download == first.download
    assert verified.source_manifest == first.download.source_manifest


def test_completed_cache_tamper_fails_without_redownload(tmp_path: Path) -> None:
    enumeration, payloads, targets = _fixture_enumeration()
    calls = 0

    def opener(asset):
        nonlocal calls
        calls += 1
        return io.BytesIO(payloads[asset.asset_identity_sha256])

    result = prepare_selected_source_cache_v3(
        enumeration=enumeration,
        cache_root=tmp_path / "cache",
        receipt_root=tmp_path / "receipts",
        open_upstream=opener,
        required_bytes_by_family=targets,
        allow_nonauthoritative_fixture=True,
    )
    first_asset = result.download.source_manifest.assets[0]
    path = (tmp_path / "cache").joinpath(
        *PurePosixPath(first_asset.relative_path).parts
    )
    tampered = b"!" + path.read_bytes()[1:]
    path.write_bytes(tampered)
    before_calls = calls
    with pytest.raises(SourceTransportError, match="differs from upstream"):
        prepare_selected_source_cache_v3(
            enumeration=enumeration,
            cache_root=tmp_path / "cache",
            receipt_root=tmp_path / "receipts",
            open_upstream=opener,
            required_bytes_by_family=targets,
            allow_nonauthoritative_fixture=True,
        )
    assert calls == before_calls
    assert path.read_bytes() == tampered


def test_external_transport_cache_uses_completed_bytes_without_overwrite(
    tmp_path: Path,
) -> None:
    resources = {
        "https://huggingface.co/datasets/allenai/dolma/resolve/rev/urls/v1_7.txt": b"parent\n",
        "https://olmo-data.org/dolma-v1_7/wiki/wiki-0000.json.gz": b"wiki-0",
        "https://olmo-data.org/dolma-v1_7/wiki/wiki-0001.json.gz": b"wiki-1",
    }
    calls: list[str] = []

    def open_url(locator: str):
        calls.append(locator)
        return io.BytesIO(resources[locator])

    cache = ExternalResourceCacheV3(tmp_path / "external", open_url=open_url)
    for locator, expected in resources.items():
        with cache.open(locator) as handle:
            assert handle.read() == expected
    assert calls == list(resources)
    paths = {
        row.locator: (tmp_path / "external").joinpath(
            *PurePosixPath(row.relative_path).parts
        )
        for row in cache.observations
    }
    before = {locator: path.stat().st_mtime_ns for locator, path in paths.items()}
    for locator, expected in resources.items():
        with cache.open(locator) as handle:
            assert handle.read() == expected
    assert calls == list(resources)
    assert before == {locator: path.stat().st_mtime_ns for locator, path in paths.items()}
    assert all(row.cache_hit for row in cache.observations)

    first_locator = next(iter(resources))
    tampered = b"!" + paths[first_locator].read_bytes()[1:]
    paths[first_locator].write_bytes(tampered)
    with pytest.raises(SourceFetchError, match="immutable receipt"):
        cache.open(first_locator)
    assert paths[first_locator].read_bytes() == tampered
    assert calls == list(resources)


def test_fresh_external_cache_bytes_are_rechecked_against_the_new_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locator = "https://olmo-data.org/dolma-v1_7/wiki/wiki-0000.json.gz"
    payload = b"fresh transport bytes"
    real_hash_file = fetch_module._hash_file

    def race_after_receipt(path: Path) -> tuple[int, str]:
        observed_bytes, unused = real_hash_file(path)
        del unused
        return observed_bytes, "0" * 64

    monkeypatch.setattr(fetch_module, "_hash_file", race_after_receipt)
    cache = ExternalResourceCacheV3(
        tmp_path / "fresh-race-cache",
        open_url=lambda unused: io.BytesIO(payload),
    )
    with pytest.raises(SourceFetchError, match="immutable receipt"):
        cache.open(locator)
    assert cache.observations == ()


def test_default_external_transport_rejects_redirect_before_caching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openers: list[_RedirectingOpener] = []

    def build(handler):
        opener = _RedirectingOpener(
            handler,
            targets=("https://unapproved.example.invalid/redirected",),
            payload=b"untrusted",
        )
        openers.append(opener)
        return opener

    monkeypatch.setattr(fetch_module, "build_opener", build)
    cache = ExternalResourceCacheV3(tmp_path / "redirect-cache")
    locator = "https://olmo-data.org/dolma-v1_7/wiki/wiki-0000.json.gz"
    with pytest.raises(SourceFetchError, match="redirect is forbidden"):
        cache.open(locator)
    assert len(openers) == 1
    assert openers[0].hop_responses[0].closed
    assert not tuple((tmp_path / "redirect-cache").rglob("*.bin"))


def test_exact_hf_manifest_redirect_is_accepted_once_and_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"fixture parent\n"
    monkeypatch.setattr(fetch_module, "_HF_DOLMA_MANIFEST_BYTES", len(payload))
    monkeypatch.setattr(fetch_module, "_HF_DOLMA_MANIFEST_SHA256", _sha(payload))
    openers: list[_RedirectingOpener] = []

    def build(handler):
        opener = _RedirectingOpener(
            handler,
            targets=(_accepted_hf_cache_target(),),
            payload=payload,
        )
        openers.append(opener)
        return opener

    monkeypatch.setattr(fetch_module, "build_opener", build)
    cache = ExternalResourceCacheV3(tmp_path / "accepted-cache")
    locator = fetch_module._HF_DOLMA_MANIFEST_ORIGIN
    with cache.open(locator) as handle:
        assert handle.read() == payload
    with cache.open(locator) as handle:
        assert handle.read() == payload
    assert len(openers) == 1
    assert openers[0].hop_responses[0].closed
    assert cache.observations[0].cache_hit
    assert (
        cache.observations[0].transport_policy_id
        == fetch_module._EXTERNAL_TRANSPORT_POLICY_ID
    )
    observation = cache.observations[0]
    assert observation.redirect_count == 1
    assert observation.redirect_kind == fetch_module._HF_CACHE_REDIRECT_KIND
    assert (
        observation.redirect_target_path
        == fetch_module._HF_DOLMA_MANIFEST_CACHE_PATH
    )
    assert observation.redirect_etag == fetch_module._HF_DOLMA_MANIFEST_ETAG
    cache_path = (tmp_path / "accepted-cache").joinpath(
        *PurePosixPath(observation.relative_path).parts
    )
    receipt_path = cache_path.with_name(cache_path.name + ".receipt.json")
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    assert hashlib.sha256(receipt_raw).hexdigest() == (
        observation.cache_entry_receipt_sha256
    )
    assert receipt["schema"] == fetch_module._EXTERNAL_CACHE_ENTRY_SCHEMA_V4
    assert receipt["transport_policy_id"] == (
        fetch_module._EXTERNAL_TRANSPORT_POLICY_ID
    )
    assert receipt["redirect_count"] == 1
    assert receipt["redirect_etag"] == fetch_module._HF_DOLMA_MANIFEST_ETAG
    assert (
        fetch_module._HF_DOLMA_MANIFEST_CACHE_QUERY.encode("ascii")
        not in receipt_raw
    )


def test_external_transport_artifact_is_stable_across_cache_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = {
        fetch_module._HF_DOLMA_MANIFEST_ORIGIN: b"parent fixture\n",
        fetch_module._PINNED_EXTERNAL_LOCATORS[1]: b"wiki fixture zero",
        fetch_module._PINNED_EXTERNAL_LOCATORS[2]: b"wiki fixture one",
    }
    parent = payloads[fetch_module._HF_DOLMA_MANIFEST_ORIGIN]
    monkeypatch.setattr(fetch_module, "_HF_DOLMA_MANIFEST_BYTES", len(parent))
    monkeypatch.setattr(fetch_module, "_HF_DOLMA_MANIFEST_SHA256", _sha(parent))
    cache = ExternalResourceCacheV3(
        tmp_path / "stable-cache",
        open_url=lambda locator: io.BytesIO(payloads[locator]),
    )
    for locator in fetch_module._PINNED_EXTERNAL_LOCATORS:
        with cache.open(locator) as handle:
            assert handle.read() == payloads[locator]
    code_identity = _attested_code_identity()
    first = build_external_transport_receipt_v1(
        cache.observations,
        require_authoritative_set=True,
        source_prep_code_identity=code_identity,
    )
    assert first.mode == fetch_module.AUTHORITATIVE_MODE
    assert not any(row.cache_hit for row in cache.observations)

    artifact_path = tmp_path / "receipts" / EXTERNAL_TRANSPORT_ARTIFACT_NAME
    with pytest.raises(SourceFetchError, match="through its finalizer"):
        write_external_transport_receipt_v1(artifact_path, first)
    mismatch_path = tmp_path / "mismatch" / EXTERNAL_TRANSPORT_ARTIFACT_NAME
    with pytest.raises(SourceFetchError, match="changed during execution"):
        fetch_module.finalize_authoritative_external_transport_receipt_v1(
            mismatch_path,
            first,
            post_execution_code_identity=_attested_code_identity("b"),
        )
    assert not mismatch_path.exists()
    first_artifact_sha256 = (
        fetch_module.finalize_authoritative_external_transport_receipt_v1(
            artifact_path,
            first,
            post_execution_code_identity=code_identity,
        )
    )
    first_raw = artifact_path.read_bytes()
    assert load_external_transport_receipt_v1(
        artifact_path,
        require_authoritative=True,
    ) == first

    for locator in fetch_module._PINNED_EXTERNAL_LOCATORS:
        with cache.open(locator) as handle:
            assert handle.read() == payloads[locator]
    second = build_external_transport_receipt_v1(
        cache.observations,
        require_authoritative_set=True,
        source_prep_code_identity=code_identity,
    )
    assert all(row.cache_hit for row in cache.observations)
    assert second == first
    assert second.receipt_sha256 == first.receipt_sha256
    assert (
        fetch_module.finalize_authoritative_external_transport_receipt_v1(
            artifact_path,
            second,
            post_execution_code_identity=code_identity,
        )
        == first_artifact_sha256
    )
    assert artifact_path.read_bytes() == first_raw
    assert b"cache_hit" not in first_raw
    assert (
        fetch_module._HF_DOLMA_MANIFEST_CACHE_QUERY.encode("ascii") not in first_raw
    )


def test_external_transport_artifact_fails_closed_on_missing_or_tampered_evidence(
    tmp_path: Path,
) -> None:
    payloads = {
        locator: f"payload:{index}".encode("ascii")
        for index, locator in enumerate(fetch_module._PINNED_EXTERNAL_LOCATORS)
    }
    cache = ExternalResourceCacheV3(
        tmp_path / "transport-cache",
        open_url=lambda locator: io.BytesIO(payloads[locator]),
    )
    for locator in fetch_module._PINNED_EXTERNAL_LOCATORS[1:]:
        with cache.open(locator):
            pass
    with pytest.raises(SourceFetchError, match="exact locator set"):
        build_external_transport_receipt_v1(
            cache.observations,
            require_authoritative_set=True,
            source_prep_code_identity=_attested_code_identity(),
        )
    receipt = build_external_transport_receipt_v1(
        cache.observations,
        require_authoritative_set=False,
        source_prep_code_identity=_fixture_code_identity(),
    )
    assert receipt.mode == fetch_module.EXTERNAL_TRANSPORT_FIXTURE_MODE
    with pytest.raises(ValueError, match="exact locator set"):
        fetch_module.ExternalTransportReceiptV1(
            schema=fetch_module.EXTERNAL_TRANSPORT_RECEIPT_SCHEMA_V1,
            mode=fetch_module.AUTHORITATIVE_MODE,
            transport_policy_id=fetch_module._EXTERNAL_TRANSPORT_POLICY_ID,
            source_prep_code_identity=_attested_code_identity(),
            entries=receipt.entries,
        )
    with pytest.raises(ValueError, match="ATTESTED_HEAD"):
        fetch_module.ExternalTransportReceiptV1(
            schema=fetch_module.EXTERNAL_TRANSPORT_RECEIPT_SCHEMA_V1,
            mode=fetch_module.AUTHORITATIVE_MODE,
            transport_policy_id=fetch_module._EXTERNAL_TRANSPORT_POLICY_ID,
            source_prep_code_identity=_fixture_code_identity(),
            entries=receipt.entries,
        )
    artifact_path = tmp_path / "transport-receipt.json"
    write_external_transport_receipt_v1(artifact_path, receipt)
    assert load_external_transport_receipt_v1(
        artifact_path,
        require_authoritative=False,
    ) == receipt
    with pytest.raises(SourceFetchError, match="not AUTHORITATIVE"):
        load_external_transport_receipt_v1(
            artifact_path,
            require_authoritative=True,
        )
    tampered = artifact_path.read_bytes().replace(
        b"DIRECT_PINNED_HTTPS",
        b"TAMPERED_TRANSPORT",
        1,
    )
    artifact_path.write_bytes(tampered)
    with pytest.raises(SourceFetchError, match="refusing overwrite"):
        write_external_transport_receipt_v1(artifact_path, receipt)
    assert artifact_path.read_bytes() == tampered


@pytest.mark.parametrize(
    "target",
    (
        _accepted_hf_cache_target().replace("https://", "http://", 1),
        _accepted_hf_cache_target().replace("huggingface.co", "evil.invalid", 1),
        _accepted_hf_cache_target().replace(
            "huggingface.co", "user@huggingface.co", 1
        ),
        _accepted_hf_cache_target().replace("huggingface.co", "huggingface.co:444", 1),
        _accepted_hf_cache_target().replace("huggingface.co", "huggingface.co:443", 1),
        _accepted_hf_cache_target().replace("/allenai/dolma/", "/other/dolma/", 1),
        _accepted_hf_cache_target().replace(
            "7f48140530a023e9ea4c5cfb141160922727d4d3",
            "0" * 40,
            1,
        ),
        _accepted_hf_cache_target().replace("urls%2Fv1_7.txt", "urls%252Fv1_7.txt"),
        _accepted_hf_cache_target().replace("urls%2Fv1_7.txt", "urls%2Fv1_7.txt/x"),
        _accepted_hf_cache_target() + "#fragment",
        _accepted_hf_cache_target().replace("etag=", "token="),
        _accepted_hf_cache_target().replace(
            "723219ebec21e7e8e2a6616b6ec45145df69aae0",
            "a" * 40,
        ),
        _accepted_hf_cache_target().replace("%2Fdatasets", "/datasets"),
        _accepted_hf_cache_target().replace("%2Fdatasets", "%2fdatasets"),
        _accepted_hf_cache_target().replace("=&etag=", "&etag="),
        _accepted_hf_cache_target().replace("=&etag=", "=&etag=%22x%22&etag="),
        _accepted_hf_cache_target().replace(
            "?%2Fdatasets",
            "?etag=%22723219ebec21e7e8e2a6616b6ec45145df69aae0%22&%2Fdatasets",
        ),
    ),
)
def test_hf_manifest_redirect_rejects_every_authority_drift(target: str) -> None:
    assert not fetch_module._is_pinned_hf_manifest_redirect(
        fetch_module._HF_DOLMA_MANIFEST_ORIGIN,
        target,
    )


def test_hf_manifest_redirect_rejects_a_second_hop_before_following(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openers: list[_RedirectingOpener] = []

    def build(handler):
        opener = _RedirectingOpener(
            handler,
            targets=(
                _accepted_hf_cache_target(),
                _accepted_hf_cache_target(),
            ),
            payload=b"unreachable",
        )
        openers.append(opener)
        return opener

    monkeypatch.setattr(fetch_module, "build_opener", build)
    with pytest.raises(SourceFetchError, match="redirect is forbidden"):
        fetch_module._default_open_url(fetch_module._HF_DOLMA_MANIFEST_ORIGIN)
    assert len(openers) == 1
    assert all(response.closed for response in openers[0].hop_responses)


def test_hf_manifest_content_drift_is_never_finalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fetch_module,
        "build_opener",
        lambda handler: _RedirectingOpener(
            handler,
            targets=(_accepted_hf_cache_target(),),
            payload=b"wrong parent bytes",
        ),
    )
    cache_root = tmp_path / "content-drift-cache"
    cache = ExternalResourceCacheV3(cache_root)
    with pytest.raises(SourceFetchError, match="frozen content identity"):
        cache.open(fetch_module._HF_DOLMA_MANIFEST_ORIGIN)
    assert not tuple(cache_root.rglob("*.bin"))
    assert not tuple(cache_root.rglob("*.receipt.json"))


def test_existing_receipt_artifact_is_never_overwritten(tmp_path: Path) -> None:
    enumeration, payloads, targets = _fixture_enumeration()
    receipt_root = tmp_path / "receipts"
    prepare_selected_source_cache_v3(
        enumeration=enumeration,
        cache_root=tmp_path / "cache",
        receipt_root=receipt_root,
        open_upstream=lambda asset: io.BytesIO(payloads[asset.asset_identity_sha256]),
        required_bytes_by_family=targets,
        allow_nonauthoritative_fixture=True,
    )
    path = receipt_root / SELECTION_ARTIFACT_NAME
    tampered = path.read_bytes().replace(b"minimal_seeded", b"tampered_seeded", 1)
    path.write_bytes(tampered)
    with pytest.raises(SourceFetchError, match="refusing overwrite"):
        prepare_selected_source_cache_v3(
            enumeration=enumeration,
            cache_root=tmp_path / "cache",
            receipt_root=receipt_root,
            open_upstream=lambda asset: io.BytesIO(
                payloads[asset.asset_identity_sha256]
            ),
            required_bytes_by_family=targets,
            allow_nonauthoritative_fixture=True,
        )
    assert path.read_bytes() == tampered
