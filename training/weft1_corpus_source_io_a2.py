"""Fail-closed source transport and record parsing for WEFT-1 corpus P-A.

The upstream enumerator proves *which* pinned assets exist.  This module owns
the next two boundaries:

* stream each selected asset into a content-addressed local cache, verify the
  strongest upstream byte identity available, and only then mint
  :class:`SourceCacheAssetV3`; and
* decode the three registered container formats through source-specific schema
  bindings.  No field guessing, fallback aliases, coercion, or malformed-row
  skipping is permitted.

Invalid UTF-8 in one otherwise framed JSONL row is an explicit whole-document
drop.  A malformed row, a truncated container, or a schema mismatch fails the
entire asset.  Parquet UTF-8 validity is enforced by Arrow's exact ``string``
type and by an additional scalar-valid UTF-8 encode before a record is emitted.
"""

from __future__ import annotations

import base64
from contextlib import closing
from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import BinaryIO, Callable, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from training.weft1_corpus_enumeration_a2 import (
    AUTHORITATIVE_MODE,
    FIXTURE_MODE,
    UpstreamAssetV3,
    UpstreamEnumerationReceiptV3,
)
from training.weft1_corpus_pa import RawDocumentV3
from training.weft1_corpus_sources_a2 import (
    QUALITY_GATED_SOURCE_FAMILIES,
    SOURCE_CACHE_SCHEMA_V3,
    CanonicalSourceRecordV3,
    SourceCacheAssetV3,
    SourceCacheManifestV3,
    VerifiedLocalCacheAssetV3,
    VerifiedLocalCacheManifestV3,
    canonical_asset_order_v3,
    verify_local_source_cache_v3,
)
from training.weft1_gtok_contract import canonical_json_bytes, canonical_sha256
from training.weft1_strict_io import assert_no_symlink_ancestors
from training.weft1_strict_io import load_canonical_json_snapshot


SOURCE_CACHE_DOWNLOAD_SCHEMA_V3 = "weft1_source_cache_download_v3"
SOURCE_CACHE_DOWNLOAD_ARTIFACT_SCHEMA_V3 = (
    "weft1_source_cache_download_receipt_artifact_v3"
)
SOURCE_PARSE_EVENT_SCHEMA_V3 = "weft1_source_parse_event_v3"
SOURCE_RECORD_OBSERVATION_SCHEMA_V3 = "weft1_source_record_observation_v3"
PARSED_SOURCE_SPOOL_SCHEMA_V3 = "weft1_parsed_source_spool_v3"
RETAIN = "RETAIN"
DROP_INVALID_UTF8 = "DROP_INVALID_UTF8"
DROP_EMPTY_TEXT = "DROP_EMPTY_TEXT"
DROP_QUALITY_LT3 = "DROP_QUALITY_LT3"
PARSE_DISPOSITIONS = (
    RETAIN,
    DROP_INVALID_UTF8,
    DROP_EMPTY_TEXT,
    DROP_QUALITY_LT3,
)

_SOURCE_STRATA = {
    "dolma_web": "general",
    "wikipedia_wikibooks": "general",
    "stackedu": "code",
    "finemath_3plus": "mathematics",
    "arxiv": "science_technical",
    "olmocr": "science_technical",
    "fineweb_edu": "general",
}
_SOURCE_CONTAINERS = {
    "dolma_web": "jsonl.zst",
    "wikipedia_wikibooks": "json.gz",
    "stackedu": "jsonl.zst",
    "finemath_3plus": "parquet",
    "arxiv": "jsonl.zst",
    "olmocr": "jsonl.zst",
    "fineweb_edu": "parquet",
}
_CONTAINER_SUFFIXES = {
    "jsonl.zst": ".jsonl.zst",
    "json.gz": ".json.gz",
    "parquet": ".parquet",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DOWNLOAD_PLAN_FACTORY_SENTINEL = object()
_DOWNLOAD_RECEIPT_FACTORY_SENTINEL = object()
_PARSED_SPOOL_FACTORY_SENTINEL = object()


class SourceIOError(RuntimeError):
    """Base class for production source-I/O failures."""


class SourceTransportError(SourceIOError):
    """The downloaded/cache bytes do not match their pinned identity."""


class SourceSchemaError(SourceIOError):
    """A selected asset does not implement its exact parser schema."""


class SourceSchemaBindingRequired(SourceSchemaError):
    """A production parser has not yet been bound from a pinned asset."""


class SourceContainerError(SourceIOError):
    """A selected compressed/container asset is malformed or truncated."""


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _safe_cache_path(root: Path, relative_path: str, *, strict: bool) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        "\\" in relative_path
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise SourceTransportError("cache path is not canonical relative POSIX")
    lexical_candidate = root.joinpath(*relative.parts)
    assert_no_symlink_ancestors(lexical_candidate)
    candidate = lexical_candidate.resolve(strict=strict)
    resolved_root = root.resolve(strict=True)
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise SourceTransportError("cache path resolves outside cache root")
    return candidate


def _git_blob_digest(algorithm: str, byte_count: int) -> object:
    digest = hashlib.new(algorithm)
    digest.update(f"blob {byte_count}\0".encode("ascii"))
    return digest


def _asset_relative_path(asset: UpstreamAssetV3) -> str:
    container = _SOURCE_CONTAINERS[asset.source_family]
    suffix = _CONTAINER_SUFFIXES[container]
    if not asset.asset_locator.endswith(suffix):
        raise SourceTransportError(
            f"{asset.source_family} asset has the wrong registered container suffix"
        )
    return f"assets/{asset.source_family}/{asset.asset_identity_sha256}{suffix}"


def _enumeration_assets(
    enumeration: UpstreamEnumerationReceiptV3,
) -> tuple[UpstreamAssetV3, ...]:
    return tuple(asset for family in enumeration.families for asset in family.assets)


def _validate_planned_assets(
    enumeration: UpstreamEnumerationReceiptV3,
    assets: tuple[UpstreamAssetV3, ...],
) -> None:
    if not isinstance(assets, tuple) or not assets:
        raise ValueError("download plan requires a nonempty asset tuple")
    if any(not isinstance(asset, UpstreamAssetV3) for asset in assets):
        raise TypeError("download plan contains an untyped upstream asset")
    complete = _enumeration_assets(enumeration)
    identities = tuple(asset.asset_identity_sha256 for asset in complete)
    if len(identities) != len(set(identities)):
        raise ValueError("full enumeration repeats an upstream asset identity")
    index = {identity: ordinal for ordinal, identity in enumerate(identities)}
    positions: list[int] = []
    for asset in assets:
        identity = asset.asset_identity_sha256
        if identity not in index or complete[index[identity]] != asset:
            raise ValueError("download plan contains an asset outside the enumeration")
        positions.append(index[identity])
    if positions != sorted(set(positions)):
        raise ValueError("download plan assets are duplicated or noncanonical")


@dataclass(frozen=True, init=False)
class SourceAssetDownloadPlanV3:
    """A canonical subset of one complete upstream enumeration."""

    enumeration_receipt_sha256: str
    enumeration_mode: str
    assets: tuple[UpstreamAssetV3, ...]

    def __new__(cls) -> "SourceAssetDownloadPlanV3":
        raise TypeError("download plans are factory-minted from an enumeration")

    @classmethod
    def _validated(
        cls,
        *,
        enumeration: UpstreamEnumerationReceiptV3,
        assets: tuple[UpstreamAssetV3, ...],
        sentinel: object,
    ) -> "SourceAssetDownloadPlanV3":
        if sentinel is not _DOWNLOAD_PLAN_FACTORY_SENTINEL:
            raise PermissionError("download plans are factory-only")
        _validate_planned_assets(enumeration, assets)
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "enumeration_receipt_sha256",
            enumeration.receipt_sha256,
        )
        object.__setattr__(instance, "enumeration_mode", enumeration.mode)
        object.__setattr__(instance, "assets", assets)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        _require_sha256(self.enumeration_receipt_sha256, "enumeration receipt")
        if self.enumeration_mode not in {AUTHORITATIVE_MODE, FIXTURE_MODE}:
            raise ValueError("download plan enumeration mode is invalid")
        if not isinstance(self.assets, tuple) or not self.assets:
            raise ValueError("download plan requires assets")

    @property
    def plan_sha256(self) -> str:
        return canonical_sha256(
            {"payload": self, "schema": "weft1_source_asset_download_plan_v3"}
        )


def plan_source_cache_assets_v3(
    enumeration: UpstreamEnumerationReceiptV3,
    assets: Sequence[UpstreamAssetV3],
) -> SourceAssetDownloadPlanV3:
    """Bind a selected canonical subsequence to its full enumeration receipt."""

    if not isinstance(enumeration, UpstreamEnumerationReceiptV3):
        raise TypeError("download planning requires a typed enumeration")
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
        raise TypeError("planned assets must be a typed sequence")
    return SourceAssetDownloadPlanV3._validated(
        enumeration=enumeration,
        assets=tuple(assets),
        sentinel=_DOWNLOAD_PLAN_FACTORY_SENTINEL,
    )


@dataclass(frozen=True)
class DownloadedAssetEvidenceV3:
    """Actual byte observation made before a cache expectation is minted."""

    upstream_asset_identity_sha256: str
    source_cache_asset_identity_sha256: str
    relative_path: str
    observed_bytes: int
    observed_sha256: str
    upstream_identity_check: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.upstream_asset_identity_sha256,
            "upstream_asset_identity_sha256",
        )
        _require_sha256(
            self.source_cache_asset_identity_sha256,
            "source_cache_asset_identity_sha256",
        )
        _require_sha256(self.observed_sha256, "observed_sha256")
        if type(self.observed_bytes) is not int or self.observed_bytes < 1:
            raise ValueError("download evidence bytes must be a positive integer")
        if self.upstream_identity_check not in {
            "content_sha256",
            "git_blob_sha1",
            "git_blob_sha256",
        }:
            raise ValueError("download evidence uses an unknown identity check")


@dataclass(frozen=True)
class SourceCachePlanMaterializationV3:
    """Downloaded bytes for one plan; no final source manifest is minted yet."""

    plan: SourceAssetDownloadPlanV3
    cache_assets: tuple[SourceCacheAssetV3, ...]
    evidence: tuple[DownloadedAssetEvidenceV3, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, SourceAssetDownloadPlanV3):
            raise TypeError("plan materialization requires a typed plan")
        if not isinstance(self.cache_assets, tuple) or not isinstance(
            self.evidence, tuple
        ):
            raise TypeError("plan materialization values must be tuples")
        if len(self.cache_assets) != len(self.plan.assets) or len(
            self.evidence
        ) != len(self.plan.assets):
            raise ValueError("plan materialization does not exactly cover its plan")
        for upstream, cached, item in zip(
            self.plan.assets,
            self.cache_assets,
            self.evidence,
            strict=True,
        ):
            if (
                cached.source_family,
                cached.repository,
                cached.config,
                cached.revision,
                cached.split,
                cached.asset_locator,
            ) != (
                upstream.source_family,
                upstream.repository,
                upstream.config,
                upstream.revision,
                upstream.split,
                upstream.asset_locator,
            ):
                raise ValueError("materialized cache asset route differs from its plan")
            if (
                item.upstream_asset_identity_sha256
                != upstream.asset_identity_sha256
                or item.source_cache_asset_identity_sha256
                != cached.asset_identity_sha256
                or item.relative_path != cached.relative_path
                or item.observed_bytes != cached.bytes
                or item.observed_sha256 != cached.sha256
            ):
                raise ValueError("materialized evidence differs from its plan/cache asset")

    @property
    def materialization_sha256(self) -> str:
        return canonical_sha256(
            {
                "payload": self,
                "schema": "weft1_source_cache_plan_materialization_v3",
            }
        )


@dataclass(frozen=True, init=False)
class SourceCacheDownloadReceiptV3:
    """Cache materialization plus an independent full local-byte re-read."""

    enumeration_receipt_sha256: str
    enumeration_mode: str
    selection_plan_sha256: str
    source_manifest: SourceCacheManifestV3
    evidence: tuple[DownloadedAssetEvidenceV3, ...]
    verification_receipt_sha256: str

    def __new__(cls) -> "SourceCacheDownloadReceiptV3":
        raise TypeError("download receipts are factory-minted after byte re-read")

    @classmethod
    def _validated(
        cls,
        *,
        enumeration_receipt_sha256: str,
        enumeration_mode: str,
        selection_plan_sha256: str,
        source_manifest: SourceCacheManifestV3,
        evidence: tuple[DownloadedAssetEvidenceV3, ...],
        verification_receipt_sha256: str,
        sentinel: object,
    ) -> "SourceCacheDownloadReceiptV3":
        if sentinel is not _DOWNLOAD_RECEIPT_FACTORY_SENTINEL:
            raise PermissionError("download receipts are factory-only")
        instance = object.__new__(cls)
        for name, value in (
            ("enumeration_receipt_sha256", enumeration_receipt_sha256),
            ("enumeration_mode", enumeration_mode),
            ("selection_plan_sha256", selection_plan_sha256),
            ("source_manifest", source_manifest),
            ("evidence", evidence),
            ("verification_receipt_sha256", verification_receipt_sha256),
        ):
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        _require_sha256(self.enumeration_receipt_sha256, "enumeration receipt")
        _require_sha256(self.selection_plan_sha256, "selection plan")
        if self.enumeration_mode not in {AUTHORITATIVE_MODE, FIXTURE_MODE}:
            raise ValueError("download receipt enumeration mode is invalid")
        if not isinstance(self.source_manifest, SourceCacheManifestV3):
            raise TypeError("download receipt requires a typed cache manifest")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("download receipt requires nonempty byte evidence")
        if len(self.evidence) != len(self.source_manifest.assets):
            raise ValueError("download evidence does not cover the source manifest")
        for item, asset in zip(
            self.evidence,
            self.source_manifest.assets,
            strict=True,
        ):
            if (
                item.source_cache_asset_identity_sha256
                != asset.asset_identity_sha256
                or item.relative_path != asset.relative_path
                or item.observed_bytes != asset.bytes
                or item.observed_sha256 != asset.sha256
            ):
                raise ValueError("download evidence and source manifest are misaligned")
        _require_sha256(self.verification_receipt_sha256, "verification receipt")

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(
            {"payload": self, "schema": SOURCE_CACHE_DOWNLOAD_SCHEMA_V3}
        )


def _manifest_bytes(manifest: SourceCacheManifestV3) -> bytes:
    return canonical_json_bytes(
        {
            "assets": [asdict(asset) for asset in manifest.assets],
            "schema": manifest.schema,
            "source_route_manifest_sha256": manifest.source_route_manifest_sha256,
        }
    )


def _verified_stream_identity(
    asset: UpstreamAssetV3,
    source: BinaryIO,
    destination: Path | None,
) -> tuple[int, str, str]:
    raw_sha256 = hashlib.sha256()
    git_sha1 = _git_blob_digest("sha1", asset.upstream_bytes)
    git_sha256 = _git_blob_digest("sha256", asset.upstream_bytes)
    byte_count = 0
    output: BinaryIO | None = None
    try:
        if destination is not None:
            output = destination.open("xb")
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if chunk is None or chunk == b"":
                break
            if not isinstance(chunk, bytes):
                raise SourceTransportError("downloader returned a non-bytes chunk")
            byte_count += len(chunk)
            if byte_count > asset.upstream_bytes:
                raise SourceTransportError("download exceeded pinned upstream bytes")
            raw_sha256.update(chunk)
            git_sha1.update(chunk)
            git_sha256.update(chunk)
            if output is not None:
                output.write(chunk)
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        if output is not None:
            output.close()
        if destination is not None and destination.exists():
            destination.unlink()
        raise
    finally:
        if output is not None and not output.closed:
            output.close()
    if byte_count != asset.upstream_bytes:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise SourceTransportError("download byte count differs from upstream metadata")
    raw_digest = raw_sha256.hexdigest()
    if asset.content_sha256 is not None:
        if raw_digest != asset.content_sha256:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise SourceTransportError("download content SHA-256 differs from upstream")
        identity_check = "content_sha256"
    elif asset.blob_identity_kind == "git_sha1":
        if git_sha1.hexdigest() != asset.blob_identity:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise SourceTransportError("download Git blob SHA-1 differs from upstream")
        identity_check = "git_blob_sha1"
    elif asset.blob_identity_kind == "git_sha256":
        if git_sha256.hexdigest() != asset.blob_identity:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise SourceTransportError("download Git blob SHA-256 differs from upstream")
        identity_check = "git_blob_sha256"
    elif asset.blob_identity_kind == "content_sha256":
        if raw_digest != asset.blob_identity:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise SourceTransportError("download content SHA-256 differs from upstream")
        identity_check = "content_sha256"
    else:  # UpstreamAssetV3 prevents this; keep the transport boundary fail-closed.
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise SourceTransportError("upstream asset has no verifiable byte identity")
    return byte_count, raw_digest, identity_check


def _stream_download(
    asset: UpstreamAssetV3,
    source: BinaryIO,
    destination: Path,
) -> tuple[int, str, str]:
    return _verified_stream_identity(asset, source, destination)


def _materialize_planned_asset(
    upstream: UpstreamAssetV3,
    root: Path,
    open_upstream: Callable[[UpstreamAssetV3], BinaryIO],
    *,
    resume_incomplete: bool,
) -> tuple[SourceCacheAssetV3, DownloadedAssetEvidenceV3]:
    relative_path = _asset_relative_path(upstream)
    final_path = _safe_cache_path(root, relative_path, strict=False)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = final_path.with_name(final_path.name + ".partial")
    if partial_path.exists():
        assert_no_symlink_ancestors(partial_path)
        if not resume_incomplete:
            raise SourceTransportError("stale partial cache asset exists")
        if not partial_path.is_file():
            raise SourceTransportError("stale partial cache asset is not a file")
        partial_path.unlink()
    if final_path.exists():
        with final_path.open("rb") as existing:
            observed_bytes, observed_sha256, identity_check = _verified_stream_identity(
                upstream, existing, None
            )
    else:
        opened = open_upstream(upstream)
        if not hasattr(opened, "read"):
            raise SourceTransportError("open_upstream returned no binary reader")
        with closing(opened):
            observed_bytes, observed_sha256, identity_check = _stream_download(
                upstream, opened, partial_path
            )
        os.replace(partial_path, final_path)
    cached = SourceCacheAssetV3(
        source_family=upstream.source_family,
        repository=upstream.repository,
        config=upstream.config,
        revision=upstream.revision,
        split=upstream.split,
        asset_locator=upstream.asset_locator,
        relative_path=relative_path,
        bytes=observed_bytes,
        sha256=observed_sha256,
    )
    return (
        cached,
        DownloadedAssetEvidenceV3(
            upstream_asset_identity_sha256=upstream.asset_identity_sha256,
            source_cache_asset_identity_sha256=cached.asset_identity_sha256,
            relative_path=relative_path,
            observed_bytes=observed_bytes,
            observed_sha256=observed_sha256,
            upstream_identity_check=identity_check,
        ),
    )


def materialize_source_cache_v3(
    enumeration: UpstreamEnumerationReceiptV3,
    plan: SourceAssetDownloadPlanV3,
    cache_root: Path,
    *,
    open_upstream: Callable[[UpstreamAssetV3], BinaryIO],
    allow_nonauthoritative_fixture: bool = False,
    resume_incomplete: bool = False,
) -> SourceCachePlanMaterializationV3:
    """Materialize only one selected plan; no final manifest is emitted."""

    if not isinstance(enumeration, UpstreamEnumerationReceiptV3):
        raise TypeError("cache materialization requires a typed enumeration")
    if not isinstance(plan, SourceAssetDownloadPlanV3):
        raise TypeError("cache materialization requires a typed download plan")
    if (
        plan.enumeration_receipt_sha256 != enumeration.receipt_sha256
        or plan.enumeration_mode != enumeration.mode
    ):
        raise SourceTransportError("download plan belongs to another enumeration")
    _validate_planned_assets(enumeration, plan.assets)
    if not enumeration.authoritative and not allow_nonauthoritative_fixture:
        raise SourceTransportError("production cache requires authoritative enumeration")
    if not isinstance(cache_root, Path):
        raise TypeError("cache root must be a pathlib.Path")
    if not callable(open_upstream):
        raise TypeError("open_upstream must be callable")
    if type(resume_incomplete) is not bool:
        raise TypeError("resume_incomplete must be an exact bool")
    assert_no_symlink_ancestors(cache_root)
    root = cache_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise SourceTransportError("cache root is not a directory")
    cached: list[SourceCacheAssetV3] = []
    evidence: list[DownloadedAssetEvidenceV3] = []
    for upstream in plan.assets:
        item, observation = _materialize_planned_asset(
            upstream,
            root,
            open_upstream,
            resume_incomplete=resume_incomplete,
        )
        cached.append(item)
        evidence.append(observation)
    return SourceCachePlanMaterializationV3(
        plan=plan,
        cache_assets=tuple(cached),
        evidence=tuple(evidence),
    )


def finalize_source_cache_v3(
    enumeration: UpstreamEnumerationReceiptV3,
    materializations: Sequence[SourceCachePlanMaterializationV3],
    cache_root: Path,
    manifest_path: Path,
    *,
    allow_nonauthoritative_fixture: bool = False,
    allow_existing_verified_manifest: bool = False,
) -> SourceCacheDownloadReceiptV3:
    """Finalize a canonical subset manifest after target/top-up scanning ends."""

    if not isinstance(enumeration, UpstreamEnumerationReceiptV3):
        raise TypeError("cache finalization requires a typed enumeration")
    if not isinstance(materializations, Sequence) or isinstance(
        materializations, (str, bytes)
    ):
        raise TypeError("cache materializations must be a typed sequence")
    if not materializations or any(
        not isinstance(item, SourceCachePlanMaterializationV3)
        for item in materializations
    ):
        raise ValueError("cache finalization requires typed materializations")
    if not enumeration.authoritative and not allow_nonauthoritative_fixture:
        raise SourceTransportError("production cache requires authoritative enumeration")
    if type(allow_existing_verified_manifest) is not bool:
        raise TypeError("allow_existing_verified_manifest must be an exact bool")
    selected_upstream = tuple(
        asset for item in materializations for asset in item.plan.assets
    )
    combined_plan = plan_source_cache_assets_v3(enumeration, selected_upstream)
    if any(
        item.plan.enumeration_receipt_sha256 != enumeration.receipt_sha256
        or item.plan.enumeration_mode != enumeration.mode
        for item in materializations
    ):
        raise SourceTransportError("materialization belongs to another enumeration")
    expected_assets = tuple(
        asset for item in materializations for asset in item.cache_assets
    )
    observations = tuple(
        evidence for item in materializations for evidence in item.evidence
    )
    if len(expected_assets) != len({asset.asset_identity_sha256 for asset in expected_assets}):
        raise SourceTransportError("cache finalization repeats a selected asset")

    if not isinstance(cache_root, Path) or not isinstance(manifest_path, Path):
        raise TypeError("cache and manifest paths must be pathlib.Path values")
    assert_no_symlink_ancestors(cache_root)
    assert_no_symlink_ancestors(manifest_path)
    root = cache_root.resolve(strict=True)
    if not root.is_dir():
        raise SourceTransportError("cache root is not a directory")
    resolved_manifest = manifest_path.resolve()
    if resolved_manifest == root or root in resolved_manifest.parents:
        raise SourceTransportError("cache manifest must live outside the asset root")
    partial_manifest = resolved_manifest.with_suffix(resolved_manifest.suffix + ".partial")
    assert_no_symlink_ancestors(partial_manifest)
    if partial_manifest.exists():
        raise SourceTransportError("stale cache manifest partial exists")
    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)

    ordered_assets = canonical_asset_order_v3(expected_assets)
    evidence_by_identity = {
        item.source_cache_asset_identity_sha256: item for item in observations
    }
    if len(evidence_by_identity) != len(observations):
        raise SourceTransportError("cache finalization repeats download evidence")
    ordered_evidence = tuple(
        evidence_by_identity[asset.asset_identity_sha256] for asset in ordered_assets
    )
    manifest = SourceCacheManifestV3(
        schema=SOURCE_CACHE_SCHEMA_V3,
        source_route_manifest_sha256=enumeration.source_route_manifest_sha256,
        assets=ordered_assets,
    )
    if resolved_manifest.exists():
        if not allow_existing_verified_manifest:
            raise SourceTransportError("refusing to overwrite a cache manifest")
        verified = verify_local_source_cache_v3(resolved_manifest, root)
        if verified.source_manifest != manifest:
            raise SourceTransportError(
                "existing cache manifest differs from the selected verified assets"
            )
    else:
        canonical_manifest = _manifest_bytes(manifest) + b"\n"
        if canonical_manifest.endswith(b"\n\n"):
            raise AssertionError("cache manifest serialization has more than one final LF")
        try:
            with partial_manifest.open("xb") as handle:
                handle.write(canonical_manifest)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial_manifest, resolved_manifest)
            verified = verify_local_source_cache_v3(resolved_manifest, root)
        except BaseException:
            partial_manifest.unlink(missing_ok=True)
            raise
    return SourceCacheDownloadReceiptV3._validated(
        enumeration_receipt_sha256=enumeration.receipt_sha256,
        enumeration_mode=enumeration.mode,
        selection_plan_sha256=combined_plan.plan_sha256,
        source_manifest=manifest,
        evidence=ordered_evidence,
        verification_receipt_sha256=verified.verification_receipt_sha256,
        sentinel=_DOWNLOAD_RECEIPT_FACTORY_SENTINEL,
    )


def _write_canonical_source_artifact(
    path: Path,
    payload: Mapping[str, object],
) -> str:
    if not isinstance(path, Path):
        raise TypeError("source receipt artifact path must be a pathlib.Path")
    assert_no_symlink_ancestors(path)
    if path.exists():
        raise SourceTransportError("refusing to overwrite a source receipt artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise SourceTransportError("stale source receipt artifact partial exists")
    raw = canonical_json_bytes(payload) + b"\n"
    try:
        with partial.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return hashlib.sha256(raw).hexdigest()


def write_source_cache_download_receipt_v3(
    receipt: SourceCacheDownloadReceiptV3,
    path: Path,
) -> str:
    """Persist a canonical complete download-receipt envelope."""

    if not isinstance(receipt, SourceCacheDownloadReceiptV3):
        raise TypeError("download artifact requires a factory receipt")
    return _write_canonical_source_artifact(
        path,
        {
            "receipt": asdict(receipt),
            "receipt_sha256": receipt.receipt_sha256,
            "schema": SOURCE_CACHE_DOWNLOAD_ARTIFACT_SCHEMA_V3,
        },
    )


def load_source_cache_download_receipt_v3(
    path: Path,
    *,
    enumeration: UpstreamEnumerationReceiptV3,
    source_manifest_path: Path,
    cache_root: Path,
    route_manifest_path: Path = Path(__file__).with_name(
        "weft1_gtok_source_routes_20260828.json"
    ),
) -> tuple[SourceCacheDownloadReceiptV3, VerifiedLocalCacheManifestV3]:
    """Reconstruct a download receipt and independently rehash every asset."""

    if not isinstance(enumeration, UpstreamEnumerationReceiptV3):
        raise TypeError("download replay requires a typed enumeration")
    artifact_bytes, payload = load_canonical_json_snapshot(path)
    if artifact_bytes != canonical_json_bytes(payload) + b"\n":
        raise SourceTransportError(
            "download receipt artifact is not canonical sorted JSON"
        )
    if set(payload) != {"receipt", "receipt_sha256", "schema"} or payload.get(
        "schema"
    ) != SOURCE_CACHE_DOWNLOAD_ARTIFACT_SCHEMA_V3:
        raise SourceTransportError("download receipt artifact envelope drifted")
    raw_receipt = payload["receipt"]
    receipt_keys = {
        "enumeration_mode",
        "enumeration_receipt_sha256",
        "evidence",
        "selection_plan_sha256",
        "source_manifest",
        "verification_receipt_sha256",
    }
    if not isinstance(raw_receipt, Mapping) or set(raw_receipt) != receipt_keys:
        raise SourceTransportError("download receipt artifact shape drifted")
    raw_evidence = raw_receipt["evidence"]
    if not isinstance(raw_evidence, list):
        raise TypeError("download artifact evidence must be a list")
    evidence_keys = {
        "observed_bytes",
        "observed_sha256",
        "relative_path",
        "source_cache_asset_identity_sha256",
        "upstream_asset_identity_sha256",
        "upstream_identity_check",
    }
    evidence: list[DownloadedAssetEvidenceV3] = []
    for raw_item in raw_evidence:
        if not isinstance(raw_item, Mapping) or set(raw_item) != evidence_keys:
            raise SourceTransportError("download artifact evidence shape drifted")
        evidence.append(DownloadedAssetEvidenceV3(**dict(raw_item)))
    raw_manifest = raw_receipt["source_manifest"]
    if not isinstance(raw_manifest, Mapping):
        raise TypeError("download artifact source manifest must be an object")
    manifest = SourceCacheManifestV3.from_mapping(raw_manifest)
    verified = verify_local_source_cache_v3(
        source_manifest_path,
        cache_root,
        route_manifest_path,
    )
    if manifest != verified.source_manifest:
        raise SourceTransportError(
            "download artifact manifest differs from independently verified bytes"
        )
    receipt = SourceCacheDownloadReceiptV3._validated(
        enumeration_receipt_sha256=raw_receipt[
            "enumeration_receipt_sha256"
        ],  # type: ignore[arg-type]
        enumeration_mode=raw_receipt["enumeration_mode"],  # type: ignore[arg-type]
        selection_plan_sha256=raw_receipt[
            "selection_plan_sha256"
        ],  # type: ignore[arg-type]
        source_manifest=manifest,
        evidence=tuple(evidence),
        verification_receipt_sha256=raw_receipt[
            "verification_receipt_sha256"
        ],  # type: ignore[arg-type]
        sentinel=_DOWNLOAD_RECEIPT_FACTORY_SENTINEL,
    )
    if (
        receipt.enumeration_receipt_sha256 != enumeration.receipt_sha256
        or receipt.enumeration_mode != enumeration.mode
        or receipt.verification_receipt_sha256
        != verified.verification_receipt_sha256
    ):
        raise SourceTransportError(
            "download artifact does not compose with enumeration/cache verification"
        )
    selected = _selected_upstream_for_cache(enumeration, verified)
    if plan_source_cache_assets_v3(enumeration, selected).plan_sha256 != (
        receipt.selection_plan_sha256
    ):
        raise SourceTransportError("download artifact selection plan drifted")
    if payload["receipt_sha256"] != receipt.receipt_sha256:
        raise SourceTransportError("download artifact receipt SHA-256 drifted")
    return receipt, verified


def materialize_complete_fixture_source_cache_v3(
    enumeration: UpstreamEnumerationReceiptV3,
    cache_root: Path,
    manifest_path: Path,
    *,
    open_upstream: Callable[[UpstreamAssetV3], BinaryIO],
    allow_nonauthoritative_fixture: bool = False,
) -> SourceCacheDownloadReceiptV3:
    """Fixture-only convenience wrapper over plan, materialize, and finalize.

    Production must use an explicit subset plan so P-A stops at its selected
    byte targets instead of downloading the complete upstream inventory.
    """

    if not isinstance(enumeration, UpstreamEnumerationReceiptV3):
        raise TypeError("cache materialization requires a typed enumeration")
    if enumeration.authoritative:
        raise SourceTransportError(
            "complete-enumeration helper is fixture-only; production requires a subset plan"
        )
    plan = plan_source_cache_assets_v3(enumeration, _enumeration_assets(enumeration))
    materialized = materialize_source_cache_v3(
        enumeration,
        plan,
        cache_root,
        open_upstream=open_upstream,
        allow_nonauthoritative_fixture=allow_nonauthoritative_fixture,
    )
    return finalize_source_cache_v3(
        enumeration,
        (materialized,),
        cache_root,
        manifest_path,
        allow_nonauthoritative_fixture=allow_nonauthoritative_fixture,
    )


@dataclass(frozen=True)
class JsonFieldV3:
    path: tuple[str, ...]
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple) or not self.path or any(
            not isinstance(item, str) or not item for item in self.path
        ):
            raise ValueError("JSON field paths must be nonempty string tuples")
        if self.kind not in {"string", "object", "integer"}:
            raise ValueError("JSON field binding uses an unsupported exact kind")


@dataclass(frozen=True)
class SourceParserBindingV3:
    source_family: str
    container: str
    authority: str
    authority_sha256: str
    text_path: tuple[str, ...]
    native_id_path: tuple[str, ...] | None
    int_score_path: tuple[str, ...] | None
    exact_json_top_level_fields: tuple[str, ...] = ()
    required_json_fields: tuple[JsonFieldV3, ...] = ()
    declared_parquet_columns: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.source_family not in _SOURCE_CONTAINERS:
            raise ValueError("parser binding uses an unknown source family")
        if self.container != _SOURCE_CONTAINERS[self.source_family]:
            raise ValueError("parser binding container differs from its source route")
        if self.authority not in {
            "PINNED_CARD_DECLARATION",
            "PINNED_ASSET_DECLARATION",
            "FIXTURE_ONLY",
        }:
            raise ValueError("parser binding authority is unknown")
        _require_sha256(self.authority_sha256, "parser authority_sha256")
        if not self.text_path:
            raise ValueError("parser binding requires an exact text path")
        if self.source_family in QUALITY_GATED_SOURCE_FAMILIES:
            if self.int_score_path is None:
                raise ValueError("quality-gated parser requires int_score path")
        elif self.int_score_path is not None:
            raise ValueError("ungated parser may not bind an int_score path")
        if self.container == "parquet":
            if not self.declared_parquet_columns or self.required_json_fields:
                raise ValueError(
                    "Parquet binding requires only its card-declared column projection"
                )
            column_names = tuple(name for name, _ in self.declared_parquet_columns)
            if len(column_names) != len(set(column_names)):
                raise ValueError("Parquet binding repeats a column")
            bound_roots = (
                self.text_path,
                self.native_id_path,
                self.int_score_path,
            )
            if any(
                path is not None
                and (len(path) != 1 or path[0] not in column_names)
                for path in bound_roots
            ):
                raise ValueError("Parquet parser path is absent from its exact schema")
        elif self.declared_parquet_columns or not self.required_json_fields:
            raise ValueError("JSON binding requires exact JSON fields only")
        else:
            if (
                self.exact_json_top_level_fields
                != tuple(sorted(set(self.exact_json_top_level_fields)))
            ):
                raise ValueError("JSON top-level fields must be unique and sorted")
            required_paths = {field.path for field in self.required_json_fields}
            for path in (self.text_path, self.native_id_path, self.int_score_path):
                if path is not None and path not in required_paths:
                    raise ValueError("JSON parser path is absent from required fields")
            if any(
                field.path[0] not in self.exact_json_top_level_fields
                for field in self.required_json_fields
            ):
                raise ValueError("JSON required field has an unbound top-level key")

    @property
    def binding_sha256(self) -> str:
        return canonical_sha256(
            {"payload": self, "schema": "weft1_source_parser_binding_v3"}
        )


_STACKEDU_FIELDS = (
    JsonFieldV3(("added",), "string"),
    JsonFieldV3(("created",), "string"),
    JsonFieldV3(("id",), "string"),
    JsonFieldV3(("metadata",), "object"),
    JsonFieldV3(("metadata", "int_score"), "integer"),
    JsonFieldV3(("source",), "string"),
    JsonFieldV3(("text",), "string"),
)
_DOLMA_WEB_FIELDS = (
    JsonFieldV3(("id",), "string"),
    JsonFieldV3(("metadata",), "object"),
    JsonFieldV3(("text",), "string"),
)
_WIKIPEDIA_FIELDS = (
    JsonFieldV3(("added",), "string"),
    JsonFieldV3(("created",), "string"),
    JsonFieldV3(("id",), "string"),
    JsonFieldV3(("metadata",), "object"),
    JsonFieldV3(("metadata", "length"), "integer"),
    JsonFieldV3(("metadata", "provenance"), "string"),
    JsonFieldV3(("metadata", "revid"), "string"),
    JsonFieldV3(("metadata", "url"), "string"),
    JsonFieldV3(("source",), "string"),
    JsonFieldV3(("text",), "string"),
    JsonFieldV3(("version",), "string"),
)
_ARXIV_FIELDS = (
    JsonFieldV3(("added",), "string"),
    JsonFieldV3(("created",), "string"),
    JsonFieldV3(("doc",), "object"),
    JsonFieldV3(("id",), "string"),
    JsonFieldV3(("metadata",), "object"),
    JsonFieldV3(("text",), "string"),
)
_OLMOCR_FIELDS = (
    JsonFieldV3(("added",), "string"),
    JsonFieldV3(("created",), "string"),
    JsonFieldV3(("id",), "string"),
    JsonFieldV3(("metadata",), "object"),
    JsonFieldV3(("source",), "string"),
    JsonFieldV3(("text",), "string"),
)

_FINEMATH_SCHEMA = (
    ("url", "string"),
    ("fetch_time", "int64"),
    ("content_mime_type", "string"),
    ("warc_filename", "string"),
    ("warc_record_offset", "int32"),
    ("warc_record_length", "int32"),
    ("text", "string"),
    ("token_count", "int32"),
    ("char_count", "int32"),
    ("metadata", "string"),
    ("score", "float64"),
    ("int_score", "int64"),
    ("crawl", "string"),
    ("snapshot_type", "string"),
    ("language", "string"),
    ("language_score", "float64"),
)
_FINEWEB_SCHEMA = (
    ("text", "string"),
    ("id", "string"),
    ("dump", "string"),
    ("url", "string"),
    ("date", "string"),
    ("file_path", "string"),
    ("language", "string"),
    ("language_score", "float64"),
    ("token_count", "int64"),
    ("score", "float64"),
    ("int_score", "int64"),
)

# These are declared parser contracts, not observation receipts.  Actual
# pinned-row evidence is minted only after re-reading cache bytes and records
# asset identity, ordinal, both row hashes, and the full observed schema.
def _declared_schema_binding_sha256(
    source_family: str,
    revision: str,
    asset_locator: str,
    fields: Sequence[JsonFieldV3],
) -> str:
    return canonical_sha256(
        {
            "asset_locator": asset_locator,
            "fields": tuple(fields),
            "revision": revision,
            "schema": "weft1_declared_pinned_asset_schema_binding_v3",
            "source_family": source_family,
        }
    )


PRODUCTION_PARSER_BINDINGS_V3: Mapping[str, SourceParserBindingV3] = {
    "dolma_web": SourceParserBindingV3(
        source_family="dolma_web",
        container="jsonl.zst",
        authority="PINNED_ASSET_DECLARATION",
        authority_sha256=_declared_schema_binding_sha256(
            "dolma_web",
            "6462556697df1a8f5c953727e9c686629ad98b68",
            "data/common_crawl-home_and_hobbies-0019/shard_00000041.jsonl.zst",
            _DOLMA_WEB_FIELDS,
        ),
        text_path=("text",),
        native_id_path=("id",),
        int_score_path=None,
        exact_json_top_level_fields=("id", "metadata", "text"),
        required_json_fields=_DOLMA_WEB_FIELDS,
    ),
    "wikipedia_wikibooks": SourceParserBindingV3(
        source_family="wikipedia_wikibooks",
        container="json.gz",
        authority="PINNED_ASSET_DECLARATION",
        authority_sha256=_declared_schema_binding_sha256(
            "wikipedia_wikibooks",
            "7f48140530a023e9ea4c5cfb141160922727d4d3",
            "https://olmo-data.org/dolma-v1_7/wiki/wiki-0000.json.gz",
            _WIKIPEDIA_FIELDS,
        ),
        text_path=("text",),
        native_id_path=("id",),
        int_score_path=None,
        exact_json_top_level_fields=(
            "added",
            "created",
            "id",
            "metadata",
            "source",
            "text",
            "version",
        ),
        required_json_fields=_WIKIPEDIA_FIELDS,
    ),
    "stackedu": SourceParserBindingV3(
        source_family="stackedu",
        container="jsonl.zst",
        authority="PINNED_ASSET_DECLARATION",
        authority_sha256=_declared_schema_binding_sha256(
            "stackedu",
            "689a3ea2d8217e64d73a5058913fa43ad15e81aa",
            "data/stack_edu-Java/shard_00000243.jsonl.zst",
            _STACKEDU_FIELDS,
        ),
        text_path=("text",),
        native_id_path=("id",),
        int_score_path=("metadata", "int_score"),
        exact_json_top_level_fields=("added", "created", "id", "metadata", "source", "text"),
        required_json_fields=_STACKEDU_FIELDS,
    ),
    "finemath_3plus": SourceParserBindingV3(
        source_family="finemath_3plus",
        container="parquet",
        authority="PINNED_CARD_DECLARATION",
        authority_sha256="77b9162c99fc6b5944da8793f18fb99c3b520ab7cdce0cc67ec8a1e47871da61",
        text_path=("text",),
        native_id_path=None,
        int_score_path=("int_score",),
        declared_parquet_columns=_FINEMATH_SCHEMA,
    ),
    "arxiv": SourceParserBindingV3(
        source_family="arxiv",
        container="jsonl.zst",
        authority="PINNED_ASSET_DECLARATION",
        authority_sha256=_declared_schema_binding_sha256(
            "arxiv",
            "689a3ea2d8217e64d73a5058913fa43ad15e81aa",
            "data/rpj-proofpile-arxiv/arxiv-train-0007.jsonl.zst",
            _ARXIV_FIELDS,
        ),
        text_path=("text",),
        native_id_path=("id",),
        int_score_path=None,
        exact_json_top_level_fields=("added", "created", "doc", "id", "metadata", "text"),
        required_json_fields=_ARXIV_FIELDS,
    ),
    "olmocr": SourceParserBindingV3(
        source_family="olmocr",
        container="jsonl.zst",
        authority="PINNED_ASSET_DECLARATION",
        authority_sha256=_declared_schema_binding_sha256(
            "olmocr",
            "689a3ea2d8217e64d73a5058913fa43ad15e81aa",
            "data/olmocr_science_pdfs-electronics_and_hardware/shard_00004701.jsonl.zst",
            _OLMOCR_FIELDS,
        ),
        text_path=("text",),
        native_id_path=("id",),
        int_score_path=None,
        exact_json_top_level_fields=("added", "created", "id", "metadata", "source", "text"),
        required_json_fields=_OLMOCR_FIELDS,
    ),
    "fineweb_edu": SourceParserBindingV3(
        source_family="fineweb_edu",
        container="parquet",
        authority="PINNED_CARD_DECLARATION",
        authority_sha256="a0cc8998a20499432b28b6575f3046b714938eb8e11b8d59a1d25ddf3716061e",
        text_path=("text",),
        native_id_path=("id",),
        int_score_path=("int_score",),
        declared_parquet_columns=_FINEWEB_SCHEMA,
    ),
}


def fixture_source_parser_binding_v3(source_family: str) -> SourceParserBindingV3:
    """Return a conspicuously non-production minimal binding for offline tests."""

    if source_family not in _SOURCE_CONTAINERS:
        raise ValueError("unknown fixture source family")
    container = _SOURCE_CONTAINERS[source_family]
    quality = source_family in QUALITY_GATED_SOURCE_FAMILIES
    native_id = None if source_family == "finemath_3plus" else ("id",)
    fields: list[JsonFieldV3] = [JsonFieldV3(("text",), "string")]
    names = ["text"]
    if native_id is not None:
        fields.append(JsonFieldV3(native_id, "string"))
        names.append("id")
    if quality:
        fields.append(JsonFieldV3(("int_score",), "integer"))
        names.append("int_score")
    if container == "parquet":
        schema = tuple(
            (name, "int64" if name == "int_score" else "string")
            for name in names
        )
        fields_tuple: tuple[JsonFieldV3, ...] = ()
        top_fields: tuple[str, ...] = ()
    else:
        schema = ()
        fields_tuple = tuple(fields)
        top_fields = tuple(sorted(names))
    return SourceParserBindingV3(
        source_family=source_family,
        container=container,
        authority="FIXTURE_ONLY",
        authority_sha256=hashlib.sha256(
            f"fixture:{source_family}".encode("ascii")
        ).hexdigest(),
        text_path=("text",),
        native_id_path=native_id,
        int_score_path=("int_score",) if quality else None,
        exact_json_top_level_fields=top_fields,
        required_json_fields=fields_tuple,
        declared_parquet_columns=schema,
    )


@dataclass(frozen=True)
class ParsedSourceRecordV3:
    canonical_record: CanonicalSourceRecordV3
    raw_document: RawDocumentV3
    parser_binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_record, CanonicalSourceRecordV3):
            raise TypeError("parsed record requires canonical source metadata")
        if not isinstance(self.raw_document, RawDocumentV3):
            raise TypeError("parsed record requires a RawDocumentV3")
        if (
            self.raw_document.source != self.canonical_record.source_family
            or self.raw_document.stable_source_record_id
            != self.canonical_record.canonical_source_record_id
        ):
            raise ValueError("parsed raw and canonical source identities disagree")
        _require_sha256(self.parser_binding_sha256, "parser_binding_sha256")


@dataclass(frozen=True)
class SourceRecordObservationV3:
    """Parent-rehashable evidence from one actual cached source record."""

    source_family: str
    source_cache_asset_identity_sha256: str
    source_asset_sha256: str
    record_ordinal: int
    row_representation: str
    raw_row_bytes: int
    raw_row_sha256: str
    canonical_row_bytes: int
    canonical_row_sha256: str
    observed_schema_canonical_json: str
    observed_schema_sha256: str
    observed_arrow_schema_ipc_hex: str | None = None

    def __post_init__(self) -> None:
        if self.source_family not in _SOURCE_CONTAINERS:
            raise ValueError("record observation uses an unknown source family")
        for value, name in (
            (
                self.source_cache_asset_identity_sha256,
                "source_cache_asset_identity_sha256",
            ),
            (self.source_asset_sha256, "source_asset_sha256"),
            (self.raw_row_sha256, "raw_row_sha256"),
            (self.canonical_row_sha256, "canonical_row_sha256"),
            (self.observed_schema_sha256, "observed_schema_sha256"),
        ):
            _require_sha256(value, name)
        if type(self.record_ordinal) is not int or self.record_ordinal < 0:
            raise ValueError("record observation ordinal must be non-negative")
        if self.row_representation not in {
            "decompressed_json_object_payload",
            "arrow_ipc_single_row",
        }:
            raise ValueError("record observation row representation is unknown")
        for value, name in (
            (self.raw_row_bytes, "raw_row_bytes"),
            (self.canonical_row_bytes, "canonical_row_bytes"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        try:
            schema_value = json.loads(self.observed_schema_canonical_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("observed schema is not canonical JSON") from error
        schema_bytes = canonical_json_bytes(schema_value)
        if schema_bytes.decode("utf-8") != self.observed_schema_canonical_json:
            raise ValueError("observed schema JSON is not canonical")
        if hashlib.sha256(schema_bytes).hexdigest() != self.observed_schema_sha256:
            raise ValueError("observed schema SHA-256 disagrees with its bytes")
        if self.row_representation == "arrow_ipc_single_row":
            if not self.observed_arrow_schema_ipc_hex:
                raise ValueError("Arrow observation requires full schema IPC bytes")
            try:
                bytes.fromhex(self.observed_arrow_schema_ipc_hex)
            except ValueError as error:
                raise ValueError("Arrow schema IPC is not canonical hex") from error
        elif self.observed_arrow_schema_ipc_hex is not None:
            raise ValueError("JSON observation may not carry Arrow schema IPC")

    @property
    def observation_sha256(self) -> str:
        return canonical_sha256(
            {"payload": self, "schema": SOURCE_RECORD_OBSERVATION_SCHEMA_V3}
        )


@dataclass(frozen=True)
class SourceParseEventV3:
    source_family: str
    source_record_ordinal: int
    disposition: str
    record: ParsedSourceRecordV3 | None
    reason: str | None = None
    observation: SourceRecordObservationV3 | None = None

    def __post_init__(self) -> None:
        if self.source_family not in _SOURCE_CONTAINERS:
            raise ValueError("parse event uses an unknown source family")
        if type(self.source_record_ordinal) is not int or self.source_record_ordinal < 0:
            raise ValueError("parse event ordinal must be non-negative")
        if self.disposition not in PARSE_DISPOSITIONS:
            raise ValueError("parse event disposition is unknown")
        if self.disposition == RETAIN:
            if self.record is None or self.reason is not None:
                raise ValueError("retained event requires only a parsed record")
        elif self.record is not None or not self.reason:
            raise ValueError("drop event requires only an explicit reason")
        if self.observation is not None:
            if not isinstance(self.observation, SourceRecordObservationV3):
                raise TypeError("parse event observation is untyped")
            if (
                self.observation.source_family != self.source_family
                or self.observation.record_ordinal != self.source_record_ordinal
            ):
                raise ValueError("parse event and record observation disagree")

    @property
    def event_sha256(self) -> str:
        return canonical_sha256(
            {"payload": self, "schema": SOURCE_PARSE_EVENT_SCHEMA_V3}
        )


def _json_no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise SourceSchemaError("JSON source row repeats a key")
        value[key] = item
    return value


def _lookup_path(row: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = row
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise SourceSchemaError("source row is missing bound field " + ".".join(path))
        current = current[part]
    return current


def _require_json_kind(value: object, field: JsonFieldV3) -> None:
    if field.kind == "string":
        valid = isinstance(value, str)
    elif field.kind == "object":
        valid = isinstance(value, Mapping)
    else:
        valid = type(value) is int
    if not valid:
        raise SourceSchemaError(
            f"source field {'.'.join(field.path)} is not exact {field.kind}"
        )


def _arrow_type_name(value: pa.DataType) -> str:
    if pa.types.is_string(value):
        return "string"
    if pa.types.is_int32(value):
        return "int32"
    if pa.types.is_int64(value):
        return "int64"
    if pa.types.is_float64(value):
        return "float64"
    return str(value)


def _json_schema_descriptor(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise SourceSchemaError("JSON source object uses a non-string key")
        return {
            "fields": tuple(
                {
                    "name": key,
                    "schema": _json_schema_descriptor(value[key]),
                }
                for key in sorted(value)
            ),
            "kind": "object",
        }
    if isinstance(value, list):
        item_descriptors = {
            canonical_json_bytes(_json_schema_descriptor(item)).decode("utf-8")
            for item in value
        }
        return {
            "item_schemas": tuple(sorted(item_descriptors)),
            "kind": "array",
            "observed_length": len(value),
        }
    if value is None:
        kind = "null"
    elif type(value) is bool:
        kind = "boolean"
    elif type(value) is int:
        kind = "integer"
    elif type(value) is float:
        kind = "number"
    elif isinstance(value, str):
        kind = "string"
    else:
        raise SourceSchemaError(
            f"JSON source uses unsupported value type {type(value).__name__}"
        )
    return {"kind": kind}


def _encoded_metadata(
    metadata: Mapping[bytes, bytes] | None,
) -> tuple[tuple[str, str], ...]:
    if metadata is None:
        return ()
    return tuple(
        sorted(
            (
                base64.b64encode(key).decode("ascii"),
                base64.b64encode(value).decode("ascii"),
            )
            for key, value in metadata.items()
        )
    )


def _arrow_schema_descriptor(schema: pa.Schema) -> Mapping[str, object]:
    return {
        "fields": tuple(
            {
                "metadata_base64": _encoded_metadata(field.metadata),
                "name": field.name,
                "nullable": field.nullable,
                "type": str(field.type),
            }
            for field in schema
        ),
        "format": "arrow_schema_ipc",
        "metadata_base64": _encoded_metadata(schema.metadata),
    }


def _record_observation_v3(
    *,
    verified_asset: VerifiedLocalCacheAssetV3,
    ordinal: int,
    row_representation: str,
    raw_row: bytes,
    canonical_row: bytes,
    schema_descriptor: Mapping[str, object],
    arrow_schema_ipc: bytes | None = None,
) -> SourceRecordObservationV3:
    schema_bytes = canonical_json_bytes(schema_descriptor)
    return SourceRecordObservationV3(
        source_family=verified_asset.expected.source_family,
        source_cache_asset_identity_sha256=(
            verified_asset.expected.asset_identity_sha256
        ),
        source_asset_sha256=verified_asset.observed_sha256,
        record_ordinal=ordinal,
        row_representation=row_representation,
        raw_row_bytes=len(raw_row),
        raw_row_sha256=hashlib.sha256(raw_row).hexdigest(),
        canonical_row_bytes=len(canonical_row),
        canonical_row_sha256=hashlib.sha256(canonical_row).hexdigest(),
        observed_schema_canonical_json=schema_bytes.decode("utf-8"),
        observed_schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
        observed_arrow_schema_ipc_hex=(
            None if arrow_schema_ipc is None else arrow_schema_ipc.hex()
        ),
    )


def _event_from_values(
    *,
    verified_asset: VerifiedLocalCacheAssetV3,
    binding: SourceParserBindingV3,
    ordinal: int,
    text: object,
    native_id: object | None,
    int_score: object | None,
    observation: SourceRecordObservationV3 | None = None,
) -> SourceParseEventV3:
    source = verified_asset.expected.source_family
    if not isinstance(text, (str, bytes)):
        raise SourceSchemaError("bound text field is not exact str or bytes")
    if isinstance(text, bytes):
        raw_text = text
        try:
            text_bytes = text.decode("utf-8", errors="strict").encode("utf-8")
        except UnicodeError:
            return SourceParseEventV3(
                source_family=source,
                source_record_ordinal=ordinal,
                disposition=DROP_INVALID_UTF8,
                record=None,
                reason="text bytes are not scalar-valid UTF-8",
                observation=observation,
            )
    else:
        raw_text = text
        try:
            text_bytes = text.encode("utf-8", errors="strict")
        except UnicodeError:
            return SourceParseEventV3(
                source_family=source,
                source_record_ordinal=ordinal,
                disposition=DROP_INVALID_UTF8,
                record=None,
                reason="text string is not scalar-valid UTF-8",
                observation=observation,
            )
    if not text_bytes:
        return SourceParseEventV3(
            source_family=source,
            source_record_ordinal=ordinal,
            disposition=DROP_EMPTY_TEXT,
            record=None,
            reason="text contains zero UTF-8 bytes",
            observation=observation,
        )
    if binding.native_id_path is not None:
        if not isinstance(native_id, str) or not native_id:
            raise SourceSchemaError("bound native ID is not a nonempty exact string")
    elif native_id is not None:
        raise SourceSchemaError("source without native ID unexpectedly supplied one")
    if source in QUALITY_GATED_SOURCE_FAMILIES:
        if type(int_score) is not int:
            raise SourceSchemaError("bound int_score is not an exact integer")
        if int_score < 3:
            return SourceParseEventV3(
                source_family=source,
                source_record_ordinal=ordinal,
                disposition=DROP_QUALITY_LT3,
                record=None,
                reason="int_score is below the bound threshold 3",
                observation=observation,
            )
    elif int_score is not None:
        raise SourceSchemaError("ungated source unexpectedly supplied int_score")
    canonical = CanonicalSourceRecordV3(
        asset=verified_asset.expected,
        source_record_ordinal=ordinal,
        retained_byte_count=len(text_bytes),
        native_record_id=native_id if isinstance(native_id, str) else None,
        int_score=int_score if type(int_score) is int else None,
    )
    raw = RawDocumentV3(
        source=source,
        stratum=_SOURCE_STRATA[source],
        stable_source_record_id=canonical.canonical_source_record_id,
        text=raw_text,
    )
    return SourceParseEventV3(
        source_family=source,
        source_record_ordinal=ordinal,
        disposition=RETAIN,
        record=ParsedSourceRecordV3(
            canonical_record=canonical,
            raw_document=raw,
            parser_binding_sha256=binding.binding_sha256,
        ),
        observation=observation,
    )


def _iter_binary_lines(handle: BinaryIO) -> Iterator[bytes]:
    while True:
        line = handle.readline()
        if not line:
            break
        yield line


def _validate_compressed_json_container(handle: BinaryIO, container: str) -> None:
    """Read the whole transport first so truncation outranks row interpretation."""

    try:
        handle.seek(0)
        if container == "jsonl.zst":
            decompressor = zstandard.ZstdDecompressor().decompressobj()
            pending = b""
            frame_count = 0
            while True:
                chunk = pending or handle.read(8 * 1024 * 1024)
                pending = b""
                if not chunk:
                    if frame_count == 0 or not decompressor.eof:
                        raise SourceContainerError(
                            "compressed JSON asset is malformed or truncated"
                        )
                    break
                decompressor.decompress(chunk)
                if decompressor.eof:
                    frame_count += 1
                    pending = decompressor.unused_data
                    decompressor.flush()
                    if pending:
                        decompressor = zstandard.ZstdDecompressor().decompressobj()
                        continue
                    following = handle.read(8 * 1024 * 1024)
                    if not following:
                        break
                    decompressor = zstandard.ZstdDecompressor().decompressobj()
                    pending = following
        elif container == "json.gz":
            with gzip.GzipFile(fileobj=handle, mode="rb") as reader:
                for _ in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                    pass
        else:
            raise SourceSchemaError("compressed validator received a non-JSON container")
    except SourceIOError:
        raise
    except (OSError, EOFError, zstandard.ZstdError) as error:
        raise SourceContainerError(
            "compressed JSON asset is malformed or truncated"
        ) from error


def _iter_jsonl_events(
    compressed: BinaryIO,
    verified_asset: VerifiedLocalCacheAssetV3,
    binding: SourceParserBindingV3,
) -> Iterator[SourceParseEventV3]:
    _validate_compressed_json_container(compressed, binding.container)
    compressed.seek(0)
    observation_emitted = False
    try:
        if binding.container == "jsonl.zst":
            reader: BinaryIO = io.BufferedReader(
                zstandard.ZstdDecompressor().stream_reader(
                    compressed, read_across_frames=True,
                )
            )
        elif binding.container == "json.gz":
            reader = gzip.GzipFile(fileobj=compressed, mode="rb")
        else:
            raise SourceSchemaError("JSON parser received a non-JSON container")
        with closing(reader):
            for ordinal, raw_line in enumerate(_iter_binary_lines(reader)):
                line = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
                if line.endswith(b"\r"):
                    line = line[:-1]
                if not line:
                    raise SourceSchemaError("JSONL source contains an empty row")
                try:
                    decoded = line.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    yield SourceParseEventV3(
                        source_family=binding.source_family,
                        source_record_ordinal=ordinal,
                        disposition=DROP_INVALID_UTF8,
                        record=None,
                        reason="whole JSONL row is not valid UTF-8",
                    )
                    continue
                try:
                    row = json.loads(
                        decoded,
                        object_pairs_hook=_json_no_duplicate_keys,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            SourceSchemaError(
                                f"JSON source row contains non-finite {value}"
                            )
                        ),
                    )
                except SourceSchemaError:
                    raise
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise SourceSchemaError("JSON source row is malformed") from error
                if not isinstance(row, Mapping):
                    raise SourceSchemaError("JSON source row is not an object")
                if tuple(sorted(row)) != binding.exact_json_top_level_fields:
                    raise SourceSchemaError("JSON source top-level schema drifted")
                for field in binding.required_json_fields:
                    value = _lookup_path(row, field.path)
                    _require_json_kind(value, field)
                observation = None
                if not observation_emitted:
                    canonical_row = canonical_json_bytes(row)
                    observation = _record_observation_v3(
                        verified_asset=verified_asset,
                        ordinal=ordinal,
                        row_representation="decompressed_json_object_payload",
                        raw_row=line,
                        canonical_row=canonical_row,
                        schema_descriptor=_json_schema_descriptor(row),
                    )
                    observation_emitted = True
                yield _event_from_values(
                    verified_asset=verified_asset,
                    binding=binding,
                    ordinal=ordinal,
                    text=_lookup_path(row, binding.text_path),
                    native_id=(
                        _lookup_path(row, binding.native_id_path)
                        if binding.native_id_path is not None
                        else None
                    ),
                    int_score=(
                        _lookup_path(row, binding.int_score_path)
                        if binding.int_score_path is not None
                        else None
                    ),
                    observation=observation,
                )
    except SourceIOError:
        raise
    except (OSError, EOFError, zstandard.ZstdError) as error:
        raise SourceContainerError("compressed JSON asset is malformed or truncated") from error


def _iter_parquet_events(
    snapshot: BinaryIO,
    verified_asset: VerifiedLocalCacheAssetV3,
    binding: SourceParserBindingV3,
) -> Iterator[SourceParseEventV3]:
    try:
        snapshot.seek(0)
        parquet = pq.ParquetFile(snapshot)
        observed_projection = tuple(
            (field.name, _arrow_type_name(field.type))
            for field in parquet.schema_arrow
        )
        if observed_projection != binding.declared_parquet_columns:
            raise SourceSchemaError(
                "Parquet card-declared column projection drifted"
            )
        full_schema = parquet.schema_arrow
        full_schema_ipc = full_schema.serialize().to_pybytes()
        schema_descriptor = _arrow_schema_descriptor(full_schema)
        ordinal = 0
        observation_emitted = False
        for batch in parquet.iter_batches(batch_size=4096):
            values = batch.to_pydict()
            for offset in range(batch.num_rows):
                observation = None
                if not observation_emitted:
                    row_batch = batch.slice(offset, 1)
                    sink = pa.BufferOutputStream()
                    with pa.ipc.new_stream(sink, row_batch.schema) as writer:
                        writer.write_batch(row_batch)
                    raw_row = sink.getvalue().to_pybytes()
                    canonical_row = canonical_json_bytes(
                        {
                            field.name: values[field.name][offset]
                            for field in full_schema
                        }
                    )
                    observation = _record_observation_v3(
                        verified_asset=verified_asset,
                        ordinal=ordinal,
                        row_representation="arrow_ipc_single_row",
                        raw_row=raw_row,
                        canonical_row=canonical_row,
                        schema_descriptor=schema_descriptor,
                        arrow_schema_ipc=full_schema_ipc,
                    )
                    observation_emitted = True
                yield _event_from_values(
                    verified_asset=verified_asset,
                    binding=binding,
                    ordinal=ordinal,
                    text=values[binding.text_path[0]][offset],
                    native_id=(
                        values[binding.native_id_path[0]][offset]
                        if binding.native_id_path is not None
                        else None
                    ),
                    int_score=(
                        values[binding.int_score_path[0]][offset]
                        if binding.int_score_path is not None
                        else None
                    ),
                    observation=observation,
                )
                ordinal += 1
    except SourceIOError:
        raise
    except (OSError, pa.ArrowException) as error:
        raise SourceContainerError("Parquet asset is malformed or truncated") from error


def iter_source_asset_events_v3(
    verified_asset: VerifiedLocalCacheAssetV3,
    cache_root: Path,
    *,
    binding: SourceParserBindingV3 | None = None,
    allow_fixture_binding: bool = False,
) -> Iterator[SourceParseEventV3]:
    """Stream one independently verified cache asset through an exact schema."""

    if not isinstance(verified_asset, VerifiedLocalCacheAssetV3):
        raise TypeError("source parser requires a verified local cache asset")
    if not isinstance(cache_root, Path):
        raise TypeError("cache_root must be a pathlib.Path")
    source = verified_asset.expected.source_family
    if binding is None:
        try:
            binding = PRODUCTION_PARSER_BINDINGS_V3[source]
        except KeyError as error:
            raise SourceSchemaBindingRequired(
                f"{source} requires pinned first-record schema inspection"
            ) from error
    if not isinstance(binding, SourceParserBindingV3):
        raise TypeError("source parser binding is untyped")
    if binding.source_family != source:
        raise SourceSchemaError("parser binding belongs to a different source family")
    if binding.authority == "FIXTURE_ONLY" and not allow_fixture_binding:
        raise SourceSchemaError("fixture-only parser binding cannot run in production")
    if binding.authority != "FIXTURE_ONLY":
        frozen = PRODUCTION_PARSER_BINDINGS_V3.get(source)
        if frozen is None or frozen.binding_sha256 != binding.binding_sha256:
            raise SourceSchemaError("production parser binding differs from frozen literal")
    path = _safe_cache_path(
        cache_root.resolve(strict=True),
        verified_asset.expected.relative_path,
        strict=True,
    )
    if not path.is_file():
        raise SourceContainerError("verified cache asset is no longer a regular file")
    stat = path.stat()
    if stat.st_size != verified_asset.observed_bytes:
        raise SourceTransportError("verified cache asset changed size before parsing")
    digest = hashlib.sha256()
    byte_count = 0
    # Parse an anonymous snapshot populated and hashed through the same source
    # handle.  Reopening the governed path after verification would leave a
    # time-of-check/time-of-use gap in which different bytes could be parsed.
    with tempfile.TemporaryFile(mode="w+b") as snapshot:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise SourceTransportError("cache reader returned non-bytes data")
                byte_count += len(chunk)
                digest.update(chunk)
                snapshot.write(chunk)
        if (
            byte_count != verified_asset.observed_bytes
            or digest.hexdigest() != verified_asset.observed_sha256
        ):
            raise SourceTransportError(
                "verified cache asset changed hash or size before parsing"
            )
        snapshot.flush()
        snapshot.seek(0)
        if binding.container in {"jsonl.zst", "json.gz"}:
            yield from _iter_jsonl_events(snapshot, verified_asset, binding)
        elif binding.container == "parquet":
            yield from _iter_parquet_events(snapshot, verified_asset, binding)
        else:
            raise SourceSchemaError("parser binding uses an unsupported container")


def iter_retained_source_records_v3(
    events: Iterable[SourceParseEventV3],
) -> Iterator[ParsedSourceRecordV3]:
    """Select retained records without suppressing malformed-asset exceptions."""

    for event in events:
        if not isinstance(event, SourceParseEventV3):
            raise TypeError("parse stream contains an untyped event")
        if event.disposition == RETAIN:
            assert event.record is not None
            yield event.record


@dataclass(frozen=True, init=False)
class ParsedSourceSpoolReceiptV3:
    """Factory proof that parsed text is backed by exact verified cache bytes."""

    enumeration_receipt_sha256: str
    enumeration_mode: str
    selection_plan_sha256: str
    download_receipt_sha256: str
    verification_receipt_sha256: str
    source_manifest_sha256: str
    parser_bindings: tuple[tuple[str, str], ...]
    observations: tuple[SourceRecordObservationV3, ...]
    event_count: int
    retained_record_count: int
    drop_counts: tuple[tuple[str, int], ...]
    retained_content_identity_sha256: str
    spool_bytes: int
    spool_sha256: str

    def __new__(cls) -> "ParsedSourceSpoolReceiptV3":
        raise TypeError("parsed spool receipts are factory-minted from cache bytes")

    @classmethod
    def _validated(
        cls,
        *,
        enumeration_receipt_sha256: str,
        enumeration_mode: str,
        selection_plan_sha256: str,
        download_receipt_sha256: str,
        verification_receipt_sha256: str,
        source_manifest_sha256: str,
        parser_bindings: tuple[tuple[str, str], ...],
        observations: tuple[SourceRecordObservationV3, ...],
        event_count: int,
        retained_record_count: int,
        drop_counts: tuple[tuple[str, int], ...],
        retained_content_identity_sha256: str,
        spool_bytes: int,
        spool_sha256: str,
        sentinel: object,
    ) -> "ParsedSourceSpoolReceiptV3":
        if sentinel is not _PARSED_SPOOL_FACTORY_SENTINEL:
            raise PermissionError("parsed spool receipts are factory-only")
        instance = object.__new__(cls)
        for name, value in locals().copy().items():
            if name not in {"cls", "sentinel", "instance"}:
                object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        for value, name in (
            (self.enumeration_receipt_sha256, "enumeration receipt"),
            (self.selection_plan_sha256, "selection plan"),
            (self.download_receipt_sha256, "download receipt"),
            (self.verification_receipt_sha256, "verification receipt"),
            (self.source_manifest_sha256, "source manifest"),
            (
                self.retained_content_identity_sha256,
                "retained content identity",
            ),
            (self.spool_sha256, "spool SHA-256"),
        ):
            _require_sha256(value, name)
        if self.enumeration_mode not in {AUTHORITATIVE_MODE, FIXTURE_MODE}:
            raise ValueError("parsed spool enumeration mode is invalid")
        if not self.parser_bindings or tuple(
            source for source, unused in self.parser_bindings
        ) != tuple(
            source
            for source in _SOURCE_CONTAINERS
            if source in {item[0] for item in self.parser_bindings}
        ):
            raise ValueError("parsed spool parser bindings are not canonical")
        for source, digest in self.parser_bindings:
            if source not in _SOURCE_CONTAINERS:
                raise ValueError("parsed spool uses an unknown parser family")
            _require_sha256(digest, "parser binding")
        if not self.observations or any(
            not isinstance(item, SourceRecordObservationV3)
            for item in self.observations
        ):
            raise ValueError("parsed spool requires typed record observations")
        for value, name, minimum in (
            (self.event_count, "event_count", 1),
            (self.retained_record_count, "retained_record_count", 0),
            (self.spool_bytes, "spool_bytes", 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} is invalid")
        if self.retained_record_count > self.event_count:
            raise ValueError("parsed spool retains more records than events")
        if tuple(name for name, count in self.drop_counts) != tuple(
            disposition for disposition in PARSE_DISPOSITIONS if disposition != RETAIN
        ):
            raise ValueError("parsed spool drop-count order is not canonical")
        if any(type(count) is not int or count < 0 for unused, count in self.drop_counts):
            raise ValueError("parsed spool drop count is invalid")
        if self.retained_record_count + sum(
            count for unused, count in self.drop_counts
        ) != self.event_count:
            raise ValueError("parsed spool disposition counts do not cover events")

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(
            {"payload": self, "schema": PARSED_SOURCE_SPOOL_SCHEMA_V3}
        )


def _selected_upstream_for_cache(
    enumeration: UpstreamEnumerationReceiptV3,
    verified_cache: VerifiedLocalCacheManifestV3,
) -> tuple[UpstreamAssetV3, ...]:
    complete = tuple(
        asset for family in enumeration.families for asset in family.assets
    )
    upstream = {
        (
            asset.source_family,
            asset.repository,
            asset.config,
            asset.revision,
            asset.split,
            asset.asset_locator,
        ): asset
        for asset in complete
    }
    selected_keys: set[tuple[str, str, str, str, str, str]] = set()
    for cached in verified_cache.source_manifest.assets:
        key = (
            cached.source_family,
            cached.repository,
            cached.config,
            cached.revision,
            cached.split,
            cached.asset_locator,
        )
        item = upstream.get(key)
        if item is None:
            raise SourceTransportError(
                "parsed cache asset is absent from its enumeration"
            )
        if item.upstream_bytes != cached.bytes:
            raise SourceTransportError(
                "parsed cache byte count differs from its enumeration"
            )
        selected_keys.add(key)
    selected = tuple(
        asset
        for asset in complete
        if (
            asset.source_family,
            asset.repository,
            asset.config,
            asset.revision,
            asset.split,
            asset.asset_locator,
        )
        in selected_keys
    )
    if len(selected) != len(selected_keys):
        raise SourceTransportError("parsed cache subset is incomplete")
    return selected


def _canonical_spool_event(
    *,
    asset_order_ordinal: int,
    verified_asset: VerifiedLocalCacheAssetV3,
    event_ordinal: int,
    event: SourceParseEventV3,
    binding: SourceParserBindingV3,
) -> Mapping[str, object]:
    retained: Mapping[str, object] | None = None
    if event.disposition == RETAIN:
        assert event.record is not None
        record = event.record
        text = record.raw_document.text
        text_bytes = text if isinstance(text, bytes) else text.encode("utf-8", errors="strict")
        text_string = text_bytes.decode("utf-8", errors="strict")
        if len(text_bytes) != record.canonical_record.retained_byte_count:
            raise SourceSchemaError("parsed text bytes disagree with canonical metadata")
        retained = {
            "canonical_record": asdict(record.canonical_record),
            "parser_binding_sha256": record.parser_binding_sha256,
            "raw_document": {
                "source": record.raw_document.source,
                "stable_source_record_id": (
                    record.raw_document.stable_source_record_id
                ),
                "stratum": record.raw_document.stratum,
                "text": text_string,
            },
            "text_utf8_bytes": len(text_bytes),
            "text_utf8_sha256": hashlib.sha256(text_bytes).hexdigest(),
        }
    return {
        "asset_order_ordinal": asset_order_ordinal,
        "disposition": event.disposition,
        "event_ordinal": event_ordinal,
        "event_sha256": event.event_sha256,
        "observation": (
            None if event.observation is None else asdict(event.observation)
        ),
        "parser_binding_sha256": binding.binding_sha256,
        "reason": event.reason,
        "retained": retained,
        "source_asset_identity_sha256": (
            verified_asset.expected.asset_identity_sha256
        ),
        "source_asset_sha256": verified_asset.observed_sha256,
        "source_family": event.source_family,
        "source_record_ordinal": event.source_record_ordinal,
    }


def verify_parsed_source_spool_v3(
    receipt: ParsedSourceSpoolReceiptV3,
    spool_path: Path,
) -> None:
    """Rehash and structurally replay a parsed spool without trusting its path."""

    if not isinstance(receipt, ParsedSourceSpoolReceiptV3):
        raise TypeError("parsed spool verifier requires a typed receipt")
    if not isinstance(spool_path, Path):
        raise TypeError("parsed spool path must be a pathlib.Path")
    assert_no_symlink_ancestors(spool_path)
    digest = hashlib.sha256()
    byte_count = 0
    rows: list[Mapping[str, object]] = []
    with spool_path.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.endswith(b"\n") or raw_line == b"\n":
                raise SourceTransportError("parsed spool has invalid JSONL framing")
            try:
                row = json.loads(
                    raw_line[:-1].decode("utf-8", errors="strict"),
                    object_pairs_hook=_json_no_duplicate_keys,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        SourceSchemaError(
                            f"parsed spool contains non-finite {value}"
                        )
                    ),
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SourceSchemaError("parsed spool is not strict JSONL") from error
            if not isinstance(row, Mapping) or canonical_json_bytes(row) + b"\n" != raw_line:
                raise SourceSchemaError("parsed spool row is not canonical JSON")
            digest.update(raw_line)
            byte_count += len(raw_line)
            rows.append(row)
    if byte_count != receipt.spool_bytes or digest.hexdigest() != receipt.spool_sha256:
        raise SourceTransportError("parsed spool bytes differ from their receipt")
    if len(rows) != receipt.event_count:
        raise SourceTransportError("parsed spool event count differs from its receipt")
    if tuple(row.get("event_ordinal") for row in rows) != tuple(range(len(rows))):
        raise SourceSchemaError("parsed spool event order is noncanonical")
    retained_rows = tuple(row for row in rows if row.get("disposition") == RETAIN)
    if len(retained_rows) != receipt.retained_record_count:
        raise SourceTransportError("parsed spool retained count drifted")
    observed_payloads = tuple(
        row["observation"] for row in rows if row.get("observation") is not None
    )
    expected_observations = tuple(asdict(item) for item in receipt.observations)
    if canonical_json_bytes(observed_payloads) != canonical_json_bytes(
        expected_observations
    ):
        raise SourceTransportError("parsed spool observations differ from receipt")
    retained_identity = canonical_sha256(
        tuple(row["retained"] for row in retained_rows)
    )
    if retained_identity != receipt.retained_content_identity_sha256:
        raise SourceTransportError("parsed spool retained-content identity drifted")


def materialize_parsed_source_spool_v3(
    enumeration: UpstreamEnumerationReceiptV3,
    download_receipt: SourceCacheDownloadReceiptV3,
    verified_cache: VerifiedLocalCacheManifestV3,
    cache_root: Path,
    spool_path: Path,
    *,
    parser_bindings: Mapping[str, SourceParserBindingV3] | None = None,
    allow_nonauthoritative_fixture: bool = False,
) -> ParsedSourceSpoolReceiptV3:
    """Parse a verified selected cache into one canonical text-bearing spool."""

    if not isinstance(enumeration, UpstreamEnumerationReceiptV3):
        raise TypeError("parsed spool requires a typed enumeration")
    if not isinstance(download_receipt, SourceCacheDownloadReceiptV3):
        raise TypeError("parsed spool requires a factory download receipt")
    if not isinstance(verified_cache, VerifiedLocalCacheManifestV3):
        raise TypeError("parsed spool requires a factory verified cache")
    if not isinstance(cache_root, Path) or not isinstance(spool_path, Path):
        raise TypeError("parsed spool paths must be pathlib.Path values")
    if not enumeration.authoritative and not allow_nonauthoritative_fixture:
        raise SourceTransportError("production parsed spool requires authoritative input")
    if (
        download_receipt.enumeration_receipt_sha256 != enumeration.receipt_sha256
        or download_receipt.enumeration_mode != enumeration.mode
        or download_receipt.source_manifest != verified_cache.source_manifest
        or download_receipt.verification_receipt_sha256
        != verified_cache.verification_receipt_sha256
    ):
        raise SourceTransportError(
            "enumeration, download, and verified-cache receipts do not compose"
        )
    selected = _selected_upstream_for_cache(enumeration, verified_cache)
    plan = plan_source_cache_assets_v3(enumeration, selected)
    if plan.plan_sha256 != download_receipt.selection_plan_sha256:
        raise SourceTransportError("parsed cache selection differs from download plan")
    bindings = (
        PRODUCTION_PARSER_BINDINGS_V3
        if parser_bindings is None
        else parser_bindings
    )
    if not isinstance(bindings, Mapping):
        raise TypeError("parser_bindings must be a mapping")
    selected_sources = tuple(
        source
        for source in _SOURCE_CONTAINERS
        if any(
            asset.expected.source_family == source
            for asset in verified_cache.assets
        )
    )
    bound = tuple((source, bindings[source]) for source in selected_sources)
    if any(not isinstance(binding, SourceParserBindingV3) for unused, binding in bound):
        raise TypeError("parsed spool contains an untyped parser binding")

    assert_no_symlink_ancestors(cache_root)
    assert_no_symlink_ancestors(spool_path)
    if spool_path.exists():
        raise SourceTransportError("refusing to overwrite a parsed spool")
    spool_path.parent.mkdir(parents=True, exist_ok=True)
    partial = spool_path.with_name(spool_path.name + ".partial")
    if partial.exists():
        raise SourceTransportError("stale parsed spool partial exists")
    spool_digest = hashlib.sha256()
    spool_bytes = 0
    event_count = 0
    retained_payloads: list[Mapping[str, object]] = []
    observations: list[SourceRecordObservationV3] = []
    drop_counts = {name: 0 for name in PARSE_DISPOSITIONS if name != RETAIN}
    try:
        with partial.open("xb") as handle:
            for asset_order_ordinal, verified_asset in enumerate(
                verified_cache.assets
            ):
                source = verified_asset.expected.source_family
                binding = bindings[source]
                asset_observation_count = 0
                for event in iter_source_asset_events_v3(
                    verified_asset,
                    cache_root,
                    binding=binding,
                    allow_fixture_binding=allow_nonauthoritative_fixture,
                ):
                    row = _canonical_spool_event(
                        asset_order_ordinal=asset_order_ordinal,
                        verified_asset=verified_asset,
                        event_ordinal=event_count,
                        event=event,
                        binding=binding,
                    )
                    raw = canonical_json_bytes(row) + b"\n"
                    handle.write(raw)
                    spool_digest.update(raw)
                    spool_bytes += len(raw)
                    event_count += 1
                    if event.observation is not None:
                        asset_observation_count += 1
                        observations.append(event.observation)
                    if event.disposition == RETAIN:
                        retained = row["retained"]
                        assert isinstance(retained, Mapping)
                        retained_payloads.append(retained)
                    else:
                        drop_counts[event.disposition] += 1
                if asset_observation_count != 1:
                    raise SourceSchemaError(
                        "each parsed cache asset must yield exactly one actual "
                        "record/schema observation"
                    )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, spool_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    receipt = ParsedSourceSpoolReceiptV3._validated(
        enumeration_receipt_sha256=enumeration.receipt_sha256,
        enumeration_mode=enumeration.mode,
        selection_plan_sha256=plan.plan_sha256,
        download_receipt_sha256=download_receipt.receipt_sha256,
        verification_receipt_sha256=verified_cache.verification_receipt_sha256,
        source_manifest_sha256=verified_cache.source_manifest.manifest_sha256,
        parser_bindings=tuple(
            (source, binding.binding_sha256) for source, binding in bound
        ),
        observations=tuple(observations),
        event_count=event_count,
        retained_record_count=len(retained_payloads),
        drop_counts=tuple((name, drop_counts[name]) for name in drop_counts),
        retained_content_identity_sha256=canonical_sha256(
            tuple(retained_payloads)
        ),
        spool_bytes=spool_bytes,
        spool_sha256=spool_digest.hexdigest(),
        sentinel=_PARSED_SPOOL_FACTORY_SENTINEL,
    )
    verify_parsed_source_spool_v3(receipt, spool_path)
    return receipt


__all__ = [
    "DROP_EMPTY_TEXT",
    "DROP_INVALID_UTF8",
    "DROP_QUALITY_LT3",
    "PARSE_DISPOSITIONS",
    "PARSED_SOURCE_SPOOL_SCHEMA_V3",
    "PRODUCTION_PARSER_BINDINGS_V3",
    "RETAIN",
    "DownloadedAssetEvidenceV3",
    "JsonFieldV3",
    "ParsedSourceRecordV3",
    "ParsedSourceSpoolReceiptV3",
    "SourceAssetDownloadPlanV3",
    "SourceCacheDownloadReceiptV3",
    "SourceCachePlanMaterializationV3",
    "SourceContainerError",
    "SourceIOError",
    "SourceParseEventV3",
    "SourceRecordObservationV3",
    "SourceParserBindingV3",
    "SourceSchemaBindingRequired",
    "SourceSchemaError",
    "SourceTransportError",
    "fixture_source_parser_binding_v3",
    "finalize_source_cache_v3",
    "iter_retained_source_records_v3",
    "iter_source_asset_events_v3",
    "load_source_cache_download_receipt_v3",
    "materialize_parsed_source_spool_v3",
    "materialize_complete_fixture_source_cache_v3",
    "materialize_source_cache_v3",
    "plan_source_cache_assets_v3",
    "verify_parsed_source_spool_v3",
    "write_source_cache_download_receipt_v3",
]
