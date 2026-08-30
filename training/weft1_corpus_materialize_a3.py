"""Forward-only V4 transport bridge for offline WEFT-1 P-A materialization.

The A2/V3 materializer is a frozen deterministic algorithm.  Amendment A3
changed only the upstream route/transport authority, producing V4 enumeration,
cache-manifest, and download receipts.  This module validates those V4
artifacts, independently rehashes the local cache, adapts the verified assets
to the frozen parser interface, and upgrades the freshly produced V3 manifests
to the V4 authority domain.  It never rewrites a banked V3 receipt or source
artifact and contains no network client.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tempfile
from typing import Any, Iterator, Mapping, TypeVar

from training.weft1_corpus_a2 import (
    A2_CAMPAIGN_ROOT_SEED,
    A2_ZSTD_CODEC_BINDING,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
    JsonlZstdShardIdentityV3,
    StableDocumentV3,
    execution_authority_v3_bound_sha256,
)
from training.weft1_corpus_a3 import (
    EffectiveSourceRouteA3,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
    execution_authority_v4_bound_sha256,
)
from training.weft1_corpus_fetch_a3 import (
    AUTHORITATIVE_MODE,
    DOWNLOAD_RECEIPT_ARTIFACT_SCHEMA_V4,
    PAExecutionBindingV4,
    UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V4,
    DownloadedAssetEvidenceV4,
    FamilyEnumerationV4,
    SourceAssetDownloadPlanV4,
    SourceCacheAssetV4,
    SourceCacheDownloadReceiptV4,
    SourceCacheManifestV4,
    UpstreamAssetV4,
    UpstreamEnumerationReceiptV4,
    VerifiedLocalCacheV4,
    load_pa_source_execution_context_v4,
)
from training.weft1_corpus_materialize_a2 import (
    GTOK_TRAINING_SEEDS,
    MATERIALIZER_ALGORITHM_VERSION,
    MATERIALIZER_SCHEMA,
    PRODUCTION_MODE,
    MaterializationInputV3,
    MaterializationPlanV3,
    MaterializationResultV3,
    materialize_corpus_pa_v3,
)
from training import weft1_corpus_pa as production_io
from training.weft1_corpus_sources_a2 import VerifiedLocalCacheAssetV3
from training.weft1_gtok_a1_contract import SOURCE_FAMILIES
from training.weft1_gtok_contract import (
    GTOK_STRATA,
    GTOK_VOCABULARY_ARMS,
    canonical_json_bytes,
)
from training.weft1_seed import derive_module_seed
from training.weft1_strict_io import (
    assert_no_symlink_ancestors,
    load_canonical_json_snapshot,
)


SOURCE_CACHE_MANIFEST_ARTIFACT_SCHEMA_V4 = (
    "weft1_local_source_cache_manifest_artifact_v4"
)
MATERIALIZATION_TRANSPORT_INPUT_SCHEMA_V4 = (
    "weft1_corpus_materialization_transport_input_v4"
)
MATERIALIZATION_BRIDGE_SCHEMA_V4 = "weft1_corpus_materialization_bridge_v4"
RELEASE_MANIFEST_SECTION_SCHEMA_V4 = "weft1_release_manifest_section_v4"
MATERIALIZER_SCHEMA_V4 = "weft1_corpus_pa_materialization_v4"
D1_READY_SCHEMA_V4 = "weft1_corpus_d1_ready_manifest_v4"
MATERIALIZED_CONTENT_SCHEMA_V4 = "weft1_corpus_materialized_content_v4"
D1_READY_IDENTITY_SCHEMA_V4 = "weft1_corpus_d1_ready_inventory_v4"
BRIDGE_RELATIVE_PATH_V4 = "transport/v4-materialization-bridge.json"
FULL_SHARD_MANIFEST_RELATIVE_PATH_V4 = "artifacts/full-shard-manifest-v4.json"
SCREEN_SUBMANIFEST_RELATIVE_PATH_V4 = "artifacts/screen-submanifest-v4.json"
D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4 = "artifacts/d6-physical-evidence-v4.json"
FULL_SHARD_MANIFEST_SCHEMA_V4 = "weft1_corpus_full_shard_manifest_v4"
SCREEN_SUBMANIFEST_SCHEMA_V4 = "weft1_corpus_screen_submanifest_v4"
D6_PHYSICAL_EVIDENCE_SCHEMA_V4 = "weft1_corpus_d6_physical_evidence_v4"
CONSUMER_ORDER_SCHEMA_V4 = "weft1_corpus_consumer_order_receipt_v4"
TOKENIZER_FIT_INPUT_SCHEMA_V4 = "weft1_gtok_tokenizer_fit_input_v4"
V4_READINESS = "AUTHORITATIVE_V4_INPUTS_D1_READY_NO_GATE_MINT"

# The A2 seed row has three distinct identities: the row's training seed, the
# module-initialization seed, and the data-order seed.  The public V4 consumer
# API remains keyed by ``training_seed`` so all four vocabulary arms share the
# same two registered rows, but the physical permutation must be driven by the
# row's derived ``data_order_seed``.  Keep the governed literals here rather
# than importing the G-TOK campaign (which depends on this materializer).
GTOK_GOVERNED_DATA_ORDER_SEED_ROWS_V4 = (
    (4_069_725_298_476_216_533, 10_666_192_988_433_719_740),
    (13_256_058_689_613_801_745, 4_197_282_192_878_334_768),
)
_DERIVED_DATA_ORDER_SEED_ROWS_V4 = tuple(
    (
        training_seed,
        derive_module_seed(
            A2_CAMPAIGN_ROOT_SEED,
            f"gtok.data.shared.{training_seed}",
        ),
    )
    for training_seed in GTOK_TRAINING_SEEDS
)
if _DERIVED_DATA_ORDER_SEED_ROWS_V4 != GTOK_GOVERNED_DATA_ORDER_SEED_ROWS_V4:
    raise RuntimeError("A2 governed V4 data-order seed derivation drifted")
GTOK_DATA_ORDER_SEED_BY_TRAINING_SEED_V4 = dict(
    GTOK_GOVERNED_DATA_ORDER_SEED_ROWS_V4
)


class CorpusMaterializationV4Error(RuntimeError):
    """A V4 transport or forward-finalization invariant failed."""


T = TypeVar("T")


def _exact_mapping(value: object, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CorpusMaterializationV4Error(f"{name} fields drifted")
    return value


def _plain_dataclass(cls: type[T], value: object, name: str) -> T:
    expected = {field.name for field in fields(cls)}
    raw = _exact_mapping(value, expected, name)
    return cls(**dict(raw))


def _binding(value: object) -> PAExecutionBindingV4:
    raw = _exact_mapping(
        value,
        {field.name for field in fields(PAExecutionBindingV4)},
        "V4 execution binding",
    )
    payload = dict(raw)
    projections = payload["family_projection_sha256s"]
    if not isinstance(projections, list) or any(
        not isinstance(row, list) or len(row) != 2 for row in projections
    ):
        raise CorpusMaterializationV4Error("V4 family projections are not pairs")
    payload["family_projection_sha256s"] = tuple(
        (str(row[0]), str(row[1])) for row in projections
    )
    return PAExecutionBindingV4(**payload)


def _route(value: object) -> EffectiveSourceRouteA3:
    return _plain_dataclass(EffectiveSourceRouteA3, value, "effective V4 route")


def _upstream_asset(value: object) -> UpstreamAssetV4:
    return _plain_dataclass(UpstreamAssetV4, value, "V4 upstream asset")


def _family_enumeration(value: object) -> FamilyEnumerationV4:
    raw = _exact_mapping(
        value,
        {field.name for field in fields(FamilyEnumerationV4)},
        "V4 family enumeration",
    )
    assets = raw["assets"]
    if not isinstance(assets, list):
        raise CorpusMaterializationV4Error("V4 family assets must be a list")
    payload = dict(raw)
    payload["route"] = _route(raw["route"])
    payload["assets"] = tuple(_upstream_asset(item) for item in assets)
    return FamilyEnumerationV4(**payload)


def _enumeration(value: object) -> UpstreamEnumerationReceiptV4:
    raw = _exact_mapping(
        value,
        {field.name for field in fields(UpstreamEnumerationReceiptV4)},
        "V4 enumeration receipt",
    )
    families = raw["families"]
    if not isinstance(families, list):
        raise CorpusMaterializationV4Error("V4 enumeration families must be a list")
    payload = dict(raw)
    payload["execution_binding"] = _binding(raw["execution_binding"])
    payload["families"] = tuple(_family_enumeration(item) for item in families)
    return UpstreamEnumerationReceiptV4(**payload)


def _cache_asset(value: object) -> SourceCacheAssetV4:
    return _plain_dataclass(SourceCacheAssetV4, value, "V4 cache asset")


def _source_manifest(value: object) -> SourceCacheManifestV4:
    raw = _exact_mapping(
        value,
        {field.name for field in fields(SourceCacheManifestV4)},
        "V4 source manifest",
    )
    assets = raw["assets"]
    if not isinstance(assets, list):
        raise CorpusMaterializationV4Error("V4 cache assets must be a list")
    payload = dict(raw)
    payload["execution_binding"] = _binding(raw["execution_binding"])
    payload["assets"] = tuple(_cache_asset(item) for item in assets)
    return SourceCacheManifestV4(**payload)


def _download_evidence(value: object) -> DownloadedAssetEvidenceV4:
    return _plain_dataclass(
        DownloadedAssetEvidenceV4,
        value,
        "V4 download evidence",
    )


def _download_receipt(value: object) -> SourceCacheDownloadReceiptV4:
    raw = _exact_mapping(
        value,
        {field.name for field in fields(SourceCacheDownloadReceiptV4)},
        "V4 download receipt",
    )
    evidence = raw["evidence"]
    if not isinstance(evidence, list):
        raise CorpusMaterializationV4Error("V4 download evidence must be a list")
    payload = dict(raw)
    payload["execution_binding"] = _binding(raw["execution_binding"])
    payload["source_manifest"] = _source_manifest(raw["source_manifest"])
    payload["evidence"] = tuple(_download_evidence(item) for item in evidence)
    return SourceCacheDownloadReceiptV4(**payload)


def _load_envelope(
    path: Path,
    *,
    schema: str,
    payload_key: str,
    decoder: Any,
    receipt_attribute: str,
) -> tuple[bytes, object]:
    if not isinstance(path, Path):
        raise TypeError("V4 artifact path must be pathlib.Path")
    assert_no_symlink_ancestors(path)
    raw, envelope = load_canonical_json_snapshot(path)
    if raw != canonical_json_bytes(envelope) + b"\n":
        raise CorpusMaterializationV4Error("V4 artifact is not canonical JSON")
    expected_keys = {payload_key, f"{payload_key}_sha256", "schema"}
    payload = _exact_mapping(envelope, expected_keys, "V4 artifact envelope")
    if payload["schema"] != schema:
        raise CorpusMaterializationV4Error("V4 artifact schema drifted")
    decoded = decoder(payload[payload_key])
    if payload[f"{payload_key}_sha256"] != getattr(decoded, receipt_attribute):
        raise CorpusMaterializationV4Error("V4 artifact typed receipt drifted")
    return raw, decoded


def load_upstream_enumeration_artifact_v4(path: Path) -> UpstreamEnumerationReceiptV4:
    _, decoded = _load_envelope(
        path,
        schema=UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V4,
        payload_key="receipt",
        decoder=_enumeration,
        receipt_attribute="receipt_sha256",
    )
    if not isinstance(decoded, UpstreamEnumerationReceiptV4):
        raise TypeError("V4 enumeration decoder returned the wrong type")
    return decoded


def load_source_manifest_artifact_v4(path: Path) -> SourceCacheManifestV4:
    _, decoded = _load_envelope(
        path,
        schema=SOURCE_CACHE_MANIFEST_ARTIFACT_SCHEMA_V4,
        payload_key="manifest",
        decoder=_source_manifest,
        receipt_attribute="receipt_sha256",
    )
    if not isinstance(decoded, SourceCacheManifestV4):
        raise TypeError("V4 manifest decoder returned the wrong type")
    return decoded


def load_cache_download_artifact_v4(path: Path) -> SourceCacheDownloadReceiptV4:
    _, decoded = _load_envelope(
        path,
        schema=DOWNLOAD_RECEIPT_ARTIFACT_SCHEMA_V4,
        payload_key="receipt",
        decoder=_download_receipt,
        receipt_attribute="receipt_sha256",
    )
    if not isinstance(decoded, SourceCacheDownloadReceiptV4):
        raise TypeError("V4 download decoder returned the wrong type")
    return decoded


@dataclass(frozen=True)
class VerifiedCacheAdapterV4:
    """V4 receipt plus V3-shaped verified assets for the frozen parsers."""

    typed_receipt: VerifiedLocalCacheV4
    assets: tuple[VerifiedLocalCacheAssetV3, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.typed_receipt, VerifiedLocalCacheV4):
            raise TypeError("V4 cache adapter requires a typed verification receipt")
        if tuple(asset.expected for asset in self.assets) != (
            self.typed_receipt.source_manifest.assets
        ):
            raise ValueError("V4 parser adapter differs from the verified manifest")

    @property
    def source_manifest(self) -> SourceCacheManifestV4:
        return self.typed_receipt.source_manifest

    @property
    def cache_root_label(self) -> str:
        return self.typed_receipt.cache_root_label

    @property
    def verification_receipt_sha256(self) -> str:
        return self.typed_receipt.receipt_sha256


def _safe_cache_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise CorpusMaterializationV4Error("V4 cache path is not canonical relative POSIX")
    lexical = root.joinpath(*pure.parts)
    assert_no_symlink_ancestors(lexical)
    resolved = lexical.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise CorpusMaterializationV4Error("V4 cache asset escapes its governed root")
    return resolved


def _verify_cache(
    manifest: SourceCacheManifestV4,
    cache_root: Path,
) -> VerifiedCacheAdapterV4:
    assert_no_symlink_ancestors(cache_root)
    root = cache_root.resolve(strict=True)
    if not root.is_dir():
        raise CorpusMaterializationV4Error("V4 cache root is not a real directory")
    observations: list[tuple[str, int, str]] = []
    parser_assets: list[VerifiedLocalCacheAssetV3] = []
    for asset in manifest.assets:
        path = _safe_cache_path(root, asset.relative_path)
        digest = hashlib.sha256()
        byte_count = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                byte_count += len(chunk)
                digest.update(chunk)
        observed_sha = digest.hexdigest()
        if byte_count != asset.bytes or observed_sha != asset.sha256:
            raise CorpusMaterializationV4Error("V4 cache bytes differ from their manifest")
        observations.append((asset.relative_path, byte_count, observed_sha))
        parser_assets.append(
            VerifiedLocalCacheAssetV3(
                expected=asset,
                observed_bytes=byte_count,
                observed_sha256=observed_sha,
            )
        )
    typed = VerifiedLocalCacheV4(
        execution_binding=manifest.execution_binding,
        source_manifest=manifest,
        cache_root_label=cache_root.name or ".",
        observations=tuple(observations),
    )
    return VerifiedCacheAdapterV4(typed_receipt=typed, assets=tuple(parser_assets))


def _verify_transport_composition(
    enumeration: UpstreamEnumerationReceiptV4,
    download: SourceCacheDownloadReceiptV4,
    verified: VerifiedCacheAdapterV4,
) -> None:
    binding = enumeration.execution_binding
    if (
        not enumeration.authoritative
        or enumeration.mode != AUTHORITATIVE_MODE
        or download.enumeration_mode != AUTHORITATIVE_MODE
        or download.execution_binding != binding
        or verified.typed_receipt.execution_binding != binding
        or download.enumeration_receipt_sha256 != enumeration.receipt_sha256
        or download.source_manifest != verified.source_manifest
        or download.verification_receipt_sha256
        != verified.verification_receipt_sha256
        or download.selection_plan_sha256 != verified.source_manifest.selection_plan_sha256
    ):
        raise CorpusMaterializationV4Error("V4 enumeration/download/cache receipts do not compose")

    upstream_by_key: dict[tuple[str, str], UpstreamAssetV4] = {}
    for family in enumeration.families:
        for asset in family.assets:
            key = (asset.source_family, asset.asset_locator)
            if key in upstream_by_key:
                raise CorpusMaterializationV4Error("V4 enumeration repeats a source locator")
            upstream_by_key[key] = asset
    selected_keys = {
        (asset.source_family, asset.asset_locator) for asset in verified.source_manifest.assets
    }
    selected_upstream = tuple(
        asset
        for family in enumeration.families
        for asset in family.assets
        if (asset.source_family, asset.asset_locator) in selected_keys
    )
    if len(selected_upstream) != len(selected_keys):
        raise CorpusMaterializationV4Error("V4 cache includes an unenrolled upstream asset")
    cache_by_key = {
        (asset.source_family, asset.asset_locator): asset
        for asset in verified.source_manifest.assets
    }
    evidence_by_cache = {
        item.source_cache_asset_identity_sha256: item for item in download.evidence
    }
    if len(evidence_by_cache) != len(download.evidence):
        raise CorpusMaterializationV4Error("V4 download evidence repeats a cache identity")
    for upstream in selected_upstream:
        key = (upstream.source_family, upstream.asset_locator)
        cached = cache_by_key[key]
        evidence = evidence_by_cache.get(cached.asset_identity_sha256)
        if evidence is None:
            raise CorpusMaterializationV4Error("V4 cache lacks download evidence")
        if (
            cached.repository,
            cached.config,
            cached.revision,
            cached.split,
            cached.bytes,
            cached.effective_route_receipt_sha256,
            cached.execution_binding_sha256,
        ) != (
            upstream.repository,
            upstream.config,
            upstream.revision,
            upstream.split,
            upstream.upstream_bytes,
            upstream.effective_route_receipt_sha256,
            upstream.execution_binding_sha256,
        ):
            raise CorpusMaterializationV4Error("V4 cached route differs from enumeration")
        if upstream.content_sha256 is not None and upstream.content_sha256 != cached.sha256:
            raise CorpusMaterializationV4Error("V4 cache hash differs from upstream content")
        if evidence.upstream_asset_identity_sha256 != upstream.asset_identity_sha256:
            raise CorpusMaterializationV4Error("V4 download evidence names another upstream asset")
    plan = SourceAssetDownloadPlanV4(
        execution_binding=binding,
        enumeration_receipt_sha256=enumeration.receipt_sha256,
        enumeration_mode=enumeration.mode,
        assets=selected_upstream,
    )
    if plan.receipt_sha256 != download.selection_plan_sha256:
        raise CorpusMaterializationV4Error("V4 selected subset/order differs from its plan")


class MaterializationInputV4(MaterializationInputV3):
    """A V4 production input accepted by the unchanged V3 algorithm surface."""

    def __post_init__(self) -> None:
        if self.mode != PRODUCTION_MODE or self.streams:
            raise CorpusMaterializationV4Error("V4 bridge is production transport only")
        if not isinstance(self.upstream_enumeration, UpstreamEnumerationReceiptV4):
            raise CorpusMaterializationV4Error("V4 bridge requires a typed enumeration")
        if not isinstance(self.verified_cache, VerifiedCacheAdapterV4):
            raise CorpusMaterializationV4Error("V4 bridge requires a verified cache adapter")
        if not isinstance(self.source_cache_download_receipt, SourceCacheDownloadReceiptV4):
            raise CorpusMaterializationV4Error("V4 bridge requires a typed download receipt")
        if not isinstance(self.cache_root, Path) or self.fixture_source_identity_sha256 is not None:
            raise CorpusMaterializationV4Error("V4 production input shape drifted")
        _verify_transport_composition(
            self.upstream_enumeration,
            self.source_cache_download_receipt,
            self.verified_cache,
        )
        root = assert_no_symlink_ancestors(self.cache_root).resolve(strict=True)
        if not root.is_dir() or (root.name or ".") != self.verified_cache.cache_root_label:
            raise CorpusMaterializationV4Error("V4 cache root differs from its verification")

    @property
    def source_identity_sha256(self) -> str:
        assert isinstance(self.upstream_enumeration, UpstreamEnumerationReceiptV4)
        assert isinstance(self.verified_cache, VerifiedCacheAdapterV4)
        assert isinstance(self.source_cache_download_receipt, SourceCacheDownloadReceiptV4)
        return execution_authority_v4_bound_sha256(
            MATERIALIZATION_TRANSPORT_INPUT_SCHEMA_V4,
            {
                "execution_binding_sha256": self.upstream_enumeration.execution_binding.receipt_sha256,
                "upstream_enumeration_receipt_sha256": self.upstream_enumeration.receipt_sha256,
                "cache_download_receipt_sha256": self.source_cache_download_receipt.receipt_sha256,
                "selection_plan_sha256": self.source_cache_download_receipt.selection_plan_sha256,
                "cache_verification_receipt_sha256": self.verified_cache.verification_receipt_sha256,
            },
        )


def load_materialization_input_v4(
    *,
    enumeration_receipt_path: Path,
    cache_download_receipt_path: Path,
    source_manifest_path: Path,
    cache_root: Path,
    breakdown_root: Path,
) -> MaterializationInputV4:
    """Load, bind to checked-in A3 authority, and rehash all V4 cache bytes."""

    enumeration = load_upstream_enumeration_artifact_v4(enumeration_receipt_path)
    manifest = load_source_manifest_artifact_v4(source_manifest_path)
    download = load_cache_download_artifact_v4(cache_download_receipt_path)
    context = load_pa_source_execution_context_v4(breakdown_root=breakdown_root)
    if (
        enumeration.execution_binding != context.binding
        or tuple(family.route for family in enumeration.families) != context.routes
        or manifest.execution_binding != context.binding
        or download.execution_binding != context.binding
        or download.source_manifest != manifest
    ):
        raise CorpusMaterializationV4Error("V4 transport differs from checked-in A3 authority")
    verified = _verify_cache(manifest, cache_root)
    return MaterializationInputV4(
        mode=PRODUCTION_MODE,
        upstream_enumeration=enumeration,  # type: ignore[arg-type]
        verified_cache=verified,  # type: ignore[arg-type]
        source_cache_download_receipt=download,  # type: ignore[arg-type]
        cache_root=cache_root,
    )


def _atomic_replace_json(path: Path, payload: Mapping[str, object]) -> str:
    raw = canonical_json_bytes(payload) + b"\n"
    partial = path.with_name(path.name + ".v4.partial")
    assert_no_symlink_ancestors(path)
    assert_no_symlink_ancestors(partial)
    partial.unlink(missing_ok=True)
    with partial.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_RAW_CONTENT_ID_V4 = re.compile(r"^[0-9a-f]{40}$")
_SCREEN_SHARD_KEYS_V4 = {
    "codec_binding_sha256",
    "content_identity_sha256",
    "identity_relative_path",
    "logical_jsonl_bytes",
    "logical_jsonl_sha256",
    "record_count",
    "relative_path",
    "retained_text_bytes",
    "stream",
    "stratum",
    "zstd_bytes",
    "zstd_sha256",
}


def _json_no_duplicates_v4(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CorpusMaterializationV4Error("physical shard JSON repeats a field")
        value[key] = item
    return value


def _framed_ascii_digest_update(digest: Any, value: str) -> None:
    encoded = value.encode("ascii")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _load_screen_shard_manifest_v4(root: Path) -> tuple[str, tuple[dict[str, object], ...]]:
    path = root / "artifacts" / "shard-manifest.json"
    raw, manifest = load_canonical_json_snapshot(path)
    manifest = _exact_mapping(
        manifest,
        {
            "codec_binding_sha256",
            "schema",
            "shards",
            "tokenizer_fit_input_receipt_sha256",
        },
        "physical screen-shard manifest",
    )
    rows = manifest.get("shards")
    if (
        manifest.get("schema") != "weft1_corpus_shard_manifest_v3"
        or manifest.get("codec_binding_sha256")
        != A2_ZSTD_CODEC_BINDING.receipt_sha256
        or not isinstance(rows, list)
        or not rows
    ):
        raise CorpusMaterializationV4Error("physical screen-shard manifest drifted")
    normalized: list[dict[str, object]] = []
    paths: list[str] = []
    for raw_row in rows:
        row = dict(
            _exact_mapping(
                raw_row, _SCREEN_SHARD_KEYS_V4, "physical screen-shard row"
            )
        )
        relative = row.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or PurePosixPath(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts)
            or row.get("stream") not in {"T", "H"}
            or row.get("stratum") not in GTOK_STRATA
            or row.get("codec_binding_sha256")
            != A2_ZSTD_CODEC_BINDING.receipt_sha256
        ):
            raise CorpusMaterializationV4Error("physical screen-shard row drifted")
        identity = JsonlZstdShardIdentityV3(
            relative_path=str(row["identity_relative_path"]),
            record_count=int(row["record_count"]),
            retained_text_bytes=int(row["retained_text_bytes"]),
            logical_jsonl_sha256=str(row["logical_jsonl_sha256"]),
            logical_jsonl_bytes=int(row["logical_jsonl_bytes"]),
            zstd_sha256=str(row["zstd_sha256"]),
            zstd_bytes=int(row["zstd_bytes"]),
            codec_binding_sha256=str(row["codec_binding_sha256"]),
        )
        if (
            relative != f"shards/{identity.relative_path}"
            or row.get("content_identity_sha256")
            != identity.content_identity_sha256
        ):
            raise CorpusMaterializationV4Error(
                "physical screen-shard typed identity drifted"
            )
        paths.append(relative)
        normalized.append(row)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise CorpusMaterializationV4Error(
            "physical screen-shard paths are noncanonical"
        )
    return hashlib.sha256(raw).hexdigest(), tuple(normalized)


def recompute_physical_d6_evidence_v4(
    *, root: Path, sqlite_path: Path
) -> tuple[dict[str, object], str]:
    """Recompute D6 solely from physical T/H zstd JSONL and raw-content IDs.

    The scratch database is disk-bounded and deleted before return.  No V3
    internal ``document_id`` enters this evidence domain.
    """

    try:
        import zstandard
    except ImportError as error:  # pragma: no cover - production runtime pins it
        raise CorpusMaterializationV4Error(
            "physical D6 evidence requires the pinned zstandard runtime"
        ) from error

    root = assert_no_symlink_ancestors(root).resolve(strict=True)
    database_path = assert_no_symlink_ancestors(sqlite_path).resolve(strict=False)
    if database_path.exists() or database_path.is_symlink():
        raise CorpusMaterializationV4Error("physical D6 scratch database must be fresh")
    screen_manifest_sha256, shard_rows = _load_screen_shard_manifest_v4(root)
    inventory = tuple(
        {
            "record_count": int(row["record_count"]),
            "relative_path": str(row["relative_path"]),
            "retained_text_bytes": int(row["retained_text_bytes"]),
            "stream": str(row["stream"]),
            "stratum": str(row["stratum"]),
            "zstd_sha256": str(row["zstd_sha256"]),
        }
        for row in shard_rows
    )
    inventory_sha256 = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
    stream_stats = {
        stream: {
            "count": 0,
            "ids": hashlib.sha256(),
            "payload": hashlib.sha256(),
            "retained": 0,
        }
        for stream in ("T", "H")
    }
    group_stats = {
        (stream, stratum): {
            "count": 0,
            "ids": hashlib.sha256(),
            "retained": 0,
        }
        for stream in ("T", "H")
        for stratum in GTOK_STRATA
    }
    fit_text = hashlib.sha256()
    fit_ids = hashlib.sha256()
    fit_count = 0
    fit_retained = 0
    fit_paths = tuple(
        str(row["relative_path"]) for row in shard_rows if row["stream"] == "T"
    )
    if not fit_paths:
        raise CorpusMaterializationV4Error("physical D6 found no T shards")

    connection = sqlite3.connect(database_path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE records ("
            "raw_content_id TEXT PRIMARY KEY, stream TEXT NOT NULL, "
            "stratum TEXT NOT NULL, manifest_ordinal INTEGER NOT NULL, "
            "record_ordinal INTEGER NOT NULL, text BLOB) WITHOUT ROWID, STRICT"
        )
        for manifest_ordinal, row in enumerate(shard_rows):
            relative = str(row["relative_path"])
            path = root.joinpath(*PurePosixPath(relative).parts)
            assert_no_symlink_ancestors(path)
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != row["zstd_bytes"]
                or _sha256_file(path) != row["zstd_sha256"]
            ):
                raise CorpusMaterializationV4Error(
                    "physical D6 shard identity drifted"
                )
            logical = hashlib.sha256()
            logical_bytes = 0
            retained = 0
            count = 0
            try:
                with path.open("rb") as compressed:
                    with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                        with io.TextIOWrapper(
                            reader, encoding="utf-8", errors="strict", newline=""
                        ) as handle:
                            for record_ordinal, line in enumerate(handle):
                                raw_line = line.encode("utf-8")
                                if not line.endswith("\n"):
                                    raise CorpusMaterializationV4Error(
                                        "physical D6 JSONL framing drifted"
                                    )
                                logical.update(raw_line)
                                logical_bytes += len(raw_line)
                                item = json.loads(
                                    line, object_pairs_hook=_json_no_duplicates_v4
                                )
                                if raw_line != canonical_json_bytes(item) + b"\n":
                                    raise CorpusMaterializationV4Error(
                                        "physical D6 JSONL record is noncanonical"
                                    )
                                item = _exact_mapping(
                                    item,
                                    {"id", "source", "stratum", "text"},
                                    "physical D6 shard record",
                                )
                                raw_id = item.get("id")
                                text = item.get("text")
                                if (
                                    not isinstance(raw_id, str)
                                    or _RAW_CONTENT_ID_V4.fullmatch(raw_id) is None
                                    or not isinstance(text, str)
                                    or item.get("stratum") != row["stratum"]
                                ):
                                    raise CorpusMaterializationV4Error(
                                        "physical D6 shard record drifted"
                                    )
                                payload = text.encode("utf-8", errors="strict")
                                if hashlib.sha1(payload).hexdigest() != raw_id:  # noqa: S324 - A2 physical ID
                                    raise CorpusMaterializationV4Error(
                                        "physical D6 raw-content identity drifted"
                                    )
                                stream = str(row["stream"])
                                stratum = str(row["stratum"])
                                try:
                                    connection.execute(
                                        "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)",
                                        (
                                            raw_id,
                                            stream,
                                            stratum,
                                            manifest_ordinal,
                                            record_ordinal,
                                            payload if stream == "T" else None,
                                        ),
                                    )
                                except sqlite3.IntegrityError as error:
                                    raise CorpusMaterializationV4Error(
                                        "physical D6 found a repeated raw-content ID"
                                    ) from error
                                _framed_ascii_digest_update(
                                    stream_stats[stream]["ids"], raw_id
                                )
                                stream_stats[stream]["payload"].update(
                                    len(payload).to_bytes(8, "big")
                                )
                                stream_stats[stream]["payload"].update(payload)
                                stream_stats[stream]["count"] += 1
                                stream_stats[stream]["retained"] += len(payload)
                                group = group_stats[(stream, stratum)]
                                _framed_ascii_digest_update(group["ids"], raw_id)
                                group["count"] += 1
                                group["retained"] += len(payload)
                                if stream == "T":
                                    fit_text.update(len(payload).to_bytes(8, "big"))
                                    fit_text.update(payload)
                                    _framed_ascii_digest_update(fit_ids, raw_id)
                                    fit_count += 1
                                    fit_retained += len(payload)
                                retained += len(payload)
                                count += 1
            except (OSError, UnicodeError, json.JSONDecodeError, zstandard.ZstdError) as error:
                raise CorpusMaterializationV4Error(
                    "physical D6 could not decode a governed shard"
                ) from error
            if (
                logical.hexdigest() != row["logical_jsonl_sha256"]
                or logical_bytes != row["logical_jsonl_bytes"]
                or retained != row["retained_text_bytes"]
                or count != row["record_count"]
            ):
                raise CorpusMaterializationV4Error(
                    "physical D6 shard accounting drifted"
                )

        training_count = int(stream_stats["T"]["count"])
        training_bytes = int(stream_stats["T"]["retained"])
        if training_count < 2 or fit_count != training_count or fit_retained != training_bytes:
            raise CorpusMaterializationV4Error(
                "physical D6 tokenizer/training coverage drifted"
            )
        multiset = hashlib.sha256()
        for row in connection.execute(
            "SELECT raw_content_id FROM records WHERE stream='T' ORDER BY raw_content_id"
        ):
            _framed_ascii_digest_update(multiset, str(row[0]))
        consumer_orders: list[dict[str, object]] = []
        for training_seed, data_order_seed in GTOK_GOVERNED_DATA_ORDER_SEED_ROWS_V4:
            connection.execute("DROP TABLE IF EXISTS consumer_order")
            connection.execute(
                "CREATE TABLE consumer_order (order_key BLOB NOT NULL, "
                "raw_content_id TEXT PRIMARY KEY) WITHOUT ROWID, STRICT"
            )
            for row in connection.execute(
                "SELECT raw_content_id FROM records WHERE stream='T' ORDER BY raw_content_id"
            ):
                raw_id = str(row[0])
                order_key = hashlib.sha256(
                    b"WEFT-1/gtok-training-order/raw-content-id/v4\x00"
                    + data_order_seed.to_bytes(8, "big")
                    + raw_id.encode("ascii")
                ).digest()
                connection.execute(
                    "INSERT INTO consumer_order VALUES (?, ?)", (order_key, raw_id)
                )
            ordered_ids = hashlib.sha256()
            framed_payload = hashlib.sha256()
            observed_count = 0
            observed_bytes = 0
            for row in connection.execute(
                "SELECT r.raw_content_id, r.text FROM consumer_order AS o "
                "JOIN records AS r ON r.raw_content_id=o.raw_content_id "
                "ORDER BY o.order_key, o.raw_content_id"
            ):
                raw_id = str(row["raw_content_id"])
                payload = bytes(row["text"])
                _framed_ascii_digest_update(ordered_ids, raw_id)
                _framed_ascii_digest_update(framed_payload, raw_id)
                framed_payload.update(len(payload).to_bytes(8, "big"))
                framed_payload.update(payload)
                observed_count += 1
                observed_bytes += len(payload)
            if observed_count != training_count or observed_bytes != training_bytes:
                raise CorpusMaterializationV4Error(
                    "physical D6 consumer ordering changed the T multiset"
                )
            receipt: dict[str, object] = {
                "document_count": training_count,
                "document_multiset_sha256": multiset.hexdigest(),
                "data_order_seed": data_order_seed,
                "framed_payload_sha256": framed_payload.hexdigest(),
                "order_key_domain": "WEFT-1/gtok-training-order/raw-content-id/v4",
                "ordered_raw_content_ids_sha256": ordered_ids.hexdigest(),
                "retained_text_bytes": training_bytes,
                "schema": CONSUMER_ORDER_SCHEMA_V4,
                "training_seed": training_seed,
            }
            receipt["receipt_sha256"] = execution_authority_v4_bound_sha256(
                CONSUMER_ORDER_SCHEMA_V4, receipt
            )
            consumer_orders.append(receipt)
        if len({row["ordered_raw_content_ids_sha256"] for row in consumer_orders}) != len(
            consumer_orders
        ):
            raise CorpusMaterializationV4Error(
                "physical D6 seeds did not produce distinct T orders"
            )
        fit_receipt: dict[str, object] = {
            "allowed_stream": "T",
            "document_count": fit_count,
            "fit_text_stream_sha256": fit_text.hexdigest(),
            "heldout_admissible": False,
            "ordered_raw_content_ids_sha256": fit_ids.hexdigest(),
            "ordered_shard_paths": fit_paths,
            "ordering": "PHYSICAL_SHARD_MANIFEST_THEN_JSONL_RECORD_ORDER",
            "retained_text_bytes": fit_retained,
            "schema": TOKENIZER_FIT_INPUT_SCHEMA_V4,
        }
        fit_receipt["receipt_sha256"] = execution_authority_v4_bound_sha256(
            TOKENIZER_FIT_INPUT_SCHEMA_V4, fit_receipt
        )
        order_by_seed = {
            int(row["training_seed"]): row for row in consumer_orders
        }
        heldout_payload = stream_stats["H"]["payload"].hexdigest()
        consumer_bindings = tuple(
            {
                "heldout_framed_retained_text_sha256": heldout_payload,
                "training_document_multiset_sha256": order_by_seed[training_seed][
                    "document_multiset_sha256"
                ],
                "training_order_receipt_sha256": order_by_seed[training_seed][
                    "receipt_sha256"
                ],
                "data_order_seed": order_by_seed[training_seed]["data_order_seed"],
                "training_ordered_raw_content_ids_sha256": order_by_seed[
                    training_seed
                ]["ordered_raw_content_ids_sha256"],
                "training_seed": training_seed,
                "vocabulary_size": vocabulary_size,
            }
            for vocabulary_size in GTOK_VOCABULARY_ARMS
            for training_seed in GTOK_TRAINING_SEEDS
        )
        stream_identities = tuple(
            {
                "document_count": int(stream_stats[stream]["count"]),
                "framed_retained_text_sha256": stream_stats[stream][
                    "payload"
                ].hexdigest(),
                "ordered_raw_content_ids_sha256": stream_stats[stream][
                    "ids"
                ].hexdigest(),
                "retained_text_bytes": int(stream_stats[stream]["retained"]),
                "stream": stream,
            }
            for stream in ("T", "H")
        )
        split_groups = tuple(
            {
                "document_count": int(group_stats[(stream, stratum)]["count"]),
                "ordered_raw_content_ids_sha256": group_stats[(stream, stratum)][
                    "ids"
                ].hexdigest(),
                "retained_text_bytes": int(
                    group_stats[(stream, stratum)]["retained"]
                ),
                "stratum": stratum,
                "stream": stream,
            }
            for stream in ("T", "H")
            for stratum in GTOK_STRATA
        )
        evidence_core: dict[str, object] = {
            "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
            "consumer_bindings": consumer_bindings,
            "consumer_order_receipts": tuple(consumer_orders),
            "document_overlap_count": 0,
            "gate": "D6",
            "gate_minted": False,
            "physical_shard_inventory": inventory,
            "physical_shard_inventory_sha256": inventory_sha256,
            "repeated_raw_content_id_count": 0,
            "schema": D6_PHYSICAL_EVIDENCE_SCHEMA_V4,
            "screen_shard_manifest_sha256": screen_manifest_sha256,
            "split_groups": split_groups,
            "status": "PHYSICAL_REREAD_PASS_NO_GATE_MINT",
            "stream_identities": stream_identities,
            "tokenizer_fit_input": fit_receipt,
        }
        evidence = dict(evidence_core)
        evidence["evidence_identity_sha256"] = execution_authority_v4_bound_sha256(
            D6_PHYSICAL_EVIDENCE_SCHEMA_V4, evidence_core
        )
        physical = hashlib.sha256(canonical_json_bytes(evidence) + b"\n").hexdigest()
        return evidence, physical
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)


def build_physical_d6_evidence_v4(
    *, root: Path, sqlite_path: Path
) -> tuple[dict[str, object], str]:
    """Recompute then exclusively publish the forward-only physical D6 artifact."""

    evidence, expected_physical = recompute_physical_d6_evidence_v4(
        root=root, sqlite_path=sqlite_path
    )
    path = root.joinpath(*PurePosixPath(D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4).parts)
    if path.exists() or path.is_symlink():
        raise CorpusMaterializationV4Error(
            "physical D6 evidence output must be fresh"
        )
    observed = _atomic_replace_json(path, evidence)
    if observed != expected_physical:
        raise CorpusMaterializationV4Error(
            "physical D6 evidence changed at publication boundary"
        )
    return evidence, observed


def validate_physical_d6_evidence_v4(
    *, root: Path, sqlite_path: Path
) -> tuple[dict[str, object], str]:
    """Independently recompute and compare the governed physical D6 artifact."""

    root = assert_no_symlink_ancestors(root).resolve(strict=True)
    path = root.joinpath(*PurePosixPath(D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4).parts)
    raw, stored = load_canonical_json_snapshot(path)
    recomputed, expected_physical = recompute_physical_d6_evidence_v4(
        root=root, sqlite_path=sqlite_path
    )
    normalized_recomputed = json.loads(canonical_json_bytes(recomputed))
    observed_physical = hashlib.sha256(raw).hexdigest()
    if (
        dict(stored) != normalized_recomputed
        or observed_physical != expected_physical
        or stored.get("schema") != D6_PHYSICAL_EVIDENCE_SCHEMA_V4
    ):
        raise CorpusMaterializationV4Error(
            "stored physical D6 evidence differs from physical T/H shards"
        )
    screen_path = root.joinpath(
        *PurePosixPath(SCREEN_SUBMANIFEST_RELATIVE_PATH_V4).parts
    )
    if screen_path.exists():
        _, screen = load_canonical_json_snapshot(screen_path)
        if (
            screen.get("d6_physical_evidence_path")
            != D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
            or screen.get("d6_physical_evidence_sha256") != observed_physical
            or screen.get("d6_physical_evidence_identity_sha256")
            != stored.get("evidence_identity_sha256")
        ):
            raise CorpusMaterializationV4Error(
                "screen submanifest does not bind physical D6 evidence"
            )
    return normalized_recomputed, observed_physical


def assert_current_physical_d6_identity_v4(
    root: Path, *, expected_physical_sha256: str
) -> dict[str, object]:
    """Rejoin a previously validated D6 identity to current control artifacts.

    This is the cheap consumer-side companion to
    :func:`validate_physical_d6_evidence_v4`.  The producer/source loader must
    first perform that full physical T/H replay and retain its returned file
    SHA.  A later T or H factory can call this assertion without rescanning the
    corpus: it authenticates the same immutable D6 artifact, its authority-
    domain identity, the current physical screen-shard manifest, and the
    screen-submanifest binding.  The individual shard iterator remains
    responsible for consuming/verifying the governed shard bytes.
    """

    if (
        not isinstance(expected_physical_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_physical_sha256) is None
    ):
        raise CorpusMaterializationV4Error(
            "expected physical D6 evidence SHA-256 is malformed"
        )
    resolved = assert_no_symlink_ancestors(root).resolve(strict=True)
    path = resolved.joinpath(
        *PurePosixPath(D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4).parts
    )
    raw, stored = load_canonical_json_snapshot(path)
    observed_physical = hashlib.sha256(raw).hexdigest()
    if observed_physical != expected_physical_sha256:
        raise CorpusMaterializationV4Error(
            "current physical D6 evidence differs from source-loaded identity"
        )
    stored = dict(stored)
    if (
        stored.get("schema") != D6_PHYSICAL_EVIDENCE_SCHEMA_V4
        or stored.get("authority_chain")
        != list(GTOK_EXECUTION_AUTHORITY_CHAIN_V4)
        or stored.get("gate") != "D6"
        or stored.get("gate_minted") is not False
        or stored.get("status") != "PHYSICAL_REREAD_PASS_NO_GATE_MINT"
    ):
        raise CorpusMaterializationV4Error(
            "current physical D6 evidence authority fields drifted"
        )
    evidence_identity = stored.get("evidence_identity_sha256")
    core = dict(stored)
    core.pop("evidence_identity_sha256", None)
    if (
        not isinstance(evidence_identity, str)
        or evidence_identity
        != execution_authority_v4_bound_sha256(
            D6_PHYSICAL_EVIDENCE_SCHEMA_V4, core
        )
    ):
        raise CorpusMaterializationV4Error(
            "current physical D6 evidence identity drifted"
        )
    screen_manifest_sha256, _ = _load_screen_shard_manifest_v4(resolved)
    if screen_manifest_sha256 != stored.get("screen_shard_manifest_sha256"):
        raise CorpusMaterializationV4Error(
            "current screen-shard manifest differs from physical D6 evidence"
        )
    screen_path = resolved.joinpath(
        *PurePosixPath(SCREEN_SUBMANIFEST_RELATIVE_PATH_V4).parts
    )
    if not screen_path.is_file() or screen_path.is_symlink():
        raise CorpusMaterializationV4Error(
            "current V4 screen submanifest is absent"
        )
    _, screen = load_canonical_json_snapshot(screen_path)
    if (
        screen.get("d6_physical_evidence_path")
        != D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
        or screen.get("d6_physical_evidence_sha256") != observed_physical
        or screen.get("d6_physical_evidence_identity_sha256")
        != evidence_identity
    ):
        raise CorpusMaterializationV4Error(
            "screen submanifest does not bind current physical D6 evidence"
        )
    return stored


def _iter_physical_stream_records_v4(
    root: Path, *, stream: str
) -> Iterator[tuple[str, bytes]]:
    """Yield raw-content ID and UTF-8 payload in physical manifest order."""

    if stream not in {"T", "H"}:
        raise ValueError("physical V4 stream must be T or H")
    try:
        import zstandard
    except ImportError as error:  # pragma: no cover
        raise CorpusMaterializationV4Error(
            "physical V4 consumer requires pinned zstandard"
        ) from error
    _, rows = _load_screen_shard_manifest_v4(root)
    for row in rows:
        if row["stream"] != stream:
            continue
        path = root.joinpath(*PurePosixPath(str(row["relative_path"])).parts)
        try:
            with path.open("rb") as compressed:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                    with io.TextIOWrapper(
                        reader, encoding="utf-8", errors="strict", newline=""
                    ) as handle:
                        for line in handle:
                            raw = line.encode("utf-8")
                            item = json.loads(
                                line, object_pairs_hook=_json_no_duplicates_v4
                            )
                            if raw != canonical_json_bytes(item) + b"\n":
                                raise CorpusMaterializationV4Error(
                                    "physical V4 consumer observed noncanonical JSONL"
                                )
                            item = _exact_mapping(
                                item,
                                {"id", "source", "stratum", "text"},
                                "physical V4 consumer record",
                            )
                            raw_id = item.get("id")
                            text = item.get("text")
                            if (
                                not isinstance(raw_id, str)
                                or _RAW_CONTENT_ID_V4.fullmatch(raw_id) is None
                                or not isinstance(text, str)
                                or item.get("stratum") != row["stratum"]
                            ):
                                raise CorpusMaterializationV4Error(
                                    "physical V4 consumer record drifted"
                                )
                            payload = text.encode("utf-8")
                            if hashlib.sha1(payload).hexdigest() != raw_id:  # noqa: S324
                                raise CorpusMaterializationV4Error(
                                    "physical V4 consumer raw ID drifted"
                                )
                            yield raw_id, payload
        except (OSError, UnicodeError, json.JSONDecodeError, zstandard.ZstdError) as error:
            raise CorpusMaterializationV4Error(
                "physical V4 consumer could not decode a shard"
            ) from error


def iter_materialized_tokenizer_fit_texts_v4(root: Path) -> Iterator[str]:
    """Yield the only tokenizer-admissible stream from V4 physical T shards."""

    resolved = assert_no_symlink_ancestors(root).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="weft1-v4-fit-verify-") as raw_work:
        evidence, _ = validate_physical_d6_evidence_v4(
            root=resolved, sqlite_path=Path(raw_work) / "d6.sqlite"
        )
        fit = evidence["tokenizer_fit_input"]
        if not isinstance(fit, Mapping) or fit.get("schema") != TOKENIZER_FIT_INPUT_SCHEMA_V4:
            raise CorpusMaterializationV4Error("physical tokenizer-fit receipt drifted")
        ids = hashlib.sha256()
        texts = hashlib.sha256()
        count = 0
        retained = 0
        for raw_id, payload in _iter_physical_stream_records_v4(resolved, stream="T"):
            _framed_ascii_digest_update(ids, raw_id)
            texts.update(len(payload).to_bytes(8, "big"))
            texts.update(payload)
            count += 1
            retained += len(payload)
            yield payload.decode("utf-8")
        if (
            ids.hexdigest() != fit.get("ordered_raw_content_ids_sha256")
            or texts.hexdigest() != fit.get("fit_text_stream_sha256")
            or count != fit.get("document_count")
            or retained != fit.get("retained_text_bytes")
        ):
            raise CorpusMaterializationV4Error(
                "physical tokenizer-fit consumption differs from its receipt"
            )


def iter_materialized_training_texts_v4(
    root: Path,
    *,
    training_seed: int,
    expected_physical_d6_evidence_sha256: str | None = None,
    expected_consumer_order_receipt: tuple[int, int, str] | None = None,
) -> Iterator[str]:
    """Yield T in the governed V4 data order for one registered seed row."""

    if training_seed not in GTOK_TRAINING_SEEDS:
        raise CorpusMaterializationV4Error("unknown V4 G-TOK training seed")
    if (expected_physical_d6_evidence_sha256 is None) != (
        expected_consumer_order_receipt is None
    ):
        raise CorpusMaterializationV4Error(
            "source-loaded V4 D6 and consumer-order expectations must be paired"
        )
    resolved = assert_no_symlink_ancestors(root).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="weft1-v4-train-order-") as raw_work:
        work = Path(raw_work)
        if expected_physical_d6_evidence_sha256 is None:
            evidence, _ = validate_physical_d6_evidence_v4(
                root=resolved, sqlite_path=work / "verify.sqlite"
            )
        else:
            evidence = assert_current_physical_d6_identity_v4(
                resolved,
                expected_physical_sha256=expected_physical_d6_evidence_sha256,
            )
        receipts = evidence.get("consumer_order_receipts")
        if not isinstance(receipts, list):
            raise CorpusMaterializationV4Error("V4 consumer receipts are absent")
        receipt = next(
            (
                row
                for row in receipts
                if isinstance(row, Mapping) and row.get("training_seed") == training_seed
            ),
            None,
        )
        if receipt is None or receipt.get("schema") != CONSUMER_ORDER_SCHEMA_V4:
            raise CorpusMaterializationV4Error("V4 consumer seed receipt is absent")
        data_order_seed = GTOK_DATA_ORDER_SEED_BY_TRAINING_SEED_V4[training_seed]
        if receipt.get("data_order_seed") != data_order_seed:
            raise CorpusMaterializationV4Error(
                "V4 consumer receipt uses the wrong governed data-order seed"
            )
        if expected_consumer_order_receipt is not None:
            if (
                not isinstance(expected_consumer_order_receipt, tuple)
                or len(expected_consumer_order_receipt) != 3
                or not isinstance(expected_consumer_order_receipt[0], int)
                or isinstance(expected_consumer_order_receipt[0], bool)
                or not isinstance(expected_consumer_order_receipt[1], int)
                or isinstance(expected_consumer_order_receipt[1], bool)
                or not isinstance(expected_consumer_order_receipt[2], str)
                or re.fullmatch(
                    r"[0-9a-f]{64}", expected_consumer_order_receipt[2]
                )
                is None
            ):
                raise CorpusMaterializationV4Error(
                    "source-loaded V4 consumer-order expectation is malformed"
                )
            observed_receipt = (
                int(receipt["training_seed"]),
                int(receipt["data_order_seed"]),
                str(receipt["ordered_raw_content_ids_sha256"]),
            )
            if observed_receipt != expected_consumer_order_receipt:
                raise CorpusMaterializationV4Error(
                    "current V4 consumer order differs from source-loaded receipt"
                )
        database = work / "consumer.sqlite"
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute(
                "CREATE TABLE ordered (order_key BLOB NOT NULL, raw_content_id TEXT "
                "PRIMARY KEY, text BLOB NOT NULL) WITHOUT ROWID, STRICT"
            )
            for raw_id, payload in _iter_physical_stream_records_v4(
                resolved, stream="T"
            ):
                key = hashlib.sha256(
                    b"WEFT-1/gtok-training-order/raw-content-id/v4\x00"
                    + data_order_seed.to_bytes(8, "big")
                    + raw_id.encode("ascii")
                ).digest()
                connection.execute(
                    "INSERT INTO ordered VALUES (?, ?, ?)", (key, raw_id, payload)
                )
            connection.commit()
            ids = hashlib.sha256()
            framed = hashlib.sha256()
            count = 0
            retained = 0
            for row in connection.execute(
                "SELECT raw_content_id, text FROM ordered ORDER BY order_key, raw_content_id"
            ):
                raw_id = str(row["raw_content_id"])
                payload = bytes(row["text"])
                _framed_ascii_digest_update(ids, raw_id)
                _framed_ascii_digest_update(framed, raw_id)
                framed.update(len(payload).to_bytes(8, "big"))
                framed.update(payload)
                count += 1
                retained += len(payload)
                yield payload.decode("utf-8")
            if (
                ids.hexdigest() != receipt.get("ordered_raw_content_ids_sha256")
                or framed.hexdigest() != receipt.get("framed_payload_sha256")
                or count != receipt.get("document_count")
                or retained != receipt.get("retained_text_bytes")
            ):
                raise CorpusMaterializationV4Error(
                    "physical V4 consumer order differs from its receipt"
                )
        finally:
            connection.close()


def _persist_full_corpus_v4(
    result: MaterializationResultV3,
    inputs: MaterializationInputV4,
) -> dict[str, object]:
    """Persist every selected full-pool document before the temporary spool dies."""

    database_path = result.work_root / "materialization.sqlite"
    assert_no_symlink_ancestors(database_path)
    if not database_path.is_file():
        raise CorpusMaterializationV4Error("V4 full-corpus spool database is absent")
    root = result.output_root.resolve(strict=True)
    full_root = root / "full-shards"
    if full_root.exists():
        raise CorpusMaterializationV4Error("V4 full-shard root must be fresh")
    connection = sqlite3.connect(database_path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    shard_rows: list[dict[str, object]] = []
    full_document_digest = hashlib.sha256()
    full_document_count = 0
    full_retained_bytes = 0
    source_document_counts = {source: 0 for source in SOURCE_FAMILIES}
    source_retained_bytes = {source: 0 for source in SOURCE_FAMILIES}
    source_strata = {
        family.route.source_family: family.route.stratum
        for family in inputs.upstream_enumeration.families
    }
    if tuple(source_strata) != SOURCE_FAMILIES or any(
        stratum not in GTOK_STRATA for stratum in source_strata.values()
    ):
        raise CorpusMaterializationV4Error(
            "V4 full-corpus routes do not cover canonical sources/strata"
        )
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE full_shard_locations_v4 ("
            "raw_content_id TEXT PRIMARY KEY, document_id TEXT NOT NULL UNIQUE, "
            "source TEXT NOT NULL, "
            "stratum TEXT NOT NULL, "
            "full_ordinal INTEGER NOT NULL UNIQUE, "
            "full_shard_relative_path TEXT NOT NULL, "
            "shard_record_ordinal INTEGER NOT NULL) WITHOUT ROWID, STRICT"
        )
        pending = 0
        for stratum in GTOK_STRATA:
            for source in SOURCE_FAMILIES:
                if source_strata[source] != stratum:
                    continue
                current = None
                shard_first_ordinal: int | None = None
                shard_last_ordinal: int | None = None
                shard_index = 0

                def close_current() -> None:
                    nonlocal current, shard_first_ordinal, shard_last_ordinal, shard_index
                    if current is None:
                        return
                    identity = production_io._close_shard(current)
                    shard_rows.append(
                        {
                            **asdict(identity),
                            "content_identity_sha256": identity.content_identity_sha256,
                            "first_full_ordinal": shard_first_ordinal,
                            "identity_relative_path": identity.relative_path,
                            "last_full_ordinal": shard_last_ordinal,
                            "relative_path": (
                                f"full-shards/{source}/{identity.relative_path}"
                            ),
                            "source": source,
                            "stream": "FULL",
                            "stratum": stratum,
                        }
                    )
                    shard_index += 1
                    current = None
                    shard_first_ordinal = None
                    shard_last_ordinal = None

                cursor = connection.execute(
                    "SELECT document_id, raw_content_id, source, stratum, "
                    "stable_source_record_id, "
                    "text, retained_bytes, full_ordinal FROM selected_documents "
                    "WHERE stratum = ? AND source = ? ORDER BY full_ordinal",
                    (stratum, source),
                )
                try:
                    for row in cursor:
                        document = StableDocumentV3(
                            source=str(row["source"]),
                            stratum=str(row["stratum"]),
                            stable_source_record_id=str(row["stable_source_record_id"]),
                            text=bytes(row["text"]).decode("utf-8", errors="strict"),
                        )
                        if (
                            document.document_id != row["document_id"]
                            or document.shard_record_id != row["raw_content_id"]
                            or document.retained_byte_count != int(row["retained_bytes"])
                            or document.source != source
                            or document.stratum != stratum
                        ):
                            raise CorpusMaterializationV4Error(
                                "V4 full shard spool document identity drifted"
                            )
                        record = production_io.canonical_jsonl_record_bytes_v3(document)
                        if current is not None and current.logical_bytes + len(record) > (
                            production_io.DEFAULT_SHARD_TARGET_BYTES
                        ):
                            close_current()
                        if current is None:
                            current = production_io._open_shard(
                                full_root / source,
                                stream="FULL",
                                stratum=stratum,
                                index=shard_index,
                            )
                            shard_first_ordinal = int(row["full_ordinal"])
                        assert current is not None
                        shard_record_ordinal = current.record_count
                        current.zstd_handle.write(record)
                        current.logical_sha256.update(record)
                        current.logical_bytes += len(record)
                        current.retained_text_bytes += document.retained_byte_count
                        current.record_count += 1
                        shard_last_ordinal = int(row["full_ordinal"])
                        full_relative_path = f"full-shards/{source}/{current.relative_path}"
                        connection.execute(
                            "INSERT INTO full_shard_locations_v4 "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                document.shard_record_id,
                                document.document_id,
                                source,
                                stratum,
                                int(row["full_ordinal"]),
                                full_relative_path,
                                shard_record_ordinal,
                            ),
                        )
                        encoded_id = document.shard_record_id.encode("ascii")
                        full_document_digest.update(len(encoded_id).to_bytes(8, "big"))
                        full_document_digest.update(encoded_id)
                        full_document_count += 1
                        full_retained_bytes += document.retained_byte_count
                        source_document_counts[source] += 1
                        source_retained_bytes[source] += document.retained_byte_count
                        pending += 1
                        if pending >= 4096:
                            connection.commit()
                            connection.execute("BEGIN IMMEDIATE")
                            pending = 0
                    close_current()
                except BaseException:
                    if current is not None:
                        try:
                            current.zstd_handle.close()
                        except BaseException:
                            pass
                        try:
                            current.raw_handle.close()
                        except BaseException:
                            pass
                        current.partial_path.unlink(missing_ok=True)
                    raise
        connection.commit()
        selected_count = int(
            connection.execute("SELECT count(*) FROM selected_documents").fetchone()[0]
        )
        selected_sources = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT source FROM selected_documents"
            )
        }
        if (
            not shard_rows
            or full_document_count != selected_count
            or not selected_sources
            or not selected_sources.issubset(set(SOURCE_FAMILIES))
            or selected_sources
            != {
                source
                for source in SOURCE_FAMILIES
                if source_document_counts[source] > 0
            }
        ):
            raise CorpusMaterializationV4Error("V4 full shards do not cover the full spool")
        ordered_shards = tuple(shard_rows)
        full_manifest_core = {
            "codec_binding_sha256": A2_ZSTD_CODEC_BINDING.receipt_sha256,
            "document_order": (
                "canonical_stratum_then_canonical_source_then_full_ordinal"
            ),
            "document_count": full_document_count,
            "ordered_raw_content_ids_sha256": full_document_digest.hexdigest(),
            "retained_text_bytes": full_retained_bytes,
            "schema": FULL_SHARD_MANIFEST_SCHEMA_V4,
            "shard_target_uncompressed_jsonl_bytes": (
                production_io.DEFAULT_SHARD_TARGET_BYTES
            ),
            "shard_order": (
                "canonical_stratum_then_canonical_source_then_shard_index"
            ),
            "shards": ordered_shards,
            "sources": tuple(
                {
                    "document_count": source_document_counts[source],
                    "retained_text_bytes": source_retained_bytes[source],
                    "source": source,
                    "stratum": source_strata[source],
                }
                for source in SOURCE_FAMILIES
            ),
        }
        full_manifest_identity = execution_authority_v4_bound_sha256(
            FULL_SHARD_MANIFEST_SCHEMA_V4, full_manifest_core
        )
        full_manifest = {
            **full_manifest_core,
            "manifest_identity_sha256": full_manifest_identity,
        }
        full_manifest_path = root.joinpath(
            *PurePosixPath(FULL_SHARD_MANIFEST_RELATIVE_PATH_V4).parts
        )
        full_manifest_sha = _atomic_replace_json(full_manifest_path, full_manifest)

        d6_evidence, d6_evidence_sha = build_physical_d6_evidence_v4(
            root=root,
            sqlite_path=result.work_root / "d6-physical-v4.sqlite",
        )

        missing = int(
            connection.execute(
                "SELECT count(*) FROM split_documents AS s LEFT JOIN "
                "selected_documents AS d ON d.document_id = s.document_id LEFT JOIN "
                "full_shard_locations_v4 AS f ON f.raw_content_id = d.raw_content_id "
                "WHERE f.document_id IS NULL"
            ).fetchone()[0]
        )
        if missing:
            raise CorpusMaterializationV4Error("V4 screen contains a non-full document")
        groups: list[dict[str, object]] = []
        screen_document_count = 0
        for stream in ("T", "H"):
            for stratum in GTOK_STRATA:
                ids = hashlib.sha256()
                locations = hashlib.sha256()
                count = 0
                retained = 0
                for row in connection.execute(
                    "SELECT d.raw_content_id, d.retained_bytes, f.source, "
                    "f.full_shard_relative_path, f.shard_record_ordinal "
                    "FROM split_documents AS s JOIN selected_documents AS d "
                    "ON d.document_id = s.document_id JOIN full_shard_locations_v4 AS f "
                    "ON f.raw_content_id = d.raw_content_id WHERE s.stream = ? AND "
                    "s.stratum = ? ORDER BY s.stream_ordinal", (stream, stratum)
                ):
                    raw_content_id = str(row["raw_content_id"])
                    encoded = raw_content_id.encode("ascii")
                    ids.update(len(encoded).to_bytes(8, "big"))
                    ids.update(encoded)
                    location = canonical_json_bytes(
                        {
                            "raw_content_id": raw_content_id,
                            "full_shard_relative_path": row["full_shard_relative_path"],
                            "shard_record_ordinal": int(row["shard_record_ordinal"]),
                            "source": row["source"],
                        }
                    )
                    locations.update(len(location).to_bytes(8, "big"))
                    locations.update(location)
                    count += 1
                    retained += int(row["retained_bytes"])
                groups.append(
                    {
                        "document_count": count,
                        "full_location_projection_sha256": locations.hexdigest(),
                        "ordered_raw_content_ids_sha256": ids.hexdigest(),
                        "retained_text_bytes": retained,
                        "stratum": stratum,
                        "stream": stream,
                    }
                )
                screen_document_count += count
        screen_shard_manifest_path = root / "artifacts" / "shard-manifest.json"
        screen_core = {
            "d6_physical_evidence_identity_sha256": d6_evidence[
                "evidence_identity_sha256"
            ],
            "d6_physical_evidence_path": D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4,
            "d6_physical_evidence_sha256": d6_evidence_sha,
            "full_manifest_identity_sha256": full_manifest_identity,
            "full_manifest_path": FULL_SHARD_MANIFEST_RELATIVE_PATH_V4,
            "full_manifest_sha256": full_manifest_sha,
            "groups": tuple(groups),
            "missing_full_document_count": 0,
            "non_screen_full_document_count": full_document_count - screen_document_count,
            "schema": SCREEN_SUBMANIFEST_SCHEMA_V4,
            "screen_document_count": screen_document_count,
            "screen_shard_manifest_path": "artifacts/shard-manifest.json",
            "screen_shard_manifest_sha256": _sha256_file(screen_shard_manifest_path),
        }
        if screen_core["non_screen_full_document_count"] < 0:
            raise CorpusMaterializationV4Error("V4 screen is larger than the full corpus")
        screen_identity = execution_authority_v4_bound_sha256(
            SCREEN_SUBMANIFEST_SCHEMA_V4, screen_core
        )
        screen_manifest = {
            **screen_core,
            "submanifest_identity_sha256": screen_identity,
        }
        screen_path = root.joinpath(
            *PurePosixPath(SCREEN_SUBMANIFEST_RELATIVE_PATH_V4).parts
        )
        screen_sha = _atomic_replace_json(screen_path, screen_manifest)
        return {
            "d6_physical_evidence_identity_sha256": d6_evidence[
                "evidence_identity_sha256"
            ],
            "d6_physical_evidence_path": D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4,
            "d6_physical_evidence_sha256": d6_evidence_sha,
            "document_count": full_document_count,
            "full_manifest_identity_sha256": full_manifest_identity,
            "full_manifest_path": FULL_SHARD_MANIFEST_RELATIVE_PATH_V4,
            "full_manifest_sha256": full_manifest_sha,
            "non_screen_full_document_count": screen_core[
                "non_screen_full_document_count"
            ],
            "retained_text_bytes": full_retained_bytes,
            "screen_submanifest_identity_sha256": screen_identity,
            "screen_submanifest_path": SCREEN_SUBMANIFEST_RELATIVE_PATH_V4,
            "screen_submanifest_sha256": screen_sha,
        }
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def finalize_materialization_output_v4(
    result: MaterializationResultV3,
    inputs: MaterializationInputV4,
) -> MaterializationResultV3:
    """Upgrade one fresh V3-algorithm output into the V4 authority domain."""

    if not isinstance(result, MaterializationResultV3) or result.mode != PRODUCTION_MODE:
        raise TypeError("V4 finalization requires a production V3 algorithm result")
    if not isinstance(inputs, MaterializationInputV4):
        raise TypeError("V4 finalization requires the typed bridge input")
    root = assert_no_symlink_ancestors(result.output_root).resolve(strict=True)
    content_path = root / "content-manifest.json"
    d1_path = root / "d1-ready-manifest.json"
    _, content = load_canonical_json_snapshot(content_path)
    _, core_d1 = load_canonical_json_snapshot(d1_path)
    content_payload = dict(content)
    core_content_identity = content_payload.pop("content_identity_sha256", None)
    expected_core_inventory = [
        {
            "bytes": path.stat().st_size,
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.name not in {"d1-ready-manifest.json", "_INCOMPLETE"}
            ),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]
    expected_core_d1_keys = {
        "content_identity_sha256",
        "d1_ready_identity_sha256",
        "file_inventory",
        "gate_minted",
        "mode",
        "readiness",
        "schema",
        "source_identity_sha256",
    }
    core_d1_payload = dict(core_d1)
    core_d1_identity = core_d1_payload.pop("d1_ready_identity_sha256", None)
    if (
        content.get("schema") != MATERIALIZER_SCHEMA
        or content.get("authority_chain")
        not in (list(GTOK_EXECUTION_AUTHORITY_CHAIN_V3), GTOK_EXECUTION_AUTHORITY_CHAIN_V3)
        or core_content_identity
        != execution_authority_v3_bound_sha256(
            "weft1_corpus_materialized_content_v3", content_payload
        )
        or content.get("source_identity_sha256") != result.source_identity_sha256
        or core_content_identity != result.content_identity_sha256
        or set(core_d1) != expected_core_d1_keys
        or core_d1_identity
        != execution_authority_v3_bound_sha256(
            "weft1_corpus_d1_ready_inventory_v3", core_d1_payload
        )
        or core_d1.get("content_identity_sha256") != core_content_identity
        or core_d1.get("file_inventory") != expected_core_inventory
        or core_d1.get("gate_minted") is not False
        or core_d1.get("mode") != PRODUCTION_MODE
        or core_d1.get("readiness")
        != "AUTHORITATIVE_INPUTS_D1_READY_NO_GATE_MINT"
        or core_d1.get("schema") != "weft1_corpus_d1_ready_manifest_v3"
        or core_d1.get("source_identity_sha256") != result.source_identity_sha256
        or _sha256_file(d1_path) != result.d1_ready_manifest_sha256
    ):
        raise CorpusMaterializationV4Error(
            "frozen V3 core output failed pre-upgrade verification"
        )
    incomplete_path = root / "_INCOMPLETE"
    if incomplete_path.exists():
        raise CorpusMaterializationV4Error(
            "frozen V3 core still carries an incomplete sentinel"
        )
    with incomplete_path.open("xb") as handle:
        handle.write(b"P-A V4 forward finalization incomplete\n")
        handle.flush()
        os.fsync(handle.fileno())
    full_corpus = _persist_full_corpus_v4(result, inputs)

    # Imported at the finalization boundary so merely importing the parser
    # bridge cannot turn release files into materialization inputs.  The helper
    # verifies the 5,746-byte authority and every exact release literal before
    # returning the attribution/claims section.
    from training.weft1_release import (
        RELEASE_AUTHORITY_SHA256,
        release_manifest_section,
    )

    release_section = release_manifest_section()
    release_section_identity = execution_authority_v4_bound_sha256(
        RELEASE_MANIFEST_SECTION_SCHEMA_V4, release_section
    )
    bridge = {
        "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
        "cache_download_receipt_sha256": inputs.source_cache_download_receipt.receipt_sha256,
        "cache_verification_receipt_sha256": inputs.verified_cache.verification_receipt_sha256,
        "core_authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
        "core_content_identity_sha256": core_content_identity,
        "core_d1_ready_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(core_d1) + b"\n"
        ).hexdigest(),
        "effective_route_identity_sha256": inputs.upstream_enumeration.execution_binding.effective_route_identity_sha256,
        "execution_binding_sha256": inputs.upstream_enumeration.execution_binding.receipt_sha256,
        "materializer_algorithm_version": MATERIALIZER_ALGORITHM_VERSION,
        "materializer_core_schema": MATERIALIZER_SCHEMA,
        "d6_physical_evidence_identity_sha256": full_corpus[
            "d6_physical_evidence_identity_sha256"
        ],
        "full_manifest_identity_sha256": full_corpus[
            "full_manifest_identity_sha256"
        ],
        "screen_submanifest_identity_sha256": full_corpus[
            "screen_submanifest_identity_sha256"
        ],
        "release_authority_sha256": RELEASE_AUTHORITY_SHA256,
        "release_manifest_section_identity_sha256": release_section_identity,
        "schema": MATERIALIZATION_BRIDGE_SCHEMA_V4,
        "source_identity_sha256": inputs.source_identity_sha256,
        "upstream_enumeration_receipt_sha256": inputs.upstream_enumeration.receipt_sha256,
    }
    bridge["bridge_identity_sha256"] = execution_authority_v4_bound_sha256(
        MATERIALIZATION_BRIDGE_SCHEMA_V4, bridge
    )
    bridge_path = root.joinpath(*PurePosixPath(BRIDGE_RELATIVE_PATH_V4).parts)
    bridge_path.parent.mkdir(parents=True, exist_ok=False)
    bridge_physical_sha256 = _atomic_replace_json(bridge_path, bridge)

    upgraded = dict(content_payload)
    upgraded.update(
        {
            "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
            "readiness": V4_READINESS,
            "schema": MATERIALIZER_SCHEMA_V4,
            "release": {
                "authority_sha256": RELEASE_AUTHORITY_SHA256,
                "manifest_section": release_section,
                "manifest_section_identity_sha256": release_section_identity,
            },
            "v4_full_corpus": full_corpus,
            "v4_transport_bridge": {
                "bridge_identity_sha256": bridge["bridge_identity_sha256"],
                "path": BRIDGE_RELATIVE_PATH_V4,
                "sha256": bridge_physical_sha256,
            },
        }
    )
    content_identity = execution_authority_v4_bound_sha256(
        MATERIALIZED_CONTENT_SCHEMA_V4, upgraded
    )
    upgraded["content_identity_sha256"] = content_identity
    _atomic_replace_json(content_path, upgraded)

    inventory = tuple(
        {
            "bytes": path.stat().st_size,
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.name not in {"d1-ready-manifest.json", "_INCOMPLETE"}
            ),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    )
    d1_payload = {
        "content_identity_sha256": content_identity,
        "file_inventory": inventory,
        "gate_minted": False,
        "mode": PRODUCTION_MODE,
        "readiness": V4_READINESS,
        "schema": D1_READY_SCHEMA_V4,
        "source_identity_sha256": result.source_identity_sha256,
    }
    d1_payload["d1_ready_identity_sha256"] = execution_authority_v4_bound_sha256(
        D1_READY_IDENTITY_SCHEMA_V4, d1_payload
    )
    d1_physical_sha256 = _atomic_replace_json(d1_path, d1_payload)
    incomplete_path.unlink()
    return MaterializationResultV3(
        mode=result.mode,
        source_identity_sha256=result.source_identity_sha256,
        content_identity_sha256=content_identity,
        d1_ready_manifest_sha256=d1_physical_sha256,
        output_root=result.output_root,
        work_root=result.work_root,
    )


def run_materialization_core_v4(
    *,
    inputs: MaterializationInputV4,
    language_classifier: object,
    output_root: Path,
    work_root: Path,
    global_execution_provenance: Mapping[str, object],
    runtime_build_receipt: Mapping[str, object],
) -> MaterializationResultV3:
    """Run the frozen algorithm once, then forward-finalize its fresh output."""

    result = materialize_corpus_pa_v3(
        inputs=inputs,
        plan=MaterializationPlanV3.production(),
        language_classifier=language_classifier,  # type: ignore[arg-type]
        output_root=output_root,
        work_root=work_root,
        global_execution_provenance=global_execution_provenance,
        runtime_build_receipt=runtime_build_receipt,
    )
    return finalize_materialization_output_v4(result, inputs)


def run_production_materialization_worker_v4(
    *,
    enumeration_receipt_path: Path,
    cache_download_receipt_path: Path,
    source_manifest_path: Path,
    cache_root: Path,
    fasttext_model_path: Path,
    breakdown_root: Path,
    execution_provenance_path: Path,
    runtime_build_receipt_path: Path,
) -> str:
    """Concrete offline V4 worker under the existing parent replay guard.

    Runtime attestation precedes every receipt/cache load and model open, just
    as in the frozen V3 worker.  The two private V3 helpers used here are
    capabilities of the unchanged algorithm implementation; no receipt is
    synthesized outside those existing checks.
    """

    paths = (
        enumeration_receipt_path,
        cache_download_receipt_path,
        source_manifest_path,
        cache_root,
        fasttext_model_path,
        breakdown_root,
        execution_provenance_path,
        runtime_build_receipt_path,
    )
    if any(not isinstance(path, Path) for path in paths):
        raise TypeError("production V4 worker inputs must be pathlib.Path values")
    for path in paths:
        assert_no_symlink_ancestors(path)
    if (
        os.environ.get("WEFT1_NETWORK_DISABLED") != "1"
        or os.environ.get("WEFT1_NETWORK_GUARD_ACTIVE") != "1"
    ):
        raise CorpusMaterializationV4Error(
            "production V4 worker requires parent offline execution"
        )
    raw_output_root = os.environ.get("WEFT1_REPLAY_OUTPUT_ROOT")
    if not raw_output_root:
        raise CorpusMaterializationV4Error("production V4 worker lacks an assigned root")
    output_root = Path(raw_output_root)
    assert_no_symlink_ancestors(output_root)
    if output_root.exists():
        raise CorpusMaterializationV4Error("production V4 worker root must be fresh")

    from training import weft1_corpus_materialize_a2 as core_v3
    from training.weft1_corpus_pa import FastTextLanguageIdAdapterV3, attest_runtime_v3

    runtime_attestation = attest_runtime_v3()
    try:
        global_execution_provenance = core_v3.validate_global_execution_provenance_v3(
            core_v3.load_canonical_json_object(execution_provenance_path)
        )
    except Exception as error:
        raise CorpusMaterializationV4Error(
            f"production V4 execution provenance failed validation: {error}"
        ) from error
    if (
        global_execution_provenance.get("environment_identity_sha256")
        != runtime_attestation.environment_identity_sha256
        or global_execution_provenance.get("environment_payload")
        != json.loads(
            canonical_json_bytes(runtime_attestation.environment_payload)
        )
        or global_execution_provenance.get("python_executable_sha256")
        != runtime_attestation.executable_sha256
        or global_execution_provenance.get("dependency_lock_sha256")
        != runtime_attestation.dependency_lock_sha256
    ):
        raise CorpusMaterializationV4Error(
            "production V4 provenance differs from child runtime attestation"
        )
    runtime_build_receipt = core_v3._validated_runtime_build_receipt_v1(
        core_v3.load_canonical_json_object(runtime_build_receipt_path),
        global_execution_provenance=global_execution_provenance,
    )
    inputs = load_materialization_input_v4(
        enumeration_receipt_path=enumeration_receipt_path,
        cache_download_receipt_path=cache_download_receipt_path,
        source_manifest_path=source_manifest_path,
        cache_root=cache_root,
        breakdown_root=breakdown_root,
    )
    classifier = FastTextLanguageIdAdapterV3(fasttext_model_path)
    output_parent = assert_no_symlink_ancestors(output_root.parent).resolve(strict=True)
    raw_local_work_parent = os.environ.get("WEFT1_REPLAY_LOCAL_WORK_PARENT")
    if not raw_local_work_parent:
        raise CorpusMaterializationV4Error(
            "production V4 worker lacks an explicit local work parent"
        )
    local_work_parent = assert_no_symlink_ancestors(
        Path(raw_local_work_parent)
    ).resolve(strict=True)
    if not local_work_parent.is_dir() or (
        local_work_parent == output_parent
        or local_work_parent in output_parent.parents
        or output_parent in local_work_parent.parents
    ):
        raise CorpusMaterializationV4Error(
            "production V4 local work and durable output parents must be disjoint"
        )
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}-work-", dir=local_work_parent
    ) as raw_temporary:
        result = run_materialization_core_v4(
            inputs=inputs,
            language_classifier=classifier,
            output_root=output_root,
            work_root=Path(raw_temporary) / "spool",
            global_execution_provenance=global_execution_provenance,
            runtime_build_receipt=runtime_build_receipt,
        )
        return core_v3._write_production_replay_child_receipt_v3(
            result,
            runtime_environment_identity_sha256=(
                runtime_attestation.environment_identity_sha256
            ),
            sentinel=core_v3._PRODUCTION_WORKER_RECEIPT_SENTINEL,
        )


__all__ = [
    "BRIDGE_RELATIVE_PATH_V4",
    "CorpusMaterializationV4Error",
    "CONSUMER_ORDER_SCHEMA_V4",
    "D1_READY_SCHEMA_V4",
    "D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4",
    "D6_PHYSICAL_EVIDENCE_SCHEMA_V4",
    "FULL_SHARD_MANIFEST_RELATIVE_PATH_V4",
    "FULL_SHARD_MANIFEST_SCHEMA_V4",
    "GTOK_DATA_ORDER_SEED_BY_TRAINING_SEED_V4",
    "GTOK_GOVERNED_DATA_ORDER_SEED_ROWS_V4",
    "MATERIALIZER_SCHEMA_V4",
    "SCREEN_SUBMANIFEST_RELATIVE_PATH_V4",
    "SCREEN_SUBMANIFEST_SCHEMA_V4",
    "TOKENIZER_FIT_INPUT_SCHEMA_V4",
    "MaterializationInputV4",
    "VerifiedCacheAdapterV4",
    "assert_current_physical_d6_identity_v4",
    "finalize_materialization_output_v4",
    "build_physical_d6_evidence_v4",
    "load_cache_download_artifact_v4",
    "load_materialization_input_v4",
    "load_source_manifest_artifact_v4",
    "load_upstream_enumeration_artifact_v4",
    "iter_materialized_tokenizer_fit_texts_v4",
    "iter_materialized_training_texts_v4",
    "run_materialization_core_v4",
    "run_production_materialization_worker_v4",
    "recompute_physical_d6_evidence_v4",
    "validate_physical_d6_evidence_v4",
]
