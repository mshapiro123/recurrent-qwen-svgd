"""Forward-only A3/V4 source preparation for WEFT-1 corpus P-A.

The A1/A2/V3 source receipts are banked evidence.  This module does not
reinterpret or overwrite them.  It consumes the resolved Amendment-A3 route
set and its single combined path-breakdown artifact, then mints a separate V4
enumeration, selection, cache-manifest, and download-receipt chain.

Only the two A3 route changes are special here:

* Dolma is restricted to the observer-proven ``0019`` top-quality groups.
* FineWeb-Edu includes every Parquet file in exactly the observer-proven 110
  ``CC-MAIN-*`` groups, including the newer numeric file names.

Network clients are instantiated only by ``prepare_pa_sources_online_v4``.
The injected enumerator is permanently branded non-authoritative, which keeps
unit tests offline and prevents fixtures from minting production evidence.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Sequence

from training.weft1_corpus_a2 import A2_CAMPAIGN_ROOT_SEED
from training.weft1_corpus_a3 import (
    A3_AUTHORITY_SHA256,
    A3_CAMPAIGN_ROOT_SEED,
    A3_EFFECTIVE_ROUTE_OVERLAY_SHA256,
    A3BreakdownPending,
    A3EffectiveRouteResolution,
    EffectiveSourceRouteA3,
    execution_authority_v4_bound_sha256,
    load_effective_route_overlay_a3,
    resolve_effective_routes_a3,
)
from training.weft1_corpus_breakdown_a3 import (
    DOLMA_SELECTED_CLASSIFICATION,
    FINEWEB_SELECTED_CLASSIFICATION,
    PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3,
    PRODUCTION_OBSERVATION_MODE_A3,
    UpstreamPathBreakdownReceiptA3,
    load_upstream_path_breakdown_snapshot_a3,
    observe_hf_tree_files_a3,
    replay_upstream_path_breakdown_a3,
)
from training.weft1_corpus_enumeration_a2 import (
    ExternalLocatorListingV3,
    UpstreamAssetV3,
    read_pinned_external_locator_listing_v3,
)
from training.weft1_corpus_fetch_a2 import (
    ExternalResourceCacheV3,
    PinnedHuggingFaceAssetOpenerV3,
)
from training.weft1_corpus_sources_a2 import SourceCacheAssetV3, asset_order_digest_v3
from training.weft1_corpus_semantic_evidence_a3 import (
    load_semantic_evidence_snapshot_a3,
    semantic_evidence_relative_path_from_breakdown_a3,
    verify_semantic_evidence_breakdown_binding_a3,
)
from training.weft1_gtok_a1_contract import (
    SOURCE_FAMILIES,
    load_source_route_manifest,
)
from training.weft1_gtok_contract import canonical_json_bytes, canonical_sha256
from training.weft1_strict_io import (
    assert_no_symlink_ancestors,
    load_canonical_json_snapshot,
)


AUTHORITATIVE_MODE = "AUTHORITATIVE"
FIXTURE_MODE = "NONAUTHORITATIVE_FIXTURE"
ENUMERATION_MODES = (AUTHORITATIVE_MODE, FIXTURE_MODE)

EXECUTION_BINDING_SCHEMA_V4 = "weft1_pa_source_execution_binding_v4"
UPSTREAM_ENUMERATION_SCHEMA_V4 = "weft1_upstream_enumeration_receipt_v4"
UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V4 = (
    "weft1_upstream_enumeration_receipt_artifact_v4"
)
PREFIX_SELECTION_SCHEMA_V4 = "weft1_pa_source_prefix_selection_v4"
PREFIX_SELECTION_ARTIFACT_SCHEMA_V4 = (
    "weft1_pa_source_prefix_selection_artifact_v4"
)
DOWNLOAD_PLAN_SCHEMA_V4 = "weft1_source_asset_download_plan_v4"
SOURCE_CACHE_MANIFEST_SCHEMA_V4 = "weft1_local_source_cache_manifest_v4"
VERIFIED_CACHE_RECEIPT_SCHEMA_V4 = "weft1_verified_local_source_cache_v4"
DOWNLOAD_RECEIPT_SCHEMA_V4 = "weft1_source_cache_download_receipt_v4"
DOWNLOAD_RECEIPT_ARTIFACT_SCHEMA_V4 = (
    "weft1_source_cache_download_receipt_artifact_v4"
)
A3_REPLAY_ATTESTATION_SCHEMA_V4 = "weft1_a3_live_replay_attestation_v4"
A3_REPLAY_ATTESTATION_ARTIFACT_SCHEMA_V4 = (
    "weft1_a3_live_replay_attestation_artifact_v4"
)

ENUMERATION_ARTIFACT_NAME_V4 = "upstream-enumeration-v4.json"
SELECTION_ARTIFACT_NAME_V4 = "source-prefix-selection-v4.json"
SOURCE_MANIFEST_NAME_V4 = "source-cache-manifest-v4.json"
DOWNLOAD_ARTIFACT_NAME_V4 = "source-cache-download-receipt-v4.json"
EXTERNAL_TRANSPORT_ARTIFACT_NAME_V4 = "external-transport-receipt-v4.json"

SOURCE_PREP_IMPLEMENTATION_REPO_PATHS_V4 = tuple(
    sorted(
        (
            "docs/STRATEGY_CORPUS_GTOK_AMENDMENT_A3_20260829.md",
            "scripts/attest_weft1_corpus_a3_replay.py",
            "scripts/run_weft1_corpus_a3_observer.py",
            "scripts/run_weft1_corpus_fetch_a3.py",
            "training/__init__.py",
            "training/weft1_corpus_a2.py",
            "training/weft1_corpus_a3.py",
            "training/weft1_corpus_breakdown_a3.py",
            "training/weft1_corpus_enumeration_a2.py",
            "training/weft1_corpus_fetch_a2.py",
            "training/weft1_corpus_fetch_a3.py",
            "training/weft1_corpus_source_io_a2.py",
            "training/weft1_corpus_sources_a2.py",
            "training/weft1_corpus_semantic_evidence_a3.py",
            "training/weft1_gtok_a1_contract.py",
            "training/weft1_gtok_contract.py",
            "training/weft1_seed.py",
            "training/weft1_strict_io.py",
        ),
        key=lambda item: item.encode("utf-8"),
    )
)

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINEWEB_GROUP = re.compile(r"^CC-MAIN-[0-9]{4}-[0-9]{2}$")
_FINEWEB_LOCATOR = re.compile(
    r"^data/(?P<group>CC-MAIN-[0-9]{4}-[0-9]{2})/"
    r"(?P<name>[^/]+[.]parquet)$"
)
_DOLMA_LOCATOR = re.compile(
    r"^data/(?P<group>common_crawl-[^/]+-(?P<bucket>[0-9]{4}))/"
    r"[^/]+[.]jsonl[.]zst$"
)
_HUGGINGFACE_HUB_DISTRIBUTION = "huggingface-hub"
_HUGGINGFACE_HUB_VERSION = "1.24.0"
_FIXTURE_ENUMERATOR_BINDING_SHA256 = hashlib.sha256(
    b"weft1:nonauthoritative-injected-enumerator:v4"
).hexdigest()
_AUTHORITATIVE_PREP_CAPABILITY = object()
_DIRECT_AVAILABLE_BYTE_BASES = (
    "pinned repository compressed asset bytes",
    "pinned repository parquet bytes",
)
_CONTAINER_SUFFIXES = {
    "dolma_web": ".jsonl.zst",
    "wikipedia_wikibooks": ".json.gz",
    "stackedu": ".jsonl.zst",
    "finemath_3plus": ".parquet",
    "arxiv": ".jsonl.zst",
    "olmocr": ".jsonl.zst",
    "fineweb_edu": ".parquet",
}


class SourceFetchV4Error(RuntimeError):
    """An A3 authority, route, selection, or transport check failed closed."""


def _require_nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive exact integer")
    return value


def _canonical_path(value: str) -> str:
    _require_nonempty(value, "repository path")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("repository path must be canonical relative POSIX")
    return value


def _glob_regex(pattern: str) -> re.Pattern[str]:
    _require_nonempty(pattern, "asset selector")
    if any(token in pattern for token in ("?", "[", "]", "**")):
        raise ValueError("effective selector uses an unsupported glob operator")
    return re.compile("^" + re.escape(pattern).replace(r"\*", "[^/]*") + "$")


@dataclass(frozen=True)
class PAExecutionBindingV4:
    """The complete A3 authority edge repeated in every V4 receipt."""

    authority_sha256: str
    effective_route_identity_sha256: str
    breakdown_artifact_physical_sha256: str
    breakdown_artifact_receipt_sha256: str
    family_projection_sha256s: tuple[tuple[str, str], ...]
    campaign_root_seed: int

    def __post_init__(self) -> None:
        if self.authority_sha256 != A3_AUTHORITY_SHA256:
            raise ValueError("V4 execution binding uses a foreign A3 authority")
        for value, name in (
            (self.effective_route_identity_sha256, "effective route identity"),
            (self.breakdown_artifact_physical_sha256, "breakdown physical SHA"),
            (self.breakdown_artifact_receipt_sha256, "breakdown receipt SHA"),
        ):
            _require_sha256(value, name)
        if tuple(family for family, _ in self.family_projection_sha256s) != (
            "dolma_web",
            "fineweb_edu",
        ):
            raise ValueError("V4 execution binding lacks both A3 family projections")
        for _, value in self.family_projection_sha256s:
            _require_sha256(value, "family projection SHA")
        if (
            type(self.campaign_root_seed) is not int
            or self.campaign_root_seed != A3_CAMPAIGN_ROOT_SEED
            or self.campaign_root_seed != A2_CAMPAIGN_ROOT_SEED
        ):
            raise ValueError("V4 execution must preserve the exact A2 campaign seed")

    @classmethod
    def from_resolution(
        cls, resolution: A3EffectiveRouteResolution
    ) -> "PAExecutionBindingV4":
        if not isinstance(resolution, A3EffectiveRouteResolution):
            raise TypeError("V4 execution requires a typed A3 route resolution")
        return cls(
            authority_sha256=A3_AUTHORITY_SHA256,
            effective_route_identity_sha256=(
                resolution.effective_route_identity_sha256
            ),
            breakdown_artifact_physical_sha256=(
                resolution.breakdown_artifact_physical_sha256
            ),
            breakdown_artifact_receipt_sha256=(
                resolution.breakdown_artifact_receipt_sha256
            ),
            family_projection_sha256s=resolution.family_projection_sha256s,
            campaign_root_seed=A3_CAMPAIGN_ROOT_SEED,
        )

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            EXECUTION_BINDING_SCHEMA_V4, self
        )


@dataclass(frozen=True)
class PASourceExecutionContextV4:
    """Resolved routes plus the exact observer group universe they depend on."""

    resolution: A3EffectiveRouteResolution
    binding: PAExecutionBindingV4
    fineweb_cc_dump_ids: tuple[str, ...]
    dolma_top_bucket_group_ids: tuple[str, ...]
    overlay_physical_sha256: str | None = None
    overlay_identity_sha256: str | None = None
    semantic_evidence_artifact_physical_sha256: str | None = None
    semantic_evidence_artifact_receipt_sha256: str | None = None
    semantic_evidence_family_receipt_sha256s: tuple[tuple[str, str], ...] = ()
    path_breakdown: UpstreamPathBreakdownReceiptA3 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resolution, A3EffectiveRouteResolution):
            raise TypeError("V4 context requires a typed route resolution")
        if self.binding != PAExecutionBindingV4.from_resolution(self.resolution):
            raise ValueError("V4 context binding differs from its route resolution")
        fineweb = tuple(
            sorted(self.fineweb_cc_dump_ids, key=lambda item: item.encode("utf-8"))
        )
        if (
            self.fineweb_cc_dump_ids != fineweb
            or len(fineweb) != 110
            or len(set(fineweb)) != 110
            or any(_FINEWEB_GROUP.fullmatch(item) is None for item in fineweb)
        ):
            raise ValueError("V4 context requires exactly 110 canonical FineWeb dumps")
        dolma = tuple(
            sorted(
                self.dolma_top_bucket_group_ids,
                key=lambda item: item.encode("utf-8"),
            )
        )
        if (
            self.dolma_top_bucket_group_ids != dolma
            or not dolma
            or len(set(dolma)) != len(dolma)
            or any(
                not item.startswith("common_crawl-") or not item.endswith("-0019")
                for item in dolma
            )
        ):
            raise ValueError("V4 context requires observer-proven Dolma 0019 groups")
        governed = (
            self.overlay_physical_sha256,
            self.overlay_identity_sha256,
            self.semantic_evidence_artifact_physical_sha256,
            self.semantic_evidence_artifact_receipt_sha256,
            self.path_breakdown,
        )
        if any(item is not None for item in governed) or self.semantic_evidence_family_receipt_sha256s:
            if any(item is None for item in governed):
                raise ValueError("V4 context has a partial governed replay binding")
            _require_sha256(self.overlay_physical_sha256, "overlay physical SHA")
            _require_sha256(self.overlay_identity_sha256, "overlay identity")
            _require_sha256(
                self.semantic_evidence_artifact_physical_sha256,
                "semantic-evidence artifact physical SHA",
            )
            _require_sha256(
                self.semantic_evidence_artifact_receipt_sha256,
                "semantic-evidence artifact typed receipt",
            )
            if not isinstance(self.path_breakdown, UpstreamPathBreakdownReceiptA3):
                raise TypeError("V4 governed context requires a typed path breakdown")
            if (
                self.path_breakdown.observation_mode
                != PRODUCTION_OBSERVATION_MODE_A3
                or self.path_breakdown.observation_client_identity
                != PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3
            ):
                raise ValueError(
                    "V4 execution rejects non-authoritative path observation"
                )
            if self.path_breakdown.receipt_sha256 != self.binding.breakdown_artifact_receipt_sha256:
                raise ValueError("V4 context breakdown receipt drifted")
            expected_evidence = tuple(
                (family.source_family, family.semantic_evidence.receipt_sha256)
                for family in self.path_breakdown.families
            )
            if self.semantic_evidence_family_receipt_sha256s != expected_evidence:
                raise ValueError("V4 context semantic-evidence receipts drifted")
            routes = {route.source_family: route for route in self.routes}
            for family in self.path_breakdown.families:
                route = routes[family.source_family]
                if (
                    family.repository != route.repository
                    or family.revision != route.revision
                    or family.selected_asset_count != route.asset_count
                    or family.selected_upstream_bytes != route.available_bytes
                    or route.breakdown_artifact_receipt_sha256
                    != self.path_breakdown.receipt_sha256
                ):
                    raise ValueError(
                        "V4 context route differs from its observer family"
                    )

    @property
    def routes(self) -> tuple[EffectiveSourceRouteA3, ...]:
        return self.resolution.routes

    @property
    def binding_sha256(self) -> str:
        return self.binding.receipt_sha256


def _breakdown_path_from_overlay(overlay: object, root: Path) -> Path:
    rows = getattr(overlay, "overlay_rows", ())
    bindings = tuple(
        row.breakdown_artifact
        for row in rows
        if row.source_family in {"dolma_web", "fineweb_edu"}
    )
    if len(bindings) != 2 or bindings[0] is None or bindings[0] != bindings[1]:
        raise SourceFetchV4Error("A3 overlay lacks one shared breakdown binding")
    binding = bindings[0]
    if not binding.is_bound or binding.relative_path is None:
        raise A3BreakdownPending("A3 execution breakdown is not yet bound")
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise SourceFetchV4Error("breakdown root must be a real directory")
    lexical = resolved_root.joinpath(*PurePosixPath(binding.relative_path).parts)
    assert_no_symlink_ancestors(lexical)
    resolved = lexical.resolve(strict=True)
    if resolved_root not in resolved.parents or not resolved.is_file():
        raise SourceFetchV4Error("A3 breakdown path escapes its governed root")
    return resolved


def load_pa_source_execution_context_v4(
    *, breakdown_root: Path
) -> PASourceExecutionContextV4:
    """Load resolved A3 authority; pending or inconsistent overlays fail closed."""

    if not isinstance(breakdown_root, Path):
        raise TypeError("breakdown_root must be a pathlib.Path")
    overlay = load_effective_route_overlay_a3()
    resolution = resolve_effective_routes_a3(
        overlay,
        breakdown_root=breakdown_root,
    )
    breakdown_path = _breakdown_path_from_overlay(overlay, breakdown_root)
    unused_raw, breakdown = load_upstream_path_breakdown_snapshot_a3(
        breakdown_path,
        expected_receipt_sha256=resolution.breakdown_artifact_receipt_sha256,
    )
    del unused_raw
    if not isinstance(breakdown, UpstreamPathBreakdownReceiptA3):
        raise TypeError("A3 breakdown loader returned an untyped receipt")
    evidence_relative = semantic_evidence_relative_path_from_breakdown_a3(
        breakdown
    )
    evidence_lexical = breakdown_root.resolve(strict=True).joinpath(
        *PurePosixPath(evidence_relative).parts
    )
    assert_no_symlink_ancestors(evidence_lexical)
    evidence_path = evidence_lexical.resolve(strict=True)
    governed_root = breakdown_root.resolve(strict=True)
    if governed_root not in evidence_path.parents or not evidence_path.is_file():
        raise SourceFetchV4Error(
            "A3 semantic-evidence artifact escapes its governed root"
        )
    unused_evidence_raw, evidence_payload, evidence_identity = (
        load_semantic_evidence_snapshot_a3(evidence_path)
    )
    del unused_evidence_raw
    verify_semantic_evidence_breakdown_binding_a3(
        evidence_payload,
        evidence_identity,
        breakdown,
    )
    # ``resolve_effective_routes_a3`` has already projected both families from
    # these exact typed bytes and compared their V4 receipts to the overlay.
    # This second load exposes the configured group IDs; its expected receipt
    # parameter prevents substituting another observer artifact between steps.
    dolma = next(
        family for family in breakdown.families if family.source_family == "dolma_web"
    )
    fineweb = next(
        family for family in breakdown.families if family.source_family == "fineweb_edu"
    )
    fineweb_selected = tuple(
        sorted(
            (
                group.group_id
                for group in fineweb.groups
                if group.classification == FINEWEB_SELECTED_CLASSIFICATION
                and group.selected
            ),
            key=lambda item: item.encode("utf-8"),
        )
    )
    if fineweb_selected != fineweb.configured_group_ids:
        raise SourceFetchV4Error(
            "A3 FineWeb selected groups differ from the configured 110 dumps"
        )
    return PASourceExecutionContextV4(
        resolution=resolution,
        binding=PAExecutionBindingV4.from_resolution(resolution),
        fineweb_cc_dump_ids=fineweb.configured_group_ids,
        dolma_top_bucket_group_ids=tuple(
            sorted(
                (
                    group.group_id
                    for group in dolma.groups
                    if group.classification == DOLMA_SELECTED_CLASSIFICATION
                    and group.selected
                ),
                key=lambda item: item.encode("utf-8"),
            )
        ),
        overlay_physical_sha256=A3_EFFECTIVE_ROUTE_OVERLAY_SHA256,
        overlay_identity_sha256=overlay.overlay_identity_sha256,
        semantic_evidence_artifact_physical_sha256=(
            evidence_identity.physical_sha256
        ),
        semantic_evidence_artifact_receipt_sha256=(
            evidence_identity.receipt_sha256
        ),
        semantic_evidence_family_receipt_sha256s=tuple(
            (family.source_family, family.semantic_evidence.receipt_sha256)
            for family in breakdown.families
        ),
        path_breakdown=breakdown,
    )


def locator_matches_effective_route_v4(
    context: PASourceExecutionContextV4,
    route: EffectiveSourceRouteA3,
    locator: str,
) -> bool:
    """Apply A3 semantic selectors without widening either changed family."""

    if not isinstance(context, PASourceExecutionContextV4):
        raise TypeError("locator matcher requires a V4 execution context")
    if not isinstance(route, EffectiveSourceRouteA3):
        raise TypeError("locator matcher requires an effective A3 route")
    _require_nonempty(locator, "asset locator")
    by_family = {item.source_family: item for item in context.routes}
    if route != by_family.get(route.source_family):
        raise ValueError("locator matcher received a route outside its resolution")
    if route.source_family == "fineweb_edu":
        match = _FINEWEB_LOCATOR.fullmatch(locator)
        return (
            match is not None
            and match.group("group") in frozenset(context.fineweb_cc_dump_ids)
        )
    if route.source_family == "dolma_web":
        match = _DOLMA_LOCATOR.fullmatch(locator)
        return (
            match is not None
            and match.group("bucket") == "0019"
            and match.group("group")
            in frozenset(context.dolma_top_bucket_group_ids)
        )
    selector = route.asset_selector
    if route.source_family == "wikipedia_wikibooks":
        parts = selector.split(" -> ")
        if len(parts) != 2 or parts[0] != "urls/v1_7.txt":
            raise ValueError("Wikipedia effective selector syntax drifted")
        selector = parts[1]
    return _glob_regex(selector).fullmatch(locator) is not None


def _field(value: object, name: str, *, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(frozen=True)
class UpstreamAssetV4:
    source_family: str
    repository: str
    config: str
    revision: str
    split: str
    asset_locator: str
    upstream_bytes: int
    blob_identity_kind: str
    blob_identity: str
    content_sha256: str | None
    effective_route_receipt_sha256: str
    execution_binding_sha256: str

    def __post_init__(self) -> None:
        if self.source_family not in SOURCE_FAMILIES:
            raise ValueError("unknown source family")
        for name in ("repository", "config", "revision", "split", "asset_locator"):
            _require_nonempty(getattr(self, name), name)
        if _SHA1.fullmatch(self.revision) is None:
            raise ValueError("upstream revision must be an exact commit SHA")
        _require_positive_int(self.upstream_bytes, "upstream_bytes")
        if self.blob_identity_kind == "git_sha1":
            if _SHA1.fullmatch(self.blob_identity) is None:
                raise ValueError("git_sha1 identity must be a lowercase SHA-1")
        elif self.blob_identity_kind in {"git_sha256", "content_sha256"}:
            _require_sha256(self.blob_identity, "blob identity")
        else:
            raise ValueError("unsupported upstream blob identity kind")
        if self.content_sha256 is not None:
            _require_sha256(self.content_sha256, "content SHA-256")
        if (
            self.blob_identity_kind == "content_sha256"
            and self.content_sha256 != self.blob_identity
        ):
            raise ValueError("external content identity and SHA disagree")
        _require_sha256(self.effective_route_receipt_sha256, "effective route receipt")
        _require_sha256(self.execution_binding_sha256, "execution binding")

    @property
    def asset_identity_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            "weft1_upstream_asset_identity_v4", self
        )


def _canonical_upstream_order_v4(
    assets: Sequence[UpstreamAssetV4],
) -> tuple[UpstreamAssetV4, ...]:
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
        raise TypeError("upstream assets must be a typed sequence")
    if any(not isinstance(asset, UpstreamAssetV4) for asset in assets):
        raise TypeError("upstream assets contain an untyped value")
    locators = tuple(asset.asset_locator for asset in assets)
    if len(locators) != len(set(locators)):
        raise ValueError("upstream enumeration repeats an asset locator")
    return tuple(
        sorted(
            assets,
            key=lambda asset: (
                asset_order_digest_v3(asset.asset_locator),
                asset.asset_locator.encode("utf-8"),
                asset.blob_identity_kind,
                asset.blob_identity,
            ),
        )
    )


@dataclass(frozen=True)
class FamilyEnumerationV4:
    route: EffectiveSourceRouteA3
    execution_binding_sha256: str
    observed_available_bytes: int
    assets: tuple[UpstreamAssetV4, ...]
    external_locator_manifest_sha256: str | None = None
    external_locator_manifest_bytes: int | None = None
    external_locator_listing_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route, EffectiveSourceRouteA3):
            raise TypeError("family enumeration requires an effective A3 route")
        _require_sha256(self.execution_binding_sha256, "execution binding")
        _require_positive_int(self.observed_available_bytes, "observed available bytes")
        if not self.assets or self.assets != _canonical_upstream_order_v4(self.assets):
            raise ValueError("family assets are absent or noncanonical")
        for asset in self.assets:
            if (
                asset.source_family,
                asset.repository,
                asset.config,
                asset.revision,
                asset.split,
                asset.effective_route_receipt_sha256,
                asset.execution_binding_sha256,
            ) != (
                self.route.source_family,
                self.route.repository,
                self.route.config,
                self.route.revision,
                self.route.split,
                self.route.receipt_sha256,
                self.execution_binding_sha256,
            ):
                raise ValueError("enumerated asset route or authority edge drifted")
        external = (
            self.external_locator_manifest_sha256,
            self.external_locator_manifest_bytes,
            self.external_locator_listing_receipt_sha256,
        )
        if self.source_family == "wikipedia_wikibooks":
            if any(value is None for value in external):
                raise ValueError("Wikipedia requires its external parent receipt")
            _require_sha256(external[0], "external locator manifest")
            _require_positive_int(external[1], "external locator manifest bytes")
            _require_sha256(external[2], "external listing receipt")
        elif any(value is not None for value in external):
            raise ValueError("only Wikipedia may bind an external locator manifest")
        if self.route.available_bytes_basis in _DIRECT_AVAILABLE_BYTE_BASES:
            if self.observed_available_bytes != self.asset_payload_bytes:
                raise ValueError("direct available bytes differ from selected files")

    @property
    def source_family(self) -> str:
        return self.route.source_family

    @property
    def asset_payload_bytes(self) -> int:
        return sum(asset.upstream_bytes for asset in self.assets)

    @property
    def declaration_matches(self) -> bool:
        return (
            len(self.assets) == self.route.asset_count
            and self.observed_available_bytes == self.route.available_bytes
        )

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            "weft1_family_upstream_enumeration_v4", self
        )


@dataclass(frozen=True)
class UpstreamEnumerationReceiptV4:
    schema: str
    mode: str
    execution_binding: PAExecutionBindingV4
    enumerator_binding_sha256: str
    replay_attestation_receipt_sha256: str | None
    families: tuple[FamilyEnumerationV4, ...]

    def __post_init__(self) -> None:
        if self.schema != UPSTREAM_ENUMERATION_SCHEMA_V4:
            raise ValueError("unexpected V4 enumeration schema")
        if self.mode not in ENUMERATION_MODES:
            raise ValueError("enumeration mode must be explicit")
        if not isinstance(self.execution_binding, PAExecutionBindingV4):
            raise TypeError("enumeration requires the full A3 execution binding")
        _require_sha256(self.enumerator_binding_sha256, "enumerator binding")
        if (
            self.mode == FIXTURE_MODE
            and self.enumerator_binding_sha256 != _FIXTURE_ENUMERATOR_BINDING_SHA256
        ):
            raise ValueError("fixture enumeration uses a foreign enumerator binding")
        if self.mode == AUTHORITATIVE_MODE:
            _require_sha256(
                self.replay_attestation_receipt_sha256,
                "replay attestation receipt",
            )
        elif self.replay_attestation_receipt_sha256 is not None:
            raise ValueError("fixture enumeration may not bind production replay authority")
        if tuple(family.source_family for family in self.families) != SOURCE_FAMILIES:
            raise ValueError("enumeration must cover every family in canonical order")
        if any(
            family.execution_binding_sha256 != self.execution_binding.receipt_sha256
            for family in self.families
        ):
            raise ValueError("enumeration family authority binding drifted")
        if self.mode == AUTHORITATIVE_MODE:
            mismatches = tuple(
                family.source_family
                for family in self.families
                if not family.declaration_matches
            )
            if mismatches:
                raise ValueError(
                    "authoritative V4 enumeration disagrees with effective routes: "
                    + ", ".join(mismatches)
                )

    @property
    def authoritative(self) -> bool:
        return self.mode == AUTHORITATIVE_MODE

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(self.schema, self)


def _normalise_hf_entry_v4(
    context: PASourceExecutionContextV4,
    route: EffectiveSourceRouteA3,
    raw: object,
) -> UpstreamAssetV4 | None:
    entry_type = _field(raw, "type")
    if entry_type is None:
        has_file_shape = _field(raw, "size") is not None and _field(raw, "blob_id") is not None
        has_folder_shape = _field(raw, "tree_id") is not None
        if has_file_shape == has_folder_shape:
            raise ValueError("Hugging Face tree entry has an ambiguous shape")
        entry_type = "file" if has_file_shape else "directory"
    if entry_type in {"directory", "dir", "tree"}:
        return None
    if entry_type not in {"file", "blob"}:
        raise ValueError("Hugging Face entry lacks an exact file type")
    path = _field(raw, "path", default=_field(raw, "rfilename"))
    if not isinstance(path, str):
        raise ValueError("Hugging Face file entry lacks a path")
    path = _canonical_path(path)
    if not locator_matches_effective_route_v4(context, route, path):
        return None
    size = _field(raw, "size")
    _require_positive_int(size, "Hugging Face file size")
    blob_id = _field(raw, "blob_id")
    if not isinstance(blob_id, str):
        raise ValueError("selected Hugging Face file lacks a blob_id")
    if _SHA1.fullmatch(blob_id):
        blob_kind = "git_sha1"
    elif _SHA256.fullmatch(blob_id):
        blob_kind = "git_sha256"
    else:
        raise ValueError("selected Hugging Face blob_id is not a lowercase hash")
    content_sha256: str | None = None
    lfs = _field(raw, "lfs")
    if lfs is not None:
        raw_sha = _field(lfs, "sha256", default=_field(lfs, "oid"))
        if isinstance(raw_sha, str) and raw_sha.startswith("sha256:"):
            raw_sha = raw_sha.removeprefix("sha256:")
        content_sha256 = _require_sha256(raw_sha, "LFS content SHA-256")
        lfs_size = _field(lfs, "size")
        if lfs_size is not None and lfs_size != size:
            raise ValueError("Hugging Face file and LFS sizes disagree")
    xet_hash = _field(raw, "xet_hash")
    if xet_hash is not None:
        _require_sha256(xet_hash, "Xet identity")
        if content_sha256 is None:
            raise ValueError("selected Xet-only asset lacks raw-content identity")
    return UpstreamAssetV4(
        source_family=route.source_family,
        repository=route.repository,
        config=route.config,
        revision=route.revision,
        split=route.split,
        asset_locator=path,
        upstream_bytes=size,  # type: ignore[arg-type]
        blob_identity_kind=blob_kind,
        blob_identity=blob_id,
        content_sha256=content_sha256,
        effective_route_receipt_sha256=route.receipt_sha256,
        execution_binding_sha256=context.binding_sha256,
    )


def _enumerate_external_wikipedia_v4(
    context: PASourceExecutionContextV4,
    route: EffectiveSourceRouteA3,
    enumerate_external_locators: Callable[..., ExternalLocatorListingV3],
    mode: str,
) -> tuple[tuple[UpstreamAssetV4, ...], int, str, int, str]:
    base_routes = {
        row.source_family: row for row in load_source_route_manifest().routes
    }
    base = base_routes[route.source_family]
    if (
        route.route_resolution != "PASSTHROUGH_A1_UNCHANGED"
        or route.external_locator_manifest_sha256
        != base.external_locator_manifest_sha256
    ):
        raise SourceFetchV4Error("Wikipedia is not an exact A1 passthrough")
    from training.weft1_corpus_sources_a2 import ExactSourceRouteV3

    exact = ExactSourceRouteV3.from_a1(base)
    listing = enumerate_external_locators(
        route=exact,
        expected_manifest_sha256=base.external_locator_manifest_sha256,
    )
    if not isinstance(listing, ExternalLocatorListingV3):
        raise TypeError("external locator enumerator returned an untyped listing")
    if mode == AUTHORITATIVE_MODE and not listing.parent_manifest_verified:
        raise ValueError("authoritative Wikipedia enumeration lacks parent verification")
    if (
        listing.external_locator_manifest_sha256
        != route.external_locator_manifest_sha256
        or listing.available_bytes_basis != route.available_bytes_basis
    ):
        raise ValueError("Wikipedia external listing authority drifted")
    assets = tuple(
        UpstreamAssetV4(
            source_family=route.source_family,
            repository=route.repository,
            config=route.config,
            revision=route.revision,
            split=route.split,
            asset_locator=item.locator,
            upstream_bytes=item.upstream_bytes,
            blob_identity_kind="content_sha256",
            blob_identity=item.content_sha256,
            content_sha256=item.content_sha256,
            effective_route_receipt_sha256=route.receipt_sha256,
            execution_binding_sha256=context.binding_sha256,
        )
        for item in listing.assets
    )
    for asset in assets:
        if not locator_matches_effective_route_v4(
            context, route, asset.asset_locator
        ):
            raise ValueError("Wikipedia locator falls outside the effective route")
    return (
        _canonical_upstream_order_v4(assets),
        listing.available_bytes,
        listing.external_locator_manifest_sha256,
        listing.external_locator_manifest_bytes,
        listing.receipt_sha256,
    )


def _replay_changed_family_trees_v4(
    context: PASourceExecutionContextV4,
    tree_cache: Mapping[tuple[str, str], Sequence[object]],
) -> None:
    """Rebuild the compact observer from full live trees before authority mints."""

    if context.path_breakdown is None:
        raise SourceFetchV4Error(
            "AUTHORITATIVE enumeration requires the typed A3 breakdown"
        )
    by_family = {route.source_family: route for route in context.routes}
    dolma_route = by_family["dolma_web"]
    fineweb_route = by_family["fineweb_edu"]
    dolma_tree = tree_cache.get((dolma_route.repository, dolma_route.revision))
    fineweb_tree = tree_cache.get((fineweb_route.repository, fineweb_route.revision))
    if dolma_tree is None or fineweb_tree is None:
        raise SourceFetchV4Error("live A3 replay lacks a changed-family tree")
    replayed = replay_upstream_path_breakdown_a3(
        context.path_breakdown,
        dolma_members=observe_hf_tree_files_a3(dolma_tree),
        fineweb_members=observe_hf_tree_files_a3(fineweb_tree),
    )
    if replayed != context.path_breakdown:
        raise SourceFetchV4Error("live A3 path replay differs from its binding")


def _enumerate_upstream_assets_impl_v4(
    *,
    context: PASourceExecutionContextV4,
    list_repo_tree: Callable[..., Iterable[object]],
    enumerate_external_locators: Callable[..., ExternalLocatorListingV3],
    mode: str,
    enumerator_binding_sha256: str,
    replay_attestation_receipt_sha256: str | None,
) -> UpstreamEnumerationReceiptV4:
    if not isinstance(context, PASourceExecutionContextV4):
        raise TypeError("V4 enumeration requires a source execution context")
    if not callable(list_repo_tree) or not callable(enumerate_external_locators):
        raise TypeError("enumerators must be callable")
    if mode not in ENUMERATION_MODES:
        raise ValueError("enumeration mode is unknown")
    tree_cache: dict[tuple[str, str], tuple[object, ...]] = {}
    families: list[FamilyEnumerationV4] = []
    for route in context.routes:
        if route.source_family == "wikipedia_wikibooks":
            (
                assets,
                observed,
                external_sha,
                external_bytes,
                external_receipt,
            ) = _enumerate_external_wikipedia_v4(
                context,
                route,
                enumerate_external_locators,
                mode,
            )
        else:
            cache_key = (route.repository, route.revision)
            if cache_key not in tree_cache:
                raw_tree = list_repo_tree(
                    repo_id=route.repository,
                    repo_type="dataset",
                    revision=route.revision,
                    recursive=True,
                    expand=PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3.expand,
                )
                if isinstance(raw_tree, (str, bytes, Mapping)):
                    raise TypeError("list_repo_tree must return repository entries")
                tree_cache[cache_key] = tuple(raw_tree)
            assets = _canonical_upstream_order_v4(
                tuple(
                    asset
                    for raw in tree_cache[cache_key]
                    if (
                        asset := _normalise_hf_entry_v4(context, route, raw)
                    )
                    is not None
                )
            )
            if not assets:
                raise ValueError(
                    f"effective selector enumerated no assets for {route.source_family}"
                )
            observed = sum(asset.upstream_bytes for asset in assets)
            external_sha = None
            external_bytes = None
            external_receipt = None
        for asset in assets:
            if not locator_matches_effective_route_v4(
                context, route, asset.asset_locator
            ):
                raise AssertionError("enumerator admitted an out-of-route asset")
        families.append(
            FamilyEnumerationV4(
                route=route,
                execution_binding_sha256=context.binding_sha256,
                observed_available_bytes=observed,
                assets=assets,
                external_locator_manifest_sha256=external_sha,
                external_locator_manifest_bytes=external_bytes,
                external_locator_listing_receipt_sha256=external_receipt,
            )
        )
    if mode == AUTHORITATIVE_MODE:
        _replay_changed_family_trees_v4(context, tree_cache)
    return UpstreamEnumerationReceiptV4(
        schema=UPSTREAM_ENUMERATION_SCHEMA_V4,
        mode=mode,
        execution_binding=context.binding,
        enumerator_binding_sha256=enumerator_binding_sha256,
        replay_attestation_receipt_sha256=replay_attestation_receipt_sha256,
        families=tuple(families),
    )


def enumerate_upstream_assets_v4(
    *,
    context: PASourceExecutionContextV4,
    list_repo_tree: Callable[..., Iterable[object]],
    enumerate_external_locators: Callable[..., ExternalLocatorListingV3],
) -> UpstreamEnumerationReceiptV4:
    """Enumerate an offline fixture; this API can never mint AUTHORITATIVE."""

    return _enumerate_upstream_assets_impl_v4(
        context=context,
        list_repo_tree=list_repo_tree,
        enumerate_external_locators=enumerate_external_locators,
        mode=FIXTURE_MODE,
        enumerator_binding_sha256=_FIXTURE_ENUMERATOR_BINDING_SHA256,
        replay_attestation_receipt_sha256=None,
    )


def _enumerate_authoritative_upstream_assets_v4(
    *,
    context: PASourceExecutionContextV4,
    open_resource: Callable[[str], BinaryIO],
    replay_attestation: A3ReplayAttestationV4,
    source_prep_code_identity: SourcePrepCodeIdentityV4,
) -> UpstreamEnumerationReceiptV4:
    """Private authoritative enumerator used only by the validated wrapper."""

    validate_a3_replay_attestation_v4(
        replay_attestation,
        context=context,
        source_prep_code_identity=source_prep_code_identity,
    )
    observed_version = metadata.version(_HUGGINGFACE_HUB_DISTRIBUTION)
    if observed_version != _HUGGINGFACE_HUB_VERSION:
        raise RuntimeError("authoritative enumeration requires huggingface_hub 1.24.0")
    from huggingface_hub import HfApi

    api = HfApi()
    if type(api).__name__ != "HfApi" or type(api).__module__ != "huggingface_hub.hf_api":
        raise RuntimeError("authoritative enumeration instantiated a foreign HfApi")
    if getattr(api, "endpoint", None) != PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3.endpoint:
        raise RuntimeError("authoritative enumeration endpoint differs from observer")
    base_by_family = {
        row.source_family: row for row in load_source_route_manifest().routes
    }

    def external_factory(**kwargs: object) -> ExternalLocatorListingV3:
        route = kwargs.get("route")
        if route is None or getattr(route, "source_family", None) != "wikipedia_wikibooks":
            raise TypeError("authoritative external factory received a foreign route")
        base = base_by_family["wikipedia_wikibooks"]
        if kwargs.get("expected_manifest_sha256") != base.external_locator_manifest_sha256:
            raise ValueError("authoritative external parent hash drifted")
        return read_pinned_external_locator_listing_v3(
            route=route,  # type: ignore[arg-type]
            declared=base,
            open_resource=open_resource,
        )

    binding_sha256 = execution_authority_v4_bound_sha256(
        "weft1_huggingface_hub_enumerator_runtime_v4",
        {
            "client_class": f"{type(api).__module__}.{type(api).__name__}",
            "distribution": _HUGGINGFACE_HUB_DISTRIBUTION,
            "execution_binding_sha256": context.binding_sha256,
            "observation_client_identity_sha256": (
                PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3.receipt_sha256
            ),
            "version": observed_version,
        },
    )
    return _enumerate_upstream_assets_impl_v4(
        context=context,
        list_repo_tree=api.list_repo_tree,
        enumerate_external_locators=external_factory,
        mode=AUTHORITATIVE_MODE,
        enumerator_binding_sha256=binding_sha256,
        replay_attestation_receipt_sha256=replay_attestation.receipt_sha256,
    )


@dataclass(frozen=True)
class FamilyPrefixSelectionV4:
    source_family: str
    execution_binding_sha256: str
    required_bytes: int
    available_asset_count: int
    available_payload_bytes: int
    selected_asset_count: int
    selected_upstream_bytes: int
    selected_asset_identities_sha256: str
    terminal_asset_identity_sha256: str
    terminal_asset_locator: str
    selection_rule: str

    def __post_init__(self) -> None:
        if self.source_family not in SOURCE_FAMILIES:
            raise ValueError("prefix selection uses an unknown source family")
        _require_sha256(self.execution_binding_sha256, "execution binding")
        for name in (
            "required_bytes",
            "available_asset_count",
            "available_payload_bytes",
            "selected_asset_count",
            "selected_upstream_bytes",
        ):
            _require_positive_int(getattr(self, name), name)
        for name in (
            "selected_asset_identities_sha256",
            "terminal_asset_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_nonempty(self.terminal_asset_locator, "terminal asset locator")
        if self.selection_rule not in {
            "minimal_seeded_prefix_reaching_required_bytes",
            "complete_pinned_wikipedia_asset_set",
        }:
            raise ValueError("prefix selection rule is unknown")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            "weft1_family_prefix_selection_v4", self
        )


@dataclass(frozen=True)
class SourceAssetDownloadPlanV4:
    execution_binding: PAExecutionBindingV4
    enumeration_receipt_sha256: str
    enumeration_mode: str
    assets: tuple[UpstreamAssetV4, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.execution_binding, PAExecutionBindingV4):
            raise TypeError("download plan requires an A3 execution binding")
        _require_sha256(self.enumeration_receipt_sha256, "enumeration receipt")
        if self.enumeration_mode not in ENUMERATION_MODES:
            raise ValueError("download plan enumeration mode is invalid")
        if not self.assets:
            raise ValueError("download plan requires assets")
        if any(
            asset.execution_binding_sha256 != self.execution_binding.receipt_sha256
            for asset in self.assets
        ):
            raise ValueError("download plan asset binding drifted")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(DOWNLOAD_PLAN_SCHEMA_V4, self)


@dataclass(frozen=True)
class PASourcePrefixSelectionReceiptV4:
    schema: str
    execution_binding: PAExecutionBindingV4
    enumeration_receipt_sha256: str
    families: tuple[FamilyPrefixSelectionV4, ...]
    selection_plan_sha256: str

    def __post_init__(self) -> None:
        if self.schema != PREFIX_SELECTION_SCHEMA_V4:
            raise ValueError("unexpected V4 prefix-selection schema")
        if not isinstance(self.execution_binding, PAExecutionBindingV4):
            raise TypeError("prefix selection requires an A3 execution binding")
        _require_sha256(self.enumeration_receipt_sha256, "enumeration receipt")
        _require_sha256(self.selection_plan_sha256, "selection plan")
        if tuple(row.source_family for row in self.families) != SOURCE_FAMILIES:
            raise ValueError("prefix selection lacks a canonical family row")
        if any(
            row.execution_binding_sha256 != self.execution_binding.receipt_sha256
            for row in self.families
        ):
            raise ValueError("prefix selection family binding drifted")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(self.schema, self)


def select_required_asset_prefixes_v4(
    enumeration: UpstreamEnumerationReceiptV4,
) -> tuple[SourceAssetDownloadPlanV4, PASourcePrefixSelectionReceiptV4]:
    if not isinstance(enumeration, UpstreamEnumerationReceiptV4):
        raise TypeError("prefix selection requires a typed V4 enumeration")
    selected_all: list[UpstreamAssetV4] = []
    rows: list[FamilyPrefixSelectionV4] = []
    for family in enumeration.families:
        target = family.route.required_bytes
        if family.source_family == "wikipedia_wikibooks":
            selected = family.assets
            rule = "complete_pinned_wikipedia_asset_set"
            if enumeration.authoritative and len(selected) != 2:
                raise SourceFetchV4Error("Wikipedia must contain both pinned assets")
        else:
            chosen: list[UpstreamAssetV4] = []
            cumulative = 0
            for asset in family.assets:
                chosen.append(asset)
                cumulative += asset.upstream_bytes
                if cumulative >= target:
                    break
            selected = tuple(chosen)
            rule = "minimal_seeded_prefix_reaching_required_bytes"
        selected_bytes = sum(asset.upstream_bytes for asset in selected)
        if not selected or selected_bytes < target:
            raise SourceFetchV4Error(
                f"enumerated {family.source_family} assets do not reach required bytes"
            )
        if rule.startswith("minimal") and selected_bytes - selected[-1].upstream_bytes >= target:
            raise AssertionError("selected source prefix is not minimal")
        identities = tuple(asset.asset_identity_sha256 for asset in selected)
        rows.append(
            FamilyPrefixSelectionV4(
                source_family=family.source_family,
                execution_binding_sha256=enumeration.execution_binding.receipt_sha256,
                required_bytes=target,
                available_asset_count=len(family.assets),
                available_payload_bytes=family.asset_payload_bytes,
                selected_asset_count=len(selected),
                selected_upstream_bytes=selected_bytes,
                selected_asset_identities_sha256=canonical_sha256(identities),
                terminal_asset_identity_sha256=selected[-1].asset_identity_sha256,
                terminal_asset_locator=selected[-1].asset_locator,
                selection_rule=rule,
            )
        )
        selected_all.extend(selected)
    plan = SourceAssetDownloadPlanV4(
        execution_binding=enumeration.execution_binding,
        enumeration_receipt_sha256=enumeration.receipt_sha256,
        enumeration_mode=enumeration.mode,
        assets=tuple(selected_all),
    )
    receipt = PASourcePrefixSelectionReceiptV4(
        schema=PREFIX_SELECTION_SCHEMA_V4,
        execution_binding=enumeration.execution_binding,
        enumeration_receipt_sha256=enumeration.receipt_sha256,
        families=tuple(rows),
        selection_plan_sha256=plan.receipt_sha256,
    )
    return plan, receipt


@dataclass(frozen=True)
class SourceCacheAssetV4(SourceCacheAssetV3):
    """A V4 cache asset; subclassing keeps A2's parsers reusable offline."""

    effective_route_receipt_sha256: str
    execution_binding_sha256: str

    def __post_init__(self) -> None:
        if self.source_family not in SOURCE_FAMILIES:
            raise ValueError("unknown source family")
        for name in (
            "repository",
            "config",
            "revision",
            "split",
            "asset_locator",
            "relative_path",
        ):
            _require_nonempty(getattr(self, name), name)
        if _SHA1.fullmatch(self.revision) is None:
            raise ValueError("cache route revision must be a commit SHA")
        _canonical_path(self.relative_path)
        _require_positive_int(self.bytes, "cache asset bytes")
        _require_sha256(self.sha256, "cache asset SHA-256")
        _require_sha256(self.effective_route_receipt_sha256, "effective route receipt")
        _require_sha256(self.execution_binding_sha256, "execution binding")

    @property
    def logical_identity_payload(self) -> Mapping[str, object]:
        return {
            "asset_locator": self.asset_locator,
            "bytes": self.bytes,
            "config": self.config,
            "effective_route_receipt_sha256": self.effective_route_receipt_sha256,
            "execution_binding_sha256": self.execution_binding_sha256,
            "repository": self.repository,
            "revision": self.revision,
            "sha256": self.sha256,
            "source_family": self.source_family,
            "split": self.split,
        }

    @property
    def asset_identity_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            "weft1_source_cache_asset_v4", self.logical_identity_payload
        )


def _canonical_cache_order_v4(
    assets: Sequence[SourceCacheAssetV4],
) -> tuple[SourceCacheAssetV4, ...]:
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
        raise TypeError("cache assets must be a typed sequence")
    if any(not isinstance(asset, SourceCacheAssetV4) for asset in assets):
        raise TypeError("cache assets contain an untyped value")
    ordered = tuple(
        sorted(
            assets,
            key=lambda asset: (
                asset_order_digest_v3(asset.asset_locator),
                asset.source_family.encode("utf-8"),
                asset.asset_locator.encode("utf-8"),
                asset.sha256,
            ),
        )
    )
    if len({(item.source_family, item.asset_locator) for item in ordered}) != len(ordered):
        raise ValueError("cache manifest repeats a source locator")
    if len({item.relative_path for item in ordered}) != len(ordered):
        raise ValueError("cache manifest repeats a local path")
    return ordered


@dataclass(frozen=True)
class SourceCacheManifestV4:
    schema: str
    execution_binding: PAExecutionBindingV4
    effective_route_identity_sha256: str
    selection_plan_sha256: str
    assets: tuple[SourceCacheAssetV4, ...]

    def __post_init__(self) -> None:
        if self.schema != SOURCE_CACHE_MANIFEST_SCHEMA_V4:
            raise ValueError("unexpected V4 source-cache schema")
        if not isinstance(self.execution_binding, PAExecutionBindingV4):
            raise TypeError("cache manifest requires an A3 execution binding")
        if (
            self.effective_route_identity_sha256
            != self.execution_binding.effective_route_identity_sha256
        ):
            raise ValueError("cache manifest effective-route identity drifted")
        _require_sha256(self.selection_plan_sha256, "selection plan")
        if not self.assets or self.assets != _canonical_cache_order_v4(self.assets):
            raise ValueError("cache manifest assets are absent or noncanonical")
        if any(
            asset.execution_binding_sha256 != self.execution_binding.receipt_sha256
            for asset in self.assets
        ):
            raise ValueError("cache manifest asset authority drifted")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            SOURCE_CACHE_MANIFEST_SCHEMA_V4, self
        )


@dataclass(frozen=True)
class DownloadedAssetEvidenceV4:
    execution_binding_sha256: str
    upstream_asset_identity_sha256: str
    source_cache_asset_identity_sha256: str
    relative_path: str
    observed_bytes: int
    observed_sha256: str
    upstream_identity_check: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.execution_binding_sha256, "execution binding"),
            (self.upstream_asset_identity_sha256, "upstream asset identity"),
            (self.source_cache_asset_identity_sha256, "cache asset identity"),
            (self.observed_sha256, "observed SHA-256"),
        ):
            _require_sha256(value, name)
        _canonical_path(self.relative_path)
        _require_positive_int(self.observed_bytes, "observed bytes")
        if self.upstream_identity_check not in {
            "content_sha256",
            "git_blob_sha1",
            "git_blob_sha256",
        }:
            raise ValueError("download evidence uses an unknown identity check")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            "weft1_downloaded_asset_evidence_v4", self
        )


@dataclass(frozen=True)
class SourceCachePlanMaterializationV4:
    execution_binding: PAExecutionBindingV4
    plan: SourceAssetDownloadPlanV4
    cache_assets: tuple[SourceCacheAssetV4, ...]
    evidence: tuple[DownloadedAssetEvidenceV4, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.execution_binding, PAExecutionBindingV4):
            raise TypeError("materialization requires an A3 execution binding")
        if self.plan.execution_binding != self.execution_binding:
            raise ValueError("materialization plan binding drifted")
        if len(self.cache_assets) != len(self.plan.assets) or len(self.evidence) != len(self.plan.assets):
            raise ValueError("materialization does not cover its plan")
        for upstream, cached, evidence in zip(
            self.plan.assets, self.cache_assets, self.evidence, strict=True
        ):
            if (
                cached.source_family,
                cached.repository,
                cached.config,
                cached.revision,
                cached.split,
                cached.asset_locator,
                cached.effective_route_receipt_sha256,
            ) != (
                upstream.source_family,
                upstream.repository,
                upstream.config,
                upstream.revision,
                upstream.split,
                upstream.asset_locator,
                upstream.effective_route_receipt_sha256,
            ):
                raise ValueError("materialized cache route differs from its plan")
            if (
                evidence.upstream_asset_identity_sha256 != upstream.asset_identity_sha256
                or evidence.source_cache_asset_identity_sha256 != cached.asset_identity_sha256
                or evidence.observed_bytes != cached.bytes
                or evidence.observed_sha256 != cached.sha256
            ):
                raise ValueError("materialized evidence differs from cache bytes")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            "weft1_source_cache_plan_materialization_v4", self
        )


@dataclass(frozen=True)
class VerifiedLocalCacheV4:
    execution_binding: PAExecutionBindingV4
    source_manifest: SourceCacheManifestV4
    cache_root_label: str
    observations: tuple[tuple[str, int, str], ...]

    def __post_init__(self) -> None:
        if self.source_manifest.execution_binding != self.execution_binding:
            raise ValueError("verified cache binding drifted")
        _require_nonempty(self.cache_root_label, "cache root label")
        if len(self.observations) != len(self.source_manifest.assets):
            raise ValueError("verified cache observations are incomplete")
        expected = tuple(
            (asset.relative_path, asset.bytes, asset.sha256)
            for asset in self.source_manifest.assets
        )
        if self.observations != expected:
            raise ValueError("verified cache observations differ from manifest")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            VERIFIED_CACHE_RECEIPT_SCHEMA_V4, self
        )


@dataclass(frozen=True)
class SourceCacheDownloadReceiptV4:
    schema: str
    execution_binding: PAExecutionBindingV4
    enumeration_receipt_sha256: str
    enumeration_mode: str
    selection_plan_sha256: str
    source_manifest: SourceCacheManifestV4
    evidence: tuple[DownloadedAssetEvidenceV4, ...]
    verification_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.schema != DOWNLOAD_RECEIPT_SCHEMA_V4:
            raise ValueError("unexpected V4 download-receipt schema")
        if self.source_manifest.execution_binding != self.execution_binding:
            raise ValueError("download receipt manifest binding drifted")
        for value, name in (
            (self.enumeration_receipt_sha256, "enumeration receipt"),
            (self.selection_plan_sha256, "selection plan"),
            (self.verification_receipt_sha256, "verification receipt"),
        ):
            _require_sha256(value, name)
        if self.enumeration_mode not in ENUMERATION_MODES:
            raise ValueError("download receipt enumeration mode is invalid")
        if len(self.evidence) != len(self.source_manifest.assets):
            raise ValueError("download evidence does not cover the manifest")
        for item, asset in zip(self.evidence, self.source_manifest.assets, strict=True):
            if (
                item.execution_binding_sha256 != self.execution_binding.receipt_sha256
                or item.source_cache_asset_identity_sha256 != asset.asset_identity_sha256
                or item.relative_path != asset.relative_path
                or item.observed_bytes != asset.bytes
                or item.observed_sha256 != asset.sha256
            ):
                raise ValueError("download evidence and cache manifest are misaligned")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(self.schema, self)


def _git_blob_digest(algorithm: str, byte_count: int) -> object:
    digest = hashlib.new(algorithm)
    digest.update(f"blob {byte_count}\0".encode("ascii"))
    return digest


def _verified_stream_identity_v4(
    asset: UpstreamAssetV4,
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
            if chunk in (None, b""):
                break
            if not isinstance(chunk, bytes):
                raise SourceFetchV4Error("downloader returned a non-bytes chunk")
            byte_count += len(chunk)
            if byte_count > asset.upstream_bytes:
                raise SourceFetchV4Error("download exceeded pinned upstream bytes")
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
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if output is not None and not output.closed:
            output.close()
    if byte_count != asset.upstream_bytes:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise SourceFetchV4Error("download byte count differs from upstream metadata")
    raw_digest = raw_sha256.hexdigest()
    if asset.content_sha256 is not None:
        valid = raw_digest == asset.content_sha256
        identity_check = "content_sha256"
    elif asset.blob_identity_kind == "git_sha1":
        valid = git_sha1.hexdigest() == asset.blob_identity
        identity_check = "git_blob_sha1"
    elif asset.blob_identity_kind == "git_sha256":
        valid = git_sha256.hexdigest() == asset.blob_identity
        identity_check = "git_blob_sha256"
    else:
        valid = raw_digest == asset.blob_identity
        identity_check = "content_sha256"
    if not valid:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise SourceFetchV4Error("download differs from its upstream byte identity")
    return byte_count, raw_digest, identity_check


def _cache_relative_path_v4(asset: UpstreamAssetV4) -> str:
    suffix = _CONTAINER_SUFFIXES[asset.source_family]
    if not asset.asset_locator.endswith(suffix):
        raise SourceFetchV4Error("asset has the wrong registered container suffix")
    return f"assets/{asset.source_family}/{asset.asset_identity_sha256}{suffix}"


def _safe_cache_path(root: Path, relative_path: str, *, strict: bool) -> Path:
    relative = PurePosixPath(_canonical_path(relative_path))
    lexical = root.joinpath(*relative.parts)
    assert_no_symlink_ancestors(lexical)
    resolved_root = root.resolve(strict=True)
    candidate = lexical.resolve(strict=strict)
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise SourceFetchV4Error("cache path escapes its root")
    return candidate


def _open_v4_with_v3_opener(
    opener: PinnedHuggingFaceAssetOpenerV3,
    asset: UpstreamAssetV4,
) -> BinaryIO:
    """Use A2's pinned transport implementation without minting a V3 receipt."""

    legacy_transport_view = UpstreamAssetV3(
        source_family=asset.source_family,
        repository=asset.repository,
        config=asset.config,
        revision=asset.revision,
        split=asset.split,
        asset_locator=asset.asset_locator,
        upstream_bytes=asset.upstream_bytes,
        blob_identity_kind=asset.blob_identity_kind,
        blob_identity=asset.blob_identity,
        content_sha256=asset.content_sha256,
    )
    return opener.open(legacy_transport_view)


def materialize_source_cache_v4(
    enumeration: UpstreamEnumerationReceiptV4,
    plan: SourceAssetDownloadPlanV4,
    cache_root: Path,
    *,
    open_upstream: Callable[[UpstreamAssetV4], BinaryIO],
    allow_nonauthoritative_fixture: bool = False,
    resume_incomplete: bool = True,
    _authoritative_capability: object | None = None,
) -> SourceCachePlanMaterializationV4:
    if not isinstance(enumeration, UpstreamEnumerationReceiptV4):
        raise TypeError("cache materialization requires a V4 enumeration")
    if (
        enumeration.authoritative
        and _authoritative_capability is not _AUTHORITATIVE_PREP_CAPABILITY
    ):
        raise SourceFetchV4Error(
            "authoritative cache materialization is private to online source prep"
        )
    if not isinstance(plan, SourceAssetDownloadPlanV4):
        raise TypeError("cache materialization requires a V4 plan")
    if (
        plan.enumeration_receipt_sha256 != enumeration.receipt_sha256
        or plan.enumeration_mode != enumeration.mode
        or plan.execution_binding != enumeration.execution_binding
    ):
        raise SourceFetchV4Error("download plan belongs to another enumeration")
    enumerated = {
        asset.asset_identity_sha256: asset
        for family in enumeration.families
        for asset in family.assets
    }
    if any(
        enumerated.get(asset.asset_identity_sha256) != asset
        for asset in plan.assets
    ):
        raise SourceFetchV4Error("download plan contains an unenrolled asset")
    if not enumeration.authoritative and not allow_nonauthoritative_fixture:
        raise SourceFetchV4Error("production cache requires authoritative enumeration")
    if not isinstance(cache_root, Path) or not callable(open_upstream):
        raise TypeError("cache root and opener must be typed")
    assert_no_symlink_ancestors(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    root = cache_root.resolve(strict=True)
    cached: list[SourceCacheAssetV4] = []
    evidence: list[DownloadedAssetEvidenceV4] = []
    for upstream in plan.assets:
        relative = _cache_relative_path_v4(upstream)
        final_path = _safe_cache_path(root, relative, strict=False)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        partial = final_path.with_name(final_path.name + ".partial")
        if partial.exists():
            if not resume_incomplete or not partial.is_file():
                raise SourceFetchV4Error("stale cache partial is not resumable")
            partial.unlink()
        if final_path.exists():
            with final_path.open("rb") as source:
                count, observed_sha, identity_check = _verified_stream_identity_v4(
                    upstream, source, None
                )
        else:
            opened = open_upstream(upstream)
            if not hasattr(opened, "read"):
                raise SourceFetchV4Error("open_upstream returned no binary reader")
            with closing(opened):
                count, observed_sha, identity_check = _verified_stream_identity_v4(
                    upstream, opened, partial
                )
            os.replace(partial, final_path)
        item = SourceCacheAssetV4(
            source_family=upstream.source_family,
            repository=upstream.repository,
            config=upstream.config,
            revision=upstream.revision,
            split=upstream.split,
            asset_locator=upstream.asset_locator,
            relative_path=relative,
            bytes=count,
            sha256=observed_sha,
            effective_route_receipt_sha256=upstream.effective_route_receipt_sha256,
            execution_binding_sha256=enumeration.execution_binding.receipt_sha256,
        )
        cached.append(item)
        evidence.append(
            DownloadedAssetEvidenceV4(
                execution_binding_sha256=enumeration.execution_binding.receipt_sha256,
                upstream_asset_identity_sha256=upstream.asset_identity_sha256,
                source_cache_asset_identity_sha256=item.asset_identity_sha256,
                relative_path=relative,
                observed_bytes=count,
                observed_sha256=observed_sha,
                upstream_identity_check=identity_check,
            )
        )
    return SourceCachePlanMaterializationV4(
        execution_binding=enumeration.execution_binding,
        plan=plan,
        cache_assets=tuple(cached),
        evidence=tuple(evidence),
    )


def _write_or_verify_artifact(path: Path, payload: Mapping[str, object]) -> str:
    if not isinstance(path, Path):
        raise TypeError("artifact path must be a pathlib.Path")
    assert_no_symlink_ancestors(path)
    expected = canonical_json_bytes(payload) + b"\n"
    if path.exists():
        observed, decoded = load_canonical_json_snapshot(path)
        del decoded
        if observed != expected:
            raise SourceFetchV4Error("existing V4 artifact differs; refusing overwrite")
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    assert_no_symlink_ancestors(partial)
    if partial.exists():
        if not partial.is_file():
            raise SourceFetchV4Error("artifact partial is not a regular file")
        partial.unlink()
    try:
        with partial.open("xb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return hashlib.sha256(expected).hexdigest()


def _verify_local_cache_v4(
    manifest: SourceCacheManifestV4,
    cache_root: Path,
) -> VerifiedLocalCacheV4:
    if not isinstance(manifest, SourceCacheManifestV4):
        raise TypeError("cache verification requires a V4 manifest")
    assert_no_symlink_ancestors(cache_root)
    root = cache_root.resolve(strict=True)
    if not root.is_dir():
        raise SourceFetchV4Error("cache root is not a directory")
    observations: list[tuple[str, int, str]] = []
    for asset in manifest.assets:
        path = _safe_cache_path(root, asset.relative_path, strict=True)
        if not path.is_file():
            raise SourceFetchV4Error("cache asset is not a regular file")
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                count += len(chunk)
                digest.update(chunk)
        if count != asset.bytes or digest.hexdigest() != asset.sha256:
            raise SourceFetchV4Error("local cache bytes differ from the V4 manifest")
        observations.append((asset.relative_path, count, digest.hexdigest()))
    return VerifiedLocalCacheV4(
        execution_binding=manifest.execution_binding,
        source_manifest=manifest,
        cache_root_label=cache_root.name or ".",
        observations=tuple(observations),
    )


def _manifest_envelope(manifest: SourceCacheManifestV4) -> Mapping[str, object]:
    return {
        "manifest": asdict(manifest),
        "manifest_sha256": manifest.receipt_sha256,
        "schema": "weft1_local_source_cache_manifest_artifact_v4",
    }


def finalize_source_cache_v4(
    enumeration: UpstreamEnumerationReceiptV4,
    materialization: SourceCachePlanMaterializationV4,
    cache_root: Path,
    manifest_path: Path,
    *,
    allow_nonauthoritative_fixture: bool = False,
    _authoritative_capability: object | None = None,
) -> tuple[SourceCacheDownloadReceiptV4, VerifiedLocalCacheV4, str]:
    if not isinstance(enumeration, UpstreamEnumerationReceiptV4):
        raise TypeError("cache finalization requires a V4 enumeration")
    if (
        enumeration.authoritative
        and _authoritative_capability is not _AUTHORITATIVE_PREP_CAPABILITY
    ):
        raise SourceFetchV4Error(
            "authoritative cache finalization is private to online source prep"
        )
    if not isinstance(materialization, SourceCachePlanMaterializationV4):
        raise TypeError("cache finalization requires a V4 materialization")
    if materialization.execution_binding != enumeration.execution_binding:
        raise SourceFetchV4Error("cache materialization belongs to another authority")
    if not enumeration.authoritative and not allow_nonauthoritative_fixture:
        raise SourceFetchV4Error("production finalization requires AUTHORITATIVE")
    ordered_assets = _canonical_cache_order_v4(materialization.cache_assets)
    evidence_by_cache = {
        item.source_cache_asset_identity_sha256: item
        for item in materialization.evidence
    }
    if len(evidence_by_cache) != len(materialization.evidence):
        raise SourceFetchV4Error("cache finalization repeats evidence")
    ordered_evidence = tuple(
        evidence_by_cache[asset.asset_identity_sha256] for asset in ordered_assets
    )
    manifest = SourceCacheManifestV4(
        schema=SOURCE_CACHE_MANIFEST_SCHEMA_V4,
        execution_binding=enumeration.execution_binding,
        effective_route_identity_sha256=(
            enumeration.execution_binding.effective_route_identity_sha256
        ),
        selection_plan_sha256=materialization.plan.receipt_sha256,
        assets=ordered_assets,
    )
    manifest_artifact_sha256 = _write_or_verify_artifact(
        manifest_path, _manifest_envelope(manifest)
    )
    verified = _verify_local_cache_v4(manifest, cache_root)
    receipt = SourceCacheDownloadReceiptV4(
        schema=DOWNLOAD_RECEIPT_SCHEMA_V4,
        execution_binding=enumeration.execution_binding,
        enumeration_receipt_sha256=enumeration.receipt_sha256,
        enumeration_mode=enumeration.mode,
        selection_plan_sha256=materialization.plan.receipt_sha256,
        source_manifest=manifest,
        evidence=ordered_evidence,
        verification_receipt_sha256=verified.receipt_sha256,
    )
    return receipt, verified, manifest_artifact_sha256


@dataclass(frozen=True)
class SourcePrepImplementationFileV4:
    repo_path: str
    bytes: int
    sha256: str
    git_blob_sha1: str

    def __post_init__(self) -> None:
        _canonical_path(self.repo_path)
        _require_positive_int(self.bytes, "implementation file bytes")
        _require_sha256(self.sha256, "implementation file SHA-256")
        if _SHA1.fullmatch(self.git_blob_sha1) is None:
            raise ValueError("implementation file requires a Git blob SHA-1")


@dataclass(frozen=True)
class SourcePrepCodeIdentityV4:
    mode: str
    execution_binding: PAExecutionBindingV4
    git_commit: str
    files: tuple[SourcePrepImplementationFileV4, ...]

    def __post_init__(self) -> None:
        if self.mode not in {AUTHORITATIVE_MODE, FIXTURE_MODE}:
            raise ValueError("source-prep code identity mode is unknown")
        if not isinstance(self.execution_binding, PAExecutionBindingV4):
            raise TypeError("source-prep code identity requires an A3 binding")
        if _SHA1.fullmatch(self.git_commit) is None:
            raise ValueError("source-prep code identity requires a commit SHA")
        paths = tuple(row.repo_path for row in self.files)
        if (
            not self.files
            or any(not isinstance(row, SourcePrepImplementationFileV4) for row in self.files)
            or paths != tuple(sorted(paths, key=lambda item: item.encode("utf-8")))
            or len(set(paths)) != len(paths)
        ):
            raise ValueError("source-prep implementation inventory is noncanonical")
        if self.mode == AUTHORITATIVE_MODE and paths != SOURCE_PREP_IMPLEMENTATION_REPO_PATHS_V4:
            raise ValueError("authoritative source-prep inventory is incomplete")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            "weft1_source_prep_code_identity_v4", self
        )


@dataclass(frozen=True)
class A3ReplayAttestationV4:
    """External clean-commit/live-replay authority required before downloads."""

    schema: str
    status: str
    authorizes_downloads: bool
    git_commit: str
    git_status: str
    authority_sha256: str
    execution_binding_sha256: str
    source_prep_code_identity_sha256: str
    semantic_evidence_artifact_physical_sha256: str
    semantic_evidence_artifact_receipt_sha256: str
    semantic_evidence_family_receipt_sha256s: tuple[tuple[str, str], ...]
    breakdown_artifact_physical_sha256: str
    breakdown_artifact_receipt_sha256: str
    overlay_artifact_physical_sha256: str
    overlay_identity_sha256: str
    effective_route_identity_sha256: str
    huggingface_hub_distribution: str
    huggingface_hub_version: str
    observation_mode: str
    observation_client_identity_sha256: str
    live_replay_status: str
    live_replay_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.schema != A3_REPLAY_ATTESTATION_SCHEMA_V4:
            raise ValueError("unexpected A3 replay-attestation schema")
        if (
            self.status != "ATTESTED_CLEAN_HEAD_LIVE_REPLAY_PASS"
            or self.authorizes_downloads is not True
            or self.git_status != "CLEAN"
            or self.live_replay_status != "PASS_EXACT_BREAKDOWN_REPLAY"
        ):
            raise ValueError("A3 replay attestation is not download-authorizing")
        if _SHA1.fullmatch(self.git_commit) is None:
            raise ValueError("A3 replay attestation requires an exact Git commit")
        if self.authority_sha256 != A3_AUTHORITY_SHA256:
            raise ValueError("A3 replay attestation uses a foreign authority")
        for value, name in (
            (self.execution_binding_sha256, "execution binding"),
            (self.source_prep_code_identity_sha256, "source-prep code identity"),
            (
                self.semantic_evidence_artifact_physical_sha256,
                "semantic-evidence physical SHA",
            ),
            (
                self.semantic_evidence_artifact_receipt_sha256,
                "semantic-evidence typed receipt",
            ),
            (self.breakdown_artifact_physical_sha256, "breakdown physical SHA"),
            (self.breakdown_artifact_receipt_sha256, "breakdown typed receipt"),
            (self.overlay_artifact_physical_sha256, "overlay physical SHA"),
            (self.overlay_identity_sha256, "overlay identity"),
            (self.effective_route_identity_sha256, "effective route identity"),
            (self.live_replay_receipt_sha256, "live replay receipt"),
        ):
            _require_sha256(value, name)
        if tuple(family for family, _ in self.semantic_evidence_family_receipt_sha256s) != (
            "dolma_web",
            "fineweb_edu",
        ):
            raise ValueError("A3 replay attestation lacks family evidence receipts")
        for _, receipt in self.semantic_evidence_family_receipt_sha256s:
            _require_sha256(receipt, "family semantic-evidence receipt")
        if (
            self.huggingface_hub_distribution != _HUGGINGFACE_HUB_DISTRIBUTION
            or self.huggingface_hub_version != _HUGGINGFACE_HUB_VERSION
            or self.observation_mode != PRODUCTION_OBSERVATION_MODE_A3
            or self.observation_client_identity_sha256
            != PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3.receipt_sha256
        ):
            raise ValueError("A3 replay attestation runtime pin drifted")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(self.schema, self)


def validate_a3_replay_attestation_v4(
    attestation: A3ReplayAttestationV4,
    *,
    context: PASourceExecutionContextV4,
    source_prep_code_identity: SourcePrepCodeIdentityV4,
) -> None:
    """Require every governed identity before constructing a download plan."""

    if not isinstance(attestation, A3ReplayAttestationV4):
        raise TypeError("online source prep requires a typed A3 replay attestation")
    if not isinstance(context.path_breakdown, UpstreamPathBreakdownReceiptA3):
        raise SourceFetchV4Error("online source prep lacks the typed A3 breakdown")
    if source_prep_code_identity.mode != AUTHORITATIVE_MODE:
        raise SourceFetchV4Error("online source prep code is not clean-head attested")
    expected = (
        source_prep_code_identity.git_commit,
        context.binding_sha256,
        source_prep_code_identity.receipt_sha256,
        context.semantic_evidence_artifact_physical_sha256,
        context.semantic_evidence_artifact_receipt_sha256,
        context.semantic_evidence_family_receipt_sha256s,
        context.binding.breakdown_artifact_physical_sha256,
        context.binding.breakdown_artifact_receipt_sha256,
        context.overlay_physical_sha256,
        context.overlay_identity_sha256,
        context.binding.effective_route_identity_sha256,
        context.path_breakdown.receipt_sha256,
    )
    observed = (
        attestation.git_commit,
        attestation.execution_binding_sha256,
        attestation.source_prep_code_identity_sha256,
        attestation.semantic_evidence_artifact_physical_sha256,
        attestation.semantic_evidence_artifact_receipt_sha256,
        attestation.semantic_evidence_family_receipt_sha256s,
        attestation.breakdown_artifact_physical_sha256,
        attestation.breakdown_artifact_receipt_sha256,
        attestation.overlay_artifact_physical_sha256,
        attestation.overlay_identity_sha256,
        attestation.effective_route_identity_sha256,
        attestation.live_replay_receipt_sha256,
    )
    if observed != expected:
        raise SourceFetchV4Error(
            "A3 replay attestation differs from code, evidence, or effective routes"
        )


def write_a3_replay_attestation_v4(
    path: Path,
    attestation: A3ReplayAttestationV4,
) -> str:
    """Write the external attestation; callers must place it outside the repo."""

    if not isinstance(attestation, A3ReplayAttestationV4):
        raise TypeError("replay attestation writer requires a typed receipt")
    if not isinstance(path, Path):
        raise TypeError("replay attestation path must be a pathlib.Path")
    repository_root = Path(__file__).resolve().parents[1]
    resolved_target = path.resolve(strict=False)
    if resolved_target == repository_root or repository_root in resolved_target.parents:
        raise SourceFetchV4Error(
            "A3 replay attestation must remain outside the repository"
        )
    return _write_or_verify_artifact(
        path,
        {
            "receipt": asdict(attestation),
            "receipt_sha256": attestation.receipt_sha256,
            "schema": A3_REPLAY_ATTESTATION_ARTIFACT_SCHEMA_V4,
        },
    )


def load_a3_replay_attestation_v4(
    path: Path,
    *,
    context: PASourceExecutionContextV4,
    source_prep_code_identity: SourcePrepCodeIdentityV4,
) -> A3ReplayAttestationV4:
    if not isinstance(path, Path):
        raise TypeError("replay-attestation path must be a pathlib.Path")
    raw, envelope = load_canonical_json_snapshot(path)
    if raw != canonical_json_bytes(envelope) + b"\n" or set(envelope) != {
        "receipt",
        "receipt_sha256",
        "schema",
    }:
        raise SourceFetchV4Error("replay-attestation artifact is noncanonical")
    if envelope["schema"] != A3_REPLAY_ATTESTATION_ARTIFACT_SCHEMA_V4:
        raise SourceFetchV4Error("replay-attestation artifact schema drifted")
    payload = envelope["receipt"]
    if not isinstance(payload, Mapping):
        raise TypeError("replay-attestation receipt must be a mapping")
    value = dict(payload)
    raw_family_receipts = value.pop("semantic_evidence_family_receipt_sha256s", None)
    if not isinstance(raw_family_receipts, list):
        raise TypeError("replay-attestation family evidence must be a list")
    attestation = A3ReplayAttestationV4(
        semantic_evidence_family_receipt_sha256s=tuple(
            tuple(item) for item in raw_family_receipts
        ),
        **value,
    )
    if envelope["receipt_sha256"] != attestation.receipt_sha256:
        raise SourceFetchV4Error("replay-attestation receipt identity drifted")
    validate_a3_replay_attestation_v4(
        attestation,
        context=context,
        source_prep_code_identity=source_prep_code_identity,
    )
    return attestation


@dataclass(frozen=True)
class ExternalTransportEntryV4:
    locator: str
    relative_path: str
    observed_bytes: int
    observed_sha256: str
    cache_entry_receipt_sha256: str
    redirect_count: int
    redirect_kind: str
    redirect_target_path: str | None
    redirect_etag: str | None

    def __post_init__(self) -> None:
        if not self.locator.startswith("https://"):
            raise ValueError("external transport locator must use HTTPS")
        _canonical_path(self.relative_path)
        _require_positive_int(self.observed_bytes, "external transport bytes")
        _require_sha256(self.observed_sha256, "external transport SHA-256")
        _require_sha256(self.cache_entry_receipt_sha256, "external cache entry")
        if type(self.redirect_count) is not int or self.redirect_count not in {0, 1}:
            raise ValueError("external transport redirect count drifted")
        _require_nonempty(self.redirect_kind, "redirect kind")


@dataclass(frozen=True)
class ExternalTransportReceiptV4:
    execution_binding: PAExecutionBindingV4
    source_prep_code_identity_sha256: str
    entries: tuple[ExternalTransportEntryV4, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.execution_binding, PAExecutionBindingV4):
            raise TypeError("external transport requires an A3 binding")
        _require_sha256(
            self.source_prep_code_identity_sha256, "source-prep code identity"
        )
        locators = tuple(row.locator for row in self.entries)
        if (
            not self.entries
            or locators != tuple(sorted(locators, key=lambda item: item.encode("utf-8")))
            or len(set(locators)) != len(locators)
        ):
            raise ValueError("external transport entries are noncanonical")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            "weft1_external_transport_receipt_v4", self
        )


def build_external_transport_receipt_v4(
    *,
    execution_binding: PAExecutionBindingV4,
    source_prep_code_identity: SourcePrepCodeIdentityV4,
    observations: Sequence[object],
) -> ExternalTransportReceiptV4:
    if source_prep_code_identity.execution_binding != execution_binding:
        raise ValueError("transport code identity belongs to another A3 binding")
    entries = tuple(
        sorted(
            (
                ExternalTransportEntryV4(
                    locator=item.locator,
                    relative_path=item.relative_path,
                    observed_bytes=item.observed_bytes,
                    observed_sha256=item.observed_sha256,
                    cache_entry_receipt_sha256=item.cache_entry_receipt_sha256,
                    redirect_count=item.redirect_count,
                    redirect_kind=item.redirect_kind,
                    redirect_target_path=item.redirect_target_path,
                    redirect_etag=item.redirect_etag,
                )
                for item in observations
            ),
            key=lambda row: row.locator.encode("utf-8"),
        )
    )
    if source_prep_code_identity.mode == AUTHORITATIVE_MODE:
        expected = {
            "https://huggingface.co/datasets/allenai/dolma/resolve/"
            "7f48140530a023e9ea4c5cfb141160922727d4d3/urls/v1_7.txt",
            "https://olmo-data.org/dolma-v1_7/wiki/wiki-0000.json.gz",
            "https://olmo-data.org/dolma-v1_7/wiki/wiki-0001.json.gz",
        }
        if {row.locator for row in entries} != expected:
            raise ValueError("authoritative transport lacks its exact external set")
    return ExternalTransportReceiptV4(
        execution_binding=execution_binding,
        source_prep_code_identity_sha256=source_prep_code_identity.receipt_sha256,
        entries=entries,
    )


def write_external_transport_receipt_v4(
    path: Path,
    receipt: ExternalTransportReceiptV4,
) -> str:
    if not isinstance(receipt, ExternalTransportReceiptV4):
        raise TypeError("external transport writer requires a V4 receipt")
    return _write_or_verify_artifact(
        path,
        {
            "receipt": asdict(receipt),
            "receipt_sha256": receipt.receipt_sha256,
            "schema": "weft1_external_transport_receipt_artifact_v4",
        },
    )


@dataclass(frozen=True)
class PASourcePreparationResultV4:
    enumeration: UpstreamEnumerationReceiptV4
    selection: PASourcePrefixSelectionReceiptV4
    plan: SourceAssetDownloadPlanV4
    download: SourceCacheDownloadReceiptV4
    verified_cache: VerifiedLocalCacheV4
    enumeration_artifact_sha256: str
    selection_artifact_sha256: str
    manifest_artifact_sha256: str
    download_artifact_sha256: str

    def __post_init__(self) -> None:
        binding = self.enumeration.execution_binding
        if any(
            item.execution_binding != binding
            for item in (self.selection, self.plan, self.download, self.verified_cache)
        ):
            raise ValueError("V4 source preparation crosses A3 authority bindings")
        if (
            self.selection.enumeration_receipt_sha256 != self.enumeration.receipt_sha256
            or self.download.enumeration_receipt_sha256 != self.enumeration.receipt_sha256
            or self.selection.selection_plan_sha256 != self.plan.receipt_sha256
            or self.download.selection_plan_sha256 != self.plan.receipt_sha256
        ):
            raise ValueError("V4 source preparation receipt chain is broken")
        for name in (
            "enumeration_artifact_sha256",
            "selection_artifact_sha256",
            "manifest_artifact_sha256",
            "download_artifact_sha256",
        ):
            _require_sha256(getattr(self, name), name)


@dataclass(frozen=True)
class PAOnlineSourcePreparationResultV4:
    preparation: PASourcePreparationResultV4
    external_transport: ExternalTransportReceiptV4

    def __post_init__(self) -> None:
        if self.external_transport.execution_binding != self.preparation.enumeration.execution_binding:
            raise ValueError("online transport and source receipts use different A3 bindings")


def prepare_selected_source_cache_v4(
    *,
    enumeration: UpstreamEnumerationReceiptV4,
    cache_root: Path,
    receipt_root: Path,
    open_upstream: Callable[[UpstreamAssetV4], BinaryIO],
    allow_nonauthoritative_fixture: bool = False,
    _authoritative_capability: object | None = None,
) -> PASourcePreparationResultV4:
    if not isinstance(enumeration, UpstreamEnumerationReceiptV4):
        raise TypeError("source preparation requires a V4 enumeration")
    if (
        enumeration.authoritative
        and _authoritative_capability is not _AUTHORITATIVE_PREP_CAPABILITY
    ):
        raise SourceFetchV4Error(
            "authoritative cache preparation is private to online source prep"
        )
    if not enumeration.authoritative and not allow_nonauthoritative_fixture:
        raise SourceFetchV4Error("production source preparation requires AUTHORITATIVE")
    if not all(isinstance(path, Path) for path in (cache_root, receipt_root)):
        raise TypeError("source preparation roots must be pathlib.Path values")
    for root in (cache_root, receipt_root):
        assert_no_symlink_ancestors(root)
        root.mkdir(parents=True, exist_ok=True)
    resolved_cache = cache_root.resolve(strict=True)
    resolved_receipts = receipt_root.resolve(strict=True)
    if (
        resolved_cache == resolved_receipts
        or resolved_cache in resolved_receipts.parents
        or resolved_receipts in resolved_cache.parents
    ):
        raise SourceFetchV4Error("cache and receipt roots must be disjoint")
    plan, selection = select_required_asset_prefixes_v4(enumeration)
    enumeration_sha = _write_or_verify_artifact(
        resolved_receipts / ENUMERATION_ARTIFACT_NAME_V4,
        {
            "receipt": asdict(enumeration),
            "receipt_sha256": enumeration.receipt_sha256,
            "schema": UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V4,
        },
    )
    selection_sha = _write_or_verify_artifact(
        resolved_receipts / SELECTION_ARTIFACT_NAME_V4,
        {
            "receipt": asdict(selection),
            "receipt_sha256": selection.receipt_sha256,
            "schema": PREFIX_SELECTION_ARTIFACT_SCHEMA_V4,
        },
    )
    materialization = materialize_source_cache_v4(
        enumeration,
        plan,
        resolved_cache,
        open_upstream=open_upstream,
        allow_nonauthoritative_fixture=allow_nonauthoritative_fixture,
        resume_incomplete=True,
        _authoritative_capability=_authoritative_capability,
    )
    download, verified, manifest_sha = finalize_source_cache_v4(
        enumeration,
        materialization,
        resolved_cache,
        resolved_receipts / SOURCE_MANIFEST_NAME_V4,
        allow_nonauthoritative_fixture=allow_nonauthoritative_fixture,
        _authoritative_capability=_authoritative_capability,
    )
    download_sha = _write_or_verify_artifact(
        resolved_receipts / DOWNLOAD_ARTIFACT_NAME_V4,
        {
            "receipt": asdict(download),
            "receipt_sha256": download.receipt_sha256,
            "schema": DOWNLOAD_RECEIPT_ARTIFACT_SCHEMA_V4,
        },
    )
    return PASourcePreparationResultV4(
        enumeration=enumeration,
        selection=selection,
        plan=plan,
        download=download,
        verified_cache=verified,
        enumeration_artifact_sha256=enumeration_sha,
        selection_artifact_sha256=selection_sha,
        manifest_artifact_sha256=manifest_sha,
        download_artifact_sha256=download_sha,
    )


def prepare_pa_sources_online_v4(
    *,
    context: PASourceExecutionContextV4,
    cache_root: Path,
    transport_cache_root: Path,
    receipt_root: Path,
    source_prep_code_identity: SourcePrepCodeIdentityV4,
    replay_attestation: A3ReplayAttestationV4,
) -> PAOnlineSourcePreparationResultV4:
    """Run the authoritative A3/V4 fetch boundary; no corpus parsing occurs."""

    if not isinstance(context, PASourceExecutionContextV4):
        raise TypeError("online source preparation requires a V4 context")
    if (
        not isinstance(source_prep_code_identity, SourcePrepCodeIdentityV4)
        or source_prep_code_identity.mode != AUTHORITATIVE_MODE
        or source_prep_code_identity.execution_binding != context.binding
    ):
        raise SourceFetchV4Error("online source preparation requires attested A3 code")
    # This external, clean-commit receipt is validated before a network cache is
    # constructed, before repository enumeration, and therefore before any
    # finite download plan can exist.
    validate_a3_replay_attestation_v4(
        replay_attestation,
        context=context,
        source_prep_code_identity=source_prep_code_identity,
    )
    roots = (cache_root, transport_cache_root, receipt_root)
    if any(not isinstance(root, Path) for root in roots):
        raise TypeError("online source roots must be pathlib.Path values")
    resolved = tuple(root.resolve(strict=False) for root in roots)
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise SourceFetchV4Error("online source roots must be pairwise disjoint")
    external_cache = ExternalResourceCacheV3(transport_cache_root / "external")
    enumeration = _enumerate_authoritative_upstream_assets_v4(
        context=context,
        open_resource=external_cache.open,
        replay_attestation=replay_attestation,
        source_prep_code_identity=source_prep_code_identity,
    )
    opener = PinnedHuggingFaceAssetOpenerV3(
        transport_cache_root / "huggingface",
        external_cache=external_cache,
    )
    preparation = prepare_selected_source_cache_v4(
        enumeration=enumeration,
        cache_root=cache_root,
        receipt_root=receipt_root,
        open_upstream=lambda asset: _open_v4_with_v3_opener(opener, asset),
        _authoritative_capability=_AUTHORITATIVE_PREP_CAPABILITY,
    )
    external_transport = build_external_transport_receipt_v4(
        execution_binding=context.binding,
        source_prep_code_identity=source_prep_code_identity,
        observations=external_cache.observations,
    )
    return PAOnlineSourcePreparationResultV4(
        preparation=preparation,
        external_transport=external_transport,
    )


__all__ = [
    "A3ReplayAttestationV4",
    "A3_REPLAY_ATTESTATION_ARTIFACT_SCHEMA_V4",
    "A3_REPLAY_ATTESTATION_SCHEMA_V4",
    "AUTHORITATIVE_MODE",
    "DOWNLOAD_ARTIFACT_NAME_V4",
    "ENUMERATION_ARTIFACT_NAME_V4",
    "EXTERNAL_TRANSPORT_ARTIFACT_NAME_V4",
    "ExternalTransportReceiptV4",
    "FIXTURE_MODE",
    "FamilyEnumerationV4",
    "FamilyPrefixSelectionV4",
    "PAExecutionBindingV4",
    "PAOnlineSourcePreparationResultV4",
    "PASourceExecutionContextV4",
    "PASourcePrefixSelectionReceiptV4",
    "PASourcePreparationResultV4",
    "SELECTION_ARTIFACT_NAME_V4",
    "SOURCE_MANIFEST_NAME_V4",
    "SourceAssetDownloadPlanV4",
    "SourceCacheAssetV4",
    "SourceCacheDownloadReceiptV4",
    "SourceCacheManifestV4",
    "SourceFetchV4Error",
    "SourcePrepCodeIdentityV4",
    "SourcePrepImplementationFileV4",
    "SOURCE_PREP_IMPLEMENTATION_REPO_PATHS_V4",
    "UpstreamAssetV4",
    "UpstreamEnumerationReceiptV4",
    "build_external_transport_receipt_v4",
    "enumerate_upstream_assets_v4",
    "load_pa_source_execution_context_v4",
    "load_a3_replay_attestation_v4",
    "locator_matches_effective_route_v4",
    "prepare_pa_sources_online_v4",
    "prepare_selected_source_cache_v4",
    "select_required_asset_prefixes_v4",
    "validate_a3_replay_attestation_v4",
    "write_a3_replay_attestation_v4",
    "write_external_transport_receipt_v4",
]
