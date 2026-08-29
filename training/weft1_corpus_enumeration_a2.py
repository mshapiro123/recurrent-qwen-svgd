"""Fail-closed upstream asset enumeration for WEFT-1 corpus P-A.

This layer observes repository metadata only.  It never downloads an asset and
does not import a network client.  Production supplies a pinned Hugging Face
``list_repo_tree`` callable and the separately pinned Dolma v1.7 external wiki
locator enumeration.  Tests supply offline callables with the same interface.

An authoritative receipt is deliberately stricter than a fixture receipt:
every one of the seven A1 source families must be present, and both its asset
count and its route-declared available-byte measure must reconcile exactly.
Fixture mode remains complete across all seven families but is branded
``NONAUTHORITATIVE_FIXTURE`` and may use smaller inventories.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Sequence

from training.weft1_corpus_a2 import execution_authority_v3_bound_sha256
from training.weft1_corpus_sources_a2 import (
    SOURCE_ROUTE_MANIFEST_SHA256,
    ExactSourceRouteV3,
    SourceCacheAssetV3,
    asset_order_digest_v3,
    load_exact_source_routes_v3,
)
from training.weft1_gtok_a1_contract import (
    SOURCE_FAMILIES,
    SOURCE_ROUTE_MANIFEST_PATH,
    SourceRouteBindingV2,
    load_source_route_manifest,
)
from training.weft1_gtok_contract import canonical_json_bytes
from training.weft1_strict_io import (
    assert_no_symlink_ancestors,
    load_canonical_json_snapshot,
)


UPSTREAM_ENUMERATION_SCHEMA_V3 = "weft1_upstream_enumeration_receipt_v3"
UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V3 = (
    "weft1_upstream_enumeration_receipt_artifact_v3"
)
AUTHORITATIVE_MODE = "AUTHORITATIVE"
FIXTURE_MODE = "NONAUTHORITATIVE_FIXTURE"
ENUMERATION_MODES = (AUTHORITATIVE_MODE, FIXTURE_MODE)

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_FACTORY_SENTINEL = object()
_EXTERNAL_LISTING_FACTORY_SENTINEL = object()
_WIKI_LOCATOR_MANIFEST_PATH = "urls/v1_7.txt"
_WIKI_LOCATOR_MANIFEST_SHA256 = (
    "9fa8c2f0eb57149ff7914b35ca2ffb8da221c02786d712bcba5f6c39d294b49e"
)
_WIKI_LOCATOR_MANIFEST_BYTES = 171_893
_WIKI_SELECTED_LOCATORS = (
    "https://olmo-data.org/dolma-v1_7/wiki/wiki-0000.json.gz",
    "https://olmo-data.org/dolma-v1_7/wiki/wiki-0001.json.gz",
)
_DIRECT_AVAILABLE_BYTE_BASES = (
    "pinned repository compressed asset bytes",
    "pinned repository parquet bytes",
)
_FIXTURE_ENUMERATOR_BINDING_SHA256 = hashlib.sha256(
    b"weft1:nonauthoritative-injected-enumerator:v3"
).hexdigest()
_HUGGINGFACE_HUB_DISTRIBUTION = "huggingface-hub"
_HUGGINGFACE_HUB_VERSION = "1.24.0"


def _require_nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive exact integer")
    return value


def _field(value: object, name: str, *, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _canonical_hf_path(value: str) -> str:
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
    """Translate the route ledger's deliberately small ``*`` glob language."""

    _require_nonempty(pattern, "asset selector glob")
    if any(token in pattern for token in ("?", "[", "]", "**")):
        raise ValueError("route selector uses an unsupported glob operator")
    return re.compile("^" + re.escape(pattern).replace(r"\*", "[^/]*") + "$")


def _selector_target(route: ExactSourceRouteV3) -> str:
    if route.source_family == "wikipedia_wikibooks":
        parts = route.asset_selector.split(" -> ")
        if len(parts) != 2 or parts[0] != "urls/v1_7.txt":
            raise ValueError("Wikipedia external selector syntax drifted")
        return parts[1]
    return route.asset_selector


def locator_matches_route_v3(route: ExactSourceRouteV3, locator: str) -> bool:
    """Return whether ``locator`` is selected by the exact A1 route literal."""

    if not isinstance(route, ExactSourceRouteV3):
        raise TypeError("route must be an ExactSourceRouteV3")
    _require_nonempty(locator, "asset locator")
    return _glob_regex(_selector_target(route)).fullmatch(locator) is not None


def _validate_selected_locator(
    route: ExactSourceRouteV3,
    locator: str,
    ordinal: int,
) -> None:
    if not locator_matches_route_v3(route, locator):
        raise ValueError(
            f"enumerated locator falls outside the pinned selector for {route.source_family}"
        )
    # Reuse the source-cache layer's independently bound route/locator checker.
    # The placeholder local path and hash are not emitted into this receipt.
    SourceCacheAssetV3(
        source_family=route.source_family,
        repository=route.repository,
        config=route.config,
        revision=route.revision,
        split=route.split,
        asset_locator=locator,
        relative_path=f"enumeration/{route.source_family}/{ordinal:08d}.asset",
        bytes=1,
        sha256="0" * 64,
    )


@dataclass(frozen=True)
class UpstreamAssetV3:
    """One selected upstream file and the exact metadata used to identify it."""

    source_family: str
    repository: str
    config: str
    revision: str
    split: str
    asset_locator: str
    upstream_bytes: int
    blob_identity_kind: str
    blob_identity: str
    content_sha256: str | None = None

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
        elif self.blob_identity_kind in {
            "git_sha256",
            "content_sha256",
        }:
            _require_sha256(self.blob_identity, "blob_identity")
        else:
            raise ValueError("unsupported upstream blob identity kind")
        if self.content_sha256 is not None:
            _require_sha256(self.content_sha256, "content_sha256")
        if (
            self.blob_identity_kind == "content_sha256"
            and self.content_sha256 != self.blob_identity
        ):
            raise ValueError("external content identity and content SHA disagree")

    @property
    def asset_identity_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_upstream_asset_identity_v3", self
        )


@dataclass(frozen=True)
class ExternalLocatorAssetV3:
    """One asset in the separately pinned Dolma v1.7 wiki URL manifest."""

    locator: str
    upstream_bytes: int
    content_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.locator, "external locator")
        _require_positive_int(self.upstream_bytes, "external upstream_bytes")
        _require_sha256(self.content_sha256, "external content_sha256")


@dataclass(frozen=True, init=False)
class ExternalLocatorListingV3:
    """Factory-minted observation of the pinned external wiki URL list."""

    source_family: str
    external_locator_manifest_sha256: str
    external_locator_manifest_locator: str
    external_locator_manifest_bytes: int
    available_bytes: int
    available_bytes_basis: str
    assets: tuple[ExternalLocatorAssetV3, ...]
    parent_manifest_verified: bool

    def __new__(cls) -> "ExternalLocatorListingV3":
        raise TypeError(
            "ExternalLocatorListingV3 is factory-minted from parent bytes"
        )

    @classmethod
    def _validated(
        cls,
        *,
        source_family: str,
        external_locator_manifest_sha256: str,
        external_locator_manifest_locator: str,
        external_locator_manifest_bytes: int,
        available_bytes: int,
        available_bytes_basis: str,
        assets: tuple[ExternalLocatorAssetV3, ...],
        parent_manifest_verified: bool,
        sentinel: object,
    ) -> "ExternalLocatorListingV3":
        if sentinel is not _EXTERNAL_LISTING_FACTORY_SENTINEL:
            raise PermissionError("external locator listings are factory-only")
        assets = tuple(sorted(assets, key=lambda item: item.locator.encode("utf-8")))
        instance = object.__new__(cls)
        for name, value in (
            ("source_family", source_family),
            (
                "external_locator_manifest_sha256",
                external_locator_manifest_sha256,
            ),
            ("external_locator_manifest_locator", external_locator_manifest_locator),
            ("external_locator_manifest_bytes", external_locator_manifest_bytes),
            ("available_bytes", available_bytes),
            ("available_bytes_basis", available_bytes_basis),
            ("assets", assets),
            ("parent_manifest_verified", parent_manifest_verified),
        ):
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    @classmethod
    def fixture(
        cls,
        *,
        source_family: str,
        external_locator_manifest_sha256: str,
        available_bytes: int,
        available_bytes_basis: str,
        assets: tuple[ExternalLocatorAssetV3, ...],
    ) -> "ExternalLocatorListingV3":
        """Mint an explicitly nonauthoritative offline fixture listing."""

        return cls._validated(
            source_family=source_family,
            external_locator_manifest_sha256=external_locator_manifest_sha256,
            external_locator_manifest_locator="fixture://urls/v1_7.txt",
            external_locator_manifest_bytes=1,
            available_bytes=available_bytes,
            available_bytes_basis=available_bytes_basis,
            assets=assets,
            parent_manifest_verified=False,
            sentinel=_EXTERNAL_LISTING_FACTORY_SENTINEL,
        )

    def __post_init__(self) -> None:
        if self.source_family != "wikipedia_wikibooks":
            raise ValueError("external locator listings are only valid for Wikipedia")
        _require_sha256(
            self.external_locator_manifest_sha256,
            "external_locator_manifest_sha256",
        )
        _require_nonempty(
            self.external_locator_manifest_locator,
            "external_locator_manifest_locator",
        )
        _require_positive_int(
            self.external_locator_manifest_bytes,
            "external_locator_manifest_bytes",
        )
        _require_positive_int(self.available_bytes, "external available_bytes")
        _require_nonempty(self.available_bytes_basis, "available_bytes_basis")
        if not isinstance(self.assets, tuple) or not self.assets:
            raise ValueError("external locator listing requires a nonempty asset tuple")
        if any(not isinstance(asset, ExternalLocatorAssetV3) for asset in self.assets):
            raise TypeError("external listing contains an untyped asset")
        locators = tuple(asset.locator for asset in self.assets)
        if len(locators) != len(set(locators)):
            raise ValueError("external locator listing repeats an asset")
        if type(self.parent_manifest_verified) is not bool:
            raise TypeError("parent_manifest_verified must be an exact bool")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_external_locator_listing_v3", self
        )


def _read_stream_bytes(
    opened: BinaryIO,
    *,
    max_bytes: int | None,
    label: str,
) -> tuple[int, str, bytes | None]:
    digest = hashlib.sha256()
    total = 0
    captured = bytearray() if max_bytes is not None else None
    while True:
        chunk = opened.read(8 * 1024 * 1024)
        if chunk is None or chunk == b"":
            break
        if not isinstance(chunk, bytes):
            raise ValueError(f"{label} reader returned a non-bytes chunk")
        total += len(chunk)
        if max_bytes is not None and total > max_bytes:
            raise ValueError(f"{label} exceeds its bounded byte limit")
        digest.update(chunk)
        if captured is not None:
            captured.extend(chunk)
    if total < 1:
        raise ValueError(f"{label} is empty")
    return total, digest.hexdigest(), None if captured is None else bytes(captured)


def read_pinned_external_locator_listing_v3(
    *,
    route: ExactSourceRouteV3,
    declared: SourceRouteBindingV2,
    open_resource: Callable[[str], BinaryIO],
    expected_manifest_sha256: str | None = None,
    allow_nonauthoritative_fixture_hash: bool = False,
) -> ExternalLocatorListingV3:
    """Read, parent-hash, parse, and observe the two pinned wiki assets.

    The injected opener is the transport only: identity is derived here from
    the bytes read through it.  Production always uses the route-ledger SHA.
    Tests may opt into a different parent hash, but the returned listing is
    then branded nonauthoritative and cannot enter an AUTHORITATIVE receipt.
    The opener may tee the two large asset streams into the later cache.
    """

    if not isinstance(route, ExactSourceRouteV3) or not isinstance(
        declared, SourceRouteBindingV2
    ):
        raise TypeError("external listing reader requires typed route rows")
    if route.source_family != "wikipedia_wikibooks" or (
        declared.source_family != route.source_family
    ):
        raise ValueError("external listing reader is only for the pinned wiki route")
    if not callable(open_resource):
        raise TypeError("open_resource must be callable")
    ledger_sha = declared.external_locator_manifest_sha256
    if ledger_sha != _WIKI_LOCATOR_MANIFEST_SHA256:
        raise ValueError("wiki locator parent hash differs from the frozen literal")
    expected = ledger_sha if expected_manifest_sha256 is None else expected_manifest_sha256
    _require_sha256(expected, "expected external locator manifest SHA-256")
    if expected != ledger_sha and not allow_nonauthoritative_fixture_hash:
        raise ValueError("production external listing must use the ledger parent hash")
    manifest_locator = (
        f"https://huggingface.co/datasets/{route.repository}/resolve/"
        f"{route.revision}/{_WIKI_LOCATOR_MANIFEST_PATH}"
    )
    opened = open_resource(manifest_locator)
    if not hasattr(opened, "read"):
        raise TypeError("open_resource returned no binary reader")
    with closing(opened):
        manifest_bytes, manifest_sha, raw = _read_stream_bytes(
            opened,
            max_bytes=64 * 1024 * 1024,
            label="wiki locator parent",
        )
    if manifest_sha != expected:
        raise ValueError("wiki locator parent SHA-256 drifted")
    if expected == ledger_sha and manifest_bytes != _WIKI_LOCATOR_MANIFEST_BYTES:
        raise ValueError("wiki locator parent byte count drifted")
    assert raw is not None
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw or b"\r" in raw:
        raise ValueError("wiki locator parent must be BOM-free LF-only UTF-8")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("wiki locator parent is not strict UTF-8") from error
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise ValueError("wiki locator parent must have exactly one final LF")
    lines = text[:-1].split("\n")
    if any(not line or line != line.strip() for line in lines):
        raise ValueError("wiki locator parent contains blank or padded rows")
    if len(lines) != len(set(lines)):
        raise ValueError("wiki locator parent repeats a URL")
    selected = tuple(
        line
        for line in lines
        if line.startswith("https://olmo-data.org/dolma-v1_7/wiki/wiki-")
        and line.endswith(".json.gz")
    )
    if selected != _WIKI_SELECTED_LOCATORS:
        raise ValueError("wiki locator parent selected URL set/order drifted")

    assets: list[ExternalLocatorAssetV3] = []
    for locator in selected:
        opened_asset = open_resource(locator)
        if not hasattr(opened_asset, "read"):
            raise TypeError("open_resource returned no binary asset reader")
        with closing(opened_asset):
            asset_bytes, asset_sha, unused = _read_stream_bytes(
                opened_asset,
                max_bytes=None,
                label="wiki external asset",
            )
        assert unused is None
        assets.append(
            ExternalLocatorAssetV3(
                locator=locator,
                upstream_bytes=asset_bytes,
                content_sha256=asset_sha,
            )
        )
    return ExternalLocatorListingV3._validated(
        source_family=route.source_family,
        external_locator_manifest_sha256=manifest_sha,
        external_locator_manifest_locator=manifest_locator,
        external_locator_manifest_bytes=manifest_bytes,
        available_bytes=declared.available_bytes,
        available_bytes_basis=declared.available_bytes_basis,
        assets=tuple(assets),
        parent_manifest_verified=expected == ledger_sha,
        sentinel=_EXTERNAL_LISTING_FACTORY_SENTINEL,
    )


def _canonical_upstream_asset_order(
    assets: Sequence[UpstreamAssetV3],
) -> tuple[UpstreamAssetV3, ...]:
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
        raise TypeError("upstream assets must be a typed sequence")
    if any(not isinstance(asset, UpstreamAssetV3) for asset in assets):
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
class FamilyEnumerationV3:
    """One complete selected inventory plus its A1 accounting comparison."""

    route: ExactSourceRouteV3
    declared_asset_count: int
    declared_available_bytes: int
    available_bytes_basis: str
    observed_available_bytes: int
    assets: tuple[UpstreamAssetV3, ...]
    external_locator_manifest_sha256: str | None = None
    external_locator_manifest_bytes: int | None = None
    external_locator_listing_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route, ExactSourceRouteV3):
            raise TypeError("family enumeration requires an exact source route")
        _require_positive_int(self.declared_asset_count, "declared_asset_count")
        _require_positive_int(
            self.declared_available_bytes, "declared_available_bytes"
        )
        _require_nonempty(self.available_bytes_basis, "available_bytes_basis")
        _require_positive_int(
            self.observed_available_bytes, "observed_available_bytes"
        )
        if not isinstance(self.assets, tuple) or not self.assets:
            raise ValueError("family enumeration requires selected upstream assets")
        if self.assets != _canonical_upstream_asset_order(self.assets):
            raise ValueError("family assets are not in deterministic seeded order")
        for ordinal, asset in enumerate(self.assets):
            if asset.source_family != self.route.source_family:
                raise ValueError("family enumeration mixes source families")
            if (
                asset.repository,
                asset.config,
                asset.revision,
                asset.split,
            ) != (
                self.route.repository,
                self.route.config,
                self.route.revision,
                self.route.split,
            ):
                raise ValueError("enumerated asset route identity drifted")
            _validate_selected_locator(self.route, asset.asset_locator, ordinal)
        if self.route.source_family == "wikipedia_wikibooks":
            if (
                self.external_locator_manifest_sha256 is None
                or self.external_locator_manifest_bytes is None
                or self.external_locator_listing_receipt_sha256 is None
            ):
                raise ValueError(
                    "Wikipedia requires its parent-hashed external locator receipt"
                )
            _require_sha256(
                self.external_locator_manifest_sha256,
                "external_locator_manifest_sha256",
            )
            _require_positive_int(
                self.external_locator_manifest_bytes,
                "external_locator_manifest_bytes",
            )
            _require_sha256(
                self.external_locator_listing_receipt_sha256,
                "external_locator_listing_receipt_sha256",
            )
        elif any(
            value is not None
            for value in (
                self.external_locator_manifest_sha256,
                self.external_locator_manifest_bytes,
                self.external_locator_listing_receipt_sha256,
            )
        ):
            raise ValueError("only Wikipedia may bind an external locator manifest")
        if self.available_bytes_basis in _DIRECT_AVAILABLE_BYTE_BASES:
            if self.observed_available_bytes != self.asset_payload_bytes:
                raise ValueError(
                    "direct repository available bytes must equal selected asset sizes"
                )

    @property
    def source_family(self) -> str:
        return self.route.source_family

    @property
    def asset_payload_bytes(self) -> int:
        return sum(asset.upstream_bytes for asset in self.assets)

    @property
    def asset_count_matches_declared(self) -> bool:
        return len(self.assets) == self.declared_asset_count

    @property
    def available_bytes_matches_declared(self) -> bool:
        return self.observed_available_bytes == self.declared_available_bytes

    @property
    def enumeration_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_family_upstream_enumeration_v3", self
        )


@dataclass(frozen=True, init=False)
class UpstreamEnumerationReceiptV3:
    """Factory-minted, complete enumeration receipt for all seven families."""

    schema: str
    mode: str
    enumerator_binding_sha256: str
    source_route_manifest_sha256: str
    families: tuple[FamilyEnumerationV3, ...]

    def __new__(cls) -> "UpstreamEnumerationReceiptV3":
        raise TypeError(
            "UpstreamEnumerationReceiptV3 is factory-minted after injected enumeration"
        )

    @classmethod
    def _validated(
        cls,
        *,
        mode: str,
        enumerator_binding_sha256: str,
        families: tuple[FamilyEnumerationV3, ...],
        sentinel: object,
    ) -> "UpstreamEnumerationReceiptV3":
        if sentinel is not _RECEIPT_FACTORY_SENTINEL:
            raise PermissionError("upstream enumeration receipts are factory-only")
        instance = object.__new__(cls)
        object.__setattr__(instance, "schema", UPSTREAM_ENUMERATION_SCHEMA_V3)
        object.__setattr__(instance, "mode", mode)
        object.__setattr__(
            instance,
            "enumerator_binding_sha256",
            enumerator_binding_sha256,
        )
        object.__setattr__(
            instance,
            "source_route_manifest_sha256",
            SOURCE_ROUTE_MANIFEST_SHA256,
        )
        object.__setattr__(instance, "families", families)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        if self.schema != UPSTREAM_ENUMERATION_SCHEMA_V3:
            raise ValueError("unexpected upstream enumeration schema")
        if self.mode not in ENUMERATION_MODES:
            raise ValueError("enumeration mode must be explicit")
        _require_sha256(self.enumerator_binding_sha256, "enumerator binding")
        if (
            self.mode == FIXTURE_MODE
            and self.enumerator_binding_sha256
            != _FIXTURE_ENUMERATOR_BINDING_SHA256
        ):
            raise ValueError("fixture enumeration uses a foreign enumerator binding")
        if self.source_route_manifest_sha256 != SOURCE_ROUTE_MANIFEST_SHA256:
            raise ValueError("enumeration is bound to the wrong source-route ledger")
        if not isinstance(self.families, tuple):
            raise TypeError("enumeration families must be a tuple")
        if tuple(family.source_family for family in self.families) != SOURCE_FAMILIES:
            raise ValueError("enumeration must cover every family in canonical order")
        if self.mode == AUTHORITATIVE_MODE:
            mismatches = tuple(
                family.source_family
                for family in self.families
                if not (
                    family.asset_count_matches_declared
                    and family.available_bytes_matches_declared
                )
            )
            if mismatches:
                raise ValueError(
                    "authoritative enumeration disagrees with route declarations: "
                    + ", ".join(mismatches)
                )

    @property
    def authoritative(self) -> bool:
        return self.mode == AUTHORITATIVE_MODE

    @property
    def total_asset_count(self) -> int:
        return sum(len(family.assets) for family in self.families)

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(self.schema, self)


def _write_canonical_receipt_artifact(path: Path, value: Mapping[str, object]) -> str:
    if not isinstance(path, Path):
        raise TypeError("receipt artifact path must be a pathlib.Path")
    assert_no_symlink_ancestors(path)
    if path.exists():
        raise ValueError("refusing to overwrite a receipt artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise ValueError("stale receipt artifact partial exists")
    raw = canonical_json_bytes(value) + b"\n"
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


def write_upstream_enumeration_receipt_v3(
    receipt: UpstreamEnumerationReceiptV3,
    path: Path,
) -> str:
    """Persist a canonical, parent-rehashable enumeration envelope."""

    if not isinstance(receipt, UpstreamEnumerationReceiptV3):
        raise TypeError("enumeration artifact requires a factory receipt")
    return _write_canonical_receipt_artifact(
        path,
        {
            "receipt": asdict(receipt),
            "receipt_sha256": receipt.receipt_sha256,
            "schema": UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V3,
        },
    )


def load_upstream_enumeration_receipt_v3(
    path: Path,
    route_manifest_path: Path = SOURCE_ROUTE_MANIFEST_PATH,
) -> UpstreamEnumerationReceiptV3:
    """Reconstruct factory authority from canonical bytes without networking."""

    artifact_bytes, payload = load_canonical_json_snapshot(path)
    if artifact_bytes != canonical_json_bytes(payload) + b"\n":
        raise ValueError("enumeration artifact is not canonical sorted JSON")
    if set(payload) != {"receipt", "receipt_sha256", "schema"} or payload.get(
        "schema"
    ) != UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V3:
        raise ValueError("enumeration artifact envelope drifted")
    raw_receipt = payload["receipt"]
    if not isinstance(raw_receipt, Mapping) or set(raw_receipt) != {
        "enumerator_binding_sha256",
        "families",
        "mode",
        "schema",
        "source_route_manifest_sha256",
    }:
        raise ValueError("enumeration artifact receipt shape drifted")
    if raw_receipt.get("schema") != UPSTREAM_ENUMERATION_SCHEMA_V3 or raw_receipt.get(
        "source_route_manifest_sha256"
    ) != SOURCE_ROUTE_MANIFEST_SHA256:
        raise ValueError("enumeration artifact authority edge drifted")
    raw_families = raw_receipt["families"]
    if not isinstance(raw_families, list):
        raise TypeError("enumeration artifact families must be a list")
    exact_routes = load_exact_source_routes_v3(route_manifest_path)
    if len(raw_families) != len(exact_routes):
        raise ValueError("enumeration artifact family count drifted")
    families: list[FamilyEnumerationV3] = []
    family_keys = {
        "assets",
        "available_bytes_basis",
        "declared_asset_count",
        "declared_available_bytes",
        "external_locator_listing_receipt_sha256",
        "external_locator_manifest_bytes",
        "external_locator_manifest_sha256",
        "observed_available_bytes",
        "route",
    }
    asset_keys = {
        "asset_locator",
        "blob_identity",
        "blob_identity_kind",
        "config",
        "content_sha256",
        "repository",
        "revision",
        "source_family",
        "split",
        "upstream_bytes",
    }
    for raw_family, route in zip(raw_families, exact_routes, strict=True):
        if not isinstance(raw_family, Mapping) or set(raw_family) != family_keys:
            raise ValueError("enumeration artifact family shape drifted")
        if canonical_json_bytes(raw_family["route"]) != canonical_json_bytes(
            asdict(route)
        ):
            raise ValueError("enumeration artifact route differs from current ledger")
        raw_assets = raw_family["assets"]
        if not isinstance(raw_assets, list):
            raise TypeError("enumeration artifact assets must be a list")
        assets: list[UpstreamAssetV3] = []
        for raw_asset in raw_assets:
            if not isinstance(raw_asset, Mapping) or set(raw_asset) != asset_keys:
                raise ValueError("enumeration artifact asset shape drifted")
            assets.append(UpstreamAssetV3(**dict(raw_asset)))
        families.append(
            FamilyEnumerationV3(
                route=route,
                declared_asset_count=raw_family["declared_asset_count"],  # type: ignore[arg-type]
                declared_available_bytes=raw_family["declared_available_bytes"],  # type: ignore[arg-type]
                available_bytes_basis=raw_family["available_bytes_basis"],  # type: ignore[arg-type]
                observed_available_bytes=raw_family["observed_available_bytes"],  # type: ignore[arg-type]
                assets=tuple(assets),
                external_locator_manifest_sha256=raw_family[
                    "external_locator_manifest_sha256"
                ],  # type: ignore[arg-type]
                external_locator_manifest_bytes=raw_family[
                    "external_locator_manifest_bytes"
                ],  # type: ignore[arg-type]
                external_locator_listing_receipt_sha256=raw_family[
                    "external_locator_listing_receipt_sha256"
                ],  # type: ignore[arg-type]
            )
        )
    receipt = UpstreamEnumerationReceiptV3._validated(
        mode=raw_receipt["mode"],  # type: ignore[arg-type]
        enumerator_binding_sha256=raw_receipt[
            "enumerator_binding_sha256"
        ],  # type: ignore[arg-type]
        families=tuple(families),
        sentinel=_RECEIPT_FACTORY_SENTINEL,
    )
    if payload["receipt_sha256"] != receipt.receipt_sha256:
        raise ValueError("enumeration artifact receipt SHA-256 drifted")
    return receipt


def _route_pairs(
    route_manifest_path: Path,
) -> tuple[tuple[ExactSourceRouteV3, SourceRouteBindingV2], ...]:
    exact_routes = load_exact_source_routes_v3(route_manifest_path)
    ledger = load_source_route_manifest(route_manifest_path)
    if ledger.manifest_sha256 != SOURCE_ROUTE_MANIFEST_SHA256:
        raise ValueError("source-route ledger identity drifted")
    if tuple(route.source_family for route in ledger.routes) != SOURCE_FAMILIES:
        raise ValueError("source-route ledger is incomplete")
    pairs = tuple(zip(exact_routes, ledger.routes, strict=True))
    for exact, declared in pairs:
        if exact.source_family != declared.source_family:
            raise ValueError("exact and declared source routes are misaligned")
        if exact.a1_route_receipt_sha256 != declared.receipt_sha256:
            raise ValueError("exact source route is bound to a different A1 row")
    return pairs


def _normalise_hf_entry(
    route: ExactSourceRouteV3,
    raw: object,
) -> UpstreamAssetV3 | None:
    entry_type = _field(raw, "type")
    if entry_type is None:
        # huggingface_hub 1.24 RepoFile/RepoFolder values do not expose a
        # synthetic ``type`` attribute.  Their fail-closed structural split is
        # RepoFile(size, blob_id) versus RepoFolder(tree_id).
        has_file_shape = (
            _field(raw, "size") is not None
            and _field(raw, "blob_id") is not None
        )
        has_folder_shape = _field(raw, "tree_id") is not None
        if has_file_shape == has_folder_shape:
            raise ValueError("Hugging Face tree entry has an ambiguous shape")
        entry_type = "file" if has_file_shape else "directory"
    if entry_type in {"directory", "dir", "tree"}:
        return None
    if entry_type not in {"file", "blob"}:
        raise ValueError("Hugging Face tree entry lacks an exact file type")
    path = _field(raw, "path", default=_field(raw, "rfilename"))
    if not isinstance(path, str):
        raise ValueError("Hugging Face file entry lacks a path")
    path = _canonical_hf_path(path)
    if not locator_matches_route_v3(route, path):
        return None
    size = _field(raw, "size")
    _require_positive_int(size, "Hugging Face file size")  # type: ignore[arg-type]
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
        if not isinstance(raw_sha, str):
            raise ValueError("selected LFS asset lacks a content SHA-256")
        content_sha256 = _require_sha256(raw_sha, "LFS content SHA-256")
        lfs_size = _field(lfs, "size")
        if lfs_size is not None and lfs_size != size:
            raise ValueError("Hugging Face file and LFS sizes disagree")
    xet_hash = _field(raw, "xet_hash")
    if xet_hash is not None:
        if not isinstance(xet_hash, str) or _SHA256.fullmatch(xet_hash) is None:
            raise ValueError("selected Xet asset exposes an invalid Xet identity")
        if content_sha256 is None:
            # The pinned client exposes a Xet Merkle identity, not a specified
            # digest of the resolved file bytes.  Until that derivation is
            # locally implemented and replay-tested, treating it as raw SHA-256
            # would be an unverifiable claim.
            raise ValueError(
                "selected Xet-only asset lacks a verifiable raw-content identity"
            )

    return UpstreamAssetV3(
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
    )


def _enumerate_hf_family(
    route: ExactSourceRouteV3,
    raw_tree: Sequence[object],
) -> tuple[UpstreamAssetV3, ...]:
    assets = tuple(
        asset
        for raw in raw_tree
        if (asset := _normalise_hf_entry(route, raw)) is not None
    )
    if not assets:
        raise ValueError(
            f"pinned selector enumerated no assets for {route.source_family}"
        )
    for ordinal, asset in enumerate(assets):
        _validate_selected_locator(route, asset.asset_locator, ordinal)
    return _canonical_upstream_asset_order(assets)


def _enumerate_external_family(
    route: ExactSourceRouteV3,
    declared: SourceRouteBindingV2,
    enumerate_external_locators: Callable[..., ExternalLocatorListingV3],
    mode: str,
) -> tuple[tuple[UpstreamAssetV3, ...], int, str, int, str]:
    expected_manifest = declared.external_locator_manifest_sha256
    if expected_manifest is None:
        raise ValueError("Wikipedia route lacks its external locator manifest")
    listing = enumerate_external_locators(
        route=route,
        expected_manifest_sha256=expected_manifest,
    )
    if not isinstance(listing, ExternalLocatorListingV3):
        raise TypeError("external locator enumerator returned an untyped listing")
    if mode == AUTHORITATIVE_MODE and not listing.parent_manifest_verified:
        raise ValueError(
            "authoritative enumeration requires a parent-hash-verified "
            "external locator listing"
        )
    if listing.external_locator_manifest_sha256 != expected_manifest:
        raise ValueError("external locator manifest identity drifted")
    if listing.available_bytes_basis != declared.available_bytes_basis:
        raise ValueError("external available-byte basis drifted")
    assets: list[UpstreamAssetV3] = []
    for ordinal, item in enumerate(listing.assets):
        _validate_selected_locator(route, item.locator, ordinal)
        assets.append(
            UpstreamAssetV3(
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
            )
        )
    return (
        _canonical_upstream_asset_order(tuple(assets)),
        listing.available_bytes,
        expected_manifest,
        listing.external_locator_manifest_bytes,
        listing.receipt_sha256,
    )


def _enumerate_upstream_assets_impl_v3(
    *,
    list_repo_tree: Callable[..., Iterable[object]],
    enumerate_external_locators: Callable[..., ExternalLocatorListingV3],
    mode: str,
    enumerator_binding_sha256: str,
    route_manifest_path: Path = SOURCE_ROUTE_MANIFEST_PATH,
) -> UpstreamEnumerationReceiptV3:
    """Enumerate exact pinned metadata and mint one complete receipt.

    ``list_repo_tree`` is invoked once per distinct ``(repository, revision)``
    with ``repo_type='dataset'`` and ``recursive=True``.  ``expand=False`` is
    explicit because the pinned client already returns size, Git OID, LFS and
    Xet metadata without the unrelated last-commit/security expansion.  A
    selected file without a blob identity still fails closed.  The external
    callable is invoked only for the pinned Wikipedia URL-list route;
    authoritative mode accepts only its concrete byte-observing factory result.
    """

    if not callable(list_repo_tree) or not callable(enumerate_external_locators):
        raise TypeError("enumerators must be callables")
    if mode not in ENUMERATION_MODES:
        raise ValueError("mode must explicitly be AUTHORITATIVE or NONAUTHORITATIVE_FIXTURE")
    if not isinstance(route_manifest_path, Path):
        raise TypeError("route_manifest_path must be a pathlib.Path")

    tree_cache: dict[tuple[str, str], tuple[object, ...]] = {}
    families: list[FamilyEnumerationV3] = []
    for route, declared in _route_pairs(route_manifest_path):
        if route.source_family == "wikipedia_wikibooks":
            (
                assets,
                observed_available_bytes,
                external_manifest,
                external_manifest_bytes,
                external_listing_receipt,
            ) = (
                _enumerate_external_family(
                    route,
                    declared,
                    enumerate_external_locators,
                    mode,
                )
            )
        else:
            cache_key = (route.repository, route.revision)
            if cache_key not in tree_cache:
                raw_tree = list_repo_tree(
                    repo_id=route.repository,
                    repo_type="dataset",
                    revision=route.revision,
                    recursive=True,
                    expand=False,
                )
                if isinstance(raw_tree, (str, bytes, Mapping)):
                    raise TypeError("list_repo_tree must return an iterable of entries")
                tree_cache[cache_key] = tuple(raw_tree)
            assets = _enumerate_hf_family(route, tree_cache[cache_key])
            observed_available_bytes = sum(
                asset.upstream_bytes for asset in assets
            )
            external_manifest = None
            external_manifest_bytes = None
            external_listing_receipt = None
        families.append(
            FamilyEnumerationV3(
                route=route,
                declared_asset_count=declared.asset_count,
                declared_available_bytes=declared.available_bytes,
                available_bytes_basis=declared.available_bytes_basis,
                observed_available_bytes=observed_available_bytes,
                assets=assets,
                external_locator_manifest_sha256=external_manifest,
                external_locator_manifest_bytes=external_manifest_bytes,
                external_locator_listing_receipt_sha256=external_listing_receipt,
            )
        )
    return UpstreamEnumerationReceiptV3._validated(
        mode=mode,
        enumerator_binding_sha256=enumerator_binding_sha256,
        families=tuple(families),
        sentinel=_RECEIPT_FACTORY_SENTINEL,
    )


def enumerate_upstream_assets_v3(
    *,
    list_repo_tree: Callable[..., Iterable[object]],
    enumerate_external_locators: Callable[..., ExternalLocatorListingV3],
    mode: str,
    route_manifest_path: Path = SOURCE_ROUTE_MANIFEST_PATH,
) -> UpstreamEnumerationReceiptV3:
    """Run only an explicitly nonauthoritative injected fixture enumeration."""

    if mode == AUTHORITATIVE_MODE:
        raise ValueError(
            "injected list_repo_tree cannot mint AUTHORITATIVE enumeration; "
            "use enumerate_authoritative_upstream_assets_v3"
        )
    return _enumerate_upstream_assets_impl_v3(
        list_repo_tree=list_repo_tree,
        enumerate_external_locators=enumerate_external_locators,
        mode=mode,
        enumerator_binding_sha256=_FIXTURE_ENUMERATOR_BINDING_SHA256,
        route_manifest_path=route_manifest_path,
    )


def enumerate_authoritative_upstream_assets_v3(
    *,
    open_resource: Callable[[str], BinaryIO],
    route_manifest_path: Path = SOURCE_ROUTE_MANIFEST_PATH,
) -> UpstreamEnumerationReceiptV3:
    """Instantiate and attest the pinned hub client before authoritative minting."""

    observed_version = metadata.version(_HUGGINGFACE_HUB_DISTRIBUTION)
    if observed_version != _HUGGINGFACE_HUB_VERSION:
        raise RuntimeError(
            "authoritative enumeration requires huggingface_hub 1.24.0 exactly"
        )
    from huggingface_hub import HfApi

    api = HfApi()
    if type(api).__name__ != "HfApi" or type(api).__module__ != "huggingface_hub.hf_api":
        raise RuntimeError("authoritative enumeration instantiated a foreign HfApi")
    declared_by_source = {
        row.source_family: row
        for row in load_source_route_manifest(route_manifest_path).routes
    }

    def external_factory(**kwargs: object) -> ExternalLocatorListingV3:
        route = kwargs.get("route")
        if not isinstance(route, ExactSourceRouteV3):
            raise TypeError("authoritative external factory received an untyped route")
        expected = kwargs.get("expected_manifest_sha256")
        if expected != declared_by_source[route.source_family].external_locator_manifest_sha256:
            raise ValueError("authoritative external parent hash argument drifted")
        return read_pinned_external_locator_listing_v3(
            route=route,
            declared=declared_by_source[route.source_family],
            open_resource=open_resource,
        )

    binding_sha256 = execution_authority_v3_bound_sha256(
        "weft1_huggingface_hub_enumerator_runtime_v3",
        {
            "client_class": f"{type(api).__module__}.{type(api).__name__}",
            "distribution": _HUGGINGFACE_HUB_DISTRIBUTION,
            "version": observed_version,
        },
    )
    return _enumerate_upstream_assets_impl_v3(
        list_repo_tree=api.list_repo_tree,
        enumerate_external_locators=external_factory,
        mode=AUTHORITATIVE_MODE,
        enumerator_binding_sha256=binding_sha256,
        route_manifest_path=route_manifest_path,
    )


__all__ = [
    "AUTHORITATIVE_MODE",
    "ENUMERATION_MODES",
    "FIXTURE_MODE",
    "UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V3",
    "UPSTREAM_ENUMERATION_SCHEMA_V3",
    "ExternalLocatorAssetV3",
    "ExternalLocatorListingV3",
    "FamilyEnumerationV3",
    "UpstreamAssetV3",
    "UpstreamEnumerationReceiptV3",
    "enumerate_authoritative_upstream_assets_v3",
    "enumerate_upstream_assets_v3",
    "load_upstream_enumeration_receipt_v3",
    "locator_matches_route_v3",
    "read_pinned_external_locator_listing_v3",
    "write_upstream_enumeration_receipt_v3",
]
