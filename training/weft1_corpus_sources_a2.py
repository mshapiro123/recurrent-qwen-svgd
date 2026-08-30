"""Exact source routing and offline-cache contracts for WEFT-1 P-A.

The checked-in A1 route ledger is the authority for repository, configuration,
revision, selector, and mechanically observed license metadata.  This module
adds A2's executable ordering rules and a local-cache verifier.  It performs no
network access and contains no downloader.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from training.weft1_corpus_a2 import (
    execution_authority_v3_bound_sha256,
    pipeline_seed,
)
from training.weft1_gtok_a1_contract import (
    SOURCE_FAMILIES,
    SOURCE_ROUTE_MANIFEST_PATH,
    SourceRouteBindingV2,
    load_source_route_manifest,
)
from training.weft1_gtok_contract import canonical_sha256
from training.weft1_strict_io import (
    assert_no_symlink_ancestors,
    load_strict_json_object,
)


SOURCE_ROUTE_MANIFEST_SHA256 = (
    "8455b63f8b0dde7f5a5bdb599bec7563ce2b8c9159a26b09f6302e6e326bb663"
)
SOURCE_CACHE_SCHEMA_V3 = "weft1_local_source_cache_manifest_v3"
VERIFIED_CACHE_RECEIPT_SCHEMA_V3 = "weft1_verified_local_source_cache_v3"
SCORED_SOURCE_FAMILIES = ("finemath_3plus", "fineweb_edu")
QUALITY_GATED_SOURCE_FAMILIES = ("stackedu", *SCORED_SOURCE_FAMILIES)
SCIENCE_SOURCE_PRECEDENCE = ("arxiv", "olmocr")
A2_ASSET_ORDER_SEED = pipeline_seed("corpus.shuffle")

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERIFIED_CACHE_FACTORY_SENTINEL = object()
_WINDOWS_DEVICE_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}

# Literal replay bindings.  A syntactically valid replacement repository,
# configuration, revision, or license is still drift and must fail closed.
_EXPECTED_ROUTE_LITERALS = {
    "dolma_web": (
        "general",
        "39_percent_general",
        "allenai/dolma3_pool",
        "default",
        "6462556697df1a8f5c953727e9c686629ad98b68",
        "train",
        "data/common_crawl-*-0019/*.jsonl.zst",
        "odc-by",
        "7a43fb286ca2d57f6e44fbc109b7208a778d59a7f6e3c8a52ccd4f172d4e0ab1",
    ),
    "wikipedia_wikibooks": (
        "general",
        "22_percent_general",
        "allenai/dolma",
        "v1_7",
        "7f48140530a023e9ea4c5cfb141160922727d4d3",
        "train",
        "urls/v1_7.txt -> https://olmo-data.org/dolma-v1_7/wiki/wiki-*.json.gz",
        "odc-by",
        "5379c6cbd2b567c0630578a27e0061346b6b89489e71039d9b2b1d176d414635",
    ),
    "stackedu": (
        "code",
        "primary",
        "allenai/dolma3_mix-6T",
        "default",
        "689a3ea2d8217e64d73a5058913fa43ad15e81aa",
        "train",
        "data/stack_edu-*/*.jsonl.zst",
        "odc-by",
        "901a231cdcae33ea8ce765d1bdef7606555780c4a108b0bcf046a65fa5d5adf9",
    ),
    "finemath_3plus": (
        "mathematics",
        "primary",
        "HuggingFaceTB/finemath",
        "finemath-3plus",
        "e92b25a616738fe95dc186b64dfb19f9c8525594",
        "train",
        "finemath-3plus/train-*.parquet",
        "odc-by",
        "77b9162c99fc6b5944da8793f18fb99c3b520ab7cdce0cc67ec8a1e47871da61",
    ),
    "arxiv": (
        "science_technical",
        "first",
        "allenai/dolma3_mix-6T",
        "default",
        "689a3ea2d8217e64d73a5058913fa43ad15e81aa",
        "train",
        "data/rpj-proofpile-arxiv/*.jsonl.zst",
        "odc-by",
        "901a231cdcae33ea8ce765d1bdef7606555780c4a108b0bcf046a65fa5d5adf9",
    ),
    "olmocr": (
        "science_technical",
        "fill_after_arxiv",
        "allenai/dolma3_mix-6T",
        "default",
        "689a3ea2d8217e64d73a5058913fa43ad15e81aa",
        "train",
        "data/olmocr_science_pdfs-*/*.jsonl.zst",
        "odc-by",
        "901a231cdcae33ea8ce765d1bdef7606555780c4a108b0bcf046a65fa5d5adf9",
    ),
    "fineweb_edu": (
        "general",
        "39_percent_general",
        "HuggingFaceFW/fineweb-edu",
        "default",
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "train",
        "data/*/train-*.parquet",
        "odc-by",
        "a0cc8998a20499432b28b6575f3046b714938eb8e11b8d59a1d25ddf3716061e",
    ),
}

_ASSET_LOCATOR_PATTERNS = {
    "dolma_web": re.compile(
        r"^data/common_crawl-[^/]+-0019/[^/]+\.jsonl\.zst$"
    ),
    "wikipedia_wikibooks": re.compile(
        r"^https://olmo-data\.org/dolma-v1_7/wiki/wiki-[^/]+\.json\.gz$"
    ),
    "stackedu": re.compile(r"^data/stack_edu-[^/]+/[^/]+\.jsonl\.zst$"),
    "finemath_3plus": re.compile(r"^finemath-3plus/train-[^/]+\.parquet$"),
    "arxiv": re.compile(r"^data/rpj-proofpile-arxiv/[^/]+\.jsonl\.zst$"),
    "olmocr": re.compile(
        r"^data/olmocr_science_pdfs-[^/]+/[^/]+\.jsonl\.zst$"
    ),
    "fineweb_edu": re.compile(r"^data/[^/]+/train-[^/]+\.parquet$"),
}


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _require_relative_posix_path(value: str) -> str:
    _require_nonempty(value, "relative_path")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(
            part.endswith((".", " "))
            or ":" in part
            or part.rstrip(". ").split(".", 1)[0].casefold()
            in _WINDOWS_DEVICE_NAMES
            for part in path.parts
        )
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise ValueError("relative_path must be a canonical relative POSIX path")
    return value


@dataclass(frozen=True)
class ExactSourceRouteV3:
    """One A1 route after checking every replay-critical literal."""

    source_family: str
    stratum: str
    role: str
    repository: str
    config: str
    revision: str
    split: str
    asset_selector: str
    declared_license: str
    card_sha256: str
    a1_route_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.source_family not in SOURCE_FAMILIES:
            raise ValueError("unknown source family")
        observed = (
            self.stratum,
            self.role,
            self.repository,
            self.config,
            self.revision,
            self.split,
            self.asset_selector,
            self.declared_license,
            self.card_sha256,
        )
        if observed != _EXPECTED_ROUTE_LITERALS[self.source_family]:
            raise ValueError(
                f"exact source route drifted for {self.source_family}"
            )
        if _SHA1.fullmatch(self.revision) is None:
            raise ValueError("source revision must be an exact commit SHA")
        _require_sha256(self.card_sha256, "card_sha256")
        _require_sha256(self.a1_route_receipt_sha256, "a1_route_receipt_sha256")

    @classmethod
    def from_a1(cls, route: SourceRouteBindingV2) -> "ExactSourceRouteV3":
        if not isinstance(route, SourceRouteBindingV2):
            raise TypeError("route must be a SourceRouteBindingV2")
        return cls(
            source_family=route.source_family,
            stratum=route.stratum,
            role=route.role,
            repository=route.repository,
            config=route.config,
            revision=route.revision,
            split=route.split,
            asset_selector=route.asset_selector,
            declared_license=route.declared_license,
            card_sha256=route.card_sha256,
            a1_route_receipt_sha256=route.receipt_sha256,
        )


def load_exact_source_routes_v3(
    route_manifest_path: Path = SOURCE_ROUTE_MANIFEST_PATH,
) -> tuple[ExactSourceRouteV3, ...]:
    """Load the checked-in ledger and reject any valid-looking literal drift."""

    manifest = load_source_route_manifest(route_manifest_path)
    if manifest.manifest_sha256 != SOURCE_ROUTE_MANIFEST_SHA256:
        raise ValueError("A1 source-route manifest identity drifted")
    routes = tuple(ExactSourceRouteV3.from_a1(route) for route in manifest.routes)
    if tuple(route.source_family for route in routes) != SOURCE_FAMILIES:
        raise ValueError("source routes are not in canonical family order")
    return routes


@dataclass(frozen=True)
class SourceCacheAssetV3:
    """One expected local cache object, fully pinned before verification."""

    source_family: str
    repository: str
    config: str
    revision: str
    split: str
    asset_locator: str
    relative_path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.source_family not in _EXPECTED_ROUTE_LITERALS:
            raise ValueError("unknown source family")
        route = _EXPECTED_ROUTE_LITERALS[self.source_family]
        if (
            self.repository,
            self.config,
            self.revision,
            self.split,
        ) != (route[2], route[3], route[4], route[5]):
            raise ValueError(f"cache asset route drifted for {self.source_family}")
        _require_nonempty(self.asset_locator, "asset_locator")
        if _ASSET_LOCATOR_PATTERNS[self.source_family].fullmatch(
            self.asset_locator
        ) is None:
            raise ValueError(
                f"cache asset locator falls outside the pinned selector for {self.source_family}"
            )
        _require_relative_posix_path(self.relative_path)
        if type(self.bytes) is not int or self.bytes < 1:
            raise ValueError("cache asset bytes must be a positive exact integer")
        _require_sha256(self.sha256, "cache asset sha256")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceCacheAssetV3":
        if not isinstance(value, Mapping):
            raise TypeError("cache asset must be a mapping")
        return cls(**dict(value))

    @property
    def logical_identity_payload(self) -> Mapping[str, object]:
        """Logical source identity, independent of the local cache layout."""

        return {
            "asset_locator": self.asset_locator,
            "bytes": self.bytes,
            "config": self.config,
            "repository": self.repository,
            "revision": self.revision,
            "sha256": self.sha256,
            "source_family": self.source_family,
            "split": self.split,
        }

    @property
    def asset_identity_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_source_cache_asset_v3", self.logical_identity_payload
        )


def canonical_asset_order_v3(
    assets: Sequence[SourceCacheAssetV3],
) -> tuple[SourceCacheAssetV3, ...]:
    """Apply A2's seeded SHA-256 order over exact UTF-8 asset locators."""

    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
        raise TypeError("assets must be a typed sequence")
    if any(not isinstance(asset, SourceCacheAssetV3) for asset in assets):
        raise TypeError("assets contain a non-SourceCacheAssetV3 value")
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
    logical_keys = tuple(
        (asset.source_family, asset.asset_locator) for asset in ordered
    )
    if len(logical_keys) != len(set(logical_keys)):
        raise ValueError("cache manifest repeats a source asset locator")
    paths = tuple(asset.relative_path for asset in ordered)
    if len(paths) != len(set(paths)):
        raise ValueError("cache manifest repeats a local relative path")
    return ordered


def asset_order_digest_v3(locator: str) -> bytes:
    """Hash ``seed_u64be || NUL || locator_utf8`` exactly as A2 binds it."""

    _require_nonempty(locator, "asset locator")
    encoded = locator.encode("utf-8", errors="strict")
    return hashlib.sha256(
        A2_ASSET_ORDER_SEED.to_bytes(8, "big") + b"\x00" + encoded
    ).digest()


@dataclass(frozen=True)
class SourceCacheManifestV3:
    """Expected source-cache contents; it is not proof that files were read."""

    schema: str
    source_route_manifest_sha256: str
    assets: tuple[SourceCacheAssetV3, ...]

    def __post_init__(self) -> None:
        if self.schema != SOURCE_CACHE_SCHEMA_V3:
            raise ValueError("unexpected source-cache manifest schema")
        if self.source_route_manifest_sha256 != SOURCE_ROUTE_MANIFEST_SHA256:
            raise ValueError("source-cache manifest is bound to the wrong route ledger")
        if not isinstance(self.assets, tuple) or not self.assets:
            raise ValueError("source-cache manifest requires a nonempty asset tuple")
        if self.assets != canonical_asset_order_v3(self.assets):
            raise ValueError("source-cache assets are not in canonical order")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceCacheManifestV3":
        if not isinstance(value, Mapping):
            raise TypeError("source-cache manifest must be a mapping")
        payload = dict(value)
        raw_assets = payload.pop("assets")
        if not isinstance(raw_assets, list):
            raise TypeError("source-cache manifest assets must be a JSON list")
        return cls(
            assets=tuple(SourceCacheAssetV3.from_mapping(item) for item in raw_assets),
            **payload,
        )

    @property
    def manifest_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_source_cache_manifest_v3", self
        )

    @property
    def offline_replay_identity_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_source_cache_offline_replay_identity_v3",
            {
                "assets": tuple(asset.logical_identity_payload for asset in self.assets),
                "source_route_manifest_sha256": self.source_route_manifest_sha256,
            },
        )


def load_source_cache_manifest_v3(path: Path) -> SourceCacheManifestV3:
    if not isinstance(path, Path):
        raise TypeError("source-cache manifest path must be a pathlib.Path")
    payload = load_strict_json_object(path)
    return SourceCacheManifestV3.from_mapping(payload)


@dataclass(frozen=True)
class VerifiedLocalCacheAssetV3:
    """One cache asset after streaming its actual local bytes."""

    expected: SourceCacheAssetV3
    observed_bytes: int
    observed_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.expected, SourceCacheAssetV3):
            raise TypeError("verified cache asset requires a typed expectation")
        if type(self.observed_bytes) is not int or self.observed_bytes < 1:
            raise ValueError("observed cache bytes must be a positive exact integer")
        _require_sha256(self.observed_sha256, "observed_sha256")
        if self.observed_bytes != self.expected.bytes:
            raise ValueError("local cache asset byte count drifted")
        if self.observed_sha256 != self.expected.sha256:
            raise ValueError("local cache asset SHA-256 drifted")


@dataclass(frozen=True, init=False)
class VerifiedLocalCacheManifestV3:
    """Typed evidence that an expected source cache was read byte-for-byte."""

    source_manifest: SourceCacheManifestV3
    assets: tuple[VerifiedLocalCacheAssetV3, ...]
    cache_root_label: str

    def __new__(cls) -> "VerifiedLocalCacheManifestV3":
        raise TypeError(
            "VerifiedLocalCacheManifestV3 is factory-minted after filesystem reads"
        )

    @classmethod
    def _validated(
        cls,
        *,
        source_manifest: SourceCacheManifestV3,
        assets: tuple[VerifiedLocalCacheAssetV3, ...],
        cache_root_label: str,
        sentinel: object,
    ) -> "VerifiedLocalCacheManifestV3":
        if sentinel is not _VERIFIED_CACHE_FACTORY_SENTINEL:
            raise PermissionError("verified-cache receipts are factory-only")
        instance = object.__new__(cls)
        object.__setattr__(instance, "source_manifest", source_manifest)
        object.__setattr__(instance, "assets", assets)
        object.__setattr__(instance, "cache_root_label", cache_root_label)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        if not isinstance(self.source_manifest, SourceCacheManifestV3):
            raise TypeError("verified cache requires a typed source manifest")
        if not isinstance(self.assets, tuple) or not self.assets:
            raise ValueError("verified cache requires a nonempty asset tuple")
        if any(not isinstance(asset, VerifiedLocalCacheAssetV3) for asset in self.assets):
            raise TypeError("verified cache contains a non-verified asset")
        if tuple(asset.expected for asset in self.assets) != self.source_manifest.assets:
            raise ValueError("verified cache does not exactly cover its source manifest")
        _require_nonempty(self.cache_root_label, "cache_root_label")

    @property
    def offline_replay_identity_sha256(self) -> str:
        return self.source_manifest.offline_replay_identity_sha256

    @property
    def verification_receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            VERIFIED_CACHE_RECEIPT_SCHEMA_V3,
            {
                "cache_root_label": self.cache_root_label,
                "offline_replay_identity_sha256": self.offline_replay_identity_sha256,
                "observations": tuple(
                    {
                        "observed_bytes": asset.observed_bytes,
                        "observed_sha256": asset.observed_sha256,
                        "relative_path": asset.expected.relative_path,
                    }
                    for asset in self.assets
                ),
                "source_manifest_sha256": self.source_manifest.manifest_sha256,
            },
        )

    @property
    def total_bytes(self) -> int:
        return sum(asset.observed_bytes for asset in self.assets)


def _hash_file(path: Path) -> tuple[int, str]:
    assert_no_symlink_ancestors(path)
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def verify_local_source_cache_v3(
    manifest_path: Path,
    cache_root: Path,
    route_manifest_path: Path = SOURCE_ROUTE_MANIFEST_PATH,
) -> VerifiedLocalCacheManifestV3:
    """Verify a local cache only; this function never opens a network resource."""

    if not all(isinstance(value, Path) for value in (manifest_path, cache_root, route_manifest_path)):
        raise TypeError("cache verifier paths must be pathlib.Path values")
    load_exact_source_routes_v3(route_manifest_path)
    manifest = load_source_cache_manifest_v3(manifest_path)
    assert_no_symlink_ancestors(cache_root)
    root = cache_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("cache_root must resolve to a directory")
    verified: list[VerifiedLocalCacheAssetV3] = []
    for expected in manifest.assets:
        lexical_candidate = root / Path(
            *PurePosixPath(expected.relative_path).parts
        )
        assert_no_symlink_ancestors(lexical_candidate)
        candidate = lexical_candidate.resolve(strict=True)
        if candidate == root or root not in candidate.parents:
            raise ValueError("cache asset resolves outside cache_root")
        if not candidate.is_file():
            raise ValueError("cache asset must resolve to a regular file")
        observed_bytes, observed_sha256 = _hash_file(candidate)
        verified.append(
            VerifiedLocalCacheAssetV3(
                expected=expected,
                observed_bytes=observed_bytes,
                observed_sha256=observed_sha256,
            )
        )
    return VerifiedLocalCacheManifestV3._validated(
        source_manifest=manifest,
        assets=tuple(verified),
        cache_root_label=cache_root.name or ".",
        sentinel=_VERIFIED_CACHE_FACTORY_SENTINEL,
    )


def verify_source_cache_manifest(
    manifest_path: Path,
    cache_root: Path,
    route_manifest_path: Path = SOURCE_ROUTE_MANIFEST_PATH,
) -> Mapping[str, object]:
    """CLI-friendly summary of a strict local source-cache verification."""

    receipt = verify_local_source_cache_v3(
        manifest_path, cache_root, route_manifest_path
    )
    return {
        "asset_count": len(receipt.assets),
        "offline_replay_identity_sha256": receipt.offline_replay_identity_sha256,
        "schema": VERIFIED_CACHE_RECEIPT_SCHEMA_V3,
        "source_cache_manifest_sha256": receipt.source_manifest.manifest_sha256,
        "source_route_manifest_sha256": SOURCE_ROUTE_MANIFEST_SHA256,
        "total_bytes": receipt.total_bytes,
        "verification_receipt_sha256": receipt.verification_receipt_sha256,
    }


@dataclass(frozen=True)
class CanonicalSourceRecordV3:
    """One source record with a stable identity and explicit order metadata."""

    asset: SourceCacheAssetV3
    source_record_ordinal: int
    retained_byte_count: int
    native_record_id: str | None = None
    int_score: int | None = None
    native_record_namespace: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.asset, SourceCacheAssetV3):
            raise TypeError("source record requires a typed cache asset")
        if type(self.source_record_ordinal) is not int or self.source_record_ordinal < 0:
            raise ValueError("source_record_ordinal must be a non-negative integer")
        if type(self.retained_byte_count) is not int or self.retained_byte_count < 1:
            raise ValueError("retained_byte_count must be a positive exact integer")
        if self.native_record_id is not None:
            _require_nonempty(self.native_record_id, "native_record_id")
        if self.native_record_namespace is not None:
            if self.native_record_id is None:
                raise ValueError(
                    "native_record_namespace requires a native_record_id"
                )
            _require_nonempty(
                self.native_record_namespace,
                "native_record_namespace",
            )
        if self.asset.source_family in QUALITY_GATED_SOURCE_FAMILIES:
            if type(self.int_score) is not int or self.int_score < 3:
                raise ValueError(
                    "StackEdu, FineMath, and FineWeb-Edu require integer int_score >= 3"
                )
        elif self.int_score is not None:
            raise ValueError("ungated source families may not carry int_score")

    @property
    def source_family(self) -> str:
        return self.asset.source_family

    @property
    def canonical_source_record_id(self) -> str:
        if self.native_record_id is not None:
            identity: Mapping[str, object] = {
                "config": self.asset.config,
                "native_record_id": self.native_record_id,
                "repository": self.asset.repository,
                "revision": self.asset.revision,
                "source_family": self.source_family,
            }
            if self.native_record_namespace is not None:
                identity = {
                    **identity,
                    "native_record_namespace": self.native_record_namespace,
                }
        else:
            identity = {
                "asset_locator": self.asset.asset_locator,
                "config": self.asset.config,
                "repository": self.asset.repository,
                "revision": self.asset.revision,
                "source_family": self.source_family,
                "source_record_ordinal": self.source_record_ordinal,
            }
        return canonical_sha256(
            {"payload": identity, "schema": "weft1_canonical_source_record_v3"}
        )


def order_family_records_v3(
    records: Sequence[CanonicalSourceRecordV3],
) -> tuple[CanonicalSourceRecordV3, ...]:
    """Apply the one bound ordering rule for a single source family."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a typed sequence")
    if not records:
        return ()
    if any(not isinstance(record, CanonicalSourceRecordV3) for record in records):
        raise TypeError("records contain a non-CanonicalSourceRecordV3 value")
    families = {record.source_family for record in records}
    if len(families) != 1:
        raise ValueError("family ordering accepts exactly one source family")
    canonical_ids = tuple(record.canonical_source_record_id for record in records)
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("source family repeats a canonical source record ID")
    family = next(iter(families))
    if family in SCORED_SOURCE_FAMILIES:
        return tuple(
            sorted(
                records,
                key=lambda record: (
                    -record.int_score,  # type: ignore[operator]
                    record.canonical_source_record_id,
                ),
            )
        )
    return tuple(
        sorted(
            records,
            key=lambda record: (
                asset_order_digest_v3(record.asset.asset_locator),
                record.source_record_ordinal,
                record.canonical_source_record_id,
            ),
        )
    )


def order_science_records_v3(
    records: Sequence[CanonicalSourceRecordV3],
) -> tuple[CanonicalSourceRecordV3, ...]:
    """Materialize all arXiv records before any residual olmOCR records."""

    if any(
        not isinstance(record, CanonicalSourceRecordV3)
        or record.source_family not in SCIENCE_SOURCE_PRECEDENCE
        for record in records
    ):
        raise ValueError("science ordering accepts only arXiv and olmOCR records")
    by_family = {
        family: tuple(record for record in records if record.source_family == family)
        for family in SCIENCE_SOURCE_PRECEDENCE
    }
    return tuple(
        record
        for family in SCIENCE_SOURCE_PRECEDENCE
        for record in order_family_records_v3(by_family[family])
    )


def fineweb_ranked_remainder_v3(
    ranked_records: Sequence[CanonicalSourceRecordV3],
    *,
    consumed_prefix_count: int,
    excluded_record_ids: frozenset[str] = frozenset(),
) -> tuple[CanonicalSourceRecordV3, ...]:
    """Resume FineWeb top-up after the consumed ranked prefix, never restart it."""

    if any(
        not isinstance(record, CanonicalSourceRecordV3)
        or record.source_family != "fineweb_edu"
        for record in ranked_records
    ):
        raise ValueError("FineWeb remainder requires only FineWeb-Edu records")
    ordered = order_family_records_v3(ranked_records)
    if tuple(ranked_records) != ordered:
        raise ValueError("FineWeb remainder input must already be canonically ranked")
    if (
        type(consumed_prefix_count) is not int
        or not 0 <= consumed_prefix_count <= len(ordered)
    ):
        raise ValueError("consumed_prefix_count is outside the ranked stream")
    if not isinstance(excluded_record_ids, frozenset) or any(
        _SHA256.fullmatch(value) is None for value in excluded_record_ids
    ):
        raise ValueError("excluded_record_ids must be a frozenset of canonical IDs")
    return tuple(
        record
        for record in ordered[consumed_prefix_count:]
        if record.canonical_source_record_id not in excluded_record_ids
    )


__all__ = [
    "CanonicalSourceRecordV3",
    "ExactSourceRouteV3",
    "SCIENCE_SOURCE_PRECEDENCE",
    "SCORED_SOURCE_FAMILIES",
    "QUALITY_GATED_SOURCE_FAMILIES",
    "SOURCE_CACHE_SCHEMA_V3",
    "SOURCE_ROUTE_MANIFEST_SHA256",
    "SourceCacheAssetV3",
    "SourceCacheManifestV3",
    "VERIFIED_CACHE_RECEIPT_SCHEMA_V3",
    "VerifiedLocalCacheAssetV3",
    "VerifiedLocalCacheManifestV3",
    "asset_order_digest_v3",
    "canonical_asset_order_v3",
    "fineweb_ranked_remainder_v3",
    "load_exact_source_routes_v3",
    "load_source_cache_manifest_v3",
    "order_family_records_v3",
    "order_science_records_v3",
    "verify_local_source_cache_v3",
    "verify_source_cache_manifest",
]
