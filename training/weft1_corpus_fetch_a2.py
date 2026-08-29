"""Concrete online source preparation for WEFT-1 P-A.

This is a deliberately narrow run-axis command.  It enumerates the exact
pinned source routes, caches the Wikipedia locator parent and its two assets,
selects the minimal deterministic compressed-byte prefix for each family,
downloads only that finite plan, and persists replayable source receipts.

It does not parse documents, materialize T/H, fit a tokenizer, create a model
checkpoint, consume sealed data, train a model, or mint a programme gate.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import os
from pathlib import Path
import re
from typing import BinaryIO, Callable, Mapping
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from training.weft1_corpus_enumeration_a2 import (
    AUTHORITATIVE_MODE,
    UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V3,
    UpstreamAssetV3,
    UpstreamEnumerationReceiptV3,
    enumerate_authoritative_upstream_assets_v3,
)
from training.weft1_corpus_source_io_a2 import (
    SOURCE_CACHE_DOWNLOAD_ARTIFACT_SCHEMA_V3,
    SourceAssetDownloadPlanV3,
    SourceCacheDownloadReceiptV3,
    SourceTransportError,
    finalize_source_cache_v3,
    materialize_source_cache_v3,
    plan_source_cache_assets_v3,
)
from training.weft1_gtok_a1_contract import (
    SOURCE_FAMILIES,
    SOURCE_ROUTE_MANIFEST_PATH,
    load_source_route_manifest,
)
from training.weft1_gtok_contract import canonical_json_bytes, canonical_sha256
from training.weft1_strict_io import (
    assert_no_symlink_ancestors,
    load_canonical_json_snapshot,
)


PA_FETCH_SELECTION_SCHEMA_V3 = "weft1_pa_source_prefix_selection_v3"
PA_FETCH_SELECTION_ARTIFACT_SCHEMA_V3 = (
    "weft1_pa_source_prefix_selection_artifact_v3"
)
ENUMERATION_ARTIFACT_NAME = "upstream-enumeration-v3.json"
SELECTION_ARTIFACT_NAME = "source-prefix-selection-v3.json"
SOURCE_MANIFEST_NAME = "source-cache-manifest-v3.json"
DOWNLOAD_ARTIFACT_NAME = "source-cache-download-receipt-v3.json"
_HUGGINGFACE_HUB_DISTRIBUTION = "huggingface-hub"
_HUGGINGFACE_HUB_VERSION = "1.24.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXTERNAL_HOSTS = frozenset({"huggingface.co", "olmo-data.org"})
_EXTERNAL_CACHE_ENTRY_SCHEMA_V3 = "weft1_external_transport_cache_entry_v3"


class SourceFetchError(RuntimeError):
    """A source-preparation authority, selection, or transport check failed."""


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _default_open_url(locator: str) -> BinaryIO:
    request = Request(
        locator,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "WEFT-1-P-A-source-fetch/3",
        },
        method="GET",
    )
    opened = urlopen(  # noqa: S310 - exact HTTPS hosts and final URL checked
        request,
        timeout=120,
    )
    final_url = opened.geturl() if hasattr(opened, "geturl") else None
    if final_url != locator:
        opened.close()
        raise SourceFetchError(
            "external transport redirect is forbidden by the pinned-origin policy"
        )
    return opened


@dataclass(frozen=True)
class ExternalCacheObservationV3:
    locator: str
    relative_path: str
    observed_bytes: int
    observed_sha256: str
    cache_hit: bool

    def __post_init__(self) -> None:
        _validate_external_locator(self.locator)
        if type(self.observed_bytes) is not int or self.observed_bytes < 1:
            raise ValueError("external cache observation requires positive bytes")
        _require_sha256(self.observed_sha256, "external observed SHA-256")
        if type(self.cache_hit) is not bool:
            raise TypeError("cache_hit must be an exact bool")


def _validate_external_locator(locator: str) -> str:
    if not isinstance(locator, str) or not locator:
        raise ValueError("external resource locator must be nonempty")
    parsed = urlsplit(locator)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _EXTERNAL_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("external resource locator is outside the pinned HTTPS hosts")
    return locator


def _hash_file(path: Path) -> tuple[int, str]:
    assert_no_symlink_ancestors(path)
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    if byte_count < 1:
        raise SourceFetchError("cached external resource is empty")
    return byte_count, digest.hexdigest()


class ExternalResourceCacheV3:
    """Non-overwriting cache for the pinned Wikipedia parent and assets.

    Cached bytes are not trusted merely because they exist.  The authoritative
    enumeration factory re-reads and validates the parent SHA and both asset
    byte identities on every invocation.
    """

    def __init__(
        self,
        root: Path,
        *,
        open_url: Callable[[str], BinaryIO] = _default_open_url,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("external cache root must be a pathlib.Path")
        if not callable(open_url):
            raise TypeError("external cache opener must be callable")
        assert_no_symlink_ancestors(root)
        root.mkdir(parents=True, exist_ok=True)
        self._root = root.resolve(strict=True)
        if not self._root.is_dir():
            raise SourceFetchError("external cache root is not a directory")
        self._open_url = open_url
        self._observations: dict[str, ExternalCacheObservationV3] = {}

    @property
    def observations(self) -> tuple[ExternalCacheObservationV3, ...]:
        return tuple(
            self._observations[key]
            for key in sorted(self._observations, key=lambda item: item.encode("utf-8"))
        )

    def _path(self, locator: str) -> tuple[Path, Path, str]:
        digest = hashlib.sha256(locator.encode("utf-8")).hexdigest()
        relative = f"resources/{digest[:2]}/{digest}.bin"
        lexical = self._root / "resources" / digest[:2] / f"{digest}.bin"
        assert_no_symlink_ancestors(lexical)
        resolved = lexical.resolve(strict=False)
        if self._root not in resolved.parents:
            raise SourceFetchError("external cache path escaped its root")
        receipt = resolved.with_name(resolved.name + ".receipt.json")
        assert_no_symlink_ancestors(receipt)
        return resolved, receipt, relative

    def _load_entry_receipt(
        self,
        path: Path,
        *,
        locator: str,
        relative_path: str,
    ) -> tuple[int, str]:
        raw, payload = load_canonical_json_snapshot(path)
        if raw != canonical_json_bytes(payload) + b"\n" or set(payload) != {
            "bytes",
            "locator",
            "relative_path",
            "schema",
            "sha256",
        }:
            raise SourceFetchError("external cache receipt shape drifted")
        if (
            payload["schema"] != _EXTERNAL_CACHE_ENTRY_SCHEMA_V3
            or payload["locator"] != locator
            or payload["relative_path"] != relative_path
            or type(payload["bytes"]) is not int
            or payload["bytes"] < 1
        ):
            raise SourceFetchError("external cache receipt authority drifted")
        observed_sha256 = payload["sha256"]
        if not isinstance(observed_sha256, str):
            raise SourceFetchError("external cache receipt SHA-256 is untyped")
        _require_sha256(observed_sha256, "external cache receipt SHA-256")
        return payload["bytes"], observed_sha256

    def _write_entry_receipt(
        self,
        path: Path,
        *,
        locator: str,
        relative_path: str,
        observed_bytes: int,
        observed_sha256: str,
    ) -> None:
        payload = {
            "bytes": observed_bytes,
            "locator": locator,
            "relative_path": relative_path,
            "schema": _EXTERNAL_CACHE_ENTRY_SCHEMA_V3,
            "sha256": observed_sha256,
        }
        raw = canonical_json_bytes(payload) + b"\n"
        partial = path.with_name(path.name + ".partial")
        assert_no_symlink_ancestors(partial)
        if path.exists():
            raise SourceFetchError("refusing to overwrite an external cache receipt")
        if partial.exists():
            if not partial.is_file():
                raise SourceFetchError("external cache receipt partial is not a file")
            partial.unlink()
        try:
            with partial.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    def open(self, locator: str) -> BinaryIO:
        locator = _validate_external_locator(locator)
        final_path, receipt_path, relative_path = self._path(locator)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        partial = final_path.with_name(final_path.name + ".partial")
        assert_no_symlink_ancestors(partial)
        if partial.exists():
            if not partial.is_file():
                raise SourceFetchError("external cache partial is not a regular file")
            partial.unlink()
        if final_path.exists() and not receipt_path.exists():
            # The atomic data rename completed but the identity receipt did
            # not.  This is an interrupted transfer, not a completed cache
            # entry, so it may be discarded and fetched again.
            if not final_path.is_file():
                raise SourceFetchError("external cache entry is not a regular file")
            final_path.unlink()
        if receipt_path.exists() and not final_path.exists():
            raise SourceFetchError("external cache receipt has no corresponding bytes")
        cache_hit = final_path.exists()
        if cache_hit:
            if not final_path.is_file():
                raise SourceFetchError("external cache entry is not a regular file")
            expected_bytes, expected_sha256 = self._load_entry_receipt(
                receipt_path,
                locator=locator,
                relative_path=relative_path,
            )
        else:
            opened = self._open_url(locator)
            if not hasattr(opened, "read"):
                raise TypeError("external transport returned no binary reader")
            try:
                digest = hashlib.sha256()
                byte_count = 0
                with closing(opened), partial.open("xb") as output:
                    while True:
                        chunk = opened.read(8 * 1024 * 1024)
                        if chunk is None or chunk == b"":
                            break
                        if not isinstance(chunk, bytes):
                            raise SourceFetchError(
                                "external transport returned non-bytes data"
                            )
                        byte_count += len(chunk)
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if byte_count < 1:
                    raise SourceFetchError("external transport returned an empty resource")
                os.replace(partial, final_path)
                self._write_entry_receipt(
                    receipt_path,
                    locator=locator,
                    relative_path=relative_path,
                    observed_bytes=byte_count,
                    observed_sha256=digest.hexdigest(),
                )
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        observed_bytes, observed_sha256 = _hash_file(final_path)
        if cache_hit and (
            observed_bytes != expected_bytes or observed_sha256 != expected_sha256
        ):
            raise SourceFetchError(
                "external cache bytes differ from their immutable receipt"
            )
        self._observations[locator] = ExternalCacheObservationV3(
            locator=locator,
            relative_path=relative_path,
            observed_bytes=observed_bytes,
            observed_sha256=observed_sha256,
            cache_hit=cache_hit,
        )
        return final_path.open("rb")


class PinnedHuggingFaceAssetOpenerV3:
    """Open pinned HF assets through huggingface_hub 1.24.0's local cache."""

    def __init__(
        self,
        root: Path,
        *,
        external_cache: ExternalResourceCacheV3,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("Hugging Face cache root must be a pathlib.Path")
        if not isinstance(external_cache, ExternalResourceCacheV3):
            raise TypeError("asset opener requires the concrete external cache")
        observed_version = metadata.version(_HUGGINGFACE_HUB_DISTRIBUTION)
        if observed_version != _HUGGINGFACE_HUB_VERSION:
            raise RuntimeError(
                "source download requires huggingface_hub 1.24.0 exactly"
            )
        from huggingface_hub import hf_hub_download

        assert_no_symlink_ancestors(root)
        root.mkdir(parents=True, exist_ok=True)
        self._root = root.resolve(strict=True)
        if not self._root.is_dir():
            raise SourceFetchError("Hugging Face cache root is not a directory")
        self._external_cache = external_cache
        self._download = hf_hub_download

    def open(self, asset: UpstreamAssetV3) -> BinaryIO:
        if not isinstance(asset, UpstreamAssetV3):
            raise TypeError("upstream opener requires a typed asset")
        if asset.asset_locator.startswith("https://"):
            return self._external_cache.open(asset.asset_locator)
        local = self._download(
            repo_id=asset.repository,
            filename=asset.asset_locator,
            revision=asset.revision,
            repo_type="dataset",
            cache_dir=str(self._root),
            force_download=False,
            local_files_only=False,
        )
        if not isinstance(local, str) or not local:
            raise SourceFetchError("huggingface_hub returned no local asset path")
        path = Path(local).resolve(strict=True)
        if path == self._root or self._root not in path.parents or not path.is_file():
            raise SourceFetchError("Hugging Face asset resolved outside its cache")
        return path.open("rb")


@dataclass(frozen=True)
class FamilyPrefixSelectionV3:
    source_family: str
    selection_rule: str
    required_bytes: int
    available_asset_count: int
    available_payload_bytes: int
    selected_asset_count: int
    selected_upstream_bytes: int
    selected_asset_identities_sha256: str
    terminal_asset_identity_sha256: str
    terminal_asset_locator: str

    def __post_init__(self) -> None:
        if self.source_family not in SOURCE_FAMILIES:
            raise ValueError("prefix selection uses an unknown source family")
        if self.selection_rule not in {
            "minimal_seeded_prefix_reaching_required_bytes",
            "complete_pinned_wikipedia_asset_set",
        }:
            raise ValueError("prefix selection rule is unknown")
        for name in (
            "required_bytes",
            "available_asset_count",
            "available_payload_bytes",
            "selected_asset_count",
            "selected_upstream_bytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.selected_asset_count > self.available_asset_count:
            raise ValueError("selected asset count exceeds the enumerated family")
        if self.selected_upstream_bytes < self.required_bytes:
            raise ValueError("selected prefix does not reach its required bytes")
        _require_sha256(
            self.selected_asset_identities_sha256,
            "selected asset identities SHA-256",
        )
        _require_sha256(
            self.terminal_asset_identity_sha256,
            "terminal asset identity SHA-256",
        )
        if not isinstance(self.terminal_asset_locator, str) or not self.terminal_asset_locator:
            raise ValueError("prefix selection requires a terminal locator")


@dataclass(frozen=True)
class PASourcePrefixSelectionReceiptV3:
    schema: str
    enumeration_receipt_sha256: str
    source_route_manifest_sha256: str
    families: tuple[FamilyPrefixSelectionV3, ...]
    selection_plan_sha256: str

    def __post_init__(self) -> None:
        if self.schema != PA_FETCH_SELECTION_SCHEMA_V3:
            raise ValueError("unexpected P-A prefix selection schema")
        _require_sha256(self.enumeration_receipt_sha256, "enumeration receipt")
        _require_sha256(self.source_route_manifest_sha256, "source route manifest")
        _require_sha256(self.selection_plan_sha256, "selection plan")
        if tuple(row.source_family for row in self.families) != SOURCE_FAMILIES:
            raise ValueError("prefix selection must cover every family in order")

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(
            {"payload": self, "schema": PA_FETCH_SELECTION_SCHEMA_V3}
        )


@dataclass(frozen=True)
class PASourcePreparationResultV3:
    enumeration: UpstreamEnumerationReceiptV3
    selection: PASourcePrefixSelectionReceiptV3
    plan: SourceAssetDownloadPlanV3
    download: SourceCacheDownloadReceiptV3
    enumeration_artifact_sha256: str
    selection_artifact_sha256: str
    download_artifact_sha256: str

    def __post_init__(self) -> None:
        if self.enumeration.receipt_sha256 != self.selection.enumeration_receipt_sha256:
            raise ValueError("source preparation selection/enumeration mismatch")
        if self.plan.plan_sha256 != self.selection.selection_plan_sha256:
            raise ValueError("source preparation selection/plan mismatch")
        if self.download.selection_plan_sha256 != self.plan.plan_sha256:
            raise ValueError("source preparation download/plan mismatch")
        for name in (
            "enumeration_artifact_sha256",
            "selection_artifact_sha256",
            "download_artifact_sha256",
        ):
            _require_sha256(getattr(self, name), name)


def _required_bytes(
    enumeration: UpstreamEnumerationReceiptV3,
    route_manifest_path: Path,
    required_bytes_by_family: Mapping[str, int] | None,
) -> dict[str, int]:
    ledger = load_source_route_manifest(route_manifest_path)
    production = {route.source_family: route.required_bytes for route in ledger.routes}
    if required_bytes_by_family is None:
        return production
    if not isinstance(required_bytes_by_family, Mapping) or set(
        required_bytes_by_family
    ) != set(SOURCE_FAMILIES):
        raise ValueError("required-byte override must cover exactly seven families")
    requested = dict(required_bytes_by_family)
    if any(type(value) is not int or value < 1 for value in requested.values()):
        raise ValueError("required-byte overrides must be positive exact integers")
    if enumeration.authoritative and requested != production:
        raise ValueError("authoritative selection must use A1 required bytes")
    return requested


def select_required_asset_prefixes_v3(
    enumeration: UpstreamEnumerationReceiptV3,
    *,
    route_manifest_path: Path = SOURCE_ROUTE_MANIFEST_PATH,
    required_bytes_by_family: Mapping[str, int] | None = None,
) -> tuple[SourceAssetDownloadPlanV3, PASourcePrefixSelectionReceiptV3]:
    """Select each canonical family prefix, including both science reserves."""

    if not isinstance(enumeration, UpstreamEnumerationReceiptV3):
        raise TypeError("prefix selection requires a typed enumeration")
    targets = _required_bytes(
        enumeration,
        route_manifest_path,
        required_bytes_by_family,
    )
    selected_all: list[UpstreamAssetV3] = []
    rows: list[FamilyPrefixSelectionV3] = []
    for family in enumeration.families:
        target = targets[family.source_family]
        if family.source_family == "wikipedia_wikibooks":
            selected = family.assets
            rule = "complete_pinned_wikipedia_asset_set"
            if enumeration.authoritative and len(selected) != 2:
                raise SourceFetchError(
                    "authoritative Wikipedia selection must contain both pinned assets"
                )
        else:
            chosen: list[UpstreamAssetV3] = []
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
            raise SourceFetchError(
                f"enumerated {family.source_family} assets do not reach A1 required bytes"
            )
        if rule.startswith("minimal") and (
            selected_bytes - selected[-1].upstream_bytes >= target
        ):
            raise AssertionError("selected source prefix is not minimal")
        selected_identities = tuple(
            asset.asset_identity_sha256 for asset in selected
        )
        rows.append(
            FamilyPrefixSelectionV3(
                source_family=family.source_family,
                selection_rule=rule,
                required_bytes=target,
                available_asset_count=len(family.assets),
                available_payload_bytes=family.asset_payload_bytes,
                selected_asset_count=len(selected),
                selected_upstream_bytes=selected_bytes,
                selected_asset_identities_sha256=canonical_sha256(
                    selected_identities
                ),
                terminal_asset_identity_sha256=selected[-1].asset_identity_sha256,
                terminal_asset_locator=selected[-1].asset_locator,
            )
        )
        selected_all.extend(selected)
    plan = plan_source_cache_assets_v3(enumeration, tuple(selected_all))
    receipt = PASourcePrefixSelectionReceiptV3(
        schema=PA_FETCH_SELECTION_SCHEMA_V3,
        enumeration_receipt_sha256=enumeration.receipt_sha256,
        source_route_manifest_sha256=enumeration.source_route_manifest_sha256,
        families=tuple(rows),
        selection_plan_sha256=plan.plan_sha256,
    )
    return plan, receipt


def _write_or_verify_artifact(path: Path, payload: Mapping[str, object]) -> str:
    if not isinstance(path, Path):
        raise TypeError("source fetch artifact path must be a pathlib.Path")
    assert_no_symlink_ancestors(path)
    expected = canonical_json_bytes(payload) + b"\n"
    if expected.endswith(b"\n\n"):
        raise AssertionError("canonical source artifact has more than one final LF")
    if path.exists():
        observed, unused = load_canonical_json_snapshot(path)
        del unused
        if observed != expected:
            raise SourceFetchError(
                "existing source fetch artifact differs; refusing overwrite"
            )
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    assert_no_symlink_ancestors(partial)
    if partial.exists():
        if not partial.is_file():
            raise SourceFetchError("source fetch artifact partial is not a file")
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


def _enumeration_envelope(
    receipt: UpstreamEnumerationReceiptV3,
) -> Mapping[str, object]:
    return {
        "receipt": asdict(receipt),
        "receipt_sha256": receipt.receipt_sha256,
        "schema": UPSTREAM_ENUMERATION_ARTIFACT_SCHEMA_V3,
    }


def _selection_envelope(
    receipt: PASourcePrefixSelectionReceiptV3,
) -> Mapping[str, object]:
    return {
        "receipt": asdict(receipt),
        "receipt_sha256": receipt.receipt_sha256,
        "schema": PA_FETCH_SELECTION_ARTIFACT_SCHEMA_V3,
    }


def _download_envelope(
    receipt: SourceCacheDownloadReceiptV3,
) -> Mapping[str, object]:
    return {
        "receipt": asdict(receipt),
        "receipt_sha256": receipt.receipt_sha256,
        "schema": SOURCE_CACHE_DOWNLOAD_ARTIFACT_SCHEMA_V3,
    }


def prepare_selected_source_cache_v3(
    *,
    enumeration: UpstreamEnumerationReceiptV3,
    cache_root: Path,
    receipt_root: Path,
    open_upstream: Callable[[UpstreamAssetV3], BinaryIO],
    route_manifest_path: Path = SOURCE_ROUTE_MANIFEST_PATH,
    required_bytes_by_family: Mapping[str, int] | None = None,
    allow_nonauthoritative_fixture: bool = False,
) -> PASourcePreparationResultV3:
    """Materialize one finite selection; injection is fixture-only by default."""

    if not isinstance(enumeration, UpstreamEnumerationReceiptV3):
        raise TypeError("source preparation requires a typed enumeration")
    if not enumeration.authoritative and not allow_nonauthoritative_fixture:
        raise SourceFetchError("production source preparation requires AUTHORITATIVE")
    if not all(isinstance(path, Path) for path in (cache_root, receipt_root)):
        raise TypeError("source preparation roots must be pathlib.Path values")
    assert_no_symlink_ancestors(cache_root)
    assert_no_symlink_ancestors(receipt_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    receipt_root.mkdir(parents=True, exist_ok=True)
    resolved_cache = cache_root.resolve(strict=True)
    resolved_receipts = receipt_root.resolve(strict=True)
    if (
        resolved_cache == resolved_receipts
        or resolved_cache in resolved_receipts.parents
        or resolved_receipts in resolved_cache.parents
    ):
        raise SourceFetchError("cache and receipt roots must be disjoint")

    plan, selection = select_required_asset_prefixes_v3(
        enumeration,
        route_manifest_path=route_manifest_path,
        required_bytes_by_family=required_bytes_by_family,
    )
    enumeration_path = resolved_receipts / ENUMERATION_ARTIFACT_NAME
    selection_path = resolved_receipts / SELECTION_ARTIFACT_NAME
    manifest_path = resolved_receipts / SOURCE_MANIFEST_NAME
    download_path = resolved_receipts / DOWNLOAD_ARTIFACT_NAME
    enumeration_artifact_sha256 = _write_or_verify_artifact(
        enumeration_path,
        _enumeration_envelope(enumeration),
    )
    selection_artifact_sha256 = _write_or_verify_artifact(
        selection_path,
        _selection_envelope(selection),
    )
    materialization = materialize_source_cache_v3(
        enumeration,
        plan,
        resolved_cache,
        open_upstream=open_upstream,
        allow_nonauthoritative_fixture=allow_nonauthoritative_fixture,
        resume_incomplete=True,
    )
    download = finalize_source_cache_v3(
        enumeration,
        (materialization,),
        resolved_cache,
        manifest_path,
        allow_nonauthoritative_fixture=allow_nonauthoritative_fixture,
        allow_existing_verified_manifest=True,
    )
    download_artifact_sha256 = _write_or_verify_artifact(
        download_path,
        _download_envelope(download),
    )
    return PASourcePreparationResultV3(
        enumeration=enumeration,
        selection=selection,
        plan=plan,
        download=download,
        enumeration_artifact_sha256=enumeration_artifact_sha256,
        selection_artifact_sha256=selection_artifact_sha256,
        download_artifact_sha256=download_artifact_sha256,
    )


def prepare_pa_sources_online_v3(
    *,
    cache_root: Path,
    transport_cache_root: Path,
    receipt_root: Path,
    route_manifest_path: Path = SOURCE_ROUTE_MANIFEST_PATH,
) -> PASourcePreparationResultV3:
    """Run the concrete, authoritative, fetch-only P-A source boundary."""

    if not all(
        isinstance(path, Path)
        for path in (cache_root, transport_cache_root, receipt_root)
    ):
        raise TypeError("online source roots must be pathlib.Path values")
    roots = (cache_root, transport_cache_root, receipt_root)
    for root in roots:
        assert_no_symlink_ancestors(root)
    resolved_roots = tuple(root.resolve(strict=False) for root in roots)
    for index, left in enumerate(resolved_roots):
        for right in resolved_roots[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise SourceFetchError(
                    "source, transport, and receipt roots must be pairwise disjoint"
                )
    external_cache = ExternalResourceCacheV3(
        transport_cache_root / "external",
    )
    enumeration = enumerate_authoritative_upstream_assets_v3(
        open_resource=external_cache.open,
        route_manifest_path=route_manifest_path,
    )
    if enumeration.mode != AUTHORITATIVE_MODE:
        raise AssertionError("concrete enumerator did not mint AUTHORITATIVE")
    opener = PinnedHuggingFaceAssetOpenerV3(
        transport_cache_root / "huggingface",
        external_cache=external_cache,
    )
    return prepare_selected_source_cache_v3(
        enumeration=enumeration,
        cache_root=cache_root,
        receipt_root=receipt_root,
        open_upstream=opener.open,
        route_manifest_path=route_manifest_path,
    )


__all__ = [
    "DOWNLOAD_ARTIFACT_NAME",
    "ENUMERATION_ARTIFACT_NAME",
    "ExternalCacheObservationV3",
    "ExternalResourceCacheV3",
    "FamilyPrefixSelectionV3",
    "PA_FETCH_SELECTION_ARTIFACT_SCHEMA_V3",
    "PA_FETCH_SELECTION_SCHEMA_V3",
    "PASourcePrefixSelectionReceiptV3",
    "PASourcePreparationResultV3",
    "PinnedHuggingFaceAssetOpenerV3",
    "SELECTION_ARTIFACT_NAME",
    "SOURCE_MANIFEST_NAME",
    "SourceFetchError",
    "prepare_pa_sources_online_v3",
    "prepare_selected_source_cache_v3",
    "select_required_asset_prefixes_v3",
]
