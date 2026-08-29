"""Production-effect helpers for WEFT-1 corpus P-A under Amendment A2.

The pure contracts live in :mod:`training.weft1_corpus_a2`.  This module is
the deliberately narrow filesystem/model boundary: it verifies a pinned
runtime and local source cache, runs the bound FastText adapter, and emits the
bound JSONL/zstd framing.  It never downloads data and never mints D1/D2.
Callers may convert a completed replay mapping into a typed
``ReplayRunReceiptV3`` and then submit two such receipts to the pure validator.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from importlib import metadata
import json
import locale
import os
from pathlib import Path, PurePosixPath
import platform
import re
import socket
import sqlite3
import stat
import sys
import sysconfig
import tempfile
from typing import Any, Protocol
import unicodedata

import zstandard

from training.weft1_corpus_a2 import (
    A2_LANGUAGE_ID_BINDING,
    A2_MATCH_NORMALIZATION_BINDING,
    A2_MINHASH_BINDING,
    A2_ZSTD_CODEC_BINDING,
    CorpusContentManifestV3,
    JsonlZstdShardIdentityV3,
    LanguageIdDecisionV3,
    MinHashRecallAuditV3,
    ProcessAttestationV3,
    ReplayRunReceiptV3,
    StableDocumentV3,
    canonical_jsonl_record_bytes_v3,
    execution_authority_v3_bound_sha256,
    language_backend_input_bytes_v3,
    language_id_decision_v3,
)
from training.weft1_gtok_a1_contract import SOURCE_FAMILIES
from training.weft1_gtok_contract import GTOK_STRATA, canonical_json_bytes
from training.weft1_strict_io import assert_no_symlink_ancestors


DEFAULT_REQUIREMENTS_LOCK = Path(__file__).with_name(
    "weft1_corpus_gtok_a2_requirements.lock"
)
DEFAULT_REQUIREMENTS_LOCK_SHA256 = (
    "bccb8e5b58b5e8fa9eee367fe9c26f59053fff5b7fadf81f23f96b83d1531860"
)
DEFAULT_SHARD_TARGET_BYTES = 512_000_000
DEFAULT_SQLITE_SOURCE_ID = (
    "2024-01-30 16:01:20 "
    "e876e51a0ed5c5b3126f52e532044363a014bc594cfefa87ffb5b82257cc467a"
)
INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA_V3 = (
    "weft1_installed_distribution_inventory_v3"
)
RUNTIME_LINKAGE_SCHEMA_V3 = "weft1_runtime_linkage_v3"
_BOOTSTRAP_DISTRIBUTIONS = frozenset({"pip"})
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_DETERMINISM_ENVIRONMENT_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "LANG",
    "LC_ALL",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
    "TZ",
)
_LOCK_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9_.-]+)==([^\s\\]+)(?:\s+\\)?$"
)
_LOCK_HASH = re.compile(r"^--hash=sha256:([0-9a-f]{64})(?:\s+\\)?$")


class CorpusProductionError(RuntimeError):
    """Fail-closed error at the P-A production boundary."""


def sha256_file(path: Path) -> str:
    """Hash one regular file without loading it into memory."""

    candidate = Path(path)
    if not candidate.is_file():
        raise CorpusProductionError(f"required regular file is absent: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _safe_relative_posix_path(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty relative POSIX path")
    if "\\" in value:
        raise ValueError(f"{name} may not contain backslashes")
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in {"", ".", ".."}
        or part.endswith((".", " "))
        or ":" in part
        for part in path.parts
    ):
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    return path.as_posix()


def _resolve_inside(root: Path, relative_path: str) -> Path:
    assert_no_symlink_ancestors(root)
    resolved_root = root.resolve(strict=True)
    lexical_candidate = resolved_root.joinpath(*PurePosixPath(relative_path).parts)
    assert_no_symlink_ancestors(lexical_candidate)
    resolved_candidate = lexical_candidate.resolve(strict=True)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise CorpusProductionError("source-cache path escapes its root") from error
    return resolved_candidate


@dataclass(frozen=True)
class SourceAssetExpectationV3:
    """One already-materialized source asset and its upstream identity."""

    source: str
    locator: str
    cache_relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if self.source not in SOURCE_FAMILIES:
            raise ValueError("source asset uses an unknown source family")
        if not isinstance(self.locator, str) or not self.locator:
            raise ValueError("source asset locator must be nonempty")
        object.__setattr__(
            self,
            "cache_relative_path",
            _safe_relative_posix_path(
                self.cache_relative_path, "source asset cache_relative_path"
            ),
        )
        if type(self.byte_count) is not int or self.byte_count < 1:
            raise ValueError("source asset byte_count must be a positive integer")
        _require_sha256(self.sha256, "source asset sha256")


@dataclass(frozen=True)
class VerifiedSourceAssetV3:
    source: str
    locator: str
    cache_relative_path: str
    byte_count: int
    sha256: str

    @property
    def identity_payload(self) -> Mapping[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceCacheVerificationV3:
    assets: tuple[VerifiedSourceAssetV3, ...]

    def __post_init__(self) -> None:
        keys = tuple(
            (asset.source, asset.locator, asset.cache_relative_path)
            for asset in self.assets
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("verified source assets require unique canonical order")

    @property
    def manifest_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_source_cache_manifest_v3",
            tuple(asset.identity_payload for asset in self.assets),
        )


def verify_source_cache_assets(
    cache_root: Path,
    expectations: Sequence[SourceAssetExpectationV3],
) -> SourceCacheVerificationV3:
    """Verify local source bytes in a deterministic, network-free pass."""

    root = Path(cache_root)
    assert_no_symlink_ancestors(root)
    if not root.is_dir():
        raise CorpusProductionError(f"source-cache directory is absent: {root}")
    if not isinstance(expectations, Sequence) or isinstance(
        expectations, (str, bytes)
    ):
        raise TypeError("source expectations must be a typed sequence")
    ordered = tuple(
        sorted(
            expectations,
            key=lambda item: (item.source, item.locator, item.cache_relative_path),
        )
    )
    if any(not isinstance(item, SourceAssetExpectationV3) for item in ordered):
        raise TypeError("source expectations contain a non-expectation")
    paths = tuple(item.cache_relative_path for item in ordered)
    if len(paths) != len(set(paths)):
        raise CorpusProductionError("source-cache manifest repeats a local path")

    verified: list[VerifiedSourceAssetV3] = []
    for expectation in ordered:
        path = _resolve_inside(root, expectation.cache_relative_path)
        if path.stat().st_size != expectation.byte_count:
            raise CorpusProductionError(
                f"source asset byte count mismatch: {expectation.cache_relative_path}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expectation.sha256:
            raise CorpusProductionError(
                f"source asset SHA-256 mismatch: {expectation.cache_relative_path}"
            )
        verified.append(
            VerifiedSourceAssetV3(
                source=expectation.source,
                locator=expectation.locator,
                cache_relative_path=expectation.cache_relative_path,
                byte_count=expectation.byte_count,
                sha256=actual_sha256,
            )
        )
    return SourceCacheVerificationV3(tuple(verified))


@dataclass(frozen=True)
class HashLockedDistributionV3:
    """One resolved distribution plus every authority-allowed archive hash."""

    distribution: str
    version: str
    artifact_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = self.distribution.casefold().replace("_", "-")
        if (
            not normalized
            or normalized != self.distribution
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for character in normalized)
        ):
            raise ValueError("hash-locked distribution name is not canonical")
        if not self.version or any(character.isspace() for character in self.version):
            raise ValueError("hash-locked distribution version is invalid")
        if (
            not self.artifact_sha256s
            or self.artifact_sha256s != tuple(sorted(self.artifact_sha256s))
            or len(self.artifact_sha256s) != len(set(self.artifact_sha256s))
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.artifact_sha256s
            )
        ):
            raise ValueError("hash-locked distribution hashes are invalid")


def parse_hash_locked_requirements_v3(
    lock_bytes: bytes,
) -> tuple[HashLockedDistributionV3, ...]:
    """Parse the exact uv hash-lock closure and reject every unhashed row."""

    try:
        text = lock_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CorpusProductionError("dependency lock is not strict UTF-8") from error

    packages: list[HashLockedDistributionV3] = []
    active_name: str | None = None
    active_version: str | None = None
    active_hashes: list[str] = []

    def finish_active() -> None:
        nonlocal active_name, active_version, active_hashes
        if active_name is None or active_version is None:
            return
        if not active_hashes:
            raise CorpusProductionError(
                f"dependency lock distribution lacks hashes: {active_name}"
            )
        try:
            packages.append(
                HashLockedDistributionV3(
                    distribution=active_name,
                    version=active_version,
                    artifact_sha256s=tuple(sorted(active_hashes)),
                )
            )
        except ValueError as error:
            raise CorpusProductionError(str(error)) from error
        active_name = None
        active_version = None
        active_hashes = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        if not line[0].isspace():
            finish_active()
            match = _LOCK_REQUIREMENT.fullmatch(line)
            if match is None:
                raise CorpusProductionError(
                    f"dependency lock row {line_number} is not an exact pin"
                )
            active_name = match.group(1).casefold().replace("_", "-")
            active_version = match.group(2)
            continue
        if active_name is None:
            raise CorpusProductionError(
                f"dependency lock row {line_number} has no distribution"
            )
        hash_match = _LOCK_HASH.fullmatch(line.strip())
        if hash_match is None:
            raise CorpusProductionError(
                f"dependency lock row {line_number} is not a SHA-256 hash"
            )
        active_hashes.append(hash_match.group(1))
    finish_active()

    if not packages:
        raise CorpusProductionError("dependency lock contains no exact pins")
    names = tuple(item.distribution for item in packages)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise CorpusProductionError("dependency lock repeats a distribution")
    return tuple(packages)


def _runtime_canonical_json_line(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _installed_inventory_identity_sha256_v3(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _runtime_canonical_json_line(
            {
                "domain": INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA_V3,
                "inventory": value,
            }
        )
    ).hexdigest()


def _canonical_installed_distribution_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CorpusProductionError("installed distribution lacks a canonical name")
    normalized = re.sub(r"[-_.]+", "-", value).casefold()
    if (
        not normalized
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in normalized
        )
    ):
        raise CorpusProductionError("installed distribution name is not canonical")
    return normalized


def _stable_installed_file_identity(path: Path) -> tuple[int, str, str]:
    """Hash one path through a stable regular-file handle.

    The before/after identity checks make an in-place write or pathname swap a
    failed attestation rather than an inventory of mixed file generations.
    """

    try:
        lexical = assert_no_symlink_ancestors(Path(path))
        before_path = lexical.lstat()
    except (OSError, RuntimeError) as error:
        raise CorpusProductionError("installed file cannot be inspected safely") from error
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise CorpusProductionError("installed artifact is not a regular non-symlink file")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with lexical.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if not stat.S_ISREG(before_handle.st_mode):
                raise CorpusProductionError("installed artifact handle is not regular")
            if (
                before_handle.st_dev != before_path.st_dev
                or before_handle.st_ino != before_path.st_ino
            ):
                raise CorpusProductionError("installed artifact changed before hashing")
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                byte_count += len(chunk)
                digest.update(chunk)
            after_handle = os.fstat(handle.fileno())
        after_path = lexical.lstat()
    except CorpusProductionError:
        raise
    except OSError as error:
        raise CorpusProductionError("installed artifact cannot be hashed") from error
    before_identity = (
        before_handle.st_dev,
        before_handle.st_ino,
        before_handle.st_size,
        before_handle.st_mtime_ns,
    )
    if (
        before_identity
        != (
            after_handle.st_dev,
            after_handle.st_ino,
            after_handle.st_size,
            after_handle.st_mtime_ns,
        )
        or before_identity
        != (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_size,
            after_path.st_mtime_ns,
        )
        or byte_count != before_handle.st_size
    ):
        raise CorpusProductionError("installed artifact changed while being hashed")
    raw_digest = digest.digest()
    return (
        byte_count,
        raw_digest.hex(),
        base64.urlsafe_b64encode(raw_digest).rstrip(b"=").decode("ascii"),
    )


def validate_installed_distribution_inventory_v3(
    value: object,
) -> dict[str, object]:
    """Validate the canonical installed-file inventory without reading disk."""

    if not isinstance(value, Mapping) or set(value) != {
        "bootstrap_distributions",
        "distributions",
        "files",
        "installation_prefix",
        "inventory_identity_sha256",
        "schema",
        "site_roots",
    }:
        raise CorpusProductionError("installed-distribution inventory fields drifted")
    if value.get("schema") != INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA_V3:
        raise CorpusProductionError("installed-distribution inventory schema drifted")
    prefix = value.get("installation_prefix")
    if not isinstance(prefix, str) or not prefix:
        raise CorpusProductionError("installed-distribution inventory prefix is invalid")

    raw_distributions = value.get("distributions")
    if not isinstance(raw_distributions, (list, tuple)) or not raw_distributions:
        raise CorpusProductionError("installed-distribution inventory is empty")
    distributions: list[dict[str, object]] = []
    for index, raw_row in enumerate(raw_distributions):
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "distribution",
            "file_count",
            "record_path",
            "record_sha256",
            "source",
            "version",
        }:
            raise CorpusProductionError(
                f"installed distribution row {index} fields drifted"
            )
        name = _canonical_installed_distribution_name(raw_row.get("distribution"))
        version = raw_row.get("version")
        record_path = raw_row.get("record_path")
        file_count = raw_row.get("file_count")
        source = raw_row.get("source")
        record_sha256 = raw_row.get("record_sha256")
        if (
            not isinstance(version, str)
            or not version
            or not isinstance(record_path, str)
            or not record_path
            or PurePosixPath(record_path).is_absolute()
            or ".." in PurePosixPath(record_path).parts
            or type(file_count) is not int
            or file_count < 1
            or source not in {"cpython_ensurepip", "hash_locked_wheel"}
            or not isinstance(record_sha256, str)
            or len(record_sha256) != 64
            or any(character not in "0123456789abcdef" for character in record_sha256)
        ):
            raise CorpusProductionError("installed distribution row is invalid")
        distributions.append(
            {
                "distribution": name,
                "file_count": file_count,
                "record_path": record_path,
                "record_sha256": record_sha256,
                "source": source,
                "version": version,
            }
        )
    distribution_names = [str(row["distribution"]) for row in distributions]
    if distribution_names != sorted(distribution_names) or len(
        distribution_names
    ) != len(set(distribution_names)):
        raise CorpusProductionError("installed distribution order or uniqueness drifted")

    raw_files = value.get("files")
    if not isinstance(raw_files, (list, tuple)) or not raw_files:
        raise CorpusProductionError("installed-distribution file inventory is empty")
    files: list[dict[str, object]] = []
    known_names = set(distribution_names)
    for index, raw_row in enumerate(raw_files):
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "bytes",
            "owners",
            "relative_path",
            "sha256",
        }:
            raise CorpusProductionError(f"installed file row {index} fields drifted")
        relative_path = raw_row.get("relative_path")
        byte_count = raw_row.get("bytes")
        sha256 = raw_row.get("sha256")
        owners = raw_row.get("owners")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or PurePosixPath(relative_path).is_absolute()
            or ".." in PurePosixPath(relative_path).parts
            or type(byte_count) is not int
            or byte_count < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(owners, (list, tuple))
            or not owners
            or list(owners) != sorted(owners)
            or len(owners) != len(set(owners))
            or any(owner not in known_names for owner in owners)
        ):
            raise CorpusProductionError("installed file row is invalid")
        files.append(
            {
                "bytes": byte_count,
                "owners": list(owners),
                "relative_path": relative_path,
                "sha256": sha256,
            }
        )
    file_paths = [str(row["relative_path"]) for row in files]
    if file_paths != sorted(file_paths) or len(file_paths) != len(set(file_paths)):
        raise CorpusProductionError("installed file order or uniqueness drifted")
    files_by_path = {str(row["relative_path"]): row for row in files}
    for row in distributions:
        record = files_by_path.get(str(row["record_path"]))
        if (
            record is None
            or row["distribution"] not in record["owners"]
            or row["record_sha256"] != record["sha256"]
        ):
            raise CorpusProductionError("distribution RECORD is not file-inventory bound")

    bootstrap = value.get("bootstrap_distributions")
    if not isinstance(bootstrap, (list, tuple)):
        raise CorpusProductionError("bootstrap distribution inventory is invalid")
    expected_bootstrap = [
        {
            "distribution": row["distribution"],
            "version": row["version"],
        }
        for row in distributions
        if row["source"] == "cpython_ensurepip"
    ]
    if list(bootstrap) != expected_bootstrap:
        raise CorpusProductionError("bootstrap distribution inventory drifted")

    site_roots = value.get("site_roots")
    if (
        not isinstance(site_roots, (list, tuple))
        or not site_roots
        or any(
            not isinstance(item, str)
            or not item
            or PurePosixPath(item).is_absolute()
            or ".." in PurePosixPath(item).parts
            for item in site_roots
        )
        or list(site_roots) != sorted(site_roots)
        or len(site_roots) != len(set(site_roots))
    ):
        raise CorpusProductionError("installed site-root inventory drifted")

    normalized = {
        "bootstrap_distributions": expected_bootstrap,
        "distributions": distributions,
        "files": files,
        "installation_prefix": prefix,
        "schema": INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA_V3,
        "site_roots": list(site_roots),
    }
    claimed_identity = value.get("inventory_identity_sha256")
    if (
        not isinstance(claimed_identity, str)
        or claimed_identity != _installed_inventory_identity_sha256_v3(normalized)
    ):
        raise CorpusProductionError("installed-distribution inventory identity drifted")
    return {**normalized, "inventory_identity_sha256": claimed_identity}


def installed_distribution_inventory_v3(
    lock_bytes: bytes,
    *,
    installation_prefix: Path | None = None,
    distributions: Iterable[metadata.Distribution] | None = None,
) -> dict[str, object]:
    """Recompute the complete installed wheel/RECORD tree for this runtime.

    All distributions must be exactly the hash-lock closure plus CPython's
    bootstrap ``pip``.  Every RECORD-listed file is hashed, every RECORD hash
    and size claim is checked, and every file below the observed site-package
    roots must have at least one RECORD owner.
    """

    locked = parse_hash_locked_requirements_v3(lock_bytes)
    locked_versions = {row.distribution: row.version for row in locked}
    expected_names = set(locked_versions) | set(_BOOTSTRAP_DISTRIBUTIONS)
    prefix = Path(sys.prefix if installation_prefix is None else installation_prefix)
    try:
        prefix = assert_no_symlink_ancestors(prefix).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CorpusProductionError("installed-distribution prefix is unsafe") from error
    if not prefix.is_dir():
        raise CorpusProductionError("installed-distribution prefix is not a directory")

    observed: dict[str, metadata.Distribution] = {}
    source = metadata.distributions() if distributions is None else distributions
    for distribution in source:
        name = _canonical_installed_distribution_name(
            distribution.metadata.get("Name")
        )
        if name in observed:
            raise CorpusProductionError(f"installed distribution is repeated: {name}")
        observed[name] = distribution
    unexpected = sorted(set(observed) - expected_names)
    missing = sorted(expected_names - set(observed))
    if unexpected:
        raise CorpusProductionError(
            "unexpected installed distributions: " + ", ".join(unexpected)
        )
    if missing:
        raise CorpusProductionError(
            "pinned distributions are absent: " + ", ".join(missing)
        )

    initial_files: dict[str, tuple[int, str]] = {}
    owners: dict[str, set[str]] = {}
    distribution_rows: list[dict[str, object]] = []
    site_roots: set[str] = set()
    for name in sorted(observed):
        distribution = observed[name]
        version = distribution.version
        if not isinstance(version, str) or not version:
            raise CorpusProductionError(f"installed distribution version is absent: {name}")
        if name in locked_versions and version != locked_versions[name]:
            raise CorpusProductionError(
                f"distribution version mismatch for {name}: "
                f"expected {locked_versions[name]}, observed {version}"
            )
        try:
            site_root = assert_no_symlink_ancestors(
                Path(distribution.locate_file(""))
            ).resolve(strict=True)
            site_relative = site_root.relative_to(prefix).as_posix()
        except (OSError, RuntimeError, ValueError) as error:
            raise CorpusProductionError(
                f"installed distribution root escapes the runtime prefix: {name}"
            ) from error
        if not site_relative or site_relative == ".":
            raise CorpusProductionError("distribution site root may not equal the prefix")
        site_roots.add(site_relative)
        package_files = distribution.files
        if not package_files:
            raise CorpusProductionError(f"installed distribution lacks RECORD files: {name}")
        seen_package_paths: set[str] = set()
        record_paths: list[str] = []
        for package_path in package_files:
            package_name = str(package_path).replace("\\", "/")
            if package_name in seen_package_paths:
                raise CorpusProductionError(f"distribution RECORD repeats a path: {name}")
            seen_package_paths.add(package_name)
            try:
                lexical = assert_no_symlink_ancestors(
                    Path(distribution.locate_file(package_path))
                )
                resolved = lexical.resolve(strict=True)
                relative_path = resolved.relative_to(prefix).as_posix()
            except (OSError, RuntimeError, ValueError) as error:
                raise CorpusProductionError(
                    f"distribution RECORD path escapes the runtime prefix: {name}"
                ) from error
            byte_count, digest, record_digest = _stable_installed_file_identity(
                resolved
            )
            declared_size = package_path.size
            declared_hash = package_path.hash
            if declared_size is not None and int(declared_size) != byte_count:
                raise CorpusProductionError(f"distribution RECORD size mismatch: {name}")
            if declared_hash is not None and (
                declared_hash.mode != "sha256" or declared_hash.value != record_digest
            ):
                raise CorpusProductionError(f"distribution RECORD hash mismatch: {name}")
            prior = initial_files.get(relative_path)
            if prior is not None and prior != (byte_count, digest):
                raise CorpusProductionError("overlapping distributions disagree on file bytes")
            initial_files[relative_path] = (byte_count, digest)
            owners.setdefault(relative_path, set()).add(name)
            pure_path = PurePosixPath(package_name)
            # Modern wheels can vendor complete nested ``*.dist-info`` trees
            # (setuptools 84 does this).  Those nested RECORD files are owned
            # payload, not the installed distribution's own metadata root.
            if (
                len(pure_path.parts) == 2
                and pure_path.name == "RECORD"
                and pure_path.parent.name.endswith(".dist-info")
            ):
                record_paths.append(relative_path)
        if len(record_paths) != 1:
            raise CorpusProductionError(f"distribution must contain one RECORD: {name}")
        record_path = record_paths[0]
        distribution_rows.append(
            {
                "distribution": name,
                "file_count": len(seen_package_paths),
                "record_path": record_path,
                "record_sha256": initial_files[record_path][1],
                "source": (
                    "cpython_ensurepip"
                    if name in _BOOTSTRAP_DISTRIBUTIONS
                    else "hash_locked_wheel"
                ),
                "version": version,
            }
        )

    # A rogue module can shadow a locked package without adding dist-info.
    # Require every regular file under each package root to have a RECORD owner.
    for relative_root in sorted(site_roots):
        root = prefix.joinpath(*PurePosixPath(relative_root).parts)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            try:
                metadata_row = path.lstat()
            except OSError as error:
                raise CorpusProductionError("installed tree cannot be enumerated") from error
            if stat.S_ISDIR(metadata_row.st_mode):
                continue
            if stat.S_ISLNK(metadata_row.st_mode) or not stat.S_ISREG(
                metadata_row.st_mode
            ):
                raise CorpusProductionError(
                    "installed tree contains a link or special artifact"
                )
            relative_path = path.relative_to(prefix).as_posix()
            if relative_path not in owners:
                raise CorpusProductionError(
                    f"installed tree contains an unowned file: {relative_path}"
                )

    file_rows: list[dict[str, object]] = []
    for relative_path in sorted(initial_files):
        path = prefix.joinpath(*PurePosixPath(relative_path).parts)
        byte_count, digest, _ = _stable_installed_file_identity(path)
        if (byte_count, digest) != initial_files[relative_path]:
            raise CorpusProductionError("installed artifact changed during inventory")
        file_rows.append(
            {
                "bytes": byte_count,
                "owners": sorted(owners[relative_path]),
                "relative_path": relative_path,
                "sha256": digest,
            }
        )

    core: dict[str, object] = {
        "bootstrap_distributions": [
            {
                "distribution": row["distribution"],
                "version": row["version"],
            }
            for row in distribution_rows
            if row["source"] == "cpython_ensurepip"
        ],
        "distributions": distribution_rows,
        "files": file_rows,
        "installation_prefix": str(prefix),
        "schema": INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA_V3,
        "site_roots": sorted(site_roots),
    }
    inventory = {
        **core,
        "inventory_identity_sha256": _installed_inventory_identity_sha256_v3(core),
    }
    return validate_installed_distribution_inventory_v3(inventory)


def _runtime_artifact_row_v3(path: Path, *, prefix: Path) -> dict[str, object]:
    try:
        lexical = assert_no_symlink_ancestors(Path(path))
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(prefix)
    except (OSError, RuntimeError, ValueError) as error:
        raise CorpusProductionError(
            "runtime linkage artifact escapes the governed prefix"
        ) from error
    byte_count, digest, _ = _stable_installed_file_identity(resolved)
    return {"bytes": byte_count, "path": str(resolved), "sha256": digest}


def validate_runtime_linkage_inventory_v3(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "executable",
        "libpython_library",
        "linkage_identity_sha256",
        "schema",
        "sqlite_extension",
        "sqlite_library",
    }:
        raise CorpusProductionError("runtime linkage inventory fields drifted")
    if value.get("schema") != RUNTIME_LINKAGE_SCHEMA_V3:
        raise CorpusProductionError("runtime linkage inventory schema drifted")
    rows: dict[str, dict[str, object]] = {}
    for name in (
        "executable",
        "libpython_library",
        "sqlite_extension",
        "sqlite_library",
    ):
        raw = value.get(name)
        if not isinstance(raw, Mapping) or set(raw) != {"bytes", "path", "sha256"}:
            raise CorpusProductionError("runtime linkage artifact fields drifted")
        path = raw.get("path")
        byte_count = raw.get("bytes")
        sha256 = raw.get("sha256")
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or type(byte_count) is not int
            or byte_count < 1
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise CorpusProductionError("runtime linkage artifact row is invalid")
        rows[name] = {"bytes": byte_count, "path": path, "sha256": sha256}
    core: dict[str, object] = {
        "executable": rows["executable"],
        "libpython_library": rows["libpython_library"],
        "schema": RUNTIME_LINKAGE_SCHEMA_V3,
        "sqlite_extension": rows["sqlite_extension"],
        "sqlite_library": rows["sqlite_library"],
    }
    identity = value.get("linkage_identity_sha256")
    if identity != execution_authority_v3_bound_sha256(
        RUNTIME_LINKAGE_SCHEMA_V3, core
    ):
        raise CorpusProductionError("runtime linkage inventory identity drifted")
    return {**core, "linkage_identity_sha256": identity}


def runtime_linkage_inventory_v3(
    executable: Path | None = None,
    *,
    maps_path: Path = Path("/proc/self/maps"),
    installation_prefix: Path | None = None,
) -> dict[str, object]:
    """Bind the exact extension and loaded shared-library bytes on Linux."""

    prefix = Path(sys.prefix if installation_prefix is None else installation_prefix)
    try:
        prefix = assert_no_symlink_ancestors(prefix).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CorpusProductionError("runtime linkage prefix is unsafe") from error
    executable_path = Path(sys.executable if executable is None else executable)
    try:
        executable_path = assert_no_symlink_ancestors(executable_path).resolve(strict=True)
        executable_path.relative_to(prefix)
    except (OSError, RuntimeError, ValueError) as error:
        raise CorpusProductionError(
            "runtime executable escapes the governed prefix"
        ) from error

    try:
        import _sqlite3

        sqlite_extension_path = Path(_sqlite3.__file__)
    except (ImportError, TypeError) as error:
        raise CorpusProductionError("runtime SQLite extension is unavailable") from error

    try:
        maps_bytes = Path(maps_path).read_bytes()
        maps_text = maps_bytes.decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise CorpusProductionError("runtime process maps are unavailable") from error
    mapped: dict[str, set[Path]] = {"libpython": set(), "libsqlite3": set()}
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw_path = fields[5]
        if raw_path.endswith(" (deleted)"):
            raise CorpusProductionError("runtime linkage contains a deleted mapping")
        basename = PurePosixPath(raw_path).name
        key: str | None = None
        if basename.startswith("libpython3.11.so"):
            key = "libpython"
        elif basename.startswith("libsqlite3.so"):
            key = "libsqlite3"
        if key is not None:
            try:
                mapped[key].add(Path(raw_path).resolve(strict=True))
            except OSError as error:
                raise CorpusProductionError(
                    "runtime mapped library is absent"
                ) from error
    if any(len(paths) != 1 for paths in mapped.values()):
        raise CorpusProductionError(
            "runtime linkage must contain exactly one libpython and libsqlite3"
        )
    libpython_path = next(iter(mapped["libpython"]))
    sqlite_library_path = next(iter(mapped["libsqlite3"]))
    core: dict[str, object] = {
        "executable": _runtime_artifact_row_v3(executable_path, prefix=prefix),
        "libpython_library": _runtime_artifact_row_v3(libpython_path, prefix=prefix),
        "schema": RUNTIME_LINKAGE_SCHEMA_V3,
        "sqlite_extension": _runtime_artifact_row_v3(
            sqlite_extension_path, prefix=prefix
        ),
        "sqlite_library": _runtime_artifact_row_v3(
            sqlite_library_path, prefix=prefix
        ),
    }
    core["linkage_identity_sha256"] = execution_authority_v3_bound_sha256(
        RUNTIME_LINKAGE_SCHEMA_V3, core
    )
    return validate_runtime_linkage_inventory_v3(core)


@dataclass(frozen=True)
class RuntimeExpectationV3:
    python_version: str = "3.11.9"
    unicode_data_version: str = "14.0.0"
    sqlite_version: str = "3.45.1"
    sqlite_source_id: str = DEFAULT_SQLITE_SOURCE_ID
    zstandard_package_version: str = "0.25.0"
    libzstd_version: str = "1.5.7"
    required_environment: tuple[tuple[str, str], ...] = (
        ("TOKENIZERS_PARALLELISM", "false"),
    )


@dataclass(frozen=True)
class RuntimeAttestationV3:
    executable_sha256: str
    dependency_lock_sha256: str
    environment_identity_sha256: str
    environment_payload: Mapping[str, object]

    def process_attestation(self, output_root: Path) -> ProcessAttestationV3:
        return ProcessAttestationV3(
            executable_sha256=self.executable_sha256,
            dependency_lock_sha256=self.dependency_lock_sha256,
            environment_identity_sha256=self.environment_identity_sha256,
            process_id=os.getpid(),
            output_root=str(Path(output_root).resolve()),
        )


def attest_runtime_v3(
    *,
    requirements_lock: Path = DEFAULT_REQUIREMENTS_LOCK,
    expected_lock_sha256: str = DEFAULT_REQUIREMENTS_LOCK_SHA256,
    expectation: RuntimeExpectationV3 = RuntimeExpectationV3(),
    executable: Path | None = None,
    version_lookup: Callable[[str], str] = metadata.version,
    inventory_builder: Callable[[bytes], Mapping[str, object]] | None = None,
    linkage_builder: Callable[[Path], Mapping[str, object]] | None = None,
) -> RuntimeAttestationV3:
    """Fail closed unless every pinned runtime component matches exactly."""

    _require_sha256(expected_lock_sha256, "expected dependency-lock SHA-256")
    lock_path = Path(requirements_lock)
    lock_bytes = lock_path.read_bytes()
    actual_lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    if actual_lock_sha256 != expected_lock_sha256:
        raise CorpusProductionError("dependency-lock SHA-256 differs from authority")

    with sqlite3.connect(":memory:") as sqlite_probe:
        sqlite_source_id = sqlite_probe.execute(
            "SELECT sqlite_source_id()"
        ).fetchone()[0]
    exact_observations = {
        "python_version": platform.python_version(),
        "unicode_data_version": unicodedata.unidata_version,
        "sqlite_version": sqlite3.sqlite_version,
        "sqlite_source_id": sqlite_source_id,
        "zstandard_package_version": zstandard.__version__,
        "libzstd_version": ".".join(str(value) for value in zstandard.ZSTD_VERSION),
    }
    expected_observations = {
        "python_version": expectation.python_version,
        "unicode_data_version": expectation.unicode_data_version,
        "sqlite_version": expectation.sqlite_version,
        "sqlite_source_id": expectation.sqlite_source_id,
        "zstandard_package_version": expectation.zstandard_package_version,
        "libzstd_version": expectation.libzstd_version,
    }
    if exact_observations != expected_observations:
        raise CorpusProductionError(
            "runtime versions differ from authority: "
            f"expected={expected_observations!r} actual={exact_observations!r}"
        )

    locked_distributions = parse_hash_locked_requirements_v3(lock_bytes)
    observed_distributions: list[dict[str, object]] = []
    for locked in locked_distributions:
        try:
            actual_version = version_lookup(locked.distribution)
        except metadata.PackageNotFoundError as error:
            raise CorpusProductionError(
                f"pinned distribution is absent: {locked.distribution}"
            ) from error
        if actual_version != locked.version:
            raise CorpusProductionError(
                f"distribution version mismatch for {locked.distribution}: "
                f"expected {locked.version}, observed {actual_version}"
            )
        observed_distributions.append(
            {
                "artifact_sha256s": locked.artifact_sha256s,
                "distribution": locked.distribution,
                "version": actual_version,
            }
        )

    try:
        installed_inventory = validate_installed_distribution_inventory_v3(
            (
                installed_distribution_inventory_v3(lock_bytes)
                if inventory_builder is None
                else inventory_builder(lock_bytes)
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, CorpusProductionError):
            raise
        raise CorpusProductionError(
            "installed-distribution integrity attestation failed"
        ) from error
    installed_rows = installed_inventory["distributions"]
    if not isinstance(installed_rows, list):
        raise CorpusProductionError("installed distribution inventory is malformed")
    installed_locked = [
        (str(row["distribution"]), str(row["version"]))
        for row in installed_rows
        if row["source"] == "hash_locked_wheel"
    ]
    expected_locked = [
        (row.distribution, row.version) for row in locked_distributions
    ]
    if installed_locked != expected_locked:
        raise CorpusProductionError(
            "installed-distribution inventory differs from the lock closure"
        )

    for key, expected_value in expectation.required_environment:
        if os.environ.get(key) != expected_value:
            raise CorpusProductionError(
                f"environment variable {key} must equal {expected_value!r}"
            )

    executable_path = Path(sys.executable if executable is None else executable)
    executable_sha256 = sha256_file(executable_path.resolve(strict=True))
    try:
        runtime_linkage = validate_runtime_linkage_inventory_v3(
            runtime_linkage_inventory_v3(executable_path)
            if linkage_builder is None
            else linkage_builder(executable_path)
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, CorpusProductionError):
            raise
        raise CorpusProductionError("runtime linkage attestation failed") from error
    executable_linkage = runtime_linkage.get("executable")
    if not isinstance(executable_linkage, Mapping) or (
        executable_linkage.get("sha256") != executable_sha256
    ):
        raise CorpusProductionError("runtime linkage attestation is not executable-bound")
    environment_payload: dict[str, object] = {
        "byteorder": sys.byteorder,
        "cache_tag": sys.implementation.cache_tag,
        "dependency_lock_sha256": actual_lock_sha256,
        "distributions": tuple(observed_distributions),
        "environment": tuple(
            (key, os.environ.get(key, "<unset>"))
            for key in _DETERMINISM_ENVIRONMENT_KEYS
        ),
        "filesystem_encoding": sys.getfilesystemencoding(),
        "implementation": platform.python_implementation(),
        "installed_distribution_inventory": installed_inventory,
        "locale": locale.setlocale(locale.LC_ALL, None),
        "machine": platform.machine(),
        "maxunicode": sys.maxunicode,
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "preferred_encoding": locale.getpreferredencoding(False),
        "python_executable_sha256": executable_sha256,
        "runtime_versions": exact_observations,
        "runtime_linkage": runtime_linkage,
    }
    environment_identity = execution_authority_v3_bound_sha256(
        "weft1_corpus_execution_environment_v3", environment_payload
    )
    return RuntimeAttestationV3(
        executable_sha256=executable_sha256,
        dependency_lock_sha256=actual_lock_sha256,
        environment_identity_sha256=environment_identity,
        environment_payload=environment_payload,
    )


class _FastTextPredict(Protocol):
    def __call__(
        self,
        text: str,
        k: int,
        threshold: float,
        on_unicode_error: str,
    ) -> Sequence[tuple[float, str]]: ...


def classify_fasttext_backend_v3(
    document: StableDocumentV3,
    *,
    predict: _FastTextPredict,
    label_count: int,
) -> LanguageIdDecisionV3:
    """Run the bound private adapter with a deterministic lexical tie break."""

    if type(label_count) is not int or label_count < 1:
        raise ValueError("FastText label_count must be a positive integer")
    scoring = language_backend_input_bytes_v3(document)
    rows = tuple(
        predict(
            scoring.decode("utf-8", errors="strict"),
            label_count,
            0.0,
            "strict",
        )
    )
    if not rows:
        raise CorpusProductionError("FastText returned no language labels")
    if len(rows) != label_count:
        raise CorpusProductionError("FastText did not return its complete label inventory")
    validated: list[tuple[float, str]] = []
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            raise CorpusProductionError("FastText returned a malformed prediction")
        probability, label = row
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise CorpusProductionError("FastText probability is not numeric")
        if not isinstance(label, str) or not label:
            raise CorpusProductionError("FastText label is not a nonempty string")
        try:
            label.encode("ascii")
        except UnicodeEncodeError as error:
            raise CorpusProductionError("FastText labels must be ASCII") from error
        probability_float = float(probability)
        if not 0.0 <= probability_float <= 1.0:
            raise CorpusProductionError("FastText probability lies outside [0, 1]")
        validated.append((probability_float, label))
    if len({label for _, label in validated}) != len(validated):
        raise CorpusProductionError("FastText returned a duplicate language label")
    probability, label = min(validated, key=lambda row: (-row[0], row[1]))
    return language_id_decision_v3(
        document,
        label=label,
        probability=probability,
    )


class FastTextLanguageIdAdapterV3:
    """Verified ``lid.176.bin`` model using ``model.f.predict`` directly."""

    def __init__(self, model_path: Path) -> None:
        path = assert_no_symlink_ancestors(Path(model_path))
        if metadata.version(A2_LANGUAGE_ID_BINDING.package) != (
            A2_LANGUAGE_ID_BINDING.package_version
        ):
            raise CorpusProductionError("fasttext-wheel version differs from A2")
        private_root = tempfile.TemporaryDirectory(prefix="weft1-fasttext-")
        private_path = Path(private_root.name) / "lid.176.bin"
        digest = hashlib.sha256()
        copied_bytes = 0
        try:
            with path.open("rb") as source:
                source_stat = os.fstat(source.fileno())
                if not stat.S_ISREG(source_stat.st_mode):
                    raise CorpusProductionError("FastText model is not a regular file")
                with private_path.open("xb") as destination:
                    for chunk in iter(lambda: source.read(_HASH_CHUNK_BYTES), b""):
                        copied_bytes += len(chunk)
                        if copied_bytes > A2_LANGUAGE_ID_BINDING.model_bytes:
                            raise CorpusProductionError(
                                "FastText model byte count differs from A2"
                            )
                        digest.update(chunk)
                        destination.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
            if copied_bytes != A2_LANGUAGE_ID_BINDING.model_bytes:
                raise CorpusProductionError("FastText model byte count differs from A2")
            if digest.hexdigest() != A2_LANGUAGE_ID_BINDING.model_sha256:
                raise CorpusProductionError("FastText model SHA-256 differs from A2")

            import fasttext  # imported only after the immutable snapshot preflight

            model = fasttext.load_model(str(private_path))
            labels = tuple(model.get_labels())
            if (
                len(labels) != len(set(labels))
                or A2_LANGUAGE_ID_BINDING.keep_label not in labels
                or any(
                    not isinstance(label, str)
                    or not label
                    or not label.isascii()
                    for label in labels
                )
            ):
                raise CorpusProductionError("FastText model label inventory is invalid")
        except OSError as error:
            private_root.cleanup()
            raise CorpusProductionError("cannot read the FastText model snapshot") from error
        except Exception:
            private_root.cleanup()
            raise
        self._model = model
        self._label_count = len(labels)
        # FastText may retain lazy access to its model file.  Keep the verified
        # private snapshot alive for exactly as long as the adapter.
        self._model_snapshot_root = private_root

    def classify(self, document: StableDocumentV3) -> LanguageIdDecisionV3:
        return classify_fasttext_backend_v3(
            document,
            predict=self._model.f.predict,
            label_count=self._label_count,
        )


@dataclass(frozen=True)
class RawDocumentV3:
    """Raw source text before strict UTF-8 validity screening."""

    source: str
    stratum: str
    stable_source_record_id: str
    text: str | bytes

    def __post_init__(self) -> None:
        if self.source not in SOURCE_FAMILIES:
            raise ValueError("raw document uses an unknown source family")
        if self.stratum not in GTOK_STRATA:
            raise ValueError("raw document uses an unknown corpus stratum")
        _require_sha256(self.stable_source_record_id, "stable_source_record_id")
        if not isinstance(self.text, (str, bytes)):
            raise TypeError("raw document text must be exact str or bytes")


@dataclass(frozen=True)
class ShardWriteResultV3:
    shards: tuple[JsonlZstdShardIdentityV3, ...]
    invalid_utf8_by_source: tuple[tuple[str, int], ...]
    valid_record_count: int
    retained_text_bytes: int
    oversized_singleton_count: int

    @property
    def invalid_utf8_total(self) -> int:
        return sum(count for _, count in self.invalid_utf8_by_source)


@dataclass
class _OpenShard:
    relative_path: str
    final_path: Path
    partial_path: Path
    raw_handle: Any
    zstd_handle: Any
    logical_sha256: Any
    logical_bytes: int = 0
    retained_text_bytes: int = 0
    record_count: int = 0


def _open_shard(
    output_root: Path,
    *,
    stream: str,
    stratum: str,
    index: int,
) -> _OpenShard:
    relative_path = f"{stratum}/{stream.casefold()}-{index:05d}.jsonl.zst"
    final_path = output_root.joinpath(*PurePosixPath(relative_path).parts)
    partial_path = final_path.with_name(final_path.name + ".partial")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists() or partial_path.exists():
        raise CorpusProductionError(f"refusing to overwrite shard: {relative_path}")
    raw_handle = partial_path.open("xb")
    compressor = zstandard.ZstdCompressor(
        level=A2_ZSTD_CODEC_BINDING.compression_level,
        threads=A2_ZSTD_CODEC_BINDING.threads,
        write_checksum=A2_ZSTD_CODEC_BINDING.write_checksum,
        write_content_size=A2_ZSTD_CODEC_BINDING.write_content_size,
        write_dict_id=A2_ZSTD_CODEC_BINDING.write_dict_id,
    )
    zstd_handle = compressor.stream_writer(raw_handle, closefd=False)
    return _OpenShard(
        relative_path=relative_path,
        final_path=final_path,
        partial_path=partial_path,
        raw_handle=raw_handle,
        zstd_handle=zstd_handle,
        logical_sha256=hashlib.sha256(),
    )


def _close_shard(shard: _OpenShard) -> JsonlZstdShardIdentityV3:
    try:
        shard.zstd_handle.close()
        shard.raw_handle.flush()
        os.fsync(shard.raw_handle.fileno())
        shard.raw_handle.close()
        os.replace(shard.partial_path, shard.final_path)
    except BaseException:
        try:
            shard.zstd_handle.close()
        except BaseException:
            pass
        try:
            shard.raw_handle.close()
        except BaseException:
            pass
        if shard.partial_path.exists():
            shard.partial_path.unlink()
        raise
    return JsonlZstdShardIdentityV3(
        relative_path=shard.relative_path,
        record_count=shard.record_count,
        retained_text_bytes=shard.retained_text_bytes,
        logical_jsonl_sha256=shard.logical_sha256.hexdigest(),
        logical_jsonl_bytes=shard.logical_bytes,
        zstd_sha256=sha256_file(shard.final_path),
        zstd_bytes=shard.final_path.stat().st_size,
    )


def _decode_raw_document(raw: RawDocumentV3) -> StableDocumentV3 | None:
    try:
        if isinstance(raw.text, bytes):
            text = raw.text.decode("utf-8", errors="strict")
        else:
            raw.text.encode("utf-8", errors="strict")
            text = raw.text
    except UnicodeError:
        return None
    return StableDocumentV3(
        source=raw.source,
        stratum=raw.stratum,
        stable_source_record_id=raw.stable_source_record_id,
        text=text,
    )


def write_jsonl_zstd_shards_v3(
    documents: Iterable[RawDocumentV3 | StableDocumentV3],
    output_root: Path,
    *,
    stream: str,
    stratum: str,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
) -> ShardWriteResultV3:
    """Emit deterministic A2 shards, dropping whole invalid-UTF-8 documents."""

    if stream not in {"T", "H"}:
        raise ValueError("stream must be T or H")
    if stratum not in GTOK_STRATA:
        raise ValueError("stratum is not registered")
    if type(shard_target_bytes) is not int or shard_target_bytes < 1:
        raise ValueError("shard_target_bytes must be a positive integer")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    invalid: Counter[str] = Counter()
    identities: list[JsonlZstdShardIdentityV3] = []
    current: _OpenShard | None = None
    valid_record_count = 0
    retained_text_bytes = 0
    oversized_singletons = 0
    try:
        for item in documents:
            if isinstance(item, StableDocumentV3):
                document = item
            elif isinstance(item, RawDocumentV3):
                document = _decode_raw_document(item)
                if document is None:
                    invalid[item.source] += 1
                    continue
            else:
                raise TypeError("shard input contains an untyped document")
            if document.stratum != stratum:
                raise ValueError("document stratum differs from shard stratum")

            record = canonical_jsonl_record_bytes_v3(document)
            if current is not None and current.logical_bytes + len(record) > (
                shard_target_bytes
            ):
                identities.append(_close_shard(current))
                current = None
            if current is None:
                current = _open_shard(
                    root,
                    stream=stream,
                    stratum=stratum,
                    index=len(identities),
                )
            current.zstd_handle.write(record)
            current.logical_sha256.update(record)
            current.logical_bytes += len(record)
            current.retained_text_bytes += document.retained_byte_count
            current.record_count += 1
            valid_record_count += 1
            retained_text_bytes += document.retained_byte_count
            if len(record) > shard_target_bytes:
                oversized_singletons += 1
                identities.append(_close_shard(current))
                current = None
        if current is not None:
            identities.append(_close_shard(current))
            current = None
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
            if current.partial_path.exists():
                current.partial_path.unlink()
        raise

    if not identities:
        raise CorpusProductionError("shard write retained no valid documents")
    return ShardWriteResultV3(
        shards=tuple(identities),
        invalid_utf8_by_source=tuple(sorted(invalid.items())),
        valid_record_count=valid_record_count,
        retained_text_bytes=retained_text_bytes,
        oversized_singleton_count=oversized_singletons,
    )


def _json_no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise CorpusProductionError(f"fixture repeats JSON key: {key}")
        output[key] = value
    return output


def _fixture_raw_document(value: object, expected_stratum: str) -> RawDocumentV3:
    if not isinstance(value, Mapping):
        raise CorpusProductionError("fixture document must be an object")
    allowed = {
        "source",
        "stratum",
        "stable_source_record_id",
        "text",
        "text_utf8_hex",
    }
    if set(value) - allowed:
        raise CorpusProductionError("fixture document contains unknown keys")
    has_text = "text" in value
    has_hex = "text_utf8_hex" in value
    if has_text == has_hex:
        raise CorpusProductionError("fixture document requires exactly one text field")
    raw_text: str | bytes
    if has_text:
        raw_text = value["text"]  # type: ignore[assignment]
        if not isinstance(raw_text, str):
            raise CorpusProductionError("fixture text must be a string")
    else:
        encoded = value["text_utf8_hex"]
        if not isinstance(encoded, str):
            raise CorpusProductionError("fixture text_utf8_hex must be a string")
        try:
            raw_text = bytes.fromhex(encoded)
        except ValueError as error:
            raise CorpusProductionError("fixture text_utf8_hex is invalid") from error
    stratum = value.get("stratum", expected_stratum)
    if stratum != expected_stratum:
        raise CorpusProductionError("fixture document stratum differs from fixture")
    try:
        return RawDocumentV3(
            source=value["source"],  # type: ignore[arg-type]
            stratum=stratum,  # type: ignore[arg-type]
            stable_source_record_id=value[
                "stable_source_record_id"
            ],  # type: ignore[arg-type]
            text=raw_text,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CorpusProductionError("fixture document is invalid") from error


def _empty_source_manifest_sha256() -> str:
    return execution_authority_v3_bound_sha256(
        "weft1_corpus_source_cache_manifest_v3", ()
    )


def run_fixture_replay(
    fixture_path: Path,
    output_root: Path,
    *,
    network_disabled: bool = True,
) -> Mapping[str, object]:
    """Run a bounded offline fixture replay and return non-authoritative evidence.

    This helper intentionally does not fabricate the MinHash recall audit and
    therefore cannot mint D1/D2.  ``typed_replay_receipt_from_mapping`` accepts
    that independently generated audit before the pure replay validator is used.
    """

    if network_disabled is not True:
        raise CorpusProductionError("fixture replay is offline-only")
    fixture = Path(fixture_path)
    fixture_bytes = fixture.read_bytes()
    try:
        payload = json.loads(
            fixture_bytes.decode("utf-8", errors="strict"),
            object_pairs_hook=_json_no_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CorpusProductionError(f"fixture uses non-finite JSON: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CorpusProductionError("fixture is not strict JSON/UTF-8") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "weft1_corpus_pa_fixture_v3"
    ):
        raise CorpusProductionError("fixture schema is not registered")
    if set(payload) - {
        "schema",
        "stream",
        "stratum",
        "shard_target_bytes",
        "documents",
        "source_cache_root",
        "source_assets",
    }:
        raise CorpusProductionError("fixture contains unknown top-level keys")

    stream = payload.get("stream")
    stratum = payload.get("stratum")
    if stream not in {"T", "H"} or stratum not in GTOK_STRATA:
        raise CorpusProductionError("fixture stream or stratum is invalid")
    document_values = payload.get("documents")
    if not isinstance(document_values, list) or not document_values:
        raise CorpusProductionError("fixture requires a nonempty document list")
    documents = tuple(
        _fixture_raw_document(value, stratum) for value in document_values
    )

    source_values = payload.get("source_assets", [])
    if not isinstance(source_values, list):
        raise CorpusProductionError("fixture source_assets must be a list")
    if source_values:
        cache_relative = payload.get("source_cache_root", "source_cache")
        if not isinstance(cache_relative, str):
            raise CorpusProductionError("fixture source_cache_root must be a string")
        cache_relative = _safe_relative_posix_path(
            cache_relative, "fixture source_cache_root"
        )
        expectations: list[SourceAssetExpectationV3] = []
        for value in source_values:
            if not isinstance(value, Mapping):
                raise CorpusProductionError("fixture source asset must be an object")
            try:
                expectations.append(SourceAssetExpectationV3(**value))
            except (TypeError, ValueError) as error:
                raise CorpusProductionError(
                    "fixture source asset is invalid"
                ) from error
        verification = verify_source_cache_assets(
            fixture.parent.joinpath(*PurePosixPath(cache_relative).parts),
            expectations,
        )
        source_manifest_sha256 = verification.manifest_sha256
    else:
        source_manifest_sha256 = _empty_source_manifest_sha256()

    runtime = attest_runtime_v3()
    process = runtime.process_attestation(output_root)
    target = payload.get("shard_target_bytes", DEFAULT_SHARD_TARGET_BYTES)
    result = write_jsonl_zstd_shards_v3(
        documents,
        output_root,
        stream=stream,
        stratum=stratum,
        shard_target_bytes=target,
    )

    valid_documents = tuple(
        document
        for raw in documents
        if (document := _decode_raw_document(raw)) is not None
    )
    ordered_document_ids = tuple(document.document_id for document in valid_documents)
    input_identity_sha256 = execution_authority_v3_bound_sha256(
        "weft1_corpus_fixture_input_v3",
        {
            "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
            "source_asset_manifest_sha256": source_manifest_sha256,
        },
    )
    dedup_ledger_sha256 = execution_authority_v3_bound_sha256(
        "weft1_corpus_fixture_noop_dedup_ledger_v3", ordered_document_ids
    )
    language_manifest_sha256 = execution_authority_v3_bound_sha256(
        "weft1_corpus_fixture_language_manifest_v3",
        {"status": "not_run_non_authoritative_fixture"},
    )
    selection_manifest_sha256 = execution_authority_v3_bound_sha256(
        "weft1_corpus_fixture_selection_manifest_v3",
        {
            "invalid_utf8_by_source": result.invalid_utf8_by_source,
            "ordered_document_ids": ordered_document_ids,
            "stream": stream,
            "stratum": stratum,
        },
    )
    algorithm_manifest_sha256 = execution_authority_v3_bound_sha256(
        "weft1_corpus_fixture_algorithm_manifest_v3",
        {
            "language_id_binding_sha256": A2_LANGUAGE_ID_BINDING.receipt_sha256,
            "match_normalization_binding_sha256": (
                A2_MATCH_NORMALIZATION_BINDING.receipt_sha256
            ),
            "minhash_binding_sha256": A2_MINHASH_BINDING.receipt_sha256,
            "runtime_compatibility_sha256": process.compatibility_identity_sha256,
            "shard_target_bytes": target,
            "zstd_binding_sha256": A2_ZSTD_CODEC_BINDING.receipt_sha256,
        },
    )
    root_identity = hashlib.sha256(
        str(Path(output_root).resolve()).encode()
    ).hexdigest()
    run_id = f"fixture-{root_identity[:12]}-{os.getpid()}"
    manifest = CorpusContentManifestV3(
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        host_name=socket.gethostname(),
        process_id=os.getpid(),
        local_output_root=str(Path(output_root).resolve()),
        source_asset_manifest_sha256=source_manifest_sha256,
        language_manifest_sha256=language_manifest_sha256,
        dedup_manifest_sha256=dedup_ledger_sha256,
        selection_manifest_sha256=selection_manifest_sha256,
        algorithm_manifest_sha256=algorithm_manifest_sha256,
        shards=result.shards,
    )
    return {
        "schema": "weft1_corpus_pa_fixture_replay_v3",
        "authoritative_gate_receipts": [],
        "network_disabled": True,
        "run_id": run_id,
        "process_attestation": asdict(process),
        "process_compatibility_identity_sha256": (
            process.compatibility_identity_sha256
        ),
        "input_identity_sha256": input_identity_sha256,
        "dedup_binding_identity_sha256": A2_MINHASH_BINDING.receipt_sha256,
        "dedup_decision_ledger_identity_sha256": dedup_ledger_sha256,
        "dedup_exact_match_rate": {"numerator": 0, "denominator": 1},
        "dedup_near_match_rate": {"numerator": 0, "denominator": 1},
        "dedup_dropped_bytes": 0,
        "dedup_topup_bytes": 0,
        "minhash_recall_audit": None,
        "content_manifest": asdict(manifest),
        "content_identity_sha256": manifest.content_identity_sha256,
        "shard_identity_sha256s": tuple(
            shard.content_identity_sha256 for shard in result.shards
        ),
        "invalid_utf8_by_source": result.invalid_utf8_by_source,
        "oversized_singleton_count": result.oversized_singleton_count,
        "typed_replay_ready": False,
        "missing_typed_evidence": ["minhash_recall_audit"],
    }


def _fraction_from_mapping(value: object, name: str) -> Fraction:
    if not isinstance(value, Mapping):
        raise CorpusProductionError(f"{name} must be an exact fraction object")
    try:
        return Fraction(value["numerator"], value["denominator"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise CorpusProductionError(f"{name} is malformed") from error


def typed_replay_receipt_from_mapping(
    replay: Mapping[str, object],
    *,
    minhash_recall_audit: MinHashRecallAuditV3,
) -> ReplayRunReceiptV3:
    """Attach real recall evidence and construct, but do not validate, a replay."""

    if replay.get("schema") != "weft1_corpus_pa_fixture_replay_v3":
        raise CorpusProductionError("replay mapping schema is not registered")
    if not isinstance(minhash_recall_audit, MinHashRecallAuditV3):
        raise TypeError("typed replay requires a MinHashRecallAuditV3")
    process_value = replay.get("process_attestation")
    manifest_value = replay.get("content_manifest")
    if not isinstance(process_value, Mapping) or not isinstance(
        manifest_value, Mapping
    ):
        raise CorpusProductionError("replay mapping lacks typed receipt fields")
    shard_values = manifest_value.get("shards")
    if not isinstance(shard_values, (tuple, list)):
        raise CorpusProductionError("replay content manifest lacks shards")
    process = ProcessAttestationV3(**process_value)
    manifest_fields = dict(manifest_value)
    manifest_fields["shards"] = tuple(
        JsonlZstdShardIdentityV3(**value) for value in shard_values
    )
    manifest = CorpusContentManifestV3(**manifest_fields)
    return ReplayRunReceiptV3(
        run_id=replay["run_id"],  # type: ignore[arg-type]
        process_attestation=process,
        input_identity_sha256=replay["input_identity_sha256"],  # type: ignore[arg-type]
        dedup_binding_identity_sha256=replay[
            "dedup_binding_identity_sha256"
        ],  # type: ignore[arg-type]
        dedup_decision_ledger_identity_sha256=replay[
            "dedup_decision_ledger_identity_sha256"
        ],  # type: ignore[arg-type]
        dedup_exact_match_rate=_fraction_from_mapping(
            replay["dedup_exact_match_rate"], "dedup_exact_match_rate"
        ),
        dedup_near_match_rate=_fraction_from_mapping(
            replay["dedup_near_match_rate"], "dedup_near_match_rate"
        ),
        dedup_dropped_bytes=replay["dedup_dropped_bytes"],  # type: ignore[arg-type]
        dedup_topup_bytes=replay["dedup_topup_bytes"],  # type: ignore[arg-type]
        minhash_recall_audit=minhash_recall_audit,
        content_manifest=manifest,
    )


__all__ = [
    "CorpusProductionError",
    "DEFAULT_REQUIREMENTS_LOCK",
    "DEFAULT_REQUIREMENTS_LOCK_SHA256",
    "DEFAULT_SHARD_TARGET_BYTES",
    "DEFAULT_SQLITE_SOURCE_ID",
    "FastTextLanguageIdAdapterV3",
    "HashLockedDistributionV3",
    "RawDocumentV3",
    "RuntimeAttestationV3",
    "RuntimeExpectationV3",
    "INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA_V3",
    "RUNTIME_LINKAGE_SCHEMA_V3",
    "ShardWriteResultV3",
    "SourceAssetExpectationV3",
    "SourceCacheVerificationV3",
    "VerifiedSourceAssetV3",
    "attest_runtime_v3",
    "classify_fasttext_backend_v3",
    "installed_distribution_inventory_v3",
    "parse_hash_locked_requirements_v3",
    "runtime_linkage_inventory_v3",
    "run_fixture_replay",
    "sha256_file",
    "typed_replay_receipt_from_mapping",
    "validate_installed_distribution_inventory_v3",
    "validate_runtime_linkage_inventory_v3",
    "verify_source_cache_assets",
    "write_jsonl_zstd_shards_v3",
]
