"""Fail-closed, offline-first CLI for WEFT-1 corpus Phase A.

This entrypoint deliberately does not download data, request an accelerator, or
mint a corpus gate.  It provides contract verification, environment
attestation, verified-cache inspection, nonauthoritative fixture replay, and a
``full-pa`` command that runs the one fixed offline production worker twice
under Linux network namespaces to produce parent-observed D1/D2 evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import socket
import sqlite3
import stat
import sys
import tempfile
import unicodedata
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.weft1_corpus_replay_a2 import (  # noqa: E402 - direct script support
    CHILD_RECEIPT_FILENAME,
    CHILD_RECEIPT_SCHEMA_V3,
    NETWORK_PROBE_RESULT,
    ParentReplayError,
    verify_parent_replays_v3,
    verify_production_materialization_replays_v3,
)
from training.weft1_strict_io import (  # noqa: E402 - direct script support
    StrictJsonError,
    StrictPathError,
    assert_no_symlink_ancestors,
    load_canonical_json_object,
)


EXPECTED_A2_SHA256 = (
    "f7a2655b30f6c699035ec4ffdccee8c03068eeab8da94894be8e5818f955ce02"
)
EXPECTED_BINDINGS_SHA256 = (
    "ee10e69a3ccd55f7960949f4c318daa4db1197c779f5e88fb67cec82ab7f263b"
)
EXPECTED_BINDINGS_SCHEMA = "weft1_corpus_gtok_a2_bindings_v3"
EXPECTED_CACHE_SCHEMA = "weft1_local_source_cache_manifest_v3"
EXPECTED_COLAB_WORKSPACE_LABEL = "Pharma Initiatives"
EXPECTED_COLAB_SUBSCRIPTION_LABEL = "Pro+"
EXPECTED_COLAB_SURFACE_LABEL = "in-app browser"

DEFAULT_BINDINGS = (
    ROOT / "training" / "weft1_corpus_gtok_a2_bindings_20260828.json"
)
DEFAULT_DEPENDENCY_LOCK = (
    ROOT / "training" / "weft1_corpus_gtok_a2_requirements.lock"
)
DEFAULT_REQUIREMENTS = (
    ROOT / "training" / "weft1_corpus_gtok_a2_requirements.txt"
)
DEFAULT_ROUTE_MANIFEST = (
    ROOT / "training" / "weft1_gtok_source_routes_20260828.json"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)$")
_RUN_LOCAL_KEYS = frozenset(
    {
        "completed_at",
        "created_at_utc",
        "host",
        "hostname",
        "host_name",
        "local_output_root",
        "output_root",
        "output_root_label",
        "pid",
        "process_id",
        "run_id",
        "started_at",
    }
)
class PreflightError(RuntimeError):
    """A fail-closed preflight or replay refusal."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the one JSON encoding used by CLI receipts and fixture files."""

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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _governed_lexical_path(path: Path, name: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    try:
        return assert_no_symlink_ancestors(path)
    except StrictPathError as error:
        raise PreflightError(f"{name} may not traverse symlinks/reparse points") from error


def _snapshot_regular_file(source: Path, destination: Path, *, name: str) -> tuple[int, str]:
    """Copy one governed file once, then return the snapshot's byte identity."""

    lexical_source = _governed_lexical_path(source, name)
    lexical_destination = _governed_lexical_path(destination, f"{name} snapshot")
    lexical_destination.parent.mkdir(parents=True, exist_ok=True)
    _governed_lexical_path(lexical_destination, f"{name} snapshot")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with lexical_source.open("rb") as opened:
            if not stat.S_ISREG(os.fstat(opened.fileno()).st_mode):
                raise PreflightError(f"{name} must be a regular file")
            with lexical_destination.open("xb") as snapshot:
                for chunk in iter(lambda: opened.read(1024 * 1024), b""):
                    digest.update(chunk)
                    byte_count += len(chunk)
                    snapshot.write(chunk)
                snapshot.flush()
                os.fsync(snapshot.fileno())
    except PreflightError:
        raise
    except OSError as error:
        raise PreflightError(f"cannot snapshot {name}: {error}") from error
    return byte_count, digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PreflightError(f"{name} must be a lowercase SHA-256")
    return value


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightError(f"{name} must be a JSON object")
    return value


def _json_object_no_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise PreflightError(f"JSON object repeats key: {key}")
        value[key] = item
    return value


def _read_json_object(path: Path, *, require_canonical: bool = False) -> Mapping[str, Any]:
    lexical_path = _governed_lexical_path(path, "JSON input")
    try:
        raw = lexical_path.read_bytes()
    except OSError as error:
        raise PreflightError(f"cannot read {path}: {error}") from error
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                PreflightError(f"JSON uses a non-finite constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"{path} is not strict UTF-8 JSON: {error}") from error
    payload = _require_mapping(value, str(path))
    if require_canonical and raw != canonical_json_bytes(payload):
        raise PreflightError(f"{path} is not canonical LF-terminated JSON")
    return payload


def _read_governed_ledger(path: Path) -> Mapping[str, Any]:
    try:
        return load_canonical_json_object(path)
    except (OSError, TypeError, StrictJsonError) as error:
        raise PreflightError(f"governed JSON ledger validation failed: {error}") from error


def _receipt(command: str, evidence: Mapping[str, Any], *, status: str = "PASS") -> dict[str, Any]:
    core = {
        "authority_sha256": EXPECTED_A2_SHA256,
        "command": command,
        "evidence": dict(evidence),
        "schema": "weft1_corpus_pa_cli_receipt_v3",
        "status": status,
    }
    return {
        "receipt": core,
        "receipt_payload_sha256": sha256_bytes(canonical_json_bytes(core)),
        "schema": "weft1_corpus_pa_cli_receipt_envelope_v3",
    }


def _emit(value: Mapping[str, Any], output_path: Path | None = None) -> None:
    raw = canonical_json_bytes(value)
    if output_path is not None:
        lexical_output = _governed_lexical_path(output_path, "receipt output")
        if lexical_output.exists():
            raise PreflightError(f"receipt output already exists: {lexical_output}")
        lexical_output.parent.mkdir(parents=True, exist_ok=True)
        lexical_output = _governed_lexical_path(
            lexical_output, "receipt output"
        )
        try:
            with lexical_output.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise PreflightError(
                f"cannot write receipt {lexical_output}: {error}"
            ) from error
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


def _load_route_manifest_identity(path: Path) -> tuple[str, Mapping[str, Any]]:
    try:
        from training.weft1_gtok_a1_contract import load_source_route_manifest

        typed = load_source_route_manifest(path)
    except (ImportError, OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise PreflightError(f"source-route manifest validation failed: {error}") from error
    return typed.manifest_sha256, _read_governed_ledger(path)


def verify_contracts(
    *,
    authority_path: Path,
    bindings_path: Path = DEFAULT_BINDINGS,
    dependency_lock_path: Path = DEFAULT_DEPENDENCY_LOCK,
    route_manifest_path: Path = DEFAULT_ROUTE_MANIFEST,
    expected_bindings_sha256: str = EXPECTED_BINDINGS_SHA256,
) -> dict[str, Any]:
    """Verify every checked-in hash edge plus the supplied A2 authority blob."""

    _require_sha256(expected_bindings_sha256, "expected_bindings_sha256")
    with tempfile.TemporaryDirectory(prefix="weft1-contract-snapshot-") as raw_root:
        snapshot_root = Path(raw_root)
        authority_snapshot = snapshot_root / "authority.md"
        bindings_snapshot = snapshot_root / "bindings.json"
        lock_snapshot = snapshot_root / "requirements.lock"
        route_snapshot = snapshot_root / "source-routes.json"
        authority_bytes, authority_sha256 = _snapshot_regular_file(
            authority_path, authority_snapshot, name="A2 authority"
        )
        bindings_bytes, bindings_sha256 = _snapshot_regular_file(
            bindings_path, bindings_snapshot, name="A2 bindings"
        )
        dependency_lock_bytes, dependency_lock_sha256 = _snapshot_regular_file(
            dependency_lock_path, lock_snapshot, name="dependency lock"
        )
        _snapshot_regular_file(
            route_manifest_path, route_snapshot, name="source-route manifest"
        )

        if authority_sha256 != EXPECTED_A2_SHA256:
            raise PreflightError(
                "A2 authority byte hash differs from the ratified SHA-256"
            )
        if bindings_sha256 != expected_bindings_sha256:
            raise PreflightError(
                "A2 bindings byte hash differs from the CLI-pinned SHA-256"
            )
        bindings = _read_governed_ledger(bindings_snapshot)
        if bindings.get("schema") != EXPECTED_BINDINGS_SCHEMA:
            raise PreflightError("A2 bindings schema drifted")
        if bindings.get("authority_sha256") != EXPECTED_A2_SHA256:
            raise PreflightError(
                "A2 bindings are not attached to the verified authority"
            )

        runtime = _require_mapping(bindings.get("runtime"), "bindings.runtime")
        bound_lock_sha256 = _require_sha256(
            runtime.get("requirements_lock_sha256"),
            "bindings.runtime.requirements_lock_sha256",
        )
        if dependency_lock_sha256 != bound_lock_sha256:
            raise PreflightError("dependency lock hash differs from the A2 binding")

        route_identity, _ = _load_route_manifest_identity(route_snapshot)
        bound_route_identity = _require_sha256(
            bindings.get("a1_route_manifest_receipt_sha256"),
            "bindings.a1_route_manifest_receipt_sha256",
        )
        if route_identity != bound_route_identity:
            raise PreflightError(
                "source-route receipt identity differs from the A2 binding"
            )

    return {
        "authority_bytes": authority_bytes,
        "authority_sha256": authority_sha256,
        "bindings_bytes": bindings_bytes,
        "bindings_sha256": bindings_sha256,
        "dependency_lock_bytes": dependency_lock_bytes,
        "dependency_lock_sha256": dependency_lock_sha256,
        "route_manifest_receipt_sha256": route_identity,
        "verified": True,
    }


def _requirements_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _PACKAGE_PIN.fullmatch(stripped)
        if match is None:
            raise PreflightError(f"requirements input has an unbound row: {stripped!r}")
        normalized = match.group(1).lower().replace("_", "-")
        if normalized in pins:
            raise PreflightError(f"duplicate requirements pin: {normalized}")
        pins[normalized] = match.group(2)
    if not pins:
        raise PreflightError("requirements input contains no exact pins")
    return dict(sorted(pins.items()))


def environment_receipt(*, require_match: bool) -> dict[str, Any]:
    """Observe only public runtime facts; never enumerate environment variables."""

    bindings = _read_governed_ledger(DEFAULT_BINDINGS)
    runtime = _require_mapping(bindings.get("runtime"), "bindings.runtime")
    pins = _requirements_pins(DEFAULT_REQUIREMENTS)
    observed_packages: dict[str, dict[str, object]] = {}
    mismatches: list[str] = []
    for package_name, expected_version in pins.items():
        try:
            observed_version: str | None = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            observed_version = None
        matches = observed_version == expected_version
        if not matches:
            mismatches.append(f"package:{package_name}")
        observed_packages[package_name] = {
            "expected": expected_version,
            "matches": matches,
            "observed": observed_version,
        }

    executable = Path(sys.executable).resolve()
    observations = {
        "dependency_lock_sha256": sha256_file(DEFAULT_DEPENDENCY_LOCK),
        "executable_sha256": sha256_file(executable),
        "implementation": platform.python_implementation(),
        "packages": observed_packages,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "sys_byteorder": sys.byteorder,
        "unicode_database": unicodedata.unidata_version,
    }
    expected_scalars = {
        "dependency_lock_sha256": runtime.get("requirements_lock_sha256"),
        "python": runtime.get("execution_python"),
        "sqlite": runtime.get("sqlite"),
        "unicode_database": runtime.get("unicode_database"),
    }
    for name, expected in expected_scalars.items():
        if observations[name] != expected:
            mismatches.append(name)
    observations["expected"] = expected_scalars
    observations["matches_bound_environment"] = not mismatches
    observations["mismatches"] = sorted(mismatches)
    observations["authoritative"] = bool(require_match and not mismatches)
    observations["credentials_inspected"] = False
    observations["environment_variables_enumerated"] = False
    if require_match and mismatches:
        raise PreflightError(
            "runtime differs from A2 bindings: " + ", ".join(sorted(mismatches))
        )
    return observations


def _canonical_cache_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PreflightError("cache_relative_path must be a nonempty POSIX path")
    if "\\" in value:
        raise PreflightError("cache_relative_path may not contain backslashes")
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in {"", ".", ".."}
        or part.endswith((".", " "))
        or ":" in part
        for part in path.parts
    ):
        raise PreflightError("cache_relative_path must be canonical and relative")
    if path.as_posix() != value:
        raise PreflightError("cache_relative_path is not canonical POSIX syntax")
    return value


def _assert_no_symlink_path(cache_root: Path, candidate: Path) -> None:
    try:
        assert_no_symlink_ancestors(cache_root)
        assert_no_symlink_ancestors(candidate)
    except StrictPathError as error:
        raise PreflightError("source-cache assets may not traverse symlinks") from error
    root = cache_root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise PreflightError("source-cache asset resolves outside cache root")


def verify_source_cache_manifest(
    *,
    manifest_path: Path,
    cache_root: Path,
    route_manifest_path: Path = DEFAULT_ROUTE_MANIFEST,
) -> dict[str, Any]:
    """Rehash a fully local, authority-bound source cache without networking."""

    try:
        assert_no_symlink_ancestors(cache_root)
    except StrictPathError as error:
        raise PreflightError("source-cache root may not traverse symlinks") from error
    if not cache_root.is_dir():
        raise PreflightError("source-cache root must be an existing directory")
    try:
        source_module = import_module("training.weft1_corpus_sources_a2")
    except ModuleNotFoundError as error:
        if error.name != "training.weft1_corpus_sources_a2":
            raise
    else:
        helper = getattr(source_module, "verify_source_cache_manifest", None)
        if helper is None:
            raise PreflightError(
                "training.weft1_corpus_sources_a2 has no cache-verification hook"
            )
        try:
            result = helper(manifest_path, cache_root, route_manifest_path)
        except (OSError, TypeError, ValueError) as error:
            raise PreflightError(f"source-cache verification failed: {error}") from error
        if not isinstance(result, Mapping):
            raise PreflightError("source-cache verifier must return a mapping")
        return {
            **dict(result),
            "network_used": False,
            "verified": True,
        }

    manifest = _read_json_object(manifest_path)
    if manifest.get("schema") != EXPECTED_CACHE_SCHEMA:
        raise PreflightError("source-cache manifest schema drifted")
    route_identity, route_payload = _load_route_manifest_identity(route_manifest_path)
    if manifest.get("source_route_manifest_sha256") != route_identity:
        raise PreflightError("source-cache manifest is bound to a different route ledger")

    route_rows_raw = route_payload.get("routes")
    if not isinstance(route_rows_raw, list):
        raise PreflightError("source-route manifest routes must be an array")
    routes = {
        row["source_family"]: row
        for row in route_rows_raw
        if isinstance(row, Mapping) and isinstance(row.get("source_family"), str)
    }
    assets_raw = manifest.get("assets")
    if not isinstance(assets_raw, list) or not assets_raw:
        raise PreflightError("source-cache manifest requires at least one asset")

    seen_paths: set[str] = set()
    verified_rows: list[dict[str, Any]] = []
    total_bytes = 0
    for index, raw_asset in enumerate(assets_raw):
        asset = _require_mapping(raw_asset, f"assets[{index}]")
        source_family = asset.get("source_family")
        if not isinstance(source_family, str) or source_family not in routes:
            raise PreflightError(f"assets[{index}] uses an unknown source family")
        route = routes[source_family]
        for key in ("repository", "config", "revision", "split"):
            if asset.get(key) != route.get(key):
                raise PreflightError(f"assets[{index}].{key} differs from its route")
        locator = asset.get("asset_locator")
        if not isinstance(locator, str) or not locator:
            raise PreflightError(f"assets[{index}].asset_locator must be nonempty")
        relative = _canonical_cache_relative_path(asset.get("relative_path"))
        if relative in seen_paths:
            raise PreflightError("source-cache manifest repeats a cache path")
        seen_paths.add(relative)
        expected_bytes = asset.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 1:
            raise PreflightError(f"assets[{index}].bytes must be positive")
        expected_sha256 = _require_sha256(
            asset.get("sha256"), f"assets[{index}].sha256"
        )

        local_path = cache_root.joinpath(*PurePosixPath(relative).parts)
        if not local_path.is_file():
            raise PreflightError(f"source-cache asset is absent: {relative}")
        _assert_no_symlink_path(cache_root, local_path)
        actual_bytes = local_path.stat().st_size
        actual_sha256 = sha256_file(local_path)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise PreflightError(f"source-cache asset failed byte verification: {relative}")
        total_bytes += actual_bytes
        verified_rows.append(
            {
                "asset_locator": locator,
                "bytes": actual_bytes,
                "config": asset.get("config"),
                "repository": asset.get("repository"),
                "revision": asset.get("revision"),
                "sha256": actual_sha256,
                "source_family": source_family,
                "split": asset.get("split"),
            }
        )

    offline_identity_payload = {
        "assets": verified_rows,
        "schema": "weft1_verified_local_source_cache_identity_v3",
        "source_route_manifest_sha256": route_identity,
    }
    computed_identity = sha256_bytes(canonical_json_bytes(offline_identity_payload))
    claimed_identity = manifest.get("offline_replay_identity_sha256")
    if claimed_identity is not None and claimed_identity != computed_identity:
        raise PreflightError("source-cache offline replay identity differs")
    return {
        "asset_count": len(verified_rows),
        "manifest_bytes_sha256": sha256_file(manifest_path),
        "network_used": False,
        "offline_replay_identity_sha256": computed_identity,
        "source_route_manifest_sha256": route_identity,
        "total_asset_bytes": total_bytes,
        "verified": True,
    }


def _install_python_network_guard() -> None:
    """Refuse Python socket use before lazily importing the production I/O layer."""

    def refused(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("WEFT-1 fixture replay forbids network access")

    socket.create_connection = refused  # type: ignore[assignment]
    socket.getaddrinfo = refused  # type: ignore[assignment]
    socket.socket.connect = refused  # type: ignore[assignment]
    socket.socket.connect_ex = refused  # type: ignore[assignment]
    socket.socket.sendto = refused  # type: ignore[assignment]
    if hasattr(socket.socket, "sendmsg"):
        socket.socket.sendmsg = refused  # type: ignore[attr-defined,assignment]


def _builtin_fixture_replay(fixture_path: Path, output_root: Path) -> Mapping[str, Any]:
    fixture = _read_json_object(fixture_path, require_canonical=True)
    artifact = {
        "fixture": fixture,
        "fixture_sha256": sha256_file(fixture_path),
        "schema": "weft1_corpus_pa_builtin_fixture_artifact_v3",
    }
    artifact_bytes = canonical_json_bytes(artifact)
    (output_root / "fixture-artifact.json").write_bytes(artifact_bytes)
    return {
        "artifact_sha256": sha256_bytes(artifact_bytes),
        "backend": "builtin_fixture_only",
        "input_identity_sha256": sha256_file(fixture_path),
        "schema": "weft1_corpus_pa_builtin_fixture_result_v3",
    }


def _lazy_fixture_replay(fixture_path: Path, output_root: Path) -> Mapping[str, Any]:
    try:
        module = import_module("training.weft1_corpus_pa")
    except ModuleNotFoundError as error:
        if error.name != "training.weft1_corpus_pa":
            raise
        return _builtin_fixture_replay(fixture_path, output_root)
    function = getattr(module, "run_fixture_replay", None)
    if function is None:
        raise PreflightError(
            "training.weft1_corpus_pa exists but has no run_fixture_replay hook"
        )
    try:
        result = function(fixture_path, output_root, network_disabled=True)
    except Exception as error:
        raise PreflightError(f"production fixture replay failed: {error}") from error
    if not isinstance(result, Mapping):
        raise PreflightError("run_fixture_replay must return a mapping")
    return result


def _tree_manifest(
    root: Path, *, exclude_relative_paths: frozenset[str] = frozenset()
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise PreflightError("fixture replay output may not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PreflightError("fixture replay output contains a non-regular entry")
        relative = path.relative_to(root).as_posix()
        if relative in exclude_relative_paths:
            continue
        rows.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise PreflightError("fixture replay produced no files")
    return rows, sha256_bytes(canonical_json_bytes(rows))


def _content_projection(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _content_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _RUN_LOCAL_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_content_projection(item) for item in value]
    return value


def _probe_blocking_network_guard() -> str:
    """Prove that the child-visible Python socket guard blocks a real attempt."""

    probe = socket.socket()
    try:
        probe.connect(("127.0.0.1", 9))
    except RuntimeError:
        return NETWORK_PROBE_RESULT
    except OSError as error:
        raise PreflightError(
            "fixture replay network probe reached the operating-system socket"
        ) from error
    finally:
        probe.close()
    raise PreflightError("fixture replay network probe was not blocked")


def _fixture_worker(
    *, fixture_path: Path, output_root: Path, builtin_only: bool = False
) -> dict[str, Any]:
    if os.environ.get("WEFT1_NETWORK_DISABLED") != "1":
        raise PreflightError("fixture worker requires WEFT1_NETWORK_DISABLED=1")
    if output_root.exists():
        raise PreflightError("fixture worker output root must be fresh")
    output_root.mkdir(parents=True)
    _install_python_network_guard()
    network_probe = _probe_blocking_network_guard()
    result = (
        _builtin_fixture_replay(fixture_path, output_root)
        if builtin_only
        else _lazy_fixture_replay(fixture_path, output_root)
    )
    rows, tree_identity = _tree_manifest(output_root)
    projection = _content_projection(result)
    run_id = os.environ.get("WEFT1_REPLAY_RUN_ID")
    if not isinstance(run_id, str) or not run_id:
        raise PreflightError("fixture worker lacks its parent-assigned replay run ID")
    input_identity = _require_sha256(
        os.environ.get("WEFT1_REPLAY_INPUT_IDENTITY_SHA256"),
        "parent-assigned replay input identity",
    )
    worker_compatibility = _require_sha256(
        os.environ.get("WEFT1_REPLAY_WORKER_COMPATIBILITY_SHA256"),
        "parent-assigned worker compatibility identity",
    )
    network_guard_sha256 = _require_sha256(
        os.environ.get("WEFT1_NETWORK_GUARD_SHA256"),
        "parent-injected network guard SHA-256",
    )
    resolved_root = output_root.resolve(strict=True)
    expected_receipt_path = resolved_root / CHILD_RECEIPT_FILENAME
    raw_receipt_path = os.environ.get("WEFT1_REPLAY_RECEIPT_PATH")
    if not isinstance(raw_receipt_path, str) or (
        Path(raw_receipt_path).resolve(strict=False) != expected_receipt_path
    ):
        raise PreflightError(
            "fixture worker receipt path differs from parent assignment"
        )
    inventory_rows = [dict(row, role="content") for row in rows]
    receipt = {
        "content_metadata": {
            "legacy_output_tree_sha256": tree_identity,
            "result": projection,
            "result_content_sha256": sha256_bytes(
                canonical_json_bytes(projection)
            ),
        },
        "dedup_evidence_complete": False,
        "dedup_metadata": None,
        "files": inventory_rows,
        "input_identity_sha256": input_identity,
        # The fixture worker attests that its Python network APIs were disabled
        # and probed.  The outer receipt separately avoids presenting that
        # guard as authoritative OS-level isolation.
        "network_disabled": True,
        "network_guard_active": (
            os.environ.get("WEFT1_NETWORK_GUARD_ACTIVE") == "1"
        ),
        "network_guard_sha256": network_guard_sha256,
        "network_probe": network_probe,
        "output_root": str(resolved_root),
        "process_id": os.getpid(),
        "run_id": run_id,
        "schema": CHILD_RECEIPT_SCHEMA_V3,
        "worker_compatibility_sha256": worker_compatibility,
    }
    receipt_bytes = canonical_json_bytes(receipt)
    try:
        with expected_receipt_path.open("xb") as handle:
            handle.write(receipt_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise PreflightError(
            "fixture worker could not persist its child receipt"
        ) from error
    return receipt


def run_two_fixture_replays(
    *,
    fixture_path: Path,
    output_parent: Path,
    use_builtin_test_backend: bool = False,
) -> dict[str, Any]:
    """Run the fixture through the parent-observed A2 replay verifier."""

    _governed_lexical_path(fixture_path, "fixture input")
    lexical_output_parent = _governed_lexical_path(
        output_parent, "fixture replay parent"
    )
    if lexical_output_parent.exists():
        raise PreflightError("fixture replay parent must not already exist")
    lexical_output_parent.mkdir(parents=True)
    lexical_output_parent = _governed_lexical_path(
        lexical_output_parent, "fixture replay parent"
    )
    fixture_snapshot = lexical_output_parent / "fixture-input.json"
    _fixture_bytes, fixture_sha256 = _snapshot_regular_file(
        fixture_path, fixture_snapshot, name="fixture input"
    )
    _read_json_object(fixture_snapshot, require_canonical=True)
    roots = (
        lexical_output_parent / "replay-a",
        lexical_output_parent / "replay-b",
    )
    worker_arguments = [
        str(Path(__file__).resolve()),
        "_fixture-worker",
        "--fixture",
        str(fixture_snapshot),
    ]
    if use_builtin_test_backend:
        worker_arguments.append("--builtin-only")
    try:
        parent_receipt = verify_parent_replays_v3(
            python_executable=Path(sys.executable),
            worker_arguments=worker_arguments,
            first_output_root=roots[0],
            second_output_root=roots[1],
            input_files={"fixture": fixture_snapshot},
            compatibility_files={
                "a2_bindings": DEFAULT_BINDINGS,
                "dependency_lock": DEFAULT_DEPENDENCY_LOCK,
                "parent_replay_verifier": ROOT
                / "training"
                / "weft1_corpus_replay_a2.py",
                "production_io": ROOT / "training" / "weft1_corpus_pa.py",
                "source_routes": DEFAULT_ROUTE_MANIFEST,
                "worker": Path(__file__).resolve(),
            },
            worker_cwd=ROOT,
        )
    except ParentReplayError as error:
        raise PreflightError(f"parent replay verification failed: {error}") from error

    rows, tree_identity = _tree_manifest(
        roots[0], exclude_relative_paths=frozenset({CHILD_RECEIPT_FILENAME})
    )

    return {
        "authoritative": parent_receipt.authoritative,
        "d1_file_replay_verified": parent_receipt.d1_file_replay_verified,
        "d2_dedup_replay_verified": parent_receipt.d2_dedup_replay_verified,
        "distinct_output_roots": True,
        "distinct_process_ids": True,
        "fixture_sha256": fixture_sha256,
        "network_disabled": False,
        "network_isolation_kind": "python_socket_guard_only",
        "parent_replay_receipt": asdict(parent_receipt),
        "parent_replay_receipt_sha256": parent_receipt.receipt_sha256,
        "production_io_hook_used": not use_builtin_test_backend,
        "output_tree": rows,
        "output_tree_sha256": tree_identity,
        "replay_result_content_sha256": (
            parent_receipt.content_projection_sha256
        ),
        "status": parent_receipt.status,
        "worker_process_ids": [
            parent_receipt.first_process_id,
            parent_receipt.second_process_id,
        ],
    }


def colab_preflight(
    *,
    workspace_label: str,
    subscription_label: str,
    surface_label: str,
    runtime_label: str,
) -> dict[str, Any]:
    """Validate externally supplied labels without touching accounts or secrets."""

    expected = {
        "subscription_label": EXPECTED_COLAB_SUBSCRIPTION_LABEL,
        "surface_label": EXPECTED_COLAB_SURFACE_LABEL,
        "workspace_label": EXPECTED_COLAB_WORKSPACE_LABEL,
    }
    observed = {
        "subscription_label": subscription_label,
        "surface_label": surface_label,
        "workspace_label": workspace_label,
    }
    if observed != expected:
        raise PreflightError("externally supplied Pharma Colab labels do not match")
    if not isinstance(runtime_label, str) or not runtime_label.strip():
        raise PreflightError("runtime_label must be supplied externally and be nonempty")
    return {
        "accelerator_inspected": False,
        "authoritative": False,
        "credentials_read": False,
        "expected_labels": expected,
        "gpu_requested": False,
        "network_used": False,
        "runtime_label": runtime_label,
        "runtime_label_source": "external_cli_argument",
        "verified": False,
    }


def full_pa_guard(
    *,
    source_cache: Path | None,
    source_cache_manifest: Path | None,
    route_manifest: Path | None,
    output_path: Path | None,
) -> dict[str, Any]:
    """Read-only legacy preflight helper; the ``full-pa`` CLI now executes."""

    supplied = {
        "output_path": output_path,
        "route_manifest": route_manifest,
        "source_cache": source_cache,
        "source_cache_manifest": source_cache_manifest,
    }
    missing = sorted(name for name, value in supplied.items() if value is None)
    if missing:
        raise PreflightError(
            "full P-A refused; required inputs missing: " + ", ".join(missing)
        )
    assert source_cache is not None
    assert source_cache_manifest is not None
    assert route_manifest is not None
    assert output_path is not None
    if output_path.exists():
        raise PreflightError("full P-A output path must be fresh")
    cache_evidence = verify_source_cache_manifest(
        manifest_path=source_cache_manifest,
        cache_root=source_cache,
        route_manifest_path=route_manifest,
    )
    return {
        "all_required_paths_supplied": True,
        "execution_enabled": False,
        "execution_status": "NOT_EXECUTED_INITIAL_READ_ONLY_REVISION",
        "gpu_requested": False,
        "network_used": False,
        "source_cache": cache_evidence,
    }


def run_full_pa_replays(
    *,
    authority_path: Path,
    enumeration_receipt_path: Path,
    cache_download_receipt_path: Path,
    source_cache_manifest_path: Path,
    source_cache: Path,
    fasttext_model_path: Path,
    output_parent: Path,
) -> dict[str, Any]:
    """Execute the governed offline P-A worker twice and reduce D1/D2."""

    paths = (
        authority_path,
        enumeration_receipt_path,
        cache_download_receipt_path,
        source_cache_manifest_path,
        source_cache,
        fasttext_model_path,
        output_parent,
    )
    if any(not isinstance(path, Path) for path in paths):
        raise TypeError("full P-A paths must be pathlib.Path values")
    governed_output_parent = _governed_lexical_path(
        output_parent, "full P-A output parent"
    )
    if governed_output_parent.exists():
        raise PreflightError("full P-A output parent must be fresh")
    governed_output_parent.mkdir(parents=True)
    _governed_lexical_path(governed_output_parent, "full P-A output parent")
    try:
        parent = verify_production_materialization_replays_v3(
            python_executable=Path(sys.executable),
            authority_path=authority_path,
            enumeration_receipt_path=enumeration_receipt_path,
            cache_download_receipt_path=cache_download_receipt_path,
            source_manifest_path=source_cache_manifest_path,
            cache_root=source_cache,
            fasttext_model_path=fasttext_model_path,
            first_output_root=governed_output_parent / "production-replay-a",
            second_output_root=governed_output_parent / "production-replay-b",
        )
    except ParentReplayError as error:
        raise PreflightError(f"production P-A replay failed: {error}") from error
    if not parent.authoritative or parent.status != "PASS":
        raise PreflightError("production P-A completed without authoritative D1/D2")
    return {
        "authoritative": True,
        "d1_file_replay_verified": parent.d1_file_replay_verified,
        "d2_dedup_replay_verified": parent.d2_dedup_replay_verified,
        "gpu_requested": False,
        "network_isolation_kind": parent.network_isolation_kind,
        "network_used": False,
        "parent_replay_receipt": asdict(parent),
        "parent_replay_receipt_sha256": parent.receipt_sha256,
        "production_profile_verified": parent.production_profile_verified,
        "status": parent.status,
    }


def _add_receipt_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--receipt-out",
        type=Path,
        help="optional fresh path for the same canonical JSON printed to stdout",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    contracts = subparsers.add_parser("verify-contracts")
    contracts.add_argument("--authority", type=Path, required=True)
    contracts.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    contracts.add_argument("--dependency-lock", type=Path, default=DEFAULT_DEPENDENCY_LOCK)
    contracts.add_argument("--route-manifest", type=Path, default=DEFAULT_ROUTE_MANIFEST)
    contracts.add_argument(
        "--bindings-sha256", default=EXPECTED_BINDINGS_SHA256
    )
    _add_receipt_output(contracts)

    environment = subparsers.add_parser("environment-receipt")
    environment.add_argument(
        "--observe-only",
        action="store_true",
        help="report mismatches instead of refusing; forbidden for production execution",
    )
    _add_receipt_output(environment)

    cache = subparsers.add_parser("verify-source-cache")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--cache-root", type=Path, required=True)
    cache.add_argument("--route-manifest", type=Path, default=DEFAULT_ROUTE_MANIFEST)
    _add_receipt_output(cache)

    replay = subparsers.add_parser("replay-fixture")
    replay.add_argument("--fixture", type=Path, required=True)
    replay.add_argument("--output-parent", type=Path, required=True)
    _add_receipt_output(replay)

    colab = subparsers.add_parser("colab-preflight")
    colab.add_argument("--workspace-label", required=True)
    colab.add_argument("--subscription-label", required=True)
    colab.add_argument("--surface-label", required=True)
    colab.add_argument("--runtime-label", required=True)
    _add_receipt_output(colab)

    full = subparsers.add_parser("full-pa")
    full.add_argument("--authority", type=Path)
    full.add_argument("--enumeration-receipt", type=Path)
    full.add_argument("--cache-download-receipt", type=Path)
    full.add_argument("--source-cache", type=Path)
    full.add_argument("--source-cache-manifest", type=Path)
    full.add_argument("--fasttext-model", type=Path)
    full.add_argument("--output-parent", type=Path)
    _add_receipt_output(full)

    worker = subparsers.add_parser("_fixture-worker")
    worker.add_argument("--fixture", type=Path, required=True)
    worker.add_argument("--output-root", type=Path)
    worker.add_argument("--builtin-only", action="store_true", help=argparse.SUPPRESS)
    return parser


def _dispatch(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.command == "verify-contracts":
        evidence = verify_contracts(
            authority_path=args.authority,
            bindings_path=args.bindings,
            dependency_lock_path=args.dependency_lock,
            route_manifest_path=args.route_manifest,
            expected_bindings_sha256=args.bindings_sha256,
        )
        return _receipt(args.command, evidence)
    if args.command == "environment-receipt":
        evidence = environment_receipt(require_match=not args.observe_only)
        return _receipt(
            args.command,
            evidence,
            status="OBSERVED" if args.observe_only else "PASS",
        )
    if args.command == "verify-source-cache":
        evidence = verify_source_cache_manifest(
            manifest_path=args.manifest,
            cache_root=args.cache_root,
            route_manifest_path=args.route_manifest,
        )
        return _receipt(args.command, evidence)
    if args.command == "replay-fixture":
        evidence = run_two_fixture_replays(
            fixture_path=args.fixture,
            output_parent=args.output_parent,
        )
        return _receipt(args.command, evidence, status=evidence["status"])
    if args.command == "colab-preflight":
        evidence = colab_preflight(
            workspace_label=args.workspace_label,
            subscription_label=args.subscription_label,
            surface_label=args.surface_label,
            runtime_label=args.runtime_label,
        )
        return _receipt(args.command, evidence, status="OBSERVED")
    if args.command == "full-pa":
        if isinstance(args.output_parent, Path) and isinstance(args.receipt_out, Path):
            output_parent = _governed_lexical_path(
                args.output_parent, "full P-A output parent"
            ).resolve(strict=False)
            receipt_out = _governed_lexical_path(
                args.receipt_out, "full P-A receipt output"
            ).resolve(strict=False)
            if receipt_out == output_parent or output_parent in receipt_out.parents:
                raise PreflightError(
                    "full P-A receipt output must be outside the governed replay tree"
                )
        evidence = run_full_pa_replays(
            authority_path=args.authority,
            enumeration_receipt_path=args.enumeration_receipt,
            cache_download_receipt_path=args.cache_download_receipt,
            source_cache=args.source_cache,
            source_cache_manifest_path=args.source_cache_manifest,
            fasttext_model_path=args.fasttext_model,
            output_parent=args.output_parent,
        )
        return _receipt(args.command, evidence, status=evidence["status"])
    if args.command == "_fixture-worker":
        assigned_root = os.environ.get("WEFT1_REPLAY_OUTPUT_ROOT")
        if args.output_root is None:
            if not isinstance(assigned_root, str) or not assigned_root:
                raise PreflightError(
                    "fixture worker requires its parent-assigned output root"
                )
            output_root = Path(assigned_root)
        else:
            output_root = args.output_root
            if assigned_root is not None and (
                output_root.resolve(strict=False)
                != Path(assigned_root).resolve(strict=False)
            ):
                raise PreflightError(
                    "fixture worker argument differs from parent-assigned output root"
                )
        return _fixture_worker(
            fixture_path=args.fixture,
            output_root=output_root,
            builtin_only=args.builtin_only,
        )
    raise PreflightError(f"unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
        output_path = getattr(args, "receipt_out", None)
        _emit(result, output_path)
    except (OSError, TypeError, ValueError, PreflightError) as error:
        failure = _receipt(
            args.command,
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "failed_closed": True,
            },
            status="FAIL",
        )
        sys.stderr.buffer.write(canonical_json_bytes(failure))
        sys.stderr.buffer.flush()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
