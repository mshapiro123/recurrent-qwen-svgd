"""Read-only closed-checkpoint observer for the WEFT-1 P-A r3 replay.

This process never writes an artifact and never mints a gate.  A report-only
condition can pass only from rows inside the maximal receipt-bound checkpoint
prefix validated by the production materializer's V3 checkpoint validator.
Open/local ledger tails, partial files, orphan chunks, watcher counters, and a
completed final ledger are never read as acceptance evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if not sys.path or sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))

from training.weft1_corpus_materialize_a2 import (  # noqa: E402
    CorpusMaterializationError,
    _parse_checkpoint_json_object_v3,
    _read_source_parse_child_once_v3,
    _source_parse_checkpoint_root_v3,
    _validate_source_parse_checkpoint_chain_v3,
)
from training import weft1_corpus_materialize_a3 as materializer_v4  # noqa: E402
from training.weft1_corpus_pa import attest_runtime_v3  # noqa: E402
from training.weft1_corpus_replay_a2 import (  # noqa: E402
    CHILD_RECEIPT_FILENAME,
)
from training.weft1_corpus_source_io_a2 import (  # noqa: E402
    STACKEDU_PYTHON_PARSER_BINDING_V3,
    resolve_production_parser_binding_v3,
)
from training.weft1_corpus_sources_a2 import (  # noqa: E402
    VerifiedLocalCacheAssetV3,
)
from training.weft1_strict_io import (  # noqa: E402
    assert_no_symlink_ancestors,
)


TARGET_PRODUCTION_COMMIT = "c19766a106e244fb8d76472c820610f8ca45557d"
TARGET_DEPENDENCY_PATHS = (
    "training/weft1_corpus_materialize_a2.py",
    "training/weft1_corpus_materialize_a3.py",
    "training/weft1_corpus_pa.py",
    "training/weft1_corpus_replay_a2.py",
    "training/weft1_corpus_source_io_a2.py",
    "training/weft1_corpus_sources_a2.py",
    "training/weft1_strict_io.py",
)
MONITOR_COMMIT_PATHS = frozenset(
    {
        ".gitattributes",
        "scripts/monitor_weft1_pa_r3_closed_checkpoints_v1.py",
        "tests/test_monitor_weft1_pa_r3_closed_checkpoints_v1.py",
    }
)
WIKIPEDIA_FAMILY = "wikipedia_wikibooks"
WIKIPEDIA_TARGET_EVENT = 2_951_022
WIKIPEDIA_TARGET_ASSET = 0
WIKIPEDIA_TARGET_RECORD = 2_951_022
WIKIPEDIA_HINT_CHUNK_INDEX = 45
STACKEDU_FAMILY = "stackedu"
STACKEDU_TARGET_ASSET = 4
STACKEDU_TARGET_RECORDS = (0, 1)
STACKEDU_DIRECT_PYTHON_BINDING_SHA256 = (
    "20dfb069e55731704bb5f562f878575658344131d43daeea292ce535fb60e64e"
)
INCOMPLETE_BYTES = b"P-A incomplete\n"
REPORT_SCHEMA = "weft1_pa_r3_closed_checkpoint_monitor_report_v1"
ERROR_SCHEMA = "weft1_pa_r3_closed_checkpoint_monitor_error_v1"
_SHA256_CHARS = frozenset("0123456789abcdef")
_GIT_HEX_CHARS = frozenset("0123456789abcdef")
_CHECKPOINT_RECEIPT = re.compile(r"^chunk-([0-9]{6})\.receipt\.json$")


class CheckpointMonitorError(RuntimeError):
    """The observer cannot establish a safe report-only condition."""


class ChildRunIdentityChangedError(CheckpointMonitorError):
    """The watched path no longer names the child opened by this process."""


class _CanonicalArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CheckpointMonitorError(f"invalid CLI arguments: {message}")


def _canonical_json_bytes(value: object) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise CheckpointMonitorError(f"{name} must be a lowercase SHA-256")
    return value


def _lexical_absolute(path: Path, *, name: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise CheckpointMonitorError(f"{name} must be an explicit absolute path")
    return assert_no_symlink_ancestors(Path(os.path.abspath(os.fspath(path))))


def _file_identity(metadata: os.stat_result) -> tuple[int, int]:
    identity = (int(metadata.st_dev), int(metadata.st_ino))
    if identity[1] <= 0:
        raise CheckpointMonitorError("filesystem does not expose a usable inode identity")
    return identity


class _ChildRunIdentityGuard:
    """Hold the original marker open and reject same-path run replacement."""

    def __init__(self, child_root: Path) -> None:
        self.child_root = assert_no_symlink_ancestors(child_root)
        self.marker_path = self.child_root / "_INCOMPLETE"
        self._directory_fd = -1
        self._marker_fd = -1
        try:
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                self._directory_fd = os.open(self.child_root, directory_flags)
                directory_metadata = os.fstat(self._directory_fd)
            except OSError:
                # Windows cannot normally open a directory with os.open.  Its
                # open marker handle prevents deletion; on POSIX the directory
                # descriptor remains open as an additional anti-reuse anchor.
                self._directory_fd = -1
                directory_metadata = os.stat(
                    self.child_root,
                    follow_symlinks=False,
                )
            if not stat.S_ISDIR(directory_metadata.st_mode):
                raise CheckpointMonitorError("child output root is not a directory")
            self._directory_identity = _file_identity(directory_metadata)

            marker_flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            self._marker_fd = os.open(self.marker_path, marker_flags)
            marker_metadata = os.fstat(self._marker_fd)
            if not stat.S_ISREG(marker_metadata.st_mode):
                raise CheckpointMonitorError("child _INCOMPLETE is not regular")
            self._marker_identity = _file_identity(marker_metadata)
            marker_bytes = self._read_marker_handle()
            if marker_bytes != INCOMPLETE_BYTES:
                raise CheckpointMonitorError("child _INCOMPLETE marker bytes drifted")
            self.identity = {
                "child_directory_device": self._directory_identity[0],
                "child_directory_inode": self._directory_identity[1],
                "incomplete_marker_device": self._marker_identity[0],
                "incomplete_marker_inode": self._marker_identity[1],
                "incomplete_marker_sha256": _sha256_bytes(marker_bytes),
            }
            self.identity_sha256 = _sha256_bytes(_canonical_json_bytes(self.identity))
            self.assert_current()
        except BaseException:
            self.close()
            raise

    def _read_marker_handle(self) -> bytes:
        os.lseek(self._marker_fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            block = os.read(self._marker_fd, 4096)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)

    def assert_current(self) -> None:
        try:
            before_directory = os.stat(self.child_root, follow_symlinks=False)
            before_marker = os.stat(self.marker_path, follow_symlinks=False)
            marker_bytes = _read_regular_once(self.marker_path)
            after_directory = os.stat(self.child_root, follow_symlinks=False)
            after_marker = os.stat(self.marker_path, follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError, OSError) as error:
            raise ChildRunIdentityChangedError(
                "watched child run path disappeared or became unreadable"
            ) from error
        identities = (
            _file_identity(before_directory),
            _file_identity(after_directory),
            _file_identity(before_marker),
            _file_identity(after_marker),
        )
        if (
            identities[0] != self._directory_identity
            or identities[1] != self._directory_identity
            or identities[2] != self._marker_identity
            or identities[3] != self._marker_identity
            or marker_bytes != INCOMPLETE_BYTES
            or self._read_marker_handle() != INCOMPLETE_BYTES
        ):
            raise ChildRunIdentityChangedError(
                "watched child run identity changed; cross-run combination refused"
            )

    def close(self) -> None:
        if self._marker_fd >= 0:
            os.close(self._marker_fd)
            self._marker_fd = -1
        if self._directory_fd >= 0:
            os.close(self._directory_fd)
            self._directory_fd = -1

    def __enter__(self) -> _ChildRunIdentityGuard:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def _read_regular_once(path: Path) -> bytes:
    lexical = assert_no_symlink_ancestors(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as error:
        raise CheckpointMonitorError(f"cannot open governed file: {lexical}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CheckpointMonitorError(f"governed child is not regular: {lexical}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise CheckpointMonitorError(
            "repository deployment check failed: " + " ".join(arguments)
        )
    return completed.stdout


def _require_git_commit(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in _GIT_HEX_CHARS for character in value)
    ):
        raise CheckpointMonitorError(f"{name} must be a lowercase full Git commit")
    return value


def _deployment_evidence(expected_monitor_commit: str) -> dict[str, object]:
    expected = _require_git_commit(
        expected_monitor_commit,
        name="expected monitor commit",
    )
    try:
        head = _git_bytes("rev-parse", "HEAD").decode("ascii").strip()
        top = Path(
            _git_bytes("rev-parse", "--show-toplevel")
            .decode("utf-8", errors="strict")
            .strip()
        ).resolve()
        parent_row = (
            _git_bytes("rev-list", "--parents", "-n", "1", head)
            .decode("ascii")
            .strip()
            .split()
        )
        changed_raw = _git_bytes(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            head,
        )
        changed = frozenset(
            item.decode("utf-8", errors="strict")
            for item in changed_raw.split(b"\0")
            if item
        )
    except (UnicodeDecodeError, OSError) as error:
        raise CheckpointMonitorError(
            "repository deployment evidence is not decodable"
        ) from error
    if head != expected:
        raise CheckpointMonitorError("repository HEAD is not the expected monitor commit")
    if top != ROOT.resolve():
        raise CheckpointMonitorError("monitor is not running from its governed worktree")
    if parent_row != [head, TARGET_PRODUCTION_COMMIT]:
        raise CheckpointMonitorError(
            "monitor commit is not a one-parent direct child of the r3 commit"
        )
    if changed != MONITOR_COMMIT_PATHS:
        raise CheckpointMonitorError(
            "monitor commit changed paths outside its exact deployment allowlist"
        )
    if _git_bytes("status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise CheckpointMonitorError("monitor worktree is not exactly clean")
    return {
        "changed_paths": sorted(changed),
        "monitor_commit": head,
        "monitor_parent_commit": TARGET_PRODUCTION_COMMIT,
        "worktree_clean": True,
    }


def _code_evidence(expected_monitor_commit: str) -> dict[str, object]:
    deployment = _deployment_evidence(expected_monitor_commit)
    observed: list[dict[str, object]] = []
    for relative in TARGET_DEPENDENCY_PATHS:
        target_raw = _git_bytes("show", f"{TARGET_PRODUCTION_COMMIT}:{relative}")
        expected = _sha256_bytes(target_raw)
        path = ROOT / relative
        actual = _sha256_bytes(_read_regular_once(path))
        if actual != expected:
            raise CheckpointMonitorError(
                f"r3 production module differs from {TARGET_PRODUCTION_COMMIT}: {relative}"
            )
        observed.append(
            {
                "path": relative,
                "sha256": actual,
            }
        )
    monitor_path = Path(__file__).resolve()
    return {
        "monitor_path": os.fspath(monitor_path),
        "deployment": deployment,
        "monitor_sha256": _sha256_bytes(_read_regular_once(monitor_path)),
        "production_modules": observed,
        "target_production_commit": TARGET_PRODUCTION_COMMIT,
    }


def _runtime_evidence() -> dict[str, object]:
    attestation = attest_runtime_v3()
    return {
        "authoritative_exact_runtime": True,
        "dependency_lock_sha256": attestation.dependency_lock_sha256,
        "environment_identity_sha256": attestation.environment_identity_sha256,
        "executable_sha256": attestation.executable_sha256,
    }


def _decode_source_manifest_snapshot(raw: bytes) -> Any:
    envelope = _parse_checkpoint_json_object_v3(
        raw,
        name="V4 source-manifest snapshot",
    )
    expected_keys = {"manifest", "manifest_sha256", "schema"}
    if set(envelope) != expected_keys:
        raise CheckpointMonitorError("V4 source-manifest envelope fields drifted")
    if envelope["schema"] != materializer_v4.SOURCE_CACHE_MANIFEST_ARTIFACT_SCHEMA_V4:
        raise CheckpointMonitorError("V4 source-manifest envelope schema drifted")
    try:
        manifest = materializer_v4._source_manifest(envelope["manifest"])
    except Exception as error:
        raise CheckpointMonitorError("V4 source-manifest payload is invalid") from error
    if envelope["manifest_sha256"] != manifest.receipt_sha256:
        raise CheckpointMonitorError("V4 source-manifest typed receipt drifted")
    return manifest


def _load_manifest_targets(
    path: Path,
    *,
    expected_physical_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    raw, manifest = _load_source_manifest_snapshot(
        path,
        expected_physical_sha256=expected_physical_sha256,
    )
    wikipedia = tuple(
        asset for asset in manifest.assets if asset.source_family == WIKIPEDIA_FAMILY
    )
    stackedu = tuple(
        asset for asset in manifest.assets if asset.source_family == STACKEDU_FAMILY
    )
    if len(wikipedia) <= WIKIPEDIA_TARGET_ASSET:
        raise CheckpointMonitorError("source manifest lacks Wikipedia asset 0")
    if len(stackedu) <= STACKEDU_TARGET_ASSET:
        raise CheckpointMonitorError("source manifest lacks StackEdu asset 4")
    wiki_asset = wikipedia[WIKIPEDIA_TARGET_ASSET]
    stack_asset = stackedu[STACKEDU_TARGET_ASSET]
    verified_stack = VerifiedLocalCacheAssetV3(
        expected=stack_asset,
        observed_bytes=stack_asset.bytes,
        observed_sha256=stack_asset.sha256,
    )
    binding = resolve_production_parser_binding_v3(verified_stack)
    if (
        binding.binding_sha256 != STACKEDU_DIRECT_PYTHON_BINDING_SHA256
        or binding.binding_sha256
        != STACKEDU_PYTHON_PARSER_BINDING_V3.binding_sha256
    ):
        raise CheckpointMonitorError(
            "StackEdu source-local asset 4 is not the direct Python binding"
        )
    expected = _require_sha256(
        expected_physical_sha256,
        name="expected source-manifest SHA-256",
    )
    common = {
        "source_manifest_physical_bytes": len(raw),
        "source_manifest_physical_sha256": expected,
        "source_manifest_receipt_sha256": manifest.receipt_sha256,
    }
    return (
        {
            **common,
            "asset_identity_sha256": wiki_asset.asset_identity_sha256,
            "asset_locator": wiki_asset.asset_locator,
            "relative_path": wiki_asset.relative_path,
            "source_family": WIKIPEDIA_FAMILY,
            "source_local_asset_ordinal": WIKIPEDIA_TARGET_ASSET,
        },
        {
            **common,
            "asset_identity_sha256": stack_asset.asset_identity_sha256,
            "asset_locator": stack_asset.asset_locator,
            "direct_python_parser_binding_sha256": binding.binding_sha256,
            "relative_path": stack_asset.relative_path,
            "source_family": STACKEDU_FAMILY,
            "source_local_asset_ordinal": STACKEDU_TARGET_ASSET,
        },
    )


def _load_source_manifest_snapshot(
    path: Path,
    *,
    expected_physical_sha256: str,
) -> tuple[bytes, Any]:
    raw = _read_regular_once(path)
    expected = _require_sha256(
        expected_physical_sha256,
        name="expected source-manifest SHA-256",
    )
    if _sha256_bytes(raw) != expected:
        raise CheckpointMonitorError("source manifest physical SHA-256 drifted")
    # Decode the exact in-memory byte snapshot whose physical hash was checked.
    # Reopening the path here would permit a same-path swap between verification
    # and target selection.
    manifest = _decode_source_manifest_snapshot(raw)
    return raw, manifest


def _checkpoint_paths(child_root: Path, source_family: str) -> tuple[Path, Path]:
    final_path = child_root / "source-parse" / f"{source_family}.jsonl"
    return final_path, _source_parse_checkpoint_root_v3(final_path)


def _directory_names(path: Path) -> tuple[str, ...]:
    assert_no_symlink_ancestors(path)
    if not path.is_dir():
        return ()
    names: list[str] = []
    with os.scandir(path) as entries:
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if stat.S_ISLNK(metadata.st_mode) or bool(
                getattr(metadata, "st_file_attributes", 0) & reparse_flag
            ):
                raise CheckpointMonitorError(
                    "checkpoint root contains a link/reparse child"
                )
            names.append(entry.name)
    return tuple(sorted(names))


def _checkpoint_root_identity(path: Path) -> tuple[int, int] | None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError):
        return None
    if not stat.S_ISDIR(metadata.st_mode):
        raise CheckpointMonitorError("checkpoint root is not a directory")
    return _file_identity(metadata)


def _wikipedia_hint(checkpoint_root: Path) -> bool:
    names = set(_directory_names(checkpoint_root))
    prefix = f"chunk-{WIKIPEDIA_HINT_CHUNK_INDEX:06d}"
    return f"{prefix}.jsonl" in names and f"{prefix}.receipt.json" in names


def _stackedu_hint(checkpoint_root: Path) -> bool:
    receipt_names = sorted(
        (
            name
            for name in _directory_names(checkpoint_root)
            if _CHECKPOINT_RECEIPT.fullmatch(name)
        ),
        reverse=True,
    )
    if not receipt_names:
        return False
    newest = checkpoint_root / receipt_names[0]
    try:
        receipt = _parse_checkpoint_json_object_v3(
            _read_source_parse_child_once_v3(newest),
            name="StackEdu readiness-hint receipt",
        )
    except Exception:
        # A malformed final-name receipt must trigger the full validator, which
        # will fail closed.  The hint itself is never acceptance evidence.
        return True
    last_asset = receipt.get("last_asset_order_ordinal")
    return isinstance(last_asset, int) and last_asset >= STACKEDU_TARGET_ASSET


def _tail_evidence(recovery: Any) -> dict[str, object]:
    if recovery.unexpected_names or recovery.orphan_receipt_names:
        raise CheckpointMonitorError(
            "checkpoint prefix has an unexpected or orphan-receipt tail"
        )
    next_index = len(recovery.receipts)
    next_chunk = f"chunk-{next_index:06d}.jsonl"
    chunk_partial = f"{next_chunk}.partial"
    receipt_partial = f"chunk-{next_index:06d}.receipt.json.partial"
    partials = set(recovery.partial_names)
    orphan_chunks = set(recovery.orphan_chunk_names)
    allowed = False
    if not partials and not orphan_chunks:
        allowed = True
    elif partials == {chunk_partial} and not orphan_chunks:
        allowed = True
    elif not partials and orphan_chunks == {next_chunk}:
        allowed = True
    elif partials == {receipt_partial} and orphan_chunks == {next_chunk}:
        allowed = True
    if not allowed:
        raise CheckpointMonitorError(
            "checkpoint prefix has a noncanonical publication tail"
        )
    return {
        "acceptance_reads_from_tail": False,
        "orphan_chunk_names": list(recovery.orphan_chunk_names),
        "orphan_receipt_names": list(recovery.orphan_receipt_names),
        "partial_names": list(recovery.partial_names),
        "status": recovery.tail_status,
        "unexpected_names": list(recovery.unexpected_names),
    }


def _receipt_path(checkpoint_root: Path, receipt: Mapping[str, object]) -> Path:
    return checkpoint_root / f"chunk-{int(receipt['chunk_index']):06d}.receipt.json"


def _reopen_receipt_and_chunk(
    checkpoint_root: Path,
    receipt: Mapping[str, object],
) -> tuple[list[tuple[dict[str, Any], str]], dict[str, object]]:
    receipt_path = _receipt_path(checkpoint_root, receipt)
    receipt_raw = _read_source_parse_child_once_v3(receipt_path)
    reopened_receipt = _parse_checkpoint_json_object_v3(
        receipt_raw,
        name="target checkpoint receipt",
    )
    if reopened_receipt != dict(receipt):
        raise CheckpointMonitorError("target checkpoint receipt changed after validation")
    chunk_path = checkpoint_root / str(receipt["chunk_name"])
    chunk_raw = _read_source_parse_child_once_v3(chunk_path)
    if (
        len(chunk_raw) != receipt["chunk_bytes"]
        or _sha256_bytes(chunk_raw) != receipt["chunk_sha256"]
    ):
        raise CheckpointMonitorError("target checkpoint chunk changed after validation")
    rows: list[tuple[dict[str, Any], str]] = []
    for line in io.BytesIO(chunk_raw):
        row = _parse_checkpoint_json_object_v3(
            line,
            name="target checkpoint event",
        )
        rows.append((row, _sha256_bytes(line)))
    return rows, {
        "chunk_bytes": len(chunk_raw),
        "chunk_index": receipt["chunk_index"],
        "chunk_name": receipt["chunk_name"],
        "chunk_sha256": receipt["chunk_sha256"],
        "event_end_ordinal_exclusive": receipt["event_end_ordinal_exclusive"],
        "event_start_ordinal": receipt["event_start_ordinal"],
        "receipt_bytes": len(receipt_raw),
        "receipt_name": receipt_path.name,
        "receipt_sha256": _sha256_bytes(receipt_raw),
    }


def _chain_evidence(
    checkpoint_root: Path,
    recovery: Any,
) -> dict[str, object]:
    if not recovery.receipts:
        raise CheckpointMonitorError("checkpoint prefix has no closed receipt pair")
    tip = recovery.receipts[-1]
    tip_path = _receipt_path(checkpoint_root, tip)
    tip_raw = _read_source_parse_child_once_v3(tip_path)
    if _parse_checkpoint_json_object_v3(
        tip_raw,
        name="checkpoint chain-tip receipt",
    ) != dict(tip):
        raise CheckpointMonitorError("checkpoint chain tip changed after validation")
    return {
        "closed_checkpoint_count": len(recovery.receipts),
        "maximal_closed_next_event_ordinal": recovery.next_event_ordinal,
        "receipt_chain_tip_name": tip_path.name,
        "receipt_chain_tip_sha256": _sha256_bytes(tip_raw),
        "tail": _tail_evidence(recovery),
    }


def _evaluate_wikipedia(
    checkpoint_root: Path,
    *,
    asset_identity_sha256: str,
    target_event: int = WIKIPEDIA_TARGET_EVENT,
    target_asset: int = WIKIPEDIA_TARGET_ASSET,
    target_record: int = WIKIPEDIA_TARGET_RECORD,
) -> dict[str, object]:
    recovery = _validate_source_parse_checkpoint_chain_v3(
        checkpoint_root,
        source_family=WIKIPEDIA_FAMILY,
    )
    chain = _chain_evidence(checkpoint_root, recovery)
    candidates = [
        receipt
        for receipt in recovery.receipts
        if int(receipt["event_start_ordinal"])
        <= target_event
        < int(receipt["event_end_ordinal_exclusive"])
    ]
    if len(candidates) != 1:
        raise CheckpointMonitorError(
            "Wikipedia target is not inside exactly one closed checkpoint"
        )
    rows, checkpoint = _reopen_receipt_and_chunk(checkpoint_root, candidates[0])
    matches = [
        (row, line_sha256)
        for row, line_sha256 in rows
        if row.get("event_ordinal") == target_event
    ]
    if len(matches) != 1:
        raise CheckpointMonitorError("Wikipedia target row is absent or repeated")
    row, line_sha256 = matches[0]
    if (
        row.get("source_family") != WIKIPEDIA_FAMILY
        or row.get("asset_order_ordinal") != target_asset
        or row.get("source_record_ordinal") != target_record
        or row.get("source_asset_identity_sha256") != asset_identity_sha256
    ):
        raise CheckpointMonitorError("Wikipedia target row identity drifted")
    return {
        "authoritative": False,
        "checkpoint": checkpoint,
        "checkpoint_chain": chain,
        "event": {
            "asset_order_ordinal": row["asset_order_ordinal"],
            "disposition": row["disposition"],
            "event_ordinal": row["event_ordinal"],
            "event_sha256": row["event_sha256"],
            "physical_line_sha256": line_sha256,
            "source_asset_identity_sha256": row[
                "source_asset_identity_sha256"
            ],
            "source_record_ordinal": row["source_record_ordinal"],
        },
        "gate_minted": False,
        "no_open_tail_reads": True,
        "report_only": True,
        "status": "PASS_CLOSED_CHECKPOINT_REPORT_ONLY_NO_GATE_MINT",
        "target": "wikipedia_event_2951022",
    }


def _evaluate_stackedu(
    checkpoint_root: Path,
    *,
    asset_identity_sha256: str,
    target_asset: int = STACKEDU_TARGET_ASSET,
    target_records: tuple[int, int] = STACKEDU_TARGET_RECORDS,
) -> dict[str, object]:
    recovery = _validate_source_parse_checkpoint_chain_v3(
        checkpoint_root,
        source_family=STACKEDU_FAMILY,
    )
    chain = _chain_evidence(checkpoint_root, recovery)
    candidates = [
        receipt
        for receipt in recovery.receipts
        if int(receipt["first_asset_order_ordinal"])
        <= target_asset
        <= int(receipt["last_asset_order_ordinal"])
    ]
    if not candidates:
        raise CheckpointMonitorError(
            "StackEdu target asset is outside the maximal closed prefix"
        )
    found: dict[int, tuple[dict[str, Any], str, dict[str, object]]] = {}
    for receipt in candidates:
        rows, checkpoint = _reopen_receipt_and_chunk(checkpoint_root, receipt)
        for row, line_sha256 in rows:
            record = row.get("source_record_ordinal")
            if (
                row.get("asset_order_ordinal") == target_asset
                and record in target_records
            ):
                if record in found:
                    raise CheckpointMonitorError("StackEdu target record is repeated")
                found[int(record)] = (row, line_sha256, checkpoint)
    if set(found) != set(target_records):
        raise CheckpointMonitorError(
            "StackEdu target records are not both inside closed checkpoints"
        )
    events: list[dict[str, object]] = []
    event_ordinals: list[int] = []
    checkpoints: dict[int, dict[str, object]] = {}
    for record in target_records:
        row, line_sha256, checkpoint = found[record]
        if (
            row.get("source_family") != STACKEDU_FAMILY
            or row.get("asset_order_ordinal") != target_asset
            or row.get("source_record_ordinal") != record
            or row.get("source_asset_identity_sha256") != asset_identity_sha256
        ):
            raise CheckpointMonitorError("StackEdu target row identity drifted")
        event_ordinal = int(row["event_ordinal"])
        event_ordinals.append(event_ordinal)
        checkpoints[int(checkpoint["chunk_index"])] = checkpoint
        events.append(
            {
                "asset_order_ordinal": row["asset_order_ordinal"],
                "disposition": row["disposition"],
                "event_ordinal": event_ordinal,
                "event_sha256": row["event_sha256"],
                "physical_line_sha256": line_sha256,
                "source_asset_identity_sha256": row[
                    "source_asset_identity_sha256"
                ],
                "source_record_ordinal": row["source_record_ordinal"],
            }
        )
    if event_ordinals[1] != event_ordinals[0] + 1:
        raise CheckpointMonitorError(
            "StackEdu records 0 and 1 do not have consecutive source event ordinals"
        )
    return {
        "authoritative": False,
        "checkpoints": [checkpoints[index] for index in sorted(checkpoints)],
        "checkpoint_chain": chain,
        "direct_python_parser_binding_sha256": (
            STACKEDU_DIRECT_PYTHON_BINDING_SHA256
        ),
        "events": events,
        "gate_minted": False,
        "no_open_tail_reads": True,
        "report_only": True,
        "status": "PASS_CLOSED_CHECKPOINT_REPORT_ONLY_NO_GATE_MINT",
        "target": "stackedu_direct_python_asset_4_records_0_1",
    }


def _pending_condition(target: str, status: str) -> dict[str, object]:
    return {
        "authoritative": False,
        "gate_minted": False,
        "no_open_tail_reads": True,
        "report_only": True,
        "status": status,
        "target": target,
    }


def _source_observation(
    child_root: Path,
    *,
    source_family: str,
    target_asset_identity_sha256: str,
) -> dict[str, object]:
    final_path, checkpoint_root = _checkpoint_paths(child_root, source_family)
    target_name = (
        "wikipedia_event_2951022"
        if source_family == WIKIPEDIA_FAMILY
        else "stackedu_direct_python_asset_4_records_0_1"
    )
    initial_root_identity = _checkpoint_root_identity(checkpoint_root)
    if initial_root_identity is None:
        if final_path.exists():
            return _pending_condition(
                target_name,
                "CHECKPOINT_WINDOW_MISSED_FINAL_PRESENT_NO_ACCEPTANCE",
            )
        return _pending_condition(
            target_name,
            "PENDING_NO_CLOSED_CHECKPOINT_ROOT",
        )

    def replacement_or_completion_status() -> str | None:
        current = _checkpoint_root_identity(checkpoint_root)
        if current is None:
            if final_path.exists():
                return "CHECKPOINT_WINDOW_MISSED_FINAL_PRESENT_NO_ACCEPTANCE"
            return "TRANSIENT_CHECKPOINT_ROOT_REPLACEMENT_NO_ACCEPTANCE"
        if current != initial_root_identity:
            return "TRANSIENT_CHECKPOINT_ROOT_REPLACEMENT_NO_ACCEPTANCE"
        return None

    try:
        hint = (
            _wikipedia_hint(checkpoint_root)
            if source_family == WIKIPEDIA_FAMILY
            else _stackedu_hint(checkpoint_root)
        )
        if not hint:
            return _pending_condition(
                target_name,
                "PENDING_CLOSED_CHECKPOINT_HINT_NOT_REACHED",
            )
        if source_family == WIKIPEDIA_FAMILY:
            return _evaluate_wikipedia(
                checkpoint_root,
                asset_identity_sha256=target_asset_identity_sha256,
            )
        return _evaluate_stackedu(
            checkpoint_root,
            asset_identity_sha256=target_asset_identity_sha256,
        )
    except (FileNotFoundError, NotADirectoryError, OSError):
        status = replacement_or_completion_status()
        return _pending_condition(
            target_name,
            status or "TRANSIENT_CHECKPOINT_IO_RACE_NO_ACCEPTANCE",
        )
    except (CorpusMaterializationError, CheckpointMonitorError):
        status = replacement_or_completion_status()
        if status is not None:
            return _pending_condition(
                target_name,
                status,
            )
        raise


def _snapshot(
    child_root: Path,
    *,
    wikipedia_target: Mapping[str, object],
    stackedu_target: Mapping[str, object],
    provenance: Mapping[str, object],
    captured_wikipedia: Mapping[str, object] | None,
    run_guard: _ChildRunIdentityGuard | None = None,
) -> dict[str, object]:
    if run_guard is not None:
        run_guard.assert_current()
    if not child_root.exists():
        wikipedia = _pending_condition(
            "wikipedia_event_2951022",
            "PENDING_CHILD_OUTPUT_ROOT_ABSENT",
        )
        stackedu = _pending_condition(
            "stackedu_direct_python_asset_4_records_0_1",
            "PENDING_CHILD_OUTPUT_ROOT_ABSENT",
        )
    else:
        assert_no_symlink_ancestors(child_root)
        if not child_root.is_dir():
            raise CheckpointMonitorError("child output root is not a directory")
        marker = child_root / "_INCOMPLETE"
        parse_root = child_root / "source-parse"
        terminal = child_root / CHILD_RECEIPT_FILENAME
        if terminal.exists():
            raise CheckpointMonitorError(
                "child terminal receipt appeared before monitor conditions completed"
            )
        if not marker.exists():
            if not parse_root.exists():
                wikipedia = _pending_condition(
                    "wikipedia_event_2951022",
                    "PENDING_CHILD_ROOT_INITIALIZING",
                )
                stackedu = _pending_condition(
                    "stackedu_direct_python_asset_4_records_0_1",
                    "PENDING_CHILD_ROOT_INITIALIZING",
                )
            else:
                raise CheckpointMonitorError(
                    "active child parse root lacks exact _INCOMPLETE marker"
                )
        else:
            if run_guard is None:
                wikipedia = _pending_condition(
                    "wikipedia_event_2951022",
                    "TRANSIENT_RUN_IDENTITY_ACQUISITION_RACE_NO_ACCEPTANCE",
                )
                stackedu = _pending_condition(
                    "stackedu_direct_python_asset_4_records_0_1",
                    "TRANSIENT_RUN_IDENTITY_ACQUISITION_RACE_NO_ACCEPTANCE",
                )
            else:
                wikipedia = (
                    dict(captured_wikipedia)
                    if captured_wikipedia is not None
                    else _source_observation(
                        child_root,
                        source_family=WIKIPEDIA_FAMILY,
                        target_asset_identity_sha256=str(
                            wikipedia_target["asset_identity_sha256"]
                        ),
                    )
                )
                stackedu = _source_observation(
                    child_root,
                    source_family=STACKEDU_FAMILY,
                    target_asset_identity_sha256=str(
                        stackedu_target["asset_identity_sha256"]
                    ),
                )
                for condition in (wikipedia, stackedu):
                    if (
                        condition.get("status")
                        == "PASS_CLOSED_CHECKPOINT_REPORT_ONLY_NO_GATE_MINT"
                    ):
                        observed_identity = condition.get(
                            "child_run_identity_sha256"
                        )
                        if observed_identity not in (
                            None,
                            run_guard.identity_sha256,
                        ):
                            raise ChildRunIdentityChangedError(
                                "captured condition belongs to a different child run"
                            )
                        condition["child_run_identity_sha256"] = (
                            run_guard.identity_sha256
                        )
                run_guard.assert_current()
    complete = all(
        condition.get("status")
        == "PASS_CLOSED_CHECKPOINT_REPORT_ONLY_NO_GATE_MINT"
        for condition in (wikipedia, stackedu)
    )
    return {
        "authoritative": False,
        "child_output_root": os.fspath(child_root),
        "child_run_identity": (
            None if run_guard is None else dict(run_guard.identity)
        ),
        "child_run_identity_sha256": (
            None if run_guard is None else run_guard.identity_sha256
        ),
        "conditions": {
            "stackedu": stackedu,
            "wikipedia": wikipedia,
        },
        "fresh_replay_required_if_interrupted": True,
        "gate_minted": False,
        "no_artifact_writes": True,
        "no_open_tail_reads": True,
        "provenance": dict(provenance),
        "report_only": True,
        "reusable_output": False,
        "schema": REPORT_SCHEMA,
        "source_targets": {
            "stackedu": dict(stackedu_target),
            "wikipedia": dict(wikipedia_target),
        },
        "status": (
            "REPORT_ONLY_CONDITIONS_SATISFIED_NO_GATE_MINT"
            if complete
            else "PENDING_REPORT_ONLY_NO_GATE_MINT"
        ),
    }


def _emit(value: Mapping[str, object]) -> None:
    sys.stdout.buffer.write(_canonical_json_bytes(dict(value)))
    sys.stdout.buffer.flush()


def _run(arguments: argparse.Namespace) -> int:
    child_root = _lexical_absolute(
        arguments.child_output_root,
        name="child output root",
    )
    source_manifest = _lexical_absolute(
        arguments.source_manifest,
        name="source manifest",
    )
    provenance = {
        "code": _code_evidence(arguments.expected_monitor_commit),
        "runtime": _runtime_evidence(),
    }
    if not source_manifest.is_file():
        raise CheckpointMonitorError("source manifest is absent")
    wikipedia_target, stackedu_target = _load_manifest_targets(
        source_manifest,
        expected_physical_sha256=arguments.expected_source_manifest_sha256,
    )
    run_guard: _ChildRunIdentityGuard | None = None

    def acquire_guard_if_ready() -> _ChildRunIdentityGuard | None:
        marker = child_root / "_INCOMPLETE"
        if not child_root.exists() or not marker.exists():
            return None
        try:
            return _ChildRunIdentityGuard(child_root)
        except (FileNotFoundError, NotADirectoryError):
            return None

    if arguments.command == "once":
        try:
            run_guard = acquire_guard_if_ready()
            snapshot = _snapshot(
                child_root,
                wikipedia_target=wikipedia_target,
                stackedu_target=stackedu_target,
                provenance=provenance,
                captured_wikipedia=None,
                run_guard=run_guard,
            )
            _emit(snapshot)
            return (
                0
                if snapshot["status"]
                == "REPORT_ONLY_CONDITIONS_SATISFIED_NO_GATE_MINT"
                else 3
            )
        finally:
            if run_guard is not None:
                run_guard.close()
    poll_seconds = arguments.poll_seconds
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, float)
        or poll_seconds <= 0.0
        or poll_seconds > 60.0
    ):
        raise CheckpointMonitorError("poll seconds must be in (0, 60]")
    captured_wikipedia: Mapping[str, object] | None = None
    previous_digest: str | None = None
    last_emit = 0.0
    try:
        while True:
            if run_guard is None:
                run_guard = acquire_guard_if_ready()
            snapshot = _snapshot(
                child_root,
                wikipedia_target=wikipedia_target,
                stackedu_target=stackedu_target,
                provenance=provenance,
                captured_wikipedia=captured_wikipedia,
                run_guard=run_guard,
            )
            wikipedia = snapshot["conditions"]["wikipedia"]  # type: ignore[index]
            if (
                captured_wikipedia is None
                and wikipedia["status"]  # type: ignore[index]
                == "PASS_CLOSED_CHECKPOINT_REPORT_ONLY_NO_GATE_MINT"
            ):
                captured_wikipedia = dict(wikipedia)  # type: ignore[arg-type]
            raw = _canonical_json_bytes(snapshot)
            digest = _sha256_bytes(raw)
            now = time.monotonic()
            if digest != previous_digest or now - last_emit >= 60.0:
                sys.stdout.buffer.write(raw)
                sys.stdout.buffer.flush()
                previous_digest = digest
                last_emit = now
            if snapshot["status"] == "REPORT_ONLY_CONDITIONS_SATISFIED_NO_GATE_MINT":
                return 0
            condition_rows = snapshot["conditions"]
            missed = [
                name
                for name in ("wikipedia", "stackedu")
                if str(condition_rows[name]["status"]).startswith(  # type: ignore[index]
                    "CHECKPOINT_WINDOW_MISSED"
                )
            ]
            if missed:
                raise CheckpointMonitorError(
                    "checkpoint window was missed for "
                    + ",".join(missed)
                    + "; no fallback is authorized"
                )
            time.sleep(poll_seconds)
    finally:
        if run_guard is not None:
            run_guard.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = _CanonicalArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("once", "watch"):
        command = commands.add_parser(name)
        command.add_argument("--child-output-root", type=Path, required=True)
        command.add_argument("--source-manifest", type=Path, required=True)
        command.add_argument(
            "--expected-source-manifest-sha256",
            required=True,
        )
        command.add_argument("--expected-monitor-commit", required=True)
        if name == "watch":
            command.add_argument("--poll-seconds", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        arguments = parser.parse_args(argv)
        return _run(arguments)
    except Exception as error:
        failure = {
            "authoritative": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "gate_minted": False,
            "no_artifact_writes": True,
            "no_open_tail_reads": True,
            "report_only": True,
            "schema": ERROR_SCHEMA,
            "status": "FAIL_CLOSED_NO_GATE_MINT",
        }
        sys.stderr.buffer.write(_canonical_json_bytes(failure))
        sys.stderr.buffer.flush()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
