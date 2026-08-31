"""Two-phase DriveFS durability probe for the WEFT-1 source-parse ledger.

``publish`` runs the exact source writer, externally kills one child only after
a closed checkpoint is visible, checks a separate successful reconstruction,
and publishes a stage manifest.  For the strong acceptance test, the operator
records the stage SHA outside the VM and replaces the backend *without* calling
``drive.flush_and_unmount``.  ``verify`` runs in the rebuilt exact runtime,
requires a different kernel mount ID, reopens every durable byte, and publishes
the final receipt.  A deliberate Drive flush/remount is only a weaker transport
canary and must not be reported as surprise-backend-loss durability.  Neither
phase mints a corpus gate, and no probe output is reusable as materialization.

``--local-dry-run`` executes the same two-phase state machine with explicit
simulated mount IDs.  It makes no DriveFS or exact-runtime claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.weft1_corpus_materialize_a2 import (  # noqa: E402
    _DurableSourceParseLedgerV3,
    _source_parse_checkpoint_root_v3,
    _source_parse_directory_fsync_v3,
    _validate_source_parse_checkpoint_chain_v3,
)
from training.weft1_corpus_pa import attest_runtime_v3  # noqa: E402
from training.weft1_corpus_replay_a2 import (  # noqa: E402
    CHILD_RECEIPT_FILENAME,
)
from training.weft1_strict_io import (  # noqa: E402
    assert_no_symlink_ancestors,
)


SOURCE_FAMILY = "wikipedia_wikibooks"
INCOMPLETE_BYTES = b"P-A incomplete\n"
KILL_CHECKPOINT_CADENCE = 2
SUCCESS_EVENT_COUNT = 5
READY_TIMEOUT_SECONDS = 120.0
_STAGE_SCHEMA = "weft1_source_parse_drivefs_durability_stage_v1"
_STAGE_ENVELOPE_SCHEMA = (
    "weft1_source_parse_drivefs_durability_stage_envelope_v1"
)
_FINAL_SCHEMA = "weft1_source_parse_drivefs_durability_final_v1"
_FINAL_ENVELOPE_SCHEMA = (
    "weft1_source_parse_drivefs_durability_final_envelope_v1"
)
_PUBLISH_RESULT_SCHEMA = (
    "weft1_source_parse_drivefs_durability_publish_result_v1"
)
_VERIFY_RESULT_SCHEMA = (
    "weft1_source_parse_drivefs_durability_verify_result_v1"
)
_READY_SCHEMA = "weft1_source_parse_drivefs_durability_probe_ready_v1"
_SHA256_CHARS = frozenset("0123456789abcdef")


class DurabilityProbeError(RuntimeError):
    """The durability probe cannot establish its fail-closed contract."""


class _CanonicalArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise DurabilityProbeError(f"invalid CLI arguments: {message}")


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


def _read_regular_once(path: Path) -> bytes:
    lexical = assert_no_symlink_ancestors(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as error:
        raise DurabilityProbeError(f"cannot open governed file: {lexical}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise DurabilityProbeError(f"governed child is not regular: {lexical}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise DurabilityProbeError(f"canonical JSON repeats key: {key}")
        value[key] = item
    return value


def _parse_canonical_object(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                DurabilityProbeError(
                    f"{name} uses non-finite JSON constant {constant}"
                )
            ),
        )
    except DurabilityProbeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise DurabilityProbeError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise DurabilityProbeError(f"{name} is not canonical JSON")
    return value


def _load_canonical_object(path: Path, *, name: str) -> dict[str, Any]:
    return _parse_canonical_object(_read_regular_once(path), name=name)


def _write_fresh_fsynced(path: Path, payload: bytes) -> str:
    lexical = assert_no_symlink_ancestors(path)
    parent = assert_no_symlink_ancestors(lexical.parent)
    if not parent.is_dir():
        raise DurabilityProbeError(f"governed parent is absent: {parent}")
    if lexical.exists() or lexical.is_symlink():
        raise DurabilityProbeError(f"governed output must be fresh: {lexical}")
    try:
        with lexical.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise DurabilityProbeError(
            f"cannot write governed output: {lexical}"
        ) from error
    directory_fsync = _source_parse_directory_fsync_v3(parent)
    if _read_regular_once(lexical) != payload:
        raise DurabilityProbeError(
            f"governed output failed close/reopen verification: {lexical}"
        )
    return directory_fsync


def _mkdir_fresh(path: Path) -> str:
    lexical = assert_no_symlink_ancestors(path)
    parent = assert_no_symlink_ancestors(lexical.parent)
    if not parent.is_dir():
        raise DurabilityProbeError(f"directory parent is absent: {parent}")
    if lexical.exists() or lexical.is_symlink():
        raise DurabilityProbeError(f"directory must be fresh: {lexical}")
    try:
        lexical.mkdir()
    except OSError as error:
        raise DurabilityProbeError(
            f"cannot create fresh directory: {lexical}"
        ) from error
    if not lexical.is_dir():
        raise DurabilityProbeError(f"fresh directory did not materialize: {lexical}")
    return _source_parse_directory_fsync_v3(parent)


def _normalized_path(path: Path) -> Path:
    return assert_no_symlink_ancestors(path)


def _is_within(path: Path, parent: Path) -> bool:
    path_text = os.path.normcase(os.fspath(path))
    parent_text = os.path.normcase(os.fspath(parent))
    try:
        return os.path.commonpath((path_text, parent_text)) == parent_text
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _decode_mountinfo_path(value: str) -> str:
    for escaped, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(escaped, decoded)
    return value


def _drivefs_mount_evidence(path: Path) -> dict[str, object]:
    if os.name != "posix":
        raise DurabilityProbeError("exact DriveFS probe requires a POSIX runtime")
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        raise DurabilityProbeError("exact DriveFS probe cannot inspect mountinfo")
    candidates: list[tuple[int, dict[str, str]]] = []
    for line in mountinfo.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator + 2 >= len(fields) or len(fields) < 5:
            continue
        mount_point = Path(_decode_mountinfo_path(fields[4]))
        if not _is_within(path, mount_point):
            continue
        candidates.append(
            (
                len(os.fspath(mount_point)),
                {
                    "filesystem_type": fields[separator + 1],
                    "major_minor": fields[2],
                    "mount_id": fields[0],
                    "mount_point": os.fspath(mount_point),
                    "mount_source": fields[separator + 2],
                    "mountinfo_line_sha256": _sha256_bytes(
                        line.encode("utf-8")
                    ),
                    "parent_mount_id": fields[1],
                },
            )
        )
    if not candidates:
        raise DurabilityProbeError("durable root has no mountinfo entry")
    evidence = max(candidates, key=lambda item: item[0])[1]
    identity = " ".join(evidence.values()).lower()
    if "drive" not in identity or (
        "fuse" not in evidence["filesystem_type"].lower()
        and "drivefs" not in identity
    ):
        raise DurabilityProbeError(
            "durable root is not backed by an identifiable DriveFS/FUSE mount"
        )
    return {
        **evidence,
        "classification": "DRIVEFS_OR_GOOGLE_DRIVE_FUSE",
        "mountinfo_verified": True,
    }


def _runtime_evidence(*, local_dry_run: bool) -> dict[str, object]:
    executable = Path(sys.executable)
    if local_dry_run:
        resolved_executable = executable.resolve()
        evidence: dict[str, object] = {
            "authoritative_exact_runtime": False,
            "executable": os.fspath(resolved_executable),
            "executable_sha256": _sha256_bytes(
                _read_regular_once(resolved_executable)
            ),
            "mode": "LOCAL_TMP_DRY_RUN_NO_DRIVE_CLAIM",
            "python_version": platform.python_version(),
        }
    else:
        attestation = attest_runtime_v3()
        evidence = {
            "authoritative_exact_runtime": True,
            "dependency_lock_sha256": attestation.dependency_lock_sha256,
            "environment_identity_sha256": attestation.environment_identity_sha256,
            "environment_payload": dict(attestation.environment_payload),
            "executable": os.fspath(executable),
            "executable_sha256": attestation.executable_sha256,
            "mode": "EXACT_RUNTIME_DRIVEFS",
            "python_version": platform.python_version(),
        }
    # Runtime attestation contains tuple-valued nested inventories.  Normalize
    # at the evidence boundary so the in-memory value and its staged JSON reload
    # use the same list-valued representation during exact verify comparison.
    return _parse_canonical_object(
        _canonical_json_bytes(evidence),
        name="runtime evidence",
    )


def _code_evidence() -> dict[str, object]:
    materializer = ROOT / "training" / "weft1_corpus_materialize_a2.py"
    probe = Path(__file__).resolve()
    return {
        "materializer_path": os.fspath(materializer),
        "materializer_sha256": _sha256_bytes(_read_regular_once(materializer)),
        "probe_path": os.fspath(probe),
        "probe_sha256": _sha256_bytes(_read_regular_once(probe)),
    }


def _simulated_mount_evidence(*, phase: str) -> dict[str, object]:
    if phase not in {"publish", "verify"}:
        raise ValueError("local mount simulation phase is invalid")
    return {
        "classification": "LOCAL_TMP_SIMULATION_NO_DRIVE_CLAIM",
        "filesystem_type": "local-dry-run",
        "mount_id": f"LOCAL_DRY_RUN_SIMULATED_{phase.upper()}_MOUNT",
        "mount_point": None,
        "mount_source": None,
        "mountinfo_verified": False,
        "simulation_phase": phase,
    }


def _event_payload(
    ordinal: int,
    *,
    asset_order_ordinal: int,
    source_record_ordinal: int,
) -> bytes:
    event = {
        "asset_order_ordinal": asset_order_ordinal,
        "disposition": "RETAIN",
        "event_ordinal": ordinal,
        "event_sha256": _sha256_bytes(f"probe-event:{ordinal}".encode("utf-8")),
        "source_asset_identity_sha256": _sha256_bytes(
            f"probe-asset:{asset_order_ordinal}".encode("utf-8")
        ),
        "source_family": SOURCE_FAMILY,
        "source_record_ordinal": source_record_ordinal,
    }
    return _canonical_json_bytes(event)


def _killed_prefix_bytes() -> bytes:
    return b"".join(
        _event_payload(
            ordinal,
            asset_order_ordinal=0,
            source_record_ordinal=ordinal,
        )
        for ordinal in range(KILL_CHECKPOINT_CADENCE)
    )


def _success_legacy_bytes() -> bytes:
    return b"".join(
        _event_payload(
            ordinal,
            asset_order_ordinal=0 if ordinal < 3 else 1,
            source_record_ordinal=ordinal if ordinal < 3 else ordinal - 3,
        )
        for ordinal in range(SUCCESS_EVENT_COUNT)
    )


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _tree_projection(root: Path) -> dict[str, object]:
    lexical_root = assert_no_symlink_ancestors(root)
    if not lexical_root.is_dir():
        raise DurabilityProbeError(f"tree root is absent: {lexical_root}")
    rows: list[dict[str, object]] = []
    directories: list[str] = []
    stack = [lexical_root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise DurabilityProbeError(f"cannot enumerate tree: {directory}") from error
        for entry in entries:
            path = Path(entry.path)
            metadata = entry.stat(follow_symlinks=False)
            if _is_link_or_reparse(metadata):
                raise DurabilityProbeError(f"tree contains link/reparse child: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                directories.append(path.relative_to(lexical_root).as_posix())
                stack.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise DurabilityProbeError(f"tree contains non-regular child: {path}")
            payload = _read_regular_once(path)
            rows.append(
                {
                    "bytes": len(payload),
                    "path": path.relative_to(lexical_root).as_posix(),
                    "sha256": _sha256_bytes(payload),
                }
            )
    return {
        "directories": sorted(directories),
        "files": sorted(rows, key=lambda row: str(row["path"])),
    }


def _assert_exact_tree(
    root: Path,
    *,
    expected_directories: set[str],
    expected_files: set[str],
) -> dict[str, object]:
    projection = _tree_projection(root)
    observed_files = {
        str(row["path"])
        for row in projection["files"]  # type: ignore[union-attr]
    }
    observed_directories = set(projection["directories"])  # type: ignore[arg-type]
    if observed_files != expected_files or observed_directories != expected_directories:
        raise DurabilityProbeError(
            "durable tree shape differs from probe contract: "
            f"expected_files={sorted(expected_files)!r} "
            f"observed_files={sorted(observed_files)!r} "
            f"expected_directories={sorted(expected_directories)!r} "
            f"observed_directories={sorted(observed_directories)!r}"
        )
    return projection


def _child_ready_marker(
    ready_marker: Path,
    receipt: Mapping[str, object],
) -> None:
    if receipt.get("chunk_index") != 0:
        raise DurabilityProbeError("kill child reached an unexpected checkpoint")
    marker = {
        "checkpoint_receipt_payload_sha256": _sha256_bytes(
            _canonical_json_bytes(dict(receipt))
        ),
        "chunk_index": receipt["chunk_index"],
        "chunk_name": receipt["chunk_name"],
        "event_end_ordinal_exclusive": receipt["event_end_ordinal_exclusive"],
        "event_start_ordinal": receipt["event_start_ordinal"],
        "schema": _READY_SCHEMA,
        "source_family": receipt["source_family"],
        "status": "FULLY_PUBLISHED_CLOSED_CHUNK_READY_FOR_PARENT_KILL",
    }
    _write_fresh_fsynced(ready_marker, _canonical_json_bytes(marker))
    while True:
        time.sleep(60.0)


def _run_hard_kill_child(arguments: argparse.Namespace) -> int:
    if arguments.require_exact_runtime:
        attest_runtime_v3()
    final_path = _normalized_path(arguments.final_path)
    local_root = _normalized_path(arguments.local_root)
    ready_marker = _normalized_path(arguments.ready_marker)
    writer = _DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=local_root,
        source_family=SOURCE_FAMILY,
        checkpoint_event_cadence=KILL_CHECKPOINT_CADENCE,
        after_checkpoint=lambda receipt: _child_ready_marker(ready_marker, receipt),
    )
    with writer:
        for ordinal in range(KILL_CHECKPOINT_CADENCE + 2):
            payload = _event_payload(
                ordinal,
                asset_order_ordinal=0,
                source_record_ordinal=ordinal,
            )
            writer.write(
                payload,
                event_ordinal=ordinal,
                asset_order_ordinal=0,
                source_record_ordinal=ordinal,
            )
            writer.commit_event(ordinal)
    raise DurabilityProbeError("kill child returned without an external hard kill")


def _wait_for_ready_and_kill(
    process: subprocess.Popen[bytes],
    *,
    ready_marker: Path,
) -> tuple[int, dict[str, Any], bytes, bytes]:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    marker: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            stdout, stderr = process.communicate()
            raise DurabilityProbeError(
                "kill child exited before parent termination: "
                f"returncode={returncode} stdout={stdout[-2000:]!r} "
                f"stderr={stderr[-4000:]!r}"
            )
        if ready_marker.exists() or ready_marker.is_symlink():
            marker = _load_canonical_object(
                ready_marker, name="hard-kill readiness marker"
            )
            break
        time.sleep(0.05)
    if marker is None:
        process.kill()
        stdout, stderr = process.communicate(timeout=30.0)
        raise DurabilityProbeError(
            "kill child did not publish readiness marker before timeout: "
            f"stdout={stdout[-2000:]!r} stderr={stderr[-4000:]!r}"
        )
    expected_marker = {
        "chunk_index": 0,
        "event_end_ordinal_exclusive": KILL_CHECKPOINT_CADENCE,
        "event_start_ordinal": 0,
        "schema": _READY_SCHEMA,
        "source_family": SOURCE_FAMILY,
        "status": "FULLY_PUBLISHED_CLOSED_CHUNK_READY_FOR_PARENT_KILL",
    }
    for key, expected in expected_marker.items():
        if marker.get(key) != expected:
            process.kill()
            process.communicate(timeout=30.0)
            raise DurabilityProbeError(
                f"hard-kill readiness marker field drifted: {key}"
            )
    process.kill()
    stdout, stderr = process.communicate(timeout=30.0)
    returncode = int(process.returncode)
    if returncode == 0:
        raise DurabilityProbeError("hard-killed child unexpectedly returned success")
    if os.name == "posix" and returncode != -signal.SIGKILL:
        raise DurabilityProbeError(
            f"child did not terminate by SIGKILL: returncode={returncode}"
        )
    return returncode, marker, stdout, stderr


def _run_killed_ledger(
    *,
    durable_root: Path,
    local_root: Path,
    require_exact_runtime: bool,
) -> dict[str, object]:
    output_root = durable_root / "killed-output"
    _mkdir_fresh(output_root)
    parse_root = output_root / "source-parse"
    _mkdir_fresh(parse_root)
    incomplete_path = output_root / "_INCOMPLETE"
    incomplete_directory_fsync = _write_fresh_fsynced(
        incomplete_path, INCOMPLETE_BYTES
    )
    final_path = parse_root / f"{SOURCE_FAMILY}.jsonl"
    child_local_root = local_root / "killed-ledger"
    _mkdir_fresh(child_local_root)
    ready_marker = local_root / "killed-ready.json"
    command = [
        sys.executable,
        "-I",
        "-B",
        os.fspath(Path(__file__).resolve()),
        "_hard-kill-child",
        "--final-path",
        os.fspath(final_path),
        "--local-root",
        os.fspath(child_local_root),
        "--ready-marker",
        os.fspath(ready_marker),
    ]
    if require_exact_runtime:
        command.append("--require-exact-runtime")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    returncode, ready, stdout, stderr = _wait_for_ready_and_kill(
        process,
        ready_marker=ready_marker,
    )
    checkpoint_root = _source_parse_checkpoint_root_v3(final_path)
    recovery = _validate_source_parse_checkpoint_chain_v3(
        checkpoint_root,
        source_family=SOURCE_FAMILY,
    )
    if (
        len(recovery.receipts) != 1
        or recovery.next_event_ordinal != KILL_CHECKPOINT_CADENCE
        or recovery.tail_status != "CLEAN"
        or recovery.partial_names
        or recovery.orphan_chunk_names
        or recovery.orphan_receipt_names
        or recovery.unexpected_names
    ):
        raise DurabilityProbeError(
            "hard-kill recovery did not yield one clean maximal closed prefix"
        )
    receipt = recovery.receipts[0]
    chunk_path = checkpoint_root / str(receipt["chunk_name"])
    prefix = _read_regular_once(chunk_path)
    expected_prefix = _killed_prefix_bytes()
    if prefix != expected_prefix:
        raise DurabilityProbeError("hard-kill maximal prefix bytes drifted")
    receipt_path = checkpoint_root / "chunk-000000.receipt.json"
    receipt_bytes = _read_regular_once(receipt_path)
    if ready.get("checkpoint_receipt_payload_sha256") != _sha256_bytes(
        receipt_bytes
    ):
        raise DurabilityProbeError(
            "readiness marker does not bind the published checkpoint receipt"
        )
    if final_path.exists() or final_path.is_symlink():
        raise DurabilityProbeError("hard-killed ledger exposed a final ledger")
    if _read_regular_once(incomplete_path) != INCOMPLETE_BYTES:
        raise DurabilityProbeError("hard-killed ledger changed _INCOMPLETE")
    checkpoint_relative = checkpoint_root.relative_to(output_root).as_posix()
    expected_files = {
        "_INCOMPLETE",
        f"{checkpoint_relative}/chunk-000000.jsonl",
        f"{checkpoint_relative}/chunk-000000.receipt.json",
    }
    survivors = _assert_exact_tree(
        output_root,
        expected_directories={
            "source-parse",
            checkpoint_relative,
        },
        expected_files=expected_files,
    )
    prohibited_terminal_paths = (
        output_root / CHILD_RECEIPT_FILENAME,
        output_root / "content-manifest.json",
        output_root / "d1-ready-manifest.json",
    )
    if any(path.exists() or path.is_symlink() for path in prohibited_terminal_paths):
        raise DurabilityProbeError("hard-killed ledger exposed a terminal receipt")
    return {
        "authoritative": False,
        "checkpoint_resume_authorized": False,
        "checkpoint_receipt": dict(receipt),
        "checkpoint_receipt_bytes": len(receipt_bytes),
        "checkpoint_receipt_sha256": _sha256_bytes(receipt_bytes),
        "child_stderr_bytes": len(stderr),
        "child_stdout_bytes": len(stdout),
        "durable_max_event_ordinal": KILL_CHECKPOINT_CADENCE - 1,
        "final_ledger_present": False,
        "fresh_replay_required": True,
        "gate_minted": False,
        "incomplete": {
            "bytes": len(INCOMPLETE_BYTES),
            "directory_fsync": incomplete_directory_fsync,
            "path": incomplete_path.relative_to(durable_root).as_posix(),
            "present_unchanged": True,
            "sha256": _sha256_bytes(INCOMPLETE_BYTES),
        },
        "maximal_closed_prefix": {
            "bytes": len(prefix),
            "event_end_ordinal_exclusive": KILL_CHECKPOINT_CADENCE,
            "event_start_ordinal": 0,
            "sha256": _sha256_bytes(prefix),
            "tail_status": recovery.tail_status,
        },
        "parent_terminal_receipt_present": False,
        "readiness_marker_sha256": _sha256_bytes(_read_regular_once(ready_marker)),
        "reusable_output": False,
        "status": "NO_GATE_MINT_INCOMPLETE_FRESH_REPLAY_REQUIRED",
        "subprocess_returncode": returncode,
        "surviving_files": survivors,
        "termination_method": "subprocess.Popen.kill",
    }


def _run_success_ledger(
    *,
    durable_root: Path,
    local_root: Path,
) -> dict[str, object]:
    output_root = durable_root / "success-output"
    _mkdir_fresh(output_root)
    parse_root = output_root / "source-parse"
    _mkdir_fresh(parse_root)
    final_path = parse_root / f"{SOURCE_FAMILY}.jsonl"
    legacy_path = local_root / "success-legacy-reference.jsonl"
    legacy_bytes = _success_legacy_bytes()
    legacy_directory_fsync = _write_fresh_fsynced(legacy_path, legacy_bytes)
    success_local_root = local_root / "success-ledger"
    _mkdir_fresh(success_local_root)
    writer = _DurableSourceParseLedgerV3(
        final_path=final_path,
        local_root=success_local_root,
        source_family=SOURCE_FAMILY,
        checkpoint_event_cadence=KILL_CHECKPOINT_CADENCE,
    )
    with writer:
        for ordinal in range(SUCCESS_EVENT_COUNT):
            asset = 0 if ordinal < 3 else 1
            source_record = ordinal if ordinal < 3 else ordinal - 3
            payload = _event_payload(
                ordinal,
                asset_order_ordinal=asset,
                source_record_ordinal=source_record,
            )
            writer.write(
                payload,
                event_ordinal=ordinal,
                asset_order_ordinal=asset,
                source_record_ordinal=source_record,
            )
            writer.commit_event(ordinal)
            if ordinal == 2:
                writer.seal_asset_boundary()
        returned_sha256 = writer.finish()
    final_bytes = _read_regular_once(final_path)
    reference_bytes = _read_regular_once(legacy_path)
    expected_sha256 = _sha256_bytes(reference_bytes)
    if (
        final_bytes != reference_bytes
        or final_bytes != legacy_bytes
        or returned_sha256 != expected_sha256
        or _sha256_bytes(final_bytes) != expected_sha256
    ):
        raise DurabilityProbeError(
            "successful durable ledger differs from direct legacy byte stream"
        )
    checkpoint_root = _source_parse_checkpoint_root_v3(final_path)
    partial = final_path.with_name(final_path.name + ".partial")
    if (
        checkpoint_root.exists()
        or checkpoint_root.is_symlink()
        or partial.exists()
        or partial.is_symlink()
    ):
        raise DurabilityProbeError(
            "successful durable ledger retained checkpoint or partial state"
        )
    final_relative = final_path.relative_to(output_root).as_posix()
    survivors = _assert_exact_tree(
        output_root,
        expected_directories={"source-parse"},
        expected_files={final_relative},
    )
    return {
        "authoritative": False,
        "checkpoint_root_cleaned": True,
        "event_count": SUCCESS_EVENT_COUNT,
        "final_bytes": len(final_bytes),
        "final_path": final_path.relative_to(durable_root).as_posix(),
        "final_sha256": _sha256_bytes(final_bytes),
        "gate_minted": False,
        "legacy_reference_bytes": len(reference_bytes),
        "legacy_reference_directory_fsync": legacy_directory_fsync,
        "legacy_reference_sha256": expected_sha256,
        "legacy_stream_exact_match": True,
        "parent_terminal_receipt_present": False,
        "reusable_output": False,
        "status": "COMPONENT_PROBE_PASS_NO_GATE_MINT",
        "surviving_files": survivors,
    }


def _publish_artifact(
    path: Path,
    envelope: Mapping[str, object],
    *,
    expected_directory_fsync: str,
) -> dict[str, object]:
    lexical = assert_no_symlink_ancestors(path)
    parent = assert_no_symlink_ancestors(lexical.parent)
    if not parent.is_dir():
        raise DurabilityProbeError("artifact parent is absent")
    partial = lexical.with_name(f".{lexical.name}.partial-{os.getpid()}")
    assert_no_symlink_ancestors(partial)
    if (
        lexical.exists()
        or lexical.is_symlink()
        or partial.exists()
        or partial.is_symlink()
    ):
        raise DurabilityProbeError("artifact and artifact partial must be fresh")
    raw = _canonical_json_bytes(dict(envelope))
    try:
        with partial.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise DurabilityProbeError("cannot stage canonical probe artifact") from error
    if _read_regular_once(partial) != raw:
        raise DurabilityProbeError("staged artifact failed close/reopen verification")
    if lexical.exists() or lexical.is_symlink():
        raise DurabilityProbeError("artifact path appeared during publication")
    try:
        os.replace(partial, lexical)
    except OSError as error:
        raise DurabilityProbeError("atomic artifact replacement failed") from error
    directory_fsync = _source_parse_directory_fsync_v3(parent)
    if directory_fsync != expected_directory_fsync:
        raise DurabilityProbeError(
            "artifact directory-fsync support changed during publication"
        )
    first = _read_regular_once(lexical)
    second = _read_regular_once(lexical)
    if first != raw or second != raw:
        raise DurabilityProbeError("published artifact failed two physical reopens")
    return {
        "bytes": len(raw),
        "directory_fsync": directory_fsync,
        "path": os.fspath(lexical),
        "reopen_count": 2,
        "sha256": _sha256_bytes(raw),
    }


def _require_sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise DurabilityProbeError(f"{name} must be a lowercase SHA-256")
    return value


def _assert_fresh_sibling_artifact(path: Path, *, durable_root: Path) -> None:
    if path.parent != durable_root.parent:
        raise DurabilityProbeError("governed artifact must be a durable-root sibling")
    if path.exists() or path.is_symlink():
        raise DurabilityProbeError(f"governed artifact must be fresh: {path}")
    prefix = f".{path.name}.partial-"
    for child in path.parent.iterdir():
        assert_no_symlink_ancestors(child)
        if child.name.startswith(prefix):
            raise DurabilityProbeError(
                f"governed artifact has a prior partial: {child}"
            )


def _prepare_publish_paths(
    arguments: argparse.Namespace,
) -> tuple[Path, Path, Path, Path | None]:
    durable_root = _normalized_path(arguments.durable_root)
    local_root = _normalized_path(arguments.local_root)
    stage_manifest = _normalized_path(arguments.stage_manifest_out)
    mount_root = (
        None
        if arguments.durable_mount_root is None
        else _normalized_path(arguments.durable_mount_root)
    )
    for candidate, name in (
        (durable_root, "durable root"),
        (local_root, "local root"),
        (stage_manifest, "stage manifest"),
    ):
        if candidate.exists() or candidate.is_symlink():
            raise DurabilityProbeError(f"{name} must be fresh: {candidate}")
        if not candidate.parent.is_dir():
            raise DurabilityProbeError(f"{name} parent is absent: {candidate.parent}")
    _assert_fresh_sibling_artifact(stage_manifest, durable_root=durable_root)
    if _paths_overlap(durable_root, local_root):
        raise DurabilityProbeError("durable and local roots must be disjoint")
    if stage_manifest == durable_root or stage_manifest == local_root:
        raise DurabilityProbeError("stage manifest collides with a probe root")
    if arguments.local_dry_run:
        if mount_root is not None:
            raise DurabilityProbeError(
                "local dry-run may not claim a durable mount root"
            )
    else:
        if mount_root is None or not mount_root.is_dir():
            raise DurabilityProbeError(
                "exact probe requires an existing --durable-mount-root"
            )
        if not _is_within(durable_root, mount_root) or not _is_within(
            stage_manifest, mount_root
        ):
            raise DurabilityProbeError(
                "durable root and stage manifest must lie below durable mount root"
            )
        if _is_within(local_root, mount_root):
            raise DurabilityProbeError("local scratch root must be outside DriveFS")
    return durable_root, local_root, stage_manifest, mount_root


def _prepare_verify_paths(
    arguments: argparse.Namespace,
) -> tuple[Path, Path, Path, Path | None]:
    durable_root = _normalized_path(arguments.durable_root)
    stage_manifest = _normalized_path(arguments.stage_manifest)
    final_receipt = _normalized_path(arguments.final_receipt_out)
    mount_root = (
        None
        if arguments.durable_mount_root is None
        else _normalized_path(arguments.durable_mount_root)
    )
    if not durable_root.is_dir():
        raise DurabilityProbeError("durable probe root is absent")
    if not stage_manifest.is_file():
        raise DurabilityProbeError("stage manifest is absent")
    if stage_manifest.parent != durable_root.parent:
        raise DurabilityProbeError("stage manifest is not a durable-root sibling")
    if final_receipt == stage_manifest:
        raise DurabilityProbeError("final receipt collides with stage manifest")
    if not final_receipt.parent.is_dir():
        raise DurabilityProbeError("final receipt parent is absent")
    _assert_fresh_sibling_artifact(final_receipt, durable_root=durable_root)
    if arguments.local_dry_run:
        if mount_root is not None:
            raise DurabilityProbeError(
                "local dry-run may not claim a durable mount root"
            )
    else:
        if mount_root is None or not mount_root.is_dir():
            raise DurabilityProbeError(
                "exact verify requires an existing --durable-mount-root"
            )
        if any(
            not _is_within(path, mount_root)
            for path in (durable_root, stage_manifest, final_receipt)
        ):
            raise DurabilityProbeError(
                "durable root and both receipts must lie below durable mount root"
            )
    return durable_root, stage_manifest, final_receipt, mount_root


def _storage_evidence(
    *,
    local_dry_run: bool,
    phase: str,
    durable_parent: Path,
    mount_root: Path | None,
) -> dict[str, object]:
    if local_dry_run:
        return _simulated_mount_evidence(phase=phase)
    assert mount_root is not None
    evidence = _drivefs_mount_evidence(durable_parent)
    evidence["declared_durable_mount_root"] = os.fspath(mount_root)
    return evidence


def _expected_full_tree(durable_root: Path) -> dict[str, object]:
    checkpoint_directory = (
        "killed-output/source-parse/"
        f".{SOURCE_FAMILY}.jsonl.checkpoints"
    )
    return _assert_exact_tree(
        durable_root,
        expected_directories={
            "killed-output",
            "killed-output/source-parse",
            checkpoint_directory,
            "success-output",
            "success-output/source-parse",
        },
        expected_files={
            "killed-output/_INCOMPLETE",
            f"{checkpoint_directory}/chunk-000000.jsonl",
            f"{checkpoint_directory}/chunk-000000.receipt.json",
            f"success-output/source-parse/{SOURCE_FAMILY}.jsonl",
        },
    )


def _stage_publication_contract(directory_fsync: str) -> dict[str, object]:
    return {
        "atomic_primitive": "os.replace",
        "directory_fsync_status": directory_fsync,
        "file_fsync_before_replace": True,
        "fresh_sibling_required": True,
        "physical_reopen_count": 2,
        "staged_partial_reopened_before_replace": True,
    }


def _load_stage_manifest(
    path: Path,
    *,
    expected_physical_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    expected = _require_sha256(
        expected_physical_sha256,
        name="expected stage-manifest SHA-256",
    )
    first = _read_regular_once(path)
    second = _read_regular_once(path)
    if first != second or _sha256_bytes(first) != expected:
        raise DurabilityProbeError(
            "stage manifest failed physical reopen or expected-hash binding"
        )
    envelope = _parse_canonical_object(first, name="stage manifest envelope")
    if set(envelope) != {"schema", "stage_manifest", "stage_payload_sha256"}:
        raise DurabilityProbeError("stage manifest envelope keys drifted")
    if envelope.get("schema") != _STAGE_ENVELOPE_SCHEMA:
        raise DurabilityProbeError("stage manifest envelope schema drifted")
    stage = envelope.get("stage_manifest")
    if not isinstance(stage, dict):
        raise DurabilityProbeError("stage manifest payload is not an object")
    payload_sha256 = _sha256_bytes(_canonical_json_bytes(stage))
    if envelope.get("stage_payload_sha256") != payload_sha256:
        raise DurabilityProbeError("stage manifest payload hash drifted")
    if stage.get("schema") != _STAGE_SCHEMA:
        raise DurabilityProbeError("stage manifest schema drifted")
    return first, stage


def _publish_probe(arguments: argparse.Namespace) -> dict[str, object]:
    durable_root, local_root, stage_manifest, mount_root = _prepare_publish_paths(
        arguments
    )
    runtime = _runtime_evidence(local_dry_run=arguments.local_dry_run)
    code = _code_evidence()
    storage = _storage_evidence(
        local_dry_run=arguments.local_dry_run,
        phase="publish",
        durable_parent=durable_root.parent,
        mount_root=mount_root,
    )
    durable_root_parent_fsync = _mkdir_fresh(durable_root)
    local_root_parent_fsync = _mkdir_fresh(local_root)
    killed = _run_killed_ledger(
        durable_root=durable_root,
        local_root=local_root,
        require_exact_runtime=not arguments.local_dry_run,
    )
    success = _run_success_ledger(durable_root=durable_root, local_root=local_root)
    durable_tree = _expected_full_tree(durable_root)
    if _code_evidence() != code:
        raise DurabilityProbeError("probe code changed during publish phase")
    if _runtime_evidence(local_dry_run=arguments.local_dry_run) != runtime:
        raise DurabilityProbeError("runtime identity changed during publish phase")
    storage_after = _storage_evidence(
        local_dry_run=arguments.local_dry_run,
        phase="publish",
        durable_parent=durable_root.parent,
        mount_root=mount_root,
    )
    if storage_after.get("mount_id") != storage.get("mount_id"):
        raise DurabilityProbeError("durable mount changed during publish phase")
    artifact_directory_fsync = _source_parse_directory_fsync_v3(
        stage_manifest.parent
    )
    stage: dict[str, object] = {
        "authoritative": False,
        "code": code,
        "durable_root": os.fspath(durable_root),
        "durable_tree_projection": durable_tree,
        "fresh_replay_required_for_corpus_work": True,
        "gate_minted": False,
        "killed_ledger": killed,
        "local_root": os.fspath(local_root),
        "probe_outputs_reusable": False,
        "publication_contract": _stage_publication_contract(
            artifact_directory_fsync
        ),
        "result_classification": "NO_GATE_MINT",
        "runtime": runtime,
        "schema": _STAGE_SCHEMA,
        "source_family": SOURCE_FAMILY,
        "status": "PUBLISHED_AWAITING_PROVIDER_REMOUNT_NO_GATE_MINT",
        "storage_before_provider_remount": storage_after,
        "success_ledger": success,
        "top_level_directory_fsync": {
            "durable_root_parent": durable_root_parent_fsync,
            "local_root_parent": local_root_parent_fsync,
        },
    }
    stage_bytes = _canonical_json_bytes(stage)
    envelope = {
        "schema": _STAGE_ENVELOPE_SCHEMA,
        "stage_manifest": stage,
        "stage_payload_sha256": _sha256_bytes(stage_bytes),
    }
    publication = _publish_artifact(
        stage_manifest,
        envelope,
        expected_directory_fsync=artifact_directory_fsync,
    )
    return {
        "authoritative": False,
        "gate_minted": False,
        "next_required_action": (
            "record the stage SHA-256 outside the VM; replace the backend "
            "without drive.flush_and_unmount; rebuild the exact runtime, mount "
            "Drive, then verify with --barrier-kind "
            "unflushed-backend-replacement"
            if not arguments.local_dry_run
            else "run verify to exercise the simulated remount boundary"
        ),
        "publication": publication,
        "schema": _PUBLISH_RESULT_SCHEMA,
        "status": "PUBLISHED_AWAITING_PROVIDER_REMOUNT_NO_GATE_MINT",
    }


def _verify_probe(arguments: argparse.Namespace) -> dict[str, object]:
    durable_root, stage_path, final_receipt, mount_root = _prepare_verify_paths(
        arguments
    )
    stage_raw, stage = _load_stage_manifest(
        stage_path,
        expected_physical_sha256=arguments.expected_stage_manifest_sha256,
    )
    expected_stage_keys = {
        "authoritative",
        "code",
        "durable_root",
        "durable_tree_projection",
        "fresh_replay_required_for_corpus_work",
        "gate_minted",
        "killed_ledger",
        "local_root",
        "probe_outputs_reusable",
        "publication_contract",
        "result_classification",
        "runtime",
        "schema",
        "source_family",
        "status",
        "storage_before_provider_remount",
        "success_ledger",
        "top_level_directory_fsync",
    }
    killed = stage.get("killed_ledger")
    success = stage.get("success_ledger")
    if (
        set(stage) != expected_stage_keys
        or stage.get("authoritative") is not False
        or stage.get("gate_minted") is not False
        or stage.get("probe_outputs_reusable") is not False
        or stage.get("fresh_replay_required_for_corpus_work") is not True
        or stage.get("result_classification") != "NO_GATE_MINT"
        or stage.get("source_family") != SOURCE_FAMILY
        or stage.get("durable_root") != os.fspath(durable_root)
        or stage.get("status")
        != "PUBLISHED_AWAITING_PROVIDER_REMOUNT_NO_GATE_MINT"
        or not isinstance(killed, dict)
        or killed.get("status")
        != "NO_GATE_MINT_INCOMPLETE_FRESH_REPLAY_REQUIRED"
        or killed.get("final_ledger_present") is not False
        or killed.get("fresh_replay_required") is not True
        or killed.get("reusable_output") is not False
        or not isinstance(success, dict)
        or success.get("status") != "COMPONENT_PROBE_PASS_NO_GATE_MINT"
        or success.get("legacy_stream_exact_match") is not True
        or success.get("reusable_output") is not False
    ):
        raise DurabilityProbeError("stage manifest safety posture drifted")
    runtime = _runtime_evidence(local_dry_run=arguments.local_dry_run)
    if runtime != stage.get("runtime"):
        raise DurabilityProbeError("verify runtime differs from publish runtime")
    code = _code_evidence()
    if code != stage.get("code"):
        raise DurabilityProbeError("verify code differs from publish code")
    current_storage = _storage_evidence(
        local_dry_run=arguments.local_dry_run,
        phase="verify",
        durable_parent=durable_root.parent,
        mount_root=mount_root,
    )
    prior_storage = stage.get("storage_before_provider_remount")
    if not isinstance(prior_storage, dict):
        raise DurabilityProbeError("stage mount evidence is absent")
    prior_mount_id = prior_storage.get("mount_id")
    current_mount_id = current_storage.get("mount_id")
    if not isinstance(prior_mount_id, str) or not isinstance(current_mount_id, str):
        raise DurabilityProbeError("mount IDs are absent")
    if prior_mount_id == current_mount_id:
        raise DurabilityProbeError(
            "provider remount is unproven because the kernel mount ID is unchanged"
        )
    if arguments.local_dry_run:
        if (
            prior_storage.get("classification")
            != "LOCAL_TMP_SIMULATION_NO_DRIVE_CLAIM"
            or current_storage.get("classification")
            != "LOCAL_TMP_SIMULATION_NO_DRIVE_CLAIM"
        ):
            raise DurabilityProbeError(
                "local remount simulation classification drifted"
            )
    else:
        for key in ("classification", "filesystem_type", "mount_point"):
            if prior_storage.get(key) != current_storage.get(key):
                raise DurabilityProbeError(
                    f"DriveFS mount identity changed unexpectedly across remount: {key}"
                )
    expected_tree = stage.get("durable_tree_projection")
    observed_tree_first = _tree_projection(durable_root)
    observed_tree_second = _tree_projection(durable_root)
    if observed_tree_first != expected_tree or observed_tree_second != expected_tree:
        raise DurabilityProbeError(
            "durable object tree changed across provider remount"
        )
    if _code_evidence() != code:
        raise DurabilityProbeError("probe code changed during verify phase")
    if _runtime_evidence(local_dry_run=arguments.local_dry_run) != runtime:
        raise DurabilityProbeError("runtime identity changed during verify phase")
    storage_after = _storage_evidence(
        local_dry_run=arguments.local_dry_run,
        phase="verify",
        durable_parent=durable_root.parent,
        mount_root=mount_root,
    )
    if storage_after.get("mount_id") != current_mount_id:
        raise DurabilityProbeError("durable mount changed during verify phase")
    artifact_directory_fsync = _source_parse_directory_fsync_v3(
        final_receipt.parent
    )
    if arguments.local_dry_run:
        barrier_classification = "SIMULATED_LOCAL_ONLY"
        barrier_action_claimed: str | None = None
    else:
        barrier_action_claimed = arguments.barrier_kind
        if barrier_action_claimed is None:
            raise DurabilityProbeError(
                "exact verify requires an explicit --barrier-kind"
            )
        barrier_classification = {
            "unflushed-backend-replacement": (
                "UNFLUSHED_BACKEND_REPLACEMENT_OPERATOR_DECLARED_"
                "AND_MOUNT_ID_CHANGED"
            ),
            "explicit-flush-remount": (
                "EXPLICIT_DRIVEFS_FLUSH_REMOUNT_OPERATOR_DECLARED_"
                "AND_MOUNT_ID_CHANGED_WEAKER_CANARY"
            ),
        }[barrier_action_claimed]
    final: dict[str, object] = {
        "authoritative": False,
        "code": code,
        "durable_root": os.fspath(durable_root),
        "durable_tree_projection": observed_tree_second,
        "fresh_replay_required_for_corpus_work": True,
        "gate_minted": False,
        "mount_transition": {
            "barrier_action_operator_declared": barrier_action_claimed,
            "current_mount": storage_after,
            "kernel_mount_id_changed": True,
            "prior_mount": prior_storage,
            "provider_barrier_classification": barrier_classification,
        },
        "probe_outputs_reusable": False,
        "publication_contract": _stage_publication_contract(
            artifact_directory_fsync
        ),
        "result_classification": "NO_GATE_MINT",
        "runtime": runtime,
        "schema": _FINAL_SCHEMA,
        "source_family": SOURCE_FAMILY,
        "stage_manifest": {
            "bytes": len(stage_raw),
            "path": os.fspath(stage_path),
            "physical_reopen_count": 2,
            "sha256": _sha256_bytes(stage_raw),
        },
        "status": "PROVIDER_REMOUNT_VERIFIED_NO_GATE_MINT_NONREUSABLE",
        "verification": {
            "durable_object_reopen_passes": 2,
            "every_durable_object_rehashed": True,
            "stage_projection_exact_match": True,
        },
    }
    final_bytes = _canonical_json_bytes(final)
    envelope = {
        "final_receipt": final,
        "final_payload_sha256": _sha256_bytes(final_bytes),
        "schema": _FINAL_ENVELOPE_SCHEMA,
    }
    publication = _publish_artifact(
        final_receipt,
        envelope,
        expected_directory_fsync=artifact_directory_fsync,
    )
    return {
        "authoritative": False,
        "gate_minted": False,
        "publication": publication,
        "schema": _VERIFY_RESULT_SCHEMA,
        "status": "PROVIDER_REMOUNT_VERIFIED_NO_GATE_MINT_NONREUSABLE",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = _CanonicalArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser(
        "publish", help="run the probe and publish its pre-remount stage"
    )
    publish.add_argument("--durable-root", type=Path, required=True)
    publish.add_argument("--local-root", type=Path, required=True)
    publish.add_argument("--stage-manifest-out", type=Path, required=True)
    publish.add_argument("--durable-mount-root", type=Path)
    publish.add_argument(
        "--local-dry-run",
        action="store_true",
        help="test mechanics locally without exact-runtime or DriveFS authority",
    )
    verify = subparsers.add_parser(
        "verify", help="verify after provider remount and publish final receipt"
    )
    verify.add_argument("--durable-root", type=Path, required=True)
    verify.add_argument("--stage-manifest", type=Path, required=True)
    verify.add_argument(
        "--expected-stage-manifest-sha256",
        required=True,
    )
    verify.add_argument("--final-receipt-out", type=Path, required=True)
    verify.add_argument("--durable-mount-root", type=Path)
    verify.add_argument(
        "--barrier-kind",
        choices=("unflushed-backend-replacement", "explicit-flush-remount"),
        help=(
            "operator-declared external action; mount-ID change and object "
            "rehash are verified independently"
        ),
    )
    verify.add_argument("--local-dry-run", action="store_true")
    child = subparsers.add_parser("_hard-kill-child", help=argparse.SUPPRESS)
    child.add_argument("--final-path", type=Path, required=True)
    child.add_argument("--local-root", type=Path, required=True)
    child.add_argument("--ready-marker", type=Path, required=True)
    child.add_argument("--require-exact-runtime", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        arguments = parser.parse_args(argv)
        if arguments.command == "_hard-kill-child":
            return _run_hard_kill_child(arguments)
        if arguments.command == "publish":
            result = _publish_probe(arguments)
        else:
            result = _verify_probe(arguments)
        sys.stdout.buffer.write(_canonical_json_bytes(result))
        sys.stdout.buffer.flush()
        return 0
    except Exception as error:
        failure = {
            "authoritative": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "gate_minted": False,
            "schema": "weft1_source_parse_drivefs_durability_probe_error_v1",
            "status": "FAIL_CLOSED_NO_GATE_MINT",
        }
        sys.stderr.buffer.write(_canonical_json_bytes(failure))
        sys.stderr.buffer.flush()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
