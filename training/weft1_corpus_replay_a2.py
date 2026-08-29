"""Parent-observed replay verification for WEFT-1 corpus Amendment A2.

The pure contracts in :mod:`training.weft1_corpus_a2` deliberately cannot
prove that a claimed replay ran in another process or that its claimed output
hashes match files on disk.  This module supplies that missing parent-side
boundary.  It launches two Python workers with :class:`subprocess.Popen`,
observes their actual PIDs, injects an offline socket guard, validates canonical
child receipts, and independently rehashes every file in both output roots.

``ParentReplayVerificationV3`` is factory-only.  It is authoritative only when
both the content artifacts and a complete, role-labelled dedup evidence set
replay byte-identically.  A replay without complete dedup evidence remains a
non-gate check even when all content files match.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

from training.weft1_corpus_a2 import (
    A2_DEDUP_SEED,
    A2_MINHASH_BINDING,
    MINHASH_RECALL_JACCARD_LEVELS,
    MinHashRecallAuditV3,
    MinHashSyntheticRecallCellV3,
    execution_authority_v3_bound_sha256,
)
from training.weft1_corpus_streaming_a2 import StreamingDedupDecisionV3
from training.weft1_gtok_contract import canonical_sha256
from training.weft1_strict_io import StrictPathError, assert_no_symlink_ancestors


CHILD_RECEIPT_SCHEMA_V3 = "weft1_corpus_parent_replay_child_receipt_v3"
DEDUP_EVIDENCE_SCHEMA_V3 = "weft1_corpus_parent_dedup_evidence_v3"
PARENT_RECEIPT_SCHEMA_V3 = "weft1_corpus_parent_replay_verification_v3"
CHILD_RECEIPT_FILENAME = "child-receipt.json"
NETWORK_PROBE_RESULT = "python_socket_connect_blocked"
OUTPUT_FILE_ROLES = frozenset({"auxiliary", "content", "dedup_evidence"})
DEDUP_LEDGER_IDENTITY_DOMAIN_V3 = b"weft1_corpus_dedup_decision_ledger_v3"
LINUX_UNSHARE_PATH_V1 = Path("/usr/bin/unshare")
LINUX_UNSHARE_SHA256_V1 = (
    "72a34e6ba98a59f1da0c7b4d8c9722b746b5ade54e4d7e8de8e519c2993858ad"
)
REPOSITORY_ROOT_V3 = Path(__file__).resolve().parents[1]
PRODUCTION_WORKER_PATH_V3 = (
    REPOSITORY_ROOT_V3 / "scripts" / "run_weft1_corpus_materialize_a2.py"
)
PRODUCTION_BINDINGS_PATH_V3 = (
    REPOSITORY_ROOT_V3
    / "training"
    / "weft1_corpus_gtok_a2_bindings_20260828.json"
)
PRODUCTION_DEPENDENCY_LOCK_PATH_V3 = (
    REPOSITORY_ROOT_V3
    / "training"
    / "weft1_corpus_gtok_a2_requirements.lock"
)
PRODUCTION_BINDINGS_SHA256_V3 = (
    "ee10e69a3ccd55f7960949f4c318daa4db1197c779f5e88fb67cec82ab7f263b"
)
PRODUCTION_AUTHORITY_SHA256_V3 = (
    "f7a2655b30f6c699035ec4ffdccee8c03068eeab8da94894be8e5818f955ce02"
)
PRODUCTION_MATERIALIZER_ALGORITHM_VERSION_V3 = 2
PRODUCTION_MATERIALIZER_SCHEMA_V3 = "weft1_corpus_pa_materialization_v3"
PRODUCTION_D1_READY_SCHEMA_V3 = "weft1_corpus_d1_ready_manifest_v3"
PRODUCTION_READINESS_V3 = "AUTHORITATIVE_INPUTS_D1_READY_NO_GATE_MINT"

# A raw boolean on the private reducer made it too easy for an unrelated caller
# to accidentally promote a diagnostic replay to production authority.  The
# token is intentionally module-private and is supplied only by the fixed
# production wrapper after its pins, snapshots, and runtime checks pass.
_PRODUCTION_PROFILE_SENTINEL = object()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_LOGICAL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_RESERVED_ENVIRONMENT_KEYS = frozenset(
    {
        "PYTHONPATH",
        "WEFT1_NETWORK_DISABLED",
        "WEFT1_NETWORK_GUARD_ACTIVE",
        "WEFT1_NETWORK_GUARD_SHA256",
        "WEFT1_REPLAY_INPUT_IDENTITY_SHA256",
        "WEFT1_REPLAY_OUTPUT_ROOT",
        "WEFT1_REPLAY_RECEIPT_PATH",
        "WEFT1_REPLAY_RUN_ID",
        "WEFT1_REPLAY_WORKER_COMPATIBILITY_SHA256",
    }
)
_PROXY_ENVIRONMENT_KEYS = (
    "ALL_PROXY",
    "FTP_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "all_proxy",
    "ftp_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
)
_RESERVED_ENVIRONMENT_KEYS = _RESERVED_ENVIRONMENT_KEYS.union(
    _PROXY_ENVIRONMENT_KEYS
)

_NETWORK_GUARD_SOURCE = b'''\
"""Injected WEFT-1 replay guard: disable Python socket networking."""
import os
import socket

class Weft1NetworkDisabledError(RuntimeError):
    pass

def _weft1_network_refused(*_args, **_kwargs):
    raise Weft1NetworkDisabledError("WEFT-1 parent replay disables network access")

socket.create_connection = _weft1_network_refused
socket.getaddrinfo = _weft1_network_refused
socket.socket.connect = _weft1_network_refused
socket.socket.connect_ex = _weft1_network_refused
socket.socket.sendto = _weft1_network_refused
if hasattr(socket.socket, "sendmsg"):
    socket.socket.sendmsg = _weft1_network_refused
os.environ["WEFT1_NETWORK_GUARD_ACTIVE"] = "1"
'''


class ParentReplayError(RuntimeError):
    """A fail-closed parent replay verification error."""


def _canonical_json_line(value: object) -> bytes:
    """Return strict, sorted, compact, LF-terminated UTF-8 JSON."""

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ParentReplayError(f"{name} must be a lowercase SHA-256")
    return value


def _require_run_id(value: object, name: str = "run_id") -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise ParentReplayError(f"{name} uses invalid canonical syntax")
    return value


def _require_exact_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParentReplayError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ParentReplayError(f"{name} keys must be strings")
    return value


def _reject_noncanonical_json_values(value: object, name: str = "JSON") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ParentReplayError(f"{name} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_noncanonical_json_values(item, f"{name}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ParentReplayError(f"{name} contains a non-string key")
            _reject_noncanonical_json_values(item, f"{name}.{key}")
        return
    raise ParentReplayError(f"{name} contains unsupported type {type(value).__name__}")


def _json_object_no_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ParentReplayError(f"child receipt repeats JSON key {key!r}")
        output[key] = value
    return output


def _read_canonical_json_object(path: Path) -> tuple[Mapping[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ParentReplayError("child receipt must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        decoded = raw.decode("utf-8", errors="strict")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ParentReplayError(
                    f"child receipt uses non-finite JSON constant {constant}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ParentReplayError("child receipt is not strict UTF-8 JSON") from error
    receipt = _require_exact_mapping(parsed, "child receipt")
    _reject_noncanonical_json_values(receipt, "child receipt")
    if raw != _canonical_json_line(receipt):
        raise ParentReplayError("child receipt is not canonical LF-terminated JSON")
    return receipt, _sha256_bytes(raw)


def _parse_canonical_json_line(
    raw: bytes, *, name: str, line_number: int
) -> Mapping[str, Any]:
    if not raw.endswith(b"\n") or raw in {b"", b"\n"}:
        raise ParentReplayError(f"{name} line {line_number} lacks canonical LF framing")
    try:
        parsed = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ParentReplayError(
                    f"{name} line {line_number} uses non-finite JSON {constant}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ParentReplayError(
            f"{name} line {line_number} is not strict UTF-8 JSON"
        ) from error
    row = _require_exact_mapping(parsed, f"{name} line {line_number}")
    _reject_noncanonical_json_values(row, f"{name} line {line_number}")
    if raw != _canonical_json_line(row):
        raise ParentReplayError(f"{name} line {line_number} is not canonical JSONL")
    return row


@dataclass(frozen=True)
class _DedupLedgerReductionV3:
    decision_count: int
    decision_ledger_identity_sha256: str
    dropped_bytes: int
    exact_match_count: int
    exact_match_rate: Fraction
    near_match_count: int
    near_match_rate: Fraction
    query_decision_count: int


def _reduce_decision_ledger(path: Path) -> _DedupLedgerReductionV3:
    digest = hashlib.sha256()
    digest.update(len(DEDUP_LEDGER_IDENTITY_DOMAIN_V3).to_bytes(8, "big"))
    digest.update(DEDUP_LEDGER_IDENTITY_DOMAIN_V3)
    decision_count = 0
    next_source_ordinal = {"dolma_web": 0, "fineweb_edu": 0}
    observed_query = False
    query_count = 0
    exact_count = 0
    near_count = 0
    dropped_bytes = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            row = _parse_canonical_json_line(
                raw, name="dedup decision ledger", line_number=line_number
            )
            try:
                decision = StreamingDedupDecisionV3(**row)
            except (TypeError, ValueError) as error:
                raise ParentReplayError(
                    f"dedup decision ledger line {line_number} is invalid"
                ) from error
            if decision.decision_ordinal != decision_count:
                raise ParentReplayError("dedup decision ordinals are not contiguous")
            expected_source_ordinal = next_source_ordinal[decision.source]
            if decision.source_order_ordinal != expected_source_ordinal:
                raise ParentReplayError("dedup source ordinals are not contiguous")
            if decision.source == "dolma_web":
                if observed_query:
                    raise ParentReplayError("Dolma decision appears after FineWeb")
            else:
                observed_query = True
                query_count += 1
                if decision.action == "DROP_EXACT":
                    exact_count += 1
                    dropped_bytes += decision.retained_byte_count
                elif decision.action == "DROP_NEAR":
                    near_count += 1
                    dropped_bytes += decision.retained_byte_count
            next_source_ordinal[decision.source] += 1
            decision_count += 1
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    if decision_count < 1 or query_count < 1:
        raise ParentReplayError(
            "complete D2 evidence requires canonical and query decisions"
        )
    return _DedupLedgerReductionV3(
        decision_count=decision_count,
        decision_ledger_identity_sha256=digest.hexdigest(),
        dropped_bytes=dropped_bytes,
        exact_match_count=exact_count,
        exact_match_rate=Fraction(exact_count, query_count),
        near_match_count=near_count,
        near_match_rate=Fraction(near_count, query_count),
        query_decision_count=query_count,
    )


def _reduce_selection_ledger(
    path: Path, *, dedup: _DedupLedgerReductionV3
) -> int:
    topup_bytes = 0
    topup_document_ids: set[str] = set()
    exact_count = 0
    near_count = 0
    dropped_bytes = 0
    row_count = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            row_count += 1
            row = _parse_canonical_json_line(
                raw, name="selection decision ledger", line_number=line_number
            )
            action = row.get("action")
            if action == "SELECT_FINEWEB_TOPUP":
                if (
                    row.get("phase") != "TOPUP"
                    or row.get("pool") != "fineweb_edu"
                    or row.get("source") != "fineweb_edu"
                    or row.get("dedup_action") != "KEEP_FINEWEB"
                ):
                    raise ParentReplayError("FineWeb top-up selection row is malformed")
                document_id = _require_sha256(
                    row.get("document_id"), "top-up document identity"
                )
                retained = row.get("retained_byte_count")
                if type(retained) is not int or retained < 1:
                    raise ParentReplayError("top-up retained bytes must be positive")
                if document_id in topup_document_ids:
                    raise ParentReplayError("top-up selection ledger repeats a document")
                topup_document_ids.add(document_id)
                topup_bytes += retained
            if action in {"DROP_EXACT", "DROP_NEAR"}:
                if row.get("dedup_action") != action:
                    raise ParentReplayError("dedup drop event disagrees with its action")
                retained = row.get("retained_byte_count")
                if type(retained) is not int or retained < 0:
                    raise ParentReplayError("dedup drop bytes must be non-negative")
                dropped_bytes += retained
                if action == "DROP_EXACT":
                    exact_count += 1
                else:
                    near_count += 1
    if row_count < 1:
        raise ParentReplayError("selection decision ledger must not be empty")
    if (
        exact_count != dedup.exact_match_count
        or near_count != dedup.near_match_count
        or dropped_bytes != dedup.dropped_bytes
    ):
        raise ParentReplayError(
            "selection decision ledger disagrees with dedup decisions"
        )
    return topup_bytes


def _read_minhash_recall_audit(path: Path) -> tuple[MinHashRecallAuditV3, str]:
    payload, physical_sha256 = _read_canonical_json_object(path)
    expected_keys = {
        "real_candidate_pairs_at_or_above_threshold",
        "real_dolma_document_count",
        "real_exact_pairs_at_or_above_threshold",
        "real_fineweb_document_count",
        "real_sample_identity_sha256",
        "seed",
        "status",
        "synthetic_cells",
    }
    if set(payload) != expected_keys:
        raise ParentReplayError("MinHash recall-audit fields drifted")
    raw_cells = payload.get("synthetic_cells")
    if not isinstance(raw_cells, list):
        raise ParentReplayError("MinHash recall-audit cells must be a list")
    cells: list[MinHashSyntheticRecallCellV3] = []
    for index, raw_cell in enumerate(raw_cells):
        cell = _require_exact_mapping(raw_cell, f"synthetic_cells[{index}]")
        if set(cell) != {"candidate_count", "exact_jaccard", "pair_count"}:
            raise ParentReplayError("MinHash synthetic cell fields drifted")
        try:
            cells.append(
                MinHashSyntheticRecallCellV3(
                    exact_jaccard=_exact_fraction_from_json(
                        cell.get("exact_jaccard"),
                        f"synthetic_cells[{index}].exact_jaccard",
                    ),
                    pair_count=cell.get("pair_count"),
                    candidate_count=cell.get("candidate_count"),
                )
            )
        except (TypeError, ValueError) as error:
            raise ParentReplayError("MinHash synthetic recall cell is invalid") from error
    try:
        audit = MinHashRecallAuditV3(
            seed=payload.get("seed"),
            synthetic_cells=tuple(cells),
            real_sample_identity_sha256=payload.get("real_sample_identity_sha256"),
            real_dolma_document_count=payload.get("real_dolma_document_count"),
            real_fineweb_document_count=payload.get("real_fineweb_document_count"),
            real_exact_pairs_at_or_above_threshold=payload.get(
                "real_exact_pairs_at_or_above_threshold"
            ),
            real_candidate_pairs_at_or_above_threshold=payload.get(
                "real_candidate_pairs_at_or_above_threshold"
            ),
            status=payload.get("status"),
        )
    except (TypeError, ValueError) as error:
        raise ParentReplayError("MinHash recall-audit receipt is invalid") from error
    if audit.seed != A2_DEDUP_SEED or tuple(
        cell.exact_jaccard for cell in audit.synthetic_cells
    ) != MINHASH_RECALL_JACCARD_LEVELS:
        raise ParentReplayError("MinHash recall-audit binding drifted")
    return audit, physical_sha256


def _canonical_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ParentReplayError("output artifact path must be a nonempty string")
    if "\\" in value:
        raise ParentReplayError("output artifact path must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise ParentReplayError("output artifact path is not canonical and relative")
    if value == CHILD_RECEIPT_FILENAME:
        raise ParentReplayError("child receipt may not inventory itself")
    return value


def _logical_file_rows(
    files: Mapping[str, Path], *, name: str
) -> tuple[dict[str, object], ...]:
    if not isinstance(files, Mapping) or not files:
        raise ParentReplayError(f"{name} must contain at least one named file")
    rows: list[dict[str, object]] = []
    for logical_name, raw_path in sorted(files.items()):
        if (
            not isinstance(logical_name, str)
            or _LOGICAL_NAME.fullmatch(logical_name) is None
            or "//" in logical_name
            or "/./" in f"/{logical_name}/"
            or "/../" in f"/{logical_name}/"
        ):
            raise ParentReplayError(f"{name} contains an invalid logical name")
        try:
            absolute = assert_no_symlink_ancestors(Path(raw_path))
        except StrictPathError as error:
            raise ParentReplayError(
                f"{name}.{logical_name} may not traverse a symlink/reparse point"
            ) from error
        path = absolute.resolve(strict=True)
        if not path.is_file():
            raise ParentReplayError(f"{name}.{logical_name} must be a regular file")
        rows.append(
            {
                "bytes": path.stat().st_size,
                "logical_name": logical_name,
                "sha256": _sha256_file(path),
            }
        )
    return tuple(rows)


def _snapshot_governed_file_v3(
    source: Path,
    destination: Path,
    *,
    name: str,
) -> Path:
    """Copy one regular governed input once into a fresh private file.

    Production workers receive the snapshot path, and the parent hashes that
    same snapshot.  This closes the validate-then-reopen gap for caller-supplied
    receipts and the FastText model without changing their registered bytes.
    """

    try:
        lexical_source = assert_no_symlink_ancestors(Path(source))
        lexical_destination = assert_no_symlink_ancestors(Path(destination))
    except StrictPathError as error:
        raise ParentReplayError(
            f"{name} may not traverse a symlink/reparse point"
        ) from error
    if lexical_destination.exists() or lexical_destination.is_symlink():
        raise ParentReplayError(f"{name} snapshot destination must be fresh")
    lexical_destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        assert_no_symlink_ancestors(lexical_destination)
        with lexical_source.open("rb") as opened:
            if not stat.S_ISREG(os.fstat(opened.fileno()).st_mode):
                raise ParentReplayError(f"{name} must be a regular file")
            with lexical_destination.open("xb") as snapshot:
                for chunk in iter(lambda: opened.read(8 * 1024 * 1024), b""):
                    snapshot.write(chunk)
                snapshot.flush()
                os.fsync(snapshot.fileno())
    except ParentReplayError:
        raise
    except (OSError, StrictPathError) as error:
        raise ParentReplayError(f"cannot snapshot {name}: {error}") from error
    return lexical_destination.resolve(strict=True)


def _resolve_python_executable(value: Path) -> Path:
    supplied = os.fspath(value)
    resolved_value = (
        shutil.which(supplied) if not Path(supplied).is_absolute() else supplied
    )
    if resolved_value is None:
        raise ParentReplayError("Python worker executable cannot be resolved")
    resolved = Path(resolved_value).resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ParentReplayError("Python worker executable must be a regular file")
    if not (
        resolved.name.casefold().startswith("python")
        or resolved.name.casefold().startswith("pypy")
    ):
        raise ParentReplayError("worker executable must be a direct Python interpreter")
    return resolved


def _resolve_unshare_executable(value: Path) -> Path:
    if os.name != "posix" or sys.platform != "linux":
        raise ParentReplayError("OS network namespaces require Linux")
    supplied = Path(value)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ParentReplayError("unshare executable must be an absolute non-symlink")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file() or resolved != LINUX_UNSHARE_PATH_V1:
        raise ParentReplayError("network namespace executable must be unshare")
    if _sha256_file(resolved) != LINUX_UNSHARE_SHA256_V1:
        raise ParentReplayError("unshare executable differs from the A2 binding")
    return resolved


def _verify_unshare_network_isolation(
    *, unshare_executable: Path, python_executable: Path
) -> None:
    probe_source = (
        "import socket; s=socket.socket(); s.settimeout(1); "
        "s.connect(('1.1.1.1',53))"
    )
    process = subprocess.run(
        (
            str(unshare_executable),
            "--net",
            "--",
            str(python_executable),
            "-c",
            probe_source,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
        shell=False,
    )
    stderr = process.stderr.decode("utf-8", errors="replace")
    if process.returncode == 0 or not (
        "Network is unreachable" in stderr or "Errno 101" in stderr
    ):
        raise ParentReplayError(
            "unshare --net did not produce parent-observed network isolation"
        )


def _resolved_fresh_roots(first: Path, second: Path) -> tuple[Path, Path]:
    try:
        lexical_roots = (
            assert_no_symlink_ancestors(Path(first)),
            assert_no_symlink_ancestors(Path(second)),
        )
    except StrictPathError as error:
        raise ParentReplayError(
            "replay output roots may not traverse symlinks/reparse points"
        ) from error
    roots = tuple(root.resolve(strict=False) for root in lexical_roots)
    if (
        roots[0] == roots[1]
        or roots[0] in roots[1].parents
        or roots[1] in roots[0].parents
    ):
        raise ParentReplayError("replay output roots must be non-overlapping")
    for root in roots:
        if root.exists() or root.is_symlink():
            raise ParentReplayError("replay output roots must be fresh and absent")
        ancestor = root.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if ancestor.is_symlink() or not ancestor.is_dir():
            raise ParentReplayError("replay output root has an unsafe ancestor")
    return roots


def _offline_environment(
    *,
    guard_directory: Path,
    guard_sha256: str,
    run_id: str,
    output_root: Path,
    input_identity_sha256: str,
    worker_compatibility_sha256: str,
    extra_environment: Mapping[str, str] | None,
) -> dict[str, str]:
    environment = os.environ.copy()
    for key in _PROXY_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    if extra_environment is not None:
        for key, value in extra_environment.items():
            if key in _RESERVED_ENVIRONMENT_KEYS:
                raise ParentReplayError(f"extra environment may not override {key}")
            if not isinstance(key, str) or not isinstance(value, str):
                raise ParentReplayError("extra environment must map strings to strings")
            environment[key] = value
    inherited_pythonpath = environment.get("PYTHONPATH")
    pythonpath_parts = [str(guard_directory)]
    if inherited_pythonpath:
        pythonpath_parts.append(inherited_pythonpath)
    environment.update(
        {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "WEFT1_NETWORK_DISABLED": "1",
            "WEFT1_NETWORK_GUARD_ACTIVE": "0",
            "WEFT1_NETWORK_GUARD_SHA256": guard_sha256,
            "WEFT1_REPLAY_INPUT_IDENTITY_SHA256": input_identity_sha256,
            "WEFT1_REPLAY_OUTPUT_ROOT": str(output_root),
            "WEFT1_REPLAY_RECEIPT_PATH": str(
                output_root / CHILD_RECEIPT_FILENAME
            ),
            "WEFT1_REPLAY_RUN_ID": run_id,
            "WEFT1_REPLAY_WORKER_COMPATIBILITY_SHA256": (
                worker_compatibility_sha256
            ),
        }
    )
    return environment


def _run_worker(
    *,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[int, bytes, bytes]:
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
        )
    except OSError as error:
        raise ParentReplayError(f"replay worker could not launch: {error}") from error
    actual_pid = process.pid
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        process.kill()
        stdout, stderr = process.communicate()
        raise ParentReplayError(
            f"replay worker PID {actual_pid} exceeded its timeout"
        ) from error
    if process.returncode != 0:
        stderr_tail = stderr.decode("utf-8", errors="replace")[-2000:]
        raise ParentReplayError(
            f"replay worker PID {actual_pid} exited {process.returncode}: {stderr_tail}"
        )
    return actual_pid, stdout, stderr


@dataclass(frozen=True)
class _VerifiedChildReplayV3:
    run_id: str
    actual_process_id: int
    output_root: str
    child_receipt_sha256: str
    stdout_sha256: str
    stderr_sha256: str
    output_file_rows: tuple[dict[str, object], ...]
    output_file_projection_sha256: str
    content_projection_sha256: str
    dedup_projection_sha256: str | None
    dedup_evidence_complete: bool
    content_metadata: Mapping[str, object]


def _validate_file_inventory(
    *, output_root: Path, claimed_files: object
) -> tuple[dict[str, object], ...]:
    if not isinstance(claimed_files, list) or not claimed_files:
        raise ParentReplayError("child receipt must inventory every output artifact")
    expected_rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    exact_file_keys = {"bytes", "path", "role", "sha256"}
    for index, raw_row in enumerate(claimed_files):
        row = _require_exact_mapping(raw_row, f"files[{index}]")
        if set(row) != exact_file_keys:
            raise ParentReplayError("child file inventory row has unexpected fields")
        relative = _canonical_relative_path(row.get("path"))
        if relative in seen_paths:
            raise ParentReplayError("child file inventory repeats a path")
        seen_paths.add(relative)
        role = row.get("role")
        if role not in OUTPUT_FILE_ROLES:
            raise ParentReplayError("child file inventory uses an unknown role")
        claimed_bytes = row.get("bytes")
        if type(claimed_bytes) is not int or claimed_bytes < 0:
            raise ParentReplayError("child file byte count must be non-negative")
        claimed_sha256 = _require_sha256(row.get("sha256"), "file SHA-256")

        path = output_root.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise ParentReplayError(
                f"inventoried output is not a regular file: {relative}"
            )
        if path.resolve(strict=True).parent != output_root.joinpath(
            *PurePosixPath(relative).parts[:-1]
        ).resolve(strict=True):
            raise ParentReplayError("inventoried output escapes through a path alias")
        actual_bytes = path.stat().st_size
        actual_sha256 = _sha256_file(path)
        if actual_bytes != claimed_bytes or actual_sha256 != claimed_sha256:
            raise ParentReplayError(
                f"parent rehash differs from child claim for {relative}"
            )
        expected_rows.append(
            {
                "bytes": actual_bytes,
                "path": relative,
                "role": role,
                "sha256": actual_sha256,
            }
        )

    canonical_rows = tuple(sorted(expected_rows, key=lambda row: str(row["path"])))
    if tuple(expected_rows) != canonical_rows:
        raise ParentReplayError("child file inventory is not in canonical path order")

    actual_paths: list[str] = []
    for path in sorted(output_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ParentReplayError("replay output tree may not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ParentReplayError("replay output tree contains a non-regular entry")
        actual_paths.append(path.relative_to(output_root).as_posix())
    if actual_paths != sorted([*seen_paths, CHILD_RECEIPT_FILENAME]):
        raise ParentReplayError(
            "child file inventory does not cover every output file exactly once"
        )
    return canonical_rows


def _exact_fraction_from_json(value: object, name: str) -> Fraction:
    mapping = _require_exact_mapping(value, name)
    if set(mapping) != {"denominator", "numerator"}:
        raise ParentReplayError(f"{name} must use exact Fraction JSON")
    numerator = mapping.get("numerator")
    denominator = mapping.get("denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise ParentReplayError(f"{name} must contain exact integer terms")
    fraction = Fraction(numerator, denominator)
    if not Fraction(0, 1) <= fraction <= Fraction(1, 1):
        raise ParentReplayError(f"{name} must lie in [0, 1]")
    if (numerator, denominator) != (fraction.numerator, fraction.denominator):
        raise ParentReplayError(f"{name} must be reduced to canonical terms")
    return fraction


def _validate_complete_dedup_metadata(
    value: object,
    *,
    output_root: Path,
    dedup_rows: tuple[dict[str, object], ...],
) -> Mapping[str, Any]:
    metadata = _require_exact_mapping(value, "dedup_metadata")
    expected_keys = {
        "binding_identity_sha256",
        "decision_count",
        "decision_ledger_identity_sha256",
        "decision_ledger_path",
        "decision_ledger_sha256",
        "dropped_bytes",
        "exact_match_rate",
        "minhash_recall_audit_path",
        "minhash_recall_audit_receipt_sha256",
        "minhash_recall_audit_sha256",
        "near_match_rate",
        "schema",
        "selection_ledger_path",
        "selection_ledger_sha256",
        "topup_bytes",
    }
    if set(metadata) != expected_keys:
        raise ParentReplayError(
            "complete dedup metadata has missing or unexpected fields"
        )
    if metadata.get("schema") != DEDUP_EVIDENCE_SCHEMA_V3:
        raise ParentReplayError("complete dedup evidence schema drifted")
    if (
        _require_sha256(
            metadata.get("binding_identity_sha256"),
            "dedup binding identity",
        )
        != A2_MINHASH_BINDING.receipt_sha256
    ):
        raise ParentReplayError("dedup evidence uses the wrong A2 MinHash binding")
    bound_artifacts = {
        _canonical_relative_path(metadata.get("decision_ledger_path")): (
            "decision ledger",
            _require_sha256(
                metadata.get("decision_ledger_sha256"),
                "dedup decision ledger SHA-256",
            ),
        ),
        _canonical_relative_path(metadata.get("selection_ledger_path")): (
            "selection ledger",
            _require_sha256(
                metadata.get("selection_ledger_sha256"),
                "selection decision ledger SHA-256",
            ),
        ),
        _canonical_relative_path(metadata.get("minhash_recall_audit_path")): (
            "MinHash recall audit",
            _require_sha256(
                metadata.get("minhash_recall_audit_sha256"),
                "MinHash recall-audit file SHA-256",
            ),
        ),
    }
    if len(bound_artifacts) != 3 or len(dedup_rows) != 3:
        raise ParentReplayError(
            "complete D2 evidence requires exactly three distinct evidence artifacts"
        )
    rows_by_path = {str(row["path"]): row for row in dedup_rows}
    if set(rows_by_path) != set(bound_artifacts):
        raise ParentReplayError("dedup metadata paths differ from evidence inventory")
    for relative, (_, expected_sha256) in bound_artifacts.items():
        if rows_by_path[relative]["sha256"] != expected_sha256:
            raise ParentReplayError(
                "dedup metadata differs from a parent-rehashed evidence artifact"
            )

    ledger_relative = _canonical_relative_path(metadata.get("decision_ledger_path"))
    selection_relative = _canonical_relative_path(metadata.get("selection_ledger_path"))
    recall_relative = _canonical_relative_path(metadata.get("minhash_recall_audit_path"))
    reduction = _reduce_decision_ledger(
        output_root.joinpath(*PurePosixPath(ledger_relative).parts)
    )
    topup_bytes = _reduce_selection_ledger(
        output_root.joinpath(*PurePosixPath(selection_relative).parts),
        dedup=reduction,
    )
    recall_audit, recall_physical_sha256 = _read_minhash_recall_audit(
        output_root.joinpath(*PurePosixPath(recall_relative).parts)
    )

    exact_rate = _exact_fraction_from_json(
        metadata.get("exact_match_rate"), "exact_match_rate"
    )
    near_rate = _exact_fraction_from_json(
        metadata.get("near_match_rate"), "near_match_rate"
    )
    recomputed = {
        "decision_count": reduction.decision_count,
        "decision_ledger_identity_sha256": (
            reduction.decision_ledger_identity_sha256
        ),
        "dropped_bytes": reduction.dropped_bytes,
        "exact_match_rate": reduction.exact_match_rate,
        "minhash_recall_audit_receipt_sha256": recall_audit.receipt_sha256,
        "minhash_recall_audit_sha256": recall_physical_sha256,
        "near_match_rate": reduction.near_match_rate,
        "topup_bytes": topup_bytes,
    }
    observed = {
        "decision_count": metadata.get("decision_count"),
        "decision_ledger_identity_sha256": _require_sha256(
            metadata.get("decision_ledger_identity_sha256"),
            "dedup decision ledger semantic identity",
        ),
        "dropped_bytes": metadata.get("dropped_bytes"),
        "exact_match_rate": exact_rate,
        "minhash_recall_audit_receipt_sha256": _require_sha256(
            metadata.get("minhash_recall_audit_receipt_sha256"),
            "MinHash recall-audit receipt SHA-256",
        ),
        "minhash_recall_audit_sha256": _require_sha256(
            metadata.get("minhash_recall_audit_sha256"),
            "MinHash recall-audit file SHA-256",
        ),
        "near_match_rate": near_rate,
        "topup_bytes": metadata.get("topup_bytes"),
    }
    for name in ("decision_count", "dropped_bytes", "topup_bytes"):
        if type(observed[name]) is not int or observed[name] < 0:
            raise ParentReplayError(f"dedup {name} must be non-negative")
    if observed != recomputed:
        differing = sorted(
            name for name in recomputed if observed.get(name) != recomputed[name]
        )
        raise ParentReplayError(
            "dedup metadata differs from parent-recomputed evidence: "
            + ", ".join(differing)
        )
    return metadata


def _validate_child_receipt(
    *,
    output_root: Path,
    expected_run_id: str,
    actual_process_id: int,
    expected_input_identity_sha256: str,
    expected_worker_compatibility_sha256: str,
    expected_network_guard_sha256: str,
    stdout: bytes,
    stderr: bytes,
) -> _VerifiedChildReplayV3:
    receipt_path = output_root / CHILD_RECEIPT_FILENAME
    receipt, receipt_sha256 = _read_canonical_json_object(receipt_path)
    exact_receipt_keys = {
        "content_metadata",
        "dedup_evidence_complete",
        "dedup_metadata",
        "files",
        "input_identity_sha256",
        "network_disabled",
        "network_guard_active",
        "network_guard_sha256",
        "network_probe",
        "output_root",
        "process_id",
        "run_id",
        "schema",
        "worker_compatibility_sha256",
    }
    if set(receipt) != exact_receipt_keys:
        raise ParentReplayError("child receipt has missing or unexpected fields")
    if receipt.get("schema") != CHILD_RECEIPT_SCHEMA_V3:
        raise ParentReplayError("child receipt schema drifted")
    if _require_run_id(receipt.get("run_id")) != expected_run_id:
        raise ParentReplayError("child receipt run ID differs from parent assignment")
    if receipt.get("process_id") != actual_process_id:
        raise ParentReplayError(
            "child receipt PID differs from parent-observed Popen PID"
        )
    if receipt.get("output_root") != str(output_root):
        raise ParentReplayError(
            "child receipt output root differs from parent assignment"
        )
    if (
        _require_sha256(receipt.get("input_identity_sha256"), "input identity")
        != expected_input_identity_sha256
    ):
        raise ParentReplayError(
            "child receipt input identity differs from parent rehash"
        )
    if (
        _require_sha256(
            receipt.get("worker_compatibility_sha256"),
            "worker compatibility identity",
        )
        != expected_worker_compatibility_sha256
    ):
        raise ParentReplayError("child receipt worker compatibility identity differs")
    if receipt.get("network_disabled") is not True:
        raise ParentReplayError("child did not attest network-disabled execution")
    if receipt.get("network_guard_active") is not True:
        raise ParentReplayError("child did not observe the injected network guard")
    if receipt.get("network_probe") != NETWORK_PROBE_RESULT:
        raise ParentReplayError("child did not complete the blocking network probe")
    if (
        _require_sha256(receipt.get("network_guard_sha256"), "network guard SHA-256")
        != expected_network_guard_sha256
    ):
        raise ParentReplayError("child receipt used a different network guard")

    content_metadata = _require_exact_mapping(
        receipt.get("content_metadata"), "content_metadata"
    )
    _reject_noncanonical_json_values(content_metadata, "content_metadata")
    if not content_metadata:
        raise ParentReplayError("content_metadata must not be empty")
    rows = _validate_file_inventory(
        output_root=output_root, claimed_files=receipt.get("files")
    )
    content_rows = tuple(row for row in rows if row["role"] == "content")
    if not content_rows:
        raise ParentReplayError("replay must contain at least one content artifact")
    content_projection = {
        "files": content_rows,
        "metadata": content_metadata,
    }

    complete = receipt.get("dedup_evidence_complete")
    if type(complete) is not bool:
        raise ParentReplayError("dedup_evidence_complete must be an exact boolean")
    dedup_rows = tuple(row for row in rows if row["role"] == "dedup_evidence")
    dedup_metadata_raw = receipt.get("dedup_metadata")
    dedup_projection_sha256: str | None
    if complete:
        dedup_metadata = _validate_complete_dedup_metadata(
            dedup_metadata_raw,
            output_root=output_root,
            dedup_rows=dedup_rows,
        )
        dedup_projection_sha256 = canonical_sha256(
            {"files": dedup_rows, "metadata": dedup_metadata}
        )
    else:
        if dedup_metadata_raw is not None or dedup_rows:
            raise ParentReplayError(
                "incomplete dedup evidence may not carry gate-shaped evidence"
            )
        dedup_projection_sha256 = None

    return _VerifiedChildReplayV3(
        run_id=expected_run_id,
        actual_process_id=actual_process_id,
        output_root=str(output_root),
        child_receipt_sha256=receipt_sha256,
        stdout_sha256=_sha256_bytes(stdout),
        stderr_sha256=_sha256_bytes(stderr),
        output_file_rows=rows,
        output_file_projection_sha256=canonical_sha256(rows),
        content_projection_sha256=canonical_sha256(content_projection),
        dedup_projection_sha256=dedup_projection_sha256,
        dedup_evidence_complete=complete,
        content_metadata=dict(content_metadata),
    )


def _validate_production_child_profile_v3(
    child: _VerifiedChildReplayV3,
    *,
    expected_environment_identity_sha256: str,
) -> None:
    """Compose a production child claim with parent-rehashed manifests.

    Generic replay workers may use application-specific ``content_metadata``.
    The production profile is narrower: every named identity must be recoverable
    from canonical artifacts that the parent independently inventoried and
    hashed.  This validator is therefore called only behind the production
    sentinel.
    """

    metadata = _require_exact_mapping(child.content_metadata, "content_metadata")
    expected_metadata_keys = {
        "content_identity_sha256",
        "d1_ready_manifest_sha256",
        "environment_identity_sha256",
        "materializer_algorithm_version",
        "source_identity_sha256",
        "tokenizer_fit_input_receipt_sha256",
    }
    if set(metadata) != expected_metadata_keys:
        raise ParentReplayError(
            "production child content metadata has missing or unexpected fields"
        )
    content_identity = _require_sha256(
        metadata.get("content_identity_sha256"), "production content identity"
    )
    d1_physical_sha256 = _require_sha256(
        metadata.get("d1_ready_manifest_sha256"), "D1-ready manifest SHA-256"
    )
    source_identity = _require_sha256(
        metadata.get("source_identity_sha256"), "production source identity"
    )
    tokenizer_fit_identity = _require_sha256(
        metadata.get("tokenizer_fit_input_receipt_sha256"),
        "tokenizer fit-input receipt identity",
    )
    environment_identity = _require_sha256(
        metadata.get("environment_identity_sha256"),
        "production runtime environment identity",
    )
    if environment_identity != expected_environment_identity_sha256:
        raise ParentReplayError(
            "production child runtime identity differs from parent attestation"
        )
    if (
        metadata.get("materializer_algorithm_version")
        != PRODUCTION_MATERIALIZER_ALGORITHM_VERSION_V3
    ):
        raise ParentReplayError("production materializer algorithm version drifted")

    output_root = Path(child.output_root)
    rows_by_path = {str(row["path"]): row for row in child.output_file_rows}
    for required_path in ("content-manifest.json", "d1-ready-manifest.json"):
        row = rows_by_path.get(required_path)
        if row is None or row.get("role") != "content":
            raise ParentReplayError(
                f"production child does not inventory {required_path} as content"
            )

    content, content_physical_sha256 = _read_canonical_json_object(
        output_root / "content-manifest.json"
    )
    d1, observed_d1_physical_sha256 = _read_canonical_json_object(
        output_root / "d1-ready-manifest.json"
    )
    if rows_by_path["content-manifest.json"]["sha256"] != content_physical_sha256:
        raise ParentReplayError("content manifest differs from parent inventory")
    if (
        rows_by_path["d1-ready-manifest.json"]["sha256"]
        != observed_d1_physical_sha256
        or observed_d1_physical_sha256 != d1_physical_sha256
    ):
        raise ParentReplayError("D1-ready manifest differs from child metadata")

    content_payload = dict(content)
    observed_content_identity = content_payload.pop("content_identity_sha256", None)
    recomputed_content_identity = execution_authority_v3_bound_sha256(
        "weft1_corpus_materialized_content_v3", content_payload
    )
    if (
        observed_content_identity != content_identity
        or recomputed_content_identity != content_identity
        or content.get("schema") != PRODUCTION_MATERIALIZER_SCHEMA_V3
        or content.get("mode") != "PRODUCTION"
        or content.get("readiness") != PRODUCTION_READINESS_V3
        or content.get("source_identity_sha256") != source_identity
        or content.get("tokenizer_fit_input_receipt_sha256")
        != tokenizer_fit_identity
    ):
        raise ParentReplayError(
            "production content manifest does not compose with child metadata"
        )

    expected_d1_keys = {
        "content_identity_sha256",
        "d1_ready_identity_sha256",
        "file_inventory",
        "gate_minted",
        "mode",
        "readiness",
        "schema",
        "source_identity_sha256",
    }
    if set(d1) != expected_d1_keys:
        raise ParentReplayError("production D1-ready manifest fields drifted")
    d1_payload = dict(d1)
    claimed_d1_identity = d1_payload.pop("d1_ready_identity_sha256", None)
    recomputed_d1_identity = execution_authority_v3_bound_sha256(
        "weft1_corpus_d1_ready_inventory_v3", d1_payload
    )
    expected_inventory = [
        {
            "bytes": row["bytes"],
            "relative_path": row["path"],
            "sha256": row["sha256"],
        }
        for row in child.output_file_rows
        if row["path"] != "d1-ready-manifest.json"
    ]
    if (
        _require_sha256(claimed_d1_identity, "D1-ready semantic identity")
        != recomputed_d1_identity
        or d1.get("file_inventory") != expected_inventory
        or d1.get("gate_minted") is not False
        or d1.get("mode") != "PRODUCTION"
        or d1.get("readiness") != PRODUCTION_READINESS_V3
        or d1.get("schema") != PRODUCTION_D1_READY_SCHEMA_V3
        or d1.get("content_identity_sha256") != content_identity
        or d1.get("source_identity_sha256") != source_identity
    ):
        raise ParentReplayError(
            "production D1-ready manifest does not compose with parent inventory"
        )


@dataclass(frozen=True, init=False)
class ParentReplayVerificationV3:
    """Factory-only parent evidence; authoritative only with full D1 and D2."""

    schema: str
    status: str
    authoritative: bool
    d1_file_replay_verified: bool
    d2_dedup_replay_verified: bool
    first_run_id: str
    second_run_id: str
    first_process_id: int
    second_process_id: int
    first_output_root: str
    second_output_root: str
    input_identity_sha256: str
    worker_compatibility_sha256: str
    network_guard_sha256: str
    network_isolation_kind: str
    network_isolation_executable_sha256: str | None
    network_isolation_authoritative: bool
    production_profile_verified: bool
    output_file_projection_sha256: str
    content_projection_sha256: str
    dedup_projection_sha256: str | None
    first_child_receipt_sha256: str
    second_child_receipt_sha256: str
    evidence_sha256: str

    def __new__(cls) -> ParentReplayVerificationV3:
        raise TypeError(
            "ParentReplayVerificationV3 is factory-minted after parent verification"
        )

    def __post_init__(self) -> None:
        if self.schema != PARENT_RECEIPT_SCHEMA_V3:
            raise ValueError("parent replay receipt schema drifted")
        if self.authoritative:
            expected = ("PASS", True, True, True)
        else:
            expected = (
                "CHECK_PASS",
                False,
                True,
                self.dedup_projection_sha256 is not None,
            )
        if (
            self.status,
            self.authoritative,
            self.d1_file_replay_verified,
            self.d2_dedup_replay_verified,
        ) != expected:
            raise ValueError("parent replay status and authority are inconsistent")
        _require_run_id(self.first_run_id, "first_run_id")
        _require_run_id(self.second_run_id, "second_run_id")
        if self.first_run_id == self.second_run_id:
            raise ValueError("parent replay requires distinct run IDs")
        if (
            type(self.first_process_id) is not int
            or type(self.second_process_id) is not int
            or self.first_process_id < 1
            or self.second_process_id < 1
            or self.first_process_id == self.second_process_id
        ):
            raise ValueError("parent replay requires distinct positive child PIDs")
        roots = (Path(self.first_output_root), Path(self.second_output_root))
        if (
            not all(root.is_absolute() for root in roots)
            or roots[0] == roots[1]
            or roots[0] in roots[1].parents
            or roots[1] in roots[0].parents
        ):
            raise ValueError("parent replay roots are not absolute and non-overlapping")
        for name in (
            "input_identity_sha256",
            "worker_compatibility_sha256",
            "network_guard_sha256",
            "output_file_projection_sha256",
            "content_projection_sha256",
            "first_child_receipt_sha256",
            "second_child_receipt_sha256",
            "evidence_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.network_isolation_authoritative) is not bool:
            raise ValueError("network isolation authority must be boolean")
        if type(self.production_profile_verified) is not bool:
            raise ValueError("production replay profile status must be boolean")
        if self.network_isolation_authoritative:
            if self.network_isolation_kind != "linux_unshare_net_v1":
                raise ValueError("authoritative replay requires Linux unshare isolation")
            _require_sha256(
                self.network_isolation_executable_sha256,
                "network isolation executable SHA-256",
            )
        elif (
            self.network_isolation_kind != "python_socket_guard_only"
            or self.network_isolation_executable_sha256 is not None
        ):
            raise ValueError("non-authoritative network isolation metadata drifted")
        if self.authoritative != (
            self.network_isolation_authoritative
            and self.production_profile_verified
            and self.dedup_projection_sha256 is not None
        ):
            raise ValueError("replay authority exceeds observed isolation/evidence")
        if self.d2_dedup_replay_verified:
            _require_sha256(self.dedup_projection_sha256, "dedup_projection_sha256")
        elif self.dedup_projection_sha256 is not None:
            raise ValueError("incomplete replay may not carry a D2 projection")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            PARENT_RECEIPT_SCHEMA_V3, self
        )


def _mint_parent_verification_v3(
    *,
    first: _VerifiedChildReplayV3,
    second: _VerifiedChildReplayV3,
    input_identity_sha256: str,
    worker_compatibility_sha256: str,
    network_guard_sha256: str,
    network_isolation_executable_sha256: str | None,
    production_profile_verified: bool,
) -> ParentReplayVerificationV3:
    network_isolation_authoritative = (
        network_isolation_executable_sha256 is not None
    )
    dedup_complete = first.dedup_evidence_complete and second.dedup_evidence_complete
    authoritative = (
        dedup_complete
        and network_isolation_authoritative
        and production_profile_verified
    )
    evidence_payload = {
        "authoritative": authoritative,
        "content_projection_sha256": first.content_projection_sha256,
        "dedup_projection_sha256": first.dedup_projection_sha256,
        "first_child_receipt_sha256": first.child_receipt_sha256,
        "first_process_id": first.actual_process_id,
        "first_run_id": first.run_id,
        "input_identity_sha256": input_identity_sha256,
        "network_guard_sha256": network_guard_sha256,
        "network_isolation_authoritative": network_isolation_authoritative,
        "network_isolation_executable_sha256": (
            network_isolation_executable_sha256
        ),
        "network_isolation_kind": (
            "linux_unshare_net_v1"
            if network_isolation_authoritative
            else "python_socket_guard_only"
        ),
        "production_profile_verified": production_profile_verified,
        "output_file_projection_sha256": first.output_file_projection_sha256,
        "second_child_receipt_sha256": second.child_receipt_sha256,
        "second_process_id": second.actual_process_id,
        "second_run_id": second.run_id,
        "worker_compatibility_sha256": worker_compatibility_sha256,
    }
    receipt = object.__new__(ParentReplayVerificationV3)
    for name, value in {
        "schema": PARENT_RECEIPT_SCHEMA_V3,
        "status": "PASS" if authoritative else "CHECK_PASS",
        "authoritative": authoritative,
        "d1_file_replay_verified": True,
        "d2_dedup_replay_verified": dedup_complete,
        "first_run_id": first.run_id,
        "second_run_id": second.run_id,
        "first_process_id": first.actual_process_id,
        "second_process_id": second.actual_process_id,
        "first_output_root": first.output_root,
        "second_output_root": second.output_root,
        "input_identity_sha256": input_identity_sha256,
        "worker_compatibility_sha256": worker_compatibility_sha256,
        "network_guard_sha256": network_guard_sha256,
        "network_isolation_kind": (
            "linux_unshare_net_v1"
            if network_isolation_authoritative
            else "python_socket_guard_only"
        ),
        "network_isolation_executable_sha256": (
            network_isolation_executable_sha256
        ),
        "network_isolation_authoritative": network_isolation_authoritative,
        "production_profile_verified": production_profile_verified,
        "output_file_projection_sha256": first.output_file_projection_sha256,
        "content_projection_sha256": first.content_projection_sha256,
        "dedup_projection_sha256": first.dedup_projection_sha256,
        "first_child_receipt_sha256": first.child_receipt_sha256,
        "second_child_receipt_sha256": second.child_receipt_sha256,
        "evidence_sha256": execution_authority_v3_bound_sha256(
            "weft1_corpus_parent_replay_evidence_v3", evidence_payload
        ),
    }.items():
        object.__setattr__(receipt, name, value)
    receipt.__post_init__()
    return receipt


def _verify_parent_replays_v3_impl(
    *,
    python_executable: Path,
    worker_arguments: Sequence[str],
    first_output_root: Path,
    second_output_root: Path,
    input_files: Mapping[str, Path],
    compatibility_files: Mapping[str, Path],
    worker_cwd: Path,
    first_run_id: str = "replay-a",
    second_run_id: str = "replay-b",
    timeout_seconds: float = 300.0,
    extra_environment: Mapping[str, str] | None = None,
    network_namespace_executable: Path | None = None,
    production_profile_sentinel: object | None = None,
) -> ParentReplayVerificationV3:
    """Launch and independently verify two offline replay workers.

    Workers receive all run-local assignments through the ``WEFT1_REPLAY_*``
    environment variables.  Each worker must write canonical
    ``child-receipt.json`` using ``CHILD_RECEIPT_SCHEMA_V3``.  The parent, not
    the worker, computes the input and compatibility identities and rehashes
    every output artifact after the worker exits.
    """

    executable = _resolve_python_executable(python_executable)
    if (
        production_profile_sentinel is not None
        and production_profile_sentinel is not _PRODUCTION_PROFILE_SENTINEL
    ):
        raise ParentReplayError("unknown production replay authority token")
    production_profile_verified = (
        production_profile_sentinel is _PRODUCTION_PROFILE_SENTINEL
    )
    unshare_executable = (
        None
        if network_namespace_executable is None
        else _resolve_unshare_executable(network_namespace_executable)
    )
    if unshare_executable is not None:
        _verify_unshare_network_isolation(
            unshare_executable=unshare_executable,
            python_executable=executable,
        )
    if not isinstance(worker_arguments, Sequence) or isinstance(
        worker_arguments, (str, bytes)
    ):
        raise ParentReplayError("worker_arguments must be a sequence of strings")
    arguments = tuple(worker_arguments)
    if any(not isinstance(argument, str) or not argument for argument in arguments):
        raise ParentReplayError("worker arguments must be nonempty exact strings")
    if any(argument in {"-E", "-I", "-S"} for argument in arguments):
        raise ParentReplayError("worker arguments may not bypass the network guard")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise ParentReplayError("timeout_seconds must be finite and positive")
    cwd = Path(worker_cwd).resolve(strict=True)
    if cwd.is_symlink() or not cwd.is_dir():
        raise ParentReplayError("worker_cwd must be a regular directory")
    run_ids = (_require_run_id(first_run_id), _require_run_id(second_run_id))
    if run_ids[0] == run_ids[1]:
        raise ParentReplayError("replay run IDs must be distinct")
    roots = _resolved_fresh_roots(first_output_root, second_output_root)

    input_rows = _logical_file_rows(input_files, name="input_files")
    compatibility_rows = _logical_file_rows(
        compatibility_files, name="compatibility_files"
    )
    input_identity_sha256 = execution_authority_v3_bound_sha256(
        "weft1_corpus_parent_replay_inputs_v3", input_rows
    )
    executable_sha256 = _sha256_file(executable)
    expected_environment_identity_sha256: str | None = None
    if production_profile_verified:
        if unshare_executable is None:
            raise ParentReplayError(
                "production replay profile requires Linux unshare isolation"
            )
        try:
            from training.weft1_corpus_pa import attest_runtime_v3

            runtime = attest_runtime_v3(
                requirements_lock=Path(input_files["dependency_lock"]),
                executable=executable,
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise ParentReplayError(
                f"production parent runtime attestation failed: {error}"
            ) from error
        if runtime.executable_sha256 != executable_sha256:
            raise ParentReplayError(
                "production parent runtime used a different Python executable"
            )
        expected_environment_identity_sha256 = (
            runtime.environment_identity_sha256
        )
    network_isolation_executable_sha256 = (
        None if unshare_executable is None else _sha256_file(unshare_executable)
    )
    worker_compatibility_sha256 = execution_authority_v3_bound_sha256(
        "weft1_corpus_parent_replay_worker_compatibility_v3",
        {
            "arguments": arguments,
            "compatibility_files": compatibility_rows,
            "python_executable_sha256": executable_sha256,
            "network_isolation_executable_sha256": (
                network_isolation_executable_sha256
            ),
            "network_isolation_kind": (
                "linux_unshare_net_v1"
                if unshare_executable is not None
                else "python_socket_guard_only"
            ),
        },
    )
    guard_sha256 = _sha256_bytes(_NETWORK_GUARD_SOURCE)

    children: list[_VerifiedChildReplayV3] = []
    with tempfile.TemporaryDirectory(prefix="weft1-replay-network-guard-") as raw_guard:
        guard_directory = Path(raw_guard).resolve(strict=True)
        guard_path = guard_directory / "sitecustomize.py"
        guard_path.write_bytes(_NETWORK_GUARD_SOURCE)
        if _sha256_file(guard_path) != guard_sha256:
            raise ParentReplayError("network guard failed its parent-side byte check")

        for run_id, output_root in zip(run_ids, roots, strict=True):
            environment = _offline_environment(
                guard_directory=guard_directory,
                guard_sha256=guard_sha256,
                run_id=run_id,
                output_root=output_root,
                input_identity_sha256=input_identity_sha256,
                worker_compatibility_sha256=worker_compatibility_sha256,
                extra_environment=extra_environment,
            )
            actual_pid, stdout, stderr = _run_worker(
                command=(
                    (
                        str(unshare_executable),
                        "--net",
                        "--",
                        str(executable),
                        *arguments,
                    )
                    if unshare_executable is not None
                    else (str(executable), *arguments)
                ),
                cwd=cwd,
                environment=environment,
                timeout_seconds=float(timeout_seconds),
            )
            if not output_root.is_dir() or output_root.is_symlink():
                raise ParentReplayError(
                    "worker did not create its assigned output root"
                )
            child = _validate_child_receipt(
                output_root=output_root,
                expected_run_id=run_id,
                actual_process_id=actual_pid,
                expected_input_identity_sha256=input_identity_sha256,
                expected_worker_compatibility_sha256=worker_compatibility_sha256,
                expected_network_guard_sha256=guard_sha256,
                stdout=stdout,
                stderr=stderr,
            )
            if production_profile_verified:
                assert expected_environment_identity_sha256 is not None
                _validate_production_child_profile_v3(
                    child,
                    expected_environment_identity_sha256=(
                        expected_environment_identity_sha256
                    ),
                )
            children.append(child)

    first, second = children
    if first.actual_process_id == second.actual_process_id:
        raise ParentReplayError("Popen returned the same PID for both replay workers")
    if first.output_file_rows != second.output_file_rows:
        raise ParentReplayError("D1 failed: parent-rehashed output files differ")
    if first.output_file_projection_sha256 != second.output_file_projection_sha256:
        raise ParentReplayError("D1 failed: output file projections differ")
    if first.content_projection_sha256 != second.content_projection_sha256:
        raise ParentReplayError("D1 failed: content projections differ")
    if first.dedup_evidence_complete != second.dedup_evidence_complete:
        raise ParentReplayError("D2 failed: dedup completeness differs by replay")
    if first.dedup_projection_sha256 != second.dedup_projection_sha256:
        raise ParentReplayError("D2 failed: dedup evidence projections differ")
    if input_rows != _logical_file_rows(input_files, name="input_files"):
        raise ParentReplayError("parent replay inputs changed during execution")
    if compatibility_rows != _logical_file_rows(
        compatibility_files, name="compatibility_files"
    ):
        raise ParentReplayError("worker compatibility files changed during execution")
    for child in (first, second):
        final_rows = _validate_file_inventory(
            output_root=Path(child.output_root),
            claimed_files=list(child.output_file_rows),
        )
        if final_rows != child.output_file_rows:
            raise ParentReplayError("replay outputs changed before parent minting")
    return _mint_parent_verification_v3(
        first=first,
        second=second,
        input_identity_sha256=input_identity_sha256,
        worker_compatibility_sha256=worker_compatibility_sha256,
        network_guard_sha256=guard_sha256,
        network_isolation_executable_sha256=(
            network_isolation_executable_sha256
        ),
        production_profile_verified=production_profile_verified,
    )


def verify_parent_replays_v3(
    *,
    python_executable: Path,
    worker_arguments: Sequence[str],
    first_output_root: Path,
    second_output_root: Path,
    input_files: Mapping[str, Path],
    compatibility_files: Mapping[str, Path],
    worker_cwd: Path,
    first_run_id: str = "replay-a",
    second_run_id: str = "replay-b",
    timeout_seconds: float = 300.0,
    extra_environment: Mapping[str, str] | None = None,
    network_namespace_executable: Path | None = None,
) -> ParentReplayVerificationV3:
    """Verify arbitrary replay workers without minting production authority.

    This public diagnostic surface can prove byte-identical D1/D2 behavior, but
    it deliberately cannot assert that the worker implements the governed P-A
    production profile.  Only the dedicated production-profile wrapper may
    call the internal reducer with that additional fact established.
    """

    return _verify_parent_replays_v3_impl(
        python_executable=python_executable,
        worker_arguments=worker_arguments,
        first_output_root=first_output_root,
        second_output_root=second_output_root,
        input_files=input_files,
        compatibility_files=compatibility_files,
        worker_cwd=worker_cwd,
        first_run_id=first_run_id,
        second_run_id=second_run_id,
        timeout_seconds=timeout_seconds,
        extra_environment=extra_environment,
        network_namespace_executable=network_namespace_executable,
        production_profile_sentinel=None,
    )


def verify_production_materialization_replays_v3(
    *,
    python_executable: Path,
    authority_path: Path,
    enumeration_receipt_path: Path,
    cache_download_receipt_path: Path,
    source_manifest_path: Path,
    cache_root: Path,
    fasttext_model_path: Path,
    first_output_root: Path,
    second_output_root: Path,
    first_run_id: str = "production-replay-a",
    second_run_id: str = "production-replay-b",
    timeout_seconds: float = 86_400.0,
) -> ParentReplayVerificationV3:
    """Run the one governed offline P-A worker twice under Linux isolation.

    Unlike :func:`verify_parent_replays_v3`, this surface does not accept an
    arbitrary command, compatibility set, route ledger, dependency lock, or
    network-isolation executable.  Those values are fixed here so the
    production-profile bit can be established by construction rather than by
    a caller assertion.
    """

    from training.weft1_corpus_a2 import A2_LANGUAGE_ID_BINDING
    from training.weft1_corpus_pa import DEFAULT_REQUIREMENTS_LOCK_SHA256
    from training.weft1_corpus_sources_a2 import (
        SOURCE_ROUTE_MANIFEST_PATH,
        SOURCE_ROUTE_MANIFEST_SHA256,
    )

    if _resolve_python_executable(python_executable) != _resolve_python_executable(
        Path(sys.executable)
    ):
        raise ParentReplayError(
            "production replay must use the currently attested Python interpreter"
        )

    fixed_files = {
        "bindings": PRODUCTION_BINDINGS_PATH_V3,
        "dependency_lock": PRODUCTION_DEPENDENCY_LOCK_PATH_V3,
        "route_manifest": SOURCE_ROUTE_MANIFEST_PATH,
        "worker": PRODUCTION_WORKER_PATH_V3,
    }
    fixed_rows = _logical_file_rows(fixed_files, name="production_fixed_files")
    fixed_hashes = {str(row["logical_name"]): str(row["sha256"]) for row in fixed_rows}
    expected_hashes = {
        "bindings": PRODUCTION_BINDINGS_SHA256_V3,
        "dependency_lock": DEFAULT_REQUIREMENTS_LOCK_SHA256,
        "route_manifest": SOURCE_ROUTE_MANIFEST_SHA256,
    }
    if any(fixed_hashes[name] != expected for name, expected in expected_hashes.items()):
        raise ParentReplayError("governed production worker inputs differ from their pins")

    try:
        cache_lexical = assert_no_symlink_ancestors(Path(cache_root))
    except StrictPathError as error:
        raise ParentReplayError(
            "production source cache may not traverse a symlink/reparse point"
        ) from error
    cache_resolved = cache_lexical.resolve(strict=True)
    if not cache_resolved.is_dir():
        raise ParentReplayError("production source cache must be a real directory")

    worker = PRODUCTION_WORKER_PATH_V3.resolve(strict=True)
    compatibility_files = {
        "corpus_contracts": REPOSITORY_ROOT_V3 / "training" / "weft1_corpus_a2.py",
        "enumeration": REPOSITORY_ROOT_V3 / "training" / "weft1_corpus_enumeration_a2.py",
        "gtok_a1_contract": REPOSITORY_ROOT_V3 / "training" / "weft1_gtok_a1_contract.py",
        "gtok_contract": REPOSITORY_ROOT_V3 / "training" / "weft1_gtok_contract.py",
        "materializer": REPOSITORY_ROOT_V3 / "training" / "weft1_corpus_materialize_a2.py",
        "models_init": REPOSITORY_ROOT_V3 / "models" / "__init__.py",
        "parent_replay": Path(__file__).resolve(strict=True),
        "production_io": REPOSITORY_ROOT_V3 / "training" / "weft1_corpus_pa.py",
        "source_io": REPOSITORY_ROOT_V3 / "training" / "weft1_corpus_source_io_a2.py",
        "source_routes": REPOSITORY_ROOT_V3 / "training" / "weft1_corpus_sources_a2.py",
        "streaming_dedup": REPOSITORY_ROOT_V3 / "training" / "weft1_corpus_streaming_a2.py",
        "strict_io": REPOSITORY_ROOT_V3 / "training" / "weft1_strict_io.py",
        "tokenizer": REPOSITORY_ROOT_V3 / "training" / "weft1_gtok_tokenizer_a2.py",
        "training_init": REPOSITORY_ROOT_V3 / "training" / "__init__.py",
        "worker": worker,
    }
    ablation_root = REPOSITORY_ROOT_V3 / "models" / "ablation_lm"
    compatibility_files.update(
        {
            f"ablation_lm/{path.name}": path
            for path in sorted(ablation_root.glob("*.py"), key=lambda item: item.name)
        }
    )

    with tempfile.TemporaryDirectory(prefix="weft1-production-inputs-") as raw_snapshot:
        snapshot_root = Path(raw_snapshot).resolve(strict=True)
        snapshots = {
            "authority": _snapshot_governed_file_v3(
                Path(authority_path), snapshot_root / "authority.md", name="A2 authority"
            ),
            "bindings": _snapshot_governed_file_v3(
                PRODUCTION_BINDINGS_PATH_V3,
                snapshot_root / "bindings.json",
                name="A2 bindings",
            ),
            "cache_download": _snapshot_governed_file_v3(
                Path(cache_download_receipt_path),
                snapshot_root / "cache-download-receipt.json",
                name="cache download receipt",
            ),
            "dependency_lock": _snapshot_governed_file_v3(
                PRODUCTION_DEPENDENCY_LOCK_PATH_V3,
                snapshot_root / "requirements.lock",
                name="dependency lock",
            ),
            "enumeration": _snapshot_governed_file_v3(
                Path(enumeration_receipt_path),
                snapshot_root / "enumeration-receipt.json",
                name="enumeration receipt",
            ),
            "fasttext_model": _snapshot_governed_file_v3(
                Path(fasttext_model_path),
                snapshot_root / "lid.176.bin",
                name="FastText model",
            ),
            "route_manifest": _snapshot_governed_file_v3(
                SOURCE_ROUTE_MANIFEST_PATH,
                snapshot_root / "source-routes.json",
                name="source route manifest",
            ),
            "source_manifest": _snapshot_governed_file_v3(
                Path(source_manifest_path),
                snapshot_root / "source-manifest.json",
                name="source manifest",
            ),
        }
        observed_inputs = _logical_file_rows(
            {
                "authority": snapshots["authority"],
                "fasttext_model": snapshots["fasttext_model"],
            },
            name="production_observed_inputs",
        )
        observed_hashes = {
            str(row["logical_name"]): (int(row["bytes"]), str(row["sha256"]))
            for row in observed_inputs
        }
        if observed_hashes["authority"][1] != PRODUCTION_AUTHORITY_SHA256_V3:
            raise ParentReplayError(
                "A2 authority artifact differs from its ratified hash"
            )
        if observed_hashes["fasttext_model"] != (
            A2_LANGUAGE_ID_BINDING.model_bytes,
            A2_LANGUAGE_ID_BINDING.model_sha256,
        ):
            raise ParentReplayError("FastText model differs from the A2 binding")

        arguments = (
            str(worker),
            "--enumeration-receipt",
            str(snapshots["enumeration"]),
            "--cache-download-receipt",
            str(snapshots["cache_download"]),
            "--source-manifest",
            str(snapshots["source_manifest"]),
            "--cache-root",
            str(cache_resolved),
            "--fasttext-model",
            str(snapshots["fasttext_model"]),
            "--route-manifest",
            str(snapshots["route_manifest"]),
        )
        input_files = {
            "a2_authority": snapshots["authority"],
            "a2_bindings": snapshots["bindings"],
            "cache_download_receipt": snapshots["cache_download"],
            "dependency_lock": snapshots["dependency_lock"],
            "enumeration_receipt": snapshots["enumeration"],
            "fasttext_model": snapshots["fasttext_model"],
            "route_manifest": snapshots["route_manifest"],
            "source_manifest": snapshots["source_manifest"],
        }
        return _verify_parent_replays_v3_impl(
            python_executable=python_executable,
            worker_arguments=arguments,
            first_output_root=first_output_root,
            second_output_root=second_output_root,
            input_files=input_files,
            compatibility_files=compatibility_files,
            worker_cwd=REPOSITORY_ROOT_V3,
            first_run_id=first_run_id,
            second_run_id=second_run_id,
            timeout_seconds=timeout_seconds,
            extra_environment=None,
            network_namespace_executable=LINUX_UNSHARE_PATH_V1,
            production_profile_sentinel=_PRODUCTION_PROFILE_SENTINEL,
        )


__all__ = [
    "CHILD_RECEIPT_FILENAME",
    "CHILD_RECEIPT_SCHEMA_V3",
    "DEDUP_EVIDENCE_SCHEMA_V3",
    "DEDUP_LEDGER_IDENTITY_DOMAIN_V3",
    "LINUX_UNSHARE_PATH_V1",
    "LINUX_UNSHARE_SHA256_V1",
    "NETWORK_PROBE_RESULT",
    "PARENT_RECEIPT_SCHEMA_V3",
    "ParentReplayError",
    "ParentReplayVerificationV3",
    "verify_production_materialization_replays_v3",
    "verify_parent_replays_v3",
]
