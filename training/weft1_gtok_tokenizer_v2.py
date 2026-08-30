"""Production tokenizer fitting for the forward-only WEFT-1 G-TOK screen.

The corpus is consumed only through the V4 physical ``T`` iterator.  A
production arm is fitted in two fresh subprocesses and two disjoint output
roots; the parent reopens and rehashes both artifacts before it can construct a
``TokenizerArmReceiptV2``.  The model-training layer never fits or mutates a
tokenizer.

This module deliberately keeps paths out of scientific identities.  Paths are
process evidence, while tokenizer.json, the physical T stream, the parent
corpus manifest, and the literal recipe are the identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from tokenizers import Tokenizer

from training.weft1_corpus_materialize_a3 import (
    D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4,
    FULL_SHARD_MANIFEST_RELATIVE_PATH_V4,
    SCREEN_SUBMANIFEST_RELATIVE_PATH_V4,
    iter_materialized_tokenizer_fit_texts_v4,
    validate_physical_d6_evidence_v4,
)
from training.weft1_gtok_contract import (
    GTOK_PRETOKENIZER_REGEX,
    GTOK_VOCABULARY_ARMS,
    canonical_json_bytes,
)
from training.weft1_gtok_offline_v2 import (
    OFFLINE_RECEIPT_ENV_V2,
    assert_offline_descendant_v2,
    load_offline_parent_receipt_v2,
)
from training.weft1_gtok_tokenizer_a2 import (
    TOKENIZERS_VERSION,
    atomic_write_tokenizer,
    fit_a2_tokenizer,
    preflight_bpe_i32_counts,
    special_token_strings,
    tokenizer_artifact_sha256,
    tokenizer_inventory_sha256,
    tokenizer_merges_sha256,
    validate_tokenizer_json,
)
from training.weft1_gtok_v2_contract import (
    TokenizerArmReceiptV2,
    gtok_v2_bound_sha256,
)
from training.weft1_strict_io import assert_no_symlink_ancestors


FIT_WORKER_SCHEMA_V2 = "weft1_gtok_v2_tokenizer_fit_worker"
DOUBLE_FIT_SCHEMA_V2 = "weft1_gtok_v2_tokenizer_double_fit"
BYTE_ROUND_TRIP_SCHEMA_V2 = "weft1_gtok_v2_tokenizer_byte_round_trip"
TOKENIZER_FILENAME_V2 = "tokenizer.json"
WORKER_RECEIPT_FILENAME_V2 = "fit-worker-receipt.json"
_HEX = frozenset("0123456789abcdef")


class GTokTokenizerV2Error(RuntimeError):
    """A tokenizer production invariant failed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    pairs_seen: list[str] = []

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GTokTokenizerV2Error(f"JSON repeats key {key!r}")
            result[key] = value
            pairs_seen.append(key)
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GTokTokenizerV2Error(f"cannot load canonical JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise GTokTokenizerV2Error(f"JSON root must be an object: {path}")
    if path.read_bytes() != canonical_json_bytes(value) + b"\n":
        raise GTokTokenizerV2Error(f"JSON is not canonical newline-terminated bytes: {path}")
    return value


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite tokenizer evidence: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(payload)


def _resolve_new_root(path: Path) -> Path:
    lexical = assert_no_symlink_ancestors(path)
    lexical.mkdir(parents=True, exist_ok=False)
    return assert_no_symlink_ancestors(lexical).resolve(strict=True)


def _corpus_fit_binding(root: Path, *, sqlite_path: Path) -> dict[str, Any]:
    resolved = assert_no_symlink_ancestors(root).resolve(strict=True)
    evidence, evidence_file_sha256 = validate_physical_d6_evidence_v4(
        root=resolved,
        sqlite_path=sqlite_path,
    )
    fit = evidence.get("tokenizer_fit_input")
    if not isinstance(fit, Mapping):
        raise GTokTokenizerV2Error("V4 physical evidence has no tokenizer-fit input")
    screen = _load_json(resolved / SCREEN_SUBMANIFEST_RELATIVE_PATH_V4)
    full_path = resolved / FULL_SHARD_MANIFEST_RELATIVE_PATH_V4
    d6_path = resolved / D6_PHYSICAL_EVIDENCE_RELATIVE_PATH_V4
    full_sha256 = _sha256_file(full_path)
    if screen.get("full_manifest_sha256") != full_sha256:
        raise GTokTokenizerV2Error("screen submanifest does not join the full manifest")
    if screen.get("d6_physical_evidence_sha256") != evidence_file_sha256:
        raise GTokTokenizerV2Error("screen submanifest does not join physical D6 evidence")
    if _sha256_file(d6_path) != evidence_file_sha256:
        raise GTokTokenizerV2Error("stored physical D6 evidence changed after validation")
    expected = {
        "document_count": fit.get("document_count"),
        "fit_stream_sha256": fit.get("fit_text_stream_sha256"),
        "full_corpus_manifest_sha256": full_sha256,
        "physical_d6_evidence_sha256": evidence_file_sha256,
        "retained_text_bytes": fit.get("retained_text_bytes"),
        "screen_submanifest_sha256": _sha256_file(
            resolved / SCREEN_SUBMANIFEST_RELATIVE_PATH_V4
        ),
        "tokenizer_fit_input_receipt_sha256": fit.get("receipt_sha256"),
    }
    if (
        type(expected["document_count"]) is not int
        or expected["document_count"] < 1
        or type(expected["retained_text_bytes"]) is not int
        or expected["retained_text_bytes"] < 1
    ):
        raise GTokTokenizerV2Error("V4 tokenizer-fit counts are invalid")
    for name in (
        "fit_stream_sha256",
        "full_corpus_manifest_sha256",
        "physical_d6_evidence_sha256",
        "screen_submanifest_sha256",
        "tokenizer_fit_input_receipt_sha256",
    ):
        _require_sha256(str(expected[name]), name)
    return expected


def _reserved_inventory_sha256() -> str:
    return _sha256_bytes(
        canonical_json_bytes(tuple(enumerate(special_token_strings())))
    )


def _pretokenizer_regex_sha256() -> str:
    return _sha256_bytes(GTOK_PRETOKENIZER_REGEX.encode("utf-8"))


def _round_trip_fixtures() -> tuple[str, ...]:
    # Exact strings, not labels, are hashed into the receipt.  These cover every
    # byte class representable by a valid UTF-8 corpus plus boundary-sensitive
    # whitespace and the literal registered protocol inventory.
    return (
        "plain ASCII 0123456789 punctuation !?",
        "nul:\x00:end",
        "line-one\nline-two\r\nline-three\rline-four",
        "combining:e\u0301 precomposed:\u00e9",
        "Greek:\u03bb Cyrillic:\u0416 Arabic:\u0645 CJK:\u6f22",
        "emoji:\U0001f9f6\u200d\U0001f52c variation:\u2764\ufe0f",
        "\t leading and trailing \u00a0\u2003 ",
        "".join(special_token_strings()),
    )


def tokenizer_byte_round_trip_receipt_v2(payload: bytes) -> dict[str, Any]:
    """Reopen one artifact and prove deterministic text-byte round trips."""

    tokenizer = Tokenizer.from_str(payload.decode("utf-8", errors="strict"))
    rows: list[dict[str, Any]] = []
    for text in _round_trip_fixtures():
        encoded = text.encode("utf-8", errors="strict")
        token_ids = tokenizer.encode(text, add_special_tokens=True).ids
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        decoded_bytes = decoded.encode("utf-8", errors="strict")
        if decoded_bytes != encoded:
            raise GTokTokenizerV2Error("tokenizer byte round-trip mismatch")
        rows.append(
            {
                "decoded_sha256": _sha256_bytes(decoded_bytes),
                "input_sha256": _sha256_bytes(encoded),
                "token_count": len(token_ids),
                "token_ids_sha256": _sha256_bytes(canonical_json_bytes(tuple(token_ids))),
            }
        )
    core = {
        "artifact_sha256": tokenizer_artifact_sha256(payload),
        "fixture_rows": tuple(rows),
        "fixture_set_sha256": _sha256_bytes(
            canonical_json_bytes(tuple(_round_trip_fixtures()))
        ),
        "status": "EXACT_UTF8_BYTES_ROUND_TRIP_PASS",
    }
    return {**core, "receipt_sha256": gtok_v2_bound_sha256(BYTE_ROUND_TRIP_SCHEMA_V2, core)}


@dataclass(frozen=True)
class FitWorkerReceiptV2:
    process_id: int
    output_root: str
    vocab_size: int
    tokenizer_json_sha256: str
    merges_sha256: str
    token_inventory_sha256: str
    reserved_inventory_sha256: str
    pretokenizer_regex_sha256: str
    fit_stream_sha256: str
    full_corpus_manifest_sha256: str
    screen_submanifest_sha256: str
    physical_d6_evidence_sha256: str
    tokenizer_fit_input_receipt_sha256: str
    bpe_safety_receipt_sha256: str
    byte_round_trip_receipt_sha256: str
    executable_sha256: str
    dependency_lock_sha256: str
    environment_identity_sha256: str
    runtime_attestation_receipt_sha256: str
    offline_network_receipt_sha256: str
    offline_network_policy_sha256: str
    tokenizers_version: str

    def __post_init__(self) -> None:
        if type(self.process_id) is not int or self.process_id < 1:
            raise ValueError("fit worker process ID must be positive")
        root = Path(self.output_root)
        if not root.is_absolute() or str(root.resolve(strict=False)) != self.output_root:
            raise ValueError("fit worker output root must be absolute and resolved")
        if self.vocab_size not in GTOK_VOCABULARY_ARMS and not 320 <= self.vocab_size <= 4096:
            raise ValueError("fit worker vocabulary is neither production nor a small fixture")
        for name in (
            "tokenizer_json_sha256",
            "merges_sha256",
            "token_inventory_sha256",
            "reserved_inventory_sha256",
            "pretokenizer_regex_sha256",
            "fit_stream_sha256",
            "full_corpus_manifest_sha256",
            "screen_submanifest_sha256",
            "physical_d6_evidence_sha256",
            "tokenizer_fit_input_receipt_sha256",
            "bpe_safety_receipt_sha256",
            "byte_round_trip_receipt_sha256",
            "executable_sha256",
            "dependency_lock_sha256",
            "environment_identity_sha256",
            "runtime_attestation_receipt_sha256",
            "offline_network_receipt_sha256",
            "offline_network_policy_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.tokenizers_version != TOKENIZERS_VERSION:
            raise ValueError("fit worker tokenizers version drifted")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256(FIT_WORKER_SCHEMA_V2, self)


def _runtime_identity(*, dependency_lock_path: Path) -> tuple[str, str, str, str]:
    # Reuse P-A's exact interpreter, distribution-inventory, SQLite/linkage, and
    # deterministic-environment attestation.  A tokenizer identity produced by
    # a merely version-compatible ambient Colab interpreter is not authoritative.
    from training.weft1_corpus_pa import attest_runtime_v3

    attestation = attest_runtime_v3(requirements_lock=dependency_lock_path)
    receipt_sha256 = gtok_v2_bound_sha256(
        "weft1_gtok_v2_tokenizer_runtime_attestation",
        {
            "dependency_lock_sha256": attestation.dependency_lock_sha256,
            "environment_identity_sha256": attestation.environment_identity_sha256,
            "environment_payload": attestation.environment_payload,
            "executable_sha256": attestation.executable_sha256,
        },
    )
    return (
        attestation.executable_sha256,
        attestation.dependency_lock_sha256,
        attestation.environment_identity_sha256,
        receipt_sha256,
    )


def run_tokenizer_fit_worker_v2(
    *,
    corpus_root: Path,
    output_root: Path,
    vocab_size: int,
    dependency_lock_path: Path,
    offline_network_receipt_path: Path,
    offline_network_receipt_sha256: str,
    offline_network_policy_sha256: str,
) -> FitWorkerReceiptV2:
    """Fit one arm once in an exclusive worker root."""

    observed_offline_sha256 = assert_offline_descendant_v2(
        offline_network_receipt_path
    )
    if observed_offline_sha256 != _require_sha256(
        offline_network_receipt_sha256,
        "offline_network_receipt_sha256",
    ):
        raise GTokTokenizerV2Error(
            "fit worker offline receipt differs from its parent binding"
        )
    offline_parent_receipt, reloaded_offline_sha256 = load_offline_parent_receipt_v2(
        offline_network_receipt_path
    )
    if (
        reloaded_offline_sha256 != observed_offline_sha256
        or offline_parent_receipt.policy_sha256
        != _require_sha256(
            offline_network_policy_sha256,
            "offline_network_policy_sha256",
        )
    ):
        raise GTokTokenizerV2Error("fit worker offline policy binding drifted")
    root = _resolve_new_root(output_root)
    try:
        binding = _corpus_fit_binding(
            corpus_root,
            sqlite_path=root / "physical-d6-validation.sqlite",
        )
        safety = preflight_bpe_i32_counts(
            iter_materialized_tokenizer_fit_texts_v4(corpus_root),
            stream_manifest_sha256=str(binding["tokenizer_fit_input_receipt_sha256"]),
        )
        if safety.status != "SAFE":
            raise GTokTokenizerV2Error(
                f"stock tokenizers BPE is blocked by i32 status {safety.status}"
            )
        payload = fit_a2_tokenizer(
            iter_materialized_tokenizer_fit_texts_v4(corpus_root),
            vocab_size=vocab_size,
            length=int(binding["document_count"]),
            safety_receipt=safety,
        )
        tokenizer_path = root / TOKENIZER_FILENAME_V2
        artifact_sha256 = atomic_write_tokenizer(
            tokenizer_path,
            payload,
            expected_vocab_size=vocab_size,
        )
        round_trip = tokenizer_byte_round_trip_receipt_v2(payload)
        (
            executable_sha256,
            lock_sha256,
            environment_sha256,
            runtime_attestation_sha256,
        ) = _runtime_identity(dependency_lock_path=dependency_lock_path)
        receipt = FitWorkerReceiptV2(
            process_id=os.getpid(),
            output_root=str(root),
            vocab_size=vocab_size,
            tokenizer_json_sha256=artifact_sha256,
            merges_sha256=tokenizer_merges_sha256(payload),
            token_inventory_sha256=tokenizer_inventory_sha256(payload),
            reserved_inventory_sha256=_reserved_inventory_sha256(),
            pretokenizer_regex_sha256=_pretokenizer_regex_sha256(),
            fit_stream_sha256=str(binding["fit_stream_sha256"]),
            full_corpus_manifest_sha256=str(binding["full_corpus_manifest_sha256"]),
            screen_submanifest_sha256=str(binding["screen_submanifest_sha256"]),
            physical_d6_evidence_sha256=str(binding["physical_d6_evidence_sha256"]),
            tokenizer_fit_input_receipt_sha256=str(
                binding["tokenizer_fit_input_receipt_sha256"]
            ),
            bpe_safety_receipt_sha256=gtok_v2_bound_sha256(
                "weft1_gtok_v2_bpe_i32_safety", safety
            ),
            byte_round_trip_receipt_sha256=str(round_trip["receipt_sha256"]),
            executable_sha256=executable_sha256,
            dependency_lock_sha256=lock_sha256,
            environment_identity_sha256=environment_sha256,
            runtime_attestation_receipt_sha256=runtime_attestation_sha256,
            offline_network_receipt_sha256=observed_offline_sha256,
            offline_network_policy_sha256=offline_parent_receipt.policy_sha256,
            tokenizers_version=TOKENIZERS_VERSION,
        )
        envelope = {
            "payload": asdict(receipt),
            "receipt_sha256": receipt.receipt_sha256,
            "schema": FIT_WORKER_SCHEMA_V2,
        }
        _exclusive_write_json(root / WORKER_RECEIPT_FILENAME_V2, envelope)
        return receipt
    except BaseException:
        # The exclusive root is intentionally retained as failed-attempt evidence.
        raise


def _parse_worker_receipt(path: Path) -> FitWorkerReceiptV2:
    envelope = _load_json(path)
    if set(envelope) != {"payload", "receipt_sha256", "schema"}:
        raise GTokTokenizerV2Error("fit worker envelope keys drifted")
    if envelope.get("schema") != FIT_WORKER_SCHEMA_V2:
        raise GTokTokenizerV2Error("fit worker schema drifted")
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        raise GTokTokenizerV2Error("fit worker payload must be an object")
    try:
        receipt = FitWorkerReceiptV2(**payload)
    except TypeError as error:
        raise GTokTokenizerV2Error("fit worker payload fields drifted") from error
    if envelope.get("receipt_sha256") != receipt.receipt_sha256:
        raise GTokTokenizerV2Error("fit worker receipt identity mismatch")
    return receipt


def _require_matching_worker_runtime_v2(
    first: FitWorkerReceiptV2, second: FitWorkerReceiptV2
) -> None:
    for name in (
        "executable_sha256",
        "dependency_lock_sha256",
        "environment_identity_sha256",
        "runtime_attestation_receipt_sha256",
        "offline_network_receipt_sha256",
        "offline_network_policy_sha256",
        "tokenizers_version",
    ):
        if getattr(first, name) != getattr(second, name):
            raise GTokTokenizerV2Error(
                f"double-fit governed worker runtime differs in {name}"
            )


def _worker_command(
    *,
    corpus_root: Path,
    output_root: Path,
    vocab_size: int,
    dependency_lock_path: Path,
    worker_executable: Path,
    repository_root: Path,
    offline_network_receipt_path: Path,
    offline_network_receipt_sha256: str,
    offline_network_policy_sha256: str,
) -> list[str]:
    source = (
        "import sys;from pathlib import Path;"
        "sys.path.insert(0,sys.argv[1]);"
        "from training.weft1_gtok_tokenizer_v2 import run_tokenizer_fit_worker_v2;"
        "run_tokenizer_fit_worker_v2(corpus_root=Path(sys.argv[2]),"
        "output_root=Path(sys.argv[3]),vocab_size=int(sys.argv[4]),"
        "dependency_lock_path=Path(sys.argv[5]),"
        "offline_network_receipt_path=Path(sys.argv[6]),"
        "offline_network_receipt_sha256=sys.argv[7],"
        "offline_network_policy_sha256=sys.argv[8])"
    )
    return [
        str(worker_executable.resolve(strict=True)),
        "-I",
        "-B",
        "-c",
        source,
        str(repository_root.resolve(strict=True)),
        str(corpus_root.resolve(strict=True)),
        str(output_root.resolve(strict=False)),
        str(vocab_size),
        str(dependency_lock_path.resolve(strict=True)),
        str(offline_network_receipt_path.resolve(strict=True)),
        _require_sha256(
            offline_network_receipt_sha256,
            "offline_network_receipt_sha256",
        ),
        _require_sha256(
            offline_network_policy_sha256,
            "offline_network_policy_sha256",
        ),
    ]


def _isolated_worker_environment(
    *,
    offline_network_receipt_sha256: str,
    offline_network_policy_sha256: str,
) -> dict[str, str]:
    """Return the closed environment used by every production fit worker."""

    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PATH": os.defpath,
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "SOURCE_DATE_EPOCH": "1712016000",
        "TOKENIZERS_PARALLELISM": "false",
        "TZ": "UTC",
        OFFLINE_RECEIPT_ENV_V2: _require_sha256(
            offline_network_receipt_sha256,
            "offline_network_receipt_sha256",
        ),
        "WEFT1_GTOK_OFFLINE_POLICY_SHA256": _require_sha256(
            offline_network_policy_sha256,
            "offline_network_policy_sha256",
        ),
    }


def fit_tokenizer_arm_double_v2(
    *,
    corpus_root: Path,
    output_parent: Path,
    vocab_size: int,
    dependency_lock_path: Path,
    offline_network_receipt_path: Path,
    offline_network_receipt_sha256: str,
    offline_network_policy_sha256: str,
    repository_root: Path | None = None,
    worker_executable: Path | None = None,
) -> tuple[TokenizerArmReceiptV2, dict[str, Any]]:
    """Fit two fresh subprocesses and parent-rehash the complete evidence."""

    if vocab_size not in GTOK_VOCABULARY_ARMS:
        raise ValueError("production double fit requires a registered vocabulary arm")
    observed_offline_sha256 = assert_offline_descendant_v2(
        offline_network_receipt_path
    )
    if observed_offline_sha256 != _require_sha256(
        offline_network_receipt_sha256,
        "offline_network_receipt_sha256",
    ):
        raise GTokTokenizerV2Error(
            "tokenizer parent offline receipt differs from its launch binding"
        )
    offline_parent_receipt, reloaded_offline_sha256 = load_offline_parent_receipt_v2(
        offline_network_receipt_path
    )
    if (
        reloaded_offline_sha256 != observed_offline_sha256
        or offline_parent_receipt.policy_sha256
        != _require_sha256(
            offline_network_policy_sha256,
            "offline_network_policy_sha256",
        )
    ):
        raise GTokTokenizerV2Error("tokenizer parent offline policy binding drifted")
    parent = _resolve_new_root(output_parent)
    first_root = parent / "fit-a"
    second_root = parent / "fit-b"
    cwd = (repository_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    executable = (worker_executable or Path(sys.executable)).resolve(strict=True)
    for worker_root in (first_root, second_root):
        completed = subprocess.run(
            _worker_command(
                corpus_root=corpus_root,
                output_root=worker_root,
                vocab_size=vocab_size,
                dependency_lock_path=dependency_lock_path,
                worker_executable=executable,
                repository_root=cwd,
                offline_network_receipt_path=offline_network_receipt_path,
                offline_network_receipt_sha256=observed_offline_sha256,
                offline_network_policy_sha256=offline_parent_receipt.policy_sha256,
            ),
            cwd=cwd,
            env=_isolated_worker_environment(
                offline_network_receipt_sha256=observed_offline_sha256,
                offline_network_policy_sha256=offline_parent_receipt.policy_sha256,
            ),
            close_fds=True,
            check=False,
        )
        if completed.returncode != 0:
            raise GTokTokenizerV2Error(
                f"tokenizer fit subprocess failed with exit {completed.returncode}"
            )
    first = _parse_worker_receipt(first_root / WORKER_RECEIPT_FILENAME_V2)
    second = _parse_worker_receipt(second_root / WORKER_RECEIPT_FILENAME_V2)
    if first.process_id == second.process_id:
        raise GTokTokenizerV2Error("double fit did not use distinct subprocesses")
    _require_matching_worker_runtime_v2(first, second)
    shared_fields = (
        "vocab_size",
        "tokenizer_json_sha256",
        "merges_sha256",
        "token_inventory_sha256",
        "reserved_inventory_sha256",
        "pretokenizer_regex_sha256",
        "fit_stream_sha256",
        "full_corpus_manifest_sha256",
        "screen_submanifest_sha256",
        "physical_d6_evidence_sha256",
        "tokenizer_fit_input_receipt_sha256",
        "bpe_safety_receipt_sha256",
        "byte_round_trip_receipt_sha256",
        "executable_sha256",
        "dependency_lock_sha256",
        "environment_identity_sha256",
        "runtime_attestation_receipt_sha256",
        "offline_network_receipt_sha256",
        "offline_network_policy_sha256",
        "tokenizers_version",
    )
    drift = tuple(name for name in shared_fields if getattr(first, name) != getattr(second, name))
    if drift:
        raise GTokTokenizerV2Error(f"double fit differs in fields: {drift!r}")
    for worker, root in ((first, first_root), (second, second_root)):
        artifact = (root / TOKENIZER_FILENAME_V2).read_bytes()
        validate_tokenizer_json(artifact, expected_vocab_size=vocab_size)
        if (
            tokenizer_artifact_sha256(artifact) != worker.tokenizer_json_sha256
            or tokenizer_merges_sha256(artifact) != worker.merges_sha256
            or tokenizer_inventory_sha256(artifact) != worker.token_inventory_sha256
        ):
            raise GTokTokenizerV2Error("parent rehash differs from fit worker evidence")
        round_trip = tokenizer_byte_round_trip_receipt_v2(artifact)
        if round_trip["receipt_sha256"] != worker.byte_round_trip_receipt_sha256:
            raise GTokTokenizerV2Error("parent byte round-trip differs from worker")
    double_core = {
        "first_process_id": first.process_id,
        "first_worker_receipt_sha256": first.receipt_sha256,
        "second_process_id": second.process_id,
        "second_worker_receipt_sha256": second.receipt_sha256,
        "status": "PARENT_REHASHED_SUBPROCESSES_MATCH",
        "tokenizer_json_sha256": first.tokenizer_json_sha256,
        "vocab_size": vocab_size,
        "offline_network_receipt_sha256": observed_offline_sha256,
        "offline_network_policy_sha256": offline_parent_receipt.policy_sha256,
    }
    double_receipt = {
        **double_core,
        "receipt_sha256": gtok_v2_bound_sha256(DOUBLE_FIT_SCHEMA_V2, double_core),
    }
    arm = TokenizerArmReceiptV2(
        vocab_size=vocab_size,
        tokenizer_json_sha256=first.tokenizer_json_sha256,
        merges_sha256=first.merges_sha256,
        token_inventory_sha256=first.token_inventory_sha256,
        reserved_inventory_sha256=first.reserved_inventory_sha256,
        pretokenizer_regex_sha256=first.pretokenizer_regex_sha256,
        fit_stream_sha256=first.fit_stream_sha256,
        full_corpus_manifest_sha256=first.full_corpus_manifest_sha256,
        double_fit_receipt_sha256=str(double_receipt["receipt_sha256"]),
        byte_round_trip_receipt_sha256=first.byte_round_trip_receipt_sha256,
        token_inventory_count=vocab_size,
    )
    parent_receipt = {
        "arm": asdict(arm),
        "arm_receipt_sha256": arm.receipt_sha256,
        "double_fit": double_receipt,
        "offline_network_receipt_sha256": observed_offline_sha256,
        "offline_network_policy_sha256": offline_parent_receipt.policy_sha256,
        "selected_artifact_relative_path": "fit-a/tokenizer.json",
    }
    _exclusive_write_json(parent / "tokenizer-arm-receipt.json", parent_receipt)
    return arm, parent_receipt


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("fit-worker")
    worker.add_argument("--corpus-root", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--vocab-size", type=int, required=True)
    worker.add_argument("--dependency-lock", type=Path, required=True)
    worker.add_argument("--offline-network-receipt", type=Path, required=True)
    worker.add_argument("--offline-network-receipt-sha256", required=True)
    worker.add_argument("--offline-network-policy-sha256", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "fit-worker":
        run_tokenizer_fit_worker_v2(
            corpus_root=arguments.corpus_root,
            output_root=arguments.output_root,
            vocab_size=arguments.vocab_size,
            dependency_lock_path=arguments.dependency_lock,
            offline_network_receipt_path=arguments.offline_network_receipt,
            offline_network_receipt_sha256=arguments.offline_network_receipt_sha256,
            offline_network_policy_sha256=arguments.offline_network_policy_sha256,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "BYTE_ROUND_TRIP_SCHEMA_V2",
    "DOUBLE_FIT_SCHEMA_V2",
    "FIT_WORKER_SCHEMA_V2",
    "FitWorkerReceiptV2",
    "GTokTokenizerV2Error",
    "fit_tokenizer_arm_double_v2",
    "run_tokenizer_fit_worker_v2",
    "tokenizer_byte_round_trip_receipt_v2",
]
