"""Forward-only hermetic P-B decontamination runner for WEFT-1.

Only the child process reads sealed inputs.  It verifies the six P3.1 CONFIRM
seals against the complete private partition ledger before deriving any prompt
fingerprint, consumes EVAL-E only through its existing anonymous index, streams
every V4 full-corpus shard, and writes one aggregate receipt.  No plaintext,
row identity, salt, or sealed identity is serialized by this module.

CONFIRM CLEAN does not depend on probabilistic LSH recall.  A deterministic
salted-shingle prefix filter has no false negatives at Jaccard >= 4/5; exact
raw-shingle Jaccard makes the decision.  The registered A2 MinHash/LSH is run
only as an internal diagnostic on safe-filter candidates.  EVAL-E's locked
anonymous index contains only signatures, so every one of its 128-component
signatures is compared directly, without an LSH prefilter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import numpy as np

from training.paper2_phase3_p31 import (
    ALL_BATTERIES,
    SPLIT_SEED as P31_SPLIT_SEED,
    canonical_sha256,
    content_sha256,
)
from training.weft1_corpus_a2 import (
    byte_shingles_v3,
    lsh_band_keys_v3,
    minhash_signature_v3,
    normalized_match_bytes,
)
from training.weft1_corpus_decon_contract import (
    CONFIRM_BATTERIES,
    DECON_BATTERIES,
    GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256,
    GOVERNED_CONFIRM_SEAL_SET_SHA256,
    GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256,
    GOVERNED_CONFIRM_SOURCE_ROWS_SHA256,
    GOVERNED_EVAL_E_ANONYMOUS_INDEX_SHA256,
    GOVERNED_EVAL_E_LOCK_SHA256,
    LEGACY_EVAL_E_CHARACTER_SHINGLE_SIZE,
    LEGACY_EVAL_E_EXACT_HASH,
    LEGACY_EVAL_E_LSH_BANDS,
    LEGACY_EVAL_E_LSH_ROWS_PER_BAND,
    LEGACY_EVAL_E_MINHASH_COMPONENTS,
    LEGACY_EVAL_E_MINHASH_SEED,
    LEGACY_EVAL_E_NORMALIZATION,
    algorithm_profiles,
    jaccard_at_least_four_fifths,
    length_can_reach_four_fifths,
    safe_prefix_size,
    signature_at_least_four_fifths,
)
from training.weft1_gtok_contract import canonical_json_bytes
from training.weft1_strict_io import assert_no_symlink_ancestors


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DECON_WORKER_RELATIVE_PATH = "scripts/run_weft1_corpus_decon.py"
DECON_CODE_RELATIVE_PATHS = (
    "docs/PAPER2_PHASE3_P31_P32_AUTHORIZED_BUILD_HANDOFF_20260810.md",
    "docs/STRATEGY_CORPUS_GTOK_AMENDMENT_A3_20260829.md",
    "docs/STRATEGY_RELEASE_POSTURE_AND_LICENSE_CLOSE_20260830.md",
    "scripts/build_weft1_pa_runtime.py",
    DECON_WORKER_RELATIVE_PATH,
    "training/paper2_phase3_p31.py",
    "training/weft1_corpus_a2.py",
    "training/weft1_corpus_a3.py",
    "training/weft1_corpus_breakdown_a3.py",
    "training/weft1_corpus_contract.py",
    "training/weft1_corpus_decon.py",
    "training/weft1_corpus_decon_contract.py",
    "training/weft1_corpus_effective_routes_a3_20260829.json",
    "training/weft1_corpus_enumeration_a2.py",
    "training/weft1_corpus_fetch_a2.py",
    "training/weft1_corpus_fetch_a3.py",
    "training/weft1_corpus_gtok_a2_bindings_20260828.json",
    "training/weft1_corpus_gtok_a2_requirements.lock",
    "training/weft1_corpus_materialize_a2.py",
    "training/weft1_corpus_materialize_a3.py",
    "training/weft1_corpus_pa.py",
    "training/weft1_corpus_pb.py",
    "training/weft1_corpus_replay_a2.py",
    "training/weft1_corpus_replay_a3.py",
    "training/weft1_corpus_semantic_evidence_a3.py",
    "training/weft1_corpus_semantic_evidence_a3_20260829.json",
    "training/weft1_corpus_source_io_a2.py",
    "training/weft1_corpus_sources_a2.py",
    "training/weft1_corpus_streaming_a2.py",
    "training/weft1_gtok_a1_contract.py",
    "training/weft1_gtok_contract.py",
    "training/weft1_gtok_source_routes_20260828.json",
    "training/weft1_gtok_tokenizer_a2.py",
    "training/weft1_release.py",
    "training/weft1_release_bindings_20260830.json",
    "training/weft1_release_card_evidence_20260830.json",
    "training/weft1_seed.py",
    "training/weft1_strict_io.py",
)
DECON_RECEIPT_FILENAME = "hermetic-decon-receipt.json"

# A2-R7 replay literals for this CPU-only sealed scan.  The 64 MiB burst is a
# bounded, representative pre-pass spread across every full shard.  The child
# refuses to begin its authoritative pass if measured throughput projects past
# twelve hours, and independent elapsed-time meters remain live throughout the
# full pass and its parent launch.  The public parent default retains a one-hour
# outer-policy margin for callers, but the effective launch deadline is always
# capped at the same twelve-hour scientific tripwire.
DECON_CALIBRATION_LOGICAL_BYTES = 64 * 1024 * 1024
# This bound is applied by ``readline(limit)`` before JSON parsing in both the
# calibration and authoritative passes.  It is an A2-R7 replay literal, not a
# corpus filter: encountering a larger canonical record stops the line.
DECON_MAX_RECORD_JSONL_BYTES = 64 * 1024 * 1024
DECON_MAX_PROJECTED_SECONDS = 12 * 60 * 60
DECON_MAX_RUNTIME_SECONDS = 12 * 60 * 60
DECON_PARENT_WATCHDOG_SECONDS = 13 * 60 * 60

LEGACY_INDEX_KIND = "paper2_tm0_eval_e_anonymous_screening_index_v1"
LEGACY_PANEL_NAME = "eval_e_hermetic_screen"
NETWORK_PROBE_RESULT = "python_socket_connect_blocked"

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SALT_HEX = re.compile(r"^[0-9a-f]+$")
_SPACE_RE = re.compile(r"\s+")
_UINT64_MAX = np.iinfo(np.uint64).max


class DeconError(RuntimeError):
    """A sealed input, corpus shard, isolation, or receipt invariant failed."""


@dataclass(frozen=True)
class DeconCalibration:
    compressed_sample_bytes: int
    logical_sample_bytes: int
    projected_seconds: float
    shard_count: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DeconError(f"{name} must be a lowercase SHA-256")
    return value


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeconError("JSON input repeats a field")
        result[key] = value
    return result


def _load_json(path: Path, name: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        safe = assert_no_symlink_ancestors(path).resolve(strict=True)
        if not safe.is_file() or safe.is_symlink():
            raise DeconError(f"{name} is not a regular file")
        raw = safe.read_bytes()
        value = json.loads(raw, object_pairs_hook=_json_no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DeconError(f"{name} is unreadable or malformed") from error
    if not isinstance(value, Mapping):
        raise DeconError(f"{name} must be a JSON object")
    return raw, value


def _exact_fields(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DeconError(f"{name} fields drifted")
    return value


def _canonical_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DeconError("full shard path is not canonical")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DeconError("full shard path is not canonical")
    return value


def _salted_frame_digest(salt: bytes, value: bytes) -> str:
    return hashlib.sha256(salt + len(value).to_bytes(8, "big") + value).hexdigest()


def _salted_shingle_digest(salt: bytes, value: bytes) -> bytes:
    return hashlib.sha256(salt + len(value).to_bytes(8, "big") + value).digest()


def _legacy_normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _legacy_salted_exact(text: str, salt: bytes) -> str:
    return hashlib.sha256(salt + _legacy_normalize(text).encode("utf-8")).hexdigest()


def _legacy_character_shingles(text: str, width: int) -> Iterable[bytes]:
    normalized = _legacy_normalize(text)
    if len(normalized) <= width:
        yield normalized.encode("utf-8")
        return
    for index in range(len(normalized) - width + 1):
        yield normalized[index : index + width].encode("utf-8")


def _legacy_minhash_coefficients(
    *, seed: int, components: int
) -> tuple[np.ndarray, np.ndarray]:
    if seed != LEGACY_EVAL_E_MINHASH_SEED:
        raise DeconError("EVAL-E MinHash seed drifted")
    if type(components) is not int or not 1 <= components <= LEGACY_EVAL_E_MINHASH_COMPONENTS:
        raise DeconError("EVAL-E MinHash component prefix is invalid")
    rng = np.random.default_rng(seed)
    multipliers = (
        rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=LEGACY_EVAL_E_MINHASH_COMPONENTS,
            dtype=np.uint64,
        )
        | np.uint64(1)
    )
    offsets = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=LEGACY_EVAL_E_MINHASH_COMPONENTS,
        dtype=np.uint64,
    )
    return multipliers[:components], offsets[:components]


def _legacy_hashed_normalized_shingle_values(normalized: str) -> np.ndarray:
    # The locked implementation takes a set before hashing.  Materializing one
    # document's set is bounded by the JSONL record already resident in memory,
    # removes duplicate small-hash calls, and exactly reproduces the legacy
    # algorithm rather than treating repeated shingles as distinct samples.
    width = LEGACY_EVAL_E_CHARACTER_SHINGLE_SIZE
    if len(normalized) <= width:
        shingles = frozenset((normalized.encode("utf-8"),))
    else:
        shingles = frozenset(
            normalized[index : index + width].encode("utf-8")
            for index in range(len(normalized) - width + 1)
        )
    if not shingles:
        raise DeconError("legacy EVAL-E signature observed no shingle")
    return np.fromiter(
        (
            int.from_bytes(hashlib.sha256(shingle).digest()[:8], "little")
            for shingle in shingles
        ),
        dtype=np.uint64,
        count=len(shingles),
    )


def _legacy_hashed_shingle_values(text: str) -> np.ndarray:
    return _legacy_hashed_normalized_shingle_values(_legacy_normalize(text))


def _legacy_signature_from_values(
    values: np.ndarray, *, seed: int, components: int
) -> tuple[int, ...]:
    if not isinstance(values, np.ndarray) or values.dtype != np.uint64 or values.ndim != 1:
        raise DeconError("legacy EVAL-E shingle values have invalid shape")
    if values.size < 1:
        raise DeconError("legacy EVAL-E signature observed no shingle")
    multipliers, offsets = _legacy_minhash_coefficients(
        seed=seed, components=components
    )
    signature = np.full(components, _UINT64_MAX, dtype=np.uint64)
    for start in range(0, len(values), 4096):
        block = values[start : start + 4096]
        signature = np.minimum(
            signature,
            (multipliers[:, None] * block[None, :] + offsets[:, None]).min(axis=1),
        )
    return tuple(int(value) for value in signature)


def legacy_minhash_signature(
    text: str,
    *,
    seed: int,
    components: int = LEGACY_EVAL_E_MINHASH_COMPONENTS,
) -> tuple[int, ...]:
    """Reproduce the locked EVAL-E signature in bounded 4096-shingle blocks."""

    values = _legacy_hashed_shingle_values(text)
    return _legacy_signature_from_values(
        values, seed=seed, components=components
    )


def render_confirm_prompt(row: Mapping[str, Any]) -> str:
    """Render only evaluation-visible prompt content; answers are never screened."""

    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, Mapping) or not isinstance(prompt.get("question"), str):
        raise DeconError("CONFIRM prompt shape is unsupported")
    rendered = [str(prompt["question"])]
    labels = prompt.get("choice_labels")
    choices = prompt.get("choice_text")
    if isinstance(labels, list) and isinstance(choices, list):
        if len(labels) != len(choices) or any(
            not isinstance(value, str) for value in (*labels, *choices)
        ):
            raise DeconError("CONFIRM labeled choices drifted")
        rendered.extend(
            f"{label}. {choice}"
            for label, choice in zip(labels, choices, strict=True)
        )
    elif isinstance(prompt.get("choices"), list):
        values = prompt["choices"]
        if any(not isinstance(value, str) for value in values):
            raise DeconError("CONFIRM choices drifted")
        rendered.extend(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(values))
    elif set(prompt) != {"question"}:
        raise DeconError("CONFIRM prompt fields drifted")
    return "\n".join(rendered)


class _ConfirmIndex:
    def __init__(self, *, salt: bytes) -> None:
        self.salt = salt
        self.exact_hashes: set[str] = set()
        self.shingle_sets: list[frozenset[bytes]] = []
        self.signatures: list[tuple[int, ...]] = []
        self.prefix: dict[bytes, set[int]] = defaultdict(set)
        self._patterns: tuple[re.Pattern[bytes], ...] | None = None

    def add(self, text: str) -> None:
        normalized = normalized_match_bytes(text)
        self.exact_hashes.add(_salted_frame_digest(self.salt, normalized))
        if not normalized:
            return
        shingles = byte_shingles_v3(normalized)
        ordinal = len(self.shingle_sets)
        self.shingle_sets.append(shingles)
        signature = minhash_signature_v3(shingles)
        self.signatures.append(signature)
        ordered = sorted(
            (_salted_shingle_digest(self.salt, value), value) for value in shingles
        )
        for _digest, value in ordered[: safe_prefix_size(len(shingles))]:
            # Salted ordering prevents the prefix from exposing a semantic
            # choice.  Lookup uses raw shingles only inside the child, which
            # removes a SHA-256 call at every corpus-byte position.
            self.prefix[value].add(ordinal)

    def finalize(self) -> None:
        if self._patterns is not None:
            raise DeconError("CONFIRM index was finalized twice")
        by_width: dict[int, list[bytes]] = defaultdict(list)
        for value in self.prefix:
            by_width[len(value)].append(value)
        patterns: list[re.Pattern[bytes]] = []
        for width in sorted(by_width):
            alternatives = b"|".join(
                re.escape(value) for value in sorted(by_width[width])
            )
            patterns.append(re.compile(b"(?=(" + alternatives + b"))"))
        if not patterns:
            raise DeconError("CONFIRM index contains no near-match prefix")
        self._patterns = tuple(patterns)

    def match(self, text: str) -> tuple[bool, bool]:
        normalized = normalized_match_bytes(text)
        if _salted_frame_digest(self.salt, normalized) in self.exact_hashes:
            return True, False
        if not normalized:
            return False, False
        if self._patterns is None:
            raise DeconError("CONFIRM index was not finalized")
        candidates: set[int] = set()
        for pattern in self._patterns:
            for match in pattern.finditer(normalized):
                candidates.update(self.prefix[match.group(1)])
        if not candidates:
            return False, False
        query = byte_shingles_v3(normalized)
        query_signature: tuple[int, ...] | None = None
        for ordinal in sorted(candidates):
            sealed = self.shingle_sets[ordinal]
            if not length_can_reach_four_fifths(len(query), len(sealed)):
                continue
            if jaccard_at_least_four_fifths(query, sealed):
                # The A2 MinHash/LSH remains a diagnostic only.  Computing it
                # here proves the registered path runs while exact Jaccard,
                # reached through the safe prefix, controls CLEAN/HIT.
                if query_signature is None:
                    query_signature = minhash_signature_v3(query)
                _ = bool(
                    set(lsh_band_keys_v3(query_signature))
                    & set(lsh_band_keys_v3(self.signatures[ordinal]))
                )
                return False, True
        return False, False


class _EvalEIndex:
    def __init__(
        self,
        *,
        salt: bytes,
        exact_hashes: frozenset[str],
        signatures: tuple[tuple[int, ...], ...],
    ) -> None:
        self.salt = salt
        self.exact_hashes = exact_hashes
        self.signatures = signatures
        self.prefix_components = 26
        self.prefix: dict[tuple[int, int], set[int]] = defaultdict(set)
        for ordinal, signature in enumerate(signatures):
            for component in range(self.prefix_components):
                self.prefix[(component, signature[component])].add(ordinal)

    def match(self, text: str) -> tuple[bool, bool]:
        normalized = _legacy_normalize(text)
        if (
            hashlib.sha256(self.salt + normalized.encode("utf-8")).hexdigest()
            in self.exact_hashes
        ):
            return True, False
        values = _legacy_hashed_normalized_shingle_values(normalized)
        prefix = _legacy_signature_from_values(
            values,
            seed=LEGACY_EVAL_E_MINHASH_SEED,
            components=self.prefix_components,
        )
        candidates: set[int] = set()
        for component, value in enumerate(prefix):
            candidates.update(self.prefix.get((component, value), ()))
        if not candidates:
            return False, False
        signature = _legacy_signature_from_values(
            values,
            seed=LEGACY_EVAL_E_MINHASH_SEED,
            components=LEGACY_EVAL_E_MINHASH_COMPONENTS,
        )
        # At threshold 4/5, at most 25 of 128 components differ.  Therefore a
        # qualifying signature must match at least one of the fixed first 26;
        # after that proof-safe filter, every component is compared.  LSH is
        # never allowed to turn a miss into CLEAN.
        return False, any(
            signature_at_least_four_fifths(signature, self.signatures[ordinal])
            for ordinal in sorted(candidates)
        )


def _iter_jsonl_bytes(raw: bytes, name: str) -> Iterable[Mapping[str, Any]]:
    try:
        with io.BytesIO(raw) as handle:
            for raw_line in handle:
                if not raw_line.endswith(b"\n") or not raw_line.strip():
                    raise DeconError(f"{name} framing drifted")
                value = json.loads(
                    raw_line.decode("utf-8", errors="strict"),
                    object_pairs_hook=_json_no_duplicates,
                )
                if not isinstance(value, Mapping):
                    raise DeconError(f"{name} contains a non-object row")
                yield value
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DeconError(f"{name} is unreadable or malformed") from error


def _validate_confirm_membership(
    *,
    seal_paths: Sequence[Path],
    seal_ledger_path: Path,
    private_rows_path: Path,
    salt: bytes,
) -> tuple[_ConfirmIndex, dict[str, str], str]:
    """Verify all P3.1 membership commitments before fingerprinting prompts."""

    if len(seal_paths) != len(CONFIRM_BATTERIES):
        raise DeconError("CONFIRM requires exactly six independent seal files")
    private_path = assert_no_symlink_ancestors(private_rows_path).resolve(strict=True)
    if not private_path.is_file() or private_path.is_symlink():
        raise DeconError("CONFIRM private rows are not a regular file")
    try:
        private_raw = private_path.read_bytes()
    except OSError as error:
        raise DeconError("CONFIRM private rows are unreadable") from error
    private_rows_sha256 = _sha256_bytes(private_raw)
    if private_rows_sha256 != GOVERNED_CONFIRM_SOURCE_ROWS_SHA256:
        raise DeconError("CONFIRM private rows are not the governed source rows")
    ledger_raw, ledger = _load_json(seal_ledger_path, "CONFIRM seal ledger")
    ledger = _exact_fields(
        ledger,
        {
            "assertions",
            "confirm_scoring_spent",
            "kind",
            "models_loaded",
            "scores_computed",
            "seal_set_sha256",
            "seals",
            "status",
        },
        "CONFIRM seal ledger",
    )
    assertions = _exact_fields(
        ledger.get("assertions"),
        {
            "confirm_membership_sealed",
            "confirm_scoring_unspent",
            "models_not_loaded",
            "scores_not_computed",
        },
        "CONFIRM seal assertions",
    )
    if (
        ledger.get("kind") != "paper2_phase3_p31_confirm_seal_ledger_v1"
        or ledger.get("status") != "sealed_before_model_scoring"
        or ledger.get("models_loaded") is not False
        or ledger.get("scores_computed") is not False
        or ledger.get("confirm_scoring_spent") is not False
        or any(value is not True for value in assertions.values())
    ):
        raise DeconError("CONFIRM seal ledger is not score-blind and unspent")

    independent: dict[str, tuple[Mapping[str, Any], str]] = {}
    seal_payload_fields = {
        "atomic_lease_required_for_future_scoring",
        "battery",
        "complete_ledger_sha256",
        "kind",
        "membership_sha256",
        "partition",
        "row_count",
        "scoring_authorized",
        "scoring_spent",
        "source_manifest_sha256",
        "source_rows_sha256",
        "split_seed",
    }
    for path in seal_paths:
        raw, seal = _load_json(path, "CONFIRM membership seal")
        seal = _exact_fields(seal, seal_payload_fields, "CONFIRM membership seal")
        battery = seal.get("battery")
        if not isinstance(battery, str) or f"CONFIRM:{battery}" not in CONFIRM_BATTERIES:
            raise DeconError("CONFIRM seal names an unknown battery")
        if battery in independent:
            raise DeconError("CONFIRM seal battery is duplicated")
        if (
            seal.get("kind") != "paper2_phase3_p31_confirm_membership_seal_v1"
            or seal.get("partition") != "confirm"
            or type(seal.get("row_count")) is not int
            or int(seal["row_count"]) < 1
            or seal.get("scoring_authorized") is not False
            or seal.get("scoring_spent") is not False
            or seal.get("atomic_lease_required_for_future_scoring") is not True
            or seal.get("source_rows_sha256") != private_rows_sha256
            or seal.get("source_manifest_sha256")
            != GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256
            or seal.get("split_seed") != P31_SPLIT_SEED
        ):
            raise DeconError("CONFIRM seal posture drifted")
        for field in (
            "complete_ledger_sha256",
            "membership_sha256",
            "source_manifest_sha256",
            "source_rows_sha256",
        ):
            _require_sha256(seal.get(field), f"CONFIRM seal {field}")
        independent[battery] = (seal, _sha256_bytes(raw))
    if set(independent) != set(ALL_BATTERIES):
        raise DeconError("CONFIRM independent seal set is incomplete")

    embedded_rows = ledger.get("seals")
    if not isinstance(embedded_rows, list) or len(embedded_rows) != len(ALL_BATTERIES):
        raise DeconError("CONFIRM seal ledger does not embed all seals")
    embedded_projection: list[dict[str, str]] = []
    observed_embedded: set[str] = set()
    for embedded in embedded_rows:
        embedded = _exact_fields(
            embedded,
            seal_payload_fields | {"path", "sha256"},
            "embedded CONFIRM seal",
        )
        battery = embedded.get("battery")
        if not isinstance(battery, str) or battery in observed_embedded:
            raise DeconError("embedded CONFIRM seal battery drifted")
        observed_embedded.add(battery)
        expected = independent.get(battery)
        if expected is None:
            raise DeconError("embedded CONFIRM seal lacks an independent file")
        payload = {key: embedded[key] for key in seal_payload_fields}
        if payload != expected[0] or embedded.get("sha256") != expected[1]:
            raise DeconError("embedded and independent CONFIRM seals disagree")
        embedded_projection.append({"battery": battery, "sha256": expected[1]})
    if (
        ledger.get("seal_set_sha256") != GOVERNED_CONFIRM_SEAL_SET_SHA256
        or canonical_sha256(embedded_projection)
        != GOVERNED_CONFIRM_SEAL_SET_SHA256
    ):
        raise DeconError("CONFIRM seal-set commitment drifted")

    # First pass: validate the entire private partition file and reproduce the
    # score-blind ledger/membership hashes.  Prompt text is not fingerprinted
    # until every check below passes.
    selected: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    all_rows: list[dict[str, str]] = []
    seen_items: set[tuple[str, str]] = set()
    for row in _iter_jsonl_bytes(private_raw, "CONFIRM private rows"):
        battery = row.get("battery")
        item_id = row.get("item_id")
        document_id = row.get("document_id")
        partition = row.get("partition")
        claimed_content = row.get("content_sha256")
        if (
            battery not in ALL_BATTERIES
            or not isinstance(item_id, str)
            or not item_id
            or not isinstance(document_id, str)
            or not document_id
            or partition not in {"verified_train", "dev", "confirm"}
            or not isinstance(claimed_content, str)
        ):
            raise DeconError("CONFIRM private row identity fields drifted")
        key = (str(battery), item_id)
        if key in seen_items:
            raise DeconError("CONFIRM private rows repeat a battery item")
        seen_items.add(key)
        if content_sha256(row) != claimed_content:
            raise DeconError("CONFIRM private row content commitment drifted")
        projection = {
            "content_sha256": claimed_content,
            "document_id": document_id,
            "item_id": item_id,
        }
        selected[(str(battery), str(partition))].append(projection)
        all_rows.append(
            {
                "battery": str(battery),
                "content_sha256": claimed_content,
                "document_id": document_id,
                "item_id": item_id,
                "partition": str(partition),
            }
        )
    all_rows.sort(
        key=lambda item: (
            item["battery"],
            item["partition"],
            item["document_id"],
            item["item_id"],
        )
    )
    complete_commitments = {
        str(seal[0]["complete_ledger_sha256"]) for seal in independent.values()
    }
    governed_complete = _require_sha256(
        GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256,
        "governed CONFIRM complete ledger",
    )
    if complete_commitments != {governed_complete} or (
        canonical_sha256(all_rows) != governed_complete
    ):
        raise DeconError("CONFIRM complete private ledger commitment drifted")
    for battery in ALL_BATTERIES:
        values = selected[(battery, "confirm")]
        values.sort(key=lambda item: (item["document_id"], item["item_id"]))
        seal = independent[battery][0]
        if (
            len(values) != seal["row_count"]
            or canonical_sha256(values) != seal["membership_sha256"]
        ):
            raise DeconError("CONFIRM sealed membership failed reproduction")

    # Second pass starts only after membership and score-blindness are proven.
    index = _ConfirmIndex(salt=salt)
    observed_confirm = 0
    for row in _iter_jsonl_bytes(private_raw, "CONFIRM private rows"):
        if row.get("partition") != "confirm":
            continue
        index.add(render_confirm_prompt(row))
        observed_confirm += 1
    expected_confirm = sum(int(value[0]["row_count"]) for value in independent.values())
    if observed_confirm != expected_confirm:
        raise DeconError("CONFIRM prompt screening coverage drifted")
    index.finalize()

    physical_rows = [
        {"battery": battery, "sha256": independent[battery][1]}
        for battery in ALL_BATTERIES
    ]
    physical_set_sha256 = _sha256_bytes(canonical_json_bytes(physical_rows))
    if physical_set_sha256 != GOVERNED_CONFIRM_SEAL_SET_SHA256:
        raise DeconError("CONFIRM independent seal-file set is not governed")
    commitments = {
        "confirm_complete_ledger_sha256": governed_complete,
        "confirm_private_rows_sha256": private_rows_sha256,
        "confirm_seal_file_set_sha256": physical_set_sha256,
        "confirm_seal_ledger_sha256": _sha256_bytes(ledger_raw),
        "confirm_source_manifest_sha256": (
            GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256
        ),
    }
    return index, commitments, physical_set_sha256


def _load_eval_e_index(
    *, index_path: Path, lock_path: Path
) -> tuple[_EvalEIndex, dict[str, str]]:
    lock_raw, lock = _load_json(lock_path, "EVAL-E lock")
    if _sha256_bytes(lock_raw) != GOVERNED_EVAL_E_LOCK_SHA256:
        raise DeconError("EVAL-E lock identity is not the governed lock")
    panels = lock.get("panels")
    if not isinstance(panels, Mapping) or not isinstance(
        panels.get(LEGACY_PANEL_NAME), Mapping
    ):
        raise DeconError("EVAL-E lock lacks the governed panel")
    parameters = dict(panels[LEGACY_PANEL_NAME])
    required_parameters = {
        "character_shingle_size",
        "estimated_jaccard_threshold",
        "exact_hash",
        "lsh_bands",
        "lsh_rows_per_band",
        "minhash_components",
        "minhash_seed",
        "no_backfill",
        "normalization",
        "partition_seed",
        "prior_partition_seed",
        "salt_hex",
    }
    if set(parameters) != required_parameters:
        raise DeconError("EVAL-E locked parameter fields drifted")
    if (
        parameters.get("normalization") != LEGACY_EVAL_E_NORMALIZATION
        or parameters.get("exact_hash") != LEGACY_EVAL_E_EXACT_HASH
        or parameters.get("character_shingle_size")
        != LEGACY_EVAL_E_CHARACTER_SHINGLE_SIZE
        or parameters.get("minhash_components")
        != LEGACY_EVAL_E_MINHASH_COMPONENTS
        or parameters.get("minhash_seed") != LEGACY_EVAL_E_MINHASH_SEED
        or parameters.get("lsh_bands") != LEGACY_EVAL_E_LSH_BANDS
        or parameters.get("lsh_rows_per_band")
        != LEGACY_EVAL_E_LSH_ROWS_PER_BAND
        or parameters.get("estimated_jaccard_threshold") != 0.8
        or parameters.get("no_backfill") is not True
    ):
        raise DeconError("EVAL-E locked algorithm profile drifted")
    salt_hex = parameters.get("salt_hex")
    if (
        not isinstance(salt_hex, str)
        or len(salt_hex) < 32
        or len(salt_hex) % 2
        or _SALT_HEX.fullmatch(salt_hex) is None
    ):
        raise DeconError("EVAL-E private salt encoding drifted")
    salt = bytes.fromhex(salt_hex)

    index_raw, payload = _load_json(index_path, "EVAL-E anonymous index")
    if _sha256_bytes(index_raw) != GOVERNED_EVAL_E_ANONYMOUS_INDEX_SHA256:
        raise DeconError("EVAL-E anonymous-index identity is not governed")
    payload = _exact_fields(
        payload,
        {
            "document_count",
            "document_ids_persisted",
            "kind",
            "metadata_persisted",
            "minhash_signatures_uint64_decimal",
            "parameters",
            "plaintext_persisted",
            "salted_exact_hashes",
        },
        "EVAL-E anonymous index",
    )
    if (
        payload.get("kind") != LEGACY_INDEX_KIND
        or payload.get("parameters") != parameters
        or payload.get("plaintext_persisted") is not False
        or payload.get("document_ids_persisted") is not False
        or payload.get("metadata_persisted") is not False
        or type(payload.get("document_count")) is not int
        or int(payload["document_count"]) < 1
    ):
        raise DeconError("EVAL-E anonymous-index posture drifted")
    exact_values = payload.get("salted_exact_hashes")
    signature_values = payload.get("minhash_signatures_uint64_decimal")
    if (
        not isinstance(exact_values, list)
        or not isinstance(signature_values, list)
        or len(exact_values) != payload["document_count"]
        or len(signature_values) != payload["document_count"]
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in exact_values
        )
        or exact_values != sorted(exact_values)
    ):
        raise DeconError("EVAL-E anonymous exact-hash inventory drifted")
    signatures: list[tuple[int, ...]] = []
    for row in signature_values:
        if not isinstance(row, list) or len(row) != LEGACY_EVAL_E_MINHASH_COMPONENTS:
            raise DeconError("EVAL-E anonymous signature shape drifted")
        parsed: list[int] = []
        for value in row:
            if not isinstance(value, str) or not value.isascii() or not value.isdigit():
                raise DeconError("EVAL-E anonymous signature encoding drifted")
            integer = int(value)
            if not 0 <= integer <= int(_UINT64_MAX):
                raise DeconError("EVAL-E anonymous signature range drifted")
            parsed.append(integer)
        signatures.append(tuple(parsed))
    if signature_values != sorted(signature_values):
        raise DeconError("EVAL-E anonymous signatures are not canonically ordered")
    return (
        _EvalEIndex(
            salt=salt,
            exact_hashes=frozenset(exact_values),
            signatures=tuple(signatures),
        ),
        {
            "eval_e_anonymous_index_sha256": _sha256_bytes(index_raw),
            "eval_e_lock_sha256": _sha256_bytes(lock_raw),
        },
    )


def _load_zstandard() -> Any:
    try:
        import zstandard
    except ImportError as error:  # pragma: no cover - the exact runtime pins it
        raise DeconError("hermetic DECON requires the pinned zstandard runtime") from error
    return zstandard


def _full_shard_path(pa: Any, manifest_row: Mapping[str, Any]) -> Path:
    relative = _canonical_relative_path(manifest_row.get("relative_path"))
    shard = pa.root.joinpath(*PurePosixPath(relative).parts)
    assert_no_symlink_ancestors(shard)
    if not shard.is_file() or shard.is_symlink():
        raise DeconError("a governed full shard is absent")
    if shard.stat().st_size != manifest_row.get("zstd_bytes"):
        raise DeconError("a governed full-shard physical identity drifted")
    return shard


def _verify_full_shard_physical(pa: Any, manifest_row: Mapping[str, Any]) -> Path:
    shard = _full_shard_path(pa, manifest_row)
    if _sha256_file(shard) != manifest_row.get("zstd_sha256"):
        raise DeconError("a governed full-shard physical identity drifted")
    return shard


def _parse_full_record(
    raw_line: bytes, manifest_row: Mapping[str, Any]
) -> tuple[str, int]:
    if not raw_line.endswith(b"\n") or raw_line == b"\n":
        raise DeconError("full shard JSONL framing drifted")
    try:
        item = json.loads(
            raw_line.decode("utf-8", errors="strict"),
            object_pairs_hook=_json_no_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DeconError("full shard contains malformed JSON") from error
    item = _exact_fields(
        item,
        {"id", "source", "stratum", "text"},
        "full shard record",
    )
    if raw_line != canonical_json_bytes(item) + b"\n":
        raise DeconError("full shard JSONL is not canonically framed")
    if (
        not isinstance(item.get("text"), str)
        or item.get("source") != manifest_row.get("source")
        or item.get("stratum") != manifest_row.get("stratum")
        or not isinstance(item.get("id"), str)
        or _SHA1.fullmatch(str(item["id"])) is None
    ):
        raise DeconError("full shard record fields drifted")
    text = str(item["text"])
    try:
        text_bytes = text.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise DeconError("full shard record text is not valid UTF-8") from error
    if hashlib.sha1(text_bytes).hexdigest() != item["id"]:  # noqa: S324 - physical contract
        raise DeconError("full shard raw-content identity drifted")
    return text, len(text_bytes)


def _iter_bounded_jsonl(buffered: io.BufferedReader) -> Iterable[bytes]:
    """Yield framed records without permitting an unbounded ``readline``."""

    while True:
        raw_line = buffered.readline(DECON_MAX_RECORD_JSONL_BYTES + 1)
        if not raw_line:
            return
        if len(raw_line) > DECON_MAX_RECORD_JSONL_BYTES:
            raise DeconError("full shard record exceeds the governed byte ceiling")
        yield raw_line


def _utf8_prefix_within(text: str, byte_budget: int) -> tuple[str, int]:
    """Return the longest prefix whose strict UTF-8 form fits ``byte_budget``."""

    if type(byte_budget) is not int or byte_budget < 1:
        raise DeconError("DECON calibration byte budget is invalid")
    encoded = text.encode("utf-8", errors="strict")
    if len(encoded) <= byte_budget:
        return text, len(encoded)
    prefix = encoded[:byte_budget].decode("utf-8", errors="ignore")
    prefix_bytes = len(prefix.encode("utf-8", errors="strict"))
    if prefix_bytes < 1:
        raise DeconError("DECON calibration budget cannot hold one UTF-8 scalar")
    return prefix, prefix_bytes


def _calibrate_screening(
    *,
    pa: Any,
    confirm: _ConfirmIndex,
    eval_e: _EvalEIndex,
    fixed_elapsed_seconds: float,
) -> DeconCalibration:
    """Measure a bounded all-shard burst and reject an intractable full pass."""

    if not pa.full_shard_rows:
        raise DeconError("DECON calibration has no full shards")
    retained_values = tuple(
        row.get("retained_text_bytes") for row in pa.full_shard_rows
    )
    compressed_values = tuple(row.get("zstd_bytes") for row in pa.full_shard_rows)
    if any(type(value) is not int or value < 1 for value in retained_values):
        raise DeconError("full-shard retained-byte accounting drifted")
    if any(type(value) is not int or value < 1 for value in compressed_values):
        raise DeconError("full-shard compressed-byte accounting drifted")
    total_retained = sum(int(value) for value in retained_values)
    total_compressed = sum(int(value) for value in compressed_values)
    logical_target = min(DECON_CALIBRATION_LOGICAL_BYTES, total_retained)
    compressed_target = min(DECON_CALIBRATION_LOGICAL_BYTES, total_compressed)
    shard_count = len(pa.full_shard_rows)
    logical_base, logical_remainder = divmod(logical_target, shard_count)
    compressed_base, compressed_remainder = divmod(
        compressed_target, shard_count
    )
    logical_quotas = tuple(
        logical_base + (index < logical_remainder)
        for index in range(shard_count)
    )
    compressed_quotas = tuple(
        compressed_base + (index < compressed_remainder)
        for index in range(shard_count)
    )
    if min(logical_quotas) < 1 or min(compressed_quotas) < 1:
        raise DeconError("DECON calibration cannot sample every full shard")
    zstandard = _load_zstandard()
    paths = tuple(
        _full_shard_path(pa, manifest_row)
        for manifest_row in pa.full_shard_rows
    )

    compressed_sample_bytes = 0
    hash_started = time.perf_counter()
    for shard, quota in zip(paths, compressed_quotas, strict=True):
        with shard.open("rb") as handle:
            sample = handle.read(quota)
        if not sample:
            raise DeconError("DECON compressed calibration sampled no bytes")
        hashlib.sha256(sample).digest()
        compressed_sample_bytes += len(sample)
    hash_elapsed = time.perf_counter() - hash_started

    logical_sample_bytes = 0
    screen_started = time.perf_counter()
    for manifest_row, shard, quota in zip(
        pa.full_shard_rows, paths, logical_quotas, strict=True
    ):
        shard_sample = 0
        try:
            with shard.open("rb") as compressed:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                    with io.BufferedReader(reader) as buffered:
                        for raw_line in _iter_bounded_jsonl(buffered):
                            text, text_bytes = _parse_full_record(raw_line, manifest_row)
                            remaining = quota - shard_sample
                            sample_text, sample_text_bytes = _utf8_prefix_within(
                                text, remaining
                            )
                            confirm.match(sample_text)
                            eval_e.match(sample_text)
                            logical_sample_bytes += sample_text_bytes
                            shard_sample += sample_text_bytes
                            if shard_sample >= quota:
                                break
        except (OSError, zstandard.ZstdError) as error:
            raise DeconError("DECON calibration could not read a full shard") from error
    screen_elapsed = time.perf_counter() - screen_started
    if (
        compressed_sample_bytes < 1
        or logical_sample_bytes < 1
        or hash_elapsed <= 0.0
        or screen_elapsed <= 0.0
    ):
        raise DeconError("DECON calibration produced no measurable work")
    calibration_elapsed = hash_elapsed + screen_elapsed
    projected_hash_seconds = (
        hash_elapsed * total_compressed / compressed_sample_bytes
    )
    projected_screen_seconds = (
        screen_elapsed * total_retained / logical_sample_bytes
    )
    projected_seconds = (
        fixed_elapsed_seconds
        + calibration_elapsed
        + projected_hash_seconds
        + projected_screen_seconds
    )
    if projected_seconds > DECON_MAX_PROJECTED_SECONDS:
        raise DeconError("DECON measured projection exceeds the twelve-hour ceiling")
    return DeconCalibration(
        compressed_sample_bytes=compressed_sample_bytes,
        logical_sample_bytes=logical_sample_bytes,
        projected_seconds=projected_seconds,
        shard_count=len(paths),
    )


def _screen_full_shards(
    *,
    pa: Any,
    confirm: _ConfirmIndex,
    eval_e: _EvalEIndex,
    deadline_monotonic: float | None = None,
) -> tuple[int, int, int]:
    """Stream and re-verify every physical V4 full shard exactly once."""

    zstandard = _load_zstandard()

    exact_match_count = 0
    near_match_count = 0
    total_documents = 0
    for manifest_row in pa.full_shard_rows:
        shard = _verify_full_shard_physical(pa, manifest_row)
        logical_sha = hashlib.sha256()
        logical_bytes = 0
        retained_bytes = 0
        record_count = 0
        try:
            with shard.open("rb") as compressed:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                    with io.BufferedReader(reader) as buffered:
                        for raw_line in _iter_bounded_jsonl(buffered):
                            if (
                                deadline_monotonic is not None
                                and time.monotonic() > deadline_monotonic
                            ):
                                raise DeconError(
                                    "DECON full scan exceeded the twelve-hour runtime ceiling"
                                )
                            logical_sha.update(raw_line)
                            logical_bytes += len(raw_line)
                            text, text_byte_count = _parse_full_record(
                                raw_line, manifest_row
                            )
                            confirm_exact, confirm_near = confirm.match(text)
                            eval_exact, eval_near = eval_e.match(text)
                            if confirm_exact or eval_exact:
                                exact_match_count += 1
                            elif confirm_near or eval_near:
                                near_match_count += 1
                            retained_bytes += text_byte_count
                            record_count += 1
                            total_documents += 1
        except (OSError, zstandard.ZstdError) as error:
            raise DeconError("full shard decompression or UTF-8 validation failed") from error
        if (
            logical_sha.hexdigest() != manifest_row.get("logical_jsonl_sha256")
            or logical_bytes != manifest_row.get("logical_jsonl_bytes")
            or retained_bytes != manifest_row.get("retained_text_bytes")
            or record_count != manifest_row.get("record_count")
        ):
            raise DeconError("full shard logical accounting drifted")
    expected_documents = sum(int(row["record_count"]) for row in pa.full_shard_rows)
    if total_documents != expected_documents:
        raise DeconError("full shard screening coverage drifted")
    if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
        raise DeconError("DECON full scan exceeded the twelve-hour runtime ceiling")
    return exact_match_count, near_match_count, total_documents


def _network_probe() -> str:
    import socket

    probe = socket.socket()
    probe.settimeout(0.1)
    try:
        probe.connect(("1.1.1.1", 53))
    except Exception as error:  # the injected guard owns the exact exception type
        if error.__class__.__name__ != "Weft1NetworkDisabledError":
            raise DeconError("Python network guard is not the parent-injected guard") from error
        return NETWORK_PROBE_RESULT
    finally:
        probe.close()
    raise DeconError("hermetic DECON network probe unexpectedly connected")


def _screen_code_commitments() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative in sorted(DECON_CODE_RELATIVE_PATHS):
        path = REPOSITORY_ROOT.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink():
            raise DeconError("a DECON code file is absent")
        rows.append({"relative_path": relative, "sha256": _sha256_file(path)})
    return rows


def _commitment(value: object) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _exclusive_canonical_json(path: Path, value: Mapping[str, Any]) -> str:
    encoded = canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise DeconError("DECON output receipt already exists") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(encoded)


def _attest_runtime_against_pa(
    pa: Any,
) -> tuple[dict[str, str], Path, Path]:
    """Re-attest this child and match it to P-A's governed environment."""

    from training import weft1_corpus_replay_a2 as replay_v3
    from training.weft1_corpus_pa import attest_runtime_v3
    from training.weft1_corpus_replay_a3 import (
        PRODUCTION_DEPENDENCY_LOCK_PATH_V4,
    )

    provenance_path = pa.root / replay_v3.GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3
    runtime_path = pa.root / replay_v3.RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1
    provenance_raw, provenance = _load_json(
        provenance_path, "P-A global execution provenance"
    )
    if provenance_raw != canonical_json_bytes(provenance) + b"\n":
        raise DeconError("P-A global execution provenance is not canonical")
    try:
        validated = replay_v3.validate_global_execution_provenance_v3(provenance)
        attestation = attest_runtime_v3(
            requirements_lock=PRODUCTION_DEPENDENCY_LOCK_PATH_V4,
            executable=Path(sys.executable),
        )
        runtime_sha = _sha256_file(runtime_path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise DeconError("DECON runtime attestation failed closed") from error
    if (
        attestation.executable_sha256
        != validated.get("python_executable_sha256")
        or attestation.dependency_lock_sha256
        != validated.get("dependency_lock_sha256")
        or attestation.environment_identity_sha256
        != validated.get("environment_identity_sha256")
        or runtime_sha != validated.get("runtime_build_receipt_sha256")
    ):
        raise DeconError("DECON runtime differs from P-A governed provenance")
    return (
        {
            "global_execution_provenance_sha256": _sha256_bytes(provenance_raw),
            "python_executable_sha256": attestation.executable_sha256,
            "runtime_build_receipt_sha256": runtime_sha,
        },
        provenance_path,
        runtime_path,
    )


def run_hermetic_decon(
    *,
    materialization_root: Path,
    confirm_seal_paths: Sequence[Path],
    confirm_seal_ledger_path: Path,
    confirm_private_rows_path: Path,
    eval_e_index_path: Path,
    eval_e_lock_path: Path,
    output_root: Path,
) -> tuple[dict[str, object], str]:
    """Run one child screen and atomically emit its sole aggregate artifact."""

    run_started_monotonic = time.monotonic()
    # Import lazily: the P-B minter imports the non-secret contract without
    # importing sealed readers, and this avoids a module-level circular import.
    from training import weft1_corpus_pb as pb

    if (
        os.environ.get("WEFT1_NETWORK_DISABLED") != "1"
        or os.environ.get("WEFT1_NETWORK_GUARD_ACTIVE") != "1"
        or _network_probe() != NETWORK_PROBE_RESULT
    ):
        raise DeconError("hermetic DECON child lacks network isolation")
    root = assert_no_symlink_ancestors(output_root)
    if root.exists() or root.is_symlink() or not root.parent.is_dir():
        raise DeconError("DECON output root must be a fresh child of an existing directory")

    pa = pb.inspect_pa_v4(materialization_root)
    profiles = algorithm_profiles()
    if tuple(pb.DECON_REQUIRED_BATTERIES) != DECON_BATTERIES:
        raise DeconError("P-B battery registry and DECON contract disagree")

    code_before = _screen_code_commitments()
    eval_e, eval_commitments = _load_eval_e_index(
        index_path=eval_e_index_path, lock_path=eval_e_lock_path
    )
    confirm, confirm_commitments, _seal_projection_sha = _validate_confirm_membership(
        seal_paths=confirm_seal_paths,
        seal_ledger_path=confirm_seal_ledger_path,
        private_rows_path=confirm_private_rows_path,
        salt=eval_e.salt,
    )
    input_commitments: dict[str, str] = {
        **confirm_commitments,
        **eval_commitments,
    }
    input_commitments["private_input_set_commitment_sha256"] = _commitment(
        input_commitments
    )

    runtime_base, provenance_path, runtime_path = _attest_runtime_against_pa(pa)
    runtime_commitments = {
        **runtime_base,
        "network_guard_sha256": _require_sha256(
            os.environ.get("WEFT1_NETWORK_GUARD_SHA256"), "network guard SHA-256"
        ),
        "unshare_executable_sha256": _require_sha256(
            os.environ.get("WEFT1_DECON_UNSHARE_SHA256"), "unshare SHA-256"
        ),
    }
    _calibrate_screening(
        pa=pa,
        confirm=confirm,
        eval_e=eval_e,
        fixed_elapsed_seconds=time.monotonic() - run_started_monotonic,
    )
    exact_count, near_count, screened_documents = _screen_full_shards(
        pa=pa,
        confirm=confirm,
        eval_e=eval_e,
        deadline_monotonic=(
            run_started_monotonic + DECON_MAX_RUNTIME_SECONDS
        ),
    )

    # Re-read every private and code input after the 38 GB stream.  A mutation
    # can only fail the job; it cannot be hidden by a pre-run commitment.
    if _screen_code_commitments() != code_before:
        raise DeconError("DECON code changed during the hermetic screen")
    expected_private = dict(input_commitments)
    expected_private.pop("private_input_set_commitment_sha256")
    observed_private = {
        "confirm_complete_ledger_sha256": (
            GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256
        ),
        "confirm_private_rows_sha256": _sha256_file(confirm_private_rows_path),
        "confirm_seal_file_set_sha256": _commitment(
            [
                {
                    "battery": str(json.loads(path.read_text(encoding="utf-8"))["battery"]),
                    "sha256": _sha256_file(path),
                }
                for path in sorted(
                    confirm_seal_paths,
                    key=lambda value: ALL_BATTERIES.index(
                        str(json.loads(value.read_text(encoding="utf-8"))["battery"])
                    ),
                )
            ]
        ),
        "confirm_seal_ledger_sha256": _sha256_file(confirm_seal_ledger_path),
        "confirm_source_manifest_sha256": (
            GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256
        ),
        "eval_e_anonymous_index_sha256": _sha256_file(eval_e_index_path),
        "eval_e_lock_sha256": _sha256_file(eval_e_lock_path),
    }
    if observed_private != expected_private:
        raise DeconError("a sealed DECON input changed during screening")
    if (
        _sha256_file(provenance_path)
        != runtime_commitments["global_execution_provenance_sha256"]
        or _sha256_file(runtime_path)
        != runtime_commitments["runtime_build_receipt_sha256"]
    ):
        raise DeconError("P-A runtime provenance changed during screening")

    screened_shards = [
        {
            "relative_path": str(row["relative_path"]),
            "zstd_sha256": str(row["zstd_sha256"]),
        }
        for row in pa.full_shard_rows
    ]
    registry_commitment = _commitment(
        {
            "algorithm_profiles": profiles,
            "input_commitments": input_commitments,
            "registered_battery_count": len(DECON_BATTERIES),
        }
    )
    receipt: dict[str, object] = {
        "algorithm_profiles": profiles,
        "authority_chain": list(pb.PB_AUTHORITY_CHAIN_V5),
        "battery_scope": list(DECON_BATTERIES),
        "corpus_content_identity_sha256": pa.content_identity_sha256,
        "corpus_manifest_sha256": pa.content_manifest_physical_sha256,
        "exact_match_count": exact_count,
        "full_shard_manifest_identity_sha256": pa.full_shard_manifest_identity_sha256,
        "full_shard_manifest_sha256": pa.full_shard_manifest_physical_sha256,
        "hermetic": True,
        "hit_action": "HARD_STOP_NO_MINT",
        "input_commitments": input_commitments,
        "near_match_count": near_count,
        "network_accessed": False,
        "plaintext_exported": False,
        "registered_battery_count": len(DECON_BATTERIES),
        "release_manifest_section_identity_sha256": pa.release_manifest_section_identity_sha256,
        "runtime_commitments": runtime_commitments,
        "salt_exported": False,
        "schema": pb.PB_DECON_SCHEMA_V5,
        "screen_code_commitments": code_before,
        "screen_code_set_commitment_sha256": _commitment(code_before),
        "screen_submanifest_identity_sha256": pa.screen_submanifest_identity_sha256,
        "screen_submanifest_sha256": pa.screen_submanifest_physical_sha256,
        "screened_battery_count": len(DECON_BATTERIES),
        "screened_battery_set_commitment_sha256": _commitment(DECON_BATTERIES),
        "screened_document_count": screened_documents,
        "screened_full_shard_count": len(screened_shards),
        "screened_full_shard_set_commitment_sha256": _commitment(screened_shards),
        "screened_full_shards": screened_shards,
        "sealed_battery_registry_commitment_sha256": registry_commitment,
        "sealed_identifiers_exported": False,
        "status": "HIT" if exact_count + near_count else "CLEAN",
        "total_match_count": exact_count + near_count,
    }
    body = dict(receipt)
    receipt["receipt_sha256"] = pb.pb_authority_bound_sha256(
        pb.PB_DECON_SCHEMA_V5, body
    )

    root.mkdir()
    receipt_path = root / DECON_RECEIPT_FILENAME
    physical = _exclusive_canonical_json(receipt_path, receipt)
    if tuple(path.name for path in root.iterdir()) != (DECON_RECEIPT_FILENAME,):
        raise DeconError("hermetic DECON emitted a non-receipt artifact")
    # The P-B loader is the final shape/identity authority.
    loaded = pb.load_hermetic_decon_receipt(receipt_path, pa=pa)
    if loaded != (physical, receipt["receipt_sha256"], receipt["status"]):
        raise DeconError("P-B rejected the freshly emitted DECON receipt")
    return receipt, physical


def _resolved_regular(path: Path, name: str) -> Path:
    try:
        lexical = assert_no_symlink_ancestors(path)
        resolved = lexical.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise DeconError(f"{name} cannot be resolved safely") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise DeconError(f"{name} must be a regular non-symlink file")
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _parent_snapshot(
    *,
    pa: Any,
    seal_paths: Sequence[Path],
    seal_ledger_path: Path,
    private_rows_path: Path,
    eval_e_index_path: Path,
    eval_e_lock_path: Path,
) -> dict[str, object]:
    def shard_stat(row: Mapping[str, object]) -> dict[str, object]:
        path = _full_shard_path(pa, row)
        try:
            observed = path.stat()
        except OSError as error:
            raise DeconError("DECON parent cannot stat a full shard") from error
        return {
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "mtime_ns": observed.st_mtime_ns,
            "relative_path": str(row["relative_path"]),
            "size": observed.st_size,
            "zstd_sha256": str(row["zstd_sha256"]),
        }

    return {
        "code": _screen_code_commitments(),
        "content_identity": pa.content_identity_sha256,
        "d1_identity": pa.d1_ready_identity_sha256,
        "eval_e_index": _sha256_file(eval_e_index_path),
        "eval_e_lock": _sha256_file(eval_e_lock_path),
        "full_manifest": pa.full_shard_manifest_physical_sha256,
        "private_rows": _sha256_file(private_rows_path),
        "seal_ledger": _sha256_file(seal_ledger_path),
        "seals": [_sha256_file(path) for path in seal_paths],
        # The child hashes every full compressed shard while streaming it.  A
        # parent byte hash here would add 76 GB of redundant I/O around the
        # child and evade calibration.  Manifest identity plus stable stat
        # identity before/after instead detects concurrent replacement; the
        # child's governed digest is the authoritative byte check.
        "shards": [shard_stat(row) for row in pa.full_shard_rows],
    }


def launch_hermetic_decon(
    *,
    materialization_root: Path,
    confirm_seal_paths: Sequence[Path],
    confirm_seal_ledger_path: Path,
    confirm_private_rows_path: Path,
    eval_e_index_path: Path,
    eval_e_lock_path: Path,
    output_root: Path,
    local_work_parent: Path,
    python_executable: Path | None = None,
    unshare_executable: Path | None = None,
    timeout_seconds: int = DECON_PARENT_WATCHDOG_SECONDS,
) -> tuple[str, str, str]:
    """Parent-launch one production child under a verified Linux net namespace."""

    launch_started_monotonic = time.monotonic()

    from training import weft1_corpus_pb as pb
    from training.weft1_corpus_replay_a2 import (
        LINUX_UNSHARE_PATH_V1,
        _ISOLATED_WORKER_BOOTSTRAP_SOURCE_V3,
        _NETWORK_GUARD_SOURCE,
        _resolve_python_executable,
        _resolve_unshare_executable,
        _verify_unshare_network_isolation,
    )

    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise DeconError("DECON timeout must be a positive exact integer")
    launch_deadline_monotonic = launch_started_monotonic + min(
        timeout_seconds, DECON_MAX_RUNTIME_SECONDS
    )
    executable = _resolve_python_executable(
        Path(sys.executable) if python_executable is None else python_executable
    )
    unshare = _resolve_unshare_executable(
        LINUX_UNSHARE_PATH_V1 if unshare_executable is None else unshare_executable
    )
    _verify_unshare_network_isolation(
        unshare_executable=unshare, python_executable=executable
    )

    materialization = assert_no_symlink_ancestors(materialization_root).resolve(strict=True)
    seal_files = tuple(
        _resolved_regular(path, "CONFIRM seal file") for path in confirm_seal_paths
    )
    ledger = _resolved_regular(confirm_seal_ledger_path, "CONFIRM seal ledger")
    private_rows = _resolved_regular(confirm_private_rows_path, "CONFIRM private rows")
    eval_index = _resolved_regular(eval_e_index_path, "EVAL-E anonymous index")
    eval_lock = _resolved_regular(eval_e_lock_path, "EVAL-E lock")
    local_parent = assert_no_symlink_ancestors(local_work_parent).resolve(strict=True)
    if not local_parent.is_dir():
        raise DeconError("DECON local-work parent must already exist")
    output = assert_no_symlink_ancestors(output_root).resolve(strict=False)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise DeconError("DECON output root must be fresh")
    protected = (materialization, ledger, private_rows, eval_index, eval_lock, *seal_files)
    if any(_is_within(path, output) or _is_within(output, path) for path in protected):
        raise DeconError("DECON output and governed inputs are not disjoint")
    if _is_within(local_parent, output) or _is_within(output, local_parent):
        raise DeconError("DECON durable output and local work are not disjoint")
    if any(
        _is_within(local_parent, path) or _is_within(path, local_parent)
        for path in protected
    ):
        raise DeconError("DECON local work and governed inputs are not disjoint")

    pa_before = pb.inspect_pa_v4(materialization)
    snapshot_before = _parent_snapshot(
        pa=pa_before,
        seal_paths=seal_files,
        seal_ledger_path=ledger,
        private_rows_path=private_rows,
        eval_e_index_path=eval_index,
        eval_e_lock_path=eval_lock,
    )
    guard_sha = _sha256_bytes(_NETWORK_GUARD_SOURCE)
    unshare_sha = _sha256_file(unshare)
    worker = REPOSITORY_ROOT.joinpath(*PurePosixPath(DECON_WORKER_RELATIVE_PATH).parts)
    arguments = [
        str(worker),
        "child",
        "--materialization-root",
        str(materialization),
        "--confirm-seal-ledger",
        str(ledger),
        "--confirm-private-rows",
        str(private_rows),
        "--eval-e-index",
        str(eval_index),
        "--eval-e-lock",
        str(eval_lock),
        "--output-root",
        str(output),
    ]
    for path in seal_files:
        arguments.extend(("--confirm-seal", str(path)))
    environment = {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.defpath,
        "PIP_NO_INDEX": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "TZ": "UTC",
        "WEFT1_DECON_UNSHARE_SHA256": unshare_sha,
        "WEFT1_NETWORK_DISABLED": "1",
        "WEFT1_NETWORK_GUARD_ACTIVE": "0",
        "WEFT1_NETWORK_GUARD_SHA256": guard_sha,
    }
    with tempfile.TemporaryDirectory(
        prefix="weft1-decon-guard-", dir=local_parent
    ) as raw_guard:
        guard = Path(raw_guard) / "sitecustomize.py"
        guard.write_bytes(_NETWORK_GUARD_SOURCE)
        if _sha256_file(guard) != guard_sha:
            raise DeconError("parent network guard bytes drifted")
        remaining_seconds = launch_deadline_monotonic - time.monotonic()
        if remaining_seconds <= 0.0:
            raise DeconError("DECON parent preflight exceeded the total runtime ceiling")
        try:
            process = subprocess.run(
                (
                    str(unshare),
                    "--net",
                    "--",
                    str(executable),
                    "-I",
                    "-B",
                    "-c",
                    _ISOLATED_WORKER_BOOTSTRAP_SOURCE_V3,
                    str(guard),
                    str(REPOSITORY_ROOT),
                    *arguments,
                ),
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=remaining_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise DeconError(
                "hermetic DECON exceeded the total twelve-hour runtime ceiling"
            ) from error
    if process.returncode != 0:
        # Never relay child output: a future dependency must not turn sealed
        # content into an exception or stderr exfiltration channel.
        raise DeconError("hermetic DECON child failed; output suppressed")
    receipt_path = output / DECON_RECEIPT_FILENAME
    if not output.is_dir() or tuple(path.name for path in output.iterdir()) != (
        DECON_RECEIPT_FILENAME,
    ):
        raise DeconError("hermetic DECON child output is not exclusive")

    pa_after = pb.inspect_pa_v4(materialization)
    snapshot_after = _parent_snapshot(
        pa=pa_after,
        seal_paths=seal_files,
        seal_ledger_path=ledger,
        private_rows_path=private_rows,
        eval_e_index_path=eval_index,
        eval_e_lock_path=eval_lock,
    )
    if snapshot_after != snapshot_before:
        raise DeconError("DECON parent detected a concurrent input mutation")
    if time.monotonic() > launch_deadline_monotonic:
        raise DeconError("DECON parent exceeded the total twelve-hour runtime ceiling")
    physical, identity, status = pb.load_hermetic_decon_receipt(
        receipt_path, pa=pa_after
    )
    return physical, identity, status


__all__ = [
    "DECON_CALIBRATION_LOGICAL_BYTES",
    "DECON_CODE_RELATIVE_PATHS",
    "DECON_MAX_PROJECTED_SECONDS",
    "DECON_MAX_RECORD_JSONL_BYTES",
    "DECON_MAX_RUNTIME_SECONDS",
    "DECON_PARENT_WATCHDOG_SECONDS",
    "DECON_RECEIPT_FILENAME",
    "DeconCalibration",
    "DeconError",
    "launch_hermetic_decon",
    "legacy_minhash_signature",
    "render_confirm_prompt",
    "run_hermetic_decon",
]
