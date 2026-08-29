"""Literal A2 tokenizer construction and replay receipts for WEFT-1.

The tokenizer artifact is the exact UTF-8 ``tokenizer.json`` byte string.
This module never selects a vocabulary arm and never trains a language model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Iterable, Iterator, Mapping, Sequence

import zstandard
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from training.weft1_corpus_a2 import JsonlZstdShardIdentityV3
from training.weft1_gtok_a1_contract import SOURCE_FAMILIES
from training.weft1_gtok_contract import (
    GTOK_PRETOKENIZER_REGEX,
    GTOK_STRATA,
    GTOK_VOCABULARY_ARMS,
    canonical_json_bytes,
)
from training.weft1_strict_io import (
    assert_no_symlink_ancestors,
    load_canonical_json_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
BINDINGS_PATH = Path(__file__).with_name("weft1_corpus_gtok_a2_bindings_20260828.json")
BINDINGS_SHA256 = "ee10e69a3ccd55f7960949f4c318daa4db1197c779f5e88fb67cec82ab7f263b"
TOKENIZERS_VERSION = "0.22.2"
ZSTANDARD_VERSION = "0.25.0"
I32_MAX = 2_147_483_647
I32_REVIEW_THRESHOLD = I32_MAX * 4 // 5
U64_MAX = 18_446_744_073_709_551_615
_RECEIPT_FACTORY_SENTINEL = object()


def _sha256(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise TypeError("SHA-256 input must be exact bytes")
    return hashlib.sha256(value).hexdigest()


def _update_text_stream_digest(
    digest: object,
    text: str,
) -> bytes:
    """Frame one exact UTF-8 text as ``u64be length || bytes`` for replay."""

    if not isinstance(text, str):
        raise TypeError("A2 tokenizer-fit stream must contain strings")
    encoded = text.encode("utf-8", errors="strict")
    if len(encoded) > U64_MAX:
        raise OverflowError("one tokenizer-fit text exceeds uint64 framing")
    digest.update(len(encoded).to_bytes(8, "big"))  # type: ignore[attr-defined]
    digest.update(encoded)  # type: ignore[attr-defined]
    return encoded


def load_a2_bindings(
    path: Path | None = None,
    *,
    nonproduction_fixture: bool = False,
) -> Mapping[str, object]:
    """Load A2 bindings; only the no-argument checked-in path is production.

    Explicit alternate paths exist solely for non-production parser fixtures.
    The production path hashes and parses one immutable byte snapshot, so a
    file replacement cannot split the identity check from the consumed value.
    """

    if type(nonproduction_fixture) is not bool:
        raise TypeError("nonproduction_fixture must be an exact bool")
    if path is None and nonproduction_fixture:
        raise ValueError("the checked-in production path may not use fixture mode")
    if path is not None and not nonproduction_fixture:
        raise ValueError(
            "alternate A2 binding paths require explicit non-production fixture mode"
        )
    production_load = path is None
    selected_path = BINDINGS_PATH if production_load else path
    if not isinstance(selected_path, Path):
        raise TypeError("A2 binding path must be a pathlib.Path")
    raw, value = load_canonical_json_snapshot(selected_path)
    if production_load and _sha256(raw) != BINDINGS_SHA256:
        raise RuntimeError(
            "checked-in A2 binding ledger SHA-256 differs from authority"
        )
    if value.get("schema") != "weft1_corpus_gtok_a2_bindings_v3":
        raise ValueError("unexpected A2 binding ledger")
    return value


def require_tokenizer_runtime() -> None:
    observed = importlib.metadata.version("tokenizers")
    if observed != TOKENIZERS_VERSION:
        raise RuntimeError(
            f"A2 requires tokenizers=={TOKENIZERS_VERSION}, observed {observed}"
        )
    observed_zstd = importlib.metadata.version("zstandard")
    if observed_zstd != ZSTANDARD_VERSION:
        raise RuntimeError(
            f"A2 requires zstandard=={ZSTANDARD_VERSION}, observed {observed_zstd}"
        )


def special_token_strings() -> tuple[str, ...]:
    bindings = load_a2_bindings()
    tokenizer_binding = bindings["tokenizer"]
    if not isinstance(tokenizer_binding, dict):
        raise TypeError("A2 tokenizer binding must be an object")
    values = tokenizer_binding["special_tokens"]
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise TypeError("A2 special-token inventory must be a string list")
    result = tuple(values)
    if len(result) != 64 or len(set(result)) != len(result):
        raise ValueError("A2 requires exactly 64 distinct protocol tokens")
    return result


def _added_special_tokens() -> list[AddedToken]:
    return [
        AddedToken(
            value,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=True,
        )
        for value in special_token_strings()
    ]


def new_a2_tokenizer() -> Tokenizer:
    """Construct the exact unfitted A2 tokenizer graph."""

    require_tokenizer_runtime()
    tokenizer = Tokenizer(
        models.BPE(
            dropout=None,
            unk_token=None,
            fuse_unk=False,
            byte_fallback=False,
            ignore_merges=False,
        )
    )
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(GTOK_PRETOKENIZER_REGEX),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False,
                trim_offsets=False,
                use_regex=False,
            ),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = None
    return tokenizer


def fit_a2_tokenizer(
    texts: Iterable[str],
    *,
    vocab_size: int,
    length: int | None = None,
    safety_receipt: BpeI32SafetyReceipt | None = None,
) -> bytes:
    """Fit once and return the canonical ``tokenizer.json`` bytes."""

    if type(vocab_size) is not int or vocab_size < 64 + 256:
        raise ValueError("vocabulary size must leave room for protocol and byte atoms")
    if vocab_size not in GTOK_VOCABULARY_ARMS and vocab_size > 4_096:
        raise ValueError("production fits require a registered G-TOK vocabulary arm")
    if safety_receipt is not None and not isinstance(
        safety_receipt, BpeI32SafetyReceipt
    ):
        raise TypeError("safety_receipt must be a BpeI32SafetyReceipt")
    production_fit = vocab_size in GTOK_VOCABULARY_ARMS
    if production_fit:
        if safety_receipt is None:
            raise ValueError("production BPE fit requires its i32 safety receipt")
        if safety_receipt.status != "SAFE":
            raise RuntimeError(
                "stock tokenizers BPE fit is blocked by the signed-i32 count bound"
            )
    if length is not None and (type(length) is not int or length < 1):
        raise ValueError("iterator length must be a positive exact integer or None")
    if safety_receipt is not None and length != safety_receipt.text_count:
        raise ValueError(
            "BPE fit length must equal the scanned safety-receipt text count"
        )

    stream_verified = safety_receipt is None

    def verified_texts() -> Iterator[str]:
        nonlocal stream_verified
        digest = hashlib.sha256()
        observed_count = 0
        for text_value in texts:
            _update_text_stream_digest(digest, text_value)
            observed_count += 1
            if observed_count > U64_MAX:
                raise OverflowError("A2 BPE fit text counter exceeded uint64")
            yield text_value
        assert safety_receipt is not None
        if observed_count != safety_receipt.text_count:
            raise RuntimeError("BPE fit text count differs from its safety scan")
        if digest.hexdigest() != safety_receipt.text_stream_sha256:
            raise RuntimeError("BPE fit text stream differs from its safety scan")
        stream_verified = True

    fit_texts: Iterable[str] = texts if safety_receipt is None else verified_texts()
    tokenizer = new_a2_tokenizer()
    initial_alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    if len(initial_alphabet) != 256 or len(set(initial_alphabet)) != 256:
        raise RuntimeError("ByteLevel.alphabet no longer exposes exactly 256 atoms")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=False,
        special_tokens=_added_special_tokens(),
        limit_alphabet=256,
        initial_alphabet=initial_alphabet,
    )
    previous = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        tokenizer.train_from_iterator(fit_texts, trainer=trainer, length=length)
    finally:
        if previous is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = previous
    if not stream_verified:
        raise RuntimeError("BPE trainer did not exhaust the safety-bound text stream")
    payload = tokenizer.to_str(pretty=False).encode("utf-8")
    validate_tokenizer_json(payload, expected_vocab_size=vocab_size)
    return payload


def _tokenizer_object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"tokenizer JSON repeats key: {key}")
        value[key] = item
    return value


def _load_strict_canonical_tokenizer(
    payload: bytes,
) -> tuple[dict[str, object], Tokenizer]:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("tokenizer artifact must contain exact bytes")
    if payload.startswith(b"\xef\xbb\xbf") or b"\x00" in payload:
        raise ValueError("tokenizer JSON must be UTF-8 without BOM or NUL")
    try:
        decoded = payload.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_tokenizer_object_without_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"tokenizer JSON uses non-finite constant: {constant}")
            ),
        )
    except ValueError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("tokenizer artifact is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("tokenizer JSON root must be an object")
    tokenizer = Tokenizer.from_str(decoded)
    canonical = tokenizer.to_str(pretty=False).encode("utf-8")
    if payload != canonical:
        raise ValueError("tokenizer JSON is not the canonical tokenizers serialization")
    return value, tokenizer


def validate_tokenizer_json(payload: bytes, *, expected_vocab_size: int) -> None:
    if type(expected_vocab_size) is not int:
        raise TypeError("expected vocabulary size must be an exact integer")
    value, tokenizer = _load_strict_canonical_tokenizer(payload)
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    if len(vocabulary) != expected_vocab_size:
        raise ValueError("tokenizer inventory does not equal the requested vocabulary size")
    for expected_id, token in enumerate(special_token_strings()):
        if vocabulary.get(token) != expected_id:
            raise ValueError("protocol token IDs do not follow the registered order")
    byte_atoms = set(pre_tokenizers.ByteLevel.alphabet())
    if not byte_atoms.issubset(vocabulary):
        raise ValueError("tokenizer does not contain all 256 ByteLevel atoms")
    expected_top_level = {
        "version",
        "truncation",
        "padding",
        "added_tokens",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
        "model",
    }
    if set(value) != expected_top_level:
        raise ValueError("tokenizer JSON top-level graph drifted")
    if value["version"] != "1.0" or value["truncation"] is not None:
        raise ValueError("A2 tokenizer version or truncation state drifted")
    if value["padding"] is not None or value["normalizer"] is not None:
        raise ValueError("A2 tokenizer padding or normalizer must be disabled")
    if value["post_processor"] is not None:
        raise ValueError("A2 tokenizer post-processor must be disabled")

    expected_added = [
        {
            "id": token_id,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token_id, token in enumerate(special_token_strings())
    ]
    if value["added_tokens"] != expected_added:
        raise ValueError("A2 AddedToken inventory or flags drifted")

    expected_pre_tokenizer = {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": GTOK_PRETOKENIZER_REGEX},
                "behavior": "Isolated",
                "invert": False,
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": False,
                "use_regex": False,
            },
        ],
    }
    if value["pre_tokenizer"] != expected_pre_tokenizer:
        raise ValueError("A2 pre-tokenizer graph or flags drifted")
    expected_decoder = {
        "type": "ByteLevel",
        "add_prefix_space": True,
        "trim_offsets": True,
        "use_regex": True,
    }
    if value["decoder"] != expected_decoder:
        raise ValueError("A2 ByteLevel decoder flags drifted")

    model = value.get("model")
    if not isinstance(model, dict):
        raise ValueError("A2 tokenizer must contain a BPE model")
    expected_model_flags = {
        "type": "BPE",
        "dropout": None,
        "unk_token": None,
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "byte_fallback": False,
        "ignore_merges": False,
    }
    if set(model) != {*expected_model_flags, "vocab", "merges"}:
        raise ValueError("A2 BPE model fields drifted")
    if any(model.get(key) != expected for key, expected in expected_model_flags.items()):
        raise ValueError("A2 BPE model flags drifted")
    if not isinstance(model["vocab"], dict) or not isinstance(model["merges"], list):
        raise ValueError("A2 BPE vocabulary or merge table is malformed")


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _checked_add(counter: Counter[object], key: object, amount: int = 1) -> None:
    updated = counter[key] + amount
    if updated > U64_MAX:
        raise OverflowError("A2 BPE safety counter exceeded uint64")
    counter[key] = updated


@dataclass(frozen=True, init=False)
class BpeI32SafetyReceipt:
    stream_manifest_sha256: str
    text_stream_sha256: str
    text_count: int
    total_initial_symbols: int
    total_adjacent_pairs: int
    maximum_symbol: str
    maximum_symbol_count: int
    maximum_pair: tuple[str, str] | None
    maximum_pair_count: int
    counter_ledger_sha256: str
    status: str

    def __new__(cls) -> BpeI32SafetyReceipt:
        raise TypeError("BpeI32SafetyReceipt is factory-minted after a complete scan")

    @classmethod
    def _validated(
        cls,
        *,
        stream_manifest_sha256: str,
        text_stream_sha256: str,
        text_count: int,
        total_initial_symbols: int,
        total_adjacent_pairs: int,
        maximum_symbol: str,
        maximum_symbol_count: int,
        maximum_pair: tuple[str, str] | None,
        maximum_pair_count: int,
        counter_ledger_sha256: str,
        status: str,
        sentinel: object,
    ) -> BpeI32SafetyReceipt:
        if sentinel is not _RECEIPT_FACTORY_SENTINEL:
            raise PermissionError("BPE safety receipts are factory-only")
        instance = object.__new__(cls)
        for name, value in (
            ("stream_manifest_sha256", stream_manifest_sha256),
            ("text_stream_sha256", text_stream_sha256),
            ("text_count", text_count),
            ("total_initial_symbols", total_initial_symbols),
            ("total_adjacent_pairs", total_adjacent_pairs),
            ("maximum_symbol", maximum_symbol),
            ("maximum_symbol_count", maximum_symbol_count),
            ("maximum_pair", maximum_pair),
            ("maximum_pair_count", maximum_pair_count),
            ("counter_ledger_sha256", counter_ledger_sha256),
            ("status", status),
        ):
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        _require_sha256(self.stream_manifest_sha256, "stream manifest identity")
        _require_sha256(self.text_stream_sha256, "text stream identity")
        _require_sha256(self.counter_ledger_sha256, "counter ledger identity")
        for name in (
            "text_count",
            "total_initial_symbols",
            "total_adjacent_pairs",
            "maximum_symbol_count",
            "maximum_pair_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > U64_MAX:
                raise ValueError(f"{name} must be an unsigned 64-bit integer")
        if self.status not in {"SAFE", "REVIEW_REQUIRED", "UNSAFE"}:
            raise ValueError("unknown BPE i32 safety status")
        expected_status = (
            "UNSAFE"
            if self.maximum_symbol_count > I32_MAX
            else "REVIEW_REQUIRED"
            if self.maximum_symbol_count >= I32_REVIEW_THRESHOLD
            else "SAFE"
        )
        if self.status != expected_status:
            raise ValueError("BPE i32 safety status does not match the measured bound")


def preflight_bpe_i32_counts(
    texts: Iterable[str],
    *,
    stream_manifest_sha256: str,
) -> BpeI32SafetyReceipt:
    """Bound every reachable BPE count using checked uint64 counters.

    Every later BPE token occurrence consumes a distinct occurrence of at least
    one initial ByteLevel symbol.  The maximum initial-symbol count therefore
    safely bounds every pair count held in tokenizers 0.22.2's signed-i32 map.
    """

    _require_sha256(stream_manifest_sha256, "stream manifest identity")
    pre_tokenizer = new_a2_tokenizer().pre_tokenizer
    if pre_tokenizer is None:
        raise RuntimeError("A2 pre-tokenizer graph is absent")
    symbols: Counter[object] = Counter()
    pairs: Counter[object] = Counter()
    total_symbols = 0
    total_pairs = 0
    text_count = 0
    text_stream_digest = hashlib.sha256()
    for text_value in texts:
        _update_text_stream_digest(text_stream_digest, text_value)
        text_count += 1
        if text_count > U64_MAX:
            raise OverflowError("A2 BPE text counter exceeded uint64")
        for token, _offset in pre_tokenizer.pre_tokenize_str(text_value):
            token_symbols = tuple(token)
            for symbol in token_symbols:
                _checked_add(symbols, symbol)
            for pair in zip(token_symbols, token_symbols[1:]):
                _checked_add(pairs, pair)
            total_symbols += len(token_symbols)
            total_pairs += max(0, len(token_symbols) - 1)
            if total_symbols > U64_MAX or total_pairs > U64_MAX:
                raise OverflowError("A2 BPE aggregate counter exceeded uint64")
    maximum_symbol, maximum_symbol_count = max(
        ((str(key), value) for key, value in symbols.items()),
        key=lambda row: (row[1], row[0]),
        default=("", 0),
    )
    maximum_pair_value, maximum_pair_count = max(
        ((key, value) for key, value in pairs.items()),
        key=lambda row: (row[1], row[0]),
        default=(None, 0),
    )
    maximum_pair = (
        None
        if maximum_pair_value is None
        else (str(maximum_pair_value[0]), str(maximum_pair_value[1]))
    )
    ledger = {
        "pairs": [
            [str(left), str(right), count]
            for (left, right), count in sorted(pairs.items())
        ],
        "stream_manifest_sha256": stream_manifest_sha256,
        "text_stream_sha256": text_stream_digest.hexdigest(),
        "symbols": [[str(symbol), count] for symbol, count in sorted(symbols.items())],
        "text_count": text_count,
        "total_adjacent_pairs": total_pairs,
        "total_initial_symbols": total_symbols,
    }
    status = (
        "UNSAFE"
        if maximum_symbol_count > I32_MAX
        else "REVIEW_REQUIRED"
        if maximum_symbol_count >= I32_REVIEW_THRESHOLD
        else "SAFE"
    )
    return BpeI32SafetyReceipt._validated(
        stream_manifest_sha256=stream_manifest_sha256,
        text_stream_sha256=text_stream_digest.hexdigest(),
        text_count=text_count,
        total_initial_symbols=total_symbols,
        total_adjacent_pairs=total_pairs,
        maximum_symbol=maximum_symbol,
        maximum_symbol_count=maximum_symbol_count,
        maximum_pair=maximum_pair,
        maximum_pair_count=maximum_pair_count,
        counter_ledger_sha256=_sha256(canonical_json_bytes(ledger)),
        status=status,
        sentinel=_RECEIPT_FACTORY_SENTINEL,
    )


def tokenizer_artifact_sha256(payload: bytes) -> str:
    _, tokenizer = _load_strict_canonical_tokenizer(payload)
    validate_tokenizer_json(
        payload,
        expected_vocab_size=len(tokenizer.get_vocab(with_added_tokens=True)),
    )
    return _sha256(payload)


def tokenizer_merges_sha256(payload: bytes) -> str:
    value, tokenizer = _load_strict_canonical_tokenizer(payload)
    validate_tokenizer_json(
        payload,
        expected_vocab_size=len(tokenizer.get_vocab(with_added_tokens=True)),
    )
    model = value.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("merges"), list):
        raise ValueError("tokenizer JSON has no literal BPE merge list")
    return _sha256(canonical_json_bytes(model["merges"]))


def tokenizer_inventory_sha256(payload: bytes) -> str:
    _, tokenizer = _load_strict_canonical_tokenizer(payload)
    validate_tokenizer_json(
        payload,
        expected_vocab_size=len(tokenizer.get_vocab(with_added_tokens=True)),
    )
    inventory = tuple(
        sorted(
            tokenizer.get_vocab(with_added_tokens=True).items(),
            key=lambda row: row[1],
        )
    )
    return _sha256(canonical_json_bytes(inventory))


def atomic_write_tokenizer(
    path: Path,
    payload: bytes,
    *,
    expected_vocab_size: int,
) -> str:
    if not isinstance(path, Path):
        raise TypeError("tokenizer path must be a pathlib.Path")
    validate_tokenizer_json(payload, expected_vocab_size=expected_vocab_size)
    lexical_path = assert_no_symlink_ancestors(path)
    lexical_path.parent.mkdir(parents=True, exist_ok=True)
    lexical_path = assert_no_symlink_ancestors(lexical_path)
    if lexical_path.exists():
        raise FileExistsError(f"tokenizer destination already exists: {lexical_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{lexical_path.name}.", dir=lexical_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link publishes the fully-fsynced inode without
        # the overwrite semantics of os.replace/os.rename.  A destination
        # created after the precheck therefore fails closed with EEXIST.
        assert_no_symlink_ancestors(lexical_path)
        os.link(temporary, lexical_path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256(payload)


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"A2 shard record repeats JSON key: {key}")
        value[key] = item
    return value


def iter_a2_shard_texts(
    shard_root: Path,
    shards: Sequence[JsonlZstdShardIdentityV3],
) -> Iterator[str]:
    """Verify registered shards completely, then yield their exact text fields."""

    require_tokenizer_runtime()
    if not isinstance(shard_root, Path):
        raise TypeError("shard_root must be a pathlib.Path")
    lexical_root = assert_no_symlink_ancestors(shard_root)
    resolved_root = lexical_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("shard_root must resolve to a directory")
    if not isinstance(shards, Sequence) or not shards:
        raise ValueError("at least one typed shard identity is required")
    if any(not isinstance(shard, JsonlZstdShardIdentityV3) for shard in shards):
        raise TypeError("shards contain a non-JsonlZstdShardIdentityV3 value")
    if len({shard.relative_path for shard in shards}) != len(shards):
        raise ValueError("shard identities repeat a relative path")
    for shard in shards:
        lexical_path = assert_no_symlink_ancestors(
            lexical_root.joinpath(*PurePosixPath(shard.relative_path).parts)
        )
        resolved_path = lexical_path.resolve(strict=True)
        if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
            raise ValueError("A2 shard resolves outside its registered root")
        if not resolved_path.is_file():
            raise ValueError("A2 shard must be a regular non-symlink file")
        if resolved_path.stat().st_size != shard.zstd_bytes:
            raise ValueError("A2 compressed shard byte count drifted")
        compressed_sha256 = hashlib.sha256()
        with resolved_path.open("rb") as compressed:
            for chunk in iter(lambda: compressed.read(8 * 1024 * 1024), b""):
                compressed_sha256.update(chunk)
        if compressed_sha256.hexdigest() != shard.zstd_sha256:
            raise ValueError("A2 compressed shard SHA-256 drifted")
        logical_sha256 = hashlib.sha256()
        logical_bytes = 0
        record_count = 0
        retained_text_bytes = 0
        with resolved_path.open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as decompressed:
                for raw_line in io.BufferedReader(decompressed):
                    if not raw_line.endswith(b"\n"):
                        raise ValueError("A2 shard record lacks its LF terminator")
                    logical_sha256.update(raw_line)
                    logical_bytes += len(raw_line)
                    row = json.loads(
                        raw_line[:-1].decode("utf-8", errors="strict"),
                        object_pairs_hook=_json_object_without_duplicate_keys,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"A2 shard uses non-finite JSON: {value}")
                        ),
                    )
                    if tuple(row) != ("id", "source", "stratum", "text"):
                        raise ValueError("A2 shard record keys or order drifted")
                    if row["source"] not in SOURCE_FAMILIES:
                        raise ValueError("A2 shard record uses an unknown source")
                    if row["stratum"] not in GTOK_STRATA:
                        raise ValueError("A2 shard record uses an unknown stratum")
                    if not shard.relative_path.startswith(f"{row['stratum']}/"):
                        raise ValueError("A2 shard record stratum differs from its path")
                    text = row["text"]
                    if not isinstance(text, str):
                        raise TypeError("A2 shard text must be a string")
                    encoded_text = text.encode("utf-8", errors="strict")
                    expected_id = hashlib.sha1(encoded_text).hexdigest()  # noqa: S324
                    if row["id"] != expected_id:
                        raise ValueError("A2 shard record ID differs from SHA-1(text)")
                    canonical = (
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                    if canonical != raw_line:
                        raise ValueError("A2 shard record is not canonical JSONL")
                    record_count += 1
                    retained_text_bytes += len(encoded_text)
                    yield text
        if record_count != shard.record_count:
            raise ValueError("A2 shard record count drifted")
        if retained_text_bytes != shard.retained_text_bytes:
            raise ValueError("A2 shard retained-text byte count drifted")
        if logical_bytes != shard.logical_jsonl_bytes:
            raise ValueError("A2 logical shard byte count drifted")
        if logical_sha256.hexdigest() != shard.logical_jsonl_sha256:
            raise ValueError("A2 logical shard SHA-256 drifted")


@dataclass(frozen=True)
class FitProcessAttestation:
    process_id: int
    output_root: str
    executable_sha256: str
    dependency_lock_sha256: str
    environment_identity_sha256: str
    input_manifest_sha256: str
    artifact_sha256: str
    merges_sha256: str
    vocab_size: int

    def __post_init__(self) -> None:
        if type(self.process_id) is not int or self.process_id < 1:
            raise ValueError("fit process ID must be a positive exact integer")
        if not isinstance(self.output_root, str) or not self.output_root:
            raise ValueError("fit output root must be a nonempty canonical string")
        output_root = Path(self.output_root)
        if not output_root.is_absolute() or str(
            output_root.resolve(strict=False)
        ) != self.output_root:
            raise ValueError("fit output root must be an absolute resolved path")
        if type(self.vocab_size) is not int or self.vocab_size < 320:
            raise ValueError("fit vocabulary size is invalid")
        for name in (
            "executable_sha256",
            "dependency_lock_sha256",
            "environment_identity_sha256",
            "input_manifest_sha256",
            "artifact_sha256",
            "merges_sha256",
        ):
            _require_sha256(getattr(self, name), name)


@dataclass(frozen=True, init=False)
class DoubleFitReceipt:
    vocab_size: int
    first_artifact_sha256: str
    second_artifact_sha256: str
    first_merges_sha256: str
    second_merges_sha256: str
    tokenizers_version: str
    authoritative: bool
    evidence_level: str
    first_process: FitProcessAttestation | None
    second_process: FitProcessAttestation | None

    def __new__(cls) -> DoubleFitReceipt:
        raise TypeError("DoubleFitReceipt is factory-minted after validation")

    @classmethod
    def _validated(
        cls,
        *,
        vocab_size: int,
        first_artifact_sha256: str,
        second_artifact_sha256: str,
        first_merges_sha256: str,
        second_merges_sha256: str,
        authoritative: bool,
        evidence_level: str,
        first_process: FitProcessAttestation | None,
        second_process: FitProcessAttestation | None,
        sentinel: object,
    ) -> DoubleFitReceipt:
        if sentinel is not _RECEIPT_FACTORY_SENTINEL:
            raise PermissionError("double-fit receipts are factory-only")
        if type(vocab_size) is not int:
            raise TypeError("vocabulary size must be an exact integer")
        for name, value in (
            ("first_artifact_sha256", first_artifact_sha256),
            ("second_artifact_sha256", second_artifact_sha256),
            ("first_merges_sha256", first_merges_sha256),
            ("second_merges_sha256", second_merges_sha256),
        ):
            _require_sha256(value, name)
        if first_artifact_sha256 != second_artifact_sha256:
            raise ValueError("A2 tokenizer double-fit artifact mismatch")
        if first_merges_sha256 != second_merges_sha256:
            raise ValueError("A2 tokenizer double-fit merge mismatch")
        if type(authoritative) is not bool:
            raise TypeError("authoritative must be a bool")
        if evidence_level not in {
            "DIAGNOSTIC_IN_MEMORY",
            "CLAIMED_PROCESS_METADATA",
            "PARENT_REHASHED_SUBPROCESSES",
        }:
            raise ValueError("unknown double-fit evidence level")
        if authoritative != (evidence_level == "PARENT_REHASHED_SUBPROCESSES"):
            raise ValueError("double-fit authority requires parent-rehashed subprocesses")
        if first_process is not None or second_process is not None:
            if first_process is None or second_process is None:
                raise ValueError("double fit requires both process attestations")
            if first_process.process_id == second_process.process_id:
                raise ValueError("double fit did not use distinct process IDs")
            first_root = Path(first_process.output_root)
            second_root = Path(second_process.output_root)
            if (
                first_root == second_root
                or first_root in second_root.parents
                or second_root in first_root.parents
            ):
                raise ValueError("double fit did not use distinct non-overlapping roots")
            for name in (
                "executable_sha256",
                "dependency_lock_sha256",
                "environment_identity_sha256",
                "input_manifest_sha256",
                "vocab_size",
            ):
                if getattr(first_process, name) != getattr(second_process, name):
                    raise ValueError(f"double-fit process {name} mismatch")
            if first_process.vocab_size != vocab_size:
                raise ValueError("double-fit attestation vocabulary size drifted")
            if first_process.artifact_sha256 != first_artifact_sha256:
                raise ValueError("first process artifact attestation mismatch")
            if second_process.artifact_sha256 != second_artifact_sha256:
                raise ValueError("second process artifact attestation mismatch")
            if first_process.merges_sha256 != first_merges_sha256:
                raise ValueError("first process merge attestation mismatch")
            if second_process.merges_sha256 != second_merges_sha256:
                raise ValueError("second process merge attestation mismatch")
        elif evidence_level != "DIAGNOSTIC_IN_MEMORY":
            raise ValueError("process evidence level requires process attestations")
        instance = object.__new__(cls)
        for name, value in (
            ("vocab_size", vocab_size),
            ("first_artifact_sha256", first_artifact_sha256),
            ("second_artifact_sha256", second_artifact_sha256),
            ("first_merges_sha256", first_merges_sha256),
            ("second_merges_sha256", second_merges_sha256),
            ("tokenizers_version", TOKENIZERS_VERSION),
            ("authoritative", authoritative),
            ("evidence_level", evidence_level),
            ("first_process", first_process),
            ("second_process", second_process),
        ):
            object.__setattr__(instance, name, value)
        return instance


def compare_double_fit(first: bytes, second: bytes, *, vocab_size: int) -> DoubleFitReceipt:
    """Compare two payloads diagnostically; this cannot mint production PASS."""

    validate_tokenizer_json(first, expected_vocab_size=vocab_size)
    validate_tokenizer_json(second, expected_vocab_size=vocab_size)
    return DoubleFitReceipt._validated(
        vocab_size=vocab_size,
        first_artifact_sha256=_sha256(first),
        second_artifact_sha256=_sha256(second),
        first_merges_sha256=tokenizer_merges_sha256(first),
        second_merges_sha256=tokenizer_merges_sha256(second),
        authoritative=False,
        evidence_level="DIAGNOSTIC_IN_MEMORY",
        first_process=None,
        second_process=None,
        sentinel=_RECEIPT_FACTORY_SENTINEL,
    )


def compare_independent_double_fit(
    first: bytes,
    second: bytes,
    *,
    vocab_size: int,
    first_process: FitProcessAttestation,
    second_process: FitProcessAttestation,
) -> DoubleFitReceipt:
    """Compare caller-supplied process claims without asserting parent proof."""

    validate_tokenizer_json(first, expected_vocab_size=vocab_size)
    validate_tokenizer_json(second, expected_vocab_size=vocab_size)
    return DoubleFitReceipt._validated(
        vocab_size=vocab_size,
        first_artifact_sha256=_sha256(first),
        second_artifact_sha256=_sha256(second),
        first_merges_sha256=tokenizer_merges_sha256(first),
        second_merges_sha256=tokenizer_merges_sha256(second),
        authoritative=False,
        evidence_level="CLAIMED_PROCESS_METADATA",
        first_process=first_process,
        second_process=second_process,
        sentinel=_RECEIPT_FACTORY_SENTINEL,
    )


__all__ = [
    "BINDINGS_PATH",
    "BINDINGS_SHA256",
    "BpeI32SafetyReceipt",
    "DoubleFitReceipt",
    "FitProcessAttestation",
    "I32_MAX",
    "I32_REVIEW_THRESHOLD",
    "TOKENIZERS_VERSION",
    "ZSTANDARD_VERSION",
    "atomic_write_tokenizer",
    "compare_double_fit",
    "compare_independent_double_fit",
    "fit_a2_tokenizer",
    "iter_a2_shard_texts",
    "load_a2_bindings",
    "new_a2_tokenizer",
    "preflight_bpe_i32_counts",
    "require_tokenizer_runtime",
    "special_token_strings",
    "tokenizer_artifact_sha256",
    "tokenizer_inventory_sha256",
    "tokenizer_merges_sha256",
    "validate_tokenizer_json",
]
