from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
import zstandard

import training.weft1_strict_io as strict_io
import training.weft1_gtok_tokenizer_a2 as tokenizer_module
from training.weft1_corpus_a2 import JsonlZstdShardIdentityV3
from training.weft1_gtok_tokenizer_a2 import (
    BINDINGS_PATH,
    BINDINGS_SHA256,
    DoubleFitReceipt,
    FitProcessAttestation,
    atomic_write_tokenizer,
    compare_double_fit,
    compare_independent_double_fit,
    fit_a2_tokenizer,
    iter_a2_shard_texts,
    load_a2_bindings,
    new_a2_tokenizer,
    preflight_bpe_i32_counts,
    require_tokenizer_runtime,
    special_token_strings,
    tokenizer_artifact_sha256,
    tokenizer_inventory_sha256,
    tokenizer_merges_sha256,
)


FIXTURE_ARTIFACT_SHA256 = (
    "96c8faf9f1931f481b91d869ce7c6bf1d2e1f808537731259f2dd16c91ee377b"
)
FIXTURE_MERGES_SHA256 = (
    "2d488ea8a2b91fcc4278c9a3cc3592d664e4bce41b90662f1fdf11493aa631da"
)
FIXTURE_INVENTORY_SHA256 = (
    "8a2d7707d37049b5b2ac9e40dfe5b9570288c7a4e724c77ab8a910175c71542a"
)


def _corpus() -> list[str]:
    return [
        f"alpha{index} beta{index * index} gamma_{index % 97} "
        f"δ{index % 31} 漢{index % 17} {chr(33 + index % 80)}"
        for index in range(5_000)
    ]


@pytest.fixture(scope="module")
def fitted() -> bytes:
    corpus = _corpus()
    return fit_a2_tokenizer(corpus, vocab_size=384, length=len(corpus))


def test_a2_tokenizer_runtime_inventory_and_graph_are_literal() -> None:
    require_tokenizer_runtime()
    bindings = load_a2_bindings()
    tokens = special_token_strings()
    assert len(tokens) == 64
    assert tokens[:16] == (
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|doc_boundary|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool|>",
        "<|tool_result|>",
        "<|developer|>",
        "<|analysis|>",
        "<|final|>",
    )
    assert bindings["tokenizer"]["post_processor"] is None
    assert bindings["tokenizer"]["protocol_token_ids"] == {
        "pad": 0,
        "bos": 1,
        "eos": 2,
        "document_boundary": 3,
        "fim_prefix": 4,
        "fim_middle": 5,
        "fim_suffix": 6,
        "fim_pad": 7,
        "system": 8,
        "user": 9,
        "assistant": 10,
        "tool": 11,
        "tool_result": 12,
        "developer": 13,
        "analysis": 14,
        "final": 15,
        "reserved_000_through_047": [16, 63],
    }
    tokenizer = new_a2_tokenizer()
    assert tokenizer.normalizer is None
    assert tokenizer.post_processor is None


def test_default_bindings_load_is_pinned_to_checked_in_bytes() -> None:
    assert hashlib.sha256(BINDINGS_PATH.read_bytes()).hexdigest() == BINDINGS_SHA256
    assert load_a2_bindings()["schema"] == "weft1_corpus_gtok_a2_bindings_v3"


def test_default_bindings_load_rejects_canonical_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tampered = tmp_path / BINDINGS_PATH.name
    original = BINDINGS_PATH.read_bytes()
    changed = original.replace(
        b'"production_surface": "Pharma Initiatives Colab Pro+ in-app browser"',
        b'"production_surface": "tampered but canonical fixture"',
        1,
    )
    assert changed != original
    tampered.write_bytes(changed)
    monkeypatch.setattr(tokenizer_module, "BINDINGS_PATH", tampered)
    with pytest.raises(RuntimeError, match="SHA-256 differs from authority"):
        load_a2_bindings()


def test_bindings_ledger_rejects_nested_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    # Supplying a path opts into the explicitly non-production parser-fixture
    # surface; production callers use the no-argument, SHA-pinned boundary.
    path = tmp_path / "duplicate-bindings-key.json"
    path.write_text(
        '{"schema":"weft1_corpus_gtok_a2_bindings_v3",'
        '"runtime":{"package":"first","package":"second"}}\n',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="explicit non-production fixture mode"):
        load_a2_bindings(path)
    with pytest.raises(ValueError, match="repeats key: package"):
        load_a2_bindings(path, nonproduction_fixture=True)


def test_double_fit_is_byte_identical_and_matches_golden(fitted: bytes) -> None:
    corpus = _corpus()
    second = fit_a2_tokenizer(iter(corpus), vocab_size=384, length=len(corpus))
    receipt = compare_double_fit(fitted, second, vocab_size=384)
    assert isinstance(receipt, DoubleFitReceipt)
    assert receipt.authoritative is False
    assert receipt.first_artifact_sha256 == FIXTURE_ARTIFACT_SHA256
    assert receipt.first_merges_sha256 == FIXTURE_MERGES_SHA256
    assert tokenizer_inventory_sha256(fitted) == FIXTURE_INVENTORY_SHA256


def test_bytelevel_round_trip_and_single_digit_split(fitted: bytes) -> None:
    tokenizer = Tokenizer.from_str(fitted.decode("utf-8"))
    fixtures = (
        "café naïve résumé",
        "Καλημέρα κόσμε",
        "漢字とかな",
        "مرحبا بالعالم",
        "typographic — ‘quotes’ …",
        "\t  mixed\r\n    indentation  ",
        "emoji 🧵🧠",
    )
    for fixture in fixtures:
        assert tokenizer.decode(tokenizer.encode(fixture).ids) == fixture
    assert len(tokenizer.encode("1234567890").ids) == 10
    assert tokenizer.token_to_id("<unk>") is None


def test_protocol_ids_are_exact_and_all_byte_atoms_survive(fitted: bytes) -> None:
    tokenizer = Tokenizer.from_str(fitted.decode("utf-8"))
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    assert len(vocabulary) == 384
    assert tuple(vocabulary[token] for token in special_token_strings()) == tuple(range(64))
    every_byte = bytes(range(256))
    round_trip = tokenizer.decode(tokenizer.encode(every_byte.decode("latin-1")).ids)
    assert round_trip == every_byte.decode("latin-1")


def test_double_fit_mismatch_fails_closed(fitted: bytes) -> None:
    value = json.loads(fitted)
    value["model"]["merges"] = value["model"]["merges"][:-1]
    changed = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with pytest.raises(ValueError, match="mismatch|inventory"):
        compare_double_fit(fitted, changed, vocab_size=384)


def test_shard_reader_requires_canonical_key_order_and_lf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = (
        {
            "id": hashlib.sha1(b"one").hexdigest(),  # noqa: S324 - A2 contract
            "source": "dolma_web",
            "stratum": "general",
            "text": "one",
        },
        {
            "id": hashlib.sha1("δύο".encode()).hexdigest(),  # noqa: S324
            "source": "fineweb_edu",
            "stratum": "general",
            "text": "δύο",
        },
    )
    logical = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    compressed = zstandard.ZstdCompressor(
        level=3,
        threads=0,
        write_checksum=True,
        write_content_size=False,
        write_dict_id=False,
    ).compress(logical)
    path = tmp_path / "general" / "fixture.jsonl.zst"
    path.parent.mkdir()
    path.write_bytes(compressed)
    identity = JsonlZstdShardIdentityV3(
        relative_path="general/fixture.jsonl.zst",
        record_count=2,
        retained_text_bytes=len("oneδύο".encode()),
        logical_jsonl_sha256=hashlib.sha256(logical).hexdigest(),
        logical_jsonl_bytes=len(logical),
        zstd_sha256=hashlib.sha256(compressed).hexdigest(),
        zstd_bytes=len(compressed),
    )
    assert tuple(iter_a2_shard_texts(tmp_path, (identity,))) == ("one", "δύο")

    alias = tmp_path / "alias"
    try:
        alias.symlink_to(tmp_path, target_is_directory=True)
    except (NotImplementedError, OSError):
        alias.mkdir()
        original_link_check = strict_io._is_link_or_reparse
        monkeypatch.setattr(
            strict_io,
            "_is_link_or_reparse",
            lambda candidate: candidate == alias or original_link_check(candidate),
        )
    with pytest.raises(ValueError, match="symlink/reparse"):
        tuple(iter_a2_shard_texts(alias, (identity,)))

    tampered_row = dict(rows[0])
    tampered_row["id"] = "0" * 40
    tampered_logical = (
        json.dumps(tampered_row, separators=(",", ":")).encode() + b"\n"
    )
    tampered_compressed = zstandard.ZstdCompressor(
        level=3,
        threads=0,
        write_checksum=True,
        write_content_size=False,
        write_dict_id=False,
    ).compress(tampered_logical)
    tampered_path = tmp_path / "general" / "tampered.jsonl.zst"
    tampered_path.write_bytes(tampered_compressed)
    tampered_identity = JsonlZstdShardIdentityV3(
        relative_path="general/tampered.jsonl.zst",
        record_count=1,
        retained_text_bytes=3,
        logical_jsonl_sha256=hashlib.sha256(tampered_logical).hexdigest(),
        logical_jsonl_bytes=len(tampered_logical),
        zstd_sha256=hashlib.sha256(tampered_compressed).hexdigest(),
        zstd_bytes=len(tampered_compressed),
    )
    with pytest.raises(ValueError, match="SHA-1"):
        tuple(iter_a2_shard_texts(tmp_path, (tampered_identity,)))


def test_fixture_hash_helpers_agree(fitted: bytes) -> None:
    assert tokenizer_artifact_sha256(fitted) == FIXTURE_ARTIFACT_SHA256
    assert tokenizer_merges_sha256(fitted) == FIXTURE_MERGES_SHA256


def test_tokenizer_identity_rejects_duplicate_and_noncanonical_json(
    fitted: bytes,
) -> None:
    duplicate = fitted.replace(
        b'"version":"1.0"',
        b'"version":"1.0","version":"1.0"',
        1,
    )
    with pytest.raises(ValueError, match="repeats key: version"):
        tokenizer_artifact_sha256(duplicate)

    noncanonical = fitted + b"\n"
    with pytest.raises(ValueError, match="canonical tokenizers serialization"):
        tokenizer_artifact_sha256(noncanonical)
    with pytest.raises(ValueError, match="canonical tokenizers serialization"):
        tokenizer_merges_sha256(noncanonical)
    with pytest.raises(ValueError, match="canonical tokenizers serialization"):
        tokenizer_inventory_sha256(noncanonical)


def test_atomic_tokenizer_publish_requires_claimed_size_and_fresh_destination(
    tmp_path: Path,
    fitted: bytes,
) -> None:
    destination = tmp_path / "freeze" / "tokenizer.json"
    with pytest.raises(ValueError, match="requested vocabulary size"):
        atomic_write_tokenizer(
            destination,
            fitted,
            expected_vocab_size=385,
        )
    assert not destination.exists()

    artifact_sha256 = atomic_write_tokenizer(
        destination,
        fitted,
        expected_vocab_size=384,
    )
    assert artifact_sha256 == FIXTURE_ARTIFACT_SHA256
    assert destination.read_bytes() == fitted
    with pytest.raises(FileExistsError, match="already exists"):
        atomic_write_tokenizer(
            destination,
            fitted,
            expected_vocab_size=384,
        )
    assert destination.read_bytes() == fitted


def test_validator_rejects_any_tokenizer_graph_flag_drift(fitted: bytes) -> None:
    value = json.loads(fitted)
    value["added_tokens"][0]["normalized"] = True
    changed = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="AddedToken"):
        tokenizer_artifact_sha256(changed)

    value = json.loads(fitted)
    value["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = True
    changed = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="pre-tokenizer"):
        tokenizer_artifact_sha256(changed)


def test_bpe_i32_preflight_has_checked_replayable_golden() -> None:
    manifest = hashlib.sha256(b"fixture").hexdigest()
    receipt = preflight_bpe_i32_counts(
        ["  x", "é"], stream_manifest_sha256=manifest
    )
    assert receipt.status == "SAFE"
    assert receipt.total_initial_symbols == 5
    assert receipt.total_adjacent_pairs == 2
    assert receipt.maximum_symbol == "Ġ"
    assert receipt.maximum_symbol_count == 2
    assert receipt.text_stream_sha256 == (
        "ffac1aed8b9c0e781aeaf7a06587f7ddf45f95f77b380fa7aadcdc5778d98a6e"
    )
    assert receipt.counter_ledger_sha256 == (
        "934af3e7675ad66636c540b4993c219b8f2e362beab1a971ea63c5367f2cb96f"
    )
    with pytest.raises(TypeError):
        type(receipt)()
    with pytest.raises(ValueError, match="i32 safety receipt"):
        fit_a2_tokenizer(["fixture"], vocab_size=16_384, length=1)


def test_bpe_fit_consumes_the_exact_stream_scanned_by_its_safety_receipt() -> None:
    manifest = hashlib.sha256(b"bound-stream").hexdigest()
    receipt = preflight_bpe_i32_counts(
        ["safe stream"], stream_manifest_sha256=manifest
    )
    payload = fit_a2_tokenizer(
        ["safe stream"],
        vocab_size=320,
        length=1,
        safety_receipt=receipt,
    )
    assert len(Tokenizer.from_str(payload.decode("utf-8")).get_vocab()) == 320
    with pytest.raises(RuntimeError, match="text stream differs"):
        fit_a2_tokenizer(
            ["different stream"],
            vocab_size=320,
            length=1,
            safety_receipt=receipt,
        )
    with pytest.raises(ValueError, match="length must equal"):
        fit_a2_tokenizer(
            ["safe stream"],
            vocab_size=320,
            length=2,
            safety_receipt=receipt,
        )


def test_authoritative_double_fit_requires_distinct_typed_processes(
    fitted: bytes,
) -> None:
    artifact = tokenizer_artifact_sha256(fitted)
    merges = tokenizer_merges_sha256(fitted)
    common = {
        "executable_sha256": hashlib.sha256(b"python").hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(b"lock").hexdigest(),
        "environment_identity_sha256": hashlib.sha256(b"environment").hexdigest(),
        "input_manifest_sha256": hashlib.sha256(b"input").hexdigest(),
        "artifact_sha256": artifact,
        "merges_sha256": merges,
        "vocab_size": 384,
    }
    first_root = str((Path.cwd() / ".test-fit-a").resolve())
    second_root = str((Path.cwd() / ".test-fit-b").resolve())
    first = FitProcessAttestation(process_id=101, output_root=first_root, **common)
    second = FitProcessAttestation(process_id=202, output_root=second_root, **common)
    receipt = compare_independent_double_fit(
        fitted,
        fitted,
        vocab_size=384,
        first_process=first,
        second_process=second,
    )
    assert receipt.authoritative is False
    assert receipt.evidence_level == "CLAIMED_PROCESS_METADATA"
    with pytest.raises(TypeError):
        DoubleFitReceipt()
    with pytest.raises(ValueError, match="process IDs"):
        compare_independent_double_fit(
            fitted,
            fitted,
            vocab_size=384,
            first_process=first,
            second_process=FitProcessAttestation(
                process_id=101,
                output_root=second_root,
                **common,
            ),
        )
