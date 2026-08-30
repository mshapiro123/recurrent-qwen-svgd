from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from training.weft1_gtok_tokenizer_a2 import fit_a2_tokenizer
from training.weft1_gtok_offline_v2 import GTokOfflineV2Error, OFFLINE_RECEIPT_ENV_V2
from training.weft1_gtok_tokenizer_v2 import (
    FitWorkerReceiptV2,
    GTokTokenizerV2Error,
    _isolated_worker_environment,
    _require_matching_worker_runtime_v2,
    _worker_command,
    tokenizer_byte_round_trip_receipt_v2,
)
from scripts import run_weft1_gtok_v2 as tokenizer_cli


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _small_tokenizer() -> bytes:
    corpus = [
        "plain ASCII and 0123456789",
        "Greek λ Cyrillic Ж Arabic م CJK 漢",
        "emoji U0001f9f6‍U0001f52c and combining é",
        "line one\nline two\r\nline three",
    ] * 8
    return fit_a2_tokenizer(corpus, vocab_size=384, length=len(corpus))


def test_tokenizer_round_trip_receipt_reopens_exact_artifact() -> None:
    payload = _small_tokenizer()
    receipt = tokenizer_byte_round_trip_receipt_v2(payload)
    assert receipt["status"] == "EXACT_UTF8_BYTES_ROUND_TRIP_PASS"
    assert receipt["artifact_sha256"] == hashlib.sha256(payload).hexdigest()
    assert len(receipt["fixture_rows"]) == 8
    assert all(row["input_sha256"] == row["decoded_sha256"] for row in receipt["fixture_rows"])


def test_round_trip_detects_decoder_corruption() -> None:
    payload = _small_tokenizer().replace(b'"type":"ByteLevel"', b'"type":"BPEDecoder"', 1)
    with pytest.raises((Exception, GTokTokenizerV2Error)):
        tokenizer_byte_round_trip_receipt_v2(payload)


def test_worker_receipt_binds_process_runtime_and_corpus(tmp_path: Path) -> None:
    hashes = {name: _hash(name) for name in (
        "artifact", "merges", "inventory", "reserved", "regex", "fit", "full",
        "screen", "d6", "fit-input", "safety", "round-trip", "exe", "lock", "env",
        "runtime-attestation", "offline-network", "offline-policy",
    )}
    receipt = FitWorkerReceiptV2(
        process_id=17,
        output_root=str(tmp_path.resolve()),
        vocab_size=384,
        tokenizer_json_sha256=hashes["artifact"],
        merges_sha256=hashes["merges"],
        token_inventory_sha256=hashes["inventory"],
        reserved_inventory_sha256=hashes["reserved"],
        pretokenizer_regex_sha256=hashes["regex"],
        fit_stream_sha256=hashes["fit"],
        full_corpus_manifest_sha256=hashes["full"],
        screen_submanifest_sha256=hashes["screen"],
        physical_d6_evidence_sha256=hashes["d6"],
        tokenizer_fit_input_receipt_sha256=hashes["fit-input"],
        bpe_safety_receipt_sha256=hashes["safety"],
        byte_round_trip_receipt_sha256=hashes["round-trip"],
        executable_sha256=hashes["exe"],
        dependency_lock_sha256=hashes["lock"],
        environment_identity_sha256=hashes["env"],
        runtime_attestation_receipt_sha256=hashes["runtime-attestation"],
        offline_network_receipt_sha256=hashes["offline-network"],
        offline_network_policy_sha256=hashes["offline-policy"],
        tokenizers_version="0.22.2",
    )
    assert len(receipt.receipt_sha256) == 64
    _require_matching_worker_runtime_v2(receipt, replace(receipt, process_id=18))
    with pytest.raises(GTokTokenizerV2Error, match="environment_identity"):
        _require_matching_worker_runtime_v2(
            receipt,
            replace(receipt, process_id=18, environment_identity_sha256=_hash("other-env")),
        )
    with pytest.raises(ValueError, match="version drifted"):
        replace(receipt, tokenizers_version="0.22.3")


def test_fit_worker_command_is_isolated_from_hostile_ambient_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in {
        "PYTHONPATH": "/hostile/python",
        "PYTHONHOME": "/hostile/home",
        "LD_LIBRARY_PATH": "/hostile/lib",
        "TOKENIZERS_PARALLELISM": "true",
        "LANG": "hostile",
    }.items():
        monkeypatch.setenv(key, value)
    offline_sha256 = _hash("offline-network")
    environment = _isolated_worker_environment(
        offline_network_receipt_sha256=offline_sha256,
        offline_network_policy_sha256=_hash("offline-policy"),
    )
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "LD_LIBRARY_PATH" not in environment
    assert environment["LANG"] == environment["LC_ALL"] == "C.UTF-8"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert environment["TZ"] == "UTC"
    assert environment[OFFLINE_RECEIPT_ENV_V2] == offline_sha256
    assert environment["WEFT1_GTOK_OFFLINE_POLICY_SHA256"] == _hash(
        "offline-policy"
    )

    corpus = tmp_path / "corpus"
    repository = tmp_path / "repository"
    lock = tmp_path / "requirements.lock"
    offline = tmp_path / "offline.json"
    executable = Path(__import__("sys").executable)
    corpus.mkdir()
    repository.mkdir()
    lock.write_text("fixture")
    offline.write_text("fixture")
    command = _worker_command(
        corpus_root=corpus,
        output_root=tmp_path / "fresh-worker",
        vocab_size=16_384,
        dependency_lock_path=lock,
        worker_executable=executable,
        repository_root=repository,
        offline_network_receipt_path=offline,
        offline_network_receipt_sha256=offline_sha256,
        offline_network_policy_sha256=_hash("offline-policy"),
    )
    assert command[1:3] == ["-I", "-B"]
    assert "-m" not in command
    assert "/hostile/python" not in command
    assert str(offline.resolve()) in command
    assert command[-2:] == [offline_sha256, _hash("offline-policy")]


def _cli_arguments(tmp_path: Path) -> list[str]:
    return [
        "--corpus-root",
        str(tmp_path / "corpus"),
        "--freeze-receipt",
        str(tmp_path / "freeze.json"),
        "--gate-bundle",
        str(tmp_path / "gates.json"),
        "--c2-evidence",
        str(tmp_path / "c2.json"),
        "--decon-receipt",
        str(tmp_path / "decon.json"),
        "--dependency-lock",
        str(tmp_path / "requirements.lock"),
        "--worker-executable",
        str(Path(__import__("sys").executable)),
        "fit-arm",
        "--vocab-size",
        "16384",
        "--output-root",
        str(tmp_path / "output"),
        "--offline-network-receipt",
        str(tmp_path / "offline.json"),
    ]


def test_tokenizer_cli_rejects_unverified_parent_before_pb_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(_path: Path) -> str:
        raise GTokOfflineV2Error("not parent launched")

    monkeypatch.setattr(tokenizer_cli, "assert_offline_campaign_child_v2", reject)
    monkeypatch.setattr(
        tokenizer_cli,
        "load_frozen_screen_corpus_v2",
        lambda **_: pytest.fail("P-B must not be opened before the offline gate"),
    )
    with pytest.raises(GTokOfflineV2Error, match="not parent launched"):
        tokenizer_cli.main(_cli_arguments(tmp_path))


def test_tokenizer_cli_propagates_physical_offline_identity_to_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_sha256 = _hash("verified-offline-launch")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        tokenizer_cli,
        "assert_offline_campaign_child_v2",
        lambda _path: offline_sha256,
    )
    monkeypatch.setattr(
        tokenizer_cli,
        "load_offline_parent_receipt_v2",
        lambda _path: (
            type("Receipt", (), {"policy_sha256": _hash("offline-policy")})(),
            offline_sha256,
        ),
    )
    monkeypatch.setattr(
        tokenizer_cli,
        "load_frozen_screen_corpus_v2",
        lambda **_: object(),
    )

    def fit(arguments, vocab_size, output, frozen):
        seen["offline_sha256"] = arguments.offline_network_receipt_sha256
        seen["offline_path"] = arguments.offline_network_receipt
        seen["offline_policy_sha256"] = arguments.offline_network_policy_sha256
        seen["vocab_size"] = vocab_size
        return {}

    monkeypatch.setattr(tokenizer_cli, "_fit_one", fit)
    assert tokenizer_cli.main(_cli_arguments(tmp_path)) == 0
    assert seen == {
        "offline_sha256": offline_sha256,
        "offline_path": tmp_path / "offline.json",
        "offline_policy_sha256": _hash("offline-policy"),
        "vocab_size": 16_384,
    }
