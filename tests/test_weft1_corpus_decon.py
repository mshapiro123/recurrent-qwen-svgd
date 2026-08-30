from __future__ import annotations

from itertools import combinations
import hashlib
import io
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pytest

import training.weft1_corpus_decon as decon
import training.weft1_corpus_decon_contract as contract
import training.weft1_corpus_pb as pb
from training.paper2_phase3_p31 import (
    ALL_BATTERIES,
    build_split_ledger,
    eval_half,
    partition_rows,
)
from training.paper2_phase3_p31_completion import seal_confirm_membership
from training.weft1_gtok_contract import canonical_json_bytes

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
from test_weft1_corpus_pb import _build_scan_fixture


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _confirm_document_id(battery: str) -> str:
    for ordinal in range(10_000):
        value = f"synthetic-{battery}-{ordinal}"
        if eval_half(value) == "confirm":
            return value
    raise AssertionError("could not construct a synthetic CONFIRM identity")


def _write_confirm_inputs(
    root: Path, *, first_prompt: str = "sealed prompt absent from corpus"
) -> tuple[tuple[Path, ...], Path, Path, tuple[str, ...]]:
    prompts: list[str] = []
    source_rows: list[dict[str, object]] = []
    for ordinal, battery in enumerate(ALL_BATTERIES):
        prompt = first_prompt if ordinal == 0 else f"sealed prompt for {battery}"
        prompts.append(prompt)
        source_rows.append(
            {
                "answer": f"private answer {battery}",
                "battery": battery,
                "document_id": _confirm_document_id(battery),
                "item_id": f"item-{battery}",
                "native_split": "evaluation",
                "programmatic_verifier_available": False,
                "prompt": prompt,
                "reader": f"reader-{battery}-v1",
                "tests": None,
            }
        )
    partitioned = partition_rows(source_rows)
    assert all(row["partition"] == "confirm" for row in partitioned)
    private_path = root / "private" / "partitioned-rows.jsonl"
    private_path.parent.mkdir(parents=True)
    with private_path.open("xb") as handle:
        for row in partitioned:
            handle.write(canonical_json_bytes(row) + b"\n")
    private_sha = hashlib.sha256(private_path.read_bytes()).hexdigest()
    ledger = build_split_ledger(
        source_rows,
        dataset_revisions={battery: "synthetic-revision" for battery in ALL_BATTERIES},
        reader_versions={battery: f"reader-{battery}-v1" for battery in ALL_BATTERIES},
    )
    seal_root = root / "seals"
    seal_confirm_membership(
        ledger,
        output_dir=seal_root,
        source_rows_sha256=private_sha,
        source_manifest_sha256="a" * 64,
    )
    seal_paths = tuple(
        seal_root / f"confirm_{battery}.seal.json" for battery in ALL_BATTERIES
    )
    return (
        seal_paths,
        seal_root / "confirm_seal_ledger.json",
        private_path,
        tuple(prompts),
    )


def _eval_e_parameters(*, salt_hex: str) -> dict[str, object]:
    return {
        "character_shingle_size": contract.LEGACY_EVAL_E_CHARACTER_SHINGLE_SIZE,
        "estimated_jaccard_threshold": 0.8,
        "exact_hash": contract.LEGACY_EVAL_E_EXACT_HASH,
        "lsh_bands": contract.LEGACY_EVAL_E_LSH_BANDS,
        "lsh_rows_per_band": contract.LEGACY_EVAL_E_LSH_ROWS_PER_BAND,
        "minhash_components": contract.LEGACY_EVAL_E_MINHASH_COMPONENTS,
        "minhash_seed": contract.LEGACY_EVAL_E_MINHASH_SEED,
        "no_backfill": True,
        "normalization": contract.LEGACY_EVAL_E_NORMALIZATION,
        "partition_seed": 17,
        "prior_partition_seed": 11,
        "salt_hex": salt_hex,
    }


def _write_eval_e_inputs(
    root: Path, *, text: str = "anonymous sealed evaluation prompt"
) -> tuple[Path, Path, str]:
    salt_hex = "71" * 32
    salt = bytes.fromhex(salt_hex)
    parameters = _eval_e_parameters(salt_hex=salt_hex)
    lock_path = root / "eval-e" / "lock.json"
    _write_json(lock_path, {"panels": {decon.LEGACY_PANEL_NAME: parameters}})
    signature = decon.legacy_minhash_signature(
        text, seed=contract.LEGACY_EVAL_E_MINHASH_SEED
    )
    index = {
        "document_count": 1,
        "document_ids_persisted": False,
        "kind": decon.LEGACY_INDEX_KIND,
        "metadata_persisted": False,
        "minhash_signatures_uint64_decimal": [
            [str(value) for value in signature]
        ],
        "parameters": parameters,
        "plaintext_persisted": False,
        "salted_exact_hashes": [decon._legacy_salted_exact(text, salt)],
    }
    index_path = root / "eval-e" / "anonymous-index.json"
    _write_json(index_path, index)
    return index_path, lock_path, salt_hex


def _runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEFT1_NETWORK_DISABLED", "1")
    monkeypatch.setenv("WEFT1_NETWORK_GUARD_ACTIVE", "1")
    monkeypatch.setenv("WEFT1_NETWORK_GUARD_SHA256", "7" * 64)
    monkeypatch.setenv("WEFT1_DECON_UNSHARE_SHA256", "8" * 64)
    monkeypatch.setattr(
        decon, "_network_probe", lambda: decon.NETWORK_PROBE_RESULT
    )


def _bind_synthetic_eval_identities(
    monkeypatch: pytest.MonkeyPatch, *, index_path: Path, lock_path: Path
) -> None:
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        decon, "GOVERNED_EVAL_E_ANONYMOUS_INDEX_SHA256", index_sha
    )
    monkeypatch.setattr(decon, "GOVERNED_EVAL_E_LOCK_SHA256", lock_sha)
    for module, name, value in (
        (pb, "GOVERNED_EVAL_E_ANONYMOUS_INDEX_SHA256", index_sha),
        (pb, "GOVERNED_EVAL_E_LOCK_SHA256", lock_sha),
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)


def _bind_synthetic_confirm_identity(
    monkeypatch: pytest.MonkeyPatch, *, seal_paths: tuple[Path, ...]
) -> None:
    first = json.loads(seal_paths[0].read_text(encoding="utf-8"))
    ledger = json.loads(
        (seal_paths[0].parent / "confirm_seal_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    values = {
        "GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256": str(
            first["complete_ledger_sha256"]
        ),
        "GOVERNED_CONFIRM_SEAL_SET_SHA256": str(ledger["seal_set_sha256"]),
        "GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256": str(
            first["source_manifest_sha256"]
        ),
        "GOVERNED_CONFIRM_SOURCE_ROWS_SHA256": str(
            first["source_rows_sha256"]
        ),
    }
    for name, value in values.items():
        monkeypatch.setattr(decon, name, value)
        monkeypatch.setattr(contract, name, value)
        if hasattr(pb, name):
            monkeypatch.setattr(pb, name, value)


def _pa_fixture(root: Path) -> tuple[object, dict[str, int]]:
    root.mkdir()
    return _build_scan_fixture(root)


def _stub_runtime_attestation(
    monkeypatch: pytest.MonkeyPatch, pa: object
) -> None:
    provenance = pa.root / pb.replay_v3.GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3
    runtime = pa.root / pb.replay_v3.RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1
    monkeypatch.setattr(
        decon,
        "_attest_runtime_against_pa",
        lambda unused: (
            {
                "global_execution_provenance_sha256": hashlib.sha256(
                    provenance.read_bytes()
                ).hexdigest(),
                "python_executable_sha256": "9" * 64,
                "runtime_build_receipt_sha256": hashlib.sha256(
                    runtime.read_bytes()
                ).hexdigest(),
            },
            provenance,
            runtime,
        ),
    )


def _run_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    first_prompt: str = "sealed prompt absent from corpus",
) -> tuple[dict[str, object], str, object, tuple[str, ...]]:
    pa, _ = _pa_fixture(tmp_path / "pa-fixture")
    seal_paths, seal_ledger, private_rows, prompts = _write_confirm_inputs(
        tmp_path / "private-inputs", first_prompt=first_prompt
    )
    _bind_synthetic_confirm_identity(monkeypatch, seal_paths=seal_paths)
    eval_index, eval_lock, _salt = _write_eval_e_inputs(tmp_path / "private-inputs")
    _bind_synthetic_eval_identities(
        monkeypatch, index_path=eval_index, lock_path=eval_lock
    )
    monkeypatch.setattr(pb, "inspect_pa_v4", lambda root: pa)
    _runtime_environment(monkeypatch)
    _stub_runtime_attestation(monkeypatch, pa)
    receipt, physical = decon.run_hermetic_decon(
        materialization_root=pa.root,
        confirm_seal_paths=seal_paths,
        confirm_seal_ledger_path=seal_ledger,
        confirm_private_rows_path=private_rows,
        eval_e_index_path=eval_index,
        eval_e_lock_path=eval_lock,
        output_root=tmp_path / "decon-output",
    )
    return receipt, physical, pa, prompts


def _subsets(values: tuple[int, ...]) -> Iterable[frozenset[int]]:
    for size in range(len(values) + 1):
        for selected in combinations(values, size):
            yield frozenset(selected)


def test_safe_prefix_filter_has_no_false_negative_exhaustively() -> None:
    universe = tuple(range(6))
    salt = b"synthetic-proof-salt"
    for sealed in _subsets(universe):
        if not sealed:
            continue
        ordered = sorted(
            sealed,
            key=lambda value: hashlib.sha256(
                salt + int(value).to_bytes(8, "big")
            ).digest(),
        )
        prefix = frozenset(ordered[: contract.safe_prefix_size(len(sealed))])
        for query in _subsets(universe):
            if contract.jaccard_at_least_four_fifths(sealed, query):
                assert prefix & query


def test_confirm_safe_prefix_reaches_exact_jaccard_path() -> None:
    sealed = "abcdefghijklmnopqrstuvwxyz0123456789" * 4
    query = sealed[:-1] + "X"
    index = decon._ConfirmIndex(salt=b"confirm-salt")
    index.add(sealed)
    index.finalize()

    assert index.match(query) == (False, True)


def test_eval_e_full_signature_path_does_not_use_lsh_as_clean_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = tuple(range(128))
    query = list(sealed)
    for band in range(16):
        query[band * 8] += 10_000
    query_tuple = tuple(query)
    assert contract.signature_at_least_four_fifths(sealed, query_tuple)
    assert not contract.lsh_shares_band(sealed, query_tuple)
    monkeypatch.setattr(
        decon,
        "_legacy_hashed_shingle_values",
        lambda unused: np.asarray((1,), dtype=np.uint64),
    )
    monkeypatch.setattr(
        decon,
        "_legacy_signature_from_values",
        lambda unused, *, seed, components: query_tuple[:components],
    )
    index = decon._EvalEIndex(
        salt=b"eval-salt", exact_hashes=frozenset(), signatures=(sealed,)
    )

    assert index.match("opaque query") == (False, True)


def test_legacy_signature_reproduces_locked_reference() -> None:
    text = "Ａ mixed  Mixed\ntext with repeated repeated shingles"
    normalized = decon._legacy_normalize(text)
    shingles = {
        normalized[index : index + 13].encode("utf-8")
        for index in range(max(1, len(normalized) - 13 + 1))
    }
    values = np.fromiter(
        (
            int.from_bytes(hashlib.sha256(value).digest()[:8], "little")
            for value in shingles
        ),
        dtype=np.uint64,
    )
    rng = np.random.default_rng(contract.LEGACY_EVAL_E_MINHASH_SEED)
    multipliers = (
        rng.integers(0, np.iinfo(np.uint64).max, size=128, dtype=np.uint64)
        | np.uint64(1)
    )
    offsets = rng.integers(
        0, np.iinfo(np.uint64).max, size=128, dtype=np.uint64
    )
    reference = np.full(128, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, len(values), 4096):
        block = values[start : start + 4096]
        reference = np.minimum(
            reference,
            (multipliers[:, None] * block[None, :] + offsets[:, None]).min(
                axis=1
            ),
        )

    assert decon.legacy_minhash_signature(
        text, seed=contract.LEGACY_EVAL_E_MINHASH_SEED
    ) == tuple(int(value) for value in reference)


def test_hermetic_runner_emits_only_aggregate_clean_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, physical, pa, prompts = _run_fixture(tmp_path, monkeypatch)
    output = tmp_path / "decon-output"
    raw = (output / decon.DECON_RECEIPT_FILENAME).read_bytes()

    assert receipt["status"] == "CLEAN"
    assert receipt["screened_document_count"] == sum(
        int(row["record_count"]) for row in pa.full_shard_rows
    )
    assert receipt["screened_battery_count"] == 7
    assert receipt["salt_exported"] is False
    assert tuple(path.name for path in output.iterdir()) == (
        decon.DECON_RECEIPT_FILENAME,
    )
    assert hashlib.sha256(raw).hexdigest() == physical
    assert all(prompt.encode("utf-8") not in raw for prompt in prompts)
    assert b"private answer" not in raw
    assert b"salt_hex" not in raw
    assert b"item-" not in raw
    assert pb.load_hermetic_decon_receipt(
        output / decon.DECON_RECEIPT_FILENAME, pa=pa
    ) == (physical, receipt["receipt_sha256"], "CLEAN")


def test_hermetic_runner_reports_hit_and_never_claims_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _physical, _pa, _prompts = _run_fixture(
        tmp_path, monkeypatch, first_prompt="general-dolma"
    )

    assert receipt["status"] == "HIT"
    assert receipt["exact_match_count"] >= 1
    assert receipt["total_match_count"] >= 1
    assert receipt["hit_action"] == "HARD_STOP_NO_MINT"


def test_membership_mutation_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, _ = _pa_fixture(tmp_path / "pa-fixture")
    seal_paths, seal_ledger, private_rows, _ = _write_confirm_inputs(
        tmp_path / "private-inputs"
    )
    _bind_synthetic_confirm_identity(monkeypatch, seal_paths=seal_paths)
    eval_index, eval_lock, _ = _write_eval_e_inputs(tmp_path / "private-inputs")
    _bind_synthetic_eval_identities(
        monkeypatch, index_path=eval_index, lock_path=eval_lock
    )
    private_rows.write_bytes(private_rows.read_bytes() + b" \n")
    monkeypatch.setattr(pb, "inspect_pa_v4", lambda root: pa)
    _runtime_environment(monkeypatch)

    with pytest.raises(
        decon.DeconError, match="posture|source rows|source_rows|framing"
    ):
        decon.run_hermetic_decon(
            materialization_root=pa.root,
            confirm_seal_paths=seal_paths,
            confirm_seal_ledger_path=seal_ledger,
            confirm_private_rows_path=private_rows,
            eval_e_index_path=eval_index,
            eval_e_lock_path=eval_lock,
            output_root=tmp_path / "decon-output",
        )
    assert not (tmp_path / "decon-output").exists()


def test_self_consistent_confirm_substitution_rejected_by_governed_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seal_paths, seal_ledger, private_rows, _ = _write_confirm_inputs(
        tmp_path / "substituted-private-inputs"
    )
    _bind_synthetic_confirm_identity(monkeypatch, seal_paths=seal_paths)
    monkeypatch.setattr(
        decon, "GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256", "f" * 64
    )

    with pytest.raises(decon.DeconError, match="complete private ledger"):
        decon._validate_confirm_membership(
            seal_paths=seal_paths,
            seal_ledger_path=seal_ledger,
            private_rows_path=private_rows,
            salt=b"synthetic-private-salt",
        )


def test_calibration_and_runtime_ceilings_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, _ = _pa_fixture(tmp_path / "pa-fixture")

    class NoMatch:
        @staticmethod
        def match(unused: str) -> tuple[bool, bool]:
            return False, False

    ticks = iter(
        (
            0.0,
            1.0,
            2.0,
            float(decon.DECON_MAX_PROJECTED_SECONDS + 3),
        )
    )
    monkeypatch.setattr(decon.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(decon.DeconError, match="projection exceeds"):
        decon._calibrate_screening(
            pa=pa,
            confirm=NoMatch(),
            eval_e=NoMatch(),
            fixed_elapsed_seconds=0.0,
        )
    with pytest.raises(decon.DeconError, match="runtime ceiling"):
        decon._screen_full_shards(
            pa=pa,
            confirm=NoMatch(),
            eval_e=NoMatch(),
            deadline_monotonic=0.0,
        )


def test_calibration_samples_bounded_bytes_and_prices_fixed_hash_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, _ = _pa_fixture(tmp_path / "pa-fixture")

    class NoMatch:
        @staticmethod
        def match(unused: str) -> tuple[bool, bool]:
            return False, False

    def forbidden_full_hash(unused: Path) -> str:
        raise AssertionError("calibration must not hash an entire shard")

    monkeypatch.setattr(decon, "_sha256_file", forbidden_full_hash)
    ticks = iter((0.0, 2.0, 10.0, 13.0))
    monkeypatch.setattr(decon.time, "perf_counter", lambda: next(ticks))
    result = decon._calibrate_screening(
        pa=pa,
        confirm=NoMatch(),
        eval_e=NoMatch(),
        fixed_elapsed_seconds=5.0,
    )
    total_compressed = sum(int(row["zstd_bytes"]) for row in pa.full_shard_rows)
    total_logical = sum(
        int(row["retained_text_bytes"]) for row in pa.full_shard_rows
    )
    expected = (
        5.0
        + 2.0
        + 3.0
        + 2.0 * total_compressed / result.compressed_sample_bytes
        + 3.0 * total_logical / result.logical_sample_bytes
    )
    assert result.projected_seconds == pytest.approx(expected)
    assert result.shard_count == len(pa.full_shard_rows)
    assert result.compressed_sample_bytes <= min(
        total_compressed, decon.DECON_CALIBRATION_LOGICAL_BYTES
    )
    assert result.logical_sample_bytes <= min(
        total_logical, decon.DECON_CALIBRATION_LOGICAL_BYTES
    )


def test_jsonl_record_reader_enforces_bound_before_unbounded_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(decon, "DECON_MAX_RECORD_JSONL_BYTES", 8)
    with io.BufferedReader(io.BytesIO(b"1234567\n")) as bounded:
        assert list(decon._iter_bounded_jsonl(bounded)) == [b"1234567\n"]
    with io.BufferedReader(io.BytesIO(b"12345678\n")) as oversized:
        with pytest.raises(decon.DeconError, match="byte ceiling"):
            list(decon._iter_bounded_jsonl(oversized))


def test_calibration_never_overshoots_exact_global_byte_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, _ = _pa_fixture(tmp_path / "pa-fixture")

    class NoMatch:
        @staticmethod
        def match(unused: str) -> tuple[bool, bool]:
            return False, False

    monkeypatch.setattr(decon, "DECON_CALIBRATION_LOGICAL_BYTES", 32)
    ticks = iter((0.0, 1e-9, 2e-9, 3e-9))
    monkeypatch.setattr(decon.time, "perf_counter", lambda: next(ticks))
    result = decon._calibrate_screening(
        pa=pa,
        confirm=NoMatch(),
        eval_e=NoMatch(),
        fixed_elapsed_seconds=0.0,
    )

    assert result.compressed_sample_bytes <= 32
    assert result.logical_sample_bytes <= 32


def test_runtime_attestation_matches_exact_pa_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, _ = _pa_fixture(tmp_path / "pa-fixture")
    provenance_path = (
        pa.root / pb.replay_v3.GLOBAL_EXECUTION_PROVENANCE_RELATIVE_PATH_V3
    )
    runtime_path = pa.root / pb.replay_v3.RUNTIME_BUILD_RECEIPT_RELATIVE_PATH_V1
    _write_json(provenance_path, {"synthetic": "validated by test double"})
    expected = {
        "dependency_lock_sha256": "1" * 64,
        "environment_identity_sha256": "2" * 64,
        "python_executable_sha256": "3" * 64,
        "runtime_build_receipt_sha256": hashlib.sha256(
            runtime_path.read_bytes()
        ).hexdigest(),
    }
    attested = dict(expected)
    observed_call: dict[str, object] = {}

    def fake_validate(unused: object) -> dict[str, str]:
        return dict(expected)

    def fake_attest(**kwargs: object) -> SimpleNamespace:
        observed_call.update(kwargs)
        return SimpleNamespace(
            dependency_lock_sha256=attested["dependency_lock_sha256"],
            environment_identity_sha256=attested["environment_identity_sha256"],
            executable_sha256=attested["python_executable_sha256"],
        )

    monkeypatch.setattr(
        pb.replay_v3, "validate_global_execution_provenance_v3", fake_validate
    )
    import training.weft1_corpus_pa as corpus_pa

    monkeypatch.setattr(corpus_pa, "attest_runtime_v3", fake_attest)
    commitments, observed_provenance, observed_runtime = (
        decon._attest_runtime_against_pa(pa)
    )

    assert observed_call["executable"] == Path(sys.executable)
    assert observed_call["requirements_lock"].name == (
        "weft1_corpus_gtok_a2_requirements.lock"
    )
    assert commitments == {
        "global_execution_provenance_sha256": hashlib.sha256(
            provenance_path.read_bytes()
        ).hexdigest(),
        "python_executable_sha256": expected["python_executable_sha256"],
        "runtime_build_receipt_sha256": expected[
            "runtime_build_receipt_sha256"
        ],
    }
    assert (observed_provenance, observed_runtime) == (
        provenance_path,
        runtime_path,
    )

    expected["environment_identity_sha256"] = "4" * 64
    with pytest.raises(decon.DeconError, match="differs from P-A"):
        decon._attest_runtime_against_pa(pa)


def test_parent_snapshot_does_not_rehash_full_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, _ = _pa_fixture(tmp_path / "pa-fixture")
    governed = tmp_path / "governed"
    governed.mkdir()
    private_rows = governed / "private.jsonl"
    seal_ledger = governed / "ledger.json"
    eval_index = governed / "eval-index.json"
    eval_lock = governed / "eval-lock.json"
    seal = governed / "seal.json"
    for ordinal, path in enumerate(
        (private_rows, seal_ledger, eval_index, eval_lock, seal), start=1
    ):
        path.write_bytes(bytes((ordinal,)))
    shard_paths = {
        (pa.root / str(row["relative_path"])).resolve()
        for row in pa.full_shard_rows
    }
    original_hash = decon._sha256_file
    hashed: list[Path] = []

    def guarded_hash(path: Path) -> str:
        resolved = path.resolve()
        if resolved in shard_paths:
            raise AssertionError("parent snapshot must not rehash full shards")
        hashed.append(resolved)
        return original_hash(path)

    monkeypatch.setattr(decon, "_sha256_file", guarded_hash)
    snapshot = decon._parent_snapshot(
        pa=pa,
        seal_paths=(seal,),
        seal_ledger_path=seal_ledger,
        private_rows_path=private_rows,
        eval_e_index_path=eval_index,
        eval_e_lock_path=eval_lock,
    )

    assert len(snapshot["shards"]) == len(pa.full_shard_rows)
    assert shard_paths.isdisjoint(hashed)
    assert all(
        row["zstd_sha256"] == source["zstd_sha256"]
        for row, source in zip(snapshot["shards"], pa.full_shard_rows, strict=True)
    )


def test_parent_launcher_binds_tz_and_total_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa, _ = _pa_fixture(tmp_path / "pa-fixture")
    private_root = tmp_path / "private"
    private_root.mkdir()
    paths = [private_root / f"input-{ordinal}.json" for ordinal in range(5)]
    for ordinal, path in enumerate(paths, start=1):
        path.write_bytes(bytes((ordinal,)))
    unshare = tmp_path / "unshare"
    unshare.write_bytes(b"verified unshare fixture")
    local_work = tmp_path / "local-work"
    local_work.mkdir()
    output = tmp_path / "decon-output"
    observed_run: dict[str, object] = {}

    import training.weft1_corpus_replay_a2 as replay_a2

    monkeypatch.setattr(pb, "inspect_pa_v4", lambda unused: pa)
    monkeypatch.setattr(decon, "_parent_snapshot", lambda **unused: {"same": True})
    monkeypatch.setattr(
        replay_a2, "_resolve_python_executable", lambda unused: Path(sys.executable)
    )
    monkeypatch.setattr(
        replay_a2, "_resolve_unshare_executable", lambda unused: unshare
    )
    monkeypatch.setattr(
        replay_a2, "_verify_unshare_network_isolation", lambda **unused: None
    )
    monkeypatch.setattr(
        pb,
        "load_hermetic_decon_receipt",
        lambda unused, *, pa: ("a" * 64, "b" * 64, "CLEAN"),
    )

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        observed_run.update(kwargs)
        output.mkdir()
        (output / decon.DECON_RECEIPT_FILENAME).write_bytes(b"{}\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(decon.subprocess, "run", fake_run)
    ticks = iter((100.0, 160.0, 170.0))
    monkeypatch.setattr(decon.time, "monotonic", lambda: next(ticks))
    result = decon.launch_hermetic_decon(
        materialization_root=pa.root,
        confirm_seal_paths=(paths[0],),
        confirm_seal_ledger_path=paths[1],
        confirm_private_rows_path=paths[2],
        eval_e_index_path=paths[3],
        eval_e_lock_path=paths[4],
        output_root=output,
        local_work_parent=local_work,
    )

    assert result == ("a" * 64, "b" * 64, "CLEAN")
    assert observed_run["env"]["TZ"] == "UTC"
    assert observed_run["env"]["LANG"] == "C.UTF-8"
    assert observed_run["timeout"] == pytest.approx(
        decon.DECON_MAX_RUNTIME_SECONDS - 60.0
    )
    assert observed_run["shell"] is False


def test_full_scan_rejects_same_size_shard_mutation(
    tmp_path: Path,
) -> None:
    pa, _ = _pa_fixture(tmp_path / "pa-fixture")

    class NoMatch:
        @staticmethod
        def match(unused: str) -> tuple[bool, bool]:
            return False, False

    row = pa.full_shard_rows[0]
    shard = pa.root / str(row["relative_path"])
    changed = bytearray(shard.read_bytes())
    changed[-1] ^= 1
    shard.write_bytes(changed)
    assert shard.stat().st_size == row["zstd_bytes"]
    with pytest.raises(decon.DeconError, match="physical identity"):
        decon._screen_full_shards(
            pa=pa,
            confirm=NoMatch(),
            eval_e=NoMatch(),
        )


def test_code_inventory_and_isolated_launcher_are_bound() -> None:
    assert decon.DECON_CODE_RELATIVE_PATHS == tuple(
        sorted(decon.DECON_CODE_RELATIVE_PATHS)
    )
    assert decon.DECON_CODE_RELATIVE_PATHS == pb._DECON_CODE_RELATIVE_PATHS
    assert decon.DECON_CALIBRATION_LOGICAL_BYTES == 64 * 1024 * 1024
    assert decon.DECON_MAX_PROJECTED_SECONDS == 12 * 60 * 60
    assert decon.DECON_MAX_RECORD_JSONL_BYTES == 64 * 1024 * 1024
    assert {
        "training/paper2_phase3_p31.py",
        "training/weft1_corpus_a2.py",
        "training/weft1_corpus_materialize_a3.py",
        "training/weft1_corpus_pa.py",
        "training/weft1_corpus_pb.py",
        "training/weft1_corpus_replay_a2.py",
        "training/weft1_corpus_replay_a3.py",
        "training/weft1_gtok_contract.py",
        "training/weft1_release.py",
        "training/weft1_release_bindings_20260830.json",
        "training/weft1_release_card_evidence_20260830.json",
        "training/weft1_strict_io.py",
    } <= set(decon.DECON_CODE_RELATIVE_PATHS)
    assert "require_network_isolation" not in inspect.signature(
        decon.run_hermetic_decon
    ).parameters
    source = inspect.getsource(decon.launch_hermetic_decon)
    assert '"--net"' in source
    assert '"-I"' in source
    assert "_verify_unshare_network_isolation" in source
    assert "stdout=subprocess.PIPE" in source
    assert "stderr=subprocess.PIPE" in source
    assert "shell=False" in source


def test_every_physical_code_commitment_has_an_explicit_lf_checkout_rule() -> None:
    rules = frozenset(
        (decon.REPOSITORY_ROOT / ".gitattributes")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    missing = tuple(
        relative
        for relative in decon.DECON_CODE_RELATIVE_PATHS
        if f"{relative} text eol=lf" not in rules
    )
    assert missing == ()


@pytest.mark.parametrize(
    "substituted_name",
    ("weft1_corpus_pb.py", "weft1_release_card_evidence_20260830.json"),
)
def test_code_inventory_detects_behavior_dependency_substitution(
    monkeypatch: pytest.MonkeyPatch,
    substituted_name: str,
) -> None:
    before = decon._screen_code_commitments()
    original_hash = decon._sha256_file

    def substituted(path: Path) -> str:
        if path.name == substituted_name:
            return "f" * 64
        return original_hash(path)

    monkeypatch.setattr(decon, "_sha256_file", substituted)
    after = decon._screen_code_commitments()
    assert after != before


def test_governed_eval_e_artifact_identities_are_exact() -> None:
    index_path = (
        decon.REPOSITORY_ROOT
        / "artifacts"
        / "tm0_20260825"
        / "sealed"
        / "tm0_eval_e_anonymous_index.json"
    )
    lock_path = decon.REPOSITORY_ROOT / "training" / "paper2_tm0_lock.json"
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == (
        contract.GOVERNED_EVAL_E_ANONYMOUS_INDEX_SHA256
    )
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == (
        contract.GOVERNED_EVAL_E_LOCK_SHA256
    )
    assert contract.GOVERNED_CONFIRM_COMPLETION_RECEIPT_SHA256 == (
        "1b6e40149034047a35cd669a6f8fd045c26330ac9b67075e0cb39b0271d1802b"
    )
    assert contract.GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256 == (
        "503d6a5551f187cb96d80de45ee0d1deff9d3202186aa4995cd80b0bfba7653f"
    )
    assert contract.GOVERNED_CONFIRM_SEAL_SET_SHA256 == (
        "6edd229f934477ae978bb70193df90a6b90830408e7cbc1286c5dea32259377b"
    )
    assert contract.GOVERNED_CONFIRM_SOURCE_ROWS_SHA256 == (
        "5e32eb1905b05076a59b2c5b315ccf9319c04eda18af450565128fd34c18ffa5"
    )
    assert contract.GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256 == (
        "1bcb847e02652881b0161718f73f13faeb30ede93f95f6a50152af900cdedef7"
    )
