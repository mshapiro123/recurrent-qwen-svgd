from __future__ import annotations

from fractions import Fraction
import base64
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
from types import SimpleNamespace
import unicodedata

import pytest
import zstandard

import training.weft1_corpus_pa as production
import training.weft1_strict_io as strict_io
from training.weft1_corpus_a2 import (
    A2_DEDUP_SEED,
    MINHASH_RECALL_JACCARD_LEVELS,
    MinHashRecallAuditV3,
    MinHashSyntheticRecallCellV3,
    StableDocumentV3,
    canonical_jsonl_record_bytes_v3,
)
from training.weft1_corpus_pa import (
    CorpusProductionError,
    FastTextLanguageIdAdapterV3,
    RawDocumentV3,
    RuntimeAttestationV3,
    RuntimeExpectationV3,
    SourceAssetExpectationV3,
    attest_runtime_v3,
    installed_distribution_inventory_v3,
    parse_hash_locked_requirements_v3,
    run_fixture_replay,
    typed_replay_receipt_from_mapping,
    verify_source_cache_assets,
    write_jsonl_zstd_shards_v3,
)


def _hash(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _record_hash(value: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode(
        "ascii"
    )


def _write_installed_distribution(
    site_root: Path, name: str, version: str, payload: bytes
) -> Path:
    import_name = name.replace("-", "_")
    module_path = site_root / f"{import_name}.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_bytes(payload)
    metadata_path = site_root / f"{import_name}-{version}.dist-info"
    metadata_path.mkdir()
    metadata_bytes = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(
        "utf-8"
    )
    (metadata_path / "METADATA").write_bytes(metadata_bytes)
    record_path = metadata_path / "RECORD"
    record_path.write_text(
        "\n".join(
            (
                f"{module_path.relative_to(site_root).as_posix()},sha256={_record_hash(payload)},{len(payload)}",
                f"{(metadata_path / 'METADATA').relative_to(site_root).as_posix()},sha256={_record_hash(metadata_bytes)},{len(metadata_bytes)}",
                f"{record_path.relative_to(site_root).as_posix()},,",
            )
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )
    return module_path


def _document(record_id: str, text: str) -> StableDocumentV3:
    return StableDocumentV3(
        source="dolma_web",
        stratum="general",
        stable_source_record_id=_hash(record_id),
        text=text,
    )


def _read_zstd(path: Path) -> bytes:
    with path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            return reader.read()


def _recall_audit() -> MinHashRecallAuditV3:
    return MinHashRecallAuditV3(
        seed=A2_DEDUP_SEED,
        synthetic_cells=tuple(
            MinHashSyntheticRecallCellV3(
                exact_jaccard=level,
                pair_count=100,
                candidate_count=95,
            )
            for level in MINHASH_RECALL_JACCARD_LEVELS
        ),
        real_sample_identity_sha256=_hash("fixture-recall-sample"),
        real_dolma_document_count=10,
        real_fineweb_document_count=10,
        real_exact_pairs_at_or_above_threshold=7,
        real_candidate_pairs_at_or_above_threshold=6,
    )


def test_runtime_attestation_hashes_exact_lock_executable_and_environment(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "requirements.lock"
    alpha_hash = "1" * 64
    bravo_hash = "2" * 64
    lock_bytes = (
        f"alpha==1.2.3 \\\n    --hash=sha256:{alpha_hash}\n"
        f"Bravo_Package==9.8.7 \\\n    --hash=sha256:{bravo_hash}\n"
    ).encode("ascii")
    lock.write_bytes(lock_bytes)
    prefix = tmp_path / "runtime"
    site_root = prefix / "lib" / "python3.11" / "site-packages"
    _write_installed_distribution(site_root, "alpha", "1.2.3", b"alpha = 1\n")
    _write_installed_distribution(
        site_root, "bravo-package", "9.8.7", b"bravo = 1\n"
    )
    _write_installed_distribution(site_root, "pip", "24.0", b"pip = 1\n")
    inventory = installed_distribution_inventory_v3(
        lock_bytes,
        installation_prefix=prefix,
        distributions=metadata.distributions(path=[str(site_root)]),
    )
    executable = tmp_path / "python-copy"
    executable.write_bytes(b"exact executable bytes")
    linkage_core = {
        "executable": {
            "bytes": len(executable.read_bytes()),
            "path": str(executable.resolve()),
            "sha256": _hash(executable.read_bytes()),
        },
        "libpython_library": {
            "bytes": 1,
            "path": str((tmp_path / "libpython.so").resolve()),
            "sha256": "3" * 64,
        },
        "schema": production.RUNTIME_LINKAGE_SCHEMA_V3,
        "sqlite_extension": {
            "bytes": 1,
            "path": str((tmp_path / "_sqlite3.so").resolve()),
            "sha256": "4" * 64,
        },
        "sqlite_library": {
            "bytes": 1,
            "path": str((tmp_path / "libsqlite3.so").resolve()),
            "sha256": "5" * 64,
        },
    }
    linkage = {
        **linkage_core,
        "linkage_identity_sha256": production.execution_authority_v3_bound_sha256(
            production.RUNTIME_LINKAGE_SCHEMA_V3, linkage_core
        ),
    }
    versions = {"alpha": "1.2.3", "bravo-package": "9.8.7"}
    expectation = RuntimeExpectationV3(
        python_version=platform.python_version(),
        unicode_data_version=unicodedata.unidata_version,
        sqlite_version=sqlite3.sqlite_version,
        zstandard_package_version=zstandard.__version__,
        libzstd_version=".".join(str(item) for item in zstandard.ZSTD_VERSION),
        required_environment=(),
    )

    attestation = attest_runtime_v3(
        requirements_lock=lock,
        expected_lock_sha256=_hash(lock_bytes),
        expectation=expectation,
        executable=executable,
        version_lookup=versions.__getitem__,
        inventory_builder=lambda _lock_bytes: inventory,
        linkage_builder=lambda _executable: linkage,
    )
    first = attestation.process_attestation(tmp_path / "first")
    second = attestation.process_attestation(tmp_path / "second")
    assert attestation.executable_sha256 == _hash(b"exact executable bytes")
    assert attestation.dependency_lock_sha256 == _hash(lock_bytes)
    assert attestation.environment_payload["distributions"] == (
        {
            "artifact_sha256s": (alpha_hash,),
            "distribution": "alpha",
            "version": "1.2.3",
        },
        {
            "artifact_sha256s": (bravo_hash,),
            "distribution": "bravo-package",
            "version": "9.8.7",
        },
    )
    assert attestation.environment_payload["installed_distribution_inventory"] == (
        inventory
    )
    assert first.compatibility_identity_sha256 == second.compatibility_identity_sha256
    assert first.output_root != second.output_root

    tampered_linkage = json.loads(json.dumps(linkage))
    tampered_linkage["sqlite_library"]["sha256"] = "9" * 64
    with pytest.raises(CorpusProductionError, match="identity drifted"):
        production.validate_runtime_linkage_inventory_v3(tampered_linkage)

    changed_core = {
        **linkage_core,
        "sqlite_library": {
            **linkage_core["sqlite_library"],
            "sha256": "9" * 64,
        },
    }
    changed_linkage = {
        **changed_core,
        "linkage_identity_sha256": production.execution_authority_v3_bound_sha256(
            production.RUNTIME_LINKAGE_SCHEMA_V3, changed_core
        ),
    }
    changed_attestation = attest_runtime_v3(
        requirements_lock=lock,
        expected_lock_sha256=_hash(lock_bytes),
        expectation=expectation,
        executable=executable,
        version_lookup=versions.__getitem__,
        inventory_builder=lambda _lock_bytes: inventory,
        linkage_builder=lambda _executable: changed_linkage,
    )
    assert (
        changed_attestation.environment_identity_sha256
        != attestation.environment_identity_sha256
    )

    with pytest.raises(CorpusProductionError, match="version mismatch"):
        attest_runtime_v3(
            requirements_lock=lock,
            expected_lock_sha256=_hash(lock_bytes),
            expectation=expectation,
            executable=executable,
            version_lookup=lambda name: "wrong" if name == "alpha" else "9.8.7",
            inventory_builder=lambda _lock_bytes: inventory,
            linkage_builder=lambda _executable: linkage,
        )
    with pytest.raises(CorpusProductionError, match="lock SHA-256"):
        attest_runtime_v3(
            requirements_lock=lock,
            expected_lock_sha256=_hash(b"different"),
            expectation=expectation,
            executable=executable,
            version_lookup=versions.__getitem__,
            inventory_builder=lambda _lock_bytes: inventory,
            linkage_builder=lambda _executable: linkage,
        )


def test_hash_locked_requirement_parser_rejects_unhashed_or_reordered_closure() -> None:
    with pytest.raises(CorpusProductionError, match="lacks hashes"):
        parse_hash_locked_requirements_v3(b"alpha==1.0\n")
    with pytest.raises(CorpusProductionError, match="repeats a distribution"):
        parse_hash_locked_requirements_v3(
            (
                "alpha==1.0 \\\n    --hash=sha256:" + "1" * 64 + "\n"
                "alpha==1.0 \\\n    --hash=sha256:" + "2" * 64 + "\n"
            ).encode("ascii")
        )


def test_installed_distribution_inventory_rejects_mutation_and_extra_distribution(
    tmp_path: Path,
) -> None:
    alpha_hash = "1" * 64
    lock_bytes = (
        f"alpha==1.0 \\\n    --hash=sha256:{alpha_hash}\n"
    ).encode("ascii")
    prefix = tmp_path / "runtime"
    site_root = prefix / "lib" / "python3.11" / "site-packages"
    alpha_path = _write_installed_distribution(
        site_root, "alpha", "1.0", b"alpha = 1\n"
    )
    _write_installed_distribution(site_root, "pip", "24.0", b"pip = 1\n")

    first = installed_distribution_inventory_v3(
        lock_bytes,
        installation_prefix=prefix,
        distributions=metadata.distributions(path=[str(site_root)]),
    )
    assert [row["distribution"] for row in first["distributions"]] == [
        "alpha",
        "pip",
    ]

    alpha_path.write_bytes(b"alpha = 2\n")
    with pytest.raises(CorpusProductionError, match="RECORD hash mismatch"):
        installed_distribution_inventory_v3(
            lock_bytes,
            installation_prefix=prefix,
            distributions=metadata.distributions(path=[str(site_root)]),
        )

    alpha_path.write_bytes(b"alpha = 1\n")
    _write_installed_distribution(site_root, "echo", "1.0", b"echo = 1\n")
    with pytest.raises(CorpusProductionError, match="unexpected installed distributions"):
        installed_distribution_inventory_v3(
            lock_bytes,
            installation_prefix=prefix,
            distributions=metadata.distributions(path=[str(site_root)]),
        )


def test_installed_distribution_inventory_accepts_owned_vendored_record(
    tmp_path: Path,
) -> None:
    lock_bytes = (
        f"setuptools==84.0.0 \\\n    --hash=sha256:{'1' * 64}\n"
    ).encode("ascii")
    prefix = tmp_path / "runtime"
    site_root = prefix / "lib" / "python3.11" / "site-packages"
    _write_installed_distribution(
        site_root, "setuptools", "84.0.0", b"setuptools = 1\n"
    )
    _write_installed_distribution(site_root, "pip", "24.0", b"pip = 1\n")
    vendored_record = (
        site_root
        / "setuptools"
        / "_vendor"
        / "example-1.0.dist-info"
        / "RECORD"
    )
    vendored_record.parent.mkdir(parents=True)
    vendored_bytes = b"vendored.py,,\n"
    vendored_record.write_bytes(vendored_bytes)
    top_record = site_root / "setuptools-84.0.0.dist-info" / "RECORD"
    with top_record.open("a", encoding="utf-8", newline="") as handle:
        handle.write(
            f"{vendored_record.relative_to(site_root).as_posix()},"
            f"sha256={_record_hash(vendored_bytes)},{len(vendored_bytes)}\n"
        )

    inventory = installed_distribution_inventory_v3(
        lock_bytes,
        installation_prefix=prefix,
        distributions=metadata.distributions(path=[str(site_root)]),
    )
    setuptools_row = next(
        row
        for row in inventory["distributions"]
        if row["distribution"] == "setuptools"
    )
    assert setuptools_row["record_path"].endswith(
        "setuptools-84.0.0.dist-info/RECORD"
    )
    vendored_row = next(
        row
        for row in inventory["files"]
        if row["relative_path"].endswith("example-1.0.dist-info/RECORD")
    )
    assert vendored_row["owners"] == ["setuptools"]


def test_source_cache_verification_is_root_independent_and_fail_closed(
    tmp_path: Path,
) -> None:
    payloads = {
        "nested/a.jsonl.zst": b"asset-a",
        "nested/b.parquet": b"asset-b",
    }
    expectations = tuple(
        SourceAssetExpectationV3(
            source="dolma_web",
            locator=f"hf://pinned/{relative}",
            cache_relative_path=relative,
            byte_count=len(data),
            sha256=_hash(data),
        )
        for relative, data in reversed(tuple(payloads.items()))
    )
    roots = (tmp_path / "cache-a", tmp_path / "cache-b")
    for root in roots:
        for relative, data in payloads.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    first = verify_source_cache_assets(roots[0], expectations)
    second = verify_source_cache_assets(roots[1], expectations)
    assert first == second
    assert first.manifest_sha256 == second.manifest_sha256
    assert tuple(item.cache_relative_path for item in first.assets) == tuple(
        sorted(payloads)
    )

    (roots[1] / "nested/a.jsonl.zst").write_bytes(b"tampered")
    with pytest.raises(CorpusProductionError, match="byte count mismatch"):
        verify_source_cache_assets(roots[1], expectations)
    with pytest.raises(ValueError, match="relative POSIX"):
        SourceAssetExpectationV3(
            source="dolma_web",
            locator="hf://pinned/escape",
            cache_relative_path="../escape",
            byte_count=1,
            sha256=_hash(b"x"),
        )


def test_source_cache_verification_rejects_symlinked_asset_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"asset"
    root = tmp_path / "cache"
    real = root / "real"
    real.mkdir(parents=True)
    (real / "asset.bin").write_bytes(payload)
    alias = root / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except (NotImplementedError, OSError):
        # Windows may deny symlink creation without Developer Mode.  Preserve
        # the source-API adversarial test by simulating the exact lstat result
        # on a real directory containing the otherwise-readable asset.
        alias.mkdir()
        (alias / "asset.bin").write_bytes(payload)
        original = strict_io._is_link_or_reparse
        monkeypatch.setattr(
            strict_io,
            "_is_link_or_reparse",
            lambda path: path == alias or original(path),
        )

    expectation = SourceAssetExpectationV3(
        source="dolma_web",
        locator="hf://pinned/asset.bin",
        cache_relative_path="alias/asset.bin",
        byte_count=len(payload),
        sha256=_hash(payload),
    )
    with pytest.raises(ValueError, match="symlink/reparse"):
        verify_source_cache_assets(root, (expectation,))


def test_fasttext_adapter_uses_only_raw_backend_and_breaks_ties_lexically() -> None:
    calls: list[tuple[str, int, float, str]] = []

    class RawBackend:
        def predict(
            self,
            text: str,
            k: int,
            threshold: float,
            on_unicode_error: str,
        ) -> list[tuple[float, str]]:
            calls.append((text, k, threshold, on_unicode_error))
            return [(0.9, "__label__fr"), (0.9, "__label__en")]

    class FakeModel:
        def __init__(self) -> None:
            self.f = RawBackend()

        def predict(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("the NumPy-incompatible public wrapper was called")

    adapter = object.__new__(FastTextLanguageIdAdapterV3)
    adapter._model = FakeModel()  # type: ignore[attr-defined]
    adapter._label_count = 2  # type: ignore[attr-defined]
    decision = adapter.classify(_document("lang", "Hello\r\nworld"))
    assert decision.label == "__label__en"
    assert decision.probability == 0.9
    assert decision.keep is True
    assert calls == [("Hello  world\n", 2, 0.0, "strict")]
    assert decision.scoring_input_sha256 == hashlib.sha256(
        b"Hello  world\n"
    ).hexdigest()

    with pytest.raises(ValueError, match="general"):
        adapter.classify(
            StableDocumentV3(
                source="stackedu",
                stratum="code",
                stable_source_record_id=_hash("code-row"),
                text="print('hello')",
            )
        )
    assert len(calls) == 1

    adapter._label_count = 3  # type: ignore[attr-defined]
    with pytest.raises(CorpusProductionError, match="complete label inventory"):
        adapter.classify(_document("incomplete", "Hello"))


def test_fasttext_adapter_loads_only_the_verified_private_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b"tiny pinned fasttext model"
    source = tmp_path / "lid.176.bin"
    source.write_bytes(original)
    binding = SimpleNamespace(
        package="fasttext-wheel",
        package_version="0.9.2",
        model_bytes=len(original),
        model_sha256=hashlib.sha256(original).hexdigest(),
        keep_label="__label__en",
    )
    monkeypatch.setattr(production, "A2_LANGUAGE_ID_BINDING", binding)
    monkeypatch.setattr(production.metadata, "version", lambda _package: "0.9.2")
    observed: dict[str, object] = {}

    class FakeBackend:
        def predict(self, *_args: object) -> list[tuple[float, str]]:
            return [(1.0, "__label__en")]

    class FakeModel:
        f = FakeBackend()

        @staticmethod
        def get_labels() -> list[str]:
            return ["__label__en"]

    def load_model(snapshot_name: str) -> FakeModel:
        snapshot = Path(snapshot_name)
        observed["snapshot"] = snapshot
        observed["bytes"] = snapshot.read_bytes()
        # A replacement after verification must not alter the bytes consumed
        # by FastText: load_model receives the private snapshot, not source.
        source.write_bytes(b"replaced after verification")
        observed["bytes_after_source_replace"] = snapshot.read_bytes()
        return FakeModel()

    monkeypatch.setitem(sys.modules, "fasttext", SimpleNamespace(load_model=load_model))
    adapter = FastTextLanguageIdAdapterV3(source)
    snapshot = observed["snapshot"]
    assert isinstance(snapshot, Path)
    assert snapshot != source
    assert snapshot.is_file()
    assert observed["bytes"] == observed["bytes_after_source_replace"] == original
    assert adapter._model_snapshot_root.name == str(snapshot.parent)  # type: ignore[attr-defined]


def test_fasttext_adapter_rejects_lexical_link_boundary_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "lid.176.bin"
    source.write_bytes(b"model")
    monkeypatch.setattr(
        production,
        "assert_no_symlink_ancestors",
        lambda _path: (_ for _ in ()).throw(ValueError("symlink/reparse fixture")),
    )
    with pytest.raises(ValueError, match="symlink/reparse"):
        FastTextLanguageIdAdapterV3(source)


def test_shards_cut_before_next_emit_oversized_singletons_and_replay_bytes(
    tmp_path: Path,
) -> None:
    first = _document("first", "alpha")
    second = _document("second", "beta")
    target = len(canonical_jsonl_record_bytes_v3(first)) + len(
        canonical_jsonl_record_bytes_v3(second)
    ) - 1
    oversized_text = "x" * (target * 2)
    documents = (
        RawDocumentV3("dolma_web", "general", _hash("first"), "alpha"),
        RawDocumentV3(
            "dolma_web", "general", _hash("invalid"), b"bad\xffutf8"
        ),
        RawDocumentV3("dolma_web", "general", _hash("second"), "beta"),
        RawDocumentV3(
            "fineweb_edu", "general", _hash("oversized"), oversized_text
        ),
        RawDocumentV3("dolma_web", "general", _hash("last"), "omega"),
    )
    roots = (tmp_path / "replay-a", tmp_path / "replay-b")
    first_result = write_jsonl_zstd_shards_v3(
        documents,
        roots[0],
        stream="T",
        stratum="general",
        shard_target_bytes=target,
    )
    second_result = write_jsonl_zstd_shards_v3(
        documents,
        roots[1],
        stream="T",
        stratum="general",
        shard_target_bytes=target,
    )

    assert first_result == second_result
    assert first_result.invalid_utf8_by_source == (("dolma_web", 1),)
    assert first_result.valid_record_count == 4
    assert first_result.oversized_singleton_count == 1
    assert len(first_result.shards) == 4
    for shard in first_result.shards:
        first_path = roots[0] / shard.relative_path
        second_path = roots[1] / shard.relative_path
        assert first_path.read_bytes() == second_path.read_bytes()
        logical = _read_zstd(first_path)
        assert hashlib.sha256(logical).hexdigest() == shard.logical_jsonl_sha256
        for line in logical.splitlines():
            row = json.loads(line)
            assert list(row) == ["id", "source", "stratum", "text"]
            assert row["id"] == hashlib.sha1(  # noqa: S324 - A2 contract
                row["text"].encode("utf-8")
            ).hexdigest()
        if shard.logical_jsonl_bytes > target:
            assert shard.record_count == 1
    assert not tuple(tmp_path.rglob("*.partial"))

    with pytest.raises(CorpusProductionError, match="overwrite"):
        write_jsonl_zstd_shards_v3(
            documents,
            roots[0],
            stream="T",
            stratum="general",
            shard_target_bytes=target,
        )


def test_fixture_replays_keep_content_identity_and_withhold_gate_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "schema": "weft1_corpus_pa_fixture_v3",
                "stream": "T",
                "stratum": "general",
                "shard_target_bytes": 170,
                "documents": [
                    {
                        "source": "dolma_web",
                        "stable_source_record_id": _hash("one"),
                        "text": "one",
                    },
                    {
                        "source": "fineweb_edu",
                        "stable_source_record_id": _hash("invalid"),
                        "text_utf8_hex": "ff",
                    },
                    {
                        "source": "dolma_web",
                        "stable_source_record_id": _hash("two"),
                        "text": "two",
                    },
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
        newline="\n",
    )
    fake_runtime = RuntimeAttestationV3(
        executable_sha256=_hash("executable"),
        dependency_lock_sha256=_hash("lock"),
        environment_identity_sha256=_hash("environment"),
        environment_payload={"fixture": True},
    )
    monkeypatch.setattr(production, "attest_runtime_v3", lambda: fake_runtime)

    first_root = tmp_path / "first-output"
    second_root = tmp_path / "second-output"
    first = run_fixture_replay(fixture, first_root)
    second = run_fixture_replay(fixture, second_root)
    assert first["content_identity_sha256"] == second["content_identity_sha256"]
    assert first["shard_identity_sha256s"] == second["shard_identity_sha256s"]
    assert first["run_id"] != second["run_id"]
    assert first["authoritative_gate_receipts"] == []
    assert first["typed_replay_ready"] is False
    assert first["invalid_utf8_by_source"] == (("fineweb_edu", 1),)
    first_files = sorted(first_root.rglob("*.jsonl.zst"))
    second_files = sorted(second_root.rglob("*.jsonl.zst"))
    assert [path.relative_to(first_root) for path in first_files] == [
        path.relative_to(second_root) for path in second_files
    ]
    assert [path.read_bytes() for path in first_files] == [
        path.read_bytes() for path in second_files
    ]

    typed = typed_replay_receipt_from_mapping(
        first,
        minhash_recall_audit=_recall_audit(),
    )
    assert typed.content_manifest.content_identity_sha256 == first[
        "content_identity_sha256"
    ]
    assert typed.minhash_recall_audit == _recall_audit()
    with pytest.raises(CorpusProductionError, match="offline-only"):
        run_fixture_replay(
            fixture,
            tmp_path / "must-not-run",
            network_disabled=False,
        )


def test_all_invalid_utf8_documents_fail_without_partial_shard(tmp_path: Path) -> None:
    root = tmp_path / "invalid-only"
    with pytest.raises(CorpusProductionError, match="no valid documents"):
        write_jsonl_zstd_shards_v3(
            (RawDocumentV3("dolma_web", "general", _hash("bad"), b"\xff"),),
            root,
            stream="H",
            stratum="general",
            shard_target_bytes=100,
        )
    assert not tuple(root.rglob("*.jsonl.zst"))
    assert not tuple(root.rglob("*.partial"))


def test_default_process_preflight_is_bound_to_current_python_executable() -> None:
    assert Path(sys.executable).is_file()
    assert os.getpid() > 0
