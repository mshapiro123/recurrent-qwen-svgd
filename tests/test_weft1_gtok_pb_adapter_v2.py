from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import training.weft1_gtok_pb_adapter_v2 as adapter
from training.weft1_corpus_materialize_a3 import D6_PHYSICAL_EVIDENCE_SCHEMA_V4
from training.weft1_gtok_contract import (
    GTOK_SCREEN_HELDOUT_STRATUM_TARGETS,
    GTOK_SCREEN_TRAIN_STRATUM_TARGETS,
    canonical_json_bytes,
)
from training.weft1_gtok_v2_contract import A2FirstFitScreenReceiptV2


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _physical_d6() -> dict[str, object]:
    groups = []
    for stream, targets in (
        ("T", GTOK_SCREEN_TRAIN_STRATUM_TARGETS),
        ("H", GTOK_SCREEN_HELDOUT_STRATUM_TARGETS),
    ):
        for stratum, target in targets:
            groups.append(
                {
                    "document_count": 10,
                    "ordered_raw_content_ids_sha256": _hash(
                        f"{stream}-{stratum}-raw-order"
                    ),
                    "retained_text_bytes": target,
                    "stratum": stratum,
                    "stream": stream,
                }
            )
    streams = []
    for stream, targets in (
        ("T", GTOK_SCREEN_TRAIN_STRATUM_TARGETS),
        ("H", GTOK_SCREEN_HELDOUT_STRATUM_TARGETS),
    ):
        streams.append(
            {
                "document_count": 40,
                "framed_retained_text_sha256": _hash(f"{stream}-framed"),
                "retained_text_bytes": sum(target for _, target in targets),
                "stream": stream,
            }
        )
    return {
        "document_overlap_count": 0,
        "evidence_identity_sha256": _hash("physical-d6-identity"),
        "schema": D6_PHYSICAL_EVIDENCE_SCHEMA_V4,
        "split_groups": groups,
        "stream_identities": streams,
    }


def _diagnostic_d6() -> dict[str, object]:
    training = dict(GTOK_SCREEN_TRAIN_STRATUM_TARGETS)
    heldout = dict(GTOK_SCREEN_HELDOUT_STRATUM_TARGETS)
    return {
        "cluster_overlap_count": 0,
        "document_overlap_count": 0,
        "split_rows": [
            {
                "heldout": {
                    "deficit_bytes": 0,
                    "document_count": 10,
                    "realized_bytes": heldout[stratum],
                    "target_bytes": heldout[stratum],
                },
                "stratum": stratum,
                "training": {
                    "deficit_bytes": 0,
                    "document_count": 10,
                    "realized_bytes": training[stratum],
                    "target_bytes": training[stratum],
                },
            }
            for stratum in training
        ],
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "materialized"
    physical = _physical_d6()
    d6_physical = _write(root / "artifacts" / "d6-physical.json", physical)
    diagnostic = _diagnostic_d6()
    d6_diagnostic = _write(root / "diagnostics" / "d6.json", diagnostic)
    pa = SimpleNamespace(
        root=root,
        full_shard_manifest_physical_sha256=_hash("full"),
        screen_submanifest_physical_sha256=_hash("screen"),
        d6_physical_evidence_physical_sha256=d6_physical,
        d6_physical_evidence_identity_sha256=physical[
            "evidence_identity_sha256"
        ],
        d6_physical_evidence_relative_path="artifacts/d6-physical.json",
        diagnostic_sha256s=(("D6", d6_diagnostic),),
    )
    paths = {
        "materialization_root": root,
        "freeze_receipt_path": tmp_path / "freeze.json",
        "gate_bundle_path": tmp_path / "gates.json",
        "c2_evidence_path": tmp_path / "c2.json",
        "decon_receipt_path": tmp_path / "decon.json",
    }
    recomputed = {
        "d1_d6_gate_bundle_sha256": _hash("gates"),
        "d6_physical_evidence_sha256": d6_physical,
        "decon_receipt_sha256": _hash("decon"),
        "freeze_receipt_identity_sha256": _hash("freeze-identity"),
        "full_shard_manifest_sha256": pa.full_shard_manifest_physical_sha256,
        "screen_submanifest_sha256": pa.screen_submanifest_physical_sha256,
        "status": "FROZEN",
    }
    _write(paths["freeze_receipt_path"], recomputed)
    calls: list[dict[str, Path]] = []

    monkeypatch.setattr(adapter, "inspect_pa_v4", lambda value: pa)

    def _revalidate(**kwargs):
        calls.append(kwargs)
        return recomputed

    monkeypatch.setattr(adapter, "build_freeze_receipt", _revalidate)
    return paths, pa, recomputed, calls


def test_adapter_revalidates_then_projects_exact_a2_first_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, pa, _, calls = _fixture(tmp_path, monkeypatch)
    corpus = adapter.load_frozen_screen_corpus_v2(**paths)
    assert isinstance(corpus.first_fit, A2FirstFitScreenReceiptV2)
    assert len(corpus.first_fit.groups) == 8
    assert corpus.training_realized_bytes == 4_000_000_000
    assert corpus.heldout_realized_bytes == 80_000_000
    assert corpus.d6_physical_evidence_sha256 == pa.d6_physical_evidence_physical_sha256
    assert calls == [
        {
            "materialization_root": paths["materialization_root"],
            "gate_bundle_path": paths["gate_bundle_path"],
            "c2_evidence_path": paths["c2_evidence_path"],
            "decon_receipt_path": paths["decon_receipt_path"],
        }
    ]


def test_adapter_rejects_mutated_stored_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _, recomputed, _ = _fixture(tmp_path, monkeypatch)
    mutated = dict(recomputed)
    mutated["d1_d6_gate_bundle_sha256"] = _hash("substituted-gates")
    _write(paths["freeze_receipt_path"], mutated)
    with pytest.raises(adapter.GTokPBAdapterError, match="differs from fresh"):
        adapter.load_frozen_screen_corpus_v2(**paths)


def test_adapter_rejects_materialization_substitution_after_pb_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, pa, recomputed, _ = _fixture(tmp_path, monkeypatch)
    substituted = dict(recomputed)
    substituted["screen_submanifest_sha256"] = _hash("different-screen")
    _write(paths["freeze_receipt_path"], substituted)
    monkeypatch.setattr(adapter, "build_freeze_receipt", lambda **_: substituted)
    assert substituted["screen_submanifest_sha256"] != pa.screen_submanifest_physical_sha256
    with pytest.raises(adapter.GTokPBAdapterError, match="substitutes"):
        adapter.load_frozen_screen_corpus_v2(**paths)


def test_adapter_rejects_raw_content_id_evidence_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, pa, _, _ = _fixture(tmp_path, monkeypatch)
    path = pa.root / pa.d6_physical_evidence_relative_path
    physical = _physical_d6()
    physical["split_groups"][0]["ordered_raw_content_ids_sha256"] = _hash("mutation")
    _write(path, physical)
    with pytest.raises(adapter.GTokPBAdapterError, match="physical D6 identity"):
        adapter.load_frozen_screen_corpus_v2(**paths)


def test_a2_contract_rejects_group_reordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _, _, _ = _fixture(tmp_path, monkeypatch)
    valid = adapter.load_frozen_screen_corpus_v2(**paths).first_fit
    with pytest.raises(ValueError, match="canonical eight"):
        replace(valid, groups=tuple(reversed(valid.groups)))
