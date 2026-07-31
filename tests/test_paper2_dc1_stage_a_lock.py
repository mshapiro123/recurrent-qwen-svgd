from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.eval_paper2_dc1_stage_a_verdict import compute_verdict


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "docs/stage_a_prereg.json"
GOVERNING = ROOT / "docs/PHASE_DC1_STAGE_A_PREREGISTRATION_DRAFT1_20260730.md"


def test_stage_a_lock_has_no_placeholders_and_exact_lineage() -> None:
    prereg = json.loads(LOCK.read_text(encoding="utf-8"))
    assert prereg["locked_before_training"] is True
    assert "TRANSCRIBE" not in LOCK.read_text(encoding="utf-8")
    assert "ENUMERATE" not in LOCK.read_text(encoding="utf-8")
    assert "NAME_AT_COMMIT" not in LOCK.read_text(encoding="utf-8")
    assert prereg["trainable"]["allowlist"] == ["horizontal_bridge.delta.weight"]
    assert prereg["eval_partition"]["manifest_sha256"] == (
        "7813d9502b06c89ba210191d4bb29ec52d217ee7177ff9397469c85831bc0123"
    )
    assert prereg["eval_partition"]["teacher_cache_sha256"] == (
        "67404785a93cc1337bdebea3d05c8b6094e20be4645e23687866281e4d89d44d"
    )


def test_governing_document_bytes_match_drive_lock() -> None:
    payload = GOVERNING.read_bytes()
    assert len(payload) == 14333
    assert hashlib.sha256(payload).hexdigest() == (
        "bd834c42d92b559dabd638c326dd76724f24adba6ade27bcdd4adb32703dc581"
    )


def test_registered_verdict_qualifies_only_with_point_and_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    prereg = json.loads(LOCK.read_text(encoding="utf-8"))
    prereg["evaluation"]["bootstrap"]["replicates"] = 8
    rows = [
        {
            "row_id": "a",
            "scored_positions": 100,
            "arms": {
                "trained_append_k1": {"helps": 5, "hurts": 4},
                "untrained_append_k1": {"helps": 2, "hurts": 20},
            },
        },
        {
            "row_id": "b",
            "scored_positions": 100,
            "arms": {
                "trained_append_k1": {"helps": 4, "hurts": 3},
                "untrained_append_k1": {"helps": 2, "hurts": 20},
            },
        },
    ]
    result = compute_verdict(rows, prereg)
    assert result["verdict"] == "qualifies"
    assert result["criteria"]["qualifies"] is True


def test_registered_verdict_partial_and_none_paths() -> None:
    prereg = json.loads(LOCK.read_text(encoding="utf-8"))
    prereg["evaluation"]["bootstrap"]["replicates"] = 8
    partial_rows = [
        {
            "row_id": "a",
            "scored_positions": 100,
            "arms": {
                "trained_append_k1": {"helps": 10, "hurts": 11},
                "untrained_append_k1": {"helps": 10, "hurts": 30},
            },
        }
    ]
    assert compute_verdict(partial_rows, prereg)["verdict"] == "partial_domestication"

    none_rows = [
        {
            "row_id": "a",
            "scored_positions": 100,
            "arms": {
                "trained_append_k1": {"helps": 4, "hurts": 25},
                "untrained_append_k1": {"helps": 10, "hurts": 30},
            },
        }
    ]
    result = compute_verdict(none_rows, prereg)
    assert result["verdict"] == "none"
    assert result["criteria"]["no_material_improvement_threshold_met"] is True
