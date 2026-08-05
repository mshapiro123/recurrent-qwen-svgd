from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.run_paper2_phase2_a2_amendment_prep import _assert_close, _markdown


ROOT = Path(__file__).resolve().parents[1]


def test_assert_close_is_strict_enough_for_receipt_reconciliation() -> None:
    _assert_close(1.0, 1.0 + 1e-11, label="close")
    with pytest.raises(RuntimeError, match="mismatch"):
        _assert_close(1.0, 1.01, label="different")


def test_markdown_states_unlocked_boundary_and_directional_contract() -> None:
    summary = {
        "verification_status": "two_seed_public_private_receipts_reconciled",
        "pathology_verdict": "clear_for_amendment_review",
        "arms": [
            {
                "seed": 0,
                "primary_cosine": {"mean": 0.72, "fraction_below_minus_0p5": 0.0},
                "raw_norm_spread": 2000.0,
                "a1_raw_norm_spread": 300000.0,
                "candidate_clip_catastrophe_tripwire": 2.4,
            }
        ],
    }
    text = _markdown(summary)
    assert "draft_unlocked_strategy_review_required" in text
    assert "at least 50%" in text
    assert "at 25%" in text
    assert "A2 training launched: `false`" in text


def test_cpu_prep_source_has_no_model_or_optimizer_path() -> None:
    source = (ROOT / "training/run_paper2_phase2_a2_amendment_prep.py").read_text(
        encoding="utf-8"
    )
    assert "import torch" not in source
    assert "torch.optim" not in source
    assert '"a2_training_authorized": False' in source
    assert '"automatic_lock_or_launch": False' in source
    assert "draft_unlocked_strategy_review_required" in source


def test_public_calibration_is_complete_and_awaiting_amendment() -> None:
    summary = json.loads(
        (
            ROOT
            / "outputs/stage5/stage5_paper2_phase2_a2_calibration_20260805/summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["status"] == "complete_with_a2_amendment_required"
    assert summary["optimizer_updates"] == 0
    assert summary["a2_training_launched"] is False
    assert [arm["status"] for arm in summary["arms"]] == [
        "complete_zero_update",
        "complete_zero_update",
    ]
