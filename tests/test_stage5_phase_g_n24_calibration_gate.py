from __future__ import annotations

from pathlib import Path

import pytest

from colab.run_stage5_phase_g_n24_calibration_gate import assess_calibration, source_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def summary(*, pooled: float = 0.80, depth: float = 0.70) -> dict:
    return {
        "overall": {"greedy_valid_rate": pooled},
        "by_depth": {str(index): {"greedy_valid_rate": depth} for index in range(1, 5)},
        "by_preimage_stratum": {
            "unique": {"greedy_valid_rate": 0.85},
            "small": {"greedy_valid_rate": 0.70},
            "large": {"greedy_valid_rate": 0.65},
        },
    }


def test_calibration_gate_requires_pooled_depth_and_each_stratum() -> None:
    assert assess_calibration(summary())["passed"] is True
    failed = summary()
    failed["by_preimage_stratum"]["large"]["greedy_valid_rate"] = 0.59
    assert assess_calibration(failed)["passed"] is False


def test_source_checkpoint_requires_complete_green_experiment1() -> None:
    source = {
        "status": "experiment1_passed",
        "abductive_gate": {"passed": True},
        "synthetic_guardrail": {"passed": True},
        "abductive_train": {
            "checkpoint_drive_backup": "/drive/checkpoint.pt",
            "checkpoint_sha256": "abc",
        },
    }
    assert source_checkpoint(source) == ("/drive/checkpoint.pt", "abc")

    source["status"] = "abductive_trained"
    with pytest.raises(RuntimeError, match="not complete"):
        source_checkpoint(source)


def test_bootstrap_exposes_n24_calibration_without_test_split() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert '"phase_g_n24_calibration_gate"' in text
    assert "STAGE5_PHASE_G_N24_CALIBRATION_CELL_VERSION" in text
    launcher = (ROOT / "colab/STAGE5_PHASE_G_N24_CALIBRATION_CELL.py").read_text(encoding="utf-8")
    assert "calibration_n24.jsonl" in launcher
    assert "test_n24.jsonl" not in launcher
