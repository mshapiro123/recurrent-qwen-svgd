from __future__ import annotations

from pathlib import Path

from colab.run_stage5_inverse_table_rebase import (
    SOURCE_CAP2_SHA256,
    assert_source_cap2,
    rebase_decision,
)


def source_payload() -> dict:
    return {
        "kind": "stage5_inverse_composition_staircase",
        "status": "blocked_at_cap_2",
        "arms": {
            "C": {
                "stages": [
                    {
                        "cap": 2,
                        "status": "advanced",
                        "checkpoint_sha256": SOURCE_CAP2_SHA256,
                        "checkpoint_drive_backup": "/drive/C_cap2.pt",
                        "gate": {"passed": True, "correct": 62, "total": 64},
                        "synthetic_guardrail": {"passed": True, "active_diagonal_min": 0.9375},
                    }
                ]
            }
        },
    }


def test_source_cap2_requires_exact_green_control_checkpoint() -> None:
    stage = assert_source_cap2(source_payload())

    assert stage["cap"] == 2
    assert stage["checkpoint_sha256"] == SOURCE_CAP2_SHA256


def test_rebase_decision_requires_caps_three_and_four_plus_guardrails() -> None:
    green = [
        {"cap": 3, "gate": {"passed": True}, "synthetic_guardrail": {"passed": True}},
        {"cap": 4, "gate": {"passed": True}, "synthetic_guardrail": {"passed": True}},
    ]
    stalled = [green[0], {"cap": 4, "gate": {"passed": False}, "synthetic_guardrail": {"passed": True}}]

    assert rebase_decision(green) == "rebase_caps3_4_green_pending_review"
    assert rebase_decision(stalled) == "rebase_blocked"


def test_bootstrap_exposes_rebase_target() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert '"inverse_table_rebase_caps3_4"' in bootstrap
    assert Path("colab/STAGE5_INVERSE_TABLE_REBASE_CELL.py").exists()
