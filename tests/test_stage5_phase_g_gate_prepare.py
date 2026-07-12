from __future__ import annotations

import pytest

from colab.run_stage5_phase_g_gate_prepare import keeper_gate_from_receipt


def eval_row(checkpoint: str, values: list[float]) -> dict:
    return {
        "checkpoint": checkpoint,
        "active_diagonal": {str(index): value for index, value in enumerate(values, start=1)},
        "active_total": {"accuracy": sum(values) / len(values)},
    }


def receipt_fixture(synthetic_min: float = 0.953125) -> dict:
    checkpoint = "/drive/keeper-step2000.pt"
    natural = [0.98, 0.96, 0.92, 0.92, 0.89, 0.90, 0.83, 0.84, 0.78, 0.72, 0.64, 0.59]
    synthetic = [1.0] * 10 + [synthetic_min, 0.96]
    return {
        "source_run_id": "natural-transfer",
        "evals": {
            "step_2000": {
                "paired_relay_d1_12": eval_row(checkpoint, natural),
                "paired_pointer_d1_12": eval_row(checkpoint, natural),
                "synthetic_frozen_v3_d1_12": eval_row(checkpoint, synthetic),
            }
        },
        "receipts": {
            "drive_checkpoint_backup": {
                "checkpoint_files": [
                    {"step": 2000, "dest": checkpoint, "sha256": "abc", "bytes": 123}
                ]
            }
        },
    }


def test_keeper_gate_locks_green_step2000_receipt() -> None:
    result = keeper_gate_from_receipt(receipt_fixture())

    assert result["status"] == "green"
    assert result["checkpoint_sha256"] == "abc"
    assert result["synthetic_full_width_min_1_12"] == pytest.approx(0.953125)
    assert result["relay_min_1_8"] == pytest.approx(0.83)
    assert result["relay_min_9_12"] == pytest.approx(0.59)


def test_keeper_gate_blocks_guardrail_regression() -> None:
    result = keeper_gate_from_receipt(receipt_fixture(synthetic_min=0.90))

    assert result["status"] == "blocked"
    assert result["synthetic_guardrail_pass"] is False


def test_keeper_gate_rejects_mismatched_checkpoint_receipts() -> None:
    receipt = receipt_fixture()
    receipt["evals"]["step_2000"]["paired_pointer_d1_12"]["checkpoint"] = "/drive/other.pt"

    with pytest.raises(RuntimeError, match="do not share one checkpoint"):
        keeper_gate_from_receipt(receipt)

