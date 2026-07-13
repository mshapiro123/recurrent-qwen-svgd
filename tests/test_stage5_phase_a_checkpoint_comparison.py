from __future__ import annotations

from pathlib import Path

from colab.run_stage5_phase_a_checkpoint_comparison import (
    CHECKPOINT_SPECS,
    build_comparison,
)


def _eval_rows(predictions: list[str]) -> list[dict]:
    source = [
        ("r1", 1, "B"),
        ("r2", 2, "D"),
        ("r3", 3, "H"),
    ]
    return [
        {
            "id": row_id,
            "depth": depth,
            "target": target,
            "prediction": prediction,
            "correct": prediction == target,
            "continuation": f"answer: {prediction}",
        }
        for (row_id, depth, target), prediction in zip(source, predictions)
    ]


def test_checkpoint_matrix_is_complete_and_unique() -> None:
    assert {(row["arm"], row["step"]) for row in CHECKPOINT_SPECS} == {
        (arm, step) for arm in "BCD" for step in (2000, 4000)
    }
    assert len({row["label"] for row in CHECKPOINT_SPECS}) == 6
    assert all("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/" in row["checkpoint"] for row in CHECKPOINT_SPECS)
    assert all((row["reference_summary"] is not None) == (row["step"] == 4000) for row in CHECKPOINT_SPECS)


def test_comparison_is_paired_and_classifies_depth2_early_errors() -> None:
    source_rows = [
        {"id": "r1", "depth": 1, "target": "B", "orbit": ["A", "B"]},
        {"id": "r2", "depth": 2, "target": "D", "orbit": ["A", "C", "D"]},
        {"id": "r3", "depth": 3, "target": "H", "orbit": ["A", "E", "F", "H"]},
    ]
    rows = {
        "B_step2000": _eval_rows(["B", "C", "F"]),
        "B_step4000": _eval_rows(["B", "D", "F"]),
        "C_step2000": _eval_rows(["B", "C", "H"]),
        "C_step4000": _eval_rows(["B", "D", "H"]),
        "D_step2000": _eval_rows(["A", "C", "G"]),
        "D_step4000": _eval_rows(["B", "C", "G"]),
    }

    summary, paired = build_comparison(rows, source_rows)

    assert summary["within_arm"]["B"]["helped"] == 1
    assert summary["within_arm"]["B"]["hurt"] == 0
    assert summary["depth2_error_classes"]["C_step2000"]["one_step_early"] == 1
    assert summary["depth2_error_classes"]["C_step4000"]["correct"] == 1
    assert paired[1]["predictions"]["C_step2000"] == "C"
    assert paired[1]["correct"]["C_step4000"] is True


def test_bootstrap_exposes_checkpoint_comparison_target() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    assert '"phase_a_checkpoint_comparison"' in bootstrap
    assert "STAGE5_PHASE_A_CHECKPOINT_COMPARISON_CELL_VERSION" in bootstrap
    assert "tests/test_stage5_phase_a_checkpoint_comparison.py" in bootstrap
