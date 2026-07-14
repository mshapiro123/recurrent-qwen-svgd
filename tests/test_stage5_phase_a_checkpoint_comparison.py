from __future__ import annotations

from pathlib import Path

from colab.run_stage5_phase_a_checkpoint_comparison import (
    CHECKPOINT_SPECS,
    _compress_completed_rows,
    build_repeatability_receipt,
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


def test_repeatability_receipt_accepts_small_nonexact_gpu_rerun_delta() -> None:
    reference = {
        "reader": "answer_marker_else_first_valid_full_symbol",
        "correct": 320,
        "total": 1792,
        "accuracy": 320 / 1792,
        "max_new_tokens": 32,
        "by_depth": {
            "1": {"correct": 15, "total": 128, "parse_failures": 0, "accuracy": 15 / 128},
            "2": {"correct": 17, "total": 128, "parse_failures": 0, "accuracy": 17 / 128},
        },
    }
    current = {
        **reference,
        "correct": 322,
        "accuracy": 322 / 1792,
        "by_depth": {
            "1": {"correct": 13, "total": 128, "parse_failures": 0, "accuracy": 13 / 128},
            "2": {"correct": 16, "total": 128, "parse_failures": 0, "accuracy": 16 / 128},
        },
    }

    receipt = build_repeatability_receipt(current, reference)

    assert receipt["status"] == "within_gpu_repeatability_envelope"
    assert receipt["exact"] is False
    assert receipt["correct_delta"] == 2
    assert receipt["max_abs_depth_correct_delta"] == 2
    assert receipt["structural_checks_pass"] is True


def test_repeatability_receipt_rejects_material_delta() -> None:
    reference = {
        "reader": "reader",
        "correct": 100,
        "total": 128,
        "accuracy": 100 / 128,
        "max_new_tokens": 32,
        "by_depth": {"1": {"correct": 100, "total": 128, "parse_failures": 0, "accuracy": 100 / 128}},
    }
    current = {
        **reference,
        "correct": 90,
        "accuracy": 90 / 128,
        "by_depth": {"1": {"correct": 90, "total": 128, "parse_failures": 0, "accuracy": 90 / 128}},
    }

    assert build_repeatability_receipt(current, reference)["status"] == "outside_gpu_repeatability_envelope"


def test_resume_compresses_completed_raw_rows_without_reevaluation(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval" / "D_step4000"
    eval_dir.mkdir(parents=True)
    raw = eval_dir / "rows.jsonl"
    raw.write_text('{"id":"row-1","correct":true}\n', encoding="utf-8")
    (eval_dir / "summary.json").write_text("{}\n", encoding="utf-8")

    assert _compress_completed_rows(eval_dir) is True
    assert not raw.exists()
    assert (eval_dir / "rows.jsonl.gz").exists()
