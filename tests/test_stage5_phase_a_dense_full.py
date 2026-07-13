from __future__ import annotations

from pathlib import Path

from colab.run_stage5_phase_a_dense_full import (
    EVAL_SHA256,
    TRAIN_SHA256,
    arm_spec,
    build_arm_rows,
    parse_arms,
)
from eval.eval_synthetic_depth_dense import extract_final_symbol, summarize_rows


def test_arm_specs_lock_full_adamw_and_model_revisions() -> None:
    b = arm_spec("B")
    c = arm_spec("C")
    d = arm_spec("D")

    assert b["training_surface"] == "direct"
    assert c["training_surface"] == "serialized_orbit_scratchpad"
    assert d["model_name"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert {b["optimizer"], c["optimizer"], d["optimizer"]} == {"adamw_full_fp32_state"}
    assert b["revision"] and d["revision"]


def test_parse_arms_is_strict_and_stable() -> None:
    assert parse_arms("B,C") == ["B", "C"]
    assert parse_arms("D") == ["D"]


def test_build_arm_rows_changes_only_completion_surface() -> None:
    source = [
        {
            "instance_id": "row-1",
            "prompt": "Question\nAnswer:",
            "completion": " C",
            "orbit": ["A", "B", "C"],
            "depth": 2,
            "target": "C",
        }
    ]

    direct = build_arm_rows(source, surface="direct")
    scratchpad = build_arm_rows(source, surface="serialized_orbit_scratchpad")

    assert direct[0]["prompt"] == scratchpad[0]["prompt"] == source[0]["prompt"]
    assert direct[0]["completion"] == " C"
    assert scratchpad[0]["completion"] == " steps: B -> C answer: C"
    assert scratchpad[0]["instance_id"] == direct[0]["instance_id"]


def test_dense_reader_uses_last_valid_symbol_for_both_surfaces() -> None:
    candidates = list("ABCD")

    assert extract_final_symbol(" C", candidates) == "C"
    assert extract_final_symbol(" steps: B -> C answer: C", candidates) == "C"
    assert extract_final_symbol("steps: A -> D; answer: D\n", candidates) == "D"
    assert extract_final_symbol("This is a guess: C", candidates) == "C"
    assert extract_final_symbol("C then D", candidates) == "C"


def test_dense_summary_is_depth_stratified() -> None:
    payload = summarize_rows(
        [
            {"depth": 1, "correct": True},
            {"depth": 1, "correct": False},
            {"depth": 2, "correct": True},
        ]
    )

    assert payload["by_depth"]["1"]["correct"] == 1
    assert payload["by_depth"]["1"]["total"] == 2
    assert payload["by_depth"]["2"]["accuracy"] == 1.0


def test_phase_a_hashes_and_bootstrap_target_are_locked() -> None:
    assert TRAIN_SHA256 == "260d5c11c0b6e97d1f09c9356b1eaedbde86cceac4053cc6bf561e53d0176bde"
    assert EVAL_SHA256 == "aaa71c3d4cc500f68fac7ee6f5f0e31d9e11570bdff90adb805c769c12c66cd3"
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    assert '"phase_a_dense_full"' in bootstrap
