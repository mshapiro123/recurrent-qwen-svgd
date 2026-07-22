from __future__ import annotations

from pathlib import Path

from colab.run_stage5_phase_a_dense_full import (
    EVAL_SHA256,
    EVAL_SOURCE,
    TRAIN_SHA256,
    TRAIN_SOURCE,
    arm_spec,
    build_arm_rows,
    parse_arms,
    sha256_jsonl_content,
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


def test_dense_reader_uses_first_completed_response_for_both_surfaces() -> None:
    candidates = list("ABCD")

    assert extract_final_symbol(" C", candidates) == "C"
    assert extract_final_symbol(" steps: B -> C answer: C", candidates) == "C"
    assert extract_final_symbol("steps: A -> D; answer: D\n", candidates) == "D"
    assert extract_final_symbol("This is a guess: C", candidates) == "C"
    assert extract_final_symbol("C then D", candidates) == "C"

    # Direct completions are a leading symbol. Later generated examples must
    # not overwrite the response to the current prompt.
    assert extract_final_symbol(
        " C\n\nStart value: A\nApply f exactly 1 times.\nAnswer: D",
        candidates,
    ) == "C"

    # Scratchpad completions end at their first answer marker. Repetition after
    # that marker is untrained continuation, not a revised answer.
    assert extract_final_symbol(
        " steps: B -> C answer: C answer: D answer: A",
        candidates,
    ) == "C"


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
    assert TRAIN_SHA256 == "cf61c14c2629f2caa7e1b6bd100adb122a468d5285b74970aaa4aebfbb56fd12"
    assert EVAL_SHA256 == "3de844669aba303063e6932f5852914ee0993e531c8e65c2a4c4b18e219b3fc8"
    assert sha256_jsonl_content(TRAIN_SOURCE) == TRAIN_SHA256
    assert sha256_jsonl_content(EVAL_SOURCE) == EVAL_SHA256
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    assert '"phase_a_dense_full"' in bootstrap
    assert 'os.environ.get("STAGE5_BOOTSTRAP_REF", "main")' in bootstrap
    assert "if is_commit_sha(REF):" in bootstrap


def test_jsonl_content_hash_is_newline_invariant(tmp_path: Path) -> None:
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    content = b'{"id":1}\n{"id":2}\n'
    lf.write_bytes(content)
    crlf.write_bytes(content.replace(b"\n", b"\r\n"))

    assert sha256_jsonl_content(lf) == sha256_jsonl_content(crlf)
