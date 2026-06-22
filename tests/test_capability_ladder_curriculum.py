from __future__ import annotations

import json

from training.build_capability_ladder_curriculum import build_records, main
from training.prepare_curriculum_jsonl import convert_curriculum_records, validate_curriculum_record


def scored_row(
    row_id: str,
    *,
    base_correct: bool,
    mid_correct: bool,
    high_correct: bool,
    decontaminated: bool = True,
    verified_by: list[str] | None = None,
) -> dict:
    answer = "42"
    return {
        "id": row_id,
        "domain": "math",
        "question": f"Problem {row_id}: what is the answer?",
        "answer": {"value": answer, "verified_by": verified_by or ["cross_model"]},
        "decontaminated": decontaminated,
        "model_results": {
            "qwen_0_5b": {
                "correct": base_correct,
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "solution": f"ANSWER: {answer}",
                "steps": 1,
            },
            "qwen_1_5b": {
                "correct": mid_correct,
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "solution": f"Compute carefully. ANSWER: {answer}",
                "steps": 3,
            },
            "qwen_3b": {
                "correct": high_correct,
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "solution": f"Use a longer derivation. ANSWER: {answer}",
                "steps": 7,
            },
        },
    }


def build_default(rows: list[dict]):
    return build_records(
        rows,
        base_key="qwen_0_5b",
        mid_key="qwen_1_5b",
        high_keys=["qwen_3b"],
        high_target_loop=3,
        allow_answer_only=False,
        assume_decontaminated=False,
    )


def test_build_records_assigns_depth_by_capability_ladder() -> None:
    records, report = build_default(
        [
            scored_row("base-known", base_correct=True, mid_correct=True, high_correct=True),
            scored_row("mid-only", base_correct=False, mid_correct=True, high_correct=True),
            scored_row("high-only", base_correct=False, mid_correct=False, high_correct=True),
        ]
    )

    assert report["exported_records"] == 3
    by_id = {record["id"]: record for record in records}
    assert by_id["base-known"]["mode"] == "direct"
    assert by_id["base-known"]["target_loop_count"] == 1
    assert by_id["base-known"]["capability_tier"] == "base_preservation"
    assert by_id["mid-only"]["mode"] == "deep_narrow"
    assert by_id["mid-only"]["target_loop_count"] == 2
    assert by_id["mid-only"]["capability_tier"] == "qwen_0_5b_miss_qwen_1_5b_solve"
    assert by_id["high-only"]["mode"] == "deep_narrow"
    assert by_id["high-only"]["target_loop_count"] == 3
    assert by_id["high-only"]["capability_tier"] == "qwen_0_5b_miss_qwen_1_5b_miss_stronger_solve"
    assert report["tier_counts"] == {
        "base_preservation": 1,
        "qwen_0_5b_miss_qwen_1_5b_miss_stronger_solve": 1,
        "qwen_0_5b_miss_qwen_1_5b_solve": 1,
    }
    assert all(validate_curriculum_record(record) == [] for record in records)


def test_build_records_skips_unverified_and_unresolved_rows() -> None:
    records, report = build_default(
        [
            scored_row("unverified", base_correct=True, mid_correct=True, high_correct=True, verified_by=["self_report"]),
            scored_row("unresolved", base_correct=False, mid_correct=False, high_correct=False),
        ]
    )

    assert records == []
    assert report["skipped"] == {
        "failed_safety_or_trace_requirements": 1,
        "unresolved_capability": 1,
    }


def test_capability_ladder_records_convert_to_positive_sft_with_metadata() -> None:
    records, _report = build_default(
        [
            scored_row("mid-only", base_correct=False, mid_correct=True, high_correct=True),
        ]
    )

    examples, sft_report = convert_curriculum_records(records)

    assert sft_report["exported_examples"] == 1
    assert examples[0]["target_loop_count"] == 2
    assert examples[0]["capability_tier"] == "qwen_0_5b_miss_qwen_1_5b_solve"
    assert examples[0]["capability_ladder"]["qwen_0_5b_correct"] is False
    assert examples[0]["capability_ladder"]["qwen_1_5b_correct"] is True
    assert examples[0]["answer_match"]["matched"] is True
    assert examples[0]["source_model"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_capability_ladder_cli_writes_gate_ready_artifacts(tmp_path) -> None:
    input_jsonl = tmp_path / "scored.jsonl"
    work_dir = tmp_path / "capability_ladder"
    rows = [
        scored_row("base-known", base_correct=True, mid_correct=True, high_correct=True),
        scored_row("mid-only", base_correct=False, mid_correct=True, high_correct=True),
        scored_row("high-only", base_correct=False, mid_correct=False, high_correct=True),
    ]
    input_jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert main(["--input_jsonl", str(input_jsonl), "--work_dir", str(work_dir)]) == 0

    summary = json.loads((work_dir / "summary.json").read_text(encoding="utf-8"))
    typed_rows = [json.loads(line) for line in (work_dir / "typed_records.jsonl").read_text(encoding="utf-8").splitlines()]
    sft_rows = [json.loads(line) for line in (work_dir / "positive_sft.jsonl").read_text(encoding="utf-8").splitlines()]
    report = json.loads((work_dir / "capability_ladder_report.json").read_text(encoding="utf-8"))

    assert summary["kind"] == "capability_ladder_curriculum_pipeline"
    assert summary["status"] == "complete"
    assert summary["counts"]["target_loop_counts"] == {"1": 1, "2": 1, "3": 1}
    assert all("\\" not in item["path"] for item in summary["artifacts"].values())
    assert len(typed_rows) == 3
    assert len(sft_rows) == 3
    assert report["tier_counts"]["base_preservation"] == 1
