from __future__ import annotations

import json

from training.build_capability_ladder_curriculum import build_records
from training.merge_capability_score_rows import main, merge_rows


def task(row_id: str, answer: str = "B") -> dict:
    return {
        "id": row_id,
        "question": f"Question {row_id}?",
        "choices": {"A": "wrong", "B": "right", "C": "also wrong"},
        "answer": answer,
        "decontaminated": True,
        "source_dataset": "unit_mcq",
    }


def result(row_id: str, prediction: str, answer: str = "B", *, solution: str | None = None) -> dict:
    payload = {
        "id": row_id,
        "prediction": prediction,
        "answer": answer,
        "hit": prediction == answer,
        "mode": "base",
        "scores": {"A": -2.0, "B": -0.5 if prediction == "B" else -1.5, "C": -1.0},
    }
    if solution:
        payload["solution"] = solution
        payload["steps"] = 3
    return payload


def test_merge_rows_builds_capability_ladder_scored_schema(tmp_path) -> None:
    base_path = tmp_path / "base.jsonl"
    mid_path = tmp_path / "mid.jsonl"
    high_path = tmp_path / "high.jsonl"
    base_path.write_text(
        json.dumps(result("easy", "B", solution="Direct solve. ANSWER: B"))
        + "\n"
        + json.dumps(result("mid", "A"))
        + "\n",
        encoding="utf-8",
    )
    mid_path.write_text(
        json.dumps(result("easy", "B"))
        + "\n"
        + json.dumps(result("mid", "B", solution="Reason carefully. ANSWER: B"))
        + "\n",
        encoding="utf-8",
    )
    high_path.write_text(
        json.dumps(result("easy", "B"))
        + "\n"
        + json.dumps(result("mid", "B", solution="Longer proof. ANSWER: B"))
        + "\n",
        encoding="utf-8",
    )

    rows, report = merge_rows(
        [task("easy"), task("mid")],
        [("qwen_0_5b", base_path), ("qwen_1_5b", mid_path), ("qwen_3b", high_path)],
        verified_by="benchmark_ground_truth",
        decontaminated=True,
        id_field="id",
        prediction_as_solution=False,
    )

    assert report["output_rows"] == 2
    assert rows[0]["answer"]["verified_by"] == ["benchmark_ground_truth"]
    assert rows[0]["model_results"]["qwen_0_5b"]["correct"] is True
    assert rows[1]["model_results"]["qwen_0_5b"]["correct"] is False
    assert rows[1]["model_results"]["qwen_1_5b"]["solution"].endswith("ANSWER: B")

    records, ladder_report = build_records(
        rows,
        base_key="qwen_0_5b",
        mid_key="qwen_1_5b",
        high_keys=["qwen_3b"],
        high_target_loop=3,
        allow_answer_only=False,
        assume_decontaminated=False,
    )

    assert ladder_report["exported_records"] == 2
    by_id = {record["id"]: record for record in records}
    assert by_id["easy"]["target_loop_count"] == 1
    assert by_id["mid"]["target_loop_count"] == 2
    assert by_id["mid"]["capability_tier"] == "qwen_0_5b_miss_qwen_1_5b_solve"


def test_merge_capability_score_rows_cli_writes_report(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    base = tmp_path / "base.jsonl"
    mid = tmp_path / "mid.jsonl"
    output = tmp_path / "scored.jsonl"

    tasks.write_text(json.dumps(task("x")) + "\n", encoding="utf-8")
    base.write_text(json.dumps(result("x", "A")) + "\n", encoding="utf-8")
    mid.write_text(json.dumps(result("x", "B", solution="ANSWER: B")) + "\n", encoding="utf-8")

    assert main(
        [
            "--tasks_jsonl",
            str(tasks),
            "--result",
            f"qwen_0_5b={base}",
            "--result",
            f"qwen_1_5b={mid}",
            "--output_jsonl",
            str(output),
            "--assume_decontaminated",
        ]
    ) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    report = json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert rows[0]["decontaminated"] is True
    assert report["correctness"]["qwen_0_5b"]["incorrect"] == 1
    assert report["correctness"]["qwen_1_5b"]["correct"] == 1
