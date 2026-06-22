from __future__ import annotations

import json

import pytest

from training.build_capability_ladder_curriculum import build_records
from training.build_capability_ladder_trace_jobs import build_trace_jobs, main as build_main
from training.collect_capability_ladder_trace_outputs import collect_trace_rows, main as collect_main


def scored_row(
    row_id: str,
    *,
    base_correct: bool,
    mid_correct: bool,
    high_correct: bool,
) -> dict:
    return {
        "id": row_id,
        "domain": "science",
        "statement": "Which option is the answer?\nA. Three\nB. Forty two\nC. Seven",
        "answer": {"value": "B", "choice_text": "Forty two", "verified_by": ["benchmark_ground_truth"]},
        "decontaminated": True,
        "model_results": {
            "qwen_0_5b": {
                "correct": base_correct,
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "prediction": "A" if not base_correct else "B",
            },
            "qwen_1_5b": {
                "correct": mid_correct,
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "prediction": "B" if mid_correct else "A",
            },
            "qwen_3b": {
                "correct": high_correct,
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prediction": "B" if high_correct else "A",
            },
        },
    }


def test_build_trace_jobs_targets_capability_tiers_and_depths() -> None:
    jobs, report = build_trace_jobs(
        [
            scored_row("base", base_correct=True, mid_correct=True, high_correct=True),
            scored_row("mid", base_correct=False, mid_correct=True, high_correct=True),
            scored_row("high", base_correct=False, mid_correct=False, high_correct=True),
        ],
        models=["opus-test"],
        base_key="qwen_0_5b",
        mid_key="qwen_1_5b",
        high_keys=["qwen_3b"],
        high_target_loop=3,
    )

    assert report["status"] == "ready"
    assert report["jobs"] == 3
    assert report["by_target_loop"] == {"1": 1, "2": 1, "3": 1}
    assert [job["stage"] for job in jobs] == ["capability_ladder_trace_solve"] * 3
    assert [job["metadata"]["target_loop_count"] for job in jobs] == [1, 2, 3]
    assert "Final line to use exactly:\nANSWER: B" in jobs[0]["prompt"]
    assert "avoid label-position shortcuts" in jobs[0]["prompt"]
    assert jobs[1]["metadata"]["solver_key"] == "qwen_1_5b"
    assert jobs[2]["metadata"]["solver_key"] == "qwen_3b"


def test_trace_job_cli_blocks_student_lineage_by_default(tmp_path) -> None:
    scored = tmp_path / "scored.jsonl"
    out = tmp_path / "jobs.jsonl"
    scored.write_text(json.dumps(scored_row("base", base_correct=True, mid_correct=True, high_correct=True)) + "\n")

    with pytest.raises(ValueError, match="Student-lineage models are blocked"):
        build_main(
            [
                "--scored_jsonl",
                str(scored),
                "--models",
                "Qwen/Qwen2.5-72B-Instruct",
                "--output_jsonl",
                str(out),
            ]
        )


def test_trace_job_cli_resolves_probe_summary_artifact(tmp_path) -> None:
    scored = tmp_path / "data" / "scored.jsonl"
    scored.parent.mkdir()
    scored.write_text(json.dumps(scored_row("mid", base_correct=False, mid_correct=True, high_correct=True)) + "\n")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"kind": "stage5_capability_ladder_mcq_probe", "artifacts": {"scored_capability_rows": str(scored)}}),
        encoding="utf-8",
    )
    out = tmp_path / "jobs.jsonl"
    report = tmp_path / "report.json"

    assert build_main(
        [
            "--summary_json",
            str(summary),
            "--models",
            "opus-test,glm-test",
            "--output_jsonl",
            str(out),
            "--report_json",
            str(report),
        ]
    ) == 0

    jobs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert len(jobs) == 2
    assert payload["tier_counts"] == {"qwen_0_5b_miss_qwen_1_5b_solve": 1}


def test_collect_trace_outputs_accepts_verified_trace_and_enables_curriculum_build() -> None:
    scored = [scored_row("mid", base_correct=False, mid_correct=True, high_correct=True)]
    jobs, _report = build_trace_jobs(
        scored,
        models=["opus-test"],
        base_key="qwen_0_5b",
        mid_key="qwen_1_5b",
        high_keys=["qwen_3b"],
        high_target_loop=3,
    )
    responses = [
        {
            "job_id": jobs[0]["job_id"],
            "response_id": "response-1",
            "resolved_model": "claude-opus-test",
            "response_text": "Check the relevant fact, compare options, then select B.\nANSWER: B",
        }
    ]

    rows, report = collect_trace_rows(scored, jobs, responses)

    assert report["status"] == "ready"
    assert report["accepted_rows"] == 1
    assert report["target_loop_counts"] == {"2": 1}
    assert rows[0]["model_results"]["qwen_1_5b"]["source_model"] == "claude-opus-test"
    assert rows[0]["model_results"]["qwen_1_5b"]["solution"].endswith("ANSWER: B")

    records, ladder_report = build_records(
        rows,
        base_key="qwen_0_5b",
        mid_key="qwen_1_5b",
        high_keys=["qwen_3b"],
        high_target_loop=3,
        allow_answer_only=False,
        assume_decontaminated=False,
    )
    assert ladder_report["exported_records"] == 1
    assert records[0]["target_loop_count"] == 2
    assert records[0]["traces"][0]["text"].endswith("ANSWER: B")


def test_collect_trace_outputs_rejects_wrong_or_missing_answer(tmp_path) -> None:
    scored = [scored_row("mid", base_correct=False, mid_correct=True, high_correct=True)]
    jobs, _report = build_trace_jobs(
        scored,
        models=["opus-test"],
        base_key="qwen_0_5b",
        mid_key="qwen_1_5b",
        high_keys=["qwen_3b"],
        high_target_loop=3,
    )
    responses = [{"job_id": jobs[0]["job_id"], "response_text": "Looks like option A.\nANSWER: A"}]

    rows, report = collect_trace_rows(scored, jobs, responses)

    assert rows == []
    assert report["status_counts"] == {"wrong_or_missing_answer": 1}


def test_collect_trace_cli_writes_rows_and_report(tmp_path) -> None:
    scored_rows = [scored_row("high", base_correct=False, mid_correct=False, high_correct=True)]
    jobs, _report = build_trace_jobs(
        scored_rows,
        models=["opus-test"],
        base_key="qwen_0_5b",
        mid_key="qwen_1_5b",
        high_keys=["qwen_3b"],
        high_target_loop=3,
    )
    scored = tmp_path / "scored.jsonl"
    jobs_path = tmp_path / "jobs.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "with_traces.jsonl"
    report = tmp_path / "report.json"
    scored.write_text("\n".join(json.dumps(row) for row in scored_rows) + "\n", encoding="utf-8")
    jobs_path.write_text("\n".join(json.dumps(job) for job in jobs) + "\n", encoding="utf-8")
    responses.write_text(
        json.dumps({"job_id": jobs[0]["job_id"], "response_id": "r1", "response_text": "Reason deeply.\nANSWER: Forty two"})
        + "\n",
        encoding="utf-8",
    )

    assert collect_main(
        [
            "--scored_jsonl",
            str(scored),
            "--jobs_jsonl",
            str(jobs_path),
            "--responses_jsonl",
            str(responses),
            "--output_jsonl",
            str(output),
            "--report_json",
            str(report),
        ]
    ) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert payload["target_loop_counts"] == {"3": 1}
