from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.run_curriculum_pipeline_from_artifacts import parse_args, read_jsonl, run_pipeline


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def args_for(tmp_path: Path, references_path: Path, *extra: str):
    return parse_args(
        [
            "--work_dir",
            str(tmp_path / "run"),
            "--seed_models",
            "gen-a",
            "--solver_models",
            "solver-a,solver-b",
            "--judge_models",
            "judge-a,judge-b",
            "--domains",
            "math",
            "--difficulties",
            "medium",
            "--target_steps",
            "4",
            "--count_per_combo",
            "1",
            "--references_jsonl",
            str(references_path),
            "--methods",
            "algebra,bounded_enumeration",
            "--min_reference_samples",
            "1",
            "--reference_samples",
            "1",
            "--min_natural_agree",
            "2",
            "--min_distinct_agree",
            "2",
            "--allow_single_model_roles",
            *extra,
        ]
    )


def response_rows_for_jobs(jobs_path: Path, responder) -> list[dict]:
    return [
        {
            "job_id": job["job_id"],
            "response_id": f"response-{job['job_id']}",
            "response_text": responder(job),
        }
        for job in read_jsonl(jobs_path)
    ]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pipeline_stops_after_building_seed_jobs(tmp_path) -> None:
    references = tmp_path / "refs.jsonl"
    write_jsonl(references, [{"id": "eval-1", "prompt": "What is 9 + 9?"}])

    summary = run_pipeline(args_for(tmp_path, references))

    assert summary["status"] == "pending_seed_responses"
    assert Path(summary["artifacts"]["jobs_seed"]["path"]).exists()
    assert summary["counts"]["seed_jobs"] == 1


def test_pipeline_requires_diverse_external_models_by_default(tmp_path) -> None:
    references = tmp_path / "refs.jsonl"
    write_jsonl(references, [{"id": "eval-1", "prompt": "What is 9 + 9?"}])

    args = parse_args(
        [
            "--work_dir",
            str(tmp_path / "run"),
            "--seed_models",
            "gen-a",
            "--solver_models",
            "solver-a,solver-b",
            "--judge_models",
            "judge-a,judge-b",
            "--references_jsonl",
            str(references),
        ]
    )

    with pytest.raises(ValueError, match="seed_models requires at least two distinct external models"):
        run_pipeline(args)


def test_pipeline_blocks_student_lineage_models_by_default(tmp_path) -> None:
    references = tmp_path / "refs.jsonl"
    write_jsonl(references, [{"id": "eval-1", "prompt": "What is 9 + 9?"}])

    args = parse_args(
        [
            "--work_dir",
            str(tmp_path / "run"),
            "--seed_models",
            "Qwen/Qwen2.5-0.5B-Instruct,glm-test",
            "--solver_models",
            "solver-a,solver-b",
            "--judge_models",
            "judge-a,judge-b",
            "--references_jsonl",
            str(references),
        ]
    )

    with pytest.raises(ValueError, match="Student-lineage models are blocked"):
        run_pipeline(args)


def test_pipeline_waits_for_partial_response_files(tmp_path) -> None:
    references = tmp_path / "refs.jsonl"
    write_jsonl(references, [{"id": "eval-1", "prompt": "What is 9 + 9?"}])

    args = args_for(tmp_path, references)
    summary = run_pipeline(args)
    paths = {name: Path(meta["path"]) for name, meta in summary["artifacts"].items()}

    write_jsonl(
        paths["responses_seed"],
        response_rows_for_jobs(
            paths["jobs_seed"],
            lambda _job: json.dumps(
                {
                    "statement": "A rectangle has side lengths 6 and 7. Find its area.",
                    "claimed_answer": "42",
                    "domain": "math",
                    "candidate_methods": ["algebra"],
                }
            ),
        ),
    )
    summary = run_pipeline(args)
    assert summary["status"] == "pending_ground_truth_responses"

    ground_jobs = read_jsonl(paths["jobs_ground_truth"])
    write_jsonl(
        paths["responses_ground_truth"],
        [
            {
                "job_id": ground_jobs[0]["job_id"],
                "response_id": "partial-ground-truth",
                "response_text": "6*7=42\nANSWER: 42",
            }
        ],
    )

    summary = run_pipeline(args)

    assert summary["status"] == "pending_ground_truth_responses"
    assert summary["counts"]["ground_truth_jobs"] == 2
    assert summary["counts"]["ground_truth_response_rows"] == 1
    assert summary["artifacts"]["verified_candidates"]["lines"] == 0


def test_pipeline_resumes_to_complete_from_artifacts(tmp_path) -> None:
    references = tmp_path / "refs.jsonl"
    work_dir = tmp_path / "run"
    write_jsonl(references, [{"id": "eval-1", "prompt": "What is 9 + 9?"}])

    args = args_for(tmp_path, references)
    summary = run_pipeline(args)
    assert summary["status"] == "pending_seed_responses"
    paths = {name: Path(meta["path"]) for name, meta in summary["artifacts"].items()}

    write_jsonl(
        paths["responses_seed"],
        response_rows_for_jobs(
            paths["jobs_seed"],
            lambda _job: json.dumps(
                {
                    "statement": "A rectangle has side lengths 6 and 7. Find its area.",
                    "claimed_answer": "43",
                    "domain": "math",
                    "candidate_methods": ["algebra", "bounded_enumeration"],
                }
            ),
        ),
    )
    summary = run_pipeline(args)
    assert summary["status"] == "pending_ground_truth_responses"
    assert read_json(paths["candidates_report"])["candidates"] == 1
    assert read_json(paths["decontam_report"])["accepted"] == 1

    write_jsonl(paths["responses_ground_truth"], response_rows_for_jobs(paths["jobs_ground_truth"], lambda _job: "6*7=42\nANSWER: 42"))
    summary = run_pipeline(args)
    assert summary["status"] == "pending_reference_attempt_responses"
    assert read_json(paths["verified_candidates_report"])["verified"] == 1

    write_jsonl(
        paths["responses_reference_attempts"],
        response_rows_for_jobs(paths["jobs_reference_attempts"], lambda _job: "Weak model misses.\nANSWER: 41"),
    )
    summary = run_pipeline(args)
    assert summary["status"] == "pending_method_or_perturbation_responses"
    assert read_json(paths["reference_attempts_report"])["attempts"] == 1
    assert read_json(paths["difficulty_report"])["measured"] == 1
    assert read_json(paths["false_answers_report"])["with_false_answer"] == 1

    write_jsonl(
        paths["responses_methods"],
        response_rows_for_jobs(paths["jobs_methods"], lambda _job: "A clean method gives 42.\nANSWER: 42"),
    )
    write_jsonl(
        paths["responses_perturbation"],
        response_rows_for_jobs(
            paths["jobs_perturbation"],
            lambda job: (
                "The false answer is wrong; 6*7=42.\nANSWER: 42"
                if job["stage"] == "false_answer_neutral"
                else f"Follow the false premise.\nANSWER: {job['metadata']['false_answer']}"
            ),
        ),
    )
    summary = run_pipeline(args)
    assert summary["status"] == "pending_judgment_responses"
    assert read_json(paths["method_solutions_report"])["solution_candidates"] == 4
    assert read_json(paths["perturbation_report"])["traces"] == 4

    write_jsonl(
        paths["responses_naturalness"],
        response_rows_for_jobs(
            paths["jobs_naturalness"],
            lambda job: json.dumps(
                {
                    "natural": True,
                    "actually_uses": job["metadata"]["method"],
                    "reason": "natural",
                }
            ),
        ),
    )
    write_jsonl(
        paths["responses_distinctness"],
        response_rows_for_jobs(paths["jobs_distinctness"], lambda _job: '{"distinct": true, "reason": "different"}'),
    )
    write_jsonl(
        paths["responses_depth"],
        response_rows_for_jobs(paths["jobs_depth"], lambda _job: '{"steps": ["recall", "multiply", "state"], "count": 3}'),
    )

    summary = run_pipeline(args)

    assert summary["status"] == "complete"
    assert summary["counts"]["typed_records"] == 1
    assert summary["counts"]["positive_sft_rows"] == 2
    typed = read_jsonl(work_dir / "typed_records.jsonl")
    assert {trace["role"] for trace in typed[0]["traces"]} == {
        "positive_wide",
        "verifier_detection",
        "verifier_rationalization",
    }
    assert read_json(paths["naturalness_report"])["judgments"] == 8
    assert read_json(paths["distinctness_report"])["judgments"] == 8
    assert read_json(paths["depth_report"])["measurements"] == 8
    assert read_json(paths["error_detection_report"])["judgments"] == 0
    assert read_json(paths["typed_records_report"])["records"] == 1
    assert read_json(paths["positive_sft_report"])["exported_examples"] == 2
    for report_name in [
        "candidates_report",
        "decontam_report",
        "verified_candidates_report",
        "reference_attempts_report",
        "difficulty_report",
        "false_answers_report",
        "method_solutions_report",
        "perturbation_report",
        "naturalness_report",
        "distinctness_report",
        "depth_report",
        "error_detection_report",
        "typed_records_report",
        "positive_sft_report",
    ]:
        assert summary["artifacts"][report_name]["exists"]
