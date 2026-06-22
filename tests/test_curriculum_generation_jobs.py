from __future__ import annotations

import json

import pytest

from training.build_curriculum_generation_jobs import (
    build_distinctness_jobs,
    build_ground_truth_jobs,
    build_method_solve_jobs,
    build_perturbation_jobs,
    build_reference_attempt_jobs,
    build_seed_jobs,
    main,
    validate_external_models,
)


def test_seed_jobs_cross_product_models_domains_difficulties_and_steps() -> None:
    jobs = build_seed_jobs(
        models=["opus-test", "glm-test"],
        domains=["math", "code"],
        difficulties=["easy", "hard"],
        target_steps=[3, 7],
        count_per_combo=2,
    )

    assert len(jobs) == 32
    assert jobs[0]["stage"] == "seed_generation"
    assert jobs[0]["role"] == "generator"
    assert jobs[0]["expects_json"] is True
    assert "Return ONLY this JSON" in jobs[0]["prompt"]
    assert jobs[0]["routing"]["requires_independent_verification"] is True


def test_student_lineage_models_are_blocked_by_default() -> None:
    with pytest.raises(ValueError, match="Student-lineage models are blocked"):
        validate_external_models(["Qwen/Qwen2.5-0.5B-Instruct"])


def test_reference_attempt_stage_allows_student_lineage_weak_reference(tmp_path) -> None:
    input_jsonl = tmp_path / "verified.jsonl"
    output_jsonl = tmp_path / "jobs.jsonl"
    input_jsonl.write_text(
        json.dumps({"id": "p1", "statement": "What is 2+2?", "domain": "math"}) + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--stage",
            "reference_attempt",
            "--models",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--input_jsonl",
            str(input_jsonl),
            "--reference_samples",
            "1",
            "--output_jsonl",
            str(output_jsonl),
        ]
    ) == 0


def test_ground_truth_jobs_use_answer_anchor_prompt() -> None:
    jobs = build_ground_truth_jobs(
        [{"id": "p1", "domain": "math", "statement": "What is 2+2?"}],
        models=["opus-test", "glm-test"],
    )

    assert len(jobs) == 2
    assert jobs[0]["stage"] == "ground_truth_solve"
    assert "ANSWER:" in jobs[0]["prompt"]
    assert jobs[0]["metadata"]["record_id"] == "p1"


def test_reference_attempt_jobs_sample_weak_reference_attempts() -> None:
    jobs = build_reference_attempt_jobs(
        [{"id": "p1", "domain": "math", "statement": "What is 2+2?"}],
        model="weak-ref",
        samples=3,
    )

    assert len(jobs) == 3
    assert {job["stage"] for job in jobs} == {"reference_attempt"}
    assert {job["role"] for job in jobs} == {"weak_reference"}
    assert [job["metadata"]["sample_id"] for job in jobs] == [0, 1, 2]
    assert "ANSWER:" in jobs[0]["prompt"]


def test_method_solve_jobs_use_explicit_or_taxonomy_methods() -> None:
    jobs = build_method_solve_jobs(
        [
            {
                "id": "p1",
                "domain": "math",
                "statement": "A right triangle has legs summing to 17 and hypotenuse 13. Find its area.",
                "candidate_methods": ["algebra", "synthetic_geometry"],
            }
        ],
        models=["opus-test"],
        fallback_methods=[],
    )

    assert len(jobs) == 2
    assert {job["metadata"]["method"] for job in jobs} == {"algebra", "synthetic_geometry"}
    assert "METHOD DOES NOT APPLY" in jobs[0]["prompt"]


def test_distinctness_jobs_pair_solution_methods_within_record() -> None:
    jobs = build_distinctness_jobs(
        [
            {
                "id": "s1",
                "record_id": "p1",
                "statement": "Find the area.",
                "method": "algebra",
                "solution": "Use A = lw.",
            },
            {
                "id": "s2",
                "record_id": "p1",
                "statement": "Find the area.",
                "method": "bounded_enumeration",
                "solution": "Count cells.",
            },
            {
                "id": "s3",
                "record_id": "p1",
                "statement": "Find the area.",
                "method": "bounded_enumeration",
                "solution": "Count rows.",
            },
        ],
        models=["opus-test", "glm-test"],
    )

    assert len(jobs) == 4
    assert jobs[0]["stage"] == "method_distinctness_judge"
    assert jobs[0]["expects_json"] is True
    assert jobs[0]["metadata"]["solution_a_id"] == "s1"
    assert jobs[0]["metadata"]["solution_b_id"] == "s2"
    assert "structurally distinct" in jobs[0]["prompt"]


def test_perturbation_jobs_emit_neutral_and_pressure_prompts() -> None:
    jobs = build_perturbation_jobs(
        [{"id": "p1", "statement": "What is 12 / 3?", "false_answer": "5"}],
        models=["opus-test"],
    )

    assert [job["stage"] for job in jobs] == ["false_answer_neutral", "false_answer_pressure"]
    assert "student claims" in jobs[0]["prompt"]
    assert "The answer to this problem is 5" in jobs[1]["prompt"]


def test_cli_writes_seed_job_jsonl_and_report(tmp_path) -> None:
    output_jsonl = tmp_path / "jobs.jsonl"
    report_json = tmp_path / "report.json"

    assert main(
        [
            "--stage",
            "seed",
            "--models",
            "opus-test,glm-test",
            "--domains",
            "math",
            "--difficulties",
            "medium",
            "--target_steps",
            "4,8",
            "--output_jsonl",
            str(output_jsonl),
            "--report_json",
            str(report_json),
        ]
    ) == 0

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert report["jobs"] == 4
    assert report["by_role"] == {"generator": 4}


def test_cli_writes_method_solve_jobs(tmp_path) -> None:
    input_jsonl = tmp_path / "verified.jsonl"
    output_jsonl = tmp_path / "jobs.jsonl"
    input_jsonl.write_text(
        json.dumps({"id": "p1", "domain": "math", "statement": "What is 3+4?"}) + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--stage",
            "method_solve",
            "--models",
            "opus-test",
            "--methods",
            "algebra,bounded_enumeration",
            "--input_jsonl",
            str(input_jsonl),
            "--output_jsonl",
            str(output_jsonl),
        ]
    ) == 0

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["stage"] == "method_constrained_solve"
